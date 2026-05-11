import argparse
import csv
import glob
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


def projection_key(path: str) -> int:
    match = re.search(r"projection_(\d+)$", os.path.basename(path))
    return int(match.group(1)) if match else 999


def find_result_dirs(result_root: Path, pattern: str) -> list[Path]:
    dirs = sorted(glob.glob(str(result_root / pattern)), key=projection_key)
    out = [Path(d) for d in dirs]
    if len(out) != 3:
        raise RuntimeError(f"Expected 3 projection dirs for {pattern}, got {len(out)}")
    for directory in out:
        for name in ("pred.npy", "true.npy"):
            if not (directory / name).exists():
                raise FileNotFoundError(directory / name)
    return out


def build_design(full_z: np.ndarray, tau_max: int) -> tuple[np.ndarray, np.ndarray]:
    total_steps, n_vars = full_z.shape
    rows = total_steps - tau_max
    design = np.empty((rows, tau_max * n_vars), dtype=np.float32)
    for lag in range(1, tau_max + 1):
        design[:, (lag - 1) * n_vars : lag * n_vars] = full_z[tau_max - lag : total_steps - lag]
    targets = full_z[tau_max:].astype(np.float32, copy=False)
    return design, targets


def sparsify_topk(delta: np.ndarray, topk: int) -> np.ndarray:
    flat = delta.reshape(-1)
    nonzero = np.flatnonzero(np.abs(flat) > 1e-12)
    if len(nonzero) <= topk:
        return delta.astype(np.float32, copy=False)
    top_local = np.argpartition(np.abs(flat[nonzero]), -topk)[-topk:]
    keep = nonzero[top_local]
    sparse = np.zeros_like(flat, dtype=np.float32)
    sparse[keep] = flat[keep].astype(np.float32, copy=False)
    return sparse.reshape(delta.shape)


def local_delta_full_design(
    design: np.ndarray,
    targets: np.ndarray,
    row_start: int,
    row_end: int,
    tau_max: int,
    n_vars: int,
    ridge_alpha: float,
    a_base: np.ndarray,
    support: np.ndarray,
    topk: int,
) -> np.ndarray:
    # Fast upper-bound approximation: use all lagged variables in one multi-output ridge,
    # then mask and sparsify by the exported static support.
    x = design[row_start:row_end].astype(np.float64, copy=False)
    y = targets[row_start:row_end].astype(np.float64, copy=False)
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean(axis=0, keepdims=True)
    gram = x_centered @ x_centered.T
    gram.flat[:: gram.shape[0] + 1] += ridge_alpha
    dual = np.linalg.solve(gram, y_centered)
    coef = x_centered.T @ dual
    local = np.zeros((n_vars, n_vars), dtype=np.float32)
    for lag in range(tau_max):
        local += coef[lag * n_vars : (lag + 1) * n_vars, :].T.astype(np.float32)
    delta = (local - a_base) * support
    return sparsify_topk(delta, topk=topk)


def mean_instant_cosine(r: np.ndarray, e: np.ndarray, clip_abs: float | None) -> float:
    if clip_abs is not None:
        r = np.clip(r, -clip_abs, clip_abs)
        e = np.clip(e, -clip_abs, clip_abs)
    dot = np.sum(r * e, axis=-1)
    nr = np.linalg.norm(r, axis=-1)
    ne = np.linalg.norm(e, axis=-1)
    valid = (nr > 1e-12) & (ne > 1e-12)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(dot[valid] / (nr[valid] * ne[valid])))


def pearson_flat(r: np.ndarray, e: np.ndarray, clip_abs: float | None) -> float:
    if clip_abs is not None:
        r = np.clip(r, -clip_abs, clip_abs)
        e = np.clip(e, -clip_abs, clip_abs)
    rf = r.reshape(-1).astype(np.float64)
    ef = e.reshape(-1).astype(np.float64)
    rf -= rf.mean()
    ef -= ef.mean()
    denom = np.sqrt(np.sum(rf * rf) * np.sum(ef * ef))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(rf * ef) / denom)


def clipped_gamma(r: np.ndarray, e: np.ndarray, clip_abs: float | None) -> float:
    if clip_abs is not None:
        r = np.clip(r, -clip_abs, clip_abs)
        e = np.clip(e, -clip_abs, clip_abs)
    num = float(np.sum(r.astype(np.float64) * e.astype(np.float64)))
    den = float(np.sum(r.astype(np.float64) * r.astype(np.float64)))
    if den <= 1e-12:
        return 0.0
    return float(np.clip(num / den, 0.0, 1.0))


def sample_gamma_stats(r: np.ndarray, e: np.ndarray, clip_abs: float | None) -> dict[str, float]:
    if clip_abs is not None:
        r = np.clip(r, -clip_abs, clip_abs)
        e = np.clip(e, -clip_abs, clip_abs)
    num = np.sum(r.astype(np.float64) * e.astype(np.float64), axis=(1, 2))
    den = np.sum(r.astype(np.float64) * r.astype(np.float64), axis=(1, 2))
    gamma = np.zeros_like(num, dtype=np.float64)
    valid = den > 1e-12
    gamma[valid] = np.clip(num[valid] / den[valid], 0.0, 1.0)
    return {
        "gamma_sample_mean": float(np.mean(gamma)),
        "gamma_sample_median": float(np.median(gamma)),
        "gamma_sample_q25": float(np.quantile(gamma, 0.25)),
        "gamma_sample_q75": float(np.quantile(gamma, 0.75)),
        "gamma_sample_frac_positive": float(np.mean(gamma > 1e-8)),
    }


def mse(x: np.ndarray) -> float:
    return float(np.mean(x.astype(np.float64) * x.astype(np.float64)))


def load_ecl_zscore(data_csv: Path, columns: list[str], train_end: int) -> np.ndarray:
    df = pd.read_csv(data_csv)
    values = df[columns].to_numpy(dtype=np.float64)
    train = values[:train_end]
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return ((values - mean) / std).astype(np.float32)


def bucket_indices(lambda_values: np.ndarray, n_buckets: int) -> list[np.ndarray]:
    order = np.argsort(lambda_values, kind="mergesort")
    return [np.asarray(x, dtype=np.int64) for x in np.array_split(order, n_buckets)]


def audit_bucket(args) -> list[dict]:
    interface_dir = Path(args.interface_dir)
    result_root = Path(args.result_root)
    manifest = json.loads((interface_dir / "interface_manifest.json").read_text(encoding="utf-8"))
    geom = manifest["window_geometry"]
    graph = manifest["graph_contract"]
    columns = manifest["dataset_contract"]["columns"]
    seq_len = int(geom["seq_len"])
    pred_len = int(geom["pred_len"])
    tau_max = int(graph["tau_max"])
    ridge_alpha = float(graph["ridge_alpha"])
    topk = int(graph["window_delta_topk"])
    test_border1 = int(geom["split_ranges"]["test"]["border1"])
    train_end = int(geom["train_interval"][1])

    lambda_test = np.load(interface_dir / "lambda_test.npy").reshape(-1).astype(np.float64)
    buckets = bucket_indices(lambda_test, args.n_buckets)

    static_dirs = find_result_dirs(result_root, args.static_pattern)
    static_preds = [np.load(d / "pred.npy", mmap_mode="r") for d in static_dirs]
    trues = [np.load(d / "true.npy", mmap_mode="r") for d in static_dirs]
    n_samples, actual_pred_len, n_vars = static_preds[0].shape
    if n_samples != len(lambda_test) or actual_pred_len != pred_len:
        raise RuntimeError(
            f"Shape mismatch: pred={static_preds[0].shape}, lambda={lambda_test.shape}, pred_len={pred_len}"
        )
    for pred, true in zip(static_preds, trues):
        if pred.shape != static_preds[0].shape or true.shape != static_preds[0].shape:
            raise RuntimeError("Projection prediction shapes do not match")

    a_base = np.load(interface_dir / "a_base_agg.npy").astype(np.float32)
    support = np.load(interface_dir / "support.npy").astype(np.float32)
    full_z = load_ecl_zscore(Path(args.data_csv), columns=columns, train_end=train_end)
    design, targets = build_design(full_z, tau_max=tau_max)

    rows = []
    for bucket_id in args.bucket_ids:
        idx = buckets[bucket_id - 1]
        if args.max_per_bucket is not None and len(idx) > args.max_per_bucket:
            rng = np.random.default_rng(args.seed + bucket_id)
            idx = np.sort(rng.choice(idx, size=args.max_per_bucket, replace=False))
        bucket_start = time.time()
        print(
            f"[Bucket {bucket_id}] n={len(idx)} lambda=[{lambda_test[idx].min():.6f}, "
            f"{lambda_test[idx].max():.6f}]"
        )

        r_by_source_seed = {source: [[] for _ in range(3)] for source in args.sources}
        e_by_seed = [[] for _ in range(3)]
        static_err_by_seed = [[] for _ in range(3)]

        for pos, sample_id in enumerate(idx):
            s_begin = test_border1 + int(sample_id)
            s_end = s_begin + seq_len
            delta = local_delta_full_design(
                design=design,
                targets=targets,
                row_start=s_begin,
                row_end=s_end - tau_max,
                tau_max=tau_max,
                n_vars=n_vars,
                ridge_alpha=ridge_alpha,
                a_base=a_base,
                support=support,
                topk=topk,
            )
            delta_t = delta.T
            for seed_i, (pred, true) in enumerate(zip(static_preds, trues)):
                y_static = np.asarray(pred[sample_id], dtype=np.float32)
                y_true = np.asarray(true[sample_id], dtype=np.float32)
                e_static = y_true - y_static
                if pos == 0:
                    e_by_seed[seed_i] = []
                    static_err_by_seed[seed_i] = []
                e_by_seed[seed_i].append(e_static)
                static_err_by_seed[seed_i].append(e_static)
                if "static_pred" in args.sources:
                    r_by_source_seed["static_pred"][seed_i].append(y_static @ delta_t)
                if "true_oracle" in args.sources:
                    r_by_source_seed["true_oracle"][seed_i].append(y_true @ delta_t)
            if (pos + 1) % args.progress_every == 0 or (pos + 1) == len(idx):
                elapsed = time.time() - bucket_start
                print(f"  processed {pos + 1}/{len(idx)} windows | elapsed={elapsed:.1f}s")

        for source in args.sources:
            for seed_i in range(3):
                r = np.stack(r_by_source_seed[source][seed_i], axis=0).astype(np.float32)
                e = np.stack(e_by_seed[seed_i], axis=0).astype(np.float32)
                static_err = np.stack(static_err_by_seed[seed_i], axis=0).astype(np.float32)
                static_mse = mse(static_err)
                abs_values = np.concatenate(
                    [
                        np.abs(r.reshape(-1))[:: args.quantile_stride],
                        np.abs(e.reshape(-1))[:: args.quantile_stride],
                    ]
                )
                clip_values = {"raw": None}
                for q in args.clip_quantiles:
                    clip_values[f"clip_abs_q{int(q * 100)}"] = float(np.quantile(abs_values, q))

                for clip_name, clip_abs in clip_values.items():
                    gamma = clipped_gamma(r, e, clip_abs=clip_abs)
                    gamma_stats = sample_gamma_stats(r, e, clip_abs=clip_abs)
                    dyn_err_gamma = static_err - gamma * r
                    dyn_err_one = static_err - r
                    row = {
                        "bucket": bucket_id,
                        "n_windows": int(len(idx)),
                        "lambda_min": float(lambda_test[idx].min()),
                        "lambda_mean": float(lambda_test[idx].mean()),
                        "lambda_max": float(lambda_test[idx].max()),
                        "source": source,
                        "projection": seed_i,
                        "clip": clip_name,
                        "clip_abs": "" if clip_abs is None else float(clip_abs),
                        "cosine_mean_it": mean_instant_cosine(r, e, clip_abs=clip_abs),
                        "pearson_flat": pearson_flat(r, e, clip_abs=clip_abs),
                        "gamma_star_clip01": gamma,
                        "static_mse": static_mse,
                        "dynamic_mse_gamma_star": mse(dyn_err_gamma),
                        "dynamic_mse_gamma_1": mse(dyn_err_one),
                        "delta_mse_gamma_star": mse(dyn_err_gamma) - static_mse,
                        "delta_mse_gamma_1": mse(dyn_err_one) - static_mse,
                    }
                    row.update(gamma_stats)
                    rows.append(row)

    return rows


def write_rows(rows: list[dict], out_csv: Path, out_json: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No rows produced")
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ECL-96 DeltaA residual alignment by lambda bucket.")
    parser.add_argument("--interface-dir", default=r"C:\Users\cyl\Desktop\data\interfaces\ECL_graph_interface_parcorr")
    parser.add_argument("--result-root", default=r"C:\Users\cyl\Desktop\iTransformer-phasec-clean\results")
    parser.add_argument("--data-csv", default=r"C:\Users\cyl\Desktop\iTransformer-phasec-clean\dataset\ECL.csv")
    parser.add_argument(
        "--static-pattern",
        default="ecl96_confirm_lr5e4_static_anchor_itr3_*projection_*",
    )
    parser.add_argument("--out-dir", default=r"C:\Users\cyl\Desktop\data\deltaA_signal_audit")
    parser.add_argument("--bucket-ids", type=int, nargs="+", default=[5])
    parser.add_argument("--n-buckets", type=int, default=5)
    parser.add_argument("--max-per-bucket", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--sources", nargs="+", default=["static_pred", "true_oracle"])
    parser.add_argument("--clip-quantiles", type=float, nargs="+", default=[0.99, 0.95])
    parser.add_argument("--quantile-stride", type=int, default=20)
    args = parser.parse_args()

    started = time.time()
    rows = audit_bucket(args)
    suffix = "b" + "-".join(str(x) for x in args.bucket_ids)
    if args.max_per_bucket is not None:
        suffix += f"_max{args.max_per_bucket}"
    out_dir = Path(args.out_dir)
    out_csv = out_dir / f"ecl96_deltaA_signal_audit_{suffix}.csv"
    out_json = out_dir / f"ecl96_deltaA_signal_audit_{suffix}.json"
    write_rows(rows, out_csv, out_json)
    print(f"[Done] wrote {out_csv}")
    print(f"[Done] wrote {out_json}")
    print(f"[Done] elapsed={(time.time() - started) / 60.0:.2f} min")


if __name__ == "__main__":
    main()

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

from step5pp_utils import (
    build_window_features,
    kmeans_simple,
    kmeans_sklearn,
    nearest_center_distance,
)


def projection_key(path: str) -> int:
    match = re.search(r"projection_(\d+)$", os.path.basename(path))
    return int(match.group(1)) if match else 999


def find_result_dirs(result_root: Path, pattern: str, pred_file: str, true_file: str) -> list[Path]:
    dirs = sorted(glob.glob(str(result_root / pattern)), key=projection_key)
    out = [Path(d) for d in dirs]
    if len(out) != 3:
        raise RuntimeError(f"Expected 3 projection dirs for {pattern}, got {len(out)}")
    for directory in out:
        for name in (pred_file, true_file):
            if not (directory / name).exists():
                raise FileNotFoundError(directory / name)
    return out


def build_design(full_z: np.ndarray, tau_max: int) -> tuple[np.ndarray, np.ndarray]:
    total_steps, n_vars = full_z.shape
    rows = total_steps - tau_max
    design = np.empty((rows, tau_max * n_vars), dtype=np.float32)
    for lag in range(1, tau_max + 1):
        design[:, (lag - 1) * n_vars : lag * n_vars] = full_z[tau_max - lag : total_steps - lag]
    return design, full_z[tau_max:].astype(np.float32, copy=False)


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


def load_ecl_zscore(data_csv: Path, columns: list[str], train_end: int) -> np.ndarray:
    values = pd.read_csv(data_csv)[columns].to_numpy(dtype=np.float64)
    train = values[:train_end]
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return ((values - mean) / std).astype(np.float32)


def sanitize_lambda(lambda_array: np.ndarray) -> np.ndarray:
    if np.isinf(lambda_array).any():
        raise ValueError("Lambda contains Inf values.")
    nan_mask = np.isnan(lambda_array)
    if not nan_mask.any():
        return lambda_array.astype(np.float32, copy=False)
    valid_idx = np.where(~nan_mask)[0]
    if len(valid_idx) == 0:
        raise ValueError("Lambda is entirely NaN.")
    return np.interp(
        np.arange(len(lambda_array), dtype=np.float64),
        valid_idx.astype(np.float64),
        lambda_array[valid_idx].astype(np.float64),
    ).astype(np.float32)


def quantile_normalize_with_reference(
    values: np.ndarray,
    reference_values: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    q10 = np.quantile(reference_values, 0.10)
    q90 = np.quantile(reference_values, 0.90)
    if not np.isfinite(q10) or not np.isfinite(q90) or q90 <= q10 + eps:
        vmin = float(reference_values.min())
        vmax = float(reference_values.max())
        if vmax <= vmin + eps:
            return np.zeros_like(values)
        return np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    return np.clip((values - q10) / (q90 - q10), 0.0, 1.0)


def compute_lambda_kmeans_trainfit(
    train_z: np.ndarray,
    full_z: np.ndarray,
    window: int,
    k: int,
    seed: int,
    max_iter: int = 100,
) -> np.ndarray:
    train_feats, _train_idx, _ = build_window_features(train_z, window=window)
    full_feats, full_idx, _ = build_window_features(full_z, window=window)
    train_feats = np.nan_to_num(train_feats, nan=0.0, posinf=0.0, neginf=0.0)
    full_feats = np.nan_to_num(full_feats, nan=0.0, posinf=0.0, neginf=0.0)
    if train_feats.shape[0] == 0 or full_feats.shape[0] == 0:
        raise ValueError("Not enough windows to compute lambda timeline.")

    feat_mean = train_feats.mean(axis=0)
    feat_std = train_feats.std(axis=0)
    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
    train_feats_std = (train_feats - feat_mean) / feat_std
    full_feats_std = (full_feats - feat_mean) / feat_std

    km_out = kmeans_sklearn(train_feats_std, k, seed)
    if km_out is None:
        _, centers = kmeans_simple(train_feats_std, k, seed, max_iter=max_iter)
    else:
        _, centers = km_out

    train_dists = nearest_center_distance(train_feats_std, centers)
    full_dists = nearest_center_distance(full_feats_std, centers)
    lambda_valid = quantile_normalize_with_reference(full_dists, train_dists)
    lambda_t = np.full((full_z.shape[0],), np.nan, dtype=np.float64)
    lambda_t[full_idx] = lambda_valid
    return sanitize_lambda(np.asarray(lambda_t, dtype=np.float32))


def resolve_split_ranges(manifest: dict, seq_len: int) -> dict[str, dict[str, int]]:
    geom = manifest["window_geometry"]
    if "split_ranges" in geom:
        return geom["split_ranges"]

    contract = manifest["dataset_contract"]
    split_mode = contract.get("split_mode", "custom_ratio")
    total_rows = int(contract["total_rows"])
    train_length = int(geom.get("train_interval", [0, contract["train_length"]])[1])
    if split_mode in {"ett_hour", "ett_minute"}:
        scale = 4 if split_mode == "ett_minute" else 1
        train_end = 12 * 30 * 24 * scale
        val_end = train_end + 4 * 30 * 24 * scale
        test_end = train_end + 8 * 30 * 24 * scale
        if test_end > total_rows:
            raise ValueError(
                f"ETT split exceeds total rows: split_mode={split_mode}, "
                f"test_end={test_end}, total_rows={total_rows}"
            )
        return {
            "train": {"border1": 0, "border2": train_end},
            "val": {"border1": train_end - seq_len, "border2": val_end},
            "test": {"border1": val_end - seq_len, "border2": test_end},
        }

    if split_mode != "custom_ratio":
        raise ValueError(f"Cannot infer split ranges for legacy split_mode={split_mode}")

    num_train = train_length
    num_test = int(total_rows * 0.2)
    num_val = total_rows - num_train - num_test
    if min(num_train, num_val, num_test) <= 0:
        raise ValueError(
            f"Invalid inferred split sizes: train={num_train}, val={num_val}, "
            f"test={num_test}, total={total_rows}"
        )
    return {
        "train": {"border1": 0, "border2": num_train},
        "val": {"border1": num_train - seq_len, "border2": num_train + num_val},
        "test": {"border1": total_rows - num_test - seq_len, "border2": total_rows},
    }


def lambda_values_for_split(lambda_t: np.ndarray, split_range: dict[str, int], seq_len: int, pred_len: int) -> np.ndarray:
    border1 = int(split_range["border1"])
    border2 = int(split_range["border2"])
    num_windows = border2 - border1 - seq_len - pred_len + 1
    if num_windows <= 0:
        raise ValueError(f"Invalid split range for lambda extraction: {split_range}")
    out = np.empty((num_windows,), dtype=np.float32)
    for sample_id in range(num_windows):
        s_begin = border1 + sample_id
        out[sample_id] = float(lambda_t[s_begin : s_begin + seq_len].mean())
    return out


def load_or_compute_lambda_splits(
    interface_dir: Path,
    manifest: dict,
    full_z: np.ndarray,
    split_ranges: dict[str, dict[str, int]],
    seq_len: int,
    pred_len: int,
    train_end: int,
) -> dict[str, np.ndarray]:
    split_values: dict[str, np.ndarray] = {}
    missing = []
    for split_name in ("train", "val", "test"):
        path = interface_dir / f"lambda_{split_name}.npy"
        if path.exists():
            split_values[split_name] = np.load(path).reshape(-1).astype(np.float64)
        else:
            missing.append(split_name)
    if not missing:
        return split_values

    contract = manifest.get("lambda_contract", {})
    window = int(contract.get("lambda_window", 40))
    k = int(contract.get("lambda_k", 2))
    seed = int(contract.get("lambda_seed", 2023))
    print(
        "[LambdaFallback] missing="
        f"{missing}; recomputing full lambda timeline with train-fit kmeans "
        f"(window={window}, k={k}, seed={seed})"
    )
    lambda_t = compute_lambda_kmeans_trainfit(
        train_z=full_z[:train_end].astype(np.float64, copy=False),
        full_z=full_z.astype(np.float64, copy=False),
        window=window,
        k=k,
        seed=seed,
    )
    for split_name in missing:
        split_values[split_name] = lambda_values_for_split(
            lambda_t,
            split_ranges[split_name],
            seq_len=seq_len,
            pred_len=pred_len,
        ).astype(np.float64)
    return split_values


def bucket_indices(lambda_values: np.ndarray, n_buckets: int) -> list[np.ndarray]:
    order = np.argsort(lambda_values, kind="mergesort")
    return [np.asarray(x, dtype=np.int64) for x in np.array_split(order, n_buckets)]


def mse_mae(err: np.ndarray) -> tuple[float, float]:
    err64 = err.astype(np.float64)
    return float(np.mean(err64 * err64)), float(np.mean(np.abs(err64)))


def build_dynamic_cache(args):
    interface_dir = Path(args.interface_dir)
    manifest = json.loads((interface_dir / "interface_manifest.json").read_text(encoding="utf-8"))
    geom = manifest["window_geometry"]
    graph = manifest["graph_contract"]
    columns = manifest["dataset_contract"]["columns"]
    seq_len = int(geom["seq_len"])
    pred_len = int(geom.get("pred_len", args.pred_len))
    tau_max = int(graph["tau_max"])
    ridge_alpha = float(graph["ridge_alpha"])
    topk = int(graph["window_delta_topk"])
    split_ranges = resolve_split_ranges(manifest, seq_len=seq_len)
    eval_split = str(args.eval_split)
    eval_border1 = int(split_ranges[eval_split]["border1"])
    train_end = int(geom["train_interval"][1])

    a_base = np.load(interface_dir / "a_base_agg.npy").astype(np.float32)
    support = np.load(interface_dir / "support.npy").astype(np.float32)
    full_z = load_ecl_zscore(Path(args.data_csv), columns=columns, train_end=train_end)
    lambda_splits = load_or_compute_lambda_splits(
        interface_dir=interface_dir,
        manifest=manifest,
        full_z=full_z,
        split_ranges=split_ranges,
        seq_len=seq_len,
        pred_len=pred_len,
        train_end=train_end,
    )
    lambda_test = lambda_splits[eval_split]
    lambda_schedule = lambda_splits[args.schedule_source]
    design, targets = build_design(full_z, tau_max=tau_max)

    n_samples = len(lambda_test)
    n_vars = a_base.shape[0]
    dyn = np.zeros((n_samples, args.pred_len, n_vars), dtype=np.float32)
    static_dirs = find_result_dirs(
        Path(args.result_root),
        args.static_pattern,
        pred_file=args.pred_file,
        true_file=args.true_file,
    )
    pred0 = np.load(static_dirs[0] / args.pred_file, mmap_mode="r")
    if pred0.shape[0] != n_samples or pred0.shape[1] != args.pred_len or pred0.shape[2] != n_vars:
        raise RuntimeError(f"Unexpected pred shape: {pred0.shape}")

    started = time.time()
    for sample_id in range(n_samples):
        s_begin = eval_border1 + int(sample_id)
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
        dyn[sample_id] = np.asarray(pred0[sample_id], dtype=np.float32) @ delta.T
        if (sample_id + 1) % args.progress_every == 0 or sample_id + 1 == n_samples:
            print(f"dynamic cache {sample_id + 1}/{n_samples} | elapsed={time.time() - started:.1f}s")
    return dyn, lambda_test, lambda_schedule, static_dirs


def run_posthoc(args) -> list[dict]:
    dynamic, lambda_test, lambda_schedule, static_dirs = build_dynamic_cache(args)
    buckets = bucket_indices(lambda_test, args.n_buckets)
    gate_masks = {
        "all": np.ones(len(lambda_test), dtype=bool),
        "bucket5": np.zeros(len(lambda_test), dtype=bool),
    }
    gate_masks["bucket5"][buckets[4]] = True
    q_low = float(np.quantile(lambda_schedule, args.linear_q_low))
    q_high = float(np.quantile(lambda_schedule, args.linear_q_high))
    if q_high <= q_low:
        raise ValueError(f"Invalid linear schedule quantiles: q_low={q_low}, q_high={q_high}")
    gamma_slope = (float(args.linear_gamma_max) - float(args.linear_gamma_min)) / (q_high - q_low)
    linear_weight = np.clip((lambda_test - q_low) / (q_high - q_low), 0.0, 1.0)
    linear_gamma = (
        float(args.linear_gamma_min)
        + (float(args.linear_gamma_max) - float(args.linear_gamma_min)) * linear_weight
    ).astype(np.float32)
    print(
        "[LinearSchedule] "
        f"source={args.schedule_source} "
        f"q{int(args.linear_q_low * 100)}={q_low:.6f} "
        f"q{int(args.linear_q_high * 100)}={q_high:.6f} "
        f"gamma_min={args.linear_gamma_min:.4f} gamma_max={args.linear_gamma_max:.4f} "
        f"slope={gamma_slope:.6f} gamma_mean_test={linear_gamma.mean():.6f}"
    )
    linear_gate_name = f"linear_q{int(args.linear_q_low * 100)}_q{int(args.linear_q_high * 100)}_{args.schedule_source}"

    rows = []
    for projection, directory in enumerate(static_dirs):
        pred = np.load(directory / args.pred_file, mmap_mode="r")
        true = np.load(directory / args.true_file, mmap_mode="r")
        static_err = np.asarray(true, dtype=np.float32) - np.asarray(pred, dtype=np.float32)
        static_mse, static_mae = mse_mae(static_err)

        for gate_name, gate_mask in gate_masks.items():
            gate = gate_mask.astype(np.float32).reshape(-1, 1, 1)
            for gamma in args.gammas:
                err = static_err - float(gamma) * gate * dynamic
                all_mse, all_mae = mse_mae(err)
                rows.append(
                    {
                        "projection": projection,
                        "gate": gate_name,
                        "gamma": float(gamma),
                        "gamma_min": float(gamma),
                        "gamma_max": float(gamma),
                        "gamma_mean": float(gamma),
                        "q_low": np.nan,
                        "q_high": np.nan,
                        "gamma_slope": np.nan,
                        "scope": "all",
                        "n": int(len(lambda_test)),
                        "lambda_min": float(lambda_test.min()),
                        "lambda_mean": float(lambda_test.mean()),
                        "lambda_max": float(lambda_test.max()),
                        "static_mse": static_mse,
                        "posthoc_mse": all_mse,
                        "delta_mse": all_mse - static_mse,
                        "rel_mse_pct": 100.0 * (all_mse - static_mse) / static_mse,
                        "static_mae": static_mae,
                        "posthoc_mae": all_mae,
                        "delta_mae": all_mae - static_mae,
                        "rel_mae_pct": 100.0 * (all_mae - static_mae) / static_mae,
                    }
                )
                for bucket_id, idx in enumerate(buckets, start=1):
                    bucket_err = static_err[idx]
                    post_err = err[idx]
                    b_static_mse, b_static_mae = mse_mae(bucket_err)
                    b_mse, b_mae = mse_mae(post_err)
                    rows.append(
                        {
                            "projection": projection,
                            "gate": gate_name,
                            "gamma": float(gamma),
                            "gamma_min": float(gamma),
                            "gamma_max": float(gamma),
                            "gamma_mean": float(gamma),
                            "q_low": np.nan,
                            "q_high": np.nan,
                            "gamma_slope": np.nan,
                            "scope": f"bucket{bucket_id}",
                            "n": int(len(idx)),
                            "lambda_min": float(lambda_test[idx].min()),
                            "lambda_mean": float(lambda_test[idx].mean()),
                            "lambda_max": float(lambda_test[idx].max()),
                            "static_mse": b_static_mse,
                            "posthoc_mse": b_mse,
                            "delta_mse": b_mse - b_static_mse,
                            "rel_mse_pct": 100.0 * (b_mse - b_static_mse) / b_static_mse,
                            "static_mae": b_static_mae,
                            "posthoc_mae": b_mae,
                            "delta_mae": b_mae - b_static_mae,
                            "rel_mae_pct": 100.0 * (b_mae - b_static_mae) / b_static_mae,
                        }
                    )
        gamma_values = linear_gamma.reshape(-1, 1, 1)
        err = static_err - gamma_values * dynamic
        all_mse, all_mae = mse_mae(err)
        rows.append(
            {
                "projection": projection,
                "gate": linear_gate_name,
                "gamma": float(linear_gamma.mean()),
                "gamma_min": float(args.linear_gamma_min),
                "gamma_max": float(args.linear_gamma_max),
                "gamma_mean": float(linear_gamma.mean()),
                "q_low": q_low,
                "q_high": q_high,
                "gamma_slope": float(gamma_slope),
                "scope": "all",
                "n": int(len(lambda_test)),
                "lambda_min": float(lambda_test.min()),
                "lambda_mean": float(lambda_test.mean()),
                "lambda_max": float(lambda_test.max()),
                "static_mse": static_mse,
                "posthoc_mse": all_mse,
                "delta_mse": all_mse - static_mse,
                "rel_mse_pct": 100.0 * (all_mse - static_mse) / static_mse,
                "static_mae": static_mae,
                "posthoc_mae": all_mae,
                "delta_mae": all_mae - static_mae,
                "rel_mae_pct": 100.0 * (all_mae - static_mae) / static_mae,
            }
        )
        for bucket_id, idx in enumerate(buckets, start=1):
            bucket_err = static_err[idx]
            post_err = err[idx]
            b_static_mse, b_static_mae = mse_mae(bucket_err)
            b_mse, b_mae = mse_mae(post_err)
            rows.append(
                {
                    "projection": projection,
                    "gate": linear_gate_name,
                    "gamma": float(linear_gamma.mean()),
                    "gamma_min": float(args.linear_gamma_min),
                    "gamma_max": float(args.linear_gamma_max),
                    "gamma_mean": float(linear_gamma[idx].mean()),
                    "q_low": q_low,
                    "q_high": q_high,
                    "gamma_slope": float(gamma_slope),
                    "scope": f"bucket{bucket_id}",
                    "n": int(len(idx)),
                    "lambda_min": float(lambda_test[idx].min()),
                    "lambda_mean": float(lambda_test[idx].mean()),
                    "lambda_max": float(lambda_test[idx].max()),
                    "static_mse": b_static_mse,
                    "posthoc_mse": b_mse,
                    "delta_mse": b_mse - b_static_mse,
                    "rel_mse_pct": 100.0 * (b_mse - b_static_mse) / b_static_mse,
                    "static_mae": b_static_mae,
                    "posthoc_mae": b_mae,
                    "delta_mae": b_mae - b_static_mae,
                    "rel_mae_pct": 100.0 * (b_mae - b_static_mae) / b_static_mae,
                }
            )
    return rows


def write_rows(rows: list[dict], out_csv: Path, out_summary_csv: Path, out_json: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["gate", "gamma", "scope"], as_index=False)
        .agg(
            n=("n", "first"),
            gamma_min=("gamma_min", "first"),
            gamma_max=("gamma_max", "first"),
            gamma_mean=("gamma_mean", "mean"),
            q_low=("q_low", "first"),
            q_high=("q_high", "first"),
            gamma_slope=("gamma_slope", "first"),
            lambda_min=("lambda_min", "first"),
            lambda_mean=("lambda_mean", "first"),
            lambda_max=("lambda_max", "first"),
            static_mse=("static_mse", "mean"),
            posthoc_mse=("posthoc_mse", "mean"),
            delta_mse=("delta_mse", "mean"),
            rel_mse_pct=("rel_mse_pct", "mean"),
            static_mae=("static_mae", "mean"),
            posthoc_mae=("posthoc_mae", "mean"),
            delta_mae=("delta_mae", "mean"),
            rel_mae_pct=("rel_mae_pct", "mean"),
        )
    )
    summary.to_csv(out_summary_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc manual-gate DeltaA audit for ECL-96.")
    parser.add_argument("--interface-dir", default=r"C:\Users\cyl\Desktop\data\interfaces\ECL_graph_interface_parcorr")
    parser.add_argument("--result-root", default=r"C:\Users\cyl\Desktop\iTransformer-phasec-clean\results")
    parser.add_argument("--data-csv", default=r"C:\Users\cyl\Desktop\iTransformer-phasec-clean\dataset\ECL.csv")
    parser.add_argument(
        "--static-pattern",
        default="ecl96_confirm_lr5e4_static_anchor_itr3_*projection_*",
    )
    parser.add_argument("--out-dir", default=r"C:\Users\cyl\Desktop\data\deltaA_signal_audit")
    parser.add_argument("--n-buckets", type=int, default=5)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.03, 0.05, 0.08, 0.10])
    parser.add_argument("--linear-gamma-min", type=float, default=0.03)
    parser.add_argument("--linear-gamma-max", type=float, default=0.08)
    parser.add_argument("--linear-q-low", type=float, default=0.20)
    parser.add_argument("--linear-q-high", type=float, default=0.80)
    parser.add_argument("--schedule-source", choices=["train", "val", "test"], default="test")
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--pred-file", default="pred.npy")
    parser.add_argument("--true-file", default="true.npy")
    parser.add_argument("--output-prefix", default="ecl96_deltaA_posthoc_manual_gate")
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    started = time.time()
    rows = run_posthoc(args)
    out_dir = Path(args.out_dir)
    out_csv = out_dir / f"{args.output_prefix}.csv"
    out_summary = out_dir / f"{args.output_prefix}_summary.csv"
    out_json = out_dir / f"{args.output_prefix}.json"
    write_rows(rows, out_csv, out_summary, out_json)
    print(f"[Done] wrote {out_csv}")
    print(f"[Done] wrote {out_summary}")
    print(f"[Done] wrote {out_json}")
    print(f"[Done] elapsed={(time.time() - started) / 60.0:.2f} min")


if __name__ == "__main__":
    main()

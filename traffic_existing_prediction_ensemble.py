from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from posthoc_calibration.evaluation import pct_gain
from posthoc_calibration.io_utils import load_result_dirs
from posthoc_calibration.profiles import PROFILES
from posthoc_ecl96_deltaA_manual_gate import mse_mae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-selected Traffic ensemble over existing baseline/static prediction assets."
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="traffic96_static")
    parser.add_argument("--alphas", default="0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0")
    parser.add_argument("--top-ks", default="2,3,4,5,6")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--select-mae-min-gain", type=float, default=0.0)
    parser.add_argument("--disable-adaptive-alpha", action="store_true")
    parser.add_argument("--var-alpha-shrink-prior-strength", type=float, default=1.0)
    parser.add_argument("--adaptive-alpha-min-denom", type=float, default=1e-12)
    parser.add_argument(
        "--out-dir",
        default=r"C:\Users\cyl\Desktop\data\deltaA_signal_audit\traffic96_existing_prediction_ensemble",
    )
    parser.add_argument("--tag", default="existing_ensemble")
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def projection_id(path: Path) -> int:
    match = re.search(r"projection_(\d+)$", path.name)
    return int(match.group(1)) if match else 999


def load_candidates(profile: dict) -> list[dict]:
    rows = []
    groups = [
        ("baseline", str(profile["baseline_pattern"])),
        ("static", str(profile["static_pattern"])),
    ]
    for group, pattern in groups:
        val_dirs = load_result_dirs(pattern, pred_file="val_pred.npy", true_file="val_true.npy")
        test_dirs = load_result_dirs(pattern, pred_file="pred.npy", true_file="true.npy")
        val_by_projection = {projection_id(path): path for path in val_dirs}
        test_by_projection = {projection_id(path): path for path in test_dirs}
        for projection in sorted(val_by_projection):
            if projection not in test_by_projection:
                raise FileNotFoundError(f"Missing test projection={projection} for group={group}")
            rows.append(
                {
                    "candidate": f"{group}_p{projection}",
                    "group": group,
                    "projection": int(projection),
                    "val_dir": val_by_projection[projection],
                    "test_dir": test_by_projection[projection],
                }
            )
    return rows


def pred_path(candidate: dict, split: str) -> Path:
    if split == "val":
        return Path(candidate["val_dir"]) / "val_pred.npy"
    if split == "test":
        return Path(candidate["test_dir"]) / "pred.npy"
    raise ValueError(split)


def true_path(candidate: dict, split: str) -> Path:
    if split == "val":
        return Path(candidate["val_dir"]) / "val_true.npy"
    if split == "test":
        return Path(candidate["test_dir"]) / "true.npy"
    raise ValueError(split)


def group_indices(candidates: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    groups = [candidate["group"] for candidate in candidates]
    baseline_idx = np.asarray([i for i, group in enumerate(groups) if group == "baseline"], dtype=np.int64)
    static_idx = np.asarray([i for i, group in enumerate(groups) if group == "static"], dtype=np.int64)
    if baseline_idx.size == 0 or static_idx.size == 0:
        raise ValueError("Adaptive alpha requires both baseline and static candidate groups")
    return baseline_idx, static_idx


def group_blend_weights(candidates: list[dict], alpha: float) -> np.ndarray:
    baseline_idx, static_idx = group_indices(candidates)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    weights = np.zeros(len(candidates), dtype=np.float64)
    weights[baseline_idx] = (1.0 - alpha) / baseline_idx.size
    weights[static_idx] = alpha / static_idx.size
    return weights


def group_mean_chunk(pred_arrays: list[np.ndarray], idx: np.ndarray, start: int, end: int) -> np.ndarray:
    out = np.zeros((end - start, *pred_arrays[0].shape[1:]), dtype=np.float32)
    scale = 1.0 / float(idx.size)
    for pred_idx in idx:
        out += scale * np.asarray(pred_arrays[int(pred_idx)][start:end], dtype=np.float32)
    return out


def evaluate_weighted(
    candidates: list[dict],
    weights: np.ndarray,
    split: str,
    chunk_size: int,
) -> dict:
    if len(candidates) != len(weights):
        raise ValueError("Candidate/weight length mismatch")
    weights = np.asarray(weights, dtype=np.float64)
    if weights.size == 0 or abs(float(weights.sum()) - 1.0) > 1e-6:
        raise ValueError(f"Invalid ensemble weights: sum={weights.sum() if weights.size else np.nan}")

    pred_arrays = [np.load(pred_path(candidate, split), mmap_mode="r") for candidate in candidates]
    true = np.load(true_path(candidates[0], split), mmap_mode="r")
    expected_shape = true.shape
    for candidate, pred in zip(candidates, pred_arrays):
        if pred.shape != expected_shape:
            raise RuntimeError(f"Unexpected pred shape for {candidate['candidate']}: {pred.shape} vs {expected_shape}")

    sse = 0.0
    sae = 0.0
    count = int(np.prod(expected_shape))
    n_samples = expected_shape[0]
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        ensemble = np.zeros((end - start, *expected_shape[1:]), dtype=np.float32)
        for weight, pred in zip(weights, pred_arrays):
            if abs(float(weight)) <= 1e-12:
                continue
            ensemble += float(weight) * np.asarray(pred[start:end], dtype=np.float32)
        err = np.asarray(true[start:end], dtype=np.float32) - ensemble
        sse += float(np.square(err, dtype=np.float32).sum(dtype=np.float64))
        sae += float(np.abs(err).sum(dtype=np.float64))
        del ensemble, err
    mse = sse / count
    mae = sae / count
    return {"mse": float(mse), "mae": float(mae)}


def evaluate_group_alpha(
    candidates: list[dict],
    alpha: float | np.ndarray,
    split: str,
    chunk_size: int,
) -> dict:
    baseline_idx, static_idx = group_indices(candidates)
    pred_arrays = [np.load(pred_path(candidate, split), mmap_mode="r") for candidate in candidates]
    true = np.load(true_path(candidates[0], split), mmap_mode="r")
    expected_shape = true.shape
    for candidate, pred in zip(candidates, pred_arrays):
        if pred.shape != expected_shape:
            raise RuntimeError(f"Unexpected pred shape for {candidate['candidate']}: {pred.shape} vs {expected_shape}")

    alpha_arr = np.asarray(alpha, dtype=np.float32)
    if alpha_arr.ndim == 0:
        alpha_view = alpha_arr.reshape(1, 1, 1)
    elif alpha_arr.ndim == 1 and alpha_arr.shape[0] == expected_shape[-1]:
        alpha_view = alpha_arr.reshape(1, 1, expected_shape[-1])
    else:
        raise ValueError(f"alpha must be scalar or length-N vector, got shape {alpha_arr.shape}")

    sse = 0.0
    sae = 0.0
    count = int(np.prod(expected_shape))
    n_samples = expected_shape[0]
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        baseline_mean = group_mean_chunk(pred_arrays, baseline_idx, start, end)
        static_mean = group_mean_chunk(pred_arrays, static_idx, start, end)
        ensemble = baseline_mean + alpha_view * (static_mean - baseline_mean)
        err = np.asarray(true[start:end], dtype=np.float32) - ensemble
        sse += float(np.square(err, dtype=np.float32).sum(dtype=np.float64))
        sae += float(np.abs(err).sum(dtype=np.float64))
        del baseline_mean, static_mean, ensemble, err
    mse = sse / count
    mae = sae / count
    return {"mse": float(mse), "mae": float(mae)}


def compute_adaptive_alphas(
    candidates: list[dict],
    split: str,
    chunk_size: int,
    shrink_prior_strength: float,
    min_denom: float,
) -> tuple[dict, pd.DataFrame]:
    if shrink_prior_strength < 0.0:
        raise ValueError("var-alpha-shrink-prior-strength must be non-negative")
    if min_denom <= 0.0:
        raise ValueError("adaptive-alpha-min-denom must be positive")

    baseline_idx, static_idx = group_indices(candidates)
    pred_arrays = [np.load(pred_path(candidate, split), mmap_mode="r") for candidate in candidates]
    true = np.load(true_path(candidates[0], split), mmap_mode="r")
    expected_shape = true.shape
    for candidate, pred in zip(candidates, pred_arrays):
        if pred.shape != expected_shape:
            raise RuntimeError(f"Unexpected pred shape for {candidate['candidate']}: {pred.shape} vs {expected_shape}")

    num_global = 0.0
    den_global = 0.0
    num_var = np.zeros(expected_shape[-1], dtype=np.float64)
    den_var = np.zeros(expected_shape[-1], dtype=np.float64)
    n_samples = expected_shape[0]
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        baseline_mean = group_mean_chunk(pred_arrays, baseline_idx, start, end)
        static_mean = group_mean_chunk(pred_arrays, static_idx, start, end)
        diff = static_mean - baseline_mean
        residual = np.asarray(true[start:end], dtype=np.float32) - baseline_mean
        prod = diff * residual
        diff_sq = np.square(diff, dtype=np.float32)
        num_global += float(prod.sum(dtype=np.float64))
        den_global += float(diff_sq.sum(dtype=np.float64))
        num_var += prod.sum(axis=(0, 1), dtype=np.float64)
        den_var += diff_sq.sum(axis=(0, 1), dtype=np.float64)
        del baseline_mean, static_mean, diff, residual, prod, diff_sq

    alpha_global_raw = num_global / den_global if den_global > min_denom else 0.5
    alpha_global = float(np.clip(alpha_global_raw, 0.0, 1.0))

    alpha_var_raw = np.full_like(den_var, alpha_global, dtype=np.float64)
    valid = den_var > min_denom
    alpha_var_raw[valid] = num_var[valid] / den_var[valid]
    alpha_var_clipped = np.clip(alpha_var_raw, 0.0, 1.0)

    valid_den = den_var[valid]
    prior_denom = float(np.median(valid_den)) if valid_den.size else 0.0
    prior_denom *= float(shrink_prior_strength)
    if prior_denom <= 0.0:
        reliability = valid.astype(np.float64)
    else:
        reliability = den_var / (den_var + prior_denom)
        reliability[~valid] = 0.0
    alpha_var_shrunk = alpha_global + reliability * (alpha_var_clipped - alpha_global)
    alpha_var_shrunk = np.clip(alpha_var_shrunk, 0.0, 1.0)

    variable_df = pd.DataFrame(
        {
            "target_index": np.arange(expected_shape[-1], dtype=np.int64),
            "alpha_raw": alpha_var_raw,
            "alpha_clipped": alpha_var_clipped,
            "alpha_shrunk": alpha_var_shrunk,
            "denominator": den_var,
            "reliability": reliability,
        }
    )
    summary = {
        "split": split,
        "alpha_global_raw": float(alpha_global_raw),
        "alpha_global_clipped": alpha_global,
        "denominator_global": float(den_global),
        "var_alpha_shrink_prior_strength": float(shrink_prior_strength),
        "var_alpha_prior_denom": float(prior_denom),
        "var_alpha_mean": float(alpha_var_shrunk.mean()),
        "var_alpha_std": float(alpha_var_shrunk.std()),
        "var_alpha_min": float(alpha_var_shrunk.min()),
        "var_alpha_max": float(alpha_var_shrunk.max()),
        "var_alpha_q05": float(np.quantile(alpha_var_shrunk, 0.05)),
        "var_alpha_q50": float(np.quantile(alpha_var_shrunk, 0.50)),
        "var_alpha_q95": float(np.quantile(alpha_var_shrunk, 0.95)),
        "valid_variable_count": int(valid.sum()),
        "variable_count": int(expected_shape[-1]),
    }
    return summary, variable_df


def candidate_weight_frame(candidates: list[dict], weights: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate": candidate["candidate"],
                "group": candidate["group"],
                "projection": candidate["projection"],
                "weight": float(weight),
            }
            for candidate, weight in zip(candidates, weights)
            if abs(float(weight)) > 1e-12
        ]
    )


def spec_weight_frame(candidates: list[dict], spec: dict) -> pd.DataFrame:
    if "weights" in spec:
        frame = candidate_weight_frame(candidates, spec["weights"])
        frame["weight_mean"] = frame["weight"]
        frame["weight_min"] = frame["weight"]
        frame["weight_max"] = frame["weight"]
        frame["weight_std"] = 0.0
        frame["alpha_mean"] = spec.get("alpha", np.nan)
        return frame

    if "alpha_vector" in spec:
        alpha = np.asarray(spec["alpha_vector"], dtype=np.float64)
        baseline_idx, static_idx = group_indices(candidates)
        rows = []
        for idx, candidate in enumerate(candidates):
            if idx in set(baseline_idx.tolist()):
                weights = (1.0 - alpha) / baseline_idx.size
            elif idx in set(static_idx.tolist()):
                weights = alpha / static_idx.size
            else:
                continue
            rows.append(
                {
                    "candidate": candidate["candidate"],
                    "group": candidate["group"],
                    "projection": candidate["projection"],
                    "weight": np.nan,
                    "weight_mean": float(weights.mean()),
                    "weight_min": float(weights.min()),
                    "weight_max": float(weights.max()),
                    "weight_std": float(weights.std()),
                    "alpha_mean": float(alpha.mean()),
                }
            )
        return pd.DataFrame(rows)

    raise ValueError(f"Unsupported spec format for {spec['ensemble']}")


def normalized_inverse(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    inv = 1.0 / np.maximum(values, eps)
    return inv / inv.sum()


def build_ensemble_specs(
    candidates: list[dict],
    candidate_val_df: pd.DataFrame,
    alphas: list[float],
    top_ks: list[int],
) -> list[dict]:
    n = len(candidates)
    names = [candidate["candidate"] for candidate in candidates]
    groups = [candidate["group"] for candidate in candidates]
    val_mse = candidate_val_df.set_index("ensemble").loc[names, "val_mse"].to_numpy(dtype=np.float64)
    order = np.argsort(val_mse)
    specs = []

    def add(name: str, kind: str, weights: np.ndarray) -> None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.sum() <= 0:
            return
        weights = weights / weights.sum()
        specs.append({"ensemble": name, "kind": kind, "weights": weights})

    for idx, candidate in enumerate(candidates):
        w = np.zeros(n, dtype=np.float64)
        w[idx] = 1.0
        add(candidate["candidate"], "single", w)

    for group in sorted(set(groups)):
        idx = np.asarray([i for i, g in enumerate(groups) if g == group], dtype=np.int64)
        w = np.zeros(n, dtype=np.float64)
        w[idx] = 1.0 / idx.size
        add(f"{group}_mean", "group_mean", w)

    add("all6_mean", "simple_mean", np.full(n, 1.0 / n, dtype=np.float64))

    baseline_idx = np.asarray([i for i, g in enumerate(groups) if g == "baseline"], dtype=np.int64)
    static_idx = np.asarray([i for i, g in enumerate(groups) if g == "static"], dtype=np.int64)
    for alpha in alphas:
        add(
            f"blend_baseline_static_alpha_{alpha:.2f}",
            "baseline_static_blend",
            group_blend_weights(candidates, float(alpha)),
        )
        specs[-1]["alpha"] = float(alpha)

    for top_k in top_ks:
        if top_k <= 0 or top_k > n:
            continue
        idx = order[:top_k]
        w = np.zeros(n, dtype=np.float64)
        w[idx] = 1.0 / top_k
        add(f"top{top_k}_mean_by_val_mse", "topk_mean", w)

        w = np.zeros(n, dtype=np.float64)
        w[idx] = normalized_inverse(val_mse[idx])
        add(f"top{top_k}_inverse_val_mse", "topk_inverse_val_mse", w)

    return specs


def add_adaptive_specs(
    specs: list[dict],
    candidates: list[dict],
    adaptive_summary: dict,
    variable_alpha_df: pd.DataFrame,
) -> None:
    alpha_global = float(adaptive_summary["alpha_global_clipped"])
    specs.append(
        {
            "ensemble": "blend_baseline_static_alpha_global_closed_form",
            "kind": "adaptive_global_alpha",
            "weights": group_blend_weights(candidates, alpha_global),
            "alpha": alpha_global,
            "alpha_source": "validation_closed_form_mse",
        }
    )
    specs.append(
        {
            "ensemble": "blend_baseline_static_alpha_variable_shrink",
            "kind": "adaptive_variable_alpha_shrink",
            "alpha_vector": variable_alpha_df["alpha_shrunk"].to_numpy(dtype=np.float64),
            "alpha_source": "validation_per_variable_closed_form_shrunk",
        }
    )


def spec_alpha_metadata(spec: dict) -> dict:
    if "alpha_vector" in spec:
        alpha = np.asarray(spec["alpha_vector"], dtype=np.float64)
        return {
            "alpha_source": spec.get("alpha_source", ""),
            "alpha": np.nan,
            "alpha_mean": float(alpha.mean()),
            "alpha_std": float(alpha.std()),
            "alpha_min": float(alpha.min()),
            "alpha_max": float(alpha.max()),
        }
    alpha = spec.get("alpha", np.nan)
    return {
        "alpha_source": spec.get("alpha_source", ""),
        "alpha": float(alpha) if pd.notna(alpha) else np.nan,
        "alpha_mean": float(alpha) if pd.notna(alpha) else np.nan,
        "alpha_std": 0.0 if pd.notna(alpha) else np.nan,
        "alpha_min": float(alpha) if pd.notna(alpha) else np.nan,
        "alpha_max": float(alpha) if pd.notna(alpha) else np.nan,
    }


def evaluate_specs(
    candidates: list[dict],
    specs: list[dict],
    split: str,
    chunk_size: int,
    progress_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    weight_rows = []
    for idx, spec in enumerate(specs):
        if "alpha_vector" in spec:
            metrics = evaluate_group_alpha(candidates, spec["alpha_vector"], split=split, chunk_size=chunk_size)
        else:
            metrics = evaluate_weighted(candidates, spec["weights"], split=split, chunk_size=chunk_size)
        row = {
            "ensemble": spec["ensemble"],
            "kind": spec["kind"],
            **spec_alpha_metadata(spec),
            f"{split}_mse": metrics["mse"],
            f"{split}_mae": metrics["mae"],
        }
        rows.append(row)
        weights = spec_weight_frame(candidates, spec)
        weights.insert(0, "kind", spec["kind"])
        weights.insert(0, "ensemble", spec["ensemble"])
        weight_rows.append(weights)
        if progress_every > 0 and ((idx + 1) % progress_every == 0 or idx + 1 == len(specs)):
            print(f"[Eval:{split}] {idx + 1}/{len(specs)}", flush=True)
    return pd.DataFrame(rows), pd.concat(weight_rows, ignore_index=True)


def add_gains(df: pd.DataFrame, reference: pd.Series, split: str) -> pd.DataFrame:
    out = df.copy()
    out[f"{split}_mse_gain_vs_best_single_pct"] = [
        pct_gain(float(reference[f"{split}_mse"]), float(value)) for value in out[f"{split}_mse"]
    ]
    out[f"{split}_mae_gain_vs_best_single_pct"] = [
        pct_gain(float(reference[f"{split}_mae"]), float(value)) for value in out[f"{split}_mae"]
    ]
    return out


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    profile = dict(PROFILES[args.profile])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.profile}_{args.tag}"

    candidates = load_candidates(profile)
    alphas = parse_float_list(args.alphas)
    top_ks = parse_int_list(args.top_ks)
    write_json(
        out_dir / f"{prefix}_manifest.json",
        {
            "profile": args.profile,
            "candidate_count": len(candidates),
            "alphas": alphas,
            "top_ks": top_ks,
            "chunk_size": args.chunk_size,
            "select_mae_min_gain": args.select_mae_min_gain,
            "adaptive_alpha_enabled": not args.disable_adaptive_alpha,
            "var_alpha_shrink_prior_strength": args.var_alpha_shrink_prior_strength,
            "adaptive_alpha_min_denom": args.adaptive_alpha_min_denom,
            "candidates": [
                {
                    "candidate": candidate["candidate"],
                    "group": candidate["group"],
                    "projection": candidate["projection"],
                    "val_dir": str(candidate["val_dir"]),
                    "test_dir": str(candidate["test_dir"]),
                }
                for candidate in candidates
            ],
        },
    )

    single_specs = []
    for idx, candidate in enumerate(candidates):
        weights = np.zeros(len(candidates), dtype=np.float64)
        weights[idx] = 1.0
        single_specs.append({"ensemble": candidate["candidate"], "kind": "single", "weights": weights})

    print("[Stage] evaluate individual candidates on validation", flush=True)
    candidate_val, _candidate_weights = evaluate_specs(
        candidates,
        single_specs,
        split="val",
        chunk_size=args.chunk_size,
        progress_every=args.progress_every,
    )
    candidate_val.to_csv(out_dir / f"{prefix}_candidate_val.csv", index=False)
    best_single_val = candidate_val.sort_values(["val_mse", "val_mae"]).iloc[0]

    adaptive_summary = None
    variable_alpha_df = None
    if not args.disable_adaptive_alpha:
        print("[Stage] estimate adaptive alpha on validation", flush=True)
        adaptive_summary, variable_alpha_df = compute_adaptive_alphas(
            candidates=candidates,
            split="val",
            chunk_size=args.chunk_size,
            shrink_prior_strength=args.var_alpha_shrink_prior_strength,
            min_denom=args.adaptive_alpha_min_denom,
        )
        write_json(out_dir / f"{prefix}_adaptive_alpha_summary.json", adaptive_summary)
        pd.DataFrame([adaptive_summary]).to_csv(out_dir / f"{prefix}_adaptive_alpha_summary.csv", index=False)
        variable_alpha_df.to_csv(out_dir / f"{prefix}_variable_alpha.csv", index=False)
        print(
            "[AdaptiveAlpha] "
            f"global_raw={adaptive_summary['alpha_global_raw']:.6f} "
            f"global_clipped={adaptive_summary['alpha_global_clipped']:.6f} "
            f"var_mean={adaptive_summary['var_alpha_mean']:.6f} "
            f"var_std={adaptive_summary['var_alpha_std']:.6f}",
            flush=True,
        )

    specs = build_ensemble_specs(
        candidates=candidates,
        candidate_val_df=candidate_val,
        alphas=alphas,
        top_ks=top_ks,
    )
    if adaptive_summary is not None and variable_alpha_df is not None:
        add_adaptive_specs(
            specs=specs,
            candidates=candidates,
            adaptive_summary=adaptive_summary,
            variable_alpha_df=variable_alpha_df,
        )
    print(f"[Stage] evaluate ensemble specs on validation: {len(specs)}", flush=True)
    val_df, weight_df = evaluate_specs(
        candidates,
        specs,
        split="val",
        chunk_size=args.chunk_size,
        progress_every=args.progress_every,
    )
    val_df = add_gains(val_df, best_single_val, split="val")
    val_df.to_csv(out_dir / f"{prefix}_val_grid.csv", index=False)
    weight_df.to_csv(out_dir / f"{prefix}_weights_all.csv", index=False)

    eligible = val_df[val_df["val_mae_gain_vs_best_single_pct"] >= float(args.select_mae_min_gain)].copy()
    if eligible.empty:
        eligible = val_df.copy()
        selection_reason = "best_val_mse_no_mae_guard_candidate"
    else:
        selection_reason = "best_val_mse_with_mae_guard"
    selected = eligible.sort_values(["val_mse", "val_mae"]).iloc[0]
    selected_weights = weight_df[weight_df["ensemble"] == selected["ensemble"]].copy()

    print(f"[Stage] evaluate selected ensemble on test: {selected['ensemble']}", flush=True)
    selected_spec = next(spec for spec in specs if spec["ensemble"] == selected["ensemble"])
    test_df, _test_weights = evaluate_specs(
        candidates,
        [selected_spec],
        split="test",
        chunk_size=args.chunk_size,
        progress_every=args.progress_every,
    )

    print("[Stage] evaluate best validation single on test for reference", flush=True)
    best_single_spec = next(spec for spec in specs if spec["ensemble"] == best_single_val["ensemble"])
    best_single_test, _ = evaluate_specs(
        candidates,
        [best_single_spec],
        split="test",
        chunk_size=args.chunk_size,
        progress_every=args.progress_every,
    )
    test_df = add_gains(test_df, best_single_test.iloc[0], split="test")

    selected_row = {
        **selected.to_dict(),
        "selection_reason": selection_reason,
        "reference_best_single": best_single_val["ensemble"],
        "reference_best_single_val_mse": float(best_single_val["val_mse"]),
        "reference_best_single_val_mae": float(best_single_val["val_mae"]),
        "reference_best_single_test_mse": float(best_single_test.iloc[0]["test_mse"]),
        "reference_best_single_test_mae": float(best_single_test.iloc[0]["test_mae"]),
        **test_df.iloc[0].to_dict(),
    }
    pd.DataFrame([selected_row]).to_csv(out_dir / f"{prefix}_selected_test_summary.csv", index=False)
    selected_weights.to_csv(out_dir / f"{prefix}_selected_weights.csv", index=False)

    print(
        "[Selected] "
        f"reason={selection_reason} ensemble={selected['ensemble']} kind={selected['kind']} "
        f"val_mse={selected['val_mse']:.6f} val_mae={selected['val_mae']:.6f} "
        f"val_mse_gain={selected['val_mse_gain_vs_best_single_pct']:.4f}% "
        f"val_mae_gain={selected['val_mae_gain_vs_best_single_pct']:.4f}%",
        flush=True,
    )
    row = test_df.iloc[0]
    print(
        "[Test] "
        f"reference_single={best_single_val['ensemble']} "
        f"test_mse={row['test_mse']:.6f} test_mae={row['test_mae']:.6f} "
        f"test_mse_gain={row['test_mse_gain_vs_best_single_pct']:.4f}% "
        f"test_mae_gain={row['test_mae_gain_vs_best_single_pct']:.4f}%",
        flush=True,
    )
    print(f"[Done] outputs written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

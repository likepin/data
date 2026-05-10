from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from posthoc_calibration.diagnostics import transform_lambda_splits
from posthoc_calibration.profiles import PROFILES, compute_selected_lambda_splits
from posthoc_calibration.schedules import gamma_from_schedule, parse_float_list
from traffic_existing_prediction_ensemble import (
    group_indices,
    group_mean_chunk,
    load_candidates,
    pred_path,
    true_path,
)
from traffic_stage3_lambda_three_source_pilot import (
    compute_dynamic_chunk,
    load_closed_loop_config,
    load_stage2_alpha,
)


DATA_ROOT = Path(r"C:\Users\cyl\Desktop\data")
DEFAULT_STAGE2_DIR = DATA_ROOT / "deltaA_signal_audit" / "etth196_existing_prediction_ensemble_parcorr"
DEFAULT_CLOSED_LOOP_DIR = DATA_ROOT / "deltaA_signal_audit" / "etth196_closed_loop_rank_quality_guard_parcorr_ridgebase_sparse"
DEFAULT_OUT_DIR = DATA_ROOT / "deltaA_signal_audit" / "etth196_stage31_target_quantile_gate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage3.1 pilot: validation-learned target-level quantile gate for "
            "lambda/dynamic correction."
        )
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="etth196_static_parcorr")
    parser.add_argument("--stage2-dir", type=Path, default=DEFAULT_STAGE2_DIR)
    parser.add_argument("--stage2-prefix", default="etth196_static_parcorr_adaptive_alpha_pilot")
    parser.add_argument("--closed-loop-dir", type=Path, default=DEFAULT_CLOSED_LOOP_DIR)
    parser.add_argument("--closed-loop-prefix", default="etth196_static_parcorr_rank_quality_guard_parcorr")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="stage31_target_quantile_gate")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-folds", type=int, default=4)
    parser.add_argument("--gamma-active-ratios", default="0.10,0.15,0.20")
    parser.add_argument("--dynamic-active-ratios", default="0.20,0.30,0.40")
    parser.add_argument(
        "--threshold-scope",
        choices=["validation_reference", "split_quantile"],
        default="split_quantile",
        help=(
            "validation_reference applies validation-set thresholds to every split; "
            "split_quantile preserves the selected active ratios inside each split "
            "using unlabeled gamma/dynamic distributions."
        ),
    )
    parser.add_argument("--eta-max", type=float, default=2.0)
    parser.add_argument("--dynamic-source", choices=["static_p0", "static_mean"], default="static_p0")
    parser.add_argument("--select-mae-min-gain", type=float, default=0.0)
    parser.add_argument("--min-positive-fold-fraction", type=float, default=0.75)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0, help="Debug cap per split. 0 means full split.")
    parser.add_argument("--shuffle-count", type=int, default=32)
    parser.add_argument("--shuffle-seed", type=int, default=20260509)
    parser.add_argument("--progress-every", type=int, default=200)
    return parser.parse_args()


def pct_gain(before: float, after: float) -> float:
    if abs(float(before)) < 1e-12:
        return 0.0
    return 100.0 * (float(before) - float(after)) / float(before)


def mse_mae_sums(err: np.ndarray) -> tuple[float, float]:
    err64 = np.asarray(err, dtype=np.float64)
    return float(np.square(err64).sum()), float(np.abs(err64).sum())


def fold_ids(n_samples: int, n_folds: int) -> np.ndarray:
    ids = np.empty(n_samples, dtype=np.int64)
    for fold, idx in enumerate(np.array_split(np.arange(n_samples), n_folds), start=1):
        ids[idx] = fold
    return ids


def quantile_threshold(values: np.ndarray, active_ratio: float) -> float:
    active_ratio = float(active_ratio)
    if active_ratio <= 0.0 or active_ratio > 1.0:
        raise ValueError(f"active ratio must be in (0, 1], got {active_ratio}")
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("cannot build quantile threshold from empty finite values")
    if active_ratio >= 1.0:
        return float(np.min(values))
    return float(np.quantile(values, 1.0 - active_ratio))


def open_prediction_arrays(candidates: list[dict], split: str) -> tuple[list[np.ndarray], np.ndarray]:
    pred_arrays = [np.load(pred_path(candidate, split), mmap_mode="r") for candidate in candidates]
    true = np.load(true_path(candidates[0], split), mmap_mode="r")
    return pred_arrays, true


def load_gamma_splits(
    *,
    profile: dict,
    closed_loop_dir: Path,
    closed_loop_prefix: str,
    seq_len: int,
    pred_len: int,
    train_ratio: float,
) -> tuple[dict[str, np.ndarray], dict, dict]:
    lambda_cfg, schedule = load_closed_loop_config(closed_loop_dir, closed_loop_prefix)
    raw_lambda_splits = compute_selected_lambda_splits(
        profile,
        lambda_cfg=lambda_cfg,
        seq_len=seq_len,
        pred_len=pred_len,
        train_ratio=train_ratio,
    )
    lambda_splits = transform_lambda_splits(raw_lambda_splits, lambda_cfg.get("lambda_transform", "raw"))
    gamma_splits = {
        split: gamma_from_schedule(values, schedule).astype(np.float32)
        for split, values in lambda_splits.items()
    }
    return gamma_splits, lambda_cfg, schedule


def dynamic_target_magnitude(dynamic: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(np.asarray(dynamic, dtype=np.float32)), axis=1))


def source_prediction(
    *,
    dynamic_source: str,
    pred_arrays: list[np.ndarray],
    static_idx: np.ndarray,
    static_mean: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    if dynamic_source == "static_mean":
        return static_mean
    return np.asarray(pred_arrays[int(static_idx[0])][start:end], dtype=np.float32)


def collect_dynamic_magnitudes(
    *,
    candidates: list[dict],
    split: str,
    delta_path: Path,
    dynamic_source: str,
    chunk_size: int,
    max_samples: int,
    progress_every: int,
) -> np.ndarray:
    pred_arrays, true = open_prediction_arrays(candidates, split)
    _, static_idx = group_indices(candidates)
    delta = np.load(delta_path, mmap_mode="r")
    n_samples = true.shape[0] if max_samples <= 0 else min(int(max_samples), true.shape[0])
    n_vars = true.shape[2]
    if delta.shape[0] < n_samples or delta.shape[1:] != (n_vars, n_vars):
        raise RuntimeError(f"Unexpected delta shape for {split}: {delta.shape}, true={true.shape}")

    values: list[np.ndarray] = []
    started = pd.Timestamp.now()
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        static_mean = group_mean_chunk(pred_arrays, static_idx, start, end)
        source_pred = source_prediction(
            dynamic_source=dynamic_source,
            pred_arrays=pred_arrays,
            static_idx=static_idx,
            static_mean=static_mean,
            start=start,
            end=end,
        )
        dynamic = compute_dynamic_chunk(source_pred, delta[start:end])
        values.append(dynamic_target_magnitude(dynamic).reshape(-1).astype(np.float32, copy=False))
        if progress_every > 0 and (end % progress_every == 0 or end == n_samples):
            elapsed = (pd.Timestamp.now() - started).total_seconds()
            print(f"[{split}:dynamic_magnitude] {end}/{n_samples} elapsed={elapsed:.1f}s", flush=True)
    return np.concatenate(values)


def build_specs(
    *,
    gamma_active_ratios: list[float],
    dynamic_active_ratios: list[float],
    gamma_thresholds: dict[float, float],
    dynamic_thresholds: dict[float, float],
) -> list[dict]:
    specs = [
        {
            "ensemble": "stage2_anchor",
            "eta_mult": 0.0,
            "eta_raw": 0.0,
            "eta_num": 0.0,
            "eta_den": 0.0,
            "eta_clip_reason": "anchor",
            "gamma_active_ratio": 0.0,
            "dynamic_active_ratio": 0.0,
            "gamma_threshold": np.inf,
            "dynamic_threshold": np.inf,
        }
    ]
    for gamma_ratio in gamma_active_ratios:
        for dynamic_ratio in dynamic_active_ratios:
            gamma_pct = int(round(float(gamma_ratio) * 100))
            dynamic_pct = int(round(float(dynamic_ratio) * 100))
            specs.append(
                {
                    "ensemble": f"target_gate_g{gamma_pct}_d{dynamic_pct}",
                    "eta_mult": np.nan,
                    "eta_raw": np.nan,
                    "eta_num": np.nan,
                    "eta_den": np.nan,
                    "eta_clip_reason": "",
                    "gamma_active_ratio": float(gamma_ratio),
                    "dynamic_active_ratio": float(dynamic_ratio),
                    "gamma_threshold": float(gamma_thresholds[float(gamma_ratio)]),
                    "dynamic_threshold": float(dynamic_thresholds[float(dynamic_ratio)]),
                }
            )
    return specs


def build_threshold_bundle(
    *,
    gamma: np.ndarray,
    dynamic_magnitudes: np.ndarray,
    gamma_active_ratios: list[float],
    dynamic_active_ratios: list[float],
) -> dict[str, dict[float, float]]:
    return {
        "gamma": {
            float(ratio): quantile_threshold(gamma, float(ratio))
            for ratio in gamma_active_ratios
        },
        "dynamic": {
            float(ratio): quantile_threshold(dynamic_magnitudes, float(ratio))
            for ratio in dynamic_active_ratios
        },
    }


def thresholds_for_spec(spec: dict, threshold_bundle: dict[str, dict[float, float]] | None) -> tuple[float, float]:
    if threshold_bundle is None or spec["ensemble"] == "stage2_anchor":
        return float(spec["gamma_threshold"]), float(spec["dynamic_threshold"])
    gamma_threshold = threshold_bundle["gamma"][float(spec["gamma_active_ratio"])]
    dynamic_threshold = threshold_bundle["dynamic"][float(spec["dynamic_active_ratio"])]
    return float(gamma_threshold), float(dynamic_threshold)


def fold_stability_summary(fold_grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ensemble, group in fold_grid.groupby("ensemble"):
        mse_gain = group["mse_gain_vs_stage2_anchor_pct"].to_numpy(dtype=np.float64)
        mae_gain = group["mae_gain_vs_stage2_anchor_pct"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "ensemble": ensemble,
                "fold_mse_gain_mean": float(mse_gain.mean()),
                "fold_mse_gain_min": float(mse_gain.min()),
                "fold_mse_positive_fraction": float(np.mean(mse_gain > 0.0)),
                "fold_mae_gain_mean": float(mae_gain.mean()),
                "fold_mae_gain_min": float(mae_gain.min()),
                "fold_mae_positive_fraction": float(np.mean(mae_gain > 0.0)),
            }
        )
    return pd.DataFrame(rows)


def estimate_eta_for_specs(
    *,
    candidates: list[dict],
    alpha: np.ndarray,
    split: str,
    delta_path: Path,
    gamma: np.ndarray,
    specs: list[dict],
    dynamic_source: str,
    chunk_size: int,
    max_samples: int,
    progress_every: int,
    eta_max: float,
) -> list[dict]:
    if eta_max < 0.0:
        raise ValueError(f"eta_max must be non-negative, got {eta_max}")
    pred_arrays, true = open_prediction_arrays(candidates, split)
    baseline_idx, static_idx = group_indices(candidates)
    delta = np.load(delta_path, mmap_mode="r")
    n_samples = true.shape[0] if max_samples <= 0 else min(int(max_samples), true.shape[0])
    n_vars = true.shape[2]
    if len(gamma) < n_samples:
        raise RuntimeError(f"Gamma too short for {split}: {len(gamma)} vs {n_samples}")
    if alpha.size != n_vars:
        raise RuntimeError(f"Alpha length mismatch: {alpha.size} vs {n_vars}")
    if delta.shape[0] < n_samples or delta.shape[1:] != (n_vars, n_vars):
        raise RuntimeError(f"Unexpected delta shape for {split}: {delta.shape}, true={true.shape}")

    sums = {
        spec["ensemble"]: {"num": 0.0, "den": 0.0}
        for spec in specs
        if spec["ensemble"] != "stage2_anchor"
    }
    alpha_view = np.asarray(alpha, dtype=np.float32).reshape(1, 1, -1)
    gamma = np.asarray(gamma[:n_samples], dtype=np.float32)
    started = pd.Timestamp.now()
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        baseline_mean = group_mean_chunk(pred_arrays, baseline_idx, start, end)
        static_mean = group_mean_chunk(pred_arrays, static_idx, start, end)
        anchor = baseline_mean + alpha_view * (static_mean - baseline_mean)
        err_anchor = np.asarray(true[start:end], dtype=np.float32) - anchor
        source_pred = source_prediction(
            dynamic_source=dynamic_source,
            pred_arrays=pred_arrays,
            static_idx=static_idx,
            static_mean=static_mean,
            start=start,
            end=end,
        )
        dynamic = compute_dynamic_chunk(source_pred, delta[start:end])
        dyn_mag = dynamic_target_magnitude(dynamic)
        gamma_chunk = gamma[start:end]
        err64 = err_anchor.astype(np.float64, copy=False)
        for spec in specs:
            if spec["ensemble"] == "stage2_anchor":
                continue
            gamma_gate = (gamma_chunk >= float(spec["gamma_threshold"])).reshape(-1, 1, 1)
            target_gate = (dyn_mag >= float(spec["dynamic_threshold"])).reshape(end - start, 1, n_vars)
            z = (
                gamma_chunk.reshape(-1, 1, 1)
                * dynamic
                * alpha_view
                * gamma_gate
                * target_gate
            ).astype(np.float64, copy=False)
            sums[spec["ensemble"]]["num"] += float((err64 * z).sum())
            sums[spec["ensemble"]]["den"] += float(np.square(z).sum())
        if progress_every > 0 and (end % progress_every == 0 or end == n_samples):
            elapsed = (pd.Timestamp.now() - started).total_seconds()
            print(f"[{split}:eta_target_gate] {end}/{n_samples} elapsed={elapsed:.1f}s", flush=True)

    out = []
    for spec in specs:
        spec = dict(spec)
        if spec["ensemble"] == "stage2_anchor":
            out.append(spec)
            continue
        num = sums[spec["ensemble"]]["num"]
        den = sums[spec["ensemble"]]["den"]
        if den <= 1e-12:
            eta_raw = 0.0
            eta = 0.0
            reason = "zero_dynamic_energy"
        else:
            eta_raw = num / den
            eta = min(max(eta_raw, 0.0), float(eta_max))
            if eta_raw < 0.0:
                reason = "clipped_low"
            elif eta_raw > float(eta_max):
                reason = "clipped_high"
            else:
                reason = "unclipped"
        spec.update(
            {
                "eta_mult": float(eta),
                "eta_raw": float(eta_raw),
                "eta_num": float(num),
                "eta_den": float(den),
                "eta_clip_reason": reason,
            }
        )
        out.append(spec)
    return out


def shuffled_target_gate(target_gate: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = np.empty_like(target_gate)
    for i in range(target_gate.shape[0]):
        shuffled[i] = rng.permutation(target_gate[i])
    return shuffled


def evaluate_specs(
    *,
    candidates: list[dict],
    alpha: np.ndarray,
    split: str,
    delta_path: Path,
    gamma: np.ndarray,
    specs: list[dict],
    dynamic_source: str,
    chunk_size: int,
    validation_folds: int,
    max_samples: int,
    progress_every: int,
    control_mode: str = "observed",
    seed: int | None = None,
    threshold_bundle: dict[str, dict[float, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_arrays, true = open_prediction_arrays(candidates, split)
    baseline_idx, static_idx = group_indices(candidates)
    delta = np.load(delta_path, mmap_mode="r")
    n_samples = true.shape[0] if max_samples <= 0 else min(int(max_samples), true.shape[0])
    n_horizon = true.shape[1]
    n_vars = true.shape[2]
    if len(gamma) < n_samples:
        raise RuntimeError(f"Gamma too short for {split}: {len(gamma)} vs {n_samples}")
    if alpha.size != n_vars:
        raise RuntimeError(f"Alpha length mismatch: {alpha.size} vs {n_vars}")
    if delta.shape[0] < n_samples or delta.shape[1:] != (n_vars, n_vars):
        raise RuntimeError(f"Unexpected delta shape for {split}: {delta.shape}, true={true.shape}")
    if control_mode not in {"observed", "shuffle_gamma", "shuffle_target"}:
        raise ValueError(f"Unknown control_mode: {control_mode}")

    rng = np.random.default_rng(seed) if seed is not None else None
    gamma_eval = np.asarray(gamma[:n_samples], dtype=np.float32)
    if control_mode == "shuffle_gamma":
        if rng is None:
            raise ValueError("shuffle_gamma requires seed")
        gamma_eval = rng.permutation(gamma_eval).astype(np.float32, copy=False)

    alpha_view = np.asarray(alpha, dtype=np.float32).reshape(1, 1, -1)
    fold_index = fold_ids(n_samples, validation_folds)
    count = n_samples * n_horizon * n_vars
    totals = {
        spec["ensemble"]: {"sse": 0.0, "sae": 0.0, "count": count, "active_units": 0, "unit_count": n_samples * n_vars}
        for spec in specs
    }
    fold_totals = {
        (spec["ensemble"], fold): {
            "sse": 0.0,
            "sae": 0.0,
            "count": int(np.sum(fold_index == fold)) * n_horizon * n_vars,
            "active_units": 0,
            "unit_count": int(np.sum(fold_index == fold)) * n_vars,
        }
        for spec in specs
        for fold in range(1, validation_folds + 1)
    }

    started = pd.Timestamp.now()
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        baseline_mean = group_mean_chunk(pred_arrays, baseline_idx, start, end)
        static_mean = group_mean_chunk(pred_arrays, static_idx, start, end)
        anchor = baseline_mean + alpha_view * (static_mean - baseline_mean)
        err_anchor = np.asarray(true[start:end], dtype=np.float32) - anchor
        source_pred = source_prediction(
            dynamic_source=dynamic_source,
            pred_arrays=pred_arrays,
            static_idx=static_idx,
            static_mean=static_mean,
            start=start,
            end=end,
        )
        dynamic = compute_dynamic_chunk(source_pred, delta[start:end])
        dyn_mag = dynamic_target_magnitude(dynamic)
        gamma_chunk = gamma_eval[start:end]
        fold_chunk = fold_index[start:end]
        for spec in specs:
            eta = float(spec["eta_mult"])
            if spec["ensemble"] == "stage2_anchor" or abs(eta) <= 1e-12:
                err = err_anchor
                active_target = np.zeros((end - start, n_vars), dtype=bool)
            else:
                gamma_threshold, dynamic_threshold = thresholds_for_spec(spec, threshold_bundle)
                gamma_gate = gamma_chunk >= gamma_threshold
                target_gate = dyn_mag >= dynamic_threshold
                if control_mode == "shuffle_target":
                    if rng is None:
                        raise ValueError("shuffle_target requires seed")
                    target_gate = shuffled_target_gate(target_gate, rng)
                active_target = gamma_gate.reshape(-1, 1) & target_gate
                z = (
                    gamma_chunk.reshape(-1, 1, 1)
                    * dynamic
                    * alpha_view
                    * active_target.reshape(end - start, 1, n_vars)
                )
                err = err_anchor - eta * z
            sse, sae = mse_mae_sums(err)
            totals[spec["ensemble"]]["sse"] += sse
            totals[spec["ensemble"]]["sae"] += sae
            totals[spec["ensemble"]]["active_units"] += int(active_target.sum())
            for fold in range(1, validation_folds + 1):
                local = np.where(fold_chunk == fold)[0]
                if local.size == 0:
                    continue
                fsse, fsae = mse_mae_sums(err[local])
                fold_totals[(spec["ensemble"], fold)]["sse"] += fsse
                fold_totals[(spec["ensemble"], fold)]["sae"] += fsae
                fold_totals[(spec["ensemble"], fold)]["active_units"] += int(active_target[local].sum())
        if progress_every > 0 and (end % progress_every == 0 or end == n_samples):
            elapsed = (pd.Timestamp.now() - started).total_seconds()
            print(f"[{split}:{control_mode}] {end}/{n_samples} elapsed={elapsed:.1f}s", flush=True)

    rows = []
    anchor = totals["stage2_anchor"]
    anchor_mse = anchor["sse"] / anchor["count"]
    anchor_mae = anchor["sae"] / anchor["count"]
    specs_by_name = {spec["ensemble"]: spec for spec in specs}
    for name, total in totals.items():
        spec = specs_by_name[name]
        gamma_threshold, dynamic_threshold = thresholds_for_spec(spec, threshold_bundle)
        mse = total["sse"] / total["count"]
        mae = total["sae"] / total["count"]
        rows.append(
            {
                "split": split,
                "control_mode": control_mode,
                "ensemble": name,
                "eta_mult": float(spec["eta_mult"]),
                "eta_raw": float(spec["eta_raw"]),
                "eta_num": float(spec["eta_num"]),
                "eta_den": float(spec["eta_den"]),
                "eta_clip_reason": spec["eta_clip_reason"],
                "gamma_active_ratio": float(spec["gamma_active_ratio"]),
                "dynamic_active_ratio": float(spec["dynamic_active_ratio"]),
                "gamma_threshold": gamma_threshold,
                "dynamic_threshold": dynamic_threshold,
                "target_gate_active_ratio": total["active_units"] / total["unit_count"] if total["unit_count"] else 0.0,
                "active_target_units": int(total["active_units"]),
                "target_unit_count": int(total["unit_count"]),
                "dynamic_source": dynamic_source,
                "mse": mse,
                "mae": mae,
                "mse_gain_vs_stage2_anchor_pct": pct_gain(anchor_mse, mse),
                "mae_gain_vs_stage2_anchor_pct": pct_gain(anchor_mae, mae),
                "n_samples": n_samples,
            }
        )

    fold_rows = []
    for (name, fold), total in fold_totals.items():
        spec = specs_by_name[name]
        if total["count"] <= 0:
            continue
        anchor_fold = fold_totals[("stage2_anchor", fold)]
        anchor_fold_mse = anchor_fold["sse"] / anchor_fold["count"]
        anchor_fold_mae = anchor_fold["sae"] / anchor_fold["count"]
        mse = total["sse"] / total["count"]
        mae = total["sae"] / total["count"]
        fold_rows.append(
            {
                "split": split,
                "fold": fold,
                "control_mode": control_mode,
                "ensemble": name,
                "eta_mult": float(spec["eta_mult"]),
                "eta_raw": float(spec["eta_raw"]),
                "eta_clip_reason": spec["eta_clip_reason"],
                "gamma_active_ratio": float(spec["gamma_active_ratio"]),
                "dynamic_active_ratio": float(spec["dynamic_active_ratio"]),
                "target_gate_active_ratio": total["active_units"] / total["unit_count"] if total["unit_count"] else 0.0,
                "mse": mse,
                "mae": mae,
                "mse_gain_vs_stage2_anchor_pct": pct_gain(anchor_fold_mse, mse),
                "mae_gain_vs_stage2_anchor_pct": pct_gain(anchor_fold_mae, mae),
                "n_samples": int(np.sum(fold_index == fold)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


def selected_control_summary(
    *,
    candidates: list[dict],
    alpha: np.ndarray,
    split: str,
    delta_path: Path,
    gamma: np.ndarray,
    selected_spec: dict,
    dynamic_source: str,
    chunk_size: int,
    validation_folds: int,
    max_samples: int,
    shuffle_count: int,
    shuffle_seed: int,
    progress_every: int,
    threshold_bundle: dict[str, dict[float, float]] | None,
) -> pd.DataFrame:
    rows = []
    if shuffle_count <= 0 or selected_spec["ensemble"] == "stage2_anchor":
        return pd.DataFrame(rows)
    for mode, seed_offset in [("shuffle_gamma", 0), ("shuffle_target", 100000)]:
        values = []
        for i in range(shuffle_count):
            grid, _ = evaluate_specs(
                candidates=candidates,
                alpha=alpha,
                split=split,
                delta_path=delta_path,
                gamma=gamma,
                specs=[selected_spec, {
                    "ensemble": "stage2_anchor",
                    "eta_mult": 0.0,
                    "eta_raw": 0.0,
                    "eta_num": 0.0,
                    "eta_den": 0.0,
                    "eta_clip_reason": "anchor",
                    "gamma_active_ratio": 0.0,
                    "dynamic_active_ratio": 0.0,
                    "gamma_threshold": np.inf,
                    "dynamic_threshold": np.inf,
                }],
                dynamic_source=dynamic_source,
                chunk_size=chunk_size,
                validation_folds=validation_folds,
                max_samples=max_samples,
                progress_every=0,
                control_mode=mode,
                seed=shuffle_seed + seed_offset + i,
                threshold_bundle=threshold_bundle,
            )
            selected = grid[grid["ensemble"] == selected_spec["ensemble"]].iloc[0]
            values.append(selected)
        frame = pd.DataFrame(values)
        rows.append(
            {
                "split": split,
                "control_mode": mode,
                "shuffle_count": int(shuffle_count),
                "shuffle_seed": int(shuffle_seed + seed_offset),
                "selected_ensemble": selected_spec["ensemble"],
                "mse_mean": float(frame["mse"].mean()),
                "mse_median": float(frame["mse"].median()),
                "mse_q05": float(frame["mse"].quantile(0.05)),
                "mse_q95": float(frame["mse"].quantile(0.95)),
                "mae_mean": float(frame["mae"].mean()),
                "mae_median": float(frame["mae"].median()),
                "mae_q05": float(frame["mae"].quantile(0.05)),
                "mae_q95": float(frame["mae"].quantile(0.95)),
                "mse_gain_vs_stage2_anchor_pct_median": float(frame["mse_gain_vs_stage2_anchor_pct"].median()),
                "mae_gain_vs_stage2_anchor_pct_median": float(frame["mae_gain_vs_stage2_anchor_pct"].median()),
                "target_gate_active_ratio_median": float(frame["target_gate_active_ratio"].median()),
            }
        )
    return pd.DataFrame(rows)


def selected_plus_anchor_specs(selected_spec: dict, anchor_spec: dict) -> list[dict]:
    if selected_spec["ensemble"] == anchor_spec["ensemble"]:
        return [anchor_spec]
    return [selected_spec, anchor_spec]


def selection_reason_label(args: argparse.Namespace, fallback: bool) -> str:
    relaxed = float(args.select_mae_min_gain) < 0.0 or float(args.min_positive_fold_fraction) <= 0.0
    if fallback:
        if relaxed:
            return "fallback_stage2_anchor_no_relaxed_mse_candidate"
        return "fallback_stage2_anchor_no_fold_stable_candidate"
    if relaxed:
        return "best_val_mse_relaxed_mae_or_fold_guard"
    return "best_val_mse_with_mae_and_fold_stability_guard"


def markdown_summary(selected_val: dict, selected_test: dict, controls: pd.DataFrame) -> str:
    lines = [
        "# Stage3.1 Target Quantile Gate Pilot",
        "",
        "Selected validation candidate:",
        (
            f"- `{selected_val['ensemble']}` with gamma active ratio "
            f"`{selected_val['gamma_active_ratio']:.2f}`, dynamic active ratio "
            f"`{selected_val['dynamic_active_ratio']:.2f}`, eta `{selected_val['eta_mult']:.6f}`."
        ),
        (
            f"- Validation: `{selected_val['mse']:.6f} / {selected_val['mae']:.6f}`, "
            f"gain vs adaptive anchor `{selected_val['mse_gain_vs_stage2_anchor_pct']:+.4f}% / "
            f"{selected_val['mae_gain_vs_stage2_anchor_pct']:+.4f}%`."
        ),
        (
            f"- Test: `{selected_test['mse']:.6f} / {selected_test['mae']:.6f}`, "
            f"gain vs adaptive anchor `{selected_test['mse_gain_vs_stage2_anchor_pct']:+.4f}% / "
            f"{selected_test['mae_gain_vs_stage2_anchor_pct']:+.4f}%`."
        ),
        "",
        "Notes:",
        "- `gamma_threshold` and `dynamic_threshold` are learned from validation-set quantiles.",
        "- In `split_quantile` mode, test thresholds are recomputed from unlabeled test-side gamma/dynamic distributions to preserve the selected active ratios.",
        "- `alpha_shrunk` is used as a continuous target confidence multiplier, not as a new hand-tuned threshold.",
        "- Shuffle controls break gamma time alignment or target alignment while preserving the candidate structure.",
    ]
    if not controls.empty:
        lines.extend(["", "Shuffle controls:"])
        for _, row in controls.iterrows():
            lines.append(
                f"- `{row['split']}` `{row['control_mode']}` median MSE/MAE: "
                f"`{row['mse_median']:.6f} / {row['mae_median']:.6f}`."
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    profile = dict(PROFILES[args.profile])
    prefix = f"{args.profile}_{args.tag}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gamma_active_ratios = [float(x) for x in parse_float_list(args.gamma_active_ratios)]
    dynamic_active_ratios = [float(x) for x in parse_float_list(args.dynamic_active_ratios)]
    alpha = load_stage2_alpha(args.stage2_dir, args.stage2_prefix)
    candidates = load_candidates(profile)
    interface_dir = Path(profile["interface_dir"])
    gamma_splits, lambda_cfg, schedule = load_gamma_splits(
        profile=profile,
        closed_loop_dir=args.closed_loop_dir,
        closed_loop_prefix=args.closed_loop_prefix,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
    )

    n_val = len(gamma_splits["val"]) if args.max_samples <= 0 else min(args.max_samples, len(gamma_splits["val"]))
    gamma_val = np.asarray(gamma_splits["val"][:n_val], dtype=np.float32)
    val_dynamic_mags = collect_dynamic_magnitudes(
        candidates=candidates,
        split="val",
        delta_path=interface_dir / "deltaA_val.npy",
        dynamic_source=args.dynamic_source,
        chunk_size=args.chunk_size,
        max_samples=args.max_samples,
        progress_every=args.progress_every,
    )
    val_threshold_bundle = build_threshold_bundle(
        gamma=gamma_val,
        dynamic_magnitudes=val_dynamic_mags,
        gamma_active_ratios=gamma_active_ratios,
        dynamic_active_ratios=dynamic_active_ratios,
    )
    gamma_thresholds = val_threshold_bundle["gamma"]
    dynamic_thresholds = val_threshold_bundle["dynamic"]
    specs = build_specs(
        gamma_active_ratios=gamma_active_ratios,
        dynamic_active_ratios=dynamic_active_ratios,
        gamma_thresholds=gamma_thresholds,
        dynamic_thresholds=dynamic_thresholds,
    )
    specs = estimate_eta_for_specs(
        candidates=candidates,
        alpha=alpha,
        split="val",
        delta_path=interface_dir / "deltaA_val.npy",
        gamma=gamma_splits["val"],
        specs=specs,
        dynamic_source=args.dynamic_source,
        chunk_size=args.chunk_size,
        max_samples=args.max_samples,
        progress_every=args.progress_every,
        eta_max=args.eta_max,
    )
    pd.DataFrame(specs).to_csv(args.out_dir / f"{prefix}_candidates.csv", index=False)

    val_grid, val_folds = evaluate_specs(
        candidates=candidates,
        alpha=alpha,
        split="val",
        delta_path=interface_dir / "deltaA_val.npy",
        gamma=gamma_splits["val"],
        specs=specs,
        dynamic_source=args.dynamic_source,
        chunk_size=args.chunk_size,
        validation_folds=args.validation_folds,
        max_samples=args.max_samples,
        progress_every=args.progress_every,
        threshold_bundle=val_threshold_bundle if args.threshold_scope == "split_quantile" else None,
    )
    val_folds.to_csv(args.out_dir / f"{prefix}_val_fold_grid.csv", index=False)
    fold_summary = fold_stability_summary(val_folds)
    fold_summary.to_csv(args.out_dir / f"{prefix}_val_fold_stability.csv", index=False)
    val_grid = val_grid.merge(fold_summary, on="ensemble", how="left")
    val_grid.to_csv(args.out_dir / f"{prefix}_val_grid.csv", index=False)

    eligible = val_grid[val_grid["mae_gain_vs_stage2_anchor_pct"] >= float(args.select_mae_min_gain)].copy()
    eligible = eligible[
        (eligible["fold_mse_positive_fraction"] >= float(args.min_positive_fold_fraction))
        & (eligible["fold_mae_positive_fraction"] >= float(args.min_positive_fold_fraction))
    ].copy()
    if eligible.empty:
        selected = val_grid[val_grid["ensemble"] == "stage2_anchor"].iloc[0]
        selection_reason = selection_reason_label(args, fallback=True)
    else:
        selected = eligible.sort_values(["mse", "mae"]).iloc[0]
        selection_reason = selection_reason_label(args, fallback=False)
    selected_spec = next(spec for spec in specs if spec["ensemble"] == selected["ensemble"])
    selected_val = {**selected.to_dict(), "selection_reason": selection_reason}
    pd.DataFrame([selected_val]).to_csv(args.out_dir / f"{prefix}_selected_val_summary.csv", index=False)

    test_threshold_bundle = None
    if args.threshold_scope == "split_quantile":
        n_test = len(gamma_splits["test"]) if args.max_samples <= 0 else min(args.max_samples, len(gamma_splits["test"]))
        test_dynamic_mags = collect_dynamic_magnitudes(
            candidates=candidates,
            split="test",
            delta_path=interface_dir / "deltaA_test.npy",
            dynamic_source=args.dynamic_source,
            chunk_size=args.chunk_size,
            max_samples=args.max_samples,
            progress_every=args.progress_every,
        )
        test_threshold_bundle = build_threshold_bundle(
            gamma=np.asarray(gamma_splits["test"][:n_test], dtype=np.float32),
            dynamic_magnitudes=test_dynamic_mags,
            gamma_active_ratios=gamma_active_ratios,
            dynamic_active_ratios=dynamic_active_ratios,
        )

    test_grid, _ = evaluate_specs(
        candidates=candidates,
        alpha=alpha,
        split="test",
        delta_path=interface_dir / "deltaA_test.npy",
        gamma=gamma_splits["test"],
        specs=selected_plus_anchor_specs(selected_spec, specs[0]),
        dynamic_source=args.dynamic_source,
        chunk_size=args.chunk_size,
        validation_folds=args.validation_folds,
        max_samples=args.max_samples,
        progress_every=args.progress_every,
        threshold_bundle=test_threshold_bundle,
    )
    test_grid.to_csv(args.out_dir / f"{prefix}_test_grid_selected.csv", index=False)
    selected_test = test_grid[test_grid["ensemble"] == selected_spec["ensemble"]].iloc[0].to_dict()
    selected_test["selection_reason"] = selection_reason
    pd.DataFrame([selected_test]).to_csv(args.out_dir / f"{prefix}_test_selected_summary.csv", index=False)

    controls = pd.concat(
        [
            selected_control_summary(
                candidates=candidates,
                alpha=alpha,
                split="val",
                delta_path=interface_dir / "deltaA_val.npy",
                gamma=gamma_splits["val"],
                selected_spec=selected_spec,
                dynamic_source=args.dynamic_source,
                chunk_size=args.chunk_size,
                validation_folds=args.validation_folds,
                max_samples=args.max_samples,
                shuffle_count=args.shuffle_count,
                shuffle_seed=args.shuffle_seed,
                progress_every=args.progress_every,
                threshold_bundle=val_threshold_bundle if args.threshold_scope == "split_quantile" else None,
            ),
            selected_control_summary(
                candidates=candidates,
                alpha=alpha,
                split="test",
                delta_path=interface_dir / "deltaA_test.npy",
                gamma=gamma_splits["test"],
                selected_spec=selected_spec,
                dynamic_source=args.dynamic_source,
                chunk_size=args.chunk_size,
                validation_folds=args.validation_folds,
                max_samples=args.max_samples,
                shuffle_count=args.shuffle_count,
                shuffle_seed=args.shuffle_seed + 1000,
                progress_every=args.progress_every,
                threshold_bundle=test_threshold_bundle,
            ),
        ],
        ignore_index=True,
    )
    controls.to_csv(args.out_dir / f"{prefix}_shuffle_controls.csv", index=False)

    manifest = {
        "profile": args.profile,
        "tag": args.tag,
        "stage2_dir": str(args.stage2_dir),
        "stage2_prefix": args.stage2_prefix,
        "closed_loop_dir": str(args.closed_loop_dir),
        "closed_loop_prefix": args.closed_loop_prefix,
        "interface_dir": str(interface_dir),
        "dynamic_source": args.dynamic_source,
        "gamma_active_ratios": gamma_active_ratios,
        "dynamic_active_ratios": dynamic_active_ratios,
        "gamma_thresholds": {str(key): value for key, value in gamma_thresholds.items()},
        "dynamic_thresholds": {str(key): value for key, value in dynamic_thresholds.items()},
        "threshold_scope": args.threshold_scope,
        "test_gamma_thresholds": (
            {str(key): value for key, value in test_threshold_bundle["gamma"].items()}
            if test_threshold_bundle is not None
            else None
        ),
        "test_dynamic_thresholds": (
            {str(key): value for key, value in test_threshold_bundle["dynamic"].items()}
            if test_threshold_bundle is not None
            else None
        ),
        "alpha_mode": "alpha_shrunk_continuous_multiplier",
        "eta_mode": "closed_form_per_gate_validation",
        "eta_max": args.eta_max,
        "lambda_cfg": lambda_cfg,
        "schedule": schedule,
        "selection_reason": selection_reason,
        "selected_ensemble": selected_spec["ensemble"],
        "min_positive_fold_fraction": args.min_positive_fold_fraction,
        "chunk_size": args.chunk_size,
        "max_samples": args.max_samples,
        "shuffle_count": args.shuffle_count,
        "shuffle_seed": args.shuffle_seed,
    }
    (args.out_dir / f"{prefix}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.out_dir / f"{prefix}_README.md").write_text(
        markdown_summary(selected_val, selected_test, controls),
        encoding="utf-8",
    )

    print(
        "[Selected] "
        f"ensemble={selected_spec['ensemble']} reason={selection_reason} "
        f"val_mse={selected_val['mse']:.6f} val_mae={selected_val['mae']:.6f} "
        f"val_gain={selected_val['mse_gain_vs_stage2_anchor_pct']:.4f}%/"
        f"{selected_val['mae_gain_vs_stage2_anchor_pct']:.4f}%",
        flush=True,
    )
    print(
        "[Test] "
        f"mse={selected_test['mse']:.6f} mae={selected_test['mae']:.6f} "
        f"gain={selected_test['mse_gain_vs_stage2_anchor_pct']:.4f}%/"
        f"{selected_test['mae_gain_vs_stage2_anchor_pct']:.4f}%",
        flush=True,
    )
    print(f"[Done] outputs written to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

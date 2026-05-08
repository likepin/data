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


DATA_ROOT = Path(r"C:\Users\cyl\Desktop\data")
DEFAULT_STAGE2_DIR = DATA_ROOT / "deltaA_signal_audit" / "traffic96_existing_prediction_ensemble_stage2_light_seed2026"
DEFAULT_CLOSED_LOOP_DIR = DATA_ROOT / "deltaA_signal_audit" / "traffic96_closed_loop_log_tail_quality_guard"
DEFAULT_OUT_DIR = DATA_ROOT / "deltaA_signal_audit" / "traffic96_stage3_lambda_three_source_pilot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage3 pilot for Traffic: validation-selected lambda-aware integration "
            "of Stage2 baseline/static anchor with post-hoc DeltaA correction."
        )
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="traffic96_static")
    parser.add_argument("--stage2-dir", type=Path, default=DEFAULT_STAGE2_DIR)
    parser.add_argument("--stage2-prefix", default="")
    parser.add_argument("--closed-loop-dir", type=Path, default=DEFAULT_CLOSED_LOOP_DIR)
    parser.add_argument("--closed-loop-prefix", default="")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="stage3_pilot")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-folds", type=int, default=4)
    parser.add_argument("--eta-mode", choices=["grid", "closed_form"], default="grid")
    parser.add_argument("--eta-mults", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--eta-max", type=float, default=2.0)
    parser.add_argument("--target-masks", default="all,top_alpha_5,top_alpha_10")
    parser.add_argument("--dynamic-source", choices=["static_p0", "static_mean"], default="static_p0")
    parser.add_argument("--select-mae-min-gain", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--shuffle-count", type=int, default=256)
    parser.add_argument("--shuffle-seed", type=int, default=20260507)
    parser.add_argument("--max-samples", type=int, default=0, help="Debug cap per split. 0 means full split.")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def read_one_row(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError(f"Expected one-row CSV: {path}, got {len(df)}")
    return df.iloc[0].to_dict()


def finite_float(value, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def load_stage2_alpha(stage2_dir: Path, prefix: str) -> np.ndarray:
    path = stage2_dir / f"{prefix}_variable_alpha.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "alpha_shrunk" not in df:
        raise ValueError(f"Missing alpha_shrunk in {path}")
    alpha = df["alpha_shrunk"].to_numpy(dtype=np.float32)
    if alpha.ndim != 1:
        raise ValueError(f"Unexpected alpha shape: {alpha.shape}")
    return alpha


def load_closed_loop_config(closed_loop_dir: Path, prefix: str) -> tuple[dict, dict]:
    lambda_row = read_one_row(closed_loop_dir / f"{prefix}_closed_loop_lambda_selected.csv")
    schedule_row = read_one_row(closed_loop_dir / f"{prefix}_closed_loop_schedule_selected.csv")
    lambda_cfg = {
        "mode": str(lambda_row["mode"]),
        "window": int(lambda_row["window"]),
        "k": int(lambda_row["k"]),
        "lambda_scale": str(lambda_row.get("lambda_scale", "legacy_clipped")),
        "tail_target_width": finite_float(lambda_row.get("tail_target_width"), 0.10),
        "tail_alpha_min": finite_float(lambda_row.get("tail_alpha_min"), 0.02),
        "tail_alpha_max": finite_float(lambda_row.get("tail_alpha_max"), 0.20),
        "stable_candidate": str(lambda_row.get("stable_candidate", "False")).lower() == "true",
        "stability_score": finite_float(lambda_row.get("stability_score"), np.nan),
        "fold_spearman_mean": finite_float(lambda_row.get("fold_spearman_mean"), np.nan),
        "fold_bucket5_lift_mean": finite_float(lambda_row.get("fold_bucket5_lift_mean"), np.nan),
        "source_file": str(lambda_row.get("source_file", "")),
        "quality_guard_reason": str(lambda_row.get("quality_guard_reason", "")),
        "quality_score": finite_float(lambda_row.get("quality_score"), np.nan),
        "lambda_transform": str(lambda_row.get("lambda_transform", "raw")),
    }
    schedule = {
        "q_low": finite_float(schedule_row["q_low"]),
        "q_high": finite_float(schedule_row["q_high"]),
        "q_low_value": finite_float(schedule_row["q_low_value"]),
        "q_high_value": finite_float(schedule_row["q_high_value"]),
        "gamma_min": finite_float(schedule_row["gamma_min"]),
        "gamma_max": finite_float(schedule_row["gamma_max"]),
        "passes_selection": str(schedule_row.get("passes_selection", "True")).lower() == "true",
        "mode_status": str(schedule_row.get("mode_status", "")),
        "mode_reason": str(schedule_row.get("mode_reason", "")),
        "selection_reason": str(schedule_row.get("selection_reason", "")),
    }
    return lambda_cfg, schedule


def build_target_masks(mask_names: list[str], alpha: np.ndarray) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    n_vars = alpha.size
    for name in mask_names:
        name = name.strip()
        if not name:
            continue
        if name == "all":
            masks[name] = np.ones(n_vars, dtype=bool)
        elif name.startswith("top_alpha_"):
            pct = float(name.removeprefix("top_alpha_")) / 100.0
            if pct <= 0 or pct > 1:
                raise ValueError(f"Invalid target mask percentile: {name}")
            k = max(1, int(np.ceil(n_vars * pct)))
            idx = np.argsort(alpha)[-k:]
            mask = np.zeros(n_vars, dtype=bool)
            mask[idx] = True
            masks[name] = mask
        else:
            raise ValueError(f"Unknown target mask: {name}")
    if "all" not in masks:
        masks = {"all": np.ones(n_vars, dtype=bool), **masks}
    return masks


def anchor_spec() -> dict:
    return {
        "ensemble": "stage2_anchor",
        "eta_mode": "anchor",
        "eta_mult": 0.0,
        "eta_raw": 0.0,
        "eta_num": 0.0,
        "eta_den": 0.0,
        "eta_clip_reason": "anchor",
        "target_mask": "all",
    }


def candidate_specs(eta_mults: list[float], target_masks: dict[str, np.ndarray]) -> list[dict]:
    specs = [anchor_spec()]
    for eta in eta_mults:
        eta = float(eta)
        if abs(eta) <= 1e-12:
            continue
        for mask_name in target_masks:
            specs.append(
                {
                    "ensemble": f"stage3_eta{eta:.3g}_{mask_name}",
                    "eta_mode": "grid",
                    "eta_mult": eta,
                    "eta_raw": eta,
                    "eta_num": np.nan,
                    "eta_den": np.nan,
                    "eta_clip_reason": "grid",
                    "target_mask": mask_name,
                }
            )
    return specs


def open_prediction_arrays(candidates: list[dict], split: str) -> tuple[list[np.ndarray], np.ndarray]:
    pred_arrays = [np.load(pred_path(candidate, split), mmap_mode="r") for candidate in candidates]
    true = np.load(true_path(candidates[0], split), mmap_mode="r")
    return pred_arrays, true


def compute_dynamic_chunk(source_pred: np.ndarray, delta_chunk: np.ndarray) -> np.ndarray:
    return np.matmul(
        np.asarray(source_pred, dtype=np.float32),
        np.transpose(np.asarray(delta_chunk, dtype=np.float32), (0, 2, 1)),
    )


def estimate_closed_form_specs(
    *,
    candidates: list[dict],
    alpha: np.ndarray,
    delta_path: Path,
    gamma: np.ndarray,
    target_masks: dict[str, np.ndarray],
    dynamic_source: str,
    chunk_size: int,
    max_samples: int,
    progress_every: int,
    eta_max: float,
) -> list[dict]:
    if eta_max < 0:
        raise ValueError(f"eta_max must be non-negative, got {eta_max}")
    pred_arrays, true = open_prediction_arrays(candidates, "val")
    baseline_idx, static_idx = group_indices(candidates)
    delta = np.load(delta_path, mmap_mode="r")
    expected_shape = true.shape
    if delta.shape[0] < expected_shape[0] or delta.shape[1] != expected_shape[2] or delta.shape[2] != expected_shape[2]:
        raise RuntimeError(f"Unexpected delta shape for val: {delta.shape}, true={expected_shape}")
    if alpha.size != expected_shape[2]:
        raise RuntimeError(f"Alpha length mismatch: {alpha.size} vs n_vars={expected_shape[2]}")

    n_samples = expected_shape[0] if max_samples <= 0 else min(int(max_samples), expected_shape[0])
    if len(gamma) < n_samples:
        raise RuntimeError(f"Gamma length mismatch for val: gamma={len(gamma)} required={n_samples}")
    gamma = np.asarray(gamma[:n_samples], dtype=np.float32)
    alpha_view = alpha.reshape(1, 1, -1)
    sums = {name: {"num": 0.0, "den": 0.0} for name in target_masks}

    started = pd.Timestamp.now()
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        baseline_mean = group_mean_chunk(pred_arrays, baseline_idx, start, end)
        static_mean = group_mean_chunk(pred_arrays, static_idx, start, end)
        anchor = baseline_mean + alpha_view * (static_mean - baseline_mean)
        err_anchor = np.asarray(true[start:end], dtype=np.float32) - anchor
        if dynamic_source == "static_mean":
            source_pred = static_mean
        else:
            source_pred = np.asarray(pred_arrays[static_idx[0]][start:end], dtype=np.float32)
        dynamic = compute_dynamic_chunk(source_pred, delta[start:end])
        gamma_chunk = gamma[start:end].reshape(-1, 1, 1)

        err64 = err_anchor.astype(np.float64, copy=False)
        for mask_name, mask in target_masks.items():
            z = (gamma_chunk * dynamic * mask.reshape(1, 1, -1)).astype(np.float64, copy=False)
            sums[mask_name]["num"] += float((err64 * z).sum())
            sums[mask_name]["den"] += float(np.square(z).sum())

        if progress_every > 0 and (end % progress_every == 0 or end == n_samples):
            elapsed = (pd.Timestamp.now() - started).total_seconds()
            print(f"[val:eta_closed_form] {end}/{n_samples} elapsed={elapsed:.1f}s", flush=True)

    specs = [anchor_spec()]
    for mask_name, values in sums.items():
        num = values["num"]
        den = values["den"]
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
        specs.append(
            {
                "ensemble": f"stage3_closed_form_{mask_name}",
                "eta_mode": "closed_form",
                "eta_mult": float(eta),
                "eta_raw": float(eta_raw),
                "eta_num": float(num),
                "eta_den": float(den),
                "eta_clip_reason": reason,
                "target_mask": mask_name,
            }
        )
    return specs


def fold_ids(n_samples: int, n_folds: int) -> np.ndarray:
    ids = np.empty(n_samples, dtype=np.int64)
    for fold, idx in enumerate(np.array_split(np.arange(n_samples), n_folds), start=1):
        ids[idx] = fold
    return ids


def evaluate_specs(
    *,
    split: str,
    candidates: list[dict],
    alpha: np.ndarray,
    delta_path: Path,
    gamma: np.ndarray,
    specs: list[dict],
    target_masks: dict[str, np.ndarray],
    dynamic_source: str,
    chunk_size: int,
    validation_folds: int,
    max_samples: int,
    progress_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_arrays, true = open_prediction_arrays(candidates, split)
    baseline_idx, static_idx = group_indices(candidates)
    delta = np.load(delta_path, mmap_mode="r")
    expected_shape = true.shape
    if delta.shape[0] < expected_shape[0] or delta.shape[1] != expected_shape[2] or delta.shape[2] != expected_shape[2]:
        raise RuntimeError(f"Unexpected delta shape for {split}: {delta.shape}, true={expected_shape}")
    if alpha.size != expected_shape[2]:
        raise RuntimeError(f"Alpha length mismatch: {alpha.size} vs n_vars={expected_shape[2]}")

    n_samples = expected_shape[0] if max_samples <= 0 else min(int(max_samples), expected_shape[0])
    if len(gamma) < n_samples:
        raise RuntimeError(f"Gamma length mismatch for {split}: gamma={len(gamma)} required={n_samples}")
    n_count = n_samples * expected_shape[1] * expected_shape[2]
    gamma = np.asarray(gamma[:n_samples], dtype=np.float32)
    fold_index = fold_ids(n_samples, validation_folds)
    alpha_view = alpha.reshape(1, 1, -1)

    totals = {
        spec["ensemble"]: {"sse": 0.0, "sae": 0.0, "count": n_count}
        for spec in specs
    }
    fold_totals = {
        (spec["ensemble"], fold): {"sse": 0.0, "sae": 0.0, "count": int(np.sum(fold_index == fold)) * expected_shape[1] * expected_shape[2]}
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
        if dynamic_source == "static_mean":
            source_pred = static_mean
        else:
            source_pred = np.asarray(pred_arrays[static_idx[0]][start:end], dtype=np.float32)
        dynamic = compute_dynamic_chunk(source_pred, delta[start:end])
        gamma_chunk = gamma[start:end].reshape(-1, 1, 1)
        fold_chunk = fold_index[start:end]

        for spec in specs:
            eta = float(spec["eta_mult"])
            if abs(eta) <= 1e-12:
                err = err_anchor
            else:
                mask = target_masks[spec["target_mask"]].reshape(1, 1, -1)
                err = err_anchor - (eta * gamma_chunk) * dynamic * mask
            sse, sae = mse_mae_sums(err)
            totals[spec["ensemble"]]["sse"] += sse
            totals[spec["ensemble"]]["sae"] += sae
            for fold in range(1, validation_folds + 1):
                local = np.where(fold_chunk == fold)[0]
                if local.size == 0:
                    continue
                fsse, fsae = mse_mae_sums(err[local])
                fold_totals[(spec["ensemble"], fold)]["sse"] += fsse
                fold_totals[(spec["ensemble"], fold)]["sae"] += fsae

        if progress_every > 0 and (end % progress_every == 0 or end == n_samples):
            elapsed = (pd.Timestamp.now() - started).total_seconds()
            print(f"[{split}] {end}/{n_samples} elapsed={elapsed:.1f}s", flush=True)

    rows = []
    spec_by_name = {spec["ensemble"]: spec for spec in specs}
    anchor_mse = totals["stage2_anchor"]["sse"] / totals["stage2_anchor"]["count"]
    anchor_mae = totals["stage2_anchor"]["sae"] / totals["stage2_anchor"]["count"]
    for name, total in totals.items():
        spec = spec_by_name[name]
        mse = total["sse"] / total["count"]
        mae = total["sae"] / total["count"]
        rows.append(
            {
                "split": split,
                "ensemble": name,
                "eta_mode": spec.get("eta_mode", "grid"),
                "eta_mult": float(spec["eta_mult"]),
                "eta_raw": float(spec.get("eta_raw", spec["eta_mult"])),
                "eta_clip_reason": spec.get("eta_clip_reason", ""),
                "target_mask": spec["target_mask"],
                "target_count": int(target_masks[spec["target_mask"]].sum()),
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
        spec = spec_by_name[name]
        if total["count"] <= 0:
            continue
        anchor_total = fold_totals[("stage2_anchor", fold)]
        anchor_fold_mse = anchor_total["sse"] / anchor_total["count"]
        anchor_fold_mae = anchor_total["sae"] / anchor_total["count"]
        mse = total["sse"] / total["count"]
        mae = total["sae"] / total["count"]
        fold_rows.append(
            {
                "split": split,
                "fold": fold,
                "ensemble": name,
                "eta_mode": spec.get("eta_mode", "grid"),
                "eta_mult": float(spec["eta_mult"]),
                "eta_raw": float(spec.get("eta_raw", spec["eta_mult"])),
                "eta_clip_reason": spec.get("eta_clip_reason", ""),
                "target_mask": spec["target_mask"],
                "target_count": int(target_masks[spec["target_mask"]].sum()),
                "dynamic_source": dynamic_source,
                "mse": mse,
                "mae": mae,
                "mse_gain_vs_stage2_anchor_pct": pct_gain(anchor_fold_mse, mse),
                "mae_gain_vs_stage2_anchor_pct": pct_gain(anchor_fold_mae, mae),
                "n_samples": int(np.sum(fold_index == fold)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


def evaluate_selected_with_sample_stats(
    *,
    split: str,
    candidates: list[dict],
    alpha: np.ndarray,
    delta_path: Path,
    gamma: np.ndarray,
    spec: dict,
    target_masks: dict[str, np.ndarray],
    dynamic_source: str,
    chunk_size: int,
    max_samples: int,
    progress_every: int,
) -> tuple[dict, pd.DataFrame]:
    pred_arrays, true = open_prediction_arrays(candidates, split)
    baseline_idx, static_idx = group_indices(candidates)
    delta = np.load(delta_path, mmap_mode="r")
    n_samples = true.shape[0] if max_samples <= 0 else min(int(max_samples), true.shape[0])
    if len(gamma) < n_samples:
        raise RuntimeError(f"Gamma length mismatch for {split}: gamma={len(gamma)} required={n_samples}")
    gamma = np.asarray(gamma[:n_samples], dtype=np.float32)
    alpha_view = alpha.reshape(1, 1, -1)
    eta = float(spec["eta_mult"])
    mask = target_masks[spec["target_mask"]].reshape(1, 1, -1).astype(np.float32)
    count_per_sample = true.shape[1] * true.shape[2]

    sample_rows = []
    total_sse = 0.0
    total_sae = 0.0
    anchor_sse_total = 0.0
    anchor_sae_total = 0.0
    started = pd.Timestamp.now()
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        baseline_mean = group_mean_chunk(pred_arrays, baseline_idx, start, end)
        static_mean = group_mean_chunk(pred_arrays, static_idx, start, end)
        anchor = baseline_mean + alpha_view * (static_mean - baseline_mean)
        err_anchor = np.asarray(true[start:end], dtype=np.float32) - anchor
        if dynamic_source == "static_mean":
            source_pred = static_mean
        else:
            source_pred = np.asarray(pred_arrays[static_idx[0]][start:end], dtype=np.float32)
        dyn = compute_dynamic_chunk(source_pred, delta[start:end]) * mask
        err = err_anchor - (eta * gamma[start:end].reshape(-1, 1, 1)) * dyn

        sse, sae = mse_mae_sums(err)
        total_sse += sse
        total_sae += sae
        anchor_sse, anchor_sae = mse_mae_sums(err_anchor)
        anchor_sse_total += anchor_sse
        anchor_sae_total += anchor_sae

        err_anchor64 = err_anchor.astype(np.float64, copy=False)
        dyn64 = dyn.astype(np.float64, copy=False)
        sample_anchor_sse = np.square(err_anchor64).sum(axis=(1, 2))
        sample_anchor_sae = np.abs(err_anchor64).sum(axis=(1, 2))
        sample_err_dot_dyn = (err_anchor64 * dyn64).sum(axis=(1, 2))
        sample_dyn_sq = np.square(dyn64).sum(axis=(1, 2))
        for local, sample_id in enumerate(range(start, end)):
            sample_rows.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "gamma": float(gamma[sample_id]),
                    "eta_mode": spec.get("eta_mode", "grid"),
                    "eta_mult": eta,
                    "eta_raw": float(spec.get("eta_raw", eta)),
                    "eta_clip_reason": spec.get("eta_clip_reason", ""),
                    "target_mask": spec["target_mask"],
                    "target_count": int(target_masks[spec["target_mask"]].sum()),
                    "anchor_sse": float(sample_anchor_sse[local]),
                    "anchor_sae": float(sample_anchor_sae[local]),
                    "err_dot_dyn": float(sample_err_dot_dyn[local]),
                    "dyn_sq": float(sample_dyn_sq[local]),
                    "count": count_per_sample,
                }
            )
        if progress_every > 0 and (end % progress_every == 0 or end == n_samples):
            elapsed = (pd.Timestamp.now() - started).total_seconds()
            print(f"[{split}:selected] {end}/{n_samples} elapsed={elapsed:.1f}s", flush=True)

    count = n_samples * count_per_sample
    summary = {
        "split": split,
        "ensemble": spec["ensemble"],
        "eta_mode": spec.get("eta_mode", "grid"),
        "eta_mult": eta,
        "eta_raw": float(spec.get("eta_raw", eta)),
        "eta_clip_reason": spec.get("eta_clip_reason", ""),
        "target_mask": spec["target_mask"],
        "target_count": int(target_masks[spec["target_mask"]].sum()),
        "dynamic_source": dynamic_source,
        "anchor_mse": anchor_sse_total / count,
        "anchor_mae": anchor_sae_total / count,
        "mse": total_sse / count,
        "mae": total_sae / count,
        "mse_gain_vs_stage2_anchor_pct": pct_gain(anchor_sse_total / count, total_sse / count),
        "mae_gain_vs_stage2_anchor_pct": pct_gain(anchor_sae_total / count, total_sae / count),
        "n_samples": n_samples,
    }
    return summary, pd.DataFrame(sample_rows)


def specs_to_frame(specs: list[dict], target_masks: dict[str, np.ndarray], dynamic_source: str) -> pd.DataFrame:
    rows = []
    for spec in specs:
        rows.append(
            {
                "ensemble": spec["ensemble"],
                "eta_mode": spec.get("eta_mode", "grid"),
                "eta_mult": float(spec["eta_mult"]),
                "eta_raw": float(spec.get("eta_raw", spec["eta_mult"])),
                "eta_num": float(spec.get("eta_num", np.nan)),
                "eta_den": float(spec.get("eta_den", np.nan)),
                "eta_clip_reason": spec.get("eta_clip_reason", ""),
                "target_mask": spec["target_mask"],
                "target_count": int(target_masks[spec["target_mask"]].sum()),
                "dynamic_source": dynamic_source,
            }
        )
    return pd.DataFrame(rows)


def mse_mae_sums(err: np.ndarray) -> tuple[float, float]:
    err64 = np.asarray(err, dtype=np.float64)
    return float(np.square(err64).sum()), float(np.abs(err64).sum())


def pct_gain(before: float, after: float) -> float:
    if abs(before) < 1e-12:
        return 0.0
    return 100.0 * (before - after) / before


def shuffled_gamma_summary(sample_stats: pd.DataFrame, shuffle_count: int, seed: int) -> dict:
    gamma = sample_stats["gamma"].to_numpy(dtype=np.float64)
    eta = float(sample_stats["eta_mult"].iloc[0])
    anchor_sse = sample_stats["anchor_sse"].to_numpy(dtype=np.float64)
    err_dot_dyn = sample_stats["err_dot_dyn"].to_numpy(dtype=np.float64)
    dyn_sq = sample_stats["dyn_sq"].to_numpy(dtype=np.float64)
    count = float(sample_stats["count"].sum())
    observed_sse = float((anchor_sse - 2.0 * eta * gamma * err_dot_dyn + (eta * gamma) ** 2 * dyn_sq).sum())
    observed_mse = observed_sse / count
    rng = np.random.default_rng(seed)
    values = np.empty(shuffle_count, dtype=np.float64)
    for i in range(shuffle_count):
        shuffled = rng.permutation(gamma)
        sse = float((anchor_sse - 2.0 * eta * shuffled * err_dot_dyn + (eta * shuffled) ** 2 * dyn_sq).sum())
        values[i] = sse / count
    return {
        "shuffle_count": int(shuffle_count),
        "shuffle_seed": int(seed),
        "observed_mse": observed_mse,
        "shuffle_mse_mean": float(values.mean()),
        "shuffle_mse_median": float(np.median(values)),
        "shuffle_mse_q05": float(np.quantile(values, 0.05)),
        "shuffle_mse_q95": float(np.quantile(values, 0.95)),
        "observed_mse_gain_vs_shuffle_median_pct": pct_gain(float(np.median(values)), observed_mse),
        "observed_rank_fraction_lower_is_better": float(np.mean(values < observed_mse)),
    }


def main() -> None:
    args = parse_args()
    profile = dict(PROFILES[args.profile])
    prefix = f"{args.profile}_{args.tag}"
    stage2_prefix = args.stage2_prefix or f"{args.profile}_adaptive_alpha"
    closed_loop_prefix = args.closed_loop_prefix or args.profile
    args.out_dir.mkdir(parents=True, exist_ok=True)

    alpha = load_stage2_alpha(args.stage2_dir, stage2_prefix)
    target_masks = build_target_masks(args.target_masks.split(","), alpha)
    lambda_cfg, schedule = load_closed_loop_config(args.closed_loop_dir, closed_loop_prefix)
    raw_lambda_splits = compute_selected_lambda_splits(
        profile,
        lambda_cfg=lambda_cfg,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
    )
    lambda_splits = transform_lambda_splits(raw_lambda_splits, lambda_cfg.get("lambda_transform", "raw"))
    gamma_splits = {
        split: gamma_from_schedule(values, schedule).astype(np.float32)
        for split, values in lambda_splits.items()
    }
    candidates = load_candidates(profile)
    interface_dir = Path(profile["interface_dir"])
    if args.eta_mode == "grid":
        specs = candidate_specs(parse_float_list(args.eta_mults), target_masks)
    else:
        specs = estimate_closed_form_specs(
            candidates=candidates,
            alpha=alpha,
            delta_path=interface_dir / "deltaA_val.npy",
            gamma=gamma_splits["val"],
            target_masks=target_masks,
            dynamic_source=args.dynamic_source,
            chunk_size=args.chunk_size,
            max_samples=args.max_samples,
            progress_every=args.progress_every,
            eta_max=args.eta_max,
        )
    specs_to_frame(specs, target_masks, args.dynamic_source).to_csv(args.out_dir / f"{prefix}_eta_candidates.csv", index=False)

    manifest = {
        "profile": args.profile,
        "tag": args.tag,
        "stage2_dir": str(args.stage2_dir),
        "closed_loop_dir": str(args.closed_loop_dir),
        "interface_dir": str(interface_dir),
        "dynamic_source": args.dynamic_source,
        "eta_mode": args.eta_mode,
        "eta_mults": parse_float_list(args.eta_mults),
        "eta_max": args.eta_max,
        "target_masks": {name: int(mask.sum()) for name, mask in target_masks.items()},
        "candidate_count": len(candidates),
        "ensemble_candidate_count": len(specs),
        "lambda_cfg": lambda_cfg,
        "schedule": schedule,
        "chunk_size": args.chunk_size,
        "max_samples": args.max_samples,
        "shuffle_count": args.shuffle_count,
        "dynamic_source_note": (
            "static_p0 matches the existing posthoc closed-loop dynamic-cache convention; "
            "static_mean is an audit option for using the mean static predictor as static_ref."
        ),
    }
    (args.out_dir / f"{prefix}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    val_grid, val_folds = evaluate_specs(
        split="val",
        candidates=candidates,
        alpha=alpha,
        delta_path=interface_dir / "deltaA_val.npy",
        gamma=gamma_splits["val"],
        specs=specs,
        target_masks=target_masks,
        dynamic_source=args.dynamic_source,
        chunk_size=args.chunk_size,
        validation_folds=args.validation_folds,
        max_samples=args.max_samples,
        progress_every=args.progress_every,
    )
    val_grid.to_csv(args.out_dir / f"{prefix}_val_grid.csv", index=False)
    val_folds.to_csv(args.out_dir / f"{prefix}_val_fold_grid.csv", index=False)

    eligible = val_grid[val_grid["mae_gain_vs_stage2_anchor_pct"] >= float(args.select_mae_min_gain)].copy()
    if eligible.empty:
        selected = val_grid[val_grid["ensemble"] == "stage2_anchor"].iloc[0]
        selection_reason = "fallback_stage2_anchor_no_mae_guard_candidate"
    else:
        selected = eligible.sort_values(["mse", "mae"]).iloc[0]
        selection_reason = "best_val_mse_with_mae_guard"
    selected_spec = next(spec for spec in specs if spec["ensemble"] == selected["ensemble"])
    selected_row = {**selected.to_dict(), "selection_reason": selection_reason}
    pd.DataFrame([selected_row]).to_csv(args.out_dir / f"{prefix}_selected_val_summary.csv", index=False)

    val_selected_summary, val_sample_stats = evaluate_selected_with_sample_stats(
        split="val",
        candidates=candidates,
        alpha=alpha,
        delta_path=interface_dir / "deltaA_val.npy",
        gamma=gamma_splits["val"],
        spec=selected_spec,
        target_masks=target_masks,
        dynamic_source=args.dynamic_source,
        chunk_size=args.chunk_size,
        max_samples=args.max_samples,
        progress_every=args.progress_every,
    )
    test_selected_summary, test_sample_stats = evaluate_selected_with_sample_stats(
        split="test",
        candidates=candidates,
        alpha=alpha,
        delta_path=interface_dir / "deltaA_test.npy",
        gamma=gamma_splits["test"],
        spec=selected_spec,
        target_masks=target_masks,
        dynamic_source=args.dynamic_source,
        chunk_size=args.chunk_size,
        max_samples=args.max_samples,
        progress_every=args.progress_every,
    )
    val_selected_summary["selection_reason"] = selection_reason
    test_selected_summary["selection_reason"] = selection_reason
    pd.DataFrame([val_selected_summary]).to_csv(args.out_dir / f"{prefix}_val_selected_recomputed_summary.csv", index=False)
    pd.DataFrame([test_selected_summary]).to_csv(args.out_dir / f"{prefix}_test_selected_summary.csv", index=False)

    shuffle_rows = []
    if args.shuffle_count > 0:
        val_shuffle = shuffled_gamma_summary(val_sample_stats, args.shuffle_count, args.shuffle_seed)
        val_shuffle["split"] = "val"
        test_shuffle = shuffled_gamma_summary(test_sample_stats, args.shuffle_count, args.shuffle_seed + 1)
        test_shuffle["split"] = "test"
        shuffle_rows.extend([val_shuffle, test_shuffle])
    pd.DataFrame(shuffle_rows).to_csv(args.out_dir / f"{prefix}_shuffled_gamma_summary.csv", index=False)

    print(
        "[Selected] "
        f"ensemble={selected_spec['ensemble']} reason={selection_reason} "
        f"val_mse={val_selected_summary['mse']:.6f} val_mae={val_selected_summary['mae']:.6f} "
        f"val_mse_gain={val_selected_summary['mse_gain_vs_stage2_anchor_pct']:.4f}% "
        f"val_mae_gain={val_selected_summary['mae_gain_vs_stage2_anchor_pct']:.4f}%",
        flush=True,
    )
    print(
        "[Test] "
        f"mse={test_selected_summary['mse']:.6f} mae={test_selected_summary['mae']:.6f} "
        f"mse_gain={test_selected_summary['mse_gain_vs_stage2_anchor_pct']:.4f}% "
        f"mae_gain={test_selected_summary['mae_gain_vs_stage2_anchor_pct']:.4f}%",
        flush=True,
    )
    print(f"[Done] outputs written to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

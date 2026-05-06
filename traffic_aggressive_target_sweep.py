from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from posthoc_calibration.diagnostics import transform_lambda_splits
from posthoc_calibration.evaluation import pct_gain
from posthoc_calibration.io_utils import load_result_dirs
from posthoc_calibration.profiles import PROFILES, compute_selected_lambda_splits
from posthoc_calibration.schedules import gamma_from_schedule, parse_float_list


@dataclass
class ValTerms:
    target_union: np.ndarray
    dynamic_union: np.ndarray
    ed_by_target: np.ndarray
    d2_by_target: np.ndarray
    static_sse: float
    static_sae: float
    total_count: int
    static_dirs: list[Path]
    pred_file: str
    true_file: str
    pred_len: int
    n_vars: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traffic-specific aggressive target-masked post-hoc calibration sweep."
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="traffic96_static")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument(
        "--lambda-selected-csv",
        default=(
            r"C:\Users\cyl\Desktop\data\deltaA_signal_audit"
            r"\traffic96_closed_loop_log_tail_quality_guard"
            r"\traffic96_static_log_tail_quality_guard_closed_loop_lambda_selected.csv"
        ),
    )
    parser.add_argument("--lambda-transform", choices=["rank", "raw"], default="rank")
    parser.add_argument("--target-sources", default="deltaA_energy,deltaA_energy_1hop")
    parser.add_argument("--target-fracs", default="0.01,0.02,0.05")
    parser.add_argument("--max-target-fracs", default="0.05,0.10")
    parser.add_argument("--active-ratios", default="0.01,0.02,0.05,0.10")
    parser.add_argument("--gamma-mins", default="0.01")
    parser.add_argument("--gamma-maxs", default="0.06,0.10,0.15,0.20")
    parser.add_argument("--risk-multipliers", default="1.0,1.5,2.0")
    parser.add_argument("--gamma-cap", type=float, default=0.60)
    parser.add_argument("--active-eps", type=float, default=1e-6)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--energy-chunk-size", type=int, default=8)
    parser.add_argument("--exact-top-n", type=int, default=40)
    parser.add_argument("--min-mae-gain-pct", type=float, default=0.0)
    parser.add_argument(
        "--out-dir",
        default=r"C:\Users\cyl\Desktop\data\deltaA_signal_audit\traffic96_aggressive_target_attack",
    )
    parser.add_argument("--tag", default="aggressive_target")
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--val-only", action="store_true")
    parser.add_argument("--oracle-split", choices=["", "val", "test"], default="")
    parser.add_argument("--energy-split", choices=["val", "test"], default="val")
    return parser.parse_args()


def read_lambda_cfg(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    row = pd.read_csv(path).iloc[0].to_dict()
    return {
        "mode": str(row["mode"]),
        "window": int(row["window"]),
        "k": int(row["k"]),
        "lambda_scale": str(row.get("lambda_scale", "legacy_clipped")),
        "tail_target_width": _finite_float(row.get("tail_target_width"), 0.10),
        "tail_alpha_min": _finite_float(row.get("tail_alpha_min"), 0.02),
        "tail_alpha_max": _finite_float(row.get("tail_alpha_max"), 0.20),
        "source_file": str(path),
    }


def _finite_float(value, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def compute_target_energy(delta_path: Path, chunk_size: int, exclude_diagonal: bool = True) -> np.ndarray:
    delta = np.load(delta_path, mmap_mode="r")
    if delta.ndim != 3 or delta.shape[1] != delta.shape[2]:
        raise ValueError(f"Expected deltaA shape (samples, vars, vars), got {delta.shape}")
    n_samples, n_vars, _ = delta.shape
    energy = np.zeros((n_vars,), dtype=np.float64)
    diag_idx = np.arange(n_vars)
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        block = np.asarray(delta[start:end], dtype=np.float32)
        abs_block = np.abs(block)
        energy += abs_block.sum(axis=(0, 2), dtype=np.float64)
        if exclude_diagonal:
            energy -= abs_block[:, diag_idx, diag_idx].sum(axis=0, dtype=np.float64)
        del block, abs_block
    return energy


def build_target_masks(
    *,
    energy: np.ndarray,
    support_path: Path,
    target_sources: list[str],
    target_fracs: list[float],
    max_target_fracs: list[float],
) -> tuple[list[dict], pd.DataFrame]:
    n_vars = int(energy.size)
    order = np.argsort(-energy)
    support = np.load(support_path)
    support_bool = np.abs(support) > 1e-12
    masks: list[dict] = []
    seen: set[tuple[int, ...]] = set()

    def add_mask(source: str, target_frac: float, max_target_frac: float, targets: np.ndarray, seed_count: int) -> None:
        targets = np.asarray(sorted(set(int(x) for x in targets)), dtype=np.int64)
        key = tuple(targets.tolist())
        if not key or key in seen:
            return
        seen.add(key)
        masks.append(
            {
                "target_source": source,
                "target_frac": float(target_frac),
                "max_target_frac": float(max_target_frac),
                "seed_count": int(seed_count),
                "target_count": int(targets.size),
                "target_frac_actual": float(targets.size / n_vars),
                "targets": targets,
            }
        )

    for source in target_sources:
        source = source.strip()
        if not source:
            continue
        if source == "all":
            add_mask("all", 1.0, 1.0, np.arange(n_vars), n_vars)
            continue
        for frac in target_fracs:
            seed_count = max(1, int(np.ceil(float(frac) * n_vars)))
            seeds = order[:seed_count]
            if source == "deltaA_energy":
                add_mask(source, frac, frac, seeds, seed_count)
            elif source == "deltaA_energy_1hop":
                for max_frac in max_target_fracs:
                    cap = max(seed_count, int(np.ceil(float(max_frac) * n_vars)))
                    expanded = set(int(x) for x in seeds)
                    for seed in seeds:
                        neighbors = np.flatnonzero(support_bool[seed, :] | support_bool[:, seed])
                        expanded.update(int(x) for x in neighbors)
                    expanded_ordered = sorted(expanded, key=lambda idx: float(energy[idx]), reverse=True)
                    add_mask(source, frac, max_frac, np.asarray(expanded_ordered[:cap]), seed_count)
            else:
                raise ValueError(f"Unsupported target source: {source}")

    rows = []
    for idx, mask in enumerate(masks):
        targets = mask["targets"]
        rows.append(
            {
                "target_mask_id": idx,
                "target_source": mask["target_source"],
                "target_frac": mask["target_frac"],
                "max_target_frac": mask["max_target_frac"],
                "seed_count": mask["seed_count"],
                "target_count": mask["target_count"],
                "target_frac_actual": mask["target_frac_actual"],
                "target_energy_sum": float(energy[targets].sum()),
                "top_targets": ",".join(str(int(x)) for x in targets[:20]),
            }
        )
        mask["target_mask_id"] = idx
    return masks, pd.DataFrame(rows)


def build_active_ratio_schedules(active_ratios: list[float], gamma_mins: list[float], gamma_maxs: list[float]) -> list[dict]:
    schedules = []
    for active_ratio in active_ratios:
        q_low = 1.0 - float(active_ratio)
        for gamma_min in gamma_mins:
            for gamma_max in gamma_maxs:
                if gamma_max <= gamma_min + 1e-12:
                    continue
                schedules.append(
                    {
                        "active_ratio_target": float(active_ratio),
                        "q_low": float(q_low),
                        "q_high": 1.0,
                        "q_low_value": float(q_low),
                        "q_high_value": 1.0,
                        "gamma_min": float(gamma_min),
                        "gamma_max": float(gamma_max),
                    }
                )
    if not schedules:
        raise ValueError("No aggressive schedules generated.")
    return schedules


def build_dynamic_for_targets(
    pred0: np.ndarray,
    delta: np.ndarray,
    sample_start: int,
    sample_end: int,
    targets: np.ndarray,
) -> np.ndarray:
    n = sample_end - sample_start
    pred_len = pred0.shape[1]
    out = np.empty((n, pred_len, len(targets)), dtype=np.float32)
    for local, sample_id in enumerate(range(sample_start, sample_end)):
        delta_targets = np.asarray(delta[sample_id, targets, :], dtype=np.float32)
        out[local] = np.asarray(pred0[sample_id], dtype=np.float32) @ delta_targets.T
    return out


def split_file_names(split: str) -> tuple[str, str]:
    if split == "val":
        return "val_pred.npy", "val_true.npy"
    if split == "test":
        return "pred.npy", "true.npy"
    raise ValueError(f"Unsupported split: {split}")


def build_split_terms(
    *,
    profile: dict,
    split: str,
    target_union: np.ndarray,
    pred_len: int,
    chunk_size: int,
    progress_every: int,
) -> ValTerms:
    interface_dir = Path(profile["interface_dir"])
    delta = np.load(interface_dir / f"deltaA_{split}.npy", mmap_mode="r")
    n_samples, n_vars, _ = delta.shape
    pred_file, true_file = split_file_names(split)
    static_dirs = load_result_dirs(str(profile["static_pattern"]), pred_file=pred_file, true_file=true_file)
    pred0 = np.load(static_dirs[0] / pred_file, mmap_mode="r")
    if pred0.shape != (n_samples, pred_len, n_vars):
        raise RuntimeError(f"Unexpected {split} pred shape: {pred0.shape}, expected {(n_samples, pred_len, n_vars)}")

    n_union = int(target_union.size)
    dynamic_union = np.empty((n_samples, pred_len, n_union), dtype=np.float32)
    ed_by_target = np.zeros((n_samples, n_union), dtype=np.float64)
    d2_by_target = np.zeros((n_samples, n_union), dtype=np.float64)
    static_sse = 0.0
    static_sae = 0.0

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        dyn = build_dynamic_for_targets(pred0, delta, start, end, target_union)
        dynamic_union[start:end] = dyn
        d2_by_target[start:end] = len(static_dirs) * np.square(dyn, dtype=np.float32).sum(axis=1, dtype=np.float64)
        for directory in static_dirs:
            pred = np.asarray(np.load(directory / pred_file, mmap_mode="r")[start:end], dtype=np.float32)
            true = np.asarray(np.load(directory / true_file, mmap_mode="r")[start:end], dtype=np.float32)
            err = true - pred
            static_sse += float(np.square(err, dtype=np.float32).sum(dtype=np.float64))
            static_sae += float(np.abs(err).sum(dtype=np.float64))
            err_union = err[:, :, target_union]
            ed_by_target[start:end] += (err_union * dyn).sum(axis=1, dtype=np.float64)
            del pred, true, err, err_union
        if progress_every > 0 and (end % progress_every == 0 or end == n_samples):
            print(f"[Terms:{split}] {end}/{n_samples}", flush=True)
        del dyn

    return ValTerms(
        target_union=target_union,
        dynamic_union=dynamic_union,
        ed_by_target=ed_by_target,
        d2_by_target=d2_by_target,
        static_sse=static_sse,
        static_sae=static_sae,
        total_count=int(len(static_dirs) * n_samples * pred_len * n_vars),
        static_dirs=static_dirs,
        pred_file=pred_file,
        true_file=true_file,
        pred_len=pred_len,
        n_vars=n_vars,
    )


def build_val_terms(
    *,
    profile: dict,
    target_union: np.ndarray,
    pred_len: int,
    chunk_size: int,
    progress_every: int,
) -> ValTerms:
    return build_split_terms(
        profile=profile,
        split="val",
        target_union=target_union,
        pred_len=pred_len,
        chunk_size=chunk_size,
        progress_every=progress_every,
    )


def score_val_grid(
    *,
    terms: ValTerms,
    masks: list[dict],
    schedules: list[dict],
    risk_multipliers: list[float],
    lambda_values: np.ndarray,
    gamma_cap: float,
    active_eps: float,
) -> pd.DataFrame:
    union_positions = {int(target): pos for pos, target in enumerate(terms.target_union)}
    static_mse = terms.static_sse / terms.total_count
    rows = []
    for mask in masks:
        positions = np.asarray([union_positions[int(x)] for x in mask["targets"]], dtype=np.int64)
        ed = terms.ed_by_target[:, positions].sum(axis=1)
        d2 = terms.d2_by_target[:, positions].sum(axis=1)
        for schedule in schedules:
            gamma_base = gamma_from_schedule(lambda_values, schedule).astype(np.float64)
            base_active_ratio = float(np.mean(gamma_base > float(schedule["gamma_min"]) + active_eps))
            for risk_multiplier in risk_multipliers:
                gamma_eff = np.minimum(gamma_base * float(risk_multiplier), float(gamma_cap))
                delta_sse = -2.0 * float(np.sum(gamma_eff * ed)) + float(np.sum(np.square(gamma_eff) * d2))
                post_sse = terms.static_sse + delta_sse
                post_mse = post_sse / terms.total_count
                rows.append(
                    {
                        "target_mask_id": mask["target_mask_id"],
                        "target_source": mask["target_source"],
                        "target_frac": mask["target_frac"],
                        "max_target_frac": mask["max_target_frac"],
                        "target_count": mask["target_count"],
                        "target_frac_actual": mask["target_frac_actual"],
                        **schedule,
                        "risk_multiplier": float(risk_multiplier),
                        "gamma_cap": float(gamma_cap),
                        "gamma_mean_actual": float(gamma_eff.mean()),
                        "gamma_max_actual": float(gamma_eff.max()),
                        "gamma_cap_hit_fraction": float(np.mean(gamma_eff >= float(gamma_cap) - 1e-12)),
                        "active_ratio_actual": base_active_ratio,
                        "static_mse": static_mse,
                        "posthoc_mse": float(post_mse),
                        "mse_gain_pct": pct_gain(static_mse, float(post_mse)),
                        "static_mae": np.nan,
                        "posthoc_mae": np.nan,
                        "mae_gain_pct": np.nan,
                        "exact_evaluated": False,
                    }
                )
    return pd.DataFrame(rows).sort_values(["posthoc_mse", "target_count"]).reset_index(drop=True)


def exact_val_metrics(terms: ValTerms, targets: np.ndarray, gamma: np.ndarray) -> dict:
    positions = np.asarray([int(np.where(terms.target_union == target)[0][0]) for target in targets], dtype=np.int64)
    static_sse = 0.0
    static_sae = 0.0
    post_sse = 0.0
    post_sae = 0.0
    chunk_size = max(1, min(32, terms.dynamic_union.shape[0]))
    for start in range(0, terms.dynamic_union.shape[0], chunk_size):
        end = min(start + chunk_size, terms.dynamic_union.shape[0])
        dyn = terms.dynamic_union[start:end, :, positions]
        g = gamma[start:end].reshape(-1, 1, 1).astype(np.float32)
        for directory in terms.static_dirs:
            pred = np.asarray(np.load(directory / terms.pred_file, mmap_mode="r")[start:end], dtype=np.float32)
            true = np.asarray(np.load(directory / terms.true_file, mmap_mode="r")[start:end], dtype=np.float32)
            err = true - pred
            err_targets = err[:, :, targets]
            post_targets = err_targets - g * dyn
            static_sse += float(np.square(err, dtype=np.float32).sum(dtype=np.float64))
            static_sae += float(np.abs(err).sum(dtype=np.float64))
            post_sse += float(np.square(err, dtype=np.float32).sum(dtype=np.float64))
            post_sae += float(np.abs(err).sum(dtype=np.float64))
            post_sse -= float(np.square(err_targets, dtype=np.float32).sum(dtype=np.float64))
            post_sae -= float(np.abs(err_targets).sum(dtype=np.float64))
            post_sse += float(np.square(post_targets, dtype=np.float32).sum(dtype=np.float64))
            post_sae += float(np.abs(post_targets).sum(dtype=np.float64))
            del pred, true, err, err_targets, post_targets
    static_mse = static_sse / terms.total_count
    post_mse = post_sse / terms.total_count
    static_mae = static_sae / terms.total_count
    post_mae = post_sae / terms.total_count
    return {
        "static_mse": static_mse,
        "posthoc_mse": post_mse,
        "mse_gain_pct": pct_gain(static_mse, post_mse),
        "static_mae": static_mae,
        "posthoc_mae": post_mae,
        "mae_gain_pct": pct_gain(static_mae, post_mae),
    }


def exact_split_metrics(
    *,
    profile: dict,
    split: str,
    targets: np.ndarray,
    schedule: dict,
    lambda_values: np.ndarray,
    risk_multiplier: float,
    gamma_cap: float,
    pred_len: int,
    chunk_size: int,
    progress_every: int,
) -> dict:
    interface_dir = Path(profile["interface_dir"])
    delta = np.load(interface_dir / f"deltaA_{split}.npy", mmap_mode="r")
    n_samples, n_vars, _ = delta.shape
    pred_file = "val_pred.npy" if split == "val" else "pred.npy"
    true_file = "val_true.npy" if split == "val" else "true.npy"
    static_dirs = load_result_dirs(str(profile["static_pattern"]), pred_file=pred_file, true_file=true_file)
    pred0 = np.load(static_dirs[0] / pred_file, mmap_mode="r")
    gamma_base = gamma_from_schedule(lambda_values, schedule).astype(np.float64)
    gamma_eff = np.minimum(gamma_base * float(risk_multiplier), float(gamma_cap)).astype(np.float32)
    static_sse = 0.0
    static_sae = 0.0
    post_sse = 0.0
    post_sae = 0.0
    total_count = int(len(static_dirs) * n_samples * pred_len * n_vars)
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        dyn = build_dynamic_for_targets(pred0, delta, start, end, targets)
        g = gamma_eff[start:end].reshape(-1, 1, 1)
        for directory in static_dirs:
            pred = np.asarray(np.load(directory / pred_file, mmap_mode="r")[start:end], dtype=np.float32)
            true = np.asarray(np.load(directory / true_file, mmap_mode="r")[start:end], dtype=np.float32)
            err = true - pred
            err_targets = err[:, :, targets]
            post_targets = err_targets - g * dyn
            static_sse += float(np.square(err, dtype=np.float32).sum(dtype=np.float64))
            static_sae += float(np.abs(err).sum(dtype=np.float64))
            post_sse += float(np.square(err, dtype=np.float32).sum(dtype=np.float64))
            post_sae += float(np.abs(err).sum(dtype=np.float64))
            post_sse -= float(np.square(err_targets, dtype=np.float32).sum(dtype=np.float64))
            post_sae -= float(np.abs(err_targets).sum(dtype=np.float64))
            post_sse += float(np.square(post_targets, dtype=np.float32).sum(dtype=np.float64))
            post_sae += float(np.abs(post_targets).sum(dtype=np.float64))
            del pred, true, err, err_targets, post_targets
        if progress_every > 0 and (end % progress_every == 0 or end == n_samples):
            print(f"[Exact:{split}] {end}/{n_samples}", flush=True)
        del dyn
    static_mse = static_sse / total_count
    post_mse = post_sse / total_count
    static_mae = static_sae / total_count
    post_mae = post_sae / total_count
    return {
        "split": split,
        "static_mse": static_mse,
        "posthoc_mse": post_mse,
        "mse_gain_pct": pct_gain(static_mse, post_mse),
        "static_mae": static_mae,
        "posthoc_mae": post_mae,
        "mae_gain_pct": pct_gain(static_mae, post_mae),
    }


def schedule_from_row(row: pd.Series) -> dict:
    return {
        "active_ratio_target": float(row["active_ratio_target"]),
        "q_low": float(row["q_low"]),
        "q_high": float(row["q_high"]),
        "q_low_value": float(row["q_low_value"]),
        "q_high_value": float(row["q_high_value"]),
        "gamma_min": float(row["gamma_min"]),
        "gamma_max": float(row["gamma_max"]),
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_oracle_sweep(
    *,
    args: argparse.Namespace,
    profile: dict,
    out_dir: Path,
    run_prefix: str,
    masks: list[dict],
    schedules: list[dict],
    risk_multipliers: list[float],
    lambda_values: np.ndarray,
) -> None:
    oracle_split = str(args.oracle_split)
    target_union = np.asarray(sorted(set(int(x) for mask in masks for x in mask["targets"])), dtype=np.int64)
    print(f"[Oracle] build {oracle_split} terms for target_union={len(target_union)}", flush=True)
    terms = build_split_terms(
        profile=profile,
        split=oracle_split,
        target_union=target_union,
        pred_len=args.pred_len,
        chunk_size=args.chunk_size,
        progress_every=args.progress_every,
    )
    if len(lambda_values) != terms.dynamic_union.shape[0]:
        raise RuntimeError(
            f"{oracle_split} lambda length mismatch: {len(lambda_values)} vs {terms.dynamic_union.shape[0]}"
        )

    print(f"[Oracle] score {oracle_split} grid", flush=True)
    oracle_grid = score_val_grid(
        terms=terms,
        masks=masks,
        schedules=schedules,
        risk_multipliers=risk_multipliers,
        lambda_values=lambda_values,
        gamma_cap=args.gamma_cap,
        active_eps=args.active_eps,
    )
    grid_path = out_dir / f"{run_prefix}_{oracle_split}_oracle_grid.csv"
    oracle_grid.to_csv(grid_path, index=False)

    print(f"[Oracle] exact metrics for top {args.exact_top_n} by MSE gain", flush=True)
    exact_indices = oracle_grid.sort_values(["mse_gain_pct", "target_count"], ascending=[False, True]).head(
        int(args.exact_top_n)
    ).index
    mask_lookup = {int(mask["target_mask_id"]): mask for mask in masks}
    for idx in exact_indices:
        row = oracle_grid.loc[idx]
        schedule = schedule_from_row(row)
        gamma_base = gamma_from_schedule(lambda_values, schedule).astype(np.float64)
        gamma_eff = np.minimum(gamma_base * float(row["risk_multiplier"]), float(args.gamma_cap)).astype(np.float32)
        targets = mask_lookup[int(row["target_mask_id"])]["targets"]
        metrics = exact_val_metrics(terms, targets, gamma_eff)
        for key, value in metrics.items():
            oracle_grid.loc[idx, key] = value
        oracle_grid.loc[idx, "exact_evaluated"] = True

    oracle_grid = oracle_grid.sort_values(["posthoc_mse", "target_count"]).reset_index(drop=True)
    oracle_grid.to_csv(grid_path, index=False)
    exact = oracle_grid[oracle_grid["exact_evaluated"].astype(bool)].copy()
    exact.to_csv(out_dir / f"{run_prefix}_{oracle_split}_oracle_top_exact.csv", index=False)

    best_mse = exact.sort_values(["mse_gain_pct", "mae_gain_pct", "target_count"], ascending=[False, False, True]).head(1)
    best_mse.to_csv(out_dir / f"{run_prefix}_{oracle_split}_oracle_best_mse.csv", index=False)
    mae_guard = exact[exact["mae_gain_pct"] >= float(args.min_mae_gain_pct)].copy()
    if not mae_guard.empty:
        best_guarded = mae_guard.sort_values(
            ["mse_gain_pct", "mae_gain_pct", "target_count"],
            ascending=[False, False, True],
        ).head(1)
        best_guarded.to_csv(out_dir / f"{run_prefix}_{oracle_split}_oracle_best_mse_mae_guard.csv", index=False)
    best_mae = exact.sort_values(["mae_gain_pct", "mse_gain_pct", "target_count"], ascending=[False, False, True]).head(1)
    best_mae.to_csv(out_dir / f"{run_prefix}_{oracle_split}_oracle_best_mae.csv", index=False)

    row = best_mse.iloc[0]
    print(
        "[OracleBestMSE] "
        f"target_source={row['target_source']} targets={int(row['target_count'])} "
        f"active={row['active_ratio_target']} gamma={row['gamma_min']}->{row['gamma_max']} "
        f"risk={row['risk_multiplier']} mse_gain={row['mse_gain_pct']:.4f}% "
        f"mae_gain={row['mae_gain_pct']:.4f}%",
        flush=True,
    )
    if not mae_guard.empty:
        row = best_guarded.iloc[0]
        print(
            "[OracleBestMSEWithMAEGuard] "
            f"target_source={row['target_source']} targets={int(row['target_count'])} "
            f"active={row['active_ratio_target']} gamma={row['gamma_min']}->{row['gamma_max']} "
            f"risk={row['risk_multiplier']} mse_gain={row['mse_gain_pct']:.4f}% "
            f"mae_gain={row['mae_gain_pct']:.4f}%",
            flush=True,
        )
    row = best_mae.iloc[0]
    print(
        "[OracleBestMAE] "
        f"target_source={row['target_source']} targets={int(row['target_count'])} "
        f"active={row['active_ratio_target']} gamma={row['gamma_min']}->{row['gamma_max']} "
        f"risk={row['risk_multiplier']} mse_gain={row['mse_gain_pct']:.4f}% "
        f"mae_gain={row['mae_gain_pct']:.4f}%",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.profile != "traffic96_static":
        print(f"[Warning] This script is intended for Traffic; running profile={args.profile}", flush=True)
    profile = dict(PROFILES[args.profile])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_prefix = f"{args.profile}_{args.tag}"

    lambda_cfg = read_lambda_cfg(Path(args.lambda_selected_csv))
    lambda_cfg["lambda_transform"] = args.lambda_transform
    raw_lambda_splits = compute_selected_lambda_splits(
        profile,
        lambda_cfg=lambda_cfg,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
    )
    lambda_splits = transform_lambda_splits(raw_lambda_splits, args.lambda_transform)

    interface_dir = Path(profile["interface_dir"])
    print(f"[Stage] compute {args.energy_split} DeltaA target energy", flush=True)
    energy = compute_target_energy(
        interface_dir / f"deltaA_{args.energy_split}.npy",
        chunk_size=args.energy_chunk_size,
        exclude_diagonal=True,
    )
    pd.DataFrame(
        {
            "target": np.arange(len(energy), dtype=np.int64),
            "deltaA_energy": energy,
        }
    ).sort_values("deltaA_energy", ascending=False).head(200).to_csv(
        out_dir / f"{run_prefix}_target_energy_top200.csv",
        index=False,
    )

    target_sources = [x.strip() for x in args.target_sources.split(",") if x.strip()]
    target_fracs = parse_float_list(args.target_fracs)
    max_target_fracs = parse_float_list(args.max_target_fracs)
    masks, mask_df = build_target_masks(
        energy=energy,
        support_path=interface_dir / "support.npy",
        target_sources=target_sources,
        target_fracs=target_fracs,
        max_target_fracs=max_target_fracs,
    )
    mask_df.to_csv(out_dir / f"{run_prefix}_target_masks.csv", index=False)
    active_ratios = parse_float_list(args.active_ratios)
    gamma_mins = parse_float_list(args.gamma_mins)
    gamma_maxs = parse_float_list(args.gamma_maxs)
    risk_multipliers = parse_float_list(args.risk_multipliers)
    schedules = build_active_ratio_schedules(active_ratios, gamma_mins, gamma_maxs)
    total_configs = len(masks) * len(schedules) * len(risk_multipliers)
    print(f"[Plan] masks={len(masks)} schedules={len(schedules)} configs={total_configs}", flush=True)

    write_json(
        out_dir / f"{run_prefix}_run_manifest.json",
        {
            "profile": args.profile,
            "lambda_cfg": lambda_cfg,
            "target_sources": target_sources,
            "target_fracs": target_fracs,
            "max_target_fracs": max_target_fracs,
            "active_ratios": active_ratios,
            "gamma_mins": gamma_mins,
            "gamma_maxs": gamma_maxs,
            "risk_multipliers": risk_multipliers,
            "gamma_cap": args.gamma_cap,
            "energy_split": args.energy_split,
            "oracle_split": args.oracle_split,
            "total_configs": total_configs,
        },
    )
    if args.dry_run:
        print(f"[Done] dry-run outputs written to {out_dir}", flush=True)
        return

    if args.oracle_split:
        run_oracle_sweep(
            args=args,
            profile=profile,
            out_dir=out_dir,
            run_prefix=run_prefix,
            masks=masks,
            schedules=schedules,
            risk_multipliers=risk_multipliers,
            lambda_values=lambda_splits[str(args.oracle_split)],
        )
        print(f"[Done] oracle outputs written to {out_dir}", flush=True)
        return

    target_union = np.asarray(sorted(set(int(x) for mask in masks for x in mask["targets"])), dtype=np.int64)
    print(f"[Stage] build validation terms for target_union={len(target_union)}", flush=True)
    terms = build_val_terms(
        profile=profile,
        target_union=target_union,
        pred_len=args.pred_len,
        chunk_size=args.chunk_size,
        progress_every=args.progress_every,
    )
    if len(lambda_splits["val"]) != terms.dynamic_union.shape[0]:
        raise RuntimeError(f"Validation lambda length mismatch: {len(lambda_splits['val'])} vs {terms.dynamic_union.shape[0]}")

    print("[Stage] score validation grid", flush=True)
    val_grid = score_val_grid(
        terms=terms,
        masks=masks,
        schedules=schedules,
        risk_multipliers=risk_multipliers,
        lambda_values=lambda_splits["val"],
        gamma_cap=args.gamma_cap,
        active_eps=args.active_eps,
    )
    val_grid_path = out_dir / f"{run_prefix}_val_grid.csv"
    val_grid.to_csv(val_grid_path, index=False)

    print(f"[Stage] exact validation metrics for top {args.exact_top_n}", flush=True)
    exact_indices = val_grid.sort_values(["mse_gain_pct", "target_count"], ascending=[False, True]).head(
        int(args.exact_top_n)
    ).index
    mask_lookup = {int(mask["target_mask_id"]): mask for mask in masks}
    for idx in exact_indices:
        row = val_grid.loc[idx]
        schedule = schedule_from_row(row)
        gamma_base = gamma_from_schedule(lambda_splits["val"], schedule).astype(np.float64)
        gamma_eff = np.minimum(gamma_base * float(row["risk_multiplier"]), float(args.gamma_cap)).astype(np.float32)
        targets = mask_lookup[int(row["target_mask_id"])]["targets"]
        metrics = exact_val_metrics(terms, targets, gamma_eff)
        for key, value in metrics.items():
            val_grid.loc[idx, key] = value
        val_grid.loc[idx, "exact_evaluated"] = True

    val_grid = val_grid.sort_values(["posthoc_mse", "target_count"]).reset_index(drop=True)
    val_grid.to_csv(val_grid_path, index=False)
    val_grid[val_grid["exact_evaluated"].astype(bool)].to_csv(
        out_dir / f"{run_prefix}_val_top_exact.csv",
        index=False,
    )
    exact_pool = val_grid[val_grid["exact_evaluated"].astype(bool)].copy()
    eligible = exact_pool[exact_pool["mae_gain_pct"] >= float(args.min_mae_gain_pct)].copy()
    if eligible.empty:
        eligible = exact_pool.copy()
        selection_reason = "best_val_mse_gain_no_mae_guard_candidate"
    else:
        selection_reason = "best_val_mse_gain_with_mae_guard"
    selected = eligible.sort_values(["mse_gain_pct", "mae_gain_pct", "target_count"], ascending=[False, False, True]).iloc[0]
    selected_dict = selected.to_dict()
    selected_dict["selection_reason"] = selection_reason
    selected_dict["lambda_mode"] = lambda_cfg["mode"]
    selected_dict["lambda_window"] = lambda_cfg["window"]
    selected_dict["lambda_k"] = lambda_cfg["k"]
    selected_dict["lambda_scale"] = lambda_cfg.get("lambda_scale", "legacy_clipped")
    selected_dict["lambda_transform"] = args.lambda_transform
    pd.DataFrame([selected_dict]).to_csv(out_dir / f"{run_prefix}_selected.csv", index=False)
    print(
        "[Selected] "
        f"reason={selection_reason} target_source={selected_dict['target_source']} "
        f"targets={int(selected_dict['target_count'])} "
        f"active={selected_dict['active_ratio_target']} "
        f"gamma={selected_dict['gamma_min']}->{selected_dict['gamma_max']} "
        f"risk={selected_dict['risk_multiplier']} "
        f"val_mse_gain={selected_dict['mse_gain_pct']:.4f}% "
        f"val_mae_gain={selected_dict['mae_gain_pct']:.4f}%",
        flush=True,
    )

    if args.val_only:
        print(f"[Done] val-only outputs written to {out_dir}", flush=True)
        return

    print("[Stage] exact test evaluation for selected config", flush=True)
    selected_mask = mask_lookup[int(selected["target_mask_id"])]
    selected_schedule = schedule_from_row(selected)
    test_metrics = exact_split_metrics(
        profile=profile,
        split="test",
        targets=selected_mask["targets"],
        schedule=selected_schedule,
        lambda_values=lambda_splits["test"],
        risk_multiplier=float(selected["risk_multiplier"]),
        gamma_cap=float(args.gamma_cap),
        pred_len=args.pred_len,
        chunk_size=args.chunk_size,
        progress_every=args.progress_every,
    )
    test_row = {
        **selected_dict,
        **{f"test_{key}": value for key, value in test_metrics.items() if key != "split"},
    }
    pd.DataFrame([test_row]).to_csv(out_dir / f"{run_prefix}_test_selected_summary.csv", index=False)
    print(
        "[Test] "
        f"mse_gain={test_metrics['mse_gain_pct']:.4f}% "
        f"mae_gain={test_metrics['mae_gain_pct']:.4f}% "
        f"posthoc_mse={test_metrics['posthoc_mse']:.6f} "
        f"posthoc_mae={test_metrics['posthoc_mae']:.6f}",
        flush=True,
    )
    print(f"[Done] outputs written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

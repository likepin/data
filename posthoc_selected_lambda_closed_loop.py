from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from posthoc_calibration.diagnostics import (
    active_ratio_fold_consistency,
    lambda_rank_diagnostics,
    lambda_quality_metrics,
    load_timestamps,
    residual_complexity_alignment_frame,
    saturated_windows,
    static_sample_errors,
    transform_lambda_splits,
    write_residual_complexity_plot,
)
from posthoc_calibration.evaluation import evaluate_selected_schedule, score_validation_grid
from posthoc_calibration.io_utils import try_load_result_dirs
from posthoc_calibration.profiles import (
    PROFILES,
    compute_selected_lambda_splits,
    dynamic_args,
    lambda_candidate_pool,
    lambda_config_from_row,
    selected_lambda_config,
    split_sample_start_rows,
)
from posthoc_calibration.schedules import build_active_ratio_schedules, build_schedules, parse_float_list
from posthoc_calibration.selection import select_schedule
from posthoc_ecl96_deltaA_manual_gate import build_dynamic_cache


def build_static_only_schedule(lambda_values, lambda_cfg: dict, q_reference: float) -> dict:
    q_value = float(pd.Series(lambda_values).quantile(float(q_reference)))
    return {
        "q_low": float(q_reference),
        "q_high": float(q_reference),
        "q_low_value": q_value,
        "q_high_value": q_value,
        "gamma_min": 0.0,
        "gamma_max": 0.0,
        "lambda_mode": lambda_cfg["mode"],
        "lambda_window": lambda_cfg["window"],
        "lambda_k": lambda_cfg["k"],
    }


def write_lambda_diagnostics(
    *,
    profile: dict,
    out_dir: Path,
    run_prefix: str,
    raw_lambda_splits: dict[str, object],
    lambda_splits: dict[str, object],
    active_ratios: list[float],
    seq_len: int,
    pred_len: int,
    train_ratio: float,
) -> None:
    rank_diag = lambda_rank_diagnostics(
        raw_splits=raw_lambda_splits,
        transformed_splits=lambda_splits,
        active_ratios=active_ratios,
    )
    rank_diag.to_csv(out_dir / f"{run_prefix}_lambda_rank_diagnostics.csv", index=False)

    timestamps = load_timestamps(
        data_csv=Path(profile["data_csv"]),
        date_col=profile.get("date_col"),
        header_mode=str(profile.get("header_mode", "infer")),
        sep=str(profile.get("sep", ",")),
    )
    window_rows = []
    for split in ("val", "test"):
        starts = split_sample_start_rows(
            profile,
            split=split,
            seq_len=seq_len,
            pred_len=pred_len,
            train_ratio=train_ratio,
        )
        window_rows.append(
            saturated_windows(
                split=split,
                raw_values=raw_lambda_splits[split],
                transformed_values=lambda_splits[split],
                sample_start_rows=starts,
                seq_len=seq_len,
                active_ratios=active_ratios,
                timestamps=timestamps,
            )
        )
    pd.concat(window_rows, ignore_index=True).to_csv(
        out_dir / f"{run_prefix}_lambda_saturated_windows.csv",
        index=False,
    )


def select_lambda_with_quality_guard(
    *,
    profile: dict,
    args: argparse.Namespace,
    out_dir: Path,
    run_prefix: str,
    active_ratios: list[float],
    val_static_mse: np.ndarray,
    val_static_mae: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    candidates = lambda_candidate_pool(profile, max_candidates=args.quality_max_candidates)
    rows = []
    selected_cfg = None
    selected_raw_splits = None
    best_quality_score = -np.inf
    diag_ratios = active_ratios or parse_float_list(args.quality_active_ratios)
    for candidate_idx, row in candidates.iterrows():
        cfg = lambda_config_from_row(row.to_dict(), source_file=row.get("_source_file", ""))
        raw_splits = compute_selected_lambda_splits(
            profile,
            lambda_cfg=cfg,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            train_ratio=args.train_ratio,
        )
        transformed = transform_lambda_splits(raw_splits, args.lambda_transform)
        metrics = lambda_quality_metrics(
            raw_values=raw_splits["val"],
            transformed_values=transformed["val"],
            static_mse=val_static_mse,
            static_mae=val_static_mae,
            active_ratios=diag_ratios,
        )
        fold_spearman_min = _float_value(row.get("fold_spearman_min"), default=-1.0)
        fold_spearman_std = _float_value(row.get("fold_spearman_std"), default=0.0)
        positive_spearman_fraction = _float_value(row.get("positive_spearman_fraction"), default=0.0)
        selection_score = _float_value(row.get("val_selection_score"), default=0.0)
        passes_quality = (
            metrics["lambda_top_tie_rate"] <= args.quality_max_top_tie_rate
            and metrics["lambda_q99_q80_spread"] >= args.quality_min_q_spread
            and metrics["lambda_raw_iqr"] >= args.quality_min_iqr
            and fold_spearman_min >= args.quality_min_fold_spearman
            and positive_spearman_fraction >= args.quality_min_positive_spearman_fraction
        )
        quality_score = (
            0.35 * metrics["residual_spearman_mse"]
            + 0.20 * selection_score
            + 0.15 * np.clip(metrics["best_active_mse_lift_pct"] / 20.0, -1.0, 1.0)
            + 0.10 * positive_spearman_fraction
            + 0.10 * np.clip(metrics["lambda_raw_iqr"], 0.0, 1.0)
            - 0.20 * metrics["lambda_top_tie_rate"]
            - 0.10 * fold_spearman_std
        )
        out_row = {
            "candidate_idx": int(candidate_idx),
            "mode": cfg["mode"],
            "window": cfg["window"],
            "k": cfg["k"],
            "stable_candidate": cfg["stable_candidate"],
            "stability_score": cfg["stability_score"],
            "fold_spearman_mean": cfg["fold_spearman_mean"],
            "fold_spearman_min": fold_spearman_min,
            "fold_spearman_std": fold_spearman_std,
            "positive_spearman_fraction": positive_spearman_fraction,
            "val_selection_score": selection_score,
            **metrics,
            "passes_quality_guard": bool(passes_quality),
            "quality_score": float(quality_score),
            "selected_by_quality_guard": False,
        }
        rows.append(out_row)
        if passes_quality and quality_score > best_quality_score:
            best_quality_score = float(quality_score)
            selected_cfg = cfg
            selected_raw_splits = raw_splits

    quality_df = pd.DataFrame(rows)
    if selected_cfg is not None:
        mask = (
            (quality_df["mode"] == selected_cfg["mode"])
            & (quality_df["window"] == selected_cfg["window"])
            & (quality_df["k"] == selected_cfg["k"])
        )
        quality_df.loc[mask, "selected_by_quality_guard"] = True
        selected_cfg["quality_guard_reason"] = "passed_lambda_quality_guard"
        selected_cfg["quality_score"] = float(quality_df.loc[mask, "quality_score"].iloc[0])
    else:
        selected_cfg = selected_lambda_config(profile)
        selected_cfg["quality_guard_reason"] = "fallback_default_no_quality_candidate"
        selected_raw_splits = compute_selected_lambda_splits(
            profile,
            lambda_cfg=selected_cfg,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            train_ratio=args.train_ratio,
        )
    quality_df = quality_df.sort_values(
        ["selected_by_quality_guard", "passes_quality_guard", "quality_score"],
        ascending=[False, False, False],
    )
    quality_df.to_csv(out_dir / f"{run_prefix}_lambda_quality_candidates.csv", index=False)
    return selected_cfg, selected_raw_splits


def write_residual_alignment_outputs(
    *,
    profile: dict,
    out_dir: Path,
    run_prefix: str,
    raw_lambda_splits: dict[str, np.ndarray],
    lambda_splits: dict[str, np.ndarray],
    val_static_mse: np.ndarray,
    val_static_mae: np.ndarray,
    active_ratios: list[float],
    seq_len: int,
    pred_len: int,
    train_ratio: float,
    validation_folds: int,
) -> None:
    timestamps = load_timestamps(
        data_csv=Path(profile["data_csv"]),
        date_col=profile.get("date_col"),
        header_mode=str(profile.get("header_mode", "infer")),
        sep=str(profile.get("sep", ",")),
    )
    starts = split_sample_start_rows(
        profile,
        split="val",
        seq_len=seq_len,
        pred_len=pred_len,
        train_ratio=train_ratio,
    )
    frame = residual_complexity_alignment_frame(
        split="val",
        raw_values=raw_lambda_splits["val"],
        transformed_values=lambda_splits["val"],
        static_mse=val_static_mse,
        static_mae=val_static_mae,
        active_ratios=active_ratios,
        sample_start_rows=starts,
        seq_len=seq_len,
        timestamps=timestamps,
        n_folds=validation_folds,
    )
    frame.to_csv(out_dir / f"{run_prefix}_residual_complexity_alignment.csv", index=False)
    write_residual_complexity_plot(
        frame,
        out_dir / f"{run_prefix}_residual_complexity_alignment.png",
        title=f"{run_prefix}: lambda vs static residual risk",
    )


def _float_value(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def run_profile(args: argparse.Namespace) -> None:
    profile = PROFILES[args.profile]
    out_dir = Path(args.out_dir) if args.out_dir else Path(profile["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_prefix = f"{args.profile}_{args.tag}" if args.tag else args.profile

    active_ratios = parse_float_list(args.active_ratios) if args.active_ratios else []
    if active_ratios and args.lambda_transform != "rank":
        raise ValueError("--active-ratios requires --lambda-transform rank")
    val_static_mse_for_diag = None
    val_static_mae_for_diag = None
    if args.lambda_quality_guard:
        print("[Stage] precompute validation static residuals for lambda quality guard", flush=True)
        quality_static_dirs = try_load_result_dirs(
            str(profile["static_pattern"]),
            pred_file="val_pred.npy",
            true_file="val_true.npy",
        )
        if quality_static_dirs is None:
            raise FileNotFoundError("Missing static validation preds required by --lambda-quality-guard")
        val_static_mse_for_diag, val_static_mae_for_diag = static_sample_errors(
            static_dirs=quality_static_dirs,
            pred_file="val_pred.npy",
            true_file="val_true.npy",
        )
        lambda_cfg, raw_lambda_splits = select_lambda_with_quality_guard(
            profile=profile,
            args=args,
            out_dir=out_dir,
            run_prefix=run_prefix,
            active_ratios=active_ratios,
            val_static_mse=val_static_mse_for_diag,
            val_static_mae=val_static_mae_for_diag,
        )
    else:
        lambda_cfg = selected_lambda_config(profile)
        raw_lambda_splits = compute_selected_lambda_splits(
            profile,
            lambda_cfg=lambda_cfg,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            train_ratio=args.train_ratio,
        )
    lambda_cfg["lambda_transform"] = args.lambda_transform
    pd.DataFrame([lambda_cfg]).to_csv(out_dir / f"{run_prefix}_closed_loop_lambda_selected.csv", index=False)
    print(
        "[LambdaSelected] "
        f"profile={args.profile} mode={lambda_cfg['mode']} "
        f"window={lambda_cfg['window']} k={lambda_cfg['k']} "
        f"stability={lambda_cfg['stability_score']:.6f} "
        f"transform={args.lambda_transform} "
        f"quality={lambda_cfg.get('quality_guard_reason', 'default_selection')}",
        flush=True,
    )

    lambda_splits = transform_lambda_splits(raw_lambda_splits, args.lambda_transform)
    if args.lambda_transform != "raw" or active_ratios:
        write_lambda_diagnostics(
            profile=profile,
            out_dir=out_dir,
            run_prefix=run_prefix,
            raw_lambda_splits=raw_lambda_splits,
            lambda_splits=lambda_splits,
            active_ratios=active_ratios,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            train_ratio=args.train_ratio,
        )
    if args.lambda_quality_guard and val_static_mse_for_diag is not None and val_static_mae_for_diag is not None:
        write_residual_alignment_outputs(
            profile=profile,
            out_dir=out_dir,
            run_prefix=run_prefix,
            raw_lambda_splits=raw_lambda_splits,
            lambda_splits=lambda_splits,
            val_static_mse=val_static_mse_for_diag,
            val_static_mae=val_static_mae_for_diag,
            active_ratios=active_ratios or parse_float_list(args.quality_active_ratios),
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            train_ratio=args.train_ratio,
            validation_folds=args.validation_folds,
        )
    q_lows = parse_float_list(args.q_lows)
    q_highs = parse_float_list(args.q_highs)
    gamma_mins = parse_float_list(args.gamma_mins)
    gamma_maxs = parse_float_list(args.gamma_maxs)
    quality_reject = bool(args.lambda_quality_guard and lambda_cfg.get("quality_guard_reason") == "fallback_default_no_quality_candidate")
    fallback_mode_reason = "lambda_quality_reject" if quality_reject else "no_valid_schedule"
    fallback_selection_reason = (
        "fallback_static_only_lambda_quality_reject"
        if quality_reject
        else "fallback_static_only_no_valid_schedule"
    )
    if quality_reject:
        schedules = []
        no_valid_schedule = True
    else:
        try:
            if active_ratios:
                schedules = build_active_ratio_schedules(
                    active_ratios=active_ratios,
                    gamma_mins=gamma_mins,
                    gamma_maxs=gamma_maxs,
                )
            else:
                schedules = build_schedules(
                    lambda_calib=lambda_splits["val"],
                    q_lows=q_lows,
                    q_highs=q_highs,
                    gamma_mins=gamma_mins,
                    gamma_maxs=gamma_maxs,
                )
            no_valid_schedule = False
        except ValueError:
            schedules = []
            no_valid_schedule = True

    print("[Stage] build validation dynamic correction", flush=True)
    val_dynamic, _legacy_lambda_val, _legacy_schedule, val_static_dirs = build_dynamic_cache(
        dynamic_args(profile, split="val", pred_len=args.pred_len, progress_every=args.progress_every)
    )
    val_baseline_dirs = try_load_result_dirs(str(profile["baseline_pattern"]), pred_file="val_pred.npy", true_file="val_true.npy")
    if val_baseline_dirs is None:
        print("[GuardFallback] missing baseline val_pred/val_true; budget guard disabled for this run", flush=True)
    if len(lambda_splits["val"]) != val_dynamic.shape[0]:
        raise RuntimeError(f"Validation lambda length mismatch: {len(lambda_splits['val'])} vs {val_dynamic.shape[0]}")
    if active_ratios:
        print("[Stage] write active-ratio fold diagnostics", flush=True)
        if val_static_mse_for_diag is None or val_static_mae_for_diag is None:
            val_static_mse, val_static_mae = static_sample_errors(
                static_dirs=val_static_dirs,
                pred_file="val_pred.npy",
                true_file="val_true.npy",
            )
        else:
            val_static_mse, val_static_mae = val_static_mse_for_diag, val_static_mae_for_diag
        fold_summary, fold_detail = active_ratio_fold_consistency(
            lambda_values=lambda_splits["val"],
            mse=val_static_mse,
            mae=val_static_mae,
            active_ratios=active_ratios,
            n_folds=args.validation_folds,
        )
        fold_summary.to_csv(out_dir / f"{run_prefix}_active_ratio_fold_consistency.csv", index=False)
        fold_detail.to_csv(out_dir / f"{run_prefix}_active_ratio_fold_details.csv", index=False)

    if no_valid_schedule:
        print(f"[GuardFallback] {fallback_mode_reason}; forcing static-only bypass", flush=True)
        schedule = build_static_only_schedule(
            lambda_values=lambda_splits["val"],
            lambda_cfg=lambda_cfg,
            q_reference=max(q_highs) if q_highs else 1.0,
        )
        val_summary, val_buckets = evaluate_selected_schedule(
            schedule=schedule,
            dynamic=val_dynamic,
            lambda_values=lambda_splits["val"],
            static_dirs=val_static_dirs,
            pred_file="val_pred.npy",
            true_file="val_true.npy",
            profile_name=args.profile,
            split="val",
            lambda_cfg=lambda_cfg,
            active_eps=args.active_eps,
            active_cutoff=args.active_cutoff,
        )
        val_summary["mode_status"] = "Bypass"
        val_summary["mode_reason"] = fallback_mode_reason
        val_summary["selection_reason"] = fallback_selection_reason
        val_buckets["mode_status"] = "Bypass"
        val_buckets["mode_reason"] = fallback_mode_reason
        val_buckets["selection_reason"] = fallback_selection_reason
        row = val_summary.iloc[0]
        selected_schedule = {
            "profile": args.profile,
            "split": "val",
            "lambda_mode": lambda_cfg["mode"],
            "lambda_window": lambda_cfg["window"],
            "lambda_k": lambda_cfg["k"],
            "lambda_transform": lambda_cfg.get("lambda_transform", "raw"),
            **schedule,
            "gamma_mean": 0.0,
            "gamma_min_actual": 0.0,
            "gamma_max_actual": 0.0,
            "gamma_above_min_fraction": 0.0,
            "active_ratio": 0.0,
            "static_mse": float(row["static_mse"]),
            "posthoc_mse": float(row["posthoc_mse"]),
            "mse_gain_pct": float(row["mse_gain_pct"]),
            "static_mae": float(row["static_mae"]),
            "posthoc_mae": float(row["posthoc_mae"]),
            "mae_gain_pct": float(row["mae_gain_pct"]),
            "baseline_mse": float("nan"),
            "baseline_mae": float("nan"),
            "passes_one_se": False,
            "passes_mae_sigma_guard": False,
            "passes_mae_budget_guard": False,
            "passes_mae_guard": False,
            "passes_selection": False,
            "budget_guard_enabled": val_baseline_dirs is not None,
            "mode_status": "Bypass",
            "mode_reason": fallback_mode_reason,
            "selected": True,
            "selection_reason": fallback_selection_reason,
        }
        val_grid = pd.DataFrame([selected_schedule])
    else:
        val_grid = score_validation_grid(
            schedules=schedules,
            dynamic=val_dynamic,
            lambda_values=lambda_splits["val"],
            static_dirs=val_static_dirs,
            baseline_dirs=val_baseline_dirs,
            pred_file="val_pred.npy",
            true_file="val_true.npy",
            profile_name=args.profile,
            lambda_cfg=lambda_cfg,
            active_eps=args.active_eps,
            progress_stride=args.grid_progress_stride,
        )
        selected_schedule, val_grid = select_schedule(
            val_grid=val_grid,
            guard_c=args.guard_c,
            guard_beta=args.guard_beta,
            active_cutoff=args.active_cutoff,
            active_eps=args.active_eps,
        )
        val_summary, val_buckets = evaluate_selected_schedule(
            schedule=selected_schedule,
            dynamic=val_dynamic,
            lambda_values=lambda_splits["val"],
            static_dirs=val_static_dirs,
            pred_file="val_pred.npy",
            true_file="val_true.npy",
            profile_name=args.profile,
            split="val",
            lambda_cfg=lambda_cfg,
            active_eps=args.active_eps,
            active_cutoff=args.active_cutoff,
        )
    val_grid_path = out_dir / f"{run_prefix}_closed_loop_val_grid.csv"
    val_grid.to_csv(val_grid_path, index=False)
    pd.DataFrame([selected_schedule]).to_csv(out_dir / f"{run_prefix}_closed_loop_schedule_selected.csv", index=False)
    print(
        "[ScheduleSelected] "
        f"reason={selected_schedule['selection_reason']} "
        f"q={selected_schedule['q_low']:.2f}-{selected_schedule['q_high']:.2f} "
        f"gamma={selected_schedule['gamma_min']:.3f}->{selected_schedule['gamma_max']:.3f} "
        f"active_ratio={selected_schedule['active_ratio']:.3f} "
        f"mode={selected_schedule['mode_status']} "
        f"reason={selected_schedule['mode_reason']} "
        f"val_mse={selected_schedule['posthoc_mse']:.6f} "
        f"gain={selected_schedule['mse_gain_pct']:.3f}% "
        f"val_mae={selected_schedule['posthoc_mae']:.6f} "
        f"mae_gain={selected_schedule['mae_gain_pct']:.3f}%",
        flush=True,
    )

    if "selection_reason" not in val_summary.columns:
        val_summary["selection_reason"] = selected_schedule["selection_reason"]
    if "selection_reason" not in val_buckets.columns:
        val_buckets["selection_reason"] = selected_schedule["selection_reason"]
    val_summary.to_csv(out_dir / f"{run_prefix}_closed_loop_val_selected_summary.csv", index=False)
    val_buckets.to_csv(out_dir / f"{run_prefix}_closed_loop_val_selected_buckets.csv", index=False)

    if args.val_only:
        print(f"[Done] val-only outputs written to {out_dir}", flush=True)
        return

    print("[Stage] build test dynamic correction", flush=True)
    test_dynamic, _legacy_lambda_test, _legacy_schedule, test_static_dirs = build_dynamic_cache(
        dynamic_args(profile, split="test", pred_len=args.pred_len, progress_every=args.progress_every)
    )
    if len(lambda_splits["test"]) != test_dynamic.shape[0]:
        raise RuntimeError(f"Test lambda length mismatch: {len(lambda_splits['test'])} vs {test_dynamic.shape[0]}")
    test_summary, test_buckets = evaluate_selected_schedule(
        schedule=selected_schedule,
        dynamic=test_dynamic,
        lambda_values=lambda_splits["test"],
        static_dirs=test_static_dirs,
        pred_file="pred.npy",
        true_file="true.npy",
        profile_name=args.profile,
        split="test",
        lambda_cfg=lambda_cfg,
        active_eps=args.active_eps,
        active_cutoff=args.active_cutoff,
    )
    test_summary["selection_reason"] = selected_schedule["selection_reason"]
    test_buckets["selection_reason"] = selected_schedule["selection_reason"]
    if no_valid_schedule:
        test_summary["mode_status"] = "Bypass"
        test_summary["mode_reason"] = fallback_mode_reason
        test_buckets["mode_status"] = "Bypass"
        test_buckets["mode_reason"] = fallback_mode_reason
    test_summary_path = out_dir / f"{run_prefix}_closed_loop_test_selected_summary.csv"
    test_buckets_path = out_dir / f"{run_prefix}_closed_loop_test_selected_buckets.csv"
    test_summary.to_csv(test_summary_path, index=False)
    test_buckets.to_csv(test_buckets_path, index=False)
    row = test_summary.iloc[0]
    print(
        "[TestSummary] "
        f"static_mse={row['static_mse']:.6f} posthoc_mse={row['posthoc_mse']:.6f} "
        f"mse_gain={row['mse_gain_pct']:.3f}% "
        f"static_mae={row['static_mae']:.6f} posthoc_mae={row['posthoc_mae']:.6f} "
        f"mae_gain={row['mae_gain_pct']:.3f}%",
        flush=True,
    )
    print(f"[Done] outputs written to {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-calibrated post-hoc lambda/DeltaA closed-loop experiment.")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--q-lows", default="0.8,0.9,0.95")
    parser.add_argument("--q-highs", default="0.9,0.95,0.99")
    parser.add_argument("--gamma-mins", default="0,0.01,0.03")
    parser.add_argument("--gamma-maxs", default="0.03,0.04,0.05,0.06")
    parser.add_argument("--guard-c", type=float, default=1.0)
    parser.add_argument("--guard-beta", type=float, default=0.1)
    parser.add_argument("--active-cutoff", type=float, default=0.5)
    parser.add_argument("--active-eps", type=float, default=1e-6)
    parser.add_argument("--grid-progress-stride", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--lambda-transform", choices=["raw", "rank"], default="raw")
    parser.add_argument("--active-ratios", default="")
    parser.add_argument("--validation-folds", type=int, default=4)
    parser.add_argument("--lambda-quality-guard", action="store_true")
    parser.add_argument("--quality-max-candidates", type=int, default=30)
    parser.add_argument("--quality-active-ratios", default="0.01,0.02,0.05,0.10,0.20")
    parser.add_argument("--quality-max-top-tie-rate", type=float, default=0.4)
    parser.add_argument("--quality-min-q-spread", type=float, default=1e-4)
    parser.add_argument("--quality-min-iqr", type=float, default=1e-4)
    parser.add_argument("--quality-min-fold-spearman", type=float, default=-1.0)
    parser.add_argument("--quality-min-positive-spearman-fraction", type=float, default=0.25)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--val-only", action="store_true")
    parser.add_argument("--tag", default="")
    run_profile(parser.parse_args())


if __name__ == "__main__":
    main()

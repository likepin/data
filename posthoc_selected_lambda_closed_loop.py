from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from posthoc_calibration.evaluation import evaluate_selected_schedule, score_validation_grid
from posthoc_calibration.io_utils import try_load_result_dirs
from posthoc_calibration.profiles import PROFILES, compute_selected_lambda_splits, dynamic_args, selected_lambda_config
from posthoc_calibration.schedules import build_schedules, parse_float_list
from posthoc_calibration.selection import select_schedule
from posthoc_ecl96_deltaA_manual_gate import build_dynamic_cache


def run_profile(args: argparse.Namespace) -> None:
    profile = PROFILES[args.profile]
    out_dir = Path(profile["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_prefix = f"{args.profile}_{args.tag}" if args.tag else args.profile

    lambda_cfg = selected_lambda_config(profile)
    pd.DataFrame([lambda_cfg]).to_csv(out_dir / f"{run_prefix}_closed_loop_lambda_selected.csv", index=False)
    print(
        "[LambdaSelected] "
        f"profile={args.profile} mode={lambda_cfg['mode']} "
        f"window={lambda_cfg['window']} k={lambda_cfg['k']} "
        f"stability={lambda_cfg['stability_score']:.6f}",
        flush=True,
    )

    lambda_splits = compute_selected_lambda_splits(
        profile,
        lambda_cfg=lambda_cfg,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
    )
    schedules = build_schedules(
        lambda_calib=lambda_splits["val"],
        q_lows=parse_float_list(args.q_lows),
        q_highs=parse_float_list(args.q_highs),
        gamma_mins=parse_float_list(args.gamma_mins),
        gamma_maxs=parse_float_list(args.gamma_maxs),
    )

    print("[Stage] build validation dynamic correction", flush=True)
    val_dynamic, _legacy_lambda_val, _legacy_schedule, val_static_dirs = build_dynamic_cache(
        dynamic_args(profile, split="val", pred_len=args.pred_len, progress_every=args.progress_every)
    )
    val_baseline_dirs = try_load_result_dirs(str(profile["baseline_pattern"]), pred_file="val_pred.npy", true_file="val_true.npy")
    if val_baseline_dirs is None:
        print("[GuardFallback] missing baseline val_pred/val_true; budget guard disabled for this run", flush=True)
    if len(lambda_splits["val"]) != val_dynamic.shape[0]:
        raise RuntimeError(f"Validation lambda length mismatch: {len(lambda_splits['val'])} vs {val_dynamic.shape[0]}")

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
    val_summary["selection_reason"] = selected_schedule["selection_reason"]
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
    parser.add_argument("--val-only", action="store_true")
    parser.add_argument("--tag", default="")
    run_profile(parser.parse_args())


if __name__ == "__main__":
    main()

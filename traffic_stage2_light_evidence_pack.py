from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

import traffic_adaptive_alpha_evidence_pack as pack


DATA_ROOT = Path(r"C:\Users\cyl\Desktop\data")
STAGE2_DIR = DATA_ROOT / "deltaA_signal_audit" / "traffic96_existing_prediction_ensemble_stage2_light_seed2026"
INTERFACE_DIR = DATA_ROOT / "interfaces" / "Traffic_graph_interface_parcorr"
RUN_LOG_DIR = DATA_ROOT / "run_logs" / "traffic_stage2_light_seed2026_20260507_0139"
PACKAGE_DIR = DATA_ROOT / "mechanism_evidence" / "traffic96_stage2_light_seed2026_20260507"
OUT_DIR = PACKAGE_DIR / "performance" / "adaptive_alpha_ensemble"
PREFIX = "traffic96_static_stage2_light_seed2026"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Traffic Stage2-Light evidence package.")
    parser.add_argument("--adaptive-dir", type=Path, default=STAGE2_DIR)
    parser.add_argument("--interface-dir", type=Path, default=INTERFACE_DIR)
    parser.add_argument("--run-log-dir", type=Path, default=RUN_LOG_DIR)
    parser.add_argument("--package-dir", type=Path, default=PACKAGE_DIR)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--shuffle-count", type=int, default=256)
    parser.add_argument("--shuffle-seed", type=int, default=20260507)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def git_head(cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def copy_logs(run_log_dir: Path, dest_dir: Path) -> list[str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(run_log_dir.glob("*")):
        if not path.is_file():
            continue
        dest = dest_dir / path.name
        shutil.copy2(path, dest)
        copied.append(path.name)
    return copied


def write_readme(
    out_dir: Path,
    selected: dict,
    adaptive_summary: dict,
    shuffle_summary: dict,
    stage15_delta: dict,
) -> None:
    content = f"""# Traffic96 Stage2-Light Adaptive-Alpha Evidence

Generated: 2026-05-07

This package freezes the light Stage2 Traffic performance branch. Stage2-Light adds one paired
seed (`projection_3`, `seed=2026`) to the existing three baseline/staticcausal projections and
reruns validation-selected adaptive-alpha ensembling over 8 candidates.

## Boundary

This is a Traffic prediction-level performance evidence package. It is not a post-hoc dynamic CACI
closed-loop result and should not be used as evidence that dynamic `deltaA` calibration improved
Traffic forecasting.

## Selection

- Selected ensemble: `{selected['ensemble']}`
- Selection reason: `{selected['selection_reason']}`
- Reference best single: `{selected['reference_best_single']}`
- Candidate count: `8` (`baseline_p0..p3`, `static_p0..p3`)
- Test split used only once for final selected evaluation.

## Key Results

- Global closed-form alpha: `{adaptive_summary['alpha_global_clipped']:.6f}`
- Per-variable alpha mean/std: `{adaptive_summary['var_alpha_mean']:.6f} / {adaptive_summary['var_alpha_std']:.6f}`
- Validation MSE/MAE: `{selected['val_mse']:.6f} / {selected['val_mae']:.6f}`
- Test MSE/MAE: `{selected['test_mse']:.6f} / {selected['test_mae']:.6f}`
- Test gain vs `static_p1`: MSE `+{selected['test_mse_gain_vs_best_single_pct']:.4f}%`, MAE `+{selected['test_mae_gain_vs_best_single_pct']:.4f}%`
- Increment vs Stage1.5 selected: MSE `+{stage15_delta['mse_rel_improve_pct']:.4f}%`, MAE `+{stage15_delta['mae_rel_improve_pct']:.4f}%`

## Negative Control

The shuffled-alpha control permutes the same 862 alpha values across targets. It preserves the
alpha distribution but breaks target identity.

- Shuffled median test MSE: `{shuffle_summary['shuffle_test_mse_median']:.6f}`
- Observed test MSE: `{shuffle_summary['observed_test_mse']:.6f}`
- Observed gain vs shuffled median: `+{shuffle_summary['observed_test_mse_gain_vs_shuffle_median_pct']:.4f}%`
- Lower-is-better test rank fraction among shuffles: `{shuffle_summary['observed_test_rank_fraction_lower_is_better']:.4f}`

## Files

- `raw_outputs/`: direct small outputs from `traffic_existing_prediction_ensemble.py --tag stage2_light_seed2026`.
- `training_logs/`: Stage2 train/backfill commands and logs; no `.npy` arrays.
- `tables/{PREFIX}_frozen_table.csv`: frozen Stage2-Light comparison table.
- `tables/{PREFIX}_target_diagnostics.csv`: per-target alpha, gains, and PCMCI graph diagnostics.
- `tables/{PREFIX}_top_alpha_targets.csv`: highest-alpha targets for mechanism inspection.
- `tables/{PREFIX}_alignment_summary.csv`: correlation and negative-control summary rows.
- `tables/{PREFIX}_shuffled_negative_control.csv`: shuffled-alpha MSE diagnostics.
- `figures/{PREFIX}_alpha_distribution.png`: alpha distribution.
- `figures/{PREFIX}_alpha_gain_scatter.png`: alpha vs validation gain.
- `figures/{PREFIX}_alpha_graph_scatter.png`: alpha vs PCMCI parent strength.
"""
    (out_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    # Reuse Stage1.5 evidence helpers, but point their prefix at Stage2 outputs.
    pack.PREFIX = PREFIX
    pack.ADAPTIVE_DIR = args.adaptive_dir

    out_dir = args.package_dir / "performance" / "adaptive_alpha_ensemble"
    raw_dir = out_dir / "raw_outputs"
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    log_dir = out_dir / "training_logs"
    for path in [raw_dir, table_dir, fig_dir, log_dir]:
        path.mkdir(parents=True, exist_ok=True)

    manifest = pack.read_json(args.adaptive_dir / f"{PREFIX}_manifest.json")
    adaptive_summary = pack.read_json(args.adaptive_dir / f"{PREFIX}_adaptive_alpha_summary.json")
    selected = pd.read_csv(args.adaptive_dir / f"{PREFIX}_selected_test_summary.csv").iloc[0].to_dict()
    alpha_df = pd.read_csv(args.adaptive_dir / f"{PREFIX}_variable_alpha.csv")
    alpha_df.attrs["alpha_global"] = adaptive_summary["alpha_global_clipped"]
    alpha = alpha_df["alpha_shrunk"].to_numpy(dtype=float)
    candidates = manifest["candidates"]

    print("[Stage] copy raw Stage2 outputs", flush=True)
    raw_files = pack.copy_raw_outputs(args.adaptive_dir, raw_dir)
    log_files = copy_logs(args.run_log_dir, log_dir)

    frozen_src_dir = args.package_dir / "performance" / "adaptive_alpha_ensemble" / "tables"
    frozen_files = [path.name for path in sorted(frozen_src_dir.glob(f"{PREFIX}_frozen_table.*"))]

    print("[Stage] compute validation/test sufficient statistics", flush=True)
    val_stats = pack.split_sufficient_stats(candidates, alpha, "val", args.chunk_size)
    test_stats = pack.split_sufficient_stats(candidates, alpha, "test", args.chunk_size)

    print("[Stage] build target diagnostics", flush=True)
    graph_df = pack.graph_frame(args.interface_dir)
    diag = pack.target_diagnostics(alpha_df, val_stats, test_stats, graph_df)
    diag_path = table_dir / f"{PREFIX}_target_diagnostics.csv"
    diag.to_csv(diag_path, index=False)

    top_path = table_dir / f"{PREFIX}_top_alpha_targets.csv"
    diag.sort_values(["alpha_shrunk", "val_adaptive_mse_gain_vs_baseline_pct"], ascending=[False, False]).head(
        args.top_k
    ).to_csv(top_path, index=False)

    print("[Stage] run shuffled-alpha negative control", flush=True)
    shuffle_df, shuffle_summary = pack.shuffled_negative_control(
        val_stats=val_stats,
        test_stats=test_stats,
        alpha=alpha,
        shuffle_count=args.shuffle_count,
        seed=args.shuffle_seed,
    )
    shuffle_path = table_dir / f"{PREFIX}_shuffled_negative_control.csv"
    shuffle_df.to_csv(shuffle_path, index=False)
    shuffle_summary_path = table_dir / f"{PREFIX}_shuffled_negative_control_summary.json"
    write_json(shuffle_summary_path, shuffle_summary)

    align_path = table_dir / f"{PREFIX}_alignment_summary.csv"
    pack.alignment_summary(diag, shuffle_summary).to_csv(align_path, index=False)

    print("[Stage] render figures", flush=True)
    figure_files = pack.make_plots(diag, adaptive_summary, fig_dir)

    stage15_table = pd.read_csv(
        DATA_ROOT
        / "mechanism_evidence"
        / "traffic96_mechanism_performance_20260506"
        / "performance"
        / "adaptive_alpha_ensemble"
        / "tables"
        / "traffic96_static_adaptive_alpha_stage15_frozen_table.csv"
    )
    stage15 = stage15_table[stage15_table["label"].astype(str).eq("per-variable shrinkage alpha")].iloc[0]
    stage15_delta = {
        "stage15_test_mse": float(stage15["test_mse"]),
        "stage15_test_mae": float(stage15["test_mae"]),
        "stage2_test_mse": float(selected["test_mse"]),
        "stage2_test_mae": float(selected["test_mae"]),
        "mse_rel_improve_pct": pack.pct_gain(float(stage15["test_mse"]), float(selected["test_mse"])),
        "mae_rel_improve_pct": pack.pct_gain(float(stage15["test_mae"]), float(selected["test_mae"])),
    }

    claim = {
        "status": "stage2_light_performance_branch",
        "claim": "One additional paired seed yields a small positive increment over Stage1.5 adaptive-alpha Traffic performance.",
        "selected_ensemble": str(selected["ensemble"]),
        "selection_reason": str(selected["selection_reason"]),
        "candidate_count": int(manifest["candidate_count"]),
        "alpha_global_closed_form": float(adaptive_summary["alpha_global_clipped"]),
        "alpha_variable_mean": float(adaptive_summary["var_alpha_mean"]),
        "alpha_variable_std": float(adaptive_summary["var_alpha_std"]),
        "val_mse": float(selected["val_mse"]),
        "val_mae": float(selected["val_mae"]),
        "test_mse": float(selected["test_mse"]),
        "test_mae": float(selected["test_mae"]),
        "test_mse_gain_vs_static_p1_pct": float(selected["test_mse_gain_vs_best_single_pct"]),
        "test_mae_gain_vs_static_p1_pct": float(selected["test_mae_gain_vs_best_single_pct"]),
        **stage15_delta,
        "shuffle_count": int(args.shuffle_count),
        "observed_test_mse_gain_vs_shuffle_median_pct": float(
            shuffle_summary["observed_test_mse_gain_vs_shuffle_median_pct"]
        ),
        "observed_test_rank_fraction_lower_is_better": float(
            shuffle_summary["observed_test_rank_fraction_lower_is_better"]
        ),
        "data_repo_head_at_packaging": git_head(DATA_ROOT),
        "itransformer_repo_head_at_packaging": git_head(Path(r"C:\Users\cyl\Desktop\iTransformer-phasec-clean")),
    }
    write_json(args.package_dir / "manifest.json", {
        "package": "traffic96_stage2_light_seed2026_20260507",
        "generated_at": "2026-05-07",
        "source_dirs": {
            "adaptive_outputs": str(args.adaptive_dir),
            "training_logs": str(args.run_log_dir),
            "traffic_interface": str(args.interface_dir),
        },
        "layout": {
            "adaptive_alpha_ensemble": "performance/adaptive_alpha_ensemble",
        },
        "copied_file_groups": {
            "raw_outputs": raw_files,
            "training_logs": log_files,
            "frozen_tables": frozen_files,
            "diagnostic_tables": [
                diag_path.name,
                top_path.name,
                shuffle_path.name,
                shuffle_summary_path.name,
                align_path.name,
            ],
            "figures": figure_files,
        },
        "claims": {
            "stage2_light_adaptive_alpha": claim,
        },
        "large_artifacts_not_copied": [
            "Traffic pred.npy/true.npy/val_pred.npy/val_true.npy arrays",
            "Traffic graph interface deltaA_*.npy arrays",
            "model checkpoints",
        ],
        "reproduction_entrypoints": [
            "traffic_stage2_light_freeze_table.py",
            "traffic_stage2_light_evidence_pack.py",
            "traffic_existing_prediction_ensemble.py",
            "backfill_posthoc_profile_preds.py",
        ],
    })
    write_readme(out_dir, selected, adaptive_summary, shuffle_summary, stage15_delta)

    root_readme = args.package_dir / "README.md"
    root_readme.write_text(
        "# Traffic96 Stage2-Light Seed2026 Evidence Package\n\n"
        "This package freezes the light Stage2 Traffic adaptive-alpha performance branch. "
        "It stores only small CSV/JSON/log/figure artifacts and intentionally excludes large `.npy` arrays.\n\n"
        f"- Selected test MSE/MAE: `{selected['test_mse']:.6f} / {selected['test_mae']:.6f}`\n"
        f"- Gain vs static_p1: MSE `+{selected['test_mse_gain_vs_best_single_pct']:.4f}%`, "
        f"MAE `+{selected['test_mae_gain_vs_best_single_pct']:.4f}%`\n"
        f"- Increment vs Stage1.5 selected: MSE `+{stage15_delta['mse_rel_improve_pct']:.4f}%`, "
        f"MAE `+{stage15_delta['mae_rel_improve_pct']:.4f}%`\n\n"
        "Main subpackage: `performance/adaptive_alpha_ensemble/`.\n",
        encoding="utf-8",
    )

    print(f"[Done] Stage2-Light evidence package: {out_dir}", flush=True)
    print(
        "[Summary] "
        f"test_mse={float(selected['test_mse']):.6f} "
        f"test_mae={float(selected['test_mae']):.6f} "
        f"stage15_mse_gain={stage15_delta['mse_rel_improve_pct']:.4f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()

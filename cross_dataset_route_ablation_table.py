from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from posthoc_calibration.evaluation import pct_gain
from posthoc_calibration.profiles import PROFILES, RESULT_ROOT


OUT_DIR = Path(r"C:\Users\cyl\Desktop\data\mechanism_evidence\cross_dataset_route_ablation_20260507")
ETTH1_ADAPTIVE_TABLE = Path(
    r"C:\Users\cyl\Desktop\data\mechanism_evidence\etth196_adaptive_alpha_20260509\etth196_adaptive_alpha_frozen_table.csv"
)
ETTH1_STAGE3_TABLE = Path(
    r"C:\Users\cyl\Desktop\data\mechanism_evidence\etth196_stage3_lambda_three_source_20260509\etth196_stage3_lambda_three_source_frozen_table.csv"
)
SOLAR_ADAPTIVE_TABLE = Path(
    r"C:\Users\cyl\Desktop\data\mechanism_evidence\solar96_192_adaptive_alpha_20260508\solar_adaptive_alpha_frozen_table.csv"
)
SOLAR_STAGE3_TABLE = Path(
    r"C:\Users\cyl\Desktop\data\mechanism_evidence\solar96_192_stage3_lambda_three_source_20260508\solar_stage3_lambda_three_source_frozen_table.csv"
)


POSTHOC_SUMMARIES = {
    "ETTh1": Path(
        r"C:\Users\cyl\Desktop\data\deltaA_signal_audit\etth196_closed_loop_rank_quality_guard_parcorr_ridgebase_sparse\etth196_static_parcorr_rank_quality_guard_parcorr_closed_loop_test_selected_summary.csv"
    ),
    "Weather": Path(
        r"C:\Users\cyl\Desktop\data\deltaA_signal_audit\weather96_closed_loop\weather96_static_full_guard_v2_closed_loop_test_selected_summary.csv"
    ),
    "ECL": Path(
        r"C:\Users\cyl\Desktop\data\deltaA_signal_audit\ecl96_closed_loop_static\ecl96_static_full_guard_v2_closed_loop_test_selected_summary.csv"
    ),
    "Solar-96": Path(
        r"C:\Users\cyl\Desktop\data\deltaA_signal_audit\solar96_closed_loop\solar96_static_closed_loop_test_selected_summary.csv"
    ),
    "Solar-192": Path(
        r"C:\Users\cyl\Desktop\data\deltaA_signal_audit\solar192_closed_loop_rank_quality_guard\solar192_static_rank_quality_guard_closed_loop_test_selected_summary.csv"
    ),
    "Traffic": Path(
        r"C:\Users\cyl\Desktop\data\deltaA_signal_audit\traffic96_closed_loop_rank_quality_guard\traffic96_static_rank_quality_guard_closed_loop_test_selected_summary.csv"
    ),
}


TRAFFIC_STAGE2_SUMMARY = Path(
    r"C:\Users\cyl\Desktop\data\deltaA_signal_audit\traffic96_existing_prediction_ensemble_stage2_light_seed2026\traffic96_static_stage2_light_seed2026_selected_test_summary.csv"
)
TRAFFIC_STAGE3_SUMMARY = Path(
    r"C:\Users\cyl\Desktop\data\deltaA_signal_audit\traffic96_stage3_lambda_three_source_closed_form_eta2\traffic96_static_stage3_closed_form_eta2_test_selected_summary.csv"
)


DATASET_PROFILES = {
    "ETTh1": "etth196_static_parcorr",
    "Weather": "weather96_static",
    "ECL": "ecl96_static",
    "Solar-96": "solar96_static",
    "Solar-192": "solar192_static",
    "Traffic": "traffic96_static",
}


DATASET_HORIZONS = {
    "ETTh1": 96,
    "Weather": 96,
    "ECL": 96,
    "Solar-96": 96,
    "Solar-192": 192,
    "Traffic": 96,
}


FINAL_HEADLINES = {
    "ETTh1": "Adaptive fusion headline; Stage3 lambda/dynamic add-on is negative, and guarded post-hoc stays Selective but weaker than the fusion route.",
    "Weather": "Static anchor headline; post-hoc dynamic is MSE-positive but MAE-negative.",
    "ECL": "Static anchor headline; strict guarded dynamic branch bypasses.",
    "Solar-96": "Adaptive fusion headline; Stage3 lambda/dynamic is a weak positive add-on.",
    "Solar-192": "Adaptive fusion headline; Stage3 lambda/dynamic falls back to the adaptive anchor.",
    "Traffic": "Adaptive fusion headline; Stage3 lambda/dynamic is a weak add-on.",
}


def route_dirs(pattern: str) -> list[Path]:
    return sorted(Path(path) for path in glob.glob(str(RESULT_ROOT / pattern)))


def metric_frame(pattern: str) -> pd.DataFrame:
    rows = []
    for directory in route_dirs(pattern):
        metrics_path = directory / "metrics.npy"
        if not metrics_path.exists():
            continue
        # iTransformer metrics.npy order is [mae, mse, rmse, mape, mspe].
        raw = np.asarray(np.load(metrics_path)).reshape(-1).tolist()
        if len(raw) >= 5:
            mae, mse, rmse, mape, mspe = raw[:5]
        elif len(raw) == 2:
            mae, mse = raw
            rmse = mape = mspe = np.nan
        else:
            raise ValueError(f"Unexpected metrics.npy length={len(raw)} at {metrics_path}")
        rows.append(
            {
                "result_dir": str(directory),
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "mape": float(mape),
                "mspe": float(mspe),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No metrics.npy found for pattern: {pattern}")
    return pd.DataFrame(rows)


def route_summary(profile_name: str) -> dict:
    profile = PROFILES[profile_name]
    baseline = metric_frame(str(profile["baseline_pattern"]))
    static = metric_frame(str(profile["static_pattern"]))
    return {
        "baseline_projection_count": int(len(baseline)),
        "static_projection_count": int(len(static)),
        "baseline_mse": float(baseline["mse"].mean()),
        "baseline_mae": float(baseline["mae"].mean()),
        "static_anchor_mse": float(static["mse"].mean()),
        "static_anchor_mae": float(static["mae"].mean()),
        "baseline_mse_std": float(baseline["mse"].std(ddof=0)),
        "baseline_mae_std": float(baseline["mae"].std(ddof=0)),
        "static_anchor_mse_std": float(static["mse"].std(ddof=0)),
        "static_anchor_mae_std": float(static["mae"].std(ddof=0)),
        "baseline_dirs": baseline["result_dir"].tolist(),
        "static_dirs": static["result_dir"].tolist(),
    }


def read_one(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty CSV: {path}")
    return df.iloc[0].to_dict()


def read_matching(path: Path, **criteria: object) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty CSV: {path}")
    mask = pd.Series(True, index=df.index)
    for key, value in criteria.items():
        if key not in df.columns:
            raise KeyError(f"Missing column {key!r} in {path}")
        mask &= df[key].astype(str) == str(value)
    matched = df.loc[mask]
    if matched.empty:
        raise ValueError(f"No row in {path} matching {criteria}")
    return matched.iloc[0].to_dict()


def f(row: dict, key: str, default=np.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def s(row: dict, key: str, default="") -> str:
    value = row.get(key, default)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return str(default)
    return str(value)


def mode_label(dataset: str, posthoc: dict) -> str:
    if "mode_status" in posthoc and str(posthoc["mode_status"]):
        return str(posthoc["mode_status"])
    if s(posthoc, "selection_reason") == "fallback_static_only":
        return "Bypass"
    active_ratio = f(posthoc, "active_ratio", default=np.nan)
    mse_gain = f(posthoc, "mse_gain_pct", default=np.nan)
    mae_gain = f(posthoc, "mae_gain_pct", default=np.nan)
    if active_ratio >= 0.95 and mse_gain > 0 and mae_gain < 0:
        return "Active_MSE_only"
    if active_ratio > 0 and mse_gain > 0 and mae_gain > 0:
        return "Selective"
    return "Diagnostic"


def markdown_table(df: pd.DataFrame) -> str:
    def fmt(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value)

    lines = ["| " + " | ".join(df.columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in df.columns) + " |")
    return "\n".join(lines) + "\n"


def metric_pair(mse: float, mae: float) -> str:
    if pd.isna(mse) or pd.isna(mae):
        return "n/a"
    return f"{float(mse):.6f} / {float(mae):.6f}"


def gain_pair(mse_gain: float, mae_gain: float) -> str:
    if pd.isna(mse_gain) or pd.isna(mae_gain):
        return "n/a"
    return f"{float(mse_gain):+.2f}% / {float(mae_gain):+.2f}%"


def build_paper_table(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in full.iterrows():
        posthoc = (
            f"{row['guarded_posthoc_mode']}; "
            f"{metric_pair(row['guarded_posthoc_mse'], row['guarded_posthoc_mae'])}; "
            f"dStatic {gain_pair(row['guarded_posthoc_mse_gain_vs_static_pct'], row['guarded_posthoc_mae_gain_vs_static_pct'])}"
        )
        adaptive = "n/a"
        if not pd.isna(row["adaptive_fusion_mse"]):
            adaptive_parts = [
                metric_pair(row["adaptive_fusion_mse"], row["adaptive_fusion_mae"]),
                f"dStatic {gain_pair(row['adaptive_fusion_mse_gain_vs_static_anchor_pct'], row['adaptive_fusion_mae_gain_vs_static_anchor_pct'])}",
            ]
            if not pd.isna(row["stage3_dynamic_mse"]):
                adaptive_parts.append(
                    f"Stage3 dAdaptive {gain_pair(row['stage3_dynamic_mse_gain_vs_adaptive_pct'], row['stage3_dynamic_mae_gain_vs_adaptive_pct'])}"
                )
            adaptive = "; ".join(adaptive_parts)
        rows.append(
            {
                "Dataset": row["dataset"],
                "Horizon": int(row["horizon"]),
                "Baseline MSE/MAE": metric_pair(row["baseline_mse"], row["baseline_mae"]),
                "Static Anchor MSE/MAE": (
                    f"{metric_pair(row['static_anchor_mse'], row['static_anchor_mae'])}; "
                    f"dBase {gain_pair(row['static_mse_gain_vs_baseline_pct'], row['static_mae_gain_vs_baseline_pct'])}"
                ),
                "Guarded Post-Hoc Dynamic": posthoc,
                "Adaptive Fusion / Stage3": adaptive,
                "Final Route": row["final_headline_route"],
            }
        )
    return pd.DataFrame(rows)


def write_readme(summary: pd.DataFrame, compact: pd.DataFrame, paper: pd.DataFrame) -> None:
    lines = [
        "# Cross-Dataset Route Ablation",
        "",
        "Purpose:",
        "- Freeze a route-level ablation table across ETTh1/Weather/ECL/Solar/Traffic at horizon 96, with Solar-192 as a cross-horizon extension.",
        "- Keep train-time static residual, guarded post-hoc dynamic calibration, and prediction-level adaptive fusion separate.",
        "- Prevent mixed claims such as treating prediction-level adaptive fusion gains as post-hoc dynamic gains.",
        "- ETTh1 is frozen under the current `ParCorr ridgebase_sparse` backend rather than the older `parcorr_regen` interface.",
        "",
        "Compact table:",
        "",
        markdown_table(paper).rstrip(),
        "",
        "Interpretation:",
        "- `Static Anchor` is the stable backbone on Weather/ECL and a useful candidate family on Traffic, but it is not itself a positive standalone route on ETTh1 or Solar.",
        "- `Post-hoc Dynamic` is selective rather than universal: ETTh1 and Solar can be Selective vs static, ECL bypasses, Weather is MSE-only, and Traffic's strict closed loop bypasses.",
        "- `Adaptive Fusion` is now a prediction-level performance branch on Traffic, Solar, and ETTh1, but it should not be conflated with guarded post-hoc dynamic calibration.",
        "- ETTh1 is no longer a hard negative overall once adaptive fusion is allowed, although its guarded post-hoc route stays below baseline and its Stage3 dynamic add-on is currently negative.",
        "- Solar-96 gets a weak extra Stage3 lambda/dynamic gain; Solar-192 falls back to the adaptive-alpha anchor.",
        "- This table should be used as method-route ablation rather than a simple component-toggle ablation.",
        "",
        "Files:",
        "- `cross_dataset_route_ablation_full.csv/md`: full numeric table.",
        "- `cross_dataset_route_ablation_compact.csv/md`: paper-facing compact table.",
        "- `cross_dataset_route_ablation_paper.csv/md`: most readable paper-facing table.",
        "- `manifest.json`: source files and route patterns.",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest = {
        "created_for": "CACI route-level ablation",
        "horizon": "mixed: 96 main benchmark plus Solar-192 extension",
        "result_root": str(RESULT_ROOT),
        "posthoc_summaries": {k: str(v) for k, v in POSTHOC_SUMMARIES.items()},
        "etth1_adaptive_table": str(ETTH1_ADAPTIVE_TABLE),
        "etth1_stage3_table": str(ETTH1_STAGE3_TABLE),
        "traffic_stage2_summary": str(TRAFFIC_STAGE2_SUMMARY),
        "traffic_stage3_summary": str(TRAFFIC_STAGE3_SUMMARY),
        "solar_adaptive_table": str(SOLAR_ADAPTIVE_TABLE),
        "solar_stage3_table": str(SOLAR_STAGE3_TABLE),
        "profiles": DATASET_PROFILES,
    }

    for dataset, profile_name in DATASET_PROFILES.items():
        horizon = DATASET_HORIZONS[dataset]
        route = route_summary(profile_name)
        posthoc = read_one(POSTHOC_SUMMARIES[dataset])
        row = {
            "dataset": dataset,
            "horizon": horizon,
            "baseline_mse": route["baseline_mse"],
            "baseline_mae": route["baseline_mae"],
            "baseline_projection_count": route["baseline_projection_count"],
            "static_anchor_mse": route["static_anchor_mse"],
            "static_anchor_mae": route["static_anchor_mae"],
            "static_projection_count": route["static_projection_count"],
            "static_mse_gain_vs_baseline_pct": pct_gain(route["baseline_mse"], route["static_anchor_mse"]),
            "static_mae_gain_vs_baseline_pct": pct_gain(route["baseline_mae"], route["static_anchor_mae"]),
            "guarded_posthoc_mode": mode_label(dataset, posthoc),
            "guarded_posthoc_selection_reason": s(posthoc, "selection_reason"),
            "guarded_posthoc_active_ratio": f(posthoc, "active_ratio"),
            "guarded_posthoc_static_mse": f(posthoc, "static_mse"),
            "guarded_posthoc_static_mae": f(posthoc, "static_mae"),
            "guarded_posthoc_mse": f(posthoc, "posthoc_mse"),
            "guarded_posthoc_mae": f(posthoc, "posthoc_mae"),
            "guarded_posthoc_mse_gain_vs_static_pct": f(posthoc, "mse_gain_pct"),
            "guarded_posthoc_mae_gain_vs_static_pct": f(posthoc, "mae_gain_pct"),
            "adaptive_fusion_mse": np.nan,
            "adaptive_fusion_mae": np.nan,
            "adaptive_fusion_mse_gain_vs_static_anchor_pct": np.nan,
            "adaptive_fusion_mae_gain_vs_static_anchor_pct": np.nan,
            "stage3_dynamic_mse": np.nan,
            "stage3_dynamic_mae": np.nan,
            "stage3_dynamic_mse_gain_vs_adaptive_pct": np.nan,
            "stage3_dynamic_mae_gain_vs_adaptive_pct": np.nan,
            "final_headline_route": FINAL_HEADLINES[dataset],
        }

        if dataset == "ETTh1":
            adaptive = read_matching(
                ETTH1_ADAPTIVE_TABLE,
                horizon=horizon,
                setting="per_variable_shrinkage_alpha",
            )
            stage3 = read_matching(
                ETTH1_STAGE3_TABLE,
                label="Stage3 closed-form eta2",
                variant="static_p0_dynamic",
            )
            row.update(
                {
                    "adaptive_fusion_mse": f(adaptive, "test_mse"),
                    "adaptive_fusion_mae": f(adaptive, "test_mae"),
                    "adaptive_fusion_mse_gain_vs_static_anchor_pct": pct_gain(
                        route["static_anchor_mse"], f(adaptive, "test_mse")
                    ),
                    "adaptive_fusion_mae_gain_vs_static_anchor_pct": pct_gain(
                        route["static_anchor_mae"], f(adaptive, "test_mae")
                    ),
                    "etth1_adaptive_gain_reference": s(adaptive, "reference_best_single"),
                    "etth1_adaptive_mse_gain_vs_baseline_mean_pct": f(
                        adaptive, "test_mse_gain_vs_baseline_mean_pct"
                    ),
                    "etth1_adaptive_mae_gain_vs_baseline_mean_pct": f(
                        adaptive, "test_mae_gain_vs_baseline_mean_pct"
                    ),
                    "etth1_adaptive_mse_gain_vs_best_single_pct": f(
                        adaptive, "test_mse_gain_vs_best_single_pct"
                    ),
                    "etth1_adaptive_mae_gain_vs_best_single_pct": f(
                        adaptive, "test_mae_gain_vs_best_single_pct"
                    ),
                    "stage3_dynamic_mse": f(stage3, "test_mse"),
                    "stage3_dynamic_mae": f(stage3, "test_mae"),
                    "stage3_dynamic_mse_gain_vs_adaptive_pct": f(
                        stage3, "test_mse_gain_vs_adaptive_anchor_pct"
                    ),
                    "stage3_dynamic_mae_gain_vs_adaptive_pct": f(
                        stage3, "test_mae_gain_vs_adaptive_anchor_pct"
                    ),
                }
            )

        if dataset.startswith("Solar-"):
            adaptive = read_matching(
                SOLAR_ADAPTIVE_TABLE,
                horizon=horizon,
                setting="per_variable_shrinkage_alpha",
            )
            stage3 = read_matching(
                SOLAR_STAGE3_TABLE,
                horizon=horizon,
                label="Stage3 closed-form eta2",
                variant="static_p0_dynamic",
            )
            row.update(
                {
                    "adaptive_fusion_mse": f(adaptive, "test_mse"),
                    "adaptive_fusion_mae": f(adaptive, "test_mae"),
                    "adaptive_fusion_mse_gain_vs_static_anchor_pct": pct_gain(
                        route["static_anchor_mse"], f(adaptive, "test_mse")
                    ),
                    "adaptive_fusion_mae_gain_vs_static_anchor_pct": pct_gain(
                        route["static_anchor_mae"], f(adaptive, "test_mae")
                    ),
                    "solar_adaptive_gain_reference": s(adaptive, "reference_best_single"),
                    "solar_adaptive_mse_gain_vs_baseline_mean_pct": f(
                        adaptive, "test_mse_gain_vs_baseline_mean_pct"
                    ),
                    "solar_adaptive_mae_gain_vs_baseline_mean_pct": f(
                        adaptive, "test_mae_gain_vs_baseline_mean_pct"
                    ),
                    "solar_adaptive_mse_gain_vs_best_single_pct": f(
                        adaptive, "test_mse_gain_vs_best_single_pct"
                    ),
                    "solar_adaptive_mae_gain_vs_best_single_pct": f(
                        adaptive, "test_mae_gain_vs_best_single_pct"
                    ),
                    "stage3_dynamic_mse": f(stage3, "test_mse"),
                    "stage3_dynamic_mae": f(stage3, "test_mae"),
                    "stage3_dynamic_mse_gain_vs_adaptive_pct": f(
                        stage3, "test_mse_gain_vs_adaptive_anchor_pct"
                    ),
                    "stage3_dynamic_mae_gain_vs_adaptive_pct": f(
                        stage3, "test_mae_gain_vs_adaptive_anchor_pct"
                    ),
                }
            )

        if dataset == "Traffic":
            stage2 = read_one(TRAFFIC_STAGE2_SUMMARY)
            stage3 = read_one(TRAFFIC_STAGE3_SUMMARY)
            row.update(
                {
                    "adaptive_fusion_mse": f(stage2, "test_mse"),
                    "adaptive_fusion_mae": f(stage2, "test_mae"),
                    "adaptive_fusion_mse_gain_vs_static_anchor_pct": pct_gain(
                        route["static_anchor_mse"], f(stage2, "test_mse")
                    ),
                    "adaptive_fusion_mae_gain_vs_static_anchor_pct": pct_gain(
                        route["static_anchor_mae"], f(stage2, "test_mae")
                    ),
                    "traffic_adaptive_gain_reference": s(stage2, "reference_best_single"),
                    "traffic_adaptive_mse_gain_vs_reference_pct": f(stage2, "test_mse_gain_vs_best_single_pct"),
                    "traffic_adaptive_mae_gain_vs_reference_pct": f(stage2, "test_mae_gain_vs_best_single_pct"),
                    "stage3_dynamic_mse": f(stage3, "mse"),
                    "stage3_dynamic_mae": f(stage3, "mae"),
                    "stage3_dynamic_mse_gain_vs_adaptive_pct": f(stage3, "mse_gain_vs_stage2_anchor_pct"),
                    "stage3_dynamic_mae_gain_vs_adaptive_pct": f(stage3, "mae_gain_vs_stage2_anchor_pct"),
                }
            )
        rows.append(row)
        manifest[f"{dataset}_route_dirs"] = {
            "baseline": route["baseline_dirs"],
            "static": route["static_dirs"],
        }

    full = pd.DataFrame(rows)
    compact_cols = [
        "dataset",
        "horizon",
        "baseline_mse",
        "baseline_mae",
        "static_anchor_mse",
        "static_anchor_mae",
        "static_mse_gain_vs_baseline_pct",
        "static_mae_gain_vs_baseline_pct",
        "guarded_posthoc_mode",
        "guarded_posthoc_mse_gain_vs_static_pct",
        "guarded_posthoc_mae_gain_vs_static_pct",
        "adaptive_fusion_mse",
        "adaptive_fusion_mae",
        "stage3_dynamic_mse_gain_vs_adaptive_pct",
        "stage3_dynamic_mae_gain_vs_adaptive_pct",
        "final_headline_route",
    ]
    compact = full[compact_cols].copy()
    paper = build_paper_table(full)

    full.to_csv(OUT_DIR / "cross_dataset_route_ablation_full.csv", index=False)
    compact.to_csv(OUT_DIR / "cross_dataset_route_ablation_compact.csv", index=False)
    paper.to_csv(OUT_DIR / "cross_dataset_route_ablation_paper.csv", index=False)
    (OUT_DIR / "cross_dataset_route_ablation_full.md").write_text(markdown_table(full), encoding="utf-8")
    (OUT_DIR / "cross_dataset_route_ablation_compact.md").write_text(markdown_table(compact), encoding="utf-8")
    (OUT_DIR / "cross_dataset_route_ablation_paper.md").write_text(markdown_table(paper), encoding="utf-8")
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(full, compact, paper)

    print(f"[Wrote] {OUT_DIR}", flush=True)
    print(compact.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from posthoc_calibration.evaluation import pct_gain
from traffic_existing_prediction_ensemble import (
    evaluate_group_alpha,
    evaluate_weighted,
    group_blend_weights,
    group_indices,
)


DATA_ROOT = Path(r"C:\Users\cyl\Desktop\data")
ADAPTIVE_DIR = DATA_ROOT / "deltaA_signal_audit" / "etth196_existing_prediction_ensemble_parcorr"
POSTHOC_SUMMARY = (
    DATA_ROOT
    / "deltaA_signal_audit"
    / "etth196_closed_loop_rank_quality_guard_parcorr_ridgebase_sparse"
    / "etth196_static_parcorr_rank_quality_guard_parcorr_closed_loop_test_selected_summary.csv"
)
OUT_DIR = DATA_ROOT / "mechanism_evidence" / "etth196_adaptive_alpha_20260509"
RAW_DIR = OUT_DIR / "raw_outputs"
PREFIX = "etth196_static_parcorr_adaptive_alpha_pilot"
HORIZON = 96


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def onehot(n: int, idx: int) -> np.ndarray:
    weights = np.zeros(n, dtype=np.float64)
    weights[idx] = 1.0
    return weights


def mean_weights(n: int, idx: np.ndarray) -> np.ndarray:
    weights = np.zeros(n, dtype=np.float64)
    weights[idx] = 1.0 / float(idx.size)
    return weights


def fmt_float(value, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_pct(value, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):+.{digits}f}%"


def markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "setting",
        "kind",
        "alpha_summary",
        "val_mse",
        "val_mae",
        "test_mse",
        "test_mae",
        "test_mse_gain_vs_baseline_mean_pct",
        "test_mae_gain_vs_baseline_mean_pct",
        "test_mse_gain_vs_static_mean_pct",
        "test_mae_gain_vs_static_mean_pct",
        "test_mse_gain_vs_best_single_pct",
        "test_mae_gain_vs_best_single_pct",
    ]
    headers = [
        "setting",
        "kind",
        "alpha",
        "val MSE",
        "val MAE",
        "test MSE",
        "test MAE",
        "test MSE vs baseline mean",
        "test MAE vs baseline mean",
        "test MSE vs static mean",
        "test MAE vs static mean",
        "test MSE vs best single",
        "test MAE vs best single",
    ]
    aligns = ["---", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for _, row in df[cols].iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["setting"]),
                    str(row["kind"]),
                    str(row["alpha_summary"]),
                    fmt_float(row["val_mse"]),
                    fmt_float(row["val_mae"]),
                    fmt_float(row["test_mse"]),
                    fmt_float(row["test_mae"]),
                    fmt_pct(row["test_mse_gain_vs_baseline_mean_pct"]),
                    fmt_pct(row["test_mae_gain_vs_baseline_mean_pct"]),
                    fmt_pct(row["test_mse_gain_vs_static_mean_pct"]),
                    fmt_pct(row["test_mae_gain_vs_static_mean_pct"]),
                    fmt_pct(row["test_mse_gain_vs_best_single_pct"]),
                    fmt_pct(row["test_mae_gain_vs_best_single_pct"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def copy_raw_outputs(paths: list[Path]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, RAW_DIR / path.name)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = read_json(ADAPTIVE_DIR / f"{PREFIX}_manifest.json")
    adaptive_summary = read_json(ADAPTIVE_DIR / f"{PREFIX}_adaptive_alpha_summary.json")
    selected = pd.read_csv(ADAPTIVE_DIR / f"{PREFIX}_selected_test_summary.csv").iloc[0]
    variable_alpha_df = pd.read_csv(ADAPTIVE_DIR / f"{PREFIX}_variable_alpha.csv")
    posthoc = pd.read_csv(POSTHOC_SUMMARY).iloc[0]

    copy_raw_outputs(
        [
            ADAPTIVE_DIR / f"{PREFIX}_adaptive_alpha_summary.csv",
            ADAPTIVE_DIR / f"{PREFIX}_adaptive_alpha_summary.json",
            ADAPTIVE_DIR / f"{PREFIX}_candidate_val.csv",
            ADAPTIVE_DIR / f"{PREFIX}_manifest.json",
            ADAPTIVE_DIR / f"{PREFIX}_selected_test_summary.csv",
            ADAPTIVE_DIR / f"{PREFIX}_selected_weights.csv",
            ADAPTIVE_DIR / f"{PREFIX}_val_grid.csv",
            ADAPTIVE_DIR / f"{PREFIX}_variable_alpha.csv",
            POSTHOC_SUMMARY,
        ]
    )

    candidates = manifest["candidates"]
    candidate_names = [candidate["candidate"] for candidate in candidates]
    n = len(candidates)
    baseline_idx, static_idx = group_indices(candidates)
    best_single_name = str(selected["reference_best_single"])
    best_single_idx = candidate_names.index(best_single_name)
    alpha_global = float(adaptive_summary["alpha_global_clipped"])
    alpha_vector = variable_alpha_df["alpha_shrunk"].to_numpy(dtype=np.float64)

    specs = [
        {
            "setting": "baseline_mean",
            "kind": "group_mean",
            "alpha_summary": "0.00 static",
            "weights": mean_weights(n, baseline_idx),
        },
        {
            "setting": "static_mean",
            "kind": "group_mean",
            "alpha_summary": "1.00 static",
            "weights": mean_weights(n, static_idx),
        },
        {
            "setting": f"best_single_{best_single_name}",
            "kind": "single_reference",
            "alpha_summary": "",
            "weights": onehot(n, best_single_idx),
        },
        {
            "setting": "global_closed_form_alpha",
            "kind": "adaptive_global_alpha",
            "alpha_summary": f"{alpha_global:.6f}",
            "weights": group_blend_weights(candidates, alpha_global),
        },
        {
            "setting": "per_variable_shrinkage_alpha",
            "kind": "adaptive_variable_alpha",
            "alpha_summary": (
                f"mean={adaptive_summary['var_alpha_mean']:.6f}; "
                f"std={adaptive_summary['var_alpha_std']:.6f}"
            ),
            "alpha_vector": alpha_vector,
        },
    ]

    rows = []
    for spec in specs:
        if "alpha_vector" in spec:
            val_metrics = evaluate_group_alpha(candidates, spec["alpha_vector"], "val", chunk_size=64)
            test_metrics = evaluate_group_alpha(candidates, spec["alpha_vector"], "test", chunk_size=64)
        else:
            val_metrics = evaluate_weighted(candidates, spec["weights"], "val", chunk_size=64)
            test_metrics = evaluate_weighted(candidates, spec["weights"], "test", chunk_size=64)
        rows.append(
            {
                "horizon": HORIZON,
                "setting": spec["setting"],
                "kind": spec["kind"],
                "alpha_summary": spec["alpha_summary"],
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                "test_mse": test_metrics["mse"],
                "test_mae": test_metrics["mae"],
                "selection_reason": str(selected["selection_reason"]),
                "reference_best_single": best_single_name,
            }
        )

    rows.append(
        {
            "horizon": HORIZON,
            "setting": "posthoc_closed_loop",
            "kind": "guarded_dynamic",
            "alpha_summary": f"{posthoc['mode_status']}:{posthoc['mode_reason']}",
            "val_mse": np.nan,
            "val_mae": np.nan,
            "test_mse": float(posthoc["posthoc_mse"]),
            "test_mae": float(posthoc["posthoc_mae"]),
            "selection_reason": str(posthoc["selection_reason"]),
            "reference_best_single": best_single_name,
        }
    )

    table = pd.DataFrame(rows)
    baseline = table[table["setting"] == "baseline_mean"].iloc[0]
    static = table[table["setting"] == "static_mean"].iloc[0]
    best_single = table[table["setting"].str.startswith("best_single_")].iloc[0]
    for metric in ("mse", "mae"):
        table[f"test_{metric}_gain_vs_baseline_mean_pct"] = [
            pct_gain(float(baseline[f"test_{metric}"]), float(value)) for value in table[f"test_{metric}"]
        ]
        table[f"test_{metric}_gain_vs_static_mean_pct"] = [
            pct_gain(float(static[f"test_{metric}"]), float(value)) for value in table[f"test_{metric}"]
        ]
        table[f"test_{metric}_gain_vs_best_single_pct"] = [
            pct_gain(float(best_single[f"test_{metric}"]), float(value)) for value in table[f"test_{metric}"]
        ]

    alpha_all = variable_alpha_df.copy()
    alpha_all.insert(0, "horizon", HORIZON)
    alpha_all["alpha_rank_desc"] = alpha_all["alpha_shrunk"].rank(ascending=False, method="first").astype(int)

    table_csv = OUT_DIR / "etth196_adaptive_alpha_frozen_table.csv"
    table_md = OUT_DIR / "etth196_adaptive_alpha_frozen_table.md"
    alpha_csv = OUT_DIR / "etth196_adaptive_alpha_variable_alpha.csv"
    readme_path = OUT_DIR / "README.md"

    table.to_csv(table_csv, index=False)
    alpha_all.to_csv(alpha_csv, index=False)
    table_md.write_text(
        "# ETTh1-96 Adaptive-Alpha Frozen Table\n\n"
        "This table evaluates prediction-level adaptive fusion over existing ETTh1 baseline/static anchor predictions. "
        "Validation selects alpha; test is used once for the selected report. No new training is performed.\n\n"
        + markdown_table(table)
        + "\n",
        encoding="utf-8",
    )

    selected_row = table[table["setting"] == "per_variable_shrinkage_alpha"].iloc[0]
    readme_lines = [
        "# ETTh1-96 Adaptive-Alpha Evidence",
        "",
        "Purpose:",
        "- Freeze the lightweight ETTh1-96 adaptive-alpha pilot over the existing baseline/static anchor prediction family.",
        "- Keep this branch separate from the guarded post-hoc closed-loop result, which remains Selective vs static but below the baseline route.",
        "- Reclassify ETTh1 at the route-ablation level: train-time static and guarded post-hoc stay weak, but prediction-level adaptive fusion is now positive overall.",
        "",
        "Main readout:",
        f"- Baseline prediction-mean ensemble: `{baseline['test_mse']:.6f} / {baseline['test_mae']:.6f}`.",
        f"- Static prediction-mean ensemble: `{static['test_mse']:.6f} / {static['test_mae']:.6f}`; "
        f"`dBase {pct_gain(float(baseline['test_mse']), float(static['test_mse'])):+.2f}% / "
        f"{pct_gain(float(baseline['test_mae']), float(static['test_mae'])):+.2f}%`.",
        f"- Guarded post-hoc: `{float(posthoc['posthoc_mse']):.6f} / {float(posthoc['posthoc_mae']):.6f}`; "
        f"`dStatic {float(posthoc['mse_gain_pct']):+.2f}% / {float(posthoc['mae_gain_pct']):+.2f}%`; "
        f"`dBase {pct_gain(float(baseline['test_mse']), float(posthoc['posthoc_mse'])):+.2f}% / "
        f"{pct_gain(float(baseline['test_mae']), float(posthoc['posthoc_mae'])):+.2f}%`.",
        f"- Adaptive fusion (per-variable shrinkage alpha): `{selected_row['test_mse']:.6f} / {selected_row['test_mae']:.6f}`; "
        f"`dBase {selected_row['test_mse_gain_vs_baseline_mean_pct']:+.2f}% / {selected_row['test_mae_gain_vs_baseline_mean_pct']:+.2f}%`; "
        f"`dStatic {selected_row['test_mse_gain_vs_static_mean_pct']:+.2f}% / {selected_row['test_mae_gain_vs_static_mean_pct']:+.2f}%`; "
        f"`dBestSingle {selected_row['test_mse_gain_vs_best_single_pct']:+.2f}% / {selected_row['test_mae_gain_vs_best_single_pct']:+.2f}%`.",
        "- Route-level projection-mean baselines are frozen separately in the cross-dataset route-ablation table; do not mix those reference numbers with the prediction-mean ensemble rows in this package.",
        "",
        "Interpretation:",
        "- ETTh1 is no longer a hard negative overall once prediction-level adaptive fusion is allowed.",
        "- The older guarded post-hoc conclusion still stands locally: dynamic windows can be Selective vs static, but that branch remains weaker than adaptive fusion.",
        "- This result should be treated as a route-ablation update, not as evidence that train-time graph injection itself is strong on ETTh1.",
        "",
        "Files:",
        "- `etth196_adaptive_alpha_frozen_table.csv/md`: frozen comparison table.",
        "- `etth196_adaptive_alpha_variable_alpha.csv`: per-target adaptive alpha values.",
        "- `raw_outputs/`: copied small raw summaries from the pilot run.",
        "- `manifest.json`: source paths and selection summary.",
    ]
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    write_json(
        OUT_DIR / "manifest.json",
        {
            "artifact": "etth196_adaptive_alpha_evidence",
            "output_dir": str(OUT_DIR),
            "horizon": HORIZON,
            "adaptive_dir": str(ADAPTIVE_DIR),
            "prefix": PREFIX,
            "posthoc_summary": str(POSTHOC_SUMMARY),
            "selected_setting": "per_variable_shrinkage_alpha",
            "reference_best_single": best_single_name,
            "adaptive_summary": adaptive_summary,
        },
    )
    print(f"[Done] wrote {table_csv}")
    print(f"[Done] wrote {readme_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
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
STAGE2_DIR = DATA_ROOT / "deltaA_signal_audit" / "traffic96_existing_prediction_ensemble_stage2_light_seed2026"
STAGE15_TABLE = (
    DATA_ROOT
    / "mechanism_evidence"
    / "traffic96_mechanism_performance_20260506"
    / "performance"
    / "adaptive_alpha_ensemble"
    / "tables"
    / "traffic96_static_adaptive_alpha_stage15_frozen_table.csv"
)
PACKAGE_DIR = DATA_ROOT / "mechanism_evidence" / "traffic96_stage2_light_seed2026_20260507"
TABLE_DIR = PACKAGE_DIR / "performance" / "adaptive_alpha_ensemble" / "tables"
PREFIX = "traffic96_static_stage2_light_seed2026"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def fmt_pct(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):+.{digits}f}%"


def markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "label",
        "kind",
        "alpha_summary",
        "val_mse",
        "val_mae",
        "test_mse",
        "test_mae",
        "test_mse_gain_vs_static_p1_pct",
        "test_mae_gain_vs_static_p1_pct",
        "test_mse_gain_vs_stage15_selected_pct",
        "test_mae_gain_vs_stage15_selected_pct",
    ]
    headers = [
        "setting",
        "kind",
        "alpha",
        "val MSE",
        "val MAE",
        "test MSE",
        "test MAE",
        "test MSE gain vs static_p1",
        "test MAE gain vs static_p1",
        "test MSE gain vs Stage1.5",
        "test MAE gain vs Stage1.5",
    ]
    aligns = ["---", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for _, row in df[cols].iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    str(row["kind"]),
                    str(row["alpha_summary"]),
                    fmt_float(row["val_mse"]),
                    fmt_float(row["val_mae"]),
                    fmt_float(row["test_mse"]),
                    fmt_float(row["test_mae"]),
                    fmt_pct(row["test_mse_gain_vs_static_p1_pct"]),
                    fmt_pct(row["test_mae_gain_vs_static_p1_pct"]),
                    fmt_pct(row["test_mse_gain_vs_stage15_selected_pct"]),
                    fmt_pct(row["test_mae_gain_vs_stage15_selected_pct"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def stage15_selected_row() -> dict:
    df = pd.read_csv(STAGE15_TABLE)
    mask = df["label"].astype(str).eq("per-variable shrinkage alpha")
    if not mask.any():
        raise RuntimeError(f"Missing Stage1.5 selected row in {STAGE15_TABLE}")
    row = df[mask].iloc[0].to_dict()
    return {
        "label": "Stage1.5 selected per-variable alpha",
        "kind": "previous_reference",
        "selection_role": "stage15_selected_reference",
        "alpha_summary": str(row["alpha_summary"]),
        "val_mse": float(row["val_mse"]),
        "val_mae": float(row["val_mae"]),
        "test_mse": float(row["test_mse"]),
        "test_mae": float(row["test_mae"]),
    }


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = read_json(STAGE2_DIR / f"{PREFIX}_manifest.json")
    adaptive_summary = read_json(STAGE2_DIR / f"{PREFIX}_adaptive_alpha_summary.json")
    variable_alpha = pd.read_csv(STAGE2_DIR / f"{PREFIX}_variable_alpha.csv")["alpha_shrunk"].to_numpy(dtype=np.float64)

    candidates = manifest["candidates"]
    n = len(candidates)
    baseline_idx, static_idx = group_indices(candidates)
    candidate_names = [candidate["candidate"] for candidate in candidates]
    static_p1_idx = candidate_names.index("static_p1")
    alpha_global = float(adaptive_summary["alpha_global_clipped"])
    stage15_ref = stage15_selected_row()

    specs = [
        {
            "label": "best single static_p1",
            "kind": "single_reference",
            "alpha_summary": "",
            "weights": onehot(n, static_p1_idx),
            "selection_role": "reference_best_single",
        },
        {
            "label": "baseline mean, 4 seeds",
            "kind": "group_mean",
            "alpha_summary": "0.00 static",
            "weights": mean_weights(n, baseline_idx),
            "selection_role": "ablation",
        },
        {
            "label": "staticcausal mean, 4 seeds",
            "kind": "group_mean",
            "alpha_summary": "1.00 static",
            "weights": mean_weights(n, static_idx),
            "selection_role": "ablation",
        },
        {
            "label": "alpha=0.50 equal blend",
            "kind": "global_blend",
            "alpha_summary": "0.50",
            "weights": group_blend_weights(candidates, 0.50),
            "selection_role": "blind_equal_blend",
        },
        {
            "label": "global closed-form alpha",
            "kind": "adaptive_global_alpha",
            "alpha_summary": f"{alpha_global:.6f}",
            "weights": group_blend_weights(candidates, alpha_global),
            "selection_role": "stage2_global_adaptive",
        },
        {
            "label": "per-variable shrinkage alpha",
            "kind": "adaptive_variable_alpha",
            "alpha_summary": (
                f"mean={adaptive_summary['var_alpha_mean']:.6f}; "
                f"std={adaptive_summary['var_alpha_std']:.6f}"
            ),
            "alpha_vector": variable_alpha,
            "selection_role": "stage2_selected",
        },
    ]

    rows = [stage15_ref]
    for spec in specs:
        if "alpha_vector" in spec:
            val_metrics = evaluate_group_alpha(candidates, spec["alpha_vector"], "val", chunk_size=64)
            test_metrics = evaluate_group_alpha(candidates, spec["alpha_vector"], "test", chunk_size=64)
        else:
            val_metrics = evaluate_weighted(candidates, spec["weights"], "val", chunk_size=64)
            test_metrics = evaluate_weighted(candidates, spec["weights"], "test", chunk_size=64)
        rows.append(
            {
                "label": spec["label"],
                "kind": spec["kind"],
                "selection_role": spec["selection_role"],
                "alpha_summary": spec["alpha_summary"],
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                "test_mse": test_metrics["mse"],
                "test_mae": test_metrics["mae"],
            }
        )

    df = pd.DataFrame(rows)
    static_ref = df[df["selection_role"] == "reference_best_single"].iloc[0]
    stage15 = df[df["selection_role"] == "stage15_selected_reference"].iloc[0]
    for split in ["val", "test"]:
        df[f"{split}_mse_gain_vs_static_p1_pct"] = [
            pct_gain(float(static_ref[f"{split}_mse"]), float(value)) for value in df[f"{split}_mse"]
        ]
        df[f"{split}_mae_gain_vs_static_p1_pct"] = [
            pct_gain(float(static_ref[f"{split}_mae"]), float(value)) for value in df[f"{split}_mae"]
        ]
        df[f"{split}_mse_gain_vs_stage15_selected_pct"] = [
            pct_gain(float(stage15[f"{split}_mse"]), float(value)) for value in df[f"{split}_mse"]
        ]
        df[f"{split}_mae_gain_vs_stage15_selected_pct"] = [
            pct_gain(float(stage15[f"{split}_mae"]), float(value)) for value in df[f"{split}_mae"]
        ]

    csv_path = TABLE_DIR / f"{PREFIX}_frozen_table.csv"
    md_path = TABLE_DIR / f"{PREFIX}_frozen_table.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(
        "# Traffic96 Stage2-Light Frozen Performance Table\n\n"
        "Stage2-Light adds one paired seed (`projection_3`, seed=2026) to the existing "
        "three baseline/staticcausal projections. Selection remains validation-only. "
        "The Stage1.5 selected row is included only as a historical reference.\n\n"
        + markdown_table(df)
        + "\n",
        encoding="utf-8",
    )
    print(f"[Done] wrote {csv_path}")
    print(f"[Done] wrote {md_path}")


if __name__ == "__main__":
    main()

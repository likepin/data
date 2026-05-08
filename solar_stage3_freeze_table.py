from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DATA_ROOT = Path(r"C:\Users\cyl\Desktop\data")
ADAPTIVE_TABLE = (
    DATA_ROOT
    / "mechanism_evidence"
    / "solar96_192_adaptive_alpha_20260508"
    / "solar_adaptive_alpha_frozen_table.csv"
)
OUT_DIR = DATA_ROOT / "mechanism_evidence" / "solar96_192_stage3_lambda_three_source_20260508"

STAGE3_RUNS = [
    {
        "horizon": 96,
        "variant": "static_p0_dynamic",
        "dir": DATA_ROOT / "deltaA_signal_audit" / "solar96_stage3_lambda_three_source_closed_form_eta2",
        "prefix": "solar96_static_stage3_closed_form_eta2",
    },
    {
        "horizon": 96,
        "variant": "static_mean_dynamic",
        "dir": DATA_ROOT / "deltaA_signal_audit" / "solar96_stage3_lambda_three_source_closed_form_eta2_staticmean",
        "prefix": "solar96_static_stage3_closed_form_eta2_staticmean",
    },
    {
        "horizon": 192,
        "variant": "static_p0_dynamic",
        "dir": DATA_ROOT / "deltaA_signal_audit" / "solar192_stage3_lambda_three_source_closed_form_eta2",
        "prefix": "solar192_static_stage3_closed_form_eta2",
    },
    {
        "horizon": 192,
        "variant": "static_mean_dynamic",
        "dir": DATA_ROOT / "deltaA_signal_audit" / "solar192_stage3_lambda_three_source_closed_form_eta2_staticmean",
        "prefix": "solar192_static_stage3_closed_form_eta2_staticmean",
    },
]


def pct_gain(before: float, after: float) -> float:
    if abs(float(before)) < 1e-12:
        return 0.0
    return 100.0 * (float(before) - float(after)) / float(before)


def read_one(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError(f"Expected one row in {path}, got {len(df)}")
    return df.iloc[0]


def fmt_float(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def fmt_pct(value: float, digits: int = 4) -> str:
    return f"{float(value):+.{digits}f}%"


def markdown_table(df: pd.DataFrame) -> str:
    columns = [
        "horizon",
        "label",
        "variant",
        "test_mse",
        "test_mae",
        "test_mse_gain_vs_adaptive_anchor_pct",
        "test_mae_gain_vs_adaptive_anchor_pct",
        "test_mse_gain_vs_baseline_mean_pct",
        "test_mae_gain_vs_baseline_mean_pct",
        "selected_ensemble",
    ]
    headers = [
        "horizon",
        "label",
        "variant",
        "test MSE",
        "test MAE",
        "MSE vs adaptive",
        "MAE vs adaptive",
        "MSE vs baseline mean",
        "MAE vs baseline mean",
        "selected",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---:", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---"]) + " |",
    ]
    for _, row in df[columns].iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["horizon"])),
                    str(row["label"]),
                    str(row["variant"]),
                    fmt_float(row["test_mse"]),
                    fmt_float(row["test_mae"]),
                    fmt_pct(row["test_mse_gain_vs_adaptive_anchor_pct"]),
                    fmt_pct(row["test_mae_gain_vs_adaptive_anchor_pct"]),
                    fmt_pct(row["test_mse_gain_vs_baseline_mean_pct"]),
                    fmt_pct(row["test_mae_gain_vs_baseline_mean_pct"]),
                    str(row["selected_ensemble"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adaptive = pd.read_csv(ADAPTIVE_TABLE)
    rows = []
    raw_refs = []
    for horizon, sub in adaptive.groupby("horizon"):
        baseline = sub[sub["setting"] == "baseline_mean"].iloc[0]
        anchor = sub[sub["setting"] == "per_variable_shrinkage_alpha"].iloc[0]
        rows.append(
            {
                "horizon": int(horizon),
                "label": "adaptive-alpha anchor",
                "variant": "anchor",
                "selected_ensemble": "per_variable_shrinkage_alpha",
                "eta_mode": "anchor",
                "eta_mult": 0.0,
                "eta_raw": 0.0,
                "eta_clip_reason": "anchor",
                "target_mask": "n/a",
                "target_count": 0,
                "dynamic_source": "n/a",
                "test_mse": float(anchor["test_mse"]),
                "test_mae": float(anchor["test_mae"]),
                "test_mse_gain_vs_adaptive_anchor_pct": 0.0,
                "test_mae_gain_vs_adaptive_anchor_pct": 0.0,
                "test_mse_gain_vs_baseline_mean_pct": pct_gain(baseline["test_mse"], anchor["test_mse"]),
                "test_mae_gain_vs_baseline_mean_pct": pct_gain(baseline["test_mae"], anchor["test_mae"]),
                "selection_reason": str(anchor["selection_reason"]),
            }
        )

    for run in STAGE3_RUNS:
        summary_path = run["dir"] / f"{run['prefix']}_test_selected_summary.csv"
        row = read_one(summary_path)
        baseline = adaptive[
            (adaptive["horizon"] == run["horizon"]) & (adaptive["setting"] == "baseline_mean")
        ].iloc[0]
        rows.append(
            {
                "horizon": int(run["horizon"]),
                "label": "Stage3 closed-form eta2",
                "variant": run["variant"],
                "selected_ensemble": str(row["ensemble"]),
                "eta_mode": str(row["eta_mode"]),
                "eta_mult": float(row["eta_mult"]),
                "eta_raw": float(row["eta_raw"]),
                "eta_clip_reason": str(row["eta_clip_reason"]),
                "target_mask": str(row["target_mask"]),
                "target_count": int(row["target_count"]),
                "dynamic_source": str(row["dynamic_source"]),
                "test_mse": float(row["mse"]),
                "test_mae": float(row["mae"]),
                "test_mse_gain_vs_adaptive_anchor_pct": float(row["mse_gain_vs_stage2_anchor_pct"]),
                "test_mae_gain_vs_adaptive_anchor_pct": float(row["mae_gain_vs_stage2_anchor_pct"]),
                "test_mse_gain_vs_baseline_mean_pct": pct_gain(baseline["test_mse"], row["mse"]),
                "test_mae_gain_vs_baseline_mean_pct": pct_gain(baseline["test_mae"], row["mae"]),
                "selection_reason": str(row["selection_reason"]),
            }
        )
        raw_refs.append(
            {
                "horizon": int(run["horizon"]),
                "variant": run["variant"],
                "summary": str(summary_path),
                "val_grid": str(run["dir"] / f"{run['prefix']}_val_grid.csv"),
                "eta_candidates": str(run["dir"] / f"{run['prefix']}_eta_candidates.csv"),
            }
        )

    table = pd.DataFrame(rows).sort_values(["horizon", "label", "variant"]).reset_index(drop=True)
    table_csv = OUT_DIR / "solar_stage3_lambda_three_source_frozen_table.csv"
    table_md = OUT_DIR / "solar_stage3_lambda_three_source_frozen_table.md"
    readme = OUT_DIR / "README.md"
    manifest = OUT_DIR / "manifest.json"

    table.to_csv(table_csv, index=False)
    table_md.write_text(
        "# Solar Stage3 Lambda Three-Source Frozen Table\n\n"
        "Stage3 adds a validation-selected lambda-gated dynamic increment on top of the "
        "Solar adaptive-alpha baseline/static anchor. Test is used only for the selected rows.\n\n"
        + markdown_table(table)
        + "\n",
        encoding="utf-8",
    )

    s96 = table[
        (table["horizon"] == 96)
        & (table["label"] == "Stage3 closed-form eta2")
        & (table["variant"] == "static_p0_dynamic")
    ].iloc[0]
    s192 = table[
        (table["horizon"] == 192)
        & (table["label"] == "Stage3 closed-form eta2")
        & (table["variant"] == "static_p0_dynamic")
    ].iloc[0]
    readme.write_text(
        "\n".join(
            [
                "# Solar96/Solar192 Stage3 Lambda Three-Source Evidence",
                "",
                "Boundary: this is a post-hoc dynamic add-on on top of the adaptive-alpha anchor. It is not a new training run.",
                "",
                "Key results:",
                (
                    f"- Solar-96: Stage3 selected `{s96['selected_ensemble']}` with "
                    f"MSE/MAE gain vs adaptive anchor {fmt_pct(s96['test_mse_gain_vs_adaptive_anchor_pct'])}/"
                    f"{fmt_pct(s96['test_mae_gain_vs_adaptive_anchor_pct'])}."
                ),
                (
                    f"- Solar-192: Stage3 selected `{s192['selected_ensemble']}` with "
                    f"MSE/MAE gain vs adaptive anchor {fmt_pct(s192['test_mse_gain_vs_adaptive_anchor_pct'])}/"
                    f"{fmt_pct(s192['test_mae_gain_vs_adaptive_anchor_pct'])}."
                ),
                "",
                "Interpretation:",
                "- Adaptive alpha is the Solar performance anchor.",
                "- Stage3 is weak positive for Solar-96 and bypasses to the anchor for Solar-192 under validation selection.",
                "",
                "Files:",
                "- `solar_stage3_lambda_three_source_frozen_table.csv/md`: frozen comparison table.",
                "- `manifest.json`: source run directories and raw summary paths.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "artifact": "solar_stage3_lambda_three_source_evidence",
                "adaptive_table": str(ADAPTIVE_TABLE),
                "raw_refs": raw_refs,
                "output_dir": str(OUT_DIR),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Done] wrote {table_csv}")
    print(f"[Done] wrote {readme}")


if __name__ == "__main__":
    main()

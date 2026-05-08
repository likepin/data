from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


DATA_ROOT = Path(r"C:\Users\cyl\Desktop\data")
ADAPTIVE_TABLE = (
    DATA_ROOT
    / "mechanism_evidence"
    / "etth196_adaptive_alpha_20260509"
    / "etth196_adaptive_alpha_frozen_table.csv"
)
OUT_DIR = DATA_ROOT / "mechanism_evidence" / "etth196_stage3_lambda_three_source_20260509"
RAW_DIR = OUT_DIR / "raw_outputs"

STAGE3_RUNS = [
    {
        "label": "Stage3 closed-form eta2",
        "variant": "static_p0_dynamic",
        "dir": DATA_ROOT / "deltaA_signal_audit" / "etth196_stage3_lambda_three_source_closed_form_eta2",
        "prefix": "etth196_static_parcorr_stage3_closed_form_eta2",
    },
    {
        "label": "Stage3 closed-form eta2",
        "variant": "static_mean_dynamic",
        "dir": DATA_ROOT / "deltaA_signal_audit" / "etth196_stage3_lambda_three_source_closed_form_eta2_staticmean",
        "prefix": "etth196_static_parcorr_stage3_closed_form_eta2_staticmean",
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
        "label",
        "variant",
        "test_mse",
        "test_mae",
        "test_mse_gain_vs_adaptive_anchor_pct",
        "test_mae_gain_vs_adaptive_anchor_pct",
        "test_mse_gain_vs_baseline_mean_pct",
        "test_mae_gain_vs_baseline_mean_pct",
        "selected_ensemble",
        "eta_mult",
        "eta_raw",
        "eta_clip_reason",
        "target_mask",
    ]
    headers = [
        "label",
        "variant",
        "test MSE",
        "test MAE",
        "MSE vs adaptive",
        "MAE vs adaptive",
        "MSE vs baseline mean",
        "MAE vs baseline mean",
        "selected",
        "eta",
        "eta_raw",
        "clip",
        "target_mask",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * 2 + ["---:"] * 6 + ["---"] + ["---:"] * 2 + ["---"] * 2) + " |",
    ]
    for _, row in df[columns].iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    str(row["variant"]),
                    fmt_float(row["test_mse"]),
                    fmt_float(row["test_mae"]),
                    fmt_pct(row["test_mse_gain_vs_adaptive_anchor_pct"]),
                    fmt_pct(row["test_mae_gain_vs_adaptive_anchor_pct"]),
                    fmt_pct(row["test_mse_gain_vs_baseline_mean_pct"]),
                    fmt_pct(row["test_mae_gain_vs_baseline_mean_pct"]),
                    str(row["selected_ensemble"]),
                    fmt_float(row["eta_mult"], digits=3),
                    fmt_float(row["eta_raw"], digits=3),
                    str(row["eta_clip_reason"]),
                    str(row["target_mask"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def copy_raw_outputs(path: Path, prefix: str) -> list[str]:
    copied = []
    for suffix in [
        "_eta_candidates.csv",
        "_manifest.json",
        "_selected_val_summary.csv",
        "_shuffled_gamma_summary.csv",
        "_test_selected_summary.csv",
        "_val_fold_grid.csv",
        "_val_grid.csv",
        "_val_selected_recomputed_summary.csv",
    ]:
        src = path / f"{prefix}{suffix}"
        if src.exists():
            dst = RAW_DIR / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    adaptive = pd.read_csv(ADAPTIVE_TABLE)
    baseline = adaptive[adaptive["setting"] == "baseline_mean"].iloc[0]
    anchor = adaptive[adaptive["setting"] == "per_variable_shrinkage_alpha"].iloc[0]

    rows = [
        {
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
            "val_mse": float(anchor["val_mse"]),
            "val_mae": float(anchor["val_mae"]),
            "test_mse": float(anchor["test_mse"]),
            "test_mae": float(anchor["test_mae"]),
            "test_mse_gain_vs_adaptive_anchor_pct": 0.0,
            "test_mae_gain_vs_adaptive_anchor_pct": 0.0,
            "test_mse_gain_vs_baseline_mean_pct": float(anchor["test_mse_gain_vs_baseline_mean_pct"]),
            "test_mae_gain_vs_baseline_mean_pct": float(anchor["test_mae_gain_vs_baseline_mean_pct"]),
            "selection_reason": str(anchor["selection_reason"]),
        }
    ]

    raw_refs = []
    for run in STAGE3_RUNS:
        test_summary = read_one(run["dir"] / f"{run['prefix']}_test_selected_summary.csv")
        val_summary = read_one(run["dir"] / f"{run['prefix']}_selected_val_summary.csv")
        copied = copy_raw_outputs(run["dir"], run["prefix"])
        rows.append(
            {
                "label": run["label"],
                "variant": run["variant"],
                "selected_ensemble": str(test_summary["ensemble"]),
                "eta_mode": str(test_summary["eta_mode"]),
                "eta_mult": float(test_summary["eta_mult"]),
                "eta_raw": float(test_summary["eta_raw"]),
                "eta_clip_reason": str(test_summary["eta_clip_reason"]),
                "target_mask": str(test_summary["target_mask"]),
                "target_count": int(test_summary["target_count"]),
                "dynamic_source": str(test_summary["dynamic_source"]),
                "val_mse": float(val_summary["mse"]),
                "val_mae": float(val_summary["mae"]),
                "test_mse": float(test_summary["mse"]),
                "test_mae": float(test_summary["mae"]),
                "test_mse_gain_vs_adaptive_anchor_pct": float(test_summary["mse_gain_vs_stage2_anchor_pct"]),
                "test_mae_gain_vs_adaptive_anchor_pct": float(test_summary["mae_gain_vs_stage2_anchor_pct"]),
                "test_mse_gain_vs_baseline_mean_pct": pct_gain(baseline["test_mse"], test_summary["mse"]),
                "test_mae_gain_vs_baseline_mean_pct": pct_gain(baseline["test_mae"], test_summary["mae"]),
                "selection_reason": str(test_summary["selection_reason"]),
            }
        )
        raw_refs.append(
            {
                "variant": run["variant"],
                "dir": str(run["dir"]),
                "prefix": run["prefix"],
                "copied_outputs": copied,
            }
        )

    table = pd.DataFrame(rows).sort_values(["label", "variant"]).reset_index(drop=True)
    table_csv = OUT_DIR / "etth196_stage3_lambda_three_source_frozen_table.csv"
    table_md = OUT_DIR / "etth196_stage3_lambda_three_source_frozen_table.md"
    readme = OUT_DIR / "README.md"
    manifest = OUT_DIR / "manifest.json"

    table.to_csv(table_csv, index=False)
    table_md.write_text(
        "# ETTh1-96 Stage3 Lambda Three-Source Frozen Table\n\n"
        "Stage3 adds a validation-selected lambda-gated dynamic increment on top of the ETTh1 adaptive-alpha anchor. "
        "This is a post-hoc dynamic add-on, not a new training run.\n\n"
        + markdown_table(table)
        + "\n",
        encoding="utf-8",
    )

    p0 = table[(table["label"] == "Stage3 closed-form eta2") & (table["variant"] == "static_p0_dynamic")].iloc[0]
    sm = table[(table["label"] == "Stage3 closed-form eta2") & (table["variant"] == "static_mean_dynamic")].iloc[0]
    readme.write_text(
        "\n".join(
            [
                "# ETTh1-96 Stage3 Lambda Three-Source Evidence",
                "",
                "Boundary:",
                "- Stage3 here means a lambda-gated dynamic increment added on top of the ETTh1 adaptive-alpha anchor.",
                "- This should be interpreted as a negative audit over the dynamic branch, not as a new main performance route.",
                "",
                "Key results:",
                (
                    f"- Adaptive-alpha anchor: `{fmt_float(anchor['test_mse'])} / {fmt_float(anchor['test_mae'])}`."
                ),
                (
                    f"- `static_p0_dynamic`: selected `{p0['selected_ensemble']}`, "
                    f"test `{fmt_float(p0['test_mse'])} / {fmt_float(p0['test_mae'])}`, "
                    f"gain vs adaptive anchor `{fmt_pct(p0['test_mse_gain_vs_adaptive_anchor_pct'])} / "
                    f"{fmt_pct(p0['test_mae_gain_vs_adaptive_anchor_pct'])}`."
                ),
                (
                    f"- `static_mean_dynamic`: selected `{sm['selected_ensemble']}`, "
                    f"test `{fmt_float(sm['test_mse'])} / {fmt_float(sm['test_mae'])}`, "
                    f"gain vs adaptive anchor `{fmt_pct(sm['test_mse_gain_vs_adaptive_anchor_pct'])} / "
                    f"{fmt_pct(sm['test_mae_gain_vs_adaptive_anchor_pct'])}`."
                ),
                (
                    f"- Both selected rows saturate at `eta_mult=2.0` with `eta_raw={fmt_float(p0['eta_raw'], 3)}` "
                    f"and `{fmt_float(sm['eta_raw'], 3)}` respectively, and both choose `target_mask=all`."
                ),
                "",
                "Interpretation:",
                "- ETTh1 adaptive fusion remains the strongest current route.",
                "- Adding the Traffic-style lambda-gated dynamic increment improves validation but hurts test under both dynamic-source choices.",
                "- Therefore ETTh1 should currently be frozen as `adaptive fusion positive, Stage3 dynamic add-on negative`.",
                "",
                "Files:",
                "- `etth196_stage3_lambda_three_source_frozen_table.csv/md`: frozen comparison table.",
                "- `raw_outputs/`: copied selected summaries, eta candidates, val grid, fold grid, and shuffle summaries.",
                "- `manifest.json`: source run directories and copied raw output paths.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "artifact": "etth196_stage3_lambda_three_source_evidence",
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

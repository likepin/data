from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


DATA_ROOT = Path(r"C:\Users\cyl\Desktop\data")
STAGE2_TABLE = (
    DATA_ROOT
    / "mechanism_evidence"
    / "traffic96_stage2_light_seed2026_20260507"
    / "performance"
    / "adaptive_alpha_ensemble"
    / "tables"
    / "traffic96_static_stage2_light_seed2026_frozen_table.csv"
)
DEFAULT_STAGE3_DIR = DATA_ROOT / "deltaA_signal_audit" / "traffic96_stage3_lambda_three_source_pilot"
STATICMEAN_STAGE3_DIR = DATA_ROOT / "deltaA_signal_audit" / "traffic96_stage3_lambda_three_source_pilot_staticmean"
PACKAGE_DIR = DATA_ROOT / "mechanism_evidence" / "traffic96_stage3_lambda_three_source_20260507"
OUT_DIR = PACKAGE_DIR / "performance" / "stage3_lambda_three_source"
TABLE_DIR = OUT_DIR / "tables"
RAW_DIR = OUT_DIR / "raw_outputs"
PREFIX = "traffic96_static_stage3_lambda_three_source"


def pct_gain(before: float, after: float) -> float:
    if abs(float(before)) < 1e-12:
        return 0.0
    return 100.0 * (float(before) - float(after)) / float(before)


def read_one(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError(f"Expected exactly one row in {path}, got {len(df)}")
    return df.iloc[0]


def stage2_row(stage2: pd.DataFrame, label: str) -> pd.Series:
    rows = stage2[stage2["label"] == label]
    if len(rows) != 1:
        raise ValueError(f"Expected one Stage2 row for label={label}, got {len(rows)}")
    return rows.iloc[0]


def metric_row(
    *,
    label: str,
    kind: str,
    selection_role: str,
    val_mse: float,
    val_mae: float,
    test_mse: float,
    test_mae: float,
    static_ref: pd.Series,
    stage15_ref: pd.Series,
    stage2_ref: pd.Series,
    notes: str,
) -> dict:
    return {
        "label": label,
        "kind": kind,
        "selection_role": selection_role,
        "val_mse": float(val_mse),
        "val_mae": float(val_mae),
        "test_mse": float(test_mse),
        "test_mae": float(test_mae),
        "val_mse_gain_vs_static_p1_pct": pct_gain(static_ref["val_mse"], val_mse),
        "val_mae_gain_vs_static_p1_pct": pct_gain(static_ref["val_mae"], val_mae),
        "test_mse_gain_vs_static_p1_pct": pct_gain(static_ref["test_mse"], test_mse),
        "test_mae_gain_vs_static_p1_pct": pct_gain(static_ref["test_mae"], test_mae),
        "val_mse_gain_vs_stage15_selected_pct": pct_gain(stage15_ref["val_mse"], val_mse),
        "val_mae_gain_vs_stage15_selected_pct": pct_gain(stage15_ref["val_mae"], val_mae),
        "test_mse_gain_vs_stage15_selected_pct": pct_gain(stage15_ref["test_mse"], test_mse),
        "test_mae_gain_vs_stage15_selected_pct": pct_gain(stage15_ref["test_mae"], test_mae),
        "val_mse_gain_vs_stage2_anchor_pct": pct_gain(stage2_ref["val_mse"], val_mse),
        "val_mae_gain_vs_stage2_anchor_pct": pct_gain(stage2_ref["val_mae"], val_mae),
        "test_mse_gain_vs_stage2_anchor_pct": pct_gain(stage2_ref["test_mse"], test_mse),
        "test_mae_gain_vs_stage2_anchor_pct": pct_gain(stage2_ref["test_mae"], test_mae),
        "notes": notes,
    }


def copy_outputs(src_dir: Path, dst_dir: Path) -> list[str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(src_dir.glob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".npy":
            raise RuntimeError(f"Refusing to package large array: {path}")
        target = dst_dir / path.name
        shutil.copy2(path, target)
        copied.append(str(target.relative_to(PACKAGE_DIR)))
    return copied


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    def fmt(value) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        text = str(value)
        return text.replace("|", "\\|")

    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    stage2 = pd.read_csv(STAGE2_TABLE)
    static_ref = stage2_row(stage2, "best single static_p1")
    stage15_ref = stage2_row(stage2, "Stage1.5 selected per-variable alpha")
    stage2_ref = stage2_row(stage2, "per-variable shrinkage alpha")

    default_val = read_one(DEFAULT_STAGE3_DIR / "traffic96_static_stage3_pilot_selected_val_summary.csv")
    default_test = read_one(DEFAULT_STAGE3_DIR / "traffic96_static_stage3_pilot_test_selected_summary.csv")
    staticmean_val = read_one(
        STATICMEAN_STAGE3_DIR / "traffic96_static_stage3_pilot_staticmean_selected_val_summary.csv"
    )
    staticmean_test = read_one(
        STATICMEAN_STAGE3_DIR / "traffic96_static_stage3_pilot_staticmean_test_selected_summary.csv"
    )

    rows = [
        metric_row(
            label="best single static_p1",
            kind="single_reference",
            selection_role="reference_best_single",
            val_mse=static_ref["val_mse"],
            val_mae=static_ref["val_mae"],
            test_mse=static_ref["test_mse"],
            test_mae=static_ref["test_mae"],
            static_ref=static_ref,
            stage15_ref=stage15_ref,
            stage2_ref=stage2_ref,
            notes="Stage2 frozen-table reference.",
        ),
        metric_row(
            label="Stage1.5 adaptive-alpha selected",
            kind="adaptive_variable_alpha",
            selection_role="previous_reference",
            val_mse=stage15_ref["val_mse"],
            val_mae=stage15_ref["val_mae"],
            test_mse=stage15_ref["test_mse"],
            test_mae=stage15_ref["test_mae"],
            static_ref=static_ref,
            stage15_ref=stage15_ref,
            stage2_ref=stage2_ref,
            notes="Previous Traffic performance reference before one additional paired seed.",
        ),
        metric_row(
            label="Stage2 adaptive-alpha anchor",
            kind="adaptive_variable_alpha",
            selection_role="stage2_anchor",
            val_mse=stage2_ref["val_mse"],
            val_mae=stage2_ref["val_mae"],
            test_mse=stage2_ref["test_mse"],
            test_mae=stage2_ref["test_mae"],
            static_ref=static_ref,
            stage15_ref=stage15_ref,
            stage2_ref=stage2_ref,
            notes="Anchor used by Stage3: baseline/static adaptive-alpha ensemble.",
        ),
        metric_row(
            label="Stage3 lambda three-source, static_p0 dynamic",
            kind="lambda_gated_dynamic_increment",
            selection_role="stage3_selected",
            val_mse=default_val["mse"],
            val_mae=default_val["mae"],
            test_mse=default_test["mse"],
            test_mae=default_test["mae"],
            static_ref=static_ref,
            stage15_ref=stage15_ref,
            stage2_ref=stage2_ref,
            notes="Default Stage3 pilot; dynamic source matches existing posthoc static_p0 convention.",
        ),
        metric_row(
            label="Stage3 lambda three-source, static_mean audit",
            kind="lambda_gated_dynamic_increment_audit",
            selection_role="audit_reference",
            val_mse=staticmean_val["mse"],
            val_mae=staticmean_val["mae"],
            test_mse=staticmean_test["mse"],
            test_mae=staticmean_test["mae"],
            static_ref=static_ref,
            stage15_ref=stage15_ref,
            stage2_ref=stage2_ref,
            notes="Audit using mean static predictor as dynamic source; confirms result is not projection_0-specific.",
        ),
    ]
    table = pd.DataFrame(rows)
    table_path = TABLE_DIR / f"{PREFIX}_frozen_table.csv"
    table.to_csv(table_path, index=False)
    write_markdown_table(table, TABLE_DIR / f"{PREFIX}_frozen_table.md")

    default_fold = pd.read_csv(DEFAULT_STAGE3_DIR / "traffic96_static_stage3_pilot_val_fold_grid.csv")
    default_fold.to_csv(TABLE_DIR / f"{PREFIX}_default_val_fold_grid.csv", index=False)
    selected_fold = default_fold[default_fold["ensemble"] == "stage3_eta1_all"].copy()
    selected_fold.to_csv(TABLE_DIR / f"{PREFIX}_default_selected_val_folds.csv", index=False)

    default_shuffle = pd.read_csv(DEFAULT_STAGE3_DIR / "traffic96_static_stage3_pilot_shuffled_gamma_summary.csv")
    staticmean_shuffle = pd.read_csv(
        STATICMEAN_STAGE3_DIR / "traffic96_static_stage3_pilot_staticmean_shuffled_gamma_summary.csv"
    )
    default_shuffle["variant"] = "static_p0_dynamic"
    staticmean_shuffle["variant"] = "static_mean_dynamic"
    shuffle = pd.concat([default_shuffle, staticmean_shuffle], ignore_index=True)
    shuffle.to_csv(TABLE_DIR / f"{PREFIX}_shuffled_gamma_summary.csv", index=False)

    copied = []
    copied += copy_outputs(DEFAULT_STAGE3_DIR, RAW_DIR / "default_static_p0_dynamic")
    copied += copy_outputs(STATICMEAN_STAGE3_DIR, RAW_DIR / "audit_static_mean_dynamic")

    selected_default = table[table["selection_role"] == "stage3_selected"].iloc[0]
    manifest = {
        "package": "traffic96_stage3_lambda_three_source_20260507",
        "status": "weak_positive_stage3_pilot",
        "default_selected": {
            "label": selected_default["label"],
            "test_mse": float(selected_default["test_mse"]),
            "test_mae": float(selected_default["test_mae"]),
            "test_mse_gain_vs_stage2_anchor_pct": float(selected_default["test_mse_gain_vs_stage2_anchor_pct"]),
            "test_mae_gain_vs_stage2_anchor_pct": float(selected_default["test_mae_gain_vs_stage2_anchor_pct"]),
            "test_mse_gain_vs_static_p1_pct": float(selected_default["test_mse_gain_vs_static_p1_pct"]),
            "test_mae_gain_vs_static_p1_pct": float(selected_default["test_mae_gain_vs_static_p1_pct"]),
        },
        "interpretation": (
            "Stage3 lambda-gated dynamic increment adds a small positive gain over the Stage2 adaptive-alpha "
            "anchor, but the test shuffled-gamma negative control is thin; treat as weak positive increment, "
            "not a strong dynamic-branch success."
        ),
        "source_outputs": {
            "default_static_p0_dynamic": str(DEFAULT_STAGE3_DIR),
            "audit_static_mean_dynamic": str(STATICMEAN_STAGE3_DIR),
            "stage2_frozen_table": str(STAGE2_TABLE),
        },
        "copied_files": copied,
    }
    (PACKAGE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (PACKAGE_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Traffic96 Stage3 Lambda Three-Source Evidence",
                "",
                "This package freezes the Stage3-Pilot result for Traffic-96.",
                "",
                "Scope:",
                "- `Stage2 anchor`: adaptive-alpha baseline/static ensemble.",
                "- `Stage3`: add a lambda-gated posthoc dynamic increment on top of the Stage2 anchor.",
                "- Default dynamic source: `static_p0`, matching the existing posthoc closed-loop convention.",
                "- Audit dynamic source: `static_mean`, confirming the result is not projection-0-specific.",
                "",
                "Interpretation:",
                "- The default Stage3 result is weak positive over Stage2.",
                "- The test shuffled-gamma negative control is thin.",
                "- Use this as a small dynamic-aware increment, not as a strong dynamic-mainline success.",
                "",
                f"Frozen table: `performance/stage3_lambda_three_source/tables/{PREFIX}_frozen_table.md`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[Wrote] {table_path}")
    print(f"[Wrote] {PACKAGE_DIR}")


if __name__ == "__main__":
    main()

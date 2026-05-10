from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path(r"C:\Users\cyl\Desktop\data")
RUN_DIR = DATA_ROOT / "deltaA_signal_audit" / "solar96_stage31_target_quantile_gate"
ADAPTIVE_DIR = DATA_ROOT / "deltaA_signal_audit" / "solar96_existing_prediction_ensemble"
ADAPTIVE_PREFIX = "solar96_static_adaptive_alpha"
OUT_DIR = DATA_ROOT / "mechanism_evidence" / "solar96_mse_primary_target_gate_20260510"

RUNS = [
    {
        "route": "strict_target_gate",
        "variant": "static_p0",
        "prefix": "solar96_static_stage31_target_quantile_gate_valref",
        "policy": "Strict route: MSE/MAE guard plus fold stability; fallback allowed.",
    },
    {
        "route": "strict_target_gate",
        "variant": "static_mean",
        "prefix": "solar96_static_stage31_target_quantile_gate_staticmean_valref",
        "policy": "Strict route: MSE/MAE guard plus fold stability; fallback allowed.",
    },
    {
        "route": "mse_primary_target_gate",
        "variant": "static_p0",
        "prefix": "solar96_static_stage31_target_quantile_gate_valref_msefirst",
        "policy": "MSE-primary route: validation MSE first; MAE is an audit/non-degradation readout.",
    },
    {
        "route": "mse_primary_target_gate",
        "variant": "static_mean",
        "prefix": "solar96_static_stage31_target_quantile_gate_staticmean_valref_msefirst",
        "policy": "MSE-primary route: validation MSE first; MAE is an audit/non-degradation readout.",
    },
]


def read_one(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"Expected one row in {path}, got {len(frame)}")
    return frame.iloc[0].to_dict()


def pct_gain(before: float, after: float) -> float:
    if abs(float(before)) < 1e-12:
        return 0.0
    return 100.0 * (float(before) - float(after)) / float(before)


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


def fmt_float(value: float, digits: int = 6) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def fmt_pct(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):+.{digits}f}%"


def markdown_table(df: pd.DataFrame) -> str:
    columns = [
        "route",
        "variant",
        "selected_ensemble",
        "test_mse",
        "test_mae",
        "test_mse_gain_vs_adaptive_anchor_pct",
        "test_mae_gain_vs_adaptive_anchor_pct",
        "val_mse_gain_vs_adaptive_anchor_pct",
        "val_mae_gain_vs_adaptive_anchor_pct",
        "selection_reason",
    ]
    headers = [
        "route",
        "variant",
        "selected",
        "test MSE",
        "test MAE",
        "test MSE vs adaptive",
        "test MAE vs adaptive",
        "val MSE vs adaptive",
        "val MAE vs adaptive",
        "selection",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---"]) + " |",
    ]
    for _, row in df[columns].iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["route"]),
                    str(row["variant"]),
                    str(row["selected_ensemble"]),
                    fmt_float(row["test_mse"]),
                    fmt_float(row["test_mae"]),
                    fmt_pct(row["test_mse_gain_vs_adaptive_anchor_pct"]),
                    fmt_pct(row["test_mae_gain_vs_adaptive_anchor_pct"]),
                    fmt_pct(row["val_mse_gain_vs_adaptive_anchor_pct"]),
                    fmt_pct(row["val_mae_gain_vs_adaptive_anchor_pct"]),
                    str(row["selection_reason"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def control_rows(prefix: str, adaptive_test_mse: float, adaptive_test_mae: float) -> list[dict]:
    path = RUN_DIR / f"{prefix}_shuffle_controls.csv"
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    if frame.empty:
        return []
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "source_prefix": prefix,
                "split": s(row, "split"),
                "control_mode": s(row, "control_mode"),
                "shuffle_count": int(f(row, "shuffle_count", 0)),
                "selected_ensemble": s(row, "selected_ensemble"),
                "mse_median": f(row, "mse_median"),
                "mae_median": f(row, "mae_median"),
                "mse_gain_vs_adaptive_anchor_pct": pct_gain(adaptive_test_mse, f(row, "mse_median"))
                if s(row, "split") == "test"
                else np.nan,
                "mae_gain_vs_adaptive_anchor_pct": pct_gain(adaptive_test_mae, f(row, "mae_median"))
                if s(row, "split") == "test"
                else np.nan,
                "target_gate_active_ratio_median": f(row, "target_gate_active_ratio_median"),
            }
        )
    return rows


def normalized_selection_reason(run: dict, test: dict) -> str:
    if run["route"] == "mse_primary_target_gate" and s(test, "ensemble") != "stage2_anchor":
        return "best_val_mse_relaxed_mae_or_fold_guard"
    return s(test, "selection_reason")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adaptive = read_one(ADAPTIVE_DIR / f"{ADAPTIVE_PREFIX}_selected_test_summary.csv")
    adaptive_val_mse = f(adaptive, "val_mse")
    adaptive_val_mae = f(adaptive, "val_mae")
    adaptive_test_mse = f(adaptive, "test_mse")
    adaptive_test_mae = f(adaptive, "test_mae")

    rows = [
        {
            "route": "adaptive_anchor",
            "variant": "per_variable_shrinkage_alpha",
            "policy": "Prediction-level adaptive baseline/static anchor.",
            "selected_ensemble": s(adaptive, "ensemble"),
            "dynamic_source": "n/a",
            "threshold_scope": "n/a",
            "gamma_active_ratio": np.nan,
            "dynamic_active_ratio": np.nan,
            "eta_mult": 0.0,
            "eta_raw": 0.0,
            "eta_clip_reason": "anchor",
            "target_gate_active_ratio": 0.0,
            "val_mse": adaptive_val_mse,
            "val_mae": adaptive_val_mae,
            "test_mse": adaptive_test_mse,
            "test_mae": adaptive_test_mae,
            "val_mse_gain_vs_adaptive_anchor_pct": 0.0,
            "val_mae_gain_vs_adaptive_anchor_pct": 0.0,
            "test_mse_gain_vs_adaptive_anchor_pct": 0.0,
            "test_mae_gain_vs_adaptive_anchor_pct": 0.0,
            "selection_reason": s(adaptive, "selection_reason"),
        }
    ]
    controls = []
    raw_refs = []
    for run in RUNS:
        val_path = RUN_DIR / f"{run['prefix']}_selected_val_summary.csv"
        test_path = RUN_DIR / f"{run['prefix']}_test_selected_summary.csv"
        manifest_path = RUN_DIR / f"{run['prefix']}_manifest.json"
        val = read_one(val_path)
        test = read_one(test_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "route": run["route"],
                "variant": run["variant"],
                "policy": run["policy"],
                "selected_ensemble": s(test, "ensemble"),
                "dynamic_source": s(test, "dynamic_source"),
                "threshold_scope": s(manifest, "threshold_scope"),
                "gamma_active_ratio": f(test, "gamma_active_ratio"),
                "dynamic_active_ratio": f(test, "dynamic_active_ratio"),
                "eta_mult": f(test, "eta_mult"),
                "eta_raw": f(test, "eta_raw"),
                "eta_clip_reason": s(test, "eta_clip_reason"),
                "target_gate_active_ratio": f(test, "target_gate_active_ratio"),
                "val_mse": f(val, "mse"),
                "val_mae": f(val, "mae"),
                "test_mse": f(test, "mse"),
                "test_mae": f(test, "mae"),
                "val_mse_gain_vs_adaptive_anchor_pct": f(val, "mse_gain_vs_stage2_anchor_pct"),
                "val_mae_gain_vs_adaptive_anchor_pct": f(val, "mae_gain_vs_stage2_anchor_pct"),
                "test_mse_gain_vs_adaptive_anchor_pct": f(test, "mse_gain_vs_stage2_anchor_pct"),
                "test_mae_gain_vs_adaptive_anchor_pct": f(test, "mae_gain_vs_stage2_anchor_pct"),
                "selection_reason": normalized_selection_reason(run, test),
            }
        )
        controls.extend(control_rows(run["prefix"], adaptive_test_mse, adaptive_test_mae))
        raw_refs.append(
            {
                **run,
                "selected_val_summary": str(val_path),
                "test_selected_summary": str(test_path),
                "manifest": str(manifest_path),
                "shuffle_controls": str(RUN_DIR / f"{run['prefix']}_shuffle_controls.csv"),
            }
        )

    table = pd.DataFrame(rows)
    controls_df = pd.DataFrame(controls)
    table.to_csv(OUT_DIR / "solar96_mse_primary_target_gate_frozen_table.csv", index=False)
    (OUT_DIR / "solar96_mse_primary_target_gate_frozen_table.md").write_text(
        "# Solar-96 MSE-Primary Target-Gated Dynamic Route\n\n"
        + markdown_table(table)
        + "\n",
        encoding="utf-8",
    )
    controls_df.to_csv(OUT_DIR / "solar96_mse_primary_target_gate_controls.csv", index=False)

    mse_primary = table[
        (table["route"] == "mse_primary_target_gate") & (table["variant"] == "static_p0")
    ].iloc[0]
    strict = table[
        (table["route"] == "strict_target_gate") & (table["variant"] == "static_p0")
    ].iloc[0]
    test_controls = controls_df[
        (controls_df["source_prefix"] == "solar96_static_stage31_target_quantile_gate_valref_msefirst")
        & (controls_df["split"] == "test")
    ]
    control_lines = []
    for _, row in test_controls.iterrows():
        control_lines.append(
            f"- `{row['control_mode']}` median: `{fmt_float(row['mse_median'])} / {fmt_float(row['mae_median'])}`, "
            f"gain vs adaptive `{fmt_pct(row['mse_gain_vs_adaptive_anchor_pct'])} / "
            f"{fmt_pct(row['mae_gain_vs_adaptive_anchor_pct'])}`."
        )

    readme_lines = [
        "# Solar-96 MSE-Primary Target-Gated Dynamic Route",
        "",
        "Purpose:",
        "- Freeze `MSE-primary target-gated dynamic route` as a secondary Stage3 route, separate from the strict CACI double-guard route.",
        "- Preserve the main strict route while documenting an MSE-sensitive route for Solar-96 volatility/risk applications.",
        "",
        "Selection rule:",
        "- Validation MSE is the primary selector.",
        "- MAE is retained as an audit/non-degradation readout rather than a hard double guard.",
        "- Test is evaluated once for the validation-selected route.",
        "- Shuffle controls break gamma time alignment or target alignment to test whether the gain is route-specific.",
        "",
        "Key results:",
        (
            f"- Strict target gate selected `{strict['selected_ensemble']}` and therefore reports the adaptive anchor: "
            f"`{fmt_float(strict['test_mse'])} / {fmt_float(strict['test_mae'])}`."
        ),
        (
            f"- MSE-primary target gate selected `{mse_primary['selected_ensemble']}` "
            f"(`gamma_active_ratio={fmt_float(mse_primary['gamma_active_ratio'], 2)}`, "
            f"`dynamic_active_ratio={fmt_float(mse_primary['dynamic_active_ratio'], 2)}`), "
            f"test `{fmt_float(mse_primary['test_mse'])} / {fmt_float(mse_primary['test_mae'])}`, "
            f"gain vs adaptive `{fmt_pct(mse_primary['test_mse_gain_vs_adaptive_anchor_pct'])} / "
            f"{fmt_pct(mse_primary['test_mae_gain_vs_adaptive_anchor_pct'])}`."
        ),
        "",
        "Controls:",
        *control_lines,
        "",
        "Interpretation:",
        "- `MSE-primary` is not a replacement for strict CACI; it is a secondary route for MSE-sensitive settings.",
        "- The observed route beats both shuffle controls, but controls retain some positive gain, so the evidence supports weak target/gamma specificity plus sparse dynamic regularization rather than a pure causal-localization claim.",
        "",
        "Files:",
        "- `solar96_mse_primary_target_gate_frozen_table.csv/md`: frozen route table.",
        "- `solar96_mse_primary_target_gate_controls.csv`: shuffle control summary.",
        "- `manifest.json`: source outputs and raw run references.",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "artifact": "solar96_mse_primary_target_gate",
                "adaptive_summary": str(ADAPTIVE_DIR / f"{ADAPTIVE_PREFIX}_selected_test_summary.csv"),
                "run_dir": str(RUN_DIR),
                "raw_refs": raw_refs,
                "output_dir": str(OUT_DIR),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Done] wrote {OUT_DIR}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

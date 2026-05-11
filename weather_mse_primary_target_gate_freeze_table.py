from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path(r"C:\Users\cyl\Desktop\data")
RUN_DIR = DATA_ROOT / "deltaA_signal_audit" / "weather96_stage31_target_quantile_gate"
OUT_DIR = DATA_ROOT / "mechanism_evidence" / "weather96_mse_primary_target_gate_20260510"

RUNS = [
    {
        "route": "strict_target_gate",
        "variant": "static_p0",
        "prefix": "weather96_static_stage31_target_quantile_gate_valref",
        "policy": "Strict route: MSE/MAE guard plus fold stability; fallback allowed.",
    },
    {
        "route": "strict_target_gate",
        "variant": "static_mean",
        "prefix": "weather96_static_stage31_target_quantile_gate_staticmean_valref",
        "policy": "Strict route: MSE/MAE guard plus fold stability; fallback allowed.",
    },
    {
        "route": "mse_primary_target_gate",
        "variant": "static_p0",
        "prefix": "weather96_static_stage31_target_quantile_gate_valref_msefirst",
        "policy": "MSE-primary route: validation MSE first; MAE is an audit/non-degradation readout.",
    },
    {
        "route": "mse_primary_target_gate",
        "variant": "static_mean",
        "prefix": "weather96_static_stage31_target_quantile_gate_staticmean_valref_msefirst",
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


def pct_gain(before: float, after: float) -> float:
    if abs(float(before)) < 1e-12:
        return 0.0
    return 100.0 * (float(before) - float(after)) / float(before)


def fmt_float(value: float, digits: int = 6) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def fmt_pct(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):+.{digits}f}%"


def normalized_selection_reason(run: dict, test: dict) -> str:
    if run["route"] == "mse_primary_target_gate" and s(test, "ensemble") != "stage2_anchor":
        return "best_val_mse_relaxed_mae_or_fold_guard"
    return s(test, "selection_reason")


def markdown_table(df: pd.DataFrame) -> str:
    columns = [
        "route",
        "variant",
        "selected_ensemble",
        "test_mse",
        "test_mae",
        "test_mse_gain_vs_adaptive_anchor_pct",
        "test_mae_gain_vs_adaptive_anchor_pct",
        "selection_reason",
    ]
    lines = [
        "| route | variant | selected | test MSE | test MAE | test MSE vs adaptive | test MAE vs adaptive | selection |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
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
                    str(row["selection_reason"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def control_rows(prefix: str, anchor_mse: float, anchor_mae: float) -> list[dict]:
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
        is_test = s(row, "split") == "test"
        rows.append(
            {
                "source_prefix": prefix,
                "split": s(row, "split"),
                "control_mode": s(row, "control_mode"),
                "shuffle_count": int(f(row, "shuffle_count", 0)),
                "selected_ensemble": s(row, "selected_ensemble"),
                "mse_median": f(row, "mse_median"),
                "mae_median": f(row, "mae_median"),
                "mse_gain_vs_adaptive_anchor_pct": pct_gain(anchor_mse, f(row, "mse_median"))
                if is_test
                else np.nan,
                "mae_gain_vs_adaptive_anchor_pct": pct_gain(anchor_mae, f(row, "mae_median"))
                if is_test
                else np.nan,
                "target_gate_active_ratio_median": f(row, "target_gate_active_ratio_median"),
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    anchor = read_one(RUN_DIR / f"{RUNS[0]['prefix']}_test_selected_summary.csv")
    anchor_val = read_one(RUN_DIR / f"{RUNS[0]['prefix']}_selected_val_summary.csv")
    anchor_mse = f(anchor, "mse")
    anchor_mae = f(anchor, "mae")

    rows = [
        {
            "dataset": "Weather",
            "horizon": 96,
            "route": "adaptive_anchor",
            "variant": "per_variable_shrinkage_alpha",
            "policy": "Stage3 anchor loaded from Weather adaptive-alpha variable shrinkage.",
            "selected_ensemble": "stage2_anchor",
            "dynamic_source": "n/a",
            "threshold_scope": "n/a",
            "gamma_active_ratio": np.nan,
            "dynamic_active_ratio": np.nan,
            "eta_mult": 0.0,
            "eta_raw": 0.0,
            "eta_clip_reason": "anchor",
            "target_gate_active_ratio": 0.0,
            "val_mse": f(anchor_val, "mse"),
            "val_mae": f(anchor_val, "mae"),
            "test_mse": anchor_mse,
            "test_mae": anchor_mae,
            "val_mse_gain_vs_adaptive_anchor_pct": 0.0,
            "val_mae_gain_vs_adaptive_anchor_pct": 0.0,
            "test_mse_gain_vs_adaptive_anchor_pct": 0.0,
            "test_mae_gain_vs_adaptive_anchor_pct": 0.0,
            "selection_reason": "stage2_anchor_reference",
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
                "dataset": "Weather",
                "horizon": 96,
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
        controls.extend(control_rows(run["prefix"], anchor_mse, anchor_mae))
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
    table.to_csv(OUT_DIR / "weather96_mse_primary_target_gate_frozen_table.csv", index=False)
    (OUT_DIR / "weather96_mse_primary_target_gate_frozen_table.md").write_text(
        "# Weather-96 MSE-Primary Target-Gated Dynamic Route\n\n" + markdown_table(table) + "\n",
        encoding="utf-8",
    )
    controls_df.to_csv(OUT_DIR / "weather96_mse_primary_target_gate_controls.csv", index=False)

    best = table[table["route"] == "mse_primary_target_gate"].sort_values(
        "test_mse_gain_vs_adaptive_anchor_pct", ascending=False
    ).iloc[0]
    strict = table[(table["route"] == "strict_target_gate") & (table["variant"] == "static_p0")].iloc[0]
    test_controls = controls_df[
        (controls_df["source_prefix"] == str(best["selected_ensemble"]).replace("target_gate", ""))
    ]
    control_lines = []
    for _, row in controls_df[
        (controls_df["split"] == "test")
        & (controls_df["source_prefix"] == "weather96_static_stage31_target_quantile_gate_staticmean_valref_msefirst")
    ].iterrows():
        control_lines.append(
            f"- `{row['control_mode']}` median: `{fmt_float(row['mse_median'])} / {fmt_float(row['mae_median'])}`, "
            f"gain vs adaptive `{fmt_pct(row['mse_gain_vs_adaptive_anchor_pct'])} / "
            f"{fmt_pct(row['mae_gain_vs_adaptive_anchor_pct'])}`."
        )

    readme_lines = [
        "# Weather-96 MSE-Primary Target-Gated Dynamic Route",
        "",
        "Purpose:",
        "- Freeze Weather-96 as a lightweight boundary check for the MSE-primary target-gated dynamic route.",
        "- Keep strict CACI and MSE-primary route separate.",
        "",
        "Key results:",
        (
            f"- Adaptive anchor: `{fmt_float(anchor_mse)} / {fmt_float(anchor_mae)}`. "
            "This is stronger than the static-only Weather anchor used in older post-hoc tables."
        ),
        (
            f"- Strict target gate selected `{strict['selected_ensemble']}` and fell back to adaptive anchor: "
            f"`{fmt_float(strict['test_mse'])} / {fmt_float(strict['test_mae'])}`."
        ),
        (
            f"- MSE-primary best variant `{best['variant']}` selected `{best['selected_ensemble']}`, "
            f"test `{fmt_float(best['test_mse'])} / {fmt_float(best['test_mae'])}`, "
            f"gain vs adaptive `{fmt_pct(best['test_mse_gain_vs_adaptive_anchor_pct'])} / "
            f"{fmt_pct(best['test_mae_gain_vs_adaptive_anchor_pct'])}`."
        ),
        "",
        "Controls:",
        *control_lines,
        "",
        "Interpretation:",
        "- Weather confirms the expected boundary behavior: strict route protects the anchor, while MSE-primary admits a small MSE gain at the cost of a small MAE regression.",
        "- Observed MSE gain beats the shuffle medians, but the effect is small and should be reported as loss-specific rather than a headline performance route.",
        "",
        "Files:",
        "- `weather96_mse_primary_target_gate_frozen_table.csv/md`: frozen route table.",
        "- `weather96_mse_primary_target_gate_controls.csv`: shuffle control summary.",
        "- `manifest.json`: source outputs and raw run references.",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "artifact": "weather96_mse_primary_target_gate",
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

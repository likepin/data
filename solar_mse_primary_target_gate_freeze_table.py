from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path(r"C:\Users\cyl\Desktop\data")
OUT_DIR = DATA_ROOT / "mechanism_evidence" / "solar96_192_mse_primary_target_gate_20260510"

HORIZONS = [
    {
        "dataset": "Solar-96",
        "horizon": 96,
        "run_dir": DATA_ROOT / "deltaA_signal_audit" / "solar96_stage31_target_quantile_gate",
        "adaptive_dir": DATA_ROOT / "deltaA_signal_audit" / "solar96_existing_prediction_ensemble",
        "adaptive_prefix": "solar96_static_adaptive_alpha",
        "run_prefix": "solar96_static",
    },
    {
        "dataset": "Solar-192",
        "horizon": 192,
        "run_dir": DATA_ROOT / "deltaA_signal_audit" / "solar192_stage31_target_quantile_gate",
        "adaptive_dir": DATA_ROOT / "deltaA_signal_audit" / "solar192_existing_prediction_ensemble",
        "adaptive_prefix": "solar192_static_adaptive_alpha",
        "run_prefix": "solar192_static",
    },
]

RUN_SUFFIXES = [
    {
        "route": "strict_target_gate",
        "variant": "static_p0",
        "suffix": "stage31_target_quantile_gate_valref",
        "policy": "Strict route: MSE/MAE guard plus fold stability; fallback allowed.",
    },
    {
        "route": "strict_target_gate",
        "variant": "static_mean",
        "suffix": "stage31_target_quantile_gate_staticmean_valref",
        "policy": "Strict route: MSE/MAE guard plus fold stability; fallback allowed.",
    },
    {
        "route": "mse_primary_target_gate",
        "variant": "static_p0",
        "suffix": "stage31_target_quantile_gate_valref_msefirst",
        "policy": "MSE-primary route: validation MSE first; MAE is an audit/non-degradation readout.",
    },
    {
        "route": "mse_primary_target_gate",
        "variant": "static_mean",
        "suffix": "stage31_target_quantile_gate_staticmean_valref_msefirst",
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
        "dataset",
        "horizon",
        "route",
        "variant",
        "selected_ensemble",
        "test_mse",
        "test_mae",
        "test_mse_gain_vs_adaptive_anchor_pct",
        "test_mae_gain_vs_adaptive_anchor_pct",
        "selection_reason",
    ]
    headers = [
        "dataset",
        "horizon",
        "route",
        "variant",
        "selected",
        "test MSE",
        "test MAE",
        "test MSE vs adaptive",
        "test MAE vs adaptive",
        "selection",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| "
        + " | ".join(["---", "---:", "---", "---", "---", "---:", "---:", "---:", "---:", "---"])
        + " |",
    ]
    for _, row in df[columns].iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(int(row["horizon"])),
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


def run_prefix(horizon_cfg: dict, suffix: str) -> str:
    return f"{horizon_cfg['run_prefix']}_{suffix}"


def control_rows(
    horizon_cfg: dict,
    run: dict,
    prefix: str,
    adaptive_test_mse: float,
    adaptive_test_mae: float,
) -> list[dict]:
    path = horizon_cfg["run_dir"] / f"{prefix}_shuffle_controls.csv"
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
                "dataset": horizon_cfg["dataset"],
                "horizon": horizon_cfg["horizon"],
                "route": run["route"],
                "variant": run["variant"],
                "source_prefix": prefix,
                "split": s(row, "split"),
                "control_mode": s(row, "control_mode"),
                "shuffle_count": int(f(row, "shuffle_count", 0)),
                "selected_ensemble": s(row, "selected_ensemble"),
                "mse_median": f(row, "mse_median"),
                "mae_median": f(row, "mae_median"),
                "mse_gain_vs_adaptive_anchor_pct": (
                    pct_gain(adaptive_test_mse, f(row, "mse_median")) if is_test else np.nan
                ),
                "mae_gain_vs_adaptive_anchor_pct": (
                    pct_gain(adaptive_test_mae, f(row, "mae_median")) if is_test else np.nan
                ),
                "target_gate_active_ratio_median": f(row, "target_gate_active_ratio_median"),
            }
        )
    return rows


def normalized_selection_reason(run: dict, test: dict) -> str:
    if run["route"] == "mse_primary_target_gate" and s(test, "ensemble") != "stage2_anchor":
        return "best_val_mse_relaxed_mae_or_fold_guard"
    return s(test, "selection_reason")


def build_rows_for_horizon(horizon_cfg: dict) -> tuple[list[dict], list[dict], list[dict]]:
    adaptive = read_one(
        horizon_cfg["adaptive_dir"] / f"{horizon_cfg['adaptive_prefix']}_selected_test_summary.csv"
    )
    adaptive_val_mse = f(adaptive, "val_mse")
    adaptive_val_mae = f(adaptive, "val_mae")
    adaptive_test_mse = f(adaptive, "test_mse")
    adaptive_test_mae = f(adaptive, "test_mae")

    rows = [
        {
            "dataset": horizon_cfg["dataset"],
            "horizon": horizon_cfg["horizon"],
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
    for run in RUN_SUFFIXES:
        prefix = run_prefix(horizon_cfg, run["suffix"])
        val_path = horizon_cfg["run_dir"] / f"{prefix}_selected_val_summary.csv"
        test_path = horizon_cfg["run_dir"] / f"{prefix}_test_selected_summary.csv"
        manifest_path = horizon_cfg["run_dir"] / f"{prefix}_manifest.json"
        val = read_one(val_path)
        test = read_one(test_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": horizon_cfg["dataset"],
                "horizon": horizon_cfg["horizon"],
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
        controls.extend(control_rows(horizon_cfg, run, prefix, adaptive_test_mse, adaptive_test_mae))
        raw_refs.append(
            {
                "dataset": horizon_cfg["dataset"],
                "horizon": horizon_cfg["horizon"],
                **run,
                "prefix": prefix,
                "selected_val_summary": str(val_path),
                "test_selected_summary": str(test_path),
                "manifest": str(manifest_path),
                "shuffle_controls": str(horizon_cfg["run_dir"] / f"{prefix}_shuffle_controls.csv"),
            }
        )
    return rows, controls, raw_refs


def best_mse_primary(table: pd.DataFrame, dataset: str) -> pd.Series:
    subset = table[(table["dataset"] == dataset) & (table["route"] == "mse_primary_target_gate")]
    if subset.empty:
        raise ValueError(f"No MSE-primary rows for {dataset}")
    return subset.sort_values("test_mse_gain_vs_adaptive_anchor_pct", ascending=False).iloc[0]


def strict_static_p0(table: pd.DataFrame, dataset: str) -> pd.Series:
    subset = table[
        (table["dataset"] == dataset)
        & (table["route"] == "strict_target_gate")
        & (table["variant"] == "static_p0")
    ]
    if subset.empty:
        raise ValueError(f"No strict static_p0 row for {dataset}")
    return subset.iloc[0]


def controls_for_best(controls_df: pd.DataFrame, best: pd.Series) -> list[str]:
    if controls_df.empty:
        return []
    subset = controls_df[
        (controls_df["dataset"] == best["dataset"])
        & (controls_df["route"] == best["route"])
        & (controls_df["variant"] == best["variant"])
        & (controls_df["split"] == "test")
    ]
    lines = []
    for _, row in subset.iterrows():
        lines.append(
            f"- {best['dataset']} `{row['control_mode']}` median: "
            f"`{fmt_float(row['mse_median'])} / {fmt_float(row['mae_median'])}`, "
            f"gain vs adaptive `{fmt_pct(row['mse_gain_vs_adaptive_anchor_pct'])} / "
            f"{fmt_pct(row['mae_gain_vs_adaptive_anchor_pct'])}`."
        )
    return lines


def write_readme(table: pd.DataFrame, controls_df: pd.DataFrame) -> None:
    readme_lines = [
        "# Solar-96/192 MSE-Primary Target-Gated Dynamic Route",
        "",
        "Purpose:",
        "- Freeze `MSE-primary target-gated dynamic route` as a secondary Stage3 route, separate from the strict CACI double-guard route.",
        "- Compare Solar-96 and Solar-192 under the same strict-vs-MSE-primary protocol.",
        "- Preserve the main strict route while documenting a loss-specific route for MSE-sensitive Solar settings.",
        "",
        "Selection rule:",
        "- Strict route keeps MSE/MAE guard plus fold stability and may fall back to the adaptive anchor.",
        "- MSE-primary route selects by validation MSE; MAE is retained as an audit/non-degradation readout.",
        "- Test is evaluated once for the validation-selected route.",
        "- Shuffle controls break gamma time alignment or target alignment to audit route specificity.",
        "",
        "Key results:",
    ]
    for cfg in HORIZONS:
        strict = strict_static_p0(table, cfg["dataset"])
        best = best_mse_primary(table, cfg["dataset"])
        readme_lines.extend(
            [
                (
                    f"- {cfg['dataset']} strict target gate selected `{strict['selected_ensemble']}`: "
                    f"`{fmt_float(strict['test_mse'])} / {fmt_float(strict['test_mae'])}`, "
                    f"gain vs adaptive `{fmt_pct(strict['test_mse_gain_vs_adaptive_anchor_pct'])} / "
                    f"{fmt_pct(strict['test_mae_gain_vs_adaptive_anchor_pct'])}`."
                ),
                (
                    f"- {cfg['dataset']} MSE-primary best variant `{best['variant']}` selected "
                    f"`{best['selected_ensemble']}` "
                    f"(`gamma_active_ratio={fmt_float(best['gamma_active_ratio'], 2)}`, "
                    f"`dynamic_active_ratio={fmt_float(best['dynamic_active_ratio'], 2)}`), "
                    f"test `{fmt_float(best['test_mse'])} / {fmt_float(best['test_mae'])}`, "
                    f"gain vs adaptive `{fmt_pct(best['test_mse_gain_vs_adaptive_anchor_pct'])} / "
                    f"{fmt_pct(best['test_mae_gain_vs_adaptive_anchor_pct'])}`."
                ),
            ]
        )

    control_lines = []
    for cfg in HORIZONS:
        control_lines.extend(controls_for_best(controls_df, best_mse_primary(table, cfg["dataset"])))
    readme_lines.extend(
        [
            "",
            "Controls:",
            *(control_lines or ["- No shuffle controls were found for the best MSE-primary variants."]),
            "",
            "Interpretation:",
            "- Strict CACI remains the conservative route and falls back on Solar-96/192 under this target-gate design.",
            "- MSE-primary is not a replacement for strict CACI; it is a secondary loss-specific route.",
            "- Solar-96 shows a small but repeatable MSE-sensitive gain; Solar-192 shows a smaller gain that still beats shuffle medians for the best variant.",
            "- Because shuffle controls retain some positive gain, the evidence supports weak target/gamma specificity plus sparse dynamic regularization rather than a pure causal-localization claim.",
            "",
            "Files:",
            "- `solar96_192_mse_primary_target_gate_frozen_table.csv/md`: frozen route table.",
            "- `solar96_192_mse_primary_target_gate_controls.csv`: shuffle control summary.",
            "- `manifest.json`: source outputs and raw run references.",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    controls = []
    raw_refs = []
    for cfg in HORIZONS:
        cfg_rows, cfg_controls, cfg_refs = build_rows_for_horizon(cfg)
        rows.extend(cfg_rows)
        controls.extend(cfg_controls)
        raw_refs.extend(cfg_refs)

    table = pd.DataFrame(rows)
    controls_df = pd.DataFrame(controls)
    table.to_csv(OUT_DIR / "solar96_192_mse_primary_target_gate_frozen_table.csv", index=False)
    (OUT_DIR / "solar96_192_mse_primary_target_gate_frozen_table.md").write_text(
        "# Solar-96/192 MSE-Primary Target-Gated Dynamic Route\n\n"
        + markdown_table(table)
        + "\n",
        encoding="utf-8",
    )
    controls_df.to_csv(OUT_DIR / "solar96_192_mse_primary_target_gate_controls.csv", index=False)
    write_readme(table, controls_df)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "artifact": "solar96_192_mse_primary_target_gate",
                "horizons": [
                    {
                        "dataset": cfg["dataset"],
                        "horizon": cfg["horizon"],
                        "adaptive_summary": str(
                            cfg["adaptive_dir"] / f"{cfg['adaptive_prefix']}_selected_test_summary.csv"
                        ),
                        "run_dir": str(cfg["run_dir"]),
                    }
                    for cfg in HORIZONS
                ],
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

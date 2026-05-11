from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\cyl\Desktop\data")
OUT_DIR = ROOT / "mechanism_evidence" / "current_frozen_tables_20260511"

CROSS_FULL = ROOT / "mechanism_evidence" / "cross_dataset_route_ablation_20260507" / "cross_dataset_route_ablation_full.csv"
WEATHER_PAT3 = ROOT / "deltaA_signal_audit" / "weather96_pat3_summary" / "weather96_pat3_adaptive_posthoc_summary.csv"

GUARDED_SUMMARIES = {
    "ETTh1": ROOT
    / "deltaA_signal_audit"
    / "etth196_closed_loop_rank_quality_guard_parcorr_ridgebase_sparse"
    / "etth196_static_parcorr_rank_quality_guard_parcorr_closed_loop_test_selected_summary.csv",
    "ECL": ROOT
    / "deltaA_signal_audit"
    / "ecl96_closed_loop_static"
    / "ecl96_static_full_guard_v2_closed_loop_test_selected_summary.csv",
    "Solar-96": ROOT
    / "deltaA_signal_audit"
    / "solar96_closed_loop"
    / "solar96_static_closed_loop_test_selected_summary.csv",
    "Solar-192": ROOT
    / "deltaA_signal_audit"
    / "solar192_closed_loop_rank_quality_guard"
    / "solar192_static_rank_quality_guard_closed_loop_test_selected_summary.csv",
    "Traffic": ROOT
    / "deltaA_signal_audit"
    / "traffic96_closed_loop_log_tail_quality_guard"
    / "traffic96_static_log_tail_quality_guard_closed_loop_test_selected_summary.csv",
}

STAGE3_TABLES = {
    "ETTh1": ROOT
    / "mechanism_evidence"
    / "etth196_stage3_lambda_three_source_20260509"
    / "etth196_stage3_lambda_three_source_frozen_table.csv",
    "Solar": ROOT
    / "mechanism_evidence"
    / "solar96_192_stage3_lambda_three_source_20260508"
    / "solar_stage3_lambda_three_source_frozen_table.csv",
    "Traffic": ROOT
    / "mechanism_evidence"
    / "traffic96_stage3_lambda_three_source_20260507"
    / "performance"
    / "stage3_lambda_three_source"
    / "tables"
    / "traffic96_static_stage3_lambda_three_source_frozen_table.csv",
}

MSE_PRIMARY_TABLES = {
    "Weather": ROOT
    / "mechanism_evidence"
    / "weather96_mse_primary_target_gate_20260510"
    / "weather96_mse_primary_target_gate_frozen_table.csv",
    "Solar": ROOT
    / "mechanism_evidence"
    / "solar96_192_mse_primary_target_gate_20260510"
    / "solar96_192_mse_primary_target_gate_frozen_table.csv",
}

DATASET_ORDER = ["ETTh1", "Weather", "ECL", "Solar-96", "Solar-192", "Traffic"]
HORIZONS = {
    "ETTh1": 96,
    "Weather": 96,
    "ECL": 96,
    "Solar-96": 96,
    "Solar-192": 192,
    "Traffic": 96,
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty CSV: {path}")
    return df


def first_row(path: Path) -> dict:
    return read_csv(path).iloc[0].to_dict()


def as_float(value: object, default: float = np.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def first_present(row: dict, *keys: str) -> object:
    for key in keys:
        if key not in row:
            continue
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            if isinstance(value, float) and np.isnan(value):
                continue
        except TypeError:
            pass
        return value
    return None


def pct_gain(old: float, new: float) -> float:
    if not np.isfinite(old) or old == 0 or not np.isfinite(new):
        return np.nan
    return (old - new) / old * 100.0


def metric_pair(mse: float, mae: float) -> str:
    if not np.isfinite(mse) or not np.isfinite(mae):
        return ""
    return f"{mse:.6f} / {mae:.6f}"


def gain_pair(mse_gain: float, mae_gain: float) -> str:
    if not np.isfinite(mse_gain) or not np.isfinite(mae_gain):
        return ""
    return f"{mse_gain:+.3f}% / {mae_gain:+.3f}%"


def mode_from_guarded(row: dict) -> str:
    mode_status = str(row.get("mode_status", "") or "")
    if mode_status:
        return mode_status
    if str(row.get("selection_reason", "")) == "fallback_static_only":
        return "Bypass"
    active_ratio = as_float(row.get("active_ratio"))
    mse_gain = as_float(row.get("mse_gain_pct"))
    if active_ratio > 0 and mse_gain >= 0:
        return "Selective"
    if active_ratio > 0:
        return "Selective_weak"
    return "Diagnostic"


def markdown_table(df: pd.DataFrame) -> str:
    def fmt(value: object) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value)

    lines = ["| " + " | ".join(df.columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in df.columns) + " |")
    return "\n".join(lines) + "\n"


def git_short_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def base_route_rows() -> dict[str, dict]:
    cross = read_csv(CROSS_FULL)
    rows = {}
    for dataset in DATASET_ORDER:
        match = cross.loc[cross["dataset"].astype(str) == dataset]
        if match.empty:
            raise ValueError(f"Missing dataset in cross table: {dataset}")
        rows[dataset] = match.iloc[0].to_dict()
    return rows


def apply_weather_pat3(row: dict) -> dict:
    weather = read_csv(WEATHER_PAT3)
    by_route = {str(item["route"]): item for _, item in weather.iterrows()}
    baseline = by_route["iTransformer baseline mean"]
    static = by_route["Static causal anchor mean"]
    adaptive = by_route["Adaptive fusion headline"]
    guarded = by_route["Guarded post-hoc closed loop"]

    row.update(
        {
            "baseline_mse": as_float(baseline["mse"]),
            "baseline_mae": as_float(baseline["mae"]),
            "static_anchor_mse": as_float(static["mse"]),
            "static_anchor_mae": as_float(static["mae"]),
            "static_mse_gain_vs_baseline_pct": as_float(static["mse_gain_vs_baseline_pct"]),
            "static_mae_gain_vs_baseline_pct": as_float(static["mae_gain_vs_baseline_pct"]),
            "guarded_posthoc_mode": str(guarded["mode_status"]),
            "guarded_posthoc_selection_reason": str(guarded["selection_reason"]),
            "guarded_posthoc_active_ratio": as_float(guarded["active_ratio"]),
            "guarded_posthoc_static_mse": as_float(static["mse"]),
            "guarded_posthoc_static_mae": as_float(static["mae"]),
            "guarded_posthoc_mse": as_float(guarded["mse"]),
            "guarded_posthoc_mae": as_float(guarded["mae"]),
            "guarded_posthoc_mse_gain_vs_static_pct": as_float(guarded["mse_gain_vs_static_pct"]),
            "guarded_posthoc_mae_gain_vs_static_pct": as_float(guarded["mae_gain_vs_static_pct"]),
            "adaptive_fusion_mse": as_float(adaptive["mse"]),
            "adaptive_fusion_mae": as_float(adaptive["mae"]),
            "adaptive_fusion_mse_gain_vs_static_anchor_pct": as_float(
                adaptive["mse_gain_vs_static_pct"]
            ),
            "adaptive_fusion_mae_gain_vs_static_anchor_pct": as_float(
                adaptive["mae_gain_vs_static_pct"]
            ),
            "stage3_dynamic_mse": np.nan,
            "stage3_dynamic_mae": np.nan,
            "stage3_dynamic_mse_gain_vs_adaptive_pct": np.nan,
            "stage3_dynamic_mae_gain_vs_adaptive_pct": np.nan,
            "final_headline_route": "Adaptive fusion headline; guarded post-hoc dynamic is Selective but tiny-positive over static.",
        }
    )
    return row


def apply_guarded_summary(dataset: str, row: dict) -> dict:
    if dataset == "Weather":
        return row
    guarded = first_row(GUARDED_SUMMARIES[dataset])
    mode = mode_from_guarded(guarded)
    row.update(
        {
            "guarded_posthoc_mode": mode,
            "guarded_posthoc_selection_reason": str(guarded.get("selection_reason", "")),
            "guarded_posthoc_active_ratio": as_float(guarded.get("active_ratio")),
            "guarded_posthoc_static_mse": as_float(guarded.get("static_mse")),
            "guarded_posthoc_static_mae": as_float(guarded.get("static_mae")),
            "guarded_posthoc_mse": as_float(guarded.get("posthoc_mse")),
            "guarded_posthoc_mae": as_float(guarded.get("posthoc_mae")),
            "guarded_posthoc_mse_gain_vs_static_pct": as_float(guarded.get("mse_gain_pct")),
            "guarded_posthoc_mae_gain_vs_static_pct": as_float(guarded.get("mae_gain_pct")),
        }
    )
    return row


def stage3_row(dataset: str) -> dict | None:
    if dataset == "ETTh1":
        df = read_csv(STAGE3_TABLES["ETTh1"])
        match = df[
            (df["label"].astype(str) == "Stage3 closed-form eta2")
            & (df["variant"].astype(str) == "static_p0_dynamic")
        ]
    elif dataset.startswith("Solar-"):
        df = read_csv(STAGE3_TABLES["Solar"])
        match = df[
            (df["horizon"].astype(str) == str(HORIZONS[dataset]))
            & (df["label"].astype(str) == "Stage3 closed-form eta2")
            & (df["variant"].astype(str) == "static_p0_dynamic")
        ]
    elif dataset == "Traffic":
        df = read_csv(STAGE3_TABLES["Traffic"])
        match = df[df["label"].astype(str) == "Stage3 lambda three-source, closed-form eta2"]
    else:
        return None

    if match.empty:
        raise ValueError(f"Missing Stage3 row for {dataset}")
    return match.iloc[0].to_dict()


def apply_stage3(dataset: str, row: dict) -> dict:
    stage3 = stage3_row(dataset)
    if stage3 is None:
        row.update(
            {
                "stage3_dynamic_mse": np.nan,
                "stage3_dynamic_mae": np.nan,
                "stage3_dynamic_mse_gain_vs_adaptive_pct": np.nan,
                "stage3_dynamic_mae_gain_vs_adaptive_pct": np.nan,
            }
        )
        return row

    row.update(
        {
            "stage3_dynamic_mse": as_float(stage3.get("test_mse")),
            "stage3_dynamic_mae": as_float(stage3.get("test_mae")),
            "stage3_dynamic_mse_gain_vs_adaptive_pct": as_float(
                first_present(
                    stage3,
                    "test_mse_gain_vs_adaptive_anchor_pct",
                    "test_mse_gain_vs_stage2_anchor_pct",
                )
            ),
            "stage3_dynamic_mae_gain_vs_adaptive_pct": as_float(
                first_present(
                    stage3,
                    "test_mae_gain_vs_adaptive_anchor_pct",
                    "test_mae_gain_vs_stage2_anchor_pct",
                )
            ),
        }
    )
    return row


def current_route_table() -> pd.DataFrame:
    rows = base_route_rows()
    out = []
    for dataset in DATASET_ORDER:
        row = dict(rows[dataset])
        if dataset == "Weather":
            row = apply_weather_pat3(row)
        row = apply_guarded_summary(dataset, row)
        row = apply_stage3(dataset, row)
        row["horizon"] = HORIZONS[dataset]
        out.append(row)
    cols = [
        "dataset",
        "horizon",
        "baseline_mse",
        "baseline_mae",
        "static_anchor_mse",
        "static_anchor_mae",
        "static_mse_gain_vs_baseline_pct",
        "static_mae_gain_vs_baseline_pct",
        "guarded_posthoc_mode",
        "guarded_posthoc_active_ratio",
        "guarded_posthoc_mse",
        "guarded_posthoc_mae",
        "guarded_posthoc_mse_gain_vs_static_pct",
        "guarded_posthoc_mae_gain_vs_static_pct",
        "adaptive_fusion_mse",
        "adaptive_fusion_mae",
        "adaptive_fusion_mse_gain_vs_static_anchor_pct",
        "adaptive_fusion_mae_gain_vs_static_anchor_pct",
        "stage3_dynamic_mse",
        "stage3_dynamic_mae",
        "stage3_dynamic_mse_gain_vs_adaptive_pct",
        "stage3_dynamic_mae_gain_vs_adaptive_pct",
        "final_headline_route",
    ]
    return pd.DataFrame(out)[cols]


def paper_route_table(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in full.iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "horizon": int(row["horizon"]),
                "baseline": metric_pair(as_float(row["baseline_mse"]), as_float(row["baseline_mae"])),
                "static_anchor": metric_pair(
                    as_float(row["static_anchor_mse"]), as_float(row["static_anchor_mae"])
                ),
                "static_gain_vs_baseline": gain_pair(
                    as_float(row["static_mse_gain_vs_baseline_pct"]),
                    as_float(row["static_mae_gain_vs_baseline_pct"]),
                ),
                "guarded_posthoc": metric_pair(
                    as_float(row["guarded_posthoc_mse"]), as_float(row["guarded_posthoc_mae"])
                ),
                "guarded_gain_vs_static": gain_pair(
                    as_float(row["guarded_posthoc_mse_gain_vs_static_pct"]),
                    as_float(row["guarded_posthoc_mae_gain_vs_static_pct"]),
                ),
                "guarded_mode": row["guarded_posthoc_mode"],
                "adaptive_fusion": metric_pair(
                    as_float(row["adaptive_fusion_mse"]), as_float(row["adaptive_fusion_mae"])
                ),
                "stage3_gain_vs_adaptive": gain_pair(
                    as_float(row["stage3_dynamic_mse_gain_vs_adaptive_pct"]),
                    as_float(row["stage3_dynamic_mae_gain_vs_adaptive_pct"]),
                ),
                "paper_reading": row["final_headline_route"],
            }
        )
    return pd.DataFrame(rows)


def guarded_dynamic_rows() -> list[dict]:
    rows = []
    for dataset in DATASET_ORDER:
        if dataset == "Weather":
            weather = read_csv(WEATHER_PAT3)
            guarded = weather.loc[weather["route"].astype(str) == "Guarded post-hoc closed loop"].iloc[0]
            rows.append(
                {
                    "dataset": "Weather",
                    "horizon": 96,
                    "route_family": "guarded_posthoc_dynamic",
                    "reference": "static_anchor_patience3",
                    "mode": str(guarded["mode_status"]),
                    "active_ratio": as_float(guarded["active_ratio"]),
                    "test_mse": as_float(guarded["mse"]),
                    "test_mae": as_float(guarded["mae"]),
                    "mse_gain_pct": as_float(guarded["mse_gain_vs_static_pct"]),
                    "mae_gain_pct": as_float(guarded["mae_gain_vs_static_pct"]),
                    "selection_reason": str(guarded["selection_reason"]),
                    "paper_status": "tiny_positive_guarded_dynamic",
                }
            )
            continue

        guarded = first_row(GUARDED_SUMMARIES[dataset])
        mse_gain = as_float(guarded.get("mse_gain_pct"))
        mae_gain = as_float(guarded.get("mae_gain_pct"))
        if mse_gain == 0 and mae_gain == 0:
            status = "bypass_or_neutral"
        elif mse_gain > 0 and mae_gain > 0:
            status = "positive_but_small"
        elif mse_gain > 0 or mae_gain > 0:
            status = "mixed_metric"
        else:
            status = "negative"
        rows.append(
            {
                "dataset": dataset,
                "horizon": HORIZONS[dataset],
                "route_family": "guarded_posthoc_dynamic",
                "reference": "static_anchor",
                "mode": mode_from_guarded(guarded),
                "active_ratio": as_float(guarded.get("active_ratio")),
                "test_mse": as_float(guarded.get("posthoc_mse")),
                "test_mae": as_float(guarded.get("posthoc_mae")),
                "mse_gain_pct": mse_gain,
                "mae_gain_pct": mae_gain,
                "selection_reason": str(guarded.get("selection_reason", "")),
                "paper_status": status,
            }
        )
    return rows


def stage3_dynamic_rows() -> list[dict]:
    rows = []
    for dataset in ["ETTh1", "Solar-96", "Solar-192", "Traffic"]:
        stage3 = stage3_row(dataset)
        assert stage3 is not None
        mse_gain = as_float(
            first_present(
                stage3,
                "test_mse_gain_vs_adaptive_anchor_pct",
                "test_mse_gain_vs_stage2_anchor_pct",
            )
        )
        mae_gain = as_float(
            first_present(
                stage3,
                "test_mae_gain_vs_adaptive_anchor_pct",
                "test_mae_gain_vs_stage2_anchor_pct",
            )
        )
        if mse_gain > 0 and mae_gain > 0:
            status = "weak_positive_addon"
        elif mse_gain == 0 and mae_gain == 0:
            status = "fallback_to_anchor"
        else:
            status = "negative_addon"
        rows.append(
            {
                "dataset": dataset,
                "horizon": HORIZONS[dataset],
                "route_family": "stage3_lambda_three_source",
                "reference": "adaptive_anchor",
                "mode": str(first_present(stage3, "selected_ensemble", "label") or ""),
                "active_ratio": np.nan,
                "test_mse": as_float(stage3.get("test_mse")),
                "test_mae": as_float(stage3.get("test_mae")),
                "mse_gain_pct": mse_gain,
                "mae_gain_pct": mae_gain,
                "selection_reason": str(stage3.get("selection_reason", "")),
                "paper_status": status,
            }
        )
    return rows


def mse_primary_rows() -> list[dict]:
    rows = []
    weather = read_csv(MSE_PRIMARY_TABLES["Weather"])
    weather = weather.loc[weather["route"].astype(str) == "mse_primary_target_gate"].copy()
    weather["test_mse_gain_vs_adaptive_anchor_pct"] = pd.to_numeric(
        weather["test_mse_gain_vs_adaptive_anchor_pct"], errors="coerce"
    )
    best_weather = weather.sort_values("test_mse_gain_vs_adaptive_anchor_pct", ascending=False).iloc[0]
    rows.append(
        {
            "dataset": "Weather",
            "horizon": 96,
            "route_family": "mse_primary_target_gate",
            "reference": "audit_adaptive_anchor_not_pat3_headline",
            "mode": str(best_weather["selected_ensemble"]),
            "active_ratio": as_float(best_weather["dynamic_active_ratio"]),
            "test_mse": as_float(best_weather["test_mse"]),
            "test_mae": as_float(best_weather["test_mae"]),
            "mse_gain_pct": as_float(best_weather["test_mse_gain_vs_adaptive_anchor_pct"]),
            "mae_gain_pct": as_float(best_weather["test_mae_gain_vs_adaptive_anchor_pct"]),
            "selection_reason": str(best_weather["selection_reason"]),
            "paper_status": "audit_only_mse_positive_mae_negative",
        }
    )

    solar = read_csv(MSE_PRIMARY_TABLES["Solar"])
    solar = solar.loc[solar["route"].astype(str) == "mse_primary_target_gate"].copy()
    solar["test_mse_gain_vs_adaptive_anchor_pct"] = pd.to_numeric(
        solar["test_mse_gain_vs_adaptive_anchor_pct"], errors="coerce"
    )
    for horizon in [96, 192]:
        match = solar.loc[solar["horizon"].astype(str) == str(horizon)]
        best = match.sort_values("test_mse_gain_vs_adaptive_anchor_pct", ascending=False).iloc[0]
        rows.append(
            {
                "dataset": f"Solar-{horizon}",
                "horizon": horizon,
                "route_family": "mse_primary_target_gate",
                "reference": "adaptive_anchor",
                "mode": str(best["selected_ensemble"]),
                "active_ratio": as_float(best["dynamic_active_ratio"]),
                "test_mse": as_float(best["test_mse"]),
                "test_mae": as_float(best["test_mae"]),
                "mse_gain_pct": as_float(best["test_mse_gain_vs_adaptive_anchor_pct"]),
                "mae_gain_pct": as_float(best["test_mae_gain_vs_adaptive_anchor_pct"]),
                "selection_reason": str(best["selection_reason"]),
                "paper_status": "mse_primary_positive_audit",
            }
        )
    return rows


def dynamic_increment_table() -> pd.DataFrame:
    rows = guarded_dynamic_rows()
    rows.extend(stage3_dynamic_rows())
    rows.extend(mse_primary_rows())
    cols = [
        "dataset",
        "horizon",
        "route_family",
        "reference",
        "mode",
        "active_ratio",
        "test_mse",
        "test_mae",
        "mse_gain_pct",
        "mae_gain_pct",
        "selection_reason",
        "paper_status",
    ]
    return pd.DataFrame(rows)[cols]


def paper_dynamic_table(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in full.iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "horizon": int(row["horizon"]),
                "route_family": row["route_family"],
                "reference": row["reference"],
                "mode": row["mode"],
                "gain_mse_mae": gain_pair(as_float(row["mse_gain_pct"]), as_float(row["mae_gain_pct"])),
                "active_ratio": (
                    "" if not np.isfinite(as_float(row["active_ratio"])) else f"{as_float(row['active_ratio']):.4f}"
                ),
                "paper_status": row["paper_status"],
            }
        )
    return pd.DataFrame(rows)


def write_readme(route: pd.DataFrame, dynamic: pd.DataFrame) -> None:
    text = f"""# Current Frozen CACI Tables 2026-05-11

This package freezes the current route-separated CACI result tables after the Weather/Solar dynamic-gate adequacy diagnostics.

## Scope

- Static and adaptive-fusion numbers are performance routes.
- Guarded post-hoc dynamic numbers are measured against the static anchor.
- Stage3 / target-gated dynamic numbers are measured against the adaptive anchor when available.
- Weather uses the paper-aligned `patience=3` summary as the canonical route table source.
- Weather MSE-primary target-gate is retained only as an audit row because its adaptive-anchor source is not the latest `patience=3` headline table.

## Main Readout

- The strongest current performance source is `adaptive_fusion`, not the dynamic graph.
- Guarded post-hoc dynamic gains are usually `0%` to `0.3%`.
- Stage3 lambda-aware add-ons are weak-positive on Traffic/Solar-96, fallback on Solar-192, and negative on ETTh1.
- The safe paper claim is route-separated: static/adaptive routes are performance routes; lambda/dynamic routes are guarded, optional, and dataset-sensitive.

## Files

- `current_route_table_full.csv/md`: numeric route-separated table.
- `current_route_table_paper.csv/md`: readable paper-facing route table.
- `dynamic_lambda_increment_full.csv/md`: dynamic/lambda increment and boundary table.
- `dynamic_lambda_increment_paper.csv/md`: readable dynamic/lambda boundary table.
- `manifest.json`: source file paths and git commit used for this freeze.

## Quick Dynamic Summary

Canonical guarded post-hoc MSE gains vs static anchor:

{markdown_table(dynamic.loc[dynamic["route_family"] == "guarded_posthoc_dynamic", ["dataset", "horizon", "mode", "mse_gain_pct", "mae_gain_pct", "paper_status"]])}

Stage3 MSE gains vs adaptive anchor:

{markdown_table(dynamic.loc[dynamic["route_family"] == "stage3_lambda_three_source", ["dataset", "horizon", "mode", "mse_gain_pct", "mae_gain_pct", "paper_status"]])}
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    route = current_route_table()
    route_paper = paper_route_table(route)
    dynamic = dynamic_increment_table()
    dynamic_paper = paper_dynamic_table(dynamic)

    route.to_csv(OUT_DIR / "current_route_table_full.csv", index=False)
    route_paper.to_csv(OUT_DIR / "current_route_table_paper.csv", index=False)
    dynamic.to_csv(OUT_DIR / "dynamic_lambda_increment_full.csv", index=False)
    dynamic_paper.to_csv(OUT_DIR / "dynamic_lambda_increment_paper.csv", index=False)

    (OUT_DIR / "current_route_table_full.md").write_text(markdown_table(route), encoding="utf-8")
    (OUT_DIR / "current_route_table_paper.md").write_text(markdown_table(route_paper), encoding="utf-8")
    (OUT_DIR / "dynamic_lambda_increment_full.md").write_text(markdown_table(dynamic), encoding="utf-8")
    (OUT_DIR / "dynamic_lambda_increment_paper.md").write_text(
        markdown_table(dynamic_paper), encoding="utf-8"
    )

    manifest = {
        "artifact": "current_frozen_caci_tables",
        "date": "2026-05-11",
        "git_head": git_short_head(),
        "output_dir": str(OUT_DIR),
        "sources": {
            "cross_dataset_route_ablation_full": str(CROSS_FULL),
            "weather_pat3_summary": str(WEATHER_PAT3),
            "guarded_summaries": {key: str(value) for key, value in GUARDED_SUMMARIES.items()},
            "stage3_tables": {key: str(value) for key, value in STAGE3_TABLES.items()},
            "mse_primary_tables": {key: str(value) for key, value in MSE_PRIMARY_TABLES.items()},
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(route, dynamic)

    print(f"[Wrote] {OUT_DIR}")
    print(route_paper.to_string(index=False))
    print()
    print(dynamic_paper.to_string(index=False))


if __name__ == "__main__":
    main()

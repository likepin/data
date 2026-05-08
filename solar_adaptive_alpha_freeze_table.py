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
OUT_DIR = DATA_ROOT / "mechanism_evidence" / "solar96_192_adaptive_alpha_20260508"

RUNS = {
    96: {
        "adaptive_dir": DATA_ROOT / "deltaA_signal_audit" / "solar96_existing_prediction_ensemble",
        "prefix": "solar96_static_adaptive_alpha",
        "posthoc_test": DATA_ROOT
        / "deltaA_signal_audit"
        / "solar96_closed_loop"
        / "solar96_static_closed_loop_test_selected_summary.csv",
    },
    192: {
        "adaptive_dir": DATA_ROOT / "deltaA_signal_audit" / "solar192_existing_prediction_ensemble",
        "prefix": "solar192_static_adaptive_alpha",
        "posthoc_test": DATA_ROOT
        / "deltaA_signal_audit"
        / "solar192_closed_loop_rank_quality_guard"
        / "solar192_static_rank_quality_guard_closed_loop_test_selected_summary.csv",
    },
}


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


def fmt_float(value: float, digits: int = 6) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_pct(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):+.{digits}f}%"


def markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "horizon",
        "setting",
        "kind",
        "test_mse",
        "test_mae",
        "test_mse_gain_vs_baseline_mean_pct",
        "test_mae_gain_vs_baseline_mean_pct",
        "test_mse_gain_vs_best_single_pct",
        "test_mae_gain_vs_best_single_pct",
    ]
    headers = [
        "horizon",
        "setting",
        "kind",
        "test MSE",
        "test MAE",
        "MSE vs baseline mean",
        "MAE vs baseline mean",
        "MSE vs best single",
        "MAE vs best single",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---:", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:"]) + " |",
    ]
    for _, row in df[cols].iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["horizon"])),
                    str(row["setting"]),
                    str(row["kind"]),
                    fmt_float(row["test_mse"]),
                    fmt_float(row["test_mae"]),
                    fmt_pct(row["test_mse_gain_vs_baseline_mean_pct"]),
                    fmt_pct(row["test_mae_gain_vs_baseline_mean_pct"]),
                    fmt_pct(row["test_mse_gain_vs_best_single_pct"]),
                    fmt_pct(row["test_mae_gain_vs_best_single_pct"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def evaluate_specs(horizon: int, cfg: dict) -> tuple[list[dict], pd.DataFrame]:
    adaptive_dir = cfg["adaptive_dir"]
    prefix = cfg["prefix"]
    manifest = read_json(adaptive_dir / f"{prefix}_manifest.json")
    alpha_summary = read_json(adaptive_dir / f"{prefix}_adaptive_alpha_summary.json")
    selected = pd.read_csv(adaptive_dir / f"{prefix}_selected_test_summary.csv").iloc[0]
    variable_alpha = pd.read_csv(adaptive_dir / f"{prefix}_variable_alpha.csv")["alpha_shrunk"].to_numpy(dtype=np.float64)

    candidates = manifest["candidates"]
    n = len(candidates)
    baseline_idx, static_idx = group_indices(candidates)
    candidate_names = [candidate["candidate"] for candidate in candidates]
    best_single_name = str(selected["reference_best_single"])
    best_single_idx = candidate_names.index(best_single_name)
    alpha_global = float(alpha_summary["alpha_global_clipped"])

    specs = [
        {
            "setting": "baseline_mean",
            "kind": "group_mean",
            "weights": mean_weights(n, baseline_idx),
        },
        {
            "setting": "static_mean",
            "kind": "group_mean",
            "weights": mean_weights(n, static_idx),
        },
        {
            "setting": f"best_single_{best_single_name}",
            "kind": "single_reference",
            "weights": onehot(n, best_single_idx),
        },
        {
            "setting": "global_closed_form_alpha",
            "kind": "adaptive_global_alpha",
            "weights": group_blend_weights(candidates, alpha_global),
            "alpha_mean": alpha_global,
            "alpha_std": 0.0,
        },
        {
            "setting": "per_variable_shrinkage_alpha",
            "kind": "adaptive_variable_alpha",
            "alpha_vector": variable_alpha,
            "alpha_mean": float(variable_alpha.mean()),
            "alpha_std": float(variable_alpha.std()),
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
                "horizon": horizon,
                "setting": spec["setting"],
                "kind": spec["kind"],
                "alpha_mean": spec.get("alpha_mean", np.nan),
                "alpha_std": spec.get("alpha_std", np.nan),
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                "test_mse": test_metrics["mse"],
                "test_mae": test_metrics["mae"],
                "reference_best_single": best_single_name,
                "selection_reason": str(selected["selection_reason"]),
            }
        )

    posthoc_path = cfg["posthoc_test"]
    if posthoc_path.exists():
        posthoc = pd.read_csv(posthoc_path).iloc[0]
        rows.append(
            {
                "horizon": horizon,
                "setting": "posthoc_closed_loop",
                "kind": "guarded_dynamic",
                "alpha_mean": np.nan,
                "alpha_std": np.nan,
                "val_mse": np.nan,
                "val_mae": np.nan,
                "test_mse": float(posthoc["posthoc_mse"]),
                "test_mae": float(posthoc["posthoc_mae"]),
                "reference_best_single": best_single_name,
                "selection_reason": f"{posthoc['mode_status']}:{posthoc['mode_reason']}",
            }
        )

    alpha_frame = pd.read_csv(adaptive_dir / f"{prefix}_variable_alpha.csv")
    alpha_frame.insert(0, "horizon", horizon)
    alpha_frame["alpha_rank_desc"] = alpha_frame["alpha_shrunk"].rank(ascending=False, method="first").astype(int)
    return rows, alpha_frame


def add_reference_gains(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for horizon, idx in out.groupby("horizon").groups.items():
        sub = out.loc[idx]
        baseline = sub[sub["setting"] == "baseline_mean"].iloc[0]
        best_single = sub[sub["setting"].str.startswith("best_single_")].iloc[0]
        for metric in ("mse", "mae"):
            out.loc[idx, f"test_{metric}_gain_vs_baseline_mean_pct"] = [
                pct_gain(float(baseline[f"test_{metric}"]), float(value)) for value in sub[f"test_{metric}"]
            ]
            out.loc[idx, f"test_{metric}_gain_vs_best_single_pct"] = [
                pct_gain(float(best_single[f"test_{metric}"]), float(value)) for value in sub[f"test_{metric}"]
            ]
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    alpha_frames = []
    for horizon, cfg in RUNS.items():
        run_rows, alpha_frame = evaluate_specs(horizon, cfg)
        rows.extend(run_rows)
        alpha_frames.append(alpha_frame)

    table = add_reference_gains(pd.DataFrame(rows))
    table = table.sort_values(["horizon", "kind", "setting"]).reset_index(drop=True)
    alpha_all = pd.concat(alpha_frames, ignore_index=True)
    top_alpha = alpha_all.sort_values(["horizon", "alpha_rank_desc"]).groupby("horizon", as_index=False).head(20)

    table_csv = OUT_DIR / "solar_adaptive_alpha_frozen_table.csv"
    table_md = OUT_DIR / "solar_adaptive_alpha_frozen_table.md"
    alpha_csv = OUT_DIR / "solar_adaptive_alpha_variable_alpha.csv"
    top_alpha_csv = OUT_DIR / "solar_adaptive_alpha_top_alpha_targets.csv"
    readme_path = OUT_DIR / "README.md"

    table.to_csv(table_csv, index=False)
    alpha_all.to_csv(alpha_csv, index=False)
    top_alpha.to_csv(top_alpha_csv, index=False)
    table_md.write_text(
        "# Solar Adaptive-Alpha Frozen Table\n\n"
        "This table evaluates prediction-level adaptive fusion over existing Solar baseline/static prediction arrays. "
        "Validation selects alpha; test is used once for the selected report. No new training is performed.\n\n"
        + markdown_table(table)
        + "\n",
        encoding="utf-8",
    )

    selected = table[table["setting"] == "per_variable_shrinkage_alpha"].copy()
    lines = [
        "# Solar96/Solar192 Adaptive-Alpha Evidence",
        "",
        "This package freezes the Solar adaptive-alpha branch after the Solar-192 closed-loop run.",
        "",
        "Boundary: this is prediction-level baseline/static fusion, not a new train-time graph model and not a formal dynamic closed-loop success claim.",
        "",
        "Key results:",
    ]
    for _, row in selected.sort_values("horizon").iterrows():
        lines.append(
            "- Solar-{h}: selected per-variable alpha, test MSE/MAE {mse:.6f}/{mae:.6f}, "
            "gain vs baseline mean {gmse:+.3f}%/{gmae:+.3f}%, gain vs best single {gsmse:+.3f}%/{gsmae:+.3f}%.".format(
                h=int(row["horizon"]),
                mse=float(row["test_mse"]),
                mae=float(row["test_mae"]),
                gmse=float(row["test_mse_gain_vs_baseline_mean_pct"]),
                gmae=float(row["test_mae_gain_vs_baseline_mean_pct"]),
                gsmse=float(row["test_mse_gain_vs_best_single_pct"]),
                gsmae=float(row["test_mae_gain_vs_best_single_pct"]),
            )
        )
    lines.extend(
        [
            "",
            "Files:",
            "- `solar_adaptive_alpha_frozen_table.csv/md`: frozen cross-horizon table.",
            "- `solar_adaptive_alpha_variable_alpha.csv`: all per-target alpha values for Solar-96 and Solar-192.",
            "- `solar_adaptive_alpha_top_alpha_targets.csv`: top-20 static-anchor targets per horizon.",
            "- `manifest.json`: source output paths.",
        ]
    )
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_json(
        OUT_DIR / "manifest.json",
        {
            "artifact": "solar_adaptive_alpha_evidence",
            "output_dir": str(OUT_DIR),
            "runs": {
                str(horizon): {
                    "adaptive_dir": str(cfg["adaptive_dir"]),
                    "prefix": cfg["prefix"],
                    "posthoc_test": str(cfg["posthoc_test"]),
                }
                for horizon, cfg in RUNS.items()
            },
        },
    )
    print(f"[Done] wrote {table_csv}")
    print(f"[Done] wrote {readme_path}")


if __name__ == "__main__":
    main()

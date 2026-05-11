from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from posthoc_calibration.profiles import PROFILES, RESULT_ROOT


OUT_DIR = Path(r"C:\Users\cyl\Desktop\data\mechanism_evidence\etth196_parcorr_selective_hard_negative_20260507")
CMIKNN_SUMMARY = Path(
    r"C:\Users\cyl\Desktop\data\deltaA_signal_audit\etth196_closed_loop_rank_quality_guard\etth196_static_rank_quality_guard_closed_loop_test_selected_summary.csv"
)
PARCORR_SUMMARY = Path(
    r"C:\Users\cyl\Desktop\data\deltaA_signal_audit\etth196_closed_loop_rank_quality_guard_parcorr_ridgebase_sparse\etth196_static_parcorr_rank_quality_guard_parcorr_closed_loop_test_selected_summary.csv"
)


def projection_key(path: str) -> int:
    name = Path(path).name
    if "projection_" not in name:
        return 999
    try:
        return int(name.rsplit("projection_", 1)[1])
    except ValueError:
        return 999


def pct_gain(anchor: float, candidate: float) -> float:
    return (float(anchor) - float(candidate)) / float(anchor) * 100.0


def metric_frame(pattern: str) -> pd.DataFrame:
    rows: list[dict] = []
    for directory in sorted(glob.glob(str(RESULT_ROOT / pattern)), key=projection_key):
        metrics_path = Path(directory) / "metrics.npy"
        if not metrics_path.exists():
            continue
        raw = np.asarray(np.load(metrics_path)).reshape(-1).tolist()
        if len(raw) >= 5:
            mae, mse = float(raw[0]), float(raw[1])
        elif len(raw) == 2:
            mae, mse = float(raw[0]), float(raw[1])
        else:
            raise ValueError(f"Unexpected metrics length at {metrics_path}: {len(raw)}")
        rows.append({"result_dir": str(directory), "mse": mse, "mae": mae})
    if not rows:
        raise FileNotFoundError(f"No metrics found for pattern: {pattern}")
    return pd.DataFrame(rows)


def read_one(path: Path) -> dict:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty CSV: {path}")
    return df.iloc[0].to_dict()


def metric_pair(mse: float, mae: float) -> str:
    return f"{mse:.6f} / {mae:.6f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    profile = PROFILES["etth196_static"]
    baseline = metric_frame(str(profile["baseline_pattern"]))
    static = metric_frame(str(profile["static_pattern"]))
    baseline_mse = float(baseline["mse"].mean())
    baseline_mae = float(baseline["mae"].mean())
    static_mse = float(static["mse"].mean())
    static_mae = float(static["mae"].mean())

    cmiknn = read_one(CMIKNN_SUMMARY)
    parcorr = read_one(PARCORR_SUMMARY)

    route_summary = pd.DataFrame(
        [
            {
                "route": "baseline",
                "backend": "projection_mean",
                "mse": baseline_mse,
                "mae": baseline_mae,
                "mse_gain_vs_baseline_pct": 0.0,
                "mae_gain_vs_baseline_pct": 0.0,
                "mse_gain_vs_static_pct": np.nan,
                "mae_gain_vs_static_pct": np.nan,
                "mode_status": "",
                "mode_reason": "",
            },
            {
                "route": "static_anchor",
                "backend": "projection_mean",
                "mse": static_mse,
                "mae": static_mae,
                "mse_gain_vs_baseline_pct": pct_gain(baseline_mse, static_mse),
                "mae_gain_vs_baseline_pct": pct_gain(baseline_mae, static_mae),
                "mse_gain_vs_static_pct": 0.0,
                "mae_gain_vs_static_pct": 0.0,
                "mode_status": "",
                "mode_reason": "",
            },
            {
                "route": "guarded_posthoc_selected",
                "backend": "ParCorr_ridgebase_sparse",
                "mse": float(parcorr["posthoc_mse"]),
                "mae": float(parcorr["posthoc_mae"]),
                "mse_gain_vs_baseline_pct": pct_gain(baseline_mse, float(parcorr["posthoc_mse"])),
                "mae_gain_vs_baseline_pct": pct_gain(baseline_mae, float(parcorr["posthoc_mae"])),
                "mse_gain_vs_static_pct": pct_gain(static_mse, float(parcorr["posthoc_mse"])),
                "mae_gain_vs_static_pct": pct_gain(static_mae, float(parcorr["posthoc_mae"])),
                "mode_status": str(parcorr.get("mode_status", "")),
                "mode_reason": str(parcorr.get("mode_reason", "")),
            },
        ]
    )

    backend_comparison = pd.DataFrame(
        [
            {
                "backend": "CMIknn_ridgebase_sparse",
                "mode_status": str(cmiknn.get("mode_status", "")),
                "mode_reason": str(cmiknn.get("mode_reason", "")),
                "lambda_mode": str(cmiknn.get("lambda_mode", "")),
                "lambda_window": int(cmiknn.get("lambda_window", 0)),
                "lambda_k": int(cmiknn.get("lambda_k", 0)),
                "static_mse": float(cmiknn["static_mse"]),
                "posthoc_mse": float(cmiknn["posthoc_mse"]),
                "mse_gain_vs_static_pct": float(cmiknn["mse_gain_pct"]),
                "posthoc_mse_gain_vs_baseline_pct": pct_gain(baseline_mse, float(cmiknn["posthoc_mse"])),
                "static_mae": float(cmiknn["static_mae"]),
                "posthoc_mae": float(cmiknn["posthoc_mae"]),
                "mae_gain_vs_static_pct": float(cmiknn["mae_gain_pct"]),
                "posthoc_mae_gain_vs_baseline_pct": pct_gain(baseline_mae, float(cmiknn["posthoc_mae"])),
                "selection_reason": str(cmiknn.get("selection_reason", "")),
                "summary_csv": str(CMIKNN_SUMMARY),
            },
            {
                "backend": "ParCorr_ridgebase_sparse",
                "mode_status": str(parcorr.get("mode_status", "")),
                "mode_reason": str(parcorr.get("mode_reason", "")),
                "lambda_mode": str(parcorr.get("lambda_mode", "")),
                "lambda_window": int(parcorr.get("lambda_window", 0)),
                "lambda_k": int(parcorr.get("lambda_k", 0)),
                "static_mse": float(parcorr["static_mse"]),
                "posthoc_mse": float(parcorr["posthoc_mse"]),
                "mse_gain_vs_static_pct": float(parcorr["mse_gain_pct"]),
                "posthoc_mse_gain_vs_baseline_pct": pct_gain(baseline_mse, float(parcorr["posthoc_mse"])),
                "static_mae": float(parcorr["static_mae"]),
                "posthoc_mae": float(parcorr["posthoc_mae"]),
                "mae_gain_vs_static_pct": float(parcorr["mae_gain_pct"]),
                "posthoc_mae_gain_vs_baseline_pct": pct_gain(baseline_mae, float(parcorr["posthoc_mae"])),
                "selection_reason": str(parcorr.get("selection_reason", "")),
                "summary_csv": str(PARCORR_SUMMARY),
            },
        ]
    )

    route_summary.to_csv(OUT_DIR / "etth196_route_summary.csv", index=False)
    backend_comparison.to_csv(OUT_DIR / "etth196_backend_comparison.csv", index=False)

    lines = [
        "# ETTh1-96 ParCorr Freeze",
        "",
        "Purpose:",
        "- Freeze the ETTh1-96 guarded closed-loop result under the current ParCorr ridgebase_sparse interface.",
        "- Keep the result comparable to the existing CMIknn sparse route without mixing old regen assets into the current protocol.",
        "- Preserve ETTh1 as a hard negative overall while documenting that post-hoc dynamic remains Selective vs static anchor.",
        "",
        "Main readout:",
        f"- Baseline mean: `{metric_pair(baseline_mse, baseline_mae)}`.",
        (
            f"- Static anchor mean: `{metric_pair(static_mse, static_mae)}`; "
            f"`dBase {pct_gain(baseline_mse, static_mse):+.2f}% / {pct_gain(baseline_mae, static_mae):+.2f}%`."
        ),
        (
            f"- ParCorr guarded post-hoc: `{metric_pair(float(parcorr['posthoc_mse']), float(parcorr['posthoc_mae']))}`; "
            f"`dStatic {float(parcorr['mse_gain_pct']):+.2f}% / {float(parcorr['mae_gain_pct']):+.2f}%`; "
            f"`dBase {pct_gain(baseline_mse, float(parcorr['posthoc_mse'])):+.2f}% / {pct_gain(baseline_mae, float(parcorr['posthoc_mae'])):+.2f}%`."
        ),
        f"- Final mode: `{parcorr.get('mode_status', '')}` with `mode_reason={parcorr.get('mode_reason', '')}`.",
        "",
        "Backend comparison:",
        (
            f"- CMIknn sparse and ParCorr ridgebase_sparse give the same qualitative verdict: "
            f"`Selective vs static`, but still below baseline overall."
        ),
        (
            f"- ParCorr is slightly more conservative than CMIknn on test: "
            f"`dStatic MSE {float(parcorr['mse_gain_pct']):+.3f}% vs {float(cmiknn['mse_gain_pct']):+.3f}%`, "
            f"`dStatic MAE {float(parcorr['mae_gain_pct']):+.3f}% vs {float(cmiknn['mae_gain_pct']):+.3f}%`."
        ),
        "",
        "Interpretation:",
        "- ETTh1 should remain a falsification-style case rather than a performance headline.",
        "- The guarded protocol is not dead on ETTh1, because dynamic windows can still pass the double guard selectively.",
        "- However, the gain only repairs a weak static anchor and does not surpass the baseline route.",
        "",
        "Files:",
        "- `etth196_route_summary.csv`: baseline/static/parcorr route summary.",
        "- `etth196_backend_comparison.csv`: side-by-side CMIknn vs ParCorr selected summaries.",
        "- `manifest.json`: source files and result directories.",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "dataset": "ETTh1",
        "horizon": 96,
        "purpose": "Freeze ETTh1-96 ParCorr hard-negative route evidence",
        "baseline_pattern": str(profile["baseline_pattern"]),
        "static_pattern": str(profile["static_pattern"]),
        "baseline_dirs": baseline["result_dir"].tolist(),
        "static_dirs": static["result_dir"].tolist(),
        "cmiknn_summary": str(CMIKNN_SUMMARY),
        "parcorr_summary": str(PARCORR_SUMMARY),
        "parcorr_interface_dir": str(PROFILES["etth196_static_parcorr"]["interface_dir"]),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[Wrote] {OUT_DIR}")


if __name__ == "__main__":
    main()

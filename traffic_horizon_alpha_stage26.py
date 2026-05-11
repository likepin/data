from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from posthoc_calibration.evaluation import pct_gain
from posthoc_calibration.profiles import PROFILES
from traffic_existing_prediction_ensemble import (
    group_indices,
    group_mean_chunk,
    load_candidates,
    pred_path,
    true_path,
)


DATA_ROOT = Path(r"C:\Users\cyl\Desktop\data")
DEFAULT_STAGE2_DIR = DATA_ROOT / "deltaA_signal_audit" / "traffic96_existing_prediction_ensemble_stage2_light_seed2026"
DEFAULT_OUT_DIR = DATA_ROOT / "deltaA_signal_audit" / "traffic96_horizon_alpha_stage26"
STAGE2_PREFIX = "traffic96_static_stage2_light_seed2026"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traffic Stage2.6: validation-only horizon-aware global alpha diagnostic."
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="traffic96_static")
    parser.add_argument("--stage2-dir", type=Path, default=DEFAULT_STAGE2_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="stage26_horizon_alpha")
    parser.add_argument("--rhos", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--select-mae-min-gain-vs-stage2", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=0, help="Debug cap per split. 0 means full split.")
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def read_one(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError(f"Expected one row in {path}, got {len(df)}")
    return df.iloc[0]


def load_stage2_alpha(stage2_dir: Path) -> tuple[float, np.ndarray]:
    summary = read_one(stage2_dir / f"{STAGE2_PREFIX}_adaptive_alpha_summary.csv")
    variable = pd.read_csv(stage2_dir / f"{STAGE2_PREFIX}_variable_alpha.csv")
    if "alpha_shrunk" not in variable:
        raise ValueError(f"Missing alpha_shrunk in {stage2_dir}")
    return float(summary["alpha_global_clipped"]), variable["alpha_shrunk"].to_numpy(dtype=np.float32)


def mse_mae_sums(err: np.ndarray) -> tuple[float, float]:
    err64 = np.asarray(err, dtype=np.float64)
    return float(np.square(err64).sum()), float(np.abs(err64).sum())


def alpha_view(alpha: float | np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    n_horizon = shape[1]
    n_vars = shape[2]
    arr = np.asarray(alpha, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1, 1, 1)
    if arr.ndim == 1 and arr.shape[0] == n_horizon:
        return arr.reshape(1, n_horizon, 1)
    if arr.ndim == 1 and arr.shape[0] == n_vars:
        return arr.reshape(1, 1, n_vars)
    if arr.ndim == 2 and arr.shape == (n_horizon, n_vars):
        return arr.reshape(1, n_horizon, n_vars)
    raise ValueError(f"Unsupported alpha shape {arr.shape}; expected scalar, horizon, variable, or horizon-variable.")


def evaluate_alpha(
    *,
    candidates: list[dict],
    alpha: float | np.ndarray,
    split: str,
    chunk_size: int,
    max_samples: int,
    progress_every: int,
) -> dict:
    baseline_idx, static_idx = group_indices(candidates)
    pred_arrays = [np.load(pred_path(candidate, split), mmap_mode="r") for candidate in candidates]
    true = np.load(true_path(candidates[0], split), mmap_mode="r")
    expected_shape = true.shape
    for candidate, pred in zip(candidates, pred_arrays):
        if pred.shape != expected_shape:
            raise RuntimeError(f"Unexpected pred shape for {candidate['candidate']}: {pred.shape} vs {expected_shape}")

    n_samples = expected_shape[0] if max_samples <= 0 else min(int(max_samples), expected_shape[0])
    view = alpha_view(alpha, expected_shape)
    sse = 0.0
    sae = 0.0
    count = n_samples * expected_shape[1] * expected_shape[2]
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        baseline_mean = group_mean_chunk(pred_arrays, baseline_idx, start, end)
        static_mean = group_mean_chunk(pred_arrays, static_idx, start, end)
        ensemble = baseline_mean + view * (static_mean - baseline_mean)
        err = np.asarray(true[start:end], dtype=np.float32) - ensemble
        chunk_sse, chunk_sae = mse_mae_sums(err)
        sse += chunk_sse
        sae += chunk_sae
        if progress_every > 0 and (end % progress_every == 0 or end == n_samples):
            print(f"[{split}:eval] {end}/{n_samples}", flush=True)
    return {"mse": sse / count, "mae": sae / count, "n_samples": n_samples}


def compute_horizon_alpha(
    *,
    candidates: list[dict],
    split: str,
    chunk_size: int,
    max_samples: int,
    progress_every: int,
    min_denom: float = 1e-12,
) -> tuple[pd.DataFrame, dict]:
    baseline_idx, static_idx = group_indices(candidates)
    pred_arrays = [np.load(pred_path(candidate, split), mmap_mode="r") for candidate in candidates]
    true = np.load(true_path(candidates[0], split), mmap_mode="r")
    expected_shape = true.shape
    for candidate, pred in zip(candidates, pred_arrays):
        if pred.shape != expected_shape:
            raise RuntimeError(f"Unexpected pred shape for {candidate['candidate']}: {pred.shape} vs {expected_shape}")

    n_samples = expected_shape[0] if max_samples <= 0 else min(int(max_samples), expected_shape[0])
    n_horizon = expected_shape[1]
    num_h = np.zeros(n_horizon, dtype=np.float64)
    den_h = np.zeros(n_horizon, dtype=np.float64)
    num_global = 0.0
    den_global = 0.0
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        baseline_mean = group_mean_chunk(pred_arrays, baseline_idx, start, end)
        static_mean = group_mean_chunk(pred_arrays, static_idx, start, end)
        diff = static_mean - baseline_mean
        residual = np.asarray(true[start:end], dtype=np.float32) - baseline_mean
        prod = diff * residual
        diff_sq = np.square(diff, dtype=np.float32)
        num_h += prod.sum(axis=(0, 2), dtype=np.float64)
        den_h += diff_sq.sum(axis=(0, 2), dtype=np.float64)
        num_global += float(prod.sum(dtype=np.float64))
        den_global += float(diff_sq.sum(dtype=np.float64))
        if progress_every > 0 and (end % progress_every == 0 or end == n_samples):
            print(f"[{split}:alpha_h] {end}/{n_samples}", flush=True)

    alpha_raw = np.divide(num_h, den_h, out=np.full_like(num_h, 0.5), where=den_h > min_denom)
    alpha_clipped = np.clip(alpha_raw, 0.0, 1.0)
    alpha_global_raw = num_global / den_global if den_global > min_denom else 0.5
    alpha_global = float(np.clip(alpha_global_raw, 0.0, 1.0))
    df = pd.DataFrame(
        {
            "horizon_index": np.arange(n_horizon, dtype=np.int64),
            "horizon_step": np.arange(1, n_horizon + 1, dtype=np.int64),
            "alpha_raw": alpha_raw,
            "alpha_clipped": alpha_clipped,
            "denominator": den_h,
            "numerator": num_h,
            "delta_vs_global": alpha_clipped - alpha_global,
        }
    )
    summary = {
        "split": split,
        "n_samples": int(n_samples),
        "n_horizon": int(n_horizon),
        "alpha_global_raw_from_horizon_run": float(alpha_global_raw),
        "alpha_global_clipped_from_horizon_run": float(alpha_global),
        "alpha_h_mean": float(alpha_clipped.mean()),
        "alpha_h_std": float(alpha_clipped.std()),
        "alpha_h_min": float(alpha_clipped.min()),
        "alpha_h_max": float(alpha_clipped.max()),
        "alpha_h_first": float(alpha_clipped[0]),
        "alpha_h_last": float(alpha_clipped[-1]),
        "alpha_h_last_minus_first": float(alpha_clipped[-1] - alpha_clipped[0]),
        "alpha_h_last_minus_global": float(alpha_clipped[-1] - alpha_global),
    }
    return df, summary


def candidate_specs(alpha_global: float, alpha_var: np.ndarray, alpha_h: np.ndarray, rhos: list[float]) -> list[dict]:
    specs = [
        {
            "ensemble": "stage2_global_alpha_reference",
            "kind": "global_alpha_reference",
            "alpha": float(alpha_global),
            "rho": np.nan,
            "alpha_source": "stage2_validation_global_closed_form",
        },
        {
            "ensemble": "stage2_variable_alpha_reference",
            "kind": "variable_alpha_reference",
            "alpha": alpha_var.astype(np.float32),
            "rho": np.nan,
            "alpha_source": "stage2_validation_variable_shrink",
        },
    ]
    for rho in rhos:
        rho = float(rho)
        if rho < 0.0 or rho > 1.0:
            raise ValueError(f"rho must be in [0,1], got {rho}")
        shrunk = alpha_global + rho * (alpha_h - alpha_global)
        shrunk = np.clip(shrunk, 0.0, 1.0).astype(np.float32)
        specs.append(
            {
                "ensemble": f"stage26_horizon_alpha_rho{rho:.2f}",
                "kind": "horizon_alpha_shrink",
                "alpha": shrunk,
                "rho": rho,
                "alpha_source": "validation_horizon_closed_form_shrunk_to_global",
            }
        )
    return specs


def evaluate_specs(
    *,
    candidates: list[dict],
    specs: list[dict],
    split: str,
    chunk_size: int,
    max_samples: int,
    progress_every: int,
) -> pd.DataFrame:
    rows = []
    for idx, spec in enumerate(specs):
        metrics = evaluate_alpha(
            candidates=candidates,
            alpha=spec["alpha"],
            split=split,
            chunk_size=chunk_size,
            max_samples=max_samples,
            progress_every=0,
        )
        alpha = np.asarray(spec["alpha"], dtype=np.float64)
        rows.append(
            {
                "split": split,
                "ensemble": spec["ensemble"],
                "kind": spec["kind"],
                "alpha_source": spec["alpha_source"],
                "rho": spec["rho"],
                "alpha_mean": float(alpha.mean()),
                "alpha_std": float(alpha.std()),
                "alpha_min": float(alpha.min()),
                "alpha_max": float(alpha.max()),
                "mse": metrics["mse"],
                "mae": metrics["mae"],
                "n_samples": metrics["n_samples"],
            }
        )
        if progress_every > 0 and ((idx + 1) % progress_every == 0 or idx + 1 == len(specs)):
            print(f"[{split}:specs] {idx + 1}/{len(specs)}", flush=True)
    return pd.DataFrame(rows)


def add_reference_gains(df: pd.DataFrame, stage2_ref: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["mse_gain_vs_stage2_variable_pct"] = [
        pct_gain(float(stage2_ref["mse"]), float(value)) for value in out["mse"]
    ]
    out["mae_gain_vs_stage2_variable_pct"] = [
        pct_gain(float(stage2_ref["mae"]), float(value)) for value in out["mae"]
    ]
    return out


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    def fmt(value) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|")

    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_trend_summary(
    *,
    out_dir: Path,
    horizon_df: pd.DataFrame,
    selected: pd.Series,
    test_selected: pd.Series,
    test_stage2: pd.Series,
) -> None:
    first = horizon_df.iloc[0]
    last = horizon_df.iloc[-1]
    max_row = horizon_df.iloc[horizon_df["alpha_clipped"].idxmax()]
    min_row = horizon_df.iloc[horizon_df["alpha_clipped"].idxmin()]
    lines = [
        "# Traffic96 Stage2.6 Horizon Alpha Diagnostic",
        "",
        "Scope: validation-only global `alpha_h` by forecast horizon; no retraining and no test-time selection.",
        "",
        "## Alpha-H Trend",
        "",
        f"- first horizon alpha: `{float(first['alpha_clipped']):.6f}`",
        f"- final horizon alpha: `{float(last['alpha_clipped']):.6f}`",
        f"- final-minus-first: `{float(last['alpha_clipped'] - first['alpha_clipped']):+.6f}`",
        f"- max alpha: horizon `{int(max_row['horizon_step'])}`, alpha `{float(max_row['alpha_clipped']):.6f}`",
        f"- min alpha: horizon `{int(min_row['horizon_step'])}`, alpha `{float(min_row['alpha_clipped']):.6f}`",
        "",
        "## Validation Selection",
        "",
        f"- selected ensemble: `{selected['ensemble']}`",
        f"- selected rho: `{float(selected['rho']):.2f}`",
        f"- validation MSE / MAE: `{float(selected['mse']):.10f}` / `{float(selected['mae']):.10f}`",
        f"- validation gain vs Stage2 variable alpha: MSE `{float(selected['mse_gain_vs_stage2_variable_pct']):+.4f}%`, MAE `{float(selected['mae_gain_vs_stage2_variable_pct']):+.4f}%`",
        "",
        "## Test Once",
        "",
        f"- Stage2 variable reference test MSE / MAE: `{float(test_stage2['mse']):.10f}` / `{float(test_stage2['mae']):.10f}`",
        f"- selected horizon-alpha test MSE / MAE: `{float(test_selected['mse']):.10f}` / `{float(test_selected['mae']):.10f}`",
        f"- test gain vs Stage2 variable alpha: MSE `{float(test_selected['mse_gain_vs_stage2_variable_pct']):+.4f}%`, MAE `{float(test_selected['mae_gain_vs_stage2_variable_pct']):+.4f}%`",
        "",
        "Interpretation rule:",
        "- If the selected horizon-alpha row does not beat Stage2 variable alpha on test, keep this as a diagnostic only.",
        "- Do not replace the Stage2 anchor unless both validation and test are positive against the variable-alpha anchor.",
        "",
    ]
    (out_dir / "traffic96_stage26_horizon_alpha_trend_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.profile != "traffic96_static":
        raise ValueError("Stage2.6 horizon alpha diagnostic is currently scoped to traffic96_static.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.profile}_{args.tag}"

    profile = dict(PROFILES[args.profile])
    candidates = load_candidates(profile)
    alpha_global, alpha_var = load_stage2_alpha(args.stage2_dir)
    rhos = parse_float_list(args.rhos)

    horizon_df, horizon_summary = compute_horizon_alpha(
        candidates=candidates,
        split="val",
        chunk_size=args.chunk_size,
        max_samples=args.max_samples,
        progress_every=args.progress_every,
    )
    alpha_h = horizon_df["alpha_clipped"].to_numpy(dtype=np.float32)
    horizon_df.to_csv(out_dir / f"{prefix}_alpha_h_by_horizon.csv", index=False)
    pd.DataFrame([horizon_summary]).to_csv(out_dir / f"{prefix}_alpha_h_summary.csv", index=False)

    specs = candidate_specs(alpha_global, alpha_var, alpha_h, rhos)
    val_grid = evaluate_specs(
        candidates=candidates,
        specs=specs,
        split="val",
        chunk_size=args.chunk_size,
        max_samples=args.max_samples,
        progress_every=args.progress_every,
    )
    stage2_val_ref = val_grid[val_grid["ensemble"] == "stage2_variable_alpha_reference"].iloc[0]
    val_grid = add_reference_gains(val_grid, stage2_val_ref)
    val_grid.to_csv(out_dir / f"{prefix}_val_grid.csv", index=False)
    write_markdown_table(val_grid, out_dir / f"{prefix}_val_grid.md")

    horizon_candidates = val_grid[val_grid["kind"] == "horizon_alpha_shrink"].copy()
    eligible = horizon_candidates[
        horizon_candidates["mae_gain_vs_stage2_variable_pct"] >= float(args.select_mae_min_gain_vs_stage2)
    ].copy()
    if eligible.empty:
        selected = horizon_candidates.sort_values(["mse", "mae"]).iloc[0]
        selection_reason = "best_horizon_val_mse_no_stage2_mae_guard_candidate"
    else:
        selected = eligible.sort_values(["mse", "mae"]).iloc[0]
        selection_reason = "best_horizon_val_mse_with_stage2_mae_guard"
    selected_spec = next(spec for spec in specs if spec["ensemble"] == selected["ensemble"])

    test_specs = [
        next(spec for spec in specs if spec["ensemble"] == "stage2_variable_alpha_reference"),
        selected_spec,
    ]
    test_grid = evaluate_specs(
        candidates=candidates,
        specs=test_specs,
        split="test",
        chunk_size=args.chunk_size,
        max_samples=args.max_samples,
        progress_every=args.progress_every,
    )
    stage2_test_ref = test_grid[test_grid["ensemble"] == "stage2_variable_alpha_reference"].iloc[0]
    test_grid = add_reference_gains(test_grid, stage2_test_ref)
    test_grid.to_csv(out_dir / f"{prefix}_test_selected.csv", index=False)
    write_markdown_table(test_grid, out_dir / f"{prefix}_test_selected.md")

    selected_test = test_grid[test_grid["ensemble"] == selected["ensemble"]].iloc[0]
    selected_row = {
        **{f"val_{k}": v for k, v in selected.to_dict().items()},
        **{f"test_{k}": v for k, v in selected_test.to_dict().items()},
        "selection_reason": selection_reason,
    }
    pd.DataFrame([selected_row]).to_csv(out_dir / f"{prefix}_selected_summary.csv", index=False)
    write_trend_summary(
        out_dir=out_dir,
        horizon_df=horizon_df,
        selected=selected,
        test_selected=selected_test,
        test_stage2=stage2_test_ref,
    )

    manifest = {
        "profile": args.profile,
        "stage2_dir": str(args.stage2_dir),
        "candidate_count": len(candidates),
        "alpha_global_from_stage2": float(alpha_global),
        "rhos": rhos,
        "select_mae_min_gain_vs_stage2": float(args.select_mae_min_gain_vs_stage2),
        "chunk_size": args.chunk_size,
        "max_samples": args.max_samples,
        "horizon_summary": horizon_summary,
        "selection_reason": selection_reason,
        "selected_ensemble": str(selected["ensemble"]),
        "selected_rho": float(selected["rho"]),
        "test_gain_vs_stage2_variable_mse_pct": float(selected_test["mse_gain_vs_stage2_variable_pct"]),
        "test_gain_vs_stage2_variable_mae_pct": float(selected_test["mae_gain_vs_stage2_variable_pct"]),
    }
    (out_dir / f"{prefix}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        "[Selected] "
        f"ensemble={selected['ensemble']} rho={float(selected['rho']):.2f} reason={selection_reason} "
        f"val_mse={float(selected['mse']):.6f} val_mae={float(selected['mae']):.6f} "
        f"val_mse_gain_vs_stage2={float(selected['mse_gain_vs_stage2_variable_pct']):+.4f}% "
        f"val_mae_gain_vs_stage2={float(selected['mae_gain_vs_stage2_variable_pct']):+.4f}%",
        flush=True,
    )
    print(
        "[Test] "
        f"mse={float(selected_test['mse']):.6f} mae={float(selected_test['mae']):.6f} "
        f"mse_gain_vs_stage2={float(selected_test['mse_gain_vs_stage2_variable_pct']):+.4f}% "
        f"mae_gain_vs_stage2={float(selected_test['mae_gain_vs_stage2_variable_pct']):+.4f}%",
        flush=True,
    )
    print(f"[Done] outputs written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

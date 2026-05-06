from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_ROOT = Path(r"C:\Users\cyl\Desktop\data")
ADAPTIVE_DIR = DATA_ROOT / "deltaA_signal_audit" / "traffic96_existing_prediction_ensemble"
INTERFACE_DIR = DATA_ROOT / "interfaces" / "Traffic_graph_interface_parcorr"
PACKAGE_DIR = DATA_ROOT / "mechanism_evidence" / "traffic96_mechanism_performance_20260506"
OUT_DIR = PACKAGE_DIR / "performance" / "adaptive_alpha_ensemble"

PREFIX = "traffic96_static_adaptive_alpha"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build lightweight Traffic adaptive-alpha mechanism/performance evidence package."
    )
    parser.add_argument("--adaptive-dir", type=Path, default=ADAPTIVE_DIR)
    parser.add_argument("--interface-dir", type=Path, default=INTERFACE_DIR)
    parser.add_argument("--package-dir", type=Path, default=PACKAGE_DIR)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--shuffle-count", type=int, default=256)
    parser.add_argument("--shuffle-seed", type=int, default=20260506)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def pct_gain(old: float, new: float) -> float:
    if old == 0.0:
        return float("nan")
    return 100.0 * (old - new) / old


def git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=DATA_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def pred_path(candidate: dict, split: str) -> Path:
    if split == "val":
        return Path(candidate["val_dir"]) / "val_pred.npy"
    if split == "test":
        return Path(candidate["test_dir"]) / "pred.npy"
    raise ValueError(split)


def true_path(candidate: dict, split: str) -> Path:
    if split == "val":
        return Path(candidate["val_dir"]) / "val_true.npy"
    if split == "test":
        return Path(candidate["test_dir"]) / "true.npy"
    raise ValueError(split)


def group_indices(candidates: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    groups = [candidate["group"] for candidate in candidates]
    baseline_idx = np.asarray([i for i, group in enumerate(groups) if group == "baseline"], dtype=np.int64)
    static_idx = np.asarray([i for i, group in enumerate(groups) if group == "static"], dtype=np.int64)
    if baseline_idx.size == 0 or static_idx.size == 0:
        raise ValueError("Expected both baseline and static candidate groups")
    return baseline_idx, static_idx


def group_mean_chunk(pred_arrays: list[np.ndarray], idx: np.ndarray, start: int, end: int) -> np.ndarray:
    out = np.zeros((end - start, *pred_arrays[0].shape[1:]), dtype=np.float32)
    scale = 1.0 / float(idx.size)
    for pred_idx in idx:
        out += scale * np.asarray(pred_arrays[int(pred_idx)][start:end], dtype=np.float32)
    return out


def split_sufficient_stats(
    candidates: list[dict],
    alpha: np.ndarray,
    split: str,
    chunk_size: int,
) -> dict:
    baseline_idx, static_idx = group_indices(candidates)
    pred_arrays = [np.load(pred_path(candidate, split), mmap_mode="r") for candidate in candidates]
    true = np.load(true_path(candidates[0], split), mmap_mode="r")
    expected_shape = true.shape
    for candidate, pred in zip(candidates, pred_arrays):
        if pred.shape != expected_shape:
            raise RuntimeError(f"Unexpected pred shape for {candidate['candidate']}: {pred.shape} vs {expected_shape}")

    n_vars = expected_shape[-1]
    if alpha.shape != (n_vars,):
        raise ValueError(f"alpha vector shape {alpha.shape} does not match variable count {n_vars}")
    alpha_view = alpha.astype(np.float32).reshape(1, 1, n_vars)

    s_rr = np.zeros(n_vars, dtype=np.float64)
    s_rd = np.zeros(n_vars, dtype=np.float64)
    s_dd = np.zeros(n_vars, dtype=np.float64)
    sae_base = np.zeros(n_vars, dtype=np.float64)
    sae_static = np.zeros(n_vars, dtype=np.float64)
    sae_alpha = np.zeros(n_vars, dtype=np.float64)
    n_samples = expected_shape[0]
    count_per_var = expected_shape[0] * expected_shape[1]

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        baseline_mean = group_mean_chunk(pred_arrays, baseline_idx, start, end)
        static_mean = group_mean_chunk(pred_arrays, static_idx, start, end)
        true_chunk = np.asarray(true[start:end], dtype=np.float32)

        residual = true_chunk - baseline_mean
        diff = static_mean - baseline_mean
        alpha_residual = residual - alpha_view * diff

        s_rr += np.square(residual, dtype=np.float32).sum(axis=(0, 1), dtype=np.float64)
        s_rd += (residual * diff).sum(axis=(0, 1), dtype=np.float64)
        s_dd += np.square(diff, dtype=np.float32).sum(axis=(0, 1), dtype=np.float64)
        sae_base += np.abs(residual).sum(axis=(0, 1), dtype=np.float64)
        sae_static += np.abs(residual - diff).sum(axis=(0, 1), dtype=np.float64)
        sae_alpha += np.abs(alpha_residual).sum(axis=(0, 1), dtype=np.float64)
        del baseline_mean, static_mean, true_chunk, residual, diff, alpha_residual

    sse_base = s_rr
    sse_static = s_rr - 2.0 * s_rd + s_dd
    sse_alpha = s_rr - 2.0 * alpha * s_rd + np.square(alpha) * s_dd
    return {
        "split": split,
        "count_per_var": int(count_per_var),
        "s_rr": s_rr,
        "s_rd": s_rd,
        "s_dd": s_dd,
        "base_mse": sse_base / count_per_var,
        "static_mse": sse_static / count_per_var,
        "alpha_mse": sse_alpha / count_per_var,
        "base_mae": sae_base / count_per_var,
        "static_mae": sae_static / count_per_var,
        "alpha_mae": sae_alpha / count_per_var,
    }


def mse_from_alpha(stats: dict, alpha: np.ndarray) -> float:
    sse = stats["s_rr"] - 2.0 * alpha * stats["s_rd"] + np.square(alpha) * stats["s_dd"]
    return float(sse.sum() / (stats["count_per_var"] * alpha.size))


def shuffled_negative_control(
    val_stats: dict,
    test_stats: dict,
    alpha: np.ndarray,
    shuffle_count: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    observed_val_mse = mse_from_alpha(val_stats, alpha)
    observed_test_mse = mse_from_alpha(test_stats, alpha)
    rows = []
    for idx in range(shuffle_count):
        shuffled = rng.permutation(alpha)
        rows.append(
            {
                "shuffle_id": idx,
                "val_mse": mse_from_alpha(val_stats, shuffled),
                "test_mse": mse_from_alpha(test_stats, shuffled),
            }
        )
    df = pd.DataFrame(rows)
    summary = {
        "shuffle_count": int(shuffle_count),
        "shuffle_seed": int(seed),
        "observed_val_mse": observed_val_mse,
        "observed_test_mse": observed_test_mse,
        "shuffle_val_mse_mean": float(df["val_mse"].mean()),
        "shuffle_val_mse_median": float(df["val_mse"].median()),
        "shuffle_val_mse_q05": float(df["val_mse"].quantile(0.05)),
        "shuffle_val_mse_q95": float(df["val_mse"].quantile(0.95)),
        "shuffle_test_mse_mean": float(df["test_mse"].mean()),
        "shuffle_test_mse_median": float(df["test_mse"].median()),
        "shuffle_test_mse_q05": float(df["test_mse"].quantile(0.05)),
        "shuffle_test_mse_q95": float(df["test_mse"].quantile(0.95)),
        "observed_val_mse_gain_vs_shuffle_median_pct": pct_gain(float(df["val_mse"].median()), observed_val_mse),
        "observed_test_mse_gain_vs_shuffle_median_pct": pct_gain(float(df["test_mse"].median()), observed_test_mse),
        "observed_val_rank_fraction_lower_is_better": float((df["val_mse"] <= observed_val_mse).mean()),
        "observed_test_rank_fraction_lower_is_better": float((df["test_mse"] <= observed_test_mse).mean()),
    }
    return df, summary


def graph_frame(interface_dir: Path) -> pd.DataFrame:
    support = np.load(interface_dir / "support.npy").astype(bool)
    a_base = np.load(interface_dir / "a_base_agg.npy").astype(np.float64)
    if support.shape != a_base.shape:
        raise ValueError(f"support/a_base shape mismatch: {support.shape} vs {a_base.shape}")
    support = support.copy()
    np.fill_diagonal(support, False)
    abs_a = np.abs(a_base) * support
    parent_count = support.sum(axis=1)
    child_count = support.sum(axis=0)
    parent_strength_sum = abs_a.sum(axis=1)
    child_strength_sum = abs_a.sum(axis=0)
    parent_strength_mean = np.divide(
        parent_strength_sum,
        np.maximum(parent_count, 1),
        out=np.zeros_like(parent_strength_sum),
        where=parent_count > 0,
    )
    parent_strength_max = abs_a.max(axis=1)
    return pd.DataFrame(
        {
            "target_index": np.arange(support.shape[0], dtype=np.int64),
            "parent_count": parent_count.astype(np.int64),
            "child_count": child_count.astype(np.int64),
            "parent_abs_strength_sum": parent_strength_sum,
            "parent_abs_strength_mean": parent_strength_mean,
            "parent_abs_strength_max": parent_strength_max,
            "child_abs_strength_sum": child_strength_sum,
        }
    )


def target_diagnostics(alpha_df: pd.DataFrame, val_stats: dict, test_stats: dict, graph_df: pd.DataFrame) -> pd.DataFrame:
    out = alpha_df.copy()
    alpha = out["alpha_shrunk"].to_numpy(dtype=np.float64)
    for split, stats in [("val", val_stats), ("test", test_stats)]:
        out[f"{split}_baseline_mean_mse"] = stats["base_mse"]
        out[f"{split}_static_mean_mse"] = stats["static_mse"]
        out[f"{split}_adaptive_alpha_mse"] = stats["alpha_mse"]
        out[f"{split}_baseline_mean_mae"] = stats["base_mae"]
        out[f"{split}_static_mean_mae"] = stats["static_mae"]
        out[f"{split}_adaptive_alpha_mae"] = stats["alpha_mae"]
        out[f"{split}_static_mse_gain_vs_baseline_pct"] = [
            pct_gain(old, new) for old, new in zip(stats["base_mse"], stats["static_mse"])
        ]
        out[f"{split}_adaptive_mse_gain_vs_baseline_pct"] = [
            pct_gain(old, new) for old, new in zip(stats["base_mse"], stats["alpha_mse"])
        ]
        out[f"{split}_adaptive_mse_gain_vs_static_pct"] = [
            pct_gain(old, new) for old, new in zip(stats["static_mse"], stats["alpha_mse"])
        ]
    out["alpha_minus_global"] = alpha - float(alpha_df.attrs.get("alpha_global", alpha.mean()))
    return out.merge(graph_df, on="target_index", how="left")


def alignment_summary(diag: pd.DataFrame, shuffle_summary: dict) -> pd.DataFrame:
    metrics = [
        "val_static_mse_gain_vs_baseline_pct",
        "val_adaptive_mse_gain_vs_baseline_pct",
        "val_adaptive_mse_gain_vs_static_pct",
        "test_static_mse_gain_vs_baseline_pct",
        "test_adaptive_mse_gain_vs_baseline_pct",
        "test_adaptive_mse_gain_vs_static_pct",
        "parent_count",
        "child_count",
        "parent_abs_strength_sum",
        "parent_abs_strength_mean",
        "parent_abs_strength_max",
        "child_abs_strength_sum",
        "reliability",
    ]
    rows = []
    for metric in metrics:
        rows.append(
            {
                "x": "alpha_shrunk",
                "y": metric,
                "pearson": float(diag["alpha_shrunk"].corr(diag[metric], method="pearson")),
                "spearman": float(diag["alpha_shrunk"].corr(diag[metric], method="spearman")),
            }
        )
    for key, value in shuffle_summary.items():
        rows.append({"x": "negative_control", "y": key, "pearson": value, "spearman": np.nan})
    return pd.DataFrame(rows)


def copy_raw_outputs(adaptive_dir: Path, raw_dir: Path) -> list[str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(adaptive_dir.glob(f"{PREFIX}_*")):
        if path.suffix.lower() == ".npy":
            continue
        dest = raw_dir / path.name
        shutil.copy2(path, dest)
        copied.append(path.name)
    return copied


def make_plots(diag: pd.DataFrame, summary: dict, fig_dir: Path) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    alpha_path = fig_dir / f"{PREFIX}_alpha_distribution.png"
    plt.figure(figsize=(8, 4.8))
    plt.hist(diag["alpha_shrunk"], bins=40, color="#28666e", alpha=0.86)
    plt.axvline(summary["alpha_global_clipped"], color="#d95d39", linestyle="--", linewidth=2, label="global alpha*")
    plt.axvline(0.60, color="#2f4858", linestyle=":", linewidth=2, label="grid alpha=0.60")
    plt.xlabel("per-variable shrunk alpha")
    plt.ylabel("target count")
    plt.title("Traffic96 adaptive alpha distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(alpha_path, dpi=180)
    plt.close()
    outputs.append(alpha_path.name)

    gain_path = fig_dir / f"{PREFIX}_alpha_gain_scatter.png"
    plt.figure(figsize=(8, 5.2))
    plt.scatter(
        diag["alpha_shrunk"],
        diag["val_adaptive_mse_gain_vs_baseline_pct"],
        c=diag["reliability"],
        s=18,
        cmap="viridis",
        alpha=0.78,
        edgecolors="none",
    )
    plt.colorbar(label="alpha reliability")
    plt.axvline(summary["alpha_global_clipped"], color="#d95d39", linestyle="--", linewidth=1.5)
    plt.axhline(0.0, color="#555555", linestyle=":", linewidth=1.2)
    plt.xlabel("per-variable shrunk alpha")
    plt.ylabel("validation MSE gain vs baseline mean (%)")
    plt.title("Adaptive alpha aligns with validation error correction")
    plt.tight_layout()
    plt.savefig(gain_path, dpi=180)
    plt.close()
    outputs.append(gain_path.name)

    graph_path = fig_dir / f"{PREFIX}_alpha_graph_scatter.png"
    plt.figure(figsize=(8, 5.2))
    plt.scatter(
        diag["parent_abs_strength_sum"],
        diag["alpha_shrunk"],
        c=diag["val_adaptive_mse_gain_vs_baseline_pct"],
        s=18,
        cmap="coolwarm",
        alpha=0.78,
        edgecolors="none",
    )
    plt.colorbar(label="validation MSE gain vs baseline mean (%)")
    plt.xlabel("PCMCI parent absolute strength sum")
    plt.ylabel("per-variable shrunk alpha")
    plt.title("Adaptive alpha vs static graph parent strength")
    plt.tight_layout()
    plt.savefig(graph_path, dpi=180)
    plt.close()
    outputs.append(graph_path.name)

    return outputs


def update_package_manifest(package_dir: Path, adaptive_claim: dict, copied_files: dict) -> None:
    manifest_path = package_dir / "manifest.json"
    manifest = read_json(manifest_path)
    manifest.setdefault("source_dirs", {})["adaptive_alpha_ensemble"] = str(ADAPTIVE_DIR)
    manifest.setdefault("claims", {})["adaptive_alpha_ensemble"] = adaptive_claim
    manifest.setdefault("layout", {})["adaptive_alpha_ensemble"] = "performance\\adaptive_alpha_ensemble"
    manifest.setdefault("copied_file_groups", {})["adaptive_alpha_ensemble"] = copied_files
    manifest.setdefault("reproduction_entrypoints", [])
    if "traffic_adaptive_alpha_evidence_pack.py" not in manifest["reproduction_entrypoints"]:
        manifest["reproduction_entrypoints"].append("traffic_adaptive_alpha_evidence_pack.py")
    manifest.setdefault("large_artifacts_not_copied", [])
    write_json(manifest_path, manifest)


def update_root_readme(package_dir: Path, selected: dict, adaptive_summary: dict, shuffle_summary: dict) -> None:
    readme_path = package_dir / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    start = "<!-- ADAPTIVE_ALPHA_SECTION_START -->"
    end = "<!-- ADAPTIVE_ALPHA_SECTION_END -->"
    section = f"""
{start}

### 4. Adaptive-Alpha Existing-Prediction Ensemble

Location:

`performance/adaptive_alpha_ensemble/`

Claim:

Validation-estimated adaptive blending strengthens the Traffic prediction-level performance branch while keeping selection on validation only.

Selected ensemble:

- `blend_baseline_static_alpha_variable_shrink`
- Global closed-form `alpha* = {adaptive_summary['alpha_global_clipped']:.6f}`
- Per-variable shrunk alpha mean/std: `{adaptive_summary['var_alpha_mean']:.6f} / {adaptive_summary['var_alpha_std']:.6f}`
- Per-variable alpha 5/50/95 percentiles: `{adaptive_summary['var_alpha_q05']:.6f} / {adaptive_summary['var_alpha_q50']:.6f} / {adaptive_summary['var_alpha_q95']:.6f}`

Selected result:

- Validation `MSE / MAE = {selected['val_mse']:.6f} / {selected['val_mae']:.6f}`
- Validation gain vs best single: `MSE +{selected['val_mse_gain_vs_best_single_pct']:.4f}%`, `MAE +{selected['val_mae_gain_vs_best_single_pct']:.4f}%`
- Test `MSE / MAE = {selected['test_mse']:.6f} / {selected['test_mae']:.6f}`
- Test gain vs best single: `MSE +{selected['test_mse_gain_vs_best_single_pct']:.4f}%`, `MAE +{selected['test_mae_gain_vs_best_single_pct']:.4f}%`

Negative control:

- Shuffled alpha median test MSE: `{shuffle_summary['shuffle_test_mse_median']:.6f}`
- Observed test MSE gain vs shuffled median: `+{shuffle_summary['observed_test_mse_gain_vs_shuffle_median_pct']:.4f}%`
- Observed lower-is-better test rank fraction among shuffles: `{shuffle_summary['observed_test_rank_fraction_lower_is_better']:.4f}`

Interpretation:

This is still a prediction-level Traffic performance branch, not post-hoc dynamic CACI calibration. The adaptive alpha diagnostics make the ensemble less arbitrary by showing how static-causal weight varies by target and by adding a shuffled-target negative control.

{end}
"""
    if start in text and end in text:
        before = text.split(start, 1)[0].rstrip()
        after = text.split(end, 1)[1].lstrip()
        text = f"{before}\n\n{section}\n{after}"
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    readme_path.write_text(text, encoding="utf-8")


def write_sub_readme(out_dir: Path, selected: dict, adaptive_summary: dict, shuffle_summary: dict) -> None:
    content = f"""# Traffic96 Adaptive-Alpha Ensemble Evidence

Generated: 2026-05-06

This subpackage extends the existing Traffic prediction-level ensemble with validation-estimated adaptive alpha.

## Selection

- Selected ensemble: `blend_baseline_static_alpha_variable_shrink`
- Selection rule: validation MSE with non-negative MAE gain guard.
- Test split is used only once for final evaluation.

## Key Results

- Global closed-form alpha: `{adaptive_summary['alpha_global_clipped']:.6f}`
- Per-variable alpha mean/std: `{adaptive_summary['var_alpha_mean']:.6f} / {adaptive_summary['var_alpha_std']:.6f}`
- Validation MSE/MAE: `{selected['val_mse']:.6f} / {selected['val_mae']:.6f}`
- Test MSE/MAE: `{selected['test_mse']:.6f} / {selected['test_mae']:.6f}`
- Test gain vs best single: MSE `+{selected['test_mse_gain_vs_best_single_pct']:.4f}%`, MAE `+{selected['test_mae_gain_vs_best_single_pct']:.4f}%`

## Negative Control

The shuffled-alpha negative control randomly permutes the same 862 alpha values across targets. It preserves the alpha distribution but breaks target identity.

- Shuffled median test MSE: `{shuffle_summary['shuffle_test_mse_median']:.6f}`
- Observed test MSE: `{shuffle_summary['observed_test_mse']:.6f}`
- Observed gain vs shuffled median: `+{shuffle_summary['observed_test_mse_gain_vs_shuffle_median_pct']:.4f}%`

## Files

- `raw_outputs/`: direct outputs from `traffic_existing_prediction_ensemble.py --tag adaptive_alpha`.
- `tables/{PREFIX}_target_diagnostics.csv`: per-target alpha, validation/test gains, and PCMCI graph metrics.
- `tables/{PREFIX}_top_alpha_targets.csv`: highest-alpha targets for mechanism inspection.
- `tables/{PREFIX}_alignment_summary.csv`: Spearman/Pearson alignment diagnostics.
- `tables/{PREFIX}_shuffled_negative_control.csv`: shuffled-alpha MSE diagnostics.
- `figures/{PREFIX}_alpha_distribution.png`: alpha distribution.
- `figures/{PREFIX}_alpha_gain_scatter.png`: alpha vs validation gain.
- `figures/{PREFIX}_alpha_graph_scatter.png`: alpha vs PCMCI parent strength.

## Reporting Boundary

Use this as Traffic performance evidence. Do not describe it as post-hoc dynamic CACI calibration gain.
"""
    (out_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    package_dir = args.package_dir
    out_dir = package_dir / "performance" / "adaptive_alpha_ensemble"
    raw_dir = out_dir / "raw_outputs"
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    for path in [raw_dir, table_dir, fig_dir]:
        path.mkdir(parents=True, exist_ok=True)

    manifest = read_json(args.adaptive_dir / f"{PREFIX}_manifest.json")
    adaptive_summary = read_json(args.adaptive_dir / f"{PREFIX}_adaptive_alpha_summary.json")
    selected = pd.read_csv(args.adaptive_dir / f"{PREFIX}_selected_test_summary.csv").iloc[0].to_dict()
    alpha_df = pd.read_csv(args.adaptive_dir / f"{PREFIX}_variable_alpha.csv")
    alpha_df.attrs["alpha_global"] = adaptive_summary["alpha_global_clipped"]
    alpha = alpha_df["alpha_shrunk"].to_numpy(dtype=np.float64)
    candidates = manifest["candidates"]

    print("[Stage] copy adaptive raw outputs", flush=True)
    raw_files = copy_raw_outputs(args.adaptive_dir, raw_dir)

    print("[Stage] compute validation sufficient statistics", flush=True)
    val_stats = split_sufficient_stats(candidates, alpha, "val", args.chunk_size)
    print("[Stage] compute test sufficient statistics", flush=True)
    test_stats = split_sufficient_stats(candidates, alpha, "test", args.chunk_size)

    print("[Stage] build target diagnostics", flush=True)
    graph_df = graph_frame(args.interface_dir)
    diag = target_diagnostics(alpha_df, val_stats, test_stats, graph_df)
    diag_path = table_dir / f"{PREFIX}_target_diagnostics.csv"
    diag.to_csv(diag_path, index=False)

    top_targets = diag.sort_values(["alpha_shrunk", "val_adaptive_mse_gain_vs_baseline_pct"], ascending=[False, False])
    top_path = table_dir / f"{PREFIX}_top_alpha_targets.csv"
    top_targets.head(args.top_k).to_csv(top_path, index=False)

    print("[Stage] run shuffled-alpha negative control", flush=True)
    shuffle_df, shuffle_summary = shuffled_negative_control(
        val_stats=val_stats,
        test_stats=test_stats,
        alpha=alpha,
        shuffle_count=args.shuffle_count,
        seed=args.shuffle_seed,
    )
    shuffle_path = table_dir / f"{PREFIX}_shuffled_negative_control.csv"
    shuffle_df.to_csv(shuffle_path, index=False)
    write_json(table_dir / f"{PREFIX}_shuffled_negative_control_summary.json", shuffle_summary)

    align = alignment_summary(diag, shuffle_summary)
    align_path = table_dir / f"{PREFIX}_alignment_summary.csv"
    align.to_csv(align_path, index=False)

    print("[Stage] render figures", flush=True)
    figure_files = make_plots(diag, adaptive_summary, fig_dir)

    copied_files = {
        "raw_outputs": raw_files,
        "tables": [
            diag_path.name,
            top_path.name,
            shuffle_path.name,
            f"{PREFIX}_shuffled_negative_control_summary.json",
            align_path.name,
        ],
        "figures": figure_files,
        "readme": ["README.md"],
    }

    adaptive_claim = {
        "status": "performance_branch",
        "claim": "Validation-estimated global and per-variable adaptive alpha improves the Traffic prediction-level ensemble while preserving validation-only selection.",
        "selected_ensemble": str(selected["ensemble"]),
        "selection_reason": str(selected["selection_reason"]),
        "reference_best_single": str(selected["reference_best_single"]),
        "alpha_global_closed_form": float(adaptive_summary["alpha_global_clipped"]),
        "alpha_variable_mean": float(adaptive_summary["var_alpha_mean"]),
        "alpha_variable_std": float(adaptive_summary["var_alpha_std"]),
        "val_mse": float(selected["val_mse"]),
        "val_mae": float(selected["val_mae"]),
        "val_mse_gain_vs_best_single_pct": float(selected["val_mse_gain_vs_best_single_pct"]),
        "val_mae_gain_vs_best_single_pct": float(selected["val_mae_gain_vs_best_single_pct"]),
        "test_mse": float(selected["test_mse"]),
        "test_mae": float(selected["test_mae"]),
        "test_mse_gain_vs_best_single_pct": float(selected["test_mse_gain_vs_best_single_pct"]),
        "test_mae_gain_vs_best_single_pct": float(selected["test_mae_gain_vs_best_single_pct"]),
        "shuffle_count": int(args.shuffle_count),
        "observed_test_mse_gain_vs_shuffle_median_pct": float(
            shuffle_summary["observed_test_mse_gain_vs_shuffle_median_pct"]
        ),
        "observed_test_rank_fraction_lower_is_better": float(
            shuffle_summary["observed_test_rank_fraction_lower_is_better"]
        ),
        "repo_head_at_packaging": git_head(),
    }

    print("[Stage] update package docs", flush=True)
    write_sub_readme(out_dir, selected, adaptive_summary, shuffle_summary)
    update_root_readme(package_dir, selected, adaptive_summary, shuffle_summary)
    update_package_manifest(package_dir, adaptive_claim, copied_files)

    print(f"[Done] adaptive-alpha evidence package: {out_dir}", flush=True)
    print(
        "[Summary] "
        f"global_alpha={adaptive_summary['alpha_global_clipped']:.6f} "
        f"selected_test_mse={float(selected['test_mse']):.6f} "
        f"shuffle_median_test_mse={shuffle_summary['shuffle_test_mse_median']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

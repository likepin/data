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


DEFAULT_STAGE2_DIR = Path(
    r"C:\Users\cyl\Desktop\data\deltaA_signal_audit\traffic96_existing_prediction_ensemble_stage2_light_seed2026"
)
DEFAULT_INTERFACE_DIR = Path(r"C:\Users\cyl\Desktop\data\interfaces\Traffic_graph_interface_parcorr")
DEFAULT_RISK_DIR = Path(
    r"C:\Users\cyl\Desktop\data\mechanism_evidence\traffic96_stage3_lambda_three_source_20260507\mechanism\risk_windows"
)
DEFAULT_OUT_DIR = Path(
    r"C:\Users\cyl\Desktop\data\mechanism_evidence\traffic96_top_causal_corridors_20260507"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Close Traffic96 mechanism evidence around Stage2 alpha_i, graph centrality, and per-target gain."
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="traffic96_static")
    parser.add_argument("--stage2-dir", type=Path, default=DEFAULT_STAGE2_DIR)
    parser.add_argument("--stage2-prefix", default="traffic96_static_stage2_light_seed2026")
    parser.add_argument("--interface-dir", type=Path, default=DEFAULT_INTERFACE_DIR)
    parser.add_argument("--risk-dir", type=Path, default=DEFAULT_RISK_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--top-pcts", default="0.01,0.05")
    parser.add_argument("--top-k-nodes", type=int, default=32)
    parser.add_argument("--top-k-neighbors", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=300)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown_table(df: pd.DataFrame, path: Path, max_rows: int | None = None) -> None:
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)

    def fmt(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value)

    lines = ["| " + " | ".join(view.columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(view.columns)) + " |")
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in view.columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_alpha_frame(stage2_dir: Path, prefix: str) -> pd.DataFrame:
    path = stage2_dir / f"{prefix}_variable_alpha.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"target_index", "alpha_raw", "alpha_clipped", "alpha_shrunk", "denominator", "reliability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df.sort_values("target_index").reset_index(drop=True)


def load_best_single_name(stage2_dir: Path, prefix: str) -> str:
    path = stage2_dir / f"{prefix}_selected_test_summary.csv"
    if not path.exists():
        return "static_p1"
    df = pd.read_csv(path)
    if "reference_best_single" not in df or df.empty:
        return "static_p1"
    value = str(df.iloc[0]["reference_best_single"])
    return value if value and value != "nan" else "static_p1"


def graph_metrics(interface_dir: Path, n_vars: int) -> pd.DataFrame:
    support_path = interface_dir / "support.npy"
    weight_path = interface_dir / "a_base_agg.npy"
    if not support_path.exists() or not weight_path.exists():
        raise FileNotFoundError(f"Missing graph arrays under {interface_dir}")

    support = np.asarray(np.load(support_path, mmap_mode="r") > 0)
    weights = np.asarray(np.load(weight_path, mmap_mode="r"), dtype=np.float64)
    if support.shape != (n_vars, n_vars) or weights.shape != (n_vars, n_vars):
        raise ValueError(f"Unexpected graph shape: support={support.shape}, weights={weights.shape}, n_vars={n_vars}")

    support = support.copy()
    weighted_mask = np.abs(weights) > 0
    weighted_mask = weighted_mask.copy()
    np.fill_diagonal(support, False)
    np.fill_diagonal(weighted_mask, False)

    abs_weights = np.abs(weights)
    abs_weights_no_diag = abs_weights.copy()
    np.fill_diagonal(abs_weights_no_diag, 0.0)

    return pd.DataFrame(
        {
            "target_index": np.arange(n_vars, dtype=np.int64),
            "support_in_degree": support.sum(axis=1).astype(np.int64),
            "support_out_degree": support.sum(axis=0).astype(np.int64),
            "support_total_degree": (support.sum(axis=1) + support.sum(axis=0)).astype(np.int64),
            "ridge_in_degree": weighted_mask.sum(axis=1).astype(np.int64),
            "ridge_out_degree": weighted_mask.sum(axis=0).astype(np.int64),
            "ridge_total_degree": (weighted_mask.sum(axis=1) + weighted_mask.sum(axis=0)).astype(np.int64),
            "weighted_in_degree": abs_weights_no_diag.sum(axis=1),
            "weighted_out_degree": abs_weights_no_diag.sum(axis=0),
            "weighted_total_degree": abs_weights_no_diag.sum(axis=1) + abs_weights_no_diag.sum(axis=0),
        }
    )


def open_prediction_arrays(candidates: list[dict], split: str) -> tuple[list[np.ndarray], np.ndarray]:
    pred_arrays = [np.load(pred_path(candidate, split), mmap_mode="r") for candidate in candidates]
    true = np.load(true_path(candidates[0], split), mmap_mode="r")
    expected_shape = true.shape
    for candidate, pred in zip(candidates, pred_arrays):
        if pred.shape != expected_shape:
            raise RuntimeError(f"Unexpected pred shape for {candidate['candidate']}: {pred.shape} vs {expected_shape}")
    return pred_arrays, true


def per_target_metrics(
    candidates: list[dict],
    alpha: np.ndarray,
    split: str,
    best_single_name: str,
    chunk_size: int,
    progress_every: int,
) -> pd.DataFrame:
    baseline_idx, static_idx = group_indices(candidates)
    name_to_idx = {candidate["candidate"]: idx for idx, candidate in enumerate(candidates)}
    if best_single_name not in name_to_idx:
        raise ValueError(f"Best single candidate {best_single_name!r} not found in candidates: {sorted(name_to_idx)}")
    best_single_idx = int(name_to_idx[best_single_name])

    pred_arrays, true = open_prediction_arrays(candidates, split)
    n_samples, n_horizon, n_vars = true.shape
    if alpha.shape != (n_vars,):
        raise ValueError(f"alpha shape mismatch: {alpha.shape} vs n_vars={n_vars}")

    sums = {
        name: np.zeros(n_vars, dtype=np.float64)
        for name in [
            "baseline_sse",
            "baseline_sae",
            "static_mean_sse",
            "static_mean_sae",
            "best_single_sse",
            "best_single_sae",
            "stage2_sse",
            "stage2_sae",
            "correction_energy",
            "baseline_err_dot_correction",
        ]
    }
    alpha_view = alpha.reshape(1, 1, -1).astype(np.float32)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        y = np.asarray(true[start:end], dtype=np.float32)
        baseline_mean = group_mean_chunk(pred_arrays, baseline_idx, start, end)
        static_mean = group_mean_chunk(pred_arrays, static_idx, start, end)
        best_single = np.asarray(pred_arrays[best_single_idx][start:end], dtype=np.float32)
        correction = static_mean - baseline_mean
        stage2 = baseline_mean + alpha_view * correction

        for prefix, pred in [
            ("baseline", baseline_mean),
            ("static_mean", static_mean),
            ("best_single", best_single),
            ("stage2", stage2),
        ]:
            err = y - pred
            sums[f"{prefix}_sse"] += np.square(err, dtype=np.float32).sum(axis=(0, 1), dtype=np.float64)
            sums[f"{prefix}_sae"] += np.abs(err).sum(axis=(0, 1), dtype=np.float64)

        baseline_err = y - baseline_mean
        sums["correction_energy"] += np.square(correction, dtype=np.float32).sum(axis=(0, 1), dtype=np.float64)
        sums["baseline_err_dot_correction"] += (baseline_err * correction).sum(axis=(0, 1), dtype=np.float64)

        if progress_every > 0 and (end % progress_every == 0 or end == n_samples):
            print(f"[{split}] processed {end}/{n_samples}", flush=True)

    count = float(n_samples * n_horizon)
    frame = pd.DataFrame({"target_index": np.arange(n_vars, dtype=np.int64), "split": split})
    for name, values in sums.items():
        frame[name] = values
    for prefix in ["baseline", "static_mean", "best_single", "stage2"]:
        frame[f"{prefix}_mse"] = frame[f"{prefix}_sse"] / count
        frame[f"{prefix}_mae"] = frame[f"{prefix}_sae"] / count

    frame["stage2_mse_gain_vs_best_single_pct"] = [
        pct_gain(ref, cur) for ref, cur in zip(frame["best_single_mse"], frame["stage2_mse"])
    ]
    frame["stage2_mae_gain_vs_best_single_pct"] = [
        pct_gain(ref, cur) for ref, cur in zip(frame["best_single_mae"], frame["stage2_mae"])
    ]
    frame["stage2_mse_gain_vs_static_mean_pct"] = [
        pct_gain(ref, cur) for ref, cur in zip(frame["static_mean_mse"], frame["stage2_mse"])
    ]
    frame["stage2_mae_gain_vs_static_mean_pct"] = [
        pct_gain(ref, cur) for ref, cur in zip(frame["static_mean_mae"], frame["stage2_mae"])
    ]
    frame["stage2_mse_gain_vs_baseline_pct"] = [
        pct_gain(ref, cur) for ref, cur in zip(frame["baseline_mse"], frame["stage2_mse"])
    ]
    frame["stage2_mae_gain_vs_baseline_pct"] = [
        pct_gain(ref, cur) for ref, cur in zip(frame["baseline_mae"], frame["stage2_mae"])
    ]
    frame["correction_energy_share_pct"] = 100.0 * frame["correction_energy"] / float(frame["correction_energy"].sum())
    return frame


def rank_pct(series: pd.Series, ascending: bool = True) -> pd.Series:
    return series.rank(method="average", pct=True, ascending=ascending)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    positive_test_gain = out["test_stage2_mse_gain_vs_best_single_pct"].clip(lower=0.0)
    out["alpha_rank_pct"] = rank_pct(out["alpha_shrunk"])
    out["weighted_degree_rank_pct"] = rank_pct(out["weighted_total_degree"])
    out["energy_rank_pct"] = rank_pct(out["val_correction_energy"])
    out["test_gain_rank_pct"] = rank_pct(positive_test_gain)
    out["static_corridor_score"] = out[["alpha_rank_pct", "weighted_degree_rank_pct", "test_gain_rank_pct"]].mean(axis=1)
    out["stress_corridor_score"] = out[["energy_rank_pct", "weighted_degree_rank_pct"]].mean(axis=1)
    out["overall_corridor_score"] = out[
        ["alpha_rank_pct", "weighted_degree_rank_pct", "energy_rank_pct", "test_gain_rank_pct"]
    ].mean(axis=1)
    return out


def top_by_pct(df: pd.DataFrame, column: str, pct: float) -> pd.DataFrame:
    n = max(1, int(np.ceil(len(df) * pct)))
    return df.sort_values(column, ascending=False).head(n).copy()


def overlap_rows(df: pd.DataFrame, top_pcts: list[float]) -> pd.DataFrame:
    rows = []
    for pct in top_pcts:
        alpha_top = set(top_by_pct(df, "alpha_shrunk", pct)["target_index"].astype(int))
        degree_top = set(top_by_pct(df, "weighted_total_degree", pct)["target_index"].astype(int))
        energy_top = set(top_by_pct(df, "val_correction_energy", pct)["target_index"].astype(int))
        gain_top = set(top_by_pct(df, "test_stage2_mse_gain_vs_best_single_pct", pct)["target_index"].astype(int))
        rows.extend(
            [
                {
                    "top_pct": pct,
                    "set_a": "alpha_shrunk",
                    "set_b": "weighted_total_degree",
                    "overlap_count": len(alpha_top & degree_top),
                    "set_size": len(alpha_top),
                    "overlap_pct_of_set": 100.0 * len(alpha_top & degree_top) / len(alpha_top),
                    "nodes": ",".join(str(x) for x in sorted(alpha_top & degree_top)),
                },
                {
                    "top_pct": pct,
                    "set_a": "alpha_shrunk",
                    "set_b": "val_correction_energy",
                    "overlap_count": len(alpha_top & energy_top),
                    "set_size": len(alpha_top),
                    "overlap_pct_of_set": 100.0 * len(alpha_top & energy_top) / len(alpha_top),
                    "nodes": ",".join(str(x) for x in sorted(alpha_top & energy_top)),
                },
                {
                    "top_pct": pct,
                    "set_a": "alpha_shrunk",
                    "set_b": "test_stage2_gain",
                    "overlap_count": len(alpha_top & gain_top),
                    "set_size": len(alpha_top),
                    "overlap_pct_of_set": 100.0 * len(alpha_top & gain_top) / len(alpha_top),
                    "nodes": ",".join(str(x) for x in sorted(alpha_top & gain_top)),
                },
                {
                    "top_pct": pct,
                    "set_a": "val_correction_energy",
                    "set_b": "test_stage2_gain",
                    "overlap_count": len(energy_top & gain_top),
                    "set_size": len(energy_top),
                    "overlap_pct_of_set": 100.0 * len(energy_top & gain_top) / len(energy_top),
                    "nodes": ",".join(str(x) for x in sorted(energy_top & gain_top)),
                },
            ]
        )
    return pd.DataFrame(rows)


def edge_rows(interface_dir: Path, nodes: list[int], top_k_neighbors: int) -> pd.DataFrame:
    weights = np.asarray(np.load(interface_dir / "a_base_agg.npy", mmap_mode="r"), dtype=np.float64)
    abs_weights = np.abs(weights)
    rows = []
    for node in nodes:
        parent_order = np.argsort(abs_weights[node, :])[::-1]
        child_order = np.argsort(abs_weights[:, node])[::-1]
        parent_count = 0
        for source in parent_order:
            source = int(source)
            if source == node or abs_weights[node, source] <= 0:
                continue
            rows.append(
                {
                    "focus_node": int(node),
                    "neighbor": source,
                    "direction": "neighbor_to_focus",
                    "target": int(node),
                    "source": source,
                    "weight": float(weights[node, source]),
                    "abs_weight": float(abs_weights[node, source]),
                }
            )
            parent_count += 1
            if parent_count >= top_k_neighbors:
                break
        child_count = 0
        for target in child_order:
            target = int(target)
            if target == node or abs_weights[target, node] <= 0:
                continue
            rows.append(
                {
                    "focus_node": int(node),
                    "neighbor": target,
                    "direction": "focus_to_neighbor",
                    "target": target,
                    "source": int(node),
                    "weight": float(weights[target, node]),
                    "abs_weight": float(abs_weights[target, node]),
                }
            )
            child_count += 1
            if child_count >= top_k_neighbors:
                break
    return pd.DataFrame(rows)


def maybe_write_plots(df: pd.DataFrame, out_dir: Path) -> list[str]:
    paths: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional diagnostic output
        print(f"[Plot] skipped: {exc}", flush=True)
        return paths

    def scatter(x: str, y: str, color: str, path: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)
        points = ax.scatter(df[x], df[y], c=df[color], s=18, cmap="viridis", alpha=0.75, linewidths=0)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
        cbar = fig.colorbar(points, ax=ax)
        cbar.set_label(color)
        fig.tight_layout()
        fig.savefig(out_dir / path)
        plt.close(fig)
        paths.append(path)

    scatter(
        "weighted_total_degree",
        "alpha_shrunk",
        "test_stage2_mse_gain_vs_best_single_pct",
        "alpha_vs_weighted_degree.png",
        "Traffic96 alpha_i vs weighted graph degree",
    )
    scatter(
        "val_correction_energy",
        "test_stage2_mse_gain_vs_best_single_pct",
        "alpha_shrunk",
        "gain_vs_correction_energy.png",
        "Traffic96 local gain vs correction energy",
    )
    return paths


def write_readme(
    out_dir: Path,
    summary: dict,
    correlations: pd.DataFrame,
    top_static: pd.DataFrame,
    top_stress: pd.DataFrame,
    top_energy: pd.DataFrame,
    key_snapshot: pd.DataFrame,
    overlap: pd.DataFrame,
) -> None:
    corr_map = {
        f"{row['metric_a']}__{row['metric_b']}": float(row["spearman"])
        for _, row in correlations.iterrows()
    }
    top_static_nodes = ", ".join(str(int(x)) for x in top_static["target_index"].head(10))
    top_stress_nodes = ", ".join(str(int(x)) for x in top_stress["target_index"].head(10))
    top_energy_nodes = ", ".join(str(int(x)) for x in top_energy["target_index"].head(10))
    key_840 = key_snapshot[key_snapshot["target_index"] == 840]
    key_840_text = "not present"
    if not key_840.empty:
        row_840 = key_840.iloc[0]
        key_840_text = (
            f"energy_share={float(row_840['val_correction_energy_share_pct']):.2f}%, "
            f"alpha={float(row_840['alpha_shrunk']):.4f}, "
            f"test_mse_gain={float(row_840['test_stage2_mse_gain_vs_best_single_pct']):.4f}%"
        )
    alpha_degree = corr_map.get("alpha_shrunk__weighted_total_degree", np.nan)
    alpha_gain = corr_map.get("alpha_shrunk__test_stage2_mse_gain_vs_best_single_pct", np.nan)
    energy_gain = corr_map.get("val_correction_energy__test_stage2_mse_gain_vs_best_single_pct", np.nan)
    overlap_alpha_degree_5 = overlap[
        (overlap["top_pct"].round(4) == 0.05) & (overlap["set_a"] == "alpha_shrunk") & (overlap["set_b"] == "weighted_total_degree")
    ]
    overlap_text = "n/a"
    if not overlap_alpha_degree_5.empty:
        row = overlap_alpha_degree_5.iloc[0]
        overlap_text = f"{int(row['overlap_count'])}/{int(row['set_size'])} ({float(row['overlap_pct_of_set']):.2f}%)"

    lines = [
        "# Traffic96 Top Causal Corridors Evidence",
        "",
        "Purpose:",
        "- Close the Traffic96 evidence around Stage2 variable alpha without adding new training or high-free-parameter search.",
        "- Separate static-anchor reliance, graph centrality, local performance gain, and window-level lambda evidence.",
        "",
        "Main readout:",
        f"- Stage2 test MSE gain vs best single anchor: `{summary['stage2_test_mse_gain_vs_best_single_pct']:.4f}%`.",
        f"- Stage2 test MAE gain vs best single anchor: `{summary['stage2_test_mae_gain_vs_best_single_pct']:.4f}%`.",
        f"- Top static-anchor corridor nodes by composite score: `{top_static_nodes}`.",
        f"- Top stress/correction-energy nodes by energy+degree score: `{top_stress_nodes}`.",
        f"- Top pure correction-energy nodes: `{top_energy_nodes}`.",
        f"- Target 840 snapshot: `{key_840_text}`.",
        "",
        "Correlation diagnostics:",
        f"- Spearman(alpha_i, weighted_total_degree): `{alpha_degree:.4f}`.",
        f"- Spearman(alpha_i, per-target test MSE gain): `{alpha_gain:.4f}`.",
        f"- Spearman(correction_energy, per-target test MSE gain): `{energy_gain:.4f}`.",
        f"- Top-5% overlap between alpha_i and weighted graph degree: `{overlap_text}`.",
        "",
        "Interpretation:",
        "- If high-alpha nodes do not strongly overlap with high correction-energy nodes, this is not a failure: alpha_i measures static-anchor reliance, while correction energy measures baseline/static disagreement scale.",
        "- Because lambda/gamma is window-level in the current protocol, this package does not claim variable-level risk response.",
        "- The strongest defensible claim is that Stage2 obtains its main Traffic gain from variable-specific static-anchor allocation; Stage3 remains a weak positive add-on rather than a reliable high-risk-window attack.",
        "",
        "Files:",
        "- `traffic96_target_node_metrics.csv`: complete per-variable table.",
        "- `traffic96_top_static_corridor_nodes.csv`: high alpha_i + graph degree + local gain ranking.",
        "- `traffic96_top_stress_nodes.csv`: correction-energy + graph degree ranking.",
        "- `traffic96_top_correction_energy_nodes.csv`: pure correction-energy ranking.",
        "- `traffic96_top_alpha_nodes.csv`: pure alpha_i ranking.",
        "- `traffic96_key_node_snapshot.csv`: union of top static, stress, correction-energy nodes plus Target 840.",
        "- `traffic96_one_hop_edges_top_static_corridors.csv`: weighted one-hop edge list around top corridor nodes.",
        "- `traffic96_corridor_overlap.csv`: top-set overlaps.",
        "- `traffic96_corridor_correlations.csv`: Spearman diagnostics.",
        "- `manifest.json`: source paths and summary metrics.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    profile = dict(PROFILES[args.profile])
    candidates = load_candidates(profile)
    best_single_name = load_best_single_name(args.stage2_dir, args.stage2_prefix)
    alpha_df = load_alpha_frame(args.stage2_dir, args.stage2_prefix)
    n_vars = len(alpha_df)
    alpha = alpha_df["alpha_shrunk"].to_numpy(dtype=np.float64)

    print("[Stage] graph metrics", flush=True)
    graph_df = graph_metrics(args.interface_dir, n_vars)

    print("[Stage] validation per-target metrics", flush=True)
    val_metrics = per_target_metrics(
        candidates=candidates,
        alpha=alpha,
        split="val",
        best_single_name=best_single_name,
        chunk_size=args.chunk_size,
        progress_every=args.progress_every,
    )

    print("[Stage] test per-target metrics", flush=True)
    test_metrics = per_target_metrics(
        candidates=candidates,
        alpha=alpha,
        split="test",
        best_single_name=best_single_name,
        chunk_size=args.chunk_size,
        progress_every=args.progress_every,
    )

    val_prefix = val_metrics.drop(columns=["split"]).add_prefix("val_").rename(columns={"val_target_index": "target_index"})
    test_prefix = test_metrics.drop(columns=["split"]).add_prefix("test_").rename(columns={"test_target_index": "target_index"})
    target_df = alpha_df.merge(graph_df, on="target_index", how="left").merge(val_prefix, on="target_index").merge(test_prefix, on="target_index")
    target_df = add_scores(target_df)

    target_df.to_csv(args.out_dir / "traffic96_target_node_metrics.csv", index=False)
    write_markdown_table(
        target_df.sort_values("overall_corridor_score", ascending=False)[
            [
                "target_index",
                "alpha_shrunk",
                "weighted_total_degree",
                "val_correction_energy",
                "val_correction_energy_share_pct",
                "test_stage2_mse_gain_vs_best_single_pct",
                "test_stage2_mae_gain_vs_best_single_pct",
                "overall_corridor_score",
            ]
        ],
        args.out_dir / "traffic96_target_node_metrics_top_overall.md",
        max_rows=args.top_k_nodes,
    )

    top_static = target_df.sort_values("static_corridor_score", ascending=False).head(args.top_k_nodes)
    top_stress = target_df.sort_values("stress_corridor_score", ascending=False).head(args.top_k_nodes)
    top_energy = target_df.sort_values("val_correction_energy", ascending=False).head(args.top_k_nodes)
    top_alpha = target_df.sort_values("alpha_shrunk", ascending=False).head(args.top_k_nodes)
    for name, frame in [
        ("traffic96_top_static_corridor_nodes", top_static),
        ("traffic96_top_stress_nodes", top_stress),
        ("traffic96_top_correction_energy_nodes", top_energy),
        ("traffic96_top_alpha_nodes", top_alpha),
    ]:
        cols = [
            "target_index",
            "alpha_shrunk",
            "reliability",
            "weighted_total_degree",
            "ridge_total_degree",
            "support_total_degree",
            "val_correction_energy",
            "val_correction_energy_share_pct",
            "test_stage2_mse_gain_vs_best_single_pct",
            "test_stage2_mae_gain_vs_best_single_pct",
            "static_corridor_score",
            "stress_corridor_score",
            "overall_corridor_score",
        ]
        frame[cols].to_csv(args.out_dir / f"{name}.csv", index=False)
        write_markdown_table(frame[cols], args.out_dir / f"{name}.md")

    key_node_ids = set(top_static["target_index"].astype(int).head(10))
    key_node_ids.update(top_stress["target_index"].astype(int).head(10))
    key_node_ids.update(top_energy["target_index"].astype(int).head(10))
    if (target_df["target_index"] == 840).any():
        key_node_ids.add(840)
    key_snapshot = target_df[target_df["target_index"].astype(int).isin(key_node_ids)].copy()
    key_snapshot["in_top_static_10"] = key_snapshot["target_index"].astype(int).isin(
        set(top_static["target_index"].astype(int).head(10))
    )
    key_snapshot["in_top_stress_10"] = key_snapshot["target_index"].astype(int).isin(
        set(top_stress["target_index"].astype(int).head(10))
    )
    key_snapshot["in_top_energy_10"] = key_snapshot["target_index"].astype(int).isin(
        set(top_energy["target_index"].astype(int).head(10))
    )
    key_snapshot = key_snapshot.sort_values(
        ["in_top_static_10", "in_top_stress_10", "in_top_energy_10", "overall_corridor_score"],
        ascending=[False, False, False, False],
    )
    key_cols = [
        "target_index",
        "in_top_static_10",
        "in_top_stress_10",
        "in_top_energy_10",
        "alpha_shrunk",
        "reliability",
        "weighted_total_degree",
        "ridge_total_degree",
        "val_correction_energy_share_pct",
        "test_stage2_mse_gain_vs_best_single_pct",
        "test_stage2_mae_gain_vs_best_single_pct",
        "static_corridor_score",
        "stress_corridor_score",
        "overall_corridor_score",
    ]
    key_snapshot[key_cols].to_csv(args.out_dir / "traffic96_key_node_snapshot.csv", index=False)
    write_markdown_table(key_snapshot[key_cols], args.out_dir / "traffic96_key_node_snapshot.md")

    top_pcts = parse_float_list(args.top_pcts)
    overlap = overlap_rows(target_df, top_pcts)
    overlap.to_csv(args.out_dir / "traffic96_corridor_overlap.csv", index=False)
    write_markdown_table(overlap, args.out_dir / "traffic96_corridor_overlap.md")

    corr_pairs = [
        ("alpha_shrunk", "weighted_total_degree"),
        ("alpha_shrunk", "val_correction_energy"),
        ("alpha_shrunk", "test_stage2_mse_gain_vs_best_single_pct"),
        ("weighted_total_degree", "test_stage2_mse_gain_vs_best_single_pct"),
        ("val_correction_energy", "test_stage2_mse_gain_vs_best_single_pct"),
        ("val_correction_energy", "weighted_total_degree"),
    ]
    correlations = pd.DataFrame(
        [
            {
                "metric_a": a,
                "metric_b": b,
                "spearman": float(target_df[[a, b]].corr(method="spearman").iloc[0, 1]),
                "pearson": float(target_df[[a, b]].corr(method="pearson").iloc[0, 1]),
            }
            for a, b in corr_pairs
        ]
    )
    correlations.to_csv(args.out_dir / "traffic96_corridor_correlations.csv", index=False)
    write_markdown_table(correlations, args.out_dir / "traffic96_corridor_correlations.md")

    focus_nodes = top_static["target_index"].astype(int).head(args.top_k_nodes).tolist()
    edges = edge_rows(args.interface_dir, focus_nodes, args.top_k_neighbors)
    edges.to_csv(args.out_dir / "traffic96_one_hop_edges_top_static_corridors.csv", index=False)
    write_markdown_table(edges.head(128), args.out_dir / "traffic96_one_hop_edges_top_static_corridors.md")

    plot_files = [] if args.skip_plots else maybe_write_plots(target_df, args.out_dir)

    # MSE/MAE denominators are identical across target rows, so aggregate from per-target means.
    summary = {
        "profile": args.profile,
        "stage2_dir": str(args.stage2_dir),
        "interface_dir": str(args.interface_dir),
        "risk_dir": str(args.risk_dir),
        "best_single_reference": best_single_name,
        "n_targets": int(n_vars),
        "candidate_count": len(candidates),
        "top_pcts": top_pcts,
        "top_k_nodes": int(args.top_k_nodes),
        "stage2_test_mse": float(test_metrics["stage2_mse"].mean()),
        "best_single_test_mse": float(test_metrics["best_single_mse"].mean()),
        "stage2_test_mae": float(test_metrics["stage2_mae"].mean()),
        "best_single_test_mae": float(test_metrics["best_single_mae"].mean()),
        "stage2_test_mse_gain_vs_best_single_pct": pct_gain(
            float(test_metrics["best_single_mse"].mean()), float(test_metrics["stage2_mse"].mean())
        ),
        "stage2_test_mae_gain_vs_best_single_pct": pct_gain(
            float(test_metrics["best_single_mae"].mean()), float(test_metrics["stage2_mae"].mean())
        ),
        "alpha_mean": float(alpha_df["alpha_shrunk"].mean()),
        "alpha_std": float(alpha_df["alpha_shrunk"].std(ddof=0)),
        "alpha_min": float(alpha_df["alpha_shrunk"].min()),
        "alpha_max": float(alpha_df["alpha_shrunk"].max()),
        "plot_files": plot_files,
    }
    write_json(args.out_dir / "manifest.json", summary)
    write_readme(args.out_dir, summary, correlations, top_static, top_stress, top_energy, key_snapshot, overlap)

    print(
        "[Done] "
        f"test_gain_mse={summary['stage2_test_mse_gain_vs_best_single_pct']:.4f}% "
        f"top_static_nodes={','.join(str(int(x)) for x in top_static['target_index'].head(5))}",
        flush=True,
    )
    print(f"[Wrote] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

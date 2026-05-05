from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def edge_entropy(weights: np.ndarray) -> tuple[float, float, float]:
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    n = int(weights.size)
    if total <= 0.0 or n <= 0:
        return 0.0, 0.0, 0.0
    p = weights.reshape(-1) / total
    p = p[p > 0.0]
    entropy = float(-np.sum(p * np.log(p)))
    return entropy, float(entropy / np.log(n)) if n > 1 else 0.0, float(np.exp(entropy))


def top_mass_share(weights: np.ndarray, k: int) -> float:
    flat = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(flat.sum())
    if total <= 0.0:
        return 0.0
    k = max(1, min(int(k), flat.size))
    idx = np.argpartition(flat, -k)[-k:]
    return float(flat[idx].sum() / total)


def summarize_group(
    delta: np.ndarray,
    indices: np.ndarray,
    eps: float,
    chunk_size: int,
    exclude_diagonal: bool,
) -> dict:
    n_windows, n_vars, _ = delta.shape
    abs_sum = np.zeros((n_vars, n_vars), dtype=np.float64)
    freq = np.zeros((n_vars, n_vars), dtype=np.int32)
    signed_sum = np.zeros((n_vars, n_vars), dtype=np.float64)
    rows = []
    for start in range(0, len(indices), int(chunk_size)):
        idx = indices[start : start + int(chunk_size)]
        block = np.asarray(delta[idx], dtype=np.float32)
        if exclude_diagonal:
            diag = np.arange(n_vars)
            block[:, diag, diag] = 0.0
        abs_block = np.abs(block, dtype=np.float32)
        nz = abs_block > eps
        abs_sum += abs_block.sum(axis=0, dtype=np.float64)
        signed_sum += block.sum(axis=0, dtype=np.float64)
        freq += nz.sum(axis=0, dtype=np.int32)
        rows.append(
            pd.DataFrame(
                {
                    "edge_count": nz.reshape(len(idx), -1).sum(axis=1),
                    "l1": abs_block.reshape(len(idx), -1).sum(axis=1),
                    "l2": np.sqrt((block.reshape(len(idx), -1).astype(np.float64) ** 2).sum(axis=1)),
                    "positive_edges": (block > eps).reshape(len(idx), -1).sum(axis=1),
                    "negative_edges": (block < -eps).reshape(len(idx), -1).sum(axis=1),
                }
            )
        )
    per_window = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    abs_in_by_target = abs_sum.sum(axis=1)
    abs_out_by_source = abs_sum.sum(axis=0)
    edge_ent, edge_ent_norm, effective_edges = edge_entropy(abs_sum)
    target_ent, target_ent_norm, effective_targets = edge_entropy(abs_in_by_target)
    source_ent, source_ent_norm, effective_sources = edge_entropy(abs_out_by_source)
    active_edges = int(np.sum(freq > 0))
    frequent_edges_10pct = int(np.sum(freq >= max(1, int(np.ceil(0.10 * len(indices))))))
    frequent_edges_50pct = int(np.sum(freq >= max(1, int(np.ceil(0.50 * len(indices))))))
    return {
        "n": int(len(indices)),
        "available_windows": int(n_windows),
        "edge_count_mean": float(per_window["edge_count"].mean()) if len(per_window) else float("nan"),
        "edge_count_std": float(per_window["edge_count"].std(ddof=0)) if len(per_window) else float("nan"),
        "positive_edges_mean": float(per_window["positive_edges"].mean()) if len(per_window) else float("nan"),
        "negative_edges_mean": float(per_window["negative_edges"].mean()) if len(per_window) else float("nan"),
        "l1_mean": float(per_window["l1"].mean()) if len(per_window) else float("nan"),
        "l1_std": float(per_window["l1"].std(ddof=0)) if len(per_window) else float("nan"),
        "l2_mean": float(per_window["l2"].mean()) if len(per_window) else float("nan"),
        "l2_std": float(per_window["l2"].std(ddof=0)) if len(per_window) else float("nan"),
        "active_edges_union": active_edges,
        "frequent_edges_10pct": frequent_edges_10pct,
        "frequent_edges_50pct": frequent_edges_50pct,
        "edge_entropy": edge_ent,
        "edge_entropy_norm": edge_ent_norm,
        "effective_edges": effective_edges,
        "target_entropy_norm": target_ent_norm,
        "effective_targets": effective_targets,
        "source_entropy_norm": source_ent_norm,
        "effective_sources": effective_sources,
        "top_10_edges_mass_share": top_mass_share(abs_sum, 10),
        "top_50_edges_mass_share": top_mass_share(abs_sum, 50),
        "top_100_edges_mass_share": top_mass_share(abs_sum, 100),
        "_abs_sum": abs_sum,
        "_signed_sum": signed_sum,
        "_freq": freq,
    }


def top_edges(group_name: str, summary: dict, columns: list[str], top_k: int) -> pd.DataFrame:
    abs_sum = summary["_abs_sum"]
    signed_mean = summary["_signed_sum"] / max(1, int(summary["n"]))
    freq = summary["_freq"] / max(1, int(summary["n"]))
    flat = abs_sum.reshape(-1)
    k = min(int(top_k), flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    n_vars = abs_sum.shape[0]
    rows = []
    for rank, flat_idx in enumerate(idx, start=1):
        tgt = int(flat_idx // n_vars)
        src = int(flat_idx % n_vars)
        rows.append(
            {
                "group": group_name,
                "rank": rank,
                "target": tgt,
                "source": src,
                "target_name": columns[tgt] if tgt < len(columns) else str(tgt),
                "source_name": columns[src] if src < len(columns) else str(src),
                "abs_delta_sum": float(abs_sum[tgt, src]),
                "signed_delta_mean": float(signed_mean[tgt, src]),
                "activation_frequency": float(freq[tgt, src]),
            }
        )
    return pd.DataFrame(rows)


def top_targets(group_name: str, summary: dict, columns: list[str], top_k: int) -> pd.DataFrame:
    abs_sum = summary["_abs_sum"]
    target_mass = abs_sum.sum(axis=1)
    total = float(target_mass.sum())
    k = min(int(top_k), target_mass.size)
    idx = np.argpartition(target_mass, -k)[-k:]
    idx = idx[np.argsort(target_mass[idx])[::-1]]
    rows = []
    for rank, target in enumerate(idx, start=1):
        edge_mass = abs_sum[int(target)]
        rows.append(
            {
                "group": group_name,
                "rank": rank,
                "target": int(target),
                "target_name": columns[int(target)] if int(target) < len(columns) else str(target),
                "abs_delta_sum": float(target_mass[int(target)]),
                "mass_share": float(target_mass[int(target)] / total) if total > 0.0 else 0.0,
                "active_source_count": int(np.sum(edge_mass > 0.0)),
                "top_source": int(np.argmax(edge_mass)),
                "top_source_mass": float(np.max(edge_mass)),
            }
        )
    return pd.DataFrame(rows)


def top_set(summary: dict, k: int) -> set[int]:
    flat = summary["_abs_sum"].reshape(-1)
    k = min(int(k), flat.size)
    return set(np.argpartition(flat, -k)[-k:].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare windowed DeltaA graph structure across validation folds.")
    parser.add_argument("--interface-dir", type=Path, required=True)
    parser.add_argument("--alignment-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="traffic96_graph_structure")
    parser.add_argument("--active-col", default="active_at_p_0.1")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--exclude-diagonal", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.interface_dir / "interface_manifest.json").read_text(encoding="utf-8"))
    columns = [str(c) for c in manifest["dataset_contract"]["columns"]]
    delta = np.load(args.interface_dir / "deltaA_val.npy", mmap_mode="r")
    alignment = pd.read_csv(args.alignment_csv)
    if len(alignment) != delta.shape[0]:
        raise RuntimeError(f"Alignment length {len(alignment)} does not match deltaA_val windows {delta.shape[0]}")
    alignment["active"] = alignment[args.active_col].astype(bool)

    group_indices = {
        "fold1_all": alignment.index[alignment["fold_id"].eq(1)].to_numpy(dtype=np.int64),
        "fold4_all": alignment.index[alignment["fold_id"].eq(4)].to_numpy(dtype=np.int64),
        "fold4_active": alignment.index[alignment["fold_id"].eq(4) & alignment["active"]].to_numpy(dtype=np.int64),
        "fold4_inactive": alignment.index[alignment["fold_id"].eq(4) & ~alignment["active"]].to_numpy(dtype=np.int64),
    }
    summaries = {}
    for name, indices in group_indices.items():
        print(f"[Group] {name} n={len(indices)}", flush=True)
        summaries[name] = summarize_group(
            delta,
            indices,
            eps=float(args.eps),
            chunk_size=int(args.chunk_size),
            exclude_diagonal=bool(args.exclude_diagonal),
        )

    public_rows = []
    for name, summary in summaries.items():
        public = {k: v for k, v in summary.items() if not k.startswith("_")}
        public["group"] = name
        public_rows.append(public)
    summary_df = pd.DataFrame(public_rows)

    compare_rows = []
    for left, right in [("fold1_all", "fold4_all"), ("fold1_all", "fold4_active"), ("fold4_inactive", "fold4_active")]:
        for k in [50, 100, 379]:
            a = top_set(summaries[left], k)
            b = top_set(summaries[right], k)
            compare_rows.append(
                {
                    "left": left,
                    "right": right,
                    "top_k": k,
                    "intersection": len(a & b),
                    "union": len(a | b),
                    "jaccard": float(len(a & b) / len(a | b)) if a or b else 0.0,
                }
            )
    compare_df = pd.DataFrame(compare_rows)
    top_df = pd.concat(
        [top_edges(name, summary, columns=columns, top_k=int(args.top_k)) for name, summary in summaries.items()],
        ignore_index=True,
    )
    top_targets_df = pd.concat(
        [top_targets(name, summary, columns=columns, top_k=int(args.top_k)) for name, summary in summaries.items()],
        ignore_index=True,
    )

    summary_path = args.out_dir / f"{args.prefix}_summary.csv"
    compare_path = args.out_dir / f"{args.prefix}_topedge_jaccard.csv"
    top_path = args.out_dir / f"{args.prefix}_top_edges.csv"
    top_targets_path = args.out_dir / f"{args.prefix}_top_targets.csv"
    summary_df.to_csv(summary_path, index=False)
    compare_df.to_csv(compare_path, index=False)
    top_df.to_csv(top_path, index=False)
    top_targets_df.to_csv(top_targets_path, index=False)
    print(f"[Done] wrote {summary_path}")
    print(f"[Done] wrote {compare_path}")
    print(f"[Done] wrote {top_path}")
    print(f"[Done] wrote {top_targets_path}")


if __name__ == "__main__":
    main()

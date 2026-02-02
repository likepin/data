import os
import argparse

import numpy as np

from step5_utils import (
    read_json,
    find_true_change_adj,
    find_pred_change_adj,
    edges_from_adj,
    confusion,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="synthetic_step3_v2")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()

    data_dir = args.data_dir
    cfg_path = args.config or os.path.join(data_dir, "step5_config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"step5_config.json not found: {cfg_path}")
    cfg = read_json(cfg_path)

    logs = []
    pred_adj, pred_source, pred_prefix, pred_scores = find_pred_change_adj(data_dir, cfg, logs)
    true_adj, _ = find_true_change_adj(data_dir, logs)

    pred_edges = edges_from_adj(pred_adj, diag_excluded=True)
    true_edges = edges_from_adj(true_adj, diag_excluded=True)
    tp, fp, fn, prec, rec, f1 = confusion(pred_edges, true_edges)

    print("=== Step5 debug dump ===")
    print(f"config: {cfg_path}")
    print(f"pred_source: {pred_source}")
    print(f"pred_prefix: {pred_prefix}")
    print(f"pred_edges: {len(pred_edges)}")
    print(f"true_edges: {len(true_edges)}")
    print(f"TP={tp} FP={fp} FN={fn} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")
    print(f"binarize: {cfg.get('binarize', {})}")

    edges_list = list(pred_edges)
    if pred_scores:
        edges_list.sort(key=lambda e: pred_scores.get(e, 0.0), reverse=True)
    else:
        edges_list.sort()

    print(f"Top-{args.top_n} pred edges:")
    for i, (src, tgt) in enumerate(edges_list[:args.top_n], start=1):
        sc = pred_scores.get((src, tgt), 1.0) if pred_scores else 1.0
        flag = 1 if (src, tgt) in true_edges else 0
        print(f"{i:2d}. {src}->{tgt} score={sc:.4f} true={flag}")


if __name__ == "__main__":
    main()

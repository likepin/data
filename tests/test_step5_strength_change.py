import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from step5_utils import read_json, find_true_change_adj, find_pred_change_adj, edges_from_adj, confusion


def test_true_change_from_deltaA():
    data_dir = "synthetic_step3_v2"
    delta_path = os.path.join(data_dir, "DeltaA.npy")
    if not os.path.isfile(delta_path):
        return
    adj_true, _ = find_true_change_adj(data_dir, logs=[])
    true_edges = edges_from_adj(adj_true, diag_excluded=True)
    assert len(true_edges) == 6


def test_pred_on_base_topk():
    data_dir = "synthetic_step3_v2"
    cfg_path = os.path.join(data_dir, "step5_config.json")
    cfg = read_json(cfg_path)
    if "_on_base" not in cfg.get("pred_change_source", ""):
        return
    adj_true, _ = find_true_change_adj(data_dir, logs=[])
    true_edges = edges_from_adj(adj_true, diag_excluded=True)
    K_true = len(true_edges)
    pred_adj, _, _, _ = find_pred_change_adj(data_dir, cfg, logs=[])
    pred_edges = edges_from_adj(pred_adj, diag_excluded=True)
    assert len(pred_edges) == K_true
    tp, fp, fn, prec, rec, f1 = confusion(pred_edges, true_edges)
    assert tp > 0


def test_adjhat_xor_disallowed():
    data_dir = "synthetic_step3_v2"
    cfg = {"pred_change_source": "adjhat_xor", "pred_prefix": "parcorr"}
    try:
        find_pred_change_adj(data_dir, cfg, logs=[])
        assert False, "adjhat_xor should raise"
    except ValueError:
        pass

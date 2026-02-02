import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from step5_utils import read_json, find_pred_change_adj, edges_from_adj, load_n_from_source


def test_pred_change_loader_basic():
    data_dir = "synthetic_step3_v2"
    cfg_path = os.path.join(data_dir, "step5_config.json")
    cfg = read_json(cfg_path)
    pred_adj, _, _, _ = find_pred_change_adj(data_dir, cfg, logs=[])

    assert pred_adj.ndim == 2
    assert pred_adj.shape[0] == pred_adj.shape[1]
    N = pred_adj.shape[0]

    # diag == 0
    assert np.all(np.diag(pred_adj) == 0)

    # values are {0,1}
    vals = np.unique(pred_adj)
    assert set(vals.tolist()).issubset({0, 1})

    # orientation check: sample edges
    edges = list(edges_from_adj(pred_adj, diag_excluded=True))
    if edges:
        rng = np.random.RandomState(0)
        sample = rng.choice(len(edges), size=min(10, len(edges)), replace=False)
        for idx in sample:
            src, tgt = edges[idx]
            assert pred_adj[tgt, src] == 1


def test_pred_change_loader_n_source():
    data_dir = "synthetic_step3_v2"
    cfg_path = os.path.join(data_dir, "step5_config.json")
    cfg = read_json(cfg_path)
    N = load_n_from_source(data_dir, cfg.get("N_source", "X.npy"))
    pred_adj, _, _, _ = find_pred_change_adj(data_dir, cfg, logs=[])
    assert pred_adj.shape[0] == N


def test_orientation_roundtrip():
    # 3-node toy adjacency: edges 0->1 and 2->0
    adj = np.zeros((3, 3), dtype=np.int32)
    adj[1, 0] = 1  # 0->1
    adj[0, 2] = 1  # 2->0

    edges = edges_from_adj(adj, diag_excluded=True)

    # rebuild adjacency from edges
    adj2 = np.zeros_like(adj)
    for src, tgt in edges:
        adj2[tgt, src] = 1

    assert np.array_equal(adj, adj2)

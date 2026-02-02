import numpy as np


def load_adj(path):
    return np.load(path)


def binarize_adj(adj, mode="binary", alpha=None, tau=None):
    if mode == "binary":
        return (adj != 0).astype(np.int32)
    if mode == "val":
        if tau is None:
            raise ValueError("binarize_adj(mode='val') requires tau.")
        return (np.abs(adj) > float(tau)).astype(np.int32)
    if mode == "pval":
        if alpha is None:
            raise ValueError("binarize_adj(mode='pval') requires alpha.")
        return (adj < float(alpha)).astype(np.int32)
    raise ValueError(f"Unknown binarize mode: {mode}")


def assert_orientation(adj, convention="tgt_src"):
    if convention != "tgt_src":
        raise ValueError("Only 'tgt_src' convention is supported.")
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Adjacency must be a square 2D array.")

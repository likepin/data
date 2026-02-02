import numpy as np


def compute_change_scores(A0, A1, mode="valdiff"):
    if mode == "valdiff":
        return np.abs(A1 - A0)
    if mode == "signflip":
        return (np.sign(A1) != np.sign(A0)).astype(np.float32) * np.abs(A1 - A0)
    raise ValueError(f"Unknown mode: {mode}")


def binarize_topk_on_base(score_tgt_src, adj_base_tgt_src, top_k, diag_excluded=True):
    score = score_tgt_src.copy()
    base_mask = (adj_base_tgt_src != 0).astype(np.float32)
    score = score * base_mask
    if diag_excluded:
        np.fill_diagonal(score, 0.0)
    flat = np.argsort(score.reshape(-1))[::-1]
    N = score.shape[0]
    pred = np.zeros_like(score, dtype=np.int32)
    scores = {}
    count = 0
    for idx in flat:
        tgt = idx // N
        src = idx % N
        if score[tgt, src] <= 0:
            continue
        pred[tgt, src] = 1
        scores[(src, tgt)] = float(score[tgt, src])
        count += 1
        if count >= top_k:
            break
    return pred, scores

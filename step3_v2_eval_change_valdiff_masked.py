import os
import numpy as np

def best_signed_val_over_lags(val_matrix):
    vals = val_matrix[:, :, 1:]
    idx = np.argmax(np.abs(vals), axis=2)
    out = np.zeros(vals.shape[:2], dtype=np.float32)
    for src in range(out.shape[0]):
        for tgt in range(out.shape[1]):
            out[src, tgt] = vals[src, tgt, idx[src, tgt]]
    return out  # (src,tgt), signed

def metrics_excluding_diag(adj_true, adj_hat):
    N = adj_true.shape[0]
    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)
    y_true = adj_true[mask].astype(int)
    y_hat = adj_hat[mask].astype(int)
    tp = int(((y_true == 1) & (y_hat == 1)).sum())
    fp = int(((y_true == 0) & (y_hat == 1)).sum())
    fn = int(((y_true == 1) & (y_hat == 0)).sum())
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = 2 * prec * rec / (prec + rec + 1e-12)
    shd  = int(np.abs(y_true - y_hat).sum())
    return tp, fp, fn, prec, rec, f1, shd

def topk_masked(score, candidate_mask, K):
    # score: (src,tgt), candidate_mask: (src,tgt) bool
    N = score.shape[0]
    score2 = score.copy()
    # remove diagonal and non-candidates
    for k in range(N):
        candidate_mask[k, k] = False
    score2[~candidate_mask] = -np.inf

    flat = np.argsort(score2.reshape(-1))[::-1]
    pred = np.zeros((N, N), dtype=np.int32)  # (tgt,src)

    chosen = 0
    for idx in flat:
        src = idx // N
        tgt = idx % N
        if not candidate_mask[src, tgt]:
            continue
        pred[tgt, src] = 1
        chosen += 1
        if chosen >= K:
            break
    return pred

def run(prefix):
    data_dir = "synthetic_step3_v2"
    adj_change = np.load(os.path.join(data_dir, "adj_change.npy"))
    adj_base   = np.load(os.path.join(data_dir, "adj_base.npy"))
    K = int(adj_change.sum())

    v0 = np.load(os.path.join(data_dir, f"{prefix}_regime0_val_matrix.npy"))
    v1 = np.load(os.path.join(data_dir, f"{prefix}_regime1_val_matrix.npy"))

    s0 = best_signed_val_over_lags(v0)  # (src,tgt)
    s1 = best_signed_val_over_lags(v1)

    # candidate set option A: use TRUE base edges (best for toy)
    # adj_base is (tgt,src), convert to (src,tgt) mask:
    cand_true = (adj_base.T == 1)

    # candidate set option B: use BOTH predicted-present edges (more realistic)
    a0 = np.load(os.path.join(data_dir, f"{prefix}_regime0_adj_hat.npy"))
    a1 = np.load(os.path.join(data_dir, f"{prefix}_regime1_adj_hat.npy"))
    cand_pred = ((a0 == 1) & (a1 == 1)).T  # to (src,tgt)

    # scores
    score_mag = np.abs(np.abs(s1) - np.abs(s0))
    score_flip = ((np.sign(s0) != np.sign(s1)).astype(np.float32)) * (np.abs(s0) + np.abs(s1))

    print(f"\n=== {prefix.upper()} change detection (val-diff) ===")
    print(f"K(true_change_edges)={K}")

    # A) masked by TRUE base edges
    predA_mag  = topk_masked(score_mag,  cand_true.copy(), K)
    predA_flip = topk_masked(score_flip, cand_true.copy(), K)

    tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_change, predA_mag)
    print(f"[TRUE-mask |mag|diff] TP={tp} FP={fp} FN={fn} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | SHD={shd}")

    tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_change, predA_flip)
    print(f"[TRUE-mask signflip]  TP={tp} FP={fp} FN={fn} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | SHD={shd}")

    # B) masked by predicted common edges
    predB_mag  = topk_masked(score_mag,  cand_pred.copy(), K)
    predB_flip = topk_masked(score_flip, cand_pred.copy(), K)

    tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_change, predB_mag)
    print(f"[PRED-mask |mag|diff] TP={tp} FP={fp} FN={fn} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | SHD={shd}")

    tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_change, predB_flip)
    print(f"[PRED-mask signflip]  TP={tp} FP={fp} FN={fn} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | SHD={shd}")

def main():
    run("parcorr")
    run("cmiknn")

if __name__ == "__main__":
    main()

import os
import numpy as np

def best_p_over_lags(p_matrix):
    # p_matrix: (N,N,tau_max+1), p_matrix[source, target, lag]
    return np.min(p_matrix[:, :, 1:], axis=2)

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

def main():
    data_dir = "synthetic_step3"

    adj_delta = np.load(os.path.join(data_dir, "adj_delta.npy"))

    p0 = np.load(os.path.join(data_dir, "parcorr_regime0_p_matrix.npy"))
    p1 = np.load(os.path.join(data_dir, "parcorr_regime1_p_matrix.npy"))

    bp0 = best_p_over_lags(p0)  # shape (source, target)
    bp1 = best_p_over_lags(p1)

    eps = 1e-12
    score = np.log(bp0 + eps) - np.log(bp1 + eps)  # bigger => more regime1-specific

    N = score.shape[0]
    # remove diagonal in (source,target) space
    for k in range(N):
        score[k, k] = -np.inf

    K = int(adj_delta.sum())  # number of true dynamic edges
    flat = np.argsort(score.reshape(-1))[::-1]  # descending

    # pred adjacency uses [target, source] convention, consistent with your eval scripts
    pred = np.zeros((N, N), dtype=np.int32)

    chosen = 0
    for idx in flat:
        src = idx // N
        tgt = idx % N
        if src == tgt:
            continue
        pred[tgt, src] = 1
        chosen += 1
        if chosen >= K:
            break

    tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_delta, pred)
    print("=== Step3 Delta Recovery by p-diff (top-K, FIXED) | diag excluded ===")
    print(f"K(true_delta_edges)={K}")
    print(f"TP={tp} FP={fp} FN={fn} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | SHD={shd}")

    # Optional: print the chosen edges for sanity
    # show top-10 (src->tgt) and whether it's truly a delta edge
    print("\nTop candidates (src->tgt): score, p0, p1, is_true_delta")
    shown = 0
    for idx in flat[:50]:
        src = idx // N
        tgt = idx % N
        if src == tgt:
            continue
        is_true = int(adj_delta[tgt, src])  # [target,source]
        print(f"{src:2d}->{tgt:2d}: score={score[src,tgt]:8.3f}  p0={bp0[src,tgt]:.3g}  p1={bp1[src,tgt]:.3g}  delta={is_true}")
        shown += 1
        if shown >= 10:
            break

if __name__ == "__main__":
    main()

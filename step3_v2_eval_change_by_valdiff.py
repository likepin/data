import os
import numpy as np

def best_val_over_lags(val_matrix):
    # val_matrix[src, tgt, lag]; take lags 1..tau_max
    return np.max(np.abs(val_matrix[:, :, 1:]), axis=2)  # (src,tgt)

def best_signed_val_over_lags(val_matrix):
    # choose lag with max |val|, keep its sign
    vals = val_matrix[:, :, 1:]
    idx = np.argmax(np.abs(vals), axis=2)  # (src,tgt) lag index in 0..tau_max-1
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

def topk_from_score(score, K):
    N = score.shape[0]
    # remove diagonal
    for k in range(N):
        score[k, k] = -np.inf
    flat = np.argsort(score.reshape(-1))[::-1]
    pred = np.zeros((N, N), dtype=np.int32)  # (tgt,src)
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
    return pred

def main():
    data_dir = "synthetic_step3_v2"
    adj_change = np.load(os.path.join(data_dir, "adj_change.npy"))
    K = int(adj_change.sum())

    v0 = np.load(os.path.join(data_dir, "parcorr_regime0_val_matrix.npy"))
    v1 = np.load(os.path.join(data_dir, "parcorr_regime1_val_matrix.npy"))

    # 1) magnitude change score
    m0 = best_val_over_lags(v0)
    m1 = best_val_over_lags(v1)
    score_mag = np.abs(m1 - m0)

    pred_mag = topk_from_score(score_mag.copy(), K)
    tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_change, pred_mag)
    print("=== Step3 V2 Change Recovery by |val|-diff (top-K) ===")
    print(f"K(true_change_edges)={K}")
    print(f"TP={tp} FP={fp} FN={fn} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | SHD={shd}")

    # 2) sign-flip score (prefer edges with sign flip and large magnitude)
    s0 = best_signed_val_over_lags(v0)
    s1 = best_signed_val_over_lags(v1)
    sign_flip = (np.sign(s0) != np.sign(s1)).astype(np.float32)
    score_flip = sign_flip * (np.abs(s0) + np.abs(s1))  # prioritize strong + flipped

    pred_flip = topk_from_score(score_flip.copy(), K)
    tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_change, pred_flip)
    print("\n=== Step3 V2 Change Recovery by sign-flip (top-K) ===")
    print(f"TP={tp} FP={fp} FN={fn} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | SHD={shd}")

if __name__ == "__main__":
    main()

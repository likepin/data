import os, json
import numpy as np

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

def best_p_over_lags(p_matrix):
    return np.min(p_matrix[:, :, 1:], axis=2)  # (src,tgt)

def delta_by_pdiff_topk(p0, p1, K):
    bp0 = best_p_over_lags(p0)
    bp1 = best_p_over_lags(p1)
    eps = 1e-12
    score = np.log(bp0 + eps) - np.log(bp1 + eps)
    N = score.shape[0]
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

def load(path):
    return np.load(path) if os.path.exists(path) else None

def eval_model(prefix, data_dir):
    # prefix examples: "parcorr", "cmiknn"
    adj_base = np.load(os.path.join(data_dir, "adj_base.npy"))
    adj_reg1 = np.load(os.path.join(data_dir, "adj_regime1.npy"))
    adj_chg  = np.load(os.path.join(data_dir, "adj_change.npy"))
    K_chg = int(adj_chg.sum())

    a0 = load(os.path.join(data_dir, f"{prefix}_regime0_adj_hat.npy"))
    a1 = load(os.path.join(data_dir, f"{prefix}_regime1_adj_hat.npy"))
    p0 = load(os.path.join(data_dir, f"{prefix}_regime0_p_matrix.npy"))
    p1 = load(os.path.join(data_dir, f"{prefix}_regime1_p_matrix.npy"))

    if a0 is None or a1 is None:
        print(f"[SKIP] {prefix}: missing regime0/regime1 outputs.")
        return

    print(f"\n=== {prefix.upper()} Eval (diag excluded) ===")

    tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_base, a0)
    print(f"[BASE]  regime0 vs adj_base    : TP={tp} FP={fp} FN={fn} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | SHD={shd}")

    tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_reg1, a1)
    print(f"[REG1]  regime1 vs adj_regime1 : TP={tp} FP={fp} FN={fn} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | SHD={shd}")

    # Hard change detection: edges that differ between regime graphs
    # Here we treat "change" as: predicted regime1 edge status != predicted regime0 edge status
    # Because V2 uses strength change on existing edges, they should remain present but significance changes;
    # so hard XOR may be weak. We still print it.
    chg_hat_xor = ((a1 == 1) ^ (a0 == 1)).astype(np.int32)
    tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_chg, chg_hat_xor)
    print(f"[CHG-XOR] (reg1 XOR reg0) vs adj_change: TP={tp} FP={fp} FN={fn} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | SHD={shd} "
          f"| edges_hat={int(chg_hat_xor.sum())} true={K_chg}")

    # p-diff top-K: best for detecting strength/significance change
    if p0 is not None and p1 is not None:
        chg_hat_pdiff = delta_by_pdiff_topk(p0, p1, K=K_chg)
        tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_chg, chg_hat_pdiff)
        print(f"[CHG-pdiff] topK={K_chg} vs adj_change: TP={tp} FP={fp} FN={fn} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | SHD={shd}")
    else:
        print("[CHG-pdiff] missing p_matrix files; skip.")

def main():
    data_dir = "synthetic_step3_v2"
    meta = json.load(open(os.path.join(data_dir, "meta.json"), "r", encoding="utf-8"))
    print("=== Step3 V2 Summary ===")
    print(meta)

    # ParCorr outputs:
    # - we saved parcorr_regime0_*, parcorr_regime1_*
    eval_model("parcorr", data_dir)

    # CMIknn outputs:
    # - we saved cmiknn_regime0_*, cmiknn_regime1_*
    eval_model("cmiknn", data_dir)

if __name__ == "__main__":
    main()


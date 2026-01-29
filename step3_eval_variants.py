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

def load_if_exists(path):
    return np.load(path) if os.path.exists(path) else None

def main():
    data_dir = "synthetic_step3"
    meta = json.load(open(os.path.join(data_dir, "meta.json"), "r", encoding="utf-8"))

    adj_base    = np.load(os.path.join(data_dir, "adj_base.npy"))
    adj_regime1 = np.load(os.path.join(data_dir, "adj_regime1.npy"))
    adj_delta   = np.load(os.path.join(data_dir, "adj_delta.npy"))

    # predicted graphs
    tags = ["trainonly", "all", "regime0", "regime1"]
    preds = {}
    for tag in tags:
        p = os.path.join(data_dir, f"parcorr_{tag}_adj_hat.npy")
        preds[tag] = load_if_exists(p)

    print("=== Step3 Eval (ParCorr) | diag excluded ===")
    print(f"Meta: t_switch={meta['t_switch']}, gamma={meta['gamma']}, sigma={meta['sigma']}, K_delta_edges={meta['K_delta_edges']}")
    print()

    # base recovery: trainonly / regime0 should match adj_base
    for tag in ["trainonly", "regime0"]:
        if preds[tag] is None: 
            continue
        tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_base, preds[tag])
        print(f"[BASE] {tag:9s}: TP={tp} FP={fp} FN={fn} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | SHD={shd}")

    # regime1 recovery: regime1 / all compare to adj_regime1
    for tag in ["regime1", "all"]:
        if preds[tag] is None:
            continue
        tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_regime1, preds[tag])
        print(f"[REG1] {tag:9s}: TP={tp} FP={fp} FN={fn} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | SHD={shd}")

    print()

    # dynamic-only recovery: (regime1 - regime0) vs adj_delta
    if preds["regime0"] is not None and preds["regime1"] is not None:
        delta_hat = ((preds["regime1"] == 1) & (preds["regime0"] == 0)).astype(np.int32)
        tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_delta, delta_hat)
        print(f"[DELTA] regime1-minus-regime0: TP={tp} FP={fp} FN={fn} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | SHD={shd}")
        print(f"        edges_true_delta={int(adj_delta.sum())}, edges_hat_delta={int(delta_hat.sum())}")

if __name__ == "__main__":
    main()

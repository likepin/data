import os
import numpy as np

def collapse_adj_from_pmatrix(p_matrix: np.ndarray, tau_max: int, alpha: float):
    N = p_matrix.shape[0]
    adj_hat = np.zeros((N, N), dtype=np.int32)
    for j in range(N):
        for i in range(N):
            if np.any(p_matrix[i, j, 1:(tau_max+1)] <= alpha):
                adj_hat[j, i] = 1
    return adj_hat

def metrics_excluding_diag(adj_true: np.ndarray, adj_hat: np.ndarray):
    N = adj_true.shape[0]
    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)

    y_true = adj_true[mask].astype(int)
    y_hat = adj_hat[mask].astype(int)

    tp = int(((y_true == 1) & (y_hat == 1)).sum())
    fp = int(((y_true == 0) & (y_hat == 1)).sum())
    fn = int(((y_true == 1) & (y_hat == 0)).sum())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    shd = int(np.abs(y_true - y_hat).sum())

    return tp, fp, fn, precision, recall, f1, shd

def eval_from_p(p_path: str, adj_true: np.ndarray, tau_max: int, alpha: float):
    p = np.load(p_path)
    adj_hat = collapse_adj_from_pmatrix(p, tau_max=tau_max, alpha=alpha)
    edges = int(adj_hat.sum())
    tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_true, adj_hat)
    return edges, tp, fp, fn, prec, rec, f1, shd

def main():
    data_dir = "synthetic_step2"
    tau_max = 2
    alphas = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    knns = [10, 20, 30]

    adj_true = np.load(os.path.join(data_dir, "adj_true.npy"))

    print("alpha | knn | edges | TP FP FN | Prec  Rec   F1   | SHD")
    print("-"*70)
    best = None  # (f1, -prec, shd, knn, alpha)

    for a in alphas:
        for k in knns:
            p_path = os.path.join(data_dir, f"cmiknn_knn{k}_p_matrix.npy")
            if not os.path.exists(p_path):
                continue
            edges, tp, fp, fn, prec, rec, f1, shd = eval_from_p(p_path, adj_true, tau_max, a)
            print(f"{a:5.3f} | {k:3d} | {edges:5d} | {tp:2d} {fp:2d} {fn:2d} | {prec:0.3f} {rec:0.3f} {f1:0.3f} | {shd:2d}")

            key = (f1, prec, -shd, -k, -a)  # tie-breakers arbitrary
            if best is None or key > best[0]:
                best = (key, (k, a, edges, tp, fp, fn, prec, rec, f1, shd))

        print("-"*70)

    if best is not None:
        k, a, edges, tp, fp, fn, prec, rec, f1, shd = best[1]
        print(f"\n[BEST] knn={k}, alpha={a:.3f}: edges={edges}, TP={tp}, FP={fp}, FN={fn}, "
              f"Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}, SHD={shd}")

if __name__ == "__main__":
    main()

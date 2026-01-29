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

def main():
    data_dir = "synthetic_step2"
    tau_max = 2

    adj_true = np.load(os.path.join(data_dir, "adj_true.npy"))

    # IMPORTANT: load the p_matrix you actually produced (knn=20)
    p_path = os.path.join(data_dir, "cmiknn_knn20_p_matrix.npy")
    p_matrix = np.load(p_path)

    alphas = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    print(f"[INFO] Loaded {p_path}, shape={p_matrix.shape}")
    print("alpha | edges | TP FP FN | Prec  Rec   F1   | SHD")
    print("-"*72)

    for a in alphas:
        adj_hat = collapse_adj_from_pmatrix(p_matrix, tau_max=tau_max, alpha=a)
        tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_true, adj_hat)
        edges = int(adj_hat.sum())
        print(f"{a:5.3f} | {edges:5d} | {tp:2d} {fp:2d} {fn:2d} | {prec:0.3f} {rec:0.3f} {f1:0.3f} | {shd:2d}")

if __name__ == "__main__":
    main()

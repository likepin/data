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

def main():
    data_dir = "synthetic_step2"
    tau_max = 2
    alpha = 0.02  # BEST from your sweep

    p_matrix = np.load(os.path.join(data_dir, "cmiknn_knn20_p_matrix.npy"))
    adj_hat = collapse_adj_from_pmatrix(p_matrix, tau_max=tau_max, alpha=alpha)

    out_path = os.path.join(data_dir, f"cmiknn_knn20_alpha{alpha:.3f}_adj_hat.npy")
    np.save(out_path, adj_hat)
    print("[OK] Saved:", out_path, "edges=", int(adj_hat.sum()))

if __name__ == "__main__":
    main()

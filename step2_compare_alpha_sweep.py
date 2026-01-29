import os
import numpy as np

def collapse_adj_from_pmatrix(p_matrix: np.ndarray, tau_max: int, alpha: float):
    """
    p_matrix: (N, N, tau_max+1) with lag dimension 0..tau_max
    Return adj_hat: (N, N) where adj_hat[j,i]=1 means i->j exists at any lag>=1
    """
    N = p_matrix.shape[0]
    adj_hat = np.zeros((N, N), dtype=np.int32)
    for j in range(N):
        for i in range(N):
            if np.any(p_matrix[i, j, 1:(tau_max + 1)] <= alpha):
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

def eval_one(p_matrix: np.ndarray, adj_true: np.ndarray, tau_max: int, alpha: float):
    adj_hat = collapse_adj_from_pmatrix(p_matrix, tau_max=tau_max, alpha=alpha)
    edges = int(adj_hat.sum())
    tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_true, adj_hat)
    return edges, tp, fp, fn, prec, rec, f1, shd

def main():
    data_dir = "synthetic_step2"
    tau_max = 2

    # ---- inputs (update filenames if yours differ) ----
    adj_true_path = os.path.join(data_dir, "adj_true.npy")
    parcorr_p_path = os.path.join(data_dir, "parcorr_p_matrix.npy")
    cmiknn_p_path  = os.path.join(data_dir, "cmiknn_knn20_p_matrix.npy")
    # ---------------------------------------------------

    adj_true = np.load(adj_true_path)
    p_par = np.load(parcorr_p_path)
    p_cmi = np.load(cmiknn_p_path)

    assert p_par.shape == p_cmi.shape, f"Shape mismatch: {p_par.shape} vs {p_cmi.shape}"

    # Choose alpha grid (you can edit)
    alphas = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]

    print(f"[INFO] data_dir={data_dir}, tau_max={tau_max}")
    print(f"[INFO] adj_true={adj_true.shape}, parcorr_p={p_par.shape}, cmiknn_p={p_cmi.shape}")
    print()
    header = (
        "alpha | "
        "ParCorr: edges TP FP FN  Prec  Rec   F1   SHD | "
        "CMIknn : edges TP FP FN  Prec  Rec   F1   SHD"
    )
    print(header)
    print("-" * len(header))

    for a in alphas:
        e1,tp1,fp1,fn1,pr1,rc1,f11,shd1 = eval_one(p_par, adj_true, tau_max, a)
        e2,tp2,fp2,fn2,pr2,rc2,f12,shd2 = eval_one(p_cmi, adj_true, tau_max, a)

        print(
            f"{a:5.3f} | "
            f"{e1:5d} {tp1:2d} {fp1:2d} {fn1:2d} {pr1:0.3f} {rc1:0.3f} {f11:0.3f} {shd1:3d} | "
            f"{e2:5d} {tp2:2d} {fp2:2d} {fn2:2d} {pr2:0.3f} {rc2:0.3f} {f12:0.3f} {shd2:3d}"
        )

    print("\n[NOTE] This is a fair comparison: same alpha grid, same tau_max, same evaluation (diag excluded).")
    print("[TIP] For an even fairer structural comparison, you can also match edge density by choosing alpha "
          "such that ParCorr and CMIknn have similar 'edges'.")

if __name__ == "__main__":
    main()

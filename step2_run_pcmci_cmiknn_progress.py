import os
import time
import traceback
import numpy as np

from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.cmiknn import CMIknn

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

def collapse_adj_from_pmatrix(p_matrix: np.ndarray, tau_max: int, alpha_level: float):
    """
    p_matrix: shape (N, N, tau_max+1) where last dim is lag 0..tau_max
    Return adj_hat: (N, N) where adj_hat[j,i]=1 means i->j exists at any lag>=1
    """
    N = p_matrix.shape[0]
    adj_hat = np.zeros((N, N), dtype=np.int32)
    for j in range(N):
        for i in range(N):
            if np.any(p_matrix[i, j, 1:(tau_max + 1)] <= alpha_level):
                adj_hat[j, i] = 1
    return adj_hat

def run_pcmci_cmiknn_once(
    dataframe: DataFrame,
    tau_max: int,
    alpha_level: float,
    knn: int,
    sig_samples: int,
    shuffle_neighbors: int,
    out_dir: str,
    adj_true: np.ndarray = None
):
    print("=" * 80)
    print(f"[START] CMIknn PCMCI | knn={knn} | sig_samples={sig_samples} | "
          f"shuffle_neighbors={shuffle_neighbors} | tau_max={tau_max} | alpha={alpha_level}")
    t0 = time.time()

    # CMIknn with shuffle_test (most robust but slow)
    cmiknn = CMIknn(
        significance='shuffle_test',
        knn=knn,
        shuffle_neighbors=shuffle_neighbors,
        sig_samples=sig_samples,
        verbosity=0
    )

    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cmiknn, verbosity=1)

    # Run PCMCI
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha_level)

    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]
    adj_hat = collapse_adj_from_pmatrix(p_matrix, tau_max=tau_max, alpha_level=alpha_level)

    # Save
    np.save(os.path.join(out_dir, f"cmiknn_knn{knn}_p_matrix.npy"), p_matrix)
    np.save(os.path.join(out_dir, f"cmiknn_knn{knn}_val_matrix.npy"), val_matrix)
    np.save(os.path.join(out_dir, f"cmiknn_knn{knn}_adj_hat.npy"), adj_hat)

    elapsed = time.time() - t0
    print(f"[DONE] knn={knn} saved. edges={int(adj_hat.sum())}. elapsed={elapsed/60:.2f} min")

    # Quick eval if adj_true provided
    if adj_true is not None:
        tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_true, adj_hat)
        print(f"[EVAL] TP={tp} FP={fp} FN={fn} | Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f} | SHD={shd}")

    return elapsed

def main():
    data_dir = "synthetic_step2"
    out_dir = data_dir
    os.makedirs(out_dir, exist_ok=True)

    # ---------- knobs ----------
    tau_max = 2
    alpha_level = 0.02

    # Quick mode (recommended first): 20
    # Formal mode later: 100 or 200
    sig_samples = 200

    # Smaller shuffle_neighbors is faster; keep 10 for now
    shuffle_neighbors = 10

    # Scan a few knn; for quick test you can set [20] only
    knn_list = [ 20]
    # ---------------------------

    print("[INFO] Loading data...")
    X = np.load(os.path.join(data_dir, "X.npy"))
    adj_true_path = os.path.join(data_dir, "adj_true.npy")
    adj_true = np.load(adj_true_path) if os.path.exists(adj_true_path) else None
    print(f"[INFO] X shape={X.shape}. adj_true={'found' if adj_true is not None else 'not found'}")

    dataframe = DataFrame(X)

    total_t0 = time.time()
    for knn in knn_list:
        try:
            run_pcmci_cmiknn_once(
                dataframe=dataframe,
                tau_max=tau_max,
                alpha_level=alpha_level,
                knn=knn,
                sig_samples=sig_samples,
                shuffle_neighbors=shuffle_neighbors,
                out_dir=out_dir,
                adj_true=adj_true
            )
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] User cancelled (Ctrl+C). Exiting cleanly.")
            return
        except Exception as e:
            print(f"\n[ERROR] knn={knn} failed: {repr(e)}")
            traceback.print_exc()
            print("[HINT] If it hangs or is too slow on Windows, try:")
            print("  - sig_samples=10 or 20 (quick)")
            print("  - knn_list=[20] only")
            print("  - reduce T in generator (e.g., 6000->2000)")
            # continue to next knn instead of stopping
            continue

    total_elapsed = time.time() - total_t0
    print("=" * 80)
    print(f"[ALL DONE] Total elapsed: {total_elapsed/60:.2f} min")

if __name__ == "__main__":
    main()

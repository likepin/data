import os
import numpy as np

from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.cmiknn import CMIknn

def run_one(dataframe, tau_max: int, alpha_level: float, knn: int, out_dir: str):
    cmiknn = CMIknn(
        significance='shuffle_test',
        knn=knn,
        shuffle_neighbors=10,
        # You can increase for more stable p-values, but slower:
        # sig_samples=200,
        sig_samples=100,
        verbosity=0
    )

    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cmiknn, verbosity=1)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha_level)

    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]

    N = p_matrix.shape[0]
    adj_hat = np.zeros((N, N), dtype=np.int32)
    for j in range(N):
        for i in range(N):
            if np.any(p_matrix[i, j, 1:(tau_max+1)] <= alpha_level):
                adj_hat[j, i] = 1

    np.save(os.path.join(out_dir, f"cmiknn_knn{knn}_p_matrix.npy"), p_matrix)
    np.save(os.path.join(out_dir, f"cmiknn_knn{knn}_val_matrix.npy"), val_matrix)
    np.save(os.path.join(out_dir, f"cmiknn_knn{knn}_adj_hat.npy"), adj_hat)

    print(f"[OK] CMIknn knn={knn}: edges={int(adj_hat.sum())}")

def main():
    data_dir = "synthetic_step2"
    X = np.load(os.path.join(data_dir, "X.npy"))  # (T, N)

    dataframe = DataFrame(X)

    tau_max = 2
    alpha_level = 0.01  # keep same default; tune later if needed

    out_dir = data_dir
    for knn in [10, 20, 30]:
        run_one(dataframe, tau_max=tau_max, alpha_level=alpha_level, knn=knn, out_dir=out_dir)

if __name__ == "__main__":
    main()


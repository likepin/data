import os
import numpy as np

from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

def main():
    data_dir = "synthetic_step2"
    X = np.load(os.path.join(data_dir, "X.npy"))  # (T, N)
    T, N = X.shape

    dataframe = DataFrame(X)
    parcorr = ParCorr(significance='analytic')

    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=1)

    tau_max = 2
    alpha_level = 0.01  # keep same default; you can also try 0.05 / 0.005

    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha_level)

    p_matrix = results["p_matrix"]      # (N, N, tau_max+1)
    val_matrix = results["val_matrix"]  # (N, N, tau_max+1)

    # collapsed adjacency over lags 1..tau_max
    adj_hat = np.zeros((N, N), dtype=np.int32)
    for j in range(N):      # target
        for i in range(N):  # source
            if np.any(p_matrix[i, j, 1:(tau_max+1)] <= alpha_level):
                adj_hat[j, i] = 1

    np.save(os.path.join(data_dir, "parcorr_p_matrix.npy"), p_matrix)
    np.save(os.path.join(data_dir, "parcorr_val_matrix.npy"), val_matrix)
    np.save(os.path.join(data_dir, "parcorr_adj_hat.npy"), adj_hat)

    print("[OK] Saved ParCorr PCMCI outputs to ./synthetic_step2/")
    print("Adj_hat edges (collapsed):", int(adj_hat.sum()))

if __name__ == "__main__":
    main()

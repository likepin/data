import os
import numpy as np

from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

def main():
    data_dir = "synthetic_step1"
    X = np.load(os.path.join(data_dir, "X.npy"))  # (T, N)
    T, N = X.shape

    # tigramite wants data in DataFrame
    dataframe = DataFrame(X)

    # Linear CI test for Step 1
    parcorr = ParCorr(significance='analytic')

    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=parcorr,
        verbosity=1
    )

    tau_max = 2  # should match L
    # You can tune alpha_level; start with 0.01 or 0.05. We'll use 0.01 for cleaner graphs.
    alpha_level = 0.01

    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha_level)

    p_matrix = results["p_matrix"]     # shape (N, N, tau_max+1) with lag 0..tau_max
    val_matrix = results["val_matrix"] # same shape, partial correlation values

    # Convert to adjacency: we care about lagged effects only (tau >= 1). Exclude lag 0.
    # Edge i -> j exists if any lag tau in [1..tau_max] has p <= alpha_level
    adj_hat = np.zeros((N, N), dtype=np.int32)
    for j in range(N):          # target
        for i in range(N):      # source
            if i == j:
                # keep self edges if you want; for now allow self-lag edges too
                pass
            # check any lagged significance
            if np.any(p_matrix[i, j, 1:(tau_max+1)] <= alpha_level):
                adj_hat[j, i] = 1  # NOTE: we store i->j as adj_hat[j,i]=1 for consistency with A_true

    np.save(os.path.join(data_dir, "pcmci_p_matrix.npy"), p_matrix)
    np.save(os.path.join(data_dir, "pcmci_val_matrix.npy"), val_matrix)
    np.save(os.path.join(data_dir, "adj_hat.npy"), adj_hat)

    print("[OK] Saved PCMCI outputs to ./synthetic_step1/")
    print("Adj_hat edges (collapsed):", int(adj_hat.sum()))

if __name__ == "__main__":
    main()

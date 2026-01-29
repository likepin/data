import os, json
import numpy as np
from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

def collapse_adj(p_matrix, tau_max, alpha_level):
    N = p_matrix.shape[0]
    adj_hat = np.zeros((N, N), dtype=np.int32)
    for j in range(N):
        for i in range(N):
            if np.any(p_matrix[i, j, 1:(tau_max+1)] <= alpha_level):
                adj_hat[j, i] = 1
    return adj_hat

def run_one(X, out_dir, tag, tau_max=2, alpha_level=0.01):
    dataframe = DataFrame(X)
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=1)

    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha_level)
    p = results["p_matrix"]
    v = results["val_matrix"]
    adj_hat = collapse_adj(p, tau_max=tau_max, alpha_level=alpha_level)

    np.save(os.path.join(out_dir, f"parcorr_{tag}_p_matrix.npy"), p)
    np.save(os.path.join(out_dir, f"parcorr_{tag}_val_matrix.npy"), v)
    np.save(os.path.join(out_dir, f"parcorr_{tag}_adj_hat.npy"), adj_hat)

    print(f"[OK] ParCorr {tag}: edges={int(adj_hat.sum())}")

def main():
    data_dir = "synthetic_step3"
    X = np.load(os.path.join(data_dir, "X.npy"))
    s = np.load(os.path.join(data_dir, "s.npy"))
    meta = json.load(open(os.path.join(data_dir, "meta.json"), "r", encoding="utf-8"))
    t_switch = int(meta["t_switch"])

    # 你可以把 alpha 调成 0.005/0.01/0.02 做敏感性
    tau_max = 2
    alpha_level = 0.01

    # 1) TrainOnly = 只用 regime0 段
    run_one(X[:t_switch], data_dir, "trainonly", tau_max=tau_max, alpha_level=alpha_level)

    # 2) All = 全量混合
    run_one(X, data_dir, "all", tau_max=tau_max, alpha_level=alpha_level)

    # 3) RegimeSplit：分别跑 0/1
    run_one(X[s == 0], data_dir, "regime0", tau_max=tau_max, alpha_level=alpha_level)
    run_one(X[s == 1], data_dir, "regime1", tau_max=tau_max, alpha_level=alpha_level)

if __name__ == "__main__":
    main()

import os, json, time, traceback
import numpy as np
from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.cmiknn import CMIknn

def collapse_adj_from_p(p_matrix: np.ndarray, tau_max: int, alpha: float) -> np.ndarray:
    N = p_matrix.shape[0]
    adj_hat = np.zeros((N, N), dtype=np.int32)
    for tgt in range(N):
        for src in range(N):
            if np.any(p_matrix[src, tgt, 1:(tau_max + 1)] <= alpha):
                adj_hat[tgt, src] = 1
    return adj_hat

def run_pcmci_cmiknn(X, out_dir, tag, tau_max, alpha_level, knn, sig_samples, shuffle_neighbors, verbosity=1):
    dataframe = DataFrame(X)
    cond_test = CMIknn(
        significance="shuffle_test",
        knn=knn,
        sig_samples=sig_samples,
        shuffle_neighbors=shuffle_neighbors,
    )
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_test, verbosity=verbosity)

    t0 = time.time()
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha_level)
    elapsed = (time.time() - t0) / 60.0

    p = results["p_matrix"]
    v = results["val_matrix"]
    adj_hat = collapse_adj_from_p(p, tau_max=tau_max, alpha=alpha_level)

    np.save(os.path.join(out_dir, f"cmiknn_{tag}_p_matrix.npy"), p)
    np.save(os.path.join(out_dir, f"cmiknn_{tag}_val_matrix.npy"), v)
    np.save(os.path.join(out_dir, f"cmiknn_{tag}_adj_hat.npy"), adj_hat)

    print(f"[DONE] {tag}: knn={knn} sig_samples={sig_samples} alpha={alpha_level} "
          f"saved. edges={int(adj_hat.sum())}. elapsed={elapsed:.2f} min")
    return p, v, adj_hat

def main():
    data_dir = "synthetic_step3_v2"
    X = np.load(os.path.join(data_dir, "X.npy"))
    s = np.load(os.path.join(data_dir, "s.npy"))
    meta = json.load(open(os.path.join(data_dir, "meta.json"), "r", encoding="utf-8"))

    tau_max = 2
    knn = 20

    fast = dict(alpha_level=0.02, sig_samples=60, shuffle_neighbors=10)
    formal = dict(alpha_level=0.02, sig_samples=200, shuffle_neighbors=10)

    RUN_STAGE = "fast"   # 改成 "formal" 出正式数
    cfg = fast if RUN_STAGE == "fast" else formal

    print("=== Step3 V2 PCMCI + CMIknn (RegimeSplit) ===")
    print(f"Meta: t_switch={meta['t_switch']}, gamma={meta['gamma']}, sigma={meta['sigma']}, "
          f"K_change_edges={meta['K_change_edges']}, gain={meta['gain']}, flip_prob={meta['flip_prob']}")
    print(f"Stage: {RUN_STAGE} | tau_max={tau_max}, knn={knn}, alpha={cfg['alpha_level']}, sig_samples={cfg['sig_samples']}")
    print()

    X0 = X[s == 0]
    X1 = X[s == 1]
    print(f"Regime0 len={len(X0)}, Regime1 len={len(X1)}")

    try:
        run_pcmci_cmiknn(X0, data_dir, "regime0", tau_max, cfg["alpha_level"], knn, cfg["sig_samples"], cfg["shuffle_neighbors"], verbosity=1)
        run_pcmci_cmiknn(X1, data_dir, "regime1", tau_max, cfg["alpha_level"], knn, cfg["sig_samples"], cfg["shuffle_neighbors"], verbosity=1)
    except KeyboardInterrupt:
        print("\n[STOP] KeyboardInterrupt. Partial outputs may exist.")
    except Exception:
        print("\n[ERROR] Exception during PCMCI+CMIknn:")
        traceback.print_exc()

if __name__ == "__main__":
    main()

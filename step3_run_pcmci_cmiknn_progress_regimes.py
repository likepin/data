import os, json, time, traceback
import numpy as np

from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.cmiknn import CMIknn


# -------------------------
# Utils: adjacency + metrics
# -------------------------
def collapse_adj_from_p(p_matrix: np.ndarray, tau_max: int, alpha: float) -> np.ndarray:
    """p_matrix[source, target, lag]; return adj_hat[target, source] collapsed over lags>=1."""
    N = p_matrix.shape[0]
    adj_hat = np.zeros((N, N), dtype=np.int32)
    for tgt in range(N):
        for src in range(N):
            if np.any(p_matrix[src, tgt, 1:(tau_max + 1)] <= alpha):
                adj_hat[tgt, src] = 1
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

    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = 2 * prec * rec / (prec + rec + 1e-12)
    shd  = int(np.abs(y_true - y_hat).sum())
    return tp, fp, fn, prec, rec, f1, shd

def best_p_over_lags(p_matrix: np.ndarray) -> np.ndarray:
    """Return best p-value over lags>=1 for each (src,tgt). Shape (src,tgt)."""
    return np.min(p_matrix[:, :, 1:], axis=2)

def delta_by_pdiff_topk(p0: np.ndarray, p1: np.ndarray, K: int) -> np.ndarray:
    """
    p0/p1 are p_matrix arrays. Compute score(src,tgt)=log(p0)-log(p1), take top-K as delta edges.
    Return pred_delta adjacency in [tgt,src] convention.
    """
    bp0 = best_p_over_lags(p0)
    bp1 = best_p_over_lags(p1)

    eps = 1e-12
    score = np.log(bp0 + eps) - np.log(bp1 + eps)  # bigger => more regime1-specific
    N = score.shape[0]
    for k in range(N):
        score[k, k] = -np.inf

    flat = np.argsort(score.reshape(-1))[::-1]
    pred = np.zeros((N, N), dtype=np.int32)

    chosen = 0
    for idx in flat:
        src = idx // N
        tgt = idx % N
        if src == tgt:
            continue
        pred[tgt, src] = 1
        chosen += 1
        if chosen >= K:
            break
    return pred


# -------------------------
# Core runner
# -------------------------
def run_pcmci_cmiknn(
    X: np.ndarray,
    out_dir: str,
    tag: str,
    tau_max: int,
    alpha_level: float,
    knn: int,
    sig_samples: int,
    shuffle_neighbors: int,
    verbosity: int = 1,
):
    """
    Runs PCMCI with CMIknn shuffle_test significance.
    Saves p_matrix/val_matrix/adj_hat in out_dir with prefix cmiknn_{tag}_...
    Returns (p_matrix, val_matrix, adj_hat).
    """
    dataframe = DataFrame(X)

    # Note: CMIknn uses shuffle_test for p-values; parameters affect runtime a lot.
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
    data_dir = "synthetic_step3"
    X = np.load(os.path.join(data_dir, "X.npy"))
    s = np.load(os.path.join(data_dir, "s.npy"))
    meta = json.load(open(os.path.join(data_dir, "meta.json"), "r", encoding="utf-8"))

    adj_base    = np.load(os.path.join(data_dir, "adj_base.npy"))
    adj_regime1 = np.load(os.path.join(data_dir, "adj_regime1.npy"))
    adj_delta   = np.load(os.path.join(data_dir, "adj_delta.npy"))
    K_true_delta = int(adj_delta.sum())

    # -------------------------
    # Config (two-stage)
    # -------------------------
    tau_max = 2
    knn = 20

    # Stage A: FAST (for iteration)
    fast = dict(alpha_level=0.02, sig_samples=60, shuffle_neighbors=10)

    # Stage B: FORMAL (paper-ish)
    formal = dict(alpha_level=0.02, sig_samples=200, shuffle_neighbors=10)

    # Choose which stage to run:
    # set RUN_STAGE = "fast" or "formal"
    RUN_STAGE = "fast"

    cfg = fast if RUN_STAGE == "fast" else formal

    print("=== Step3 PCMCI + CMIknn (RegimeSplit) ===")
    print(f"Meta: t_switch={meta['t_switch']}, gamma={meta['gamma']}, sigma={meta['sigma']}, K_delta_edges={meta['K_delta_edges']}")
    print(f"Stage: {RUN_STAGE} | tau_max={tau_max}, knn={knn}, alpha={cfg['alpha_level']}, sig_samples={cfg['sig_samples']}")
    print()

    X0 = X[s == 0]
    X1 = X[s == 1]
    print(f"Regime0 len={len(X0)}, Regime1 len={len(X1)}")

    # -------------------------
    # Run
    # -------------------------
    try:
        p0, v0, adj0 = run_pcmci_cmiknn(
            X0, data_dir, "regime0",
            tau_max=tau_max, alpha_level=cfg["alpha_level"],
            knn=knn, sig_samples=cfg["sig_samples"],
            shuffle_neighbors=cfg["shuffle_neighbors"],
            verbosity=1
        )

        p1, v1, adj1 = run_pcmci_cmiknn(
            X1, data_dir, "regime1",
            tau_max=tau_max, alpha_level=cfg["alpha_level"],
            knn=knn, sig_samples=cfg["sig_samples"],
            shuffle_neighbors=cfg["shuffle_neighbors"],
            verbosity=1
        )

    except KeyboardInterrupt:
        print("\n[STOP] KeyboardInterrupt. Partial outputs may exist.")
        return
    except Exception:
        print("\n[ERROR] Exception during PCMCI+CMIknn:")
        traceback.print_exc()
        return

    # -------------------------
    # Eval: base / regime1
    # -------------------------
    print("\n=== Eval (diag excluded) ===")

    tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_base, adj0)
    print(f"[BASE]  cmiknn_regime0: TP={tp} FP={fp} FN={fn} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | SHD={shd}")

    tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_regime1, adj1)
    print(f"[REG1]  cmiknn_regime1: TP={tp} FP={fp} FN={fn} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | SHD={shd}")

    # -------------------------
    # Delta (hard difference)
    # -------------------------
    delta_hat_hard = ((adj1 == 1) & (adj0 == 0)).astype(np.int32)
    tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_delta, delta_hat_hard)
    print(f"[DELTA-hard] regime1-minus-regime0: TP={tp} FP={fp} FN={fn} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | SHD={shd} "
          f"| edges_hat_delta={int(delta_hat_hard.sum())} true={K_true_delta}")

    # -------------------------
    # Delta (p-diff top-K)
    # -------------------------
    delta_hat_pdiff = delta_by_pdiff_topk(p0, p1, K=K_true_delta)
    tp, fp, fn, prec, rec, f1, shd = metrics_excluding_diag(adj_delta, delta_hat_pdiff)
    print(f"[DELTA-pdiff] topK={K_true_delta}: TP={tp} FP={fp} FN={fn} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} | SHD={shd}")

    print("\n[NEXT] If fast stage looks promising, set RUN_STAGE='formal' for final numbers.")

if __name__ == "__main__":
    main()

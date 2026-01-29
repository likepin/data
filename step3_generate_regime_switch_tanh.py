import os, json
import numpy as np

def set_seed(seed: int = 42):
    np.random.seed(seed)

def make_sparse_var_params(
    N: int, L: int, indeg: int = 2,
    w_min: float = 0.3, w_max: float = 0.7,
    self_lag: bool = True, self_lag_min: float = 0.2, self_lag_max: float = 0.4,
) -> np.ndarray:
    A = np.zeros((L, N, N), dtype=np.float32)

    if self_lag:
        for j in range(N):
            A[0, j, j] = np.random.uniform(self_lag_min, self_lag_max)

    for j in range(N):
        candidates = [i for i in range(N) if i != j]
        parents = np.random.choice(candidates, size=min(indeg, len(candidates)), replace=False)
        for i in parents:
            lag = np.random.randint(0, L)
            sign = np.random.choice([-1.0, 1.0])
            A[lag, j, i] = sign * np.random.uniform(w_min, w_max)

    return A

def spectral_radius_companion(A: np.ndarray) -> float:
    L, N, _ = A.shape
    C = np.zeros((N * L, N * L), dtype=np.float64)
    for l in range(L):
        C[0:N, l * N:(l + 1) * N] = A[l].astype(np.float64)
    if L > 1:
        C[N:, 0:-N] = np.eye(N * (L - 1), dtype=np.float64)
    eigvals = np.linalg.eigvals(C)
    return float(np.max(np.abs(eigvals)))

def stabilize_var_params(A: np.ndarray, target_rho: float = 0.95, max_iter: int = 8) -> np.ndarray:
    A2 = A.astype(np.float32)
    for it in range(max_iter):
        rho = spectral_radius_companion(A2)
        if not np.isfinite(rho) or rho <= 0:
            print("[stabilize] spectral radius invalid, stop.")
            return A2
        if rho <= target_rho:
            print(f"[stabilize] rho {rho:.3f} <= {target_rho} (iters={it})")
            return A2
        scale = target_rho / rho
        A2 = (A2 * scale).astype(np.float32)
        rho_new = spectral_radius_companion(A2)
        print(f"[stabilize] iter {it+1}: rho {rho:.3f} > {target_rho}, scale {scale:.4f} -> rho {rho_new:.3f}")
    print(f"[stabilize] max_iter reached, rho={spectral_radius_companion(A2):.3f}")
    return A2

def sample_deltaA(
    A_base: np.ndarray,
    K_edges: int = 6,
    w_min: float = 0.3,
    w_max: float = 0.6,
) -> np.ndarray:
    """
    Create DeltaA with K new edges not present in A_base.
    Only add cross-variable edges (i != j). Lags chosen randomly.
    """
    L, N, _ = A_base.shape
    DeltaA = np.zeros_like(A_base, dtype=np.float32)

    # eligible slots: where base is zero, and i != j
    candidates = []
    for l in range(L):
        for j in range(N):
            for i in range(N):
                if i == j:
                    continue
                if abs(float(A_base[l, j, i])) < 1e-8:
                    candidates.append((l, j, i))

    np.random.shuffle(candidates)
    chosen = candidates[:K_edges]

    for (l, j, i) in chosen:
        sign = np.random.choice([-1.0, 1.0])
        DeltaA[l, j, i] = sign * np.random.uniform(w_min, w_max)

    return DeltaA

def simulate_regime_switch_tanh(
    T: int, A0: np.ndarray, A1: np.ndarray, t_switch: int,
    sigma: float = 0.25, burnin: int = 200, gamma: float = 4.0
):
    L, N, _ = A0.shape
    total_T = T + burnin
    X = np.zeros((total_T, N), dtype=np.float32)
    s = np.zeros((total_T,), dtype=np.int32)

    X[:L] = np.random.normal(0, 1, size=(L, N)).astype(np.float32)

    for t in range(L, total_T):
        tt = t - burnin  # map to visible time
        if tt >= t_switch:
            A = A1
            s[t] = 1
        else:
            A = A0
            s[t] = 0

        z = np.zeros((N,), dtype=np.float32)
        for l in range(L):
            z += A[l] @ X[t - (l + 1)]
        x = np.tanh(gamma * z) + np.random.normal(0, sigma, size=(N,)).astype(np.float32)
        X[t] = x

    return X[burnin:], s[burnin:]

def collapsed_adj(A: np.ndarray) -> np.ndarray:
    return (np.abs(A).sum(axis=0) > 1e-8).astype(np.int32)

def main():
    set_seed(123)

    out_dir = "synthetic_step3"
    os.makedirs(out_dir, exist_ok=True)

    # ---- config ----
    N = 10
    L = 2
    T = 6000
    switch_ratio = 0.60
    t_switch = int(T * switch_ratio)

    indeg = 2
    sigma = 0.25
    gamma = 3.0
    burnin = 200

    K_delta_edges = 6   # 动态新增边数量（你可改 4/6/8）
    # ----------------

    A_base = make_sparse_var_params(N=N, L=L, indeg=indeg)
    A_base = stabilize_var_params(A_base, target_rho=0.95, max_iter=8)

    DeltaA = sample_deltaA(A_base, K_edges=K_delta_edges, w_min=0.3, w_max=0.6)

    # regime1 matrix
    A_regime1 = (A_base + DeltaA).astype(np.float32)
    A_regime1 = stabilize_var_params(A_regime1, target_rho=0.95, max_iter=8)

    X, s = simulate_regime_switch_tanh(
        T=T, A0=A_base, A1=A_regime1, t_switch=t_switch,
        sigma=sigma, burnin=burnin, gamma=gamma
    )

    assert np.isfinite(X).all(), "X contains NaN/Inf."

    adj_base = collapsed_adj(A_base)
    adj_regime1 = collapsed_adj(A_regime1)
    adj_delta = ((adj_regime1 == 1) & (adj_base == 0)).astype(np.int32)

    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "s.npy"), s)
    np.save(os.path.join(out_dir, "A_base.npy"), A_base)
    np.save(os.path.join(out_dir, "DeltaA.npy"), DeltaA)
    np.save(os.path.join(out_dir, "A_regime1.npy"), A_regime1)

    np.save(os.path.join(out_dir, "adj_base.npy"), adj_base)
    np.save(os.path.join(out_dir, "adj_regime1.npy"), adj_regime1)
    np.save(os.path.join(out_dir, "adj_delta.npy"), adj_delta)

    meta = dict(
        N=N, L=L, T=T, t_switch=t_switch, switch_ratio=switch_ratio,
        indeg=indeg, sigma=sigma, gamma=gamma, burnin=burnin,
        K_delta_edges=K_delta_edges
    )
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Saved to ./{out_dir}/")
    print(f"X: {X.shape}, s: {s.shape}, t_switch={t_switch} (ratio={switch_ratio})")
    print(f"Edges base={int(adj_base.sum())}, regime1={int(adj_regime1.sum())}, delta(add-only)={int(adj_delta.sum())}")

if __name__ == "__main__":
    main()

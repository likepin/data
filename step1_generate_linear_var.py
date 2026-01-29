import os
import numpy as np

def set_seed(seed: int = 42):
    np.random.seed(seed)

def make_sparse_var_params(
    N: int,
    L: int,
    indeg: int = 2,
    w_min: float = 0.3,
    w_max: float = 0.7,
    self_lag: bool = True,
    self_lag_min: float = 0.2,
    self_lag_max: float = 0.4,
) -> np.ndarray:
    """
    Create sparse VAR(L) coefficients.
    A shape: (L, N, N), where A[l, j, i] is coeff for x_{t-(l+1)}^i -> x_t^j.
    """
    A = np.zeros((L, N, N), dtype=np.float32)

    # Optional self-lag for stability / realism
    if self_lag:
        for j in range(N):
            A[0, j, j] = np.random.uniform(self_lag_min, self_lag_max)

    # Random sparse parents for each target j
    for j in range(N):
        candidates = [i for i in range(N) if i != j]
        if len(candidates) == 0:
            continue
        parents = np.random.choice(candidates, size=min(indeg, len(candidates)), replace=False)
        for i in parents:
            lag = np.random.randint(0, L)  # which lag carries this edge
            sign = np.random.choice([-1.0, 1.0])
            A[lag, j, i] = sign * np.random.uniform(w_min, w_max)

    return A

def spectral_radius_companion(A: np.ndarray) -> float:
    """
    Compute spectral radius of VAR(L) companion matrix.
    A: (L, N, N), A[l, j, i] is coeff i->j at lag l+1
    """
    L, N, _ = A.shape
    C = np.zeros((N * L, N * L), dtype=np.float64)

    # Top block row: [A1 A2 ... AL]
    for l in range(L):
        C[0:N, l * N:(l + 1) * N] = A[l].astype(np.float64)

    # Subdiagonal identity blocks
    if L > 1:
        C[N:, 0:-N] = np.eye(N * (L - 1), dtype=np.float64)

    eigvals = np.linalg.eigvals(C)
    rho = float(np.max(np.abs(eigvals)))
    return rho

def stabilize_var_params(A: np.ndarray, target_rho: float = 0.95, max_iter: int = 5) -> np.ndarray:
    """
    Iteratively scale VAR(L) coefficients so that companion spectral radius <= target_rho.
    This version is stricter and handles float32 rounding.
    """
    A2 = A.astype(np.float32)
    for it in range(max_iter):
        rho = spectral_radius_companion(A2)
        if not np.isfinite(rho) or rho <= 0:
            print("[stabilize] spectral radius is invalid, stop.")
            return A2

        if rho <= target_rho:
            if it == 0:
                print(f"[stabilize] rho {rho:.3f} <= {target_rho}, no scaling.")
            else:
                print(f"[stabilize] converged: rho {rho:.3f} <= {target_rho} in {it} iters.")
            return A2

        scale = target_rho / rho
        A2 = (A2 * scale).astype(np.float32)
        rho_new = spectral_radius_companion(A2)
        print(f"[stabilize] iter {it+1}: rho {rho:.3f} > {target_rho}, scale {scale:.4f} -> rho {rho_new:.3f}")

    # final check
    rho_final = spectral_radius_companion(A2)
    print(f"[stabilize] reached max_iter, final rho {rho_final:.3f} (target {target_rho})")
    return A2


def simulate_var(T: int, A: np.ndarray, sigma: float = 0.3, burnin: int = 200) -> np.ndarray:
    """
    Simulate VAR: x_t = sum_l A[l] x_{t-(l+1)} + eps_t
    """
    L, N, _ = A.shape
    total_T = T + burnin
    X = np.zeros((total_T, N), dtype=np.float32)

    # init
    X[:L] = np.random.normal(0, 1, size=(L, N)).astype(np.float32)

    for t in range(L, total_T):
        x = np.zeros((N,), dtype=np.float32)
        for l in range(L):
            # A[l] is (N,N), X[t-(l+1)] is (N,)
            x += A[l] @ X[t - (l + 1)]
        x += np.random.normal(0, sigma, size=(N,)).astype(np.float32)
        X[t] = x

    return X[burnin:]

def main():
    set_seed(123)

    out_dir = "synthetic_step1"
    os.makedirs(out_dir, exist_ok=True)

    # Default config for Step 1
    N = 10
    L = 2
    T = 6000
    indeg = 2
    sigma = 0.3
    burnin = 200

    # 1) sample sparse params
    A_true = make_sparse_var_params(
        N=N, L=L, indeg=indeg,
        w_min=0.3, w_max=0.7,
        self_lag=True, self_lag_min=0.2, self_lag_max=0.4
    )

    # 2) stabilize (guarantee no explosion)
    A_true = stabilize_var_params(A_true, target_rho=0.95)

    # 3) simulate
    X = simulate_var(T=T, A=A_true, sigma=sigma, burnin=burnin)

    # 4) safety check
    if not np.isfinite(X).all():
        nan_cnt = int(np.isnan(X).sum())
        inf_cnt = int(np.isinf(X).sum())
        raise ValueError(f"X has NaN/Inf: nan={nan_cnt}, inf={inf_cnt}")

    # adjacency collapsed across lags: edge i->j exists if any lag has nonzero coeff
    adj_true = (np.abs(A_true).sum(axis=0) > 1e-8).astype(np.int32)

    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "A_true.npy"), A_true)
    np.save(os.path.join(out_dir, "adj_true.npy"), adj_true)

    print(f"[OK] Saved to ./{out_dir}/")
    print("X:", X.shape, "A_true:", A_true.shape, "adj_true:", adj_true.shape)
    print("True edges (collapsed):", int(adj_true.sum()))

if __name__ == "__main__":
    main()

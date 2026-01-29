import os
import numpy as np
import matplotlib.pyplot as plt

def bin_metrics(adj_true: np.ndarray, adj_hat: np.ndarray):
    # Flatten, exclude diagonal if you want (often done). Here we exclude diag by default.
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

    shd = int(np.abs(y_true - y_hat).sum())  # edge disagreements excluding diag

    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1, shd=shd)

def plot_adj(adj, title, path):
    plt.figure(figsize=(5, 5))
    plt.imshow(adj, interpolation="nearest")
    plt.title(title)
    plt.xlabel("source i")
    plt.ylabel("target j")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    data_dir = "synthetic_step1"
    adj_true = np.load(os.path.join(data_dir, "adj_true.npy"))
    adj_hat = np.load(os.path.join(data_dir, "adj_hat.npy"))

    m = bin_metrics(adj_true, adj_hat)
    print("=== Step 1 Graph Recovery (collapsed over lags, diag excluded) ===")
    print(f"TP={m['tp']} FP={m['fp']} FN={m['fn']}")
    print(f"Precision={m['precision']:.3f} Recall={m['recall']:.3f} F1={m['f1']:.3f}")
    print(f"SHD={m['shd']}")

    plot_adj(adj_true, "True adjacency (collapsed)", os.path.join(data_dir, "adj_true.png"))
    plot_adj(adj_hat,  "PCMCI ParCorr adjacency (collapsed)", os.path.join(data_dir, "adj_hat.png"))

    print("[OK] Saved plots: adj_true.png, adj_hat.png")

if __name__ == "__main__":
    main()

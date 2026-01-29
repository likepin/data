import os
import numpy as np
import matplotlib.pyplot as plt

import re  # 【新增】：导入 re 模块，用于正则匹配文件名，解析 knn 和 alpha

# bin_metrics 函数未变：计算排除对角线的性能指标，返回字典
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

# plot_adj 函数未变：绘制并保存单个热力图
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
    data_dir = "synthetic_step2"
    adj_true = np.load(os.path.join(data_dir, "adj_true.npy"))

    # 先处理 ParCorr 文件（如果存在），作为线性方法的基准
    # 这部分是可选的，如果 parcorr_adj_hat.npy 存在，就计算并画图
    parcorr_file = "parcorr_adj_hat.npy"
    parcorr_path = os.path.join(data_dir, parcorr_file)
    if os.path.exists(parcorr_path):
        adj_hat = np.load(parcorr_path)
        m = bin_metrics(adj_true, adj_hat)
        print("=== ParCorr Graph Recovery (collapsed over lags, diag excluded) ===")
        print(f"TP={m['tp']} FP={m['fp']} FN={m['fn']}")
        print(f"Precision={m['precision']:.3f} Recall={m['recall']:.3f} F1={m['f1']:.3f}")
        print(f"SHD={m['shd']}")
        
        # 保存 ParCorr 专属热力图
        plot_adj(adj_hat, "PCMCI ParCorr adjacency (collapsed)", 
                 os.path.join(data_dir, "adj_hat_parcorr.png"))
        print("[OK] Processed ParCorr and saved adj_hat_parcorr.png")

   
    # 定义正则模式，匹配 "cmiknn_knnXX_alphaY.YYY_adj_hat.npy"
    # 组1: knn 的数字，组2: alpha 的浮点数
    pattern = re.compile(r'cmiknn_knn(\d+)_alpha(\d+\.\d+)_adj_hat\.npy')
    
    # 遍历目录所有文件
    for filename in os.listdir(data_dir):
        match = pattern.match(filename)  # 尝试匹配
        if match:                        # 如果匹配成功
            knn = int(match.group(1))    # 提取 knn（整数）
            alpha = float(match.group(2))# 提取 alpha（浮点数）
            
            # 加载该文件
            adj_hat_path = os.path.join(data_dir, filename)
            adj_hat = np.load(adj_hat_path)
            
            # 计算并打印指标（打印中带上 knn 和 alpha）
            m = bin_metrics(adj_true, adj_hat)
            print(f"=== CMIKNN (knn={knn}, alpha={alpha:.3f}) Graph Recovery (collapsed over lags, diag excluded) ===")
            print(f"TP={m['tp']} FP={m['fp']} FN={m['fn']}")
            print(f"Precision={m['precision']:.3f} Recall={m['recall']:.3f} F1={m['f1']:.3f}")
            print(f"SHD={m['shd']}")
            
            # 保存对应的热力图，文件名动态生成（带 knn 和 alpha）
            plot_path = os.path.join(data_dir, f"adj_hat_cmiknn_knn{knn}_alpha{alpha:.3f}.png")
            plot_adj(adj_hat, f"PCMCI CMIKNN (knn={knn}, alpha={alpha:.3f}) adjacency (collapsed)", plot_path)
            print(f"[OK] Processed {filename} and saved {os.path.basename(plot_path)}")

    # 真实图只画一次（在最后，避免重复）
    plot_adj(adj_true, "True adjacency (collapsed)", os.path.join(data_dir, "adj_true.png"))
    print("[OK] Saved adj_true.png")

if __name__ == "__main__":
    main()
### P0：建立“单一真相”的图语义与二值化接口（必须先做）

1. 新建 `graph_io.py`
   - `load_adj(path) -> np.ndarray`
   - `binarize_adj(adj, mode, alpha=None, tau=None) -> np.int32`
     - `mode="binary"`：直接 (adj!=0)
     - `mode="val"`：abs(adj) > tau
     - `mode="pval"`：pval < alpha
   - `assert_orientation(adj, convention="tgt_src")`：只做注释+可选检查
2. 新建 `project_convention.md`
   - 明确：`adj[target, source]=1` 表示 `source->target`

### P1：把 `find_pred_change_adj` 重写成“配置驱动”而不是“猜”

1. 新增 `step5_config.json`（你已经有雏形）
   - 明确字段：
     - `N_source`: `"X.npy"` or `"adj_base.npy"`
     - `pred_change_source`: `"valdiff"` / `"signflip"` / `"adjhat_xor"` / `"topk_csv"`
     - `pred_prefix`: 比如 `"parcorr"` 或 `"cmiknn"`
     - `binarize`: `{ "mode": "pval", "alpha": 0.02 }` 或 `{ "mode": "val", "tau": 0.3 }`
2. `find_pred_change_adj(data_dir, cfg)` 的行为：
   - **只按 cfg 指定策略找**；找不到就报错并列出候选文件（不要 silent fallback）
   - XOR 时：
     - 先 `adj0_bin = binarize_adj(adj0, ...)`
     - 再 `pred = adj0_bin ^ adj1_bin`
   - CSV 时：
     - `N` 统一从 `X.shape[1]`（或 cfg 指定）
     - 严格读取 `src,tgt,score`（并校验 src/tgt < N）

### P2：为 Step5 增加“loader 自检 + 单元测试”（避免再次踩坑）

1. 新建 `tests/test_pred_change_loader.py`
   - 用 synthetic_step3_v2 的已知 `adj_true_change.npy`
   - 测试每个策略输出：
     - shape == (N,N)
     - diag==0
     - 值域是 {0,1}
   - 测试方向：随机抽 10 条边 `(src,tgt)`，验证在矩阵位置是 `[tgt,src]==1`
2. 新增 `step5_debug_dump.py`
   - 打印：
     - pred_change edges 数、top10 边（含坐标）
     - 与 true_change 的重合度（TP/FP/FN）
     - 选用的文件名与 binarize 参数

### P3：修复 Step5 “表格 nan / 极端值” 的统计逻辑

（你现在表里出现 `nan`，多半是分母为 0）

1. 在 retention 统计中，所有 ratio 都要：
   - `den = max(den, 1e-12)`
   - 或者如果 den=0，显式输出 `0.0` 并标记 `den_is_zero=True`
2. 输出表格同时写入：
   - `n_pred_edges`, `n_true_edges`, `n_active_edges_high`, `n_active_edges_low`
   - 这样你一眼能看出是不是“pred 太大/太小导致 ratio 崩
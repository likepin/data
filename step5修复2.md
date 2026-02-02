## Phase 0 — 目录扫描与定位（只做信息收集，不改逻辑）

### 0.1 输出项目关键文件树

**目标**：让后续改动精确落点（不要猜文件名）。

**动作**

- 在 repo 根目录写一个脚本或直接在现有脚本里加 `--print_tree`：
  - 打印 `synthetic_step3_v2/` 下所有与以下关键字有关的文件：
    - `DeltaA`, `adj_change`, `adj_base`, `regime0`, `regime1`, `val_matrix`, `p_matrix`, `chg_pred`, `valdiff`, `signflip`, `cmiknn`, `parcorr`
- 将匹配结果写到 `synthetic_step3_v2/exports_step5/debug_tree.txt`

**验收**

- `debug_tree.txt` 存在，且能看到：
  - `DeltaA.npy`
  - `*_regime0_*`、`*_regime1_*` 的估计图或 val/p 矩阵
  - `adj_base.npy`（或等价文件）

------

## Phase 1 — 统一 true change：强制从 DeltaA 派生（P0 必做）

### 1.1 在 step5_utils 中新增真值派生函数

**文件**：`step5_utils.py`

**新增函数**

- `find_true_change_adj(data_dir, logs, prefer_deltaA=True)`

**逻辑**

- 若 `prefer_deltaA=True` 且存在 `DeltaA.npy`：
  - load `DeltaA` shape `(L,N,N)`
  - `adj_true = (np.abs(DeltaA) > 0).any(axis=0).astype(np.int32)`  # tgt_src convention
  - 保存缓存：`adj_change_true_from_DeltaA.npy`（可选）
  - return `adj_true`
- 否则 fallback 到旧的 `adj_change_true.npy / adj_change.npy`

**额外 sanity**

- 打印并写入 logs：
  - `DeltaA shape`
  - `K_true=len(edges_from_adj(adj_true))`
  - 检查变化边是否都属于 base：`adj_base[tgt,src]==1`，否则打印 violations

**验收命令**

```
python step5_proxy_gating_effect.py --data_dir synthetic_step3_v2 --sanity
```

**通过标准**

- 日志显示 `true_change from DeltaA`
- `K_true == 6`（或与你生成器一致）
- `violations == 0`

------

## Phase 2 — 禁用 adjhat_xor：强度变化实验中不允许结构 XOR（P0 必做）

### 2.1 修改 find_pred_change_adj：禁止 adjhat_xor

**文件**：`step5_utils.py`（或你目前实现 `find_pred_change_adj` 的文件）

**改动**

- 如果 `cfg["pred_change_source"] == "adjhat_xor"`：
  - 直接 `raise ValueError("Step3_v2 strength-change does NOT support adjhat_xor. Use valdiff/signflip.")`

**验收**

- 把 config 临时设成 adjhat_xor 时，脚本应立刻报错且提示正确做法。

------

## Phase 3 — 实现 pred_change_source：valdiff/signflip（只在 base edges 上取 TopK）（P0 必做）

### 3.1 新增变化分数计算函数

**文件**：建议新建 `step5_pred.py`，并在 utils 中引用（避免 step5_utils 过肥）

**新增函数**

- `compute_change_scores(A0, A1, mode="valdiff"|"signflip") -> score_matrix`
  - valdiff：`abs(A1 - A0)`
  - signflip：`(sign(A1) != sign(A0)) * abs(A1 - A0)`（推荐）

### 3.2 新增“只在 base edges 上取 TopK”的二值化函数

**新增函数**

- `binarize_topk_on_base(score, adj_base, top_k, diag_excluded=True) -> pred_adj, score_dict`
  - score 先乘 `(adj_base!=0)`，只保留 base edges
  - 排除对角线
  - 取 top_k 个 (tgt,src) 位置置 1
  - `score_dict` 用 `{(src,tgt): score[tgt,src]}`

### 3.3 修改 find_pred_change_adj：支持 valdiff/signflip_on_base

**文件**：`step5_utils.py`

**新增 pred_source**

- `"valdiff_on_base"`
- `"signflip_on_base"`

**加载输入**

- 从 `data_dir` 中加载同 prefix 的两段“带权重图”
  - 优先：`{prefix}_regime0_val_matrix.npy` / `{prefix}_regime1_val_matrix.npy`
  - 若没有 val_matrix，则退而求其次用 `{prefix}_regime0_adj_hat.npy` / `{prefix}_regime1_adj_hat.npy`
- 加载 `adj_base.npy`（或 `find_base_adj` 返回的 base）

**输出**

- `pred_adj` 二值变化边（TopK，默认 K_true）
- `pred_scores`（score_dict）

### 3.4 更新 step5_config.json 默认配置（强制 cmiknn）

**文件**：`synthetic_step3_v2/step5_config.json`

**默认写**

```
{
  "pred_change_source": "valdiff_on_base",
  "pred_prefix": "cmiknn",
  "top_k": 6
}
```

> pred_prefix 需要与你目录里的 cmiknn 文件前缀一致（可能是 cmiknn_knn20_alpha0.020 等）。Codex 应自动匹配：如果 exact prefix 不存在，就列出候选并报错（不要 silently fallback）。

**验收命令**

```
python step5_proxy_gating_effect.py --data_dir synthetic_step3_v2 --sanity
```

**通过标准**

- `K_true == 6`
- `K_pred == 6`
- `TP > 0`（理想 ≥4）
- 不再出现 `TP=0` 的死局

------

## Phase 4 — Step5 输出增强：门控方向与子集退化处理（P1 强烈建议）

### 4.1 输出 gate 方向一致性统计

**文件**：`step5_proxy_gating_effect.py`

**新增输出列/日志**

- `mean_lambda_high/low/all`
- `mean_gate_weight_high/low/all`（默认 gate_weight = 1 - lambda）
- 对 hard gate：`p_active = mean(lambda < tau)`
- 对 soft gate：`p_active = mean(gate_weight > w_thresh)` 或你现有 soft_mode 的一致版本

**通过标准**

- `mean_lambda_high > mean_lambda_low`
- `p_active_high < p_active_low`（危险区更少启用 ΔA）

### 4.2 修复 high subset 退化（high_thr=1.0 全饱和）

**改动**

- 若 high subset 全部 λ=1（或方差过小）：
  - 自动提升 high_q（如 0.95→0.98）
  - 或切换到 “topk_non_saturated” 模式（排除 λ==1 再取 quantile）
- 将最终采用的 subset 切法写进 `config_used.json`

**通过标准**

- high 子集不是全 1（或至少有足够变化点），否则输出 WARN 并记录处理策略

------

## Phase 5 — 回归测试（防止以后又被 XOR/前缀 fallback 搞坏）（P1）

### 5.1 新增 tests（或最小 sanity 脚本）

**文件**：`tests/test_step5_strength_change.py`（没有 tests 目录就新建）

**测试内容**

- 若 `DeltaA.npy` 存在：
  - `K_true` 应等于生成器设定（默认 6）
- 若 `pred_change_source` 包含 `_on_base` 且 `top_k=K_true`：
  - `K_pred == K_true`
  - `TP > 0`
- 如果配置为 `adjhat_xor`：
  - 必须 raise（强度变化版禁止）

**通过标准**

- `pytest -q` 通过（或 python 运行测试脚本输出 PASS）

------

# Codex 执行顺序建议（不要跳）

1. Phase 0 打印树确认文件名
2. Phase 1 真值从 DeltaA 派生
3. Phase 2 禁用 XOR
4. Phase 3 实现 valdiff/signflip_on_base 并默认 cmiknn
5. Phase 4 输出增强 + 子集退化修复
6. Phase 5 加回归测试
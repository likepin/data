# Phase 0 — 新增脚本与配置（组织结构）

### 0.1 新建脚本

**新增文件**：`step5pp_simulate_gated_graph.py`

### 0.2 配置文件

在 `synthetic_step3_v2/step5pp_config.json` 写默认配置（若已存在则兼容）
 建议字段：

```
{
  "pred_prefix": "cmiknn",
  "score_type": "valdiff", 
  "delta_mode": "A1_minus_A0", 
  "gate_mode": "soft",
  "tau_hard": 0.8,
  "w_soft": null,
  "subset_high_q": 0.90,
  "subset_low_q": 0.50,
  "edge_mask": "base_only",
  "norm": "none",
  "output_topk_edges": 20
}
```

**验收**

- `python step5pp_simulate_gated_graph.py --data_dir synthetic_step3_v2 --sanity` 能跑起来并生成 out_dir

------

# Phase 1 — 统一加载：A_base / A0 / A1 / λ / t_switch（P0）

### 1.1 加载图矩阵并统一方向

**复用/增强函数**（如果已有就直接用）：

- `load_adj(...)`：返回 float 矩阵（保留强度）
- `assert_orientation(adj, "tgt_src")`

**加载项**

- `A_base`：从 base 文件（binary 也行，后面会 cast float）
- `A0`：cmiknn regime0（优先 val_matrix，其次 adj_hat）
- `A1`：cmiknn regime1（同上）
- `lambda_t, valid_mask, t_switch`：复用 step5 的 lambda loader

**sanity 输出**

- `A_base nnz`
- `A0/A1 min/max/mean, nnz`
- `lambda stats` + 是否 high_thr=1 饱和

**通过标准**

- 方向检查 OK
- A0/A1 不是全 0/全 NaN

------

# Phase 2 — 构造 ΔA_proxy 与变化边集合（P0）

### 2.1 ΔA_proxy 定义（强度变化版本）

**核心定义（默认）**：

- `ΔA_proxy = A1 - A0`（保留符号）
- `ΔA_mag = abs(ΔA_proxy)`（变化幅度）

### 2.2 边掩码（只在 base edges 上评估）

如果 `edge_mask="base_only"`：

- `ΔA_proxy *= (A_base != 0)`
- `ΔA_mag *= (A_base != 0)`
- 排除对角线

### 2.3 变化边集合（pred/topK）

- `pred_change_edges_topK`: 从 `ΔA_mag` 取 topK（默认 K_true=6 或 config 指定）
- 输出 csv：`exports_step5pp/pred_topk_edges.csv`（含 src,tgt,delta_mag,delta_signed）

**通过标准**

- topK 边可打印检查（含你看到过的那几条真边）
- K_pred 可控

------

# Phase 3 — 门控函数 g(t) 与有效图 A_eff(t)（P0）

### 3.1 定义 gate_weight(t)

提供两种模式，默认 soft：

- **soft**：`g(t) = 1 - lambda_t`
- **hard**：`g(t) = 1(lambda_t < tau_hard)`

（注意：你的哲学是 λ 大→关 ΔA，所以 g(t) 必须随 λ 单调下降）

### 3.2 构造有效图序列

我们不需要存 T 个 N×N（太大），只做统计即可。

定义：

- `A_eff(t) = A_base + g(t) * ΔA_proxy`
  - 这里 `A_base` 若是 binary，可先转 float
  - 可选 `clip`：限制 A_eff 在合理范围（例如 [-1,1]），默认不 clip

------

# Phase 4 — 关键评估指标（P0）

> 你的目标是证明：**危险区回归 base**、安全区允许动态修正靠近变化态。
>  所以必须比较 `A_eff` 与 `A_base/A0/A1` 的距离。

### 4.1 距离定义（建议两类都做）

对矩阵差分只在 `edge_mask` 范围内计算：

- `dist_base(t) = ||A_eff(t) - A_base||_1 / M`
- `dist_reg0(t) = ||A_eff(t) - A0||_1 / M`
- `dist_reg1(t) = ||A_eff(t) - A1||_1 / M`

其中 M 是被评估边数（避免尺度随 N 变化）

### 4.2 high/low 子集统计

对 mask：

- `high_mask`（默认 top10% λ）
- `low_mask`（默认 bottom50% λ）

输出表（CSV & MD）：

- `mean(dist_base)`、`mean(dist_reg0)`、`mean(dist_reg1)`
- `mean(g(t))`、`p_active`
- `mean_lambda`

**核心预期（写进结论）**

- high 子集：`dist_base` 最小（最接近 base）
- low 子集：`dist_reg1` 或 `dist_reg0` 更小（取决于切换后/前定义），至少应明显偏离 base

------

# Phase 5 — “变化边强度保留曲线”（P1，论文卖点很强）

### 5.1 在 pred_topK_edges 上评估门控保留

对每个 t：

- `retained_strength(t) = mean_{e in topK} |g(t) * ΔA_proxy[e]|`
- `raw_strength = mean_{e in topK} |ΔA_proxy[e]|`（常数）
- 输出 `retained_ratio(t) = retained_strength(t) / (raw_strength + eps)`

### 5.2 分子集统计

- high 子集 retained_ratio 应接近 0
- low 子集 retained_ratio 应接近 1（soft）或 1（hard）视 tau

输出：

- `retained_curve.png`
- `retained_summary.csv`

------

# Phase 6 — 处理 high_thr=1 饱和（必须纳入 step5++ 输出）（P1）

你已经遇到：high subset 全 1，导致 gate 完全关，这是预期但需要“可解释”。

### 6.1 新增“非饱和 high 子集”（可选）

新增一个备选 mask：

- `high_mask_non_sat`: 在 `lambda < 1` 的点里取 top q（如 0.90）
- 若非饱和点过少则跳过并 WARN

输出比较：

- high (sat) vs high_non_sat 的距离/保留率差异

这样你能在论文里写：

> 当 λ 饱和时门控完全关闭；在非饱和高 λ 区域门控仍呈连续衰减。

------

# Phase 7 — 可视化输出（P0）

生成一张 PPT 友好图：`gated_graph_simulation.png`（建议 3 个panel）

1. λ(t) 与 g(t) 曲线（标注 t_switch）
2. dist_base/dist_reg0/dist_reg1 三条距离曲线（随 t）
3. retained_ratio(t) 曲线（随 t）

并输出一份 `exports_step5pp/step5pp_summary.md`：

- 表格：high/low/all 的核心均值指标
- 结论 bullet（自动生成）

------

# Phase 8 — 回归测试（P1）

新增 `tests/test_step5pp.py` 或最小 sanity：

- 断言：
  - `mean_gate_high < mean_gate_low`
  - `mean_dist_base_high < mean_dist_base_low`（危险区更接近 base）
  - `retained_ratio_high` 接近 0（阈值比如 <0.05）

------

# ✅ 运行命令（Codex 必须写入 README / logs）

```
python step5pp_simulate_gated_graph.py --data_dir synthetic_step3_v2 --sanity
```

输出目录：

- `synthetic_step3_v2/exports_step5pp/`
  - `step5pp_summary.csv`
  - `step5pp_summary.md`
  - `pred_topk_edges.csv`
  - `gated_graph_simulation.png`
  - `retained_curve.png`
  - `config_used.json`
  - `logs.txt`

------

# ✅ 成功标准（你看一眼就能判断）

1. high 子集：

- `mean_lambda_high≈1`
- `mean_gate_high≈0`
- `mean(dist_base)` 最小
- `retained_ratio_high≈0`

1. low 子集：

- `mean_gate_low` 明显大
- `dist_base` 变大（说明允许偏离 base）
- `retained_ratio_low` 明显 > 0（soft 模式接近 0.9）
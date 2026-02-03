## P0-1：修复 union 掩码“稀释信号”问题（rel 变常数的根因）

### 目标

在 `dist_mask_mode=union_base_predchange`（或其它 union 类）时，距离计算不要平均在一堆“无 regime 差异的边”上。
 要让 dist_mask 聚焦到 **真正有 regime 差异的边（|A1-A0| 大）**，这样 rel 才不会恒定。

### 修改文件

- `step5pp_simulate_gated_graph.py`

### 新增配置项（step5pp_config.json）

```
{
  "dist_mask_mode": "union_delta_topk",
  "dist_topk": 6
}
```

可选支持：

- `"dist_delta_thr": 0.01`（阈值模式）
- topk 和 thr 二选一，优先 topk

### 实现方式（核心逻辑）

新增 `dist_mask_mode = "union_delta_topk"`：

1. 先算参考差异（只在候选 mask 上取）

- `delta_ref = abs(A1_eff - A0_eff)`（注意 A0_eff/A1_eff 在 mask 外置 0 或不计）

1. 候选集合：

- `cand = union_mask > 0`（union_base_predchange / union_base_predchange / etc）

1. 在 `cand` 内选 topk：

- 找出 cand 位置的 `delta_ref` 展平
- 取 topk 的阈值（第 K 大）
- dist_mask = cand & (delta_ref >= thr)

> 直观：union 仍可用，但 dist 的统计只用最“有 regime 差异”的那几条边。

### 验收标准

执行：

```
python step5pp_simulate_gated_graph.py --data_dir synthetic_step3_v2 --sanity
```

期望：

- `rel_pre_std` 不再为 0
- `dist_std_reg0` 与 `dist_std_reg1` 不再几乎完全相同（至少均值或 std 有区分）
- 输出里 `dist_mask nnz` 应接近 `dist_topk`（例如 6）

------

## P0-2：明确 pre/post 与 A0/A1 的 regime 对应（防止方向错）

### 目标

确定 `A0` 对应 regime0、`A1` 对应 regime1 是否与 `t_switch` 对齐。
 避免出现“margin 正但 rel 负 / align 不一致”的现象。

### 修改文件

- `step5pp_simulate_gated_graph.py`

### 新增 sanity 打印（必须）

基于 `t_switch` 拆分：

- `pre_mask = valid_mask & (t < t_switch)`
- `post_mask = valid_mask & (t >= t_switch)`

打印 4 个均值：

- `mean(dist_reg0_pre), mean(dist_reg1_pre)`
- `mean(dist_reg0_post), mean(dist_reg1_post)`

### 验收标准（逻辑判断）

按你的叙事：**pre 段更像 A0，post 段（低λ gate大）更像 A1**
 所以期望：

- pre：`mean(dist_reg0_pre) < mean(dist_reg1_pre)`
- post：`mean(dist_reg1_post) < mean(dist_reg0_post)`（至少在 low 子集更明显）

如果反过来：
 → 需要在 sanity 输出里给出提示：

- `WARN: A0/A1 may be swapped wrt t_switch.`

------

## P0-3：统一 rel / margin / align 的符号与定义（修掉“不一致”）

### 目标

让三个指标表达同一件事，避免出现：

- `align_pre=1` 但 `rel_pre_mean<0` 这种冲突

### 统一推荐定义（强制）

- `rel(t) = dist_reg0(t) - dist_reg1(t)`
  - rel > 0 表示 **更像 reg1**（因为 dist_reg1 更小）
- `margin_pre/post = mean(rel in segment)`
- `align_pre/post = mean( rel > 0 )`（更像 reg1 的比例）
- overall align 可选 `mean( rel > 0 )`（或只算 post 段）

### 修改点

确认现有 rel/align/margin 的计算是否一致，如果不是：

- 改成上面定义
- sanity 打印 `check: align_pre ≈ mean(rel_pre>0)`，差异过大则报警

### 验收标准

sanity 输出应满足：

- `align_pre == fraction(rel_pre > 0)`（误差 < 1e-6）
- `margin_pre` 与 `rel_pre_mean` 同号、数值一致
- pre/post 的 margin 符号符合预期（后面还要配合门控方向）

------

## P1-4：写成强断言（sanity 模式下自动判定“方向对不对”）

### 目标

不靠肉眼看图，sanity 一跑就知道是否符合设计叙事。

### 断言/检查（仅 sanity 模式）

1. 门控方向检查（你想要 λ大→gate小）

- `mean_gate_high < mean_gate_low`

1. “高 λ（危险区）更像 A0”

- 在 `high_non_sat` 或 `high_mask` 上：
  - `mean(dist_reg0[mask]) < mean(dist_reg1[mask])`

1. “低 λ（安全区）更像 A1”

- 在 `low_mask` 上：
  - `mean(dist_reg1[low]) < mean(dist_reg0[low])`

若失败：

- 打印明确原因：
  - “Gate direction mismatch” / “A0/A1 swapped” / “rel sign convention mismatch”
     不要直接 crash（可以先 warning，再可选 `--strict` 才 assert）。

### 验收标准

sanity 输出里显示：

- `[OK] gate direction`
- `[OK] high subset closer to A0`
- `[OK] low subset closer to A1`

------

## P1-5：受控极端验证（g=0 与 g=1）

### 目标

彻底排除“距离函数/掩码逻辑”写错的可能性。

### 实现

sanity 下做两次强制门控：

- `g_force=0`: `A_eff=A0` → `dist(A_eff,A0)=0`
- `g_force=1`: `A_eff=A1` → `dist(A_eff,A1)=0`

输出：

- `g_force=0 dist_to_A0=... dist_to_A1=...`
- `g_force=1 dist_to_A0=... dist_to_A1=...`

### 验收标准

- `g_force=0 dist_to_A0 < 1e-8`
- `g_force=1 dist_to_A1 < 1e-8`

------

## P2-6：补充 compare 输出表（自动汇总 base_only vs union vs union_delta_topk）

### 目标

把你现在手动对比的结果自动写到一张表里（你已经做了一半）。

### 输出文件

- `exports_step5pp/compare_masks.md`
- `exports_step5pp/compare_masks.csv`

每行至少包含：

- mask 模式（delta_mask_mode / dist_mask_mode）
- dist_std_base/reg0/reg1
- mean_dist_base_high/low
- mean_retained_high/low
- align_post_low（或 align_low_post）
- dist_mask nnz

### 验收标准

执行一次脚本即可生成 compare 表，且 union_delta_topk 在 rel/std 上表现更“有区分度”。

------

# 建议的默认配置（给你/给 Codex）

先跑三套：

### cfg_base_only.json

```
{
  "pred_prefix": "cmiknn",
  "delta_mask_mode": "base_only",
  "dist_mask_mode": "base_only",
  "gate_mode": "soft"
}
```

### cfg_union.json

```
{
  "pred_prefix": "cmiknn",
  "delta_mask_mode": "union_base_predchange",
  "dist_mask_mode": "union_base_predchange",
  "gate_mode": "soft"
}
```

### cfg_union_delta_topk.json（重点）

```
{
  "pred_prefix": "cmiknn",
  "delta_mask_mode": "union_base_predchange",
  "dist_mask_mode": "union_delta_topk",
  "dist_topk": 6,
  "gate_mode": "soft"
}
```

------

# 最终验收命令（一次性）

```
python step5pp_simulate_gated_graph.py --data_dir synthetic_step3_v2 --config cfg_base_only.json --sanity
python step5pp_simulate_gated_graph.py --data_dir synthetic_step3_v2 --config cfg_union.json --sanity
python step5pp_simulate_gated_graph.py --data_dir synthetic_step3_v2 --config cfg_union_delta_topk.json --sanity
```

你要看到的关键现象：

- union_delta_topk：`rel_pre_std/post_std` 非 0，`dist_std_reg0` 与 `dist_std_reg1` 有区分
- high_non_sat（或 high）：更像 A0（dist_reg0 更小）
- low：更像 A1（dist_reg1 更小）
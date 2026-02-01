# ✅ Step5 工程化计划

## 0) 总目标（写在 README 顶部）

在 `synthetic_step3_v2/` 上做 **proxy 门控有效性评估**，验证：

- High-λ（危险样本）上：门控应当 **显著减少** 错误变化边（FP/SHD）
- Low-λ（安全样本）上：门控应当 **尽量不损失** 变化边恢复（F1）

最终产物：

- 一张可直接放论文的 **Table 5-1/5-2/5-3**（md + csv）
- 一张关键图 **Figure 5-1**（λ 曲线 + gated 边数曲线）
- 一个 top-K 边明细 csv（可解释）

------

## 1) 目录规范（必须按此输出）

在 `synthetic_step3_v2/` 下创建：

```
exports_step5/
  step5_proxy_summary.csv
  step5_proxy_summary.md
  step5_tau_sweep.csv
  step5_tau_sweep.md
  step5_edge_retention.csv
  step5_edge_retention.md
  top_edges_highlambda.csv
  top_edges_lowlambda.csv
  gating_curve_demo.png
  config_used.json
  logs.txt
```

------

## 2) 输入自动探测规则（无脑可跑）

脚本默认 `--data_dir synthetic_step3_v2`，自动找以下文件；找不到才要求手动传参。

### 2.1 λ 相关（Step4 输出）

优先级：

1. `exports_step4/best_lambda_t.npy`
2. `exports_step4/lambda_t.npy`
3. `lambda_indexed.npz`（读取 `lambda_t` 与 `valid_mask`）

valid mask：

- `exports_step4/lambda_valid_mask.npy` 或 `lambda_indexed.npz` 里的 `valid_mask`
- 没有就全 True（但要在日志里 warn）

### 2.2 变化边与真值（Step3 V2 输出）

真变化边：

- `adj_change_true.npy` 或文件名含 `change_true`

基础图（静态骨架）：

- `adj_base_true.npy` 或 `adj_regime0_true.npy`（优先 base）

预测变化边（ΔA_pred）至少支持两种来源：

- `chg_pred_by_valdiff.npy`（文件名含 `valdiff` 或 `chg_pred`）
- `chg_pred_by_signflip.npy`（可选）
   如果都没有，就退而求其次：用 `adj_regime1_hat XOR adj_regime0_hat`（但要 warn）

> 备注：所有矩阵都按 **diag excluded** 评估（对角线不计）。

------

## 3) 统一的输出 schema（列名固定，方便复制到论文）

### 3.1 `step5_proxy_summary.csv`（对应 Table 5-1）

每行 = 一个 setting（包含 subset 与门控与否）

列名（必须完全一致）：

- `lambda_config` 例如 `(50,3)`
- `deltaA_source` 例如 `valdiff_topK6`
- `gate_type` 取值：`ungated` / `hard` / `soft`
- `tau`（hard/soft 的阈值；ungated 为空）
- `subset` 取值：`high` / `low` / `all`
- `subset_q` 例如 `high_q=0.90,low_q=0.50`
- `K_true_change`
- `TP` `FP` `FN` `Prec` `Rec` `F1` `SHD`
- `SHD_gain_vs_ungated`（仅 high/all 填：`SHD_ungated - SHD_gated`）
- `F1_delta_vs_ungated`（仅 low/all 填：`F1_gated - F1_ungated`）

### 3.2 `step5_tau_sweep.csv`（对应 Table 5-2）

每行 = 一个 τ setting（固定 high/low）

列名：

- `lambda_config`
- `deltaA_source`
- `gate_type`（hard/soft）
- `tau`
- `high_SHD_gain`
- `low_F1_delta`
- `high_F1` `high_SHD`
- `low_F1` `low_SHD`
- `pick_best`（bool：是否 Pareto 最优，或你指定规则最佳）

### 3.3 `step5_edge_retention.csv`（对应 Table 5-3）

每行 = 一个 subset（high/low），统计变化边保留情况

列名：

- `lambda_config`
- `deltaA_source`
- `gate_type`
- `tau`
- `subset`（high/low）
- `K_pred`（pred_change 边数）
- `TP_change`（pred ∩ true_change）
- `FP_change`（pred \ true_change）
- `retained_ratio`（门控后保留的 pred_change 比例）
- `true_retained_ratio`（门控后保留的 TP_change 比例）
- `fp_removed_ratio`（门控去掉 FP 的比例）

### 3.4 `top_edges_highlambda.csv / lowlambda.csv`

列名：

- `rank`
- `src`
- `tgt`
- `score`（如果有 valdiff/score）
- `is_true_change`（0/1）
- `mean_lambda`（该 subset 的 λ 均值）
- `mean_gate_weight`（soft gating 时的 `mean(1-λ)`）
- `count_active`（hard gating 下在 subset 内被激活次数）

------

## 4) 门控实现（必须支持 hard + soft 两套）

### 4.1 Hard gate（阈值 τ）

定义 gate mask：

- `active_t = (lambda_t < tau)`
   在 subset 内：
- 若 inactive（高 λ）：pred_change 不生效（只用 base）
- 若 active：pred_change 全生效

你不需要真的生成 A(t) 全序列图；只要在评估时把“哪些时间点算 ungated/gated”处理掉即可。

### 4.2 Soft gate（更贴近公式）

对变化边集合 `E_pred`，定义每个 t 的权重：

- `w_t = 1 - lambda_t`

soft→hard 的两种方式（都要实现，可选）：

- `mean_w`：若 `mean(w_t over subset) < w_thresh` 则删该边
- `frac_active`：若 `P(w_t > w_thresh) < p_thresh` 则删

默认：`mean_w`，阈值扫 `w_thresh ∈ {0.2,0.3,0.4}`

------

## 5) subset 划分（必须按 valid_mask）

在 valid 的时间点上计算分位数：

- `high = lambda >= quantile(high_q)`
- `low  = lambda <= quantile(low_q)`

默认：

- `high_q = 0.90`
- `low_q = 0.50`

------

## 6) 评价函数（必须复用一套）

写通用函数：

- `edges_from_adj(adj, diag_excluded=True) -> set[(i,j)]`
- `confusion(pred_edges, true_edges) -> TP,FP,FN,Prec,Rec,F1`
- `shd(pred_edges, true_edges)`（对 directed edges：SHD = FP+FN）

注意：

- 变化边任务用 `true_change_edges` 作为真值
- 也要支持“全图”任务（可选）：比较 gated/ungated 的 `adj_dyn` 与 `adj_regime1_true`（如果你想扩展）

------

## 7) 绘图（必须输出 Figure 5-1）

`gating_curve_demo.png`：

- 上图：λ_t（标 t_switch 若有）
- 下图：`active_edge_count(t)`
  - ungated：恒定 `K_pred`
  - gated hard：在 inactive 时 0，active 时 K_pred
  - gated soft：画 `K_pred * (1-λ_t)` 的连续曲线（可解释）

------

## 8) 脚本拆分（Codex 交付 3 个脚本 + 1 个 utils）

### 8.1 `step5_proxy_gating_effect.py`（主入口）

功能：

- 探测输入
- 生成 pred_change_edges
- 运行 ungated/hard/soft，输出 summary + 图 + retention + top edges

CLI 示例：

```
python step5_proxy_gating_effect.py --data_dir synthetic_step3_v2 --out_dir exports_step5
```

可选参数：

- `--high_q 0.90 --low_q 0.50`
- `--tau_list 0.7,0.8,0.9`
- `--soft_w_list 0.2,0.3,0.4`
- `--lambda_config "(50,3)"`（仅写进报告，不影响数据）

### 8.2 `step5_tau_sweep.py`

功能：

- 从多个 τ 结果中生成 `step5_tau_sweep.csv/md`
- 自动标记 `pick_best`（规则：最大化 high_SHD_gain，且 low_F1_delta >= -0.02）

### 8.3 `step5_make_tables_md.py`

功能：

- 把三个 csv 渲染为论文表格 md（Table 5-1/5-2/5-3）
- 顺带生成一个 `caption_templates.md`

### 8.4 `step5_utils.py`

通用函数、加载、自动探测、评估、绘图

------

## 9) 验收标准（跑完你就知道有没有成功）

执行：

```
python step5_proxy_gating_effect.py --data_dir synthetic_step3_v2
```

必须满足（至少在某个 τ 上）：

- `high_SHD_gain > 0`（越大越好）
- `low_F1_delta ≈ 0`（建议 ≥ -0.02）
   并且输出文件齐全（见第 1 节目录）。
Phase A 总目标（给 Codex 的一句话）

把合成实验整理成标准化评估流水线：固定配置、批量跑三种 λ 策略 + 负对照、输出统一 CSV/MD、生成核心可视化图，并自动汇总比较结果。

A0. 目录与产物规范（先做）
ToDo A0.1：统一输出目录结构

让 Codex 建议并实现以下结构（在 synthetic_step3_v2/exports_step5pp/ 下）：

runs/

score_equal/

score_gating/

score_regime/

lambda_shuffle/

lambda_constant_05/

lambda_constant_10/

compare/

compare_phaseA_configs.csv

compare_phaseA_configs.md

compare_phaseA_subsets.csv

compare_phaseA_checks.csv

figs/

fig_lambda_gate_overlay.png

fig_distance_curves.png

fig_retained_rel.png

fig_strategy_bar_align.png

fig_strategy_bar_margin.png

fig_strategy_bar_retained_gap.png

configs/

cfg_phaseA_base.json

cfg_phaseA_truechange_eval.json

cfg_lambda_equal.json

cfg_lambda_gating.json

cfg_lambda_regime.json

cfg_lambda_shuffle.json

cfg_lambda_const05.json

cfg_lambda_const10.json

ToDo A0.2：统一“单次运行输出文件名”

确保每个 run 子目录都输出同名文件，方便比较脚本读取：

config_used.json

sanity_metrics.json ← 新增（重点）

subset_summary.csv

subset_summary.md

curve_stats.csv ← 新增

checks.json ← 新增（gate_direction 等）

logs.txt

gated_graph_simulation.png

retained_curve.png

A1. 固化 Step5++ 评估协议（标准化）
ToDo A1.1：把 sanity 打印同步写入结构化 JSON

你现在 sanity 信息很多在 stdout 里，Codex 要把它们同时写入 sanity_metrics.json。

sanity_metrics.json 建议字段
{
  "config_name": "...",
  "delta_mask_mode": "...",
  "dist_mask_mode": "...",
  "delta_mask_nnz": 20,
  "dist_mask_nnz": 6,
  "A0_eff_nnz": 30,
  "A1_eff_nnz": 30,

  "subset_strategy": "non_sat_quantile",
  "high_thr": 0.595086,
  "low_thr": 0.257808,

  "high_non_sat_count": 536,
  "high_non_sat_mean_lambda": 0.740751,
  "high_non_sat_mean_gate_weight": 0.259249,
  "low_count": 2976,
  "low_mean_lambda": 0.105766,
  "low_mean_gate_weight": 0.894234,
  "all_count": 5951,
  "all_mean_lambda": 0.342089,
  "all_mean_gate_weight": 0.657911,

  "dist_std_base": 0.004718,
  "dist_std_reg0": 0.014162,
  "dist_std_reg1": 0.014162,

  "align_pre": 0.750493,
  "align_post": 0.751250,
  "align_overall": 0.750798,
  "margin_pre": 0.015325,
  "margin_post": 0.013502,
  "rel_pre_mean": 0.015325,
  "rel_pre_std": 0.032633,
  "rel_post_mean": 0.013502,
  "rel_post_std": 0.020293,

  "mean_dist_reg0_pre": 0.030760,
  "mean_dist_reg1_pre": 0.015436,
  "mean_dist_reg0_post": 0.029849,
  "mean_dist_reg1_post": 0.016347,

  "gate_direction": true,
  "high_closer_A0": true,
  "low_closer_A1": true,
  "pre_post_direction": false,
  "overall_check": false,

  "regime_swapped": false,
  "swap_reason": "auto_swap_regimes_disabled"
}
ToDo A1.2：输出 checks.json

单独再输出一份简洁检查文件，方便汇总脚本读取：

gate_direction

high_closer_A0

low_closer_A1

pre_post_direction

overall_check

regime_swapped

swap_reason

ToDo A1.3：输出 curve_stats.csv

把曲线统计单独导出（不要只在日志里）
建议字段：

metric, value

dist_std_base

dist_std_reg0

dist_std_reg1

rel_pre_mean

rel_pre_std

rel_post_mean

rel_post_std

retained_high_mean

retained_low_mean

retained_gap（新增：low - high）

ToDo A1.4：统一 subset 表（subset_summary.csv）

确保固定包含这几行：

high_non_sat

low

all

固定字段：

subset

count

mean_lambda

mean_gate_weight

p_active

mean_dist_base

mean_dist_reg0

mean_dist_reg1

mean_retained_ratio

A2. Step4 三种 λ 策略联动对比（核心）

目标：比较 score_equal / score_gating / score_regime 哪个更适合门控。

ToDo A2.1：为 Step5++ 增加 lambda_source_mode（或外部 lambda 文件输入）

Codex 需要让 step5pp_simulate_gated_graph.py 支持显式指定 λ 来源，例如：

方案（推荐）

增加命令行参数：

--lambda_file path/to/lambda.npy（优先级最高）

--lambda_tag score_gating（仅用于记录名字）

这样你可以复用同一套 Step5++ 逻辑，直接喂不同 λ 序列。

ToDo A2.2：写一个 λ 导出脚本（如果 Step4 还没导出）

新增脚本：

step4_export_lambda_variants.py

功能：

从 Step4 结果中导出：

lambda_equal.npy

lambda_gating.npy

lambda_regime.npy

同时导出：

lambda_metadata.csv（每个策略对应的 score、选中配置、阈值信息）

可选导出：

lambda_shuffle.npy

lambda_const_05.npy

lambda_const_10.npy

ToDo A2.3：写批量运行脚本（最重要）

新增脚本：

run_phaseA_batch.py

功能：

定义标准 Step5++ 配置（Phase A 统一协议）

delta_mask_mode=union_base_predchange

dist_mask_mode=true_change_only

subset_strategy=non_sat_quantile

auto_swap_regimes=False（建议先固定）

依次运行：

score_equal

score_gating

score_regime

lambda_shuffle

lambda_constant_05

lambda_constant_10

每次运行输出到对应 runs/<name>/

自动收集 sanity_metrics.json 和 subset_summary.csv

ToDo A2.4：自动汇总比较表（CSV + MD）

新增脚本：

compare_phaseA_runs.py

输出 3 张表：

表1：compare_phaseA_configs.csv/md（配置级）

每行一个 run，字段建议：

config_name

lambda_strategy

delta_mask_mode

dist_mask_mode

delta_mask_nnz

dist_mask_nnz

align_pre

align_post

align_overall

margin_pre

margin_post

dist_std_base

dist_std_reg0

dist_std_reg1

gate_direction

high_closer_A0

low_closer_A1

regime_swapped

swap_reason

表2：compare_phaseA_subsets.csv

每行一个（run, subset）组合，字段：

lambda_strategy

subset

count

mean_lambda

mean_gate_weight

mean_dist_base

mean_dist_reg0

mean_dist_reg1

mean_retained_ratio

表3：compare_phaseA_checks.csv

更简洁，适合汇报：

lambda_strategy

gate_direction

high_closer_A0

low_closer_A1

align_overall

margin_pre

margin_post

retained_gap

pass_core_checks（布尔：前三项都 True）

A3. 负对照实验（证明 λ 不是随便都行）
ToDo A3.1：实现 shuffle lambda 模式

在 step4_export_lambda_variants.py 或单独小脚本中：

读取一个基准 λ（建议 score_gating）

固定随机种子（如 2026）

打乱时间顺序，保存 lambda_shuffle.npy

要求：

保持长度、值分布、valid_mask 对齐不变

ToDo A3.2：实现常数 λ 模式

生成：

lambda_const_05.npy（全 0.5）

lambda_const_10.npy（全 1.0）
（可选再加 const_00）

ToDo A3.3：在汇总表里单独标记对照类型

在 compare_phaseA_configs.csv 增加字段：

run_type: main / negative_control

ToDo A3.4：增加一个“结论性指标” retained_gap

从 subset_summary.csv 提取：

retained_gap = mean_retained_ratio(low) - mean_retained_ratio(high_non_sat)

预期：

好的 λ：retained_gap 明显 > 0

shuffle/constant：retained_gap 变小或失效

A4. 可视化（批量生成，交给 Codex）

你刚问过“可视化要做什么”，这里直接列成 Codex 任务。

ToDo A4.1：单次运行图（每个 run 一套）

让 step5pp_simulate_gated_graph.py 输出（你已有部分）：

gated_graph_simulation.png

子图1：lambda(t) + g(t) + 高/低 subset 背景色

子图2：dist_base/dist_reg0/dist_reg1

子图3：retained_ratio(t) + rel(t)=dist_reg0-dist_reg1

retained_curve.png（保留）

dist_curve_only.png（新增，单独看距离曲线）

lambda_gate_only.png（新增）

ToDo A4.2：策略对比图（跨 run 汇总）

新增脚本：

plot_phaseA_comparison.py

生成以下图（统一放 figs/）：

图1：fig_strategy_bar_align.png

柱状图（x=策略）

y=align_overall

策略：equal / gating / regime / shuffle / const05 / const10

图2：fig_strategy_bar_margin.png

双柱图（每个策略两根）

margin_pre

margin_post

图3：fig_strategy_bar_retained_gap.png

柱状图：

retained_gap = retained_low - retained_high_non_sat

图4：fig_strategy_bar_diststd.png

三组柱：

dist_std_base

dist_std_reg0

dist_std_reg1

图5：fig_strategy_checks_heatmap.png（可选但很有用）

热图（0/1）：

行：策略

列：gate_direction, high_closer_A0, low_closer_A1

ToDo A4.3：论文/PPT友好版本导图

给 plot_phaseA_comparison.py 增加参数：

--paper_style（字体大一点、标题简洁）

导出 *_paper.png

A5. 自动生成“结论摘要”（非常实用）
ToDo A5.1：生成 phaseA_summary.md

新增脚本：

summarize_phaseA_results.py

从 compare_phaseA_configs.csv 自动写出一份简短总结，格式类似：

最佳门控策略（按 align_overall）: score_gating

核心检查是否通过:

gate_direction: ✅

high_closer_A0: ✅

low_closer_A1: ✅

负对照结果:

shuffle λ 的 align_overall 从 X 降到 Y

const λ 的 retained_gap 接近 0（或显著下降）

说明:

λ 携带结构相关时序信息，不是任意序列都能产生同等门控效果

ToDo A5.2：生成 phaseA_summary.json

方便后面 PPT/论文脚本直接读取：

best_strategy_by_align

best_strategy_by_retained_gap

main_runs_pass_rate

negative_control_drop

A6. 代码质量与复现性（Codex必须做）
ToDo A6.1：固定随机种子

涉及 shuffle / 任何随机过程：

numpy.random.seed(2026)

如果有 random 也设

ToDo A6.2：参数与版本记录

每个 run 保存：

config_used.json

lambda_source_info.json（策略名、文件路径、生成方式）

可选：git_commit.txt（如果仓库里能取到）

ToDo A6.3：失败保护

批量脚本里：

单个 run 失败不影响后续

在 compare/failed_runs.log 记录错误栈

A7. 给 Codex 的执行顺序（建议）

这个顺序能避免反复返工：

A1（先结构化输出：JSON/CSV）

A2.1 / A2.2（支持外部 λ + 导出 λ）

A2.3 / A2.4（批量跑 + 汇总表）

A3（负对照）

A4（可视化）

A5（自动总结）

A6（复现性清理）

给 Codex 的验收标准（Checklist）

Codex 完成后，你只看这几项：

 runs/ 下至少有 6 个 run（3主策略 + 3对照）

 每个 run 都有 sanity_metrics.json / subset_summary.csv / checks.json

 compare_phaseA_configs.csv 成功生成

 compare_phaseA_subsets.csv 成功生成

 至少 3 张对比图成功生成（align / margin / retained_gap）

 phaseA_summary.md 自动生成且内容可读

 shuffle/constant 对照结果明显弱于主策略（至少某些指标下降）
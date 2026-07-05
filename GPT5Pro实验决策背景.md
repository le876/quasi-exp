# GPT-5 Pro 实验决策背景：张力一致性优化

生成日期：2026-06-08  
项目路径：`/mnt/ML_projects/quasi_exp`  
配套材料：请同时发送 `原论文.md` 给 GPT-5 Pro。

## 0. 给 GPT-5 Pro 的任务说明

本文档是一个中立的实验决策包，目标不是让你复述已有实验，而是基于已有事实提出一个完整、可执行、可验收的下一阶段实验方案。

当前最需要优化的主目标是 **张力一致性**：

- 同一或相近构型下，张力标签应尽量唯一、连续、可学习。
- 模型张力预测 MAE 应下降，但不要只用模型误差掩盖标签本身的不连续。
- EE 误差、theta 误差、准静态残差、张力上限、饱和率都是约束或副指标。

希望你重点判断：

- 当前张力不一致主要来自物理建模/张力求解多解，还是来自工作空间到构型的多分支映射？
- 下一阶段应该优先改 canonical tension allocator、采样策略、模型结构，还是评价指标定义？
- 如何设计分阶段实验，使 20k 级别试验能可靠预测是否值得扩展到 100k。

## 1. 项目与学习任务

本项目基于 `原论文.md` 和 `整体项目原文.md`，复现一个连续体/绳驱机器人数据集。当前核心监督学习任务是从末端工作空间坐标预测机器人内部状态与张力：

```text
input:  x_m, y_m, z_m
output: theta_1..30_rad + tension_1..12_n
```

训练脚本位于 `scripts/baselines/run_baselines.py`。当前 baseline 使用 `dataset.parquet` 中的：

- 输入列：`x_m`, `y_m`, `z_m`
- 输出列：30 个 `theta_*_rad`，12 个 `tension_*_n`
- 特征工程：常用 `poly_heavy`
- 模型输出后，用 `FK(pred_theta)` 回代得到末端点，并与输入 `x,y,z` 比较，得到 `ee_pos_*_mm`

评价指标主要包括：

- `theta_mae_deg`, `theta_rmse_deg`
- `tension_mae_n`, `tension_rmse_n`
- `ee_pos_p95_mm`, `ee_pos_rmse_mm`, `ee_pos_max_mm`
- `tension_lt0_ratio`, `tension_gt_tmax_ratio`
- 数据质量侧的 `rms_rnorm`, `max_tension`, `tension saturation ratio`
- 局部连续性侧的近邻 `theta_rms_deg_p95`, `tension_mae_n_p50/p90/p95`

当前常用 split：

- `iid`：随机划分。
- `radius`：按 `r = ||p||` 排序，外层工作空间作为 test。
- `beta_block`：按 6 维 reduced beta 幅值外推。
- `angular_sector`：按 yz 平面方向扇区外推。

## 2. 已确认的力学建模口径

机器人共有 30 个圆盘，按 10 个圆盘分为三段：

- 第一关节：`Disk 1..10`
- 第二关节：`Disk 11..20`
- 第三关节：`Disk 21..30`

12 根绳的终止盘号采用论文 Eq.(6) 口径：

- `{1,2,11,12}` 终止于第 10 盘
- `{3,4,9,10}` 终止于第 20 盘
- `{5,6,7,8}` 终止于第 30 盘

重要物理理解：

- `{1,2,11,12}` 主要决定第一关节角，但第一关节仍受到其余 8 根远端绳穿越摩擦影响。
- `{3,4,9,10}` 主要决定第二关节角，但第二关节仍受到 `{5,6,7,8}` 穿越摩擦影响。
- `{5,6,7,8}` 主要决定第三关节两个角。

当前实现中，某盘/段是否受到某根绳影响按“该绳是否穿过该段”判断：

```text
active[i, j] = true if end_disk_by_j[j] >= i
```

这等价于：

- `Disk 1..10`：12 根绳均参与。
- `Disk 11..20`：8 根绳参与，即 `{3,4,5,6,7,8,9,10}`。
- `Disk 21..30`：4 根绳参与，即 `{5,6,7,8}`。

该逻辑目前被认为符合我们已确认的物理口径。参考文档：`力学建模明确事项.md`。

仍未完全解决或需要 GPT-5 Pro 判断的点：

- 张力传递方向：当前实现主要复用从基座向远端的正向传递；真正分段求解可能需要更严格的反向/边界传递定义。
- 摩擦 Case 1/2 的路径依赖：当前多按全局绳长变化选择，分段情况下是否应按每段历史选择仍不明确。
- 论文 Eq.(52)+(53) 是否本身缺少唯一化张力的 tie-break 条件。
- 给定同一个末端 xyz 时，是否存在多个 theta/configuration branch；如果存在，workspace 近邻连续性可能不是正确的唯一评价方式。

## 3. 原始 PSO/论文式约束的关键问题

早期采用论文式 PSO 或接近论文 Eq.(52)+(53) 的约束目标来求解 12 路基座张力。核心现象是：很多 theta 下可以找到满足残差阈值的张力，但不同 seed 会落到相差很大的可行张力解。

关键诊断：

- 文件：`runs/diagnostics/paper_strict_seed_stability_standard_100k/summary.json`
- 配置：`configs/robot_rods_only_standard_2k_paper_strict.yaml`
- 数据：`data/standard_beta_sweep_100k/dataset.parquet`
- theta 数：10
- seeds：8 个
- feasible rate：`1.0`
- objective：`paper_constraint`
- `median_pairwise_mae_n = 731.85 N`
- `p95_pairwise_mae_n = 1108.30 N`
- gate：fail

解释：这不是“PSO 找不到可行解”，而是“可行解太多且没有唯一化规则”。这会导致 MLP 看到同一个或近似 theta/xyz 对应冲突张力标签，张力预测自然很弱。

旧数据/旧模型也体现了这个问题。`runs/acceptance/seed_compare_summary.json` 中早期 MLP/TF MLP 张力 MAE 大约在 `435-443 N` 量级。

## 4. Segmented canonical 张力求解

为解决原始 PSO 多解与随机性问题，我们实现了分段 deterministic/canonical 张力标注：

求解顺序固定为远端到近端：

1. 第三段：先求 `{5,6,7,8}`，残差切片为 `Disk 21..30`。
2. 第二段：再求 `{3,4,9,10}`，残差切片为 `Disk 11..20`，并考虑第三段绳穿越影响。
3. 第一段：最后求 `{1,2,11,12}`，残差切片为 `Disk 1..10`，得到最终 12 路基座张力。

canonical tie-break 早期采用 `t_ref_n = 800.0` 等确定性项，让同一个 theta 得到稳定张力。

核心入口：

- `src/quasi_exp/opt/segmented_tension.py`
- `src/quasi_exp/opt/tension_labeler.py`
- `scripts/generate_dataset.py`
- `scripts/worker_generate_sample.py`

### 4.1 求解速度与一致性诊断

文件：`runs/diagnostics/segmented_tension_standard_10k_20x3/summary.json`

20 个 theta、每个重复 2 次，并与 3 seed PSO 对比：

| 指标 | segmented | PSO |
|---|---:|---:|
| feasible rate | 1.0 | 1.0 |
| rms_rnorm mean | 0.02840 | 0.05092 |
| rms_rnorm p95 | 0.03215 | 0.05942 |
| elapsed mean | 0.1629 s | 1.2557 s |
| pairwise MAE median | 0 N | 501.27 N |
| pairwise MAE p95 | 0 N | 935.04 N |
| speedup | 约 7.71x | baseline |

100 theta 跳过 PSO 的诊断中：

- 文件：`runs/diagnostics/segmented_tension_standard_10k_100_skip_pso/summary.json`
- segmented mean time：`0.1574 s/sample`
- `repeat_max_abs_delta_n = 0.0`
- `pairwise_mae_n = 0.0`
- 估算单进程 100k：约 `15743 s`

实际正式生成使用多进程，因此 100k 生成时间显著低于单进程估计。

## 5. 标准角度网格 100k：证明标签可以很平滑

标准网格数据集：

- 配置：`configs/robot_rods_only_standard_100k_segmented_canonical.yaml`
- 数据：`data/standard_beta_sweep_100k_segmented_canonical/dataset.parquet`
- 记录：`segmented_canonical_100k执行记录.md`

关键生成结果：

- rows：`100000`
- tried = accepted = `100000`
- elapsed：`2446.62 s`，约 `40 分 47 秒`
- speed：`0.02447 s/sample`，约 `40.87 samples/s`

数据质量：

- `tension_solver_method = segmented_canonical`：`100000/100000`
- `segmented_success_rate = 1.0`
- `rms_rnorm p95 = 0.035676`
- `rms_rnorm max = 0.039803`
- tension max p95：`929.98 N`
- tension max max：`1010.52 N`
- saturation ratio：`0`

局部连续性：

- 文件：`runs/diagnostics/standard_beta_sweep_100k_segmented_canonical_local_continuity.json`
- passed：true
- 10mm pairs：`7,900,000`
- `theta_rms_deg_p95 = 0.25335 deg`
- `tension_mae_n_p50 = 0.1652 N`
- `tension_mae_n_p95 = 5.8468 N`
- `tension_max_abs_n_p95 = 12.9831 N`

MLP 结果：

| 模型 | theta MAE | T MAE | EE p95 | 备注 |
|---|---:|---:|---:|---|
| Classic MLP | 0.03205 deg | 4.1979 N | 4.9772 mm | `runs/baselines_standard_beta_sweep_100k_segmented_canonical_classic_iid/all_metrics.json` |
| TF MLP | 0.03768 deg | 3.3610 N | 5.8180 mm | `runs/baselines_standard_beta_sweep_100k_segmented_canonical_tf_iid/all_metrics.json` |

结论：在标准角度网格/简单分布上，segmented canonical 标签非常平滑，MLP 也能把张力 MAE 学到 `3-4 N`。这说明“模型完全学不了张力”不是根本事实；根本问题更可能在复杂采样下的标签连续性/多分支性。

## 6. 标准 100k 多模型与 OOD split：IID 指标不能过度解读

IID 多模型比较：

- 文件：`runs/baselines_standard_beta_sweep_100k_segmented_canonical_model_compare_iid/COMPARISON.md`
- split：IID
- feature set：`poly_heavy`

IID 结果按张力 MAE 排名：

| rank | model | T MAE | EE p95 | theta MAE |
|---:|---|---:|---:|---:|
| 1 | knn | 0.093 N | 0.058 mm | 0.00042 deg |
| 2 | lgbm | 0.201 N | 0.535 mm | 0.00097 deg |
| 3 | rf | 0.425 N | 1.766 mm | 0.00794 deg |
| 4 | mlp_large | 3.072 N | 5.500 mm | 0.03992 deg |
| 5 | tf_mlp | 3.361 N | 5.818 mm | 0.03768 deg |
| 6 | mlp | 4.198 N | 4.977 mm | 0.03205 deg |

但是 radius split 明显更难：

- 文件：`runs/baselines_standard_beta_sweep_100k_segmented_canonical_model_compare_radius/RADIUS_VS_IID.md`
- split rule：按 `r = ||p||` 排序，训练内层 80%，验证下一层 10%，测试最外层 10%。

radius split 排名：

| model | radius T MAE | radius EE p95 | radius theta MAE |
|---|---:|---:|---:|
| lgbm | 21.701 N | 40.814 mm | 0.3691 deg |
| knn | 22.364 N | 39.883 mm | 0.3615 deg |
| mlp_large | 24.900 N | 24.356 mm | 0.2515 deg |
| rf | 25.639 N | 29.124 mm | 0.3089 deg |
| mlp | 29.108 N | 14.984 mm | 0.2380 deg |

结论：

- 标准网格 IID 下 KNN/LGBM/RF 近乎完美，主要说明 dense-grid interpolation 很强。
- radius holdout 下全部模型退化，说明当前标准 100k 不是充分的 OOD 泛化证据。
- 张力 MAE 和 EE p95 不总是同向排序，例如 radius split 下 plain MLP 张力较差但 EE p95 最好。

## 7. 数据复杂化：mixed beta + distal preferred

为了提升数据复杂度，我们从标准网格转向 mixed beta 采样，并加入“优先移动第三关节”的原则：

- `theta1..4` 限制在 `5 deg` 以内。
- `theta5..6` 限制在 `10 deg` 以内。
- 如果多个构型能到达相近位置，倾向选择第三关节角度更大、前两个关节角度更小的样本。

mixed 采样组件：

- `sobol_full`
- `lhs_full`
- `workspace_balanced`
- `distal_biased`

用户曾要求对这些组件分别做 continuity，用于判断不同采样来源是否具有不同连续性问题。

### 7.1 Mixed 20k distal-preferred

文件：`runs/diagnostics/mixed_beta_20k_distal_preferred_report.md`

数据：

- 配置：`configs/robot_rods_only_mixed_20k_distal_preferred_segmented_canonical.yaml`
- 数据：`data/mixed_beta_20k_distal_preferred_segmented_canonical/dataset.parquet`
- rows：`20000`
- elapsed：`483.24 s`
- speed：`0.02416 s/sample`
- source counts：`sobol_full=8000`, `lhs_full=4000`, `workspace_balanced=4000`, `distal_biased=4000`

硬质量门槛：

- beta1..4 <= 5 deg：PASS
- beta5..6 <= 10 deg：PASS
- `rms_rnorm p95 = 0.03857`：PASS
- max tension：`1478.78 N`：PASS
- saturation ratio：`0.0`：PASS

局部连续性诊断：

- passed：false
- nearest-neighbor dxyz p50：`9.544 mm`
- nearest-neighbor dxyz p90：`18.187 mm`
- nearest-neighbor dxyz p95：`22.695 mm`
- 10mm `theta_rms_deg_p95 = 8.354 deg`
- 10mm `tension_mae_n_p95 = 250.08 N`
- 20mm `theta_rms_deg_p95 = 8.409 deg`
- 20mm `tension_mae_n_p95 = 250.81 N`

fast4 baseline 最佳结果：

| split | best model | T MAE | EE p95 | theta MAE |
|---|---|---:|---:|---:|
| iid | mlp_large | 99.61 N | 47.71 mm | 2.403 deg |
| radius | mlp | 104.67 N | 43.18 mm | 1.952 deg |
| beta_block | mlp | 114.75 N | 69.08 mm | 3.474 deg |
| angular_sector | mlp | 93.52 N | 79.17 mm | 2.523 deg |

当时结论：硬质量门槛通过，但模型拟合门槛失败。该数据集是复杂度压力测试，不适合直接扩展前就断言可学。

### 7.2 Mixed 100k distal-preferred

文件：

- `runs/diagnostics/mixed_beta_100k_distal_preferred_report.md`
- `runs/diagnostics/mixed_beta_100k_distal_preferred_baseline_report.md`

数据：

- 配置：`configs/robot_rods_only_mixed_100k_distal_preferred_segmented_canonical.yaml`
- 数据：`data/mixed_beta_100k_distal_preferred_segmented_canonical/dataset.parquet`
- rows：`100000`
- report elapsed：`2426 s`
- dataset_report elapsed：`2373.23 s`
- speed：约 `0.024 s/sample`
- source counts：`sobol_full=40000`, `lhs_full=20000`, `workspace_balanced=20000`, `distal_biased=20000`

硬质量门槛：

- beta1..4 <= 5 deg：PASS
- beta5..6 <= 10 deg：PASS
- `rms_rnorm p95 = 0.0385919342`：PASS
- max tension：`1478.784 N`：PASS
- saturation ratio：`0.0`：PASS
- distal preference median：`group3=0.78565 > max(group1, group2)=0.30113`：PASS

数据分布：

- `theta_max_abs_deg q95 = 9.79929 deg`
- `theta_max_abs_deg max = 9.99994 deg`
- tension max q95：`1133.75 N`
- tension max q99：`1238.32 N`
- workspace x：`0.5512 to 1.2139 m`
- workspace y：`-0.7436 to 0.7392 m`
- workspace z：`-0.7000 to 0.6957 m`

局部连续性诊断：

- passed：false
- nearest-neighbor dxyz p50：`5.570 mm`
- nearest-neighbor dxyz p90：`10.694 mm`
- nearest-neighbor dxyz p95：`13.371 mm`
- 10mm `theta_rms_deg_p95 = 8.386 deg`
- 10mm `tension_mae_n_p95 = 249.69 N`
- 20mm `theta_rms_deg_p95 = 8.358 deg`
- 20mm `tension_mae_n_p95 = 248.59 N`

100k fast4 baseline 最佳结果：

| split | best model | T MAE | T RMSE | EE p95 | theta MAE |
|---|---|---:|---:|---:|---:|
| iid | mlp_large | 98.80 N | 129.83 N | 48.74 mm | 2.408 deg |
| radius | mlp_large | 100.52 N | 130.12 N | 49.66 mm | 1.916 deg |
| beta_block | mlp | 114.99 N | 156.80 N | 69.39 mm | 3.428 deg |
| angular_sector | mlp_large | 91.31 N | 117.99 N | 84.96 mm | 2.475 deg |

100k 相对 20k 的 best T MAE 改善：

| split | 20k best | 100k best | delta |
|---|---:|---:|---:|
| iid | 99.61 N | 98.80 N | -0.8% |
| radius | 104.67 N | 100.52 N | -4.0% |
| beta_block | 114.75 N | 114.99 N | +0.2% |
| angular_sector | 93.52 N | 91.31 N | -2.4% |

结论：简单扩大数据量不是主要瓶颈。mixed 数据满足硬质量门槛，但从 workspace 角度看局部不连续，当前直接回归器难以学到低张力 MAE。

## 8. Cross-sample canonical：anchor relabel v1/v2

在 mixed 数据上，segmented canonical 虽然对同一个 theta 是确定的，但跨样本仍可能不够平滑。于是尝试了 same-component beta-kNN anchor relabel：

- 在同一个采样组件内找 beta 空间 kNN。
- 用邻域张力作为 anchor/reference。
- 在 segmented tension solve 中加入 `w_anchor`，使张力场更接近邻域 canonical 分配。
- 保持硬门槛：`rms_rnorm q95 <= 0.06`、张力不超过上限、饱和率为 0、`segmented_success_all`。

关键脚本：

- `scripts/analysis/relabel_anchor_canonical.py`
- `scripts/analysis/run_anchor_canonical_sweep.py`
- `src/quasi_exp/opt/segmented_tension.py`

### 8.1 Anchor v1

v1 设置：

- same-component beta-kNN
- `k = 16`
- `w_anchor = 10.0`
- 数据：`data/mixed_beta_20k_distal_preferred_anchor_v1`

在 v2 sweep 报告中，v1 作为 reference：

- hard passed：true
- 10mm T p50：`94.17 N`
- 10mm T p90：`169.11 N`
- 10mm T p95：`190.60 N`
- 10mm theta p95：`8.34 deg`
- `rms_rnorm q95 = 0.03982`
- max tension：`1300.66 N`

v1 比原 mixed segmented canonical 有改善，但局部张力仍然偏大。

### 8.2 Anchor v2 sweep

文件：

- `runs/diagnostics/anchor_sweep_v2/REPORT.md`
- `runs/diagnostics/anchor_sweep_v2/summary.json`

测试变体：

- `anchor_v2_k16_w20`
- `anchor_v2_k16_w30`
- `anchor_v2_k32_w10`
- `anchor_v2_k32_w20`
- `anchor_v2_k32_w30`
- 从最佳 first-stage 做 second-stage graph smoothing/refinement

重要观察：

- 仅提高 `w_anchor` 到 20/30 但仍用 `k=16`，改善很小，且出现 `segmented_success_all=false`。
- 增大 same-component kNN 到 `k=32` 是主要改善来源。
- `k32/w20` 是最佳 first-stage hard-passing 候选。
- 基于 `k32/w20` 再做 second-stage refinement 是当前 20k 上最好的结果。

v2 质量表：

| variant | hard | 10mm T p50 | 10mm T p90 | 10mm T p95 | theta p95 | rms q95 | max T |
|---|---|---:|---:|---:|---:|---:|---:|
| anchor_v1 | true | 94.17 | 169.11 | 190.60 | 8.34 | 0.03982 | 1300.66 |
| k16/w20 | false | 93.86 | 168.96 | 190.38 | 8.34 | 0.04398 | 1300.39 |
| k16/w30 | false | 93.62 | 168.94 | 190.34 | 8.34 | 0.04646 | 1300.29 |
| k32/w10 | true | 88.27 | 158.05 | 179.11 | 8.34 | 0.04000 | 1251.83 |
| k32/w20 | true | 87.63 | 157.57 | 178.73 | 8.34 | 0.04433 | 1253.23 |
| k32/w30 | false | 87.40 | 157.60 | 178.61 | 8.34 | 0.04695 | 1253.80 |
| second-stage from k32/w20 | true | 67.38 | 118.66 | 134.73 | 8.34 | 0.04479 | 1138.58 |

当前最佳数据：

- `data/mixed_beta_20k_distal_preferred_anchor_v2_second_stage_from_anchor_v2_k32_w20/dataset.parquet`
- `data/mixed_beta_20k_distal_preferred_anchor_v2_second_stage_from_anchor_v2_k32_w20/dataset_meta.parquet`
- `data/mixed_beta_20k_distal_preferred_anchor_v2_second_stage_from_anchor_v2_k32_w20/dataset_report.json`

二阶段 relabel 报告：

- rows：`20000`
- `accepted_infeasible = 0`
- elapsed：`322.84 s`
- speed：`0.01614 s/sample`
- `anchor_k = 32`
- `w_anchor = 20.0`
- `component_scope = same_component`
- `distance_space = beta`

### 8.3 Anchor v2 模型结果

`anchor_v2_second_stage_from_anchor_v2_k32_w20` 的最佳模型指标：

| split | model | T MAE | T RMSE | EE p95 | theta MAE |
|---|---|---:|---:|---:|---:|
| iid | mlp | 49.58 N | 66.16 N | 49.75 mm | 2.458 deg |
| radius | mlp_large | 43.84 N | 58.65 N | 55.64 mm | 1.948 deg |
| beta_block | mlp | 59.23 N | 80.83 N | 77.37 mm | 3.445 deg |
| angular_sector | mlp_large | 51.09 N | 65.47 N | 89.38 mm | 2.567 deg |

相对 anchor v1 的 T MAE 改善：

- iid：`-29.9%`，70.74N -> 49.58N
- radius：`-32.1%`，64.59N -> 43.84N
- beta_block：`-29.4%`，83.93N -> 59.23N
- angular_sector：`-28.9%`，71.86N -> 51.09N

当前数值判断：

- 如果说“相邻位置一致性差在 70N”，更准确应表述为：最佳 v2 second-stage 的 10mm 近邻张力 MAE p50 是 `67.38N`，p90 是 `118.66N`，p95 是 `134.73N`。
- 如果说“模型拟合 MAE 平均在 50N”，更准确应表述为：四个 split 最佳模型 T MAE 平均约 `50.94N`。

局限：

- 10mm `theta_rms_deg_p95` 仍为 `8.34 deg`，anchor 主要降低了张力差，没有解决 workspace 近邻构型差异。
- EE p95 并非所有 split 都改善；radius/beta_block 在 second-stage 下 EE p95 有时更差。
- 这仍只是 20k 结果，100k anchor v2 second-stage 尚未正式扩展。

## 9. 当前关键实验结论

### 9.1 已比较明确的结论

1. 论文式 PSO/约束目标本身不足以唯一化张力。
   - 同 theta 多 seed 都可行，但张力 pairwise MAE 可到数百到上千牛。

2. Segmented deterministic/canonical 有效解决了同 theta 随机性。
   - 重复求解张力差为 `0N`。
   - 比 PSO 快约 `7.7x`。
   - 标准网格 100k 上张力 continuity p95 低至 `5.85N`。

3. 标准网格数据证明模型可以学张力。
   - Classic/TF MLP 张力 MAE 为 `4.20N/3.36N`。
   - KNN/LGBM/RF 在 IID dense grid 上更低，但这主要是插值能力。

4. mixed/distal-preferred 数据显著更难。
   - 20k/100k 都通过硬质量门槛。
   - 但 10mm workspace 近邻 tension p95 仍约 `250N`。
   - 100k 相比 20k 的直接回归提升只有 `0-4%`，说明不是简单样本数问题。

5. Anchor canonical v2 有明显收益但还没彻底解决。
   - 10mm T p95 从 v1 `190.60N` 降到 second-stage `134.73N`。
   - 模型 T MAE 平均降到约 `50.94N`。
   - 但 workspace 近邻 theta p95 仍 `8.34deg`，说明可能还有多构型/多分支问题。

### 9.2 需要谨慎解释的点

- `workspace xyz` 近邻不一定是同一构型分支近邻。mixed 数据可能在同一小空间区域内包含不同 beta/theta 分支。
- 如果评价 local continuity 时跨分支配对，张力差大不一定全是求解器差；也可能是物理上不同构型对应不同张力。
- 但是模型输入目前只有 xyz。如果同一个 xyz 附近确实存在多种 theta/tension，那么直接单值回归本身就会困难。
- anchor relabel 使用 beta 空间 same-component kNN，本质是在试图构造一个更平滑的单值张力场，但它不一定能解决 xyz -> state 的多分支性。

## 10. 当前主要瓶颈假设

请 GPT-5 Pro 针对以下假设做判断，并给出可验证实验：

### 假设 A：张力 allocator 仍不够 canonical

即使在同一 beta/component 近邻内，张力仍存在冗余分配。需要更强的 canonical allocator，例如：

- 更系统的 graph smoothing / Laplacian regularization。
- 全局或局部 convex/least-squares canonical 分配。
- 最小张力范数、最小最大张力、最小张力变化、接近 nominal 张力等明确 tie-break。
- 分段张力 allocator 中加入跨样本连续约束，而不是事后 relabel。

### 假设 B：workspace 到 theta/tension 是多分支映射

输入只有 `x,y,z`，但同一或相近 workspace 可能对应多组 beta/theta。此时单值回归会把多个 branch 平均掉，导致：

- theta MAE 增大。
- tension MAE 增大。
- EE 回代可能不稳定。

可能方向：

- 重新定义 continuity，只在 same branch / same component / beta-kNN 内评估。
- 输入增加 branch/source/component/initial beta 等条件。
- 训练 mixture-of-experts 或 classifier + expert。
- 改任务定义：先预测 branch/canonical beta，再预测 theta/tension。

### 假设 C：采样策略造成局部混叠

mixed_beta 的 `sobol_full/lhs_full/workspace_balanced/distal_biased` 混合后，可能在 workspace 中制造了近邻混叠。需要分别诊断：

- 每个 source component 单独 local continuity。
- 跨 component local continuity。
- same-component vs cross-component 近邻张力差。
- beta 空间近邻 vs xyz 空间近邻的一致性差异。

### 假设 D：模型结构不适合当前输出

当前直接预测 30 theta + 12 tension。可能需要结构化模型：

- theta head 与 tension head 分离。
- 分段 tension heads：第三段、第二段、第一段。
- group-specific heads：`{5,6,7,8}`、`{3,4,9,10}`、`{1,2,11,12}`。
- source/region-conditioned experts。
- 物理一致性 loss：FK loss、tension bounds、residual surrogate。

但需要注意：如果标签本身跨 xyz 不单值，模型结构只能缓解，不能根治。

## 11. 希望 GPT-5 Pro 输出的实验方案要求

请输出一个分阶段方案，至少包括：

1. 诊断实验
   - 如何区分“张力 allocator 不唯一”和“workspace 多分支映射”。
   - 如何设计 same-component、same-beta-neighbor、same-xyz-neighbor 的 continuity 对比。
   - 如何判断 local continuity 应该在哪个空间中定义。

2. 张力 canonical 优化实验
   - 是否继续 anchor v2 second-stage。
   - 是否引入更强 graph smoothing / global canonical allocator。
   - 是否把 canonical 目标集成进 solver，而不是 relabel 后处理。
   - 每种方案的预计成本、风险和验收指标。

3. 数据采样实验
   - 是否保留 `sobol_full/lhs_full/workspace_balanced/distal_biased`。
   - 是否需要分 source 训练/评估。
   - 是否需要主动采样低混叠区域或 branch-consistent 区域。
   - 是否先做 20k ablation，再扩展 100k。

4. 模型实验
   - 当前 MLP/MLP_large/KNN/RF/LGBM baseline 已有，下一步该测哪些结构化模型。
   - 是否需要 mixture-of-experts、分段张力头、branch classifier、multi-task loss。
   - 如何避免模型实验掩盖标签问题。

5. 验收标准
   - 数据质量硬门槛：`rms_rnorm q95 <= 0.06`、max tension <= 2000N、saturation ratio = 0、`segmented_success_all = true`。
   - 张力连续性门槛：请给出合理的 p50/p90/p95 目标，区分 same-branch 与 workspace-neighbor。
   - 模型门槛：请给出 T MAE、EE p95、theta MAE 的分 split 目标。
   - 计算成本门槛：请说明 20k/100k 可接受耗时。

## 12. 可复现实验入口与关键文件

### 文档

- `原论文.md`
- `整体项目原文.md`
- `力学建模明确事项.md`
- `segmented_canonical_100k执行记录.md`
- `后续优化方案.md`

### 核心代码

- `src/quasi_exp/opt/segmented_tension.py`
- `src/quasi_exp/opt/tension_canonical.py`
- `src/quasi_exp/opt/pso_inverse.py`
- `src/quasi_exp/model/sampling.py`
- `scripts/generate_dataset.py`
- `scripts/worker_generate_sample.py`
- `scripts/baselines/run_baselines.py`
- `scripts/baselines/splits.py`
- `scripts/analysis/relabel_anchor_canonical.py`
- `scripts/analysis/run_anchor_canonical_sweep.py`

### 配置

- `configs/robot_rods_only_standard_100k_segmented_canonical.yaml`
- `configs/robot_rods_only_mixed_20k_distal_preferred_segmented_canonical.yaml`
- `configs/robot_rods_only_mixed_100k_distal_preferred_segmented_canonical.yaml`
- `configs/robot_rods_only_mixed_20k_distal_preferred_anchor_v1.yaml`

### 数据与报告

- `data/standard_beta_sweep_100k_segmented_canonical/`
- `data/mixed_beta_20k_distal_preferred_segmented_canonical/`
- `data/mixed_beta_100k_distal_preferred_segmented_canonical/`
- `data/mixed_beta_20k_distal_preferred_anchor_v2_second_stage_from_anchor_v2_k32_w20/`
- `runs/diagnostics/paper_strict_seed_stability_standard_100k/summary.json`
- `runs/diagnostics/segmented_tension_standard_10k_20x3/summary.json`
- `runs/diagnostics/standard_beta_sweep_100k_segmented_canonical_local_continuity.json`
- `runs/diagnostics/mixed_beta_20k_distal_preferred_report.md`
- `runs/diagnostics/mixed_beta_100k_distal_preferred_report.md`
- `runs/diagnostics/mixed_beta_100k_distal_preferred_baseline_report.md`
- `runs/diagnostics/anchor_sweep_v2/REPORT.md`
- `runs/diagnostics/anchor_sweep_v2/summary.json`
- `runs/baselines_standard_beta_sweep_100k_segmented_canonical_model_compare_iid/COMPARISON.md`
- `runs/baselines_standard_beta_sweep_100k_segmented_canonical_model_compare_radius/RADIUS_VS_IID.md`

注意：`data/` 和 `runs/` 下大部分 artifacts 可能是 gitignored，但本机当前项目目录中存在。

## 13. 一句话当前状态

我们已经证明：张力标签如果被 segmented canonical 唯一化，并且采样分布简单，MLP 可以把张力 MAE 学到 `3-4N`；但在 mixed/distal-preferred 复杂采样下，即使用 anchor v2 二阶段 canonical，局部 10mm 张力一致性仍约为 p50 `67N`、p95 `135N`，模型四 split 平均张力 MAE 约 `51N`。下一阶段的核心不是盲目扩数据，而是判断并解决“张力 canonical 不足”和“workspace 多分支映射”这两个可能瓶颈。

# Phase 5 固定层单支流形实验结果

本文记录根据 `docs/GPT5pro 三次回答.md` 后续执行的实验。核心问题是：扩展 workspace 泛化性后，张力一致性从分段 canonical 小数据集的约 50N 退化到 100N 左右，怀疑原因不是单纯张力 allocator，而是 workspace 中混入多个 branch / 多个近似逆解，导致 `xyz -> theta/T` 不是稳定单值映射。

## 实验目的

本阶段不再继续在 mixed pool 上事后过滤，而是测试“固定层单支流形”是否能让数据从生成阶段就变成局部单值：

- 固定层定义为 `(s1, s2)`，将第三关节主角 `beta5,beta6` 作为主采样对象；
- 使用 `beta=[s1*beta5, s1*beta6, s2*beta5, s2*beta6, beta5, beta6]`；
- 这样仍然遵守“优先第三关节，其次第二关节，其次第一关节”的原则；
- 每个固定层单独生成 2k pilot，避免把多个连续子流形硬压成一个全局单值模型。

## 新增脚本

- `scripts/generate_fixed_layer_manifold_dataset.py`
  - 生成固定 `(s1,s2)` 的 beta manifold 数据集；
  - 默认每个 2k 数据包含 1681 个规则网格点、160 个 Sobol 点、159 个 LHS 点；
  - 输出 `dataset.parquet`、`dataset_meta.parquet`、`fixed_layer_generation_report.json`。

- `scripts/analysis/audit_tension_discontinuity_sources.py`
  - 对 10mm workspace 邻域且 beta 接近的样本对做张力不连续审计；
  - 区分 `no_flip`、`case_flip_only`、`active_set_flip_only`、`both_flip`；
  - 统计张力 p95、case flip 比例、active-set flip 比例和主要跳变段。

- `scripts/analysis/relabel_tension_graph_canonical.py`
  - 在同一固定层内部构造 kNN anchor；
  - 用邻域张力的 median 或 huber mean 作为 anchor；
  - 重新求解张力，目标是进一步降低局部张力不连续。

- `scripts/pipelines/run_phase5_fixed_layer_experiments.py`
  - 串联 fixed-layer 2k sweep、audit、relabel ablation；
  - 产出统一 summary。

## 验收标准

固定层 2k pilot 初筛通过条件：

- hard gate 通过；
- `all10_theta_p95_deg <= 1.5 deg`；
- `all10_tension_p95_n <= 110 N`；
- `beta_close_tension_p95_n <= 90 N`；
- `multi_branch_ball_ratio <= 0.15`；
- `xyz_nn_tension_mae_n <= 35 N`；
- `beta_nn_tension_mae_n <= 20 N`。

relabel silver 条件：

- hard gate 通过；
- `rms_rnorm_q95 <= 0.06`；
- `max_tension_n <= 2000 N`；
- `all10_tension_p95_n <= 90 N`；
- `beta_close_tension_p95_n <= 70 N`；
- `same_beta_tension_p95_n <= 60 N`；
- `xyz_nn_tension_mae_n <= 28 N`；
- `beta_nn_tension_mae_n <= 18 N`。

## 2k 固定层 sweep 结果

输出目录：

- 数据集：`data/priority_grid_fixed_layer_*_2k/`
- 汇总：`runs/diagnostics/fixed_layer_2k_sweep_v1/summary.md`
- audit：`runs/diagnostics/tension_discontinuity_audit_fixed_layer_v1/summary.json`

7 个固定层全部生成 2000 行，全部通过初筛。

| rank | layer | pass | all10 theta p95 deg | all10 T p95 N | beta-close T p95 N | multi-branch | xyz-NN T | beta-NN T |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `s1_0125_s2_0250` | True | 0.099 | 36.317 | 36.317 | 0.000 | 11.821 | 11.890 |
| 2 | `s1_0250_s2_0250` | True | 0.084 | 39.107 | 39.107 | 0.000 | 13.430 | 13.294 |
| 3 | `s1_0000_s2_0250` | True | 0.130 | 45.364 | 45.364 | 0.000 | 10.320 | 10.560 |
| 4 | `s1_0000_s2_0500` | True | 0.102 | 45.188 | 45.188 | 0.000 | 13.621 | 13.304 |
| 5 | `s1_0125_s2_0500` | True | 0.087 | 45.796 | 45.796 | 0.000 | 14.759 | 15.469 |
| 6 | `s1_0000_s2_0000` | True | 0.204 | 68.506 | 68.506 | 0.000 | 8.616 | 9.072 |
| 7 | `s1_0250_s2_0500` | True | 0.073 | 59.993 | 59.993 | 0.000 | 15.397 | 15.973 |

注意：早期 smoke test 曾使用过同名 `s1_0000_s2_0000_2k` 目录并留下 12 行数据。正式结论已经补算并确认 7 个目录均为 2000 行。

## 张力不连续来源审计

对所有固定层重跑 audit 后，关键结论是：

- 多 branch 指标为 0，说明固定层路线有效去掉了 workspace 近邻内的 branch 混叠；
- active-set flip 在 audit 中为 0，说明当前主要问题不是上下限饱和集合切换；
- 残余张力跳变主要来自 case flip，且最常见的贡献段是第三关节相关绳段；
- 在没有 case flip 的 pair 内，最优固定层非常平滑：
  - `s1_0125_s2_0250`: no-flip T p95 = 7.406N；
  - `s1_0250_s2_0250`: no-flip T p95 = 2.633N。

代表性 audit：

| layer | pair count | all T p95 N | no-flip T p95 N | case-flip-only T p95 N | case flip ratio | active-set flip ratio | top segment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `s1_0125_s2_0250` | 222 | 36.317 | 7.406 | 56.064 | 0.077 | 0.000 | third |
| `s1_0250_s2_0250` | 145 | 39.107 | 2.633 | 137.710 | 0.069 | 0.000 | third |
| `s1_0000_s2_0250` | 409 | 45.364 | 43.733 | 95.274 | 0.066 | 0.000 | third |
| `s1_0000_s2_0500` | 202 | 45.188 | 44.108 | 81.210 | 0.064 | 0.000 | third/second |

解释：固定层已经解决了“同一 workspace 附近混入多 branch”的大问题。剩余张力不连续主要是摩擦 case 的离散切换造成的，尤其集中在第三关节段。

## graph-anchored relabel 结果

只对排名第一的固定层 `s1_0125_s2_0250` 做 8 个 relabel ablation：

- `k = 16, 32`；
- `w_anchor = 20, 40`；
- `anchor_stat = median, huber_mean`。

输出目录：

- 数据集：`data/priority_grid_fixed_layer_s1_0125_s2_0250_2k_relabel_t1_*`
- 汇总：`runs/diagnostics/tension_relabel_fixed_layer_ablation_v1/summary.md`

最佳结果：

| rank | k | w | stat | hard gate | all10 T p95 N | beta-close T p95 N | same-beta T p95 N | xyz-NN T | beta-NN T | rms q95 |
| ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 40 | huber_mean | True | 13.012 | 13.012 | 61.851 | 7.822 | 8.439 | 0.04223 |
| 2 | 32 | 20 | huber_mean | True | 16.067 | 16.067 | 62.057 | 7.986 | 8.663 | 0.04061 |
| 3 | 32 | 40 | median | True | 27.054 | 27.054 | 76.426 | 8.103 | 8.686 | 未在表内展开 |
| 4 | 32 | 20 | median | True | 31.590 | 31.590 | 76.852 | 8.332 | 8.924 | 未在表内展开 |

相对原始 top layer：

- `all10_tension_p95_n`: 36.317N -> 13.012N；
- `beta_close_tension_p95_n`: 36.317N -> 13.012N；
- `xyz_nn_tension_mae_n`: 11.821N -> 7.822N；
- `beta_nn_tension_mae_n`: 11.890N -> 8.439N；
- `same_beta_tension_p95_n`: 约 104.27N -> 61.85N。

这个结果没有通过 silver，只差在 `same_beta_tension_p95_n <= 60N`，当前最佳为 61.85N。该门槛是我们人为设定的严格标准，不代表方向失败。

补测更高 `w_anchor=60` 时，出现单样本硬门槛失败：

- sample_id = 1422；
- `rms_rnorm = 0.06277 > 0.06`；
- `canonical_success = False`；
- 说明继续增大 anchor 权重会开始破坏力平衡可行性。

因此当前推荐配置是：

- 固定层：`s1_0125_s2_0250`；
- relabel：`k=32, w_anchor=40, anchor_stat=huber_mean`；
- 不建议直接继续把 `w_anchor` 提到 60 以上。

## 计算速度

固定层生成：

- 2k 每个 layer 大约 325-349 秒；
- 约 0.16-0.175 秒/样本；
- 7 个 layer 正式生成总耗时约 2387 秒，包含一次此前污染修正前的完整 sweep。

graph-anchored relabel：

- 2k 单变体约 31-32 秒；
- 约 0.0158 秒/样本；
- 8 个变体总耗时约 265 秒。

如果按当前速度估算：

- 固定层 20k：约 54-58 分钟/层；
- 固定层 100k：约 4.5-4.9 小时/层；
- relabel 20k 单变体：约 5.3 分钟；
- relabel 100k 单变体：约 26 分钟。

## 当前结论

1. 之前扩展 workspace 后张力一致性退化到 100N 左右，主要不是 PSO 随机性本身，而是 mixed workspace 数据把多个逆解 branch 混在同一个 `xyz -> theta/T` 映射中。

2. 固定层单支流形是目前最有效的数据生成方向。它把 2k pilot 的 `multi_branch_ball_ratio` 压到 0，并且所有固定层都通过初筛。

3. 最好的固定层 `s1_0125_s2_0250` 在不做 relabel 时已经达到：
   - theta p95 约 0.10 deg；
   - 10mm 张力 p95 约 36N；
   - no-flip 张力 p95 约 7.4N。

4. graph-anchored relabel 可以继续显著改善张力一致性。最佳配置把 10mm 张力 p95 从 36N 降到 13N，但过强 anchor 会破坏力平衡可行性。

5. 下一步如果目标是提升模型拟合性能，建议优先训练以下两个版本做对比：
   - 原始固定层 `s1_0125_s2_0250` 20k；
   - 固定层 + `k=32,w=40,huber_mean` relabel 20k。

6. 不建议马上把多个固定层混合成一个单模型训练集。更合理的路线是：
   - 每个固定层先单独训练，确认单 branch 的模型误差下限；
   - 若需要更大 workspace，再训练 layer-aware / branch-aware 模型，而不是强迫 `xyz -> theta/T` 在全局单值。

# Fixed-Layer 20k/100k 扩展实验记录

## 当前已知结论

- fixed-layer 单支路方法已经把 2k pilot 的 `multi_branch_ball_ratio` 压到 0。
- 当前最佳固定层是 `s1_0125_s2_0250`。
- 2k raw 指标：`all10_theta_p95_deg=0.099`，`all10_tension_p95_n=36.317N`，`xyz_nn_tension_mae_n=11.821N`。
- 2k relabel 最佳配置是 `k=32,w_anchor=40,anchor_stat=huber_mean`。
- 2k relabel 最佳指标：`all10_tension_p95_n=13.012N`，`same_beta_tension_p95_n=61.851N`。
- 张力断点审计显示，剩余张力跳变主要来自 friction case flip，active-set flip 比例为 0。

## 本轮计划

- 先生成 `s1_0125_s2_0250` 的 20k raw fixed-layer 数据。
- 对 20k raw 做 `k=32,w=40,huber_mean` graph-anchored relabel。
- raw 和 relabel 都跑质量诊断，并训练 `mlp,mlp_large,tf_mlp,tf_mlp_large`。
- 只要 20k relabel 质量门槛通过，就生成 100k raw + relabel，并继续做同口径诊断与训练。

## 输出位置

- 运行汇总：`runs/diagnostics/fixed_layer_scale_s1_0125_s2_0250_v1/summary.md`
- 20k raw：`data/priority_grid_fixed_layer_s1_0125_s2_0250_20k`
- 20k relabel：`data/priority_grid_fixed_layer_s1_0125_s2_0250_20k_relabel_t1_k32_w40_huber_mean_iter1`
- 100k raw：`data/priority_grid_fixed_layer_s1_0125_s2_0250_100k`
- 100k relabel：`data/priority_grid_fixed_layer_s1_0125_s2_0250_100k_relabel_t1_k32_w40_huber_mean_iter1`

## 最近运行结果

# Fixed-Layer 20k/100k Scale Experiment

- layer: `s1_0125_s2_0250`
- mixed baseline avg best T MAE: `50.94 N`

## Dataset Quality

| dataset | rows | hard | theta p95 deg | T p95 N | same-beta T p95 N | multi-branch | xyz-NN T | beta-NN T |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20k_raw | 20000 | True | 0.093 | 43.978 | 57.190 | 0.0000 | 3.974 | 4.069 |
| 20k_relabel | 20000 | True | 0.093 | 31.131 | 39.950 | 0.0000 | 3.183 | 3.275 |
| 100k_raw | 100000 | True | 0.093 | 43.336 | 7.463 | 0.0000 | 1.776 | 1.887 |
| 100k_relabel | 100000 | True | 0.093 | 34.694 | 9.445 | 0.0000 | 1.624 | 1.709 |

## Promotion

- promote_to_100k: `True`
- reason: 20k_quality_passed

## Training

### 100k classic MLP 训练记录

本轮原计划训练 `mlp,mlp_large,tf_mlp,tf_mlp_large`。执行前检查发现当前
`dante_env` 里的 TensorFlow 仍是 namespace package：`tf.__version__ = None`，
且没有 `tf.random`，因此 TF MLP 路径不可用。为了完成 100k 数据本身的模型验收，
本轮训练改为 classic-only：`mlp,mlp_large`。

- 训练 runner：`scripts/pipelines/run_fixed_layer_100k_classic_training.py`
- 训练 summary：`runs/diagnostics/fixed_layer_scale_s1_0125_s2_0250_v1/classic_100k_training_summary.json`
- 平铺指标：`runs/diagnostics/fixed_layer_scale_s1_0125_s2_0250_v1/classic_100k_metrics_flat.json`
- raw 输出：`runs/baselines_fixed_layer_s1_0125_s2_0250_100k_raw_v1`
- relabel 输出：`runs/baselines_fixed_layer_s1_0125_s2_0250_100k_relabel_v1`
- longrun：`fixed_layer_100k_classic_train`
- longrun 耗时：`2026-06-20 17:38:35` 到 `2026-06-20 17:54:21`，约 `15.8 min`
- 并行度：`max_parallel=4`
- 结果：`rc=0`

### 100k raw/relabel 训练全量指标

| dataset | split | model | theta MAE deg | T MAE N | T RMSE N | EE p95 mm | fit s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw | iid | mlp | 0.0255 | 8.65 | 15.41 | 6.70 | 51.5 |
| raw | iid | mlp_large | 0.0384 | 6.67 | 13.40 | 8.68 | 79.9 |
| raw | radius | mlp | 0.1460 | 45.04 | 60.03 | 37.40 | 311.3 |
| raw | radius | mlp_large | 0.2171 | 41.50 | 53.45 | 62.11 | 321.4 |
| raw | beta_block | mlp | 0.1927 | 41.44 | 61.46 | 46.81 | 110.8 |
| raw | beta_block | mlp_large | 0.1533 | 33.06 | 55.51 | 41.67 | 301.4 |
| raw | angular_sector | mlp | 0.3336 | 81.51 | 106.91 | 86.37 | 188.3 |
| raw | angular_sector | mlp_large | 0.3320 | 60.96 | 78.59 | 83.98 | 266.3 |
| relabel | iid | mlp | 0.0225 | 7.88 | 14.12 | 5.72 | 223.3 |
| relabel | iid | mlp_large | 0.0267 | 5.79 | 10.63 | 6.19 | 404.8 |
| relabel | radius | mlp | 0.1889 | 86.75 | 114.84 | 50.20 | 207.5 |
| relabel | radius | mlp_large | 0.2842 | 43.59 | 58.53 | 72.90 | 201.8 |
| relabel | beta_block | mlp | 0.1317 | 41.79 | 66.15 | 42.83 | 207.6 |
| relabel | beta_block | mlp_large | 0.1331 | 36.76 | 58.29 | 30.80 | 196.0 |
| relabel | angular_sector | mlp | 0.3766 | 65.09 | 84.53 | 112.09 | 159.5 |
| relabel | angular_sector | mlp_large | 0.2186 | 57.60 | 76.20 | 52.82 | 127.5 |

### best-model 对比

按每个 split 内 `T MAE N` 最低选择 best model，本轮所有 best model 都是
`mlp_large`。

| split | raw best T MAE N | relabel best T MAE N | relabel delta N | raw best EE p95 mm | relabel best EE p95 mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| iid | 6.67 | 5.79 | -0.88 | 8.68 | 6.19 |
| radius | 41.50 | 43.59 | +2.08 | 62.11 | 72.90 |
| beta_block | 33.06 | 36.76 | +3.70 | 41.67 | 30.80 |
| angular_sector | 60.96 | 57.60 | -3.36 | 83.98 | 52.82 |

best-model 平均：

| dataset | avg T MAE N | OOD avg T MAE N | avg theta MAE deg | avg EE p95 mm | OOD avg EE p95 mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 35.55 | 45.17 | 0.1852 | 49.11 | 62.59 |
| relabel | 35.93 | 45.98 | 0.1657 | 40.68 | 52.17 |

固定 `mlp_large` 口径：

| dataset | avg T MAE N | OOD avg T MAE N | avg theta MAE deg | avg EE p95 mm | OOD avg EE p95 mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 35.55 | 45.17 | 0.1852 | 49.11 | 62.59 |
| relabel | 35.93 | 45.98 | 0.1657 | 40.68 | 52.17 |

### 当前结论

1. 100k fixed-layer 单支路数据的质量验收通过。`multi_branch_ball_ratio=0`，
   `all10_theta_p95_deg=0.093`，raw/relabel 的 10mm 张力 p95 分别为
   `43.34N` 和 `34.69N`。
2. 100k 相比之前 mixed 100k 已经明显改善模型可学性。当前 best-model 平均张力
   MAE 约 `35.5-35.9N`，低于 mixed baseline 的 `50.94N`。
3. 但 relabel 并没有让张力 MAE 全面下降。它改善了 IID 和 angular_sector 的张力
   MAE，但 radius、beta_block 上略差；best-model 平均张力 MAE 基本持平。
4. relabel 对几何回代误差更有价值：best-model 平均 EE p95 从 `49.11mm` 降到
   `40.68mm`，OOD 平均 EE p95 从 `62.59mm` 降到 `52.17mm`。
5. 主要瓶颈已经不是局部 branch 混叠或张力局部跳变，而是 OOD 泛化。IID 张力 MAE
   已到 `5.8-6.7N`，但 angular_sector 仍在 `57.6-61.0N`，radius/beta_block
   也在 `33-44N`。
6. 下一步若目标继续提升模型拟合性能，应优先做 OOD 泛化实验：更合理的 split-aware
   采样、输入特征/branch hint、或按固定层/sector 分专家模型；单纯继续加强 relabel
   不是第一优先级。

## 2026-06-24：100k 全模型对比与 MLP 容量扫描

本轮目的有两个：

1. 把当前环境中可稳定运行的对比模型补齐，确认是否有模型能超过 `mlp_large`。
2. 系统扩大 MLP 参数规模，观察是否能通过更大容量降低误差，并判断是否出现过拟合。

### 运行产物

- 模型对比 runner：`scripts/pipelines/run_fixed_layer_100k_model_comparison.py`
- MLP 容量扫描 runner：`scripts/pipelines/run_fixed_layer_mlp_capacity_sweep.py`
- 模型对比 summary：`runs/diagnostics/fixed_layer_scale_s1_0125_s2_0250_v1/model_comparison_summary.json`
- 模型对比报告：`runs/diagnostics/fixed_layer_scale_s1_0125_s2_0250_v1/model_comparison_summary.md`
- 容量扫描逐 trial summary：`runs/diagnostics/fixed_layer_scale_s1_0125_s2_0250_v1/mlp_capacity_sweep_summary.json`
- 容量扫描最终分析：`runs/diagnostics/fixed_layer_scale_s1_0125_s2_0250_v1/mlp_capacity_sweep_final_analysis.md`

### 模型对比结果

补跑模型包括 direct `knn/rf/lgbm`，以及 beta-first `mlp/rf/knn` 的
`direct` 和 `beta_knn` 张力模式。第一次尝试把 `knn,rf,lgbm` 放在同一个
100k job 中，默认 RF/LGBM 过重，`direct_raw_iid` 在约 `119 min` 后被
`SIGKILL(-9)`。因此改为 per-model job：KNN 全量，RF/LGBM 使用轻量参数和
`--max-train-rows=30000`，作为采样基线。

最终模型对比完成 `88` 行 test 指标。OOD 平均张力 MAE 排名前几项：

| rank | family | dataset | model | OOD avg T MAE N | avg T MAE N |
| ---: | --- | --- | --- | ---: | ---: |
| 1 | direct | raw | mlp_large | 45.17 | 35.55 |
| 2 | direct | relabel | mlp_large | 45.98 | 35.93 |
| 3 | beta_first | relabel | rf__direct | 54.34 | 41.15 |
| 4 | beta_first | raw | rf__direct | 54.74 | 41.55 |
| 5 | direct | raw | mlp | 56.00 | 44.16 |
| 6 | direct | relabel | knn | 57.38 | 43.61 |

结论：补充的 KNN/RF/LGBM 和 beta-first 结构化模型都没有超过原来的
direct `mlp_large`。当前最好的可部署单模型仍然是 direct `mlp_large`。

### MLP 容量扫描结果

容量定义：

| capacity | hidden_layer_sizes |
| --- | --- |
| L0 | `(256, 128, 64, 32)` |
| L1 | `(384, 192, 96, 48)` |
| L2 | `(512, 256, 128, 64)` |
| L3 | `(768, 384, 192, 96)` |
| L4 | `(1024, 512, 256, 128)` |
| L5 | `(1536, 768, 384, 192)` |

probe 阶段先在 relabel 的 `iid` 和 `angular_sector` 上跑 `alpha=1e-8`、
`early_stopping=False`。IID 上容量变大确实能继续降低张力误差：

| split | capacity | test T MAE N | fit s |
| --- | --- | ---: | ---: |
| iid | L0 | 4.34 | 242.2 |
| iid | L1 | 4.26 | 361.6 |
| iid | L2 | 3.41 | 453.5 |
| iid | L3 | 3.85 | 526.2 |
| iid | L4 | 3.10 | 1097.6 |
| iid | L5 | 2.95 | 1623.0 |

但 angular-sector OOD 不随容量稳定改善：

| split | capacity | test T MAE N | overfit | fit s |
| --- | --- | ---: | --- | ---: |
| angular_sector | L0 | 72.62 | false | 247.0 |
| angular_sector | L1 | 63.29 | false | 307.3 |
| angular_sector | L2 | 73.67 | true | 404.8 |
| angular_sector | L3 | 69.09 | true | 734.4 |

grid 阶段完成了 L0/L2 的三组 alpha，并完成了 L4 的 `alpha=1e-7` 与
部分 `alpha=1e-6`。完整四 split 配置如下：

| capacity | alpha | avg T MAE N | OOD avg T MAE N |
| --- | ---: | ---: | ---: |
| L2 | 1e-5 | 31.09 | 40.02 |
| L2 | 1e-6 | 33.22 | 43.09 |
| L0 | 1e-5 | 34.04 | 43.94 |
| L4 | 1e-7 | 34.53 | 44.83 |
| L0 | 1e-6 | 35.26 | 45.34 |
| L2 | 1e-7 | 37.00 | 48.15 |
| L0 | 1e-7 | 40.08 | 51.70 |

按单 split 最优：

| split | best capacity | alpha | T MAE N | theta MAE deg |
| --- | --- | ---: | ---: | ---: |
| iid | L4 | 1e-6 | 3.37 | 0.0331 |
| radius | L0 | 1e-5 | 30.18 | 0.2628 |
| beta_block | L4 | 1e-7 | 25.65 | 0.1432 |
| angular_sector | L2 | 1e-5 | 59.05 | 0.2248 |

### 当前结论

1. 扩大 MLP 容量和调 alpha 是有效的，尤其对 IID、radius、beta_block 有明显收益。
   最好的完整配置 `L2 alpha=1e-5` 把 OOD 平均张力 MAE 从原 relabel
   `45.98N` 降到 `40.02N`。
2. 但容量扫描没有解决最难的 `angular_sector`。当前 best 是 `59.05N`，
   仍略差于原 relabel `mlp_large` 的 `57.60N`。
3. L4/L5 在 IID 上可以继续压低误差，但训练代价大，而且 L4 `alpha=1e-7`
   在 angular_sector 是 `72.25N`，没有继续跑完整 L4 网格的性价比。
4. 下一步若目标是继续提高整体拟合性能，不应只继续扩大 MLP。更合理的方向是：
   angular-sector-aware 采样、sector/region expert、加入更明确的几何/角度提示特征，
   或者对 angular holdout 单独做采样补强与模型诊断。

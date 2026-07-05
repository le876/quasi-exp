# Canonical Layer-Field U3 Pilot 实验结果

日期：2026-07-05

## 1. 实验目的

本轮实验执行 `GPT5Pro四次反馈_构型控制策略与EEPose逆解一致性问题.md` 末尾的建议：不再从 6D mixed beta pool 里直接筛数据，而是人为定义一个 3D canonical 构型流形：

```text
u = (a, b, eta)
u -> beta6 -> theta30 -> FK xyz
```

其中 `a,b` 是第三关节两个方向的 bending command，`eta` 是连续 layer / shape-distribution 变量。最终模型仍然学习 `xyz -> theta`，`u` 只用于数据生成阶段，目的是让数据从生成源头更接近单值、连续、可学习。

本阶段是 FK-only 几何实验，没有计算张力，也没有调用 PSO。

## 2. 新增实现

新增脚本：

```text
scripts/generate_canonical_layer_field_dataset.py
```

新增测试：

```text
tests/test_canonical_layer_field_dataset.py
```

核心生成规则：

```text
beta1 = s1(eta) * a
beta2 = s1(eta) * b
beta3 = s2(eta) * a
beta4 = s2(eta) * b
beta5 = a
beta6 = b
```

测试的四条 path：

| path | s1(eta) | s2(eta) |
|---|---|---|
| `path_a_sync_plus` | `0.125 + 0.075 eta` | `0.250 + 0.100 eta` |
| `path_b_redistribute_12` | `0.125 + 0.075 eta` | `0.250 - 0.100 eta` |
| `path_c_redistribute_21` | `0.125 - 0.075 eta` | `0.250 + 0.100 eta` |
| `path_d_wide_redistribute` | `0.125 + 0.125 eta` | `0.250 - 0.150 eta` |

pilot grid：

```text
a,b in [-15, 15] deg, step 0.5 deg
eta in [-1, 1], 21 levels
rows per path = 61 * 61 * 21 = 78141
```

Jacobian gate：

```text
rho >= 2 deg
sigma3 > 0.002 m
kappa < 50
```

这里的 Jacobian 是：

```text
J_h = d FK(g(u)) / d u
```

用归一化变量 `(a/15deg, b/15deg, eta)` 做中心差分计算。

## 3. 输出位置

数据输出：

```text
data/canonical_layer_field_u3_pilot_v1/
```

诊断输出：

```text
runs/diagnostics/canonical_layer_field_u3_pilot_v1/
```

关键汇总文件：

```text
runs/diagnostics/canonical_layer_field_u3_pilot_v1/path_summary.csv
runs/diagnostics/canonical_layer_field_u3_pilot_v1/layer_field_generation_report.json
runs/diagnostics/canonical_layer_field_u3_pilot_v1/README.md
```

模型训练结果：

```text
runs/baselines_canonical_layer_field_u3_path_d_20k_theta_v1/
```

## 4. 四条 path 的几何 gate 结果

| path | x-slab rows | Jacobian pass rows | balanced rows | pass ratio | 10mm theta p95 | multi-branch ratio |
|---|---:|---:|---:|---:|---:|---:|
| `path_a_sync_plus` | 37282 | 25014 | 20000 | 0.6709 | 0.418 deg | 0.0000 |
| `path_b_redistribute_12` | 35696 | 12162 | 12158 | 0.3407 | 0.291 deg | 0.0000 |
| `path_c_redistribute_21` | 35696 | 12162 | 12158 | 0.3407 | 0.291 deg | 0.0000 |
| `path_d_wide_redistribute` | 35586 | 22786 | 20000 | 0.6403 | 0.235 deg | 0.0000 |

主要结论：

- 四条 path 的 `10mm theta RMS p95` 都远低于此前 mixed hierarchical beta 数据的约 `9-10 deg`。
- 四条 path 的 `multi_branch_ball_ratio` 都是 `0`，说明这个 canonical layer-field 生成方式确实消除了局部 workspace 多 beta/theta branch 混入。
- `path_d_wide_redistribute` 的局部 theta 连续性最好，`10mm theta p95 = 0.235 deg`。
- `path_a_sync_plus` 的 Jacobian pass 后可用点最多，`25014`，但 balanced NN p95 稍差。

## 5. workspace 覆盖和稠密度

| path | Jacobian pass x-range | balanced NN p95 | yz-cell x-range p95 | balanced 3D voxels |
|---|---:|---:|---:|---:|
| `path_a_sync_plus` | `1.0000 .. 1.1812 m` | 10.98 mm | 71.87 mm | 15805 |
| `path_b_redistribute_12` | `1.0000 .. 1.0848 m` | 2.10 mm | 42.91 mm | 3596 |
| `path_c_redistribute_21` | `1.0000 .. 1.0848 m` | 2.10 mm | 42.91 mm | 3596 |
| `path_d_wide_redistribute` | `1.0000 .. 1.1444 m` | 4.13 mm | 73.31 mm | 9272 |

解释：

- 原始 x-slab pool 都覆盖到 `x≈1.2m`，但经过 strict Jacobian gate 后，高 x 区域被大量剔除。
- 这符合 GPT5Pro 的提醒：近直线或接近最大伸长边界时，`eta` 对末端位移的独立影响变弱，`J_h` 容易退化。
- 如果目标必须完整覆盖 `x=1.0..1.2m`，下一轮需要放宽 gate 或把近直边界单独作为 chart/trajectory continuation 处理。
- 如果目标是先验证单值可学习机制，`path_d` 已经明显优于 mixed hierarchical beta 数据。

## 6. path_d 20k 模型结果

使用数据：

```text
data/canonical_layer_field_u3_pilot_v1/path_d_wide_redistribute/balanced_20000.parquet
```

训练脚本：

```text
scripts/baselines/run_theta_fk_baseline.py
```

模型：

```text
mlp_l2, mlp_large, lgbm
```

完整结果：

```text
runs/baselines_canonical_layer_field_u3_path_d_20k_theta_v1/metrics_flat.csv
runs/baselines_canonical_layer_field_u3_path_d_20k_theta_v1/all_metrics.json
```

关键 test 指标：

| split | best model by EE p95 | theta MAE | theta p95 | EE p95 |
|---|---|---:|---:|---:|
| iid | `mlp_large` | 0.036 deg | 0.054 deg | 3.79 mm |
| x_slab | `mlp_large` | 0.095 deg | 0.156 deg | 15.24 mm |
| radius | `mlp_l2` | 0.118 deg | 0.203 deg | 12.28 mm |
| beta_block | `mlp_large` | 0.078 deg | 0.126 deg | 6.38 mm |
| angular_sector | `mlp_l2` | 0.263 deg | 0.533 deg | 28.11 mm |

对比此前 hierarchical beta FK-only 数据：

- 旧 100k MLP 的 iid theta MAE 约 `2.405 deg`，EE p95 约 `82.9 mm`。
- 现在 path_d 20k 的 iid MLP large theta MAE 是 `0.036 deg`，EE p95 是 `3.79 mm`。
- 这说明模型拟合瓶颈确实主要来自 mixed inverse branch，而不是 MLP 参数规模不够。

## 7. 当前结论

本轮实验支持 GPT5Pro 的核心判断：

```text
要提高 xyz -> theta 拟合性能，优先解决数据生成阶段的单值构型策略，而不是继续扩大 mixed beta 数据集。
```

canonical layer-field 的效果非常明显：

- 局部 theta 一致性从 mixed 数据的约 `9-10 deg p95` 降到 `0.23-0.42 deg p95`。
- MLP/LGBM 的 `xyz -> theta` 拟合误差从度级下降到 `0.03-0.3 deg` 量级。
- FK 回代 EE p95 从几十毫米到近百毫米，下降到 iid `3.8 mm`、常规 OOD `6-15 mm`、最难 angular sector `28 mm`。

剩余问题：

- Strict Jacobian gate 会把接近最大伸长的高 x 区域剔除，导致 path_d 的可用 x 覆盖只有 `1.000 .. 1.144m`。
- 如果后续目标轨迹需要靠近 `x=1.2m`，不能直接用单个 strict path_d chart。
- 下一步应在两个方向中选一个：
  - 继续做多 chart：例如一个 path_d 主 chart，加一个专门覆盖高 x 边界的 near-straight chart；
  - 或对高 x 区域放宽 `sigma3/kappa`，并单独评估模型误差是否仍可接受。

## 8. 验证命令

单元与诊断 smoke 测试：

```bash
/mnt/ML_projects/conda_envs/dante_env/bin/python -m pytest \
  tests/test_canonical_layer_field_dataset.py \
  tests/test_local_continuity.py \
  tests/test_branch_aware_continuity.py -q
```

结果：

```text
7 passed
```

正式 pilot 命令：

```bash
/mnt/ML_projects/conda_envs/dante_env/bin/python scripts/generate_canonical_layer_field_dataset.py \
  --config configs/robot_rods_only_priority_grid_third_joint_first_v1.yaml \
  --out-dir data/canonical_layer_field_u3_pilot_v1 \
  --report-dir runs/diagnostics/canonical_layer_field_u3_pilot_v1 \
  --a-deg=-15,15,0.5 \
  --b-deg=-15,15,0.5 \
  --eta-count 21 \
  --x-min 1.0 \
  --x-max 1.2 \
  --rho-min-deg 2.0 \
  --sigma3-min-m 0.002 \
  --kappa-max 50 \
  --balanced-rows 20000 \
  --progress
```

模型训练命令：

```bash
/mnt/ML_projects/conda_envs/dante_env/bin/python scripts/baselines/run_theta_fk_baseline.py \
  --dataset data/canonical_layer_field_u3_pilot_v1/path_d_wide_redistribute/balanced_20000.parquet \
  --robot-config configs/robot_rods_only_priority_grid_third_joint_first_v1.yaml \
  --out-dir runs/baselines_canonical_layer_field_u3_path_d_20k_theta_v1 \
  --splits iid,x_slab,radius,beta_block,angular_sector \
  --models mlp_l2,mlp_large,lgbm \
  --max-iter 500 \
  --n-jobs 8
```

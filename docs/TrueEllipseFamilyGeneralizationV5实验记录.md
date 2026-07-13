# True Ellipse Family Generalization V5 实验记录

## 1. 实验目标与结论口径

V5 以 `amp_xy=87.5 mm`、`amp_z=131.25 mm` 的 strict support-backed 为主目标，并使用相同硬门槛独立挑战 `amp_xy=100 mm`、`amp_z=150 mm`。本轮沿用完整 `beta6` canonical branch、360 点轨迹、5×5 法向 tube 与 `xyz -> beta6` 模型路线，不引入 multi-chart、direct-theta、张力、FK loss 或输出后 IK refinement，不放宽 V3/V4 的任何 gate。

严格结论必须同时具备：

1. 固定 family 在目标半径完成 pointwise IK；
2. 360 点 canonical branch 通过中心线、正反一致性与确定性复跑 gate；
3. 完整 360×25 tube 通过局部一致性 gate 并物化；
4. 训练分区对目标半径通过 strict support；
5. 完整目标半径从训练隔离后，最终模型至少 4/5 seeds 通过模型 gate。

任何上游 gate 失败时，后续模型调参不用于 strict 结论。

## 2. 固化输入与实现

- V2/V3/V4 冻结起点：`740d8c0`；
- 只读输入：V2 1M full-`beta6` reachability pool、V2 E100 候选、V3 selected centerline 和 9,000-row tube；
- V5 输出目录：`runs/true_ellipse_family_expansion_v5/` 与 `runs/true_ellipse_family_training_v5/`；
- 搜索、pointwise、branch 与 tube 使用独立 subprocess 并逐任务 checkpoint；
- family 的 center 与 phase 只在搜索阶段确定一次，随后在全部半径固定不变；
- 正式 branch 验证全部 5 条 pointwise 代表 family，避免主目标、stretch 与原始 V3 baseline 之间的候选覆盖漏检。

正式半径锚点为：

`75, 80, 82.5, 85, 87.5, 90, 92.5, 95, 97.5, 100 mm`。

## 3. 固定 family 搜索

- seed geometry：原始 V3 selected family，加两个去重 E100 center；
- 每个 seed 生成 2,048 个 bounded Sobol perturbations；
- 加回 3 个原始 seed 后，共评估 6,147 条固定 family；
- 所有支持指标均从 V2 原始 1M pool 精确重算。

主目标排名第一：

- candidate：`c0273_a100_py210_pz330_s0752`；
- center：`(1.1195695282, 0.0666662281, -0.1607720225) m`；
- phase：`(3.5651668037, 5.9087515950) rad`；
- 75→87.5 mm 最坏 NN P95/最大值：`4.64795/5.79592 mm`。

stretch 排名第一：

- candidate：`c0273_a100_py210_pz330_s0243`；
- center：`(1.1123509584, 0.0925287277, -0.1565356437) m`；
- phase：`(3.6386194114, 5.9601921917) rad`；
- 75→100 mm 最坏 NN P95/最大值：`5.23174/6.66733 mm`。

搜索阶段只用于候选排序，不代替精确 IK、canonical branch 或 tube 结论。

## 4. Pointwise IK

正式评估 5 条代表 family，每个半径 72 个角度。seed budget 最大为 `16→32→64`；单角度获得 `<=2 mm` 可达证书后停止剩余 seed。

- 87.5 mm：`5/5` family 通过全部 72 个角度；
- 100 mm：`2/5` family 通过，分别为 `c0273_a100_py210_pz330_s0243` 和 `c0426_a100_py180_pz60_s1789`；
- stretch 第一候选在 100 mm 的 residual P95/最大值：`0.001683/0.016823 mm`；
- stretch 第二候选在 100 mm 的 residual P95：`0.000307 mm`。

因此，87.5 mm 与 100 mm 都不是被逐点可达性本身否决；pointwise 通过也不等于存在唯一、平滑、可复现的 canonical branch。

## 5. 360 点 canonical branch robustness

正式验证的 5 条 family：

1. `c0273_a100_py210_pz330_s0243`：75、80 mm 通过；82.5 mm 的 forward/reverse branch difference P95 为 `1.008592°`，超过 `1.0°` 硬门槛，后续半径停止；
2. `c0273_a100_py210_pz330_s0752`：75 mm 的 forward/reverse P95 为 `2.045109°`，失败；
3. `c0426_a100_py180_pz60_s1789`：75 mm 的 forward/reverse P95 为 `2.405607°`，且 delta beta max 为 `2.416497°`，失败；
4. `v3_selected_s1008`：75、80、82.5、85 mm 通过；87.5 mm 的 forward/reverse P95 为 `1.313548°`，失败；
5. 原始 `v3_selected`：75 mm 通过；80 mm 的 forward/reverse P95 为 `1.513728°`，失败。

最接近主目标的 `v3_selected_s1008@87.5` 中心线本身表现为：

- residual P95/最大值：`0.000108/0.000130 mm`；
- delta beta P95/最大值：`0.04847/0.75783°`；
- delta2 beta P95：`0.002543°`；
- seam beta RMS：`0.031953°`；
- kappa P95：`40.0015`；
- sigma3 P05：`0.157444 m`。

这些中心线指标均通过；正式失败项是 forward/reverse canonical branch 一致性，而不是 Cartesian tracking、平滑性或条件数。五条 family 中最远的连续 branch 严格通过半径为 `85 mm`。

## 6. 5×5 法向 tube

正式 tube 对每个通过 branch 的 `family@radius` 使用固定 offsets `[-5,-2.5,0,2.5,5] mm²`，每条包含 `360×25=9,000` 个样本。

全部 7 条通过 branch 的 tube 都通过了 V3 固定 gate：

| family | radius (mm) | residual P95 (mm) | residual max (mm) | tube10 beta RMS P95 (deg) | multi-branch ratio |
|---|---:|---:|---:|---:|---:|
| `c0273_a100_py210_pz330_s0243` | 75 | `2.7458e-05` | `5.1020e-05` | `0.34030` | `0` |
| `c0273_a100_py210_pz330_s0243` | 80 | `3.0632e-05` | `6.3435e-05` | `0.36520` | `0` |
| `v3_selected` | 75 | `9.3239e-05` | `0.001053` | `0.52340` | `0` |
| `v3_selected_s1008` | 75 | `3.8506e-05` | `0.000145` | `0.40794` | `0` |
| `v3_selected_s1008` | 80 | `6.1676e-05` | `0.000417` | `0.47923` | `0` |
| `v3_selected_s1008` | 82.5 | `9.2719e-05` | `0.001005` | `0.52965` | `0` |
| `v3_selected_s1008` | 85 | `0.000143` | `0.003872` | `0.58600` | `0` |

所有 tube 的 target success ratio 与 normal-grid coverage 都为 `1.0`，每条均为 9,000 行。`v3_selected_s1008` 从 75→85 mm 连续通过，因此 V5 的 robust trajectory materialization 上限为 `85 mm`。87.5 mm 未进入 tube，不是 tube solver 失败，而是其上游 branch robustness 已失败。

## 7. 数据集、模型与 strict support

全部通过 tube 的诊断 union 包含：

- `63,000` 行；
- `7` 条 trajectory；
- `3` 条候选 family；
- 2 mm voxel 内有 `4` 个 conflict voxels；
- 最大 voxel beta RMS 为 `7.24404°`；
- 跨-family union 的 branch-conflict gate 为 `False`。

不同候选 family 的冲突标签不直接混入监督训练。按照固定-family 的确定性选择规则，正式数据集选中 `v3_selected_s1008`：

- 半径：`75, 80, 82.5, 85 mm`；
- `4` 条完整 trajectory；
- `36,000` 行；
- family 内 conflict voxels：`0`；
- family 内最大 voxel beta RMS：`0.760565°`；
- 数据集 gate：`True`。

训练 preflight 的结果：

- required columns：通过；
- unique sample IDs：通过；
- multiple trajectories/radii：通过；
- 85 mm validation radius materialized：通过；
- source dataset gate：通过；
- 87.5 mm primary radius materialized：`失败`。

因此，已经实现并测试的 24 组模型受控网格与 5-seed 正式训练没有启动。继续训练会缺少 87.5 mm 完整 tube 真值，无法构造计划要求的 whole-radius holdout，也不能产生 strict support-backed 结论。该停止是预注册 gate 的结果，不是运行中断。

## 8. 主目标与 stretch 判定

| 目标 | 判定 | 最早失败阶段 | 证据 |
|---|---|---|---|
| 87.5 mm strict support-backed | `False` | `branch` | `v3_selected_s1008@87.5` 中心线各项通过，但 forward/reverse P95=`1.313548° > 1°`；无正式 tube/whole-radius holdout |
| 100 mm strict support-backed | `False` | `branch` | 2/5 family 的 pointwise IK 通过 100 mm，但没有 family 形成从 75→100 mm 连续、正反一致的 canonical branch |
| robust trajectory materialization | `85 mm` | 87.5 mm branch | `v3_selected_s1008` 的 75/80/82.5/85 mm tube 全通过 |
| multi-radius fixed-family dataset | `True` | 主半径仍缺失 | 36,000 行、4 半径、内部 conflict voxel=0 |

本轮不能把 85 mm 写成新的 strict support 半径。因为正式模型未训练、87.5 mm whole-radius holdout 不存在，当前冻结的 strict support-backed 上限仍是 V4 的 `81.25 mm`；V5 新增的是 `85 mm robust-trajectory-only` 证据。

对 100 mm 的准确表述是：实现已经能够搜索并逐点求解到 100 mm，说明当前预算下它不是简单的逐点不可达；但本轮固定 family 与 branch 参数不能把它提升为严格可复现轨迹，更不能形成 strict model/support 结论。这不是 100 mm 物理不可达证明，而是当前方法和候选预算下的严格负结果。

下一轮若保持现有方法论，应优先优化上游 branch，而不是先扩大 MLP 网格：

1. 把 coarse forward/reverse hysteresis 加入 family search 排名，而不只按 pool NN 支持排序；
2. 对 `lambda_center`、canonical posture 权重和 radial continuation 步长做受控 sweep，硬 gate 保持不变；
3. 优先复查接近门槛的 `s0243@82.5`（`1.008592°`），再验证能否连续推进到 87.5 mm；
4. 只有 87.5 mm 完整 tube 物化后，才执行已经实现的 24-config/5-seed 模型参数优化；
5. 若单 canonical branch 在多轮受控 sweep 后仍无法通过，再把 multi-chart 作为独立方法学升级，而不是与本 V5 结论混合。

## 9. 关键产物

- `runs/true_ellipse_family_expansion_v5/01_family_search/search_report.json`；
- `runs/true_ellipse_family_expansion_v5/02_pointwise/pointwise_report.json`；
- `runs/true_ellipse_family_expansion_v5/03_branch/branch_report.json`；
- `runs/true_ellipse_family_expansion_v5/03_branch/branch_radius_summary.csv`；
- `runs/true_ellipse_family_expansion_v5/04_tube/tube_report.json`；
- `runs/true_ellipse_family_expansion_v5/05_dataset/dataset_report.json`；
- `runs/true_ellipse_family_expansion_v5/06_summary/expansion_summary.json`；
- `runs/true_ellipse_family_training_v5/05_summary/final_goal_report.json`（仅在全部训练前置 gate 通过时生成）。

## 10. 复现实验命令

```bash
/mnt/ML_projects/conda_envs/dante_env/bin/python \
  scripts/analysis/run_true_ellipse_family_expansion_v5.py \
  --phases all --workers 8 --skip-existing
```

若 expansion 数据集包含完整 87.5 mm tube 且通过数据集 gate，再执行：

```bash
/mnt/ML_projects/conda_envs/dante_env/bin/python \
  scripts/analysis/run_true_ellipse_family_training_v5.py \
  --preset formal --phases all --workers 3 --skip-existing
```

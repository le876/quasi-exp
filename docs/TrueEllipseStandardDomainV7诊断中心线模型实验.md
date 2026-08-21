# True Ellipse Standard-Domain V7-D1 诊断中心线模型实验

## 1. 实验目的与结论边界

正式 V7 因 105 mm strict radial checkpoint 未通过而 fail-closed，没有授权 tube、正式 dataset 或正式模型训练。V7-D1 回答一个更窄的问题：已经生成并通过路径审计的 75–104 mm 连续中心线，能否被静态 `xyz → beta6` MLP 拟合，并在 FK 后保持椭圆跟踪。

V7-D1 从设计上不是正式 V7 的补跑。所有报告、模型包、预测与图片均固定写入：

```text
diagnostic_only=true
formal_claims_allowed=false
changes_v7_formal_gate=false
static_inverse_claim_radius_mm=null
```

所以本实验即使在某个半径上通过诊断门，也不能改写正式 V7 的 `formal_radial_gate_pass=false`、`formal_tube_gate_pass=false`，也不能把 V4 的正式可拟合半径上调。

## 2. 数据集与隔离策略

上游来自固定 family `c0273_a100_py210_pz330_s0243` 的 `strict_path_manifest.json`。清单共包含 35 条完整 360 点连续中心线：

```text
75, 76, 77, 78, 79, 80, 81, 82, 82.5, 83.5,
84.5, 85, 86, 87, 87.5, 88.5, 89.5, 90, 91, 92,
92.5, 93.5, 94.5, 95, 96, 97, 97.5, 98.5, 99.5, 100,
101, 102, 102.5, 103.5, 104 mm
```

逻辑总量为 12,600 行。模型 screen 使用一个物理隔离的 parquet：

| 分区 | 半径 | trajectory | 行数 | 进入拟合 |
|---|---|---:|---:|---|
| train | 所有 `R≤100 mm` | 30 | 10,800 | 是 |
| validation | `102.5 mm` | 1 | 360 | 否 |
| post-selection | `101,102,103.5,104 mm` | 4 | 1,440 | 否；screen 文件中物理不存在 |

配置选择只读取 102.5 mm。`selection_report.json` 写盘并计算 SHA-256 后，评估阶段才打开 101、102、103.5 和 104 mm。最终报告记录：

```text
selection_frozen_before_outer_read=true
screen training rows=10800
screen validation rows=360
post-selection rows physically absent=true
trajectory/radius leakage=0
```

审计阶段也遵守同一物理隔离：75–100 mm 与 102.5 mm 的 31 条 screen 输入会立即核对内容 SHA；101/102/103.5/104 mm 只核对“路径存在且 manifest 已注册预期 SHA”，不会读取文件内容。四条外层曲线的实际 SHA、family 与 360 点完整性统一推迟到 selection report 冻结后的 materialization 阶段核验。因此 provenance 审计本身不会绕过 post-selection 隔离。

screen parquet、选型/评估 CSV、预测 parquet 与 prediction manifest 均逐行或逐文件携带同一组非正式声明；缓存复用除了核对上游 task fingerprint，也使用不依赖 size/mtime 缓存的 SHA-256 核对 worker result JSON、模型包、下游 CSV、Parquet、manifest 和 PNG 的当前字节。Screen 与 stability 的结果产物指纹还会直接写入 `final_report.json`。`--preset full` 禁止 `--screen-config-limit`，不能把截断 screen 标成 full。

本实验只训练 centerline，没有生成或使用 tube。因此它衡量的是一条固定 family 路径面上的静态逆映射能力，不代表 tube 邻域鲁棒性。

## 3. 模型协议

输入固定为：

```text
x_target_m, y_target_m, z_target_m
```

模型网格复用 V5 的 24 个 base config，并与两种输出 link 做笛卡尔积：

```text
24 base configs × {identity, tanh_bounds} = 48 configs
```

所有 48 个配置均使用 screen seed `20260711`、完整 10,800 行训练集、`batch_size=256`、`max_iter=800`。每个配置都保存独立 joblib 模型包，不只保存最佳模型。

102.5 mm 选择门同时包含：

- 360 个真实整数相位标签；
- 360 个 `(k+0.5)°` 解析目标点；其 beta 诊断真值由相邻整数相位标签做周期插值。

排序先要求 integer 与 half-phase 两组严格跟踪门同时通过，再依次比较最坏 EE P95、beta P95、逐轴 P95、最小关节余量、拟合时间与稳定 config ID。若没有配置过门，协议会显式写出 `fallback_best_no_gate`；本次 full screen 有 21/48 个配置通过双门，因此没有触发 fallback。

最佳配置为：

```text
mlp_beta6_wide_poly_medium_relu_a1em04__identity
```

48 配置中，identity 有 9 个通过双门，tanh-bounds 有 12 个通过双门。最佳配置仍是 identity，说明 bounded link 能消除 raw-bound 风险，但并不保证获得最低 Cartesian 误差。

## 4. Screen 结果

最佳配置在 102.5 mm 的选择指标为：

| 评估 | EE P95/max (mm) | beta P95 (deg) | axis P95 max (mm) | 最小预测关节余量 (deg) | raw bound violations | gate |
|---|---:|---:|---:|---:|---:|---|
| integer | 2.184463 / 3.572605 | 0.134317 | 1.792841 | 1.967424 | 0 | pass |
| half-phase | 2.252207 / 3.570558 | 0.134145 | 1.796642 | 1.967396 | 0 | pass |

前五名依次为：

| config | worst EE P95 (mm) | worst beta P95 (deg) | min margin (deg) |
|---|---:|---:|---:|
| `wide_poly_medium_relu_a1em04__identity` | 2.252207 | 0.134317 | 1.967396 |
| `wide_poly_heavy_relu_a1em06__tanh_bounds` | 2.494881 | 0.135359 | 1.968642 |
| `large_raw_relu_a1em06__identity` | 2.745886 | 0.146676 | 1.969109 |
| `large_poly_medium_relu_a1em04__tanh_bounds` | 2.779453 | 0.143419 | 1.962093 |
| `large_poly_heavy_relu_a1em04__tanh_bounds` | 2.791618 | 0.149888 | 1.965801 |

48 个 screen 模型的拟合时间总和为 1,547.39 s，中位数为 31.54 s，最大值为 58.11 s。训练以两个 worker、每个 BLAS 单线程执行。

## 5. 五 seed 稳定性与连续半径评估

选型后使用 seeds `20260711..20260715`。首个 seed 直接复用 screen 包，其余四个 seed 重新训练。接受连续路径使用 720 点密集轨迹：360 个整数相位和 360 个半相位。结果如下：

| radius (mm) | 通过 seed | 要求 | 稳定门 | worst EE P95 (mm) | worst beta P95 (deg) | min margin (deg) | bound violations |
|---:|---:|---:|---|---:|---:|---:|---:|
| 100.0 | 5/5 | 4 | pass | 1.640998 | 0.033845 | 1.977177 | 0 |
| 101.0 | 5/5 | 4 | pass | 2.121714 | 0.063437 | 1.971643 | 0 |
| 102.0 | 5/5 | 4 | pass | 2.957370 | 0.130154 | 1.966251 | 0 |
| 102.5 | 4/5 | 4 | pass | 3.733303 | 0.171256 | 1.963540 | 0 |
| 103.5 | 2/5 | 4 | fail | 5.098838 | 0.268421 | 1.958103 | 0 |
| 104.0 | 1/5 | 4 | fail | 6.048756 | 0.367762 | 1.955514 | 0 |

因此，V7-D1 的“诊断稳定、已评估连续半径上界”为 102.5 mm。这个数字不是正式静态逆模型 claim。

### 5.1 103.5/104 mm 为什么失败

失败不再来自 beta3/beta4 越界：全部五个 seed 在所有接受半径上的 raw-bound violation 都为 0，最小预测关节余量仍约 1.96°，beta P95 也远低于 1°。

决定性项变成 Cartesian 逐轴误差：

- 103.5 mm：seed 20260712/13/14 的 axis P95 分别为 3.349/3.295/4.701 mm，超过 3 mm；seed 20260714 的 EE max 还达到 11.107 mm；
- 104 mm：只有 seed 20260711 通过；其余 seed 的 axis P95 为 3.322–5.190 mm，seed 20260712/14 的 EE P95 也超过 5 mm，seed 20260714 的 EE max 为 12.590 mm。

这说明标准域已经解决 V6 的贴界失败，但单 family、centerline-only 的静态 MLP 在 103.5–104 mm 附近出现 seed-sensitive Cartesian 泛化误差。继续扩大正式拟合半径需要新的数据覆盖或模型约束，不能只放宽 raw-bound gate。

## 6. 105–120 mm 目标外推

105、107.5、110、112.5、115、117.5 和 120 mm 没有连续 beta 标签。V7-D1 只从固定 family 的解析椭圆生成 720 个目标点，执行：

```text
target xyz → selected model beta → FK xyz
```

这些记录没有 `true_beta*` 字段，`beta_truth_available=false`，只报告 Cartesian 跟踪、预测连续性和关节边界。最佳可视 seed `20260711` 的结果为：

| radius (mm) | EE P95/max (mm) | axis P95 max (mm) | min margin (deg) | bound violations | Cartesian diagnostic gate |
|---:|---:|---:|---:|---:|---|
| 105.0 | 4.418498 / 5.058687 | 2.960450 | 1.955090 | 0 | pass |
| 107.5 | 6.685637 / 6.949486 | 5.233141 | 1.943628 | 0 | fail |
| 110.0 | 9.280791 / 9.778979 | 7.837224 | 1.932582 | 0 | fail |
| 112.5 | 12.391853 / 12.654009 | 10.707914 | 1.921804 | 0 | fail |
| 115.0 | 15.668273 / 16.473774 | 13.865711 | 1.910515 | 0 | fail |
| 117.5 | 19.425559 / 20.678783 | 17.372740 | 1.899168 | 0 | fail |
| 120.0 | 23.614544 / 25.206425 | 21.175291 | 1.887703 | 0 | fail |

105 mm 的 Cartesian 指标在这个单 seed 反事实下仍过门，但它没有 beta 真值、连续 IK 证书、tube support 或正式 holdout，不能被解释成 105 mm 可拟合阳性。

## 7. 模型轨迹与训练表现图

所有图片沿用项目现有 Matplotlib/Agg 方式。目标为深蓝虚线，模型预测经 FK 后为橙色或红色实线。3D 相机由目标轨迹 SVD 平面法向自动计算，并加 12° 方位偏移；同时给出严格等比例的主平面投影，所以椭圆不会再因侧视角退化成直线。

最佳模型的 100、102.5 和 104 mm 轨迹、主平面投影与逐轴误差：

![V7-D1 best model trajectories](../runs/true_ellipse_standard_domain_training_v7_diagnostic_centerline/05_visualization/individual_models/mlp_beta6_wide_poly_medium_relu_a1em04__identity.png)

24 个 identity 输出模型在 104 mm 的 contact sheet：

![V7-D1 identity model contact sheet](../runs/true_ellipse_standard_domain_training_v7_diagnostic_centerline/05_visualization/contact_sheet_identity.png)

24 个 tanh-bounds 输出模型在 104 mm 的 contact sheet：

![V7-D1 bounded model contact sheet](../runs/true_ellipse_standard_domain_training_v7_diagnostic_centerline/05_visualization/contact_sheet_tanh_bounds.png)

48 个模型在六个接受连续半径上的 EE P95 热图：

![V7-D1 model error heatmap](../runs/true_ellipse_standard_domain_training_v7_diagnostic_centerline/05_visualization/screen_model_error_heatmap.png)

最佳配置五 seed 对比：

![V7-D1 selected configuration across seeds](../runs/true_ellipse_standard_domain_training_v7_diagnostic_centerline/05_visualization/selected_config_seed_comparison.png)

最佳可视 seed 的 105/110/115/120 mm 目标外推：

![V7-D1 unsupported extrapolation](../runs/true_ellipse_standard_domain_training_v7_diagnostic_centerline/05_visualization/selected_config_unsupported_extrapolation.png)

另外，每个模型都有一张独立的 100/102.5/104 mm 复合图，位于：

```text
runs/true_ellipse_standard_domain_training_v7_diagnostic_centerline/
  05_visualization/individual_models/<config_id>.png
```

## 8. 关键产物与哈希

输出根目录：

```text
runs/true_ellipse_standard_domain_training_v7_diagnostic_centerline/
```

总大小约 415 MiB，包含 48 个 screen 模型包、最佳配置额外四个 seed 包、预测 parquet、48 张单模型图与 7 张汇总图。

| 产物 | SHA-256 |
|---|---|
| `00_audit/audit_report.json` | `a997f3c65f22befc4d7b4259438cab2f83caa55b53767133c93ab8fe12200214` |
| `01_dataset/screen_train_validation_centerlines.parquet` | `b2e5e2a63b272d556bde6919d37d13ab41b1b0f48550be1e846b35e535601427` |
| `02_screen/selection_report.json` | `9bcf2654a485f7c03c800909de96707fe153e82857305be2dea32a34cfed284a` |
| `02_screen/screen_config_ranking.csv` | `d44449814b36d4acb9e0e89cadbaf99a1bc2a8aa80a9ec4f9d97ff609aa490d5` |
| `03_stability/stability_report.json` | `c251c612c350de76cae8b4d486e66010c910f5b718228cc3bf50b2d1d57a14b2` |
| `04_evaluation/evaluation_report.json` | `601dc655bf6cf950954ba2f8e97357bb2582332dc1fd3bcf87d47fa3fa1073c7` |
| `05_visualization/visualization_report.json` | `783735c3d1df19521b71399fc1cf09a5236876a2de0bbbdd170e03de4c8975a0` |
| `06_summary/final_report.json` | `3f82911888dd4cbbe9770ad5c58e0ada425829f31a76c884a4d4ecb1be94c4cb` |
| `06_summary/selected_config_accepted_radius_summary.csv` | `8193e648e53e1bbe04baf7e2945bf2a5113fc85fca671689833f16f0c1cfb4c7` |

注意：图片标签修正或重新渲染会改变 visualization/final report 的文件哈希；报告内部同时保存每张图片与输入预测的当前 SHA，交付时以目录中的最终文件为准。

## 9. 运行命令

```bash
env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/tmp/quasi-exp-py311-packages \
  LD_LIBRARY_PATH=/tmp/quasi-exp-libs:/mnt/ML_projects/conda_envs/quasi_exp/lib \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  /mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11 \
  scripts/analysis/run_true_ellipse_standard_domain_diagnostic_training_v7.py \
  --preset full \
  --phases all \
  --workers 2 \
  --skip-existing
```

独立重绘：

```bash
env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/tmp/quasi-exp-py311-packages \
  LD_LIBRARY_PATH=/tmp/quasi-exp-libs:/mnt/ML_projects/conda_envs/quasi_exp/lib \
  /mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11 \
  scripts/analysis/plot_true_ellipse_standard_domain_model_trajectories_v7.py
```

## 10. 最终结论

V7-D1 证实这些 V7 中心线确实训练了模型，不是“只有数据没有拟合”。48 配置 full screen 与最佳配置五 seed 均已完成，且每个模型都有对应的椭圆跟踪图。

在当前 centerline-only、单 family、静态 MLP 协议下，100–102.5 mm 达到五 seed 诊断稳定门，103.5/104 mm 因 Cartesian 逐轴误差的 seed 波动而失败。关节越界已不再是瓶颈。105–120 mm 仅能作为无 beta 真值的目标外推观察。正式 V7 仍没有获准模型 claim，`static_inverse_claim_radius_mm=null`。

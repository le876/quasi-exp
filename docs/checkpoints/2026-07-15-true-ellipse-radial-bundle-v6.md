---
title: True Ellipse Radial Bundle V6 100 mm Checkpoint
date: 2026-07-15
status: frozen
tags:
  - quasi-exp
  - checkpoint
  - true-ellipse
  - radial-bundle
  - beta6
  - 100mm
---

# True Ellipse Radial Bundle V6 100 mm Checkpoint

> [!abstract] 固化目标
> V6 在不修改 V3/V5 centerline、tube、support 或模型 gate 的前提下，把固定 family `c0273_a100_py210_pz330_s0243` 作为完整 `beta(phase,radius)` 路径面延拓到 `amp_xy=100 mm / amp_z=150 mm`，生成十半径、单 family、无冲突的 90,000 行数据集，并使用 92.5/100 mm 整半径 validation/test 验证静态 `xyz -> beta6` 泛化。

## Git 边界

- 分支：`codex/true-ellipse-v6-100mm`；
- V5 固定起点：`6e03dc5b323362f0a919258ed3c7b094a225b84a`；
- V6 checkpoint 提交：本文所在提交；
- 结果文档：[[../TrueEllipseRadialBundleV6实验记录|True Ellipse Radial Bundle V6 实验记录]]。

本 checkpoint 纳入：

```text
scripts/analysis/true_ellipse_radial_bundle_v6_utils.py
scripts/analysis/run_true_ellipse_radial_bundle_v6.py
scripts/analysis/run_true_ellipse_radial_bundle_training_v6.py
tests/test_true_ellipse_radial_bundle_v6.py
tests/test_true_ellipse_radial_bundle_v6_runner.py
tests/test_true_ellipse_radial_bundle_training_v6.py
docs/TrueEllipseRadialBundleV6实验记录.md
docs/checkpoints/2026-07-15-true-ellipse-radial-bundle-v6.md
```

`runs/` 大体积产物继续由 `.gitignore` 排除；本文记录 gate、路径与 SHA-256。

## 阶段结果

| 阶段 | 固化结果 | Gate |
|---|---|---|
| legacy audit | 22 个 V2–V5/robot 关键输入完整；V5 100 mm pointwise 证书保持 72 点，72/72 成功，P95/max=`0.001683/0.016823 mm`；注册目标差异 0 | formal audit pass |
| radial bundle | 同 fixed family 从 75/80 mm 延拓；正式 checkpoint 全通过；100 mm 的 8 个 predictor/cut job 全通过，cut pair P95 max=`0.030487°`，bitwise repeat差异 0 | formal radial pass，last radius=100 mm |
| normal tube | 十条 360×25 tube 全部物化；100 mm 初始 beta P95=`1.019249°`，保留失败产物后用 10 条 outer-shell joint correction 降至 `0.937835°`；success=`0.991222`、P95/max=`0.129934/2.483489 mm`、conflict ratio=0 | formal tube pass |
| dataset | 90,000 行/10 trajectories/10 radii/1 family；sample ID 唯一；conflict voxels=0；最大 voxel beta RMS=`0.972208°` | formal dataset pass |
| split/support | 92.5 mm validation、100 mm test；训练为其余 8 个整半径的 69,120 条非中心线样本；trajectory/radius leakage=0；100 mm NN P95/max=`1.488390/1.626673 mm`、count P10=`871.8` | whole-radius split 与 training-only support pass |
| model | formal 24-config screen 锁定 `mlp_beta6_large_poly_heavy_relu_a1em04`；92.5 mm validation=`1/5`，100 mm held-out-from-fitting/formal-selection evaluation=`0/5`；所有失败 seed 的唯一失败字段是 beta3/beta4 bound violation | `formal_model_gate_pass=false`，strict static-model 半径保持 V4 `81.25 mm` |

## 关键方法边界

1. family center、phase 与 ID 从 V5 固定，扩半径过程中不得重新搜索或混入第二 family。
2. radial predictor 使用全部 360 个 parent angle；`parent_copy` 和 `radial_secant` 都必须跨四个 cyclic cut 通过。
3. `conservative/balanced/loose` 的每个 stage 保留正 radial anchor；candidate graph 只在直接 joint correction 失败时启用。
4. exact rerun 必须复用实际被选中的初始化；若选中 candidate graph，不能误复跑原始 predictor。
5. 100 mm tube 的 outer-shell fallback 处理完整 `n2=±5` 曲线，不删除 70°–110° 困难区间，也不改变 5×5 网格。
6. 92.5/100 mm 的中心线与 offsets 全部从训练隔离；support 还额外排除训练半径中心线。
7. 模型只能读取 XYZ；radius、angle、family、branch 与 split 仅用于审计。
8. 100 mm 不得参与拟合或 24-config validation 排序；审查后进一步禁止 smoke/pilot 读取注册 test。

## 100 mm 关键证据

### Centerline

```text
residual P95/max       0.0166425 / 0.0401516 mm
delta beta P95/max     0.0619091 / 0.1348196 deg
delta2 beta P95        0.0073969 deg
seam beta RMS          0.0337905 deg
cut pair P95 max       0.0304867 deg
sigma3 P05             0.1226281 m
kappa P95              51.4401
exact rerun max diff   0 rad
```

### Tube

```text
initial success        0.9935556
initial residual P95   0.0005611 mm
initial residual max   2.4792995 mm
initial beta P95       1.0192489 deg  (fail)

final success          0.9912222
final coverage         1.0
final residual P95     0.1299336 mm
final residual max     2.4834894 mm
final beta P95         0.9378345 deg
final multi-branch     0.0
```

## 声明环境

```text
/mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11
Python 3.11.5
numpy 1.26.4
scipy 1.11.4
pandas 2.2.2
pyarrow 16.1.0
scikit-learn 1.3.0
joblib 1.2.0
matplotlib 3.10.0
pytest 8.2.2
```

运行时设置 `PYTHONNOUSERSITE=1`，并只使用 `/tmp/quasi-exp-py311-packages` 与 `/tmp/quasi-exp-libs` 补齐声明环境，没有修改 Conda 环境。

## 验证状态

- V6 定向测试：`31 passed`；
- Python 编译与两个 CLI `--help`：通过；
- 正式 radial/tube/dataset：通过；
- smoke training：exit 0；
- formal training：exit 0，validation/test=`1/5,0/5`，正式模型 gate 关闭；
- hash-compatible formal training 复跑：`0.52 s`，选型、task fingerprint 与 5-seed 结论不变；
- 审查后 smoke 全链：exit 0，`test_100_evaluated=false`，holdout predictions 只有 `validation_92p5.parquet`；
- dataset/summary 重生：通过，dataset SHA-256 不变；
- 全仓回归：`299 passed`；
- 提交边界双轴审查及 consolidated 复审：Standards 无 hard/blocking finding，`Spec: no findings`。

## 关键忽略产物

| 阶段 | 相对路径（以 `runs/true_ellipse_radial_bundle_v6/` 为根） | SHA-256 |
|---|---|---|
| audit | `00_audit/audit_report.json` | `83cea23dd72f69d5608bc129a48d351b04291fab04b134e46218bcae0dc1a849` |
| radial | `01_radial/radial_report.json` | `2209d6bfb2228aecf32e7fd5f0db61508248675a56381799ac30178b612ceee6` |
| tube | `02_tube/tube_report.json` | `2070ba9a2c8a5b05ae9193e8747c5aca993033c35fef8a74801e0dccbcde80ff` |
| tube 100 | `02_tube/r100p00/tube_small.parquet` | `5e916b1c7810ea8e4a06c136ff26be030b756eda74e6d93cf69a5bdfd33de69a` |
| dataset | `03_dataset/dataset_report.json` | `afab2a546b2ac70d83d5e5c6da59f67732ba577915f1eac0f4bfb7194ed6d2dc` |
| dataset | `03_dataset/true_ellipse_radial_bundle_tubes_v6.parquet` | `72da6126635a10aa8b44c069a12448052b17d9fb6addfb6aa41824ee16dc63b6` |

formal training 根目录为 `runs/true_ellipse_radial_bundle_training_v6/`：

| 阶段 | 相对路径 | SHA-256 |
|---|---|---|
| audit | `00_audit/audit_report.json` | `e226f9cd822ab8cd179162162dd94946cfd7eb28f9954bfaa9b19149b7e66962` |
| split | `01_split/split_assignment.parquet` | `1923facd987ac0c421db3623db42616c008628005d57500ca12c8f1f41b64791` |
| screen | `02_model_screen/selection_report.json` | `3f7ce4ce97907aed3c865d2ef68b2200d4e18767d25531b8b5659072d6db7361` |
| final | `03_final_models/final_training_report.json` | `edb32a94d4f9e734f8ab60d14cfe0a565ff3e5ea9586133780aacf25f4865914` |
| validation | `03_final_models/validation_92p5_metrics_all_seeds.csv` | `0523a3a8f3592f139584fd9b905a7959939a32e3921765a6fe60b16ef369550e` |
| test | `03_final_models/test_100_metrics_all_seeds.csv` | `760e8e5c9c2eda01ad6076d71e4a946ab6166fd3bbd0b06aa27a011ae763cb31` |
| summary | `04_summary/final_report.json` | `aa7732f03c30ed9bc5922734357c2db71e7aca850ea08d741eae68d448aea724` |

> [!warning] 冻结结论边界
> V6 已严格证明 100 mm robust branch/tube/dataset/support 可以生成，但没有证明当前无约束静态模型在 100 mm 严格通过。初版 smoke 曾在 formal 前读取 100 mm，故不再声称 untouched；100 mm 仍确认未进入拟合或 formal 选型。事后忽略 bound gate 或裁剪输出都会得到 5/5 的诊断性反事实，但不得改写 `formal_model_gate_pass=false`。

---
title: True Ellipse Family Generalization V5 Checkpoint
date: 2026-07-14
status: frozen
tags:
  - quasi-exp
  - checkpoint
  - true-ellipse
  - family-generalization
  - beta6
---

# True Ellipse Family Generalization V5 Checkpoint

> [!abstract] 固化结论
> 本 checkpoint 固化以 `87.5 mm strict support-backed` 为主目标、`100 mm` 为独立 stretch 的 V5 固定 family 扩展实验。Git 固化实现、测试、计划与结果文档；`runs/` 大体积产物继续忽略，但本文记录关键 gate、路径、大小与 SHA-256。失败结果同样是正式实验结论，不通过放宽门槛改写。

## Git 边界

- 分支：`codex/true-ellipse-v5`；
- V2/V3/V4 固定起点：`740d8c00fd8faf9085e3e05ba25eb212d202048f`；
- V5 checkpoint 提交：本文所在提交；
- 计划：[[../plans/2026-07-13-true-ellipse-family-v5|True Ellipse Family Generalization V5 实验计划]]；
- 结果：[[../TrueEllipseFamilyGeneralizationV5实验记录|True Ellipse Family Generalization V5 实验记录]]。

本 checkpoint 纳入：

```text
scripts/analysis/true_ellipse_atlas_utils.py
scripts/analysis/true_ellipse_family_v5_utils.py
scripts/analysis/run_true_ellipse_family_expansion_v5.py
scripts/analysis/run_true_ellipse_family_training_v5.py
tests/test_true_ellipse_family_v5_utils.py
tests/test_true_ellipse_family_expansion_v5.py
tests/test_true_ellipse_family_training_v5.py
docs/plans/2026-07-13-true-ellipse-family-v5.md
docs/TrueEllipseFamilyGeneralizationV5实验记录.md
docs/checkpoints/2026-07-14-true-ellipse-v5.md
```

## 阶段结果

| 阶段 | 固化结果 | Gate |
|---|---|---|
| family search | 3 个 seed、每 seed 2,048 Sobol perturbations，加原始 seed 共 6,147 条固定 geometry family；精确读取 V2 1M pool 排名 | 完成，不以 NN 排名替代 IK/branch |
| pointwise IK | 87.5 mm 为 5/5 family 全角度通过；100 mm 为 2/5 family 全角度通过 | 87.5/100 pointwise 均通过 |
| canonical branch | 五条代表 family 全部正式验证；`v3_selected_s1008` 连续通过至 85 mm，87.5 mm forward/reverse P95=`1.313548°`；stretch 第一候选在 82.5 mm 为 `1.008592°` | 87.5/100 branch 均失败；最远严格 branch=85 mm |
| normal tube | 7 条通过 branch 的 360×25 tube 全部通过；`v3_selected_s1008@85` beta RMS P95=`0.585999°`、multi-branch ratio=`0` | connected robust tube=85 mm |
| dataset | 跨-family union 63,000 行/7 trajectories/3 families，有 4 个冲突 voxel；正式选择 `v3_selected_s1008` 单 family，36,000 行/4 半径、内部冲突 0 | fixed-family dataset gate pass |
| training preflight | 列、ID、多轨迹、多半径、85 mm validation 与源数据集全部通过；`primary_radius_materialized=false` | 24-config/5-seed 训练按停止规则 gated out |
| 主目标 | 87.5 mm 无 robust tube、无 whole-radius holdout、无新 strict model/support 证据 | `87.5 mm strict=false` |
| stretch | 100 mm 有 pointwise 可达证据，但没有连续 canonical branch/tube | `100 mm strict=false` |

## 结论边界

> [!warning] 结论边界
> `85 mm` 是 V5 的 robust-trajectory-only 上限，不是 strict support-backed 上限。由于 87.5 mm branch robustness 失败，正式模型训练没有启动；当前 strict support-backed 仍冻结在 V4 的 `81.25 mm`。100 mm 的负结果约束本轮固定 family、branch 算法与候选预算，不构成物理不可达证明。

本 checkpoint 还固化以下数据治理边界：

1. 不把不同候选 family 在重叠 Cartesian voxel 内的冲突 beta 标签直接混入监督训练；跨-family union 只作为诊断。
2. 正式数据集必须来自一条内部无冲突、至少覆盖 3 个半径的固定 family，并优先主半径、stretch 与最大连续半径。
3. 87.5/100 的 strict 结论必须同时具备 trajectory materialization、training-only support 与 whole-radius 4/5 seed model gate；缺一不可。
4. MLP 网格已经实现和单元测试，但没有在缺少主半径 holdout 时做无效的正式训练。

## 实际运行环境

解释器：

```text
/mnt/ML_projects/conda_envs/dante_env/bin/python
```

实际生成 V5 产物和执行验证的版本：

```text
Python 3.10.19
numpy 1.26.4
scipy 1.10.1
pandas 1.5.3
scikit-learn 1.5.2
joblib 1.5.2
pyarrow 23.0.0
```

`pyproject.toml` 的正式项目环境面向 Python 3.11；上述内容记录本次实验的实际执行环境。跨 Python、NumPy、SciPy 或 scikit-learn 版本迁移时应重跑 gate，不假设优化器与 MLP 数值完全一致。

## 关键忽略产物指纹

这些文件位于 `.gitignore` 排除的 `runs/` 下；摘要、大小和 SHA-256 用于核验本机产物是否与 checkpoint 一致。

| 阶段 | 相对路径 | 字节 | SHA-256 |
|---|---|---:|---|
| search | `01_family_search/search_report.json` | 1739 | `f6f51a220b69f0e7688d7afba9116f0d6d7c09f761b827d05eafbb19508299ed` |
| pointwise | `02_pointwise/pointwise_report.json` | 8813 | `9a97cdb6e553bc1432130725518f0b0655ff281af4686f8ed824274bfb455819` |
| branch | `03_branch/branch_report.json` | 2362 | `3db828e15a4e27c4d5581838fb25f9d4c8fc003222066119ecbe5c1b6e112375` |
| branch | `03_branch/branch_radius_summary.csv` | 13441 | `3897920b54004bbed00a634b3b6d0dafbd5bc288936a04f46846a3348b36393a` |
| tube | `04_tube/tube_report.json` | 1632 | `7f5bdea533fdfb308db8f7c6fae632db8fb586499cdc0ae2dd2afa8caee319b9` |
| tube | `04_tube/tube_radius_summary.csv` | 6081 | `7dfc9fb95342d692b0b16858e4cd03d73922f053f99a4c7c3a541d789e51575d` |
| tube | `04_tube/v3_selected_s1008/r085p00/tube_small.parquet` | 3600197 | `780ebf3f713d8b1b766b6f4e301af99f0d7c74b7d46c09e8ab7894cebc75aada` |
| dataset | `05_dataset/dataset_report.json` | 1886 | `dd06ec9d3f8ae49ee09293b1afd0295462be4a63b457877fda0cf5ed46444986` |
| dataset | `05_dataset/true_ellipse_family_tubes_v5.parquet` | 14525512 | `08331f74949e61c49ef6ecfd13f486898fd87b6c9bd42bb42106acc148f61421` |
| summary | `06_summary/expansion_summary.json` | 2502 | `71c89c597fdd824ec544b6656a119fa7f66999b9a62159517487761da9c29321` |
| training audit | `00_audit/audit_report.json` | 748 | `6b3b6670e2c38f8fe9c442ce34778810ac4946daae5c5456f3c91b34e568b5d6` |

前 10 条以 `runs/true_ellipse_family_expansion_v5/` 为根；最后一条以 `runs/true_ellipse_family_training_v5/` 为根。校验示例：

```bash
sha256sum \
  runs/true_ellipse_family_expansion_v5/05_dataset/true_ellipse_family_tubes_v5.parquet \
  runs/true_ellipse_family_expansion_v5/06_summary/expansion_summary.json \
  runs/true_ellipse_family_training_v5/00_audit/audit_report.json
```

## 验证命令

语法检查：

```bash
/mnt/ML_projects/conda_envs/dante_env/bin/python -m py_compile \
  scripts/analysis/true_ellipse_atlas_utils.py \
  scripts/analysis/true_ellipse_family_v5_utils.py \
  scripts/analysis/run_true_ellipse_family_expansion_v5.py \
  scripts/analysis/run_true_ellipse_family_training_v5.py
```

针对性测试：

```bash
/mnt/ML_projects/conda_envs/dante_env/bin/python -m pytest \
  tests/test_true_ellipse_family_v5_utils.py \
  tests/test_true_ellipse_family_expansion_v5.py \
  tests/test_true_ellipse_family_training_v5.py -q
```

全量回归：

```bash
/mnt/ML_projects/conda_envs/dante_env/bin/python -m pytest -q
```

CLI 导入检查：

```bash
/mnt/ML_projects/conda_envs/dante_env/bin/python scripts/analysis/run_true_ellipse_family_expansion_v5.py --help
/mnt/ML_projects/conda_envs/dante_env/bin/python scripts/analysis/run_true_ellipse_family_training_v5.py --help
```

## 后续工作边界

保持 V5 方法论时，下一轮优先级是 branch-aware 参数优化，而不是扩大模型容量：

1. 在几何候选排名中加入 coarse forward/reverse hysteresis；
2. 受控 sweep `lambda_center`、canonical posture 权重和更细 radial continuation 步长；
3. 首先复查离门槛仅 `0.008592°` 的 `c0273...s0243@82.5`，硬阈值继续固定为 1°；
4. 只有 87.5 mm 完整 tube 通过后才运行 24-config/5-seed 正式模型实验；
5. 若单 branch 在独立实验中仍失败，再另立 multi-chart 规格，不改写本 checkpoint。

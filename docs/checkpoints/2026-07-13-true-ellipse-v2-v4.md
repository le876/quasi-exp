---
title: True Ellipse V2-V4 Checkpoint
date: 2026-07-13
status: frozen
tags:
  - quasi-exp
  - checkpoint
  - true-ellipse
  - branch-lifting
  - beta6
---

# True Ellipse V2-V4 Checkpoint

> [!abstract] 固化结论
> 本 checkpoint 固化从全局 `beta6` 可达域搜索，到 E75 连续 canonical branch lifting、局部法向 tube 生成，再到 `xyz -> beta6 -> theta30 -> FK` 模型训练的 V2/V3/V4 链路。Git 固化内容包含实现、测试、规格与结果文档；`runs/` 下的大体积运行产物仍按仓库规则忽略，但关键产物路径、大小和 SHA-256 记录在本文中。

## Git 边界

- 分支：`canonical-layer-field-u3`
- 固定基点：`8dc94894c52ba2c70ee3c666473520003bffac2f`
- checkpoint 提交：本文所在提交
- 规格来源：[[../GPT5Pro七次反馈_真实椭圆半径扩展与数据生成优化问题|GPT5Pro 七次反馈与 V2 实验计划]]
- V2 记录：[[../TrueEllipseReachabilityAtlasV2实验记录]]
- V3 记录：[[../TrueEllipseBranchLiftingV3实验记录]]
- V4 记录：[[../TrueEllipseBeta6ModelV4实验记录]]

本 checkpoint 只纳入以下实现链路：

```text
scripts/analysis/true_ellipse_atlas_utils.py
scripts/analysis/run_true_ellipse_reachability_atlas_v2.py
scripts/analysis/run_true_ellipse_branch_lifting_v3.py
scripts/analysis/run_true_ellipse_beta6_training_v4.py
```

以及对应四个测试文件、四份规格/结果文档和 `pyproject.toml` 的绘图依赖声明。工作区中更早的 LargeEllipse、TubeAware、张力标注和其他未跟踪实验文件不属于本 checkpoint。

## 阶段结果

| 阶段 | 固化结果 | Gate |
|---|---|---|
| V2 全局可达域 | 1M full-`beta6` Sobol pool；E75 top 5 逐点 IK 全通过；A100 top 5 仍有约 8–10% 点失败 | E75 pointwise pass；V2 strict cyclic branch fail |
| V3 branch lifting | continuation 把相邻 beta p95 从 `2.15666°` 降到 `0.188475°`，但原始 72 点路径 seam=`0.803237°`，不满足当前硬门槛；严格重判后最终选中 `v2_02_forward`，360 点 p95=`0.0379594°`、seam=`0.665756°` | final centerline、selected-branch robustness、tube 全通过 |
| V3 local tube | 360 angles × 25 offsets，共 9000 行；tube residual p95=`9.11061e-05mm`；tube10 beta RMS p95=`0.519174°`；multi-branch ratio=`0` | tube pass，允许训练 |
| V4 模型 | 最优模型 `mlp_beta6_large_poly_heavy_relu_a1em06`；E75 五个 seed 全通过 | E75 5/5 seed gate pass |
| V4 半径 | strict support-backed `81.25/121.875mm`；relaxed `83.25mm`；model-only `84.25mm` | 正式上限只报告 strict support-backed |

> [!warning] 结论边界
> `84.25mm` 是 model-only 外推，不是数据支撑结论。V4 只覆盖当前中心、`py=120°`、`pz=30°` 的局部 canonical tube；本阶段没有张力、LGBM、direct-theta 或输出后 IK refinement。`330–360°` sector holdout 的 EE p95 为 `3.76146mm`；该 open-sector 诊断本轮通过 model gate，但不替代完整轨迹 holdout。

## 规格完成边界

本次固化的是一条可复现的 V2 → V3 → V4 实验链，不等价于七次反馈计划中的所有扩展项已经完成：

1. V2 已完成 1M 全 `beta6` Sobol 可达域、粗粒度 ellipse family 搜索和候选逐点 IK，但尚未实现计划中的 Phase 2 局部精搜；报告里的 `refined` 候选仍是 coarse top 的透传结果。逐点 IK 也尚未补齐多来源 seed、失败重试与候选级 360 点 fine sweep，因此 E100 只能视为当前搜索预算下的负结果，不能作为不可达证明。
2. V3 已把接缝硬门槛校正为计划要求的 `seam <= 0.75°`，并对缓存 report 重新计算 gate，避免旧布尔值绕过新阈值。当前 atlas 实质上仍是单条 canonical branch 上的单 chart 局部 tube pilot；尚无多 chart、chart overlap 一致性与切换 gate。
3. V4 已验证单条 360 点轨迹的 exact regeneration、支持集隔离与五 seed 稳定半径，但还没有形成三条及以上轨迹的数据集，也没有按完整轨迹进行 train/validation/test holdout。因此当前半径结论只能约束这条局部轨迹及其 tube，不能外推为全局机器人工作空间结论。

V3 的 robustness 许可由“被选 canonical branch 在 forward/reverse 间可复现、确定性复跑一致且自身 centerline gate 通过”共同决定；全部候选 run 的通过率保留为搜索诊断，不再让无关的平滑分支否决已验证的被选分支。

## 实际运行环境

实验与本次验证使用：

```text
Python 3.10.19
numpy 1.26.4
scipy 1.10.1
pandas 1.5.3
scikit-learn 1.5.2
joblib 1.5.2
pyarrow 23.0.0
matplotlib 3.10.0
```

解释器：

```text
/mnt/ML_projects/conda_envs/dante_env/bin/python
```

`pyproject.toml` 的正式项目环境仍面向 Python 3.11；上面记录的是生成当前 V2/V3/V4 产物时的实际历史环境。环境迁移时必须重新运行测试和关键 gate，不能假设跨版本数值完全一致。

## 关键忽略产物指纹

这些文件位于被 `.gitignore` 排除的 `runs/` 下。SHA-256 用于判断本机产物是否仍是本 checkpoint 对应版本。

| 阶段 | 相对路径 | 字节 | SHA-256 |
|---|---|---:|---|
| V2 | `02_ellipse_family_search/top_candidates_by_radius.csv` | 23764 | `f3006a3ac4207bd0502fcc9627ab93dbf85cf6a9dd5299537337d141518fe504` |
| V2 | `03_fullbeta_pointwise_ik/candidate_feasibility_summary.csv` | 504 | `683bf224e8116ad8b1f8e651ebd6534c18c7e9f8f740bafa1c3d96e35fbd215c` |
| V2 | `04_cyclic_branch_linking/branch_candidate_summary.csv` | 416 | `5f4c05c64dbbe0c5afdb62b9d12df54228b15553ae598e8e02c3dda22818e273` |
| V2 | `10_failure_diagnostics/branch_threshold_sweep.csv` | 4400 | `e4389d8e3b417e03bc3c6319260d1fb2f7b32d2a7ed488a4fe6d2e1fca5f4720` |
| V3 | `05_robustness/selected_centerline_360.parquet` | 194551 | `a91fba77eb0837d8af0ca02879ec453498cc6e73967526c77c3222e0fe966bb0` |
| V3 | `06_local_tube/tube_small.parquet` | 3596378 | `2db9f0f993b56b379e9c90872eff1afe89af48c9925302f8cae5de6ec46916ad` |
| V3 | `06_local_tube/tube_quality_report.json` | 478 | `2a762bd447fcf527b2bdbd731aece50a8a969f522df31fcb78b20551cfe29ae2` |
| V3 | `07_summary/final_gate_summary.json` | 12435 | `d1c29dfa07aa4f6b2740def36085e7d12aa2f1c0f35fd611edc3627e3f1cf300` |
| V4 | `00_input_audit/audit_report.json` | 1157 | `591f423fce38b27e1df39f7e758ab25ec2e2cbd81e9bc279b64180cb18dd31c1` |
| V4 | `03_final_models/final_training_report.json` | 2037 | `592a9f6e5f8b16169ea5104d0a9886db82d416f9166765d898ae58a0b473ecaf` |
| V4 | `04_radius_sweep/stable_radius_report.json` | 1775 | `159dd0e8e2c0b4ac68a365f7e292861654097d803ea90acd20027c12014fbb84` |
| V4 | `06_summary/final_report.json` | 608 | `2bcd7ce3d8d17cdd99e8bb6628af89a4c0feb18986ba84212de26eb88f935a3e` |

表中相对路径分别以以下目录为根：

- V2：`runs/true_ellipse_reachability_atlas_v2/`
- V3：`runs/true_ellipse_branch_lifting_v3/`
- V4：`runs/true_ellipse_beta6_training_v4/`

校验示例：

```bash
sha256sum \
  runs/true_ellipse_branch_lifting_v3/06_local_tube/tube_small.parquet \
  runs/true_ellipse_beta6_training_v4/04_radius_sweep/stable_radius_report.json
```

## 验证命令

语法检查：

```bash
/mnt/ML_projects/conda_envs/dante_env/bin/python -m py_compile \
  scripts/analysis/true_ellipse_atlas_utils.py \
  scripts/analysis/run_true_ellipse_reachability_atlas_v2.py \
  scripts/analysis/run_true_ellipse_branch_lifting_v3.py \
  scripts/analysis/run_true_ellipse_beta6_training_v4.py
```

针对性测试：

```bash
/mnt/ML_projects/conda_envs/dante_env/bin/python -m pytest \
  tests/test_true_ellipse_atlas_utils.py \
  tests/test_true_ellipse_reachability_atlas_v2.py \
  tests/test_true_ellipse_branch_lifting_v3.py \
  tests/test_true_ellipse_beta6_training_v4.py -q
```

全量回归：

```bash
/mnt/ML_projects/conda_envs/dante_env/bin/python -m pytest -q
```

CLI 导入检查：

```bash
/mnt/ML_projects/conda_envs/dante_env/bin/python scripts/analysis/run_true_ellipse_reachability_atlas_v2.py --help
/mnt/ML_projects/conda_envs/dante_env/bin/python scripts/analysis/run_true_ellipse_branch_lifting_v3.py --help
/mnt/ML_projects/conda_envs/dante_env/bin/python scripts/analysis/run_true_ellipse_beta6_training_v4.py --help
```

## 后续工作边界

本 checkpoint 之后的首要科学任务是给 V3/V4 canonical tube 增加 12 路张力标签并验证张力连续性。下列结构性改进留待独立重构，不应与该实验 checkpoint 混合：

1. 统一 V2/V3/V4 的 schema 和 JSON 序列化 helper；
2. 把 runner 的 worker、phase、reporting/visualization 拆为独立模块；
3. 取消脚本级 `sys.path` 注入，形成可安装的 analysis package；
4. 将单轨迹 tube 扩展为多中心、多相位、多半径的 atlas，并按完整轨迹做 holdout。

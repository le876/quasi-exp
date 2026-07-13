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
| canonical branch | 五条代表 family 全部正式验证；`v3_selected_s1008` 连续通过至 85 mm，87.5 mm forward/reverse P95=`1.313585°`；stretch 第一候选在 82.5 mm 为 `1.020413°` | 87.5/100 branch 均失败；最远严格 branch=85 mm |
| normal tube | 7 条通过 branch 的 360×25 tube 全部通过；`v3_selected_s1008@85` beta RMS P95=`0.586004°`、multi-branch ratio=`0` | connected robust tube=85 mm |
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

## 双轴验收补强

提交边界后的 Standards/Spec 双轴审查发现并关闭了 strict-claim 防护缺口：

1. formal 模式强制恰好 5 个唯一 seed，稳定 gate 需要绝对数量至少 4 个通过；smoke/pilot 永不产生 strict 阳性；
2. 若最终选择周期优化路径，确定性复跑现在使用同一份未被修改的初值和同一优化算法，不再复跑原始 continuation；
3. training audit 强制正式数据集只有一个 fixed family，worker 再按 evaluation family 防御性过滤；
4. radius sweep 只在有物化 centerline 真值时计算 beta P95，缺 beta 真值时 strict model gate 必然失败；
5. branch 无论何时命中 dual goal 都继续验证全部 5 条代表 family；
6. 正式 tube offsets 精确固定为 `[-5,-2.5,0,2.5,5] mm`；
7. 100 mm 未物化时明确记录 `stretch_whole_radius_held_out=false`，不再无条件写为 true。

最终复审先后关闭五类 blocking：

8. `formal_expansion_protocol_gate` 固定完整目标、锚点、5/5 family、72/360 点、预算、随机种子和精确 offsets；正式训练只接受同时通过 formal protocol 与 formal dataset gate 的来源。
9. expansion 每阶段/每分块与 training audit/split/screen/train/sweep/worker 均绑定策略版本、上游内容哈希和任务指纹；`--skip-existing` 不再按“文件存在”盲复用，评估半径若进入训练分区会被 worker 直接拒绝。
10. formal dataset report 记录并认证最终 parquet、trajectory manifest 与 robot config 的 resolved path、SHA-256 和字节数；training audit 重新核对当前文件、协议指纹、目标/anchors、精确 360×25 完整性以及 manifest 对应关系。任一绑定输入或输出被改写时 expansion/training 缓存都会失效。
11. `formal_training_protocol_gate` 强制 `formal` preset、87.5/100 mm 目标、85 mm validation、完整 anchors、恰好 5 个唯一 seed、`screen_config_limit=0`、完整 24-config 网格及 V4 baseline；screen、train、sweep 与最终 strict claim 必须逐级继承这条证据链，缩减 screen 或改 validation 不能产生 formal 阳性。
12. `formal_family_coverage_gate` 不再把 CLI 的“最多 5 条”当成实际覆盖：pointwise report 与 selected CSV 必须共同证明恰好 5 个唯一 family，branch 必须选中并执行同一组 5 个唯一 family，且 `all_selected_families_executed=true`。dataset gate 绑定三份上游文件的 path+SHA，training audit 再独立读取并复算覆盖指纹。

正式产物审计显示，当前 12 条已执行 branch 记录的 `selected_path` 全部为 `forward/reverse` continuation，没有旧 optimized-path 证据；实际 offsets 也精确为规定的 25 个组合。最终在声明环境中强制失效旧缓存并重算全部 5 条 family 和 7 条 360×25 tube；formal expansion/family-coverage/dataset/training-protocol gate 均为 `true`。training audit 的 36 项检查只有 `primary_radius_materialized=false`，因此 strict claim 链按设计关闭。数值存在小幅环境漂移，但 85、87.5 或 100 mm 的实验判定不变。

## 实际运行环境

解释器：

```text
/mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11
```

最终生成 V5 正式产物和执行验证的版本：

```text
Python 3.11.5
numpy 1.26.4
scipy 1.11.4
pandas 2.2.2
scikit-learn 1.3.0
joblib 1.2.0
pyarrow 16.1.0
PyYAML 6.0.1
tqdm 4.66.4
matplotlib 3.10.0
pytest 8.2.2
```

本机 `quasi_exp` 环境的若干共享库链接带有不可解析的 reparse tag，且 `ml` extra 缺少 matplotlib；本次仅在 `/tmp` 建立 SONAME 修复链接与 `matplotlib==3.10.0` 隔离层，并设置 `PYTHONNOUSERSITE=1` 排除用户级同名 `tests` 包，没有修改 Conda 环境。该环境与 `pyproject.toml` 的 Python、直接依赖及 `ml/dev` 固定版本一致。声明环境重算使部分优化指标发生小幅漂移，例如 `s0243@82.5` forward/reverse P95 从旧环境的 `1.008592°` 变为 `1.020413°`，但 gate 与最终结论不变。

## 关键忽略产物指纹

这些文件位于 `.gitignore` 排除的 `runs/` 下；摘要、大小和 SHA-256 用于核验本机产物是否与 checkpoint 一致。

| 阶段 | 相对路径 | 字节 | SHA-256 |
|---|---|---:|---|
| search | `01_family_search/search_report.json` | 3002 | `86c56f6b07be60486e02bf8fb77d987d2ae5107da3b99dc8ae54e6c5dea28ec3` |
| pointwise | `02_pointwise/pointwise_report.json` | 9023 | `e4336dcbd89a7d91dce77fdc094f0ac798dfbab95d7a3adb595d676331350011` |
| pointwise | `02_pointwise/selected_families.csv` | 1220 | `add5c5b3c673cb73920f3fcf27d482bbb96946beae2259e8e06a05752c691c92` |
| branch | `03_branch/branch_report.json` | 3505 | `cd07ea0142c08e6e2c769a7fa1cd0e8ad1b0108b19851f78e73d09e95cd5be59` |
| branch | `03_branch/branch_radius_summary.csv` | 11458 | `cd922cce6b71e9bdb22fb094432c39c5e9379663590e0768fc291ea96dd5b92c` |
| tube | `04_tube/tube_report.json` | 2733 | `a9b96a962b8e0a77170715a5e442f59ebc9a9c4ea16c7235016c0e6657c1338b` |
| tube | `04_tube/tube_radius_summary.csv` | 6488 | `f79306e9adb3df6b37194edb52b14b9ff6531f7a7c9d3b09d8211e9a7e942963` |
| tube | `04_tube/v3_selected_s1008/r085p00/tube_small.parquet` | 3606893 | `58508355fe81d7cf53e0a78489073e42c66bfa973e34033e24a4c028cfcc4d49` |
| dataset | `05_dataset/dataset_report.json` | 5816 | `9929c7c18ad2676b86da26f16aae2cfdad186ac8dee5d23c542daf58f5481c3d` |
| dataset | `05_dataset/true_ellipse_family_tubes_v5.parquet` | 14721577 | `b0424c1f1fbd2f7a4228bbeff51a24549e0f7b547784e90b05ed5f6647ad1c26` |
| dataset | `05_dataset/trajectory_manifest.csv` | 1751 | `4ea07fccfffae9158f6d4769231b16f5c4b8ba8cb6d78dc9261e651e74c12a30` |
| summary | `06_summary/expansion_summary.json` | 7867 | `74edead6c766f90c35529fd4c4b02b480bf573075d7e70a5a392ef1d85cd3cf4` |
| run | `run_report.json` | 37582 | `148bbfdb0008f98b7bc8ceba4c1c8a96fcc6b15468e64d8755917ecd8fd4aeb7` |
| training audit | `00_audit/audit_report.json` | 5785 | `05b6ef979e828c11f30e796fe02ed720094098be326e9059dc8eb6506d97d40d` |

前 13 条以 `runs/true_ellipse_family_expansion_v5/` 为根；最后一条以 `runs/true_ellipse_family_training_v5/` 为根。校验示例：

```bash
sha256sum \
  runs/true_ellipse_family_expansion_v5/05_dataset/true_ellipse_family_tubes_v5.parquet \
  runs/true_ellipse_family_expansion_v5/05_dataset/trajectory_manifest.csv \
  runs/true_ellipse_family_expansion_v5/06_summary/expansion_summary.json \
  runs/true_ellipse_family_training_v5/00_audit/audit_report.json
```

## 验证命令

语法检查：

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/tmp/quasi-exp-py311-packages \
LD_LIBRARY_PATH=/tmp/quasi-exp-libs:/mnt/ML_projects/conda_envs/quasi_exp/lib \
/mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11 -m py_compile \
  scripts/analysis/true_ellipse_atlas_utils.py \
  scripts/analysis/true_ellipse_family_v5_utils.py \
  scripts/analysis/run_true_ellipse_family_expansion_v5.py \
  scripts/analysis/run_true_ellipse_family_training_v5.py
```

针对性测试：

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/tmp/quasi-exp-py311-packages \
LD_LIBRARY_PATH=/tmp/quasi-exp-libs:/mnt/ML_projects/conda_envs/quasi_exp/lib \
/mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11 -m pytest \
  tests/test_true_ellipse_family_v5_utils.py \
  tests/test_true_ellipse_family_expansion_v5.py \
  tests/test_true_ellipse_family_training_v5.py -q
```

全量回归：

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/tmp/quasi-exp-py311-packages \
LD_LIBRARY_PATH=/tmp/quasi-exp-libs:/mnt/ML_projects/conda_envs/quasi_exp/lib \
/mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11 -m pytest -q
```

最终验证结果：V5 针对性测试 `64/64 passed`，全量回归 `268/268 passed`；语法检查与两个 CLI `--help` 均退出 0。正式 expansion 的 `--skip-existing` 全链复跑退出 0，14 个关键产物的字节数与 SHA-256 均与上表一致；正式 training audit 按停止规则退出 1，36 项检查中只有 `primary_radius_materialized=false`，dataset/manifest/robot 与 5-family 上游证据的 path+hash、expansion/training 协议、360×25 完整性全部为 true，未进入 split/screen/train/sweep。

CLI 导入检查：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/quasi-exp-py311-packages \
LD_LIBRARY_PATH=/tmp/quasi-exp-libs:/mnt/ML_projects/conda_envs/quasi_exp/lib \
/mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11 scripts/analysis/run_true_ellipse_family_expansion_v5.py --help

PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/quasi-exp-py311-packages \
LD_LIBRARY_PATH=/tmp/quasi-exp-libs:/mnt/ML_projects/conda_envs/quasi_exp/lib \
/mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11 scripts/analysis/run_true_ellipse_family_training_v5.py --help
```

## 后续工作边界

保持 V5 方法论时，下一轮优先级是 branch-aware 参数优化，而不是扩大模型容量：

1. 在几何候选排名中加入 coarse forward/reverse hysteresis；
2. 受控 sweep `lambda_center`、canonical posture 权重和更细 radial continuation 步长；
3. 首先复查离门槛 `0.020413°` 的 `c0273...s0243@82.5`，硬阈值继续固定为 1°；
4. 只有 87.5 mm 完整 tube 通过后才运行 24-config/5-seed 正式模型实验；
5. 若单 branch 在独立实验中仍失败，再另立 multi-chart 规格，不改写本 checkpoint。

---
question_id: Q08
question_number: 8
question_confirmed_by_user: true
question_confirmation_summary: "判断当前数据集质量瓶颈、审视是否让模型实际表现参与 gate，并重构通向 1 m 椭圆的整体方法"
date: "2026-07-19"
status: ready-to-send
project: "quasi_exp / True Ellipse beta6 inverse"
evidence_cutoff: "2026-07-19 17:52:26 CST (UTC+08:00)"
source_snapshot: "主工作树 canonical-layer-field-u3@740d8c00fd8faf9085e3e05ba25eb212d202048f（dirty）；V7 工作树 codex/true-ellipse-v7-standard-domain-120mm@85730c92ba38cf1c0a1e4edeb2ac55b457633dac（dirty，当前 Gate-v2 实现与记录未全部提交）"
---

# GPT-5 Pro 第八次交接：数据集质量瓶颈、Gate 与 1 m 椭圆方法审视

## 0. 经用户确认的 GPT-5 Pro 任务

用户已经确认的任务原意：

> 请根据最近几轮 True Ellipse 实验的完整结果，判断目前主要影响数据集质量的瓶颈是什么；判断是否需要放宽或重构 gate，使模型的实际拟合与轨迹表现能够参与验证数据集质量；提出进一步提升数据集质量和模型可拟合椭圆半径的方法。理想目标是椭圆达到 1 m 半径，因此需要审视当前整套方法。当前困境不是简单的量变，而是质变。

确认范围：对 V4、V6、原始 V7、V7 Gate-v2、外层 tube 失败、正式模型、候选 κ sweep 与候选标签 lineage 的现有证据作整体判断，并给出有明确判断依据的、可执行的整体重构与实验计划。旧的过渡期上下文文档保持独立，本文件是新的 Q08，不追溯覆盖旧文档。

## 1. 当前有效事实与最短摘要

### 1.1 证据截止时间

本文件只使用截至 `2026-07-19 17:52:26 CST (UTC+08:00)` 已存在于项目中的代码、配置、正式报告、诊断报告、图片和 checkpoint。该时间之后的运行或修改不属于本次证据。

### 1.2 当前有效事实

1. 当前正式、可审计的静态逆模型结论已经从 V4 的 `81.25 mm` 提升到 Gate-v2 的 `100.0 mm`：固定单 family 的 90,000 行数据集通过正式数据 gate，`xyz -> beta1..6` 的五个 MLP seed 在 92.5 mm validation 与 100 mm test 的 integer/half-phase 四组中心线评估均为 `5/5` 通过，`formal_model_gate_pass=true`。
2. 当前更大半径的限制不只是某一个频繁出现的 gate。101.25 mm 与 102.5 mm tube 同时失败三个未修改 gate：`success ratio >= 0.99`、`residual max <= 3 mm`、`beta local RMS P95 <= 1°`；105 mm 径向 checkpoint 在所有预注册 κ 阈值 150/200/250/300/400 下都未形成完整通过路径。
3. 固定 family 与当前机器人几何本身给出了接近现有前沿的硬上界。机器人末端从原点的最大理论伸展为 `1.215498 m`；固定 family 在相位 `φ=π/2` 时有 `x=cx+R`，其中 `cx=1.1123509584333815 m`，所以零残差的必要条件为 `R <= 0.1031470415666185 m = 103.1470415666 mm`。这与 102.5 mm 径向前沿及 105 mm 首次失败几乎重合。
4. 用户提出的“1 m 椭圆半径”在当前项目中存在术语边界：当前参数 `R` 是三个正弦分量的尺度，不等于三维椭圆的主半轴。对固定相位，实际主半轴约为 `1.953403629R` 与 `0.658949362R`。无论把 1 m 理解为 `R=1 m`，还是理解为实际长半轴为 1 m，在保持当前机器人、基座、中心和 family 不变时，轨迹都超出最大伸展范围。
5. V6 与 Gate-v2 的对照证明“几何/数据通过”和“标签是否适合无约束静态回归”是不同层级。V6 的 100 mm 几何、tube、数据和 support 通过，但标签几乎贴住 beta3/beta4 硬边界，正式模型因微小 raw-bound violation 得到 validation `1/5`、test `0/5`；V7 标准关节域和 margin barrier 使全数据最小关节余量提高到 `1.800186°`，Gate-v2 同类静态 MLP 得到四组中心线 `5/5`。
6. 模型表现还揭示了数据标签 lineage 的可学习性差异：正式与 κ=150 候选数据的 90,000 个 XYZ 完全相同，81,000 行 beta 标签相同，只有 100 mm 的 9,000 行标签不同；候选标签把 100 mm test beta P95 从正式 lineage 的 `0.259–0.275°` 降到 `0.088–0.140°`，但该运行预先标为 evidence-only，不能改写正式 claim。

### 1.3 关键边界

1. 当前正式 claim 只是：固定一个 family、仅以目标 `xyz` 为输入、输出 `beta1..6`、再映射为 30 个 theta 并用 FK 评估的静态中心线逆模型，在 100 mm 注册 test 上通过。它不是多 family、整工作空间、动力学、时序控制、张力控制或真实机器人部署结论。
2. 100 mm tube 模型评估是预先标记的 diagnostic，不是正式中心线 gate。其五个 seed 只有 `1/5` 通过；四个失败 seed 只因少数点的 `EE max` 达到 `11.5966–15.4900 mm` 而失败，说明中心线阳性不能自动外推为整个 ±5 mm tube 的尾部风险已解决。
3. κ sweep、候选标签与候选模型均为 evidence-only；V6 的事后裁剪为 post-hoc。它们可以说明机制差异或生成假设，不能授予新的正式半径。
4. “1 m”尚未被定义为当前公式的 `R`、三维椭圆的长半轴、某个平面内半径、直径、末端相对中心的位移，还是更大尺寸机器人上的归一化目标；当前证据也没有改变机器人长度、基座、轨迹中心或 family 后的 1 m 实验。

## 2. 项目目标与本轮边界

### 2.1 用户给定的项目目标

项目最终目标来自用户和原论文复现任务：构建能够支持连续体机器人控制的数据与模型，最终关系包含末端目标位置 `(x,y,z)`、30 个关节角 `theta1..theta30` 以及 12 路张力 `T1..T12`。本次用户进一步给出理想目标：椭圆尺度最终应达到 1 m，并要求把它视为需要整体方法重构的质变问题，而不是在 100 mm 附近继续做小幅参数调整。

### 2.2 当前实际覆盖范围

当前 True Ellipse 链只覆盖：

```text
固定 family 的目标 xyz
    -> 由连续 IK / 路径面生成 beta1..beta6 标签
    -> beta6 交替展开为 theta1..theta30
    -> forward kinematics 得到末端 xyz
    -> 静态 MLP 学习无约束 xyz -> beta1..beta6
```

每个中心线半径使用 360 个相位点；tube 使用每个相位上的 `5×5` 法向偏移网格，即每个半径 9,000 行。当前 family 固定为 `c0273_a100_py210_pz330_s0243`，没有通过搜索另一个 family 来绕开外层困难区。

### 2.3 当前证据未覆盖的内容

- 没有 `T1..T12` 张力标签、准静态平衡质量、摩擦参数辨识或张力可实现性实验。
- 没有末端姿态目标、速度/加速度、历史状态、滞回、动力学、MPC 或闭环控制实验。
- 没有多 family 留一验证、跨 family 泛化或全工作空间静态逆模型结论。
- 没有把 phase、radius、family、上一时刻 beta、分支 ID 或局部 chart 作为模型输入的正式对照；当前正式模型输入只有三个目标 Cartesian 坐标。
- 没有改变机器人尺度、基座、轨迹中心或几何 family 后的 1 m 可达性与数据生成实验。

## 3. 相对上次材料新增的事实

| 类型 | 已发生的变化 | 实际证据 | 证据等级 |
|---|---|---|---|
| 实现/配置 | V7 把 beta3/beta4 标准域改为 ±10°，引入 balanced joint-margin barrier；Gate-v2 又把 `target_105_achieved` 与当前前沿的 tube/data/model 授权解耦，并把重复的 `tube_success_ratio == 1.0` 恢复为项目既有的 `>=0.99`，其他 tube gate 不变 | `v7_original_summary_report.json`、`v7_gate_v2_checkpoint.md` | V7/Gate-v2 当轮 formal；实现来自 dirty worktree |
| 实验 | 生成 Gate-v2 90,000 行正式数据，完成 48-config screen、五 seed 正式训练和倾斜 3D 轨迹可视化；100 mm 中心线正式模型通过 | `gate_v2_dataset_report.json`、`gate_v2_final_training_report.json`、`gate_v2_seed_20260713.png` | formal；图片为 visualization-only |
| 失败补证 | 完成 101.25/102.5 mm 外层 tube 联合修正和 κ=150/200/250/300/400 sweep；三个未修改 tube gate 仍阻止外层进入数据，所有 κ 候选仍在 105 mm 失败 | `gate_v2_formal_tube_summary.csv`、`gate_v2_candidate_tube_summary.csv`、`gate_v2_kappa_sweep_report.json`、`v7_gate_v2_checkpoint.md` | formal replay + evidence-only / diagnostic-only |
| 结论修正 | “没有模型拟合”不再成立；正式静态中心线模型当前已拟合并通过到 100 mm。但“达到 105 mm”与“整 tube 模型稳定”仍不成立 | `gate_v2_final_training_report.json`、`v7_gate_v2_checkpoint.md` | formal + diagnostic boundary |
| 新增确定性边界 | 由当前机器人长度、末端偏移、family 中心与目标公式可直接推出固定 family 的零残差必要上界 `R<=103.147 mm`，以及当前几何下的 1 m 目标不可达 | `robot_config_standard_100k.yaml`、`robot_lengths.csv`、`robot_end_effector.csv`、`v6_audit_report.json` 与本文件 §4.2 的公式 | deterministic derived fact，不是新的优化实验 |

## 4. 实验方法与复现口径

### 4.1 代码与工作树 fixed point

| 项目 | 值 |
|---|---|
| 项目根 | `/mnt/ML_projects/quasi_exp` |
| 主工作树分支 / HEAD | `canonical-layer-field-u3` / `740d8c00fd8faf9085e3e05ba25eb212d202048f` |
| V7 工作树分支 / HEAD | `codex/true-ellipse-v7-standard-domain-120mm` / `85730c92ba38cf1c0a1e4edeb2ac55b457633dac` |
| V7 worktree | `/mnt/ML_projects/quasi_exp/.worktrees/true-ellipse-v7-standard-domain-120mm` |
| dirty 状态 | V7 runner、training runner、atlas/V6 utils 与对应测试有已修改文件；Gate-v2 runner、gate sweep、tube replay、绘图脚本、实验记录和 checkpoint 中有未跟踪文件 |
| 仅 checkout HEAD 是否足以复现 | 否。当前 Gate-v2 实现和记录没有全部进入 `85730c9`，必须先冻结 dirty diff 与未跟踪文件；现有 artifact 仍可按其 SHA 和 task fingerprint 审计 |

### 4.2 目标几何、机器人和确定性可达性边界

固定 family 参数：

```text
family_id = c0273_a100_py210_pz330_s0243
center    = (1.1123509584333815,
             0.0925287277087819,
            -0.1565356436707808) m
phase_y   = 3.638619411393083 rad
phase_z   = 5.960192191688433 rad
```

目标公式为：

```text
x(φ) = cx + R sin(φ)
y(φ) = cy + R sin(φ + phase_y)
z(φ) = cz + 1.5 R sin(φ + phase_z)
```

因此 `R` 是生成公式中的尺度。把式子写成 `c + A[sinφ, cosφ]^T` 后，固定 family 的 `A` 两个奇异值为 `1.953403629R` 和 `0.658949362R`，即真实三维椭圆的两个主半轴。

机器人配置为三段、每段十盘、共 30 个 theta；`theta_sign=-1`。FK 使用 `lengths[0:30]`，其和为 `1.160000 m`，再加末端局部偏移 `0.055498 m`，直线姿态末端为：

```text
p(theta=0) = (1.215498, 0, 0) m
```

串联链任意姿态的末端欧氏距离不能超过全部有效链长与末端偏移之和 `1.215498 m`。在 `φ=π/2` 时 `x=cx+R`，所以仅由 x 坐标就有必要条件：

```text
cx + R <= 1.215498
R <= 1.215498 - 1.1123509584333815
R <= 0.1031470415666185 m = 103.1470415666 mm
```

这个必要上界不依赖 IK 优化器、标签、模型或 gate。作为对照：100/101.25/102.5/104/105 mm 在 `φ=π/2` 的目标点欧氏距离分别约为 1.212444/1.213671/1.214901/1.216382/1.217372 m；104 和 105 mm 的该点已经超出理论伸展约 0.884 和 1.874 mm。有限 residual gate 可能暂时接受靠近不可达点的近似解，但不能把它变成零残差可达。

对 1 m 的两种常见解释：

- 若 `R=1 m`，则 `x(π/2)=2.1123509584 m > 1.215498 m`，且实际长半轴约为 1.9534 m。
- 若“实际长半轴=1 m”，则 `R=1/1.953403629=0.5119269695 m`，仍有 `x(π/2)=1.6242779279 m > 1.215498 m`。

所以保持当前机器人、基座、中心与 family 不变时，这两种 1 m 定义都不可达。该结论不等价于“任何重新定义后的 1 m 任务都不可行”。

### 4.3 路径与标签生成方法

当前 `xyz -> beta6` 是欠定的 `6 -> 3` 逆问题。IK 可多解；当前所谓 canonical branch 是由初始 seed、半径 continuation、正 radial anchor、whole-curve correction 与 candidate graph 共同选择的连续分支，不是已经证明的全局、seed-independent 唯一逆映射。

V6/V7 不再逐相位独立求 IK，而把 `beta(phase, radius)` 当作路径面：

1. 每个半径同时使用完整 360 点 parent curve、`parent_copy` 和 `radial_secant` predictor；
2. 每个候选经过 0/90/180/270 四个 cyclic cut；
3. 直接联合修正失败时使用 candidate graph rescue；
4. V7 在标准关节域内加入 joint-margin barrier；
5. tube 以中心线局部法向的 `[-5,-2.5,0,2.5,5] mm × [-5,-2.5,0,2.5,5] mm` 网格生成 25 条外壳轨迹；
6. 外壳失败后执行 whole-surface / whole-curve joint correction，不删除困难相位。

| 参数 | 实际值 | 来源 |
|---|---:|---|
| beta1/2 bounds | ±5° | `robot_config_standard_100k.yaml` |
| beta3/4 bounds | ±10° | `robot_config_standard_100k.yaml` |
| beta5/6 bounds | ±15° | `robot_config_standard_100k.yaml` |
| centerline phase points | 360 | `v7_original_summary_report.json` → `protocol.final_points` |
| cyclic cuts | 0, 90, 180, 270 | `v7_original_summary_report.json` → `protocol.cut_indices` |
| radial base/retry step | 1 / 0.5 / 0.25 mm | `v7_original_summary_report.json` |
| legacy strict κ P95 gate | ≤150 | `v7_gate_v2_checkpoint.md` |
| candidate κ sweep | 150, 200, 250, 300, 400 | `gate_v2_kappa_sweep_report.json` |
| tube success floor | ≥0.99 | `v7_gate_v2_checkpoint.md` |

### 4.4 数据生成与划分

| 项目 | 实际口径 |
|---|---|
| 数据来源 | Gate-v2 通过的十个完整半径 tube；单一固定 family |
| 生成方式 | 360 phase × 25 normal offsets = 9,000 行/半径；十半径共 90,000 行 |
| 半径 | 75, 80, 82.5, 85, 87.5, 90, 92.5, 95, 97.5, 100 mm |
| train | 除 validation/test 之外的八个整半径；69,120 条非中心线训练行 |
| validation | 整半径隔离 92.5 mm；另有 half-phase centerline challenge |
| test | 整半径隔离 100 mm；另有 half-phase centerline challenge |
| seed | 20260711, 20260712, 20260713, 20260714, 20260715 |
| 泄漏审计 | trajectory leakage=0，radius leakage=0；smoke/pilot 不读取注册 test，formal final 才读取 test |
| 历史暴露边界 | 100 mm 在 V6 已经被历史实验读取，因此 Gate-v2 test 在本协议内隔离，但不是全项目历史上的 virgin test |

### 4.5 baseline、评价和 gate

| 项目 | 实际定义 | 是否正式 |
|---|---|---|
| baseline / selected model | 静态 MLP，输入仅 `x_target_m,y_target_m,z_target_m`；锁定配置 `[512,256,128,64] + poly_medium + ReLU + alpha=1e-6 + identity output` | Gate-v2 formal |
| geometry metrics | Cartesian residual P95/max、success ratio、coverage、Jacobian sigma3/κ、四 cut、cut invariance、确定性重跑 | 对各自注册协议 formal；外层候选为 evidence-only |
| label metrics | beta 一阶/二阶变化、seam、voxel conflict、joint bounds/margins、multi-branch | 对各自注册协议 formal |
| support/split metrics | training-only NN distance/count、完整半径隔离、trajectory/radius leakage | Gate-v2 formal |
| model metrics | FK 后 EE P95/max、逐轴偏差、beta P95、平滑、seam、raw bounds/margins、至少 4/5 seed | 中心线四组为 formal |
| tube model metric | 同类指标，但 `formal_gate_role=false` | diagnostic-only |

Gate-v2 的 0.99 修改只删除 V7 新增的重复 `==1.0` 条件；它没有放宽 residual、局部 beta、coverage、multi-branch、margin 或 cut gate，而且在读取 101.25/102.5 mm 最终结果前固定并写入新的 fingerprint。

### 4.6 复现入口与产物

V7 原始正式命令在实验记录中为：

```text
env PYTHONNOUSERSITE=1 \
  PYTHONPATH=/tmp/quasi-exp-py311-packages \
  LD_LIBRARY_PATH=/tmp/quasi-exp-libs:/mnt/ML_projects/conda_envs/quasi_exp/lib \
  /mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11 \
  scripts/analysis/run_true_ellipse_standard_domain_v7.py \
  --preset formal --phases radial --workers 12
```

Gate-v2 相关入口位于 V7 dirty worktree：

```text
scripts/analysis/run_true_ellipse_standard_domain_tube_gate_replay_v7.py
scripts/analysis/run_true_ellipse_standard_domain_training_v7.py
scripts/analysis/run_true_ellipse_standard_domain_gate_sweep_v7.py
scripts/analysis/plot_true_ellipse_standard_domain_model_trajectories_v7.py
```

机器人输入验证已在证据截止前执行并通过：

```text
python3 scripts/validate_robot_inputs.py \
  --config configs/robot_rods_only_standard_100k.yaml
```

主要产物：

- `/mnt/ML_projects/quasi_exp/runs/true_ellipse_standard_domain_v7_gate_v2_99pct/03_dataset/`
- `/mnt/ML_projects/quasi_exp/runs/true_ellipse_standard_domain_training_v7_gate_v2_99pct/`
- `/mnt/ML_projects/quasi_exp/runs/true_ellipse_standard_domain_gate_sweep_v7/`

由于 Gate-v2 工作树 dirty，上述入口不应被误解为仅凭 HEAD 就能字节级复现现有 artifact；报告中的 SHA、task fingerprint 和本交接包 manifest 是当前证据审计边界。

## 5. 实验结果

| Fact ID | 已观察结果 | 数值 | 证据等级 | 实际附件/JSON key | 适用边界 |
|---|---|---:|---|---|---|
| F1 | 当前机器人直线最大伸展 | 1.215498 m | deterministic config/FK fact | `robot_lengths.csv`、`robot_end_effector.csv`、`robot_config_standard_100k.yaml` | 当前机器人输入与 FK 约定 |
| F2 | 固定 family 的零残差必要尺度上界 | R ≤ 103.1470415666 mm | deterministic derived fact | F1 + `v6_audit_report.json` → `family.center_x_m` + §4.2 公式 | 当前中心、基座、family 不变 |
| F3 | 当前 R 与实际椭圆主半轴比例 | 1.953403629R / 0.658949362R | deterministic derived fact | `v6_audit_report.json` 的 family 相位 + §4.2 公式 | 当前固定 family |
| F4 | V4 strict/support-backed 静态模型前沿 | 81.25 mm；relaxed 83.25；model-only 84.25 | historical（V4 当轮 formal/diagnostic） | `v4_stable_radius_report.json` → `stable_radii` | 不代表当前 Gate-v2 结论 |
| F5 | V6 100 mm 数据链 | 90,000 行；geometry/tube/dataset/support 通过 | historical（V6 当轮 formal） | `v6_checkpoint.md` | V6 窄 beta3/4 域 lineage |
| F6 | V6 正式模型 | validation 1/5，test 0/5；formal fail | historical（V6 当轮 formal） | `v6_final_training_report.json` | 失败 seed 唯一失败字段为 raw beta3/4 bounds，详见 checkpoint |
| F7 | V6 标签贴界 | 100 mm beta3/beta4 最小余量 0.009656806° / 5.728769e-9°；约 26.2%–30.6% tube 行在任一边界 0.05° 内 | historical diagnostic | `v6_checkpoint.md` | 不能自动外推为 V7 标签状态 |
| F8 | 原始 V7 径向前沿 | strict 102.5 mm；105 mm 首次失败；下游未授权 | V7 formal | `v7_original_summary_report.json` | 原始 V7 耦合 KPI 的协议 |
| F9 | Gate-v2 100 mm tube | success 0.993111；coverage 1；residual P95/max 0.053237/2.479538 mm；local RMS P95 0.955817° | Gate-v2 formal replay | `gate_v2_formal_tube_report.json`、`gate_v2_formal_tube_summary.csv`、`v7_gate_v2_checkpoint.md` | 100 mm 数据准入 |
| F10 | 101.25 mm tube 首次外层失败 | best/final success 0.989333/0.988667；residual max 3.685917 mm；local RMS 1.175044° | evidence-only / diagnostic | `gate_v2_formal_tube_summary.csv`、`gate_v2_candidate_tube_summary.csv`、`v7_gate_v2_checkpoint.md` | 失败 success、max residual、local RMS 三门 |
| F11 | 102.5 mm tube 最终失败 | best/final success 0.979222/0.978667；residual P95/max 0.140529/4.896392 mm；local RMS 1.307521° | evidence-only / diagnostic | `gate_v2_candidate_tube_summary.csv`、`v7_gate_v2_checkpoint.md` | 外层不进入正式数据 |
| F12 | 外层 tube 失败定位 | 101.25 mm：102 个失败、98 个在 n2=-5 mm；102.5 mm：192 个失败；主要为约 76°–103°，两者最坏点均为 84°、(+5,-5) mm | diagnostic-only | `v7_gate_v2_checkpoint.md` | 只证明失败聚集，不证明唯一根因 |
| F13 | Gate-v2 正式数据质量 | 90,000 行；conflict voxel 0；max voxel beta RMS 0.852693°；margin min/P01/P05 1.800186/1.941607/2.066736°；无越界/贴界；split/support/challenge 通过 | formal | `gate_v2_dataset_report.json` | 单 family、十半径、test=100 mm |
| F14 | Gate-v2 正式模型 100 mm | integer EE P95 1.649803–2.515722 mm、max 2.448309–3.302809 mm、beta P95 0.259335–0.274777°；half-phase 同样 5/5 | formal | `gate_v2_final_training_report.json` | 中心线静态逆模型 |
| F15 | 100 mm tube 模型尾部 | 1/5；四个失败 seed 仅 EE max 超 10 mm，范围 11.5966–15.4900 mm | diagnostic-only | `gate_v2_final_training_report.json`、`gate_v2_seed_20260713.png` | 不参与 formal centerline gate |
| F16 | κ sweep | 150/200/250/300/400 的注册径向前沿均为 102.5 mm，105 mm 均首次失败；无候选可注册 | evidence-only | `gate_v2_kappa_sweep_report.json` | 不修改 legacy κ=150 |
| F17 | κ=400 的分支差异 | 105 mm 四个 parent_copy 较早 stage κ≈338.24–338.26 可过候选阈值；四个 radial_secant κ≈471.37–471.38 仍失败；106.25/106.5 graph 仅 343/341 角层 | evidence-only | `gate_v2_kappa_sweep_report.json`、`v7_gate_v2_checkpoint.md` | 候选路径未形成完整 360 点正式证据 |
| F18 | 正式/候选标签 lineage 对照 | XYZ 全同；81,000/90,000 beta 同；100 mm 9,000 行 beta RMS 差异 P95/max 0.121382/0.813683° | evidence-only comparison | `v7_gate_v2_checkpoint.md` | 不能据此事后替换正式标签 |
| F19 | 候选 lineage 模型 100 mm beta 拟合 | beta P95 0.088070–0.139516°，显著低于正式 lineage 的 0.259324–0.275467°；Cartesian 指标相同范围 | evidence-only | `v7_gate_v2_checkpoint.md` | 说明标签 lineage 可学习性不同，不授予正式 claim |

### 5.1 通过项

- Gate-v2 已证明 100 mm 固定 family 的完整 geometry/tube/data/support/challenge/centerline-model 证据链可以闭合。
- 标准 beta3/beta4 域与 margin policy 下，90,000 行数据没有 raw joint 越界、贴界样本或 3° voxel conflict，五 seed 中心线模型也没有 output clipping 或 bound violation。
- 3D 轨迹图使用目标轨迹 SVD 主平面法向再偏转 12°，可视化中椭圆不会因侧视角退化成直线；`gate_v2_seed_20260713.png` 是五个 seed 中的一张代表图。
- Gate-v2 的 train/validation/test 按整半径隔离，并对 validation/test 增加 half-phase challenge；当前报告中的 trajectory/radius leakage 均为 0。

### 5.2 失败与反例

- 放宽单一 κ gate 到 400 没有提高注册径向前沿；105 mm 的 radial_secant 和更外层不完整角层仍阻断路径。
- 把 tube success 从错误重复的 100% 恢复到 99% 使 100 mm 数据链闭合，但 101.25/102.5 mm 同时失败三项未修改 gate，因此不能把外层失败概括成一个过严 success gate。
- 100 mm 中心线模型 5/5 不代表 100 mm 整 tube 已经稳定：tube diagnostic 仍有四个 seed 的少数最大误差超 10 mm。
- V6 反例显示，几何、覆盖与数据 gate 全过并不保证无约束静态模型可稳定输出合法关节值；标签的 joint-margin 与分支选择会改变模型可学性。
- 正式/候选 100 mm 标签使用同一 XYZ 却产生不同 beta 拟合误差，说明“FK 可行且局部连续”不足以唯一确定最易学习的 inverse label surface。
- 当前 fixed family 的硬几何必要上界约 103.147 mm；在此定义下，通过继续加数据、改 MLP 或放宽 gate 不可能把零残差 R 推到 1 m。

## 6. 证据等级、失败结果与未知事项

### 6.1 正式事实

- V4 的 strict/support-backed 静态半径是 81.25 mm；V6 当轮模型 formal fail；原始 V7 strict radial 前沿是 102.5 mm；Gate-v2 当前正式静态中心线 claim 是 100 mm。
- Gate-v2 100 mm 正式数据集为 90,000 行、十完整半径、一个 family，正式 SHA-256 为 `997260c4b9798a3f492a4ba994e6afcd8fca1a68c6a884ad22253221dfb655dc`。
- Gate-v2 五 seed 的四组正式 centerline evaluation 均 5/5，通过至少 4/5 的稳定性要求；`formal_model_gate_pass=true`，`static_inverse_claim_radius=100.0 mm`，`target_105_achieved=false`。
- 机器人当前输入的最大链长与末端偏移和为 1.215498 m；固定 family 的 `R<=103.147 mm` 是由几何直接得到的必要条件，不是 gate 选择。

### 6.2 诊断、探索、evidence-only、post-hoc 或历史事实

| 内容 | 等级 | 能支持什么 | 不能支持什么 |
|---|---|---|---|
| V6 raw-bound failure 与标签贴界 | historical + diagnostic | 说明无关节余量的标签可使很小回归误差触发硬边界失败 | 不能证明 Gate-v2 当前仍贴界 |
| V6 事后裁剪后 5/5 | post-hoc | 说明当轮失败字段对裁剪敏感 | 不能把 V6 改判为正式 100 mm 阳性，也不能直接确立部署策略 |
| 101.25/102.5 mm tube 修正与失败定位 | diagnostic/evidence-only | 说明外层失败的相位、offset 和同时失败的 gate | 不能形成更大正式数据半径 |
| κ sweep | evidence-only | 比较预注册阈值下的完整路径证据 | 不能修改 legacy κ 或授予 105 mm claim |
| 正式/候选标签模型差异 | evidence-only | 说明同一 XYZ 的 inverse label lineage 可具有不同可学习性 | 不能事后选择更好 test 标签并称为正式结果 |
| seed 轨迹 PNG | visualization-only | 直观展示中心线目标/预测轨迹与视角 | 不能替代 JSON 数值 gate |

### 6.3 已发现的实现或证据缺口

- 当前逆映射是 3D 目标到 6D beta 的多解问题，但数据只保存一个由 continuation/seed 选择的 branch；尚无全局唯一性证明，也没有显式 branch/state 输入。
- candidate generator 会记录 predictor conditioning，但现有 sweep 没有证明 candidate graph 在每个解候选层面真正用 conditioning 选择了更优完整路径；阈值 sweep 主要改变路径准入，不等价于解决分支几何。
- 当前正式模型只优化静态中心线 claim；tube 模型诊断的最大误差尾部没有进入正式目标。
- 所有正式数据来自一个 family；当前 conflict voxel 与 support 统计不能替代跨 family 的多值性和泛化审计。
- 当前 V7/Gate-v2 实现位于 dirty worktree，checkout HEAD 不足以复现；在方法继续演进前需要把代码 fixed point 与 artifact lineage 一起冻结，否则后续对照可能混入实现变化。

### 6.4 尚未知或尚未执行

- 用户的 1 m 目标在物理上究竟指 `R`、实际主半轴、直径、某个平面半径，还是缩放机器人后的归一化尺度，尚未定义。
- 没有评估改变轨迹中心、family 相位、机器人总长、基座位置或硬件尺度后，1 m 目标的可达集合与所需 joint domain。
- 没有执行多 family atlas、分支分类/混合专家、带历史状态模型、局部 chart 模型或前向可微约束训练的正式对照。
- 没有证明 101.25/102.5 mm tube 失败中，多少比例来自绝对 workspace 不可达，多少来自求解器/分支选择，多少来自 tube 标签平滑性要求。
- 没有对 tube 模型 `EE max` 尾部逐点做与 reachability、conditioning、support density、offset 和标签 lineage 的联合归因。
- 没有真实机器人闭环、噪声、张力可实现性、动力学或安全验证。

### 6.5 当前不能声称

- 不能声称当前方法可在同一机器人、同一中心和同一 family 上把 `R` 扩展到 1 m。
- 不能声称放宽 κ、success、residual 或平滑 gate 就能产生更高质量数据；现有证据只显示某些门的结果和外层失败结构。
- 不能声称模型通过可以替代几何可达性、物理关节边界、数据 lineage、split 泄漏和安全 gate。
- 不能声称 100 mm centerline 5/5 等同于整个 100 mm tube、多个 family 或控制部署通过。
- 不能声称候选标签比正式标签“真实”或“正确”；只知道它们在同一 XYZ 上不同，且当前静态模型对候选 lineage 的 beta 拟合误差更低。

## 7. 决策所需的客观对照

本节只比较已经执行的方法、协议和结果，不给出路线排序。

### 7.1 历史阶段对照

| 对照对象 | 已执行输入/方法 | 已观察结果 | 证据等级 | 已知限制 |
|---|---|---|---|---|
| V4 | 旧 family 的静态 MLP 半径 sweep 与 support gate | strict 81.25 mm；relaxed 83.25；model-only 84.25 | historical | family、数据 lineage 与 V7 不同 |
| V6 | 固定 family；连续路径面；窄 beta3/4 域；90k 数据 | geometry/data 到 100 mm，但模型 1/5、0/5，raw beta3/4 越界 | historical formal | 标签几乎贴关节边界 |
| 原始 V7 | 标准 beta3/4 域 + margin barrier；105 KPI 与下游授权耦合 | strict radial 102.5；105 fail；未训练 | formal | KPI 阻断了较小完整前沿的模型评估 |
| Gate-v2 formal | KPI 解耦；tube success 重复门恢复 0.99；正式 90k + 48 config + 5 seeds | 100 mm centerline 5/5，formal pass；105 仍 false | formal | 单 family、静态 xyz 输入、tube 模型尾部仍有失败 |
| κ=150 候选下游 | 与正式 XYZ 相同，100 mm 标签 lineage 不同 | centerline 5/5；beta P95 更低；policy 仍不允许注册 | evidence-only | 不能事后改写正式标签或 κ policy |

### 7.2 各层 gate 当前回答的问题

| 层级 | 当前 gate 能排除的失败 | 已观察的阳性 | 已观察的阴性或缺口 |
|---|---|---|---|
| 物理/工作空间 | 目标超过总链长、关节硬界 | 100 mm 中心线目标仍在必要 reach 边界内 | 固定 family 零残差必要上界约 103.147 mm；1 m 当前定义不可达 |
| 路径/几何 | residual、连续性、seam、四 cut、conditioning | strict radial 到 102.5 mm | 105 mm 失败；κ sweep 不提升完整前沿 |
| tube/标签面 | 25 offset coverage、success、residual tail、local beta、margin | 完整通过到 100 mm | 101.25/102.5 同时失败三门；失败集中于局部外壳 |
| 数据集 | 完整半径、冲突、margin、support、split/challenge | 90k Gate-v2 数据全部通过 | 单 family；未证明多分支唯一性或跨 family 泛化 |
| 中心线模型 | 静态 inverse 的跨 seed FK/beta/边界表现 | 92.5/100 mm 四组 5/5 | 只验证中心线，且 100 mm 有历史暴露 |
| tube 模型诊断 | 局部邻域的模型尾部风险 | EE P95、beta、margin 多数通过 | 1/5；四 seed 的少数 EE max >10 mm |

### 7.3 “让模型表现参与数据质量”已有的客观证据

| 对照 | 数据几何 | 标签变化 | 模型观察 | 能支持的最窄结论 |
|---|---|---|---|---|
| V6 vs Gate-v2 | 都有 100 mm 完整数据链 | V6 标签贴 beta3/4 边界；Gate-v2 有 ≥1.8° 全局最小 margin | V6 formal fail；Gate-v2 formal pass | 关节余量是静态无约束模型合法输出的重要可观测属性之一 |
| Gate-v2 formal vs κ=150 candidate | 90k XYZ 相同；81k beta 相同 | 仅 100 mm 的 9k beta 不同 | Cartesian 范围相同，candidate beta P95 更低 | 模型误差能作为同一可行几何上的标签可学习性诊断信号 |
| Gate-v2 centerline vs tube diagnostic | 同一训练模型 | 评估域从中心线扩到 ±5 mm tube | centerline 5/5；tube 1/5，差在 EE max tail | 中心线模型 gate 不能覆盖邻域最坏点质量 |

这些对照没有回答模型指标应获得多大正式权重，也没有证明应放宽哪些物理 gate；这正是本次请 GPT-5 Pro 判断的事项。

## 8. 经用户确认的问题

1. 综合 V4、V6、V7、Gate-v2、外层 tube、κ sweep、正式/候选标签与模型结果，目前主要影响数据集质量和模型可拟合半径的瓶颈分别是什么？请区分已经被证据证明的限制、合理但尚未验证的解释，以及纯粹未知项。
2. 是否应该放宽或重构当前 gate，使模型的实际拟合、FK 轨迹和邻域表现参与验证数据集质量？如果应该，物理不可达/硬边界类 gate、路径与标签质量 gate、数据 support/split gate、中心线模型 gate、tube 模型诊断应怎样分层，哪些可以作为软排序或证据，哪些仍必须 fail-closed？
3. 在不把探索性或事后结果冒充正式结论的前提下，有什么方法可以实质提升标签/数据集质量，从而提高模型能够拟合的椭圆尺度？现有“单 canonical branch + 静态 xyz->beta6 MLP + 固定 family + tube gate”是否需要被替换或扩展？
4. 面向用户理想的 1 m 目标，考虑当前机器人最大伸展 1.215498 m、固定 family 的必要上界 R≈103.147 mm，以及 R 与实际主半轴并不相同，应如何重新定义目标并重构整体方法？请判断这是否需要改变机器人/基座/轨迹几何、任务参数化、inverse representation、数据生成与模型验证范式，而不是继续在 100 mm 附近做同类微调。

## 9. 经用户确认的回答要求

请给出有证据链的明确判断，而不只是罗列可能性；并给出具体、可执行的整体方法重构与实验计划，说明各阶段要改变什么、如何验证是否有效，以及如何逐步逼近重新定义后物理可实现的 1 m 目标。若现有证据不足以作出某项决定，请明确指出未知项和所需的最小补证实验。正式事实、诊断/探索证据和推断必须分开，不得用模型表现覆盖物理不可达或数据泄漏等硬约束。

## 10. 附件清单与复现信息

GPT-5 Pro 实际看到的是“上传文件名”列；原始 SSH 路径只用于 provenance。

| 上传文件名 | 原始 SSH 路径 | 大小 | 证据等级 | 支持事实 | 上传理由 |
|---|---|---:|---|---|---|
| `robot_config_standard_100k.yaml` | `/mnt/ML_projects/quasi_exp/configs/robot_rods_only_standard_100k.yaml` | 1,619 B | formal config | F1、joint bounds、theta sign | 核验当前机器人离散结构与标准 beta 域 |
| `robot_lengths.csv` | `/mnt/ML_projects/quasi_exp/data/robot/generated/lengths.csv` | 285 B | source data | F1、F2 | 核验有效链长 1.160000 m |
| `robot_end_effector.csv` | `/mnt/ML_projects/quasi_exp/data/robot/generated/end_effector.csv` | 47 B | source data | F1、F2 | 核验末端局部偏移 0.055498 m |
| `v4_stable_radius_report.json` | `/mnt/ML_projects/quasi_exp/runs/true_ellipse_beta6_training_v4/04_radius_sweep/stable_radius_report.json` | 1,775 B | historical | F4 | 给出 V4 strict/relaxed/model-only 的直接报告 |
| `v6_audit_report.json` | `/mnt/ML_projects/quasi_exp/runs/true_ellipse_radial_bundle_v6/00_audit/audit_report.json` | 5,918 B | historical（V6 当轮 formal audit） | F2、F3、固定 family 参数 | 核验中心、相位和 V6 输入 lineage |
| `v6_final_training_report.json` | `/mnt/ML_projects/quasi_exp/runs/true_ellipse_radial_bundle_training_v6/03_final_models/final_training_report.json` | 1,854 B | historical（V6 当轮 formal） | F6 | 直接核验 V6 的 1/5、0/5 与 formal fail |
| `v6_checkpoint.md` | `/mnt/ML_projects/quasi_exp/.worktrees/true-ellipse-v6-100mm/docs/checkpoints/2026-07-15-true-ellipse-radial-bundle-v6.md` | 8,019 B | historical checkpoint | F5–F7 | 补充 V6 geometry/tube/data/model 边界与 bound-failure 定位 |
| `v7_original_summary_report.json` | `/mnt/ML_projects/quasi_exp/runs/true_ellipse_standard_domain_v7/04_summary/summary_report.json` | 3,722 B | formal | F8 | 核验原始 V7 strict radial 102.5 和下游阻断状态 |
| `v7_conditioning_frontier.png` | `/mnt/ML_projects/quasi_exp/runs/true_ellipse_standard_domain_v7/05_visualization/conditioning_frontier.png` | 315,829 B | visualization-only / diagnostic | F8、conditioning frontier | 展示半径扩张与 conditioning 前沿，不替代数值报告 |
| `v7_gate_v2_checkpoint.md` | `/mnt/ML_projects/quasi_exp/.worktrees/true-ellipse-v7-standard-domain-120mm/docs/checkpoints/2026-07-17-true-ellipse-standard-domain-v7-gate-v2.md` | 15,515 B | formal/evidence-only 汇总，dirty worktree | F9–F19 | 最紧凑地闭合 Gate-v2、outer tube、model、candidate lineage 与证据边界 |
| `gate_v2_formal_tube_report.json` | `/mnt/ML_projects/quasi_exp/runs/true_ellipse_standard_domain_v7_gate_v2_99pct/02_tube/tube_report.json` | 5,948 B | formal replay | F9、formal tube frontier | 核验 0.99 gate-only replay 的协议、准入与正式前沿 |
| `gate_v2_formal_tube_summary.csv` | `/mnt/ML_projects/quasi_exp/runs/true_ellipse_standard_domain_v7_gate_v2_99pct/02_tube/tube_radius_summary.csv` | 4,741 B | formal replay + failed outer evidence | F9、F10 | 直接给出 75–101.25 mm 的逐半径 tube 数值与 SHA |
| `gate_v2_candidate_tube_summary.csv` | `/mnt/ML_projects/quasi_exp/runs/true_ellipse_standard_domain_gate_sweep_v7/02_downstream/r102p50_k150/02_tube/tube_radius_summary.csv` | 4,197 B | evidence-only | F10、F11 | 直接给出完成联合修正后的 101.25/102.5 mm 失败数值 |
| `gate_v2_dataset_report.json` | `/mnt/ML_projects/quasi_exp/runs/true_ellipse_standard_domain_v7_gate_v2_99pct/03_dataset/dataset_report.json` | 43,466 B | formal | F13 | 核验 90k、split、conflict、margin、support 与 challenge |
| `true_ellipse_standard_domain_tubes_v7.parquet` | `/mnt/ML_projects/quasi_exp/runs/true_ellipse_standard_domain_v7_gate_v2_99pct/03_dataset/true_ellipse_standard_domain_tubes_v7.parquet` | 43,569,908 B | formal raw dataset | F13；逐行复核与新派生统计 | 提供完整 90,000 行原始数据，使 GPT-5 Pro 在运行环境支持 Parquet 时可独立检查分布、多值性与异常样本 |
| `gate_v2_final_training_report.json` | `/mnt/ML_projects/quasi_exp/runs/true_ellipse_standard_domain_training_v7_gate_v2_99pct/03_final_models/final_training_report.json` | 14,291 B | formal + diagnostic tube section | F14、F15 | 核验锁定模型、五 seed 中心线正式结果和 tube diagnostic |
| `gate_v2_kappa_sweep_report.json` | `/mnt/ML_projects/quasi_exp/runs/true_ellipse_standard_domain_gate_sweep_v7/04_summary/gate_sweep_report.json` | 48,444 B | evidence-only | F16、F17 | 核验五个 κ 候选、105 mm 失败和不可注册原因 |
| `gate_v2_seed_20260713.png` | `/mnt/ML_projects/quasi_exp/runs/true_ellipse_standard_domain_training_v7_gate_v2_99pct/04_visualization/seed_trajectories/seed_20260713.png` | 958,094 B | visualization-only | F14、F15 | 代表性展示 validation/test 的倾斜 3D、主平面投影和逐轴误差 |

### 完整原始数据与仍未上传的派生物说明

完整正式 Parquet 已按用户要求作为 `true_ellipse_standard_domain_tubes_v7.parquet` 纳入本交接包：

```text
source: /mnt/ML_projects/quasi_exp/runs/true_ellipse_standard_domain_v7_gate_v2_99pct/03_dataset/true_ellipse_standard_domain_tubes_v7.parquet
rows:   90,000
bytes:  43,569,908
sha256: 997260c4b9798a3f492a4ba994e6afcd8fca1a68c6a884ad22253221dfb655dc
```

本交接包没有另行抽样、转码或生成替代数据；该文件就是 Gate-v2 正式数据报告所对应的完整 90,000 行输入。因此，在运行环境能够读取 Parquet 时，GPT-5 Pro 可以在报告级核验之外重新计算任意必要的逐行分布、局部多值性、joint-margin、support 或异常样本统计。若运行环境不能解析 Parquet，应明确说明这一限制，并只使用随附的正式报告和 checkpoint，不得假定已经完成逐行复核。

模型 `joblib`、全部 prediction Parquet、其余四张重复 seed 图、完整 radial Parquet 与整个 `runs/` 树仍未上传。它们的 SHA 与路径由 checkpoint/报告记录；省略它们不授权对未展示的逐点预测、其他 seed 图像或径向中间状态作新结论。

## 11. 给下一位讨论者的最短事实摘要

1. 当前正式结论不是“没有模型”：Gate-v2 的 90k 单-family 数据和 `xyz->beta6` 静态 MLP 已在 92.5/100 mm 四组中心线评估中五 seed 全通过，正式可拟合尺度为 100 mm；整 tube 模型仍有 `EE max` 尾部失败。
2. 101.25/102.5 mm tube 同时失败 success、residual max、local-beta-RMS 三个未修改 gate；κ 放宽到 400 也没有把完整径向前沿推过 102.5 mm。正式/候选 100 mm 标签差异又表明标签 branch 的可学习性需要单独审视。
3. 当前 fixed family 有确定性物理边界：最大伸展 1.215498 m，`x(π/2)=cx+R` 导出零残差必要条件 `R<=103.147 mm`。当前 `R` 也不等于实际主半轴；保持机器人、中心和 family 时，无论把 1 m 理解为 R 或实际长半轴都不可达。
4. 用户已确认的问题是：判断数据质量与拟合半径的主要瓶颈，判断是否以及如何让模型实际表现参与分层 gate，提出提升数据质量的方法，并面向重新定义后物理可实现的 1 m 目标给出整体方法重构和可执行实验计划。
---
# 下一步总方案：并行验证“教师标签、轨迹几何、学生表示、混合求解、跨 family 泛化”五条路线

## 一、先给出当前最重要的判断

下一步不适合继续单独扩大数据量，也不适合只调 MLP。现有证据已经把问题分成了四个彼此独立的层级：

1. **固定 family 的物理边界。** 当前机器人最大伸展为 (1.215498\text{m})，固定 family 的零残差必要上界约为 (R\le103.147\text{mm})。因此，继续在同一中心和 family 上追求远大于 (103\text{mm}) 的 (R)，不是数据或模型能够解决的问题。
2. **教师标签曲面在外层发生变化。** 正式标签和候选标签使用完全相同的 (90{,}000) 个目标 (XYZ)，仅 (100\text{mm}) 的 (9{,}000) 行标签不同，但候选标签使模型的 test beta p95 从约 (0.26^\circ\sim0.28^\circ) 降到 (0.09^\circ\sim0.14^\circ)。这说明“物理可行、局部连续”的标签仍可能存在明显的可学习性差异。
3. **中心线通过不等于 tube 通过。** 当前静态 MLP 在 (100\text{mm}) 中心线四组评价中全部 (5/5) 通过，但 (100\text{mm}) tube 只有 (1/5) seed 通过，四个失败 seed 都是少量点的 EE max 超过 (10\text{mm})。
4. **单一 family 不能支持“全工作空间 IK”的结论。** 当前数据只有一个 family，虽然在该 family 内没有 voxel conflict，但这不能证明多个 family 重叠区域中的静态 (XYZ\to\beta_6) 仍然单值。

我另外直接复核了上传的 (90{,}000) 行 Parquet，发现了几个对下一步非常关键的事实：

* (100\text{mm}) tube 的 (9{,}000) 个目标中，有 **108 个目标的欧氏距离已经超过理论最大伸展 (1.215498\text{m})**；
* 有 62 行 `tube_success=false`，其中 **60 行位于 (\delta n_2=-5\text{mm})**；
* 失败集中于约 (55^\circ\sim97^\circ)，最坏点位于约 (84^\circ)、((\delta n_1,\delta n_2)=(+5,-5)\text{mm})；
* (97.5\to100\text{mm}) 的相邻半径标签变化突然增大，(\beta) RMS 差异 p95 约为 (0.481^\circ)、max 约为 (1.234^\circ)，而此前大多数相邻半径步的 p95 只有约 (0.13^\circ\sim0.27^\circ)；
* (100\text{mm}) 的 (\kappa) p95 约为 (49.5)，高于 (97.5\text{mm}) 的约 (35.4)，同时 (\sigma_3) p05 从约 (0.178) 降到 (0.127)。

因此，**当前真正的主矛盾不是总体样本数量，而是靠近最大伸展边界时，tube 中存在不可达点、局部条件恶化和标签 branch/lineage 转换。**

---

# 二、实验总体设计原则

我建议建立一个统一的并行实验项目：

```text
Canonical Teacher–Student Parallel Study V9
```

总目录：

```text
runs/canonical_teacher_student_parallel_v9/
  00_common_protocol/
  01_track_A_teacher_labels/
  02_track_B_tube_geometry/
  03_track_C_student_models/
  04_track_D_hybrid_solver/
  05_track_E_multi_family_atlas/
  06_track_F_one_meter_feasibility/
  07_combined_winner/
  08_formal_evaluation/
  09_paper_summary/
```

这六条路线不是互相替代，而是分别回答六个不同问题：

| 路线 | 要回答的问题                              |
| -- | ----------------------------------- |
| A  | 标签选得不好，是否是当前模型误差的主因？                |
| B  | tube 几何越过物理边界，是否是尾部失败主因？            |
| C  | 静态 (XYZ\to\beta) 表示是否不够？            |
| D  | 学习模型加少量 Jacobian 修正，能否兼顾速度与精度？      |
| E  | 单一 family 之外，是否必须使用 atlas / expert？ |
| F  | 1 m 目标需要怎样的任务或机器人几何重构？              |

---

# 三、重新分层 Gate：模型表现应参与数据质量，但不能覆盖物理硬约束

## 3.1 第一层：物理与几何硬 Gate

以下条件必须 fail-closed，模型效果再好也不能覆盖：

1. 目标点在允许的物理工作空间中；
2. joint bounds 不越界；
3. teacher FK residual 满足约束；
4. 不存在数据泄漏；
5. 轨迹几何定义正确；
6. tube claim 中的所有目标必须满足预先定义的可达性策略。

对于“完整对称 (\pm5\text{mm}) tube”正式结论，建议加入：

$$
|x_{\mathrm{target}}|\le L_{\max}-m_{\mathrm{safe}}
$$

其中：

$$
m_{\mathrm{safe}}=0.5\sim1.0\text{mm}
$$

如果 target 已超出理论最大伸展，就不能把它作为“精确可达 tube target”。

---

## 3.2 第二层：教师标签硬 Gate

用于静态单值模型的数据，应满足：

$$
\text{voxel conflict ratio}=0
$$

$$
\text{local beta RMS p95}\le1^\circ
$$

$$
\text{joint margin min}\ge1.5^\circ
$$

$$
\text{teacher repeatability p95}\le0.1^\circ\sim0.2^\circ
$$

以及沿 phase、radius 和 tube normal 的标签连续性。

---

## 3.3 第三层：数据 support 和 split Gate

继续保持：

* trajectory leakage (=0)；
* radius leakage (=0)；
* 完整轨迹/完整半径隔离；
* validation/test 只能在 protocol 锁定后读取。

---

## 3.4 第四层：模型可学习性 Gate

**模型表现可以参与 physically valid 数据集之间的排序，但不能用来接受物理无效数据。**

建议采用两级制度：

### Data validity

仅由物理、标签、support、split 决定。

### Data learnability

在所有 validity 通过的数据版本中，用模型验证结果排序。

排序采用 lexicographic rule，而不是随意加权：

1. 先比较 validation tube 通过 seed 数；
2. 再比较 EE max 的中位数；
3. 再比较 EE p95；
4. 再比较 beta p95；
5. 最后比较标签一阶、二阶曲率。

测试集绝不能参与标签 lineage 的选择。

---

## 3.5 第五层：Claim-specific 模型 Gate

必须把结论拆开：

* **中心线 claim**：中心线模型正式 gate；
* **tube claim**：tube 模型必须成为正式 gate，而不再只是 diagnostic；
* **跨 family claim**：必须有完整 family holdout；
* **轨迹控制 claim**：必须做 closed-loop rollout。

---

# 四、共同基础实验：所有路线开始前必须完成

## Phase 0.1 冻结当前 fixed point

当前 Gate-v2 位于 dirty worktree。Codex 首先要：

1. 保存完整 diff；
2. 将未跟踪脚本纳入 commit 或生成不可变 patch；
3. 记录代码 SHA、配置 SHA、数据 SHA；
4. 当前 (90{,}000) 行数据只读，不得覆盖。

输出：

```text
00_common_protocol/code_manifest.json
00_common_protocol/artifact_manifest.json
00_common_protocol/protocol_v9.yaml
```

---

## Phase 0.2 统一基线报告

统一重新计算：

* reachability margin；
* residual；
* (\sigma_3,\kappa)；
* phase/radius/normal beta 一阶、二阶差分；
* physically unreachable target count；
* joint margin；
* voxel conflict；
* 模型各 seed 的 per-angle/per-offset error。

输出：

```text
00_common_protocol/baseline_recomputed.json
00_common_protocol/hard_sector_points.parquet
00_common_protocol/radial_label_jump.csv
```

`hard_sector_points.parquet` 应至少保存：

```text
radius_mm
angle_idx
delta_n1_mm
delta_n2_mm
reach_margin_mm
teacher_residual_mm
sigma3_m
kappa
tube_success
```

---

## Phase 0.3 重新定义开发集和正式测试集

由于当前 (100\text{mm}) 已经被多轮实验读取，不再把它当作全项目 virgin test。

建议：

* 当前 family 的 (75\sim100\text{mm})：development / diagnostic；
* 后续新生成的 family F1、F2：训练；
* F3：validation；
* F4：正式 virgin test。

所有并行路线先在 development 上比较，锁定方案后才打开 F4。

---

# 五、并行路线 A：教师标签曲面优化

## 目标

验证当前瓶颈是否主要来自 teacher label lineage，而不是模型本身。

Guided Policy Search 的核心思想就是由轨迹优化器产生教师样本，再通过监督学习训练学生，并在后续迭代中使教师生成的数据更适合学生学习。([Proceedings of Machine Learning Research][1])

## A0：当前正式标签

作为基线：

```text
label_lineage = formal_gate_v2
```

---

## A1：当前 κ=150 候选标签

使用 evidence-only 候选 lineage，但只作为平行实验，不改变已有正式结论。

---

## A2：纯物理平滑标签面

在相同 (XYZ) 上重新选择/优化 (\beta)，目标为：

$$
J_A=
\lambda_x\sum_i|F(\beta_i)-x_i|^2
+
\lambda_\phi\sum_i|D_\phi\beta_i|_W^2
+
\lambda_r\sum_i|D_r\beta_i|_W^2
$$

$$
+
\lambda_n\sum_i
\left(
|D_{n_1}\beta_i|*W^2+
|D*{n_2}\beta_i|_W^2
\right)
+
\lambda_2\sum_i|D^2\beta_i|_W^2
$$

$$
+
\lambda_m\sum_iP_{\mathrm{margin}}(\beta_i)
+
\lambda_\kappa\sum_iP_\kappa(\beta_i)
$$

其中：

$$
W=\operatorname{diag}(4,4,2,2,1,1)
$$

保留 distal-priority 软偏好。

硬约束：

* FK residual；
* joint bounds；
* minimum margin；
* tube success；
* no voxel conflict。

---

## A3：模型感知标签面

在 A2 基础上加入学生一致项：

$$
J_{A3}=J_A+
\lambda_\pi
\sum_{i\in\mathcal D_{\mathrm{train/dev}}}
|\beta_i-\hat\beta_\pi(x_i)|^2
$$

执行 2–3 轮：

1. teacher 生成/选择标签；
2. 训练临时 MLP；
3. 根据 validation 误差修正标签候选选择；
4. 锁定后停止。

这一步只能使用 train/validation，不能访问正式 test。

---

## A4：离散候选图 + 连续修正

参考 IKLink“每个 waypoint 保留多组 IK 候选，再用动态规划连接”的思路：先从每个 surface node 的候选集合中进行全局图选择，再做连续 surface correction。([arXiv][2])

---

## A 路线对照矩阵

| ID | 标签来源                                    | 是否使用模型反馈 |
| -- | --------------------------------------- | -------- |
| A0 | 当前 formal                               | 否        |
| A1 | 当前 candidate                            | 否        |
| A2 | 物理平滑 surface                            | 否        |
| A3 | 物理平滑 + student agreement                | 是        |
| A4 | candidate graph + continuous correction | 否        |

---

## A 路线验收

所有版本必须使用相同目标 (XYZ)。

硬 gate 不得比 A0 退化。

额外目标：

$$
\text{radial beta RMS p95}_{97.5\to100}
\le0.25^\circ
$$

当前约为：

$$
0.481^\circ
$$

目标 max：

$$
\le0.75^\circ
$$

当前约为：

$$
1.234^\circ
$$

模型 validation 目标：

* beta p95 相比 A0 至少降低 (25%)；
* FK EE p95 不恶化；
* tube EE max 中位数下降至少 (20%)。

---

# 六、并行路线 B：重新设计物理可达的 tube 和 family

## 目标

验证 tube 尾部是否主要是因为 uniform (\pm5\text{mm}) tube 穿过了物理工作空间边界。

当前 (100\text{mm}) 中心线距离最大伸展边界只剩约 (3.05\text{mm})，但 tube 向外扩展到 (5\text{mm})，因此完整对称 tube 必然包含不可达点。

---

## B0：现有 (97.5\text{mm}) 对称 tube

作为 physically feasible baseline。

---

## B1：现有 (100\text{mm}) 对称 tube

作为 negative control，不改变 gate。

---

## B2：中心内移的 (100\text{mm}) 对称 tube

保持 (R=100\text{mm})，搜索：

$$
\Delta c_x\in{-2.5,-5,-7.5,-10}\text{mm}
$$

并允许小范围调整：

$$
\Delta c_y,\Delta c_z\in[-10,10]\text{mm}
$$

目标是最大化：

$$
m_{\mathrm{tube}}
=================

\min_{t,\delta_1,\delta_2}
\left[
L_{\max}
--------

|p(t)+\delta_1n_1+\delta_2n_2|
\right]
$$

要求：

$$
m_{\mathrm{tube}}\ge1\text{mm}
$$

再运行完整 teacher/tube/data/model 流程。

---

## B3：phase/family 重搜索

搜索新的：

$$
(c_x,c_y,c_z,\phi_y,\phi_z)
$$

保持 (R=100\text{mm}) 和三个轴幅值比例不变。

优化目标同时考虑：

* 最小 reachability margin；
* centerline (\kappa)；
* tube (\sigma_3)；
* teacher surface smoothness。

---

## B4：reachability-aware 非对称 tube

仅作为“任务相关邻域”实验，不得声称完整对称 tube。

定义 tube：

$$
\mathcal T_{\mathrm{safe}}
==========================

{
x+\delta_1n_1+\delta_2n_2:
|x+\delta_1n_1+\delta_2n_2|
\le L_{\max}-m_{\mathrm{safe}}
}
$$

必须明确报告：

* 原始 25 offset 中保留比例；
* 各 angle 的法向厚度；
* 不得把裁剪后的 tube 称作完整 (\pm5\text{mm}) tube。

---

## B5：困难扇区主动加密

对当前已知困难区：

$$
\phi\in[50^\circ,110^\circ]
$$

重点加密：

$$
\delta n_2=-5\text{mm}
$$

特别是：

$$
\delta n_1\in[0,5]\text{mm}
$$

采样密度提高 (4\sim8) 倍，并生成多个 teacher candidates，再使用 A 路线的最佳标签策略选择 branch。

---

## B 路线验收

### 对称 tube 正式 gate

* 所有 target 物理可达；
* tube success (\ge0.99)；
* residual max (\le3\text{mm})；
* local beta RMS p95 (\le1^\circ)；
* 4/5 模型 seed：

$$
EE_{95}\le3\text{mm}
$$

$$
EE_{\max}\le10\text{mm}
$$

### 强目标

$$
EE_{\max}\le5\text{mm}
$$

---

# 七、并行路线 C：学生模型表示实验

## 目标

判断当前标签究竟是静态函数、上下文函数，还是有状态策略。

---

## C0：静态基线

$$
(x,y,z)\to\beta_6
$$

当前 MLP 配置保持不变。

---

## C1：物理损失静态 MLP

损失：

$$
\mathcal L=
\lambda_\beta|\hat\beta-\beta|^2
+
\lambda_{FK}|F(\hat\beta)-x|^2
+
\lambda_mP_{\mathrm{margin}}(\hat\beta)
$$

同时测试 Jacobian sensitivity weighting：

$$
\mathcal L_{\mathrm{sens}}
==========================

(F(\hat\beta)-x)^\top
W_x
(F(\hat\beta)-x)
$$

---

## C2：轨迹上下文静态模型

对于 inspection path，phase、radius、family 都是已知的，可以训练：

$$
(x,y,z,R,\sin\phi,\cos\phi,\mathrm{family\ embedding})
\to\beta_6
$$

这不是通用静态 IK claim，而是 task-conditioned inverse policy。

---

## C3：有状态 MLP

输入：

$$
(x_t,\Delta x_t,\beta_{t-1})
$$

输出：

$$
\Delta\beta_t
$$

更新：

$$
\hat\beta_t=\hat\beta_{t-1}+\Delta\hat\beta_t
$$

---

## C4：GRU / LSTM

序列长度：

$$
16,\ 32
$$

输入每步：

$$
[x_t,\Delta x_t,\beta_{t-1}]
$$

输出：

$$
\Delta\beta_t
$$

必须做 closed-loop rollout，不能只做 teacher forcing。

---

## C5：causal Transformer

建议小型结构：

* (d_{\mathrm{model}}=128)；
* 4 heads；
* 2 layers；
* sequence length 32；
* 输出 (\Delta\beta_t)。

---

## C6：局部 Mixture-of-Experts

按 phase 和 conditioning 划为 6–8 个有重叠 chart。

比较：

1. hard chart gating；
2. soft MoE；
3. previous-chart-conditioned gating。

重叠区要求：

$$
|\beta^{(k)}(x)-\beta^{(k+1)}(x)|_{\mathrm{RMS}}
\le0.5^\circ\sim1^\circ
$$

---

## C 路线评价

### 静态模型

* centerline；
* tube；
* radius holdout；
* family holdout。

### 有状态模型

* closed-loop EE p95/max；
* rollout drift；
* seam；
* (\Delta\beta)、(\Delta^2\beta)；
* 不同初始 (\beta_0) 的 branch 稳定性。

### 判断规则

* 若 C0/C1 与 C3–C5 接近，则教师近似路径无关；
* 若 stateful 显著优于 static，则最终任务应写成：

$$
(x_t,\beta_{t-1})\to\beta_t
$$

* 若 C6 明显优于单模型，则需要 canonical atlas，而不是单一全局逆函数。

---

# 八、并行路线 D：学习模型 + 少量 Jacobian 修正

## 目标

验证是否可以采用 Fang 类型的混合方法：神经网络提供快速初值，Jacobian 迭代完成高精度修正。Fang 等学习 FK 和 Jacobian 后使用 Jacobian 迭代求解 IK，核心目的也是把昂贵模型计算转移到离线阶段，同时保留迭代求解的 branch 连续性。([arXiv][3])

---

## D0：完整慢速教师

作为精度和时间基线。

---

## D1：纯 MLP

当前模型。

---

## D2：MLP + 1 次 DLS

$$
\beta^{(1)}
===========

\Pi_{\mathcal B}
\left[
\hat\beta+
J_W^#(x-F(\hat\beta))
\right]
$$

---

## D3：MLP + 2 次 DLS

---

## D4：MLP + 4 次 DLS

---

## D5：有状态 MLP + 1–2 次 DLS

DLS 使用：

$$
J_W^#
=====

W^{-1}J^\top
\left(
JW^{-1}J^\top+\mu^2I
\right)^{-1}
$$

并在零空间中加入：

$$
-\alpha(I-J_W^#J)\nabla C_{\mathrm{posture}}
$$

---

## D 路线评价

记录：

* EE p95/max；
* beta 改变量；
* bound violations；
* inference time；
* teacher time；
* speedup；
* 失败率和 fallback 比例。

成功标准：

* (100\text{mm}) tube 4/5 seed 通过；
* EE max (\le5\sim10\text{mm})；
* 相比完整 teacher 至少快 (10\times)；
* 在线平均时间满足论文实时性目标。

这条路线即便不是最终纯 MLP，也非常符合“慢速物理教师离线生成数据，快速学习模型在线推理，必要时少量修正”的论文逻辑。

---

# 九、并行路线 E：Multi-family canonical atlas

## 目标

验证“全工作空间静态 IK”是否需要 family/chart 条件，而不是把单一 family 的结果外推成全局结论。

---

## E1：生成 4 个 family

使用 B 路线搜索出的 physically safe family：

* F1、F2：训练；
* F3：validation；
* F4：virgin test。

每个 family 先生成：

$$
R\in{85,92.5,100}\text{mm}
$$

Pilot 使用：

* 180 phase；
* (3\times3) tube offsets；

正式使用：

* 360 phase；
* (5\times5) offsets。

---

## E2：跨 family 冲突审计

对不同 family 中满足：

$$
|x_i-x_j|\le2\text{mm}
$$

的样本统计：

$$
d_\beta(i,j)
$$

### 若 p95 很小

可以合并训练静态模型。

### 若出现少数稳定 cluster

训练：

$$
xyz\to\text{family/chart id}
$$

再训练 expert。

### 若同一 (xyz) 的标签取决于前一状态

使用 stateful student。

---

## E3：模型比较

1. 单一 global static MLP；
2. family-conditioned MLP；
3. chart classifier + expert；
4. stateful GRU；
5. MoE。

正式 test 必须是完整 F4，不允许在任何调参阶段读取。

---

## E 路线成功标准

F4 holdout：

$$
EE_{95}\le5\text{mm}
$$

$$
EE_{\max}\le10\text{mm}
$$

并且 4/5 seeds 通过。

只有这一条路线通过后，才能把论文表述从“固定 family 的 task-relevant inverse”扩展为“多个任务区域上的 canonical atlas”。

---

# 十、并行路线 F：1 m 目标的独立可行性研究

当前机器人和 family 下的 (1\text{m}) 目标不可通过数据或模型实现。因此必须单独开一条工程可行性路线。

## F0：先定义“1 m”

必须分别分析：

1. (R=1\text{m})；
2. 实际长半轴 (=1\text{m})；
3. 直径 (=1\text{m})；
4. 轨迹总宽度 (=1\text{m})。

---

## F1：中心和 phase 全局优化

在不改变机器人长度时，搜索最有利的中心和平面方向。

判断哪一种 1 m 定义可能在现有链长内可达。

---

## F2：机器人尺度设计

如果当前几何不可达，求最小所需：

* 总链长；
* 单段长度；
* 基座位置；
* center；
* joint domain。

目标：

$$
L_{\mathrm{required}}
\ge
\max_t|p(t)|+m_{\mathrm{tube}}
$$

---

## F3：无量纲目标

建议论文和实验使用：

$$
\bar R=\frac{R}{L_{\mathrm{robot}}}
$$

以及：

$$
\bar a_{\mathrm{major}}
=======================

\frac{a_{\mathrm{major}}}{L_{\mathrm{robot}}}
$$

这样可以把当前 (100\text{mm}) 结果与未来放大机器人进行公平比较。

F 路线不与当前 (90k) 数据混合，也不参与当前模型正式结论。

---

# 十一、实验执行顺序与并行关系

## Stage 0：公共冻结与审计

先完成 Phase 0。

---

## Stage 1：可同时执行

以下四条可并行：

```text
Track A：标签 lineage 与 surface 优化
Track B：tube/family 几何重构
Track C：静态、context、stateful、MoE 模型
Track D：MLP + Jacobian correction
```

其中 C 在 A0、A1 上可以立即开始；A2/A3 完成后再补训练。

---

## Stage 2：组合实验

选择：

* A 路线最优标签策略；
* B 路线最优 physically safe family/tube；
* C/D 路线最优学生形式。

组合生成：

```text
07_combined_winner/
```

只在 development/validation 上锁定方案。

---

## Stage 3：Multi-family atlas

使用组合赢家执行 E 路线，生成 F1–F4。

---

## Stage 4：正式评价

只有完成预注册和 artifact freeze 后，才能打开 F4 test。

---

## Stage 5：1 m feasibility

F 路线可以和以上实验独立并行，但不得混入当前结论。

---

# 十二、推荐的具体 Pilot 规模

## A 路线

* 5 个标签版本；
* 先用 radii (75\sim97.5\text{mm})；
* 每版本 3 seeds screen；
* top 2 再做 5 seeds。

## B 路线

* 20 个 center/phase 候选快速 reachability screen；
* top 4 生成完整 (100\text{mm}) tube；
* top 2 训练模型。

## C 路线

| 模型           | Pilot seeds |
| ------------ | ----------: |
| static MLP   |           3 |
| FK-loss MLP  |           3 |
| context MLP  |           3 |
| stateful MLP |           3 |
| GRU/LSTM     |           3 |
| Transformer  |           3 |
| MoE          |           3 |

top 3 再运行 5 seeds。

## D 路线

无需重新训练即可先评估：

```text
0 / 1 / 2 / 4 DLS steps
```

然后只对最佳步数重新训练或联合训练。

## E 路线

Pilot：

$$
4\text{ families}\times3\text{ radii}\times180\text{ phase}\times9\text{ offsets}
=================================================================================

17{,}496\text{ samples}
$$

通过后再扩展正式 (5\times5) tube。

---

# 十三、所有实验的统一成功门槛

## 物理与标签

* physically unreachable target count (=0)；
* tube success (\ge0.99)；
* residual max (\le3\text{mm})；
* local beta RMS p95 (\le1^\circ)；
* conflict voxel ratio (=0)；
* joint margin min (\ge1.5^\circ)。

## 中心线模型

* 4/5 seeds；
* EE p95 (\le3\text{mm})；
* EE max (\le5\text{mm})；
* beta p95 (\le0.3^\circ)；
* no bound violation。

## Tube 模型

* 4/5 seeds；
* EE p95 (\le3\text{mm})；
* EE max (\le10\text{mm})，强目标 (\le5\text{mm})；
* no fixed bias；
* no rollout drift；
* no bound violation。

## Multi-family

* F4 holdout 4/5 seeds；
* EE p95 (\le5\text{mm})；
* EE max (\le10\text{mm})。

## Hybrid

* 精度 gate 通过；
* 相比完整 teacher 至少 (10\times) speedup。

---

# 十四、最终决策规则

## 若 A 显著优于 A0

说明当前主要瓶颈是 teacher label lineage。
后续论文应把 canonical teacher 的定义作为方法核心。

## 若 B2/B3 显著优于当前 family

说明 tube 失败主要来自物理边界，而不是模型。
应调整轨迹中心/family，而不是放宽 gate。

## 若 C3–C5 显著优于静态模型

说明教师策略存在路径依赖。
最终模型应改为 stateful 或 trajectory-conditioned。

## 若 D 路线仅需 1–2 次迭代即可通过

优先采用：

$$
\text{MLP initializer}+\text{few-step Jacobian correction}
$$

它最符合 Fang 类型的高效混合求解路线。

## 若 E 中 global MLP 失败而 expert/stateful 通过

说明不存在适合整个任务区域的单一静态 canonical inverse，应使用 atlas。

## 若所有路线在 (100\text{mm}) 附近都受物理边界限制

停止继续扩大当前 fixed geometry，转入 F 路线。

---

# 十五、Codex 最终交付物

每条 track 必须输出：

```text
experiment_manifest.json
protocol.yaml
dataset_report.json
model_report.json
per_seed_metrics.csv
per_angle_per_offset_metrics.parquet
failure_points.parquet
plots/
checkpoint.md
```

总汇总：

```text
09_paper_summary/
  parallel_experiment_scoreboard.csv
  hard_gate_summary.csv
  label_learnability_summary.csv
  model_comparison_summary.csv
  teacher_vs_student_timing.csv
  multi_family_generalization.csv
  one_meter_feasibility.md
  final_recommendation.md
```

`final_recommendation.md` 必须区分：

* formal facts；
* evidence-only results；
* diagnostic results；
* failed experiments；
* selected next formal protocol。

---

# 最终推荐

最值得优先并行启动的是：

1. **A2/A3：标签曲面平滑与模型感知 teacher；**
2. **B2/B3：重新搜索具有完整对称 tube margin 的 (100\text{mm}) family；**
3. **D2–D4：MLP 加 1–4 次 Jacobian 修正；**
4. **C3/C4/C6：stateful 与 local-expert 对照。**

其中，A 和 B 决定“数据是否正确”，C 决定“静态表示是否足够”，D 决定“工程上是否必须坚持纯一次前向推理”。

当前最有希望获得实质提升的组合不是单纯更大的 MLP，而是：

$$
\boxed{
\text{physically safe family}
+
\text{model-aware canonical teacher}
+
\text{structured/stateful student}
+
\text{optional few-step Jacobian correction}
}
$$

这套并行实验能够同时回答：是标签、轨迹几何、模型表示还是在线求解架构限制了当前结果，而不会再把所有失败笼统归因于“数据量不够”。

[1]: https://proceedings.mlr.press/v28/levine13.html?utm_source=chatgpt.com "Guided Policy Search"
[2]: https://arxiv.org/html/2402.16154v1?utm_source=chatgpt.com "End-Effector Trajectory Tracking with Minimal Reconfigurations"
[3]: https://arxiv.org/abs/2012.13965?utm_source=chatgpt.com "Efficient Jacobian-Based Inverse Kinematics with Sim-to-Real Transfer of Soft Robots by Learning"

---

你说得对。上一版我把“标签、tube、模型、混合求解”等拆成了多条相对独立的实验线，偏离了你真正想要的主线。

你想要的是一个**统一的教师—学生数据生成范式**：

$$
\boxed{
\text{准静态虚拟样机}
\rightarrow
\text{慢速 trajectory-level canonical teacher}
\rightarrow
\text{一致性审计}
\rightarrow
\text{快速学生模型}
}
$$

而你提到的 Fang、Bern、Thuruthel、Lee/Fang local model，并不是四套互相排斥的完整方案，而应当分别成为这条主流程中的不同模块：

* **Fang/Bern**：物理模型或可微正向模型作为环境；
* **Thuruthel**：整条轨迹优化器作为教师；
* **Fang**：上一时刻解 warm-start 和 continuation，保持 branch 连续；
* **Lee/Fang local models**：当一个全局逆映射不成立时，用多个局部 chart 表达同一任务区域。

下面给出我最推荐的完整实施方案。它可以直接作为 Codex 下一轮的主实验设计。

---

# 一、方法的总定义

建议将下一阶段命名为：

```text
Trajectory-Level Canonical Teacher Distillation V10
```

核心不再是从随机 FK 样本中直接训练：

$$
xyz\rightarrow\beta_6
$$

而是先定义一个慢速教师策略：

$$
\Pi_T:
\left(
x_{0:T}^{\star},
\beta_0,
\text{policy parameters}
\right)
\longrightarrow
\beta_{0:T}^{\star}
$$

其中：

* (x_{0:T}^{\star}) 是完整目标轨迹；
* (\beta_0) 是起始构型或 root anchor；
* (\beta_{0:T}^{\star}) 是教师生成的连续构型轨迹；
* 教师同时考虑 FK 精度、构型连续性、关节余量、条件数和 distal-priority。

再将教师行为蒸馏为学生：

### 若教师近似路径无关

$$
\pi_\phi(x)\approx\beta^\star
$$

### 若教师依赖上一状态

$$
\pi_\phi(x_t,\beta_{t-1})
\approx
\Delta\beta_t
$$

### 若教师需要多个局部 chart

$$
\pi_c(x)\rightarrow\text{chart id}
$$

$$
\pi_{\phi,k}(x)\rightarrow\beta
$$

也就是说，**学生模型形式由教师数据的一致性审计结果决定，而不是提前假设一定是静态 (xyz\to\beta_6)。**

---

# 二、为什么这是当前项目最合适的主路线

Fang 等没有直接训练冗余系统的一对多 IK，而是学习 FK 和 Jacobian，再通过 Jacobian 迭代求解；他们明确指出，冗余系统的 task-space-to-actuator-space 映射是一对多的，直接监督回归容易失败，并通过最小化驱动变化生成平滑运动。

Bern 等同样学习单值的可微正向模型：

$$
u\rightarrow y(u)
$$

然后通过梯度优化搜索控制输入。对连续轨迹，他们将上一目标点的解作为下一目标点的 warm-start，从而提高收敛性并保持局部 branch。([Computational Robotics Lab][1])

Thuruthel 等的工作进一步采用“模型 + 轨迹优化 + 监督学习策略”的结构：用循环网络表示正向动态模型，通过 trajectory optimization 生成控制行为，再监督训练闭环策略。([IEEE Xplore][2])

Lee/Fang 类型的局部学习方法则说明，当一个全局映射过于复杂或不唯一时，可以只在局部区域学习 inverse mapping，并在线更新与当前状态最相关的 local model。([HKU Scholars Hub][3])

你的项目当前已经部分实现了 continuation、radial predictor、cyclic cuts、candidate graph、whole-curve correction 和 tube surface correction；但这些目前更像“标签生成技巧”，还没有被统一定义成一个正式的 canonical teacher。当前数据也只来自一个 family，静态中心线模型在 (100)mm 上通过，但整个 tube 仍存在尾部失败，而且不同标签 lineage 在相同 (XYZ) 上呈现不同的模型可学习性。

因此，下一步应当做的不是再发明一个新的固定 (u)-公式，而是：

> 把现有 continuation、candidate graph、whole-curve optimization 和 local chart 统一为一个可重复、可审计的慢速教师。

---

# 三、整体系统架构

建议实现以下结构：

```text
Target trajectory library
        │
        ▼
Exact quasi-static virtual prototype F(beta), J(beta)
        │
        ├──────────────┐
        ▼              ▼
Candidate IK       Jacobian continuation
generation             │
        └──────┬───────┘
               ▼
       Cyclic branch linking
               ▼
     Whole-trajectory optimization
               ▼
        Local chart / atlas
               ▼
      Teacher trajectory dataset
               ▼
        Consistency audit
       ┌───────┼──────────┐
       ▼       ▼          ▼
 Static MLP  Stateful   Chart experts
 xyz→beta   policy      / local model
```

其中推荐的正式教师是：

$$
\boxed{
\text{多候选 IK}
+
\text{continuation}
+
\text{循环图连接}
+
\text{整轨迹联合优化}
+
\text{局部 chart tube 扩展}
}
$$

它是 Fang、Bern、Thuruthel 和 local-model 思想的组合，而不是其中任何一个方法的简单复现。

---

# 四、教师的输入、输出和确定性定义

## 4.1 教师输入

每次教师运行输入一整条目标轨迹：

$$
X^\star=
{x_0^\star,\ldots,x_{T-1}^\star}
$$

以及：

```text
trajectory_id
family_id
radius
center
phase/orientation
tube offset
root_configuration
teacher_policy_id
solver_seed
```

对于闭合椭圆：

$$
x_0^\star=x_T^\star
$$

## 4.2 教师输出

教师输出：

$$
B^\star=
{\beta_0^\star,\ldots,\beta_{T-1}^\star}
$$

同时保存：

```text
theta_1 ... theta_30
FK achieved xyz
FK residual
Jacobian singular values
condition number
joint margin
posture cost
first-order smoothness
second-order smoothness
chart id
branch id
teacher convergence state
```

## 4.3 教师必须是确定性的

在相同的：

* 目标轨迹；
* root configuration；
* policy 权重；
* 代码版本；
* seed；

下，应得到近似相同结果：

$$
\operatorname{p95}
\left(
|\beta_t^{(1)}-\beta_t^{(2)}|_{\mathrm{RMS}}
\right)
\le0.1^\circ\sim0.2^\circ
$$

否则学生训练的不是一个明确策略。

---

# 五、模块 1：准静态虚拟样机作为正向环境

## 5.1 正式环境

使用论文前半部分建立的准静态运动学—动力学环境作为标签的最终判定器：

$$
F:\beta_6\rightarrow x
$$

并计算：

$$
J(\beta)
========

\frac{\partial F}{\partial\beta}
\in\mathbb R^{3\times6}
$$

正式标签必须由精确虚拟样机验证。

## 5.2 可选快速正向代理

可以并行训练：

$$
\hat F(\beta)\approx F(\beta)
$$

以及：

$$
\hat J(\beta)\approx J(\beta)
$$

其作用仅限于：

* 生成 IK 初值；
* 快速筛掉明显失败 candidate；
* 加速 trajectory optimization 的前几轮；
* 扩大教师搜索规模。

最终每个标签仍必须回到精确环境中校正和验证：

$$
|F(\beta^\star)-x^\star|\le\epsilon
$$

这避免 learned forward model 的误差污染教师标签。

## 5.3 正向环境验证

Codex 先执行：

### 数值 Jacobian 校验

随机取 2000 个 (\beta)，比较：

$$
J(\beta)\delta\beta
$$

和：

$$
F(\beta+\delta\beta)-F(\beta)
$$

输出：

* relative Jacobian error；
* error vs perturbation scale；
* singular value distribution。

### IK synthetic recovery

随机采样：

$$
x_i=F(\beta_i)
$$

只给 (x_i) 求 IK，要求：

$$
\mathrm{residual}_{95}\le0.5\text{mm}
$$

这一步只验证环境和求解器，不要求恢复原始 (\beta_i)。

---

# 六、模块 2：trajectory-level canonical teacher

这是整个方案的核心。

## 6.1 轨迹级目标函数

对完整轨迹 (\beta_{0:T-1})，教师优化：

$$
J_{\mathrm{teacher}}
====================

\lambda_xJ_{\mathrm{track}}
+
\lambda_vJ_{\mathrm{vel}}
+
\lambda_aJ_{\mathrm{acc}}
+
\lambda_pJ_{\mathrm{posture}}
+
\lambda_mJ_{\mathrm{margin}}
+
\lambda_\kappa J_{\mathrm{cond}}
+
\lambda_cJ_{\mathrm{closure}}
$$

### 跟踪项

$$
J_{\mathrm{track}}
==================

\sum_{t=0}^{T-1}
\left|
F(\beta_t)-x_t^\star
\right|^2
$$

### 一阶连续项

$$
J_{\mathrm{vel}}
================

\sum_{t=0}^{T-1}
\left|
\beta_{t+1}-\beta_t
\right|_W^2
$$

闭合索引：

$$
\beta_T=\beta_0
$$

### 二阶连续项

$$
J_{\mathrm{acc}}
================

\sum_{t=0}^{T-1}
\left|
\beta_{t+1}-2\beta_t+\beta_{t-1}
\right|_W^2
$$

### distal-priority 构型偏好

$$
J_{\mathrm{posture}}
====================

\sum_t
\left[
w_1|\beta_{1:2,t}|^2
+
w_2|\beta_{3:4,t}|^2
+
w_3|\beta_{5:6,t}|^2
\right]
$$

其中：

$$
w_1>w_2>w_3>0
$$

初始建议：

$$
w_1:w_2:w_3=4:2:1
$$

注意不能使用：

$$
-|\beta_{5:6}|^2
$$

因为这会鼓励第三段走向边界。

### joint-margin barrier

对每个主动角：

$$
d_i(\beta)=
\min
\left(
\beta_i-\beta_i^{\min},
\beta_i^{\max}-\beta_i
\right)
$$

当：

$$
d_i<m_{\mathrm{safe}}
$$

时施加强惩罚。

建议：

$$
m_{\mathrm{safe}}=1.5^\circ\sim2^\circ
$$

### 条件数项

可以使用：

$$
J_{\mathrm{cond}}
=================

\sum_t
\psi\left(\kappa(J(\beta_t))\right)
$$

或：

$$
J_{\mathrm{cond}}
=================

\sum_t
\frac{1}{\sigma_{\min}(J(\beta_t))+\varepsilon}
$$

### 闭环项

$$
J_{\mathrm{closure}}
====================

|\beta_{T-1}-\beta_0|_W^2
$$

---

# 七、模块 3：continuation 教师求解器

仅用整轨迹优化通常容易依赖初值，因此必须结合 Fang/Bern 式 continuation。

## 7.1 Weighted damped least squares

定义：

$$
W=
\operatorname{diag}(4,4,2,2,1,1)
$$

加权阻尼伪逆：

$$
J_W^#
=====

W^{-1}J^\top
\left(
JW^{-1}J^\top+\mu^2I
\right)^{-1}
$$

任务误差：

$$
e_t=x_t^\star-F(\beta_t)
$$

基础更新：

$$
\Delta\beta_{\mathrm{task}}
===========================

J_W^#e_t
$$

## 7.2 零空间 canonical motion

零空间投影：

$$
N=I-J_W^#J
$$

次级更新：

$$
\Delta\beta_{\mathrm{null}}
===========================

-\alpha
N
\nabla C_{\mathrm{posture+margin+cond}}
$$

最终：

$$
\Delta\beta
===========

\Delta\beta_{\mathrm{task}}
+
\Delta\beta_{\mathrm{null}}
$$

## 7.3 初值 continuation

对下一 waypoint：

### previous-copy

$$
\beta_t^{(0)}=\beta_{t-1}^\star
$$

### secant predictor

$$
\beta_t^{(0)}
=============

\beta_{t-1}^\star
+
\left(
\beta_{t-1}^\star-\beta_{t-2}^\star
\right)
$$

### Jacobian predictor

$$
\beta_t^{(0)}
=============

\beta_{t-1}^\star
+
J_W^#
\left(
x_t^\star-x_{t-1}^\star
\right)
$$

三个预测器均保留，作为不同候选来源。

## 7.4 前向、反向和 cyclic cut

对闭合轨迹至少运行：

* 正方向 continuation；
* 反方向 continuation；
* cyclic cuts：(0^\circ,90^\circ,180^\circ,270^\circ)。

如果结果依赖起点或方向过强，说明教师还未定义稳定 branch。

---

# 八、模块 4：多候选图连接

每个 waypoint 不只保留一个 IK 解，而是保留：

$$
\mathcal C_t=
{\beta_{t,1},\ldots,\beta_{t,K}}
$$

建议 Pilot：

$$
K=16,\ 32,\ 64
$$

候选来源：

1. previous-copy continuation；
2. secant predictor；
3. Jacobian predictor；
4. full-beta multi-seed local IK；
5. null-space perturbation；
6. 当前 family / parent radius candidate；
7. 相邻 chart candidate。

## 8.1 候选聚类

在归一化 (\beta_6) 空间中聚类。

不同 cluster 阈值：

$$
d_{\beta,\mathrm{RMS}}\ge0.5^\circ\sim1^\circ
$$

## 8.2 图代价

节点代价：

$$
C_t(k)
======

\alpha_x
|F(\beta_{t,k})-x_t^\star|^2
+
\alpha_pC_{\mathrm{posture}}
+
\alpha_mC_{\mathrm{margin}}
+
\alpha_\kappa C_{\mathrm{cond}}
$$

边代价：

$$
E_t(k,l)
========

\lambda_v
|\beta_{t+1,l}-\beta_{t,k}|_W^2
$$

若采用二阶动态规划，还加入：

$$
\lambda_a
|\beta_{t+1,l}-2\beta_{t,k}+\beta_{t-1,j}|_W^2
$$

最后一点与第一点必须连闭环边。

## 8.3 不再只用硬 jump threshold

硬阈值可以作为剪枝：

$$
|\Delta\beta|_{\mathrm{RMS}}\le3^\circ\sim5^\circ
$$

但最终选择应依赖全局累计代价，而不是“超过 (3^\circ) 就全部删除”。

图连接输出的是整轨迹优化的初值，不是最终标签。

---

# 九、模块 5：整轨迹联合修正

图连接得到离散 branch 后，再对全部 (\beta_{0:T-1}) 做联合连续优化。

## 9.1 分阶段优化

### Stage A：先保证 tracking

使用较高 (\lambda_x)，较低平滑权重，把 residual 压入门槛。

### Stage B：提高一阶与二阶平滑

在 residual 约束下增大：

$$
\lambda_v,\lambda_a
$$

### Stage C：加入 posture、margin 和 conditioning

逐渐增大：

$$
\lambda_p,\lambda_m,\lambda_\kappa
$$

避免一开始就被次级项推离可达 branch。

## 9.2 推荐 Pilot 权重组

在变量和位置归一化后，测试：

| config | (\lambda_v) | (\lambda_a) | (\lambda_p) | (\lambda_\kappa) |
| ------ | ----------: | ----------: | ----------: | ---------------: |
| T1     |         0.1 |           0 |        0.01 |                0 |
| T2     |           1 |         0.1 |        0.05 |             0.01 |
| T3     |           5 |         0.5 |         0.1 |             0.05 |
| T4     |          10 |           1 |         0.2 |              0.1 |

joint-margin barrier 作为硬约束或高权重 barrier，不参与随意 sweep。

---

# 十、模块 6：局部 chart / atlas

局部 chart 不是另一个独立终点，而是教师在全局单 chart 无法稳定覆盖时的局部表达工具。

## 10.1 Chart 定义

第 (k) 张 chart 保存：

```text
chart_id
anchor_xyz
anchor_beta
Jacobian
weighted pseudoinverse
validity radius
condition range
neighbor charts
```

局部预测：

$$
\beta_{\mathrm{pred}}
=====================

\beta_c
+
J_{W,c}^#
(x-x_c)
$$

再由 exact simulator 做 corrector。

## 10.2 新建 chart 的条件

满足任一条件时新建：

1. 预测 residual (>1\sim2)mm；
2. (\kappa) 超过预设阈值；
3. 与 anchor 的 (\beta) 距离过大；
4. corrector 迭代次数明显增加；
5. 同一局部区域出现稳定的第二个 branch cluster。

## 10.3 Chart overlap gate

相邻 chart 在重叠区域应满足：

$$
|\beta^{(k)}(x)-\beta^{(k+1)}(x)|_{\mathrm{RMS}}
\le0.5^\circ
$$

如果重叠区域差异很大，它们不是同一 branch 的两张坐标图，而是两个不同 branch。

## 10.4 用 atlas 生成 tube

对中心线附近的法向扰动：

$$
x_{t,i,j}
=========

x_t^\star+
\delta_1n_{1,t}+
\delta_2n_{2,t}
$$

使用最近 chart predictor：

$$
\beta^{(0)}=
\beta_t^\star+
J_{W,t}^#
\left(
x_{t,i,j}-x_t^\star
\right)
$$

然后 exact corrector，并沿：

* phase；
* radius；
* (n_1)；
* (n_2)

四个方向做 continuation。

这一步正是 local-model 思路在数据生成阶段的实现。

---

# 十一、教师内部应并行比较的四个版本

这才是你原本希望的“几个路线并行”。

它们共享同一正向环境、同一目标轨迹和同一硬 gate。

| 教师 ID | 方法                                      | 对应借鉴                           |
| ----- | --------------------------------------- | ------------------------------ |
| T0    | 当前 V7 continuation + graph + correction | 当前项目基线                         |
| T1    | 纯 Jacobian/DLS continuation             | Fang/Bern                      |
| T2    | 全轨迹直接优化                                 | Thuruthel 式 trajectory teacher |
| T3    | 多候选图连接 + trajectory optimization        | **推荐主教师**                      |
| T4    | T3 + local chart atlas                  | Lee/Fang local model 扩展        |

## 推荐优先级

正式推荐：

$$
\boxed{T3}
$$

如果 T3 的 tube 或跨 family 区域仍不稳定，再升级到：

$$
\boxed{T4}
$$

T1、T2 主要用于机制消融：

* T1 检查 continuation 是否足够；
* T2 检查纯联合优化是否能消除 jump；
* T3 验证离散 branch 选择和连续优化组合；
* T4 验证全局单 chart 是否是瓶颈。

---

# 十二、教师数据质量 Gate

## 12.1 物理硬 Gate

任何模型表现都不能覆盖：

* target 物理不可达；
* joint bounds 越界；
* FK residual 超标；
* 轨迹几何错误；
* 数据泄漏。

## 12.2 中心线 Gate

建议：

$$
\mathrm{FK\ residual}_{95}\le1\text{mm}
$$

$$
\mathrm{FK\ residual}_{\max}\le3\text{mm}
$$

$$
\Delta\beta_{\mathrm{RMS,p95}}\le1^\circ
$$

$$
\Delta\beta_{\mathrm{RMS,max}}\le2^\circ
$$

$$
\Delta^2\beta_{\mathrm{RMS,p95}}\le0.25^\circ\sim0.5^\circ
$$

$$
\mathrm{seam}\le0.5^\circ\sim1^\circ
$$

$$
\mathrm{joint\ margin}_{\min}\ge1.5^\circ
$$

## 12.3 Tube Gate

$$
\mathrm{tube\ success}\ge0.99
$$

$$
\mathrm{residual}_{95}\le1.5\text{mm}
$$

$$
\mathrm{residual}_{\max}\le3\text{mm}
$$

$$
\mathrm{local\ beta\ RMS}_{95}\le1^\circ
$$

$$
\mathrm{multi\ branch\ ratio}=0
$$

## 12.4 重复性 Gate

相同 teacher config，不同运行：

$$
\mathrm{teacher\ repeatability}_{95}
\le0.1^\circ\sim0.2^\circ
$$

正向、反向 traversal：

$$
\mathrm{direction\ gap}_{95}\le0.2^\circ\sim0.5^\circ
$$

不同 cyclic cut：

$$
\mathrm{cut\ gap}_{95}\le0.2^\circ\sim0.5^\circ
$$

---

# 十三、一致性审计：决定学生到底应该是什么

教师数据生成完成后，先不训练正式学生，而做三类审计。

## 13.1 静态路径无关性审计

在不同轨迹、不同方向、不同 family 中找：

$$
|x_i-x_j|\le5\text{mm}
$$

统计：

$$
d_\beta(i,j)
============

|\beta_i-\beta_j|_{\mathrm{RMS}}
$$

### 若：

$$
d_{\beta,95}\le1^\circ
$$

说明可以近似训练：

$$
xyz\to\beta_6
$$

### 若出现少数稳定 cluster

说明应训练：

$$
xyz\to\mathrm{chart/branch\ id}
$$

再由 expert 输出 (\beta_6)。

### 若标签明显依赖 (\beta_{t-1})

说明教师定义的是有状态策略：

$$
(x_t,\beta_{t-1})\to\Delta\beta_t
$$

不能再强行训练静态 MLP。

---

## 13.2 局部 Jacobian 与 label curvature 审计

统计：

* (\sigma_{\min})；
* (\kappa)；
* (|\partial\beta/\partial x|)；
* phase/radius/normal 一阶差分；
* 二阶差分；
* chart overlap gap。

这决定学生模型是否需要局部专家。

---

## 13.3 Probe-student 可学习性审计

对于所有物理 hard gate 已通过的 teacher 版本，训练同一个小型 probe MLP。

Probe 只用于排序标签面，不用于覆盖物理 gate。

优先级：

1. validation tube 通过 seed 数；
2. EE max 中位数；
3. EE p95；
4. beta p95；
5. label curvature。

当前项目已经证明，相同 (XYZ) 的不同 label lineage 会产生不同的 beta 拟合误差，因此把 probe student 作为 teacher surface 的软排序指标是合理的。

---

# 十四、学生模型并行实验

通过一致性审计后，再并行训练以下学生。

## S0：静态 MLP

$$
xyz\to\beta_6
$$

## S1：静态 MLP + FK loss

$$
\mathcal L
==========

\lambda_\beta|\hat\beta-\beta^\star|^2
+
\lambda_{FK}|F(\hat\beta)-x^\star|^2
$$

## S2：有状态 MLP

$$
(x_t,\Delta x_t,\beta_{t-1})
\to
\Delta\beta_t
$$

## S3：GRU/LSTM

输入长度：

$$
16,\ 32
$$

每步输入：

$$
[x_t,\Delta x_t,\beta_{t-1}]
$$

## S4：Chart classifier + local expert

$$
xyz\to\mathrm{chart\ id}
$$

$$
(xyz,\mathrm{chart\ id})\to\beta_6
$$

## S5：Soft mixture-of-experts

用于 chart 边界平滑过渡。

模型选择不按复杂度决定，而按教师一致性审计结果决定。

---

# 十五、数据集结构

每一行至少保存：

```text
target_x
target_y
target_z

teacher_beta1 ... teacher_beta6
theta_1 ... theta_30

previous_beta1 ... previous_beta6
next_beta1 ... next_beta6

trajectory_id
family_id
radius
phase
tube_n1
tube_n2
chart_id
branch_id

teacher_fk_residual
teacher_posture_cost
teacher_velocity_cost
teacher_acceleration_cost
teacher_condition_cost

sigma1
sigma2
sigma3
kappa
joint_margin_min

root_configuration_id
teacher_policy_id
solver_seed
traversal_direction
cyclic_cut
```

即使最后训练静态 MLP，也不能删除 history、chart 和 branch 字段，因为它们用于审计数据是否真的静态单值。

---

# 十六、实验实施顺序

## Phase 0：代码与协议冻结

当前实现位于 dirty worktree，先冻结 commit、diff、配置和数据 SHA；否则 teacher 对照会混入代码变化。当前正式事实、数据规模和单-family 边界已经记录，但尚无多 family、历史状态和局部 chart 正式对照。

## Phase 1：正向环境验证

完成 FK/Jacobian/synthetic recovery。

## Phase 2：单 family 教师 Pilot

使用当前固定 family：

$$
R\in{75,85,92.5,97.5,100}\text{mm}
$$

先用：

* 180 phase；
* (3\times3) tube；

比较 T0–T4。

## Phase 3：教师版本筛选

先通过 hard gate，再用 probe student 排序。

选 top 2。

## Phase 4：正式单-family 教师数据

* 360 phase；
* (5\times5) tube；
* 十个半径；
* 目标规模约 90k。

## Phase 5：跨 trajectory 一致性审计

决定 static / stateful / chart-aware 学生。

## Phase 6：学生并行训练

S0–S5。

## Phase 7：多 family Pilot

建议：

* 4 个 family；
* 3 个半径；
* 180 phase；
* (3\times3) tube。

总量：

$$
4\times3\times180\times9
========================

19{,}440
$$

其中：

* F1、F2：train；
* F3：validation；
* F4：virgin test。

## Phase 8：正式多-family 数据

根据 Pilot 结果扩大到：

* 8–12 个 family；
* 5–8 个半径；
* 360 phase；
* (5\times5) tube。

---

# 十七、最终评价：教师与学生都要比较

## 教师评价

* FK residual；
* trajectory smoothness；
* tube success；
* joint margin；
* conditioning；
* repeatability；
* wall time。

## 学生评价

* teacher imitation beta error；
* FK EE p95/max；
* tube error；
* holdout family；
* holdout radius；
* closed-loop rollout；
* inference time。

## 核心速度指标

报告：

$$
\mathrm{speedup}
================

\frac{
t_{\mathrm{teacher}}
}{
t_{\mathrm{student}}
}
$$

并比较：

1. 完整慢速教师；
2. 纯学生一次前向；
3. 学生 + 1–2 次 Jacobian corrector。

---

# 十八、最小可执行 Pilot

建议 Codex 先只执行以下三步。

## Pilot A：教师求解器并行对照

对象：

```text
当前 100 mm fixed family
180 phase
3×3 tube
```

方法：

```text
T0 current
T1 Jacobian continuation
T2 whole-trajectory optimization
T3 graph + trajectory optimization
T4 T3 + local atlas
```

要求：

* 同一目标；
* 相同 bounds；
* 相同 hard gate；
* 相同 root candidates。

## Pilot B：一致性审计

比较 top 2 teacher：

* seed repeatability；
* forward/reverse；
* cyclic cut；
* label curvature；
* probe MLP。

## Pilot C：学生表示对照

只对最佳 teacher 训练：

```text
S0 static MLP
S2 stateful MLP
S4 chart expert
```

这三个模型足以判断下一步该走静态、状态还是 atlas。

---

# 十九、Codex 的主任务描述

可以直接将下面这段交给 Codex：

> Implement a unified slow-to-fast canonical inverse pipeline. Use the exact quasi-static virtual prototype as the authoritative forward environment. For each complete target trajectory, generate multiple IK candidates using weighted damped-Jacobian continuation, previous-state and secant warm starts, full-beta multi-seed search, and null-space exploration. Link candidates with a cyclic graph, then jointly optimize the complete beta trajectory for FK tracking, first- and second-order smoothness, distal-priority posture, joint margin, conditioning, and closed-loop consistency. When a single global mapping becomes unstable, construct overlapping local charts and use predictor-corrector continuation to generate tube samples. Audit repeatability, traversal invariance, cross-trajectory label consistency, branch conflicts, chart overlaps, and student learnability. Based on the audit, train and compare a static (xyz\to\beta_6) student, a stateful ((xyz,\beta_{\mathrm{prev}})\to\Delta\beta) student, and a chart-classifier-plus-expert student. Compare teacher accuracy and computation time against student FK accuracy, trajectory smoothness, tube performance, and online inference speed.

---

# 最终判断

你最开始提出的路线是正确的，而且应该成为整个项目的**主方法**，不是一个附属实验：

$$
\boxed{
\text{准确的准静态虚拟环境}
\rightarrow
\text{慢速连续 canonical teacher}
\rightarrow
\text{一致性与路径依赖审计}
\rightarrow
\text{与数据结构匹配的快速学生}
}
$$

其中：

* Fang/Bern 决定环境和 continuation；
* Thuruthel 决定教师应优化整条轨迹，而不是逐点产生标签；
* Lee/Fang local models 决定单一全局逆映射失败时，应转向局部 chart；
* 当前项目的 candidate graph、radial continuation 和 whole-curve correction，可以直接成为这套教师系统的初始实现，而不需要全部推倒重来。

[1]: https://crl.ethz.ch/papers/RoboSoft2020.pdf "Soft Robot Control with a Learned Differentiable Model"
[2]: https://ieeexplore.ieee.org/document/8531756/?utm_source=chatgpt.com "Model-Based Reinforcement Learning for Closed-Loop ..."
[3]: https://hub.hku.hk/bitstream/10722/272700/1/Content.pdf "Microsoft Word - visualservo_ICRA2019_final_v2.doc"

---
question_id: Q14
question_number: 14
question_confirmed_by_user: true
question_confirmation_summary: "用户要求再次制作 GPT-5 Pro 交接文档，清楚描述当前 V12.13 实验状态与其认可的成功部分，并请 GPT-5 Pro 给出下一步实验方案，以进一步提升数据集和 Student 的泛化性，使其不再局限于当前窄范围的椭圆 family。"
date: "2026-07-29"
status: ready-to-send
project: "quasi_exp / Branch-Aware Canonical Region Atlas"
evidence_cutoff: "2026-07-29 14:43:54 +08:00"
source_snapshot: "主项目 canonical-layer-field-u3@740d8c00fd8faf9085e3e05ba25eb212d202048f；V12 worktree codex/bacra-v12-12-fixed-point@303729e278f8edd679b63b087f3c31ed9cbe5445；两处工作树均为 dirty，V12.13 实现、配置和测试含未提交文件"
---

# GPT-5 Pro 第十四次交接：从当前椭圆走廊扩展到多几何泛化数据集

## 0. 经用户确认的 GPT-5 Pro 任务

用户已经确认的任务原意：

> 再次制作一个 handoff 文档，描述清楚当前的实验情况以及用户认为成功的部分；基于这些结果，讨论下一步应该如何进一步提升数据集泛化性，使数据集和 Student 不再局限于当前这一个椭圆。

这里的“当前这一个椭圆”是用户对当前覆盖域较窄的概括。严格来说，V12.13 已经包含多条不同中心、平面姿态、axis ratio 和 corridor 位置的椭圆 family，而不是只重复一条完全相同的轨迹；但它们的 major semiaxis 仍主要集中在约 0.49–0.50 m，最终独立验证的 axis ratio 也只覆盖约 0.326–0.342，因此尚未形成跨椭圆尺度和大范围几何参数的泛化证据。

确认范围：请基于本交接包中的 V12.12 antecedent、V12.13 数据生成、训练、独立 final geometry、逐 phase 跟踪数据和半径归一化 Gate 证据，给出下一阶段实验方案，以进一步提升数据集与 Student 对不同椭圆几何的泛化性，同时保留当前已经成功的部分。除此之外，本交接不自动增加其他独立问题。

## 1. 当前有效事实与最短摘要

### 1.1 证据截止时间

本文只使用截至 2026-07-29 14:43:54（Asia/Shanghai）已经落盘的 V12.12 和 V12.13 artifact、冻结配置、原始 Parquet/CSV、Gate、模型锁、可视化和只读数据审计。没有把尚未执行的多尺度椭圆扩展、候选方法或 Codex 的路线偏好写成实验事实。

### 1.2 当前有效事实

1. V12.12 在冻结模型之后生成 4 个 independent core 和 4 个 independent stress geometry。4/4 core Teacher 与 4 个 frozen Student 全部通过；stress00 Teacher 可行但 Student 的 FK P95/max 明显失败。这是 V12.13 扩展实验的直接 antecedent。
2. V12.13 协议为 `branch-aware-canonical-region-atlas-v12.13-exploratory-geometry-expansion-student-repair`，claim scope 是 `simulation_exploratory_geometry_expansion_and_student_repair`，不是 deployment claim。
3. V12.13 在不把 stress00 加入 D2/D3 的前提下，构造了 32 条 core→stress00 bridge centerline：12 条 direct、12 条 lateral、8 条 LHS。32 条 dense Teacher centerline 全部 training-eligible。
4. 从 12 条代表性 bridge family 构造 `3×3`、±1 mm tube，共尝试 108 条 offset trajectory；94 条为 training-eligible Silver，14 条 Reject。
5. 最终 D3 有 93,600 行、130 个 family ID：2,880 行 core Gold、23,040 行 bridge Silver、67,680 行 tube Silver。train 为 80,640 行、112 个 family ID；validation 为 12,960 行、18 个 family ID；stress00 明确不在 D3。
6. V12.13 共完成 42 个 Student training task；实际 TensorFlow 2.21.0 可见 `GPU:0`。冻结选择规则选中 `D3 + current architecture + row_loss_weight=1.0 + dls_steps=0`，并锁定 seed `20260738/39/40` 三个模型。
7. 可比的 D0 Student 在 stress00 上 0/3 seed 通过，平均 FK P95 为 10.5451 mm、最坏 FK max 为 13.5210 mm；不包含 stress00 的 D2 和 D3 分别达到 3/3 seed 通过，平均 FK P95 为 3.1227 mm 和 3.0848 mm。该结果支持“在当前 corridor 内，增加周围 Teacher-feasible geometry 能修复该 development target 的 FK 泛化失败”，但 stress00 已参与选模，不能再作为最终独立测试。
8. 在三个模型锁定后，V12.13 才以 seed `20260765` 生成 8 条 final geometry：3 条 core-like、3 条 bridge-like、2 条 boundary-like。8/8 Teacher 均生成 training-eligible Silver 闭环；纯 Student、`dls_steps=0` 的 24/24 family-seed 全部通过。
9. final 8 条 family 的 major semiaxis 为 0.491024–0.497940 m，axis ratio 为 0.326248–0.341898。最坏纯 Student FK P95 为 4.639477 mm，最坏 FK max 为 5.901130 mm，最小预测 joint margin 为 1.017825°。
10. 用户已明确认为当前 Student 合格且优秀。随后进行的 V12.13.1 post-hoc 尺度归一化重判没有重跑 Teacher 或 Student：以实际 major semiaxis 为尺度，FK P95 ≤ 1%、FK max ≤ 2%；24/24 通过，最坏 P95 为半长轴的 0.940614%，最坏 max 为 1.196403%，质量层级记录为 `excellent_simulation_tracking`。
11. `deployment_claim_gate_pass=false` 保持不变；当前成功结论只属于 simulation tracking。Teacher final family 的最低实际 joint margin 约 0.251°，不构成真实硬件 uncertainty-budget 证明。

### 1.3 关键边界

1. 当前已经证明的是约 0.5 m major semiaxis 附近、core→stress00 capability corridor 内、Teacher-feasible 闭环上的局部 task-geometry 泛化；没有证明任意大小、任意 axis ratio、任意工作空间位置或任意平面姿态的全局泛化。
2. V12.13 的 stress00 是 development target，D3 不包含它，但其结果参与模型选择；最强的 decision-independent 证据来自模型锁定后才生成的 8 条 final family。
3. final 8 条 geometry 是三个冻结模型之后生成，且本轮只读检查未发现它们与 D3 轨迹的精确 task-space 重合；但是只有 8 条，不能据此估计大范围椭圆分布中的总体通过概率。
4. D3 标称 130 个 family ID，但 12 条 `u=0,v=0` tube centerline 与已有 bridge centerline 在 task-space 上重复，因此按排序后的 720 个 `(x,y,z)` 精确曲线计数，只有 118 条独立 task-space 曲线。
5. D3 的 train/validation `family_id` 无重叠，但只读 audit 发现两组 task-space 曲线跨 split 精确重复：`bridge_lhs_02` 对 `bridge_lhs_02_u+0_v+0`，以及 `bridge_lateral_11` 对 `bridge_lateral_11_u+0_v+0`。因此内部 validation 不是完全 geometry-disjoint；这一问题不影响模型锁定后的 final 8 条独立 geometry，但限制了对 D3 内部 validation 的解释。
6. 对 12 组 task-space 重复 centerline，两次 Teacher solve 的 beta 标签并非全部相同：per-phase 六维 beta RMS 差异的 family-level P95 为 0.178–0.279°，局部最大 beta RMS 为 1.673°，局部最大单 joint 差异为 3.703°。它没有阻止当前 Student 通过，但表明逆映射多值性和 canonical label consistency 仍是未来多几何扩展需要处理的事实。
7. stress02 的 known-chart expert 为 3/3 seed 通过，但 `chart_classifier_trained=false`；当前只能说已知 chart ID 时 chart-B expert 可用，不能说 Student 已能自动在多个 branch/chart 间路由。
8. 当前没有执行新的多尺度椭圆数据生成、半径分层 holdout、跨尺度外推、真实硬件、噪声/动力学扰动或自动多 chart routing 实验。

## 2. 项目目标与本轮边界

### 2.1 用户给定的项目目标

基于论文和已冻结的机器人力学模型，生成覆盖目标闭环 task geometry、具有可行且连续 Teacher 标签的数据集，训练能够正确拟合并跟踪不同椭圆轨迹的 Student。用户当前将 V12.13 Student 视为已经合格且优秀，希望下一阶段把这种成功从当前窄椭圆 family 扩展到更广泛的椭圆几何。

### 2.2 当前实际覆盖范围

当前实现覆盖：

- 同一机器人与同一 simulation mechanics；
- 720-phase 闭合椭圆轨迹；
- core→stress00 之间的单条主要 capability corridor；
- bridge direct、lateral、LHS 和局部 ±1 mm tube；
- major semiaxis 主要在约 0.49–0.50 m；
- D3 的 chart-A 主数据与一个已知 chart ID 的 stress02 chart-B expert；
- 纯 Student 静态预测和可选的 1–2 步 DLS 诊断；
- 模型锁定后 8 条 final geometry 的纯 Student simulation tracking；
- 以实际 major semiaxis 归一化的 P95/max 跟踪 Gate。

### 2.3 当前证据未覆盖的内容

- 多个显著不同 major semiaxis 或完整 radius schedule；
- 更大范围的 axis ratio、中心位置和平面姿态组合；
- 任意椭圆 family 的 Teacher-feasibility coverage rate；
- 跨 capability component/chart 的自动识别与路由；
- 非椭圆轨迹；
- 动态时序、速度、加速度、控制器闭环、执行器饱和和噪声；
- 标定误差、结构模型误差、真实硬件和 uncertainty budget；
- 多尺度扩展后的新独立 holdout；
- 以统计方式估计的 family-level 泛化成功概率。

## 3. 相对上次材料新增的事实

Q13 的证据截止于 V12.12，当时 core 正式通过，但 stress00 出现 Teacher-feasible/Student-FK-fail 的反例。Q14 新增的是已经实际执行完成的 V12.13 数据扩展、Student 修复和 final geometry 结果。

| 类型 | 已发生的变化 | 实际证据 | 证据等级 |
| --- | --- | --- | --- |
| 数据设计 | 从 4 条 core centerline 扩展为 32 条 bridge centerline，并对 12 条代表性 family 构造 `3×3` tube | `bridge_family_catalog.parquet`、`teacher_dense_summary.csv`、`tube_summary.csv` | exploratory protocol artifact |
| 数据集 | 建立 D0/D1/D2/D3 对照；最终 D3 为 93,600 行，D2/D3 都排除 stress00 | `D3_manifest.json`、`D3_dataset.parquet` | exploratory protocol artifact |
| Student | 训练 42 个 model task，选择 D3/current/row-loss=1.0 的 3 个纯 Student | `model_selection.json`、`final_model_lock.json` | exploratory selection，模型随后冻结 |
| development target | D0 对 stress00 为 0/3，D2/D3 在未训练 stress00 的条件下达到 3/3 | `model_selection.json` | development evidence，不能当 final holdout |
| 独立 final geometry | 模型锁定后生成 8 条新 family；8/8 Teacher feasible，24/24 纯 Student family-seed 通过 | `final_family_catalog.parquet`、`final_teacher_summary.csv`、`final_metrics_per_family_seed.csv`、`final_gate.json` | decision-independent final geometry evidence |
| Gate 重判 | 使用实际 major semiaxis 的 1% P95 / 2% max Gate 对冻结结果重判，未重跑模型 | `radius_gate.json`、`radius_family_seed_gate.csv` | post-hoc simulation qualification |
| 用户判断 | 用户明确认为当前 Student 已合格且优秀，并希望保留这一成功继续扩展 | 当前对话 | user-stated interpretation |
| 数据审计 | 发现 12 条 task-space 重复中心线、118 条独立曲线和两组跨 split task-space 重复 | `D3_dataset.parquet` 的 post-hoc 只读审计 | post-hoc data-quality evidence |

## 4. 实验方法与复现口径

### 4.1 代码与工作树 fixed point

| 项目 | 值 |
| --- | --- |
| 项目根 | `/mnt/ML_projects/quasi_exp` |
| 主项目分支 / HEAD | `canonical-layer-field-u3` / `740d8c00fd8faf9085e3e05ba25eb212d202048f` |
| V12 worktree | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v12-1-capability-bridge` |
| V12 worktree 分支 / HEAD | `codex/bacra-v12-12-fixed-point` / `303729e278f8edd679b63b087f3c31ed9cbe5445` |
| runtime 记录的 source root / Git SHA | 同一 V12 worktree / `303729e278f8edd679b63b087f3c31ed9cbe5445` |
| dirty 状态 | 主项目和 V12 worktree 均 dirty；V12.13 config、runner、module、tests 和后续 radius Gate 文件含未提交修改/未跟踪文件 |
| 仅 checkout HEAD 是否足以复现 | 否。V12.13 依赖 dirty worktree 中的 `bacra_v12_13_exploratory.yaml`、`bacra_exploratory_expansion.py` 和 `run_bacra_v12_13_exploratory.py`；本包附上当前字节版本与 artifact 的 frozen config/source manifest，但不能把 checkout HEAD 称为完整源码 fixed point |

当前附件中的实现 SHA256：

- `bacra_exploratory_expansion.py`：`ce664f3c86d8fb2b0a7e7686b0284b779ff3cfd8d0b3f08d7e3721a1baa08ebb`
- `run_bacra_v12_13_exploratory.py`：`62cfa537b19c935f0d7b0646d013f30d8989fa0a057ce6e4ecf5c303b09c17f6`
- `bacra_v12_13_exploratory.yaml`：`64a2b1fe1f64becdb93799576a6a7108ac16077975abe2eaaa1853afd7ccfb54`

### 4.2 方法与参数

#### Bridge family

| 参数 | 实际值 | 来源 |
| --- | ---: | --- |
| direct family | 12 | `bridge_family_catalog.parquet` |
| lateral family | 12 | 同上 |
| LHS family | 8 | 同上 |
| 总 bridge family | 32 | `01_family_catalog/gate.json` |
| screen phase | 180 | `v12_13_frozen_config.json` |
| dense phase | 720 | 同上 |
| Silver minimum margin | 0.25° | 同上 |
| dense training transition RMS max | 3.0° | 同上 |
| residual P95 / max | ≤1 / ≤3 mm | 同上 |

#### Tube

| 参数 | 实际值 | 来源 |
| --- | ---: | --- |
| 代表性 base family | 12 | `v12_13_frozen_config.json` |
| 局部 offset | `{-1,0,+1} mm × {-1,0,+1} mm` | 同上 |
| 尝试 trajectory | 108 | `tube_summary.csv` |
| training-eligible | 94 | 同上 |
| Reject | 14 | 同上 |

#### Student

| 参数 | 实际值 | 来源 |
| --- | ---: | --- |
| training seed | 20260738、20260739、20260740 | `v12_13_frozen_config.json` |
| batch size | 256 | 同上 |
| learning rate | 0.0003 | 同上 |
| max optimizer steps | 4,500 | 同上 |
| selected architecture | current | `model_selection.json` |
| selected dataset | D3 | 同上 |
| selected row loss weight | 1.0 | 同上 |
| selected online correction | `dls_steps=0` | 同上 |
| GPU | TensorFlow 2.21.0，`GPU:0` visible | `03_training/gate.json` |

### 4.3 数据生成与划分

| 项目 | 实际口径 |
| --- | --- |
| D0 | 4 个原 core Gold family，2,880 行 |
| D1 | D0 + stress00，5 个 family，3,600 行；只用于直接拟合诊断 |
| D2 | D0 + 32 bridge centerline，36 个 family，25,920 行；不含 stress00 |
| D3 | D2 + 94 training-eligible tube trajectory，130 个 family ID，93,600 行；不含 stress00 |
| D3 train | 80,640 行，112 个 family ID |
| D3 validation | 12,960 行，18 个 family ID |
| final test | 模型锁定后生成 8 条新 family，3 seed × 8 family；不进入训练或模型选择 |
| seed | bridge lateral `20260760`、LHS `20260761`、final family `20260765` |
| 标签质量 | D3：2,880 行 Gold、90,720 行 Silver |
| 模型输入/标签 | task-space `(x_m,y_m,z_m)` 到六维 `beta1_rad…beta6_rad`；同时保存 Jacobian、权重、quality 和 split 字段 |
| 泄漏审计 | family ID split overlap 为 0；post-hoc task-space curve audit 发现 2 组跨 split 中心线重复；final 8 与 D3 无精确 task-space curve overlap |

#### Post-hoc task-space curve audit 口径

本轮只读 audit 使用完整 `D3_dataset.parquet`：

1. 对每个 `family_id` 按 `phase_idx` 稳定排序；
2. 取 720×3 的 `(x_m,y_m,z_m)`；
3. 数值四舍五入到小数点后 12 位后比较完整字节序列；
4. 统计相同 curve hash、train/validation curve overlap；
5. 对相同 task-space curve 的六维 beta 计算逐 phase RMS 和单 joint absolute difference；
6. 以相同方式比较 `final_teacher_reference.parquet` 与 D3。

源 D3 SHA256 为 `13ba2c1a72960cff2cb1fd7f31f9483bf3ae5d39a58ffa84b5d12856d3d6b9bc`；源 final Teacher reference SHA256 为 `383c1f3e09844188bbeac1e2b875e1678751cdb0654f9a1b5ee141fa6e925b29`。完整源文件均已上传，以上审计不是抽样。

### 4.4 baseline、评价和 Gate

| 项目 | 实际定义 | 是否正式 |
| --- | --- | --- |
| V12.12 antecedent | frozen V12.11 Student 在独立 core/stress geometry 上验证 | V12.12 formal simulation claim |
| V12.13 D0 baseline | 原 core 数据、相同 current architecture 和 task-space loss 对照 | exploratory baseline |
| development target | stress00；D2/D3 不训练它，但它参与 V12.13 模型选择 | development evidence |
| bridge validation | 4 条 bridge validation family，3 个 locked seed，12/12 纯 Student pass | exploratory supplement |
| final geometry | 三模型先锁定，再生成 8 条 family；纯 Student 24/24 pass | decision-independent final geometry evidence |
| 原 V12.13 Student Gate | FK P95 ≤5 mm、FK max ≤10 mm、实际 bounds、transition/seam | exploratory protocol Gate |
| V12.13.1 radius Gate | FK P95 ≤ actual major semiaxis 的1%，FK max ≤2%；joint quantities 保持角度单位 | post-hoc simulation qualification |
| deployment Gate | `false` | 明确未通过/未建立 |

### 4.5 复现入口与产物

V12.13 当前 runner 的等价入口：

```text
cd /mnt/ML_projects/quasi_exp/.worktrees/bacra-v12-1-capability-bridge
/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python \
  scripts/analysis/run_bacra_v12_13_exploratory.py \
  --config configs/bacra_v12_13_exploratory.yaml \
  --preset formal \
  --stage all
```

注意：这是附件中当前 dirty-worktree runner 的 CLI；已有 artifact 按阶段落盘，并非本次交接重新运行。由于 V12.13 runner 没有被纳入一个干净 commit，不应声称仅 checkout HEAD 可以逐字节复现。

主要产物：

- `/mnt/ML_projects/quasi_exp/runs/bacra_v12_13_exploratory_expansion`
- `/mnt/ML_projects/quasi_exp/runs/bacra_v12_13_radius_normalized_student_gate`

本次交接没有重跑 Teacher、Student 或测试；只核对已有 artifact、执行只读数据审计并打包证据。

## 5. 实验结果

### 5.1 从 V12.12 失败到 V12.13 修复

| 对象 | 是否训练 stress00 | stress00 seed pass | mean FK P95 | worst FK max | 证据等级 | 附件 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| V12.13 D0/current/row-loss=1.0 | 否 | 0/3 | 10.545119 mm | 13.521029 mm | exploratory baseline | `model_selection.json` |
| V12.13 D2/current/row-loss=1.0 | 否 | 3/3 | 3.122722 mm | 3.975961 mm | development evidence | `model_selection.json` |
| V12.13 D3/current/row-loss=1.0 | 否 | 3/3 | 3.084816 mm | 4.086677 mm | selected development result | `model_selection.json` |

D1 直接加入 stress00，只能回答模型能否拟合该轨迹，不用于证明 stress00 泛化。D2/D3 明确排除 stress00，因此上表 D2/D3 支持当前 corridor coverage 对该 development target 的修复效果。

### 5.2 Final 8 条新 family

| family | group | Teacher quality | 3 seed 中 worst FK P95 | worst FK max | min Student margin | radius Gate |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| final_core_00 | core | Silver | 3.142318 mm | 3.920608 mm | 1.155730° | pass |
| final_core_01 | core | Silver | 4.639477 mm | 5.901130 mm | 1.150250° | pass |
| final_core_02 | core | Silver | 3.587939 mm | 4.748331 mm | 1.147320° | pass |
| final_bridge_00 | bridge | Silver | 2.674617 mm | 3.733209 mm | 1.214597° | pass |
| final_bridge_01 | bridge | Silver | 2.340053 mm | 3.058676 mm | 1.017825° | pass |
| final_bridge_02 | bridge | Silver | 2.622782 mm | 3.274833 mm | 1.093144° | pass |
| final_boundary_00 | boundary | Silver | 3.095895 mm | 4.242676 mm | 1.187769° | pass |
| final_boundary_01 | boundary | Silver | 2.163615 mm | 3.132435 mm | 1.213241° | pass |

总体结果：

- Teacher：8/8 training-eligible；
- 纯 Student：24/24 family-seed 通过；
- family：8/8 达到 3/3 seed pass；
- 最坏 P95/major-semiaxis：0.940614%，Gate 为1%；
- 最坏 max/major-semiaxis：1.196403%，Gate 为2%；
- DLS：最终成功不依赖 DLS，`dls_steps=0`；
- deployment claim：false。

逐 family-seed 原始数据在 `final_metrics_per_family_seed.csv`，逐 phase 原始跟踪数据在 `final_tracking_details.parquet`。`final_core01_plane_tracking.png` 展示最坏 P95 family，`final_boundary00_plane_tracking.png` 展示一条 boundary-like family。

### 5.3 Chart-B 已执行结果

对已知 `chart_id` 的 stress02 chart-B expert：

- 3 个 expert 均训练完成；
- 3/3 pure seed pass；
- stress02 hard cycle 已登记；
- `chart_classifier_trained=false`；
- 因此结果只支持 known-chart expert，不支持自动全局路由。

### 5.4 数据质量审计

| 审计项 | 结果 | 证据等级 | 影响边界 |
| --- | ---: | --- | --- |
| D3 family ID | 130 | artifact | 不能直接等同于独立 task-space curve 数 |
| 独立 task-space curve | 118 | post-hoc full-data audit | 12 条 `u=0,v=0` 是已有 centerline 的重复 |
| family ID train/validation overlap | 0 | artifact/full-data audit | family ID split 正确 |
| task-space curve train/validation overlap | 2 组 | post-hoc full-data audit | D3 内部 validation 不是完全 geometry-disjoint |
| duplicate-label beta RMS family P95 | 0.178–0.279° | post-hoc full-data audit | 相同 task point 存在轻度到局部明显的多值标签 |
| duplicate-label局部 max beta RMS | 1.673° | post-hoc full-data audit | 不能假设 static mapping 标签严格唯一 |
| duplicate-label局部 max joint abs | 3.703° | post-hoc full-data audit | 需要与 branch/canonical 语义共同解释 |
| final 8 与 D3 exact curve overlap | 0 | post-hoc full-data audit | 支持 final geometry 的非重复性 |

## 6. 证据等级、失败结果与未知事项

### 6.1 正式或当前有效事实

- V12.12 的 simulation decision-independent core claim 已通过；stress00 是明确的 Teacher-feasible/Student-FK-fail antecedent。
- V12.13 的 32 条 bridge centerline、94 条 tube、D3 数据集、42 个模型任务和 final 8 geometry 均有落盘 artifact。
- D3 不包含 stress00。
- 三个锁定 D3 Student 在 final 8 条新 family 上 24/24 纯 Student 通过。
- 用户把当前 Student 视为合格且优秀。
- V12.13.1 radius-normalized Gate 对冻结指标的重判为 pass，但不增加新实验数据。

### 6.2 诊断、探索、post-hoc 和用户判断

| 内容 | 等级 | 能支持什么 | 不能支持什么 |
| --- | --- | --- | --- |
| D0→D2/D3 的 stress00 修复 | development/exploratory | 当前 corridor geometry coverage 与 stress00 FK 改善共同出现，且 D2/D3 未训练 stress00 | 不能把 stress00 再当 unbiased final test；不能自动外推到其他尺度 |
| final 8 geometry | decision-independent final geometry | 锁定模型在 8 条新且不与 D3 精确重合的可行闭环上表现良好 | 不能估计全椭圆参数域的通过率 |
| radius-normalized Gate | post-hoc qualification | 用统一相对尺度描述当前冻结轨迹误差 | 不能证明未实际测试半径上的误差仍按比例变化 |
| 用户认为 Student 优秀 | user-stated interpretation | 定义当前探索阶段的接受状态和下一阶段起点 | 不等同于 deployment 或论文强结论 |
| duplicate/split audit | post-hoc full-data audit | 暴露 D3 的有效独立曲线数、两组 split 重复和标签多值性 | 不证明这些重复是当前模型成功或失败的根因 |
| chart-B expert | exploratory known-chart result | 已知 chart 时 stress02 expert 可工作 | 不证明自动 chart classifier 或 global multi-chart generalization |

### 6.3 已发现的实现或证据缺口

- V12.13 代码与配置来自 dirty worktree，未封存在单一 clean commit；当前附件保存字节版本，但不是严格干净源码 fixed point。
- D3 的 `(u=0,v=0)` tube 没有去除已存在的 centerline，造成 12 条 task-space 重复和两组跨 split 几何重合。
- 同一 task-space curve 的重复 Teacher solve 会产生不同 beta label，尚未冻结统一的 duplicate/canonical-label policy。
- 当前 split 主要按 family ID，而不是按 task-geometry fingerprint 或 parameter block 分组。
- final 8 family 数量较少，且位于设计好的 core/bridge/boundary corridor；没有覆盖跨半径的大尺度矩阵。
- Teacher-feasible family 的条件性能与整个 proposal distribution 的 Teacher-feasibility coverage 尚未分开报告。
- chart-B 仍依赖 known chart ID，没有自动 router。

### 6.4 尚未知或尚未执行

- 不同 major semiaxis 下 Teacher capability component 是否保持连通；
- Student 的绝对误差是否随 major semiaxis 线性变化，当前 1%/2% Gate 只是未跨尺度验证的规范；
- 当前 D3 Student 在明显更小或更大椭圆上的插值/外推边界；
- axis ratio、中心偏移、平面倾角和 radius 之间是否存在强交互；
- 多尺度数据是否需要单一 global Student、scale-conditioned Student 或 multiple experts；
- 多尺度扩展会不会损害当前 0.5 m corridor 的 core retention；
- 跨 chart 的自动路由准确率和路由错误代价；
- Teacher 在更广 proposal distribution 上的可标注比例和主要拒绝机制；
- 真实控制闭环、噪声、动力学和硬件表现。

### 6.5 当前不能声称

- 不能声称当前数据集覆盖任意椭圆或完整机器人工作空间。
- 不能声称 1%/2% radius Gate 已在不同半径上得到经验校准。
- 不能声称 130 个 family ID 等于 130 条独立几何。
- 不能声称 D3 的 validation 完全 geometry-disjoint。
- 不能声称 Student 已能自动处理 chart A/B 或任意逆解 branch。
- 不能声称当前结果具备 deployment safety 或 sim-to-real 证据。
- 不能因 final 8 条全部通过而推断大范围椭圆分布的总体成功率。

## 7. 决策所需的客观对照

本节只比较已经实际执行的方案与结果，不给出 Codex 推荐或下一步路线排序。

| 对照对象 | 已执行输入/方法 | 已观察结果 | 证据等级 | 已知限制 |
| --- | --- | --- | --- | --- |
| D0 | 4 条 core Gold centerline | stress00 0/3；mean P95 10.545 mm | exploratory baseline | 几何覆盖最窄 |
| D1 | D0 + stress00 | 用于直接拟合能力诊断 | diagnostic | 不能证明 stress00 泛化 |
| D2 | D0 + 32 bridge centerline，不含 stress00 | stress00 3/3；mean P95 3.123 mm | development | 仍主要是 centerline manifold |
| D3 | D2 + 94 tube trajectory，不含 stress00 | stress00 3/3；mean P95 3.085 mm；最终被选中 | development + final model source | 主要集中在单一 corridor 和约 0.5 m 尺度 |
| final 8 | 模型锁定后生成 3 core + 3 bridge + 2 boundary family | Teacher 8/8；纯 Student 24/24 | decision-independent final geometry | 只有8条，且尺度范围窄 |
| chart-B expert | stress02 已知 chart ID 的独立 expert | pure seed 3/3 | exploratory | 没有 router |
| radius Gate | 对现有 frozen final metrics 进行 major-semiaxis 归一化 | P95 0.941%、max 1.196%，24/24 pass | post-hoc | 没有跨尺度校准 |

## 8. 经用户确认的问题

1. 基于 V12.13 已经成功实现的局部 capability-corridor 数据扩展、stress00 修复、三个 frozen Student 在 8 条模型锁定后新 geometry 上 24/24 通过，以及半径归一化跟踪结果，下一步应该如何设计实验和数据扩展，才能进一步提升数据集与 Student 的泛化性，使其不再局限于当前约 0.5 m 的窄椭圆 family？

## 9. 经用户确认的回答要求

请给出下一阶段实验方案，目标是把当前已经成功的 simulation tracking 能力扩展到更广泛的椭圆几何，同时明确保留当前成功部分。回答应以本包已核实的当前数据、结果、边界和未知事项为基础，不把尚未执行的多尺度实验当成既成事实。

## 10. 附件清单与复现信息

GPT-5 Pro 实际可见的是“上传文件名”列；原始 SSH 路径只用于 provenance。

唯一上传 ZIP 的成员数（`QUESTION.md + 实际证据附件`）：34。无论成员数多少，网页端和 Windows 传输都只使用打包器生成的 `Q14_GPT5Pro_upload.zip`。

| 上传文件名 | 原始 SSH 路径 | 约大小 | 证据等级 | 支持事实 | 上传理由 |
| --- | --- | ---: | --- | --- | --- |
| `v12_12_new_geometry_catalog.csv` | `runs/branch_aware_canonical_region_atlas_v12_12_independent_gold_set_geometry_holdout_retry3/00_protocol/new_geometry_catalog.csv` | 4 KiB | formal antecedent | V12.12 core/stress geometry 范围 | 允许对照扩展前几何 |
| `v12_12_teacher_gate.json` | `runs/branch_aware_canonical_region_atlas_v12_12_independent_gold_set_geometry_holdout_retry3/01_teacher_reference/gate.json` | 5 KiB | formal antecedent | V12.12 Teacher core/stress 状态 | 证明 stress00 Teacher 可行和 stress 边界 |
| `v12_12_student_gate.json` | `runs/branch_aware_canonical_region_atlas_v12_12_independent_gold_set_geometry_holdout_retry3/02_student_evaluation/gate.json` | 5 KiB | formal antecedent | V12.12 Student core pass/stress fail | 证明 V12.13 的直接起点 |
| `v12_12_family_summary.csv` | `runs/branch_aware_canonical_region_atlas_v12_12_independent_gold_set_geometry_holdout_retry3/04_summary/family_summary.csv` | 2 KiB | formal antecedent | V12.12 逐 family 汇总 | 提供紧凑对照 |
| `v12_12_summary_gate.json` | `runs/branch_aware_canonical_region_atlas_v12_12_independent_gold_set_geometry_holdout_retry3/04_summary/gate.json` | 1 KiB | formal antecedent | V12.12 总 claim 边界 | 区分 core claim 和 stress evidence |
| `v12_13_frozen_config.json` | `runs/bacra_v12_13_exploratory_expansion/00_protocol/frozen_config.json` | 24 KiB | artifact | V12.13 全部实际参数 | 完整方法口径 |
| `v12_13_runtime.json` | `runs/bacra_v12_13_exploratory_expansion/00_protocol/runtime.json` | 1 KiB | artifact | Python/TF/Git/source root | 环境与 provenance |
| `v12_13_source_manifest.json` | `runs/bacra_v12_13_exploratory_expansion/00_protocol/source_manifest.json` | 3 KiB | artifact | V12.11/V12.12 输入 SHA | 源 lineage |
| `v12_13_protocol_gate.json` | `runs/bacra_v12_13_exploratory_expansion/00_protocol/gate.json` | 2 KiB | artifact | V12.13 source fixed point checks | 协议入口 |
| `bacra_v12_13_exploratory.yaml` | `.worktrees/bacra-v12-1-capability-bridge/configs/bacra_v12_13_exploratory.yaml` | 3 KiB | current implementation snapshot | 人类可读配置 | 便于 Pro 检查设计参数 |
| `bacra_exploratory_expansion.py` | `.worktrees/bacra-v12-1-capability-bridge/src/quasi_exp/teacher/bacra_exploratory_expansion.py` | 27 KiB | current implementation snapshot | family、tube、dataset、Student 辅助实现 | 允许核验数据语义 |
| `run_bacra_v12_13_exploratory.py` | `.worktrees/bacra-v12-1-capability-bridge/scripts/analysis/run_bacra_v12_13_exploratory.py` | 101 KiB | current implementation snapshot | V12.13 runner | 允许核验完整执行路径；dirty 状态已披露 |
| `bridge_family_catalog.parquet` | `runs/bacra_v12_13_exploratory_expansion/01_family_catalog/bridge_family_catalog.parquet` | 19 KiB | artifact | 32 条 bridge 的完整参数 | 判断当前几何覆盖 |
| `teacher_dense_summary.csv` | `runs/bacra_v12_13_exploratory_expansion/01_family_catalog/teacher_dense_summary.csv` | 6 KiB | artifact | 32 条 dense Teacher 质量 | 判断标签可用性 |
| `tube_summary.csv` | `runs/bacra_v12_13_exploratory_expansion/01_family_catalog/tube/tube_summary.csv` | 20 KiB | artifact | 108 条 tube 的 pass/Reject | 判断局部覆盖与失败 |
| `D3_manifest.json` | `runs/bacra_v12_13_exploratory_expansion/02_datasets/D3_manifest.json` | 1 KiB | artifact | D3 行数、family、split、SHA | 数据集身份 |
| `D3_dataset.parquet` | `runs/bacra_v12_13_exploratory_expansion/02_datasets/D3_dataset.parquet` | 25.4 MiB | artifact，完整原始数据 | 93,600 行训练/validation、标签、Jacobian、split | 允许 Pro 独立检查分布、多值性和扩展接口 |
| `model_selection.json` | `runs/bacra_v12_13_exploratory_expansion/03_training/model_selection.json` | 7 KiB | exploratory selection | D0–D3、loss、DLS、seed 排名 | 证明 stress00 修复与选模 |
| `bridge_evaluation_gate.json` | `runs/bacra_v12_13_exploratory_expansion/03_training/bridge_evaluation_gate.json` | 2 KiB | exploratory supplement | 4 bridge family ×3 seed 的12/12 | 新 family tracking 补充 |
| `active_enrichment_gate.json` | `runs/bacra_v12_13_exploratory_expansion/04_active_enrichment/gate.json` | 1 KiB | exploratory checkpoint | active enrichment 为 `not_required` | 说明已执行决策 |
| `chart_b_gate.json` | `runs/bacra_v12_13_exploratory_expansion/05_chart_b_experts_retry1/gate.json` | 5 KiB | exploratory | chart-B 3/3、无 classifier | 多 chart 当前边界 |
| `final_model_lock.json` | `runs/bacra_v12_13_exploratory_expansion/06_final/model_lock.json` | 2 KiB | final protocol artifact | 三模型 SHA、catalog after lock | 证明 final geometry 决策独立 |
| `final_family_catalog.parquet` | `runs/bacra_v12_13_exploratory_expansion/06_final/new_family_catalog.parquet` | 20 KiB | final artifact | 8 条 final geometry 参数与 fingerprint | 判断 final 覆盖范围 |
| `final_teacher_summary.csv` | `runs/bacra_v12_13_exploratory_expansion/06_final/teacher_summary.csv` | 2 KiB | final artifact | 8/8 Teacher Silver | 标签有效性 |
| `final_teacher_reference.parquet` | `runs/bacra_v12_13_exploratory_expansion/06_final/teacher_reference.parquet` | 2.1 MiB | final artifact，完整原始数据 | 8×720 Teacher reference 和 certificate 字段 | 允许逐 phase 核验 Teacher 与独立曲线 |
| `final_metrics_per_family_seed.csv` | `runs/bacra_v12_13_exploratory_expansion/06_final/final_metrics_per_family_seed.csv` | 11 KiB | final artifact | 3 seed×8 family×DLS0/1 指标 | 核验24/24纯 Student |
| `final_tracking_details.parquet` | `runs/bacra_v12_13_exploratory_expansion/06_final/final_tracking_details.parquet` | 3.7 MiB | final artifact，完整原始数据 | 逐 phase target/prediction/FK/margin | 判断误差形态与扩展风险 |
| `final_gate.json` | `runs/bacra_v12_13_exploratory_expansion/06_final/gate.json` | 4 KiB | final artifact | 8/8 Teacher、24/24 Student 和并行证据 | 当前总结果 |
| `radius_gate.json` | `runs/bacra_v12_13_radius_normalized_student_gate/gate.json` | 2 KiB | post-hoc qualification | 1%/2% Gate 和总通过 | 用户当前接受语义 |
| `radius_family_summary.csv` | `runs/bacra_v12_13_radius_normalized_student_gate/family_gate_summary.csv` | 2 KiB | post-hoc qualification | 逐 family 相对半径汇总 | 比较 family 难度 |
| `radius_family_seed_gate.csv` | `runs/bacra_v12_13_radius_normalized_student_gate/family_seed_gate.csv` | 9 KiB | post-hoc qualification | 24 条逐 seed 相对指标 | 核验最坏 seed |
| `final_core01_plane_tracking.png` | `runs/bacra_v12_13_exploratory_expansion/06_final/final_tracking_plots_plane_aligned/final_core_01_plane_aligned_tracking.png` | 499 KiB | visualization | 最坏 FK P95 final family | 展示用户认可的实际椭圆平面跟踪 |
| `final_boundary00_plane_tracking.png` | `runs/bacra_v12_13_exploratory_expansion/06_final/final_tracking_plots_plane_aligned/final_boundary_00_plane_aligned_tracking.png` | 502 KiB | visualization | boundary-like family | 展示边界几何跟踪 |

### 未上传的大文件或派生数据说明

- 本包上传完整 `D3_dataset.parquet`、`final_teacher_reference.parquet` 和 `final_tracking_details.parquet`，没有用抽样文件代替核心数据。
- 本包不上传 `.keras` 模型、42 个训练目录、全部日志、全部 8 张重复 tracking 图、D0/D1/D2 全量数据、V12.1–V12.11 整个历史 run 或完整 V12.13 目录。这些内容会显著增加体积，而现有附件已经能够核验当前数据结构、训练对照、final geometry 和逐 phase tracking。
- duplicate/split audit 是对完整 D3 的 post-hoc 只读计算；没有另行生成抽样数据。正文给出了源 SHA、完整行数和计算口径，GPT 可用上传的完整 D3 独立复核。
- 两张 PNG 分别支持最坏 P95 family 和 boundary-like family，不代表额外数值证据；数值以 CSV/Parquet/Gate 为准。
- 当前实现源码是 dirty-worktree 的字节快照，正文已明确披露其复现边界。

## 11. 给下一位讨论者的最短事实摘要

1. V12.12 在新 core geometry 上成功，但 stress00 暴露 Teacher-feasible/Student-FK-fail；V12.13 随后在不训练 stress00 的条件下构造 core→stress00 bridge 和 tube。
2. D3 有93,600行、130个family ID；按独立 task-space curve 为118条。D3 Student 把 stress00 从D0的0/3修复到3/3，但stress00是development target。
3. 三个 D3 Student 在模型锁定后生成的8条新core/bridge/boundary geometry上实现24/24纯Student通过；最坏P95为半长轴0.941%，最坏max为1.196%。用户认可其为合格且优秀的simulation Student。
4. 当前覆盖仍主要是约0.49–0.50 m的窄椭圆family；D3还存在12条中心线重复、两组跨split几何重合、相同task point的多值beta标签，以及无自动router的chart-B边界。
5. 当前没有多尺度椭圆、完整axis/position/tilt矩阵、跨尺度holdout或真实硬件证据。
6. 用户已确认的问题：下一阶段应如何扩展数据集与实验，使当前成功的Student不再局限于这一窄椭圆family，并获得更广泛的椭圆几何泛化能力？

---

# 核心判断

你的判断是对的：**下一阶段应停止把“椭圆 family 参数”当作训练数据的主要生成坐标系，转而直接构造一个三维任务空间区域中的一致逆映射数据集。**

V12.13 已经证明两件关键事实：

1. 当前 Teacher 技术路线能够稳定生成大量可用闭环标签：32 条 bridge centerline 全部可用于训练，108 条局部 tube 轨迹中有 94 条可用。
2. 当前 Student 已经足以拟合 Teacher：D2、D3 在没有直接训练 stress00 的情况下都达到 (3/3) seed 通过，模型锁定后生成的 8 条新椭圆 family 又实现了 (24/24) family-seed 通过。

而且，D2 只有 (25{,}920) 行中心线数据，D3 增加到 (93{,}600) 行后，stress00 的 FK P95 只从 (3.123)mm 改善到 (3.085)mm。也就是说，额外增加约 (67{,}680) 行窄 tube 样本，只带来很小的边际收益；真正产生主要提升的是**增加不同几何 family 的覆盖**。这说明下一步应该优先增加任务空间覆盖，而不是继续在已有椭圆附近加密。

因此，下一轮建议改成：

$$
\boxed{
\text{当前成功 chart 作为种子}
\rightarrow
\text{三维任务空间区域生长}
\rightarrow
\text{Teacher 多路径一致标注}
\rightarrow
\text{区域内稠密采样}
\rightarrow
\text{Student 拟合}
\rightarrow
\text{非椭圆轨迹测试}
}
$$

本轮不追求完整工作空间，也不处理所有困难区域。目标是构造一个**足够大、单值、连续、Teacher 容易标注的连通区域**。遇到 branch 冲突、难以到达或条件很差的区域，先从数据集中排除。

---

# 一、为什么不能直接生成随机末端点并逐点求 IK

可以生成“一团末端点”，但不能对每个点完全独立运行 Teacher。

若对每个目标点独立求解：

$$
\beta_i=
\operatorname*{IK}(x_i;\text{random seed}),
$$

即使：

$$
x_i\approx x_j,
$$

也可能得到：

$$
\beta_i\not\approx\beta_j.
$$

因为你的系统是：

$$
F:\mathbb R^6\rightarrow\mathbb R^3,
$$

同一个 (xyz) 对应一个三维左右的冗余解集合。独立 seed、不同求解方向和不同局部初值都可能选择不同构型。

所以，正确的数据生成方式不是：

$$
\text{随机 }xyz
\rightarrow
\text{独立 IK},
$$

而是：

$$
\boxed{
\text{连通的 }xyz\text{ 点云}
\rightarrow
\text{从已有一致标签向邻域逐步传播}
}
$$

Teacher 仍然执行“跟踪”，但跟踪的不是一条大椭圆，而是任务空间图中的许多**短局部边**。

---

# 二、下一轮方法：Region-Grown Canonical Teacher

建议将新实验命名为：

```text
BACRA V12.14 — Region-Grown Canonical Workspace Dataset
```

本轮只构造一个主 chart：

```text
chart_id = chart_A_region
```

暂不纳入 stress02 的 chart-B，也不做自动 chart classifier。对于出现明显多值性的任务区域，先排除，不进入当前静态：

$$
xyz\rightarrow\beta_6
$$

训练集。

---

# 三、数据生成的数学定义

设需要覆盖的任务区域为：

$$
\Omega\subset\mathbb R^3.
$$

在区域中生成稀疏节点：

$$
V_X={x_1,\ldots,x_N}.
$$

对于一个新节点 (x_i)，从若干已经获得标签的邻居：

$$
\mathcal P_i=
{(x_{p_1},\beta_{p_1}),\ldots,(x_{p_K},\beta_{p_K})}
$$

分别进行局部 continuation。

从父节点 (p) 出发的候选定义为：

$$
\beta_{i|p}
===========

\operatorname*{argmin}_{\beta}
\left[
\frac{|F(\beta)-x_i|^2}{\sigma_x^2}
+
\lambda_a
|\beta-\beta_p|*W^2
+
\lambda_c C*{\mathrm{posture}}(\beta)
\right],
$$

其中：

$$
W=\operatorname{diag}(4,4,2,2,1,1).
$$

若父节点到目标距离较远，则把线段：

$$
x_p\rightarrow x_i
$$

分成若干不超过 (5)mm 的小步，并让前一步的 (\beta) 作为下一步初值。

对同一个 (x_i)，至少从三个不同父节点生成候选：

$$
\beta_{i|p_1},
\beta_{i|p_2},
\beta_{i|p_3}.
$$

若这些候选彼此接近，则说明该点的标签对到达路径不敏感：

$$
\max_{p,q}
d_\beta
\left(
\beta_{i|p},\beta_{i|q}
\right)
\le\epsilon_{\mathrm{cons}}.
$$

其中：

$$
d_\beta(\beta,\beta')
=====================

\sqrt{
\frac{1}{6}
\sum_{j=1}^{6}
(\beta_j-\beta'_j)^2
}.
$$

最终标签选择候选 medoid：

$$
\beta_i =
\operatorname*{argmin}_{\beta_{i|p}}
\sum_q
d_\beta^2
\left(
\beta_{i|p},\beta_{i|q}
\right).
$$

这会产生一个确定的、局部一致的 Teacher 数据集。

---

# 四、Phase 0：清理当前 D3，建立一致的种子集

当前 D3 有 (93{,}600) 行、130 个 family ID，但只有 118 条独立 task-space 曲线。12 条 (u=0,v=0) tube centerline 与已有 bridge centerline 重复；相同 task-space 曲线的两次 Teacher 标签还可能存在约 (0.178^\circ\sim0.279^\circ) 的 family-level P95 差异，局部最大六维 RMS 达 (1.673^\circ)。

因此，不能直接把全部 D3 当作无冲突区域种子。

## 4.1 精确重复数据处理

对完全相同的：

$$
(x,y,z,\text{phase})
$$

只保留一个标签。

优先级：

1. 原始 core/bridge centerline；
2. 非 (u=0,v=0) tube；
3. FK residual 更低；
4. local beta discontinuity 更低；
5. margin 更大。

具体来说：

- 删除所有与现有 bridge centerline 完全重复的 `u=0,v=0` tube 轨迹；
- 对其他精确重复点，只保留一个 canonical label；
- 不允许对两个相差较大的 (\beta) 直接做算术平均。

## 4.2 生成种子子集

从清理后的 D3 中使用 task-space farthest-point sampling 选取：

$$
N_{\mathrm{seed}}=500
$$

个空间分布较均匀的种子。

每个种子保存：

```text
seed_id
x_m, y_m, z_m
beta1...beta6
teacher_residual
joint_margin
kappa
source_family
source_chart
```

这些种子不再携带“椭圆轨迹必须继续延伸”的语义，它们只是 chart-A 中已经验证的任务空间锚点。

---

# 五、Phase 1：定义一个安全但更宽的三维任务区域

## 5.1 复用 branch-agnostic FK capability pool

使用现有 full-(\beta_6) Sobol FK pool，建立任务空间 voxel map。

Pilot：

$$
\Delta_{\mathrm{voxel}}=10\text{mm}.
$$

正式：

$$
\Delta_{\mathrm{voxel}}=5\text{mm}.
$$

每个 voxel 至少记录：

```text
sample_count
nearest_fk_distance
best_joint_margin
best_kappa
distance_to_D3_cloud
connected_component_id
```

## 5.2 区域生长方式

不再定义椭圆半径、中心、平面或 axis ratio。

从包含当前 500 个种子的 voxels 开始，向六邻域或二十六邻域 flood-fill：

```text
current accepted voxels
    ↓
adjacent capability-supported voxels
    ↓
Teacher consistency check
    ↓
accepted / rejected
```

初始扩展半径建议为：

$$
20\sim30\text{mm}
$$

的 voxel-geodesic 厚度。

区域候选 voxel 的宽松条件：

1. capability pool 中存在足够的 FK support；
2. 位于当前机器人实际工作空间内；
3. 存在实际 bounds 内的候选；
4. conditioning 不超过当前 D3 已接受数据的 P95 或其 (1.5) 倍；
5. 与已接受区域连通。

本轮不试图覆盖边界、空洞和高难度区域。只保留包含当前 D3 种子的最大连通 component。

---

# 六、Phase 2：稀疏区域生长

## 6.1 任务点采样

从候选区域中使用 Poisson-disk 或 farthest-point sampling 生成：

### Smoke

$$
N_X=200.
$$

### Pilot

$$
N_X=2{,}000.
$$

### Formal exploratory

$$
N_X=5{,}000.
$$

其中建议：

- (70%) 区域内部；
- (20%) 当前 chart 边界附近；
- (10%) 当前 D3 覆盖稀疏但 capability 较好的位置。

## 6.2 标签传播顺序

按照任务点到已标注集合的图距离，从近到远处理。

每次优先选择：

- 已标注邻居至少 3 个；
- 与父节点距离不超过 (10\sim15)mm；
- 所在线段均处于候选 voxel component 内；

的新节点。

对每个目标节点运行三个父节点 continuation。

## 6.3 标签质量等级

### Gold

满足：

$$
e_{\mathrm{FK}}\le1\text{mm},
$$

$$
d_{\mathrm{parent,max}}\le0.5^\circ,
$$

并在 actual bounds 内。

### Silver

满足：

$$
e_{\mathrm{FK}}\le3\text{mm},
$$

$$
d_{\mathrm{parent,max}}\le1.0^\circ,
$$

并在 actual bounds 内。

### Reject

任一情况：

$$
e_{\mathrm{FK}}>3\text{mm},
$$

或：

$$
d_{\mathrm{parent,max}}>1.0^\circ,
$$

或 solver 不收敛、实际关节越界。

Gold 和 Silver 均可用于本轮探索性训练，Silver 使用较低权重。Reject voxel 不继续向外传播，因此困难区域会自然形成 chart 边界。

## 6.4 邻接一致性

对于任务空间中距离不超过 (10)mm 的两个已接受节点，要求：

$$
d_\beta(\beta_i,\beta_j)\le1^\circ.
$$

若超过 (1^\circ)：

- 比较两个节点的 parent lineage；
- 保留与周围多数邻居更一致的标签；
- 无法消除时，删除该节点或切断对应图边；
- 本轮不建立第二 chart。

最终只保留最大的单值连通 component。

---

# 七、Phase 3：轻量路径一致性审计

本轮不需要重建复杂 whole-surface fixed-point 系统，但仍需做最低限度的一致性检查。

## 7.1 三角闭环

从 task graph 中随机抽取：

$$
100
$$

个短三角环：

$$
x_a\rightarrow x_b\rightarrow x_c\rightarrow x_a.
$$

比较返回构型：

$$
d_{\mathrm{return}}
===================

d_\beta
\left(
\beta_a^{\mathrm{return}},
\beta_a
\right).
$$

目标：

$$
d_{\mathrm{return,P95}}\le0.5^\circ,
$$

最大值允许：

$$
\le1^\circ.
$$

## 7.2 双路径到达

对 100 个节点，从两个不同父节点路径传播到目标点，比较最终标签。

目标：

$$
d_{\mathrm{two-path,P95}}\le0.5^\circ.
$$

若某个局部区域失败，直接把该区域裁掉，不要求修复所有 branch 拓扑。

---

# 八、Phase 4：在稀疏 chart 内稠密生成区域点

当 (2{,}000\sim5{,}000) 个稀疏节点完成后，再生成真正的大数据集。

## 8.1 任务空间采样

在已接受 voxels 中均匀采样：

### Dense Pilot

$$
N_{\mathrm{dense}}=50{,}000.
$$

### Formal exploratory

$$
N_{\mathrm{dense}}=100{,}000.
$$

可选最终：

$$
150{,}000.
$$

## 8.2 初值方法一：局部 tetrahedral interpolation

为稀疏节点建立局部 Delaunay tetrahedra。

若目标位于：

$$
{x_1,x_2,x_3,x_4}
$$

构成的 tetrahedron 中，计算重心系数：

$$
x=\sum_{j=1}^{4}\lambda_jx_j,
\qquad
\sum_j\lambda_j=1.
$$

构型初值：

$$
\beta_{\mathrm{init}}
=====================

\sum_{j=1}^{4}\lambda_j\beta_j.
$$

随后使用 exact Teacher corrector 求：

$$
F(\beta^\star)\approx x.
$$

## 8.3 初值方法二：多邻居 Jacobian predictor

如果没有合法 tetrahedron，则从三个最近 anchor 分别预测：

$$
\beta_{\mathrm{pred},p}
=======================

\beta_p
+
J_{W,p}^{#}
(x-x_p).
$$

三组 prediction 分别 correct，再做 multi-parent agreement。

## 8.4 Dense 样本接受条件

与稀疏节点相同：

- Gold：residual (\le1)mm，双初始化差异 (\le0.5^\circ)；
- Silver：residual (\le3)mm，差异 (\le1^\circ)；
- Reject：超过这些条件。

Dense 阶段不得用独立随机 IK 重新选择 branch。

---

# 九、最终数据集构成

建议构造三个数据版本。

## R0：当前曲线数据

清理后的 D3：

```text
current curve-corridor data
```

保持约 (80\text{k}\sim90\text{k}) 行，作为当前基线。

## R1：纯区域数据

从新 chart 中均匀采样：

$$
100{,}000
$$

行。

它不按 ellipse family 或 phase 组织，只按 task-space voxel 组织。

## R2：混合数据

建议：

- (20%) 清理后的 D3；
- (80%) 新区域数据。

总量：

$$
120{,}000\sim150{,}000.
$$

保留少量旧轨迹数据，是为了避免当前已经优秀的椭圆能力被完全冲淡；但区域点应成为主要部分。

---

# 十、数据采样平衡

当前 D3 中每条轨迹有 720 行，导致局部轨迹密度很高。

新的区域数据应采用 voxel-balanced sampling。

对于 voxel (v) 中有 (n_v) 个样本，样本权重取：

$$
w_v\propto\frac{1}{n_v}.
$$

训练 batch 建议：

- (20%) 旧 D3；
- (60%) 区域内部；
- (20%) 区域边界或高误差区域。

Gold 权重：

$$
w_{\mathrm{quality}}=1.
$$

Silver 权重：

$$
w_{\mathrm{quality}}=0.5.
$$

不需要在本轮继续使用 family-balanced sampler 作为主要机制。

---

# 十一、Student 实验设计

当前 Student 架构已经证明足够有效，因此本轮不进行大规模架构搜索。

固定：

```text
current architecture
bounded tanh beta output
beta loss
FK loss
row-space loss weight = 1.0
dls_steps = 0
```

每个数据版本训练：

$$
3
$$

个 seed。

比较：

| 数据 | 目的 |
| -- | ---------------- |
| R0 | 当前椭圆 corridor 基线 |
| R1 | 纯区域泛化能力 |
| R2 | 区域能力与原椭圆能力折中 |

Student 输入仍然只有：

$$
(x,y,z).
$$

输出：

$$
\beta_1,\ldots,\beta_6.
$$

本轮不加入 phase、family、chart ID 或 previous beta，因为我们刻意只保留单一、路径一致的 chart-A 区域。

---

# 十二、测试设计：不再只用椭圆

评价分成两部分。

## 12.1 随机任务点测试

在 region 数据生成后，另外采样：

$$
10{,}000
$$

个不参与训练的 task-space 点。

由 Teacher 给出参考标签，评价：

- Student (\beta) error；
- FK P50/P95/max；
- joint bounds；
- error 对训练 NN distance；
- error 对 chart 边界距离。

主要目标：

$$
FK_{\mathrm{P95}}\le5\text{mm},
$$

$$
FK_{\max}\le10\text{mm}.
$$

## 12.2 未见轨迹测试

在区域内生成 20 条 Teacher-feasible 轨迹：

| 轨迹类型 | 数量 |
| -------------------------------- | -: |
| 随机中心、姿态、尺度的椭圆 | 4 |
| 随机平面圆 | 4 |
| 三维 Lissajous | 4 |
| 闭合 cubic B-spline | 4 |
| 开放式平滑 B-spline / inspection path | 4 |

每条使用：

$$
360
$$

或：

$$
720
$$

个点。

这些轨迹不参与训练；只要求它们完全位于已接受 chart 内。

每条轨迹计算：

- FK P95；
- FK max；
- phase/path smoothness；
- cyclic seam，若为闭环；
- 3 个 Student seed 的通过情况。

成功目标：

- 至少 (16/20) 条轨迹在 (2/3) seed 上满足 (5/10)mm；
- 当前 V12.13 的 8 条 final 椭圆继续满足现有 (1%/2%) 相对尺度标准。

---

# 十三、一次主动加密

第一轮 Student 训练后，对 10k random point test 计算每个 voxel 的平均 FK error。

选择误差最高的：

$$
10%
$$

voxel，但只选择 Teacher 仍然能稳定标注的部分。

在这些 voxels 中增加：

$$
20{,}000
$$

个样本。

重新训练表现最好的 R1/R2 配置。

最多进行一次主动加密。本轮目标只是证明区域数据优于椭圆 corridor 数据，不需要无限循环。

---

# 十四、实验成功的判定

本轮不要求完整工作空间覆盖。

只需同时满足：

## Teacher 区域

- 至少 (80%) 的预选 safe voxels 能生成 Gold 或 Silver 标签；
- 最大单值连通 component 包含至少 (2{,}000) 个稀疏节点；
- multi-parent consistency P95 (\le0.5^\circ)；
- Reject/conflict 区域被排除。

## Student 随机点

$$
FK_{\mathrm{P95}}\le5\text{mm},
$$

$$
FK_{\max}\le10\text{mm}.
$$

## Student 未见轨迹

至少：

$$
16/20
$$

条未见轨迹通过。

## 当前能力保持

当前 8 条 final ellipse 不出现明显退化。

如果满足以上条件，就已经能够声称：

> Student 不仅拟合了一组椭圆 family，而是在一个三维任务空间区域内近似了 Teacher 定义的 canonical inverse mapping，并能够跟踪区域内多种未见轨迹。

---

# 十五、对比实验的科学价值

至少保留三个对照。

## C0：当前 D3 曲线数据

验证当前 narrow corridor 表现。

## C1：相同样本量的 region-grown 数据

例如都使用：

$$
93{,}600
$$

行。

这样可以判断提升来自数据几何分布，而不是纯粹样本数。

## C2：D3 + region-grown 混合数据

判断是否能同时保持椭圆表现和区域泛化。

可选增加一个小规模失败基线：

## C3：随机 (xyz) + 独立 pointwise IK

只生成 (20{,}000) 行，用于展示独立 IK 数据更容易产生 local beta conflicts，不需要为它投入完整训练预算。

---

# 十六、Codex 实现阶段

建议输出目录：

```text
runs/bacra_v12_14_region_grown_dataset/
  00_protocol/
  01_seed_cleanup/
  02_capability_region/
  03_sparse_region_growth/
  04_consistency_audit/
  05_dense_region/
  06_datasets/
  07_students/
  08_random_point_test/
  09_unseen_trajectory_test/
  10_active_enrichment/
  11_summary/
```

建议新增脚本：

```text
scripts/analysis/run_bacra_v12_14_region_growth.py
scripts/analysis/train_bacra_v12_14_students.py
scripts/analysis/evaluate_bacra_v12_14_region.py
```

核心模块：

```text
src/quasi_exp/teacher/seed_canonicalization.py
src/quasi_exp/teacher/task_region_growth.py
src/quasi_exp/teacher/multiparent_continuation.py
src/quasi_exp/teacher/region_densification.py
src/quasi_exp/teacher/region_path_generation.py
```

Runner stages：

```text
seed_cleanup
capability_region
sparse_growth
consistency
dense_generation
dataset_assemble
train
random_point_test
trajectory_test
active_enrich
all
```

---

# 十七、Codex 必须输出的产物

```text
01_seed_cleanup/
  duplicate_curve_report.csv
  duplicate_label_report.csv
  canonical_seed_dataset.parquet

02_capability_region/
  voxel_map.parquet
  connected_components.csv
  selected_region.json

03_sparse_region_growth/
  sparse_nodes.parquet
  sparse_edges.parquet
  multiparent_candidates.parquet
  rejected_nodes.parquet
  growth_report.json

04_consistency_audit/
  two_path_consistency.csv
  loop_consistency.csv
  conflict_voxels.parquet

05_dense_region/
  dense_region_50k.parquet
  dense_region_100k.parquet
  dense_acceptance_report.json

06_datasets/
  R0_manifest.json
  R1_manifest.json
  R2_manifest.json
  spatial_split_manifest.csv

07_students/
  metrics_per_seed.csv
  selected_model.json

08_random_point_test/
  random_targets.parquet
  random_point_metrics.csv
  error_vs_support.csv

09_unseen_trajectory_test/
  trajectory_catalog.csv
  teacher_references.parquet
  student_tracking.parquet
  trajectory_metrics.csv
  plots/

10_active_enrichment/
  high_error_voxels.csv
  enrichment_dataset.parquet
  retrained_metrics.csv

11_summary/
  dataset_comparison.csv
  coverage_report.csv
  student_generalization_summary.csv
  final_recommendation.md
```

---

# 十八、可直接交给 Codex 的主任务

> Implement BACRA V12.14 as a region-first canonical dataset experiment. Stop using ellipse parameters as the primary training-data coordinates. Begin from the cleaned and deduplicated chart-A labels already validated in V12.13, build a connected capability-supported Cartesian voxel region around them, and sample an unordered three-dimensional task-space point cloud. Label each new point through short local Teacher continuations from at least three independently labeled neighboring anchors. Accept a point only when the exact FK corrector succeeds and the resulting beta candidates agree within the registered local consistency threshold; exclude branch-conflicting or difficult voxels rather than creating a second chart in this experiment. Grow the largest consistent connected component, then densify it using local tetrahedral beta interpolation or multi-anchor weighted-Jacobian prediction followed by exact correction and independent-anchor agreement. Remove exact task-space duplicates and select a single canonical label rather than averaging incompatible beta labels. Assemble a pure region dataset and a mixed dataset dominated by uniformly sampled Cartesian region points, train the currently selected bounded MLP with FK and row-space loss without changing the architecture, and compare it against the existing D3 curve-corridor dataset at equal sample counts. Evaluate on 10,000 held-out Cartesian points and on unseen ellipses, circles, Lissajous curves, closed 3-D B-splines, and open smooth inspection paths contained inside the accepted chart. Perform at most one active enrichment round in high-error but Teacher-consistent voxels. The experiment succeeds when the region-grown Student maintains the existing ellipse performance while achieving the registered FK error on most held-out points and at least sixteen of twenty unseen trajectories.

---

# 最终结论

当前最值得做的变化不是继续扩展椭圆参数范围，也不是继续沿某一条椭圆增加 tube 密度，而是：

$$
\boxed{
\text{从“轨迹 family 数据集”切换到“连通任务空间区域数据集”}
}
$$

Teacher 保持一致性的关键不是椭圆，而是：

1. 使用当前成功 chart 作为锚点；
2. 只做短距离局部 continuation；
3. 对同一点使用多个父节点独立标注；
4. 只有多个路径给出一致 (\beta) 时才接受；
5. 遇到冲突或困难区域直接停止扩展；
6. 在已经确认单值的 chart 内再进行大规模稠密采样。

这样既利用了当前已经成功的 Teacher–Student 路线，又能让最终 Student 的能力从“拟合一组相近椭圆”提升为“拟合一块真实三维任务区域中的 canonical inverse mapping”。

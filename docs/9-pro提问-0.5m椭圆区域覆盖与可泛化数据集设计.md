---
question_id: Q09
question_number: 9
question_confirmed_by_user: true
question_confirmation_summary: "判断当前泛化证据是否不足，并围绕实际长半轴0.5m椭圆定义区域、设计teacher采样、覆盖与标签质量指标、整轨迹隔离split、Gate、停止条件和分阶段数据集生成计划；0.75m仅作为压力测试证据。"
date: "2026-07-20"
status: ready-to-send
project: "quasi_exp"
evidence_cutoff: "2026-07-20T20:15:31+08:00"
source_snapshot: "主工作区 canonical-layer-field-u3@740d8c0（dirty，既有用户文件未改动）；V10证据工作树 codex/trajectory-canonical-teacher-v10@7a73f2acbdc8c90f7872bb2b20691d5a65abe434（clean）"
---

# GPT-5 Pro 第九次交接：0.5 m 椭圆区域覆盖与可泛化数据集设计

## 0. 经用户确认的 GPT-5 Pro 任务

用户已于 2026-07-20 明确确认：基于最新 V10 实验，审计当前结果的泛化证据边界，并围绕实际三维长半轴为 `0.5 m` 的椭圆设计下一阶段可泛化数据集。回答范围严格限定为以下八项：

1. 判断“当前泛化证据不足”是否成立，并划清现有结果允许与不允许支持的结论。
2. 定义 0.5 m 椭圆“对应区域”的数学范围和覆盖对象。
3. 设计可泛化数据集的 teacher 标签生成与采样方法。
4. 定义空间覆盖、标签连续性、多值性、关节余量和可学习性指标。
5. 设计按整条轨迹、姿态、family 隔离的 IID、interpolation、OOD split。
6. 给出数据准入 Gate、失败回退策略和停止条件。
7. 给出由小规模 pilot 到正式数据集的可执行分阶段计划。
8. 说明 0.75 m 结果应如何作为压力测试证据。

本次没有要求设计 1.0 m 数据集，也没有授权把现有某条候选路线预设为答案。

## 1. 当前有效事实与最短摘要

### 1.1 证据截止时间

本交接只使用截至 `2026-07-20T20:15:31+08:00` 已存在并重新核对的项目 artifact、配置、代码和实验记录。旧 GPT 回答只作为历史背景，不作为实验事实。

### 1.2 当前有效事实

1. V10 已把挑战尺度重新定义为实际三维长半轴。0.5 m 固定 pose 的 T3 teacher 在 720 个相位上得到低残差闭环：残差 P95/max 为 `0.007080/0.153161 mm`，但最小关节余量为 `1.470676°`，仍低于既有 strict centerline Gate 的 `1.5°`。
2. 0.5 m student 数据实际只有一条固定中心、固定平面、固定长短轴比、固定 family 的椭圆中心线：`720` 行、`1` 个 trajectory、`1` 个 family、`1` 个 chart、`1` 个 branch；所有 `tube_n1_mm=tube_n2_mm=0`。没有生成该椭圆周围的 tube、体积邻域、中心扰动、姿态扰动或 family 扰动数据。
3. 0.5 m 数据的 train/validation/sealed-test 是同一条闭合曲线上的交错相位切分：`360/180/180`。全标签部署重拟合后的 `1440` 个测试点是同一条椭圆 2880 网格中的奇数相位，因此它验证的是同轨迹相位插值，不是跨轨迹或区域泛化。
4. 0.5-only 部署模型在这 1440 个同轨迹新相位上的 P95/max 末端误差为 `4.607734/4.979476 mm`，关节界内率为 `1.0`。按后来定义的 `max error <= 2% × major semiaxis` 应用级 Gate，0.5 m 阈值为 `10 mm`，该结果通过；按独立报告的 `3 mm` 点阈值，成功率为 `0.715278`。
5. joint 模型并未给出稳定的跨尺度正迁移：部署阶段 joint 模型在 0.5 m 的 max 为 `12.329131 mm`，超过 10 mm；在 0.75 m 的 max 为 `12.765676 mm`，低于 15 mm。五 seed sealed test 中，0.75 m joint 只有 `2/5` seed 通过相对 Gate，0.75-only 为 `0/5`。
6. 0.75 m dense teacher 有 `698/720` 个训练合格标签，teacher 残差 P95/max 为 `0.915796/27.293174 mm`，最小关节余量为 `0°`；22 个不合格点不进入训练，但仍保留在整环模型评估中。

### 1.3 关键边界

1. 现有结果可以支持“固定 pose、固定 family、单条中心线上的相位插值已经取得较低误差”；不能据此支持“0.5 m 椭圆邻域或工作空间区域内已经泛化”。
2. “当前泛化能力差”尚未被区域 OOD 实验直接证明；已被直接证明的是当前泛化证据范围很窄，且简单 joint 训练存在负迁移和 seed 稳定性问题。
3. V10.3 strict teacher/data 协议下，0.5 m centerline 因最小关节余量不足而 fail-closed，因此没有运行正式 `3×3, ±1 mm` tube；后续 student 实验采用的是“数值有限、关节界内、teacher FK 残差不超过 3 mm”的标签资格规则，不等于 strict dataset Gate 已通过。
4. `2%` 相对 tracking Gate 是在原 V10 student 结果已经存在之后新增并重评的应用级口径。本交接将其标为 post-hoc，不把相对 Gate 通过写成预注册 formal claim。

## 2. 项目目标与本轮边界

### 2.1 用户给定的项目目标

项目总体目标是依据原论文和项目力学约束复现连续体机器人数据集，并让机器学习模型在具有说服力的数据覆盖和物理约束下完成椭圆轨迹逆映射与跟踪。本轮用户把优先级限定为：先围绕实际长半轴 0.5 m 的椭圆建立可解释、可审计、可泛化的数据空间，再讨论训练说服力。

### 2.2 当前实际覆盖范围

- 准静态虚拟机器人；机器人配置为 `robot_rods_only_standard_100k.yaml`。
- 标准 beta6 关节域为 `±5°, ±5°, ±10°, ±10°, ±15°, ±15°`。
- 0.5 m 与 0.75 m 各自只有一个优化 pose、一条固定比例的三维椭圆中心线和一个 T3 dense teacher traversal。
- student 输入是目标位置，输出为有界 beta6；已比较静态 MLP、带可微 FK loss 的静态 MLP和自回归 GRU。
- 当前 student 训练没有 tube 数据、硬件动力学、真实机器人测量、DLS/IK 在线后处理或姿态重搜索。

### 2.3 当前证据未覆盖的内容

- 0.5 m 中心线周围法平面 tube、三维局部体积或任意明确区域的覆盖。
- 不同中心、轨迹平面、朝向、长短轴比、尺度、trajectory family 和 canonical branch 的整轨迹留出泛化。
- 多值逆解在空间重叠区的系统审计，以及不同 branch 能否安全合并为静态 `xyz -> beta6` 标签。
- 训练集覆盖度与模型误差之间的因果关系。
- 动力学、张力、摩擦、制造误差、测量噪声和真实硬件泛化。

## 3. 相对上次材料新增的事实

本节所称“上次材料”是 Q08。

| 类型 | 已发生的变化 | 实际证据 | 证据等级 |
|---|---|---|---|
| 尺度定义 | 挑战对象从旧 `R=75–105 mm` 改为实际长半轴 `0.5/0.75/1.0 m` | `v10_teacher_checkpoint.md`、`v10_challenge_report.json` | V10.3 formal protocol 内的 teacher/challenge 事实 |
| 0.5 m teacher | 新 pose 上得到 720 相位 T3 dense teacher，720/720 满足 student 标签资格 | `v10_teacher_summary.json`、`v10_0p5m_dense_teacher.parquet` | student-training artifact；不等同 strict dataset pass |
| 0.75 m teacher | 720 相位中 698 个标签合格，22 个不合格点被隔离 | `v10_teacher_summary.json`、`v10_0p75m_dense_teacher.parquet` | student-training artifact / stress evidence |
| 模型 | 选择 `S1_l1_tanh`；0.5 m 专用部署在同中心线奇数相位上 max `4.979476 mm` | `v10_selection.json`、`v10_deployment_summary.json` | sealed selection + same-trajectory interpolation evidence |
| 稳定性 | 0.5 m 两个 scope 的 sealed test 均 5/5 seed 通过；0.75 m joint 2/5、0.75-only 0/5 | `v10_final_summary.json` | post-selection sealed-test evidence；相对 Gate 判定为 post-hoc |
| 负迁移 | 全标签重拟合时 joint 在 0.5 m 未通过 10 mm，而 0.5-only 通过 | `v10_deployment_summary.json` | deterministic deployment / post-hoc relative-Gate comparison |
| Gate 口径 | 新增 `T(a)=0.02a` 的最坏点 student tracking Gate | `v10_relative_tracking_gate.md` | post-hoc application-level reassessment |
| 完整性边界 | 18:59 的 `artifact_verification.json` 早于 19:02 重写的最终 summary/prediction/deployment artifact；旧 `verification_pass` 不覆盖最终字节 | 文件时间、旧记录内 SHA 与当前 SHA 对照 | evidence-integrity gap；本 Q09 bundle 重新计算自身 SHA |

## 4. 实验方法与复现口径

### 4.1 代码与工作树 fixed point

| 项目 | 值 |
|---|---|
| 项目根 | `/mnt/ML_projects/quasi_exp` |
| 主工作区 | `canonical-layer-field-u3@740d8c0`，dirty；存在用户既有修改和未跟踪文件，本次没有覆盖 |
| V10 证据工作树 | `/mnt/ML_projects/quasi_exp/.worktrees/trajectory-canonical-teacher-v10` |
| V10 分支 | `codex/trajectory-canonical-teacher-v10` |
| V10 HEAD | `7a73f2acbdc8c90f7872bb2b20691d5a65abe434` |
| V10 dirty 状态 | clean（证据截止时） |
| 关键历史 fixed points | student 初始运行记录 `06d1271`；deterministic deployment `fece63c`；相对 Gate `4338b3a`；legacy evidence 隔离 `7a73f2a` |
| 仅 checkout 当前 HEAD 是否足以复现 | 代码与配置可以定位；`runs/` artifact 不在提交中，必须连同本交接附件核验；已删除的临时重评脚本不能从当前 HEAD 直接调用 |

### 4.2 V10 teacher 方法与参数

| 参数 | 实际值 | 来源 |
|---|---:|---|
| challenge definition | actual 3D major semiaxis | `v10_config.yaml` |
| 0.5 m minor semiaxis | `0.168666975 m` | `v10_teacher_summary.json` |
| 0.75 m minor semiaxis | `0.253000462 m` | `v10_teacher_summary.json` |
| standard beta6 limits | `±5°, ±5°, ±10°, ±10°, ±15°, ±15°` | `v10_teacher_checkpoint.md` |
| pose search | 24 phases, 3 restarts, joint-margin-aware objective | `v10_config.yaml` |
| reachability atlas | 131,072 Sobol + zero configuration = 131,073 rows | `v10_teacher_checkpoint.md` |
| dense teacher | T3, 720 phases/scale | `v10_teacher_summary.json` |
| strict centerline residual Gate | P95 ≤1 mm; max ≤3 mm | `v10_config.yaml` |
| strict centerline joint margin Gate | min ≥1.5° | `v10_config.yaml` |
| formal tube definition | offsets `[-1,0,1] mm × [-1,0,1] mm` | `v10_config.yaml` |
| student label eligibility | finite, within joint bounds, teacher residual ≤3 mm | `v10_student_experiment_report.md` |

0.5 m 优化 pose：

```text
center = [1.0616378098, -0.0471136671, -0.0545792158] m
major_direction = [0.0348533993, 0.9043158939, 0.4254386027]
minor_direction = [0.1436338845, -0.4258103730, 0.8933391481]
major/minor semiaxes = 0.5 / 0.168666975 m
```

0.75 m 优化 pose：

```text
center = [0.8774353980, 0.0030441281, -0.0457380223] m
major_direction = [-0.0585162938, -0.7905498679, -0.6095955625]
minor_direction = [0.0401478710, -0.6120115267, 0.7898291205]
major/minor semiaxes = 0.75 / 0.253000462 m
```

### 4.3 student 模型、数据生成与划分

每个尺度重新生成 720 相位 T3 teacher。`label_eligible` 只由标签资格规则决定；`used_for_training` 在初始 split 中仅标记训练相位。原始 Parquet 已完整上传，没有用抽样文件替代。

| 项目 | 0.5 m | 0.75 m |
|---|---:|---:|
| 总行数 | 720 | 720 |
| trajectory / family / chart / branch | 1 / 1 / 1 / 1 | 1 / 1 / 1 / 1 |
| tube offset 组合 | 1，且仅 `(0,0) mm` | 1，且仅 `(0,0) mm` |
| 标签合格 | 720 | 698 |
| train split | 360，其中合格 360 | 360，其中合格 349 |
| validation split | 180，其中合格 180 | 180，其中合格 174 |
| sealed test split | 180，其中合格 180 | 180，其中合格 175 |
| split 规则 | 偶数 phase train；`phase_idx % 4 == 1` validation；`%4 == 3` sealed test | 同左 |

候选模型：

- `S0`：静态 MLP，关节空间监督；
- `S1`：静态 MLP，关节空间监督 + 可微 FK loss；
- `S3`：GRU，自回归 beta 预测；
- identity / `tanh` 输出和 FK loss 权重 `0.1/1.0` 的组合。

只使用 0.5 m validation 和三个 screen seeds 选择候选；锁定 `S1_l1_tanh` 后才打开 sealed test，并以五个新 seeds 评估。0.75 m 标签不参与候选选择。部署模型用各 scope 的全部合格标签重拟合，然后在同一椭圆 2880 网格中未作为 teacher 标签的 1440 个奇数相位上评估。

### 4.4 baseline、评价和 gate

| 对象 | 实际定义 | 证据身份 |
|---|---|---|
| strict teacher centerline | residual、相邻 beta、加速度、seam、joint margin 全部通过才准入 | V10.3 formal |
| strict tube | 只有 centerline formal pass 才运行 `3×3, ±1 mm` tube | V10.3 formal；0.5/0.75 m 均未运行 |
| student selection | 只看 0.5 m validation，sealed test 选择后打开 | protocol v10.1 selection evidence |
| student metrics | FK residual P50/P95/max、3 mm success rate、joint-bounds rate | direct artifact metrics |
| relative student Gate | `max_i e_i <= 0.02a`；0.5/0.75 m 阈值 10/15 mm | post-hoc application-level gate |
| deterministic deployment | fixed seed，全部合格 dense labels 重拟合，同曲线奇数相位评估 | deterministic same-trajectory interpolation evidence |

### 4.5 当前代码入口与产物

当前 clean V10 HEAD 中的主要入口为：

```text
PYTHONPATH=src /mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python \
  scripts/analysis/run_large_scale_dense_teacher_v10.py \
  --config configs/large_scale_ellipse_challenge_v10.yaml \
  --robot-config /mnt/ML_projects/quasi_exp/configs/robot_rods_only_standard_100k.yaml \
  --output-root /mnt/ML_projects/quasi_exp/runs/trajectory_canonical_teacher_v10/08_large_scale_student_tracking

PYTHONPATH=src /mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python \
  scripts/analysis/train_large_scale_students_v10.py \
  --stage all \
  --robot-config /mnt/ML_projects/quasi_exp/configs/robot_rods_only_standard_100k.yaml \
  --output-root /mnt/ML_projects/quasi_exp/runs/trajectory_canonical_teacher_v10/08_large_scale_student_tracking

PYTHONPATH=src MPLCONFIGDIR=/tmp/quasi-exp-v10-mpl \
  /mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python \
  scripts/analysis/deploy_large_scale_students_v10.py \
  --challenge-config configs/large_scale_ellipse_challenge_v10.yaml \
  --robot-config /mnt/ML_projects/quasi_exp/configs/robot_rods_only_standard_100k.yaml \
  --output-root /mnt/ML_projects/quasi_exp/runs/trajectory_canonical_teacher_v10/08_large_scale_student_tracking
```

这些是当前 CLI 入口，不声称上述三条命令单独重建了已经存在的全部 V10 历史中间态；完整重放还受 `runs/07_large_scale_challenge` 初始化 artifact 和 fixed seed 环境约束。

## 5. 实验结果

| Fact ID | 已观察结果 | 数值 | 证据等级 | 实际附件/JSON key | 适用边界 |
|---|---|---:|---|---|---|
| F1 | 0.5 m formal T3 180-phase centerline 精确且平滑，但 strict Gate 失败 | residual P95/max `0.050025/0.338349 mm`; margin `1.432188° < 1.5°` | formal | `v10_challenge_report.json / challenges[0].formal_teachers.T3.centerline` | teacher centerline；没有 tube/data pass |
| F2 | 0.5 m dense teacher 标签质量 | 720/720 eligible；P95/max `0.007080/0.153161 mm`; margin `1.470676°` | student-training artifact | `v10_teacher_summary.json / reports[0]` | 单 pose、单中心线 |
| F3 | 0.5 m 数据覆盖结构 | 720 行；1 trajectory/family/chart/branch；唯一 tube offset `(0,0)` | raw-data direct audit | `v10_0p5m_dense_teacher.parquet` | 不能代表区域覆盖 |
| F4 | 0.5-only sealed test 稳定性 | P95 median `2.574064 mm`; max median `2.809890 mm`; relative-Gate 5/5 | sealed test；relative Gate post-hoc | `v10_final_summary.json / records[scale0p5,a0p500m]` | 同轨迹交错相位 test |
| F5 | 0.5-only 全标签部署 | P95/max `4.607734/4.979476 mm`; 3 mm success `0.715278`; bounds `1.0` | deterministic deployment；relative Gate post-hoc | `v10_deployment_summary.json / reports[scale0p5].scales.a0p500m` | 同一固定椭圆的 1440 奇数相位 |
| F6 | joint 在 0.5 m 出现负迁移 | P95/max `11.946357/12.329131 mm`，超过 10 mm | deterministic comparison；post-hoc Gate | `v10_deployment_summary.json / reports[joint].scales.a0p500m` | 只比较当前两尺度 mixture |
| F7 | 0.75 m dense teacher 存在尾部失败 | 698/720 eligible；P95/max `0.915796/27.293174 mm`; margin `0°` | stress evidence | `v10_teacher_summary.json / reports[1]` | 不代表 0.75 m 完整合格数据集 |
| F8 | 0.75 m seed 稳定性不足 | joint 2/5、0.75-only 0/5 relative-Gate pass | sealed stress test；relative Gate post-hoc | `v10_final_summary.json` | 可作为压力测试，不可外推 0.5 m 区域性能 |
| F9 | 0.75 m deployment 单 seed 可过相对 Gate | joint max `12.765676 mm`; 0.75-only max `10.650252 mm` | deterministic deployment；post-hoc Gate | `v10_deployment_summary.json` | 单固定 pose、单中心线、单部署 seed |

### 5.1 已通过的内容

- 0.5 m 的 T3 teacher 在固定 pose 中给出低残差、平滑、闭环的中心线标签；失败项集中在 strict joint-margin Gate。
- 候选选择与 0.75 m stress labels 隔离；sealed test 在选择锁定后打开。
- 0.5-only 模型在同一中心线的新相位上取得低于 5 mm 的 P95 和最坏点误差，且输出全部在关节界内。
- 固定 deployment seed 的预测可重复；当前 bundle 会对最终上传字节重新计算 SHA。

### 5.2 失败与反例

- 0.5 m strict centerline Gate 未通过，因此不存在 formal tube pass 或 formal dataset pass。
- 0.5 m joint 部署比 0.5-only 更差，并在 2% 相对 Gate 下失败；“增加另一个尺度数据必然提升泛化”被当前结果反驳。
- 0.75 m 只有 698 个合格 teacher 标签，且五 seed sealed test 不稳定；单 deployment seed 通过不能覆盖该不稳定性。
- S3 GRU 在无 teacher 注入的自回归 rollout 中发生严重误差累积，8 个切口/方向的最坏 P95 为 324–426 mm；当前实现不可用。

## 6. 证据等级、失败结果与未知事项

### 6.1 正式事实

- V10.3 teacher challenge 的几何、centerline、joint margin 和 fail-closed tube 状态属于其冻结协议范围内的正式事实。
- 0.5 m formal T3 centerline 没有通过 joint-margin Gate，0.75 m coarse screen 没有通过 hard Gate；两者都没有运行正式 tube。
- v10.1 候选选择在 0.75 m stress evaluation 和 sealed test 打开前锁定。

### 6.2 非正式、post-hoc 和压力测试证据

| 内容 | 等级 | 能支持什么 | 不能支持什么 |
|---|---|---|---|
| 2% 长半轴相对 Gate | post-hoc application-level | 用统一相对口径重述已有 student tracking 误差 | 不能改写 strict teacher/data Gate，也不是预注册成功证据 |
| 1440 odd-phase deployment | same-trajectory interpolation evidence | 检查同一解析曲线上训练标签间的插值和固定 seed 部署 | 不能支持 tube、pose、family 或 workspace OOD 泛化 |
| 0.75 m joint/scale-only | post-selection stress evidence | 暴露 seed 敏感、标签缺口和 joint 负迁移 | 不能直接定义 0.5 m 区域，也不能替代 0.5 m OOD split |
| 0.5/0.75 tracking PNG | visualization-only | 直观看轨迹形状和相位误差分布 | 不替代数值报告和 raw data |

### 6.3 已发现的实现或证据缺口

- 当前 dataset schema 已有 `tube_n1_mm`、`tube_n2_mm`、`chart_id`、`branch_id`、`teacher_policy_id` 等字段，但本轮实际 0.5 m 数据只填充一个零偏移、一个 chart 和一个 branch，尚未形成区域覆盖。
- 当前 split 是同一轨迹交错相位 split；没有 trajectory-group、pose-group 或 family-group 的 sealed OOD split。
- strict teacher/data Gate 和 student label eligibility 是两套不同口径；现有报告同时使用两者，若不分开容易误读为 strict dataset 已通过。
- `artifact_verification.json` 的验证时间早于最终 post-hoc 重评文件；不能把其中的 `verification_pass` 作为当前最终字节的完整性证明。

### 6.4 尚未知或尚未执行

- “0.5 m 椭圆对应区域”应该是法平面 tube、轨迹参数邻域、可达空间壳层，还是这些对象的分层组合，尚未定义。
- 在保持 canonical branch 连续与安全余量时，0.5 m 周围实际可覆盖多大的 tube、中心、姿态和形状参数范围，尚未测量。
- 区域内存在多少可行逆分支、分支交叉和不可避免的多值区，尚未审计。
- 哪种覆盖指标与 student 的整轨迹 OOD 误差最相关，尚无实验结论。
- 当前静态 `xyz -> beta6` 模型是否足以覆盖多 pose/multi-family 区域，尚未通过 group-held-out 对照验证。

### 6.5 当前不能声称

- 不能声称当前模型已经学会 0.5 m 椭圆周围区域的逆映射。
- 不能声称 720 个相位或 1440 个奇数相位构成充分的三维工作空间覆盖。
- 不能声称当前模型泛化能力已经被证明很差；区域泛化尚未被执行。更准确的当前结论是“泛化证据不足”。
- 不能声称 0.5 m strict dataset Gate 已通过，或已经存在 formal tube 数据。
- 不能根据 0.75 m 单 seed deployment 通过声称跨尺度泛化稳定。

## 7. 决策所需的客观对照

本节只比较已经执行的对象，不预设 GPT-5 Pro 应选择的下一条路线。

| 对照对象 | 已执行输入/方法 | 已观察结果 | 证据等级 | 已知限制 |
|---|---|---|---|---|
| 0.5-only | 720 个同中心线合格标签；部署用全部标签 | 奇数相位 P95/max `4.608/4.979 mm` | deterministic interpolation / post-hoc Gate | 单 pose、单 curve、零 tube |
| joint 0.5+0.75 在 0.5 m | 720 + 698 个合格标签混合 | P95/max `11.946/12.329 mm` | deterministic comparison | 只有两个离散尺度，且 pose/family 同时变化；不能单独归因 |
| joint 0.5+0.75 在 0.75 m | 同上 | deployment max `12.766 mm`，但 sealed seeds 2/5 通过 | post-selection stress | 单 deployment seed 与五 seed 结论不同 |
| 0.75-only | 698 个合格标签，22 个困难相位隔离 | deployment max `10.650 mm`；sealed seeds 0/5 通过 | post-selection stress | teacher 尾部失败、margin 0° |
| S3 GRU | 多 cut、多方向窗口，自回归 rollout | 最坏 P95 324–426 mm | diagnostic failure | 不能作为当前可用 stateful baseline |
| strict V10.3 tube | 仅 centerline formal pass 后运行 `3×3, ±1 mm` | 0.5/0.75 m 都未运行 | formal fail-closed | 尚无区域数据可评价 |

## 8. 经用户确认的问题

1. “当前泛化证据不足”这一判断是否成立？请划清现有结果允许支持和不允许支持的结论。
2. 应如何定义实际长半轴 0.5 m 椭圆“对应区域”的数学范围和覆盖对象？
3. 应如何生成并采样可泛化数据集的 teacher 标签？
4. 应如何定义空间覆盖、标签连续性、多值性、关节余量和可学习性指标？
5. 应如何设计按整条轨迹、姿态和 family 隔离的 IID、interpolation 与 OOD split？
6. 应设置哪些数据准入 Gate、失败回退策略和停止条件？
7. 应如何把方案组织为从小规模 pilot 到正式数据集的可执行分阶段计划？
8. 0.75 m 的 teacher 缺口、seed 不稳定和 joint/scale-only 结果应如何作为压力测试证据？

## 9. 经用户确认的回答要求

请给出一份完整回答，逐项覆盖第 8 节的八个问题：先审计当前泛化证据边界；再定义 0.5 m 椭圆对应区域；设计 teacher 标签生成与采样；给出空间覆盖、标签连续性、多值性、关节余量和可学习性指标；设计按整条轨迹、姿态、family 隔离的 IID/interpolation/OOD split；给出准入 Gate、失败回退和停止条件；形成从 pilot 到正式数据集的可执行分阶段计划；最后说明 0.75 m 如何仅作为压力测试证据使用。

## 10. 附件清单与复现信息

GPT-5 Pro 实际可见的是“上传文件名”列。原始 SSH 路径只用于 provenance。

GPT 可见文件数为 `14`：`QUESTION.md + 13` 个实际证据附件。低于 15 个安全阈值，采用逐文件上传；不生成 ZIP。

| 上传文件名 | 原始 SSH 路径 | 大小 | 证据等级 | 支持事实 | 上传理由 |
|---|---|---:|---|---|---|
| `v10_student_experiment_report.md` | `/mnt/ML_projects/quasi_exp/runs/trajectory_canonical_teacher_v10/08_large_scale_student_tracking/EXPERIMENT_REPORT.md` | 5,766 B | mixed summary | F2–F9 | 最短阅读入口，包含 student 协议、split、模型和结果 |
| `v10_teacher_checkpoint.md` | `/mnt/ML_projects/quasi_exp/.worktrees/trajectory-canonical-teacher-v10/docs/checkpoints/2026-07-20-large-scale-ellipse-challenge-v10.md` | 5,855 B | formal checkpoint | F1、strict Gate 边界 | 核验 0.5/0.75 m teacher 与 tube early-stop |
| `v10_relative_tracking_gate.md` | `/mnt/ML_projects/quasi_exp/.worktrees/trajectory-canonical-teacher-v10/docs/TrackingErrorRelativeGateV10.md` | 1,323 B | post-hoc gate definition | F4–F9 | 核验 2% 相对 Gate 的数学定义和适用范围 |
| `v10_teacher_summary.json` | `/mnt/ML_projects/quasi_exp/runs/trajectory_canonical_teacher_v10/08_large_scale_student_tracking/teacher_summary.json` | 5,837 B | machine-readable teacher summary | F2、F7 | 核验 phase、eligible、residual、margin、split hash |
| `v10_selection.json` | `/mnt/ML_projects/quasi_exp/runs/trajectory_canonical_teacher_v10/08_large_scale_student_tracking/screen/selection.json` | 534 B | selection-lock evidence | F4 | 核验只用 0.5 m validation 选择且 stress 未参与 |
| `v10_final_summary.json` | `/mnt/ML_projects/quasi_exp/runs/trajectory_canonical_teacher_v10/08_large_scale_student_tracking/final/final_summary.json` | 1,311 B | sealed-test summary | F4、F8 | 核验五 seed 中位数与稳定性 |
| `v10_deployment_summary.json` | `/mnt/ML_projects/quasi_exp/runs/trajectory_canonical_teacher_v10/08_large_scale_student_tracking/deployment/deployment_summary.json` | 7,097 B | deterministic deployment summary | F5、F6、F9 | 核验 1440 奇数相位、模型 scope 和误差 |
| `v10_challenge_report.json` | `/mnt/ML_projects/quasi_exp/runs/trajectory_canonical_teacher_v10/07_large_scale_challenge/report.json` | 22,492 B | formal machine-readable teacher report | F1 | 核验 pose、centerline checks 和 fail-closed 状态 |
| `v10_config.yaml` | `/mnt/ML_projects/quasi_exp/.worktrees/trajectory-canonical-teacher-v10/configs/large_scale_ellipse_challenge_v10.yaml` | 1,377 B | frozen config | F1、Gate/teacher 参数 | 核验尺度、atlas、pose search、centerline/tube Gate |
| `v10_0p5m_dense_teacher.parquet` | `/mnt/ML_projects/quasi_exp/runs/trajectory_canonical_teacher_v10/08_large_scale_student_tracking/a0p500m/teacher_dense/dense_teacher.parquet` | 478,717 B | full raw 0.5 m labels | F2、F3 | 允许独立核验 schema、唯一 offset、branch/chart 和标签分布 |
| `v10_0p75m_dense_teacher.parquet` | `/mnt/ML_projects/quasi_exp/runs/trajectory_canonical_teacher_v10/08_large_scale_student_tracking/a0p750m/teacher_dense/dense_teacher.parquet` | 476,286 B | full raw stress labels | F7、F8 | 允许独立检查 22 个不合格点和边界尾部 |
| `v10_0p5m_tracking.png` | `/mnt/ML_projects/quasi_exp/runs/trajectory_canonical_teacher_v10/08_large_scale_student_tracking/deployment/scale0p5/a0p500m_tracking.png` | 339,179 B | visualization-only | F5 | 展示 0.5-only 同中心线跟踪与相位误差 |
| `v10_0p75m_joint_tracking.png` | `/mnt/ML_projects/quasi_exp/runs/trajectory_canonical_teacher_v10/08_large_scale_student_tracking/deployment/joint/a0p750m_tracking.png` | 393,287 B | visualization-only / stress | F9 | 展示 joint 在 0.75 m 的固定轨迹压力测试 |

### 未上传的大文件或派生数据说明

- 两个 dense teacher Parquet 都是完整 720 行原始文件，不是抽样或派生替代品；其源 SHA 将由 bundle manifest 重新记录。
- 未上传模型 checkpoint、训练中间目录、所有 seed prediction、完整 reachability atlas 和重复图片，因为它们不是回答八个确认问题的最小必要证据。
- 正文中的 trajectory/family/chart/branch/offset 计数和目标坐标范围是从已上传的完整 Parquet 直接只读统计；没有另行生成摘要文件，也没有删减原始行。
- 没有上传已过期的 `artifact_verification.json`；其时间与 SHA 不覆盖 19:02 后最终文件，这一缺口已在正文披露。bundle 的 `manifest.json` 将对实际上传副本重新计算 SHA256。

## 11. 给下一位讨论者的最短事实摘要

1. 0.5-only student 在固定 0.5 m 椭圆同轨迹奇数相位上的 P95/max 为 `4.608/4.979 mm`，但这只是同曲线插值。
2. 0.5 m 训练数据只有 720 个中心线相位、单 trajectory/family/chart/branch，tube offset 只有 `(0,0)`；没有区域覆盖证据。
3. strict V10.3 teacher/data Gate 因 joint margin 未通过而没有运行正式 tube；student 标签资格不能替代 strict dataset pass。
4. joint 在 0.5 m 发生负迁移；0.75 m teacher 有 22 个不合格点且五 seed 稳定性不足，可作为压力测试事实。
5. 用户确认的问题是：判断泛化证据边界，定义 0.5 m 区域，设计 teacher 采样、覆盖与标签质量指标、整轨迹/姿态/family 隔离 split、Gate/回退/停止条件和 pilot-to-formal 数据集计划，并说明 0.75 m 的压力测试角色。

---
# 结论

当前结果已经证明：**在固定中心、固定平面、固定长短轴比的一条 (0.5,\mathrm m) 椭圆中心线上，慢速 T3 teacher 能生成精确、平滑、闭合的 (\beta_6) 标签，学生模型也能完成同一曲线上的相位插值。** 但当前数据只有 (720) 行、一个 trajectory、一个 family、一个 chart、一个 branch，而且所有 tube offset 都是 ((0,0))；train、validation、test 也只是同一条曲线上的交错相位。因此，它本质上还是一个**一维曲线插值数据集**，没有形成任何可称为“空间区域”的训练覆盖。

现有结果不能说明模型区域泛化能力差，只能说明：

> **目前没有区域泛化证据。**

0.5-only 模型在同一椭圆奇数相位上的 P95/max 为 (4.608/4.979,\mathrm{mm})，这是有效的同轨迹插值结果，但不能支持 tube、不同中心、不同椭圆平面或不同 family 的泛化结论。

下一阶段最推荐的目标不是随意增加更多椭圆，而是依次构造：

$$
\boxed{
\text{0.5 m鲁棒中心线}
\rightarrow
\text{三维实心canonical tube}
\rightarrow
\text{邻近椭圆family集合}
\rightarrow
\text{跨family一致性审计}
\rightarrow
\text{整轨迹隔离训练与测试}
}
$$

建议将实验命名为：

```text
Generalized 0.5 m Canonical Region Dataset V11
```

---

# 一、先划清当前实验允许支持的结论

## 已经成立

当前证据支持：

1. 在已优化的固定 pose 下，(0.5,\mathrm m) 椭圆中心线逐点可达；
2. T3 teacher 的标签残差、闭环性和轨迹平滑性很好；
3. 静态 MLP 加 FK loss 能学习该中心线上的映射；
4. 在同一条解析曲线的未采样相位上，模型能实现约 (5,\mathrm{mm}) 以内的最坏点误差。

当前 dense teacher 的残差 P95/max 为 (0.0071/0.1532,\mathrm{mm})，且相邻 (\beta) RMS p95 仅约 (0.067^\circ)，说明中心线标签本身已经非常平滑。

## 尚未成立

当前不能声称：

* 模型学习了 (0.5,\mathrm m) 椭圆附近的三维区域；
* 模型能泛化到新的中心、平面、朝向或长短轴比；
* 一个 (xyz\to\beta_6) 静态函数能覆盖多个 family；
* 0.5 m strict dataset Gate 已经通过；
* 0.75 m 数据可以自然改善 0.5 m 模型。

0.5 m strict centerline Gate 因最小关节余量 (1.470676^\circ<1.5^\circ) 而 fail-closed，所以正式 (\pm1,\mathrm{mm}) tube 根本没有运行。

---

# 二、0.5 m 椭圆“对应区域”应该如何定义

应把区域分成两个层级。

## 2.1 第一层：中心线周围的三维实心 tube

当前椭圆写成：

$$
\gamma_0(\phi)
==============

c_0
+
a_0q_1\cos\phi
+
b_0q_2\sin\phi
$$

其中：

$$
a_0=0.5,\mathrm m
$$

$$
b_0=0.168666975,\mathrm m
$$

(q_1,q_2) 是当前优化椭圆平面内的单位长轴和短轴方向。

定义平面法向：

$$
n_\Pi=q_1\times q_2
$$

定义单位切向：

$$
\tau(\phi)
==========

\frac{\gamma_0'(\phi)}
{|\gamma_0'(\phi)|}
$$

定义椭圆平面内、垂直于轨迹切向的法向：

$$
n_r(\phi)=n_\Pi\times\tau(\phi)
$$

于是最自然的三维区域是：

$$
\Omega_{\mathrm{tube}}
======================

\left{
\gamma_0(\phi)
+
u,n_r(\phi)
+
v,n_\Pi
\right}
$$

并约束：

$$
\left(\frac{u}{\rho_r}\right)^2
+
\left(\frac{v}{\rho_\Pi}\right)^2
\le1
$$

这里：

* (\phi) 提供沿轨迹的一个维度；
* (u) 提供椭圆平面内法向厚度；
* (v) 提供椭圆平面外厚度。

因此：

$$
(\phi,u,v)\longrightarrow(x,y,z)
$$

恰好是一个三维任务区域，与模型的三维输入 (xyz) 维数匹配。

这应该是下一阶段最先建立的“区域泛化”对象。

---

## 2.2 第二层：邻近椭圆 family 的联合区域

仅有一条 tube 仍然只支持单一轨迹附近的任务。进一步定义邻近 family：

$$
\gamma_\xi(\phi)
================

c_\xi
+
a_\xi q_{1,\xi}\cos\phi
+
b_\xi q_{2,\xi}\sin\phi
$$

其中 family 参数可写为：

$$
\xi=
\left(
a,\frac ba,
\Delta c_1,\Delta c_2,\Delta c_3,
\omega_1,\omega_2
\right)
$$

分别表示：

* 实际长半轴 (a)；
* 长短轴比 (b/a)；
* 沿 (q_1,q_2,n_\Pi) 的中心平移；
* 椭圆平面的两个小角度倾斜。

联合区域为：

$$
\Omega_{\mathrm{family}}
========================

\bigcup_{\xi\in\mathcal F}
\Omega_{\mathrm{tube}}(\xi)
$$

但要注意：多个 family 的 tube 可能在同一个 (xyz) 附近产生不同 (\beta) 标签。因此，多 family 数据不能直接混合，必须先做 branch-conflict 审计。

---

# 三、第一步不是生成 tube，而是修复 0.5 m anchor 的安全余量

当前 0.5 m 中心线几乎通过 strict Gate，但最小关节余量只差约：

$$
1.5^\circ-1.470676^\circ
========================

0.029324^\circ
$$

我对原始 720 行标签的逐行检查显示：

* 全局最小余量主要由 (\beta_1) 在约 (175^\circ) 相位处产生；
* (\beta_2,\beta_6) 在约 (304^\circ) 附近也接近边界；
* (\beta_3) 在约 (325^\circ) 附近接近边界；
* 当前中心线的 (\kappa) p95 约为 (19.6)，因此中心线的主要问题不是严重病态，而是局部关节余量不足。

因此，应先执行一个新的、预注册的 anchor re-search，而不是事后把 (1.5^\circ) Gate 放宽。

## 3.1 保持不变

保持：

* 实际长半轴 (0.5,\mathrm m)；
* 当前长短轴比；
* 标准 (\beta_6) 关节域；
* FK residual、smoothness、seam Gate 不变。

## 3.2 搜索变量

允许小范围改变：

$$
\Delta c_{q_1},\Delta c_{q_2}\in[-20,20],\mathrm{mm}
$$

$$
\Delta c_{n_\Pi}\in[-10,10],\mathrm{mm}
$$

$$
\omega_1,\omega_2\in[-5^\circ,5^\circ]
$$

并允许：

* 多 root configuration；
* 多 branch seed；
* 不同 cyclic cut；
* 正向和反向 traversal。

## 3.3 anchor 目标函数

先以硬约束保证：

$$
\operatorname{FKResidual}_{95}\le1,\mathrm{mm}
$$

$$
\operatorname{FKResidual}_{\max}\le3,\mathrm{mm}
$$

$$
\Delta\beta_{\mathrm{RMS,p95}}\le1^\circ
$$

$$
\mathrm{seam}\le1^\circ
$$

再优化：

$$
J_{\mathrm{anchor}}
===================

-w_m,m_{\beta,\min}
+
w_\kappa\kappa_{95}
+
w_a\Delta^2\beta_{95}
+
w_tP_{\mathrm{tube-predictor}}
$$

其中 (m_{\beta,\min}) 是整条轨迹的最小关节余量。

## 3.4 anchor 选择标准

正式最低要求：

$$
m_{\beta,\min}\ge1.5^\circ
$$

推荐保留 tube 余量的目标：

$$
m_{\beta,\min}\ge2.0^\circ
$$

选择 top 3–5 个 robust anchors，而不是只保留一个。

---

# 四、如何生成三维 canonical tube 标签

## 4.1 不应逐点独立求 IK

如果每个 tube 点独立求解，会重新出现：

$$
\text{相近 }xyz
\longrightarrow
\text{不同 branch}
$$

所以应把整个 tube 看成一个三维标签流形：

$$
\beta(\phi,u,v)
$$

而不是独立样本集合。

## 4.2 使用 T3 中心线作为 anchor

当前 0.5 m 中心线 T3 比 T4 更适合作为基础：

* T3 residual 更低；
* T3 没有 chart-overlap 失败；
* T4 在当前 0.5 m formal 实验中 overlap gap 达 (2.512^\circ)，未通过 (0.5^\circ) Gate。

因此建议：

> 用 T3 产生中心线，用局部 chart 只作为 tube predictor/corrector 工具，而不是直接采用当前 T4 的标签 lineage。

---

## 4.3 Tube predictor

在中心线构型 (\beta_0(\phi)) 处计算：

$$
J(\beta_0)
==========

\frac{\partial F}{\partial\beta}
\in\mathbb R^{3\times6}
$$

使用 distal-priority 加权矩阵：

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

对 tube 位移：

$$
\Delta x
========

u,n_r+v,n_\Pi
$$

生成初值：

$$
\beta_{\mathrm{pred}}
=====================

\beta_0+J_W^#\Delta x
$$

---

## 4.4 Exact corrector

再用论文的精确准静态 FK 环境修正：

$$
\min_\beta
\frac{|F(\beta)-x_{\mathrm{target}}|^2}{\sigma_x^2}
+
\lambda_b|\beta-\beta_{\mathrm{pred}}|*W^2
+
\lambda_pC*{\mathrm{posture}}
+
\lambda_mC_{\mathrm{margin}}
+
\lambda_\kappa C_{\mathrm{cond}}
$$

这里：

$$
C_{\mathrm{posture}}
====================

4|\beta_{1:2}|^2
+
2|\beta_{3:4}|^2
+
|\beta_{5:6}|^2
$$

---

## 4.5 整个 tube 的联合优化

设离散标签为：

$$
\beta_{i,j,k}
=============

\beta(\phi_i,u_j,v_k)
$$

建议使用整体目标：

$$
J_{\mathrm{surface}}
====================

\lambda_x
\sum_{i,j,k}
|F(\beta_{i,j,k})-x_{i,j,k}|^2
$$

$$
+
\lambda_\phi
\sum_{i,j,k}
|D_\phi\beta_{i,j,k}|*W^2
+
\lambda_u
\sum*{i,j,k}
|D_u\beta_{i,j,k}|*W^2
+
\lambda_v
\sum*{i,j,k}
|D_v\beta_{i,j,k}|_W^2
$$

$$
+
\lambda_2
\sum_{i,j,k}
\left(
|D_\phi^2\beta|_W^2+
|D_u^2\beta|_W^2+
|D_v^2\beta|_W^2
\right)
$$

$$
+
\lambda_p
\sum C_{\mathrm{posture}}
+
\lambda_m
\sum C_{\mathrm{margin}}
+
\lambda_\kappa
\sum C_{\mathrm{cond}}
$$

并对 phase 维度采用循环边界：

$$
\beta(0,u,v)=\beta(2\pi,u,v)
$$

这相当于把当前 trajectory-level canonical teacher 扩展为：

> **trajectory-surface-level canonical teacher。**

---

# 五、Tube 宽度应如何逐步扩大

不要直接从零 offset 跳到 (\pm5,\mathrm{mm})。

## 5.1 并行测试的 tube

建议测试：

| Tube ID | (\rho_r) | (\rho_\Pi) |
| ------- | -------: | ---------: |
| T0      |   0.5 mm |     0.5 mm |
| T1      |     1 mm |       1 mm |
| T2      |     2 mm |       1 mm |
| T3      |     2 mm |       2 mm |
| T4      |     5 mm |       2 mm |
| T5      |     5 mm |       5 mm |

这样可以判断：

* 平面内法向和出平面法向哪个更受限制；
* 是否需要各向异性 tube；
* 失败是 joint margin、conditioning 还是 branch continuity 导致。

## 5.2 正式区域必须是完整 tube

如果某些相位失败，不能把失败点删除后仍称作完整区域。

有两种合法处理：

1. 缩小统一 tube 宽度；
2. 将不规则可达 tube 作为 diagnostic，明确不能声称完整对称区域。

对于正式数据，推荐：

$$
\text{全部审计目标的teacher success}=100%
$$

---

# 六、如何从一条 tube 扩展为可泛化的多 family 数据集

## 6.1 Pilot family 参数范围

建议先使用小范围邻域，不要立即混入 0.75 m。

### 长半轴

$$
a\in[0.46,0.50],\mathrm m
$$

### 长短轴比

当前约为：

$$
\frac ba\approx0.337
$$

Pilot 取：

$$
\frac ba\in[0.31,0.36]
$$

### 中心平移

$$
\Delta c_{q_1},\Delta c_{q_2}\in[-10,10],\mathrm{mm}
$$

$$
\Delta c_{n_\Pi}\in[-5,5],\mathrm{mm}
$$

### 平面倾斜

$$
\omega_1,\omega_2\in[-3^\circ,3^\circ]
$$

这些只是 proposal ranges；每一个 family 都必须由 teacher 重新验证，不得因属于参数盒而自动接受。

## 6.2 采样方式

不要做完整笛卡尔组合，建议使用：

* Latin hypercube；
* Sobol design；
* maximin space-filling design。

Pilot 生成：

$$
N_{\mathrm{family}}=24
$$

其中：

* 14 个训练 family；
* 5 个 validation family；
* 5 个 sealed-test family。

family 的分配必须在标签生成前冻结。

---

# 七、样本规模建议

## 7.1 Core tube Pilot

* 180 phase；
* 每个截面 13 或 25 个点；
* 3–5 个 robust anchors。

规模约：

$$
180\times25\times5=22{,}500
$$

## 7.2 Core tube 正式版本

* 720 phase；
* 每个截面 81 个 Sobol + boundary points。

规模：

$$
720\times81=58{,}320
$$

## 7.3 Multi-family Pilot

* 24 families；
* 每条 180 phase；
* 每个截面 25 点。

规模：

$$
24\times180\times25
===================

108{,}000
$$

## 7.4 正式 family 数据集

筛掉未通过 teacher Gate 的 family 后，保留约 20–30 条 family：

$$
20\times360\times25
===================

180{,}000
$$

可以另外生成嵌套数据规模：

```text
D20k
D50k
D100k
D180k
```

用于建立学习曲线，证明性能提升来自区域覆盖，而不是单次随机训练结果。

---

# 八、如何衡量“空间覆盖程度”

不能再只统计 (xyz) min/max。

## 8.1 参数区域覆盖

在归一化参数：

$$
(\phi,\hat u,\hat v,\xi)
$$

中统计：

* occupied-cell ratio；
* discrepancy；
* boundary coverage；
* 每个 family 的 phase 完整率；
* 每个法向截面的覆盖率。

## 8.2 Cartesian fill distance

在一个独立的 dense audit grid (\mathcal G) 上计算：

$$
h(D,\Omega)
===========

\max_{x\in\mathcal G}
\min_{x_i\in D}
|x-x_i|
$$

同时报告：

* NN p50；
* NN p95；
* NN max；
* 无邻居点比例。

推荐正式目标：

$$
\mathrm{NN}_{95}\le3,\mathrm{mm}
$$

$$
\mathrm{NN}_{\max}\le5,\mathrm{mm}
$$

## 8.3 法向厚度

对局部样本协方差的三个特征值：

$$
\lambda_1\ge\lambda_2\ge\lambda_3
$$

要求两个法向方向都具有非零厚度，而不是所有点重新退化到一条曲线上。

## 8.4 Family 覆盖

报告：

* center range；
* plane-tilt range；
* long-axis range；
* aspect-ratio range；
* 每一维 train/validation/test 的覆盖和间隔。

---

# 九、如何审计标签连续性和多值性

这是决定能否继续用静态 (xyz\to\beta_6) 的关键。

## 9.1 局部连续性

对满足：

$$
|x_i-x_j|\le5,\mathrm{mm}
$$

和：

$$
|x_i-x_j|\le10,\mathrm{mm}
$$

的样本，统计：

$$
d_\beta(i,j)
============

\sqrt{
\frac{1}{6}
\sum_{k=1}^6
(\beta_{i,k}-\beta_{j,k})^2
}
$$

推荐：

$$
d_{\beta,95}^{5\mathrm{mm}}\le0.5^\circ
$$

$$
d_{\beta,95}^{10\mathrm{mm}}\le1^\circ
$$

## 9.2 跨 family 冲突

对于来自不同 family 但：

$$
|x_i-x_j|\le2,\mathrm{mm}
$$

的样本，如果：

$$
d_\beta(i,j)>1^\circ
$$

则标记为 inverse-label conflict。

若要训练单一静态模型，要求：

$$
\text{conflict voxel ratio}=0
$$

如果冲突形成稳定的少数 cluster，则不能直接混合，应采用：

$$
xyz\to\text{chart/family id}
$$

再：

$$
(xyz,\text{chart id})\to\beta_6
$$

如果标签取决于前一时刻构型，则需要：

$$
(x_t,\beta_{t-1})\to\Delta\beta_t
$$

---

# 十、Teacher 数据质量 Gate

## 10.1 物理和标签硬 Gate

正式数据必须满足：

$$
\mathrm{FKResidual}_{95}\le1,\mathrm{mm}
$$

$$
\mathrm{FKResidual}_{\max}\le3,\mathrm{mm}
$$

$$
\mathrm{joint\ margin}_{\min}\ge1.5^\circ
$$

$$
\mathrm{joint\ bounds\ rate}=1
$$

$$
\mathrm{teacher\ success}=100%
$$

$$
\mathrm{multi\ branch\ conflict}=0
$$

## 10.2 轨迹和 surface 连续性

$$
\Delta\beta_{\mathrm{phase,p95}}\le1^\circ
$$

$$
\Delta^2\beta_{\mathrm{phase,p95}}\le0.5^\circ
$$

$$
\mathrm{seam}\le0.5^\circ\sim1^\circ
$$

tube 法向相邻点：

$$
\Delta\beta_{\mathrm{normal,p95}}\le1^\circ
$$

## 10.3 Teacher 重复性

相同 trajectory、root、policy 和 seed 重复运行：

$$
d_{\beta,\mathrm{repeat,p95}}
\le0.1^\circ\sim0.2^\circ
$$

正向与反向 traversal：

$$
d_{\beta,\mathrm{direction,p95}}
\le0.2^\circ\sim0.5^\circ
$$

---

# 十一、训练、验证和测试必须按整条轨迹隔离

当前的偶数相位训练、奇数相位测试只能保留为 interpolation diagnostic，不能作为区域泛化证据。

## 11.1 Row-IID

随机行切分，仅用于调试，不进入正式结论。

## 11.2 Intra-family interpolation

完整保留某些 tube offset、phase block 或截面环，用于检查单 family 内部插值。

## 11.3 Trajectory-family interpolation

留出完整 family，但其参数位于训练 family 参数空间的内部。

例如：

* 训练有中心偏移 (-10,0,+10,\mathrm{mm})；
* 验证使用 (+5,\mathrm{mm})。

## 11.4 Near-OOD family

留出位于参数盒边缘的完整 family：

* (a=0.5,\mathrm m)；
* 最大允许平面倾斜；
* 最大中心偏移；
* 极端长短轴比。

## 11.5 Sealed test

测试 family 的所有 phase 和所有 tube offset 都不能出现在训练或模型选择中。

任何一条 trajectory 的点都必须全部属于同一个 split。

---

# 十二、学生模型实验

区域数据通过后，第一轮仍以当前最有效的模型为主：

$$
xyz\to\beta_6
$$

模型：

```text
S1_l1_tanh
```

即：

* 有界 tanh 输出；
* (\beta) supervision；
* 可微 FK loss。

当前 S1 在单曲线实验中明显优于普通 MLP，是合理的区域 baseline。

并行比较：

1. 静态 (xyz\to\beta_6)；
2. family/chart-conditioned MLP；
3. 只有在路径依赖审计失败时，再测试 stateful 模型。

当前 GRU 自回归 rollout 已发生 (324\sim426,\mathrm{mm}) 误差累积，因此不应在数据区域尚未建立前作为主路线。

---

# 十三、Student 正式 Gate

建议为新的 V11 实验预注册以下 Gate。

## 区域 interpolation

$$
EE_{95}\le5,\mathrm{mm}
$$

$$
EE_{\max}\le10,\mathrm{mm}
$$

## Near-OOD family

$$
EE_{95}\le7.5,\mathrm{mm}
$$

$$
EE_{\max}\le10,\mathrm{mm}
$$

其中 (10,\mathrm{mm}) 对应 (0.5,\mathrm m) 长半轴的 (2%)；这一次必须在实验开始前预注册，不能再作为 post-hoc 判据。此前的 (2%) Gate 是结果产生后新增的，只能作为历史重评口径。

同时要求：

* 关节界内率 (=1)；
* 4/5 seeds 通过；
* 不允许固定轴偏移；
* 不允许只凭 P95 覆盖少量大尾部错误。

---

# 十四、失败回退策略

## 情况 A：0.5 m anchor 仍无法达到 (1.5^\circ) margin

处理顺序：

1. 增加 branch/root 搜索；
2. 小范围重搜中心和平面；
3. 调整 canonical posture 权重；
4. 若仍失败，则只能保留 centerline evidence，不能生成 formal tube。

不得事后放宽 margin Gate 改判。

## 情况 B：(\pm1,\mathrm{mm}) tube 失败

先确定失败方向：

* 若只在 (n_r) 方向失败，使用各向异性 tube；
* 若只在 (n_\Pi) 方向失败，缩小出平面厚度；
* 若某些 phase 全部失败，重新搜索 anchor；
* 不允许删除失败 phase 后称作完整 tube。

## 情况 C：多个 family 出现标签冲突

停止合并静态数据。

转向：

* chart classifier + expert；
* family/context 输入；
* 或有状态策略。

## 情况 D：Teacher Gate 通过但模型泛化仍差

依次检查：

1. Cartesian fill distance；
2. family 参数覆盖；
3. teacher label curvature；
4. model capacity；
5. FK-loss 权重；
6. 是否需要 local experts。

此时才属于学生模型问题。

---

# 十五、分阶段可执行计划

## Phase 0：证据与协议冻结

冻结：

* 当前代码 SHA；
* 当前 0.5 m teacher 数据；
* V11 Gate；
* family 参数范围；
* split family 清单；
* sealed test family。

输出：

```text
00_protocol/
  protocol_v11.yaml
  artifact_manifest.json
  split_manifest.csv
```

---

## Phase 1：0.5 m robust anchor re-search

并行运行：

```text
A0 current pose/current branch
A1 center re-search
A2 center + plane re-search
A3 multi-root branch search
A4 center + plane + branch joint search
```

每个候选用 180 phase 筛选，top 3 用 720 phase 验证。

停止条件：

* 没有候选满足 margin (\ge1.5^\circ)；
* 或 residual/smoothness 退化。

---

## Phase 2：完整 tube 前沿

对 top anchors 测试 T0–T5 六种 tube 宽度。

先：

* 180 phase；
* 9 或 25 cross-section points。

通过后：

* 360 phase；
* dense audit grid。

输出最大完整、均匀、无空洞 tube：

$$
(\rho_r^\star,\rho_\Pi^\star)
$$

---

## Phase 3：Core tube 数据集

使用最佳 anchor 和 tube：

* 720 phase；
* 每截面 81 个 nested Sobol + boundary points；
* 约 (58{,}320) 行。

生成：

```text
D_core_0p5m.parquet
```

并执行完整 teacher、coverage、conflict 和 repeatability 审计。

---

## Phase 4：Multi-family Pilot

用 24 个 LHS family：

* 180 phase；
* 每截面 25 点；
* 约 (108{,}000) 行。

输出：

```text
D_family_pilot.parquet
family_quality.csv
cross_family_conflicts.parquet
```

只保留完整通过 teacher Gate 的 family；不能仅删除 family 内的失败点。

---

## Phase 5：确定学生形式

根据跨 family 审计决定：

### 无冲突

训练静态：

$$
xyz\to\beta_6
$$

### 少数稳定 chart

训练：

$$
xyz\to chart
$$

和 local expert。

### 明显路径依赖

训练有状态策略，但必须重新建立稳定 rollout baseline。

---

## Phase 6：正式区域数据集

建议保留 20–30 个完整 family：

* 360 phase；
* 每截面 25 点；
* (180{,}000\sim270{,}000) 行。

生成嵌套版本：

```text
D20k
D50k
D100k
D200k
```

用于学习曲线和覆盖—误差因果分析。

---

## Phase 7：正式 group-held-out 训练

至少 5 seeds。

评价：

* family interpolation；
* center OOD；
* plane OOD；
* axis-ratio OOD；
* exact (0.5,\mathrm m) unseen family；
* tube boundary；
* same-trajectory phase interpolation，仅作辅助。

---

# 十六、0.75 m 应如何使用

0.75 m 当前只能作为压力测试：

* (698/720) 标签合格；
* teacher residual max 为 (27.293,\mathrm{mm})；
* joint margin 为 (0^\circ)；
* 五 seed 中 joint 仅 (2/5) 通过，0.75-only 为 (0/5)。

因此：

1. 不把当前 0.75 m 数据加入 0.5 m 正式训练；
2. 不用 0.75 m 调模型或选择超参数；
3. 在 0.5 m 模型锁定后，才作为 scale-OOD stress test；
4. 必须报告所有 22 个 teacher 缺口；
5. 不允许通过删除困难相位美化整环结果；
6. 若未来要联合训练，必须先为 0.75 m 重新生成完整、无缺口、branch 对齐的数据。

当前 joint 0.5+0.75 模型在 0.5 m 上表现反而变差，说明简单混合两个不完整、pose 不同的尺度数据会产生负迁移，而不是自动提高泛化。

---

# 十七、建议 Codex 的直接主任务

可以将以下内容直接作为执行要求：

> Build a generalized canonical task-space region around the validated 0.5 m ellipse. First re-search the ellipse center, plane orientation, root configuration, and canonical branch to obtain a robust T3 centerline with at least 2 deg preferred joint-margin reserve while preserving the existing tracking and smoothness gates. Then construct a complete three-dimensional tube parameterized by phase, in-plane normal offset, and out-of-plane normal offset. Generate labels with weighted-Jacobian predictor, exact quasi-static corrector, cyclic continuation, and whole-surface optimization; do not solve tube points independently and do not delete failed points from a claimed complete region. Determine the maximum uniform feasible anisotropic tube by testing 0.5, 1, 2, and 5 mm normal widths. After the core tube passes, generate 24 nearby ellipse families with space-filling perturbations in center, plane orientation, major semiaxis, and axis ratio. Audit Cartesian fill distance, parameter-space coverage, teacher residual, joint margin, conditioning, phase/normal smoothness, repeatability, chart overlap, and cross-family inverse-label conflicts. Use complete trajectory/family group splits rather than interleaved phase splits. Train the bounded MLP with FK loss only if cross-family labels remain approximately single-valued; otherwise use chart-conditioned experts. Keep the existing 0.75 m data completely outside model selection and use it only as a sealed scale-stress test.

---

# 最终建议

下一阶段最重要的不是把一条椭圆采得更密，而是完成两个质变：

$$
\boxed{
\text{1D中心线}
\longrightarrow
\text{3D实心tube}
}
$$

以及：

$$
\boxed{
\text{单一固定family}
\longrightarrow
\text{经过冲突审计的多family区域}
}
$$

只有完成这两步，并使用整条 trajectory、完整 pose 和 family 隔离的测试，才能有说服力地声称模型在 (0.5,\mathrm m) 椭圆对应的任务区域内具有泛化能力，而不再只是记住一条参数曲线。

# BACRA V12.15 单模型蒸馏与空间块封存验证实验记录

## 1. 实验结论

V12.15 已完成正式运行，最终探索性 Gate 通过。

本次实验把 V12.14 的复合 Student 作为训练期 soft-target teacher，蒸馏为每个 seed 一个独立的 bounded MLP；模型锁定以后不再依赖复合模型。与此同时，验证集不再按随机行拆分，而是先在 task position 空间中构造 15 mm macro voxel，再把完整空间块分别分配给 train、validation 和 sealed holdout，并在训练块周围排除 5 mm buffer。

正式结果支持以下有限结论：

> 单个 bounded MLP Student 在当前单 chart capability region 内通过了空间块封存的随机点验证和全新闭环轨迹验证，同时保留了 V12.13 final8 椭圆 family 的既有能力。

该结果不支持 workspace-wide、multi-chart、真实硬件或 deployment claim。正式报告中的 `deployment_claim_gate_pass` 固定为 `false`。

## 2. 协议与代码 fixed point

- protocol id：`branch-aware-canonical-region-atlas-v12.15-single-student-spatial-block-holdout`
- gate semantics：`v12_15_locked_single_student_spatial_block_generalization`
- claim scope：`simulation_single_chart_spatial_block_generalization`
- 实现 worktree：`/mnt/ML_projects/quasi_exp/.worktrees/bacra-v12-15-single-student`
- 实现分支：`codex/bacra-v12-15-single-student`
- 正式运行实现 commit：`a79c51c4db43cde1cbd072031292239fc729e870`
- 标准解释器：`/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python`
- 正式结果根：`/mnt/ML_projects/quasi_exp/runs/bacra_v12_15_single_student_formal_margin_tail_retry1_20260730`

主要新增实现：

- `configs/bacra_v12_15_single_student.yaml`
- `scripts/analysis/run_bacra_v12_15_single_student.py`
- `src/quasi_exp/teacher/retention_distillation.py`
- `tests/test_bacra_v12_15_single_student.py`

V12.15 绑定并校验了 V12.14 Gate、R2 dataset、final8 Teacher/catalog 等源 artifact 的 SHA256，源 closure 记录在 `00_protocol/source_manifest.json`。

## 3. 真正按空间块封存的数据划分

空间划分使用预先冻结的参数：

- macro voxel：15 mm
- train/sealed buffer：5 mm
- partition seed：20260792
- region macro blocks：197
- train blocks：137
- validation blocks：30
- sealed blocks：30

125,000 个源数据点的分配结果：

| 部分 | 行数 | 用途 |
|---|---:|---|
| train | 59,178 | 模型参数更新 |
| validation | 17,220 | variant/seed 选择和模型锁定 |
| sealed | 14,503 | 模型锁定前不读取标签 |
| buffer excluded | 34,099 | 防止相邻空间块泄漏 |

蒸馏训练文件包含 76,398 行，即 train 与 development validation；其中没有写入 sealed rows。sealed block registry 在训练前冻结，模型锁文件记录其 SHA256；最终 Gate 同时验证：

- model 在 sealed holdout 打开前已经锁定；
- locked model 中记录的 sealed registry hash 与实际 registry 完全相同；
- 全部 sealed trajectory point 均属于已登记的 sealed 空间块；
- sealed random 和 sealed path 的 Teacher reference 均完整生成。

关键哈希：

- sealed block registry：`05b6fc3377fdf2213ecaa980ce479ec0242dbafddfd0ff23801acf5099c33bed`
- distillation dataset manifest：`5c389945d31c5e051982c140a117c93a5ac05c301b64cebabcb44bfbe47a0591`
- final model lock：`633bf07e51eeaa754ba8c46bcec79464417c9ec74858ea39d430c8dc14146da2`
- final Gate：`bcb6715136c601474dfbefb4bd0ec733abe07cec586e514d131543a08f13d668`

## 4. Retention-aware distillation

V12.15 的训练目标同时包含：

- V12.14 Teacher beta；
- V12.14 composite Student soft target；
- FK loss；
- safe-margin loss；
- row-space consistency loss。

replay 组成冻结为：

- old：30%
- interior：50%
- boundary：20%

首轮正式候选的 FK 和 final8 retention 已经合格，但 17,220 行 validation block 上三个 seed 的全局最小 joint margin 只有约 1.23°–1.38°，因此在模型选择阶段停止，sealed holdout 没有被打开。

对未封存 validation 的诊断表明，复合 Student soft target 在同一 validation block 上的最小 margin 分别约为 1.656°、1.777° 和 1.721°，说明既定 1.5° Gate 仍然可实现，问题是普通平均损失没有充分约束少量 margin tail，而不是必须放宽 Gate。

retry1 保持 Gate 不变，只进行局部训练工程修复：

- 对 composite margin 小于 2.2° 的样本施加 4 倍训练权重；
- 提高 distillation 和 safe-margin loss；
- 保留同一空间划分、同一 sealed registry、同一模型结构和同一 seed。

两个候选 variant 均达到 development admission。冻结排序最终选择 `S0_margin_tail`：

- learning rate：`5e-5`
- steps：9,000
- teacher beta loss：0.10
- distillation loss：2.0
- FK loss：0.75
- margin loss：4.0
- row-space loss：0.50

锁定的三个模型均是独立的 `single_bounded_mlp`：

| seed | model SHA256 |
|---:|---|
| 20260738 | `26e2d5b562fd2c6bb133a1a20d8b3f452c7829efce80b11d0f3d78f5dab3e379` |
| 20260739 | `219caf5bb2ec888aa3bfd463590aabd02687f1bc7028f78bdb295a5ab688f765` |
| 20260740 | `31b0660777d65548f5c0d1889fefbd8b5378a2949ba245c083688a2999525d1d` |

## 5. 正式结果

### 5.1 Development validation block

`S0_margin_tail`：

| seed | FK P95 | FK max | minimum margin | seed Gate |
|---:|---:|---:|---:|---|
| 20260738 | 1.468 mm | 5.227 mm | 1.509° | pass |
| 20260739 | 1.142 mm | 3.562 mm | 1.425° | fail：margin only |
| 20260740 | 1.549 mm | 7.009 mm | 1.519° | pass |

按预注册的 2/3 seed 语义，variant 可以锁定。

### 5.2 Sealed random spatial-block test

模型锁定后，从 sealed block 中评估 10,000 个随机点：

| seed | FK P95 | FK max | minimum margin | seed Gate |
|---:|---:|---:|---:|---|
| 20260738 | 1.343 mm | 2.688 mm | 1.510° | pass |
| 20260739 | 0.902 mm | 3.351 mm | 1.385° | fail：margin only |
| 20260740 | 1.242 mm | 2.802 mm | 1.523° | pass |

结果为 2/3 seed 通过。seed 20260739 的 FK 和实际机械 bounds 均通过，唯一失败项仍是最小 margin。

### 5.3 Sealed unseen trajectories

- 新封存闭环 family：20
- family Gate：20/20 通过
- 三个 seed 的 family-seed 结果：60/60 通过
- Teacher reference：完整
- path point sealed registration：完整

这部分说明 Student 不只是对独立散点有效，也能在被冻结的新空间块中形成完整连续闭环跟踪。

### 5.4 V12.13 final8 retention

- retained families：8/8
- family-seed：23/24

唯一未通过的 family-seed 是 seed 20260740 的 `final_core_01`：

- major semiaxis：493.239 mm
- FK P95：5.0065 mm，即半长轴的 1.015%
- FK max：5.6622 mm，即半长轴的 1.148%

该 family 的另外两个 seed 通过，因此按照既有 2/3 seed family Gate，final8 仍为 8/8 retained。这里不能写成 24/24，也不能将单 seed 的轻微越线隐藏掉。

### 5.5 最终 Gate

`08_summary/gate.json` 中九项检查全部为 `true`：

- sealed random 2/3 seed Gate；
- sealed paths 20/20 family Gate；
- final8 8/8 retention；
- locked artifact 是单模型；
- model lock 早于 holdout open；
- sealed registry hash 一致；
- sealed random/path Teacher reference 完整；
- 所有 path point 均是 registered sealed point。

最终：

- `gate_pass=true`
- `deployment_claim_gate_pass=false`

## 6. 硬件利用

Student 训练使用 GTX 1080 Ti。单 GPU 上按 seed/variant 串行运行 6 个训练任务，以避免多个 TensorFlow 进程争抢显存：

- requested/effective GPU workers：1/1
- 总 wall time：5,689.6 s
- aggregate child CPU time：6,017.6 s
- peak concurrent RSS：约 1,887.7 MiB
- `S0_margin_tail` 单任务约 809–820 s
- `S1_margin_tail_strong` 单任务约 1,081–1,086 s

Teacher sealed labeling 是 CPU 密集阶段，使用 12 个独立 subprocess worker，并固定每 worker 的 BLAS/OMP thread 为 1：

- sealed random：12 workers，wall time 43.83 s，aggregate CPU utilization 9.11
- sealed paths：12 workers，wall time 13.59 s，aggregate CPU utilization 8.79

因此本次没有把可独立的 Teacher labeling 串行化；Student 阶段的单 worker 是单 GPU 的主动资源调度，不是退回 CPU 单核。

## 7. 失败证据与未覆盖范围

首轮正式运行保留在：

`/mnt/ML_projects/quasi_exp/runs/bacra_v12_15_single_student_formal_20260730`

它在 development selection 因 minimum margin tail 停止，sealed holdout 没有打开。该失败证据用于定位训练目标不足，没有被覆盖。

成功运行保留在：

`/mnt/ML_projects/quasi_exp/runs/bacra_v12_15_single_student_formal_margin_tail_retry1_20260730`

当前仍需明确保留的边界：

1. 这是同一 capability chart 内的 spatial-block generalization，不是整个机器人 workspace 的任意几何泛化。
2. seed 20260739 在 development 和 sealed random 上都存在稳定复现的 margin tail；总体 Gate 依赖预注册的 2/3 seed 规则。
3. final8 存在 1/24 个 family-seed 的相对 FK P95 轻微越线。
4. 所有结论来自 simulation；没有执行器、标定误差、硬件噪声或真实部署不确定性预算。

## 8. 验证

已运行：

- V12.15 focused tests、V12.14 runner compatibility 和 longrun wait tests：16 passed。
- 使用标准 V12.15 worktree `PYTHONPATH` 与 `--import-mode=importlib` 的全量 pytest：100% passed，只有 warnings。
- GPU smoke pipeline：全部 stage 通过。
- 正式 retry1：long-run terminal return code 0，最终 Gate 通过。

项目根直接执行裸命令：

`/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python -m pytest -q`

仍会在 collection 阶段受到已知的第三方 `tests` package 名称碰撞影响，错误为 `ModuleNotFoundError: tests.test_segmented_tension_solver`。这与 V12.15 的 focused/full importlib 验证结果分开记录，不能把裸命令的收集问题写成 V12.15 数值失败。

## 9. 关键 artifact

- 最终 Gate：`08_summary/gate.json`
- 最终建议：`08_summary/final_recommendation.md`
- 空间划分：`01_spatial_seal/partition_report.json`
- sealed registry：`01_spatial_seal/sealed_block_registry.json`
- 数据集 manifest：`02_distillation_dataset/dataset_manifest.json`
- 训练并行证据：`03_single_students/training_parallel_manifest.json`
- development selection：`04_development_selection/selection.json`
- model lock：`05_model_lock/final_model_lock.json`
- sealed random metrics：`06_sealed_block_test/metrics.csv`
- sealed trajectory metrics：`07_sealed_trajectory_test/trajectory_metrics.csv`
- final8 retention：`07_sealed_trajectory_test/final8_retention_metrics.csv`

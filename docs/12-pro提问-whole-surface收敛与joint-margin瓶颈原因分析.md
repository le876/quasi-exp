---
question_id: Q12
question_number: 12
question_confirmed_by_user: true
question_confirmation_summary: "仅分析 whole-surface block convergence 失败与 minimum joint margin < 1.5° 的可能机制、相互关系及证据边界；不要求设计下一步实验方案。"
date: "2026-07-24"
status: ready-to-send
project: "quasi_exp"
evidence_cutoff: "2026-07-24 19:02:30 CST (+0800)"
source_snapshot: "实验由 codex/generalized-ellipse-region-v11@5c72a9e 生成；当前同一 worktree HEAD 为 0cbe2fd，5c72a9e..0cbe2fd 在 src/scripts/configs/tests 无差异，worktree clean；项目主工作树另有不属于本实验的既有 dirty 文件。"
---

# GPT-5 Pro 第十二次交接：whole-surface 收敛与 joint-margin 瓶颈原因分析

## 0. 经用户确认的 GPT-5 Pro 任务

用户已经确认的任务原意：

> 请 GPT-5 Pro 基于 V11.4 的正式 centerline、tube surface、Gate 和并行运行证据，分析 whole-surface block convergence 失败与 minimum joint margin < 1.5° 这两个瓶颈的可能机制、相互关系以及当前证据能支持和不能支持的结论。

确认范围：

- 只做原因分析、机制区分和证据边界审计；
- 不要求设计下一步实验方案；
- 附件包含 tube/bridge Gate 实际计算所依赖的全量原始 surface 数据、直接输入
  centerline、task 参数、Gate/report、配置、机器人原始输入和指标计算代码；
- 不把未保存的中间状态、未运行的实验或 Codex 推断写成事实。

## 1. 当前有效事实与最短摘要

### 1.1 证据截止时间

本交接以 `2026-07-24 19:02:30 CST (+0800)` 前已经落盘并重新核验的正式
artifact 为准。正式 downstream tube Gate 和 bridge Gate 的文件时间均为
`2026-07-24 18:42:55 +0800`。

### 1.2 当前有效事实

1. V11.4 在两个 candidate 上都找到了通过严格 Gate 的 720-phase 完整闭环
   centerline；Formal Gate 通过，两个 candidate 的 formal decision 均为 A。
2. downstream tube 对两个 Formal centerline 分别计算了四种宽度组合，共
   8 个 `180 phase × 9 cross-section = 1620 row` surface；8 个 surface 的
   `teacher_success` 全部为 true，合计原始行数为 12,960，没有删点或抽样。
3. 8 个 surface 的正式 report 均为 `gate_pass=false`；全部共同失败于
   `joint_margin` 和 `whole_surface_block_convergence`。
4. 因 screen 没有通过项，`dense_candidate_count=0`、`selected_tube=null`；
   full-tube branch audit、区域数据集、Student 训练和 sealed evaluation 均未运行。
5. 任务进程退出码为 0，表示预注册流水线在阴性 Gate 处正常早停，不表示
   tube 或 bridge Gate 通过。

### 1.3 关键边界

1. 当前证据正式支持“这 8 个已执行 surface 没有通过严格 tube Gate”，不支持
   “所有可能 solver、sweep 数、centerline 或 geometry 都不可能通过”。
2. `surface.parquet` 保存最终 surface 的完整 target、achieved XYZ 和 beta6
   字段；每轮 sweep 的完整中间 beta field 没有持久化。report 只保存每轮
   `update_rms_max_deg` 标量，因此无法仅从现有 Parquet 独立重建每轮最大更新
   出现在哪个 node、phase 或 joint。
3. margin 的最终数值可以由 `surface.parquet` 中的 beta 与正式机器人
   `beta_ranges_rad` 重新计算；block convergence 的最后数值只能与 report
   中持久化的 sweep history 和实现代码交叉核验。

## 2. 项目目标与本轮边界

### 2.1 用户给定的项目目标

项目最终目标是为 0.5 m 三维椭圆区域建立可信、可泛化的监督数据集，并训练
`xyz -> beta6` Student。用户本次只要求 GPT-5 Pro 分析阻止区域 teacher
进入 Student 训练的两个直接瓶颈，不要求给出解决方案。

### 2.2 当前实际覆盖范围

本轮实际覆盖：

- 两个通过严格 Formal 的 720-phase centerline；
- 两个 anchor 的 0.5/1.0 mm 径向与平面向组合；
- 8 个 180-phase、9-point cross-section surface；
- surface metrics、local consistency、tube frontier 和 bridge early stop；
- 确定性 subprocess 并行与资源证据。

### 2.3 当前证据未覆盖的内容

- 没有执行超过 4 次 block sweep 的正式对照；
- 没有执行 under-relaxation、Jacobi、Anderson acceleration 或其他 surface
  fixed-point 算法的正式对照；
- 没有保存逐 sweep 的完整 beta field；
- 没有执行以 hard 1.5° interior bounds 代替 soft margin penalty 的 tube 对照；
- 没有生成正式区域数据集，也没有 Student 训练或泛化指标。

## 3. 相对上次材料新增的事实

| 类型 | 已发生的变化 | 实际证据 | 证据等级 |
|---|---|---|---|
| 实现/配置 | V11 downstream surface 按独立 surface 使用确定性 subprocess；Formal 请求 8 workers，每 worker BLAS/OpenMP 线程固定为 1 | `run_v11_region.py`、`downstream_config.yaml`、`tube_parallel_manifest.json` | formal implementation |
| 实验 | 8 个正式 screen surface 全部完成，worker 无失败，但没有 surface 通过 Gate | `tube_screen_frontier.csv`、8 个 `surface_*.parquet`、8 个 `report_*.json` | formal |
| 结论修正 | monitor 的 process success 不能解释为科学 Gate 通过；实际 tube/bridge Gate 均为 false，且不存在 V11.4 end-to-end marker | `tube_gate.json`、`bridge_gate.json`、`longrun.log` | formal |

## 4. 实验方法与复现口径

### 4.1 代码与工作树 fixed point

| 项目 | 值 |
|---|---|
| 项目根 | `/mnt/ML_projects/quasi_exp` |
| 实验分支 | `codex/generalized-ellipse-region-v11` |
| artifact 生成代码 commit | `5c72a9ec13b91b6c916f720a755d5475d648759b` |
| 当前 HEAD | `0cbe2fda13503a853ad97f1a107b9d924dfb0b5c` |
| worktree | `/mnt/ML_projects/quasi_exp/.worktrees/generalized-ellipse-region-v11` |
| worktree dirty 状态 | clean |
| 代码差异 | `5c72a9e..0cbe2fd` 在 `src/`、`scripts/`、`configs/`、`tests/` 无差异；后一 commit 只更新实验记录 |
| 项目主工作树 | 存在用户既有 dirty/untracked 文件；未用于修改本轮 artifact |
| 仅 checkout 当前 HEAD 是否足以复现 | 代码足够；还需要本附件中的 robot CSV、正式 centerline 和有效配置 |

`downstream_config.yaml` 内的 `source_fixed_point=7a73f2a...` 是继承 V11
downstream protocol 的冻结来源标识；实际并行 artifact 的 runtime code
fixed point 是上表中的 `5c72a9e`。两者代表不同层次，不应合并。

### 4.2 Surface 方法与参数

每个 surface 的执行步骤由 `region.py`、`canonical.py` 和
`run_v11_region.py` 定义：

1. 使用 9-point cross-section，按稳定半径顺序先从 center 向外 bootstrap；
2. 每个 node 求解一条完整 cyclic trajectory；
3. 已求解的 Delaunay neighbour beta 均值作为 initializer/neighbor anchor；
4. 完成 bootstrap 后执行 4 个 block-coordinate sweeps；
5. 每个 sweep 依次执行 outward 和 inward 两个方向；
6. 每次重新求解所有 node，包括 center node；
7. sweep 前后对全部 `node × phase × 6 beta` 计算 beta 差；
8. 对 6 个 beta 取 RMS、转换为 degree，再对全部 node/phase 取 maximum；
9. 最后一次 maximum `<=0.05°` 才记为 block converged。

| 参数 | 实际值 | 来源 |
|---|---:|---|
| screen phase count | 180 | `downstream_config.yaml` |
| cross-section count | 9 | `downstream_config.yaml`、各 task JSON |
| max surface sweeps | 4 | `downstream_config.yaml` |
| sweep directions | outward, inward | `downstream_config.yaml` |
| convergence tolerance | 0.05° | `downstream_config.yaml` |
| surface neighbor weight | 0.1 | `downstream_config.yaml` |
| surface neighbour scale | 1.0° | `downstream_config.yaml` |
| safe joint margin | 1.5° | `downstream_config.yaml` |
| downstream safe-margin repulsion step | 0.0° | `downstream_config.yaml` |
| Formal centerline safe-margin repulsion step | 0.05° | `formal_config.yaml` |
| solver seed | 20260720 | `downstream_config.yaml` |
| cross-section seed | 20260722 | `downstream_config.yaml` |
| candidate budget | 16 | `downstream_config.yaml` |
| max corrector iterations | 100 | `downstream_config.yaml` |

`canonical.py` 中与 margin 直接相关的已核实实现事实：

- trajectory least-squares 的优化变量硬 bounds 是机器人的实际 beta 上下界；
- 1.5° safe margin 通过高权重单侧 shell penalty 加入 residual，而不是作为
  `least_squares` 的 hard interior bounds；
- whole-trajectory 优化后还会逐点调用 corrector；
- downstream 的 `safe_margin_repulsion_step_deg=0.0`；
- trajectory success 只要求 tracking residual 和实际机械 bounds，不要求
  `joint_margin >=1.5°`；1.5° 由后续 surface Gate 独立裁决。

这些是实现事实，不等价于“已经证明 margin 失败由某一个配置导致”。

### 4.3 数据生成与划分

| 项目 | 实际口径 |
|---|---|
| 直接输入 centerline | 两个 720-row `centerline_A2_*.parquet` |
| screen 输入 | 每个 task 从对应 720-row centerline 稳定 subsample 到 180 phase |
| surface 生成方式 | 2 anchors × 4 width combinations × 180 phases × 9 cross-sections |
| surface 原始总行数 | 12,960 |
| 单个 surface schema | 23 columns：ID/offset、target XYZ、achieved XYZ、success、chart、6 个 teacher beta |
| train/validation/test | 未生成 |
| seed | solver 20260720；cross-section 20260722；audit 20260724 |
| 泄漏审计 | Student 阶段未开始，不适用 |

本 bundle 上传全部 8 个 `surface.parquet`，没有抽样、筛选或派生替代。

### 4.4 正式 metrics 和 Gate

| check | metric | Gate |
|---|---|---:|
| all rows success | `success_rate` | `>=1.0` |
| residual p95 | `residual_p95_mm` | `<=1.0 mm` |
| residual max | `residual_max_mm` | `<=3.0 mm` |
| joint margin | `joint_margin_min_deg` | `>=1.5°` |
| phase smoothness | `phase_beta_rms_p95_deg` | `<=1.0°` |
| phase acceleration | `acceleration_beta_rms_p95_deg` | `<=0.5°` |
| cyclic seam | `seam_beta_rms_max_deg` | `<=1.0°` |
| normal smoothness | `surface_edge_beta_rms_p95_deg` | `<=1.0°` |
| normal Laplacian | `surface_laplacian_beta_rms_p95_deg` | `<=1.0°` |
| block convergence | `surface_block_update_rms_max_deg` | `<=0.05°` |
| local 5 mm | `local_5mm_beta_gap_p95_deg` | `<=0.5°` |
| local 10 mm | `local_10mm_beta_gap_p95_deg` | `<=1.0°` |

Surface Gate 是上述 checks 的逻辑 AND。Tube stage 要求至少一个 `(0.5,0.5)`
screen 通过并形成通过 dense/repeat/reverse audit 的 selected tube。Bridge
看到 tube Gate false 后写入 `minimum_tube_pass=false` 并早停。

### 4.5 复现入口与产物

正式命令：

```text
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python \
scripts/analysis/run_generalized_ellipse_full_loop_v11_4.py \
--preset formal --stage bridge --project-root /mnt/ML_projects/quasi_exp
```

主要产物原路径：

- `runs/generalized_ellipse_region_v11_full_loop_downstream/01_anchor/bridged/`
- `runs/generalized_ellipse_region_v11_full_loop_downstream/02_tube/`
- `runs/generalized_ellipse_region_v11_full_loop_feasible_branch/05_downstream_bridge/gate.json`

## 5. 实验结果

### 5.1 八个 surface

| Anchor | radial/plane mm | residual p95/max mm | margin min ° | final block update ° | sweep history ° | 失败 checks | 附件 |
|---|---:|---:|---:|---:|---|---|---|
| A2_143 | 0.5/0.5 | 0.013271/0.044377 | 1.492904 | 0.097718 | 0.212008, 0.100661, 0.100497, 0.097718 | margin, block convergence | `surface_A2_143_r0p5_p0p5.parquet`, `report_A2_143_r0p5_p0p5.json` |
| A2_143 | 0.5/1.0 | 0.013447/0.044938 | 1.492632 | 0.099032 | 0.212270, 0.100702, 0.101468, 0.099032 | margin, block convergence | `surface_A2_143_r0p5_p1.parquet`, `report_A2_143_r0p5_p1.json` |
| A2_143 | 1.0/0.5 | 0.013431/0.045025 | 1.492836 | 0.098081 | 0.212054, 0.100911, 0.101009, 0.098081 | margin, block convergence | `surface_A2_143_r1_p0p5.parquet`, `report_A2_143_r1_p0p5.json` |
| A2_143 | 1.0/1.0 | 0.013411/0.045613 | 1.492581 | 0.099175 | 0.212201, 0.100999, 0.101650, 0.099175 | margin, block convergence | `surface_A2_143_r1_p1.parquet`, `report_A2_143_r1_p1.json` |
| A2_178 | 0.5/0.5 | 0.024481/0.087059 | 1.487527 | 0.137110 | 0.266074, 0.131556, 0.133011, 0.137110 | margin, block convergence | `surface_A2_178_r0p5_p0p5.parquet`, `report_A2_178_r0p5_p0p5.json` |
| A2_178 | 0.5/1.0 | 0.024487/0.088819 | 1.485992 | 0.136762 | 0.265695, 0.131562, 0.132987, 0.136762 | margin, block convergence | `surface_A2_178_r0p5_p1.parquet`, `report_A2_178_r0p5_p1.json` |
| A2_178 | 1.0/0.5 | 0.024214/0.088651 | 1.487116 | 0.137295 | 0.266361, 0.132210, 0.133162, 0.137295 | margin, block convergence | `surface_A2_178_r1_p0p5.parquet`, `report_A2_178_r1_p0p5.json` |
| A2_178 | 1.0/1.0 | 0.465061/0.986709 | 0.214145 | 7.090321 | 8.271214, 6.730563, 4.505781, 7.090321 | margin, local 5/10 mm, normal edge/Laplacian, block convergence | `surface_A2_178_r1_p1.parquet`, `report_A2_178_r1_p1.json` |

### 5.2 可由最终 Parquet 重算的 margin minimum 位置

以下是本交接制作时从全量 surface beta 和 `robot_config.yaml` bounds 重新计算的
post-hoc 定位；没有修改正式 Gate：

| Anchor | radial/plane mm | min joint/side | phase | cross-section | physical offsets mm |
|---|---:|---|---:|---:|---:|
| A2_143 | 0.5/0.5 | beta2 upper | 138 | 6 | -0.354/-0.354 |
| A2_143 | 0.5/1.0 | beta2 upper | 137 | 7 | -0.000/-1.000 |
| A2_143 | 1.0/0.5 | beta2 upper | 138 | 6 | -0.707/-0.354 |
| A2_143 | 1.0/1.0 | beta2 upper | 137 | 6 | -0.707/-0.707 |
| A2_178 | 0.5/0.5 | beta2 upper | 132 | 6 | -0.354/-0.354 |
| A2_178 | 0.5/1.0 | beta2 upper | 131 | 7 | -0.000/-1.000 |
| A2_178 | 1.0/0.5 | beta2 upper | 132 | 6 | -0.707/-0.354 |
| A2_178 | 1.0/1.0 | beta6 lower | 137 | 5 | -1.000/0.000 |

该定位支持“margin failure 在哪些最终点出现”，不支持仅凭位置确定其动力学或
算法根因。

### 5.3 通过项

- 8 个 surface 均为 1620 rows，`success_rate=1.0`；
- 8 个 surface 均通过 residual p95/max、phase smoothness、phase
  acceleration 和 cyclic seam；
- 除 A2_178 的 1.0/1.0 mm surface 外，其余 7 个均通过 normal smoothness、
  normal Laplacian 和 local 5/10 mm consistency；
- 两个 Formal centerline 均通过 720-phase strict cycle 和 audit。

### 5.4 失败与早停

- 8/8 surface 的 `joint_margin=false`；
- 8/8 surface 的 `whole_surface_block_convergence=false`；
- A2_178 1.0/1.0 surface 还出现多项局部/法向失稳；
- `tube_gate.json` 的四项 checks 全部为 false；
- `bridge_gate.json` 为 `minimum_tube_pass=false`；
- `tube_dense_frontier.csv` 只有换行，表示没有 dense task；
- 不存在 full-tube branch audit manifest 或 `V11_4_END_TO_END_COMPLETED.json`。

## 6. 证据等级、失败结果与未知事项

### 6.1 正式事实

- Formal centerline Gate、8 个 surface report、tube Gate、bridge Gate 和
  parallel manifest 均来自同一正式流水线。
- Gate 阈值在运行前由有效配置确定，没有根据结果事后放宽。
- 8 个 worker task 全部成功完成；阴性结果不是 worker crash、缺失切片或部分
  merge。

### 6.2 Post-hoc 但可复核的事实

| 内容 | 等级 | 能支持什么 | 不能支持什么 |
|---|---|---|---|
| 从最终 surface beta 与机器人 bounds 重算 margin minimum 的 joint/phase/node | post-hoc deterministic audit | 定位最终 margin minimum | 证明 margin 缺口由哪个优化步骤造成 |
| 对比 Formal 和 downstream 的 safe-margin repulsion 配置 | implementation comparison | 证明两个有效配置值不同 | 证明该差异是唯一或主要原因 |
| 阅读 block sweep 实现 | implementation audit | 证明更新顺序、停止量和 center 会被重解 | 预测增加 sweep 后一定收敛或发散 |

### 6.3 已发现的实现或证据缺口

- report 保存了 4 个 sweep 的 maximum update history，但没有保存每轮完整
  `beta[node, phase, joint]`；
- 没有保存 maximum update 的 argmax node、phase、joint；
- trajectory candidate diagnostics 没有逐 surface 持久化到
  `surface.parquet`；
- 当前 artifact 无法区分稳定 limit cycle、慢收敛、局部 branch switching、
  Gauss-Seidel 顺序效应或最后 corrector 回拉等机制。

### 6.4 尚未知或尚未执行

- 若保持完全相同方法只增加 sweep，A2_143 的约 0.098–0.099° plateau 是否会
  继续下降；
- outward/inward 顺序是否产生二周期或更高周期；
- margin 低于 1.5° 是由 trajectory optimizer、逐点 corrector、neighbor
  coupling、center re-solve 还是几何可行域共同导致；
- A2_143 的小 margin deficit 与 block plateau 是否来自同一组 phase/node；
- A2_178 1.0/1.0 的明显失稳是否与其他 7 个 surface 属于同一种机制；
- hard interior margin constraints 下是否仍有同一 target surface 的可行解。

### 6.5 当前不能声称

- 不能声称只把 `max_surface_sweeps` 增大就会通过；
- 不能声称只把 downstream `safe_margin_repulsion_step_deg` 改为 0.05 就会通过；
- 不能声称 0.007° margin deficit 是浮点误差；
- 不能声称当前 geometry 在 1.5° margin 下绝对不可行；
- 不能声称已存在可以用于 Student 的区域标签；
- 不能把 A2_178 1.0/1.0 的明显失稳自动外推到其他 7 个 surface。

## 7. 决策所需的客观对照

本节只比较已存在的配置和结果，不给出路线建议。

| 对照对象 | 已执行输入/方法 | 已观察结果 | 证据等级 | 已知限制 |
|---|---|---|---|---|
| A2_143 Formal centerline | 720 phases；Formal policy 的 repulsion step 0.05° | strict cycle/audit pass；margin min 约 1.500000° | formal | 是单条 centerline，不是区域 |
| A2_143 tube surfaces | 180×9；downstream repulsion step 0.0°；4 outward/inward sweeps | 4/4 margin 约 1.4926–1.4929°；block update 约 0.0977–0.0992° | formal | 没有逐 sweep beta field |
| A2_178 Formal centerline | 720 phases；Formal policy 的 repulsion step 0.05° | strict cycle/audit pass；margin min 约 1.500022° | formal | 是单条 centerline，不是区域 |
| A2_178 三个非 1/1 surfaces | 180×9；4 sweeps | margin 约 1.4860–1.4875°；block update 约 0.1368–0.1373° | formal | 没有逐 sweep beta field |
| A2_178 1/1 surface | 180×9；4 sweeps | margin 0.2141°；block update 7.0903°；多项 normal/local Gate 失败 | formal | 单个宽度反例，不能自动外推 |

## 8. 经用户确认的问题

1. 请 GPT-5 Pro 基于 V11.4 的正式 centerline、tube surface、Gate 和并行运行证据，分析 whole-surface block convergence 失败与 minimum joint margin < 1.5° 这两个瓶颈的可能机制、相互关系以及当前证据能支持和不能支持的结论。

## 9. 经用户确认的回答要求

这次只做原因分析：

- 分析两个瓶颈各自可能对应的机制；
- 分析它们可能相互独立、共享原因或互相影响的方式；
- 明确区分 artifact/代码已经支持的事实、合理但未验证的解释以及现有证据无法
  判断的事项；
- 不设计下一步实验方案，不给出实验优先级、实现计划或 Student 训练方案。

## 10. 附件清单与复现信息

GPT-5 Pro 实际可见文件数为 63：`QUESTION.md` 加 62 个实际证据附件。由于
超过 15 个文件，网页端只上传打包器生成的 `Q12_GPT5Pro_upload.zip`。

### 10.1 Formal 与 centerline 证据

| 上传文件名 | 原始 SSH 路径 | 大小 | 证据等级 | 支持事实/上传理由 |
|---|---|---:|---|---|
| `formal_config.yaml` | `.worktrees/generalized-ellipse-region-v11/configs/generalized_ellipse_region_v11_full_loop.yaml` | 5,224 B | formal protocol | Formal centerline policy、Gate 和 downstream 授权口径 |
| `formal_gate.json` | `runs/generalized_ellipse_region_v11_full_loop_feasible_branch/04_formal/gate.json` | 29,241 B | formal | 两个 candidate 的 Formal 总 Gate |
| `formal_parallel_manifest.json` | `runs/generalized_ellipse_region_v11_full_loop_feasible_branch/04_formal/formal_parallel_manifest.json` | 2,055 B | formal execution | Formal subprocess 完整性 |
| `formal_report_A2_143.json` | `runs/generalized_ellipse_region_v11_full_loop_feasible_branch/04_formal/A4_A2_143_r1_reverse_c0045/formal_report.json` | 9,248 B | formal | A2_143 strict cycle/audit 和 input hashes |
| `formal_report_A2_178.json` | `runs/generalized_ellipse_region_v11_full_loop_feasible_branch/04_formal/A4_A2_178_r0_reverse_c0045/formal_report.json` | 9,249 B | formal | A2_178 strict cycle/audit 和 input hashes |
| `formal_cycle_A2_143.parquet` | `runs/generalized_ellipse_region_v11_full_loop_feasible_branch/04_formal/A4_A2_143_r1_reverse_c0045/selected_cycle/cycle.parquet` | 71,576 B | formal raw | A2_143 选中闭环原始表 |
| `formal_cycle_A2_178.parquet` | `runs/generalized_ellipse_region_v11_full_loop_feasible_branch/04_formal/A4_A2_178_r0_reverse_c0045/selected_cycle/cycle.parquet` | 71,576 B | formal raw | A2_178 选中闭环原始表 |
| `formal_cycle_gate_A2_143.json` | `runs/generalized_ellipse_region_v11_full_loop_feasible_branch/04_formal/A4_A2_143_r1_reverse_c0045/selected_cycle/gate.json` | 18,094 B | formal | A2_143 cycle metrics/checks |
| `formal_cycle_gate_A2_178.json` | `runs/generalized_ellipse_region_v11_full_loop_feasible_branch/04_formal/A4_A2_178_r0_reverse_c0045/selected_cycle/gate.json` | 18,637 B | formal | A2_178 cycle metrics/checks |
| `full_loop_completed.json` | `runs/generalized_ellipse_region_v11_full_loop_feasible_branch/FULL_LOOP_EXPERIMENT_COMPLETED.json` | 340 B | formal marker | 只证明 centerline Formal 完成及 downstream 授权 |
| `anchor_gate.json` | `runs/generalized_ellipse_region_v11_full_loop_downstream/01_anchor/gate.json` | 1,023 B | formal | bridged anchor inventory 和 hash |
| `centerline_A2_143.parquet` | `runs/generalized_ellipse_region_v11_full_loop_downstream/01_anchor/bridged/A4_A2_143_r1_reverse_c0045/centerline.parquet` | 49,542 B | formal raw | A2_143 tube solver 的直接 720-row beta 输入 |
| `centerline_A2_178.parquet` | `runs/generalized_ellipse_region_v11_full_loop_downstream/01_anchor/bridged/A4_A2_178_r0_reverse_c0045/centerline.parquet` | 49,542 B | formal raw | A2_178 tube solver 的直接 720-row beta 输入 |

### 10.2 Tube/bridge 汇总与执行证据

| 上传文件名 | 原始 SSH 路径 | 大小 | 证据等级 | 支持事实/上传理由 |
|---|---|---:|---|---|
| `downstream_config.yaml` | `runs/generalized_ellipse_region_v11_full_loop_feasible_branch/05_downstream_bridge/generalized_ellipse_region_v11_4_downstream.yaml` | 5,213 B | formal effective config | 实际 surface policy、threshold、seed 和并行参数 |
| `tube_screen_frontier.csv` | `runs/generalized_ellipse_region_v11_full_loop_downstream/02_tube/screen_frontier.csv` | 2,820 B | formal derived | 8 个 report 的稳定顺序汇总 |
| `tube_dense_frontier.csv` | `runs/generalized_ellipse_region_v11_full_loop_downstream/02_tube/dense_frontier.csv` | 1 B | formal derived | 证明 screen 无通过项，dense 为空 |
| `tube_gate.json` | `runs/generalized_ellipse_region_v11_full_loop_downstream/02_tube/gate.json` | 5,543 B | formal | Tube checks、artifact hashes 和最终 false Gate |
| `tube_parallel_manifest.json` | `runs/generalized_ellipse_region_v11_full_loop_downstream/02_tube/tube_parallel_manifest.json` | 12,574 B | formal execution | 8/8 task 完整性、CPU/RSS、批次和日志路径 |
| `bridge_gate.json` | `runs/generalized_ellipse_region_v11_full_loop_feasible_branch/05_downstream_bridge/gate.json` | 7,767 B | formal | `minimum_tube_pass=false` 及 early stop |
| `longrun.log` | `.worktrees/generalized-ellipse-region-v11/runs/maintenance/longrun/v11_4_downstream_bridge_parallel8_20260724.log` | 实际文件大小见 manifest | formal execution | 命令终态和 process exit code |

### 10.3 全量 raw surface

以下 8 个文件是 Gate 所使用的全部最终 surface Parquet，共 1,678,778 B、
12,960 rows；没有抽样：

- `surface_A2_143_r0p5_p0p5.parquet`
- `surface_A2_143_r0p5_p1.parquet`
- `surface_A2_143_r1_p0p5.parquet`
- `surface_A2_143_r1_p1.parquet`
- `surface_A2_178_r0p5_p0p5.parquet`
- `surface_A2_178_r0p5_p1.parquet`
- `surface_A2_178_r1_p0p5.parquet`
- `surface_A2_178_r1_p1.parquet`

它们分别来自：

```text
runs/generalized_ellipse_region_v11_full_loop_downstream/02_tube/
  anchors/<candidate>/screen/<width>/surface.parquet
```

以下 8 个原始 report 共 21,117 B，包含 metrics、checks、Gate、surface SHA、
family geometry 和四轮 sweep scalar history：

- `report_A2_143_r0p5_p0p5.json`
- `report_A2_143_r0p5_p1.json`
- `report_A2_143_r1_p0p5.json`
- `report_A2_143_r1_p1.json`
- `report_A2_178_r0p5_p0p5.json`
- `report_A2_178_r0p5_p1.json`
- `report_A2_178_r1_p0p5.json`
- `report_A2_178_r1_p1.json`

以下 8 个 task JSON 共 9,516 B，记录每个 worker 的 centerline 路径、family
geometry、phase/cross-section、width、cut、direction、seed offset 和输出目录：

- `task_A2_143_r0p5_p0p5.json`
- `task_A2_143_r0p5_p1.json`
- `task_A2_143_r1_p0p5.json`
- `task_A2_143_r1_p1.json`
- `task_A2_178_r0p5_p0p5.json`
- `task_A2_178_r0p5_p1.json`
- `task_A2_178_r1_p0p5.json`
- `task_A2_178_r1_p1.json`

### 10.4 机器人原始输入和计算代码

| 上传文件名 | 原始 SSH 路径 | 证据等级 | 支持事实/上传理由 |
|---|---|---|---|
| `robot_config.yaml` | `configs/robot_rods_only_standard_100k.yaml` | formal input | beta bounds、FK 输入路径和物理配置 |
| `robot_lengths.csv` | `data/robot/generated/lengths.csv` | formal raw input | FK rod lengths |
| `robot_holes.csv` | `data/robot/generated/holes.csv` | formal raw input | 机器人孔位原始表 |
| `robot_disks.csv` | `data/robot/generated/disks.csv` | formal raw input | disk geometry |
| `robot_material.csv` | `data/robot/generated/material_rods_only.csv` | formal raw input | rods-only material |
| `robot_end_effector.csv` | `data/robot/generated/end_effector.csv` | formal raw input | end-effector local point |
| `region.py` | `.worktrees/generalized-ellipse-region-v11/src/quasi_exp/teacher/region.py` | implementation | surface bootstrap、Gauss-Seidel sweep、metrics |
| `region_audit.py` | `.worktrees/generalized-ellipse-region-v11/src/quasi_exp/teacher/region_audit.py` | implementation | surface Gate checks |
| `canonical.py` | `.worktrees/generalized-ellipse-region-v11/src/quasi_exp/teacher/canonical.py` | implementation | trajectory solve、soft margin shell、corrector |
| `forward.py` | `.worktrees/generalized-ellipse-region-v11/src/quasi_exp/teacher/forward.py` | implementation | beta-space FK environment 和 hard bounds |
| `kinematics.py` | `.worktrees/generalized-ellipse-region-v11/src/quasi_exp/model/kinematics.py` | implementation | forward kinematics |
| `sampling.py` | `.worktrees/generalized-ellipse-region-v11/src/quasi_exp/model/sampling.py` | implementation | beta-to-theta |
| `io_config.py` | `.worktrees/generalized-ellipse-region-v11/src/quasi_exp/io/config.py` | implementation | robot config loader |
| `robot_inputs.py` | `.worktrees/generalized-ellipse-region-v11/src/quasi_exp/io/robot_inputs.py` | implementation | robot CSV loader |
| `run_v10_teacher.py` | `.worktrees/generalized-ellipse-region-v11/scripts/analysis/run_trajectory_canonical_teacher_v10.py` | implementation | environment construction |
| `run_v11_region.py` | `.worktrees/generalized-ellipse-region-v11/scripts/analysis/run_generalized_ellipse_region_v11.py` | implementation | surface worker、local metrics、frontier 和 tube Gate |
| `run_v11_4_full_loop.py` | `.worktrees/generalized-ellipse-region-v11/scripts/analysis/run_generalized_ellipse_full_loop_v11_4.py` | implementation | centerline bridge、derived config 和 early stop |
| `experiment_record.md` | `.worktrees/generalized-ellipse-region-v11/docs/GeneralizedEllipseRegionV11FullLoopFeasible实验记录.md` | formal record | 执行时间、并行现象、验证和终态 |

### 未上传的大文件或派生数据说明

- 本 bundle 没有用摘要替代 tube Gate 的原始 surface；8 个 surface 全部上传。
- 没有上传约 140 MiB 的完整 Formal root/graph candidate 搜索矩阵，因为它们
  不是这两个 tube failure metrics 的直接计算输入；上传了 Formal 选中 cycle、
  cycle Gate、formal report、Formal 总 Gate，以及 tube 实际读取的两条 bridged
  centerline。
- 没有上传空 stderr 和与 report 内容重复的 8 个 worker stdout。
- 没有生成或上传逐 sweep beta field，因为正式运行没有持久化该数据；不能
  伪造或事后重建。
- §5.2 的 margin argmin 表是从上传的全量 Parquet 和 robot bounds 做的确定性
  post-hoc 定位，没有作为独立附件替代原始数据。

## 11. 给下一位讨论者的最短事实摘要

1. 两条 720-phase Formal centerline 均 strict pass，但它们的 8 个
   180×9 tube surfaces 全部 Gate fail。
2. 共同失败项是 `joint_margin_min_deg <1.5°` 和最后 sweep
   `surface_block_update_rms_max_deg >0.05°`；A2_178 1/1 另有明显局部/法向
   失稳。
3. margin 可从最终 raw surface 重算；block convergence 的逐 sweep 完整状态
   没有保存，只有 scalar history。
4. 已确认问题仅要求分析两个瓶颈的可能机制、相互关系和证据边界，不要求设计
   下一步实验方案。
---
# BACRA-V12：Branch-Aware Canonical Region Atlas 完整实验计划

## 一、实验定位

本轮不再以“为一条椭圆单独生成标签”为主线，而是建立一个任务相关三维工作空间区域中的 canonical inverse atlas：

$$
\boxed{
\text{精确准静态正向环境}
\rightarrow
\text{可达能力图}
\rightarrow
\text{稀疏任务空间点云}
\rightarrow
\text{每点多组严格可行 IK}
\rightarrow
\text{branch-aware product graph}
\rightarrow
\text{canonical chart/atlas}
\rightarrow
\text{区域稠密化}
\rightarrow
\text{Student 蒸馏}
\rightarrow
\text{未见轨迹测试}
}
$$

当前实验已经证明，单条 (0.5,\mathrm m) 椭圆中心线可以生成高精度标签并被 MLP 拟合，但该数据只有 (720) 行，且只有一个 trajectory、family、chart 和 branch，本质上只能证明同一曲线上的相位插值，不能证明三维区域泛化。

V11.4 已经进一步尝试从中心线扩展 tube：两个通过严格 Gate 的 (720)-phase 中心线共生成了八个 (180\times9) surface，但八个 surface 均因 `joint_margin` 和 `whole_surface_block_convergence` 失败，没有进入区域数据集或 Student 阶段。

历史上的 Gate-v2 则证明：在单一 family 中，可以生成 (90{,}000) 行、十个半径的高一致性数据，并使静态 MLP 在 (100,\mathrm{mm}) 中心线任务上稳定通过；但它仍然是单 family，且不能证明跨 family 或更一般工作空间中的单值逆映射。

因此，本轮的核心科学任务是：

> 在一个真实三维任务区域内，先构造严格可行、局部连续、路径一致的一张或多张 inverse chart；椭圆、圆、Lissajous 和随机样条随后只作为该区域数据集的测试轨迹。

---

# 二、实验所依据的核心原理

## 2.1 正向环境是权威物理环境

定义六个主动构型变量：

$$
\beta=
(\beta_1,\beta_2,\beta_3,\beta_4,\beta_5,\beta_6)
\in\mathcal B\subset\mathbb R^6
$$

论文前半部分给出的准静态环境定义：

$$
x=F(\beta),\qquad x=(x,y,z)\in\mathbb R^3.
$$

最终 Student 不学习任意逆解，而是学习一个 canonical selector：

$$
c:\Omega\rightarrow\mathcal B
$$

满足：

$$
F(c(x))\approx x.
$$

由于系统是 (6\to3)，固定一个 (x) 后通常仍有多个连续构型自由度。因此，随机采完整 (\beta_6) 后交换输入输出，会重新产生多 branch 标签混叠。

---

## 2.2 数据生成的本质是选择一个连续截面

对于任务点 (x_i)，严格可行逆解集合定义为：

$$
\mathcal C_i^{\mathrm{hard}}
============================

\left{
\beta\ \middle|
\begin{aligned}
&|F(\beta)-x_i|\le\epsilon_{\mathrm{node}},\
&\beta_{\min}\le\beta\le\beta_{\max},\
&\text{solver converged}
\end{aligned}
\right}.
$$

可进一步定义部署安全子集：

$$
\mathcal C_i^{\mathrm{gold}}
============================

\left{
\beta\in\mathcal C_i^{\mathrm{hard}}
:
m(\beta)\ge1.5^\circ
\right},
$$

其中：

$$
m(\beta)
========

\min_j
\min
\left(
\beta_j-\beta_j^{\min},
\beta_j^{\max}-\beta_j
\right).
$$

目标不是逐点选择 residual 最小的候选，而是在所有任务节点上联合选择：

$$
\beta_i\in\mathcal C_i
$$

并尽量满足：

$$
x_i\approx x_j
\Rightarrow
\beta_i\approx\beta_j.
$$

---

## 2.3 借鉴的主流方法如何组合

Fang 等针对软机器人学习的是正向运动学和 Jacobian，然后通过 Jacobian 迭代求解 IK，从而避免直接学习多值逆映射；其轨迹求解依赖相邻 waypoint 的 continuation。([arXiv][1])

Thuruthel 等采用正向模型、轨迹优化器和监督学习策略，形成“慢速优化教师 (\to) 快速学生”的框架。([Zenodo][2])

TORM 将冗余 IK、末端路径跟踪和关节轨迹平滑放在完整轨迹优化中，而不是逐点独立求解。([arXiv][3])

IKLink 为每个 waypoint 生成多组 IK 解，再通过图和动态规划寻找全局连接。([arXiv][4])

Lee 等使用局部非参数模型来处理软连续体机器人中全局映射复杂、局部环境变化和冗余控制问题。([Sage Journals][5])

Atlas 类方法则通过多张局部 chart 表示无法被单一全局坐标系覆盖的约束流形。([arXiv][6])

本实验将这些思想整合为：

$$
\boxed{
\text{多候选 IK}
+
\text{局部 continuation}
+
\text{product graph}
+
\text{chart/atlas}
+
\text{精确 corrector}
+
\text{Student distillation}
}
$$

---

# 三、需要验证的核心假设

## H1：目标区域内存在至少一个严格可行的局部 inverse chart

即存在：

$$
c_k:\Omega_k\rightarrow\mathcal B
$$

使：

$$
F(c_k(x))\approx x
$$

且局部连续。

## H2：逐点独立 IK 不能可靠构造该 chart

需要比较：

$$
\text{pointwise IK}
$$

与：

$$
\text{branch-aware graph selection}.
$$

## H3：当前 whole-surface Gate 失败主要是旧 surface 求解器的问题，而不一定说明区域内不存在 canonical chart

新方案不再依赖四轮 Gauss–Seidel surface sweep，而通过严格候选图和局部 predictor-corrector 建立 atlas。

## H4：单一静态 (xyz\to\beta_6) 是否成立，应由路径一致性实验决定

若不同路径到达同一点得到不同 (\beta)，则必须使用 chart-conditioned 或 stateful Student。

## H5：区域覆盖程度应与 Student OOD 误差相关

需要建立：

$$
\text{fill distance}
\leftrightarrow
\text{Student FK error}
$$

的经验关系。

## H6：BACRA 数据比 mixed FK 和 pointwise IK 数据更适合监督学习

通过统一样本量、统一模型、统一 split 的消融实验验证。

---

# 四、实验目录与代码结构

建议项目名：

```text
Branch-Aware Canonical Region Atlas V12
```

输出目录：

```text
runs/branch_aware_canonical_region_atlas_v12/
  00_protocol/
  01_capability_map/
  02_task_region/
  03_candidate_solver_benchmark/
  04_sparse_candidates/
  05_product_graph/
  06_chart_extraction/
  07_consistency_audit/
  08_dense_generation/
  09_dataset_assembly/
  10_student_models/
  11_sealed_trajectory_benchmark/
  12_ablations/
  13_summary/
```

建议新增代码：

```text
src/quasi_exp/teacher/capability_map.py
src/quasi_exp/teacher/task_region.py
src/quasi_exp/teacher/multi_ik_candidates.py
src/quasi_exp/teacher/product_graph.py
src/quasi_exp/teacher/canonical_atlas.py
src/quasi_exp/teacher/path_consistency.py
src/quasi_exp/teacher/dense_chart_sampling.py
src/quasi_exp/teacher/atlas_audit.py

scripts/analysis/run_bacra_v12.py
scripts/analysis/train_bacra_students_v12.py
scripts/analysis/evaluate_bacra_v12.py

configs/bacra_v12.yaml
```

---

# 五、Phase 0：冻结协议与基线

## 5.1 目标

保证后续变化只来自数据生成方法，而不是 FK、robot config 或 Gate 的隐式变化。

## 5.2 冻结内容

必须记录：

```text
robot_config SHA
FK implementation SHA
beta bounds
beta-to-theta mapping SHA
solver versions
random seeds
task-region definition
Gold/Silver/Reject semantics
train/validation/test spatial partition
sealed trajectory generation seed
```

## 5.3 基线数据

保留三类历史基线：

### D0：mixed-FK baseline

从完整 (\beta_6) 中采样后 FK，直接交换为 (xyz\to\beta_6)。

### D1：pointwise-IK baseline

对工作空间点逐点独立求一个 canonical-cost 最小 IK。

### D2：trajectory/tube baseline

当前 V10/V11 的轨迹 continuation / tube surface 方法。

### D3：BACRA

本轮推荐方法。

## 5.4 输出

```text
00_protocol/protocol_v12.yaml
00_protocol/artifact_manifest.json
00_protocol/baseline_inventory.json
```

---

# 六、Phase 1：建立 branch-agnostic capability map

## 6.1 目的

先确定哪些工作空间区域值得进行昂贵的 inverse labeling。

Capability map 只回答：

* 某区域是否有可达构型；
* 候选构型是否有安全余量；
* Jacobian 条件是否较好；
* 是否接近工作空间边界。

它不直接作为 inverse 训练数据。

## 6.2 输入采样

使用标准 (\beta_6) 域 Sobol 采样：

$$
N_{\mathrm{FK}}=2^{20}=1{,}048{,}576.
$$

若已有 full-beta 1M pool 的机器人配置、FK 实现和 bounds 与 V12 完全一致，可以复用；否则重新生成。

每个样本保存：

```text
beta1...beta6
theta1...theta30
x,y,z
joint_margin
sigma1,sigma2,sigma3
kappa
posture_cost
```

## 6.3 Voxel 化

Pilot：

$$
\Delta x_{\mathrm{voxel}}=10\text{mm}
$$

Formal：

$$
\Delta x_{\mathrm{voxel}}=5\text{mm}.
$$

每个 voxel 保存：

* FK sample count；
* 最大 margin；
* 最小 (\kappa)；
* (\sigma_3) 分布；
* (\beta)-cluster 数量；
* 到已知 (0.5,\mathrm m) 椭圆的距离。

## 6.4 区域等级

### Core-safe voxel

至少存在：

$$
m(\beta)\ge1.5^\circ
$$

的 sample。

### Feasible-boundary voxel

存在实际机械界内的样本，但：

$$
0<m(\beta)<1.5^\circ.
$$

### Unsupported voxel

当前 pool 未发现有效候选。

Core-safe 用于正式 Gold atlas；boundary 只用于探索和边界诊断。

## 6.5 需要验证

1. 1M 与 4M pool 的 voxel occupancy 是否收敛；
2. 已知 (0.5,\mathrm m) 椭圆是否落在同一连通 component 中；
3. 目标区域周围是否存在足够 Core-safe voxel；
4. capability map 的 NN distance 是否能预测后续 IK 成功率。

## 6.6 Gate

Pilot 必须满足：

$$
\text{已知中心线 Core-safe support ratio}\ge95%.
$$

否则先重新定义目标区域或使用 boundary exploratory track，不直接进入 Gold atlas。

---

# 七、Phase 2：定义三维任务区域并采样点云

## 7.1 区域定义

已知 (0.5,\mathrm m) 椭圆记为：

$$
\Gamma_{0.5}
============

{\gamma(\phi):\phi\in[0,2\pi)}.
$$

建立嵌套任务区域：

$$
\Omega_r
========

\left{
x:
\operatorname{dist}(x,\Gamma_{0.5})\le r
\right}
\cap
\mathcal W_{\mathrm{capability}}.
$$

初始测试：

$$
r\in{2,5,10,20}\text{mm}.
$$

但最终区域不要求保持规则 tube。实际采用的是：

> capability map 中与中心线支撑 voxel 连通的 Core-safe component，并限制在 (\Omega_r) 中。

因此中心线只用于定位工作空间 component，而不用于给点排序或产生标签。

## 7.2 点云采样

Pilot：

$$
N_x=2{,}000.
$$

Formal：

$$
N_x=5{,}000.
$$

采样方法：

1. Core-safe voxel 中进行 farthest-point / Poisson-disk sampling；
2. (60%) 采内部；
3. (25%) 采 capability 边界；
4. (15%) 采已知困难扇区、高 (\kappa) 或候选稀少区。

## 7.3 Task-space graph

构建：

$$
G_X=(V_X,E_X).
$$

每个节点连接：

$$
k_x=12
$$

个最近邻。

最大边长：

$$
|x_i-x_j|
\le1.5h,
$$

其中 (h) 是点云中位最近邻距离。

过长边不连，避免强行跨越 workspace 空洞。

## 7.4 输出

```text
02_task_region/task_nodes.parquet
02_task_region/task_edges.parquet
02_task_region/region_report.json
02_task_region/region_visualization.png
```

## 7.5 需要验证

* 点云是否为单一任务空间连通分量；
* fill distance；
* 边界和内部采样比例；
* 是否存在孤立节点；
* 是否覆盖已知 (0.5,\mathrm m) 椭圆及其邻域。

---

# 八、Phase 3：多 IK 候选生成器基准

## 8.1 目的

每个 (x_i) 都需要多组严格可行 IK 候选，不能让一个 seed 决定标签。

## 8.2 候选来源

### C1：FK pool nearest-cluster seeds

从 1M capability pool 中取最近 64 个样本，并在 (\beta) 空间聚类，保留 8 个代表 seed。

### C2：加权 DLS

使用：

$$
J_W^#
=====

W^{-1}J^\top
\left(
JW^{-1}J^\top+\mu^2I
\right)^{-1},
$$

其中：

$$
W=\operatorname{diag}(4,4,2,2,1,1).
$$

更新：

$$
\Delta\beta
===========

J_W^#
\left(
x_i-F(\beta)
\right)
-------

\alpha
\left(
I-J_W^#J
\right)
\nabla C_{\mathrm{posture+margin}}.
$$

### C3：bounded SQP / least squares

直接对完整 (\beta_6) 求解：

$$
\min_\beta
|F(\beta)-x_i|^2
+
\lambda_pC_{\mathrm{posture}}(\beta)
$$

并施加实际机械 bounds。

### C4：null-space exploration

从已收敛解沿：

$$
N=I-J^#J
$$

的基向量做正负扰动，再 correct 到目标点。

### C5：邻域 warm-start

使用 task graph 邻居已有候选作为 seed。

## 8.3 并行求解器

比较：

| Solver | 内容                                           |
| ------ | -------------------------------------------- |
| S-C1   | DLS only                                     |
| S-C2   | bounded SQP only                             |
| S-C3   | DLS + SQP 合并                                 |
| S-C4   | DLS + SQP + null-space + neighbor warm-start |

推荐正式使用 S-C4，但必须由 Pilot 结果决定。

## 8.4 候选预算

Pilot：

$$
16\text{–}32\text{ seeds/node}.
$$

Formal：

* 普通节点：16 seeds；
* 困难节点：32–64 seeds。

## 8.5 候选严格准入

候选类分为：

### Gold candidate

$$
|F(\beta)-x_i|\le3\text{mm},
$$

整体候选集后续要求 P95 不超过 (1)mm，并且：

$$
m(\beta)\ge1.5^\circ,
$$

actual bounds 内、solver converged。

### Silver candidate

tracking 和 actual bounds 通过，但：

$$
0<m(\beta)<1.5^\circ.
$$

只用于诊断和边界探索，不用于主 Gold Student。

### Reject

* solver 未收敛；
* actual bounds 越界；
* residual (>3)mm；
* NaN/Inf；
* 通过 clip 才可行。

不得为凑候选数而保留 Reject。

## 8.6 聚类距离

同时保存两种距离。

角度 RMS：

$$
d_{\deg}(\beta,\beta')
======================

\sqrt{
\frac16
\sum_{j=1}^{6}
(\beta_j-\beta_j')^2
}.
$$

关节范围归一化距离：

$$
d_{\mathrm{norm}}
=================

\sqrt{
\frac16
\sum_{j=1}^{6}
\left(
\frac{\beta_j-\beta_j'}{\beta_j^{\max}-\beta_j^{\min}}
\right)^2
}.
$$

候选聚类不能只依赖固定 (0.5^\circ)，必须同时报告阈值敏感性。

## 8.7 Pilot 数据

* capability pool 中 500 个 synthetic targets；
* task region 中 500 个随机 nodes；
* 200 个 boundary/hard nodes。

## 8.8 通过标准

Synthetic recovery：

$$
\text{success rate}\ge99.5%.
$$

Task-region：

$$
\text{至少一个 Gold candidate 的节点比例}\ge95%.
$$

普通节点 Gold candidate 中位数：

$$
\ge4.
$$

若某些节点无 Gold candidate，必须保存其 Silver/Reject 原因，不能静默删除。

---

# 九、Phase 4：构建 task/configuration product graph

## 9.1 图节点

每个严格候选形成一个节点：

$$
v_{i,k}=(x_i,\beta_{i,k}).
$$

## 9.2 图边不能只按 (\beta) 距离判断

对于每条 task edge：

$$
(x_i,x_j)\in E_X,
$$

从 (\beta_{i,k}) 做 Jacobian predictor：

$$
\beta_{\mathrm{pred}}
=====================

\beta_{i,k}
+
J_{W,i,k}^{#}(x_j-x_i).
$$

再使用 exact corrector 求 (x_j)。

若 corrector 解：

1. tracking 与 bounds 通过；
2. 属于 (x_j) 的候选 cluster (l)；
3. 与 (\beta_{j,l}) 的 RMS gap 小于匹配阈值；

则连接：

$$
v_{i,k}\leftrightarrow v_{j,l}.
$$

这样，图边表示：

> 从候选 (\beta_{i,k}) 出发，可以沿局部任务位移连续到达候选 (\beta_{j,l})。

## 9.3 Edge cost

$$
E_{ik,jl}
=========

w_d d_W^2(\beta_{i,k},\beta_{j,l})
+
w_p C_{\mathrm{posture}}
+
w_\kappa\log(1+\kappa)
+
w_mP_{\mathrm{margin}}.
$$

但物理 residual 和 actual bounds 不作为软代价，而是候选准入硬条件。

## 9.4 输出

```text
05_product_graph/product_nodes.parquet
05_product_graph/product_edges.parquet
05_product_graph/component_report.csv
```

## 9.5 需要验证

* product graph connected-component 数量；
* 每个 task node 属于多少 component；
* 是否有任务节点在某 component 中出现多个相距较大的候选；
* component 的 task-space 覆盖率；
* Gold 与 Silver component 的差异。

---

# 十、Phase 5：提取 canonical chart/atlas

## 10.1 Root 的新角色

使用：

$$
8\text{–}16
$$

个 root anchors。

Root 不是最终标签规则，而只是探索不同 product-graph component 的入口。

每个 root candidate 生成一张 root-induced section。

## 10.2 Root-induced section

对某个 root candidate，在 product graph 中计算到所有 candidate nodes 的最短累计代价。

对于每个 task node (x_i)，选择：

$$
s_i
===

\operatorname*{argmin}*k
D(v*{\mathrm{root}},v_{i,k}).
$$

得到初始 section：

$$
c_{\mathrm{root}}(x_i)=\beta_{i,s_i}.
$$

## 10.3 图上联合标签优化

定义离散能量：

$$
E(s)
====

\sum_i U_i(s_i)
+
\lambda
\sum_{(i,j)\in E_X}
V_{ij}(s_i,s_j).
$$

其中：

$$
U_i(k)
======

w_pC_{\mathrm{posture}}
+
w_\kappa\log(1+\kappa)
+
w_mP_{\mathrm{margin}},
$$

而：

$$
V_{ij}(k,l)
===========

\begin{cases}
d_W^2(\beta_{i,k},\beta_{j,l}),
&\text{若 product edge 存在},\
+\infty,
&\text{否则}.
\end{cases}
$$

求解方法：

1. root-shortest-path 初始化；
2. ICM 局部更新；
3. 多 root restart；
4. 保留 top (M) 个 section proposals。

Pilot 不要求实现通用 NP-hard 全局求解器；ICM、loopy belief propagation 或 tree-reweighted approximation 均可作为对照。

## 10.4 Chart 评分采用字典序

先比较：

1. Gold task-node 覆盖率；
2. task-space 连通性；
3. cycle consistency；
4. path independence；
5. 最小 joint margin；
6. residual；
7. local label smoothness；
8. posture；
9. condition。

即：

$$
\boxed{
\text{区域可行性}
\succ
\text{canonical posture preference}
}
$$

## 10.5 Single-valuedness

同一 chart 中，同一 task node 只能保留一个候选。

如果一个 component 在同一 (x_i) 上包含多个相距：

$$
d_\beta>1^\circ
$$

的稳定候选，则不能视作单一 chart，应拆分为不同 `chart_id`。

## 10.6 Chart 合并

对两个 chart 的重叠任务节点：

$$
\mathcal O_{kl}
===============

\Omega_k\cap\Omega_l.
$$

计算：

$$
d_{\mathrm{overlap,p95}}
========================

\operatorname{P95}*{x\in\mathcal O*{kl}}
|c_k(x)-c_l(x)|_{\mathrm{RMS}}.
$$

### 可合并

$$
d_{\mathrm{overlap,p95}}\le0.5^\circ.
$$

### 保持分离

$$
d_{\mathrm{overlap,p95}}>1^\circ.
$$

---

# 十一、Phase 6：路径无关性和闭环一致性审计

这是决定最终 Student 形式的关键。

## 11.1 多路径一致性

从同一 anchor 到某个目标 node，随机生成三条不同 task-space path：

$$
\gamma_1,\gamma_2,\gamma_3.
$$

每条 path 通过 Jacobian predictor + exact corrector 传播 (\beta)。

比较：

$$
d_{\mathrm{path}}(x)
====================

\max_{a,b}
|\beta_{\gamma_a}(x)-\beta_{\gamma_b}(x)|_{\mathrm{RMS}}.
$$

Pilot：

* 100 个 endpoints；
* 每个 endpoint 3 条 paths。

Formal：

* 500 endpoints；
* 每个 endpoint 5 条 paths。

## 11.2 Task-space loop audit

随机抽取闭环：

$$
x_0\rightarrow x_1\rightarrow\cdots\rightarrow x_0.
$$

要求：

$$
d_{\mathrm{return}}
===================

|\beta_{\mathrm{return}}-\beta_0|_{\mathrm{RMS}}.
$$

Pilot 100 个 loops，Formal 500 个 loops。

## 11.3 Forward/reverse audit

对同一路径正向和反向运行：

$$
d_{\mathrm{direction,p95}}
\le0.5^\circ.
$$

## 11.4 Repeatability

相同 root、path 和 seed 再运行：

$$
d_{\mathrm{repeat,p95}}
\le0.2^\circ.
$$

## 11.5 Gate

静态 chart 进入正式数据需要：

$$
d_{\mathrm{path,p95}}\le0.5^\circ,
$$

$$
d_{\mathrm{return,p95}}\le0.5^\circ,
$$

$$
d_{\mathrm{overlap,p95}}\le0.5^\circ,
$$

$$
d_{\mathrm{repeat,p95}}\le0.2^\circ.
$$

最大值建议：

$$
\le1^\circ.
$$

## 11.6 决策

### 低路径依赖、单 chart

训练：

$$
xyz\rightarrow\beta_6.
$$

### 多个稳定 chart

训练：

$$
xyz\rightarrow chart_id,
$$

以及：

$$
(xyz,chart_id)\rightarrow\beta_6.
$$

### 同一 chart 内仍明显路径依赖

训练：

$$
(x_t,\beta_{t-1})\rightarrow\Delta\beta_t.
$$

此时不得把不同历史标签混成静态数据。

---

# 十二、Phase 7：在 chart 内稠密生成数据

## 12.1 数据量目标

### Dense Pilot

$$
50{,}000\text{ rows}.
$$

### Formal

$$
100{,}000\sim200{,}000\text{ rows}.
$$

只有学习曲线显示 (200k) 仍显著优于 (100k) 时，才扩展到 (250k\sim500k)。

---

## 12.2 方法 A：任务空间 tetrahedral interpolation

对每张 chart 的 task nodes 做局部 Delaunay tetrahedralization，或寻找四个仿射独立近邻。

对 tetrahedron 顶点：

$$
(x_1,\beta_1),\ldots,(x_4,\beta_4)
$$

采样重心系数：

$$
\lambda_j\ge0,\qquad
\sum_{j=1}^{4}\lambda_j=1.
$$

目标点：

$$
x=
\sum_{j=1}^{4}
\lambda_jx_j.
$$

构型初值：

$$
\beta_{\mathrm{init}}
=====================

\sum_{j=1}^{4}
\lambda_j\beta_j.
$$

随后 exact corrector：

$$
\beta^\star
===========

\operatorname{Correct}
(x,\beta_{\mathrm{init}}).
$$

这比每点全局随机 IK 更能保持 chart identity。

## 12.3 方法 B：Jacobian predictor

在 chart boundary 或无合法 tetrahedron 区域：

$$
\beta_{\mathrm{pred}}
=====================

\beta_a+
J_{W,a}^{#}(x-x_a).
$$

再 exact corrector。

## 12.4 双 anchor 一致性

对每个 dense target，从两个独立邻近 anchor 生成两个解：

$$
\beta^{(1)},\beta^{(2)}.
$$

要求：

$$
|\beta^{(1)}-\beta^{(2)}|_{\mathrm{RMS}}
\le0.5^\circ.
$$

否则标记为 chart conflict，不加入静态 Gold 数据。

## 12.5 最终 FK 验证

每个 accepted sample 必须重新计算：

$$
x_{\mathrm{FK}}=F(\beta^\star)
$$

并保存 residual。

## 12.6 Active fill-distance enrichment

在独立 audit set (\mathcal G) 上计算：

$$
d(x,D)=
\min_{x_i\in D}
|x-x_i|.
$$

优先补点区域：

1. Cartesian NN distance 大；
2. chart 边界；
3. candidate count 少；
4. 高 (\kappa)；
5. label curvature 高；
6. probe Student ensemble disagreement 高。

---

# 十三、区域数据质量指标

## 13.1 Cartesian coverage

报告：

$$
\mathrm{NN}*{50},
\mathrm{NN}*{95},
\mathrm{NN}_{\max}.
$$

Pilot 目标：

$$
\mathrm{NN}_{95}\le3\text{mm},
$$

$$
\mathrm{NN}_{\max}\le5\text{mm}.
$$

## 13.2 Voxel occupancy

在 (5)mm 和 (10)mm voxel 下报告：

* occupied ratio；
* empty-hole ratio；
* samples per voxel；
* boundary occupancy。

## 13.3 局部标签连续性

对：

$$
|x_i-x_j|\le5\text{mm}
$$

要求：

$$
d_{\beta,\mathrm{p95}}\le0.5^\circ.
$$

对：

$$
|x_i-x_j|\le10\text{mm}
$$

要求：

$$
d_{\beta,\mathrm{p95}}\le1^\circ.
$$

## 13.4 局部 Lipschitz 指标

$$
L_{ij}
======

\frac{
d_\beta(i,j)
}{
|x_i-x_j|
}.
$$

报告 P50、P95、max 和空间热区。

## 13.5 冲突率

在 (2)mm task voxel 内，若：

$$
d_\beta>1^\circ,
$$

记作 conflict。

Gold 静态数据要求：

$$
\text{conflict voxel ratio}=0.
$$

## 13.6 Teacher 精度

正式要求：

$$
e_{\mathrm{FK,p95}}\le1\text{mm},
$$

$$
e_{\mathrm{FK,max}}\le3\text{mm}.
$$

## 13.7 Margin

数据同时输出两种身份。

### Gold

$$
m_{\min}\ge1.5^\circ.
$$

### Silver

actual bounds 内、tracking 和一致性通过，但：

$$
0<m_{\min}<1.5^\circ.
$$

Silver 可以用于算法诊断和 probe Student，不得用于 Gold 部署结论。

---

# 十四、Phase 8：数据集装配与 Split

## 14.1 禁止正式 row-IID 结论

随机行 split 只用于 debug。

## 14.2 Spatial block split

将区域划分为 (20)mm macro-voxels。

完整 macro-voxel 只能属于一个 split：

* train：70%；
* validation：15%；
* sealed test：15%。

在 validation/test 宏块周围设置 (5)mm buffer，buffer 样本不进入 train，减少最近邻泄漏。

## 14.3 Family/trajectory split

额外生成多组完整轨迹：

* 椭圆；
* 圆；
* Lissajous；
* 3D B-spline；
* 随机平滑闭环。

同一条完整轨迹的全部 phase 必须属于同一个 split。

## 14.4 Sealed trajectory

在 atlas、Student 架构和训练超参数冻结后，使用独立 seed 生成：

* 3 条未见椭圆；
* 2 条圆；
* 2 条 Lissajous；
* 3 条随机 B-spline。

这些轨迹在模型选择期间不可读取。

---

# 十五、Phase 9：Student 模型训练

## 15.1 主模型

使用历史上已经有效的结构作为主基线：

$$
xyz\rightarrow\beta_6.
$$

采用：

* bounded tanh output；
* (\beta) supervision；
* differentiable FK loss。

损失：

$$
\mathcal L
==========

\lambda_\beta
|\hat\beta-\beta^\star|^2
+
\lambda_{\mathrm{FK}}
|F(\hat\beta)-x|^2.
$$

## 15.2 模型并行比较

### 静态单 chart

* MLP；
* ResMLP；
* LGBM baseline；
* RBF/KRR 小数据 baseline。

### 多 chart

* chart classifier + local MLP experts；
* soft mixture-of-experts。

### 路径依赖

只有 Phase 6 证明存在历史依赖时才运行：

* stateful MLP；
* GRU/LSTM。

当前不优先上 Transformer，因为首先需要确定状态依赖是否真实存在。

## 15.3 五 seed 正式实验

所有正式模型至少使用五个 seeds。

## 15.4 Student Gate

对于 (0.5,\mathrm m) 任务区域，预注册：

### 区域 interpolation

$$
EE_{\mathrm{p95}}\le5\text{mm},
$$

$$
EE_{\max}\le10\text{mm}.
$$

### Near-OOD spatial/family

$$
EE_{\mathrm{p95}}\le7.5\text{mm},
$$

$$
EE_{\max}\le10\text{mm}.
$$

同时：

* joint bounds rate (=1)；
* 至少 (4/5) seeds 通过；
* 无固定轴偏移；
* 无 chart boundary error spike。

---

# 十六、Phase 10：Teacher 与 Student 速度比较

记录：

1. pointwise multi-seed Teacher 时间；
2. sparse atlas 构建总时间；
3. dense corrector 每点时间；
4. Student CPU batch-1 时间；
5. Student GPU batch-1 时间；
6. chart classifier + expert 时间；
7. 可选 Student + 1–2 DLS corrector 时间。

速度提升定义：

$$
\mathrm{speedup}
================

\frac{
t_{\mathrm{full\ teacher}}
}{
t_{\mathrm{student}}
}.
$$

必须同时报告精度和速度，不能只报告网络推理时间而忽略 chart classification 或前后处理。

---

# 十七、关键消融实验

为证明 BACRA 的价值，必须在同一个任务区域、相同样本量、相同 Student 和相同 split 下比较：

| 数据 ID | 生成方法                              |
| ----- | --------------------------------- |
| D0    | 随机 (\beta_6) FK mixed data        |
| D1    | 每个 (xyz) 独立 pointwise IK          |
| D2    | trajectory/tube continuation data |
| D3    | BACRA product-graph atlas data    |

固定：

$$
N=50{,}000
$$

用于第一轮消融。

比较：

* voxel conflict；
* local beta p95；
* fill distance；
* MLP beta p95；
* FK EE p95/max；
* family/trajectory holdout；
* training stability；
* inference time。

若 D3 只因样本更多而更好，不能归因给 atlas；因此样本量必须相同。

---

# 十八、分阶段执行计划

## Stage A：Smoke

规模：

* FK pool：(2^{16})；
* task nodes：200；
* seeds/node：8；
* dense rows：2k。

目标：验证文件、图、候选、chart 和审计流水线可运行。

---

## Stage B：Pilot

规模：

* capability map：1M；
* task nodes：2k；
* seeds/node：16–32；
* task kNN：12；
* dense data：50k；
* random loops：100；
* path endpoints：100。

进入下一阶段的最低条件：

1. Gold candidate node ratio (\ge95%)；
2. 至少一张 chart 覆盖 (\ge80%) 的 Core-safe task nodes；
3. path/cycle P95 (\le0.5^\circ)；
4. conflict voxel ratio (=0)；
5. dense acceptance ratio (\ge95%)；
6. probe MLP 明显优于 D0/D1。

---

## Stage C：Formal Sparse Atlas

规模：

* task nodes：5k；
* difficult nodes：最高 64 seeds；
* random loops：500；
* path endpoints：500；
* root/chart proposals：8–16。

目标：

* 单 chart 或稳定多 chart atlas；
* 冻结 chart IDs；
* 冻结 path consistency；
* 冻结 Gold region。

---

## Stage D：Formal Dense Dataset

生成：

$$
100k,\quad150k,\quad200k
$$

三个嵌套版本。

学习曲线判断：

* 若 (100k\to150k) 提升显著，继续到 (200k)；
* 若提升小于预注册阈值，例如 EE p95 改善不足 (5%)，停止扩充；
* 不默认追求 (500k)。

---

## Stage E：Model Selection

只使用 train/validation spatial blocks 和 validation trajectories。

锁定：

* Student 类型；
* 网络结构；
* FK loss 权重；
* chart strategy；
* epoch budget。

---

## Stage F：Sealed Evaluation

打开：

* sealed spatial blocks；
* sealed families；
* sealed trajectories。

运行五 seeds，生成最终模型表格和图。

---

# 十九、失败回退与停止规则

## 情况 A：大量 task node 没有 Gold candidate

若：

$$
\text{Gold node ratio}<80%,
$$

说明当前区域与 (1.5^\circ) Gold 余量不兼容。

动作：

1. 缩小区域；
2. 重搜任务区域 center/geometry；
3. 保留 Silver map 作为诊断；
4. 不通过删除失败节点伪造完整区域。

---

## 情况 B：每点都有可行候选，但没有大覆盖 chart

说明存在 branch barriers 或工作空间拓扑分裂。

动作：

* 构建多 chart atlas；
* 若 charts 在相同 (xyz) 区域重叠且标签冲突，使用 chart context 或 stateful model。

---

## 情况 C：路径一致性失败

若不同路径到同一点：

$$
d_{\beta,\mathrm{path,p95}}>1^\circ,
$$

禁止训练静态 (xyz\to\beta_6)。

转向：

$$
(xyz,chart_id)\to\beta_6
$$

或：

$$
(x_t,\beta_{t-1})\to\Delta\beta_t.
$$

---

## 情况 D：稀疏 atlas 通过，但 dense corrector 冲突

说明 chart sampling 过稀或存在局部 fold。

动作：

* 增加 task nodes；
* 缩短 graph edges；
* 在冲突区主动加密；
* 拆分 chart。

---

## 情况 E：数据 Gate 通过但 Student 失败

依次检查：

1. fill distance；
2. label curvature；
3. chart boundary；
4. model capacity；
5. FK loss；
6. 是否存在被遗漏的路径依赖。

此时才属于 Student 模型问题。

---

# 二十、Codex 必须输出的产物

```text
13_summary/
  protocol_summary.md
  capability_map_summary.json
  sparse_candidate_summary.csv
  product_graph_components.csv
  chart_coverage.csv
  path_consistency_report.csv
  cycle_consistency_report.csv
  chart_overlap_report.csv
  dense_dataset_report.json
  cross_chart_conflicts.parquet
  spatial_split_manifest.csv
  data_ablation_table.csv
  student_metrics_per_seed.csv
  teacher_student_timing.csv
  final_recommendation.md
```

关键可视化：

```text
capability_map_3d.png
task_graph_3d.png
product_graph_component_projection.png
canonical_chart_xyz.png
canonical_chart_beta.png
chart_overlap_heatmap.png
fill_distance_heatmap.png
label_lipschitz_heatmap.png
sealed_trajectory_tracking_grid.png
```

---

# 二十一、可直接交给 Codex 的主任务

> Implement Branch-Aware Canonical Region Atlas V12. Freeze the exact quasi-static forward environment, robot configuration, joint domain, data gates, spatial splits, and sealed trajectory seeds. Build or verify a one-million-sample branch-agnostic FK capability map, then define a connected three-dimensional task region around the validated 0.5 m task while sampling the region as an unordered task-space point cloud rather than a single trajectory. For each task node, generate multiple distinct IK candidates using FK-pool seeds, weighted damped-Jacobian iteration, bounded SQP, null-space exploration, and neighboring-node warm starts. Never pad candidate sets with unconverged, clipped, or residual-failing seeds. Construct a task/configuration product graph whose edges are verified by predictor-corrector continuation. Extract root-induced candidate sections, optimize one candidate label per task node on the graph, and split the result into local charts whenever the same task node retains multiple incompatible configuration clusters. Audit path independence, random task-space loops, forward/reverse consistency, repeatability, and chart overlaps before authorizing static inverse data. Densify validated charts using task-space tetrahedral interpolation, weighted-Jacobian prediction, and exact FK correction, requiring independent-anchor agreement and final FK verification. Actively enrich Cartesian holes, chart boundaries, high-conditioning regions, and high-label-curvature areas until the registered fill-distance target is reached. Assemble nested 50k, 100k, 150k, and 200k datasets using complete spatial blocks, families, and trajectories as split units. Compare mixed-FK, independent pointwise IK, trajectory/tube, and BACRA data at equal sample counts and with the same Student model. Train a bounded beta6 MLP with differentiable FK loss when the atlas is path-independent; otherwise train chart-conditioned experts or a stateful policy according to the audit result. Keep sealed spatial blocks and trajectories inaccessible until the atlas and model protocol are frozen, and report Teacher accuracy, Student FK accuracy, trajectory smoothness, branch conflicts, coverage, and computation-time speedup separately.

---

# 最终方法概括

本实验不再把问题定义成：

$$
\text{在工作空间采点}
\rightarrow
\text{每点随便求一个 IK}
\rightarrow
\text{训练 MLP}.
$$

而是：

$$
\boxed{
\text{先找出严格可行的局部 inverse charts}
\rightarrow
\text{验证 charts 是否路径一致}
\rightarrow
\text{在 chart 内稠密生成数据}
\rightarrow
\text{根据 atlas 结构选择 Student}
}
$$

这样生成的 (100k\sim200k) 数据才真正代表一个三维任务区域，而不是一条椭圆或一组互相冲突的随机 IK 标签。

[1]: https://arxiv.org/abs/2012.13965?utm_source=chatgpt.com "Efficient Jacobian-Based Inverse Kinematics with Sim-to-Real Transfer of Soft Robots by Learning"
[2]: https://zenodo.org/records/3759636?utm_source=chatgpt.com "Model-Based Reinforcement Learning for Closed-Loop Dynamic Control of Soft Robotic Manipulators | Zenodo"
[3]: https://arxiv.org/abs/1909.12517?utm_source=chatgpt.com "TORM: Fast and Accurate Trajectory Optimization of Redundant Manipulator given an End-Effector Path"
[4]: https://arxiv.org/abs/2402.16154?utm_source=chatgpt.com "IKLink: End-Effector Trajectory Tracking with Minimal Reconfigurations"
[5]: https://journals.sagepub.com/doi/abs/10.1089/soro.2016.0065?utm_source=chatgpt.com "Nonparametric Online Learning Control for Soft Continuum Robot: An Enabling Technique for Effective Endoscopic Navigation - Kit-Hang Lee, Denny K.C. Fu, Martin C.W. Leong, Marco Chow, Hing-Choi Fu, Kaspar Althoefer, Kam Yim Sze, Chung-Kwong Yeung, Ka-Wai Kwok, 2017"
[6]: https://arxiv.org/abs/1705.07637?utm_source=chatgpt.com "Kinodynamic Planning on Constraint Manifolds"

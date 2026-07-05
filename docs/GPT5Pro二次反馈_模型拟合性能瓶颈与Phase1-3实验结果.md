# GPT5Pro 二次反馈：模型拟合性能瓶颈与 Phase 1-3 实验结果

生成日期：2026-06-08  
项目路径：`/mnt/ML_projects/quasi_exp`

## 0. 给 GPT5Pro 的新任务说明

这是上次发送 `GPT5Pro实验决策背景.md`、`gpt5pro实验计划.md` 和原论文之后的二次反馈材料。我们已经按上次方案做了一轮 Phase 1-3 诊断与 baseline 实验，现在希望你基于新的证据重新判断下一阶段实验路线。

当前最终目标需要明确调整为：

```text
提高模型拟合性能：让 xyz -> theta_1..30_rad + tension_1..12_n 的预测误差显著下降。
```

张力一致性、theta/branch 生成、canonical allocator、采样策略都不是最终目标本身，而是为了让监督学习目标变得更单值、更连续、更可学习。现在的证据显示，模型拟合性能差很可能同时受两个上游问题限制：

1. `xyz -> theta/beta` 不是足够单值，复杂采样下同一 workspace 小区域存在多个构型 branch。
2. `theta/beta -> tension` 的 canonical 张力标签在复杂采样下仍不够连续，尤其是 same-family/same-branch 邻域的张力 p95 还偏高。

因此这次希望你重点回答：

- 如果主目标是模型拟合性能，下一步应该优先优化 `theta/branch` 生成方法，还是优先优化张力 canonical allocator？
- 是否应该先构造 single-branch canonical dataset，而不是继续保留当前 mixed 多 branch 数据？
- 如何把“优先第三关节、前两段小”的原则转化为严格、可复现、可验收的 branch/theta selection policy？
- 在 beta-first baseline 未超过 direct baseline 后，是否仍建议把 beta 作为中间监督/辅助 loss？
- 在 20k 规模上应达到什么验收标准，才值得扩展到 100k？

## 1. 当前参考数据集与旧基线

本轮诊断主要围绕当前最好的 20k 参考集：

- 数据：`data/mixed_beta_20k_distal_preferred_anchor_v2_second_stage_from_anchor_v2_k32_w20/dataset.parquet`
- meta：`data/mixed_beta_20k_distal_preferred_anchor_v2_second_stage_from_anchor_v2_k32_w20/dataset_meta.parquet`
- 旧 direct baseline：
  - `runs/baselines_mixed_beta_20k_distal_preferred_anchor_v2_second_stage_from_anchor_v2_k32_w20_iid_fast4/SUMMARY.md`
  - `runs/baselines_mixed_beta_20k_distal_preferred_anchor_v2_second_stage_from_anchor_v2_k32_w20_radius_fast4/SUMMARY.md`
  - `runs/baselines_mixed_beta_20k_distal_preferred_anchor_v2_second_stage_from_anchor_v2_k32_w20_beta_block_fast4/SUMMARY.md`
  - `runs/baselines_mixed_beta_20k_distal_preferred_anchor_v2_second_stage_from_anchor_v2_k32_w20_angular_sector_fast4/SUMMARY.md`

旧 direct baseline 的最佳结果：

| split | direct best model | theta MAE deg | T MAE N | EE p95 mm |
| --- | --- | ---: | ---: | ---: |
| iid | mlp | 2.458 | 49.58 | 49.75 |
| radius | mlp_large | 1.948 | 43.84 | 55.64 |
| beta_block | mlp | 3.445 | 59.23 | 77.37 |
| angular_sector | mlp_large | 2.567 | 51.09 | 89.38 |

四个 split 的最佳张力 MAE 平均约 `50.94 N`。这就是当前需要突破的模型拟合基线。

## 2. Phase 1：诊断主瓶颈

输出文件：

- `runs/diagnostics/gpt5pro_plan_phase1_20k/diagnosis_report.md`

Phase 1 的主结论：

```text
primary_bottleneck = workspace_multibranch_bottleneck
```

原因是：hard gate 通过，beta-close/same-branch continuity 明显好于 all-workspace continuity，但 all-workspace 近邻仍有很高的 theta/T 跳变。

### 2.1 hard gate

当前 20k 参考集质量门槛通过：

| 指标 | 数值 |
| --- | ---: |
| rows | 20000 |
| rms_rnorm q95 | 0.038570 |
| max tension | 1478.78 N |
| saturation ratio | 0 |
| tension < 0 ratio | 0 |
| tension > 2000 ratio | 0 |

这说明数据不是因为残差、超张力或求解失败而不可用。

### 2.2 局部连续性

`anchor_v2_second_stage` 的 10mm workspace 邻域结果：

| group | pairs | T p50 N | T p90 N | T p95 N | theta p95 deg |
| --- | ---: | ---: | ---: | ---: | ---: |
| all_xyz_<=10mm | 10363 | 67.38 | 118.65 | 134.72 | 8.341 |
| xyz_<=10mm_beta_close | 291 | 16.32 | 35.25 | 41.55 | 1.196 |
| xyz_<=10mm_beta_far | 10072 | 68.52 | 119.37 | 135.70 | 8.365 |

解释：

- 如果在 workspace 里直接找近邻，theta p95 仍达到 `8.341 deg`，张力 p95 仍达到 `134.72 N`。
- 如果限制为 beta-close 邻域，theta p95 降到 `1.196 deg`，张力 p95 降到 `41.55 N`。
- 这说明同一 workspace 小区域附近存在 beta/theta 差异很大的点对，单值 `xyz -> theta/T` 学习会被迫平均多个 branch。

### 2.3 branch clustering 与 oracle floor

branch clustering 结果：

| 指标 | 数值 |
| --- | ---: |
| balls evaluated | 536 |
| multi-branch ball ratio | 1.000 |
| branch count p50 / p90 | 6 / 9 |
| within-branch T p50 / p95 | 0 / 0 N |
| between-branch T p50 / p95 | 76.01 / 100.33 N |
| between-branch variance ratio p50 / p95 | 1.000 / 1.000 |

Oracle floor：

| oracle | T MAE N | T p95 N | theta MAE deg | EE p95 mm |
| --- | ---: | ---: | ---: | ---: |
| xyz_nn_oracle | 67.18 | 130.77 | 3.238 | 23.14 |
| branch_aware_xyz_nn_oracle | 62.92 | 126.78 | 3.129 | 33.95 |
| beta_nn_oracle | 23.12 | 47.83 | 0.622 | 150.75 |

解释：

- `xyz_nn_oracle` 本身已经和模型误差同量级，说明输入 `xyz` 对当前标签不是充分信息。
- `beta_nn_oracle` 的张力 MAE 明显更低，说明如果构型 branch 已知，张力标签更可学习。
- 但 `beta_nn_oracle` 的 EE p95 较高，提示 beta 近邻并不等同于 workspace 近邻，后续不能只用 beta 平滑而忽略末端位置误差。

Phase 1 结论：当前最大问题不是 simple model capacity，而是 workspace 多 branch 导致 `xyz -> theta/T` 监督目标不够单值。

## 3. Phase 2：source ablation、branch filtering 与 branch-aware baseline

输出文件：

- `runs/diagnostics/gpt5pro_plan_phase2_summary.md`
- `runs/diagnostics/gpt5pro_plan_phase2_source_ablation/source_ablation_report.md`
- `runs/diagnostics/gpt5pro_plan_phase2_branch_filtered_s1/diagnosis_report.md`

Phase 2 的目标是判断：branch 混叠是否来自某个特定 source，简单 voxel filtering 能否修复，exact voxel branch label 能否用于 branch-aware 模型。

### 3.1 source ablation

10mm 风险排序：

| rank | source | rows | risk score | cross pairs | cross beta-far pairs | within T p95 N |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | sobol_full | 8000 | 16975.517 | 4242 | 4191 | 132.49 |
| 2 | workspace_balanced | 4000 | 13114.842 | 3278 | 3235 | 136.92 |
| 3 | distal_biased | 4000 | 12823.073 | 2990 | 2957 | 126.78 |
| 4 | lhs_full | 4000 | 11123.946 | 2864 | 2827 | 112.01 |

解释：

- branch 混叠不是只由 `workspace_balanced` 引入。
- 几乎所有 cross-source 10mm pair 都是 beta-far。
- `sobol_full` 风险最高 partly because 它占 8000 行，是其他 source 的两倍。
- 单纯删掉某一个 source 不能根治问题，更需要全局 branch selection policy。

### 3.2 branch-consistent filtering S1

输出数据：

- `data/mixed_beta_20k_distal_preferred_anchor_v2_branch_filtered_s1`

过滤策略：

- workspace voxel：10mm
- beta DBSCAN eps：0.15
- policy：`distal_v1`
- score：低 `distal_preference_score`，弱 max-tension/rms penalty，branch-size stabilizer

过滤结果：

| metric | value |
| --- | ---: |
| rows in | 20000 |
| rows out | 17818 |
| retention ratio | 0.8909 |
| voxels | 17808 |
| total branches | 19989 |
| multi-branch voxel ratio | 0.1083 |

S1 hard gate 通过，但没有真正修复 all-workspace branch mixing：

| group | pairs | T p95 N | theta p95 deg |
| --- | ---: | ---: | ---: |
| all_xyz_<=10mm | 5613 | 130.99 | 8.358 |
| xyz_<=10mm_beta_close | 201 | 41.13 | 1.192 |
| xyz_<=10mm_beta_far | 5412 | 131.79 | 8.415 |

解释：

- S1 只在单个 voxel 内选 branch，去掉了一部分重复分支。
- 但相邻 voxel / workspace ball 之间仍然存在多 branch 混叠。
- all-workspace theta p95 仍是 `8.358 deg`，没有达到单 branch 数据集应有的 <3deg 目标。

### 3.3 exact voxel branch-aware baseline

输出：

- `runs/baselines_mixed_beta_20k_anchor_v2_branch_aware_knn`
- `runs/baselines_mixed_beta_20k_anchor_v2_branch_aware_knn_min1`

结果：

| setting | branch count | other ratio | classifier acc | classifier-expert T MAE N | xyz-NN T MAE N |
| --- | ---: | ---: | ---: | ---: | ---: |
| min_branch_size=3 | 1 | 1.000 | 1.000 | 66.92 | 66.92 |
| min_branch_size=1 | 19989 | 0.000 | 0.001 | 69.04 | 66.92 |

解释：

- `min_branch_size=3` 时，几乎所有 branch 都塌缩成 `other_branch`，没有监督意义。
- `min_branch_size=1` 时，branch label 接近 per-sample，无法从 xyz 分类。
- exact `voxel_key + local beta cluster` 不适合作为 supervised branch class。

Phase 2 结论：需要更粗、更全局、更物理可解释的 branch/theta policy，而不是 exact voxel branch label。

## 4. Phase 3：global beta families 与 beta-first baseline

输出文件：

- `runs/diagnostics/gpt5pro_plan_phase3_summary.md`
- `runs/diagnostics/gpt5pro_plan_phase3_beta_families_effective_theta/GLOBAL_BETA_FAMILIES_REPORT.md`
- `runs/baselines_mixed_beta_20k_anchor_v2_beta_first_effective_theta_summary.json`

Phase 3 目标是测试两个方向：

1. 在 global normalized beta 空间定义 coarse branch families。
2. 训练 beta-first baseline：`xyz/poly -> beta6 -> theta/T`。

### 4.1 重要 schema/sign 发现

当前数据的 `dataset_meta.parquet` 里的 `beta*_rad` 不能直接当作 dataset theta 的几何 beta。

核验结果：

| beta source | vs dataset theta |
| --- | ---: |
| `beta_to_theta(meta_beta)` | MAE 6.6759 deg，p95 17.5957 deg |
| `beta_to_theta(-meta_beta)` | MAE 0 deg |
| `effective_beta_from_theta(theta)` | 近似 0 deg |

因此后续 Phase 3 脚本默认改为：

```text
--beta-source effective_theta
```

它从真实 30 维 theta 中按每 10 个盘的 odd/even 均值反推 beta6。这一点很重要：否则 beta-first 的 theta/EE 指标会被符号问题污染。

### 4.2 global normalized beta families

门槛：

- min family size >= `100`
- classifier accuracy >= `0.70`
- same-family 10mm tension p95 <= `70 N`

结果：

| k | pass | min size | RF acc | KNN acc | same-family T p95 N | cross-family T p95 N |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | False | 1909 | 0.454 | 0.414 | 105.03 | 141.32 |
| 16 | False | 946 | 0.377 | 0.350 | 90.01 | 140.80 |
| 32 | False | 474 | 0.260 | 0.222 | 82.39 | 138.87 |
| 64 | False | 228 | 0.193 | 0.170 | 73.06 | 137.62 |

解释：

- coarse beta family 可以减少一部分跨 branch 跳变，但 same-family 张力 p95 仍没有达到 `70 N` 门槛。
- family 分类从 xyz/poly 特征预测很弱，RF acc 从 `0.454` 降到 `0.193`。
- 这说明现在不适合直接做 hard branch classifier/gate；误分支会很严重。

### 4.3 beta-first baseline

beta-first 训练结构：

```text
xyz/poly -> beta6(effective_theta) -> theta_1..30 + tension_1..12
```

最佳结果：

| split | best beta-first model | beta/theta MAE deg | T MAE N | T p95 N | EE p95 mm |
| --- | --- | ---: | ---: | ---: | ---: |
| iid | rf__direct | 2.454 | 52.18 | 100.99 | 50.29 |
| radius | mlp__direct | 1.945 | 44.74 | 80.28 | 47.50 |
| beta_block | mlp__direct | 3.443 | 63.22 | 120.03 | 75.61 |
| angular_sector | mlp__direct | 2.531 | 52.63 | 90.05 | 103.94 |

四 split 平均 best T MAE：`53.19 N`。

### 4.4 beta-first 与 direct baseline 对比

| split | direct best model | direct T MAE N | beta-first T MAE N | direct EE p95 mm | beta-first EE p95 mm |
| --- | --- | ---: | ---: | ---: | ---: |
| iid | mlp | 49.58 | 52.18 | 49.75 | 50.29 |
| radius | mlp_large | 43.84 | 44.74 | 55.64 | 47.50 |
| beta_block | mlp | 59.23 | 63.22 | 77.37 | 75.61 |
| angular_sector | mlp_large | 51.09 | 52.63 | 89.38 | 103.94 |

解释：

- beta-first 没有超过 direct regression。
- 修正 beta-source/sign 后，它的 EE 指标恢复到合理量级，但张力 MAE 没有改善。
- 这说明把 beta 作为连续中间变量有诊断价值，但不是当前性能突破点。
- 如果 `xyz -> beta` 本身仍然多值，beta-first 仍会学到 branch 平均解。

Phase 3 结论：当前不能指望仅靠 `xyz -> beta6 -> theta/T` 结构突破模型拟合瓶颈。需要先解决 branch/theta 生成策略和张力 canonical 连续性。

## 5. 当前总体判断

综合 Phase 1-3，现在的结论应从“张力一致性优化”进一步改成：

```text
模型拟合性能差，是因为学习目标本身在复杂采样下不够单值、不够连续。
```

更具体地说：

1. `xyz -> theta/beta` 存在 workspace 多 branch。  
   all-workspace 10mm theta p95 约 `8.34 deg`，而 beta-close theta p95 约 `1.20 deg`。这直接影响 theta 拟合和 EE 回代误差。

2. `theta/beta -> tension` 的 canonical allocator 在复杂采样下仍不够平滑。  
   beta-close 张力 p95 能到 `41.55 N`，但 same-family/global family 仍有 `73-105 N` p95；all-workspace 更高。

3. direct baseline 已经接近当前数据定义下的可学习下限。  
   四 split best T MAE 约 `50.94 N`；beta-first 约 `53.19 N`，没有突破。

4. exact branch label 不可用。  
   per-voxel branch label 要么塌缩，要么接近每样本唯一，无法从 xyz 监督分类。

5. 简单扩数据或简单堆模型不太可能解决根因。  
   之前 mixed 100k 对 20k 的直接回归改善很小，说明数据量不是唯一瓶颈。

## 6. 需要 GPT5Pro 重新判断的问题

请基于以上新证据，重新给出一个以“提高模型拟合性能”为核心目标的实验方案。希望重点回答以下问题。

### 6.1 theta/branch generation

1. 是否应该先构造 single-branch canonical dataset？
2. 如果是，branch selection policy 应如何定义？
3. 我们提出的原则是“优先第三关节、前两段小”：同一个 workspace 位置附近，如果多个构型可达，优先选择第三关节幅度更大、第一/第二关节幅度更小的 theta。这个原则应该如何数学化？
4. policy 应该在 beta 空间、theta 空间还是 workspace voxel/ball 中执行？
5. 只靠 `distal_preference_score` 是否足够，还是还需要加入 max tension、residual、branch density、路径连续性？

### 6.2 tension canonical allocator

1. 在 branch policy 固定之前，继续优化张力 allocator 是否会被多 branch 问题掩盖？
2. 是否应该把张力 canonical 作为 branch selection score 的一部分，而不只是后处理 relabel？
3. 现有 anchor v2 second-stage 已把 all-workspace T p95 从约 `250 N` 降到 `134.72 N`，但 theta p95 没降。下一步 allocator 应该如何避免只优化 T、不解决 theta branch？
4. 是否建议做 integrated allocator，而不是 relabel 后处理？
5. same-branch / beta-close 张力 p95 应降低到多少，才足以支撑模型 T MAE 降到 35-40N 以下？

### 6.3 beta-first 和模型结构

1. beta-first 没有超过 direct baseline 后，还应不应该保留 beta 作为辅助监督？
2. 是否应该训练 `xyz -> canonical beta`，但不直接用它替代 direct model，而是作为 consistency loss 或 branch regularizer？
3. 如果 branch classifier 准确率只有 0.19-0.45，是否还有必要做 MoE/classifier+expert？
4. 模型是否必须增加额外输入才能让 inverse mapping 单值，例如 branch hint、目标姿态、路径历史、上一时刻 theta/T？
5. 如果坚持输入只有 `x,y,z`，理论上能达到怎样的 theta/T 下限？

### 6.4 20k 到 100k 扩展门槛

请给出新的 20k 晋级 100k 的门槛。我们倾向于如下方向，但希望你修正：

| 类别 | 建议门槛 |
| --- | --- |
| hard gate | rms_rnorm q95 <= 0.06，max tension <= 2000N，saturation=0 |
| single-branch theta | all-workspace 10mm theta p95 < 3deg |
| single-branch tension | all-workspace 10mm T p95 < 100N |
| same-branch tension | same-branch 10mm T p95 < 50-70N |
| 20k model | 四 split 平均 T MAE <= 35-40N |
| EE | EE p95 不比当前 anchor v2 明显恶化 |

如果你认为这些门槛不合理，请给出更适合“模型拟合性能提升”的门槛。

## 7. 我们当前倾向的下一步，但需要 GPT5Pro 判断

目前我们倾向于下一阶段不要先做复杂模型，而是先做一个新的数据/标签生成实验：

1. 在 workspace 局部邻域内显式执行 canonical branch selection。
2. branch score 以 distal-preferred 为主：

```text
score = + w3 * ||beta_5:6||
        - w12 * ||beta_1:4||
        - wT * normalized_max_tension
        - wR * normalized_rms_rnorm
        - wJ * local_theta_jump
```

3. 生成或过滤一个 20k single-branch/distal-preferred canonical dataset。
4. 对这个数据集跑：
   - hard gate
   - all-workspace 10mm continuity
   - beta-close / same-branch continuity
   - direct baseline fast4
   - beta-first diagnostic baseline
5. 如果 20k 的 theta/T/EE 显著改善，再生成 100k。

但这里还有几个关键不确定点：

- branch selection 是过滤已有样本，还是重新生成样本时主动选择？
- workspace voxel 大小应该是 5mm、10mm，还是基于局部密度自适应？
- `优先第三关节` 是否会牺牲可达空间覆盖或导致张力过高？
- 如果某个 workspace 区域只能由前两段大角度到达，policy 是否应保留它，还是从训练分布中剔除？
- 是否应该把输入从 `xyz` 扩展到 `xyz + branch policy hint`，以承认仅靠 xyz 逆解不唯一？

请基于原论文和以上实验结果，给出下一阶段最小可行但能真正提升模型拟合性能的实验方案。

# 回答
根据 2026-06-08 的实验反馈文档，下一步的主线应当改为：

[
\boxed{\text{先构造可由 }xyz\text{ 单值监督学习的 canonical branch/theta 数据集，再在该 branch 内优化张力 allocator。}}
]

不要让 Codex 先继续堆模型、做 MoE、调 beta-first 或单独做全局张力平滑。当前证据已经比较清楚：模型拟合差的主要矛盾不是模型容量，而是监督目标本身在复杂 mixed 采样下不够单值、不够连续。任务目标也已经从单纯“张力一致性”调整为“提高 (xyz\rightarrow \theta_{1:30}+T_{1:12}) 的模型拟合性能”，并明确指出张力一致性、branch 生成、canonical allocator 都只是让监督学习目标更可学的手段。

---

## 1. 当前实验原理审查：结论方向是对的，但下一步不能再按 voxel 过滤或 hard branch classifier 做

### 1.1 Phase 1 的判断是正确的：主要矛盾已经变成 workspace 多 branch

当前 20k anchor v2 数据的 hard gate 是通过的：rows = 20000，`rms_rnorm q95 = 0.038570`，最大张力 1478.78 N，饱和率和越界率为 0；所以数据不可学不是因为残差失控、张力超限或求解失败。

关键证据是 10mm workspace 邻域：

| group                 |    T p95 | theta p95 |
| --------------------- | -------: | --------: |
| all_xyz_<=10mm        | 134.72 N |    8.341° |
| xyz_<=10mm_beta_close |  41.55 N |    1.196° |
| xyz_<=10mm_beta_far   | 135.70 N |    8.365° |

这说明同一个 workspace 小区域内混有 beta/theta 差异很大的 branch。单值 (xyz\rightarrow \theta/T) 模型被迫学习多个 branch 的平均值，所以 theta、T、FK 回代都会受损。

branch clustering 和 oracle 也支持这个判断：multi-branch ball ratio = 1.000，branch count p50/p90 = 6/9；`xyz_nn_oracle` 的 T MAE 已经达到 67.18 N，而 `beta_nn_oracle` 的 T MAE 只有 23.12 N。这说明 xyz 本身对当前标签不是充分信息，而如果构型 branch 已知，张力明显更可学。

### 1.2 Phase 2 的失败也合理：S1 只是“单 voxel 局部过滤”，不是全局 branch policy

S1 用 10mm voxel、beta DBSCAN、`distal_v1` policy 做 branch filtering，保留率 0.8909，但 all-workspace 10mm theta p95 仍为 8.358°，T p95 仍为 130.99 N；beta-close 区域仍好，beta-far 区域仍坏。也就是说，S1 只是去掉了单 voxel 内的一部分重复分支，没有解决相邻 voxel / workspace ball 之间的 branch 跳变。

source ablation 也说明问题不是某一个 source 引起的：`sobol_full/workspace_balanced/distal_biased/lhs_full` 都存在高 cross beta-far 风险，单纯删除某个 source 不能根治问题。

exact voxel branch-aware baseline 也不应该继续：`min_branch_size=3` 时 branch 塌缩成 `other_branch`，`min_branch_size=1` 时接近 per-sample label，无法从 xyz 分类。这说明 exact `voxel_key + local beta cluster` 不是可监督的 branch class。

### 1.3 Phase 3 的 beta-first 结果说明：beta 有诊断价值，但不能作为当前突破口

Phase 3 发现 `dataset_meta.parquet` 的 `beta*_rad` 与 dataset theta 存在符号问题：`beta_to_theta(meta_beta)` 对 theta 的 MAE 为 6.6759°，而 `beta_to_theta(-meta_beta)` 为 0°；因此后续应使用 `effective_beta_from_theta(theta)`。这个修正非常重要，否则所有 beta-first、branch family、beta continuity 都会被符号污染。

但修正后，beta-first 仍没有超过 direct baseline。direct 四 split 平均 T MAE 约 50.94 N，beta-first 约 53.19 N；文档也明确指出 beta-first 没有改善张力 MAE，只说明 beta 作为连续中间变量有诊断价值，不是当前性能突破点。

global beta families 也不适合直接拿来做 hard branch classifier：k=8 到 64 的 RF accuracy 只有 0.454 降到 0.193，same-family T p95 仍为 73–105 N，没有达到 70 N 门槛。

---

## 2. 主次矛盾判断

### 主要矛盾

当前主要矛盾是：

[
\boxed{xyz\rightarrow \theta/\beta/T \text{ 的监督标签不是单值函数}}
]

这不是单纯张力问题。anchor v2 second-stage 已经把张力 p95 从约 250 N 降到 134.72 N，但 theta p95 仍约 8.34°，说明张力后处理无法解决构型分支混叠。

### 主要矛盾的主要方面

主要方面是：

[
\boxed{theta/branch\ generation}
]

因为只要同一个 (xyz) 附近混有多个 theta branch，任何 (T(\theta)) 的平滑化都会被 all-workspace 近邻评价掩盖。你可以把同一 branch 内的张力做得很好，但模型仍然会因为输入 (xyz) 不知道该选哪条 branch 而学平均解。

### 次要矛盾

次要矛盾是：

[
\boxed{theta/beta\rightarrow T \text{ 的 canonical allocator 在复杂 branch 内仍不够平滑}}
]

beta-close T p95 已经降到 41.55 N，但 same-family/global family 仍有 73–105 N p95，这说明 branch 选好之后仍需要 allocator 优化。

### 主要矛盾的演化

之前主要矛盾是 PSO/论文式张力多解。原始 PSO 在同 theta 下 8 seeds 可行率 1.0，但 median pairwise MAE 731.85 N，p95 1108.30 N，说明 Eq.(52)+(53) 类约束缺少唯一化 tie-break。segmented canonical 解决了 same-theta 随机性后，主要矛盾转移到 cross-sample / workspace branch 混叠。

---

## 3. 下一步实验方向：不要先做复杂模型，先做 Phase 4 single-branch canonical dataset

我建议交给 Codex 的最小可行实验是：

[
\boxed{\text{Phase 4: Graph-consistent distal-preferred single-branch canonical dataset, 20k pilot}}
]

当前 mixed 多 branch 数据要保留，但只作为 stress test / OOD 诊断集，不再作为主训练集。原论文确实是从 3D Cartesian end point 预测 30 个 joint angles 和 12 个 tensions 的 MLP，输出维度 42；这一路线只有在你人为定义了一个 deterministic canonical inverse map 后才成立。

---

## 4. Codex 应该实现的核心实验

### 实验 4A：先做 graph-policy filtering dry run，不直接大规模重生成

目的：用现有 20k 数据快速验证“全局 branch selection policy”是否能显著降低 theta/T 跳变。

这一步不是重复 S1。S1 的错误是按单个 10mm voxel 独立选 branch；新方案必须在 workspace graph 上选 branch，使相邻 voxel/ball 的 branch 选择互相约束。

Codex 需要实现：

```text
scripts/analysis/select_canonical_branch_graph.py
```

输入：

```text
data/mixed_beta_20k_distal_preferred_anchor_v2_second_stage_from_anchor_v2_k32_w20/dataset.parquet
data/mixed_beta_20k_distal_preferred_anchor_v2_second_stage_from_anchor_v2_k32_w20/dataset_meta.parquet
```

必须先生成：

```text
effective_beta_1..6
```

不要直接用 meta beta，因为 Phase 3 已经证明 meta beta 与 theta 有符号不一致问题。

输出：

```text
data/mixed_beta_20k_anchor_v2_graph_branch_filtered_s2/
```

核心方法：

1. 用 workspace voxel 建图，先用 10mm voxel，边连接 26-neighborhood 或中心距离 (\leq 15mm) 的 voxel。
2. 每个 voxel/ball 内按 effective_beta/theta 聚类，得到候选 branch。
3. 对每个候选 branch 计算 unary score。
4. 在相邻 voxel 之间加 pairwise smoothness cost。
5. 用 deterministic greedy propagation + ICM 迭代优化即可，不必一开始上复杂图割。

总能量：

[
E(b)=\sum_i U_i(b_i)+\lambda\sum_{(i,j)\in \mathcal{E}}V_{ij}(b_i,b_j)
]

其中 (i,j) 是 workspace graph 节点，(b_i) 是该节点选中的 branch。

---

## 5. “优先第三关节、前两段小”应数学化为 branch selection policy，而不是采样口号

### 5.1 先定义 effective beta

对每个样本，从真实 theta 反推：

[
\beta_1=\operatorname{mean}(\theta_{1,3,5,7,9}),\quad
\beta_2=\operatorname{mean}(\theta_{2,4,6,8,10})
]

[
\beta_3=\operatorname{mean}(\theta_{11,13,15,17,19}),\quad
\beta_4=\operatorname{mean}(\theta_{12,14,16,18,20})
]

[
\beta_5=\operatorname{mean}(\theta_{21,23,25,27,29}),\quad
\beta_6=\operatorname{mean}(\theta_{22,24,26,28,30})
]

然后定义三段幅值：

[
B_1=\sqrt{\beta_1^2+\beta_2^2}
]

[
B_2=\sqrt{\beta_3^2+\beta_4^2}
]

[
B_3=\sqrt{\beta_5^2+\beta_6^2}
]

### 5.2 branch unary score

不要只用 distal preference。建议用硬约束 + 连续性 + distal preference + 力学安全共同定义。

先做硬过滤：

[
|FK(\theta)-p_{target}|\le \epsilon_p
]

[
rms_rnorm\le 0.06,\quad 0\le T_j\le 2000N,\quad saturation=0
]

再定义可最大化的 score：

[
S_i(b)=
w_3\tilde B_3
-w_2\tilde B_2
-w_1\tilde B_1
-w_T\widetilde{T}_{max}
-w_R\widetilde r
+w_N\log(1+n_b)
]

其中：

* (\tilde B_1,\tilde B_2,\tilde B_3) 用各自允许范围或 q95 归一化。
* (\widetilde{T}*{max}=T*{max}/2000)。
* (\widetilde r=rms_rnorm/0.06)。
* (n_b) 是该 branch 在局部 ball 内的样本数，避免选 singleton branch。

初始权重建议：

```text
w3 = 1.0
w2 = 0.45
w1 = 0.65
wT = 0.25
wR = 0.40
wN = 0.05
```

注意：这些权重只是初始值。Codex 应该实现 weight sweep，而不是把它们写死。

### 5.3 pairwise smoothness cost

相邻 workspace node 的 branch 不应突然跳变：

[
V_{ij}(b_i,b_j)
===============

\lambda_\theta\operatorname{rmsdeg}(\theta_i^{b_i}-\theta_j^{b_j})^2
+\lambda_\beta|\beta_i^{b_i}-\beta_j^{b_j}|_2^2
+\lambda_T\operatorname{MAE}(T_i^{b_i}-T_j^{b_j})^2
]

初始建议：

```text
lambda_theta = 1.0
lambda_beta  = 0.5
lambda_T     = 0.15
```

解释：先保证 theta/branch 连续，再保证 tension 平滑。否则会再次出现“张力降低了，但 theta p95 没变”的问题。

### 5.4 policy 执行空间

policy 不应只在 beta 空间、theta 空间或单 voxel 中执行，而应当：

```text
workspace graph 上执行选择；
effective_beta/theta 空间中定义 branch identity；
tension/residual 作为力学安全和 tie-break。
```

也就是说，branch 的选择索引在 workspace 中，branch 的相似性度量在 beta/theta 中。

---

## 6. 实验 4B：主动重生成 single-branch 20k，而不是只过滤已有样本

Graph filtering 只能作为 dry run。真正要提升模型拟合性能，应该主动生成新数据：

```text
data/mixed_beta_20k_single_branch_distal_v2/
```

原因是 S1 已证明过滤已有样本很难修复相邻 voxel 的 branch mixing，而且过滤会造成 workspace density 不均匀。

Codex 应实现：

```text
scripts/generate_single_branch_dataset.py
```

生成流程：

1. 从当前 20k/100k mixed 数据估计 reachable workspace hull。
2. 采样 20k 个 target xyz，优先保证 workspace coverage，而不是 beta coverage。
3. 每个 target xyz 生成候选 beta/theta：

   * 从已有数据找 (R=20\sim30mm) 内候选。
   * 加入 local beta perturbation。
   * 必要时做小规模 local optimization，使 FK 靠近 target。
4. 对候选 theta 调用 segmented/integrated tension solver。
5. 用上面的 graph branch policy 选 canonical branch。
6. 输出 dataset + meta + branch_policy_report。

这里最重要的是：每个 workspace target 最终只保留一个 canonical branch。不是“同一个位置附近采很多构型”，而是“同一个位置附近生成很多候选，然后只选一个可复现 branch”。

---

## 7. 张力 allocator 下一步怎么做

### 7.1 不要在 branch policy 固定前单独做 global tension smoothing

在 branch policy 未固定前继续调张力 allocator，会被多 branch 问题掩盖。all-workspace T p95 可能下降，但 theta p95 不会下降，模型仍然学平均 branch。

### 7.2 应该把 tension canonical 纳入 branch score，而不是只做后处理

张力应参与 branch 选择：

[
-w_T\widetilde{T}_{max}
-w_R\widetilde r
]

但它不能主导 branch 选择。优先级应是：

```text
hard feasibility
> FK/coverage
> theta/beta branch continuity
> distal preference
> tension/residual tie-break
```

### 7.3 branch 固定后，再做 integrated allocator

现有 anchor v2 是有效 baseline：它把 mixed 数据 T p95 明显降低，并保持 hard gate 通过，但 theta p95 没降。下一步应改成 selected-branch graph 内的 integrated allocator：

[
T_i^*=
\arg\min_T
|r(\theta_i,T)|^2
+\lambda_{ref}|T-T^{ref}*i|^2
+\lambda*{nom}|T-T_{nom}|^2
+\lambda_{max}\operatorname{smoothmax}(T)
]

其中 (T_i^{ref}) 只能来自 selected same-branch graph 邻居，不能来自 all-workspace 邻居。

Codex 可复用：

```text
scripts/analysis/relabel_anchor_canonical.py
src/quasi_exp/opt/segmented_tension.py
```

但要新增参数：

```text
--anchor-scope selected_branch_graph
--distance-space effective_beta
--no-cross-branch-anchor
```

---

## 8. beta-first 之后，beta 还要不要保留？

要保留，但不要作为硬级联瓶颈。

不推荐继续：

[
xyz \rightarrow \beta \rightarrow \theta/T
]

因为 beta-first 已经没有超过 direct regression，且 (xyz\rightarrow\beta) 本身仍然多值。

推荐改为 auxiliary loss：

[
h=f_\phi(x,y,z)
]

[
\hat\theta=g_\theta(h),\quad \hat T=g_T(h),\quad \hat\beta=g_\beta(h)
]

loss：

[
\mathcal{L}
===========

\mathcal{L}*\theta
+\mathcal{L}*T
+\lambda*{FK}|FK(\hat\theta)-xyz|
+\lambda*\beta|\hat\beta-\beta_{eff}|
+\lambda_{cons}|\beta_{eff}(\hat\theta)-\hat\beta|
]

其中 (\lambda_\beta,\lambda_{cons}) 不要太大，建议从 0.05、0.1、0.2 sweep。

beta 的作用是 regularizer / branch consistency signal，不是强制中间瓶颈。

---

## 9. 现在不建议做什么

### 不建议 1：不建议先做 MoE / hard branch classifier

理由：当前 family classifier accuracy 只有 0.19–0.45，hard gate 误分支会非常严重。

只有当新的 canonical branch policy 形成可分类的粗 branch，且 classifier accuracy 至少达到 0.80，才值得做 MoE。

### 不建议 2：不建议继续简单扩到 100k

mixed 100k 相比 20k 的 T MAE 改善很小：iid 只 -0.8%，radius -4.0%，beta_block 反而 +0.2%，angular_sector -2.4%；文档结论也明确说简单扩大数据量不是主要瓶颈。

### 不建议 3：不建议继续只调 anchor k/w

anchor v2 已经证明能降 T，但 theta branch 没解决。继续调 k/w 大概率只能小幅降张力，无法从根上降低 (xyz\rightarrow\theta) 的多值性。

---

## 10. 20k 晋级 100k 的验收标准

我建议把门槛分成四层。

### 10.1 hard gate

必须全部通过：

| 指标                             |                   门槛 |
| ------------------------------ | -------------------: |
| rows                           | ≥ 18000，有效目标最好 20000 |
| rms_rnorm q95                  |               ≤ 0.06 |
| max tension                    |             ≤ 2000 N |
| saturation ratio               |                    0 |
| tension < 0 ratio              |                    0 |
| tension > 2000 ratio           |                    0 |
| segmented / integrated success |        1.0 或 ≥ 0.995 |

### 10.2 single-branch continuity

20k silver：

| 指标                                      |          门槛 |
| --------------------------------------- | ----------: |
| all-workspace 10mm theta p95            |      ≤ 3.0° |
| all-workspace 10mm T p95                |     ≤ 100 N |
| beta-close / same-branch 10mm theta p95 |      ≤ 1.5° |
| beta-close / same-branch 10mm T p95     |      ≤ 50 N |
| 10mm beta-far pair ratio                | 比当前下降 ≥ 60% |
| multi-branch ball ratio                 | 比当前下降 ≥ 70% |

20k gold：

| 指标                           |     门槛 |
| ---------------------------- | -----: |
| all-workspace 10mm theta p95 | ≤ 2.0° |
| all-workspace 10mm T p95     | ≤ 70 N |
| same-branch 10mm T p95       | ≤ 35 N |

### 10.3 oracle floor

必须新增一个 readiness check：

| oracle                           |                20k 晋级门槛 |
| -------------------------------- | ----------------------: |
| selected-branch xyz-NN T MAE     |                  ≤ 35 N |
| selected-branch xyz-NN theta MAE |                  ≤ 2.0° |
| selected-branch xyz-NN EE p95    |                 ≤ 40 mm |
| beta-NN T MAE                    | ≤ 25 N，同时 EE p95 不能离谱恶化 |

如果 oracle 自己都到不了 35–40 N，MLP 不可能稳定到 35–40 N。

### 10.4 model gate

与当前 anchor v2 direct baseline 的四 split 平均 T MAE 50.94 N 对比，20k 必须至少达到：

| split          | T MAE 目标 | theta MAE 目标 | EE p95 目标 |
| -------------- | -------: | -----------: | --------: |
| iid            |   ≤ 35 N |       ≤ 2.2° |   ≤ 50 mm |
| radius         |   ≤ 38 N |       ≤ 2.0° |   ≤ 60 mm |
| beta_block     |   ≤ 48 N |       ≤ 3.2° |   ≤ 80 mm |
| angular_sector |   ≤ 42 N |       ≤ 2.6° |   ≤ 90 mm |

晋级 100k 的硬条件：

[
\text{四 split 平均 T MAE} \le 40N
]

更理想条件：

[
\text{四 split 平均 T MAE} \le 38N
]

并且 EE p95 不得比当前 anchor v2 明显恶化。当前 anchor v2 的 EE p95 为 iid 49.75 mm、radius 55.64 mm、beta_block 77.37 mm、angular_sector 89.38 mm，可作为对照。

---

## 11. 可以直接交给 Codex 的任务清单

### Codex Task 0：修正 beta schema / sign

实现或检查：

```text
scripts/analysis/derive_effective_beta.py
```

要求：

```text
input: dataset.parquet
output columns:
effective_beta_1_rad ... effective_beta_6_rad
effective_beta_source = theta_odd_even_mean
```

单元测试：

```text
beta_to_theta(effective_beta_from_theta(theta)) 与原 theta 的 MAE < 1e-8 rad
```

禁止再把 `dataset_meta.parquet` 的原始 beta 直接作为几何 beta。

---

### Codex Task 1：graph branch policy dry run

实现：

```text
scripts/analysis/select_canonical_branch_graph.py
```

CLI 示例：

```bash
python scripts/analysis/select_canonical_branch_graph.py \
  --dataset data/mixed_beta_20k_distal_preferred_anchor_v2_second_stage_from_anchor_v2_k32_w20/dataset.parquet \
  --meta data/mixed_beta_20k_distal_preferred_anchor_v2_second_stage_from_anchor_v2_k32_w20/dataset_meta.parquet \
  --beta-source effective_theta \
  --voxel-mm 10 \
  --ball-mm 15 \
  --beta-dbscan-eps 0.15 \
  --policy distal_v2_graph \
  --out data/mixed_beta_20k_anchor_v2_graph_branch_filtered_s2
```

输出报告必须包含：

```text
hard gate
retention ratio
workspace coverage shrinkage
all-workspace 10mm theta/T continuity
beta-close and beta-far continuity
multi-branch ball ratio
oracle floor
```

验收：如果 all-workspace theta p95 仍 > 5°，说明 filtering 不够，直接进入 active regeneration。

---

### Codex Task 2：主动生成 single-branch 20k

实现：

```text
scripts/generate_single_branch_dataset.py
configs/robot_rods_only_mixed_20k_single_branch_distal_v2.yaml
```

要求：

```text
每个 target xyz 生成多个 candidate beta/theta
每个 candidate 解 tension
用 graph policy 选一个 canonical branch
只输出 selected branch
```

输出：

```text
data/mixed_beta_20k_single_branch_distal_v2/dataset.parquet
data/mixed_beta_20k_single_branch_distal_v2/dataset_meta.parquet
data/mixed_beta_20k_single_branch_distal_v2/branch_policy_report.md
```

---

### Codex Task 3：selected-branch integrated allocator

实现：

```text
scripts/analysis/relabel_selected_branch_integrated_anchor.py
```

或扩展现有：

```text
scripts/analysis/relabel_anchor_canonical.py
```

新增参数：

```text
--anchor-scope selected_branch_graph
--distance-space effective_beta
--no-cross-branch-anchor
--w-anchor-sweep 5,10,20,40
```

目标不是单纯降低 all-workspace T，而是在 single-branch 数据上降低 same-branch / all-workspace T，同时不破坏 theta continuity 和 hard gate。

---

### Codex Task 4：baseline 只做必要模型

先跑：

```text
direct MLP / MLP_large / RF / LGBM / KNN
```

再跑一个轻量 beta auxiliary MLP：

```text
xyz/poly -> shared trunk -> theta head + tension head + beta_aux head
```

不要先做 MoE。

输出：

```text
runs/baselines_single_branch_distal_v2_fast4/
runs/baselines_single_branch_distal_v2_beta_aux/
```

每个 split 都要和 anchor v2 direct baseline 对比。

---

## 12. 论文原理层面的修正

原论文的思路是用 virtual prototype 生成数据，再训练 MLP 从 3D Cartesian 坐标预测 30 个 joint angles 和 12 个 cable tensions；输出层 42 维。 这个表述本身缺了一个关键前提：对于冗余/多段连续体机器人，(xyz\rightarrow\theta/T) 并不天然单值。

因此论文中 model inverse solution 部分应该补一句核心原则：

[
\text{The inverse model is trained on a deterministic canonical branch of the inverse solution manifold.}
]

也就是说，Eq.(52)+(53) 或 PSO 得到的是 feasible inverse solution，不自动保证 supervised label 的唯一性。你现在要补的是：

1. 给定 (xyz) 时，先定义 canonical branch selection policy。
2. 在该 branch 内求解 canonical tension。
3. MLP 学的是这个 canonical inverse map，而不是所有 feasible inverse solutions 的混合。

---

## 最终建议

让 Codex 立即进入 Phase 4：

```text
Phase 4A: graph branch policy filtering dry run
Phase 4B: active single-branch distal_v2 20k generation
Phase 4C: selected-branch integrated tension relabel
Phase 4D: direct + beta_aux baseline
```

只有当 20k single-branch 数据达到：

```text
all-workspace 10mm theta p95 <= 3deg
all-workspace 10mm T p95 <= 100N
same-branch 10mm T p95 <= 50N
四 split 平均 T MAE <= 40N
EE p95 不比 anchor v2 明显恶化
```

才扩展到 100k。当前 mixed 多 branch 数据不要丢，但它应该作为复杂度压力测试，而不是下一阶段主训练数据。

---

## 13. Phase 4 实际执行结果（2026-06-08）

### 13.1 已实现内容

本轮已经按上面的 Phase 4 计划实现了以下工具和测试：

```text
scripts/analysis/derive_effective_beta.py
scripts/analysis/select_canonical_branch_graph.py
scripts/generate_single_branch_dataset.py
scripts/analysis/relabel_anchor_canonical.py
scripts/baselines/run_beta_aux_baselines.py
scripts/pipelines/run_phase4_single_branch_experiment.py
configs/robot_rods_only_mixed_20k_single_branch_distal_v2.yaml
```

关键实现点：

1. `effective_beta_1_rad ... effective_beta_6_rad` 从 30 维 `theta` 的 odd/even section mean 反推，不再直接信任 meta 里的原始 beta。
2. `select_canonical_branch_graph.py` 使用 workspace voxel 内 beta DBSCAN + voxel graph ICM，按 distal-preferred policy 选择每个 voxel 的 canonical branch。
3. `graph_component_id` 已修正为 selected workspace graph 的真实连通分量，不再是 per-voxel id。这个修正很重要，因为旧的 per-voxel id 会让 branch-aware oracle 和 no-cross-branch anchor 诊断过于乐观。
4. `generate_single_branch_dataset.py` 当前实现是 candidate-pool single-branch 版本：从已有 100k segmented canonical + 20k anchor v2 pool 中做图一致性选择，再 workspace-balanced 抽样 20k。
5. `relabel_anchor_canonical.py` 已支持 `selected_branch_graph`、`effective_beta` distance space、`no-cross-branch-anchor`。
6. Phase 4 runner 已加 gate：如果 10mm all-workspace theta continuity 失败，则停止在 relabel/model training 之前。原因是 relabel 只改张力，不会修复 theta branch 混叠。

相关测试：

```text
tests/test_beta_effective_schema.py
tests/test_select_canonical_branch_graph.py
tests/test_single_branch_candidate_pool_generation.py
tests/test_selected_branch_anchor_scope.py
tests/test_beta_aux_baseline.py
tests/test_phase4_runner_gates.py
```

### 13.2 Phase 4A：graph branch policy dry run

输入：

```text
data/mixed_beta_20k_distal_preferred_anchor_v2_second_stage_from_anchor_v2_k32_w20
```

输出：

```text
data/mixed_beta_20k_anchor_v2_graph_branch_filtered_s2
```

结果：

```text
rows_in: 20000
rows_out: 17818
retention_ratio: 0.8909
voxels: 17808
graph_edges: 29229
graph_components: 4014
largest_graph_component_voxels: 10469
multi_branch_voxel_ratio: 0.1083
ICM iterations: 4
```

诊断结论：

```text
all-workspace 10mm theta p95: 8.16 deg
all-workspace 10mm T p95: 131.63 N
beta-close 10mm theta p95: 1.20 deg
beta-close 10mm T p95: 40.14 N
```

解释：

graph filtering 可以让 beta-close 情况变得较干净，但 all-workspace theta p95 仍然远大于 5 deg。这说明只在已有 20k anchor v2 数据上过滤，不足以得到 deterministic `xyz -> theta/T` label manifold。

### 13.3 Phase 4B：candidate-pool single-branch 20k

输入候选池：

```text
data/mixed_beta_100k_distal_preferred_segmented_canonical
data/mixed_beta_20k_distal_preferred_anchor_v2_second_stage_from_anchor_v2_k32_w20
```

输出：

```text
data/mixed_beta_20k_single_branch_distal_v2
```

候选池选择结果：

```text
graph-selected rows in pool: 76395
output rows: 20000
retention_ratio vs selected pool: 0.2618
workspace balance: radius_bins=5, z_bins=5, angle_bins=8
```

物理 hard gate：

```text
passed: true
rms_rnorm_q95: 0.03884
rms_rnorm_max: 0.05046
max_tension_n: 1448.46
max_tension_q95_n: 1055.18
saturation_ratio: 0.0
tension_lt0_ratio: 0.0
tension_gt_tmax_ratio: 0.0
```

连续性 gate：

| metric | result | target |
| --- | ---: | ---: |
| all-workspace 10mm theta p95 | 7.137 deg | <= 3 deg |
| all-workspace 10mm T p95 | 194.19 N | <= 100 N |
| beta-close 10mm theta p95 | 0.979 deg | diagnostic |
| beta-close 10mm T p95 | 105.60 N | <= 50 N |
| same-component 10mm theta p95 | 6.734 deg | diagnostic |
| same-component 10mm T p95 | 175.28 N | diagnostic |
| cross-component 10mm theta p95 | 7.656 deg | diagnostic |
| cross-component 10mm T p95 | 214.96 N | diagnostic |

branch clustering：

```text
balls_evaluated: 1093
multi_branch_ball_ratio: 1.0
branch_count_p50: 5
branch_count_p90: 6
branch_purity_p50: 0.3333
within_branch_tension_mae_n_p95: 101.76
between_branch_tension_mae_n_p95: 156.06
```

oracle floor：

| oracle | theta MAE | theta p95 | T MAE | T p95 | EE p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| xyz-NN | 1.449 deg | 4.444 deg | 69.07 N | 185.25 N | 16.05 mm |
| branch-aware xyz-NN | 1.422 deg | 4.426 deg | 76.33 N | 184.61 N | 14.99 mm |
| beta-NN | 0.316 deg | 0.781 deg | 38.42 N | 122.04 N | 110.16 mm |

### 13.4 当前结论

Phase 4B 生成的 20k 数据在物理上可行，但仍不是足够单值的 `xyz -> theta/T` 监督标签流形。

最关键证据：

1. all-workspace 10mm theta p95 仍为 7.14 deg，超过停止阈值 5 deg，也远高于目标 3 deg。
2. all-workspace 10mm T p95 为 194 N，超过目标 100 N。
3. beta-close theta 很好（0.98 deg），但 beta-close T p95 仍为 105.6 N。这说明即便几何 branch 接近，张力 canonical allocator 仍有一致性问题。
4. `multi_branch_ball_ratio = 1.0`，说明当前 candidate-pool filtering 并没有真正消除局部 workspace 多 branch 混合。
5. `beta-NN` 张力 MAE 为 38.42 N，说明在 beta 空间 label 相对可学；但 `xyz-NN` 张力 MAE 为 69.07 N，说明瓶颈仍然是 `xyz -> beta/theta` 多解性。

因此本轮没有继续执行：

```text
Phase 4C selected-branch tension relabel
Phase 4D direct / beta_aux model training
```

停止原因：

```text
theta continuity gate failed before relabel/model training.
```

工程判断：

张力 relabel 只能改善 T label，不会改变 theta label；当 theta branch mixing 仍然存在时，继续训练 MLP 只会重新得到平均化/混叠的结果，不能解决拟合性能瓶颈。

### 13.5 需要 GPT5Pro 判断的下一步

目前 evidence 指向：candidate-pool filtering 不够。下一步应优先研究真正的 active single-branch generation，而不是继续加模型复杂度。

建议让 GPT5Pro 判断以下方向：

1. 是否应从 `xyz` target 出发，为每个 target 主动生成多个 beta/theta candidates，再用 global branch policy 选一个，而不是从已有 mixed pool 中事后过滤。
2. 是否应把 distal-preferred policy 变成连续 optimization objective，例如固定 `xyz` 约束下最大化第三段 beta、惩罚第一/二段 beta，同时加局部 Jacobian/trajectory continuity 正则。
3. 是否应以 beta manifold 作为主采样空间，然后只保留满足 workspace coverage 且局部 injective 的子流形，而不是强行覆盖所有 reachable xyz。
4. 张力 allocator 是否需要在 beta-close 条件下继续做更强的 graph smoothing / anchor relabel，但这应放在 theta branch 被控制之后。
5. 如果必须保留多 branch，模型路线应改成 `xyz -> beta/family -> theta/T` 或 MoE，而不是单值 direct MLP。

本轮报告文件：

```text
runs/diagnostics/phase4_single_branch_distal_v2_summary_v2.md
runs/diagnostics/phase4_single_branch_distal_v2_summary_v2.json
runs/diagnostics/phase4_single_branch_distal_v2_continuity_v2.json
runs/diagnostics/phase4_single_branch_distal_v2_branch_clustering_v2.json
runs/diagnostics/phase4_single_branch_distal_v2_oracle_v2.json
runs/diagnostics/phase4_single_branch_distal_v2_quality_gates_v2.json
```

### 13.6 Phase 4E：active xyz single-branch 2k pilot（2026-06-08）

根据 13.5 的判断，本轮继续实现了一个真正主动生成版本，而不是继续在已有 mixed pool 上事后 filtering。

新增工具：

```text
scripts/generate_active_single_branch_dataset.py
configs/robot_rods_only_active_xyz_2k_single_branch_distal_v3.yaml
tests/test_active_xyz_target_sampler.py
tests/test_active_xyz_candidate_generation.py
tests/test_active_xyz_global_branch_selector.py
tests/test_active_xyz_runner_gates.py
```

实现逻辑：

1. 先从受限 beta 空间生成 reachable target pool，并用 forward kinematics 得到 target xyz。
2. 对每个 target xyz 主动生成多个 beta/theta candidates：
   - candidate 0 使用 source beta，保证 target 来自可达流形；
   - 其余 candidate 使用 inverse PSO 从同一 target xyz 反解；
   - 每个 candidate 再用 segmented/canonical tension labeler 标注张力。
3. 保留满足 `xyz_err <= 6mm`、`rms_rnorm <= 0.06`、张力有限的 candidates。
4. 在 target kNN graph 上做 global active branch selection，代价同时考虑 theta、effective beta 和 tension 的局部连续性。
5. dataset 中 `x_m/y_m/z_m` 保存 desired target xyz；meta 中额外保存 `achieved_x/y/z` 和 `xyz_err_m`，避免把反解误差混入监督输入。

本轮 2k pilot 配置：

```text
num_targets: 2000
candidates_per_target: 3
target_pool_size: 50000
target_oversample: 2
beta limit: beta1-4 <= 5deg, beta5-6 <= 10deg
graph_k_neighbors: 16
graph_radius_m: 0.02
policy_version: active_xyz_distal_v3_graph
```

运行结果：

| metric | value |
| --- | ---: |
| targets tried | 2035 |
| targets accepted | 2000 |
| candidate rows | 5572 |
| selected rows | 2000 |
| mean candidates / selected target | 2.786 |
| elapsed | 6319.71 s |
| sec / selected target | 3.160 s |
| target graph edges | 2620 |
| target graph components | 1065 |
| ICM iterations | 4 |

hard gate：

| metric | value | gate |
| --- | ---: | --- |
| xyz err p95 | 3.964 mm | pass |
| xyz err max | 5.967 mm | pass |
| rms_rnorm q95 | 0.0386 | pass |
| hard gate passed | true | pass |

continuity gate：

| metric | value | target |
| --- | ---: | ---: |
| all-workspace 10mm theta p95 | 8.917 deg | <= 3 deg |
| all-workspace 10mm T p95 | 237.01 N | <= 100 N |
| beta-close 10mm theta p95 | 1.044 deg | diagnostic |
| beta-close 10mm T p95 | 119.43 N | <= 50 N |
| multi-branch ball ratio | 1.0 | near 0 |
| xyz-NN T MAE | 114.21 N | diagnostic |
| beta-NN T MAE | 64.19 N | diagnostic |
| acceptance gate passed | false | fail |
| stop before relabel | true | stop |

关键分组诊断：

| group | pairs | theta p50 | theta p95 | T p50 | T p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| all xyz <= 5mm | 43 | 4.512 deg | 9.149 deg | 106.68 N | 231.69 N |
| xyz <= 5mm beta-close | 4 | 0.594 deg | 0.910 deg | 34.74 N | 70.90 N |
| xyz <= 5mm beta-far | 39 | 4.672 deg | 9.252 deg | 113.64 N | 234.09 N |
| all xyz <= 10mm | 379 | 4.819 deg | 8.917 deg | 118.72 N | 237.01 N |
| xyz <= 10mm beta-close | 39 | 0.759 deg | 1.044 deg | 14.54 N | 119.43 N |
| xyz <= 10mm beta-far | 340 | 5.371 deg | 8.937 deg | 128.86 N | 241.68 N |
| same beta kNN k8 | 10926 | 1.293 deg | 2.487 deg | 74.36 N | 186.03 N |

oracle floor：

| oracle | theta MAE | theta p95 | T MAE | T p95 | EE p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| xyz-NN | 2.855 deg | 5.952 deg | 114.21 N | 230.59 N | 47.24 mm |
| branch-aware xyz-NN | 1.711 deg | 5.603 deg | 67.17 N | 213.75 N | 18.60 mm |
| beta-NN | 0.787 deg | 1.444 deg | 64.19 N | 170.49 N | 221.88 mm |

结论：

1. active xyz generation 成功解决了可达性和基本物理质量问题：target 可达、反解误差小、rms 约束过线。
2. 但它没有真正生成 single-branch workspace label manifold。`multi_branch_ball_ratio = 1.0`，且 10mm 近邻里 beta-far pairs 占 340/379，仍然主导 theta/T 不连续。
3. beta-close 近邻的 theta 很稳定，10mm theta p95 为 1.044 deg；这说明 forward/inverse kinematics 本身不是主要问题，问题是同一 workspace 邻域内仍混入多个 beta branch。
4. 本轮按 gate 停止在 relabel 之前。原因是 selected-branch tension relabel 只能改张力，不能修复 theta branch 混叠；在 theta p95 仍为 8.917 deg 时继续 relabel 或训练 direct MLP 没有决策价值。
5. 速度方面，2k、3 candidates 的实测速度为 3.16 s/selected target。直接扩展到 20k 约 17.6 小时，100k 约 87.8 小时；如果使用 8 candidates，会进一步变慢。因此下一步需要先改生成策略，而不是直接扩大规模。

当前判断：

active xyz target + candidate selection 仍然不够，因为它的 target pool 本身来自 mixed reachable beta 空间。只要允许所有 reachable xyz 进入目标集合，workspace 上天然会出现多个 beta branch。global selection 只能在候选之间选局部较平滑的分支，不能保证目标集合本身位于一个局部 injective 的 beta manifold。

下一步更合理的实验方向：

1. 从 `active xyz target` 转向 `active beta manifold`：先在 effective beta 空间构造连续、distal-preferred、局部 injective 的 6D/低维子流形，再映射到 workspace。
2. 或者在 target selection 阶段加入 Jacobian / local inverse uniqueness gate：拒绝那些 workspace 近邻中存在多个 beta-far 解的 target。
3. 目标不应是覆盖所有 reachable xyz，而应是先构造一个对 `xyz -> theta/T` 近似单值的训练子域；模型拟合稳定后，再逐步扩展 branch family 或使用 branch-aware/MoE。

本轮报告文件：

```text
data/active_xyz_2k_single_branch_distal_v3_c3/active_generation_report.json
data/active_xyz_2k_single_branch_distal_v3_c3/diagnostics/active_diagnostics.json
data/active_xyz_2k_single_branch_distal_v3_c3/dataset.parquet
data/active_xyz_2k_single_branch_distal_v3_c3/dataset_meta.parquet
data/active_xyz_2k_single_branch_distal_v3_c3/candidates.parquet
data/active_xyz_2k_single_branch_distal_v3_c3/candidates_meta.parquet
```

### 13.7 Phase 4F：active beta-manifold 2k pilots（2026-06-09）

基于 Phase 4E 的失败结论，本轮不再从 mixed workspace target 出发，而是实现并测试了新的主动生成器：

```text
scripts/generate_active_beta_manifold_dataset.py
scripts/pipelines/run_active_beta_manifold_overnight.py
configs/robot_rods_only_active_beta_manifold_distal_v1.yaml
```

核心策略：

1. 先在 effective beta 空间构造连续、distal-preferred 的候选 manifold；
2. 再 forward map 到 workspace，避免从 mixed reachable xyz 里抽目标；
3. 对候选做 workspace local injectivity check；
4. 先跑 2k pilot，只有通过严格 gate 才生成 20k；
5. 20k 通过后才跑 MLP / TF MLP baseline。

本轮 pipeline 命令：

```text
/mnt/ML_projects/conda_envs/dante_env/bin/python scripts/pipelines/run_active_beta_manifold_overnight.py \
  --config configs/robot_rods_only_active_beta_manifold_distal_v1.yaml \
  --num-samples-2k 2000 \
  --pool-size-2k 20000 \
  --num-samples-20k 20000 \
  --pool-size-20k 200000 \
  --out-root data/active_beta_manifold_distal_v1 \
  --runs-root runs/active_beta_manifold_distal_v1
```

#### 13.7.1 本轮验收标准

2k pilot 必须同时满足：

| metric | gate |
| --- | ---: |
| hard gate passed | true |
| all-workspace 10mm pairs | >= 100 |
| all-workspace 10mm theta p95 | <= 3 deg |
| all-workspace 10mm T p95 | <= 100 N |
| beta-close 10mm T p95 | <= 50 N |
| multi-branch ball ratio | <= 0.25 |

注意：这里的 gate 是为了判断是否值得扩到 20k 并训练模型。它比“物理可行”更严格，主要检查监督标签是否足够单值、连续、可学习。

#### 13.7.2 2k pilot 结果

最终状态：

```text
runs/active_beta_manifold_distal_v1/summary.json
status = stopped_no_2k_variant_passed
best_variant = None
twenty_k = None
baselines = None
```

也就是说：三个 2k variant 都没有通过严格 gate，因此 pipeline 按计划停止，没有生成 20k，也没有训练 MLP / TF MLP。

| variant | strict gate | sec/row | keep ratio | all10 pairs | all10 theta p95 | all10 T p95 | same-beta theta p95 | same-beta T p95 | beta-close T p95 | multi-branch ratio | xyz-NN T MAE | beta-NN T MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| manifold_3d_full_angle | false | 0.169 | 1.000 | 819 | 0.727 deg | 113.0 N | 0.714 deg | 117.9 N | 113.0 N | 0.049 | 24.7 N | 18.1 N |
| filtered_6d_distal_cloud | false | 0.171 | 0.220 | 107 | 1.092 deg | 118.7 N | 1.595 deg | 141.7 N | 75.4 N | 0.000 | 65.4 N | 36.7 N |
| manifold_3d_quadrant | false | 0.171 | 1.000 | 3188 | 0.750 deg | 109.2 N | 0.460 deg | 98.5 N | 109.2 N | 0.123 | 22.5 N | 12.9 N |

#### 13.7.3 主要观察

1. active beta-manifold 明显改善了 theta/branch 问题。
   - Phase 4E active xyz 的 all-workspace 10mm theta p95 是 `8.917 deg`；
   - 本轮三个 active beta-manifold variant 的 all-workspace 10mm theta p95 降到 `0.727-1.092 deg`；
   - `manifold_3d_quadrant` 的 same-beta theta p95 进一步降到 `0.460 deg`。

2. branch 混叠已不再是当前最主要的失败点。
   - `filtered_6d_distal_cloud` 的 multi-branch ball ratio 为 `0.0`；
   - `manifold_3d_full_angle` 为 `0.049`；
   - `manifold_3d_quadrant` 为 `0.123`，仍低于 gate 的 `0.25`。

3. 当前严格 gate 失败主要由张力连续性导致。
   - all10 T p95 仍在 `109.2-118.7 N`；
   - beta-close T p95 仍在 `75.4-113.0 N`；
   - gate 要求分别是 `<=100 N` 和 `<=50 N`。

4. `manifold_3d_quadrant` 是当前最接近成功的方向。
   - 它的 all10 theta p95 为 `0.750 deg`；
   - same-beta theta p95 为 `0.460 deg`；
   - same-beta T p95 为 `98.5 N`；
   - beta-NN T MAE 为 `12.9 N`；
   - 但 beta-close T p95 仍为 `109.2 N`，没有过张力 gate。

5. 生成速度稳定。
   - 三个 2k pilot 的速度约 `0.169-0.171 s/row`；
   - 20k 估算约 `56-57 min`，100k 估算约 `4.7 h`，不再像 Phase 4E active xyz candidate selection 那样慢。

#### 13.7.4 当前结论

active beta-manifold 是一个明显更正确的 single-branch 生成方向：它已经把 `xyz -> theta` 的局部不连续压到 1 度以内，说明“从 beta manifold 主动生成 workspace 子域”比“从 mixed workspace target 事后选 branch”更有效。

但它还没有解决 `xyz -> tension` 的严格连续性问题。当前失败不是因为 forward kinematics 或 theta branch 仍然混乱，而是因为 segmented/canonical tension labeler 在局部邻域仍会产生约 `100 N` 量级的 p95 张力跳变。若直接放宽 gate 生成 20k 并训练模型，预计 theta 会明显好于之前，但张力 MAE 仍可能被标签噪声/allocator 分支选择限制。

因此，本轮没有扩到 20k 是合理的：继续扩大同一标注机制的数据量，未必能突破当前张力拟合上限。

#### 13.7.5 建议给下一轮决策的问题

下一步不应再优先处理 workspace branch，而应重点判断张力标签是否需要和 beta-manifold branch policy 联合 canonical：

1. 是否应该在 active beta-manifold 内做 tension relabel，而不是沿用当前 segmented/canonical tension labeler？
2. 是否需要让 tension allocator 引入邻域锚定项，例如同一 beta manifold 上的 kNN graph smoothing / second-stage canonical refinement？
3. 当前 beta-close T p95 的 `50 N` gate 是否过严？如果最终模型张力 MAE 目标是约 `30-50 N`，那么 beta-close T p95 应该设在什么水平才有训练意义？
4. 是否应该先用 `manifold_3d_quadrant` 放宽张力 gate 生成 20k，只训练 theta-only 或 theta+EE baseline，确认几何分支已经可学？
5. 是否应该把张力任务拆成二阶段：先训练 `xyz -> theta/beta`，再由 physics allocator 根据预测 theta 计算 canonical tension，而不是直接监督 `xyz -> T`？

本轮报告文件：

```text
runs/active_beta_manifold_distal_v1/summary.json
runs/active_beta_manifold_distal_v1/pilot_2k_summary.json
data/active_beta_manifold_distal_v1/pilot_2k/manifold_3d_full_angle/active_generation_report.json
data/active_beta_manifold_distal_v1/pilot_2k/manifold_3d_full_angle/diagnostics/active_diagnostics.json
data/active_beta_manifold_distal_v1/pilot_2k/filtered_6d_distal_cloud/active_generation_report.json
data/active_beta_manifold_distal_v1/pilot_2k/filtered_6d_distal_cloud/diagnostics/active_diagnostics.json
data/active_beta_manifold_distal_v1/pilot_2k/manifold_3d_quadrant/active_generation_report.json
data/active_beta_manifold_distal_v1/pilot_2k/manifold_3d_quadrant/diagnostics/active_diagnostics.json
```

### 13.8 Phase 4G：第三关节优先基础网格数据集（2026-06-10）

基于新的人工规则，本轮实现了“第三关节优先基础网格”路线：

```text
scripts/generate_priority_grid_dataset.py
configs/robot_rods_only_priority_grid_third_joint_first_v1.yaml
```

规则：

```text
beta5, beta6: -10 deg 到 10 deg，各 41 档
s2: [0, 0.25, 0.5]
s1: [0, 0.125, 0.25]
只保留 s1 <= s2

beta1 = s1 * beta5
beta2 = s1 * beta6
beta3 = s2 * beta5
beta4 = s2 * beta6
beta5 = beta5
beta6 = beta6
```

这等价于：第三关节完整参与，第二关节按较小比例参与，第一关节按更小比例参与。严格优先组合共 `7` 组，完整网格为 `41 * 41 * 7 = 11767` 行。

同时实现了基础库参照选解工具：

1. target xyz 必须在基础库 workspace 10mm 邻域内；
2. 多个候选 theta 同时可达时，先按第一关节 `beta1/beta2` 距离基础库最近排序；
3. 第一关节相同或接近时，再按第二关节 `beta3/beta4`；
4. 再按第三关节、xyz 误差、rms、max tension 排序。

#### 13.8.1 生成结果

完整生成命令：

```text
/mnt/ML_projects/conda_envs/dante_env/bin/python scripts/generate_priority_grid_dataset.py \
  --config configs/robot_rods_only_priority_grid_third_joint_first_v1.yaml \
  --out-dir data/priority_grid_third_joint_first_v1
```

结果：

| metric | value |
| --- | ---: |
| selected rows | 11767 |
| label failures | 0 |
| elapsed | 2003.18 s |
| sec / row | 0.170 s |
| hard gate | true |
| rms_rnorm q95 | 0.0379 |

workspace 覆盖：

| axis / metric | value |
| --- | ---: |
| x range | 684.96 - 1215.50 mm |
| y range | -609.28 - 609.28 mm |
| z range | -569.32 - 569.32 mm |
| radius range | 973.98 - 1215.50 mm |
| yz hull area | 1,233,167.91 mm^2 |

对比：

| dataset | yz hull area |
| --- | ---: |
| standard sweep 100k segmented canonical | 493,407.55 mm^2 |
| active beta-manifold quadrant 2k | 236,901.68 mm^2 |
| active beta-manifold full-angle 2k | 899,268.16 mm^2 |
| priority grid 11767 | 1,233,167.91 mm^2 |
| mixed 100k distal-preferred | 1,846,090.67 mm^2 |

结论：第三关节优先基础网格确实从“两条曲线”扩展成了一个更大的 workspace 面，覆盖面积约为 standard sweep 的 `2.5x`，也大于 active beta-manifold full-angle 2k。

#### 13.8.2 连续性结果

| group | pairs | theta p95 | T p95 |
| --- | ---: | ---: | ---: |
| all xyz <= 5mm | 487 | 0.938 deg | 102.66 N |
| xyz <= 5mm beta-close | 463 | 0.823 deg | 95.77 N |
| xyz <= 5mm beta-far | 24 | 1.164 deg | 150.81 N |
| all xyz <= 10mm | 4969 | 1.096 deg | 117.44 N |
| xyz <= 10mm beta-close | 4691 | 0.992 deg | 111.36 N |
| xyz <= 10mm beta-far | 278 | 1.470 deg | 145.89 N |
| same beta kNN k8 | 48144 | 0.331 deg | 97.15 N |

branch / oracle：

| metric | value |
| --- | ---: |
| multi_branch_ball_ratio | 0.4696 |
| branch_count_p50 / p90 | 1 / 2 |
| branch_purity_p50 | 1.0 |
| within_branch T p95 | 60.51 N |
| between_branch T p95 | 107.58 N |
| xyz-NN theta MAE | 0.472 deg |
| xyz-NN T MAE | 31.37 N |
| beta-NN theta MAE | 0.118 deg |
| beta-NN T MAE | 16.26 N |

#### 13.8.3 结论

1. 这个方案成功扩大了 workspace 覆盖，并且 theta 连续性明显达标。`all-workspace 10mm theta p95 = 1.096 deg`，满足 `<= 1-2 deg` 的目标。
2. 张力没有改善。`all-workspace 10mm T p95 = 117.44 N`，比 active beta-manifold quadrant 的 `109.17 N` 更差，因此不能直接作为 20k/100k 主训练数据。
3. full 7 层比例网格本身仍有局部多分支重叠。`multi_branch_ball_ratio = 0.4696`，虽然 branch_count p50 只有 1、p90 为 2，但仍未达到 `<= 0.25` 的目标。
4. `within_branch T p95 = 60.51 N`，说明如果先把同一 workspace 邻域内的比例层/分支选成单支，张力有机会接近可接受区间；但 all-workspace 直接训练仍会被比例层重叠和张力跳变影响。
5. 当前最合理的下一步不是直接训练模型，而是在这个基础网格上做一次“基础网格自身的 canonical 单支选择”：同一 workspace 邻域内只保留最贴近第三关节优先规则的一层比例，再重新评估 T p95。

本轮报告文件：

```text
data/priority_grid_third_joint_first_v1/priority_grid_generation_report.json
data/priority_grid_third_joint_first_v1/diagnostics/priority_grid_diagnostics.json
data/priority_grid_third_joint_first_v1/dataset.parquet
data/priority_grid_third_joint_first_v1/dataset_meta.parquet
data/priority_grid_third_joint_first_v1/priority_grid_pool.parquet
```

### 13.9 Phase 4H：基础网格内部 canonical 单支选择（2026-06-10）

按上一节结论，本轮在 `11767` 行第三关节优先基础网格内部做了一次 canonical 单支选择：

```text
scripts/analysis/select_priority_grid_single_branch.py
data/priority_grid_third_joint_first_v1_single_branch/
```

策略：

1. 以 workspace `10mm` voxel 作为局部邻域；
2. 同一个 voxel 内按 `(s1, s2)` 分层；
3. 每个 voxel 只保留一层 `(s1, s2)`；
4. 分层排序优先级为：更小 `s1`，更小 `s2`，更小 `rms_rnorm`，更小 `max_tension`，更大层内样本数；
5. 该策略对应“同一末端位置附近，如果多种比例层都能达到，则优先使用第一关节更小、第二关节更小的解”，也就是尽量把弯曲留给第三关节。

#### 13.9.1 筛选结果

| metric | before | after |
| --- | ---: | ---: |
| rows | 11767 | 11163 |
| retention ratio | 100.00% | 94.87% |
| workspace voxels | - | 10969 |
| multi-layer voxels | - | 509 |
| hard gate | true | true |

筛选后每层保留数量：

| s1 | s2 | rows |
| ---: | ---: | ---: |
| 0.000 | 0.000 | 1681 |
| 0.000 | 0.250 | 1586 |
| 0.000 | 0.500 | 1576 |
| 0.125 | 0.250 | 1604 |
| 0.125 | 0.500 | 1564 |
| 0.250 | 0.250 | 1616 |
| 0.250 | 0.500 | 1536 |

核心指标对比：

| metric | before full 7-layer grid | after voxel single-branch |
| --- | ---: | ---: |
| all xyz <= 5mm theta p95 | 0.938 deg | 1.104 deg |
| all xyz <= 5mm T p95 | 102.66 N | 125.15 N |
| all xyz <= 10mm theta p95 | 1.096 deg | 1.104 deg |
| all xyz <= 10mm T p95 | 117.44 N | 115.28 N |
| xyz <= 10mm beta-close T p95 | 111.36 N | 110.45 N |
| xyz <= 10mm beta-far T p95 | 145.89 N | 145.10 N |
| same beta kNN k8 theta p95 | 0.331 deg | 0.433 deg |
| same beta kNN k8 T p95 | 97.15 N | 98.05 N |
| multi_branch_ball_ratio | 0.4696 | 0.5882 |
| within_branch T p95 | 60.51 N | 82.07 N |
| between_branch T p95 | 107.58 N | 128.91 N |
| xyz-NN T MAE | 31.37 N | 32.00 N |
| beta-NN T MAE | 16.26 N | 16.62 N |

#### 13.9.2 结论

这次“基础网格内部 10mm voxel 单支选择”没有达到预期。

主要原因不是排序规则错了，而是这个基础网格的多解重叠并不主要发生在“同一个 10mm voxel 里有多层 `(s1,s2)`”这种形式。实际只有 `509 / 10969` 个 voxel 存在多层候选，因此筛选只删除 `604` 行，数据主体几乎没变。

更关键的是，当前 branch 诊断使用的是 `10mm ball`，而不是互不重叠的 voxel。即使每个 voxel 内只保留一层，半径 10mm 的球邻域仍会跨越多个相邻 voxel；相邻 voxel 之间仍可能选择不同 `(s1,s2)` 层。结果就是：

1. `all xyz <= 10mm T p95` 只从 `117.44 N` 小幅降到 `115.28 N`，改善很小；
2. `multi_branch_ball_ratio` 从 `0.4696` 反而升到 `0.5882`；
3. `all xyz <= 5mm T p95` 从 `102.66 N` 变差到 `125.15 N`；
4. `within_branch T p95` 从 `60.51 N` 变差到 `82.07 N`。

因此，不能把“voxel 内单支”当成最终 canonical 策略。它只能去掉非常局部、同 voxel 的重复层，但不能保证整个 workspace 局部邻域连续一致。

#### 13.9.3 下一步判断

如果继续沿这个方向，需要从“每个 voxel 独立选层”升级为“全局连续的层场选择”：

1. 在 workspace kNN graph 上选择 `(s1,s2)` 层，使相邻节点尽量保持同层或平滑切换；
2. 选择目标不只看单点第三关节优先，还要加入邻域一致性代价；
3. 允许删除更多边界样本，形成真正局部单值的 workspace patch，而不是保留 95% 样本；
4. 或者更直接地，把 `(s1,s2)` 固定成少数连续子流形分别训练，不再试图把 7 层比例网格压成一个全局单值模型。

当前结果说明：基础网格路线在几何覆盖上是成功的，但张力/分支一致性仍没有解决。下一轮如果目标是提升模型拟合性能，优先级应从“局部 voxel 去重”转向“全局连续单分支生成或连续层场选择”。

本轮报告文件：

```text
data/priority_grid_third_joint_first_v1_single_branch/priority_grid_single_branch_report.json
data/priority_grid_third_joint_first_v1_single_branch/diagnostics/priority_grid_single_branch_diagnostics.json
data/priority_grid_third_joint_first_v1_single_branch/priority_grid_single_branch_voxel_decisions.csv
data/priority_grid_third_joint_first_v1_single_branch/dataset.parquet
data/priority_grid_third_joint_first_v1_single_branch/dataset_meta.parquet
```

### 13.10 Phase 4I：全局连续层场选择与固定 `(s1,s2)` 子流形专家（2026-06-13）

上一轮 `10mm voxel` 内单支选择失败后，本轮同时尝试两条路线：

1. **全局连续层场选择**：在 workspace 邻接图上选择 `(s1,s2)` 层，使相邻 voxel 尽量同层或平滑换层。
2. **固定层子流形分开训练**：把 7 个固定 `(s1,s2)` 层拆开，分别作为连续子流形训练，不再硬压成一个全局单值 `xyz -> theta/T` 模型。

新增脚本：

```text
scripts/analysis/select_priority_grid_layer_field.py
scripts/pipelines/run_priority_grid_layer_experiments.py
configs/priority_grid_mlp_quiet.yaml
```

输出目录：

```text
data/priority_grid_third_joint_first_v1_layer_field_v1/
data/priority_grid_third_joint_first_v1_layers/
runs/diagnostics/priority_grid_layer_experiments_v1/
runs/baselines_priority_grid_original/
runs/baselines_priority_grid_layer_field_v1/
runs/baselines_layer_s1_*/
```

#### 13.10.1 全局连续层场选择策略

层场选择以 `10mm` workspace voxel 为节点。每个节点的候选标签不再是 DBSCAN 局部分支，而是 priority grid 中实际存在的固定 `(s1,s2)` 层。

默认图参数：

| parameter | value |
| --- | ---: |
| voxel size | 10 mm |
| graph radius | 20 mm |
| graph k | 16 |
| max ICM iterations | 30 |
| fixed-layer restarts | true |

能量项：

1. unary：小 `s1`、小 `s2`、低 `rms_rnorm`、低 `max_tension`、较大层内样本数；
2. pairwise：相邻 voxel 尽量同层；
3. 若换层，则惩罚 `s1/s2` 跳变、theta 均值跳变、beta 均值跳变、张力均值跳变。

这相当于把“第三关节优先”从逐 voxel 排序，升级为 workspace 图上的连续层场。

#### 13.10.2 层场选择结果

| metric | value |
| --- | ---: |
| rows in | 11767 |
| rows out | 11163 |
| retention ratio | 94.87% |
| hard gate rows | 11767 |
| rejected rows | 0 |
| voxels | 10969 |
| graph edges | 25006 |
| graph components | 2005 |
| largest component voxels | 6723 |
| total layer candidates | 11573 |
| multi-layer voxel ratio | 0.0464 |
| ICM iterations | 2 |
| best restart | s1_0125_s2_0500 |

保留层分布：

| layer | rows |
| --- | ---: |
| s1_0000_s2_0000 | 1681 |
| s1_0000_s2_0250 | 1586 |
| s1_0000_s2_0500 | 1558 |
| s1_0125_s2_0250 | 1584 |
| s1_0125_s2_0500 | 1592 |
| s1_0250_s2_0250 | 1610 |
| s1_0250_s2_0500 | 1552 |

核心连续性指标：

| metric | original grid | voxel single-branch | layer-field v1 |
| --- | ---: | ---: | ---: |
| rows | 11767 | 11163 | 11163 |
| all10 theta p95 | 1.096 deg | 1.104 deg | 1.104 deg |
| all10 T p95 | 117.44 N | 115.28 N | 115.46 N |
| beta-close T p95 | 111.36 N | 110.45 N | 110.50 N |
| multi_branch_ball_ratio | 0.4696 | 0.5882 | 0.5882 |
| within_branch T p95 | 60.51 N | 82.07 N | 82.07 N |
| between_branch T p95 | 107.58 N | 128.91 N | 128.91 N |
| xyz-NN T MAE | 31.37 N | 32.00 N | 31.94 N |
| branch-aware xyz-NN T MAE | 31.37 N | 31.99 N | 15.92 N |
| beta-NN T MAE | 16.26 N | 16.62 N | 16.74 N |

结论：这个全局层场版本没有通过预设验收。`all10 T p95` 只从 `117.44 N` 降到 `115.46 N`，`multi_branch_ball_ratio` 仍为 `0.5882`。主要原因仍是候选重叠太稀疏：只有 `4.64%` voxel 有多层候选，因此图优化能改动的空间很有限，结果和上一轮 voxel 单支非常接近。

但有一个重要信号：`branch-aware xyz-NN T MAE` 从约 `31.37 N` 降到 `15.92 N`。这说明如果模型知道当前点属于哪一层，张力局部可预测性会明显提升；问题不在单层内部完全不可学，而在多层合并后 `xyz` 信息不足。

#### 13.10.3 固定 `(s1,s2)` 子流形诊断

7 个固定层各保留 `1681` 行。每层 `multi_branch_ball_ratio = 0`。oracle floor 明显低于全局混合数据：

| layer | xyz-NN T MAE | beta-NN T MAE |
| --- | ---: | ---: |
| s1_0000_s2_0000 | 11.74 N | 10.40 N |
| s1_0000_s2_0250 | 12.88 N | 13.20 N |
| s1_0000_s2_0500 | 17.42 N | 17.15 N |
| s1_0125_s2_0250 | 15.82 N | 15.92 N |
| s1_0125_s2_0500 | 19.12 N | 19.24 N |
| s1_0250_s2_0250 | 17.83 N | 18.45 N |
| s1_0250_s2_0500 | 19.26 N | 19.52 N |

固定层内部很多层在 `10mm` radius 下没有足够 pair，所以 `all10 theta/T p95` 为 `None`。这不是训练失败，而是每个固定层映射到 workspace 后点间距离较稀，10mm 局部球不足以形成稳定统计。这个现象说明：如果后续要走固定层专家路线，需要增加每层采样密度，而不是只用 `41x41` 网格。

#### 13.10.4 MLP 训练结果

本轮训练使用 classic sklearn MLP；TF MLP 未完成，因为当前 `dante_env` 中的 `tensorflow` 是 namespace package，`tf.__version__ = None` 且缺少 `tf.random`，在 `tf.random.set_seed()` 处失败。这是环境问题，不是数据问题。失败证据保留在训练日志中；本轮模型对比以 classic MLP 为准。

MLP 对比结果：

| split | original T MAE | layer-field T MAE | fixed-layer macro T MAE |
| --- | ---: | ---: | ---: |
| IID | 25.44 N | 27.60 N | 24.21 N |
| radius | 53.77 N | 61.80 N | 54.59 N |
| angular-sector | 72.41 N | 76.30 N | 69.16 N |

theta 与 EE 指标：

| split | original theta MAE | layer-field theta MAE | fixed-layer macro theta MAE |
| --- | ---: | ---: | ---: |
| IID | 0.2627 deg | 0.2715 deg | 0.1354 deg |
| radius | 0.3502 deg | 0.3465 deg | 0.3623 deg |
| angular-sector | 0.4813 deg | 0.5843 deg | 0.4371 deg |

| split | original EE p95 | layer-field EE p95 | fixed-layer macro EE p95 |
| --- | ---: | ---: | ---: |
| IID | 15.71 mm | 15.41 mm | 18.25 mm |
| radius | 25.62 mm | 24.79 mm | 55.75 mm |
| angular-sector | 93.72 mm | 96.29 mm | 91.02 mm |

训练耗时也支持固定层专家路线的可扩展性：全局 MLP 每个 split 约 `9-13s`，固定层单专家每个 split 约 `1.5s`。7 个专家总耗时与一个全局模型同量级，但可以并行。

#### 13.10.5 本轮结论

1. **全局连续层场选择没有解决当前 11767 网格的问题。**  
   它和 voxel 单支结果几乎一样，原因是多层候选在同一 voxel 内太少。图优化无法凭空选择不存在的候选层。

2. **固定层专家路线更符合证据，但当前数据密度还不够。**  
   固定层 oracle floor 从全局约 `31N` 降到 `12-19N`，说明单层内部确实更可学；但 MLP 张力 MAE 只小幅改善，尤其 OOD radius 几乎没改善，说明每层 `1681` 行太稀，且张力标签本身仍有外推不连续。

3. **分层主要改善 theta，不足以单独解决 tension。**  
   IID 下 fixed-layer macro theta MAE 从 `0.2627 deg` 降到 `0.1354 deg`，这说明分层能减少构型平均；但 tension MAE 只从 `25.44 N` 降到 `24.21 N`，张力仍受 label continuity / allocator 限制。

4. **branch hint 很有价值。**  
   layer-field 数据的 `branch-aware xyz-NN T MAE = 15.92 N`，远低于 xyz-only `31.94 N`。这支持后续模型加入 `layer_label` / `(s1,s2)` 作为输入或采用 classifier + expert，而不是继续强迫纯 `xyz -> theta/T` 单值模型。

5. **下一步不建议继续在 11767 行上过滤。**  
   更合理的是生成更密的 fixed-layer / layer-aware 数据：例如每个 `(s1,s2)` 层从 `41x41` 提升到更高密度，或在每层内部做 workspace-balanced active sampling，再训练 layer-aware expert。

#### 13.10.6 建议下一步

优先级建议：

1. 先做 **dense fixed-layer pilot**：每层增加采样密度，例如 `81x81` 或更高，保持 7 层分开；
2. 对每层重新计算 continuity，并把 radius 从 `10mm` 扩展到 `20mm/30mm` 辅助判断采样稀疏性；
3. 训练两类模型：
   - `xyz + one-hot(layer_label) -> theta/T`；
   - 7 个 `xyz -> theta/T` expert；
4. 如果 dense fixed-layer 仍然 tension MAE 改善有限，再回到张力 allocator / anchor smoothing，而不是继续改 branch selection。

本轮主要报告文件：

```text
data/priority_grid_third_joint_first_v1_layer_field_v1/selection_summary.json
data/priority_grid_third_joint_first_v1_layer_field_v1/diagnostics/layer_field_diagnostics.json
data/priority_grid_third_joint_first_v1_layers/fixed_layers_summary.json
runs/diagnostics/priority_grid_layer_experiments_v1/summary.json
runs/diagnostics/priority_grid_layer_experiments_v1/summary.md
runs/diagnostics/priority_grid_layer_experiments_v1/mlp_training_summary.json
runs/diagnostics/priority_grid_layer_experiments_v1/mlp_training_summary.md
```

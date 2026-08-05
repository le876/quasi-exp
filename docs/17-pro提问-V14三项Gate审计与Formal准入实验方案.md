---
question_id: Q17
question_number: 17
question_confirmed_by_user: true
question_confirmation_summary: "审计 BACRA V14 5k-cell Pilot 的 Reach convergence、branch-discovery saturation 和 deployable representation 三项 Gate，判断 Gate 是否影响实验逻辑、能否直接放宽、下一轮如何推进及满足何种证据后可进入 Formal，并明确 inverse representation 的转向条件。"
date: "2026-08-05"
status: ready-to-send
project: "quasi_exp / BACRA"
evidence_cutoff: "2026-08-05T10:42:47+08:00"
source_snapshot: "V14 实现位于 clean worktree codex/bacra-v14-omega200-workspace-atlas@cd892503227a27627345504eec79553d6401ef0b；5k-cell Pilot artifacts 位于 runs/bacra_v14_omega200_workspace_atlas_pilot 并锁定同一 source Git SHA。Q17 文档和 post-hoc evidence builder 位于另一个 dirty 主工作树 canonical-layer-field-u3@740d8c00fd8faf9085e3e05ba25eb212d202048f，不属于 V14 实现 commit。"
---

# GPT-5 Pro 第 17 次交接：V14 三项 Gate 审计与 Formal 准入实验方案

## 0. 经用户确认的 GPT-5 Pro 任务

用户已经确认的任务原意：

> 请 GPT-5 Pro 基于 BACRA V14 5k-cell 科学 Pilot 的真实代码与 artifacts，审计 Reach convergence、branch-discovery saturation 和 deployable representation 三项 Gate 未通过的科学含义，区分实验定义或实现缺口、搜索与图连接预算不足，以及 \(6\to3\) 冗余 IK 本身的表示限制，并判断下一轮应该如何推进，当前 Gate 是否影响实验逻辑，是否可以直接放宽到通过、满足什么证据条件后才有资格进入 Formal。

用户确认的回答要求：

> 返回一份原因审计和可执行的下一轮实验方案，说明需要优先验证或修改的数学定义与实现模块、必要对照、指标、Gate、停止或转向条件及计算预算优先级；同时明确无状态 \(xyz\)-only inverse 是否仍应保留，以及何种证据会要求改用 stateful、branch token、multiple candidates 或 abstention。

确认范围：本次只审计已完成 V14 Pilot 的三项科学 Gate、Gate 与实验逻辑的关系、是否可放宽、Formal 准入证据、下一轮实验以及 inverse representation 的决策边界。本文不替用户预先选择路线，不把 post-hoc 统计升级为正式结论，也不要求 GPT-5 Pro 在本轮直接修改代码。

## 1. 当前有效事实与最短摘要

### 1.1 证据截止时间

本文只使用截至 `2026-08-05T10:42:47+08:00` 已存在并重新核验的 V14 代码、冻结配置、运行日志和 artifacts。V14 Pilot 在 `2026-08-05 02:46:06+08:00` 开始，在 `08:40:40+08:00` 以 `rc=0` 结束，墙钟时间 `21,274 s`，即 `5 h 54 min 34 s`。

已独立重算并核对：

- artifact manifest 的 `96/96` 个文件均存在，大小和 SHA256 全部匹配；
- source manifest 的 `71/71` 个 implementation sources 均存在，大小和 SHA256 全部匹配；
- `9/9` 个 external sources 的 SHA256 全部匹配；
- V14 source worktree 当前 clean，HEAD 仍为 `cd892503227a27627345504eec79553d6401ef0b`。

### 1.2 当前有效事实

1. V14 已实现 Q16 计划经审查后的主要修正：独立 scrambled Reach replicas、frontier inverse discovery、Correction/Diversity candidate modes、branch saturation audit、cell multi-probe、one-candidate-per-task section、全 fundamental-cycle fresh audit、chart stitchability Gate、primary/expert/stateful 数据 contract，以及 fail-closed representation/Formal Gate。
2. 5k-cell Pilot 的 10 个 operational stages 全部完成，各 stage `gate_pass=true`；但总科学状态为 `pilot_gate_pass=false`、`scientific_gate_pass=false`、`formal_authorized_by_this_run=false`、`deployment_claim_gate_pass=false`。Operational 完成不等于科学 Gate 通过。
3. Reach Gate 未通过：第三轮 volume-weighted Jaccard 为 `0.927166`，低于 `0.95`；new-volume ratio 为 `0.027618`，高于 `0.01`；boundary-change ratio 为 `0.150640`，高于 `0.02`。frontier-new-volume ratio `0.000430` 单项通过 `<=0.01`，但不足以使总体收敛。
4. Branch saturation Gate 未通过：500 个高预算审计 cell 中 `500/500` 都按当前 `>1 deg` candidate-family gap 规则发现新 family；普通预算 cluster 平均 `3.444`，高预算平均 `32.988`。该结果证明有限 candidate discovery 未饱和，但不自动证明平均存在约 33 个拓扑 branch。
5. Atlas 的局部一致性审计通过，但全域 representation 被阻断：16 个 section 各只选择 5 个 probes；5,000 个 Pilot cells 中只有 7 个 `resolved_single_under_budget`、4 个 `resolved_multichart`、4,989 个 `teacher_unresolved`；25,000 个 probes 中只有 5 个属于 primary `chart_06`。
6. 77,453 条 robust product edges 中，77,426 条只跨同一 cell 内的不同 probes，只有 27 条跨 cell，对应 25 个不同 cell pairs。该事实说明跨-cell graph support 很稀疏，但现有证据未证明其根因。
7. 6 个实际 chart overlaps 全部为 `non_stitchable`，beta-gap P95 范围 `1.1295..8.9290 deg`。现有实现没有用 FK residual 在两个同样有效的 IK branches 之间选择 canonical branch。
8. 当前 11 个 labelable cells 对应的预算算术为 `N_min=30,050 <= 200,000`，所以 budget feasibility Gate 通过；但 representation 为 `blocked`，因此监督数据 materialization 行数为 0，Student 未训练，Formal 未执行。

### 1.3 关键边界

1. Reach A/B replicas 是独立 scrambled Sobol lineage，但仍是有限前向采样得到的 empirical proxy，不是连续 `Reach(FK)` 的数学证明；numerical IK failure 没有被升级为 certified unreachable。
2. Artifact 字段 `new_stable_branch` 实际由 candidate-family gap criterion 触发。对于 \(6\to3\) 冗余映射，距离超过 `1 deg` 的解可能仍属于同一连续 fiber/component；现有 Pilot 没有完成 same-fiber versus distinct-component 分类。
3. 16 个局部 section 的 path/cycle/multipath audit 通过，只支持这些小 section 的局部一致性；不能支持全工作空间存在连续、无状态、单值 `xyz -> beta6` inverse。
4. 跨-cell robust edge 很少是已观察事实，不是已证明根因。它可能与 task adjacency、candidate pairing、continuation seed/阈值、section extraction 或真实几何结构有关，当前 artifacts 无法裁决。
5. 当前没有 50k Pilot supervision dataset、200k Formal dataset、任何 V14 Student、sealed spatial result、trajectory result 或 deployment claim。

## 2. 项目目标与本轮边界

### 2.1 用户给定的项目目标

用户的目标是在固定约 `200,000` 条监督数据预算下，不再只对一个局部椭球壳体加密，而是逼近：

```text
Omega_200 = Reach(FK, registered beta bounds)
            intersect {1.015498 m <= x <= 1.215498 m}
```

目标系统需要在扩大任务空间覆盖的同时，显式处理冗余 IK 的多值性、chart 连续性、canonical labeling 和 Student 可部署表示；不能通过删除 unresolved 或 multichart 区域制造虚假的全域 coverage。

### 2.2 当前实际覆盖范围

V14 当前实际执行到科学 Pilot：

1. 三轮独立 A/B Reach occupancy 与 tip-focused/frontier discovery；
2. `20/10/5 mm` 自适应 cell registry；
3. 从 43,442 个 eligible cells 分层选择 5,000 cells，覆盖 19 个 x-bins 及 interior、boundary、tip、retention strata；
4. 每 cell 1 个代表点加 4 个 measure probes，共 25,000 task probes；
5. 普通 candidate search 与 10% 高预算 saturation audit；
6. bidirectional continuation product graph、explicit section extraction、fresh cycle/multipath audit 和 overlap stitchability 分类；
7. representation Gate、fixed-budget arithmetic 和下游 fail-closed skip。

### 2.3 当前证据未覆盖的内容

- 没有证明当前 Reach proxy 已达到预注册收敛要求，也没有连续 Reach 的 certified upper/lower enclosure。
- 没有把 candidate clusters 分类为同一 fiber 内连续族或不同拓扑 component。
- 没有完成大尺度、跨 cell 的 atlas section 覆盖。
- 没有证明无状态 `xyz-only` partition 的切换边界可 stitch。
- 没有执行任何 relaxed-Gate 对照，因而没有证据说明放宽哪个阈值不会改变 claim 含义。
- 没有 materialize 50k/200k 监督表，没有训练或比较 global、router-expert、stateful Student。
- 没有 Formal run；Formal output root 当前不存在。
- 没有真实机器人部署、张力标签生产或原论文最终 42 维输出的本轮证据。

## 3. 相对上次材料新增的事实

上次材料是 Q16。`q16_previous_protocol.md` 同时包含当时用户问题和历史 GPT-5 Pro 方案；后半部分属于历史建议，不是 V14 已执行事实。本 Q17 只把实际代码和 artifacts 作为当前证据。

| 类型 | 已发生的变化 | 实际证据 | 证据等级 |
| --- | --- | --- | --- |
| 实现/配置 | 新增 V14 workspace Reach、registry、candidate bank、atlas、representation、dataset 和 Student modules，共 28 个提交文件、13,985 行新增 | `v14_context.md`、`v14_config.yaml`、`v14_runner.py`、`v14_source_manifest.json` | implementation fixed point |
| 实验 | Smoke 后实际完成 5k-cell Pilot；10 个 operational stages 完成，96 个 artifacts 封存 | `v14_pilot_log.txt`、`v14_operational_gate.json`、`v14_artifact_manifest.json` | exploratory scientific Pilot with sealed provenance |
| 结论修正 | 预算算术可行不等于可生成数据；三项科学 Gate 未通过使 representation、dataset、Student、Formal fail closed | `v14_pilot_gate.json`、`v14_budget_report.json`、`v14_dataset_report.json`、`v14_student_report.json` | current Pilot result |
| 诊断 | 全表聚合发现 robust edges 几乎都在同一 cell 内；生成 27 条跨-cell edge 完整清单 | `q17_pilot_gate_audit.json`、`q17_cross_cell_robust_edges.csv` | post-hoc diagnostic-only |

## 4. 实验方法与复现口径

### 4.1 代码与工作树 fixed point

| 项目 | 值 |
| --- | --- |
| 项目根 | `/mnt/ML_projects/quasi_exp` |
| V14 分支 | `codex/bacra-v14-omega200-workspace-atlas` |
| V14 HEAD | `cd892503227a27627345504eec79553d6401ef0b` |
| V14 worktree | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v14-omega200-workspace-atlas` |
| V14 dirty 状态 | clean |
| V14 commit subject | `Implement BACRA V14 workspace atlas pilot` |
| 仅 checkout V14 HEAD 是否足以复现代码 | 是；真实输入数据还必须满足 `v14_source_manifest.json` 中 9 个 external-source hashes |
| Q17 文档与 evidence builder | 位于 dirty 主工作树 `canonical-layer-field-u3@740d8c00...`，是 post-hoc 交接材料，不属于 V14 commit |

标准 Python：

```text
/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python
```

### 4.2 方法与参数

| 模块 | 实际 Pilot 参数 | 来源 |
| --- | --- | --- |
| Target domain | `x=[1.015498,1.215498] m`，grid levels `[20,10,5] mm` | `v14_frozen_config.json: domain` |
| Reach A/B | powers `17,18,18`，A seeds `20260841/843/845`，B seeds `20260842/844/846` | `v14_frozen_config.json: reach.rounds` |
| Reach Gate | weighted Jaccard `>=0.95`，new volume `<=0.01`，boundary change `<=0.02`，frontier new volume `<=0.01`，连续轮数 `2` | `v14_frozen_config.json: reach` |
| Frontier | `2` rounds，`128` cells/round，`8` starts/cell | `v14_frozen_config.json: reach` |
| Pilot cells | `5,000`；每 cell `4` measure probes 加代表点；每代表点 `4` seeds | `v14_frozen_config.json: registry` |
| Candidate base | normal starts `4`，difficult starts `8`，nullspace starts `2`；DLS、bounded least squares、SLSQP | `v14_frozen_config.json: candidates` |
| Saturation | `10%` cells，`32` starts，new-family gap `1 deg`，cluster merge `0.5 deg` | `v14_frozen_config.json: candidates` |
| Atlas | 32 roots，最多 16 sections；edge match `0.5 deg`；cycle P95/max `0.5/1.0 deg` | `v14_frozen_config.json: atlas` |
| Representation | primary measure coverage `>=0.80`，每 x-bin coverage `>=0.60`，abstention `<=0.20` | `v14_frozen_config.json: atlas` |
| Branch saturation | overall Wilson upper `<=0.02`，stratum upper `<=0.05` | `v14_frozen_config.json: atlas` |
| Budget | total `200,000`，hard max `300,000`，Pilot target `50,000` | `v14_frozen_config.json: budget` |
| Split | 40 mm macroblock；train/validation/sealed/active fractions `0.625/0.15/0.15/0.075` | `v14_frozen_config.json: split` |
| Parallel | candidate workers `4`；配置声明 continuation workers `4`；worker timeout `14,400 s` | `v14_frozen_config.json: parallel` |

### 4.3 数据生成与划分

| 项目 | 实际口径 |
| --- | --- |
| Reach 数据来源 | 已锁 capability pool 加两组独立 scrambled Sobol FK replicas、tip-focused pool 与 frontier inverse probes |
| Reach 实际规模 | A slab `297,172` rows；B slab `297,099` rows；lower/upper cells `18,689/19,301` |
| Registry | 43,442 eligible cells，分层选择 5,000；25,000 task probes；20,000 seed rows；10,715 task edges |
| Candidate generation | 5,000/5,000 representative tasks 至少有一个 candidate；90,888 solver attempts；500 saturation audit cells |
| Atlas | 160,884 atlas candidate rows；71,631 product candidates；376,020 continuation attempts；77,453 robust edges |
| Supervision rows | `0`，因为 representation Gate blocked；没有 padding |
| train/validation/sealed | 未 materialize，配置存在但未执行 split |
| leakage audit | 未执行；只有 40 mm macroblock split 实现和配置，没有数据可审计 |

### 4.4 baseline、评价和 Gate

| 项目 | 实际定义 | 本轮状态 |
| --- | --- | --- |
| Operational stage Gate | 检查每阶段 artifacts、exact sets、skip semantics 和上游 closure | 10/10 stages pass |
| Reach convergence Gate | 独立 A/B occupancy 及新体积、边界和 frontier 变化 | fail |
| Branch saturation Gate | 高预算审计后新 candidate-family rate 的 overall/stratum Wilson upper | fail |
| Atlas local audit | fresh path、direction、all fundamental cycles、multipath、repeat | pass |
| Representation Gate | primary measure/x-bin coverage、overlap stitchability、abstention与 branch saturation | fail，mode=`blocked` |
| Budget feasibility | 当前 labelable cells 的最低监督预算是否 `<=200k` | pass，但不授权数据生成 |
| Student baseline | global bounded MLP、xyz-only router-experts、stateful delta-beta paths 已实现 | 未训练，无法比较 |
| Spatial/trajectory evaluation | 需要锁定 Student 和 sealed splits | 未授权并安全跳过 |

### 4.5 复现入口与产物

实际 Pilot 命令：

```text
/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python \
  scripts/analysis/run_bacra_v14_omega200_atlas.py --preset pilot
```

主要产物：

```text
/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot
/mnt/ML_projects/quasi_exp/runs/maintenance/bacra_v14_pilot.log
```

Formal 入口存在，但 runner 在 formal protocol 阶段强制读取 persisted Pilot Gate。当前会产生四个拒绝原因：

```text
pilot_gate_failed
reach_convergence_not_proven
branch_discovery_not_saturated
deployable_representation_not_authorized
```

### 4.6 测试状态

未经修改的标准入口：

```text
/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python -m pytest -q
```

在 collection 阶段以 `rc=2` 失败，错误为：

```text
ModuleNotFoundError: No module named 'tests.test_segmented_tension_solver'
```

实际环境中 `/home/ubuntu/.local/lib/python3.11/site-packages/tests/__init__.py` 遮蔽仓库 `tests` namespace。使用 process-local repo-tests namespace injection 后，输出包含 756 个 test dots 和 `[100%]`，以 `rc=0` 结束。二者必须分开表述：当前不能声称未经修改的标准 pytest 入口通过。

## 5. 实验结果

| Fact ID | 已观察结果 | 数值 | 证据等级 | 实际附件/JSON key | 适用边界 |
| --- | --- | ---: | --- | --- | --- |
| F1 | 运行与 artifact closure 完成 | 10/10 stages；96 artifacts | operational | `v14_summary_report.json: stage_gate_pass`、`v14_artifact_manifest.json` | 不等于 scientific Gate pass |
| F2 | Reach lower/upper proxy | 18,689 / 19,301 cells | exploratory | `v14_workspace_proxy_report.json` | empirical proxy，不是连续 Reach 证明 |
| F3 | 第三轮 weighted Jaccard | 0.927166，阈值 0.95 | exploratory fail | `v14_workspace_proxy_report.json: metrics[2]` | A/B finite occupancy |
| F4 | 第三轮 new-volume/boundary change | 0.027618 / 0.150640，阈值 0.01 / 0.02 | exploratory fail | `v14_workspace_proxy_report.json: metrics[2]` | 不能单独判断根因 |
| F5 | Frontier new volume | 0.000430；8/256 probes 新增 support | exploratory pass component | `v14_workspace_proxy_report.json`、`v14_frontier_inverse_probes.parquet` | 单项通过不使 Reach Gate 通过 |
| F6 | 普通 candidate coverage | 5,000/5,000 tasks resolved under budget；90,888 attempts | exploratory | `v14_candidate_report.json` | 至少一解，不是 branch completeness |
| F7 | 高预算发现新 family | 500/500 cells；observed rate/Wilson upper 1.0 | exploratory fail | `v14_candidate_report.json`、`v14_branch_saturation_audit.parquet` | `>1 deg` family，不等于拓扑 branch |
| F8 | base versus saturation clusters | mean 3.444 versus 32.988；saturation range 17..49 | exploratory | `q17_pilot_gate_audit.json: branch_discovery` | 依赖当前 clustering/seed/solver 定义 |
| F9 | 局部 atlas consistency | path/direction/loop/multipath P95 `0.3467/0.3467/0.3956/0.4072 deg` | exploratory pass | `v14_workspace_atlas_report.json: fresh_audit.aggregate` | 只针对已提取小 charts |
| F10 | Atlas chart 数与大小 | 16 charts，每 chart selection count 恰为 5 | exploratory | `v14_charts.parquet`、`q17_pilot_gate_audit.json` | 每 chart 只覆盖一个 cell 的 probes |
| F11 | Cell classification | single 7，multichart 4，unresolved 4,989 | exploratory fail | `v14_selected_cell_classification.parquet` | 仅 5k stratified Pilot cells |
| F12 | 全 registry measure ratio | single 0.0084477%，multichart 0.0042239%，unresolved 99.9873284% | exploratory fail | `v14_workspace_atlas_report.json: domain_class_measure_ratio` | empirical registered domain measure |
| F13 | Primary partition | chart_06 仅 5 probes；uncovered 24,995 | exploratory fail | `v14_primary_chart_partition.parquet` | 不能支持静态全域 section |
| F14 | Robust edge 分布 | within-cell 77,426；cross-cell 27；25 distinct cell pairs | post-hoc diagnostic | `q17_pilot_gate_audit.json`、`q17_cross_cell_robust_edges.csv` | 全表聚合；不证明原因 |
| F15 | Chart overlaps | 6/6 non-stitchable；P95 gap 1.1295..8.9290 deg | exploratory fail component | `v14_chart_overlaps.parquet` | 只针对已发现 overlaps |
| F16 | Budget arithmetic | `N_min=30,050 <= 200,000` | exploratory pass | `v14_budget_report.json` | 基于当前仅 11 labelable cells |
| F17 | Dataset/Student | supervision rows 0；training unauthorized | fail-closed skip | `v14_dataset_report.json`、`v14_student_report.json` | 没有模型性能结果 |
| F18 | Formal/deployment | both unauthorized | current Gate fact | `v14_pilot_gate.json`、`v14_summary_report.json` | 不能形成 Formal claim |

### 5.1 通过项

- V14 fixed point、71 个 implementation files、9 个 external sources 和 96 个 Pilot artifacts 的 hash/size closure 通过。
- 两组 Reach replicas 的 scrambled Sobol seed lineage 独立；没有把 numerical failure 标为 certified unreachable。
- 5,000-cell 分层选择没有 padding，并覆盖全部 19 个 eligible x-bins。
- 局部 section extraction、fresh path/direction audit、全部 96 个 fundamental cycles、multipath 和 repeat audit 通过。
- 当前 labelable cells 下的 200k budget arithmetic 通过。
- Representation blocked 时数据、Student、spatial、trajectory stages 正确以显式空表/skip artifact 结束，没有制造冲突监督 rows。

### 5.2 失败与反例

- Reach 的三项主要 convergence indicators 没有达到冻结阈值，也没有满足连续两轮通过。
- 500/500 高预算审计 cells 均发现新 candidate family，远高于 overall `0.02` 和 stratum `0.05` Wilson upper Gate。
- 16 个局部 charts 没有形成 domain-scale atlas；跨-cell robust connections 只有 27 条。
- primary measure/x-bin coverage 不足，representation mode 为 `blocked`。
- 6 个已发现 overlaps 全部 non-stitchable，不能用“两个 expert 都有低 FK residual”证明 canonical branch 已确定。
- 50k/200k 数据、Student、sealed spatial 和 trajectory evidence 全部不存在。

## 6. 证据等级、失败结果与未知事项

### 6.1 正式事实

本轮没有 V14 Formal 实验事实。可以正式陈述的只有 provenance 和执行状态：V14 Pilot 绑定 clean commit，artifacts/hash closure 完整，Pilot scientific Gate 为 false，Formal 未授权且未执行。

### 6.2 诊断、探索、evidence-only、post-hoc 或历史事实

| 内容 | 等级 | 能支持什么 | 不能支持什么 |
| --- | --- | --- | --- |
| Q16 计划及历史 GPT 回答 | historical | 解释 V14 设计来源 | 证明 V14 已执行或 Gate 合理 |
| 5k-cell Pilot | exploratory with sealed provenance | 比较当前固定配置下三项 Gate、成本和 failure modes | 全域、Formal 或部署结论 |
| Reach A/B occupancy | exploratory empirical proxy | 独立 replica 的有限占用差异 | 连续 Reach 的完整性/不可达证明 |
| Candidate saturation audit | exploratory | 普通预算 candidate-family discovery 未饱和 | candidate family 等同拓扑 branch |
| Local atlas audit | exploratory | 已选 section 内 continuation consistency | 全域静态 inverse 存在 |
| `q17_pilot_gate_audit.json` | post-hoc diagnostic-only | 复核全表计数、阈值比较、cross-cell edges | 改变 Gate、选择路线或证明根因 |
| namespace-injected pytest | verification workaround | 说明当前进程注入下 756 tests 到达 100% | 未修改标准 pytest 入口通过 |

### 6.3 已发现的实现或证据缺口

- 当前 branch audit 的核心标签仍叫 `new_stable_branch`，而其实际语义是超过 `new_branch_gap_deg` 的新 candidate family；缺少 fiber connectivity/component classification。
- `parallel.continuation_workers=4` 已写入配置，但当前 Atlas graph/audit 主流程仍主要单进程执行；Pilot 总时长接近 6 小时。
- Atlas graph 的绝大多数 robust edges 都是同一 cell 内 probe edges；现有报告原先没有直接暴露 within/cross-cell 分解，Q17 post-hoc 聚合才给出该计数。
- Representation Gate 的失败原因记录为 `primary_measure_coverage_gate_failed`，但当前 artifacts 没有将 coverage 缺口进一步归因到 task adjacency、candidate pairing、continuation、section extraction 或真实 topology。
- 没有 relaxed-threshold 或 increased-budget 对照，无法从单个 Pilot 区分 Gate 过严与方法未达到目标。

### 6.4 尚未知或尚未执行

- Reach 再增加独立 rounds 后是否收敛，以及收敛速度是否与边界/薄 preimage strata 有系统关系。
- 当前 17..49 个高预算 clusters 中，哪些可通过 null-space continuation 连成同一 fiber component。
- 若固定 candidates 和 task graph，仅扩大 edge proposal/continuation budget，跨-cell robust edge 数和 primary coverage 会如何变化。
- 若固定 continuation budget，仅修改 task adjacency、candidate matching 或 section extraction，会发生何种变化。
- 是否存在覆盖主要 measure 且所有必要切换都 stitchable 的无状态 primary section。
- Stateful、branch token、multiple candidates 和 abstention 在同一 sealed spatial/trajectory protocol 下的比较结果。
- 任意 Gate 放宽后是否只改变“通过/失败”标签，还是改变目标 claim 的科学含义。

### 6.5 当前不能声称

- 不能声称 Reach proxy 已稳定或 `Omega_200` 已完整发现。
- 不能声称普通搜索已发现所有 IK branches。
- 不能把 32.988 clusters 解释为平均 32.988 个拓扑 branches。
- 不能声称 16 个 charts 构成覆盖工作空间的 atlas。
- 不能声称纯 `xyz-only` Student 可部署，也不能声称 stateful 一定优于其他表示。
- 不能声称 200k 预算已被实际验证为充分；当前 `N_min=30,050` 只来自 11 个 labelable cells。
- 不能声称已有 50k/200k 数据集、V14 Student、Formal 或 deployment 结果。
- 不能在没有对照的情况下把 Gate 直接改成当前观测值并称为科学通过。

## 7. 决策所需的客观对照

本节只比较已执行的方法、配置或结果，不给出 Codex 推荐或路线排序。

| 对照对象 | 已执行输入/方法 | 已观察结果 | 证据等级 | 已知限制 |
| --- | --- | --- | --- | --- |
| Operational Gate | 每阶段 artifact/exact-set/skip closure | 10/10 pass | operational | 不要求三项科学目标成立 |
| Scientific Pilot Gate | Reach + branch saturation + representation + downstream evidence | fail | exploratory | 是否阈值合理仍是待审计问题 |
| Reach round 1 | Sobol power 17 A/B | weighted Jaccard 0.784315 | exploratory | 首轮 boundary/new-volume 定义为 1.0 |
| Reach round 2 | 追加 power 18 A/B | weighted Jaccard 0.892841；new volume 0.104913 | exploratory | 未收敛 |
| Reach round 3 | 再追加 power 18 A/B | weighted Jaccard 0.927166；new volume 0.027618 | exploratory | 趋势改善但仍未达到冻结 Gate |
| 普通 candidate budget | 4/8 starts、2 nullspace starts | mean clusters 3.444 | exploratory | 只能说明当前预算下发现内容 |
| saturation budget | 32 starts、所有 solver/diversity attempts | mean clusters 32.988；500/500 新 family | exploratory | family/component 语义未分离 |
| Atlas local audit | 16 个 5-probe sections | path/cycle/multipath pass | exploratory | 空间覆盖极小 |
| Atlas global support | 25k probes、77,453 robust edges | 27 cross-cell edges；5 primary probes | post-hoc plus exploratory | 原因未区分 |
| Stitchability | 6 个已发现 overlaps | 6/6 non-stitchable | exploratory | 未发现 overlap 不能代表不存在 overlap |
| Budget Gate | 当前 11 labelable cells | `N_min=30,050` pass | exploratory | labelable domain 极小 |
| Static/global Student | code path 已实现 | 未训练 | implementation-only | 无性能证据 |
| xyz-only router-experts | code path已实现；不接受 external known chart | 未训练 | implementation-only | representation Gate blocked |
| Stateful delta-beta | code path已实现 | 未训练 | implementation-only | 未注册历史状态/trajectory supervision |
| Standard pytest | 未修改环境 | collection rc=2 | verification failure | site-packages tests namespace collision |
| Namespace-injected pytest | process-local tests namespace | 756 dots、100%、rc=0 | workaround verification | 不是标准入口 |

## 8. 经用户确认的问题

1. 请基于真实 V14 代码与 Pilot artifacts，审计 Reach convergence、branch-discovery saturation 和 deployable representation 三项 Gate 未通过的科学含义；区分实验定义或实现缺口、搜索与图连接预算不足，以及 \(6\to3\) 冗余 IK 本身的表示限制；判断下一轮应如何推进，当前 Gate 是否影响实验逻辑、是否可以直接放宽到通过，以及满足什么证据条件后才有资格进入 Formal。

## 9. 经用户确认的回答要求

请返回一份原因审计和可执行的下一轮实验方案，并覆盖用户明确要求的以下内容：

- 需要优先验证或修改的数学定义与实现模块；
- 必要对照；
- 指标；
- Gate；
- 停止或转向条件；
- 计算预算优先级；
- 明确无状态 `xyz-only inverse` 是否仍应保留；
- 明确何种证据会要求改用 stateful、branch token、multiple candidates 或 abstention。

请把已有 artifact 直接支持的事实、由代码推导的机制、仍需实验验证的解释和最终建议分开，不把 Pilot 中的 candidate-family cluster 数直接当成拓扑 branch 数，也不把 operational Gate 通过误写成 scientific Gate 通过。

## 10. 附件清单与复现信息

GPT-5 Pro 实际可见的是“上传文件名”列；原始 SSH 路径只用于 provenance。

唯一上传 ZIP 的成员数（`QUESTION.md + 51` 个实际证据附件）：`52`。无论成员数多少，网页端只上传打包器生成的 `Q17_GPT5Pro_upload.zip`。

| 上传文件名 | 原始 SSH 路径 | 大小 | 证据等级 | 支持事实 | 上传理由 |
| --- | --- | ---: | --- | --- | --- |
| `q16_previous_protocol.md` | `/mnt/ML_projects/quasi_exp/docs/16-pro提问-200mm工作空间覆盖与20万点泛化数据集设计.md` | 79,303 B | historical | V14 设计来源 | 区分上轮协议/建议与本轮执行事实 |
| `v14_context.md` | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v14-omega200-workspace-atlas/CONTEXT.md` | 2,230 B | implementation | fixed point、范围 | 最短实现上下文 |
| `v14_config.yaml` | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v14-omega200-workspace-atlas/configs/bacra_v14_omega200_workspace_atlas.yaml` | 8,003 B | implementation | thresholds、presets | 审计 Gate 定义和 Formal override |
| `v14_runner.py` | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v14-omega200-workspace-atlas/scripts/analysis/run_bacra_v14_omega200_atlas.py` | 156,732 B | implementation | 全 pipeline、Gate、skip | 审计实际控制流 |
| `v14_workspace_protocol.py` | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v14-omega200-workspace-atlas/src/quasi_exp/teacher/workspace_protocol.py` | 35,260 B | implementation | schema、manifest、Gate contract | 审计 fail-closed 语义 |
| `v14_workspace_reach.py` | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v14-omega200-workspace-atlas/src/quasi_exp/teacher/workspace_reach.py` | 16,130 B | implementation | independent replicas、occupancy | Reach Gate 实现 |
| `v14_workspace_registry.py` | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v14-omega200-workspace-atlas/src/quasi_exp/teacher/workspace_registry.py` | 32,293 B | implementation | 20/10/5 mm cells、multi-probe | 审计 cell measure |
| `v14_workspace_candidate_bank.py` | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v14-omega200-workspace-atlas/src/quasi_exp/teacher/workspace_candidate_bank.py` | 19,290 B | implementation | correction/diversity、cluster | Branch audit 实现 |
| `v14_workspace_atlas.py` | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v14-omega200-workspace-atlas/src/quasi_exp/teacher/workspace_atlas.py` | 27,689 B | implementation | product graph、section extraction | Atlas 原理审计 |
| `v14_workspace_atlas_integration.py` | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v14-omega200-workspace-atlas/src/quasi_exp/teacher/workspace_atlas_integration.py` | 87,115 B | implementation | continuation/cycle/representation | 三项 Gate 的主要集成实现 |
| `v14_workspace_inverse.py` | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v14-omega200-workspace-atlas/src/quasi_exp/teacher/workspace_inverse.py` | 16,411 B | implementation | solver 与 normalized Jacobian | 搜索/病态性审计 |
| `v14_workspace_dataset.py` | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v14-omega200-workspace-atlas/src/quasi_exp/teacher/workspace_dataset.py` | 20,985 B | implementation | primary/expert/stateful contract | 审计 blocked 后的数据语义 |
| `v14_workspace_student.py` | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v14-omega200-workspace-atlas/src/quasi_exp/teacher/workspace_student.py` | 28,959 B | implementation | global/router/stateful code paths | Representation 备选实现边界 |
| `v14_pilot_log.txt` | `/mnt/ML_projects/quasi_exp/runs/maintenance/bacra_v14_pilot.log` | 9,948 B | operational/exploratory | 命令、时间、rc、stage summaries | 证明实际 Pilot 执行 |
| `v14_standard_pytest_failure.log` | `/mnt/ML_projects/quasi_exp/runs/maintenance/bacra_v14_full_pytest.log` | 1,225 B | verification | 标准入口 rc=2 | 保留测试失败边界 |
| `v14_namespace_pytest_pass.log` | `/mnt/ML_projects/quasi_exp/runs/maintenance/bacra_v14_full_pytest_namespace_injected_retry1.log` | 8,383 B | workaround verification | 756 tests、100%、rc=0 | 与标准入口分开核验 |
| `v14_frozen_config.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/00_protocol/frozen_config.json` | 5,112 B | operational | 实际 Pilot 参数 | 防止只看 source config |
| `v14_source_manifest.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/00_protocol/source_manifest.json` | 13,912 B | operational | 71 source、9 external hashes | source closure |
| `v14_artifact_manifest.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/11_summary/artifact_manifest.json` | 17,690 B | operational | 96 artifact hashes | artifact closure |
| `v14_summary_report.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/11_summary/summary_report.json` | 1,118 B | exploratory | 总科学状态 | 区分 operational/scientific |
| `v14_pilot_gate.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/11_summary/pilot_gate.json` | 868 B | exploratory | 三项 Gate、Formal authorization | 核心裁决附件 |
| `v14_operational_gate.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/11_summary/gate.json` | 1,449 B | operational | stage completion | 防止将 `gate_pass=true` 误解为科学通过 |
| `v14_workspace_proxy_report.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/01_workspace_proxy/workspace_proxy_report.json` | 1,503 B | exploratory | Reach rounds、lower/upper、frontier | Reach 核心结果 |
| `v14_occupancy_convergence.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/01_workspace_proxy/occupancy_convergence.json` | 1,009 B | exploratory | 三轮 occupancy | 收敛趋势审计 |
| `v14_domain_cells.parquet` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/01_workspace_proxy/domain_cells.parquet` | 146,468 B | exploratory | lower/upper cell classification | 复核 proxy cell 表 |
| `v14_replica_a_slab.parquet` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/01_workspace_proxy/replica_a_slab.parquet` | 24,027,526 B | exploratory raw | 297,172 A rows | 允许独立重算 occupancy |
| `v14_replica_b_slab.parquet` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/01_workspace_proxy/replica_b_slab.parquet` | 24,022,267 B | exploratory raw | 297,099 B rows | 允许独立重算 occupancy |
| `v14_frontier_inverse_probes.parquet` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/01_workspace_proxy/frontier_inverse_probes.parquet` | 18,260 B | exploratory raw | 256 probes、8 new support | frontier 结果复核 |
| `v14_registry_report.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/02_domain_registry/registry_report.json` | 772 B | exploratory | 5k selection、25k probes | Cell Pilot 设计 |
| `v14_pilot_task_probes.parquet` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/02_domain_registry/pilot_task_probes.parquet` | 2,726,222 B | exploratory raw | 25,000 task probes | 复核 multi-probe 空间分布 |
| `v14_candidate_report.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/03_candidate_bank/candidate_report.json` | 3,145 B | exploratory | solver attempts、saturation Gate | Candidate 核心结果 |
| `v14_branch_saturation_audit.parquet` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/03_candidate_bank/branch_saturation_audit.parquet` | 19,116 B | exploratory raw | 500-cell audit | 复核 cluster/gap 分布 |
| `v14_cell_candidates.parquet` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/03_candidate_bank/cell_candidates.parquet` | 19,238,629 B | exploratory raw | 普通预算完整 candidate 表 | 比较 base candidate families |
| `v14_saturation_candidates.parquet` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/03_candidate_bank/saturation_candidates.parquet` | 13,269,149 B | exploratory raw | 500-cell 高预算 candidate 表 | 审计 same-fiber/component 问题 |
| `v14_workspace_atlas_report.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/04_workspace_atlas/workspace_atlas_report.json` | 14,705 B | exploratory | coverage、fresh audit、representation | Atlas 主报告 |
| `v14_atlas_report.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/04_workspace_atlas/atlas_report.json` | 14,110 B | exploratory | 16 chart 明细 | 局部 consistency 复核 |
| `v14_atlas_task_nodes.parquet` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/04_workspace_atlas/atlas_task_nodes.parquet` | 1,337,241 B | exploratory raw | 25k task nodes、cell tuple | Join product edges 到 cell |
| `v14_atlas_candidates.parquet` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/04_workspace_atlas/atlas_candidates.parquet` | 16,836,363 B | exploratory raw | 160,884 candidate rows | 复核 edge/candidate 构型 |
| `v14_product_edges.parquet` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/04_workspace_atlas/product_edges.parquet` | 2,430,834 B | exploratory raw | 77,453 robust edges | 审计跨-cell连接 |
| `v14_charts.parquet` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/04_workspace_atlas/charts.parquet` | 8,869 B | exploratory raw | 16 sections | 复核 chart size/audit |
| `v14_chart_overlaps.parquet` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/04_workspace_atlas/chart_overlaps.parquet` | 4,918 B | exploratory raw | 6 non-stitchable overlaps | Representation 决策证据 |
| `v14_selected_cell_classification.parquet` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/04_workspace_atlas/selected_cell_classification.parquet` | 15,461 B | exploratory raw | 7/4/4,989 cell status | 复核 cell-level coverage |
| `v14_primary_chart_partition.parquet` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/04_workspace_atlas/primary_chart_partition.parquet` | 150,730 B | exploratory raw | 5 primary / 24,995 uncovered probes | 静态 section Gate |
| `v14_task_probe_labels.parquet` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/04_workspace_atlas/task_probe_labels.parquet` | 1,317,451 B | exploratory raw | probe labels/candidates | 复核 labelability |
| `v14_domain_classification.parquet` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/04_workspace_atlas/domain_classification.parquet` | 874,663 B | exploratory raw | 全 registry domain class measure | 复核 99.9873% unresolved |
| `v14_budget_report.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/05_budget_allocation/budget_report.json` | 513 B | exploratory | `N_min=30,050` | 区分预算 Gate 与 representation Gate |
| `v14_dataset_report.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/06_dataset/dataset_report.json` | 217 B | fail-closed | rows 0、blocked | 证明没有伪造数据 |
| `v14_student_report.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v14_omega200_workspace_atlas_pilot/07_student/student_report.json` | 156 B | fail-closed | training unauthorized | 证明没有 Student 结果 |
| `q17_evidence_builder.py` | `/mnt/ML_projects/quasi_exp/scripts/analysis/build_q17_handoff_evidence.py` | 20,574 B | post-hoc code | 派生逻辑 | 允许复核全表聚合方法 |
| `q17_pilot_gate_audit.json` | `/mnt/ML_projects/quasi_exp/runs/gpt5pro_handoff_evidence/Q17/q17_pilot_gate_audit.json` | 81,681 B | post-hoc diagnostic | thresholds、counts、source hashes | 一站式事实索引，不改变 Gate |
| `q17_cross_cell_robust_edges.csv` | `/mnt/ML_projects/quasi_exp/runs/gpt5pro_handoff_evidence/Q17/q17_cross_cell_robust_edges.csv` | 3,546 B | post-hoc diagnostic | 全部 27 cross-cell edges | 允许逐行检查连接 |

### 未上传的大文件或派生数据说明

本包上传了判断三项 Gate 所需的完整 Reach A/B slabs、普通与 saturation candidate tables、Atlas candidates、task nodes 和 product edges，不是仅上传摘要。

未上传整个 Pilot 目录，也未上传重复 shard files、完整 node-report JSON、tip pool、seed banks、candidate diagnostics 的重复副本、product nodes、candidate clusters、空 prediction Parquets 或历史 V13 全数据。它们仍由 `v14_artifact_manifest.json` 锁定，但 GPT-5 Pro 无法从本 ZIP 逐行检查这些未上传表。本问题的三个 Gate 可以由已上传 raw tables、报告、config 和代码直接审计；若 GPT 的结论依赖某个未上传表，应在回答中明确指出所需文件，而不是假定已读取。

`q17_pilot_gate_audit.json` 是全表、无抽样的 post-hoc aggregate：

- 对每个列出的源文件记录 path、bytes、SHA256，Parquet 还记录 row/column schema；
- `random_seed=null`；
- 所有列出的 Parquet source 均读取全部 rows；
- 不重跑 IK、不修改阈值、不修改任何 Gate；
- 不能证明三项失败的根因。

`q17_cross_cell_robust_edges.csv` 从完整 `77,453` 行 `v14_product_edges.parquet` 与完整 `25,000` 行 `v14_atlas_task_nodes.parquet` join 后，筛选左右 `cell_level_mm/cell_ix/cell_iy/cell_iz` tuple 不同的所有行得到；输出 `27` 行，无抽样、无随机种子。其用途是展示实际跨-cell edges，不能单独解释为何数量很少。

所有附件的最终 bundle SHA256、大小和 ZIP 内名称由交接目录中的 `manifest.json` 与 `ATTACHMENTS.md` 记录。

## 11. 给下一位讨论者的最短事实摘要

1. V14 clean fixed point `cd892503...` 已完整运行 5k-cell Pilot；10 个 operational stages 和 artifact closure 通过，但三项 scientific Gate 未通过，Formal 未授权。
2. Reach 第三轮 Jaccard/new-volume/boundary-change 为 `0.9272/0.0276/0.1506`；500/500 saturation cells 发现新 `>1 deg` candidate family；这分别说明当前 proxy 未达冻结收敛、candidate-family discovery 未饱和，但后者不等于 500 个新拓扑 branch。
3. 局部 16 charts 的 fresh cycle/multipath audit 通过，但各只含 5 probes；77,453 robust edges 中仅 27 条跨 cell，25k probes 中只有 5 个属于 primary section，6 个 overlaps 全部 non-stitchable，故 representation 被 blocked。
4. Budget arithmetic `N_min=30,050<=200k` 通过，但监督 rows、Student、spatial、trajectory、Formal 均为 0/未执行；不能以预算 Gate 代替 representation Gate。
5. 用户已确认的问题：审计三项 Gate 的科学含义和是否可直接放宽，区分定义/实现、预算/图连接与冗余 IK 表示限制，并给出含对照、指标、Gate、停止条件、预算优先级和 representation 转向证据的下一轮可执行方案。

---

# 核心判断

当前 V14 Pilot 的真正瓶颈已经非常明确：

$$
\boxed{
\text{不是单点 IK 求不出来，也不是 Student 不够强，}
\quad
\text{而是 canonical branch 无法从一个 cell 传播到相邻 cell。}
}
$$

现有实验中：

- (5000/5000) 个代表点都至少找到了一组有效 IK candidate；
- 高预算搜索还能在每个审计 cell 中发现大量不同的 (\beta) 解；
- 每个单 cell 内部的 5 个 probes 可以形成局部连续、闭合、可重复的小 chart；
- 但 (77{,}453) 条 robust product edges 中，只有 (27) 条跨 cell，只涉及 (25) 个相邻 cell pair；
- 最终 16 个 charts 每个恰好只包含 5 个 probes，实际上就是 16 个“单 cell chart”；
- 因而 (5000) 个 cell 中 (4989) 个被判为 `teacher_unresolved`，最终数据行数为 (0)。

所以当前主要矛盾是：

$$
\boxed{
\text{广域任务空间覆盖}
\quad\longleftrightarrow\quad
\text{跨 cell 的 canonical branch 连续传播}
}
$$

其中当前起决定作用的主要方面，是**跨 cell 图构造与 section 提取方法**。Reach proxy 尚未完全收敛、候选解仍会继续增加，都是真问题，但不是目前让 atlas 几乎完全坍缩成单 cell 的直接原因。

下一轮不应再直接重跑完整 5k Pilot，也不应简单放宽三个 Gate。应先实施一个规模较小但信息增益极高的：

# BACRA V14.1 — Cross-Cell Canonical Atlas Repair

它的目标不是立刻证明完整 (\Omega_{200})，而是先回答：

> 在相邻任务单元之间，当前 Teacher 是否能够沿同一 branch 连续传播，并形成真正跨越几十至几百个 cell 的局部 atlas？

只有这个问题解决，后续 50k、200k 数据集和 Student 才有意义。

---

# 一、当前实验已经直接证明了什么

## 1. 单点可达性不是当前主要瓶颈

普通预算下，(5000/5000) 个代表任务点都找到了至少一个 candidate，共运行了 (90{,}888) 次 candidate attempts。这说明在选出的 Pilot cells 上，Teacher 并没有普遍遭遇“目标点没有 IK”的问题。

因此，当前失败不能解释为：

> (\Omega_{200}) 中绝大部分点不可达。

更准确地说是：

> 点可以逐个到达，但当前算法没有把各点的逆解连接成跨空间连续的 section。

---

## 2. 局部 canonical consistency 已经成立

16 个已提取的小 chart 均通过了 fresh path、正反方向、fundamental cycle、multipath 和 repeat 审计。聚合指标约为：

$$
d_{\mathrm{path,P95}}=0.347^\circ,
$$

$$
d_{\mathrm{loop,P95}}=0.396^\circ,
$$

$$
d_{\mathrm{multipath,P95}}=0.407^\circ.
$$

这些指标都低于当前 (0.5^\circ) 的局部 Gate。

这说明当前方法不是完全错误。它已经能在非常小的邻域内建立稳定 canonical label。

---

## 3. 真正的断点发生在跨 cell 连接

我进一步检查了 ZIP 中的 `v14_workspace_protocol.py` 和 `v14_workspace_atlas_integration.py`，发现当前图结构具有两个关键特征。

### Cell 内部

每个 cell 有 1 个 representative 和 4 个 measure probes。代码将同一 cell 中的 5 个 probes 两两连接，形成完全图：

$$
K_5.
$$

因此每个 cell 天然有：

$$
\binom52=10
$$

条 task edges，并产生 6 个 fundamental cycles。

这与实验中每个小 chart 恰好：

- 5 个 probes；
- 10 条 selected task edges；
- 6 个 fundamental cycles；

完全吻合。

也就是说，当前“local atlas audit pass”主要证明的是：

> 同一 cell 内部的五个点可以构成稳定小 chart。

### Cell 之间

跨 cell 只连接两个 cell 的 representative probe，而且只做一次直接 predictor-corrector。若 continuation 得到的目标 (\beta) 与目标 cell 中**预先已经存在**的 candidate 相差超过：

$$
0.5^\circ,
$$

这条 edge 就会被拒绝。

即使 continuation 实际成功、residual 很低，只要该终点没有被原 candidate bank 预先采中，也不会被加入 target candidate bank。

于是当前流程实际上是：

$$
\text{先独立生成静态 candidate bank}
\rightarrow
\text{再要求邻居 continuation 恰好落入已有 cluster}.
$$

对于一个 (6\to3) 的冗余逆问题，这个要求非常苛刻。相邻点属于同一连续 branch，并不意味着两个独立多起点 IK 搜索恰好采到了彼此相差 (0.5^\circ) 以内的代表解。

按照当前 registry 的 (10{,}715) 个 cell 邻接关系计算，最终只有 25 个 cell pair 出现跨 cell robust connection，连接率约为：

$$
\frac{25}{10715}\approx0.23%.
$$

这已经足以解释为什么 atlas 没有生长起来。跨 cell robust edges 极少是完整数据直接支持的事实；Q17 材料本身也强调，该现象尚未被归因到具体实现机制。

---

## 4. Primary partition 的实现又进一步放大了失败

当前 `_primary_partition()` 对每个 task-space connected component 寻找：

> 能够覆盖该 component **全部节点**的单一 chart。

只有满足：

$$
V_{\mathrm{component}}
\subseteq
V_{\mathrm{chart}}
$$

的 chart 才能被用于该 component。

这与 multi-chart atlas 的原理并不相符。一个大的 task-space 连通区域，本来就应当允许：

$$
\Omega
======

\Omega_1\cup\Omega_2\cup\cdots\cup\Omega_K,
$$

由多张局部 chart 分片覆盖。它不应要求一张 chart 单独覆盖整个 task component。

当前 charts 只有单 cell 大小，自然不可能覆盖由大量 cell 构成的 task component，于是几乎全部被标为 `uncovered_component`。最终 primary partition 只有 `chart_06` 的 5 个 probes，另外 (24{,}995) 个 probes 全部未覆盖。

因此当前 representation failure 包含一个明确的实现层缺口：

$$
\boxed{
\text{当前 partition 规则要求“单 chart 覆盖整个 component”，}
}
$$

而真正需要的是：

$$
\boxed{
\text{多张局部 chart 对一个 component 做空间连贯的分片覆盖。}
}
$$

---

# 二、三个 Gate 应该分别如何处理

三个 Gate 不能统一解释，也不能统一放宽。

---

## 1. Reach convergence Gate：继续运行，但不应阻塞 atlas 修复 Pilot

三轮 Reach 指标为：

| Round | Weighted Jaccard | New volume | Boundary change |
| ----- | ---------------: | ---------: | --------------: |
| 1     |         (0.7843) |   (1.0000) |        (1.0000) |
| 2     |         (0.8928) |   (0.1049) |        (0.4478) |
| 3     |         (0.9272) |   (0.0276) |        (0.1506) |

它们显示了明确的收敛趋势，但尚未达到原始标准：

$$
J\ge0.95,\qquad
r_{\mathrm{new}}\le0.01,\qquad
r_{\mathrm{boundary}}\le0.02.
$$

同时当前 Reach lower/upper cells 为 (18{,}689/19{,}301)，二者的相对差距约为：

$$
\frac{19301-18689}{19301}\approx3.17%.
$$

说明工作空间总测度已经比 (15.1%) 的 boundary-change 数字更稳定；后者对边界 cell 的进出较敏感。现有材料明确把 Reach 解释为有限前向采样的经验代理，而不是连续可达域的数学证明。

### 建议

- 再追加 (1\sim2) 轮独立 scrambled Sobol A/B 采样；
- frontier probing 继续只作用于 A/B disagreement 和高 (x) tip；
- 暂时保留原 Gate，不把当前数值事后改判为通过；
- **V14.1 atlas 修复 Pilot 直接使用稳定 lower proxy：**

$$
\Omega_{\mathrm{pilot}}
=======================

\Omega_A\cap\Omega_B.
$$

因此 Reach sampling 可以后台并行运行，不再阻塞跨 cell atlas 方法验证。

### Reach 的后续 Formal 条件

优先继续使用原条件。如果 boundary-change 长期不低于 (2%)，但：

- lower/upper measure gap (\le3%)；
- weighted Jaccard (\ge0.95)；
- new volume (\le1%)；
- frontier new volume (\le1%)；

则可以在一个**新协议**中，把 raw boundary-cell change 替换为 lower/upper measure gap，而不是直接把 (0.02) 改成 (0.15)。

---

## 2. Branch saturation Gate：不是放宽，而是应当替换

当前 Gate 的逻辑是：

> 高预算搜索只要发现一组距普通预算候选超过 (1^\circ) 的 candidate family，就记作 `new_stable_branch=true`。

在正常满秩点附近：

$$
F:\mathbb R^6\rightarrow\mathbb R^3
$$

的固定目标逆像通常局部具有：

$$
6-3=3
$$

维自由度。

因此，在同一个连续 self-motion fiber 上，增加更多 seed 几乎必然会不断找到彼此相差超过 (1^\circ) 的解。要求：

$$
P(\text{new candidate family})\le2%
$$

并不是合理的“branch saturation”标准。

500/500 高预算 cell 都发现新 family，只能证明：

> 普通预算没有穷尽整个逆解集合。

它不能证明：

> 每个 cell 都出现了新的拓扑 branch。

Q17 文档也明确指出，当前 `new_stable_branch` 实际只是 candidate-family gap，而尚未区分 same-fiber 与 distinct-component。

### 新 Gate：Canonical Selection Stability

真正需要稳定的不是所有候选集合，而是最终被选中的 canonical label。

对同一批审计 cells，设置预算：

$$
B\in{4,8,16,32}.
$$

每个预算下完成：

1. candidate discovery；
2. same-target candidate component 分析；
3. 与邻居的 continuation connectivity；
4. canonical candidate 选择。

定义：

$$
\beta_B^\star(x)
================

\text{预算 }B\text{ 下被选中的 canonical label}.
$$

检查：

$$
d_{\mathrm{select}}(x)
======================

d_\beta
\left(
\beta_{16}^\star(x),
\beta_{32}^\star(x)
\right).
$$

建议探索性 Gate：

$$
d_{\mathrm{select,P95}}\le0.5^\circ,
$$

$$
d_{\mathrm{select,max}}\le1^\circ,
$$

$$
\text{selected component switch ratio}\le5%,
$$

$$
\text{outgoing cross-cell edge set change}\le5%.
$$

只要最终 canonical label、所属 component 和邻接结构稳定，即使增加预算后仍然发现更多 self-motion candidates，也不应阻止实验。

---

## 3. Representation Gate：不能直接放宽，必须先修复图和 partition

当前 representation failure 不是“80% coverage 阈值比结果略高”。

当前实际覆盖是：

- 5 个 primary probes；
- (24{,}995) 个 uncovered probes；
- 全 registry unresolved measure 约 (99.9873%)。

这与 (80%) 的目标相差四个数量级，不属于阈值微调问题。

因此不能把 representation coverage 从 (80%) 改为 (1%) 来继续 Formal。

但为了推进探索性实验，可以新增两级 Gate：

### Exploratory dataset Gate

允许从已解决的大连通区域生成数据并训练 probe Student：

$$
R_{\mathrm{resolved}}\ge30%,
$$

$$
R_{\mathrm{largest\ component}}\ge20%,
$$

$$
R_{\mathrm{unresolved}}\le70%.
$$

它只能支持：

> large connected atlas patch

不能称为完整 (\Omega_{200})。

### Formal Gate

保持更高标准：

$$
R_{\mathrm{labelable}}\ge60%,
$$

强成功目标：

$$
R_{\mathrm{labelable}}\ge80%,
$$

并且：

$$
R_{\mathrm{unresolved}}\le20%.
$$

---

# 三、下一轮实验的总体结构

建议新实验命名为：

```text
BACRA V14.1 — Cross-Cell Canonical Atlas Repair
```

新实验首先不跑完整 5k cells，而是在一组紧凑的空间 patches 上做机制消融。

总流程为：

$$
\boxed{
\begin{aligned}
&\text{局部连通 patch 选择}\
&\rightarrow\text{跨 cell edge 构造消融}\
&\rightarrow\text{动态 candidate 传播}\
&\rightarrow\text{piecewise chart partition}\
&\rightarrow\text{canonical selection stability}\
&\rightarrow\text{5k-cell rerun}\
&\rightarrow\text{50k exploratory dataset}\
&\rightarrow\text{probe Student}
\end{aligned}
}
$$

---

# 四、Phase 0：建立机制诊断 Patch 集

## 4.1 Patch 选择

从当前 lower Reach proxy 中选取 (12) 个连通 patches。

每个 patch 包含：

$$
4\times4\times4=64
$$

个相邻 10 mm cells。

总计：

$$
12\times64=768
$$

个 cells。

Patch 应覆盖：

- 低 (x)、中 (x)、高 (x)；
- interior；
- Reach boundary；
- tip；
- V13 retention 邻域；
- 低、中、高 (\kappa)；
- 当前有/无 cross-cell edge 的区域。

所有消融方法使用完全相同的 patch cells、task probes、初始 candidate bank 和随机种子。

---

# 五、Phase 1：跨 cell graph 构造消融

这是整个下一轮的最高优先级。

## E0：当前方法基线

- 每个 cell 内 5 probes 完全连接；
- 跨 cell 只连接 representative-to-representative；
- 一次直接 predictor-corrector；
- outcome 必须落入已有 target candidate (0.5^\circ) 范围；
- 不插入新 candidate。

它应复现当前极低的跨 cell edge rate。

---

## E1：Shared-face 多 probe 连接

对两个面邻接 cells，不再只比较 representative。

在共享面附近，为每一对 cells 选择：

- 最近的 (2\sim4) 对 probe；
- representative pair；
- capability medoid pair。

对每个 candidate pair 都尝试 continuation。

这用于检验当前失败是否主要由 task adjacency 过稀造成。

---

## E2：分段 continuation

对相邻任务点 (x_a,x_b)，不再一步完成：

$$
x_a\rightarrow x_b.
$$

将路径分段：

$$
x^{(r)} = x_a+
\frac{r}{M}(x_b-x_a),
$$

并要求：

$$
|x^{(r+1)}-x^{(r)}|\le5\text{ mm}.
$$

每一步都用上一构型作为 seed，并 exact correct。

这用于检验 10–20 mm 单步位移是否超过了局部 continuation 的有效半径。

---

## E3：Edge-induced candidate insertion

这是最关键的改动。

若从 source candidate 向 target point 的 continuation：

- solver 成功；
- FK residual (\le3)mm；
- actual bounds 内；

但与 target 现有 candidates 的距离全部大于 (0.5^\circ)，当前代码会拒绝该 edge。

V14.1 应改为：

1. 将 continuation endpoint 作为 target node 的新 candidate；
2. 按 (0.5^\circ) 聚类；
3. 标记 provenance：

```text
source = propagated_from_neighbor
```

1. 使用该新 candidate 向目标其他邻居继续传播。

即：

$$
\text{successful continuation}
\Rightarrow
\text{candidate propagation},
$$

而不是：

$$
\text{successful continuation}
\land
\text{pre-existing cluster match}
\Rightarrow
\text{edge}.
$$

最多运行 (2\sim3) 轮传播 wave，直到：

- 没有新 cell 被连接；
- 或达到 candidate budget。

每个 node 每个 lineage 最多保留 (8\sim12) 个候选，避免指数爆炸。

---

## E4：Frontier-only 高预算增广

仅对以下节点增加到 (16\sim32) starts：

- 当前 chart frontier；
- 无 cross-cell edge；
- 两个 propagation fronts 相遇；
- suspected multichart；
- 高 (\kappa)。

不应再给全部 5000 cells 统一增加到 32 starts。现有实验已经证明“候选数量不足”不是唯一问题；把预算集中在 frontier 更有效。

---

# 六、Phase 1 的评价指标

对每个消融方法报告：

## 1. Cell-pair connection rate

$$
R_E
===

\frac{
\text{有至少一条 robust candidate edge 的相邻 cell pairs}
}{
\text{被尝试的相邻 cell pairs}
}.
$$

当前基线约为 (0.23%)。

## 2. Connected-cell coverage

$$
R_C
===

\frac{
\text{至少连接到一个其他 cell 的 cells}
}{
\text{patch 总 cells}
}.
$$

## 3. Largest component ratio

$$
R_{\max}
========

\frac{
\text{最大跨 cell component 中的 cells}
}{
\text{patch 总 cells}
}.
$$

## 4. Chart 尺度

报告：

- chart cell count P50/P95/max；
- 单 cell chart 比例；
- 最大 chart 直径；
- 跨 (x)-bin 数量。

## 5. 一致性

继续要求：

$$
d_{\mathrm{path,P95}}\le0.5^\circ,
$$

$$
d_{\mathrm{loop,P95}}\le0.5^\circ,
$$

$$
d_{\mathrm{return,max}}\le1^\circ,
$$

$$
e_{\mathrm{FK,max}}\le3\text{ mm}.
$$

## 探索性通过标准

最低推进条件：

$$
R_E\ge5%
$$

且至少相对 E0 提升 (20\times)；

$$
R_{\max}\ge30%.
$$

强通过标准：

$$
R_E\ge25%,
$$

$$
R_{\max}\ge60%.
$$

如果 E3 相对 E2 出现数量级提升，可以确认当前主要瓶颈是：

> 静态 candidate bank 与 continuation endpoint 不匹配。

---

# 七、Phase 2：修复 Primary Partition

当前“一个 chart 覆盖整个 task component”的规则应被替换。

## 7.1 正确的 section 变量

对每个 task node (i)，选择一个 candidate：

$$
s_i\in{1,\ldots,K_i}.
$$

定义能量：

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
U_i(s_i)
========

C_{\mathrm{posture}}
+
C_{\mathrm{margin}}
+
C_{\mathrm{condition}},
$$

而：

$$
V_{ij}(s_i,s_j)
===============

\begin{cases}
d_W^2(\beta_{i,s_i},\beta_{j,s_j}),
&\text{存在 robust product edge},\
+\infty,
&\text{不存在 robust edge}.
\end{cases}
$$

## 7.2 Piecewise chart partition

一个大的 task component 可以由多张 chart 分片覆盖：

$$
\Omega_C
========

\Omega_{C,1}\cup\cdots\cup\Omega_{C,K}.
$$

使用：

- multi-source region growing；
- ICM/Potts graph labeling；
- 或最短路初始化加局部更新；

生成空间连贯的 `primary_chart_id`。

不得再要求：

$$
\exists k:\Omega_C\subseteq\Omega_k.
$$

## 7.3 Overlap 处理

### Stitchable

若同一物理点处：

$$
d_{\beta,\mathrm{P95}}\le0.5^\circ,
$$

可以合并或平滑切换。

### Non-stitchable

若：

$$
d_\beta>1^\circ,
$$

保留为 branch boundary，不强制平均。

在探索性静态数据中：

- 选择一侧作为 primary chart；
- 在边界设置 abstention band；
- alternative branch 仅保存到 expert/candidate table。

不能仅因为一个 task component 中出现 non-stitchable overlap，就把整个 component 判定为 stateful。

---

# 八、Phase 3：Canonical Selection Stability Audit

在原有 500 个 saturation cells 上重新运行。

## 8.1 Same-fiber connectivity

对同一 (x) 的候选 (\beta_a,\beta_b)，通过：

- Jacobian null-space 小步移动；
- 每步 exact correct 回到同一 (x)；
- joint bounds 检查；

判断二者能否通过连续 self-motion 路径连接。

由此区分：

- 同一个 fiber component 中的不同采样点；
- 可能不同的 inverse components。

不再把所有相差 (1^\circ) 的候选都称为新 branch。

## 8.2 预算序列

$$
B\in{4,8,16,32}.
$$

每个预算下记录：

- raw candidate count；
- estimated fiber-component count；
- canonical selected candidate；
- canonical selected component；
- outgoing robust cell edges；
- canonical cost。

## 8.3 新 Gate

$$
\operatorname{P95}
d_\beta
\left(
\beta^\star_{16},
\beta^\star_{32}
\right)
\le0.5^\circ,
$$

$$
\max
d_\beta
\left(
\beta^\star_{16},
\beta^\star_{32}
\right)
\le1^\circ,
$$

$$
\text{selected-component switch ratio}\le5%,
$$

$$
\text{outgoing-edge-set change ratio}\le5%.
$$

新 candidates 可以继续出现，只要 canonical section 的选择不再变化即可。

---

# 九、Phase 4：Reach 采样并行推进

在 cross-cell repair 实验运行期间，后台继续完成：

- Reach A round 4、5；
- Reach B round 4、5；
- frontier disagreements；
- high-(x) tip。

V14.1 Patch Pilot 只使用：

$$
\Omega_A\cap\Omega_B.
$$

因此 Reach 尚未完全收敛不阻止机制实验。

只有准备进入完整 Formal 时，才要求 Reach Gate 通过。

---

# 十、Phase 5：重新运行 5k-cell Pilot

只有 Patch Pilot 中至少一个方法达到最低推进条件，才重跑 5k。

推荐冻结：

- Shared-face probe edges；
- (5)mm 分段 continuation；
- edge-induced candidate insertion；
- 最多 3 个 propagation waves；
- frontier-only high-budget search；
- piecewise primary partition；
- canonical selection stability Gate。

## 5k exploratory Gate

建议先采用：

$$
R_{\mathrm{resolved}}\ge30%,
$$

$$
R_{\mathrm{largest\ primary\ region}}\ge20%,
$$

$$
R_{\mathrm{unresolved}}\le70%,
$$

$$
R_E\ge5%.
$$

所有保留的 charts 仍须通过 local path/cycle/multipath audit。

这不授权 Formal，但授权生成一个 (20k\sim50k) 行的探索性数据集。

---

# 十一、Phase 6：先训练探索性 Student

当前项目已经在局部 chart 上证明 Student 能拟合 Teacher，因此一旦得到真正跨 cell 的 atlas，就应尽快训练 probe Student，而不是再次因为全域 Gate 没达到 (80%) 而停住。

## 11.1 数据集

从 resolved regions 中生成：

$$
20{,}000\sim50{,}000
$$

行数据。

每个物理 (xyz) 在 primary 表中只保存一组：

$$
(x,\beta^\star,\mathrm{primary\ chart\ id}).
$$

alternative branches 单独保存到：

```text
expert_branch_dataset.parquet
```

## 11.2 模型对照

### S0：全局静态 MLP

$$
xyz\rightarrow\beta_6.
$$

只训练最大 primary chart 或经过空间 partition 后的 primary labels。

### S1：xyz-only router + experts

$$
r(x)\rightarrow chart,
$$

$$
g_k(x)\rightarrow\beta.
$$

Router 目标来自固定 primary partition，而不是用 FK residual在不同有效 branch 之间选择。

### S2：Student + 1–2 次 DLS

用于判断 Student 是否已经进入正确局部 basin。

## 11.3 暂不直接训练 stateful 模型

目前 6 个 non-stitchable overlaps 都来自单 cell 大小的小 charts；它们不足以证明整个工作空间需要历史状态。

只有在跨 cell atlas 修复后仍观察到：

1. 相同 (xyz) 存在稳定的 non-stitchable branches；
2. 不同路径到达同一点得到相差 (>1^\circ) 的 (\beta)；
3. static primary partition 在轨迹中产生 branch jump；
4. 无法通过空间连贯 partition 和 abstention 消除；

才转向：

$$
(x_t,\beta_{t-1})\rightarrow\Delta\beta_t.
$$

---

# 十二、Inverse Representation 的明确转向条件

## 保留静态 (xyz\to\beta_6)

当：

- 一个 coherent primary section 覆盖主要 resolved region；
- graph cycles 和 multipath 一致；
- chart boundary 可 stitch 或可通过小 abstention band 隔离；
- Student 输出沿轨迹连续。

## 使用 xyz-only router + experts

当：

- 有多张局部 chart；
- chart domains 在任务空间中可以分区；
- 每个 (xyz) 有确定的 primary chart；
- 不依赖上一构型也能稳定路由。

## 使用 stateful inverse

当：

$$
\beta_t
\text{ 的正确选择依赖 }
\beta_{t-1},
$$

并且这种依赖在多路径实验中稳定复现。

形式为：

$$
(x_t,\beta_{t-1})\rightarrow\Delta\beta_t.
$$

## 使用 branch token / multiple candidates

当同一个 (xyz) 上确实存在多个稳定、非 stitchable 且任务上都需要保留的 branch：

$$
(x,branch_id)\rightarrow\beta
$$

或：

$$
x\rightarrow{\beta^{(1)},\ldots,\beta^{(K)}}.
$$

## 使用 abstention

在：

- Reach proxy 不确定；
- teacher unresolved；
- chart non-stitchable boundary；
- router 低置信度；
- 所有 expert FK residual 过大；

的区域拒绝输出。

---

# 十三、Formal 准入条件

V14.1 完整 5k Pilot 只有满足以下条件后才应授权 Formal。

## 1. Reach

原 Reach Gate通过，或新协议预先注册的 lower/upper measure Gate通过。

## 2. Canonical selection saturation

使用新的 selection stability，不再使用 raw candidate-family exhaustion。

## 3. Atlas coverage

最低：

$$
R_{\mathrm{labelable}}\ge60%,
$$

$$
R_{\mathrm{unresolved}}\le20%.
$$

强成功：

$$
R_{\mathrm{labelable}}\ge80%.
$$

## 4. Cross-cell connectivity

- 跨 cell robust edges 不再只占极少数；
- 最大 connected chart/primary region 跨越多个 x-bins；
- 单 cell charts 不应占主要比例；
- 所有 retained charts 通过 cycle/multipath。

## 5. Representation

必须明确冻结为：

- global static；
- router-expert；
- stateful；
- 或 candidate-set。

不能再由一个 `blocked/stateful` 布尔结果代替完整表示决策。

## 6. Budget

使用真实 labelable cells 重新计算：

$$
N_{\min}.
$$

当前 (30{,}050) 的预算通过没有意义，因为它只来自 11 个 labelable cells。

---

# 十四、计算预算优先级

建议按以下顺序投入资源：

1. **768-cell Patch 的 E0–E4 cross-cell 消融**；
2. **修复 `_primary_partition()`**；
3. **canonical selection stability audit**；
4. **Reach round 4/5 后台并行**；
5. **修复后重新运行 5k Pilot**；
6. **生成 20k–50k 探索性数据并训练 Student**；
7. **达到 Formal 准入条件后才运行完整 registry 和 200k 数据**。

不建议直接再跑一次接近 6 小时的完整 5k Pilot，因为当前代码结构不变时，结果大概率仍然是大量单 cell charts。

---

# 十五、可直接交给 Codex 的任务定义

> Implement BACRA V14.1 as a cross-cell canonical-atlas repair experiment. Preserve the existing V14 Pilot as the E0 baseline. First select twelve deterministic connected patches of sixty-four adjacent 10 mm cells spanning interior, reach-boundary, tip, retention, conditioning, and x-bin strata. On exactly the same patches compare the current representative-to-representative graph against shared-face multi-probe edges, continuation subdivided into at most 5 mm Cartesian steps, dynamic insertion of successful continuation endpoints into the target candidate bank, and frontier-only high-budget candidate enrichment. A successful continuation that satisfies FK residual and actual bounds must no longer be rejected solely because no pre-existing target candidate lies within 0.5 degrees; it should become a new lineage candidate and be propagated for at most three waves.
>
> Replace the current primary-partition rule that requires one chart to cover an entire task-space component. Extract one candidate per task node and allow multiple spatially connected chart regions to cover one task component. Use graph-constrained region growing or discrete labeling with infinite cost where no robust product edge exists. Merge only same-target stitchable overlaps, retain non-stitchable overlaps as explicit branch boundaries, and allow abstention bands rather than declaring an entire component stateful.
>
> Replace the raw new-candidate-family branch-saturation Gate with a canonical-selection-stability audit. For the same audit cells compare budgets 4, 8, 16, and 32; estimate same-fiber connectivity through null-space predictor-corrector paths; measure the stability of the selected canonical beta, selected component, and outgoing cross-cell edge set. New distant candidates may continue to appear, but the selected section must stabilize.
>
> Report cross-cell neighbor-pair connection rate, connected-cell ratio, largest component ratio, chart cell-count distribution, unresolved ratio, candidate-propagation acceptance, cycle/multipath consistency, residual, and runtime for every ablation. Authorize a repaired 5k-cell Pilot only after cross-cell connection increases by at least twenty times over the current baseline and the largest patch component covers at least thirty percent of patch cells while retaining the existing local consistency thresholds.
>
> Continue independent Reach A/B sampling in parallel, but use the stable intersection proxy for the repair Pilot. After a repaired 5k run resolves at least thirty percent of selected cells in sizeable connected chart regions, materialize a 20k–50k exploratory dataset and train the existing global bounded MLP, an xyz-only router with local experts, and a Student-plus-one/two-step-DLS baseline. Do not select stateful representation solely from the six current one-cell overlaps; require reproducible non-stitchable path dependence after cross-cell atlas repair.
>
> Formal generation remains blocked until Reach convergence, canonical-selection stability, cross-cell atlas coverage, representation choice, and the real labelable-cell budget all pass their registered criteria.

---

# 最终结论

当前实验不是因为 Gate “太严格”而停住，也不是因为连续体机器人天然无法形成广域数据集而停住。

当前主要失败链条是：

$$
\boxed{
\begin{aligned}
&\text{跨 cell 只连接 representative}\
&+\text{一步 continuation}\
&+\text{必须匹配静态 target candidate}\
&+\text{不插入成功 propagation endpoint}\
&+\text{primary partition 要求单 chart 覆盖整个 component}\
&\Downarrow\
&\text{atlas 被限制成单 cell charts}\
&\Downarrow\
&\text{99.987% domain 被标为 unresolved}\
&\Downarrow\
&\text{representation blocked，数据和 Student 均为 0}
\end{aligned}
}
$$

因此，下一步推进实验的最优先事项是：

$$
\boxed{
\text{修复跨 cell branch propagation 和 piecewise chart partition}
}
$$

Reach 继续补采；branch saturation 改为 canonical selection stability；stateful representation 暂不提前决定。

只要跨 cell atlas 能从当前的单 cell 尺度扩展到几十、几百个连通 cell，就应立即生成一个探索性区域数据集并训练 Student。这样才能尽快判断整套方法是否真正具备从局部成功走向工作空间泛化的能力，而不是继续在三个不合适的总 Gate 前反复停住。

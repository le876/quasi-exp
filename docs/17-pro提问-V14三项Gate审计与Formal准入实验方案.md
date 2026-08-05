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
|---|---|---|---|
| 实现/配置 | 新增 V14 workspace Reach、registry、candidate bank、atlas、representation、dataset 和 Student modules，共 28 个提交文件、13,985 行新增 | `v14_context.md`、`v14_config.yaml`、`v14_runner.py`、`v14_source_manifest.json` | implementation fixed point |
| 实验 | Smoke 后实际完成 5k-cell Pilot；10 个 operational stages 完成，96 个 artifacts 封存 | `v14_pilot_log.txt`、`v14_operational_gate.json`、`v14_artifact_manifest.json` | exploratory scientific Pilot with sealed provenance |
| 结论修正 | 预算算术可行不等于可生成数据；三项科学 Gate 未通过使 representation、dataset、Student、Formal fail closed | `v14_pilot_gate.json`、`v14_budget_report.json`、`v14_dataset_report.json`、`v14_student_report.json` | current Pilot result |
| 诊断 | 全表聚合发现 robust edges 几乎都在同一 cell 内；生成 27 条跨-cell edge 完整清单 | `q17_pilot_gate_audit.json`、`q17_cross_cell_robust_edges.csv` | post-hoc diagnostic-only |

## 4. 实验方法与复现口径

### 4.1 代码与工作树 fixed point

| 项目 | 值 |
|---|---|
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
|---|---|---|
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
|---|---|
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
|---|---|---|
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
|---|---|---:|---|---|---|
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
|---|---|---|---|
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
|---|---|---|---|---|
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
|---|---|---:|---|---|---|
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

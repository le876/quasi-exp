---
question_id: Q21
question_number: 21
question_confirmed_by_user: true
question_confirmation_summary: "基于 retry7 的 12-patch 全通过和 512-cell meso 结果，判断当前结果成因，并基于项目目标与该阶段实验目标分析下一轮实验如何推进；证据包包含最具信息价值的关键数据集。"
date: "2026-08-19"
status: ready-to-send
project: "quasi_exp / BACRA"
evidence_cutoff: "2026-08-19 23:02:00 CST (+08:00)"
source_snapshot: "codex/bacra-v14-2r-retry7-gauge-lock @ 33862ad483059ad8bbaa22678e7c434d201c2766, clean worktree"
---

# GPT-5 Pro 第二十一次交接：retry7 中尺度 Gate 失败成因与下一轮实验推进

## 0. 经用户确认的 GPT-5 Pro 任务

用户已经确认的任务原意：

> 基于 retry7 的 12-patch 全通过和 512-cell meso 结果，请判断当前结果的成因；结合项目最终实验目标和本阶段实验目标，分析下一轮应该如何推进实验。证据包应包含关键且最具信息价值的数据集。

确认范围：对当前结果做原因分析，并给出基于既定实验目标的下一轮实验推进方案。本文没有替用户预设应保留或放弃 `xyz-only`、应放宽 Gate、应采用何种新表示，也没有预设候选路线优先级。

## 1. 当前有效事实与最短摘要

### 1.1 证据截止时间

本文只使用截至 2026-08-19 23:02:00 CST 已存在的 retry7 retry4 代码、配置和 artifacts。当前运行已经结束，不存在仍需等待的 canonical 实验进程。

### 1.2 当前有效事实

1. retry7 retry4 在 clean fixed point `33862ad483059ad8bbaa22678e7c434d201c2766` 上 operationally 完成；最终 summary 为 `operational_completion=true`、`scientific_gate_pass=false`、`repaired_5k_authorized=false`。V14.3 repaired 5k/Student 未启动。
2. lineage、K/R ablation、holonomy diagnostics、gauge kernel selection、patch_07 repair、four-patch Gate 和 fresh 12-patch confirmation 均通过；12-patch 结果为 development `8/8`、sealed confirmation `4/4`。
3. Reach Round 7 和 Round 8 的单轮 convergence checks 均通过，但 lower/upper measure gap 为 `0.06151960784313725`，超过注册上限 `0.05`，因此 Formal Reach 未获授权。
4. 512-parent connected meso bridge 使用 2560 task probes、6038 task edges。baseline 8 roots 和 leave-one-non-anchor-root-out 7 roots 均未选择 canonical anchor component，最终 `coverage_ratio=0`、`abstention_ratio=1`、primary certificate 为空。
5. meso 不是“没有 IK candidate”：baseline 生长得到 3939 个 post-pruning hypotheses，8 张 rooted charts 中有 4 张通过 chart qualification；其 support fractions 为 `0.246875`、`0.059375`、`0.371484` 和 `0.339062`。
6. 4 张 qualified charts 的六组 pairwise overlaps 均证据充分但被判为 `nonstitchable`；overlap beta P95 gap 范围为约 `4.066647°` 至 `9.704019°`，明显高于 stitch P95 上限 `0.5°`。
7. 当前 canonical policy 要求注册 anchor 所在 singleton/stitchable component 同时满足 coverage `>=0.90` 和 coherent measure `>=0.60`。没有 component 达标时，代码返回 `no_registered_anchor_component_qualified`、令 selected component 为空，并按 fail-closed 规则把所有 task probes 标为 abstained。

### 1.3 关键边界

1. meso 的 `0% coverage / 100% abstention` 是当前 all-or-nothing canonical-component materialization 规则下的部署结果，不等价于 2560 个 probes 全部不可达或全部 IK solver 失败。
2. 四张 qualified charts 的非 stitchable overlap 是已观察事实，但现有结果尚不能区分其主要原因是 candidate/search budget、长距离 chart growth、canonical anchor 选择口径、task graph 尺度，还是冗余 IK 的实际表示限制。
3. 当前没有 paired same-`xyz` history experiment、stateful Student、branch-token Student 或 multiple-candidate Student 结果，不能据此声称历史状态是必要输入。

## 2. 项目目标与本轮边界

### 2.1 用户给定的项目目标

项目最终目标是在约 200,000 条监督数据预算下覆盖 Omega200 工作空间，构造可审计的 Teacher/canonical inverse 数据，并训练可部署的 Student，使其从末端位置预测六维内部参数，同时保持工作空间覆盖、标签一致性和 FK 精度。

### 2.2 本阶段实验目标

retry7 本阶段不是直接生成 200,000 条数据，而是验证从局部机制证据进入 repaired 5k Pilot 的前置条件：

1. 在 64-parent local patches 上验证固定 canonical anchor、K/R stability、gauge repair、node-induced certificate 和 root-budget stability；
2. 通过独立 Reach replicas 与 frontier probing 约束 Omega200 empirical proxy；
3. 用一个 connected 512-parent meso bridge 检验局部 certified section 是否能跨更长距离、更多节点和非局部 cycle 扩展；
4. 只有这些 Gate 通过，才授权 repaired 5k，再进入数据 materialization 和 Student。

### 2.3 当前实际覆盖范围

- 已执行 retry7 local mechanism、12-patch confirmation、Reach Round 8 和 512-cell meso bridge。
- meso baseline 与 root-dropout 均使用相同 canonical anchor registry 的冻结前缀语义。
- 当前算法为 `anchor_locked_gauge_transport`，使用 rooted charts、chart qualification、stitchability、canonical anchor component selection、局部 abstention 和 fresh primary certificate。

### 2.4 当前证据未覆盖的内容

- repaired 5k Pilot、20k–50k Student、约 200k 数据 materialization 和 Formal 均未执行；
- 没有对 meso 失败做看过结果后的阈值放宽；
- 没有在本轮对 alternative anchor、不同 meso domain decomposition、stateful、branch token 或 multiple candidates 做新实验；
- 没有证明一张覆盖 90% 的无状态 global section 在数学上存在或不存在。

## 3. 相对上次材料新增的事实

| 类型 | 已发生的变化 | 实际证据 | 证据等级 |
| --- | --- | --- | --- |
| 实现/配置 | retry7 使用固定 canonical anchor、analytic endpoint kinematics、显式 least-squares Jacobian、48 logical shards/12 workers、screening-first 和 fail-closed empty primary schema | `retry7_config.yaml`、`canonical_gauge.py`、`section_atlas_repair.py` | 当前 fixed point 实现 |
| 实验 | patch_07 repair、four-patch Gate 和 12-patch confirmation 已完成并通过 | `twelve_patch_confirmation_gate.json`、`retry7_scientific_report.json` | 注册协议；部分上游 stage 为 hash-verified sealed evidence reuse |
| 实验 | Reach Round 8 operationally 完成，但 lower/upper gap 未通过 Formal Gate | `reach_round8_gate.json` | 注册 Reach Gate |
| 实验 | 新增 connected 512-parent meso bridge baseline/root-dropout，二者均 operationally 完成但 Gate 失败 | `meso_bridge_gate.json` 及 meso Parquet attachments | 注册 meso Gate；fresh retry4 execution |
| 结论修正 | 先前 empty-primary serialization/merge 实现错误已修复；本轮不再异常退出，而是产生完整 fail-closed scientific result | `source_fixed_point_retry7.json`、`retry7_artifact_manifest.json`、最终 summary | 当前 fixed point operational fact |

## 4. 实验方法与复现口径

### 4.1 代码与工作树 fixed point

| 项目 | 值 |
| --- | --- |
| 项目根 | `/mnt/ML_projects/quasi_exp` |
| 分支 | `codex/bacra-v14-2r-retry7-gauge-lock` |
| HEAD | `33862ad483059ad8bbaa22678e7c434d201c2766` |
| worktree | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v14-2r-retry7-gauge-lock` |
| dirty 状态 | clean；`git status --porcelain=v1` 无输出 |
| config SHA256 | `f3ce4e3fd5f01a6f520907265705a96607029d78b5e94f6ff815176f7bcfbea1` |
| runtime SHA256 | `83ce7d8efc457a42c75feb5f54effc34b0033756ff0315f6fb0c8ba49c77b10d` |
| 仅 checkout HEAD 是否足以复现 | 代码充分；正式复现仍需要 `source_fixed_point_retry7.json` 中记录的 upstream artifact closures 和同一 runtime |

### 4.2 主要方法与参数

| 参数 | 实际值 | 来源 |
| --- | ---: | --- |
| registered method | `anchor_locked_gauge_transport` | `retry7_config.yaml` |
| root count | 8；root-dropout 为 7 | `retry7_config.yaml`、meso reports |
| maximum growth waves | 16 | `retry7_config.yaml` |
| beam width | 配置 4；当前 report 的 selected field 为 1 | config/report；不将二者混写为同一对象 |
| FK residual max | 3 mm | `retry7_config.yaml` |
| geometry P95/max | 0.5° / 1.0° | `retry7_config.yaml` |
| repeat P95 max | 0.2° | `retry7_config.yaml` |
| repeats | 每方向 3 次；forward/reverse | `retry7_config.yaml`、audit datasets |
| retry tiers | R0 5 mm/100；R1 2.5 mm/200；R2 1.25 mm/400 + SLSQP | `retry7_config.yaml` |
| chart minimum support | max(8 nodes, 2% nodes)，spread >=10 mm | `retry7_config.yaml` |
| stitch threshold | overlap P95 <=0.5°，max <=1.0° | `retry7_config.yaml` |
| canonical anchor component | coverage >=0.90，coherent measure >=0.60，anchor required | `retry7_config.yaml`、`canonical_gauge.py` |
| meso scale | 512 parents，2560 task probes，6038 edges | `meso_bridge_gate.json`、meso input datasets |
| numerical execution | analytic endpoint kinematics + explicit least-squares Jacobian | `retry7_config.yaml` |
| parallelism | 48 logical shards，最多 12 个单线程 workers，cost-balanced LPT | `retry7_config.yaml` |

### 4.3 数据来源与划分

| 项目 | 实际口径 |
| --- | --- |
| 12-patch development | `patch_00`–`patch_07`，8 patches |
| diagnostic-only | `patch_09`，不计入 sealed confirmation |
| sealed confirmation | `patch_08`、`patch_10`、`patch_11`、replacement `patch_12` |
| meso | 一个 connected 512-parent macro-patch；每 parent 5 probes，共 2560 nodes |
| meso input | `meso_input_task_nodes.parquet`、`meso_input_task_edges.parquet`、`meso_input_assignments.parquet`、`meso_input_candidate_clusters.parquet` |
| roots | canonical anchor first；R5/R8 prefix 与 leave-one-non-anchor-root-out 语义冻结 |
| Reach replicas | 独立 seed lineages；Round 8 seeds A/B 为 `20260865/20260866`，每 replica Sobol power 19 |
| Student split | 未执行，不适用于当前阶段 |

### 4.4 baseline、评价和 Gate

| 项目 | 实际定义 | 是否正式 |
| --- | --- | --- |
| local baseline | 64-parent patches 上的 registered anchor/gauge method | retry7 注册协议 |
| meso baseline | `gauge_K1_R8`，8 roots | retry7 注册 meso Gate |
| meso control | `gauge_meso_root_dropout`，leave one non-anchor root out | retry7 注册 meso Gate |
| local Gate | coverage >=0.90、coherent >=0.60、geometry/repeat/solver/certificate/edge/cycle closure | retry7 注册协议 |
| confirmation Gate | development >=6/8、confirmation >=3/4 | retry7 注册协议；实际 8/8 与 4/4 |
| Reach Gate | 最近两轮检查均通过且 lower/upper gap <=0.05 | retry7 注册协议 |
| meso Gate | exact scale、coverage、coherent region、node-induced certificate、long path、nonlocal cycles、root-dropout stability、resource bound | retry7 注册协议 |

### 4.5 复现入口与产物

实际 pipeline：

```text
/mnt/ML_projects/quasi_exp/.worktrees/bacra-v14-2r-retry7-gauge-lock/scripts/pipelines/run_bacra_v14_2r_retry7_then_v14_3.sh
```

本轮 retry7 output root：

```text
/mnt/ML_projects/quasi_exp/runs/bacra_v14_2r_stitched_atlas_retry7_abstention_retry4
```

V14.3 output 未生成，因为 `repaired_5k_authorized=false`。

## 5. 实验结果

| Fact ID | 已观察结果 | 数值 | 证据等级 | 实际附件/JSON key | 适用边界 |
| --- | --- | ---: | --- | --- | --- |
| F1 | retry7 operationally 完成，但最终科学授权失败 | operational `true`；scientific `false` | current final summary | `retry7_summary_gate.json` | 决定 V14.3 是否启动 |
| F2 | local/confirmation 路线在 12 patches 上通过 | development `8/8`；confirmation `4/4` | sealed confirmation | `twelve_patch_confirmation_gate.json` | 64-parent local patch 尺度 |
| F3 | Reach 最近两轮检查通过，但 lower/upper gap 超阈值 | gap `0.0615196` > `0.05` | registered Reach Gate | `reach_round8_gate.json` | Formal Reach proxy；不等于 meso 失败原因 |
| F4 | meso input scale | 512 parents；2560 nodes；6038 edges | registered meso input | `meso_bridge_gate.json`、`meso_input_*.parquet` | 一个 connected macro-patch |
| F5 | baseline chart growth 产生大量候选 | 4540 proposals；3939 post-pruning hypotheses | fresh meso execution | `meso_baseline_report.json`、`meso_baseline_section_hypotheses.parquet` | 不能解释为“没有 candidate” |
| F6 | baseline 有 4 张 qualified charts | support fractions `0.246875/0.059375/0.371484/0.339062` | fresh meso execution | `meso_baseline_chart_qualification.parquet` | chart-level qualification，不是 primary deployment coverage |
| F7 | original anchor chart 未 qualified | support `0.252734`；persistent numerical count `3` | fresh meso execution | `meso_baseline_chart_qualification.parquet`、audit schedules/executions | 只说明该 chart 在当前 audit 下未通过 |
| F8 | 另一未通过 chart 的 persistent numerical count | `15` | fresh meso execution | 同上 | 不代表全部 charts solver 失败 |
| F9 | 六对 qualified chart overlaps 全部 nonstitchable | overlap P95 gap `4.066647°`–`9.704019°` | fresh meso execution | `meso_baseline_chart_stitchability.parquet` | 当前 stitch 定义与候选 fields |
| F10 | canonical anchor component 未选择 | selected `false`；fallback `no_registered_anchor_component_qualified` | registered policy result | `meso_baseline_report.json` | coverage/coherent threshold 的 all-or-nothing selection |
| F11 | primary atlas 全域 abstain且没有 primary audits | coverage `0`；abstention `1`；continuation execution `0` | registered fail-closed result | `meso_baseline_primary_atlas.parquet`、`meso_bridge_gate.json` | 不是 2560 次 solver 失败统计 |
| F12 | root-dropout 复现同类失败 | 3930 hypotheses；4 qualified charts；coverage `0` | registered control | `meso_dropout_*` | 非 anchor root 减一对照 |
| F13 | meso 资源上限通过 | `8.2970 s/parent` <= `120 s/parent` | registered meso Gate | `meso_bridge_gate.json` | 说明此次失败不是 resource Gate |

### 5.1 通过项

- 12-patch confirmation 的 development 和 sealed confirmation 均通过，且 confirmation 未用于调阈值。
- meso exact connected scale 和 resource scaling checks 通过。
- meso 中 4 张局部 charts 分别通过 geometry、solver、repeat 和 certificate qualification。
- root budget filled；baseline 8 roots、dropout 7 roots 均完成。

### 5.2 失败与反例

- Formal Reach 尚未通过，唯一直接超限项是 lower/upper measure gap `6.15196% > 5%`；最近两轮各自通过。
- meso 的 primary coverage、largest coherent region、node-induced certificate、long-path geometry、nonlocal-cycle nonvacuity 和 root-dropout stability checks 均失败。
- local 12-patch 的全域 singleton anchor section 没有在 connected 512-parent domain 中复现。
- qualified charts 在大量共同 nodes 上有 `4°–10°` 量级的 beta gap，因此不能按当前 stitch Gate 合并为同一连续 section。

## 6. 证据等级、失败结果与未知事项

### 6.1 注册协议内的事实

- retry7 漏斗、阈值和扩大顺序由 `retry7_protocol.md` 和 `retry7_config.yaml` 冻结。
- 12-patch confirmation Gate 通过；meso Gate 和 Reach Formal Gate 失败；最终 repaired 5k authorization 为 false。
- 当前 meso primary atlas 的全域 abstention 是 registered policy 的实际输出。

### 6.2 证据身份边界

| 内容 | 等级 | 能支持什么 | 不能支持什么 |
| --- | --- | --- | --- |
| lineage/KR/holonomy/gauge/patch repair/four-patch/confirmation | hash-verified sealed evidence reuse | 当前 retry7 summary 的上游闭合事实 | 不能冒充 retry4 新 fresh execution |
| Reach Round 8 | registered Reach evidence；hash-verified reuse | 当前 Formal Reach authorization 状态 | 不能证明解析 Reach 集合完备 |
| meso baseline/root-dropout | fresh registered meso experiment | 中尺度 canonical-component、chart 和 audit 结果 | 不能外推整个 Omega200 或证明 statefulness 必要 |
| GPT-5 Pro 后续回答 | 待生成建议 | 可用于设计下一轮 | 不能作为已经执行的实验事实 |

### 6.3 已发现的实现或度量边界

- `select_anchor_locked_component()` 先要求一个注册 anchor component 达到 90% coverage/60% coherent measure；失败时 selected component 为空。
- primary materialization 发生在 component 选择之后，因此 qualified singleton coverage `37.15%` 不会作为 partial primary 输出，而会在最终 deployable field 中坍缩成 `0%`。
- 最终 report 的 `solver_gate=false` 与 `continuation_execution_count=0` 同时出现；这里是空 primary certificate 的结果，不能解释为全部 meso nodes 的 continuation solver 均失败。
- `retry7_scientific_report.json` 内的 `scientific_gate=true` 表示上游 mechanism 漏斗曾授权进入 meso；最终授权事实以 `retry7_summary_gate.json` 的 `scientific_gate_pass=false` 为准。

### 6.4 尚未知或尚未执行

- 最大 qualified singleton 仅覆盖 37.15% 的原因中，search/growth budget、task-graph尺度、anchor choice、domain topology 和 branch structure 各占多少；
- 若固定另一 canonical root、局部延伸当前 anchor、改变 meso domain partition 或只在边界 abstain，能否得到显著更大的 certified component；
- qualified chart union 的空间连通形态、边界物理宽度和可分区性是否足以支持无状态 spatial routing；
- same-`xyz` 的不同有效 branch 是否需要历史状态才能选择；
- 增加 Reach 采样或 frontier budget 后 lower/upper gap 是否会降到 5% 以下；
- repaired 5k 和 Student 的真实效果。

### 6.5 当前不能声称

- 不能声称 2560 meso probes 均不可达或均没有 IK；
- 不能声称 solver 在 meso 全域失败；
- 不能声称 `xyz-only -> beta6` 已被数学上否定；
- 不能声称仅放宽 90% Gate 就能得到连续、可部署的 canonical inverse；
- 不能把四张 nonstitchable charts 逐点混合为单值监督标签；
- 不能授权 repaired 5k、Student 或 Formal。

## 7. 决策所需的客观对照

本节只比较已经执行的输入和结果，不给出路线推荐。

| 对照对象 | 已执行输入/方法 | 已观察结果 | 证据等级 | 已知限制 |
| --- | --- | --- | --- | --- |
| 12 local patches | 每 patch 64 parents；registered anchor/gauge；R5/R8 stress | development 8/8、confirmation 4/4；local primary coverage 通过 | sealed confirmation | 不能验证长距离传播和大环 |
| meso baseline | 512 connected parents；8 roots | 4 qualified charts；largest qualified singleton 37.15%；最终 primary 0% | fresh registered meso | all-or-nothing component threshold 位于 materialization 前 |
| meso root-dropout | 同域；移除一个 non-anchor root | 4 qualified charts；最终 primary 0% | fresh registered control | primary 为空，无法计算普通 beta-field stability |
| Reach Round 7 | 独立 replicas/frontier | weighted Jaccard `0.968137`；单轮 pass | registered Reach evidence | lower/upper aggregate 尚未达标 |
| Reach Round 8 | 新独立 replicas；128 frontier probes | weighted Jaccard `0.969683`；0 个新 frontier cells；单轮 pass | registered Reach evidence | lower/upper gap仍为 `0.06152` |
| qualified chart overlaps | 六组 evidence-sufficient overlaps | beta P95 `4.07°–9.70°`；全部 nonstitchable | fresh meso entity evidence | 尚未测试其他 canonical representation |

## 8. 经用户确认的问题

1. 基于 retry7 的 12-patch 全通过和 512-cell meso 结果，请判断当前结果的成因；结合项目最终实验目标和本阶段实验目标，分析下一轮应该如何推进实验。

## 9. 经用户确认的回答要求

请返回一份基于当前真实代码与 artifacts 的原因分析和下一轮实验推进方案。判断应围绕项目最终目标和当前阶段目标，不把尚未执行的路线写成已验证事实。

## 10. 附件清单与复现信息

GPT-5 Pro 实际可见的是“上传文件名”列；原始 SSH 路径只用于 provenance。

唯一上传 ZIP 的成员数（`QUESTION.md + 实际证据附件`）：34。网页端只上传 `Q21_GPT5Pro_upload.zip`。

| 上传文件名 | 原始 SSH 路径 | 大小 | 证据等级 | 支持事实 | 上传理由 |
| --- | --- | ---: | --- | --- | --- |
| `source_fixed_point_retry7.json` | `runs/bacra_v14_2r_stitched_atlas_retry7_abstention_retry4/00_inventory/source_fixed_point.json` | 1,065 B | provenance | F1 | 绑定 source/config/runtime/upstream closures |
| `retry7_summary_gate.json` | `runs/.../10_summary/retry7_summary_gate.json` | 183 B | final summary | F1 | 最终授权事实 |
| `retry7_scientific_report.json` | `runs/.../10_summary/retry7_scientific_report.json` | 125,876 B | current summary | F1–F13 | 汇总所有阶段并保留事实边界 |
| `retry7_artifact_manifest.json` | `runs/.../10_summary/retry7_artifact_manifest.json` | 403,191 B | provenance | F1 | 完整 artifact closure 索引，不代替具体数据 |
| `reach_round8_gate.json` | `runs/.../07_reach_round8/gate.json` | 2,454 B | registered Gate | F3 | Reach Round 6–8 指标与失败项 |
| `twelve_patch_confirmation_gate.json` | `runs/.../08_twelve_patch_confirmation/gate.json` | 83,398 B | sealed confirmation | F2 | 12 patches 逐 patch 结果与统计 |
| `meso_bridge_gate.json` | `runs/.../09_meso_bridge/gate.json` | 7,442 B | registered meso Gate | F4–F13 | baseline/dropout 和全部 meso checks |
| `meso_input_task_nodes.parquet` | `runs/.../09_meso_bridge/input/meso_task_nodes.parquet` | 110,048 B | raw meso input | F4 | 2560 probes 的坐标、cell 和邻接输入 |
| `meso_input_task_edges.parquet` | `runs/.../09_meso_bridge/input/meso_task_edges.parquet` | 24,593 B | raw meso input | F4 | 6038 task edges |
| `meso_input_assignments.parquet` | `runs/.../09_meso_bridge/input/meso_assignments.parquet` | 10,980 B | raw meso input | F4 | 512-parent macro-patch assignments |
| `meso_input_candidate_clusters.parquet` | `runs/.../09_meso_bridge/input/meso_candidate_clusters.parquet` | 810,272 B | raw meso input | F4–F5 | meso candidate distribution和 branch clusters |
| `meso_baseline_report.json` | `runs/.../09_meso_bridge/meso_512/gauge_K1_R8/report.json` | 3,268 B | fresh meso result | F5–F13 | baseline 紧凑结果 |
| `meso_baseline_root_report.json` | `runs/.../gauge_K1_R8/_growth_checkpoint/root_report.json` | 661 B | fresh meso provenance | F5 | root registry、anchor 和 budget |
| `meso_baseline_chart_qualification.parquet` | `runs/.../gauge_K1_R8/chart_qualification.parquet` | 15,194 B | raw entity result | F6–F8 | 8 charts 的 support 和各 Gate |
| `meso_baseline_chart_stitchability.parquet` | `runs/.../gauge_K1_R8/chart_stitchability.parquet` | 18,663 B | raw entity result | F9 | 六组 overlap、beta gap 和 stitch status |
| `meso_baseline_section_hypotheses.parquet` | `runs/.../gauge_K1_R8/section_hypotheses.parquet` | 406,976 B | raw entity result | F5–F9 | 3939 hypotheses 的 branch-specific beta 和质量字段 |
| `meso_baseline_growth_events.parquet` | `runs/.../gauge_K1_R8/growth_events.parquet` | 33,120 B | raw diagnostic | F5 | chart growth 接受/拒绝事件 |
| `meso_baseline_chart_audit_schedules.parquet` | `runs/.../gauge_K1_R8/chart_audit_v2_schedules.parquet` | 641,950 B | raw audit registry | F7–F9 | exact chart audit entities和 paths |
| `meso_baseline_chart_audit_executions.parquet` | `runs/.../gauge_K1_R8/chart_audit_v2_executions.parquet` | 4,496,848 B | raw audit execution | F7–F9 | forward/reverse/repeat/tier 的最细数值证据 |
| `meso_baseline_primary_atlas.parquet` | `runs/.../gauge_K1_R8/primary_atlas.parquet` | 12,343 B | raw deployment result | F10–F11 | 2560 nodes 的 abstention mask 和空 primary beta |
| `meso_dropout_report.json` | `runs/.../gauge_meso_root_dropout/report.json` | 3,304 B | registered control | F12 | root-dropout 紧凑结果 |
| `meso_dropout_root_report.json` | `runs/.../gauge_meso_root_dropout/_growth_checkpoint/root_report.json` | 687 B | control provenance | F12 | 7-root registry 与 anchor |
| `meso_dropout_chart_qualification.parquet` | `runs/.../gauge_meso_root_dropout/chart_qualification.parquet` | 15,202 B | raw control | F12 | dropout 后 chart qualification |
| `meso_dropout_chart_stitchability.parquet` | `runs/.../gauge_meso_root_dropout/chart_stitchability.parquet` | 18,663 B | raw control | F12 | dropout 后 overlap/stitch evidence |
| `meso_dropout_section_hypotheses.parquet` | `runs/.../gauge_meso_root_dropout/section_hypotheses.parquet` | 405,967 B | raw control | F12 | dropout 后 3930 hypotheses |
| `meso_dropout_primary_atlas.parquet` | `runs/.../gauge_meso_root_dropout/primary_atlas.parquet` | 12,343 B | raw control | F12 | dropout 后全域 abstention |
| `retry7_config.yaml` | `.worktrees/bacra-v14-2r-retry7-gauge-lock/configs/bacra_v14_2r_stitched_atlas_retry7.yaml` | 5,257 B | frozen config | F1–F13 | 阈值、roots、audit、Reach 和 meso 定义 |
| `canonical_gauge.py` | `.worktrees/.../src/quasi_exp/teacher/canonical_gauge.py` | 21,744 B | source | F10–F11 | anchor component selection 实现 |
| `section_atlas_repair.py` | `.worktrees/.../src/quasi_exp/teacher/section_atlas_repair.py` | 91,405 B | source | F6–F11 | qualification、stitch、materialization、abstention 和 certificate 实现 |
| `retry7_runner.py` | `.worktrees/.../scripts/analysis/run_bacra_v14_2r_retry7.py` | 86,867 B | source | F1–F4、F12–F13 | stage Gate、Reach、meso 和 summary 聚合 |
| `stitched_atlas_runner.py` | `.worktrees/.../scripts/analysis/run_bacra_v14_2r_stitched_atlas.py` | 175,324 B | source | F5–F12 | root registry、anchor policy 和 patch execution wiring |
| `retry7_pipeline.sh` | `.worktrees/.../scripts/pipelines/run_bacra_v14_2r_retry7_then_v14_3.sh` | 1,723 B | source | F1 | 漏斗顺序和 V14.3 授权边界 |
| `retry7_protocol.md` | `.worktrees/.../docs/24-BACRA-V14.2R-retry7执行协议.md` | 4,593 B | frozen protocol | 全部 | 本阶段科学假设、Gate 和停止规则 |

### 未上传的大文件或派生数据说明

- 未上传整个 `runs/`、全部 12-patch numerical executions、模型、缓存和所有 shard completion files。
- 未生成抽样数据；上传的 Parquet 均为所列对象的完整原始表，不是抽样。
- baseline 的完整 chart-audit schedules/executions 已上传，便于独立检查 numerical failures；root-dropout 只上传 chart/hypothesis/primary 级完整数据和 report，未重复上传约 4.5 MB 的高度相似 execution table。
- `retry7_artifact_manifest.json` 可用于核对未上传 artifacts 的名称和 SHA，但 GPT-5 Pro 无法仅凭 manifest 检查未上传文件内容。

## 11. 给下一位讨论者的最短事实摘要

1. retry7 local mechanism 与 12-patch confirmation 全部通过，但 Formal Reach 和 connected 512-parent meso Gate 未通过，所以 repaired 5k/V14.3 未启动。
2. meso 已生长出约 3939 hypotheses 和 4 张 qualified charts；失败不是没有 IK candidates，而是没有任何注册 anchor singleton/stitchable component 达到 90% coverage/60% coherent measure。
3. 四张 qualified charts 在共同 nodes 上存在约 4°–10° beta gap，六对均 nonstitchable；当前 all-or-nothing policy 因此选空 component，并将最终 primary coverage 记为 0%、abstention 记为 100%。
4. 该结果尚不能证明 `xyz-only` 数学上不可能，也不能授权直接放宽 Gate。
5. 用户已确认的问题：判断上述结果的成因，并结合项目最终目标与本阶段目标，给出下一轮实验推进方案。

---

# 结论先说清楚

这次不是 **Codex CLI 本身执行异常**，也不是数值任务跑错了；它是在一个**存在语义缺口的实验规格下，正确执行了一个比我们真正想验证的问题更窄的实验**。

需要区分两件事：

1. **meso 没通过 Gate**：这是计划允许出现、可以接受的科学结果。
2. **meso 实际没有测试“多空间锚点、同 canonical lineage 的 atlas 扩展”**：这不是我们真正希望得到的 meso 实验，应视为**实验计划描述不够明确，加上 Codex 复用了局部 patch 的 root 实现所造成的规格执行偏差**。

同样，(37.15%) 变成最终报告的 (0%)，不是这 (37.15%) 的标签突然失效，而是代码采用了：

$$
\text{先判断是否达到 }90%/60%
\rightarrow
\text{达不到就不选择任何 component}
\rightarrow
\text{不生成 partial primary}
$$

的 all-or-nothing 规则。

所以：

$$
\boxed{
37.15%\text{ 是已认证的局部 section 覆盖，}
\quad
0%\text{ 是当前严格部署策略下的最终 primary coverage。}
}
$$

两者描述的是不同层级。当前报告把它们压成了一个数字，这是实验可观测性和推进策略上的问题。

---

# 核心判断

**当前实验总体上满足了此前规划，而且这次 meso Gate 失败是有价值的科学结果，不是实验跑偏。**

此前规划的漏斗是：

$$
\text{局部机制修复}
\rightarrow
\text{四 patch}
\rightarrow
\text{12-patch confirmation}
\rightarrow
\text{512-cell meso}
\rightarrow
\text{repaired 5k}
$$

retry7 已经完成了 lineage、(K/R) 消融、holonomy 诊断、gauge kernel、`patch_07` 修复、四 patch、12-patch confirmation 和 Reach Round 8；12-patch 达到 development (8/8)、sealed confirmation (4/4)。随后在 512-parent connected meso 上失败，因此停止 5k，完全符合之前设定的 fail-closed 路线。

但有一个重要偏差：

> 当前 meso 实际验证的是“从一个空间位置上的多个构型 root 出发，单张 rooted chart 能否覆盖 90% 的 512-cell 区域”，尚未真正验证“用多个任务空间位置上的同 lineage relay roots 构造广域 atlas”。

我直接核对了 ZIP 内的 [meso root report](sandbox:/mnt/data/q21_py/meso_baseline_root_report.json)：8 个 root 的 `node_id` 全部是 `17755`，差别只是该点上的 `candidate_id`。也就是说，这里的“8 roots”是**同一 (xyz) 上的 8 个逆解 branch**，不是 8 个 task-space maximin 空间锚点。再加上最大 propagation waves 只有 (16)，当前 meso 还没有充分测试我们原本期望的“多空间锚点、同 canonical lineage 的 atlas 扩展”。

所以更准确地说：

- **实验流程满足规划；**
- **局部方法验证满足规划；**
- **meso 阶段成功发现了尺度瓶颈；**
- **但当前 meso 实现只完成了广域 atlas 问题的一个较窄子问题。**

---

# 一、当前究竟是哪一个 Gate 失败了

当前有两个彼此独立的未通过项。

## 1. 直接阻止 repaired 5k 的是 meso Gate

512-cell meso 的直接结果是：

- (512) 个 parent cells；
- (2560) 个 task probes；
- (6038) 条 task edges；
- 生成 (4540) 个 proposals；
- 保留 (3939) 个 hypotheses；
- 8 张 rooted charts 中有 4 张通过 qualification；
- 但没有任何注册 anchor component 满足：
  [
  \text{coverage}\ge0.90,
  \qquad
  \text{coherent measure}\ge0.60.
  ]

因此代码返回：

```text
no_registered_anchor_component_qualified
```

继而：

```text
selected_component = empty
coverage_ratio = 0
abstention_ratio = 1
```

这才是 repaired 5k 未获授权的直接原因。

## 2. Reach Gate 只阻止 Formal

Round 7 和 Round 8 的单轮检查都通过，但 lower/upper measure gap 是：

$$
0.0615196>0.05.
$$

因此 Formal Reach 尚未授权。它**不是当前 meso 失败的原因，也不应阻止 meso 修复或 exploratory Student**。

当前 lower/upper cell 数分别是 (19145) 和 (20400)，差异为：

$$
20400-19145=1255.
$$

要达到 (5%) 阈值，差异最多约为：

$$
0.05\times20400=1020,
$$

至少还需消除约 (235) 个 cell 的不确定性。

---

# 二、meso Gate 的真实因果链

当前 Gate 面板看起来有很多失败项：

- primary coverage 失败；
- largest coherent region 失败；
- certificate 失败；
- long-path geometry 失败；
- nonlocal-cycle nonvacuity 失败；
- root-dropout stability 失败。

但这些并不是六个独立科学反例。真正的因果链只有一条：

$$
\begin{aligned}
&\text{没有 anchor singleton/stitchable component 达到 }90%/60%\
\Downarrow\quad
&\text{selected component 为空}\
\Downarrow\quad
&\text{没有 primary labels}\
\Downarrow\quad
&\text{没有 induced edges、long paths、cycles 和 certificate schedules}\
\Downarrow\quad
&\text{coverage/coherent/certificate/long-path/cycle/root-dropout 全部失败}
\end{aligned}
$$

所以：

- `solver_gate=false` 并不是全域 solver 失败，因为 primary continuation execution 数是 (0)；
- `root_dropout_stable=false` 也不是已经发现 root dropout 导致 beta 不稳定，因为两边 primary 都为空，实际比较节点数是 (0)；
- long-path 和 nonlocal-cycle 失败并不是发现了坏的长路径或坏的闭环，而是根本没有 primary graph 可供审计。

材料也明确指出，当前的 (0%) coverage 是 all-or-nothing materialization 的结果，而不是 (2560) 个点都不可达。

因此，**当前最直接的 Gate 名称应理解为“canonical component selection failure”，而不是“solver failure”或“cycle failure”。**

---

# 三、当前结果究竟说明了什么

## 1. 已经证明：局部 section-first 方法成立

12 个互相独立的 64-parent patches 全部通过，而且每个 patch 都能选择一张覆盖 (100%) 的 anchor singleton chart，geometry max 远低于 (1^\circ)。这说明 retry6 发现的 gauge、root-budget 和 holonomy 问题已经被 retry7 有效修复。

因此当前不再是：

- candidate flood；
- pointwise selector 漂移；
- gauge 未锁定；
- `patch_07` 局部闭环；
- solver 普遍不稳定。

当前主要矛盾已经发生了阶段转换。

## 2. 当前主要矛盾

现在的主要矛盾是：

$$
\boxed{
\text{局部 canonical section 的强一致性}
\quad\text{与}\quad
\text{该 section 在大连接区域内的空间可扩展性}
}
$$

当前起决定作用的主要方面已经从“如何选择一个 branch”转变为：

> **如何在不改变 branch 身份的情况下，把一个局部 section 扩展成广域 atlas。**

## 3. 没有证明：IK 不存在或 xyz-only 必然失败

meso 已经找到 (3939) 个 hypotheses 和 4 张通过审计的 charts，其 support fractions 分别约为：

$$
24.69%,\quad5.94%,\quad37.15%,\quad33.91%.
$$

所以 IK 与局部 section 都大量存在。

真正的问题是：这几张 charts 来自同一空间点上的不同 beta branches，它们在公共节点上的 P95 beta gap 为：

$$
4.07^\circ\sim9.70^\circ,
$$

远高于 stitch 阈值 (0.5^\circ)。它们不能被强行拼成一张连续 section。

但这只证明：

> 不同 root branches 不能互相 stitch。

它没有证明：

> 选定其中一个 branch 后，不能用多个同 lineage 的空间 relay charts 把它扩展出去。

## 4. 当前 (0%) coverage 夸大了“方法失效”的表象

当前代码是在 component 达到 (90%) 之前不 materialize 任何 partial primary，因此最大 qualified singleton 的 (37.15%) 最终被报告为 (0%)。

这一 fail-closed 语义对部署 Gate 是合理的，但对机制诊断不够。下一轮必须同时报告：

$$
\begin{aligned}
&\text{max qualified singleton coverage},\
&\text{qualified-chart union coverage},\
&\text{same-lineage stitchable coverage},\
&\text{nonstitchable boundary measure},\
&\text{deployment primary coverage}.
\end{aligned}
$$

不能再只输出最后一个 (0%)。

---

# 四、最可能的成因排序

## 第一位：空间 root 和传播尺度没有随 meso 尺度扩大

这是我认为当前最优先验证的原因。

当前：

- 8 个 roots 全部位于同一个 task node；
- 它们只是 8 个不同 beta branches；
- maximum growth waves 为 (16)；
- meso 域从 64 parents 扩大到了 512 parents。

因此局部 patch 中“一张 root chart 覆盖全域”能够成立，但在 meso 上每张 chart 只覆盖 (6%\sim37%)，很可能首先是**空间传播半径或 growth horizon 不足**。当前证据也明确把 growth budget、task-graph 尺度列为尚未区分的原因。

这不是简单说“继续加 waves”。应该先判断：

- wave 16 时是否仍有大量可扩展 frontier；
- coverage 是否随 wave 数继续增长；
- charts 是到达 wave cap 停止，还是因为 continuation/consensus failure 停止。

## 第二位：缺少同 canonical lineage 的 relay charts

当前 4 张 qualified charts 是不同 branch charts，彼此相差 (4^\circ\sim10^\circ)。将它们 stitch 本来就不应该成功。

正确的大尺度 atlas 不应由：

$$
\text{同一点的不同 IK branches}
$$

组成，而应由：

$$
\text{同一个 anchor branch 在不同空间位置上的 relay charts}
$$

组成。

relay root 的 beta 应来自已认证 canonical chart 内部节点，而不是在新位置重新独立求一个 IK。这样 relay chart 才继承同一 branch identity。

## 第三位：当前 hard meso domain 具有边界与高条件数偏置

当前 512-cell domain 是从 boundary/high-condition seed 进行 connected BFS 得到的，是一个刻意偏难的 stress domain，而不是 Omega200 的平均区域。

所以：

- 它很适合当压力测试；
- 但不能单独代表整个 Omega200；
- 也不能因为这个域失败就宣布所有中尺度区域都无法建立静态 section。

## 第四位：K1 在中尺度上可能过强剪枝

meso baseline 实际是 `gauge_K1_R8`。局部 patch 上 (K=1) 与 (K=4) 已稳定，但不代表在更长传播距离上 alternative hypothesis 永远没有价值。

因此需要一个廉价的 meso growth-only (K1/K4) 对照，但目前还没有证据要求直接实现复杂 true beam。

## 第五位：少量数值失败

原 anchor chart 有 3 个 persistent numerical entities，另一张未通过 chart 有 15 个。它们值得定向 retry，但不是主要原因：

- 即使原 anchor 的 3 个 failure 全部恢复；
- 它当前 support 也只有约 (25.27%)；
- 仍然远低于 (90%)。

所以不能把下一轮主要计算投入到 solver 微调。

---

# 五、建议下一步：BACRA V14.2R retry8

建议下一轮命名为：

```text
BACRA V14.2R retry8
— Meso Scale Decomposition and Same-Lineage Relay Atlas
```

暂时不要启动 repaired 5k，也不要重跑 12-patch。

---

## Phase 0：现有 meso artifacts 的无 solver 诊断

### 实验目的

先判断当前 partial coverage 是：

- wave cap；
- frontier propagation failure；
- candidate 缺失；
- task graph 结构；
- 还是 branch barrier。

### 实验步骤

对每张 chart 输出：

1. 每个 wave 的累计 coverage；
2. wave 16 时仍处于 frontier 的 node 数；
3. accepted/rejected frontier 的原因；
4. root 到各节点的 graph-hop distance；
5. support 与 boundary、condition、cell strata 的关系；
6. qualified charts 的 union coverage；
7. charts 的空间 support 是否嵌套；
8. uncovered nodes 中已有 IK candidates 的比例；
9. 每张 chart 到停止 frontier 的物理距离。

新增：

```text
meso_chart_support_by_wave.parquet
meso_frontier_failure_taxonomy.parquet
meso_qualified_union.parquet
meso_support_spatial_components.parquet
meso_root_spatial_registry.json
```

本阶段不运行 FK/IK，也不设 Gate。

---

## Phase 1：廉价的尺度因素消融

只运行 growth 和单次 screening，不运行完整 primary certificate。

建议四个候选：

| Variant | (K) | waves |                 空间 roots |
| ------- | --: | ----: | -----------------------: |
| M0      |   1 |    16 |             当前单空间 anchor |
| M1      |   4 |    16 |             当前单空间 anchor |
| M2      |   1 |    32 |             当前单空间 anchor |
| M3      |   1 |    16 | 4 个同-lineage relay roots |

只有 M0 是已有结果，可以完全复用。

### 解释规则

- M1 明显改善：meso 上 K1 剪枝过强；
- M2 明显改善：主要是 growth horizon；
- M3 明显改善：主要是空间 root 布局；
- M1/M2/M3 都不改善：优先考虑 branch/domain barrier；
- M3 的 relay charts 在 overlap 上仍相差 (>1^\circ)：同一 lineage 本身存在中尺度路径依赖，需要进一步 representation diagnostic。

### 计算停止规则

每增加 8 waves 或 4 个 relay roots，若：

$$
\Delta\text{coverage}<0.02,
$$

则停止该方向扩展。

只有 screening coverage 达到约 (80%) 的方案，才值得运行昂贵的完整 certificate；最终科学 Gate 仍保持 (90%/60%)。

---

## Phase 2：构造 same-lineage relay atlas

这是下一轮最关键的方法修复。

### relay root 的构造

1. 先对原 canonical anchor chart 的 3 个 persistent failures 做定向 R1/R2；
2. 若恢复，则继续使用原 canonical anchor；
3. 若不能恢复，按照**事先注册的 anchor fallback 顺序**选择第一张 qualified chart，不能看 coverage 后临时挑最大；
4. 在该 chart 已认证区域的 frontier 内侧选择 task-space maximin 节点；
5. 直接使用这些节点现有的 canonical beta 作为 relay roots；
6. relay chart 继承：

   - `canonical_lineage_id`；
   - original anchor root key；
   - parent relay/root provenance。

不能在 relay 位置重新独立做 diversity IK 后把结果当成同一个 canonical branch。

### 同-lineage stitch Gate

relay charts 的公共节点仍要求：

$$
Q_{0.95}(d_\beta)\le0.5^\circ,
$$

$$
\max d_\beta\le1^\circ.
$$

阈值不放宽。

如果同-lineage relay charts 能 stitch，而当前不同 branch charts不能 stitch，就证明当前失败主要是 atlas root 设计问题，而不是无状态 section 不存在。

---

## Phase 3：重新定义 meso 报告，但不放宽部署 Gate

保留原来的 global-section Gate：

$$
\text{coverage}\ge0.90,
\qquad
\text{largest coherent component}\ge0.60.
$$

但同时增加 diagnostic partial-atlas 指标：

```text
max_single_chart_coverage
qualified_union_coverage
same_lineage_stitchable_coverage
largest_partial_component
nonstitchable_boundary_measure
unresolved_measure
```

并将以下情况区分开：

```text
not_evaluated_due_empty_primary
```

而不是统一记为 `false`：

- long-path geometry；
- nonlocal cycle；
- root-dropout beta stability；
- primary certificate。

当前 root-dropout 在 compared-node count (=0) 时应标为 `not_applicable`，不应解释成实际 root instability。

---

## Phase 4：第二个 meso domain

当前 boundary/high-condition meso 必须保留为 stress result，不能替换。

另外预注册一个：

```text
interior / low-condition / connected 512-parent meso
```

可以先做 256-parent growth screening；只有候选方案接近 Gate 时再扩展到 512。

### 结果解释

- interior 与 hard-boundary 都通过：继续 global xyz-only 主线；
- interior 通过、hard-boundary 只支持 partial atlas：进入“global section + boundary abstention”；
- 两个域都只能形成多个 nonstitchable components：提升 router/experts 或 stateful diagnostic 的优先级；
- 两个域的 qualified union 均低于 (30%)：停止 5k，Teacher atlas 仍未解决。

---

# 六、何时才能进入 repaired 5k

## 路线 A：静态 global section

满足：

- hard meso 或至少两个代表性 meso 中，same-lineage component coverage (\ge90%)；
- largest coherent region (\ge60%)；
- geometry/cycle/certificate 全部通过；
- root/wave stability 通过。

此时可直接进入 repaired 5k。

## 路线 B：显式 multi-chart atlas

如果无法形成单一 (90%) component，但满足：

- qualified union coverage (\ge30%)；
- largest certified component (\ge20%)；
- unresolved + boundary abstention (\le70%)；
- chart domains 在 (xyz) 中具有稳定空间分区；
- 所有被保留的 local charts certificate 通过；

则不应强行 stitch，也不应直接训练 global MLP。应先在 meso 数据上做一个小规模：

$$
\text{global MLP}
\quad\text{vs}\quad
\text{xyz router + local experts}
\quad\text{vs}\quad
\text{abstention}
$$

representation probe，再决定是否进入 5k。

## 路线 C：same-lineage 仍路径依赖

只有当同一个 canonical lineage 的 relay charts从不同路径到同一点后仍稳定相差 (>1^\circ)，并且跨 solver、wave、root 稳定存在，才运行 paired same-(xyz) history experiment，考虑 stateful。

当前不同 branch charts 相差 (4^\circ\sim10^\circ)，本身还不足以触发 stateful。

---

# 七、Reach 下一步

Reach 不应再盲目追加一整轮大规模 Sobol。

Round 8：

- weighted Jaccard 已约 (0.9697)；
- frontier 新 cell 为 (0)；
  -真正未通过的是 lower/upper 的 (1255) 个 disagreement cells。

建议：

1. 输出 A-only/B-only cells 的：

   - sample count；
   - boundary distance；
   - occupancy confidence；
     -邻域支持；
2. 对 disagreement cells 做分层 targeted probes；
3. 优先处理低计数、靠近稳定 interior 的 cells；
4. 达到至少约 (235) 个 cell 的有效归类后，再判断是否需要 Round 9 full Sobol。

这可以与 retry8 meso 并行，不阻塞 exploratory 路线。

---

# 八、计算效率建议

当前不是资源瓶颈：meso 资源指标为约 (8.30) 秒/parent，远低于注册上限 (120) 秒/parent。

但运行结构还有明显优化空间：

- baseline 约 (70.8) 分钟；
- root-dropout 约 (73.1) 分钟；
- baseline 的 root growth 只有约 (149) 秒；
- atlas repair/audits 约 (4098) 秒，占总时间约 (96.5%)。

因此下一轮应：

1. **不重跑 12-patch。**
2. **不再对空 primary 运行完整 root-dropout。**
3. 所有 M0–M3 先只做 growth + 一方向一次 repeat screening。
4. 复用现有 growth checkpoint，只扩展 frontier。
5. 只有一个候选方法运行：
   [
   2\text{ directions}\times3\text{ repeats}
   ]
   的完整 certificate。
6. relay roots 按 4 个一批增加，边际 coverage 小于 (1%\sim2%) 时停止。
7. chart 没有达到最小 support 时，不运行完整 path/cycle audit。
8. baseline、wave extension 和 relay variants 共享持久化 FK/Jacobian cache。
9. 全局仍保持最多 12 个单线程 numerical workers，不需要增加并发。

---

# 最终决策

当前不应解释为：

> retry7 方法已经失败，应该转 stateful。

也不应解释为：

> 只需把 (90%) coverage Gate 降到 (37%)。

更准确的结论是：

$$
\boxed{
\text{retry7 已解决局部 canonical 一致性，}
\text{但当前 meso 仍使用单空间锚点、有限传播半径和不同 branch roots，}
\text{尚未真正构造同-lineage 的广域 relay atlas。}
}
$$

下一步优先级应为：

$$
\boxed{
\text{无 solver 尺度诊断}
\rightarrow
\text{wave/K/spatial-root 小消融}
\rightarrow
\text{same-lineage relay atlas}
\rightarrow
\text{最终 meso certificate}
}
$$

只有这一步仍失败，才应该把主要矛盾从“atlas 构造不足”进一步转换为“静态 global inverse representation 可能不适用”
---

# 一、为什么当前 meso 没有测试“多空间锚点、同 lineage 扩展”

## 1. 当前的“8 roots”其实是同一个空间点上的 8 个 branch roots

当前 root report 里，8 个 root 的形式都是：

```text
[17755, candidate_id]
```

它们的 `task_node_id` 全部是 `17755`，变化的只是 candidate ID。也就是说，当前实验使用的是：

$$
(x_r,\beta_r^{(1)}),
(x_r,\beta_r^{(2)}),
\dots,
(x_r,\beta_r^{(8)})
$$

而不是：

$$
(x_{r,1},\beta_{r,1}),
(x_{r,2},\beta_{r,2}),
\dots,
(x_{r,8},\beta_{r,8}),
\qquad x_{r,i}\neq x_{r,j}.
$$

可以直接查看当前 root registry：

[meso baseline root report](sandbox:/mnt/data/q21_py/meso_baseline_root_report.json)

因此，这 8 张 chart 测试的是：

> 从同一末端位置上的 8 组不同 IK branch 出发，各自能覆盖 meso 区域的多少。

它没有测试：

> 先固定一组 canonical branch，再在该 branch 已经认证的不同空间位置建立 relay roots，共同扩张同一 lineage。

这正是为什么当前 4 张 qualified charts 在重叠点上相差 (4^\circ\sim10^\circ)：它们本来就是同一 (xyz) 上不同的 IK branches，而不是同一 branch 在不同空间位置上的局部坐标图。当前 artifacts 也明确显示，4 张 qualified charts 的 overlap P95 gap 为 (4.07^\circ\sim9.70^\circ)，无法按 (0.5^\circ) 的 stitch 阈值合并。

---

## 2. 这是 Codex 没按计划执行吗？

更准确地说，是**计划语义没有写死，Codex 没有主动识别科学对象发生了偷换**。

最初的整体 V14.3 计划明确写过：

- 5000 parent cells；
- 32 个 **task-space maximin roots**；
- 每个 root 独立增长。

交接方案对 root 的定义也明确包括 task-space maximin roots。

但在后来单独加入 retry7 meso 阶段时，计划只写了：

> 一个 connected 512-cell meso bridge，用来检查长距离传播和非局部 cycle。

却没有进一步明确：

- meso roots 必须位于不同 task nodes；
- branch roots 与 spatial relay roots 是两种不同对象；
- relay root 的 beta 必须继承自同一个 canonical lineage；
- 不允许直接复用局部 patch 的“一个 root node 上做 candidate diversity enrichment”。

于是 Codex 复用了现有 patch runner 的 root 生成逻辑：

1. 先找到一个 root node；
2. 在该 node 上搜索多组 beta candidates；
3. 将这些 candidates 当成 (R=8) roots；
4. 每个 candidate 生长一张 chart。

从工程复用角度看，这是自然实现；从科学问题看，它没有完成我们真正想要的 meso atlas 扩展。

因此责任判断是：

| 问题 | 判断 |
| ------------------------------------ | ------------------------ |
| Codex CLI 是否故障 | 否 |
| 运行是否偏离冻结代码和配置 | 否 |
| Codex 是否实现了 config 中的 `root_count=8` | 是 |
| 计划是否明确规定 8 个不同空间节点 | 没有 |
| Codex 是否应主动发现 root 语义不对 | 应该，但没有 |
| 根本原因 | **实验规格缺少 root 类型和强制不变量** |

所以这次不能简单归咎于 Codex，也不能说完全在计划之内。准确说法是：

> **Gate 失败本身在计划之内；meso 被实现成“同一点多 branch root”而不是“多空间同-lineage relay root”，不在我们的真实实验意图之内。**

---

# 二、为什么 (37.15%) 最终被报告成 (0%)

## 1. 当前代码的实际逻辑

当前 canonical component selector 要求：

$$
\text{component coverage}\ge0.90
$$

且：

$$
\text{coherent measure}\ge0.60.
$$

同时：

```text
require_anchor_selected = true
```

当没有注册 anchor component 同时达到两个阈值时，selector 返回：

```text
component = empty
fallback_reason = no_registered_anchor_component_qualified
```

然后 primary materialization 才开始。由于 selected component 是空的，最终所有节点都被标记为 abstention。当前材料也明确记录了这一实现边界：最大 qualified singleton 的 (37.15%) 不会被写成 partial primary，而会在 deployable field 中坍缩成 (0%)。

因此当前有三个不同概念：

### Chart-level certified support

$$
\mu_{\mathrm{chart}}=37.15%.
$$

表示某张 chart 已在约 (37.15%) 的 meso probes 上给出通过 qualification 的 labels。

### Partial canonical field coverage

理论上应该是：

$$
\mu_{\mathrm{partial-primary}}=37.15%.
$$

剩余区域标记为 abstention。

但当前代码没有生成这份表。

### Strict global authorization coverage

由于没达到 (90%)，当前部署策略认为：

$$
\mu_{\mathrm{authorized-primary}}=0.
$$

这个 (0%) 只表示：

> 当前没有获得“覆盖至少 (90%) 的全域 canonical component”授权。

它不表示：

> 没有 (37.15%) 的可用 labels。

---

## 2. 这是过于严苛吗？

**作为 Formal 或部署 Gate，不算错误；作为探索性 meso 阶段的唯一输出和下游阻断条件，确实过于严苛。**

更明显的问题是，原本 repaired 5k 的探索性 Gate 是：

$$
\text{labelable measure}\ge30%,
$$

$$
\text{largest coherent region}\ge20%.
$$

达到之后就应立即生成数据并训练 Student，以尽早验证数据是否可学。

但现在 meso 前置 Gate 却要求：

$$
90%/60%.
$$

于是产生了一个不一致：

> 要进入一个只要求 (30%/20%) 的 5k 探索性实验，必须先在更难的 boundary/high-condition meso 域达到 (90%/60%)。

这会让 meso 从“中尺度风险检查”变成比下游探索性 5k 更严格的终局证明。

所以需要修正的不是简单把所有科学阈值都删掉，而是：

$$
\boxed{
\text{把 Formal/部署 Gate 与探索性推进 Gate 分开。}
}
$$

---

## 3. 这个结果是否在计划预期内？

分三层回答。

### 预期内

- 512-cell meso 可能失败；
- 一张局部 chart 不能扩展到 (90%)；
- 不同 branch charts 不能 stitch；
- Gate 失败后不自动进入 Formal。

这些都是合理且预期允许出现的结果。

### 不完全在预期内

- 8 个 roots 全部位于同一空间节点；
- 没有真正构造 spatial relay atlas；
- partial certified coverage 不被 materialize；
- long-path、cycle、root-dropout 因 primary 为空而全部连锁记为失败；
- repaired 5k 和 Student 又因此完全跳过。

这些说明 meso 阶段的“诊断输出”和“推进策略”设计不足。

一句话总结：

$$
\boxed{
\text{失败结果可以接受，}
\quad
\text{但当前实验没有完整回答我们真正提出的问题。}
}
$$

---

# 三、Gate 应该如何放宽

我同意下一轮要优先跑通实验，但不建议把所有 Gate 无差别放宽。

## 1. 可以大幅放宽的 Gate

这些是**阶段推进 Gate**，可以放宽：

| Gate                         |     当前 |                              下一轮探索性值 |
| ---------------------------- | -----: | -----------------------------------: |
| meso primary coverage        | (0.90) |                               (0.30) |
| meso coherent component      | (0.60) |                               (0.20) |
| anchor 必须达到全域 Gate           |   true |                 false，允许确定性 fallback |
| primary 必须全局达标后才 materialize |   true | false，始终 materialize partial primary |
| root-dropout 必须通过才进入 5k      |   true |                        改为 diagnostic |
| nonlocal cycle 在空 primary 时  |   fail |                      `not_evaluated` |
| Student 最低监督量                |    20k |              先做 5k–10k smoke Student |
| Student 首轮 seeds             |      3 |                       1 个 smoke seed |
| Student smoke FK P95         |   5 mm |                                15 mm |

## 2. 不应为了跑通而放宽的约束

这些定义了数据是否仍然是合法监督数据：

- 同一个 (xyz) 在主监督表中只能有一个 beta；
- 不允许把相差 (4^\circ\sim10^\circ) 的 branch charts 强行 stitch；
- target canonical beta 不得作为 continuation seed；
- 不允许 padding 或复制行；
- 不允许把 unresolved 节点当作有效 label；
- retained labels 必须满足 actual bounds；
- alternative branches 必须与 primary table 分开。

特别是 stitch threshold。

当前 charts 差的是 (4^\circ\sim10^\circ)，把 stitch threshold 从 (0.5^\circ) 放宽到 (1^\circ) 或 (2^\circ) 也没有意义；放宽到 (10^\circ) 则是在直接混合不同 branch，会重新制造最早的多标签问题。

## 3. 数值质量 Gate 的处理

下一轮可以同时报告两套指标：

### Screening Gate

用于便宜地挑方法：

$$
\text{geometry P95}\le1^\circ,
$$

$$
\text{geometry max}\le2^\circ,
$$

$$
\text{repeat P95}\le0.5^\circ,
$$

$$
\text{FK residual max}\le5\ \mathrm{mm}.
$$

### Certified Primary Gate

真正进入 Student 主监督表的 labels 仍建议保持：

$$
\text{geometry P95}\le0.5^\circ,
$$

$$
\text{geometry max}\le1^\circ,
$$

$$
\text{repeat P95}\le0.2^\circ,
$$

$$
\text{FK residual max}\le3\ \mathrm{mm}.
$$

当前已经有 4 张 charts 通过 qualification，所以问题主要不是数值阈值，而是覆盖与 root 设计。直接放宽几何阈值不会解决核心问题。

---

# 四、交给 Codex 的下一步可执行计划

# BACRA V14.2R retry8

## Spatial-Relay Partial Atlas and Fast Student Probe

## 1. 本轮核心目标

本轮必须完成两个目标：

1. 真正测试**多空间锚点、同 canonical lineage 的 relay atlas**；
2. 即使不能形成 (90%) 全域 section，也必须 materialize 已认证 partial primary，并推进到小规模 Student。

核心问题是：

$$
\boxed{
\text{从一个已认证 branch 出发，}
\text{多个不同空间位置的 relay roots 能否扩大同一 canonical field？}
}
$$

不得再把：

```text
同一 task node 上的多个 beta candidates
```

当作：

```text
多个 task-space roots
```

---

## 2. Root 类型必须显式区分

新增：

```python
class RootKind(str, Enum):
    BRANCH_ORIGIN = "branch_origin"
    SPATIAL_RELAY = "spatial_relay"
    ALTERNATIVE_BRANCH = "alternative_branch"
```

新增 root schema：

```python
@dataclass(frozen=True)
class RootSpec:
    root_id: str
    root_kind: RootKind
    task_node_id: int
    candidate_id: str
    canonical_lineage_id: str
    parent_root_id: str | None
    inherited_beta_rad: np.ndarray
```

### 强制不变量

正式 meso 运行前必须检查：

```text
branch_origin_count == 1
spatial_relay_root_count >= 4
unique_spatial_root_node_count == spatial_relay_root_count
```

并要求：

$$
\min_{i\neq j}|x_{r,i}-x_{r,j}|\ge20\ \mathrm{mm}.
$$

每个 relay root 的 beta 必须来自已经认证的同-lineage primary label：

$$
d_\beta\left(
\beta_{\mathrm{relay}},
c_{\mathrm{parent}}(x_{\mathrm{relay}})
\right)
\le0.1^\circ.
$$

如果所有 root node ID 仍然相同，runner 必须 operationally stop，不得继续长任务。

---

## 3. 冻结 exploratory canonical lineage

### 3.1 优先恢复原 anchor

先只对原 anchor chart 的 3 个 persistent numerical entities 运行 R1/R2。

若恢复后 chart qualification 通过，则：

```text
canonical_lineage_id = retry7_anchor_lineage
```

### 3.2 确定性 fallback

若原 anchor 仍不 qualified，不再让整个实验停止。

从 retry7 已有 qualified charts 中，按以下冻结排序选择一个 exploratory lineage：

1. 最大 certified support；
2. 最少 persistent failures；
3. 最小 geometry P95；
4. 最大 normalized margin；
5. 最小稳定 root key。

一旦选中，写入：

```text
frozen_exploratory_lineage.json
```

后续不得因为新 chart coverage 更高而切换 lineage。

其他 charts 只保存在：

```text
alternative_branch_evidence.parquet
```

不得作为同-lineage relay roots。

---

## 4. Spatial relay root 的生成

在冻结 lineage 已认证的 support 内选择 relay nodes。

### 第一批：4 个 roots

使用 task-space farthest-point/maximin，优先选择：

- 位于当前 chart frontier 内侧；
- graph geodesic distance 较大；
- task-space 距离分散；
- margin 和 condition 仍合格；
- 不处于 persistent failure 附近。

每个 relay root 直接继承该位置的 canonical beta。

### 第二批：8 个 roots

只有第一批使 coverage 增长至少 (2) 个百分点时，才增加到 8 roots。

### 第三批：16 个 roots

只有 8 roots 后每增加 4 roots 仍带来至少 (1) 个百分点 coverage 增长时才执行。

不得一开始直接运行 32 roots。

---

## 5. 低成本 screening 消融

复用 retry7 的当前结果作为 M0，不重算。

| Variant | 空间 roots |    waves | (K) | 含义                               |
| ------- | -------: | -------: | --: | -------------------------------- |
| M0      |        1 |       16 |   1 | 当前结果，只读复用                        |
| M1      |        1 |       32 |   1 | 单纯测试 growth horizon              |
| M2      |        4 |       16 |   1 | 测试 spatial relay                 |
| M3      |        4 |       32 |   1 | relay + 更长传播                     |
| M4      |        8 |       32 |   1 | 仅在 M3 有明显增益时                     |
| M5      | 最佳 roots | 最佳 waves |   4 | 仅在 K1 出现明显 alternative pruning 时 |

### Screening 阶段只运行

- growth；
- chart support；
- overlap beta；
  -一方向、一次 repeat 的 edge screening；
- frontier failure taxonomy。

不要为所有 M1–M5 运行完整 cycle/path certificate。

### 方法选择指标

每个 variant 输出：

```text
max_single_chart_coverage
same_lineage_union_coverage
same_lineage_stitchable_coverage
new_coverage_per_relay_root
new_coverage_per_8_waves
frontier_remaining_count
frontier_failure_reason_counts
screening_geometry_p95/max
runtime
```

### 提前停止

若增加 8 waves 后：

$$
\Delta\text{coverage}<1%,
$$

停止继续增加 waves。

若增加 4 roots 后：

$$
\Delta\text{coverage}<1%,
$$

停止继续增加 roots。

---

## 6. 必须始终 materialize partial primary

修改 `select_anchor_locked_component()` 和 primary materialization 语义。

无论是否达到 progression Gate，都要输出：

```text
partial_primary_atlas.parquet
partial_primary_certificate.parquet
component_coverage_summary.parquet
unlabeled_abstention_nodes.parquet
```

报告中必须分开：

```text
observed_partial_coverage
certified_partial_coverage
exploratory_progression_authorized
formal_or_deployment_authorized
```

禁止再出现：

> chart coverage (37.15%)，但唯一报告的 primary coverage 是 (0%)。

正确语义应是：

```text
certified_partial_coverage = 0.3715
exploratory_progression_authorized = true
global_90pct_authorized = false
```

---

## 7. 新的三层 Gate

## 7.1 Screening Gate

用于选择 M1–M5：

| 指标 | 阈值 |
| ---------------- | -------------: |
| partial coverage | (\ge20%) |
| geometry P95 | (\le1^\circ) |
| geometry max | (\le2^\circ) |
| repeat P95 | (\le0.5^\circ) |
| FK residual max | (\le5) mm |

## 7.2 Exploratory Progression Gate

用于进入 1k/5k 和 Student：

| 指标 | 阈值 |
| ---------------------------------------------- | -------: |
| certified primary coverage | (\ge30%) |
| largest coherent component | (\ge20%) |
| retained labels strict certificate | 通过 |
| nonstitchable branches 是否混入 primary | 否 |
| unresolved critical entities in retained graph | (0) |
| 至少 4 个 distinct spatial roots | 是 |

按照当前最大 singleton (37.15%) 的结果，只要把它作为确定性 fallback component materialize，并通过 retained certificate，这一探索性 Gate 就有可能直接通过。

## 7.3 Formal Gate

继续保留原严格标准，不在 retry8 修改：

-广域覆盖；

- Reach；
- (200k) 预算；
- Student 多 seed；
- original geometry/certificate thresholds。

---

# 五、快速推进到 5k 与 Student

## Phase A：1000-parent end-to-end smoke

在直接运行完整 5k 前，先从 frozen lower proxy 中选一个 1000-parent connected domain。

使用：

- 8 个 task-space maximin relay roots；
- frozen canonical lineage；
- partial materialization；
  -同一套 strict retained certificate。

只要得到：

$$
\text{labelable measure}\ge15%,
$$

或至少 (5000) 条真实唯一 labels，就进入 smoke Student。

这个阶段的目的不是发表结论，而是确保：

$$
\text{Teacher atlas}
\rightarrow
\text{data materialization}
\rightarrow
\text{split}
\rightarrow
\text{Student}
\rightarrow
\text{FK evaluation}
$$

整条链终于跑通。

---

## Phase B：Smoke Student

### 数据

- 5k–10k 条唯一 primary labels；
- 一个 coherent component；
- 不混入 alternative branches；
- 40 mm macroblock split；
- train/validation/test 为 (70/15/15)。

### 模型

只先训练：

```text
global MLP: xyz -> beta6
```

一个 seed，early stopping。

暂不并行训练：

- router；
- LGBM；
- KNN；
- stateful；
  -三 seed full suite。

### Smoke Gate

| 指标 | 阈值 |
| --------------------- | ---------: |
| FK P95 | (\le15) mm |
| FK P99 | (\le30) mm |
| bounds violation | (0) |
| 推理结果可运行 DLS | 是 |
| 明显 beta discontinuity | 单独报告，不直接阻断 |

如果 smoke Student 成功，说明：

> 当前 partial canonical data 至少是可学习的。

此后再扩大 atlas coverage，比继续长期停留在 Teacher Gate 更有价值。

如果 smoke Student 失败，应先判断：

- 数据标签是否仍不连续；
  -空间 split 是否过难；
  -输出归一化和模型容量是否不足；

而不是直接继续增加 roots。

---

## Phase C：完整 repaired 5k

Smoke Student 成功后执行完整 5000-parent Pilot。

Roots 渐进增加：

```text
8 -> 16 -> 24
```

不默认跑满 32。

探索性 Gate调整为：

| 指标 | 首轮可接受 |
| --------------------------- | -------: |
| labelable measure | (\ge20%) |
| largest coherent component | (\ge10%) |
| 三个 x tertiles | 至少两个有覆盖 |
| strict retained certificate | 通过 |
| unique labels | (\ge10k) |

达到后训练 10k Student。

原来的 (30%/20%) 作为主探索结果目标，不再作为第一次 Student 的绝对阻断项。

---

# 六、Codex 必须增加的测试

1. `root_count=8` 但 8 个 roots 的 node ID 相同，测试必须失败。
2. branch origin 与 spatial relay root 必须使用不同枚举类型。
3. relay root beta 必须继承 canonical field，不能重新独立 IK。
4. alternative branch 不能被标为 same-lineage。
5. component 低于 (90%) 时仍输出 partial primary。
6. `observed_partial_coverage` 不能因 authorization false 变成 0。
7. `not_evaluated_due_empty_primary` 不能写成 solver failure。
8. progress Gate 和 Formal Gate 必须是两个独立 JSON。
9. fallback component 选择必须确定性、可复现。
10. nonstitchable (4^\circ\sim10^\circ) charts 不得因为 threshold relaxation 被合并。
11. 只有 retained component 的 labels 进入 Student。
12. 无 padding、无重复 xyz 冲突标签。

---

# 七、计算效率要求

下一轮不要重跑已经通过的：

- lineage；
- K/R local；
- holonomy；
- patch repair；
  -四 patch；
- 12-patch；
- Reach Round 8。

直接从 meso artifacts 分叉。

执行漏斗：

$$
\boxed{
\text{root semantics 修复}
\rightarrow
\text{4 spatial roots growth screening}
\rightarrow
\text{最佳方案完整 certificate}
\rightarrow
\text{1000-parent smoke}
\rightarrow
\text{Student}
\rightarrow
\text{5k}
}
$$

其他效率要求：

- M0 完全复用；
- M1 从现有 frontier checkpoint 继续到 wave 32；
- M2/M3 共享 FK/Jacobian cache；
  -只有最佳 variant 执行双方向三 repeat 全 certificate；
- chart support 小于 (10%) 时不运行全 cycle basis；
  -全机最多 12 个单线程 numerical workers；
  -不与 Student GPU 训练同时运行 12 个重型 Teacher workers；
  -每增加一批 roots 或 waves 都报告边际 coverage，达到停止条件立即停。

---

# 最终判断

这次不能总结为“Codex 把实验做错了”，也不能总结为“计划本来就是这样”。

最准确的结论是：

$$
\boxed{
\text{Codex 正确执行了一个规格不够明确的 meso 实验，}
\text{但该实验只测试了同一点的多 branch charts，}
\text{没有测试多空间同-lineage relay atlas。}
}
$$

而 (37.15%\rightarrow0%) 是：

$$
\boxed{
\text{有意的 fail-closed 授权逻辑}
\quad+\quad
\text{不适合探索阶段的 all-or-nothing materialization}
}
$$

下一轮应当放宽的是**推进权限**，不是把不同 branch 混在一起的科学底线。最重要的路线调整是：

$$
\boxed{
\text{先 materialize 已认证 partial section，}
\text{真正加入 spatial relay roots，}
\text{然后尽快跑到 Student。}
}
$$

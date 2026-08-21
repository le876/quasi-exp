# BACRA V14.2R/V14.3 修订执行协议

状态：预注册执行协议。本文只定义后续实验，不声明新的实验结果。

## 1. 科学目标与表示假设

当前目标是在约 200,000 条最终监督数据预算下，为 Omega200 工作空间构造一张可部署的 canonical inverse field。默认表示继续保留无状态

\[
(x,y,z)\mapsto\beta_6,
\]

但监督表只能包含唯一 primary label。未被选择、且与 primary 相差超过阈值的 alternative branches 仅作为诊断证据，不得进入 global MLP 或 xyz-only router/expert 的监督标签。

只有 paired same-xyz 历史实验同时证明 within-history gap 不超过 \(0.2^\circ\)、between-history gap 大于 \(1^\circ\)，并且差异跨 solver/budget 稳定、不能被空间分区或 abstention 隔离、stateful Student 能显著修复时，才允许转向 stateful。若仅需显式选择已知 branch，则优先 branch token 或 multiple candidates；若失败只局限于小的空间边界，则优先 abstention。

## 2. Formal 前必须闭合的协议修订

1. `patch_09` 为 diagnostic-only。新的独立 confirmation 集合是 `patch_08`、`patch_10`、`patch_11`、`patch_12`。`patch_12` 必须在任何新科学结果产生前，从未使用的 5k cells 中确定性选择并封存 assignment、source SHA 和 hashes。
2. retained primary graph 必须是 retained nodes 在原 task graph 上的 node-induced graph。禁止保留两个 endpoints 却删除其失败邻边。必须有 `edge_completeness_ratio == 1`，并报告 abstention 前后 cycle rank、实际审计 cycle 数和 cycle coverage。
3. `indeterminate` 与 `nonstitchable` 分开。stitch 可由有空间分散度的 overlap 证书或独立 boundary transitions 加跨 boundary cycle 证书建立。证据不足时先运行定向边界延伸；alternative chart 的存在本身不能制造 primary abstention。
4. qualified full-domain singleton 优先于 multi-chart optimization。确需多 chart 时，运行三初值 ICM；64-cell diagnostic patches 用全 probe、全 induced edge 的 MILP reference 检查 objective 与物理 beta field。
5. fresh 12-patch confirmation 与 repaired 5k 之间加入一个未使用、连通的 512-cell meso bridge，检查长路径、非局部闭环、root dropout 和资源增长率。

## 3. Gate 语义

`geometry_gate` 只统计已完成 fresh continuation 的几何差异；`missing_count` 不属于 geometry failure。first-pass solver failure 只作诊断。最终 retained critical entities 在 registered retry 后必须全部解决，否则 `certificate_gate` 失败。

保留的硬阈值：FK residual 不超过 3 mm；geometry P95 不超过 \(0.5^\circ\)，maximum 不超过 \(1^\circ\)；真实微扰 repeat P95 不超过 \(0.2^\circ\)；target canonical beta 禁止作为 continuation seed。

chart qualification 同时使用最小绝对 support、最小 domain fraction 和最小物理 spread。tiny chart 以其承担的 primary measure 比例量化。跨运行稳定性比较 physical beta field、selected component、abstention mask 和 verified physical edges，不直接比较可置换的 chart ID。

## 4. 执行阶段与停止条件

V14.2R 顺序为：clean inventory、replacement `patch_12`、历史 entity-level 诊断、四 diagnostic patches、registered retry、`patch_07` 局部反例、嵌套搜索稳定性、mechanism Gate、独立 Reach Round 7、fresh 12-patch confirmation、512-cell meso bridge。

任一阶段的 schema/hash/source closure 失败属于 operational failure 并立即停止。科学 Gate 失败仍完整保存 artifacts，但不得进入受其保护的下游阶段。Reach Round 7 可与局部机制实验并行，但阻塞 Formal。

V14.3 只有在 fresh confirmation 和 meso bridge 同时通过后才能开始。5k coverage 必须报告 volume-weighted fractional estimate、strict all-probes measure、cell bootstrap 95% interval，以及互斥的 labelable/abstain/unresolved 分解。Formal 的统一口径是 labelable measure 至少 80%，等价地 abstain 加 unresolved 不超过 20%。

## 5. 数据与 Student

新增监督点从至少两个不同 certified local parents 做 source-only predictor-corrector；只有多 parent endpoints 的 beta gap 通过后才 materialize。所有真正参与训练的 rows 计入预算，不允许 duplicate padding。

Student 顺序为 global bounded MLP、primary-field xyz-only router/experts、Student 加 1--2 次 DLS。随机空间测试使用 FK P95/P99/P99.9；随机集 maximum 只报告。最大 10 mm 硬 Gate 仅用于预注册、固定数量和固定 waypoints 的轨迹集。另报告 CPU batch-1 latency。非 primary alternative branches 不得参与 xyz-only Student 监督。

## 6. 计算资源

patch/root/data-shard 级并行固定为 12 个单线程 worker；每个 worker 设置 BLAS、OpenMP、TensorFlow 数值线程为 1，避免嵌套过度订阅。单一图集 integration 等不可安全拆分的阶段可少于 12 个进程，但不得通过重复无科学意义的任务伪造利用率。全部长任务仍由项目 canonical `longrun_tmux.sh` 和 `$long-wait` 监控协议启动。

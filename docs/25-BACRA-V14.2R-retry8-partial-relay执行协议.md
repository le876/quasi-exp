# BACRA V14.2R retry8 partial-relay 执行协议

## 目的

retry8 从 retry7 已封存的 512-parent meso artifacts 分叉，只回答三个探索问题：

1. 一个已认证局部 inverse section 能否始终 materialize 为 `P0`；
2. 同一 canonical lineage 上、位于不同任务节点的 spatial relay roots 能否扩大该 section；
3. 合法 partial labels 能否尽快跑通 Smoke Student，并在不混 branch 的前提下授权 repaired 5k Teacher。

retry8 不重新运行 retry7 已通过的 12-patch、Reach、holonomy、K/R 和局部 repair。

## Root 身份

- `BRANCH_ORIGIN`：冻结 canonical lineage 的原始 root；
- `SPATIAL_RELAY`：从同一 frozen field 直接继承 beta，且任务节点与其他 roots 不同；
- `ALTERNATIVE_BRANCH`：同一任务点的其他 IK family，只作诊断，禁止进入 relay registry。

`D0_raw_retry7` 只复现旧 meso 现象，永不可选。`P0_frozen_partial_singleton` 始终是可选 fallback，并始终 materialize partial primary。

## 计算漏斗

1. 所有 relay 方法先做 growth-only；
2. 有正 coverage 增益的方法进入 sampled screening：relay/root boundary edges、new coverage boundary edges、最多 256 interior edges、最多 64 cycles、forward、baseline 加一个 perturbation；
3. 最多前三名依次做完整双向三 repeat fresh certificate；
4. 三者均失败时回到 P0；P0 只允许一次最小 abstention repair。

## 正交 Gate

- `data_legality_gate`：single lineage、无冲突标签、bounds、FK residual、node-induced certificate 和 unresolved critical entities；
- `relay_method_validation_pass`：至少两个 spatial relays、coverage 相对 P0 正增益、same-lineage stitch 通过；
- `smoke_execution_authorized`：partial certificate 且至少 3000 条合法唯一标签；
- `global_coverage_gate_pass`：coverage 至少 30%，largest component 至少 20%，只决定结论强度；
- `five_k_teacher_execution_authorized`：数据合法、Smoke Teacher/data pipeline 完成、无 label integrity failure、无未修复训练实现失败；
- `formal_or_deployment_authorized`：retry8 永远为 false。

禁止将这些布尔量做一次 `all(...)` 后跳过全部后续阶段。

## 数据与 Student

- Gold：两个独立 parents 一致、forward/reverse、beta gap 不超过 0.5 度、residual 不超过 3 mm；
- Silver：一个 certified parent、reverse return 不超过 0.5 度、与两个最近 certified neighbours 的 beta jump 不超过 1 度、residual 不超过 3 mm；
- Smoke 可用 Gold+Silver，必须分别报告；Formal 默认只用 Gold；
- split 按 40→30→20 mm 选择最粗可行 macroblock，禁止为凑数量泄漏空间邻域；
- Smoke Student 的有限精度 miss 不阻止 5k Teacher；只有数据完整性失败或一次预注册修复后仍存在训练实现失败才阻止。

## Smoke 与 repaired 5k 的域

- balanced Smoke 固定为 1000 parent / 5000 probes；首先保留 P0 的全部 parent，再保留 128–256 个 hard-meso parent，剩余预算优先补充 interior/core-safe cells；原 512-parent hard meso 继续作为独立 stress domain，不要求全部进入 Smoke；
- Smoke 和 repaired 5k 的 roots 均由 frozen lineage labels 按精确 xyz 映射，并在任务空间做 deterministic maximin；不得重新从同点 candidate bank 选择 branch；
- Smoke 使用 8→16→24 root budget，不再默认运行 32 roots；
- repaired 5k 的执行授权只依赖 partial certificate、Smoke 数据完整性和训练实现可运行，不依赖 Smoke 精度 Gate；
- repaired 5k 的 20% labelable、10% coherent、两个 x tertiles 和 10000 labels 是结果目标，不是执行前置条件；低于目标时仍保留 partial dataset，并在合法标签不少于 3000 时训练一版 exploratory Student。

## 可执行入口

- retry8 Teacher：`scripts/pipelines/run_bacra_v14_2r_retry8_partial_relay.sh`；
- balanced Smoke：`scripts/pipelines/run_bacra_v14_3_retry8_smoke.sh`；
- repaired 5k：`scripts/pipelines/run_bacra_v14_3_retry8_5k.sh`；
- 串联入口：`scripts/pipelines/run_bacra_retry8_then_v14_3.sh`。

串联入口只有在 Smoke summary 明确写出
`five_k_teacher_execution_authorized=true` 时进入 5k；该字段不得由
`smoke_student_quality_pass` 单独否决。

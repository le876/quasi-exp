# Scientific objective and claim boundary

## Reproduction objective

项目的基础目标是依据 [原论文](../原论文.md)、[数据集复现方法](../数据集复现方法.md) 和 [力学建模明确事项](../力学建模明确事项.md)，以运动学、准静态力学、采样和数值优化复现连续体机器人的数据集。具体 robot 参数、schema、样本预算与求解约束属于 config/code/tests，不由本文再次冻结。

`整体项目原文.md` 当前为空，不能独立支持事实。上述来源或用户指定实验记录存在空缺或冲突时，必须报告缺口；不得以常识补造参数、方法或结果。

## BACRA research scope

同一 task-space point 可以有多组逆解。BACRA 在有限计算预算下研究如何构造 deterministic、branch-consistent、连续性可审计、可供 Student 学习的 canonical inverse representation，包括 candidate families、inverse sections、charts/atlases、canonical labels、audit evidence、teacher records 和 Student evaluation。

当前实验的精确问题、方法、family 隔离、Gate、停止规则和授权条件由 [registry](../spec/registry.yaml) 指向的 governing record/protocol、config/runner arguments、runner 与 tests 定义。本文不声明任一版本已通过，也不复制它们的数值阈值。

## Claim boundary

- 有限 cells、probes、paths 或 solver budget 的 reach evidence 只支持其注册采样范围，不证明完整连续 reachable set。
- 在注册预算下得到一个 stable candidate family，不证明逆解数学唯一。
- chart/atlas consistency 只覆盖注册的 entities、edges、cycles、paths 和 tolerances，不自动外推到未审计区域。
- canonical label 是注册 selection policy 的输出，不是自然界唯一正确的 inverse。
- Student 结果只支持注册数据域、split、trajectory 和 Gate；不得扩大为未测试 geometry/workspace 的泛化结论。
- operational completion、artifact completeness、scientific pass、downstream authorization 和 formal claim authorization 必须分别由其 protocol/artifact 证据支持。registry binding、结构 verifier、普通 Markdown、commit message 或终端日志不能替代它们。

严格术语见 [CONTEXT.md](../CONTEXT.md)。改变方法、阈值、confirmation family、stage order 或上述边界时，先建立新的 protocol revision；已见 confirmation 结果不得回写已执行协议。

## Evidence required for a formal claim

每项正式结论至少要能指向：scientific source fixed point、protocol source、executable config、相关 runner/tests、runtime 与上游 closure、sealed Gate/report/manifest，以及适用的数据域和实验日期。缺任一必要环节时，应报告为未评估或 gap，而不是从 operational completion 或 artifact 存在推断通过。

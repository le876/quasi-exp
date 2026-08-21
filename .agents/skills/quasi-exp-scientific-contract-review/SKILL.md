---
name: quasi-exp-scientific-contract-review
description: Review a quasi-exp experiment's scientific contract without launching it. Use to trace sources, protocol, config inventory, runner and authorization behavior, negative tests, provenance, sealed artifacts, Gates, downstream authorization, or formal claim boundaries.
---

# Quasi-exp Scientific Contract Review

默认只读。Registry definition 只表示 executable binding 存在，不表示当前正在运行、formal-ready、Gate pass 或 authorization safe。

## 证据链

从 spec/registry.yaml 解析用户选择的 experiment；不要按最高版本号猜测。逐段追踪：

~~~text
scientific source
→ protocol
→ config and source inventory
→ runner producer and authorization consumer
→ negative tests
→ provenance and source closure
→ sealed artifacts
~~~

对每一段记录实际路径、读取到的 contract、生产者、消费者和缺口。只在任务需要结果判断时只读打开引用 artifact；不要从 current-state 摘要代替 artifact。

## 分轴结论

分别报告：

1. operational completion；
2. artifact completeness；
3. scientific Gate；
4. downstream authorization；
5. formal claim authorization。

任何一轴未知时写 not_evaluated 或 gap。不得从 governance verifier 绿灯、registry binding、manifest 非空、进程结束或文字摘要推断科学通过。

## Consultation adoption

审阅 GPT-5 Pro 回答时，逐项核对其引用的 source、attempt 和 artifact，把 observation 与 interpretation 分开。只有用户明确采纳的 method、Gate、停止、authorization 或 claim boundary 才进入新 protocol/repair addendum；可执行 binding 随后登记为新的 registry definition。

## 变更与运行边界

- Review、audit、diagnose 保持只读。
- 在用变更后的 scientific implementation 生成可比较或可引用 evidence 前，建立新的 source fixed point 与相关 tests；只有方法、Gate、停止、authorization 或 claim semantics 改变时才新增 protocol revision/repair addendum。新 evidence 使用 fresh output root。
- Agent Note 只记录 durable rationale，不能代替 scientific protocol。
- 不自动启动 formal 或 long-running experiment；已授权且预计超过五分钟的命令由 $long-wait 唯一管理。

输出应把已验证事实、推断、未读 artifact、未运行测试和 remaining gaps 分开。不要为了让链路“闭合”而补造 source、修改阈值或回写协议。

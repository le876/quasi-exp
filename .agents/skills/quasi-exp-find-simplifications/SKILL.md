---
name: quasi-exp-find-simplifications
description: Find evidence-backed simplifications in quasi-exp by tracing real callers and scientific consumers. Use for scoped simplification passes, dead or duplicated code, speculative indirection, redundant representations, dependency swaps, or over-built experiment flows; not for routine experiment execution.
---

# Quasi-exp Find Simplifications

对一个有边界的实现流查找净删除，以 caller、consumer、call site 和 owning tests 为证据；audit 与 implementation 分开，不制造候选。

## Scope and mode

1. 先确认实际 worktree、branch、HEAD、dirty state、用户指定 scope 和活动实验边界。
2. 默认只检查一个 experiment flow、runner 或 shared subsystem。由 registry 或用户入口定位 protocol、config、runner、tests 及 artifact/report consumers。
3. 用户明确要求只读 review/audit 时只报告。普通 scoped pass 可以写 proposed Agent Note 或小型 marker，但不在同一次 pass 中修改功能。
4. 用户要求 breadth、many candidates、整个实验族、多个 subsystem 或全仓 survey 时，直接把独立范围交给只读 subagents；可按 protocol/config/runner、numerical/model、tests/negative contracts、artifact/provenance、pipeline/tooling 划分。主代理继续自己的 scope，跟踪并统一核对、去重和持久化；不向 subagent 提供预期 finding，也不让它们并发编辑。

## Evidence workflow

1. 用 `rg` 搜索 exact symbol、config key、artifact field、function call 和动态字符串，再阅读真实 call site。
2. 搜索相关 active Agent Notes，确认结构是否有仍然成立的理由。
3. 只对能删除、折叠、降级、内联或由 dependency 替换的 surface 建候选，并计算：

   ```text
   removed implementation + removed dedicated tests/docs
   - new glue/wrapper/configuration
   ```

4. 有现实 consumer、没有净删除、只产生无关 churn，或仅仅“看起来复杂”的候选直接拒绝。结果可以是零个强候选。

对现实候选标明 consumer：`execution consumer` 是 config、runner、launcher、runtime loader 或当前 executable flow；`scientific/evidence consumer` 是 protocol、Gate、negative/fail-closed test、authorization、provenance、artifact/report consumer；`historical fixed-point only` 只服务旧 source commit；`no verified consumer` 表示已查范围内没有 reader。只有最后一类可直接进入删除候选。历史限定的 surface 只有在不再是 standing authority、没有当前 binding 或 artifact consumer，且 Git 仍可读取原 source commit 时，才可提议从当前 checkout 删除。

Tests/docs 按具体 assertion 或命题判断，不按目录处理。Gate、authorization、source identity、provenance、artifact closure、negative control 和 formal protocol 是现实 consumer；只保护已删除 API 或重复当前 owner 的材料可随实现删除。原论文、实验 chronology、sealed artifacts 和 GPT handoff/reference input 不进入普通 cleanup。

## Behavior, duplication, and dependencies

简化可以产生合理且容易解释、维护的轻微行为差异。每个候选说明差异、合理性和 owning tests；不要求 bitwise identical。helper placement、内部数据结构、batching/并行/调度、cache、日志/异常组织、等价 dependency，以及无现实 consumer 的旧行为都可调整。若候选明确改变 governing protocol、Gate、停止、authorization 或 claim，转给 `$quasi-exp-scientific-contract-review`；本 Skill 不复制该审查流程。

只有出现手工同步、真实 drift、同步代码、无 reader 副本或可证明净删除时，才审查镜像表示。删除前只问：第二份是否保持 self-contained executable config？是否供下游独立验证 source、hash、Gate、manifest 或 tampering？任一为是就保留。相同字段、hash、YAML 或 Markdown 不构成证据；stage identity、registry/config identity 和 provenance hash 若被 validator 消费，也不是冗余。

Dependency candidate 必须由稳定维护的库覆盖当前 surface，并明确 residual glue、transitive footprint 及实现、专用 tests/docs 的净删除。numerical 或 persisted surface 用 owning tests、既有 tolerance 或 artifact consumer 验证可用性。Audit 只提出候选，不安装 package。

## Persist and report

强且 durable 的候选立即新增或更新 `.agents/notes/proposed/`，沿用 `Problem`、`Proposal`、`Alternatives considered`、`Acceptance criteria`、`Risks`；先搜索重叠 owner，不建 placeholder。小候选写稳定 tag 和具体动作：`FIXME` 阻塞下一次 release，除非用户明确接受；由 `TO` 与 `DO` 直接相连的中优先级 marker 在资源允许时尽快处理；`XXX` 最低优先级且无承诺。不要写 speculative complaint。

按以下标题报告：

```text
Scope and fixed point
Strong candidates
Retained or rejected candidates
Notes or markers written
Recommended validation
Unexamined surfaces
```

每个强候选至少列出 target surface、real callers、scientific/evidence consumers、existing rationale、准确删除或折叠方案、net deletion、allowed behavior difference、risks 和 owning tests。

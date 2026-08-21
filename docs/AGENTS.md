# Documentation maintenance

本文件约束 `docs/` 及其子目录；根 [AGENTS.md](../AGENTS.md) 的事实、运行和工作树安全约束仍然适用。先使用 [documentation map](README.md) 找唯一 owner，再编辑文字。

## Before writing

1. 绑定实际 worktree、HEAD、dirty state 和任务范围；不要把主工作树或最新文件名假定为当前 experiment。
2. 从 [current state](current-state.md) 取得当前选择，或使用用户指定的 registry definition，再读取对应 protocol、config、runner 和 tests。需要结论时单独只读 artifact。
3. 源文件为空、缺失或冲突时写明 gap。`整体项目原文.md` 为空，不能单独支持新事实。

## Placement and boundaries

- 目标与 claim boundary 写 [scientific objective](scientific-objective.md)；系统结构写 [Architecture](../ARCHITECTURE.md)；术语写 `CONTEXT.md`；长期理由写 `.agents/notes/`。
- governing record/protocol 拥有方法、Gate、停止和 authorization 语义；config 与明确 runner arguments/defaults 拥有可执行数值；runner/tests 拥有行为与 negative evidence；artifact 拥有结果和 hash。
- `current-state.md` 只保存有时间的 experiment、attempt 与 locator。不要在那里复制 source binding、Gate 指标或总结结果。
- dated experiment record/audit 保存当时的合同、证据范围、chronology 与缺口；只有 registry 明确绑定时才作为该实验的 governing record，且不会因此成为 formal protocol。
- GPT-5 Pro 咨询使用 `docs/{N}-pro提问-{主题}.md` 与 `docs/{N}-pro回答-{同一主题}.md`；一个问题可比较多个 attempts，但要列明 roots、fixed points 与 evidence cutoff。回答首行为 `# GPT-5 Pro 第 N 次回答：<主题>`，无 YAML，只在原文前记录 `Question` 相对链接、`Received`、`Source` 和 `Authority: advisory scientific interpretation and candidate plan`。
- 等待回答时 `current_consultation_locator` 指向提问，审阅回答时指向回答；用户采纳并形成 protocol 后或没有当前咨询时置为 `null`。采纳的方法、Gate、停止或 authorization 另写 protocol。
- Skill 只保存可复用 workflow，不拥有科学 contract、runtime behavior 或实验结果。
- 不在导航 Markdown 手工同步 YAML threshold/seed 表，也不建立第二套 Agent Note owner。

## Trim boundary

删除 reasoning transcript、review choreography 或 change narration 只适用于 current state、Architecture、README、AGENTS、Skill 和普通代码注释，且必须保留完整事实命题。原论文、原始事实来源、formal protocol、实验记录/故障 chronology、provenance、GPT handoff/reference input、archived Agent Note 与 sealed artifact 不受这类通用精简规则约束。

## Standing-doc budgets

[`doc-budgets.yaml`](../scripts/spec/doc-budgets.yaml) 只列容易累积的 standing docs，并以 Unicode characters 设置 ceiling；不自动预算 Agent Notes、Skills、领域 reference 或实验记录。`--list-budgets` 只报告 usage，普通实验执行和 scientific code/test 修改不运行预算 gate。

默认 verifier 报告超限时依次 relocate → condense → raise。确需提高单个 ceiling 时在当前 change 说明理由；只有预算 policy 或其 durable rationale 改变时才更新 Agent Note。

## Lifecycle

已用于 formal run 的 protocol 不原地改变科学含义。新方法、阈值、确认集、stage order 或 claim boundary 使用新 revision，并更新其 executable binding；Agent Note 不能代替它。历史文档除非已验证 source/provenance closure，否则保持原路径。

## Validation

按 [testing.md](testing.md) 选择与变更直接相关的检查。文档/authority 变更通常运行默认 spec closure 和 `git diff --check`；结构检查只验证 owner、路径、binding 和 Git identity，不能验证 runner 语义或 scientific result。报告实际运行的命令、未运行的 artifact/formal 检查及原因。

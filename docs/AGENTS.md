# Documentation maintenance

本文件约束 `docs/` 及其子目录；根 [AGENTS.md](../AGENTS.md) 的事实、运行和工作树安全约束仍然适用。先使用 [documentation map](README.md) 找唯一 owner，再编辑文字。

## Before writing

1. 绑定实际 worktree、HEAD、dirty state 和任务范围；不要把主工作树或最新文件名假定为当前 experiment。
2. 按任务读取用户指定的 experiment definition 或 [current state](current-state.md) 所选 definition；需要结论时单独只读 artifact。
3. 源文件为空、缺失或冲突时写明 gap。`整体项目原文.md` 为空，不能单独支持新事实。

## Placement and boundaries

- 先按 [documentation map](README.md) 选择唯一 owner；本文件只规定 `docs/` placement，不复述 registry/current-state schema。
- 目标与 claim boundary 写 [scientific objective](scientific-objective.md)；系统结构写 [Architecture](../ARCHITECTURE.md)；术语写 `CONTEXT.md`；长期理由写 `.agents/notes/`。
- dated experiment record/audit 保存当时的合同、证据范围、chronology 与缺口；formal method、Gate、停止或 authorization 写独立 protocol/revision。
- GPT-5 Pro 咨询使用 `docs/{N}-pro提问-{主题}.md` 与 `docs/{N}-pro回答-{同一主题}.md`；一个问题可比较多个 attempts，但要列明 roots、fixed points 与 evidence cutoff。回答首行为 `# GPT-5 Pro 第 N 次回答：<主题>`，无 YAML，只在原文前记录 `Question` 相对链接、`Received`、`Source` 和 `Authority: advisory scientific interpretation and candidate plan`。
- Skill 只保存可复用 workflow；导航 Markdown 不复制 YAML threshold/seed、artifact 结果或 Agent Note rationale。

## Trim boundary

删除 reasoning transcript、review choreography 或 change narration 只适用于 current state、Architecture、README、AGENTS、Skill 和普通代码注释，且必须保留完整事实命题。原论文、原始事实来源、formal protocol、实验记录/故障 chronology、provenance、GPT handoff/reference input、archived Agent Note 与 sealed artifact 不受这类通用精简规则约束。

## Standing-doc budgets

[`doc-budgets.yaml`](../scripts/spec/doc-budgets.yaml) 用 Unicode-character ceiling 发现 standing-doc 膨胀。超限是 warning，不阻塞 governance 或实验；需要处理时依次 relocate → condense → raise。只有 budget schema、非法值或声明文件缺失属于结构错误。

## Lifecycle

已用于 formal run 的 protocol 不原地改变科学含义。新方法、阈值、确认集、stage order 或 claim boundary 使用新 revision，并更新其 executable binding；Agent Note 不能代替它。历史文档除非已验证 source/provenance closure，否则保持原路径。

## Validation

按 [testing.md](testing.md) 选择与变更直接相关的检查。文档/authority 变更通常运行默认 spec closure 和 `git diff --check`；结构检查只验证 owner、路径、binding 和 Git identity，不能验证 runner 语义或 scientific result。报告实际运行的命令、未运行的 artifact/formal 检查及原因。

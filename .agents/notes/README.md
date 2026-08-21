# Agent Notes

`.agents/notes` 是本仓库 durable rationale 的唯一 owner。Note 不拥有 scientific protocol、executable config、current status 或实验结果。

## When to write one

一次 change 只要改变 behavior、architecture、跨文件 shared contract、process/tooling、testing strategy、config/on-disk/wire format，或形成其他需要跨 session 保留且未来可能重访的 durable decision，就属于 `non-trivial`，必须在同一 change 中新增或更新至少一个 Agent Note。

先搜索 active Notes；已有 Note 拥有该取舍时更新它，只有没有合适 owner 时才新建。Note 记录 code、protocol、config、tests 或 sealed artifact 无法自行表达的 durable rationale，不复制它们拥有的阈值、结果或实现细节。

只有两个例外：

- Mechanical/local：纯格式、拼写、无语义链接修复或局部机械修改，且不改变上述六类内容或 durable rationale。
- Execution-only research：严格按既有 protocol、config 和 runner 执行实验、生成或封存预期 artifact，或只刷新 pointer-only locator；一旦改变方法、Gate、停止准则、claim、authorization、schema、代码行为或 testing strategy，就不再属于例外。

科学方法、Gate、证据要求、authorization 或 claim boundary 变化仍需新 protocol revision/repair addendum；Agent Note 不能替代它。

## Layout and naming

文件名使用 `YYYY-MM-DD-short-slug.md`。日期是该主题首次提出的日期；lifecycle 移动和后续更新不改变文件名，后续历史由 Git 保存。Active lifecycle 只有：

- `proposed/`：尚未生效；
- `implemented/`：已由当前系统或 governing specification 实现；
- `rejected/`：明确拒绝，且理由仍能阻止一次现实的重复错误。

Note 不使用 YAML frontmatter。首部固定为：

~~~markdown
# Agent Note: <title>

Status: proposed | implemented
~~~

rejected 使用 `Status: rejected — <明确理由>`。

## Required sections

- proposed：`Problem`、`Proposal`、`Alternatives considered`、`Acceptance criteria`、`Risks`。
- implemented：`Problem`、`Decision`、`Alternatives considered`、`Consequences`。
- rejected：`Problem`、`Proposal`、`Alternatives considered`。

每份 Note 的第一个二级标题必须是 `## Problem`。`Verification` 可在有实际命令、测试、artifact 或 fixed point 时使用，但不是必需章节。`Revisit condition` 不是标准章节；有价值的条件写入 `Consequences`、`Deferred` 或相关正文。

implemented Note 的 `Decision` 用 present tense 描述已经实现并持续成立的现实。当 active Note 已经陈述、或用于定位其决定的路径、名称、symbol、key、default 或机制变化时，在同一 change 中更新相应事实；与该决定无关的局部重命名不触发 Note 维护。文件日期不变，后续历史由 Git 保存。implemented Note 不追加 change history，也不保留 `Proposal`、`Plan`、`Migration plan`、`Acceptance criteria` 等 proposal-era 章节。

状态改变时移动文件并重写 `Status:` 及 lifecycle 对应正文。proposed → implemented 时，把 proposal 改写为当前现实；实际检查可放入 `Testing` 或可选的 `Verification`，取舍写入 `Consequences`。不得把同一文件改写成相反决策。

## Supersession

每个新 Note 都对同一决策或机制做 scoped supersession search。只有真正发生 supersession 时才在正文使用相对 Markdown 链接；不预留空 metadata。

- Partial supersession：双方保持 active 并交叉链接，明确各自仍拥有的范围。
- Full supersession：当前 owner 先吸收旧 Note 独有的 rationale、alternatives、consequences 和 coverage gaps，再合并或归档旧 implemented Note。

出现候选、归档或语料去重任务时使用 [$quasi-exp-archive-agent-notes](../skills/quasi-exp-archive-agent-notes/SKILL.md)。

## Archive

`archived/` 只接收 implemented Note；保留 `Status: implemented`，并在下一行增加 `Archived: YYYY-MM-DD`。是否归档取决于未来决策价值，不取决于年龄、长度或配额。

第一次真实归档时，由下列命令创建 `archived/manifest.json` 并以 SHA-256 冻结每个 archived Markdown；没有 archived Note 时不保留空 manifest：

~~~bash
/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python scripts/spec/verify_spec_system.py --seal-archive
~~~

manifest 一旦存在，该入口只追加新 seal；已有路径、hash 或删除必须 fail closed。封存后不得编辑、移动或删除文件。Archived Note 只检查身份 header、归档日期和 seal，不追溯套用以后新增的 active-body 格式规则。普通搜索由根 `.rgignore` 排除 archive，需要历史证据时显式读取。

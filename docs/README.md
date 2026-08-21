# Documentation map

本目录是导航和记录层，不是第二套 config、结果数据库或 Agent Note 系统。一个事实只保留一个 owner；其他文档只链接或路由到它。

| Question | Owner | Do not duplicate it in |
| --- | --- | --- |
| 项目目标与可作主张的边界 | [scientific-objective.md](scientific-objective.md) | current state、Architecture、普通实验记录。 |
| 系统组成、依赖与代码放置 | [ARCHITECTURE.md](../ARCHITECTURE.md) | protocol、版本日志。 |
| 术语 | [CONTEXT.md](../CONTEXT.md) | Architecture、protocol 摘要。 |
| 可执行 experiment definition/binding | [registry.yaml](../spec/registry.yaml)；binding 在其 scientific source fixed point 中解释 | current state、README、release map。 |
| 方法、stage order、Gate、停止与 authorization 语义 | registry 绑定的 governing record/protocol | registry、导航文档、Agent Note。 |
| 可执行数值、seed、路径和 output root | YAML config 与明确的 runner arguments/defaults | 导航 Markdown。 |
| 协议的可执行实现与 fail-closed enforcement | registry 绑定的 runner；tests 提供验证 | pipeline、current state。 |
| 当前 experiment、attempt 与 locator | [current-state.md](current-state.md) | registry、结果报告、Architecture。 |
| 实际结果、Gate、hash、manifest | `data/`、`runs/` artifact；formal claim 还需 seal | current state、commit message。 |
| GPT-5 Pro 科学咨询 | 成对的 `docs/{N}-pro提问-*` / `docs/{N}-pro回答-*` | artifact、protocol、current state。 |
| 实验 chronology 与一次性审计 | 对应 dated record 或 `docs/audits/` | current state、Agent Note。 |
| 长期设计取舍 | [.agents/notes](../.agents/notes/README.md) | protocol、临时日志。 |
| scientific/public SHA 对应与未发布 fixed-point 集合 | [release-map.yaml](../spec/release-map.yaml) | registry experiment membership、current state、release note。 |
| 测试层与命令选择 | [testing.md](testing.md) | AGENTS、协议正文。 |
| 文档编辑约束 | [docs/AGENTS.md](AGENTS.md) | 每份具体文档。 |
| 可复用维护流程 | `.agents/skills/` | AGENTS、科学 contract、结果记录。 |

## Harness concepts

- **standing order**：Codex 自动加载的根或子树 `AGENTS.md` 规则；只保存稳定约束与任务路由。
- **experiment definition**：registry 中 source/protocol/config/runner/optional launcher/tests/upstream 的静态对应；不等于当前选择、正在运行、通过或授权。历史 binding 在声明的 source commit 内自洽即可，当前 checkout 不承担保留旧文件或旧 bytes 的责任。
- **scientific source fixed point**：某次科学实现绑定的 Git commit，不等于 public release SHA；release map 只记录 SHA 对应与尚未发布的 SHA，不重复 experiment membership。
- **mutable pointer**：带观察时间的 current-state locator，可更新但不拥有结果。
- **consultation input**：GPT-5 Pro 对证据的外部解释与候选计划；用户采纳并写入 protocol 前不支配实验。
- **sealed evidence**：同时具有 source/provenance closure 的 artifact；普通 unsealed report 仍是结果材料，但不能支持 formal claim。
- **Agent Note**：需要跨 session 保留的 durable rationale；不能代替 protocol。
- **archived memory**：不再 active、但仍有未来证据价值且经 SHA-256 封存的 implemented Note。
- **governance structure**：owner、路径、schema、Skill、Note 与 seal 的结构一致性；standing-doc budget 超限只是膨胀提示，不阻塞该轴。
- **scientific contract**：source → governing record/protocol → config/arguments → runner/tests → artifact/provenance 的语义链。
- **persistence**：authority 资产已被 Git 跟踪、存在于 `HEAD` 且工作树与其一致。

## Use

按 [根 AGENTS.md](../AGENTS.md) 依任务加载 owner；结果任务直接读 artifact，当前定位才读 current state。不要从版本名、commit message、binding 或 verifier 绿灯推断 scientific pass。修改 `docs/` 前读 [docs/AGENTS.md](AGENTS.md)，验证选择见 [testing.md](testing.md)；历史缺口只报告，不补造来源。

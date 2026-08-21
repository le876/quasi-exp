# quasi-exp architecture

本文是当前代码与证据流的结构图，不保存版本化阈值、运行进度、Gate 结果或历史取舍。它们分别属于 governing record/protocol、config 或 runner arguments、artifact 和 `.agents/notes/`。

## System boundary

项目以机器人/力学输入、目标工作空间和一次明确的 scientific contract 为输入，经 config 与 runner parameters 驱动运动学、准静态、优化、数据生成和评测代码，产生可核对的 datasets 与 reports。有限采样、有限搜索或 artifact 都是注册范围内的经验性证据，不能单独证明连续可达域、数学唯一逆解或全局连续性；术语见 [CONTEXT.md](CONTEXT.md)，可作出的主张见 [scientific objective](docs/scientific-objective.md)。

```text
source facts + governing record/protocol
        -> config + explicit runner parameters
        -> reusable io / model / optimization code
        -> experiment runner + direct tests
        -> dataset and report artifacts

registry catalogs source, contract, config, runner, optional launcher and tests
current-state selects one definition and points to one attempt
pipeline supplies runtime lifecycle when one exists; it does not own science
```

`spec/registry.yaml` 记录静态 executable experiment definitions；`docs/current-state.md` 单独记录当前选择、attempt 与 locator。两者都不执行阶段，也不拥有结果或授权。

## Code layers and seams

| Layer | Owner | Responsibility |
| --- | --- | --- |
| Input and configuration helpers | `src/quasi_exp/io/` | 读取与校验机器人/实验输入。 |
| Physical model | `src/quasi_exp/model/` | kinematics、sampling、quasi-static 与 tension transmission。 |
| Numerical methods | `src/quasi_exp/opt/` | inverse optimization、PSO、segmented/canonical tension 求解。 |
| BACRA Teacher and progression | `src/quasi_exp/teacher/` | workspace inverse、atlas、partial relay、retry progression 与数据/Student 组合所需的可复用逻辑。 |
| Provenance | `src/quasi_exp/provenance.py` | source/config/runtime identity 与可审计 provenance 边界。 |
| Dataset and experiment composition | `scripts/`、`scripts/analysis/` | 将 config/CLI contract 接到库代码，生成数据、诊断和版本化实验结果。 |
| Runtime lifecycle | `scripts/pipelines/` | 环境、进程和 longrun 入口；不拥有数值参数或科学 Gate 语义。 |
| Verification | `tests/` | contract、negative control、regression 与 fail-closed 行为。 |

Runner 可以显式表达 stage order、Gate flow、resume 和 provenance closure，也可以作为没有独立 launcher 的 direct entrypoint。只有至少两个真实 flow 共享同一计算，或复制已经造成行为漂移时，才把它下沉为可复用实现；不要为单一 consumer 创建转发层。

## Control and evidence flow

执行链路是：current selection → registry definition → governing contract/config/arguments → runner（可选 launcher）→ dataset/report 或 stage artifacts。Formal flow 在读取被封存的上游输入和已完成 stage 时验证 closure；普通 pilot 可能没有独立 manifest。Scientific contract 拥有 Gate、停止、失败与 authorization 语义，runner 实现，tests 验证。

架构上必须保持下列分离：

- `io`、`model` 和 `opt` 不依赖 shell/longrun 生命周期。
- runner 是 composition root，不重新实现可复用物理或优化算法。
- pipeline 不定义或放宽 scientific threshold、confirmation family 或 claim semantics。
- `data/`、`runs/` 是输入/输出边界；已封存 output root 不由文档或代码维护原地修补。
- Gate、manifest、operational completion 与 authorization 以各自 artifact 字段读取；实验记录可以解释证据，但不是机器结果 owner。
- public release SHA 与 scientific source SHA 通过 [release map](spec/release-map.yaml) 关联，不从 commit message 推断。

## Where a change belongs

| Change | Place it with its verification |
| --- | --- |
| 物理、运动学或张力传递 | `src/quasi_exp/model/` 和直接 tests。 |
| 优化器或 canonical rule | `src/quasi_exp/opt/`，或真实拥有该实验组合的 runner，并配直接 tests。 |
| 数据生成或分析流程 | 对应 `scripts/` entrypoint 与 owning/downstream tests。 |
| 新实验或变更 scientific behavior | protocol revision + config + runner + tests + 新 registry definition；必须在新 output root 取证。 |
| 运行/调度机制 | `scripts/pipelines/`、runbook 与必要的 Agent Note；只有语义改变时才改 protocol。 |
| 测试选择或解释 | [docs/testing.md](docs/testing.md)。 |
| 当前 locator | [docs/current-state.md](docs/current-state.md)；真实状态仍取 artifact/runtime owner。 |
| 持久的设计取舍 | `.agents/notes/`。 |

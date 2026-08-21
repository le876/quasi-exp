# 项目目的

基于 `原论文.md` 及整体项目原文中的机器人项目，复现原论文的数据集。

## 事实来源

- 以 `原论文.md`、`数据集复现方法.md`、`力学建模明确事项.md` 及用户明确指定的实验记录为事实来源。
- 引用的源文件为空、缺失或相互冲突时，明确报告缺口；不得自行补造论文事实、参数或实验结果。

## 目录职责

- `src/`：可复用库代码。
- `scripts/`：命令入口与实验流水线。
- `configs/`：实验配置。
- `tests/`：自动化验证。
- `docs/`：实验记录与项目文档。

## 环境与验证

- 标准 Python 为 `/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python`。
- 每次改动都运行与改动行为直接相关的测试。
- 修改公共数值逻辑、数据 schema、数据生成器或跨模块基础设施时，运行 owning tests 与可识别的受影响下游测试。只有影响无法可靠缩小、跨多个实验族，或用户明确要求时才运行全量测试：
  `/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python -m pytest -q`
- 全量验证成本过高或环境不可用时，明确列出已运行的检查、未运行的检查及原因。
- 不假定项目存在未在仓库中定义的 lint 或 typecheck 命令。

## 长任务与实验结果

- 预计运行超过 5 分钟，或用户明确要求长时间后台自动完成的非交互式本地命令，必须触发全局 `$long-wait` skill；monitor 编排以该 `SKILL.md` 为唯一流程真源，本文件不复刻。
- 项目长任务遵循 `docs/长任务防中断运行SOP.md`，使用既有的 `scripts/pipelines/longrun_tmux.sh` 作为进程状态和退出码的事实来源。
- 未经用户明确请求，不覆盖已有 `data/` 或 `runs/` 结果，不重复启动同一长任务。

## 主要思想

- 本项目是科研任务，实现以实验原理验证，推进实验为核心目的，不要太追求工程鲁棒性，要提升token的效率和实现的效率。
- 要准确理解，思考用户的真实意图
- 不要过度保守，注意效率
- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, or migrations.
- Choose the simplest implementation that fully meets the current requirements. Avoid speculative abstractions, configuration, and indirection.
- Grow the system in layers. Start from the smallest version that works end to end, and add each new capability on top of a product that already works. Never trade a working product for unfinished complexity.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce overall complexity or improve reliability. Do not reimplement common functionality without a clear reason.
- Lean on the dependencies already in the project before writing your own implementation or adding packages. Do not assume a library lacks a capability without checking its documentation and types.
- Make architectural decisions for the long term. Do not accept a stopgap that only works for now and is meant to be replaced later.

## 科研 Harness 路由

- 实验任务先读 `spec/registry.yaml`，再只读所选 experiment 绑定的 protocol、config、runner 与 tests；不要按版本名猜测。
- runtime/status 任务才读 `docs/current-state.md` 和所指 runtime/artifact；术语任务才读 `CONTEXT.md`；架构任务才读 `ARCHITECTURE.md` 与相关 Agent Note。
- 用 [$quasi-exp-harness-maintenance](.agents/skills/quasi-exp-harness-maintenance/SKILL.md) 维护 authority、binding、pointer、standing docs、repo Skills 与治理验证；用 [$quasi-exp-scientific-contract-review](.agents/skills/quasi-exp-scientific-contract-review/SKILL.md) 审查科学证据链。
- 需要 GPT-5 Pro 参与关键科学决断时使用全局 `$gpt5pro-handoff`；回答只有经用户采纳并写入 protocol 后才成为实验合同。
- 每个符合 [Agent Note rules](.agents/notes/README.md) 定义的 `non-trivial` change 必须在同一 change 更新已有 owner Note，确无 owner 时才新建；Note 不能替代 scientific protocol。
- registry binding 或 governance verifier 通过均不证明 scientific pass、下游授权或 formal claim 成立。

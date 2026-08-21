# Validation layers

标准 Python：

```bash
/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python
```

选择验证时先看变更真正落在哪个 owner；注册了某个 test path 不表示该 test 已在本次工作中运行或通过。

| Change | Minimum direct check | What it does not establish |
| --- | --- | --- |
| 文档、authority map、registry 或 release map | `scripts/spec/verify_spec_system.py` 与 `git diff --check` | scientific Gate、runner 语义或 artifact 内容。 |
| verifier 自身 | 上述检查加 `python -m pytest -q tests/test_spec_system.py` | 任一实验 runner 的 contract。 |
| config、runner、Gate、authorization | registry 绑定的 self-contained contract/negative tests；只有 binding、source inventory 或治理 owner 同时改变时才加 spec closure | sealed artifact 的新结果。 |
| 公共数值逻辑、schema、生成器、跨模块基础设施 | owning tests 与可识别的受影响下游 tests；影响无法可靠缩小、跨多个实验族或用户要求时才运行全量 pytest | formal run 或外部 artifact closure。 |
| artifact、resume、publication 或 formal claim | protocol preflight、对应 artifact checks 与 runner tests | 长任务是否真的已启动/完成。 |

## Structural closure

默认命令：

```bash
/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python scripts/spec/verify_spec_system.py
```

它检查 catalog 在各 definition 的 source commit 内是否自洽、current pointer 结构及其余 repository-governance surfaces；不要求历史 binding 仍存在于当前 checkout，也不读取 ignored `runs/` 或认证科学结果。standing-doc 超预算只产生 warning，`--list-budgets` 显示 usage；`--strict-history` 是当前 checkout 的历史来源审计；`--committed` 只证明 Harness authority 与记录已持久化；第一次真实归档由 `--seal-archive` 创建 manifest。

## Contract, artifact and formal checks

自包含 runner tests 应使用临时目录/合成输入或已追踪 source，不把缺失 artifact 的 skip 说成通过。需要 `runs/` 的检查必须同时说明 artifact root、source/config/runtime closure 和缺失处理。当前工作树的绑定边界记录在 [activation audit](audits/2026-08-20-research-harness-activation.md)；默认 verifier 通过不会把 pilot artifact 变成 formal evidence。

会启动完整协议、大量计算或长时间进程的验证属于 formal/longrun 范围。预计超过五分钟时遵从根 AGENTS 的 `$long-wait` 触发条件和 [长任务 SOP](长任务防中断运行SOP.md)；普通测试命令不得被报告成 formal validation。没有完整 marker 分类时，也不要用一个宽泛的 pytest marker 表达“所有 artifact/formal 测试均已排除”。

交付时始终列出已运行命令、结果、未运行检查及原因。

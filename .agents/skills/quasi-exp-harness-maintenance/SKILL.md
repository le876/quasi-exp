---
name: quasi-exp-harness-maintenance
description: Maintain quasi-exp authority, static experiment definitions, current pointers, standing docs, repo Skills, governance verification, persistence, and owning Agent Notes. Use when those governance surfaces change; not for routine experiment execution or ordinary scientific code and tests.
---

# Quasi-exp Harness Maintenance

维护 repository memory 与治理结构；科学方法、Gate 和结果仍由 protocol、config、runner、tests 与 sealed artifacts 拥有。

## 工作流

1. 绑定实际 worktree、branch、HEAD、dirty state 和 live-process boundary。
2. 以用户指定路径或当前 scoped diff 为范围；不要自动扩大为全仓 prose audit。
3. 从 docs/README.md 找唯一 owner，只读取本次变化需要的 registry entry、文档和 Note。
4. 使用 docs/README.md 定义的 governance_structure、scientific_contract、persistence 三轴；只有任务需要科学合同判断时才使用 $quasi-exp-scientific-contract-review，持久化轴由 --committed 检查。
5. 只改权威 owner 及必要指针；不在多个文档复制同一命题。
6. 收尾时按 [Agent Note rules](../../notes/README.md) 对 scoped diff 分类；先更新已有 owner，没有合适 owner 时才新建，不在本 Skill 复制触发定义。
7. implemented Note 跟随当前现实；只同步该 Note 已经陈述、或用于定位其决定的事实，不追加 change history。Note 只保存其他 owner 无法表达的 durable rationale。
8. 运行最窄相关 verifier/tests 和 git diff --check，只报告实际执行结果。

## 维护边界

- Registry、current-state、release map、protocol 与 artifact 的边界只以 [authority matrix](../../../docs/README.md) 为准；本 Skill 不复制字段清单。
- AGENTS.md 只保留常驻规则和按任务路由；详细 SOP 放入 Skill 或唯一 owner。
- 普通实验执行和 ordinary scientific code/test change 不因本 Skill 自动增加 governance checks；是否需要 Note 只按 Agent Note rules 判断。
- 精简时遵循 docs/AGENTS.md 的适用范围与排除项，并保留完整命题。
- standing-doc budget 只用于发现膨胀；超限 warning 不阻塞 governance 或实验，结构错误仍失败。
- Audit、review、diagnose 请求保持只读。

## 验证选择

默认运行：

~~~bash
/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python scripts/spec/verify_spec_system.py
~~~

`--list-budgets` 只报告 usage；历史来源审计加 `--strict-history`；只有要证明 Git 持久化时才加 `--committed`。归档任务改用 $quasi-exp-archive-agent-notes。不要默认刷新 live state、读取 ignored runs/、启动实验或运行全量 pytest。预计超过五分钟的已授权命令仍由 $long-wait 唯一管理。

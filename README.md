# quasi-exp

`quasi-exp` 用解析运动学、准静态力学、数值优化和 BACRA workspace-inverse 方法复现并扩展连续体机器人数据集生成。仓库同时承担科学协议、可执行实验、provenance closure 与训练/评测代码；任何结果主张都必须绑定具体 source、config、runtime 和 artifact。

## 从这里开始

- 当前主线、attempt 与 locator：[docs/current-state.md](docs/current-state.md)
- experiment definitions：[spec/registry.yaml](spec/registry.yaml)
- 系统地图：[ARCHITECTURE.md](ARCHITECTURE.md)
- 领域术语：[CONTEXT.md](CONTEXT.md)
- 文档权威关系：[docs/README.md](docs/README.md)
- Codex 工作约束：[AGENTS.md](AGENTS.md)

当前状态页是导航，不是科学证据。先由它选择 registry definition，再从所指 runtime/artifact 读取现场信息和结果并核对 source/provenance seal。

## Harness 变更验证

```bash
/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python scripts/spec/verify_spec_system.py
/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python -m pytest -q tests/test_spec_system.py
git diff --check
```

这些命令验证 repository harness，不是每次实验或 scientific code change 的固定前置。其他任务按 [docs/testing.md](docs/testing.md) 选择最窄相关检查。正式长任务必须遵循 [长任务防中断运行 SOP](docs/长任务防中断运行SOP.md)，不得把普通 pytest 或文档更新当作 formal scientific validation。

## 重要边界

- YAML config 与明确的 runner arguments/defaults 是 executable value owner；知识体系 Markdown 不新增阈值副本。
- protocol 规定允许的方法、Gate 语义和注册值的冻结理由；既有 protocol 中的 immutable 数值证据保留，用过的 formal protocol 不原地回改。
- runner 与 tests 执行和证明 fail-closed contract。
- `data/`、`runs/` artifact 拥有实际结果；`docs/current-state.md` 只指向它。未封存 artifact 不被 Harness 升级为 formal evidence。
- `spec/release-map.yaml` 分开记录 scientific source 与 filtered public release。
- `data/`、`runs/` 和活动实验 worktree 默认只读。

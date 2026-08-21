# quasi-exp

`quasi-exp` 用解析运动学、准静态力学、数值优化和 BACRA workspace-inverse 方法复现并扩展连续体机器人数据集生成。仓库同时承担科学协议、可执行实验、provenance closure 与训练/评测代码；任何结果主张都必须绑定具体 source、config、runtime 和 artifact。

## 从这里开始

- 当前主线、attempt 与 locator：[docs/current-state.md](docs/current-state.md)
- experiment definitions：[spec/registry.yaml](spec/registry.yaml)
- 系统地图：[ARCHITECTURE.md](ARCHITECTURE.md)
- 领域术语：[CONTEXT.md](CONTEXT.md)
- 文档权威关系：[docs/README.md](docs/README.md)
- Codex 工作约束：[AGENTS.md](AGENTS.md)

当前状态页和 registry 都是导航，不是科学证据；结果与 Gate 必须从所指 runtime/artifact 读取并核对 source/provenance closure。完整 owner 边界见 [documentation map](docs/README.md)。

## Harness 变更验证

```bash
/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python scripts/spec/verify_spec_system.py
/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python -m pytest -q tests/test_spec_system.py
git diff --check
```

这些命令验证 repository harness，不是每次实验或 scientific code change 的固定前置。其他任务按 [docs/testing.md](docs/testing.md) 选择最窄相关检查。正式长任务必须遵循 [长任务防中断运行 SOP](docs/长任务防中断运行SOP.md)，不得把普通 pytest 或文档更新当作 formal scientific validation。

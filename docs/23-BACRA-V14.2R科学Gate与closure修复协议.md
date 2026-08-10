# BACRA V14.2R 科学 Gate 与 closure 修复协议

## 1. 适用范围

本协议从 `697e1d9` 的 work-conserving 调度固定点向前应用。旧 retry2 属于因性能与
聚合缺口而中止的历史运行；`697e1d9` patch-07 benchmark 只作为调度性能基线，不能
授权本协议修订后的 Formal。新科学运行使用独立的 retry4 output root。

## 2. repeat 证书

每个方向的三次注册扰动 continuation 继续独立执行。repeat P95 不超过
`0.2 deg` 同时进入 chart qualification、fragment repair 和 fresh primary
certificate。repeat-only failure 不得出现 `certificate_gate=true`，也不得延迟到
patch 最终 Gate 才被发现。patch-07 failure localization 同时覆盖 solver、geometry
和 repeat failure。

## 3. registered retry chain

只允许以下 chain：

- R0: `predictor -> bounded_ls`；
- R1: `weighted_dls -> bounded_ls`；
- R2: `weighted_dls -> bounded_ls -> slsqp`。

未知或变更的 chain fail closed。每条 execution 保存 registered chain、executed
chain 和 chain SHA。该修复冻结并审计既有数值路径，不擅自更换 corrector 算法。

## 4. fixed-point closure

inventory 递归验证 retry4 与 V14.2 上游 manifests 声明的全部 artifacts。runtime
closure 至少绑定 Python、解释器、平台、NumPy/SciPy/Pandas/PyArrow/sklearn/
TensorFlow 版本与 NumPy build/BLAS 配置，并进入 inventory、growth checkpoint 和
audit bundle input hash。

patch 完成只由最后写入的 `completion_manifest.json` 认定。该 manifest 绑定
source/config/runtime/input、report 及全部科学 artifacts；单独存在 `report.json`
不得触发 resume skip，任一 artifact 被修改均使 patch 重新进入执行路径。

顶层 stage 也必须在 `gate.json` 之外有最后写入的
`completion_manifest.json`。launcher 每次断点恢复都通过 runner 核验
source/config/runtime 与 stage artifacts；不得仅凭 `gate.json` 存在跳过。

inventory 保存 reviewed files 及 combined tree hash。在文档过滤后的公开
snapshot 尚未生成时，`release_sha` 必须明确为空并标记
`pending_filtered_publication`；不得把 scientific SHA 重复填入伪装成已验证的
public mapping。

## 5. summary 语义

summary 明确分开：

- `operational_completion`；
- `artifact_manifest_complete`；
- `scientific_gate_pass`；
- `deployment_authorized`。

summary 的 `gate_pass` 使用科学语义，不得再用“manifest 非空”替代科学通过。

## 6. 性能准入

新 source SHA 必须在独立 output root 重跑 patch-07 benchmark。除 exact-set、source/
config/runtime closure 和绝对时间 Gate 外，所有 required audit phases 必须有 12 个
complete shard 进度证书，且 start/finish 区间证明实际最大并发为 12。每个 shard
保存 wall/CPU time，每个 patch 保存 input、growth、atlas repair/audit、strong
reference 和 artifact write 分阶段耗时。

## 7. 暂缓的优化

root-chart 并行、cost-balanced sharding、Reach scalar multiprocessing、vectorized
FK 和显式 Jacobian 均需独立等价性与 scaling benchmark。它们不得与本轮科学 Gate
修复混为一个实验变量；尤其不能缓存 repeat、reverse、fragment re-audit 或 fresh
primary certificate 的数值结果。

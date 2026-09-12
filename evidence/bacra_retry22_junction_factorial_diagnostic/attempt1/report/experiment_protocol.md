# BACRA retry22：交界采样与 β 标签牵制的配对对照

2026-09-13，用户授权针对已怀疑问题做实验验证。前置只读审计见[壳内扩充与两域衔接](../audits/2026-09-13-retry21-shell-junction.md)。本轮在扩充监督数据前，检验“接缝失败能否仅通过抽样曝光改变”和“β 标签项是否牵制 FK 优化”。不把去掉标签损失等同于证明 Teacher 标签错误，不把补采样等同于新增几何覆盖。

## 冻结分母与预算

只读复用 retry21 attempt2 的已验证输入/35条有序路径、3000 off-grid test，以及 retry20 L0/L1。L1 74696行，train/validation/test=58629/7795/8272；L0 train仅用于归一化。原35路径/全test与先前失败位置已用于选研究方向，是复用诊断，不称新盲测。不得修改坐标、split、标签、原行权重或旧 sealed artifacts；新增监督为0。

沿用35条路径的预检 registry 和几何尺寸，因为目标完全相同；新建本实验身份、预算与输入来源的六文件合同并调用通用校验。学习成功仍not_bounded，diagnostic_only。推理无IK/DLS，无专家β平均，不授权全域替换、部署或正式连续空间主张。

每seed先新训500步beta-only warmup，然后四组各从完全一致的模型、Adam、全部采样随机状态分叉，继续到10000步。两seed共 `2*(500+4*9500)=77000` 次optimizer更新，最多7200秒。复用上游数据/审计/冻结轨迹，不复用旧final模型作为配对起点，因为其没有本轮四组所需的共同seed2 warmup状态。训练完成后以重载.keras评估。

## 唯一干预与公平性

固定现有[128,128,64] GELU网络、signed XYZ/L0归一化、有界输出、501步启用零位结构、Adam epsilon1e-7、batch1024；lr=.001至6000步，.0003至9000步，随后.0001。500步warmup只用原beta loss和均匀抽样。

| arm | 501步起采样 | loss |
| --- | --- | --- |
| control | core205+transition256+tip154+uniform409 | task + .01 beta |
| junction | core205+transition256+tip154+uniform281+junction128 | task + .01 beta |
| task_only | 与control一致 | task |
| junction_task_only | 与junction一致 | task |

task和beta的数值定义保持retry21不变。junction pool只从train中取 `abs(X-1015.498)<=60 mm` 的点，覆盖真实交界及既有坏点邻域，所有rho和方位保留。它是基于旧诊断选定的干预区域，不能当独立发现集。各pool内无放回，跨pool允许重复。junction组保留原regions抽样的前896行，以独立seed+20000随机流替换最后128个uniform样本；其余随机流按同样次序消耗，使相同采样组在不同loss下得到完全相同batch。训练前冻结train pool及test的k32混合域mask与±5/10/20mm带。

保存每seed四组初始state SHA及完整batch序列SHA；相同sampling的两loss组batch SHA必须相同。control额外对照旧retry21同seedfinal预测，量化复现差异但不强求跨实现浮点逐位一致。每500步只记录validation；固定final-step，无early-stop、无test选模、无结果后续加训。

## 判断与终止

所有模型锁定后，参考scalar FK重算完整train/validation/test、35路径、3000offgrid、精确零位。保存P50/P95/P99/max、≤3/10mm与bounds。单独报告完整test与混合邻域的>50/>100mm点数、相对配对control逐点增量、原严重坏点的修复与新增坏点，以及八条近轴路径是否仍全点≤3mm。

预先登记“两个seed完整test的>50mm点数均至少减半且原八条近轴路径全点≤3mm”为本轮**尾部改善候选判据**。这是诊断筛选，不是global pass；任何seed仍有大误差、任何原路径退化均完整报告。以配对seed分别报告采样效应、去β项效应及二者交互，不把两个seed当统计显著性的充分样本。若control已无>50mm点，不能宣称对应干预使其减半，记该项不适用。

采样有效支持曝光/权重是可干预因素；去β项有效支持该损失约束参与失败，不唯一证明分支冲突；均无效则不能据此证明必须扩数据，仍有优化/容量因素。无论结果如何，当前实验没有新增厚度，不能证明增厚后的全域能力。

非有限loss/gradient立即停止并保存failure；7200秒到时保存partial checkpoint并终止，partial不作final。使用fresh long_wait_monitor及既有longrun_tmux.sh独立状态/会话；不改旧进程。完成后独立复算关键指标，交付报告和source/binding/public provenance。

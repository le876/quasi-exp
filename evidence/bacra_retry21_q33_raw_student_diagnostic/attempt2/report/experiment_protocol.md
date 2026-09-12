# BACRA retry21：Q33 raw Student 原框架复核与 outer 定位

## 授权与方法来源

用户于 2026-09-12 采纳 Q33 回答，确认本轮终点为“原框架复核并定位 outer”，并授权执行计划。本实验独立于 retry20，采用 Q33 第二轮候选方法而不覆盖旧产物。Q33 的 CPU/PyTorch 数值属于外部诊断，不冒充本地 TensorFlow 结果。原咨询材料保留在用户提供的 `docs/Q33回复`；配置绑定原 ZIP 的 SHA-256，runner 验证其 96 个 manifest 成员，仅提取证据，不执行外部脚本。

合法 IK 标签的 beta 均值不保证合法 FK，是本轮待复核机制；它不自动证明所有 outer 退化由分支冲突引起。实验只授权诊断，不授权全域模型替换、canonical Teacher 蒸馏完成、full-workspace、力学可行性、硬件或闭环控制主张。

## 冻结输入与分母

复用 retry20 candidate attempt1 的 L0/L1 各 74,696 行（train 58,629、validation 7,795、test 8,272），原 L0/L1 锁定模型及原16条路径和3,000个 off-grid test；加 Q31 attempt1 的3条大轨迹、Q33附件的16条尺寸/位置对照，共35条完整有序路径。哈希、split、target ID、坐标和原始权重冻结，不新增监督、不重求训练标签、不平移或缩小轨迹。只使用原 L1 train 监督、原 L0 train 归一化。

00_objective_feasibility 冻结路径与点的分母、训练数据复用、30,500次更新与7,200秒预算。训练预算无法保证所有路径达标，required objective 使用 not_bounded，gate 为 diagnostic_only。原路径、旧test和Q33失败点已用于研究方向选择，属于复用诊断，不称新盲测。附件 matched-controls 可达性仅为外部证据，本地不重复 IK 验证或将其用作标签。

## 模型、训练与选择

signed XYZ D0、L0 train population normalization、hidden [128,128,64] GELU、有界 beta，axial/radial scale600/220mm，batch1024。beta loss 为 sample-weighted sum([16,16,4,4,1,1]*(prediction-label)^2)；NaN sample_weight 按原合同填1。任务项为 sample-weighted sum(((reference-compatible differentiable FK(prediction)-xyz)/0.1m)^2)。组合loss=0.01*beta loss+task loss，不单独声称分离了下调标签权重与增加FK项的贡献。

第一轮：seed20260925，新训500步beta-only warmup，四组从完全相同的模型、Adam及采样状态分叉，各继续2500步至3000：beta_uniform、beta_balanced、fk_uniform、fk_balanced。balanced为256行core加768行全train；其余均匀。Adam epsilon1e-7，lr.001，2501步起.0003。保存并比较分叉状态hash。原锁定模型没有完整训练Adam状态，仅作为评估基线。

第二轮：seed20260925和20260926各从头独立训练10000步，前500步普通beta监督；501步起使用 beta=b*tanh(h(phi(p))-h(phi(p0)))，启用组合loss，保留优化器状态。学习率.001，6001步起.0003，9001步起.0001。每batch：core205、transition256、tip154、全train409。定义u=(p0_x-x)*1000mm，rho=norm(y,z)*1000mm：core为60≤u≤200,rho≤80；transition为60≤u≤200,80<rho≤280；tip为0≤u<60,rho≤160，零附近仅容忍1e-6mm浮点偏差。各pool内无放回，不同pool可与均匀部分重合，原行weight不变。

两轮合计30500次optimizer updates。每100步更新进度，每500步保存validation全域与分区指标。固定final-step选模，无early stopping，无测试选模。各模型保存.keras、Adam与采样状态；最终评估使用重载模型。零位为结构恒等式；推理为XYZ到beta的直接有界前向，不做IK/DLS、后投影、配准或专家beta软平均。零位latent在训练中随权重更新，不使用过时cache。

## 数值验证与诊断

训练FK采用项目已有TensorFlow实现；最终指标以原参考scalar FK重算。先核验镜像解反例，再对35条路径和整个on-grid test执行train-only k1/2/4/8/16/32标签均值诊断：FK误差、beta范数收缩、Jacobian条件数。train全图k16邻接记录primary–outer跨域边、端点残差、beta gap和beta中点相对于XYZ弦中点的FK失真；此项不是连通性认证。

所有模型锁定后评估完整train/validation/test、35路径、3000 off-grid与精确零位。报告P50/P95/P99/max、有限值分母、非有限数、全分母≤3/10mm比例、bounds；路径另报质心偏移、去质心尺度比、平面误差、beta相邻跳变及端点gap。缺失点不删分母。

outer诊断固定为附件两种子test top40，以及本地两种子相对原L1退化最大各40个outer点的并集；保留split身份，不加入训练。每点取32个train邻居，保存domain、距离、beta，比较最近标签与其余邻居在alpha=.25/.5/.75的插值失真，并比较原Teacher和所有raw模型的beta偏离、范数、FK和Jacobian。每个分区报告逐点误差增量及大于1/10mm的退化数，完整尾部保留。

两种子八条近轴轨迹各自全点≤3mm才报告复现Q33该项结果。outer退化按实际结果报告，不改门槛或追加训练；本轮全域替换始终未授权。第二轮是结构、采样与预算的组合复核，不冒充等预算部件消融；下一轮是否需要局部分支仍待证据。

## 执行、终止与交付

训练前通过owning tests及可识别下游测试，提交source并登记binding；独立工作区、fresh output root。使用long-wait和现有longrun_tmux.sh，独立launcher状态目录与session避免影响旧进程。总任务以7200秒timeout约束，runner在训练步间检查同一上限并保存中断checkpoint；到时不续预算，中途模型不作final。异常保留failure/progress，非有限loss或gradient停止。

完成后保存model_lock、run_identity、逐点Parquet、CSV、summary及completion manifest。报告使用项目数据集可视化Skill，复用原监督数据展示空间分布和35条路径的raw结果，说明Teacher与matched-controls证据边界；不生成PDF。验收后按Harness在隔离发布worktree发布实现与HTML报告，保留旧公开evidence。operational completion、artifact completeness、局部复现结果及formal授权分别报告，测试或HTML通过不替代科研结论。

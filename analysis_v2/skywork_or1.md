skywork_or1 | Skywork Open Reasoner 1 Technical Report | Skywork AI(昆仑万维 Kunlun Inc),Jujie He、Jiacai Liu 共同一作,通讯 Jujie He | 2025-05-29 arXiv v2(2505.22312)·技术报告+Notion 博客 | 主题线 L3(RLVR/长 CoT 模型 RL)·相关性 高

**原始论文**:https://arxiv.org/abs/2505.22312

## 一眼看懂
- 🟦 TL;DR:面向"已蒸馏的长 CoT 模型"(DeepSeek-R1-Distill 系列)的**高效可扩展 RL 配方**,在改造版 GRPO(命名 **MAGIC** = Multi-stage Adaptive entropy scheduling for GRPO In Convergence)上从 数据/训练策略/损失 三方面组合多项设计并逐组件消融;系统研究**过早熵坍缩**,论证"缓解过早熵坍缩对提升测试性能至关重要"。32B 在 AIME24/25+LiveCodeBench 平均 57.8→72.8(+15.0%)、7B 43.6→57.5(+13.9%),代码/数据/权重全开源【原文 abstract+§1】。
- 最巧的一步:**Adaptive Entropy Control(自适应熵控制)**。它用 target-entropy(tgt-ent=0.2)给策略熵托一个动态下界——抽掉它(或其等价的 clip-higher)则熵过早坍缩、过度 exploitation、测试性能变差(§4.5)。这是把"维持探索"操作化的核心机制,也是全文"熵坍缩研究"的落脚点。

## 为什么做
- 研究背景:DeepSeek-R1 证明 online RL + 简单规则奖励即可大幅提升 base 模型推理(此前 RL 多靠 MCTS/PRM)。R1-Distill 系列生成的长 CoT(AIME24 上平均超 10K token)远长于 Qwen2.5/Llama3.1。已有复现(Logic-RL/Open-Reasoner-Zero/DAPO/VAPO)多聚焦对 **base 模型** RL【原文§1】。
- 解决的具体痛点:① 如何高效、可扩展地用 RL 提升**已做过 SFT 的长 CoT 模型**仍不清楚;② DeepScaleR/Light-R1/DeepCoder 等虽有初步进展,但**未系统拆解各算法组件**在 RL 训练中的独立贡献;③ 训练普遍出现 premature entropy collapse(过早熵坍缩、过度 exploitation),成因与缓解缺系统研究【原文§1】。
- 相关工作 & 各自不足(本轮按引文链补全):
  - **对 base 模型的 zero-RL 复现**:Logic-RL、Open-Reasoner-Zero、**DAPO**(贡献 clip-higher/dynamic sampling/token-level loss,被 MAGIC 部分纳入对照)、**VAPO**——证 RL 有效但**未迁移到长 CoT 蒸馏模型**。
  - **对长 CoT 模型的早期 RL**:**DeepScaleR**(多阶段长度调度 8K→16K→24K,**被 MAGIC 的 Multi-Stage Training 直接借鉴**)、Light-R1、DeepCoder——开始探索但**未系统拆组件**,也未系统研究熵坍缩。
  - **探索-利用/熵**:DeepSeek-Math/GRPO 原始工作给 group-norm + clip + k3-KL;本文把 DAPO 的 clip-higher 当"缓解熵坍缩"的对照手段之一(§4.5),并提出自适应熵控制作为替代。
  - 精确差异:vs DeepScaleR——MAGIC **系统消融了 6 个组件的独立贡献**(数据 mixture/多阶段/advantage mask/高温/自适应熵/无 KL)+ **专门做了熵坍缩的多角度对照研究**(熵坍缩速度 vs 性能、off-policy 影响、batch/温度影响、缓解手段),并得出几个反直觉结论(如"对截断响应加负 advantage 的 advantage mask 反而不利于大 context 长度的 scaling,故不用")。
- 动机链:R1 证明 RL 有效但"对已蒸馏长 CoT 模型怎么做"不清楚 → 已有复现未拆组件贡献、且普遍遇熵坍缩 → 提出 MAGIC 配方并逐组件消融 + 系统研究熵坍缩 → 证"缓解过早熵坍缩=提升测试性能关键" + 全开源【原文§1 末贡献列表】。

## 怎么做 + 靠不靠谱
- 方法流水线(MAGIC,§3.1 三类,输入→输出):输入已蒸馏长 CoT 模型(R1-Distill 7B/32B)→ ① **数据收集**:严格预处理+更准验证器;Offline+Online 过滤(训练前去掉 base 正确率为 0/1 的题;每阶段开始丢弃上阶段已全对的题,保持只训难题)+ **Rejection Sampling**(batch 只保留含非零 advantage 的组,见 \(\tilde T_k\) 定义)→ ② **训练策略**:Multi-Stage Training(逐阶段增大 context \(T\),先短后长 8K→16K→…)；**不用任何 advantage mask**；High-Temperature Sampling(τ=1)；On-Policy Training(7B/32B 严格 on-policy)→ ③ **损失函数**:去掉 GRPO 的 \(1/|y_{ij}|\) 长度归一项→token 级 policy loss + entropy loss；Adaptive Entropy Control(target-entropy 动态调 \(\alpha_k\) 托住熵下界）；No KL Loss → 输出 Skywork-OR1-{Math-7B,7B,32B}【原文§3.1】。
- 逐组件必要性(均有独立消融 §3.2.1-3.2.6):
  - **数据 mixture(§3.2.1)**:严格过滤的 mixture 优于松阈值 baseline mixture(后者只是学得更慢,非更差)——有消融✓。
  - **Multi-Stage(§3.2.2)**:初期省算力、后期保 scalability;同样最终精度但 token 效率显著高(平均长度 12.5K→5.4K)——有消融✓。
  - **Advantage Mask(§3.2.3,关键反直觉)**:对截断响应赋负 advantage 看似合理(避免噪声信号),但实验证其**不利于大 context(如 32K)的 later-stage scaling**,故"we do not employ any advantage mask"(原文 §3.2.3 直引)——有消融✓,结论是"不用"。
  - **High-Temp(§3.2.4)**:τ=1 早期测试精度低但最终增益更大(τ=0.6 时 math 立即/code 很快进入低熵态)——有消融✓。
  - **Adaptive Entropy Control(§3.2.5)**:把熵 lower-bound 在 target-entropy,维持探索与高 plasticity,测试性能稳步升——有消融✓(核心组件)。
  - **No KL Loss(§3.2.6)**:KL 罚在多阶段后期妨碍提升,故去掉——有消融✓。
- 关键机制/公式(本轮据 PDF 正文 Eq.2.1-2.5/3.1 补全,真符号 MathJax):
  - **RL 目标(Eq.2.1)**:\(\max_\pi J(\pi)=\mathbb{E}_{x\sim D}\,\mathbb{E}_{y\sim\pi(\cdot|x)}[r(x,y)]\),batch 级代理(Eq.2.2)。
  - **vanilla PG(Eq.2.3)**:\(L^{\text{PG}}_k(\theta)=-\mathbb{E}\big[\sum_t \tfrac{\pi_\theta(a^t_i|s^t_i)}{\pi_k(a^t_i|s^t_i)}A^{\pi_k}(s^t_i,a^t_i)\big]\)。
  - **GRPO(Eq.2.4)**——MAGIC 的出发点,含长度归一与 k3-KL:
    \(\displaystyle L^{\text{GRPO}}_k(\theta)=-\mathbb{E}\Big[\tfrac1M\sum_{i=1}^{M}\tfrac{1}{|y_{ij}|}\sum_{t=0}^{|y_{ij}|-1}\min\big(\rho^t_{ij}A^t_{ij},\ \mathrm{clip}(\rho^t_{ij},1-\varepsilon,1+\varepsilon)A^t_{ij}\big)-\beta D^t_{ij}(\theta)\Big]\)
    其中 \(\rho^t_{ij}=\dfrac{\pi_\theta(a^t_{ij}|s^t_{ij})}{\pi_k(a^t_{ij}|s^t_{ij})}\),**k3-KL 罚** \(D^t_{ij}(\theta)=\dfrac{\pi_{\text{ref}}(a^t_{ij}|s^t_{ij})}{\pi_\theta(a^t_{ij}|s^t_{ij})}-\log\dfrac{\pi_{\text{ref}}(a^t_{ij}|s^t_{ij})}{\pi_\theta(a^t_{ij}|s^t_{ij})}-1\),系数 β。
  - **组归一 token 级 advantage(Eq.2.5)**:\(\forall t:\ A^t_{ij}=\dfrac{r(x_i,y_{ij})-\mathrm{mean}(r(x_i,y_{i1}),\dots,r(x_i,y_{iM}))}{\mathrm{std}(\cdots)}\),binary 奖励 \(r\in\{0,1\}\) 由规则验证器给。
  - **Rejection Sampling 集合**:只保留 \(\tilde T_k=\{i\in[N]:\exists j\in[M],\ \hat A_{ij}\ne0\}\)(零优势组不进 batch,避免它们隐式抬高 KL/entropy 罚的相对权重致不稳)。
  - **MAGIC 损失(Eq.3.1,两处关键改动)**:
    \(\displaystyle L^{\text{MAGIC}}(\theta)=-\frac{1}{T_k}\sum_{i\in\tilde T_k}\sum_{j=1}^{M}\Big[\sum_{t=0}^{|y_{ij}|-1}\min\big(\rho^t_{ij}A^t_{ij},\mathrm{clip}(\rho^t_{ij},1-\varepsilon,1+\varepsilon)A^t_{ij}\big)+\alpha_k H^t_{ij}(\theta)\Big]\)
    ① **去长度归一**:把原 GRPO 的 \(1/|y_{ij}|\) 删掉,改成对**全 batch token** 求平均(\(T_k=\sum_{i\in\tilde T_k}\sum_j|y_{ij}|\) 为 batch 总 token 数),消除长度偏置;② **加 token 级熵项** \(\alpha_k H^t_{ij}(\theta)\),\(H^t_{ij}=H(\pi_\theta(\cdot|s^t_{ij}))\),系数 \(\alpha_k\ge0\) 由自适应熵控制动态调;③ **无 KL**(去掉 Eq.2.4 的 \(\beta D^t_{ij}\) 项)。
  - **Adaptive Entropy Control(§3.2.5,机制)**:引入超参 tgt-ent;\(\alpha_k\) 按"当前熵与 tgt-ent 之差"动态调整,确保当前熵被 tgt-ent **托住下界**(熵跌破下界→自动加大熵奖励系数把它顶回)。直觉:熵=探索度;这是把"clip-higher 等手动 trick"换成"目标熵反馈控制器"。〔主文给的是机制描述与 tgt-ent=0.2,**未给闭式 \(\alpha_k\) 更新方程**,故此处不杜撰其解析式。〕
- 实验与证据:
  - 数据集/设置:训练=自建 mixture(严格难度过滤,含从 NuminaMath-1.5 过滤的 hard 题),对照 DeepScaleR mixture(AIME/AMC/Omni-MATH/STILL);含数学+代码,配 Math Verifiers + Code Sandboxes。评测=AIME24、AIME25、LiveCodeBench(2024-08~2025-02),主指标 avg@K（非 pass@1）。基座=R1-Distill 7B/32B。消融设置(Table 1):batch 64 / mini-batch 32 / group 16 / Entropy Control=tgt-ent 0.2 / KL=No(基于 R1-Distill-Qwen-7B)【原文§6-§8+§3.2】。
  - 关键数字【原文 abstract+§1,本轮 PDF 直读确认】:32B 57.8→72.8(+15.0)、7B 43.6→57.5(+13.9);32B 具体 AIME24 82.2 / AIME25 73.3 / LiveCodeBench 63.0,超 DeepSeek-R1 与 Qwen3-32B(math)、LCB 持平;7B AIME24 70.2 / AIME25 54.6 / LCB 47.6;Math-7B AIME24 69.8 / AIME25 52.3 / LCB 43.6。多阶段:同最终精度但平均长度 12.5K→5.4K、省算力。
  - baseline 公平吗:消融均在自建 mixture + R1-Distill 同基座下做内部对照,公平;但与外部模型(R1/Qwen3-32B)的 SOTA 比较是终点比较,非控制变量。
  - 看着强但没回答核心:工程报告属性强,单个组件多借鉴已有(DeepScaleR 多阶段、DAPO clip-higher),真贡献在"组合+系统消融+熵机理+开源";诸多结论(高温影响大、batch/group 影响小)绑定特定 mixture+R1-Distill 基座。
- 假设与失效边界:
  - 显式【原文】:On-policy mitigates entropy collapse(§4);off-policy(增大 mini-batch/数据复用 \(N_{\text{SGD}}\))加速熵坍缩并劣化性能(§4.4);batch/group 增大对熵动态影响小、温度影响大(§4.3+§3.2.4)。
  - 隐式【推断】:"advantage mask 反而不利 scaling"依赖其多阶段长度调度场景,外推到非多阶段设置需谨慎;tgt-ent=0.2 等关键超参缺跨基座/任务的可迁移性论证;评测面集中 AIME+LiveCodeBench,泛化基准较窄。
  - 一致性瑕疵【原文】:Skywork-OR1-Math-7B 用**两步梯度更新**(非严格 on-policy),与 7B/32B 严格 on-policy 不一致;论文§3.1 自承"此设置先于我们对 off-policy↔熵坍缩关系的完整理解",但靠自适应熵控制仍达强性能——削弱"on-policy 减缓熵坍缩"在 Math-7B 上的纯净性〔本轮 PDF §3.1 已直读确认此细节,原 v1 标"推断"可上调为【原文】〕。
- 祛魅总结【推断】:真贡献=面向长 CoT 模型 RL 的可复现配方(MAGIC)+ 6 组件逐项消融 + 熵坍缩的多角度实证研究(熵坍缩速度 vs 性能、off-policy/温度/batch 对熵的影响、自适应熵 vs clip-higher 缓解)+ 全开源(代码/数据/权重)。包装/高估:abstract 主推 +15/+13.9 与"超 R1/Qwen3-32B",但多数组件是已有 trick 的组合;"高温影响大、batch 影响小"等结论的普适性受限于其特定数据+基座。低估:熵坍缩研究本身(§4)作为独立科学贡献(把"过早熵坍缩→更差性能"做成可观测规律)价值高于单纯刷分。

## 结构化抽取
- 🎯 机制速览6轴:
  - **学什么信号**:规则奖励(binary 0/1,Math Verifier + Code Sandbox)→ 组归一 token 级 advantage(Eq.2.5);辅以熵信号(当前熵 vs target-entropy)调 entropy loss 系数 \(\alpha_k\)。
  - **改什么**:策略模型 \(\pi_\theta\) 全参数(改造版 GRPO/MAGIC 更新)。
  - **何时改**:多阶段 online RL 全程(逐阶段增大 context \(T\):8K→16K→…);每步按熵差动态调熵系数。
  - **免梯度?**:否,梯度策略优化(GRPO 变体)。
  - **记忆-技能生命周期**:无记忆/技能库;能力固化进参数;online+offline 数据过滤动态维持"在难题上训练"。
  - **防遗忘机制**:无显式跨任务防遗忘;去 KL(§3.2.6,KL 罚后期妨碍提升)→不锚定参考策略;靠 on-policy + 自适应熵防"过早熵坍缩/过度 exploitation"(可视为防"探索能力遗忘")。
- ⑦ 开源代码+框架/harness:https://github.com/SkyworkAI/Skywork-OR1 (v1 验证已克隆约 11MB,commit 64e96af)。框架=**veRL 定制 fork**(仓内自带 `verl/` + `or1_scripts/` + `or1_data/`,含 Math/Code 验证器与 code sandbox);RL 算法=改造版 GRPO（MAGIC）。数据 HF `Skywork/Skywork-OR1-RL-Data`;权重 `Skywork-OR1-{Math-7B,7B,32B}`(及 Preview)。代码/数据/权重全开源,可复现性高。
- 💰 资源/成本与可扩展性:多阶段先短后长显著省算力(同最终精度更少累计训练小时,平均响应长度 12.5K→5.4K token);§5 专论训练资源分配(固定算力下提效率 §5.1 / 增算力提性能 §5.2)。具体 GPU 卡数/总卡时正文未以单一数字给出(分散在 §5/§8 及 Fig 的"cumulative training hours")→部分原文未以汇总数列明。
- 🎯 对"探索-巩固"对标:**支撑/可借组件(探索侧)**。一句判定:本文为"探索"提供了可直接借用的工程化机制——**Adaptive Entropy Control 用 target-entropy 给熵托下界**,正是把"维持探索能力/不过早 exploitation"操作化,与本项目"探索=发现有效路径、不要过早收敛到自己已会的开头"高度契合;其"过早熵坍缩→更差性能"的实证(§4)为"巩固阶段也要保留探索"提供经验依据。可借:target-entropy 下界、on-policy 减缓熵坍缩、去长度归一的 token 级 loss、online 难题过滤。缺口:纯 RLVR 配方,无 teacher 蒸馏/无脚手架/无路径恢复/无记忆技能/无 MTP 前瞻;"巩固"仅指参数固化,无防遗忘机制。依据:§3.2.5 自适应熵 + §4 熵坍缩研究。
- 🔭 开放问题/未来方向:【原文】缓解过早熵坍缩对测试性能至关重要(主结论),可继续探索熵控制手段(自适应系数 vs clip-higher)的权衡(§4.5)。【推断】target-entropy 等超参的跨基座/任务可迁移性;把"advantage mask 反而损 scaling"的结论在非多阶段设置下重检;扩展评测面(超出 AIME+LiveCodeBench);把熵下界机制与 teacher 脚手架/OPD 结合,检验"维持探索"是否能让 student 更好地学到"路径恢复"。

RETURN: skywork_or1 | 读PDF?是(40页/102885字,abstract+§1贡献+§2 Eq.2.1-2.5 全policy梯度族+§3.1 MAGIC定义/Eq.3.1+rejection-sampling 集合+§3.2消融结论+§4熵坍缩 全直读;MAGIC 全称"Multi-stage Adaptive entropy scheduling for GRPO In Convergence"已 PDF §3 核实) | 加厚?是(补全 Eq.2.1-2.5/3.1 全套真符号公式含 k3-KL 显式形式、去长度归一+token 级熵项的 MAGIC loss、rejection-sampling \(\tilde T_k\) 定义;related-work 引文链 DAPO/VAPO/DeepScaleR 精确借鉴关系;Math-7B 两步梯度细节升为【原文】) | LaTeX 公式条数 5(RL目标/vanilla PG/GRPO Eq.2.4含k3-KL/组归一advantage Eq.2.5/MAGIC loss Eq.3.1) | 待核 0(Math-7B 两步梯度 本轮 PDF §3.1 已直读坐实,原 v1 残留待核清除)

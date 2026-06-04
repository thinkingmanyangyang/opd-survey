dapo | DAPO: An Open-Source LLM Reinforcement Learning System at Scale | ByteDance Seed + 清华 AIR(SIA-Lab) + 港大;通讯 Hao Zhou、Mingxuan Wang | arXiv 2503.14476(v1 2025-03-17，v2 2025-05-20)·技术报告 | 主题线 L3(RLVR/GRPO)·相关性 高

**原始论文**:https://arxiv.org/abs/2503.14476

## 一眼看懂
- 🟦 TL;DR:朴素 GRPO 在 Qwen2.5-32B base 上跑数学 RL 只到 30 分(AIME 2024),作者诊断出 4 个病灶——熵坍缩、无效梯度、长序列梯度失衡、超长样本奖励噪声——逐一开药(Clip-Higher / Dynamic Sampling / Token-Level Loss / Overlong Reward Shaping),组成 DAPO,把 AIME 提到 50 分且只用一半训练步就超过 DeepSeek-R1-Zero-Qwen-32B(47),并完整开源算法+代码(基于 verl)+数据【原文 Abstract, Fig.1, §1】。
- 最巧的一步:**Dynamic Sampling**(动态采样)。看 Table 1 消融,前面四项技术叠加只到 42,最后加上 Dynamic Sampling 一步直接跳到 50——这一步贡献最大(+8 分),抽掉它整个系统从 SOTA 掉回平庸【原文 §4.2 Table 1】。为什么:它过滤掉组内"全对/全错"(advantage=0、零梯度)的 prompt,保证每个 batch 都是有效梯度,避免随训练推进有效 prompt 数持续缩水(Fig.3b 显示 acc=1 的样本比例一路上升)【原文 §3.2】。

## 为什么做
- 研究背景:test-time scaling(o1/R1)靠长 CoT + 大规模 RL 撑起推理能力革命,RL 是核心引擎;但 SOTA 模型(o1、R1)的算法与配方细节被技术报告藏起来,社区难复现工业级结果【原文 §1】。
- 解决的具体痛点:作者自己在 Qwen2.5-32B base 上跑朴素 GRPO 只得 30 分(远低于 DeepSeek 报告的 47),诊断出朴素 GRPO 有熵坍缩、reward noise、训练不稳定三类问题;且 R1 论文省略了构建可复现大规模 RL 系统所需的工程细节【原文 §1】。
- 相关工作 & 各自不足(来龙去脉 + 并行路线 + 精确差异):
  - **奠基:PPO(Schulman 2017)**【原文 §2.1】。actor-critic 范式,用 GAE 估优势 \(\hat A_t=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}\),\(\delta_l=R_l+\gamma V(s_{l+1})-V(s_l)\)。短板:需训一个与策略同规模的价值网络 \(V\),显存/算力翻倍;LLM 场景奖励仅在末 token 给出,逐 token 价值难学。DAPO 站在 PPO 的 clip 机制肩上(沿用 clipped surrogate),但去掉 critic。
  - **直系前作:GRPO(DeepSeekMath 2024,源自 §2.2)**【原文 §2.2】。去 critic、用组内相对奖励当 baseline:\(\hat A_{i,t}=\dfrac{r_i-\mathrm{mean}(\{R_i\}_{i=1}^G)}{\mathrm{std}(\{R_i\}_{i=1}^G)}\)。这是 DAPO 的**直接改造对象**——DAPO=GRPO+4 补丁。GRPO 短板:在长 CoT 大规模场景暴露 4 病灶(熵坍缩、零梯度 prompt、sample-level 聚合稀释长序列、截断噪声),且其 objective 在 **sample-level** 聚合(先对每序列内 token 求均、再对样本求均),长序列被稀释。
  - **并行路线 ①——闭源工业系统(o1/R1)**:遥遥领先但训练码与配方不公开;R1 只放权重不放训练码,工程细节缺失。DAPO 的差异化定位正是"把闭源系统达到的水平用全开源算法+码+数据复现出来"。
  - **并行路线 ②——RLHF 范式(Ouyang 2022,§2.3)**:RLHF 用 KL 约束 \(-\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})\) 防止偏离初始模型。DAPO 的精确分歧:长 CoT RL 下模型分布**本就应大幅偏移**初始模型,KL 反成累赘,故 DAPO 直接去掉 KL 项(§2.3)。
  - **并行路线 ③——奖励建模(neural RM vs rule-based)**:用神经/过程奖励模型易 reward hacking(§2.4 引 [24–29]),DAPO 改用规则二值结果奖励 \(R(\hat y,y)=+1\,[\text{is\_equivalent}]\,/\,-1\)(§2.4)。
- 动机链:现状(长 CoT RL 是提升推理的核心但闭源)→缺陷(朴素 GRPO 复现只到 30 分且不稳定,工程细节缺失)→所以必须(完全开源一套达 SOTA 的系统:算法+代码+数据,把 4 个关键技术公开)【原文 §1】。
- 与最近邻工作的Δ:相对 GRPO/R1-Zero,差在"把朴素 GRPO 的隐性失败点显式拆成 4 个可独立验证的工程改造并逐项消融",且彻底开源(R1 只放权重不放训练码)。关键有用点:这 4 项是即插即用的解耦补丁,后续被 verl 收为官方 recipe、被社区广泛复用。

## 怎么做 + 靠不靠谱

### 方法总览(读完可复现)
输入 Qwen2.5-32B base + DAPO-Math-17K(17K 题,作者把 AoPS 等竞赛题答案统一转成整数以便 rule-based 校验)→ 先搭一个**去 KL + 规则二值奖励**的极简 GRPO 底座 → 叠加 4 个解耦补丁 → 输出 AIME 2024 达 50 分的推理模型。下面给每一层的输入→输出、机制、公式真实形式与默认超参。

**底座目标函数(GRPO,作为对照锚点)**【原文 Eq.5】:
\(\displaystyle J_{\mathrm{GRPO}}(\theta)=\mathbb{E}_{(q,a)\sim\mathcal D,\;\{o_i\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}(\cdot|q)}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big(\min\big(r_{i,t}(\theta)\hat A_{i,t},\;\mathrm{clip}(r_{i,t}(\theta),1-\varepsilon,1+\varepsilon)\hat A_{i,t}\big)-\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})\Big)\Bigg]\)
其中重要性比 \(r_{i,t}(\theta)=\dfrac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})}\)【Eq.6】。注意外层 \(\frac1G\sum_i\frac1{|o_i|}\sum_t\) 是 **sample-level 聚合**——这是 §3.3 要修的地方。

**最终 DAPO 目标(四补丁叠加后,Eq.8)**【原文 Eq.8/10/11/12,四式同构、逐步加约束】:
\(\displaystyle J_{\mathrm{DAPO}}(\theta)=\mathbb{E}_{(q,a)\sim\mathcal D,\;\{o_i\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}(\cdot|q)}\Bigg[\frac{1}{\sum_{i=1}^{G}|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\min\big(r_{i,t}(\theta)\hat A_{i,t},\;\mathrm{clip}(r_{i,t}(\theta),1-\varepsilon_{\mathrm{low}},1+\varepsilon_{\mathrm{high}})\hat A_{i,t}\big)\Bigg]\)
\(\displaystyle \text{s.t.}\quad 0<\big|\{o_i\mid \text{is\_equivalent}(a,o_i)\}\big|<G,\qquad \hat A_{i,t}=\frac{R_i-\mathrm{mean}(\{R_i\}_{i=1}^G)}{\mathrm{std}(\{R_i\}_{i=1}^G)}\)
对照底座式可一眼看出 4 处改动:**(a)** clip 上下界解耦为 \(\varepsilon_{\mathrm{low}},\varepsilon_{\mathrm{high}}\);**(b)** 去掉 \(-\beta D_{\mathrm{KL}}\) 项;**(c)** 外层归一从 \(\frac1G\sum_i\frac1{|o_i|}\) 改为 \(\frac{1}{\sum_i|o_i|}\sum_i\)(token-level);**(d)** 约束 \(0<|\{\text{正确}\}|<G\) 即 Dynamic Sampling 过滤全对/全错。

### 逐组件必要性(均有 Table 1 累积消融,给机制 + 公式 + 直觉)
- **① Clip-Higher**(+2,36→38)【原文 §3.1, Eq.10, Fig.2/3a】。
  - 干什么:把 PPO-Clip 的对称裁剪 \(\mathrm{clip}(\cdot,1-\varepsilon,1+\varepsilon)\) 解耦为非对称 \(\mathrm{clip}(\cdot,1-\varepsilon_{\mathrm{low}},1+\varepsilon_{\mathrm{high}})\),**只抬高上界 \(\varepsilon_{\mathrm{high}}\)**(默认 \(\varepsilon_{\mathrm{low}}=0.2,\varepsilon_{\mathrm{high}}=0.28\))。
  - 直觉(关键):上裁剪对"已经高概率的 exploitation token"几乎不设限,却把"低概率 exploration token"的上行幅度卡死——因为低概率 token 即使要翻倍也撞不到 \(1+\varepsilon\) 的天花板。放宽 \(\varepsilon_{\mathrm{high}}\) 即给低概率探索 token 解锁上升空间。作者特意**不动 \(\varepsilon_{\mathrm{low}}\)**:抬高下界会把这些 token 概率压到 0,反而坍缩采样空间(§3.1 原话)。
  - 证据:实测被上裁剪 token 的概率 <0.2(Fig.3a),解耦后熵回升(Fig.2b)。没它→熵坍缩、采样近乎同质。
- **② Dynamic Sampling**(+8,42→50,**最大单项**)【原文 §3.2, Eq.11, Fig.3b/6】。
  - 干什么:对每个 prompt 过采样,**过滤掉组内全对(acc=1)或全错(acc=0)的 prompt**,持续采样直到 batch 被"有效 prompt"填满(`buffer size < N` 则 continue,见 Algorithm 1 第 6–8 行)。约束写进目标 s.t. \(0<|\{\text{正确}\}|<G\)。
  - 直觉:组内全对/全错 → 奖励全同 → \(\mathrm{std}=0\) 归一后 \(\hat A_{i,t}=0\) → 零梯度。随训练推进 acc=1 的样本比例单调上升(Fig.3b),有效 prompt 数持续缩水,batch 梯度方差变大、信号被稀释。过滤保证每个 batch 全是有效梯度。
  - 反驳"变慢"质疑:作者澄清它不显著拖慢训练——同步 RL 系统的生成时间本由长尾样本主导,且收敛更快(Fig.6 同性能用更少步达到)。
- **③ Token-Level Loss**(+1,41→42,贡献最小)【原文 §3.3, Eq.12, Fig.4】。
  - 干什么:把 sample-level 聚合 \(\frac1G\sum_i\frac1{|o_i|}\sum_t(\cdot)\) 改为 token-level \(\frac{1}{\sum_i|o_i|}\sum_i\sum_t(\cdot)\)——即对 batch 内**所有 token 求和后按总 token 数归一**,而非先序列内平均再样本间平均。
  - 直觉:sample-level 下每个序列权重相同,长序列里每个 token 的权重被 \(1/|o_i|\) 稀释,导致(i)高质量长样本里的推理模式学不进去,(ii)长样本里的乱码/重复模式罚不动 → 熵和长度不健康膨胀(Fig.4a/b)。token-level 让长序列对梯度有正比贡献,且"某生成模式若改变 reward,就被等强度提升/抑制,与所在响应长度无关"(§3.3 原话)。
  - 作者明说"性能增益较小但提升训练稳定性、让长度更健康增长"。
- **④ Overlong Reward Shaping**(Overlong Filtering +6,30→36;Soft Punishment +3,38→41)【原文 §3.4, Eq.13, Fig.5】。
  - 干什么:两步。(i)**Overlong Filtering**:把被截断(超长)样本的 loss **mask 掉**(不计梯度);(ii)**Soft Overlong Punishment**:在 \([\,L_{\max}-L_{\mathrm{cache}},\,L_{\max}]\) 区间内给长度加一个**软惩罚**,叠加到规则正确性奖励上:
  \(\displaystyle R_{\mathrm{length}}(y)=\begin{cases}0,&|y|\le L_{\max}-L_{\mathrm{cache}}\\[4pt]\dfrac{(L_{\max}-L_{\mathrm{cache}})-|y|}{L_{\mathrm{cache}}},&L_{\max}-L_{\mathrm{cache}}<|y|\le L_{\max}\\[6pt]-1,&L_{\max}<|y|\end{cases}\)
  - 直觉:若一律对截断样本重罚,会把"推理正确只是太长"的样本错误惩罚,注入 reward 噪声、让模型困惑于"我的推理到底对不对"。先 mask 截断样本消除噪声(Fig.5 精度/熵更稳),再用线性软惩罚温和地把长度往回拉(越长罚越多,撞 \(L_{\max}\) 才给 −1)。默认 \(L_{\max}=16384,\;L_{\mathrm{cache}}=4096\)(即生成上限 20480)。

### 训练流程(Algorithm 1,逐步)
对 step \(=1..M\):① 从 \(\mathcal D\) 采 batch \(\mathcal D_b\);② \(\pi_{\theta_{\mathrm{old}}}\leftarrow\pi_\theta\);③ 对每题采 \(G\) 个输出 \(\{o_i\}\sim\pi_{\theta_{\mathrm{old}}}\);④ 用规则 \(R\) 算奖励;⑤ **Dynamic Sampling 过滤**全对/全错并入 buffer;⑥ 若 buffer 不满 \(N\) 则 continue(回到 ① 继续采);⑦ 对 buffer 内每个 token 算 \(\hat A_{i,t}\);⑧ 内层 \(\mu\) 次梯度迭代最大化 \(J_{\mathrm{DAPO}}\)(Eq.8)【原文 Algorithm 1】。

### 关键超参默认值(复现锚点)【原文 §4.1, §4.2】
框架 verl;基座 Qwen2.5-32B base;baseline=朴素 GRPO+组奖励归一。优化器 AdamW、常数 lr \(1\times10^{-6}\)、20 rollout step 线性 warm-up;prompt batch 512、每 prompt \(G=16\) 响应;train mini-batch 512(即每 rollout 做 16 次梯度更新,\(\mu=16\));\(\varepsilon_{\mathrm{low}}=0.2,\;\varepsilon_{\mathrm{high}}=0.28\);\(L_{\max}=16384\) + \(L_{\mathrm{cache}}=4096\) soft punish buffer = 最大生成 20480 token;无 KL、无 reference;评测 temperature 1.0、top-p 0.7,AIME 重复 32 次报 avg@32。

### 实验与证据 / 靠不靠谱
- 主结果:AIME 2024 约 0%→50%,半数步即超 DeepSeek-R1-Zero-Qwen-32B(47);Table 1 给 4 技术逐项累积消融(30→36→38→41→42→50,最后 +8 来自 Dynamic Sampling)。
- baseline 公平性:同基座(Qwen2.5-32B)同评测对照 R1-Zero-Qwen-32B,公平;但**主结果集中在单一基座 + 单一评测(AIME 2024)**,跨基座/跨任务论文内未验证。"看着强但没回答"的点:50 分是否泛化到非数学/更小模型,论文未答(声称"可迁移到其它任务"但未给证据)【推断,据 §4.1 仅做数学】。
- 假设与失效边界:【原文】去 KL 的前提是"长 CoT RL 下模型分布本应大幅偏离初始模型"(§2.3),故在需要贴近初始模型的对齐场景不适用;规则二值奖励要求任务可验证(数学/代码)。【推断】\(\varepsilon_{\mathrm{high}}=0.28\) 等为经验取值无敏感性分析,换基座/任务可能需重调;在更小/更弱 base 上,Clip-Higher 放宽探索是否仍稳定未知。
- 祛魅总结:真贡献=把朴素 GRPO 的失败点工程化拆解为 4 个可复现补丁 + 彻底开源(算法+码+数据),复现价值极高,这是它被社区奉为标准 recipe 的根本原因。包装/高估处="key techniques that make large-scale RL a success"的普适性主要靠社区后续复现背书,论文内只在数学单域+单基座验证;Token-Level Loss 被列为四大技术之一但实测仅 +1 分,贡献被叙事略微抬高【推断,据 Table 1】。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=规则二值结果奖励(答案对+1/错−1,可验证任务)| **改什么**=策略参数(on-policy 梯度更新)| **何时改**=训练期,每 rollout step 16 次梯度更新 | **免梯度?**=否(全程梯度 RL)| **记忆-技能生命周期**=不涉及(无显式记忆/技能库,能力固化进权重)| **防遗忘机制**=去掉 KL 惩罚(反而主动允许偏离),无专门防遗忘设计【原文 §2.3, §4.1】。
- ⑦ 开源代码+框架/harness:https://github.com/BytedTsinghua-SIA/DAPO(recipe/eval/数据说明;**训练逻辑实现在 verl**:https://github.com/volcengine/verl,DAPO 作为 verl 的一个 recipe);框架 **veRL**(requirements 含 vllm==0.8.3、ray[serve])。本地已 clone(~4.1MB)。【原文 Abstract 脚注 a 明确"built on the verl framework"】
- 💰 资源/成本与可扩展性:论文未给出 GPU 卡时/训练成本的明确数字(原文未说明);仅知 rollout prompt batch 512×16 响应、生成上限 20480 token、Dynamic Sampling 会增加采样量但作者称总训练时间不显著增加(Fig.6 收敛更快)【原文 §4.1-4.2】。
- 🎯 对"探索-巩固"对标:**支撑(探索侧)** —— 一句判定:DAPO 的 Clip-Higher 与 Dynamic Sampling 是"如何在 on-policy RL 中保住探索、不让策略过早确定化"的高质量工程方案,与"探索=发现有效行为/路径"的诉求直接同构;依据:Clip-Higher 专门给低概率探索 token 解锁上行空间(§3.1)、Dynamic Sampling 保证有效梯度即保证持续探索信号(§3.2)。但它**不涉及"巩固/回轨"也不涉及 teacher 脚手架**——纯 on-policy 自演化,无错误前缀注入、无路径恢复目标(对比 denoiserl)。可借组件:去 KL + token 级聚合损失 + Soft Overlong Punishment 可直接搬进任何 GRPO 底座;缺口:无 path-recovery/巩固机制、无 MTP 前瞻、无防遗忘。
- 🔭 开放问题/未来方向:【原文】§4.3 讨论训练动态——长度并非单调上升、训练 reward 与验证精度弱相关(过拟合训练集),提示需结合长度+验证精度共同监控;监控熵需维持在合适区间。【推断】跨基座/跨任务泛化、超参敏感性、与过程奖励或路径级信用分配的结合、在更小模型上的可扩展性均未解。

---
RETURN: dapo | 读到PDF? 是(16页全文,公式从 PDF 精确抄录) | L3(RLVR/GRPO) | 探索侧强支撑(Clip-Higher/Dynamic Sampling 是保探索工程范式),无巩固/回轨/teacher 脚手架,纯 on-policy | 残留待核 0

deepseekmath | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO 起源) | DeepSeek-AI(Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Daya Guo 等) | arXiv 2402.03300(v3 2024-04-27)·2024-02 首发·模型报告 | 主题线 L3(RLVR/GRPO 算法基石)·相关性 高

**原始论文**:https://arxiv.org/abs/2402.03300

## 一眼看懂

> 一句话导读:GRPO 的诞生地。一句话讲它干了什么——PPO 要额外养一个跟主模型一样大的"评分网络"(critic)来估每一步好坏,GRPO 干脆不要这东西,改用"同一道题采好几个答案、互相比一比"来代替,省一半显存算力;顺带还提出一套统一框架把各种后训练方法都装进同一个梯度公式里。

- 🟦 TL;DR:本文首次提出 **GRPO**。核心做法是用"同一道题采多个输出、组内互相比较得到的相对奖励"来替代 PPO 的价值网络(critic),从而省掉一个与策略同规模的 critic,大幅降低显存和算力。同时它给出一个统一梯度范式,把 SFT / RFT / Online-RFT / DPO / PPO / GRPO 都归一到一起,差异只落在三点上:数据源、奖励函数、梯度系数。
  - 效果:DeepSeekMath-RL 7B 在 GSM8K=88.2%、MATH=51.7%(纯 CoT,不用工具、不用投票);而且仅用 GSM8K/MATH 的 CoT 数据训练,却在所有基准(含 OOD)上都超过 DeepSeekMath-Instruct 7B【原文 Abstract, §4, §5.2】。
- 最巧的一步:**用组内多次采样的均值 / 标准差归一来当 advantage baseline**(取代价值网络)。为什么可行:奖励模型本质衡量的就是"同一道题里不同答案谁更好",所以"组内相对奖励"天然契合奖励模型的这种比较性质,可以直接当一个无偏的 baseline,根本不必再单独训一个价值网络。
  - 抽掉它就退回 PPO——要训一个与策略同规模的 critic(显存 / 算力翻倍),GRPO 的"高效"卖点就垮了【原文 §4.1.1】。

## 为什么做

> 一句话导读:SFT 之后接 RL 来提升数学推理,业界标配是 PPO,但 PPO 那个 critic 又贵又难训(稀疏奖励下尤其难学)。本文想省掉它,顺便把 RFT/DPO/PPO/GRPO 这些零散方法收进一个统一框架,讲清楚 RL 到底为什么有效。

- 研究背景:RL 已经被证明能在 SFT 之后进一步提升 LLM 的数学推理,这一阶段业界广泛使用 PPO(actor-critic 架构)。本文是在大规模数学语料预训练(120B math tokens,得到 DeepSeekMath-Base/Instruct 7B)的基础上,引入一个新的 RL 算法【原文 §1, §4】。
- 解决的具体痛点,三条:
  - ① PPO 的价值模型开销大——要训一个与策略同规模的 critic,显存和算力都很大;
  - ② 稀疏奖励下 critic 难训——LLM 场景通常只在最后一个 token 由奖励模型打分,要学一个逐 token 的精确价值函数很难;
  - ③ 缺乏一个统一的理论框架来理解各类后训练方法(RFT / DPO / PPO / GRPO)【原文 §4.1.1, §5.2】。
- 相关工作 & 各自不足(来龙去脉 + 精确差异):
  - **奠基:PPO(Schulman 2017,§4.1.1)**——actor-critic,用 GAE 估优势 \(\hat A_t\) 时需要价值函数 \(V\) 当 baseline。**精确短板**:critic 与策略同规模,显存 / 算力翻倍;而且 LLM 只在末 token 给奖励,逐 token 精确价值难学。GRPO 站在 PPO 的 clip 机制肩上,但**去掉 critic、改用组内均值当 baseline**。
  - **并行路线 ①——RFT / Online-RFT(拒绝采样微调)**:RFT 在 SFT 模型自己采到的正确响应上做 SFT;Online-RFT 则实时用当前策略采样。**精确短板**:RFT 是离线的(数据来自一个固定的 SFT 模型);而且两者都**不惩罚错误、对所有正确响应一视同仁地等强度强化**(梯度系数恒为一个正常数)。GRPO 的差异是**按奖励值差异化正负梯度系数**——这正是 §5.2 实证 GRPO>Online-RFT 的根因。
  - **并行路线 ②——DPO(直接偏好优化)**:用偏好对 \((o^+,o^-)\) 隐式地做 RL,没有显式奖励。本文把它也纳入统一范式(梯度系数见 Eq.14),但它属于离线偏好对、和在线 RL 不是一族。
  - **并行路线 ③——过程奖励(PRM,Wang 2023b / Lightman 2023 PRM800K)**:信号更细(每一步都打分),但**标注本身含约 20% 的错误**(§5.2.3),且需要额外训一个过程奖励模型。GRPO 兼容过程监督(§4.1.3),但默认用的是结果监督。
  - **站位**:GRPO 并不否定 PPO 框架,而是在其内部做两处替换——用"组内相对奖励"替掉"价值网络 baseline",并把 KL 从"加进奖励"改成**直接加进损失**(以免把优势计算复杂化)。
- 动机链:现状是 SFT 后用 PPO 做 RL 来提升数学;缺陷在于 critic 贵、稀疏奖励难训、各方法零散没有统一理解;所以本文的做法是用组内相对奖励省掉 critic(即 GRPO),并提供一个统一范式来分析在线 / 离线、结果 / 过程监督、单轮 / 迭代,从而理解 RL 为何有效【原文 §4.1, §5.2】。
- 与最近邻工作的Δ:相对 PPO,差在去掉价值网络、改用组内 baseline + KL 直接加损失(而非加进奖励);相对 RFT / Online-RFT,差在 GRPO 按奖励值差异化调整正负梯度系数。有用点:GRPO 成了 R1、DAPO 以及绝大多数 RLVR 工作的算法基石,被 veRL / TRL / OpenRLHF / ms-swift 广泛实现。

## 怎么做 + 靠不靠谱

> 一句话导读:A 是 GRPO 主目标(组内相对优势 + clip + KL 加在损失里),B 是一个保证非负的 KL 估计器,C 给出"结果监督 / 过程监督"两种算优势的方式,E 是把所有方法统一成一个梯度公式的"三件套"框架。

### A. GRPO 目标函数(§4.1.1,Eq.3,读完可实现)
对每道题 \(q\),从旧策略 \(\pi_{\theta_{\mathrm{old}}}\) 采一组 \(G\) 个输出 \(\{o_1,\dots,o_G\}\),然后最大化:
\(\displaystyle \mathcal J_{\mathrm{GRPO}}(\theta)=\mathbb E_{q\sim P(Q),\,\{o_i\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}(O|q)}\frac1G\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big[\min\Big(\tfrac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}|q,o_{i,<t})}\hat A_{i,t},\ \mathrm{clip}\big(\tfrac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}|q,o_{i,<t})},1-\varepsilon,1+\varepsilon\big)\hat A_{i,t}\Big)-\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})\Big] \tag{3}\)
这里有两处关键设计:(i)优势 \(\hat A_{i,t}\) **只由组内相对奖励**来估计(不用价值函数),正好契合奖励模型"比较"的本质;(ii)KL **直接加进损失**(不像 PPO 那样把 KL 加进奖励),这样就不会把 \(\hat A_{i,t}\) 的计算搞复杂。

### B. KL 无偏估计器(§4.1.1,Eq.4)
和 PPO 把 KL 加进奖励不同,GRPO 用的是 Schulman(2020)提出的、**保证非负**的无偏估计器:
\(\displaystyle D_{\mathrm{KL}}\big(\pi_\theta\|\pi_{\mathrm{ref}}\big)=\frac{\pi_{\mathrm{ref}}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})}-\log\frac{\pi_{\mathrm{ref}}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})}-1 \tag{4}\)
直觉:它形如 \(x-\log x-1\geq 0\)(其中 \(x=\pi_{\mathrm{ref}}/\pi_\theta\)),恒为非负,数值上很稳定。

### C. 优势估计两变体
> 一句话导读:结果监督只在答案末尾打一个分、整条输出共用;过程监督则给每一步打分,某个 token 的优势是"它之后所有步"的奖励之和(谁后面贡献大就给谁更多功劳)。
- **结果监督(§4.1.2)**:奖励模型对每条输出打分,得到 \(\mathbf r=\{r_1,\dots,r_G\}\);组内归一后,**整条输出里所有 token 的优势都相同**:
\(\displaystyle \hat A_{i,t}=\tilde r_i=\frac{r_i-\mathrm{mean}(\mathbf r)}{\mathrm{std}(\mathbf r)}\)
- **过程监督(§4.1.3)**:过程奖励模型对每个推理步的末 token 打分,得到 \(\{r_i^{\mathrm{index}(j)}\}\)(这里 \(\mathrm{index}(j)\) 是第 \(j\) 步末 token 的索引);组内归一为 \(\tilde r_i^{\mathrm{index}(j)}\)。某个 token 的优势 = **它后续各步的标准化奖励之和**(这是一种 step-aware 的信用分配):
\(\displaystyle \hat A_{i,t}=\sum_{\mathrm{index}(j)\ge t}\tilde r_i^{\mathrm{index}(j)}\)
- **迭代 RL(§4.1.4)**:训练过程中,用策略采样的结果**重新训练奖励模型**(replay 里掺 10% 历史数据),然后把 reference 设为当前策略、用新的奖励模型继续训。

### D. 训练流程(Algorithm 1,逐步)
外层 iteration \(=1..I\):① 把 reference 设为当前策略(reference \(\leftarrow\) 当前策略);② 内层 step \(=1..M\),依次做——采一个 batch \(\mathcal D_b\),令 \(\pi_{\theta_{\mathrm{old}}}\leftarrow\pi_\theta\),对每题采 \(G\) 个输出,用奖励模型 \(r_\varphi\) 打分,估出组内相对优势 \(\hat A_{i,t}\),再在内层做 \(\mu\) 次 GRPO 目标的最大化;③ 用 replay 持续训练 \(r_\varphi\)。

### E. 统一梯度范式(§5.2.1,Eq.5——本文第二大贡献)
> 一句话导读:任何后训练方法的梯度都能拆成同一个公式,只在三个地方不一样——数据从哪来、奖励怎么给、把它们怎么变成梯度系数;SFT/RFT/PPO/GRPO 不过是这三个旋钮的不同取值。
任何训练方法对 \(\theta\) 的梯度都可以写成同一个统一形式:
\(\displaystyle \nabla_\theta\mathcal J_{\mathcal A}(\theta)=\mathbb E_{\underbrace{(q,o)\sim\mathcal D}_{\text{Data Source}}}\left(\frac{1}{|o|}\sum_{t=1}^{|o|}\underbrace{GC_{\mathcal A}(q,o,t,\pi_{rf})}_{\text{Gradient Coefficient}}\,\nabla_\theta\log\pi_\theta(o_t|q,o_{<t})\right) \tag{5}\)
它由三个组件构成:
- **数据源 \(\mathcal D\)**:决定训练数据从哪来;
- **奖励函数 \(\pi_{rf}\)**:决定奖励信号的来源;
- **算法 \(\mathcal A\)**:把数据和奖励处理成梯度系数 \(GC\),也就是决定强化 / 惩罚的幅度。

据此,Table 10 把各方法归一对照:SFT(\(GC\equiv1\),数据 = 人选的 SFT 集)/ RFT(数据 = SFT 模型离线采样,Rule 奖励)/ Online-RFT(数据 = 实时策略采样)/ PPO(Model 奖励)/ GRPO(组采样 + Model 奖励)——**差异仅落在这三点上**。

### 逐组件必要性(§5.2 系统对照,均有 Fig.5/6)
- **组内 baseline(去掉 critic)**:这是核心。GRPO 相比 PPO 省掉了 critic,却仍能达到 SOTA【原文 §4.1.1, §4.2】。
- **在线 vs 离线**:Online RFT 显著超过 RFT(早期接近、后期拉开)——因为后期 actor 与原 SFT 模型差异变大,这时用实时采样更有利【原文 §5.2.1, Fig.5】。
- **差异化梯度系数**:GRPO 超过 Online RFT。原因是 GRPO 会按奖励值给出有差异的正负梯度,而 Online RFT 不惩罚错误、对所有正确响应等强度强化【原文 §5.2.1】。
- **过程监督 vs 结果监督**:GRPO+PS 超过 GRPO+OS,说明细粒度的 step-aware 梯度系数是有益的【原文 §5.2.1, Fig.5】。
- **迭代 RL**:两轮迭代带来显著提升(尤其第一轮)【原文 §4.1.4, §5.2.1, Fig.6】。

### F. 关键超参与实验(复现锚点)【原文 §4.2】
RL 基于 DeepSeekMath-Instruct 7B;RL 数据 约 144K GSM8K/MATH CoT 题(故意排除其它 SFT 题以观察 RL 对缺数据基准的影响);奖励模型基于 Base 7B(lr 2e-5);GRPO 超参:策略 lr **1e-6**、**KL β=0.04**、每题 **G=64**、max length **1024**、batch **1024**、每探索阶段策略仅更新一次。结果(Table 5):RL 7B GSM8K **88.2%**、MATH **51.7%**(CoT),超 7B-70B 全部开源 + 多数闭源;仅训 GSM8K/MATH CoT 却在所有(含 OOD 如 MGSM-zh/CMATH/工具集成)基准超 Instruct 7B。

### 靠不靠谱
> 一句话导读:GRPO 和统一范式是真贡献,而且作者难得地诚实——他自己揭穿了"RL 只是把已有正确答案顶上来(升 Maj@K),并没真正扩展能力边界(不升 Pass@K)"。但要注意:全部结论只在数学单域、7B 单规模上验证,熵坍缩等坑是后来 DAPO 才补的。
- baseline 公平性:GRPO / Online-RFT / RFT / PPO 都在同 base、同数据下对照(Fig.5),这点公平。**一个关键的去魅证据(§5.2.2 Fig.7)**:RL 能提升 Maj@K,但**不提升 Pass@K**——这说明 RL 主要是让输出分布更鲁棒(把本就正确的答案从 TopK 里顶上来),而**不是提升根本能力**;作者自己也承认,这可能是因为只用了 SFT 阶段的题 + 朴素 nucleus 采样【原文 §5.2.2】。
- 假设与失效边界:
  - 【原文】所有方法都"完全信任奖励信号",但奖励并不总可靠(连 PRM800K 都约 20% 标错),复杂任务下会失效;因此作者提出需要 weak-to-strong / 抗噪奖励算法(§5.2.3)。"RL 只稳定分布、不提升根本能力"这一点,在奖励模型无法泛化到 OOD 时会更明显。
  - 【推断】结论集中在数学单域、7B 单规模;\(G=64\)、max length 1024 的设定在更长 CoT 下不适用(后续 R1/DAPO 用到 32768+);"在线优于离线""GRPO+PS 更优"这些结论的可推广性还需外部验证(事实上 DAPO 后来就指出了朴素 GRPO 的熵坍缩问题)。
- 祛魅总结:
  - 真贡献:GRPO(一个省掉 critic 的高效 RLVR 算法,后来成了整个领域的基石);统一梯度范式(把方法拆成数据源 / 奖励 / 梯度系数三件,解释力很强);以及罕见的诚实去魅(明说 RL 只升 Maj@K、不升 Pass@K,即分布锐化而非能力提升)。
  - 包装 / 高估处:GRPO 被后续奉为"标准",但本文只在数学 7B 上验证过,熵坍缩等问题是留给 DAPO 等去修补的;过程监督 / 迭代 RL 依赖额外的过程 / 迭代奖励模型,其工程成本与稳定性本文没有深入。注:这份模型报告的主要篇幅其实是预训练语料工程(120B math corpus、code training 的益处、arXiv 无用等),GRPO 只占后半【推断,据全文结构】。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=奖励模型/规则对完整输出(结果监督)或每推理步(过程监督)的标量奖励,组内归一 | **改什么**=策略全参数(on-policy 梯度)| **何时改**=RL 训练期,每探索阶段策略更新一次;可迭代(同时重训奖励模型)| **免梯度?**=否(梯度 RL)| **记忆-技能生命周期**=不涉及显式记忆/技能库,数学推理能力固化进权重 | **防遗忘机制**=GRPO KL 惩罚(β=0.04,直接加损失约束 ref);无专门防遗忘【原文 §4.1】。
- ⑦ 开源代码+框架/harness:https://github.com/deepseek-ai/DeepSeek-Math;模型权重 DeepSeekMath-Base/Instruct/RL 7B(HuggingFace)。**仓库主要为评测/推理脚本与模型发布,GRPO 训练代码不在仓内**(实际 RL 为 DeepSeek 内部代码)。框架 custom/none。GRPO 算法后被 veRL/TRL/OpenRLHF/ms-swift 广泛实现。【训练码未开源,复现依赖论文公式 + 第三方框架;本条为模型报告,无训练代码】
- 💰 资源/成本与可扩展性:论文未给 RL 阶段的 GPU 卡时(原文未说明);GRPO 的核心卖点正是降成本——省掉与策略同规模的价值网络,显著优化 PPO 的显存占用(Abstract"optimizing the memory usage of PPO")。预训练 120B math tokens(独立大成本,与 RL 分离)【原文 Abstract】。
- 🎯 对"探索-巩固"对标:**支撑(探索侧算法底座) + 一个关键警示** —— 一句判定:GRPO 是本项目几乎所有 on-policy RL/蒸馏实验要用的算法地基(探索=策略自采样多输出、按相对奖励强化走得通的路径),其统一范式(数据源/奖励/梯度系数)正好提供了思考"稀疏脚手架如何改梯度系数"的语言;依据:§4.1 组内相对优势 + §5.2.1 梯度系数分解(Eq.5)。**关键警示(对本项目最有价值)**:§5.2.2 实证 RL 只升 Maj@K 不升 Pass@K——若本项目的"探索-巩固"也只是把已有正确路径顶上来而非真正扩展能力边界,就会落入同样陷阱,这正是 denoiserl 等"注入新状态/负样本"工作和本项目"teacher 脚手架扩探索空间"想要突破的点。可借组件:①过程监督的 step-aware 优势 \(\hat A_{i,t}=\sum_{\mathrm{index}(j)\ge t}\tilde r_i^{\mathrm{index}(j)}\) 可迁移为"路径恢复/巩固的逐步信用分配";②迭代 RL(策略进化→奖励模型同步进化)对"持续学习不遗忘"有结构启发。缺口:GRPO 本身无 teacher 脚手架、无路径恢复、无 MTP、无记忆/技能库,纯结果/过程奖励驱动。
- 🔭 开放问题/未来方向:【原文】§5.2.3 三方向——数据源(OOD prompt + tree-search 等高级采样以突破 Pass@K 瓶颈)、算法(抗噪奖励、weak-to-strong 对齐)、奖励函数(泛化性、不确定性建模、高质量过程奖励模型);否则"RL 仅稳定分布而非提升根本能力"。【推断】更大规模/更长 CoT 下的熵坍缩(后被 DAPO 揭示)、把 GRPO 与 on-policy 蒸馏/路径恢复结合、用前瞻(MTP)信号改进探索效率均未涉及。

---
RETURN: deepseekmath | 读到PDF? 是(读GRPO §4 + 统一范式 §5.2,公式从正文.txt 精确抄录) | L3(GRPO 算法基石) | 探索侧算法底座(本项目 RL/蒸馏实验地基);§5.2.2"RL只升Maj@K不升Pass@K"是对"探索-巩固是否真扩能力边界"的关键警示;无teacher脚手架/路径恢复/MTP | 残留待核 0(为模型报告,GRPO 训练代码未开源)

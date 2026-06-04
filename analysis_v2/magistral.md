magistral | Magistral(Mistral 首个推理模型与自建可扩展 RLVR 流水线)| Mistral AI | 2025-06-12(arXiv 2506.10910, v1);技术报告(Magistral Small 24B 权重 Apache 2.0) | 主题线 L3 RLVR/GRPO + L6 长 CoT·相关性 中

**原始论文**:https://arxiv.org/abs/2506.10910

## 一眼看懂

> 一句话导读:这是 Mistral 的模型技术报告——不靠任何外部蒸馏轨迹,纯用自家模型 + 自研异步在线 RL,把一个通用模型纯 RLVR 训成推理模型,并诚实地公开了一批失败实验。

- 🟦 TL;DR:Mistral 自底向上、**不依赖任何外部蒸馏轨迹**,只用自家模型 + 自研异步在线 RL 基础设施,纯 RLVR(改造版 GRPO + 四维奖励整形)把 Mistral Medium 3 训成了推理模型 Magistral Medium(AIME-24 pass@1 从 26.8→73.6,近 +47 分)。

  报告给出三样东西:
  - 让推理语言强制与用户语言一致的简单方法;
  - 纯文本 RL 不损、甚至提升多模态/指令/函数调用的实证;
  - 一批失败实验(诚实负结果)。【原文 Abstract, §1】

  (术语:RLVR = 可验证奖励 RL,只在答案能被规则验证为正确时给奖励;CoT = chain-of-thought 长链推理。)

- 最巧的一步:**放宽 trust-region 上界(Clip-Higher,\(\varepsilon_{\text{high}}\) 调到 0.26-0.28)+ 过滤掉零优势组**(§2.1)。

  - 抽掉 Clip-Higher:低概率但可能关键的推理 token 涨不上去 → 熵塌缩、探索被扼杀,纯 RL 难以持续涨;
  - 过滤全对/全错组:保证每个 batch 都有有效梯度。

  二者共同支撑了"无冷启动、纯 RL 也能持续提升"这一核心主张。

## 为什么做

> 一句话导读:社区普遍认为小模型纯 RL 不如从大模型蒸馏,Mistral 想用受控实验正面挑战这个观点,同时解决大规模在线 RL 的基础设施、语言混杂、纯文本 RL 是否伤多模态等实际问题。

- 研究背景:推理模型(o1/R1)靠长 CoT;DeepSeek-R1 给出了 RLVR 配方,但社区多依赖"先蒸馏 R1 轨迹冷启动、再 RL"。Mistral 想验证**纯 RL(不用任何已有推理模型的蒸馏)能走多远**,并完全用自有技术栈复现/扩展。【原文 §1】

- 解决的具体痛点:
  - (1) 此前观点(DeepSeek-R1)认为**小模型纯 RL 不如从大模型蒸馏**——本文要挑战;
  - (2) 大规模在线 RL 的基础设施(异步生成、on-policyness 与吞吐之间的权衡);
  - (3) 推理模型常出现**语言混杂**(英中俄混说);
  - (4) 纯文本 RL 是否会损害多模态/工具能力。【原文 §1, §2.2.4, §7】

- 相关工作 & 各自具体短板:
  - **DeepSeek-R1 / R1-Zero**:给出了 RLVR 配方,但其结论是"小模型应优先从大模型蒸馏、而非纯 RL"——本文用受控实验**正面反驳**(24B 纯 RL 可达蒸馏版相近,MATH/GPQA 上部分更优)。
  - **DAPO(Yu 2025)**:贡献了 Clip-Higher、动态采样——本文沿用 Clip-Higher、过滤零优势组,但**明确否定 entropy bonus**(见消融),改用调 \(\varepsilon_{\text{high}}\) 代替。
  - **Dr.GRPO / Liu 2025、Hu 2025**:做 loss/优势归一去偏——本文集成进自己的 GRPO 改造,但在"优势归一方式"上给出**负结果**(三种方式无显著差异)。
  - **共同短板(本文要补的)**:这些工作多在算法点上做单点优化,缺一套**端到端、工业级、异步在线**的 RLVR 基础设施,缺系统性的"纯 RL vs 蒸馏 / 跨域泛化 / 对其它能力影响"受控研究,也缺诚实的失败实验记录。

- 动机链(逐步推进):
  - 想要自主推理能力;
  - 不蒸馏、走纯 RLVR;
  - 需要稳定的 GRPO 改造(去 KL/Clip-Higher/优势归一/过滤零优势)+ 四维奖励 + 异步在线 RL 基础设施;
  - 用它验证纯 RL vs 蒸馏、跨域泛化、对其它能力的影响。

- 与最近邻工作的精确 Δ:相比 R1 的"小模型靠蒸馏",Magistral 的 Δ 是**实证"纯 RL 在 24B 小模型上可达蒸馏版相近、部分(MATH/GPQA)更优,且把 RL 叠在蒸馏 checkpoint 之上还能再 +5~10 分"**(§6.2, §8);并给出了语言一致性奖励、batch=minibatch 的 off-policy 边界等实用小招与负结果。

## 怎么做 + 靠不靠谱

> 一句话导读:数据难度过滤 + 一套去 KL/Clip-Higher/过滤零优势的 GRPO 改造 + 四维奖励 + 自研异步在线 RL 设施;诚实地报告了若干负结果。最大祛魅是:算法点多沿用 DAPO/Dr.GRPO 原创增量有限,增益部分来自"学会写更长 CoT",且 Medium 权重未开源、无训练代码、核心规模不可独立复现。

- 方法流水线:
  - ① 数据 curation:数学 700k → 格式过滤 501k → 难度过滤(两阶段,用 RL 训过的更强模型重新评分)38k;代码 35k(测试用例一致性过滤);
  - ② 改造 GRPO 做在线 RL(见下)+ 四维奖励;
  - ③ 异步基础设施(Trainer/Generator/Verifier 三类 worker,生成器不停跑、权重经 NCCL 热更新、in-flight 序列不丢);
  - ④ 多阶段课程(难度↑、生成长度上限↑、batch↓);
  - ⑤ Magistral Medium(纯 RL);Magistral Small(先用 Medium 轨迹 SFT 冷启动、再 RL)。【原文 §2-§5】

- GRPO 改造(§2.1,逐项即组件,真实符号):
  - 起点为标准 GRPO 目标(§2.1):\(J_{\text{GRPO}}(\theta)=\mathbb{E}_{q,\{o_i\}}\big[\tfrac{1}{G}\sum_i\tfrac{1}{|o_i|}\sum_t\min(r_{i,t}\hat A_{i,t},\ \mathrm{clip}(r_{i,t},1-\varepsilon,1+\varepsilon)\hat A_{i,t})-\beta D_{\mathrm{KL}}[\pi_\theta\|\pi_{\text{ref}}]\big]\),其中 \(r_{i,t}=\pi_\theta(o_{i,t}|q,o_{i,<t})/\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})\),组归一优势 \(\hat A_{i,t}=(r_i-\mu)/\sigma\)。
  - **去 KL 惩罚**(\(\beta=0\)):策略本就大幅偏离,维护 ref 模型不划算 → 直接去掉。
  - **loss 归一**:先把组内所有 token、所有 generation 的 token-wise loss 相加,再除以组内总长度 \(\sum_{i=1}^G|o_i|\)(去掉组内长度偏置)。
  - **优势归一**:先 \(\hat A_{i,t}=\hat A_i=r_i-\mu\),再在 minibatch 内标准化 \(\hat A_{i,t}^{\text{norm}}=(\hat A_i-\hat A_{\text{mean}})/\hat A_{\text{std}}\)(序列级均值/标准差,Andrychowicz 2020)。
  - **Clip-Higher(放宽上界)**:把上裁阈值升到 \(\varepsilon_{\text{high}}\),给低概率 token 增长空间、提熵促探索(核心);\(\varepsilon_{\text{high}}\) 调在 0.26-0.28 以稳住 group entropy。
  - **过滤零优势组**:全对/全错的组优势为 0、无梯度且对噪声敏感 → 组 batch 时剔除。
  - 最终改造后 loss(§2.1,把红色修改全合进来):
  \(\displaystyle J_{\text{GRPO}}(\theta)=\mathbb{E}_{q,\{o_i\}}\!\left[\frac{1}{\sum_{i=1}^{G}|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\min\!\Big(r_{i,t}\hat A_{i,t}^{\text{norm}},\ \mathrm{clip}(r_{i,t},1-\varepsilon_{\text{low}},1+\varepsilon_{\text{high}})\,\hat A_{i,t}^{\text{norm}}\Big)\right],\)
  约束 \(\ \text{s.t. }\exists\,1\le m<n\le G,\ r_m\neq r_n\)(即只保留"组内有奖励差异"的非平凡组)。注意:相对标准 GRPO,这里**没有 KL 项**、上下裁剪不对称(\(\varepsilon_{\text{low}}\) vs \(\varepsilon_{\text{high}}\))、优势用的是 minibatch 归一版。

- 四维奖励整形(§2.2,真实数值):
  - **格式**(§2.2.1):必须含 `<think></think>` 且答案段有 `\boxed{}`(代码题为代码块);任一条件不满足 → reward \(=0\) 且不再评分;满足则得 **0.1** 进入评分。
  - **正确性**(§2.2.2-3):数学用规则验证器(归一两边、SymPy 比对)、代码编译并跑 ≤20 个测试(有内存上限);正确额外 **+0.9**(总 1.0)。
  - **长度软惩罚**:接近生成上限时渐扣。
  - **语言一致性**(§2.2.4):把 (problem, thoughts, answer) 三段各去掉 LaTeX/代码块后,用 fastText 分类器判是否同一语言(法/西/意/德/中/俄之一);同语言额外 **+0.1**。

- 逐组件必要性(消融在 §6-§7,含诚实负结果):
  - **Clip-Higher vs entropy bonus**(§7.4.2):entropy bonus 在数学-only 反而降熵、数学+代码混合时熵爆炸;改用调 \(\varepsilon_{\text{high}}\) 更稳——**明确否定 entropy bonus**。
  - **batch=minibatch 且足够大**(§6.3):一个 batch 拆成 >2 个 minibatch(更 off-policy)会骤降性能;应保持 \(n_{\text{async}}/n_{\text{batch}}\le2\)。
  - **优势归一方式**(§6.4):minibatch/group/none 三种**对最终性能无显著差异**(诚实负结果)。
  - **代码部分奖励**(§7.4.1):按通过测试比例给分 → 训练更快、但**最终略低**(LiveCodeBench -2%),故弃用。

- 关键机制/公式(直觉):二值奖励 + Clip-Higher 让"罕见但有洞见的推理步"有空间被强化;§7.1 用 PCA 发现 **RL 主要沿一个"长度方向"移动权重**,reward 随输出长度对数增长——即"会推理"在很大程度上是"学会写更长的 CoT"。系统提示里写 "Be as casual and as long as you want" 能提熵、促探索(§2.2)。

- 实验与证据:
  - Magistral Medium(纯 RL,Table 2):AIME'24 pass@1 26.8→**73.6**(maj@64 90.0)、LiveCodeBench v5 29.1→59.4、GPQA 59.6→70.8;
  - Magistral Small(Table 3):SFT+RL 最优(AIME'24 70.7 > 纯 RL 65.8 > 纯 SFT 65.4);
  - 跨域泛化(Table 5):只训数学也提代码,反之亦然;
  - 多模态白吃午餐(§7.2):纯文本 RL 后 MMMU/MathVista 不降反升;
  - 函数调用/IFEval 基本不变或微升(Table 6);
  - baseline 主要对比 DeepSeek-R1 的报告值(非完全同设),但纯 RL vs 蒸馏的关键对比是**自家受控实验**(§6.2/§8),较可信。

- 假设与失效边界:【原文】纯 RL 的强结果建立在**强基座(Mistral Medium/Small 3)+ 干净且难度适中的数据**上;§6.2 承认代码上纯 RL 略逊蒸馏。【推断】结论依赖工业级算力与自研异步 RL 基础设施;"纯文本 RL 提升多模态"的因果未深究(可能基座多模态本就强);Magistral Medium 权重未开源(只开 Small 24B),核心规模不可独立复现。

- 祛魅总结:【推断】真贡献 = **系统级证明"无蒸馏纯 RLVR 在中小模型上也能强,且 RL 可叠加在蒸馏之上再涨" + 一套可工作的 GRPO 改造与四维奖励 + 诚实的失败实验**。被高估处:很多是工程/规模成果,算法点(去 KL/Clip-Higher/过滤零优势)多沿用 DAPO/Dr.GRPO,**原创算法增量有限**;"RL 沿长度方向移动"暗示增益部分来自"写更长"而非更深推理。被低估处:语言一致性奖励、batch=minibatch 的 off-policy 边界、entropy-bonus 不如调 \(\varepsilon_{\text{high}}\) 等工程结论很实用。**未完整克隆**:仅开源 Magistral Small 24B 权重(Apache 2.0,HF),**无训练代码、Medium 权重不公开**;自研异步 RL 系统未释出。

## 结构化抽取

- 🎯 机制速览6轴:
  - 学什么信号 = 二值可验证奖励(数学规则/代码测试)+ 格式(0.1)/长度软惩罚/语言一致性(0.1)三项整形奖励;正确再 +0.9。
  - 改什么 = 策略参数(全参 RL)。
  - 何时改 = 异步在线 RL 训练期(生成器不停跑、权重经 NCCL 热更新,in-flight 序列继续用稍旧的 KV-cache);多阶段课程。
  - 免梯度? = 否(GRPO 策略梯度)。
  - 记忆-技能生命周期 = 无外部记忆/技能库;能力固化进参数。
  - 防遗忘机制 = 无显式蒸馏/KL(KL 被去掉,\(\beta=0\));靠"纯文本 RL 不损其它能力"的实证 + 课程;Small 用 10% 通用指令数据保住非推理能力。
- ⑦ 开源代码 + 框架/harness:仅开源 Magistral Small (24B, Apache 2.0) 权重(https://huggingface.co/mistralai/Magistral-Small-2506),**无训练代码**(weights-only / paper-only)。框架 = **custom**(自研异步在线 RL 系统:Trainer/Generator/Verifier + NCCL 权重广播)。【原文 §1 脚注 + §3 + 元信息】
- 💰 资源/成本与可扩展性:未给总 GPU 时/$;提及大 GPU 集群、单次权重更新 <5s(NCCL GPU→GPU 广播)、贪心 collation 减 padding 19%。【原文 §3;总成本原文未说明】
- 🎯 对"探索-巩固"对标:**中等支撑(探索侧 + 一条与本课题相关的反证据)**。
  - 探索侧:Clip-Higher / 系统提示提熵 / 过滤零优势组,都服务"保持探索、别过早收敛到单一路径",契合"探索 = 发现有效路径";
  - **关键相关点**:§8 实证"**先用强教师(Magistral Medium)轨迹 SFT 冷启动、再 RL**"显著优于纯 RL(AIME'25 +约 10 分)——这正是"teacher 脚手架(冷启动)+ 自主探索(RL)"两段式的支撑;但其冷启动是**离线 SFT 整轨蒸馏**,非本课题的 on-policy 稀疏接管;
  - **缺口/区别**:无 reverse-KL on-policy 蒸馏、无 path-recovery、无 MTP、无关键步定位;且去掉了 KL(本课题若要"贴着教师巩固"可能需要保留某种锚)。

  一句判定:Magistral 给出了"教师冷启动 + RL 探索"有效的工业证据,但其巩固方式是粗粒度的离线 SFT,可被本课题的 on-policy 稀疏脚手架替代/精细化。
- 🔭 开放问题/未来方向:【原文 §9】更优的 loss/优化算法、用模型自身推理轨迹 bootstrapping 能拿多少增益、扩到下一量级算力、tool-use/多模态/agent。【推断】
  - (1) 把离线 SFT 冷启动换成 **on-policy 稀疏教师接管**(只在关键分叉步用教师纠偏),减蒸馏成本、避免整轨模仿的暴露偏差;
  - (2) RL 沿"长度方向"移动 → 用 MTP 前瞻判断"是否真需要更长",而非一味加长;
  - (3) 在去 KL 的纯 RL 中引入轻量 reverse-KL 巩固项防漂移。

— 残留待核:0(GRPO 改造最终 loss、\(\hat A_{i,t}=(r_i-\mu)/\sigma\) 与 minibatch 归一、\(\varepsilon_{\text{high}}\)=0.26-0.28、四维奖励 0.1/+0.9/+0.1 均据 PDF §2 抄准;Medium 权重未开源、无训练代码已核。)

one_shot_rlvr | Reinforcement Learning for Reasoning in LLMs with One Training Example | UW · USC · Microsoft · UC Santa Cruz · Georgia Tech(Yiping Wang*, Qing Yang, Zhiyuan Zeng, …, Jianfeng Gao, Weizhu Chen, Shuohang Wang*, Simon S. Du*, Yelong Shen*) | 2025-10-24 · arXiv 2504.20571 v3 · NeurIPS 2025 | 主题线 L3(RLVR/GRPO·数据效率+机理)，兼 L6(自反思/CoT) · 相关性 高

**原始论文**：https://arxiv.org/abs/2504.20571

## 一眼看懂
- 🟦 TL;DR：**只用一条训练样本**做 RLVR(GRPO/PPO)，就能把 Qwen2.5-Math-1.5B 的 MATH500 从 36.0%→73.6%、6 数学基准均值从 17.6%→35.7%，几乎追平含该样本的 1.2k DSR 子集(73.6/35.9)，甚至 2 样本(74.8/36.6)略超 1.2k 子集、追平 7.5k MATH 全集。配套发现一串现象:post-saturation generalization(训练准确率早早饱和到 100%、测试仍持续涨)、跨类泛化、自反思词频与响应长度上升、增益主要来自 policy gradient loss(区别于 grokking)、适度 entropy loss 助探索。强力支持"RLVR 主要是**激发**base 已有推理潜能、而非**注入**新能力"。【原文 §Abstract/§1/Fig.1-2/§4】
- 最巧的一步：**把训练集压到 1 条**这个极限实验设计本身(配 historical variance 选样)。抽掉"压到极限"——若仍用千条数据——就观察不到 post-saturation generalization(训练准确率一直没饱和，测试反而先掉，Fig.2 右),也就无法把"数据量"从增益里剥离、无法论证"base 已埋藏能力、RL 只是点火"。**这个极简化才是让机理结论可见的关键**(类似 dr_grpo 用极简配方让偏置可见)。【原文 §3.2.2/Fig.2】

## 为什么做
- 研究背景：RLVR(规则化二元 outcome reward 做 RL)已大幅推进 LLM 数学推理(o1、DeepSeek-R1、Kimi-1.5)，并伴随自反思等认知行为涌现与跨任务泛化。但社区精力多在**算法侧**(PPO/GRPO/DAPO 调稳),**数据侧**(要多少数据、什么数据最有效)相对被忽视。【原文 §1】
- 解决的具体痛点：① RLVR 训练集究竟能压缩到什么极限尚未探明;② 数据质量/数量如何关联自反思、跨任务泛化等经验现象不清楚。最相关前作 LIMR 用 LIM 分数把数据减约 6×仍保性能，但没探"能减到多狠"。【原文 §1】
- 相关工作 & 各自不足：LIMR(Li 2025，LIM 分数选样，减 6×)——没碰极限;"base 已具推理能力"系列(Liu 2025/Yue 2025/Gandhi 2025)——提出主张但未用"单样本即可"这种强证据佐证;算法侧大量工作(精炼 PPO/GRPO)——与数据效率正交。共性缺口:RLVR 的**数据下限**与"激发 vs 注入"机理无人系统回答。【原文 §1】
- 动机链：现状(RLVR 靠大数据集、增益机理被"涌现"叙事笼统解释)→ 问题("能把训练集减到多少还不掉点?")→ 实验发现(减到 1 条仍≈全集)→ 推断(单样本即可激发⇒能力本就在 base 里⇒RLVR 偏"激发/对齐"非"灌输")→ 所以(把研究重心从算法复杂度转向数据效率与机理)。【原文 §1】
- 与最近邻工作的 Δ：相对 **LIMR**——把"减 6×"推到"减到 1 条"，差在**揭示了数据量几乎可被压到零**这一质变，并由此暴露 post-saturation 等只在单样本下可见的现象。相对"base 已具能力"系列——本文给出**最锋利的单点证据**(1 example ≈ 1.2k ≈ 7.5k)，且跨模型(Qwen-Math-1.5B/7B、Llama-3.2-3B、R1-Distill)、跨算法(GRPO/PPO)、跨样本一致。【原文 §1/Fig.1/§3.3】

## 怎么做 + 靠不靠谱
- 方法流水线：输入(base 模型 + 单条数学题 + 二元 verifier) → ① 选样:先用全集(DSR-sub 1209 条)RLVR 训 E 个 epoch(论文用 500 步),逐样本记录"逐 epoch 训练准确率列表"→ 算其**方差(Historical Variance Score, Eq.1)** → 降序排名(Eq.2)，高方差样本如 π1/π13 → ② 把单条样本**复制到 128 份**凑够 batch(verl 的 dataloader 要求数据≥batch) → ③ 标准 verl GRPO:二元 0/1 outcome reward、KL 系数 β=0.001、熵系数 α=−0.001、rollout 温度 0.6(vLLM)、batch=mini-batch=128、每 prompt 采 8 条→每 rollout 步 8 次梯度更新、max prompt 1024/response 3072 → ④ 评测用 Qwen2.5-Math 官方 pipeline。【原文 §2/§3.1】
- 逐组件必要性：
  - **单样本(1-shot)本身**：核心对象。Tab.3 显示**很多样本**(高/中/低方差)单独训练都能涨≥30%(除错标 π1207、超难 π1208)——说明这是**普遍现象**而非特例。【原文 §3.2.3/Tab.3】
  - **Historical Variance Score 选样**：**非必要、非最优**——作者明确说许多中低方差样本也行;它只在 Qwen2.5-Math-7B 上略胜随机选(Tab.4)。**有消融**(隐含于"很多样本都行"的对照)。【原文 §2 脚注 2/§3.2.3】
  - **policy gradient loss**：增益**主因**。消融(Tab.5):只加 PG loss(Row 2)就接近完整 loss(Row 5);再加 weight decay(Row 3)/KL(Row 4)几乎无影响。【原文 §4.1/Tab.5】
  - **entropy loss(适度)**：助 post-saturation——无熵 loss 时 step150 后(训练准确率饱和点)几乎不再涨;加熵 loss 再涨 MATH500 +4.0%/AIME24 +2.5%(Tab.5 Row5),温度升到 1.0 再 +0.8%。**系数过大反而不稳**(Row 6)。**有消融**。【原文 §4.1/Fig.5】
  - **复制凑 batch**：纯工程必要(verl drop_last=True),非方法贡献。【原文 §3.1 脚注 3】
- 关键机制/公式(直觉)：选样准则就一个——**方差**:某样本在不同 epoch 训练准确率忽高忽低(方差大)，说明它"刚好卡在模型会与不会的边界"，反复给出有用的对错信号。但作者反复强调这**不是关键**——关键是"几乎任何不太难也没标错的题"都能点火。机理直觉:base(尤其 Qwen-Math)预训练已几乎会做 π1(Fig.3 显示 base 已能完成除最后一步开立方外的所有关键步,57.8% 输出"12.7"),RLVR 只是**把已有的正确推理路径的概率抬上来 + 鼓励探索其它表达**——所以单样本够用、且能跨类泛化(被点火的是"通用推理表达"而非"该题知识")。post-saturation 的诡异之处:训到 step1860 时模型对**训练样本**的输出退化成"多语言乱码夹杂正确算式"(Fig.3),但对**测试样本**的输出仍正常可读、准确率仍 74%——即过拟合了训练样本的表面形式却没损害泛化(作者归因于 zero-mean advantage 的 anti-overfitting:训练准确率饱和后 advantage→0、梯度自然变小)。【原文 §2/§3.2.1-3.2.2/Fig.3/§5】
- 实验与证据：
  - **核心并列(Fig.1/Tab.3)**：1-shot{π13}=35.7% ≈ 1.2k DSR-sub=35.9%;2-shot{π1,π13}=36.6% ≳ DSR-sub，≈ 7.5k MATH(36.7%)。MATH500:1-shot{π1}=74.0、{π13}=74.4 vs DSR-sub 75.2、MATH 75.4(基本追平)。【原文 Fig.1/Tab.3】
  - **跨模型/算法/样本**：Qwen2.5-Math-1.5B/7B、Llama-3.2-3B-Instruct、R1-Distill-Qwen-1.5B 均见大幅提升;GRPO 与 PPO 均可;Tab.3 多个样本均涨。【原文 §3.3/Abstract】
  - **非数学泛化(Tab.1)**：数学单样本训练竟提升 ARC-Easy/Challenge，且**超过全集 RLVR**(如 {π13}:ARC-E 55.8 vs MATH 全集 51.6 vs DSR-sub 42.2)。【原文 Tab.1】
  - **entropy-only(Tab.6)**：**只加熵 loss、完全无 outcome reward** 也能涨(1.5B:M500 36.0→63.4),但弱于 format-reward baseline(65.0);Llama/7B 同样(仅前几步)。说明"鼓励探索"本身就有(弱)增益。【原文 §4.2/Tab.6】
  - **label 鲁棒性(Tab.5 Row11-13)**：把标签从 12.8 改成正确的 12.7 性能≈;改成"4"(可猜可过拟合的错标)反而最差;改成"9292725"(完全猜不到的错标)居中(≈entropy-only)。说明轻微标签误差无害,但"可过拟合的错标"危害大于"完全错的标"。【原文 §4.2/Tab.5】
  - baseline 公平吗：与 format-reward baseline(只奖励"能解析出答案")对照以剥离"格式修正"增益,较严谨;全集/子集/单样本用同 pipeline、同模型对比,归因清晰。
  - "看着强但没回答核心问题"：核心(数据可压到 1 条 + 激发非注入)证据非常足;但"为何乱码仍泛化"只给观察未给完整理论。
- 假设与失效边界：
  - 【原文 §3.2.3/Tab.3】**失效样本**:错标(π1207)、超难到模型几乎采不到正确答案(π1208)——无有效 policy gradient 信号则失效。
  - 【原文 §3.3/Tab.1】Llama-3.2-3B-Instruct 增益相对有限(ARC 提升小、entropy-only 仅前几步)——暗示**对预训练较弱/非数学专长的 base，"激发"叙事打折**。
  - 【推断】现象集中在 **Qwen2.5-Math 系**(其 base 已在数学语料充分预训练)。"激发已有潜能"在数学潜能本就强的 base 上最成立;对真正缺乏该能力的模型,单样本无从"激发"。依据:跨模型实验里 Qwen-Math 增益最大、Llama 最小。
  - 【推断】**选择阶段不省算力**:Historical Variance Score 需先跑全集 RLVR 才能算,"1-shot 省的"只是 RL 训练阶段的数据,不是端到端算力。依据:§2 选样流程需先训 500 步。
  - 【原文 §3.2.2 + 推断】post-saturation 下训练样本输出退化为多语言乱码却仍泛化——作者给观察(归因 zero-mean advantage 的 anti-overfitting),**机理未充分解释**;过训(>1.4k 步)对超参(熵系数、步数)敏感。
- 祛魅总结【推断】：
  - 真贡献：**数据效率的极限证据**(1 example ≈ 全集)干净、可复核、冲击力强,且跨模型/算法/样本稳健,已成为"base 已具能力、RLVR 是激发"这一观点的标志性实证。post-saturation/跨类泛化/自反思增多等现象丰富,entropy-only 与 label 鲁棒性消融把功劳定位到 policy gradient loss + 探索而非数据量/正则。
  - 包装/被高估处：① **"1-shot 省算力"易被误读**——省的是 RL 数据不是选样算力;② **强依赖 Qwen-Math 系**,通用性有边界;③ "self-reflection 增多"是相关性观察(且 dr_grpo 已指出自反思≠更准),不宜过度解读为"学会反思";④ "single example sufficient"的措辞掩盖了"需是不太难、标对、且 base 已接近会做的题"这些前提(Tab.3 的失效样本)。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：规则验证器的二元 0/1 outcome reward(数学题答案对错);policy gradient(GRPO/PPO)的组内归一化优势驱动。属 RLVR,无教师/无蒸馏。
  - **改什么**：改模型参数(全量 RL 微调);不改结构/不加记忆。本质是"把 base 已有正确推理路径的概率抬高"。
  - **何时改**：在线 RL——每步对单样本采 8 条→算优势→8 次更新;训练可持续到百万次 rollout(单样本采 1024 次/步)。
  - **免梯度?**：否,核心是策略梯度;但揭示"policy gradient loss 是主因、weight decay/KL 可去",且 entropy-only(无 reward)也有弱增益。
  - **记忆-技能生命周期**：无外部记忆/技能库;"技能"=被激发的推理表达,固化进参数。无防遗忘设计——但意外展示 RLVR 的 anti-overfitting(zero-mean advantage 使训练样本被采样百万次仍不退化测试性能)。
  - **防遗忘机制**：无显式机制;zero-mean advantage 提供了一种"被动抗过拟合"(训练饱和后梯度自然趋零)。
- ⑦ 开源代码+框架/harness：https://github.com/ypwang61/One-Shot-RLVR (已开源，本地已 clone ~70MB)。**框架=veRL**(改编自 verl + rllm/DeepScaleR,仓库内含完整 `verl/` 目录),算法 GRPO(也验证 PPO);推理 vLLM 0.6.3;评测复用 Qwen2.5-Math 官方 pipeline(`Qwen2.5-Eval/`,latex2sympy)。模型/数据 HF ypwang61/one-shot-rlvr、ypwang61/One-Shot-RLVR-Datasets。关键超参:β=0.001、α=−0.001、温度 0.6、batch=mini-batch=128、每 prompt 8 rollout、max prompt 1024/response 3072。代码可得性高。【v1 仓库核查 + 本批 repo ls】
- 💰 资源/成本与可扩展性：单样本复制成 128 凑 batch;1.5B/7B/3B 规模(context 4096);相对全集省 RL **数据**(非选样算力);训练可跑到百万 rollout(单样本)。具体卡数原文未集中说明。【原文 §3.1】
- 🎯 对"探索-巩固"对标：**中-强支撑(机理层:base 蕴含可激发能力 + 探索的价值)+ 警示;非脚手架/蒸馏方法**。判定依据：① **"探索=发现自己能走通的开头"获最强背书**——Fig.3 显示 base 已能完成 π1 几乎全部关键步、只在最后一步发散,RLVR 把"自己本就能走通的路径"概率抬上来;这与本课题"teacher 偏向 student 自己能走通的开头/student on-policy 自选"的设计**正面共振**:可激发的上限由 base 决定。② **entropy loss 助 post-saturation** 支撑"探索是巩固持续生效的前提"——巩固阶段若不留探索,增益会随训练饱和而停。③ **zero-mean advantage 的 anti-overfitting** 给"巩固而不灾难性遗忘"一个朴素机制提示(训练目标饱和后梯度自衰)。**可借组件**:把 1-shot/few-shot 当**探索探针**——用极少"自己几乎能走通"的种子样本点火,再配 MTP 前瞻挑选种子;Historical Variance(卡在会/不会边界)的选样思路可迁移为"挑临界路径/关键步"。**缺口/警示**:① **无 teacher、无蒸馏、无回轨单点接管**(纯 self-play RL);② **无记忆/技能库、无 MTP**;③ **警示一**:增益强依赖 base 已具能力——对"student 走不通的开头"它无能为力,反证了本课题"teacher 脚手架"在 base 能力不足处的必要性;④ **警示二**:post-saturation 下训练样本退化为乱码——提醒"过度巩固单一路径"会侵蚀该路径的可读性/结构,本课题"固化进参数"需防此类塌缩。
- 🔭 开放问题/未来方向：
  - 【原文 §5/Appendix D.4】更好的数据选择/采集;把结论推广到其它领域与更弱 base;理解 post-saturation 乱码却泛化的机理;探索的更优诱导方式。
  - 【推断】用 MTP 前瞻识别"base 临界可走通"的种子题做 1-shot 点火,再用 teacher 稀疏脚手架补"走不通的开头";把 zero-mean advantage 的抗过拟合性质与"巩固防遗忘"显式结合,研究"巩固到何种程度路径开始塌缩"的边界。

RETURN: one_shot_rlvr|读到PDF=是(§Abstract/§1-5全文+Fig.1-3/Tab.1/Tab.3/Tab.5-6/§4.1-4.2机理)|L线=L3(兼L6)|对标=中-强支撑(base蕴含可激发能力背书"走通的开头"、探索助巩固;1-shot可作探索探针;但无teacher/回轨/MTP,且强依赖base能力+过巩固致路径塌缩是警示)|残留待核=0

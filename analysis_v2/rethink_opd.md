rethink_opd | Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe | 清华 THUNLP + 上海科技大学 + UIUC + 人大（Yaxuan Li/Yuxin Zuo/Bingxiang He 共一；Zhiyuan Liu/Ning Ding 通讯） | 2026-04-15 v2 · arXiv preprint cs.LG | 主题线 L1(OPD/自蒸馏)+L6(Token信用)·相关性 高

**原始论文**:https://arxiv.org/abs/2604.13016

## 一眼看懂
> 一句话导读:OPD 已是工业后训练标配,但它为什么有时灵、有时连更强的老师都教不动,一直是黑箱;本文把它解剖成"成败条件 + token 级机制 + 救活配方 + 适用边界"四块,核心发现是——OPD 真正学的是师生共享的少数高概率 token,而非吃分数差。

- 🟦 TL;DR:OPD(让学生自己采样,再用 teacher 的逐 token log-prob 当作稠密 reward 来纠正)已被 Qwen3/MiMo/GLM-5 大规模采用,但它的训练动力学是黑箱、而且很脆——**更强的 teacher 有时完全教不动学生,更弱的 teacher 反而能成功**。本文系统拆解四块:
  1. **现象学**——OPD 的成败由两个条件共同决定:(a) 师生的"思维模式"要兼容(即两者 top-k 分布的 overlap 要高);(b) 光是分高、思维一致还不够,teacher 还必须带有学生训练时没见过的"新知识";
  2. **机制**——成功的 OPD 会表现出:在学生访问的状态上,师生 top-k 高概率 token 的 overlap 渐进上升、熵差收窄;而且**这批 overlap token 数量虽少,却集中了 97-99% 的概率质量,单独训它们就够了**;
  3. **recipe**——两招救活失败的 OPD:① off-policy cold start(先用 teacher 的 rollout 做 SFT,把初始 overlap 拉高);② teacher-aligned prompt(改用 teacher 后训练数据的题目/模板);
  4. **代价**——稠密 reward 的可靠性会随轨迹变长而退化,且失稳是从轨迹尾部向前传播的,这暗示 OPD 难以直接扩到长程/agentic 场景。【Abstract,§1,图1】
- 最巧的一步:**§4.2 那个"只训 overlap token 就等价于训全量 top-k"的因果消融**。它的价值在于:抽掉它,全文就只剩"overlap 上升与成功相关"这种相关性观察。正是因为作者把 top-k 拆成 overlap / non-overlap 两部分分别训(发现 Overlap ≈ Student-TopK,而 Non-Overlap 显著更弱),再加上揭示"overlap 自增强"现象(reverse-KL 会把质量越堆越多、并把竞争 token 挤出 top-k),才把"高概率 token 对齐"确立为 OPD 的**因果作用点**,而不是一个副产品。【§4.2,图7】

## 为什么做
> 一句话导读:OPD 火、但为何/何时失败没人系统讲过;过去的蒸馏研究(经典 KD、OPD 谱系、容量差)要么只证"OPD 好"、要么只在 off-policy 设定下分析,留下了"OPD 何时灵"这个空白,本文来填它。

- 研究背景:OPD 已成为 LLM 后训练的核心技术(Qwen3/MiMo/GLM-5 都在用,Thinking Machines Lab 还以极低的 RL 成本复现了 Qwen3 的 recipe)。它与离策略蒸馏不同:离策略蒸馏在固定的 teacher 序列上训、会有 exposure bias(暴露偏差),而 OPD 是在学生**实际访问的状态**上、用 teacher 的逐 token log-prob 当 dense reward;它还被扩展到了自蒸馏(单个模型当自己的 teacher)。【§1,L67-82】
- 解决的具体痛点:OPD 存在一个显著的失败模式——更强的 teacher 反而完全教不动学生,更弱的 teacher 却能成功;而且很少有研究解释 teacher 的 token 级信号为什么/在什么时候能把学生分布推向期望方向、又在什么时候失败。【§1,L83-87】
- 相关工作的来龙去脉与精确短板(据 §7 三段):
  - **经典 KD / 序列级 KD**:Hinton 2015 把大模型的 soft 分布教给小模型;Kim&Rush 2016 把它扩到序列级(在 teacher 生成的序列上训学生),奠定了 off-policy 蒸馏基线(TinyBERT/DistilBERT/MiniLM 都属此族)。**共同短板**=训练与推理的分布失配,即 exposure bias(Bengio 2015):学生是在 teacher/参考序列上优化的,推理时却要从自己的分布生成,误差会沿着长生成不断累积——这正是"把蒸馏搬到学生 on-policy 分布上"的动机。【§7 L1464-1476】
  - **OPD 谱系(本文站在谁肩上)**,三项:
    1. **MiniLLM**(Gu 2023):首次为 LLM 形式化 OPD,用 reverse-KL + policy gradient,并论证 reverse-KL 的 mode-seeking 特性能阻止学生把概率质量摊到 teacher 认为不可能的区域;
    2. **GKD**(Agarwal 2024):给出统一框架,在 on/off-policy 数据与多种散度之间插值;
    3. **Yang2026b**(本文的最近邻理论):把 OPD 形式化为"dense KL-约束 RL"的一个特例,证明 teacher 的逐 token log-ratio 就等于一个隐式 reward,且**把这个 reward 放大到超过标准权重,可以把学生推过 teacher 的性能上界**。
    - OPD 已被工业后训练广泛采用(Qwen3/MiMo/GLM-5 等多篇 2026),并扩到 self-distillation(单模型借助"特权信息"——如 ground-truth 解、执行反馈——来当自己的 teacher,见 Hübotter/Shenfeld/Zhao 2026 等十余篇)。**共同短板**=这些工作都在"展示 OPD 好"(稠密 reward、缓解 exposure bias)、横跨各种 objective/任务/师生对,但**都没有系统分析它何时/为何失败**。【§7 L1477-1493】
  - **capacity gap / distillability(并行路线)**:Cho&Hariharan 2019 发现 teacher 远强于学生时蒸馏反而有害;Mirzadeh 2020 用一个中等的 TA(teacher assistant)来桥接;Busbridge 2025 给出"蒸馏 scaling law"——学生 loss 是 teacher 质量、学生尺寸、数据量的幂律,且存在一个 **U 型容量区**(teacher 过强反而降低蒸馏效率);Li2025 记录了"learnability gap"——小模型去学强推理 teacher 的长 CoT,效果反不如简单方法。**共同短板**=这些分析几乎全在 **off-policy KD** 下做,OPD 的 capacity gap/distillability **仍未被探索**。【§7 L1494-1508】
- 动机链:OPD 很火但是黑箱,且观察到"强 teacher 反而失败"这个反常现象 → 先找出成败条件(现象学) → 再看这些条件如何在 token 级体现出来(机制) → 既然 thinking-pattern gap 是可以训练缩小的 → 那就给出修复 recipe → 但稠密 reward 真的是免费午餐吗?最后再查 reward 本身的可靠性边界(代价)。
- 与最近邻工作的精确Δ:
  - vs **Yang2026b**(把 OPD 当 dense KL-RL、并证明放大 teacher reward 可越过 teacher 上界):本文不提新的 objective,而是去**诊断成败条件 + 揭示 overlap-token 的作用机制 + 给出救活配方**。
  - vs RL 阵营对 GRPO 的拆解:本文把"成功的签名"定位到 token 级的 overlap 动力学(overlap ratio↑、overlap-token advantage→0、熵差↓),并用 reverse distillation 干净地把"训练动力学"与"benchmark 分数"**完全解耦**(§3.3:R1-7B 分明明更高,却把学生拖回到与弱 1.5B teacher 相同的退化水平)。
  - 关键论断:**"OPD 本质是学 thinking pattern,而不是吃分数差"**——这是全文最反直觉、也最有信息量的一条。【§3.3,L579-598】

## 怎么做(到"读完能复现"的粒度)
> 一句话导读:本文不是新算法,而是一套"诊断工具箱"——三种粒度的 OPD 实现 + 三个能画成曲线的健康度指标(overlap/advantage/熵差)+ 一串受控因果实验,用它们逐层回答"OPD 何时成、机制是什么、怎么救、代价在哪"。

> 本文是**诊断范式**而非新算法:核心"方法"=一套可监控的 token 级动态指标 + 三类监督粒度的 OPD 实现 + 一组受控因果实验。下面把每一块写到输入→输出、咬合关系、默认超参。

### A. OPD 的三种监督粒度(数据流动 + 显存代价)
统一目标=**在学生自采轨迹上做序列级 reverse-KL**。具体地:给一个 prompt \(x\sim\mathcal D_x\),学生自回归地采样出 \(\hat y=(\hat y_1,\dots,\hat y_T)\sim\pi_\theta(\cdot\mid x)\);在学生生成的前缀 \(\hat y_{<t}\) 上,同时取学生和 teacher 的下一 token 分布 \(p_t(v)\triangleq\pi_\theta(v\mid x,\hat y_{<t})\) 与 \(q_t(v)\triangleq\pi_T(v\mid x,\hat y_{<t})\)。序列目标为
\(\displaystyle \mathcal L_{\text{OPD}}(\theta)=\mathbb E_{x\sim\mathcal D_x}\big[D_{\mathrm{KL}}\!\big(\pi_\theta(\cdot\mid x)\,\Vert\,\pi_T(\cdot\mid x)\big)\big],\)
按自回归因子分解成逐 token 之和(注意 KL 的方向是 \(p\Vert q\),即 reverse-KL):
\(\displaystyle \mathcal L_{\text{OPD}}(\theta)=\mathbb E_{x\sim\mathcal D_x,\ \hat y\sim\pi_\theta(\cdot\mid x)}\Big[\textstyle\sum_{t=1}^{T}D_{\mathrm{KL}}(p_t\Vert q_t)\Big].\)
三种实现的唯一区别是"每个位置评估多少个 token"(它们的输入都是同一批学生 rollout,输出都是回传给 \(\theta\) 的梯度):
1. **Sampled-token OPD(最常用、最轻)**:只评估学生实际采到的那个 token \(\hat y_t\sim p_t\),逐 token 损失为 \(\ell^{\text{sample}}_t\triangleq\log p_t(\hat y_t)-\log q_t(\hat y_t)\)。由于 \(\mathbb E_{\hat y_t\sim p_t}[\ell^{\text{sample}}_t]=D_{\mathrm{KL}}(p_t\Vert q_t)\),它是逐 token reverse-KL 的**无偏单样本估计**;每个位置只需查 teacher 的 1 个 log-prob,代价最低。Thinking Machines blog / MiMo / Yang2026b 都用这个。【§2.2 式3】
2. **Full-vocabulary OPD**:每个前缀都算全词表的 KL,梯度最稠密,但显存开销是 \(O(BTM)\)(\(B\) 为 batch、\(T\) 为序列长、\(M=|\mathcal V|\) 为词表大小)——大词表下很昂贵。【§2.2 式4】
3. **Top-\(k\) OPD(折中)**:把散度限制在学生的 top-\(k\) 子集 \(S_t=\text{TopK}(p_t,k)\) 上,在 \(S_t\) 内重归一化得到 \(\bar p^{(S_t)}_t,\bar q^{(S_t)}_t\),再算子集 KL \(D_{\mathrm{KL}}(\bar p^{(S_t)}_t\Vert\bar q^{(S_t)}_t)\)。它丢掉了 \(S_t\) 之外的质量、仍是 full 版的一个近似,但 teacher 查询代价显著降低,又保留了学生高概率区的多 token 监督。**默认 \(k=16\),用 Student Top-K 策略**。【§2.2 式5,Table 2】

### B. 三个诊断指标(全文都用来画监控曲线,均在学生 rollout 上计算,top-\(k\) 默认 16)
先约定记号:学生、teacher 各自的 top-\(k\) 集合记为 \(S^{(p)}_t=\text{TopK}(p_t,k)\) 与 \(S^{(q)}_t=\text{TopK}(q_t,k)\)。
- **Overlap Ratio(师生候选空间的对齐度)**:\(\;M_{\text{overlap}}\triangleq\mathbb E_t\big[\,|S^{(p)}_t\cap S^{(q)}_t|/k\,\big].\) 它趋近 1,说明学生已找到 teacher 的支撑区;它偏低,说明 mode mismatch(两者关注的候选集对不上)。【式6】
- **Overlap-Token Advantage(交集内的分布对齐质量)**:先在交集上重归一,得到 \(A_t(v)\triangleq\bar p_t(v)\big(\log\bar q_t(v)-\log\bar p_t(v)\big)\),再取均值 \(M_{\text{adv}}\triangleq\mathbb E_t\big[\frac{1}{|S^{(p)}_t\cap S^{(q)}_t|}\sum_{v}A_t(v)\big].\) **越趋近 0 越好**(说明学生在 teacher 偏好的 token 上,质量与置信都恰当);若是大负值,说明学生比 teacher 过自信(\(p_t\) 很高但 \(q_t\) 很低)。【式7】
- **Entropy Gap(熵差)**:\(\Delta H_t=|H(q_t)-H(p_t)|\),这是一个状态级的 mode 对齐指标,趋近 0 说明学生匹配上了 teacher 的不确定性轮廓。【式8】
- 附录还补了一个 **overlap mass**:\(M^{(p)}_{\text{overlap-mass}}=\mathbb E_t[\sum_{v\in S^{(p)}_t\cap S^{(q)}_t}p_t(v)]\)(teacher 侧同理)。实测发现师生的 overlap token 全程都占据 97%-99% 的概率质量(图18)——这正是"只训 overlap 就够"的概率依据。【附录B.1 式9/10】

### C. 现象学:两条件的受控实验(每个条件配一组对照)
- **条件1:思维模式要一致**(§3.1)。设置:学生固定为 Qwen3-1.7B-Base;两个 teacher 是 Qwen3-4B(Non-thinking)与 Qwen3-4B-Base-GRPO(对 Qwen3-4B-Base 跑 zero-RL/GRPO 得到)。两个 teacher 的 benchmark 分数相近(图3),但 base 学生与 GRPO teacher 的思维更接近——于是 **GRPO teacher 的初始 overlap 更高,蒸馏效果也更好**。值得注意:两条 overlap 曲线后期会收敛,但性能差却一直存在 → 说明**早期 thinking-pattern mismatch 造成的损失,后期补不回来**。【§3.1,图2/3】
- **条件2:新知识 ≠ 分数**(§3.2)。两族对照,各比较"同 pipeline 的 teacher" vs "经过额外 RL 的 teacher":
  - DeepSeek 族:学生 R1-Distill-1.5B,teacher 是 R1-Distill-7B vs Skywork-OR1-Math-7B(后者是对 7B 再加 RL);
  - Qwen 族:学生 Qwen3-1.7B(NT),teacher 是 Qwen3-4B(NT)vs Qwen3-4B-NT-RL-Math(后者在 DeepMath 57K 子集上加 RL)。
  - 指标用 **gap recovery rate(差距回补率)** \(=\dfrac{\text{Acc}_{\text{after OPD}}-\text{Acc}_{\text{before OPD}}}{\text{Acc}_{\text{teacher}}-\text{Acc}_{\text{before OPD}}}\)。
  - 结果:同 pipeline 的 teacher 增益有限;而经过 RL 的 teacher 回补率显著更高(**Qwen3-RL-Math 58.6% vs Qwen3 15.6%;Skywork 16.9% vs DeepSeek 5.3%**),且两者 overlap 都已经很高 → 说明增益来自 teacher 经 RL 获得的**新能力**,而非单纯的尺度更大。【§3.2,图4】
- **reverse distillation(一招同时验两个条件,§3.3 关键)**。背景:JustRL-1.5B 是对 R1-Distill-1.5B 做 RL 得到的;现在反过来,用 JustRL-1.5B 当学生,去蒸馏它自己的 pre-RL checkpoint(即 R1-Distill-1.5B),也试了更大更强的 R1-Distill-7B 当 teacher。两个结果:
  1. 蒸回它自己的 pre-RL checkpoint → 学生**几乎精确地退回 pre-RL 的性能**(RL 增益被抹掉)→ 证明 OPD 会主动吸收 teacher 的 thinking pattern、并**覆盖掉学生原有的**;
  2. 换成更大更强的 R1-Distill-7B → 训练轨迹**几乎无法区分、同样退化** → 因为在学生访问的状态上,reverse-KL 让这两个 teacher 诱导出了几乎相同的局部目标分布。
  - 三条结论:思维模式才是主导、benchmark 分数预测不了 OPD 的结果、高分 ≠ 新知识。【§3.3,图5】

### D. 机制:overlap-token 自增强(§4)
- **成功 vs 失败的签名**(§4.1):同一个学生 R1-Distill-1.5B,teacher 用 JustRL-1.5B(成功)对比 R1-Distill-7B(失败,尽管它更强)。成功的那一 run:overlap ratio 稳步上升、overlap-token advantage 升向 0、entropy gap 收窄——即学生在逐步定位 teacher 的高概率区、校准其中的质量、匹配局部置信;而失败的 run 这三个指标全部停滞。两点观察:① overlap token 全程占 97-99% 质量(所以 overlap 上升不是集合巧合);② overlap-token advantage 的改善说明 OPD 的主优化信号是在**交集区内重新分配质量**,而非在交集之外。【§4.1,图6】
- **因果消融(§4.2)**:固定那个成功的设置(JustRL-1.5B → R1-Distill-1.5B),只改一件事——"损失覆盖哪些 token",三选一(\(k=16\)):
  - (i) Student Top-\(k\):学生的全部 \(S^{(p)}_t\);
  - (ii) Overlap Top-\(k\):交集 \(S^{(p)}_t\cap S^{(q)}_t\);
  - (iii) Non-Overlap Top-\(k\):对称差 \(S^{(p)}_t\triangle S^{(q)}_t\)。
  - 结果:**Overlap Top-\(k\) 几乎等于 Student Top-\(k\)**(三个 benchmark 都如此),Non-Overlap 则显著更弱;且 Student / Overlap 两者的 overlap-token advantage 曲线几乎重合。结论:**OPD 的主增益来自共享高概率区的梯度**。【§4.2,图7】
- **自增强机制(直觉 + 真实形式)**:Student Top-\(k\) 和 Overlap Top-\(k\) 都把 overlap ratio 从约 72% 稳步升到 >91%,而 Non-Overlap 是先降后部分恢复。机制就是 reverse-KL 的 mode-seeking:一旦某个 token 进入了共享高概率区、且被 teacher 偏好,reverse-KL 的更新就会**给它堆更多质量、并把竞争的 non-overlap token 挤出学生的 top-\(k\)**——于是 overlap 区"因优化而越变越大",形成良性循环。统一论断:OPD 的主效应,就是**在学生访问的状态上,把学生分布在 teacher 支撑的那些高概率 token 上逐步精修**。【§4.2 末】

### E. Recipe:救活失败 OPD 的两招(§5)
- **off-policy cold start(冷启动,§5.1)**:分两阶段——先用 teacher 生成的 rollout 对学生做 SFT(拉近师生思维模式),再接标准 OPD。设置:学生 Qwen3-1.7B-Base,teacher Qwen3-4B(NT),prompt 取自 OpenThoughts3-1.2M 的数学子集;teacher 先生成 **200K** 条响应做 SFT、得到 Qwen3-1.7B-SFT,再用去重后约 **30K** 个 prompt 接 OPD。对照组是纯 OPD、直接从 Base 起步。结果:经 SFT 初始化的学生**起始 overlap 更高、轨迹更稳、最终天花板也更高**,且性能差全程持续(说明 cold start 既改善了早期优化、也抬高了最终上限);而纯 OPD 起点低、早期剧烈不稳。【§5.1,图8】
- **teacher-aligned prompt(让 prompt 向 teacher 对齐,§5.2)**,分两个粒度:
  - **模板层面**(teacher JustRL-1.5B,学生 R1-Distill-1.5B,题集同为 DAPO-Math-17K,只换模板):把标准 DAPO 模板换成 JustRL 后训练时用的那句 "Please reason step by step, and put your final answer within \boxed{}"。结果三个 benchmark 的精度和 overlap 增长都上升——说明连"换模板"这种微小改动,都能让学生的生成状态更兼容 teacher。【§5.2,图9】
  - **内容层面**(teacher Qwen3-4B-Base-GRPO,学生 Qwen3-1.7B-Base,等量对比 DAPO-Math-17K[与 teacher 的 RL 数据对齐] vs 去重后的 DeepMath 子集):teacher-aligned 的内容让下游更好,但有个反直觉之处——**overlap ratio 反而更低,而学生在 overlap token 上的累计质量却显著更高**(即质量集中到了更少、但共享得更强的 token 上)。代价是:**显著压低了学生的熵** → 所以作者建议混入一些 OOD prompt 来防熵坍缩。【§5.2,图10】

### F. 代价:稠密 reward 的可靠性边界(§6,全文最重要的一盆冷水)
- **长度甜区(§6.1)**:学生 R1-Distill-1.5B vs teacher JustRL-1.5B,跑 6 个不同的 max response length、各 200 步。结果:0.5K/1K 时监督 token 太少;**3K/7K 最强**;10K/15K 则 plateau 或下降,而且 10K/15K 在后期 overlap 会骤然崩塌,并伴随学生熵 / grad norm 的尖峰。【§6.1,图11a/12】
- **失稳是从尾部向前传播的(§6.1)**:在 15K 设置下,把学生的熵按输出位置画成热图,可见高熵**先出现在响应末尾,然后随训练逐步向前传播**(back-to-front,图13);teacher 的熵也呈现 suffix→prefix 的趋势——这与一个解释吻合:teacher 在越靠后的位置遇到越来越陌生的前缀,产出的 reward 越来越噪,进而把学生带崩。【§6.1,图13】
- **teacher 的续写质量随前缀加深而退化**:用 2K 个 DAPO prompt 生成 >16K 的学生 rollout,在多个位置截断后让 teacher 接着续写。teacher 的精度优势**从 1K 前缀时的 +0.37 单调降到 16K 前缀时的 +0.02**(图11b 给出 +0.3659/+0.2709/+0.1522/+0.0237)。结论:dense reward 在中等长度有效,但可靠性随深度退化 → **OPD 可能难以干净地扩到长 CoT/agentic 多轮**。【§6.1,图11b】
- **全局有信息 ≠ 局部可利用(§6.2)**:对每条学生 rollout 算一个序列均值 reward \(\bar r(y)=\frac1T\sum_t[\log\pi_T(y_t\mid x,y_{<t})-\log\pi_\theta(y_t\mid x,y_{<t})]\),再比较对、错 rollout 的分布。发现:**失败的 7B teacher 和成功的 1.5B teacher,都能让答对的 rollout 拿到更高的 reward,AUROC 相当(0.73 vs 0.75)** → 所以失败的原因不是信号质量差,而是局部优化的几何问题。
  - **各向异性梯度假说**(§6.2,作者明说**未直接验证**):7B teacher 的逐 token advantage 虽然单点上更大,但跨位置是各向异性的,聚合成梯度时会部分相消,导致有效梯度反而很小(实测确实如此:7B 的 overlap-token advantage 幅值更大、但 grad norm 持续更小);相比之下 JustRL-1.5B 因为思维兼容,advantage 集中在一个更连贯的 token 子集上,梯度方向一致,能被 reverse-KL 的 mode-seeking 放大。【§6.2,图14 + 附录B.2】
- **support size \(k\) 的影响(§6.3)**:学生 R1-Distill-1.5B、teacher JustRL-1.5B,比较 sampled-token 与 Top-\(k\)(\(k\in\{1,4,16,64\}\))。结果:**sampled-token ≈ Top-{4,16,64};只有 Top-1 明显更差**(图15:Top-4 为 0.473/0.331/0.793 vs Top-1 为 0.446/0.310/0.772 等)。Top-1 不稳定的原因是它总取 argmax,策略稍有变动就会翻转 rank-1、导致 reward 不稳;而 sampled-token 每步按学生分布抽不同 token,提供了对高概率区的无偏覆盖。结论:**\(k\) 不是关键设计,只要避开 Top-1 即可**。【§6.3,图15/16】

### 默认超参(可复现锚点,Table 1/2)
- **OPD 默认(Table 2)**:training temperature 1.0、global/mini batch 64、rollout number 4、LogProb top-\(K\)=16、Top-K 策略=Student Top-K、top-\(p\)=1.0、max prompt 1024 / max response 7168、lr 1e-6、epoch 1、**KL coeff = 0.0**。
- **teacher 的 GRPO(Table 1,造 Qwen3-4B-Base-GRPO)**:base Qwen3-4B-Base、rollout \(n=8\)、max prompt 1024 / max response 7168 / validation max 31744、lr 1e-6、temperature 1.0 top-\(p\) 1.0、关 KL、token-mean loss、1 epoch、8×A800-80G。
- **评测口径**:AIME2024/AIME2025/AMC2023,每题采 16 解,T=0.7、top-\(p\) 0.95、validation max 31744,主指标 avg@16。【§3.1,附录A.2 Table2】

## 靠不靠谱
> 一句话导读:对照设计极干净、且诚实标注了未验证的假设;主要软肋是全部实验都在数学域 + 中小模型,且"新知识"这个核心条件偏定性、没被干净量化。真正分量最重的是 §6 那盆"长程退化"冷水——它给整条 OPD 路线划了边界。

- baseline 公平吗:对照设计极干净(同学生/同数据/同 recipe,只换 teacher 或只换模板),reverse distillation 还用"同族不同 scale"来隔离"分数 vs 思维模式",公平性强;且作者诚实标注了未验证的假设(§6.2 的各向异性梯度假说就写明"未直接验证,留作 future work")。
- 看着强但没回答核心问题:全部实验都在**数学域 + 中小模型(1.5B-7B)**,作者自己承认没验证 code/开放域(§8);"新知识"这个条件依赖预训练语料的差异,但跨族蒸馏又混淆了 tokenizer/架构,而受控的预训练消融成本太高——所以"新知识"到底是什么,仍未被干净隔离(§8 已承认);另外 §6.2 那个"全局有信息 ≠ 局部可利用"的各向异性梯度解释,也只是一个 suggestive 的假说,未经证明。
- 假设与失效边界:
  - 【原文】① 所有结论都隐含一个前提——"teacher 的逐 token reward 在学生访问的状态上是可靠的",而这个前提会随轨迹深度而破裂(§6.1):reward 质量有甜区(3K-7K),太短则监督不足、太长(10K/15K)则后期崩,且失稳会 back-to-front 向前传播 → OPD 难干净扩到长 CoT/agentic;② sampled-token 够用、Top-1 不行。
  - 【推断】"思维模式一致是必要条件"意味着:跨族/跨范式(thinking↔non-thinking)的蒸馏天然吃亏;一个更大但思维迥异的 teacher,即便更强,也可能在学生策略附近诱导出一个局部平坦的 reward landscape(§6.2 假说),让 token 梯度失效。
- 祛魅总结:真贡献=把 OPD 从一份"经验配方"提升成了一套系统理解——它配齐了诊断指标(overlap/advantage/entropy-gap)、因果机制(overlap-token 自增强)、成败条件、救活 recipe、以及明确的代价(长程退化);其中 reverse distillation 那个"分数与动力学解耦"的结果极有冲击力。
  - 【推断】可能被高估的:"新知识"这个条件偏定性、没被干净量化,可能掩盖了"预训练数据 + 容量 + 范式"三者的混淆;且结论强绑定数学域、中小模型。
  - 被低估的是 §6 的"长程退化"——它是对当前 OPD 乐观叙事最重要的一盆冷水,却被放在了 Discussion 里。总之它**不是**提一个新方法,而是给整条 OPD 路线划清了边界、并配了一套体检指标。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=teacher 逐 token log-prob(reverse-KL 隐式 reward),核心作用集中在师生 top-k overlap token | **改什么**=学生参数 θ;OPD 本质改写学生在访问状态上的高概率 token 分布(并覆盖学生原有思维模式) | **何时改**=on-policy 在线、学生自采轨迹上;recipe 建议先 off-policy SFT cold start 再 OPD | **免梯度?**=否(reverse-KL policy-gradient);但揭示"sampled-token 单 token 即够"=极轻量 | **记忆-技能生命周期**=无显式记忆/技能库;知识=teacher 的思维模式被吸收进参数 | **防遗忘机制**=本文反而揭示 OPD 会"覆盖/抹掉"学生既有能力(reverse distillation 把 RL 增益抹回 pre-RL),即 OPD 在师生思维不一致时本身是一种**遗忘风险**;recipe 的 cold start/混 OOD prompt 间接缓解熵坍缩
- ⑦ 开源代码+框架/harness:https://github.com/thunlp/OPD(v1 元信息已核:约 195MB,verl fork + LlamaFactory);框架 **verl v0.7.0(OPD+GRPO,rollout 用 vLLM,verifier math-verify)+ LlamaFactory v0.9.5(SFT cold start)**;OPD 经自定义 ADV_ESTIMATOR=token_reward_direct 实现(teacher 当 reward model 给 token reward);其 overlap 诊断指标(overlap_ratio / overlap_token_advantage)已合并入官方 verl(PR #6469)。评测复用 thunlp/JustRL pipeline。【§1,v1 元信息】
- 💰 资源/成本与可扩展性:原文未给总 GPU 时(给了 teacher GRPO 用 8×A800-80G、1 epoch);给了关键扩展性结论=**稠密监督有"长度天花板"**(3K-7K 甜区,>10K 退化),且 sampled-token(每位置 1 token)即可达 Top-k 水平→可显著省 teacher 查询成本;cold start 需额外生成 200K teacher rollout 做 SFT。【§5.1,§6.1,§6.3,附录A.1】
- 🎯 对"探索-巩固"对标:**理论基石级支撑,直接为 TSRD 提供机制依据,同时也是一份边界警示**。
  - 强支撑(三条):
    1. "OPD 学的是 thinking pattern 而非分数" + "overlap token 是作用点" → 为 TSRD"让 teacher 当稀疏脚手架、教 path-selection"提供了 token 级理论依据:真正起作用的本就是**少量高概率的共享 token**,这与 TSRD 想做的"稀疏化/单点接管"天然契合(§4.2 已证只训 overlap 就够,正是"稀疏脚手架"的实证支持);
    2. "思维模式一致是必要条件" → TSRD 让 student 自选"能走通的开头/恢复分支",其实正是在保证师生兼容、抬高初始 overlap,与本文 §5(cold start / teacher-aligned prompt 拉 overlap)同向;
    3. "高分 ≠ 新知识" → 提醒 TSRD 在选 teacher 时,要找能提供 student 没见过的路径的,而不是单纯更强的。
  - **可借组件**:① overlap_ratio / overlap_token_advantage / entropy_gap 这三件套,可直接拿来当 TSRD 训练时的健康度探针;② cold start 拉 overlap 的那套两阶段范式。
  - **重大警示/缺口**:§6 的"长程 reward 退化 + 失稳从尾部向前传"正中 TSRD 痛点——TSRD 想用 MTP 做长程前瞻探针,但本文表明 teacher 的 dense 信号在长轨迹尾部本就不可靠。换个角度看,**这恰恰是 TSRD 用"MTP 前瞻 + 稀疏关键步接管"想解决的问题**:把稠密 reward 已经退化的尾部交给 sparse_critical 单点接管,而不是全程都上稠密监督。
  - 判定:**强支撑 + 给 TSRD 划清了"为何要稀疏/前瞻"的动机边界**。
- 🔭 开放问题/未来方向:【原文 §8】① 数学外(code/开放域)是否同条件同机制未知;② 预训练语料对"新知识"条件的影响难隔离(跨族混淆 tokenizer/架构);③ 自蒸馏动力学(思维一致天然满足、新知识来自特权信息)是自然下一步;④ 长程/agentic:提倡 hybrid——短段用 dense token 监督 + 长程用 sparse outcome reward,以及"渐进延长监督 horizon"的 curriculum。【推断】把 overlap-token 自增强机制与 MTP 前瞻结合做"提前对齐将访问的高概率 token";验证 §6.2 各向异性梯度假说并设计能利用各向异性 reward 的 objective;把"只训 overlap token"推到极致=显式稀疏脚手架蒸馏(与 TSRD sparse_critical 直接对接)。

RETURN: rethink_opd|读PDF?是(_txt 2273行,核到§2-§8全文+附录A/B超参表)|加厚?是(方法扩为A-F六块+默认超参,相关工作三段精确化,9条公式转MathJax)|LaTeX公式条数 12|待核数 1(§6.2各向异性梯度假说作者未验证)

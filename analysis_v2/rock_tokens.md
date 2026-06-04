rock_tokens | Cornerstones or Stumbling Blocks? Deciphering the Rock Tokens in On-Policy Distillation | UMBC + Case Western + Arizona State + VU Amsterdam(Yuxuan Jiang/Runchao Li/Shubhashis Roy Dipta 共同一作;通讯 Zhao Yang) | 2026-05-29 v2 · arXiv preprint(cs.CL) | L1 OPD/自蒸馏 · 相关性中

**原始论文**:https://arxiv.org/abs/2605.09253

## 一眼看懂
> 一句话导读:OPD 训练里有一批 token(多为格式符、空白、"So"/"Wait" 这类话语标记)始终学不动、却白白吃掉大量梯度;本文把它们命名为 Rock Tokens,并用因果探针证明绝大多数对答题没用,于是从一开始就冻结它们来换提速。

- 🟦 TL;DR:RLVR 领域早已知道"少数关键 token 撑起了推理增益",但 OPD 的 token 级动力学一直没人研究。作者发现:OPD 训练表观上饱和之后,仍有约 6% 的词表、占输出 18% 频次的 token 持续保持高 KL loss——把它们命名为 **Rock Tokens(顽石 token)**,内容多是格式符、空白、话语标记(如 "So"/"Wait")。它们带来两个悖论:① 贡献了不成比例的梯度范数,却在训练中"原地不动"、抵抗 teacher 的修正;② 用因果敲除(knock-out)一测,发现其中绝大多数对推理准确率并无关键贡献。结论:OPD 的"均匀 token 加权"把大量优化带宽浪费在了学生"学不会、也不必学"的结构性残差上;而从训练一开始就冻结这些 token 的梯度,可以在几乎不掉点的前提下提速(正文称 1.7×、take-away 处称 1.4×)。【原文】Abstract、§1、§4.3
- 最巧的一步:**inference-time token knock-out(推理时的 token 敲除)**——在每一步解码时,把某个 token 的 logit 强制设为 \(-\infty\),再看准确率变化 \(\Delta\)。这是一个把"梯度成本高"和"功能贡献大"两件事解耦开来的因果探针。抽掉它,全文就只剩"高 loss token 很顽固"这种描述性观察,既回答不了标题之问(它们究竟是基石还是绊脚石),也证明不了"冻结它们是安全的"。【原文】§3.2 Eq.17

## 为什么做
> 一句话导读:OPD 靠的是"每个 token 都算 KL"的稠密监督,直觉上高 loss 的 token 应该是关键修正点、会随训练收敛;但实测它们顽固不动还高频,于是本文追问:这些 token 到底是基石还是绊脚石?——这是 RLVR token 研究从没覆盖的角落。

- 研究背景:OPD(On-Policy Distillation)已是现代 LLM 后训练的基石(DeepSeek-V4、MiMo、Qwen-3 都在 SFT/RLVR 之外用它进一步榨取推理能力),本质上依赖"全 token 的稠密 KL 监督";而 self-distillation 的研究暗示,OPD 是一种放大潜能的通用机制,而非单纯的模仿。【原文】§1、§6.1
- 解决的具体痛点:在 OPD 的 per-token KL loss 里,high-loss token 是师生失配最直接的信号,按既有理解它们应随收敛而减少;但实证恰恰相反——有一批 token 持续保持高 loss(即 Rock Tokens)。由此引出两个悖论:(1) 它们因为高频,贡献了巨大的梯度范数,自身却停滞不动;(2) 从因果上看,它们对推理性能的贡献又可忽略。结果就是大量优化带宽被花在了结构/话语的残差上。【原文】Abstract、§1
- 相关工作的来龙去脉与精确短板(据 §6):
  - **OPD 用于后训练(§6.1)**:OPD 超越 SFT/RLVR 的地方在于"轨迹探索 + 稠密的 token 级指导"(Lu&Lab 2025);DeepSeek-V4/MiMo/Qwen-3 作为后训练阶段证明了它有效。但**已有研究 OPD 失败原因的工作,多盯着全局因子**——词表、推理风格、teacher 选择、模型族(即 rethink_opd / why_sd_degrade 那一类)。**本文要填补的空白**=聚焦**细粒度**的停滞来源(Rock Tokens),判定 OPD 的监督何时有用、何时只是无用噪声。
  - **token 级学习动力学(§6.2,直接对标)**:已有工作发现推理被少数 "anchor" token 不成比例地支撑着(推理时一扰动它们就显著掉点);RLVR 研究也揭示训练是非均匀的——集中在高熵的 "forking token"(决定性的分叉点,Wang et al.[70/71]),且 token 分布漂移主要发生在稀疏但关键的 token 上。**与本文的精确差异**:RLVR 是从 **entropy/reward** 来推断 token 重要性的,而本文用 OPD 独有的 **per-token reverse-KL** 直接度量师生失配、据此定义 token 类型,并发现"**高 KL ≠ 高功能价值**"——这个方向恰与 RLVR 的"高熵 = 高价值"相反;且本文还因果证明了 Pillarhood(基石性)与 entropy/频率/KL **正交**(|r|<0.07),从而警告:按 loss/熵加权的重要性方案,有压制 Pillar 的风险。
  - (§6.3 大段是无关领域的引用堆砌,与本工作弱相关。)【原文】§6.1-6.3
- 动机链:OPD 靠稠密 KL → 直觉上认为高 loss token = 关键修正点、应当收敛 → 实测它们却顽固不动且高频 → 那它们到底是 Pillar(基石)还是 Stumbling Block(绊脚石)?→ 用 What/Why/How 三阶段来诊断,并据此挑战"均匀加权"的做法。【原文】§1
- 与最近邻工作的Δ:见上 §6.2;核心一句——**用 per-token KL 来定义 OPD 的 token 类型,并发现高 KL 与因果重要性是正交的**,这正是 RLVR token 研究没覆盖到的角落。

## 怎么做(到"读完能复现"的粒度)
> 一句话导读:先给出 OPD 的 token 级 loss,再用两级打分(频率加权 + 上下文感知)把真正"跨相似上下文反复学不动"的 Rock Token 筛出来,然后用三个递进的 RQ 回答——它们还提供有用梯度吗?有没有因果功能?能不能安全冻结来提速?

### A. OPD 的 token 级 loss(基底)
OPD 在学生自采的轨迹上最小化逐 token 的 reverse-KL:
\(\displaystyle \mathcal L_{\text{OPD}}(\theta)=\mathbb E_{x_{1:T}\sim\pi_\theta}\Big[\textstyle\sum_{t=1}^{T}D_{\mathrm{KL}}\big(\pi_\theta(\cdot\mid x_{<t})\,\Vert\,\pi_T(\cdot\mid x_{<t})\big)\Big].\)
位置 \(t\) 处的 token 级蒸馏信号是 \(\ell_t=\log\pi_\theta(x_t\mid x_{<t})-\log\pi_T(x_t\mid x_{<t})\)。理想情况下 \(\ell_t\) 会随师生对齐而系统性减小、直到 plateau;但在长推理轨迹里,某些 token 的 occurrence(出现实例)即便在总目标已饱和后仍持续大失配——而这种持续高 loss,可能来自 **token 身份本身**、也可能来自 **局部上下文**(对空白/换行/标点/数字这类高频 token,尤其要把两者区分开)。【§2.1 式1/2】

### B. 识别 Rock Tokens:两级打分(§2.2)
两级打分的设计动机:一级先按"高频 × 高 loss"粗筛候选,但这会漏判——一个高频 token 可能只在少数位置难;所以再加二级,用上下文感知过滤掉那些"只在孤立位置难"的,只留真正"跨相似上下文反复难"的。
- **一级:聚合 Rock Score(候选筛选)**。先把 OPD loss 按 token 类型 \(v\) 分解:\(\mathcal L_{\text{OPD}}=\sum_v \mathrm{Freq}(v)\cdot\mathbb E[\ell_t\mid x_t=v]\),据此定义初始分数
\(\displaystyle R(v)=\bar\ell_v\cdot\mathrm{Freq}(v),\)
其中 \(\bar\ell_v\) 是末个 checkpoint 上 token \(v\) 的经验平均 token 级 loss。这里乘频率项是为了**抑制小样本噪声**:稀有 token 即便观测到的 loss 极端,也会被限权;只有"高频、且持续超基线 loss"的才拿高分。但 \(R(v)\) 单独还不够(高频 token 可能只少数 occurrence 难),所以加二级。
- **二级:occurrence 级 + 上下文感知过滤**。先把"持续高 loss 的 occurrence"定义为训练前、后都高 loss 的那些:\(O_{\text{PH}}=\{(i,t):\ell^{\text{pre}}_{i,t}\ge\tau_{\text{pre}},\ \ell^{\text{post}}_{i,t}\ge\tau_{\text{post}}\}\)。再给每个 occurrence 取一个局部上下文窗 \(c^{(w)}_{i,t}=x^{(i)}_{t-w:t+w}\)、编码成 \(h^{(w)}_{i,t}=f_{\text{ctx}}(c^{(w)}_{i,t})\),两个 occurrence 的上下文相似度记为 \(s(o,o')=\mathrm{sim}(h^{(w)}_{i,t},h^{(w)}_{j,k})\)。**只在同一 token 类型内**比较(\(O_{\text{PH}}(v)=\{o\in O_{\text{PH}}:x_o=v\}\)),定义上下文一致性分
\(\displaystyle \rho(o)=\frac{1}{|O_{\text{PH}}(v)|-1}\sum_{o'\in O_{\text{PH}}(v),\,o'\ne o}\mathbb 1[s(o,o')\ge\gamma],\)
保留 \(\rho(o)\ge\eta\) 的 occurrence,得到 rock occurrence 集 \(\mathcal R(v)\),其"上下文一致 rock 率"为 \(\mathrm{CCR}(v)=|\mathcal R(v)|/\mathrm{Freq}(v)\)。**最终的上下文感知 Rock Score**:
\(\displaystyle R_{\text{ctx}}(v)=R(v)\cdot\mathrm{CCR}(v),\qquad v\in V_{\text{rock}}\iff R_{\text{ctx}}(v)\ge\tau_R.\)
**这一步的作用**:防止那种"高频、但只在孤立位置难"的 token(比如个别空白/数字)被误判为全局顽固——它们的 CCR 低,最终分会被压下去;只有"跨相似上下文反复高 loss"的,才会被保留为真正的 Rock Token。【§2.2 式3-14】
- **经验估计**:用学生 rollout 来估计 per-token KL,即 \(\hat b_{\ell v}=\hat{\mathbb E}[D_{\mathrm{KL}}(\pi_\theta\Vert\pi_T)\mid x_t=v]\)(式15)。在 N=500 条 MATH-500 轨迹上画出"频率-KL"平面:稀有 token 由噪声主导,而 Rock Score 能把真正的 Rock Token(图中红点)隔在稳定频率带的上沿。【§2.3 式15,图2a】
- **cutoff \(K\) 的选取**(§2.4):扫一遍 Top-K——K 小则稳但覆盖低,K 大则覆盖高但抗噪差;**最优交点是 K=100**(覆盖约 60% 的语料级 KL,且 Jaccard 在 \(n\in[50,400]\) 区间稳健)。最终 Rock Tokens 约占 6% 词表、中位数占 18.5% 的输出频次。把它们 decode 出来是四类:LaTeX/数学定界符、Markdown/空白结构、话语标记("So"/"Wait")、数字——也就是说,**学生抵抗的是"怎么组织推理结构",而不是"生成什么内容"**(作为对照,频率匹配的对照集 \(S_{\text{ctrl}}\) 多是内容词)。【§2.4-2.5,图2c/d】

### C. RQ1:Rock Tokens 还提供有用的学习信号吗?——Gradient Paradox(梯度悖论,§2.6)
做法:对每个 token 类型算它的均值 logit 梯度 \(\bar g_t=\frac1{n_t}\sum_i g_i\),再把它对全局下降的贡献分解为
\(\displaystyle \mathrm{contrib}(t)=n_t\cdot\Vert\bar g_t\Vert\cdot\cos(\bar g_t,\ G_{\text{balanced}}),\qquad G_{\text{balanced}}=\textstyle\sum_t\bar g_t,\)
其中 \(G_{\text{balanced}}\) 是一个**频率均衡的参考方向**(让每个 token 类型等权,故意去掉常见 token 的频率主导)。三个发现:
- **梯度幅值低、但靠高频取胜(图3a)**:Rock 单次的梯度幅值远小于稀有高 KL token(中位数 \(\Vert\bar g_t\Vert\approx0.016\) vs \(0.54\),Mann-Whitney \(p<10^{-30}\));但上面 Eq.16 的乘法分解显示,它的主导项其实来自**频率因子 \(n_t\)**——也就是"小信号 × 海量出现",最终在权重更新里形成了主导性的聚合力。
- **方向却与全局优化对齐(图3b)**:这一点**反直觉**——Rock 的梯度方向与 \(G_{\text{balanced}}\) 是**正对齐**的(中位数 cos \(\approx0.040\) > 高 KL 的 0.025 > random 的 0.006,尾部甚至 cos>0.3)。也就是说,每个 rock 在很多位置都贡献着"小、但方向一致"的信号;从 teacher 的视角看,Rock Tokens 其实指向的是"正确"的优化方向。
- **可它在训练中停滞(图3c/d)**:对比早、晚两个 checkpoint 的 per-token KL——稀有高 KL token 显著下降(中位数 2.02→0.85,降约 58%,Wilcoxon \(p<10^{-8}\));而 **Rock 几乎不变(0.21→0.19),\(\Delta\)KL 集中在 0**;random 则是对称噪声、无净变化。几何上的解释是:稀有高 KL token 的梯度方向独特,聚焦更新就能把它单独解决掉;而 Rock 的梯度高度对齐全局方向,**"专门去削减它"与"跟随全局下降"两件事变得无法区分**,于是它得不到一个 token 专属的学习信号去克服自身的 loss。
- **一个假设(作者明说未验证)——Adam 抑制**:在这种重尾的类别不平衡下,预训练时 Adam 压抑高频 token 本是好事(防过拟合),但 OPD 恰恰需要更新这些高频结构 token 来匹配 teacher 的风格;Adam 会膨胀 Rock 的二阶矩、系统性压低它的有效学习率,把学生困在既有习惯里。【§2.6 式16,图3,Adam 假说留作 future work】

### D. RQ2/RQ3:因果功能价值 + 选择性蒸馏(§3-§4)
- **inference-time knockout(§3.2,RQ2 的因果探针)**:构造一个 knockout 策略 \(\pi^{\backslash v}_\theta\)(在解码的每一步,把 token \(v\) 的 logit 设为 \(-\infty\)),度量它的 causal delta
\(\displaystyle \Delta_B(v)=\mathrm{Acc}_B(\pi^{\backslash v}_\theta)-\mathrm{Acc}_B(\pi_\theta).\)
若 \(\Delta_B(v)\le-0.01\)(即敲掉它准确率明显下降)就判为 Strong Pillar。候选池取 \(|\tilde{\mathcal R}|=200\)(刻意超出核心的 K=100,以便搜到稀有的因果效应),用 paired-bootstrap、\(\alpha=0.05\)、10000 次 resample。结果(图4):**MATH-500 上是 7 个 Pillar / 0 个 Stumbling / 193 个 Neutral;IFEval 上是 3 / 0 / 197**。三个论断:
  1. Pillar 罕见但致命(只占 3.5% / 1.5%);
  2. **方向不对称——Strong Stumbling 为 0**:学生顽固偏离 teacher,几乎从不有害,要么是必要的锚点、要么是无害的风格偏好;
  3. **Pillarhood 与传统指标正交**(与 entropy/log-freq/residual KL 的 |r| 都 <0.07)→ 所以按 loss/熵加权的重要性方案,有压制 Pillar 的风险。【§3.2-3.3 式17,图4】
- **window-aware 选择性蒸馏(§4.2,RQ3)**:因为 Rock 是通过"局部高散度窗"识别出来的(不是孤立 token 的尖峰),所以重加权的对象是"Rock token ∪ 它的局部窗"。设 Persistence Set 为 \(\mathcal R\)、每个候选 token 的局部高散度窗为 \(W(v)\),取并集 \(W_{\mathcal R}=\bigcup_{v\in\mathcal R}W(v)\)(式18)。加权后的 OPD 目标:
\(\displaystyle \mathcal L_{\text{weighted}}=\mathbb E_{x\sim\pi_\theta}\sum_{t=1}^{T}w(x_t,t)\cdot\ell_t,\qquad w(x_t,t)=\begin{cases}\lambda,&x_t\in\mathcal R\ \text{或}\ t\in W_{\mathcal R}\\1,&\text{otherwise}\end{cases}\)
对比三个 regime:① **Baseline OPD**(\(\lambda=1\));② **Rock-Freeze(本文方法,\(\lambda=0\),即 "Just Not Train")**;③ **Freq-Matched Window Freeze(Random 对照,对频率/窗长都匹配的随机窗 \(W_S\) 同样取 \(\lambda=0\))**。设 Random 对照是为了排除"冻任意 token 都无害"这个平凡解。判定逻辑:冻掉 Rock 窗后,若性能维持/提升 → 它们更像 Stumbling Block;若显著掉点 → 它们更像 Pillar。【§4.2 式18/19】
- **结果(§4.3,图5)**,三条:
  1. Rock Tokens **不是纯噪声陷阱**——它们提供的梯度对优化是有锚定作用的(Original OPD vs 本文方法存在性能差,全冻会掉点);
  2. **Random 对照严重掉点,而 Rock-Freeze 维持了高准确率上限**——说明"选择性降权"远比"无差别减信号"安全;
  3. Rock 约占 18% 的输出 token,降低对它们的优化压力可换得 **1.7× 的 wall-clock 提速**(但 take-away 处又说对 30% 高成本 token 做 freeze-weighting 得 **1.4×**,两处口径不一,宜取保守值)。【§4.3,图5】

### 实验设置(复现锚点,§5)
- **两阶段蒸馏(KDFlow 框架)**:teacher = Qwen3-30B-A3B-Instruct-2507(MoE,激活 3B),student = Qwen3-4B-Instruct-2507,**两者都关闭 thinking**。
  - Stage-1(off-policy):在 OpenThoughts3 的 **20k** 条 teacher 解上训 1 epoch,用 forward KL(kd_ratio=0.5)+ 交叉熵,做一次全词表的初始对齐;
  - Stage-2(on-policy):从 Stage-1 的 checkpoint 起,在另外 **10k** 个 prompt 上、每个 prompt 采 4 条 rollout(共存 7 个 checkpoint),用纯 reverse KL(kd_ratio=1.0)训 1 epoch;其余超参沿用 KDFlow 默认(附录C)。
- **硬件**:4×H100(80GB),FSDP2 + bf16 + gradient checkpointing。
- **评测**:用 LM-Eval-Harness 做 zero-shot Pass@1(取 5 次平均)。主指标 = AIME24 + AIME25 + HMMT25-Feb(各 30 题,合 90 题);MATH-500 + IFEval 则用于做大样本的 token 级统计。图5 横轴每 200 步 = 8000 prompt × 4 rollout。【§5.1-5.2】

## 靠不靠谱
> 一句话导读:实验只在单一师生对、关 thinking 模式下做,且核心假设(Adam 抑制)未验证;最该记住的纠偏是——它不是说"Rock Tokens 全无用",而是说"选择性、缓和地给高成本 token 降权来提速",因为其中仍有 1.5-3.5% 是不可或缺的 Pillar。

- 假设与失效边界:
  - 【原文】只用了单一师生对(30B-A3B → 4B)、且关闭 thinking 模式;"Adam 抑制"只是一个**假设**,作者明说留待未来做形式化验证(§2.6)。
  - 【推断】① "功能贡献可忽略"这个结论是基于数学/IFEval 的准确率得出的,对那些强依赖格式/话语连贯的长文档/对话任务未必成立;② knockout(推理时屏蔽)与 freeze(训练时不更新)本是两种不同的干预,二者结论之间的可迁移性并未被严格论证。
  - 【推断】Rock 集合是基于"最终 checkpoint + MATH-500 轨迹"得到的,它在跨 tokenizer、或更大跨度师生下的迁移性还未验证。
- 祛魅总结:
  - 真贡献:首次对 OPD 的 token 级动力学做了系统诊断;提出了两级(含 context-aware)的 Rock Score 与 knockout 因果探针;揭示了"Gradient Paradox"以及"Pillar 极少但致命、Stumbling 为零"这个非对称结构;并给出了一个"非均匀梯度分配"的可落地提速方案。【推断】
  - 比 v1 旧分析更准的关键纠偏——**它并不是说"Rock Tokens 全无用"**:
    - RQ1 表明它们提供的梯度方向是正确的(全冻会掉点);
    - knockout 表明其中 1.5-3.5% 是不可或缺的 Pillar,且这些 Pillar 与 entropy/频率/KL **正交**(|r|<0.07);
    - 作者据此警告"按 loss/熵加权的重要性方案有压制 Pillar 的风险"。
    - 所以本文真正的贡献是"**选择性、缓和地**给高成本 token 降权来换提速",而不是"把它们砍掉"。【原文】§2.6、§3.3
  - 可能高估之处:加速幅度文中两处口径不一(1.7× vs 1.4×、18% vs 30% token),宜取保守值。【推断】

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号:per-token reverse-KL(师生失配)+ knockout 因果 Δaccuracy。
  - 改什么:OPD 的 token 级梯度加权(对 Rock token 及其窗口赋权 λ)。
  - 何时改:从训练**一开始**就冻(非后期),且基于离线识别的固定 Rock 集。
  - 免梯度?否,是梯度蒸馏(reverse-KL OPD);"freeze"指对子集梯度置零以省算力。
  - 记忆-技能生命周期:不涉及外部记忆/技能库,纯参数内蒸馏。
  - 防遗忘机制:无显式防遗忘;但"只降权冗余 token、保留 Pillar"间接保护学生既有推理框架(path dependency 解释)。
- ⑦ 开源代码+框架/harness:https://github.com/YuxuanJiang1/Rock-Token(v1 已克隆 ~7.4MB,三目录 KDFlow_localopd/rock_detection/stumbling)。框架=自定义 **KDFlow**(songmzhang/KDFlow),后端 SGLang+Ray+FSDP2+bf16;README 致谢显示模型/分布式抽象沿用 **OpenRLHF**、on-policy KD 的 Ray placement-group 与 SGLang 权重更新借鉴 **slime**,rollout 用 SGLang;评测 LM-Eval-Harness。即 OpenRLHF(训练抽象)+slime(on-policy KD)+SGLang(rollout)。【原文】§5.1 + 仓库(v1 核查)
- 💰 资源/成本与可扩展性:硬件 4×H100(80GB),FSDP2+bf16+gradient checkpointing,超参沿用 KDFlow 默认(Appendix C);收益=对高成本 token 冻梯度换 1.4-1.7× wall-clock 提速。【原文】§5.1、§4.3
- 🎯 对"探索-巩固"对标:以**可借组件**为主,是弱竞品。
  - 判定:它的"识别学生顽固抵抗的脚手架 token + 区分 Pillar/Neutral",与本项目的"path-token 选择性监督"同源——为"哪些 token 该被 teacher 监督、哪些是学生自有、可保留的脚手架"提供了一个**因果级判据(knockout)**,可直接迁移到 TSRD 的"巩固/回轨 token 选择"。但它做的是**降权/冻结**(减少信号),与"探索 = 偏向自己走得通的开头"恰是反向操作,且它没有 path-recovery 的概念,所以不构成竞品。
  - **关键警示**:§3.3 发现 Pillar 与传统指标正交(|r|<0.07)→ 提示本项目**不能简单地按高熵/高 KL 去选监督 token**(那样会误伤 Pillar)。
  - 这与 rethink_opd 的"overlap token 是作用点"、rho1 的"excess-loss 选 token"形成了一种有益的张力——三者都在追问"哪些 token 该被监督",但 rock_tokens 给出的是一个反向警告:"高 KL ≠ 就该监督"。依据:§3.3。【推断】
- 🔭 开放问题/未来方向:【原文】Adam-suppression 假设的形式化验证(§2.6);Pillar 的可预测性(目前与所有传统指标正交,§3.3)。【推断】跨任务/跨 tokenizer 的 Rock 集迁移;thinking 模式与更大跨度师生下是否成立;把 knockout 的因果 Pillar 信号反哺到 token 级加权方案(替代按 loss/熵加权)。

RETURN: rock_tokens|读PDF?是(_txt 850+行,核到§2-§6全文+式1-19含v1漏掉的context-aware式5-14)|加厚?是(补全两级Rock Score+window-aware式18/19+两阶段KDFlow配置,相关工作§6.2精确化,19条公式转MathJax)|LaTeX公式条数 11|待核数 1(跨任务/跨tokenizer Rock集迁移性)

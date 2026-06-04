beyond_loglik | Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum | UIUC（共一 Gaotang Li、Ruizhong Qiu、Xiusi Chen;Heng Ji、Hanghang Tong）| 2026-05（arXiv:2510.00526v3;ICML 2026，PMLR 306）| 主题线 L2 统一SFT-RL/SFT损失设计·相关性 中

**原始论文**：https://arxiv.org/abs/2510.00526

## 一眼看懂

> 一句话导读：大家都怪"SFT 泛化差是因为它在搞模仿";本文说真正的病根是 SFT 默认用的那个损失（NLL）。它把损失推广成一个可调的参数族,并指出该用哪种损失,取决于 base 模型在该任务上有多强。

- 🟦 TL;DR：SFT 默认用 NLL（即 \(-\log p\)，负对数似然,等价交叉熵）。一直以来大家把"SFT 泛化差"归咎于模仿学习这套范式。本文说不对——病根其实是 NLL 这个**默认目标**本身。理由是:NLL 只有在"从零训练一个小分类器"时才最优（Cox 1958 等）;而后训练的场景完全不同——base 模型已经有先验、监督序列又长又含噪,这时还逼模型去逐 token 复刻冗长的 CoT,反而伤泛化。于是本文把 NLL 推广成一个参数族 \(f^\alpha(p)=\frac{1-p^\alpha}{\alpha}\)，并提出一个统一刻画——**模型能力连续谱（model-capability continuum）**:
  - base 先验强（如数学）时,那些"下调低概率 token 权重"的 prior-leaning 目标（如 \(-p\)）持续胜过 NLL（Qwen2.5-Math-1.5B 上 **+15.75**）;
  - base 先验弱（如 figfont 谜题）时,NLL 反而占优;
  - 中间地带（如医疗）则两者难分高下。
- 最巧的一步：**对梯度权重 \(W_f(p)=-f'(p)\,p\,(1-p)\) 做凸凹分析（Lemma 3.1 + Prop 3.2）**。\(W_f(p)\) 描述的是"一个 token 按它当前的预测概率,该贡献多大的梯度"。抽掉这个分析,全文就只剩"换个损失有时更好"的经验观察。正是 \(W_f\) 把下面这件事讲清楚了——凸目标（如 \(-\log p\)）的峰值落在 \([0,\tfrac12]\),会强调低概率 token,属于 prior-averse;凹目标（如 \(-p\)）的峰值落在 \([\tfrac12,1]\),会强调高概率 token,属于 prior-leaning。有了这把刀,才能把一堆散落的损失（\(-p\)、focal、entropic matching、均匀重加权）统一进同一个谱系,并预测它们各自何时该用。

## 为什么做

> 一句话导读：把"SFT 泛化差"全怪到模仿范式头上,等于忽略了一直没人质疑的默认损失 NLL;而 NLL 的最优性在后训练场景根本不成立。

- 研究背景：SFT 是 LLM 后训练的标准做法（Zhang 2026b、Chung 2024）,但**泛化常常受限**,大家普遍把原因归到"模仿学习范式本身"（§1,引 Ouyang 2022、Chu 2025）。
- 解决的具体痛点：把锅甩给"模仿范式",其实**忽略了默认目标 NLL 本身**。NLL 在"从零训练分类器"上确实经典最优,但后训练的场景不一样:base 模型已经编码了任务先验、而且通常 **well-calibrated（置信度校准得不错）**（Zhu 2023、Xie 2024）;监督又是上千 token 的长 CoT、可能含噪。在这种情况下,**强行要求 base 逐 token 复刻,反而伤泛化**。与此同时,RL 启发的各种 SFT 改进虽然各自在某些域有效,却**缺乏一个"何时该用哪种目标"的统一刻画**。
- 相关工作 & 各自不足（§2）：
  - **从 RL 视角改 SFT**——把 SFT/DPO 看隐式 reward learning（小 lr + 散度目标,Wang 2026）、引入 importance sampling（Qin & Springenberg 2025）、PPO-style clip 约束 drift（Zhu 2026）、**均匀重加权梯度系数（Wu 2026,本文证明等价 \(-p\)）**——各自有效但**不知何时失效**。
  - **其他 SFT 损失**——MSE / focal loss（Lin 2017）/ Huber（Huber 1992）/ entropic distribution matching（Li 2025,加正则项）——本文框架可统一解释为 prior-leaning。
  - **定位（§positioning）**——提供**首个 capability-based 的 SFT 目标统一刻画**;明确**不主张单一万能损失**（"not advocating a single one-size-fits-all loss"）,而是"establish a principled account of **when and why** objectives trade advantages"。
- 动机链：
  - 起点是现状:SFT 泛化差、大家归咎于模仿范式、且一律默认用 NLL。
  - 暴露的缺陷有两条:一是 NLL 的最优性假设在后训练被违反;二是没有一个"何时该用何损失"的统一理论。
  - 于是提出 \(f^\alpha\) 族 + 能力连续谱。
  - 为什么不干脆推一个"更好的损失"就完事？因为实证发现损失的好坏会**随 base 能力反转**——在 MS 端（base 强）\(-p\) 赢,在 MW 端（base 弱）NLL 赢。既然如此,单一损失不可能万能。
- 与最近邻工作的 Δ：
  - vs **Wu 2026（均匀重加权,本文证明它等价于 \(-p\)）**——本文证明这种 prior-leaning 更新**只在 MS 端有效、在 MW 端反而有害**,从而把它的单点结论纳入整条连续谱。
  - vs 各种 SFT 损失工作——本文提供 \(W_f\) 凸凹这把统一的分析刀。
  - 关键点:损失的选择不是绝对的优劣,而是与 base 先验强度的**匹配**问题。

## 怎么做 + 靠不靠谱

> 一句话导读：这不是一个新算法,而是一套"该用哪种损失"的分析框架。核心就一把刀——梯度权重 \(W_f(p)\)：它的形状（凸还是凹）决定了梯度更偏向低概率 token 还是高概率 token,进而决定该损失适合 base 强还是 base 弱的任务。

> 注:本文是**分析/认识论框架**而非新算法。设定:后训练的 base 为 \(p_\theta\)（已 well-calibrated、含任务先验）;SFT 数据 \(\mathcal T\) 是一批 \((x,\tilde y)\) 对,其中 \(\tilde y=(y_1,\dots,y_N)\);decoding 第 \(t\) 步的 logits 记为 \(z_t\)，对应概率 \(p_t=\mathrm{softmax}(z_t)\)。

**【统一框架（§3）——可复现的目标族】**
1. **标准 SFT = NLL/交叉熵**：\(\mathcal L_{\log(p)}(\theta)=\mathbb E_{(x,\tilde y)\sim\mathcal T}\big[-\sum_{t=1}^N\log p_\theta(y_t\mid y_{<t},x)\big]\)。
2. **通用概率目标族**：对任意可微非增 \(f:[0,1]\to\mathbb R\)，
   \(\displaystyle \mathcal L_{f(p)}(\theta)=\mathbb E_{(x,\tilde y)\sim\mathcal T}\Big[\sum_{t=1}^N f\big(p_\theta(y_t\mid y_{<t},x)\big)\Big].\)
   一个有用实例:\(f^\alpha(p)=\dfrac{1-p^\alpha}{\alpha}\)。**\(\alpha\to0\) 退化 NLL**（\(f^\alpha\to-\log p\)）;**\(\alpha=1\) 即 \(-p\)（plain-p，=最大化期望平均预测准确率,因 \(1-p\) 对应 0-1 loss 期望）**;**\(\alpha\ge1\) 凹 / \(0\le\alpha\le1\) 凸**。
3. **梯度形状（Lemma 3.1,理论核心）**：对 logits 的梯度为
   \(\displaystyle \frac{\partial \mathcal L_f}{\partial z_{t,i}}=s_f(p_{t,y})\,(\delta_{i,y}-p_{t,i}),\qquad s_f(p):=-f'(p)\,p\ \ge0,\)
   其中 \(\delta_{i,y}=\mathbb 1\{i=y\}\)。对正确类 \(i=y\)：
   \(\displaystyle \frac{\partial \mathcal L_f}{\partial z_{t,y}}=s_f(p_{t,y})(1-p_{t,y})=W_f(p_{t,y}),\qquad \boxed{W_f(p):=-f'(p)\,p\,(1-p)}.\)
   \(W_f(p)\) **决定每个 token 按其当前预测概率贡献多少正确类梯度**。
4. **凸凹归类（Prop 3.2,把"损失形状"和"该强调哪类 token"对应起来）**：设 \(f\in C^2[0,1]\)、\(f'(p)<0\)。结论是:
   - **\(f\) 凹 ⇒ \(W_f\) 的峰值落在 \([\tfrac12,1]\)**——强调高概率 token,属 prior-leaning。
   - **\(f\) 凸 ⇒ \(W_f\) 的峰值落在 \([0,\tfrac12]\)**——强调低概率 token,属 prior-averse。
   - 套到参数族 \(W_f(p)=p^\alpha(1-p)\) 上看几个特例:\(\alpha\to0\)（即 NLL）时 \(W_f\to(1-p)\)，强压低概率 token;\(\alpha\ge1\)（即 \(-p\)）时,低概率处的信号迅速衰减;另一个特例 \(f(p)=-\log(1-p)\) 得 \(W_f=p\)，完全偏向高概率 token。
5. **prior-leaning vs prior-averse（Def 3.3）**：按 \(W_f\) 把梯度的质量集中在阈值 \(\tau\) 之上还是之下来分类。集中在 \(\tau\) 之上（中-高概率 token）意味着"利用先验、精炼已经可信的预测",即 prior-leaning;集中在 \(\tau\) 之下（低概率 token）意味着"逼模型从不可信的预测里学",即 prior-averse。阈值 \(\tau\) 不唯一、随任务变,但 \(-\log p\) 与 \(-p\) 之间有明确对比。
6. **hard-thresholding 变体（Eq.4,关键消融工具）**：只在概率区间 \(I\subseteq[0,1]\) 内更新——
   \(\displaystyle \mathcal L_{HT(I),f(p)}(\theta)=\mathbb E_{(x,\tilde y)\sim\mathcal T}\Big[\sum_t f\big(p(y_t\mid\cdot)\big)\,\mathbb 1\{p(y_t\mid\cdot)\in I\}\Big].\)
   用于隔离各概率段 token 的贡献（分位阈值由**训练前**的 base 模型预测概率算出）。
7. **能力连续谱定位（§3 末）**：怎么判断一个任务该放在谱的哪一端?给了两个视角。
   - **数据侧**——看语料里相关 token 的占比:LLaMA-3 报告约 25% 是数学 token,所以数学属 MS（base 强）;figfont 完全不在语料里,属 MW（base 弱）;医疗只部分覆盖,属 MI（中间）。
   - **模型侧**——用**训练集上的平均预测概率**当先验强度的代理:数学 0.76–0.81、医疗约 0.50、figfont 约 0.01（思路类比 lm-eval 用 log-likelihood 评 base）。
   - 落到选择上:MS 端用 prior-leaning,MW 端用 prior-averse（即 NLL）,MI 端两者皆可。
- **逐组件必要性**：
  - **\(W_f\) 凸凹分析**——理论核心,没它无法预测谱位置。
  - **hard-thresholding 变体**——§5.1 Fig.5 用它隔离各概率段:① **低概率 token 一致有害**（所有目标在低分位都掉）;② **只训 top10% token 时所有目标都超标准 SFT**;③ \(-\log p\) 训全 token 的退化主要归因于 bottom 10% 分位。
  - **训练集平均似然作能力代理**——§5.3 Fig.6 验证:MS 端 \(-p\) 训后平均似然更高、MW 端 \(-\log p\) 更高,与下游准确率趋势一致（似然估计 \(\frac1N\sum_i\sum_j p_\theta(\tilde y_{i,j})\)）。
- **关键机制/公式（直觉）**：\(W_f\) 决定了梯度沿 token 概率的分布形状（Fig.3 画了各条曲线的峰值位置）。三个代表性损失对照看:
  - NLL:\(W_f\to(1-p)\)，强压低概率 token——逼模型从所有 token、尤其是出错处广泛地学。
  - \(-p\):\(W_f=p(1-p)\)，低概率处信号迅速衰减——只精炼那些高置信的 token。
  - \(-\log(1-p)\):\(W_f=p\)，完全偏向高概率 token。
  - **直觉**:base 强时,低概率 token 大多是噪声（§5.1 实证原文 "low-probability tokens act primarily as noise to the strong model"）,应当 de-emphasize;base 弱时,低概率 token 大多正是该学的出错处,应当强调。所以损失的凸凹,本质上是"先验被尊重程度"的代理。
- **实验与证据（§4-5）**：规模很大——8 个 backbone（LLaMA-3.1-3B/8B、3.2-3B、DeepSeekMath-7B、Qwen2.5-Math-1.5B/7B、Qwen2.5-1.5B~32B、Qwen2.5-Coder-7B）× 27 个 benchmark / 7 个域。三个锚点任务分别代表谱的三段:**MS=NuminaMath（数学）、MI=m23k（医疗）、MW=Reasoning Gym figfont**。
  - **MS（Table 1）**：\(-p\) 与阈值化 \(-\log p\)（\(p\ge0.2\)）一致超 NLL。Qwen2.5-Math-1.5B 17.00→32.75（**+15.75**）;DeepSeekMath-7B 11.10→19.11;Qwen2.5-Math-7B 22.67→36.51。
  - **MI（Table 2）**：\(-p\) 与 \(-\log p\) 几乎相同（差异在统计波动内,如 LLaMA-3.1-8B 47.23 vs 45.89、Qwen2.5-Math-7B 33.56 vs 33.83）。**MI 区无单一最优,改善应转向数据/监督质量**。
  - **MW（Table 3）**：\(-\log p\) 一致大幅超 \(-p\),\(-p\) 的 Exact Match 常为 0（如 Qwen2.5-7B \(-\log p\) EM 35.20 vs \(-p\) 0.00;Jaro-Winkler 82.48 vs 10.15）。
  - **通用指令微调（§4.3 Fig.4）**：固定数据/协议只变规模 3B→14B,AlpacaEval2 上偏好从 NLL 平滑转向 \(-p\)（**单设定内复现连续谱**）。
  - **凸凹扫描（§5.2 Fig.6）**：\(\alpha\) 从 0.1→10,MS 端准确率随 \(\alpha\)↑ 改善、峰在 \(\alpha\approx1\) 后稳;MW 端在 \(\alpha=0.1\) 最优、过凸凹边界后急剧恶化——**凸凹对性能的影响在两端方向相反**。
  - 编码（Appen C.3:\(-p\) 胜）/ 低资源多语（Appen C.4:\(-\log p\) 胜）/ 固定数学规模扫描 3B→32B（\(-p\) 优势随规模增大）分别复现 MS/MW 端。
- **假设与失效边界**：【原文】MI 区"无单一最优"——改善应转向数据/监督质量而非损失（§4.2）;**能力位置靠训练后用训练集似然量化**（§3,非先验可操作）;MW 进步靠更强/更针对性监督或其他注入知识方式（§4.2）。【推断】最大说服力来自实证规模（8×27）,但"训练集平均似然"需事后估计,落地时仍需试;**MI（最常见的中等先验任务）反而给不出明确建议,实践指导有限**。
- **祛魅总结**：
  - 【推断】真贡献有三块:一是 **"没有单一最优的 SFT 损失,该用哪种取决于 base 能力"这一统一认识论框架**;二是 \(W_f\) 凸凹这把刀,把散落的损失归到一起;三是大规模实证撑起的连续谱（包含单一设定内做规模扫描的内部复现）。而且**作者很诚实,没把它包装成新算法**——§positioning 明说自己讲的是"when and why",不是某个 best loss。
  - 真正要祛魅的是它的落地价值:**它不是一个新损失,也不给一个先验可操作的选择准则**——价值在"解释",而非"可直接拿来用的方法"。
  - 与 OPD 的关系是**间接**的:它"prior-leaning、下调低概率 token"的思路,和 token 加权 / 重要性采样相通;但本文是**固定 ground-truth 的 SFT 设定**,而 OPD 是 student-on-policy + teacher 分布监督,不能直接照搬——两者的"低概率 token"语义并不相同。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=ground-truth 序列的 token 概率（经 \(f\) 变换,不同 \(f\) 改变各概率段 token 的梯度权重 \(W_f\)）｜**改什么**=参数 + **SFT 训练目标 \(f\) 本身**（\(-\log p\) / \(-p\) / \(-p^{10}\) / 阈值化 / hard-threshold）｜**何时改**=离线 SFT（训练目标层面,非 per-step 动态）｜**免梯度?**=否（标准 SFT 梯度下降,只改损失函数形状）｜**记忆-技能生命周期**=不适用｜**防遗忘机制**=不适用（间接:prior-leaning 少改 base 已会的低概率 token,可视为"少破坏已有先验",但非持续学习防遗忘）
- ⑦ 开源代码+框架/harness：https://github.com/GaotangLi/Beyond-Log-Likelihood 。框架=**VeRL**（旧 analysis 记 `main_verl`,目录含 `scripts/{training,evaluation,ablation,one_click}`、`data`、`evaluations`,覆盖 27 benchmark）。【待核】框架与目录细节复用旧 analysis 对仓库的核查,本次基于 PDF 重写未重新进仓核对。复现=固定数据/评测协议、仅改 SFT 目标 \(f\) 即可;hard-thresholding 变体（Eq.4）用于消融。
- 💰 资源/成本与可扩展性：【原文未明确给单次训练成本】实验规模大（8 backbone×27 benchmark×7 域 + 规模扫描 3B→32B）。方法本身**零额外成本**（只换损失函数,不增采样/不增阶段/不增参数）。
- 🎯 对"探索-巩固"对标：**可借组件（机制层启发）**。判定:本文与 OPD/TSRD **不是同一个设定**（它是固定 ground-truth 的 SFT）,但它的核心洞察对"巩固"环节有直接启发——"**base 先验强时,该 de-emphasize 低概率 token、少破坏模型已经会的东西**"。换言之,巩固有效行为时,应当保护 base 已掌握的稳妥路径（高概率段）,而把学习信号集中到少数关键 token 上（这与"只训 top10% token 就能超标准 SFT"的发现呼应）。
  - **可借组件**:① \(W_f(p)=-f'(p)p(1-p)\) 这把"该强调哪类 token"的分析刀,可移植到 TSRD 去判断"path-recovery 时该改哪些 token 的 logits";② hard-thresholding（Eq.4）这种"按概率分位隔离 token 贡献"的消融方法。
  - **缺口/差异**:本文没有 on-policy、没有 teacher 分布、没有 step/path 信用分配、也没有 MTP;它的结论建立在固定监督序列上,而 TSRD 是 student 自采轨迹 + teacher 脚手架,迁移时需谨慎。
- 🔭 开放问题/未来方向：【原文】MI 区改善应转向数据/监督质量（§4.2）;MW 进步靠更强/更有针对性监督或其他注入知识方式（§4.2）。【推断】先验可操作的目标选择准则（免事后估计似然）;把连续谱思想推广到 on-policy/OPD 的 token 加权;MI 区损失之外的杠杆;把 RL/蒸馏的 token 级信用分配统一在 \(W_f\) 框架下。

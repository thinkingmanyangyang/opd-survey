beyond_loglik | Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum | UIUC（共一 Gaotang Li、Ruizhong Qiu、Xiusi Chen;Heng Ji、Hanghang Tong）| 2026-05（arXiv:2510.00526v3;ICML 2026，PMLR 306）| 主题线 L2 统一SFT-RL/SFT损失设计·相关性 中

**原始论文**：https://arxiv.org/abs/2510.00526

## 一眼看懂
- 🟦 TL;DR：SFT 默认用 NLL（\(-\log p\)），大家把"SFT 泛化差"归咎于模仿范式。本文说不对——病根是 NLL 这个**默认目标**:NLL 只在"从零训练小分类"才最优（Cox 1958 等),后训练时 base 已有先验、监督序列长且含噪,逼模型逐 token 复刻冗长 CoT 反而伤泛化。把 NLL 推广成参数族 \(f^\alpha(p)=\frac{1-p^\alpha}{\alpha}\)，并提出统一刻画——**模型能力连续谱（model-capability continuum）**:base 先验强（数学）时下调低概率 token 的 prior-leaning 目标（如 \(-p\)）持续胜 NLL（Qwen2.5-Math-1.5B **+15.75**）;先验弱（figfont 谜题）时 NLL 主导;中间（医疗）两者难分。
- 最巧的一步：**梯度权重 \(W_f(p)=-f'(p)\,p\,(1-p)\) 的凸凹分析（Lemma 3.1 + Prop 3.2）**。抽掉它就只剩"换损失有时更好"的经验观察;正是 \(W_f\) 把"凸目标(\(-\log p\))峰值在 \([0,\tfrac12]\) 强调低概率 token = prior-averse / 凹目标(\(-p\))峰值在 \([\tfrac12,1]\) 强调高概率 token = prior-leaning"讲清楚,才把一堆散落的损失（\(-p\)、focal、entropic matching、均匀重加权）统一进同一谱系并预测它们何时该用。

## 为什么做
- 研究背景：SFT 是 LLM 后训练标准做法（Zhang 2026b、Chung 2024）,但**泛化常受限**,普遍归因于"模仿学习范式本身"（§1,引 Ouyang 2022、Chu 2025）。
- 解决的具体痛点：把锅甩给"模仿范式"**忽略了默认目标 NLL 本身**。NLL 在从零训练分类上经典最优,但后训练范式不同:base 已编码任务先验且通常 **well-calibrated**（Zhu 2023、Xie 2024）,监督是上千 token 的长 CoT 且可能含噪——**要求 base 逐 token 复刻会伤泛化**。而 RL 启发的各种 SFT 改进各自在某些域有效,但**缺乏"何时用哪种目标"的统一刻画**。
- 相关工作 & 各自不足（§2）：
  - **从 RL 视角改 SFT**——把 SFT/DPO 看隐式 reward learning（小 lr + 散度目标,Wang 2026）、引入 importance sampling（Qin & Springenberg 2025）、PPO-style clip 约束 drift（Zhu 2026）、**均匀重加权梯度系数（Wu 2026,本文证明等价 \(-p\)）**——各自有效但**不知何时失效**。
  - **其他 SFT 损失**——MSE / focal loss（Lin 2017）/ Huber（Huber 1992）/ entropic distribution matching（Li 2025,加正则项）——本文框架可统一解释为 prior-leaning。
  - **定位（§positioning）**——提供**首个 capability-based 的 SFT 目标统一刻画**;明确**不主张单一万能损失**（"not advocating a single one-size-fits-all loss"）,而是"establish a principled account of **when and why** objectives trade advantages"。
- 动机链：现状（SFT 泛化差,归咎模仿范式,默认用 NLL）→ 缺陷（NLL 最优性假设在后训练被违反 + 无"何时用何损失"统一理论）→ 提 \(f^\alpha\) 族 + 能力连续谱。为什么不直接推一个"更好的损失"？因为实证发现损失好坏会随 base 能力**反转**（MS 端 \(-p\) 赢、MW 端 NLL 赢）,单一损失不可能万能。
- 与最近邻工作的Δ：vs **Wu 2026（均匀重加权,等价 \(-p\)）**——本文证明这种 prior-leaning 更新**只在 MS 端有效、MW 端会害**,把单点结论纳入连续谱;vs 各种 SFT 损失工作——提供 \(W_f\) 凸凹这把统一刀。关键点:损失选择不是绝对优劣,而是与 base 先验强度的**匹配**。

## 怎么做 + 靠不靠谱

> 注:这是**分析/认识论框架**而非新算法。设定:后训练 base \(p_\theta\)（已 well-calibrated、含任务先验）;SFT 数据 \(\mathcal T\) 为 \((x,\tilde y)\) 对,\(\tilde y=(y_1,\dots,y_N)\);decoding 步 \(t\) logits \(z_t\)，\(p_t=\mathrm{softmax}(z_t)\)。

**【统一框架（§3）——可复现的目标族】**
1. **标准 SFT = NLL/交叉熵**：\(\mathcal L_{\log(p)}(\theta)=\mathbb E_{(x,\tilde y)\sim\mathcal T}\big[-\sum_{t=1}^N\log p_\theta(y_t\mid y_{<t},x)\big]\)。
2. **通用概率目标族**：对任意可微非增 \(f:[0,1]\to\mathbb R\)，
   \[ \mathcal L_{f(p)}(\theta)=\mathbb E_{(x,\tilde y)\sim\mathcal T}\Big[\sum_{t=1}^N f\big(p_\theta(y_t\mid y_{<t},x)\big)\Big]. \]
   一个有用实例:\(f^\alpha(p)=\dfrac{1-p^\alpha}{\alpha}\)。**\(\alpha\to0\) 退化 NLL**（\(f^\alpha\to-\log p\)）;**\(\alpha=1\) 即 \(-p\)（plain-p，=最大化期望平均预测准确率,因 \(1-p\) 对应 0-1 loss 期望）**;**\(\alpha\ge1\) 凹 / \(0\le\alpha\le1\) 凸**。
3. **梯度形状（Lemma 3.1,理论核心）**：对 logits 的梯度为
   \[ \frac{\partial \mathcal L_f}{\partial z_{t,i}}=s_f(p_{t,y})\,(\delta_{i,y}-p_{t,i}),\qquad s_f(p):=-f'(p)\,p\ \ge0, \]
   其中 \(\delta_{i,y}=\mathbb 1\{i=y\}\)。对正确类 \(i=y\)：
   \[ \frac{\partial \mathcal L_f}{\partial z_{t,y}}=s_f(p_{t,y})(1-p_{t,y})=W_f(p_{t,y}),\qquad \boxed{W_f(p):=-f'(p)\,p\,(1-p)}. \]
   \(W_f(p)\) **决定每个 token 按其当前预测概率贡献多少正确类梯度**。
4. **凸凹归类（Prop 3.2）**：设 \(f\in C^2[0,1]\)，\(f'(p)<0\)。**\(f\) 凹 ⇒ \(W_f\) 的极大点落在 \([\tfrac12,1]\)（强调高概率 token = prior-leaning）;\(f\) 凸 ⇒ 极大点落在 \([0,\tfrac12]\)（强调低概率 token = prior-averse）**。对参数族 \(W_f(p)=p^\alpha(1-p)\):\(\alpha\to0\)（NLL）得 \(W_f\to(1-p)\) 强压低概率 token;\(\alpha\ge1\)（\(-p\)）低概率信号迅速衰减;特例 \(f(p)=-\log(1-p)\) 得 \(W_f=p\) 完全偏高概率。
5. **prior-leaning vs prior-averse（Def 3.3）**：按 \(W_f\) 把梯度质量集中在阈值 \(\tau\) 之上（中-高概率 token,leverage 先验精炼已可信预测）还是之下（低概率 token,逼模型从不可信预测学习）来分类。边界 \(\tau\) 非唯一、随任务变,但 \(-\log p\) vs \(-p\) 有明确对比。
6. **hard-thresholding 变体（Eq.4,关键消融工具）**：只在概率区间 \(I\subseteq[0,1]\) 内更新——
   \[ \mathcal L_{HT(I),f(p)}(\theta)=\mathbb E_{(x,\tilde y)\sim\mathcal T}\Big[\sum_t f\big(p(y_t\mid\cdot)\big)\,\mathbb 1\{p(y_t\mid\cdot)\in I\}\Big]. \]
   用于隔离各概率段 token 的贡献（分位阈值由**训练前**的 base 模型预测概率算出）。
7. **能力连续谱定位（§3 末）**：两视角——**数据侧**（语料相关 token 占比:LLaMA-3 报告 ~25% 数学 token → 数学 MS;figfont 完全不在语料 → MW;医疗部分覆盖 → MI）+ **模型侧**（用**训练集平均预测概率**作先验强度代理:数学 0.76–0.81、医疗 ~0.50、figfont ~0.01,类比 lm-eval 用 log-likelihood 评 base）。MS 端用 prior-leaning、MW 端用 prior-averse(NLL)、MI 端两者皆可。
- **逐组件必要性**：
  - **\(W_f\) 凸凹分析**——理论核心,没它无法预测谱位置。
  - **hard-thresholding 变体**——§5.1 Fig.5 用它隔离各概率段:① **低概率 token 一致有害**（所有目标在低分位都掉）;② **只训 top10% token 时所有目标都超标准 SFT**;③ \(-\log p\) 训全 token 的退化主要归因于 bottom 10% 分位。
  - **训练集平均似然作能力代理**——§5.3 Fig.6 验证:MS 端 \(-p\) 训后平均似然更高、MW 端 \(-\log p\) 更高,与下游准确率趋势一致（似然估计 \(\frac1N\sum_i\sum_j p_\theta(\tilde y_{i,j})\)）。
- **关键机制/公式（直觉）**：\(W_f\) 决定梯度按 token 概率的分布形状（Fig.3 各曲线峰值位置）。NLL 的 \(W_f\to(1-p)\) 强压低概率 token（逼模型从所有 token 尤其错处广泛学）;\(-p\) 的 \(W_f=p(1-p)\) 低概率信号迅速衰减（只精炼高置信 token）;\(-\log(1-p)\) 的 \(W_f=p\) 完全偏高概率。**直觉**:base 强时低概率 token 多是噪声（§5.1 实证"low-probability tokens act primarily as noise to the strong model"）,该 de-emphasize;base 弱时低概率 token 多是该学的错处,该强调。凸凹是"先验被尊重程度"的代理。
- **实验与证据（§4-5）**：8 backbone（LLaMA-3.1-3B/8B、3.2-3B、DeepSeekMath-7B、Qwen2.5-Math-1.5B/7B、Qwen2.5-1.5B~32B、Qwen2.5-Coder-7B）× 27 benchmark / 7 域。锚点:**MS=NuminaMath(数学)、MI=m23k(医疗)、MW=Reasoning Gym figfont**。
  - **MS（Table 1）**：\(-p\) 与阈值化 \(-\log p\)（\(p\ge0.2\)）一致超 NLL。Qwen2.5-Math-1.5B 17.00→32.75（**+15.75**）;DeepSeekMath-7B 11.10→19.11;Qwen2.5-Math-7B 22.67→36.51。
  - **MI（Table 2）**：\(-p\) 与 \(-\log p\) 几乎相同（差异在统计波动内,如 LLaMA-3.1-8B 47.23 vs 45.89、Qwen2.5-Math-7B 33.56 vs 33.83）。**MI 区无单一最优,改善应转向数据/监督质量**。
  - **MW（Table 3）**：\(-\log p\) 一致大幅超 \(-p\),\(-p\) 的 Exact Match 常为 0（如 Qwen2.5-7B \(-\log p\) EM 35.20 vs \(-p\) 0.00;Jaro-Winkler 82.48 vs 10.15）。
  - **通用指令微调（§4.3 Fig.4）**：固定数据/协议只变规模 3B→14B,AlpacaEval2 上偏好从 NLL 平滑转向 \(-p\)（**单设定内复现连续谱**）。
  - **凸凹扫描（§5.2 Fig.6）**：\(\alpha\) 从 0.1→10,MS 端准确率随 \(\alpha\)↑ 改善、峰在 \(\alpha\approx1\) 后稳;MW 端在 \(\alpha=0.1\) 最优、过凸凹边界后急剧恶化——**凸凹对性能的影响在两端方向相反**。
  - 编码（Appen C.3:\(-p\) 胜）/ 低资源多语（Appen C.4:\(-\log p\) 胜）/ 固定数学规模扫描 3B→32B（\(-p\) 优势随规模增大）分别复现 MS/MW 端。
- **假设与失效边界**：【原文】MI 区"无单一最优"——改善应转向数据/监督质量而非损失（§4.2）;**能力位置靠训练后用训练集似然量化**（§3,非先验可操作）;MW 进步靠更强/更针对性监督或其他注入知识方式（§4.2）。【推断】最大说服力来自实证规模（8×27）,但"训练集平均似然"需事后估计,落地时仍需试;**MI（最常见的中等先验任务）反而给不出明确建议,实践指导有限**。
- **祛魅总结**：【推断】真贡献是 **"无单一最优 SFT 损失,取决于 base 能力"这一统一认识论框架** + \(W_f\) 凸凹这把刀把散落损失归一 + 大规模实证的连续谱（含单设定规模扫描的内部复现）。**作者很诚实地不包装成新算法**（§positioning 明说"when and why"而非 best loss）。要祛魅的是其落地价值:**它不是新损失、不给先验可操作的选择准则**——价值在解释而非可直接拿来用的方法;与 OPD 的关系是**间接**:其"prior-leaning 下调低概率 token"与 token 加权/重要性采样相通,但本文是**固定 ground-truth SFT 设定**,OPD 是 student-on-policy + teacher 分布监督,不能直接套（OPD 的"低概率 token"语义不同于固定 ground-truth 的低概率 token）。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=ground-truth 序列的 token 概率（经 \(f\) 变换,不同 \(f\) 改变各概率段 token 的梯度权重 \(W_f\)）｜**改什么**=参数 + **SFT 训练目标 \(f\) 本身**（\(-\log p\) / \(-p\) / \(-p^{10}\) / 阈值化 / hard-threshold）｜**何时改**=离线 SFT（训练目标层面,非 per-step 动态）｜**免梯度?**=否（标准 SFT 梯度下降,只改损失函数形状）｜**记忆-技能生命周期**=不适用｜**防遗忘机制**=不适用（间接:prior-leaning 少改 base 已会的低概率 token,可视为"少破坏已有先验",但非持续学习防遗忘）
- ⑦ 开源代码+框架/harness：https://github.com/GaotangLi/Beyond-Log-Likelihood 。框架=**VeRL**（旧 analysis 记 `main_verl`,目录含 `scripts/{training,evaluation,ablation,one_click}`、`data`、`evaluations`,覆盖 27 benchmark）。【待核】框架与目录细节复用旧 analysis 对仓库的核查,本次基于 PDF 重写未重新进仓核对。复现=固定数据/评测协议、仅改 SFT 目标 \(f\) 即可;hard-thresholding 变体（Eq.4）用于消融。
- 💰 资源/成本与可扩展性：【原文未明确给单次训练成本】实验规模大（8 backbone×27 benchmark×7 域 + 规模扫描 3B→32B）。方法本身**零额外成本**（只换损失函数,不增采样/不增阶段/不增参数）。
- 🎯 对"探索-巩固"对标：**可借组件（机制层启发）**。判定:本文与 OPD/TSRD **非同设定**（固定 ground-truth SFT）,但其核心洞察——"**base 先验强时该 de-emphasize 低概率 token、少破坏已会的**"——对"巩固"环节有直接启发:巩固有效行为时应保护 base 已掌握的稳妥路径（高概率段）,而把学习信号集中在少数关键 token（与"只训 top10% token 超标准 SFT"呼应）。**可借组件**:① \(W_f(p)=-f'(p)p(1-p)\) 这把"该强调哪类 token"的分析刀,可移植到 TSRD 判断"path-recovery 时该改哪些 token 的 logits";② hard-thresholding（Eq.4）按概率分位隔离 token 贡献的消融法。**缺口/差异**:无 on-policy、无 teacher 分布、无 step/path 信用分配、无 MTP;其结论建立在固定监督序列上,TSRD 是 student 自采轨迹 + teacher 脚手架,需谨慎迁移。
- 🔭 开放问题/未来方向：【原文】MI 区改善应转向数据/监督质量（§4.2）;MW 进步靠更强/更有针对性监督或其他注入知识方式（§4.2）。【推断】先验可操作的目标选择准则（免事后估计似然）;把连续谱思想推广到 on-policy/OPD 的 token 加权;MI 区损失之外的杠杆;把 RL/蒸馏的 token 级信用分配统一在 \(W_f\) 框架下。

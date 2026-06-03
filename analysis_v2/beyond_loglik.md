beyond_loglik | Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum | UIUC（共一 Gaotang Li、Ruizhong Qiu、Xiusi Chen；Heng Ji、Hanghang Tong）| 2026-05（arXiv:2510.00526v3；ICML 2026，PMLR 306）| 主题线 L2 统一SFT-RL/SFT损失设计·相关性 中

**原始论文**：https://arxiv.org/abs/2510.00526

## 一眼看懂
- 🟦 TL;DR：SFT 默认用 NLL（−log p），大家把"SFT 泛化差"归咎于模仿范式。本文说不对——病根是 NLL 这个**默认目标**：NLL 只在"从零训练小分类"才最优，后训练时 base 已有先验、监督序列长且含噪，逼模型逐 token 复刻冗长 CoT 反而伤泛化。把 NLL 推广成参数族 f_α(p)=(1−p^α)/α，并提出统一刻画——**模型能力连续谱**：base 先验强（数学）时下调低概率 token 的 prior-leaning 目标（如 −p）持续胜 NLL（+15.75）；先验弱（figfont 谜题）时 NLL 主导；中间（医疗）两者难分。
- 最巧的一步：**梯度权重 W_f(p)=−f'(p)·p·(1−p) 的凸凹分析（Lemma 3.1 + Prop 3.2）**。抽掉它就只剩"换损失有时更好"的经验观察；正是 W_f 把"凸目标(−log p)峰值在[0,0.5]强调低概率 token=prior-averse / 凹目标(−p)峰值在[0.5,1]强调高概率 token=prior-leaning"讲清楚，才把一堆散落的损失（−p、focal、entropic matching、均匀重加权）统一进同一谱系并预测它们何时该用。

## 为什么做
- 研究背景：SFT 是 LLM 后训练标准做法，但**泛化常受限**，普遍归因于"模仿学习范式本身"（§1，引 Chu 2025）。
- 解决的具体痛点：把锅甩给"模仿范式"**忽略了默认目标 NLL 本身**。NLL 在从零训练分类上经典最优（Cox 1958 等），但后训练范式不同：base 已编码任务先验且通常 well-calibrated，监督是上千 token 的长 CoT 且可能含噪——**要求 base 逐 token 复刻会伤泛化**。而 RL 启发的各种 SFT 改进各自在某些域有效，但**缺乏"何时用哪种目标"的统一刻画**。
- 相关工作 & 各自不足：① 从 RL 视角改 SFT——把 SFT/DPO 看隐式 reward learning（小 lr+散度目标）、引入 importance sampling、PPO-style clip 约束 drift、**均匀重加权梯度系数（Wu 2026，等价本文 −p）**——各自有效但**不知何时失效**；② 其他 SFT 损失（MSE/focal/Huber/entropic distribution matching）——本文框架可统一解释为 prior-leaning。
- 动机链：现状（SFT 泛化差，归咎模仿范式，默认用 NLL）→ 缺陷（NLL 的最优性假设在后训练被违反，且无"何时用何损失"的统一理论）→ 所以提 f_α 族 + 能力连续谱。为什么不直接推一个"更好的损失"？作者明确**不主张单一万能损失**（§1），而是建立"何时用何损失"的认识论框架——因为实证发现损失好坏会随 base 能力**反转**（MS 端 −p 赢、MW 端 NLL 赢）。
- 与最近邻工作的Δ：vs Wu 2026（均匀重加权，等价 −p）——本文证明这种 prior-leaning 更新**只在 MS 端有效、MW 端会害**，把单点结论纳入连续谱；vs 各种 SFT 损失工作——提供**首个 capability-based 的统一刻画**（W_f 凸凹）。关键点：损失选择不是绝对优劣，而是与 base 先验强度的匹配。

## 怎么做 + 靠不靠谱
- 方法流水线（这是**分析/认识论框架**而非新算法）：① 统一框架——把 SFT 目标写成 L_f=E[f(p_θ(y|x))]，f 可微非增（Eq.2）；f_α(p)=(1−p^α)/α，α→0 退化 NLL、α=1 即 −p（=最大化期望平均预测准确率），α≥1 凹/0≤α≤1 凸（Eq.3）；② 凸凹归类——W_f(p)=p^α(1−p)，Prop 3.2 证凹目标 W_f 峰在[0.5,1]（prior-leaning）、凸目标峰在[0,0.5]（prior-averse），据此定义 prior-leaning/prior-averse（Def 3.3）；③ 能力连续谱三段——MS（先验强，如数学）用 prior-leaning、MW（无先验，如 figfont）用 NLL、MI（医疗）两者难分；④ 能力位置量化——用"训练集平均预测概率"（数学 0.76–0.81、医疗~0.50、figfont~0.01）；⑤ 理论——梯度流下给充分条件：MS 端 −p 损失下降更大、MW 端 NLL 更大。
- 逐组件必要性：**W_f 凸凹分析**（理论核心，没它无法预测谱位置）；**hard-thresholding 变体 L_HT(I)**（Eq.4，只在概率区间 I 内更新）——是关键**消融工具**，§5.1 Fig.5 用它隔离各概率段贡献，发现"低概率 token 一致有害 + 只训 top10% token 所有目标都超标准 SFT"；**训练集平均似然作能力代理**——§5.3 验证与定性预期吻合且随谱反转。整篇是"理论预测→多设定实证验证"，消融充分（27 benchmark）。
- 关键机制/公式（直觉）：W_f(p)=−f'(p)·p·(1−p) 决定每个 token 贡献多少梯度。NLL 的 W_f→(1−p) 强压低概率 token（逼模型从所有 token 尤其错处广泛学）；−p 的 W_f=p(1−p) 低概率信号迅速衰减（只精炼高置信 token）；−log(1−p) 的 W_f=p 完全偏高概率。直觉：base 强时低概率 token 多是噪声（§5.1 实证"低概率 token 主要作为噪声"），该 de-emphasize；base 弱时低概率 token 多是该学的错处，该强调。
- 实验与证据：8 backbone（LLaMA-3.1-3B/8B、3.2-3B、DeepSeekMath-7B、Qwen2.5-Math-1.5B/7B、Qwen2.5-1.5B~32B、Qwen2.5-Coder-7B）×27 benchmark/7 域。锚点：MS=NuminaMath(数学)、MI=m23k(医疗)、MW=Reasoning Gym figfont。**关键数字**：MS（Table 1）−p 与阈值化 −log p 一致超 NLL，Qwen2.5-Math-1.5B 17.00→32.75（**+15.75**）；MI（Table 2）−p 与 −log p 几乎相同（差异在统计波动内，如 LLaMA-3.1-8B 47.23 vs 45.89）；MW（Table 3）−log p 一致大幅超 −p，−p 的 Exact Match 常为 0；通用指令微调（Fig.4）固定数据只变规模 3B→14B，偏好从 NLL 平滑转向 −p（单设定内复现连续谱）；编码/低资源多语分别复现 MS/MW 端。
- 假设与失效边界：【原文】MI 区"无单一最优"——改善应转向数据/监督质量而非损失（§4.2）；能力位置靠**训练后用训练集似然量化**（§3，非先验可操作）。【推断】最大说服力来自实证规模（8×27），但"训练集平均似然"需事后估计，落地时仍需试；MI（最常见的中等先验任务）反而给不出明确建议，实践指导有限。
- 祛魅总结：【推断】真贡献是 **"无单一最优 SFT 损失，取决于 base 能力"这一统一认识论框架** + W_f 凸凹这把刀把散落损失归一 + 大规模实证的连续谱（含单设定规模扫描的内部复现）。**作者很诚实地不包装成新算法**（§positioning 明说"establish a principled account of when and why"而非 best loss）。要祛魅的是其落地价值：**它不是新损失、不给先验可操作的选择准则**——价值在解释而非可直接拿来用的方法；与 OPD 的关系是**间接**：其"prior-leaning 下调低概率 token"与 token 加权/重要性采样相通，但本文是固定 ground-truth SFT 设定，OPD 是 student-on-policy + teacher 分布监督，不能直接套。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=ground-truth 序列的 token 概率（经 f 变换，不同 f 改变各概率段 token 的梯度权重）｜**改什么**=参数 + **SFT 训练目标 f 本身**（−log p / −p / 阈值化 / hard-threshold）｜**何时改**=离线 SFT（训练目标层面，非 per-step 动态）｜**免梯度?**=否（标准 SFT 梯度下降，只改损失函数形状）｜**记忆-技能生命周期**=不适用｜**防遗忘机制**=不适用（间接：prior-leaning 少改 base 已会的低概率 token，可视为"少破坏已有先验"，但非持续学习防遗忘）
- ⑦ 开源代码+框架/harness：https://github.com/GaotangLi/Beyond-Log-Likelihood 。框架=**VeRL**（旧 analysis 记 `main_verl`，目录含 `scripts/{training,evaluation,ablation,one_click}`、`data`、`evaluations`，覆盖 27 benchmark）。【待核】框架与目录细节复用旧 analysis 对仓库的核查，本次未重新进仓核对。复现=固定数据/评测协议、仅改 SFT 目标即可；hard-thresholding 变体用于消融。
- 💰 资源/成本与可扩展性：【原文未明确给单次训练成本】实验规模大（8 backbone×27 benchmark×7 域 + 规模扫描 3B→32B）。方法本身**零额外成本**（只换损失函数，不增采样/不增阶段）。
- 🎯 对"探索-巩固"对标：**可借组件（机制层启发）**。判定：本文与 OPD/TSRD **非同设定**（固定 ground-truth SFT），但其核心洞察——"**base 先验强时该 de-emphasize 低概率 token、少破坏已会的**"——对"巩固"环节有直接启发：巩固有效行为时应保护 base 已掌握的稳妥路径（高概率段），而把学习信号集中在少数关键 token。**可借组件**：① W_f 凸凹这把"该强调哪类 token"的分析刀，可移植到 TSRD 判断"path-recovery 时该改哪些 token 的 logits"；② hard-thresholding 按概率分位隔离 token 贡献的消融法。**缺口/差异**：无 on-policy、无 teacher 分布、无 step/path 信用分配、无 MTP；其结论建立在固定监督序列上，TSRD 是 student 自采轨迹 + teacher 脚手架，需谨慎迁移（OPD 的"低概率 token"语义不同于固定 ground-truth 的低概率 token）。
- 🔭 开放问题/未来方向：【原文】MI 区改善应转向数据/监督质量（§4.2）；MW 进步靠更强/更有针对性监督或其他注入知识方式（§4.2）。【推断】先验可操作的目标选择准则（免事后估计似然）；把连续谱思想推广到 on-policy/OPD 的 token 加权；MI 区的损失之外的杠杆；与 RL/蒸馏的 token 级信用分配统一在 W_f 框架下。

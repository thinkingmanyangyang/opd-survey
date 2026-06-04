rl_plus | RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization | 北京大学 + 阿里通义实验室 + University of Alberta（Yihong Dong、Xue Jiang 一作@通义实习；Zhi Jin、Ge Li、Yongbin Li 等） | arXiv 2508.00222（v5 2026-04-15；标注 Preprint July 2025）· cs.AI | 主题线 L2(统一SFT-RL)+L3(RLVR/GRPO)·相关性 较高

**原始论文**:https://arxiv.org/abs/2508.00222

## 一眼看懂
> 一句话导读:纯 RLVR 只会榨干基座已有的能力、却把可解题集越练越窄;RL-PLUS 用一套"稳定吸收外部示范 + 专门奖励低概率正确路径"的混合策略,让模型真正学到基座之外的新解法、突破 pass@k 天花板。

- 🟦 TL;DR:纯 on-policy RLVR(如 GRPO)会"收窄"基座本来能解的问题集——表现为 pass@1 上升、但 pass@128 反而低于基座。作者称之为 capability boundary collapse(能力边界坍缩);成因是巨大的动作空间 + 稀疏奖励,逼着模型只做"向内利用(exploit)"而不敢"向外探索(explore)"。RL-PLUS 的对策是用混合策略,把外部数据(SFT 示范)稳定地吸进 RL,三招:
  1. **Multiple Importance Sampling(MIS,多重重要性采样)**:把外部样本看作"旧策略 \(\pi_{\theta_{\text{old}}}\) 与未知外部策略 \(\pi_\omega\) 的混合"所生成,重要性比率的分母改用 \(\pi_\omega+\pi_{\theta_{\text{old}}}\)(其中未知的 \(\pi_\omega\) 用贝叶斯估计为 \(\pi_\omega\approx\frac12\pi_{\theta_{\text{old}}}+\frac12\) 均匀分布)。这里 \(\pi_{\theta_{\text{old}}}\) 充当"方差护栏",让比值始终有界,从而解决分布失配;
  2. **Exploration-Based Advantage(基于探索的优势,focal 式)**:给"正确、但当前策略给的概率却很低"的 token 放大优势(权重为 \((1-\mathrm{detach}(\pi_\theta))^\gamma\)),逼模型去关注那些低概率的正确路径;
  3. **去掉 clip**:因为 clip 会压制"高信息、低概率事件"的梯度,而这恰恰是模型最该学的新知识。
- 结果:6 个数学基准 SOTA、6 个 OOD 任务领先,且 pass@k 曲线全程高于基座(说明是真的突破了天花板)。【Abstract,§3,图1】
- 最巧的一步:**MIS 的分母构造(式4)**。看消融就知道它多关键:抽掉它(Table4 的 "−MIS"),平均分从 53.4 暴跌到 45.5,等于退回普通 GRPO 水平。它是让"从外部 off-policy 数据稳定学习"变得可行的关键——没有它,要么用 on-policy 代理(会有系统偏差,Lemma A.5),要么用纯 off-policy 权重(会支撑失配 A.6 + 高方差 A.7),两条路都会让外部数据的吸收失败。其中"\(\pi_{\theta_{\text{old}}}\) 放在分母里,即使 \(\pi_\omega\) 很差、比值也仍有界"(Theorem 3.1)正是整个方法稳定性的支柱。【§3.1,Table4】

## 为什么做
> 一句话导读:RLVR 只会"思而不学"(榨基座、不吸收新知识),顺序 SFT→RL 又"学而不思"(易灾难性遗忘);本文想要一个既能稳定从外部数据"学"、又能显式激励模型"思"出低概率正确路径的混合策略,难点在分布失配和提取外部数据价值两道关。

- 研究背景:RLVR(o1/DeepSeek-R1/Kimi)靠可验证奖励驱动 LLM 延展 CoT、自发反思与探索,被视为通向更强 AI 的一条路径。【§1,L34-42】
- 解决的具体痛点:多项工作(Havrilla 2024、Shao 2024、Yue 2025a)指出,现行 RLVR 并不能让模型获得**新**的推理能力、只是复用基座已有的模式——表现为 pass@1 升、pass@128 反低于基座(图1a),即可解问题集被收窄(**capability boundary collapse**)。根因是:解空间巨大 + 奖励稀疏,长链上单步出错就清零整条轨迹的奖励,于是模型被迫"向内利用"而非"向外探索";并伴随 **entropy collapse**(熵坍缩,即过度确定、丧失探索)。【§1,§2.2,L43-106】
- 相关工作的来龙去脉与精确短板(据 §2.2):
  - **on-policy RLVR**:GRPO 是基底(用组内归一化的 reward 估 advantage、免掉 value model);PRIME-Zero(Cui)用隐式过程奖励,Oat-Zero(Liu)简化 advantage 计算。**短板**=受限于基座已有知识,且两病并发——Capability Boundary Collapse(在大 k 处 pass@k 被基座反超,Yue 2025a)+ Entropy Collapse(Cui 2025b)。
  - **混合 SFT-RL(并行路线)**,四类:
    - ① **顺序 SFT→RL**(InstructGPT):概念简单,但容易**灾难性遗忘**掉 SFT 学到的知识,且低效;
    - ② **统一/交错框架**:如 ReLIFT(Ma,交替 RL 与对难题的在线微调)、LUFFY(Yan,并发,用混合策略选择性模仿高质量外部轨迹)、TAPO(Wu,注入抽象"思维模式")、SASR(Chen)/SuperRL(Liu,自适应切换 SFT/RL)。**短板**=常依赖复杂、可能不稳的启发式;
    - ③ **"GRPO w/ SFT Loss"**(简单把 SFT loss 直接加到 RL 上):**反而掉点**(Table1 中 40.1 < GRPO 的 45.5),说明二者融合并不平凡;
    - ④ **UFT**(Wang,统一 SFT-RL 以加速收敛):但**没有显式解决"稳定 off-policy 更新 + 同时导向新解探索"**这个核心。【§2.2,L207-251】
- 动机链(作者借孔子"学而不思则罔,思而不学则殆"作隐喻):当前 RLVR 是"思而不学"(只埋头想、不学外部知识),SFT 是"学而不思"(只模仿、不内化,遇新题就脆) → 需要一个既能稳定从外部 off-policy 数据"学"、又能显式激励"思"出低概率正确路径的混合策略。两大挑战:
  1. **分布失配**:标准重要性采样(IS)不够用——on-policy 代理有系统偏差,纯 off-policy 又高方差 + 支撑失配,而且外部策略 \(\pi_\omega\) 本身还是未知的;
  2. **高效提取外部数据的价值**:模型天生偏好高概率 token,但新知识恰恰常藏在那些被忽略的低概率正确 token 里。【§1,§2.2 Motivation】
- 与最近邻工作的精确Δ:
  - vs **LUFFY**(并发,同样用混合策略吸收外部轨迹):LUFFY 把外部策略概率近似为 1(当成完美 oracle);RL-PLUS 则用贝叶斯估计 \(\pi_\omega\approx\frac12\pi_{\theta_{\text{old}}}+\frac12 U\),消融显示这比 oracle 近似(44.6)和 \(\pi_{\theta_{\text{old}}}\) 近似(41.7)各高约 3 点(本文估计为 47.5)。
  - vs **SFT+GRPO**:RL-PLUS 平均高 +5.2 点(53.4 vs 48.2),且 OOD 不退化(而 SFT 系在 OOD 编程任务上严重退化,如 LeetCode 22.8→8.3)。
  - vs **DAPO/Dr.GRPO** 等纯 on-policy 改进:它们仍困在基座内,RL-PLUS 靠"外部数据 + 低概率探索"突破 pass@k 天花板。
  - 关键 Δ:**用 MIS 把"稳定吸收外部数据"理论化(给出有界方差)+ 用 focal 式 advantage 显式奖励低概率正确路径**,从而同时解决"稳定学"和"激励思"。

## 怎么做(到"读完能复现"的粒度)
> 一句话导读:在 GRPO 基底上加两个组件——MIS 比率(驯服 off-policy 外部数据)和 focal 式探索优势(放大低概率正确 token),再把二者拼成"内部利用 + 外部探索"的复合目标,并去掉 clip。下面逐块写到可复现。

### A. MDP 与 GRPO 基底
先把推理生成建模成一个 MDP:状态 \(s_t=q\oplus y_{<t}\)(问题拼上已生成前缀),动作 \(a_t\) = 选下一个 token \(y_t\),策略为 \(\pi_\theta\),reward \(R(q,y)\) 只在整条序列完成时才给(RLVR 下是稀疏二元:答对给 1、否则给 0)。标准 GRPO 目标:
\(\displaystyle J_{\text{RL}}(\theta)=\mathbb E_{(q,y)\sim\mathcal D_{\text{on}}}\Big[\sum_{t=1}^{|y|}\min\!\big(r_{i,t}(\theta)A_i,\ \mathrm{clip}(r_{i,t}(\theta),1-\epsilon,1+\epsilon)A_i\big)\Big]-\beta D_{\mathrm{KL}}[\pi_\theta\Vert\pi_{\text{ref}}],\)
其中比率 \(r_{i,t}(\theta)=\dfrac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q,o_{i,<t})}\),组相对 advantage \(A_i=\dfrac{R_i-\mathrm{mean}(\{R_1,\dots,R_G\})}{\mathrm{std}(\{R_1,\dots,R_G\})}\)。评测用 pass@k(采 k 次至少答对一次的比例,衡量的是"可解问题集"而非只看 pass@1)。【§2.1 式1-3】

### B. 组件1:MIS 驯服 off-policy(§3.1,方差护栏 + 贝叶斯估计)
- **问题**:从一个静态的外部数据集 \(\mathcal D_e=\{e_i\}\) 学习时,目标策略 \(\pi_\theta\) 与生成这批数据的未知行为策略 \(\pi_\omega\) 之间存在分布漂移。直接套用三种现成权重都不行:
  - on-policy IS(分母用 \(\pi_{\theta_{\text{old}}}\) 当代理)对外部数据有系统偏差(Lemma A.5);
  - 理论上正确的 off-policy 权重 \(r^e_t(\theta)=\pi_\theta(e_t\mid e_{<t})/\pi_\omega(e_t\mid e_{<t})\) 又会支撑失配(A.6)+ 高方差(A.7);
  - 而且 \(\pi_\omega\) 根本就是未知的。
- **MIS 比率**:把外部样本看作由 \(\pi_{\theta_{\text{old}}}\) 与 \(\pi_\omega\) 的混合策略生成,逐 token 比率为
\(\displaystyle r^m_{i,t}(\theta)=\frac{2\,\pi_\theta(e_{i,t}\mid q,e_{i,<t})}{\pi_\omega(e_{i,t}\mid q,e_{i,<t})+\pi_{\theta_{\text{old}}}(e_{i,t}\mid q,e_{i,<t})}.\)
**直觉**:分母里的 \(\pi_{\theta_{\text{old}}}\)(它被刻意保持接近 \(\pi_\theta\))起到"方差护栏"的作用——即便 \(\pi_\omega\) 烂到极点,这个比值也仍有上界,从而把"理论正确但高方差的 off-policy"驯化成"可稳定训练"(对应 Remarks A.8/A.9 的有界畸变误差)。【Theorem 3.1:只要行为池里存在一个策略 ≈ \(\pi_\theta\),MIS 的方差就低,且对池中其他任意"坏"行为策略都不敏感】
- **估计未知的 \(\pi_\omega\)(贝叶斯)**:把 \(\pi_\omega\) 的模型空间限定为两个候选——一个是具体的代理 \(\pi_{\theta_{\text{old}}}\)(代表已有的可用信息),另一个是无信息的均匀策略 \(U(\tau)=1/V\)(代表最大不确定)。按 Principle of Indifference(无差别原则)给二者赋相等先验 \(P(\pi_\omega=\pi_{\theta_{\text{old}}})=P(\pi_\omega=U)=\frac12\),则最小化 Bayes 风险(期望 L2 误差)的估计器就是这两者的贝叶斯模型平均:
\(\displaystyle \hat\pi^*_\omega(\tau)=\tfrac12\,\pi_{\theta_{\text{old}}}(\tau)+\tfrac12\,U(\tau).\) 【Theorem 3.2,证明在附录A.5】

### C. 组件2:Exploration-Based Advantage(§3.2,focal 重加权)
做法:给外部 token 的 advantage 乘一个 focal 权重,以放大"正确、但难探索(低概率)"的路径:
\(\displaystyle A^c_{i,t}=\underbrace{\frac{R_i-\mathrm{mean}(\{R_1,\dots,R_G\})}{\mathrm{std}(\{R_1,\dots,R_G\})}}_{\text{标准化 reward(含内/外轨迹)}}\cdot\ C_{i,t},\qquad C_{i,t}=\big(1-\mathrm{detach}(\pi_\theta(e_{i,t}\mid q,e_{i,<t}))\big)^{\gamma}.\)
**直觉**(借自 focal loss):模型对某个正确外部 token 越没把握(即 \(\pi_\theta\) 越小),权重 \(C_{i,t}\) 就越大,于是把优势信号放大到"被忽视的低概率正确区域"。式中的 **detach/stop-gradient** 用来阻止梯度经概率项回传,增强稳定性。\(\gamma\) 控制放大强度,默认取 \(\gamma=0.5\)。【§3.2 式5/6】

### D. 复合目标 + 去 clip(§3.3)
\(\displaystyle J_{\text{RL-PLUS}}(\theta)=\underbrace{\mathbb E_{(o_i,A_i)\sim\mathcal D_o}\big[r_{i,t}(\theta)A_i\big]}_{\text{内部利用(Thinking)}}\ +\ \underbrace{\mathbb E_{(e_i,A^c_{i,t})\sim\mathcal D_e}\big[r^m_{i,t}(\theta)A^c_{i,t}\big]}_{\text{外部数据探索(Learning)}}.\)
两项各司其职:内部项就是标准 GRPO 的 policy gradient(稳住并精炼已有能力);外部项 = MIS 比率 × focal advantage(吸收外部新知识)。**去掉 clip 的理由**:\(\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\) 会砍掉"高信息低概率事件"的梯度——而这正是想学的新知识;去掉 clip 后,模型一旦遇到有价值的外部信息就能迈更大的步子,加速吸收、更有效地扩边界。**梯度直觉**(附录):梯度 \(\propto A_i\cdot(1-p_t)^\gamma\),当 \(p_t\to0\) 权重趋于 1、当 \(p_t\to1\) 权重趋于 0,即把更新聚焦到低概率的正确动作上。【§3.3 式7 + 附录梯度分析】

### E. 逐组件必要性(Table4 消融,base=Qwen2.5-Math-7B,Avg over 6 数学基准)
- **去 MIS**:53.4 → **45.5**(=回到 GRPO),最大跌幅——稳定吸收外部知识的核心。
- **去 Exploration-Based Advantage**:53.4 → **50.9**(−2.5)——高效探索低概率正确路径的贡献。
- **\(\pi_\omega\) 估计方式对比**(三 naive 变体):\(\pi_\theta/\pi_{\theta_{\text{old}}}\) 近似 41.7;\(\pi_\theta/\pi_\omega\) oracle≈1(LUFFY 式)44.6;**本文贝叶斯估计 47.5**(比 oracle 再 +2.9)。
- **去 clip / detach**:理论支撑(去 clip 保低概率梯度;detach 防概率项回传增稳),无独立数值消融。
- **\(\gamma\) 超参**(附录图6):不敏感但 \(\gamma=0.5\) 峰值;任何 \(\gamma\) 都超 GRPO。【§4,Table4】

### 数据/模型/评测(复现锚点)
- 基座:Qwen2.5-Math-7B(主);泛化在 **LLaMA-3.1-8B / DeepSeek-Math-7B / Qwen2.5-Math-1.5B** 上验证(LLaMA-3.1-8B 上 GRPO 几乎不涨,RL-PLUS 显著提升)。训练约 600 步级别(图2 横轴);框架 VeRL + DeepScaleR。
- ID 基准:AIME24/25、AMC、MATH-500、Minerva、Olympiad;OOD 基准:HumanEval/LeetCode/LiveCodeBench(编程)+ ARC-c/GPQA-diamond/MMLU-Pro(科学 QA)。
- **关键数字**:
  - ID 平均:RL-PLUS **53.4** > LUFFY 50.1 > TAPO 49.5 > ReLIFT 49.4 > SFT+GRPO 48.2 > GRPO 45.5;
  - OOD 平均 48.8 > GRPO 的 44.9(比次优高 +3.9);跨族相对增益最高达 **+69.2%**;作为对照,Qwen2.5-Math-7B base 仅 9.0/19.0;
  - pass@k(图1b/图3):RL-PLUS 全程都高于基座(说明真突破了天花板),而 GRPO 在大 k 处被基座反超;
  - 训练动态(图2):baseline 的熵坍缩到≈0;naive 地直接塞外部数据会导致"熵爆炸"(输出混乱);RL-PLUS 的熵不归零(保留探索潜力)、且 response length 稳步增长。【Table1/2/3,图1-3】

## 靠不靠谱
> 一句话导读:对照设计干净、pass@k/熵动态都正面回应了"是否真扩了边界";但要看清——它的"突破天花板"其实是"把外部数据里的新知识高效注入进来",新能力来自外部数据而非算法凭空创造,所以收益与外部数据质量强绑定。

- baseline 公平吗:同基座 Qwen2.5-Math-7B 横扫十余个 baseline(含并发的 LUFFY/ReLIFT/TAPO),且专门设了 "SFT / GRPO / GRPO w-SFT Loss / SFT+GRPO" 四个直接对照来隔离"外部知识 vs 自探索 vs 二者结合",公平性强;pass@k 与熵动态也正面回应了"是否真扩了边界"(而不是只刷 pass@1)。
- 看着强但没回答核心问题:**外部数据 \(\mathcal D_e\) 的质量/来源是个前提**——核心收益来自"稳定吸收高质量外部示范",但对 \(\mathcal D_e\) 该如何构造(用什么 teacher、怎么生成数据)正文着墨很少,本质上是把"能否突破边界"外包给了"外部数据有多好"。严格说,"突破基座天花板"= "突破基座 + 注入了外部数据所含的新知识",而非凭空产生新能力(这与 rethink_opd 的"高分≠新知识、需 teacher 带来新知识"异曲同工)。
- 假设与失效边界:
  - 【原文】① 需要一个静态的外部数据集 \(\mathcal D_e\)(且含正确轨迹);② \(\pi_\omega\) 未知,用 \(\frac12\pi_{\theta_{\text{old}}}+\frac12U\) 做贝叶斯估计(其假设是"具体代理 vs 最大不确定"各占 ½ 先验);③ 主实验在数学域 + Qwen2.5-Math-7B,泛化只在 4 个模型上验证。
  - 【推断】三处潜在脆弱点:① MIS 的方差护栏依赖"\(\pi_{\theta_{\text{old}}}\) 始终接近 \(\pi_\theta\)"——若训练中策略漂移过大(长训、或去 clip 后步子太大),护栏可能松动;② focal 权重对"正确但低概率"的判定依赖 reward 的正确性,一旦 verifier 噪声大、或外部数据含错误轨迹,就会把错误的低概率路径也一并放大;③ 去 clip 在外部数据质量差时可能放大不稳(naive 塞外部数据已出现"熵爆炸",RL-PLUS 靠 MIS 压住了,但这个边界并未被充分探明)。
- 祛魅总结:真贡献有二——① 用 pass@k 把"capability boundary collapse"刻画清楚,并给出一个有理论支撑(有界方差 MIS)的混合策略解;② 用 focal 式 exploration advantage 显式奖励低概率正确路径、配合去 clip,实证了 pass@k 能突破基座。
  - 【推断】可能被高估的是"突破天花板"这个叙事——更准确的说法是"高效注入了外部数据所含的知识 + 保住了探索熵",新能力来自外部数据、而非算法本身凭空创造。
  - 被低估的是 MIS 方差护栏这件可迁移的稳定性工具(它比 LUFFY 的 oracle 近似更 principled)。整体是混合 SFT-RL 谱系里"理论 + 探索"结合较好的一篇,但收益与外部数据质量强绑定。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=verifier 二元 reward → GRPO 组相对 advantage(内部)+ MIS 加权的外部数据 advantage,focal 重加权低概率正确 token | **改什么**=策略参数 θ;复合目标 = 内部利用项 + 外部探索项 | **何时改**=在线 RLVR,内外数据同批次联合优化(无 clip) | **免梯度?**=否(policy gradient);但 focal 权重内用 detach/stop-gradient 阻断概率项回传 | **记忆-技能生命周期**=无显式记忆/技能库;外部知识=静态数据集 De 经 MIS 吸收进参数 | **防遗忘机制**=核心针对"capability boundary collapse(=一种探索能力遗忘)";靠"内部利用项稳住已有能力 + 不归零的熵 + MIS 稳定融合"避免顺序 SFT→RL 的灾难性遗忘
- ⑦ 开源代码+框架/harness:https://github.com/YihongDong/RL-PLUS(v1 元信息已核:含 rl_plus/{deepscaler,verl,scripts,setup.py}、exp_scripts/、eval_scripts/、data/);框架 **VeRL + DeepScaleR**。【§1 脚注 L49】
- 💰 资源/成本与可扩展性:原文未给 GPU 时/卡数;训练 600 步级别(图2 横轴);在 4 个模型族(1.5B-8B)验证泛化。可推断成本=常规 7B RLVR 量级 + 外部数据生成开销;去 clip + MIS 不增显著算力。【§4,Table3】
- 🎯 对"探索-巩固"对标:**强支撑(探索侧),直接同源于 path-recovery,可借组件多**。
  - 映射:RL-PLUS 的"capability boundary collapse + 向内利用 vs 向外探索"正是 TSRD"探索/选路"要对抗的失败模式;其 **focal 式 exploration advantage(放大"正确但低概率"的路径)** 与 TSRD 的"path-recovery(走偏后自选恢复分支、并固化那条低概率但正确的恢复路径)"高度同源——双方都在显式奖励"模型自己不容易走到、但正确"的路径。
  - **可借组件**(四件):
    1. **MIS 方差护栏(\(\pi_{\theta_{\text{old}}}\) 在分母兜底)**:若 TSRD 用 teacher/外部恢复轨迹做 path-recovery 监督,MIS 是比 oracle 近似更稳的融合工具,可直接用于"把 teacher 的恢复分支当外部 off-policy 数据稳定吸收";
    2. **focal \((1-\pi_\theta)^\gamma\) 重加权**:可用于给 TSRD 的"低概率关键恢复 token"加权(与 rho1 的 excess-loss、revisit_entropy 的 token 异质性思路互补);
    3. **detach/stop-gradient 用在权重项**:与项目 memory 记录的"forward-hard/backward-soft 解耦"signature 同构(权重值参与前向、梯度被截断),是可迁移的稳定性技巧;
    4. pass@k 作为"是否真扩了可恢复路径集"的评测指标。
  - **缺口/差异**:RL-PLUS 是 RLVR + 外部静态数据,既无 teacher 的 token 级 dense 监督(它不是蒸馏)、也无 on-policy 自蒸馏、无 MTP 前瞻、无记忆库;它的"外部数据"是预先备好的静态集,而 TSRD 想要的是 teacher 在线当稀疏脚手架、动态给出恢复分支。
  - 判定:**探索/path-recovery 侧的强方法参考 + 多个可直接移植的稳定性组件(MIS 护栏、focal 重加权、stop-grad)**。
- 🔭 开放问题/未来方向:【原文】Conclusion 强调 pass@k 与训练动态证明突破天花板;γ 仍有细调空间(附录图6);未设详尽 Future Work。【推断】把静态外部数据 De 换成 teacher 在线生成的恢复分支(向 on-policy/OPD 靠拢)、把 MIS 护栏用于稳定 OPD/自蒸馏中的 off-policy 成分、把 focal exploration advantage 与 MTP 前瞻结合(用前瞻定位"将走偏需恢复"的低概率正确 token 优先放大)、研究"突破边界"中外部数据质量的下界与 verifier 噪声鲁棒性。

RETURN: rl_plus|读PDF?是(_txt 1000+行,核到§2-§4全文+Table4消融真实数值+式1-7)|加厚?是(方法扩为A-E五块+Theorem3.1/3.2+复现锚点,相关工作两线精确化,7条公式转MathJax)|LaTeX公式条数 8|待核数 0

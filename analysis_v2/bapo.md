bapo | BAPO: Stabilizing Off-Policy Reinforcement Learning for LLMs via Balanced Policy Optimization with Adaptive Clipping | 复旦 FudanNLP / 上海稷迹智锋 / 上海创新研究院（共一 Zhiheng Xi、Xin Guo;通讯 Tao Gui、Qi Zhang、黄萱菁）| 2025-10（arXiv:2510.18927v1，cs.LG）| 主题线 L3 RLVR/GRPO·相关性 中

**原始论文**：https://arxiv.org/abs/2510.18927

## 一眼看懂
- 🟦 TL;DR：用过期数据训 LLM 的 **off-policy RL**（rollout 行为策略 ≠ 训练目标策略,经验复用 experience replay / partial rollout）样本效率高、容忍 staleness,但数据越陈旧越容易**梯度爆炸 + 熵崩溃**（Fig.2:GRPO 在 staleness 0/2/4/8× 下熵骤降、grad norm 飙、训练崩）。BAPO 先诊断两个根因——(i) **负优势样本在数量和梯度贡献上都压倒正样本**;(ii) **固定对称裁剪系统性挡掉"增熵更新"**（Entropy-Clip Rule）——再据此**每个 batch 动态搜裁剪上下界 \((c_{\text{low}},c_{\text{high}})\)**，让"正 token 对策略梯度损失的贡献占比"达到目标 \(\rho_0\)，从而既防爆炸又保熵。
- 最巧的一步：**把"正 token 贡献占比 \(\ge\rho_0\)"当作可观测的自适应调节目标（§4.2 Eq.8）**。抽掉它,BAPO 就退回 DAPO 式的固定非对称裁剪——而 §4.1 验证实验恰恰说明固定阈值"僵硬、需手调、缺乏适应"（"relatively rigid and manually specified"）。是这个"占比目标 + 动态搜界"让裁剪窗能 per-batch 自调,省掉手工调参。

## 为什么做
- 研究背景：RL 已是对齐/强化 LLM（推理 Guo 2025、代码 Anthropic 2025、agentic Bai 2025-Kimi K2）的核心范式。其中 **off-policy RL**（Arnal 2025、Roux 2025）样本效率高、容忍 staleness,契合现代基础设施（**partial rollout**:超长轨迹切段、超 budget 的部分存 replay buffer 后续续跑,Fu 2025 / Kimi Team 2025）,适合超长/难任务（§1）。
- 解决的具体痛点：直接把 off-policy RL 套到 LLM 上,随 staleness 增大出现**优化不稳、梯度爆炸、甚至崩溃,同时策略熵骤降**（Fig.2 用 GRPO 复现:staleness↑ → 熵降更剧、被裁 token 更多、训练更不稳）;on-policy 则全程稳定。已有非对称裁剪（Clip-Higher/DAPO-Yu 2025、Asymmetric REINFORCE-Arnal 2025、target-entropy-He 2025、80/20 高熵 token-Wang 2025a、DCPO-Yang 2025b）**靠手动固定阈值**,僵硬。
- 相关工作 & 各自不足（§6）：
  - **DAPO/Clip-Higher（Yu 2025）**——把裁剪上界设 1.28,纳入更多低概率正 token、稳熵;但**只放宽上界、不抑制负 token 主导**,且固定阈值。
  - **高熵 token 子集（Wang 2025a）**——只训 top20% 最高熵 token 保探索;**target-entropy（He 2025）**——把熵拉到目标值。思路相近但**各自固定策略**。
  - **熵稳定性系列（Cui 2025、Cheng 2025 entropy perspective、Liu 2025、Zheng 2025）**——系统研究怎么维持熵稳定。
  - **off-policy 非对称裁剪（Roux 2025、Arnal 2025）**——引入非对称裁剪平衡正负奖励。
  - **最像的 DCPO（Yang 2025b）**——按 **token 先验概率**调 **token 级**裁剪;但 BAPO 自称取"**整体优化视角**":从损失贡献失衡 + Entropy-Clip Rule 出发动态调**全局**裁剪界,且做了更大规模实验。
- 动机链（§3 实证 + 理论双线）：现状（off-policy RL 高效但 LLM 上易崩）→ **诊断一(失衡)**：§3 Fig.4 显示正样本在数量和损失贡献上都是少数,归因两点——难题上轨迹更长→负样本 token 更多（Fig.6 负样本平均长度更长）+ 训练早期能力不足→负样本比例高;且低概率负 token 累积（\(\pi_\theta(y_t)\to0\) 使 log 项→\(-\infty\)）触发梯度爆炸 → **诊断二(熵崩)**：推导 Entropy-Clip Rule（见下）→ §4.1 验证实验（Fig.7:增 \(c_{\text{high}}\) 纳入低概率正 token 提性能+抑熵降;放宽 \(c_{\text{low}}\) 纳入低概率负 token 反而降性能+加速熵崩）→ "用占比目标驱动的自适应裁剪"。为什么不用固定非对称裁剪？§4.1 试过 Clip=[0.8,1.5]/[0.8,1.2]/[0.5,1.2]——僵硬、需手调、单一阈值不能 per-batch 适应难度变化。
- 与最近邻工作的Δ：vs DAPO/Clip-Higher——BAPO **同时**动态调上下界（\(c_{\text{high}}\) 纳入低概率正 token 增熵、\(c_{\text{low}}\) 过滤低概率负 token 防爆炸）,且由"正 token 贡献占比 \(\rho_0\)"自适应驱动而非固定;vs DCPO——从损失贡献失衡的**全局视角**调全局裁剪界,并配 **Entropy-Clip Rule** 理论。关键点:把"该裁多少"从手工超参变成"满足贡献占比"的可搜索量。

## 怎么做 + 靠不靠谱

> 基础设定（§2）：prompt \(x\)，response \(\boldsymbol y=(y_1,\dots,y_T)\)，\(\pi_\theta(\boldsymbol y\mid x)=\prod_t\pi_\theta(y_t\mid x,\boldsymbol y_{<t})\)。RL 目标 \(J(\theta)=\mathbb E_{x,\boldsymbol y\sim\pi_\theta}[R(x,\boldsymbol y)]\)，policy gradient \(\nabla_\theta J=\mathbb E[\sum_t\nabla_\theta\log\pi_\theta(y_t)\cdot A_t]\)。PPO 代理目标用 importance weight \(r_t=\frac{\pi_\theta(y_t\mid x,\boldsymbol y_{<t})}{\pi_{\theta_{\text{rollout}}}(y_t\mid x,\boldsymbol y_{<t})}\) 修正分布失配:
> \(\displaystyle J_{\text{PPO}}(\theta)=\mathbb E_{x,\boldsymbol y\sim\pi_{\theta_{\text{rollout}}}}\Big[\textstyle\sum_{t=1}^T\min\big(r_t A_t,\ \mathrm{clip}(r_t,1-\varepsilon,1+\varepsilon)A_t\big)\Big].\)

**【两条诊断的形式化】**
- **失衡（§3 Eq.5）**：把 PG 按正负 advantage 拆开,裁剪体现为指示函数——
  \(\displaystyle \nabla J_{\text{PPO}}=\underbrace{\sum_{A_t>0}\pi_\theta(y_t)\,\mathbb I\{r_t<1+\varepsilon\}\,A_t\,\nabla\log\pi_\theta(y_t)}_{\text{正 token}}+\underbrace{\sum_{A_t<0}\pi_\theta(y_t)\,\mathbb I\{r_t>1-\varepsilon\}\,A_t\,\nabla\log\pi_\theta(y_t)}_{\text{负 token}}.\)
  上界 \(1+\varepsilon\) 把**低概率正 token（\(r_t\) 偏大）挡在外**,下界 \(1-\varepsilon\) 让低概率负 token 留下并累积。
- **Entropy-Clip Rule（§3 Eq.6,Appendix B 证明）**：熵变近似为对数概率与"裁后 advantage"的负协方差——
  \(\displaystyle \Delta H(\pi_\theta)\approx -\eta\cdot\mathrm{Cov}_{\boldsymbol y\sim\pi_\theta}\big[\log\pi_\theta(y_t\mid x,\boldsymbol y_{<t}),\ A_t\cdot\mathcal X(y_t)+C\big],\)
  其中 \(C\) 为常数,\(\mathcal X(y_t)=1\) 当（\(A_t>0\) 且 \(r_t<1+\epsilon\)）或（\(A_t<0\) 且 \(r_t>1-\epsilon\)）即"该 token 未被裁",否则 0。**直觉**:更新"正高概率 + 负低概率"token 会**锐化分布、降熵**;更新"负高概率 + 正低概率"token 会**平滑分布、增熵**。Fig.5/10 实证:IS 权重偏离 1（极高或极低）的 token 往往是低概率 token 且高熵——固定对称裁剪 [0.8,1.2] 把大量低概率正 token 挡在外,**系统性排除增熵更新→熵持续下降→探索能力枯竭**。BAPO 据此扩 \(c_{\text{high}}\) 把这些正 token 放回来。

**【BAPO 算法流水线（§4.2 Algorithm 1）——读完可复现】**
1. **采样**：更新 rollout 策略 \(\pi_{\theta_{\text{rollout}}}\leftarrow\pi_\theta\);从第 \(s\) 个 batch 采 \(G\) 条 response \(\{\boldsymbol y_i\}\sim\pi_{\theta_{\text{rollout}}}(\cdot\mid x)\);算 reward 与 advantage（GRPO group-relative）。
2. **对每个 staleness 层动态搜裁剪界**（核心循环）：初始化 \(c_{\text{low}}=a^-,\ c_{\text{high}}=a^+\);**while（正 token 贡献 \(\rho<\rho_0\) 且 \(c_{\text{low}}+\delta_2\le b^-\)）**：若 \(c_{\text{high}}+\delta_1\le b^+\) 则 **先增 \(c_{\text{high}}\)**（步长 \(\delta_1\)）,到顶后再增 \(c_{\text{low}}\)（步长 \(\delta_2\)）。目标占比定义（**Eq.8**）：
   \(\displaystyle \frac{\big|\sum_{A_t>0}\pi_{\theta_{\text{rollout}}}(y_t)\cdot\min(r_tA_t,\mathrm{clip}(r_t,0,c_{\text{high}})A_t)\big|}{\big|\sum_{A_t}\pi_{\theta_{\text{rollout}}}(y_t)\cdot\min(r_tA_t,\mathrm{clip}(r_t,c_{\text{low}},c_{\text{high}})A_t)\big|}\ \ge\ \rho_0.\)
   即"正 token 对 PG 损失的贡献绝对值 / 全部 token 的贡献绝对值"要达到 \(\rho_0\)。
3. **更新**：用搜到的 \((c_{\text{low}},c_{\text{high}})\) 跑非对称裁剪的 PPO/GRPO 目标
   \(\displaystyle J_{\text{BAPO}}(\theta)=\mathbb E_{\boldsymbol y\sim\pi_{\theta_{\text{rollout}}}}\Big[\textstyle\sum_t\min\big(r_tA_t,\ \mathrm{clip}(r_t,c_{\text{low}},c_{\text{high}})A_t\big)\Big].\)
- **逐组件必要性（§4.1 Fig.7 验证 + §4.3 Fig.8/9 训练动态）**：
  - **\(c_{\text{high}}\) 上调**——Fig.7（Clip=[0.8,1.5] vs [0.8,1.2]）:纳入低概率正 token **提性能 + 抑熵降**。
  - **\(c_{\text{low}}\) 上调（收紧下界过滤负 token）**——Fig.7 反向验证:放宽 \(c_{\text{low}}\)（[0.5,1.2]）反而**降性能 + 加速熵崩**,所以方向是"收紧下界过滤低概率负 token"。
  - **\(\rho_0\) 目标**——Fig.8 显示训练中上下界都在波动（证确实自适应,不是固定值）;**防熵失控**（设正 token 目标占比、避免熵无控增长）+ **防 tail degradation**（过拟合易题却答不了难题,Ding 2025）。
  - 三重收益（§4.2 原文）:①平衡正负贡献防爆炸;②纳入低概率正、过滤低概率负→保熵;③设占比上限→防正 token 淹没损失 + 防 tail degradation。
- **关键机制/公式（直觉）**：见上 Entropy-Clip Rule。机制图 Fig.3:GRPO 对称固定裁剪 → 强化高概率正 + 过罚低概率负 → 分布锐化、熵崩;BAPO 按正 token 损失贡献动态调 \(c_{\text{high}}/c_{\text{low}}\) → 排掉过负 token、放回被裁的正 token → 分布更平滑、熵稳。与 prior work 的连接（§4.3）:Clip-Higher（1.28）、80/20 高熵 token、target-entropy 都可视为"用某种方式纳入低概率正 token / 稳熵"的特例。
- **实验与证据（§5）**：RL 数据 **SkyWork-OR1-RL-Data**;评测 **AIME24/25**（16 rollouts 平均）。backbone:R1-Distill-Qwen-7B/32B、OctoThinker-Llama3.2-3B-Long-Zero,外加自训两个 SFT 起点 **BP-Math-7B/32B**（Qwen2.5-Math 微调）。
  - **主表 Table 1**：BP-Math-7B(BAPO) AIME24/25 = **70.8/62.5**（超 SkyWork-OR1-7B 70.2/54.6,AIME25 **+7.9**;超自家 GRPO 起点 69.2/59.2）;BP-Math-32B(BAPO) = **87.1/80.0**（超 Qwen3-32B 81.4/72.9 即 +5.7/+7.1、超 SkyWork-OR1-32B +4.9/+6.7;超 DeepSeek-R1-671B 79.8/70.0 即 +7.3/+10.0;比肩 o3-mini-medium 79.6/76.7）。
  - **Llama（Table 2）**：GRPO→BAPO,AIME24 2.5%→5.4%、AIME25 2.9%→5.8%、MATH 58.4%→66.0%。
  - **稳定性**：不同 staleness（Fig.11:staleness 2/4 下均超 baseline 与 clip-higher）、partial rollout（Fig.12:budget 2k/4k 下 reward 更高、熵更稳）均比 GRPO 稳。
- **假设与失效边界**：【原文】超参 \(\rho_0=0.4\)、可移动区间 \(a^-=0.6/b^-=0.9/a^+=1.2/b^+=3.0\)、步长 \(\delta_1=0.05/\delta_2=0.02\)"**未精调**"（§5.1,原文 "not finely tuned"）;预备/验证用 R1-Distill-7B、max len 8k、lr \(2\times10^{-6}\)、temp 0.6;BP-Math 主实验 max len 64k;评测 16 rollouts 平均。【推断】**核心评测只在 AIME24/25 两个小测试集**,Llama 才补 MATH,泛化证据弱;与 SkyWork 的对比依赖自训强起点 BP-Math,且 BAPO 相对自家 GRPO 在 32B 上增益小（AIME24 84.6→87.1）,削弱"方法本身"的归因;\(\rho_0\)/区间/步长的鲁棒性未系统扫描。
- **祛魅总结**：【推断】真贡献是 **Entropy-Clip Rule 这条把"裁剪→挡增熵更新→熵崩溃"讲通的理论** + "用正 token 贡献占比作自适应目标"这一干净的工程化。被包装的是算法增量——它与既有非对称裁剪家族（Clip-Higher/KL-Cov/CE-GPPO/80-20/DCPO）**同源**,新意主要在"占比驱动 + 理论解释"。主结果用"超 DeepSeek-R1/o3-mini"做标题党,但那部分靠强 SFT 起点 BP-Math。**重要不一致**：旧 analysis 核查指出**开源代码 `recipe/bapo/policy_loss.py` 的搜索顺序与论文 Algorithm 1 相反**（代码先调 \(c_{\text{low}}\) 再 \(c_{\text{high}}\);论文先 \(c_{\text{high}}\) 后 \(c_{\text{low}}\)）,且代码含 dual-clip（clip_ratio_c=3.0,对应论文 \(b^+=3.0\)）——复现以代码为准,但这是方法描述与实现的不一致。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=token 级 advantage（GRPO group-relative）+ 正/负 token 对策略梯度损失的贡献占比（用于调裁剪界）｜**改什么**=策略参数 + **裁剪上下界 \((c_{\text{low}},c_{\text{high}})\) 这一优化超参**（per-batch 动态搜索）｜**何时改**=在线 per-step（每 batch 搜界 + 更新）｜**免梯度?**=否（PPO/GRPO 梯度上升）｜**记忆-技能生命周期**=不适用｜**防遗忘机制**=不适用（间接:保熵防探索退化/tail degradation,但非持续学习意义的防遗忘）
- ⑦ 开源代码+框架/harness：https://github.com/WooooDyy/BAPO （旧 analysis 记已 clone 约 5.2MB）。框架=**veRL**，以 **GRPO** 为基础算法（`python -m verl.trainer.main_ppo` 主循环,方法在 `recipe/bapo`）。【待核】仓库默认 config（`bapo_trainer.yaml`/`run_bapo_example.sh`）是**示例值非论文值**（旧 analysis 已核:adv_ratio_target=1、ratio_lower 0.6→0.8 等）,复现论文数字需手动改回论文超参（\(\rho_0=0.4,a^-=0.6,b^-=0.9,a^+=1.2,b^+=3.0,\delta_1=0.05,\delta_2=0.02\)）;且代码搜索顺序与 Algorithm 1 相反（见祛魅）。本次基于 PDF 重写未重新进仓核对。
- 💰 资源/成本与可扩展性：【原文】预备/验证 R1-Distill-7B、max len 8k、lr 2e-6、temp 0.6;staleness 实验 SkyWork-OR1-RL、max len 32k;BP-Math 主实验 max len 64k。staleness 经 `ppo_epoch` 经验复用 + partial rollout 引入。每 step 多一次裁剪界搜索（轻量,在已算好的 advantage 上做,无额外前向）。
- 🎯 对"探索-巩固"对标：**竞品/边缘相关**。判定:BAPO 与 OPD/蒸馏**无直接关系**（纯 RLVR 裁剪稳定化）,对 TSRD 的价值是**间接借鉴**——其 Entropy-Clip Rule 解释了"为什么 off-policy/经验回放训练会熵崩溃",这对 TSRD 若用 partial rollout / teacher 轨迹回放（off-policy 成分）时**保持探索性**有参考;"保正 token、过滤负 token"的占比思想也可类比"巩固有效路径、不过度惩罚走偏分支"。**缺口**:信用分配仍是 outcome 级 group advantage（同轨迹所有 token 同一 \(A\)）,无 step/path 级,无蒸馏/teacher 脚手架/记忆,无 MTP。
- 🔭 开放问题/未来方向：【原文】"为 LLM RL 社区提供关键洞见"（§7,未给具体 future work）。【推断】\(\rho_0\)/区间/步长的自适应或学习化;扩到更广基准（非 AIME）;把 Entropy-Clip Rule 用于 agentic/multi-turn 的熵管理;统一代码与论文 Algorithm 1 的搜索顺序不一致。

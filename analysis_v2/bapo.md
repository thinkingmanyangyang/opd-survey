bapo | BAPO: Stabilizing Off-Policy Reinforcement Learning for LLMs via Balanced Policy Optimization with Adaptive Clipping | 复旦 FudanNLP / 上海稷迹智锋 / 上海创新研究院（共一 Zhiheng Xi、Xin Guo;通讯 Tao Gui、Qi Zhang、黄萱菁）| 2025-10（arXiv:2510.18927v1，cs.LG）| 主题线 L3 RLVR/GRPO·相关性 中

**原始论文**：https://arxiv.org/abs/2510.18927

## 一眼看懂

> 一句话导读：off-policy RL 用"攒下来的旧数据"训 LLM,省采样但容易训崩;BAPO 把"该裁掉多少梯度"从手工固定值改成每个 batch 自动搜出来的值,专门防崩、保探索。

- 🟦 TL;DR：所谓 **off-policy RL**,就是采数据的"行为策略"和正在优化的"目标策略"不是同一个——可以复用旧经验（experience replay）、把超长轨迹切段续跑（partial rollout）。好处是样本效率高、对数据陈旧（staleness）有容忍度。坏处是:数据越陈旧越容易**梯度爆炸 + 熵崩溃**。Fig.2 就复现了这点:GRPO 在 staleness 0/2/4/8× 下,熵骤降、梯度范数飙升、训练直接崩。
- BAPO 先把崩的根因诊断成两条。(i) **负优势样本在数量和梯度贡献上都压倒正样本**;(ii) **固定的对称裁剪会系统性地挡掉"能增熵的更新"**（即下文的 Entropy-Clip Rule）。
- 据此 BAPO 的做法是:**每个 batch 动态搜一对裁剪上下界 \((c_{\text{low}},c_{\text{high}})\)**,让"正 token 对策略梯度损失的贡献占比"达到一个目标值 \(\rho_0\)。这样既防爆炸,又保住熵（即保住探索）。
- 最巧的一步：**把"正 token 贡献占比 \(\ge\rho_0\)"当成一个可观测的自适应调节目标（§4.2 Eq.8）**。把它抽掉,BAPO 就退回成 DAPO 式的"固定非对称裁剪"。而 §4.1 的验证实验恰好说明固定阈值"僵硬、要手调、缺乏适应"（原文 "relatively rigid and manually specified"）。正是这个"占比目标 + 动态搜界"的组合,让裁剪窗口能逐 batch 自调,省掉手工调参。

## 为什么做

> 一句话导读：off-policy RL（复用旧数据）很省采样,但越省越容易把 LLM 训崩;现有的"非对称裁剪"补丁都靠手调一个固定阈值,僵硬。BAPO 想让这个阈值自己动起来。

- 研究背景：RL 已是对齐/强化 LLM 的核心范式（推理 Guo 2025、代码 Anthropic 2025、agentic Bai 2025-Kimi K2）。其中 **off-policy RL**（Arnal 2025、Roux 2025）样本效率高、容忍数据陈旧（staleness）。它也契合现代训练基础设施——比如 **partial rollout**:把超长轨迹切段、超出 budget 的部分先存进 replay buffer 后续再续跑（Fu 2025 / Kimi Team 2025）,因此适合超长、超难的任务（§1）。
- 解决的具体痛点：把 off-policy RL 直接套到 LLM 上,随着 staleness 增大,会出现**优化不稳、梯度爆炸、甚至训练崩溃,同时策略熵骤降**。Fig.2 用 GRPO 复现了这点:staleness 越大,熵降得越剧烈、被裁掉的 token 越多、训练越不稳;而纯 on-policy 则全程稳定。
- 已有的补丁是各种"非对称裁剪"——Clip-Higher/DAPO（Yu 2025）、Asymmetric REINFORCE（Arnal 2025）、target-entropy（He 2025）、只训 80/20 高熵 token（Wang 2025a）、DCPO（Yang 2025b）。但它们**都靠手动设一个固定阈值**,僵硬。
- 相关工作 & 各自不足（§6）：
  - **DAPO/Clip-Higher（Yu 2025）**——把裁剪上界设 1.28,纳入更多低概率正 token、稳熵;但**只放宽上界、不抑制负 token 主导**,且固定阈值。
  - **高熵 token 子集（Wang 2025a）**——只训 top20% 最高熵 token 保探索;**target-entropy（He 2025）**——把熵拉到目标值。思路相近但**各自固定策略**。
  - **熵稳定性系列（Cui 2025、Cheng 2025 entropy perspective、Liu 2025、Zheng 2025）**——系统研究怎么维持熵稳定。
  - **off-policy 非对称裁剪（Roux 2025、Arnal 2025）**——引入非对称裁剪平衡正负奖励。
  - **最像的 DCPO（Yang 2025b）**——按 **token 先验概率**调 **token 级**裁剪;但 BAPO 自称取"**整体优化视角**":从损失贡献失衡 + Entropy-Clip Rule 出发动态调**全局**裁剪界,且做了更大规模实验。
- 动机链（§3 实证 + 理论双线,分两步走）：
  - 起点是现状:off-policy RL 高效,但在 LLM 上易崩。
  - **诊断一（失衡）**：§3 Fig.4 显示正样本在"数量"和"损失贡献"上都是少数。原因有两条——难题上轨迹更长,导致负样本 token 更多（Fig.6:负样本平均长度更长）;训练早期模型能力不足,导致负样本比例本就高。更糟的是,低概率负 token 会累积:当 \(\pi_\theta(y_t)\to0\) 时 log 项趋向 \(-\infty\),直接触发梯度爆炸。
  - **诊断二（熵崩）**：据此推导出 Entropy-Clip Rule（见下文）。§4.1 的验证实验印证了它——Fig.7 显示,调高 \(c_{\text{high}}\) 把低概率正 token 纳进来,能提性能、抑熵降;反过来放宽 \(c_{\text{low}}\) 把低概率负 token 纳进来,反而降性能、加速熵崩。
  - 两条诊断合起来,导出"用占比目标驱动的自适应裁剪"。
  - 为什么不直接用固定的非对称裁剪？§4.1 试过 Clip=[0.8,1.5]、[0.8,1.2]、[0.5,1.2] 等——都僵硬、要手调,单一阈值无法逐 batch 适应难度变化。
- 与最近邻工作的 Δ：
  - vs DAPO/Clip-Higher——BAPO **同时**动态调上、下界:\(c_{\text{high}}\) 用来纳入低概率正 token 以增熵,\(c_{\text{low}}\) 用来过滤低概率负 token 以防爆炸;且整个调节由"正 token 贡献占比 \(\rho_0\)"自适应驱动,而不是写死。
  - vs DCPO——BAPO 站在"损失贡献失衡"的**全局视角**调全局裁剪界,并配上 **Entropy-Clip Rule** 这层理论解释。
  - 关键点:把"该裁多少"从一个手工超参,变成"满足贡献占比"的可搜索量。

## 怎么做 + 靠不靠谱

> 一句话导读：先把"训崩"用两条公式讲清楚——一条说"负 token 在梯度里压倒正 token",一条（Entropy-Clip Rule）说"固定裁剪专门挡掉能增熵的更新";再给出 BAPO 的动态搜界算法把这两条都对症下药。

> 基础设定（§2）：给定 prompt \(x\) 和 response \(\boldsymbol y=(y_1,\dots,y_T)\)，整条轨迹的概率按自回归分解 \(\pi_\theta(\boldsymbol y\mid x)=\prod_t\pi_\theta(y_t\mid x,\boldsymbol y_{<t})\)。RL 目标是最大化期望奖励 \(J(\theta)=\mathbb E_{x,\boldsymbol y\sim\pi_\theta}[R(x,\boldsymbol y)]\)，对应的 policy gradient 是 \(\nabla_\theta J=\mathbb E[\sum_t\nabla_\theta\log\pi_\theta(y_t)\cdot A_t]\)。由于 off-policy 下采样分布和当前分布不一致,PPO 用 importance weight（重要性权重,即新旧策略概率之比）\(r_t=\frac{\pi_\theta(y_t\mid x,\boldsymbol y_{<t})}{\pi_{\theta_{\text{rollout}}}(y_t\mid x,\boldsymbol y_{<t})}\) 来修正这种失配:
> \(\displaystyle J_{\text{PPO}}(\theta)=\mathbb E_{x,\boldsymbol y\sim\pi_{\theta_{\text{rollout}}}}\Big[\textstyle\sum_{t=1}^T\min\big(r_t A_t,\ \mathrm{clip}(r_t,1-\varepsilon,1+\varepsilon)A_t\big)\Big].\)

**【两条诊断的形式化】**
- **失衡（§3 Eq.5）**：把 policy gradient 按 advantage 的正负拆成两半,裁剪在公式里体现为指示函数（被裁的 token 系数置 0）——
  \(\displaystyle \nabla J_{\text{PPO}}=\underbrace{\sum_{A_t>0}\pi_\theta(y_t)\,\mathbb I\{r_t<1+\varepsilon\}\,A_t\,\nabla\log\pi_\theta(y_t)}_{\text{正 token}}+\underbrace{\sum_{A_t<0}\pi_\theta(y_t)\,\mathbb I\{r_t>1-\varepsilon\}\,A_t\,\nabla\log\pi_\theta(y_t)}_{\text{负 token}}.\)
  读法:上界 \(1+\varepsilon\) 会把**低概率正 token（它们的 \(r_t\) 偏大）挡在更新之外**;下界 \(1-\varepsilon\) 却让低概率负 token 留下来不断累积。一挡一留,正负就失衡了。
- **Entropy-Clip Rule（§3 Eq.6,Appendix B 给证明）**：这条规则把"熵的变化"近似写成"对数概率"与"裁剪后 advantage"的负协方差——
  \(\displaystyle \Delta H(\pi_\theta)\approx -\eta\cdot\mathrm{Cov}_{\boldsymbol y\sim\pi_\theta}\big[\log\pi_\theta(y_t\mid x,\boldsymbol y_{<t}),\ A_t\cdot\mathcal X(y_t)+C\big],\)
  其中 \(C\) 是常数;\(\mathcal X(y_t)=1\) 表示"该 token 没被裁"——具体是当（\(A_t>0\) 且 \(r_t<1+\epsilon\)）或（\(A_t<0\) 且 \(r_t>1-\epsilon\)）时取 1,否则取 0。
  **直觉**:更新"高概率正 token + 低概率负 token"会**锐化分布、降熵**;更新"高概率负 token + 低概率正 token"则会**平滑分布、增熵**。
  Fig.5/10 给了实证:importance weight 偏离 1（要么极高要么极低）的 token,往往就是那些低概率、高熵的 token。固定的对称裁剪 [0.8,1.2] 把大量低概率正 token 挡在外面,于是**系统性地排除了增熵更新**,结果就是熵持续下降、探索能力枯竭。BAPO 据此把 \(c_{\text{high}}\) 调大,把这些正 token 放回来。

**【BAPO 算法流水线（§4.2 Algorithm 1）——读完可复现】**
1. **采样**：先把 rollout 策略同步到当前策略 \(\pi_{\theta_{\text{rollout}}}\leftarrow\pi_\theta\);从第 \(s\) 个 batch 采 \(G\) 条 response \(\{\boldsymbol y_i\}\sim\pi_{\theta_{\text{rollout}}}(\cdot\mid x)\);算出 reward 和 advantage（用 GRPO 的组内相对方式）。
2. **对每个 staleness 层动态搜裁剪界**（核心循环）：初始化 \(c_{\text{low}}=a^-,\ c_{\text{high}}=a^+\)。然后在 **while（正 token 贡献 \(\rho<\rho_0\) 且 \(c_{\text{low}}+\delta_2\le b^-\)）** 的条件下循环:若 \(c_{\text{high}}+\delta_1\le b^+\),就**先增 \(c_{\text{high}}\)**（步长 \(\delta_1\)）;等 \(c_{\text{high}}\) 到顶,再增 \(c_{\text{low}}\)（步长 \(\delta_2\)）。这里的目标占比 \(\rho\) 定义如下（**Eq.8**）：
   \(\displaystyle \frac{\big|\sum_{A_t>0}\pi_{\theta_{\text{rollout}}}(y_t)\cdot\min(r_tA_t,\mathrm{clip}(r_t,0,c_{\text{high}})A_t)\big|}{\big|\sum_{A_t}\pi_{\theta_{\text{rollout}}}(y_t)\cdot\min(r_tA_t,\mathrm{clip}(r_t,c_{\text{low}},c_{\text{high}})A_t)\big|}\ \ge\ \rho_0.\)
   读法:分子是"正 token 对 policy gradient 损失的贡献绝对值",分母是"全部 token 的贡献绝对值",二者之比要达到 \(\rho_0\)。
3. **更新**：用搜到的这对 \((c_{\text{low}},c_{\text{high}})\) 跑非对称裁剪的 PPO/GRPO 目标——
   \(\displaystyle J_{\text{BAPO}}(\theta)=\mathbb E_{\boldsymbol y\sim\pi_{\theta_{\text{rollout}}}}\Big[\textstyle\sum_t\min\big(r_tA_t,\ \mathrm{clip}(r_t,c_{\text{low}},c_{\text{high}})A_t\big)\Big].\)
- **逐组件必要性（§4.1 Fig.7 验证 + §4.3 Fig.8/9 训练动态）**：
  - **调高 \(c_{\text{high}}\)**——Fig.7（Clip=[0.8,1.5] vs [0.8,1.2]）显示:把低概率正 token 纳进来能**提性能 + 抑制熵降**。
  - **调高 \(c_{\text{low}}\)（即收紧下界、过滤负 token）**——Fig.7 给了反向证据:反过来放宽 \(c_{\text{low}}\)（[0.5,1.2]）会**降性能 + 加速熵崩**。所以正确方向是"收紧下界,过滤低概率负 token"。
  - **\(\rho_0\) 这个目标本身**——Fig.8 显示训练中上下界一直在波动,这就证明它确实是自适应的、不是固定值。它的作用有二:一是**防熵失控**（给正 token 设个目标占比,避免熵无控制地增长）;二是**防 tail degradation**（即过拟合简单题、却答不了难题,Ding 2025）。
  - 综合起来,§4.2 原文给出三重收益:① 平衡正负贡献,防爆炸;② 纳入低概率正、过滤低概率负,保熵;③ 给占比设上限,既防正 token 淹没损失,又防 tail degradation。
- **关键机制/公式（直觉）**：核心就是上面的 Entropy-Clip Rule。机制图 Fig.3 把两种做法对照得很清楚:GRPO 用对称固定裁剪,会强化高概率正 token、过罚低概率负 token,导致分布锐化、熵崩;BAPO 则按正 token 的损失贡献动态调 \(c_{\text{high}}/c_{\text{low}}\),把过负的 token 排掉、把被误裁的正 token 放回来,于是分布更平滑、熵更稳。与 prior work 的连接（§4.3）:Clip-Higher（上界 1.28）、80/20 高熵 token、target-entropy,都可以看成"用某种方式纳入低概率正 token / 稳熵"的特例。
- **实验与证据（§5）**：RL 数据用 **SkyWork-OR1-RL-Data**;评测在 **AIME24/25**（取 16 次 rollout 的平均）。backbone 有 R1-Distill-Qwen-7B/32B、OctoThinker-Llama3.2-3B-Long-Zero,另外自训了两个 SFT 起点 **BP-Math-7B/32B**（在 Qwen2.5-Math 上微调）。
  - **主表 Table 1**：BP-Math-7B(BAPO) 在 AIME24/25 拿到 **70.8/62.5**——超过 SkyWork-OR1-7B 的 70.2/54.6（AIME25 上 **+7.9**）,也超过自家 GRPO 起点 69.2/59.2。BP-Math-32B(BAPO) 拿到 **87.1/80.0**——超 Qwen3-32B 81.4/72.9（+5.7/+7.1）、超 SkyWork-OR1-32B（+4.9/+6.7）、超 DeepSeek-R1-671B 79.8/70.0（+7.3/+10.0）,并比肩 o3-mini-medium 79.6/76.7。
  - **Llama（Table 2）**：GRPO 换成 BAPO 后,AIME24 从 2.5% 升到 5.4%、AIME25 从 2.9% 升到 5.8%、MATH 从 58.4% 升到 66.0%。
  - **稳定性**：无论是不同 staleness（Fig.11:staleness 2/4 下均超 baseline 和 clip-higher）,还是 partial rollout（Fig.12:budget 2k/4k 下 reward 更高、熵更稳）,BAPO 都比 GRPO 稳。
- **假设与失效边界**：
  - 【原文】超参 \(\rho_0=0.4\)、可移动区间 \(a^-=0.6/b^-=0.9/a^+=1.2/b^+=3.0\)、步长 \(\delta_1=0.05/\delta_2=0.02\),作者自称"**未精调**"（§5.1,原文 "not finely tuned"）。预备/验证阶段用 R1-Distill-7B、max len 8k、lr \(2\times10^{-6}\)、temp 0.6;BP-Math 主实验 max len 64k;评测取 16 次 rollout 平均。
  - 【推断】这里有几处证据偏薄。一是**核心评测只在 AIME24/25 这两个小测试集**,只有 Llama 才补了 MATH,泛化证据弱。二是与 SkyWork 的对比依赖自训的强起点 BP-Math,而 BAPO 相对自家 GRPO 在 32B 上增益其实不大（AIME24 84.6→87.1）,这削弱了"增益来自方法本身"的归因。三是 \(\rho_0\)、区间、步长的鲁棒性都没系统扫描过。
- **祛魅总结**：
  - 【推断】真贡献有两块:一是 **Entropy-Clip Rule——这条把"裁剪 → 挡掉增熵更新 → 熵崩溃"这条因果链讲通的理论**;二是"用正 token 贡献占比当自适应目标"这一干净的工程化。
  - 被包装的是算法增量本身——它与既有的非对称裁剪家族（Clip-Higher/KL-Cov/CE-GPPO/80-20/DCPO）**同源**,新意主要落在"占比驱动 + 理论解释"上。主结果用"超 DeepSeek-R1/o3-mini"做标题,但那部分其实靠的是强 SFT 起点 BP-Math。
  - **一处重要不一致**：旧 analysis 核查发现,**开源代码 `recipe/bapo/policy_loss.py` 的搜索顺序与论文 Algorithm 1 相反**——代码是先调 \(c_{\text{low}}\) 再 \(c_{\text{high}}\),论文却是先 \(c_{\text{high}}\) 后 \(c_{\text{low}}\)。而且代码含 dual-clip（clip_ratio_c=3.0,对应论文的 \(b^+=3.0\)）。复现应以代码为准,但这确实是"方法描述"与"实现"之间的不一致。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=token 级 advantage（GRPO group-relative）+ 正/负 token 对策略梯度损失的贡献占比（用于调裁剪界）｜**改什么**=策略参数 + **裁剪上下界 \((c_{\text{low}},c_{\text{high}})\) 这一优化超参**（per-batch 动态搜索）｜**何时改**=在线 per-step（每 batch 搜界 + 更新）｜**免梯度?**=否（PPO/GRPO 梯度上升）｜**记忆-技能生命周期**=不适用｜**防遗忘机制**=不适用（间接:保熵防探索退化/tail degradation,但非持续学习意义的防遗忘）
- ⑦ 开源代码+框架/harness：https://github.com/WooooDyy/BAPO （旧 analysis 记已 clone 约 5.2MB）。框架=**veRL**，以 **GRPO** 为基础算法（`python -m verl.trainer.main_ppo` 主循环,方法在 `recipe/bapo`）。【待核】仓库默认 config（`bapo_trainer.yaml`/`run_bapo_example.sh`）是**示例值非论文值**（旧 analysis 已核:adv_ratio_target=1、ratio_lower 0.6→0.8 等）,复现论文数字需手动改回论文超参（\(\rho_0=0.4,a^-=0.6,b^-=0.9,a^+=1.2,b^+=3.0,\delta_1=0.05,\delta_2=0.02\)）;且代码搜索顺序与 Algorithm 1 相反（见祛魅）。本次基于 PDF 重写未重新进仓核对。
- 💰 资源/成本与可扩展性：【原文】预备/验证 R1-Distill-7B、max len 8k、lr 2e-6、temp 0.6;staleness 实验 SkyWork-OR1-RL、max len 32k;BP-Math 主实验 max len 64k。staleness 经 `ppo_epoch` 经验复用 + partial rollout 引入。每 step 多一次裁剪界搜索（轻量,在已算好的 advantage 上做,无额外前向）。
- 🎯 对"探索-巩固"对标：**竞品/边缘相关**。判定:BAPO 是纯 RLVR 的裁剪稳定化,与 OPD/蒸馏**没有直接关系**,对 TSRD 的价值只是**间接借鉴**。具体两点:① 它的 Entropy-Clip Rule 解释了"为什么 off-policy / 经验回放训练会熵崩溃"——如果 TSRD 用到 partial rollout 或 teacher 轨迹回放（这些都带 off-policy 成分）,这条规则对"如何保持探索性"有参考价值;② 它"保正 token、过滤负 token"的占比思想,也可类比成"巩固有效路径、但不过度惩罚走偏的分支"。**缺口**:它的信用分配仍是 outcome 级的 group advantage（同一条轨迹的所有 token 共享同一个 \(A\)）,既没有 step/path 级粒度,也没有蒸馏/teacher 脚手架/记忆,更没有 MTP。
- 🔭 开放问题/未来方向：【原文】"为 LLM RL 社区提供关键洞见"（§7,未给具体 future work）。【推断】\(\rho_0\)/区间/步长的自适应或学习化;扩到更广基准（非 AIME）;把 Entropy-Clip Rule 用于 agentic/multi-turn 的熵管理;统一代码与论文 Algorithm 1 的搜索顺序不一致。

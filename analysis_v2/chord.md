chord | On-Policy RL Meets Off-Policy Experts: Harmonizing SFT and RL via Dynamic Weighting (CHORD) | 阿里巴巴集团（Wenhao Zhang、Yuexiang Xie、Yuchang Sun、Yanxi Chen、Guoyin Wang、Yaliang Li(通讯)、Bolin Ding、Jingren Zhou） | v3 2026-03-17 · ICLR 2026 · cs.LG | 主题线 L2（统一 SFT-RL / GFT 类代表作）·相关性 高

**原始论文**：https://arxiv.org/abs/2508.11408

## 一眼看懂

> 一句话导读：别再"先 SFT 再 RL"分两段跑。CHORD 把 SFT 变成 RL 里的一个辅助项，用一个全局系数控制"先模仿专家、后自主探索"的过渡，再用一条抛物线门控决定每个 token 学多少——只在"模型还没把握"的 token 上使劲学。

- 🟦 TL;DR：作者实测发现，在 instruct 模型上"SFT-then-RL"会走出一条 **"shift-readapt-overfit"（漂移-再适应-过拟合）** 的三阶段曲线，而且不一定比纯 RL 好【原文 §3.1, Fig.1/2】。
- CHORD 把 SFT 重构成 on-policy RL 里一个**动态加权的辅助目标**，两层控制：
  - **全局系数 \(\mu\)**（带衰减调度）：控制专家信号占多少比重，让训练从"以模仿为主"平滑过渡到"以探索为主"。统一损失写成 \(\mathcal{L}_{\text{Hybrid}}=(1-\mu)\mathcal{L}_{\text{GRPO}}+\mu\mathcal{L}_{\text{SFT}}\)。
  - **token 级权重 \(\phi(p)=p(1-p)\)**（\(p\) 是策略对该专家 token 的概率）：对"已经很可能（\(p\to1\)）"或"极不可能（\(p\to0\)）"的专家 token 都下调学习信号，只在 \(p\approx0.5\) 处学得最多【原文 §3.2-3.3, 式(3)(5)(6)】。
- 最巧的一步：**token 级 \(\phi(p)=p(1-p)\) 这条抛物线**。
  - 光有全局 \(\mu\) 不够——case study 显示 CHORD-µ 会让模型整体照搬专家的冗长风格、覆盖掉自身的简洁性（原文："\(\mu\) lacks fine-grained precision … forces the model to indiscriminately adopt expert patterns"）【原文 §3.2 L378-383】。
  - 直接用重要性采样（IS，权重 \(p.\text{detach}\)）又会让熵**急剧坍缩**（过度强化高概率 token、忽视低概率新 token → 过自信）；而完全不加 IS 则熵**暴涨**（established pattern 被破坏）【原文 §3.3 Fig.5, L423-431】。
  - \(\phi=p(1-p)\) 同时下调两端，把学习集中在"模型还不确定"的 token 上，制造一个"learning sweet spot"。
  - 抽掉 \(\phi\)：要么熵坍缩、要么熵暴涨，且模型被专家风格整体覆盖——CHORD-φ 的全面最优就垮了。

## 为什么做

> 一句话导读：SFT（模仿专家）和 RL（从自身反馈学）各有长短，传统"先 SFT 后 RL"是个生硬的二元开关，会经历漂移-再适应-过拟合，还不一定赢纯 RL。CHORD 想把这个开关换成连续衰减，再加 token 级精度。

- 研究背景：两大后训练范式各有性格。
  - SFT（off-policy）：靠高质量专家轨迹模仿响应模式，对数据质量/数量敏感，易过拟合、有 exposure bias（暴露偏差，训练时只见正确前缀、推理时却要面对自己生成的前缀），"SFT memorizes"、难泛化；
  - RL（on-policy）：从自己生成的直接反馈学、泛化好，但探索低效，易 entropy collapse（熵坍缩）或 over-exploitation（过度利用）【原文 §1 L32-44】。
  - 常规直觉是"SFT 学专家模式引导 RL 越过局部最优、RL 缓解 SFT 的 exposure bias"【§1 L45-49】。
- 解决的具体痛点，三条：
  - ①SFT-then-RL **不一定胜过纯 RL**（Fig.1，Qwen2.5-1.5B + Open-R1：40k/10k SFT+RL 都被 Pure RL 反超）【原文 §1 L99-101】；
  - ②SFT 训练曲线呈 **shift-readapt-overfit**（Fig.2，Qwen2.5-7B + DeepSeek-R1 专家，MATH-500），三阶段：
    - (a) **Policy Shift**——被迫跟随模式差异巨大的 off-policy 示范，established pattern 被破坏 + exposure bias 加剧 → 掉点；
    - (b) **Readapt**——逐渐整合专家模式、减少对自身模式的依赖 → 回升；
    - (c) **Overfit**——在有限静态专家数据上过拟合、丧失多样性与后续 RL 所需的探索能力【原文 §3.1 L219-232】；
  - ③SFT→RL 的转换时机高度任务相关（数学上 SFT-best+RL 最好、工具调用上 SFT-light+RL 更好），需大量调参，而且两阶段分离本身就次优【§4.2 L613-619】。
- 相关工作 & 各自不足【原文 §5 L883-896】：
  - **数据混合**：简单 dataset mixing（SimpleMix）/ 把专家轨迹混进 on-policy rollout 组（LUFFY、SRFT）；
  - **专家引导生成**：UFT、BREAD、Prefix-Sampling 用专家数据引导生成；
  - **交错 SFT/RL 步**：按预定/自适应调度（SASR）或对最难题（Ma 2025）交错；
  - **统一框架**：SRFT（Fu 2025）提出 sample-level SFT loss，把"数据混合 + SFT 损失"统一起来。
  - 作者强调本文聚焦"已经建立了自身回答模式的 **instruct 模型**"（原文："a more challenging yet practical scenario compared to existing works that finetune a base model"，如 LUFFY/SRFT 多在 base 上做）【§5 L893-896】。
- 动机链：SFT-then-RL 的二元开关（\(\mu=1\to0\)）太僵硬、会经历 shift-readapt-overfit → 改成可衰减的 \(\mu\) 调度，能在过拟合前平滑退出（受 scheduled sampling 启发，把"混专家与自生成数据"从 token 级推广到 loss 级）→ 但全局 \(\mu\) 缺精度（会整体照搬专家冗长）→ 需要 token 级 \(\phi\) 把专家信号集中在"模型不确定"的 token → \(\phi=p(1-p)\) 同时避熵坍缩（下调 \(p\to1\)）与避干扰（下调 \(p\to0\)）。为什么不用更简单做法：固定 \(\mu\) 一律差于动态 \(\mu\)（Fig.7）；纯 IS 熵坍缩、无 IS 熵暴涨（Fig.5）——所以既要衰减 \(\mu\)、又要 \(\phi\)。
- 与最近邻工作的Δ：
  - vs SFT-then-RL——把"独立阶段"变成"动态加权辅助目标"，\(\mu\) 连续衰减而非二元开关；
  - vs LUFFY/SRFT（把专家混进 rollout 组/重塑 IS 比）——CHORD 用 `expert_mask` 区分专家/非专家数据，专家走 SFT 损失、非专家走 GRPO，做**凸组合**；
  - vs SASR（概率交错 SFT/RL 步，按输出与专家相似度调焦点）——CHORD 是损失级凸组合 + token 级精度。
  - **关键有用点**：\(\phi\) 提供"选择性吸收"——按任务自适应吸收专家模式（Table 2 长度证据），而非无差别模仿。

## 怎么做 + 靠不靠谱

> 一句话导读：一个 mini-batch 里既有 RL 任务、也有专家示范，分别走 GRPO 损失和加权 SFT 损失，再按 \((1-\mu):\mu\) 合成。\(\phi=p(1-p)\) 这个 token 门控是关键。论文-代码高度一致、消融充分，主要短板是规模有限（最大 7B）。

- 方法流水线（读完可复现）【原文 §3, Fig.3】，五步：
  1. 一个 mini-batch 混合 RL 任务 prompt + 专家 \((x,y^*)\)；
  2. 对 RL 任务采 \(K\) 条 on-policy rollout、算 reward 与组归一化优势 \(A_k\)；
  3. 非专家数据走 GRPO 损失（PPO clipped surrogate），专家数据走 token 级 SFT 损失（\(-\phi(p)\cdot\log\pi\)）；
  4. 凸组合 \(\mathcal{L}=(1-\mu)\mathcal{L}_{\text{GRPO}}+\mu\mathcal{L}_{\text{SFT-}\phi}\)，\(\mu\) 随步数动态调度；
  5. 更新策略。
  - Fig.3 画得很清楚：专家分支算 `SUM(φ(·)*LogProbs)`，RL 分支算 `SUM(Advantages*LogProbs)`，再按 \((1-\mu):\mu\) 合成。
- 关键公式（真实形式 + 直觉）：
  - **SFT 损失**（式1，token 级 NLL）：
    \(\displaystyle \mathcal{L}_{\text{SFT}}(\theta)=-\frac{1}{\sum_{i=1}^B|y^*_i|}\sum_{i=1}^B\sum_{t=1}^{|y^*_i|}\log\pi_\theta(y^*_{i,t}\mid x_i,y^*_{i,<t})\)
  - **GRPO 损失**（式2，PPO clipped，**不含 KL 项**以免限制性能）：
    \(\displaystyle \mathcal{L}_{\text{GRPO}}(\theta)=-\frac{1}{\sum_{i}\sum_{k}|\tau_{i,k}|}\sum_{i}\sum_{k}\sum_{t}\min\bigl(r_{i,k,t}(\theta)A_{i,k},\;\text{clip}(r_{i,k,t}(\theta),1-\epsilon,1+\epsilon)A_{i,k}\bigr)\)
    其中优势 \(A_k=\dfrac{R(\tau_k)-\mu_R}{\sigma_R+\epsilon_z}\)（组内均值/标准差归一化），IS 比 \(r_{i,k,t}(\theta)\triangleq\dfrac{\pi_\theta(\tau_{i,k,t}\mid x,\tau_{i,k,<t})}{\pi_{\text{sample}}(\tau_{i,k,t}\mid x,\tau_{i,k,<t})}\)。strict on-policy（\(\pi_{\text{sample}}=\pi_\theta\)）时这个比恒为 1，梯度退化为 \(\nabla_\theta\log\pi_\theta(\tau^*_{i,k,t}\mid\cdot)\)【§2 L193-206】。
  - **统一损失**（式3，CHORD 的"全局控制"）：
    \(\displaystyle \mathcal{L}_{\text{Hybrid}}(\theta)=(1-\mu)\,\mathcal{L}_{\text{GRPO}}(\theta)+\mu\,\mathcal{L}_{\text{SFT}}(\theta),\quad\mu\in[0,1]\)
    SFT-then-RL = 二元调度（\(\mu\) 从 1 切到 0）的特例；交错 SFT/RL = 周期 \(\mu\) 调度的特例【§3.2 L308-313】。
  - **被否决的 IS 变体**（式4，给 \(\phi\) 当对照）：\(\mathcal{L}_{\text{SFT-IS}}=\mathbb{E}\bigl[-\sum_t\text{sg}\bigl(\tfrac{\pi_\theta(y^*_t\mid\cdot)}{\pi_{\text{sample}}(y^*_t\mid\cdot)}\bigr)\cdot\log\pi_\theta(y^*_t\mid\cdot)\bigr]\)，按惯例假设分母=1（把专家当 ground-truth 分布）。IS 会下调低概率 token 防破坏，但**激进强化高概率 token** → 熵坍缩 → 过自信、困在次优【§3.3 L398-431】。
  - **token 级权重**（式5，CHORD 的"细粒度控制"核心）：
    \(\displaystyle \phi(y^*_t;\pi_\theta)=p_t(1-p_t),\qquad p_t=\pi_\theta(y^*_t\mid x,y^*_{<t})\)
    一条抛物线，峰在 \(p_t=0.5\)、两端趋零。信息论上 \(p_t(1-p_t)\) 是"生成该 token 这一二元事件"的策略**不确定性**度量，偏向"模型最不确定"的 token，制造 learning sweet spot（原文："novel enough to be informative but not so divergent as to disrupt"——足够新到有信息量、又不至于离谱到破坏已有模式）【§3.3 L464-469】。
  - **最终 SFT 目标**（式6）：
    \(\displaystyle \mathcal{L}_{\text{SFT-}\phi}(\theta)=-\mathbb{E}_{(x,y^*)\sim D_{\text{SFT}}}\!\left[\sum_{t=1}^{|y^*|}\phi(y^*_t;\pi_\theta)\cdot\log\pi_\theta(y^*_t\mid x,y^*_{<t})\right]\)
    把式3 里的 \(\mathcal{L}_{\text{SFT}}\) 换成 \(\mathcal{L}_{\text{SFT-}\phi}\)，就是 CHORD 的最终目标（\(\mu\) 全局 + \(\phi\) token 级双控）。
- 逐组件必要性：
  - **全局 \(\mu\) 衰减**：负责"模仿→探索"的平滑过渡。**Fig.7 消融**——固定 \(\mu\) 一律差于动态 \(\mu\)、甚至可能不及纯 RL；小固定 \(\mu\)(0.02) 减损但提升不显著；大固定 \(\mu\)(0.1/0.5) 明显更差【§4.3 L735-749】。
  - **token 级 \(\phi=p(1-p)\)**：负责"防熵坍缩/干扰 + 选择性吸收"。**Fig.8/9**——CHORD-φ（固定 \(\mu=0.1\)）既防熵过早坍缩、又避免熵暴涨，reward 稳定上升、显著优于纯 RL；用了 \(\phi\) 后对 \(\mu\) 的选择变鲁棒（原文："a complex and decaying schedule for \(\mu\) is no longer essential"）【§4.3 L755-826】。**Table 2**——CHORD-φ 数学响应长度 2444、工具调用 120（vs CHORD-µ 被拉到专家级冗长 6081；专家 6132/原模型 659）。
  - **expert_mask 区分数据**：负责"哪条走 SFT、哪条走 GRPO"，是凸组合的前提。
  - **两个实例（CHORD-µ / CHORD-φ）**：CHORD-µ 只用全局 \(\mu\) 衰减调度（关掉 \(\phi\)）；CHORD-φ 固定 \(\mu\)（如 0.1）+ 启用 \(\phi\)。论文明示**不存在跨所有任务最优的单一 \(\phi\)**，但 \(p(1-p)\) 是"computationally efficient（只需 element-wise 乘已有 forward 概率）"且鲁棒的实例；附录 B.2 还试过 entropy-based / clipping / focal loss 变体【§4.3 L828-844】。
- 实验与证据【原文 §4】：
  - **数据集/设置**：
    - 数学——OpenR1-Math-220k（**5k SFT / 20k RL 无重叠**），policy=Qwen2.5-7B-Instruct（与专家 DeepSeek-R1 风格差异显著），评测 AIME24/25/AMC，MMLU-Pro 监控通用推理；
    - 工具调用——ToolAce 单轮（**5k RL / 500 SFT**，专家 DeepSeek-R1 同 system prompt 生成），policy=LLaMA3.2-3B-Instruct，评测 BFCL(Live/Non-live)【原文 §4.1 L478-490】。
  - **主表（Table 1）**：CHORD-µ 数学超强基线 SFT-best+RL（AMC 58.4→60.8 即 +2.4、AIME24 17.1→18.1 即 +1.0、AIME25 16.3→17.9 即 +1.6），工具调用整体也更好（BFCL Overall 77.6）；CHORD-φ 进一步在数学 + 工具调用**全面最优**（AMC 62.5、AIME24 18.2、**MMLU-Pro 56.2** 显著高于其余、BFCL Overall 78.5）。
  - **响应长度（Table 2）**：专家 DeepSeek-R1 远长于原模型（数学 6132 vs 659；工具 315 vs 147）；SFT 模型初期模仿冗长（SFT-light 数学 9966）；后续 RL 缩短（SFT-light+RL 1322 < SFT-best+RL 4830）；CHORD-µ 被拉到专家级冗长（6081），**CHORD-φ 取得更细的平衡（数学 2444、工具 120）**——token 级加权使模型按任务**选择性**吸收专家模式。
  - **进一步分析（§4.4）**，三点：
    1. 换专家源（DeepSeek-R1 vs 风格更近 LLaMA 的 Qwen2.5-72B），CHORD 均超基线，偏模仿的方法（SFT+RL/CHORD-µ）在专家风格相近时增益更大——印证"效果取决于专家质量 + 引入的 pattern shift 程度"；
    2. 扩展到非可验证域（RaR-Medicine 医疗 QA），CHORD-µ/φ 均显著超纯 RL，CHORD-φ 收敛更快；
    3. 弱模型（Qwen2.5-3B）上 naive 模仿/SFT+RL 不稳易崩，CHORD-φ 更鲁棒【§4.4 L847-880】。
  - **baseline 公平吗**：基线很全（Original / SFT-light / SFT-best / SFT-light+RL / SFT-best+RL / SASR / 纯 RL / LUFFY）。**诚实标注**：为公平，LUFFY 数学用 20k（原论文 45k，得 AMC 50.9/AIME24 17.7/AIME25 14.8）、工具调用用 500 SFT（原 5k）——透明，但也意味着 LUFFY 在此不是它的最佳配置【原文 §4.2 脚注1 L507-509】。
  - **看着强但没回答核心问题**：主实验规模有限（7B/3B/1.5B policy），更大模型 + 异构专家混合是 future work；模式漂移分析主要是经验性的，缺"不同 CoT 模式如何影响学习"的机理理论【§6 L908-918】。
- 假设与失效边界：
  - 【原文】聚焦 instruct 模型（已建立回答模式）；base 模型场景未必同理【§5 L893-896】。
  - 【原文 §3.3 L419-422】\(\mathcal{L}_{\text{SFT-IS}}\) 假设专家数据 IS 分母=1（把专家当 ground-truth 分布）——常见但是近似。
  - 【原文/推断】\(\mu,\phi\) 配置跨任务/数据/模型会变，无单一普适最优（作者明示）；自适应 reward-aware \(\mu=\max(0,\tau-\text{reward\_mean})\) 可行，但"requires heavy hyper-parameter tuning"【§4.3 L750-754】。
  - 【推断】\(\phi=p(1-p)\) 的"sweet spot"假设"中等概率 token = 最值得学的新信息"——若专家轨迹的关键信息恰好落在模型已高概率或极低概率的 token 上，\(\phi\) 会抑制它（这是失效边界）。
- 祛魅总结【推断】：真贡献是**"把 SFT 当 RL 动态加权辅助目标"这一统一视角 + 一个简单鲁棒实例（\(\mu\) 衰减 + \(\phi=p(1-p)\)）**，而非全新算法。论文-代码高度一致（损失公式、\(\mu\) schedule、\(\phi\)、expert_mask 均在 `chord_policy_loss.py` 复现），自洽性强，且诚实承认 \(\phi\) 非唯一最优、配置随 setup 变。被高估的可能是"统一视角"的新颖性（SRFT 等已有相关统一框架）；被低估的是 **shift-readapt-overfit 的系统刻画**（Fig.2）这一诊断本身的价值——它把"为什么 SFT-then-RL 次优"讲清楚了。规模有限是主要短板。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：on-policy 部分 = 可验证 reward 的 GRPO 优势（数学用分层 reward 兼顾正确性+格式）；off-policy 部分 = 专家示范的 token 级 NLL，经 \(\phi(p)=p(1-p)\) 重加权（只学中等概率 token）。
  - **改什么**：参数（策略 \(\pi_\theta\)）；通过凸组合损失同时受 RL 梯度与加权 SFT 梯度更新。
  - **何时改**：在线 per-step（RL 训练循环内，每步同时算 GRPO loss + SFT loss）；\(\mu\) 随训练步动态衰减。
  - **免梯度?**：否（全梯度）；\(\phi\) 权重用 sg/detach（\(p.\text{detach}\)），不反传到权重本身。
  - **记忆-技能生命周期**：专家数据集 \(D_{\text{SFT}}\) 是固定的 off-policy 示范池（写入：外部专家如 DeepSeek-R1 生成；检索：每 batch 按 `expert_data_ratio` 采样混入；无遗忘/共享/更新机制）。
  - **防遗忘机制**：\(\mu\) 衰减 = 防"过拟合专家"（在 overfit 前退出专家影响）；\(\phi\) = 防"established pattern 被破坏（熵暴涨）"与"过自信（熵坍缩）"——两者共同保护模型自身能力不被专家覆盖，可视为一种"防灾难性覆盖"机制。
- ⑦ 开源代码+框架/harness：https://github.com/modelscope/Trinity-RFT 。示例 `examples/mix_chord/`（`mix_chord.yaml` 数学、`mix_chord_toolace.yaml` 工具、`get_openr1_data.py`、README）；损失实现 `trinity/algorithm/policy_loss_fn/chord_policy_loss.py`（`MIXCHORDPolicyLossFn` + `SFTPhiLossFn`/`SFTISLossFn`/`SFTLossFn` + `mu_schedule_function`）；另有 ms-swift 集成【既有 analysis 代码核查】。框架 **Trinity-RFT（modelscope，基于 veRL 后端 + Ray，Pan 2025）**，算法注册名 `mix_chord`，用 `expert_mask` + `expert_data_ratio`（示例 0.20）区分专家/非专家。代码可得性好（损失实现 + 复现脚本 + 超参齐全）。
- 💰 资源/成本与可扩展性：相比纯 RL，额外成本主要是每 batch 多算一次 SFT loss（专家数据 forward），无额外采样。policy 规模 7B/3B/1.5B，仓库默认示例 Qwen2.5-1.5B-Instruct。超参（附录 A.1）：Adam（\(\beta_1=0.9,\beta_2=0.999\)）、lr ∈{1e-6,5e-6,1e-5}、温度 1.0、max response 16k、SFT≤3 epochs、strict on-policy \(K=8\) rollout/prompt；数学 batch SFT/RL=64/32、RL≤1500 步；工具 batch=96、RL≤100 步；**\(\mu\) 衰减 0.9→0.05**（A.1 称"over the first 30 steps"，§4.3 Fig.7 的 µ-only 消融称"over the first 200 steps"，二者是不同实验配置，复现以 repo 配置为准）。可扩展性：声称非可验证域（RaR-Medicine）、弱模型（3B）、异构专家均适用，但更大模型未验证（future work）。具体 GPU 数/wall-clock 正文未给（详见 Appendix）。
- 🎯 对"探索-巩固"对标：**强支撑 + 高度可借（L2 主线代表作）**。
  - CHORD 直击本项目"巩固/固化"环节的核心矛盾——**如何把外部/专家知识固化进参数而不破坏（遗忘）模型自身已有能力**。其"shift-readapt-overfit"诊断 = "巩固时若无差别覆盖会破坏 established pattern"，正是本项目"巩固且不遗忘"要规避的失败模式。
  - **与探索-巩固的精确对标**，三点：
    1. \(\mu\) 衰减 ≈ **脚手架撤除曲线**（从模仿专家 → 自主探索，与 CBRL 退火、TSRD"撤脚手架"同构）；
    2. \(\phi=p(1-p)\) ≈ **选择性巩固**——只在"模型不确定的 token"上吸收专家信号，与本项目"在关键步/高熵处接管"的 path-selection 思路高度同源（MEMORY 记录的"切高熵/低置信关键步"几乎是同一直觉）；
    3. expert_mask 凸组合 = on-policy 探索与 off-policy 巩固在**同一 batch 内共存**，而非分阶段——这对本项目"边探索边巩固"的设计极有参考价值。
  - **最可借组件**：\(\phi(p)=p(1-p)\) 的 token 级门控 + \(\mu\) 调度 + `chord_policy_loss.py` 的现成实现（`SFTPhiLossFn`/`mu_schedule_function`），可直接作为"巩固损失"的起点。
  - **Δ/缺口**：CHORD 的专家是静态外部示范池，无 path-recovery 的"走偏后自选恢复分支"、无 MTP 前瞻、无技能/记忆生命周期——它解决"如何吸收已有专家轨迹"，不解决"如何发现/生成要巩固的好行为"（那是探索侧）。
  - 一句判定：本项目"巩固/防遗忘"环节最直接可借的范式与代码来源，\(\phi+\mu\) 双控是现成模板；但需补上探索侧（脚手架生成、path-recovery、前瞻）。
- 🔭 开放问题/未来方向：
  - 【原文 §6 L908-918】①更稳的训练动力学算法改进、降 \(\mu/\phi\) 的超参依赖（更优的自适应调度函数）；②"不同 CoT 模式如何影响学习"的机理理论（当前漂移分析偏经验）；③扩展到**异构专家混合**，研究不同模型生成的推理模式如何涌现/演化。
  - 【推断】\(\phi\) 的设计空间未穷尽（论文也试过 entropy-based/clipping/focal 变体）——可探索"前瞻感知"的 token 门控（用 MTP 预测未来收益决定该 token 学多少），把 CHORD 的"当前不确定性"\(\phi\) 升级为"未来价值"\(\phi\)，直接对接本项目 MTP 前瞻；把静态专家池换成"模型自己探索出的成功轨迹"（self-distillation 闭环），让探索侧产出直接喂给 CHORD 巩固侧。

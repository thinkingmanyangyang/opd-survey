cbrl | Context Bootstrapped Reinforcement Learning (CBRL) | UC Santa Barbara + Cisco Research（Saaket Agashe、Jayanth Srinivasa、Gaowen Liu、Ramana Kompella、Xin Eric Wang） | 2026-03-19 · arXiv preprint v1 · cs.LG | 主题线 L3（RLVR/GRPO）兼 L6（探索引导）·相关性 高

**原始论文**：https://arxiv.org/abs/2603.18953 · 项目页 https://context-bootstrapped-rl.github.io

## 一眼看懂

> 一句话导读：RLVR 在陌生领域常常一条正确 rollout 都采不出来,直接卡死。CBRL 的招很朴素——训练早期把 few-shot 示范随机塞进 prompt 帮模型"蒙对",拿到学习信号后再把注入概率慢慢退火到 0,逼它自己内化。只动输入、不动 RL 本身。

- 🟦 TL;DR：RLVR（可验证奖励的强化学习）的死穴是"探索低效"。模型遇到新推理模式或陌生领域时,几乎采不出一条正确的 rollout;而只要一组 rollout 全错,GRPO 的组内优势就会坍缩为 0、没有梯度、训练卡在 reward 平台期【原文 §1 L23-27, §5.1 L468-471】。CBRL 的做法极简:把 few-shot 示范当成"临时脚手架"——训练早期以一个较高的概率（\(p_{\text{start}}\approx0.5\)）随机把示范前置到 prompt,帮模型蒙对、拿到学习信号;然后按一条线性课程,把注入概率退火到 0,逼模型把推理模式"内化"、而不是依赖示范【原文 §2, Algorithm 1】。它只改训练的**输入分布**,不动 RL 的目标/损失/优化器,所以是算法无关的、推理时零开销【原文 §2.4 L141-146】。
- 最巧的一步：**退火到 0 这一步**。如果一直注入示范（\(p\) 恒定不变）,模型会学成"有示范才会做题",到测试时（没有示范）就垮了;论文明说 "If examples appeared on every prompt, the policy might learn to depend on their presence rather than internalize the demonstrated reasoning patterns"【原文 §2.2 L99-101】。所以退火本身就是一条隐式课程,把"被引导 → 独立"压成一条单调下降的概率曲线。把退火抽掉,CBRL 就退化成普通的 few-shot 注入,"内化"这个主张也就没了核心支撑（Figure 4 那个"退火后不崩"的证据,也就无从谈起）。

## 为什么做

> 一句话导读：RLVR 一旦采不出正确 rollout 就零梯度、卡死。现有的三类救法（混 off-policy 轨迹、中途插真值片段、上课程）各有代价;CBRL 想用"prompt 前缀给整段示范 + 退火"同时绕开这三种代价。

- 研究背景：RLVR（可验证奖励的 RL,Lambert/Tülu-3 2024；DeepSeek-R1 2025）已是推理后训练的主流。二元的 verifier 奖励在数学/代码/工具调用上推动了显著进展;GRPO（Shao 2024）用组内归一化去掉了 value 网络,成了事实标准算法,还被证明能诱发"aha moment"式的自我反思;后续的 zero-RL 路线（SimpleRL-Zoo、DAPO、Liu 对 r1-zero 的批判）则干脆在 base 模型上直接跑 RLVR、不做 SFT 预热【原文 §1 L18-22, §5.1 L454-467】。
- 解决的具体痛点：当模型可靠地采不出正确 rollout 时,学习信号极弱（收敛慢、甚至训不动）。在"领域在预训练里欠表示"（比如 Q 这种小众编程语言）或"需要全新推理模式"时尤其严重——其机理论文写得很直白:**一组 rollout 全错 → 组优势 = 0 → 零梯度**【原文 §1 L23-27, §5.1 L468-471】。
- 相关工作 & 各自不足【原文 §5.2 L472-494】：把现有的"救活探索"路线分成三类——
  - (i) **mixed-policy training**（在 RL 中交错 SFT,或把一部分 on-policy rollout 替换成高质量的 off-policy 轨迹,如 Yan/LUFFY、Zhang）:LUFFY 用 "regularized importance sampling",在引入 off-policy 推理轨迹的同时保留自驱探索;但论文也指出 "over-reliance on off-policy data can misguide policy updates toward non-generalizable solution paths"。
  - (ii) **partial supervision / hints**（用 ground-truth 解的片段去救活失败的 rollout,如 HINT、GHPO、BREAD）:在 rollout **中途**插入真值片段,引导它沿正确轨迹走;有效,但仍属于 off-policy 干预。
  - (iii) **curriculum**（课程式,如 E2H Reasoner 从易到难、淡出简单题以防过拟合;Absolute Zero 用 code executor 自演化课程并自验答案）:需要难度估计或自生成题目等**外部基础设施**。
- 动机链：
  - RLVR 探索低效,零样本下无法 bootstrap（引导启动）出新模式,所以需要注入引导。
  - 但持续注入又会让模型产生依赖,所以需要靠退火把脚手架撤掉。
  - 为什么不用更简单的现成做法?因为三类各有代价——mixed-policy 会破坏 on-policy 的探索动力学,partial supervision 是中途干预,curriculum 要外部设施。
  - CBRL 用"ICL（上下文学习）当临时脚手架 + 一个线性退火"同时避开了这三者的代价,并且保持完全 on-policy（示范只进 prompt 上下文,不是要模仿的轨迹）【原文 §5.2 L487-494】。
- 与最近邻工作的 Δ（论文逐条对照）：
  - vs mixed-policy：CBRL **完全 on-policy**——"In-context examples serve as **context** rather than **trajectories to imitate**, preserving the exploration dynamics that enable robust generalization"【§5.2 L487-489】。
  - vs partial supervision：CBRL "does not intervene mid-rollout"（不在 rollout 中途干预）,只在 prompt 前缀给完整示范,靠原生的 ICL 从一开始就 bootstrap【§5.2 L489-491】。
  - vs curriculum：CBRL 在**固定分布**上跑,加一个单一退火就构成了"从被引导到独立"的隐式课程,不需要难度估计或题目生成【§5.2 L492-494】。
  - 关键有用点:它把"获取学习信号"和"破坏探索"解耦了——在前缀给整段示范,并不会污染 rollout 中段的 on-policy 性。

## 怎么做 + 靠不靠谱

> 一句话导读：三个组件——一个 few-shot 示范库、一个"掷硬币决定要不要塞示范"的随机注入、一条把注入概率从高降到 0 的线性退火。唯一的公式就是那条退火直线;RL 目标完全没动。

- 方法流水线（读完可复现）【原文 §2, Algorithm 1 L120-139】：三大组件 = ① few-shot 示范库 \(\mathcal{B}\)、② 随机注入机制、③ 退火课程。逐步如下:
  1. **构造示范库** \(\mathcal{B}\)：每条 \(e\in\mathcal{B}\) 是一个三元组 \(\langle q,r,a\rangle\)——含问题 \(q\)、可选的推理轨迹 \(r\)、答案 \(a\);来源可以是专家示范、更强模型的解、或手工构造【§2.1 L70-73】。具体到两个实验:Reasoning Gym 每任务 20 条,答案程序化求解、推理由 **GPT-5.2** 生成;Q 编程 50 条,**只含代码、没有推理注释**,且都是已验证的样本【§3.3 L225-232】。
  2. **每个训练步设定注入概率** \(p\leftarrow p_i\)（\(p_i\) 来自退火调度,公式见下一条）。
  3. **逐 prompt 注入**：对 batch 内每个 \(q_i\),先从 \(\mathcal{B}\) 采 \(k\) 条示范 \(E_i\leftarrow\text{Sample}(\mathcal{B},k)\),再掷一枚硬币 \(b_i\sim\text{Bernoulli}(p)\),最后 \(x_i\leftarrow\text{Compose}(b_i,E_i,q_i)\)。两种情形:
     - 若 \(b_i=1\):把 \(k\) 条示范作为**前置的 user-assistant 对话轮**,拼在目标 query 之前。Reasoning Gym 走 chat template（见附录 C.1.1 的 `[User]…[Assistant]<think>…</think><answer>…</answer>` 多轮格式）;Q 编程走 raw prompt,直接把示范题面拼接上去。
     - 若 \(b_i=0\):只给目标 query + 一个空的 assistant 轮【§2.2 L75-97, Alg.1 L5-10】。
  4. **rollout + 策略更新**：先 \(\mathcal{D}_t\leftarrow\text{RolloutBatch}(\pi_\theta,\{x_i\})\),再 \(\pi_\theta\leftarrow\text{PolicyUpdate}(\pi_\theta,\mathcal{D}_t)\)。注意**奖励只对生成的响应计算**——示范本身既不进 reward、也不进 loss【§2.2 L76-77, Alg.1 L11-13】。
  5. **推理时** \(p=0\)、不注入、零额外开销【§2 L67-68】。
- 关键公式（直觉 + 真实形式）：
  - **线性退火调度**（论文唯一的显式公式,eq.1）：
    \(\displaystyle p_i \;=\; p_{\text{start}} \;+\; \frac{t-1}{T-1}\,\bigl(p_{\text{end}}-p_{\text{start}}\bigr)\)
    其中 \(p_{\text{start}}\) 是初始注入概率（典型 0.5~1.0）,\(p_{\text{end}}\) 是终值（典型 0.0）,\(T\) 是总训练步数,\(t\) 是当前步【原文 §2.3 L103-107】。直觉很简单:就是一条从 \(p_{\text{start}}\) 线性降到 \(p_{\text{end}}\) 的直线;而且**退火速率会随 \(T\) 自动适配任意训练预算**（原文 "automatically adjusts to any training budget \(T\)"）【§2.3 L115-116】。
  - **底层 RL 目标完全没动**：论文不写 GRPO 公式,明确说 "without altering the underlying RL objective, loss functions, or optimization procedure"【§2.4 L141-143】。【推断】GRPO 的优势仍是组内归一化 \(\hat A_i=(r_i-\text{mean}(\mathbf r))/\text{std}(\mathbf r)\)，CBRL 只换了"产生 \(r_i\) 的那个 prompt 分布"——这正是它"算法无关"的技术根据。
  - 核心直觉一句话:示范给的不是"照抄的目标",而是"该怎么起步尝试"的引导,所以 on-policy 的探索动力学不被破坏。
- 逐组件必要性：
  - **stochastic 注入（Bernoulli 而非每条都给）**：负责"即便早期也强制独立尝试"，没它模型会学成依赖示范【§2.2 L98-101】。无单独消融，但理由论述清晰。
  - **退火课程**：负责"内化而非依赖"，有 Figure 4（三个 setting 下退火到 0 后性能不崩）+ Figure 3（注入概率消融）双重支撑【§4.2 L317-363】。
  - **初始注入概率 \(p_i\)**：Figure 3 在 ARC-1D（500 步）上做了消融——倒 V 形，\(p_i=0.5\) 峰值（success 0.31），\(p_i=0\)（0.26）、\(p_i=1.0\)（0.23）都更低；过高压制自身探索、过低脚手架不足【§4.2 L433-438】。**这是论文唯一的定量消融**。
  - **bank 采样策略（均匀 vs tag 过滤）**：Reasoning Gym 均匀采样；Q 编程**按 tag**（Array / Dynamic Programming 等）过滤出与当前训练题共享标签的子集后再采【§3.3 L227-232】。没做"均匀 vs 过滤"的对照消融。
- 实验与证据【原文 §4】：
  - **数据集/设置**：Reasoning Gym 5 任务（ARC-1D / Manipulate Matrix / Spell Backward / Word Sorting / Puzzle-24，程序化生成、可控 seed、无限训练数据），骨干 Qwen2.5-3B-Instruct（近零基线，受控）+ Llama-3.2-3B-Instruct（跨族验证）；Q Programming（Hogan 2025 的 678 题 LeetCode 式，542 训/136 测，含 array/search/sort/DP），骨干 **QQwen-7B-Pretrain**（在 Q 语料上预训过的 Qwen，起点更强但 RLVR 前 Q 能力仍弱）【§3.1-3.2 L153-211】。
  - **主结果（Table 1）**：全 10 个 model-environment 对都超 GRPO-only，增益 +1.3%（Spell Backward, Llama 95.67→97.00）~ +22.3%（Word Sorting, Qwen 53.33→75.67）。Qwen 最大增益 Word Sorting（+22.34）、Puzzle-24（48→60.67, +12.67）；Llama 在 ARC-1D（17→25, +8.0）、Manipulate Matrix（3.33→8.33, +5.0）。
  - **Q 编程（Table 2）**：Avg Pass 27.3→43.0%，Success(Pass@1) 5.0→26.3%。值得注意：CBRL 的 Valid Q 率**反低于** GRPO（80.9 vs 89.1）——论文解读：GRPO 主学"写合法 Q 语法"，CBRL 进一步学"用这些构造**真正解题**"【§4.1 L275-279】。
  - **算法无关（Table 3, RLOO）**：Word Sorting 20.33→67.33、Puzzle-24 23→66、Spell Backward 63.67→89.67（RLOO 增益甚至大于 GRPO，作者归因"RLOO's higher-variance gradient estimates"让早期更需引导）；但 ARC-1D（10.33→8.00, −2.3）、Manipulate Matrix（8.67→1.67, −7.0）**反而掉**【§4.2 L282-302】。
  - **训练动态（Figure 4）**：三 setting（Q-GRPO / Word Sorting-GRPO / Word Sorting-RLOO）CBRL 早期 mean reward 显著更高（阴影区 \(p_i>0.25\) 高注入期），退火到 0 后性能不崩【§4.2 L317-363】。
  - **定性（Figure 5）**：CBRL 模型在 Word Sorting 上**显式检索 ASCII 码**（'v'=118、'a'=97）逐字符比较、排序、映射回单词得正确答案，紧贴 few-shot 示范的 step-by-step ASCII 推理范式；baseline 只泛泛提"letter cases"不算码点、给错序【§4.2 L439-451】。
  - **baseline 公平吗**：同骨干、同步数、同超参对照（Table 1 含 zero-shot baseline + few-shot prompting baseline + GRPO + CBRL-GRPO），公平。但只跟"无引导 GRPO/RLOO"比，**未与 LUFFY/HINT/BREAD 等同类引导方法做直接对照**——这是缺口。
  - **看着强但没回答核心问题**：主张"内化"但只给行为层证据（退火后不崩 + 定性 Figure 5 模仿示范的 ASCII 推理），**未给参数/表征层的内化证明**。
- 假设与失效边界：
  - 【原文】RLOO 下 ARC-1D/Manipulate Matrix 退化，作者明说"the effectiveness of CBRL depends on the **match between selected context and task structure**"【§4.2 L314-316】——脚手架与任务结构需匹配，否则有害；task-aware context selection 列为 future work。
  - 【推断】依赖"有合理的 few-shot bank"——bank 质量/覆盖度差时无效（论文用程序化求解或验证过的代码示例，质量有保证）。
  - 【推断】增益对"GRPO 自身已能探索"的任务收益小（Spell Backward Llama 仅 +1.3%）；对零基线任务（Qwen Word Sorting near-zero baseline）收益最大——这是"救活探索"而非"普涨"的方法。
- 祛魅总结【推断】：
  - 真贡献是"用 ICL 当临时脚手架 + 退火内化"这个组合,再加上比较系统的验证（2 个模型族 × 5 个任务 × 2 个算法,外加一个真实 DSL）;它不是算法创新——本质上就是带退火课程的 few-shot prompt 注入,完全不碰 RL loss。
  - 它**高估了"内化"**:只有行为层的证据。
  - 但它也**诚实地暴露了 RLOO 两个任务上的退化**（这印证了"脚手架要和任务结构匹配"的假设）。
  - 最大的短板是评测面窄——只有合成任务 + 单一 DSL,没有 AIME/MATH/LiveCodeBench 这类基准,所以规模化和泛化的证据都有限。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：可验证二元奖励（Reasoning Gym：format +0.2 / answer +1.0；Q 编程：base = \(\frac{\text{passed tests}}{\text{total tests}}\times w\)，全过 +2.0 bonus，每题≤5 测例、10s 超时）【原文 §3.3 L215-219, Table 13 L811-823】；示范本身不进 loss，只改输入分布。
  - **改什么**：参数（策略 \(\pi_\theta\) 的权重，经 GRPO/RLOO 更新）；**训练时临时改 prompt**（前置示范），但这是临时的、退火到 0。
  - **何时改**：在线 per-step（每训练步做策略更新）；注入概率随步数线性退火。
  - **免梯度?**：否（标准 policy-gradient RL）。
  - **记忆-技能生命周期**：有一个外部 few-shot bank \(\mathcal{B}\)（写入：专家/更强模型/手工构造；检索：均匀采样或按 tag 过滤；**无遗忘/共享/更新机制**——bank 固定）。这是"非参数上下文"而非可演化记忆。
  - **防遗忘机制**：无显式防遗忘；但"退火"本身可视为防"过拟合到示范依赖"的机制（类比 scheduled sampling 缓解 exposure bias）。
- ⑦ 开源代码+框架/harness：https://github.com/context-bootstrapped-rl/cbrl （项目页 https://context-bootstrapped-rl.github.io；既有记录已 clone 约 1.3MB 研究原型）。框架 **verl + vLLM + PyTorch + FSDP**，Hydra 配置；自带 `cbrl/trainers`【既有 analysis「元信息」+ §3.3「FSDP/TP=4」佐证 verl 栈】。【待核】既有 v2 记的精确版本号（verl 0.3.0.post2 / vLLM 0.8.5 / PyTorch 2.6.0）正文未列，以 repo 配置为准。
- 💰 资源/成本与可扩展性：Reasoning Gym——500 步、batch 32、mini-batch 16、micro-batch 4/GPU、lr \(1\times10^{-6}\)、无 warmup、grad clip 1.0、PPO epochs 1、clip ratio \(\epsilon=0.2\)、entropy coef 0.001、KL coef 0.001（`low_var_kl`）、温度 1.0/top-p 1.0、**4×A6000 + TP=4 + FSDP**；CBRL 注入 \(p_{\text{start}}=0.5\to p_{\text{end}}=0\)、\(k=2\) 示范【原文 Table 4/5/8】。Q 编程——batch 64、group/n=8、64 epochs=512 步、entropy coef 0.0、**4×GH200 + TP=4 + FSDP**【Table 10-13】。评测温度 0.6/top-p 0.9、bf16、100 题×3 次（Q 编程×5 次）。**推理零额外开销**（测试时 \(p=0\) 不注入）是核心卖点【§2 L67-68】。可扩展性：声称算法无关、可与其他增强组合；但只在 3B/7B 验证，大模型 + 长程/agentic 设置列为 future work【§7】。
- 🎯 对"探索-巩固"对标：**支撑（prompt 层面的极简对照）**。CBRL 可以理解为"教师脚手架（用 few-shot 示范当稀疏脚手架）+ 后撤（退火）"的最朴素实现,与 TSRD 那套"探索/选路 + 走偏后引导,再撤除脚手架"的直觉高度同构。
  - **关键 Δ**:CBRL 的脚手架是"把整段示范放在 prompt 前缀"——非参数、上下文级,既不触及 per-step 的 token 信用,也没有 path-recovery 那种"走偏后单点接管"的机制。换句话说,它是"开头给引导",而不是"中途纠偏"。
  - 可借组件:**退火调度 \(p_{\text{start}}\to 0\)（eq.1）可以直接当作"脚手架撤除曲线"的现成模板**;而且作者自己也把 "longer-horizon/multi-step/agentic" 列为 future work【§7 L516-518】,这正好是本项目方向上的缺口。
  - 一句判定:概念同源、但机制最浅（它在 prompt 层,本项目要的是参数/token 层）,所以它是一个优秀的"极简基线 / 直觉锚点",而非直接竞品。
- 🔭 开放问题/未来方向：
  - 【原文 §7 L510-519】①注入调度自适应化（按 reward 趋势/成功率调 \(p_i\) 而非固定线性）；②原则化的示范构造/选择（学习式检索自动对齐示范与训练实例）；③扩展到 longer-horizon（多步推理链 + agentic workflow，应对"早期失误→无法发现正确路径"的复合探索难题）；④CBRL × model scale 的交互、与 off-policy 学习组合。
  - 【推断】"内化"的表征层证据缺失——可探测注入退火后模型内部是否真的"编码"了示范模式（probing/representation analysis）；RLOO 两任务退化提示需要"task-aware context selection"，这与本项目 path-selection 直接相关。

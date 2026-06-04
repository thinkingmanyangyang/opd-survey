behavior_priming | Beneficial Reasoning Behaviors in Agentic Search and Effective Post-training to Obtain Them | CMU LTI（Jiahe Jin、Abhijay Paladugu、Chenyan Xiong）| 2026-01（arXiv:2510.06534v3，cs.AI）| 主题线 L4 Agent/工具/多轮自进化·相关性 高

**原始论文**：https://arxiv.org/abs/2510.06534

## 一眼看懂

> 一句话导读：搜索 agent 能不能靠 RL 训好,关键看 base 模型有没有几种特定推理行为。本文先把这些行为找出来,再用 SFT"种"进模型;最惊人的发现是——种行为时答案对不对都无所谓,RL 后效果一样好。

- 🟦 TL;DR：搜索 agent（指多轮检索、边搜边答的 LLM agent）要靠 RL 训得好,前提是 base 模型本身已经具备某些推理行为。本文分两步。第一步,用一条 **LLM pipeline 从"强模型成功 vs 弱模型失败"的轨迹里,提炼出四种有益行为**:Information Verification（信息验证）、Authority Evaluation（权威评估）、Adaptive Search（自适应搜索）、Error Recovery（纠错回轨）。第二步,用 **Behavior Priming**——先 SFT 把这四种行为"种"进模型,再跑 GRPO 强化。
- 最炸的发现:用"展现了这四种行为、但**最终答案是错的**"的轨迹去做 SFT,RL 之后的效果**约等于**用答案正确的轨迹（Table 4,RL 后 Overall 23.5 vs 23.6）。这说明**解锁 RL 的关键是推理行为（path）本身,而不是结果（outcome）对不对**。
- 最巧的一步：**Behavior Prime (Incorrect) vs (Correct) 的对照消融（§5.2 Table 4）**。这两组 SFT 数据,行为频率都是 100%,但 accuracy 一组 0%、一组 100%——也就是只改变 outcome 这一个变量。把这个干净的因果隔离实验抽掉,"行为 > 正确性"就只能算相关、而非因果结论;正是它把整篇的核心主张钉死。

## 为什么做

> 一句话导读：数学/代码领域早就发现"RL 能不能成,取决于 base 模型已有的推理模式";但这个结论几乎只在数学上验证过。本文想把它搬到搜索 agent,并搞清楚到底该种哪些行为。

- 研究背景：Agentic search 指 LLM 用多步检索来解决复杂信息需求——把任务分解、多步去搜、再综合（引 Jin 2025a、Zheng 2025）。它商用已经起飞（ChatGPT Deep Research、Google AI Mode）,开源侧靠 RL 训练进展也快（Search-R1、R1-searcher、DeepResearcher）。范式分两类（§2）:① 多 agent 系统,走预定义 workflow（Open Deep Search、WebThinker）;② **单 agent 端到端**,更容易端到端训练,本文聚焦这一类。训练手段则有三种:SFT 合成轨迹蒸馏（Self-RAG、Chain-of-Retrieval）、online RL、以及 **SFT-then-RL 两段式初始化**（WebSailor、RAG-R1）。
- 解决的具体痛点：在数学/代码上,大家已经发现"**RL 能否成功,高度依赖 base 模型是否已有特定推理模式**"——比如 verification（自我验证）、backtracking（回溯）这类行为（引 Gandhi 2025 cognitive behaviors、Yeo 2025 demystify long-CoT、Liu 2025）。问题是,这些研究**几乎只针对数学**。对 agentic search 来说,"哪些具体行为有益、又该如何系统培养",都不清楚;何况 agentic search 还有自己的独特挑战:要在海量检索结果里识别有用信息、要解决来源之间的冲突、要在长轨迹中始终保持目标聚焦（§1）。一个关键观察是:**没被 prime 过的模型,在 RL 过程中根本长不出这些行为**（§7 Table 6:Direct RL 之后 Authority Eval −8.3、Error Recovery −28.5,多数行为的频率反而是降的）。
- 相关工作 & 各自不足（§2）：
  - **多 agent vs 单 agent**——后者更易端到端训,本文选后者。
  - **SFT 合成轨迹蒸馏（Asai 2023 Self-RAG、Wang 2025b CoRAG）/ online RL（Search-R1、DeepResearcher）/ SFT-then-RL 初始化（Li 2025a、Tan 2025）**——但**没人系统研究 agentic search 该种入哪些行为**。
  - **"RL 成功取决于 base 已有行为"（Gandhi 2025 / Yeo 2025 / Liu 2025）**——集中在数学,**agentic 域空白**。
- 动机链：
  - 起点是现状:agentic search 靠 RL,但成功与否取决于 base 已有的行为,而这个结论目前只在数学上有。
  - 由此暴露两个缺陷:一是不知道 agentic 到底该种哪些行为;二是没被 prime 的模型在 RL 里自己长不出来。
  - 所以路线定为:先"识别有益行为",再"用 SFT 种进去 + RL 精炼"。
  - 为什么不干脆用 process reward（过程奖励）引导 RL 自己长？§7 试过——奖励设为 \(R=R_{\text{outcome}}+0.1\times N\)，其中 \(N\) 是轨迹里展现的行为种数。结果行为频率确实大涨（Info Verif +39.0、Adaptive +64.6）,但**性能反而比标准 RL 还差**。原因是模型在 **reward hack**:只去模仿行为的表面模式、没掌握其精髓（这和 DeepSeek-R1 在数学/代码上的观察一致）。
- 与最近邻工作的 Δ：
  - vs 数学域的 Gandhi 2025（"cognitive behaviors enable self-improving reasoners"）——本文把"base 行为决定 RL 上限"这个结论从数学**迁移并系统化到 agentic search**,还新增了"错误轨迹消融"来证明起作用的是行为、而非正确性。
  - vs 普通 SFT-then-RL（其中 Distillation 是随机采样轨迹、Outcome-driven 只筛答对的轨迹）——本文改按"是否展现全部四种行为"来筛,并证明这样筛**超过了专门筛正确轨迹的 Outcome-driven**。
  - 关键点:**SFT 阶段真正该优化的目标,不是"答对",而是"展现好行为"**。

## 怎么做 + 靠不靠谱

> 一句话导读：方法分两部分——先用一条"对比强弱轨迹 → 抽行为 → 合并"的 LLM 流水线把四种有益行为找出来;再用 SFT 把它们种进模型、用 GRPO 精炼。注意 RL 阶段仍是 outcome 级奖励,这与"行为重要"的主张存在一处内部张力。

**【第一部分:识别行为（§3）——可复现的方法论】**
1. **标准 agentic search 框架（§3.1）**：这是个迭代过程。第 \(k\) 步给定历史 \(\mathrm{ctx}_k\)，模型输出 \(y_k=\langle t_k,a_k\rangle\)——其中 \(t_k\) 是 reasoning（思考）、\(a_k\) 是 action（动作）。动作只有三种:search（检索）、answer（终止给答案）、**summary（把历史压缩一下,用来管理上下文）**。历史的更新规则是:
   \(\displaystyle \mathrm{ctx}_{k+1}=\begin{cases}\mathrm{ctx}_k+y_k+\mathrm{info}_k, & a_k=\text{search}\\ a_k, & a_k=\text{summary}\end{cases}\)
   读法:走 search 时,把这一步的输出和检索结果 \(\mathrm{info}_k\) 追加到历史后面;走 summary 时,把累积的历史整个替换成压缩版,好让有限上下文的模型也能撑住。完整 prompt 见 Appendix B.1。
2. **配对轨迹**：取 **500 题**,这些题都是 Gemini 2.5 Flash（强模型）能做对、而 Qwen3-1.7B（弱模型）做错的;于是每题就有一对——强模型的正确轨迹 vs 弱模型的错误轨迹。数据来自 Li 2025c（Chain-of-Agents）的 web agent SFT 集（后面造语料也用同一来源）。
3. **三阶段 LLM pipeline（§3.2,受 AutoRule-Wang&Xiong 2025 启发）**：依次是——
   - ① **轨迹比较**:让 Gemini 2.5 Flash 详细分析"为什么一条成功、一条失败"。
   - ② **行为抽取**:让 Gemini 2.5 Flash 从中提炼出贡献了成功的关键行为。
   - ③ **行为合并**:让 Gemini 2.5 **Pro** 去重、合并相似项、只留下普适的,再加人工复核。
   - 最终得到**四种行为**:
   - **Information Verification**:跨多源验证检索结果、reasoning 中引证。
   - **Authority Evaluation**:识别来源冲突、评可信度、优先权威信息。
   - **Adaptive Search**:据前次结果动态改搜索策略。
   - **Error Recovery**:识别并纠正先前步的错误（≈path-recovery/回轨）。
4. **跨模型验证普适性**：在 Gemini 2.5 Flash、DeepSeek-R1、Llama3.2-3B、Qwen3-1.7B 上跑三个 web benchmark,结果**行为频率与性能强正相关**（Fig.2;这里"频率"指展现该行为的轨迹占比）。

**【第二部分:Behavior Priming 训练（§4）——读完可复现】**
5. **SFT 种行为（§4.1）**：对 web/QA 两类任务,各从 Gemini 2.5 Flash **每题采 10 条**轨迹（问/答仍来自 Li 2025c）。然后用 LLM 判断每条是否**同时展现了全部四种行为**,只保留全展现的。训练样本的构造是**把轨迹的每一步当成独立样本**:
   \(\displaystyle \mathcal D_{\text{SFT}}=\{\langle x^i_k,y^i_k\rangle\mid 1\le k\le L_i\},\quad T_i=(\langle x^i_1,y^i_1\rangle,\dots,\langle x^i_{L_i},y^i_{L_i}\rangle).\)
   数据统计见 Table 1:web 侧 Behavior Prime 有 2.9k 条轨迹、平均 6.8 步,合计 20k 个 step 样本（accuracy 49.8%）;对照的 Incorrect 2.6k 条×7.6 步（acc 0%）、Correct 3.4k 条×5.9 步（acc 100%）;QA 侧 2.2k 条×4.6 步 = 10k。注意一个细节:Behavior Prime 的轨迹**更长**（步数更多,意味着探索更充分）。
6. **RL 精炼（§4.2）**：在 primed 模型上跑 **GRPO**,并**按 step 聚合**做更新:
   \(\displaystyle J_{\text{GRPO}}(\theta)=\mathbb E_{q\sim\mathcal D,\{T_i\}_{i=1}^G\sim\pi_{\theta_{\text{old}}}}\Big[\sum_{i=1}^G\sum_{k=1}^{L_i}\sum_{t=1}^{|y_k|}\frac{1}{|y_k|}\min\big(r_{i,k,t}(\theta)\hat A_i,\ \mathrm{clip}(r_{i,k,t}(\theta),1\pm\varepsilon)\hat A_i\big)\Big],\)
   其中 \(r_{i,k,t}\) 是 importance ratio（重要性比值）。奖励是 **outcome 二值奖励**:LLM-judge 判最终答案对/错,给 \(R_i\in\{0,1\}\)。关键是 **\(R_i\) 和优势 \(\hat A_i\) 对一条轨迹内的所有 step/token 都是恒定的**——这正是与"path 重要"主张之间的内部张力所在:SFT 阶段在喂 path,RL 阶段却退回了 outcome 级。
- **逐组件必要性（消融较完整）**：
  - **四行为 vs 单行为（§5.4 Table 5）**——只筛 Information Verification 的 IV-Only-10k 比 Direct RL 好（Overall 17.4 vs 13.9）,但**始终被四种行为齐全的 Behavior Prime-10k 超过（19.7）**。这说明复合行为之间有协同,缺一不可。
  - **SFT 种行为 vs RL 自己长（§7 Table 6）**——Direct RL 之后,行为频率大多是降的,证明确实需要先显式给一个行为先验。
  - **行为 vs outcome（§5.2 Table 4）**——这是核心消融（见 TL;DR）:SFT 之后 Incorrect 9.3 < Correct 11.3,差距明显;但 **RL 之后两者被抹平到 23.5≈23.6**。
  - **SFT 数据规模（§5.3 Fig.3）**——从 5k 加到 10k 再到 20k,但 WebWalkerQA 在 **5k 之后就 plateau（饱和）了**:行为一旦学会,继续 priming 的收益就递减。
  - **process reward 替代（§7 Table 6）**——这条失败了:行为频率大涨,性能却反降（典型的 reward hack）。
- **关键机制/公式（直觉）**：本文**没有引入新损失**,用的是 GRPO 标准目标。机制解释主要靠**训练动态**的几个观察:
  - SFT 之后,行为频率、Pass@8、平均步数都上升（Fig.1/Fig.4,且 Behavior Prime 的涨幅明显大于 Distillation）。这带来更丰富的探索和更多 search 动作,也就为 test-time scaling 打下基础。
  - RL 过程中,**primed 模型能维持更高的 policy entropy（策略熵）、不会早早崩掉**（Fig.5a）;而 Direct RL 的熵骤降、很早就收敛到一个低 plateau（Fig.5b:primed 收敛更慢,但天花板更高）。一句话:更高的熵 = 更丰富的探索 = 更高的 RL 上限。
  - 还有一个反直觉的发现（Fig.5c）:Direct RL 起步时 valid-action-ratio（合法动作比例）很低,但 20 步内就学会格式并稳住;primed 模型的 valid-action-ratio 反而**不太稳**。这恰好说明增益来自**推理行为**,而不是对工具语法的熟悉。
- **实验与证据**：backbone 是 Qwen3-1.7B 和 Llama3.2-3B-Instruct;评测包含 3 个 web benchmark（GAIA 用 103 个文本例、WebWalkerQA、HLE）+ 7 个 multi-hop QA（NQ/TQ/HotpotQA/2Wiki/MuSiQue/Bamboogle/PopQA）。
  - **主表 Table 2/3**：相对 Direct RL,web 上**相对提升 +37.2%**、QA 上 **+6.2%**;两个 SFT-then-RL 基线（Distillation/Outcome-driven）都被一致超过;并且比肩或超过 Search-R1、R1-Searcher、DeepResearcher 等 prior work（Qwen3-1.7B QA Overall 65.5 vs DeepResearcher 61.8）。
  - baseline 是否公平:两个 SFT-then-RL 基线都从**同一份 Gemini 语料**里选轨迹,只是选择策略不同,所以是公平的。
- **假设与失效边界**：
  - 【原文】(§A) SFT 跑 3 epoch、batch size 8;RL 在 **verl-agent（Feng 2025 GiGPO 系）** 上跑 300 步、bs32、GRPO group 8,耗时 **8×H100 约 20h**;评测时 temp 0、pass@k 用 temp1,打分用 **GPT-4o-mini**;web 轨迹最多 25 步、QA 最多 15 步。
  - 【推断】几个限制:① **模型规模偏小**（1.7B/3B）,更大模型是否仍需显式先验,未知。② 四种行为是由 **LLM-judge 识别和筛选**的,所以整套结论都依赖判别 prompt 的质量、以及 Gemini 判断的一致性——这里有个**循环依赖**:用 LLM 来定义什么是"好行为",又用 LLM 来评判是否展现了这些行为。③ web 增益大(+37%)、QA 温和(+6.2%),说明方法对任务类型相当敏感。
- **祛魅总结**：
  - 【推断】真贡献有三块:一是为 **"path 监督 > outcome 监督"这个可迁移结论提供了强因果证据**（错误轨迹消融做得干净）;二是把数学域的发现系统地迁移到 agentic search;三是 process-reward 失败给出的反面教训。
  - 被包装、或者说存在张力的地方是:**SFT 阶段明明是 path 监督,RL 阶段却又退回了 outcome 级 GRPO**——\(R_i,\hat A_i\) 在全轨迹恒定,没有任何 step/turn 级信用分配。这与"path 重要"的主张有内部张力:它只在 SFT 喂 path,RL 仍靠 outcome reward。另外,"行为定义"高度依赖特定的 LLM-judge,可迁移性也没充分检验。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=四种推理行为的轨迹示范（SFT,LLM-judge 筛"全展现"）+ outcome 二值奖励（RL,\(R_i\in\{0,1\}\)）｜**改什么**=参数（SFT 全量微调 + RL 更新策略）｜**何时改**=离线两段式（先 SFT 种行为 → 再 GRPO 在线精炼）｜**免梯度?**=否｜**记忆-技能生命周期**=部分相关——"行为"≈可复用技能,但只写入参数,无外部技能库/检索/共享;summary action 是轻量上下文管理（非持久记忆）｜**防遗忘机制**=无显式机制（SFT→RL 顺序,未处理灾难性遗忘）
- ⑦ 开源代码+框架/harness：主仓库 https://github.com/cxcscmu/Behavior-Priming-for-Agentic-Search （旧 analysis 已核:旧 KEYS 的 `cxcscmu/Behavior-Priming` 是 404,此为正确仓库名）。**SFT 框架=LLaMA-Factory**（vendored,`scripts/train.sh`→`llamafactory-cli train`）;agent rollout/评测靠 vLLM;**RL 框架=GRPO,在独立仓库** https://github.com/cxcscmu/verl-agent-deepresearch （veRL 系,verl-agent/GiGPO 的 deep-research 变体）,主仓库**不含 RL 训练代码**。harness=自建 agentic search 框架（search/answer/summary 三动作,Appendix B 给完整 prompt,含 internal-thinking 与无 thinking 两版）。【待核】框架细节复用旧 analysis 核查,本次基于 PDF 重写未重新进仓核对。
- 💰 资源/成本与可扩展性：【原文】RL 单次运行 8×H100 约 20h;SFT 数据 web 20k steps / QA 10k steps（步级样本）;每题采 10 条 Gemini 轨迹做语料。
- 🎯 对"探索-巩固"对标：**强支撑（核心对标论文之一）**。判定:本文是 TSRD"teacher 当稀疏脚手架 → student 内化 → RL"思路的**直接经验先例**。几个对应关系:
  - 四种行为里,**Error Recovery（识别并纠正先前错误）几乎就是 path-recovery / 回轨**,Adaptive Search 约等于选路调整。
  - "path 监督 > outcome 监督"为 TSRD"教选路 + 回轨比教对错更重要"提供了**最直接的因果证据**（Table 4）。
  - "primed 模型在 RL 中维持更高熵 = 更好探索"（Fig.5a）支撑了"巩固有效行为后探索空间更大"这一直觉。
  - **可借组件**:① 用强弱轨迹对比 + LLM pipeline 自动提炼"有益行为/path"的方法论（§3.2 三阶段）;② 把单步当独立样本的 step 级 SFT 构造（\(\mathcal D_{\text{SFT}}\)）。
  - **缺口/竞品面**:RL 仍是 outcome 级、没有 step 信用分配（TSRD 要的是更细的 path-recovery 单点接管）;行为靠 LLM-judge 识别,而非用 logit/MTP 做前瞻;也没有 teacher 在线脚手架（这里 teacher 只在离线造 SFT 数据）。
- 🔭 开放问题/未来方向：【原文】"targeted post-training 增强推理行为以提升 RL 有效性"是有前景方向（§8）;探讨过 process reward 作替代但失败（§7）。【推断】更大模型上的必要性;行为识别去 LLM-judge 化（用更客观信号,如 logit/熵）;把 path 监督贯穿到 RL 阶段（step/turn 级信用分配）而非 SFT 后退回 outcome;teacher 在线脚手架而非仅离线造数据;与 MTP 前瞻结合识别"该回轨"的时点。

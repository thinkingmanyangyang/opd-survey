behavior_priming | Beneficial Reasoning Behaviors in Agentic Search and Effective Post-training to Obtain Them | CMU LTI（Jiahe Jin、Abhijay Paladugu、Chenyan Xiong）| 2026-01（arXiv:2510.06534v3，cs.AI）| 主题线 L4 Agent/工具/多轮自进化·相关性 高

**原始论文**：https://arxiv.org/abs/2510.06534

## 一眼看懂
- 🟦 TL;DR：搜索 agent（多轮检索答题）要 RL 训得好,得看 base 模型有没有特定推理行为。本文先用 **LLM pipeline 从"强模型成功 vs 弱模型失败"轨迹里提炼出四种有益行为**（Information Verification 信息验证 / Authority Evaluation 权威评估 / Adaptive Search 自适应搜索 / Error Recovery 纠错回轨）,再用 **Behavior Priming**——先 SFT 把这四种行为"种"进模型、再跑 GRPO。最炸的发现:用"展现这四种行为但**答案错误**"的轨迹做 SFT,RL 后效果 **≈** 用答案正确的轨迹（Table 4,RL 后 Overall 23.5 vs 23.6）,说明**解锁 RL 的关键是推理行为(path),不是 outcome 正确性**。
- 最巧的一步：**Behavior Prime (Incorrect) vs (Correct) 的对照消融（§5.2 Table 4）**——两组 SFT 数据行为频率都 100%、accuracy 分别 0%/100%,只变 outcome。抽掉这个干净的因果隔离实验,"行为>正确性"就只是相关而非因果论断;正是它把整篇核心主张钉死。

## 为什么做
- 研究背景：Agentic search（LLM 多步检索解复杂信息需求:分解任务→多步搜→综合,引 Jin 2025a、Zheng 2025）商用已起飞（ChatGPT Deep Research、Google AI Mode）,开源侧靠 RL 训练进展快（Search-R1、R1-searcher、DeepResearcher）。范式分两类（§2）:① 多 agent 系统（预定义 workflow,Open Deep Search、WebThinker）;② **单 agent 端到端**（更易端到端训,本文聚焦）。训练手段:SFT 合成轨迹蒸馏（Self-RAG、Chain-of-Retrieval）/ online RL / **SFT-then-RL 两段式初始化**（WebSailor、RAG-R1）。
- 解决的具体痛点：数学/代码已发现"**RL 成功高度依赖 base 是否已有特定推理模式**"（verification/backtracking,引 Gandhi 2025 cognitive behaviors / Yeo 2025 demystify long-CoT / Liu 2025），但**几乎只研究数学**。对 agentic search,"哪些具体行为有益、如何系统培养"不清楚;且 agentic search 有独特挑战（海量结果中识别有用信息、解决来源冲突、长轨迹保持目标聚焦,§1）。关键观察:**non-primed 模型 RL 中无法内生长出这些行为**（§7 Table 6:Direct RL 后 Authority Eval −8.3、Error Recovery −28.5,多数行为频率反降）。
- 相关工作 & 各自不足（§2）：
  - **多 agent vs 单 agent**——后者更易端到端训,本文选后者。
  - **SFT 合成轨迹蒸馏（Asai 2023 Self-RAG、Wang 2025b CoRAG）/ online RL（Search-R1、DeepResearcher）/ SFT-then-RL 初始化（Li 2025a、Tan 2025）**——但**没人系统研究 agentic search 该种入哪些行为**。
  - **"RL 成功取决于 base 已有行为"（Gandhi 2025 / Yeo 2025 / Liu 2025）**——集中在数学,**agentic 域空白**。
- 动机链：现状（agentic search 靠 RL 但成功取决于 base 行为,且只在数学有结论）→ 缺陷（不知 agentic 该种什么行为 + non-primed 模型 RL 自己长不出来）→ 先"识别有益行为"再"SFT 种进去 + RL 精炼"。为什么不用 process reward 引导 RL 自己长？§7 试过（\(R=R_{\text{outcome}}+0.1\times N\)，\(N\)=轨迹中展现的行为种数）——行为频率确实大涨（Info Verif +39.0、Adaptive +64.6）但**性能反比标准 RL 差**,因为模型 **reward hack** 模仿表面模式而非掌握精髓（与 DeepSeek-R1 在数学/代码的观察一致）。
- 与最近邻工作的Δ：vs 数学域 Gandhi 2025（"cognitive behaviors enable self-improving reasoners"）——把"base 行为决定 RL 上限"从数学**迁移并系统化到 agentic search**,并新增"错误轨迹消融"证明是行为而非正确性;vs 普通 SFT-then-RL（Distillation 随机采样 / Outcome-driven 只筛正确轨迹）——本文按"是否展现全部四行为"筛,且证明它**超过专门筛正确轨迹的 Outcome-driven**。关键点:**SFT 阶段该优化的目标不是"答对"而是"展现好行为"**。

## 怎么做 + 靠不靠谱

**【第一部分:识别行为（§3）——可复现的方法论】**
1. **标准 agentic search 框架（§3.1）**：迭代式。第 \(k\) 步给定历史 \(\mathrm{ctx}_k\)，模型输出 \(y_k=\langle t_k,a_k\rangle\)（\(t_k\)=reasoning、\(a_k\)=action）。动作 \(\in\{\)search 检索, answer 终止, **summary 压缩历史管上下文**\(\}\)。历史更新规则:
   \(\displaystyle \mathrm{ctx}_{k+1}=\begin{cases}\mathrm{ctx}_k+y_k+\mathrm{info}_k, & a_k=\text{search}\\ a_k, & a_k=\text{summary}\end{cases}\)
   （search 把检索结果 \(\mathrm{info}_k\) 追加;summary 把累积历史替换为压缩版,支撑有限上下文模型。完整 prompt 见 Appendix B.1。）
2. **配对轨迹**：Gemini 2.5 Flash（强）成功 而 Qwen3-1.7B（弱）失败 的 **500 题**,各自的正确轨迹 vs 错误轨迹。数据来自 Li 2025c（Chain-of-Agents）的 web agent SFT 集（后续造语料同源）。
3. **三阶段 LLM pipeline（§3.2,受 AutoRule-Wang&Xiong 2025 启发）**：① **轨迹比较**（详析"为何一成一败",用 Gemini 2.5 Flash）→ ② **行为抽取**（提炼贡献成功的关键行为,Gemini 2.5 Flash）→ ③ **行为合并**（去重/合并相似/留普适,Gemini 2.5 **Pro**）+ 人工复核 → 得**四行为**:
   - **Information Verification**:跨多源验证检索结果、reasoning 中引证。
   - **Authority Evaluation**:识别来源冲突、评可信度、优先权威信息。
   - **Adaptive Search**:据前次结果动态改搜索策略。
   - **Error Recovery**:识别并纠正先前步的错误（≈path-recovery/回轨）。
4. **跨模型验证普适性**：在 Gemini 2.5 Flash / DeepSeek-R1 / Llama3.2-3B / Qwen3-1.7B 上测三 web benchmark,**行为频率与性能强正相关**（Fig.2,频率=展现该行为的轨迹占比）。

**【第二部分:Behavior Priming 训练（§4）——读完可复现】**
5. **SFT 种行为（§4.1）**：web/QA 两类任务各从 Gemini 2.5 Flash **每题采 10 条**轨迹（问/答来自 Li 2025c）;用 LLM 判每条是否**同时展现全部四行为**,只留全展现的;**把轨迹每一步当独立训练样本**:
   \(\displaystyle \mathcal D_{\text{SFT}}=\{\langle x^i_k,y^i_k\rangle\mid 1\le k\le L_i\},\quad T_i=(\langle x^i_1,y^i_1\rangle,\dots,\langle x^i_{L_i},y^i_{L_i}\rangle).\)
   数据统计（Table 1）:web Behavior Prime 2.9k 轨迹×平均 6.8 步=20k step 样本（accuracy 49.8%）、Incorrect 2.6k×7.6 步（acc 0%）、Correct 3.4k×5.9 步（acc 100%）;QA 2.2k×4.6 步=10k。注意:Behavior Prime 轨迹**更长**（步数多 → 探索更充分）。
6. **RL 精炼（§4.2）**：primed 模型上跑 **GRPO**,**按 step 聚合**更新:
   \(\displaystyle J_{\text{GRPO}}(\theta)=\mathbb E_{q\sim\mathcal D,\{T_i\}_{i=1}^G\sim\pi_{\theta_{\text{old}}}}\Big[\sum_{i=1}^G\sum_{k=1}^{L_i}\sum_{t=1}^{|y_k|}\frac{1}{|y_k|}\min\big(r_{i,k,t}(\theta)\hat A_i,\ \mathrm{clip}(r_{i,k,t}(\theta),1\pm\varepsilon)\hat A_i\big)\Big],\)
   其中 \(r_{i,k,t}\) 是 importance ratio。**outcome 二值奖励**:LLM-judge 判最终答案对/错给 \(R_i\in\{0,1\}\)，且 **\(R_i\) 和 \(\hat A_i\) 对轨迹内所有 step/token 恒定**（这是与"path 重要"主张的内部张力所在——RL 阶段退回 outcome 级）。
- **逐组件必要性（消融较完整）**：
  - **四行为 vs 单行为（§5.4 Table 5）**——IV-Only-10k（只筛 Information Verification）比 Direct RL 好（Overall 17.4 vs 13.9）但**一致被全四行为 Behavior Prime-10k 超过（19.7）**,证复合行为协同必要。
  - **SFT 种行为 vs RL 自己长（§7 Table 6）**——Direct RL 后行为频率多数下降,证需显式先验。
  - **行为 vs outcome（§5.2 Table 4）**——核心消融（见 TL;DR）:SFT 后 Incorrect 9.3 < Correct 11.3,但 **RL 后抹平为 23.5≈23.6**。
  - **SFT 数据规模（§5.3 Fig.3）**——5k/10k/20k 递增,但 WebWalkerQA **5k 后 plateau**（行为一旦学会,额外 priming 收益递减）。
  - **process reward 替代（§7 Table 6）**——失败:行为频率大涨但性能反降（reward hack）。
- **关键机制/公式（直觉）**：**无新损失**,GRPO 标准目标。机制解释靠**训练动态**:
  - SFT 后行为频率↑、Pass@8↑、平均步数↑（Fig.1/Fig.4,且 Behavior Prime 涨幅显著大于 Distillation）→ 更丰富探索 + 更多 search 动作（test-time scaling 基础）。
  - RL 中 **primed 模型维持更高 policy entropy、不早崩**（Fig.5a）;Direct RL 熵骤降、早收敛到低 plateau（Fig.5b primed 收敛更慢但天花板更高）→ 更高熵=更丰富探索=更高 RL 上限。
  - 另一发现（Fig.5c）:Direct RL 起步 valid-action-ratio 低但 20 步内掌握格式并维持高;primed 模型 valid-action-ratio 反而**不太稳**——说明增益来自**推理行为**而非熟悉工具语法。
- **实验与证据**：backbone Qwen3-1.7B、Llama3.2-3B-Instruct;评测 3 web benchmark（GAIA 用 103 文本例、WebWalkerQA、HLE）+ 7 multi-hop QA（NQ/TQ/HotpotQA/2Wiki/MuSiQue/Bamboogle/PopQA）。
  - **主表 Table 2/3**：相对 Direct RL,web **+37.2%**、QA **+6.2%**（相对提升）;一致超 Distillation/Outcome-driven 两基线;且比肩/超 Search-R1、R1-Searcher、DeepResearcher 等 prior work（Qwen3-1.7B QA Overall 65.5 vs DeepResearcher 61.8）。
  - baseline 公平:两 SFT-then-RL 基线从**同一 Gemini 语料**选轨迹,仅选择策略不同。
- **假设与失效边界**：【原文】(§A) SFT 3 epoch、bs8;RL 在 **verl-agent（Feng 2025 GiGPO 系）** 上 300 步、bs32、GRPO group 8、**8×H100 约 20h**;评测 temp0、pass@k temp1,**GPT-4o-mini 评分**;web 轨迹 max 25 步、QA max 15 步。【推断】**规模偏小**（1.7B/3B）,更大模型是否仍需显式先验未知;四行为由 **LLM-judge 识别与筛选**,整套结论依赖判别 prompt 质量与 Gemini 判断一致性（**用 LLM 定义"好行为"又用 LLM 评是否展现,存循环依赖**）;web 增益大(+37%)、QA 温和(+6.2%) 说明对任务类型敏感。
- **祛魅总结**：【推断】真贡献是 **"path 监督 > outcome 监督"这一可迁移结论的强因果证据**（错误轨迹消融干净）+ 把数学域发现系统迁移到 agentic search + process-reward 失败的反面教训。被包装/张力的是:**SFT 阶段是 path 监督、RL 阶段又退回 outcome 级 GRPO**（\(R_i,\hat A_i\) 全轨迹恒定,无 step/turn 信用分配）,与"path 重要"的主张存在内部张力——它只在 SFT 喂 path,RL 仍靠 outcome reward;"行为定义"高度依赖特定 LLM-judge,可迁移性未充分检验。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=四种推理行为的轨迹示范（SFT,LLM-judge 筛"全展现"）+ outcome 二值奖励（RL,\(R_i\in\{0,1\}\)）｜**改什么**=参数（SFT 全量微调 + RL 更新策略）｜**何时改**=离线两段式（先 SFT 种行为 → 再 GRPO 在线精炼）｜**免梯度?**=否｜**记忆-技能生命周期**=部分相关——"行为"≈可复用技能,但只写入参数,无外部技能库/检索/共享;summary action 是轻量上下文管理（非持久记忆）｜**防遗忘机制**=无显式机制（SFT→RL 顺序,未处理灾难性遗忘）
- ⑦ 开源代码+框架/harness：主仓库 https://github.com/cxcscmu/Behavior-Priming-for-Agentic-Search （旧 analysis 已核:旧 KEYS 的 `cxcscmu/Behavior-Priming` 是 404,此为正确仓库名）。**SFT 框架=LLaMA-Factory**（vendored,`scripts/train.sh`→`llamafactory-cli train`）;agent rollout/评测靠 vLLM;**RL 框架=GRPO,在独立仓库** https://github.com/cxcscmu/verl-agent-deepresearch （veRL 系,verl-agent/GiGPO 的 deep-research 变体）,主仓库**不含 RL 训练代码**。harness=自建 agentic search 框架（search/answer/summary 三动作,Appendix B 给完整 prompt,含 internal-thinking 与无 thinking 两版）。【待核】框架细节复用旧 analysis 核查,本次基于 PDF 重写未重新进仓核对。
- 💰 资源/成本与可扩展性：【原文】RL 单次运行 8×H100 约 20h;SFT 数据 web 20k steps / QA 10k steps（步级样本）;每题采 10 条 Gemini 轨迹做语料。
- 🎯 对"探索-巩固"对标：**强支撑（核心对标论文之一）**。判定:本文 = TSRD"teacher 当稀疏脚手架→student 内化→RL"的**直接经验先例**——四行为里 **Error Recovery（识别并纠正先前错误）几乎就是 path-recovery/回轨**、Adaptive Search≈选路调整;"path 监督 > outcome 监督"为 TSRD"教选路+回轨比教对错更重要"提供**最直接的因果证据**（Table 4）;"primed 模型 RL 中维持更高熵=更好探索"（Fig.5a）支撑"巩固有效行为后探索空间更大"。**可借组件**:① 用强弱轨迹对比 + LLM pipeline 自动提炼"有益行为/path"的方法论（§3.2 三阶段）;② 把单步当独立样本的 step 级 SFT（\(\mathcal D_{\text{SFT}}\) 构造）。**缺口/竞品面**:RL 仍 outcome 级无 step 信用分配（TSRD 要更细的 path-recovery 单点接管）;行为靠 LLM-judge 识别而非 logit/MTP 前瞻;无 teacher 在线脚手架（teacher 只在离线造 SFT 数据）。
- 🔭 开放问题/未来方向：【原文】"targeted post-training 增强推理行为以提升 RL 有效性"是有前景方向（§8）;探讨过 process reward 作替代但失败（§7）。【推断】更大模型上的必要性;行为识别去 LLM-judge 化（用更客观信号,如 logit/熵）;把 path 监督贯穿到 RL 阶段（step/turn 级信用分配）而非 SFT 后退回 outcome;teacher 在线脚手架而非仅离线造数据;与 MTP 前瞻结合识别"该回轨"的时点。

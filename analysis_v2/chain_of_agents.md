chain_of_agents | Chain-of-Agents: End-to-End Agent Foundation Models via Multi-Agent Distillation and Agentic RL | OPPO AI Agent Team（通讯 Wangchunshu Zhou，30+ 作者） | 2025-08-06 · arXiv preprint v1（标注 2025-08-20）· cs.AI | 主题线 L4（Agent/工具/多轮）兼 L2（SFT+RL recipe）·相关性 中（"distillation" 为 agent-level 轨迹 SFT，非 token 级 OPD）

**原始论文**：https://arxiv.org/abs/2508.13167

## 一眼看懂
- 🟦 TL;DR：把"多智能体系统（MAS）的协作能力"压进**单一模型**。MAS 性能强但靠人工 prompt/workflow 工程、agent 间通信冗余、不能 data-centric 学习；TIR（工具集成推理）能端到端训练但只支持 ReAct 单一"think-action-observation"轨迹【原文 §1 L181-200, Fig.2】。CoA（Chain-of-Agents）让一个模型在**单次解码内**动态激活不同 role-playing agent（Thinking/Plan/Reflection/Verification）与 tool agent（Search/Crawl/Code），像 MAS 那样多轮多工具求解。训练 recipe 两步：**(I) multi-agent distillation**——让 SOTA 多智能体系统 OAgents 执行任务，把成功轨迹转成 CoA 格式做 agentic SFT 冷启动（带 observation masking）；**(II) agentic RL**——在可验证任务上用 **DAPO** 优化，outcome 二元奖励【原文 §3, Fig.4】。得到的模型叫 Agent Foundation Model（AFM）。
- 最巧的一步：**把 MAS 的执行轨迹"录制"成单模型可学的 CoA 序列（式5 \(\tau=\{(S_t,\phi_t,o_t)\}_{t=1}^T\)）**。这是把"多个 agent 协作"降维成"一个模型按状态转移 \(S_t=f_\theta(S_{t-1},\phi_{t-1},o_{t-1})\) 动态激活角色 \(\phi_t\)"的关键——抽掉这一步（没有 OAgents 当 teacher 提供高质量协作轨迹），阶段 I 冷启动就没有数据来源，后续 RL 也无从在合理的 CoA 行为空间里探索。本质是**用一个强 MAS 当 teacher，把它的序列决策模式蒸进 student**（agent-level / sequence-level 蒸馏，"distill the sequential decision-making patterns of expert multi-agent systems rather than word-level distributions"）【原文 §3.2.1 L362-365】。

## 为什么做
- 研究背景：MAS（deep research、vibe coding）靠多角色/工具 agent 协作解复杂任务，性能强；TIR（Search-R1/WebThinker）把工具调用编进推理、端到端训练，实证优于纯 prompt 工程的 ReAct agent【原文 §1, §2】。
- 解决的具体痛点【原文 §1 L184-200】：①MAS 计算低效（agent 间冗余通信 + 复杂 workflow）；②难泛化（换域要重做 prompt/workflow 工程）；③无法 data-centric 学习（性能不能靠训练 agentic 任务提升）；④MAS 的 backbone LLM 通常**静态、没针对多轮/多工具/多角色训练**，只靠 prompt 凑（"the backbone LLMs used in multi-agent systems are generally not trained … for agentic use"）。TIR 虽端到端可训但**只支持 ReAct 单一轨迹模式**，无法端到端训练 LLM 去支持完整 MAS。
- 相关工作 & 各自不足（Table 1 四维对比，Tool Integration / End-to-end Execution / Multi-agent Collaboration / Data-centric Optimization）：ReAct（Tool✓/端到端✗/多agent✗/data-centric✗）、Multi-Agent System（端到端✗/data-centric✗）、Tool-Integrated Reasoning（多agent✗）——只有 **Chain-of-Agents 四项全✓**【原文 Table 1 L366-391】。
- 动机链：MAS 优于 ReAct 但不可端到端训练 → TIR 可端到端但只 ReAct → 想要"既有 MAS 的动态多角色编排、又能 SFT+RL 直接优化"→ 把 MAS 的动态角色编排能力蒸进单模型（用 system prompt 定义多 agent、Thinking Agent 状态转移动态激活）→ CoA + 两阶段 recipe。为什么不用更简单做法：纯 prompt 工程的 MAS 不可训练（"LLM backbones in agent frameworks are static and cannot be directly optimized"）、TIR 表达力不够——所以要"蒸 MAS + agentic RL"。
- 与最近邻工作的Δ：vs TIR（Search-R1/WebThinker）——CoA 支持"any workflow that can be modeled by a multi-agent system"（多 role-playing + 多 tool agent），非仅 ReAct；vs MAS——CoA 在**单次解码内**编排、保持上下文连续（"maintains contextual continuity"）、省去 agent 间通信开销、可端到端训练。**关键有用点**：单模型 + token 消耗减少 **84.6%**（vs 传统 MAS）仍有竞争力【原文 §1 L228-229, §3.1 L350-356】。

## 怎么做 + 靠不靠谱
- 方法流水线（读完可复现）【原文 §3, Fig.4】：①agentic 任务生成+过滤（沿用 Shi et al. 程序，depth/width 扩展增复杂度）→②让 OAgents（SOTA 开源 MAS）执行，录制成 CoA 兼容轨迹 \(\tau=\{(S_t,\phi_t,o_t)\}\)→③四阶段渐进质量过滤→④用 LLaMA-Factory 做 agentic SFT（observation masking）冷启动→⑤在 SFT 未用过的 QA 上 tool-aware rollout，用 veRL 跑 DAPO，outcome 二元奖励，只选难题（\(r_q\le0.3\)）训练。
- 关键公式（真实形式 + 直觉）：
  - **CoA 动态编排状态转移**（式4，Thinking Agent 当总调度）：
    \(\displaystyle S_t=f_\theta(S_{t-1},\phi_{t-1},o_{t-1}),\qquad \phi_t\sim P(\phi\mid S_t)\)
    \(S_t\) 维持持久推理状态，\(\phi_t\in\{\phi_{\text{think}},\phi_{\text{plan}},\phi_{\text{search}},\dots\}\) 是被激活的角色——看当前状态决定下一步激活哪个角色/工具【§3.1 L342-345】。
  - **蒸馏轨迹定义**（式5，agent-level / sequence-level KD 目标）：
    \(\displaystyle \tau=\{(S_t,\phi_t,o_t)\}_{t=1}^{T}\)
    \(S_t\) 推理状态、\(\phi_t\sim P(\phi\mid S_t)\) 激活的 agent、\(o_t\) 该 agent 的观测——把 OAgents 执行过程的"agent 激活序列 + 推理状态"录成可学序列（非 word 分布）【§3.2.1 L448-454】。
  - **轨迹构造迭代循环**（式6）：\(S_t=\Gamma(S_{t-1},o_{t-1})\)，\(a_t\sim\pi(\cdot\mid S_t)\)，\(o_t=\Phi(a_t)\)（\(\Gamma\) 状态转移、\(\pi\) 动作策略、\(\Phi\) agent 执行环境）【§3.2.1 L459-463】。
  - **SFT 目标 + observation masking**（式7，核心训练损失）：
    \(\displaystyle \mathcal{L}_{\text{SFT}}=-\sum_{t\notin O}\log\pi_\theta(\tau_t\mid\tau_{<t},q)\)
    其中 \(O\) 为工具观测 token 集——**观测不计入 loss，防环境噪声反传**（"prevent environmental noise propagation"）；训练轨迹格式 `<think>C_cot</think><tools>α_m(α_p)</tools><observation>O_t</observation><reflection>F_t</reflection>...<answer>A_t</answer>`【§3.2.1 L541-549】。
  - **RL 难题选择信号**（式8-9，web agent）：\(r_q=\frac1N\sum_{i=1}^N\mathbb{I}[\text{EM}(a_i,y_{gt})=1]\)（\(N=32\) 次预测的无工具通过率，量化"参数知识污染风险"），剔除 \(r_q>0.3\)（易/被污染），只在 \(Q_{\text{RL}}=\{q_j\mid r_{q_j}\le0.3\}\) 上 RL【§3.3.1 L562-583】。
  - **奖励函数**（式10-11）：\(R_{\text{web}}(\tau)=\text{score}_{\text{answer}}\)（LLM-as-Judge 二元，免格式奖励因 SFT 已保证格式）；\(R_{\text{code}}(\tau)=\text{score}_{\text{answer}}\cdot\text{score}_{\text{format}}\)（沙箱过全部测例 + math 用 Math-Verify；format 检查 `<code>```py...```</code>`，**两者都满足才得满分**）【§3.3.2 L600-616】。
- 逐组件必要性：
  - **multi-agent distillation（OAgents teacher）**：提供冷启动数据；无它无 CoA 行为先验。核心组件，但**无"换不同 teacher MAS"的消融**——重度依赖 OAgents 质量。
  - **四阶段过滤**：①复杂度（<5 agent-tool 交互剔除）②质量（剔错答/冗余 tool 输入/不遵指令，QA/search 用 LLM 判、code 用测例、math 用精确匹配）③reflection enrichment（缺反思的下采样；math/code 直接丢无反思轨迹）④error-correction 上采样（search/QA 中 `<double_check>` 初始低可信〔GRM 可信度〕但最终纠对的轨迹）【原文 §3.2.1 L468-534】。产物三特征：全需多工具协作、推理链 5-20 hop（远超标准基准 2-3 hop）、富含含纠错的反思轨迹。全是人工启发式阈值，无敏感性分析。
  - **observation masking**：O（工具观测）不计入 loss，防环境噪声反传【式7 L544-549】。agentic SFT 的标准但必要设计。
  - **agentic RL（DAPO）+ \(r_q\le0.3\) 难题选择**：用 Qwen-2.5-72B 判 question solvability，\(r_q>0.3\)（易/被污染）剔除；code agent 用 7B AFM 采 8 次、全过则丢（"insufficiently challenging"）【原文 §3.3.1 L584-588】。**无"有无 RL 阶段"的干净并列消融**（正文未见 SFT-only vs SFT+RL 表）。
- 实验与证据【原文 §1, §4】：
  - **数据集/设置**：近 20 个 agent benchmark。Web/QA：NQ/TriviaQA/PopQA/HotpotQA/2Wiki/MuSiQue/Bamboogle/GAIA（103 text-only 题，与 [29,65] 公平对比）/WebWalker/BrowseComp/HLE（2500 多模态题）；Code：LiveCodeBench/CodeContests；数学 AIME25。backbone **Qwen2.5-3B/7B/32B-Instruct**（默认 32B）【§4.1 L699-762, L1032】。
  - **训练实现（关键复现细节）**：**SFT 用 LLaMA-Factory**【§4 L1037】；**RL 用 DAPO（[69]）on veRL（[48]）**，"each training iteration processes 64 prompts, generating 8" rollouts（与 Search-R1 设置一致）【§4 L1040-1044】。
  - **主结果（Figure 1 + §4）**：AFM-32B 多 benchmark SOTA——GAIA 55.3%、BrowseComp 11.1%、HLE 18.0%（web）；LiveCodeBench v5 47.9%、CodeContests 32.7%（code）；AIME2025 **59.8%**（较此前最佳 TIR ReTool/SimpleTIR 绝对 +10.5%+）【§1 L223-226】。7B web agent 接近用更强 QwQ-32B backbone 的 WebThinker-RL。
  - **效率**：相比传统 MAS，**token 消耗减少 84.6%**，性能仍竞争力【§1 L228-229】。
  - **baseline 公平吗**：**多数 SOTA 在同 backbone（Qwen2.5）受控对比下成立**；但 Figure 1 也跨 backbone 比（如 AFM vs 用 QwQ-32B 的 WebThinker-RL、vs QwQ-32B）——**跨 backbone 比较应保留**（更强 backbone 的对手不构成同条件对照）。
  - **看着强但没回答核心问题**：方法学创新有限（组合既有组件）；"distillation"是 sequence-level 轨迹 SFT，**无 token 级信用分配**；reward 是 outcome 二元信号，**过程监督缺失**——多轮 agent 的中间步好坏未被直接监督。
- 假设与失效边界：
  - 【原文/推断】SOTA 在跨 backbone 比较下不一定稳健（受控对比多在同 Qwen2.5）。
  - 【推断】严重依赖教师 MAS（OAgents）的质量与四阶段过滤的人工启发式阈值（<5 交互、\(r_q\le0.3\)、采 8 次全过则丢等）——teacher 弱或阈值不当则数据质量下滑。
  - 【推断】需自建 tool servers（web search/crawl + code sandbox）+ 较大算力（32B + 多轮 rollout），复现门槛高。
- 祛魅总结【推断】：作为**系统/工程论文**整体自洽、结果扎实、开源完整（模型权重 + 数据 + 训练/评测代码全开）。但**方法创新有限**——核心是 OAgents 蒸成 CoA 轨迹 + LLaMA-Factory SFT + veRL/DAPO 的组合 recipe，每个组件都是既有的。"distillation"一词在此是 **agent-level/sequence-level 轨迹 SFT**，与本项目 token-level on-policy 蒸馏**不同源**，关联较弱。被高估的是"SOTA"主张（部分靠跨 backbone 比 + 重度依赖 teacher MAS）；被低估/真正有价值的是 **agentic 数据构造流水线（四阶段过滤 + error-correction 上采样）+ observation masking** 这套可复用的工程方法，以及 84.6% token 削减的效率证据。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：阶段I=OAgents 多智能体轨迹的 sequence-level 模式（agent 激活序列 \(\phi_t\) + 推理状态 \(S_t\)，经 observation-masked NLL SFT，式7）；阶段II=outcome 二元奖励（web：LLM-judge \(R_{\text{web}}=\text{score}_{\text{answer}}\)；code：\(R_{\text{code}}=\text{score}_{\text{answer}}\cdot\text{score}_{\text{format}}\)）的 DAPO 优势。
  - **改什么**：参数（整个 AFM 模型，SFT + RL 全参微调）；harness 层面引入多 agent 的 system-prompt 编排（固定范式，非被学的对象）。
  - **何时改**：离线两阶段（先 SFT 冷启动，再 agentic RL per-episode）；非 test-time。
  - **免梯度?**：否（SFT + RL 全梯度）。
  - **记忆-技能生命周期**：无持久记忆/技能库；"技能"= 不同 role/tool agent，由 system prompt 定义、Thinking Agent 在单次解码内动态激活 \(\phi_t\sim P(\phi\mid S_t)\)（写入=训练进参数；无跨 episode 检索/遗忘/共享）。tool servers（search/crawl/code sandbox）是外部环境，非记忆。
  - **防遗忘机制**：无显式防遗忘（单次两阶段训练，不涉及持续学习）。
- ⑦ 开源代码+框架/harness：https://github.com/OPPO-PersonalAI/Agent_Foundation_Models （Apache-2.0，~61MB，模型权重/数据/训练评测代码全开）。**repo 结构**：`AFM/`（data/evaluation/models/tool_servers/train，含 `train/code_agent/rl/train_dapo_code_agent.sh`、`train/web_agent/rl/train_dapo_web_agent.sh`）、`LLaMA-Factory/`（SFT）、`verl/`（RL，含 `docs/algo/dapo.md`）、`assets/`。框架：**SFT 用 LLaMA-Factory；RL 用 veRL 跑 DAPO**【原文 §4 L1037-1044 佐证】；backbone Qwen2.5-3B/7B/32B-Instruct。代码可得性好，但需自建 tool servers + 较大算力。
- 💰 资源/成本与可扩展性：训练数据规模大（MHQA SFT 8.8k avg 4.35 hops、RL 169k；Web Agent SFT 7.6k avg 7.29 hops）；backbone 到 32B；需 web search/crawl API + code sandbox；DAPO 每 iter 64 prompts × 8 rollouts【原文 Table 2/3, §4 L1040】。**核心效率卖点**：推理 token 消耗比传统 MAS 减 84.6%（单模型省去 agent 间通信）【§1, §3.1】。可扩展性：3B/7B/32B 三规模验证；7B 已接近更大 backbone 的 baseline。
- 🎯 对"探索-巩固"对标：**弱支撑/部分竞品（L4 自进化 Agent 维度），但蒸馏机制不同源**。CoA 触及本项目"自进化 Agent"映射的**探索侧**——发现有效的多工具/多角色行为路径（Plan/Reflection/Verification/double_check 正是"选路 + 走偏后纠错"的显式角色化）。其 **error-correction 上采样**（初始低可信→迭代纠对的轨迹被上采样）与本项目"path-recovery/走偏后自选恢复分支"直觉**部分同构**——它显式偏好"能识别并改正初始错误"的轨迹。**关键 Δ/缺口**：(1) CoA 的"巩固"是 sequence-level 轨迹 SFT（式7）+ outcome RL（式10-11），**无 token 级信用、无过程监督**——本项目要的"在关键步接管/前瞻"它没有；(2) 纠错是靠**数据层**（上采样含纠错的轨迹）实现，非机制层（无在线 path-recovery 的单点接管）；(3) 无技能库/记忆生命周期/防遗忘——不是持续学习意义上的"自进化"。**可借组件**：①agentic 数据构造（任务 depth/width 扩展 + 四阶段过滤 + 把成功 MAS 轨迹录成可学序列 \(\tau=\{(S_t,\phi_t,o_t)\}\)）是本项目构造"探索-巩固训练数据"的现成流水线；②observation masking（式7，环境噪声不反传）对任何多轮 agent 训练通用；③"显式角色化 Reflection/Verification/double_check"提供了把"纠错/回轨"做成可监督子序列的思路。一句判定：agentic 系统层面与本项目探索侧（发现有效行为 + 纠错）相关、数据工程可借，但其蒸馏是 agent-level 轨迹 SFT 而非 token-level OPD，且无前瞻/记忆/防遗忘，与本项目核心机制不同源，属"参考价值在数据与系统、不在蒸馏机制"。
- 🔭 开放问题/未来方向：
  - 【原文/推断】把更多 SOTA MAS 蒸进 CoA（降低对单一 OAgents 的依赖）；更细的过程奖励替代 outcome 二元信号。
  - 【推断】给多轮 agent 引入 token/step 级信用分配（对接本项目 L6/CEPO 式 credit assignment，解决"哪一步工具调用/反思真正决定成败"）；把 error-correction 从"数据上采样"升级为"在线 path-recovery 机制"（走偏检测 + 单点接管），并加入跨 episode 的技能/记忆沉淀与防遗忘——把 CoA 从"一次性训练的 AFM"推向真正的持续自进化 Agent；用 MTP 前瞻指导 Thinking Agent 的角色激活（\(\phi_t\) 选择前瞻化）。

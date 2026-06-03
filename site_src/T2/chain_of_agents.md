# chain_of_agents — Chain-of-Agents: End-to-End Agent Foundation Models via Multi-Agent Distillation and Agentic RL

> **一句话重点 (TL;DR)**：把"多智能体协作"压进单一模型——先用 multi-agent distillation 把 SOTA 多智能体系统（OAgents）的执行轨迹转成 CoA 格式做 agentic SFT 冷启动，再用 agentic RL（DAPO）在可验证任务上优化，得到能原生动态激活不同 tool/role agent 的 Agent Foundation Model（AFM）。本质是组合既有组件的大体量工程 recipe，与 token 级 OPD 不同源。

**元信息**：arXiv 2508.13167（v1，2025-08-06；论文标注 Date: 2025-08-20）｜ OPPO AI Agent Team（30+ 作者，通讯 Wangchunshu Zhou）｜ Preprint ｜ 主题：agent foundation model 训练 recipe（与 OPD/distillation 主线关联较弱——这里的 "distillation" 是 agent-level/sequence-level 轨迹 SFT，不是 token-level on-policy 蒸馏）｜ 代码 https://github.com/OPPO-PersonalAI/Agent_Foundation_Models （Apache-2.0，~61MB，模型权重/数据/训练评测代码全开源）｜ 框架 SFT 用 LLaMA-Factory、RL 用 veRL 跑 DAPO

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/chain_of_agents/fig_01.png)

*Figure 1 Performance comparison of AFM with the proposed Chain-of-Action paradigm against state-of-the-art tool-integrated reasoning (TIR) methods on GAIA, BrowseComp, HLE, and AIME25 benchmarks. AFM demonstrates consistent effectiveness across web agent and code agent benchmarks.*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/chain_of_agents/fig_04.png)

*Figure 4 Overview of the training framework. (I) The SFT stage utilizes reformatted ReAct data with both short and long chains of thought for cold start. (II) The RL stage performs tool-aware rollouts on unused QA pairs and optimizes the policy.*

## 1. 相关工作与进展
- **多智能体系统（MAS）**：靠多个角色/工具的 agent 协作解复杂任务（deep research、vibe coding），性能强但靠人工 prompt/workflow 工程。
- **Tool-Integrated Reasoning（TIR）**：把工具调用显式编进推理（Search-R1、WebThinker），让 LLM 端到端支持 ReAct 式 "think-action-observation"。实证显示 TIR 训练优于纯 prompt 工程的 ReAct agent。
- 但 TIR 只能表达 ReAct 单一轨迹模式，无法端到端训练 LLM 去支持完整 MAS。CoA 想填这个 gap。

## 2. 现有工作存在的问题
1. 手工多智能体 workflow 计算低效（agent 间冗余通信）、难泛化（换域要重做 prompt/workflow 工程）、不可端到端 data-centric 学习。
2. MAS 里的 backbone LLM 通常静态、没针对 agentic（多轮/多工具/多角色）用法训练。
3. TIR 只支持 ReAct（think-action-observation），轨迹表达能力有限。

## 3. Motivation
把多智能体协作建模进**单一模型**：让它端到端、原生地动态激活不同 tool agent（search/crawl/code）与 role-playing agent（think/plan/reflect/verify），像 MAS 那样做多轮多工具求解，同时可被 SFT + RL 直接优化；并大幅削减 MAS 的 agent 间通信 token 开销。

## 4. 主要灵感 / 核心直觉
- 既然 MAS 优于 ReAct，就把 MAS 的"动态角色编排"能力蒸进一个模型，用 system prompt 定义多个 agent、由 Thinking Agent 通过状态转移 S_t = f_θ(S_{t-1}, ϕ_{t-1}, o_{t-1}) 动态激活角色 ϕ_t。
- 蒸馏层级是 **agent-level / sequence-level**（学 MAS 的序列决策模式），而非 word-level 分布——这是它与 token 级 KD/OPD 的本质区别。

## 5. 主要解决思路（一段话讲清核心）
两阶段 recipe：**(I) Multi-Agent Distillation（agentic SFT 冷启动）**——沿用 Shi et al. 的 agentic 任务生成+过滤，让 SOTA 多智能体系统 OAgents 执行这些任务，把成功的多智能体协作过程录成 CoA 兼容轨迹，经四阶段质量过滤后用 LLaMA-Factory 做 SFT（带 observation masking 防环境噪声反传）。**(II) Agentic RL**——在 SFT 未用过的 QA 上做 tool-aware rollout，用 veRL 跑 DAPO，outcome-driven 二元奖励，只选难题训练。

## 6. 方法详解（通俗、分步骤）
**CoA 范式**：role-playing agents（Thinking 编排、Plan 分解、Reflection 自评、Verification 校验）+ tool agents（Search、Crawl、Code Generate）；单次解码过程内由 Thinking Agent 动态编排，保持上下文连续、省去 MAS 的 agent 间通信开销。
**阶段 I 数据生成与四阶段渐进质量过滤**：① 复杂度（<5 agent-tool 交互剔除）② 质量（剔错答/冗余/脏数据）③ reflection enrichment（缺反思的下采样）④ error-correction 上采样（search/QA 中带 `<double_check>` agent 纠错的轨迹）。轨迹格式带 observation masking（O 不计入 loss）。
**阶段 II Agentic RL**：tool-aware rollout + DAPO；reward 用 outcome-driven 二元信号（web agent 用 LLM judge M_j 二元判断、免格式奖励；code agent reward = score_answer · score_format）；RL 数据选难题（rq≤0.3）。
**backbone**：Qwen2.5-3B/7B/32B-Instruct。整体是组合既有组件（OAgents 蒸馏 + LLaMA-Factory + veRL/DAPO）的工程 recipe，方法学创新点有限。

## 7. 实验数据集
近 20 个 agent benchmark。Web/QA：NQ、TriviaQA、PopQA、HotpotQA、2Wiki、MuSiQue、Bamboogle、GAIA、WebWalker、BrowseComp、HLE 等。Code agent：LiveCodeBench、CodeContests 等。数学：AIME25。RL 数据 setting 同 Search-R1。

## 8. 实验结果与主要发现
- **AFM-32B 多 benchmark SOTA**（Figure 1）：GAIA 55.3%、BrowseComp 11.1%、HLE 18.0%（web agent）；LiveCodeBench v5 47.9%、CodeContests 32.7%（code agent）；AIME2025 59.8%（数学，较此前最佳 TIR 如 ReTool/SimpleTIR 绝对 +10.5%+）。
- **效率**：相比传统多智能体系统，token 消耗减少 84.6%，性能仍有竞争力。
- **7B 版** web agent 上接近用更强 QwQ-32B backbone 的 WebThinker-RL。
- 全开源（模型权重 + 数据 + 训练/评测代码）。

## 9. 结果如何支撑其主张
- "单模型能像 MAS 一样工作" 由跨 web/code/math 近 20 基准的 SOTA 与 84.6% token 削减支撑。
- "端到端可训练" 由两阶段 recipe 本身 + RL 进一步提升支撑。
- **但 SOTA 主张需谨慎**：多数 SOTA 是在**同 backbone（Qwen2.5）受控对比**下成立；跨 backbone 比较（如对比用更强 QwQ-32B 的 WebThinker-RL）应保留。

## 10. 逻辑自洽性（中性评估）
作为系统/工程论文整体自洽、结果扎实、开源完整。方法创新有限——核心是把 OAgents 蒸馏成 CoA 轨迹 + LLaMA-Factory SFT + veRL/DAPO 的组合 recipe，每个组件都是既有的。"distillation"一词在此是 agent-level 轨迹 SFT，与本项目 token-level on-policy 蒸馏不同源，关联较弱。

## 11. 残留问题 / 局限
- SOTA 在跨 backbone 比较下不一定稳健（受控对比多在同 Qwen2.5 backbone）。
- 严重依赖教师 MAS（OAgents）的质量与四阶段过滤的人工启发式（阈值如 <5 交互、rq≤0.3 等）。
- 蒸馏是 sequence-level，无 token 级信用分配；reward 是 outcome 二元信号，过程监督缺失。
- 对本项目（MTP/OPD 方向）参考价值主要在 agentic 数据构造与 observation masking，而非蒸馏机制本身。

## 12. 开源代码与框架（链接 + 框架 + 代码可得性）
- 仓库：https://github.com/OPPO-PersonalAI/Agent_Foundation_Models （Apache-2.0，~61MB clone）。结构：`AFM/`（data / evaluation / models / tool_servers / train）、`LLaMA-Factory/`（SFT）、`verl/`（RL）、tool servers（web search/crawl + code sandbox）。
- 框架：SFT 用 **LLaMA-Factory**；RL 用 **veRL** 跑 **DAPO**；backbone Qwen2.5-3B/7B/32B-Instruct。代码可得性：模型权重、训练数据、训练 + 评测代码全开源，复现条件好（但需自建 tool servers 与较大算力）。

# sweet_rl — SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks

> **一句话重点 (TL;DR)**：在多轮 agent 任务上，用训练期才可见、actor 看不到的额外信息（最终结果 + 参考解）训练一个非对称的 turn-level critic 做 step 级信用分配；关键设计是"直接用动作 log-prob 参数化 advantage 函数 + Bradley-Terry 目标"，避开在 LLM 上训 value head 损泛化的问题；并配套开源多轮协作 benchmark ColBench。

**元信息**：arXiv 2503.15478（v1, 2025-03-19, cs.LG）｜ FAIR at Meta + UC Berkeley（Yifei Zhou, Song Jiang, Yuandong Tian, Jason Weston, Sergey Levine, Sainbayar Sukhbaatar\*, Xian Li\*）｜ arXiv 2025-03 ｜ 主题 多轮 agent step 级信用分配 + 新 benchmark / 相关性 中高（非对称 critic 用 teacher 特权信息做信用分配，与 TSRD 把 teacher 参考路径/foresight 用于 scaffold/信用分配同构，可作 asymmetric-critic 路线代表 baseline；不涉及 MTP）｜ 代码 https://github.com/facebookresearch/sweet_rl ｜ 框架 OpenRLHF 定制 fork（`YifeiZhou02/collab_openrlhf`，支持 multi-turn DPO + length normalization）

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/sweet_rl/fig_01.png)

*Figure 1 (Left) Overview of our ColBench including Backend Programming and Frontend Design that supports cheap and reliable evaluation of multi-turn RL algorithms for agents in realistic settings. (Right) The high-level motivation behind SWEET-RL that uses additional training-time information along*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/sweet_rl/fig_02.png)

*Figure 2 textbfAn overview of the training procedure of SWEET-RL. At a high level, we first apply Bradley-Terry objective to directly train a step-wise advantage function with access to additional training-time information. Once the advantage function is trained, we perform policy improvement by usi*

## 1. 相关工作与进展
LLM agent 需在真实任务中做多轮交互；要在序列决策任务上拿到最好性能，需直接优化多轮目标（如成功率），比 next-token 预训练只模仿每轮最可能动作更难。现有多轮 RL 路线包括 RAFT/DPO/PPO 系单轮方法外推、value-function/TD-learning、Path Consistency 等。

## 2. 现有工作存在的问题

- 把单轮 RLHF 方法（RAFT/DPO/PPO）直接用于多轮、不做跨轮显式信用分配，长 horizon 下高方差、样本复杂度差。
- value function 学习（TD-learning）需在 LLM 表征上训任务专用 value head，微调数据有限时泛化差（与 next-token 预训练目标差异大）。
- 缺乏同时满足三条件的 benchmark：(1) 任务多样性避免过拟合；(2) 足够复杂挑战推理/泛化；(3) 最小工程开销便于快速原型。现有 benchmark 无一兼具。

## 3. Motivation
利用一个常被忽视的事实：训练期可拿到 actor 推理时拿不到的额外信息（最终 outcome、reference solution）。这种非对称信息可为"信息搜寻型"动作的信用分配提供捷径。

## 4. 主要灵感 / 核心直觉
直接拿训练期额外信息训标量 value 会偏离 next-token 目标、损泛化；故改为**直接学 advantage 函数**（用每轮动作平均 log-prob 参数化）并用偏好（Bradley-Terry）目标对齐 LLM——这比"在 hidden state 上训 value head"更契合预训练 LLM。

## 5. 主要解决思路(一段话讲清核心)
两阶段：先用训练期额外信息 c（actor 不可见）训一个 turn-level critic，直接建模 turn-wise advantage（用动作 log-prob 参数化、BT 轨迹级目标），再把该 advantage 当每轮 reward model 对 actor 做 RLHF 式优化。配套提出 ColBench 多轮协作 benchmark 做评测。

## 6. 方法详解(通俗、分步骤)
两部分贡献：

1. **ColBench（Collaborative Agent Benchmark）**：聚焦 artifact creation 的多轮人机协作任务，含 Backend Programming（写 Python 函数）与 Frontend Design（做网页）。用 LLM 当 human simulator 且给其 reference artifact 以保证忠实模拟；用 functional evaluator（单元测试 / 图像相似度）客观度量产出与参考的相似度。>10k 程序化生成任务，可扩展、可调难度。
2. **SWEET-RL（RL with Step-WisE Evaluation from Training-time information）——两阶段**：
   - 阶段一（训 critic/advantage）：用训练期额外信息 c（如 reference solution，actor 不可见）训 turn-level critic，利用 critic 与 actor 的**非对称观测空间**。**直接建模 turn-wise advantage 函数**（用每轮动作平均 log-prob 参数化），不先学 value 再求 advantage。同任务下取两条轨迹按累积奖励标 chosen/rejected，在轨迹级用 **Bradley-Terry 目标**训该 advantage 函数。
   - 阶段二（policy improvement）：把训好的 advantage 函数当每轮 reward model，对 actor 做 RLHF 式 per-step 优化。

## 7. 实验数据集
ColBench 两任务：Backend Programming、Frontend Design。Backend 数据生成：用 Llama-3.1-70B-Instruct 从 DCLM 抽取片段生成 Python 函数 + 高层描述 + 单元测试，仅保留能过单测的；train 10k、test 1k（test 经作者人工检查）。15k 离线训练轨迹由 Llama-3.1-8B-Instruct 当 agent、Llama-3.1-70B-Instruct 当 human simulator 零样本生成。backbone：Llama-3.1-8B（agent）。

## 8. 实验结果与主要发现

- 相较其他 SOTA 多轮 RL，ColBench 成功率与 win rate 绝对 +6%。
- 使 Llama-3.1-8B 在协作内容创作上匹配/超过 GPT-4o（部分超 o1-mini）。
- baselines 含 RAFT/DPO/PPO 系、value-function/Bellman bootstrapping/Path Consistency 路线，及闭源 GPT-4o、o1-mini。

## 9. 结果如何支撑其主张
在自建 ColBench 上对一组 SOTA 多轮 RL 的一致 +6% 绝对提升，加上匹配 GPT-4o 的端到端表现，支撑"非对称 critic + 直接学 advantage + BT 目标"做 step 级信用分配的有效性。但提升幅度属稳健而非数量级，且评测局限于自家 benchmark。

## 10. 逻辑自洽性(中性评估)
设计自洽：非对称信息→直接学 advantage（避 value head 泛化问题）→BT 目标对齐 LLM→当 reward model 优化 policy。代码框架（OpenRLHF fork 的 multi-turn DPO）与方法的偏好学习目标一致。最大假设依赖在于"训练期可得 reference solution"——这是方法成立的前提，但也是其外推到真实场景的主要限制。

## 11. 残留问题 / 局限

- 增量是"非对称 critic + 直接学 advantage + BT 目标"的组合，依赖训练期可得 reference solution 这一较强假设（真实场景未必有）。
- ColBench 的"人类"用 LLM simulator 近似，且 simulator 依赖 reference artifact，模拟忠实度受 simulator 能力上限约束。
- 提升幅度（+6% 绝对）稳健但非数量级；仅在自建 benchmark 验证，跨 benchmark 泛化未证。
- 与 TSRD 是思路同构（特权信息做信用分配）而非直接可用方案；最大可复用资产是开源 benchmark + 数据。

## 12. 开源代码与框架(链接+框架+代码可得性)

- https://github.com/facebookresearch/sweet_rl （ColBench + SWEET-RL 官方实现）；数据集 https://huggingface.co/datasets/facebook/collaborative_agent_bench 。
- 框架：基于 **OpenRLHF 的定制 fork**（`YifeiZhou02/collab_openrlhf`），改造以支持 multi-turn DPO 与 length normalization（README 训练用 `openrlhf.cli.train_dpo`）。Frontend Design 评测需额外装 GeckoDriver + Firefox（渲染 HTML）。
- 流程：生成任务与离线轨迹 → 构 chosen/rejected 偏好对 → BT 目标训 turn-wise advantage（critic 用训练期信息 c）→ 用 advantage 当 per-turn reward 优化 policy。human simulator 用 Llama-3.1-70B-Instruct。

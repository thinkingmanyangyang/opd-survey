behavior_priming | Beneficial Reasoning Behaviors in Agentic Search and Effective Post-training to Obtain Them | CMU LTI（Jiahe Jin、Abhijay Paladugu、Chenyan Xiong）| 2026-01（arXiv:2510.06534v3，cs.AI）| 主题线 L4 Agent/工具/多轮自进化·相关性 高

**原始论文**：https://arxiv.org/abs/2510.06534

## 一眼看懂
- 🟦 TL;DR：搜索 agent（多轮检索答题）要 RL 训得好，得看 base 模型有没有特定推理行为。本文先用 LLM pipeline 从"强模型成功 vs 弱模型失败"轨迹里提炼出**四种有益行为**（信息验证、权威评估、自适应搜索、纠错），再用 **Behavior Priming**——先 SFT 把这四种行为"种"进模型、再跑 GRPO。最炸的发现：用"展现这四种行为但答案错误"的轨迹做 SFT，RL 后效果 **≈** 用答案正确的轨迹（Table 4），说明**解锁 RL 的关键是推理行为(path)，不是 outcome 正确性**。
- 最巧的一步：**Behavior Prime (Incorrect) vs (Correct) 的对照消融（§5.2 Table 4）**——两组 SFT 数据行为频率都 100%、accuracy 分别 0%/100%，只变 outcome。抽掉这个干净的因果隔离实验，"行为>正确性"就只是相关而非因果论断；正是它把整篇的核心主张钉死。

## 为什么做
- 研究背景：Agentic search（LLM 多步检索解复杂信息需求：分解任务→多步搜→综合）商用已起飞（ChatGPT Deep Research、Google AI Mode），开源侧靠 RL 训练进展快（Search-R1、R1-searcher、DeepResearcher）。范式分两类：早期靠强模型轨迹蒸馏/SFT，近期主流端到端 online RL，不少用 SFT-then-RL 两段式先初始化（§2）。
- 解决的具体痛点：数学/代码已发现"**RL 成功高度依赖 base 是否已有特定推理模式**"（verification/backtracking，引 Gandhi 2025/Yeo 2025），但**几乎只研究数学**。对 agentic search，"哪些具体行为有益、如何系统培养"不清楚；且 agentic search 有独特挑战（海量结果中识别有用信息、解决来源冲突、长轨迹保持目标聚焦）。关键观察：non-primed 模型 RL 中**无法内生**长出这些行为（§7 Table 6：Direct RL 后多数行为频率反而降）。
- 相关工作 & 各自不足：① 多 agent 系统（预定义 workflow）vs 单 agent 端到端——后者更易端到端训，本文聚焦后者；② 训练靠 SFT 合成轨迹蒸馏 / online RL / SFT-then-RL 初始化——但**没人系统研究 agentic search 该种入哪些行为**；③ "RL 成功取决于 base 已有行为"的发现集中在数学，agentic 域空白。
- 动机链：现状（agentic search 靠 RL 但成功取决于 base 行为，且只在数学有结论）→ 缺陷（不知 agentic 该种什么行为 + non-primed 模型 RL 自己长不出来）→ 所以先"识别有益行为"再"SFT 种进去 + RL 精炼"。为什么不用 process reward 引导 RL 自己长？§7 试过（R=R_outcome+0.1×N，N=行为数）——行为频率确实升但性能**反而比标准 RL 差**，因为模型 reward hack 模仿表面模式而非掌握精髓。
- 与最近邻工作的Δ：vs 数学域"cognitive behaviors enable self-improving reasoners"（Gandhi 2025）——把"base 行为决定 RL 上限"从数学**迁移并系统化到 agentic search**，并新增"错误轨迹消融"证明是行为而非正确性；vs 普通 SFT-then-RL（Distillation 随机采样 / Outcome-driven 只筛正确轨迹）——本文按"是否展现全部四行为"筛，且证明它**超过专门筛正确轨迹的 Outcome-driven**。关键点：SFT 阶段该优化的目标不是"答对"而是"展现好行为"。

## 怎么做 + 靠不靠谱
- 方法流水线：**第一部分识别行为**——① 建标准 agentic search 框架（每步出 reasoning+action，action∈{search, answer, **summary**(压历史管上下文)}）；② 配对轨迹：Gemini 2.5 Flash(强)成功而 Qwen3-1.7B(弱)失败的 500 题；③ 三阶段 LLM pipeline：轨迹比较→行为抽取（用 Gemini 2.5 Flash）→行为合并去重（Gemini 2.5 Pro）+ 人工复核 → 得**四行为**（Information Verification 跨源验证引证 / Authority Evaluation 识冲突评可信度 / Adaptive Search 动态调策略 / Error Recovery 识别并纠正先前错误）；④ 跨 Gemini/DeepSeek-R1/Llama/Qwen 测"行为频率与性能强相关"（Fig.2）。**第二部分 Behavior Priming**——⑤ SFT：Gemini 2.5 Flash 每题采 10 条，筛"同时展现全部四行为"的轨迹，**每一步当独立训练样本** D_SFT={(x_k,y_k)}；⑥ RL：primed 模型上跑 GRPO，按 step 聚合更新，**outcome 二值奖励**（LLM-judge 判最终答案对/错，R 和 Â 对轨迹内所有 step 恒定）。
- 逐组件必要性：**四行为 vs 单行为**——§5.4 Table 5：IV-only-10k 比 Direct RL 好但被全四行为 Behavior Prime-10k 一致超过（证明复合行为协同必要）；**SFT 种行为 vs RL 自己长**——§7 Table 6：Direct RL 后行为频率多数下降（证明需显式先验）；**行为 vs outcome**——§5.2 Table 4 核心消融（见上）；**SFT 数据规模**——§5.3 Fig.3：5k/10k/20k 递增但 WebWalkerQA 在 5k 后 plateau（行为一旦学会，额外 priming 收益递减）。消融较完整。
- 关键机制/公式（直觉）：无新损失，GRPO 标准目标（§4.2）。机制解释靠**训练动态**：SFT 后行为频率↑、pass@8↑、平均步数↑（Fig.1/Fig.4）；RL 中 primed 模型**维持更高 policy entropy、不早崩**，Direct RL 熵骤降早收敛（Fig.5a）——更高熵=更丰富探索=更高 RL 上限。另一发现：primed 模型 valid-action-ratio 反而不太稳（Fig.5c），说明增益来自推理行为而非熟悉工具语法。
- 实验与证据：backbone Qwen3-1.7B、Llama3.2-3B-Instruct；评测 3 个 web agent benchmark（GAIA 103 文本例、WebWalkerQA、HLE）+ 7 个 multi-hop QA（NQ/TQ/HotpotQA/2Wiki/MuSiQue/Bamboogle/PopQA）。**关键数字**：相对 Direct RL，web **+37.2%**、QA **+6.2%**（相对提升，Table 2/3）；一致超 Distillation/Outcome-driven 两基线；Table 4：Incorrect 组 RL 后 Overall 23.5 ≈ Correct 组 23.6（SFT 后分别 9.3 vs 11.3，差距 RL 后抹平）。baseline 公平：两 SFT-then-RL 基线从**同一 Gemini 语料**选轨迹，仅选择策略不同。
- 假设与失效边界：【原文】(§A) SFT 3 epoch bs8；RL 在 verl-agent 上 300 步、bs32、GRPO group8、8×H100 约 20h；评测 temp0、pass@k temp1，GPT-4o-mini 评分。【推断】**规模偏小**（1.7B/3B），更大模型是否仍需显式先验未知；四行为由 **LLM-judge 识别与筛选**，整套结论依赖判别 prompt 质量与 Gemini 判断一致性（用 LLM 定义"好行为"又用 LLM 评是否展现，存循环依赖）；web 增益大(+37%)、QA 温和(+6.2%)说明对任务类型敏感。
- 祛魅总结：【推断】真贡献是 **"path 监督 > outcome 监督"这一可迁移结论的强因果证据**（错误轨迹消融干净）+ 把数学域发现系统迁移到 agentic search + process-reward 失败的反面教训。被包装/张力的是：**SFT 阶段是 path 监督、RL 阶段又退回 outcome 级 GRPO**（无 step/turn 信用分配），与"path 重要"的主张存在内部张力——它只在 SFT 喂 path，RL 仍靠 outcome reward；"行为定义"高度依赖特定 LLM-judge，可迁移性未充分检验。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=四种推理行为的轨迹示范（SFT，LLM-judge 筛选）+ outcome 二值奖励（RL）｜**改什么**=参数（SFT 全量微调 + RL 更新策略）｜**何时改**=离线两段式（先 SFT 种行为 → 再 GRPO 在线精炼）｜**免梯度?**=否｜**记忆-技能生命周期**=部分相关——"行为"≈可复用技能，但只写入参数，无外部技能库/检索/共享；summary action 是轻量上下文管理（非持久记忆）｜**防遗忘机制**=无显式机制（SFT→RL 顺序，未处理灾难性遗忘）
- ⑦ 开源代码+框架/harness：主仓库 https://github.com/cxcscmu/Behavior-Priming-for-Agentic-Search （旧 analysis 已核：旧 KEYS 的 `cxcscmu/Behavior-Priming` 是 404，此为正确仓库名）。**SFT 框架=LLaMA-Factory**（vendored，`scripts/train.sh`→`llamafactory-cli train`）；agent rollout/评测靠 vLLM；**RL 框架=GRPO，在独立仓库** https://github.com/cxcscmu/verl-agent-deepresearch （veRL 系，verl-agent 的 deep-research 变体），主仓库**不含 RL 训练代码**。harness=自建 agentic search 框架（search/answer/summary 三动作，Appendix B 给完整 prompt）。
- 💰 资源/成本与可扩展性：【原文】RL 单次运行 8×H100 约 20h；SFT 数据 web 20k steps / QA 10k steps（步级样本）；每题采 10 条 Gemini 轨迹做语料。
- 🎯 对"探索-巩固"对标：**强支撑（核心对标论文之一）**。判定：本文 = TSRD"teacher 当稀疏脚手架→student 内化→RL"的**直接经验先例**——四行为里 **Error Recovery（识别并纠正先前错误）几乎就是 path-recovery/回轨**、Adaptive Search≈选路调整；"path 监督 > outcome 监督"为 TSRD"教选路+回轨比教对错更重要"提供**最直接的因果证据**；"primed 模型 RL 中维持更高熵=更好探索"支撑"巩固有效行为后探索空间更大"。**可借组件**：① 用强弱轨迹对比 + LLM pipeline 自动提炼"有益行为/path"的方法论；② 把单步当独立样本的 step 级 SFT。**缺口/竞品面**：RL 仍 outcome 级无 step 信用分配（TSRD 要更细的 path-recovery 单点接管）；行为靠 LLM-judge 识别而非 logit/MTP 前瞻；无 teacher 在线脚手架（teacher 只在离线造 SFT 数据）。
- 🔭 开放问题/未来方向：【原文】"targeted post-training 增强推理行为以提升 RL 有效性"是有前景方向（§8）；探讨过 process reward 作替代但失败（§7）。【推断】更大模型上的必要性；行为识别去 LLM-judge 化（用更客观信号）；把 path 监督贯穿到 RL 阶段（step/turn 级信用分配）而非 SFT 后退回 outcome；teacher 在线脚手架而非仅离线造数据；与 MTP 前瞻结合识别"该回轨"的时点。

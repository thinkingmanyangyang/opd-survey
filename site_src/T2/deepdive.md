# deepdive — DeepDive: Advancing Deep Search Agents with Knowledge Graphs and Multi-Turn RL

> **一句话重点 (TL;DR)**：DeepDive 用知识图谱（KG）随机游走 + 属性模糊化自动合成"难找"deep-search QA，再用端到端多轮 GRPO（带 redundancy penalty 抑制重复查询）训练浏览 agent，DeepDive-32B 在 BrowseComp 上达 15.3%（KG 数据），加半自动 i.i.d. 数据可达 22.2%。

**元信息**：arXiv 2509.10446（v2 2025-10-14，under review）｜ 清华 / Z.AI / 东北大学（Rui Lu*, Zhenyu Hou*, Zihan Wang* 等；Jie Tang、Yuxiao Dong）｜ 主题 T2（deep search agent / 多轮 RL），与 mtp_opd 外围相关（纯多轮 RL 非蒸馏，可作长 horizon 工具调用 agent RL 对照；redundancy penalty / test-time scaling 对 path 多样性有参考价值）｜ 代码 https://github.com/THUDM/DeepDive（仅 KG 数据合成，**不含 RL 训练代码**）｜ 框架 slime（RL）+ 多轮 GRPO

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/deepdive/fig_01.png)

*Figure 1: Left : Adding the redundancy penalty reduces tool call counts during RL training. Middle : DeepDive drives the model's deep search ability with maximum tool calls, which improves performance on BrowseComp. Right : Multi-turn reinforcement learning consistently enhances DeepDive-32B on four*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/deepdive/fig_04.png)

*Figure 4: Overview of multi-turn RL in DeepDive.*

## 1. 相关工作与进展
给 LLM 加浏览工具可成为 deep search agent，对数百在线来源推理检索以定位复杂、难找信息（如 BrowseComp 题目）。R1-Searcher / ReSearch / DeepResearcher 等开源工作主要在 HotpotQA 类多跳 QA 上训练评测；OpenAI DeepResearch 等专有系统遥遥领先。GLM-4.5/4.6 采用了 DeepDive 的数据并贡献其 BrowseComp 强表现。

## 2. 现有工作存在的问题

- **数据太简单**：HotpotQA、2Wiki、Bamboogle、Musique 等靠搜几个清晰实体即可解，不反映真正"hard-to-find"案例；而 BrowseComp 含多个模糊实体（blurry entity）、需长 horizon 推理 + deep search。
- **训练是开放问题**：如何把长 horizon 推理与 deep search 工具使用有效结合仍未解决；即便 DeepSeek-R1 也只做浅层工具调用且易幻觉。

## 3. Motivation
(1) 自动从开放 **知识图谱** 合成复杂、难找问题，补足难数据；(2) 用端到端多轮 RL 提升"长 horizon 推理 + deep search"；(3) 用 redundancy penalty 鼓励多样、减少冗余重复查询。

## 4. 主要灵感 / 核心直觉
KG 天然支持多跳连接、每实体多属性——沿 KG 随机游走可抽出长多跳路径，故意模糊部分属性即可制造"blurry entity"，从而构造刺激长 horizon 推理与 deep search 的难题；同时 deep search 中重复相似查询是浪费，可用查询相似度直接惩罚。

## 5. 主要解决思路(一段话讲清核心)
KG 难数据合成提供训练信号，端到端多轮 GRPO 让 LLM 在真实 web 环境中边搜边推、按最终答案给奖励，redundancy penalty 用 Jaccard 相似度惩罚重复查询以提升搜索多样性与效率。

## 6. 方法详解(通俗、分步骤)

1. **KG 难数据合成**：先按出度区间 [d_min, d_max] 过滤候选节点（出度过高→答案过于流行可预测；过低→难扩展路径）；在 KG 上随机游走抽长多跳路径；再用 LLM 进一步混淆关键线索形成 blurry entity。i.i.d. side study 随机游走参数：k∈[5,9]、d=3、d_min=4、d_max=8（核自 §4），实体混淆用 Gemini-2.5-Pro。
2. **端到端多轮 RL（multi-turn GRPO）**：LLM 与 web 环境交互，按构造 QA 的最终答案给奖励；为促探索保留 KL penalty。
3. **Redundancy penalty**：用 Jaccard 相似度度量任意两次查询的重复程度并惩罚（系数 λ=0.1），鼓励多样高效搜索（图1左：显著减少 RL 训练中工具调用冗余计数）。

## 7. 实验数据集

- 评测：**BrowseComp、BrowseComp-ZH、SEAL-0、Xbench-DeepSearch**（四个 deep search benchmark）。
- 训练数据：摘要/引言称构造 **3,090** 条 KG 自动合成 deep search QA；§实验细则述随机游走流程实际产出 **3,250** 条，随机切分为 **1,016 SFT + 2,234 RL**（RL 用全部 2,234）。另含半自动 i.i.d. 合成数据（side study，用 Gemini-2.5-Pro 混淆实体）。〔3,090 vs 3,250 为论文内部叙述口径差异，已核。〕

## 8. 实验结果与主要发现

- 基模型：GLM-Z1-9B-0414 与 QwQ-32B。SFT 阶段 3 epochs、global batch 32、lr=1e-5、max context 104,800；RL 用 slime、全部 2,234 样本，rollout size=8、每 prompt 16 samples（GRPO），redundancy penalty λ=0.1。训练期 checkpoint 选择用 BrowseComp-266（从 1,266 题随机预采样子集），turn limit 提至 128。
- 结果：DeepDive-32B 在 BrowseComp 达 **15.3%**（KG 数据），超过 WebSailor、Search-o1、DeepSeek-R1-Browse；加半自动 i.i.d. 数据进一步到 **22.2%**。多轮 RL 在四个 benchmark 一致提升（+5.8/+6.7/+1.6/+3.3% 等）。展示工具调用与并行采样的 **test-time scaling**（增大 max tool calls 提升成功率）。做了 n-gram 污染分析（表4）。其数据被 GLM-4.5/4.6 采用。

## 9. 结果如何支撑其主张
四个 benchmark 一致正增益支撑"多轮 RL 有效"；redundancy penalty 的工具调用计数下降图（图1左）支撑"减少冗余"；test-time scaling 曲线支撑"更多工具调用→更高成功率"；n-gram 污染分析回应数据泄漏质疑，间接支撑结果可信度。

## 10. 逻辑自洽性(中性评估)
数据合成→RL 训练→多样性惩罚三段逻辑连贯，benchmark 选择（BrowseComp 系）契合"hard-to-find"主张。但部分增益（如 Xbench +1.6%）幅度较小，且 BrowseComp 绝对值仍偏低（15.3%/22.2%），说明任务远未解决；checkpoint 选择用 266 子集存在一定调参–评测耦合风险。

## 11. 残留问题 / 局限

- 训练数据规模小（RL 仅 2,234 样本），合成数据多样性与难度分布对结果影响未充分隔离。
- BrowseComp 绝对准确率仍低，距专有 DeepResearch 有差距。
- RL 训练代码不在仓内（依赖外部 slime），独立复现需自行接入。
- 3,090 / 3,250 的数据计数口径在论文内不一致（摘要 vs 实验细则），需以实验节为准。

## 12. 开源代码与框架(链接+框架+代码可得性)

- 仓库：https://github.com/THUDM/DeepDive （已 clone，~2.9MB；含 qa_synthetic/（KG 数据合成）、assets/）。**仅含 KG 数据合成代码，不含 RL 训练代码**（训练在外部框架完成）。
- RL 框架：**slime**（THUDM/slime，"LLM post-training framework for RL scaling"，Zhu et al. 2025）；论文 §实验设置明确"we conduct training using the open-source Slime framework with all 2,234 data samples"。多轮 **GRPO** 算法。
- 数据/模型公开：3,090（KG 自动）+ 2,234（RL）+ 半自动 i.i.d. 合成数据；DeepDive 模型。

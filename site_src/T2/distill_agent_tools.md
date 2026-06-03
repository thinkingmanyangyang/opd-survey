# distill_agent_tools — Distilling LLM Agents into Small Models with Retrieval and Code Tools (Agent Distillation)

> **一句话重点 (TL;DR)**：不只蒸馏教师的"推理"，而是把教师 agent 的完整"think + act（检索/代码工具）"任务求解行为蒸馏到小模型；配两项改进——用 first-thought prefix 提高教师轨迹质量、用 self-consistent action generation 提高学生测试时鲁棒性——使小模型 agent 能匹配甚至超过比它大 2–4× 的 CoT 蒸馏模型。

**元信息**：arXiv 2505.17612 (v2, 2025-11-05) ｜ KAIST / DeepAuto.ai（Minki Kang*, Seanie Lee, Sung Ju Hwang）、UW-Madison（Jongwon Jeong*）、KRAFTON（Jaewoong Cho）｜ NeurIPS 2025 ｜ 主题 T2（agent/tool 蒸馏，离线 SFT 式，非 on-policy，可作 OPD-agent 对照基线）｜ 代码 https://github.com/Nardien/agent-distillation（已克隆约 7.6MB）｜ 框架 smolagents v1.13.0.dev0 + TRL SFT trainer + LoRA；检索环境沿用 Search-R1（pyserini + faiss）。

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/distill_agent_tools/fig_01.png)

*Figure 1: Performance comparison of different sizes of Qwen2.5-Instruct models [1] on the average accuracy of four factual reasoning tasks (HotpotQA [2], Bamboogle [3], MuSiQue [4], 2WikiMultiHopQA [5]) and four mathematical reasoning tasks (MATH [6], GSM-Hard [7], AIME [8], OlymMATH [9]). Distillat*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/distill_agent_tools/fig_03.png)

*Figure 3: (a) First-thought Prefix: We prompt teacher with a CoT prompt to induce step-by-step reasoning. The first reasoning step is used as a prefix to generate an agentic trajectory, which is then distilled to a student agent to teach CoT-style reasoning initialization. (b) Self-consistent Action*

## 1. 相关工作与进展

- **CoT 蒸馏**：用教师 LLM 的 Chain-of-Thought trace 通过 next-token 蒸馏把推理能力迁移到小模型（sLM）。
- **检索增强（RAG）**：在蒸馏与推理时引入外部知识，作为公平对比基线。
- **CodeAct 范式**：每步由 Thought–Action（Python 代码）–Observation 组成，本文 agent 形式化沿用之。
- **prefix-attack（jailbreak）**：往模型回答前缀注入内容来引导生成，是 first-thought prefix 的直接灵感来源。
- **majority voting / self-consistency**：对多次采样结果投票提升鲁棒性。

## 2. 现有工作存在的问题

- 纯 CoT 蒸馏在需要**罕见事实知识**或**精确计算**时失效——小模型容易幻觉、算错（如"$100 投资 Apple 2010→2020 值多少"需查历史股价、拆股，再精确计算）。
- 静态推理 trace 模仿无法在测试时获取新知识或保证计算正确。
- 直接让 instruction-tuned 教师（Qwen2.5-32B-Instruct）当 agent 时，其**初始推理质量不佳**，生成的轨迹不够好。
- 小蒸馏 agent 常生成**无效/不可解析的动作**（代码报错、库函数误用），阻碍与环境交互。

## 3. Motivation
迁移的不应只是"推理"，而是 LLM agent 的**完整任务求解行为**（think + act）：让小模型学会用检索工具获取事实、用代码工具精确计算，从而对幻觉更鲁棒、对 OOD 任务泛化更强。核心问题：如何在更小模型中保留 LLM 级问题求解能力。

## 4. 主要灵感 / 核心直觉

- **教师轨迹质量瓶颈在"第一步思考"**：instruction-tuned 教师直接做 agent 时开头推理弱；借 prefix-attack 思路，先用 CoT prompt 诱导教师生成 step-by-step 推理，把这"第一步思考"作为前缀注入教师 agent → 抬高整条轨迹质量。学生推理时不需要该前缀。
- **学生测试时靠采样+投票纠错**：对每步采多条 thought-action（高温 nucleus 采样增多样性），无效动作可由其报错 observation 引导自纠，并对结果 observation 做 majority voting，降低单次无效动作的影响。

## 5. 主要解决思路（一段话讲清核心）
用 first-thought prefix 增强后的教师（Qwen2.5-32B-Instruct）在 smolagents 中对训练问题各采一条"Thought-Action(code/search)-Observation"轨迹，过滤掉答错的，得到约 2,000 条轨迹；用 TRL 的 SFT trainer + LoRA 把这些 agent 轨迹蒸馏进 Qwen2.5-Instruct 小模型（0.5B/1.5B/3B/7B）；测试时小 agent 配 self-consistent action generation（每步采 N=8、温度 0.4，投票），得到实用的 tool-using 小 agent。

## 6. 方法详解（通俗、分步骤）
**Agent Distillation 框架，两条互补改进轴：**

1. **First-thought prefix (ftp)**：用 CoT prompt 诱导教师产生 step-by-step 推理，截取其作为前缀注入教师 agent 的"第一步思考"，再让教师在 smolagents 中续走 reason-act-observe。只用于改善教师轨迹采集，学生推理时不需要。
2. **Self-consistent action generation (sag)**：测试时不用 greedy，对每一步用 nucleus 采样以高温采 N 条 thought-action（主实验 N=8、温度 0.4）；无效动作的报错 observation 反馈回去供后续步自纠；对各条得到的 observation 做 majority voting 选最终动作结果。

**流程**：教师=Qwen2.5-32B-Instruct；学生=Qwen2.5-Instruct 四个尺寸（0.5B/1.5B/3B/7B，蒸馏前已 instruction-tuned）。每题从教师采 1 条轨迹并过滤错误轨迹 → 约 2,000 条训练数据 → LoRA（rank 64，所有线性层）SFT 学生（2 epoch、batch 8、lr 2e-4、4×A100 80GB）→ 测试时学生 agent 配 sag、max steps=5、greedy 主解码。

## 7. 实验数据集

- **训练数据**：1,000 HotPotQA + 2,000 MATH examples（每题采 1 条教师轨迹、过滤错误后约 2,000 条用于蒸馏）。〔已核：训练源含 HotPotQA 与 MATH 两部分，非仅 MATH〕
- **检索环境**：Wikipedia 2018 作知识库（agent 与 RAG 共用），e5-base-v2 做 doc/query embedding；沿用 Search-R1 的 retriever。
- **评测（8 任务，每集限 500 例）**：
  - 事实多跳 QA：HotpotQA（in-domain）、MuSiQue、Bamboogle、2WikiMultiHopQA（OOD）。
  - 数学：MATH500（in-domain）、GSM-Hard、AIME、OlympiadMATH（OOD）。
  - 指标：数学用 exact match；事实用 LLM-as-judge（gpt-4o-mini）。

## 8. 实验结果与主要发现

- **agent 蒸馏全面优于 CoT 蒸馏**，尤其在 OOD 任务上。蒸馏前除 7B 外多数尺寸靠 prompting 无法产生有效 agentic 输出（常出不可解析代码）。
- **跨档匹配**（论文主张）：0.5B agent ≈ 1.5B CoT 蒸馏；1.5B agent ≈ 3B CoT；3B agent 超过 7B CoT；7B agent 甚至超过 32B CoT 教师。
- **逐项数值（Table 2 摘选，Avg. 为 8 任务均值）**：
  - 教师 32B：CoT Prompting 39.54；Agent Prompting 46.00。
  - 7B 学生：CoT Distill 33.54 → Agent Distill 39.85 → +ftp 42.26 → +sag 41.86 → +ftpsag 42.68。
  - 3B 学生：CoT Distill 27.72 → Agent Distill 33.60 → +ftpsag 36.60。
- ftp 与 sag 在 0.5B–7B 全尺寸、跨两域基本稳定提升（但 3B+sag 在 AIME 上掉到 0.0，说明 sag 单用并非处处增益）。

## 9. 结果如何支撑其主张

- "迁移工具使用行为提升泛化" → agent 蒸馏在 4 个 OOD 任务上相对 CoT 蒸馏增益最明显，与"测试时可查事实/算数→对幻觉更鲁棒"的机制一致。
- "小模型匹配更大 CoT 模型" → 表中 7B Agent(ftpsag) 42.68 > 32B CoT 39.54，3B Agent(ftpsag) 36.60 > 7B CoT 33.54，支撑跨档主张。
- ftp/sag 的增量贡献由"Distill → +ftp/+sag/+ftpsag"逐行递增体现，但并非每个单元格都单调（sag 在个别难任务上可能反伤），说明组件增益是平均意义上的。

## 10. 逻辑自洽性（中性评估）
框架自洽、实现干净：ftp 只动教师数据采集、sag 只动学生测试，互不耦合，消融清晰。但"跨档匹配"这种比较高度依赖平均分聚合（8 个异质任务直接平均），单任务上结论不稳（如 AIME 这类小样本基准方差大、出现 0.0/15.6 的剧烈跳变）。sag 用 majority voting 实质是 test-time 多次采样，提升来自额外推理算力而非模型本身能力——与"高效小 agent"的卖点存在一定张力（推理成本被转移到 test-time）。整体是合理的工程组合，机制解释成立但增益的统计稳健性有限。

## 11. 残留问题 / 局限

- **离线 SFT、非 on-policy**：纯模仿教师轨迹，未做 on-policy/RL 矫正，学生在工具调用上的分布漂移未被训练时纠正。
- **sag 的 test-time 成本**：每步采 N=8 条，推理开销显著上升，"小模型省算力"的优势部分被抵消。
- **小样本基准方差大**：AIME/OlympiadMATH 等任务样本少，Avg. 聚合掩盖了单任务的剧烈波动（如 3B+sag AIME=0.0）。
- **依赖外部判分**：事实任务用 gpt-4o-mini 做 LLM-as-judge，引入评测器偏差。
- **教师/检索环境固定**：仅 Wikipedia 2018 + 单一 retriever，真实开放检索下的表现未验证。

## 12. 开源代码与框架（链接+框架+代码可得性）

- 仓库 https://github.com/Nardien/agent-distillation（已克隆，约 7.6MB）。顶层含 `src/smolagents/`（fork 的 smolagents）、`data_processor/{math_dataset,qa_dataset}`、`exps_research/first_thought_prefix`、`scripts/{training,inference}`、`search/`、`serve_vllm.py`、`e2b.toml`。
- 框架：基于 **smolagents v1.13.0.dev0**（logging→可训练轨迹 / training / benchmarking）；训练用 **TRL** SFT trainer + **LoRA**（rank 64）；检索沿用 **Search-R1**（pyserini + faiss-gpu）；代码工具走本地或 e2b 沙箱。
- 资产：HF `agent-distillation/*` 提供教师轨迹（2k baseline 与 first-thought-prefix 两版）与 agent-distilled Qwen2.5-1.5B-Instruct 等模型。
- 代码可得性：高（框架、数据处理、训练/推理脚本、轨迹与模型权重齐全；README TODO 标注 ftp 详细说明待补）。

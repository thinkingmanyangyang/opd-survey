# latent_agents — Latent Agents: A Post-Training Procedure for Internalized Multi-Agent Debate

> **一句话重点 (TL;DR)**：用"SFT 学辩论结构 + GRPO 把显式辩论压进潜空间"的两阶段微调，把多个 agent 多轮辩论（multi-agent debate）内化进单个 LLM（IMAD），用 Debate 6.3%~21.1% 的 token（5–16× 提效）匹配/超过显式辩论；并发现内化后存在线性可分的"agent 子空间"，可做行为控制。属概念验证级（944 条算术 trace、LoRA）。

**元信息**：arXiv 2604.24881v1（2026-04-27）｜ Boston University（John Seon Keun Yi, Aaron Mueller, Dokyun Lee）｜ ACL 2026 Oral（据 README；论文 PDF 内未见该字样，〔待核-以 README 为准〕）｜ 主题 多智能体辩论的内化/蒸馏，与 TSRD/OPD 间接相关（"内化昂贵外部过程进单模型"的 SFT→RL 两阶段范式、length-annealing 把显式长推理压成隐式潜推理）；不涉及 MTP，非 teacher-student logit 蒸馏，奖励为 outcome+format 非 path 监督｜ 代码 https://github.com/johnsk95/latent_agents ｜ 框架 TRL (GRPOTrainer) + PEFT/LoRA。

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/latent_agents/fig_03.png)

*Figure 3: Agent behavior steering suppressed malicious traits with less damage to task performance. While both models display suppression of evil and hallucination traits when applying negative steering (solid lines), IMAD is more resistant to performance drops when steering at higher positive and n*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/latent_agents/fig_01.png)

*Figure 1: Overview of the Internalized Multi-Agent Debate (IMAD) Pipeline. 1. We first collect a debate dataset using the standard multi-agent debate protocol on an arithmetic task. Using this dataset, a single LLM agent is trained via supervised fine-tuning to learn the debate structure. The same a*

## 1. 相关工作与进展
Multi-agent debate（Du et al. 2023; Liang et al. 2024）通过多模型多轮对话降幻觉、提升事实准确性。DebateGPT（Subramaniam et al. 2024）仅用最终 consensus 输出蒸馏。length-pruning 借鉴 ThinkPrune（Hou et al. 2025，代码注释明示）。

## 2. 现有工作存在的问题
- 显式辩论 token 开销巨大（多模型多轮 transcript 才给答案）。
- 仅蒸馏最终 consensus（DebateGPT）token 最省但性能普遍不如显式辩论，因丢掉了驱动增益的中间交互。
- 缺乏对"内化后辩论结构是否仍保留、能否被控制"的机制性理解。

## 3. Motivation
用一次性微调投入，换单模型的效率 + 多智能体辩论的推理能力；并探究内化后模型是否学到可恢复的"每个 agent 的表示"，进而用于行为控制（如抑制恶意 agent）。

## 4. 主要灵感 / 核心直觉
辩论的增益来自中间多视角交互而非仅最终结论，所以要在完整 trace 上学结构；学会结构后，用"格式奖励衰减 + 长度上限收缩"逼模型把多视角分析从显式文本转入潜空间，直接产答案。内化后不同 agent 的"声音"应在表示空间留下线性可分的痕迹。

## 5. 主要解决思路(一段话讲清核心)
三阶段 IMAD：(1) 用标准 multi-agent debate（n=3 agents, m=2 rounds, GPT-3.5-turbo 当 agent）在算术题上生成 944 条带结构标签的辩论 trace；(2) 在完整 trace 上做 next-token CE 学辩论格式（SFT）；(3) GRPO 内化，奖励 `r=w_fmt·R_fmt+w_clip·R(y;l)`，R_fmt 为结构标签匹配的格式奖励（权重 w_fmt 随训练 1.0→0.05 衰减），R(y;l) 为"正确答案出现在前 l token 内记 1"的长度裁剪奖励（l 随训练 2000→500 退火），两者协同迫使模型把分析压进潜空间。

## 6. 方法详解(通俗、分步骤)
1. **数据收集**：标准 debate（n=3, m=2, GPT-3.5-turbo）在 6 个两位数表达式算术题上生成 transcript；过滤无 majority consensus 的；加结构标签 `<|Agent 1|>`/`<|Round 1|>`/`<|Consensus|>`/`<|endofdebate|>`；共 **944** 条 {Question, Trace, Answer}。
2. **Debate Structure Learning（SFT）**：在完整辩论 trace（非仅最终输出）上做自回归 CE。
3. **RL for Internalization（GRPO）**：`r=w_fmt·R_fmt+w_clip·R(y;l)`；w_fmt 1.0→0.05 衰减、l 2000→500 退火，把多视角分析转入潜空间直接产答案。
4. **机制分析**：用 difference-in-means 提取 agent-specific steering vector，发现内化产生线性可分的 agent 子空间；对恶意 agent 子空间做 negative steering 可在保任务性能下抑制有害行为，且内化后比直接 steer base model 更有效。
- 两阶段均用 LoRA；GRPO 阶段从 SFT 的 LoRA checkpoint 起再叠一层 LoRA。

## 7. 实验数据集
- 训练：仅 944 条算术题辩论 trace。
- 评测：GSM8K（多步数学）、MMLU-Pro（多领域多选）、BigBench Hard（多样推理）；各随机采 1000 题，三次运行报均值±标准误。训练仅算术、评测跨域跨格式，测泛化。

## 8. 实验结果与主要发现
- **效率**：所有模型上 IMAD 仅用 Debate 的 **6.3%~21.1%** token（5–16× 提效）。
- LLaMA-3.1-8B-Instruct 上三个 benchmark 全面超 Debate；Mistral-Nemo-12B 在 GSM8K 上超 Debate **18.97** 个百分点；Qwen2.5-7B 增益温和。
- 仅算术训练却能跨域泛化（附录另做多任务扩展数据集，性能更强）。
- DebateGPT（仅最终输出蒸馏）token 最省但性能不及 IMAD，印证保留中间交互的价值。

## 9. 结果如何支撑其主张
"内化保性能+提效"由三 benchmark 上 IMAD 以极少 token 匹配/超 Debate 支撑；"中间交互重要"由 IMAD 超 DebateGPT 支撑；"agent 子空间可控"由 difference-in-means steering 实验（negative steering 抑制恶意 agent、内化后比 base 更有效）支撑。

## 10. 逻辑自洽性(中性评估)
两阶段范式与奖励设计逻辑清晰，机制实验为"内化"提供了表示层证据。但整体属概念验证：训练数据极小（944 条、仅算术、n=3/m=2 最小辩论），增益跨 backbone 差异大（Qwen 仅"温和"），机制结论主要靠 steering 实验单一证据链支撑，泛化主张依赖附录扩展数据。

## 11. 残留问题 / 局限
- 训练数据与规模都很小（944 条算术 trace、LoRA、n=3/m=2），结论可扩展性存疑。
- 增益在不同 backbone 间差异显著（LLaMA/Mistral 强、Qwen 弱），未充分解释。
- 教师辩论用 GPT-3.5-turbo，质量天花板受限；机制结论（agent 子空间可控）证据链较窄。
- ACL 2026 Oral 据 README，论文 PDF 内未见该字样，〔待核〕。

## 12. 开源代码与框架(链接+框架+代码可得性)
- 链接：https://github.com/johnsk95/latent_agents （README 写 ACL 2026 Oral）。含 `sft.py`、`grpo.py`、`grpo_persona.py`、`steering/`、`eval/`、`data/`、`utils/generate_arithmetic*.py`。
- 框架：TRL（`GRPOConfig`/`GRPOTrainer`，trl>=0.11）+ PEFT/LoRA（peft>=0.13）+ transformers>=4.46 + accelerate；length-pruning 注释借鉴 ThinkPrune。
- Backbone：LLaMA-3.1-8B-Instruct、Qwen2.5-7B、Mistral-Nemo-12B。SFT 3–6 epoch、GRPO 2 epoch。代码可得、流程可复现。

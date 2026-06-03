# glm45 — GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models

> **一句话重点 (TL;DR)**：开源 MoE（355B 总 / 32B 激活）混合推理模型（thinking + direct 双模式）。后训练分两阶段——先分域训三个专家（Reasoning/Agent/General-chat，各 cold-start SFT + 专家 RL），再用 **self-distillation** 把多专家统一进一个通才；agent RL 中用 **iterative (self-)distillation** 在昂贵 RL 之间快速抬升起点。

**元信息**：arXiv 2508.06471（v1, 2025-08-08）｜ Zhipu AI 智谱 & 清华 ｜ 2025-08 ｜ 主题 T1/T3，相关性 High（多专家→统一的 self-distillation/迭代蒸馏范式；去 KL 的 GRPO + 难度课程）｜ 代码 https://github.com/zai-org/GLM-4.5（模型/权重发布仓，**不含后训练训练脚本**）；RL 框架 **Slime** 单独开源 https://github.com/THUDM/slime ｜ 框架 Slime（Megatron 训练 + SGLang/Router rollout + Data Buffer）。

> 注：这是**模型技术报告**而非单一方法论文；slime 是其 RL 框架（不等于完整后训练脚本，社区微调多用 LLaMA-Factory/ms-swift）。CloneTier=B，仅记录不 clone。

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/glm45/fig_01.png)

*Figure 1: Average performance on agentic, reasoning, and coding (ARC) benchmarks. Overall, GLM-4.5 achieves a rank of 3rd, with GLM-4.5-Air following at rank 6th. The models listed are evaluated as of July 28, 2025.*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/glm45/fig_03.png)

*Figure 3: Pre-training and mid-training stages for GLM-4.5. We adapt a multi-stage training recipe and extend the sequence length from 4K to 128K.*

## 1. 相关工作与进展

- 对标 o1/o3、Claude Sonnet 4、DeepSeek-R1、Kimi K2、Qwen3-235B 等；目标是单个开源模型同时擅长 Agentic / Reasoning / Coding（ARC）。
- 架构借鉴 DeepSeek-V3 / Kimi K2 的 MoE，但**减宽增深**（更少 routed experts、更多层），并用 2.5× 注意力头、QK-Norm、loss-free balance routing、partial RoPE。
- 含一层 **MTP（Multi-Token Prediction）MoE 层**用于推理时 speculative decoding（与本课题 MTP 主题有接触点，但此处仅用于加速解码，非 foresight 训练信号）。

## 2. 现有工作存在的问题

- RL 中模型能力随训练演化，与**静态训练数据失配**：后期太简单(reward 全 1)、早期太难(reward 全 0)，都缺 reward 方差→无梯度信号。
- 此前主张"多阶段渐增输出长度"的 RL，会让模型在短长度阶段"遗忘"长上下文能力，造成**不可逆**性能下降。
- Agent 任务 RL 耗时；function call 参数含代码时 JSON 转义负担重。

## 3. Motivation

- 用专家模型迭代（分域训练）+ self-distillation 统一，得到既能深思又能快答的混合推理通才；
- 用难度课程、单阶段长输出 RL、动态采样温度等技巧稳定高效地 scale reasoning RL；
- 用迭代自蒸馏在昂贵的 agent RL 之间快速抬升起点。

## 4. 主要灵感 / 核心直觉

- "先专后通"：单独训练的领域专家上限更高，再蒸成一个统一模型可兼得各域能力 + 双响应模式。
- RL 数据应随模型能力滚动调难度（课程），始终保持 reward 方差；长输出能力一旦在短阶段退化就难恢复，故宁可直接在目标长度训。

## 5. 主要解决思路(一段话讲清核心)
**Stage 1 专家训练**：分别构建 Reasoning / Agent / General-chat 专家，每个先 cold-start SFT（少量 extended-CoT）再专家 RL。**Stage 2 统一训练**：Overall SFT 收集各专家数百万样本（数学/代码/科学/chat/agentic/长上下文，ctx 最长 128K），平衡"带完整推理"与"无显式思考"数据 → 通过 **self-distillation** 得到 reflective + immediate 双模式混合推理模型。RL 全程基于 **去 KL 项的 GRPO**。

## 6. 方法详解(通俗、分步骤)
**Reasoning RL 技巧**：

- **两阶段难度课程**：第二阶段切到极难题（pass@8=0 但 pass@512>0，且仅取有验证正确答案的题），持续突破上限。
- **单阶段 64K 长输出 RL**：直接在 64K 目标长度做 RL，优于渐增长度的多阶段（后者不可逆掉点）。
- **动态采样温度**：reward 收敛时升温增多样性；用 held-out 验证集做质控（取性能跌幅<1% 的最大温度）。
- **Code/Science RL**：code RL 用 **token-weighted mean loss**（优于 sequence-mean，收敛更快、缓解长度偏置）；science RL 仅用少量专家验证的高质量选择题效果最佳（GPQA 65.8% vs 混合数据 62.9%）。

**Agent RL**：

- group-wise policy optimization（仅 model-generated token 入 loss，忽略环境反馈）；
- **process format penalty**：工具格式错则中止 trace 并给零奖励；
- **Iterative Distillation**：RL 训到 plateau 后，用 RL 模型输出替换 cold-start 数据生成更强 SFT 模型，再继续渐增难度 RL，循环抬升；
- 通过增加交互轮数实现 test-time scaling（BrowseComp 随轮数平滑提升）。

**General RL**：Holistic（rule + RLHF + RLAIF 混合反馈）、Instruction-Following、Function-Calling（step-wise rule-based + end-to-end multi-turn）、Pathology RL（针对语言混杂/重复/格式错）。
**function call 模板**：用 XML-like 特殊 token 包裹 key/value，大幅减少代码段转义负担（不损调用执行性能）。

## 7. 实验数据集

- **预训练 23T tokens**（前 15T 后调整数据权重；MTP loss 权重 0.3→0.1，bias 更新率 0.001→0），多阶段把序列长 4K→32K→128K。
- **后训练**：百万级 SFT 样本（数学/代码/科学/chat/agentic/长上下文）；RL 用规则可验证数学/代码/科学题（难度筛选）、agent 轨迹。
- **评测**：12 ARC 基准——TAU-Bench、BFCL V3、BrowseComp、AIME24、MATH-500、GPQA、HLE、LCB、SWE-bench Verified、Terminal-Bench、MMLU-Pro、SciCode。

## 8. 实验结果与主要发现

- TAU-Bench 70.1%、AIME24 91.0%、SWE-bench Verified 64.2%、BFCL V3 77.8%、GPQA 79.1%、BrowseComp 26.4%。
- 12 基准总排名第 3、agentic 第 2、coding 第 3，且参数远少于竞品（DeepSeek-R1 的一半、Kimi K2 的 1/3），位于 SWE-bench/参数 Pareto 前沿。
- **关键消融（小实验模型，非 GLM-4.5 本体）**：难度课程 AIME24 81.8%→83.4%；单阶段 64K（83.4%）优于多阶段（80.6%，且早期不可逆掉点）；code RL token-weighted mean 收敛更快。

## 9. 结果如何支撑其主张

- "专家迭代 + self-distillation 统一"由 ARC 全面强表现与高参数效率（Pareto 前沿）支撑；具体 RL 技巧由小模型对照消融支撑（难度课程、单阶段长输出、loss 计算、science 数据质量）。

## 10. 逻辑自洽性(中性评估)

- 工程报告型，技巧与动机对应清晰，关键设计都有消融。
- **重要保留**：几乎所有消融与对比曲线**在"smaller experimental model"上做**，非 GLM-4.5 本体，结论能否完全外推到 355B 本体未直接验证（作者已注明）。
- self-distillation 的具体损失形式（logit KL vs SFT 交叉熵）、数据配比等细节披露有限。

## 11. 残留问题 / 局限

- 后训练**训练代码不开源**（仅放权重 + 评测工具 + 通用 RL 框架 Slime），方法复现门槛高、细节缺失。
- 消融在小模型上，规模外推性存疑；多专家 self-distillation 的统一是否引入能力折损（相对各专家峰值）未量化。
- "去 KL 项的 GRPO" 在 355B 规模的稳定性细节、与 GSPO 等序列级算法的对比未给出。
- 报告自报 benchmark 为主，部分用 LLM 自动判分（HLE 用 GPT-4o、reasoning 用 LLM 验证），存在评测偏差风险。

## 12. 开源代码与框架(链接+框架+代码可得性)

- 模型/权重：https://github.com/zai-org/GLM-4.5 与 https://huggingface.co/zai-org/GLM-4.5（355B + Air 106B）；评测工具 https://github.com/zai-org/glm-simple-evals。
- RL 框架 **Slime**（自研开源）：https://github.com/THUDM/slime——三模块 Training(Megatron) / Rollout(SGLang+Router) / Data Buffer，支持 colocated 同步 与 disaggregated 异步、BF16 训练 + FP8 推理。
- **后训练训练脚本不在主仓**；Slime 是通用 RL 框架，不等于 GLM-4.5 完整后训练流程。CloneTier=B，仅记录不 clone。

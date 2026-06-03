# apple_ssd — SSD: Embarrassingly Simple Self-Distillation Improves Code Generation

> **一句话重点 (TL;DR)**：不用任何 teacher / verifier / reward / RL / 代码执行环境，只让模型在"调过温度 + 截断"的设置下采样自己的**原始未验证**输出，再用标准交叉熵 SFT，最后评估时单独调一个解码温度——就能把代码生成显著提升(Qwen3-30B-Instruct LiveCodeBench v6 pass@1 42.4%→55.3%)。值得看的点：它给"自蒸馏为何起作用"提出一个清晰机制——代码解码存在 **precision-exploration conflict**，SSD 通过上下文相关地重塑 token 分布(该压的地方压、该留多样性的地方留)拿到固定解码拿不到的增益。

**元信息**：arXiv 2604.01193v1 ｜ Apple（Ruixiang Zhang、Yizhe Zhang 等共一）｜ 2026-04 arXiv preprint ｜ 主题 自蒸馏(off-policy 纯 SFT) / 与 OPD **对照** ｜ 代码 https://github.com/apple/ml-ssd（**部分**：含数据生成 + 评测，SFT 训练本体不在仓库，README 注明为"复现")｜ 框架 仅需 采样 + 标准 SFT(交叉熵) + 评估时单独调温；无 RL/verifier/teacher/执行环境

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/apple_ssd/fig_01.png)

*Figure 1 Simple self-distillation (SSD) is embarrassingly simple, yet yields substantial LiveCodeBench v6 gains across five models spanning two families, three scales, with both instruct and thinking variants. Left: SSD samples from the base model with training-time decoding temperature T train, fin*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/apple_ssd/fig_02.png)

*Table 1 Comparison of training paradigms.*

## 1. 相关工作与进展(这条线现在做到哪一步)
LLM 编码任务越来越难，高质量监督信号成瓶颈：人工解贵；合成数据要么需要更强 teacher、要么需对每题做基于执行的验证。提升路径主要有：(1) **teacher 蒸馏**——继承 teacher 上限；(2) **RLVR**——操作复杂、可能不稳；(3) **基于内禀奖励的无监督自提升**(majority voting/熵最小化)——有早期成效但面临 reward hacking 与长训崩溃。SSD 处在"无外部信号自提升"这条线，但走的是最朴素的 off-policy 自蒸馏。

## 2. 现有工作存在的问题(本文针对的痛点)
核心问题：**能否让模型在完全不借助任何外部标注或验证的情况下自我提升？** 现有路径都需要 teacher / verifier / reward / 执行环境 / 人工标注解之一；无监督内禀奖励法又易 reward hacking、长训崩溃。

## 3. Motivation(为什么做这件事)
作者想要一个**最简、无任何外部依赖**的后训练方法，并搞清"自蒸馏到底为何 work"。他们把目光放到代码生成——因为代码的任务结构让底层机制**特别可见**，便于分析。

## 4. 主要灵感 / 核心直觉
关键直觉是 **precision-exploration conflict(精度-探索冲突)**：代码里同时存在两类位置——

- **fork 位置**：多个续写都合理(对应不同解法)，需要**多样性**(高温有利)；
- **lock 位置**：语法语义近乎唯一，但仍有一条低概率 **distractor(干扰)尾巴**，需要**精度**(低温有利、压住 distractor)。

两类位置对解码温度 Teval 的需求**相反**，所以任何**全局固定温度**都必然是折中。而 SSD 在"温度偏移 + 截断"样本上训练，能**按上下文**重塑分布：在 lock 处最狠地压 distractor，同时在 fork 处保留有用多样性——这是单纯改解码温度无法恢复的增益。

## 5. 主要解决思路(一段话把核心机制讲清)
三步：从冻结的 base 模型用非单位温度 Ttrain + 截断 τ 采样若干候选解；对这些**原始、未验证**的输出做标准交叉熵 SFT；评估时用一个**单独调过的**温度 Teval 解码。训练的净效果被形式化为对 token 分布的 **support compression(支撑压缩)** 与 **within-support reshaping(支撑内重塑)**(论文 Eq.4)，并用受控仿真 + 真实模型分析 + 理论(附录 B)支撑这一机制解释。

## 6. 方法详解(通俗、分步骤;关键公式用白话解释,必要时给伪代码)
**SSD 三步**：

1. **Sample**：冻结 base，以温度 Ttrain(非 1.0) + 截断配置 τ(top-k/top-p)对每个 prompt 采样 N 个候选解。
2. **Fine-tune**：对这 N 个**未经任何验证**的原始输出做标准交叉熵 SFT。
3. **Decode**：评估时用单独调过的温度 Teval 解码。

**机制白话**：在低温 + 截断下采样，等于把分布尾巴(包括 lock 处的 distractor)先削掉再当训练目标，于是 SFT 学到的新分布在 lock 处**自动**压住 distractor；而 fork 处因为本来就有多个高概率续写，截断削不掉它们，多样性得以保留。这就是"按上下文重塑"——比全局调温更精细。

## 7. 实验数据集

- 主基准：**LiveCodeBench v6**(按 Easy/Medium/Hard 分难度报告 pass@1 与 pass@5/coverage)；另含 **LCB v5**(374 题)。
- 模型(2 家族 × 3 规模 × instruct/thinking，共 5 个)：Qwen3-4B-Instruct、Qwen3-30B-Instruct、Qwen3-4B-Thinking、Qwen3-30B-Thinking、Llama-3.1-8B-Instruct。

## 8. 实验结果与主要发现(关键数字)

- **Qwen3-30B-Instruct**：LCB v6 pass@1 42.4% → **55.3%**(+12.9pp，相对 +30.4%)；LCB v5 45.8% → **54.3%**(+8.5pp)。
- **增益集中在中/难题**：hard pass@5 从 31.1% → **54.1%**——说明保留了跨解法分支的探索，而非只锐化单一主模。
- **5 个模型全部提升**(pass@1)：Llama-8B +3.5pp、4B-Instruct +7.5pp、4B-Thinking +3.3pp、30B-Thinking +2.1pp、30B-Instruct +12.9pp；跨家族/规模/变体均泛化。
（数值已对照论文 _txt 第 96、296 行核实。）

## 9. 结果如何支撑其主张(证据链是否到位)

- "极简自蒸馏有效"——5 模型一致提升，证据强。
- "机制是 precision-exploration conflict"——用三路证据互证：**受控仿真**(玩具分布上复现 support compression/reshaping)、**真实模型分析**(Section 4.2)、**理论**(附录 B)。hard pass@5 大涨(31.1→54.1)是关键旁证：若 SSD 只是锐化单模，coverage 不该升反该降。
- "固定解码无法恢复增益"——论文论证 Teval 调温的天花板低于 SSD，支撑"重塑分布"而非"换个温度"的主张。

## 10. 逻辑自洽性(中性评估:哪里站得住、哪里牵强)

- **站得住**：机制解释与 coverage 上升的经验现象自洽；用未验证原始输出仍提升，说明增益来自分布重塑而非数据筛选，逻辑闭环。
- **牵强/需注意**：(1) 对"未验证输出里必然混入错误解"为何不损害性能，靠"低温采样的解平均质量较高 + SFT 对高频正确模式更敏感"间接解释，但缺乏对错误解占比与最终增益的定量关系；(2) 机制主要在代码域验证，"代码任务结构让机制可见"反过来也意味着该机制在非代码域是否成立尚不确定；(3) Ttrain/τ/Teval 三个温度/截断超参的选取对结果影响大，调参成本被"embarrassingly simple"的叙事淡化。

## 11. 残留问题 / 局限

- 仅在代码生成域验证机制；跨域(数学/通用推理)是否成立未知。
- 训练用未验证输出，存在引入错误模式的风险，长程多轮自蒸馏是否会崩溃(类似熵最小化的崩溃)未测。
- 三组解码超参(Ttrain、τ、Teval)需调，最优组合的可迁移性待考。
- SFT 训练本体不在仓库，复现需自备 Megatron-LM 等外部框架。

## 12. 开源代码与框架(链接 + 框架 + 代码可得性)

- 链接：https://github.com/apple/ml-ssd （Apple 许可；约 0.85MB，社区关注度高约 772 stars）。预训练模型(4B/30B)在 HuggingFace。
- **可得性部分**：仓库含 `data_generation/`(采样流水线 `generate.py` + 模板)与 `evaluation/`(LiveCodeBench 评测工具)；**SFT 训练本体不在仓库**——标准交叉熵，依赖 Megatron-LM 等外部框架，README 注明为"复现"。
- 框架：方法极简，仅需 采样 + 标准 SFT + 评估时单独调温；无 RL、无 verifier、无 teacher、无执行环境。

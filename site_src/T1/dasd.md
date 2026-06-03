# dasd — Distribution-Aligned Sequence Distillation (DASD-4B-Thinking)

> **一句话重点 (TL;DR)**：DASD 从"分布对齐"视角改造序列级蒸馏（即在 teacher 响应上做 SFT），用三件套（温度调度学习、散度感知采样、混合策略蒸馏）补回缺失的师生交互，仅 **448K** 样本就让 Qwen3-4B 在推理基准上达到同量级 SOTA，部分基准超过若干 32B 模型。

**元信息**：arXiv 2601.09088v1（2026-01-14 提交，技术报告）｜ 阿里云（Shaotian Yan*, Kaiyuan Liu*, Chen Shen*†, Bing Wang*, Sinan Fan* 等）｜ 主题 序列级蒸馏改进、Long-CoT 推理（与 OPD/distillation 强相关；含一个轻量 mixed-policy 阶段直接缓解 exposure bias，呼应 prefix-OPD / 路径恢复思路）｜ **纯 SFT 式蒸馏，不含 RL** ｜ 代码 https://github.com/D2I-ai/dasd-thinking ｜ 框架 LLaMA-Factory + DeepSpeed ZeRO-3 + Liger-Kernel

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/dasd/fig_01.png)

*Figure 1: Performance of DASD-4B-Thinking on benchmark datasets. All metrics for the comparison models are taken from their official reports.*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/dasd/fig_02.png)

*Figure 2: Overall training pipeline of DASD-4B-Thinking.*

## 1. 相关工作与进展
DeepSeek-R1 首次证明"从强 teacher 蒸馏"可大幅赋能小模型推理，引发社区大量复刻（OpenR1、OpenThoughts、a-m-team、AceReason、LIMO、s1、Light-R1 等）。主流范式即 **SFT on teacher-generated responses = 序列级蒸馏（Kim & Rush 2016）**：简单高效、不限师生架构、无需 token-level logits。另一范式是 logit 蒸馏（Qwen3/Gemma 的 on-policy 变体、Thinking Machines Lab 实现），但需访问 logits 且跨 tokenizer 难对齐。

## 2. 现有工作存在的问题
现有序列级蒸馏多停留在 SFT 视角，只设计启发式数据过滤规则，**忽视蒸馏本质——让 student 学到 teacher 完整输出分布以继承泛化能力**。三大缺陷：

- (i) **teacher 序列级分布表征不充分**：随机采样 + 质量过滤难覆盖分布全支撑，模式覆盖差，或过度表征低概率/噪声序列；
- (ii) **teacher 输出分布与 student 学习能力错配**：SFT 只抬高 ground-truth token 概率，会产生误导梯度（对 teacher 低概率但 student 已高概率的 token 仍继续抬高）；
- (iii) **exposure bias**：teacher-forced 训练与自回归（free-running）推理不一致。
根因是全程缺乏显式师生交互。

## 3. Motivation
不改用 logit 蒸馏（保留序列级蒸馏简单/高效/无 logit 依赖的优点），而是从"分布对齐"角度补回师生交互：让 student 更好覆盖并对齐 teacher 序列级分布、找到更利于 student 学习的目标分布、再缓解 exposure bias。

## 4. 主要灵感 / 核心直觉
温度控制覆盖–一致性的权衡（低温样本一致易学但覆盖窄，高温覆盖广但难学）；师生预测概率在候选响应上的差异可归为若干典型模式，其中"teacher 高置信 + student 低概率"这一高散度模式与测试性能提升正相关；exposure bias 可用"student 生成前缀 + teacher 续写"的少量混合数据廉价修正。

## 5. 主要解决思路(一段话讲清核心)
在两阶段 off-policy SFT 的基础上，用**温度调度**先学一致模式再扩覆盖、用**散度感知采样**优先喂"高散度"实例以对齐 student 学习能力、再加一个轻量**混合策略蒸馏**阶段缓解 exposure bias，构成一条增强的序列级蒸馏 pipeline。

## 6. 方法详解(通俗、分步骤)

1. **Temperature-scheduled Learning（温度调度学习）**：低温样本模式一致、易学但覆盖窄；高温样本覆盖 teacher 更多模式但多样大、学习效率低。故两阶段——先在低温高置信样本上训抓一致模式，再渐进加入高温样本扩覆盖，优于单一温度的单阶段训练。
2. **Divergence-aware Sampling（散度感知采样）**：提出分布分解框架，分析师生预测概率在候选响应上的差异，归纳四种典型分布模式；发现"**teacher 高置信 + student 低概率**"的高散度模式与测试性能提升一致相关，故引导 student 优先学习此类实例，自然缓解误导梯度。〔该思路与 ICLR 2026 论文 arXiv:2512.20908 "Where Did This Sentence Come From? Tracing Provenance…" 相呼应。〕
3. **Mixed-policy Distillation（混合策略蒸馏）**：在初始 off-policy SFT 后加一个轻量阶段——随机选小子集，让训练后的 student 生成完整响应，随机截断前缀，再由 **teacher 从截断点续写**，仅保留通过质量过滤的续写用于 student SFT。少量数据/少量步即缓解 exposure bias 并使输出更简洁。

## 7. 实验数据集

- teacher **gpt-oss-120b**；student **Qwen3-4B**（MoE 变体用作 DASD-30B-A3B）。师生在规模/架构/词表/tokenizer/预训练语料上差异巨大，验证跨族兼容性。
- 训练集仅 **448K** 样本（比多数开源工作少一个数量级），跨数学/代码/科学推理/复杂指令多域；温度消融用 50K 数学响应（gpt-oss-120b，高/低温）。内部名 Apsara-Reason-v1-SFT-stage1/stage2（YAML），对外发布名 Superior-Reasoning-SFT-gpt-oss-120b（README，stage1 低温 105K + stage2 高温 330K）。
- 评测：AIME24、AIME25、LiveCodeBench v5、GPQA-Diamond 等。

## 8. 实验结果与主要发现

- 两阶段 SFT（stage1.yaml / stage2.yaml，DeepSpeed ZeRO-3）：均为 full SFT、cutoff_len=65536、packing=true、per_device_bs=1、grad_accum=4、lr=5e-5（cosine_with_min_lr，min_lr=1e-5）、warmup_ratio=0.1、weight_decay=0.1、num_train_epochs=6、bf16。Stage-2 从 stage1_checkpoints 续训。
- 两阶段数据均经 divergence-aware sampling 生成以对齐分布与能力，并沿用严格质控（过滤截断/重复）；最后加 mixed-policy 阶段。DASD-30B-A3B-Preview 因时间限制仅用 Stage-1 数据训练。
- 结果：DASD-4B-Thinking 在同量级开源模型中达 SOTA，并在关键基准上超过若干 32B 级模型（论文柱状图对照 AM-Think-v1-32B、Qwen3-32B、GLM-Z1-32B 等）。

## 9. 结果如何支撑其主张
温度消融（50K 数学）支撑"调度优于单温度"；散度感知采样的性能相关性分析支撑"高散度模式对齐能力"；mixed-policy 的输出长度/一致性改善支撑"缓解 exposure bias"。仅 448K 样本达到/超越更大模型，支撑"分布对齐比堆数据更重要"的核心论点。

## 10. 逻辑自洽性(中性评估)
三件套各自对应 §2 的三个缺陷，逻辑闭环清晰。但论文是**技术报告**，三个机制多为联合呈现，缺少严格的逐项可加性消融与统计显著性；"高散度模式与性能正相关"为观测性结论，因果关系未做隔离验证。

## 11. 残留问题 / 局限

- **论文文本与发布代码不一致**：正文多处写 student 为 **Qwen3-4B-Instruct-2507**（§lines 160/287/319/324），但 train/stage1.yaml 的 `model_name_or_path` 实为 **Qwen3-4B-Thinking-2507**，README 性能表也以 Thinking-2507 为同族基线对照——**以代码为准应为 Thinking 变体**。〔已逐一核对，结论维持。〕
- 纯 SFT 式蒸馏，未与 RL 结合，无 token-level 对齐。
- 技术报告体例，消融与显著性较弱。
- 散度感知采样依赖能拿到师生 logprob（发布了 -Logprob 数据集），跨任务可迁移性待验证。

## 12. 开源代码与框架(链接+框架+代码可得性)

- 仓库：https://github.com/D2I-ai/dasd-thinking （已 clone，~5.2MB；含 train/stage1.yaml、stage2.yaml、deepspeed/ds_z3_config.json、assets/、dasd_technical_report.pdf）。**含训练配置但不含数据生成/采样代码**。
- 框架：**LLaMA-Factory**（README 明确"We utilize LLaMA-Factory framework for training"）+ DeepSpeed ZeRO-3 + Liger-Kernel（enable_liger_kernel=true）；纯 SFT/序列级蒸馏栈，非 RL 框架。
- 模型/数据开源于 HF & ModelScope：DASD-4B-Thinking、DASD-30B-A3B-Thinking-Preview、数据集 Superior-Reasoning-SFT-gpt-oss-120b（及 -Logprob 版）。

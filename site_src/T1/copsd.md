# copsd — Crosslingual On-Policy Self-Distillation for Multilingual Reasoning (COPSD)

> **一句话重点 (TL;DR)**：低资源语言（尤其非洲语）数学推理差，是因为模型"有 latent 能力但调不出来"。COPSD 把 OPSD 的特权上下文 self-distillation 搬到跨语言场景：student 只看译成低资源语的题、必须用目标语推理；teacher 是同一个模型但额外拿到英文题 + 英文参考解，从而诱导出更可靠分布；在 student 自己的 rollout 上做逐 token reverse-KL 蒸馏。无需外部 teacher、无需目标语 rationale。

**元信息**：arXiv 2605.09548（v1，2026-05-10，cs.CL）｜ LMU Munich (CIS) + MCML（Yihong Liu*、Raoyuan Zhao*、Michael A. Hedderich、Hinrich Schütze）｜ Preprint ｜ 主题：OPSD 的跨语言变体（与本项目 OPD 主线直接同源——把 siyan-zhao/OPSD 那套特权上下文自蒸馏换个输入条件搬到低资源语言数学推理）｜ 代码 https://github.com/cisnlp/COPSD （README 声明 fork 自原始 OPSD 代码库 siyan-zhao/OPSD，已 clone）｜ 框架 HuggingFace TRL + Accelerate/DeepSpeed，LoRA，A100/H200

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/copsd/fig_01.png)

*Figure 1: Radar comparison of Qwen3-1.7B performance on AfriMGSM under a 4096-token generation budget. Each axis corresponds to one of the 17 lowresource African languages, with axis-specific scaling based on the maximum observed performance for that language. COPSD consistently outperforms both the*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/copsd/fig_02.png)

*Figure 2: Overview of COPSD. Each problem is translated into a low-resource language as the student's input. The same LLM acts as both student and teacher: the student generates an on-policy rollout, while the teacher evaluates it with privileged English context and the reference solution. By minimi*

## 1. 相关工作与进展
- **On-Policy Distillation**：结合 student 自生成轨迹的 on-policy 监督与 dense token 级 teacher 反馈，缓解 train-inference mismatch、避免稀疏序列奖励。OPSD（Zhao 2026b）/ Sang 2026 / Zhang 2026a 进一步用**同一模型**在不同条件下分饰 student/teacher，无需外部 teacher。
- **多语言推理**：LLM 跨语言性能差距大，低资源语言尤甚，且常产生语言混杂或不一致的推理轨迹。现有解法：translate-and-test、SFT（机翻 rationale）、self-training、RL——多需翻译过的推理 rationale 或稀疏 outcome 奖励。

## 2. 现有工作存在的问题
- **机翻 reasoning trace 噪声大、off-policy**：数学表达/数量/逻辑依赖易在翻译中出错，且与模型自身推理行为不匹配，造成 train-inference 分布失配。
- **outcome-only RL（GRPO）稀疏、不稳**：低资源语言下模型很少产生正确答案，二元奖励几乎无信号、样本低效。
- 模型有"latent ability"，但在低资源语言下调不出来。

## 3. Motivation
让模型把自己"用英文时"可及的推理行为迁移到低资源语言：student 只看低资源问题、必须用目标语推理（匹配推理时条件）；teacher 是同一个模型，但额外获得 privileged 英文上下文（英文问题 + 英文参考解），从而诱导出更可靠的分布。这样既无需外部 teacher、也无需目标语 rationale，且 dense token 级监督比稀疏 outcome 奖励信号强得多。

## 4. 主要灵感 / 核心直觉
- OPSD 公式不变，只把 teacher 的特权信息从"参考解"扩成"英文问题 + 英文解"，输入端从原语言换成低资源语言——这是 OPSD 的领域迁移应用，增量有限。
- 用 prompt-hacking 强制 student 和 teacher 都用目标语推理：在 `<think>` 后插入语言特定前缀（否则模型会切回英文推理）。

## 5. 主要解决思路（一段话讲清核心）
两个 policy 来自同一 LLM p_θ。Student p_S(·|x_L) 只看低资源问题 x_L；Teacher p_T(·|x_L, x_H, y*) 额外条件英文问题 x_H 与英文参考解 y*。Student 在线生成目标语 rollout ŷ；两者在同一前缀上算逐 token 分布，最小化轨迹平均的 token-level 散度 D(p_T‖p_S)，论文实例化为 **reverse KL + full-vocabulary logit distillation**；梯度只过 student，teacher 作为固定（frozen）分布目标。本质 = OPSD 公式不变，只是把特权信息扩成"英文题+英文解"、输入换成低资源语言。

## 6. 方法详解（通俗、分步骤）
1) 取英文数学题 x_H + 英文参考解 y*；2) 用 Gemini-3-Flash 把问题机翻到目标低资源语 x_L；3) student 看 x_L 在线生成目标语 rollout ŷ（max 2048 token），用 prompt-hacking 强制目标语推理；4) teacher（frozen 同模型）以英文特权上下文评估同一 ŷ；5) 逐 token reverse-KL（full-vocab）蒸馏，LoRA 更新 student。

**〔代码核查〕实现细节与论文表述的出入**：
- **散度**：`generalized_jsd_loss` 支持 forward/reverse KL 与广义 JSD，由 `beta` 选择（beta=0 即 reverse KL）；发布的 4B 训练脚本 `--beta 0`，与论文 reverse-KL 一致。
- **"full-vocabulary" 与发布脚本不符**：4B 脚本实际用 `--top_k 20`（只在 teacher top-20 token 上算散度并重归一化，而非论文正文所述 full-vocab），并加 `--jsd_token_clip 0.05`（逐 token 散度截断，抑制 style token 主导梯度）——与论文"full-vocabulary"表述略有出入。
- **teacher 固定方式**：`fixed_teacher` 通过在 teacher forward 时 `disable_adapter()` 实现——即用 base 模型（无 LoRA adapter）当 teacher，梯度只过带 adapter 的 student（同一份基座权重）。4B 脚本用 `--fixed_teacher`。
- **LoRA**：4B 脚本 `lora_r=64`、`lora_alpha=128`，目标模块 q/k/v/o/gate/up/down_proj，lr 5e-6。
- **代码额外能力（论文未强调）**：trainer 还实现了 EMA teacher（`use_ema_teacher`，与 fixed_teacher 互斥）、`reason_first` 模式（teacher 先对英文参考解推理再评估 student），以及 thinking-machines RL 式 reverse-KL（仅在采样 token 上的 policy-gradient `advantage=(teacher_logp − student_logp).detach()`）——发布主实验用的是 JSD/KL 全分布路径 + fixed_teacher。

## 7. 实验数据集
- **训练**：OpenThoughts（Guha 2025）采样 0.5K 数学题，问题用 Gemini-3-Flash 译成 17 种 AfriMGSM 非洲语；英文问题 + 英文解作 teacher 特权信息。
- **评测**：AfriMGSM（Adelani 2025，17 语 × 250 题，Pass@12，主表 4096-token budget）；PolyMath（更难，8 语 × 125 题，8192-token budget）。
- **模型**：Qwen3-1.7B / 4B / 8B。
- 指标：Pass@12（每题采 12，至少一条正确即算对），Math-Verify 抽 \boxed{} 比对。

## 8. 实验结果与主要发现
- **AfriMGSM 主表（Table 1，Pass@12，4096-token）**：1.7B 平均 9.11→15.53、4B 19.20→20.61、8B 19.41→23.55，全面超 base 与 GRPO；GRPO 几乎无提升（1.7B 仅 9.11→9.18，多语言上偶尔还低于 base）——印证低资源下 outcome RL 信号太稀疏。1.7B 相对提升超 70%，在类型学/正字法多样的语言上普遍受益。
- **训练动态（Figure 3）**：COPSD 早期快速提升 Pass@12 与 format rate；4B/8B 几步内达峰后缓降（可能因目标语生成能力弱、teacher 信号有限、继续训练过拟合不完美 teacher 信号）；GRPO 无明显上升趋势。
- **format adherence**：format rate 与 Pass@12 强正相关（mean Pearson 0.628/0.838/0.728），说明低资源失败部分源于"在有限 token 预算内产不出规定格式答案"。
- **test-time scaling（Table 3）**：大模型更稳定受益于更长生成预算；8B COPSD 从 1024→4096 token 提升 30.0%（GRPO 仅 13.8%），即 COPSD 强化了利用更长目标语推理轨迹的能力。
- **repeat rate（Figure 5）**：COPSD 显著降低 4-gram 重复率，缓解多语言推理常见的重复退化。
- **PolyMath 泛化（Figure 6）**：跨难度普遍超 base，**低资源语言增益最大**（medium 难度 Swahili +32.0、Telugu +32.8、Bengali +15.2；high 难度 Swahili +18.4、Telugu +16.8），高资源语（日/中/俄/西）增益小——支撑"模型已有 latent 能力但难经低资源语表达"的假设。

## 9. 结果如何支撑其主张
- "dense 跨语言监督 > 稀疏 outcome RL" 由 COPSD 全面超 GRPO（GRPO 几乎不动）直接支撑。
- "迁移 latent 英文推理能力" 由 PolyMath 上低资源语增益远大于高资源语支撑（高资源语本就被 base 覆盖、无可迁移空间）。
- 但需在背景下解读：**teacher 拿到了英文参考解 y* 本身**，等于把答案信息蒸进 student——这是"特权"但也意味着 teacher 分布很接近"直接给答案"，提升幅度应在此背景下看待。

## 10. 逻辑自洽性（中性评估）
方法自洽、主张与证据对齐，是 OPSD 的清晰跨语言迁移。增量有限——OPSD 公式不变，新意在"用英文上下文当跨语言特权信息"的设定与 17 语系统验证。需注意两点诚实暴露的张力：(1) 论文称 full-vocab，发布脚本实为 top_k=20 + token clip，表述与实现有出入；(2) teacher 见参考解，使"特权"接近"漏答案"，绝对增益（尤其 4B 仅 19.2→20.61）应保守解读。4B/8B 几步达峰后下降也说明 teacher 信号在弱目标语上很快饱和。

## 11. 残留问题 / 局限
- **依赖英文参考解**：高质量英文监督不可得、或另一高资源语更合适时受限。
- **训练题靠机翻**：题面翻译 artifact 仍可能影响训练质量（虽不需翻译 rationale）。
- **同模型当 teacher**：目标语能力弱时 teacher 分布仍不完美，学习信号易快速饱和甚至继续训练后退化（4B/8B 已观察到）。
- 论文"full-vocabulary"表述与发布脚本（top_k=20 + jsd_token_clip）不符，复现需以脚本为准。

## 12. 开源代码与框架（链接 + 框架 + 代码可得性）
- 仓库：https://github.com/cisnlp/COPSD （README 明确 fork 自 siyan-zhao/OPSD，已 clone）。含 `multilingual_opsd_trainer.py`（self-distillation loss，OPSDTrainer 继承 TRL SFTTrainer，含 fixed_teacher/EMA/reason_first/JSD 与 thinking-machines reverse-KL 两条 loss 路径）、`multilingual_data_collator.py`（student/teacher prompt 构造）、`multilingual_grpo_train.py`（GRPO baseline）、`language_config.py`、17 非洲语 + PolyMath 评测脚本、`multilingual_scripts/run_all_opsd_4b_3000.sh`（4B 复现脚本）。数据集放在 HF。
- 框架：HuggingFace TRL + Accelerate/DeepSpeed（含 vLLM rollout、ZeRO-3 支持），LoRA 训练（4B：r=64/α=128，lr 5e-6，beta=0 即 reverse KL，top_k=20，jsd_token_clip=0.05，fixed_teacher）。A100/H200。代码可得性：训练 + 评测脚本齐全，但复现需注意脚本超参与论文正文的若干出入。

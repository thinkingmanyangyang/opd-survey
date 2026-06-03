# dr_grpo — Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)

> **一句话重点 (TL;DR)**：批判性审视 R1-Zero 范式的两大成分——base 模型与 RL 算法——指出 Qwen2.5 base 已"类 SFT"、"Aha moment"在 base 中早已存在；并发现 GRPO 目标里的 1/|o_i|（响应级长度偏置）与 std(R)（题目级难度偏置）会人为推高（尤其错误）响应长度；去掉这两项得到无偏的 **Dr. GRPO**，在不损推理性能下大幅缩短错误响应、提升 token 效率，并给出 7B 极简 SOTA 配方（AIME24 43.3%，8×A100/27h）。

**元信息**：arXiv 2503.20783（v1 2025-03-21，v2 2025-10-06）｜ Sea AI Lab、新加坡国立大学(NUS)、新加坡管理大学(SMU)；Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi 等 ｜ COLM 2025；ICML 2025 AI4Math Workshop Best Paper Honorable Mention ｜ 主题 T3（RLVR / R1-Zero 训练机理 + RL 算法，High）｜ 代码 https://github.com/sail-sg/understand-r1-zero（已克隆约 56MB，基于自研 RL 框架 Oat）。

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/dr_grpo/fig_02.png)

*Figure 2: Model performance comparison. Oat-Zero-7B is RL-tuned with our minimalist recipe described in Sec. 1 (third paragraph). Please see Sec. B for more results.*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/dr_grpo/fig_01.png)

*Figure 1: Left : Dr. GRPO introduces simple yet significant modifications to address the biases in GRPO (Shao et al., 2024), by removing the length and std normalization terms. Right : Our unbiased optimizer effectively prevents the model from generating progressively longer incorrect responses, the*

## 1. 相关工作与进展

- **R1-Zero 范式**：DeepSeek-R1-Zero 证明可不经 SFT、直接对 base 模型大规模 RL 提升推理，并伴随 RL scaling（响应长度持续增长）与 "Aha moment"（自反思涌现）。
- **社区复现**：多用 Qwen2.5 系列 + GRPO（SimpleRL-Zero、ORZ 等）。
- **GRPO**（DeepSeekMath）：组内相对优势 + 长度归一化的 PPO 变体。

## 2. 现有工作存在的问题

- **base 模型的预训练偏置被忽视**：Qwen2.5 base 不用对话模板时性能反而提升约 60%（疑似预训练用了拼接的 question-answer 文本），使其"已类 SFT"；"Aha moment" 在 base 模型（含 DeepSeek-V3-Base）中早已存在，并非纯 RL 涌现。许多"RL 涌现"的归因因此站不住。
- **GRPO 的优化偏置**：目标中 1/|o_i|（响应级长度归一化）和 std(R)（题目级难度归一化）会人为推高响应长度，尤其推高**错误响应**的长度；trl/OpenRLHF/verl/SimpleRL-Zero/ORZ 等多个开源 PPO 实现也存在 length bias。

## 3. Motivation
从 base 模型与 RL 两个核心成分批判性审视 R1-Zero：厘清哪些现象是真涌现、哪些是预训练遗留；去除 GRPO 的优化偏置以提升 token 效率；并给出一个极简（minimalist）的 R1-Zero 配方。

## 4. 主要灵感 / 核心直觉

- **GRPO 的长度增长可能是 bug 不是 feature**：1/|o_i| 让长响应每 token 梯度被稀释、短响应被放大，对负优势样本会**奖励变长**（拖长错误回答以摊薄惩罚）；std(R) 归一化让难/易题权重失衡。去掉它们即恢复无偏 PPO 风格优势（蒙特卡洛回报 + 无偏 baseline）。
- **模板与 base 的匹配至关重要**：模板-模型不匹配会先破坏能力、再由 RL 重建；领域预训练能抬高 RL 上限。

## 5. 主要解决思路（一段话讲清核心）
Dr. GRPO（GRPO Done Right）= 在 GRPO 目标中移除 1/|o_i| 长度归一化项与 std(R) 难度归一化项：把 token 级 loss 的 mask 归一化从"除以本响应长度"改为"除以一个常数（生成预算 MAX_TOKENS）"，并把优势计算改为只减组均值、**不除以组内 std**。其余沿用 PPO 风格。这样在保持推理性能的同时显著抑制响应长度无意义增长、大幅缩短错误响应、缓解 overthinking。

## 6. 方法详解（通俗、分步骤）

1. **移除长度偏置（Modification 1）**：把 masked_mean（除以 mask.sum=本响应长度）换成 masked_sum 除以常数。
   - 代码（`train_zero_math.py` 第 288–290，已核对）：`masked_sum(..., constant_normalizer=args.generate_max_length) if critic_type=="drgrpo" else masked_mean`。
2. **移除难度偏置（Modification 2）**：优势只减组均值、不除以 std。
   - 代码（`compute_monte_carlo_advantages`，第 294–308，已核对）：`advantages = rewards - values`；仅当 `critic_type=="grpo"` 时才 `advantages /= (std_grouped_rewards + 1e-8)`，drgrpo 不除。
3. **效果**：响应长度不再失控增长，错误响应长度大幅下降，token 效率更高；两者最终 reward 相近。
4. **配套分析**：模板对 base 作答至关重要；模板-模型不匹配先破坏能力再由 RL 重建；领域（数学）预训练提升 RL 上限（Llama-3.2-3B + FineMath/NuminaQA 续训后 RL 更强）。

## 7. 实验数据集

- **训练**：MATH（Hendrycks）level 3–5（主配方）；分析中另用 ORZ-57k / MATH-12k / GSM-8k / ASDiv-2k 研究"模板 × 题集覆盖度"交互。
- **评测**：AIME 2024、AMC、MATH500、Minerva Math、OlympiadBench。
- **base 模型分析**：Qwen2.5-Math-1.5B/7B、Qwen2.5-7B、Llama-3.1-8B、DeepSeek-Math-7B、DeepSeek-V3-Base-685B。

## 8. 实验结果与主要发现

- **极简 SOTA 配方**：用 Dr. GRPO 对 Qwen2.5-Math-7B 在 MATH lv.3-5 + Qwen-Math 模板上 RL，达 AIME 2024 **43.3%** 的 7B SOTA，仅 8×A100、27 小时。
- **算法对比**（Qwen2.5-1.5B base + R1 模板，奖励 Math-Verify 0/1）：vanilla GRPO 与 Dr. GRPO 的 reward 曲线相近，但 Dr. GRPO **阻止响应长度失控、错误响应长度大幅下降**，token 效率更高。
- **base 发现**：Qwen2.5 base 去模板性能反升约 60%；多个 base（含 DeepSeek-V3-Base）已有 "Aha moment"。
- **模板/领域**：模板-模型不匹配先掉能力再由 RL 恢复；数学领域续训提升 RL 上限。

## 9. 结果如何支撑其主张

- "GRPO 有长度偏置" → 去掉 1/|o_i| 后错误响应长度显著下降而 reward/准确率不降，直接验证"长度增长部分来自优化偏置而非真实推理需要"。
- "Aha 非纯涌现" → 在未 RL 的 base 模型（含 V3-Base）上即观测到自反思 token，削弱"RL 涌现自反思"的强主张。
- "base 已类 SFT" → 去模板性能反升的对照支撑 Qwen2.5 base 预训练已含 QA 拼接的推断（间接证据，非直接数据审计）。

## 10. 逻辑自洽性（中性评估）
两处算法修正有清晰的数学动机（恢复无偏 PPO 优势）且与代码精确对应（masked_sum 常数归一化、优势不除 std），是该工作最扎实的部分。base 分析多为观察性/相关性证据："Qwen2.5 预训练用了 QA 拼接"是从"去模板性能反升"反推的合理猜测而非确证；"Aha moment 早已存在"依赖对 base 生成的定性识别，缺统一的量化指标。因此"批判 R1-Zero 范式"的算法侧结论强、归因侧结论偏推测。整体论证自洽，但部分批判性结论的证据强度低于其修辞强度。

## 11. 残留问题 / 局限

- **base 归因偏推测**：Qwen2.5 预训练数据未公开，"已类 SFT"为间接推断；"Aha moment 早已存在"缺量化判据。
- **去 std 的代价未充分讨论**：std 归一化本意是稳定不同难度题的梯度尺度，移除后在极端难度分布下是否引入新不稳定，论文着墨不多。
- **验证集中于数学**：评测与训练几乎全是数学推理，去偏置结论在代码/通用推理上的迁移性未验证。
- **常数归一化引入新超参**：用 generate_max_length 作常数，其取值对梯度尺度有影响，需配合 lr 调整。

## 12. 开源代码与框架（链接+框架+代码可得性）

- 仓库 https://github.com/sail-sg/understand-r1-zero（已克隆，约 56MB）。训练入口 `train_zero_math.py`（含 Modification 1：masked_sum 常数归一化；Modification 2：MC 优势不除 std，均已核对）；算法包 `understand_r1_zero/`（含 `math_grader.py` 答案判定）；另有 `analysis/`、`datasets/`、`examples/`、`deploy_dpsk/`、`evaluate_model.py`。
- 框架：自研 **Oat**（sail-sg 的模块化 LLM online alignment / RL 框架；底层 vLLM + DeepSpeed），依赖 math-verify、pylatexenc、fire 等；奖励用 Math-Verify。
- 资产：HF `sail/Oat-Zero` collection（Oat-Zero-7B 等）。
- 代码可得性：高（训练入口、算法、判分、评测齐全，可复现极简 7B 配方与 1.5B 对比实验）。

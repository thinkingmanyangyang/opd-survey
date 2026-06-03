# prism — Beyond SFT-to-RL: Pre-alignment via Black-Box On-Policy Distillation for Multimodal RL (PRISM)

> **一句话重点 (TL;DR)**：在 SFT 与 RLVR 之间插入一个独立 "预对齐" 阶段，用 black-box（无需教师 logits）的对抗式 OPD——policy 对一个含感知/推理双专家的 MoE 判别器做极小极大博弈——修复 SFT 引入的（且对感知/推理异质的）分布漂移，为下游多模态 RLVR 提供更好初始化。

**元信息**：arXiv 2604.28123（v2 2026-05-01）｜ HKUST(广州) + 清华 + 南洋理工 + 人大 + 中科大 + 国科大 ｜ 2026-05 预印本 ｜ 主题 OPD 新用法（SFT/RLVR 之间的预对齐），与 mtp_opd 相关 ｜ 代码 github.com/XIAO4579/PRISM（MIT，已克隆约 133MB，完整）｜ 框架 三阶段：LLaMA-Factory（SFT）+ verl（对齐/RLVR）+ vendored transformers-4.57.0 + MoE 判别器。

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/prism/fig_01.png)

*Figure 1: Overview of the PRISM pipeline. (a) SFT introduces distributional drift between the policy and the supervision distribution. (b) The alignment stage uses an MoE discriminator with dedicated perception and reasoning experts to repair this drift via adversarial on-policy distillation. (c) Th*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/prism/fig_02.png)

*Figure 2: Architecture of the distribution-alignment stage. An MoE discriminator with perception and reasoning experts is trained via Bradley-Terry loss to distinguish supervision from policy outputs; the policy is updated via policy gradient to maximize the combined MoE reward.*

## 1. 相关工作与进展
LMM 标准后训练为两阶段：先在 curated 示范上做 SFT（能力 bootstrap），再用 RLVR 精炼（决定最终性能）。一系列工作分别改进两阶段的有效性与稳定性，包括重设计 importance weighting/clipping 的 GRPO 变体。OPD 表明从自身 on-policy 生成学习可缓解 exposure bias。

## 2. 现有工作存在的问题
近期发现 SFT 会引入**分布漂移**：既不充分匹配示范策略分布，又丢失模型原有有利分布——SFT 成为漂移源而非纯改进；模型越强，token 级模仿外部示范越易挤占其原生强项。多模态下更突出且**异质**：感知（visual grounding）错误与推理失败的漂移模式不同，会在后续 RL 中复合放大，单一纠正目标无法兼顾。

## 3. Motivation
如何在进入 RL 前修复 SFT 引入的、且对感知/推理异质的分布漂移？据 OPD 思想把对齐重定位为 SFT 与 RLVR 之间的独立预对齐阶段；并提出 logit-free（black-box）形式以摆脱外部教师依赖（适配 Gemini 等无法访问 logits 的黑盒监督源）；用 MoE 判别器为感知/推理提供解耦纠正信号。

## 4. 主要灵感 / 核心直觉
"偏离监督分布的方向本身就是奖励信号"：与其用静态 teacher-forced 目标，不如让一个判别器去分辨 policy rollout 与监督池，并把 "像监督" 的程度当奖励；再把这个判别器拆成感知/推理两个专家，分别给出解耦的纠正信号，避免单一标量把异质漂移混为一谈。

## 5. 主要解决思路(一段话讲清核心)
三阶段流水线：SFT 冷启动 → PRISM 对齐（policy 与 MoE 判别器做 response 级对抗博弈，判别器分作奖励、GRPO 式更新 policy，跑固定步数得 checkpoint）→ 在对齐 checkpoint 上做标准结果型 RLVR（GRPO/DAPO/GSPO）。对齐阶段仅需监督池样本、无需教师 logits。

## 6. 方法详解(通俗、分步骤)
- **Stage1 冷启动 SFT**：高质量示范上得到初始多模态推理策略。
- **Stage2 分布对齐（adversarial OPD）**：建模为 policy 与 MoE 判别器的 response 级极小极大博弈。判别器含**感知专家 D_v**（查 visual grounding/物体）与**推理专家 D_r**（查推理一致性），判别分为两专家加权组合（Eq.1）。架构沿用 Qwen3-VL-MoE，由**四个 Qwen3-VL-2B 组装为专家、Top-2 路由**（已核 §4 行 467 "four Qwen3-VL-2B"、行 327 "Top-2"）；两专家用 Bradley-Terry loss 学习区分 policy rollout 与 supervision pool，从同一预训练 backbone 初始化并各在视觉描述/推理偏好对上 warm-start，带 load-balancing 辅助损失防专家坍塌。〔已核 PDF §3.2、Fig.2、行 439-443〕
- **对齐优化方式（已核 §3.2.3 / 行 439-443、1272）**：交替进行——policy 用 GRPO 式目标（组内归一化、奖励为 MoE 判别器分）更新，判别器两专家用各自 BT loss 更新；**显式去掉 KL 正则**（KL 系数=0，锚定 SFT 初始化会与 "修正 SFT 漂移" 冲突）；跑固定步数后 checkpoint 作 RLVR 初始化，对感知/推理漂移给出解耦奖励。
- **Stage3 RLVR**：在对齐后策略上做结果型 RLVR（GRPO/DAPO/GSPO）。

## 7. 实验数据集
- 监督语料：从 **Gemini 3 Flash** 蒸馏的 **113K** 高质量多模态推理语料（针对当前 LMM 零通过率最难题，含密集视觉 grounding 与逐步推理）；其中 **107K 用于 SFT、6K 最高质量留给对齐与 RL**，并补充同 Gemini 系 **1.26M** 公开示范，合计约 **1.37M** SFT 语料。〔已核行 268-277〕
- Backbone：Qwen3-VL-4B/8B。
- 评测：多种多模态基准（用 lmms-eval）。

## 8. 实验结果与主要发现
- 流程：SFT（1.37M）→ PRISM 对齐（MoE 判别器对抗式 OPD，repair 漂移）→ RLVR；监督池同时作 SFT 基础与对齐参考。
- 主结论（Qwen3-VL）：PRISM 在 GRPO/DAPO/GSPO 多算法、多基准上持续提升下游 RLVR；**PRISM+GRPO 较 SFT→GRPO 基线在 4B/8B 上平均 +4.4 / +6.0 点**（已核行 43、206），DAPO/GSPO 有类似增益。
- 分析：对齐 checkpoint（SFT 后、RLVR 前）精度与 SFT checkpoint 相当但显著缩小了与监督分布的间隙；SFT 单独会在两个尺度上平均拉低 Instruct 起点，PRISM 对齐缓解之。

## 9. 结果如何支撑其主张
"对齐能改善下游 RL" 由跨三种 RL 算法（GRPO/DAPO/GSPO）一致 +Δ 支撑；"修复漂移" 由对齐前后分布间隙缩小的分析支撑；"感知/推理异质" 由双专家解耦奖励设计 + 相应分析支撑。多算法一致性是较强证据。

## 10. 逻辑自洽性(中性评估)
对抗式 OPD + 双专家判别器的设计与 "异质漂移需解耦纠正" 的动机自洽；去 KL 的理由（与修漂移目标冲突）合理。但 "black-box" 仅指无需教师 logits，监督仍来自 Gemini 蒸馏语料，因此并非完全无外部强模型依赖；判别器奖励的可靠性依赖 BT 训练质量，存在对抗训练常见的不稳定/奖励 hacking 风险（论文用 load-balancing 与固定步数缓解）。

## 11. 残留问题 / 局限
- 监督语料强依赖 Gemini 3 Flash 蒸馏（含 113K + 1.26M），方法的 "data-free/teacher-free" 程度有限，仅 logit-free。
- 对抗博弈引入额外不稳定性与算力（4×Qwen3-VL-2B 判别器需独立 warm-start）。
- 评测仅多模态基准；判别器奖励 vs 真值 RLVR 奖励的相互作用、是否引入新偏置未深究。
- +4.4/+6.0 为平均增益，单基准方差与负向案例披露有限。

## 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库：https://github.com/XIAO4579/PRISM（MIT，vendored 上游 Apache-2.0；本地约 133MB，含 vendored transformers-4.57.0、verl、moe（MoE 判别器）、scripts、tools）。数据/checkpoints 在 HuggingFace（prism-vlm）。项目页 https://xiao4579.github.io/PRISM/。
- 框架：Stage1 SFT 用 LLaMA-Factory；Stage2 对齐用 `qwen3_vl_prism`（verl + transformers-4.57.0 + moe）；Stage3 RLVR 用 `qwen3_vl_xxpo_after_prism`（GRPO/DAPO/GSPO）；评测 lmms-eval。
- 代码可得性：完整开源（含判别器与三阶段脚本）。

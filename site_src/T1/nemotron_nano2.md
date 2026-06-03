# nemotron_nano2 — NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model

> **一句话重点 (TL;DR)**：基于 Nemotron-H 的混合 Mamba-Transformer 推理模型，先预训练 12B base(20T tokens, FP8)，多分支对齐(SFT+GRPO+DPO+RLHF+模型合并)后用 Minitron 剪枝 + 仅 forward-KL 的 logit 蒸馏压到 9B，目标在单张 A10G(22GiB)上做 128k 推理，较 Qwen3-8B 同精度下吞吐高 3×–6×。

**元信息**：arXiv 2508.14444 (v4, 2025-09-02) ｜ NVIDIA ｜ 2025-09 ｜ 主题 T?/中等相关（模型压缩/蒸馏 + 多算法对齐；含 iterative on-policy DPO 与 forward-KL 蒸馏，非通用 on-policy KD 方法论）｜ 代码 无完整训练代码（NVIDIA 模型/数据发布）｜ 框架 NeMo / Megatron

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/nemotron_nano2/fig_01.png)

*Figure 1 | Comparison of Nemotron Nano 2 and Qwen3-8B in terms of accuracy and throughput. Nemotron Nano 2 achieves comparable or better accuracies on complex reasoning benchmarks, while achieving up to 6.3 × higher throughput for such workloads. We abbreviate input sequence length to ISL and output*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/nemotron_nano2/fig_04.png)

*Figure 4 | Flow of alignment procedures followed to arrive at the final "Merged" Nemotron Nano 2 12B checkpoint.*

## 1. 相关工作与进展
推理模型需生成长 thinking 轨迹，对吞吐压力大。Nemotron-H 提出把多数自注意力层替换为 Mamba-2 的混合架构以提升长序列生成速度。压缩侧承袭 Minitron（剪枝 + 知识蒸馏），Mamba 重要性估计沿用 Taghibakhshi 2025。对齐侧组合 SFT/GRPO/DPO/RLHF 等成熟算法。

## 2. 现有工作存在的问题
- 12B bf16 权重 22.9GiB > A10G 22GiB，必须压缩才能单卡 128k 推理；
- 推理模型在不同 thinking budget 下鲁棒性差：budget 截断后仍停留 thinking 模式、生成多余 `</think>`，well-formedness 下降；
- Stage-1 SFT 的 128k 拼接会损害工具调用学习；
- 推理能力与 chat 能力存在权衡。

## 3. Motivation
在硬件内存约束（单 A10G）下，造一个既快（混合架构）又准、且支持 128k 上下文与可控 thinking budget 的紧凑推理模型；通过多分支对齐 + 合并缓解能力权衡，通过截断训练提升 budget 鲁棒性，通过 Minitron 剪枝 + 蒸馏在精度损失可控下压缩。

## 4. 主要灵感 / 核心直觉
- 用 Mamba-2 替换大部分自注意力层，把长生成的吞吐瓶颈打开；
- 压缩本质是"先训大再剪小再蒸馏恢复"，logit 蒸馏（forward KL）比普通微调更能恢复精度；
- 不同能力（IFEval/工具/Arena-Hard）彼此干扰，分支独立优化后做 checkpoint 插值合并是低成本折中；
- 训练时混入"突然截断的推理轨迹"，模型学会在 budget 内收尾。

## 5. 主要解决思路(一段话讲清核心)
预训练 12B 混合 Mamba-Transformer base(20T tokens, FP8) → 多阶段 SFT + 多分支 RL/DPO/RLHF 对齐并合并得对齐 12B → 用 Minitron(剪枝 + forward-KL logit 蒸馏)压到 9B，得最终 Nano 2 推理/base 模型。

## 6. 方法详解(通俗、分步骤)
**对齐（Base → 3 阶段 SFT → DPO/GRPO/RLHF 分支 → Merged）**：
- **Stage-1 SFT**：全量数据，混入约 10% 去除推理轨迹的"空 trace"样本以支持 reasoning-off 直答；拼接成长序列（约 128k）。
- **Stage-2 SFT**：专攻工具调用，**不拼接**（修复 Stage-1 拼接对工具学习的破坏）。
- **Stage-3 SFT**：强化长上下文，并加入把推理轨迹**突然截断到 1–2k token**（保留最终答案）的增强样本，提升 thinking budget 鲁棒性〔原稿误记为截断到 12k，已据正文 §3.x 改为 1–2k〕。
- **IFeval RL**、**iterative on-policy DPO**（工具调用）、**RLHF(GRPO)**（HelpSteer3，Qwen-based 奖励模型）分支独立优化。
- **DPO 保持 on-policy**：在 WorkBench 多步可验证工具调用环境中，对每个 checkpoint 生成 on-policy 正样本(成功调用)/负样本(失败生成)，迭代 DPO，保证在线性；BFCL v3 评测。
- **模型合并**：对推理强/chat 强 checkpoint 做插值得 Merged 12B。

**剪枝 + 蒸馏（Minitron，压 12B→9B 推理模型分阶段进行）**：
1. 深度剪枝到 56 层；KD 约 60B tokens @8192 序列长。
2. 宽度剪枝 + KD：约 50B @8192、约 25B @49152、约 1B @262144。
3. DPO → 4. GRPO → 5. KD 约 0.4B @262144 恢复 post-RL 回退 → 6. RLHF → 7. 在 step 5/6 间做 0.5 线性插值合并。
- **重要性估计（仅前向）**：层重要性用逐层临时移除后与原 logits 的 MSE；FFN 神经元/embedding 通道用 1024 样本校准聚合；Mamba head 按 Taghibakhshi 2025（本工作压缩比小，剪 Mamba head 收益有限，故只剪 FFN+embedding 维度 + 深度）。
- **架构搜索**：按 128k/bs1 内存排候选，top-3 各做短 KD 后选 **Candidate 2（精度 63.02）**。
- **重训蒸馏**：对剪枝模型做 **logit-based 蒸馏，仅用 forward KL 散度损失**（accuracy recovery 阶段专用，优于普通微调）；消融(Table 11, ~6B tokens KD)：reasoning-SFT 数据占比 50/50→57.5、70/30→58.5、90/10→57.2，**70%/30% 最佳**。

## 7. 实验数据集
- 预训练 20T tokens(FP8)；约 5% 预训练数据含刻意截断的推理轨迹，用于推理时 budget 控制。
- 后训练总计约 **90B tokens**（多为单轮 prompt-response），SFT 域分布(Table 7)：Math 1.5M、Coding 1.1M、Science 2.0M、Tool-calling 400K、Conversational 1.5M、Safety 2K、Multilingual(全域) 5.0M。响应多由 DeepSeek-R1-0528 与 Qwen3-235B-A22B 生成〔原稿"SFT ~80B"已据 abstract 改为后训练 ~90B〕。
- 评测：AIME-2024/2025、MATH-500、GPQA-Diamond、LiveCodeBench、SciCode、HLE、IFEval、BFCL v3、Arena-Hard、Global-MMLU-Lite、MGSM 等。

## 8. 实验结果与主要发现
- Nemotron-Nano-9B-v2 在推理基准上与 Qwen3-8B 相当或更优，在 8k 输入/16k 输出等生成密集场景吞吐高约 3×–6×（最高 6×）。
- 阶段分析（Figure 6）：DPO 与 GRPO 显著提升 function-calling(BFCL v3) 与 instruction-following(IFEval)，但 GRPO 暂时损害 MMLU-Pro（post-GRPO KD 恢复）；RLHF 提升 Arena-Hard 对齐但引入回退，经模型合并恢复。
- 截断训练显著改善短 budget 下的 well-formedness（Figure 5a→5b）。
- 成功在单 A10G(22GiB, bf16)实现 128k 推理。

## 9. 结果如何支撑其主张
吞吐对比直接支撑"混合 Mamba-Transformer 提升长生成速度"；单卡 128k 推理达成支撑"压缩到内存约束内"；Figure 6 的逐阶段曲线把各对齐/恢复步骤的作用拆开，支撑多分支 + 合并的设计；Table 11 的数据配比消融支撑 forward-KL KD 的数据选择。

## 10. 逻辑自洽性(中性评估)
作为工业级技术报告，pipeline 描述详尽、消融到位，主张与证据基本对齐。但局限明显：(1) 这是模型/数据发布而非可复现的方法论论文，完整训练代码不在发布仓内，外部难以独立复现；(2) "on-policy DPO"与本课题关心的 on-policy KD 关系较弱，蒸馏部分用的是 forward-KL logit 蒸馏（教师固定数据/logits，本质 off-policy KD），与 on-policy distillation 不是同一范式；(3) 大量超参/阶段为工程经验选择，缺乏对替代方案的系统对照。

## 11. 残留问题 / 局限
- 训练代码闭源（NeMo/Megatron 内部栈），仅放模型与多数数据集，方法可复现性受限。
- forward-KL KD + 固定教师 logits，无 on-policy 蒸馏的分布真实性收益；reverse vs forward KL 取舍未讨论。
- 多分支合并(0.5 线性插值)为经验做法，缺乏对合并系数/方式的系统消融。
- thinking budget 控制依赖训练时截断样本与推理时强制插入 `</think>`，对极短 budget 的鲁棒性仍有边界。
- 压缩比较小(<15%)使其"剪枝 + 蒸馏"结论未必外推到激进压缩。

## 12. 开源代码与框架(链接+框架+代码可得性)
- 权重：HF nvidia（Nemotron-Nano-9B-v2、12B-v2-Base、9B-v2-Base）。
- 数据：Nemotron-Post-Training-Dataset-v1、Nemotron-Personas、Aegis Content-Safety v2 等多数预/后训练数据集开源。
- 训练栈：NVIDIA NeMo / Megatron（FP8 预训练）；评测用 lm-evaluation-harness、math-verify。**完整训练代码不在发布仓内**（属模型/数据发布，记录不 clone）。
- 对齐算法组合：3 阶段 SFT + IFeval RL + GRPO(RLHF, Qwen-based RM) + iterative on-policy DPO(工具) + 模型合并；压缩：Minitron 剪枝 + forward-KL logit KD。

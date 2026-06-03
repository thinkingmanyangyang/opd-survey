# dapo — DAPO: An Open-Source LLM Reinforcement Learning System at Scale

> **一句话重点 (TL;DR)**：DAPO 把朴素 GRPO 拆解出 4 个关键改造（Clip-Higher、动态采样、token 级损失、超长奖励整形），并完整开源算法+数据+代码，在 Qwen2.5-32B base 上把 AIME 2024 从约 0 提到 **50 分**，以约一半训练步数超越 DeepSeek-R1-Zero-Qwen-32B（47 分）。

**元信息**：arXiv 2503.14476（v1 2025-03-17，v2 2025-05-20）｜ ByteDance Seed + 清华 AIR（SIA-Lab）+ 港大；通讯 Hao Zhou、Mingxuan Wang ｜ 主题 T3（RLVR / RL 算法系统），相关性 High ｜ 代码 https://github.com/BytedTsinghua-SIA/DAPO（recipe/eval/数据说明，训练逻辑在 veRL）｜ 框架 veRL（vllm==0.8.3、ray[serve]）

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/dapo/fig_01.png)

*Figure 1 AIME 2024 scores of DAPO on the Qwen2.5-32B base model, outperforming the previous SoTA DeepSeekR1-Zero-Qwen-32B using 50% training steps. The x-axis represents the gradient update steps.*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/dapo/fig_02.png)

*Figure 2 The accuracy on the AIME test set and the entropy of the actor model's generated probabilities during the RL training process, both before and after applying Clip-Higher strategy.*

## 1. 相关工作与进展
o1/R1 引领 test-time scaling，长 CoT + 大规模 RL 成为提升推理能力的核心技术路线。但 SOTA 推理模型（OpenAI o1、DeepSeek R1）的关键训练细节被隐藏，社区难以复现工业级结果。GRPO（源自 DeepSeekMath）成为主流的免 critic RLVR 算法基石。

## 2. 现有工作存在的问题
作者在 Qwen2.5-32B base 上用朴素 GRPO 起步，AIME 2024 仅约 30 分（远低于 DeepSeek 报告的 47 分）。复现中暴露的关键问题：
- **熵坍缩（entropy collapse）**：策略熵迅速下降、探索受限、过早确定化。
- **reward noise** 与训练不稳定。
- **R1 论文省略了构建可复现大规模 RL 系统所需的工程细节**。

## 3. Motivation
完全开源一套达到 SOTA 的大规模 LLM RL 系统（算法 + 代码 + 数据），把"让大规模长 CoT RL 成功"的关键技术公开，democratize 工业级 RL 结果。

## 4. 主要灵感 / 核心直觉
熵坍缩的一个直接诱因是裁剪上界压制了"低概率探索 token"的提升空间；无效梯度（整组全对或全错的 prompt）浪费 batch；长序列在 sample-level 归一下贡献被稀释；超长截断样本引入奖励噪声。逐一对症即可在不加 KL 惩罚、仅用规则二值奖励的极简设定下稳定训练。

## 5. 主要解决思路(一段话讲清核心)
在"去掉 KL 惩罚 + 规则化二值奖励（正确 +1 / 否则 −1）"的极简 GRPO 之上，叠加 4 个解耦改造来分别缓解熵坍缩、无效梯度、长序列梯度失衡和超长奖励噪声，组合成 **DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）**。

## 6. 方法详解(通俗、分步骤)
1. **Clip-Higher**：解耦上下裁剪范围 ε_low / ε_high（取 **0.2 / 0.28**）。提高 ε_high 给低概率"探索"token 留出上行空间（实测被上裁剪的 token 概率 <0.2），缓解熵坍缩。
2. **Dynamic Sampling（动态采样）**：过采样并过滤掉准确率为 0 或 1 的 prompt（其组内优势全为 0、无梯度），保证每个 batch 都含有效梯度，稳定训练效率。
3. **Token-Level Policy Gradient Loss**：把 GRPO 的 sample-level 损失改为 token 级聚合（对所有 token 求和后按总 token 数归一），让长序列对梯度有更合理贡献，抑制长样本中的乱码/重复。
4. **Overlong Reward Shaping**：先做 Overlong Filtering（屏蔽超长截断样本的 loss），再用 Soft Overlong Punishment（在长度缓冲区内长度越长惩罚越大），降低 reward 噪声。

## 7. 实验数据集
- 训练：**DAPO-Math-17K**（17K 数学题，答案转为整数便于规则解析；来源网络抓取 + 竞赛官网 + 人工标注）。
- 评测：AIME 2024（重复评测 32 次，报告 **avg@32**）。

## 8. 实验结果与主要发现
- 基座 Qwen2.5-32B base，基线为朴素 GRPO + group reward normalization。
- 优化器 AdamW，常数 lr 1e-6，20 rollout step 线性 warm-up。
- Rollout：prompt batch 512，每 prompt 采 16 个响应；训练 mini-batch 512（每 rollout 做 16 次梯度更新）。
- 长度：期望最大 16384 token + 4096 soft punish 缓冲 → 最大生成 20480 token。
- 评测推理：temperature 1.0、top-p 0.7。
- 结果：AIME 2024 从约 0% → **50 分**，以约 50% 训练步数超越 DeepSeek-R1-Zero-Qwen-32B（47 分）。论文 Table 1 给出 4 项技术逐项消融的累积贡献。

## 9. 结果如何支撑其主张
逐项消融表（Table 1）显示每项技术对 AIME avg@32 的边际贡献，支撑"这 4 项改造是大规模长 CoT RL 成功关键"的主张；熵曲线对比（w/ vs w/o Clip-Higher）支撑熵坍缩缓解；以更少步数超越同基座 R1-Zero 支撑系统级有效性与可复现性。

## 10. 逻辑自洽性(中性评估)
论证链条自洽：每个问题（熵坍缩/无效梯度/长序列/超长噪声）对应一个改造并有消融支撑。需注意主结果集中在单一基座（Qwen2.5-32B）与单一评测（AIME 2024），普适性主要靠社区后续复现而非论文内多任务验证。

## 11. 残留问题 / 局限
- 主评测窄（AIME 2024 + Qwen2.5-32B base），跨基座/跨任务的稳健性论文内验证有限。
- ε_high=0.28 等超参为经验取值，缺少敏感性分析。
- 去 KL + 二值奖励的极简设定在更长/更难任务上的可扩展性需外部验证。
- 训练逻辑不在 DAPO 仓内，需配合 veRL 才能复现。

## 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库：https://github.com/BytedTsinghua-SIA/DAPO （本地已 clone，~4.1MB；含 recipe/eval/数据说明，**训练逻辑实现在 veRL**：https://github.com/volcengine/verl）。
- 框架：veRL（Volcano Engine RL），requirements 含 vllm==0.8.3、ray[serve]；DAPO 作为 verl 的一个 recipe。
- 数据/权重：DAPO-Math-17K（HF: BytedTsinghua-SIA/DAPO-Math-17k）；DAPO-Qwen-32B。代码可得性高（开源系统）。

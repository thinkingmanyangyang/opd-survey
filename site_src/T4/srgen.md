# srgen — Self-Reflective Generation at Test Time (SRGen)

> **一句话重点 (TL;DR)**：零训练的测试时方法，在解码过程中用动态熵阈值识别"高熵 critical token"，在该处短暂暂停、在线优化一个瞬态修正向量 δ 注入 hidden state 再发射下一 token，实现"主动错误预防"；数学/AIME 类任务增益显著，但在 AMC/GPQA/EvalPlus 上增益常仅 +0.2~+2.8pp。

**元信息**：arXiv 2510.02919（v1 2025-10-03；v2 2026-05-29，预印本未注明会议）｜ 港科大（广州）/ 南洋理工 / 爱丁堡 / 香港城大 / 港中深（Jian Mu, Qixin Zhang 等，通讯 Yao Shu）｜ 主题 测试时推理增强（零训练）/ 相关性 中（与 MTP foresight probe / 路径恢复概念相关——都在生成中识别不确定点并干预，但实现是推理时 hidden-state 修正而非训练时蒸馏，属正交可叠加方向）｜ 代码 https://github.com/2020-qqtcg/SRGen （已克隆 ~18MB，含框架 + evaluator + server，可跑）｜ 框架 HuggingFace Transformers 即插即用（含 vLLM/OpenAI 兼容 server）

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/srgen/fig_03.png)

*Figure 3: Cons@k and Pass@k accuracy of Qwen2.5-Math-7B on AMC and MATH500*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/srgen/fig_01.png)

*Figure 1: An overview of the Self-Reflective Generation (SRGen) framework. This framework consists of two main stages. (1) Uncertainty Monitoring . A threshold is dynamically computed from the mean and standard deviation of token entropies within a recent history window of size N. (2) Self-Reflectiv*

## 1. 相关工作与进展
LLM 靠长 CoT 解决复杂推理。已有纠错分两类：(1) post-hoc 迭代精修（如 Self-Refine：对完整草稿批判重写，延迟/算力高）；(2) 训练内生自纠（如 RL：需昂贵训练，且只能在错误产生后干预）。另有 SLOT、MI-Peak 等测试时方法作对照。

## 2. 现有工作存在的问题
前向自回归解码"只能向前、无法回改"，早期 token 错误会级联放大、毁掉整条轨迹。现有自反思方法本质都是 **reactive（被动）**——只在错误已发生后才纠正；"主动错误预防"（错误被提交前就把模型引开）仍是空白。post-hoc 成本随全序列长度线性增长；RL 纠错需先产出错误片段才能介入。

## 3. Motivation
能否在**单次解码过程中**实时识别并在潜在错误点干预，以最小额外成本提升推理可靠性？关键前提：token 信息量不同，"critical token"可由高预测熵识别。

## 4. 主要灵感 / 核心直觉
不在错误之后纠正，而在"风险时刻"（高熵点）之前介入——短暂暂停、在线优化一个小修正向量 δ 注入 hidden state，再发射下一 token。干预局部、瞬态、不需额外完整前向草稿。

## 5. 主要解决思路(一段话讲清核心)
两阶段 monitor-reflect-optimize 循环：用滑窗熵统计的动态阈值识别 critical token；触发时暂停解码，在投影头前的 hidden state 上优化一个瞬态 δ（最小化当前步熵同时保真历史前缀），用优化后的 logits 生成该 token 后即丢弃 δ。

## 6. 方法详解(通俗、分步骤)
- **Stage 1 动态不确定性监测**：每步算 next-token 分布熵 H_t；维护大小 N 的滑窗，算均值 μ、标准差 σ；当 H_t > μ + k·σ 时触发反思（动态阈值适配不同模型的熵分布，避免固定阈值失效；代码另含 `minimal_threshold` 下限）。
- **Stage 2 自反思优化**：触发时暂停解码，优化瞬态向量 δ∈R^d（初始化 0），加到投影头前 hidden state：logits' = W(h_{t−1}+δ)。混合损失 L = (1−λ)·L_CE + λ·L_AEM：
  - **L_CE（回溯上下文损失）**：对已生成前缀施加同一 δ，惩罚破坏既有上下文预测的修正（保真度）；
  - **L_AEM（前瞻熵最小化）**：最小化当前步 next-token 分布熵（让决策更果断）。
  内层优化几步得 δ*，生成 y_t 后丢弃（每次干预局部化）。Theorem 1：该混合损失等价于"min L_AEM s.t. L_CE ≤ ε"约束优化的 Lagrangian，λ 隐式决定保真容忍 ε。

## 7. 实验数据集
数学推理：AIME2024、AIME2025、HMMT2025、AMC；通用推理：GPQA；代码：EvalPlus；效率分析在 MATH500。基座：Qwen2.5-Math-7B、DeepSeek-R1-Distill-Qwen-7B、DeepSeek-R1-Distill-Llama-8B、Qwen3-32B（覆盖两架构族、7B~32B、distill/SFT/RL 多种后训练）。

## 8. 实验结果与主要发现
- 超参（论文正文）：内层步 T=3、lr η=0.01、熵窗 N=25、std 系数 k=4；解码 T=0.6 / top-p=0.95（Qwen2.5-Math-7B 另报 T=0）；max_gen 4096（Qwen2.5-Math-7B）/32768（其余）；准确率取 5 次 pass@1 均值。
- 数学增益显著：AIME2024 上 DS-R1-Qwen-7B +12.0pp、Qwen2.5-Math-7B +7.4pp、Qwen3-32B +6.0pp，普遍优于 Self-Refine。
- 效率（MATH500/Qwen2.5-Math-7B，100 题）：wall-clock 1025s→1198s、token 7.2w→8.0w，远低于 Self-Refine（2316s）与 MI-Peak（1744s）；可与 SLOT 等叠加。
- 在 AMC、GPQA、EvalPlus 上增益往往仅 +0.2~+2.8pp，部分接近噪声。

## 9. 结果如何支撑其主张
数学任务的稳定大增益 + 远低于 Self-Refine 的开销，支撑"低成本主动纠错"主张。但非数学任务的微弱/噪声级增益削弱了"通用可靠性提升"的泛化论断。

## 10. 逻辑自洽性(中性评估)
方法机制自洽（高熵→触发→局部 δ 优化→丢弃）。Theorem 1 只是把加权和重述为 Lagrangian，理论新意有限，不保证 δ 优化得到更"正确"的 token，仅更"自信且保真"。代码 argparse 默认 N=20、K=2、lr=0.1，与论文报告 N=25、k=4、lr=0.01 不同；但 README 的推荐命令与 config 示例正是 N=25/K=4/lr=0.01，与论文一致——即默认值与论文实验设置不同、推荐设置一致（以论文/README 推荐为准）。代码确含 `--adaptive_entropy`/`--minimal_threshold` 开关印证动态熵阈值机制。

## 11. 残留问题 / 局限
- 增益高度集中在"早期 slip 易翻盘"的数学/AIME 类任务，对任务类型挑剔。
- "零训练"但有真实推理开销：每个触发点 T 步反向传播优化 δ，且 L_CE 需对整段前缀重算，长前缀下单次干预成本随已生成长度增长（与论文"只随干预次数缩放"的表述存在张力，触发频繁或前缀很长时开销不可忽略）。
- 熵阈值法对已低熵的强 RL 模型可能很少触发，非数学领域泛化证据弱。
- 与本项目 MTP 仅是松散概念类比（都关注不确定点），实现路径完全不同；可借鉴价值在"动态熵阈值 + 局部修正向量"这一测试时机制本身。

## 12. 开源代码与框架(链接+框架+代码可得性)
- https://github.com/2020-qqtcg/SRGen （已克隆 ~18MB，含 `SRGen/` 框架与 aime/gsm8k/math/gpqa evaluator、`analysis/`、`scripts/`、OpenAI 兼容 `srgen_server.py`，代码完整可跑）。
- 框架：基于 HuggingFace Transformers 的即插即用推理框架（任意 HF 模型可用）；含 vLLM/OpenAI 兼容 server；evaluator 覆盖 AIME/GSM8K/MATH/GPQA。硬件 NVIDIA A800-80G。
- 关键实现：`SRGen/tnot_decorator.py`（熵滑窗阈值 `mean_history + K·std_history`、触发逻辑）、`SRGen/base_evaluator.py`（argparse 超参，默认 N=20/K=2/lr=0.1，推荐用论文 N=25/K=4/lr=0.01）。

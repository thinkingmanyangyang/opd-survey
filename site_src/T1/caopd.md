# caopd — The Illusion of Certainty: Decoupling Capability and Calibration in On-Policy Distillation (CaOPD)

> **一句话重点 (TL;DR)**：标准 on-policy distillation（OPD/自蒸馏）在提升准确率的同时会系统性地把模型推入"过度自信"区；CaOPD 把"答什么（能力）"和"多确信（置信）"在监督目标上解耦——只把轨迹里的置信片段替换成学生自己 rollout 估出的经验成功率，其余照常做 reverse-KL 蒸馏，从而在几乎零额外改动下同时获得能力与校准。

**元信息**：arXiv 2604.16830（v1，2026-04-18）｜ Salesforce AI Research（Jiaxin Zhang、Xiangyu Peng、Qinglin Chen、Qinyuan Ye、Caiming Xiong、Chien-Sheng Wu）｜ Preprint，2026-04 ｜ 主题：OPD 的置信度校准（与 OPD 主线直接相关，高相关）｜ 代码 https://github.com/SalesforceAIResearch/CaOPD （已 clone，含完整训练/评测代码与两域数据）｜ 框架 TRL 0.24 + vLLM 0.12 + DeepSpeed 0.18 + transformers 4.57（自建 distil trainer，在 SDFT/SDPO 自蒸馏管线上做 target replacement）

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/caopd/fig_01.png)

*Figure 1: The Scaling Law of Miscalibration. (Left) Mean Confidence vs. Accuracy on Science Q&A. Almost all modern LLMs are trapped in the red Overconfidence Zone , exhibiting massive calibration gaps. Scaling up capability does not resolve this blind optimism. (Right) CaOPD structurally eliminates*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/caopd/fig_05.png)

*Figure 5: Overview of the CaOPD Framework. The pipeline consists of five stages: (1) querying the student for a base response with verbalized confidence, (2) approximating the true empirical confidence via student rollouts and an objective verifier, (3) revising the completion by replacing the origi*

## 1. 相关工作与进展

- **OPD / 自蒸馏后训练**：近年后训练越来越依赖 on-policy distillation（Agarwal 2024；Lu & TML 2025）与自蒸馏框架。代表作 SDPO（Hübotter 2026）、SDFT（Shenfeld 2026）、OPSD（Zhao 2026）的共同套路是：让 teacher 在"特权上下文（privileged context）"——verifier 反馈、专家示范、或 ground-truth 解——下生成高质量轨迹，student 仅凭 prompt 模仿。这类方法在迁移推理能力上很成功，但几乎都是 capability-centric，不关心置信度。
- **置信度校准**：过度自信问题已被广泛记录。主流修法是 RL + reward shaping，把 Brier/proper scoring rule 罚项塞进 PPO/RL 目标（RLCR、Rewarding Doubt、CAR、Taming Overconfidence）。
- **Test-time 不确定性估计**：SelfCheckGPT、SAC³ 等用多次采样的一致性来估不确定性，可靠但推理成本 O(K)。

## 2. 现有工作存在的问题

1. **"误校准的 Scaling Law"**：作者实测发现，不仅小模型，连前沿大模型（GPT、Claude、Gemini、DeepSeek、Kimi、Qwen3.5-397B 等）普遍落在"过自信区"——能力变强（横轴准确率右移）并不能自动修复盲目乐观。
2. **OPD 本身会加剧过自信**：在 OPD 提升准确率的同时，把平均置信推向饱和（Tool Use 上达到 0.996），过自信缺口（OCG）不降反升。
3. **RL reward shaping 有"能力税"**：RLCR、CAR 等虽能压低绝对置信，但为了躲避罚项让模型变得过度保守，准确率明显掉队（Qwen3-8B Science Q&A 上 RLCR 65.8%、CAR 61.6%，远低于 GRPO 74.5% / SDPO 80.6%）。

## 3. Motivation
根因被归结为 **训练-部署的信息不对称（information asymmetry）**：teacher"开卷"（拿着特权上下文）产生低熵、近确定性的轨迹；student"闭卷"（只有 prompt）。当 student 去最小化对这个特权分布的 per-token reverse KL 时，被迫人为锐化 logits（熵坍缩），并继承成功轨迹那种"笃定"的表述风格（乐观偏置）。结论是：**能力可以靠模仿迁移，但"该有多确信"不能跨信息状态安全迁移**。因此应把能力和置信在监督目标上拆开。

## 4. 主要灵感 / 核心直觉

- "答什么" vs "多确信"是两个正交目标，OPD 的标准 loss 把它们纠缠在一起：reasoning 段学到了能力，confidence 段却被逼着模仿 teacher 那个≈1.0 的笃定。
- 真正该作为置信目标的，是 **student 在部署条件下的成功率 µ(x)**——这恰好是 X-可测的最优预测（见命题 1）。而 µ(x) 可以用 student 自己的多次 rollout + verifier 经验估出来 µ̂(x)。
- 多采样估不确定性本身很贵（test-time O(K)），但可以把这笔账"摊销"到训练阶段：训练时算好 µ̂(x) 蒸进参数，部署时单次前向就能输出校准过的置信。

## 5. 主要解决思路（一段话讲清核心）
CaOPD = **目标解耦 + target replacement**。对每个输入 x，先用 student 采 K 条 rollout、用 verifier 算经验成功率 µ̂(x)=ΣR/K；再做一次"目标替换"：把待蒸馏轨迹里的置信片段 c 改写成 µ̂(x)，同时把 teacher 特权上下文里那个原本≈1.0 的置信也改写成 µ̂(x)。然后照常跑原来的 per-token reverse-KL OPD loss——reasoning 段因为前缀和 teacher 内容都没变，等价于标准 OPD（能力克隆原样保留）；confidence 段则因为监督目标变成了 student-grounded 的 µ̂(x)，把熵坍缩和乐观偏置直接掐掉。reverse-KL 机制完全不动，无 reward 改造、无额外优化阶段。在 SDPO 下，µ̂(x) 直接复用基础训练循环已经生成的 rollout，几乎零额外成本。

## 6. 方法详解（通俗、分步骤）
**生成格式约定**：每条生成 y=(a, c) 切成两段——推理段 a（含最终答案）+ 置信段 c（形如 "Confidence: 0.85" 的 verbalized confidence）。val(c)∈[0,1] 是解析出的标量置信。

**理论三命题（Appendix A 给全证明）**：

- **命题 1（信息差 → 不可辨识）**：当 teacher 特权上下文 Z 对正确性 R 有超出 X 的信息（条件互信息 I(R;Z|X)>0）时，teacher 条件成功率 µT(X,Z) 对 X 不可测（不存在 g(X)=µT 几乎处处成立）；且在平方误差下，X-可测的最优预测恰好是 student 部署成功率 µ(X)，残差严格为正。→ 用 teacher 的笃定当置信目标，从信息论上就是错的。
- **命题 2（特权条件 → 熵坍缩）**：当 I(A;Z|X)>0，teacher 轨迹分布的期望熵严格低于仅给 X 的条件熵；最小化对该分布的 reverse KL，会逼 student 内部 logits 人为锐化。
- **命题 3（选择偏置 → 乐观）**：特权上下文通常取自成功/高质量样本（Dhelpful），使 teacher 期望正确率 ≥ student 边际能力，于是蒸进 student 的隐式目标是对真实部署成功率的**系统性上偏**估计。

**算法（Algorithm 1）**：

1. **学生 grounded 置信估计**：对 x 采 K 条 (a_k,c_k)~π_θ(·|x)，verifier 打分，µ̂(x)=1/K·Σ R(x,a_k)。（SDPO 下复用已有 rollout，只多一次轻量 verifier 评估。）
2. **目标替换**：另采一条待蒸馏轨迹 y=(a,c)；(i) 把 c 换成 µ̂(x) 得 ỹ=(a, µ̂(x))；(ii) 构造特权上下文 z，把其中原≈1.0 的置信改写成 µ̂(x) 得 z̃。
3. **蒸馏**：在修订轨迹 ỹ 上算 per-token reverse KL（式 7）：推理位置 t∈Ia 保留"能力克隆"（与标准 OPD 等价），置信位置 t∈Ic 变成"student-grounded 校准"。AdamW 更新。

开放域无 verifier 时，µ̂(x) 可退化用 Teacher-Anchored Self-Consistency 近似（Appendix B.6）。

## 7. 实验数据集

- **两域**：Science Q&A（Chemistry，SciKnowEval）与 Tool Use（ToolAlpaca）。
- **主模型**：Qwen3-8B、Olmo-3-7B-Instruct；scaling 分析覆盖 Qwen3 全家族 0.6B→32B。
- **对比前沿模型校准**：GPT-5.x、Claude、Gemini、DeepSeek-V3.1、Kimi-K2.5、Qwen3.5-397B 等（仅用于画"误校准 Scaling Law"散点）。
- **设置**：覆盖 SDFT / SDPO 两种 OPD 范式，外加 OOD 迁移与持续学习（CT）。
- **指标**：能力用 Accuracy；校准用 ECE、Brier Score；另引入 OCG（Overconfidence Gap = 平均置信 − 准确率）量化过自信方向，SPR（Strict Pairwise Ranking，对正确答案严格高于错误答案的概率，对置信饱和重罚）量化区分度。

## 8. 实验结果与主要发现

- **OPD 确实加剧过自信**：Qwen3-8B Science Q&A 上，base OCG 已 +58.7%；SDFT/SDPO 进一步推到 +48.1%/+12.9%，Tool Use 上 SDFT 平均置信达 0.996（OCG +32.0%）。
- **CaOPD 结构性纠偏且不掉能力**：Qwen3-8B Tool Use OCG 从 SDFT 的 +32.0% 收到 −0.7%；主表（vs SDFT）上 ECE/BS 大幅下降、SPR 大幅回升（如 Tool Use SDFT 的 SPR 0.085 几乎丧失区分度，CaOPD 恢复到 0.555），准确率持平或略升。
- **避开 RL 的能力税**：vs SDPO 基线，CaOPD 在 Qwen3-8B Tool Use 上准确率 66.2%→70.9%，同时 ECE 0.298→0.133、BS 0.303→0.164；而 RLCR/CAR 校准虽好但准确率明显更低。
- **每步耗时与 SDPO 几乎一致**（Figure 2 右），准确率曲线与 SDPO 重合（左），校准 loss 快速收敛（中）。
- **泛化/持续学习**：OOD（Tool Use→Chemistry）下 SDFT 校准崩（ECE 0.599），CaOPD ECE 0.358（相对降约 40%）；CT 下 CaOPD 解决"校准遗忘"，CT ECE 0.126、SPR 0.662。
- **Scaling**：0.6B→32B 下 SDFT 平均置信恒为接近 1.0 的水平线；CaOPD 让置信随真实准确率动态对齐，在 Reliability(1-BS) 与 SPR 的 Pareto 前沿上全程占优；使 8B 模型校准质量媲美前沿大模型。
- **K 消融**：准确率对 K∈{1..32} 平坦（能力与置信采样方差解耦）；K≥8 是校准的高性价比甜点，K 太小目标过度量化反而困在过自信。

## 9. 结果如何支撑其主张

- "OPD 加剧过自信"由 Table 1 的 OCG 红色扩大直接支撑，且与命题 2/3 的方向一致。
- "能力-校准可解耦"由两条曲线（准确率与 SDPO 重合、校准 loss 独立收敛）+ K 消融（准确率对 K 平坦）共同支撑，逻辑自洽。
- "避开能力税"由 vs RLCR/CAR 的准确率对照支撑，明确归因于"target replacement 不与 RL 优化器对抗"。
- "命题 1 的目标选择正确性"是理论层面给的，实证上由置信目标换成 µ̂(x) 后 ECE/BS 下降侧面印证。

## 10. 逻辑自洽性（中性评估）
整体自洽、诊断清晰、工程改动极小（只改置信片段的监督目标，reverse-KL 机制不动）。理论命题与实证现象方向吻合。值得注意的边界条件：方法只对"verbalized confidence"这一显式格式生效；校准收益几乎全部来自置信段，对推理能力本身没有直接增益（准确率提升主要来自底层 OPD/SDPO，而非 CaOPD 的校准机制）。

## 11. 残留问题 / 局限

- **依赖可验证 verifier 估 µ̂(x)**：开放域只能退到 self-consistency 近似，质量与覆盖面待考。
- **依赖可解析的置信格式**：测试时偶发格式失败（与所有 verbalized uncertainty 方法共有的问题）。
- **训练成本**：每 prompt 需 K 次 rollout（K=8 够用），SDFT 下不一定能像 SDPO 那样零成本复用。
- **能力上界受 base 模型约束**：CaOPD 不提升推理能力，只校准置信。
- **仅 utterance-level 置信**：长程/agentic 多步推理的 step-level 校准是 future work。
- 与 why_sd_degrade（Kim 2026，"自蒸馏为何退化推理"）是同一信息不对称现象的两种切法——此处改监督目标修过自信，彼处只做退化诊断。

## 12. 开源代码与框架（链接 + 框架 + 代码可得性）

- 仓库：https://github.com/SalesforceAIResearch/CaOPD （已 clone，约 40M）。含 `main.py`、`distil_trainer.py`、`distil_config.py`、`eval_science.py`/`eval_tooluse.py`、Tool Use 与 Chemistry Q&A 两域数据、`scripts/`、`figures/`、Salesforce 标准合规文件（LICENSE、SECURITY 等）。
- 框架：TRL 0.24 + vLLM 0.12 + DeepSpeed 0.18 + transformers 4.57；在 SDFT/SDPO 自蒸馏管线上以"采样估 µ̂ → 替换置信 token → 原 reverse-KL 训练"的方式插入，核心改动集中在 distil trainer 的 target replacement。代码可得性：训练 + 评测 + 数据齐全，可复现性较好。

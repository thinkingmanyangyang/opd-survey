# hapo — Heterogeneous Adaptive Policy Optimization: Tailoring Optimization to Every Token's Nature

> **一句话重点 (TL;DR)**：现有 RLVR 对所有 token 一视同仁，违背语言生成的异质本质。HAPO 把 **token 熵当作贯穿采样/优势/裁剪全流程的连续优化驱动量**（而非离散过滤或事后正则），用四个组件对每个 token 做细粒度差异化处理，在多模型规模上一致优于 DAPO。

**元信息**：arXiv 2509.16591（v2, 2026-04-29）｜ 北京大学 / 上海 AI Lab / 北航（共一 Zheng Liu、Mengjie Liu；通讯 Wentao Zhang，README 另标 Lijun Wu）｜ Preprint ｜ 主题 token 异质性驱动的 RLVR（on-policy RL，对标 DAPO），与 OPD 关联**间接**（"token 级熵信号做细粒度加权"思路可迁移到蒸馏的 token 级加权，但本身非蒸馏）｜ 代码 https://github.com/starriver030515/HAPO（已 clone ~8.6MB）｜ 框架 **verl + vLLM**。

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/hapo/fig_02.png)

*Figure 2: Dual-Entropy Token Frequency-Entropy Landscape*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/hapo/fig_01.png)

*Figure 1: HAPO's overall framework. HAPO integrates Adaptive Temperature Sampling, Token-Level Group Average Advantage Estimation, Differential Advantage Redistribution and Asymmetric Adaptive Clipping, leveraging entropy as a core optimization driver to achieve fine-grained, token-level optimizatio*

## 1. 相关工作与进展

- RLHF/RLVR 是提升 LLM 推理的核心（o1、DeepSeek-R1、Qwen3）。
- 用熵区分 token 角色的近期工作：DAPO-with-forking-tokens（少数高熵 token 引导优化）、Archer（高/低熵两组，对高熵放宽 clip）、Entropy-Adv / EDGE-GRPO（用熵调优势）、80/20 规则（高熵少数 token 驱动 RL）。

## 2. 现有工作存在的问题
现有算法对所有 token **统一优化**，无法区分关键推理路径 token 与常规模式 token；已有用熵的方法只把熵当**离散过滤器或事后 bonus**，核心优化机制本身未变。三处具体痛点：

- **采样**：温度困境——低温保精度但压制关键(高熵)token，高温增其出现却引入噪声；高熵关键 token 本就稀少。
- **优势**：序列级优势忽略序列内 token 异质；DAPO token-mean loss 下长负样本主导梯度。
- **裁剪**：低熵 token（多为格式符）易撞左界、阻止其降概率；高熵 token（关键推理）易撞右界、限制探索——统一裁剪"保护噪声、约束探索"。

## 3. Motivation
把 entropy 从"辅助调节器"提升为"核心优化驱动量"：将采样/优势/裁剪的优化参数表达为 **token 熵的连续函数**，在每一阶段嵌入 token 级细粒度处理，实现连续（而非离散分组）的异质优化。

## 4. 主要灵感 / 核心直觉

- 语言生成本质异质：少数高熵 token 是"决策分叉点"（关键推理），多数低熵 token 是"既定模式"（格式/连接词）。
- 二者应被反向对待：高熵处放开探索、低熵处允许激进降噪——统一优化恰好做反了。

## 5. 主要解决思路(一段话讲清核心)
以 token 熵为连续信号，在 RLVR 全流程嵌入四个组件：(1) 采样时按熵实时调温；(2) 优势用 token 级组平均（全局均值/方差归一）；(3) 用熵+重要性比对优势做差分重分配；(4) 反向非对称自适应裁剪（低熵扩左界、高熵扩右界）。先做系统实证（熵-频率 landscape、熵与训练动态、温度-精度曲线）论证三痛点，再逐阶段嵌入。

## 6. 方法详解(通俗、分步骤)

1. **Adaptive Temperature Sampling**：按 token 熵实时调采样温度（高熵升温促探索、低熵降温保连贯），解采样阶段精度-探索权衡。〔code: `rollout.adaptive_temperature=True`, `temperature_tau=0.05`〕
2. **Token-Level Group Average Advantage**：组内按 token 级估优势 \(A_{i,t} = (a_{i,t} - \mu_{\mathrm{glob}})/\sigma_{\mathrm{glob}}\)，兼顾序列长度（保留 token-mean 对长序列的好处）且对正负样本无偏。〔code: `adaptive_advantage=True`〕
3. **Differential Advantage Redistribution**：用熵+重要性比调制优势——对高熵且比值极端的 token 放大优势、对低熵接近 1 的抑制，做细粒度信号归因。
4. **Asymmetric Adaptive Clipping**：反向非对称裁剪界——低熵 token 扩左界（允许激进降噪声概率），高熵 token 扩右界（关键决策点放开探索）。〔code: `adaptive_clip=True`, `clip_alpha=1.0`, `entropy_pivot=0.8`〕

## 7. 实验数据集

- **任务**：数学、代码、逻辑三类，跨多模型规模。
- **数学基准（Table 1/3）**：AIME24(30)、AIME25(30)、AMC(83)、Math(500)、OlympiadBench(675)、Minerva(272)，六基准均值；模型 Qwen2.5-Math-1.5B/7B、Qwen3-8B（recipe 另含 qwen3_14b、llama3.1-8B-Instruct、llama3.2-3B-Instruct）。
- **代码/逻辑**：Logic RL(4–7 ppl 子集)、LiveCodeBench(V5: 24/8-25/2; V6: 25/2-25/5)，模型 Qwen2.5-7B-Instruct-1M。训练数据 dapo-math-17k。
- **基线**：Vanilla GRPO/DAPO、DAPO w/ Forking Tokens、Archer、Entropy Env、EDGE-GRPO；主对照 DAPO 系。

## 8. 实验结果与主要发现

- **主结果（均值）**：Qwen2.5-Math-7B HAPO **50.04 vs DAPO 46.97**；Qwen2.5-Math-1.5B **40.62 vs 38.34**；Qwen3-8B 亦优于各基线，跨规模一致优于 DAPO。
- **逐组件消融（Table 4，A=自适应温度采样 / B=token 级组平均 / C=差分优势重分配 / D=非对称裁剪）**：单加 A 增益最大（46.97→48.85），B/C/D 单加各达 48.56/48.28/48.02，四件齐备 50.04；作者称 A 最关键（主导 token 分布与熵）。

## 9. 结果如何支撑其主张

- "异质优化优于统一优化"由跨规模一致超 DAPO 支撑；"熵作核心驱动量"由消融中四组件各有正贡献、且自适应温度采样(A)主导支撑；三处痛点先有实证 landscape 再被对应组件解决，论证结构清晰。

## 10. 逻辑自洽性(中性评估)

- 四组件都以 token 熵为统一信号，主线自洽；代码 recipe 与论文四组件一一对应（adaptive_temperature / adaptive_advantage / adaptive_clip / entropy_pivot）。
- 保留意见：(1) 四组件叠加相对 A+B+C(49.42) 再增约 0.6，单组件间差异不大，需注意是否含调参收益；(2) "连续 entropy 函数"相比 Archer 离散分组的本质增益、迁移到非 DAPO 基线的稳健性，论文未单独验证；(3) token 级超参遵循 80/20，熵分位 ρ=80%（entropy_pivot=0.8，取 top-20% 高熵 token）。

## 11. 残留问题 / 局限

- 仅在 DAPO 基线上验证四组件协同；对其他 RL 基线（GRPO/GSPO 等）的可迁移性未测。
- 引入多个 token 级超参（温度 τ、clip_alpha、entropy_pivot 等），调参面变大，部分增益可能来自调参而非机制本身。
- 与 OPD/蒸馏仅间接相关——其 token 级熵加权思想可借鉴，但本文是 on-policy RLVR，不涉及 teacher 监督。
- **工程瑕疵**：README 的 arXiv badge/链接仍是占位符 `1234.12345`，badge 文字写 2507.21848，但 README 内 BibTeX 与 HF collection 均指向真实 ID **2509.16591**（与本标头一致）。〔已核〕

## 12. 开源代码与框架(链接+框架+代码可得性)

- https://github.com/starriver030515/HAPO（已 clone ~8.6MB；含 `verl/`、`recipe/`、`vllm/`、`pyproject.toml`）。
- 框架 **verl + vLLM**；四组件作为 verl 训练流程的配置项接入（见 `recipe/qwen2.5_math_7b.sh` 等）。HF 有对应模型 collection。

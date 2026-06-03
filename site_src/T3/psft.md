# psft — Proximal Supervised Fine-Tuning (PSFT)

> **一句话重点 (TL;DR)**：把 SFT 视为 "advantage 恒为正(A=1)、采样自固定离线数据集" 的策略梯度特例，给它套上 PPO/TRPO 式信任域裁剪约束策略漂移；代价是 in-domain 略低于普通 SFT，换来更强 OOD 泛化、不熵坍缩、以及作为后续 RL 起点更优。

**元信息**：arXiv 2508.17784（v2 2026-04-12）｜ 上海交大 + 上海创智学院 + 腾讯大模型部 + 澳门大学 ｜ ICLR 2026 ｜ 主题 T3（SFT-RL 统一视角下的 "改进版 SFT"），相关性 High ｜ 代码 github.com/zwhong714/PSFT（CloneTier=A，已克隆约 23MB，完整）｜ 框架 veRL（recipe/psft）。

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/psft/fig_02.png)

*Figure 2: Training dynamics of in-domain/out-of-domain performance in SFT experiments. In each subfigure, Qwen2.5-7B-Instruct is shown on the left, and Llama3.1-8B-Instruct on the right.*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/psft/fig_01.png)

*Figure 1: Training dynamics of entropy. Each epoch contains 178 steps.*

## 1. 相关工作与进展
社区大量利用强推理模型产出的高质量轨迹，通过 SFT（一种蒸馏）注入推理能力，因其相比 RL 简单高效。SFT-RL 统一视角（如 Prefix-RFT 等）逐渐兴起。

## 2. 现有工作存在的问题

- SFT 本质是 behavior cloning，泛化差：数据次优或与预训练分布失配时会引发过大的策略更新，损害原有能力。
- SFT 易致**熵坍缩**（作者图示约 150 step 后熵骤降至近零），削弱探索，约束后续 RL。
- 挑战：如何同时改善 SFT 模型的泛化与探索能力。

## 3. Motivation
建立 SFT 与 RL 的理论联系：SFT 是 **advantage 恒为正(A=1)、采样自固定离线数据集**的策略梯度特例。借鉴 TRPO/PPO 的信任域思想，给 SFT 加 PPO 式 clipped surrogate，约束策略漂移，避免死记硬背并防熵坍缩。

## 4. 主要灵感 / 核心直觉
"SFT 之所以伤泛化、垮探索，是因为它对每个 demo token 都施加无界的最大似然推力"；若把这股推力用重要性比裁剪封顶（信任域），既保留模仿、又把更新限制在策略附近，就能在不引入显式 KL/reference 的情况下保住熵与原有能力。

## 5. 主要解决思路(一段话讲清核心)
把 demonstration 当作 "trajectory"、advantage 设为常数正值(A=1)，用 \(r_t=\pi_\theta/\pi_{\mathrm{old}}\) 的重要性比 + PPO 式非对称 clip\((1-\varepsilon_{\mathrm{low}},\,1+\varepsilon_{\mathrm{high}})\) 构造信任域约束的代理目标，替代普通 SFT 的最大似然，靠 trust-region（非显式 KL）约束漂移。

## 6. 方法详解(通俗、分步骤)

- 将 SFT 改写为带重要性比裁剪的代理目标；advantage 恒正(A=1)。
- 非对称 clip：run 脚本 clip_ratio_low=0.2、clip_ratio_high=0.28（high>low，与 DAPO clip-higher 同思路，鼓励探索）；use_kl_in_reward=False、use_kl_loss=False、kl_coef=0（靠 trust-region 而非显式 KL）。〔已核 verl/recipe/psft/run_psft.sh:8-16,80（clip_ratio_c=10.0）〕
- **adv_estimator=psft 的实现**：并非独立 `@register_adv_est` 函数，而是在 ray_trainer.py 的 `compute_advantage` 中内联分支——当 adv_estimator==PSFT 时直接令 \(\mathrm{advantages}=\mathrm{returns}=\mathbf{1}(\mathrm{response\_mask})\cdot \mathrm{response\_mask}\)，即每个 response token 赋恒定 advantage=1（被 response_mask 掩到回复 token）。随后 actor 用 PPO 非对称 clip surrogate 更新。〔已核 verl/verl/trainer/ppo/ray_trainer.py:274-276、core_algos.py:104 枚举 PSFT="psft"〕
- π_θold 动态更新（每 4/8/16 步刷新一次，过快或不更新都损害效果，论文取 update.8）；可选 warm-up SFT 先对齐 π_θold。
- 效果：训练全程不熵坍缩，保持生成多样性。

## 7. 实验数据集

- 数学推理（主实验，SFT 阶段）：训练用 **OpenR1-Math-8192** long-CoT 数据集（HF Open-R1，§4.1.1 明示）；RL 阶段用 **DAPO-MATH-17k**（§4.2，clip-higher 0.28）。〔注：开源 run_psft.sh 默认 TRAIN_FILE 指向 `wh-zhu/train_openr1_4k`（4k 变体），与正文主实验 8192 版略有出入，属仓库默认脚本配置差异，已核〕
- 基座：Qwen2.5-7B-Instruct、Llama3.1-8B-Instruct。
- 评测：in-domain 数学 = AIME-24/25、AMC（avg@32）、MATH-500、OlympiadBench、Minerva（avg@8）；OOD = GPQA、ARC-C、TruthfulQA、IFEval（avg@8）、MMLU-Pro、SuperGPQA、HeadQA（pass@1）。推理长度 10,240 token（IFEval 4,096），top-p 0.95，temperature 0.7。
- 普适性另覆盖：human-value alignment（Qwen3-4B-Base + UltraFeedback，后接 DPO）与多模态（Qwen2.5-7B-VL-Instruct，文本 OpenR1-Math-4096、几何 Geometry-3k，后接 GRPO）。

## 8. 实验结果与主要发现
（Table 1，Qwen2.5-7B-Instruct，全部已核对 PDF 行号）：

- **in-domain：PSFT 略低于标准 SFT**，并非更高。AIME-24 SFT 22.08 / PSFT 19.38 / PSFT-warm-up 22.92（行 261-265）；in-domain 6 项均值 SFT 47.99 / PSFT 46.98 / warm-up 48.17（行 328-331）。即 "持平、warm-up 后可反超"，绝非凭空领先。
- **OOD 泛化 PSFT 明显更优**：OOD 7 项均值 SFT 57.90 / PSFT 61.26（行 417-420）；GPQA 32.89 / 33.21；TruthfulQA SFT 63.14 / PSFT 67.16（行 362-365）；IFEval SFT 54.42 / PSFT 73.03（行 406-409，SFT 严重损伤指令遵循，PSFT 基本保住）。
- Llama3.1-8B-Instruct 同向：OOD 均值 SFT 50.49 / PSFT 59.25（行 422-425）。
- **作为 RL 起点更优（Table 2）**：PSFT→GRPO 在 in-domain 与 OOD 均超 SFT→GRPO——Qwen in-domain 均值 53.31 vs 52.40、OOD 均值 64.06 vs 59.90（行 535-538、608-611）；PSFT 起步虽慢但因保留更高熵、探索空间大，RL 后反超。
- 普适性：human-value alignment（PSFT→DPO 在 AlpacaEval2/Arena-Hard/MT-Bench 优于 SFT→DPO 并降 alignment tax）与多模态（PSFT→GRPO 在 MMMU/MMMU-Pro 等保持泛化而 SFT 显著退化）。

## 9. 结果如何支撑其主张
主张是 "牺牲少量 in-domain 拟合换取泛化+探索+更好 RL 起点"，证据链与之**完全对齐且不夸大**：(a) in-domain 略降被如实报告；(b) OOD 全面提升（IFEval/TruthfulQA 提升尤显）；(c) 熵曲线不坍缩；(d) Table 2 下游 RL 反超。证据方向一致、内部统一。

## 10. 逻辑自洽性(中性评估)
理论联系（SFT=A≡1 的 PG）清晰，方法是该视角的直接实例化，代码实现与正文 "A=1" 严格一致（已逐行核）。关键优点：论文坦诚 in-domain 略逊，不做选择性汇报——这与上一轮被修正的 "凭空领先" 表述相反，本次复核确认现有 §8 数字与论文 Table 1/2 **逐项吻合，无新捏造**。clip_ratio_high>low 与 DAPO 同思路，自洽。

## 11. 残留问题 / 局限

- in-domain 拟合略逊普通 SFT，对追求纯 in-domain 峰值的场景不利。
- 依赖 π_θold 更新频率（4/8/16）这一敏感超参，论文经验性取 8。
- 〔注〕Table 2 表头标 SFT→GRPO/PSFT→GRPO，而 §4.2 RL setup 文字用 "DAPO(clip-higher 0.28)+DAPO-MATH-17k"；论文 GRPO/DAPO 表述并存（DAPO 即带 clip-higher 的 GRPO 变体），本文 §7 用 DAPO、§8 沿用 Table 2 标签 GRPO，均忠实原文。
- 仓库默认脚本数据（4k）与正文主实验（8192）不一致，可能影响直接复现峰值。

## 12. 开源代码与框架(链接+框架+代码可得性)

- 仓库：https://github.com/zwhong714/PSFT（CloneTier=A，本地约 23MB，完整）。实现：verl/recipe/psft/（main_psft.py、psft_ray_trainer.py、config/psft_trainer.yaml、run_psft.sh / run_sft.sh）；adv_estimator=psft 内联于 ray_trainer.py。模型权重：HF collection `wh-zhu/psft-...`。
- 框架：veRL（recipe 内实现）；torch2.6.0+cu124+vllm0.8.5。
- 代码可得性：完整开源，PSFT 的核心（A=1 内联分支 + 非对称 clip）可逐行核验。

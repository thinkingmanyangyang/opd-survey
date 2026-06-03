# aligndistil — AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation

> **一句话重点 (TL;DR)**：把"带 token-level reward 的 RLHF 目标"在理论上**等价改写成一个蒸馏问题**——其 teacher 分布是 DPO 模型与 reference 模型 logit 的线性组合，于是 RLHF 对齐就变成"向一个自动合成的 teacher 分布做 token 级蒸馏"。值得看的点：它给"DPO 的 token-level reward 分解"找到了一个干净的蒸馏对应，并用一个 reverse-DPO 对比 + 逐 token 自适应外插权重把这个对应做得更稳更准。

**元信息**：arXiv 2503.02832v3（2025-07-23，cs.CL）｜ 北京交通大学(交通大数据与人工智能教育部重点实验室、计算机学院) + 腾讯（Songming Zhang 实习于腾讯完成；通讯 Yufeng Chen、Jinan Xu）｜ ACL 2025 ｜ 主题 T1/T3 On-Policy Distillation / RLHF 对齐，**相关性 High** ｜ 代码 https://github.com/songmzhang/AlignDistil（**完整**，含定制版 OpenRLHF）｜ 框架 OpenRLHF v0.5.2.post2（+ vLLM for on-policy）

## 关键图示

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/aligndistil/fig_01.png)

*Figure 1: An overview of our AlignDistil. At token position t , the distribution from the current policy π θ ( t ) is guided by a teacher distribution π ∗ ( t ) , which is constructed from an adaptive extrapolation between logit distributions from a DPO model and a reverse DPO model with a weight α*

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/aligndistil/fig_02.png)

*Figure 2: Convergence curves of token averaged reward from optimization on the sentence-level, token-level scalar-type, and token-level distributional reward.*

## 1. 相关工作与进展(这条线现在做到哪一步)
LLM 对齐主流两条线：(1) **RLHF**——先训 response-level reward model，再用 PPO 等优化策略并约束不偏离初始模型；(2) **直接偏好学习(DPO 等)**——用策略模型自身参数化 reward，直接在偏好数据上训练，省掉显式 RL。近期出现 token-level 的偏好优化(如 TDPO)与把 RLHF 视为蒸馏的视角(如 RTO)。AlignDistil 处在"把 token-level reward 优化与蒸馏统一起来"的交叉点。

## 2. 现有工作存在的问题(本文针对的痛点)

- 现有对齐多用**稀疏的 response-level reward / 偏好标注**去优化一整条回复里的所有 token，**粒度太粗**：无法反映每个 token 的个体贡献，可能错误惩罚一条好回复里的高质量 token、或鼓励差回复里的低质量 token，拖慢收敛、限制上限。
- DPO 自带的 token-level reward 虽可分解，但**精度不如纯 reward model**，且在不同 token 上存在欠优化/过优化不均衡。

## 3. Motivation(为什么做这件事)
作者想要 token-level 的奖励优化，又不想额外训练一个昂贵的 reward model。注意到 **DPO reward 本身可在 token 级分解**，于是设问：能否把"带 DPO token-level reward 的 RLHF 目标"直接转成一个**蒸馏**过程？如果能，token-level 奖励优化就等价于"向某个 teacher 分布蒸馏"，实现简单且稳定。

## 4. 主要灵感 / 核心直觉
两个直觉：(1) **RLHF + DPO-reward ⇔ 蒸馏**：理论上可证 sequence-level RLHF 目标等价于一个 token-level 蒸馏目标，其 teacher 分布 = DPO 模型与 reference 模型 logit 的线性组合(本质是"沿着 DPO 改进方向外插")。(2) **正反两个 DPO 更有判别力**：再训一个把 chosen/rejected 交换的 **reverse DPO**，让它捕捉"低质量数据的负面特征"，与正常 DPO 组成对比，reward 更准、可训练参数隐式翻倍。

## 5. 主要解决思路(一段话把核心机制讲清)
从 RLHF 目标出发引入 DPO reward，证明该目标等价于"把当前策略向一个合成 teacher 分布做 reverse-KL 蒸馏"；teacher 分布由 **forward DPO 模型**与 **reverse DPO 模型**(充当 reference)的 logit **外插(extrapolation)** 而成，推动策略**越过** DPO 模型；再用一个**逐 token 自适应外插权重 γ_t** 给每个 token 位置定制 teacher 强度，缓解欠/过优化。可在 on-policy(采样数据，效果更好)与 off-policy(偏好数据，更高效)间切换。

## 6. 方法详解(通俗、分步骤;关键公式用白话解释,必要时给伪代码)
三步流程：**① 训正常 DPO** → **② 训 reverse DPO**(交换 chosen/rejected) → **③ AlignDistil**：用两者 logit 合成 teacher 分布，蒸馏到当前策略。

关键公式（已逐行核对 `aligndistil_trainer.py`，`reward_boost_type=="aligndistil"` 分支，代码注释明确 **teacher = forward DPO 模型，reference = reverse DPO 模型**）：

- **逐 token 外插权重**：`tvd = |p_tea − p_ref|.sum`（forward-DPO 与 reverse-DPO 概率分布的全变差距离 TVD），`weight = tvd·β2 + 1e-3`。白话：两个 DPO 模型在某 token 上分歧越大(TVD 大)，外插越激进。
- **合成 teacher logits**：\(\text{final\_tea\_logits} = \text{weight} \cdot (\text{teacher\_logits} - \text{reference\_logits}) + \text{teacher\_logits}\)。即在 forward DPO 的基础上，再沿"forward−reverse"差分方向外推一段，让 teacher 比单纯 DPO 更"对齐"。
- **蒸馏损失**(reverse-KL 形)：\(\text{rlhf\_loss} = \left(\text{stu\_probs} \cdot (\text{stu\_lprobs} - \text{final\_tea\_lprobs})\right).\text{sum} \times \beta_2\)，其中 \(\beta_2 = \beta / \text{weight}\)。
- 对照分支：`theorem1` 用 \(\text{weight} \cdot \text{teacher} + (1 - \text{weight}) \cdot \text{reference}\) 的常数线性组合；`theorem1_contrast` 用差分形式但常数权重；`*_adaptive` / `aligndistil` 则把权重换成逐 token 的 TVD 自适应——消融正是为隔离这一项。

## 7. 实验数据集

- 初始模型：Qwen2-1.5B-Instruct、Qwen2.5-1.5B-Instruct。
- 偏好/训练数据：UltraFeedback。
- 评测：AlpacaEval 2.0(length-controlled win rate)、MT-Bench、Arena-Hard；裁判用 Qwen2.5-72B-Instruct(作者测得与 GPT-4 判断相当但更便宜)。

## 8. 实验结果与主要发现(关键数字)

- on/off-policy 两版 AlignDistil 均显著超基线；AlpacaEval 2.0 LC win rate 较 DPO **提升 >6%**；优于 TDPO1/2、RTO、PPO、DPO(β=0.01) 等。
- Table 2：对比 DPO reward 的 reward accuracy(UltraFeedback 训练/测试各 1000 样本)优于 vanilla DPO reward，甚至优于显式 reward model。
- Table 3：token 自适应外插优于常数外插(为隔离数据影响，用 off-policy 设置)。
- 〔待核〕Table 1 各 benchmark 精确数值未逐一抄录。

## 9. 结果如何支撑其主张(证据链是否到位)
证据链较完整：理论(RLHF⇔蒸馏的等价推导) + Table 2(对比 reward 更准) + Table 3(自适应外插 > 常数外插) + 主表(三 benchmark 一致超基线)，分别对应方法的三个主张(蒸馏等价、对比 reward、token 自适应)。on-policy 优于 off-policy 也与"采样数据更贴近策略分布"的预期一致。

## 10. 逻辑自洽性(中性评估:哪里站得住、哪里牵强)

- **站得住**：等价性推导 + 代码实现一一对应(外插公式、TVD 权重、reverse-KL 损失均已核实)；"用 reverse DPO 作 reference 而非固定初始模型"是一个有依据的设计，对比实验支持。
- **牵强/可质疑**：(1) "外插越过 DPO"假设 DPO 方向在更大步长上仍正确，过度外插可能放大噪声，论文靠 `1e-3` 下限和 TVD 自适应缓解但无理论上界；(2) 规模仅 1.5B、评测偏对话/指令，未验证更大模型或推理域；(3) 需训三个模型(DPO、reverse DPO、最终策略)，成本不低。

## 11. 残留问题 / 局限

- 实验规模有限(1.5B)，未覆盖大模型与数学/代码等推理任务。
- 三阶段训练(DPO→reverse DPO→蒸馏)成本与超参(β、β2、kd_temperature)敏感性需更多消融。
- 外插权重虽自适应但仍依赖人工设定的 β/β2 与 `1e-3` 下限。
- 〔待核〕主表精确数值未抄录。

## 12. 开源代码与框架(链接 + 框架 + 代码可得性)

- 链接：https://github.com/songmzhang/AlignDistil （已 clone 到 resource/repos/aligndistil，约 2.4M，**完整**，自带定制版 OpenRLHF 子目录）。
- 框架：**OpenRLHF v0.5.2.post2**（`OpenRLHF/version.txt` 实测；安装 `cd AlignDistil/OpenRLHF && pip install -e ./`，on-policy 版另需 vLLM）。
- 训练脚本：`train_scripts/ultrafeedback/qwen2.5-1.5b/`（dpo / reverse_dpo / aligndistil_off_policy / aligndistil_on_policy）。核心 loss 在 `openrlhf/trainer/aligndistil_trainer.py`（`reward_boost_type=="aligndistil"`）。

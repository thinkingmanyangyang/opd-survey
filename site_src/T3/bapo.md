# bapo — BAPO: Stabilizing Off-Policy Reinforcement Learning for LLMs via Balanced Policy Optimization with Adaptive Clipping

> **一句话重点 (TL;DR)**：异策略（off-policy）RL 训练 LLM 时，数据越陈旧（staleness 越大）越容易梯度爆炸、熵崩溃。BAPO 找到两个根因——负优势样本主导梯度、固定对称裁剪系统性挡掉"增熵更新"——并据此**每个 batch 动态调裁剪上下界**，让"正 token 贡献占比"达到目标 ρ0，从而既防爆炸又保熵。

**元信息**：arXiv:2510.18927v1（2025-10-21，cs.LG）｜ 复旦 FudanNLP / 上海稷迹智锋 / 上海创新研究院（共一 Zhiheng Xi、Xin Guo；通讯 Tao Gui、Qi Zhang）｜ 2025-10 预印本｜ 主题：off-policy RL 稳定化（自适应裁剪 + 熵保持），与 OPD **直接关联弱**，但其 Entropy-Clip Rule 对理解 partial rollout / 经验回放等异策略训练的熵崩溃有参考价值｜ 代码 https://github.com/WooooDyy/BAPO （已 clone 约 5.2MB，基于 veRL，方法在 `recipe/bapo`，真实可用）｜ 框架 veRL（GRPO 为基础算法）。

---

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/bapo/fig_01.png)

*Figure 1 | Performance of BAlanced Policy Optimization with Adaptive Clipping (BAPO).*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/bapo/fig_02.png)

*Figure 2 | Preliminary results with different data staleness. As the staleness increases, the model suffers from unstable optimization, decreasing entropy, and even a sudden collapse in training.*

## 1. 相关工作与进展

- RL 已成对齐/强化 LLM 的核心范式（推理、代码、agentic）。
- **off-policy RL**（rollout 行为策略 ≠ 训练目标策略，使用过期数据）样本效率高、契合现代基础设施（**partial rollout**、经验复用），适合超长/难任务。
- 与之相关的"非对称裁剪"家族已有多项：Clip-Higher（DAPO）、KL-Cov、CE-GPPO、80/20 等，思路都是"放宽对低概率正 token 的裁剪"。

## 2. 现有工作存在的问题
直接套用 off-policy RL 到 LLM：随 staleness 增大（Fig.2），出现**优化不稳、梯度爆炸、甚至崩溃**，同时**策略熵急剧下降**（探索退化、过度利用）。已有非对称裁剪方法**手动固定阈值**，僵硬、缺乏自适应。

## 3. Motivation
作者通过理论+实证锁定两个根因：

- **(i) 优化失衡**：负优势样本在数量与对策略梯度损失贡献上都占主导，过度惩罚会压制有用/中性行为；且低概率负 token（log 项 → −∞）易引发梯度爆炸。难题/训练早期会进一步加剧负样本占比。
- **(ii) Entropy-Clip Rule（理论推导）**：PPO 类目标的**固定对称裁剪**会系统性阻断"增熵更新"——把大量低概率正 token 挡在更新之外，同时过度惩罚低概率负 token，使分布锐化、熵崩溃。
据此目标：平衡正负贡献 + 防梯度爆炸 + 保熵维持探索。验证实验表明非对称裁剪（增大上界 c_high 纳入更多低概率正 token）能提升性能并抑制熵下降，但固定阈值不够灵活——故引出自适应版本。

## 4. 主要灵感 / 核心直觉
既然问题是"正 token 被裁太多、负 token 罚太狠"，那就**用'正优势信号对策略梯度损失的贡献占比'作为可观测的调节目标**，每个 batch 自适应地扩裁剪窗，直到正 token 贡献达到目标占比 ρ0——免去手动调阈值。

## 5. 主要解决思路（一段话讲清核心）
BAPO（Balanced Policy Optimization with Adaptive Clipping）在 GRPO/PPO 代理目标上**动态调整裁剪上下界 (c_high, c_low)**：每个 batch 搜索一对界，使正优势信号对策略梯度损失的贡献占比 ≥ 目标 ρ0；扩 c_high 纳入更多低概率正 token（增熵），扩 c_low 过滤过多低概率负 token（防爆炸），设 ρ0 又能防止熵无控增长。

## 6. 方法详解（通俗、分步骤）

- 代理目标仍是 PPO/GRPO 的 min(r·A, clip(r,1−ε,1+ε)·A)，但裁剪界不再固定。
- **论文 Algorithm 1 的搜索顺序**：从 c_low=a−、c_high=a+ 起，while 循环内**先增 c_high**（步长 δ1，优先纳入更多低概率正 token）至 b+；不够再增 c_low（步长 δ2，过滤低概率负 token）——即"先 c_high 后 c_low"，单 while 内交替。
- **〔code 不一致，已核〕**：开源实现 `recipe/bapo/policy_loss.py::compute_policy_loss_bapo` 顺序与论文**相反**——先 "adjust lower clip range first"（增 c_low 到 ratio_lower_max），不满足再 "increase upper bound"（增 c_high），且为**两段顺序循环**而非论文的单 while 交替。差异不改方法本质（都为满足正 token 贡献占比而扩裁剪窗），但与 Algorithm 1 表述不符，**复现以代码实际行为为准**。
- **〔code 额外细节，已核〕**：代码对负优势样本另含 **dual-clip**（clip_ratio_c 默认 3.0），论文 Eq.8 正文未强调。
- 净效果：增大正 token 贡献、防负 token 主导与梯度爆炸；纳入低概率正 token + 过滤低概率负 token 来保熵；ρ0 防熵失控与尾部退化。免去 DAPO/手动非对称裁剪的繁琐调参。

## 7. 实验数据集

- **RL 训练数据**：SkyWork-OR1-RL-Data。
- **评测**：AIME 2024、AIME 2025（report 16 rollouts 平均）。
- **骨干**：DeepSeek-R1-Distill-Qwen-7B/32B、OctoThinker-Llama3.2-3B-Long-Zero；并自训两个 SFT 模型 BP-Math-7B/32B（由 Qwen2.5-Math 微调而来）。

## 8. 实验结果与主要发现

- **BP-Math-7B(BAPO)**：AIME24/25 = **70.8 / 62.5**，超 SkyWork-OR1-7B（70.2 / 54.6，AIME25 +7.9）。
- **BP-Math-32B(BAPO)**：**87.1 / 80.0**，同规模 SOTA，并超 o3-mini-medium、Gemini-2.5-Flash-Thinking、DeepSeek-R1(671B)。
- **Llama**（GRPO→BAPO）：AIME24 2.5%→5.4%、MATH 58.4%→66.0%。
- partial rollout 与不同 staleness 下均比 GRPO 更稳（Fig.2 复现的崩溃在 BAPO 下消失）。

## 9. 结果如何支撑其主张
主张是"自适应裁剪能稳定 off-policy RL 并保熵"。支撑：(a) staleness 扫描显示 GRPO 崩溃而 BAPO 稳定（直接对应 motivation）；(b) 主基准上超 SkyWork-OR1 与多个专有系统。但支撑链有缺口——见 §10/§11。

## 10. 逻辑自洽性（中性评估）

- 理论贡献 **Entropy-Clip Rule** 较清晰，把"裁剪→挡增熵更新→熵崩溃"链条讲通，且与 motivating 实验一致。
- 但**主结果泛化覆盖窄**：核心评测只在 AIME24/25 两个小测试集，Llama 才补 MATH。
- 与 SkyWork-OR1 的对比部分依赖**自训的强 SFT 起点 BP-Math**；BAPO 相对自家 GRPO 的增益在 32B 上较小（AIME24 84.6→87.1），削弱了"方法本身"的归因强度。
- 仓库默认 config/example 是示例值而非论文实验值，复现需手动改超参（见 §12）——这本身不影响逻辑，但增加复现摩擦。

## 11. 残留问题 / 局限

- 评测基准窄（AIME 为主），泛化性证据不足。
- 自适应搜索的目标占比 ρ0、可移动区间、步长仍是超参（论文称"未精调"），其鲁棒性未系统扫描。
- **代码与论文 Algorithm 1 搜索顺序相反**，方法描述与实现存在不一致（已核，见 §6）。
- 与既有非对称裁剪家族（Clip-Higher/KL-Cov/CE-GPPO/80-20）同源，**创新点集中在"用正 token 贡献占比作自适应目标"+ Entropy-Clip Rule 理论**，算法增量有限。

## 12. 开源代码与框架（链接+框架+代码可得性）

- 代码：https://github.com/WooooDyy/BAPO （resource/repos/bapo，约 5.2MB；含 verl 子目录 + `recipe/bapo`）。依赖 hydra-core、liger-kernel、accelerate 等。
- 框架：**veRL**，以 **GRPO** 为基础算法。`python -m verl.trainer.main_ppo` 主循环。
- 训练设定：预备/验证实验用 R1-Distill-Qwen-7B，max len 8k、lr 2e-6、temp 0.6（staleness 实验用 SkyWork-OR1-RL、max len 32k）；BP-Math 主实验 max len 64k。通过 ppo_epoch 经验复用与 partial rollout 引入 staleness。
- **论文 BAPO 超参（未精调）**：ρ0=0.4，可移动区间 a−=0.6/b−=0.9、a+=1.2/b+=3.0，步长 δ1=0.05/δ2=0.02。
- **〔复现注意，已核〕**：仓库默认 config（`recipe/bapo/config/bapo_trainer.yaml`）与 `run_bapo_example.sh` 给的是**示例值非论文值**——config 默认 adv_ratio_target=1、ratio_lower 0.6→0.8 step0.05、ratio_upper 1.2→2.0 step0.05；example.sh 用 ppo_epochs=2、low_max=0.95、upper_step 0.1。复现论文数字需手动改回上述论文超参。
- 代码可得性：方法实现完整可跑，仅需对齐超参。

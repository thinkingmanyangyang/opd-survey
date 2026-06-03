# dft_reweight — On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification (DFT / Dynamic Fine-Tuning)

> **一句话重点 (TL;DR)**：把标准 SFT 梯度还原成"策略梯度 + 隐含奖励"形式后发现其奖励被 1/πθ（逆概率）加权而病态；只需把每个 token 的交叉熵损失乘以该 token 的预测概率（detach 阻断梯度），就能把隐含奖励整平为常数 1，得到更稳定、更接近 RL 风格的更新，从而显著改善 SFT 的泛化——核心改动只有一行代码。

**元信息**：arXiv 2508.05629 (v3, 2026-02-27) ｜ 东南大学、UCLA、上海交大、南洋理工、UC Berkeley、武汉大学、UC Merced 等；Yongliang Wu, Yizhou Zhou 等 ｜ ICLR 2026 ｜ 主题 T3（High，"用 RL 视角理解并改进 SFT"的代表性单行改动法，ASFT 的直接前置工作）｜ 代码 https://github.com/yongliang-wu/DFT（已克隆约 66MB，已被 TRL / LLaMA-Factory / ms-swift 收录）｜ 框架 veRL（FSDP SFT 训练器 + Liger），评测沿用 Qwen2.5-Math 仓库。

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/dft_reweight/fig_01.png)

*Figure 1: Accuracy progression for Qwen2.5-Math-1.5B across mathematical benchmarks, illustrating faster convergence and better performance achieved by DFT relative to SFT.*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/dft_reweight/fig_02.png)

*Figure 2: Token probability distributions on the training set before training and after fine-tuning with DFT, SFT, and various RL methods. A logarithmic scale is used on the y-axis for clarity.*

## 1. 相关工作与进展

- **SFT vs RL 的权衡**是 LLM 对齐的核心议题。SFT 模仿专家 demonstration、简单高效（类机器人 behavioral cloning），但"SFT memorizes, RL generalizes"——SFT 易过拟合、泛化弱于 RL。
- **混合范式**：SFT 预训练 + RL 精修（InstructGPT 式）是最常见策略；近期也有交错式方法。
- **偏好/拒绝采样类**：DPO、RFT/RAFT 等可视为缓解稀疏奖励问题。
- 已有少量工作引入基于数据生成策略的重要性加权，但未把它落到"修正 SFT 隐含奖励"这一点上。

## 2. 现有工作存在的问题

- RL 泛化好但算力贵、需显式奖励、调参敏感，且在"只有正样本 demonstration、无负样本/无奖励模型"时不可用——此时 SFT 是唯一可行选项。
- 因此核心未解问题是：**SFT 本身能否被根本性改进？** 作者通过数学分析定位症结：把 SFT 梯度解释为策略梯度时，其隐含奖励是 (i) 稀疏的（只在精确匹配专家 token 时非零），且 (ii) 被一个 1/πθ 的重要性权重缩放——当模型给专家动作的概率很低时，权重爆炸→梯度异常大→优化不稳、过拟合稀有精确匹配样本，限制泛化。

## 3. Motivation
既然 SFT 的隐含奖励因 1/πθ 逆概率加权而病态，那么就用一个"校正性逆比"——即乘以策略概率 πθ——来抵消该畸变，把隐含奖励整平为均匀常数。这样不引入采样、不引入奖励模型、不需要 reference model 或大 batch，就能把 SFT 梯度从有偏不稳定估计变成更接近 RL 风格的稳定更新。

## 4. 主要灵感 / 核心直觉

- **把 SFT 写成 RL**：SFT 的梯度可解释为 on-policy 策略梯度，其奖励是匹配专家轨迹的稀疏指示函数 r(x,y)=1[y=y⋆]，但被重要性权重 1/πθ 偏置（论文 eq.6）。
- **病态来源是 1/πθ**：低概率专家 token → 巨大权重 → 不成比例的大梯度。
- **校正即乘回 πθ**：乘以 sg(πθ(y⋆|x)) 后，隐含奖励对所有专家轨迹均匀变为 1，类似 RLVR 给所有正确样本统一奖励，避免对低概率参考 token 的过度集中。

## 5. 主要解决思路（一段话讲清核心）
在标准 SFT 的逐 token 交叉熵损失上，乘以该 token 的模型预测概率（用 stop-gradient/detach 使该系数在反向传播中视为常数）。这一步把 SFT 隐含奖励里的 1/πθ 因子抵消掉，使每个 token 的有效奖励变为常数 1。整个方法仍以纯 SFT 形式实现（无需采样、无 reference model、无奖励模型），只是 loss 多乘一个 detach 后的概率权重。

## 6. 方法详解（通俗、分步骤）

1. **理论刻画**（eq.6）：\(\mathbb{E}[w(y \mid x) \cdot \nabla \log \pi_\theta(y \mid x) \cdot r(x,y)]\)，\(r(x,y) = \mathbb{1}[y = y^\star],\quad w = 1/\pi_\theta\)；这说明标准 SFT 是带病态奖励的特殊策略梯度。
2. **奖励整流**（eq.7）：\(\mathrm{sg}(1/w) = \mathrm{sg}(\pi_\theta(y^\star \mid x))\)，抵消 1/πθ；sg 保证梯度不流过该缩放项。
3. \(L = \mathbb{E}\big[ -\mathrm{sg}(\pi_\theta(y^\star \mid x)) \cdot \log \pi_\theta(y^\star \mid x) \big]\)。
4. **token 级稳定化**（eq.9，实际使用版本）：因整轨迹重要性权重会数值不稳，仿 PPO 在 **token 级**做重要性采样：\(L = -\sum_t \mathrm{sg}(\pi_\theta(y^\star_t \mid y^\star_{<t}, x)) \cdot \log \pi_\theta(y^\star_t \mid \cdots)\)。
5. **一行代码实现**（README/仓库 `fsdp_dft_trainer.py` 第 369–371 行，已核对）：
   ```python
   probs = torch.softmax(shift_logits, dim=-1)
   prob_coefficients = probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
   loss = loss * prob_coefficients.detach()
   ```

6. **效果直觉**（Fig.2 分析）：标准 SFT 把所有概率一律推向训练集；DFT 选择性地提高一部分、压低另一部分，弱拟合 token 的比例上升，相当于一种正则化。

## 7. 实验数据集

- **数学 SFT 训练**：从 NuminaMath-CoT 随机采 100k 条。基座 Qwen2.5-Math-1.5B/7B 等多模型多规模。
- **数学评测**：MATH-500、OlympiadBench、AIME 2024、AMC 2023、Minerva Math 等。
- **offline RL 设定**（Qwen2.5-Math-1.5B）：构造 100k 正负偏好对训 DPO；对比 DPO、RFT/RAFT（offline）与 PPO、GRPO（online，n=4 for GRPO）。
- **跨域**：代码生成（Table 3）、多模态推理数学（Table 4）。

## 8. 实验结果与主要发现

- **数学主实验**：在 Qwen2.5-Math 上 DFT 比标准 SFT 增益数倍；标准 SFT 常在 OlympiadBench/AIME/AMC 上**退化**，而 DFT 持续改善并提升泛化；增益跨模型、规模、数据量稳定。示例超参：train_batch=256、max_length=2048、lr=5e-5、1 epoch、use_liger=True、bf16。
- **offline RL（Table 2，Qwen2.5-Math-1.5B）**：DFT 平均 **35.43**，比最佳 RL 基线 GRPO 高 +3.43。逐项：MATH500 64.71（GRPO 62.86、PPO 56.10、RFT 48.23）；AMC23 48.44（比 GRPO +7.19、比 RFT +17.66）；Minerva 25.16（比 GRPO +6.23、PPO +9.75）。即 DFT 同时优于 DPO/RFT（offline）与 PPO/GRPO（online），且不需要 reference model 或大 batch。
- **跨域**：在代码生成与多模态推理上同样改善。
- **局限发现**：DFT 在"非确定性多解轨迹"任务（数学/复杂代码 CoT、信息丰富的多模态 CoT）上强，在"单一确定答案、低熵约束 CoT"任务上偏弱。

## 9. 结果如何支撑其主张

- "1/πθ 是泛化症结" → 校正后（DFT）相对 SFT 在多个困难基准上由退化转为持续提升，且 Fig.2 显示概率分布从"一律拉高"变为"选择性调整"，与理论预测的"整平病态奖励→更好正则化"一致。
- "可与 RL 竞争" → Table 2 上 DFT 平均分超过 PPO/GRPO，且不需奖励模型/reference model，支撑"streamlined alternative"主张。
- 但严格说，eq.6 的"SFT=策略梯度"刻画依赖若干假设（把数据生成策略当成 πθ、指示函数奖励），是一种**理论透镜**而非严格等价；论文也明确承认"RL-style characterization serves solely as a theoretical lens"。

## 10. 逻辑自洽性（中性评估）
推导链条（eq.5→6→7→8→9）清晰且与代码一一对应（detach 概率系数 = sg(πθ)），实现极简、可复核。最大概念张力在于：所谓"隐含奖励"是把 SFT 强行套进策略梯度框架后的产物，其"病态"本质上等价于"加权交叉熵的梯度对低概率 token 敏感"这一已知事实；DFT 的整流可被看作一种**置信度加权 / focal-loss 反向版**（降低对难拟合 token 的权重）。换言之，RL 透镜提供了优雅动机，但方法本身也能在不诉诸 RL 的情况下被理解为"按模型置信度重加权 SFT"。结论与证据自洽，只是机制解释存在多重等价视角。

## 11. 残留问题 / 局限

- **任务依赖**：在低熵/单一确定答案、强约束 CoT 任务上增益弱甚至可能不利（作者自陈）。
- **理论刻画的假设性**：eq.6 的策略梯度等价依赖把专家分布视作 Dirac、奖励视作指示函数等假设，是透镜而非严格证明。
- **置信度加权的双刃**：降低低概率 token 权重虽稳定训练，但也可能弱化对"模型当前不会、恰恰最该学"的 token 的学习；论文未充分讨论这一潜在反作用。
- **offline RL 对比的公平性**：DPO/RFT/PPO/GRPO 各自超参与数据构造不同，跨方法平均分比较的可比性需谨慎看待。〔待核〕Table 2 各基线是否经过同等调参未完全交代。

## 12. 开源代码与框架（链接+框架+代码可得性）

- 仓库 https://github.com/yongliang-wu/DFT（已克隆，约 66MB）。顶层含 `verl/`（训练）与 `math_evaluation/`（评测）。
- 训练器：`verl/verl/trainer/fsdp_dft_trainer.py`（FSDP SFT 训练器，第 369–371 行即 detach 概率重加权，已核对）；FSDP + Liger + bf16。
- 评测沿用 Qwen2.5-Math 仓库（latex 答案匹配）。
- 生态收录：TRL、LLaMA-Factory（`examples/extras/dft`）、ms-swift（`examples/train/full/dft.sh`）均一行启用；模型权重在 HF collection `Liang0223/dft-...`。
- 代码可得性：高（核心改动一行，多框架可直接复现）。

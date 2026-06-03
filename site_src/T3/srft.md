# srft — SRFT: A Single-Stage Method with Supervised and Reinforcement Fine-Tuning for Reasoning

> **一句话重点 (TL;DR)**：用熵感知（entropy-aware）的自适应权重，在单阶段内同时对同一模型施加 SFT（demonstration）与 RL（自探索 rollout），按当前策略熵动态平衡模仿与探索；建立在 LUFFY 之上，平均 59.1%、较 zero-RL 在 5 个数学基准 +9.0%、3 个 OOD 基准 +10.9%。

**元信息**：arXiv 2506.19767（v1, 2025-06-24，OpenReview n6E0r6kQWQ）｜ 中科院自动化所 / 国科大 / 美团 / 上海交大（Yuqian Fu, Tinghong Chen, Jiajun Chai 等，通讯 Dongbin Zhao 团队）｜ ICLR 2026 ｜ 主题 T3/T4（High，GFT 类 / SFT-RL 单阶段统一）｜ 代码 https://github.com/fyqqyf/SRFT （Tier A，已克隆 ~2.4MB，可跑）｜ 框架 veRL + vLLM（底座沿用 LUFFY 的 mix_src 结构与 deepscaler 奖励）

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/srft/fig_03.png)

*Figure 3: Learning dynamics during different fine-tuning paradigms in three-dimensional probability space. The number denotes the final performance of each training process.*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/srft/fig_01.png)


## 1. 相关工作与进展
SFT 与 RL 的结合是后训练的核心问题。传统做法两阶段串行（SFT 做 instruction-following → RL 做 alignment/reasoning），二者被视为独立阶段。近期 GFT/统一后训练方向（如 LUFFY 的 off-policy RL、混合策略训练）尝试把 demonstration 与 rollout 信号融合。本文直接建立在 LUFFY 之上（README 致谢，代码沿用 `mix_src`），创新点是熵感知的 SFT/RL 自适应加权。

## 2. 现有工作存在的问题

- 两阶段串行：SFT 易记忆模式而非习得真实推理、易过拟合；RL 样本效率低、探索困难、易 mode collapse。
- 集成不足导致误差传播、限制 RL 提升；过度依赖 demonstration 又过拟合、约束探索。如何在 SFT 的知识蒸馏与 RL 的策略优化之间定权重，是核心痛点。

## 3. Motivation
从熵视角对 SFT/RL 做机理分析，据此在两个范式间做"按需平衡"的权重调度，使单阶段统一训练既不过拟合 demonstration 又保持探索。

## 4. 主要灵感 / 核心直觉
作者三条关键发现：(1) SFT 对策略分布做"粗粒度全局"改变，RL 做"细粒度选择性"修改；(2) 单阶段集成优于串行；(3) 熵动态是训练有效性的关键指标，可据此在两范式间平衡加权——熵高时减弱模仿、保持探索，熵低时加强模仿。

## 5. 主要解决思路(一段话讲清核心)
单阶段同时对同一模型施加 SFT（demonstration）与 RL（on-policy rollout），用 entropy-aware weighting 把两者合成为单一 loss 反向：以当前策略熵自适应调节 demonstration 模仿强度与 RL 探索强度。

## 6. 方法详解(通俗、分步骤)
**SRFT（Supervised Reinforcement Fine-Tuning）**：

- 论文（Sec.4）两个熵感知权重解析式：SFT 权重 `w_SFT = 0.5·stop_grad(exp(−H(π_θ)))`（论文叙述：熵高时减弱模仿、熵低时加强）；RL 正样本目标权重 `w_RL = 0.1·stop_grad(exp(H(π_θ)))`（熵高时维持探索）。
- 代码层面（`mix_src/mix_core_alg.py` 的 `compute_token_on_off_policy_loss`）：
  - 正 advantage 项乘 `pos_entropy_exp_coeff = 0.1 * (entropy).exp().detach()`（L134，与论文 w_RL 一致 ✓）。
  - `entropy_exp_coeff = entropy.exp().detach()`（L162），`sft_loss = -entropy_exp_coeff * log_prob`（L163）；再在 `mix_actor.py` 中以 `policy_loss = policy_loss - sft_loss·sft_loss_coef`、训练脚本 `sft_loss_coef=-0.5`（train.sh L63）合成，即等效 `+0.5·exp(H)·(−log_prob)`。
  - ⚠️ **paper-code 符号差异（保留 flagged）**：论文写 SFT 权重 `exp(−H)`，代码实际用 `exp(+H)`，**符号相反**——以代码为准时 SFT 权重随熵升高而增大（与论文叙述方向相反）；0.5 系数则一致。RL 侧 `exp(+H)` 则 paper-code 一致。该差异已在论文与代码间双向核对确认。
- `mix_actor.py` 同时含 `policy_loss = pg_loss - entropy_loss·entropy_coeff`（`entropy_coeff=0.001`，train.sh L59）。
- **adaptive temperature**（可学习 `log_alpha` 对齐 target entropy，类 SAC 温度自适应）是代码中的**可选项、默认关闭**：config `use_adaptive_temperature: False`、`adaptive_temperature_target_entropy: 1.0`（`mix_ppo_trainer.yaml` L78/L81），主训练脚本未启用——非核心方法必备组件。

## 7. 实验数据集

- 训练：OpenR1-Math-46k-8192（openr1.parquet；OpenR1-Math-220k 的 46k 子集，源自 NuminaMath 1.5，带高质量推理 demonstration）+ on-policy rollout。
- 评测：5 个数学推理基准 + 3 个 OOD 基准（AIME24/AMC 用 avg@32；推理 temperature=0.6、max_gen=8192）。基座 Qwen2.5-Math-7B(-16k-think)。训练 64×A100（脚本 n_gpus_per_node=8、nnodes=4）。

## 8. 实验结果与主要发现
平均准确率 59.1%，较 zero-RL 方法在 5 个数学基准 +9.0%、3 个 OOD 基准 +10.9%（三项与 arXiv v1 摘要一致，已核）。熵感知调度使单阶段统一训练在数学与 OOD 上均稳定优于两阶段及纯 SFT/RL 基线。

## 9. 结果如何支撑其主张
跨数学 + OOD 双类基准的一致增益，配合熵动态分析（SFT 全局/RL 选择性、单阶段优于串行），支撑"用熵调度统一 SFT-RL"的主张。但增益相对 LUFFY 等强基线的具体边际、以及熵权重的消融贡献需看正文逐表（见局限）。

## 10. 逻辑自洽性(中性评估)
方法叙事自洽：熵机理分析 → 熵感知权重 → 单阶段统一。最大隐患是上述 paper-code 符号差异——论文用 `exp(−H)` 论证"熵高减弱模仿"，而开源代码实为 `exp(+H)`（熵高加强模仿），二者机制方向相反。若以代码为准，则论文对 w_SFT 的直觉解释不成立；这是使用方必须注意的自洽性裂缝（可能是论文笔误或代码 bug，作者未澄清）。

## 11. 残留问题 / 局限

- **paper-code 符号矛盾未解释**（见 §6/§10），影响对"熵如何调度模仿"的理解。
- adaptive temperature 默认关闭，论文若将其计入贡献叙事需谨慎。
- 仅 Qwen2.5-Math-7B 单基座、单训练集（OpenR1-46k），跨模型族/跨数据泛化未验证。
- 〔待核〕anonymous 项目页与 OpenReview 版本的逐表数字、以及熵权重各项的消融未逐一交叉核对。

## 12. 开源代码与框架(链接+框架+代码可得性)

- https://github.com/fyqqyf/SRFT （Tier A，已克隆 ~2.4MB，代码完整）。核心实现：`srft/verl/verl/mix_src/`（`mix_core_alg.py` 的 `compute_token_on_off_policy_loss`、`mix_actor.py` 的 loss 组装、`mix_trainer.py`）。模型权重：HuggingFace `Yuqian-Fu/SRFT`。
- 框架 **veRL + vLLM**（rollout/评测），底座沿用 LUFFY 的 mix_src 与 deepscaler 奖励。
- 关键超参（exp_scripts/train.sh）：train_batch_size=128、ppo_mini_batch=64、max_prompt=1024 / max_response=8192、actor lr=1e-6、temperature=1.0、val_temperature=0.6、kl_loss_coef=0、kl_loss_type=low_var_kl、entropy_coeff=0.001、sft_loss_coef=-0.5、tp=2、use_dynamic_bsz；adaptive temperature 默认关闭。

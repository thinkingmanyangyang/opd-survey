# cepo — CEPO: RLVR Self-Distillation using Contrastive Evidence Policy Optimization

> **一句话重点 (TL;DR)**：GRPO 给一条轨迹里所有 token 同一个优势，浪费在 filler 上、低估决定性步。CEPO 在每个 token 问"正确答案偏好它**且**错误答案反对它吗？"——用对比比率 \(P^{+}_T(y_t) / P^{-}_T(y_t)\)（正确/错误答案两个 teacher，错误答案取自组内已有的 rejected rollout）在 stop-gradient 下调制 GRPO 优势幅度，符号仍由 verifier 锚定；由于学生先验 P_S 被消掉，从构造上消除了 RLSD 的 fluency confound，在决定性 token 处锐化、filler 处恰好失效。

**元信息**：arXiv 2605.19436（v1，2026-05-19，cs.LG）｜ MBZUAI + Linköping University + Australian National University（Ahmed Heakl、Abdelrahman M. Shaker、Youssef Mohamed、Rania Elbadry、Omar Fetouh、Fahad Shahbaz Khan、Salman Khan）｜ Preprint，2026-05 ｜ 主题：RLVR self-distillation 的 token 级 credit assignment（多模态数学推理；与本项目 token 级监督/path-selection 相关，但偏 RLVR 而非纯 logit-OPD）｜ 代码 https://github.com/ahmedheakl/CEPO （已 clone，约 16MB）｜ 框架 veRL（经由 EasyR1）+ FSDP + vLLM

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/cepo/fig_01.png)

*Figure 1: Accuracy over 50 training steps. CEPO improves faster than GRPO and RLSD, reaching its largest gap around step 40 before partially converging by the final checkpoint.*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/cepo/fig_02.png)

*Figure 2: CEPO training pipeline and its relationship to GRPO and RLSD. Given a question x , the policy π θ produces G rollouts that are partitioned into correct ( G + ) and wrong ( G -) sets by a verifiable reward. CEPO conditions two frozen teachers on a sampled correct rationale r + ∈ G + and rej*

## 1. 相关工作与进展

- **RLVR / credit assignment**：RLVR 采样 rollout、verifier 打分、更新策略；GRPO 去掉 value 网络但赋予整条轨迹**均匀**的序列级优势。token 级方法要么靠 Monte Carlo 重模拟（VinePPO、SPO），要么训练独立的 PRM——都需昂贵重采样或辅助网络。
- **带特权信息的 on-policy 自蒸馏**：用正确答案 r⁺ 当 teacher 产生 dense token 级信号，无需辅助网络。OPSD（Zhao 2026）最小化 P⁺_T 与 student 的 per-token KL；SDPO（Hübotter 2026）扩成 JSD + EMA teacher；HDPO 专门用于全错的 prompt。
- **RLSD**（Yang 2026，CEPO 的直接前身）：在 stop-gradient 下只在采样 token 上算 evidence ratio \(P^{+}_T(y_t) / P_S(y_t)\)，仅用于调制 GRPO 优势幅度、符号锚定 verifier——做到既用特权信息又"leakage-free"（梯度里无 vocab-wide r-条件求和）。
- Table 1 用 Priv./Leak-free/Contr./No-Aux 四维定位 CEPO 是唯一四项全 ✓ 的方法。

## 2. 现有工作存在的问题

- **GRPO 太钝**：正确轨迹每 token 同正优势、错误轨迹每 token 同负信号；数学推理里单步算错或单个正确推断即可决定整条 CoT 成败，均匀 credit 把梯度浪费在 filler（连接词、格式、样板）上、低估少数决定性 token。
- **"用 r⁺ 当 teacher 做分布匹配"会泄漏**：RLSD 证明任何把 P⁺_T 当分布目标的 divergence 目标，其梯度都含 vocab-wide 的 r-条件求和（式 3），方差 ∝ I(Y_t;R⁺|X)，训练后期主导，逼模型编码 x→r⁺ 的伪相关（information leakage，与实现无关、不可约）。
- **RLSD 虽解决泄漏，但信号质量有三缺陷**：(1) **fluency confound**——分母 P_S 反映 base-rate 流畅度而非语义相关，常见 token 无论 r⁺ 多偏好它都被压低；(2) **asymmetric negative**——对错误轨迹的惩罚缺乏对 r⁻ 的显式 grounding；(3) **one-sided evidence**——P⁺_T/P_S 无法区分"正反答案都同等支持的 filler"与"r⁺ 支持 / r⁻ 反对的决定性步"，两者 ratio 相同时被同等加权。

## 3. Motivation
在每个 token 问更尖锐的问题：不仅"正确答案是否偏好该 token"，而是"正确答案偏好**且**错误答案反对吗"。两者皆满足→真正推理步；皆不满足→filler。且 wrong-answer teacher 可从 batch 中已有的 rejected rollout 构造，无额外采样开销。

## 4. 主要灵感 / 核心直觉

- 把单参照（只看 r⁺）升级为对比双参照（r⁺ vs r⁻）。用对比比率 P⁺_T/P⁻_T 后，**学生先验 P_S 在 stop-gradient 下完全抵消**，fluency confound 从构造上消失。
- 对比证据 delta 有清晰的贝叶斯解释（应用 RLSD Theorem 4 于两个 teacher 相减）：∆^CE_t = [r⁺ 的信念更新] − [r⁻ 的信念更新]，即"token y_t 同时把 r⁺ 后验抬高、把 r⁻ 后验压低多少"。决定性步得大正值，filler 趋近 0。
- filler 处改进恰好消失不是缺陷而是**正确性准则**——在信息中性位置放大梯度只会引入噪声。

## 5. 主要解决思路（一段话讲清核心）
CEPO = Contrastive Evidence Policy Optimization：定义三个共享参数 θ 但条件不同的分布——student P_S(y_t)=π_θ(y_t|x,y<t)、正确 teacher P⁺_T（条件 r⁺）、错误 teacher P⁻_T（条件 r⁻）。r⁻ 取组内最低 reward 的 rejected rollout 的最终答案（无额外采样）。对比证据 delta \(\Delta^{\mathrm{CE}}_t = \mathrm{sg}\!\left(\log P^{+}_T(y_t) - \log P^{-}_T(y_t)\right)\)；对比权重 \(w^{\mathrm{CE}}_t = \exp(\mathrm{sign}(A) \cdot \Delta^{\mathrm{CE}}_t) = (P^{+}_T / P^{-}_T)^{\mathrm{sign}(A)}\)；token 级优势 \(\hat{A}_t = A \cdot \left[(1-\lambda) + \lambda \cdot \mathrm{clip}(w^{\mathrm{CE}}_t, 1-\varepsilon_w, 1+\varepsilon_w)\right]\)，λ 从 λ₀ 线性退火到 0。把 Â_t 代入标准 PPO clipped surrogate 更新。G⁻=∅ 时令 P⁻_T=P_S，精确退回 RLSD。

## 6. 方法详解（通俗、分步骤）
**算法 1**：每次迭代、每个 (x,r⁺)：(1) 采 G 条 rollout，按式(1) 算组归一化序列优势 A，分成正确组 G⁺/错误组 G⁻；(2) r⁻ ← argmin_{j∈G⁻} R 的最终答案，G⁻=∅ 则 P⁻_T←P_S；(3) 对每条轨迹每个位置 t：∆_t ← sg(log P⁺_T(y_t) − log P⁻_T(y_t))，Â_t ← A·[(1−λ)+λ·clip(e^{sign(A)∆_t}, 1−ε_w, 1+ε_w)]；(4) 用 Â_t 做 PPO clipped surrogate 更新。

**理论保证（Theorem 1，证明见 Appendix A）**：(i) 方向锚定 \(\mathrm{sign}(\hat{A}_t) = \mathrm{sign}(A)\)（特权信息不能翻转任何 token 更新方向）；(ii) leakage-free 梯度（无 vocab-wide r-条件求和，r⁺/r⁻ 只作为采样 token 处的 stop-gradient 标量进入）；(iii) RLSD 包含性（P⁻_T=P_S 时精确退回 RLSD）。
**Proposition 1（区分锐度）**：正确轨迹下 w^CE_t > w^RLSD_t 当且仅当 P⁻_T(y_t) < P_S(y_t)（即错误答案相对学生先验更不喜欢该 token，恰是决定性位置）；错误轨迹对称；filler 处三者皆近 1、改进消失。

**成本**：CEPO 比 RLSD 多一次 teacher forward（r⁺/r⁻ 各一次），即比 GRPO 多两次 forward；无额外采样。Geo3k 50 步 wall-clock：GRPO 5h58m、SDPO 6h14m、RLSD 6h15m、CEPO 6h34m（多约 36 分钟）。

## 7. 实验数据集

- **训练**：Geo3K（3,000 道带可验证数值答案的几何题）。
- **评测**：5 个 held-out 多模态数学推理基准——DynaMath、LogicVista、MathVision-mini、MMMU、WeMath（不含 MathVista）。
- **模型**：Qwen3-VL-2B-Instruct、Qwen3-VL-4B-Instruct，LoRA（rank 16 / α 32）微调，50 步。
- **关键超参（Table 6/7，以论文为准）**：AdamW，lr 1e-6（**CEPO 用 5e-6**），cosine decay + 5 步 warmup，batch 32 prompts，group G=8，温度 1.0，max len 2048，PPO clip 0.20/0.28，无 KL/熵正则，rule-based 数值 verifier；CEPO 默认 λ₀=0.5、25 步线性退火到 0、ε_w=0.5。评测用 lmms-eval（温度 1.0、top-p 1.0、top-k 40、presence penalty 2.0、max 32k token）。
- 硬件：NVIDIA RTX6000 Pro Blackwell 100GB。
- 〔已核〕仓库脚本 `geo/cepo.sh`（2B）历史上有 lr 5e-6、cepo_lambda_init=0.5、warmup 25、eps_w 等设置，与论文超参基本一致（个别脚本数值可能与论文表略有出入，以论文 Table 6/7 为准）。

## 8. 实验结果与主要发现

- **主结果（Table 2）**：2B 五基准平均 CEPO 43.43% vs GRPO 41.17%（+2.26pp）、base 39.73%（+3.70pp）、OPSD 34.96%、SDPO 35.70%；4B CEPO 60.56% vs GRPO 57.43%（+3.13pp）、base 58.36%（+2.20pp）、OPSD 56.23%。增益最大在 LogicVista（4B 上 +6.18 over GRPO）与 MathVision-mini（2B 上 +4.94），最小在 MMMU（短链知识检索，2B 上 +1.67）。
- **OPSD/SDPO 跌破未训练 base**（2B 上 34.96/35.70 vs 39.73；4B 同样），实证印证 information leakage 理论预测，且不随规模消失。
- **消融**：teacher source（Table 4）actor-policy teacher 最优 43.43%（fixed reference 42.18、每 25 步同步 42.74）——teacher 新鲜度/on-policy 对齐比拉大师生分布差更重要，且 actor 共享权重省显存。feedback source（Table 5）"ground-truth r⁺ + peer answer-only r⁻"最优 43.43%；用整条 peer rollout 当负参照次之（42.74）；只 prefix/suffix 截断反而 < GRPO。超参（Figure 3）λ=0.5 与 25 步退火都 > GRPO，λ=1.0 反而引入噪声变差；ε_w 在 [0.4,0.5] 峰值，过小退回 GRPO、过大失稳。
- **机理分析**：训练中正 delta 占比上升、负 delta 占比下降（Figure 4）；token 权重热图（Figure 5）显示 CEPO 把 credit 锐化到关键代数推导与最终答案、把 filler 压到近 1，clip 率更低（49.5% vs RLSD 71.3%，即更宽的有效动态范围）。

## 9. 结果如何支撑其主张

- "CEPO > GRPO/RLSD" 由 Table 2 在两个规模、五基准上的一致增益支撑。
- "结构安全是实践前提" 由 OPSD/SDPO 跌破 base 这一反例强支撑——这也是论文最有说服力的实证。
- "在决定性 token 锐化、filler 处失效" 由 Prop.1 + token 热图 + delta 占比演化共同支撑，理论-实证闭环较好。
- "对比比率 ≠ 对比 KL"：脚注证明 \(\nabla\!\left[D_{\mathrm{KL}}(P^{+} \,\|\, P_S) - D_{\mathrm{KL}}(P^{-} \,\|\, P_S)\right]\) 会产生与 OPSD 同构的 vocab-wide leakage，凸显 CEPO 用 ratio + stop-gradient 的必要性。

## 10. 逻辑自洽性（中性评估）
理论（三保证 + Prop.1）与实证（OPSD/SDPO 退化、热图、delta 演化）方向一致，自洽性强。但**增益绝对值偏小**（base→CEPO 仅 +2.2~3.7pp），且训练只跑 50 步、单一数据集（Geo3k）、小模型（2/4B）、LoRA——更像"低预算下的快速收敛优势"（Figure 1 显示 CEPO 早期更快、约 step 40 差距最大、最终部分收敛）。强主张（"结构安全是实践前提"）证据扎实，但泛化主张需更大规模验证。

## 11. 残留问题 / 局限

- 仅在 Qwen3-VL 2/4B + Geo3k + 50 步 + LoRA 上验证；更大模型、纯文本推理、代码生成是 future work（作者自承）。
- 依赖组内同时有正确与错误 rollout 来构造 r⁻；全对/全错的 prompt 上对比信号退化（全错时退回 RLSD）。
- 增益小且部分收敛，长训是否保持优势未知。
- r⁻ 只取"最低 reward rollout 的最终答案"，对负参照的选择策略敏感（feedback source 消融显示 prefix/suffix 反而有害）。

## 12. 开源代码与框架（链接 + 框架 + 代码可得性）

- 仓库：https://github.com/ahmedheakl/CEPO （已 clone，约 16MB）。含 `examples/`、`experiments/`、`scripts/`（含 `geo/cepo.sh` 等复现脚本）、`verl/`（内置 veRL）、`Dockerfile`、`requirements.txt`、`setup.py`。
- 框架：veRL，经由 **EasyR1**（Zheng 2025，基于 verl 的多模态 RLVR 框架）+ FSDP + vLLM 加速 rollout。代码可得性：训练 + 评测脚本齐全，依赖明确，可复现性较好（个别脚本超参与论文表略有出入，以论文为准）。

# hpd — Hybrid Policy Distillation for LLMs (HPD)

> **一句话重点 (TL;DR)**：把 SFT、FKLD、RKLD 统一为 token 级 reweighted log-likelihood 目标，用 K1 估计器（Schulman 2020）在每个 token 上判 student 对 expert token 的 under/over-estimate，据此自适应混合 forward/reverse-KL 并重分配概率质量——既保留 one-hot 监督的计算效率，又兼容 off-policy 数据 + 轻量近似 on-policy 采样，从而以更省算力逼近 dense 蒸馏。

**元信息**：arXiv 2604.20244v1（2026-04-22）｜ 上海交大 / 上海创智学院 / 腾讯，通讯 Rui Wang(SJTU)、Ruobing Xie(Tencent)｜ Preprint（README 称 ICML 2026，但论文正文无该字样，〔待核〕）｜ 主题 知识蒸馏统一视角 + 混合 KL/混合 on-off-policy，与 OPD 直接相关（把 OPD 视为数据 regime 之一并用轻量采样降其开销）｜ 代码 https://github.com/zwhong714/Hybrid-Policy-Distillation （已 clone，约 40MB）｜ 框架 LlamaFactory（SFT 路径）+ veRL（RL 路径）。

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/hpd/fig_01.png)

*Figure 1. Comparison of training dynamics between SFT and HPD.*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/hpd/fig_04.png)

*Figure 4. Self-distillation Evolving. Stage 1: SFT + DPO/PPO initialization. Stage 2: Iterative self-distillation with teacher model updates, while keeping the SFT data fixed.*

## 1. 相关工作与进展
白盒 KD 可用 teacher logits 做分布级匹配（KLD）。FKL 促 mode coverage 但过平滑；RKL 促 mode-seeking 但在师生差距大时不稳。OPD（on-policy distillation）避免 train–inference 失配但开销大。既有 GKD/DistiLLM 系工作分别处理散度方向、优化策略（loss vs reward）、数据 regime（on/off-policy）。

## 2. 现有工作存在的问题
散度方向、优化策略、数据 regime 三轴被孤立选择、缺乏统一视角；单向散度各有缺陷；OPD 虽避失配但 teacher 侧对 student 输出有分布漂移且算力开销大。

## 3. Motivation
希望同时拿到双向散度的互补性与 one-hot 监督的计算效率，并天然兼容 off-policy 与轻量 on-policy，从而在算力受限场景务实地逼近 dense 蒸馏效果。

## 4. 主要灵感 / 核心直觉
把 SFT/FKLD/RKLD 看作同一 token 级 reweighted log-likelihood 目标的不同权重特例：FKL/RKL 用 teacher 全分布给 dense 监督但丢了 one-hot 的效率。用 K1 估计器在每个 token 上廉价判断"student 是否低估了 expert token"，据此决定走 FKL 还是 RKL，并把被抑制 token 的概率质量重分配回 expert token。

## 5. 主要解决思路(一段话讲清核心)
对每个 offline expert token 算 \(k_1 = q_\theta(a^* \mid s) \cdot (\log p(a^* \mid s) - \log q_\theta(a^* \mid s))\)：k1>0（student 低估 expert）触发 forward-KL 式增强、k1≤0（高估）取负值抑制；同时让学生在 offline 前缀下采一个替代 token、对其算 k'1，仅当 k'1<0（高估非 expert token）才以负权重抑制；当 k1>0 且 k'1<0 同时成立则把 expert 权重加倍，将从被抑制 token 释放的概率质量定向回流给 expert token。整套在保留 one-hot 效率的同时混合 FKL/RKL，且无额外超参。

## 6. 方法详解(通俗、分步骤)

- **expert 权重 w\*（Eq.12/14）**：`k1>0` → \(p(a^* \mid s) + k_1\)（forward-KL）；`k1≤0` → 取 `k1`（负，抑制）。
- **sampled token 权重 wₜ（Eq.13）**：学生在 offline 前缀下采 `aₜ≠a*`，算 `k'1`；仅 `k'1<0` 保留为负权重，`k'1≥0` 置 0。
- **reinforce 加倍（Eq.14）**：`k1>0 且 k'1<0` → expert 权重升为 \(2p(a^* \mid s) + k_1\)，把释放的概率质量重分配回 expert token。
- **效率/兼容性**：保留 one-hot 监督，天然兼容 off-policy 数据 + 轻量近似 on-policy 采样。
- 〔已核-代码〕`LlamaFactory/src/llamafactory/train/hpd.py::compute_hpd_loss` 与 Eq.11-15/Algorithm 1 逐式吻合：\(k_{1,\mathrm{gt\_raw}} = (\mathrm{teacher\_nll} - \mathrm{student\_nll}) \cdot \exp(\mathrm{student\_nll})\)、`mask3=mask1&mask2` 时 `adv1+=exp(teacher_nll)` 实现加倍；loss=`−student_nll·adv1 − adv2·sampled_student_nll·(labels≠sampled)`。论文消融明确 "HPD introduces no additional hyperparameters"（已核-PDF）。

## 7. 实验数据集

- **训练（蒸馏源）**：数学长 CoT 用 **OpenR1-Math-8192**（已核-PDF）；个性化/对话用 Ultrafeedback prompt；代码用 WizardCoder prompt。
- **评测**：数学 AIME24/AIME25/AMC/MATH/OlympiadBench/GPQA(OOD)；对话 AlpacaEval2(LC/WR)、Arena-Hard、MT-Bench；代码 HumanEval/MBPP（EvalPlus pass@1）。
- **模型/规模**：student=Qwen2.5(1.5B/3B, teacher 7B) 与 LLaMA3(1B/3B, teacher 8B)；代码 Qwen2.5-Coder(7B→1.5B)、DeepSeek-Coder(6.7B→1.3B)。teacher **非现成 instruct，而是先在 offline 数据 SFT 再用 GRPO 精调**（PSFT+RL，论文 §7 明确 "select Qwen2.5-7B-Base model as the teacher … SFT all the base models then …"）。

## 8. 实验结果与主要发现

- 数学（off-policy, avg.）：Qwen2.5-3B 28.25→**39.83**（+41%）、LLaMA3-3B 19.43→**34.56**（+77.9%），均显著超 SFT/SeqKD/RKLD/JSD。
- on-policy：HPD 单独即超 "SFT→OPD" 两阶段；HPD 作 OPD 初始化（HPD+OPD）再获最高分（Qwen2.5-1.5B **30.24→33.41**）。
- 另演示 HPD+DPO、迭代自蒸馏。

## 9. 结果如何支撑其主张
跨两个模型族、多规模、多任务（数学/对话/代码）的一致增益支撑"统一视角 + K1 混合"的有效性；HPD 单独超两阶段 SFT→OPD、且作 OPD 初始化再涨，支撑"高效逼近/补充 OPD"的卖点；消融逐项移除 K1 组件验证各部件贡献。

## 10. 逻辑自洽性(中性评估)
代码与公式逐式吻合，消融自洽。但"reweighted log-likelihood 统一"基本是对既有 GKD/DistiLLM 系工作的重述式归纳，真正新颖性在 K1-based 混合与质量重分配规则；"无额外超参"成立，但 teacher 经 SFT+GRPO 加工，增益与 teacher 质量耦合，统一框架的解释力被这一工程依赖部分稀释。

## 11. 残留问题 / 局限

- "approximate on-policy" 实为在 **offline ground-truth 前缀**下让学生采单个替代 token（非从学生 rollout 采整条轨迹），严格说是 off-policy 框架内的单步偏离纠正，称谓略宽松。
- 数学实验 student 仅 1B–3B、teacher 仅 7B/8B，长链推理增益是否随规模保持未充分验证。
- teacher 本身经 SFT+GRPO，蒸馏增益与 teacher 质量耦合。
- Preprint（2026-04），README 称 ICML 2026 但论文正文无此字样、未见正式接收证据，〔待核〕。

## 12. 开源代码与框架(链接+框架+代码可得性)

- 链接：https://github.com/zwhong714/Hybrid-Policy-Distillation （已 clone，约 40MB）。含 `LlamaFactory/`（SFT 式蒸馏 + HPD loss）、`verl/`（RL 式后训练）、`evaluation/`；提供 Qwen2.5-1.5B（从 Qwen2.5-7B-PSFT-RL teacher 蒸出，README 命名为 Qwen2.5-7B-Thinking）checkpoint。
- 框架：双后端 LlamaFactory（SFT 路径，全参/LoRA）+ veRL（RL 路径）。代码可得、核心 loss 可逐式对照。

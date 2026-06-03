# why_sd_degrade — Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?

> **一句话重点 (TL;DR)**：自蒸馏在数学推理上会"response 变短但性能反降（最高 ~40%）"，根因是 teacher 在富 context 下生成的自信轨迹压制了 epistemic verbalization（wait/hmm 等不确定性表达）；该压制由 conditioning 信息丰富度驱动，其危害随任务覆盖度增大而显现于 OOD。

**元信息**：arXiv 2603.24472v3（cs.CL，2026-05-20）｜ Microsoft Research + KAIST + SNU（Jeonghye Kim 实习于 MSR；Xufang Luo 通讯）｜ Preprint, under review ｜ 主题 自蒸馏/OPD 失败模式分析（**诊断/机理论文，无新算法**），与 OPD 直接相关：解释 privileged-context 自蒸馏为何在数学上"变短变差" ｜ 代码 https://github.com/beanie00/self-distillation-analysis （53M，已 clone，built upon lasgroup/SDPO）｜ 框架 veRL（仓库含 `verl/` 目录、megatron/sglang 脚本）

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/why_sd_degrade/fig_01.png)

*Figure 1: (a) Training score and response length changes for GRPO and Self-Distillation (SDPO) (H¨ ubotter et al., 2026) in Chemistry, using results from SDPO Wandb logs (Biewald, 2020) (link) . (b) Training score and response length changes on DAPO-Math-17k with GRPO and SDPO.*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/why_sd_degrade/fig_12.png)

*Figure 12: Comparison of per-word frequency shifts ( ∆ m ( w ) ) for epistemic tokens ( T ) versus non-epistemic tokens ( V \ T ) in GRPO- and SDPO-trained models relative to the base model. Epistemic tokens exhibit disproportionately large frequency changes compared to the general vocabulary, indic*

## 1. 相关工作与进展
自蒸馏（Snell 2022）：同一模型在 privileged context（如 ground-truth 解、环境反馈）下当 teacher，给无 context 的 student 提供 dense 信号。SDPO(Hübotter 2026) 条件于自身正确轨迹/环境反馈；OPSD(Zhao 2026) 用 ground-truth 解作 privileged 信息。这类方法在 agentic、科学推理等域高效提分，且常"response 变短、性能变好"。相关线：epistemic verbalization 框架(Kim 2026)；推理压缩(GFPO/OPSDC/ConPress/Accordion/CEEH)。本文建于 SDPO codebase 之上，专门追问"何时/为何自蒸馏会退化"。

## 2. 现有工作存在的问题
既有自蒸馏工作展示了有效性，但**未研究何时/为何会退化**——尤其在"模型需完全自主解题、无外部环境交互"的数学推理场景。把同一套自蒸馏(SDPO)用到数学时出现反常：response 仍随训练变短，但性能显著下降（最高掉约 40%）。"为何朝正确答案训练反而退化？"

## 3. Motivation
追因到 epistemic verbalization 被压制。把数学推理视为 self-Bayesian 推理：逐步在已生成 token 上更新对中间假设的信念；不确定性表达不是冗余，而是保留备选假设、支持渐进纠错的信号，过早自信会锁死错误假设无从恢复（Fig.2a）。teacher 拿到富 context（c=解）后生成低不确定性的自信轨迹，student 模仿即丢掉这些 epistemic 信号。

## 4. 主要灵感 / 核心直觉
用条件互信息 I(y*;c|x)=H(y*|x)−H(y*|x,c) 量化 context c 对正确答案 y* 的信息量。直觉：c 越富 → teacher 轨迹越简洁自信、epistemic token 越少；这在窄覆盖任务上加速 in-domain 收敛，但在宽覆盖/OOD 上因丢失不确定性信号而退化。标准训练目标不惩罚这种"风格漂移"，故 OOD 受损是"隐性"的。

## 5. 主要解决思路(一段话讲清核心)
不是提新算法，而是受控实证：固定/系统改变两个因子——(1) conditioning context 信息丰富度（用 I(y*;c|x) 形式化），(2) 任务覆盖度（训练题数 |D|）——观测 response length、score、10 个 epistemic token（wait/hmm/perhaps/maybe/actually/alternatively/seems/might/likely/check）频率随之如何变化，并对比 GRPO vs SDPO 的 in-domain 与 OOD 表现，从而定位"epistemic 压制 ↔ OOD 退化"的关联。

## 6. 方法详解(通俗、分步骤)
1. **自蒸馏目标（Eq.1）**：L=Σ_t KL(πθ(·|x,y_<t) ‖ stopgrad πθ(·|x,c,y_<t))，即 **student 向 teacher 对齐的标准 KL(student‖teacher)，是前向方向的自蒸馏目标**。〔已核并保留：此前 analysis 一度写"reverse-KL"不准——论文 Eq.1 是 forward-direction KL(student‖teacher)。〕
2. **§3 信息丰富度受控对比**：DeepSeek-R1-Distill-Qwen-7B 在 DAPO-Math-17k 选 100 题（base 8-rollout 准确率∈[0.125,0.5]），比较 4 种 c：(1)unguided c=∅、(2)solution c=s、(3)c=s\think、(4)regeneration c=yr。MI 排序 (1)<(3)≤(4)≤(2)。**Table 1**：随 I 增大，length 与 epistemic 计数单调下降——(1) score 0.30 / len 13,054 / E 182.5；(2) 0.98 / 1,873 / 8.8；(3) 0.78 / 12,036 / 159.8；(4) 0.95 / 2,808 / 24.1。
3. **§4 off-policy SFT 对照（Table 2）**：在 800 条正确轨迹上微调 DeepSeek-7B——D_ug（unguided，高 E、~12k tok）几乎不掉；D_sg（solution-guided，低 E、~2k tok）全面大跌（AIME24 54.79→20.21、AIME25 37.92→12.71、AMC23 89.06→57.03、MATH500 92.19→65.52）。证明"即便全是正确答案，过度压制 epistemic 也实质损害推理"。
4. **§5 on-policy 自蒸馏**：GRPO vs SDPO，DAPO-Math-17k 上跑 ~100+ step，比较 c=s 与 c=s\think。teacher **固定为初始策略（EMA rate 0.0）**优于移动 target。DeepSeek-7B：SDPO(c=s) 使 AIME24 ~40%、AMC23 ~15% 跌；c=s\think 缓解但仍低于 base。Qwen3-8B(think on/off)、Olmo3-7B 同向。Fig.3d/4d：SDPO 比 GRPO 更激进压制 epistemic token（尤以 wait 为甚）。
5. **§5.4 EMA 消融**：EMA 0.05（慢更新 teacher）比固定 teacher 退化更大——形成"越自信→teacher 越自信"的反馈回路。
6. **§6 任务覆盖度**：变 |D|∈{1,8,64,128,512}（Qwen3-8B think off）。|D|≤128 时 SDPO 快速高分且 len 减 8×（窄覆盖高效）；|D|=512 时 SDPO 反伤 score、且 OOD 全程低于 base，而 GRPO 随 |D| 增大靠增 epistemic 表达持续提升。

## 7. 实验数据集
- **作者自跑核心实验在数学**：DAPO-Math-17k（训练，14,000 distinct 题）；AIME24/25、AMC23、MATH-500（OOD 评测，题型不与训练重叠）。
- **Chemistry(ScienceQ&A) 与 LiveCodeBench v6 并非作者重跑**：Fig.1a 的 Chemistry 曲线直接取自 SDPO(Hübotter 2026) 的 W&B 日志，与 LiveCodeBench v6 一起仅用于 §6 的**任务覆盖度对比**（Table 3：Chemistry 共 2,400 题但仅 6 类题型/90-10 split；LiveCodeBench v6 仅 131 题且 train/eval 全重叠；vs DAPO-Math 14,000 题不重叠题型）。〔已核并保留此审计事实。〕
- 模型：Qwen3-1.7B/8B（think on/off）、DeepSeek-R1-Distill-Qwen-7B、Olmo3-7B-Instruct。

## 8. 实验结果与主要发现
- **Takeaway 1**：c 越富 → 越自信、epistemic 越少（Table 1 单调）。
- **Takeaway 2**：即便训练全是正确轨迹，过度压制 epistemic 也大幅降推理（Table 2 D_sg 全面跌）。
- **Takeaway 3**：on-policy 自蒸馏随 c 变富压制 epistemic、缩短 response，幅度依 base 原有不确定性水平而定。
- **Takeaway 4**：epistemic 的价值随泛化需求增长——窄覆盖近冗余可删，覆盖越宽越重要。
- 关键现象：math 上 SDPO 持续低于 GRPO 与 base（AIME24 ~40% 跌）；c=s\think 缓解但不消除；EMA 加剧；epistemic 频移（Fig.12）远大于全词表均移（30–40×），说明训练**专门**作用于 epistemic 表达而非均匀漂移。

## 9. 结果如何支撑其主张
- "信息丰富度→epistemic 压制"：Table 1 + 逐 token Fig.9 强支撑（单调、且 wait/maybe/perhaps 变化最大）。
- "压制实质有害（非纯风格）"：Table 2 off-policy SFT 是较干净的因果对照（同为正确轨迹，仅 epistemic 密度不同→大幅差异）。
- "覆盖度调制危害"：§6 |D| sweep + GRPO/SDPO 反向趋势支撑。
- "为何 chem 涨 math 跌"：Table 3 用覆盖度差异（题型数/train-eval 重叠）解释，但 chem 数据为**引用**而非重跑，属间接论证。
- 频移对照（Fig.12）排除"通用词表漂移"混淆，强化"训练特异性作用于 epistemic"。

## 10. 逻辑自洽性(中性评估)
- 机理叙事自洽：I(y*;c|x)→epistemic 压制→OOD 退化，由 MI 排序、Table 1/2、|D| sweep 串起。
- 但"epistemic verbalization 因果驱动正确率"主要靠相关性 + 受控对比，**尚非严格因果**（未做"强行注入/移除 epistemic token 看正确率"的反事实干预）。
- 跨域比较一半证据为引用 SDPO 日志，口径不完全一致；epistemic 用 10 个固定词近似（LLM-as-Judge 在 Appendix 作补充验证），多词短语未被单 token 计数捕获。
- 结论稳健但偏现象学，主张落在"后训练目标应显式保留不确定性表达"。

## 11. 残留问题 / 局限
- 缺反事实因果实验，"epistemic→正确率"为强相关而非证明。
- Chemistry/LiveCodeBench 为引用结果，跨域对比间接。
- 仅数学 OOD 基准；"何种程度的不确定性是恰当的"无可操作判据。
- 未给出修复算法（自承为分析论文）；与 caopd 同源（privileged-context 信息不对称→过自信），但 caopd 修 verbalized confidence 校准、本文关注 reasoning 链内 epistemic token 对正确率影响，二者互补。

## 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库：https://github.com/beanie00/self-distillation-analysis （53M，已 clone）。含 `analyzing_reasoning_behavior`、`experiments/math`（GRPO/SDPO 训练脚本）、`eval`、`data/math`（§6.2 缩减题集 + 评测集）、Docker、HF↔mcore 转换。
- **框架：veRL**（仓库含 `verl/` 目录；修改 `verl/trainer/ppo/ray_trainer.py` 的 `_remove_thinking_trace`）；代码 built upon lasgroup/SDPO，环境按 SDPO README 安装；transformers + vLLM/SGLang。
- 关键超参（README）：train_batch_size=256、max_response_length=20480、teacher_update_rate=0（主）/0.05（消融）、remove_thinking_from_demonstration 区分 c=s 与 c=s\think；训练在 4×B200。
- 代码可得性：完整开源 + HF 释出 Qwen3-8B(think on/off)/DeepSeek-Distill-7B checkpoint + W&B 日志。

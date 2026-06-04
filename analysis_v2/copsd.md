copsd | Crosslingual On-Policy Self-Distillation for Multilingual Reasoning (COPSD) | LMU Munich (CIS) + MCML（Yihong Liu*、Raoyuan Zhao*、Michael A. Hedderich、Hinrich Schütze） | 2026-05-10 · arXiv preprint v1 · cs.CL | 主题线 L1（OPD/自蒸馏）·相关性 高（与本项目 OPD 主轴直接同源）

**原始论文**：https://arxiv.org/abs/2605.09548

## 一眼看懂

> 一句话导读：低资源语言（如非洲语）的数学推理差，不是模型不会，而是"会但用这门语言调不出来"。COPSD 让同一个模型分饰两角：学生只看低资源语的题、必须用低资源语推理；老师额外偷看英文题 + 英文答案，因而能给出更靠谱的分布；学生在自己走的路径上逐 token 对齐老师。

- 🟦 TL;DR：核心观察——低资源语言的数学推理差，往往不是模型没能力，而是"有 latent（潜在）能力但在低资源语下调不出来"。原文："a model may possess the latent ability to solve a problem, yet fail to access that ability when … expressed in a low-resource language"【原文 §1 L101-104】。
- COPSD 把 OPSD（同一个模型分饰 student/teacher 的特权上下文自蒸馏）搬到跨语言场景：
  - student 只看译成低资源语的题 \(x^{(L)}\)，必须用目标语推理（这样才匹配推理时的真实条件）；
  - teacher 是**同一个模型**，额外拿到英文题 \(x^{(H)}\) + 英文参考解 \(y^*\)，因而能诱导出更可靠的分布；
  - 在 student 自己的 on-policy rollout 上做逐 token 散度蒸馏，梯度只过 student【原文 §4.1-4.2, Fig.2】。
- 优点：无需外部 teacher、无需目标语 rationale，dense（逐 token）信号远强于稀疏的 outcome（最终对错）奖励。
- 最巧的一步：**teacher 的特权信息 = 英文题 + 英文参考解，而 student 只看低资源语题**。
  - 这个"跨语言信息不对称"是全部增益的来源——teacher 看到英文解能 rationalize（合理化推导）出强分布，student 在自己的目标语 rollout 上对齐它，等于把"英文可及的推理行为"迁移到低资源语【原文 §4 L385-395】。
  - 抽掉"英文参考解 \(y^*\)"，teacher 就退化成"只多看一份英文翻译"，特权信号大幅削弱。
  - 但这也是双刃剑——teacher 见到 \(y^*\) 等于把答案信息蒸进 student，使"特权"接近"漏答案"（见祛魅）。

## 为什么做

> 一句话导读：低资源语推理差是因为能力调不出来。机翻推理过程噪声大又 off-policy，纯靠最终对错做 RL 信号又太稀疏。需要一个既密集、又跟模型自己生成的轨迹对齐的信号——OPSD 正好提供。

- 研究背景：LLM 数学推理靠 step-by-step CoT（思维链，Wei 2022），但跨语言极不均衡，低资源语在预训练里曝光少、后训练的高质量推理监督也罕见【原文 §1 L39-104】。
- 解决的具体痛点，两条：
  - ①**机翻 reasoning trace 噪声大且 off-policy**——数学表达/数量/逻辑依赖容易翻错（原文："prone to inconsistencies or errors in mathematical expressions, quantities, and logical dependencies"），而且与模型自身的推理行为不匹配，造成 train-inference distribution mismatch（训练-推理分布失配）【原文 §1 L110-122】；
  - ②**outcome-only RL（GRPO）稀疏不稳**——低资源下模型很少产出正确答案，二元奖励"provides little information about how intermediate reasoning should be improved"，样本低效、还可能不稳【原文 §1 L123-137】。
  - 需要的是"既 dense 又 scalable、且与模型在低资源语实际产生的轨迹对齐"的信号。
- 相关工作 & 各自不足【原文 §2】：
  - **OPD**（Agarwal/GKD、Gu/MiniLLM、Lu & Thinking-Machines 2025）：结合 on-policy 监督 + dense token teacher 反馈，缓解失配、避开稀疏序列奖励，但**需要外部 teacher**；近期还发现"effective OPD requires compatible teacher-student thinking patterns"（Li 2026）——师生思路不匹配会阻碍迁移；
  - **OPSD**（Zhao 2026b / Zhang 2026a / Kim 2026 / Sang 2026）：用同一个模型分饰 student/teacher，免去外部 teacher——COPSD 直接 build on 这一支；
  - **多语言推理**现有解：translate-and-test pipeline、SFT（机翻 rationale）、self-training、RL——但"typically require translated reasoning rationales or sparse outcome rewards"（要么需要翻译好的推理过程、要么靠稀疏 outcome 奖励），而 COPSD 两者都不需要（只需英文上下文当特权信息）【原文 §2 L248-255】。
- 动机链：低资源推理差（latent 能力调不出）→ 机翻 rationale 噪声 + off-policy、outcome RL 太稀疏 → 需要 dense + on-policy + 对齐目标语轨迹的信号 → OPSD 正好提供（同模型、dense、on-policy）→ 把它的特权信息从"参考解"扩成"英文题 + 英文解"、输入换成低资源语 → COPSD。为什么不用更简单做法：SFT 机翻 rationale 引入噪声 + 失配，GRPO 信号太稀疏，二者在低资源数学上都被实证打败（GRPO 几乎不动，1.7B 9.11→9.18）。
- 与最近邻工作的Δ：相对 OPSD（siyan-zhao/OPSD），**核心公式完全不变**，只把 teacher 的特权信息从"reference solution \(y^*\)"扩成"英文问题 \(x^{(H)}\) + 英文解 \(y^*\)"，student 输入从原语言换成低资源语 \(x^{(L)}\)【原文 §4.1 L407-416】；额外加了 prompt-hacking 强制 student/teacher 都用目标语推理（在 `<think>` 后插语言特定前缀，否则模型会切回英文）【原文 §5.2 L668-680】。有用点：把"特权信息"具体化为"高资源语上下文"，**首次系统验证 17 非洲语 + PolyMath 8 语**。

## 怎么做 + 靠不靠谱

> 一句话导读：取英文题 + 英文解，机翻出低资源语的题；学生用低资源语生成轨迹，老师偷看英文答案在同一前缀上给分布，逐 token 对齐。验证很系统（17 语），但有两点诚实张力：论文说"全词表"实际代码是 top-20、老师看到答案使"特权"接近"漏答案"。

- 方法流水线（读完可复现）【原文 §3-4, Fig.2】，五步：
  1. **取英文题 \(x^{(H)}\) + 英文参考解 \(y^*\)**（数据集 OpenThoughts，采 0.5K 数学题），再**用 Gemini-3-Flash 把题机翻到目标低资源语 \(x^{(L)}\)**（只翻题，不翻 rationale）。
  2. **从同一模型 \(p_\theta\) 实例化两个策略**（只是 conditioning 不同）：student \(p_S(\cdot\mid x^{(L)})\triangleq p_\theta(\cdot\mid x^{(L)})\)（只看低资源题，匹配推理时条件）；teacher \(p_T(\cdot\mid x^{(L)},x^{(H)},y^*)\triangleq p_\theta(\cdot\mid x^{(L)},x^{(H)},y^*)\)（额外吃英文特权上下文）。
  3. **student 在线生成目标语 rollout** \(\hat y^{(L)}=(\hat y^{(L)}_1,\dots)\sim p_S(\cdot\mid x^{(L)})\)，用 prompt-hacking（`<think>` 后插目标语前缀）强制目标语推理。
  4. **两策略评估同一 student 前缀**：在每步 \(n\)，\(p^n_S\triangleq p_S(\cdot\mid x^{(L)},\hat y^{(L)}_{<n})\)、\(p^n_T\triangleq p_T(\cdot\mid x^{(L)},x^{(H)},y^*,\hat y^{(L)}_{<n})\)。
  5. **逐 token 散度蒸馏**，梯度**只过 student**（teacher 冻结，作固定目标）。
- 关键公式（真实形式 + 直觉）：
  - **OPSD 两策略**（§3.1）：\(p_S(\cdot\mid x)\triangleq p_\theta(\cdot\mid x)\)，\(p_T(\cdot\mid x,y^*)\triangleq p_\theta(\cdot\mid x,y^*)\)——共享参数 \(\theta\)，teacher 多 condition 在特权信息 \(y^*\) 上。
  - **轨迹平均的逐 token 散度**（§3.3 / §4.2）：
    \(\displaystyle D_{\text{COPSD}}(\hat y^{(L)}\mid x^{(L)})=\frac{1}{|\hat y^{(L)}|}\sum_{n=1}^{|\hat y^{(L)}|}D\bigl(p^n_T\,\Vert\,p^n_S\bigr)\)
  - **总目标**（梯度只过 student）：
    \(\displaystyle \mathcal{L}_{\text{COPSD}}(\theta)=\mathbb{E}_{(x^{(L)},x^{(H)},y^*)\sim D}\;\mathbb{E}_{\hat y^{(L)}\sim p_S(\cdot\mid x^{(L)})}\Bigl[D_{\text{COPSD}}(\hat y^{(L)}\mid x^{(L)})\Bigr]\)
    直觉：teacher 看着答案"边讲边纠"，student 在自己走的目标语路径上每步对齐这个更靠谱的分布。
  - **散度的真实实现**（代码核查，generalized JSD，docstring 引 GKD 论文 2306.13649）：用插值系数 \(\beta\)，发布脚本全部取 \(\beta=0\)：
    \(\displaystyle D=\sum_t \texttt{F.kl\_div}(\log p^t_S,\;\log p^t_T,\;\text{log\_target=True})=\sum_t\sum_v p^t_T(v)\bigl(\log p^t_T(v)-\log p^t_S(v)\bigr)\)
    即 \(D_{KL}(p_T\Vert p_S)\)（以 teacher 为 target 的 KL）。
    - 【待核】论文正文称此为 **"reverse KL"**（§5.3 L698），但按 PyTorch `F.kl_div(input,target)` 约定，\(\beta=0\) 分支以 teacher 为 target、是 GKD 习惯下的"前向/teacher-anchored KL"。命名取决于"以谁为参照"的视角，复现以代码 \(\beta=0\) 分支为准（本次直接核 `multilingual_opsd_trainer.py` L467-468 确认）。
- 逐组件必要性：
  - **特权英文上下文（\(x^{(H)}+y^*\)）**：核心，是 teacher 强分布的来源；没它就退回普通自蒸馏、无跨语言迁移。无单独"\(x^{(H)}\) vs \(y^*\)"的消融，但概念上 \(y^*\) 是关键。
  - **on-policy（在 student 自己的 rollout 上蒸）**：对齐推理时条件、减少失配——这是 OPD 优于机翻-SFT 的根本【原文 §3.4】。
  - **prompt-hacking 强制目标语**：负责防"模型切回英文推理"——若不强制，student 不会真正用低资源语推理，迁移就落空【原文 §5.2 L668-680】。
  - **top_k + jsd_token_clip（代码层，论文未强调）**：【代码核查】发布脚本（1.7B/4B/8B 及 3000 变体全部）实为 **`--top_k 20`**（只在 teacher top-20 token 上算散度并重归一化，docstring：reduces memory and focuses distillation on teacher's most probable tokens）+ **`--jsd_token_clip 0.05`**（逐 token 散度截断，docstring：prevents style tokens from dominating the gradient over math tokens）+ **`--fixed_teacher`**——与论文 §5.3"full-vocabulary logit distillation"的表述**有出入**，复现以脚本为准（本次直接核 repo 确认 top_k=20、jsd_token_clip=0.05、beta=0）。
- 实验与证据【原文 §5-6】：
  - **数据集/设置**：训练 OpenThoughts 采 0.5K 数学题，题面用 Gemini-3-Flash 译成 17 非洲语；英文题 + 英文解作 teacher 特权；评测 **AfriMGSM**（17 语 × 250 题，**Pass@12**，主表 4096-token 预算）；**PolyMath**（更难，8 语 × 125 题，8192-token，含 low/medium/high 难度）；模型 **Qwen3-1.7B/4B/8B**；指标 Pass@12（采 12 条，Math-Verify 抽 `\boxed{}` 比对）。GRPO baseline 训练用 16K-token 预算。
  - **AfriMGSM 主表（Table 1）**：1.7B 平均 **9.11(base)/9.18(GRPO)→15.53(COPSD)**；4B **19.20/19.36→20.61**；8B **19.41/19.22→23.55**。**GRPO 几乎无提升**（1.7B 9.11→9.18，多语言上偶尔还低于 base，如 4B SWA 47.60→46.00、8B LUG 15.20→8.40）——印证低资源 outcome RL 信号太稀疏。1.7B 相对提升 **>70%**。
  - **训练动态（Fig.3, 1024-token 预算）**：COPSD 早期快速提升 Pass@12 与 format rate；1.7B 最终 plateau；**4B/8B 几步达峰后缓降**（原文："useful signal may be limited … continued updates may begin to overfit to imperfect teacher signals"，因为目标语生成弱、teacher 信号有限）；GRPO 无上升趋势【§6.1 L860-880】。
  - **format adherence（Table 2）**：format rate 与 Pass@12 **强正相关**（per-language Pearson 0.628/0.838/0.728 for 1.7B/4B/8B；pooled 更低但仍正）——大模型达峰后 format rate 下降也解释了 Pass@12 的回落【§6.1 L973-996】。
  - **test-time scaling（Table 3, Fig.4）**：大模型更稳定地受益于更长生成预算（8B COPSD 1024→4096 **+30.0%** vs GRPO +13.8%；Zulu 8B 4096-token 达 ~28% vs base/GRPO ~16%）；1.7B 增益较小、GRPO 在 2048 预算上 scaling 不稳【§6.2】。
  - **repeat rate（Fig.5）**：COPSD 显著降低 4-gram 重复率（缓解低资源推理常见的"repetitive loops/circular repetition"失败模式）【§6.3】。
  - **PolyMath 泛化（Fig.6）**：**低资源语增益最大**（medium 难度 Swahili +32.0、Telugu +32.8、Bengali +15.2；high 难度 SWA +18.4/TEL +16.8），高资源语（日/中/俄/西）增益小甚至为负（medium THA −2.4）——直接支撑"latent 能力难经低资源语表达"的假设（原文："most effective when the model already possesses latent reasoning ability but struggles to express it through lower-resource language contexts"）【§6.4 L1289-1292】。
  - **baseline 公平吗**：base / GRPO / COPSD 同模型同数据对照，公平。但**只跟 base + GRPO 比，没和"SFT 机翻 rationale"等同类多语言方法直接对照**——这是缺口。
  - **看着强但没回答核心问题**：4B 绝对增益很小（19.20→20.61，仅 +1.41），且"teacher 见参考解"使增益部分来自"答案泄漏"而非纯粹的推理迁移（见祛魅）。
- 假设与失效边界：
  - 【原文 Limitations】依赖**英文参考解**——若高质量英文监督不可得、或另一门高资源语更合适时，受限。
  - 【原文 Limitations】训练题靠机翻——题面翻译的 artifact（瑕疵）仍可能影响训练质量（虽然不需翻译 rationale）。
  - 【原文 Limitations】同模型当 teacher——目标语能力弱时 teacher 分布仍不完美，学习信号"may saturate quickly or degrade with continued training"（4B/8B 已观察到达峰后降）。
  - 【代码核查·待核】论文"full-vocab / reverse KL"的表述与发布脚本（top_k=20 + jsd_token_clip=0.05 + beta=0 的 teacher-anchored KL）有出入——见上"关键公式"。
- 祛魅总结【推断】：真贡献是把 OPSD 清晰地迁移到跨语言 + 17 语系统验证，把"特权信息 = 高资源语上下文"这一设定落地。增量有限——OPSD 公式不变，新意在输入条件与系统性。两点诚实张力：(1)**论文"full-vocab"与发布脚本（top_k=20 + clip）不符**，"全分布"程度被高估了；(2)**teacher 见 \(y^*\) 使"特权"接近"漏答案"**——绝对增益（尤其 4B 仅 +1.41）应在此背景下保守解读，这不是"无答案的纯推理迁移"。4B/8B 几步达峰后下降也暴露 teacher 信号在弱目标语上很快饱和。被低估的可能是"format adherence / repeat rate 改善"这类副产品对低资源生成质量的实际价值。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：teacher（同模型 + 英文特权上下文）在 student 自己 rollout 上的 dense token 级分布（\(\beta=0\) 的 teacher-anchored KL，实现上 top-20 + clip）。
  - **改什么**：参数（student 的 LoRA adapter；teacher 用 `--fixed_teacher`/`disable_adapter()` 即 base 模型，梯度只过带 adapter 的 student，同一份基座权重）。
  - **何时改**：在线 per-step（on-policy，在 student 当前 rollout 上蒸）；训练若干步即达峰（4B/8B 几步）。
  - **免梯度?**：否（梯度蒸馏）；teacher 冻结、作固定目标。
  - **记忆-技能生命周期**：无外部记忆/技能库；"知识"是 teacher 的英文推理分布，即时蒸进 student 参数。
  - **防遗忘机制**：无显式防遗忘；LoRA 本身限制了改动范围；但论文观察到"过训练→退化"（达峰后降），这反而是缺乏防过拟合机制的体现。
- ⑦ 开源代码+框架/harness：https://github.com/cisnlp/COPSD （README 明确 fork 自 siyan-zhao/OPSD，已 clone；本次核查目录有 `multilingual_opsd_trainer.py`/`multilingual_opsd_train.py`/`multilingual_data_collator.py`/`multilingual_grpo_train.py`/`language_config.py` + `african_langs_scripts/`（17 非洲语 run_all_opsd_{1.7b,4b,8b}_train.sh）+ `polymath_eval/` + `multilingual_scripts/run_all_opsd_4b_3000.sh`）。框架 **HuggingFace TRL（OPSDTrainer 继承 SFTTrainer，generalized_jsd_loss 即 GKDTrainer 血统）+ Accelerate/DeepSpeed（ZeRO-3）+ vLLM rollout**，LoRA。**4B 脚本核实超参**：base=Qwen/Qwen3-4B，lr **5e-6**，max_grad_norm **0.1**，lora_r=**64**/α=**128**，per_device_batch=**4**，num_train_epochs=30 / max_steps=**100**，top_k=**20**，beta=**0**，jsd_token_clip=**0.05**，`--fixed_teacher`，student max gen len 2048。**代码额外能力（论文未强调）**：trainer 还实现了 EMA teacher（`use_ema_teacher`，与 fixed_teacher 互斥）、`reason_first`（teacher 先对英文解推理再评估 student）、thinking-machines RL 式 reverse-KL（仅采样 token 上 policy-gradient，advantage=(teacher_logp−student_logp).detach()）；主实验用 generalized-JSD 路径（beta=0）+ fixed_teacher【既有 analysis 代码核查 + 本次 repo 复核 L408-483】。
- 💰 资源/成本与可扩展性：LoRA 训练（r=64/α=128），per_device_batch=4，A100 或 H200【4B 脚本核实】；训练数据仅 0.5K 题、每语单独训。可扩展性：三规模（1.7/4/8B）都验证，大模型更受益于长生成预算；但 teacher 信号在弱目标语上快速饱和，限制了"多训更好"。
- 🎯 对"探索-巩固"对标：**支撑（OPD 主轴的直接同源工作）**。
  - COPSD = 标准 OPSD 的跨语言变体，与本项目"teacher 当稀疏脚手架、on-policy 自选轨迹、dense token 监督"高度同源：它正是"teacher 提供更可靠分布 + student 在自己 rollout 上对齐"的范式。
  - **关键 Δ vs 本项目**，三点：
    1. teacher 特权 = 静态英文上下文（看完整答案），不是本项目设想的"稀疏脚手架/前瞻探针"——它更像"全程看答案的家教"，而非"只在关键点给提示"；
    2. 无 path-recovery 的"走偏后单点接管/自选恢复分支"机制，是全程逐 token 对齐；
    3. 无 MTP 前瞻、无技能/记忆。
  - **可借组件**，三点：
    1. OPSD/COPSD 的代码与 trainer（TRL SFTTrainer + 双前缀 collator + fixed_teacher via disable_adapter）是现成的"同模型双角色"实现模板；
    2. `top_k=20` + `jsd_token_clip=0.05` 抑制 style token 主导梯度，对本项目"只在关键 token 上蒸"有直接借鉴（与高熵/低置信切关键步的思路相关）；
    3. 它暴露的"teacher 信号饱和 → 过训练退化"是本项目设计"何时停/何时撤脚手架"的反面教材。
  - **缺口**：特权信息太强（漏答案）、无选择性/前瞻——本项目要的是更"稀疏、点状、前瞻"的脚手架。
  - 一句判定：同范式同源、直接可借代码与 token 级蒸馏技巧，但其"全程看答案"的特权设定与本项目"稀疏脚手架"哲学相反，是有用的对照与零件来源。
- 🔭 开放问题/未来方向：
  - 【原文 Limitations/未来】摆脱对英文参考解的依赖（高资源语监督不可得时）；减小机翻 artifact 的影响；缓解同模型 teacher 在弱目标语上的信号饱和/过训练退化。
  - 【推断】把"全程看答案"的特权弱化为"稀疏/点状提示"（更贴近脚手架直觉），看是否仍能迁移——这正是本项目方向；选择性蒸馏（只在 student 走偏/高熵步对齐 teacher），避免无差别逐 token 对齐带来的饱和；teacher 新鲜度调度（EMA vs fixed，代码已有 `use_ema_teacher` 但未系统比）。

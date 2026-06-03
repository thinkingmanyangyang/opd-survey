why_sd_degrade | Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs? | Microsoft Research + KAIST + Seoul National University（Jeonghye Kim 实习于 MSR；Xufang Luo 通讯） | 2026-05-20·arXiv preprint, under review·v3（cs.CL） | 主题线 L1（OPD/自蒸馏，失败模式诊断）+ L6（CoT/不确定性表达）·相关性 高

**原始论文**：https://arxiv.org/abs/2603.24472

## 一眼看懂
- 🟦 TL;DR：自蒸馏（同一模型在富 context=正确解 下当 teacher，给无 context 的 student 打稠密信号）在很多域（化学/工具/代码）能"response 变短、性能变好"，但搬到**数学推理**上会反常——response 照样变短、性能却大跌（最高掉约 40%）。本文不提新算法，而是用受控实证把根因锁定在 **epistemic verbalization（认知性不确定表达，如 wait/hmm/maybe）被压制**：teacher 拿到富信息后生成的轨迹简洁自信、几乎不带不确定词，student 模仿就丢掉了这些"标记备选假设、支持渐进纠错"的信号；危害是否显现取决于**任务覆盖度**——窄覆盖时压制反而高效，覆盖越宽/越 OOD 越有害【原文 Abstract、§1、Takeaway 1-4】。
- 最巧的一步：**用条件互信息 I(y*;c|x)=H(y*|x)−H(y*|x,c) 把"context 信息丰富度"量化**，再构造 4 种 c（unguided ∅ / 全解 s / 去 think 的 s\think / 用 s 再生成的 yr），论证其 MI 排序 (1)<(3)≤(4)≤(2)，并观测 length、score、10 个 epistemic token 计数随 I 单调变化（Table 1）。抽掉这个 MI 框架，"信息越富→越自信→epistemic 越少"就只是定性观察、无法把"风格漂移"和"性能退化"挂到一个可控变量上【原文 §3、式2-3】。

## 为什么做
- 研究背景：自蒸馏（Snell 2022）作为 LLM 后训练范式，配 RLVR 在 agentic/科学推理域高效提分，且常伴随"response 变短、性能变好"（SDPO Hübotter 2026、OPSD Zhao 2026 等）【原文 §1】。
- 解决的具体痛点：把同一套自蒸馏（SDPO）用到数学时出现反常——response 仍变短但性能显著跌（最高 ~40%）。既有自蒸馏工作只展示有效性，**未研究"何时/为何会退化"**，尤其在"模型需完全自主解题、无外部环境交互"的数学场景【原文 §1、附录 A】。
- 相关工作 & 各自不足：① SDPO（Hübotter 2026）条件于自身正确轨迹/环境反馈做 dense 信用分配、② OPSD（Zhao 2026）用 ground-truth 解作 privileged 信息——均展示有效但不问退化条件，本文**建于 SDPO codebase 之上**；③ epistemic verbalization 框架（Kim 2026）证不确定表达对鲁棒推理"信息上必要"，但答不出"何时被鼓励/压制、为何压制在化学有益数学有害"——本文正是回答这两问；④ 推理压缩线 GFPO/OPSDC/ConPress/Accordion/CEEH——本文补充说明"压制不确定表达的压缩即便答对也害 OOD"【原文 §1、附录 A】。
- 动机链：自蒸馏数学上反常退化→追问"朝正确答案训为何更差"→数学=self-Bayesian 推理，不确定表达保留备选假设、支持纠错，过早自信会锁死错误假设（Fig.2a）→teacher 拿富 context 生成低不确定自信轨迹→student 模仿丢掉 epistemic 信号→标准目标不惩罚这种风格漂移→OOD 退化是"隐性"的【原文 §1-2】。
- 与最近邻工作的Δ：相对 Kim 2026（提出 epistemic verbalization 必要性）的Δ——本文识别出**两个调制因子（信息丰富度 I(y*;c|x) + 任务覆盖度 |D|）**并系统扫描多模型，解释"同一压制机制为何化学涨数学跌"。为什么有用：把一个看似矛盾的现象（自蒸馏有时好有时坏）统一到"覆盖度 × epistemic 保留"的可操作框架，提示后训练目标须显式保留不确定表达。

## 怎么做 + 靠不靠谱
- 方法流水线（诊断而非算法，输入→输出）：固定/系统改变两因子 →（§3）变 c 的信息丰富度测 length/score/epistemic 计数 →（§4）off-policy SFT 对照（同为正确轨迹、仅 epistemic 密度不同）→（§5）on-policy GRPO vs SDPO 多模型 + EMA 消融 →（§6）变任务覆盖度 |D| →输出"epistemic 压制 ↔ OOD 退化、受覆盖度调制"的关联与四条 takeaway【原文 §3-6】。
- 逐组件（这里"组件"=各支证据，逐一看其必要性/能否被驳）：
  - **§3 信息丰富度受控对比（Table 1）**：DeepSeek-R1-Distill-Qwen-7B 在 DAPO-Math-17k 选 100 题（base 8-rollout 准确率∈[0.125,0.5]），4 种 c。结果随 I 增大 length 与 epistemic 单调降：(1)unguided score 0.30/len 13,054/E 182.5；(2)solution score 0.98/len 1,873/E 8.8；(3)s\think 0.78/12,036/159.8；(4)regeneration 0.95/2,808/24.1。**强支撑 Takeaway 1**【原文 Table 1、式4-5】。
  - **§4 off-policy SFT 因果对照（Table 2）**：在 800 条正确轨迹上微调——D_ug（unguided，高 E、~12k tok）几乎不掉（AIME24 54.79→51.04 等）；D_sg（solution-guided，低 E、~2k tok）全面大跌（AIME24 54.79→20.21、AIME25 37.92→12.71、AMC23 89.06→57.03、MATH500 92.19→65.52）。**这是最干净的因果对照**：两数据集都是正确答案，唯一差异是 epistemic 密度→大幅性能差，证明压制是实质有害而非纯风格【原文 Table 2、Takeaway 2】。
  - **§5 on-policy GRPO vs SDPO**：DAPO-Math-17k 上 ~100+ step。DeepSeek-7B：SDPO(c=s) 使 AIME24 ~40%、AMC23 ~15% 跌，c=s\think 缓解但仍低于 base；Qwen3-8B(think on/off)、Olmo3-7B 同向。Fig.3d/4d：SDPO 比 GRPO 更激进压制 epistemic（尤以 wait 为甚）。teacher **固定为初始策略**【原文 §5.1-5.3、Fig.3-5】。
  - **§5.4 EMA 消融**：EMA 0.0（固定 teacher）优于 EMA 0.05（慢更新）——慢更新形成"越自信→teacher 越自信"反馈回路，退化更大。**有消融，支撑反馈回路解释**【原文 §5.4、Fig.6】。
  - **§6 任务覆盖度 sweep**：变 |D|∈{1,8,64,128,512}（Qwen3-8B think off）。|D|≤128 时 SDPO 快速高分、len 减 8×（窄覆盖高效）；|D|=512 时 SDPO 反伤 score、且 OOD 全程低于 base，而 GRPO 随 |D| 增大靠增 epistemic 持续提升。**支撑 Takeaway 4**【原文 §6.2、Fig.7-8、Fig.11】。
  - **Fig.12 频移对照（排除混淆）**：全词表平均每词 shift |Δ|<1，而 10 个 epistemic token shift 大 30-40×（SDPO 达 −11.9/−12.2）——证明训练**特异性**作用于 epistemic 表达而非均匀词表漂移。**这步排除了"通用风格漂移"的替代解释，很关键**【原文 附录 B.4、Fig.12】。
- 关键机制/公式（直觉）：把数学推理看成 self-Bayesian——每步在已生成 token 上更新对中间假设的信念；epistemic token 不是冗余，而是"先别锁死、保留备选"的标记（Fig.2a）。I(y*;c|x) 量化 c 给正确答案的信息量：c 越富 teacher 越能"照抄"提示→轨迹简洁自信→epistemic 越少。覆盖度调制：|D| 大时模型需容纳更多推理模式，GRPO 靠增 epistemic 表达适应，SDPO 却逼简洁自信→宽覆盖受限【原文 §2-3、§6.2】。
- 实验与证据：
  - 数据集/设置：**作者自跑核心实验全在数学**——DAPO-Math-17k（训练，14,000 distinct 题，§6.1 Table 3 报为"100 步内 25,600 采样中 78% distinct"）；AIME24/25、AMC23、MATH-500（OOD 评测，题型不与训练重叠）。模型 Qwen3-1.7B/8B(think on/off)、DeepSeek-R1-Distill-Qwen-7B、Olmo3-7B-Instruct。
  - **跨域比较一半是引用而非重跑**：Fig.1a 的 Chemistry 曲线直接取自 SDPO(Hübotter 2026) 的 W&B 日志；§6 用 LiveCodeBench v6 一起做覆盖度对比（Table 3：Chemistry 2,400 题但仅 6 类题型/90-10 split；LiveCodeBench v6 仅 131 题且 train/eval 全重叠；vs DAPO-Math 14,000 题不重叠）。**此审计事实重要——跨域结论是间接论证**【原文 §6.1、Table 3、Fig.1 脚注】。
  - baseline 公平吗：GRPO vs SDPO 在同数据/同模型/同 step 比较，公平；off-policy SFT 对照（Table 2）控制极干净。问题在跨域（数学 vs 化学/代码）一半证据是引用 SDPO 日志，口径不完全一致。
  - 有无"看着强但没回答核心问题"：核心问题答得清楚（why degrade）；但"epistemic verbalization 因果驱动正确率"主要靠相关性 + 受控对比，**未做"强行注入/移除 epistemic token 看正确率"的反事实干预**——属强相关而非严格因果【推断，依据全文无 counterfactual injection 实验】。
- 假设与失效边界：
  - 显式假设【原文】：数学推理可视为 self-Bayesian 推理（§2，引 Kim 2026）；10 个固定词（wait/hmm/perhaps/maybe/actually/alternatively/seems/might/likely/check）可作 epistemic 的实用代理（§3，附录 B.5 用 GPT-5.4 as judge 补充验证多词短语）。
  - 隐式假设【推断】：epistemic token 频率与"真实不确定推理"正相关（10 词近似可能漏掉多 token 短语、误计数学条件句"if...then"——作者 prompt 里显式排除后者，但单 token 计数仍粗）。
  - 何时失效【推断】：窄覆盖/train-eval 高重叠任务上压制 epistemic 不仅无害还高效（Takeaway 4）——所以结论是"取决于覆盖度"，不是"自蒸馏一律有害"；"何种程度不确定是恰当的"无可操作判据。
- 祛魅总结【推断】：真贡献是把一个反直觉现象（自蒸馏在数学上变短变差）做了扎实的现象学拆解，Table 2 的 off-policy 对照 + Fig.12 的频移排除混淆是两处硬证据，结论"后训练目标应显式保留不确定表达、不能只看答案正确"有分量。被适度高估的是因果强度——通篇无反事实干预，"epistemic→正确率"是强相关；且跨域(化学/代码)对比一半是引用 SDPO 日志、非重跑。定位准确：这是诊断/机理论文，不提修复算法（作者自承）。

## 结构化抽取
- 🎯 机制速览6轴（注：本文是"诊断"，6 轴描述其分析的对象=自蒸馏训练，非提出新机制）：
  - **学什么信号**：被分析的自蒸馏目标 = student 向"富 context teacher"对齐的 **forward-direction KL(student‖stopgrad teacher)**（式1，注：是前向 KL(πθ‖teacher_with_c)，非 reverse-KL）；本文发现这等价于"学一种预设了推理时不可得信息的自信风格"【原文 §2 式1】。
  - **改什么**：student 参数；本文关注训练如何**改变 reasoning 行为/风格**（epistemic token 密度、response 长度）而非仅答案正确率【原文 §1、§5】。
  - **何时改**：off-policy SFT（§4）与 on-policy（§5）两种；on-policy 中 teacher 固定为初始策略最优（§5.4）。
  - **免梯度?**：否（分析对象是 SFT/GRPO/SDPO 这些基于梯度的后训练；本文自身是实证分析无训练新增）。
  - **记忆-技能生命周期**：N/A（无记忆/技能库；论点是"巩固不当会侵蚀不确定推理这一能力"——与本项目"巩固不应遗忘"直接呼应）【推断】。
  - **防遗忘机制**：本文恰是**反例诊断**——指出自蒸馏会"隐性遗忘 epistemic 表达能力"，标准目标不惩罚→OOD 退化；隐含建议是训练目标须显式保留不确定表达（未给具体机制）【原文 §1、§7】。
- ⑦ 开源代码+框架/harness：仓库 https://github.com/beanie00/self-distillation-analysis （v1 记 53M 已 clone）。**框架：veRL/HybridFlow**（References 引 Sheng 2024 HybridFlow；仓库含 `verl/` 目录，修改 `verl/trainer/ppo/ray_trainer.py` 的 `_remove_thinking_trace`）；built upon lasgroup/SDPO；transformers + vLLM/SGLang。关键超参（README）：train_batch_size=256、max_response_length=20480、teacher_update_rate=0（主）/0.05（消融）、训练在 4×B200。释出 Qwen3-8B(think on/off)/DeepSeek-Distill-7B checkpoint + W&B 日志【v1 analysis 已核仓库结构，本次据 PDF References 确认 HybridFlow=veRL】。
- 💰 资源/成本与可扩展性：原文正文未给卡时/wall-clock；README 标 4×B200、max_response_length=20480、batch=256。诊断实验规模适中（每设置 ~100-150 step、100 题/800 轨迹级）【原文未在正文说明完整成本，据仓库 README】。
- 🎯 对"探索-巩固"对标：**竞品/警示（强相关，反向支撑）**——一句判定：这是对"privileged-context 自蒸馏"做巩固时的**失效边界警告**，直接约束本项目"巩固/回轨"分支的设计。依据：本项目 idea 要 teacher 当稀疏脚手架教 student"走偏后自选恢复"——而本文证明：若 teacher（富 context）生成的恢复轨迹过于自信简洁、抹掉了 epistemic 表达（wait/hmm 这些正是"察觉走偏→回轨"的语言信号），student 模仿后在宽覆盖/OOD 上反而丧失自主纠错能力（AIME 掉 ~40%）。**可借组件**：(1) 用 I(y*;c|x) 度量脚手架信息量、避免脚手架"过强"压垮 student 自主性——这对 idea 里"稀疏脚手架"的"稀疏"二字给了量化抓手；(2) Fig.12 式的频移诊断可用于监控巩固阶段是否在均匀漂移之外特异性抹掉关键 token。**缺口/张力**：本文用的是 forward-KL(student‖teacher) 自蒸馏，而本项目/vla_opd 倾向 reverse-KL；reverse-KL 是否也压制 epistemic 本文未测——这是一个该项目需自查的开放点（reverse-KL mode-seeking 锁主模，可能同样丢长尾的不确定表达）。与 survey-grpo-step-segmentation 记忆里"切高熵/低置信关键步"高度互补：epistemic token 往往出现在高熵处，本文佐证"不该无脑压低这些步的熵"。
- 🔭 开放问题/未来方向：
  - 【原文】后训练目标须兼顾答案正确性 + 保留/诱导 uncertainty-aware 推理行为；呼吁开发显式保留不确定表达的更鲁棒训练策略（§7）。
  - 【原文/隐含】"何种程度的不确定是恰当的"——需可操作判据（当前只有定性"过度压制有害"）。
  - 【推断】补反事实因果实验（注入/移除 epistemic token 测正确率）；测 reverse-KL 自蒸馏是否同样压制 epistemic；把覆盖度 |D| 与 epistemic 保留写进一个自适应目标（按覆盖度动态调脚手架强度）——后者正是本项目可落地的设计方向。

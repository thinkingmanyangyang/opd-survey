spurious_rewards | Spurious Rewards: Rethinking Training Signals in RLVR | University of Washington / AI2 / UC Berkeley(Rulin Shao、Shuyue Stella Li、Rui Xin、Scott Geng 共同一作;…Nathan Lambert、Sewon Min、Pang Wei Koh、Luke Zettlemoyer) | 2026-02-25 arXiv v2·Preprint | 主题线 L3(RLVR/GRPO 机理)·相关性 高

**原始论文**:https://arxiv.org/abs/2506.10947

## 一眼看懂
- 🟦 TL;DR:在 Qwen2.5-Math 上,即使用**虚假奖励**(随机/格式/错误标签——与正确答案零相关甚至负相关)做 RLVR(GRPO)也能大幅涨分:随机奖励 MATH-500 +21.4%,接近真值的 +29.1%【原文 abstract+§2 Fig.2】。机理:GRPO 的 **clip 项产生与奖励无关的梯度偏置**,系统性放大 base model 中**已有高先验的 token/行为**(如 Qwen 的 code reasoning),而非注入新能力。该效应**强烈依赖模型族**——在 Llama3/OLMo2 上几乎消失甚至掉点。
- 最巧的一步:**clipping bias 推导 + no-clipping 消融**。抽掉"去 clip 实验"(Fig.4)就垮——正是"关掉 clip 后随机奖励的期望梯度为零、不再涨分"(§4)这一对照,把"随机奖励为何有效"从玄学钉死为 GRPO clip 机制的数学产物;没有它,"放大先验"只是猜想。

## 为什么做
- 研究背景:RLVR 已成提升 LLM 数学推理的主流后训练范式,但增益机理不清;开源社区高度依赖 Qwen2.5-Math 作事实标准 base model,大量"zero-RL"结论建立在这单一模型族上【原文§1】。
- 解决的具体痛点:多数 RLVR 方法只在 Qwen 上验证;社区默认"Qwen 上的增益=真实推理能力提升",忽视预训练先验对 RL 动力学的塑造,可能高估了 RL 算法本身贡献【原文§1+§3 practical warning】。
- 相关工作 & 各自不足:TTRL(多数投票伪标签做无监督 RL,Zuo et al.)、One-Shot RL(Wang et al.,单样本)、各类弱/噪声监督奖励。前人已用"少量真值/噪声标签"暗示过"RLVR 触发潜在能力而非教新能力",但本文把这一假说推到极端(连随机/错误奖励都能涨)【原文§2 末】。
- 动机链:RLVR 增益机理不清 + Qwen 中心化 → 设计从弱到虚假递进的奖励当诊断探针、跨模型族跑 → 发现 Qwen 上虚假奖励也涨、他族不涨 → 推断涨分来自"放大先验"而非"注入能力" → 用 clip 偏置推导 + code reasoning 行为分析 + 干预实验闭环验证【原文 §1-§5】。
- 与最近邻工作的Δ:vs TTRL/One-Shot RL——本文不提新训练方案,而是把"奖励是否需准确"的隐含假设推到极端(随机/错误奖励)做**诊断**;并明确把这些方法纳入"Qwen 上涨他族不涨"的同一模式(§3 用 Figure 15 复现 TTRL/One-Shot RL 也是 Qwen-only 有效)。为什么有用:提供了一个"虚假奖励=dummy baseline"的方法论检验标准。

## 怎么做 + 靠不靠谱
- 方法流水线:① 设计奖励梯度链(Ground Truth → Majority Vote 64-rollout 多数投票伪标签 → Format 含非空 \boxed{} 即给 1 → Random γ=0.5 概率随机给 1 → Incorrect Label 只奖励多数投票得到的错误答案)→ ② 只替换奖励函数、固定其余超参,在 DeepScaleR 上跑 GRPO 300 步 → ③ 跨模型族对比(Qwen2.5-Math/Qwen2.5/Llama3.1-3.2/OLMo2)→ ④ clip 偏置理论推导 + no-clipping 三种消融 → ⑤ code reasoning 行为分析 + prompt/RL-based 干预验证【原文§2.1+§4+§5】。
- 逐组件必要性:
  - **奖励梯度链(5 档)**:负责"检验需多少有效监督";Incorrect Label 专门设计来**解耦**"多数投票有效是因标签更可能正确,还是因它是高概率模型输出"(§2 第 5 档说明)——这是关键诊断设计。
  - **跨模型族对照**:Qwen 涨 / Llama3·OLMo2 不涨(OLMo2 GT +0.4,gains from GT only),证"先验依赖";无它则无法排除"虚假奖励普遍有效"【原文 Fig.1/Fig.3】。
  - **no-clipping 消融(Fig.4)**:三变体(关 clip / 增大 mini-batch 到 rollout size / 减小 rollout size 使每 rollout 单次梯度更新,后两者强制 πθ=πold→ρt=1 消除 clip)下随机奖励**不再稳定涨分**,仅开 clip 才涨——直接证 clip 是因果来源。
  - **code reasoning 干预**:prompt/RL-based 显式抬高 code reasoning 与词汇重复→Qwen 涨分,反向印证中介机理。
- 关键机制/公式(直觉):GRPO 目标含 min(ρ·Â, clip(ρ,1−ε,1+ε)·Â)(Eq.1)。**clip 偏置**(Eq.2):随机奖励下,析出奖励无关标量后,期望梯度 ∝ 对 Rθ=πθ/πold<1−ε 的 token 推高其概率、对 Rθ>1+ε 的压低、中间区为 0。直觉(论文 Example,ε=0.2):高先验 token(πold=0.85)上界 1.02 永不可达(πθ≤1)→只受非负梯度偏置→持续被推高;低先验 token(πold=0.02)上界 0.024 极易越过→被压。**净效果:无论奖励是否有信息,系统性放大已高先验的 token**。
- 实验与证据:
  - 数据集/设置:训练 DeepScaleR(及其多数投票伪标注/错误标注派生集);评测 MATH-500(pass@1)、AMC(avg@8)、AIME2024/2025;模型 Qwen2.5-Math-7B/1.5B、Qwen2.5-7B/1.5B、Llama3.1-8B(-Instruct)、Llama3.2-3B(-Instruct)、OLMo2-7B(-SFT)【原文§2.1+§3】。
  - 关键数字【原文 Fig.1/Fig.2,本轮 PDF 直读确认】:Qwen2.5-Math-7B MATH-500——GT +29.1、Majority +27.1、Incorrect +24.1、Random +21.4、Format +13.8/+15.5、某弱奖励 −6.4;AMC 上 Format/Incorrect/Random = +13.8/+24.1/+21.4,逼近 GT~27-29%。跨族:Llama3.1-8B-Instruct 多为负(−6.4/−8.3/−2.1/−11.5),OLMo2-7B 仅 GT +0.4。code reasoning:含代码答案准确率 **60.9%** vs 无代码 **28.0%**;Qwen2.5-Math-7B 基线 code 频率 **65.0%**,虚假奖励下升至 90%+(随机奖励最高 95.6%)【Python 直读 §5 表确认】。
  - baseline 公平吗:只换奖励函数、固定其余超参,跨族同设置,极公平(本就是诊断对照设计)。
  - 看着强但没回答核心:作者**明确自限**——结论限于"当前开源后训练算力规模"(§2 末),并**不推荐**虚假奖励作训练方案(§2 黑体 Note)。code reasoning 是相关性中介证据,未必穷尽全部机理。
- 假设与失效边界:
  - 显式【原文】:结论绑定 Qwen2.5-Math 这一"预训练即含大量含代码数学解"的特殊 base model;限于开源算力规模;γ=0 时 loss 恒定、梯度全零(§2 第 4 档)。
  - 隐式【推断】:clip 偏置推导基于若干简化(省略 KL 项,Eq.1 脚注 2);"放大先验"假说外推到通用模型/更大算力需谨慎(依据:Llama/OLMo 不涨 + 作者自限算力);已 RL 后训练的模型几乎无增益(Appendix J)。
- 祛魅总结【推断】:真贡献=用"虚假奖励诊断探针 + clip 偏置推导 + 跨族对照 + 干预验证"四证闭环,有力支撑"RLVR 在开源算力规模下主要激发/放大已有潜在能力、而非教新能力",并给出"虚假奖励=dummy baseline"的方法论警示——这是对整个 RLVR/zero-RL 文献的一记重要"祛魅"。包装/高估风险低(作者多处主动自限)。低估:clip 偏置作为"与奖励无关的优化伪影"对**所有** GRPO 工作(含本项目可能用到的 RLVR/OPD 混合)的潜在污染,论文点到但未量化其在真值奖励下的份额。

## 结构化抽取
- 🎯 机制速览6轴:
  - **学什么信号**:本质上"什么都不学新的"——clip 偏置让模型学的是"已高先验的 token/行为"(奖励信号被证可有可无);可观测中介=code reasoning 频率、词汇重复。
  - **改什么**:策略模型 πθ 全参数(标准 GRPO 更新),但有效改动方向被 clip 偏置导向高先验区。
  - **何时改**:RLVR 训练全程(300 步);虚假奖励下前 50 步即显著涨。
  - **免梯度?**:否,标准 GRPO 梯度训练。
  - **记忆-技能生命周期**:无记忆/技能库;"能力"被论证为**预训练已存的潜在行为被放大固化**,RLVR 不新增。
  - **防遗忘机制**:不适用/反例——论文恰好揭示 RLVR 在此规模下不引入新知识,故无"新技能需防遗忘"之说;反倒提示 RLVR 可能只是把先验行为推到饱和。
- ⑦ 开源代码+框架/harness:https://github.com/ruixin31/Rethink_RLVR (v1 验证本地已克隆约 259MB,Tier A,代码完整可跑)。模型集合 HF `stellalisy/spurious-rewards`。框架=**OpenRLHF**(README 自述改编自 PRIME-RL 的 TTRL,TTRL 本身为 OpenRLHF 衍生;`code/ttrl/` 多处引 OpenRLHF issue,脚本用 OpenRLHF 风格参数 `micro_train_batch_size`/`n_samples_per_prompt`/`--advantage_estimator group_norm`/`--normalize_reward`)。算法主体 **GRPO**(advantage 用 group_norm)。〔承 v1 更正:原稿曾误记 veRL,经 README+`ttrl/` 代码+论文正文确认为 OpenRLHF。〕奖励函数实现 `ttrl/verifier/qwen/qwen_eval.py`、`ttrl/verifier/auto_verify.py`(`random_reward_fn(rate=0.5)`/`inverse_qwen_reward_fn`/`box_only_format_reward_fn`)。
- 💰 资源/成本与可扩展性:训练 300 步、binary 0-1 奖励、train_batch_size=128、n_samples_per_prompt=16、micro_train_batch_size=4、gamma=1.0;仅改 TASK/REWARD;无 chat template 模型加 `_r1_style` 后缀。环境 python 3.10 + flash_attn 2.7.0.post2。具体 GPU 数原文未明确列在正文,Appendix 含多随机种子复现。
- 🎯 对"探索-巩固"对标:**竞品/警示(对核心追问直接相关)**。一句判定:本文是本项目"教师脚手架蒸馏是否真教会新能力"这一核心追问的**最尖锐反方证据**——它论证在开源算力规模下 RLVR(乃至 GRPO 类训练)主要是"放大 base 已有高先验行为"而非注入新能力,因此本项目若用 RLVR/on-policy 信号做"巩固",必须警惕**clip 偏置污染**(涨分可能来自先验放大而非真学到"路径恢复")。可借:① 把"虚假奖励 dummy baseline"纳入本项目评估,区分"teacher 脚手架真贡献" vs "GRPO clip 伪影";② 跨模型族验证(勿只用 Qwen)。缺口:本文是纯诊断、无任何蒸馏/记忆/技能机制,不提供"巩固"方案。依据:§4 clip 偏置 + §5 先验放大 + §3 跨族失效。
- 🔭 开放问题/未来方向:【原文】未来 RLVR 研究应在非 Qwen 模型上确认、并用虚假奖励作 dummy baseline(§3);OLMo 开放训练数据可助未来研究推理行为起源(§3)。【推断】量化"clip 偏置放大先验"在**真值奖励**增益中所占份额(分离"真学习" vs "先验放大");设计抑制 clip 偏置或对先验不敏感的 RL 目标;把"先验放大"机理迁移到非数学/agent 任务,检验本项目 path-recovery 信号是否也被先验放大所污染。

RETURN: spurious_rewards | 读到PDF?是(41页/135430字,abstract+§2 Fig.1/2+§4 clip偏置Eq.1/2+§5 code reasoning表 全直读核实) | L3(RLVR/GRPO机理) | 对标=核心追问的尖锐反方/竞品,警示RLVR增益可能是clip偏置放大先验而非学新能力;可借"虚假奖励dummy baseline+跨族验证";无巩固方案 | 残留待核0

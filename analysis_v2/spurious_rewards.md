spurious_rewards | Spurious Rewards: Rethinking Training Signals in RLVR | University of Washington / AI2 / UC Berkeley(Rulin Shao、Shuyue Stella Li、Rui Xin、Scott Geng 共同一作;…Nathan Lambert、Sewon Min、Pang Wei Koh、Luke Zettlemoyer) | 2026-02-25 arXiv v2·Preprint | 主题线 L3(RLVR/GRPO 机理)·相关性 高

**原始论文**:https://arxiv.org/abs/2506.10947

## 一眼看懂

> 一句话导读:在 Qwen2.5-Math 上,哪怕用"完全虚假的奖励"(随机给分、看格式给分、甚至按错误答案给分)跑 RLVR 也能大涨分。原因不是学到了新能力,而是 GRPO 的 clip 机制带来一股"与奖励无关的梯度偏置",系统性地放大模型本来就擅长的行为(比如 Qwen 爱用代码解题)。换个模型族(Llama/OLMo)就不灵了。

- 🟦 TL;DR:在 Qwen2.5-Math 上,即使用**虚假奖励**(随机/格式/错误标签——与正确答案零相关、甚至负相关)做 RLVR(GRPO),也能大幅涨分:随机奖励 MATH-500 +21.4%,接近真值的 +29.1%【原文 abstract+§2 Fig.2】。
  - 机理:GRPO 的 **clip 项会产生一股与奖励无关的梯度偏置**,系统性地放大 base model 里**本来就有高先验的 token/行为**(比如 Qwen 的 code reasoning),而不是注入任何新能力;
  - 这个效应**强烈依赖模型族**——换到 Llama3/OLMo2 上几乎消失、甚至掉点。
- 最巧的一步:**clipping bias 推导 + no-clipping 消融**。抽掉"去掉 clip 的实验"(Fig.4)就垮:正是"关掉 clip 后,随机奖励的期望梯度变为零、于是不再涨分"(§4)这个对照,把"随机奖励为何有效"从玄学钉死成 GRPO clip 机制的数学产物;没有它,"放大先验"就只是个猜想。

## 为什么做

> 一句话导读:大家用 RLVR 涨数学分,却没搞清增益从哪来;而且整个开源社区几乎只在 Qwen2.5-Math 这一个 base 模型上验证——本文怀疑很多"RL 提升"其实是 Qwen 自带的预训练先验在起作用。

- 研究背景:RLVR 已是提升 LLM 数学推理的主流后训练范式(Lambert et al. 2024;DeepSeek-R1;SimpleRL-Zero;DeepScaleR),但增益机理不清;开源社区高度依赖 Qwen2.5-Math 作为事实标准 base model,大量"zero-RL"结论都建立在这单一模型族上【原文§1】。
- 解决的具体痛点:多数 RLVR 方法只在 Qwen 上验证;社区默认"Qwen 上的增益 = 真实推理能力提升",忽视了预训练先验对 RL 动力学的塑造,因此可能高估了 RL 算法本身的贡献【原文§1+§3 practical warning】。
- 相关工作 & 各自不足(本轮按引文链补全):
  - **"RLVR 只激发潜能、不教新能力"假说族**——本文是这一假说的**极端化压力测试**。前人证据按"激进程度"排序:
    - Wang et al. 2024:用**少量**真值标签即足够(暗示只是触发潜能);
    - Zuo et al. 2025 **TTRL**:用多数投票伪标签做无监督 RL,即**噪声标签**也行;
    - Yue et al. 2025 / Liu et al. 2025 / Gandhi et al. 2025:从 pass@k、推理行为等角度论证"RL 收窄而非拓宽能力边界"；
    - 本文把梯度推到**连随机/错误奖励都涨**这一端点,是该假说最尖锐的实证【原文§2 末】。
  - **Qwen-only 方法的可复现性隐患**:本文把 **TTRL**(Zuo et al.)与 **One-Shot RL**(Wang et al. 2025b,单样本)归入"Qwen 涨、他族不涨"同一模式,§3+Fig.15(Appendix E)直接复现二者"在 Qwen 上匹配真值、在他族失效"——指出它们的结论可能是 Qwen 先验伪影、而非方法本身有效【原文§3】。
  - **clip 偏置的并行发现**:Yu et al. 2025(DAPO/clip-higher 系)在**真值奖励**下指出 clip 偏置会"减少探索、增加 exploitation";本文把同一 clip 偏置分析**迁移到随机奖励**场景并给出完整推导,二者机理一致(§4 末"echo the intuition of Yu et al.")。
  - 精确差异:相比上述所有工作——本文**不提任何新训练方案**,而是把"奖励是否需要准确"这一隐含假设推到极端做**诊断**,并给出"虚假奖励 = dummy baseline"的方法论检验标准。
- 动机链(逐步推):
  - RLVR 增益机理不清 + 社区过度 Qwen 中心化;
  - 于是设计一组"从弱到虚假递进"的奖励当诊断探针,跨模型族跑;
  - 发现 Qwen 上虚假奖励也涨、他族不涨;
  - 推断涨分来自"放大先验"而非"注入能力";
  - 用 clip 偏置推导 + code reasoning 行为分析 + 干预实验闭环验证【原文 §1-§5】。

## 怎么做 + 靠不靠谱

> 一句话导读:固定其余一切,只换"答案→0/1 奖励"那一个函数,设 5 档奖励(从真值到纯随机到错误标签),跨 10 个模型跑 GRPO 300 步;再用 clip 偏置的数学推导 + 关掉 clip 的消融把因果钉死;最后用 code reasoning 行为追踪与干预闭环验证。

- 方法流水线(这是一个诊断协议,输入→输出):
  - ① **奖励梯度链(5 档)**:每档只替换"把 rollout 答案映射到 0/1 奖励"的那一个函数,其余超参全固定;
  - ② 在 DeepScaleR 训练集上跑 **GRPO 300 步**(二值 0/1 奖励、train_batch=128、n_samples_per_prompt=16、micro_train_batch=4、γ=1.0、KL 关闭);
  - ③ 同一协议跨 10 个模型(Qwen2.5-Math-7B/1.5B、Qwen2.5-7B/1.5B、Llama3.1-8B(-Instruct)、Llama3.2-3B(-Instruct)、OLMo2-7B(-SFT));
  - ④ clip 偏置理论推导(主文 Eq.1/2 + Appendix B 全推导)+ no-clipping 三种消融(Fig.4);
  - ⑤ code reasoning 行为追踪 + prompt/RL-based 干预验证(§5);
  - 输出:对每档奖励/每个模型的 MATH-500/AMC/AIME 增益曲线 + code 频率曲线【原文§2.1+§4+§5】。
- 5 档奖励的精确定义(可复现关键,§2.2):
  1. **Ground Truth**:rollout 答案经 verifier 与真值匹配则 r=1,否则 0(监督质量上界)。
  2. **Majority Vote**:训练前用 base model 对每题采 **64 条** rollout,取多数答案当伪标签;rollout 匹配该伪标签则 r=1。
  3. **Format**:含**非空** `\boxed{...}` 即 r=1(不看正误;`\boxed{}` 出现在 Qwen2.5-Math 的系统提示里,故等价于"奖励遵循提示")。
  4. **Random**:与响应无关,以固定概率 γ 给 1(主实验 γ=0.5;Appendix B 验 γ∈{0.001,0.3,0.7} 增益相似、仅收敛速度不同;**γ=0 时 loss 恒定、梯度全零**,符合解析预期)。
  5. **(Majority-voted) Incorrect**:先用多数投票标注全集,**只取标签错误的子集**训练,且只奖励"答案匹配该错误伪标签"的 rollout。这是**专门设计来解耦**一个问题:多数投票有效,究竟是因为它标签更可能正确,还是因为它本就是高概率的模型输出?——若 incorrect 也涨,就说明是后者。为保持与其它档相近的 reward 稀疏度,只奖励**特定**错误答案而非任意错误答案。
- 逐组件必要性:
  - **奖励梯度链(5 档)**:检验"到底需要多少有效监督";Incorrect 档是关键诊断设计(解耦"对 vs 高概率")。
  - **跨模型族对照**:Qwen 涨 / Llama3·OLMo2 不涨(OLMo2 仅 GT +0.4),证"先验依赖";无它则无法排除"虚假奖励普遍有效"【原文 Fig.1/Fig.3】。同族内部行为一致(共享预训练分布);**越大的模型越受益于随机奖励**(推测大模型保留了更多可被放大的先验)。
  - **no-clipping 消融(Fig.4,因果钉死)**:三个变体——
    - (a) 在实现里直接关掉 clip;
    - (b) 把 mini-batch 增到等于 rollout size;
    - (c) 把 rollout size 减到"每个 rollout 只做一次梯度更新";
    - 后两者强制 \(\pi_\theta=\pi_{\text{old}}\Rightarrow\rho_t=1\),从构造上消除 clip 效应;
    - 三者下随机奖励**不再稳定涨分**,唯独开 clip(第四面板)稳定涨——直接证明 clip 是因果来源。
  - **code reasoning 干预**:用 prompt("Let's solve this using Python.")和 RL("含 python 字符串即 +1")两路显式抬高 code reasoning → Qwen 涨分,反向印证中介机理。
- 关键机制/公式(本轮据 PDF 正文 Eq.1/2 + Appendix B 全推导补全,真符号 MathJax):
  - **GRPO 目标(Eq.1,省略 KL,因实验关闭 KL)**:
    \(\displaystyle J(\theta)=\mathbb{E}_{x\sim D,\,y\sim\pi_{\text{old}}(\cdot|x)}\!\left[\sum_{t=1}^{|y|}\min\!\Big(\rho_t(y;\theta)\,\hat A(x,y),\ \mathrm{clip}\big(\rho_t(y;\theta),\,1-\epsilon_c,\,1+\epsilon_c\big)\hat A(x,y)\Big)\right]\)
    其中 token 级重要性比 \(\rho_t(y;\theta)=\dfrac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_{\text{old}}(y_t\mid x,y_{<t})}\),优势 \(\hat A(x,y)=\dfrac{r(x,y)-\bar r_x}{\sigma_x}\)(组内均值 \(\bar r_x\)、组内标准差 \(\sigma_x\))。
  - **随机奖励下期望优势为零(Appendix B.1.1)**:对随机 Bernoulli(γ) 奖励,组内归一化后的优势之和恒为 0(构造性),且奖励独立于样本,故 \(\mathbb{E}[\hat A]=0\)。
  - **clipping bias 定义(Appendix B.1.2)**:把 clip 引入的额外梯度定义为
    \(\displaystyle \mathrm{Bias}(\nabla_\theta J(\theta))=\mathbb{E}_{x,y}[\nabla_\theta J(\theta)]-\mathbb{E}_{x,y}[\nabla_\theta L_{\text{unclipped}}(\theta)]\)
    由于 \(\mathbb{E}[\hat A]=0\) 且 \(\hat A\) 独立于其余项,**无 clip 项时期望梯度恒为 0**:\(\mathbb{E}_{x,y}[\nabla_\theta L_{\text{unclipped}}]=\mathbb{E}[\hat A]\cdot\mathbb{E}_{x,y}[\tfrac{1}{|y|}\sum_t\nabla_\theta\rho_t]=0\)。所以 \(\mathrm{Bias}=\mathbb{E}[\nabla_\theta J]\) 本身——即"整股梯度都是 clip 偏置贡献的"。
  - **按 \(\hat A\) 符号分情形求梯度(Appendix B.1.2 核心,记 \(R_\theta=\rho_t\))**:
    \(\displaystyle \nabla_\theta J(\theta)=\hat A\cdot \begin{cases} \nabla_\theta R_\theta, & \hat A\ge 0\ \text{且}\ R_\theta<1+\epsilon_c,\\ 0, & \hat A\ge 0\ \text{且}\ R_\theta>1+\epsilon_c,\\ 0, & \hat A<0\ \text{且}\ R_\theta<1-\epsilon_c,\\ \nabla_\theta R_\theta, & \hat A<0\ \text{且}\ R_\theta>1-\epsilon_c. \end{cases}\)
    令 \(\mu=P(\hat A\ge0)\,\mathbb{E}[\hat A\mid\hat A\ge0]=-P(\hat A<0)\,\mathbb{E}[\hat A\mid\hat A<0]>0\)(由 \(\mathbb{E}[\hat A]=0\) 推出此等式且 \(\mu\) 恒正)。代回、并把不可导点的梯度置 0,得**最终 clip 偏置(Eq.2)**:
    \(\displaystyle \mathrm{Bias}(\nabla_\theta J)=\mu\cdot\mathbb{E}_{x,y} \begin{cases} \nabla_\theta R_\theta, & \pi_{\theta,x}(y_t)<\pi_{\text{old},x}(y_t)(1-\epsilon_c),\\ 0, & |R_\theta-1|\le\epsilon_c,\\ -\nabla_\theta R_\theta, & \pi_{\theta,x}(y_t)>\pi_{\text{old},x}(y_t)(1+\epsilon_c). \end{cases}\)
  - **直觉(论文 Example,\(\epsilon_c=0.2\))**:相对 clip 阈值落在 \([0,1]\) 概率区间上,会对高/低概率 token 造成**非对称的绝对 clip 范围**:
    - 高先验 token(\(\pi_{\text{old}}=0.85\)):上界 \(0.85\times1.02\approx1.02>1\),因 \(\pi_\theta\le1\) 而**永远碰不到上界**→ 只受非负梯度偏置 → 被持续推高;
    - 低先验 token(\(\pi_{\text{old}}=0.02\)):上界 \(0.02\times1.2=0.024\),稍微一增就越界 → 受负偏置被压;
    - 净效果:不管奖励有没有信息,都系统性地放大本就高先验的 token、压低低先验的 token——这就是随机奖励仍能涨分的数学根源。
- 实验与证据:
  - 数据集/设置:训练 DeepScaleR(及其多数投票伪标注/错误标注派生集);评测 MATH-500(pass@1)、AMC(avg@8)、AIME2024/2025;模型见上述 10 个【原文§2.1+§3】。评测沿用 **OpenRLHF 默认评测设置**(原文§2.1 明引 "default evaluation setup in the popular RL framework OpenRLHF")。
  - 关键数字【原文 Fig.1/Fig.2/Table 1,本轮 PDF 直读确认】:
    - Qwen2.5-Math-7B MATH-500:GT +29.1、Majority +27.1、Incorrect +24.1、Random +21.4、Format +13.8/+15.5、某弱奖励 −6.4;AMC 上 Format/Incorrect/Random = +13.8/+24.1/+21.4;
    - AIME2024:Format +10.3 逼近 GT +15.3;AIME2025(题目在所有模型知识截止之后):GT 明显占优,其余 −0.4~+4.5;
    - 跨族:Llama3.1-8B-Instruct 多为负(−6.4/−8.3/−2.1/−11.5),OLMo2-7B 仅 GT +0.4;
    - **code reasoning(Table 1)**:Qwen2.5-Math-7B 含代码答案准确率 **60.9%** vs 无代码 **35.0%**(注:本轮 PDF Table 1 显示 w/Lang=35.0,v1 曾记 28.0,以 PDF Table 1 为准);Qwen2.5-Math-1.5B 52.6 vs 17.2;**反例**:Qwen2.5-7B 含代码反而更低(39.9 vs 61.5)、OLMo2-7B-SFT 21.0 vs 40.0(故称 Bad-Code 模型);
    - Code 频率:Qwen2.5-Math-7B 基线 **65.0%**,虚假奖励下 15 步内升至 ~90%、随机奖励最高 **95.6%**;Python-reward 干预 20 步内 >99%;
    - **prompt 干预(Table 2)**:强制以 "LET'S SOLVE THIS USING PYTHON." 开头,Qwen2.5-Math-1.5B/7B/Qwen2.5-1.5B 分别 +24.2/+15.0/+10.0,而 Qwen2.5-7B −19.4、Llama3.2-3B-Inst −28.6、Llama3.1-8B-Inst −21.6、OLMo2 −1.2/−2.8。
  - baseline 公平吗:只换奖励函数、固定其余超参,跨族同设置,极公平(本就是诊断对照设计)。
  - 看着强但没回答核心:作者**明确自限**——结论限于"当前开源后训练算力规模"(§2 末),并**不推荐**虚假奖励作为训练方案(§2 黑体 Note 与 §1 末重复声明)。code reasoning 只是相关性中介证据(论文明说"not a complete explanation",另有 lexical repetition 等行为,Appendix G),未必穷尽全部机理。
- 假设与失效边界:
  - 显式【原文】:结论绑定 Qwen2.5-Math 这一"预训练即含大量含代码数学解"的特殊 base model;限于开源算力规模;γ=0 时梯度全零;**已 RL 后训练的模型几乎无增益(Appendix J)**。
  - 隐式【推断】:
    - clip 偏置推导基于若干简化(省略 KL 项,Eq.1 脚注 2;不可导点梯度取 0,PyTorch 默认);
    - "放大先验"假说外推到通用模型/更大算力需谨慎(依据:Llama/OLMo 不涨 + 作者自限算力)。
- 祛魅总结【推断】:
  - 真贡献 = 用"虚假奖励诊断探针 + clip 偏置完整推导 + 跨族对照 + 干预验证"四证闭环,有力支撑"RLVR 在开源算力规模下主要是激发/放大已有潜在能力、而非教新能力",并给出"虚假奖励 = dummy baseline"的方法论警示——这是对整个 RLVR/zero-RL 文献的一记重要"祛魅"。
  - 包装/高估风险低(作者多处主动自限)。
  - 低估:clip 偏置作为"与奖励无关的优化伪影",对**所有** GRPO 工作(含本项目可能用到的 RLVR/OPD 混合)都有潜在污染,论文点到但未量化它在真值奖励下所占的份额。

## 结构化抽取

> 一句话导读:六轴速览 + 开源(基于 OpenRLHF/TTRL)+ 资源成本;对本项目而言,这是"教师脚手架蒸馏是否真教会新能力"这一核心追问的**最尖锐反方证据**——警示 clip 偏置可能让涨分来自先验放大而非真学到东西。

- 🎯 机制速览6轴:
  - **学什么信号**:本质上"什么新东西都没学"——clip 偏置让模型学的是"已高先验的 token/行为"(奖励信号被证明可有可无);可观测中介 = code reasoning 频率、词汇重复。
  - **改什么**:策略模型 \(\pi_\theta\) 的全参数(标准 GRPO 更新),但有效改动方向被 clip 偏置导向高先验区。
  - **何时改**:RLVR 训练全程(300 步);虚假奖励下前 50 步即显著涨。
  - **免梯度?**:否,是标准 GRPO 梯度训练。
  - **记忆-技能生命周期**:无记忆库/技能库;"能力"被论证为**预训练已存的潜在行为被放大固化**,RLVR 并不新增。
  - **防遗忘机制**:不适用/反例——论文恰好揭示在此规模下 RLVR 不引入新知识,故无"新技能需防遗忘"之说;反倒提示 RLVR 可能只是把先验行为推到饱和。
- ⑦ 开源代码+框架/harness:https://github.com/ruixin31/Rethink_RLVR (v1 验证本地已克隆约 259MB,Tier A,代码完整可跑)。模型集合 HF `stellalisy/spurious-rewards`。
  - 框架 = **OpenRLHF**〔本轮据**实际仓库文件**再确认〕:`code/README.md:25` 自述 "Our codebase is based on TTRL (https://github.com/PRIME-RL/TTRL)";TTRL/PRIME-RL 本身构建在 **OpenRLHF + DeepSpeed** 之上(仓内 `code/ttrl/helper/deepspeed.py`、`ttrl/models/actor.py` 引用 OpenRLHF;训练脚本如 `scripts/rlvr_deepscaler_grpo_qwen_random.sh` 用 OpenRLHF 风格 CLI:`--micro_train_batch_size 4`、`--n_samples_per_prompt 16`、`--advantage_estimator "group_norm"`、`--normalize_reward`)。
  - 算法主体 = **GRPO**(advantage=group_norm)。
  - 〔承 v1 更正并已二次坐实〕原稿曾误记为 veRL,经 README + `ttrl/` 代码 + 脚本确认为 **OpenRLHF 谱系**,非 veRL。
  - 奖励函数实现:`ttrl/verifier/qwen/qwen_eval.py`、`ttrl/verifier/auto_verify.py`(含 `random_reward_fn(rate=0.5)`/`inverse_qwen_reward_fn`/`box_only_format_reward_fn`)。
- 💰 资源/成本与可扩展性:
  - 训练 300 步、二值 0-1 奖励、train_batch_size=128、n_samples_per_prompt=16、micro_train_batch_size=4、gamma=1.0;仅改 TASK/REWARD;无 chat template 的模型加 `_r1_style` 后缀;
  - 环境 python 3.10 + flash_attn 2.7.0.post2;
  - 具体 GPU 数原文未在正文集中列(Appendix 含多随机种子复现)。
- 🎯 对"探索-巩固"对标:**竞品/警示(对核心追问直接相关)**。
  - 一句判定:本文是本项目"教师脚手架蒸馏是否真教会新能力"这一核心追问的**最尖锐反方证据**——它论证在开源算力规模下,RLVR(乃至 GRPO 类训练)主要是"放大 base 已有高先验行为"而非注入新能力;因此本项目若用 RLVR/on-policy 信号做"巩固",必须警惕 **clip 偏置污染**(涨分可能来自先验放大而非真学到了"路径恢复")。
  - 可借:
    - ① 把"虚假奖励 dummy baseline"纳入本项目评估,区分"teacher 脚手架真贡献" vs "GRPO clip 伪影";
    - ② 跨模型族验证(别只用 Qwen);
    - ③ clip 偏置"对高先验 token 非对称放大"这一点提示:若本项目 path-recovery 信号集中在低概率分支,可能被 clip 偏置**系统性压制**,需特别检查。
  - 缺口:本文是纯诊断、无任何蒸馏/记忆/技能机制,不提供"巩固"方案。
  - 依据:§4 clip 偏置 + §5 先验放大 + §3 跨族失效。
- 🔭 开放问题/未来方向:
  - 【原文】未来 RLVR 研究应在非 Qwen 模型上确认,并用虚假奖励作 dummy baseline(§3);OLMo 开放的训练数据可助未来研究推理行为的起源(§3)。
  - 【推断】
    - 量化"clip 偏置放大先验"在**真值奖励**增益中所占的份额(把"真学习" vs "先验放大"分离开);
    - 设计能抑制 clip 偏置、或对先验不敏感的 RL 目标;
    - 把"先验放大"机理迁移到非数学/agent 任务,检验本项目 path-recovery 信号是否也被先验放大所污染。

RETURN: spurious_rewards | 读PDF?是(41页/135430字,abstract+§2 Fig.1/2/Table1/2+§4 clip偏置Eq.1/2+Appendix B 全推导+§5 code reasoning 全直读核实) | 加厚?是(补全 Appendix B 完整 clipping-bias 推导含按 Â 符号分情形与 μ 定义、5 档奖励精确定义、Table1/2 反例数据、related-work 引文链 TTRL/One-Shot/Yu et al. 并行发现;坐实 OpenRLHF 框架 via README+脚本 CLI) | LaTeX 公式条数 6(GRPO目标/优势/bias定义/unclipped期望零/符号分情形梯度/最终bias Eq.2) | 待核 0

srft | SRFT: A Single-Stage Method with Supervised and Reinforcement Fine-Tuning for Reasoning | 中科院自动化所/国科大 + 美团 + 上海交大(通讯 Dongbin Zhao 团队;Yuqian Fu、Tinghong Chen 共同一作) | 2025-06-24 arXiv v1·ICLR 2026·OpenReview n6E0r6kQWQ | 主题线 L2(统一SFT-RL/GFT类)·相关性 高

**原始论文**:https://arxiv.org/abs/2506.19767

## 一眼看懂
- 🟦 TL;DR:把 SFT 和 RL 塞进**单阶段同一个 loss** 一起训(不再先 SFT 再 RL 两段),用"当前策略熵"做开关自适应调两者权重——熵高(不确定)时压低 demonstration 模仿、保探索,正样本 RL 项熵高时加权维持探索。基于 LUFFY 的混合策略 off-policy RL 改造,Qwen2.5-Math-7B 上平均 59.1%,比 zero-RL 基线在 5 个数学基准 +9.0%、3 个 OOD 基准 +10.9%【原文 abstract+§5.2+结论】。
- 最巧的一步:**熵感知权重 + 单阶段融合**。抽掉熵权重退化为 LUFFY 式 SFT+RL 朴素相加(§3.2.2 的 L_SFT+RL=L_SFT+L_RL),论文主张正是"靠熵动态在模仿/探索间按需定权"。但需注意:论文给 SFT 权重 `exp(−H)`(熵高减弱模仿),开源代码实际用 `exp(+H)`(熵高加强模仿),方向相反【待核,见结构化抽取】——所以"最巧"这步的方向性在论文与代码间未对齐,是该工作最大的自洽裂缝。

## 为什么做
- 研究背景:LLM 推理后训练里 SFT 与 RL 如何最优结合是核心未解问题;传统两阶段串行(SFT 做指令跟随→RL 做对齐/推理)把二者当独立阶段【原文§1】。
- 解决的具体痛点:① 两阶段串行——SFT 易记忆模式而非真推理、过拟合数据集;RL 样本效率低、探索难、易 mode collapse【原文§1】。② 集成不足→误差传播、限制 RL 提升;过度依赖 demonstration→过拟合、约束探索。如何在"SFT 知识蒸馏"与"RL 策略优化"之间定权重是痛点【原文§1 末】。
- 相关工作 & 各自不足:LUFFY(off-policy RL/混合策略,直接被 SRFT 当底座)、ReLIFT(交错 RL 与对最难题在线 fine-tune)、TAPO(GRPO 内融结构化外部知识)、SimpleRL-Zero/OpenReasoner-Zero/PRIME-Zero(各类 zero-RL)。论文把这些归为"或两阶段、或动态切换",未在熵视角下系统定权【原文§1+§5.1 baseline 列表】。
- 动机链:两阶段各有缺陷(SFT 过拟合/RL 探索难)→ 单阶段融合更优但难定权(集成不足 vs 过度依赖 demo)→ 从熵视角做机理分析发现"SFT 全局粗调、RL 选择性细调、熵是有效性指标"→ 所以用熵感知权重在单阶段内自适应平衡【原文§1 Key Findings+§3】。
- 与最近邻工作(LUFFY)的Δ:LUFFY 把 demonstration 直接拼进 on-policy rollout group 做混合策略 GRPO(SRFT 的 Eq.6/7 优势估计沿用此),Δ 在于 SRFT **额外加了两个熵感知自适应权重**(demo 的 SFT 项 `0.5·exp(−H)`、self-rollout 正样本项 `0.1·exp(+H)`)+ 显式把 self-rollout RL 按正/负样本拆分(Eq.11)。为什么有用:论文称熵动态揭示训练机制,据此定权能在"熵高保探索 / 熵低强模仿"间动态平衡,避免 demo 与当前策略分布失配导致的退化【原文§4.1-4.2】。

## 怎么做 + 靠不靠谱
- 方法流水线:输入 prompt → ① 当前策略 πθ 生成 8 条 on-policy rollout(max 8192 token)→ ② 把 demonstration(DeepSeek-R1 生成的 OpenR1-Math 高质量解)拼进 rollout group 组成异构 batch Gaug.,组内统一算 group-norm advantage(Eq.7)→ ③ 四路 loss 合成单阶段反向:demo-SFT(熵权 `w_SFT=0.5·sg(exp(−H))`,Eq.8)+ demo-RL(LUFFY 式 off-policy,设 πβ=1、去 clip,Eq.9)+ self-rollout 正样本(熵权 `w_RL=0.1·sg(exp(+H))`,Eq.12)+ self-rollout 负样本(likelihood 最小化,Eq.12)→ 输出更新后的 πθ【原文§4.1-4.3 Eq.5-13】。
- 逐组件必要性:
  - **demo-SFT 熵权 `0.5·exp(−H)`**:负责"粗粒度行为策略逼近";论文称没它则 demo 与当前策略分布失配会致性能退化(Eq.8 上下文)。无单独消融该系数/方向的表(正文未见独立消融表)→**标出:熵权方向与系数缺独立消融**。
  - **demo-RL(off-policy,πβ=1 去 clip)**:负责"细粒度行为策略学习";沿用 LUFFY,设 πβ=1 避免 tokenization 复杂度、去 clip 因 πβ=1 时 clip 失衡【原文§4.1】。
  - **self-rollout 正/负拆分 + 正样本熵权 `0.1·exp(+H)`**:论文观察 binary 奖励{1,−1}下 RL 目标可自然拆成正样本(似 SFT 的 max-likelihood)+ 负样本(likelihood 最小化);self-exploration 致熵快速下降损探索,故对正样本项用 `exp(+H)` 维持探索多样性【原文§4.2 Eq.11-12】。
  - **单阶段 vs 两阶段**:有对照(Table 1:SFT→RL=52.5 > RL=49.4 > SFT=47.3 > RL→SFT=37.4;§3.2.2 Fig.5 单阶段 SFT+RL 训练效率优于 SFT→RL)——该对照支撑"单阶段更优"。
- 关键机制/公式(直觉):核心是两个"stop-grad 的指数熵权"当软开关。直觉:熵 H 是策略不确定性;`exp(−H)` 随熵升高而**减小**(论文用于 SFT:不确定时少模仿)、`exp(+H)` 随熵升高而**增大**(用于 RL 正样本:不确定时多保探索)。stop_grad 保证权重只调幅度不回传梯度。总 loss = demo-SFT + demo-RL + self-rollout-RL 三块(Eq.13,self-rollout 内含正负两项)。
- 实验与证据:
  - 数据集/设置:训练用 OpenR1-Math-46k-8192(OpenR1-Math-220k 的 46k 子集,源自 NuminaMath 1.5,DeepSeek-R1 生成解,经 Math-Verify 过滤去掉 >8192 token 与不可验证项);基座 Qwen2.5-Math-7B;8 rollout/prompt、500 训练步【原文§5.1】。
  - 关键实验+数字:5 个数学基准(AIME24/AMC 用 avg@32,Minerva/Olympiad/MATH500 用 pass@1)平均 **59.1**,较最佳 RL 基线 +9.0、较 SFT 方法 +4.8、较 SFT+RL 方法 +3.4;3 个 OOD 基准(ARC-C/GPQA-D/MMLU-Pro,选项乱序防泄漏)平均 **62.5**,较最佳基线 +4.7【原文§5.2 Table 2】。abstract/结论的"+9.0%/+10.9%"口径是**对 zero-RL 基线**(非对 Table 中最佳 SFT+RL 基线)【原文 abstract+结论 L1074-1075】。
  - baseline 公平吗:含 LUFFY/ReLIFT/TAPO 等同期强基线,且 LUFFY/SRFT 用同一 46k 数据集,基座统一 Qwen2.5-Math-7B,较公平;但部分 zero-RL 基线(SimpleRL/ORZ/PRIME)用不同/更大训练集(24k~150k),"对 zero-RL +9.0/+10.9"的口径混入了数据规模差异。
  - 看着强但没回答核心:"熵权各项贡献"缺独立消融表(正文未见拆解 `0.5·exp(−H)` / `0.1·exp(+H)` 各自增益),故"增益来自熵感知"vs"来自单阶段融合(LUFFY 已有)"的归因不够干净。
- 假设与失效边界:
  - 显式【原文】:依赖高质量 demonstration(Limitations 明说"假设可获高质量 demo,imperfect demo 待研究");设 πβ=1 简化(off-the-shelf 数据无需重算 behavior policy 概率)。
  - 隐式【推断】:仅 Qwen2.5-Math-7B 单基座 + OpenR1-46k 单训练集,跨模型族/跨数据未验证(依据:§5.1 仅此一配置);熵权的指数形式是"basic exponential",论文自承简单(Limitations)。
- 祛魅总结【推断】:真贡献=熵视角的 SFT/RL 机理分析(SFT 全局粗调 vs RL 选择性细调、单阶段优于串行、熵作指标)+ 在 LUFFY 上加两个熵自适应权重的工程实现。包装/高估:① abstract 的"+9.0/+10.9"是对 zero-RL 而非对最强基线(对最强 SFT+RL 仅 +3.4/+4.7),相对弱基线放大了观感;② "熵感知"机制的 SFT 权重方向在论文(exp−H)与开源代码(exp+H)间相反且作者未澄清,削弱"熵如何调度模仿"叙事的可信度。低估:单阶段融合的训练效率优势(Fig.5/6)论证较扎实但 abstract 未突出。

## 结构化抽取
- 🎯 机制速览6轴:
  - **学什么信号**:demonstration 的 token 级 NLL(SFT 模仿)+ on-policy rollout 的 group-norm advantage(RL,二值奖励{1,−1})+ 当前策略熵 H(πθ)作为调权信号。
  - **改什么**:同一策略模型 πθ 全参数(单 loss 反向)。
  - **何时改**:单阶段训练全程,500 步;每步按当前熵动态调 SFT/RL 权重。
  - **免梯度?**:否——是梯度训练(SFT+RL 联合 loss);熵权用 stop_grad 仅阻断"权重自身"的梯度,主 loss 仍正常回传。
  - **记忆-技能生命周期**:无显式记忆/技能库;"技能"即固化进 πθ 参数;demonstration 提供外部专家轨迹做粗粒度逼近,self-rollout 做细粒度精修。
  - **防遗忘机制**:单阶段融合本身缓解"RL 阶段灾难性遗忘 SFT 知识"(§3.2.2 指出两阶段 SFT→RL 的 RL 段会遗忘 SFT 知识致 transient 退化);熵感知权重防 demo 过拟合/RL 过早熵坍缩。无显式 KL-to-SFT 正则(kl_loss_coef=0)。
- ⑦ 开源代码+框架/harness:https://github.com/fyqqyf/SRFT (Tier A,v1 验证已克隆约 2.4MB,代码完整)。框架 **veRL + vLLM**(rollout/评测),底座沿用 **LUFFY 的 mix_src 结构** 与 deepscaler 奖励。核心实现 `srft/verl/verl/mix_src/`(`mix_core_alg.py` 的 `compute_token_on_off_policy_loss`、`mix_actor.py` loss 组装)。模型权重 HF `Yuqian-Fu/SRFT`。
  - **〔待核/已 flagged 的 paper-code 符号差异〕**(承 v1):论文 Eq.8 写 SFT 权重 `exp(−H)`,而开源代码 `mix_core_alg.py` 实为 `entropy_exp_coeff = entropy.exp().detach()`(即 `exp(+H)`),方向相反——以代码为准则"熵高加强模仿",与论文叙述(熵高减弱模仿)矛盾,作者未澄清(可能论文笔误或代码 bug)。RL 侧 `exp(+H)` 则 paper-code 一致。0.5/0.1 系数一致。本轮 PDF 已二次确认论文侧确为 `exp(−H)`(§4.1 Eq.8 L566 原文)。
  - adaptive temperature(可学习 log_alpha 对齐 target entropy,类 SAC)是代码可选项、**默认关闭**(`use_adaptive_temperature: False`),非核心组件;论文正文未把它列入方法/贡献(Limitations 只说"future work 可探索 adaptive entropy scheduling")。
- 💰 资源/成本与可扩展性:训练 64×A100(脚本 n_gpus_per_node=8、nnodes=4),500 步;8 rollout/prompt,max_prompt=1024/max_response=8192,actor lr=1e-6,train_batch=128/ppo_mini_batch=64,kl_loss_coef=0、entropy_coeff=0.001,sft_loss_coef=-0.5,tp=2,use_dynamic_bsz【元信息+v1 脚本核】。仅单基座单数据集验证,跨规模可扩展性原文未说明。
- 🎯 对"探索-巩固"对标:**支撑/可借组件**。一句判定:SRFT 是"统一 SFT-RL"线里与本项目 idea 最贴近的一类——它用**熵**当信号在"模仿专家(巩固)"与"自探索(探索)"间动态定权,正是"探索-巩固"的一种 loss 级实现;可借组件=熵感知自适应权重(把 H 当探索/利用调度的连续旋钮),以及"正样本 RL≈on-policy SFT"的拆解直觉(Eq.11)对设计 teacher 脚手架的稀疏监督有参考。依据:§4.1-4.2 熵权设计 + §3.1.1 "RL 在初始邻域做选择性细调"恰对应"偏向自己能走通的开头"的 on-policy 自选。缺口:SRFT 全程 dense 融合,无"稀疏脚手架/单点接管/走偏后自选恢复分支"机制;teacher 信号是整段 demonstration 而非关键步介入;无 MTP/前瞻。
- 🔭 开放问题/未来方向:【原文】熵利用目前仅"basic exponential weighting",可探索 adaptive entropy scheduling / 多时间尺度熵分析;假设高质量 demo,可研究 imperfect demonstration 训练(Limitations)。【推断】解决 paper-code 的 `exp(±H)` 符号矛盾(关乎"熵如何调度模仿"的正确方向);把熵感知权重从"整段 demo 均匀施加"细化到"按关键步/高熵 token 选择性施加",更接近稀疏脚手架;跨模型族/跨数据集泛化验证。

RETURN: srft | 读到PDF?是(22页/85100字) | L2(统一SFT-RL/GFT类,旁及L3) | 对标=支撑+可借熵感知探索-巩固调权组件,缺稀疏脚手架/单点接管/MTP | 残留待核1(paper-code `exp(−H)` vs `exp(+H)` SFT权重符号矛盾,作者未澄清)

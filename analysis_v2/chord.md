chord | On-Policy RL Meets Off-Policy Experts: Harmonizing SFT and RL via Dynamic Weighting (CHORD) | 阿里巴巴集团（Wenhao Zhang、Yaliang Li(通讯)、Jingren Zhou 等） | v3 2026-03-17 · ICLR 2026 · cs.LG | 主题线 L2（统一 SFT-RL / GFT 类代表作）·相关性 高

**原始论文**：https://arxiv.org/abs/2508.11408

## 一眼看懂
- 🟦 TL;DR：别再把 SFT 当 RL 前面的独立阶段。作者实测 SFT-then-RL 在 instruct 模型上会经历 **"shift-readapt-overfit"(漂移-再适应-过拟合)** 三阶段曲线,且不一定胜过纯 RL【原文 §3.1, Fig.1/2】。CHORD 把 SFT 重构成 on-policy RL 里一个**动态加权的辅助目标**:统一损失 L=(1−μ)·L_GRPO + μ·L_SFT;**全局系数 μ**(带 warmup 的余弦/退火调度)控制专家信号占比从"以模仿为主"平滑过渡到"以探索为主";**token 级权重 φ(p)=p(1−p)**(p 为策略对该专家 token 的概率)对"已很可能(p→1)"或"极不可能(p→0)"的专家 token 下调学习信号,只在 p≈0.5 处学得最多【原文 §3.2-3.3, 式(3)(5)(6)】。
- 最巧的一步：**token 级 φ(p)=p(1−p) 这条抛物线**。光有全局 μ 不够——case study 显示 CHORD-µ 会让模型整体照搬专家的冗长风格、覆盖自身简洁性(μ 缺乏精度)【原文 §3.2 L378-383】。直接用重要性采样(IS,权重 p.detach)又会让熵急剧坍缩(过度强化高概率 token、忽视低概率新 token→过自信),而不加 IS 则熵暴涨(established pattern 被破坏)【原文 §3.3 Fig.5, L423-431】。φ=p(1−p) 同时下调两端,在"模型还不确定"的 token 上集中学习,制造"learning sweet spot"。抽掉 φ:要么熵坍缩、要么熵暴涨,且模型被专家风格整体覆盖——CHORD-φ 的全面最优就垮了。

## 为什么做
- 研究背景：SFT(off-policy,静态专家示范,易过拟合/exposure bias)与 RL(on-policy,直接反馈、泛化好,但探索低效/熵坍缩)是两大后训练范式;常规 SFT-then-RL 直觉上"SFT 学专家模式引导 RL 越过局部最优、RL 缓解 SFT 的 exposure bias"【原文 §1 L32-49】。
- 解决的具体痛点：①SFT-then-RL **不一定胜过纯 RL**(Fig.1,Qwen2.5-1.5B + Open-R1)【原文 §1 L99-101】;②SFT 训练曲线呈 **shift-readapt-overfit**(Fig.2,Qwen2.5-7B + DeepSeek-R1 专家,MATH-500):突然策略漂移掉点(exposure bias 加剧)→再适应专家模式回升→过拟合到有限静态样本、丧失多样性与探索能力【原文 §3.1 L219-232】;③SFT→RL 转换时机高度任务相关(数学上 SFT-best+RL 最好、工具调用上 SFT-light+RL 更好),需大量调参,且两阶段分离本身次优。
- 相关工作 & 各自不足：dataset 混合(SimpleMix)、专家轨迹混进 rollout 组(LUFFY、SRFT)、专家数据引导生成(UFT、BREAD)、按预定/自适应调度交错 RL/SFT 步(SASR、Ma 2025)——SRFT 提 sample-level SFT loss 统一框架。作者强调本文聚焦"已建立自身回答模式的 **instruct 模型**"(比微调 base 更难也更实用)【原文 §1, §3.1】。
- 动机链：SFT-then-RL 二元开关(μ=1→0)太僵硬、经历 shift-readapt-overfit→改成可衰减 μ 调度可在过拟合前平滑退出(类比 scheduled sampling 缓解 exposure bias)→但全局 μ 缺精度(整体照搬专家冗长)→需 token 级 φ 把专家信号集中在"模型不确定"的 token→φ=p(1−p) 同时避熵坍缩(下调 p→1)与避干扰(下调 p→0)。为什么不用更简单做法:固定 μ 一律差于动态 μ(Fig.7);纯 IS 熵坍缩、无 IS 熵暴涨(Fig.5)——所以既要衰减 μ 又要 φ。
- 与最近邻工作的Δ：vs SFT-then-RL——把"独立阶段"变"动态加权辅助目标",μ 连续衰减而非二元开关;vs LUFFY/SRFT(把专家混进 rollout 组)——CHORD 用 expert_mask 区分专家/非专家数据,专家走 SFT 损失、非专家走 GRPO,凸组合;vs SASR(概率交错 SFT/RL 步)——CHORD 是损失级凸组合 + token 级精度。**关键有用点**:φ 提供"选择性吸收"——按任务自适应吸收专家模式(Table 2 长度证据),而非无差别模仿。

## 怎么做 + 靠不靠谱
- 方法流水线【原文 §3, Fig.3】：①一个 batch 混合 RL 任务 prompt + 专家(prompt,response)→②对 RL 任务采 K 条 on-policy rollout、算 reward 与组归一化优势 A→③非专家数据走 GRPO 损失(PPO clipped surrogate),专家数据走 token 级 SFT 损失(−φ(p)·logπ)→④凸组合 L=(1−μ)·L_GRPO + μ·L_SFT,μ 随步数动态调度→⑤更新策略。
- 逐组件必要性：
  - **全局 μ 衰减**：负责"模仿→探索"平滑过渡;Fig.7 消融——固定 μ 一律差于动态 μ、甚至可能不及纯 RL,小固定 μ(0.02)减损但提升不显著【原文 §4 Fig.7】。
  - **token 级 φ=p(1−p)**：负责"防熵坍缩/干扰 + 选择性吸收";Fig.8/9——CHORD-φ(固定 μ=0.1)既防熵过早坍缩、又避免熵暴涨,reward 稳定上升、显著优于纯 RL;用了 φ 后对 μ 选择鲁棒(不再需复杂 μ schedule)。Table 2——CHORD-φ 数学响应长度 2444、工具调用 120(vs CHORD-µ 被拉到专家级冗长 6081)。
  - **expert_mask 区分数据**：负责"哪条走 SFT、哪条走 GRPO";是凸组合的前提。
  - **两个实例(CHORD-µ / CHORD-φ)**:CHORD-µ 只用全局 μ 调度(关 φ),CHORD-φ 启用 φ + μ 取较小常值——论文明示不存在跨所有任务最优的单一 φ,但 p(1−p) 是有效鲁棒实例【原文 §3.3 L470-473 + 既有 analysis 代码核查】。
- 关键机制/公式(直觉)：L=(1−μ)L_GRPO+μL_SFT【式3】;φ(y*_t)=p_t(1−p_t)【式5】——信息论上 p(1−p) 是"生成该 token 这一二元事件"的策略不确定性度量,偏向"模型最不确定"的 token,制造 learning sweet spot【原文 §3.3 L464-469】。L_SFT-IS(式4)是 IS 变体(权重 p.detach,假设分母=1、把专家当 ground-truth 分布),被 φ 取代。
- 实验与证据【原文 §4】：
  - **数据集/设置**:数学——OpenR1-Math-220k(5k SFT/20k RL 无重叠),policy=Qwen2.5-7B-Instruct(与专家 DeepSeek-R1 风格差异显著),评测 AIME24/25/AMC,MMLU-Pro 监控通用推理;工具调用——ToolAce 单轮(5k RL/500 SFT,专家 DeepSeek-R1),policy=LLaMA3.2-3B-Instruct,评测 BFCL(Live/Non-live)【原文 §4.1】。
  - **主表(Table 1)**:CHORD-µ 数学超强基线 SFT-best+RL(AMC +2.4/AIME24 +1.0/AIME25 +1.6),工具调用整体也更好;CHORD-φ 进一步在数学+工具调用全面最优(MMLU-Pro 56.2 显著高、BFCL Overall 78.5)【既有 analysis,基于 Table 1】。
  - **响应长度(Table 2)**:专家 DeepSeek-R1 远长于原模型(数学 6132 vs 659 token);CHORD-µ 被拉到专家级冗长(6081),CHORD-φ 取得更细平衡(2444)——token 级加权使模型按任务**选择性**吸收专家模式。
  - **进一步分析**:换专家源(DeepSeek-R1 vs 风格更近的 Qwen2.5-72B)CHORD 均超基线,偏模仿方法(SFT+RL/CHORD-µ)在专家风格相近时增益更大;扩展非可验证域(RaR-Medicine)CHORD 仍超纯 RL;弱模型(Qwen2.5-3B)上 naive 模仿/SFT+RL 不稳,CHORD-φ 更鲁棒【既有 analysis】。
  - **baseline 公平吗**:基线很全(Original/SFT-light/SFT-best/SFT-light+RL/SFT-best+RL/SASR/纯 RL/LUFFY)。**诚实标注**:为公平,LUFFY 数学用 20k(原论文 45k)、工具调用用 500 SFT(原 5k),并列出原论文分数——透明但意味着 LUFFY 在此非其最佳配置【原文 §4.2 脚注1 L507-509】。
  - **看着强但没回答核心问题**：主实验规模有限(7B/3B/1.5B policy),更大模型 + 异构专家混合是 future work;模式漂移分析主要是经验性的,缺"不同 CoT 模式如何影响学习"的机理理论。
- 假设与失效边界：
  - 【原文】聚焦 instruct 模型(已建立回答模式);base 模型场景未必同理。
  - 【原文 §3.3 L419-422】L_SFT-IS 假设专家数据分母 IS 比为 1(把专家当 ground-truth 分布)——常见但近似。
  - 【原文/推断】μ、φ 配置跨任务/数据/模型会变,无单一普适最优(作者明示);自适应 reward-aware μ 可行但调参重。
  - 【推断】φ=p(1−p) 的"sweet spot"假设"中等概率 token = 最值得学的新信息"——若专家轨迹的关键信息恰落在模型已高概率或极低概率的 token 上,φ 会抑制它(失效边界)。
- 祛魅总结【推断】：真贡献是**"把 SFT 当 RL 动态加权辅助目标"这一统一视角 + 一个简单鲁棒实例(μ 衰减 + φ=p(1−p))**,而非全新算法。论文-代码高度一致(损失公式、μ schedule、φ、expert_mask 均在 `chord_policy_loss.py` 复现),自洽性强,诚实承认 φ 非唯一最优、配置随 setup 变。被高估的可能是"统一视角"的新颖性(SRFT 等已有相关统一框架);被低估的是 **shift-readapt-overfit 的系统刻画**(Fig.2)这一诊断本身的价值——它把"为什么 SFT-then-RL 次优"讲清楚了。规模有限是主要短板。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：on-policy 部分=可验证 reward 的 GRPO 优势;off-policy 部分=专家示范的 token 级 NLL,经 φ(p)=p(1−p) 重加权(只学中等概率 token)。
  - **改什么**：参数(策略 π_θ);通过凸组合损失同时受 RL 梯度与加权 SFT 梯度更新。
  - **何时改**：在线 per-step(RL 训练循环内,每步同时算 GRPO loss + SFT loss);μ 随训练步动态衰减。
  - **免梯度?**：否(全梯度);φ 权重用 sg/detach(p.detach)不反传到权重本身。
  - **记忆-技能生命周期**：专家数据集 D_SFT 是固定 off-policy 示范池(写入:外部专家如 DeepSeek-R1 生成;检索:每 batch 按 expert_data_ratio 采样混入;无遗忘/共享/更新机制)。
  - **防遗忘机制**：μ 衰减 = 防"过拟合专家"(在 overfit 前退出专家影响);φ = 防"established pattern 被破坏(熵暴涨)"与"过自信(熵坍缩)"——两者共同保护模型自身能力不被专家覆盖,可视为一种"防灾难性覆盖"机制。
- ⑦ 开源代码+框架/harness：https://github.com/modelscope/Trinity-RFT 。示例 `examples/mix_chord/`(`mix_chord.yaml` 数学、`mix_chord_toolace.yaml` 工具、`get_openr1_data.py`、README);损失实现 `trinity/algorithm/policy_loss_fn/chord_policy_loss.py`(`MIXCHORDPolicyLossFn` + `SFTPhiLossFn`/`SFTISLossFn`/`SFTLossFn` + `mu_schedule_function`);另有 ms-swift 集成【既有 analysis 代码核查】。框架 **Trinity-RFT(modelscope,基于 veRL 后端 + Ray)**,算法注册名 `mix_chord`,用 `expert_mask` + `expert_data_ratio`(示例 0.20)区分专家/非专家。代码可得性好(损失实现 + 复现脚本 + 超参齐全)。
- 💰 资源/成本与可扩展性：相比纯 RL,额外成本主要是每 batch 多算一次 SFT loss(专家数据 forward),无额外采样。policy 规模 7B/3B/1.5B,仓库默认示例 Qwen2.5-1.5B-Instruct。可扩展性:声称非可验证域(RaR-Medicine)、弱模型(3B)、异构专家均适用,但更大模型未验证(future work)。具体 GPU 数/wall-clock 原文未在正文给出(详见 Appendix A,本次未展开)。
- 🎯 对"探索-巩固"对标：**强支撑 + 高度可借(L2 主线代表作)**。CHORD 直击本项目"巩固/固化"环节的核心矛盾——**如何把外部/专家知识固化进参数而不破坏(遗忘)模型自身已有能力**。其"shift-readapt-overfit"诊断 = "巩固时若无差别覆盖会破坏 established pattern",正是本项目"巩固且不遗忘"要规避的失败模式。**与探索-巩固的精确对标**:(1)μ 衰减 ≈ **脚手架撤除曲线**(从模仿专家→自主探索,与 CBRL 退火、TSRD"撤脚手架"同构);(2)φ=p(1−p) ≈ **选择性巩固**——只在"模型不确定的 token"上吸收专家信号,与本项目"在关键步/高熵处接管"的 path-selection 思路高度同源(MEMORY 记录的"切高熵/低置信关键步"几乎是同一直觉);(3)expert_mask 凸组合 = on-policy 探索与 off-policy 巩固在**同一 batch 内共存**,而非分阶段——这对本项目"边探索边巩固"的设计极有参考价值。**最可借组件**:φ(p)=p(1−p) 的 token 级门控 + μ 调度 + `chord_policy_loss.py` 的现成实现(SFTPhiLossFn/mu_schedule_function),可直接作为"巩固损失"的起点。**Δ/缺口**:CHORD 的专家是静态外部示范池,无 path-recovery 的"走偏后自选恢复分支"、无 MTP 前瞻、无技能/记忆生命周期——它解决"如何吸收已有专家轨迹",不解决"如何发现/生成要巩固的好行为"(那是探索侧)。一句判定:本项目"巩固/防遗忘"环节的最直接可借范式与代码来源,φ+μ 双控是现成模板;但需补上探索侧(脚手架生成、path-recovery、前瞻)。
- 🔭 开放问题/未来方向：
  - 【原文】自适应 reward-aware μ(按 reward 趋势调,但调参重);更大模型 + 异构专家混合;"不同 CoT 模式如何影响学习"的机理理论(当前漂移分析偏经验)。
  - 【推断】φ 的设计空间未穷尽(论文也试过 entropy-based/clipping/focal 变体)——可探索"前瞻感知"的 token 门控(用 MTP 预测未来收益决定该 token 学多少),把 CHORD 的"当前不确定性"φ 升级为"未来价值"φ,直接对接本项目 MTP 前瞻;把静态专家池换成"模型自己探索出的成功轨迹"(self-distillation 闭环),让探索侧产出直接喂给 CHORD 巩固侧。

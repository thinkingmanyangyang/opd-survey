open_agentrl | RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System | Gen-Verse / Princeton（Yinjie Wang, Tianbao Xie, Ke Shen, Mengdi Wang, Ling Yang 通讯） | 2026-02-02·arXiv·v1（仓库 README 标 ICML 2026,PDF 自身未声明 venue） | 主题线 L4(Agent/工具·多轮自进化)·相关性 中

**原始论文**:https://arxiv.org/abs/2602.02488 （代码 github.com/Gen-Verse/Open-AgentRL）

## 一眼看懂
- 🟦 TL;DR:一个"完全动态"的闭环 RL 系统,**同时进化环境、策略、生成式奖励模型三者**——策略用 step-wise+outcome 融合反馈训练,奖励模型经一致性反馈联合优化产出可靠的逐步监督,环境据策略当前能力自适应调难度;并论证"优化后的逐步奖励信号优于人工写的 outcome 标签"。【原文】abstract+§2
- 最巧的一步:**把"环境难度自适应"用理论(Theorem 1&2)接进闭环**。抽掉环境自适应——奖励模型的训练环境(由策略轨迹诱导)在任务过难/过易时会让 p+、p− 的重要性采样极不平衡、违反 reward-precision 目标 μ=p++p−>1(Theorem 2),奖励模型质量下降,进而策略也学不好。正是"据 acc 落在 [0.2,0.8] 外时调难度 + 质量门控"把这条因果闭上,使三者互相放大(Table 1:每加一个动态组件都一致提升)。

## 为什么做
- 研究背景:RLVR 提升 LLM 推理;但当策略与环境长轨迹迭代交互时,二值 outcome 奖励监督不足。step-wise 信号常由生成式奖励模型给出(借语言模型推理能力,优于标量奖励模型),但训练这类模型需高质量任务专属监督。环境质量(任务难度与策略能力匹配、任务多样性)对 RL 扩展同样关键。【原文】§1
- 解决的具体痛点:① 长轨迹下二元 outcome 奖励信号不足;② 奖励模型与策略割裂,且生成式奖励模型难获得可靠 step-wise 监督;③ 环境任务难度与策略当前能力不匹配,影响策略与奖励模型双方训练动态;④ 三要素(环境/策略/奖励模型)通常被分开处理。【原文】§1
- 相关工作 & 各自不足:RLVR 单 outcome 奖励可联合优化奖励模型+策略,但不直接扩到多轮(缺 step-wise 监督);调难度可改善策略(Zeng 2025 多难度引擎、Xue 2026 可验证任务合成),但这些系统缺长 horizon 交互所需的 step-wise 信号。【原文】§4.1+§4.2
- 动机链:单 outcome 在长轨迹太稀疏 → 需 step-wise(生成式奖励模型)→ 但奖励模型需可靠监督且与策略割裂 → 让策略轨迹当奖励模型的训练环境(consistency feedback)→ 但奖励模型质量取决于任务难度是否平衡(Theorem 1&2)→ 所以环境也要据策略能力动态调 → 三者闭环。【原文】§1+§2
- 与最近邻工作的Δ:vs 标准 agentic RL/RLVR——不是只优化策略,而是**环境/策略/奖励模型三者同时动态、互为反馈**;vs 调难度类工作——难度调整由"奖励模型对失败步的诊断摘要"驱动(targeted),而非盲调,且有理论说明为何调难度同时利好奖励模型。关键点:论证 step-wise(奖励模型)信号可**替代甚至超过人工 outcome 标签**(§3.2.6),为大规模自进化 agent 去掉人工评测脚本铺路。【原文】§2+§3.2.6

## 怎么做 + 靠不靠谱
- 方法流水线(输入→输出,Algorithm 1):① 采任务 q,策略 π_θ 采一组轨迹 τ(outcome O_τ∈{−1,1})→ ② 奖励模型 r_φ 对每步 τ_i 独立评 m=3 次得 S_τi,j∈{−1,1} 及评估推理 r_τi,j → ③ step reward R_τi = O_τ + (λ/m)Σ_j S_τi,j(Eq.1,λ=1),在同一步 index 跨轨迹标准化得优势训策略 → ④ 奖励模型监督信号 RS_τi,j = R_τi·S_τi,j(Eq.2,跨 j 标准化)训其评估推理 → ⑤ 据 rollout acc:落在 α_low=0.2/α_high=0.8 外时,把"失败步(S=−1)的评估摘要"喂给 LM(Qwen3-4B)提议更难/更易任务 q',质量门控(变难仅当 α_low<acc(q')<acc(q);变易仅当 acc(q)<acc(q')<α_high)后替换 → 循环。【原文】Algorithm 1+§2.1-2.4
- 逐组件必要性:
  - **integrated feedback(step+outcome,Eq.1)**:有消融(Fig.6a),outcome-only 在长轨迹太稀疏——必要,尤其长 horizon。
  - **联合优化奖励模型(Eq.2)**:Table 1 中 Policy→Policy+Reward 一致提升(且 reward process/outcome acc 上升)——必要。
  - **环境自适应**:Table 1 中 Policy+Reward→+Env 进一步提升所有指标(尤其 OOD 与奖励模型 acc),Fig.4 训练曲线更高收敛——必要,且 Theorem 2 给理论依据。
  - **λ(step vs outcome 权衡)**:有消融(Table 2,Alf World,policy 在 λ=1 最优 acc 54.1;λ 越大奖励模型 process acc 越高但 outcome acc 略降)——λ=1 默认。
  - **质量门控/接受机制**:Fig.7a + §3.2.9,用更强模型(Qwen3-VL-32B/Qwen3-32B)验证接受任务 pass-at-least-one 96.0%/96.7%/94.2%——证过滤掉错误合成任务。
- 关键机制/公式(直觉):reward precision A=P(S_τ+>S_τ−|O_τ+=1,O_τ−=−1)——奖励模型不仅要判单步对错,还要预测该步对最终 outcome 的影响。Theorem 1:A→1 当且仅当 μ=p++p−>1(且 A≥1−e^{−m(μ−1)²/4});Theorem 2:任务过难(P(O=−1)→1)或过易(P(O=1)→1)会使 f+/f− 重要性权重比 →0 或 →∞,破坏 μ 目标。直觉:要让奖励模型学得准,得让任务难度处在"有对有错"的平衡区——这就是为何"环境自适应"既利策略也利奖励模型。【原文】§2.3
- 实验与证据:三场景——① GUI:OSWorld,策略/奖励均 Qwen3-VL-8B-Thinking,任务改写 Qwen3-4B,RL rollout 最大 30 步/评测 50 步,每步采 12 任务×8 rollout,奖励模型每步评 3 次;230 in-domain + 139 OOD(训练排除 Multiple Apps/Chrome 作 OOD)。② Alf World:策略 Qwen2.5-7B-Instruct、奖励 Qwen2.5-14B-Instruct、改写 Qwen3-4B;官方 3.5k 训练/140 in-domain/134 OOD;每步采 16 任务×8 rollout。③ coding:同 Alf World 模型组合,无交互环境;LiveCodeBench-V2/CodeContests/LiveBench 评测,CodeContests(难度≤2,4.5k 训/200 评)训练;每步 64 任务×(32 解+32 单元测试)。关键数字(Table 1,Policy+Reward+Env 相对 Before):GUI in-domain +11.7、OOD +5.2;LLM agent in-domain +21.2、OOD **+18.7**;LiveBench code +11.9、UT +51.1、Detect +28.9。abstract 口径:OSWorld **+9.1%**(§3.2.4)、Alf World +18.7%、LiveBench +11.9%。baseline 公平性:逐组件消融(每加一个动态组件)较公平;但 step reward 把 O_τ 直接加进每步,step-wise 与 outcome 部分耦合。【原文】Table 1+§3.2.4
- 假设与失效边界:
  - 【原文】§3.2.6:用"仅奖励模型 step-wise 监督、无 outcome 脚本"训练甚至超过用 verifiable outcome——但这是在 GUI 这类"人工写评测脚本"的设定下;隐含假设=奖励模型能可靠评步。
  - 【推断】论文**无独立 Limitations 节**(extracted text 未见 Conclusion/Limitations 标题,待核附录)。reward precision 理论在 m→∞ 渐近,有限 m=3 下偏差未充分量化;环境改写依赖 LLM 质量与阈值/门控;奖励模型与环境改写均由 LLM 担任,存在自评/自改写的潜在偏置;评测集中 OSWorld/Alf World/coding 三域。
- 祛魅总结:【推断】真贡献=① 把"环境-策略-奖励模型"三者纳入一个有理论支撑(Theorem 1&2 把 reward precision 与任务难度平衡连成因果)的闭环,这个 framing 较新且自洽;② "优化后 step-wise 信号可替代/超过人工 outcome 标签"(§3.2.6)若稳健,对去人工评测、扩 agentic 环境有实际意义。被适度高估:step reward 把 outcome 直接加进每步(Eq.1),"step-wise 优于 outcome"的对比需在此耦合下解读;三组件全由 LLM 自评/自改写,可能积累偏置;与 OPD/蒸馏主线**无直接关联**(本文不含任何蒸馏),只作"过程信号价值 + 环境难度自适应"的外部论据。

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号:策略学 step+outcome 融合优势(Eq.1);奖励模型学"评估推理"的一致性优势(Eq.2);环境学"据 acc 调难度"。
  - 改什么:策略 π_θ、奖励模型 r_φ 两套参数 + 环境任务集 Q(任务被改写替换)。
  - 何时改:每个 RL step 三者同步更新;环境仅当 acc 落在 [0.2,0.8] 外且通过门控时改任务。
  - 免梯度?策略/奖励模型走梯度(GRPO 类 group-based);环境自适应是**免梯度**的 LLM 任务改写 + 阈值门控。
  - 记忆-技能生命周期:无显式记忆/技能库;"经验"以"奖励模型诊断摘要→改写任务"的形式驱动 active learning from experience;任务集 Q 随训练演进(接受任务近似线性增长)。
  - 防遗忘机制:原始训练任务保留在所有 setup(§3.1.4 为公平比较);无显式防遗忘机制(非持续学习设定)。
- ⑦ 开源代码+框架/harness:github.com/Gen-Verse/Open-AgentRL(已 clone,~38M,同仓库另含 DemyAgent;HF 有 Policy & Reward 模型 collection)。框架 **veRL**(仓库内置 verl/ 目录,GRPO 类 group-based RL);含 recipe/、train/、reward/、按任务的 alfworld_rl.py/coding_rl.py/osworld_rl.py 训练入口与 *_eval.py,以及 OSWorld-main/、alfworld_master/ 环境;requirements_rlanything.txt / requirements_sglang.txt / requirements-npu.txt。代码可得性高(完整训练/评测/环境脚本齐全)。【原文/仓库】
- 💰 资源/成本与可扩展性:多模型同时在线(策略+奖励模型+任务改写 LM Qwen3-4B,Alf World 奖励模型 14B 大于策略 7B);GUI 每步 12×8 rollout、奖励每步评 3 次,长 horizon(GUI 30 步/Alf 40 步 RL rollout);GUI 预造 47/230 任务的扰动版(含 evaluator/verifier 文件)。具体 GPU 数/卡时论文正文未明说。【原文】§3.1
- 🎯 对"探索-巩固"对标:**间接支撑(探索侧)+ 可借的环境调度组件**。一句判定:本文与 OPD/蒸馏主线无方法交集,但它的"**环境难度自适应**"恰是"探索=发现有效行为/路径"的环境侧实现——把任务维持在"有对有错"的平衡区正对应"偏向自己能走通的开头/边界"。依据:Theorem 2 + §2.4 据 acc 调难度。可借组件:① **据 rollout acc 的难度门控调度**(α_low/α_high + 质量门控)——可直接用于"探索-巩固"的课程,把训练样本维持在学生能力前沿;② **奖励模型对失败步的诊断摘要→targeted 任务改写**(Fig.3:0.0→0.25)——这与"识别走偏的关键步并给针对性脚手架/hint"思路相通,是 path-recovery 的环境侧类比;③ step-wise 过程奖励对长 horizon 的必要性(Fig.6),支撑"过程级信号优于纯结果"。缺口:① 无任何蒸馏/特权信息/on-policy 自蒸馏(纯 RL + 生成式奖励模型);② 无 MTP/前瞻;③ 无"把能力固化进参数/记忆且不遗忘"的巩固机制(任务集演进 ≠ 参数级巩固);④ step-wise 信号来自外部奖励模型而非"看过答案的自己",与 teacher-scaffold 思路不同源。
- 🔭 开放问题/未来方向:
  - 【原文】abstract+§3.2.6:enabling large-scale self-evolving agents in real-world environments(用奖励模型替人工评测脚本、扩环境);新环境任务线性扩展。
  - 【推断】把"奖励模型诊断摘要→改写任务"的 targeted 难度调度,与 OPD 的"teacher 稀疏脚手架"结合——用奖励模型定位策略走偏的关键步,既调环境难度又触发 path-recovery 监督;把 MTP 前瞻接到奖励模型的"预测该步对 outcome 影响"(reward precision A 本质就是前瞻),可能让过程奖励更早、更准地定位关键分支步。

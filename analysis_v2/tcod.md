tcod | TCOD: Exploring Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents | Tongyi Lab, Alibaba Group + 香港中文大学(Jiaqi Wang†, Wenhao Zhang, Weijie Shi, Yaliang Li, James Cheng†) | arXiv 2604.24005 v3(2026-04-29, cs.LG) | 主题线 L1(On-Policy Distillation)+ L4(多轮自主 agent)·相关性 **高**(与 mtp_opd 直接相关)

**原始论文**:https://arxiv.org/abs/2604.24005

## 一眼看懂
- 🟦 TL;DR:on-policy 蒸馏(OPD)在单轮数学/QA 上很稳,但搬到**多轮 agent** 就崩——作者命名为 **Trajectory-Level KL Instability**:训练中 KL 不降反升、成功率坍塌到近零。根因是**跨轮误差累积**(学生动作/观测追加进历史,把学生推出 teacher 有效支撑,teacher 给学生 token 赋低概率、监督信号失真)。TCOD 用一条**时间课程**逐步放大暴露给学生的轨迹深度 \(k\)(线性 pacing),两个变体:F2B(学生从早段学起、\(k\) 由浅到深)、B2F(teacher 先把环境导航到近终止态、学生从"成功门口"接管、\(k\) 反向递增到端到端)。最高 +18 SR,且能在 teacher 失败的 hard split 上反超 teacher。【原文 Abstract, §4, Table 2-3】
- 最巧的一步:**沿时间维度控制轨迹深度 \(k\) 当课程难度**(\(k=k_{start}+\lfloor n/\eta\rfloor\),无需外部难度评估器、只用学生自生成数据)。抽掉它(=vanilla OPD 一开始就端到端跑完整轨迹)就必然进入 teacher 支撑外、KL 飙升坍塌(Qwen3-1.7B 跨 benchmark avg 仅 0.17,Table 3)。B2F 里还有一步抽不得:课程末端把 teacher 前缀从 \(L-1\) 拉到 0,消除 train-test 分布漂移(Appendix D.5 验证端到端成功率随训练步稳升)。【原文 §4.2, lines 525-532】

## 为什么做
- 研究背景:OPD(Agarwal 2024 GKD;Lu & Thinking Machines Lab 2025)用稠密蒸馏信号替代稀疏标量奖励、提升样本效率,在单轮推理上很成功;ReAct 式多轮 agent(reason+act 交错)是主流范式,但 OPD 在多轮设定下行为未被探索。【原文 §1-2】
- 解决的具体痛点:**vanilla OPD 在多轮的失效模式**(ALFWorld 实证):① KL 持续升高 + 成功率坍塌至近零(与单轮 KL 收敛递减相反,Fig.2a/2b,小模型显著);② 即使最终收敛,初始 KL 极高(~1000 vs 收敛 ~60,Fig.2c);机理=跨轮误差累积使 per-turn KL 随轮次单调增(Fig.2d)。【原文 §4.1, lines 292-318】
- 相关工作 & 各自不足(把 OPD 改进与课程学习两条线铺开,定位 TCOD 的空白):
  - **OPD 目标设计这一路(都只管单轮)**:forward/backward KL 平衡(Jang 2026)、entropy-aware OPD 按 teacher 熵调 loss(Jin 2026)——改的是"散度方向/加权",不碰多轮误差累积。【原文 §2, lines 151-159】
  - **OPD 优化启发一路**:reward clipping(Ko 2026)等稳定 trick,仍是单轮。
  - **OPD 监督来源一路**:context distillation(Ye 2026)、self-distillation(Zhao 2026)——换的是"监督从哪来",非多轮。
  - **课程学习一路(近期用于 LLM RL)**:Bengio 2009 课程学习思想,近期接进 GRPO 等 RL——但**多依赖外部模型评估样本难度**(需额外打分器);TCOD 的难度=轨迹深度 \(k\),零外部难度模型。【原文 §2】
  - **最近邻的反例——Lauffer 2025**:用 expert 的"后续纠错动作"训学生,但这**破坏了 on-policy 设定**(混入了 expert 动作的监督);TCOD 的 F2B 全程用学生自生成数据保持 on-policy,B2F 的 teacher 步只导航不贡献梯度(见下),严格区别于此。【原文 §2】
  - 动机链:OPD 单轮稳→搬多轮 KL 失稳→根因是长 horizon 误差累积(固有属性)→借课程学习"先易后难"→把"轨迹深度"当难度、只用学生数据、避开外部难度评估器→TCOD(F2B/B2F)。
  - 与最近邻工作的精确Δ:vs 单轮 OPD 改进,关键差异=首次定位并解决"多轮误差累积致 KL 失稳",对策是沿时间轴做课程(而非改 KL 目标/加 clipping);vs 课程学习/Lauffer,关键差异=难度=轨迹深度 \(k\)(无需外部难度模型)、且全程用学生自生成数据保持 on-policy。【原文 §2, lines 151-159】

## 怎么做 + 靠不靠谱
- 方法流水线(输入→输出,逐组件):
  - **OPD 目标(reverse-KL,student 状态分布下对齐 teacher)**:\(L_{\mathrm{OPD}}(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\big[\sum_{t=0}^{T-1}D_{KL}\big(\pi_\phi(a_t\mid h_t)\,\|\,\pi_\theta(a_t\mid h_t)\big)\big]\),teacher \(\pi_\phi\)、student \(\pi_\theta\)、\(h_t\)=完整交互历史,\(D_{KL}(\pi_\phi\|\pi_\theta)=\sum_{a_t}\pi_\phi(a_t\mid h_t)\log\frac{\pi_\phi(a_t\mid h_t)}{\pi_\theta(a_t\mid h_t)}\)(Eq.2)。
  - **线性 pacing(课程难度调度)**:\(k=k_{start}+\lfloor n/\eta\rfloor,\ n\in\{1,\dots,N\}\)(Eq.4),\(n\)=当前训练步、\(N\)=总步数、\(k_{start}\)=初始交互步数、\(\eta\)=增长率;主实验 \(k_{start}=1,\eta=2\)。
  - **F2B(Forward-to-Backward,Algorithm 1)**:每训练步 \(k\leftarrow\min(k_{start}+\lfloor n/\eta\rfloor,\ T_{max})\);从 \(h_0\) 起学生 rollout \(k\) 步(\(a_t\sim\pi_\theta(\cdot\mid h_t)\),执行、更新 \(h_{t+1}\));loss \(L=\sum_{t=0}^{k-1}D_{KL}\big(\pi_\phi(a_t\mid h_t)\,\|\,\pi_\theta(a_t\mid h_t)\big)\)(Eq.3);更新 \(\theta\leftarrow\theta-\nabla_\theta L\)。先学早轮信号→渐进端到端;**无需任何示范**。
  - **B2F(Backward-to-Forward,Algorithm 2)**:预采集 teacher 成功轨迹 \(\mathcal{T}^*=\{\tau^*\}\)(pass@10 采样、保留成功)。每步 \(k\leftarrow\min(k_{start}+\lfloor n/\eta\rfloor,\ L)\);teacher 执行 \(\tau^*\) 前 \(L-k\) 步(**stop-gradient**,\(a^*_t\) 把环境带到中间态)→学生从 \(t=L-k\) 接管到 \(L\)、仅对学生步算 KL:\(L_{\mathrm{TCOD\text{-}B2F}}(\theta)=\mathbb{E}_{\tau\sim(\pi_\phi,\pi_\theta)}\big[\sum_{t=L-k+1}^{T-1}D_{KL}\big(\pi_\phi(a_t\mid h_t)\,\|\,\pi_\theta(a_t\mid h_t)\big)\big]\)(Eq.5);\(k\) 递增直到学生从初始态端到端。teacher 步只"把学生放到成功门口"、不贡献梯度。【原文 §4.2, Eq.2/3/4/5, Alg.1/2】
- 逐组件必要性:
  - **线性 pacing \(k\)**:核心。去掉=vanilla OPD,小模型崩(Table 3 Qwen3-1.7B avg 0.17 vs TCOD ~18.6)。
  - **F2B vs B2F**:两个并列变体——F2B 无需示范(drop-in),B2F 需预采 teacher 成功轨迹但能借 teacher 导航避早期误差累积。Table 2 上 3B 学生 F2B 更强(Unseen +18.74),7B 学生 B2F 更强(Hard +7.44)。
  - **课程末端回到端到端(B2F)**:消解 train-test mismatch——teacher 前缀从 \(L-1\) 步降到 0,确保训练末学生从初始态全程自走、对齐 train/test 分布,Appendix D.5 佐证(端到端 SR 随步稳升)。
  - **异步训练 + staleness-aware 子轨迹回放**:把长 \(n\) 轨迹拆成 \(n\) 个递归前缀子轨迹入 buffer;staleness filter 丢弃 \(n_{current}-n_{old}>\Delta_{max}\) 的经验,经验上 \(\Delta_{max}=2\) 最优。属工程稳定性细节(无独立消融,但给出经验最优值)。【原文 §4.3】
  - **\(\eta\)(增长率)消融**:\(\eta\in\{2,4,6\}\) 性能波动<2%(Table 3)→ 对 \(\eta\) 不敏感、易部署;但更大 \(\eta\) → KL 更稳(学生在当前深度多练几步)。
- 关键机制/公式(直觉):误差累积是长 horizon 固有属性——学生一步走偏,错误观测/动作追加进历史,后续每步都在"被污染的历史"上推理,per-turn KL 随轮次放大,最终被推出 teacher 支撑、监督失真。Remark 1 关键区分:**long-CoT 只在同一环境状态下增长响应长度,而多轮 agent 每轮更新环境状态,才真正放大累积误差**——这解释了为何单轮 OPD 稳、多轮崩。对策直觉:与其端到端进 OOD,不如沿时间做课程,让学生始终停在 teacher 有效指导范围内(F2B 限制 rollout 深度、B2F 用 teacher 把学生放到成功门口)。【原文 §4.1, Remark 1】
- 实验与证据:
  - 数据集=3 个多轮 agent benchmark(Table 1):**ALFWorld**(具身,max 30 turns,seen/unseen/hard split)、**WebShop**(电商,max 15 turns)、**ScienceWorld**(科学推理,max 30 turns)。Hard split=121 个 teacher 在 train split 上 pass@10 仍失败的任务。
  - 师生对:ALFWorld 主实验 student=Qwen2.5-{3,7}B、teacher=GRPO 训练的 Qwen2.5-7B(域内 SR 85.71);跨 benchmark student=Qwen3-{1.7,4}B、teacher=Qwen3-30B-A3B-Instruct(通用、域内偏弱)。8× H20(96GB)。【原文 §5.1】
  - 关键数字:Q1(Table 2)——Qwen2.5-3B 上 F2B 在 Valid Unseen 较 OPD **+18.74 SR**、Seen +15.71;7B 上 B2F Seen +11.06、Hard +7.44;平均减少 2.97 动作轮次。跨 benchmark(Table 3)——Qwen3-1.7B 下 vanilla OPD 几乎全崩(avg 0.17),TCOD 拉到 ~18.6(**+18.4~18.7**);Qwen3-4B 提升温和(avg +0.3~+1.9)。Q2——Hard split 上 teacher SR 仅 6.61,**B2F 反超最高 +14**(7B 达 20.66)→ 非单纯模仿。Q3——总训练时间较 OPD 减少近 32%(F2B 比 B2F 更省)。【原文 §5.2-5.4】
  - baseline 公平吗:对比 Zero-Shot(下界)、teacher(Oracle 上界)、SFT、vanilla OPD,同 backbone 同环境——公平。
  - "看着强但没回答核心":作者诚实——Qwen3-4B 上 TCOD 仅与 OPD 相当(Table 3,非显著提升);收益高度依赖"学生本会崩 + teacher 域内够强"。
- 假设与失效边界:
  - 【原文】收益条件性强:teacher **域内性能**(而非规模)是上界主因——域内强的 7B-RL teacher 下 B2F 可略超 teacher(+0.7);通用 30B teacher 下 OPD/TCOD 均差 teacher ~2 分(§5.4 Domain-Specific vs Larger Teacher;Appendix B Obs.2:师生容量匹配比绝对 teacher 强度更重要)。
  - 【原文】B2F 依赖预采集 teacher 成功轨迹(额外采样开销);F2B 是无示范替代。
  - 【原文】仅在 3 个**文本**多轮 benchmark 验证,未覆盖多模态/物理具身(Appendix A 自陈)。
  - 【原文/推断】一个机理张力:作者自承"KL 升高既可能是学生模仿不能、也可能是学生进 OOD 致 teacher 不确定"二者难区分——但两种解释指向同一对策,不影响方法有效性(lines 312-314)。
  - 【推断】固定 pacing 鲁棒,但最优节奏可能随环境/师生对变化(作者建议未来用 KL 的 EMA 自适应 horizon,未实现)。
- 祛魅总结【推断】:真贡献=(a) 清晰识别并命名多轮 OPD 的失效模式(Trajectory-Level KL Instability)+ 给出机理(误差累积)+ Remark 1 区分 long-CoT vs 多轮;(b) 一个极简、几乎零额外代码的时间课程修复方案 + 配套异步训练工程。包装上 "surpass teacher" 真实但条件性强(主要在 hard split + 域内强 teacher)。高估:把"超 teacher"读成普遍性(Qwen3-4B 上仅持平 OPD)。低估:失效模式的诊断本身对整个多轮 OPD 方向很有价值,且 +32% 提速是实打实的工程收益。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=token-level reverse-KL(teacher vs student),仅在课程允许的轨迹深度内计算 | **改什么**=student 策略参数 \(\theta\)(标准 OPD 梯度) | **何时改**=每训练步,课程深度 \(k\) 随步线性增长 | **免梯度?**=否,学生执行步算 KL 梯度;**teacher 执行步 stop-gradient**(B2F 导航段) | **记忆-技能生命周期**=不涉及外部记忆/技能库(纯参数蒸馏) | **防遗忘机制**=隐式——课程渐进 + 子轨迹经验回放 + staleness filter(\(\Delta_{max}=2\))维持 on-policy、防分布漂移;无显式回放旧任务式防遗忘
- ⑦ 开源代码+框架/harness:https://github.com/kokolerk/TCOD(已克隆,README 明确"What Is Implemented":三环境各一套 multi-turn OPD + TCOD-f2b + TCOD-b2f workflow)。**框架=Trinity-RFT(Pan 2025,Ray-based RFT,Alibaba)**;`trinity run --config <yaml>`,依赖 vLLM 类推理 + flash-attn;关键配置 `algorithm.advantage_fn: multi_turn_opd`、`rollout_args.logprobs` 必开(算蒸馏 gap)、TCOD 用 `workflow_args.checkpoint_strategy: linear`(线性 pacing)。模型:ModelScope `wjqkoko/TCOD` / HF `kolerk/tcod`。环境 ALFWorld(pip)/ WebShop(需 Java17+、~1TB 内存)/ ScienceWorld(jar)。
- 💰 资源/成本与可扩展性:8× H20(96GB):4 actor + 2 learner + 2 teacher,lock-free ring buffer 异步;TCOD 较 OPD 训练时间 **−32%**(早期短轨迹采集更快);F2B 比 B2F 更省(B2F 接管后仍多探索几步)。WebShop 需 ~1TB 内存是部署门槛。【原文 §4.3, Fig.6】
- 🎯 对"探索-巩固"对标:**强支撑 + 直接竞品/可借组件**。一句判定:TCOD 是本项目 idea 在"多轮 agent OPD"上的最近邻实现——**B2F 几乎就是"teacher 稀疏脚手架 + 学生从成功门口接管 + 渐进撤掉脚手架"的 path-recovery 雏形**(teacher 导航到近终止态=把学生放回正轨,stop-gradient=teacher 不被学生梯度污染,\(k\) 反向递增=逐步把控制权交还学生);F2B 对应"探索/选路从早段稳定前缀学起"。可借组件:① B2F 的"teacher 前缀导航 + stop-gradient + 课程撤离"可直接迁移为 path-recovery 的脚手架调度;② Trajectory-Level KL Instability 的诊断方式(per-turn KL 随轮增)是检验"走偏"的现成探针。缺口/Δ:TCOD 是**轨迹深度**层面的课程(粗粒度,按 turn),本项目要的是**单点/关键步**接管(细粒度)+ on-policy 自选恢复分支 + MTP 前瞻——TCOD 既无 step 内关键点定位、也无 MTP、也不"让学生自选恢复分支"(B2F 是 teacher 固定前缀导航,非学生自选)。可作最强对照 baseline,但需在"切点粒度"和"恢复分支由谁选"两处做出区分。【推断,依据 §4.2 B2F 机制与本项目 path-recovery 定义的逐项对应】
- 🔭 开放问题/未来方向:【原文】用 KL 的 EMA 做自适应 horizon(替代固定 pacing);扩到多模态/物理具身环境(Appendix A)。【推断】把"轨迹深度课程"细化到"step 内关键点接管"(对标本项目切关键步);让学生在 B2F 接管点**自选**恢复分支而非沿 teacher 固定前缀;引入 MTP 前瞻作为"是否该让 teacher 介入"的探针;与 token 级重要性(TIP)结合,在课程允许的深度内再做 token 选择。

〔本篇与既有 analysis/tcod.md 核对:无事实冲突。本次增强:5 条核心公式(OPD 目标 Eq.2 + \(D_{KL}\) 展开、pacing Eq.4、F2B loss Eq.3、B2F loss Eq.5)转 MathJax 并从 PDF 抄准;相关工作铺开 OPD 改进三条支线(目标/优化/监督来源)+ 课程学习依赖外部难度模型的缺陷 + Lauffer 破坏 on-policy 的反例;方法流水线把 F2B/B2F 两 Algorithm 拆到"每步 k 更新→rollout/导航→loss→更新"逐行,标 stop-gradient 与 \(k_{start}=1,\eta=2\) 默认值。无新增待核项。〕

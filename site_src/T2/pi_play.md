# pi_play — π-Play: Multi-Agent Self-Play via Privileged Self-Distillation without External Data

> **一句话重点 (TL;DR)**：自博弈在造题时天然产出一条 "问题构造路径(QCP)"，本文把它当作零成本的内禀特权信息，让同规模 teacher 据此对 student 做 token 级 reverse-KL 自蒸馏，从而把稀疏奖励自博弈变成稠密反馈的 data-free 自演化。

**元信息**：arXiv 2604.14054（v2 2026-05-25）｜ 中科院自动化所 + 国科大 + 美团 ｜ 2026-04 预印本 ｜ 主题 self-play + self-distillation for 深度搜索 agent，与 TSRD（QCP=foresight/教师脚手架）相关 ｜ 代码 github.com/zhyaoch/pi-play（README 注明 "Code will be released soon"，当前仅 README+images，**训练代码未发布**）｜ 框架 论文未绑定特定开源框架，student 用 GRPO。

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/pi_play/fig_01.png)

*Figure 1: Comparison of π -Play with other self-evolution frameworks. All models (examiner, teacher, and student) in π -Play are initialized from the same base LLM and function as search agents. π -Play uses alternating optimization to evolve multiple agents in a closed loop. Compared to self-play,*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/pi_play/fig_02.png)

*Figure 2: Overview of QCP-guided self-distillation in π -Play. The examiner is equipped with search tools and interacts with the search engine to obtain factual information, ensuring the correctness of both the synthesized QA pairs and their construction paths c . The teacher policy π T ψ leverages*

## 1. 相关工作与进展
深度搜索 agent 结合 LLM 推理与外部搜索引擎做多轮检索分析；RL 能显著提升其推理与搜索行为（Search-R1、ToolForge 等监督 RL）。自博弈（Dr.Zero、SQLM、SSP）让同规模模型轮流当 examiner/student 以降低数据依赖；自蒸馏可改善 credit assignment，但需高质量特权信息。

## 2. 现有工作存在的问题
(1) 自博弈只给学生**稀疏结果奖励**，多轮搜索任务学习低效、信用分配弱；(2) 自博弈实际产出的不止最终 QA 对 (q,o)，还有被忽视的中间产物——**QCP(c)**，记录从答案反向构造问题的逆向解题过程，但现有自博弈无法直接拿它做 SFT；(3) 自蒸馏所需高质量特权信息获取代价高，且常依赖人类专家/更强模型与 curated QA 数据，难规模化。

## 3. Motivation
关键观察：自博弈天然、低成本、可规模化地产出 QCP，而 QCP 记录了问题如何从事实证据构造而来，正是一种**内禀特权信息**——可让同规模 teacher 在条件于 c 时生成比仅看 q 的 student 更准确的 rollout。由此把自博弈产生的 QCP 直接喂给自蒸馏，无需人类反馈或人工特权数据。

## 4. 主要灵感 / 核心直觉
"造题过程本身就是答案的脚手架"：既然 examiner 是从事实/答案反向构造问题，那条构造轨迹 c 天然包含解题所需的特权线索；用它做 teacher 的额外上下文，就能把稀疏的结果奖励转成稠密的逐 token 监督，而不引入任何外部数据。

## 5. 主要解决思路(一段话讲清核心)
三角色（examiner / teacher / student，均由同一 base LLM 初始化、均为带搜索工具的搜索 agent）交替优化：examiner 造出 (q,c,o)，teacher 条件于特权上下文 c、student 仅见 q；student 在结果奖励 + teacher 的逐 token reverse-KL 蒸馏联合作用下学习，形成 data-free 协同自演化闭环。

## 6. 方法详解(通俗、分步骤)
- **Examiner**：带搜索工具与搜索引擎交互获取事实，生成三元组 (q, c, o)；目标为造出多样且有难度的题（含难度奖励，并 −β·KL 约束；难度奖励随正确预测数线性衰减）。
- **Teacher**：理想目标 Eq.2 兼顾 "生成更准 rollout" 与 "不过度偏离 student"，但**实际实现并不直接优化该目标**，而是把 teacher 参数取为 student 的 **EMA 软更新**：ψ←(1−τ)ψ+τθ（Eq.11，soft-update 权重 **τ=0.05**，已核 Table 7），低成本提供稳定且随 student 缓慢演化的监督。
- **Student**：在结果奖励 I(ô=o)（GRPO）与教师指导（−per-token reverse-KL D(πS_θ(·|q) ‖ stopgrad[πT_ψ(·|q,c)])）联合下学习。
- **蒸馏权重衰减调度**：用衰减的 **λ**（distillation 系数）逐步弱化教师指导——Qwen3-4B/8B 取 0.03/0.003/0.002，Qwen3-4B-Instruct 取 0.1/0.03/0.03。〔修正：原分析在 §6/§8 误用 "β" 指代蒸馏衰减系数；论文中蒸馏权重为 λ，β 是 examiner 难度奖励/KL 系数——已统一改为 λ〕

## 7. 实验数据集
论文为 data-free 自演化，**不依赖任何外部训练数据**（无人工 demo/问题/答案）。〔已核 PDF §3.1〕
- 基座：Qwen3-4B、Qwen3-4B-Instruct、Qwen3-8B。
- 评测：3 个 one-hop/General QA（NQ、TriviaQA、PopQA）+ 4 个 multi-hop QA（HotpotQA、2WikiMQA、MuSiQue、Bamboogle）；统一 exact-match，检索器 E5-base，语料英文 Wikipedia dump。
- 基线：training-free（ReAct）、监督 RL（Search-R1、ToolForge）、自博弈（Dr.Zero、SQLM*）。

## 8. 实验结果与主要发现
- 流程（Algorithm 1）：每轮先训 examiner 50 步 → 生成学生训练数据 → 训 student 50 步（GRPO + teacher reverse-KL），每个 student step 后 teacher 做 EMA 软更新；共 3 轮 = **150 步**（远少于 Search-R1 等基线）。examiner 默认造 1/2/3/4-hop 比例 4:3:2:1。
- 主结论：data-free 的 π-Play 超过全监督搜索 agent——平均较 Search-R1 **+6.3% / +4.2% / +15.4%**（Qwen3-4B / 4B-Instruct / 8B）；演化效率较传统自博弈提升 **2–3×**（首轮即可媲美 Dr.Zero 三轮收敛值）。〔已核行 838、43/251、1057-1058〕
- 消融：QCP 优于其他特权信息形式（含 ground-truth）；衰减 λ 调度优于固定 λ。

## 9. 结果如何支撑其主张
"QCP=有用特权信息" 由 Table 3 消融支撑（QCP > Partial QCP > 无/替代特权信息）；"高效率" 由首轮即追平 Dr.Zero 三轮收敛、150 步胜全监督基线两点支撑。主张与证据方向一致。

## 10. 逻辑自洽性(中性评估)
内在逻辑自洽：把被忽视的造题副产物变成监督信号是清晰且新颖的因果链。但需注意 teacher 实际只是 student 的 EMA，"教师" 的额外能力完全来自条件于 c 的特权上下文而非独立训练，这与 §2.4 "理想 teacher 目标" 之间存在 "理想 vs 实现" 落差，论文已坦承。

## 11. 残留问题 / 局限
- **代码未发布**：框架/实现细节（搜索工具栈、reverse-KL 具体实现、examiner 难度奖励工程）无法核验，复现性受限。〔待核：完整训练代码待官方释出〕
- 评测局限于英文 Wikipedia QA（one/multi-hop）+ exact-match，未覆盖更开放的搜索任务；检索器/语料固定。
- "2–3× 效率" 的对比依赖与 Dr.Zero 等的同设定可比性。
- EMA teacher 是否在更大规模/更长训练下仍稳定优于直接优化 Eq.2，未做对照。

## 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库：https://github.com/zhyaoch/pi-play（本地已克隆约 1.5MB，**仅含 README 与 images/，代码尚未发布**；TODO：paper 已发布、code 未发布）。
- 框架：论文未在正文绑定特定开源训练框架；student 用 GRPO，teacher 为 EMA 软更新，三角色交替优化。〔待核：框架实现细节待 code 发布确认〕
- 代码可得性：paper-only（"coming soon"）。

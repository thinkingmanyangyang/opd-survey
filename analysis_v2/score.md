score | From Correction to Mastery: Reinforced Distillation of LLM Agents (SCoRe) | 中科大 USTC(Yuanjie Lyu、Tong Xu 通讯)+ Independent Researcher(Chengyu Wang 通讯、Jun Huang,原阿里 EasyDistill 团队) | 2025-09 arXiv 2509.14257(v2 2025-10-09)·预印本 | 主题线 L4(Agent/工具多轮自进化)+L1(蒸馏)·相关性 高

**原始论文**:https://arxiv.org/abs/2509.14257

## 一眼看懂

> 一句话导读:让 7B 小学生整条照抄 72B 大老师的解题轨迹是行不通的——任何一步走错都会把它推到没见过的状态、误差越滚越大。SCoRe 反过来让学生自己主导、老师只在"最早出错那一步"插手纠一下,学生从纠正后接着走;先在这种轨迹上 SFT、再从"已验证前缀"起做短程 RL。

- 🟦 TL;DR:小学生 agent(7B)做不动大 teacher(72B)整条轨迹的模仿——任何一步走错就会被推到 OOD(分布外),误差按 \(O(H^2)\) 滚雪球(\(H\) 是轨迹步数)。
  - SCoRe 反过来:让**学生自己主导**生成轨迹,teacher **只纠"最早出错的那一步" \(\sigma_k\)**,然后学生从被纠正后的前缀 \((\sigma_1,\dots,\sigma_{k-1},\sigma'_k)\) 继续往下走。
  - 训练分两步:先在这种"学生中心、最小干预"的纠错轨迹上做 SFT;再做**短 horizon(短程)RL**——从"已验证前缀"起 rollout,而不是从题目开头起——并给**关键步稠密奖励**。
  - 结果:7B 学生在 8 项数学 + 事实推理上 Avg = 50.8,只比 72B teacher(51.7)低 0.9。
- 最巧的一步:**"teacher 只纠最早错误 + 学生从该前缀续写"**。
  - 抽掉它,就退回普通行为克隆(即 teacher-acts / student-clones,老师演、学生抄),误差链又恢复成 \(O(H^2)\)(Thm 3.1);而且后面的短 horizon RL 也失去了"已验证前缀"这个起跑点(等于整个 RL 创新失去支点)。
  - 这一步同时实现了论文标榜的两个性质:Capability Matching(轨迹复杂度匹配学生能力)+ Deficiency Localization(用前缀 + 关键步精确暴露弱点)。【原文】§1、§3.2、Fig.1

## 为什么做

> 一句话导读:把大模型蒸到小模型,最直接的办法是让小学生整条照抄老师轨迹(行为克隆 BC),但有两道天生的差距(推理能力、知识储备)填不平,更要命的是误差会沿 horizon 按 \(O(H^2)\) 复合放大。SCoRe 的思路是把监督和探索都锚在"学生真实能力边界(最早出错点)"上。

- **研究背景的来龙去脉**:LLM agent 靠 ReAct 式的"推理-动作-观测"循环 + 外部工具(代码解释器 / 搜索)来解复杂任务——一条轨迹是 \(\tau=(t_1,c_1,o_1,\dots,t_H,c_H,o_H)\),每一步 \((t_i,c_i)\sim\pi(\cdot\mid s_i)\) 先生成 thought \(t_i\)、再生成 ReAct action \(c_i\),执行 \(c_i\) 得到 observation \(o_i\)。
  - 但高性能的 agent 需要 GPT-4 / Qwen-72B 这种超大 backbone,延迟与成本都高(一个复杂任务要几十次模型调用)。
  - Agent Distillation(Kang 2025)把 teacher 的行为拆成 [Thought, Action, Observation] 让小学生模仿。本文建在 **CodeAct**(Wang 2024,即 action = 可执行代码)之上;选 CodeAct 的理由有三:(i) 确定性操作、可复现;(ii) 师生在预训练时都熟悉代码,能降低能力差;(iii) 图灵完备、统一了动作空间。【原文】§1、§2、§3.1
- **解决的具体痛点("teacher-acts, student-clones"全轨迹模仿的两个 gap + 误差链)**:
  - (1) **Reasoning Ability Gap**:小模型复现不了 teacher 的逻辑分解;
  - (2) **Knowledge Capability Gap**:即便照搬了计划,也可能因为知识不足而执行不了复杂动作。
  - 这两个 gap 都源于 emergent abilities(涌现能力),无法完全迁移。
  - 更致命的是误差链:BC 目标 \(\mathcal{L}_{\mathrm{BC}}(\theta)=-\mathbb{E}_{\tau\sim D_T}\big[\sum_{i=1}^{|\tau|}\log\hat\pi(a_i\mid s_i;\theta)\big]\)(Eq.1,其中 \(a_i=(t_i,c_i)\))在协变量偏移(covariate shift)下,任意一步出错就会把学生推入 OOD,误差随 horizon 按 \(O(H^2\varepsilon)\) 复合(Ross 2011)。【原文】§1、§2、Thm 3.1
- **并行技术路线 + 各自具体短板**(§5):
  - **DAgger / HG-DAgger**(Ross 2011、Kelly 2019):能缓解 exposure bias(暴露偏差),但仍是 **teacher-led**(teacher 在学生分布上全程标注),且轨迹复杂度常与学生能力**错配**。
  - **Toolformer 式模仿**(Schick 2023、Gao 2023):能迁移规划 / 工具技能,但仍属 **BC**,继承了 reasoning / knowledge gap 和 \(O(H^2)\) 误差链。
  - **agentic RL**(GRPO Shao 2024 / ARPO Dong 2025b):能探索,但 long-context 的 rollout 不稳(Schulman 2017)、稀疏 / 延迟奖励难做信用分配(Andrychowicz 2017)。
  - 这些短板促成了"把 trajectory-level planning 与 local、可验证的 corrections 结合"的稳定细粒度方案。【原文】§5
- **动机链**:大 backbone 贵 → 蒸到小模型 → 但全轨迹 BC 有双 gap + \(O(H^2)\) 误差链 → 改成"学生主导、teacher 只纠最早错",把分布偏移限制在单步、误差降到 \(O(H)\) → 再加短 horizon RL(从已验证前缀起跑、给关键步稠密奖励),把"模仿"推向"自主解题"。
- **与最近邻的精确差异**:
  - vs DAgger——SCoRe 不是让 teacher 在学生分布上全程标注,而是**只纠最早一步** \(\sigma_k\to\sigma'_k\)、且由学生续写;
  - vs GRPO / ARPO(全 horizon RL)——SCoRe 的 RL **从最早错误之前的前缀 \((\sigma_1,\dots,\sigma_{k-1})\) 起 rollout**(缩短 horizon、降方差),并用关键步奖励而非纯结果奖励。
  - 关键差别就是"把监督 / 探索都锚在学生真实的能力边界(即最早出错点)"。【原文】§3.2-3.3、§5

## 怎么做 + 靠不靠谱

> 一句话导读:三阶段——先用 20% teacher 轨迹做冷启 SFT(故意只用一点、让学生会走多步循环但保持弱,当探索器);再让学生独立解剩下 80% 题、老师只纠最早错,产出两类监督(纠错轨迹 + 关键步偏好对);最后做 GRPO,两个创新是"从已验证前缀起 rollout(缩短 horizon)"和"关键步三档稠密奖励"。要冷静看:三个"理论"其实是经典结论的复述,"小成本"只指部署期、蒸馏期仍需 72B 老师,且 RL 用的 verifier(7B)与评测 judge(72B)不同、存在口径不一致风险。

- **方法流水线(三阶段,输入→输出)**:
  1. **Cold-Start BC(冷启动行为克隆)**(§3.1):用 teacher(Qwen2.5-72B)按"先给 high-level plan(`<first thought>`)、再走 Thought-Code-Observation 循环"的双 prompt 生成轨迹 \(D_T\),用 rejection sampling 只留最终答案正确的;然后在其中 **20%** 的数据上做 SFT,得到 \(\hat\pi_{\mathrm{init}}\)。
     - 这里故意只用 20%,是为了让学生先学会走多步循环、但能力上保持弱,好充当一个 explorer(探索器)。
     - **输出**:一个会走 T-C-O 循环、且不会立刻崩的 \(\hat\pi_{\mathrm{init}}\)。
  2. **Mentored Problem-Solving(MPS,有指导的解题)+ SCoRe-SFT**(§3.2):剩下的 **80%** 题,让 \(\hat\pi_{\mathrm{init}}\) 独立生成完整轨迹 \(\tau_S=(\sigma_1,\dots,\sigma_H)\) → teacher \(\pi_E\) 查正误,若错则定位**最早偏离步 \(\sigma_k\)** 并替换为 \(\sigma'_k\)、学生从 \((\sigma_1,\dots,\sigma_{k-1},\sigma'_k)\) 续写;如果在更靠后的 \(m>k\) 处又错,就再纠一个 \(\sigma'_m\)(本文上限 5 次);**最后用任务成功来隐式验证**这些修正是对的,留作训练数据。
     - 产出两类互补监督:
       - ① 最终的纠错轨迹(大部分由学生生成、teacher 只做稀疏编辑)→ 用作 capability-aligned 的 SFT 示范;
       - ② 每个关键步的纠正构成一个 preference pair(在同前缀 \((\sigma_1,\dots,\sigma_{k-1})\) 上 \(\sigma'_k\succ\sigma_k\))→ 供 RL 使用。
     - 把这些"最小干预"纠错轨迹的**一半**拿去 SFT、得到 SCoRe-SFT 模型,另一半留给 RL。
     - **输出**:能力匹配 + 弱点定位的轨迹数据 + SCoRe-SFT 模型。
  3. **SCoRe-RL(GRPO)的两个创新**(§3.3):
     - (a) **Short-Horizon Rollout**——从已验证前缀 \((\sigma_1,\dots,\sigma_{k-1})\) 起 rollout,把 horizon 从 \(H\) 缩到 \(H'=H-(k-1)\),以降方差;
     - (b) **Key-Step Reward**(关键步奖励)——
       \(\displaystyle R=\begin{cases}R_{\mathrm{final}},& \text{最终答案正确}\\[2pt]R_{\mathrm{key}},& a_k=a^{\pi_E}_k\ (\text{复现 teacher 纠正})\\[2pt]R_{\mathrm{avoid}},& a_k\ne a^{\mathrm{orig}}_k\ \wedge\ a_k\ne a^{\pi_E}_k\ (\text{避开原错})\\[2pt]0,& \text{否则(等于原错)}\end{cases}\)
       实现取 \(R_{\mathrm{key}}=0.5,\ R_{\mathrm{avoid}}=0.1,\ R_{\mathrm{final}}=1\)(Fig.2c);动作是否等价,由代码 + 执行结果可靠判定;奖励的正误,由一个**轻量的 Qwen2.5-7B-Instruct verifier** 判语义等价(注意:它与评测时用的 72B judge **不是同一个**)。
     - **输出**:SCoRe-RL 模型。
- **逐组件必要性(Table 3 消融,Qwen2.5-7B,8 项 Avg)**:
  - Initial Distillation(只用 BC 的 20% 数据)= 41.7 —— 故意弱,当 MPS 的 explorer(对比 Table 1 里用全量 teacher 数据的 BC baseline = 42.5)。
  - + MPS / SCoRe-SFT = 45.8(较 Initial +4.1)→ 证明 MPS 数据有效地强化了弱链。
  - **去掉 short-horizon rollout** → 48.4(满配是 50.8,掉了 2.4)→ 证明缩短 horizon 降方差有用。
  - **去掉 key-step rewards** → 49.7(掉了 1.1)→ 证明关键步稠密奖励有用。
  - 满配 SCoRe-RL = 50.8。两个组件都做了消融、缺一就掉点,必要性成立。【原文】Table 3、§4.3
  - 另:Fig.3a 给出了 teacher 的干预频次——大多数题只需**单次**纠正(Math:71% 0 次 / 15.3% 1 次 / 2.7% ≥2 次;Search:58.8% / 25.3% / 2.2%),其中 hard data(teacher 也教不会的)占比很小——证明"单步纠正通常就够用"。【原文】§4.3、Fig.3a
- **关键机制 / 公式(三个定理,都是直觉化论证、非新数学)**:设有限 horizon \(H\)、每步代价 \(c_t(s)\in[0,1]\)、总期望代价 \(c(\pi)=\mathbb{E}[\sum_{t=1}^{H}c_t(s_t)]\)。
  - **Thm 3.1(BC 误差界)**:若在 teacher 的状态分布 \(d^{\pi_E}_t\) 上有 \(\mathbb{P}[\hat\pi(s)\ne\pi_E(s)]\le\varepsilon\),则 \(c(\hat\pi)\le c(\pi_E)+\frac{H(H-1)}{2}\varepsilon=c(\pi_E)+O(H^2\varepsilon)\)(经典的 covariate-shift 结论,Ross 2011)。
  - **Thm 3.2(SCoRe 首错纠正界)**:因为 SCoRe 的训练数据来自**学生自身的分布** \(d^{\hat\pi}_t\)(在其上 \(\mathbb{P}[\hat\pi(s)\ne\pi_E(s)]\le\varepsilon\)),且"最多一个未纠正的错误就会回到 expert 路径",误差界就收紧到 \(c(\hat\pi)\le c(\pi_E)+H\varepsilon=c(\pi_E)+O(H\varepsilon)\)。
    - 证明思路(附录 A):BC 下 \(d^{\hat\pi}_t=(1-p_{t-1})d^{\pi_E}_t+p_{t-1}q_t\),其中 \(p_{t-1}\le(t-1)\varepsilon\),逐步求和得 \(\sum_t(t-1)\varepsilon=\frac{(H-1)H}{2}\varepsilon\);而 SCoRe 截断了误差传播,所以只剩线性项。
  - **Thm 3.3(短 rollout 方差界)**:策略梯度 \(g_k=\sum_{t=k}^{H}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\cdot G_t\),其中 \(G_t=\sum_{t'=t}^{H}\gamma^{t'-t}r_{t'}\);设 \(|r_t|\le R_{\max}\)、\(\|\nabla_\theta\log\pi_\theta\|\le G_{\max}\),则 \(\mathrm{Var}[g_k]\le\frac{C}{(1-\gamma)^2}\big((H-k+1)-\gamma\frac{1-\gamma^{H-k+1}}{1-\gamma}\big)^2\)(其中 \(C=G_{\max}^2R_{\max}^2\)),这个界随 \(k\) 增大而**单调下降**——也就是说,从更靠后的 verified prefix 起 rollout,方差更小。【原文】Thm 3.1-3.3、附录 A、Eq.3-9
- **训推数据如何流动**:种子 QA(35k,主要取自 Tool-Star:NuminaMath / Omni-Math + HotpotQA / 2Wiki / WebWalker)→ 其中 20% 由 teacher 全标注、做 BC 冷启(用 LLaMA-Factory SFT)→ 另外 80% 经 MPS(用 LangGraph 编排 agent 轨迹 + teacher 单步纠错)产出纠错轨迹 → 这批轨迹一半给 SCoRe-SFT、一半给 SCoRe-RL(用 veRL,max rollout 步数 = 8,从 verified prefix 起 rollout,由 7B verifier 判奖励)。推理用 LLaMA-Factory 的 api(vllm / sglang),teacher 经 Qwen2.5-72B API 或本地部署。【原文】§4.1 Implementation
- **实验与证据**:
  - **数据集**:12 个 benchmark,分三类——数学(AIME24/25、MATH500、OlympiadMath)、事实多跳 QA(HotpotQA、2Wiki、MuSiQue、Bamboogle,用 token-F1)、深度搜索(GAIA、WebWalker、HLE、xBench,用 WebThinker 的 text-only split)。
  - **关键数字**:
    - Qwen2.5-7B SCoRe-RL = **50.8**(对比 BC = 42.5、GRPO = 48.4、ARPO = 49.3),仅比 72B teacher(51.7)低 0.9;Qwen2.5-3B = 46.7(+8.4 over BC)、Llama3.1-8B = 47.5(+10.2 over BC);
    - 深度搜索(Qwen3-8B,Table 2):SCoRe-RL Avg = **30.5**(+7.7 over BC、+8.3 over GRPO、超 TIR-72B teacher +3.2),GAIA-Avg 从 27.2(BC)→40.8;
    - Fig.3b:数学 hard data(200 题,已从训练里排除)的正确率 0% →(SFT)17.3% →(RL)24.3%。【原文】Table 1/2、Fig.3
  - **baseline 公平吗**:GRPO / ARPO 的数值多取自 ARPO 原文(Dong 2025b),评测集的组织也 follow ARPO,口径基本对齐;而 BC baseline 用的是**全量** teacher 数据,SCoRe 的 cold-start BC 却只用 20%——这对 SCoRe 反而更苛刻(脚注 1),所以较公平。
  - **看着强但没回答核心问题?**:"逼近 teacher"是成立的(7B 仅低 0.9);但"教会 student 反超那些它本来教不了的题"(Fig.3b 的 hard data 24.3%)虽然亮眼,绝对值仍偏低;另外 RL 的 verifier(7B)和评测的 judge(72B)不是同一个,存在训练-评测口径不一致 / reward hacking 的潜在风险,论文没做交叉验证。【推断】
- **假设与失效边界**:
  - 【原文】"用最终任务成功来隐式验证 teacher 的修正是对的"——这是**弱验证**:任务成功 ≠ 每一步修正都对,可能引入噪声标签,论文没量化误纠率。
  - 【原文】Thm 3.2 假设"每步纠正后就回到 expert-aligned 路径、且至多一个未纠正的错误",但现实中 teacher 定位"最早错误"这件事本身就可能判错。
  - 【推断】方法依赖 teacher 是 72B 级的强模型 + 用可执行代码做确定性的动作等价判定(CodeAct);一旦换到无法用代码精确判等价的任务域,key-step reward 的可靠性就会下降。
- **祛魅总结**:
  - 真贡献 = **"最早错误纠正 + 从已验证前缀做短 horizon RL"这一条完整 pipeline**,它把 BC 的误差链从 \(O(H^2)\) 降到 \(O(H)\)、并稳定了 RL,消融扎实、跨 3 个 student 一致。
  - 包装 / 高估处:
    - (1) 论文 prose §4.2 写的 7B "+6.3 over GRPO" 与它自己的 Table 1(48.4 vs 50.8 = **+2.4**)矛盾,应以表为准;
    - (2) "小成本"只指部署期的 student,蒸馏期仍需要 72B teacher + 多轮 rollout,成本并不低,容易被低估;
    - (3) 三个"理论 justification"其实是经典结论(Ross 2011 的 covariate shift + 策略梯度方差)的复述,不是新理论。【推断】

## 结构化抽取
- 🎯 机制速览6轴:
  - **学什么信号** = teacher 对"最早错误步 \(\sigma_k\)"的纠正(用于 SFT)+ 关键步是否复现该纠正 / 避开原错的稠密奖励(\(R_{\mathrm{key}}=0.5/R_{\mathrm{avoid}}=0.1\))+ 最终答案正确性(\(R_{\mathrm{final}}=1\),用于 RL);
  - **改什么** = student 参数(SFT 全参 + GRPO 策略更新);
  - **何时改** = 三阶段离线串行(BC → MPS-SFT → 短 horizon RL);
  - **免梯度?** = 否,SFT + RL 都更新参数;
  - **记忆-技能生命周期** = 无显式记忆 / 技能库,知识固化进参数(轨迹 → SFT → RL);
  - **防遗忘机制** = 无专门的防遗忘设计,靠"前 20% BC 冷启动 + 能力匹配数据"来隐式维持基础技能。【原文】§3
- ⑦ 开源代码+框架/harness:https://github.com/modelscope/easydistill(SCoRe 在 `projects/SCoRe/`,整库 clone 约 221MB)。框架是**一套组合工具链、而非单一框架**:
  - Cold-start / SCoRe-SFT 用 **LLaMA-Factory**;
  - agent 轨迹生成用 **LangGraph**(`MPS/graph/{graph.py,graph_repair.py}`);
  - SCoRe-RL 用 **veRL**(`verl/tools/search_tool.py`);
  - 推理用 LLaMA-Factory 的 api(vllm / sglang);teacher 经 Qwen2.5-72B API 或本地部署。
  - 代码可得性 Tier A。【原文】GitHub + 既有 analysis 仓库核查
- 💰 资源/成本与可扩展性:RL 最大 rollout 步数=8;训练用 Qwen2.5-72B 作 teacher 标注 + 生成 35k 题轨迹,成本主要在蒸馏期(原文未给 GPU·小时);部署期仅 7B student。完整超参(组大小/lr/KL)未在正文给出〔待核:附录 B〕。【原文】Implementation
- 🎯 对"探索-巩固"对标:**强支撑 + 高度同构**。
  - SCoRe 的"学生主导探索 + teacher 只纠最早错 \(\sigma_k\) + 从被纠正后的前缀 \((\sigma_1,\dots,\sigma_{k-1},\sigma'_k)\) 续写",正是本项目 idea 里的"teacher 当稀疏脚手架,教 path-selection(偏向自己走得通的开头)+ path-recovery(走偏后从恢复分支续写并固化)";on-policy 自选体现在"学生先独立生成完整轨迹";巩固 = 纠错轨迹 SFT + 关键步 RL 固化进参数。
  - **可借组件(3 件)**:
    - (a) "最早错误定位 + 单点接管",可作 path-recovery 的离散实现;
    - (b) **short-horizon rollout from verified prefix**(Thm 3.3 证明了方差随起点右移而单调下降)——可直接移植到 OPD 的 RL 阶段降方差,且给出了"从哪起 rollout"的理论依据;
    - (c) **key-step reward 的三档结构**(复现纠正 0.5 / 避错 0.1 / 等于原错 0),是离散版的"接管点稠密信用"。
  - **缺口 / 差异**:SCoRe 用离散的"动作等价"来判关键步,没有 MTP 式的前瞻信号,也没有 token 级的稠密信用;而且纠正点是由 teacher 判定的、不是学生自选恢复分支(而 idea 强调"学生自选恢复"),这一点有偏差。
  - 一句判定:**它是最贴近 idea 的 agent 落地之一,可当 path-selection / recovery 的离散基线,但缺 MTP 前瞻与学生自主选恢复点**。【推断,依据 §3.2/Fig.1 + Thm 3.3 vs idea】
- 🔭 开放问题/未来方向:【原文·§6】改进 reward 设计、扩展到多模态任务。【推断】(1) 量化 teacher 误纠率、用更强 verifier 或多数投票降噪声标签;(2) 把"最早错误"的离散纠正换成 token/段级稠密信用(可接 MTP 前瞻或 IG 归因)以减少对 teacher 判断的依赖;(3) 让学生自主选择"从哪个分支恢复"而非 teacher 指定,更贴近真正的 path-recovery;(4) RL verifier 与评测 judge 统一以排除口径不一致风险。

RETURN:score | 读到PDF? 是(29页全文,Eq.1-9 + Thm 3.1-3.3 + 附录 A 证明 + Table 1/2/3 + Fig.3 全核) | L4(+L1) | 对标=强支撑/高度同构,可作 path-selection+recovery 离散基线 + Thm 3.3 给"从哪起 rollout"理论依据 + key-step 三档 reward,缺 MTP 前瞻与学生自选恢复点 | 残留待核 1(附录 B 完整 RL 超参:组大小/lr/KL)

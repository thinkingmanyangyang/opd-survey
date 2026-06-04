ca_survey | From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models | Chenchen Zhang（独立研究者,单作者 survey）| 2026-04（arXiv:2604.09459v2，cs.CL）| 主题线 L3 RLVR/信用分配（综述）·相关性 中

**原始论文**：https://arxiv.org/abs/2604.09459

> 注:本篇是 **survey + awesome-list**,非新方法论文。以下"怎么做/逐组件"栏改写为其 **taxonomy（分类法）与覆盖范围**。

## 一眼看懂
> 一句话导读：这是一篇综述，用"信用分配"这一把尺子重审 LLM RL——所谓信用分配，就是把一条轨迹最后那个稀疏的对错奖励，摊回到具体哪个 token/步/轮去；它把 47 篇方法按"在哪个粒度分配 × 用什么方法算"排成网格。

- 🟦 TL;DR：本文以"**信用分配(credit assignment, CA)**"为中心透镜重审 LLM RL。这里 CA 指的是——把稀疏的 outcome 奖励（只有最终对错一个分数）分配到具体的 token / segment / step / turn / agent 上。
  - 核心产物：把 2024 到 2026 初的 **47 篇方法（41 篇 core + 6 篇 adjacent enabler）**按**粒度 × 方法学的二维分类**整理（Fig.2/3），并配上机器可读清单、reporting checklist、benchmark 协议，以及一棵方法选择决策树。
  - 主要判断：reasoning 的 CA 正在成熟化（收敛到 PRM + critic-free 的组内比较）；agentic 的 CA 则催生出真正的新方法（hindsight counterfactual、privileged asymmetric critic、turn-level MDP）。
- 最巧的一步（综述层面）：**"PRM 本质上就是 CA"这一概念澄清（§2.5 boxed）**。它把 PRM 文献（Math-Shepherd/OmegaPRM/PURE）和 CA 文献（VinePPO/SPRO/SCAR）统一成"同一问题的两个视角"。抽掉这个统一视角，整篇就退化成方法罗列；正是它让"reasoning CA 已成熟"这个判断站得住——因为 PRM 密集就等于 step 级 CA 密集。

## 为什么做
> 一句话导读：LLM RL 从单条 CoT 的 reasoning 走到了多轮交互的 agentic，轨迹越来越长；而 GRPO 这类"整条轨迹给每个 token 同一个优势"的做法，在长轨迹上会把好动作和差动作一视同仁、信噪比崩掉——所以需要一篇专讲信用分配、且横跨这两类场景的综述。

- 研究背景：LLM RL 经历两波。
  - **reasoning RL**（如 DeepSeek-R1/o1）：单条 CoT，长度 \(L\sim10^3\)–\(3\times10^4\) token，1 个 turn，纯 outcome 奖励。
  - **agentic RL**：多轮环境交互，\(T\sim10\)–\(100+\) turn，长度 \(L\sim10^5\)–\(10^6\) token，奖励是稀疏的 terminal 奖励。
  - 经典 RL 早有一套 CA 工具箱——TD/GAE、return decomposition(RUDDER)、hindsight(HCA)、counterfactual/difference reward，这些是 LLM 方法的基础（§2.5）。
- 解决的具体痛点（即这篇综述要填的 gap），三条：
  - ① **Episode 级方法（GRPO/REINFORCE）给每个 token 同一个 advantage**，轨迹一长就失效。先看动作的多粒度层级(§2.2 Eq.1)：\(\tau=[\text{Turn}_1,\dots,\text{Turn}_T]=[\text{Seg}_{1,1},\dots]=[a_{1,1,1},\dots]\)，即 Episode ⊃ Turn ⊃ Segment ⊃ Token。GRPO 的 advantage（§2.3 Eq.2）为：
    \(\displaystyle \hat A^{\text{GRPO}}_i=R(\tau_i)-\frac1G\sum_{j=1}^G R(\tau_j),\)
    它给整条轨迹里每个 token 同一个 \(\hat A_i\)。在 reasoning RL（\(L\sim10^3\)、1 turn）上还凑合，因为关键决策少；但在 agentic RL（\(L\sim10^5\)、100+ turn）上，它会把"选对 API"和"格式化输出"赋予同样的信用，于是信噪比崩掉。
    - 形式化看：REINFORCE 的梯度方差正比于 \((R(\tau)-b)^2\)；当同一 baseline 用于全部 \(T\) 个动作时，总方差是 \(O(T\cdot\mathrm{Var}[R])\)，于是 **\(T=100\) 时每个动作的信噪比约差 100×**。
    - 后果是所谓 "echo trap"（Wang 2025d：agentic 模型在 episode 级信用下会收敛到重复行为）。
  - ② **agentic 让 CA 变成双层级问题**：先判断哪个 turn 关键，再判断 turn 内哪些 token 重要；同时还叠加了随机转移、部分可观测、超长 horizon 等难点。
  - ③ **现有相邻综述都不够**：Pignatelli 2023 只覆盖经典 RL；Zhang 2025a 那篇 100 页的 agentic 综述只把 CA 当一个子话题。结果是——**还没人系统覆盖 reasoning + agentic 两个 regime 的 CA**。
- 相关工作 & 各自不足：就是上面第③条——现有工作要么只讲经典 RL，要么把 CA 当附属，**缺一篇专门、且跨这两个 regime 的 CA 综述**。
- 动机链（一步步推下来）：现状是 LLM RL 从 reasoning 演进到 agentic，CA 成了核心难题；缺陷是既没有统一 CA 透镜的综述，episode 级方法又在长 horizon 上失效；所以做一篇专门的 CA 综述，外加一批可复用产物（清单 / checklist / benchmark 协议 / 决策树）。
- 与最近邻工作的差异：
  - 相对 Pignatelli 2023（前 LLM 时代的 CA）：补上 LLM 时代。
  - 相对 Zhang 2025a（agentic 综述）：以 CA 为唯一透镜，并横跨 reasoning + agentic，且专门论证"为何 agentic 的 CA 比 reasoning 更难、催生了哪些新技术"。
  - 关键点：把所有方法（包括 PRM）都拉到"如何把稀疏奖励分配到动作"这同一个问题之下。

## 怎么做 + 靠不靠谱（= taxonomy 与覆盖）
> 一句话导读：这篇没有新方法，"怎么做"就是它的分类法本身——两根轴（在哪个粒度分、用什么方法算）搭成一张网格，再把 47 篇方法逐一填进格子，并把每种做法对应回经典 RL 的工具。

**【二维正交分类（§2.4，核心贡献，Fig.2 网格 / Fig.3 层级树）】** 覆盖 2024-01 到 2026-04 的 47 篇（41 core + 6 adjacent）。
- **粒度轴（在什么层级分配信用）**，从细到粗：
  - **Token**：生成内的单个 token。
  - **Segment**：一段语义跨度，如一个推理步。
  - **Step-Turn**：一个完整的 LLM 响应或一个工具调用周期。
  - **Multi-Agent**：在协作的多个 agent 之间分解。
- **方法学轴（如何算信用）**：
  - **Monte Carlo (MC)**：从中间状态做 rollout。
  - **Temporal Difference + value**：学一个值函数并 bootstrap，如 GAE。
  - **Model-based / LLM-as-Critic**：让 LLM 自身对中间状态给出自然语言评价。**论文强调这一轴没有经典 RL 的对应，是 LLM-native 的新轴**。
  - **Game-theoretic**：如 Shapley 值、counterfactual baseline。
  - **Information-theoretic**：如信息增益、熵。

**【方法填格（Fig.2/3，真实条目）】** 下面把方法按"场景 → 子类（括号内是其方法学/特点）"列出：
- **Reasoning RL**：
  - Token 级——VinePPO(MC)、RED(redistribution)、T-REG(self-gen)、From r to Q*(implicit)。
  - Segment 级——SPO(MC)、SCAR(Shapley)、TEMPO(tree-TD)。
  - Step 级——PURE(PRM/min-form)、SPRO(masked adv)、CAPO(LLM-critic)、HICRA(hierarchy，区分 planning 与 procedural)、PRL(entropy)、InT(intervention)、FinePO(fine PRM)、ACPO(attribution)。
- **Agentic RL**：
  - Turn-level PRM——AgentPRM(TD+GAE)、SWEET-RL(privileged asymmetric critic)、Turn-PPO(turn MDP)、SORL(bias-corr)、TARL(LLM-judge)、ITPO(implicit)。
  - Hindsight/Counterfactual——HCAPO(hindsight + generative verification)、C3(LOO)、CCPO(causal)、CriticSearch(retrospective)。
  - Critic-Free/Step——GiGPO(group-in-group)、POAD(action decomp)、CARL(entropy)、iStar(implicit DPO)、IGPO(info-theoretic)。
  - Hierarchical——ArCHer(TD hierarchy / off-policy critic)、PilotRL(progressive)、StepAgent(IRL)。
- **Multi-Agent**：M-GRPO(hierarchical)、SHARP(Shapley)、MAPPA(per-action PRM)、Dr.MAS(agent-wise adv)、LLM-MCA(LLM-critic)、QLLM(LLM-gen)。
- **Adjacent enablers（6 篇，#36–41）**：SPA-RL、Agent Lightning、RAGEN、SCRIBE、PRS、AdaptSeg。

**【经典→LLM 映射（§2.5）】** 每条都是"某个经典 RL 工具 → 它在 LLM 时代的对应方法"：
- **TD/GAE**：\(\hat A^{\text{GAE}(\gamma,\lambda)}_t=\sum_{l=0}^\infty(\gamma\lambda)^l\delta_{t+l},\ \delta_t=r_t+\gamma V(s_{t+1})-V(s_t)\)，对应到 learned critics（ArCHer、AgentPRM）。
- **return decomposition（RUDDER）**：\(c_t=\hat R(s_{0:t})-\hat R(s_{0:t-1})\)，对应到 reward redistribution（RED 做 token 级、SPA-RL 用 MLP 估进度、IGPO 用信息增益）。
- **hindsight（HCA）**：用未来的结果回头重估过去的动作，对应到 retrospective 类（HCAPO，基于 generative verification）。
- **counterfactual / difference reward**：拿动作与默认动作对比，对应到两支——leave-one-out（C3、CCPO 的 turn 级 LOO）与 Shapley（SCAR）。
- **唯一没有经典对应的是 LLM-as-Critic**：即让 LLM 给中间状态自然语言评价（Xie 2025 / Zhou 2025 / Qu 2025）。
- **PRM=CA（§2.5 boxed）**：PRM 给每一步打一个分 \(r_i\)，本质就是对 terminal 奖励 \(R(\tau)\) 做 step 级分解。所以 PRM 文献(Math-Shepherd/OmegaPRM/PURE) 与 CA 文献(VinePPO/SPRO/SCAR) 是同一问题的两个视角。
- **RL 算法的 CA 谱（§2.6）**，从粗到细：
  - REINFORCE/GRPO = episode 级（最粗；GRPO 用 group baseline \(\hat A_i=R(\tau_i)-\frac1G\sum_j R(\tau_j)\)，免 critic）。
  - PPO = token 级，靠 learned critic（更细，但是近似，且值网难训）。
  - DPO = 隐式的 token 级（"From r to Q*" 证明 DPO 隐式学到了 token 级 Q 值；iStar/ITPO 据此抽 step 信用）。
  - RLOO/REINFORCE++ = 仍是 episode 级。

**【综合判断（覆盖范围/gaps，对应"实验与证据"）】**
- 两条主结论：
  - **reasoning CA 正收敛到 process reward model + critic-free 的组内比较**（即成熟化，evidence [SE]）。
  - **agentic CA 催生出真正的新方法**——hindsight counterfactual(HCAPO)、privileged asymmetric critic(SWEET-RL)、turn-level MDP 重构(Turn-PPO)，这些在 reasoning RL 里没有先例（[LS]）。
  - Fig.2 网格直观地显示方法的迁移趋势：从 reasoning（左上密集，token/segment 细粒度）走向 agentic（右下，step/turn 较粗但带环境感知）。其中 **step/turn 级的聚类最密**（这是 LLM agent 的自然粒度），而 **multi-agent 与超长 horizon 最薄弱**。
  - 五条 takeaway 都标了 evidence level：[SE] 强实证 / [LS] 有限但暗示 / [AS] 作者综合。
- **§7 systematic comparison**：按计算成本、是否需辅助模型、适用场景、实证性能，逐项比较各方法（含一个 GRPO-family 的 meta-comparison）。
- **三类可复用产物**：
  - ① **机器可读的 47 方法清单**：带 taxonomy 标签、baseline family（G/P/D/O/T）、evidence level（S/L/A）、主 benchmark；将以 CSV/JSON 发布（§B Table 10）。
  - ② **reporting checklist**（§C Table 11）：已用 HICRA/GiGPO/M-GRPO 三篇做验证，发现系统性 gap——0/41 报 GPU-hours、2/41 报方差/CI、0/41 有 compute-controlled baseline。
  - ③ **benchmark 协议规范** + **方法选择决策树**（Fig.4/Table 8，§9）：协议规范含任务族、元数据要求、以及 **controlled bifurcation tasks（受控分叉任务）**。
- **假设与失效边界（§9.4 作者自陈局限）**：
  - 【原文】**单作者**——所有筛选/分类/evidence 编码都由一人执行（已释出 screening log 供核验）。
  - 【原文】preprint 易变——这是 2026-04 的 snapshot，方法/结果/标题都可能变。
  - 【原文】cross-paper 的数字是非受控对比。
  - 【原文】core/adjacent 与 reasoning/agentic 之间的边界存在判断空间。
  - 【原文】multi-agent CA、超长 horizon、exploration 与 credit 的互作用，都还是开放前沿。
  - 【推断】部分被引方法极新且 evidence level 参差（2026 的 HCAPO/C3/CCPO 等，其中 3 篇 counterfactual 在 2026-03 同一周出），所以"成熟化"的判断对早期方法可能偏乐观。
  - 【推断】机器可读清单是"发表时发布"，截到 snapshot 未必齐全。
- **祛魅总结**：
  - 【推断】真贡献是 **"PRM=CA"的统一透镜 + 跨 reasoning/agentic 的二维分类网格 + 一批可复用产物（清单/checklist/决策树）**，对快速定位方法、选 baseline、找相邻工作很实用。
  - 要祛魅的几点：它本质是索引 + 分类 + 标准化提案，**没有新方法、没有新实验**；"reasoning CA 已成熟"这类判断是综述作者的归纳判断（单作者，对极新方法可能乐观，不过已标 evidence level 自陈不确定性）；机器可读清单截到 snapshot 未必齐全；仓库是 living list，已超出论文 snapshot。

## 结构化抽取
- 🎯 机制速览6轴（对综述而言，这里填的是它 taxonomy 所覆盖的机制谱）：
  - **学什么信号**=覆盖全谱——把 outcome 奖励再分配到 token/segment/step/turn/agent，手段包括 MC rollout、TD value、LLM-critic 评价、Shapley-counterfactual、信息增益-熵。
  - **改什么**=综述对象大多是"改 advantage/reward 的分配"（即 policy gradient 的信用），含 learned critic / PRM / group baseline。
  - **何时改**=覆盖 in-training（每步/每轮的 CA 计算）。
  - **免梯度?**=对综述本身不适用；其覆盖的方法多为梯度 RL，少数是 critic-free（组内比较）。
  - **记忆-技能生命周期**=不适用。但 §9.3 提到"CA 与 memory"——对存储/检索/更新 summary 这类记忆动作如何分信用是个开放问题，需要把 eligibility trace 的思想类比扩展到语义记忆。
  - **防遗忘机制**=不适用。
- ⑦ 开源代码+框架/harness：列表在 https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL 。
  - 旧 analysis 已核：旧 KEYS 里的 `xxzcc/Awesome-Credit-Assignment` 是 404，上面这个才是正确的仓库名；MIT 协议，是 living list，2026.05 又新增了 agentic/coding-agent/uncertainty-control/multi-agent 等论文。
  - 内容=`README.md`（分类表）+ `gen_figures.py`（生成 taxonomy 网格图与树图）+ `assets/`。
  - **框架=不适用**——它是论文清单 + 分类可视化脚本，不是训练代码。
  - 【待核】仓库细节沿用旧 analysis 的核查；机器可读的 CSV/JSON 清单论文称"发表时发布"，需留意是否已上传。
- 💰 资源/成本与可扩展性：原文未说明（综述本身无训练）。
  - §7 做了 systematic comparison：按计算成本、是否需辅助模型、适用场景、实证性能逐项比较各方法（含 GRPO-family meta-comparison）。
  - §9.2 提出"**computation-signal trade-off / CA efficiency frontier**"——在固定算力下，是选"更多 rollout + 粗的 episode 信用(GRPO)"还是"更少 rollout + 细的信用(VinePPO/HCAPO)"。作者猜测随轨迹变长应偏向细信用，但没给系统答案。
- 🎯 对"探索-巩固"对标（**索引/定位工具，属上位框架**）：
  - 判定：CA（把稀疏 outcome 奖励分配到 token/step/turn）正是 TSRD"path-selection / path-recovery 细粒度信用分配"的**上位框架**——TSRD 的"单点接管/在关键步给信号"本质上就是 step/segment 级 CA。
  - 可借一：用它的二维分类（粒度 × 方法学）给 TSRD 定位（大致属于 step/segment 级 + LLM-as-Critic 或 counterfactual 一类）。
  - 可借二：从它的 47 篇里挑 baseline（如 VinePPO/SPO/SCAR/PURE/HICRA/HCAPO/SWEET-RL/GiGPO）与相邻工作。
  - 可借三：§9 benchmark 协议里的 **controlled bifurcation tasks（受控分叉任务）**，思想很契合"选路/回轨"的评测设计。
  - 可借四：reporting checklist 可用来自查 TSRD 论文（是否报了 GPU-hours / 方差 / compute-controlled baseline）。
  - 可借五：§9.2 "CA 与 exploration 结合"——IGPO 用信息论来框信用，而信用最不确定处恰好就是最该探索处，这与 TSRD 选路高度相关。
  - **缺口**：它本身不涉及蒸馏/OPD/MTP/path-recovery 的具体机制，只是一张地图、不是方法。
- 🔭 开放问题/未来方向：
  - 【原文】(§9) 以下都是开放前沿——multi-agent credit（可扩展分解 / 通信信用 / 部分可观测）、超长 horizon CA、exploration 与 credit 的互作用、CA-memory、reasoning→agentic 的技术迁移（如把 VinePPO 的 vine 扩到 turn 边界、把 PURE 的 min-form 扩成 turn-PRM、把 HICRA 的 planning-procedural 扩到 agentic）、formal guarantees（多数方法无收敛保证，仅 VinePPO/PURE/CCPO 有部分）。文中给了一条 roadmap：Classical RL → Reasoning RL → Agentic RL → Future Multi-Agent。
  - 【推断】把 "controlled bifurcation tasks" 做成标准的 path-selection/recovery 评测。
  - 【推断】在 CA 框架下统一 PRM 与蒸馏的 token 级信号。
  - 【推断】把 MTP 前瞻当作一种新的 CA 信号——用未来 token 的可预测性给当前步分配信用，可纳入 information-theoretic 那一轴。

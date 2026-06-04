ca_survey | From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models | Chenchen Zhang（独立研究者,单作者 survey）| 2026-04（arXiv:2604.09459v2，cs.CL）| 主题线 L3 RLVR/信用分配（综述）·相关性 中

**原始论文**：https://arxiv.org/abs/2604.09459

> 注:本篇是 **survey + awesome-list**,非新方法论文。以下"怎么做/逐组件"栏改写为其 **taxonomy（分类法）与覆盖范围**。

## 一眼看懂
- 🟦 TL;DR：一篇以"**信用分配(credit assignment, CA)**"为中心透镜重审 LLM RL 的综述。CA = 把稀疏的 outcome 奖励分配到具体 token/segment/step/turn/agent。核心产物:把 2024–2026 初的 **47 篇方法（41 core + 6 adjacent enabler）**按**粒度 × 方法学二维分类**（Fig.2/3）,配机器可读清单、reporting checklist、benchmark 协议 + 方法选择决策树。判断:reasoning CA 正成熟化（收敛到 PRM + critic-free group comparison）,agentic CA 催生真正新方法（hindsight counterfactual、privileged asymmetric critic、turn-level MDP）。
- 最巧的一步（综述层面）：**"PRM 本质就是 CA"这一概念澄清（§2.5 boxed）**——把 PRM 文献（Math-Shepherd/OmegaPRM/PURE）和 CA 文献（VinePPO/SPRO/SCAR）统一成"同一问题的两个视角"。抽掉这个统一视角,整篇就只是方法罗列;正是它让"reasoning CA 已成熟"的判断站得住（PRM 密集=step 级 CA 密集）。

## 为什么做
- 研究背景：LLM RL 两波——**reasoning RL**（DeepSeek-R1/o1,单条 CoT \(L\sim10^3\)–\(3\times10^4\) token,1 turn,纯 outcome 奖励）→ **agentic RL**（多轮环境交互,\(T\sim10\)–\(100+\) turn,\(L\sim10^5\)–\(10^6\) token,稀疏 terminal 奖励）。经典 RL 已有 CA 工具箱（TD/GAE、return decomposition RUDDER、hindsight HCA、counterfactual/difference reward）,是 LLM 方法的基础（§2.5）。
- 解决的具体痛点（survey 要填的 gap）：
  - ① **Episode 级方法（GRPO/REINFORCE）给每 token 同一 advantage**,轨迹一长就失效。多粒度动作层级(§2.2 Eq.1):\(\tau=[\text{Turn}_1,\dots,\text{Turn}_T]=[\text{Seg}_{1,1},\dots]=[a_{1,1,1},\dots]\)（Episode⊃Turn⊃Segment⊃Token）。GRPO advantage（§2.3 Eq.2）:
    \(\displaystyle \hat A^{\text{GRPO}}_i=R(\tau_i)-\frac1G\sum_{j=1}^G R(\tau_j),\)
    每 token 同一 \(\hat A_i\)。reasoning RL（\(L\sim10^3\)，1 turn）尚可（关键决策少）;agentic RL（\(L\sim10^5\)，100+ turn）把"选对 API"和"格式化输出"赋同样信用 → 信噪比崩。形式化:REINFORCE 梯度方差 \(\propto(R(\tau)-b)^2\)，同一 baseline 用于全 \(T\) 动作时总方差 \(O(T\cdot\mathrm{Var}[R])\)，**\(T=100\) 时每动作信噪比约差 100×** → "echo trap"（Wang 2025d:agentic 模型在 episode 级信用下收敛到重复行为）。
  - ② **agentic 使 CA 双层级化**（先判哪个 turn 关键,再判 turn 内哪些 token 重要）+ 随机转移/部分可观测/超长 horizon。
  - ③ **现有相邻综述不够**（Pignatelli 2023 只覆盖经典 RL;Zhang 2025a agentic 100 页综述把 CA 当子话题）——**无人系统覆盖 reasoning+agentic 两个 regime 的 CA**。
- 相关工作 & 各自不足：见上③——要么只经典 RL,要么 CA 是附属,**缺专门、跨两 regime 的 CA 综述**。
- 动机链：现状（LLM RL 从 reasoning 演进到 agentic,CA 是核心难题）→ 缺陷（无统一 CA 透镜的综述 + episode 级方法在长 horizon 失效）→ 做专门 CA 综述 + 可复用产物（清单/checklist/benchmark 协议/决策树）。
- 与最近邻工作的Δ：vs Pignatelli 2023（前 LLM CA）——补 LLM 时代;vs Zhang 2025a（agentic 综述）——以 CA 为唯一透镜并跨 reasoning+agentic,且论证"为何 agentic 比 reasoning 的 CA 更难、催生哪些新技术"。关键点:把所有方法（含 PRM）都拉到"如何把稀疏奖励分配到动作"这一统一问题下。

## 怎么做 + 靠不靠谱（= taxonomy 与覆盖）

**【二维正交分类（§2.4,核心贡献,Fig.2 网格 / Fig.3 层级树）】** 覆盖 2024-01~2026-04 的 47 篇（41 core + 6 adjacent）。
- **粒度轴（在什么层级分配信用）**:**Token**（生成内单个 token）/ **Segment**（语义跨度,如一个推理步）/ **Step-Turn**（一个完整 LLM 响应或工具调用周期）/ **Multi-Agent**（协作 agent 间分解）。
- **方法学轴（如何算信用）**:**Monte Carlo (MC)**（从中间状态 rollout）/ **Temporal Difference + value**（学值函数 + bootstrap,GAE）/ **Model-based / LLM-as-Critic**（LLM 自身给中间状态自然语言评价,**论文强调它无经典 RL 对应、是 LLM-native 新轴**）/ **Game-theoretic**（Shapley、counterfactual baseline）/ **Information-theoretic**（信息增益、熵）。

**【方法填格（Fig.2/3,真实条目）】**
- **Reasoning RL**:Token 级——VinePPO(MC)、RED(redistribution)、T-REG(self-gen)、From r to Q*(implicit);Segment 级——SPO(MC)、SCAR(Shapley)、TEMPO(tree-TD);Step 级——PURE(PRM/min-form)、SPRO(masked adv)、CAPO(LLM-critic)、HICRA(hierarchy,planning vs procedural)、PRL(entropy)、InT(intervention)、FinePO(fine PRM)、ACPO(attribution)。
- **Agentic RL**:Turn-level PRM——AgentPRM(TD+GAE)、SWEET-RL(privileged asymmetric critic)、Turn-PPO(turn MDP)、SORL(bias-corr)、TARL(LLM-judge)、ITPO(implicit);Hindsight/Counterfactual——HCAPO(hindsight+generative verification)、C3(LOO)、CCPO(causal)、CriticSearch(retrospective);Critic-Free/Step——GiGPO(group-in-group)、POAD(action decomp)、CARL(entropy)、iStar(implicit DPO)、IGPO(info-theoretic);Hierarchical——ArCHer(TD hierarchy/off-policy critic)、PilotRL(progressive)、StepAgent(IRL)。
- **Multi-Agent**:M-GRPO(hierarchical)、SHARP(Shapley)、MAPPA(per-action PRM)、Dr.MAS(agent-wise adv)、LLM-MCA(LLM-critic)、QLLM(LLM-gen)。
- **Adjacent enablers（6 篇,#36–41）**:SPA-RL、Agent Lightning、RAGEN、SCRIBE、PRS、AdaptSeg。

**【经典→LLM 映射（§2.5）】**
- **TD/GAE**:\(\hat A^{\text{GAE}(\gamma,\lambda)}_t=\sum_{l=0}^\infty(\gamma\lambda)^l\delta_{t+l},\ \delta_t=r_t+\gamma V(s_{t+1})-V(s_t)\) → learned critics（ArCHer、AgentPRM）。
- **return decomposition（RUDDER）**:\(c_t=\hat R(s_{0:t})-\hat R(s_{0:t-1})\) → reward redistribution（RED token 级、SPA-RL MLP 进度、IGPO 信息增益）。
- **hindsight（HCA）**:用未来结果重估过去动作 → retrospective（HCAPO 基于 generative verification）。
- **counterfactual / difference reward**:与默认动作对比 → leave-one-out（C3、CCPO turn 级 LOO）与 Shapley（SCAR）。
- **唯一无经典对应的是 LLM-as-Critic**（LLM 给中间状态自然语言评价,Xie 2025/Zhou 2025/Qu 2025）。
- **PRM=CA（§2.5 boxed)**:PRM 给每步 \(r_i\) 打分 = 对 terminal \(R(\tau)\) 做 step 级分解 → PRM 文献(Math-Shepherd/OmegaPRM/PURE) 与 CA 文献(VinePPO/SPRO/SCAR) 是同一问题两视角。
- **RL 算法的 CA 谱（§2.6)**:REINFORCE/GRPO=episode 级（最粗,GRPO 用 group baseline \(\hat A_i=R(\tau_i)-\frac1G\sum_j R(\tau_j)\) 免 critic）;PPO=token 级 via learned critic（细但近似,值网难训）;DPO=隐式 token 级（"From r to Q*" 证 DPO 隐学 token Q 值,iStar/ITPO 据此抽 step 信用）;RLOO/REINFORCE++=仍 episode 级。

**【综合判断（覆盖范围/gaps,对应"实验与证据"）】**
- **reasoning CA 正收敛到 process reward model + critic-free group comparison（成熟化,evidence [SE]）**;**agentic CA 催生真正新方法——hindsight counterfactual(HCAPO)、privileged asymmetric critic(SWEET-RL)、turn-level MDP 重构(Turn-PPO),无 reasoning RL 先例（[LS]）**。Fig.2 网格直观显示方法从 reasoning（左上密集,token/segment 细粒度）→ agentic（右下,step/turn 粗但环境感知）迁移;**step/turn 级聚类最密**（LLM agent 自然粒度）;**multi-agent 与超长 horizon 最薄弱**。五条 takeaway 都标了 evidence level（[SE]强实证/[LS]有限但暗示/[AS]作者综合）。
- **§7 systematic comparison**:按计算成本/是否需辅助模型/适用场景/实证性能比较各方法（含 GRPO-family meta-comparison）。
- **三类可复用产物**:① **机器可读 47 方法清单**（taxonomy 标签/baseline family（G/P/D/O/T）/evidence level（S/L/A）/主 benchmark,将以 CSV/JSON 发布,§B Table 10）;② **reporting checklist**（§C Table 11,已对 HICRA/GiGPO/M-GRPO 三篇验证找系统 gap:0/41 报 GPU-hours、2/41 报方差/CI、0/41 有 compute-controlled baseline）;③ **benchmark 协议规范**（任务族、元数据要求、**controlled bifurcation tasks** 受控分叉任务）+ **方法选择决策树**（Fig.4/Table 8,§9）。
- **假设与失效边界（§9.4 自陈局限）**：【原文】**单作者**（所有筛选/分类/evidence 编码单人执行,释出 screening log 供核验）;preprint 易变（snapshot 2026-04,方法/结果/标题可能变）;cross-paper 数字非受控对比;core/adjacent 与 reasoning/agentic 边界有判断空间;multi-agent CA、超长 horizon、exploration 与 credit 互作用是开放前沿。【推断】部分被引方法（2026 的 HCAPO/C3/CCPO 等,3 篇 counterfactual 同一周 2026-03 出）极新、evidence level 参差,"成熟化"判断对早期方法可能偏乐观;机器可读清单"发表时发布",截 snapshot 未必齐全。
- **祛魅总结**：【推断】真贡献是 **"PRM=CA"的统一透镜 + 跨 reasoning/agentic 的二维分类网格 + 可复用产物（清单/checklist/决策树）**,对快速定位方法、选 baseline、找相邻工作很实用。要祛魅的是:**它是索引+分类+标准化提案,无新方法、无新实验**;"reasoning CA 已成熟"这类判断是综述作者的归纳判断（单作者、对极新方法可能乐观,且已标 evidence level 自陈不确定性）;机器可读清单截 snapshot 未必齐全;仓库是 living list 已超出论文 snapshot。

## 结构化抽取
- 🎯 机制速览6轴（对综述=其 taxonomy 覆盖的机制谱）：**学什么信号**=覆盖全谱——outcome 奖励再分配到 token/segment/step/turn/agent（MC rollout / TD value / LLM-critic 评价 / Shapley-counterfactual / 信息增益-熵）｜**改什么**=综述对象多为"改 advantage/reward 的分配"（policy gradient 的信用）,含 learned critic / PRM / group baseline｜**何时改**=覆盖 in-training（per-step/per-turn 的 CA 计算）｜**免梯度?**=不适用（综述本身）;覆盖的方法多为梯度 RL,少数 critic-free（group comparison）｜**记忆-技能生命周期**=不适用（但 §9.3 提"CA 与 memory":对存储/检索/更新 summary 等记忆动作如何分信用是开放问题,需类比 eligibility trace 扩到语义记忆）｜**防遗忘机制**=不适用
- ⑦ 开源代码+框架/harness：列表 https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL （旧 analysis 已核:旧 KEYS 的 `xxzcc/Awesome-Credit-Assignment` 是 404,此为正确仓库名;MIT,living list,2026.05 又加 agentic/coding-agent/uncertainty-control/multi-agent 新论文）。内容=`README.md`（分类表）+ `gen_figures.py`（生成 taxonomy 网格图与树图）+ `assets/`。**框架=不适用**——是论文清单 + 分类可视化脚本,非训练代码。【待核】仓库细节复用旧 analysis 核查;机器可读 CSV/JSON 清单论文称"发表时发布",需留意是否已上传。
- 💰 资源/成本与可扩展性：原文未说明（综述无训练）。§7 做 systematic comparison（按计算成本/是否需辅助模型/适用场景/实证性能比较各方法,含 GRPO-family meta-comparison）;§9.2 提出"**computation-signal trade-off / CA efficiency frontier**"——固定算力下"更多 rollout + 粗 episode 信用(GRPO)" vs "更少 rollout + 细信用(VinePPO/HCAPO)",猜测随轨迹变长应偏向细信用,但无系统答案。
- 🎯 对"探索-巩固"对标：**索引/定位工具（上位框架）**。判定:CA（把稀疏 outcome 奖励分配到 token/step/turn）正是 TSRD"path-selection / path-recovery 细粒度信用分配"的**上位框架**——TSRD 的"单点接管/在关键步给信号"本质是 step/segment 级 CA。**可借**:① 用其二维分类（粒度×方法学）给 TSRD 定位（属 step/segment 级 + LLM-as-Critic 或 counterfactual 一类）;② 从其 47 篇里挑 baseline（如 VinePPO/SPO/SCAR/PURE/HICRA/HCAPO/SWEET-RL/GiGPO）与相邻工作;③ §9 benchmark 协议的 **controlled bifurcation tasks（受控分叉任务）** 思想很契合"选路/回轨"的评测设计;④ reporting checklist 可用于自查 TSRD 论文（报 GPU-hours/方差/compute-controlled baseline）;⑤ §9.2 "CA 与 exploration 结合"（IGPO 用信息论框信用,信用最不确定处正是该探索处）与 TSRD 选路高度相关。**缺口**:本身不涉及蒸馏/OPD/MTP/path-recovery 的具体机制,只是地图不是方法。
- 🔭 开放问题/未来方向：【原文】(§9) multi-agent credit（可扩展分解/通信信用/部分可观测）、超长 horizon CA、exploration 与 credit 互作用、CA-memory、reasoning→agentic 技术迁移（VinePPO vine 扩到 turn 边界、PURE min-form 扩 turn-PRM、HICRA planning-procedural 扩 agentic）、formal guarantees（多数方法无收敛保证,仅 VinePPO/PURE/CCPO 有部分）是开放前沿;给 roadmap（Classical RL → Reasoning RL → Agentic RL → Future Multi-Agent）。【推断】把"controlled bifurcation tasks"做成标准 path-selection/recovery 评测;统一 PRM 与蒸馏 token 级信号在 CA 框架下;MTP 前瞻作为一种新的 CA 信号（用未来 token 可预测性给当前步分配信用,可纳入 information-theoretic 轴）。

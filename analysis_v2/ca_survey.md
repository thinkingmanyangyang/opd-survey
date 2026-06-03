ca_survey | From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models | Chenchen Zhang（独立研究者，单作者 survey）| 2026-04（arXiv:2604.09459v2，cs.CL）| 主题线 L3 RLVR/信用分配（综述）·相关性 中

**原始论文**：https://arxiv.org/abs/2604.09459

> 注：本篇是 **survey + awesome-list**，非新方法论文。以下"怎么做/逐组件"栏改写为其 **taxonomy（分类法）与覆盖范围**。

## 一眼看懂
- 🟦 TL;DR：一篇以"信用分配(credit assignment, CA)"为中心透镜重审 LLM RL 的综述。CA = 把稀疏的 outcome 奖励分配到具体 token/step/turn/agent。核心产物：把 2024–2026 初的 **47 篇方法（41 core + 6 adjacent enabler）**按**粒度×方法学二维分类**（Fig.2/3），配机器可读清单、reporting checklist、benchmark 协议 + 方法选择决策树。判断：reasoning CA 正成熟化（收敛到 PRM + critic-free group comparison），agentic CA 催生真正新方法（hindsight counterfactual、privileged asymmetric critic、turn-level MDP）。
- 最巧的一步（综述层面）：**"PRM 本质就是 CA"这一概念澄清（§2.5 boxed）**——把 PRM 文献（Math-Shepherd/OmegaPRM/PURE）和 CA 文献（VinePPO/SPRO/SCAR）统一成"同一问题的两个视角"。抽掉这个统一视角，整篇就只是方法罗列；正是它让"reasoning CA 已成熟"的判断站得住（PRM 密集=step 级 CA 密集）。

## 为什么做
- 研究背景：LLM RL 两波——**reasoning RL**（DeepSeek-R1/o1，单条 CoT 500–30K+ token，纯 outcome 奖励）→ **agentic RL**（多轮环境交互，10–100+ turn，100K–1M token，稀疏 terminal 奖励）。经典 RL 已有 CA 工具箱（TD/GAE、return decomposition RUDDER、hindsight HCA、counterfactual/difference reward），是 LLM 方法的基础（§2.5）。
- 解决的具体痛点（survey 要填的 gap）：① **Episode 级方法**（GRPO/REINFORCE）给每 token 同一 advantage，轨迹一长就失效（turn 3 的错误 tool-call 与后续几十个正确动作受同样惩罚，"echo trap"；§4 论证 T=100 时单动作信噪比约差 100×）；② agentic 使 CA **双层级化**（先判哪个 turn 关键，再判 turn 内哪些 token 重要）+ 随机转移/部分可观测/超长 horizon；③ 现有相邻综述（Pignatelli 2023 只覆盖经典 RL；Zhang 2025a agentic 100 页综述把 CA 当子话题）——**无人系统覆盖 reasoning+agentic 两个 regime 的 CA**。
- 相关工作 & 各自不足：见上③——要么只经典 RL，要么 CA 是附属，**缺专门、跨两 regime 的 CA 综述**。
- 动机链：现状（LLM RL 从 reasoning 演进到 agentic，CA 是核心难题）→ 缺陷（无统一 CA 透镜的综述 + episode 级方法在长 horizon 失效）→ 所以做专门 CA 综述 + 可复用产物（清单/checklist/benchmark 协议/决策树）。
- 与最近邻工作的Δ：vs Pignatelli 2023（前 LLM CA）——补 LLM 时代；vs Zhang 2025a（agentic 综述）——以 CA 为唯一透镜并跨 reasoning+agentic，且论证"为何 agentic 比 reasoning 的 CA 更难、催生哪些新技术"。关键点：把所有方法（含 PRM）都拉到"如何把稀疏奖励分配到动作"这一统一问题下。

## 怎么做 + 靠不靠谱（= taxonomy 与覆盖）
- 分类法（§2.4，核心贡献）：覆盖 2024-01~2026-04 的 47 篇（41 core + 6 adjacent）。**二维正交轴**——**粒度轴**：Token / Segment / Step-Turn / Multi-Agent；**方法学轴**：Monte Carlo（VinePPO/SPO）/ Temporal Difference+value（GAE；AgentPRM 用 TD+GAE 学 turn-level value、ArCHer 用 off-policy critic）/ **Model-based / LLM-as-Critic**（LLM 自身给中间状态自然语言评价，CAPO 等，**论文强调它无经典 RL 对应、是 LLM-native 新轴**）/ Game-theoretic（Shapley—SCAR / counterfactual LOO—C3、CCPO）/ Information-theoretic（信息增益—IGPO / 熵—CARL）。
- 经典→LLM 映射（§2.5）：return decomposition(RUDDER)→reward redistribution（RED token 级、SPA-RL）；hindsight(HCA)→retrospective（HCAPO 基于 generative verification）；counterfactual→leave-one-out 与 Shapley（C3/CCPO 做 turn 级 LOO、SCAR 用 Shapley）；TD/GAE→learned critics（ArCHer、AgentPRM）。**唯一无经典对应的是 LLM-as-Critic**。
- 综合判断（覆盖范围/gaps，对应"实验与证据"）：**reasoning CA 正收敛到 process reward model + critic-free group comparison（成熟化）**；**agentic CA 催生真正新方法——hindsight counterfactual（HCAPO）、privileged asymmetric critic（SWEET-RL）、turn-level MDP 重构（Turn-PPO），无 reasoning RL 先例**。Fig.2 网格直观显示方法从 reasoning（左上密集，token/segment 细粒度）→ agentic（右下，step/turn 粗但环境感知）迁移；**step/turn 级聚类最密**（LLM agent 自然粒度）；**multi-agent 与超长 horizon 最薄弱**。
- 三类可复用产物（论证其"支撑"靠系统性而非实验）：① 机器可读 47 方法清单（taxonomy 标签/baseline family/evidence level/主 benchmark，将以 CSV/JSON 发布，§B）；② **reporting checklist**（§C，已对现有文献验证以找系统性 gap）；③ **benchmark 协议规范**（任务族、元数据要求、controlled bifurcation tasks）+ **方法选择决策树**（Fig.4/Table 8，§9）。
- 假设与失效边界（局限）：【原文】(§9.4 自陈) **单作者**，覆盖可能有缺口；multi-agent CA、超长 horizon、exploration 与 credit 的互作用是开放前沿；core vs adjacent 边界 case 在 §9.4 讨论；CSV/JSON 清单"发表时发布"。【推断】47 篇含 6 篇 adjacent（如 RAGEN/Agent Lightning/PRS），core/adjacent 边界有判断空间；部分被引方法（2026 的 HCAPO/C3/CCPO 等）极新、evidence level 参差，"成熟化"判断对早期方法可能偏乐观；检索协议（arXiv/Semantic Scholar/Google Scholar 关键词 + VinePPO/ArCHer/GRPO/R1 前后向引用 + 监控 NeurIPS/ICML/ICLR/ACL 2025 与 HF Daily）虽系统但单人执行。
- 祛魅总结：【推断】真贡献是 **"PRM=CA"的统一透镜 + 跨 reasoning/agentic 的二维分类网格 + 可复用产物（清单/checklist/决策树）**，对快速定位方法、选 baseline、找相邻工作很实用。要祛魅的是：**它是索引+分类+标准化提案，无新方法、无新实验**；"reasoning CA 已成熟"这类判断是综述作者的归纳判断（单作者、对极新方法可能乐观）；机器可读清单截至 snapshot 未必齐全（"发表时发布"）；仓库是 living list 已超出论文 snapshot。

## 结构化抽取
- 🎯 机制速览6轴（对综述=其 taxonomy 覆盖的机制谱）：**学什么信号**=覆盖全谱——outcome 奖励再分配到 token/segment/step/turn/agent（MC rollout / TD value / LLM-critic 评价 / Shapley-counterfactual / 信息增益-熵）｜**改什么**=综述对象多为"改 advantage/reward 的分配"（policy gradient 的信用），含 learned critic / PRM / group baseline｜**何时改**=覆盖 in-training（per-step/per-turn 的 CA 计算）｜**免梯度?**=不适用（综述本身）；覆盖的方法多为梯度 RL，少数 critic-free（group comparison）｜**记忆-技能生命周期**=不适用｜**防遗忘机制**=不适用
- ⑦ 开源代码+框架/harness：列表 https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL （旧 analysis 已核：旧 KEYS 的 `xxzcc/Awesome-Credit-Assignment` 是 404，此为正确仓库名；MIT，living list，2026.05 又加 agentic/coding-agent/uncertainty-control/multi-agent 新论文）。内容=`README.md`（分类表）+ `gen_figures.py`（生成 taxonomy 网格图与树图）+ `assets/`。**框架=不适用**——是论文清单 + 分类可视化脚本，非训练代码。【待核】仓库细节复用旧 analysis 核查；机器可读 CSV/JSON 清单论文称"发表时发布"，需留意是否已上传。
- 💰 资源/成本与可扩展性：原文未说明（综述无训练）。§7 做 systematic comparison（按计算成本/是否需辅助模型/适用场景/实证性能比较各方法，含 GRPO-family meta-comparison）。
- 🎯 对"探索-巩固"对标：**索引/定位工具（上位框架）**。判定：CA（把稀疏 outcome 奖励分配到 token/step/turn）正是 TSRD"path-selection / path-recovery 细粒度信用分配"的**上位框架**——TSRD 的"单点接管/在关键步给信号"本质是 step/segment 级 CA。**可借**：① 用其二维分类（粒度×方法学）给 TSRD 定位（属 step/segment 级 + LLM-as-Critic 或 counterfactual 一类）；② 从其 47 篇里挑 baseline（如 VinePPO/SCAR/HCAPO/SWEET-RL）与相邻工作；③ §9 benchmark 协议的 **controlled bifurcation tasks**（受控分叉任务）思想很契合"选路/回轨"的评测设计；④ reporting checklist 可用于自查 TSRD 论文。**缺口**：本身不涉及蒸馏/OPD/MTP/path-recovery 的具体机制，只是地图不是方法。
- 🔭 开放问题/未来方向：【原文】(§9) multi-agent credit、超长 horizon CA、exploration 与 credit 的互作用是开放前沿；给出 roadmap（Classical RL → Reasoning RL → Agentic RL → Future Multi-Agent）。【推断】把"controlled bifurcation tasks"做成标准 path-selection/recovery 评测；统一 PRM 与蒸馏 token 级信号在 CA 框架下；MTP 前瞻作为一种新的 CA 信号（用未来 token 可预测性给当前步分配信用）。

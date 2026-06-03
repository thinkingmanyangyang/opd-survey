score | From Correction to Mastery: Reinforced Distillation of LLM Agents (SCoRe) | 中科大 USTC(Yuanjie Lyu、Tong Xu 通讯)+ Independent Researcher(Chengyu Wang 通讯、Jun Huang,原阿里 EasyDistill 团队) | 2025-09 arXiv 2509.14257(v2 2025-10-09)·预印本 | 主题线 L4(Agent/工具多轮自进化)+L1(蒸馏)·相关性 高

**原始论文**:https://arxiv.org/abs/2509.14257

## 一眼看懂
- 🟦 TL;DR:小学生 agent(7B)做不动大 teacher(72B)整条轨迹的模仿——任何一步走错就被推到 OOD、误差按 O(H²) 滚雪球。SCoRe 反过来让**学生自己主导**生成轨迹,teacher **只纠"最早出错的那一步"**,学生从被纠正的前缀继续走;先在这种"学生中心、最小干预"的纠错轨迹上做 SFT,再做**短 horizon RL**(从"已验证前缀"起 rollout,而不是从题目开头)并给**关键步稠密奖励**。结果 7B 学生 8 项数学+事实推理 Avg=50.8,只比 72B teacher(51.7)低 0.9。
- 最巧的一步:**"teacher 只纠最早错误 + 学生从该前缀续写"**。抽掉它,就退回普通行为克隆(teacher-acts/student-clones),误差链恢复成 O(H²),后面的短 horizon RL 也失去"已验证前缀"这个起跑点(RL 创新整个失去支点)。这一步同时实现论文吹的两个性质:Capability Matching(轨迹复杂度匹配学生能力)+ Deficiency Localization(前缀+关键步精确暴露弱点)。【原文】§1、§3.2、Fig.1

## 为什么做
- 研究背景:LLM agent 靠 ReAct 式"推理-动作-观测"循环 + 外部工具(代码解释器/搜索)解复杂任务,但要 GPT-4/Qwen-72B 这类超大 backbone,延迟成本高(复杂任务要几十次模型调用)。Agent Distillation(Kang 2025)把 teacher 行为拆成 [Thought,Action,Observation] 让小学生模仿。本文建在 **CodeAct**(action=可执行代码)之上。【原文】§1
- 解决的具体痛点:"teacher-acts, student-clones"全轨迹模仿有两个 gap——(1) Reasoning Ability Gap:小模型复现不了 teacher 的逻辑分解;(2) Knowledge Capability Gap:照搬计划也可能因知识不足执行不了复杂动作。二者都源于 emergent abilities,不可完全迁移。更致命的是 BC 在协变量偏移下任一步出错把学生推入 OOD,误差按 **O(H²ε)** 随 horizon 复合(Ross 2011)。【原文】§1、§2、定理 3.1
- 相关工作 & 各自不足:DAgger/HG-DAgger(Ross 2011、Kelly 2019)缓解 exposure bias 但仍 teacher-led、轨迹复杂度常与学生能力错配;Toolformer 式模仿迁移规划/工具技能但仍是 BC;agentic RL(GRPO/ARPO)能探索但长 rollout 不稳、稀疏奖励难做信用分配。【原文】§5
- 动机链:大 backbone 贵 → 蒸到小模型 → 全轨迹 BC 有双 gap + O(H²) 误差链 → 改"学生主导、teacher 只纠最早错"把分布偏移限制在单步、误差降到 O(H) → 再加短 horizon RL 从已验证前缀起跑、关键步稠密奖励,把"模仿"推向"自主解题"。
- 与最近邻工作的Δ:vs DAgger——SCoRe 不是 teacher 在学生分布上全程标注,而是**只纠最早一步**且学生续写;vs GRPO/ARPO(全 horizon RL)——SCoRe 的 RL **从最早错误前的前缀起 rollout**(缩短 horizon、降方差)且用关键步奖励而非纯结果奖励。关键差别就是"把监督/探索都锚在学生真实能力边界(最早出错点)"。

## 怎么做 + 靠不靠谱
- 方法流水线(三阶段):
  1. **Cold-Start BC**:用 teacher(Qwen2.5-72B)经"先 high-level plan 再 Thought-Code-Observation 循环"双 prompt 生成轨迹,rejection sampling 只留最终答案正确者;在其中 **20%** 数据上 SFT 得 ˆπinit,让学生先会走多步循环。【原文】§3.1、脚注 1
  2. **Mentored Problem-Solving(MPS)+ SCoRe-SFT**:剩 **80%** 题让 ˆπinit 独立做 → teacher 检查、定位**最早偏离步 σk**、替换为 σ′k、学生从 (σ1..σk−1,σ′k) 续写;再错就再纠(本文上限 5 次);最终任务成功**隐式验证**修正正确。把这些"最小干预"纠错轨迹的一半拿来 SFT。【原文】§3.2
  3. **SCoRe-RL(GRPO)**两创新:(a) **Short-Horizon Rollout**——从已验证前缀 (σ1..σk−1) 起 rollout,horizon 从 H 缩到 H−k+1;(b) **Key-Step Reward**——最终答案对 reward=1;否则关键步=teacher 修正→0.5,关键步既非原错也非 teacher 修正→0.1,等于原错→0。奖励正误由**轻量 Qwen2.5-7B-Instruct verifier** 判语义等价(注意:与评测用的 72B judge 不同)。【原文】§3.3、Fig.2(c)
- 逐组件必要性(Table 3 消融,Qwen2.5-7B,8 项 Avg):
  - Initial Distillation(只 BC 20% 数据)= 41.7 —— 故意弱,作为 MPS 的 explorer。
  - + MPS/SCoRe-SFT = 45.8(较 Initial +4.1)→ 证 MPS 数据有效强化弱链。
  - **去 short-horizon rollout** → 48.4(满配 50.8,掉 2.4)→ 证缩 horizon 降方差有用。
  - **去 key-step rewards** → 49.7(掉 1.1)→ 证关键步稠密奖励有用。
  - 满配 SCoRe-RL = 50.8。两组件都做了消融,缺一掉点,必要性成立。【原文】Table 3、§4.3
- 关键机制/公式(直觉):定理 3.1 把 BC 误差界给成 c(ˆπ)≤c(πE)+H(H−1)/2·ε=O(H²ε)(在 teacher 分布 dπE 上算 ε);定理 3.2 在**学生自身分布 dˆπ** 上算 ε,因"最多一个未纠错误就回到 expert 路径",界收紧到 O(Hε)。定理 3.3 给 short-horizon 的策略梯度方差界随 k 增大单调下降。三个定理都是直觉化的 covariate-shift/方差论证,非新数学。【原文】定理 3.1–3.3、附录 A
- 实验与证据:
  - **数据集**:12 benchmark 三类——数学(AIME24/25、MATH500、OlympiadMath)、事实多跳 QA(HotpotQA、2Wiki、MuSiQue、Bamboogle,用 token-F1)、深度搜索(GAIA、WebWalker、HLE、xBench,WebThinker text-only split)。种子主要取自 **Tool-Star**(NuminaMath/Omni-Math + HotpotQA/2Wiki/WebWalker),共 **35k QA 对**。【原文】§4.1、Implementation
  - **关键数字**:Qwen2.5-7B SCoRe-RL=**50.8**(BC=42.5、GRPO=48.4、ARPO=49.3),仅低 72B teacher(51.7)0.9;Qwen2.5-3B=46.7(+8.4 over BC)、Llama3.1-8B=47.5(+10.2 over BC)。深度搜索(Qwen3-8B,Table 2):SCoRe-RL Avg=**30.5**(+7.7 over BC、+8.3 over GRPO、超 TIR-72B teacher +3.2),GAIA-Avg 27.2(BC)→40.8。Fig.3:数学 hard data(200 题,已排除训练)正确率 0%→17.3%(SFT)→24.3%(RL)。【原文】Table 1/2、Fig.3
  - **baseline 公平吗**:GRPO/ARPO 数值多取自 ARPO 原文(Dong 2025b),评测集组织也follow ARPO,口径基本对齐;BC baseline 用**全量** teacher 数据,而 SCoRe 的 cold-start BC 只用 20%——这对 SCoRe 反而更苛刻(脚注 1),较公平。
  - **看着强但没回答核心问题?**:"逼近 teacher"成立(7B 仅低 0.9),但"教 student 反超它教不了的题"(Fig.3 hard data 24.3%)虽亮眼,绝对值仍低;且 RL verifier(7B)与评测 judge(72B)不同,存在训练-评测口径不一致 / reward hacking 的潜在风险,论文未交叉验证。【推断】
- 假设与失效边界:
  - 【原文】"最终任务成功隐式验证 teacher 修正正确"——是**弱验证**:任务成功 ≠ 每步修正都对,可能引入噪声标签,论文未量化误纠率。
  - 【原文】定理 3.2 假设"每步纠正后回到 expert-aligned 路径、至多一个未纠错误",现实中 teacher 定位"最早错误"本身可能错判。
  - 【推断】方法依赖 teacher 是 72B 级强模型 + 可执行代码做确定性动作等价判定(CodeAct);换到无法用代码精确判等价的任务域,key-step reward 的可靠性下降。
- 祛魅总结:真贡献=**"最早错误纠正 + 从已验证前缀短 horizon RL"这条把 BC 误差链 O(H²)→O(H) 并稳定 RL 的完整 pipeline**,消融扎实、跨 3 个 student 一致。包装/高估处:(1) 论文 prose §4.2 写 7B "+6.3 over GRPO"与自身 Table 1(48.4 vs 50.8=**+2.4**)矛盾,应以表为准;(2) "小成本"只指部署期 student,蒸馏期仍需 72B teacher + 多轮 rollout,成本不低,易被低估;(3) 三个"理论 justification"是经典结论的复述,非新理论。【推断】

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=teacher 对"最早错误步"的纠正(SFT)+ 关键步是否复现该纠正/避开原错的稠密奖励 + 最终答案正确性(RL)｜**改什么**=student 参数(SFT 全参 + GRPO 策略更新)｜**何时改**=三阶段离线串行(BC→MPS-SFT→短 horizon RL)｜**免梯度?**=否,SFT+RL 都更新参数｜**记忆-技能生命周期**=无显式记忆/技能库,知识固化进参数(轨迹→SFT→RL)｜**防遗忘机制**=无专门防遗忘设计,靠"前 20% BC 冷启动 + 能力匹配数据"隐式维持基础技能。【原文】§3
- ⑦ 开源代码+框架/harness:https://github.com/modelscope/easydistill(SCoRe 在 `projects/SCoRe/`,整库 clone ~221MB)。框架=**组合工具链非单一**:Cold-start/SCoRe-SFT 用 **LLaMA-Factory**;agent 轨迹生成用 **LangGraph**(`MPS/graph/{graph.py,graph_repair.py}`);SCoRe-RL 用 **veRL**(`verl/tools/search_tool.py`);推理 LLaMA-Factory api(vllm/sglang);teacher 经 Qwen2.5-72B API 或本地部署。代码可得性 Tier A。【原文】GitHub + 既有 analysis 仓库核查
- 💰 资源/成本与可扩展性:RL 最大 rollout 步数=8;训练用 Qwen2.5-72B 作 teacher 标注 + 生成 35k 题轨迹,成本主要在蒸馏期(原文未给 GPU·小时);部署期仅 7B student。完整超参(组大小/lr/KL)未在正文给出〔待核:附录 B〕。【原文】Implementation
- 🎯 对"探索-巩固"对标:**强支撑 + 高度同构**。SCoRe 的"学生主导探索 + teacher 只纠最早错 + 从被纠前缀续写"正是 idea 里"teacher 当稀疏脚手架教 path-selection(偏向自己走通的开头)+ path-recovery(走偏后从恢复分支续写并固化)";on-policy 自选体现在"学生先独立生成全轨迹";巩固=纠错轨迹 SFT + 关键步 RL 固化进参数。**可借组件**:(a)"最早错误定位 + 单点接管"做 path-recovery 的离散实现;(b) short-horizon rollout from verified prefix 直接可移植到 OPD 的 RL 阶段降方差。**缺口/差异**:SCoRe 用离散"动作等价"判关键步,无 MTP 式前瞻信号,也无 token 级稠密信用;且纠正点由 teacher 判而非学生自选恢复分支(idea 强调"学生自选恢复"),这点有偏差。一句判定:**最贴近 idea 的 agent 落地之一,可作 path-selection/recovery 的离散基线,但缺 MTP 前瞻与学生自主选恢复点**。【推断,依据 §3.2/Fig.1 vs idea】
- 🔭 开放问题/未来方向:【原文】改进 reward 设计、扩展到多模态任务(§6 Conclusion)。【推断】(1) 量化 teacher 误纠率、用更强 verifier 或多数投票降噪声标签;(2) 把"最早错误"的离散纠正换成 token/段级稠密信用(可接 MTP 前瞻或 IG 归因)以减少对 teacher 判断的依赖;(3) 让学生自主选择"从哪个分支恢复"而非 teacher 指定,更贴近真正的 path-recovery;(4) RL verifier 与评测 judge 统一以排除口径不一致风险。

RETURN:score | 读到PDF? 是(23页全文+全表全消融全证明) | L4(+L1) | 对标=强支撑/高度同构,可作 path-selection+recovery 离散基线,缺 MTP 前瞻与学生自选恢复点 | 残留待核 1(附录 B 完整 RL 超参)

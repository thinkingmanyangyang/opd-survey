sweet_rl | SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks | FAIR at Meta + UC Berkeley(Yifei Zhou, Song Jiang, Yuandong Tian, Jason Weston, Sergey Levine, Sainbayar Sukhbaatar*, Xian Li*) | arXiv 2503.15478 v1(2025-03-19, cs.LG) | 主题线 L4(多轮 agent step 级信用分配 + 新 benchmark)·相关性 中高

**原始论文**:https://arxiv.org/abs/2503.15478

## 一眼看懂
- 🟦 TL;DR:多轮 agent 任务里,把单轮 RLHF(RAFT/DPO/PPO)直接外推做不好"跨轮信用分配";学 value head 又泛化差。SWEET-RL 用一个**非对称 critic**——它能看 actor 推理时看不到的训练期特权信息(最终结果 + 参考解 \(c\))——直接学 turn-level **advantage 函数**(用每轮动作平均 log-prob 参数化),用 **Bradley-Terry 轨迹级目标**训练,再把它当每轮 reward 对 actor 做 DPO。配套开源多轮协作 benchmark ColBench。让 Llama-3.1-8B 在协作内容创作上匹配/超 GPT-4o。【原文 §4, Table 2】
- 最巧的一步:**直接学 advantage 而非先学 value、且用动作 log-prob 参数化 advantage**(Eq 3:\(A_\theta(o_t,a_t,c)=\frac{1}{L}\sum_{l=1}^{L}\log\frac{\pi_\theta(a^l_t\mid o_t,a^{1:l-1}_t,c)}{\pi_{ref}(a^l_t\mid o_t,a^{1:l-1}_t,c)}\))。抽掉这步(改回训 value head / regression head),Best-of-N 与最终成功率都明显掉(Fig.3a / Table 3),因为 value head 偏离 next-token 预训练目标、泛化差;另一抽不得的细节是 advantage 的 **\(1/L\) 长度归一**——去掉它 actor 会坍塌成越来越短的回答(Table 3 w/o Normalization)。【原文 §4.2-4.3, §5.4】

## 为什么做
- 研究背景:LLM agent 要在真实任务里多轮交互、直接优化多轮目标(成功率),比 next-token 预训练只模仿每轮最可能动作更难。【原文 §1, lines 81-83】
- 解决的具体痛点:(1) 把单轮 RLHF(RAFT/DPO/PPO)直接套多轮、不做跨轮显式信用分配,长 horizon 下高方差、样本复杂度差;(2) value-function/TD-learning 需在 LLM 表征上训任务专用 value head,微调数据有限时泛化差(与 next-token 目标差异大);(3) 缺乏同时满足"任务多样性 + 推理复杂度 + 最小工程开销"三条件的 benchmark(Table 1 现有 benchmark 无一兼具)。【原文 §1, §2, Table 1】
- 相关工作 & 各自不足(把多轮 RL 信用分配的几条路线铺开):
  - **单轮 RLHF 外推到多轮**:RAFT(拒绝采样微调)、DPO、PPO/REINFORCE 直接把整条轨迹当一个"长动作"或对轨迹做偏好——无跨轮显式信用分配,长 horizon 高方差、样本复杂度差。【原文 §2, lines 211-217】
  - **value-function / TD bootstrapping 一路**:Bai 2024 / Zhou 2024 / Snell 2023 先学 value 函数再导出 advantage,Bellman bootstrapping / Path Consistency 降方差。短板:在推理密集任务上学准 value 本身很难、且训任务专用 value head 与 next-token 预训练目标差异大、微调数据有限时泛化差。SWEET-RL 因此**跳过 value、直接学 advantage**。【原文 §2, §4.2, lines 417-423】
  - **过程奖励模型(PRM)一路**:step-wise critic 与 PRM 思路相近(都给中间步打分),但 PRM 多用于 **test-time search** 或加速探索;SWEET-RL 把 step-wise critic 当"reward 代理"**直接优化 policy**,无需额外采交互数据。【原文 §2, lines 223】
  - **机器人里的非对称 actor-critic 一路**:critic 看 latent/特权状态(actor 看不到)在机器人控制中已有先例,但少有人把它用到**推理密集的 LLM 任务**上——SWEET-RL 把"训练期可得参考解/终局"当这份特权信息。【原文 §2, lines 127-128】
  - **benchmark 缺口**:Table 1 对比现有多轮 benchmark,无一同时满足"任务多样性 + 推理复杂度 + 最小工程开销",故自建 ColBench。
  - 动机链:多轮目标需跨轮信用分配 → 单轮外推高方差、value head 泛化差 → 注意到训练期可得 actor 看不到的特权信息(outcome/参考解)→ 用它给非对称 critic 做信用分配捷径 → 但训标量 value 偏离预训练 → 改为直接学 advantage(log-prob 参数化)+ BT 目标 → SWEET-RL。【原文 §1, lines 110-130】
  - 与最近邻工作的精确Δ:vs Multi-Turn DPO(最强基线,无 critic、直接对比轨迹做 DPO),关键差异=显式训一个 turn-level reward model 做信用分配(借训练期特权信息),而非仅轨迹级偏好;vs value-function 路线,关键差异=直接学 advantage + 用 LLM head 参数化(契合预训练)而非训 regression/value head。有用是因为信用分配落到每轮、且不偏离 LLM 预训练目标 → 泛化更好(Fig.3a Best-of-N 缩放最优)。【原文 §5.3-5.4】

## 怎么做 + 靠不靠谱
- 问题形式(POMDP):\(M=\{O,C,A,T,\mu_1,R,N\}\),\(O\) 可观测态、\(C\) 隐藏态;episode 初抽初始指令 \(o_1\) 与隐藏训练期信息 \(c\in C\)(如参考解,episode 内不变)。第 \(t\) 轮 agent 见交互历史 \(o_t\)、出动作 \(a_t=a^{1:L}_t\)(token 序列),环境把最新交互追加进历史得新态;每步可得标量奖励 \(r(o_t,a_t,c)\)。目标 \(\max\sum_{t=1}^N r(o_t,a_t,c)\)。**offline 设定**(在线人类交互太贵)。Q/V/A:\(A^\pi(o_t,a_t,c)=Q^\pi(o_t,a_t,c)-V^\pi(o_t,c)\)。【原文 §4.1, lines 385-410】
- 方法流水线(两阶段,输入→输出):
  1. **训 critic/advantage(BT 目标)**:同任务下取两条离线轨迹,按累计回报标 chosen \(\tau^+\) / rejected \(\tau^-\) → 用 Bradley-Terry 目标在**轨迹级**训:原始 BT \(J_{BT}=-\log\sigma\big(\sum_t\beta r(o^+_t,a^+_t,c)-\sum_t\beta r(o^-_t,a^-_t,c)\big)\)(Eq 1),用 advantage 改写为 \(J_A(\theta)=-\log\sigma\big(\sum_t\beta A_\theta(o^+_t,a^+_t,c)-\sum_t\beta A_\theta(o^-_t,a^-_t,c)\big)\)(Eq 2);effect=抬高 \(\tau^+\) 每动作 advantage、压低 \(\tau^-\) 每动作 advantage。advantage 用动作平均 log-prob 比参数化(复用 LLM head,critic 输入含特权信息 \(c\)):
     \[
     A_\theta(o_t,a_t,c)=\frac{1}{L}\sum_{l=1}^{L}\log\frac{\pi_\theta(a^l_t\mid o_t,a^{1:l-1}_t,c)}{\pi_{ref}(a^l_t\mid o_t,a^{1:l-1}_t,c)},
     \]
     (Eq 3,\(\pi_\theta\) 是待训 advantage LLM,\(\pi_{ref}\) 是冻结 seed,\(1/L\) 长度归一稳定训练)。
  2. **policy improvement(DPO)**:把训好的 advantage 当每轮 reward model,对 actor \(\pi_\phi\)(只看交互历史 \(o_t\)、不看 \(c\))做 DPO——每轮采 16 候选动作、按 advantage 排序,top-50% 分位随机取为 chosen \(a^+\)、bottom-50% 取为 rejected \(a^-\),标准 DPO loss:
     \[
     J_\pi(\phi)=-\log\sigma\!\left(\beta'\frac{\log\pi_\phi(a^+\mid o_t)}{\log\pi_{ref}(a^+\mid o_t)}-\beta'\frac{\log\pi_\phi(a^-\mid o_t)}{\log\pi_{ref}(a^-\mid o_t)}\right).
     \]
     (Eq 4)此阶段无需人类交互(offline)。【原文 §4.1-4.3, lines 510-523】
- 逐组件必要性(消融齐全):
  - **训练期特权信息 \(c\)**:核心。Fig.3a"SWEET-RL w/o Training-Time Information"Best-of-N 成功率大幅低于 SWEET-RL,Table 3 同结论 → 没它信用分配能力骤降。
  - **直接学 advantage + log-prob 参数化**:vs"w/ Value Function"(分类 head 预测成功率)Fig.3a 缩放显著更差;vs"w/ Regression Head"(均值池化上接回归头)Table 3 泛化差 → 参数化选择关键。
  - **\(1/L\) 长度归一**:Table 3"w/o Normalization"→ actor 坍塌成越来越短回答 → 必需。
  - **BT 轨迹级目标**:把单轮偏好优化的成功外推到轨迹级(Eq 1→2),理论推导见 Appendix B。
  - **多轮交互本身**:Table 2 单轮 vs 协作对比——所有模型成功率翻倍(Llama-8B 6.9%→22.4%)→ 多轮收集信息是任务关键。
- 关键机制/公式(直觉):actor 在部分可观测环境里做"信息搜寻"动作,其价值难评(要到终局才知);但训练期 critic 能看参考解 \(c\),等于开了上帝视角判断"这步是否在正轨上",从而给信息搜寻动作合理信用。把 advantage 写成 log-prob 比值,使 critic 训练就是 LLM 偏好微调,契合预训练、泛化好。【原文 §4.2-4.3, lines 499-504】
- 实验与证据:
  - 数据集=自建 **ColBench** 两任务:**Backend Programming**(写 ≤50 行 Python 函数,10 轮内,10 个隐藏单测给 0/1 终局奖励;train 10k/test 1k 人工检查;15k 离线轨迹由 Llama-3.1-8B 当 agent、Llama-3.1-70B 当 human simulator 零样本生成)与 **Frontend Design**(写 ~100 行 HTML,CLIP 图像相似度当 0-1 奖励;tasks 来自 WebSight,train 10k/test 500;6k 离线轨迹,human simulator=Qwen2-VL-72B)。【原文 §3.2-3.3】
  - 设置:actor backbone=Llama-3.1-8B-Instruct;Backend 的 advantage model=同架构 LLM;**Frontend 的 advantage model=Qwen2-VL-7B-Instruct + 回归头**(因参考网页是多模态)。**offline RL**(明确 PPO/REINFORCE 不适用,因在线人类交互昂贵)。【原文 §5.1, lines 616-625】
  - 关键数字(Table 2,Llama-3.1-8B-Instruct):Backend Success Rate——SWEET-RL **40.4** vs Multi-Turn DPO 34.4(**+6.0**)vs Rejection-FT 28.2 vs Zero-Shot 22.4;Frontend Win Rate——SWEET-RL **48.2** vs Multi-Turn DPO 42.8(**+5.4**)。SWEET-RL 的 Llama-8B 匹配 Llama-3.1-70B(Backend 35.0 / Frontend win 39.8)、接近 GPT-4o(40.4 / 50.0)与 o1-mini。Fig.3b 数据缩放:SWEET-RL 初期需更多数据学可靠 critic,但很快追上并收敛更高(15k 样本达 41.6%)。【原文 Table 2, Fig.3b】
  - baseline 公平吗:同 backbone(Llama-3.1-8B)、同 offline 数据,对比 RAFT/Multi-Turn DPO + 闭源 GPT-4o/o1-mini——较公平。一个隐性优势=SWEET-RL critic 用了训练期特权信息(参考解),这是其前提也是相对其他方法的"信息不对称增益",论文已诚实说明(Fig.3a 标注此曲线非 test-time scaling)。
  - "看着强但没回答核心":提升幅度 +6% 属稳健而非数量级;且只在自建 ColBench 验证,跨 benchmark 泛化未证。
- 假设与失效边界:
  - 【原文】**最强假设=训练期可得 reference solution / outcome**(\(c\))。这是方法成立前提,也是外推真实场景的主要限制(真实协作"人"往往只有模糊想法、无清晰参考解)——论文承认但辩称"有清晰参考是合理假设"(lines 247-255)。
  - 【原文】offline 设定(不做在线人类交互)。
  - 【推断】ColBench 的"人类"用 LLM simulator 近似,且 simulator 依赖 reference artifact,模拟忠实度受 simulator 能力上限约束;Frontend 用 CLIP 相似度当奖励,语义保真度有限。
  - 【推断】只在两个 artifact-creation 任务验证,长 horizon(>10 轮)或非创作型多轮任务的有效性未知。
- 祛魅总结【推断】:真贡献=(a) 一个轻量但有效的多轮信用分配方案(非对称 critic + 直接学 advantage + log-prob 参数化 + BT),(b) 一个工程开销极低、可程序化扩展的开源多轮 RL benchmark(ColBench)。包装上"6% over SOTA + 匹配 GPT-4o"很亮眼,但 SOTA 多轮 RL 基线本身较弱、且 critic 吃了训练期特权信息。高估:把"匹配 GPT-4o"读成方法本身的通用强度(实为特定任务 + 特权信息下)。低估:ColBench + 15k 离线轨迹 + 评测器是对社区很实在的资产,可直接复用。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=turn-level advantage(由训练期特权信息 \(c\) 指导、BT 偏好目标),再当每轮 reward 给 actor | **改什么**=两个模型:先训 critic(advantage 函数),再用它训 actor 策略 | **何时改**=两阶段、offline(先训 critic 再训 policy,均不在线交互) | **免梯度?**=否,critic 与 policy 都是梯度训练;但 policy 改进不需在线 rollout(用离线轨迹 + critic 打分做 DPO) | **记忆-技能生命周期**=不涉及(无外部记忆/技能库) | **防遗忘机制**=无显式机制;用 LLM head 参数化 advantage 以"契合预训练"间接保泛化
- ⑦ 开源代码+框架/harness:https://github.com/facebookresearch/sweet_rl(ColBench + SWEET-RL 官方实现,已克隆);数据 https://huggingface.co/datasets/facebook/collaborative_agent_bench 。**框架=OpenRLHF 的定制 fork(`YifeiZhou02/collab_openrlhf`)**,改造支持 multi-turn DPO + length normalization;训练命令 `deepspeed --module openrlhf.cli.train_dpo`(DeepSpeed 后端);Frontend Design 评测需额外装 GeckoDriver + Firefox 渲染 HTML(已核对 README ✓)。仓库 `sweet_rl/{environments,models,utils}` + `scripts/{simulate_interactions, sample_best_of_n, rank_best_of_n, evaluate_code/html}`。
- 💰 资源/成本与可扩展性:ColBench 卖点之一就是低开销——只需 LLM API + 跑 Python/渲染 HTML 的包;>10k 程序化生成任务,难度可调、可扩展。训练用 8B/7B-VLM 量级,具体卡数原文未在正文详列。【原文 §3.1】
- 🎯 对"探索-巩固"对标:**支撑(思路同构)+ 可借资产**。一句判定:SWEET-RL 的"非对称 critic 用 teacher 特权信息(参考解/终局)做 step 级信用分配"与本项目"teacher 当稀疏脚手架、用其参考路径/foresight 给学生 path-selection 信用"高度同构,可作 asymmetric-critic 路线的代表 baseline;但它不涉及 MTP 前瞻、不涉及 on-policy 自生成 rollout(offline 设定),信用分配是 turn 级(非 token 级)。可借组件=(a) ColBench 多轮协作 benchmark + 数据可直接拿来评测;(b)"训练期特权信息→advantage→当 reward"的范式可移植到"teacher 参考路径→给学生每步信用"。缺口=offline 限制、需 reference solution、无 token 级/MTP、无记忆固化。【推断,依据 §4.3 信用分配范式与本项目 idea 的结构对应】
- 🔭 开放问题/未来方向:【原文】把方法推广到数学推理等其他"有训练期隐藏信息"的任务(reference solution 普遍存在,lines 501-504);更好地利用非对称信息做信用分配。【推断】放宽"需 reference solution"假设(真实协作常无清晰参考解);从 offline 扩到 online/on-policy;把 turn 级信用细化到 token/step 级(对标 TIP/本项目);更长 horizon 的稳定性验证。

〔本篇与既有 analysis/sweet_rl.md 核对:无事实冲突。本次增强:5 条核心公式(Eq 1/2/3/4 + POMDP 的 Q/V/A 定义)转 MathJax 并从 PDF 抄准;补 POMDP 问题形式与 offline 设定缘由;相关工作铺开多轮信用分配四条路线(单轮外推/value-TD/PRM/机器人非对称 AC)各自短板与精确Δ;方法流水线拆成两阶段、每阶段输入→输出 + chosen/rejected 采样细节(16 候选、top/bottom-50%)。无新增待核项。〕

raft_reinforce_rej | A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce (RAFT / RAFT++ / Reinforce-Rej) | Salesforce AI Research + UIUC(Wei Xiong & Hanze Dong 共一通讯;Jiarui Yao, Yuhui Xu, Bo Pang, Lei Wang, Doyen Sahoo, Junnan Li, Nan Jiang, Tong Zhang, Caiming Xiong) | 2025-06-12 v2 · arXiv preprint(arXiv:2504.11343v2)cs.LG | 主题线 L2(统一 SFT-RL)+L3(RLVR/GRPO)·相关性 高

**原始论文**:https://arxiv.org/abs/2504.11343

## 一眼看懂
- 🟦 TL;DR:大家默认"用上负样本的 RL(GRPO/PPO)一定比只用正样本的 SFT 类(拒绝采样 RAFT)强"。本文把 GRPO 从 reinforce-like 视角拆开做对照实验,发现:(1) 最简单的拒绝采样 **RAFT**(只在自己采样出的正确答案上做 SFT)与 GRPO 差距小得惊人(Qwen 上 RAFT++ 56.1 vs GRPO 56.3),早期还收敛更快;(2) GRPO 真正的增益不是来自奖励归一化(减均值/除标准差),而是来自**隐式丢掉了"整组全错"的 prompt**;(3) 据此提出 **Reinforce-Rej**——一个只过滤掉"全对"和"全错"两类 prompt 的极简 policy gradient,KL 效率和稳定性都更好(Qwen 56.4、LLaMA 28.5,≈GRPO)。核心立场:**样本选择 > 算法设计**,并呼吁未来用"更 principled 的负样本纳入机制",而非不加区分地用负反馈。【原文 Abstract 行 18-30,§5,§5.1】
- 最巧的一步:**"Remove all wrong"(去全错组)这一步**。抽掉它,GRPO 相对 vanilla Reinforce 的优势基本消失(§5.1 Figure 4:去掉全错样本带来 reward 最大增益,而去掉全对样本几乎没用)。本文核心论点(归一化无关、过滤才是关键)整个建立在这个对照上。直觉:全错组的负梯度"方差高、误导性强,会主导更新并带偏学习"(行 705-707)。

## 为什么做
- 研究背景:RLVR(可验证奖励的 RL)成为提升 LLM 数学推理的主流;PPO 是主导但需额外 critic(开销+复杂度),且 LLM 的**确定性转移**使方差较低、PPO 许多复杂组件在此设定下可能多余 → 催生"更简单 RL 算法"研究(ReMax/RLOO/GRPO/Reinforce++)。GRPO 因 DeepSeek-R1 走红被默认采用,但其算法细节"largely undocumented",成功到底来自算法本身还是路径依赖不清楚。【原文 §1 行 33-57】
- 解决的具体痛点:① GRPO 相对 vanilla Reinforce 的增益来源不明(**归一化** vs **隐式过滤**);② 普遍信念"负样本让 RL 远超只用正样本的 SFT"在作者初步实验里站不住(gap 很小,RAFT 早期还更快,前 100-200 步,行 68-70);③ 究竟是"算法设计"还是"样本选择"在起作用,缺系统拆解。【原文 §1 行 65-89】
- 相关工作 & 各自不足(§2,精确到机制):
  - **Reinforce 系简化**:ReMax(Li 2023)、RLOO(Ahmadian 2024/Kool 2019)、GRPO(Shao 2024)、Reinforce++(Hu 2025)——各做局部改动,但未系统回答"GRPO 为何强"。
  - **RAFT / 拒绝采样**(Anthony 2017、Dong 2023;亦即 rejection sampling fine-tuning):简单可解释、prior literature 一贯有好表现,但被当**二等基线**。近亲 **STaR**(Zelikman 2022)——也训自采 CoT,但每轮**从原始预训练模型重训**(非当前模型)、贪心解码只采 1 条、难题在 prompt 里给答案。
  - **偏好类**:Slic-HF(Zhao 2023)、DPO(Rafailov 2023)——优化成对对比目标;iter-DPO(Liu/Xiong/Xu/Dong 等)用中间 checkpoint 自采 on-policy 数据迭代提升。
  - **数据过滤**(RLHF/偏好优化):Yuan 2024、Dong 2024、Xiong 2024、Shen&Zhang 2024 丢除 top/bottom 之外候选降噪;Yu 2025a 把 reward+length 纳入过滤;推理任务常剔过易/过难题(Yang 2024b、Zhao 2024)——但这些**多是训练前一次性**剔题。**本文强调在线、逐轮按组(all-correct/all-wrong)动态过滤**(行 104-106),并揭示 GRPO 强表现与这种**隐式数据过滤**的联系。
- 动机链:GRPO 火但黑箱 → 默认"负样本=强"未经检验 → 做控制实验逐一隔离 GRPO 组件(负样本 vs 归一化)→ 发现过滤(尤其去全错)才是关键、归一化无关 → 提出最小变体 Reinforce-Rej + 重新抬高 RAFT 作为可解释基线。
- 与最近邻工作的Δ:vs Dr.GRPO/DAPO(也在改 GRPO 的归一化/clip)——本文不提新损失技巧,而用**消融拆穿"归一化贡献"这个被默认的来源**,把功劳归给"隐式数据过滤";vs 传统数据过滤——把一次性剔题改成**在线按组动态过滤**。差异关键点:把"算法 vs 数据"之争用对照实验定性,结论是"样本选择 > 算法设计"。【原文 §2 行 104-106,§5.1 行 717-720】

## 怎么做 + 靠不靠谱
- 符号(§3):policy \(\pi(a\mid x)\);二元奖励 \(r(x,a)\in\{-1,+1\}\)(verifier=Math-Verify 实现);每 prompt 采 \(n\) 个候选 \(a_1,\dots,a_n\),奖励 \(r_1,\dots,r_n\);token 级重要性比 \(s_t(\theta)=\dfrac{\pi_\theta(a_t\mid x,a_{1:t-1})}{\pi_{\theta_{\text{old}}}(a_t\mid x,a_{1:t-1})}\)。
- 方法流水线(以 Reinforce-Rej 为主线,输入→输出讲清):
  - **① 采样**:\(\pi_{\theta_{\text{old}}}\) 对每个 prompt 采 \(n=4\) 个回答;**② 打分**:Math-Verify 给二元 \(r\in\{-1,+1\}\);**③ 按组过滤**:丢弃"全对"和"全错"的 prompt 组(其余进训练);**④ 更新**:在保留样本上做带 importance-sampling+clip 的 token 级 policy gradient,**不做 std 归一化**;**⑤ 迭代**。输出=数学推理增强的 policy。
  - **各损失的真实形式**:
    - **RAFT(Eq.1)**:三步=数据采集→拒绝采样(只留最高奖励、通常 r=1 的正样本入 \(D\))→ 在 \(D\) 上最大化对数似然(纯 SFT):\(L_{\text{RAFT}}(\theta)=\sum_{(x,a)\in D}\log\pi_\theta(a\mid x)\)。
    - **Reinforce(token 级,Eq.5)**:\(L_{\text{Reinforce}}(\theta)=\dfrac{1}{|D|}\sum_{x,a\in D}\dfrac{1}{|a|}\sum_{t=1}^{|a|}\min\!\big(s_t(\theta)\,r(x,a),\ \operatorname{clip}(s_t(\theta),1-\epsilon,1+\epsilon)\,r(x,a)\big)\)(由 PG 目标 \(J(\theta)=\mathbb{E}_{x\sim d_0}\mathbb{E}_{a\sim\pi_\theta}[r(x,a)]\) 经重要性采样 Eq.3 + PPO clip 得到)。
    - **GRPO**:与 Eq.5 同形,但把 \(r(x,a)\) 换成组内标准化优势 \(A_t(x,a_i)=\dfrac{r_i-\operatorname{mean}(r_1,\dots,r_n)}{\operatorname{std}(r_1,\dots,r_n)}\)。
    - **RAFT++(Eq.6)**:给 RAFT 加上 importance-sampling + clip(把 RAFT 视为"在 replay buffer 上多步更新的 off-policy 混合算法"),并用指示函数 \(I\{r(x,a)=\arg\max_i r(x,a_i)\}\) 只在最高奖励(正)样本上训练:\(L_{\text{RAFT++}}(\theta)=\dfrac{1}{|D|}\sum\dfrac{1}{|a|}\sum_t \min\!\big(s_t(\theta),\operatorname{clip}(s_t(\theta),1-\epsilon,1+\epsilon)\big)\,I\{\cdot\}\)。
    - **Reinforce-Rej**:= "Reinforce + Remove both"(全对+全错组都过滤),在保留样本上用 Eq.5,**不加 std 归一化**。
  - **数据流动**:逐轮 on-policy(近 on-policy,多步 mini-batch + IS 修正);过滤在更新**之前**按组进行。
- 逐组件必要性(本文本身就是一篇大消融,§5.1 设计 6 个 Reinforce 变体对照,Figure 4):
  - **Remove all wrong(去全错)**:**最关键**——带来 reward 最大增益;没它则等同 vanilla Reinforce,增益消失(全错组负梯度高方差、误导)。【行 704-707】
  - **Remove all correct(去全对)**:单独"does not help much"。【行 707-708】
  - **Remove both(=Reinforce-Rej)**:entropy 更"well-behaved"、reward 略好,兼顾探索。【行 708-711】
  - **Mean-Zero(减均值)**:**有害**——抬高 KL、不涨 reward,潜在不稳。【行 712-713】
  - **Normalize Std(除标准差,在 Remove both 之上)**:"little additional gain"——证明归一化不是关键。【行 713-716】
  - **RAFT++ 的 clip**:**必要**——去掉 clip(只 IS 不 clip)反而比 vanilla RAFT 更差(Figure 2);作者反驳 Ahmadian 2024"clip 罕见故无用"的观点:clip 虽不频繁,但**一旦发生(\(\pi_\theta/\pi_{\theta_{\text{old}}}\) 远偏 1)其无界更新会严重违反 on-policy 假设、致不稳**(行 476-482)。
  - **clip-higher(\(\epsilon_1=0.2,\epsilon_2=0.28\) 非对称)**:把 Yu 2025b 的 clip-higher 用于 LLaMA,**稳住 policy entropy**、让 RAFT++ 后期回升、反超原 RAFT++(行 566-571)。
- 关键机制/公式(直觉):
  - RAFT=在"自采正例"上做 SFT;Reinforce 二元奖励可理解为"**在正例上 fine-tune + 在负例上 unlearning**";当负信号只由"最终答案对错"定义、太粗时,**unlearning 比 fine-tune 更不稳**——这是 RAFT 有时反超 vanilla Reinforce 的直觉(LLaMA 上 Reinforce 24.2 < RAFT++ 27.6)。【行 418-422】
  - **熵坍缩机制(§5.1)**:RAFT++ 只用正样本 → policy entropy **快速下降**(Qwen/LLaMA 一致);熵稳在低位后性能提升放缓(低熵=少探索、难产出多样推理路径);同时其对初始 policy 的 KL 早期增长更快(反映早期 test-acc 优势),但因缺持续探索很快 plateau,被 GRPO 反超。**负样本的作用=维持探索、防分布坍缩**——这是 RAFT++ 与 RL 法性能差距的可能根因(行 552-564)。
- 实验与证据(数字已对 Table 1 逐项核对):
  - 数据 **Numina-Math**(~860k 数学题,中国高中→国际奥赛);模型 **Qwen2.5-Math-7B-base** 与 **LLaMA-3.2-3B-instruct**;CoT prompt "Let's think step by step ... \boxed{}";评测 MATH500 / Minerva Math / OlympiadBench,指标 **avg@16**(T=1.0,4096 token)。**有意去掉 AIME**(只 30 题、噪声大,行 397-399)。框架 **veRL**。超参:AdamW lr 1e-6、1024 prompts/iter、n=4、mini-batch 512、max 4096 token(行 298-303)。
  - **关键数字(Table 1,Avg = MATH500/Minerva/Olympiad 三者均值)**:
    - **Qwen2.5-Math-7B-base**:Base **23.6** → RAFT **52.3**、RAFT++ **56.1**、iter-DPO **48.8**、Reinforce **53.9**、GRPO **56.3**、PPO **52.5**、**Reinforce-Rej 56.4**。即 Reinforce-Rej≈GRPO(且略高),RAFT++ 仅差 GRPO 0.2。
    - **LLaMA-3.2-3B-instruct**:Base **13.1** → RAFT 25.9、RAFT++ **27.6**、Reinforce **24.2**、GRPO **28.4**、PPO 26.9、**Reinforce-Rej 28.5**。注:此处 vanilla **Reinforce 24.2 < RAFT++ 27.6**(负信号太粗反伤)。
  - 学习曲线(Figure 1/2/3):RAFT++ 早期快、**~iter 100 后增速放缓**、被 GRPO 反超;Clip-Higher(\(\epsilon_2=0.28\))稳熵、让 RAFT++ 后期回升。
  - baseline 公平性:作者声明对所有算法"fully optimizing hyper-parameters"(含 batch/mini-batch/lr)调到各自最优,且共享 GRPO 脚本配置,公平性较好;用 avg@16 多基准、去 AIME 降单点噪声。【Table 1 caption 行 390-394】
  - 看着强但没回答的:① 规模偏小(3B/7B、单数据集数学域、4096 token 短上下文),"过滤>归一化"是否在更大模型/长 CoT/agentic 设定下成立未验证【推断】;② 只用二元答案级奖励,作者自陈"负样本太粗"是 unlearning 不稳的根因,但没给出更细粒度负信号方案(留作 future work)。
- 假设与失效边界:【原文】负样本由"最终答案对错"二值定义,粒度粗(行 418);任务限定数学推理、verifier 可靠。【推断】当一个 prompt 组里既无正例也无负例(全对/全错)本就被丢,说明方法依赖"组内有正有负"的混合;在极难/极易题占多数(rare-success 或饱和)时可用样本会锐减——这与 RESD 关注的 rare-success 退化是同一软肋;熵坍缩结论依赖"只用正样本"这一设定。
- 祛魅总结:真贡献=用干净对照把"GRPO 增益来源"从"归一化"重新归因到"隐式过滤(尤其去全错)",并把被低估的 RAFT 抬成强基线、给出极简 Reinforce-Rej。【推断】被高估的可能是"普适性"——结论很可能是 Qwen-Math/NuminaMath 这类"基座已会、RL 主要在收敛分布"场景的产物;被低估的是 RAFT 作为"自蒸馏/自采 SFT"的地位。它**没有**提出让模型学到新能力的机制(与 ProRL/rl_plus 关心的 capability boundary 正交)。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=verifier 二元答案奖励(±1),按 prompt 组聚合 | **改什么**=策略参数 \(\theta\)(on-policy/近 on-policy 更新) | **何时改**=在线逐轮,先过滤(去全对+全错组)后更新 | **免梯度?**=RAFT/RAFT++=纯 SFT 即免 RL 梯度(只对正例做 MLE);Reinforce-Rej=要 policy gradient,但比 PPO 少一个 critic | **记忆-技能生命周期**=无显式记忆/技能库,知识固化进参数 | **防遗忘机制**=无专门机制;靠 clip + 去全错维持熵/探索、间接防分布坍缩(熵坍缩=一种"遗忘多样性")
- ⑦ 开源代码+框架/harness:https://github.com/RLHFlow/Minimal-RL(Abstract 行 91、行 402 两处给出);框架 **veRL**(内置 verl 源码,入口 verl.trainer.main_ppo,FSDP+vLLM);脚本 run_raft / run_raftpp / run_reinforce_rej / run_grpo / run_ppo,vanilla 通过 policy_loss=vanilla 切换,RAFT++ 用 policy_loss=plusplus(脚本目录见 v1 元信息已核;本地 repos/raft_reinforce_rej 已克隆)。【行 91,行 289】
- 💰 资源/成本与可扩展性:原文未给 GPU 时/卡数。可推断的成本特征:RAFT/RAFT++ 省一个 critic 网络(比 PPO 轻),Reinforce-Rej 也无 critic;n=4 采样、1024 prompts/iter、mini-batch 512、lr 1e-6、max 4096 token(行 298-303)。卖点之一即"lightweight"。
- 🎯 对"探索-巩固"对标:**竞品偏分析、可借组件强**。与 TSRD 的关系:本文的"去全错组"对应"student 当前完全走不通的题先别学"(避免在无正例可锚的失败上做不稳定 unlearning),与"teacher 偏向给 student 能走通的开头"的探索直觉同向;"只在正例上 SFT(RAFT)→ 熵快速坍缩"反向印证了 TSRD 需要保留探索/负信号才能不过早巩固。可借:**在线按组过滤(all-correct/all-wrong gating)** 作为 OPD/path-recovery 的样本选择闸门;**clip-higher 稳熵**作长程训练防坍缩工具。缺口:本文负信号太粗(仅答案级),与 TSRD 想要的"token/步级 path-recovery 信号"差一个粒度——正是本文 future work 喊话的方向(更 principled / fine-grained 的负样本纳入)。判定:**强支撑性参考(样本选择维度),非直接竞品**。
- 🔭 开放问题/未来方向:【原文】呼吁设计"more principled / fine-grained 的负样本纳入机制",而非不加区分地用负反馈(Abstract 行 28-30、Conclusion)。【推断】把"全错组"进一步细分(部分步骤正确的负样本可作 path-recovery 监督)、在长 CoT/agentic/更大模型上验证"过滤>归一化"是否仍成立、把在线组过滤与 token 级 dense 信号(OPD teacher logprob)结合。

— RETURN —
raft_reinforce_rej | 读到PDF? 是(13页/44k字,§1-§5.1+Eq.1-6全核,Table 1 数字逐项核对一致) | L线 L2(SFT-RL统一)+L3(RLVR/GRPO机制拆解) | 对标结论:强支撑性参考(样本选择维度)——"在线按组过滤(去全对/全错)"作OPD/path-recovery样本闸门、clip-higher稳熵作防坍缩工具、"去全错组"同向"student走不通先别学";但负信号仅答案级太粗、无让模型学新能力的机制、非直接竞品,与TSRD想要的token/步级path-recovery信号差一个粒度(正是其future work方向) | 残留待核数:0(Table 1 所有数字逐项核对一致;Eq.1-6真实形式从PDF抄准;无组件消融以外的待核项)

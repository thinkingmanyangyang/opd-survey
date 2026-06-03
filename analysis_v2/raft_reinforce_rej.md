raft_reinforce_rej | A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce (RAFT / RAFT++ / Reinforce-Rej) | Salesforce AI Research + UIUC（Wei Xiong, Hanze Dong 共一通讯；Nan Jiang, Tong Zhang, Caiming Xiong） | 2025-06-12 v2 · arXiv preprint cs.LG | 主题线 L2(统一SFT-RL)+L3(RLVR/GRPO)·相关性 高

**原始论文**:https://arxiv.org/abs/2504.11343

## 一眼看懂
- 🟦 TL;DR:大家默认"用上负样本的 RL(GRPO/PPO)一定比只用正样本的 SFT 类(拒绝采样 RAFT)强"。本文把 GRPO 拆开做对照实验,发现:(1) 最简单的拒绝采样 RAFT(只在自己采样出的正确答案上做 SFT)与 GRPO 差距小得惊人,早期还收敛更快;(2) GRPO 真正的增益不是来自奖励归一化(减均值/除标准差),而是来自**隐式丢掉了"整组全错"的 prompt**;(3) 据此提出 Reinforce-Rej——一个只过滤掉"全对"和"全错"两类 prompt 的极简 policy gradient,KL 效率和稳定性都更好。【原文 Abstract,§5】
- 最巧的一步:**"Remove all wrong"这一步**。抽掉它,GRPO 相对 vanilla Reinforce 的优势基本消失(§5.1 图4:去掉全错样本带来 reward 最大增益,而去掉全对样本几乎没用)。本文的核心论点(归一化无关、过滤才是关键)整个建立在这个对照上。

## 为什么做
- 研究背景:RLVR(可验证奖励的 RL)成为提升 LLM 数学推理的主流;GRPO 因 DeepSeek-R1 走红被默认采用,但其算法细节"largely undocumented",成功到底来自算法本身还是路径依赖不清楚。【§1,L52-57】
- 解决的具体痛点:① GRPO 相对 vanilla Reinforce 的增益来源不明(归一化 vs 隐式过滤)。② 普遍信念"负样本让 RL 远超只用正样本的 SFT"在作者初步实验里站不住(gap 很小,RAFT 早期还更快)。③ 究竟是"算法设计"还是"样本选择"在起作用,缺系统拆解。【§1,L65-89】
- 相关工作 & 各自不足:Reinforce 系简化(ReMax/RLOO/GRPO/Reinforce++)各做局部改动但未回答"GRPO 为何强";RAFT/拒绝采样(Dong 2023)简单可解释但被当二等基线;数据过滤工作(RIP、Policy Filtration 等)多是**训练前一次性**剔题,而本文强调**在线、逐轮**过滤。【§2,L97-120】
- 动机链:GRPO 火但黑箱 → 默认"负样本=强"未经检验 → 做控制实验逐一隔离 GRPO 组件 → 发现过滤(尤其去全错)才是关键、归一化无关 → 提出最小变体 Reinforce-Rej + 重新抬高 RAFT 作为可解释基线。
- 与最近邻工作的Δ:vs Dr.GRPO/DAPO(也在改 GRPO 的归一化/clip)——本文不提新损失技巧,而是用**消融拆穿"归一化贡献"这个被默认的来源**,把功劳归给"隐式数据过滤";vs 传统数据过滤——把一次性剔题改成**在线按组(all-correct/all-wrong)动态过滤**。差异关键点:把"算法 vs 数据"之争用对照实验定性,结论是"样本选择 > 算法设计"。【§5.1,L89】

## 怎么做 + 靠不靠谱
- 方法流水线(以 Reinforce-Rej 为主线):① πθold 对每个 prompt 采 n 个回答;② verifier(Math-Verify)给二元 reward r∈{−1,+1};③ **按组过滤**:丢弃"全对"和"全错"的 prompt 组(其余进训练);④ 在保留样本上做带 importance-sampling+clip 的 token 级 policy gradient(式5),**不做 std 归一化**;⑤ 迭代。RAFT/RAFT++ 则是只保留正样本做 SFT(式1)、RAFT++ 额外加 IS+clip(式6)。【§3-§5.1】
- 逐组件必要性(本文本身就是一篇大消融,设计6个 Reinforce 变体对照,§5.1 图4):
  - Remove all wrong(去全错):**最关键**,带来 reward 最大增益;没它则等同 vanilla Reinforce,增益消失。【L704-708】
  - Remove all correct(去全对):单独几乎无用(reward "still not satisfactory")。【L708】
  - Remove both(=Reinforce-Rej):entropy/KL 更稳、reward 略好,兼顾探索。【L710-720】
  - Mean-Zero(减均值):**有害**——抬高 KL、不涨 reward。【L712-714】
  - Normalize Std(除标准差):在 Remove both 之上"minimal additional benefit"——证明归一化不是关键。【L714-716】
  - RAFT++ 的 clip:**必要**——去掉 clip(只 IS 不 clip)反而比 vanilla RAFT 更差(§5 图2),作者归因于 πθ/πθold 偏离 1 时无界更新破坏 on-policy 假设。【L475-482】
- 关键机制/公式(直觉):RAFT=在"自采正例"上做 SFT;Reinforce 二元奖励可理解为"在正例上 fine-tune + 在负例上 unlearning";当负信号只由"最终答案对错"定义、太粗时,unlearning 比 fine-tune 更不稳——这是 RAFT 有时反超 vanilla Reinforce 的直觉(LLaMA 上 Reinforce 24.2 < RAFT++ 27.6)。【L418-422,表1】
- 实验与证据:数据 NuminaMath(~860k);模型 Qwen2.5-Math-7B-base 与 LLaMA-3.2-3B-instruct;评测 MATH500 / Minerva / OlympiadBench,指标 avg@16(T=1.0,4096 token)。**关键数字(Qwen,表1)**:Base 23.6 → RAFT 52.3、RAFT++ 56.1、GRPO 56.3、Reinforce-Rej 56.4、PPO 52.5、iter-DPO 48.8。即 Reinforce-Rej≈GRPO,RAFT++ 仅差 0.2。LLaMA 上 Reinforce-Rej 28.5≈GRPO 28.4。学习曲线(图1/3):RAFT++ 早期快但 ~100 步后熵坍缩、被 GRPO 反超;Clip-Higher(ε2=0.28)能稳熵、让 RAFT++ 后期回升。【表1,§5,图1-3】
- baseline 公平吗:作者声明对所有算法"fully optimizing hyper-parameters"调到各自最优,且共享 GRPO 脚本配置,公平性较好;有意去掉 AIME(只30题、噪声大)用 avg@16 多基准,降低单点噪声。【表1 caption,L395-401】
- 看着强但没回答核心问题:规模偏小(3B/7B、单数据集数学域、4096 token 短上下文),"过滤>归一化"的结论是否在更大模型/长 CoT/agentic 设定下成立未验证【推断】;只用二元答案级奖励,作者自己也指出"负样本太粗"是 unlearning 不稳的根因,但没给出更细粒度负信号的方案(留作 future work)。
- 假设与失效边界:【原文】负样本由"最终答案对错"二值定义,粒度粗(L418);任务限定数学推理、verifier 可靠。【推断】当一个 prompt 组里既无正例也无负例(全对/全错)时本就被丢,说明方法依赖"组内有正有负"的混合,在极难/极易题占多数(rare-success 或饱和)时可用样本会锐减——这与 RESD 关注的 rare-success 退化是同一软肋;熵坍缩结论依赖"只用正样本"这一设定。
- 祛魅总结:真贡献=用干净对照把"GRPO 增益来源"从"归一化"重新归因到"隐式过滤(尤其去全错)",并把被低估的 RAFT 抬成强基线、给出极简 Reinforce-Rej。【推断】被高估的可能是"普适性"——结论很可能是 Qwen-Math/NuminaMath 这类"基座已会、RL 主要在收敛分布"场景的产物;被低估的是 RAFT 作为"自蒸馏/自采 SFT"的地位。它**没有**提出让模型学到新能力的机制(与 rl_plus 关心的 capability boundary 正交)。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=verifier 二元答案奖励(±1),按 prompt 组聚合 | **改什么**=策略参数 θ(on-policy/近 on-policy 更新) | **何时改**=在线逐轮,先过滤(去全对+全错组)后更新 | **免梯度?**=RAFT/RAFT++=纯 SFT 即免 RL 梯度(只对正例做 MLE);Reinforce-Rej=要 policy gradient,但比 PPO 少一个 critic | **记忆-技能生命周期**=无显式记忆/技能库,知识固化进参数 | **防遗忘机制**=无专门机制;靠 clip + 去全错维持熵/探索、间接防分布坍缩(熵坍缩=一种"遗忘多样性")
- ⑦ 开源代码+框架/harness:https://github.com/RLHFlow/Minimal-RL ;框架 **veRL**(内置 verl 源码,入口 verl.trainer.main_ppo,FSDP+vLLM);脚本 run_raft / run_raftpp / run_reinforce_rej / run_grpo / run_ppo,vanilla 通过 policy_loss=vanilla 切换,RAFT++ 用 policy_loss=plusplus(脚本目录见 v1 元信息已核)。【L91,L289】
- 💰 资源/成本与可扩展性:原文未给 GPU 时/卡数。可推断的成本特征:RAFT/RAFT++ 省一个 critic 网络(比 PPO 轻),Reinforce-Rej 也无 critic;n=4 采样、1024 prompts/iter、mini-batch 512、lr 1e-6、max 4096 token。【L298-304】卖点之一即"lightweight"。
- 🎯 对"探索-巩固"对标:**竞品偏分析、可借组件强**。与 TSRD 的关系:本文的"去全错组"对应"student 当前完全走不通的题先别学"(避免在无正例可锚的失败上做不稳定 unlearning),与"teacher 偏向给 student 能走通的开头"的探索直觉同向;"只在正例上 SFT(RAFT)→ 熵快速坍缩"反向印证了 TSRD 需要保留探索/负信号才能不过早巩固。可借:**在线按组过滤(all-correct/all-wrong gating)**作为 OPD/path-recovery 的样本选择闸门。缺口:本文负信号太粗(仅答案级),与 TSRD 想要的"token/步级 path-recovery 信号"差一个粒度——正是本文 future work 喊话的方向。判定:**强支撑性参考(样本选择维度),非直接竞品**。
- 🔭 开放问题/未来方向:【原文】呼吁设计"more principled / fine-grained 的负样本纳入机制",而非不加区分地用负反馈(Abstract、Conclusion L873-876)。【推断】把"全错组"进一步细分(部分步骤正确的负样本可作 path-recovery 监督)、在长 CoT/agentic/更大模型上验证"过滤>归一化"是否仍成立、把在线组过滤与 token 级 dense 信号(OPD teacher logprob)结合。

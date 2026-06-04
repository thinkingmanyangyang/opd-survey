nft | NFT: Bridging Supervised Learning and Reinforcement Learning in Math Reasoning | Tsinghua · NVIDIA · UIUC · Stanford(Huayu Chen, Kaiwen Zheng, Qinsheng Zhang, Ganqu Cui, Lifan Yuan, …, Jun Zhu*, Haoxiang Wang) | 2026-03-01 · arXiv 2505.18116 v3 · ICLR 2026 | 主题线 L2(统一 SFT-RL/GFT 类)，兼 L3(RLVR/GRPO) · 相关性 高

**原始论文**：https://arxiv.org/abs/2505.18116

## 一眼看懂
- 🟦 TL;DR：人们普遍认为"从错误中自我反思"是 RL 的专利、监督学习(SL)只会记正样本。本文用一个 Bayes 分解反驳：生成策略 `\(\pi\)` 可拆成"正策略 `\(\pi^+\)`"与"负策略 `\(\pi^-\)`"，且二者被一条线性关系锁死 `\(r_q\,\pi^+ + (1-r_q)\,\pi^- = \pi_{\text{old}}\)`(`\(r_q\)`=该题正确率)。于是把**负策略隐式重参数化为目标正策略** `\(\pi^-_\theta=\frac{\pi_{\text{old}}-r_q\,\pi^+_\theta}{1-r_q}\)`——对负样本做最大似然，等于在直接优化 `\(\pi^+_\theta\)`。这样纯 SL(取名 NFT,Negative-aware Fine-Tuning)也能"利用负样本"，全程只存一份模型。理论上证明 **on-policy 时 NFT 的梯度与 GRPO 完全相等**(Prop.4.2)，实证上 7B 略超 DAPO(51.7 vs 51.2)、32B 与 DAPO 基本持平(59.2 vs 59.9)。【原文 §Abstract/§3.2/§4/Table 1】
- 最巧的一步：**隐式负策略重参数化**(Eq.7 的耦合 → `\(\pi^-_\theta\)` 用 `\(\pi^+_\theta\)` 表达)。抽掉它就退回 RFT(只在正样本上微调、丢掉所有负样本)——而 RFT 正是被诟病"学不会反思"的那个 baseline。正是这一步让"在负样本上算 loss"可以反向传播去**塑造正策略**，从而把负样本的信息榨出来。【原文 §3.2/Theorem 3.1】

## 为什么做
- 研究背景：LLM 数学推理近年的跃升，靠的是从"模仿(用参考答案 SFT)"转向"自我改进(只需题目 + 二元 verifier 判对错，模型自采样自纠错)"。RL(PPO/GRPO/DAPO)被视作这套 verification-driven 训练的天然载体。【原文 §1】
- 解决的具体痛点：① SL 一侧的代表 RFT(拒绝采样微调:采样→verifier 留对的→在对的上 SFT)**完全丢弃负样本**，被普遍当作"SL 落后 RL"的根因；② "自我反思是 RL 专属、SL 天生不能从错误学"这一论断缺乏理论澄清；③ SL 与 RL(尤其 GRPO)之间到底是什么关系不清楚。【原文 §1/§3.1 Discussion】
- 相关工作 & 各自不足(站谁肩上 + 精确差异):
  - **策略梯度族**(Sutton 1999 PG → Schulman 2017 PPO/重要性采样 → Shao 2024 GRPO):本文式(1)-(4)正是从 PG 经 IS 推到 GRPO 的标准链；GRPO 用组内归一优势 `\(\hat A_{q,a}=(r-\text{mean}\{r_{1:K}\})/\text{std}\{r_{1:K}\}\)`(式4)免 critic,DAPO(Yu 2025)进一步去 KL/熵正则、解耦 clip。它们**能用负样本但被当作"RL 才有的本事"**——本文的 Δ 是从纯 SL 复刻这个能力并证明梯度等价。
  - **RFT / 拒绝采样**(Dong 2023, Yuan 2023, Xiong 2025):只强化已会的正样本、不碰负样本——这是 NFT 要超越且要"统一进来"的 SL baseline。
  - **隐式模型重参数化的思想源头**:DPO(Rafailov 2023,用策略网络隐式参数化奖励模型)、扩散/视觉生成里"用生成网络隐式定义条件/残差模型"——**思想同源**(都靠隐式定义的模型做直接优化),但**没人把它用到"从负样本反推正策略"**。
  - **Dr.GRPO**(Liu 2025b):建议去掉式(4)的 std 项——本文 `\(\omega(q)=1-r_q\)` 这一权重选择正好与之对齐。
  - 共性缺口:没人在**纯 SL 框架内**实现可与 RL 比肩的负样本利用。【原文 §1/§2.1/§3.1】
- 动机链：现状("自我改进 = RL 专属"成主流叙事，SL 只配做模仿)→ 质疑(SL 真不能从负样本学吗? 若能，SL 与 RL 的差距是不是只来自"会不会用负样本"而非 RL 本身更优?)→ 数学发现(Bayes 把 `\(\pi\)` 拆成 `\(\pi^+/\pi^-\)`，二者线性耦合 → 学负样本可塑造正策略)→ 所以(造 NFT:隐式负策略 + 正样本 MLE,纯 SL 也能反思)。【原文 §1/§3】
- 与最近邻工作的 Δ：相对 **RFT**——NFT 多了"在负样本上做隐式负似然比的 MLE"这一项(Eq.9/10 的 `\((1-r)\)` 分支),差在**把负样本从"丢弃"变成"可监督信号"**(消融:32B 上 RFT 贡献总增益 ~80%、负样本贡献剩余 ~20%)。相对 **GRPO/DAPO**——NFT 从 MLE 出发、GRPO 从 PG 出发，但 Prop.4.2 证明 on-policy 下二者梯度**恒等**；唯一差异在 off-policy:GRPO 把偏离 `\(\pi_{\text{old}}\)` 的梯度**硬置零**(clip 的指示函数),NFT 用**软衰减**(Fig.4 的 `\(1/R^t_\theta\)`)。【原文 §4/Prop.4.1-4.2/§5.3】

## 怎么做(到可复现)
### 总体流水线(Algorithm 1,在线迭代;输入→输出)
输入:LLM `\(\pi\)`、题集 `\(\{q_{1:N}\}\)`、二元 verifier `\(r(\cdot)\in\{0,1\}\)`。每轮:
1. **数据收集**:对每题采 `\(K\)` 个答案 `\(a_{1:K}\sim\pi\)`、verifier 判对错 `\(r_{1:K}\)`;**预存**每题正确率估计 `\(\hat r_q=\text{mean}\{r_{1:K}\}\)` 与 token 级旧似然 `\(\{\pi_{\text{old}}(a_t|q,a_{<t})\}\)`(单模型省内存的关键:`\(\pi_{\text{old}}\)` 在采样时一次性记下,无需保 ref 模型)。
2. **prompt 过滤**:只留 `\(0<r_q<1\)` 的题(全对/全错优势=0、无梯度信息)。
3. **构造 token 级似然比**:正样本 `\(R^t_\theta=\frac{\pi^+_\theta(a_t|q,a_{<t})}{\pi_{\text{old}}(a_t|q,a_{<t})}\)`;若该样本为负(r=0)则替换为**隐式负似然比** `\(R^t_\theta\leftarrow\frac{1-\hat r_q R^t_\theta}{1-\hat r_q}\)`;再用 **straight-through max** 把它裁到下界 `\(\epsilon\)`(保对数参数为正、梯度可回传)。
4. **最大似然更新** `\(\theta\leftarrow\theta+\lambda\nabla_\theta\sum_t\log R^t_\theta\)`,按 prompt 权重 `\(\omega(q)\)` 给难题(低 `\(r_q\)`)更高权重。
5. `\(\pi\leftarrow\pi^+_\theta\)`,进入下一轮。

### 核心数学(真实形式 + 直觉)
- **正/负策略的 Bayes 定义(式5-6)**:
\[
\pi^+(a|q):=\pi(a|q,r{=}1)=\frac{\pi_{\text{old}}(a|q)\,p(r{=}1|q,a)}{\sum_A \pi_{\text{old}}(a|q)\,p(r{=}1|q,a)},\quad
\pi^-(a|q):=\pi(a|q,r{=}0)=\frac{\pi_{\text{old}}(a|q)[1-p(r{=}1|q,a)]}{\sum_A \pi_{\text{old}}(a|q)[1-p(r{=}1|q,a)]}.
\]
- **守恒/耦合式(式7,全文枢纽)**:
\[
r_q\,\pi^+(a|q)+(1-r_q)\,\pi^-(a|q)=\pi_{\text{old}}(a|q),\qquad r_q:=\sum_A \pi_{\text{old}}(a|q)p(r{=}1|q,a)=p(r{=}1|q).
\]
直觉:某题你采 `\(K\)` 个答案、正确率 `\(r_q\)`,那么"全部生成"= `\(r_q\)` 份"正确生成分布 `\(\pi^+\)`" + `\((1-r_q)\)` 份"错误生成分布 `\(\pi^-\)`"。既然 `\(\pi\)` 和 `\(r_q\)` 已知,`\(\pi^+\)` 和 `\(\pi^-\)` 就**一损俱损:学 `\(\pi^-\)` 等于反着学 `\(\pi^+\)`**——错误答案不是噪声,它和正确答案是同一枚硬币的两面。
- **隐式负策略重参数化**:由式7解出
\[
\pi^-_\theta(a|q):=\frac{\pi_{\text{old}}(a|q)-r_q\,\pi^+_\theta(a|q)}{1-r_q}.
\]
- **Theorem 3.1(可行性保证)**:对 `\(\pi^-_\theta\)` 做 MLE,
\[
\max_\theta \mathbb{E}_{p(q)\pi^-(a|q)}\big[\log\pi^-_\theta(a|q)\big]\Leftrightarrow \min_\theta \Big(-\mathbb{E}_{(q,a)\sim D^-}\log\tfrac{\pi_{\text{old}}(a|q)-r_q\pi^+_\theta(a|q)}{1-r_q}\Big),
\]
在**无限数据 + 无限模型容量**下其最优解恰为真正策略:`\(\pi^+_{\theta^*}(a|q)=\pi^+(a|q)\)`。即"只在负样本上学也能学到正策略"。
- **NFT 损失(式9,正+负合一;减去与 θ 无关的 baseline `\(-\log\pi_{\text{old}}\)`,初始时 loss=0)**:
\[
L^{\text{NFT}}_{(a,q,r)\sim D}(\theta)=r\Big(-\log\tfrac{\pi^+_\theta(a|q)}{\pi_{\text{old}}(a|q)}\Big)+(1-r)\Big(-\log\tfrac{1-r_q\frac{\pi^+_\theta(a|q)}{\pi_{\text{old}}(a|q)}}{1-r_q}\Big).
\]
- **实用目标(式10,token 级 + 难度加权 + 负似然比裁剪)**:
\[
L^{\text{NFT}}_D(\theta)=-\sum_{q,a,r}\omega(q)\sum_t\Big[r\log R^t_\theta(q,a)+(1-r)\log\,\text{max}_v\!\Big(\tfrac{1-\hat r_q R^t_\theta(q,a)}{1-\hat r_q},\,\epsilon\Big)\Big],\quad R^t_\theta=\tfrac{\pi^+_\theta(a_t|q,a_{<t})}{\pi_{\text{old}}(a_t|q,a_{<t})}.
\]
其中 `\(\text{max}_v(x,\epsilon)=\text{stopgrad}[\max(x,\epsilon)-x]+x\)`(直通梯度:前向裁到 `\(\epsilon\)`,反向仍按 `\(x\)` 流)。

### 逐组件必要性
- **隐式负策略重参数化(核心)**：没它→退回 RFT,负样本白丢。Theorem 3.1 保证理想条件下最优解=真 `\(\pi^+\)`。【原文 §3.2/Theorem 3.1】
- **token 级 loss(不是序列级)**：序列似然 `\(\propto\)` 答案长度、方差大、数值不稳;改为把每个 token 当独立单元求和(承袭 PPO/DAPO)。**无单独消融**(作为标准设计直接采用)。【原文 §3.3 "Token-level loss"】
- **负似然比 straight-through 裁剪(下界 `\(\epsilon\)`)**:负 loss 含 log,其参数 `\((1-\hat r_q R^t_\theta)/(1-\hat r_q)\)` 必须 `\(>0\)`;未优化时可能变负→训练坍缩。强制下界 `\(\epsilon\)`、用直通梯度保流。仓库默认 `clamp_negative=1.0`(即 `\(\epsilon=1.0\)`)。没它→可能崩。【原文 §3.3 "Clipping" + 仓库 train_7B.sh】
- **prompt 加权 `\(\omega(q)\)`**：给难题(低 `\(r_q\)`)更高权重;选 `\(\omega(q)=\sqrt{(1-r_q)/r_q}\)` 时正好让 NFT 梯度对齐 GRPO,选 `\(\omega(q)=1-r_q\)` 对齐 Dr.GRPO。**有消融**(Fig.9):三种权重(1 / `\(1-r_q\)` / `\(\sqrt{(1-r_q)/r_q}\)`)性能相近,`\(\sqrt{}\)` 形略好。【原文 §3.3/§5.4/Fig.9】
- **prompt 过滤(去全对/全错)**：advantage=0、无梯度,留着浪费算力。标准 GRPO 式做法。【原文 Algorithm 1 L8】

### NFT=GRPO 的梯度证明(Prop.4.1,直觉)
取 `\(\omega(q)=\sqrt{(1-\hat r_q)/\hat r_q}\)`,定义归一优势 `\(A^+_q=\sqrt{(1-\hat r_q)/\hat r_q}\)`、`\(A^-_q=-\sqrt{\hat r_q/(1-\hat r_q)}\)`,则
\[
\nabla_\theta L^{\text{GRPO}}_D=-\sum\Big\{rA^+_q\,\mathbb{I}[R^t_\theta<1{+}\epsilon']+(1{-}r)A^-_q\,\mathbb{I}[R^t_\theta>1{-}\epsilon']\Big\}\nabla_\theta R^t_\theta,
\]
\[
\nabla_\theta L^{\text{NFT}}_D=-\sum\Big\{rA^+_q\cdot\tfrac{1}{R^t_\theta}+(1{-}r)A^-_q\cdot\big(\text{max}(\tfrac{1-\hat r_q R^t_\theta}{1-\hat r_q},\epsilon)\big)^{-1}\Big\}\nabla_\theta R^t_\theta.
\]
**唯一差异在 off-policy 时的梯度权重**(Fig.4):on-policy(`\(R^t_\theta=1\)`)处二者**完全相同**;偏离时 GRPO 用指示函数**硬置零**(clip),NFT 用 `\(1/R^t_\theta\)` 等**软衰减**。连 GRPO 那个"看似经验技巧"的组内归一化(标准化优势)在 NFT loss 里都是**自动冒出来的**(因为 `\(\sqrt{(1-r_q)/r_q}\)` 这个权重 = 归一化优势的尺度)。直觉:"SL 和 RL 在二元反馈下其实是一回事,差别只在如何处理 off-policy 漂移。"【原文 §3.2/§4/Fig.4】

## 靠不靠谱
- 实验与证据:
  - **算法对比(Table 1,同数据/同基础设施/同通用超参)**：7B(Qwen2.5-Math-7B base 31.6)——NFT 51.7 > DAPO 51.2 > Dr.GRPO 49.8 > GRPO 49.5 > DPO 48.9 > RFT 48.3。32B(base 29.6)——DAPO 59.9 ≳ NFT 59.2 ≫ RFT 52.8。即 7B 略超 SOTA RL、32B 基本持平(略低 0.7)。【原文 Table 1/§5.2】
  - **训练动态(Fig.6/Fig.8)**：NFT 收敛速度与最终性能与 DAPO 相当(3-4 次独立运行报 mean±std);**熵曲线**——RFT 随训练熵下降(越练越保守),而 NFT 与 DAPO 都熵上升(更多探索),被作者认为是 NFT>RFT 的潜在原因。【原文 §5.2-5.3/Fig.6/Fig.8】
  - **负样本贡献(Fig.7,32B)**：正样本(RFT 部分)贡献最优模型总增益 ~80%,负样本贡献剩余 ~20%,且**模型越大负样本越重要**——作者解释:大模型已记得够好,"反思错误"成了新瓶颈。【原文 §5.3/Fig.7/Fig.11】
  - baseline 公平吗:所有算法用**相同 DAPO-Math-17k 训练集、相同基础设施、相同通用超参**,且 NFT 实现直接 fork DAPO 环境、继承其绝大多数超参,归因清晰——本文方法学最扎实处。
  - "看着强但没回答核心问题":核心问题"SL 能否匹敌 RL"被正面回答(能);但**性能本身不是卖点**(32B 还略低),真贡献是"统一视角",论文坦诚。
- 假设与失效边界:
  - 【原文 §3.2/Theorem 3.1】等价/最优性证明依赖"**无限数据 + 无限模型容量**"理想假设。
  - 【原文 §4/Prop.4.2】"NFT=GRPO"严格只在 **on-policy(`\(R^t_\theta=1\)`)** 成立;off-policy 下按不同裁剪策略分道扬镳,NFT 软衰减**无收敛性理论保证**(纯经验)。
  - 【原文 §3.2 "Continuous Reward"】NFT 可直接推广到连续奖励 `\(r\in[0,1]\)`(Appendix B 证收敛不变),但**正文实验只做二元奖励 + 整数答案数学题**。
  - 【推断】仅在**数学推理 + 答案可二元验证**场景验证;对开放式/不可验证/非二元奖励任务是否成立未实测。依据:训练集 DAPO-Math-17k 全是整数答案数学题、6 个评测基准全是数学。
  - 【推断】负似然比裁剪下界 `\(\epsilon\)` 是数值稳定关键超参,取值不当(过小)可能仍坍缩;论文给默认值但未系统扫 `\(\epsilon\)` 稳健区间。
- 祛魅总结【推断】：
  - 真贡献:**理论统一**——Bayes 分解 → 隐式重参数化 → Theorem 3.1 → on-policy 梯度等价(Prop.4.2)这条链清晰、有解释力,并顺带"解释了 GRPO 组内归一化为何 work"(它在 SL 视角下是自动出现的归一化优势)。把"SL vs RL"之争收敛为"是否利用负样本 + 如何处理 off-policy 漂移"两个具体维度。"负样本利用是 RFT↔RL 差距主因"有受控实验支撑。
  - 包装/被高估处:**"self-reflection(自我反思)"一词偏宣传**——本质是"在负样本上做带符号的隐式似然更新",与人直觉的"反思"语义不完全对应。**性能增益有限**:32B 仅与 DAPO 持平甚至略低;等价性是 on-policy 的、依赖理想容量假设,off-policy 行为仍是经验性的。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：verifier 的二元 0/1 正误标签(无奖励模型、无教师 logits);正样本走正似然比、负样本走隐式负似然比,本质是带符号的最大似然。
  - **改什么**：只改**训练目标/loss**(RFT 的纯正样本 MLE → NFT 的正+隐式负 MLE);模型参数全量更新,单模型。
  - **何时改**：在线迭代——每轮自采样 K 答案→过滤→更新→`\(\pi\leftarrow\pi^+_\theta\)` 进入下一轮(on-policy 收集 + 可 off-policy 多步更新)。
  - **免梯度?**：否,是梯度优化(最大似然反传);但**不需要 critic/value 模型、不需要 reference 模型**(单模型,`\(\pi_{\text{old}}\)` 似然在采样时预存),内存开销极小。
  - **记忆-技能生命周期**：无外部记忆/技能库;"技能"=数学推理能力,靠在线 SL 在 base 已有能力上放大并固化进参数。无显式持续学习机制。
  - **防遗忘机制**：无显式防遗忘(且明确去掉 KL/熵正则,对齐 DAPO);靠 on-policy 自采样使分布不至剧烈漂移,但不针对灾难性遗忘。
- ⑦ 开源代码+框架/harness：https://github.com/NVlabs/NFT (完整开源,本地已 clone)。核心文件 `main_nft.py`、`ray_nft_trainer.py`、`dp_actor.py`(NFT loss)、`experience_maker.py`、`verifier.py`;脚本 `train_7B.sh`/`train_32B.sh`、`eval_local_7B/32B.sh`、`download_data.sh`。**框架=VeRL**(volcengine/verl,fork 自官方 DAPO 环境、固定 commit)+ FSDP + Ray。关键超参(train_7B.sh 实测):`neg_weight=1.0`(1.0=NFT / 0.0=RFT / −1.0=DAPO;32B 建议试 0.5)、`normalize=1`(1=Dr.GRPO 对齐 / 2=标准 GRPO 对齐)、`ratio_type="token"`、`clamp_negative=1.0`/`clamp_positive=0.0`(隐式负/正似然比的 straight-through 裁剪下界 `\(\epsilon\)`)、`lr=1e-6`、`clip_ratio_low/high=0.2/0.28`。权重 HF nvidia/NFT-7B、nvidia/NFT-32B。代码可得性高。【v1 仓库核查 + 本批 repo ls + train_7B.sh】
- 💰 资源/成本与可扩展性：7B 用 4×8 H100,32B 用 16×8 H100;约 5,000 梯度步、batch 512、生成温度 1.0、训练集 DAPO-Math-17k。单模型(无 critic/ref)使内存友好。【原文 §5.1 + 仓库核】
- 🎯 对"探索-巩固"对标：**中等支撑(巩固层的统一信号视角)+ 可借的"负样本=可监督信号"机制;非脚手架/回轨范式**。判定依据:① **巩固对标**——NFT 的"在负样本上做隐式 MLE 去塑造正策略"与本课题"巩固=把走通的有效路径固化进参数"高度同构:它把"压低错误分支"重写为"抬高正确分支",是一种**单模型、免 teacher 的自巩固**。② **探索对标**——熵曲线证据(NFT 像 RL 一样熵上升、RFT 熵下降)支撑"利用负反馈→更多探索"。**可借组件**:Bayes 耦合 `\(r_q\pi^++(1-r_q)\pi^-=\pi_{\text{old}}\)` 给"path-recovery(走偏后回轨)"提供干净数学接口——把"恢复分支"看作 `\(\pi^+\)`、"走偏分支"看作 `\(\pi^-\)`,用隐式重参数化在不引入 teacher 的情况下从失败轨迹学回轨信号;`neg_weight` 旋钮可调"对失败分支的惩罚强度"。**缺口**:① **无 teacher 脚手架/无蒸馏**(纯 self-improvement SL);② 信号是**序列/token 级二元正误**,非"关键步稀疏接管"——没有"只在走偏那一步介入"的稀疏性;③ **无 MTP/前瞻、无记忆/技能库**;④ on-policy 等价性是它与 OPD 的桥(opd_survey §7.3 把 NFT/DPO 归为同族),但 NFT 本身不做学生轨迹上的教师 KL。一句话:**它是"用负样本自巩固"的极简基座,可作回轨信号的数学接口,但不提供脚手架与前瞻。**
- 🔭 开放问题/未来方向：
  - 【原文】把 NFT 推广到连续奖励/非二元反馈(Appendix B 给理论缺实验);理解 off-policy 软衰减裁剪的最优性;为何大模型更依赖负样本(§5.3 给观察未给理论)。
  - 【推断】把"隐式负策略"嫁接到 OPD:在学生自采样轨迹上,用教师 KL 定义"正/负方向"再做隐式重参数化,得到"teacher 引导的负样本利用"(介于 NFT 与 on-policy 蒸馏之间);把 `\(\omega(q)\)` 难度加权换成"关键步/高熵 token 加权",向稀疏脚手架靠拢。

RETURN: nft|读到PDF=是(§Abstract/§1-4全文+Eq.1-10/Theorem3.1/Prop.4.1-4.2/Algorithm1/Table1/Fig.4/Fig.6-9)|L线=L2(兼L3)|对标=中等支撑(负样本隐式重参数化=单模型自巩固,Bayes耦合可作回轨数学接口,on-policy时=GRPO是与OPD的桥;但无teacher脚手架/无稀疏关键步/无MTP)|残留待核=0

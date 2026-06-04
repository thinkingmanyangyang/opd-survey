entropy_mechanism | The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models | 上海 AI Lab / 清华 / UIUC / 北大 / 南大 / CUHK（PRIME-RL 团队，Ganqu Cui、Yuchen Zhang、Jiacheng Chen 等共同一作；Ning Ding/Bowen Zhou/Yu Cheng 通讯） | 2025-05-28 (arXiv:2505.22617 v1，预印本) | L3 RLVR/GRPO · 相关性高

**原始论文**：https://arxiv.org/abs/2505.22617

## 一眼看懂

> 一句话导读：RLVR 训推理模型时,策略熵(衡量模型还愿不愿意探索)会很快塌到接近 0,模型从此过度自信、不再探索,性能也就到顶。本文先发现"性能上限可由熵预测"的经验定律,再从优化动力学上把塌熵的根因锁定到一个量——协方差,然后只动极少数"关键 token"就能续命探索。

- 🟦 TL;DR：在可验证奖励 RL(RLVR)训练推理模型时,策略熵会在头 200 步就急剧坍缩到接近 0;模型随之变得过度自信、再也探索不动,性能也跟着触顶。
  - 而且这个上限是**可预测**的:验证性能 \(R\) 与熵 \(H\) 满足一条经验定律 \(R=-a\exp(H)+b\),当 \(H=0\) 时 \(R=-a+b\) 就是天花板【原文 Abstract, §2.4, Fig.1】。
  - 本文进一步从动力学上证明:相邻两步的熵变 \(\approx -\eta\,\mathrm{Cov}(\log\pi(a),\,\pi(a)A(a))\),也就是"动作 log 概率"与"它的 logit 变化"之间的协方差;而在策略梯度下,logit 变化 \(\propto A\)(优势)。
  - 于是提出 **Clip-Cov / KL-Cov** 两个干预——只动一小撮"高协方差 token",就能持续维持探索、突破熵瓶颈【原文 Abstract, §3, §4.2, Listing 1】。
- 最巧的一步：把"调熵"精准化到**协方差**这一个量上。
  - 抽掉"只干预高协方差 token"这步、退回对全局加熵正则/KL 正则——论文 §4.1 实测朴素正则要么压不住坍缩、要么过度扰动优化(总之失效)。
  - 而 Clip-Cov/KL-Cov 只干预 \(2\times10^{-4}\sim2\times10^{-3}\) 比例的 token,就能彻底改变熵曲线,说明少数"pivotal token(关键 token)"主宰了 LLM 的熵(Table 1:Top 0.02% token 的平均协方差是 5.654,全体均值才 0.003,差约 1800 倍)。
  - 所以命门就是"用协方差当阈值、只动 pivotal token"【原文 §4.1, §4.2 Table 1, §4.5】。

## 为什么做

> 一句话导读：传统 RL 一直靠熵正则/KL 正则主动控熵保探索;但在 LLM RLVR 上这套被证明不灵——本文要回答"熵到底为什么塌、塌到哪、怎么按根因去救"。

- 研究背景：RLVR 被视作后训练算力的下一增长极(在数学/代码等有客观奖励的任务上持续提升推理)。RL 的核心是探索-利用权衡(Sutton 1988),而策略熵正好度量探索潜力;传统 RL 把"抑制熵下降"当作多数算法的必备(Williams&Peng 1991、Williams 1992、Eysenbach&Levine 2021),常用熵正则/KL 正则来主动控熵(Ziebart 2008、Schulman 2017b、Haarnoja 2018)【原文 §1, §5】。
- 解决的具体痛点(三条):
  - ① LLM 策略熵的典型行为缺乏定量刻画——大家都见过"训练后期过度自信、探索枯竭、性能饱和",但没人总结出规律;
  - ② **熵坍缩**:无干预时熵早期就急降到 ~0、性能触顶,意味着再投算力边际收益趋零,这直接限制了"扩 RL 算力"的价值;
  - ③ 朴素的熵/KL 正则在 LLM RLVR 上被证明无效(§4.1:熵正则系数太小则无效、取 0.01 则熵爆炸、取 0.005 能稳但超不过 baseline;KL 正则虽然稳住了熵,分数反而降)【原文 §1, §2.3, §4.1】。
- 相关工作 & 各自不足(来龙去脉):
  - ① **最大熵 RL / 熵正则**(Ziebart08 IRL、Haarnoja18 SAC、Schulman17b PPO 的熵 bonus)——在传统 RL 里是抗早熟、保探索的必备手段,但本文 §4.1 证明朴素照搬到 LLM RLVR 无效(要么超参敏感、要么损性能)。这是本文最直接要打破的"默认做法"。
  - ② **RL for LLM 里"要不要保留熵正则"的分歧**——Ouyang22(InstructGPT)保留了熵 bonus;而近期 RLVR 主力(Shao24 GRPO/DeepSeekMath、He25、Cui25 PRIME、Hu25、Yu25 DAPO、Liu25)多直接去掉了熵/KL 项。本文为"为什么去掉它们也没解决坍缩"给出机理解释。
  - ③ **RL 可预测性 / Scaling**——此前只在非 LLM agent 上研究过"策略性能 vs 算力"的 scaling(Hilton23、Rybkin25 ProRL);Gao22 用 KL 预测 reward 建模的 RLHF over-optimization(与本文"用熵预测性能"思路一致,还共享"系数随模型尺寸 log-linear"的发现);但 LLM RLVR 的 entropy-performance 可预测性此前无人系统建立。
  - ④ **clip-higher 类调熵 trick**(DAPO,Yu25)——把 PPO 上界 \(\epsilon\) 从 0.2 调到 0.28,间接抬熵;本文把它解释为"实际是在加低协方差 token",并把它当作最强 baseline 直接对比。
  - ⑤ **"RL 只是激发 base 已有行为、打不破天花板"之争**(Yue25)——本文 §2.6 给的是条件性支持:若发生熵坍缩,则天花板确实存在且可预测,但论证这天花板来自"熵机制",而非 RL 本身的内禀限制。
- 动机链:熵坍缩会可预测地锁死性能上限 → 那就应当:(1) 给它建一条像 Scaling Law 那样可外推的经验定律(用早期/小模型预测终态),(2) 从优化动力学里找坍缩根因,(3) 据此设计能持续注入探索、突破熵瓶颈的可扩展方法【原文 §1, §2】。
- 与最近邻工作的Δ(精确差异):最近邻是 clip-higher(DAPO)与各种熵正则 trick。
  - 本文的 Δ:不只给一个 trick,而是建立"经验定律→动力学根因(协方差)→按根因设计干预"的完整因果链;并把 clip-higher 解释为"实际是在加平均协方差 ~−0.03 的低协方差 token",而本文是直接用协方差当阈值,控熵更精准、还能双向(既能抬也能压)【原文 §4.5】。
  - 为什么这更有用:理论→方法→效果的闭环,比单报一个 trick 更有说服力,而且能定量预测上限。

## 怎么做（细到可复现）

> 一句话导读：这节分三块——先给"性能-熵"的经验定律(§1),再推导"熵为什么塌"的动力学根因(§2,落到协方差),最后据此给两个只改几行 loss 的干预(§3)。公式不少,但主线只有一个量:协方差。

### 0. 记号与目标（§2.1）
- 输入 prompt \(x\),LLM 策略 \(\pi_\theta\) 自回归生成 \(y=\{y_1,\dots,y_T\}\),verifier 给奖励 \(r(y)\)。目标是 \(\max_\theta J(\theta)=\mathbb{E}_{x\sim D,\,y\sim\pi_\theta(x)}[r(y)]\)(Eq.1)。
- 策略梯度 \(\nabla_\theta J(\theta)=\mathbb{E}\big[\sum_{t}\nabla_\theta\log\pi_\theta(y_t\mid y_{<t})\,A_t\big]\)(Eq.2)。三种算法在优势 \(A_t\) 的取法上不同:REINFORCE 取 \(A_t=r(y)\);GRPO/RLOO 用组内归一 \(A_t=\dfrac{r(y)-\mathrm{mean}(r(y_{1:K}))}{\mathrm{std}(r(y_{1:K}))}\)(Eq.3,每 prompt 采 \(K\) 个响应);PPO 用 clipped surrogate(Eq.4)。
- **策略熵定义**(Eq.5,token 级、按响应长归一):\(H(\pi_\theta,D)=-\dfrac{1}{|D|}\sum_{x\in D}\dfrac{1}{|y|}\sum_{t=1}^{|y|}\mathbb{E}_{y_t\sim\pi_\theta}[\log\pi_\theta(y_t\mid y_{<t},x)]\),逐 batch 计算。

### 1. 经验定律（§2.3–2.5）：性能与熵的可预测关系
- **现象(Fig.1/2)**:跨 11 个 RL run,头 200 步就占了约 73% 的熵消耗 + 76% 的性能增益;头 800 步占 >93% 增益 + 94% 熵损失——也就是说,后 2/3 的步只剩边际收益。
- **定律**:\(R=-a\exp(H)+b\)(只用 2 个系数就拟合了 >200 个数据点)。对它微分得 \(\dfrac{dR}{dH}=-a\exp(H)\),所以 \(a\) 就是"把熵换成性能"的转化率,\(-a+b\)(即 \(H=0\) 处)就是熵耗尽时的性能天花板。〔注:原文 §2 takeaway 框内排版作 \(R=-a\exp(H+b)\),但 Fig.1、微分式 \(dR/dH=-a\exp H\) 与各处一致表明真实形式是 \(R=-a\exp(H)+b\);以 Fig.1 为准〕
- **可外推性的两个证据**:
  - ① 用前 36 步(约 15%)拟合就能外推到 200 步,数学 RMSE 0.9%、代码 1.2%,终步 0.5%/1.9%(Fig.5);
  - ② 系数 \(a,b\) 与模型参数量(去掉 embedding)呈 **log-linear**(Fig.7),所以能用小模型外推大模型的终态——即"训个小模型→预测大模型 RL 后的性能、不必真去训大的"。
  - 而且系数与 RL 算法无关(GRPO/RLOO/PRIME/REINFORCE++ 都拟合到同一条曲线,Fig.6),说明 \(a,b\) 是"策略模型 + 数据"的内禀属性。

### 2. 熵动力学理论（§3）：坍缩的根因 = 协方差
> 这一小块四步推导,层层把"熵变"归到"协方差"上:先写熵差(Lemma 1)→ 代入 PG 的 logit 更新(Prop 1)→ 得到 PG/NPG 下的熵变公式(Thm 1/2)。
- **Lemma 1(softmax 策略的熵差,一阶近似)**:对 tabular softmax 策略 \(\pi_\theta(a\mid s)=\dfrac{\exp(z_{s,a})}{\sum_{a'}\exp(z_{s,a'})}\)(Eq.7),相邻两步的熵差为
\(\displaystyle H(\pi^{k+1}_\theta\mid s)-H(\pi^{k}_\theta\mid s)\approx-\,\mathrm{Cov}_{a\sim\pi^k_\theta(\cdot\mid s)}\big(\log\pi^k_\theta(a\mid s),\ z^{k+1}_{s,a}-z^{k}_{s,a}\big).\)
直觉:某动作更新前已是高概率,更新后 logit 又被抬升→就会降熵。
- **Proposition 1(PG 下的 logit 变化)**:vanilla PG 的更新是 \(z^{k+1}_{s,a}-z^{k}_{s,a}=\eta\,\pi_\theta(a\mid s)\,A(s,a)\)——即 logit 变化正比于"概率 × 优势"。
- **Theorem 1(PG 下熵变)**:把 Prop 1 代进 Lemma 1,得 \(H(\pi^{k+1}_\theta\mid s)-H(\pi^{k}_\theta\mid s)\approx-\eta\,\mathrm{Cov}_{a\sim\pi^k_\theta}\big(\log\pi^k_\theta(a\mid s),\ \pi^k_\theta(a\mid s)\,A(s,a)\big)\)。
- **Theorem 2(NPG 下熵变,去掉 \(\pi\) 权重)**:\(\approx-\eta\,\mathrm{Cov}\big(\log\pi^k_\theta(a\mid s),\ A(s,a)\big)\)(KL-Cov 就基于这一式)。
- **结论**:看协方差的符号——\(P(a)\) 与 \(A(a)\) 强正相关→平均降熵;负相关→升熵。换句话说,"高概率 × 高 advantage"的动作会被进一步强化、迅速降熵;只有"稀有 × 高 advantage"才会升熵。
- **实证(§3.3,Fig.8)**:用 bandit 设定(把 prompt 当 state、整条响应当 action),按 Eq.8 算组内协方差、Eq.9 按响应长归一 log-prob。实测结果:\(-d(H)\) 与 \(\mathrm{Cov}(\cdot)\) 两条曲线高度同步,且 \(\mathrm{Cov}\) 全程为正→所以熵持续降;另外难题(低准确率)的协方差更小(说明模型未校准),易题的协方差更大。

### 3. 两个干预（§4.2，Listing 1，只改几行 loss）
> 两个干预共用同一个想法:先算每个 token 的协方差,挑出最"危险"的一小撮,再对这撮 token 改 loss——Clip-Cov 是直接停更它们,KL-Cov 是给它们额外加 KL 惩罚。
- **token 级协方差**(Eq.10,对一 batch \(N\) 个 token 做中心化叉积):
\(\displaystyle \mathrm{Cov}(y_i)=\Big(\log\pi_\theta(y_i)-\tfrac1N\textstyle\sum_j\log\pi_\theta(y_j)\Big)\cdot\Big(A(y_i)-\tfrac1N\textstyle\sum_j A(y_j)\Big).\)
- **Clip-Cov**:先按 Eq.10 算协方差;从落在 \([\omega_{\text{low}},\omega_{\text{high}}]\) 区间的高协方差 token 里**随机**选 \(\lfloor r\cdot N\rfloor\) 个(Eq.11);把这些 token **从策略梯度 detach**(即停止更新):
\(\displaystyle L_{\text{Clip-Cov}}(\theta)=\begin{cases}\mathbb{E}_t\big[\tfrac{\pi_\theta(y_t\mid y_{<t})}{\pi_{\theta_{\text{old}}}(y_t\mid y_{<t})}A_t\big], & t\notin I_{\text{clip}}\\[4pt]0, & t\in I_{\text{clip}}\end{cases}\quad(\text{Eq.12}).\)
- **KL-Cov**:取协方差 **Top-\(k\)** 比例的 token(Eq.13);对它们额外加一个"当前策略 ↔ rollout 策略"的 KL 惩罚:
\(\displaystyle L_{\text{KL-Cov}}(\theta)=\begin{cases}\mathbb{E}_t\big[\tfrac{\pi_\theta}{\pi_{\theta_{\text{old}}}}A_t\big], & t\notin I_{\text{KL}}\\[4pt]\mathbb{E}_t\big[\tfrac{\pi_\theta}{\pi_{\theta_{\text{old}}}}A_t-\beta\,D_{\text{KL}}(\pi_{\theta_{\text{old}}}\Vert\pi_\theta)\big], & t\in I_{\text{KL}}\end{cases}\quad(\text{Eq.14}).\)
- **数据流动**:rollout 采样 → 算 token 级的 \(\log\pi,A\) → 用 Eq.10 算每个 token 的协方差 → 选出 pivotal token 的索引集 → 对这撮 token 改写 loss(detach 或加 KL)→ 反传更新全参。整条管线相对 vanilla GRPO 只多了"算协方差 + 选索引 + 改这撮 token 的 loss"几行(伪代码 Listing 1 就是官方 veRL `core_algos.py` 的实现原型)。
- **关键超参默认值(§4.3)**:
  - Clip-Cov:\(r=2\times10^{-4}\)、\(\omega_{\text{low}}=1,\omega_{\text{high}}=5\)(这两个阈值都 >500× 的平均协方差);
  - KL-Cov:\(k=2\times10^{-3}\)(7B)/\(2\times10^{-4}\)(32B)、\(\beta=1\);
  - 训练设置:DAPO-MATH,batch 256 prompt × 8 响应、温度 1、每 rollout 做 8 次更新、max gen 8192;评测时 AIME/AMC 用温度 0.6,其余贪心。

### 逐组件必要性
- **经验定律 \(R=-a\exp(H)+b\)**：负责回答"why it matters"——证明性能是用熵换来的、上限可预测。跨 Qwen2.5/Mistral/LLaMA/DeepSeek-Math 多族多尺寸(0.5–32B)都拟合良好【§2.4, Fig.3–7】。
- **协方差理论**：负责回答"why 坍缩"。Fig.8 实证 \(\mathrm{Cov}\) 与 \(-d(H)\) 精确同步、全程为正;它是 softmax+PG 的一阶分析【§3.3】。
- **Clip-Cov / KL-Cov**：负责回答"how to fix"。对比 vanilla GRPO(坍缩饱和)、对比 clip-higher(早期抬熵但后期不稳、会回落),这两个干预都更稳且持续提分;其中 KL-Cov 的熵曲线比 Clip-Cov 更稳【§4.3–4.4, Fig.11–12】。
- **反向消融**：去掉协方差选择、退回全局熵/KL 正则 → §4.1 实测失效(这就是本文的"没它会怎样")。

## 靠不靠谱
- 实验与证据：
  - **拟合实验(覆盖广)**:4 族 11 个 base(0.5–32B)、数学+代码、8 个公开 benchmark、4 种 RL 算法;Zero 设置从 base 起 RL,veRL 框架。超参:lr 5e-7(policy)/1e-6(PRIME 的隐式 PRM)、KL 系数默认 0、\(\epsilon=0.2\)、batch 256、micro-batch 128、rollout 512 prompt×8 响应、过滤掉全对/全错的 prompt【原文 §2.2】。
  - **Clip-Cov/KL-Cov 主实验(Table 2)**:在 Qwen2.5-7B/32B 上,测 MATH500/AIME24/AIME25/AMC/OMNI-MATH/OlympiadBench/Minerva(AIME/AMC 为 avg@32)。**关键数字**:较 GRPO 平均提升 **7B +2.0%、32B +6.4%**;32B 上 KL-Cov 在 AIME24 拿 36.8(GRPO 21.8)、AIME25 拿 30.8(GRPO 16.2),即 AIME 上 **+15.0%/+14.6%**;而且 KL-Cov 能维持比 baseline 高 10× 的熵、响应长度稳定增长【原文 §4.3, Table 2, Fig.11】。
  - **baseline 公平吗**:较公平——主 baseline 是 vanilla GRPO 与 GRPO+clip-higher(\(\epsilon\)→0.28),同 DAPO-MATH、同采样预算,而且把 clip-higher 的机理拆开来对比。
  - **"看着强但没回答核心问题"**:维持熵≠一定更好——§4.5 自承"没观察到干预后熵与性能之间的明确关系,是否存在最优熵值仍 open"。也就是说,方法能提分,但"熵该维持到多高"还没有自适应准则。
- 假设与失效边界：
  - 【原文】定律/理论建立在 softmax 策略 + Policy Gradient/Natural PG 上;用可验证任务(数学/代码)来避免 reward hacking(§2.1);Lemma/Prop 都基于 tabular softmax 的一阶近似(证明在 Appendix E)。
  - 【原文】§2.6 自承:换不同策略模型(Luo25 ProRL)、或用 off-policy 数据(Yan25),会观察到不同的熵模式,所以可预测性"非普适"。
  - 【原文】§4.5 明确:最优熵值未知;干预后熵与性能之间没有明确的单调关系。
  - 【推断】\(R=-a\exp(H)+b\) 是拟合关系、不是严格因果,外推到极大算力/更强模型是否仍成立没验证(依据:拟合只做到当前实验规模,§2.4–2.5)。
  - 【推断】协方差理论是局部一阶的 tabular 分析,对加了大量工程 trick(各种 clip、长度惩罚)的实际 RLVR 只是近似(依据:§3 推导基于纯 PG/NPG)。
  - 【推断】干预只是把"调熵正则"换成了"调协方差阈值"(\(r,\omega,k,\beta\)),并没有消除调参负担,而且最优值与任务/尺寸相关(依据:§4.3 里不同模型用了不同的 \(k\))。
- 祛魅总结：
  - 真贡献【推断】:把"熵坍缩"从一个经验观察,升级为"定量定律 + 动力学根因(协方差)+ 按根因设计的最小干预"三件套,三者由"协方差"这一个概念贯穿,因果链完整、且可复现(已合入官方 veRL)。
  - 包装/高估【推断】:"性能上限完全可预测"在已测规模上成立,但若当成普适的 scaling-law 式结论就容易高估它的外推力;另外干预的提分在 7B 上其实不大(+2%),主要亮点在 32B(+6.4%、AIME +15%)。
  - 低估【推断】:"只动 \(2\times10^{-4}\sim2\times10^{-3}\) 的 token 就能改写整条熵曲线"这一"pivotal token"现象,它对理解 LLM RL 的意义可能比提分本身更深远,但论文没深挖这些 token 到底是什么。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=可验证奖励（数学/代码对错）下的 advantage，外加"token 级协方差 \(\mathrm{Cov}(\log\pi,\Delta\text{logit})\)"作为控熵的诊断/干预信号｜**改什么**=策略全参（替换 PPO surrogate loss 的 clip/KL 项）｜**何时改**=训练期在线，每步对当批高协方差 token 施加干预｜**免梯度?**=否（仍是策略梯度 RL；Clip-Cov 对选中 token 局部停梯度=detach，KL-Cov 加 KL 惩罚）｜**记忆-技能生命周期**=无（纯 RL，无记忆/技能库）｜**防遗忘机制**=无显式机制（但"维持探索/抗坍缩"间接缓解过早收敛到单一模式）。
- ⑦ 开源代码+框架/harness：github.com/PRIME-RL/Entropy-Mechanism-of-RL（本地已 clone ~5.9MB，Tier A）；方法**已合入官方 veRL（PR #1830）**，上游 `recipe/entropy/`，可经 `loss_mode=clip_cov`/`kl_cov` 直接用。框架=**veRL**，fork 自 **DAPO recipe**（`recipe/dapo/`），conda env `entropy`（environment.yaml）。代码已审计：`core_algos.py:compute_policy_loss_clip_cov` 在正协方差(lb~ub)token 上随机选 clip_ratio 比例、把校正系数置 0（停更）；`compute_policy_loss_kl_cov` 对协方差 Top-k% token 加 KL 惩罚——与论文 Listing 1/Eq.12/14 一致。运行：`bash recipe/dapo/7b_kl_cov.sh`（单节点）、`recipe/dapo/32b_*.sh`（多节点）。
- 💰 资源/成本与可扩展性：拟合实验跨 0.5–32B、4 RL 算法、8 benchmark，规模大；主干预实验 Qwen2.5-7B/32B、2400+ 步、batch 256、rollout 512×8、max gen 8192。干预本身近乎零额外算力（只改几行 loss、干预 \(\le2\times10^{-3}\) token）。具体 GPU 数/时长原文未说明【原文 §2.2, §4.3；GPU 成本：原文未说明】。
- 🎯 对"探索-巩固"对标：**支撑(机理层)+ 可借组件**。一句判定:本文为"探索"这一支提供了最直接的机理工具——它定量解释了"为什么 on-policy RL 会过早收敛、探索枯竭",并给出了"按协方差精准维持探索"的可移植手段。
  - 可借组件:
    - ① **协方差阈值控熵**——可嵌入 MTP/OPD 的 RL 阶段,防止学生过早锁死到单一路径、保住"自选可走通开头"的多样性;
    - ② "**少数 pivotal token 主宰熵**"——对"在关键步/高熵步接管"(path-recovery 单点接管)是强支撑(与 memory 里 sparse_critical 同构);
    - ③ Theorem 1/2 把"高概率 × 高 advantage→降熵"形式化——可当作"何时该让 teacher 重新打开探索"的判据。
  - 缺口:纯 RL、无 teacher 蒸馏、无 MTP、无巩固/防遗忘机制;它只解决"别太快停止探索",不解决"探索到的怎么固化"。依据:§4.5 协方差只控探索强度,§4.2 干预只动 pivotal token。
- 🔭 开放问题/未来方向：【原文】是否存在平衡探索与稳定的最优熵值仍 open（§4.5）；scaling RL 不止于熵最小化（§6）。【推断】把"高协方差/pivotal token"与"高熵关键步/path-recovery 接管点"对齐，可能统一"控熵"与"在关键步注入 teacher 监督"两条线；将协方差诊断用于 on-policy 蒸馏，判断学生在哪些 token 已过度自信、该由 teacher 重新打开探索（依据：本文 pivotal token 现象 + 本课题 path-recovery 单点接管）。

读到PDF? 是（PyMuPDF 全文 22 页；正文 §1–4 + Lemma1/Thm1-2 + Eq.1–14 + Listing 1 全核，附录证明 E.2–E.4 未逐页核）｜L线 L3（RLVR 训练机理+算法；触及 L6 token 信用）｜对标结论 支撑探索支（控熵机理）+ 可借组件（协方差阈值控熵、pivotal token↔path-recovery 接管点、Thm1/2 降熵判据）；缺巩固/蒸馏/MTP｜残留待核 0

lp_reg | Low-probability Tokens Sustain Exploration in RL with Verifiable Reward (Lp-Reg) | 腾讯 LLM Department(Guanhua Huang/Tingqiang Xu 共一,Bo Zhou 通讯;含清华/北大/港中文实习生) | 2025-10 v1 → 2025-11-07 v2;arXiv 2510.03222 | 主题线 L3(RLVR/GRPO 熵崩溃·探索保持)·相关性 中

**原始论文**:https://arxiv.org/abs/2510.03222

## 一眼看懂
- 🟦 TL;DR:RLVR 训久了会"熵崩溃"——策略熵快速衰减、探索消失、性能进平台期。本文指出根因不是"整体熵不够",而是**有价值的低概率探索 token 被 GRPO 系统性惩罚消除**。这些 token 叫 **Reasoning Sparks**(如 "wait"/"however"/"perhaps",开启新推理路径)。做法:构造一个**去噪代理分布 \(\pi_{\text{proxy}}\)**(把概率低于阈值 \(\tau\) 的 token 当噪声丢掉、把质量重归一化到剩下的 token 上,从而放大 spark 的相对概率),再在 GRPO 上加一个**前向 KL 正则 \(D_{\mathrm{KL}}(\pi_{\text{proxy}}\,\|\,\pi_\theta)\)**,但只对"低概率 ∩ 非噪声 ∩ 负优势"**三条件同时满足**的 token 触发,定向阻止 spark 被消除(§4)。Qwen3-14B-Base 五数学集均值 **60.17%**、较 80/20 高熵法 **+2.66%**,且能在基线崩溃的长训练区间持续 scaling 不崩。
- 最巧的一步:**"先用概率阈值滤掉噪声、再保护剩余低概率 token"这一区分 spark/noise 的两步**(而非无差别提熵)。抽掉噪声过滤(对所有低概率 token 一视同仁保护),就退化成"无差别最大化随机性"——会放大噪声、**加速崩溃**(论文实测 GRPO+Entropy Loss 比 baseline 崩得更快,§Abstract/§5.3 移除过滤的消融)。正是"spark 平均概率一贯高于 noise"这一可分性(§Fig.1a/§6.1 统计观察 + Nguyen 2025/Xu 2025"高相对概率 token 更契合上下文"的依据),让定向保护成为可能。

## 为什么做
- 研究背景:RLVR(可验证奖励 RL,数学答案对/代码过测才给奖励)推动复杂推理(OpenAI o1/DeepSeek-R1),但训练常在平台期崩溃,伴随策略熵快速衰减、探索丧失。形式上 RLVR 目标 \(J_{\mathrm{RL}}(\theta)=\mathbb{E}_{(q,a)\sim D,o\sim\pi_\theta(\cdot|q)}[r(o,a)]\)(§3.1 Eq.1),实践用 GRPO 优化。
- 解决的具体痛点:依赖"整体熵"是**间接(correlational)而非因果的**工具——无差别最大化随机性会放大噪声、加速崩溃。真正问题在于**有价值的低概率探索 token 被系统性消除**,这一根因没被现有熵方法精准命中。
- 相关工作 & 各自具体短板(§2 三类):
  - **RL for LLMs / RLVR**:GRPO(Shao 2024)、DAPO(Yu 2025)、VAPO(Yue 2025)等改 clip/归一提稳定性与可扩展性——但都在"分数/稳定性"层面,未触及"低概率 spark 被消除"这一机制根因。
  - **熵崩溃缓解**:高熵"forking token"选择性正则(Wang 2025)、探索位优势放大(Cheng 2025)、改裁剪(Yu/Zhao/Cui/Zheng 2025)、权重裁剪(MiniMax 2025/Su 2025)。**共同短板**:它们"primarily operate by monitoring policy entropy, which is correlational rather than causal to exploration"——监控整体熵而非直接看 NTP 分布,无法区分"有用的低概率 spark"与"无关的低概率 noise",故要么不够、要么放大噪声。
  - **LLM intrinsic confidence**:研究表明 NTP 分布里的相对概率反映模型内在置信,且高相对概率 token 通常更契合上下文(Nguyen 2025 min-p、Xu 2025、Fu 2025);熵最小化(Gao/Agarwal 2025)反向锐化置信以提推理。**本文与之共用洞见**——用模型内在置信在低概率区间**区分 spark vs noise**(而非像熵最小化那样一味锐化或像熵正则那样一味提熵)。
- 动机链:崩溃伴随熵衰减 → 但提整体熵会放大噪声 → 观察到 spark 与 noise 在低概率区间**可分**(spark 平均概率更高)→ 故先滤噪声再定向保护剩余低概率 token。
- 与最近邻工作的精确Δ:相对熵正则/forking-token 选择性更新(Wang 2025),差在**从"整体熵"或"高熵分叉点"下沉到"低概率 token 的语义筛选(spark vs noise)"**,用**前向 KL + 三重门控**做**定向**保护而非无差别提熵——更有针对性、不放大噪声;且明确论证"高熵是探索的劣质代理"。

## 怎么做 + 靠不靠谱
- 基线 GRPO 回顾(§3.2,作为修改起点):优势 \(A_{i,t}=\dfrac{R(o_i)-\mathrm{mean}(G)}{\mathrm{std}(G)}\)(Eq.2,组内 \(G\) 个采样的 reward 归一);概率比 \(r_{i,t}=\dfrac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})}\)(Eq.4);标准 PPO 代理目标带对称裁剪 \([1-\epsilon,1+\epsilon]\) 与 KL 项(Eq.3)。
- 方法流水线(Lp-Reg,集成进 GRPO):
  - ① **构造 \(\pi_{\text{proxy}}\)(两步)**(§4.1):
    - (a) **过滤噪声**——定义"噪声 token"为概率 \(\pi_\theta(o|\cdot)\le\tau\) 者并丢弃。阈值 \(\tau\) 两种:**固定**(常数,如 \(\tau=0.02\));**min-p**(Nguyen 2025,随分布锐度自适应)\(\tau=\kappa\cdot\max_{o'}\pi_\theta(o'|\cdot)\),主实验 \(\kappa=0.02\)。
    - (b) **重归一化**——把丢弃 token 的质量重分到剩余 token(Eq.5):
    \(\displaystyle \pi_{\text{proxy}}(o|\cdot)= \begin{cases} \dfrac{\pi_\theta(o|\cdot)}{\sum_{o'\,:\,\pi_\theta(o'|\cdot)>\tau}\pi_\theta(o'|\cdot)} & \text{if }\pi_\theta(o|\cdot)>\tau\\ 0 & \text{otherwise.} \end{cases}\)
    直觉:把低相对概率 token 当噪声清零,剩下的"高置信参考"被整体放大——其中真正的 spark(平均概率高于 noise)被保留并相对加权(§Fig.2)。注:proxy 由数据生成策略 \(\pi_{\theta_{\text{old}}}\) 构造(§5.1)。
  - ② **目标函数**(§4.2,Eq.6):第一项 GRPO 策略梯度,但**去掉裁剪下界**(改为 \(\mathrm{clip}(r_{i,t},0,U)\),避免裁掉低概率探索动作)、加大上界 \(U=10\)(数值稳定);第二项 Lp-Reg 惩罚——**仅对三条件同时满足的 token 触发**前向 KL:
  \(\displaystyle J_{\text{Lp-Reg}}(\theta)=\mathbb{E}\!\left[\frac{1}{\sum_i|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\Big(\mathrm{clip}(r_{i,t},0,U)\,A_{i,t}-\beta\cdot\mathbb{I}[\,\cdot\,]\cdot D_{\mathrm{KL}}\big(\pi_{\text{proxy}}(\cdot|q,o_{i,<t})\,\|\,\pi_\theta(\cdot|q,o_{i,<t})\big)\Big)\right],\)
  其中指示器 \(\mathbb{I}[\,\cdot\,]\) 同时要求:**低概率** \(\pi_\theta(o_{i,t}|\cdot)<\delta_B^\rho\)(\(\delta_B^\rho\)=当前批 \(B\) 内所有 token 采样概率的最低 \(\rho\) 分位)∧ **非噪声** \(\pi_{\text{proxy}}(o_{i,t}|\cdot)>0\) ∧ **负优势** \(A_{i,t}<0\)。第三条件确保只在"负学习信号"上施正则(防止把正确探索过度惩罚,同时不动正样本更新)。
- 逐组件必要性(消融 §5.3):
  - **噪声过滤阈值 \(\tau\)(min-p)**:关键。去掉过滤(对所有低于 \(\tau\) 的 token 也"fork"进来保护)→ 放大噪声、性能不稳(§5.3 "Importance of Noise Filtering");min-p 动态阈值优于固定阈值(§5.3 + Fig.8,dynamic \(\tau\) > fixed \(\tau\),因 min-p 跨题对模型置信的估计更稳)。
  - **三条件门控**:论文专门逐条测(§4.2 末 + §5.3);缺任一条件保护就失准(对正样本/噪声/高概率 token 误触发)。
  - **前向 KL(方向选择)**:§5.3 + Fig.10 证明 forward KL \(D_{\mathrm{KL}}(\pi_{\text{proxy}}\|\pi_\theta)\) 优于 reverse KL \(D_{\mathrm{KL}}(\pi_\theta\|\pi_{\text{proxy}})\)——因 proxy 只是**启发式参考而非理想目标**:reverse KL 会强迫策略严格模仿 proxy(把它当真理),forward KL 只在"策略快把 proxy 里仍有概率的 token 压到 0"时惩罚,提供**软**正则、保留越过 proxy 探索的自由。
  - **去裁剪下界 + 大上界 \(U\)**:配合放行低概率动作梯度(对称下界会裁掉它们),属机制必要部分。
- 关键机制/公式(直觉):前向 KL \(D_{\mathrm{KL}}(\pi_{\text{proxy}}\|\pi_\theta)=\sum_o \pi_{\text{proxy}}(o)\log\dfrac{\pi_{\text{proxy}}(o)}{\pi_\theta(o)}\),当某 spark 在 proxy 里有概率(\(\pi_{\text{proxy}}>0\))但策略快把它压到 0(\(\pi_\theta\to0\))时,\(\log(\pi_{\text{proxy}}/\pi_\theta)\to\infty\) 给巨大惩罚,把它"拉回来";反之若 proxy 已置零(认定为噪声),则该 token 在 KL 里权重为 0、不被保护。这正是"定向防止 spark 被消除,又不强制策略完全匹配 proxy"(§4.2)。
- 实验与证据:RL 训练 Dapo-Math-17K,max 8192,global batch 256;\(U=10\)、min-p \(\kappa=0.02\)、percentile \(\rho\)(14B \(=1\%\)、32B \(=0.5\%\))。评测 5 数学集(AIME24/25、MATH-500、OlympiadBench、Minerva;AIME24/25 采 16 次 temp 0.6、其余 greedy)。骨干 Qwen3-14B-Base(主)、Qwen2.5-32B-Base。关键数字:Qwen3-14B-Base 五基准均值 **60.17%**,超次优(80/20)**+2.66%**;**稳定性测试 Qwen2.5-32B 训 3000 步、81,204 GPU·h 不崩**。基线 GRPO/GRPO+Entropy Loss/Clip-Higher/80-20/KL-Cov/GSPO,较全面公平。
- 假设与失效边界:
  - 【原文】引入多阈值超参(\(\tau/\kappa\)、\(\rho\) 分位、\(\beta\)、\(U\)),14B/32B 取值不同(\(\rho=1\%/0.5\%\));仅在**数学域**验证。
  - 【推断】核心增益 **+2.66% 属中等增量**,且**大量算力(8 万 GPU·h)主要用于证明"长训练不崩"而非绝对分数跃升**——卖点是"稳定 scaling"而非"刷高分"。"spark vs noise"以**平均概率**统计区分,个例上二者可能重叠,**门控误判的代价未量化**;阈值多→调参负担大、跨骨干泛化稳健性需更多验证。当 spark/noise 概率分布重叠严重(如某些非数学域)时应失效。
  - 【推断】工程不便:主仓为占位 README,实际实现需切到 dev 仓。
- 祛魅总结【推断】:真贡献是**把"防熵崩溃"从"提整体熵"这个钝器,精化成"按语义保护低概率 spark"的精准手术**,机制叙事(观察可分性→去噪 proxy→前向 KL 定向保护)环环相扣、方向选择(forward vs reverse KL)有"proxy 只是启发式参考"的清晰论证,这是相对熵正则的实质进步。需要祛魅的是把它当"性能跃升方法"——它的真价值在**长训练稳定性/持续探索**,绝对增益温和;且阈值多、仅数学域,普适性待证。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=token 级(GRPO 策略梯度 + 对低概率 spark 的前向 KL 保护信号);**改什么**=策略参数(actor);**何时改**=on-policy RL 每步,但保护项**仅在"低概率∩非噪声∩负优势"token 上触发**(选择性);**免梯度?**=否(梯度,且专门"去裁剪下界"以放行低概率动作梯度);**记忆-技能生命周期**=无;**防遗忘机制**=核心就是一种"防遗忘"——**定向防止已学会的探索行为(spark)在训练中被惩罚消除**,可视为"保护探索能力不被巩固过程抹除"的机制(虽非针对参数级灾难遗忘)。
- ⑦ 开源代码+框架/harness:主仓 https://github.com/CarlanLark/Lp-Reg(仅 README 占位,称整合进最新 veRL);**实现在 dev 仓** https://github.com/CarlanLark/Lp-Reg-dev(已克隆 ~4.6MB,**verl** 底座,含 `recipe/lp_reg` 与 `recipe/dapo`)。框架 **verl**(GRPO 基础)。可得性:dev 仓真实可复现;主仓占位需自行切 dev 仓。〔实现以 Lp-Reg-dev 为准〕
- 💰 资源/成本与可扩展性:verl 上 GRPO,lr=1e-6 常数无 warmup,group=8,IS 比率上界 \(U=10\),off-policy mini-batch 32(每 rollout 8 次梯度更新)。算力:14B ~1000 步(8000 GPU·h / 32×H20),32B ~800 步(16000 GPU·h / 64×H20);稳定性测试 32B 训 3000 步、81,204 GPU·h。**算力开销大,主要花在证明长训练不崩**。
- 🎯 对"探索-巩固"对标:**中等支撑(直接服务"探索"侧的保护)**——Lp-Reg 精准对应 idea 中"巩固时别把探索能力练没了":它在 token 级**保护开启新推理路径的关键 spark**,正是"探索=发现有效路径"在 RL 训练中不被消除的机制保障。可借组件:(a) **spark vs noise 的低概率可分性 + min-p 去噪 proxy**——可用于在 OPD/MTP 训练中识别并保护"student 自己摸索出的关键探索 token";(b) **前向 KL 在 \(\pi_\theta\to0\) 时大惩罚的"定向防消除"**思想,可迁移到"防止 student 在巩固 teacher 信号时丢掉自己走得通的探索分支"(直接呼应 idea 的"偏向自己能走通的开头");(c) **三条件门控**(低概率∩非噪声∩负优势)是一个可复用的"何时才介入保护"判据。缺口:无 teacher/无蒸馏/无 MTP 前瞻/无记忆库,纯 RL 内部的探索保持;"巩固/回轨"侧无对应机制。判定依据:§4 三条件门控 + §Abstract spark 定义。
- 🔭 开放问题/未来方向:【原文】更多骨干验证泛化;阈值自适应。【推断】把"spark 保护"从纯 RL 扩到 **OPD/mixed-policy**(保护 student 在受 teacher 监督时仍保留自身关键探索 token,正是 LUFFY policy-shaping 的互补);用 MTP 前瞻判定"哪个低概率 token 真是有效 spark";门控误判代价的量化与跨域(非数学)有效性验证。

— 残留待核:0(Eq.1–6 与 min-p/三条件/forward-vs-reverse KL 论证均据 PDF §3–§5 抄准。)

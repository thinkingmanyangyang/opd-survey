revisit_entropy | Revisiting Entropy in Reinforcement Learning for Large Reasoning Models | 天津大学 TJUNLP Lab + 天津师范大学 + 独立研究者（Renren Jin 一作；Deyi Xiong 熊德意 通讯） | 2026-04-19 v3（v1 2025-11-08）· arXiv preprint cs.CL | 主题线 L3(RLVR/GRPO)+L6(Token信用)·相关性 中

**原始论文**:https://arxiv.org/abs/2511.05993

## 一眼看懂
> 一句话导读:RLVR 训练里"策略熵坍缩"(模型越来越确定、不再探索)是个老大难;本文做了一次系统体检,发现真正的元凶是"正优势 token",并据此给出一个调节这类 token 损失权重的小旋钮来稳住熵。

- 🟦 TL;DR:用 RLVR(可验证奖励 RL,即用"答案对不对"这种可机械判定的信号做 RL)训练 LLM 时,策略熵常会"坍缩"——概率质量集中到少数 token、模型丧失探索,从而过早收敛。本文做了一次系统性的熵体检,四件事:
  1. 实证熵与"回答多样性 / 校准 / 性能"的关系:发现熵**不是**性能的可靠代理,二者相关性高度依赖任务;熵坍缩越严重,模型越过自信、校准越差;而且**性能可以在不牺牲熵的前提下持续提升**;
  2. 找出影响熵动态的三个因素:clip 阈值、off-policy 更新次数、训练数据多样性;
  3. 用理论 + 实证证明**正优势(positive-advantage)token 是熵坍缩的主因**;
  4. 据此提出 Positive-Advantage Reweighting:只是动态调节"正优势 token"的损失权重来调熵,同时保住性能。【Abstract,§1】
- 最巧的一步:**§7.1 的梯度符号分析(式3/4)+ §7.2 的"只训 Adv≥0 vs 只训 Adv≤0"对照**。抽掉它,本文就只剩一堆"熵现象观察 + 调参经验"的集合。正是这套机制论证——"正优势 token 会推高已采样的高概率 token、同时压低未采样的低概率 token,导致质量集中、进而熵坍缩"——把前面分散的现象统一了起来(clip 的三个因素其实都是通过"改变正/负优势 token 的相对梯度贡献"来影响熵的),并直接导出了干预手段:给正优势 token 的 loss 乘一个可调权重 \(\lambda\)。【§7.1-7.2,图5】

## 为什么做
> 一句话导读:大家都知道 RLVR 会熵坍缩、也有一堆零散的缓解招,但缺一个"熵-性能-校准-多样性"的全景体检 + 把矛头明确指向某类 token 的因果归因;本文就来补这个系统性空白。

- 研究背景:RLVR(由 o1/DeepSeek-R1/Kimi k1.5 开创)已成为提升 LLM 推理的主流路线;但越来越多工作发现 RLVR 会驱使熵坍缩——概率质量集中到少数 token,模型越来越偏向 exploitation(利用)、不再探索新推理路径,过早陷入局部最优。【§1,L42-70】
- 解决的具体痛点:已有的缓解法很多,但**对 RLVR 中的熵缺乏系统研究**,三个核心问题没人答清楚:(1) 熵与性能如何相关?(§5)(2) 哪些因素(理论 + 实证)在支配熵动态?(§6)(3) 如何有效调熵以提升性能?(§7)【§1,L83-90】
- 相关工作 & 各自不足(来龙去脉,据 §2):
  - **用熵作 RL 的探索正则**历史悠久(Ziebart 的 max-ent RL)。
  - **LLM 时代用熵当信号**:Dong/Zhang 等用熵来决定 agent 何时去求经验指导、何时补采样;还有把**熵最大化目标**并入 RLVR 的(Shen 2025;He 2025),以鼓励探索、防过早收敛。**短板**=正则系数难调:系数大则熵暴涨,小则没效果。
  - **clip 类**:DAPO Clip-Higher(Yu 2025)抬高重要性比率的上界,防止低概率的正优势 token 被 clip 掉。**短板**=只是缓解、不显式控熵,熵仍会无控漂移。
  - **协方差类理论切片**:Liu 2025 与 Cui 2025(Clip-Cov、KL-Cov)从理论上指出"log-prob 与 advantage 强正协方差的 token"主导坍缩,据此去限制这类 token 的更新。**短板**=干预手段是"限制更新"(硬截断),不是连续可调的,而且只切了单一理论切面。
  - **CE-GPPO**(Su 2025):用 stop-gradient **保留被 clip 掉的 token 的梯度**来缓解坍缩(与项目 memory 记录的 forward-hard/backward-soft 解耦同构)。
  - 此外还有:**只用高熵 token 训**(Wang 2025b);**把熵并入 advantage**(Cheng 2025,即 Entropy-Adv)。
  - **共同空白**:这些大多是点状缓解或单一理论切片,缺三样东西——"熵-性能-校准-多样性"的全景、跨三因素(clip/off-policy/数据)的统一实证、以及 token 级的因果归因。【§2,L145-180】
- 动机链:RLVR 虽强但会熵坍缩 → 缺探索、过早收敛 → 那就先搞清熵与性能到底什么关系(是不是越高越好) → 再找熵被什么支配 → 再定位到底是哪类 token 在推坍缩 → 最后针对该类 token 设计最小干预。
- 与最近邻工作的精确Δ:
  - vs **Cui(Clip-Cov/KL-Cov)**:双方都把矛头指向"高协方差/正优势 token",但 Cui 的干预是"限制这类 token 的更新",本文 Pos-Adv-Reweight 则是"给正优势 token 的 loss 乘一个**可调连续权重 \(\lambda\)**"——是更细的连续调控,还配了 stage/epoch/entropy-guided 三种调度。
  - vs **DAPO Clip-Higher**:本文把 Clip-Higher/Lower/Tighter/Free 放进统一的 clip 阈值消融,用梯度公式解释了为何 Clip-Higher 能抗坍缩;并指出 Clip-Higher 不显式控熵、熵会无控漂移,而 Pos-Adv-Reweight 能把熵稳到目标 \(\delta\) 附近。
  - 关键 Δ:**把"熵 = 性能代理"这一常见假设证伪**(Table1 的 Spearman 相关高度依赖任务:AIME24 Avg@64 为 0.417、LiveCodeBench 为 −0.894、IF-Eval 为 0.627),并给出"性能可不牺牲熵而升"的反例(§5.1)。【§5,Table1】

## 怎么做(到"读完能复现"的粒度)
> 一句话导读:全文主体 = 一套诊断扫描(沿 clip/off-policy/数据三轴看熵怎么变)+ 一个 token 级因果归因(锁定正优势 token)+ 一个轻量方法 Pos-Adv-Reweight(给正优势 token 的损失加可调权重)。下面把基底目标、三轴扫描、梯度归因、方法三变体逐一写到可复现。

### A. 基底:token-level、去 KL 的 GRPO(全文训练用)
做法:对每个 prompt \(x\) 采 \(G\) 个响应 \(\{y_i\}\),用组内 reward 的均值/标准差归一化得到组相对 advantage \(\hat A_{i,t}\)(沿用 DAPO 的两点设定:token-level loss + 去掉 KL 惩罚)。目标函数:
\(\displaystyle J(\theta)=\mathbb E_{x\sim\mathcal D,\{y_i\}\sim\pi_{\theta_{\text{old}}}}\!\Big[\tfrac{1}{\sum_i|y_i|}\sum_{i=1}^{G}\sum_{t=1}^{|y_i|}\min\!\big(r_{i,t}(\theta)\hat A_{i,t},\ \mathrm{clip}(r_{i,t}(\theta),1-\varepsilon_{\text{low}},1+\varepsilon_{\text{high}})\hat A_{i,t}\big)\Big],\)
其中 \(r_{i,t}(\theta)\) 是重要性比率(新旧策略对该 token 概率之比)。熵的定义 \(H(\pi_\theta)\) 取 \(\pi_\theta\) 的 token 级平均熵;若加熵正则,就是在目标上加一项 \(\alpha H(\pi_\theta)\)。【附录A.1】

### B. 诊断扫描(同模型/同数据,沿三轴)
平台:**Qwen2.5-Math-7B + GRPO + veRL + DAPO-Math-17K**,全程监测熵、reward、ID/OOD 性能、校准。
- **§5 熵 vs 性能/多样性/校准(证伪"熵 = 性能")**,五个发现:
  1. 熵与 N-gram 多样性强正相关(4 个设置平均 Spearman 0.8795)——熵低就意味着输出更不多样;
  2. 训练中 response 熵和 prompt 熵都在降,且 ID(分布内)prompt 的熵降得更快;
  3. **熵不是可靠的性能代理**:Table1 里熵-性能的 Spearman 高度依赖任务(AIME24 Avg@64 为 0.417、LiveCodeBench 为 −0.894 强负相关、IF-Eval 为 0.627 正相关)——取决于具体任务与评测指标;
  4. 熵坍缩越重,模型校准越差(越过自信);
  5. **性能可以不牺牲熵而提升**(§5.1,用 Ada-Ent-Reg 把熵稳在训练前的水平)。【§5,Table1】
- **§6.1 clip 阈值**:统一对比四种——Clip-Higher / Clip-Lower(\(\varepsilon_{\text{low}}=0.28\))/ Clip-Tighter(\(\varepsilon_{\text{low}}=0.12\))/ Clip-Free(干脆去掉 clip)。结果:Clip-Higher 抗坍缩、甚至会升熵;Clip-Lower 坍缩最重;Clip-Tighter 有缓解;**Clip-Free 的熵排第二高、ID/OOD 性能也排第二高**——说明 off-policy 更新少的时候,去掉 clip 并不损稳定。【§6.1】
- **§6.2 off-policy 更新数 \(N_{\text{update}}\in\{1,2,4\}\)**:更多更新会放大熵的变化、提高训练 reward,但 ID Avg@64 的增益不足 1%、Pass@64 反而下降(过拟合)。表内数据:\(N=1\) 熵 0.118、\(N=2\) 熵 0.0229、DAPO(\(N=4\))熵 0.479。注意此处不单调——\(N=2\) 反而坍缩最重,而 \(N=4\) 因为带了 Clip-Higher 所以熵反高。【§6.2,Table2】
- **§6.3 数据多样性**:训练数据多样性越低,熵坍缩越重(用 K-means 选出的子集就比随机子集熵更低)。一个降本启示:**约 616 样本 ≈ 17k 样本的性能**,说明数据规模并非唯一决定因素。【§6.3】

### C. token 级因果归因:正优势 token 是主因(§7,全文支柱)
- **梯度符号分析(§7.1,式3/4)**:对某个 token \(v\) 的 logit \(z_v\) 求梯度,其符号由三件事共同决定——该步是否采样到 \(v\)、advantage 的正负、以及是否被 clip。当 \(v\) **未被采样**时(式3):
\(\displaystyle \frac{\partial J(\theta)}{\partial z_v}=\begin{cases}-\,r_t(\theta)\,\pi_\theta(v\mid x,y_{<t})\,\hat A_t,&\hat A_t>0\ \text{且}\ r_t(\theta)<1+\varepsilon_{\text{high}}\\[2pt]0,&\hat A_t>0\ \text{且}\ r_t(\theta)>1+\varepsilon_{\text{high}}\\[2pt]-\,r_t(\theta)\,\pi_\theta(v\mid x,y_{<t})\,\hat A_t,&\hat A_t<0\ \text{且}\ r_t(\theta)>1-\varepsilon_{\text{low}}\\[2pt]0,&\hat A_t<0\ \text{且}\ r_t(\theta)<1-\varepsilon_{\text{low}}\end{cases}\)
当 \(v\) **被采样**时(式4),把 \(\pi_\theta(v)\) 换成 \((1-\pi_\theta(v))\) 并去掉负号:\(\frac{\partial J}{\partial z_v}=r_t(\theta)\,(1-\pi_\theta(v\mid x,y_{<t}))\,\hat A_t\)(clip 边界处同样归零)。读这个公式的直觉:**因为 RLVR 做的是梯度上升**——
  - 正优势会**抬高已采样 token 的概率、压低未采样 token**;而高概率 token 又更容易被采样,于是质量越堆越集中 → **熵坍缩**;
  - 负优势则相反:**压低已采样的高概率 token、抬高未采样的** → 抗坍缩;
  - clip 把比率截断时梯度直接归零,所以 **clip 阈值的本质就是在调节正/负优势 token 的相对梯度贡献** → 这与 §6.1 的实证一致。【§7.1,推导见附录E.1】
- **直接实证(§7.2,关键)**:做一个对照——只训 Adv≥0 的 token vs 只训 Adv≤0 的 token。结果**只训 Adv≥0 → 熵坍缩最重(熵 ≈ 0.0146)**;**只训 Adv≤0 → 熵很高(≈ 0.884)**(图5)。这一对照与 Ada-Ent-Reg / Clip-Cov / KL-Cov / Entropy-Adv / Rand-Pos-Clip 同台比较。它是"正优势 token 主导坍缩"这一结论的因果支柱。【§7.2,图5】

### D. 方法:Positive-Advantage Reweighting(§7.3)
核心做法:对 advantage \(\hat A>0\) 的 token,给它的损失乘一个可调权重 \(\lambda\)(取 \(\lambda=0\) 就等于完全不训正优势 token、只用负优势 token)。\(\lambda\) 有三种调度:
1. **Stage-based**:把训练分成两个等长阶段;前半段 \(\lambda=0\)(只用非正优势 token),后半段 \(\lambda\) 从 0 线性升到 1。
2. **Epoch-wise**:\(\lambda=(e-1)/(E-1)\)(\(E\) 是总 epoch 数、\(e\) 是当前 epoch),逐 epoch 线性上升。
3. **Entropy-guided(主推)**:按当前熵自适应调整——熵超过阈值 \(\delta\) 就 \(\lambda+\Delta\) 来压熵,否则 \(\lambda-\Delta\) 来升熵,把熵稳定在 \(\delta\) 附近:
\(\displaystyle \lambda_{k+1}=\begin{cases}\mathrm{clip}(\lambda_k-\Delta,\,0,\,1),&H_k(\pi_\theta)<\delta\\[2pt]\mathrm{clip}(\lambda_k+\Delta,\,0,\,1),&\text{otherwise}\end{cases}\)
默认 \(\delta=0.2\)(与 Ada-Ent-Reg 对齐)、\(\Delta=0.05\)、\(\lambda_0=0\)。结果:Stage/Epoch 两版的熵都是先升后降;Entropy-guided 把熵稳在 0.2 附近,拿到最高的 ID Avg@64(45.66),并在 7 个 benchmark 中有 6 个的 Avg@64 超过 Clip-Higher。**消融的额外发现**:连 Rand-Pos-Clip(随机把一小撮正优势 token 的梯度置零)都已能缓解坍缩、且 Avg@64 ≈ Clip-Cov——反过来印证了"调正优势 token 的损失权重确实是有效的控熵手段"。【§7.3,Table2,图5】

### 数据/模型/评测(复现锚点)
- 模型:Qwen2.5-Math-7B(主)+ Llama-3.1-8B-Instruct(泛化验证,§7.3 末确认结论可迁移);训练 GRPO@veRL@DAPO-Math-17K(token-level loss,去 KL)。
- ID 基准:AIME24/25、MATH500、AMC23、Minerva;OOD 基准:LiveCodeBench(代码)、IF-Eval(指令遵循);指标 Avg@64 / Pass@64。
- 关键数字:Table2 各变体 ID Avg@64 多在 43-45.7(差距小),熵跨度大(0.015~0.88);GRPO(\(N=1\)) ID Avg@64 44.12;Pos-Adv-Reweight(Entropy-guided) 45.66。【Table1/2,§5-7】

## 靠不靠谱
> 一句话导读:对照扫得很全、很公平,但主实验只在一个数学 7B 模型上做;而且要看清——本文真正值钱的是诊断结论(熵≠性能、600≈17k 样本),方法本身涨点很微弱,更像一个"稳熵旋钮"。

- baseline 公平吗:同模型/同数据/同框架横扫十余个变体(clip×4、\(N_{\text{update}}\)×3、数据多档、方法×多),且 Entropy-guided 特意与 Ada-Ent-Reg 对齐 \(\delta=0.2\) 以便可比,公平性强。但**主实验仅用 Qwen2.5-Math-7B + 数学训练数据**(Llama-3.1-8B 只是泛化点验,OOD 只是用来评测、不是训练域的多样化)。
- 看着强但没回答核心问题:Pos-Adv-Reweight 的性能增益其实很小(ID Avg@64 45.66 vs 默认 GRPO 44.12,约 +1.5 点),主卖点更像"能稳住熵且不掉点"而非"显著涨点"——价值偏向"可控性/诊断"而非"性能突破"。另外"熵不是可靠性能代理"这个结论很有用,但它也反过来意味着:调熵本身并不直接等于提性能。
- 假设与失效边界:
  - 【原文】① 主体基于 GRPO + Qwen2.5-Math-7B + 数学数据;② 梯度分析(式3/4)是在"token 未/已采样 + clip 边界"假设下的近似(推导在附录E.1);③ "性能可不牺牲熵而升"是建立在把熵稳在训练前水平之上的(\(\delta\) 由 1000 prompt 实测得到)。
  - 【推断】结论强绑定 GRPO 族的 "clip + advantage" 结构。换一种 reward 形态(比如 OPD 用的是 teacher 的 token reward、而非二元 advantage)时,"正优势 token 主导坍缩"这套机制不一定能平移——OPD 没有显式的 advantage 正负之分,其熵动态由师生分布的对齐主导(见 rethink_opd),二者机制不同。
- 祛魅总结:真贡献=把 RLVR 的熵从"经验调参"提升为一个"现象全景(熵 vs 多样性/校准/性能)+ 三因素 + 正优势 token 是主因"的理论-实证闭环,并给出一个可解释的最小干预 Pos-Adv-Reweight。两个最有用的论断:**"熵不是性能的可靠代理(且高度任务相关)"**,以及 **"正优势 token 是熵坍缩的主因"**。
  - 【推断】被高估的是方法本身(增益微弱,更像一个"可控旋钮");被低估、真正有价值的是那些诊断结论(尤其 §5.2 证伪"熵=性能"、§6.3 的"600≈17k 样本")。整体是 RLVR 熵这条线上扎实的"综述+体检"型工作,不是范式突破。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=verifier 二元奖励→GRPO 组相对 advantage(本文重点是 advantage 的正/负如何经梯度影响熵) | **改什么**=策略参数 θ;干预手段=正优势 token 的损失权重 λ | **何时改**=在线 RLVR;λ 按 stage/epoch/entropy 三种调度动态调整 | **免梯度?**=否(GRPO policy gradient) | **记忆-技能生命周期**=无显式记忆/技能库 | **防遗忘机制**=无;但"防熵坍缩=保探索"间接防"多样性遗忘"(熵坍缩↔过自信/校准变差是本文核心关联)
- ⑦ 开源代码+框架/harness:https://github.com/cordercorder/EntropyRL(v1 元信息已核:约 7.8MB,以 veRL 为底,train_scripts/ 提供 ada_ent_reg.sh、clip_cov.sh、kl_cov.sh、entropy_adv.sh、pos_adv_reweight.sh、rand_pos_clip.sh 等;Clip-Higher/Adv≤0/≥0 等只需对 veRL 小改、未单独给脚本);框架 **veRL(GRPO)**。【§1 脚注,§4】
- 💰 资源/成本与可扩展性:原文未给 GPU 时/卡数。可推断:单模型 7B + GRPO,扫了十余配置(clip×4、Nupdate×3、数据子集多档、方法×多)——总算力不小但单 run 是常规 7B RLVR 规模;§6.3 "616 样本≈17k 样本"是个降本启示。【§6】
- 🎯 对"探索-巩固"对标:**间接支撑(探索侧)+ 可借诊断,非直接竞品**。
  - 映射:本文的"探索 = 高熵 = 保留多样推理路径"对应 TSRD 的"探索/选路"。其核心警示——"正优势 token 推动熵坍缩、导致过度 exploitation"——提醒 TSRD:在"巩固/固化有效路径"的同时,要防止把探索能力也一并固化掉(过早熵坍缩 = 过早巩固)。
  - 可借组件(三件):① **token 级正/负优势的梯度符号分析(式3/4)**,可迁移来分析 TSRD 中"哪些 token 的更新在收缩 vs 扩张策略支撑",尤其判断 path-recovery 监督会不会反而引发熵坍缩;② **"熵≠性能、要按任务看相关性"的方法论**(评估 TSRD 时别把熵当目标);③ Entropy-guided 的 λ 调度,可作"巩固强度"的自适应旋钮思路(熵高时多固化、熵低时松手探索),与"探索-巩固"的动态平衡天然同构。
  - 缺口:本文是纯 RLVR(无 teacher、无蒸馏、无记忆),与 TSRD 的"teacher 脚手架 + MTP 前瞻 + 记忆固化"三要素都不沾;且它的机制(靠 advantage 正负)在 OPD 设定下并不直接适用。
  - 判定:**探索侧的机制参考与诊断工具来源,不是方法竞品**。
- 🔭 开放问题/未来方向:【原文】未设独立 Future Work 章;隐含方向=把熵调控与性能解耦后如何"既保探索又稳定提升"、Pos-Adv-Reweight 在更多模型/域的验证。【推断】把"正优势 token 主导坍缩"的分析推广到 OPD/自蒸馏的 token reward 设定(检验是否同样有"某类 token 主导坍缩");把 Entropy-guided λ 与 OPD 的 overlap-token 动力学结合,做"在巩固高概率共享 token 的同时用 λ 守住探索熵"的混合调控;在 TSRD 中用该梯度框架定位 path-recovery 单点接管是否引入熵坍缩。

RETURN: revisit_entropy|读PDF?是(_txt 3400+行,核到§5-§7全文+附录A.1 GRPO目标+式3/4/5)|加厚?是(方法扩为A-D四块+复现锚点,相关工作6条精确化,4条公式转MathJax含分段梯度)|LaTeX公式条数 4|待核数 0

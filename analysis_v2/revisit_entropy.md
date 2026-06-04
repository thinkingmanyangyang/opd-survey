revisit_entropy | Revisiting Entropy in Reinforcement Learning for Large Reasoning Models | 天津大学 TJUNLP Lab + 天津师范大学 + 独立研究者（Renren Jin 一作；Deyi Xiong 熊德意 通讯） | 2026-04-19 v3（v1 2025-11-08）· arXiv preprint cs.CL | 主题线 L3(RLVR/GRPO)+L6(Token信用)·相关性 中

**原始论文**:https://arxiv.org/abs/2511.05993

## 一眼看懂
- 🟦 TL;DR:RLVR(可验证奖励 RL)训练 LLM 时策略熵常"坍缩"(概率质量集中到少数 token、丧失探索),导致过早收敛。本文做了一次系统性熵体检:① 实证熵与"回答多样性/校准/性能"的关系(发现熵≠性能的可靠代理,二者相关性高度任务相关;熵坍缩越严重模型越过自信/校准越差;**性能可以在不牺牲熵的情况下持续提升**);② 找出影响熵动态的三因素:clip 阈值、off-policy 更新次数、训练数据多样性;③ 理论+实证证明**正优势(positive-advantage)token 是熵坍缩的主因**;④ 据此提出 Positive-Advantage Reweighting——简单地动态调节"正优势 token"的损失权重来调熵,同时保持性能。【Abstract,§1】
- 最巧的一步:**§7.1 的梯度符号分析(式3/4)+ §7.2 的"只训 Adv≥0 vs 只训 Adv≤0"对照**。抽掉它,本文就只是一篇"熵现象观察 + 调参经验"集合;正是这个"正优势 token 推高已采样高概率 token、压低未采样低概率 token→质量集中→熵坍缩"的机制论证,把分散的现象(clip 三因素都通过"改变正/负优势 token 的相对梯度贡献"影响熵)统一起来,并直接导出干预手段(给正优势 token 的 loss 加可调权重 \(\lambda\))。【§7.1-7.2,图5】

## 为什么做
- 研究背景:RLVR(o1/DeepSeek-R1/Kimi k1.5 开创)成为提升 LLM 推理的主流;但越来越多工作发现 RLVR 会驱使熵坍缩——概率质量集中到少数 token,模型越来越偏 exploitation、不再探索新推理路径,过早陷局部最优。【§1,L42-70】
- 解决的具体痛点:已有缓解法众多,但**对 RLVR 中的熵缺乏系统研究**,三个核心问题未答:(1) 熵与性能如何相关?(§5)(2) 何因素(理论+实证)支配熵动态?(§6)(3) 如何有效调熵以提升性能?(§7)【§1,L83-90】
- 相关工作 & 各自不足(来龙去脉,据 §2):
  - **熵作为 RL 探索正则**历史悠久(Ziebart max-ent RL)。
  - **LLM 时代用熵当信号**:Dong/Zhang 等用熵决定 agent 何时求经验指导/补采样;**熵最大化目标**并入 RLVR(Shen 2025;He 2025)以鼓励探索防过早收敛——**短板**=系数难调(大则熵暴涨、小则无效)。
  - **clip 类**:**DAPO Clip-Higher**(Yu 2025)抬高重要性比率上界,防低概率正优势 token 被 clip 掉——**短板**=只缓解、不显式控熵,熵会无控漂移。
  - **协方差类理论切片**:Liu 2025 / **Cui 2025(Clip-Cov、KL-Cov)** 理论指出"log-prob 与 advantage 强正协方差的 token"主导坍缩,据此限制这类 token 更新——**短板**=干预是"限制更新"(硬截断),非连续可调,且只切单一理论切面。
  - **CE-GPPO**(Su 2025):用 stop-gradient **保留被 clip token 的梯度**缓解坍缩(与项目 memory 的 forward-hard/backward-soft 同构)。
  - **只用高熵 token 训**(Wang 2025b);**把熵并入 advantage**(Cheng 2025,Entropy-Adv)。
  - **共同空白**:多是点状缓解或单一理论切片,缺"熵-性能-校准-多样性"全景 + 跨三因素(clip/off-policy/数据)统一实证 + token 级因果归因。【§2,L145-180】
- 动机链:RLVR 强但熵坍缩→缺探索→过早收敛→需先搞清熵与性能到底什么关系(是不是越高越好)→再找熵被什么支配→再定位是哪类 token 在推坍缩→针对该类 token 设计最小干预。
- 与最近邻工作的精确Δ:vs **Cui(Clip-Cov/KL-Cov)**——同样把矛头指向"高协方差/正优势 token",但 Cui 干预="限制这类 token 更新",本文 Pos-Adv-Reweight="给正优势 token 的 loss 乘**可调连续权重 \(\lambda\)**"(更细的连续调控 + stage/epoch/entropy-guided 三种调度);vs **DAPO Clip-Higher**——本文把 Clip-Higher/Lower/Tighter/Free 放进统一 clip 阈值消融,用梯度公式解释为何 Clip-Higher 抗坍缩,且指出 Clip-Higher 不显式控熵会无控漂移,而 Pos-Adv-Reweight 能把熵稳到目标 \(\delta\) 附近。关键 Δ:**把"熵=性能代理"这一常见假设证伪**(Table1 Spearman 高度任务相关:AIME24 Avg@64 0.417、LiveCodeBench −0.894、IF-Eval 0.627),并给出"性能可不牺牲熵而升"的反例(§5.1)。【§5,Table1】

## 怎么做(到"读完能复现"的粒度)
> 本文主体=诊断扫描 + 一个轻量方法 Pos-Adv-Reweight。下面把基底目标、三轴扫描、token 级梯度归因、方法三变体逐一写到可复现。

### A. 基底:token-level、去 KL 的 GRPO(全文训练用)
对每 prompt \(x\) 采 \(G\) 个响应 \(\{y_i\}\),组相对 advantage \(\hat A_{i,t}\) 用组内 reward 的均值/标准差归一(沿 DAPO 用 token-level loss + 去掉 KL 惩罚):
\(\displaystyle J(\theta)=\mathbb E_{x\sim\mathcal D,\{y_i\}\sim\pi_{\theta_{\text{old}}}}\!\Big[\tfrac{1}{\sum_i|y_i|}\sum_{i=1}^{G}\sum_{t=1}^{|y_i|}\min\!\big(r_{i,t}(\theta)\hat A_{i,t},\ \mathrm{clip}(r_{i,t}(\theta),1-\varepsilon_{\text{low}},1+\varepsilon_{\text{high}})\hat A_{i,t}\big)\Big],\)
其中 \(r_{i,t}(\theta)\) 是重要性比率。熵的定义 \(H(\pi_\theta)\) = \(\pi_\theta\) 的 token 级平均熵;熵正则把目标加项 \(\alpha H(\pi_\theta)\)。【附录A.1】

### B. 诊断扫描(同模型/同数据,沿三轴)
平台:**Qwen2.5-Math-7B + GRPO + veRL + DAPO-Math-17K**,监测熵、reward、ID/OOD 性能、校准。
- **§5 熵 vs 性能/多样性/校准(证伪"熵=性能")**:① 熵与 N-gram 多样性强正相关(4 设置平均 Spearman 0.8795)——低熵=输出更不多样;② 训练中 response/prompt 熵都降,ID prompt 熵降更快;③ **熵不是可靠性能代理**:Table1 熵-性能 Spearman 高度任务相关(AIME24 Avg@64 0.417、LiveCodeBench −0.894 强负、IF-Eval 0.627 正)→ 取决于任务与评测指标;④ 熵坍缩越重、模型校准越差(过自信);⑤ **性能可不牺牲熵而升**(§5.1,用 Ada-Ent-Reg 把熵稳在训练前水平)。【§5,Table1】
- **§6.1 clip 阈值**:统一比 Clip-Higher / Clip-Lower(\(\varepsilon_{\text{low}}=0.28\)) / Clip-Tighter(\(\varepsilon_{\text{low}}=0.12\)) / Clip-Free(去 clip)。Clip-Higher 抗坍缩甚至升熵;Clip-Lower 坍缩最重;Clip-Tighter 缓解;**Clip-Free 熵第二高、ID/OOD 性能也第二高**(说明 off-policy 更新少时去 clip 不损稳定)。【§6.1】
- **§6.2 off-policy 更新数 \(N_{\text{update}}\in\{1,2,4\}\)**:更多更新放大熵变化、提训练 reward,但 ID Avg@64 增益 <1%、Pass@64 反降(过拟合)。表内:\(N=1\) 熵 0.118、\(N=2\) 熵 0.0229、DAPO(\(N=4\)) 熵 0.479(注:\(N=2\) 反而坍缩最重,\(N=4\) 因 Clip-Higher 熵高)。【§6.2,Table2】
- **§6.3 数据多样性**:多样性越低熵坍缩越重(K-means 选的子集比随机子集熵更低);**约 616 样本 ≈ 17k 样本性能**(数据规模非唯一决定因素,是降本启示)。【§6.3】

### C. token 级因果归因:正优势 token 是主因(§7,全文支柱)
- **梯度符号分析(§7.1,式3/4)**:对某 token \(v\) 的 logit \(z_v\) 求梯度,符号取决于"该步是否采样到 \(v\) × advantage 正负 × 是否被 clip"。当 \(v\) **未被采样**时(式3):
\(\displaystyle \frac{\partial J(\theta)}{\partial z_v}=\begin{cases}-\,r_t(\theta)\,\pi_\theta(v\mid x,y_{<t})\,\hat A_t,&\hat A_t>0\ \text{且}\ r_t(\theta)<1+\varepsilon_{\text{high}}\\[2pt]0,&\hat A_t>0\ \text{且}\ r_t(\theta)>1+\varepsilon_{\text{high}}\\[2pt]-\,r_t(\theta)\,\pi_\theta(v\mid x,y_{<t})\,\hat A_t,&\hat A_t<0\ \text{且}\ r_t(\theta)>1-\varepsilon_{\text{low}}\\[2pt]0,&\hat A_t<0\ \text{且}\ r_t(\theta)<1-\varepsilon_{\text{low}}\end{cases}\)
当 \(v\) **被采样**时(式4),把 \(\pi_\theta(v)\) 换成 \((1-\pi_\theta(v))\) 且去掉负号:\(\frac{\partial J}{\partial z_v}=r_t(\theta)\,(1-\pi_\theta(v\mid x,y_{<t}))\,\hat A_t\)(同样的 clip 边界归零)。**因 RLVR 做梯度上升**:正优势 → 抬高已采样 token 概率、压低未采样 token;因高概率 token 更易被采样,质量越堆越集中 → **熵坍缩**。负优势反之 → 压高概率已采样 token、抬未采样的 → 抗坍缩。clip 把比率截断时梯度归零,所以 **clip 阈值 = 在调节正/负优势 token 的相对梯度贡献** → 与 §6.1 实证一致。【§7.1,推导见附录E.1】
- **直接实证(§7.2,关键)**:只训 Adv≥0 token vs 只训 Adv≤0 token。**只训 Adv≥0 → 熵坍缩最重(熵 ≈ 0.0146)**;**只训 Adv≤0 → 熵很高(≈ 0.884)**(图5)。对照 Ada-Ent-Reg / Clip-Cov / KL-Cov / Entropy-Adv / Rand-Pos-Clip 同台。这是"正优势 token 主导坍缩"的因果支柱。【§7.2,图5】

### D. 方法:Positive-Advantage Reweighting(§7.3)
对 advantage \(\hat A>0\) 的 token,其损失乘可调权重 \(\lambda\)(\(\lambda=0\) 即完全不训正优势 token、只用负优势 token)。三种 \(\lambda\) 调度:
1. **Stage-based**:训练分两等阶段;前半 \(\lambda=0\)(只用非正优势 token),后半 \(\lambda\) 线性 0→1。
2. **Epoch-wise**:\(\lambda=(e-1)/(E-1)\)(\(E\) 总 epoch、\(e\) 当前 epoch),逐 epoch 线性升。
3. **Entropy-guided(主推)**:按当前熵自适应——熵超阈值 \(\delta\) 则 \(\lambda+\Delta\) 压熵,否则 \(\lambda-\Delta\) 升熵,把熵稳在 \(\delta\) 附近:
\(\displaystyle \lambda_{k+1}=\begin{cases}\mathrm{clip}(\lambda_k-\Delta,\,0,\,1),&H_k(\pi_\theta)<\delta\\[2pt]\mathrm{clip}(\lambda_k+\Delta,\,0,\,1),&\text{otherwise}\end{cases}\)
默认 \(\delta=0.2\)(与 Ada-Ent-Reg 对齐)、\(\Delta=0.05\)、\(\lambda_0=0\)。结果:Stage/Epoch 版熵先升后降;Entropy-guided 把熵稳在 0.2 附近,且拿到最高 ID Avg@64(45.66),并在 6/7 benchmark 上 Avg@64 超 Clip-Higher。**消融额外发现**:Rand-Pos-Clip(随机把一小撮正优势 token 梯度置零)就已能缓解坍缩且 Avg@64≈Clip-Cov——印证"调正优势 token 的损失权重是有效的控熵手段"。【§7.3,Table2,图5】

### 数据/模型/评测(复现锚点)
- 模型:Qwen2.5-Math-7B(主)+ Llama-3.1-8B-Instruct(泛化验证,§7.3 末确认结论可迁移);训练 GRPO@veRL@DAPO-Math-17K(token-level loss,去 KL)。
- ID 基准:AIME24/25、MATH500、AMC23、Minerva;OOD 基准:LiveCodeBench(代码)、IF-Eval(指令遵循);指标 Avg@64 / Pass@64。
- 关键数字:Table2 各变体 ID Avg@64 多在 43-45.7(差距小),熵跨度大(0.015~0.88);GRPO(\(N=1\)) ID Avg@64 44.12;Pos-Adv-Reweight(Entropy-guided) 45.66。【Table1/2,§5-7】

## 靠不靠谱
- baseline 公平吗:同模型/同数据/同框架横扫十余变体(clip×4、\(N_{\text{update}}\)×3、数据多档、方法×多),Entropy-guided 与 Ada-Ent-Reg 特意对齐 \(\delta=0.2\) 以可比,公平性强。但**主实验仅 Qwen2.5-Math-7B + 数学训练数据**(Llama-3.1-8B 只作泛化点验,OOD 只是评测不是训练域多样)。
- 看着强但没回答核心问题:Pos-Adv-Reweight 的性能增益其实很小(ID Avg@64 45.66 vs 默认 GRPO 44.12,约 +1.5 点),主卖点更像"能稳熵且不掉点"而非"显著涨点"——价值偏"可控性/诊断"而非"性能突破";"熵不是可靠性能代理"很有用但也意味着调熵本身不直接等于提性能。
- 假设与失效边界:【原文】① 主体基于 GRPO + Qwen2.5-Math-7B + 数学数据;② 梯度分析(式3/4)是"token 未/已采样 + clip 边界"下的近似(推导在附录E.1);③ "性能可不牺牲熵而升"基于把熵稳在训练前水平(\(\delta\) 由 1000 prompt 实测)。【推断】结论强绑定 GRPO 族的 clip+advantage 结构;换 reward 形态(如 OPD 的 teacher token reward 而非二元 advantage)时,"正优势 token 主导坍缩"机制不一定平移——OPD 无显式 advantage 正负之分,熵动态由师生分布对齐主导(见 rethink_opd),二者机制不同。
- 祛魅总结:真贡献=把 RLVR 的熵从"经验调参"提升为"现象全景(熵 vs 多样性/校准/性能)+ 三因素 + 正优势 token 主因的理论-实证闭环",并给出可解释的最小干预 Pos-Adv-Reweight。两个最有用论断:**"熵不是性能的可靠代理(高度任务相关)"**和**"正优势 token 是熵坍缩主因"**。【推断】被高估的是方法本身(增益微弱,更像"可控旋钮");被低估/真有价值的是诊断结论(尤其 §5.2 证伪"熵=性能"、§6.3 "600≈17k 样本")。它是 RLVR 熵线的扎实"综述+体检"型工作,非范式突破。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=verifier 二元奖励→GRPO 组相对 advantage(本文重点是 advantage 的正/负如何经梯度影响熵) | **改什么**=策略参数 θ;干预手段=正优势 token 的损失权重 λ | **何时改**=在线 RLVR;λ 按 stage/epoch/entropy 三种调度动态调整 | **免梯度?**=否(GRPO policy gradient) | **记忆-技能生命周期**=无显式记忆/技能库 | **防遗忘机制**=无;但"防熵坍缩=保探索"间接防"多样性遗忘"(熵坍缩↔过自信/校准变差是本文核心关联)
- ⑦ 开源代码+框架/harness:https://github.com/cordercorder/EntropyRL(v1 元信息已核:约 7.8MB,以 veRL 为底,train_scripts/ 提供 ada_ent_reg.sh、clip_cov.sh、kl_cov.sh、entropy_adv.sh、pos_adv_reweight.sh、rand_pos_clip.sh 等;Clip-Higher/Adv≤0/≥0 等只需对 veRL 小改、未单独给脚本);框架 **veRL(GRPO)**。【§1 脚注,§4】
- 💰 资源/成本与可扩展性:原文未给 GPU 时/卡数。可推断:单模型 7B + GRPO,扫了十余配置(clip×4、Nupdate×3、数据子集多档、方法×多)——总算力不小但单 run 是常规 7B RLVR 规模;§6.3 "616 样本≈17k 样本"是个降本启示。【§6】
- 🎯 对"探索-巩固"对标:**间接支撑(探索侧)+ 可借诊断,非直接竞品**。映射:本文的"探索=高熵=保留多样推理路径"对应 TSRD 的"探索/选路"——其核心警示"正优势 token 推动熵坍缩、过度 exploitation"提醒 TSRD 在"巩固/固化有效路径"时要防止把探索能力一并固化掉(过早熵坍缩=过早巩固)。可借组件:① **token 级正/负优势梯度符号分析(式3/4)**——可迁移用于分析 TSRD 中"哪些 token 的更新在收缩 vs 扩张策略支撑",尤其判断 path-recovery 监督会不会误导致熵坍缩;② **熵≠性能、按任务看相关性**的方法论(评估 TSRD 时别把熵当目标);③ Entropy-guided λ 调度作为"巩固强度"的自适应旋钮思路(熵高时多固化、熵低时松手探索)——与"探索-巩固"的动态平衡天然同构。缺口:本文是纯 RLVR(无 teacher、无蒸馏、无记忆),与 TSRD 的"teacher 脚手架 + MTP 前瞻 + 记忆固化"三要素都不沾;且其机制(advantage 正负)在 OPD 设定下不直接适用。判定:**探索侧的机制参考与诊断工具来源,非方法竞品**。
- 🔭 开放问题/未来方向:【原文】未设独立 Future Work 章;隐含方向=把熵调控与性能解耦后如何"既保探索又稳定提升"、Pos-Adv-Reweight 在更多模型/域的验证。【推断】把"正优势 token 主导坍缩"的分析推广到 OPD/自蒸馏的 token reward 设定(检验是否同样有"某类 token 主导坍缩");把 Entropy-guided λ 与 OPD 的 overlap-token 动力学结合,做"在巩固高概率共享 token 的同时用 λ 守住探索熵"的混合调控;在 TSRD 中用该梯度框架定位 path-recovery 单点接管是否引入熵坍缩。

RETURN: revisit_entropy|读PDF?是(_txt 3400+行,核到§5-§7全文+附录A.1 GRPO目标+式3/4/5)|加厚?是(方法扩为A-D四块+复现锚点,相关工作6条精确化,4条公式转MathJax含分段梯度)|LaTeX公式条数 4|待核数 0

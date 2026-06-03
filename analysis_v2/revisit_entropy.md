revisit_entropy | Revisiting Entropy in Reinforcement Learning for Large Reasoning Models | 天津大学 TJUNLP Lab + 天津师范大学 + 独立研究者（Renren Jin 一作；Deyi Xiong 熊德意 通讯） | 2026-04-19 v3（v1 2025-11-08）· arXiv preprint cs.CL | 主题线 L3(RLVR/GRPO)+L6(Token信用)·相关性 中

**原始论文**:https://arxiv.org/abs/2511.05993

## 一眼看懂
- 🟦 TL;DR:RLVR(可验证奖励 RL)训练 LLM 时策略熵常"坍缩"(概率质量集中到少数 token、丧失探索),导致过早收敛。本文做了一次系统性熵体检:① 实证熵与"回答多样性/校准/性能"的关系(发现熵≠性能的可靠代理,二者相关性高度任务相关;熵坍缩越严重模型越过自信/校准越差;**性能可以在不牺牲熵的情况下持续提升**);② 找出影响熵动态的三因素:clip 阈值、off-policy 更新次数、训练数据多样性;③ 理论+实证证明**正优势(positive-advantage)token 是熵坍缩的主因**;④ 据此提出 Positive-Advantage Reweighting——简单地动态调节"正优势 token"的损失权重来调熵,同时保持性能。【Abstract,§1】
- 最巧的一步:**§7.1 的梯度符号分析(式3/4)+ §7.2 的"只训 Adv≥0 vs 只训 Adv≤0"对照**。抽掉它,本文就只是一篇"熵现象观察 + 调参经验"集合;正是这个"正优势 token 推高已采样高概率 token、压低未采样低概率 token→质量集中→熵坍缩"的机制论证,把分散的现象(clip 三因素都通过"改变正/负优势 token 的相对梯度贡献"影响熵)统一起来,并直接导出干预手段(给正优势 token 的 loss 加可调权重 λ)。

## 为什么做
- 研究背景:RLVR(o1/DeepSeek-R1/Kimi k1.5 开创)成为提升 LLM 推理的主流;但越来越多工作发现 RLVR 会驱使熵坍缩——概率质量集中到少数 token,模型越来越偏 exploitation、不再探索新推理路径,过早陷局部最优。【§1,L42-70】
- 解决的具体痛点:已有缓解法众多(熵最大化项、DAPO Clip-Higher、Clip-Cov/KL-Cov 限制高 log-prob×advantage 协方差 token、CE-GPPO stop-grad 保留被 clip 梯度、只用高熵 token 训练等),但**对 RLVR 中的熵缺乏系统研究**,三个核心问题未答:(1) 熵与性能如何相关?(2) 何因素(理论+实证)支配熵动态?(3) 如何有效调熵以提升性能?【§1,L71-90】
- 相关工作 & 各自不足:熵作为 RL 探索正则历史悠久;LLM 时代有"用熵决定何时求经验指导/补采样"(Dong/Zhang)、熵最大化目标(Shen;He)、Clip-Higher(DAPO)、Clip-Cov/KL-Cov(Cui,理论指出强正协方差 token 主导坍缩)、CE-GPPO(Su,stop-grad 保梯度)、只用高熵 token(Wang)、把熵并入 advantage(Cheng)。各自多是**点状缓解或单一理论切片**,缺"熵-性能-校准-多样性"全景 + 跨三因素的统一实证。【§2,L145-180】
- 动机链:RLVR 强但熵坍缩→缺探索→过早收敛→需先搞清熵与性能到底什么关系(是不是越高越好)→再找熵被什么支配→再定位是哪类 token 在推坍缩→针对该类 token 设计最小干预。
- 与最近邻工作的Δ:vs Cui(Clip-Cov/KL-Cov)——同样把矛头指向"高协方差/正优势 token",但 Cui 的干预是"限制这类 token 更新",本文的 Pos-Adv-Reweight 是"给正优势 token 的 loss 乘可调权重 λ"(更细的连续调控,且给出 stage/epoch/entropy-guided 三种调度);vs DAPO Clip-Higher——本文把 Clip-Higher/Lower/Tighter/Free 放进统一 clip 阈值消融,用梯度公式解释为何 Clip-Higher 抗坍缩(放宽上界让低概率正优势 token 不被 clip、得以涨概率)。关键 Δ:**把"熵=性能代理"这一常见假设证伪**(Table1 Spearman 相关高度任务相关,LiveCodeBench 强负相关 -0.89、IF-Eval 正相关 0.63),并给出"性能可不牺牲熵而升"的反例(§5.1 Ada-Ent-Reg)。【§5.1-5.2,Table1】

## 怎么做 + 靠不靠谱
- 方法流水线:本文主体是诊断+一个轻量方法。诊断:在 Qwen2.5-Math-7B + GRPO + veRL + DAPO-Math-17K 上,沿三轴扫描(clip 阈值 §6.1 / off-policy 更新数 Nupdate §6.2 / 数据多样性 §6.3),监测熵、reward、ID/OOD 性能、校准。方法 Pos-Adv-Reweight:对优势 Â>0 的 token 的损失乘权重 λ∈[0,1],三种调度:① Stage-based(前半 λ=0 只用非正优势 token,后半 λ 线性 0→1);② Epoch-wise(λ=(e-1)/(E-1) 按 epoch 线性升);③ Entropy-guided(式5:熵>δ 则 λ+=Δ 压熵,否则 λ-=Δ 升熵,δ=0.2、Δ=0.05、λ0=0)。【§4,§6,§7.3】
- 逐组件必要性(本文消融极充分,Table2 列了 ~17 个变体):
  - **clip 阈值**(§6.1):Clip-Higher 抗坍缩甚至升熵;Clip-Lower 坍缩最重;Clip-Tighter 缓解;Clip-Free 熵第二高、ID/OOD 性能也第二高(说明 off-policy 更新少时去 clip 不损稳定)。
  - **off-policy 更新数**(§6.2):Nupdate∈{1,2,4},更多更新放大熵变化、提训练 reward,但 ID Avg@64 增益<1%、Pass@64 反降(过拟合)。
  - **数据多样性**(§6.3):多样性越低熵坍缩越重(K-means 子集比随机子集熵更低);**616 样本≈17k 样本性能**(数据规模非唯一决定因素)。
  - **正优势 token 是主因**(§7.2 关键验证):只训 Adv≥0→熵坍缩最重(熵 0.0146);只训 Adv≤0→熵很高(0.884)。这是全文因果支柱。
  - **Pos-Adv-Reweight 三变体**(§7.3):Entropy-guided 拿到最高 ID Avg@64(45.66),且能把熵稳在 δ 附近;stage/epoch 版熵先升后降。
  - baseline 对照(Ada-Ent-Reg / Clip-Cov / KL-Cov / Entropy-Adv / Rand-Pos-Clip)均在 Table2/图5 同台。
- 关键机制/公式(直觉,不复述符号):GRPO 做梯度上升时,对某 token logit z_v 的梯度符号取决于"该 token 这步是否被采样 × 优势正负 × 是否被 clip"(式3/4)。结论:**正优势→抬高已采样 token 概率、压低未采样 token**;因高概率 token 更易被采样,这就把质量越堆越集中→熵坍缩。负优势反之→把高概率已采样 token 压下、抬未采样的→抗坍缩。clip 把比率截断时梯度归零,所以 clip 阈值=在调节"正优势 vs 负优势 token 的相对梯度贡献"→与 §6.1 实证一致。
- 实验与证据:模型 Qwen2.5-Math-7B;训练 GRPO@veRL@DAPO-Math-17K;ID 基准 AIME24/25、MATH500、AMC23、Minerva;OOD 基准 LiveCodeBench(代码)、IF-Eval(指令遵循);指标 Avg@64 / Pass@64。关键数字:Table1 熵-性能 Spearman(AIME24 Avg@64 0.417、LiveCodeBench -0.894、IF-Eval 0.627→高度任务相关);Table2 各变体 ID Avg@64 多在 43-45.7 区间(差距小),熵值跨度大(0.015~0.88);616 vs 17k 样本性能相当。【Table1/2,§5-7】
- baseline 公平吗:同模型/同数据/同框架横扫十余变体,Entropy-guided 与 Ada-Ent-Reg 特意对齐 δ=0.2 以可比,公平性强。但**所有实验仅 Qwen2.5-Math-7B 单模型 + 数学训练数据**,泛化性未验证(OOD 只是评测,不是训练域多样)。
- 看着强但没回答核心问题:Pos-Adv-Reweight 的性能增益其实很小(ID Avg@64 45.66 vs 默认 GRPO 44.12,约 +1.5 点),主卖点更像是"能稳熵且不掉点"而非"显著涨点"——方法价值偏"可控性/诊断"而非"性能突破";"熵不是可靠性能代理"的结论很有用但也意味着调熵本身不直接等于提性能。
- 假设与失效边界:【原文】① 全部基于 GRPO + 单模型 Qwen2.5-Math-7B + 数学数据;② 梯度分析(式3/4)在"token 未/已被采样 + clip 边界"下的近似(详细推导在附录 E.1);③ "性能可不牺牲熵而升"基于 Ada-Ent-Reg 把熵稳在训练前水平(δ 由 1000 prompt 实测)。【推断】结论强绑定 GRPO 族的 clip+advantage 结构;换 reward 形态(如 OPD 的 teacher token reward 而非二元 advantage)时,"正优势 token 主导坍缩"的机制不一定平移——OPD 没有显式 advantage 正负之分,熵动态由师生分布对齐主导(见 rethink_opd),二者机制不同。
- 祛魅总结:真贡献=把 RLVR 的熵从"经验调参"提升为"现象全景(熵 vs 多样性/校准/性能)+ 三因素 + 正优势 token 主因的理论-实证闭环",并给出可解释的最小干预 Pos-Adv-Reweight。两个最有用的论断:**"熵不是性能的可靠代理(高度任务相关)"**和**"正优势 token 是熵坍缩主因"**。【推断】被高估的是方法本身(性能增益微弱,更像"可控旋钮");被低估/真正有价值的是诊断结论(尤其 §5.2 证伪"熵=性能"、§6.3 "600≈17k 样本")。它是 RLVR 熵线的扎实"综述+体检"型工作,非范式突破。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=verifier 二元奖励→GRPO 组相对 advantage(本文重点是 advantage 的正/负如何经梯度影响熵) | **改什么**=策略参数 θ;干预手段=正优势 token 的损失权重 λ | **何时改**=在线 RLVR;λ 按 stage/epoch/entropy 三种调度动态调整 | **免梯度?**=否(GRPO policy gradient) | **记忆-技能生命周期**=无显式记忆/技能库 | **防遗忘机制**=无;但"防熵坍缩=保探索"间接防"多样性遗忘"(熵坍缩↔过自信/校准变差是本文核心关联)
- ⑦ 开源代码+框架/harness:https://github.com/cordercorder/EntropyRL(v1 元信息已核:约 7.8MB,以 veRL 为底,train_scripts/ 提供 ada_ent_reg.sh、clip_cov.sh、kl_cov.sh、entropy_adv.sh、pos_adv_reweight.sh、rand_pos_clip.sh 等;Clip-Higher/Adv≤0/≥0 等只需对 veRL 小改、未单独给脚本);框架 **veRL(GRPO)**。【§1 脚注,§4】
- 💰 资源/成本与可扩展性:原文未给 GPU 时/卡数。可推断:单模型 7B + GRPO,扫了十余配置(clip×4、Nupdate×3、数据子集多档、方法×多)——总算力不小但单 run 是常规 7B RLVR 规模;§6.3 "616 样本≈17k 样本"是个降本启示。【§6】
- 🎯 对"探索-巩固"对标:**间接支撑(探索侧)+ 可借诊断,非直接竞品**。映射:本文的"探索=高熵=保留多样推理路径"对应 TSRD 的"探索/选路"——其核心警示"正优势 token 推动熵坍缩、过度 exploitation"提醒 TSRD 在"巩固/固化有效路径"时要防止把探索能力一并固化掉(过早熵坍缩=过早巩固)。可借组件:① **token 级正/负优势梯度符号分析(式3/4)**——可迁移用于分析 TSRD 中"哪些 token 的更新在收缩 vs 扩张策略支撑",尤其判断 path-recovery 监督会不会误导致熵坍缩;② **熵≠性能、按任务看相关性**的方法论(评估 TSRD 时别把熵当目标);③ Entropy-guided λ 调度作为"巩固强度"的自适应旋钮思路(熵高时多固化、熵低时松手探索)——这与"探索-巩固"的动态平衡天然同构。缺口:本文是纯 RLVR(无 teacher、无蒸馏、无记忆),与 TSRD 的"teacher 脚手架 + MTP 前瞻 + 记忆固化"三要素都不沾;且其机制(advantage 正负)在 OPD 设定下不直接适用。判定:**探索侧的机制参考与诊断工具来源,非方法竞品**。
- 🔭 开放问题/未来方向:【原文】未设独立 Future Work 章;隐含方向=把熵调控与性能解耦后如何"既保探索又稳定提升"、Pos-Adv-Reweight 在更多模型/域的验证。【推断】把"正优势 token 主导坍缩"的分析推广到 OPD/自蒸馏的 token reward 设定(检验是否同样有"某类 token 主导坍缩");把 Entropy-guided λ 与 OPD 的 overlap-token 动力学结合,做"在巩固高概率共享 token 的同时用 λ 守住探索熵"的混合调控;在 TSRD 中用该梯度框架定位 path-recovery 单点接管是否引入熵坍缩。

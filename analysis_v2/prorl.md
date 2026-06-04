prorl | ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models | NVIDIA(Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, Yi Dong;长程 RLVR / 推理边界谱系) | 2025-05 预印本·arXiv v1(arXiv:2505.24864v1,2025-05-30)·Under review | 主题线 L3(RLVR/GRPO 训练与稳定化)·相关性 高(长程探索-稳定性,与"探索"一侧直接相关)

**原始论文**:https://arxiv.org/abs/2505.24864

## 一眼看懂
- 🟦 TL;DR:学界争论"RL 到底是发现新推理能力,还是只放大 base 已潜在的高奖励输出"(后者由 Yue/Dang/Zhao 等用 pass@k 论证)。本文用一套能跑 **2k+ 步不崩**的长程 RL 配方(ProRL = GRPO + DAPO 解耦 clip/动态采样 + **KL 正则 + 周期性把参考策略硬重置为近期在线快照**),在 **136K 五域可验证数据**上训练,证明:在 **base 即使大量采样(pass@128/256)也完全做不出**的任务上(boxnet:base pass@k 全 0.000,RL 模型达 1.000),RL 后能做对,从而主张 RL **确能扩展**推理边界(而非仅放大)——但前提是足够长的训练 + 良好初始化 + base 本来弱的领域。产出 **Nemotron-Research-Reasoning-Qwen-1.5B**(号称当时最强 1.5B 推理模型)。【原文 Abstract 行 14-34,§4.2-4.3,Figure 4-5,Table 3】
- 最巧的一步:**参考策略硬重置(Reference Policy Reset)**。抽掉它,长程 RL 会卡死——因为随训练推进 KL 项**渐主导损失**、policy 更新变小、过早收敛;周期性把 \(\pi_{\text{ref}}\) 硬重置为近期在线快照 \(\pi_\theta\)、并重置优化器状态,就能"在保留 KL 稳定收益的同时持续提升"。它和"保留 KL"是一对:单留 KL 会停滞,单去 KL(DAPO/ORZ 主流做法)会熵坍缩;ProRL 用"KL + 定期重置"两者兼得。【原文 §2.3.1 行 223-228】
  - 注:这是**机制层面**的命门(论文用训练曲线 + 直觉论证),但论文**未做"逐组件剥离"消融**(无 w/o reset / w/o KL 的对照实验),故"最巧一步"的因果强度部分依赖叙述而非受控实验。【推断:全文无 ablation 表】

## 为什么做
- 研究背景:o1/DeepSeek-R1 等推理模型靠 test-time scaling(长 CoT + 探索/验证/回溯)在数学/代码大幅提升,RLVR(针对可验证奖励的 RL)是核心驱动且能缓解 reward hacking。【原文 §1 行 37-47】
- 解决的具体痛点:① 流行观点(Yue et al.[13]、Dang et al.[14]、Zhao et al.[15])认为 RL 不带来超越 base 的新能力,只放大已潜在输出(base 大量采样仍做不出的题,RL 后也不行);② 长程 RL 面临**熵坍缩**与不稳定——输出分布过早变尖、熵骤降、探索受限,而 GRPO 的学习信号**依赖一组多样采样**来估相对优势,熵坍缩使更新偏置、训练停滞;③ 单纯提高采样温度只**延缓**而非阻止熵坍缩(论文实测,行 188-190)。【原文 §1 行 104-114,§2.2.1 行 182-190】
- 相关工作 & 各自不足(精确到论点):
  - **[13] Yue et al.**——用 pass@k 论证 RLVR 不扩展能力(甚至退化);本文反驳其**两个方法论局限**:(1) 过度依赖**数学**这种"预训练+后训练都过度训练"的领域,本就压缩了探索空间;(2) RL 步数太短(typically <几百步,行 114),模型还没充分探索就停。
  - **[14] Dang et al.**——推导 pass@k 上界并观察训练中 pass@k 下降。
  - **[15] Zhao et al.("Echo chamber")**——RL 收敛到主导分布、只放大预训练模式。
  - **去 KL 阵营 [4,7,5,18]（DAPO/ORZ 等）**——主张移除 KL,理由是 CoT 任务模型训练中"自然发散";本文观察到**这一观点多适用于未经 SFT 的 base 模型**,而 ProRL 从**已 SFT、能产连贯 CoT 的良好起点**(DeepSeek-R1-Distill-Qwen-1.5B)出发,此时保留 KL 对稳定与持续熵仍有益(行 218-222)。
  - 关键差异:本文不只改 clip/归一化,而是把"**长程 + 多域 + 稳定化(KL+重置)+ 良好起点**"四者绑定,并用 **Creativity Index**(轨迹与预训练语料 DOLMA 的重叠,越低=越新)+ **base 全 0 任务上 RL 做对**作为"扩展 vs 放大"的更强证据。
- 动机链:质疑"RL 不扩展能力"假设 → 怀疑那是"领域过窄 + 训练过短"两个方法论缺陷所致 → 设计能长跑(2k+ 步)且覆盖五域(数学/代码/STEM/逻辑谜题 Reasoning Gym/指令遵循)的 ProRL,并用稳定化压住熵坍缩 → 验证"扩展边界"与 base 任务能力、训练时长强相关。【原文 §1 行 115-126】
- 与最近邻工作的Δ:vs DAPO/ORZ 等"去 KL"主流——ProRL **反其道保留 KL**,并把适用前提**条件化**("从已能产连贯 CoT 的良好初始化出发"才保 KL);又新增**参考策略重置**解决"KL 项后期主导→更新停滞"。vs [13/14/15]——用 Creativity Index + base 全 0 任务 RL 做对,作"扩展"的更强证据,且诚实给出"只在 base 弱领域扩展、base 强领域 pass@128 反降"的边界。

## 怎么做 + 靠不靠谱
- 方法流水线(§2,输入→输出讲清):
  - **输入**:良好初始化 base \(\pi_\theta\)(DeepSeek-R1-Distill-Qwen-1.5B,已能产连贯 CoT);136K 五域可验证数据(每域配二值或连续奖励)。**输出**:Nemotron-Research-Reasoning-Qwen-1.5B,一个跨域 generalist。
  - **① 基座 GRPO(去 critic)**:
    \[
    L_{\text{GRPO}}(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\!\left[\min\!\Big(r_\theta(\tau)A(\tau),\ \operatorname{clip}(r_\theta(\tau),1-\epsilon,1+\epsilon)A(\tau)\Big)\right],\quad r_\theta(\tau)=\frac{\pi_\theta(\tau)}{\pi_{\text{old}}(\tau)}
    \]
    优势用组内标准化(去 PPO 的 critic):\(A(\tau)=\dfrac{R_\tau-\operatorname{mean}(\{R_i\}_{i\in G(\tau)})}{\operatorname{std}(\{R_i\}_{i\in G(\tau)})}\)(Eq.2)。
  - **② 叠 DAPO 两组件**(§2.3,抗熵坍缩):
    - **解耦 clip**(Eq.3):上下界拆成独立超参 \(\operatorname{clip}(r_\theta(\tau),1-\epsilon_{\text{low}},1+\epsilon_{\text{high}})\);调高 \(\epsilon_{\text{high}}\)="clip-higher",**抬升低概率 token 的上行空间**、保熵、减 mode collapse(实测有效,行 200-202)。本文取 \(\epsilon_{\text{low}}=0.2,\ \epsilon_{\text{high}}=0.4\)。
    - **动态采样**:过滤掉 acc=0 或 1 的 prompt(无学习信号),聚焦中等难度,维持多样学习信号。
  - **③ 叠 KL 正则**(Eq.4):\(L_{\text{KL-RL}}(\theta)=L_{\text{GRPO}}(\theta)-\beta\,D_{\text{KL}}(\pi_\theta\,\|\,\pi_{\text{ref}})\)。作用三重:维持熵、防 online policy 偏离稳定参考太远、**抑制对 spurious reward 的过拟合**(行 215-217)。
  - **④ 参考策略硬重置**(§2.3.1):验证集(blended validation)停滞/退化时,把 \(\pi_{\text{ref}}\leftarrow\) 近期在线 \(\pi_\theta\) 的快照 + **重置优化器状态**;全程反复应用,避免过早收敛、鼓励长训练。
  - **数据流动 / 何时重置**:用从评测基准混出的 validation set 监控;性能停滞即触发硬重置(既恢复稳定性,又**促进 policy 进一步偏离 base**,行 261-263)。多数训练 response 上限 8k token,**末段提至 16k**(行 264+)。
- 逐组件必要性:**全文无受控 ablation 表**。各组件必要性靠"机制论证 + 训练曲线"支撑:
  - 温度只**延缓**熵坍缩(行 188-190,实测论证,且仍用高温 1.2 抬初始熵);
  - clip-higher 保熵(行 200-202,观察);
  - KL 保熵 + 防漂移 + 抑 spurious-reward 过拟合(行 215-217);
  - 重置防"KL 后期主导→更新趋零→停滞"(行 223-228)。
  - **没有"去掉某组件后掉多少分"的对照实验**——这是方法学明显短板:无法定量归因"重置"vs"仅 KL"vs"仅 DAPO"三者的边际贡献。【推断:全文 grep 无 ablation/无 w/o reset 实验】
- 关键机制/公式(直觉):
  - 核心张力=**熵坍缩 vs 探索预算**。GRPO 需一组多样采样才能估好相对优势 \(A(\tau)\),一旦熵坍缩(分布变尖)就退化为偏置更新、停滞。clip-higher 给低概率 token 更大上行空间(保探索);KL 把策略**拴在稳定参考**附近(防崩);而 KL 久了会"拽太紧"(\(\beta D_{\text{KL}}\) 项主导)使更新趋零——**重置=松开缰绳再系到新位置**(\(\pi_{\text{ref}}\) 跟进 + 优化器清零),故能"长跑且不早收敛"。
  - **pass@k 上界**(引 Dang et al.):上界随期望 pass@1 升、随其方差降;ProRL 把整体分布**右移**(pass@1 提升)足以盖过"分布变尖→方差变化"的负效应,从而在多数复杂任务上 pass@128 也升——但**数学等 base 已饱和领域**例外(见下"诚实反例")。
- 实验与证据:
  - 数据集/设置:自构 **136K** 可验证问题(数学/代码/STEM/逻辑谜题 Reasoning Gym/指令遵循,每类配二值或连续奖励,组成见 Appendix D);基座 **DeepSeek-R1-Distill-Qwen-1.5B**。框架 **veRL**。超参:\(\epsilon_{\text{low}}=0.2,\ \epsilon_{\text{high}}=0.4\)、动态采样过滤 acc∈{0,1}、每 prompt **n=16**、context 8096、**温度 1.2**、**batch 256 / mini-batch 64**(每 rollout 4 次梯度更新)、AdamW 常数 lr **2×10⁻⁶**、**4×8 H100-80GB**、约 **16k GPU·小时**;多数训练 response 上限 8k,末段提 16k。【§3.2 行 250-257】
  - 关键数字 — **增益数值有两套口径(论文内部不一致,已并列记录)**:
    - **§1(行 123-124)**:数学 **+14.7%**、代码 **+13.9%**、逻辑 **+54.8%**、STEM **+25.1%**、指令 **+18.1%**。
    - **§3(行 236-237)**:数学 **+15.7%**、代码 **+14.4%**、STEM **+25.9%**、指令 **+22.0%**、逻辑 **+54.8%**;并超领域专用基线(数学 +4.6%、代码 +6.5%)。
    - 两套差异主要在指令(18.1% vs 22.0%)、数学(14.7% vs 15.7%)、STEM/代码;**原因论文未说明**,本分析以 §3/结果节为准(与 Table 对应)。【已核行 123-124 vs 236-237】〔待核:两套口径差异原文无解释〕
  - **边界扩展最强证据(Figure 5,boxnet OOD)**:base pass@k 在 k=1,2,4,…,256 **全为 0.000**;RL 模型 final 达 **1.000**(intermediate 0.994@128→1.000@256),且 final 在所有 k 上一致 ≥ intermediate(行 1032-1062)。Table 3 数值:base boxnet **0.00**、R1-7B 1.71、Nemotron-1.5B 远超(行 363-379)。
  - **三 regime(Figure 4,关键诚实反例)**:(1) **Diminished**:部分**数学**基准 pass@1 升但 **pass@128 反降**(base 本就强、RL 只是把分布变尖、牺牲多样性,与 [13] 一致,行 992-997);(2) **Plateau**:pass@1/128 都升但**早期即饱和**,ProRL 额外步数收益微(行 1000-1003);(3) **Sustained**:复杂任务(如 coding)随长训练持续提升、真正扩展边界(行 1004-1008)。即**ProRL 并非到处都扩展边界**,只在 base 弱的复杂任务上扩展。§4.2(行 517-527)进一步:边界扩展(pass@128)与 base 初始能力**负相关**——base 高 pass@128 的任务 RL 后边界反而收窄/负增益,base 低 pass@128 的任务 RL 最有效。
  - **新颖性证据**:Creativity Index(对 DOLMA 语料的非重叠度)随训练上升 **3.84→4.42→4.70**(Figure 1 中);base 已熟悉的任务 creativity 低(行 530-535)。
  - baseline 公平性:与同 base(R1-Distill-1.5B)及领域专用基线(DeepScaleR、DeepCoder 等)对比,提供 R1-7B 作灰色参考;较公平。但**仅 1.5B 单规模、单 base**。
  - 看着强但没回答的:① 全程**无组件消融**,稳定化各组件贡献无法定量归因;② "扩展边界"的判定高度依赖 pass@128 与 Creativity Index 两个测度(pass@k 对 k/温度敏感,Creativity Index 是语料重叠代理);③ §1 vs §3 增益数值不一致。
- 假设与失效边界:【原文】① 适用前提="良好初始化(已蒸馏、能产连贯 CoT)的起点",base 模型直接 RL 不在此列(行 218-222);② 边界扩展只在 base 初始弱的领域显著,base 强领域反而 pass@128 退化(Figure 3/4,§4.2);③ KL 项后期会主导→需重置(行 223-224)。【推断】④ 仅 1.5B、单一 base,更大模型/不同起点的可迁移性未验证;⑤ "扩展 vs 放大"结论强度依赖 pass@128(大 k)与 Creativity Index 的稳健性,换测度可能动摇定性结论;⑥ 16k GPU·hr 成本对小团队复现门槛高,且无训练代码、需自行在 veRL 上拼装。
- 祛魅总结:【推断】真贡献=① 以**强反例**(boxnet base 全 0、RL 满分)+ Creativity Index 把"RL 能扩展边界"从口号变成可观测现象,并诚实给出"只在 base 弱领域、Diminished/Plateau/Sustained 三态"的边界条件;② "KL + 参考重置"是长程 RL 实用稳定化配方。被高估处:标题"Prolonged RL Expands Reasoning Boundaries"偏强——实证里数学等领域 pass@128 反降,普适性被三 regime 打折;"扩展边界"对测度敏感。被低估处:对"去 KL 是否普适"的细致反驳(**限定良好初始化前提**)其实调和了 ORZ/DAPO 的对立主张,这一**前提条件化**洞见比标题更有价值。**数值一致性瑕疵**:§1(行 123-124)vs §3(行 236-237)两套增益数不同,原因论文未说明,本分析采 §3。

## 结构化抽取
- 🎯 机制速览6轴:
  - **学什么信号**:可验证结果奖励(二值/连续,跨数学/代码/STEM/逻辑/指令五域),GRPO 组内归一优势 \(A(\tau)\);无 teacher/无 KD。
  - **改什么**:policy 参数 \(\theta\)(GRPO 梯度);参考策略 \(\pi_{\text{ref}}\) 周期性被硬重置(非梯度,是 copy 快照 + 重置优化器)。
  - **何时改**:纯在线 on-policy RL,长达 2k+ 步;重置由 blended validation 停滞触发。
  - **免梯度?**:否,policy 全程梯度更新。
  - **记忆-技能生命周期**:无外部记忆/技能库;多域"技能"通过长程 RL 固化进单一 generalist 参数;\(\pi_{\text{ref}}\) 重置=滚动更新"稳定锚点"。
  - **防遗忘机制**:**KL-to-reference 正则(Eq.4)是核心稳定/防漂移手段**(防偏离稳定参考、抑制对 spurious reward 过拟合);参考重置避免 KL 拽死。这是"防(过快)漂移/防熵坍缩"而非"跨任务防遗忘";多域联合训练间接缓解单域过拟合。
- ⑦ 开源代码+框架/harness:**仅发布模型权重**——https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B(Abstract 行 33-34)。**无独立训练代码仓库**(CloneTier=B,只记链接不 clone)。框架=**veRL**(行 250);算法=GRPO + DAPO(解耦 clip + 动态采样)+ KL 正则 + 参考策略重置。复现需依论文超参在 veRL 上自行拼装。【未完整克隆原因:weights-only 无训练代码,无可 clone 的方法仓;权重仓为大模型权重,按 B 层规则只记链接】
- 💰 资源/成本与可扩展性:**4×8 NVIDIA H100-80GB、约 16k GPU·小时**(行 256-257);n=16 rollout、batch 256、2k+ 步、response 上限 8k(末段 16k)。成本较高,小团队复现门槛大;可扩展性=单 1.5B 验证,更大规模未试(论文自陈)。
- 🎯 对"探索-巩固"对标:**支撑"探索"一侧(探索预算/边界扩展),但与"教师脚手架"范式正交**。ProRL 证明的是"给足探索预算 + 压住熵坍缩,RL 能发现 base 走不通的新路径"——这正对应 idea 里"探索=发现有效行为/路径"的可行性论据(且给了"base 弱领域才扩展"的边界条件,对 TSRD 选题域有参考)。但它**没有 teacher、没有脚手架、没有 path-recovery、没有 MTP 前瞻**,巩固靠 KL 锚定 + 参考重置而非教师指导。可借组件:① **参考策略重置**这一"滚动锚点"技巧可直接用于 TSRD 长程训练防停滞;② **KL 保留 + 前提条件化**(良好初始化才保 KL)的洞见,提示"教师脚手架蒸馏 + KL 锚定"在 on-policy 蒸馏中如何设;③ **pass@128 / Creativity Index** 作"是否真扩展(而非放大)"的评测协议,可验证 TSRD 是否教会了新路径(尤其"base 弱领域才扩展"这一规律,提示脚手架若想突破 base 强领域的 Diminished 困境需要新机制)。缺口:无"主动选路/偏向可走通开头"机制(纯靠随机采样 + 高温),无"自选恢复分支"巩固,无前瞻。一句判定:**作"探索可行性 + 长程稳定化技巧 + 边界评测协议"的支撑/工具来源**,而非"探索-巩固"机制的直接基线。
- 🔭 开放问题/未来方向:【原文】long-horizon RL for reasoning 的进一步研究(Abstract/Conclusion);把"扩展边界"推到更难/更 OOD 任务。【推断】① 补做组件消融(隔离 KL / 重置 / clip-higher 的边际贡献),把"最巧一步"从叙述坐实为受控证据;② 扩到更大模型与非蒸馏起点,验证"良好初始化才保 KL"前提的普适性;③ 把"随机采样探索"换成"主动/前瞻引导探索"(如 MTP 前瞻或教师脚手架选路),看能否在 base 强领域也突破 pass@128 的 Diminished 困境;④ 寻找比 pass@128 / Creativity Index 更稳健的"边界扩展"测度。

— RETURN —
prorl | 读到PDF? 是(26页/101k字重抽清洗null字节后,§1-§6+三regime/boxnet/Creativity Index全核,所有超参与§1/§3增益不一致点逐项核对) | L线 L3(长程RLVR/稳定化) | 对标结论:支撑"探索可行性"(给足预算+压熵坍缩能发现base走不通的新路径,boxnet base全0→RL满分;且给出"仅base弱领域才扩展、base强领域pass@128反降"边界条件)并提供参考策略重置/KL前提化/pass@128+CreativityIndex评测协议三件可借工具,但无teacher/脚手架/回轨/前瞻,作支撑与工具来源非直接基线 | 残留待核数:2(全文无组件消融,KL/重置/clip-higher边际贡献未隔离;§1 vs §3 增益数值不一致原文无解释、已并列采§3)

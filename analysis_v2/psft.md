psft | Proximal Supervised Fine-Tuning (PSFT) | 上海交大 + 上海创智学院 + 腾讯大模型部 + 澳门大学(Wenhong Zhu, Ruobing Xie, Rui Wang, Xingwu Sun, Di Wang, Pengfei Liu;SFT-RL 统一/改进版 SFT 谱系,与 iw-SFT/DFT/LUFFY/HPT 同族) | ICLR 2026·arXiv·v2(arXiv:2508.17784v2,2026-04-12) | 主题线 L2(统一 SFT-RL 视角下的"改进版 SFT")·相关性 高

**原始论文**:https://arxiv.org/abs/2508.17784

## 一眼看懂
- 🟦 TL;DR:普通 SFT 是 behavior cloning,泛化差(在特定任务上微调后原有通用能力退化),且训练中会**熵坍缩**(熵曲线呈"锯齿":每个 epoch=178 步后骤降一次,见 Figure 1),把后续 RL 的探索空间压没。PSFT 把 SFT 重新看成"**advantage 恒为正(\(\hat A_t=1\))、采样自固定离线数据集**的策略梯度特例",于是可以借 TRPO/PPO 的信任域思想给它套上 PPO 式 clipped surrogate(用 \(r_t(\theta)=\pi_\theta/\pi_{\theta_{\text{old}}}\) 的重要性比 + clip),把每步更新限制在策略附近。代价是 in-domain **基本持平/略低**于普通 SFT(Qwen 均值 46.98 vs 47.99),换来明显更强的 OOD 泛化(61.26 vs 57.90)、不熵坍缩、以及作为后续 RL/DPO 起点更优。【原文 Abstract 行 10-23,§2.1 行 107-109,§3 行 149-198,Table 1 行 326-420】
- 最巧的一步:**把 advantage 设为常数 \(\hat A_t=1\),再套 PPO 的 clip**。这一步等价于"给每个 demo token 的最大似然推力封顶"——梯度分析(Eq.7)显示:当离线数据分布偏离模型分布很多(\(r_t>1+\epsilon\))时,这些 token **梯度被直接置零**(\(I_{\text{trust}}=0\)),从而不被"硬覆写"。抽掉 clip 就退回普通 SFT(熵坍缩 + 泛化垮);抽掉"\(\hat A_t=1\)"它就不再是 SFT 而变成需要真 advantage 的 RL。两者缺一不可——clip 是机制,\(\hat A_t=1\) 是把 SFT 纳入 PG 框架的桥。【原文 §2.1 行 107-109,§3 Eq.6 行 159-167,Eq.7 行 187-194】

## 为什么做
- 研究背景:后训练已成训练流程关键一环。RL(PPO/GRPO)在推理任务上随训练规模扩展能产出高质量长 CoT 轨迹;社区大量用 SFT(一种"蒸馏")把这些轨迹的推理能力注入模型,因其比 RL 简单高效(引 Guha 2025、Li 2025)。【原文 §1 行 26-34】
- 解决的具体痛点:① SFT=behavior cloning,当微调数据**次优**或与**预训练分布失配**时会引发**过大的策略更新**(引 Schulman 2015)、损害原有能力(泛化差,引 Chu 2025 "SFT memorizes, RL generalizes");② SFT 易致**熵坍缩**(引 Cui 2025 entropy mechanism),削弱探索能力——SFT 常作 RL 冷启动,但过度依赖会把探索磨没,约束后续 RL(引 Xie 2024、Yu 2025)。**核心挑战:同时改善 SFT 模型的泛化与探索**(原文加粗句,行 46-47)。【原文 §1 行 35-47】
- 相关工作 & 各自不足(§6 三族,精确到机制):
  - **改 SFT 目标族**:(a) **iw-SFT**(Qin & Springenberg 2025)——把标准 SFT 重释为 RL 目标的**松下界**(loose lower bound),该下界随模型分布漂离参考策略而越来越松;iw-SFT 用**重要性重加权**(给更偏好的轨迹更高权)收紧下界、逼近真 RL 目标;(b) **DFT**(Wu et al. 2025)——把 SFT 看成**有缺陷的策略梯度**(flawed PG),用基于概率的**重加权**改善 target 性能。二者都聚焦"提升 target 性能/收紧界",而 **PSFT 的目标是"防熵坍缩 + 保通用能力 + 给后续优化留空间"**(原文加粗句,行 1046-1048)。
  - **RL 族**:PG(Sutton 2000)不稳/高方差 → TRPO(硬 KL 信任域)→ PPO(clip 软信任域)→ GRPO(去 critic、组内相对奖励、序列内 token 等权)。PSFT="把这些 RL 方法的好处搬到监督设定"(行 1056)。
  - **SFT+RL 混合族**:**LUFFY**(Yan 2025)——每 batch 按**固定比例**混 offline demo + online rollout;**HPT**(Lv 2025)——按 rollout 表现**自适应切换** SFT/RL,平衡指导与探索。PSFT 自陈**与这些方法正交**(orthogonal,行 1064)。另有 **SFT-KL**(在 loss 加 KL 约束、系数 0.5)作直接对照基线(§4.1 行 217-219)。
- 动机链:SFT 伤泛化/垮探索 ∵ 对每个 demo token 施加**无界**最大似然推力 → 借 TRPO/PPO 信任域把这股推力用重要性比裁剪封顶 → 既保留模仿、又把更新限制在策略附近 → **不引入显式 KL/reference** 也能保熵 + 保原有能力 + 给 RL 留空间(对照 SFT-KL:KL 把策略拴在初始 ref 上,反而限制能从离线数据学到的知识)。【原文 §1 行 47-51,§3 行 168-180】
- 与最近邻工作的Δ:
  - vs **普通 SFT**:加 clip 信任域(\(\epsilon=0.28\)),\(\hat A_t=1\)。
  - vs **SFT-KL**:**靠 trust-region(裁剪)而非显式 KL/reference 锚定**——KL 把策略拴在初始 \(\pi_{\theta_{\text{ref}}}\) 上,把优化局限在"以 ref 为中心的信任域",**限制能从离线数据有效利用的知识**(原文 §3 行 177-180 明说);裁剪只在 token 级封顶,而 \(\pi_{\theta_{\text{old}}}\) 还能**动态演化**。
  - vs **iw-SFT/DFT**:它们重加权**所有** token 以收紧/修正 PG;PSFT 用 clip **直接置零**远偏 token 的梯度,且目标定位在"留 RL 空间/防熵坍缩"而非收紧界。
  - 关键有用点:Eq.7 的梯度门控(\(r_t>1+\epsilon\) 置零)直接对应"不被次优 demo 硬覆写",这是泛化与防熵坍缩的来源。

## 怎么做 + 靠不靠谱
- 方法流水线(§3,输入→输出讲清):
  - **输入**:离线长 CoT 数据集 \(D\)(实验用 OpenR1-Math-8192);基座 policy \(\pi_\theta\)(Qwen2.5-7B-Instruct / Llama3.1-8B-Instruct)。**输出**:微调后 policy,熵不坍缩、OOD 泛化更好、可作 RL/DPO 更优起点。
  - **MDP 形式化**(§2):把自回归生成建成 MDP——状态 \(s_t=(x,y_{<t})\)(当前前缀),动作 \(a_t\in A\)(下一 token),policy \(\pi_\theta(y\mid x)=\prod_{t=1}^{n}\pi_\theta(a_t\mid s_t)\)。
  - **核心目标(Eq.6)**:把 SFT 改写为带重要性比裁剪的代理目标——
    \(\displaystyle L_{\text{PSFT}}(\theta)=\mathbb{E}_{(s_t,a_t)\sim D}\!\left[\min\!\left(\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)},\ \operatorname{clip}\!\Big(\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)},\,1-\epsilon,\,1+\epsilon\Big)\right)\right]\)
    其中 advantage 恒正且简化为 \(\hat A_t=1\)(对应"所有离线 token 视为正确"),\((s_t,a_t)\) 采自固定 \(D\)。注意:因 \(\hat A_t=1>0\),min 实际只在**上侧**起作用——这把 SFT 的"无界对数似然上推"封顶到 \(1+\epsilon\)。
  - **clip 参数**:\(\epsilon=0.2\) 或 **0.28**(实验主用 0.28,与 DAPO clip-higher 同思路鼓励探索);**\(\text{use\_kl}=\text{False}\),\(\text{kl\_coef}=0\)**(靠 trust-region,非显式 KL)。"更大 \(\epsilon\) → 更大梯度"(行 202)。
  - **\(\pi_{\theta_{\text{old}}}\) 动态更新(关键)**:让旧策略**动态演化**(每 4/8/16 步刷新一次快照),而非固定为初始 \(\pi_{\theta_{\text{ref}}}\);固定会把优化局限在 ref 中心的信任域、限制可学知识(§3 行 176-180,§5.3 实证)。
  - **可选 warm-up**:先在 \(D\) 上做一段普通 SFT,把 \(\pi_{\theta_{\text{old}}}\) 的分布对齐到 \(D\),缓解初期 \(r_t\) 的偏估,**改善 in-domain**(更长 warm-up → in-domain 更高,Finding 3)。
  - **数据流动**:每步从 \(D\) 取 \((s_t,a_t)\) → 用当前 \(\pi_\theta\) 与快照 \(\pi_{\theta_{\text{old}}}\) 算比值 \(r_t\) → 按 Eq.7 门控梯度(\(r_t>1+\epsilon\) 置零,否则正常 MLE 梯度)→ 更新 \(\theta\);每 4/8/16 步把 \(\pi_{\theta_{\text{old}}}\leftarrow\pi_\theta\)。效果:训练全程不熵坍缩、保持生成多样性。
- 逐组件必要性:
  - **clip 信任域(vs 普通 SFT / SFT-KL)**:有对照(Table 1 + Figure 1 熵曲线 + Figure 8 clip 值消融)。普通 SFT/SFT-KL 熵曲线呈"锯齿"、每个 epoch 后骤降(过拟合,Finding 1);PSFT 熵曲线平滑。Figure 8:**PSFT 不加 clip → 熵更高但梯度范数大而不稳、下游剧烈波动**;加 clip 后稳定;中等 \(\epsilon\)(0.28)在稳定/性能间最优。【行 228-232,行 965-970】
  - **\(\hat A_t=1\)(SFT 纳入 PG 的桥)**:理论推导(§2.1 行 107-109),实现上即"给序列所有 advantage 赋常数"(reproducibility 声明,行 1078-1079)——非实验消融,但与"SFT=\(\hat A\equiv1\) 的 PG"自洽。
  - **\(\pi_{\theta_{\text{old}}}\) 动态更新频率**:有消融(Table 6,§5.3)。**no-upd(固定 ref)= 差**(AIME-24 **13.02**、AIME-25 11.15,仅略超 base 11.25/8.75);**upd.16=21.56 / 23.54**、**upd.8=19.38 / 21.98**、**upd.4=22.50 / 22.92**;vs SFT=22.08/23.02。结论:动态更新是必要的;更频繁(每 4 步,batch 256 + mini-batch 小 + 高 lr)in-domain 增益最大,但**因高波动牺牲泛化**(IFEval upd.4=72.08 < upd.8=73.03)——故主实验取 **upd.8**(in-domain 与 OOD 折中)。【Table 6 行 977-1012,分析行 1013-1028】
  - **warm-up**:有对照(Finding 3,Table 1)。warm-up 让 in-domain 稳步提升、可反超普通 SFT(Qwen in-domain 均值 warm-up **48.17** > SFT **47.99** > PSFT **46.98**);更长 warm-up 更高 in-domain。【行 431-434】
  - **非对称 clip 本身**:取 0.28(clip-higher 思路);high/low 非对称性的**单独**消融未单列,只在 §5.2 给了"不同 \(\epsilon\) 值"的整体消融(Figure 8)〔推断:Eq.6 实际写的是对称 \([1-\epsilon,1+\epsilon]\),"非对称"指实现里 clip_ratio_high=0.28 这一工程取值,论文未做 high≠low 的对照〕。
- 关键机制/公式(直觉):
  - **SFT 即 \(\hat A\equiv1\) 的 PG**(§2.1):标准 SFT 损失 \(L_{\text{SFT}}=-\hat{\mathbb E}_{(s_t,a_t^*)\sim D}[\log\pi_\theta(a_t^*\mid s_t)]\);PG 目标 \(L_{\text{PG}}=\hat{\mathbb E}_{(s_t,a_t)\sim\pi_\theta}[\log\pi_\theta(a_t\mid s_t)\,\hat A_t]\)。令采样自固定 \(D\) 且 \(\hat A_t=1\),PG 即退化为最大似然=SFT。
  - **clip 在此是正则器,不是 off-policy 修正**:原文明确"PSFT 不是 PG-RL"(§3 行 171-175)——SFT 没有 advantage 估计/分布偏移问题,clip 主要当**正则器限制 token 概率剧烈变化 + 重加权梯度**。
  - **梯度门控(Eq.7,核心直觉)**:
    \(\displaystyle \nabla_\theta L_{\text{PSFT}}(\theta)=\mathbb{E}_{(s_t,a_t)\sim D}\big[r_t(\theta)\cdot I_{\text{trust}}(r_t(\theta))\cdot\nabla_\theta\log\pi_\theta(a_t\mid s_t)\big],\quad I_{\text{trust}}(r_t)=\begin{cases}0,& r_t>1+\epsilon\\[2pt]1,&\text{otherwise}\end{cases}\)
    直觉:模型已基本同意的 token(\(r_t\) 适中)正常推;模型**强烈不同意**的 token(\(r_t>1+\epsilon\),即离线分布远偏于模型)**梯度置零**,避免"次优 demo 把原能力硬覆写"——这正是保泛化 + 防熵坍缩的来源。
  - **被裁剪的是什么 token(§5.1 的精彩观察)**:Figure 7 显示被 clip 的 token 集中在 **"wait"、"alternatively"** 等不确定性词——即"长思考模式(long thinking pattern)"标志词;随训练推进这些 token 的 clip 权重更显著、其它 token 变小。解读:PSFT 把"思考模式"**平滑、渐进**地注入模型,而对通用能力扰动最小。〔这是"clip 到底在管什么"的直接证据,比抽象 Eq.7 更可感〕。【原文 §5.1 行 953-962】
- 实验与证据(所有数字均已对 Table 1/2/6 逐项核对):
  - 数据集/设置:SFT 阶段训练用 **OpenR1-Math-8192** long-CoT(§4.1.1);RL 阶段用 **DAPO-MATH-17k**(DAPO,clip-higher 0.28,称"GPPO 的稳定变体",§4.2 行 622-623)。基座 **Qwen2.5-7B-Instruct、Llama3.1-8B-Instruct**。in-domain=AIME-24/25、AMC(avg@32)、MATH-500、OlympiadBench、Minerva(avg@8);OOD=GPQA、ARC-C、TruthfulQA、IFEval(avg@8)、MMLU-Pro、SuperGPQA、HeadQA(pass@1)。checkpoint 选取:Qwen 上 SFT@700、SFT-KL@900、PSFT@1300 步(按 in-domain 选,行 442-443)。推理长度 10,240(IFEval 4,096),top-p 0.95,温度 0.7。另覆盖 human-value alignment(Qwen3-4B-Base + UltraFeedback → DPO)与多模态(Qwen2.5-VL,文本长 CoT 注入 → Geometry-3K 上 GRPO)。【§4.1-4.4 行 212-219,行 442-445,行 656-670,行 914-921】
  - **关键数字 — SFT 阶段(Table 1,Qwen2.5-7B-Instruct;Base/SFT/SFT-KL/PSFT-warmup/PSFT)**:
    - in-domain AIME-24:11.25 / **22.08** / 19.27 / 22.92 / **19.38**;AIME-25:8.75 / 23.02 / 21.56 / 23.02 / 21.98;in-domain 6 项**均值**:37.98 / **47.99** / 47.08 / **48.17** / **46.98**(行 326-336)。→ PSFT in-domain 略低 SFT、warm-up 反超。
    - OOD **IFEval**:base 73.94 / SFT **54.42** / SFT-KL 55.44 / warmup 55.07 / PSFT **73.03**(行 404-414)——**SFT/SFT-KL/warmup 都把指令遵循能力打掉 ~20 点,唯独 vanilla PSFT 基本保住 base**;TruthfulQA:66.10 / 63.14 / 61.31 / 66.37 / **67.16**;OOD 7 项**均值**:59.85 / **57.90** / 57.38 / 58.53 / **61.26**(行 416-420)。
    - Llama3.1-8B 同向:OOD 均值 base 52.70 / SFT **50.49** / PSFT **59.25**;IFEval base 78.10 / SFT 33.45 / PSFT 69.75(行 410-425)。
  - **关键数字 — RL 阶段(Table 2,作为 RL 起点;SFT / SFT→GRPO / PSFT / PSFT→GRPO)**:Qwen in-domain 均值 47.99 / **52.40** / 46.98 / **53.31**;Qwen OOD 均值 57.90 / **59.90** / 61.26 / **64.06**;IFEval SFT→GRPO **53.89**(几乎没救回)vs PSFT→GRPO **73.73**(行 534-615)。Llama OOD 均值 SFT→GRPO 53.19 vs PSFT→GRPO **61.96**。→ PSFT 初始 in-domain 落后,但**经 RL 后 in-domain 与 OOD 双双反超 SFT→GRPO**(高熵留下的探索空间被 RL 兑现,Finding 2 "slow start, rapid catch-up")。
  - 普适性补充:**对齐域(Table 3)**——Qwen3-4B-Base + UltraFeedback,SFT→DPO vs PSFT→DPO:AlpacaEval2 LC SFT→DPO 16.96 vs PSFT→DPO **23.29**、Arena-Hard 26.50 vs **36.40**(PSFT 显著降 alignment tax,Figure 5);**多模态(Table 5)**——Qwen2.5-VL 文本长 CoT 注入后 SFT 在 MMMU-Pro 掉到 23.58(GRPO 后更掉到 19.60),PSFT 保住 27.63(GRPO 后 26.07)。【Table 3 行 678-724,Table 5 行 865-907】
  - baseline 公平性:**强且诚实**——SFT、SFT-KL(KL 0.5)、PSFT、PSFT-warmup 同基座同数据;**论文如实报告 in-domain 略逊普通 SFT**,不做选择性汇报。
  - 看着强但没回答的:① in-domain 略降是真实代价(纯追 in-domain 峰值者不利);② "留 RL 空间"优势(Table 2)虽明显,但 RL 阶段用 DAPO+DAPO-MATH-17k,与 SFT 阶段数据不同,起点-终点归因略复杂;③ \(\hat A_t=1\) 假设依赖"所有 demo token 正确",含噪/含错 demo 下不成立(论文未给鲁棒变体)。
- 假设与失效边界:【原文】① 离线数据所有 token 视为"correct"(\(\hat A_t>0\),简化为 1)——含错误/噪声 demo 时假设不再合理(§3 行 157-158);② \(\epsilon=0.2/0.28\),更大 \(\epsilon\)→大梯度(行 202);③ \(\pi_{\theta_{\text{old}}}\) 需动态更新(固定差,§5.3);④ warm-up 可选,改善 in-domain。【推断】⑤ in-domain 拟合略逊普通 SFT,不适合"只要 in-domain 峰值"场景;⑥ 高频更新(每 4 步)提 in-domain 但伤泛化(高波动),最优频率任务相关、经验取 8;⑦ "信任域=保泛化"的因果在数学/对齐/多模态三域 + Qwen/Llama 双族验证,但**更工业级/更大模型未试**(Conclusion 行 1071-1072 自陈)。
- 祛魅总结:【推断】真贡献=① 一个干净的理论联系(**SFT=\(\hat A\equiv1\) 的 PG**)+ ② 据此把 PPO clip 搬进 SFT 这一**极简改动**(reproducibility 声明:主流 RL 框架"rollout 换 demo + advantage 赋常数"即可),并**诚实报告 in-domain 略逊、OOD 全面提升 + 不熵坍缩 + RL 起点更优**的完整权衡。被高估处:与 iw-SFT/DFT 同属"SFT-as-PG 重加权/约束"一族,**理论新意有限**(\(\hat A=1\) 视角 DFT/Prefix-RFT 都用过);PSFT 的差异主要在"用 clip **置零**远偏 token 而非重加权"+"目标定位防熵坍缩/留 RL 空间"。被低估处:IFEval 54.42→73.03 这种"防 SFT 把指令遵循能力打没"的实用价值很大;Figure 7(被 clip 的恰是 "wait/alternatively" 这类思考模式词)给"PSFT 在管什么"提供了少见的可解释证据;Table 2 的"RL 起点更优"对 SFT→RL 流水线有直接工程意义。

## 结构化抽取
- 🎯 机制速览6轴:
  - **学什么信号**:离线 demonstration 的 ground-truth token(交叉熵/最大似然),但 advantage 恒=1、且 token 级梯度被信任域裁剪门控(\(r_t>1+\epsilon\) 置零);无 teacher logits、无奖励。
  - **改什么**:policy 参数 \(\theta\)(裁剪后的最大似然梯度,Eq.7);\(\pi_{\theta_{\text{old}}}\) 动态滚动更新(copy 快照,非梯度)。
  - **何时改**:离线 SFT 阶段(非在线 RL);\(\pi_{\theta_{\text{old}}}\) 每 4/8/16 步刷新;可选 warm-up 先对齐 \(\pi_{\theta_{\text{old}}}\)。
  - **免梯度?**:否,\(\theta\) 梯度更新(但远偏 token \(r_t>1+\epsilon\) 梯度置零)。
  - **记忆-技能生命周期**:无外部记忆/技能库;离线数据 \(D\) 是静态知识源;"技能"通过受约束的模仿固化进参数,刻意**留出后续 RL 优化空间**(不把熵学没)。
  - **防遗忘机制**:**核心卖点即"防 SFT 遗忘原有能力"**——靠 trust-region 裁剪(Eq.7 置零远偏 token 梯度)限制策略漂移、防熵坍缩,从而保住预训练/通用能力(OOD 全面提升、IFEval 不被打垮即证据)。非跨任务持续学习式防遗忘,而是"单次 SFT 不灾难性覆写原分布"。
- ⑦ 开源代码+框架/harness:仓库 https://github.com/zwhong714/PSFT(本地已克隆 **23MB**,完整)。框架=**veRL**,实现在 `verl/recipe/psft/`(main_psft.py、psft_ray_trainer.py、config/psft_trainer.yaml、run_psft.sh / run_sft.sh);adv_estimator=psft 内联于 ray_trainer.py(adv=ones_like·mask)。模型权重在 HF(`wh-zhu/psft-*`)。reproducibility 声明(行 1076-1081):主流 RL 框架最小改动即可(rollout 换 demo + 给序列所有 advantage 赋常数),代码 built upon verl,随 supplementary 提供。【已核本地仓 verl/recipe/psft 文件树 + du 23M;v1 已逐行核 adv 内联分支】
- 💰 资源/成本与可扩展性:SFT 阶段成本与普通 SFT 同量级(只是 loss 加裁剪 + 维护 \(\pi_{\theta_{\text{old}}}\) 快照);update.16=batch 256 / mini-batch 16、update.8=mini-batch 32、lr 1e-6(§5.3 行 1025-1026)。绝对 GPU 卡时主文未给(见 Appendix C)〔原文未在主文给绝对算力〕。可扩展性:数学/对齐/多模态三域 + Qwen/Llama 双族验证,工业级/更大模型留作 future work。
- 🎯 对"探索-巩固"对标:**强支撑"巩固"一侧 + 保探索的前置条件**。PSFT 解决的正是 idea 里"巩固=固化进参数**且不遗忘**"的核心痛点——它给出"如何在 SFT 阶段把(教师)demonstration 固化进 student 而不灾难性覆写原能力/不熵坍缩"的极简答案,这对 TSRD 的"巩固/回轨固化"阶段直接可用。与"探索"的关系是**前置使能**:不熵坍缩=给后续 on-policy 探索留空间(Table 2 "slow start, rapid catch-up" + RL 起点更优,行 632-639)。可借组件:① **\(\hat A_t=1\) + PPO clip** 这一"受约束模仿"目标可直接替换 TSRD 巩固阶段的普通 SFT,防止教师脚手架把 student 原有可走通路径覆写掉;② Eq.7 的**梯度门控**(远偏 token 置零)与本项目"forward-hard/backward-soft 解耦"哲学相通(都用"有界/受限梯度"避免破坏性更新),可类比"巩固时只固化与现有策略不冲突的部分";③ "防熵坍缩=保探索空间"的熵曲线可作 TSRD 巩固阶段的健康度监控;④ Figure 7 提示——被裁剪的恰是"wait/alternatively"这类**思考模式词**,暗示"巩固阶段应温和注入'思考结构'而非硬背 token",与 path-recovery 想注入"回轨步骤"而非整段覆写同构。缺口:PSFT 是**纯离线模仿**,无 on-policy 自采、无 teacher 软标签(只有 ground-truth token)、无 path-selection/recovery、无前瞻——它只管"模仿得温和不伤身",不管"探索/选路"。一句判定:**"巩固不遗忘"这一半的强可借基线**(受约束模仿 + 梯度门控 + 防熵坍缩保探索空间),与本项目 forward-hard/backward-soft 哲学同构,但不覆盖"探索/选路/前瞻"另一半。
- 🔭 开放问题/未来方向:【原文】在更多样、工业级数据集与模型上验证 PSFT 的普适性(Conclusion 行 1071-1072)。【推断】① 把 \(\hat A_t=1\) 放松为"教师/前瞻给的软 advantage",使巩固阶段也能区分 demo token 的重要性(向 on-policy 蒸馏过渡);② 把"远偏 token 置零"与"高熵/低置信关键步切分"结合,做选择性巩固(只固化关键步,呼应 Figure 7 的思考模式词观察);③ 噪声/含错 demo 下 \(\hat A_t=1\) 假设失效时的鲁棒变体;④ \(\pi_{\theta_{\text{old}}}\) 更新频率的自适应(in-domain vs OOD 权衡)而非固定 8;⑤ 与 MTP 前瞻信号结合,把"信任域裁剪"扩展到"前瞻一致的 token 才放行"。

— RETURN —
psft | 读到PDF? 是(23页/66k字,§1-§7+§5.1-5.3消融+Table 1/2/3/5/6 数字逐项重核,确认无回退) | L线 L2(SFT-RL统一/改进版SFT) | 对标结论:"巩固不遗忘"这一半的强可借基线——A=1+PPO clip受约束模仿+Eq.7梯度门控(远偏token置零)与本项目forward-hard/backward-soft哲学同构,防熵坍缩=保探索空间作巩固健康度指标,Figure 7"被裁剪=思考模式词"提示温和注入思考结构;但纯离线模仿、无on-policy/软标签/选路/前瞻,不覆盖"探索"另一半 | 残留待核数:0(Table 1/2/6 全部数字逐项核对一致;此前抽检的编造数字未复现,确认真实;非对称clip的high≠low单独消融原文确无、已标注为推断;绝对GPU卡时主文未给见Appendix C)

rock_tokens | Cornerstones or Stumbling Blocks? Deciphering the Rock Tokens in On-Policy Distillation | UMBC + Case Western + Arizona State + VU Amsterdam(Yuxuan Jiang/Runchao Li/Shubhashis Roy Dipta 共同一作;通讯 Zhao Yang) | 2026-05-29 v2 · arXiv preprint(cs.CL) | L1 OPD/自蒸馏 · 相关性中

**原始论文**:https://arxiv.org/abs/2605.09253

## 一眼看懂
- 🟦 TL;DR:RLVR 已知"少数关键 token 撑起推理增益",但 OPD 的 token 级动力学没人研究。作者发现 OPD 训练表观饱和后,仍有约 6% 词表、占输出 18% 频次的 token 持续高 KL loss(命名 **Rock Tokens**,多为格式符/空白/话语标记如 "So"/"Wait")。两大悖论:它们贡献了不成比例的梯度范数却在训练中"原地不动"抵抗 teacher 修正;且因果敲除(knock-out)显示绝大多数对推理准确率无关键贡献。结论:OPD 的均匀 token 加权浪费了大量优化带宽在学生"学不会也不必学"的结构残差上,从训练起冻结这些 token 的梯度可在几乎不掉点下提速(文中正文 1.7×、take-away 1.4×)。【原文】Abstract、§1、§4.3
- 最巧的一步:**inference-time token knock-out**(把某 token 的 logit 在每步解码强制设 \(-\infty\),看准确率变化 \(\Delta\))——这是把"梯度成本高"和"功能贡献"解耦的因果探针。抽掉它,全文就只剩"高 loss token 很顽固"的描述性观察,无法回答标题之问(是基石还是绊脚石),也无法证明冻结它们安全。【原文】§3.2 Eq.17

## 为什么做
- 研究背景:OPD(On-Policy Distillation)已是现代 LLM 后训练基石(DeepSeek-V4、MiMo、Qwen-3 在 SFT/RLVR 之外用它进一步榨推理),本质依赖 dense 全 token KL 监督;self-distillation 研究暗示 OPD 是放大潜能的通用机制而非单纯模仿。【原文】§1、§6.1
- 解决的具体痛点:OPD 的 per-token KL loss 中 high-loss token 是师生失配最直接信号,按既有理解应随收敛而减少;但实证相反——一批 token 持续高 loss(Rock Tokens)。由此两悖论:(1) 高频导致贡献巨大梯度范数却自身停滞;(2) 因果上对推理性能贡献可忽略。大量优化带宽花在结构/话语残差上。【原文】Abstract、§1
- 相关工作的来龙去脉与精确短板(据 §6):
  - **OPD for 后训练(§6.1)**:OPD 超越 SFT/RLVR 之处=轨迹探索 + dense token 级指导(Lu&Lab 2025);DeepSeek-V4/MiMo/Qwen-3 作后训练阶段证其效。**已有对 OPD 失败的研究多看全局因子**——词表、推理风格、teacher 选择、模型族(即 rethink_opd / why_sd_degrade 那一类)。**本文短板填补**=聚焦**细粒度**停滞源(Rock Tokens),判定 OPD 监督何时有用 vs 无用噪声。
  - **token 级学习动力学(§6.2,直接对标)**:推理被少数 "anchor" token 不成比例支撑(扰动它们推理时显著掉点);RLVR 研究揭示训练非均匀——集中在高熵 "forking token"(决定性分叉点,Wang et al.[70/71]),且 token 分布漂移主要发生在稀疏但关键 token 上。**精确差异**:RLVR 的 token 重要性从 **entropy/reward** 推断,本文用 OPD 独有的 **per-token reverse-KL** 直接度量师生失配定义 token 类型,并发现"**高 KL ≠ 高功能价值**",方向与 RLVR"高熵=高价值"相反;且本文因果证明 Pillarhood 与 entropy/频率/KL **正交**(|r|<0.07)——警告按 loss/熵加权的重要性方案有压制 Pillar 的风险。
  - (§6.3 大段是无关领域的引用堆砌,与本工作弱相关。)【原文】§6.1-6.3
- 动机链:OPD 靠 dense KL → 直觉认为高 loss token = 关键修正点应收敛 → 实测它们顽固不动且高频 → 它们到底是 Pillar 还是 Stumbling Block?→ 用 What/Why/How 三阶段诊断并据此挑战均匀加权。【原文】§1
- 与最近邻工作的Δ:见上 §6.2;核心一句——**用 per-token KL 定义 OPD 的 token 类型,且高 KL 与因果重要性正交**,这是 RLVR token 研究没覆盖的角落。

## 怎么做(到"读完能复现"的粒度)

### A. OPD 的 token 级 loss(基底)
OPD 在学生自采轨迹上最小化逐 token reverse-KL:
\[\mathcal L_{\text{OPD}}(\theta)=\mathbb E_{x_{1:T}\sim\pi_\theta}\Big[\textstyle\sum_{t=1}^{T}D_{\mathrm{KL}}\big(\pi_\theta(\cdot\mid x_{<t})\,\Vert\,\pi_T(\cdot\mid x_{<t})\big)\Big].\]
位置 \(t\) 的 token 级蒸馏信号 \(\ell_t=\log\pi_\theta(x_t\mid x_{<t})-\log\pi_T(x_t\mid x_{<t})\)。理想下 \(\ell_t\) 随对齐系统性减小至 plateau;但长推理轨迹里某些 token occurrence 即便总目标饱和仍持续大失配——且这种持续高 loss 可能源自 **token 身份本身** 或 **局部上下文**(对空白/换行/标点/数字这类高频 token 尤其需区分)。【§2.1 式1/2】

### B. 识别 Rock Tokens:两级打分(§2.2,v1 漏掉的 context-aware 部分在此补全)
- **一级:聚合 Rock Score(候选筛选)**。OPD loss 按 token type \(v\) 分解 \(\mathcal L_{\text{OPD}}=\sum_v \mathrm{Freq}(v)\cdot\mathbb E[\ell_t\mid x_t=v]\),定义初始
\[R(v)=\bar\ell_v\cdot\mathrm{Freq}(v),\]
\(\bar\ell_v\) = 末 checkpoint 上 token \(v\) 的经验平均 token 级 loss。频率项**抑制小样本噪声**(稀有 token 即便观测 loss 极端也限权,高频且持续超基线 loss 才高分)。但 \(R(v)\) 单独不够(高频 token 可能只少数 occurrence 难),故加二级。
- **二级:occurrence 级 + 上下文感知过滤**。把"持续高 loss occurrence"定义为训练前后都高 loss:\(O_{\text{PH}}=\{(i,t):\ell^{\text{pre}}_{i,t}\ge\tau_{\text{pre}},\ \ell^{\text{post}}_{i,t}\ge\tau_{\text{post}}\}\)。给每个 occurrence 取局部上下文窗 \(c^{(w)}_{i,t}=x^{(i)}_{t-w:t+w}\),编码 \(h^{(w)}_{i,t}=f_{\text{ctx}}(c^{(w)}_{i,t})\),两 occurrence 上下文相似度 \(s(o,o')=\mathrm{sim}(h^{(w)}_{i,t},h^{(w)}_{j,k})\)。**仅在同 token type 内**比较(\(O_{\text{PH}}(v)=\{o\in O_{\text{PH}}:x_o=v\}\)),定义上下文一致性分
\[\rho(o)=\frac{1}{|O_{\text{PH}}(v)|-1}\sum_{o'\in O_{\text{PH}}(v),\,o'\ne o}\mathbb 1[s(o,o')\ge\gamma],\]
保留 \(\rho(o)\ge\eta\) 的 occurrence 得 rock occurrence 集 \(\mathcal R(v)\),其上下文一致 rock 率 \(\mathrm{CCR}(v)=|\mathcal R(v)|/\mathrm{Freq}(v)\)。**最终上下文感知 Rock Score**:
\[R_{\text{ctx}}(v)=R(v)\cdot\mathrm{CCR}(v),\qquad v\in V_{\text{rock}}\iff R_{\text{ctx}}(v)\ge\tau_R.\]
**作用**:防"高频但仅孤立位置难"的 token(如个别空白/数字)被误判为全局顽固——它们 CCR 低、最终分被压下;只有"跨相似上下文反复高 loss"的才留为真 Rock Token。【§2.2 式3-14】
- **经验估计**:用学生 rollout 估 per-token KL \(\hat b_{\ell v}=\hat{\mathbb E}[D_{\mathrm{KL}}(\pi_\theta\Vert\pi_T)\mid x_t=v]\)(式15)。在 N=500 MATH-500 轨迹上画 频率-KL 平面:稀有 token 噪声主导,Rock Score 把真 Rock Token(红)隔在稳定频率带上沿。【§2.3 式15,图2a】
- **cutoff \(K\)**(§2.4):扫 Top-K——小 K 稳但覆盖低、大 K 覆盖高但抗噪差;**最优交点 K=100**(覆盖约 60% 语料级 KL,Jaccard 在 \(n\in[50,400]\) 稳健)。Rock Tokens 约占 6% 词表、median 18.5% 输出频次。decode 出来是四类:LaTeX/数学定界符、Markdown/空白结构、话语标记("So"/"Wait")、数字——**学生抵抗的是"怎么组织推理结构"而非"生成什么内容"**(频率匹配对照 \(S_{\text{ctrl}}\) 多是内容词)。【§2.4-2.5,图2c/d】

### C. RQ1:Rock Tokens 还提供有用学习信号吗?——Gradient Paradox(§2.6)
对每个 token type 算均值 logit 梯度 \(\bar g_t=\frac1{n_t}\sum_i g_i\),分解其对全局下降的贡献:
\[\mathrm{contrib}(t)=n_t\cdot\Vert\bar g_t\Vert\cdot\cos(\bar g_t,\ G_{\text{balanced}}),\qquad G_{\text{balanced}}=\textstyle\sum_t\bar g_t,\]
其中 \(G_{\text{balanced}}\) 是**频率均衡参考**(每 token type 等权,故意去掉常见 token 的频率主导)。三个发现:
- **低梯度幅值 vs 高频(图3a)**:Rock 的每次梯度幅值远小于稀有高 KL token(median \(\Vert\bar g_t\Vert\approx0.016\) vs \(0.54\),Mann-Whitney \(p<10^{-30}\)),但 Eq.16 的乘法分解显示其主导来自**频率因子 \(n_t\)**——小信号 ×海量出现=对权更新的主导聚合力。
- **与全局优化方向对齐(图3b)**:**反直觉**——Rock 的梯度方向与 \(G_{\text{balanced}}\) **正对齐**(median cos \(\approx0.040\) > 高 KL \(0.025\) > random \(0.006\),尾部 cos>0.3)。即每个 rock 在很多位置贡献小而方向一致的信号;从 teacher 视角,Rock Tokens 确实指向"正确"的优化方向。
- **训练中停滞(图3c/d)**:比早晚 checkpoint 的 per-token KL——稀有高 KL token 显著降(median 2.02→0.85,~58%,Wilcoxon \(p<10^{-8}\));**Rock 几乎不变(0.21→0.19),\(\Delta\)KL 集中在 0**;random 对称噪声无净变。几何解释:稀有高 KL token 梯度方向独特,聚焦更新可分离解决;Rock 因梯度高度对齐全局,**对它的定向削减与跟随全局下降不可区分**,故得不到 token 专属学习信号去克服自身 loss。
- **假设(作者明说未验证)**:重尾类不平衡下 **Adam 抑制**——预训练时 Adam 压抑高频 token 防过拟合是好事,但 OPD 恰需更新这些高频结构 token 去匹配 teacher 风格;Adam 膨胀 Rock 的二阶矩、系统性压低其有效学习率,把学生困在既有习惯。【§2.6 式16,图3,Adam 假说留 future work】

### D. RQ2/RQ3:因果功能价值 + 选择性蒸馏(§3-§4)
- **inference-time knockout(§3.2,RQ2 因果探针)**:构造 knockout 策略 \(\pi^{\backslash v}_\theta\)(解码每步把 token \(v\) 的 logit 设 \(-\infty\)),量 causal delta
\[\Delta_B(v)=\mathrm{Acc}_B(\pi^{\backslash v}_\theta)-\mathrm{Acc}_B(\pi_\theta).\]
\(\Delta_B(v)\le-0.01\) 判为 Strong Pillar。候选池 \(|\tilde{\mathcal R}|=200\)(超出核心 K=100 以搜稀有因果效应),paired-bootstrap \(\alpha=0.05\)、10000 resample。结果(图4):**MATH-500: 7 Pillar / 0 Stumbling / 193 Neutral;IFEval: 3 Pillar / 0 Stumbling / 197 Neutral**。三论断:① Pillar 罕见但致命(仅 3.5% / 1.5%);② **方向不对称——Strong Stumbling 为 0**(学生顽固偏离 teacher 极少有害,要么是必要锚点要么是无害风格偏好);③ **Pillarhood 与传统指标正交**(|r|<0.07 vs entropy/log-freq/residual KL)→ 按 loss/熵加权的重要性方案有压制 Pillar 的风险。【§3.2-3.3 式17,图4】
- **window-aware 选择性蒸馏(§4.2,RQ3)**:因 Rock 是经局部高散度窗识别(非孤立 token 尖峰),故重加权"Rock token ∪ 其局部窗"。设 Persistence Set \(\mathcal R\)、每候选 token 的局部高散度窗 \(W(v)\),并集 \(W_{\mathcal R}=\bigcup_{v\in\mathcal R}W(v)\)(式18)。加权 OPD:
\[\mathcal L_{\text{weighted}}=\mathbb E_{x\sim\pi_\theta}\sum_{t=1}^{T}w(x_t,t)\cdot\ell_t,\qquad w(x_t,t)=\begin{cases}\lambda,&x_t\in\mathcal R\ \text{或}\ t\in W_{\mathcal R}\\1,&\text{otherwise}\end{cases}\]
三 regime:**Baseline OPD(\(\lambda=1\))** / **Rock-Freeze(ours,\(\lambda=0\),即 "Just Not Train")** / **Freq-Matched Window Freeze(Random,对频率/窗长匹配的随机窗 \(W_S\) 同样 \(\lambda=0\))**。Random 对照排除"冻任意 token 都无害"的平凡解。逻辑:冻 Rock 窗若维持/提升性能 → 它们更像 Stumbling Block;若显著掉点 → 更像 Pillar/pillar-like。【§4.2 式18/19】
- **结果(§4.3,图5)**:① Rock Tokens **不是纯噪声陷阱**——含它们提供的梯度锚定优化(Original OPD vs Ours 有性能差,全冻会掉点);② **Random 严重掉点而 Rock-Freeze 维持高准确率上限**(选择性降权远比无差别减信号安全);③ Rock 约占 18% 输出 token,降其优化压力得 **1.7× wall-clock**(take-away 处又称对 30% 高成本 token freeze-weighting 得 **1.4×**,口径不一宜保守)。【§4.3,图5】

### 实验设置(复现锚点,§5)
- **两阶段蒸馏(KDFlow 框架)**:teacher = Qwen3-30B-A3B-Instruct-2507(MoE,3B active),student = Qwen3-4B-Instruct-2507,**均关 thinking**。Stage-1(off-policy):在 OpenThoughts3 的 **20k** teacher 解上训 1 epoch,forward KL(kd_ratio=0.5)+ 交叉熵,做全词表初始对齐。Stage-2(on-policy):从 Stage-1 checkpoint 起,在另 **10k** prompt 上每 prompt 采 4 rollout(共 7 checkpoint),用纯 reverse KL(kd_ratio=1.0)训 1 epoch;其余超参用 KDFlow 默认(附录C)。
- **硬件**:4×H100(80GB),FSDP2 + bf16 + gradient checkpointing。
- **评测**:LM-Eval-Harness zero-shot Pass@1(5 次平均)。主指标=AIME24 + AIME25 + HMMT25-Feb(各 30 题,合 90 题);MATH-500 + IFEval 用于大样本 token 级统计。图5 横轴每 200 步=8000 prompt × 4 rollout。【§5.1-5.2】

## 靠不靠谱
- 假设与失效边界:
  - 【原文】仅单一师生对(30B-A3B→4B)、关 thinking 模式;Adam 抑制为**假设**,作者明说留待未来形式化验证(§2.6)。
  - 【推断】"功能贡献可忽略"基于数学/IFEval 准确率;对强依赖格式/话语连贯的长文档/对话任务未必成立。knockout(推理时屏蔽)与 freeze(训练时不更新)是两种不同干预,二者结论的可迁移性未严格论证。
  - 【推断】Rock 集合基于最终 checkpoint + MATH-500 轨迹,跨 tokenizer / 更大跨度师生的迁移性未验证。
- 祛魅总结:
  - 真贡献:首个 OPD token 级动力学的系统诊断;提出两级(含 context-aware)Rock Score 与 knockout 因果探针;揭示"Gradient Paradox"与"Pillar 极少但致命、Stumbling 为零"的非对称结构;给出"非均匀梯度分配"的可落地提速。【推断】
  - 比 v1 旧分析更准的关键纠偏:**并非"Rock Tokens 全无用"**——RQ1 表明它们提供方向正确的梯度(全冻会掉点),knockout 表明 1.5-3.5% 是不可或缺 Pillar 且**与 entropy/频率/KL 正交**(|r|<0.07),作者据此警告"按 loss/熵加权的重要性方案有压制 Pillar 的风险"。所以贡献是"**选择性、缓和**地降权高成本 token 求提速",而非"砍掉它们"。【原文】§2.6、§3.3
  - 高估:加速幅度文中两处口径不一(1.7× vs 1.4×、18% vs 30% token),宜保守。【推断】

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号:per-token reverse-KL(师生失配)+ knockout 因果 Δaccuracy。
  - 改什么:OPD 的 token 级梯度加权(对 Rock token 及其窗口赋权 λ)。
  - 何时改:从训练**一开始**就冻(非后期),且基于离线识别的固定 Rock 集。
  - 免梯度?否,是梯度蒸馏(reverse-KL OPD);"freeze"指对子集梯度置零以省算力。
  - 记忆-技能生命周期:不涉及外部记忆/技能库,纯参数内蒸馏。
  - 防遗忘机制:无显式防遗忘;但"只降权冗余 token、保留 Pillar"间接保护学生既有推理框架(path dependency 解释)。
- ⑦ 开源代码+框架/harness:https://github.com/YuxuanJiang1/Rock-Token(v1 已克隆 ~7.4MB,三目录 KDFlow_localopd/rock_detection/stumbling)。框架=自定义 **KDFlow**(songmzhang/KDFlow),后端 SGLang+Ray+FSDP2+bf16;README 致谢显示模型/分布式抽象沿用 **OpenRLHF**、on-policy KD 的 Ray placement-group 与 SGLang 权重更新借鉴 **slime**,rollout 用 SGLang;评测 LM-Eval-Harness。即 OpenRLHF(训练抽象)+slime(on-policy KD)+SGLang(rollout)。【原文】§5.1 + 仓库(v1 核查)
- 💰 资源/成本与可扩展性:硬件 4×H100(80GB),FSDP2+bf16+gradient checkpointing,超参沿用 KDFlow 默认(Appendix C);收益=对高成本 token 冻梯度换 1.4-1.7× wall-clock 提速。【原文】§5.1、§4.3
- 🎯 对"探索-巩固"对标:**可借组件**为主,弱竞品。判定:它的"识别学生顽固抵抗的脚手架 token + 区分 Pillar/Neutral"与本项目"path-token 选择性监督"同源——给"哪些 token 该被 teacher 监督、哪些是学生自有可保留的脚手架"提供了**因果级判据(knockout)**,可直接迁移到 TSRD 的"巩固/回轨 token 选择";但它做的是**降权/冻结**(减信号),与"探索=偏向自己走得通的开头"是反向操作,且无 path-recovery 概念,故不构成竞品。**关键警示**:§3.3 Pillar 与传统指标正交(|r|<0.07)→ 提示本项目**不能简单按高熵/高 KL 选监督 token**(会误伤 Pillar);这与 rethink_opd"overlap token 是作用点"、rho1"excess-loss 选 token"形成有益张力——三者都在追问"哪些 token 该被监督",但 rock_tokens 给出的是"高 KL≠该监督"的反向警告。依据:§3.3。【推断】
- 🔭 开放问题/未来方向:【原文】Adam-suppression 假设的形式化验证(§2.6);Pillar 的可预测性(目前与所有传统指标正交,§3.3)。【推断】跨任务/跨 tokenizer 的 Rock 集迁移;thinking 模式与更大跨度师生下是否成立;把 knockout 的因果 Pillar 信号反哺到 token 级加权方案(替代按 loss/熵加权)。

RETURN: rock_tokens|读PDF?是(_txt 850+行,核到§2-§6全文+式1-19含v1漏掉的context-aware式5-14)|加厚?是(补全两级Rock Score+window-aware式18/19+两阶段KDFlow配置,相关工作§6.2精确化,19条公式转MathJax)|LaTeX公式条数 11|待核数 1(跨任务/跨tokenizer Rock集迁移性)

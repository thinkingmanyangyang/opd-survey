rock_tokens | Cornerstones or Stumbling Blocks? Deciphering the Rock Tokens in On-Policy Distillation | UMBC + Case Western + Arizona State + VU Amsterdam(Yuxuan Jiang/Runchao Li/Shubhashis Roy Dipta 共同一作;通讯 Zhao Yang) | 2026-05-29 v2 · arXiv preprint(cs.CL) | L1 OPD/自蒸馏 · 相关性中

**原始论文**:https://arxiv.org/abs/2605.09253

## 一眼看懂
- 🟦 TL;DR:RLVR 已知"少数关键 token 撑起推理增益",但 OPD 的 token 级动力学没人研究。作者发现 OPD 训练表观饱和后,仍有约 6% 词表、占输出 18% 频次的 token 持续高 KL loss(命名 **Rock Tokens**,多为格式符/空白/话语标记如 "So"/"Wait")。两大悖论:它们贡献了不成比例的梯度范数却在训练中"原地不动"抵抗 teacher 修正;且因果敲除(knock-out)显示绝大多数对推理准确率无关键贡献。结论:OPD 的均匀 token 加权浪费了大量优化带宽在学生"学不会也不必学"的结构残差上,从训练起冻结这些 token 的梯度可在几乎不掉点下提速(文中正文 1.7×、take-away 1.4×)。【原文】Abstract、§1、§4.3
- 最巧的一步:**inference-time token knock-out**(把某 token 的 logit 在每步解码强制设 −∞,看准确率变化 Δ)——这是把"梯度成本高"和"功能贡献"解耦的因果探针。抽掉它,全文就只剩"高 loss token 很顽固"的描述性观察,无法回答标题之问(是基石还是绊脚石),也无法证明冻结它们安全。【原文】§3.2 Eq.17

## 为什么做
- 研究背景:OPD(On-Policy Distillation)已是现代 LLM 后训练基石(DeepSeek-V4、MiMo、Qwen-3 在 SFT/RLVR 之外用它进一步榨推理),本质依赖 dense 全 token KL 监督。【原文】§1
- 解决的具体痛点:OPD 的 per-token KL loss 中 high-loss token 是师生失配最直接信号,按既有理解应随收敛而减少;但实证相反——一批 token 持续高 loss(Rock Tokens)。由此两悖论:(1) 高频导致贡献巨大梯度范数却自身停滞;(2) 因果上对推理性能贡献可忽略。大量优化带宽花在结构/话语残差上。【原文】Abstract、§1
- 相关工作 & 各自不足:RLVR 侧已知 token 非等价(高熵 forking token 驱动增益,[71] Wang et al.),但 OPD 的 token 级理解几乎空白;现有"high loss=重要修正点"的直觉([63])未经 OPD 验证。【原文】§1
- 动机链:OPD 靠 dense KL → 直觉认为高 loss token = 关键修正点应收敛 → 实测它们顽固不动且高频 → 它们到底是 Pillar 还是 Stumbling Block?→ 用 What/Why/How 三阶段诊断并据此挑战均匀加权。【原文】§1
- 与最近邻工作的Δ:RLVR 的 token 重要性来自 entropy/reward 推断;本文用 **per-token KL 直接度量师生失配**这一 OPD 独有信号定义 token 类型,并发现"高 KL ≠ 高功能价值",方向与 RLVR"高熵=高价值"相反。【原文】§1

## 怎么做 + 靠不靠谱
- 方法流水线:① 在 N=500 条 MATH-500 学生 rollout 上算 per-token reverse-KL b_ℓv(Rock Score),频率-KL 平面分离出高频高 KL 的 Rock Tokens,Top-K=100(覆盖约 60% 语料 KL、Jaccard 稳定性在 n∈[50,400] 稳健)→ ② 梯度几何分析(RQ1)与 inference-time knockout 因果探针(RQ2/3)分类 Pillar/Neutral/Stumbling → ③ window-aware 选择性蒸馏:从训练起对 Rock 相关 token 及其局部高散度窗口冻结梯度(λ=0)→ ④ 对比 Baseline OPD(λ=1)/ Rock-Freeze(ours)/ Freq-Matched Random Freeze 三 regime。【原文】§2.2、§3.2、§4.2
- 逐组件必要性:
  - **Rock Score + Top-K 选择**:定义身份。消融充分——Top-K 扫描图(Fig.2c/d)证 K=100 是覆盖/稳定最佳交点,更大 cutoff 反退化。【原文】Fig.2
  - **梯度几何分解(Eq.16,RQ1)**:回答"是否仍提供有用信号"。无它则无法解释"为何停滞"。结论是**反直觉的"Gradient Paradox"**——Rock Tokens 的每次梯度幅值小(median 0.016 vs 稀有高KL 0.54)但方向与 frequency-balanced 全局梯度正对齐(cos 0.040 > 高KL 0.025 > random 0.006),故它们**确实指向"正确"优化方向**,但因方向与全局下降不可分离而得不到 token 专属学习信号,加上(假设)Adam 对高频 token 二阶矩膨胀压低有效学习率,导致停滞。【原文】§2.6
  - **knockout(Eq.17,RQ2/3)**:核心因果证据。MATH-500: 7 Pillar / 0 Stumbling / 193 Neutral;IFEval: 3 Pillar / 0 Stumbling / 197 Neutral(候选池 |R̃|=200,paired-bootstrap α=0.05)。【原文】Fig.4、§3.3
  - **window-aware freeze + Random 控制**:利用阶段。Random(频率/窗长匹配)对照排除"冻任意 token 都无害"的平凡解;Fig.5 显示 Random 严重掉点而 Rock-Freeze 维持高准确率上限。【原文】§4.2、§4.3
- 关键机制/公式(直觉):Rock Score = 学生 rollout 下条件于某 token 的期望 reverse-KL;knockout Δ = 屏蔽该 token logit 后的准确率差;选择性蒸馏对"Rock token ∪ 其高散度窗口 W_R"赋权 λ∈[0,1],λ=0 即"Just Not Train"。【原文】§2.2、Eq.17、Eq.18-19
- 实验与证据:teacher=Qwen3-30B-A3B-Instruct-2507(MoE,3B active)→ student=Qwen3-4B-Instruct-2507,均关 thinking。两阶段蒸馏:Stage-1 离策略用 OpenThoughts3 的 20k teacher 解,Stage-2 在线策略从另 10k prompt 采样、7 个 checkpoint。评测 LM-Eval-Harness zero-shot Pass@1(5 次平均):AIME24/AIME25/HMMT25-Feb(合 90 题主指标)+ MATH-500/IFEval(大样本 token 统计)。关键数字:Pillars 仅占 Rock Tokens 的 1.5%–3.5%、Strong Stumbling 为 0;冻结 ΔKL 精确集中于零;对 30% 高成本 token freeze-weighting 得 1.4× wall-clock(正文另称 18% token、1.7×)。baseline 公平(Random 频率/窗长匹配)。【原文】§3.3、§4.3、§5.1
- 假设与失效边界:
  - 【原文】仅单一师生对(30B-A3B→4B)、关 thinking 模式;Adam 抑制为**假设**,作者明说留待未来形式化验证(§2.6)。
  - 【推断】"功能贡献可忽略"基于数学/IFEval 准确率;对强依赖格式/话语连贯的长文档/对话任务未必成立。knockout(推理时屏蔽)与 freeze(训练时不更新)是两种不同干预,二者结论的可迁移性未严格论证。
  - 【推断】Rock 集合基于最终 checkpoint + MATH-500 轨迹,跨 tokenizer / 更大跨度师生的迁移性未验证。
- 祛魅总结:
  - 真贡献:首个 OPD token 级动力学的系统诊断;提出 Rock Score 与 knockout 因果探针;揭示"Gradient Paradox"与"Pillar 极少但致命、Stumbling 为零"的非对称结构;给出"非均匀梯度分配"的可落地提速。【推断】
  - 比 v1 旧分析更准的关键纠偏:**并非"Rock Tokens 全无用"**——RQ1 表明它们提供方向正确的梯度(冻全部会掉点),knockout 表明 1.5–3.5% 是不可或缺 Pillar 且**与 entropy/频率/KL 等传统指标正交**(|r|<0.07),作者据此警告"按 loss/熵加权的重要性方案有压制 Pillar 的风险"。所以贡献是"**选择性、缓和**地降权高成本 token 求提速",而非"砍掉它们"。【原文】§2.6、§3.3
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
- 💰 资源/成本与可扩展性:硬件 4×H100(80GB),FSDP2+bf16+gradient checkpointing,超参沿用 KDFlow 默认(Appendix C);收益=对高成本 token 冻梯度换 1.4–1.7× wall-clock 提速。【原文】§5.1、§4.3
- 🎯 对"探索-巩固"对标:**可借组件**为主,弱竞品。判定:它的"识别学生顽固抵抗的脚手架 token + 区分 Pillar/Neutral"与本项目"path-token 选择性监督"同源——给"哪些 token 该被 teacher 监督、哪些是学生自有可保留的脚手架"提供了**因果级判据(knockout)**,可直接迁移到 TSRD 的"巩固/回轨 token 选择";但它做的是**降权/冻结**(减信号),与"探索=偏向自己走得通的开头"是反向操作,且无 path-recovery 概念,故不构成竞品。依据:§3.3 Pillar 与传统指标正交 → 提示本项目不能简单按高熵/高 KL 选监督 token。【推断】
- 🔭 开放问题/未来方向:【原文】Adam-suppression 假设的形式化验证(§2.6);Pillar 的可预测性(目前与所有传统指标正交,§3.3)。 【推断】跨任务/跨 tokenizer 的 Rock 集迁移;thinking 模式与更大跨度师生下是否成立;把 knockout 的因果 Pillar 信号反哺到 token 级加权方案(替代按 loss/熵加权)。

RETURN: rock_tokens|读到PDF?是(23页,_txt 88k字)|L1|对标=可借组件(knockout 因果判据→path-token 选择;非竞品,做的是降权而非探索)|残留待核 1(跨任务 Rock 集迁移性)

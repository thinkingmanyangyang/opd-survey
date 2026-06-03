brts | On-Policy Distillation with Best-of-N Teacher Rollout Selection (BRTS) | JHU + TikTok + UCSD + 复旦(Ke Zhang JHU/TikTok 实习;通讯 Di Fu, TikTok) | 2026-05·arXiv 预印本·v2(2026-05-13, 标 cs.CV) | 主题线 L1(在线策略蒸馏)·相关性 高

**原始论文**:https://arxiv.org/abs/2605.09725 （arXiv:2605.09725v2）

## 一眼看懂
- 🟦 TL;DR:标准在线策略蒸馏(OPD,即"学生用自己采样的轨迹、由 teacher 逐 token 打 log-prob 当监督")有两个隐患——监督算在"学生自己跑出来、可能跑歪"的噪声前缀上,而且每道题通常只用**一条随机抽的 teacher 轨迹**,方差极大,抽到错的/不匹配的就会被放大。BRTS 的做法:每道题先采 N 条 teacher 轨迹,按"先看答对没(correctness)、再看像不像学生当前会走的(alignment)"挑一条;若 N 条全错,就把 ground-truth 答案当"静默校验"偷偷塞进 prompt 让 teacher 重新自然推导出一条对的(path-recovery);把选中的这条轨迹作为一条额外的 **teacher-context 蒸馏分支**,和标准 student-context OPD 一起训(总损失 = L_stu-ctx + λ·L_tea-ctx,λ=10)。
- 最巧的一步:**抽掉"correctness-first 选择 + Tier-2 ground-truth 恢复"这一层,方法就退化回普通 OPD**。因为 BRTS 真正新增的不是损失形式(teacher-context KL 本身很朴素),而是"在 OPD 内循环里把随机 teacher rollout 结构化成可靠监督"这个 curation 步骤——尤其是难题上全错时还能靠注答案"救"出一条对的轨迹(§3.2 Tier-2),保证最该给监督的地方分支不空转。为什么是它:论文全部增益(Table 1 候选池 0/2/4 递增、Table 2 Tier-2)都挂在"挑得准/救得回"上,损失形式不变(§3.3 明说 "does not replace OPD; it augments")。

## 为什么做
- 研究背景:OPD 已成 LLM 后训练标准工具(§1 列 Qwen3/MiMo/GLM-5 等工业 pipeline 采用,以一小部分 RL 算力获可比增益)。相比在 teacher 文本上做 SFT 或序列级蒸馏,OPD 因"监督定义在学生推理时真实访问的状态上"而更少 exposure bias。
- 解决的具体痛点:① 监督算在"不完美学生生成、可能漂到噪声推理态"的前缀上(student context 噪声,§1);② 每 prompt 通常只依赖**单条随机 teacher rollout**,而 teacher 本身随机——同题不同采样在正确性/推理风格/与学生接近度上差异大(尤其难题),单样本是对 teacher 该题能力的高方差估计,抽到错的/不匹配的就放大噪声(§1 + Fig.1a)。
- 相关工作 & 各自不足:(a) 解析 OPD 何时成功的工作[26]指出师生需"兼容推理模式 + teacher 能产出超学生探索范围的自然正确解";小模型难模仿风格不匹配的强 reasoner[9,14,27]。(b) best-of-N / 拒绝采样 / 自训练[6,8,28,39,43,53]做的是 **outer-loop 离线过滤**保留高质量样本;(c) 用 privileged 信息(ground-truth/示范)[7,18,32,38,40,48,55]增强监督。BRTS 的差异:把选择放进 **OPD 内循环**,选中轨迹不是离线训练数据而是"为当前这步 OPD 更新定制的 teacher-context 信号"(§2 第三段)。
- 动机链:OPD 好用但 teacher 监督在难题上不可靠(噪声前缀+单样本高方差)→ 单纯换 KL 散度或加更多 student rollout 不解决"teacher 信号本身是错的/不匹配"→ 所以必须先**识别/恢复一条可靠 teacher 轨迹**再蒸馏,且该分支要 correctness-aware(防错放大)+ alignment-aware(选最像学生的)。为什么不用更简单的现成做法:离线 rejection sampling 会引入分布失配且不针对当前学生状态;只加 student rollout 不能注入"超出学生范围的新能力"(§2)。
- 与最近邻工作的Δ:最像的是离线 best-of-N/拒绝采样和 privileged-hint 方法。**关键差一点**:BRTS 把"选 + 救"放在 OPD 训练步内,且新增的是一条**反向条件**的 teacher-context 分支(在 teacher 自己的可靠前缀上、用 teacher 的 top-K 候选监督学生),而非把选中样本当 SFT 目标。为什么有用:teacher-context 分支让学生见到"完整、连贯、且选得贴近自己当前分布"的推理路径,补上 student-context 分支看不到的"完整正确解"(§3.3)。

## 怎么做 + 靠不靠谱
- 方法流水线(Algorithm 1,§3):
  1. 输入:prompt x、ground-truth y*、teacher π_T、student π_S、Tier-1 采样数 N、辅助权重 λ。
  2. 学生采 1 条 on-policy rollout ŷ_S;teacher 采 N 条 unconditioned 轨迹 {y_T,i}。
  3. **Tier-1 选择**:逐条按"末答案是否=y*"判对错。若 ≥1 条对 → 在对的里选与学生 top-K 候选集 **token 重叠最高**者(alignment proxy)。
  4. **Tier-2 恢复**(全错时):构造 x_gt——把 y* 作为"SILENT_VALIDATION_KEY,严禁在 think/答案里提及,只能推完后当 sanity-check"(附录 B 给出完整 prompt 模板),采 1 条 guided rollout,**仅当其答案正确才保留**。
  5. **Tier-3 fallback**(仍无对的):回退到 Tier-1 里 student-overlap 最高的那条(可能错,但避免引入离学生很远的任意轨迹)。
  6. 计算 student-context 损失(在 ŷ_S 上,用学生 top-K)+ teacher-context 损失(在选中 y' 上,用 teacher top-K);L_total = L_stu-ctx + λ·L_tea-ctx;对学生参数走一步梯度。输出:更新后的学生。
- 逐组件必要性:
  - **Tier-1 候选池 N**:负责降低"抽到错轨迹"的方差。有消融——Table 1 候选 0/2/4 递增,AIME24 mean 0.3917→(1/2)0.3750→(1/4)**0.400**;Fig.6 显示 Tier-1 命中率 2 候选 52.73% → 4 候选 66.70%。没它就退化回单样本高方差。
  - **alignment(student top-K 重叠)选择**:负责选"学生够得着"的轨迹。**未单独消融** alignment vs 随机选(论文只比候选池大小,没做"correct 里随机选 vs 选最像学生"的对照)——这是一处证据缺口〔推断〕。
  - **Tier-2 ground-truth 恢复**:负责难题上"全错→救出对的"。有消融——Table 2 加 Tier-2 后 AIME25 mean 0.2667→**0.3000**;Fig.4 显示 Tier-2 在更难的 AIME25 上早期持续抬升;Fig.6 显示 Tier-2 在 1 个 Tier-1 候选时多救 25%、2 个时多救 16.8%。
  - **teacher-context 用 teacher top-K(§3.4)**:引入学生当前 top 之外的 teacher-preferred token,这是"注入新能力"的载体。**未单独消融**(没对比"teacher-context 也用 student top-K")〔推断〕。
  - **λ=10**:§3.3 明说"λ=1 时该分支原始贡献太小,经验取 λ=10 给它有意义的尺度且稳定,全实验用此值"。**未给 λ 敏感性扫描**〔推断:仅给单值,无 5/10/20 对照〕。
  - **prompt 扰动**(附录 B):给两条 teacher rollout 之一加 "rethink in detail" 指令做去相关。Table 3 显示 AMC23 早期 majority 0.6663→0.6764;Fig.5 解释"未加多样性时 Tier-1 命中率远低于 i.i.d. 理论曲线 1−(1−p)^n,说明 teacher 样本强相关,轻微扰动能部分去相关"。
- 关键机制/公式(直觉,不复述符号):
  - 两条 KL **方向相反**(这是本文一个被低调处理、但很重要的设计):student-context(Eq.3)是 **reverse-KL**(π_S 在前,KL(π_S‖π_T)),"在学生访问的态上把学生拉向 teacher";teacher-context(Eq.4)是 **forward 方向**(π_T 在前,KL(π_T‖π_S)),"在 teacher 可靠前缀上,把学生的局部分布拉去覆盖 teacher 的分布"。直觉:reverse-KL 是 mode-seeking(让学生在自己态上别犯错),forward-KL 是 mode-covering(让学生别漏掉 teacher 在好路径上偏好的 token)。**论文未论证为何两支选相反方向**——这是逻辑链上一个未解释的设计选择〔推断〕。
  - Tier-2 的"静默校验"巧思:不让 teacher 直接抄答案,而是逼它"自己推、推完再拿答案验证",从而得到一条**自然推导**而非"看答案倒推"的轨迹——这才符合 OPD"teacher 产出自然正确解"的成功前提(§3.2 + 附录 B 模板)。
- 实验与证据:
  - 数据集/设置:训练 prompt = DAPO-Math-17K(可验证短答案);评测 AIME 2024 / AIME 2025 / AMC 2023,k=4 解、temp 0.7、top-p 0.95,报 mean/best/majority;主实验 teacher=**JustRL-1.5B**、student=**DeepSeek-R1-Distill-Qwen-1.5B**(同尺度 1.5B,共享 tokenizer 以简化 top-K 对齐);teacher-swap 换 DeepSeek-R1-Distill-Qwen-7B;8×B200;lr 1e-6 常数无 warmup、token-mean、mini-batch 64、bf16、关掉对 frozen ref 的 KL(附录 A)。
  - 支撑核心主张的关键数字:Table 1 候选池递增 AIME24 mean 0.3917→0.400(+约 0.8 个百分点)、majority 0.4146→**0.4306**;Table 2 Tier-2 使 AIME25 mean→0.3000、best 0.4309;Table 4 teacher-swap 到 7B 后早期 AIME24 mean 0.3167→**0.3667**、majority→0.3952(换 teacher 仍有效)。
  - baseline 公平吗:基线是"2 条 student rollout",BRTS 是"1 student + 1 teacher(从 N 候选选)"——**总用于 loss 的轨迹数相同(都是 2 条)**,额外成本只在选择阶段采样(每多 1 候选 +约 59s/step,Table 6:1→4 候选 281s→460s)。这个对照设计是公平的(控制了 loss 轨迹数)。
  - "看着强但没回答核心问题"的结果:AMC23 几乎无提升(Table 1 各设置 mean 0.67–0.68 ≈ 基线 0.6777),论文归因"student-only 基线已强";增益**绝对值小**(AIME24 mean 仅 +0.8pt),"显著改善"的措辞与量级有张力〔推断〕。另外主结果是同尺度 1.5B 师生——"teacher 产出超学生范围的解"这一 OPD 成功前提在同尺度下能否成立,靠的是"teacher JustRL-1.5B 经强 RL 故仍强于 distill student",但论文未直接量化师生能力差〔推断〕。
- 假设与失效边界:
  - 显式假设【原文】:teacher 能在某些题上产出正确自然解;师生**共享 tokenizer**(否则 top-K 对齐失效,附录 A 明说"share the same tokenizer simplifies top-K alignment");有可验证 ground-truth 答案(Tier-2 依赖 y*)。
  - 隐式假设【推断】:teacher 与 student 推理模式兼容(否则 alignment 选出来的也不可用);top-K token 重叠是有效的"学生可达性"代理(论文只当 proxy,未验证它真比别的对齐度量好)。
  - 何时失效【推断】:① 无 ground-truth 可验证的开放生成任务(Tier-2 失效,附录 D 自己承认需换成 retrieval/verifier/更强模型);② 师生 tokenizer 不一致;③ teacher 本身在该领域很弱(N 条全错且 Tier-2 也救不回,只能 fallback 到错轨迹)。
- 祛魅总结【推断】:
  - 真贡献:把"correctness-first + student-alignment 选择"和"ground-truth 静默校验恢复"这两个**轻量 curation 步骤**塞进 OPD 内循环,且控制 loss 轨迹数做公平对照——这是干净、可复用的工程组件,尤其 Tier-2 的"自然推导式恢复"对"难题上分支不空转"有清晰价值。
  - 包装/高估:"significantly improves"在 AIME 上的绝对增益其实很小(<1pt mean),AMC 基本持平;评测仅竞赛数学三个集,泛化覆盖窄;同尺度师生使"OPD 注入新能力"的叙事打了折扣。
  - 低估之处:作者其实给了一个挺通用的框架视角(附录 D:Tier-2 可推广到 retrieval/tool-feedback/learned-verifier/更大 teacher 引导小 student),但正文实验没兑现,把一个"可推广的可靠监督构造范式"局限在了数学。

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号:teacher 在两种 context 下的 **token 级 log-prob / top-K 分布**(student 前缀上的 reverse-KL + 选中 teacher 前缀上的 forward-KL)。
  - 改什么:**参数**(梯度更新 student;改的是 logits 分布对齐)。
  - 何时改:**在线 per-step**(每训练步现采 teacher 候选、现选、现算两支损失)。
  - 免梯度?:否(标准梯度下降训 student)。
  - 记忆-技能生命周期:**无显式记忆/技能库**;teacher 轨迹是 per-step 临时构造、用完即弃(不写入持久存储)。
  - 防遗忘机制:间接——OPD 范式本身(监督在学生访问态上、关掉对 ref 的 KL 但靠 teacher 分布锚定)被认为比 SFT 少遗忘;论文还提"早期达峰→可缩短训练→少遗忘"(§4.2 Computational Cost),但**未做遗忘基准评测**,防遗忘是继承自 OPD 的论断而非本文实测〔推断〕。
- ⑦ 开源代码 + 框架/harness:https://github.com/BWGZK-keke/BRTS (已 clone 约 14MB,Tier A 可跑);框架 **veRL**(`python -m verl.trainer.main_ppo`,`ADV_ESTIMATOR=token_reward_direct`,rollout 用 vLLM,teacher 当 reward_model 提供 token 级 reward,swanlab 记录);与 thunlp/OPD(rethink_opd)同源 fork。脚本:on_policy_distillation.sh / grpo.sh / forward.sh(Tier-1 only)/ forward_tier2.sh(含 Tier-2)。
  - **〔待核·文-码不一致〕**:论文 §3.3 与附录 A 均明确 λ=10;但 v1 分析记录仓库脚本默认 `AUX_TEACHER_CTX_KD_COEF=0.05`。复现需对齐(本轮未重新打开脚本核对,沿用 v1 记录,标待核)。
- 💰 资源/成本与可扩展性:8×B200 单节点;每多 1 个 teacher 候选 +约 59s/step(Table 6:1/2/3/4 候选 = 281/338/400/460 s/step),线性增长;**不增加** loss 用的轨迹数(选中后只用 1 条);teacher 解通常更短更直接故 forward 开销可控;max response 7168(训练)/31744(验证)。可扩展性:论文称可加大候选池、用多条选中轨迹做损失(附录 D),但要权衡采样成本 vs 质量。
- 🎯 对"探索-巩固"对标:**强支撑 + 可借组件**。判定:BRTS 的"Tier-2 ground-truth 静默校验恢复"几乎就是我们"巩固/回轨"里"走偏后从恢复分支自选一条能接着做对的"的一种 teacher-scaffolded 实现——只不过它是 teacher 端恢复、再蒸给 student;"alignment-second 选最像学生的轨迹"对应"偏向自己能走通的开头/方向"。依据:§3.2 三层 curation + 附录 B 静默校验模板;对我们的 idea,可直接借的积木是"correctness-first→student-alignment-second 的轨迹优先级规则"和"把 ground-truth 当静默 sanity-check 逼出自然推导(而非看答案倒推)"这两个机制。缺口:它是 teacher 外部恢复 + 同尺度,而我们要 student **全程 on-policy 自选**恢复分支;BRTS 不涉及 MTP/前瞻。
- 🔭 开放问题/未来方向:
  - 【原文】(附录 D)推广到无 ground-truth 的开放生成(Tier-2 改用 retrieval-hint / tool-feedback / learned verifier / 更强模型如 32B 引导 1.5B);更有表现力的师生兼容性度量(替代 top-K 重叠);把 OPD 做成**课程式**——自动识别题目对当前学生是 easy/learnable/too-hard 并自适应调 teacher 监督("像人类老师先识别学生能力再传授");自适应选 λ、候选数、选中轨迹条数。
  - 【推断】两支 KL 方向相反的理论依据值得补;同尺度师生下"注入新能力"vs"仅降方差"需解耦;应补遗忘基准、补"alignment 选择 vs 随机选"和"teacher-context top-K 方向"的消融以坐实各组件归因。

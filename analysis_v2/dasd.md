dasd | Distribution-Aligned Sequence Distillation for Superior Long-CoT Reasoning (DASD-4B-Thinking) | 阿里云(Shaotian Yan*, Kaiyuan Liu*, Chen Shen*†, Bing Wang*, Sinan Fan* 等) | arXiv 2601.09088v1(2026-01-14)·技术报告 | 主题线 L1(序列级蒸馏/含轻量 on-policy 阶段)·相关性 高

**原始论文**:https://arxiv.org/abs/2601.09088

## 一眼看懂
- 🟦 TL;DR:主流"在 teacher 生成的响应上做 SFT"(序列级蒸馏)只把它当数据过滤问题,忽视了蒸馏本质——让 student 学到 teacher 完整序列分布。DASD 从"分布对齐"视角补回师生交互三件套:温度调度学习(先低温学一致模式再高温扩覆盖)、散度感知采样(优先喂"teacher 高置信+student 低概率"的句子)、混合策略蒸馏(student 自生成前缀→teacher 续写,缓解 exposure bias),只用 448K 样本就让 Qwen3-4B 在 AIME24/25、LCB、GPQA 上达同量级 SOTA,部分基准超 32B 模型【原文 Abstract, §1, Table 6】。
- 最巧的一步:**散度感知采样(Divergence-aware Sampling, DAS)**。抽掉它,整套就退回"随机采样+质量过滤"的旧范式,核心论点(分布对齐比堆数据重要)就垮——Table 3 显示 DAS(50K)在 AIME25 上(79.2)甚至超过随机采样翻倍数据(100K RS,78.9)【原文 §4 Table 3】。为什么:它把"哪些 teacher 句子最该学"从启发式规则升级为"按师生概率差(teacher 高/student 低=高散度)筛选",直接对齐 student 学习能力、规避 SFT 的误导梯度(对 teacher 低概率但 student 已高概率的 token 继续抬高的反向梯度)【原文 §4】。

## 为什么做
- 研究背景:DeepSeek-R1 首次证明从强 teacher 蒸馏可大幅赋能小模型推理,引发社区大量复刻;主流范式即"SFT on teacher responses = 序列级蒸馏(Kim&Rush 2016)",简单高效、不限师生架构、不需 token 级 logits【原文 §1, §2】。
- 解决的具体痛点:现有工作只停留在 SFT 视角、专注设计启发式数据过滤规则,**忽视蒸馏本质(继承 teacher 泛化能力)**,根因是全程缺乏显式师生交互,导致三大缺陷:(i)teacher 序列分布表征不足(随机采样覆盖窄、或过度表征低概率噪声序列);(ii)teacher 分布与 student 学习能力错配(SFT 只抬 ground-truth token 概率→误导梯度);(iii)exposure bias(teacher-forced 训练 vs 自回归推理不一致)【原文 §1, §2】。
- 相关工作 & 各自不足(来龙去脉 + 三条并行路线 + 精确差异):
  - **路线 ①——序列级蒸馏(本文改进对象,Kim&Rush 2016)**【原文 §1, §2】:把 teacher 生成响应当 SFT 数据;社区复刻潮(OpenR1 / OpenThoughts / a-m-team / AceReason / LIMO / s1 / Light-R1)全属此族,优势是简单、不限架构、不需 logits。**精确短板**:全都把问题简化为"过滤高质量 SFT 数据",缺师生交互,落入上述三缺陷。DASD 站在它肩上(保留序列级简单性),但把"采样+选数据"从启发式升级为分布对齐。
  - **路线 ②——logit 蒸馏(Hinton 2015)+ 其 on-policy 变体(Qwen3、Gemma、Thinking Machines Lab)**【原文 §1 line 173–182】:对齐师生**逐位置 next-token 分布**(最小化 token 级 KL);其 on-policy 变体先用 student 自生成序列、再对齐 logit。**精确短板**:(a)需 teacher 全词表 logits;(b)师生不同 tokenizer 时输出空间错位、难对齐;(c)即便简化到 token-level 概率,on-policy 蒸馏仍需"对 student 自生成的每个 token 拿到师生双方概率",而 **teacher 对 student 输出的概率对闭源模型通常不可得**(§4 line 55–58)。DASD 用**句子级**(几何均值)分析规避这些约束——只需 teacher 对**自己生成**响应的 token 概率(采样时天然得到,且很多闭源 API 也暴露)+ student 本地算。
  - **路线 ③——on-policy distillation(token 级监督)**【原文 §4 line 626/831】:要求师生**同 tokenizer/词表**(token 对齐的硬约束)。DASD 的句子级分解(只比概率大小、不要求 token 对齐)正是为绕开此约束而设计。
- 动机链:现状(序列级蒸馏火但只当 SFT 数据问题)→缺陷(缺师生交互→三大问题)→所以必须(不改用 logit 蒸馏、保留序列级简单性,但从分布对齐角度补回师生交互:更好覆盖+更优目标分布+缓解 exposure bias)【原文 §1, §3】。
- 与最近邻工作的Δ:相对纯 SFT 蒸馏(OpenR1/LIMO 等),差在"把采样策略和数据选择从启发式提升为分布对齐(温度调度+DAS)+ 加一个轻量 on-policy 混合阶段";相对 on-policy/logit 蒸馏,差在"只需师生对每个 teacher token 的概率(句子级 geo-mean),不需全词表 logits、不要求同 tokenizer"。有用点:用极少(448K)数据达 SOTA,且 DAS 数据可跨 student 复用(§4 line 830:为 4B 筛的数据可迁移到 30B-A3B)。

## 怎么做 + 靠不靠谱

### 0. 理论锚点:为什么"SFT on teacher data"本身就是蒸馏(Eq.1-5,本文立论基石)
给输入 \(x\),序列级蒸馏让 student \(p_S\) 在**整条响应**层面逼近 teacher \(p_T\)【原文 §2, Eq.1-5】:
\(\displaystyle \min_{\theta}\;D_{\mathrm{KL}}\!\big(p_T(y\in\mathcal Y\mid x)\,\|\,p_S(y\in\mathcal Y\mid x)\big),\qquad \mathcal Y=\text{teacher 对 }x\text{ 所有可能响应} \tag{1}\)
展开 KL 并丢掉与 \(\theta\) 无关的常数项 \(p_T\log p_T\):
\(\displaystyle \mathcal L_{\mathrm{SEQ}}=\sum_{y\in\mathcal Y}p_T(y\mid x)\big[\log p_T(y\mid x)-\log p_S(y\mid x)\big] \;\;\longrightarrow\;\; \mathcal L_{\mathrm{SEQ}}=-\sum_{y\in\mathcal Y}p_T(y\mid x)\log p_S(y\mid x) \tag{2,3}\)
\(\mathcal Y\) 指数大、不可枚举,故用**采样响应 \(\hat y\) 的点质量**近似 teacher 分布:
\(\displaystyle p_T(y\mid x)\approx \mathbf 1\{y=\hat y\}\;\Longrightarrow\;\mathcal L_{\mathrm{SEQ}}\sim-\sum_{y\in\mathcal Y}\mathbf 1\{y=\hat y\}\log p_S(y\mid x)=-\log p_S(\hat y\mid x) \tag{4,5}\)
Eq.5 **正好就是 teacher 数据上的标准 SFT loss**——这一步是全文的"立论"：它解释了为何"SFT on teacher 数据"有效(本质是用单样本点质量近似 teacher 序列分布),也直接推出"**怎么选 \(\hat y\)(采样策略)决定了对 \(p_T\) 的近似质量**"——这是温度调度与 DAS 的出发点。Kim&Rush 用 beam search 取 \(\hat y\)≈众数,近期工作用随机采样;两者都只覆盖 \(p_T\) 的窄子集。

### 1. 温度调度学习(Temperature-scheduled Learning,补"覆盖不足")
- 干什么:teacher 用低温 \(T=0.6\) 和高温 \(T=1.0\) 各采多响应;**两阶段 SFT**——先在低温样本上 cold-start(50K,T=0.6),再在高温样本上续训(50K,T=1.0)【原文 §3, §6】。
- 直觉与证据:低温分布尖锐、质量集中、易学(loss 平滑下降),但覆盖窄;高温分布平展、覆盖广、数据多样,但引入大量罕见 teacher 模式/噪声样本,小 student 难学(loss 居高)。先低温稳住早期学习、再高温扩覆盖。Table 1:静态单温度对比中 T=1.0 一致优于 T=0.6(AIME24 +1.4 / AIME25 +4.2),但纯高温堆数据边际递减(100K T=1.0 相对 50K 在 AIME24 零增益、AIME25 仅 +2.8)→说明 student 吸收多样 teacher 行为的能力是瓶颈,故需调度而非单纯加量。
- 表征工具:为刻画一条响应的整体似然,用其 **token 概率的几何均值**(§3 Fig.3 注、§4 line 615)。

### 2. 散度感知采样(DAS,补"师生能力错配"——最核心)
- 前置发现:**四类句子分解**【原文 §4, Fig.5/6】。把每条响应切成句子,算 teacher/原始 student/蒸馏后 distilled 三模型对每句的概率(几何均值)\(p_T,p_S,p_D\),按相对大小分四类:
  - **Teacher Sentence**:\(p_T\gg p_S\) 且 distilled 仍输出该句 → 主要源自 teacher;此时 student 可在 SFT 下**放心抬高概率而无误导梯度之虞**。
  - **Student Sentence**:\(p_S\gg p_T\) → 主要源自 student。
  - **Shared Sentence**:三模型概率相近 → 师生本就共有、蒸馏未改变。
  - **Boosted Sentence**:\(p_T\approx p_S\) 但 \(p_D\) 显著(通常更高)→ 师生本有但被蒸馏放大。
- 关键实证(Fig.6):按句位置统计各类句子概率与答案正确性的相关。**Teacher Sentence 在正确答案中概率持续更高**(浅绿实线恒高于虚线,面积差 \(\Delta\) 为正);**Boosted Sentence 反而负相关**(\(\Delta<0\),作者推测源自次优误导梯度);Shared/Student Sentence 概率低、影响小。结论:**优先学 Teacher Sentence**。
- 可前置识别(DAS 之所以可行):虽然完整四类分解需要 distilled 模型(训练后才有),但 Teacher/Student Sentence **训练前即可识别**——只看 teacher 是否对该句赋显著高于/低于 student 的概率即可【原文 §4 line 793–798】。
- DAS 机制:优先选**Teacher Sentence 丰富**的训练样本,从而隐式逼近一个"更契合 student 学习能力"的 teacher 派生序列分布,天然规避误导梯度。**资源足迹极小**:每个 teacher token 只需师生双方概率(teacher 侧采样时即得、闭源 API 多暴露;student 侧本地算),不需全词表 logits、不需 teacher 对 student 输出打分(后者闭源不可得)。
- 证据(Table 3/4):同采样预算 DAS 一致优于随机采样(50K DAS T=1.0 AIME25=79.2 vs 50K RS=76.1,且超 100K RS=78.9);换 teacher(Qwen3-Next-80B-A3B-Thinking)、跨域(数学/代码/科学)均成立;且 DAS 数据**跨 student 复用**(为 4B 筛的数据迁到 30B-A3B 仍有效,§4 line 830)。

### 3. 混合策略蒸馏(Mixed-policy Distillation,补"exposure bias"——迷你 path-recovery)
- 协议(§5, §6.3 line 1071–1078,可复现):从 DAS 集采 50K 题让 **student 自己生成**解 → 在**超过总长一半的随机位置截断**、丢弃后半 → **teacher 把丢弃部分重写(续写)** → 通过质量过滤的 teacher 续写保留 → 得 **12.7K** 混合策略样本,加入 student 进一步微调。
- 直觉:student 在长响应上会逐渐偏离 teacher(Fig.7 cut-off 率随长度上升),纯 teacher-forced SFT 训练态与自回归推理态不一致(exposure bias)。让 student 先在自生成(on-policy)前缀上"走到一半",再由 teacher 在 student 真实落点处给出修正续写,使训练直面 student 自己的分布。
- **关键消融(Table 5,有价值的负结果)**:仅 7.7K 混合数据(no-mask)即提升(baseline 83.3/74.2 → 83.3/74.8);但若**把 student 自生成段 mask 掉、只在 teacher 续写段算 loss,反而更差(80.8/72.3)**——证明训练时**保留 on-policy student 段不可去**(去掉就丢了 exposure bias 的纠正价值)。

### 训练流程与关键超参(复现锚点)【原文 §6, §7】
- 数据(总 ≈448K):跨域难题(数学 105K + 代码 + 科学 + 指令);teacher 在 T=0.6/T=1.0 各采多响应 → DAS 筛 → 结构/长度/重复过滤(显式剔除含 function call 的响应,§6.3 line 1035)→ 105K 低温 + 330K 高温 + 12.7K 混合策略。
- 师生:teacher **gpt-oss-120b**(另验 Qwen3-Next-80B-A3B-Thinking);student **Qwen3-4B-Instruct-2507**(MoE 版另用 Qwen3-30B-A3B-Instruct-2507)。〔v1 待核 RESOLVED:正文 §6.1 line 937 与 Table 1/3/4 一致写 student=Qwen3-4B-Instruct-2507,以此为准;"-Thinking-2507" 系发布命名,非训练 base〕
- 训练:full SFT、cutoff length **64K**、greedy packing、**ZeRO-3 + Liger kernels**、global batch **64**、**6 epochs**、lr **5e-5→1e-5**(cosine)。
- 评测:统一 temperature 1.0 / top-p 1.0、每题采 64 报均值、AIME 最大生成 102400 token、LCB/GPQA 81920。
- 主结果(Table 6):DASD-4B = AIME24 88.5 / AIME25 83.3 / LCB v5 69.3 / LCB v6 67.5 / GPQA-D 68.4,超 Qwen3-4B-Thinking-2507(AIME25 81.3)、Qwen3-32B(72.9)、AM-thinking-v1-32B(74.4,且用 2.9M 数据 vs 本文 448K)。

### 靠不靠谱
- baseline 公平性:对照分"开权重"与"开权重+开数据"两族,数据规模对比突出(448K vs 2.9M/30M),较公平;但**三件套多为逐项独立消融(各在不同小规模子集),缺少在最终 448K full pipeline 上的逐项可加性消融**,无法判定三者叠加时各自净贡献【推断,据 §3-5 消融均为独立子实验】。
- 假设与失效边界:【原文】DAS 需拿到师生对每个 teacher token 的概率(发布了 -Logprob 数据集);结构过滤显式剔除含 function call 的 teacher 响应(§6.3,工具调用留给未来工作)——故当前不适用于工具/agent 蒸馏。【推断】"Teacher Sentence 与正确性正相关"为观测性结论(Fig.6 面积差 \(\Delta\)),因果未隔离;四类句子分解依赖句子切分,跨语言/无明确句界的输出可能失效;teacher 概率对闭源 API 不总可得(虽作者称 many closed-source APIs 也暴露 teacher 概率)。
- 祛魅总结:真贡献=把序列级蒸馏从"数据过滤"重新框定为"分布对齐"(Eq.1-5 立论 + 四类句子分解 + DAS),给出可操作的 DAS(只需师生 token 概率,跨 tokenizer 可用)+ 数据效率惊人(448K 达 SOTA)+ mixed-policy 的 mask 消融是有价值的负结果(证明 on-policy 段不可去)。包装/高估处:title/abstract 的"distribution-aligned"听起来像理论对齐,实际 DAS 是基于经验观测(Teacher Sentence 正相关)的启发式采样,无理论保证;"SOTA 超 32B"成立但基于 4B-Instruct 这一强 base + 强 teacher gpt-oss-120b,蒸馏增益与 base/teacher 质量耦合,论文未拆分【推断】。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=teacher 序列级输出分布(经 DAS 筛的 Teacher Sentence 丰富响应)+ mixed-policy 阶段的 teacher 对 student 错误前缀的续写修正 | **改什么**=student 全参数(full SFT)| **何时改**=离线两阶段 SFT + 一个轻量 mixed-policy 阶段(均为训练期)| **免梯度?**=否(SFT 梯度训练)| **记忆-技能生命周期**=不涉及显式记忆/技能库,推理能力固化进权重 | **防遗忘机制**=无专门防遗忘;温度调度的低温 cold-start 可视为稳定早期学习(弱相关)【原文 §6.4】。
- ⑦ 开源代码+框架/harness:https://github.com/D2I-ai/dasd-thinking(含 train/stage1.yaml、stage2.yaml、deepspeed/ds_z3 配置、技术报告 PDF;**含训练配置但不含数据生成/DAS 采样代码**);框架 **LLaMA-Factory**(README 明确"utilize LLaMA-Factory")+ DeepSpeed ZeRO-3 + Liger-Kernel。本地已 clone(~5.2MB)。模型/数据在 HF & ModelScope(DASD-4B-Thinking、DASD-30B-A3B-Thinking-Preview、Superior-Reasoning-SFT-gpt-oss-120b 及 -Logprob 版)。【纯 SFT 栈,非 RL 框架】
- 💰 资源/成本与可扩展性:论文未给 GPU 卡时/总训练成本(原文未说明);仅知 64K context 训练靠 ZeRO-3+Liger 降显存、global batch 64、6 epochs;数据效率是其卖点(448K vs 同类 2.9M/30M)【原文 §6.4.1】。
- 🎯 对"探索-巩固"对标:**部分支撑(巩固侧 + 一个迷你回轨机制)** —— 一句判定:DASD 的 mixed-policy 阶段(student 自生成前缀 → 随机截断 → teacher 从截断点续写修正)正是"走偏后由 teacher 提供 path-recovery 监督"的离线 SFT 版,与"巩固/回轨"诉求直接同构;依据:§5"randomly cut off student solutions and prompt teacher to continue...targeted guidance on student's errors",且 Table 5 证明保留 on-policy student 段(不 mask)才有效。可借组件:① mixed-policy 的"student 前缀+teacher 续写"构造可直接搬到 OPD 的路径恢复数据生成;② DAS 的"按师生概率差选高散度 token"思想可迁移到 OPD 中"选哪些 token/步该由 teacher 接管"(呼应 MEMORY 中 survey-grpo-step-segmentation 的高熵/低置信切分)。缺口/竞品差异:DASD 是**离线 SFT(teacher-forced 续写),不是 on-policy 自选恢复分支**——teacher 决定从哪截断(随机过半位置)、写什么,student 不自选;无 MTP 前瞻;无 RL/免梯度信号。与本项目"on-policy 自选恢复 + 稀疏脚手架"相比,DASD 的 teacher 介入更密、更 off-policy。
- 🔭 开放问题/未来方向:【原文】§5 称 mixed-policy 是"promising direction"值得继续探索;§6.3 把 function calling/工具调用蒸馏明确列为 future work(当前过滤掉)。【推断】三件套在 full pipeline 的可加性消融、DAS 的理论化(目前纯经验)、跨语言句子分解鲁棒性、与 RL/on-policy KD 的结合、teacher 概率不可得时的退化方案均未解。

---
RETURN: dasd | 读到PDF? 是(20页全文,Eq.1-5 与四类句子定义从 PDF 精确抄录) | L1(序列级蒸馏+轻量on-policy) | 巩固/回轨侧部分支撑(mixed-policy=teacher 续写的离线 path-recovery,但 teacher-forced 非 on-policy 自选);DAS 高散度选样可借鉴 | 残留待核 0(v1 待核已 RESOLVE:正文 §6.1/Table 1/3/4 一致为 Qwen3-4B-Instruct-2507)

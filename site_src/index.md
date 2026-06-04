# 自进化 / 持续学习 LLM Agent — 论文精读手册

> 视角:**探索-巩固(explore-consolidate)**。每篇含「一眼看懂 / 为什么做 / 怎么做+靠不靠谱 / 结构化抽取(6轴·代码·成本·对标·开放问题)」,严格读原文、三标注(【原文】/【推断】/【待核】)、带原始论文链接。

> 共 **117** 篇有代码深读论文。**点左侧 L1–L6 主题即可一页读完该线全部论文**;下表为一句话速览。

## 一句话速览表


### L1 · 在线策略蒸馏 / 自蒸馏 (OPD)（43）

| 论文 | 一句话重点(TL;DR) | 相关性 | 对标 | 原文 |
|---|---|---|---|---|
| [aligndistil](L1.md#p-aligndistil) | 本文把"带 token-level reward 的 RLHF 目标"在理论上**等价改写成一个蒸馏问题**（§3 Theorem 1）。这里的 teacher 分布并非现成大模型，而是 DPO 模型与 reference 模型的 logit 做线性组合后合成出来的。于是 RLH | 高 |  | [📄](https://arxiv.org/abs/2503.02832) |
| [apple_ssd](L1.md#p-apple_ssd) | SSD 不用任何 teacher/verifier/reward/RL/代码执行环境，只让模型在"调过温度 \(T_{\mathrm{train}}\neq1\) + 截断 \(\rho\)"的设定下采样自己的**原始、未验证**输出，再用标准交叉熵 SFT，评估时单独调一个解码 | 中 | **对照/边界案例(而非直接竞品)**——SSD 是**纯 off-policy 自蒸馏**，与 | [📄](https://arxiv.org/abs/2604.01193) |
| [avsd](L1.md#p-avsd) | 自蒸馏让**同一个模型**当 student 又当 teacher，teacher 偷看"特权信息"（完整解 / 部分推理 / 只给答案），在 student 自采的轨迹上给 dense 的 token 级信号。问题是没有哪种特权视图永远最好（§1 Fig.1：Qwen3-4B  | 高 | **支撑 + 可借组件**。判定:AVSD 的"几何共识=可靠方向、门控残差=谨慎补充"与 TS | [📄](https://arxiv.org/abs/2605.20643) |
| [brts](L1.md#p-brts) | 先说什么是 OPD（在线策略蒸馏）——学生用自己采样的轨迹、由 teacher 逐 token 打 log-prob 当监督。标准 OPD 有两个隐患: | 高 | **强支撑 + 可借组件**。判定:BRTS 的"Tier-2 ground-truth 静默校 | [📄](https://arxiv.org/abs/2605.09725) |
| [caopd](L1.md#p-caopd) | on-policy distillation（OPD,这里是自蒸馏形式）在提升准确率的同时,会系统性地把模型推进"过度自信"区。作者把这个现象叫 **Scaling Law of Miscalibration（误校准的标度律）**——连 GPT-5.x/Claude/Gemini | 高 | **支撑 + 警示（边界提供者）**。判定:CaOPD 揭示的"**信息不对称 → teache | [📄](https://arxiv.org/abs/2604.16830) |
| [copsd](L1.md#p-copsd) | 核心观察——低资源语言的数学推理差，往往不是模型没能力，而是"有 latent（潜在）能力但在低资源语下调不出来"。原文："a model may possess the latent ability to solve a problem, yet fail to access  | 高 | **支撑（OPD 主轴的直接同源工作）**。 | [📄](https://arxiv.org/abs/2605.09548) |
| [csd](L1.md#p-csd) | 知识蒸馏（KD，knowledge distillation：让小 student 模仿大 teacher）的主流做法是在每个 token 上做 **softmax 概率匹配**（KL 或 f-散度）。 | 中 | **外围相关（loss 层面）**。 | [📄](https://arxiv.org/abs/2509.25837) |
| [dasd](L1.md#p-dasd) | 主流做法是"在 teacher 生成的响应上做 SFT"(即序列级蒸馏)。但这种做法只把蒸馏当成一个数据过滤问题,忽视了蒸馏的本质——让 student 学到 teacher 的完整序列分布。DASD 改从"分布对齐"视角出发,补回三件师生交互工具: | 高 | **部分支撑(巩固侧 + 一个迷你回轨机制)** —— 一句判定:DASD 的 mixed-po | [📄](https://arxiv.org/abs/2601.09088) |
| [denoiserl](L1.md#p-denoiserl) | 把**弱模型生成的错误推理前缀**当成一种"结构化噪声",注入到策略的 rollout 里,再用 RL 训练策略从这个错误的中间状态"去噪并恢复"到正确答案。这样做的好处是:不引入更强的 teacher、也不构造难数据,就把"自纠错 / 从错误中恢复"从一种涌现行为,提升为**显 | 高 | **强支撑(巩固/回轨侧最近邻竞品 + 探索侧扩状态多样性)** —— 一句判定:Denoise | [📄](https://arxiv.org/abs/2605.28421) |
| [distillm](L1.md#p-distillm) | 自回归 LM 的白盒蒸馏有两大老毛病。 | 高 | **中等支撑(on-policy 数据效率 + 梯度稳定的方法学)**。判定依据:DistiLL | [📄](https://arxiv.org/abs/2402.03898) |
| [distillm2](L1.md#p-distillm2) | 以往蒸馏对两类数据用**同一个 loss**,忽略了"loss 形式 × 数据类型"的协同。这两类数据是:TGO(Teacher-Generated Output,教师生成的数据)和 SGO(Student-Generated Output,学生生成的数据)。 | 高 | **中等支撑(差异化处理 + 防退化的损失设计)**。判定依据:CALD 的"对不同来源数据施加 | [📄](https://arxiv.org/abs/2503.07067) |
| [gad](L1.md#p-gad) | 先说困境。当教师是只吐文本的闭源 API(如 GPT-5-Chat)时,学生拿不到 logits(模型输出的原始概率分数),做不了白盒蒸馏,也做不了 likelihood 蒸馏(对齐似然/概率的蒸馏)。更麻烦的是,连"在自己生成上学"(on-policy)都没法做——因为教师无法 | 高 | **竞品 / 可借组件**。 | [📄](https://arxiv.org/abs/2511.10643) |
| [gemma2](L1.md#p-gemma2) | Gemma 2 的 2B/9B 在**预训练**阶段用知识蒸馏来训小模型——用逐 token 软标签交叉熵 \(\min_{P_S}\sum_x -P_T(x\mid x_c)\log P_S(x\mid x_c)\) 替代标准 next-token 预测(即用教师给的概率分布当 | 中 | **弱支撑(立场层)/ 背景**。 | [📄](https://arxiv.org/abs/2408.00118) |
| [gkd](L1.md#p-gkd) | 自回归 LM(逐 token 往后生成的语言模型)的知识蒸馏有个老毛病——训练时学生拟合的是"固定数据集(真值或教师生成)"的序列,但推理时学生要从自己的部分输出自回归续写,会走到训练没见过的状态,早期 token 的误差会级联放大。这就是 train-inference 分布失 | 高 | **强支撑(OPD 源头)+ 可借组件**。 | [📄](https://arxiv.org/abs/2306.13649) |
| [glm45](L1.md#p-glm45) | GLM-4.5 是开源 MoE(总参 355B / 激活 32B,另有 106B 的 Air)**混合推理**模型(支持 thinking + direct 双模,即深思与快答两种模式)。**注意它是模型技术报告,不是单一方法论文。** | 高 | Agentic, Reasoning, and Coding (ARC) Foundation  | [📄](https://arxiv.org/abs/2508.06471) |
| [gopd](L1.md#p-gopd) | 本文先在理论上证明一件事——**标准 OPD 其实是一种 dense KL-约束 RL 的特例**。这里 dense 指奖励落在每个 token 上(而非只在结尾给一个总分)。它的隐藏约束是：reward 项与 KL 正则**永远 1:1 等权(β=1)**，且作为对照锚点的 r | 高 | **强对标(支撑+可借组件)**。判定依据： | [📄](https://arxiv.org/abs/2602.12125) |
| [hpd](L1.md#p-hpd) | 白盒知识蒸馏(white-box KD,即让小模型直接对齐大模型输出的概率分布,而不只是看大模型生成的文本)长期被三个互相纠缠的旋钮卡住: | 高 | **支撑 + 可借组件**。 | [📄](https://arxiv.org/abs/2604.20244) |
| [lightning_opd](L1.md#p-lightning_opd) | 标准 on-policy distillation(OPD,在线策略蒸馏)要在整个训练期常驻一个大 teacher server(教师服务器),对学生每步新采的 rollout(自生成轨迹)在线查 teacher log-prob(教师对数概率)。问题:GPU 被学生+教师共置而 | 高 | **支撑(OPD 工程基座)+ 可借组件**。判定:这是本项目 OPD 主轴的**直接、核心命中 | [📄](https://arxiv.org/abs/2604.13010) |
| [madopd](L1.md#p-madopd) | OPD(on-policy distillation,学生在自己生成的轨迹上、由 teacher 逐 token 监督)有两个老毛病: | 高 | **强支撑(直接对标 idea 的 OPD 骨架)**——MAD-OPD 本身就是 on-pol | [📄](https://arxiv.org/abs/2605.01347) |
| [minillm](L1.md#p-minillm) | 标准知识蒸馏(KD,即用大模型当老师教小模型)让小学生用 forward-KL 去"覆盖"大教师分布的所有模式。但学生容量小、覆盖不了,就会把概率塞到教师几乎没概率的"空白区(void region)",自由生成时吐垃圾。 | 高 | **强支撑(谱系源头)**。MiniLLM = "teacher 当脚手架、student 在自 | [📄](https://arxiv.org/abs/2306.08543) |
| [nemotron_cascade2](L1.md#p-nemotron_cascade2) | 在"按领域顺序逐段做 RL"的 **Cascade RL** 里,插入一个 **多领域 on-policy 蒸馏(MOPD)** 稳定化阶段: | 高 | **最强支撑(工业级同构实例)**。MOPD 几乎逐条命中本课题: | [📄](https://arxiv.org/abs/2603.19220) |
| [nemotron_nano2](L1.md#p-nemotron_nano2) | 造一个能在**单张 A10G(22GiB)上做 128k 推理**的紧凑推理模型。路线分三步: | 中 | **弱-中支撑(仅工业混用 SFT+RL+KD 的背景对照 + on-policy DPO 一支 | [📄](https://arxiv.org/abs/2508.14444) |
| [opd_blog](L1.md#p-opd_blog) | 本文系统推广了 **on-policy distillation**。它的核心是:学生在**自己采样出来的轨迹**上,以更强教师的**逐 token 分布(用 reverse KL 度量)** 作为**唯一的稠密监督**(环境不给任何 reward)。这样做兼顾了两件事—— | 高 | **强支撑(它是本课题 OPD 一侧的直接范式蓝本);是核心对标对象**。判定依据如下: | [📄](https://thinkingmachines.ai/blog/on-policy-distillation/) |
| [opd_survey](L1.md#p-opd_survey) | 这是本课题(MTP+OPD / TSRD)的核心背景综述。它把 On-Policy Distillation(OPD)统一刻画为"**学生在自己采样的轨迹上最小化 f-散度**"(Eq.8;f-散度是一族用来度量两个分布差异的量,KL 是其特例),并沿**三条设计轴**——优化什 | 高 | **强支撑(它是本课题的理论地图与设计指南);是定位 TSRD 的坐标系**。判定依据如下: | [📄](https://arxiv.org/abs/2604.00626) |
| [ophsd](L1.md#p-ophsd) | 先解释两个词。**harness(脚手架)**指推理时套在模型外面的编排流程,例如检索增强、先 plan 再 solve、先 draft 再 verify 等。**自蒸馏**指让模型自己当自己的老师、用 KL 散度去贴近老师分布。本文的做法是: | 高 | **最强结构同构(直接对标 TSRD)**。 | [📄](https://arxiv.org/abs/2605.08741) |
| [opsa](L1.md#p-opsa) | 先解释两个词。**safety tax(安全税)**指为提升安全而牺牲通用推理的代价;**特权上下文**指只在训练时给老师、不给学生的额外指令。本文做法是: | 中 | **领域专用支撑(巩固侧)+ 一个可借的"关键步定位"准则**。 | [📄](https://arxiv.org/abs/2605.15239) |
| [opsd](L1.md#p-opsd) | 这里"on-policy 采样"指让学生用自己当前策略生成轨迹(而非在固定数据上学)。本文做法是: | 高 | **强支撑(原型级)+ 有缺口**。 | [📄](https://arxiv.org/abs/2601.18734) |
| [pi_play](L1.md#p-pi_play) | 这里 **QCP(question construction path,问题构造路径)**指出题人从答案 / 证据反向把题目拼出来的全过程。本文做法是: | 高 | **强支撑**。 | [📄](https://arxiv.org/abs/2604.14054) |
| [prism](L1.md#p-prism) | 多模态后训练的标准做法是 SFT→RLVR,但 SFT 会引入**分布漂移**——所谓分布漂移,是指模型微调后输出分布偏离了目标(此处既没充分匹配示范分布,又丢掉了模型原有的强项)。多模态下这种漂移还**异质**:视觉感知错误和推理错误是两种不同模式,会在后续 RL 里各自复合放 | 高 | **中等支撑,偏"巩固/纠偏"一侧 + 流程位置启发**。 | [📄](https://arxiv.org/abs/2604.28123) |
| [qwen3](L1.md#p-qwen3) | Qwen3 是覆盖 0.6B–235B 的 dense + MoE 系列。 | 高 | **强支撑 OPD 主线 + 对"探索"提供厂级实证,但无脚手架/前瞻/记忆**。 | [📄](https://arxiv.org/abs/2505.09388) |
| [resd](L1.md#p-resd) | 先讲背景——在线策略自蒸馏(self-distillation,即让模型自己当老师)在 rare-success(几乎没成功轨迹)时会失灵。 | 高 | **强支撑 + 高度同源,既是直接竞品、又有可借组件**。 | [📄](https://arxiv.org/abs/2605.12741) |
| [rethink_opd](L1.md#p-rethink_opd) | OPD(让学生自己采样,再用 teacher 的逐 token log-prob 当作稠密 reward 来纠正)已被 Qwen3/MiMo/GLM-5 大规模采用,但它的训练动力学是黑箱、而且很脆——**更强的 teacher 有时完全教不动学生,更弱的 teacher 反而能 | 高 | **理论基石级支撑,直接为 TSRD 提供机制依据,同时也是一份边界警示**。 | [📄](https://arxiv.org/abs/2604.13016) |
| [rock_tokens](L1.md#p-rock_tokens) | RLVR 领域早已知道"少数关键 token 撑起了推理增益",但 OPD 的 token 级动力学一直没人研究。作者发现:OPD 训练表观上饱和之后,仍有约 6% 的词表、占输出 18% 频次的 token 持续保持高 KL loss——把它们命名为 **Rock Tokens | 中 | 已有工作发现推理被少数 "anchor" token 不成比例地支撑着(推理时一扰动它们就显著掉 | [📄](https://arxiv.org/abs/2605.09253) |
| [rosd](L1.md#p-rosd) | 先交代背景。在线策略自蒸馏(OPSD,即用一个和学生同源的"自教师"在学生自己采样的轨迹上提供稠密的、逐 token 的监督)本意是给训练补充细粒度信号。但标准做法有两个毛病: | 高 | **最强对标(直接竞品 + 高度可借)**。判定:ROSD 几乎就是本项目"教 student  | [📄](https://arxiv.org/abs/2605.28014) |
| [safesteer](L1.md#p-safesteer) | 先说现状。安全对齐常以牺牲通用能力(alignment tax,即"对齐税")为代价,现有方法靠混入海量通用数据、或借助辅助的 reward model 来做这种双目标权衡。 | 中 | **可借组件(方法同源),非主题竞品**。判定: | [📄](https://arxiv.org/abs/2606.02530) |
| [scope](L1.md#p-scope) | 标准 OPD 把 teacher 的稠密 token-level KL 监督**一视同仁**地施加到所有 rollout,忽视了不同信号的质量差异。SCOPE 按轨迹的正确性,把 on-policy rollout 路由成两条互补的路径: | 中 | **强对标(支撑 + 部分竞品)**。判定: | [📄](https://arxiv.org/abs/2604.10688) |
| [sdcl](L1.md#p-sdcl) | 目标是让模型**学新技能/新知识又不忘旧的**(持续学习)。两条已有路都有坑: | 高 | **强支撑(尤其"巩固/防遗忘"维度)+ 可借组件**。 | [📄](https://arxiv.org/abs/2601.19897) |
| [sod_stepwise](L1.md#p-sod_stepwise) | OPD(on-policy distillation)指"老师在学生自己生成的轨迹上,逐 token 给密集监督"。把它用到**小模型 agent 的工具集成推理(TIR,即一边推理一边调工具)**上会训练崩溃。原因链条是: | 高 | **最强支撑/直接对标(可借核心组件)**。 | [📄](https://arxiv.org/abs/2605.07725) |
| [tcod](L1.md#p-tcod) | on-policy 蒸馏(OPD,学生跑自己的轨迹、逐 token 受 teacher 监督)在单轮数学/QA 上很稳,但搬到**多轮 agent** 就崩——作者把这个现象命名为 **Trajectory-Level KL Instability**:训练中 KL(师生分布差距 |  | **强支撑 + 直接竞品/可借组件**。一句判定:TCOD 是本项目 idea 在"多轮 age | [📄](https://arxiv.org/abs/2604.24005) |
| [trl_v1](L1.md#p-trl_v1) | TRL 是 HuggingFace 的统一 LLM 后训练库(月下载约 300 万次),把以下方法都收进了一个库:SFT、Reward Modeling、偏好优化（DPO/KTO/ORPO/CPO/IPO）、RLVR（PPO/GRPO/RLOO/GSPO）、on-policy 蒸 | 中 | **基础设施（工具,非方法对标）**——一句判定:TRL 不提"探索/巩固"的新机制,但它是本项 |  |
| [unisd](L1.md#p-unisd) | 自蒸馏(同一模型从自身行为导监督,不靠外部强 teacher)在自回归 LLM 上难做——自生成轨迹是自由文本、正确性又跟任务相关,即便 rationale 看着合理也可能给出不稳/不可靠的监督。过去的方法各自只研究一个设计选择,谁起作用、如何交互都不清楚。 | 中 | **可借组件库（弱支撑）**——一句判定:UniSD 不直接做"探索/回轨",但它沉淀了一套"o | [📄](https://arxiv.org/abs/2605.06597) |
| [vla_opd](L1.md#p-vla_opd) | 机器人 VLA 后训练有个老大难—— | 中 | **支撑（域外平行证据）**——一句判定:VLA-OPD 是"teacher 当稀疏脚手架教 s | [📄](https://arxiv.org/abs/2603.26666) |
| [why_sd_degrade](L1.md#p-why_sd_degrade) | 自蒸馏指同一模型在富 context（这里 context=正确解）下当 teacher，给无 context 的 student 打稠密信号。它在很多域（化学/工具/代码）能让 response 变短、性能变好。但搬到**数学推理**上会反常——response 照样变短、性能 | 高 | **竞品/警示（强相关,反向支撑）**——一句判定:这是对"privileged-context | [📄](https://arxiv.org/abs/2603.24472) |

### L2 · 统一 SFT-RL · 奖励微调 (GFT 类)（20）

| 论文 | 一句话重点(TL;DR) | 相关性 | 对标 | 原文 |
|---|---|---|---|---|
| [amft](L2.md#p-amft) | AMFT 把 SFT 和 RL 揉进**单阶段单循环**，并把"模仿(SFT) vs 探索(RL) 的配比 \(\mu\in[0,1]\)"当成**可学习参数**。它用 meta-gradient(元梯度，即"对超参本身求梯度")以"最大化最终验证集表现"为元目标，**前瞻式** | 高 | **最强概念同构者**——AMFT 的"模仿(SFT/path-level) vs 探索(RL/ | [📄](https://arxiv.org/abs/2508.06944) |
| [asft](L2.md#p-asft) | DFT(用 `sg[π_θ]` 给交叉熵重加权的 SFT 变体；`sg` 是 stop-gradient)在推理域好用，但在知识域(如医疗)**不稳**。ASFT 用 **reward-weighted regression(RWR) 框架**做了三件事：证明 DFT 等价于一个 | 高 | **巩固/防遗忘侧的强相关者(竞品+可借组件)**——ASFT 的 "DFT 重加权(把学习集中 | [📄](https://arxiv.org/abs/2509.23753) |
| [beyond_loglik](L2.md#p-beyond_loglik) | SFT 默认用 NLL（即 \(-\log p\)，负对数似然,等价交叉熵）。一直以来大家把"SFT 泛化差"归咎于模仿学习这套范式。本文说不对——病根其实是 NLL 这个**默认目标**本身。理由是:NLL 只有在"从零训练一个小分类器"时才最优（Cox 1958 等）;而后训 | 中 | **可借组件（机制层启发）**。判定:本文与 OPD/TSRD **不是同一个设定**（它是固定 | [📄](https://arxiv.org/abs/2510.00526) |
| [chord](L2.md#p-chord) | 作者实测发现，在 instruct 模型上"SFT-then-RL"会走出一条 **"shift-readapt-overfit"（漂移-再适应-过拟合）** 的三阶段曲线，而且不一定比纯 RL 好【原文 §3.1, Fig.1/2】。 | 高 | **强支撑 + 高度可借（L2 主线代表作）**。 | [📄](https://arxiv.org/abs/2508.11408) |
| [dft_reweight](L2.md#p-dft_reweight) | 先把标准 SFT 的梯度用重要性采样改写成"策略梯度"的形式,改完会发现它隐含的奖励长这样——是一个**稀疏的指示函数(只有精确匹配专家 token 时才 =1),并且被 \(1/\pi_\theta\)(逆概率)加权**。问题就出在这个逆概率权重上:当模型给某个专家 token | 高 | **中等支撑(巩固侧的损失工具)+ 部分竞品(同为单行 SFT 改造)**。判定依据:DFT 是 | [📄](https://arxiv.org/abs/2508.05629) |
| [fest](L2.md#p-fest) | demonstration-guided RLVR(在 RL 采样失败时,拿 SFT 示范来补充外部知识)很有效,但代价是要数 K~50K 条精选的 SFT 数据,太贵。 | 中 | **可借组件 + 部分竞品**。一句判定:FEST 的"衰减权重防过拟合 + Pass@8 不塌 | [📄](https://arxiv.org/abs/2605.15012) |
| [hpt_upge](L2.md#p-hpt_upge) | 把 SFT 和各种 RL 后训练(PPO / GRPO / REINFORCE / CISPO / GSPO / SRFT / LUFFY)的策略梯度,统一写成同一个表达式——**Unified Policy Gradient Estimator(UPGE):`grad_Uni  | 高 | **强支撑 + 可借框架**。 | [📄](https://arxiv.org/abs/2509.04419) |
| [luffy](L2.md#p-luffy) | 纯 on-policy RL(GRPO)只能在模型"自己已会"的范围里放大,弱模型/难题很快撞天花板;纯 SFT/蒸馏又是死记硬背(behavior cloning),泛化差、熵塌。(术语:on-policy 指只用模型自己采样的轨迹来学;behavior cloning 指照搬 | 高 | **强支撑(且是最近邻竞品)**——LUFFY 几乎就是 idea 的一个已实现版本:teach | [📄](https://arxiv.org/abs/2504.14945) |
| [nft](L2.md#p-nft) | 人们普遍认为"从错误中自我反思"是 RL 的专利、监督学习(SL)只会记正样本。本文用一个 Bayes 分解反驳: | 高 | **中等支撑(巩固层的统一信号视角)+ 可借的"负样本=可监督信号"机制;非脚手架/回轨范式** | [📄](https://arxiv.org/abs/2505.18116) |
| [on_policy_sft](L2.md#p-on_policy_sft) | 做"高效推理"(目标是缩短 CoT、同时保住准确率)时,主流做法是把"长度惩罚"做成额外奖励塞进 GRPO,结果训练不稳、精度-效率的权衡也不理想。本文做的是**原理化简**——它先指出高效推理有两条 RLHF 任务所没有的性质: | 高 | **中-强支撑(巩固层:on-policy 数据是巩固之钥 + 低效 token 诊断接近"关键 | [📄](https://arxiv.org/abs/2602.13407) |
| [prefix_rft](L2.md#p-prefix_rft) | 先解释两个词。**SFT(监督微调)**模仿示范、泛化差;**RFT(强化微调)**奖励驱动、依赖初始策略、难突破上界。本文做法是: | 高 | **强支撑,且机制非常贴近**。 | [📄](https://arxiv.org/abs/2507.01679) |
| [psft](L2.md#p-psft) | 先说普通 SFT 的两个毛病。 | 高 | **强支撑"巩固"一侧 + 保探索的前置条件**。 | [📄](https://arxiv.org/abs/2508.17784) |
| [raft_reinforce_rej](L2.md#p-raft_reinforce_rej) | 先说大家默认的一个信念——"用上负样本的 RL(GRPO/PPO),一定比只用正样本的 SFT 类(拒绝采样 RAFT)强"。本文把 GRPO 从 reinforce-like(类 Reinforce)的视角拆开做对照实验,发现三点: | 高 | **竞品偏分析、可借组件强**。 | [📄](https://arxiv.org/abs/2504.11343) |
| [rl_plus](L2.md#p-rl_plus) | 纯 on-policy RLVR(如 GRPO)会"收窄"基座本来能解的问题集——表现为 pass@1 上升、但 pass@128 反而低于基座。作者称之为 capability boundary collapse(能力边界坍缩);成因是巨大的动作空间 + 稀疏奖励,逼着模型只做 |  | **强支撑(探索侧),直接同源于 path-recovery,可借组件多**。 | [📄](https://arxiv.org/abs/2508.00222) |
| [sed_sft](L2.md#p-sed_sft) | 标准 SFT 的交叉熵(CE)会把概率全压向唯一正确路径,造成 **mode collapse**(生成多样性塌缩),限制后续 RL 的探索效率。 | 中 | **弱-中支撑,落在"探索/保探索空间"这一侧,但不涉 teacher 蒸馏也不涉 path-r | [📄](https://arxiv.org/abs/2602.07464) |
| [srft](L2.md#p-srft) | 把 SFT 和 RL 塞进**单阶段的同一个 loss** 一起训(不再先 SFT 再 RL 两段),用"当前策略熵"做开关来自适应调两者权重。 | 高 | **支撑/可借组件**。 | [📄](https://arxiv.org/abs/2506.19767) |
| [sstoken](L2.md#p-sstoken) | SFT(监督微调,用标注数据继续训模型)里有个观察——"不是所有 token 都值得学"。现有的 token 选择方法(RHO-1、TokenCleaning)有两个痛点: | 中 | **可借组件(弱)**。一句判定:ssToken 是 SFT 侧的 token 选择,与本项目" | [📄](https://arxiv.org/abs/2510.18250) |
| [superrl](L2.md#p-superrl) | RLVR(用可验证奖励做 RL,如答案对错)在稀疏奖励下常因"一个 prompt 的所有 rollout(采样轨迹)都拿零奖励"而无梯度可学。SuperRL 在 RL 流程内做**实例级 reward-gated 切换**: | 中 | **支撑(部分同构)+ 竞品基线**。一句判定:SuperRL 的"全失败→回退离线示范监督"与 | [📄](https://arxiv.org/abs/2506.01096) |
| [trapo](L2.md#p-trapo) | 两阶段 SFT→RL 有个根本矛盾——SFT 把模型锁进刻板模仿,抑制 RL 的探索,还引发遗忘。TRAPO 在**每个训练实例内部**交织 SFT 与 RL: |  | **强支撑(论文 idea 层面)+ 对照基线,但代码不可直接复用**。一句判定:TRAPO 的 | [📄](https://arxiv.org/abs/2512.17636) |
| [vcore](L2.md#p-vcore) | 长 CoT SFT(在 teacher 蒸馏出的长推理迹上做监督微调)里,标准交叉熵对所有 token 等权——但很多 token 要么过易、要么过歧义,学习价值低;自动蒸馏出的长 CoT(动辄 >1k token)还常含幻觉/错位的 spurious token,等权会让噪声主 | 中 | **可借组件（中等,偏巩固侧的信用分配）**——一句判定:VCORE 不做探索/回轨,但它把"哪 | [📄](https://arxiv.org/abs/2510.27462) |

### L3 · RLVR · GRPO 家族推理训练（29）

| 论文 | 一句话重点(TL;DR) | 相关性 | 对标 | 原文 |
|---|---|---|---|---|
| [ampo](L3.md#p-ampo) | on-policy RLVR(GRPO) 的探索被困在模型自身知识边界内，难题持续全失败会导致稀疏奖励、训练不稳。AMPO 在 GRPO 之上做两件事：第一，**只在某 query 整组采样全部失败时**(guidance-on-demand)，才从一个**多教师池**里挑正确解 | 高 | **最强支撑/可直接借鉴者之一**——AMPO 的"on-demand 仅在自探索全失败时才注入 | [📄](https://arxiv.org/abs/2510.02227) |
| [bapo](L3.md#p-bapo) | 所谓 **off-policy RL**,就是采数据的"行为策略"和正在优化的"目标策略"不是同一个——可以复用旧经验（experience replay）、把超长轨迹切段续跑（partial rollout）。好处是样本效率高、对数据陈旧（staleness）有容忍度。坏处是: | 中 | **竞品/边缘相关**。判定:BAPO 是纯 RLVR 的裁剪稳定化,与 OPD/蒸馏**没有直 | [📄](https://arxiv.org/abs/2510.18927) |
| [ca_survey](L3.md#p-ca_survey) | 本文以"**信用分配(credit assignment, CA)**"为中心透镜重审 LLM RL。这里 CA 指的是——把稀疏的 outcome 奖励（只有最终对错一个分数）分配到具体的 token / segment / step / turn / agent 上。 | 中 |  | [📄](https://arxiv.org/abs/2604.09459) |
| [cbrl](L3.md#p-cbrl) | RLVR（可验证奖励的强化学习）的死穴是"探索低效"。模型遇到新推理模式或陌生领域时,几乎采不出一条正确的 rollout;而只要一组 rollout 全错,GRPO 的组内优势就会坍缩为 0、没有梯度、训练卡在 reward 平台期【原文 §1 L23-27, §5.1 L46 | 高 | **支撑（prompt 层面的极简对照）**。CBRL 可以理解为"教师脚手架（用 few-sh | [📄](https://arxiv.org/abs/2603.18953) |
| [cepo](L3.md#p-cepo) | GRPO 的做法是，一条正确轨迹里**所有 token 拿同一个正优势**，一条错误轨迹里所有 token 拿同一个负信号。这样梯度被浪费在 filler（连接词、格式、套话）上，又低估了真正决定成败的少数推理步——论文原话："The credit assignment prob | 高 | **可借组件 + 部分竞品**。CEPO 直击"token 级信用分配"——与本项目 L6（To | [📄](https://arxiv.org/abs/2605.19436) |
| [dapo](L3.md#p-dapo) | 朴素 GRPO 在 Qwen2.5-32B base 上跑数学 RL 只到 30 分(AIME 2024)。作者诊断出 4 个病灶——熵坍缩、无效梯度、长序列梯度失衡、超长样本奖励噪声——逐一开药(Clip-Higher / Dynamic Sampling / Token-Le | 高 | **支撑(探索侧)**。 | [📄](https://arxiv.org/abs/2503.14476) |
| [deepseek_r1](L3.md#p-deepseek_r1) | 本文分两层。 | 高 |  | [📄](https://arxiv.org/abs/2501.12948) |
| [deepseekmath](L3.md#p-deepseekmath) | 本文首次提出 **GRPO**。核心做法是用"同一道题采多个输出、组内互相比较得到的相对奖励"来替代 PPO 的价值网络(critic),从而省掉一个与策略同规模的 critic,大幅降低显存和算力。同时它给出一个统一梯度范式,把 SFT / RFT / Online-RFT / | 高 | **支撑(探索侧算法底座) + 一个关键警示** —— 一句判定:GRPO 是本项目几乎所有 o | [📄](https://arxiv.org/abs/2402.03300) |
| [dr_grpo](L3.md#p-dr_grpo) | 批判性审视 R1-Zero 范式(不经 SFT、直接对 base 模型大规模 RL)的两大成分。 | 高 | **中-强支撑(机理诊断层)+ 直接可借的训练信号工具**。判定依据有两条: | [📄](https://arxiv.org/abs/2503.20783) |
| [entropy_mechanism](L3.md#p-entropy_mechanism) | 在可验证奖励 RL(RLVR)训练推理模型时,策略熵会在头 200 步就急剧坍缩到接近 0;模型随之变得过度自信、再也探索不动,性能也跟着触顶。 | 高 | **支撑(机理层)+ 可借组件**。一句判定:本文为"探索"这一支提供了最直接的机理工具——它定 | [📄](https://arxiv.org/abs/2505.22617) |
| [gmpo](L3.md#p-gmpo) | GRPO 优化的是 token 级奖励的**算术平均**,这对离群的重要性采样比 \(\rho_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\ | 中 | **可借组件(RL 优化层),非直接对标**。 | [📄](https://arxiv.org/abs/2507.20673) |
| [gspo](L3.md#p-gspo) | 先说病灶。GRPO 在 **token 级**做重要性采样校正是「病态(ill-posed)」的。原因：重要性采样的原理要求在行为分布上对**多样本(N≫1)** 取平均，权重 πtar/πbeh 才能完成「从行为分布校正到目标分布」；但 GRPO 在**每个 token 位置* | 中 | **可借组件/竞品(RL 优化层)，非直接对标**。判定依据： | [📄](https://arxiv.org/abs/2507.18071) |
| [hapo](L3.md#p-hapo) | 现有 RLVR(GRPO/DAPO)对所有 token **一视同仁**地优化，这违背了语言生成的异质本质——关键推理决策 token 与常规格式 token 的角色完全不同。已有用熵的方法只把熵当成**离散过滤器或事后的 bonus**，核心优化机制并没变。 | 中 | **在「探索/选路」一侧相关、有可借组件；「巩固」一侧无贡献**。判定依据： | [📄](https://arxiv.org/abs/2509.16591) |
| [holderpo](L3.md#p-holderpo) | GRPO 系算法里有这么一步——把序列内每个 token 的重要性比 r_{i,t}=πθ/πθold **聚合成一个序列级标量**。过去这步用的都是**固定算子**：GRPO 用算术平均(对应 p=1)、GMPO/GSPO 用几何平均(对应 p→0)。 | 中 | 【原文】(1) 调度区间 endpoint 任务/base 相关、非通用(§4.4);(2) 方 | [📄](https://arxiv.org/abs/2605.12058) |
| [limit_rlvr](L3.md#p-limit_rlvr) | 本文不提新方法,而是用一把"尺子"去量 RLVR(可验证奖励 RL)到底有没有让模型获得 base 模型没有的新推理能力。 | 高 | **间接支撑(方向校准 + 度量工具)**。 | [📄](https://arxiv.org/abs/2504.13837) |
| [lite_ppo](L3.md#p-lite_ppo) | 2025 年 RL4LLM(给 LLM 做 RL)的各路 trick(归一化、Clip、loss 聚合、overlong filtering 超长过滤)互相打架: | 中 | **弱/工具性支撑**——不直接做探索-巩固,但给本项目两个可借的工程默认: | [📄](https://arxiv.org/abs/2508.08221) |
| [lp_reg](L3.md#p-lp_reg) | RLVR 训久了会"熵崩溃"——策略熵快速衰减、探索消失、性能进入平台期。(术语:熵衡量策略输出的随机性/多样性,熵塌缩意味着模型越来越只走一条固定路径。) | 中 | **中等支撑(直接服务"探索"侧的保护)**——Lp-Reg 精准对应 idea 中"巩固时别把 | [📄](https://arxiv.org/abs/2510.03222) |
| [magistral](L3.md#p-magistral) | Mistral 自底向上、**不依赖任何外部蒸馏轨迹**,只用自家模型 + 自研异步在线 RL 基础设施,纯 RLVR(改造版 GRPO + 四维奖励整形)把 Mistral Medium 3 训成了推理模型 Magistral Medium(AIME-24 pass@1 从 2 | 中 | 相对标准 GRPO,这里**没有 KL 项**、上下裁剪不对称(\(\varepsilon_{\ | [📄](https://arxiv.org/abs/2506.10910) |
| [minimax_m1](L3.md#p-minimax_m1) | 首个开源权重的**大规模混合注意力推理模型**。这里"混合注意力"指 MoE(混合专家)叠加 lightning attention(一种线性注意力变体)。规模为 456B 总参 / 45.9B 激活,原生 1M 上下文;生成 10 万 token 时 FLOPs 仅 DeepS | 高 | CISPO 的"用 sg(IS) 当系数、保全梯度"形式与本课题关注的"forward-hard | [📄](https://arxiv.org/abs/2506.13585) |
| [negative_reinforce](L3.md#p-negative_reinforce) | 把 RLVR(可验证奖励 RL,即用对/错二元信号训练)的学习信号(对 +1 / 错 -1)解析地拆成两条独立范式: | 高 | **强支撑 + 直接可借组件**。NSR 三性质几乎是本课题"巩固/回轨"想要的: | [📄](https://arxiv.org/abs/2506.01347) |
| [one_shot_rlvr](L3.md#p-one_shot_rlvr) | **只用一条训练样本**做 RLVR(用 GRPO 或 PPO),就能把 Qwen2.5-Math-1.5B 在 MATH500 上从 36.0% 提到 73.6%、6 个数学基准的均值从 17.6% 提到 35.7%。这几乎追平了包含该样本的 1.2k DSR 子集(它是 73 | 高 | **中-强支撑(机理层:base 本就蕴含可激发的能力 + 探索本身有价值),外加两条警示;但它 | [📄](https://arxiv.org/abs/2504.20571) |
| [orz](L3.md#p-orz) | 这里 "scale up reasoning RL" 指随训练步增加,性能与响应长度持续同步上涨而不饱和。本文做法是: | 低 | **竞品/背景,非直接组件**。 | [📄](https://arxiv.org/abs/2503.24290) |
| [prorl](L3.md#p-prorl) | 学界争论的核心是——RL 到底是发现新的推理能力,还是只放大 base 模型已潜在的高奖励输出?(后一种观点由 Yue/Dang/Zhao 等人用 pass@k 来论证;pass@k 指采样 k 次里至少答对一次的概率,k 越大越能看出模型潜在的解题上限。) | 高 | **支撑"探索"一侧(探索预算/边界扩展),但与"教师脚手架"范式正交**。 | [📄](https://arxiv.org/abs/2505.24864) |
| [revisit_entropy](L3.md#p-revisit_entropy) | 用 RLVR(可验证奖励 RL,即用"答案对不对"这种可机械判定的信号做 RL)训练 LLM 时,策略熵常会"坍缩"——概率质量集中到少数 token、模型丧失探索,从而过早收敛。本文做了一次系统性的熵体检,四件事: | 中 | **间接支撑(探索侧)+ 可借诊断,非直接竞品**。 | [📄](https://arxiv.org/abs/2511.05993) |
| [rl_survey_lrm](L3.md#p-rl_survey_lrm) | 本文系统综述了 DeepSeek-R1 以来"RL for 大型推理模型(LRM)"这一范式,用一张总图(Fig.1)把它组织成五层——**基础组件(§3)/ 五对开放争议(§4)/ 训练资源(§5)/ 下游应用(§6)/ 未来方向(§7)**。核心论断:**RLVR(可验证奖励 | 中 | **坐标系/背景支撑,而非方法对标**。判定:它为本项目提供的是文献定位—— | [📄](https://arxiv.org/abs/2509.08827) |
| [scaf_grpo](L3.md#p-scaf_grpo) | RLVR(可验证奖励的 RL,如 DeepSeek-R1 范式)有个"**学习悬崖**"问题,链条是: | 高 | **强支撑(尤其"探索 / 选路"维度)+ 可借组件 / 部分竞品**。判定: | [📄](https://arxiv.org/abs/2510.19807) |
| [simplerl_zoo](L3.md#p-simplerl_zoo) | DeepSeek-R1 证明从 base 模型直接做规则奖励的纯 RL("zero RL")能自发涌现长 CoT 与自反思("aha moment"),但社区复现几乎都集中在 Qwen2.5(它本身已经很强,不代表"野生"base)。 | 高 | **中支撑,提供"探索质量如何度量与保护"的方法论与多条警示**。 | [📄](https://arxiv.org/abs/2503.18892) |
| [skywork_or1](L3.md#p-skywork_or1) | 本文面向"已蒸馏的长 CoT 模型"(指 DeepSeek-R1-Distill 系列——它们已经能输出很长的逐步推理)提出一套**高效可扩展的 RL 配方**。配方建在改造版 GRPO 上,命名 **MAGIC**(= Multi-stage Adaptive entropy  | 高 | **支撑/可借组件(探索侧)**。 | [📄](https://arxiv.org/abs/2505.22312) |
| [spurious_rewards](L3.md#p-spurious_rewards) | 在 Qwen2.5-Math 上,即使用**虚假奖励**(随机/格式/错误标签——与正确答案零相关、甚至负相关)做 RLVR(GRPO),也能大幅涨分:随机奖励 MATH-500 +21.4%,接近真值的 +29.1%【原文 abstract+§2 Fig.2】。 | 高 | **竞品/警示(对核心追问直接相关)**。 | [📄](https://arxiv.org/abs/2506.10947) |

### L4 · Agent / 工具 / 多轮自进化（17）

| 论文 | 一句话重点(TL;DR) | 相关性 | 对标 | 原文 |
|---|---|---|---|---|
| [behavior_priming](L4.md#p-behavior_priming) | 搜索 agent（指多轮检索、边搜边答的 LLM agent）要靠 RL 训得好,前提是 base 模型本身已经具备某些推理行为。本文分两步。第一步,用一条 **LLM pipeline 从"强模型成功 vs 弱模型失败"的轨迹里,提炼出四种有益行为**:Information  | 高 | **强支撑（核心对标论文之一）**。判定:本文是 TSRD"teacher 当稀疏脚手架 → s | [📄](https://arxiv.org/abs/2510.06534) |
| [chain_of_agents](L4.md#p-chain_of_agents) | 核心目标是把"多智能体系统（MAS，multi-agent system）的协作能力"压进**单一模型**。MAS 性能强，但有三个老毛病——靠人工写 prompt/workflow、agent 之间通信冗余、不能用数据驱动来学。而 TIR（工具集成推理，tool-integra | 中 | **弱支撑/部分竞品（L4 自进化 Agent 维度），但蒸馏机制不同源**。 | [📄](https://arxiv.org/abs/2508.13167) |
| [deepdive](L4.md#p-deepdive) | 开源 LLM 当 deep-search agent(即在数百个网页里找"难找信息"的浏览智能体)表现很差,原因是两条——缺够难的数据,且缺多轮 RL。DeepDive 用两招应对: | 中 | **弱相关(纯多轮工具 RL,作长 horizon agent RL 对照)** —— 一句判定 | [📄](https://arxiv.org/abs/2509.10446) |
| [distill_agent_tools](L4.md#p-distill_agent_tools) | 本文不只蒸馏教师的"推理(CoT)",而是把教师 **agent 完整的"think + act(检索 / 代码工具)"任务求解行为**蒸馏进小模型(sLM)。痛点很具体:纯 CoT 蒸馏出来的小模型,在需要罕见事实知识或精确计算时容易幻觉 / 算错(经典例子:"\$100 投资 | 高 | **强对标(L4 agent 蒸馏的直接对照基线)+ 多个可借组件,但范式相反(离线 SFT v | [📄](https://arxiv.org/abs/2505.17612) |
| [gigpo](L4.md#p-gigpo) | GRPO 在单轮数学 / 代码上很强(奖励即时、信用分配简单、critic-free 即不用价值网络)。但搬到长 horizon 多轮 agent(horizon = 一条任务的步数;如 ALFWorld 一集可达 50 步、20k+ token、奖励稀疏 / 延迟)就出问题:它 | 中 | **竞品 / 可借组件(信用分配侧)**。 | [📄](https://arxiv.org/abs/2505.10978) |
| [icrl](L4.md#p-icrl) | 同一个 backbone(主干模型)套两套 role prompt(角色提示词),同时扮演 solver(解题手)和 critic(批评家——解题失败后写一段自然语言批评指出问题),两个角色联合做 RL。目标是把"有 critique(批评)才能做对"内化成"无 critique | 高 | **强支撑(形式同构)**。判定:ICRL 的 "critique 引导 solver 走出失败 | [📄](https://arxiv.org/abs/2605.15224) |
| [latent_agents](L4.md#p-latent_agents) | 把"多个 agent 多轮辩论(multi-agent debate, MAD)"这套昂贵的外部推理过程,内化进**单个 LLM**(方法叫 IMAD)。两阶段微调: | 中 | **部分支撑(范式同构)+ 弱竞品**。判定:IMAD 的 "SFT 学外部过程结构 → RL  | [📄](https://arxiv.org/abs/2604.24881) |
| [meow_tea_taro](L4.md#p-meow_tea_taro) | 不是新算法,而是一份**多轮 agentic RL 的受控实证配方**。把设计空间拆成 environment / reward / policy 三支柱系统消融,在 TextWorld/ALFWorld(具身文字)+ SWE-Gym(软件工程)上得出一句可操作的配方: | 中 | **中等支撑(多轮/agent 侧的对标,偏方法学约束)**。 | [📄](https://arxiv.org/abs/2510.01132) |
| [open_agentrl](L4.md#p-open_agentrl) | 这是一个"完全动态"的闭环 RL 系统,**同时进化环境、策略、生成式奖励模型三者**。具体是: | 中 | **间接支撑(在探索这一侧)+ 一个可借的环境调度组件**。 | [📄](https://arxiv.org/abs/2602.02488) |
| [openclaw_rl](L4.md#p-openclaw_rl) | 核心做法是,把 agent 每次交互产生的 "next-state 信号"(用户回复 / 工具输出 / 终端·GUI 状态变化)当作一个在线实时的学习源回收回来,从中抽出两类互补信号: | 高 | **强支撑(它把 OPD 用进了在线 agentic,而且专治失配稳定性)**。 | [📄](https://arxiv.org/abs/2603.10165) |
| [rstar2](L4.md#p-rstar2) | 目标是让 14B 模型"想得更聪明而非更长"。靠三件套: | 高 | **部分支撑(可借采样思想),非直接同构**。判定: | [📄](https://arxiv.org/abs/2508.20722) |
| [score](L4.md#p-score) | 小学生 agent(7B)做不动大 teacher(72B)整条轨迹的模仿——任何一步走错就会被推到 OOD(分布外),误差按 \(O(H^2)\) 滚雪球(\(H\) 是轨迹步数)。 | 高 | **强支撑 + 高度同构**。 | [📄](https://arxiv.org/abs/2509.14257) |
| [search_dont_guess](L4.md#p-search_dont_guess) | 小模型(SLM,通常 <4B)做"带搜索的 agent"时有个反直觉现象——它**比大模型更少搜索、更爱拿内部知识瞎猜**(即 parametric hallucination,凭参数里记的东西编答案)。 | 中 | **弱-中支撑,主要价值在"巩固有效行为先验"这一侧**。 | [📄](https://arxiv.org/abs/2604.04651) |
| [search_r1](L4.md#p-search_r1) | 把"调用搜索引擎"嵌进 RL 训练循环——让 LLM 在 `<think>` 里推理、需要时发 `<search>` query、把检索结果包进 `<information>` 拼回上下文继续推理、最后 `<answer>` 给答案。 | 高 | **底座/对照型支撑,主要在 L4 的"自主探索有效工具调用"这一侧**。 | [📄](https://arxiv.org/abs/2503.09516) |
| [spear](L4.md#p-spear) | 在多轮 agent RL 里,"机械地最大化策略熵"促探索很脆弱(环境反馈里的低概率 token 会累积,导致分布漂移 → mode collapse 或 runaway divergence)。SPEAR 的方案是"**课程化自模仿学习(SIL)+ 内在奖励塑形**": | 高 | **支撑/竞品(思路相通,可借组件)**。 | [📄](https://arxiv.org/abs/2509.22601) |
| [sweet_rl](L4.md#p-sweet_rl) | 在多轮 agent 任务里,把单轮 RLHF(RAFT/DPO/PPO)直接外推会做不好"跨轮信用分配"(判断哪一轮动作对最终结果有功/有过);学一个 value head(预测状态价值的小网络)又泛化差。SWEET-RL 的做法: | 中 | **支撑(思路同构)+ 可借资产**。一句判定:SWEET-RL 的"非对称 critic 用  | [📄](https://arxiv.org/abs/2503.15478) |
| [webagent_r1](L4.md#p-webagent_r1) | 本文把单轮数学 RL 里成功的 GRPO 思想搬到"多轮 web agent"场景——agent 在真实 web 沙箱里点击/输入/滚动地完成任务,只拿一个终态二值奖励(成没成)。方法用端到端多轮 on-policy RL（M-GRPO + 动态上下文压缩 + 并行轨迹 roll | 中 | **竞品/参照（无 teacher 的多轮 agent RL）**——一句判定:这是"纯 on- | [📄](https://arxiv.org/abs/2505.16421) |

### L5 · 记忆/技能库 & 持续学习 · 防遗忘（1）

| 论文 | 一句话重点(TL;DR) | 相关性 | 对标 | 原文 |
|---|---|---|---|---|
| [eaft](L5.md#p-eaft) | SFT 损害通用能力,主因不是"训练流程"本身,而是一类特定 token——**模型自己很笃定(低熵)、标签却逼它输出另一个 token(对该标签给的概率很低)**。论文给它起名叫"Confident Conflict(自信冲突)"。 | 高 | **强支撑(巩固/防遗忘侧)+ 弱竞品(探索侧无)**。判定依据:本课题的"巩固"= 把有效行为 | [📄](https://arxiv.org/abs/2601.02151) |

### L6 · 思维链 / Token 级信用 & 前瞻 (MTP)（7）

| 论文 | 一句话重点(TL;DR) | 相关性 | 对标 | 原文 |
|---|---|---|---|---|
| [adaspec](L6.md#p-adaspec) | 投机解码(speculative decoding)用一个小的 draft 模型先快速生成、再由大模型验证。给这个 draft 做蒸馏时，常规做法是"所有 token 一视同仁地对齐大模型"。AdaSPEC 换了个思路：先训一个与 draft 同初始化的"参考模型"当难度探针，用 | 中 | **可借组件**——"用参考模型损失作基准定义可学性(\(\Delta L\))，只在学生够得着 | [📄](https://arxiv.org/abs/2510.19779) |
| [lightreasoner](L6.md#p-lightreasoner) | 反直觉地用"小模型(amateur,业余者)教大模型(expert,专家)"。 | 高 | **部分支撑(关键步定位)+ 方向反转的竞品**。判定:LightReasoner 的"**用分 | [📄](https://arxiv.org/abs/2510.07962) |
| [llm_future_mtp](L6.md#p-llm_future_mtp) | 这是一个纯**推理加速(speculative)**的工作,既不提升推理质量、也不是蒸馏。 | 高 | **弱支撑/工具性**。它为 idea 中"MTP 当前瞻探针"提供了**机制基础与实证**(模 | [📄](https://arxiv.org/abs/2507.11851) |
| [rho1](L6.md#p-rho1) | 预训练的惯常做法是对**所有** token 一视同仁地施加 next-token 损失(预测下一个词的交叉熵)。本文主张"语料里并非所有 token 都同等重要",提出 Selective Language Modeling(SLM,选择性语言建模)。三步: | 中 | **思想先验级支撑(token 异质性 + token 信用),非方法竞品**。 | [📄](https://arxiv.org/abs/2404.07965) |
| [segment_attrib](L6.md#p-segment_attrib) | 长 CoT 里只有一小部分 token 真正帮到答案,大量是重复/截断/废话;直接在全 CoT 上 SFT,会让模型把这些冗余一起学坏。 | 中 | **中-高支撑,直击"巩固时该固化哪些 token/段"这一信用分配问题**。 | [📄](https://arxiv.org/abs/2602.00425) |
| [srgen](L6.md#p-srgen) | 这是一个**零训练的测试时方法**。解码时它的流程是: | 中 | **正交、松散类比(可借测试时机制)**。 | [📄](https://arxiv.org/abs/2510.02919) |
| [tip](L6.md#p-tip) | OPD(on-policy distillation)里"哪些 token 有学习信号"由两根轴决定: |  | 把它当"大幅提升 OPD"——数学上多为 +1~3 点。低估之处:Q3 盲区的诊断 + Deep | [📄](https://arxiv.org/abs/2604.14084) |

scaf_grpo | Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning | HKUST / CUHK / HKU(Xichen Zhang*, Sitong Wu*, …, Jiaya Jia 通讯;*共同一作) | 2026-02·ICLR 2026·v2(2026-02-28, cs.CL) | 主题线 L3(RLVR/GRPO 家族)+ L1(教师引导)·相关性 高

**原始论文**:https://arxiv.org/abs/2510.19807 （arXiv:2510.19807v2;ICLR 2026 接收）

## 一眼看懂
- 🟦 TL;DR:RLVR(可验证奖励的 RL,如 DeepSeek-R1 范式)有个"**学习悬崖**"——题目远超当前能力时所有 rollout 全错→GRPO 里这组 advantage 全归零→梯度消失→这些难题对训练"隐形",卡在长尾。Scaf-GRPO 的招:只在**学习真停滞**时,按"知识→规划→解答"三层、由抽象到具体**增量注入 in-prompt 提示**,直到**当前策略 π_θ 自己**在带提示的 prompt 上采出一条正确轨迹 o*_h,用它**替换 rollout buffer 里某条失败轨迹**,从而恢复非零 advantage、让学习重启。因为成功轨迹仍是当前策略采的、重要性比率也对"带提示的 prompt"算,所以**全程 on-policy**(不是 teacher 续写 golden 前缀那种 off-policy)。
- 最巧的一步:**抽掉"用 π_θ 自采样的 o*_h 替换失败轨迹、且比率对 q⊕h* 计算"这一步,方法就垮成 off-policy 前缀法**。整个方法不改 GRPO 损失形式(§3.2 明说"does not alter the mathematical form…only modifies the data"),真正的魔法是"提示只进 prompt、答案仍由学生自己产、比率对带提示 prompt 算"——这同时保住了 on-policy 一致性(避免前缀法的师生分布失配)和探索自主性(提示是路标不是铁轨)。为什么是它:Eq.4 那个分段比率(o*_h 的比率用 q⊕h* 而非 q)是论文反复强调的"on-policy integrity"所在,附录 F.5 还专门证了用 π_θ(·|q)/π_θold(·|q⊕h*) 这种失配比率会训崩。

## 为什么做
- 研究背景:RLVR 用稀疏 binary 结果奖励即可让模型自主习得推理策略,免逐步人工标注(§1/§2)。后续两条线:(a) 稳定性/去偏(Dr.GRPO、DAPO);(b) 更稠密奖励(长度惩罚抑 overthinking、token 级反馈)。
- 解决的具体痛点:**学习悬崖**——超能力题持续全错→零奖励→GRPO advantage = (R−μ_G)/(σ_G+ε) 在全零时 μ_G=σ_G=0 → Â_i=0 → 梯度消失,难题"隐形"形成长尾瓶颈(§1 + Fig.2 实证 Qwen2.5-Math-1.5B 上 vanilla GRPO 在零奖励题上停滞)。
- 相关工作 & 各自不足:主流应对是 **off-policy 教师引导/前缀续写**——LUFFY(整条专家轨迹与多条 rollout 混批)、Huang 等(cosine-decay 调前缀长度)、Zhang 等(多级不同长度 hint,从多起点探索)。两弊:(1) teacher 前缀与 student 后缀**分布失配**,要 policy-shaping / 混合 SFT-RL 等补丁,引入偏置+不稳;(2) "**on-rails**"把模型逼上预定路径,扼杀对替代/更优策略的探索(§1/§2)。
- 动机链:RLVR 好但难题学不动(悬崖)→ 前缀续写能给正奖励但失配+扼杀探索→ 所以借教育学**脚手架**(Berk & Winsler 1995):仅在停滞时给"会随能力提升撤除的、临时最小分层"引导,且让"题+提示"在**同一统一策略**下处理(避失配)、提示只指方向不定路径(保探索)。为什么不用更简单的现成做法:前缀法虽现成但破坏 on-policy 一致性、需复杂补丁(§3.2 + 附录 F.5 证失配比率训崩)。
- 与最近邻工作的Δ:最像 LUFFY/多级 hint 前缀法。**关键差一点**:提示**只进 prompt(in-prompt),答案完全由当前策略自采**,且重要性比率对"带提示 prompt"计算 → 严格 on-policy;而前缀法是把 teacher 的 token 直接拼进轨迹。为什么有用:既给了"学习悬崖"上缺的正奖励信号,又指向"模型自己够得着的最高效推理路径",还不引入失配偏置(主结果相对 LUFFY +9.2%、GPQA OOD 与 LUFFY 持平甚至超出,§4.2)。

## 怎么做 + 靠不靠谱
- 方法流水线(§3.2 + Fig.3):
  1. 输入:query q;预定义三层提示 H={H_knowledge, H_planning, H_solution}(由 DeepSeek-R1 基于 ground-truth 解题步骤生成,§4.1)。
  2. **Phase 1 — 诊断 true-hard(guidance exemption period)**:训练前 **15% 步数**纯 on-policy 探索;监控"零奖励 query 被解决的速率",该速率停滞后,仍持续失败的题判 **true-hard**(区别于格式/早期技能不熟的 **pseudo-hard**——多训即可解),才成引导候选。这保证"提示只留给真悬崖"。
  3. 标准 GRPO 步:对 q 采一组 N 条轨迹 G;若 ≥1 条对 → 正常 GRPO(不干预,Eq.5 J_Scaf ≡ J_GRPO)。
  4. **Phase 2 — 全错时触发分层引导**:确定性搜索,从最抽象层(Knowledge)起、层内**增量给提示**,一旦 π_θ 在 q⊕h 上采出正确解即终止 → 得**最小有效提示 h***(附录 D.1 给算法)。
  5. **on-policy batch 增强**:用 o*_h∼π_θ(·|q⊕h*) **替换**一条随机失败轨迹 o_j → G_final;advantage 在 G_final 上重算(Eq.2),恢复 μ_G_final>0、非零 Â';损失仍是 GRPO clipped surrogate(Eq.3),但比率分段(Eq.4):普通轨迹比率对 q 算,o*_h 的比率对 q⊕h* 算。
  6. KL penalty=0(最大化探索);输出:能持续从 true-hard 题学习的策略。
- 逐组件必要性(消融在 Table 2,Qwen2.5-Math-7B,基线 No Guidance=vanilla GRPO 50.9 vs 45.2):
  - **guidance exemption period(15%)**:防"一开始就给提示→产生 hint 依赖"。消融充分——附录 G.1:从头就 scaffold 掉 **9.2%**;且 10%–40% 区间稳(plateau 49.5–50.9%),故选 15%。
  - **三层 KPS 分层**:每层有独特功能。消融充分——逐层删:w/o Knowledge(P→S)49.2、w/o Planning(K→S)48.6、**w/o Solution(K→P)48.0 降幅最大(−5.7%)**——证三层互补非冗余(抽象层促高阶推理,具体层是必要兜底)。
  - **progressive/hierarchical(抽象→具体)vs Solution-Only**:Solution-Only(直接给最具体提示)48.4(−4.9%)——证"先逼模型啃高阶推理"促更可迁移技能。
  - **incremental chunking(层内增量给)vs Full Hint(一次给整层)**:Full Hint 47.7(−6.3%,Table 2 降幅最大的一项)——证最小增量干预对保自主性、防过度依赖最关键。
  - **hint 质量**:有分析——§4.3 + 附录 H,用 LLM-as-Judge(accuracy/minimality/clarity/coherence),DeepSeek-R1 生成的 hint 质量高于 Qwen2.5-72B,带来学生 +4% 相对增益。
  - **数据过滤(Too Easy 丢/Too Hard 留/Potentially Solvable 50% 子采样)**:有消融——Table 3:filtered 数据对 vanilla GRPO 仅 +0.5% 而对 Scaf-GRPO +6.0%(7B);附录 G.2/Table 12:此过滤比全量 +10.2%。说明"难课程"只在配合 Scaf-GRPO 这种能消化难题的框架时才有效。
- 关键机制/公式(直觉):
  - **为何仍是 on-policy**(§3.2 "on-policy integrity"):off-policy 前缀法的比率是 π_θ/π_ϕ(从别的策略 ϕ 导入轨迹)需高方差重要性校正且失配;Scaf-GRPO 的 o*_h 直接从当前 π_θ 采,比率只是"在改过的输入(q⊕h*)上的标准 on-policy 比率"——把当前与旧策略都 condition 在**同一带提示 prompt** 上,信号更稳。附录 F.5 实证:用 π_θ(·|q)/π_θold(·|q⊕h*) 这种"分子分母 condition 不一致"的失配比率会训崩。
  - **"最小有效引导促内化"**:用尽可能抽象的提示解出题→逼模型内化推理技能而非记解法(§1/§4 Internalizing skills);Fig.5 展示一道题"靠 Solution 提示对→靠 Planning 对→靠 Knowledge 对→无提示自解"的"毕业"轨迹。
- 实验与证据:
  - 数据集/设置:训练数据派生自 **DeepScaleR-Preview-Dataset**,按每模型初始能力动态过滤;三层 hint 由 prompt DeepSeek-R1 基于 ground-truth 步骤生成。评测 7 基准(pass@1 greedy):AIME24/AIME25/AMC/Minerva/MATH-500/OlympiadBench/GaoKao2023en;OOD 用 **GPQA-Diamond**。模型 5 个:Qwen2.5-Math-7B、Qwen2.5-Math-1.5B、Qwen2.5-7B、Llama-3.2-3B-Instruct(非 Qwen)、DeepSeek-R1-Distill-Qwen-1.5B(长 CoT)。框架 **verl**;训 **10 epochs**;max response 2048(长 CoT 模型 8192);**KL penalty=0**;报 best checkpoint。vanilla GRPO 用本文数据+超参训,LUFFY 用本文数据+其原参,Simple-RL/Oat-Zero 用公开权重。
  - 支撑核心主张的关键数字:主结果(Qwen2.5-Math-7B,Table 1)整体相对 vanilla GRPO **+12.6%**(50.9 vs 45.2)、相对 LUFFY **+9.2%**;AIME24 pass@1 30.0→**43.3**(相对 +44.3%);相对 Simple-RL +19.5%、Oat-Zero +9.5%。**5 个模型全有正增益**(Llama-3.2-3B +10.3%、长 CoT 1.5B +5.9%)。OOD(Table 6,GPQA-Diamond):7B 相对 vanilla GRPO **+15.5%**(与 LUFFY 持平)、Qwen2.5-7B-Base **+7.5%**(超 LUFFY)。skill-graduation(Table 4)从"hint 依赖→自解"事件数显著多于 vanilla(1.5B +137.8%、7B +11.3%、Llama +70.9%)。
  - baseline 公平吗:vanilla GRPO 用同数据同超参(可比);LUFFY 用同数据其原参(基本可比);Simple-RL/Oat-Zero 用公开权重(口径略不同,作者标注为"contextualize against optimized public benchmarks")。整体相对可比。
  - "看着强但没回答核心问题":主结果多报 **best checkpoint** 而非固定步数(跨方法早停一致性需信任作者实现,§限制);消融只到**层级粒度**,**没做"训练题 vs 持出题"的记忆探针**直接排除"学生在 hint 下记住了该题"〔推断:Fig.5 是定性单例,Table 4 graduation 是间接证据,均非记忆探针〕。
- 假设与失效边界:
  - 显式假设【原文】:有可验证答案(verifier,RLVR 前提);有一个**更强教师(DeepSeek-R1)**来生成分层 hint(§4.1);题目可被三层提示"够到"(否则搜索到 Solution 层仍失败 → fallback,论文未细述全失败比例)。
  - 隐式假设【推断】:三层 hint 的语义粒度划分对不同领域普适(论文只在数学验证);"零奖励解决速率停滞"能可靠区分 true/pseudo-hard(阈值/窗口是经验设定)。
  - 何时失效:【原文】无 verifier 的开放任务不适用(RLVR 框架限制);hint 质量受教师上限约束(§4.3 hint 质量影响显著)。【推断】非数学域(代码/通用推理)未验证(GPQA 仅作 OOD 评测、非训练域);exemption 比例/KL=0/温度等鲁棒性除 G.1 外未系统消融。
- 祛魅总结【推断】:
  - 真贡献:用"in-prompt 分层提示 + 当前策略自采替换 + 比率对带提示 prompt 算"这三件,**在不改 GRPO 损失的前提下严格保住 on-policy 一致性地攻克学习悬崖**——这是对前缀法(LUFFY 等)一个干净且概念清晰的改进,消融(逐层、progressive、incremental、exemption、过滤)做得相当扎实,跨 5 模型一致增益增强了普适性论断。
  - 包装/高估:"真零外部知识/自主"并不成立——hint 由更强的 DeepSeek-R1 生成,本质仍是教师知识下放;"internalization"靠 graduation 计数和单例图,缺记忆探针硬证;best-checkpoint 报告 + 仅数学域,使"unlocking ability beyond reach"的普适性有夸张空间。
  - 低估之处:"数据难度需配能消化它的框架"(Table 3:难数据只对 Scaf-GRPO 有大增益)其实是个对 RLVR 课程设计很有价值的观察,但被放在 further analysis;exemption 10–40% 稳的鲁棒性也说明方法对该超参不敏感,这点实用性未被突出。
  - 〔待核·论文内部小笔误〕Table 1 标题把非 Qwen 模型写作 "Llama-3.2-8B-Instruct",但 §4.1 Models 与表内行、§4.2 正文均为 **Llama-3.2-3B-Instruct**——应以 3B 为准(标题处疑笔误)。

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号:可验证 **结果奖励 R(对/错)**经 GRPO 组内归一化 advantage(轨迹级);提示本身不直接进损失,只改采样输入。
  - 改什么:**参数**(GRPO 策略更新);训练期还动态改 **prompt**(注入分层 hint)和 **rollout buffer**(替换失败轨迹)。
  - 何时改:**在线 per-step / 触发式**(仅在某 query 整组全错=学习悬崖时才触发引导;Qwen2.5-Math-7B 上仅 17.4% 样本触发)。
  - 免梯度?:否(标准 GRPO 梯度;引导只是数据增强)。
  - 记忆-技能生命周期:**无持久记忆/技能库**;三层 hint 是**离线预生成的静态资源**(训练前由 DeepSeek-R1 造好,按需注入);成功轨迹 o*_h 用完即弃(替换进当前 batch,不存储)。技能"固化"靠参数更新。
  - 防遗忘机制:**无显式防遗忘**;KL penalty=0(反而最大化探索而非锚定)。仅"exemption period 防 hint 依赖"算一种弱意义的"防过拟合到提示"。
- ⑦ 开源代码 + 框架/harness:https://github.com/JIA-Lab-research/Scaf-GRPO (已 clone 约 7.2MB,Tier A,origin 已核)。框架 **veRL 0.4.1.dev**(仓库含 `verl/` 与 `hint_mix_grpo/{main_ppo.py, trainer/, dataset/, config/}`;rollout 用 vLLM;conda env `scaf-grpo`/py3.10)。入口 `train.sh` → `bash sh/hint_mix_grpo/bs256_6k_mix.sh`;baseline/评测在 `sh/{baseline,generation_eval}`。数据集 HuggingFace `hkuzxc/scaf-grpo-dataset`。
- 💰 资源/成本与可扩展性:Qwen2.5-Math-7B 训练 ~73GB peak（vanilla ~72GB,几乎无额外显存,Table 5）；**引导仅 17.4% 样本触发**故吞吐多用于标准生成；到最佳 checkpoint **~12h**(vanilla GRPO 需 ~13h 才到更低峰值)——即更省总时间还更高分。max response 2048（长 CoT 8192）；hint 一次性离线生成（DeepSeek-R1）。可扩展性:方法与 GRPO 损失解耦、只改数据，原则上可挂在任意 GRPO 实现上。
- 🎯 对"探索-巩固"对标:**强支撑(尤其"探索/选路"维度)+ 可借组件 / 部分竞品**。判定:Scaf-GRPO 是我们"探索-巩固"里**探索/选路**思路的最贴近实现之一——它的"路标 vs 铁轨"哲学、"提示只指方向、让 student 自己采出能走通的解"几乎就是我们"岔路口偏向自己能走通的开头/方向 + student 全程 on-policy 自选"的 RLVR 版落地。同时它是 path-recovery 的一种范式:全错(走偏)时,用最小提示帮 student 自采一条对的来"接着做对"。依据:§3.2 on-policy 替换 + Fig.5 毕业轨迹。可直接借的积木:① **触发式干预**(只在整组全错/advantage 坍缩时才介入,平时纯自主)——天然契合我们"只在岔路口/走偏时让 teacher 当稀疏脚手架";② **比率对带提示 prompt 计算**这一保 on-policy 的技术细节(Eq.4),对我们"student 自选恢复分支后仍要 on-policy 更新"直接可用;③ **exemption period** 防过早依赖脚手架的课程设计;④ **三层抽象→具体 + 增量**作为"最小有效脚手架"的粒度模板。部分竞品:它已占据"教师分层提示 + 保 on-policy + 保探索"这块,与我们 idea 重叠度高,需在"MTP 前瞻探针 + student 自选恢复分支(而非确定性搜索预设三层 hint)+ 自蒸馏固化"上做出区分。缺口:hint 是**预生成静态三层**(非 student 自己发现的恢复分支),且**无 MTP/前瞻**、**无防遗忘/技能库**。
- 🔭 开放问题/未来方向:
  - 【原文】正文未设独立"Future Work"节;隐含方向:更鲁棒的 true/pseudo-hard 判定、把脚手架推广到更广模型/任务(论文反复强调 model-agnostic 但只测数学)。
  - 【推断】扩到代码/通用推理域;补"训练题 vs 持出题"记忆探针以坐实"内化非记忆";用固定步数而非 best-checkpoint 复核增益;把"预生成静态三层 hint"换成"student 自采/自生成的恢复分支"以更接近真自进化;研究 exemption/温度/组大小等超参的系统鲁棒性。

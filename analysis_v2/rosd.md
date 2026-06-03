rosd | ROSD: Reflective On-Policy Self-Distillation for Language Model Reasoning across Domains | 香港理工 + 百度 + 山东大学 + 莱顿大学(Ziqi Zhao 一作;通讯 Xiao-Ming Wu) | 2026-05-27 v1 · arXiv preprint(cs.CL) | L1 OPD/自蒸馏 · 相关性高

**原始论文**:https://arxiv.org/abs/2605.28014

## 一眼看懂
- 🟦 TL;DR:在线策略自蒸馏(OPSD)给学生自采样轨迹提供稠密 token 级监督,但标准做法把"自教师"条件于**完整正确解**,会让学生去模仿训练域参考轨迹而非纠正自己的错误,且对整条 response 蒸馏会覆盖已正确的前缀——结果域内不稳、域外(OOD)崩。ROSD 改为"反思引导 + 错误定位":对每条错误 rollout,用 self-reflector 抽出"纠错关键思路 e"和"错误引语 q(第一处出错的精确子串)";e 注入自教师做**针对性纠错**,q 把蒸馏 loss **限制在出错后缀**(保留有效前缀)。结果域内更强、OOD 远好于 SDPO(4B 上 ToolUse 训练后 OOD 均值 SDPO 跌到 2.88%,ROSD 维持 41.31%)。【原文】Abstract、§3、Table 3
- 最巧的一步:**用 error quote q 做"第一处出错"定位 + 只蒸出错后缀(mask t<k 置 0)**。抽掉这步退化成"用反思思路 e 但仍全响应蒸馏",有效前缀会被覆盖、训练域风格(如 "according to the reference solution")会注入正确推理段,OOD 泛化优势就没了。e(纠错思路替代全解模仿)与 q(定位)二者缺一不可,但 q 的"局部化"是把 OOD 救回来的关键。【原文】§3.2、Eq.7-9

## 为什么做
- 研究背景:后训练对 LLM 推理至关重要。RLVR(GRPO)只用 outcome reward 算 response 级 advantage,无 token 级监督;OPSD(on-policy self-distillation)用"与学生同权重的自教师"在 token 级提供稠密监督来弥补——给学生 on-policy rollout,让自教师条件于一个正确解,对师生分布算 token 级 KL。【原文】§1
- 解决的具体痛点:标准 OPSD 即便域内也不稳健(pilot study/Fig.1:域内早期升、后期不稳甚至降;OOD 快速恶化)。两归因:(1) 自教师条件于"已验证正确解"鼓励学生模仿参考轨迹而非纠错,内化域特定模式;(2) 全响应蒸馏覆盖已正确前缀、惩罚有效替代前缀,并把训练域偏好注入正确推理(自教师还会产生 reference-dependent 措辞)。【原文】§1
- 相关工作 & 各自不足:RLVR/GRPO(无 token 监督);SDPO(Hübotter et al. 2026,用成功 rollout 作自教师上下文,但全响应蒸馏、引入参考依赖);OPSD(Zhao 2026a)/ SDFT(Shenfeld 2026)——依赖人工 golden 解,故未纳入主对比。【原文】§1、§4.1
- 动机链:错误通常是**局部**的 → 只该修"第一处出错起的后缀" → 用"纠错思路"而非完整参考解引导教师(把"参考解模仿"转为"针对性纠错")→ 保留有效前缀 → 域内更稳 + OOD 改善。【原文】§1、§3
- 与最近邻工作(SDPO)的Δ:SDPO 全响应蒸馏 + 条件于完整正确解;ROSD 改两点:**条件于"反思抽出的纠错思路 e"(特权信息但非全解)+ 蒸馏只从错误引语 q 定位的后缀起**。有用之处:把自教师增益的来源从"灌训练域风格"剥离为纯"特权信息条件",从而不伤 OOD。【原文】§3、Fig.1

## 怎么做 + 靠不靠谱
- 方法流水线:① 每题采 G 条 on-policy rollout,按最终答案对错分 Y⁺/Y⁻ → ② 对每条错误 rollout y⁻,配同组**最短**正确 rollout y*,self-reflector 输出 `<error_quote>`(q,出错精确子串)+ `<explanation>`(e,错因+修法+正确逻辑);正确 rollout 只单独反思得 e("为何有效")→ ③ 把 e 注入自教师 prompt 作 auxiliary context;用 q 在 y⁻ 中定位 k=Locate(q,y⁻),mask m_t=0 (t<k)/1 (t≥k),匹配失败回退全响应 k=0,正确 rollout m_t=1 全 token → ④ 对 mask 后 token 算师生散度,教师分布 π_θ(·|x,e,y<t) 取 stopgrad。【原文】§3.1-3.2
- 逐组件必要性:
  - **error-focused self-reflector(e + q)**:核心组件 1。e 替代全解模仿(避免灌训练域风格),q 提供定位锚点。正确 rollout 也反思(只得 e),保留成功推理信息而不直接全解模仿。【原文】§3.1
  - **quote-localized distillation(mask)**:核心组件 2。只蒸出错后缀,保留有效前缀。论文给了主对比(vs GRPO/SDPO)与训练动力学(Fig.3),**但缺"全响应 vs 局部"的直接消融**——"局部化是 OOD 关键"主要靠"ROSD 远超 SDPO"间接支撑,而非控制变量消融。【推断】
  - **共享 base(student=teacher=reflector)**:排除"更大外部教师"混淆变量,增益纯来自特权信息条件。【原文】§4.1
- 关键机制/公式(直觉):L_ROSD = Σ_t m_t · KL(π_student(·|x,y<t) ‖ stopgrad[π_teacher(·|x,e,y<t)])(Eq.9)。直觉:教师"偷看"了纠错思路 e(特权信息),学生只在"自己开始出错的地方"向这个更明白的教师对齐,出错前的正确部分不动。【原文】Eq.9
- 实验与证据:backbone=Qwen3-4B / Qwen3-8B(student/teacher/reflector 同 base)。在 **5 个数据集**各单独训练:SciKnowEval(L3) 的化学/物理/生物/材料 4 个本科科学 QA + ToolAlpaca 工具调用(ToolUse);额外用 AIME2024 评数学。"训一域评所有域"。每 prompt 采 8 条 rollout,评估每题 16 条报 mean@16。baseline:强化版 GRPO(非对称 clip + 无偏归一化 + off-policy 校正)、SDPO。
  - 域内(Table 2):ROSD 平均 4B **72.83%**(较 GRPO/SDPO +2.97/+5.81)、8B **73.45%**(+1.46/+0.95,8B 增益小)。注意:GRPO 与 SDPO 域内平均接近,说明 SDPO 的稠密监督未明显超过 outcome-level RL(论文归因 SDPO 引入噪声监督)。【原文】§4.2、Table 2
  - OOD(Table 3):**关键且诚实的发现——GRPO 的 OOD 鲁棒性最强**(因无 token 级 KL 约束),ROSD 介于 GRPO 与 SDPO 之间;ROSD 在所有训练域、两规模上**稳超 SDPO** 但**多数设置下不超 GRPO**。极端跨域(ToolUse→其他)SDPO 崩溃:4B OOD 均值 2.88%(且 ToolUse 列多处 0.49/0.83),ROSD 维持 41.31%(GRPO 43.57%);Chemistry 训练时 SDPO AIME2024=0.00、ROSD=30.56。【原文】§4.3、Table 3
  - baseline 公平性:同数据划分/verifier/eval prompt,GRPO 用强化实现,公平。【原文】§4.1
- 假设与失效边界:
  - 【原文】q 定位依赖 reflector 抽"精确子串"并能在 y⁻ 匹配;匹配失败回退全响应(k=0)——论文未报回退比例。【待核:回退率】
  - 【原文】散度实现:PDF 正文明确"following Hübotter et al. 2026; Li et al. 2026a,用 token 级散度 with **Jensen–Shannon divergence (JSD)**"(§3.2)。〔注:v1 代码核查发现仓库 `core_algos.py` 中由 `alpha` 配置——alpha=1 为 reverse KL(SDPO 默认)、alpha=0 forward KL、中间为广义 JSD;即 JSD 是论文报告设置,但代码层可切换。PDF 与代码无冲突,论文报告的是 JSD。〕
  - 【推断】"第一处出错"假设单点错误;多处分散错误或错误前缀本身无效时,"保留前缀"会保留错误前提。
  - 【推断】reflector 质量依赖同一 base,弱基座下反思可能不可靠;8B 上相对 SDPO 增益已很小(域内 +0.95),方法收益随基座变强而衰减,规模化外推存疑。
- 祛魅总结:
  - 真贡献:把"自教师该看什么(纠错思路而非全解)+ 该改哪里(出错后缀)"两个直觉用 reflection+quote 落地,显著缓解 OPSD 的 OOD 崩溃问题,且无需更大外部教师。【推断】
  - 包装/需冷静看:**TL;DR/摘要的"显著改善 OOD"是相对 SDPO 而言**;Table 3 显示 ROSD 在 OOD 上多数设置仍**不及最朴素的 GRPO**——即"加 token 级监督本身就会牺牲 OOD,ROSD 只是把这个代价压小、没消除"。SDPO 域内也没真正超过 GRPO,凸显 OPSD 这条线整体收益的脆弱性。【推断·依据 Table 2/3】
  - 缺少对"局部化"本身的消融,是方法论上的小缺口。【推断】

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号:token 级师生散度(JSD),教师条件于"纠错思路 e"(特权信息);路由信号=答案对错。
  - 改什么:OPSD 的**蒸馏作用范围**(只蒸出错后缀)+ 教师**条件上下文**(e 替代全解)。
  - 何时改:每步在线,按当前 rollout 的对错与 reflector 输出动态定位。
  - 免梯度?否,梯度蒸馏(masked token-level KL/JSD,教师 stopgrad)。
  - 记忆-技能生命周期:无外部记忆/技能库;反思 e 是一次性的 in-context 信号,不持久化。
  - 防遗忘机制:**核心卖点即一种防(OOD)遗忘**——局部化蒸馏 + 纠错思路条件,避免把训练域风格灌进学生从而保留跨域能力;但非显式 replay/正则,而是"少改"。
- ⑦ 开源代码+框架/harness:https://github.com/ZiqiZhao1/ROSD(Apache-2.0,v1 已克隆 ~53MB,含完整 verl + 实验脚本 + reward)。框架=**verl**(verl-project/verl)扩展自 **SDPO**(lasgroup/SDPO);FSDP actor + vLLM rollout,8×A800 80G。关键文件:`core_algos.py`(`compute_self_distillation_loss`,alpha 配 KL/JSD)、`dp_actor.py`(自蒸馏 loss 分发)、`ray_trainer.py`(rollout/reward/reflection/distillation batch)、`config/sdpo.yaml`;主入口 `experiments/local/run_sdpo_processreflection_ood_all.sh`。【原文】Abstract + 仓库(v1 核查)
- 💰 资源/成本与可扩展性:8×A800 80G;每 prompt 8 rollout + 每条错误轨迹需额外一次 reflector 前向(同 base,无额外存储)+ 自教师前向,推理成本高于纯 GRPO。【原文】§4.1 + 仓库
- 🎯 对"探索-巩固"对标:**最强对标(直接竞品 + 高度可借)**。判定:ROSD 几乎就是本项目"教 student 两能力"的一个文本实例——**保留有效前缀≈"偏向自己走得通的开头"(探索/选路);从第一处出错起由"看了纠错思路的自教师"接管纠错≈ path-recovery(走偏后恢复)**;on-policy 自采样、自教师同 base 也契合"自选"。可直接借:① error quote 做"第一处走偏点"定位的工程实现;② "教师只给纠错思路而非全解"以避免风格灌注。关键缺口/Δ vs 本项目:(a) ROSD 用 **JSD 全后缀蒸馏**,本项目主张"**单点接管**"(更细的 step/token 级)而非整段后缀;(b) 无 MTP 前瞻(本项目用 MTP 做 foresight probe 找切点);(c) 无"固化进参数/记忆且不遗忘"的显式机制,只是"少改以保 OOD"。依据:§3.1-3.2 + Table 3。【推断】
- 🔭 开放问题/未来方向:【原文】(论文聚焦实证,未列大量未来方向) 【推断】回退率(quote 匹配失败)对结果的影响量化;多处错误/无效前缀的处理;把"整段后缀蒸馏"细化到"单点/关键步接管"(对接本项目);8B 以上规模收益衰减的成因;为何 ROSD 仍不及 GRPO 的 OOD——能否在保 token 监督的同时彻底追平。

RETURN: rosd|读到PDF?是(12页,_txt 46k字)|L1|对标=最强对标:保留前缀=探索/出错后缀自教师接管=path-recovery,可借 error-quote 定位;Δ=整段后缀蒸馏(非单点接管)且无 MTP|残留待核 1(quote 匹配失败回退率,论文未报)

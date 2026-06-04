rosd | ROSD: Reflective On-Policy Self-Distillation for Language Model Reasoning across Domains | 香港理工 + 百度 + 山东大学 + 莱顿大学(Ziqi Zhao 一作;通讯 Xiao-Ming Wu) | 2026-05-27 v1 · arXiv preprint(cs.CL) | L1 OPD/自蒸馏 · 相关性高

**原始论文**:https://arxiv.org/abs/2605.28014

## 一眼看懂
- 🟦 TL;DR:在线策略自蒸馏(OPSD)给学生自采样轨迹提供稠密 token 级监督,但标准做法把"自教师"条件于**完整正确解**,会让学生去模仿训练域参考轨迹而非纠正自己的错误,且对整条 response 蒸馏会覆盖已正确的前缀——结果域内不稳、域外(OOD)崩。ROSD 改为"反思引导 + 错误定位":对每条错误 rollout,用 self-reflector 抽出"纠错关键思路 \(e\)"和"错误引语 \(q\)(第一处出错的精确子串)";\(e\) 注入自教师做**针对性纠错**,\(q\) 把蒸馏 loss **限制在出错后缀**(保留有效前缀)。结果域内更强、OOD 远好于 SDPO(4B 上 ToolUse 训练后 OOD 均值 SDPO 跌到 2.88%,ROSD 维持 41.31%)。【原文】Abstract、§3、Table 3
- 最巧的一步:**用 error quote \(q\) 做"第一处出错"定位 + 只蒸出错后缀(mask \(m_t=0,\ t<k\))**。抽掉这步退化成"用反思思路 \(e\) 但仍全响应蒸馏"(即消融 w/o Localized Distillation),Table 4 证此变体 OOD 平均虽仍优于 SDPO 但低于完整 ROSD(4B OOD 38.91 vs 42.46);而抽掉反思 \(e\)(w/o Reflection,仍局部化但教师条件于正确解)Chemistry 训练时 OOD 从 38.54 掉到 31.80。两者缺一不可,但定位(\(q\))与反思条件(\(e\))分别针对"覆盖有效前缀"与"灌训练域风格"两个病灶。【原文】§3.2、Eq.7-9、Table 4

## 为什么做
- **研究背景的来龙去脉**:后训练是 LLM 推理能力的核心驱动(GPT-4、Kimi-k1.5)。其中 RLVR(以 GRPO 为代表)用规则可验证的 outcome reward 训练,免去逐步人工标注;但 GRPO 只算 **response 级** group-relative advantage \(A_i=\dfrac{r_i-\mu_G}{\sigma_G}\)(Eq.1),同一条 rollout 内所有 token 共享一个 \(A_i\),**无 token 级监督**,无法区分一条错误轨迹里哪些 token 仍有效、哪些该改——这是稠密信号缺失的根因(§2.1)。
- **并行技术路线一:稠密化 outcome 信号**。OPSD(on-policy self-distillation)这条线用"与学生同权重的自教师"在 token 级补稠密监督:给学生 on-policy rollout \(y\),让自教师条件于某个正确解 \(c\),对师生分布算 token 级 KL。代表作 SDPO(Hübotter et al. 2026,RL setting,自教师条件于成功 rollout)、OPSD(Zhao 2026a,SFT setting,条件于 golden 解)、SDFT(Shenfeld 2026,SFT,条件于 golden 解,评 in+out domain)。Table 1 把这几家按 training-setting / supervision-source / eval-scope 三轴对齐:SFT 类依赖数据集提供的 golden 解,RL 类靠 verifier 信号——ROSD 与 SDPO 同属 RL/verifier-only setting,故只与 SDPO 主对比,OPSD/SDFT 因依赖人工 golden 解被排除出主对比。【原文】§2.2、Table 1、§4.1
- **并行技术路线二:GRPO 变体**(改 objective/normalization/clipping/sampling,如 DAPO 的非对称 clip、Dr.GRPO 的无偏归一化、GSPO/VAPO/GPG),但这条线**仍是 response 级**,不提供 token 监督——ROSD 的强化版 GRPO baseline 即吸收了 DAPO 非对称 clip + Dr.GRPO 无偏归一化 + Yao 2025 off-policy 校正三项最佳实践,作为公平对照。【原文】§2.1、§4.1
- **SDPO 的具体短板(本文靶子)**:pilot study(Fig.1,Material 训练、Qwen3-8B)显示 SDPO 即便**域内**也不稳健——早期升、后期不稳甚至降;OOD 快速恶化。两归因:(1) **自教师条件于"已验证正确解"** → 鼓励学生模仿参考轨迹而非定位/纠正自己的错;正确解只给"对的样子"却不指"学生哪步错、该如何局部修",学生因而内化域特定参考模式;(2) **全响应蒸馏** → 覆盖已正确前缀、惩罚有效替代前缀,并把自教师受参考解诱导产生的 "according to the reference solution" 式 reference-dependent 措辞 / 域特定风格注入正确推理段。两者合力把模型推向"模仿训练域参考轨迹"而非"做针对性纠错",伤 OOD。【原文】§1、§3.1-3.2
- **本文站在谁肩上 + 与最近邻(SDPO)的精确差异**:沿用 SDPO 的"on-policy + 自教师条件于特权信息 + token 级散度"骨架,只改两点——① 自教师条件上下文从"完整正确解 \(c\)"换成"反思抽出的纠错思路 \(e\)"(特权信息但非全解,剥离"参考解模仿"成分);② 蒸馏作用范围从"全响应"收窄到"error quote \(q\) 定位的后缀"(保留有效前缀)。本质是把 SDPO 自教师增益的来源从"灌训练域风格"提纯为"纯特权信息条件",从而保住 OOD。【原文】§3、Fig.1

## 怎么做 + 靠不靠谱
- **方法流水线(输入→输出,逐模块)**:
  1. **采样 + 分组**(§3.1):对每题 \(x\),当前策略 \(\pi_\theta\) 采 \(G\) 条 on-policy rollout \(Y(x)=\{y_1,\dots,y_G\}\)(实现:\(G=8\)),按最终答案对错切成正确集 \(Y^+(x)\) 与错误集 \(Y^-(x)\)。**输出**:对错标签 + 两个子集。
  2. **错误聚焦自反思**(§3.1,核心组件 1):对每条错误 rollout \(y^-\in Y^-(x)\),配同组**最短**正确 rollout \(y^*\in Y^+(x)\)(选最短意在鼓励反思偏向更高效推理路径);self-reflector(复用同一 base 权重)接收 \((x,y^*,y^-)\),按固定模板(Fig.2:`<error_quote>` + `<explanation>`)输出
     \(\displaystyle (e,q)\sim\pi_\theta(\cdot\mid x,y^*,y^-)\)
     其中 \(e\)=纠错关键思路(为何错 + 怎么修 + 正确逻辑),\(q\)=错误引语(从 \(y^-\) 抄出的**精确子串**,标第一处出错跨度)。对正确 rollout \(y^+\) 用另一模板只反思得 \(e\sim\pi_\theta(\cdot\mid x,y^+)\)(总结"为何有效"),**不直接复用整条 \(y^+\) 当教师上下文**——保留成功推理信息而不诱发全解模仿。**输出**:每条 rollout 配一个 \(e\),错误 rollout 额外配 \(q\)。
  3. **quote 定位 + masking**(§3.2,核心组件 2):对错误 rollout,把 \(q\) 对齐回 \(y^-\) 得起始 token 下标
     \(\displaystyle k=\mathrm{Locate}(q,y^-)\)
     匹配失败则回退全响应蒸馏(\(k=0\));再构 mask
     \(\displaystyle m_t=\begin{cases}0,& t<k,\\ 1,& t\ge k.\end{cases}\)
     正确 rollout 无 \(q\),取 \(m_t=1\)(全 token)。**输出**:每条 rollout 的 token 级 mask。
  4. **掩码 token 级蒸馏**(§3.2):自教师条件于 \(e\) 并被指示"输出修正解";在位置 \(t\),学生分布 \(\pi_\theta(\cdot\mid x,y_{<t})\)、自教师分布 \(\pi_\theta(\cdot\mid x,e,y_{<t})\)(教师取 stopgrad)。训练目标
     \(\displaystyle \mathcal{L}_{\mathrm{ROSD}}(\theta)=\sum_{t=1}^{T} m_t\,\mathrm{KL}\!\Big(\pi_\theta(\cdot\mid x,y_{<t})\ \big\|\ \mathrm{stopgrad}\big[\pi_\theta(\cdot\mid x,e,y_{<t})\big]\Big).\)
     对照 SDPO 的全响应目标 \(\mathcal{L}_{\mathrm{SDPO}}(\theta)=\sum_{t=1}^{T}\mathrm{KL}\big(\pi_\theta(\cdot\mid x,y_{<t})\,\|\,\mathrm{stopgrad}[\pi_\theta(\cdot\mid x,c,y_{<t})]\big)\)(Eq.3):ROSD 把条件上下文 \(c\to e\)、并乘上 mask \(m_t\)。**实现细节**(附录 B):散度实例化为 **Jensen–Shannon divergence (JSD),\(\alpha=0.5\)**,distillation top-\(k=100\),自教师/反思器训练期**冻结**。**输出**:学生参数更新。
- **逐组件必要性(基于 Table 4 真实消融,4B/8B,mean@16%)**:
  - **error-focused reflector(\(e+q\))**:w/o Reflection(保留 \(q\) 定位 + 局部化,但教师退回条件于正确解,如 SDPO)→ 域内与 SDPO 相当、OOD 大幅改善(说明"局部化"本身就压住了全响应蒸馏的泛化损伤);但其 Chemistry 列 OOD 仍明显弱于完整 ROSD(4B 31.80 vs 38.54)——证 \(e\) 这条"纠错思路替代正确解条件"提供了更有信息、更少噪声的教师信号。【原文】§4.5、Table 4
  - **quote-localized distillation(mask)**:w/o Localized Distillation(用反思条件教师 \(e\) 但全响应蒸馏)→ 域内/OOD 均优于 SDPO(证 \(e\) 有效),但仍低于完整 ROSD(4B OOD 38.91 vs 42.46)——证局部化在 \(e\) 之上还有独立增益。**注:v1 曾记"缺全响应 vs 局部消融",本轮 PDF 复核 Table 4 实含此消融,予以更正。**【原文】§4.5、Table 4
  - **error localization dynamics(Fig.5)**:训练中 quote 匹配率约 **0.5**(8B 略高于 4B,与其更强 instruction-following 一致);归一化错误位置随训练**逐渐右移**(错误更少更靠后)——这正解释了"为何全响应蒸馏后期变有害":早期推理已多数正确时,继续蒸全响应会无谓扰动正确前缀。【原文】§4.6、Fig.5
  - **共享 base(student=teacher=reflector 同权重)**:排除"更大外部教师"混淆变量,增益纯来自特权信息条件,且 reflector 复用权重不增存储。【原文】§4.1
- **关键机制/公式(直觉)**:教师"偷看"了纠错思路 \(e\)(特权信息),学生只在"自己开始出错的地方(\(t\ge k\))"向这个更明白的教师对齐,出错前的正确部分(\(t<k\),\(m_t=0\))不动。JSD(\(\alpha=0.5\))相比纯 reverse-KL 更对称、训练更稳;top-\(k=100\) 把蒸馏限制到教师高概率 token,降噪。【原文】Eq.9、附录 B
- **训推数据如何流动**:数据流为 rollout(vLLM 生成)→ 对错切分 → reflector 前向(同 base)产 \((e,q)\)→ teacher 前向(条件 \(e\),stopgrad)产分布 → masked JSD 反传更新 FSDP actor。每 prompt 1 次 rollout(8 条)+ 每条错误轨迹 1 次额外 reflector 前向 + 自教师前向,故推理成本高于纯 GRPO;但反思 prompt 上限 8k token、反思输出上限 4k token,且训练后期因学生响应更短反而比 baseline 更省时(附录 C)。
- **实验与证据**:backbone=Qwen3-4B / Qwen3-8B(student/teacher/reflector 同 base)。**5 个数据集各单独训练**:SciKnowEval(L3) 的化学/物理/生物/材料 4 个本科科学 QA + ToolAlpaca 工具调用(ToolUse);额外用 AIME2024 评数学。"训一域评所有域"。每 prompt 采 8 rollout,评估每题 16 条报 mean@16。**训练超参(附录 B)**:8×NVIDIA A800 80G、verl(FSDP actor + vLLM rollout)、science 任务 10 epochs / ToolUse 5 epochs、batch size 32。baseline:强化版 GRPO(DAPO 非对称 clip + Dr.GRPO 无偏归一化 + off-policy 校正)、SDPO。
  - 域内(Table 2):ROSD 平均 4B **72.83%**(较 GRPO/SDPO +2.97/+5.81)、8B **73.45%**(+1.46/+0.95,8B 增益小)。逐域看 4B 上 Material 80.18、Physics 76.56、Chemistry 82.47、ToolUse 67.46 多为最佳。GRPO 与 SDPO 域内平均接近(4B 69.86 vs 67.02),说明 SDPO 的稠密监督未明显超过 outcome-level RL(论文归因 SDPO 引入噪声监督)。【原文】§4.2、Table 2
  - OOD(Table 3):**关键且诚实的发现——GRPO 的 OOD 鲁棒性最强**(因无 token 级 KL 约束),ROSD 介于 GRPO 与 SDPO 之间;ROSD 在所有训练域、两规模上**稳超 SDPO** 但**多数设置下不超 GRPO**。极端跨域(ToolUse→其他)SDPO 崩溃:4B OOD 均值 2.88%(ToolUse 列多处 0.49/0.83),ROSD 维持 41.31%(GRPO 43.57%);Chemistry 训练时 SDPO AIME2024=0.00、ROSD=30.56。【原文】§4.3、Table 3
  - 训练动力学(Fig.3-4):ROSD 收敛快且全程稳;SDPO 早期升后期降。Fig.4 揭示 SDPO 在 Material 上 rollout accuracy 末期仍升但 test score 反降(过拟合自采 rollout)。【原文】§4.4
  - baseline 公平性:同数据划分/verifier/eval prompt,GRPO 用强化实现,公平。【原文】§4.1
- **假设与失效边界**:
  - 【原文】\(q\) 定位依赖 reflector 抽"精确子串"并能在 \(y^-\) 匹配;匹配失败回退全响应(\(k=0\))——Fig.5 报匹配率约 0.5,即约一半错误轨迹退化为全响应蒸馏。
  - 【原文】散度=JSD(\(\alpha=0.5\),附录 B 明确;§3.2 亦说"following Hübotter 2026; Li 2026a 用 token 级散度 with JSD")。〔注:v1 代码核查发现仓库 `core_algos.py` 由 `alpha` 配置——\(\alpha=1\) reverse KL(SDPO 默认)、\(\alpha=0\) forward KL、中间为广义 JSD;论文报告设置 \(\alpha=0.5\) 即对称 JSD,代码可切换。PDF 与代码无冲突。〕
  - 【原文·Limitations】方法仍**不及 GRPO 的 OOD**,作者承认跨域泛化需进一步加强;且需在更广 setting 验证普适性。
  - 【推断】"第一处出错"假设单点错误;多处分散错误或错误前缀本身无效时,"保留前缀"会保留错误前提。
  - 【推断】reflector 质量依赖同一 base,弱基座下反思可能不可靠;8B 上相对 SDPO 增益已很小(域内 +0.95),方法收益随基座变强而衰减,规模化外推存疑。
- **祛魅总结**:
  - 真贡献:把"自教师该看什么(纠错思路 \(e\) 而非全解)+ 该改哪里(出错后缀,mask \(m_t\))"两个直觉用 reflection+quote 落地,显著缓解 OPSD 的 OOD 崩溃,且无需更大外部教师;Table 4 双消融 + Fig.5 定位动力学构成较完整证据链。【推断】
  - 包装/需冷静看:**TL;DR/摘要的"显著改善 OOD"是相对 SDPO 而言**;Table 3 显示 ROSD 在 OOD 上多数设置仍**不及最朴素的 GRPO**——即"加 token 级监督本身就会牺牲 OOD,ROSD 只是把这个代价压小、没消除"。SDPO 域内也没真正超过 GRPO,凸显 OPSD 这条线整体收益的脆弱性。【推断·依据 Table 2/3 + Limitations】

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号:token 级师生散度(JSD,\(\alpha=0.5\)),教师条件于"纠错思路 \(e\)"(特权信息);路由信号=答案对错。
  - 改什么:OPSD 的**蒸馏作用范围**(只蒸出错后缀,mask \(m_t\))+ 教师**条件上下文**(\(e\) 替代全解 \(c\))。
  - 何时改:每步在线,按当前 rollout 的对错与 reflector 输出动态定位 \(k\)。
  - 免梯度?否,梯度蒸馏(masked token-level JSD,教师 stopgrad)。
  - 记忆-技能生命周期:无外部记忆/技能库;反思 \(e\) 是一次性的 in-context 信号,不持久化。
  - 防遗忘机制:**核心卖点即一种防(OOD)遗忘**——局部化蒸馏 + 纠错思路条件,避免把训练域风格灌进学生从而保留跨域能力;但非显式 replay/正则,而是"少改"。
- ⑦ 开源代码+框架/harness:https://github.com/ZiqiZhao1/ROSD(Apache-2.0,v1 已克隆 ~53MB,含完整 verl + 实验脚本 + reward)。框架=**verl**(verl-project/verl)扩展自 **SDPO**(lasgroup/SDPO);FSDP actor + vLLM rollout,8×A800 80G。关键文件:`core_algos.py`(`compute_self_distillation_loss`,alpha 配 KL/JSD)、`dp_actor.py`(自蒸馏 loss 分发)、`ray_trainer.py`(rollout/reward/reflection/distillation batch)、`config/sdpo.yaml`;主入口 `experiments/local/run_sdpo_processreflection_ood_all.sh`。【原文】Abstract + 附录 B + 仓库(v1 核查)
- 💰 资源/成本与可扩展性:8×A800 80G;science 10 epochs / ToolUse 5 epochs,batch 32;每 prompt 8 rollout + 每条错误轨迹额外一次 reflector 前向(同 base,无额外存储)+ 自教师前向,推理成本高于纯 GRPO,但训练后期因响应更短反更省时(附录 C)。【原文】附录 B-C + 仓库
- 🎯 对"探索-巩固"对标:**最强对标(直接竞品 + 高度可借)**。判定:ROSD 几乎就是本项目"教 student 两能力"的一个文本实例——**保留有效前缀(\(m_t=0,t<k\))≈"偏向自己走得通的开头"(探索/选路);从第一处出错(\(k\))起由"看了纠错思路 \(e\) 的自教师"接管纠错≈ path-recovery(走偏后恢复)**;on-policy 自采样、自教师同 base 也契合"自选"。可直接借:① error quote 做"第一处走偏点"定位的工程实现(`Locate` + mask);② "教师只给纠错思路 \(e\) 而非全解 \(c\)"以避免风格灌注。关键缺口/Δ vs 本项目:(a) ROSD 用 **JSD 全后缀蒸馏**(从 \(k\) 到 \(T\) 整段),本项目主张"**单点接管**"(更细的 step/token 级)而非整段后缀;(b) 无 MTP 前瞻(本项目用 MTP 做 foresight probe 找切点,ROSD 靠 reflector 抽 quote);(c) 无"固化进参数/记忆且不遗忘"的显式机制,只是"少改以保 OOD"。依据:§3.1-3.2 + Table 3-4。【推断】
- 🔭 开放问题/未来方向:【原文·Limitations】跨域泛化需进一步加强(仍不及 GRPO);更广 setting 验证普适性。【推断】quote 匹配失败(约半数)对结果的影响量化;多处错误/无效前缀的处理;把"整段后缀蒸馏"细化到"单点/关键步接管"(对接本项目);8B 以上规模收益衰减的成因;为何 ROSD 仍不及 GRPO 的 OOD——能否在保 token 监督的同时彻底追平。

RETURN: rosd|读到PDF?是(12页,_txt 46k字,Eq.1-9 + Table 4 消融 + 附录 B 超参全核)|L1|对标=最强对标:保留前缀=探索/出错后缀自教师接管=path-recovery,可借 error-quote 定位;Δ=整段后缀蒸馏(非单点接管)且无 MTP|残留待核 0(v1 标"缺局部化消融"已更正为 Table 4 实含;JSD α=0.5/topk=100/8×A800/batch32 均从附录 B 核到)

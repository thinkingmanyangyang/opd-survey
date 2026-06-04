rosd | ROSD: Reflective On-Policy Self-Distillation for Language Model Reasoning across Domains | 香港理工 + 百度 + 山东大学 + 莱顿大学(Ziqi Zhao 一作;通讯 Xiao-Ming Wu) | 2026-05-27 v1 · arXiv preprint(cs.CL) | L1 OPD/自蒸馏 · 相关性高

**原始论文**:https://arxiv.org/abs/2605.28014

## 一眼看懂

> 一句话导读:让模型给自己当老师纠错时,别拿"标准答案"去比对(会变成抄答案),而是抽出"哪里错了、该怎么改"的思路,且只在出错的那一段上纠——这样域内更强、换个领域也不崩。

- 🟦 TL;DR:先交代背景。在线策略自蒸馏(OPSD,即用一个和学生同源的"自教师"在学生自己采样的轨迹上提供稠密的、逐 token 的监督)本意是给训练补充细粒度信号。但标准做法有两个毛病:
  - 自教师被**条件于完整正确解**,于是学生学的是"模仿训练域里的参考轨迹",而不是纠正自己的错误;
  - 对整条 response 都做蒸馏,会把学生本来已经写对的前缀也一起覆盖掉。
  - 结果就是:域内训练不稳,换到域外(OOD,即训练时没见过的领域)直接崩。
- ROSD 的改法是"反思引导 + 错误定位":对每条答错的 rollout(rollout = 模型自己采样出来的一条完整解答),用一个 self-reflector(自反思器)抽出两样东西——纠错关键思路 \(e\),和错误引语 \(q\)(\(q\) 是从原答案里逐字抄出的、标出第一处出错的精确子串)。
  - \(e\) 注入自教师,让它做**针对性纠错**;
  - \(q\) 把蒸馏 loss **限制在出错后缀**上,保住前面有效的部分。
- 效果:域内更强,OOD 远好于 SDPO(代表性数字:4B 模型用 ToolUse 训练后,OOD 均值 SDPO 跌到 2.88%,ROSD 维持 41.31%)。【原文】Abstract、§3、Table 3
- 最巧的一步:**用 error quote \(q\) 定位"第一处出错"的位置,然后只蒸出错后缀(mask \(m_t=0,\ t<k\),即出错点之前的 token 不参与蒸馏)**。
  - 抽掉这步,方法就退化成"仍然用反思思路 \(e\),但对全响应蒸馏"——这正是消融实验 w/o Localized Distillation。Table 4 显示:这个变体 OOD 平均虽仍优于 SDPO,但低于完整 ROSD(4B OOD 38.91 vs 42.46)。
  - 反过来,如果抽掉反思 \(e\)(消融 w/o Reflection:仍做局部化定位,但教师退回条件于正确解),Chemistry 训练时 OOD 从 38.54 掉到 31.80。
  - 结论:定位(\(q\))与反思条件(\(e\))缺一不可,二者分别治"覆盖有效前缀"和"灌训练域风格"这两个病灶。【原文】§3.2、Eq.7-9、Table 4

## 为什么做

> 一句话导读:RLVR 这套训练只给"整条对/错"一个奖励,看不清一条错答案里哪几步其实是对的;现有的 OPSD 想补细粒度监督,却因为"抄正确解 + 全程蒸"反而把训练域的腔调灌进了学生,换领域就崩——ROSD 就是冲着这两个毛病去的。

- **研究背景的来龙去脉**:后训练是 LLM 推理能力的核心驱动(GPT-4、Kimi-k1.5)。其中 RLVR(可验证奖励的强化学习,以 GRPO 为代表)用规则可验证的 outcome reward(结果奖励,即只看最终答案对不对)来训练,免去逐步人工标注。
  - 但 GRPO 只算 **response 级**(整条响应一个分数)的 group-relative advantage(组内相对优势)\(A_i=\dfrac{r_i-\mu_G}{\sigma_G}\)(Eq.1)。
  - 同一条 rollout 内所有 token 共享同一个 \(A_i\),所以**没有 token 级监督**,无法区分一条错误轨迹里哪些 token 仍有效、哪些该改——这是稠密信号缺失的根因(§2.1)。
- **并行技术路线一:稠密化 outcome 信号**。OPSD(on-policy self-distillation,在线策略自蒸馏)这条线用"与学生同权重的自教师"在 token 级补稠密监督:给学生 on-policy rollout(学生自己采的) \(y\),让自教师条件于某个正确解 \(c\),再对师生分布算 token 级 KL。
  - 代表作三家:SDPO(Hübotter et al. 2026,RL setting,自教师条件于成功的 rollout);OPSD(Zhao 2026a,SFT setting,条件于 golden 解,即数据集自带的标准答案);SDFT(Shenfeld 2026,SFT,条件于 golden 解,同时评域内+域外)。
  - Table 1 把这几家按三个维度对齐:训练设置(training-setting)/ 监督来源(supervision-source)/ 评测范围(eval-scope)。SFT 类依赖数据集提供的 golden 解,RL 类靠 verifier 信号。
  - ROSD 与 SDPO 同属 RL / 只用 verifier 的设置,所以只跟 SDPO 做主对比;OPSD/SDFT 因为依赖人工 golden 解,被排除出主对比。【原文】§2.2、Table 1、§4.1
- **并行技术路线二:GRPO 变体**(改 objective / normalization / clipping / sampling,例如 DAPO 的非对称 clip、Dr.GRPO 的无偏归一化、GSPO/VAPO/GPG)。但这条线**仍是 response 级**,不提供 token 监督。ROSD 的强化版 GRPO baseline 就吸收了三项最佳实践当公平对照:DAPO 非对称 clip + Dr.GRPO 无偏归一化 + Yao 2025 的 off-policy 校正。【原文】§2.1、§4.1
- **SDPO 的具体短板(本文靶子)**:pilot study(初步实验,Fig.1,Material 数据训练、Qwen3-8B)显示 SDPO 即便在**域内**也不稳——早期涨、后期不稳甚至下降;OOD 则快速恶化。论文把毛病归到两点:
  - (1) **自教师条件于"已验证正确解"**:这会鼓励学生模仿参考轨迹,而不是定位并纠正自己的错。正确解只给"对的样子",却没指出"学生哪步错、该如何局部修",学生于是内化了域特定的参考模式。
  - (2) **全响应蒸馏**:会覆盖已经正确的前缀、惩罚同样有效的替代前缀;还会把自教师受参考解诱导产生的措辞(例如 "according to the reference solution" 这种依赖参考解的腔调)和域特定风格注入到本来正确的推理段里。
  - 两者合力,把模型推向"模仿训练域参考轨迹"而非"做针对性纠错",于是伤 OOD。【原文】§1、§3.1-3.2
- **本文站在谁肩上 + 与最近邻(SDPO)的精确差异**:ROSD 沿用 SDPO 的骨架——on-policy + 自教师条件于特权信息(privileged information,即学生看不到、只给教师的额外信息)+ token 级散度——只改两点:
  - ① 自教师的条件上下文,从"完整正确解 \(c\)"换成"反思抽出的纠错思路 \(e\)"。\(e\) 仍是特权信息,但不是全解,这就剥离了"模仿参考解"的成分。
  - ② 蒸馏作用范围,从"全响应"收窄到"error quote \(q\) 定位出的后缀",保住有效前缀。
  - 本质上,是把 SDPO 里"自教师带来的增益"从"灌训练域风格"提纯为"纯特权信息条件",从而保住 OOD。【原文】§3、Fig.1

## 怎么做 + 靠不靠谱

> 一句话导读:流程四步——学生自己采一批答案、分对错;对错的那条让"反思器"抽出纠错思路 \(e\) 和出错引语 \(q\);用 \(q\) 找到出错位置、做出一个 mask 只盖住出错后缀;最后让看了 \(e\) 的自教师在这段后缀上教学生。两个消融 + 定位动力学撑起证据链,但要冷静看:它只是把"加 token 监督会牺牲 OOD"的代价压小,没消除。

- **方法流水线(输入→输出,逐模块)**:
  1. **采样 + 分组**(§3.1):对每道题 \(x\),用当前策略 \(\pi_\theta\) 采 \(G\) 条 on-policy rollout \(Y(x)=\{y_1,\dots,y_G\}\)(实现取 \(G=8\)),按最终答案对错切成正确集 \(Y^+(x)\) 与错误集 \(Y^-(x)\)。**输出**:对错标签 + 两个子集。
  2. **错误聚焦自反思**(§3.1,核心组件 1):对每条错误 rollout \(y^-\in Y^-(x)\),给它配上同组里**最短**的那条正确 rollout \(y^*\in Y^+(x)\)(选最短是为了让反思偏向更高效的推理路径)。
     - self-reflector(自反思器,复用同一 base 权重)接收 \((x,y^*,y^-)\),按固定模板(Fig.2:`<error_quote>` + `<explanation>`)输出
       \(\displaystyle (e,q)\sim\pi_\theta(\cdot\mid x,y^*,y^-)\)
     - 其中 \(e\) = 纠错关键思路(为何错 + 怎么修 + 正确逻辑),\(q\) = 错误引语(从 \(y^-\) 里抄出的**精确子串**,标出第一处出错的跨度)。
     - 对正确 rollout \(y^+\),用另一个模板只反思得 \(e\sim\pi_\theta(\cdot\mid x,y^+)\)(总结"为何有效"),并**不直接拿整条 \(y^+\) 当教师上下文**——这样既保留成功推理的信息,又不诱发全解模仿。
     - **输出**:每条 rollout 配一个 \(e\),错误 rollout 额外配一个 \(q\)。
  3. **quote 定位 + masking**(§3.2,核心组件 2):对错误 rollout,把 \(q\) 对齐回 \(y^-\),拿到出错的起始 token 下标
     \(\displaystyle k=\mathrm{Locate}(q,y^-)\)
     - 匹配失败就回退到全响应蒸馏(\(k=0\));再构造 mask
       \(\displaystyle m_t=\begin{cases}0,& t<k,\\ 1,& t\ge k.\end{cases}\)
     - 正确 rollout 没有 \(q\),取 \(m_t=1\)(全部 token 都参与)。
     - **输出**:每条 rollout 的 token 级 mask。
  4. **掩码 token 级蒸馏**(§3.2):自教师条件于 \(e\),并被指示"输出修正解";在位置 \(t\),学生分布是 \(\pi_\theta(\cdot\mid x,y_{<t})\)、自教师分布是 \(\pi_\theta(\cdot\mid x,e,y_{<t})\)(教师取 stopgrad,即不回传梯度)。训练目标
     \(\displaystyle \mathcal{L}_{\mathrm{ROSD}}(\theta)=\sum_{t=1}^{T} m_t\,\mathrm{KL}\!\Big(\pi_\theta(\cdot\mid x,y_{<t})\ \big\|\ \mathrm{stopgrad}\big[\pi_\theta(\cdot\mid x,e,y_{<t})\big]\Big).\)
     - 对照 SDPO 的全响应目标 \(\mathcal{L}_{\mathrm{SDPO}}(\theta)=\sum_{t=1}^{T}\mathrm{KL}\big(\pi_\theta(\cdot\mid x,y_{<t})\,\|\,\mathrm{stopgrad}[\pi_\theta(\cdot\mid x,c,y_{<t})]\big)\)(Eq.3):ROSD 做了两处改动——把条件上下文从 \(c\) 换成 \(e\),并乘上 mask \(m_t\)。
     - **实现细节**(附录 B):散度具体取 **Jensen–Shannon divergence (JSD),\(\alpha=0.5\)**;distillation top-\(k=100\);自教师 / 反思器在训练期**冻结**。
     - **输出**:学生参数更新。
- **逐组件必要性(基于 Table 4 真实消融,4B/8B,mean@16%)**:
  - **error-focused reflector(\(e+q\))**:消融 w/o Reflection(保留 \(q\) 定位 + 局部化,但教师退回条件于正确解,跟 SDPO 一样)→ 域内与 SDPO 相当、OOD 大幅改善。这说明"局部化"本身就压住了全响应蒸馏的泛化损伤;但它的 Chemistry 列 OOD 仍明显弱于完整 ROSD(4B 31.80 vs 38.54)——证明 \(e\)(用纠错思路替代正确解条件)提供了更有信息、更少噪声的教师信号。【原文】§4.5、Table 4
  - **quote-localized distillation(mask)**:消融 w/o Localized Distillation(用反思条件的教师 \(e\),但做全响应蒸馏)→ 域内 / OOD 都优于 SDPO(证 \(e\) 有效),但仍低于完整 ROSD(4B OOD 38.91 vs 42.46)——证明局部化在 \(e\) 之上还有独立的增益。**注:v1 曾记"缺全响应 vs 局部消融",本轮 PDF 复核 Table 4 实含此消融,予以更正。**【原文】§4.5、Table 4
  - **error localization dynamics(Fig.5)**:训练过程中,quote 匹配率约 **0.5**(8B 略高于 4B,这与它更强的指令遵循能力一致);归一化的错误位置随训练**逐渐右移**(错误变得更少、更靠后)。这正好解释了"为何全响应蒸馏到后期会变有害"——当早期推理已经大多正确时,继续蒸全响应只会无谓地扰动正确前缀。【原文】§4.6、Fig.5
  - **共享 base(student = teacher = reflector 同权重)**:这样设计排除了"更大的外部教师"这个混淆变量,使增益纯粹来自特权信息条件;而且 reflector 复用权重,不增加存储。【原文】§4.1
- **关键机制/公式(直觉)**:教师"偷看"了纠错思路 \(e\)(特权信息),学生只在"自己开始出错的地方(\(t\ge k\))"向这个更明白的教师对齐;出错之前的正确部分(\(t<k\),\(m_t=0\))保持不动。JSD(\(\alpha=0.5\))相比纯 reverse-KL 更对称、训练更稳;top-\(k=100\) 把蒸馏限制到教师高概率的 token 上,起降噪作用。【原文】Eq.9、附录 B
- **训推数据如何流动**:数据流是 rollout(vLLM 生成)→ 对错切分 → reflector 前向(同 base)产出 \((e,q)\)→ teacher 前向(条件于 \(e\),stopgrad)产出分布 → masked JSD 反传更新 FSDP actor。
  - 每个 prompt:1 次 rollout(8 条)+ 每条错误轨迹额外 1 次 reflector 前向 + 1 次自教师前向,所以推理成本高于纯 GRPO。
  - 但反思 prompt 上限 8k token、反思输出上限 4k token,且训练后期因为学生响应变短,反而比 baseline 更省时(附录 C)。
- **实验与证据**:backbone = Qwen3-4B / Qwen3-8B(student / teacher / reflector 同 base)。
  - 数据:**5 个数据集各单独训练**——SciKnowEval(L3)里化学 / 物理 / 生物 / 材料 4 个本科科学 QA + ToolAlpaca 工具调用(ToolUse);额外用 AIME2024 评数学。评测方式是"训一个域、评所有域"。每 prompt 采 8 rollout,评估时每题取 16 条报 mean@16。
  - **训练超参(附录 B)**:8×NVIDIA A800 80G;verl(FSDP actor + vLLM rollout);science 任务 10 epochs / ToolUse 5 epochs;batch size 32。
  - baseline:强化版 GRPO(DAPO 非对称 clip + Dr.GRPO 无偏归一化 + off-policy 校正)、SDPO。
  - 域内(Table 2):ROSD 平均 4B **72.83%**(较 GRPO / SDPO 分别 +2.97 / +5.81)、8B **73.45%**(+1.46 / +0.95,8B 上增益小)。逐域看,4B 上 Material 80.18、Physics 76.56、Chemistry 82.47、ToolUse 67.46,多为最佳。GRPO 与 SDPO 的域内平均很接近(4B 69.86 vs 67.02),说明 SDPO 的稠密监督并没明显超过 outcome-level RL(论文归因于 SDPO 引入了噪声监督)。【原文】§4.2、Table 2
  - OOD(Table 3):**这里有一个关键且诚实的发现——GRPO 的 OOD 鲁棒性最强**(因为它没有 token 级 KL 约束)。ROSD 介于 GRPO 与 SDPO 之间:在所有训练域、两个规模上都**稳超 SDPO**,但**多数设置下不超 GRPO**。
    - 极端跨域(ToolUse → 其他)时 SDPO 崩溃:4B OOD 均值 2.88%(ToolUse 列多处 0.49 / 0.83),ROSD 维持 41.31%(GRPO 43.57%);Chemistry 训练时 SDPO AIME2024 = 0.00、ROSD = 30.56。【原文】§4.3、Table 3
  - 训练动力学(Fig.3-4):ROSD 收敛快、全程稳;SDPO 则早期涨、后期降。Fig.4 还揭示 SDPO 在 Material 上的 rollout accuracy 到末期仍在涨、但 test score 反而下降(过拟合到自采 rollout)。【原文】§4.4
  - baseline 公平性:同数据划分 / verifier / eval prompt,且 GRPO 用的是强化实现,公平。【原文】§4.1
- **假设与失效边界**:
  - 【原文】\(q\) 定位依赖 reflector 抽出"精确子串"并能在 \(y^-\) 里匹配上;匹配失败就回退全响应(\(k=0\))。Fig.5 报匹配率约 0.5,即约一半错误轨迹其实退化成了全响应蒸馏。
  - 【原文】散度 = JSD(\(\alpha=0.5\),附录 B 明确;§3.2 也说"following Hübotter 2026; Li 2026a 用 token 级散度 with JSD")。〔注:v1 代码核查发现仓库 `core_algos.py` 由 `alpha` 配置:\(\alpha=1\) 是 reverse KL(SDPO 默认)、\(\alpha=0\) 是 forward KL、中间值为广义 JSD;论文报告的设置 \(\alpha=0.5\) 即对称 JSD,代码可切换。PDF 与代码无冲突。〕
  - 【原文·Limitations】方法仍**不及 GRPO 的 OOD**,作者承认跨域泛化需进一步加强;且需要在更广的 setting 上验证普适性。
  - 【推断】"第一处出错"这个假设默认是单点错误;一旦遇到多处分散错误、或错误前缀本身就无效的情况,"保留前缀"反而会保留错误前提。
  - 【推断】reflector 质量依赖同一 base,基座弱时反思可能不可靠;而且 8B 上相对 SDPO 的增益已经很小(域内 +0.95),方法收益随基座变强而衰减,能否外推到更大规模存疑。
- **祛魅总结**:
  - 真贡献:把两个直觉——"自教师该看什么(纠错思路 \(e\) 而非全解)"和"该改哪里(出错后缀,mask \(m_t\))"——用 reflection + quote 落了地,显著缓解了 OPSD 的 OOD 崩溃,而且不需要更大的外部教师;Table 4 的双消融 + Fig.5 的定位动力学构成了较完整的证据链。【推断】
  - 包装 / 需冷静看:**TL;DR / 摘要里的"显著改善 OOD"是相对 SDPO 而言的**。Table 3 显示 ROSD 在 OOD 上多数设置仍**不及最朴素的 GRPO**——也就是说,"加 token 级监督本身就会牺牲 OOD,ROSD 只是把这个代价压小了、没消除"。SDPO 在域内也没真正超过 GRPO,这更凸显了 OPSD 这条线整体收益的脆弱性。【推断·依据 Table 2/3 + Limitations】

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
- 🎯 对"探索-巩固"对标:**最强对标(直接竞品 + 高度可借)**。判定:ROSD 几乎就是本项目"教 student 两种能力"的一个文本实例:
  - 保留有效前缀(\(m_t=0,t<k\))≈ "偏向自己走得通的开头"(探索 / 选路);
  - 从第一处出错(\(k\))起、由"看了纠错思路 \(e\) 的自教师"接管纠错 ≈ path-recovery(走偏后恢复);
  - on-policy 自采样、自教师同 base 也契合"自选"。
  - 可直接借的:① error quote 做"第一处走偏点"定位的工程实现(`Locate` + mask);② "教师只给纠错思路 \(e\) 而非全解 \(c\)"以避免风格灌注。
  - 关键缺口 / Δ vs 本项目:
    - (a) ROSD 用 **JSD 全后缀蒸馏**(从 \(k\) 到 \(T\) 一整段),而本项目主张"**单点接管**"(更细的 step / token 级),不是整段后缀;
    - (b) 无 MTP 前瞻(本项目用 MTP 做 foresight probe 找切点,ROSD 靠 reflector 抽 quote);
    - (c) 无"固化进参数 / 记忆且不遗忘"的显式机制,只是"少改以保 OOD"。
  - 依据:§3.1-3.2 + Table 3-4。【推断】
- 🔭 开放问题/未来方向:【原文·Limitations】跨域泛化需进一步加强(仍不及 GRPO);更广 setting 验证普适性。【推断】quote 匹配失败(约半数)对结果的影响量化;多处错误/无效前缀的处理;把"整段后缀蒸馏"细化到"单点/关键步接管"(对接本项目);8B 以上规模收益衰减的成因;为何 ROSD 仍不及 GRPO 的 OOD——能否在保 token 监督的同时彻底追平。

RETURN: rosd|读到PDF?是(12页,_txt 46k字,Eq.1-9 + Table 4 消融 + 附录 B 超参全核)|L1|对标=最强对标:保留前缀=探索/出错后缀自教师接管=path-recovery,可借 error-quote 定位;Δ=整段后缀蒸馏(非单点接管)且无 MTP|残留待核 0(v1 标"缺局部化消融"已更正为 Table 4 实含;JSD α=0.5/topk=100/8×A800/batch32 均从附录 B 核到)

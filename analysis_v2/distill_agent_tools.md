distill_agent_tools | Distilling LLM Agent into Small Models with Retrieval and Code Tools (Agent Distillation) | KAIST / DeepAuto.ai(Minki Kang*, Seanie Lee, Sung Ju Hwang)· UW-Madison(Jongwon Jeong*)· KRAFTON(Jaewoong Cho) | 2025-11-05 · arXiv 2505.17612 v2 · NeurIPS 2025 | 主题线 L4(Agent/工具蒸馏)，兼 L1 · 相关性 高

**原始论文**：https://arxiv.org/abs/2505.17612

## 一眼看懂

> 一句话导读：以往蒸馏只教小模型"怎么一步步想(CoT)",可碰到要查冷门事实或做精确计算时,小模型照样瞎编 / 算错。本文改成连"怎么用工具(查检索、跑代码)"一起教——把老师 agent 完整的"想一步、做一个动作、看结果"的轨迹蒸给小模型,让 3B 以下的小 agent 也能打平甚至超过比它大 2–4 倍的纯 CoT 模型。

- 🟦 TL;DR：本文不只蒸馏教师的"推理(CoT)",而是把教师 **agent 完整的"think + act(检索 / 代码工具)"任务求解行为**蒸馏进小模型(sLM)。痛点很具体:纯 CoT 蒸馏出来的小模型,在需要罕见事实知识或精确计算时容易幻觉 / 算错(经典例子:"\$100 投资 Apple、2010→2020 值多少",这需要查历史股价 + 处理拆股 + 精确计算)。
  - 做法:让教师 agent 在 CodeAct 框架里产生一条"Thought–Action(code/search)–Observation"的轨迹,过滤掉答错的之后 SFT 进小模型;再配两项改进——**first-thought prefix(ftp)** 用来提高教师轨迹的质量,**self-consistent action generation(sag)** 用来提高学生测试时的鲁棒性。最终让 0.5B / 1.5B / 3B 的 agent 匹配、甚至超过比它大 2–4× 的 CoT 蒸馏模型。【原文 §Abstract / §4 / Fig.1–2】
- 最巧的一步：**first-thought prefix**——先用 CoT prompt 诱导教师产生第一步推理 \(y_1\),截取它当作前缀、注入教师 agent 的"第一步思考",再让教师从这里续走 reason-act-observe 循环(注意:学生推理时并不需要这个前缀)。
  - 为什么不能抽:instruction-tuned 的教师**直接当 agent 时会发生"分布漂移"**——agent 指令和它原本的 CoT 推理模式相冲突(见 §D.1:32B 当 agent 时在 MATH500 上反而比 CoT prompting 更差);而"最初的推理步会关键性地决定最终结论"[63–65],所以必须保证教师**开局就走对方向**。这一步正好对应本课题里的"探索 = 偏向自己能走通的开头"。【原文 §4 First-thought prefix / Eq.5】

## 为什么做

> 一句话导读：以前蒸馏给的都是"死的"推理示范——小模型只能照抄、不能在测试时上网查或算一遍。本文要让小模型学会"用工具去解决问题"这种活的行为;难点在于这种 agent 行为对师生双方都很陌生,所以专门加了两个针对小模型的补丁。

- 研究背景:sLM 的推理能力靠 CoT 蒸馏来迁移(也就是模仿教师的 step-by-step trace、做 next-token 拟合),这已经是后训练的标准件;再往前一步,是引入检索 / 代码工具,帮 sLM"学一套可迁移的策略、而不是死记"。但多数方法仍然停留在**静态 demonstration、没有环境交互**(原文 §2.1:"most existing approaches rely on static demonstrations and lack interaction with the environment")。本文的核心研究问题是:"how can we preserve LLM-level problem-solving ability in much smaller models?"【原文 §1 / §2.1】
- 解决的具体痛点:纯 CoT 蒸馏出来的小模型有两个毛病——(a)**幻觉 / 算错**(碰到需要新知识或精确计算时就失效,Fig.2);(b)静态 trace 没法在测试时去获取新知识、也没法保证计算正确。对应两个核心诉求:① 让 sLM 学"如何用工具去 reason+act",而不是去记知识 / 记算术;② 对那些需要未见事实 / 计算的新 query,泛化得更好。【原文 §1】
- 相关工作 & 各自不足(原文 §2,逐条点名)：
  - **CoT 蒸馏 / 工具增强 CoT**:训 sLM 去复现 step-by-step trace [16,17,24–27];较新的工作加了检索 [19,28,29] 或代码执行 [30–32] 来帮它学可迁移策略——**但仍是静态 trace、没有环境交互、容易死记**。
  - **语言 agent(ReAct [20] / CodeAct [21])**:做 observe-think-act 交互,但大多靠为强 LLM 设计的 prompt(few-shot),或者**在强 LLM 的轨迹上微调一个较大(≥7B)的模型**(例如 FireAct [34] 在 GPT-4 轨迹上微调)。
  - **agentic 检索 / 工具推理**:这条线里,agentic retrieval 支持多跳 [48–50],tool-augmented reasoning 用代码解数学 [51–55],agentic reasoning 靠 prompt / RL 增强规划 [56–58]。
  - **共性不足(本文的 Δ)**:此前的 agent 微调主要针对 **≥7B** 的模型,**把 agentic 能力蒸进 ≤3B 的小模型,是一个还没被充分探索的设定**;而且要解决两个专门针对小模型的新挑战——"教师轨迹的质量"和"学生测试时的鲁棒性"。【原文 §2.2 末】
- 动机链:现状是 CoT 蒸馏小模型;缺陷在于罕见事实 / 精确计算上会幻觉 / 算错,而且静态 trace 没法在测试时补救;所以本文改成蒸"完整的任务求解行为(think+act)",让 sLM 用工具去查事实 / 算数、而不是去记。但这又带来两个子挑战:① agentic 行为对教师和学生都是 OOD 的,可能反伤它们已经优化好的 CoT 能力;② sLM 很难产生可执行的代码,常常格式错误 / 误用库函数。两项改进就是针对这两点——ftp 修教师轨迹质量,sag 修学生测试鲁棒性。【原文 §1 / §4 两挑战】
- 与最近邻工作的 Δ:相对 **CoT 蒸馏**——它迁移的是"用工具的行为"而非"静态推理",所以 OOD 泛化更好;相对 **FireAct 等 agent 微调**——它的目标是一个**远更小**的学生(≤3B),并在工具调用上配了 ftp / sag 这两个针对小模型的工程修补。差异点就在"小模型 + 工具行为 + 两项稳定化"。【原文 §2.2 末】

## 怎么做 + 靠不靠谱

> 一句话导读：核心 loss 其实就是"在教师的工具轨迹上做 next-token SFT,但环境返回的 observation 不算 loss"(Eq.4)。剩下两个关键件:ftp 在采集教师数据时让它开局走对(Eq.5),sag 在学生测试时多采几条、过滤报错、再投票挑最一致的动作。

- **方法流水线(可复现级)**,一条龙是这样:
  1. 输入训练问题 \(x\);
  2. 教师(Qwen2.5-32B-Instruct)在 smolagents / CodeAct 里采一条"(r,a,o) 循环"的轨迹(用 ftp 增强);
  3. 过滤掉答错的轨迹,得到约 2,000 条;
  4. 用 TRL 的 SFT trainer + LoRA(rank 64,全线性层)训学生(**注意 observation 不计 loss**);
  5. 测试时给学生 agent 配上 sag(每一步采 N=8 条 thought-action、温度 0.4,过滤掉报错的,再对 observation 投票);
  6. 输出一个工具型的小 agent。【原文 §4 / §5 / Eq.3–5】
- **核心目标的真实形式 + 直觉**：
  - 一般 KD(Eq.1):\(\min_\theta\mathbb{E}_{(x,y)\sim\mathcal{D}_{\text{train}}}\tfrac{1}{L_y}\sum_{n=1}^{L_y}D\big(p_T(\cdot\mid y_{<n},x)\,\|\,p_S(\cdot\mid y_{<n},x;\theta)\big)\),\(D\) 为 KL/JSD。
  - 推理(CoT)蒸馏(Eq.2):用 CoT prompt \(I_{\mathrm{CoT}}\)(如 "Let's think step by step")让教师生成 rationale,学生 next-token 模仿 \(\min_\theta -\mathbb{E}_{x,\,y\sim p_T(\cdot\mid x,I_{\mathrm{CoT}})}\sum_{n=1}^{L_y}\log p_S(y_n\mid x,y_{<n};\theta)\)。
  - **agent 轨迹(Eq.3)**:在 agent 指令 \(I_{\text{agent}}\)(如 "...in a cycle of Thought:, Code:, and Observation: sequences")下,教师生成 \(\tau=((r_1,a_1,o_1),\dots,(r_{L_\tau},a_{L_\tau},o_{L_\tau}))\sim p_T(\cdot\mid x,I_{\text{agent}})\),其中每个 observation \(o\) 来自**环境对 action 的响应,非模型生成**。
  - **agent 蒸馏 loss(Eq.4,关键:排除 observation)**:\(\displaystyle \min_\theta\;-\mathbb{E}_{x,\,\tau\sim\pi_T(\cdot\mid x,I_{\text{agent}})}\sum_{t=1}^{L_\tau}\log p_S\big(r_t,a_t\mid x,\tau_{<t};\theta\big),\qquad \tau_{<t}=((r_1,a_1,o_1),\dots,(r_{t-1},a_{t-1},o_{t-1})).\)直觉:\(o\) 来自环境、非模型该"生成"的内容,故只学 (thought, action)、observation 排除在 loss 外。
  - **first-thought prefix(Eq.5)**:\(\displaystyle y_1\sim p_T(\cdot\mid x,I_{\mathrm{CoT}}),\qquad \tau=\{(r_1',a_1,o_1),\dots,(r_{L_\tau},a_{L_\tau},o_{L_\tau})\}\sim p_T(\cdot\mid x,y_1,I_{\text{agent}}),\)即先用 CoT prompt 取教师**第一步推理 \(y_1\)**,把它当前缀让教师 agent 续走、补完第一思考 \(r_1'\)。灵感来自 jailbreak 的 **prefix-attack [66–68]**(往回答前缀注入内容引导生成方向)。**仅用于教师轨迹采集;学生推理时不需要**。
  - **self-consistent action generation(sag)**:学生 agent 在每一步不用 greedy 解码,而是改成这样几步——先用高温 nucleus 采样出 \(N\) 条 thought-action;再用一个轻量的代码解释器**把解析 / 执行报错的那些过滤掉**;如果全都失败,就保留一条、把报错信息喂回去当 observation,供后面自我纠正;最后对剩下那些结果的 observation 做 **majority voting**,挑出最一致的那个 action。一句话直觉就是:"教师靠 ftp 保证开局走对 + 学生靠 sag 在测试时多采几条、投票纠错"。【原文 §4 / Eq.1–5 / Fig.3】
- 逐组件必要性(消融见 Table 2 "Distill→+ftp/+sag/+ftpsag")：
  - **agent 蒸馏(核心)**：没它(纯 CoT 蒸馏)在 OOD 上明显更弱;且蒸馏前除 7B 外多数尺寸靠 prompting **无法产生有效 agentic 输出**(常出不可解析代码,0.5B 仅 1.80、1.5B 7.94)。【原文 §6 Overall / Table 2】
  - **first-thought prefix(ftp)**：只动教师数据采集、提高轨迹质量。Table 2 多数尺寸 +ftp 抬平均分(7B 39.85→42.26、3B 33.60→34.49)。【原文 §4 / Table 2】
  - **self-consistent action generation(sag)**：只动学生测试。多数尺寸 +sag 抬平均(1.5B 28.06→30.29),但**非处处增益**——3B+sag 在 AIME 掉到 **0.0**(单用 sag 在个别难任务反伤)。【原文 §4 / Table 2】
  - ftp 与 sag **互不耦合**(一个改教师采集端、一个改学生推理端),消融清晰;ftpsag 合用通常最好。【原文 §4 / §6】
- **关键超参默认值(可直接照搬)**：
  - 训练:4×A100 80GB、LoRA rank **64**(全线性层)、**2 epoch**、batch **8**、lr **2e-4**、约 2,000 条轨迹(每题采 1 条、过滤掉错误的);教师是 Qwen2.5-32B-Instruct,学生是 Qwen2.5-Instruct 的 0.5/1.5/3/7B(都已先做过 instruction-tuned)。
  - 推理:基线用 greedy、max steps **5**;sag 主实验取 **N=8、温度 0.4**。
  - 检索环境:**Wikipedia 2018** + **e5-base-v2**(沿用 Search-R1 [49],agent 与 RAG 共用)。
  - 指标:数学用 exact match,事实用 LLM-as-judge(gpt-4o-mini)。【原文 §5】
- 实验与证据：
  > 一句话导读：最抓眼的结论是"跨档匹配"——小一两档的工具 agent 能打平甚至超过大的纯 CoT 模型。但要打两个折:这个结论靠的是 8 个异质任务的平均分(单任务波动很大),而且 sag 的增益部分其实是 test-time 多采样换来的算力,不全是模型本身变强。
  - 训练数据:1,000 HotPotQA + 2,000 MATH,每题采 1 条教师轨迹、过滤错误后约 2,000 条。【原文 §5】
  - 评测(8 任务,每集 ≤500):事实多跳 QA(HotpotQA in-domain;MuSiQue/Bamboogle/2WikiQA OOD)+ 数学(MATH500 in-domain;GSM-Hard/AIME/OlymMATH OOD),各 1 in-domain + 3 OOD 测泛化。【原文 §5 / Table 1】
  - 关键数字(Table 2 Avg.):教师 32B CoT 39.54 / Agent 46.00。**跨档匹配**:7B Agent(ftpsag) **42.68 > 32B CoT 39.54**;3B Agent(ftpsag) **36.60 > 7B CoT 33.19/Distill 33.54**;1.5B Agent(ftpsag) 30.55 ≈ 3B CoT;0.5B Agent(ftpsag) 21.90 > 1.5B CoT。原文总结:"0.5B agent matches 1.5B CoT, 1.5B reaches 3B, 3B surpasses 7B CoT, 7B even outperforms 32B CoT"。【原文 §6 / Table 2 / Fig.1】
  - baseline 公平吗:对 CoT 蒸馏额外加了 **Distill+RAG** 基线(同样给外部知识)以公平对比工具增益,较规范;但**跨档匹配高度依赖 8 个异质任务的平均聚合**——单任务结论不稳(AIME/OlymMATH 小样本方差大,出现 0.0 等剧烈跳变)。【原文 §5 / §6 / Table 2】
  - "看着强但没回答核心问题":sag 的提升部分来自**测试时多次采样的额外算力**(N=8),而非模型本身能力——与"高效小 agent"卖点有张力(推理成本被转移到 test-time)。【推断,依据 §4 sag 机制 + N=8】
- 假设与失效边界：
  - 【原文 §4 两挑战】agentic 行为相对预训练/指令微调分布 **OOD**——蒸它可能反伤学生已优化的 CoT 能力(Table 2 个别单元格 Distill < CoT);sLM 难产生可执行代码(misformatted/误用库函数)。
  - 【推断】**离线 SFT、非 on-policy**:纯模仿教师轨迹,学生在工具调用上的分布漂移未在训练时被纠正(只靠 test-time sag 兜底)。依据:Eq.4 是标准 next-token SFT(对教师轨迹的 (r,a) 做 MLE)。
  - 【推断】**教师/检索环境固定**(仅 Wikipedia 2018 + 单一 retriever e5-base-v2),真实开放检索下表现未验证;事实任务依赖 gpt-4o-mini 判分,引入评测器偏差;小样本基准(AIME/OlymMATH)方差大、Avg 聚合掩盖单任务剧烈波动。【原文 §5】
- 祛魅总结【推断】：
  - 真贡献:把"蒸推理"扩展到"蒸完整工具使用行为",并用两个**正交、可消融**的小修补(ftp 改教师采集、sag 改学生测试)让 ≤3B 小模型可用——工程组合干净,"小 agent 可匹配更大 CoT 模型"在平均意义上成立,对实用小 agent 有价值。
  - 包装/被高估处:"跨档匹配"依赖平均分聚合,统计稳健性有限(单任务可剧烈反复);sag 的增益本质是 test-time 多采样投票(算力换准确率),部分抵消"省算力"卖点;纯离线 SFT,未做 on-policy/RL 矫正(这正是与本课题 OPD 的最大差距)。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：教师 agent 的 (thought, action) 序列(observation 不计 loss);序列级 next-token SFT(非散度/非 logits)。
  - **改什么**：改**训练数据形态**(静态 CoT trace → 交互式 reason-act-observe 轨迹)+ 教师采集策略(ftp)+ 学生测试解码(sag);学生用 LoRA 微调。
  - **何时改**：训练时蒸轨迹(ftp 在教师采集端);测试时 sag(学生推理端,与训练解耦)。
  - **免梯度?**：是离线 SFT(next-token MLE,Eq.4),无 RL/策略梯度/奖励;sag 是纯推理时采样投票。
  - **记忆-技能生命周期**：技能=工具使用行为,**外化到工具(检索=外部知识、代码解释器=精确计算)** 而非压进参数——这点很关键:它主动选择"不把事实/计算记进权重,而是学会用工具取"。无显式技能库/记忆库的持久积累,工具是无状态调用。
  - **防遗忘机制**：无显式防遗忘;反而注意 §4 警示——蒸 agentic 行为可能**反伤**学生原有 CoT 能力(一种诱发遗忘的风险),未给缓解机制。
- ⑦ 开源代码+框架/harness：https://github.com/Nardien/agent-distillation (本地已克隆 ~7.6MB)。**框架=smolagents v1.13.0.dev0(fork) + TRL SFT trainer + LoRA(rank 64)**;检索沿用 **Search-R1**(pyserini + faiss-gpu),代码工具走本地或 e2b 沙箱(`e2b.toml`)。含 `data_processor/{math,qa}`、`exps_research/first_thought_prefix`、`scripts/{training,inference}`、`serve_vllm.py`。HF `agent-distillation/*` 提供教师轨迹(2k baseline + ftp 两版)与 agent-distilled Qwen2.5-1.5B 等权重。代码可得性高(README TODO 标注 ftp 详细说明待补)。【本地仓库核查】
- 💰 资源/成本与可扩展性：训练 4×A100 80GB、LoRA rank 64、2 epoch、batch 8、lr 2e-4、~2,000 轨迹(数据量极小)。推理:max steps=5,sag 主实验 N=8/温度 0.4(test-time 成本随 N 上升)。学生尺度 0.5B–7B、教师 32B。【原文 §5】
- 🎯 对"探索-巩固"对标：**强对标(L4 agent 蒸馏的直接对照基线)+ 多个可借组件,但范式相反(离线 SFT vs on-policy)**。判定依据:
  - **支撑/同构**:① **first-thought prefix ≈ 本课题"探索:偏向自己能走通的开头"**——ftp 用"开局走对方向决定全局"的洞察(援引 [63–65]"初始步关键")确保轨迹质量,与"teacher 当稀疏脚手架在开头给方向、student 续走"高度同构,且为 MTP 前瞻探针/选路提供文献依据。② **sag 的"采多条→过滤无效→投票一致"** 与"走偏后自选恢复分支"有相通处(用执行反馈 + 一致性筛掉走不通的分支)。③ **工具外化(查事实/算数不进参数)** 对"巩固"的边界划定有启发:并非所有能力都该固化进权重。
  - **竞品/缺口(关键)**:这是**离线 SFT、纯行为克隆**(Eq.4)——正是本课题 OPD 要超越的"非 on-policy"基线。它**没有**:① on-policy student 自选(教师轨迹固定,学生分布漂移只靠 test-time sag 兜底、未在训练时纠正);② 真正的"巩固进参数/记忆/技能库且防遗忘"(反而有反伤 CoT 的遗忘风险);③ MTP/前瞻探针(ftp 是 prompt 前缀技巧,非可学的前瞻);④ 稀疏关键步介入(ftp 只在第一步、sag 每步全采,非"关键步单点接管")。
  - 一句判定:**最值得对标的 L4 工作**——ftp/工具外化可直接借鉴到 MTP+OPD 的"探索/脚手架"侧;但它停在离线 SFT,本课题的增量正应在"把 ftp 式开局引导 + sag 式回轨筛选**做成 on-policy、可学、稀疏关键步介入**"。
- 🔭 开放问题/未来方向：
  - 【原文】把 agent 蒸馏推广到更多工具/更大开放检索;提升小 agent 生成可执行代码的可靠性。
  - 【推断】把离线轨迹蒸馏升级为 **on-policy agent 蒸馏**(学生自 rollout + 教师在关键步/走偏处稀疏纠正),把 ftp 的"开局引导"与 sag 的"一致性筛选"做成训练时可学信号而非纯推理时技巧——这正是本课题"探索-巩固"对 agent 自进化的直接延伸点;并研究蒸 agentic 行为时如何防止反伤原 CoT(防遗忘)。

RETURN: distill_agent_tools | 读PDF=是(§1–6 全文 + Eq.1–5/Table1–2 全量数字/Fig.1–3/两挑战/§D.1 教师当agent掉分) | 加厚=是(新增 KD Eq.1、CoT蒸馏 Eq.2、agent轨迹 Eq.3、agent蒸馏loss Eq.4、ftp Eq.5 全 MathJax;related work逐条点名 CoT蒸馏[16–32]/ReAct[20]/CodeAct[21]/FireAct[34];补 prefix-attack 来源与超参默认值) | LaTeX公式=5条(KD Eq.1、CoT Eq.2、agent轨迹Eq.3、agent蒸馏loss Eq.4、ftp Eq.5) | 待核=0

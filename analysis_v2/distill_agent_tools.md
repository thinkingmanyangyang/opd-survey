distill_agent_tools | Distilling LLM Agent into Small Models with Retrieval and Code Tools (Agent Distillation) | KAIST / DeepAuto.ai(Minki Kang*, Seanie Lee, Sung Ju Hwang)· UW-Madison(Jongwon Jeong*)· KRAFTON(Jaewoong Cho) | 2025-11-05 · arXiv 2505.17612 v2 · NeurIPS 2025 | 主题线 L4(Agent/工具蒸馏)，兼 L1 · 相关性 高

**原始论文**：https://arxiv.org/abs/2505.17612

## 一眼看懂
- 🟦 TL;DR：不只蒸馏教师的"推理(CoT)"，而是把教师 **agent 的完整"think + act(检索/代码工具)"任务求解行为**蒸馏进小模型(sLM)。痛点:纯 CoT 蒸馏在需要罕见事实知识或精确计算时小模型易幻觉/算错(经典例:"$100 投资 Apple 2010→2020 值多少"需查历史股价+拆股+精确算)。做法:让教师 agent 在 CodeAct 框架里产生"Thought-Action(code/search)-Observation"轨迹，过滤错误后 SFT 进小模型；配两项改进——**first-thought prefix(ftp)** 提高教师轨迹质量、**self-consistent action generation(sag)** 提高学生测试时鲁棒性——使 0.5B/1.5B/3B agent 匹配甚至超过比它大 2–4× 的 CoT 蒸馏模型。【原文 §Abstract/§4/Fig.1-2】
- 最巧的一步：**first-thought prefix**(用 CoT prompt 诱导教师产生第一步推理，截取作前缀注入教师 agent 的"第一步思考"，再让教师续走 reason-act-observe；学生推理时不需要此前缀)。抽掉它教师轨迹质量就垮——论文动机:instruction-tuned 教师直接当 agent 时会有"分布漂移"(agent 指令与其原 CoT 推理模式冲突)，而"初始推理步关键决定最终结论"[63-65]，所以必须确保教师**开局走对方向**。这一步直接对应本课题"探索=偏向自己能走通的开头"。【原文 §4 First-thought prefix/Eq.5】

## 为什么做
- 研究背景：sLM 推理能力靠 CoT 蒸馏(模仿教师 step-by-step trace，next-token 拟合)迁移，已是后训练标准件；进一步引入检索/代码工具帮 sLM "学可迁移策略而非死记"。但多数方法仍是**静态 demonstration、无环境交互**。【原文 §1/§2.1】
- 解决的具体痛点：纯 CoT 蒸馏的小模型 (a) **幻觉/算错**(需新知识或精确计算时失效，Fig.2)；(b) 静态 trace 无法在测试时获取新知识或保证计算正确。核心问题:如何在远小的模型里保留 LLM 级问题求解能力。【原文 §1】
- 相关工作 & 各自不足：
  - **CoT 蒸馏 / RAG**：静态 trace 或加检索，仍不交互、易死记。【原文 §2.1】
  - **语言 agent(ReAct/CodeAct)**：观察-思考-行动，但多靠为强 LLM 设计 prompt、或在强 LLM 轨迹上微调**较大(≥7B)** 模型(如 FireAct 在 GPT-4 轨迹上微调)。【原文 §2.2】
  - 共性不足:此前 agent 微调主要针对 ≥7B 模型，**把 agentic 能力蒸进 ≤3B 小模型是未充分探索的设定**。【原文 §2.2 末】
- 动机链：现状(CoT 蒸馏小模型)→ 缺陷(罕见事实/精确计算上幻觉/算错；静态 trace 不能测试时补救)→ 所以(蒸"完整任务求解行为 think+act"，让 sLM 用工具查事实/算数而非记)→ 子挑战(① agentic 行为对教师/学生都 OOD，可能反伤已优化的 CoT 能力；② sLM 难产生可执行代码)→ 两项改进(ftp 修教师轨迹质量、sag 修学生测试鲁棒)。【原文 §1/§4 两挑战】
- 与最近邻工作的 Δ：相对 **CoT 蒸馏**——迁移的是"用工具的行为"而非"静态推理"，OOD 泛化更好。相对 **FireAct 等 agent 微调**——目标是**远更小**的学生(≤3B)且在工具调用上配 ftp/sag 两个针对小模型的工程修补。差在"小模型 + 工具行为 + 两项稳定化"。【原文 §2.2 末】

## 怎么做 + 靠不靠谱
- 方法流水线：输入(训练问题 x) → 教师(Qwen2.5-32B-Instruct)在 smolagents/CodeAct 中按 Eq.3 采一条"(r,a,o) 循环"轨迹(ftp 增强:先 CoT 诱导首步 y1 当前缀，Eq.5) → 过滤答错轨迹得 ~2,000 条 → 用 TRL SFT trainer + LoRA(rank 64)训学生(Eq.4，**observation 不计 loss**) → 测试时学生 agent 配 sag(每步采 N=8 条 thought-action、温度 0.4、过滤报错、对 observation 投票) → 输出工具型小 agent。【原文 §4/§5/Eq.3-5】
- 逐组件必要性(消融见 Table 2 "Distill→+ftp/+sag/+ftpsag")：
  - **agent 蒸馏(核心)**：没它(纯 CoT 蒸馏)在 OOD 上明显更弱；且蒸馏前除 7B 外多数尺寸靠 prompting **无法产生有效 agentic 输出**(常出不可解析代码)。【原文 §6 Overall/Table 2】
  - **first-thought prefix(ftp)**：只动教师数据采集、提高轨迹质量。Table 2 上多数尺寸 +ftp 抬平均分(如 7B 39.85→42.26、3B 33.60→34.49)。【原文 §4/Table 2】
  - **self-consistent action generation(sag)**：只动学生测试。多数尺寸 +sag 抬平均(如 1.5B 28.06→30.29)，但**非处处增益**——3B+sag 在 AIME 掉到 **0.0**(单用 sag 在个别难任务反伤)。【原文 §4/Table 2】
  - ftp 与 sag **互不耦合**(一个改教师采集、一个改学生推理)，消融清晰；ftpsag 合用通常最好。【原文 §4/§6】
- 关键机制/公式(直觉)：蒸馏目标 Eq.4 把 observation 排除在 loss 外(o 来自环境、非模型生成，不该学着"生成"它)，只学 (thought, action)。ftp(Eq.5):y1~教师CoT(首步)，τ~教师agent(以 y1 为前缀续走)——灵感来自 jailbreak 的 prefix-attack(往回答前缀注入内容引导生成)。sag:高温 nucleus 采样增多样性 → 过滤执行/解析报错(全失败则留一条把报错喂回当 observation 供后续自纠)→ 对结果 observation 做 majority voting。直觉:"教师开局走对(ftp)+ 学生测试时多采几条投票纠错(sag)"。【原文 §4/Eq.4-5】
- 实验与证据：
  - 训练数据:1,000 HotPotQA + 2,000 MATH，每题采 1 条教师轨迹、过滤错误后 ~2,000 条。检索环境:Wikipedia 2018 + e5-base-v2(沿用 Search-R1 retriever，agent 与 RAG 共用)。【原文 §5】
  - 评测(8 任务，每集 ≤500):事实多跳 QA(HotpotQA in-domain；MuSiQue/Bamboogle/2WikiQA OOD)+ 数学(MATH500 in-domain；GSM-Hard/AIME/OlymMATH OOD)。指标:数学 exact match，事实 LLM-as-judge(gpt-4o-mini)。【原文 §5/Table 1】
  - 关键数字(Table 2 Avg.):教师 32B CoT 39.54 / Agent 46.00。**跨档匹配**:7B Agent(ftpsag) **42.68 > 32B CoT 39.54**；3B Agent(ftpsag) **36.60 > 7B CoT 33.19/Distill 33.54**；1.5B Agent(ftpsag) 30.55 ≈ 3B CoT(27.72)甚至更高；0.5B Agent(ftpsag) 21.90 > 1.5B CoT(21.28)。蒸馏前 agent prompting 多数尺寸极弱(0.5B 仅 1.80、1.5B 7.94)。【原文 §6/Table 2】
  - baseline 公平吗:对 CoT 蒸馏额外加了 **Distill+RAG** 基线(同样给外部知识)以公平对比工具增益，较规范；但**跨档匹配高度依赖 8 个异质任务的平均聚合**——单任务结论不稳(AIME/OlymMATH 小样本方差大，出现 0.0/15.6 剧烈跳变)。【原文 §5/§6/Table 2】
  - "看着强但没回答核心问题":sag 的提升部分来自**测试时多次采样的额外算力**(N=8)，而非模型本身能力——与"高效小 agent"卖点有张力(推理成本被转移到 test-time)。【推断，依据 §4 sag 机制 + N=8】
- 假设与失效边界：
  - 【原文 §4 两挑战】agentic 行为相对预训练/指令微调分布 **OOD**——蒸它可能反伤学生已优化的 CoT 能力(Table 2 个别单元格 Distill < CoT，如 3B HotPotQA 38.6→26.8)；sLM 难产生可执行代码(misformatted/误用库函数)。
  - 【推断】**离线 SFT、非 on-policy**:纯模仿教师轨迹，学生在工具调用上的分布漂移未在训练时被纠正(只靠 test-time sag 兜底)。依据:Eq.4 是标准 next-token SFT。
  - 【推断】**教师/检索环境固定**(仅 Wikipedia 2018 + 单一 retriever)，真实开放检索下表现未验证；事实任务依赖 gpt-4o-mini 判分，引入评测器偏差；小样本基准(AIME/OlymMATH)方差大、Avg 聚合掩盖单任务剧烈波动。【原文 §5】
- 祛魅总结【推断】：
  - 真贡献:把"蒸推理"扩展到"蒸完整工具使用行为"，并用两个**正交、可消融**的小修补(ftp 改教师采集、sag 改学生测试)让 ≤3B 小模型可用——工程组合干净，"小 agent 可匹配更大 CoT 模型"在平均意义上成立，对实用小 agent 有价值。
  - 包装/被高估处:"跨档匹配"依赖平均分聚合，统计稳健性有限(单任务可剧烈反复)；sag 的增益本质是 test-time 多采样投票(算力换准确率)，部分抵消"省算力"卖点；纯离线 SFT，未做 on-policy/RL 矫正(这正是与本课题 OPD 的最大差距)。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：教师 agent 的 (thought, action) 序列(observation 不计 loss)；序列级 next-token SFT(非散度/非 logits)。
  - **改什么**：改**训练数据形态**(静态 CoT trace → 交互式 reason-act-observe 轨迹)+ 教师采集策略(ftp)+ 学生测试解码(sag)；学生用 LoRA 微调。
  - **何时改**：训练时蒸轨迹(ftp 在教师采集端)；测试时 sag(学生推理端，与训练解耦)。
  - **免梯度?**：是离线 SFT(next-token MLE)，无 RL/策略梯度/奖励；sag 是纯推理时采样投票。
  - **记忆-技能生命周期**：技能=工具使用行为，**外化到工具(检索=外部知识、代码解释器=精确计算)** 而非压进参数——这点很关键:它主动选择"不把事实/计算记进权重，而是学会用工具取"。无显式技能库/记忆库的持久积累，工具是无状态调用。
  - **防遗忘机制**：无显式防遗忘；反而注意 §4 警示——蒸 agentic 行为可能**反伤**学生原有 CoT 能力(一种诱发遗忘的风险)，未给缓解机制。
- ⑦ 开源代码+框架/harness：https://github.com/Nardien/agent-distillation (v1 记已克隆 ~7.6MB)。**框架=smolagents v1.13.0.dev0(fork) + TRL SFT trainer + LoRA(rank 64)**;检索沿用 **Search-R1**(pyserini + faiss-gpu)，代码工具走本地或 e2b 沙箱(`e2b.toml`)。含 `data_processor/{math,qa}`、`exps_research/first_thought_prefix`、`scripts/{training,inference}`、`serve_vllm.py`。HF `agent-distillation/*` 提供教师轨迹(2k baseline + ftp 两版)与 agent-distilled Qwen2.5-1.5B 等权重。代码可得性高(README TODO 标注 ftp 详细说明待补)。【v1 仓库核查】
- 💰 资源/成本与可扩展性：训练 4×A100 80GB、LoRA rank 64、2 epoch、batch 8、lr 2e-4、~2,000 轨迹(数据量极小)。推理:max steps=5，sag 主实验 N=8/温度 0.4(test-time 成本随 N 上升)。学生尺度 0.5B-7B、教师 32B。【原文 §5】
- 🎯 对"探索-巩固"对标：**强对标(L4 agent 蒸馏的直接对照基线)+ 多个可借组件，但范式相反(离线 SFT vs on-policy)**。判定依据:
  - **支撑/同构**:① **first-thought prefix ≈ 本课题"探索:偏向自己能走通的开头"**——ftp 用"开局走对方向决定全局"的洞察确保轨迹质量，与"teacher 当稀疏脚手架在开头给方向、student 续走"高度同构(且 ftp 论证了"初始步关键"[63-65]，为 MTP 前瞻探针/选路提供文献依据)。② **sag 的"采多条→过滤无效→投票一致"** 与"走偏后自选恢复分支"有相通处(用执行反馈+一致性筛掉走不通的分支)。③ **工具外化(查事实/算数不进参数)** 对"巩固"的边界划定有启发:并非所有能力都该固化进权重。
  - **竞品/缺口(关键)**:这是**离线 SFT、纯行为克隆**——正是本课题 OPD 要超越的"非 on-policy"基线。它**没有**:① on-policy student 自选(教师轨迹固定，学生分布漂移只靠 test-time sag 兜底、未在训练时纠正)；② 真正的"巩固进参数/记忆/技能库且防遗忘"(反而有反伤 CoT 的遗忘风险)；③ MTP/前瞻探针(ftp 是 prompt 前缀技巧，非可学的前瞻)；④ 稀疏关键步介入(ftp 只在第一步、sag 每步全采，非"关键步单点接管")。
  - 一句判定:**最值得对标的 L4 工作**——ftp/工具外化可直接借鉴到 MTP+OPD 的"探索/脚手架"侧；但它停在离线 SFT，本课题的增量正应在"把 ftp 式开局引导 + sag 式回轨筛选**做成 on-policy、可学、稀疏关键步介入**"。
- 🔭 开放问题/未来方向：
  - 【原文】把 agent 蒸馏推广到更多工具/更大开放检索；提升小 agent 生成可执行代码的可靠性。
  - 【推断】把离线轨迹蒸馏升级为 **on-policy agent 蒸馏**(学生自 rollout + 教师在关键步/走偏处稀疏纠正)，把 ftp 的"开局引导"与 sag 的"一致性筛选"做成训练时可学信号而非纯推理时技巧——这正是本课题"探索-巩固"对 agent 自进化的直接延伸点；并研究蒸 agentic 行为时如何防止反伤原 CoT(防遗忘)。

RETURN: distill_agent_tools|读到PDF=是(全文+Eq.1-5/Table1-3全量数字/Fig.1-3/两挑战)|L线=L4(兼L1)|对标=强对标基线(ftp≈"走通的开头"、sag≈回轨筛选、工具外化可借;但离线SFT非on-policy、无稀疏关键步介入、有反伤CoT风险——本课题增量正在此)|残留待核=0

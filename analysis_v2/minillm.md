minillm | MiniLLM: On-Policy Knowledge Distillation of Large Language Models | 清华 CoAI(Yuxian Gu,微软实习)+ 微软研究院(Li Dong/Furu Wei),Minlie Huang 通讯 | 2023-06 首发,持续更新至 2026-01(arXiv v6);ICLR 2024 | 主题线 L1 OPD/自蒸馏·相关性 高

**原始论文**:https://arxiv.org/abs/2306.08543

## 一眼看懂
- 🟦 TL;DR:标准知识蒸馏(KD)让小学生用 forward-KL 去"覆盖"大教师分布的所有模式,但学生容量小、覆盖不了,就会把概率塞到教师几乎没概率的区域,自由生成时吐垃圾。MiniLLM 改成 **reverse-KL**(让学生"挑大模式、忽略长尾"),并用**策略梯度做 on-policy 优化**——即在学生**自己采样**的输出上,用教师打分当奖励来更新学生。配三招稳定化,在 120M~13B、GPT-2/OPT/LLaMA 三族上一致超过 SFT/word-KD/SeqKD。【原文 Abstract, §2.1】
- 最巧的一步:把 KD 目标从 forward-KL 换成 **reverse-KL**(§2.1)。抽掉它,整个"学生在自己分布上挑教师认可的主模式"的 mode-seeking 行为就没了,退化成普通模仿(SeqKD/word-KD),长文本就会出现暴露偏差和低质生成。reverse-KL 的另一面是它天然写成"学生采样 + 教师当 reward"的策略梯度形式(§2.2),这正是 on-policy 蒸馏的来源。

## 为什么做
- 研究背景:LLM 太大难部署,KD 是主流压缩手段。已有 KD 分两类——black-box(只拿到教师生成的文本,如 Alpaca/Vicuna 蒸 ChatGPT)和 white-box(能拿到教师的输出分布/隐状态)。white-box KD 过去只在 <1B 的分类/理解模型上研究,对**生成式 LLM 的 white-box KD 尚属空白**。【原文 §1, §4】
- 解决的具体痛点:标准 KD 本质在最小化 forward-KL `KL[p‖q_θ]`,它逼学生覆盖教师 p 的所有模式;分类任务模式少没问题,但**开放式生成**里 p 模式极多、低容量学生 q 覆盖不了,于是 q 在 p 的"空白区(void region)"被赋了不合理的高概率 → 自由生成时产出"在 p 下极不可能"的样本(低质/幻觉)。【原文 §1, §2】
- 相关工作 & 各自不足:(1) SeqKD/word-KD——本质 forward-KL,导致上面的过覆盖问题;(2) 分类模型蒸馏(DistilBERT/TinyBERT/MiniLM)——只到理解任务,没解决生成;(3) 并行工作 GKD(Agarwal 2023)、f-divergence KD(Wen 2023)——也在探索分布差异度量,但本文聚焦 LLM+reverse-KL+完整 on-policy 算法。【原文 §4】
- 动机链:LLM 部署贵 → 要 white-box KD → 但标准 forward-KL 对生成式模型次优(过覆盖空白区) → 改用 reverse-KL 让学生挑主模式 → reverse-KL 的梯度天然是 on-policy 策略梯度 → 但策略梯度高方差+reward hacking+偏好短句 → 加三招稳定化。
- 与最近邻工作的Δ:相比并行的 GKD(也用学生自采样 + 可调散度),MiniLLM 的关键差在**显式论证 reverse-KL 的 mode-seeking 对"真实性/可靠性"的价值**,并给出**三项让 PG 真正能跑稳**的工程化分解(单步分解降方差、教师混采防 hacking、长度归一去短句偏好),且做了 120M→13B 的跨族规模验证。

## 怎么做 + 靠不靠谱
- 方法流水线:① 学生先在任务数据 D 上做 SFT,取验证损失最低的 checkpoint 作初始化 → ② 进入 on-policy 蒸馏循环:从 D 采 prompt,学生(实为教师-学生混合分布 `p̃=α·p+(1-α)·q`)采样响应 → ③ 教师给每个 token 算 `r_t=log p/q` 作为质量信号,累积成回报 `R_t` → ④ 用策略梯度更新学生,梯度拆成 (∇L)_Single(单步,可对词表直接求和、无需 MC)+ (∇L)_Long(长程,带重要性权重 w_t 与裁剪)+ 语言建模损失 ∇L_PT(防遗忘通用能力) → 收敛输出学生。【原文 §2.2, §2.3, Alg.1】
- 逐组件必要性(均有消融,Table 4):
  - **reverse-KL(核心)**:决定 mode-seeking;无它退化为标准 KD。toy 实验(Fig.2)直观展示 reverse 挑主模式、forward 摊平到空白区。
  - **单步分解(Single-Step Decomposition)**:把单步质量 r_t 从累积 R_t 中拆出直接求梯度,降方差/加速收敛。去掉:验证 R-L 27.4→27.0(轻微),但训练方差变大(Fig.8)。
  - **教师混采(Teacher-Mixed Sampling, α=0.2)**:防 reward hacking(小学生易生成重复/短句骗高教师分)。去掉:R-L 27.4→22.3,明显垮(Table 4),且训练中很快学会刷重复串。
  - **长度归一(Length Normalization)**:R_t 偏好短句→空响应;归一后消除。去掉:R-L 27.4→17.4,**最关键的稳定项**(Table 4)。
  - **L_PT(预训练语言建模损失)**:仿 InstructGPT,保通用 benchmark 能力(防遗忘)。
- 关键机制/公式(直觉):reverse-KL 的策略梯度 `∇L = -E[Σ(R_t-1)∇log q_θ]`(式2),直觉是"让生成在教师下概率高(↑p)、同时学生自己保持多样(↓q)"。式9 给出最终带重要性权重+裁剪+长度归一的可实现梯度。附录 A.1 还给了一个 Inverse-RL 视角:KD≈behavior cloning,而 reverse-KL on-policy≈用教师 logits 当奖励的 max-entropy IRL,所以能比纯模仿强。
- 实验与证据:teacher=GPT-2-1.5B/OPT-13B/LLaMA-13B,student 跨 GPT-2(120M-760M)/OPT(1.3B-6.7B)/LLaMA-7B;指令遵循 5 数据集(DollyEval/SelfInst/VicunaEval/S-NI/UnNI),指标 Rouge-L + GPT-4 打分 + 人评。结果:几乎所有格子超 SFT/KD/SeqKD;部分(Vicuna/S-NI/UnNI)学生 R-L 甚至超教师(归因暴露偏差被缓解);ExAccErr 暴露偏差显著低且长文本停止累积(Fig.6);校准 ECE 更接近教师(Table 2);Dist-4 多样性基本不掉(Table 3)。教师规模律(Fig.5):教师越大学生越好(不出现"教师太强反而蒸不好"的退化)。baseline 公平(同初始化、同数据、同评测)。
- 假设与失效边界:【原文】reverse-KL 易丢模式影响多样性,作者主张"很多 NLP 应用只需一个正确答案"故可接受(§3.3)——这是**显式假设**,对需要高多样性/创意生成的场景失效。【推断】需要**white-box**(教师输出分布可得),对只有 API 的闭源教师不适用;教师与学生需同词表(否则 token 级 KL 无法对齐);任务以指令遵循/相对短-中长度生成为主,超长 agent 轨迹未验证。
- 祛魅总结:【推断】真贡献=**最早把"reverse-KL + on-policy 策略梯度"系统化用于生成式 LLM 蒸馏并跑稳+跨规模验证**,是后续一切 OPD(GKD、DistiLLM、本课题 mtp_opd)的奠基与命名来源("On-Policy Distillation"概念)。被低估的是三项稳定化的工程价值(长度归一/教师混采几乎是 on-policy 蒸馏能否跑通的开关)。被高估处:它是 2023 的工作(虽更新到 2026),骨干是 GPT-2/OPT 级小模型 + Dolly 指令数据,**未触及推理(reasoning)/RLVR/长 CoT 场景**;"超过教师"主要因暴露偏差,不代表知识增量。

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号 = 教师在**学生自采样轨迹**上的 token 级对数概率比 `log p/q`(reverse-KL 视角的密集奖励);非外部 verifier。
  - 改什么 = 学生模型参数(全参微调)。
  - 何时改 = 训练期 on-policy 循环,每步用最新学生(混教师)采样后更新。
  - 免梯度? = 否,核心是策略梯度(PG)。
  - 记忆-技能生命周期 = 无外部记忆/技能库;"知识"固化进学生参数;靠 L_PT 防通用能力遗忘。
  - 防遗忘机制 = 加 L_PT(预训练语料语言建模损失),类 InstructGPT。
- ⑦ 开源代码+框架/harness:https://github.com/microsoft/LMOps/tree/main/minillm(本地已 clone 整仓 ~348MB,Tier 保留)。框架=**自研**(改版 HuggingFace Transformers + DeepSpeed + Accelerate,搭出类 RLHF 的 on-policy pipeline;非 veRL/TRL/OpenRLHF 等现成 RL 框架)。【原文 §1 脚注链接 + 元信息】
- 💰 资源/成本与可扩展性:原文未给 GPU 时/成本数字;仅说方法"可扩展至 120M–13B 跨族"。【原文未说明具体算力】
- 🎯 对"探索-巩固"对标:**强支撑(谱系源头)**。MiniLLM = "teacher 当脚手架、student 在自己分布上挑能走通的主模式"的最直接实现:on-policy 自采样=探索;reverse-KL 挑教师认可的主模式=朝"自己能走通的开头"收敛;L_PT=巩固时防遗忘。**可借组件**:① 教师混采 α 作"稀疏脚手架强度"旋钮(本课题"teacher 稀疏接管"的现成实现);② 单步分解可对接 MTP 的逐步前瞻信号;③ 长度归一是任何"自采样轨迹+token 奖励"训练的必备项。**缺口**:无"走偏后自选恢复分支"的显式 path-recovery 机制(它是均匀地在整条自采样轨迹上软对齐,不挑关键步/高熵分叉点),也无 MTP 式前瞻。一句判定:本课题的"探索-巩固"可视为 MiniLLM 的"加 path-recovery 单点接管 + MTP 前瞻探针"的升级版。
- 🔭 开放问题/未来方向:【原文】把更多分布差异度量(TVD/最优传输等)用于 LLM-KD;改进 reverse-KL 的多样性损失。【推断】(1) 把 token 级软对齐升级为**关键步/分叉点**的稀疏对齐(只在高熵/低置信处接管),贴合本课题;(2) 迁移到长 CoT/推理与 agent 多轮轨迹;(3) 教师混采 α 的自适应调度(随训练把脚手架逐步撤掉);(4) 与 RLVR 信号(可验证奖励)联合,而非纯教师 KL。

— 残留待核:0

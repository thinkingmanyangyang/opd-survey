dasd | Distribution-Aligned Sequence Distillation for Superior Long-CoT Reasoning (DASD-4B-Thinking) | 阿里云(Shaotian Yan*, Kaiyuan Liu*, Chen Shen*†, Bing Wang*, Sinan Fan* 等) | arXiv 2601.09088v1(2026-01-14)·技术报告 | 主题线 L1(序列级蒸馏/含轻量 on-policy 阶段)·相关性 高

**原始论文**:https://arxiv.org/abs/2601.09088

## 一眼看懂
- 🟦 TL;DR:主流"在 teacher 生成的响应上做 SFT"(序列级蒸馏)只把它当数据过滤问题,忽视了蒸馏本质——让 student 学到 teacher 完整序列分布。DASD 从"分布对齐"视角补回师生交互三件套:温度调度学习(先低温学一致模式再高温扩覆盖)、散度感知采样(优先喂"teacher 高置信+student 低概率"的句子)、混合策略蒸馏(student 自生成前缀→teacher 续写,缓解 exposure bias),只用 448K 样本就让 Qwen3-4B 在 AIME24/25、LCB、GPQA 上达同量级 SOTA,部分基准超 32B 模型【原文 Abstract, §1, Table 6】。
- 最巧的一步:**散度感知采样(Divergence-aware Sampling, DAS)**。抽掉它,整套就退回"随机采样+质量过滤"的旧范式,核心论点(分布对齐比堆数据重要)就垮——Table 3 显示 DAS(50K)在 AIME25 上(79.2)甚至超过随机采样翻倍数据(100K RS,78.9)【原文 §4 Table 3】。为什么:它把"哪些 teacher 句子最该学"从启发式规则升级为"按师生概率差(teacher 高/student 低=高散度)筛选",直接对齐 student 学习能力、规避 SFT 的误导梯度(对 teacher 低概率但 student 已高概率的 token 继续抬高的反向梯度)【原文 §4】。

## 为什么做
- 研究背景:DeepSeek-R1 首次证明从强 teacher 蒸馏可大幅赋能小模型推理,引发社区大量复刻(OpenR1/OpenThoughts/a-m-team/AceReason/LIMO/s1/Light-R1 等);主流范式即"SFT on teacher responses = 序列级蒸馏(Kim&Rush 2016)",简单高效、不限师生架构、不需 token 级 logits【原文 §1】。
- 解决的具体痛点:现有工作只停留在 SFT 视角、专注设计启发式数据过滤规则,**忽视蒸馏本质(继承 teacher 泛化能力)**,根因是全程缺乏显式师生交互,导致三大缺陷:(i)teacher 序列分布表征不足(随机采样覆盖窄、或过度表征低概率噪声序列);(ii)teacher 分布与 student 学习能力错配(SFT 只抬 ground-truth token 概率→误导梯度);(iii)exposure bias(teacher-forced 训练 vs 自回归推理不一致)【原文 §1, §2】。
- 相关工作 & 各自不足:序列级蒸馏(本文改进对象)简单但只做数据过滤;logit 蒸馏(Hinton;Qwen3/Gemma 的 on-policy 变体、Thinking Machines Lab 实现)需 token 级 logits 且跨 tokenizer 难对齐(输出空间错位)【原文 §1】;on-policy distillation 要求师生同 tokenizer/词表(token 级监督的约束),DASD 用句子级分析规避此约束【原文 §4 line 626】。
- 动机链:现状(序列级蒸馏火但只当 SFT 数据问题)→缺陷(缺师生交互→三大问题)→所以必须(不改用 logit 蒸馏、保留序列级简单性,但从分布对齐角度补回师生交互:更好覆盖+更优目标分布+缓解 exposure bias)【原文 §1, §3】。
- 与最近邻工作的Δ:相对纯 SFT 蒸馏(OpenR1/LIMO 等),差在"把采样策略和数据选择从启发式提升为分布对齐(温度调度+DAS)+ 加一个轻量 on-policy 混合阶段";相对 on-policy/logit 蒸馏,差在"只需师生对每个 teacher token 的概率(句子级 geo-mean),不需全词表 logits、不要求同 tokenizer"。有用点:用极少(448K)数据达 SOTA,且 DAS 数据可跨 student 复用(§4 line 830:为 4B 筛的数据可迁移到 30B-A3B)。

## 怎么做 + 靠不靠谱
- 方法流水线:输入跨域难题(数学 105K + 代码 + 科学 + 指令)→ ①teacher(gpt-oss-120b)在低温(T=0.6)、高温(T=1.0)各采多响应 → ②DAS 筛"高散度"(Teacher Sentence 丰富)的实例 → ③过滤(长度/结构/重复)得 105K 低温 + 330K 高温 → ④温度调度两阶段 SFT(先低温 cold-start 再高温续训)→ ⑤混合策略蒸馏:从 DAS 集采 50K 题让 student 生成、截断后 teacher 续写、过滤得 12.7K → ⑥输出 DASD-4B-Thinking(总数据 ≈ 448K)【原文 §6, Fig.2】。
- 逐组件必要性(均有独立消融):
  - **温度调度学习**(Table 1):有消融。低温样本一致易学但覆盖窄,高温覆盖广但难学(loss 居高);两阶段(50K T=0.6 cold-start → 50K T=1.0)在 AIME24/25 全面超单温度静态基线(如 81.7/71.9 → 85.2/81.3)。没它→要么覆盖窄要么早期学习不稳【原文 §3 Table 1】。
  - **散度感知采样 DAS**(Table 3/4):有消融。同采样预算下 DAS 一致优于随机采样(50K DAS AIME25=79.2 vs 50K RS=76.1,甚至超 100K RS=78.9);跨 teacher、跨域均成立。没它→退回随机采样、受误导梯度拖累【原文 §4 Table 3/4】。
  - **混合策略蒸馏**(Table 5):有消融,且含关键对照。仅 7.7K 数据即提升(baseline 83.3/74.2 → 7.7K no-mask 83.3/74.8)。**关键发现:mask 掉 student 生成段(只留 teacher 续写)反而更差(80.8/72.3)**——证明训练时保留 on-policy student 段很重要。没它→exposure bias 未缓解(student 在长响应上偏离 teacher,Fig.7 cut-off 率随长度上升)【原文 §5 Table 5, Fig.7】。
- 关键机制/公式(直觉):①序列级蒸馏目标 = min KL(p_T(y|x)‖p_S(y|x)),用采样响应 ŷ 的点质量近似 p_T 后即退化为标准 SFT loss(Eq.1-5)——这解释了"为何 SFT on teacher data 是有效蒸馏";②**四类句子分解**(Fig.5):按 teacher/student/distilled 三模型对每句的概率(token 概率几何均值)差异分为 Teacher Sentence(p_T≫p_S,student 可放心抬高、无误导梯度)/Student Sentence/Shared Sentence/Boosted Sentence;实证 Teacher Sentence 与答案正确性正相关(Fig.6 实线持续高于虚线),Boosted Sentence 反而负相关;③DAS 在训练前即可识别 Teacher/Student Sentence(只看师生概率差),优先喂 Teacher Sentence 丰富的样本【原文 §2, §4 Fig.5/6】。
- 实验与证据:teacher gpt-oss-120b(另验 Qwen3-Next-80B-A3B-Thinking),student Qwen3-4B-Instruct-2507(MoE 版 30B-A3B);训练 full SFT、cutoff 64K、greedy packing、ZeRO-3 + Liger kernels、global batch 64、6 epochs、lr 5e-5→1e-5(cosine);评测统一 temperature 1.0/top-p 1.0、每题采 64 报均值、AIME 最大生成 102400 token、LCB/GPQA 81920。主结果 Table 6:DASD-4B = AIME24 88.5 / AIME25 83.3 / LCB v5 69.3 / LCB v6 67.5 / GPQA-D 68.4,超 Qwen3-4B-Thinking-2507(AIME25 81.3)、Qwen3-32B(72.9)、AM-thinking-v1-32B(74.4,且用 2.9M 数据 vs 本文 448K)【原文 §6.4, §7】。baseline 公平性:对照分"开权重"与"开权重+开数据"两族,数据规模对比突出(448K vs 2.9M/30M),较公平;但"看着强但没回答"的点:三件套多为**逐项独立消融(各在不同小规模子集),缺少在最终 448K full pipeline 上的逐项可加性消融**,无法判定三者叠加时各自净贡献【推断,据 §3-5 消融均为独立子实验】。
- 假设与失效边界:【原文】DAS 需拿到师生对每个 teacher token 的概率(发布了 -Logprob 数据集);结构过滤显式剔除含 function call 的 teacher 响应(line 1056,工具调用留给未来工作)——故当前不适用于工具/agent 蒸馏。【推断】"Teacher Sentence 与正确性正相关"为观测性结论,因果未隔离;四类句子分解依赖句子切分,跨语言/无明确句界的输出可能失效;teacher 概率对闭源 API 不总可得(虽作者称"many closed-source APIs 也暴露 teacher 概率")。
- 祛魅总结:真贡献=把序列级蒸馏从"数据过滤"重新框定为"分布对齐",并给出可操作的 DAS(只需师生 token 概率,跨 tokenizer 可用)+ 数据效率惊人(448K 达 SOTA)+ mixed-policy 的 mask 消融是有价值的负结果(证明 on-policy 段不可去)。包装/高估处:title/abstract 的"distribution-aligned"听起来像理论对齐,实际 DAS 是基于经验观测(Teacher Sentence 正相关)的启发式采样,无理论保证;"SOTA 超 32B"成立但基于 4B-Instruct 这一强 base + 强 teacher gpt-oss-120b,蒸馏增益与 base/teacher 质量耦合,论文未拆分【推断】。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=teacher 序列级输出分布(经 DAS 筛的 Teacher Sentence 丰富响应)+ mixed-policy 阶段的 teacher 对 student 错误前缀的续写修正 | **改什么**=student 全参数(full SFT)| **何时改**=离线两阶段 SFT + 一个轻量 mixed-policy 阶段(均为训练期)| **免梯度?**=否(SFT 梯度训练)| **记忆-技能生命周期**=不涉及显式记忆/技能库,推理能力固化进权重 | **防遗忘机制**=无专门防遗忘;温度调度的低温 cold-start 可视为稳定早期学习(弱相关)【原文 §6.4】。
- ⑦ 开源代码+框架/harness:https://github.com/D2I-ai/dasd-thinking(含 train/stage1.yaml、stage2.yaml、deepspeed/ds_z3 配置、技术报告 PDF;**含训练配置但不含数据生成/DAS 采样代码**);框架 **LLaMA-Factory**(README 明确"utilize LLaMA-Factory")+ DeepSpeed ZeRO-3 + Liger-Kernel。本地已 clone(~5.2MB)。模型/数据在 HF & ModelScope(DASD-4B-Thinking、DASD-30B-A3B-Thinking-Preview、Superior-Reasoning-SFT-gpt-oss-120b 及 -Logprob 版)。【纯 SFT 栈,非 RL 框架】
- 💰 资源/成本与可扩展性:论文未给 GPU 卡时/总训练成本(原文未说明);仅知 64K context 训练靠 ZeRO-3+Liger 降显存、global batch 64、6 epochs;数据效率是其卖点(448K vs 同类 2.9M/30M)【原文 §6.4.1】。
- 🎯 对"探索-巩固"对标:**部分支撑(巩固侧 + 一个迷你回轨机制)** —— 一句判定:DASD 的 mixed-policy 阶段(student 自生成前缀 → 随机截断 → teacher 从截断点续写修正)正是"走偏后由 teacher 提供 path-recovery 监督"的离线 SFT 版,与"巩固/回轨"诉求直接同构;依据:§5"randomly cut off student solutions and prompt teacher to continue...targeted guidance on student's errors",且 Table 5 证明保留 on-policy student 段(不 mask)才有效。可借组件:① mixed-policy 的"student 前缀+teacher 续写"构造可直接搬到 OPD 的路径恢复数据生成;② DAS 的"按师生概率差选高散度 token"思想可迁移到 OPD 中"选哪些 token/步该由 teacher 接管"。缺口/竞品差异:DASD 是**离线 SFT(teacher-forced 续写),不是 on-policy 自选恢复分支**——teacher 决定从哪截断、写什么,student 不自选;无 MTP 前瞻;无 RL/免梯度信号。与本项目"on-policy 自选恢复 + 稀疏脚手架"相比,DASD 的 teacher 介入更密、更 off-policy。
- 🔭 开放问题/未来方向:【原文】§5 称 mixed-policy 是"promising direction"值得继续探索;§6.3 把 function calling/工具调用蒸馏明确列为 future work(当前过滤掉)。【推断】三件套在 full pipeline 的可加性消融、DAS 的理论化(目前纯经验)、跨语言句子分解鲁棒性、与 RL/on-policy KD 的结合、teacher 概率不可得时的退化方案均未解。

---
RETURN: dasd | 读到PDF? 是(20页全文) | L1(序列级蒸馏+轻量on-policy) | 巩固/回轨侧部分支撑(mixed-policy=teacher 续写的离线 path-recovery,但 teacher-forced 非 on-policy 自选);DAS 高散度选样可借鉴 | 残留待核 1(正文/Table 均写 student=Qwen3-4B-Instruct-2507,而 v1 记发布 YAML 为 Qwen3-4B-Thinking-2507——文本与代码不一致,以代码为准存疑)

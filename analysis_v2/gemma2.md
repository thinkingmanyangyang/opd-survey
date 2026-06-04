gemma2 | Gemma 2: Improving Open Language Models at a Practical Size | Gemma Team, Google DeepMind | 2024-08（arXiv:2408.00118 技术报告，原始发布 2024-06-27） | L1 OPD/自蒸馏 · 相关性中（蒸馏训练小模型 + 后训练点名 on-policy KD）

**原始论文**：https://arxiv.org/abs/2408.00118

## 一眼看懂
- 🟦 TL;DR：Gemma 2 的 2B/9B 在**预训练**阶段用知识蒸馏（逐 token 软标签交叉熵 \(\min_{P_S}\sum_x -P_T(x\mid x_c)\log P_S(x\mid x_c)\)）替代标准 next-token 预测来训小模型；27B 因没有更大的同族教师，仍用 next-token。**后训练 SFT** 阶段明确"在学生自身分布上从教师蒸馏（on-policy distillation，引 Agarwal 2024=GKD / Gu 2024=MiniLLM）"，再叠 RLHF，最后对各阶段模型做模型平均（model merging）。"后训练 on-policy KD"是它与本综述（on-policy 蒸馏）最直接的连接点【原文 §3.2, §4】。
- 最巧的一步：**用大教师的逐 token 软标签替代 one-hot 硬标签**，并**故意训远超 token 数的量**（2B/9B 训 50× compute-optimal token）来"模拟训练超出可用 token 数"。抽掉这一步、退回 from-scratch next-token——Table 6 显示 2B 在 500B token 上从 distilled **67.7** 掉到 **60.3**（3 benchmark 平均，抽掉蒸馏 −7.4 分）。直觉：小模型瓶颈是"每 token 学到多少信息"，教师完整分布比 one-hot 信息密度大得多。但作为技术报告，蒸馏温度/教师规模/on-policy KD 配方均未披露，命门虽明、细节不可复现【原文 §1, §3.2, Table 6；不可复现为推断】。

## 为什么做
- 研究背景：小模型性能提升此前主要靠"延长训练"（最新小模型用到 15T token 才提 1–2% SOTA），而这随数据量只 log 增长（Hoffmann22 Chinchilla）。知识蒸馏（Hinton 2015）用大模型软标签指导小模型，比硬标签含更多信息、给更丰富梯度；Gemini 1.5 已用蒸馏。on-policy 蒸馏（Agarwal 2024 GKD、Gu 2024 MiniLLM）让学生在自己生成分布上接受教师监督，缓解 train/inference 分布不一致【原文 §1, §3.2, §4】。
- 解决的具体痛点：小模型若用常规 next-token 预训练，在固定参数/算力预算下"仍 under-trained"，难逼近 2–3× 更大的模型；单纯扩数据对小模型边际收益递减（数据效率瓶颈）【原文 §1】。
- 相关工作 & 各自不足（来龙去脉）：
  - ① **标准 next-token 预训练 + 延长训练**（Gemma 1、LLaMA、Mistral）——只随数据量 log 增长，小模型数据效率瓶颈，是本文要替换的主路线。
  - ② **监督 KD**（Hinton15 软标签、Sanh19 DistilBERT）——通常用于"缩短小模型训练时间"，Gemini 1.5 用过；Gemma 2 反其道用它"训超量 token 以模拟更多数据"。
  - ③ **on-policy KD**（Agarwal24 GKD、Gu24 MiniLLM）——缓解 train/inference 分布失配，Gemma 2 后训练 SFT 采用，但**不复现其对照**（只引用并采用）。
  - ④ **架构侧**：interleaving local-global attention（Beltagy20 Longformer）、GQA（Ainslie23）、logit soft-capping（Bello16）、model merging（Ramé24 WARP/WARM）——均为"已知技术模块"的组合，非本文新提。
  - 报告未对上述做横向消融，只引用并采用【原文 §1, §4】。
- 动机链：要在"实用规模"（2B/9B/27B，可消费级/单机部署）榨出同尺寸最佳性能 → 小模型瓶颈在 token 信息密度 → 用大教师软标签蒸馏（虚拟扩大信息量）训预训练小模型 → 后训练再让蒸馏发生在学生自身输出分布上（on-policy），对齐推理真实分布 → 各阶段模型平均稳性能。
- 与最近邻工作的Δ（精确差异）：最近邻是 Gemma 1（next-token 预训练）与 Gemini 1.5（已用蒸馏）。Δ：Gemma 2 把"预训练蒸馏"系统用于开源实用规模（2B/9B）、且**刻意训 50× compute-optimal token**（用蒸馏模拟超量数据），并在后训练显式叠加 on-policy KD + 模型平均。为什么有用：Table 6/7 在控制 token 预算下证明蒸馏增益，且 9B/27B 达当时同尺寸开源 SOTA【原文 §1, §6】。

## 怎么做（细到可复现的程度——技术报告，部分配方闭源）
### 1. 预训练蒸馏（§3.2）
- 给定大教师，逐 token 用教师对下一 token 的分布 \(P_T(x\mid x_c)\) 当软目标，最小化教师与学生分布的负对数似然（=逐 token 软标签交叉熵）：
\[
\min_{P_S}\ \sum_{x}\ -\,P_T(x\mid x_c)\,\log P_S(x\mid x_c).
\]
- 2B/9B 用此蒸馏；27B 用标准 next-token（无更大同族教师）。预训练 token：27B 用 13T、9B 用 8T、2B 用 2T（英文为主 web/code/science）；SentencePiece 256k 词表。

### 2. 架构要点（§2，非蒸馏核心但影响复现）
- **local/global 注意力逐层交替**：局部滑窗(4096) 与全局(8192) 每隔一层交替。
- **GQA**：num_groups=2（vs MHA 几乎无损但省参/快，Table 8: 50.3 vs 50.8）。
- **Logit soft-capping**：注意力层与最终层把 logits 压到 \([-\text{soft\_cap},+\text{soft\_cap}]\)：
\[
\text{logits}\leftarrow \text{soft\_cap}\cdot\tanh\!\big(\text{logits}/\text{soft\_cap}\big),
\]
注意力层 soft_cap=50.0、最终层=30.0。
- **pre-norm + post-norm（均 RMSNorm）**：对每个子层输入输出都归一。
- **深 > 宽**：9B 深网络略优（Table 9: 52.0 vs 50.8）。具体规模：2B(d=2304,26 层)/9B(d=3584,42 层)/27B(d=4608,46 层)；GeGLU、RoPE、tied embedding、context 8192。

### 3. 后训练（§4）
- **SFT（behavioral cloning）**：在合成+真实 prompt 上，响应**主要由更大教师合成**；同时**在学生自身分布上从教师蒸馏（on-policy KD，引 Agarwal 2024 / Gu 2024）**——原文："We also run distillation from the teacher on the student's distribution"。
- **RLHF**：算法沿 Gemma 1.1，但 reward model 大一个数量级、更偏多轮对话；policy 基于与 SFT 相同 prompt。
- **模型平均（model merging）**：对用不同超参跑出的多模型做平均（Ramé24）。
- **数据**：后训练用 LMSYS-chat-1M 的 **prompt（不用其答案）**；过滤聚焦 helpfulness、降 safety/hallucination 危害、去评测集污染、降 recitation 风险；发现加入"鼓励 in-context attribution/hedging/refusal"的子集能提 factuality 而不损其他指标。

### 数据流动 / 逐组件必要性
- 预训练：教师对每位置输出完整分布 → 学生用软标签 CE 拟合（替代 one-hot）→ 训 50× compute-optimal token。后训练：教师合成响应 SFT + 学生自生成分布上的 on-policy KD → RLHF → 多模型平均。
- **预训练蒸馏**：核心，有对照消融——Table 6（2B distilled 67.7 vs from-scratch 60.3，500B token，7B 教师）、Table 7（不同尺寸 perplexity）。
- **后训练 on-policy KD**：被点名采用但**无独立消融**——报告未给它相对 off-policy KD 的对照，其独立贡献无法从本文判断【§4，推断】。
- **RLHF / 模型平均**：标准后训练组件，无单独消融数字。

## 靠不靠谱
- 实验与证据：
  - **蒸馏 vs from-scratch（Table 6）**：2B 在 500B token（10× compute-optimal）上，distilled（7B 教师）**67.7** vs from-scratch **60.3**（3 benchmark 平均）。
  - **不同尺寸的蒸馏增益（Table 7）**：验证集 perplexity——200M/400M/1B：distilled(7B 教师) **21/17/15** vs from-scratch **23/19/17**。论文结论是"**the gain remains as the model size is scaled**"（增益随尺寸放大而保持，约恒定 ~2 ppl），**非**"模型越小蒸馏越有用"〔订正：旧 analysis 称"蒸馏收益随模型变小更明显"与 Table 7 原文表述不符——各尺寸 ppl 差近似恒定，原文措辞为 gain remains〕。
  - **架构消融**：MHA vs GQA（Table 8 50.3/50.8）、wide vs deep（Table 9 50.8/52.0）、滑窗 4096/2048/1024 推理 ppl 1.63/1.63/1.64（Table 10，可调省推理）、MMLU 格式鲁棒性（Table 11，Gemma 2 2B std 2.1 vs Mistral 7B 6.9）。
  - **指令模型**：Chatbot Arena/人类评测；标准基准 MMLU/GSM8K/ARC/HellaSwag/HumanEval/MBPP/AGIEval/BBH/WinoGrande【§6】。
  - **baseline 公平吗**：Table 6/7 的"蒸馏 vs from-scratch"在控制 token 预算/教师规模下是干净对照（教师固定 7B、模拟 27B→9B 的师生 gap）；但全报告无 on-policy KD 单独消融。
  - **"看着强但没回答核心问题"**：作为技术报告，蒸馏部分消融粗略，缺乏对 on-policy KD 单独贡献的隔离实验；"蒸馏=以虚拟信息量训练"是直觉说法而非可量化论证。
- 假设与失效边界：
  - 【原文】27B 未蒸馏（同族无更大教师），故"蒸馏更优"严格说只在 2B/9B 上成立（§3.2, §6.1）。
  - 【原文】后训练数据用 LMSYS-chat-1M 的 **prompt（不用其答案）**；多级数据过滤（§4）。
  - 【推断】完全闭源训练——教师模型、数据混合、蒸馏温度、on-policy KD 配方、模型平均权重均未公开，**不可复现**（依据：§3.2/§4 均无超参/配方细节）。
  - 【推断】on-policy KD 的独立贡献无法从本文判断（无对照），其"on-policy"成分到底带来多少增益未知（依据：§4 仅引用 GKD/MiniLLM，无消融）。
  - 【推断】标准基准分数仍受数据混合/污染过滤质量影响，跨模型对比需谨慎（依据：§4 提 recitation/污染过滤但效果未量化）。
- 祛魅总结：
  - 真贡献【推断】：在开源实用规模（2B/9B）上把"预训练蒸馏"做成主力训练范式（并刻意训超量 token）并公开权重，且明确把"后训练 on-policy KD + 模型平均"纳入配方——为社区提供了"用大模型造强小模型"的工程范本与强基线权重。
  - 包装/高估【推断】：作为技术报告，"蒸馏有效"的结论可信但消融粗（尤其 on-policy KD 无隔离实验）；"以虚拟信息量训练"是叙事性直觉，非严格论证。本文对本综述的价值主要是"立场+权重"，而非可复现的 OPD 方法。
  - 低估/订正【推断】：旧 analysis 称"蒸馏收益随模型变小更明显"应订正为"增益随尺寸放大而保持"（Table 7 原文 gain remains，各尺寸 ppl 差近似恒定）。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=预训练用教师逐 token 完整概率分布（软标签 CE）；后训练 SFT 用教师在学生自生成分布上的 token 监督（on-policy KD）+ RLHF 奖励｜**改什么**=学生全参（预训练+后训练全程）｜**何时改**=预训练期（软标签替代 next-token）+ 后训练 SFT 期（on-policy KD）+ RLHF 期｜**免梯度?**=否（全监督/RL 梯度）｜**记忆-技能生命周期**=无显式记忆/技能库，知识固化进参数｜**防遗忘机制**=**模型平均（model merging/averaging）** 对各阶段/各超参模型做平均提升整体性能（WARP/WARM 思路），间接缓解阶段间能力遗忘。
- ⑦ 开源代码+框架/harness：**仅发布权重**（huggingface.co/google/gemma-2-2b、-9b、-27b 及对应 -it 指令版），**无训练/蒸馏代码**（n/a）。CloneTier=B，仅记录不 clone。框架=内部 Google **JAX + TPU** 栈（GSPMD 分片 + Pathways 跨 pod + MegaScale XLA + ZeRO-3 式 optimizer 分片），属"大厂技术报告/权重发布仓，无训练代码"。
- 💰 资源/成本与可扩展性：预训练 TPU——2B 用 512 TPUv5e（512-way 数据×1-way 模型分片）、9B 用 4096 TPUv4（1024×4）、27B 用 6144 TPUv5p（768×8）。预训练碳足迹估 1247.61 tCO2eq（碳中和）。需先有更大同族教师才能蒸馏（27B 因无教师未蒸馏）【原文 §3.3, §3.4, Table 3】。
- 🎯 对"探索-巩固"对标：**弱支撑（立场层）/ 背景**。一句判定：Gemma 2 主要价值是"工业界确认 on-policy KD 进入主流后训练配方"这一立场背书 + 一个可用的强基线权重，但它**不提供可借的 OPD 机制细节**（配方全闭源）。与"探索-巩固"对标点稀薄：① 预训练蒸馏是稠密软标签 CE，无"探索/选路"也无"path-recovery"；② 后训练点名 on-policy KD 但无细节、无单点接管、无 MTP；③ **模型平均**可作"巩固/防遗忘"的一个粗粒度参考（合并多阶段/多超参能力），但与"固化进记忆/技能库"相去甚远。缺口：无可复现方法、无路径级监督、无 MTP、无记忆/技能生命周期。依据：§3.2/§4 仅给立场与软标签公式、无配方。
- 🔭 开放问题/未来方向：【原文】27B 无同族教师故未蒸馏——更大规模如何蒸馏（自蒸馏/弱到强？）开放（§3.2, §6.1）；on-policy KD 细节缺失（§4）。【推断】把"预训练稠密软标签蒸馏"与"后训练稀疏 on-policy 脚手架（path-recovery 单点接管）"分工，可在小模型训练里区分"打底（稠密 KD）"与"纠偏（稀疏 OPD）"；用 MTP 前瞻替代/补充教师软标签，让学生在预训练期就学"前瞻一致性"；模型平均可升级为"按技能选择性合并"以更精细防遗忘（依据：本文模型平均 + 本课题记忆/技能固化）。

读到PDF? 是（PyMuPDF 全文 21 页；§3.2 蒸馏公式 + §4 后训练 + Table 6–11 + 架构表/soft-cap 公式全核）｜L线 L1（蒸馏训练小模型 + 后训练 on-policy KD 立场）｜对标结论 弱支撑/背景（工业界 on-policy KD 立场背书 + 强基线权重；配方全闭源、无可借机制、无路径级/MTP/记忆）；含一处对旧 analysis 的订正（Table 7 增益随尺寸"保持"而非"越小越明显"）｜残留待核 0

# gemma2 — Gemma 2: Improving Open Language Models at a Practical Size

> **一句话重点 (TL;DR)**：Gemma 2 的 2B/9B 用**知识蒸馏(逐 token 软标签的交叉熵)替代 next-token 预测**来预训练小模型；后训练 SFT 阶段明确"在学生自身分布上从教师蒸馏(on-policy KD，引 GKD/MiniLLM)"——这是它与本综述(on-policy 蒸馏)最直接的连接点。

**元信息**：arXiv:2408.00118（技术报告，原始发布 2024-06-27）｜ Gemma Team, Google DeepMind ｜ 主题 T1（蒸馏训练小模型）/ 相关性中 ｜ 代码 仅发布权重(huggingface.co/google/gemma-2-9b 等)，**无训练/蒸馏代码**（CloneTier=B，不 clone）｜ 框架 内部 Google JAX + TPU 栈。

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/gemma2/fig_01.png)

*Figure 1 | Comparing memorization rates. We find significantly lower memorization rates across-the-board. (Left) Overall memorization across model families. (Right) Exact and approximate memorization per data source.*

## 1. 相关工作与进展

- **知识蒸馏**(Hinton et al. 2015)：用大模型(教师)的软标签指导小模型,比硬标签含更多信息。
- **on-policy 蒸馏**(Agarwal et al. 2024 GKD、Gu et al. 2024 MiniLLM)：让学生在**自己生成的分布**上接受教师监督,缓解 train/inference 分布不一致。
- **架构改进**:局部滑窗 + 全局注意力交替、GQA、logit soft-capping、pre+post RMSNorm 等(沿 Gemini 1.5 的蒸馏经验)。

## 2. 现有工作存在的问题

- 小模型若用常规 next-token 预训练,在固定参数/算力预算下性能受限,难以逼近 2–3× 更大的模型。
- 单纯靠扩数据对小模型边际收益递减(数据效率瓶颈)。

## 3. Motivation
在"实用规模"(2B/9B/27B,可在消费级/单机部署)下榨出同尺寸最佳性能。思路:用大教师的知识蒸馏训练小模型,相当于以远超 token 数的"虚拟"信息量训练,缓解小模型的数据效率问题。

## 4. 主要灵感 / 核心直觉
小模型的瓶颈不在算法而在"每个 token 能学到多少信息"。教师对每个位置给出的**完整概率分布**比一个 one-hot 硬标签信息量大得多;用它当软目标,等于把训练信号的密度大幅提高。后训练再让蒸馏发生在学生自己的输出分布上(on-policy),进一步对齐推理时的真实分布。

## 5. 主要解决思路(一段话讲清核心)
预训练阶段(2B/9B):不再用 next-token 预测,而是逐 token 最小化"教师概率 P_T 与学生概率 P_S 的交叉熵"(软标签蒸馏);27B 因没有更大的同族教师,仍用标准 next-token。后训练阶段:SFT 做行为克隆(响应主要由更大教师合成)并**在学生分布上从教师蒸馏(on-policy KD)**,再叠加 RLHF,最后对各阶段模型做**模型平均**。

## 6. 方法详解(通俗、分步骤)

- **预训练知识蒸馏(§3.2)**:给定大教师,逐 token 用其概率分布 \(P_T(x \mid x_c)\) 作软目标,最小化 \(\min_{P_S} \sum_x -P_T(x \mid x_c) \log P_S(x \mid x_c)\)。2B、9B 用此蒸馏训练;27B 用标准 next-token(无更大同族教师)。
- **后训练(§4)**:
  - **SFT**:在合成+真实 prompt 上行为克隆,响应主要由更大教师合成;并**在学生自身分布上从教师蒸馏(on-policy distillation,引 GKD Agarwal 2024 / MiniLLM Gu 2024)**。
  - **RLHF**:沿用 Gemma 1.1 类似算法,但 reward model 大一个数量级,策略基于与 SFT 相同的 prompt,英文偏好数据训 reward model。
  - **模型平均(model merging/averaging)**:对各阶段后得到的模型做平均提升整体性能(WARP/WARM 思路)。
- **架构要点**:局部滑窗注意力(窗口 4096)与全局注意力(8192)逐层交替、GQA(num_groups=2)、logit soft-capping(注意力层 50.0、最终层 30.0,经 tanh 软截断:\(\mathrm{logits} \leftarrow \mathrm{soft\_cap} \cdot \tanh(\mathrm{logits}/\mathrm{soft\_cap})\))、pre+post RMSNorm。
- **数据**:后训练用 LMSYS-chat-1M 的 **prompt(不用其答案)**;过滤聚焦提升 helpfulness、降低 safety/hallucination 危害、去评测集污染、降 recitation 风险。

## 7. 实验数据集

- **预训练**:27B 用 13T tokens、9B 用 8T、2B 用 2T(以英文为主的 web/code/science),SentencePiece 256k 词表。
- **评测**:MMLU、GSM8K、ARC、HellaSwag、HumanEval、MBPP、AGIEval、BBH、WinoGrande 等标准基准;指令模型用 Chatbot Arena/人类评测。

## 8. 实验结果与主要发现

- **蒸馏 vs 从头训(Table 6/7)**:在固定 token 预算下,蒸馏训练的小模型显著优于从头 next-token 训练;且蒸馏的收益随模型变小而更明显——支撑"蒸馏对小模型尤其有效"。
- 2B/9B(蒸馏)相比同 token 数的 Gemma 1 大幅提质;9B/27B 在同尺寸开源模型中达到当时领先水平。
- **滑窗大小可调(Table 10)**:推理时调整滑窗大小对质量影响有限,提供部署灵活性。

## 9. 结果如何支撑其主张
主张是"蒸馏能在实用规模下让小模型更强"。Table 6/7 的"蒸馏 vs from-scratch"对照在控制 token 预算下直接验证蒸馏增益,且呈现"模型越小蒸馏越有用"的趋势,与 motivation(缓解小模型数据效率瓶颈)一致。但报告为工程技术报告,蒸馏部分的消融相对粗略、缺乏对 on-policy KD 单独贡献的隔离实验。

## 10. 逻辑自洽性(中性评估)
整体自洽,蒸馏(预训练)+on-policy KD(后训练)+模型平均构成一条连贯的"用大模型造好小模型"流水线。需注意:(1)作为技术报告,许多关键设置(教师规模、蒸馏温度、on-policy KD 的具体配方、模型平均权重)未充分披露,**不可复现**;(2)"蒸馏=以虚拟信息量训练"是直觉性说法而非可量化论证;(3)on-policy KD 虽被点名,但论文未给它相对 off-policy KD 的对照,故其独立贡献无法从本文判断。

## 11. 残留问题 / 局限

- **完全闭源训练**:无蒸馏/后训练代码,教师模型、数据混合、超参均未公开,本综述只能据论文记录其方法立场。
- **on-policy KD 细节缺失**:仅引用 GKD/MiniLLM 并称"在学生分布上蒸馏",未给损失形式、采样策略、占比等。
- **27B 未蒸馏**:同族无更大教师,故"蒸馏更优"的结论严格说只在 2B/9B 上成立。
- **评测污染/recitation** 虽做过滤,但标准基准上的分数仍受数据混合影响,跨模型对比需谨慎。

## 12. 开源代码与框架(链接+框架+代码可得性)

- **权重**:huggingface.co/google/gemma-2-9b(及 2b、27b 及对应 -it 指令版)。
- **代码可得性**:**仅发布权重,无训练/蒸馏代码**(n/a)。CloneTier=B,仅记录不 clone。
- **框架**:预训练用逐 token 软标签蒸馏(min CE between P_T 与 P_S);后训练 SFT(行为克隆 + on-policy KD,引 GKD/MiniLLM)+ RLHF + 模型平均;内部 Google JAX + TPU 栈(2B 用 512 TPUv5e、9B 用 4096 TPUv4、27B 用 6144 TPUv5p),GSPMD 分片 + Pathways。

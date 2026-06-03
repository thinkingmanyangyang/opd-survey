# eaft — Entropy-Adaptive Fine-Tuning: Resolving Confident Conflicts to Mitigate Forgetting (EAFT)

> **一句话重点 (TL;DR)**：SFT 之所以损害通用能力，主要源于一类"模型很自信、却被强迫去学相悖标签"的 token（Confident Conflicts）；EAFT 用逐 token 的归一化熵作为门控系数去缩放交叉熵——模型自信(低熵)就压低梯度、不确定(高熵)就正常学习，从而在几乎不损失目标任务的同时显著缓解遗忘。

**元信息**：arXiv:2601.02151（2026-01-07，预印本，曾居 HuggingFace Daily Paper #1）｜ 北京邮电大学 PRIS-CV / 中关村学院（Muxi Diao、Lele Yang、Wuxuan Gong 共同一作；通讯 Zhanyu Ma）｜ 主题 T3（SFT 改进 / 缓解遗忘）/ 相关性中-高 ｜ 代码 github.com/PRIS-CV/EAFT（主仓库仅 README+assets，实现已合入 LLaMA-Factory 与 ms-swift；本地两个 submodule 指针未拉取，**无法直接审计损失实现**）｜ 框架 LLaMA-Factory（主）+ ms-swift。

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/eaft/fig_01.png)

*Figure 1: (a) Conceptual illustration. When SFT forces the model to override its strong priors (e.g., labeling a 'ball' as a 'truncated icosahedron'), it creates a Confident Conflict . Fitting these conflicts distorts the model's existing representations, leading to catastrophic forgetting. (b) Toke*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/eaft/fig_03.png)

*Figure 3: Gradient Magnitude Landscape. Left: SFT exerts strong optimization pressure (dark purple) on Confident Conflicts in the bottom-left. Right: EAFT effectively suppresses these gradients (light yellow), protecting the model's existing representations.*

## 1. 相关工作与进展

- **领域适配的主流范式是 SFT**：行为克隆地拟合外部监督数据，简单高效，是把通用模型适配到数学、医学、Agent 等垂直域的标准做法。
- **缓解遗忘的已有思路**：加 KL 正则（SFT-KL，约束新模型不偏离原模型）、对 token 按预测概率重加权（DFT、FLOW、TALR 等，思路是"模型已会的少学、不会的多学"）。
- **RL 与 SFT 的对照观察**：近期工作（Chu et al. 2025 等）发现 on-policy RL 在适配目标任务时对通用能力的损害远小于 SFT，因为 RL 对齐的是模型"自己的信念"，而 SFT 强迫拟合"外部标签"。本文正是从这一对照出发。

## 2. 现有工作存在的问题

- **KL 正则**：粗粒度约束整个分布，既会抑制有害更新也会抑制有益学习，往往得不偿失。
- **基于概率的重加权（DFT 等）**：只看预测概率不足以判断该不该学。低概率 token 有两种截然不同的来源——(a) **epistemic uncertainty**：模型确实不会、应该学的有效知识；(b) **Confident Conflict**：模型其实很有把握(自己想输出别的)，标签却要它输出一个相悖 token。概率重加权在后者上反而**放大**破坏性梯度，加速遗忘。

## 3. Motivation
作者系统统计训练数据的逐 token 概率与熵，刻画出 SFT 与 on-policy rollout 之间的"**分布差异 (distributional gap)**"，并定位出最有害的一类样本：**低概率 + 低熵**的 token——模型对自己的预测高度确信(低熵)，却被 ground truth 强行拉向另一个 token(低概率)。论文称之为 **Confident Conflicts**，并通过 pilot study 验证：单纯**屏蔽**这些 token 就能明显减小通用能力下降，说明遗忘主要来自"在冲突样本上的破坏性梯度"，而非 SFT 流程本身。

## 4. 主要灵感 / 核心直觉
区分"该学"与"该抑制"的信号不是概率、而是**熵**：

- 熵高 = 模型在该位置不确定/在探索 → 这是真该学的新知识，保持高学习权重；
- 熵低 = 模型很笃定 → 若此时标签相悖即为 Confident Conflict，应压低梯度，避免破坏已有的通用表征。
用一个连续的熵门控替代 pilot study 里的硬屏蔽(后者会丢数据、且依赖敏感阈值 τ、δ)，得到一个软的、自调节的学习信号。

## 5. 主要解决思路(一段话讲清核心)
在标准 SFT 的逐 token 交叉熵损失前，乘上一个**由模型当前预测熵算出的归一化门控系数** H̃_t∈[0,1]：模型在该 token 越自信(熵越低)系数越接近 0、梯度被抑制；越不确定(熵越高)系数越接近 1、退化为普通 SFT。整个改动**只替换损失函数**，不引入 RL rollout、不引入额外模型、流程与标准全参微调完全一致。

## 6. 方法详解(通俗、分步骤)
**EAFT 损失(论文 Eq.2)**：

\(\mathcal{L}_{\mathrm{EAFT}}(\theta) = -\sum_{t=1}^{T} \tilde{H}_t \cdot \log P_\theta(y_t \mid x, y_{<t})\)

其中前一项 H̃_t 是"自适应门控信号"，后一项是标准监督项。门控信号(Eq.3)用 **Top-K 熵**近似全词表熵以省算力：

\(\tilde{H}_t = H^{\text{top-}K}_t / \ln(K) \approx H^{\text{top-}20}_t / 3.0\)  (K=20，ln(20)≈3.0 为 K 个结果的最大熵，归一化到 [0,1])

H_top-K_t 是在该 token 的 Top-20 概率分布上算的熵。自调节机制(论文原文)：

- **Conflict Suppression（H̃_t→0）**：模型固执(低熵)时权重趋零，屏蔽冲突标签的破坏性梯度；
- **Knowledge Acquisition（H̃_t→1）**：模型不确定/探索(高熵)时权重接近 1，恢复标准 SFT 目标以学习新模式。

与 DFT/FLOW/TALR 的关键区别：用**熵而非概率**作门控，从而避免在 Confident Conflict 上放大梯度。
〔待核：门控系数是否对其本身停止梯度(detach)——论文正文只说"用熵缩放监督"，未显式写 stop-gradient；实现细节需以 LlamaFactory-EAFT 损失代码为准，本地 submodule 为空无法核对。〕

## 7. 实验数据集

- **训练数据构造**：prompt 取自 NuminaMath 与 Nemotron-CrossThink，用 **Qwen3-235B-A22B-Instruct** 对 prompt 合成回答，随机选取 **19k 条被验证为正确**的实例作为数学域训练集(医学/Agent 域各自另构)。
- **Backbone**：Qwen3 与 GLM-4 系列，4B–32B(如 Qwen3-4B-Instruct、Qwen3-4B-Thinking、GLM-4 系列、32B 级)。
- **目标任务评测**：数学(AIME24/AIME25/GSM8K)、医学(MedMCQA/MedQA/PubMedQA)、Agent(BFCL v3)。
- **通用能力(遗忘)评测**：MMLU、IFEval、CLUEWSC。
- **基线**：SFT、SFT-KL(β=0.5)、FLOW、DFT、TALR。

## 8. 实验结果与主要发现

- 以 Qwen3-4B-Instruct 数学域为例(Table 1)：目标任务上 EAFT 的 Math Avg 与标准 SFT 相当(SFT 把 Math Avg 从 68.3 提到 69.4)；**通用域 General Avg 几乎不掉(EAFT 80.1，仅 -1.0；而 SFT 76.5，掉 -4.6)**——即在不牺牲目标提升的前提下把遗忘量级缩到约 1/4~1/5。
- pilot study(硬屏蔽 Confident Conflict)即可减小通用能力下降，验证了"遗忘主因是冲突样本"的假设；EAFT 的软门控比硬屏蔽更好(不丢数据、无敏感阈值)。
- 跨域(医学、Agent)与跨模型(4B–32B)上趋势一致，支撑方法的通用性主张。

## 9. 结果如何支撑其主张
论文的核心因果链是"Confident Conflict → 破坏性梯度 → 遗忘"。pilot study 的屏蔽实验直接验证了"去掉这些 token 就能减小遗忘"，为因果方向提供了证据；主表则证明软门控版本(EAFT)能同时保住目标任务收益与通用能力——两者结合较好地支撑了主张。增益主要体现在**遗忘量(General Avg 下降幅度)**而非目标任务绝对分(目标分与 SFT 基本持平)。

## 10. 逻辑自洽性(中性评估)
机制清晰且自洽：熵门控的两端行为(低熵抑制、高熵学习)与"区分 epistemic uncertainty vs Confident Conflict"的动机直接对应；用 Top-20 熵近似全词表熵、ln(K) 归一化都合理。一个需注意的内在张力：门控只看模型**自身**的熵，并不直接判断"标签是否真的与模型相悖"——低熵但标签恰好一致的 token 也会被同等压权重，这在理论上可能轻微拖慢部分正确知识的学习；论文用"目标任务不掉分"的实验结果间接回应了这一担忧。

## 11. 残留问题 / 局限

- **实现不可直接审计**：核心贡献(熵门控损失)在两个集成分支里，本地 submodule 为空；detach/数值稳定/Top-K 取法等细节只能据论文+README 推断。
- **熵门控可能抑制部分有益学习**：低熵≠冲突，门控对"低熵且标签一致"的 token 也降权，是否最优未深入消融。
- **超参 K 与归一化**：K=20、ln(K) 归一化为经验选择，未见对 K 的系统敏感性分析(正文只给计算开销分析 Sec 5.2)。
- **评测以中英文通用基准为主**：遗忘的衡量依赖 MMLU/IFEval/CLUEWSC 三项，覆盖面有限。

## 12. 开源代码与框架(链接+框架+代码可得性)

- 主仓库 github.com/PRIS-CV/EAFT(README+assets，无独立训练代码)；项目页 ymxyll.github.io/EAFT/。
- **实现集成**：LLaMA-Factory(已合入官方，经 `use_eaft_loss` 启用；作者分支 github.com/ymxyll/LlamaFactory-EAFT)与 ms-swift(github.com/ymxyll/ms-swift-EAFT，支持 megatron/deepspeed)。
- **框架**：标准全参/SFT 流程(DeepSpeed/Megatron/FSDP2/LoRA)，EAFT 仅替换损失，无 RL、无额外模型。
- **代码可得性**：〔本地 clone 的 `resource/repos/eaft/` 下 `LlamaFactory-EAFT/`、`ms-swift-EAFT/` 两个子目录为空(submodule 指针未拉取)，损失实现需到上述 fork 仓库查看，本地无法审计。〕

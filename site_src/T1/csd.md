# csd — Distillation of Large Language Models via Concrete Score Matching (Concrete Score Distillation, CSD)

> **一句话重点 (TL;DR)**：CSD 提出一种 **logit 级**的离线知识蒸馏目标，用"离散 concrete score 匹配"替代传统的 softmax 概率匹配（KL/f-散度），既避免 softmax 把 teacher 的大 logit 差异"压平"，又比直接对齐 logit（DLD）拥有更大的最优解集（对 logit 常数平移保持不变），并给出 O(|V|) 线性时间梯度。

**元信息**：arXiv 2509.25837 ｜ KAIST + summary.ai（Yeongmin Kim, Donghyeok Shin, Mina Kang, Byeonghu Na, Il-Chul Moon）｜ v3 2026-05-30（ICLR 2026 接收，cs.LG）｜ 主题 离线 logit-level KD（与本项目仅在"蒸馏 loss 设计"层面外围相关，非在线/RL 推理蒸馏）｜ 代码 https://github.com/aailab-kaist/CSD（仅 README 占位，**训练代码尚未释出**）｜ 框架 即插式 KD 目标，可嵌入 ImitKD/GKD/DistiLLM

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/csd/fig_01.png)

*Figure 1: Logit vectors, probabilities, and parameter space illustrating how concrete-score distillation (CSD) differs from the divergence-based baseline.*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/csd/fig_02.png)

*Figure 2: Schematic for L CSD (Eq. (8)).*

## 1. 相关工作与进展
知识蒸馏（KD）让小 student 继承大 teacher，以降低推理成本。主流 KD 在每个 token 上做 softmax 概率匹配——最常见是前向/反向 KL，以及各类 f-散度与平滑变体。近期 on-policy KD（ImitKD、GKD、DistiLLM）改进的是"用谁生成的数据训练"（student on-policy / 混合 / 自适应选择），而 CSD 改进的是与之正交的"散度目标本身"。CSD 借鉴能量模型中的 score matching（Hyvärinen 2005）与离散变量的 concrete score（Meng et al. 2022），把后者适配到自回归 LLM 蒸馏。

## 2. 现有工作存在的问题

- **softmax 平滑抹掉 logit 级知识**：当 teacher 的 logit 差异很大时（如 [−1,−4,4] vs [1,−9,6]），经 softmax 后两者概率几乎一致、梯度近乎相同，淹没了 teacher logit 里编码的细粒度知识。在大词表下概率分布极度稀疏——论文统计 GPT-2-1.5B 上仅 **0.0023%** 的 token 概率大于 0.01。
- **Direct Logit Distillation (DLD) 解集受限**：DLD 直接对齐 logit、绕过 softmax，但其最优解**不允许 logit 的常数平移不变性**（而 softmax 推理下 logits 整体加常数等价）。这严重收窄解集，在 teacher/student 容量差异大时尤甚。

## 3. Motivation
能否设计一个 logit 级目标，**同时**克服"softmax 平滑"与"DLD 解集受限"，并带可证明的最优性保证（解集应是 DLD 解集的超集）？

## 4. 主要灵感 / 核心直觉
score matching 在能量模型中可绕开 sum-to-one 归一化约束。把它的离散版本（concrete score，刻画"换到另一 token"的相对概率变化）搬到 LLM 蒸馏上：只要 student/teacher 在**所有词表对**上的相对 logit 差对齐即可——这天然对 logit 常数平移不变，从而比 DLD 多出一整族等价解。

## 5. 主要解决思路(一段话讲清核心)
\(s_\theta(y) = [q_\theta(x)/q_\theta(y)]_{x \in V}\)，把蒸馏目标设为匹配 student 与 teacher 的 concrete score。为适配 LLM，做两处工程处理：(a) \(q_\theta(x)/q_\theta(y_t)\)导致训练不稳，改用其 **log 变换**形式；(b) 朴素双重词表求和是 O(|V|²)，在可分权重假设下降到 O(|V|)。最终目标归约为"匹配所有词表对上的相对 logit 差，权重可调"，并可在同一框架内实例化 mode-seeking 与 mode-covering 两类行为。

## 6. 方法详解(通俗、分步骤)

1. **构造 concrete score**：对每个位置，用 student logits 算出"当前 token 换成词表中任一其它 token"的相对概率比，对 teacher 同理。
2. **log 变换稳定训练**：直接用概率比会发散，改对齐 log 形式（论文 §"adopt the logarithm"），得到 CSD 目标 L_CSD。
3. **理论保证**：
   - **Proposition 1（一致性）**：模型容量趋于无穷时，匹配 log-concrete-score 可使 student 收敛到 teacher。
   - \(\Theta^*_{\mathrm{CSD}} \supsetneq \Theta^*_{\mathrm{DLD}}\)——DLD 能达到的解 CSD 都能达到，且 CSD 因对 logit 常数平移不变而拥有更多解。
   - **Theorem 3（高效梯度）**：在权重可分假设 \(w(y_t, x) = w_1(y_t) \cdot w_2(x)\) 下，梯度可在 **O(|V|)** 线性时间算出（Algorithm 1）。
4. **更一般权重的退路**：若不接受可分假设，可用 **Monte Carlo 估计**梯度（不需独立性假设，但方差更大、收敛略慢）。
5. **即插使用**：把 CSD(S,S) 与 DLD(S) 损失叠加进 ImitKD/GKD/DistiLLM——这三者的差异在于训练数据来源（ImitKD 纯 student on-policy、GKD 混合、DistiLLM 按验证损失自适应选择）。

## 7. 实验数据集

- 蒸馏数据：databricks-dolly-15k（沿用 DistiLLM 设置）；评测集 Dolly Eval、Self-Instruct 等。
- 任务：task-agnostic 指令跟随、task-specific（摘要/数学/翻译）、通用 chat 蒸馏。
- backbone/teacher：GPT-2(0.1B/0.3B/1.5B)、OpenLLaMA-7B、Gemma-7B-IT、Qwen2.5-7B-IT、**Gemma2-9B-IT**（最大 teacher 9B）。

## 8. 实验结果与主要发现

- 先在 dolly 上微调 teacher，再蒸馏 student；ROUGE-L 跨 5 个随机种子取平均，并用 Self-BLEU 衡量多样性。
- 基线覆盖 KL/RKL、各类 f-散度、DLD（及 DLD-mean 中心化变体）、ImitKD、GKD、DistiLLM。
- 结论：CSD 一致优于近期概率匹配目标与 DLD，并位于"多样性–保真"前沿；与 on-policy 技术（ImitKD/GKD/DistiLLM）结合时呈互补增益。

## 9. 结果如何支撑其主张
理论侧（Prop.1 / Thm.2 / Thm.3）直接支撑"解集更大 + 可线性计算"的核心卖点；实验侧通过相对概率匹配目标与 DLD 的对比、以及与三种 on-policy 框架叠加的增益，支撑"logit 级知识被更好保留"的主张。多样性/保真前沿图支撑"不牺牲多样性"的说法。

## 10. 逻辑自洽性(中性评估)
内在逻辑自洽：从"softmax 抹平 logit 差 + DLD 解集受限"两个具体缺陷出发，给出兼顾两者的目标并配理论。但需注意 Thm.3 的 O(|V|) 依赖**权重可分假设**，更一般情形退回方差更大的 Monte Carlo——即"高效"与"通用"二者不可兼得，这一权衡论文有交代。

## 11. 残留问题 / 局限

- 假设 teacher 与 student **共享词表/tokenizer**，跨族蒸馏不适用。
- O(|V|) 仅在可分权重下成立；一般权重需 Monte Carlo（方差更大）。
- 评测以 ROUGE-L/Self-BLEU 等**代理指标**为主，模型规模偏中小（最大 9B teacher），未在大规模推理任务上验证。
- 属**离线 logit-level KD**，与本项目关注的在线/RL/推理路径蒸馏只在 loss 设计层面相关。
- 〔待核：GitHub 代码尚未填充，复现性无法独立验证〕

## 12. 开源代码与框架(链接+框架+代码可得性)

- 仓库：https://github.com/aailab-kaist/CSD （已 clone，~83KB；**当前仅 README 占位**"Official repo for CSD (ICLR 26)"，无训练代码）。RepoExists=YES 但内容近乎空。
- 框架：非独立训练框架，而是一个可替换 KL 的 **logit-level 蒸馏目标**，设计为嵌入现有 KD 流程（ImitKD/GKD/DistiLLM）。

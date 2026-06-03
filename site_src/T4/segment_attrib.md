# segment_attrib — Segment-Level Attribution for Selective Learning of Long Reasoning Traces

> **一句话重点 (TL;DR)**：用 integrated gradients 直接量化每个 token 对"正确答案预测"的贡献，聚合为段落级"强度 + 方向一致性"两指标，挑出"高强度但中等一致性"的反思性段落做 loss-mask 选择性 SFT；相对 full-CoT SFT 准确率最高 +4.7%、长度最多 −18%，但温度采样下增益明显收窄，框架本身沿用 Rho-1 selective SFT。

**元信息**：arXiv 2602.00425v1（2026-01-31 提交） ｜ University of Southern California（Siyuan Wang, Yanchen Liu, Xiang Ren） ｜ PDF 页眉标注 "Published as a conference paper at ICLR 2026" ｜ 主题 长 CoT 的 SFT 段落级选择性学习 / credit-assignment（与 OPD/蒸馏"不是所有 token 都值得学"直接相关） ｜ 代码 https://github.com/SiyuanWangw/SegmentSelectiveSFT （已克隆，约 81MB，Attribution/SelectiveSFT/Eval 三阶段齐全） ｜ 框架 SFT 用 unsloth+trl，归因阶段自写 IG 脚本，评测含 latex2sympy

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/segment_attrib/fig_02.png)

*Figure 2: Change ( ∆ ) in correct answer confidence across different segment types.*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/segment_attrib/fig_03.png)

*Figure 3: Log perplexity, entropy and BLEU similarity of important versus unimportant segments.*

## 1. 相关工作与进展
长 CoT（o1/R1/Qwen 系）靠 test-time scaling 提升推理，也成为 cold-start SFT 的监督资源（s1/Muennighoff 2025）。围绕"识别长链中重要部分以构造压缩监督"已有：token 级分析（TokenSkip/Xia 2025b）、段落级 perplexity（Cui 2025a）、段落级 entropy（Li 2025b）。selective SFT（Rho-1，Lin 2024）提供"只在部分 token 上算 loss、其余 mask"的框架。段落切分沿用 Lu 2025 的转折关键词法。

## 2. 现有工作存在的问题

- token 级度量忽略语义完整性，不构成可解释推理单元。
- 段落级 perplexity/entropy 是与真实重要性"不完全一致"的间接指标：既有假阳（过度强调"让我们一步步算"这类脚手架桥接文本——删它会破坏后文连贯但贡献甚微），也有假阴（漏掉独立验证/中间结论这类低熵、删除不影响流畅度、但显著提升正确答案概率的段落）。
- 基于剪枝的压缩监督方法（删冗余）往往掉精度。

## 3. Motivation
需要一个直接度量"段落对正确答案预测的影响"的指标（能同时捕捉直接与间接贡献），并以可解释的段落为单位，从而比 perplexity/entropy 更准地区分真正重要段落与各类冗余。核心实证：30~40% 的段落累计贡献 >80% 的总归因（correct/incorrect CoT 皆然，Fig.1 right-bottom CDF），说明长 CoT 存在大量冗余。

## 4. 主要灵感 / 核心直觉

- 用 IG（Sundararajan 2017）沿 baseline→真实嵌入的直线路径积分梯度，捕捉 token 的直接+间接影响，优于"顺序追加段落看答案概率变化"或 leave-one-out（后两者会低估间接贡献且对后文完整时不敏感）。
- 方向一致性的直觉：极高一致性（token IG 几乎全正或全负）= 浅层/已定方向（冗余澄清或严重错误探索）；中等一致性 = 混合支持与纠正的反思性推理，更有学习价值。

## 5. 主要解决思路(一段话讲清核心)
按转折关键词切段 → 对每个 token 算 IG 归因 → 聚合为段落级"强度"与"方向一致性"两指标 → 按强度降序取累计达阈值 τ 的 top-k 段落、再用一致性阈值 β 过滤掉极高一致性者，得到"重要段落集" → 只在重要段落 token 上算交叉熵、其余 mask，做全参选择性 SFT。

## 6. 方法详解(通俗、分步骤)

- **切段**：按 "\n\nWait"、"\n\nAlternatively" 等转折关键词将 CoT 切为 {S1..Sn}（完整关键词表见 Appendix C.2）。
- **IG 归因**：IGi(x)=(xi−x'i)·∫∂F/∂xi dα，J 步插值近似，baseline x' 取 padding token embedding；token 归因 IG(x)=Σ_i IGi(x)。用绝对值捕捉影响幅度（负 IG 可能是必要的探索性推理，不应丢弃）。
- **两个段落指标**（Eq.3）：Strength(S)=Σ|IG(on)|/√N（√N 长度归一防偏长段），再在 CoT 内跨段归一化（Eq.4）；Consistency(S)=|Σ IG(on)| ÷ Σ|IG(on)|。
- **重要段落判据**（Eq.5-7）：按归一化 strength 降序，取累计 ≥ τ 的最小 top-k* 段落集，其中 Consistency ≤ β 者为"重要"。
- **超参**：贪心搜索（最大化重要/不重要段落的"正确答案置信度变化 ∆"之差）定 τ=0.7、β=0.8；τ=0.7 时平均 ~33% 段落被判重要、占 CoT 中 ~45% token（重要段落偏长）。
- **选择性 SFT**（Eq.9）：L=−(1/Σ I(ot))Σ_t I(ot)·log P(ot|·)，I(ot) 标记 token 是否属重要段落；mask 其余、保留完整轨迹连贯性。

## 7. 实验数据集

- 训练：LIMO 数学数据集 817 题（用 provided CoT 或 R1-Distill-Qwen-7B 自生成、从 32 候选取最短正确/错误解）。
- 评测——In-domain：MATH500、AMC23、AIME24；Out-of-domain：GPQA-Diamond、Minerva、OlympiadBench。指标：greedy 准确率，或温度采样 T=0.6、max_len 32768 下 pass@1/pass@6，并报平均输出 token 数。
- 基座：R1-Distill-Qwen-1.5B/7B、Qwen2.5-7B-Instruct。IG 归因模型与训练基座一致（1.5B/7B），IG_STEPS=50。

## 8. 实验结果与主要发现

- **段落分析**（§3）：高强度+中等一致性段落带来最大的正确答案置信度增益（Fig.2）；重要段落 perplexity/entropy 反而更低（Fig.3）；不重要段落 BLEU 自相似更高（重复），49% 被判截断 vs 重要段落仅 26%。
- **主结果**（Table 1，greedy）：R1-Distill-Qwen-1.5B overall 44.8→46.9（+4.7%），长度 16520→13506（−18.2%）；7B 62.1→（部分基准已见 MATH500 91.2→95.2、AIME24 50.0→56.7）。
- 消融：段落级优于 token 级 IG 选择、优于随机段落、优于仅取高 strength；对比 First-Correct-Solution / Confidence-Gain / Perplexity / Entropy 等度量均更优。

## 9. 结果如何支撑其主张
"IG 段落归因能识别真正重要段落"由 §3 三类证据支撑（置信度增益、低 perplexity/entropy、低重复率），且消融证明两指标（strength+consistency）联合优于单指标与各 baseline 度量；"选择性学习提升精度+效率"由 Table 1 的 +4.7%/−18% 支撑。逻辑链完整，但 greedy 下的增益在温度采样下显著缩水。

## 10. 逻辑自洽性(中性评估)
方法-动机自洽：用 IG 直接归因解决"间接指标不一致"的痛点，用"中等一致性"操作化"反思性推理"。一个张力点：§3 称重要段落 perplexity/entropy 更低，恰说明"低熵≠不重要"，反向支撑其相对 entropy 度量的优势，但也意味着 strength 与 entropy 度量在某些段落上结论相反，可解释性有赖 IG 假设成立。

## 11. 残留问题 / 局限

- 框架创新有限："selective SFT + loss mask"直接来自 Rho-1（Lin 2024），本文核心新意在"IG 归因 + strength/consistency 两指标"的段落重要性度量。
- 解码敏感：温度采样下增益明显收窄（1.5B pass@1 仅 +1.6%、7B 仅 +0.5%），作者归因于采样随机性抹平训练优势，说明部分增益对解码策略敏感。
- 算力开销：IG 需 J=50 步插值，成本不低，论文未量化归因阶段的额外算力。
- 泛化证据有限：仅数学 LIMO 单一训练源、≤7B 规模。
- 超参可迁移性：τ/β 由 1.5B 的 confidence-gain 贪心搜索确定后直接套用到所有模型/数据源，论证较弱。

## 12. 开源代码与框架(链接+框架+代码可得性)

- 仓库 https://github.com/SiyuanWangw/SegmentSelectiveSFT （已克隆，约 81MB）。
- 三阶段脚本齐全且核心可定位：(1) Attribution——`segment_split.py` 切段、`grad_analyze.py` 算 IG、`get_important_segments.py` 按 τ/β 聚合选段、`cal_attribution.sh` 串流程；(2) SelectiveSFT——`train_mask.py`（`--mask` 开关，对非重要段落把 labels 置 −100 实现 loss-mask 全参 SFT；requirements 锁定 unsloth==2025.11.3 + trl==0.23.0 + transformers 4.57.1），`run_train.sh` 启动；(3) Eval——`evaluate.py`/`grader.py` + `latex2sympy/` 子目录，`CoT_generation.sh`/`run_eval` 评测。
- 代码与论文方法一致，loss-mask 实现已确认；归因脚本可运行，复现门槛主要在 IG 算力。

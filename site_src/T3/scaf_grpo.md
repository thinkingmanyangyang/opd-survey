# scaf_grpo — Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning

> **一句话重点 (TL;DR)**：针对 RLVR 的"学习悬崖"(难题持续零奖励→GRPO advantage 坍缩→梯度消失),只在学习停滞时按"知识→规划→解答"三层、增量注入 in-prompt 提示,让**当前策略自己**采样出成功轨迹来替换失败轨迹,从而在保持 on-policy 一致性、不破坏探索自主性的前提下攻克长尾难题。

**元信息**：arXiv 2510.19807 (v2, 2026-02-28) ｜ HKUST / CUHK / HKU(Xichen Zhang*、Sitong Wu*、Yinghao Zhu、Haoru Tan、Shaozuo Yu、Ziyi He、Jiaya Jia 通讯;*共同一作) ｜ ICLR 2026 接收 ｜ 主题 T3(RLVR 算法/引导式探索),与 T1 教师引导相关,Relevance=High(其"教师按需提供分层最小提示、保 on-policy、保探索"与本项目 TSRD 路径选择思路高度契合) ｜ 代码 github.com/JIA-Lab-research/Scaf-GRPO(已克隆 ~7.2MB,Tier A,origin 已核) ｜ 框架 veRL 0.4.1.dev(vLLM rollout)

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/scaf_grpo/fig_01.png)

*Figure 1: Scaf-GRPO overcomes the learning cliff with minimal guidance, outperforming vanilla GRPO (Shao et al., 2024) and the prefix-based LUFFY (Yan et al., 2025) across challenging math benchmarks on Qwen2.5-Math-7B. By injecting strategic, hierarchical hints, our method unlocks the model's poten*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/scaf_grpo/fig_03.png)

*Figure 3: Overview of the Scaf-GRPO framework. For a given query, the model generates multiple solutions. (Left) If any solution is correct, standard GRPO proceeds. (Right) If all solutions fail (the learning cliff), Scaf-GRPO initiates hierarchical hint-guided exploration. It injects progressively*

## 1. 相关工作与进展
RLVR(DeepSeek-R1 范式)用稀疏 binary 结果奖励即可让模型自主习得推理策略,免去逐步人工标注。后续工作沿两条线推进:(a) 算法稳定性/去偏(Dr.GRPO、DAPO 等);(b) 更稠密/更有信息量的奖励(长度惩罚抑制 overthinking、token 级稠密反馈)。针对"学习悬崖",已有一类 **off-policy 教师引导**工作:LUFFY(整条专家轨迹与多条 rollout 混批)、cosine-decay 前缀长度调度(Huang et al.)、多级不同长度 hint(Zhang et al.)等,主流形式是"golden 解前缀续写"。

## 2. 现有工作存在的问题
- **学习悬崖(learning cliff)**:面对远超当前能力的题目,所有探索尝试持续失败 → 持续零奖励 → GRPO 中 advantage 坍缩为 0 → 梯度消失,这些难题对策略更新"不可见",形成训练停滞的长尾瓶颈(论文 Figure 2 实证 Qwen2.5-Math-1.5B 上 vanilla GRPO 在零奖励题上停滞)。
- 现有"前缀续写"应对带来两弊:(1) 教师前缀与学生后缀**分布失配**,需 policy shaping / 混合 SFT-RL 等补丁,引入偏置与不稳定;(2) **"on-rails"**把模型逼上预定路径,扼杀对替代/更优策略的探索。

## 3. Motivation
借鉴教育学**脚手架(Scaffolding, Berk & Winsler 1995)**理论:仅在学习停滞时提供"会随能力提升而撤除"的临时、最小、分层引导,以"路标(signpost)"而非"铁轨(railroad)"方式引导。两条设计目标:(a) **policy consistency**——同一统一策略同时处理"题目+提示",避免前缀法的分布失配;(b) **保留探索灵活性**——提示只指方向不定路径。

## 4. 主要灵感 / 核心直觉
"最小有效引导"原则:奖励模型**用尽可能抽象的提示**解出问题,促其内化推理技能而非记忆解法。提示分三层(抽象→具体)、增量给出而非一次性给全,既精确诊断学生缺口,又最大化技能迁移。

## 5. 主要解决思路(一段话讲清核心)
两阶段:先用"豁免期"区分真难题与伪难题,只对**真难题**触发引导;引导时按"知识→规划→解答"三层、由抽象到具体增量注入 in-prompt 提示,直到当前策略 π_θ **自己**采样出正确解 o*_h,用该成功轨迹替换 rollout buffer 中某条失败轨迹。因为成功轨迹仍是当前策略在 hint-augmented prompt 上采样所得、重要性比率也对该 prompt 计算,所以整个过程仍是 **on-policy**(而非教师 off-policy 续写),既给出有意义学习信号又指向模型可达的最高效推理路径。

## 6. 方法详解(通俗、分步骤)
1. **Phase 1 — 诊断 true-hard 问题(guidance exemption period)**:训练前 **15% 步数**纯 on-policy 探索。监控"零奖励 query 被解决的速率",一旦该速率停滞,仍持续失败的题判为 **true-hard**(区别于因格式/早期技能不熟而"多训即可解"的 pseudo-hard),才成为引导候选。
2. **Phase 2 — 分层 hint 引导**:预定义三层提示 H = {H_knowledge(知识)、H_planning(规划)、H_solution(解答)},抽象→具体。框架做确定性搜索:从最抽象层起、层内增量给提示,一旦模型生成正确解即终止,得到**最小有效引导**。提示为 in-prompt 注入(不破坏 on-policy)。
3. **训练机制**:在 veRL 上保持 GRPO on-policy 性质,仅在停滞时**策略性扩充 rollout buffer**(用 π_θ 自采样的 o*_h 替换一条失败轨迹)。KL penalty=0 以最大化探索;hint-guided 探索仅对 17.4% 样本触发(Qwen2.5-Math-7B)。

## 7. 实验数据集
- **训练**:派生自 DeepScaleR-Preview-Dataset,按每模型初始能力动态过滤(Too Easy 丢弃 / Too Hard 保留 / Potentially Solvable 50% 子采样);三层 hint 由 prompt DeepSeek-R1 基于 ground-truth 解题步骤生成。
- **评测**(7 基准,pass@1 greedy decoding):AIME24、AIME25、AMC、Minerva、MATH-500、OlympiadBench、GaoKao2023en;OOD 泛化用 **GPQA-Diamond**(Table 6)。
- **模型**(Table 1,5 个):Qwen2.5-Math-7B、Qwen2.5-Math-1.5B、Qwen2.5-7B、Llama-3.2-3B-Instruct(非 Qwen 架构)、DeepSeek-R1-Distill-Qwen-1.5B(长 CoT)。跨架构/规模(1.5B–7B)/专长验证。

## 8. 实验结果与主要发现
- **主结果**(Qwen2.5-Math-7B):整体相对 vanilla GRPO **+12.6%**、相对 LUFFY **+9.2%**;AIME24 pass@1 较 vanilla GRPO 相对 **+44.3%**;均值 0.509,较 Simple-RL +19.5%、较 Oat-Zero +9.5%(Figure 1 / Table 1)。
- **OOD(GPQA-Diamond, Table 6)**:Qwen2.5-Math-7B 较 vanilla GRPO **+15.5%**(且与 LUFFY 持平);Qwen2.5-7B-Base **+7.5%**(且超 LUFFY)。
- **消融**:去 Solution 层掉 5.7%(三层中最严重),证三层 KPS 互补;增量给提示优于一次性 "Full Hint"(后者掉 6.3%)。
- **效率**:Scaf-GRPO 约 12h 达最佳 checkpoint(50.9% avg.),vanilla GRPO 需 13h 才达更低峰值;计算开销小(仅 17.4% 样本触发引导)。

## 9. 结果如何支撑其主张
"攻克学习悬崖"由 Figure 2 训练动态(持续解掉零奖励题)与 7 基准一致增益共同支撑;"跨架构/规模/专长普适"由 5 个模型(含 Llama、长 CoT 1.5B)的一致提升支撑;"内化而非记忆"由"去 Solution 层降幅最大 + 增量优于 Full Hint"两个消融支撑;"保 on-policy 优于前缀法"由相对 LUFFY +9.2% 与 GPQA OOD 增益支撑。逻辑链较完整。

## 10. 逻辑自洽性(中性评估)
方法叙事与实验对齐,核心机制(自采样替换→保 on-policy)在概念上成立。但有几点需中性看待:(1) true-hard/pseudo-hard 的判定靠"零奖励解决速率停滞",阈值与窗口是经验设定,论文未给敏感性分析;(2) "最小有效引导促内化"是合理叙事,但缺直接证据排除"学生在 hint 下记住了该题"——消融只在层级粒度,未做"训练题 vs 持出题"的记忆探针;(3) 主结果多基于 best checkpoint 报告,存在 checkpoint 挑选偏差风险(基线同口径,影响相对可比)。

## 11. 残留问题 / 局限
- 引导依赖一个更强教师(DeepSeek-R1)生成分层 hint,真"零外部知识"并不成立;hint 质量上限受教师约束。
- 仅在数学域验证;代码/通用推理是否同样有效未测(GPQA 仅作 OOD 评测,非训练域)。
- 15% 豁免期、KL=0、温度等超参的鲁棒性未系统消融;〔待核〕组大小、lr 等完整超参在附录(正文未全列)。
- 报告 best checkpoint 而非固定步数,跨方法早停策略一致性需信任作者实现。

## 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库:https://github.com/JIA-Lab-research/Scaf-GRPO(本地已克隆 ~7.2MB,Tier A;origin 已核为该地址)。
- 框架:**veRL 0.4.1.dev**(仓库含 `verl/` 与 `hint_mix_grpo/{main_ppo.py, trainer/, dataset/, config/}`);rollout 用 vLLM。README 安装步骤列出 SGLang、Megatron-Core,但**明确注明本实现禁用、不需要**(原分析称其为"核心依赖"略夸大,已更正)。conda env `scaf-grpo`(python 3.10)。
- 入口:`train.sh`(conda activate `scaf-grpo` → `bash sh/hint_mix_grpo/bs256_6k_mix.sh`);提示混合逻辑在 `hint_mix_grpo/`,baseline/评测脚本在 `sh/{baseline,generation_eval}`。
- 数据集:HuggingFace `hkuzxc/scaf-grpo-dataset`。

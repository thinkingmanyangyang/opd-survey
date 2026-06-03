# search_dont_guess — Search, Do not Guess: Teaching Small Language Models to Be Effective Search Agents

> **一句话重点 (TL;DR)**：小模型(SLM)做 search agent 时反而比大模型**更少搜索、更易幻觉**(under-searching),且"自适应搜索"在 SLM 上会掉点(Adaptive Search Trap);本文用 **Always-Search Policy(ASP)** 显式约束"总是搜索、别猜",经 SFT/OPD/Mixed 三种实现把搜索行为蒸进 SLM,使 1.7B 逼近甚至局部超过 8B。

**元信息**：arXiv 2604.04651 (v1, 2026-04-06, cs.AI) ｜ NYU + UIUC + NTU(Yizhou Liu*、Qi Sun*、Yulin Chen、Siyue Zhang、Chen Zhao;*共同一作) ｜ arXiv 预印本 ｜ 主题 agentic RAG / SLM 蒸馏,Relevance=Med(把 **OPD** 作为三训练变体之一与 SFT、Mixed 实测对比,提供 OPD 在"让小模型坚持搜索"上的具体实证) ｜ 代码 github.com/yizhou0409/Agentic-Rag(100% Python) ｜ 框架 检索 E5+BM25 / Search-o1 式迭代 agent / 训练 SFT+OPD+RFT

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/search_dont_guess/fig_01.png)

*Figure 1: Left: With Always-Search Policy, the distilled SLM significantly narrows the performance gap with the teacher model. Right: SLMs suffer from adaptive search and ASP is the most effective policy.*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/search_dont_guess/fig_02.png)

*Figure 2: Scaling of agentic search performance. Small models trail behind in both performance and retrieval capability on HotpotQA.*

## 1. 相关工作与进展
带搜索工具的 agent 是知识密集任务的有效方案,但依赖 ≥7B LLM,部署成本高。近期工作把 agentic 行为蒸到 <4B SLM(Agent Distillation 路线)。本文在此之上做全面评测并提出针对 SLM 的训练 recipe。

## 2. 现有工作存在的问题
全面评测发现:尽管参数知识更少,SLM 反而**更少调用搜索、更易幻觉**(under-searching,Figure 2)。直接蒸 LLM 轨迹效果差——LLM 轨迹常含"I remember…"式不搜索直接答的行为,被 SLM 学坏。更关键:让模型自己决定要不要搜的**自适应搜索在 SLM 上反而掉点**("Adaptive Search Trap")。

## 3. Motivation
作者用 **confidence probe** 检验 SLM 内部知识可靠性:SLM 高置信 query 远少于 teacher。⇒ SLM 不该依赖内部知识"猜",应"总是搜索、别猜"。需要一种训练显式约束搜索行为、强制 grounded 生成。

## 4. 主要灵感 / 核心直觉
对 SLM 而言,"一致地总是搜索"优于"自适应地按置信度决定是否搜索"——后者把不可靠的内部置信引入决策,放大幻觉。把这一先验直接编码进训练目标。

## 5. 主要解决思路(一段话讲清核心)
提出 **Always-Search Policy(ASP)**:训练时显式约束搜索行为,三种实现——(1) **SFT**:只保留"始终用搜索工具获取信息"(剔除"I remember"类)的高质量轨迹做监督;(2) **OPD(on-policy distillation)**:不显式过滤轨迹,而是用 system prompt 要求模型总是搜索,期望 teacher 的 log-prob 分布在 student 自身 rollout 上调控其搜索行为;(3) **Mixed**:先 ASP-SFT 再 OPD 强化。下游再加 **RFT(Rejection Fine-Tuning)** 选择性强化高质量 agentic 行为。整体是工程化 recipe,非新算法。

## 6. 方法详解(通俗、分步骤)
- 推理:Search-o1 式迭代 search agent,检索 = E5(embedding)+ BM25(keyword)双检索,summarizer = Qwen3-32B。
- **ASP-SFT**:从 HotpotQA 采 **18,000** 条 Qwen3-32B teacher 轨迹(String-F1>0.65 保留;且只留"始终搜索"轨迹),**3 epochs、lr 1e-5** 蒸馏。
- **ASP-OPD**:**3,000** 个 HotpotQA 训练题,每题学生采样 **8** 条轨迹,teacher 给 token 分布、最小化 KL,**4 epochs、lr 2e-6**;靠 system prompt 强制搜索而非显式过滤。
- **RFT**:学生自生成 **10,000** 条轨迹(与蒸馏用不同题),**2 epochs、lr 5e-6**,拒绝采样保留高质量行为。

## 7. 实验数据集
- **训练仅用 HotpotQA 训练集**(teacher = **Qwen3-32B**)。
- 评测:结构化多跳 HotpotQA / 2WikiMultiHopQA / Bamboogle / MuSiQue;agentic 信息检索 BrowseComp-plus / Frames / LongSeAL。指标 **String-F1**(也报 EM)。
- 检索器 e5-large-v2 + fullwiki-20210620 语料(BrowseComp-Plus 用 Qwen3-Embedding-8B + 自带语料)。
- 模型:Qwen3-0.6B/1.7B/4B/8B/32B、Llama-3.2-1B/3B。

## 8. 实验结果与主要发现
- **逼近大模型**(Table 1,String-F1):三种 ASP 法均使 1.7B 逼近 Qwen3-8B——HotpotQA 上 Mixed-1.7B = **58.2** ≈ 8B(58.2);**2Wiki 上 OPD-1.7B = 62.9,超过 8B(58.1)**。(注:上一轮已更正"OPD 56.2→62.9"的误读——56.2 是 OPD 的 HotpotQA 分、62.9 是其 2Wiki 分;本轮再核 Table 1 OPD-1.7B 行 = 56.2/62.9/61.4/… 属实。)
- **泛化**:只在 HotpotQA 训练却泛化到 OOD(BrowseComp/Frames/LongSeAL)。
- **搜索频率**(§4.2):vanilla 1.72 → ASP-SFT 2.47 → ASP-OPD 2.84 搜索/题(Vanilla-8B 亦 2.84)。OPD 在多个表上搜索频率最高,是本文 OPD 有效性的具体实证点。
- **噪声鲁棒**(10% 检索失败):vanilla SLM / 普通蒸馏掉 **12.1** 分,ASP 仅掉 **2.3 / 1.7**,显示更强恢复能力。
- **Adaptive Search Trap**:Adaptive Distill(Top-5/10/20% confidence)在 Bamboogle 等上劣于一致搜索的 ASP。

## 9. 结果如何支撑其主张
"SLM under-search"由 Figure 2 + confidence probe 支撑;"一致搜索优于自适应"由 Adaptive Distill 各档掉点支撑;"ASP 让 SLM 逼近 LLM"由 1.7B≈8B、2Wiki OPD>8B 支撑;"更 grounded/抗噪"由噪声实验 2.3/1.7 vs 12.1 支撑。支撑较直接。但"训练仅用 HotpotQA"既是泛化卖点也是局限——单源单任务,泛化结论的可靠性受训练分布单一性限制。

## 10. 逻辑自洽性(中性评估)
recipe 与诊断对齐,叙事自洽。中性看待:(1) 三种 ASP 实现孰优缺乏统一胜者(SFT/OPD/Mixed 在不同 benchmark 互有高低),论文未给清晰选择准则;OPD 的优势主要体现在搜索频率与个别 benchmark,而非全面占优;(2) "总是搜索"在 query 本可由内部知识快速回答时会增加延迟/成本(Table 5 显示 OPD-1.7B 端到端 ~3.1s),论文承认但未量化"过度搜索"代价;(3) 仅 Qwen3 系评测,跨家族结论保守(论文亦自述局限)。

## 11. 残留问题 / 局限
- 训练单一(仅 HotpotQA 单跳/多跳混合),跨域/跨任务训练分布缺失。
- "总是搜索"对简单 query 引入不必要搜索开销,无最优搜索预算分析。
- 检索噪声处理仅做 10% 失败注入的鲁棒性测试,缺更系统的对抗检索机制(论文列为 future work)。
- 评测集中于 Qwen3 家族(Llama-3.2 仅部分),跨架构普适性未充分验证。
- 仓库以语料/检索/推理脚手架为主,SFT/OPD/RFT 训练实现细节相对简略(在 appendix B)。

## 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库:https://github.com/yizhou0409/Agentic-Rag(100% Python)。含 `build_corpus/`(`extract_wiki.py`、`build_e5_corpus.py`、`merge_e5_splits.py`:Wikipedia 语料构建/索引)、`e5_retriever.py` + `bm25_retriever.py`(双检索)、`main.py`(推理 pipeline)、`prompts/`(`default_QA.yaml`、`default_retrieval_summary.yaml`、`free_QA.yaml`)、`utils.py`。
- 框架:检索 E5(embedding)+ BM25(keyword);推理 Search-o1 式迭代 search agent;训练含 SFT、OPD、RFT 三阶段(trainer 实现见 appendix,仓库内偏脚手架)。
- 代码可得性:Tier B——语料/检索/推理可跑,但训练(SFT/OPD/RFT)脚本不完整,复现需补 appendix B 细节。

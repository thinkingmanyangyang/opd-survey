search_dont_guess | Search, Do not Guess: Teaching Small Language Models to Be Effective Search Agents | NYU + UIUC + NTU(Yizhou Liu*、Qi Sun*、Yulin Chen、Siyue Zhang、Chen Zhao) | 2026-04-06 arXiv 2604.04651(v1, cs.AI)·预印本 | 主题线 L4(Agent/工具多轮)+L1(OPD/蒸馏)·相关性 中

**原始论文**:https://arxiv.org/abs/2604.04651

## 一眼看懂
- 🟦 TL;DR:小模型(SLM,<4B)做"带搜索的 agent"时反直觉地**比大模型更少搜索、更爱拿内部知识瞎猜(parametric hallucination)**,直接蒸 LLM 轨迹只涨一点(F1 50.6→53.2)。本文提 **Always-Search Policy(ASP)**:训练时显式强制"凡事都先搜、别用脑内知识猜",用三种实现(ASP-SFT / ASP-OPD / Mixed)把"坚持搜索"这个先验编码进 SLM,使 Qwen3-1.7B 逼近甚至局部超过 8B(2Wiki 上 OPD-1.7B=62.9 > 8B 的 58.1)。
- 最巧的一步:**ASP 这个"总是搜索、知识置零"的硬约束本身**(配套 system prompt "You are a Knowledge-Free agent... MUST search EVERY entity")。抽掉它就退回普通 agent distillation(只涨 50.6→53.2),under-searching 与幻觉照旧。论文还有一记反证:让 SLM "自适应"决定要不要搜(Adaptive Search Trap),反而掉点——说明"一致地搜"才是 SLM 该有的策略。【原文】§1、§4、§5、Fig.1、附录 D 的 prompt

## 为什么做
- 研究背景:带搜索工具的 agent(如 Search-o1)解知识密集多跳问题很强,但依赖 ≥7B LLM,延迟/预算高;部署偏好 <4B 的 SLM(Belcak 2025)。中心问题:agentic search 能力怎么有效蒸进 SLM?【原文】§1
- 解决的具体痛点:全面评测(Fig.2)发现 SLM 参数知识更少却**更少调搜索工具**——要么幻觉答案,要么连工具调用的语法格式都写不对(structural constraint 跟不上)。直接蒸 LLM 轨迹收效甚微,因 LLM 轨迹里常含"I remember…"式不搜直接答的步骤,被 SLM 学坏。【原文】§3.1、§3.2、附录 C
- 相关工作 & 各自不足(并行路线 + 各自短板 + 站谁肩上 + 与最近邻精确差异):
  - **(路线 A)CoT 蒸馏**(Wei 2023、Magister 2023):把大模型的推理链蒸给小模型提推理。**短板**【原文 §6】——只提"推理"不解"搜索行为";小模型即便会推理,仍会 under-search + 用内部知识猜。
  - **(路线 B)工具使用 / agent 蒸馏**(FireAct、AgentTuning、Agent Distillation/Kang 2025):模仿 teacher 的成功工具调用轨迹。**短板**【原文 §6、§3.1】——"**保留 teacher 本性**",而 teacher 本性恰恰**会用内部知识不搜**(LLM 轨迹含 "I remember" 步骤),SLM 把这坏习惯一并学走。Search-dont-guess 站在 Agent Distillation(Kang 2025)肩上,但**显式过滤"始终搜索"轨迹 + 强制 system prompt**改掉这一点。
  - **(路线 C,最近邻/反向对照)Adaptive / Selective Search**(Eisenstein 2025 等):按模型置信度省检索(高置信就不搜),省成本。**短板/被反驳**【原文 §5、Table 2】——本文实证 SLM 上**自适应反而掉点**(P=5% 时 SFT-1.7B 掉 4.8、P=10% 掉 15.0,而 32B 到 P=10% 仍稳),因为 SLM 内部置信**不可靠**(confidence probe 显示其高置信 query 远少于 teacher)。本文差异:把"是否搜索的决策权"从不可靠的 SLM 内部置信中**直接拿走**,改硬约束总搜。
  - **(路线 D)on-policy distillation 框架**(Agarwal 2024 GKD;Lu & Lab 2025 / Thinking Machines blog):学生采样、teacher 在其 rollout 上给 token 分布做 KL。本文 ASP-OPD **直接沿用**这个框架,新意在"用 system prompt 而非数据过滤来注入'总搜'先验"。
  - **与最近邻精确差异**:vs 标准 Agent Distillation(Kang 2025)——不只是模仿成功轨迹,而是**显式过滤"始终搜索"轨迹 + 强制 system prompt**;vs Adaptive Search(Eisenstein 2025)——本文实证 SLM 上自适应反而掉点,给"SLM 该一致搜索"的反向证据。承重的一点就是**把决策权从 SLM 内部置信拿走、硬性总搜**。
- 动机链:SLM 部署省成本 → 但 vanilla SLM under-search+幻觉 → 普通蒸馏没用(学到 teacher 的"不搜")→ 用 confidence probe 证 SLM 内部知识不可靠(高置信 query 远少于 teacher,§5/附录 F)→ 所以应"总是搜、别猜",把这先验直接编码进训练。

## 怎么做(到可复现粒度)
- **推理底座 = Search-o1 式迭代 agent**:`<search>`query`</search>` → 检索 top-k → `<information>`摘要`</information>` 拼回 → 继续推理 → `<answer>`。检索器 = **E5(e5-large-v2)+ BM25 双检索**;summarizer = **Qwen3-32B**;语料 = fullwiki-20210620。
- **ASP 推理时的强制循环(Algorithm 2,可复现)**【原文 附录】:输入 query \(x\)、策略 \(\pi_\theta\)、工具 \(S\)、约束 prompt \(P_{\mathrm{force}}\)、最大重试 \(K\)。
  1. 初始轨迹 \(\tau\leftarrow[x\oplus P_{\mathrm{force}}]\)(**把"强制依赖工具"的约束 prompt 全局注入到开头**)。
  2. **for** hop \(t=1,\dots,H\):
     - 内层 repeat:采样响应 \(s_t\leftarrow\pi_\theta(\tau)\),\(k\leftarrow k+1\);**直到** \(s_t\) 含 action 或 含 answer,或 \(k\geq K\)。
     - 若仍既无 action 又无 answer → **Raise Error 丢弃该轨迹**(强制搜索失败)。
     - 若 \(s_t\) 含 `<search>q_t</search>`:\(o_t\leftarrow S(q_t)\),\(\tau\leftarrow\tau\oplus s_t\oplus\)`<information>\(o_t\)</information>`。
     - 若 \(s_t\) 含 `<answer>y</answer>`:返回 \(y,\tau\)(成功)。
  3. 到 \(H\) hop 上限仍未答 → Failure。
  关键:**\(K\) 次重试 + 失败即丢**是"强制总搜"的运行时保障——逼模型在 action/answer 间二选一,堵住"既不搜也不答"的退路。
- **三种把 ASP 注入训练的变体**【原文 §4.1、B.2】:
  1. **ASP-SFT(过滤式)**:从 HotpotQA 训练集采 **18,000** 条 Qwen3-32B teacher 轨迹,**两道过滤**——① String-F1 > **0.65** 留对的;② **search tool checking + keyword filtering**——只留"始终用搜索工具、不出现 'I remember' 这类不搜直答"的轨迹。在过滤后轨迹上做标准 SFT(序列级 CE),损失即
     \(\displaystyle \mathcal{L}_{\text{ASP-SFT}}(\theta)=-\sum_{t}\log\pi_\theta(y_t\,|\,y_{<t},x)\quad\text{(仅在通过 ASP 过滤的轨迹上)}\)
     超参:**3.0 epochs,AdamW,lr 1e-5,batch 4**。
  2. **ASP-OPD(on-policy distillation,prompt 式)**:**不显式过滤轨迹**,改用 system prompt 强制总搜,**靠 teacher 的 log-prob 分布来约束学生行为**。流程:HotpotQA **3,000** 题,每题**学生自己采 8 条轨迹**,以 **4 题/batch** 把这些 on-policy 轨迹送 teacher 取 token 概率分布,**最小化 KL 散度**:
     \(\displaystyle \mathcal{L}_{\text{ASP-OPD}}(\theta)=\mathbb{E}_{x}\,\mathbb{E}_{y\sim\pi_\theta(\cdot|x,P_{\mathrm{force}})}\Big[\textstyle\sum_t D_{\mathrm{KL}}\!\big(\pi_{\text{teacher}}(\cdot|y_{<t},x)\,\|\,\pi_\theta(\cdot|y_{<t},x)\big)\Big]\)
     即"学生采样、teacher 在学生 rollout 上逐 token 给分布"的标准 OPD 形态(Agarwal 2024;teacher "观察并约束学生动作")。超参:**4.0 epochs,AdamW,lr 2e-6**。〔KL 方向论文只写 "optimize the KL Divergence loss",未显式标 forward/reverse;按 GKD 默认与"teacher 约束学生"语义,记为以 teacher 为参考分布的 token 级 KL,方向【待核】。〕
  3. **Mixed**:先 ASP-SFT 再在其上做 ASP-OPD 强化(§4.1)。
- **下游增强:Rejection Fine-Tuning(RFT)**【原文 B.2,Yuan 2023】:用**学生自生成** 10,000 条轨迹(HotpotQA,与蒸馏用**不同题**)、**拒绝采样**留高质量 agentic 行为再 SFT。超参:**2.0 epochs,AdamW,lr 5e-6,batch 4**。
- **诊断工具:confidence probe**【原文 §5、附录 F】:三层 MLP 接在 SLM **最后 4 层 hidden** 上,**backbone frozen**,预测"模型能否凭内部知识答对该 query"。实测 ECE ~0.043、Acc ~93.9%(校准良好)——用它证 SLM 高置信 query 远少于 teacher,佐证"内部知识不可靠 → 该总搜"。
- 数据流动小结:HotpotQA query → 注入 \(P_{\mathrm{force}}\) → (SFT:teacher 轨迹经双过滤后 CE | OPD:学生采 8 条 → teacher 给分布 → KL | Mixed:先 SFT 后 OPD)→ 可选 RFT(学生自生成 + 拒绝采样)→ 部署时按 Alg.2 强制总搜循环推理。
- 逐组件必要性:
  - **ASP 强制(SFT 过滤 + prompt)**:有效——三种 ASP 法都让 1.7B≈8B(Table 1),无 ASP 的普通蒸馏只 53.2。但**三者孰优无统一胜者**(SFT/OPD/Mixed 在不同 benchmark 互有高低),论文未给选择准则;没有"只用 prompt 不过滤" vs "只过滤不 prompt"的细粒度消融来拆开两个因素各自贡献。【推断】
  - **RFT**:作为下游增强,但 Table 1 主表似主要呈现 SFT/OPD/Mixed,RFT 的独立增益在正文未充分展开。【推断】
  - **搜索频率**(§4.2):vanilla 1.72 → ASP-SFT 2.47 → ASP-OPD 2.84(= Vanilla-8B 频率),OPD 搜索频率最高,是本文 OPD 有效性的具体实证点。
- 关键机制/直觉:ASP-OPD 的核心是**在学生自己 rollout 的 8 条轨迹上、用 teacher 的 token 概率分布做 KL 监督**——这正是 on-policy distillation 的标准形态(Agarwal 2024 + Thinking Machines 2025 blog),靠 prompt(\(P_{\mathrm{force}}\))而非数据过滤来引导"总搜"行为(因 on-policy 下学生本就在 \(P_{\mathrm{force}}\) 条件下采样)。confidence probe = 三层 MLP 接 SLM 最后 4 层 hidden、frozen backbone,预测"能否凭内部知识答对",ECE~0.043、Acc~93.9%。【原文】§4.1、附录 E/F
- 关键超参默认值(汇总,B.2)【原文】:ASP-SFT 18k 轨迹 / 3ep / lr 1e-5 / batch 4 / F1 阈 0.65;ASP-OPD 3k 题 ×8 rollout / 4 题 batch / 4ep / lr 2e-6;RFT 10k 自生成 / 2ep / lr 5e-6 / batch 4;teacher=Qwen3-32B;检索 e5-large-v2 + BM25;摘要 Qwen3-32B;语料 fullwiki-20210620。模型 Qwen3-0.6/1.7/4/8/32B、Llama-3.2-1B/3B。

## 靠不靠谱
- 实验与证据:
  - **数据集**:训练**仅用 HotpotQA 训练集**(teacher=Qwen3-32B)。评测:结构化多跳 HotpotQA/2Wiki/Bamboogle/MuSiQue + agentic 检索 BrowseComp-plus/Frames/LongSeAL,指标 String-F1(也报 EM)。检索 e5-large-v2 + fullwiki-20210620 语料(BrowseComp-Plus 用 Qwen3-Embedding-8B + 自带语料)。模型 Qwen3-0.6/1.7/4/8/32B、Llama-3.2-1B/3B。【原文】§4.1、附录 B
  - **关键数字**:摘要头条=ASP 比 LLM 蒸馏 **+17.3(Bamboogle)、+15.3(HotpotQA)**。Table 1:HotpotQA Mixed-1.7B=58.2≈8B(58.2);2Wiki OPD-1.7B=**62.9 > 8B(58.1)**;只在 HotpotQA 训练却泛化到 OOD(BrowseComp/Frames/LongSeAL)。噪声鲁棒(10% 检索失败):vanilla/普通蒸馏掉 12.1,ASP 仅掉 2.3/1.7。Adaptive Search Trap(Table 2):SLM 在 P=5% 自答就明显掉点,而 32B 到 P=10% 仍稳。【原文】Table 1/2、§4.2
  - **baseline 公平吗**:vanilla Qwen3 各尺寸 + 普通 Distilled 都在同检索器/语料下评,口径一致;但**所有训练只用 HotpotQA 单源**,既是泛化卖点也是天花板——OOD 增益的可靠性受单一训练分布限制。
  - **看着强但没回答核心问题?**:"1.7B≈8B"成立,但"总是搜"对本可内部答的简单 query 增加延迟(Table 5:SFT-1.7B ~3.1s vs vanilla 1.7B ~1.8s),论文承认却未量化"过度搜索"代价,也未给最优搜索预算。【推断】
- 假设与失效边界:
  - 【原文】Limitations 自述:ASP 隐含假设"检索信息总是准确可靠"——而真实搜索有噪声/误导内容(附录 H 只做了 10% 失败注入的初步鲁棒性);评测集中于 Qwen3 家族,跨架构未充分验证;训练实现"相对简单",更高级框架待探索。
  - 【推断】"知识置零、凡事都搜"在 query 本可由常识/内部知识秒答时是次优的;对需要内部推理(非检索)的任务,ASP 帮不上(论文也说 agentic search 的推理不像数学那样重)。
- 祛魅总结:真贡献=**一个清晰的诊断(SLM under-search 是主因,且自适应搜索在 SLM 上反而有害)+ 一个工程化 recipe(ASP 三变体)**,诊断有 confidence probe + 误差分析(33/66 幻觉、23/66 检索不足)支撑,较扎实。包装/局限:(1) 方法本身是**工程化 recipe 非新算法**(SFT/OPD/RFT 都是现成);(2) 三变体无统一胜者、缺选择准则,OPD 的优势主要在搜索频率与个别 benchmark,而非全面占优;(3) 单源(HotpotQA)单家族(Qwen3 为主)证据面偏窄。【推断】

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=ASP-SFT 学"始终搜索"的过滤后轨迹(序列级模仿);ASP-OPD 学 teacher 在学生 rollout 上的 token 分布(KL,on-policy)｜**改什么**=SLM 参数(SFT/OPD/RFT 全更新)｜**何时改**=离线串行(SFT→OPD→RFT,Mixed 是 SFT 后 OPD)｜**免梯度?**=否｜**记忆-技能生命周期**=无显式记忆/技能库,"总是搜索"的行为先验固化进参数；外部知识来自实时检索语料(非参数记忆)｜**防遗忘机制**=无专门设计;附录 I 反而显示 ASP 训练后通用推理(MMLU/GSM8K/GPQA)不降反升(1.7B MMLU 62.6→73.3),即未见明显遗忘。【原文】§4、附录 I
- ⑦ 开源代码+框架/harness:https://github.com/yizhou0409/Agentic-Rag(100% Python)。含 `build_corpus/`(Wikipedia 语料/E5 索引)、`e5_retriever.py`+`bm25_retriever.py`(双检索)、`main.py`(推理 pipeline)、`prompts/`。框架=检索 E5+BM25 / 推理 Search-o1 式迭代 agent / 训练 SFT+OPD+RFT 三阶段(trainer 细节在 appendix,仓库偏脚手架)。代码可得性 Tier B——语料/检索/推理可跑,SFT/OPD/RFT 训练脚本不完整,复现需补附录 B(已把 B.2 全超参抄录于"关键超参默认值")。【原文】GitHub + 既有 analysis 核查
- 💰 资源/成本与可扩展性:训练数据极省(SFT 18k、OPD 3k×8 rollout、RFT 10k,均来自 HotpotQA)。Table 5 端到端延迟(4×H20、vLLM、FAISS-GPU):ASP-SFT-1.7B ~3.1s vs Qwen3-32B ~10.3s,约 3× 加速达到可比性能。【原文】附录 B/G
- 🎯 对"探索-巩固"对标:**弱-中支撑,主要价值在"巩固有效行为先验"这一侧**。ASP 把"凡事先搜、别凭脑内知识猜"这个**有效行为(探索到的好习惯)固化进参数**——对应 idea 的"巩固=固化有效行为进参数";噪声鲁棒(检索失败时 ASP 模型能 re-plan 恢复,附录 H)轻度对应"path-recovery/回轨"。**竞品/差异**:本文是"一刀切强制搜索"的硬先验,**与 idea 强调的"on-policy 自选 / teacher 当稀疏脚手架"相反**——它恰恰**取消**了学生的自适应选择(Adaptive Search Trap 证明 SLM 自选不行)。**可借组件**:(a) ASP-OPD 的"学生 rollout 8 条 + teacher token 分布 KL"是干净的 OPD 实现样板(超参已抄录:3k 题 ×8 / lr 2e-6 / 4ep);(b) confidence probe(frozen backbone + 三层 MLP 接最后 4 层 hidden)可作"判断某步该不该探索/检索"的轻量门控原型。**缺口**:无 path-selection 的细粒度信用、无 MTP 前瞻、决策权交给硬先验而非学生。一句判定:**OPD 落地的实证样板 + "巩固有效行为"的弱对标,但其"取消自适应"的结论与 idea 的"学生自选"方向相左,作对照而非直接借鉴**。【推断,依据 §4/§5/附录 H vs idea】
- 🔭 开放问题/未来方向:【原文】把 ASP 集成进更高级训练框架;刻画 SLM agent 能力上界(受推理能力等多因素影响);开发对噪声/误导检索的鲁棒机制;跨更多 LM 架构验证(Limitations)。【推断】(1) 在"硬约束总搜"与"自适应搜"之间找中间地带——用更可靠的外部信号(而非 SLM 自身置信)做选择性检索,避开 Adaptive Search Trap;(2) 拆开 ASP-SFT 的"过滤"与"prompt"两因素做消融;(3) 给三变体一个可操作的选择准则。

RETURN:search_dont_guess | 读PDF? 是(13页全文+全附录含 Alg.2/B.2 全训练超参/confidence probe/prompt/误差分析/延迟) | 加厚? 是(方法到可复现:Alg.2 强制总搜循环逐步+ASP-SFT/OPD/RFT 三变体全超参+OPD KL 与 SFT CE 公式化+confidence probe 结构+数据流;相关工作四路线A-D精确短板+最近邻差异) | LaTeX公式条数 3 | 待核数 1(ASP-OPD 的 KL 方向 forward/reverse 原文未显式标注) | 残留待核(对标侧)0

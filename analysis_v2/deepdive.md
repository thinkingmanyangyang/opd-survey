deepdive | DeepDive: Advancing Deep Search Agents with Knowledge Graphs and Multi-Turn RL | 清华 / Z.AI / 东北大学(Rui Lu*, Zhenyu Hou*, Zihan Wang* 等;Jie Tang、Yuxiao Dong) | arXiv 2509.10446(v2 2025-10-14)·under review | 主题线 L4(Agent/工具/多轮自进化)·相关性 中

**原始论文**:https://arxiv.org/abs/2509.10446

## 一眼看懂

> 一句话导读:让开源大模型学会"在几百个网页里把藏得很深的信息挖出来"。难点是没有足够难的训练题、也没有靠谱的多轮 RL;本文用知识图谱自动造难题 + 多轮 GRPO 训练来补,造出的题甚至被自家旗舰 GLM-4.5/4.6 拿去用了。

- 🟦 TL;DR:开源 LLM 当 deep-search agent(即在数百个网页里找"难找信息"的浏览智能体)表现很差,原因是两条——缺够难的数据,且缺多轮 RL。DeepDive 用两招应对:
  - ① 从知识图谱(KG)里做随机游走,抽出长的多跳路径,再把属性模糊化,自动合成"hard-to-find"的 QA;
  - ② 端到端做多轮 GRPO 训练浏览 agent,并加一个"冗余惩罚"(用 Jaccard 相似度惩罚重复查询),促使搜索更多样、更高效。
  - 效果:DeepDive-32B 在 BrowseComp 上用 KG 数据达到 15.3%,再加上半自动的 i.i.d. 数据可达 22.2%;它造的数据还被 GLM-4.5/4.6 采用【原文 Abstract, §1, Table 1】。
- 最巧的一步:**KG 难数据合成**。它之所以是核心,是因为 KG 同时满足了三个性质:可验证(实体-关系三元组可追溯)、可控多跳深度(由随机游走步数 \(k\) 决定)、可控模糊度(选择性遮蔽属性来造 blurry entity,即"模糊实体")。三者齐备才能造出"逼着模型反复搜索-验证、而不是走捷径"的难题。
  - 抽掉它(改用 HotpotQA 这类数据),整套就退化到现有 deep-search agent 的水平——Table 2 消融显示,用 HotpotQA 轨迹做 SFT/RL 只有微弱增益,换成合成数据才在所有 benchmark 上大涨【原文 §2.1, §3.4 Table 2】。

## 为什么做

> 一句话导读:浏览 agent 是趋势,但开源模型被专有系统甩开一大截。作者把差距归到两件事——训练题太简单、且没有成熟的多轮 RL,于是分别用"KG 造难题"和"端到端多轮 GRPO"来补。

- 研究背景:给 LLM 加上浏览工具,它就能成为 deep search agent——在数百个在线来源上反复推理、检索,以定位复杂难找的信息(典型如 BrowseComp 里的题)。但开源模型在这件事上远远落后于 OpenAI DeepResearch 等专有系统【原文 §1】。
- 解决的具体痛点:作者把差距归因于两点。
  - ① **数据太简单**:HotpotQA / 2Wiki / Bamboogle / Musique 这些题,搜几个清晰实体就能解,根本不反映"hard-to-find";而 BrowseComp 含多个模糊实体(blurry entity),需要长 horizon 推理 + deep search。
  - ② **多轮 RL 训练缺失**:如何把长 horizon 推理和 deep search 的工具使用有效结合,仍是开放问题;即便 DeepSeek-R1 也只做浅层的工具调用,而且容易幻觉【原文 §1】。
- 相关工作 & 各自不足(并行路线,各自的精确差异):
  - **路线 ①——开源 search-RL agent(R1-Searcher / ReSearch / DeepResearcher / Search-o1)**:主要在 HotpotQA 这类多跳 QA 上训练和评测。**精确短板**:任务太简单(几个清晰实体直接搜就有),学不到"在数百个模糊实体里做长 horizon 深搜"。DeepDive 的差异在于:用 KG 随机游走 + 属性模糊来**系统性地造 blurry-entity 难题**,而不是靠人工标注或简单多跳。
  - **路线 ②——专有 DeepResearch(OpenAI / Grok-DeepResearch)**:遥遥领先(BrowseComp 51.5%),但闭源、配方拿不到。DeepDive 把自己定位为"开源里有竞争力的结果"(尽管绝对值仍低)。
  - **路线 ③——通用浏览 agent / WebSailor / WebDancer 等**:多数是为直接搜索任务设计的。DeepDive 的差异在于用 **redundancy penalty 显式抑制重复查询**(一个 Jaccard 正则),促使搜索更多样。
  - **算法底座**:多轮 RL 用的是标准 **GRPO(Shao 2024,DeepSeekMath)**。DeepDive 不改 RL 算法本身,只在 reward 上加了冗余惩罚和严格的格式约束。
- 动机链:现状是 deep search agent 已成趋势、但开源落后;缺陷在于难数据稀缺、且缺多轮 RL;所以本文的做法是用 KG 自动合成难数据 + 端到端多轮 GRPO + 冗余惩罚促多样【原文 §1】。
- 与最近邻工作的Δ:相对 R1-Searcher / WebSailor 等,差在两点——用 KG 随机游走 + 属性模糊系统性地造 blurry-entity 难题,以及用 redundancy penalty 显式抑制重复查询。有用点:KG 数据可验证、难度可控、可规模化,而且直接被 GLM-4.5/4.6 采用来提升它们的 BrowseComp 表现。

## 怎么做 + 靠不靠谱

> 一句话导读:四块——A 用知识图谱造难题(随机走出长路径、把信息模糊化、再用强模型筛掉太容易的),B 是 agent 的"想-动-看"多轮循环,C 是多轮 GRPO 加冗余惩罚和严格格式奖励,D/E 是消融与超参。

### A. KG 难数据合成(读完可复现,§2.1,Eq.2-5)
先把 KG 形式化为一个有向图 \(\mathcal G=(V,E)\),每个实体 \(v_i\in V\) 带一组属性 \(A(v)=\{a_i^0,a_i^1,\dots,a_i^t\}\)。流水线如下:
1. **随机游走抽长路径**:从 \(v_0\) 出发走 \(k\) 步,得到路径 \(P=[v_0,v_1,\dots,v_k]\),每一步 \((v_i,v_{i+1})\in E\)。取较长的路径(\(k>5\),主实验取 \(k\in[5,9]\))来提升推理复杂度。
2. **属性富化 + 模糊化**(目的是造出 blurry entity):先把每个节点和它的属性拼成一条属性丰富的路径
\(\displaystyle P_A=\big[v_0,(a_0^0,a_0^1,\dots),\;v_1,(a_1^0,a_1^1,\dots),\;\dots,\;v_k,(a_k^0,a_k^1,\dots)\big] \tag{2}\)
取终点 \(v_k\) 的某个属性 \(a_k^i\) 当 ground-truth 答案;再让 LLM 沿整条路径把信息**模糊化**(例如把具体日期泛化成一个区间):
\(\displaystyle (q,a_k^i)=\text{LLM-obscure}(P_A) \tag{3}\)
3. **路径质量两约束**:
   - (i)按出度区间过滤候选的下一节点——出度太高意味着太流行、答案容易被猜到,太低则难以继续扩展路径:
\(\displaystyle \mathcal N(v_i)=\{u\mid (v_i,u)\in E\;\wedge\;d_{\min}\le d(u)\le d_{\max}\}\quad(d_{\min}=4,\,d_{\max}=8) \tag{4}\)
   - (ii)再让 LLM 在候选里挑出**逻辑最连贯**的下一节点(\(d=3\) 是每步候选数的设定):
\(\displaystyle v_{i+1}=\text{LLM-select}(P_i,\mathcal N(v_i)) \tag{5}\)
4. **难度过滤**:用 frontier 模型(GPT-4o + 基础搜索)对每道题试 4 次,**只要有一次答对就丢弃**,只保留 4 次全错的题 → 最终得到 **3,250 QA**,切成 **1,016 SFT + 2,234 RL**。

### B. agent 交互范式(ReAct 式多轮循环,§2)
agent 反复执行 reason → tool execution → observe 这个循环:在第 \(t\) 步先输出推理 \(c_t\),再输出动作 \(a_t\)(即工具调用),然后观察返回的网页内容 \(o_t\);如此循环,直到它判定信息已经足够、给出最终答案为止。一条完整轨迹记作 \(T=[q,(c_1,a_1,o_1),\dots,c_{\text{ans}},a_{\text{eos}}]\)。工具用的是 Serper API(search,返回 top-10 页)和 Jina API(click/open)。

### C. 端到端多轮 GRPO + 复合奖励(§2.2,Eq.6-9)
> 一句话导读:奖励由三块拼成——只看最终答案对不对(严格二值)、惩罚查询互相重复(冗余惩罚)、还要求每一步格式都合规。
- **多轮 GRPO 目标(Eq.6)**:对每道题,从当前策略 \(\pi_\theta\) 采 \(G\) 条**轨迹**(注意是整条轨迹,不是单个响应),按最终 reward 在组内归一得到优势 \(A_i=\big(r_i-\mathrm{mean}\{r_k\}_{k=1}^G\big)/\mathrm{std}\{r_k\}_{k=1}^G\),再最大化下面的 clip 代理目标:
\(\displaystyle \mathcal L(\theta)=\frac1G\sum_{i=1}^{G}\Big[\min\big(\rho_i A_i,\;\mathrm{clip}(\rho_i,1-\epsilon,1+\epsilon)A_i\big)-\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})\Big],\quad \rho_i=\frac{\pi_\theta(T)}{\pi_{\theta_{\mathrm{old}}}(T)} \tag{6}\)
注意:这里的重要性比 \(\rho_i\) 是**整条轨迹级**的 \(\pi_\theta(T)/\pi_{\theta_{\mathrm{old}}}(T)\);实际训练时设 \(\beta=0\)(即去掉 KL 以鼓励探索,详见超参)。
- **冗余惩罚(Eq.7,目的是促搜索多样)**:一条轨迹内的查询集记作 \(Q=[q_1,\dots,q_T]\),每个查询是一组关键词 \(q_i=\{w_{i,1},\dots,w_{i,n_i}\}\)。两个查询之间的 Jaccard 相似度定义为 \(\mathrm{sim}(q_i,q_j)=|q_i\cap q_j|/|q_i\cup q_j|\)(交集占并集的比例);整条轨迹的相似度则是两两平均:
\(\displaystyle S(T)=\frac{1}{T(T-1)}\sum_{i\ne j}\mathrm{sim}(q_i,q_j)\;\in[0,1] \tag{7}\)
\(S=1\) 表示所有查询完全相同,\(S=0\) 表示完全不重叠;\(S\) 越低,说明搜索越多样。
- **严格二值奖励(Eq.8,全有或全无)**:只有当**每一步格式都正确**(reason \(c_i\)、action \(a_i\) 都合规)**且**最终答案经 LLM judge 判对,才给 +1:
\(\displaystyle r(T)=\begin{cases}1,&\big(\forall i,\;\text{Format}(c_i,a_i)\big)\wedge \text{Judge}(a_{\text{eos}},a^*)\\0,&\text{otherwise}\end{cases} \tag{8}\)
- **复合奖励(Eq.9)**:把冗余惩罚叠加:
\(\displaystyle r'(T)=r(T)-\lambda\cdot S(T),\qquad \lambda=0.1 \tag{9}\)

### D. 逐组件必要性(消融)
- **KG 难数据合成**:有消融(Table 2)。HotpotQA 轨迹做 SFT/RL 只有微弱增益,换成合成数据则在四个 benchmark 上大涨(尤其 BrowseComp-266)。没它,agent 学不到长 horizon 深搜【原文 §3.4 Table 2】。
- **多轮 GRPO**:有对照(Fig.1 右)。SFT-only 的 32B 在 BrowseComp 上是 9.5%,RL 之后涨到 15.3%;四个 benchmark 一致提升(+5.8/+6.7/+1.6/+3.3%)。注意 9B 模型增益很小(SFT-only 5.6% → RL 6.3%),作者归因于它推理容量有限、容易过拟合到合成数据【原文 §3.2, Table 1】。
- **Redundancy Penalty**:有消融(Fig.7b/c)。后期精度提升约 20%、工具调用次数减少约 14%。没它,会有重复查询、白白浪费、搜索效率低【原文 §3.4】。
- **Strict Format Reward**:有消融(Fig.7a)。去掉它,精度停在约 8.0 几乎不涨;加上它,精度稳定上升、约高 2 个绝对点。没它,学习不稳【原文 §3.4】。

### E. 训练超参与流程(复现锚点)【原文 §3.1】
- 基模型 GLM-Z1-9B-0414 与 QwQ-32B。
- **SFT(cold-start)**:Claude-4-Sonnet-Thinking 拒绝采样得 858 高质量轨迹;3 epochs、global batch 32、lr 1e-5、max ctx 104,800。
- **RL**:**slime 框架**、全部 2,234 样本、rollout size 8、每 prompt 16 samples、global batch 128、temperature 1.0、max ctx 51,200、\(\lambda=0.1\)、**KL \(\beta=0\)(显式去 KL 促探索)**、lr 1e-6。
- 评测:BrowseComp/BrowseComp-ZH/Xbench-DeepSearch/SEAL-0,LLM-as-Judge(Llama-3.1-70B);训练期 checkpoint 选择用 BrowseComp-266 子集(turn ≤75),饱和后评全集 1,266(turn ≤128),每集评两次取均值。
- 结果(Table 1):DeepDive-32B BrowseComp 15.3%(开源最佳,超 WebSailor-32B 10.5%、Search-o1-32B 2.8%),仅次于 OpenAI DeepResearch(51.5%)。

### 靠不靠谱
> 一句话导读:KG 造难题这条 pipeline 是真功劳(被 GLM-4.5/4.6 实际采用就是最硬的背书);但要清醒——BrowseComp 绝对值还很低、离专有系统差得远,而且涨分主要来自数据,不是 RL 算法本身的创新。
- baseline 公平性:对 GLM-Z1-9B / QwQ-32B 也开了 function calling 做公平对照,这点合理。但有个"看着强、其实没真正解决问题"的点:**BrowseComp 的绝对值仍然很低(15.3% / 22.2%),距离专有 DeepResearch 差距巨大,任务远未解决**;另外训练期用 266 子集来选 checkpoint,存在调参-评测耦合的风险【推断,据 §3.1 用 266 选 checkpoint 再评全集】。
- 假设与失效边界:
  - 【原文】KG 数据合成依赖高质量 KG(KILT / AMiner)的可验证三元组;难度过滤靠 frontier 模型(GPT-4o)4 次答不对;评测靠 LLM judge。
  - 【推断】合成数据的难度分布对结果的影响没有被隔离;RL 数据规模极小(2,234),其多样性是否足以支撑泛化未充分检验;9B 上增益小说明方法对模型容量敏感(小模型可能过拟合到合成数据)。
- 祛魅总结:
  - 真贡献:一条基于 KG、可验证、难度可控的难数据合成 pipeline(被 GLM-4.5/4.6 实际采用,这是最硬的外部背书);冗余惩罚这个简单有效的搜索多样性正则;以及系统的 test-time scaling 分析(工具调用越多越准,且"用最少工具调用的那条"来选答案,优于多数投票)。
  - 包装 / 高估处:"new open-source competitive result"成立,但绝对值低(15.3%);标题强调 KG + 多轮 RL,但实际增益主要来自数据(Table 2),而非 RL 算法创新(用的就是标准多轮 GRPO);i.i.d. 数据达 22.2% 靠的是半自动人工标注(o3 辅助),并非全自动【推断,据 §4 semi-automated】。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=最终答案正确性的 strict binary reward(LLM judge)+ 冗余惩罚(Jaccard)| **改什么**=策略参数(多轮 on-policy 梯度更新)| **何时改**=RL 训练期(cold-start SFT 后)| **免梯度?**=否(GRPO 梯度训练;但 test-time scaling 部分是免训练的推理期增益)| **记忆-技能生命周期**=不涉及持久记忆/技能库;agent 在单轨迹内通过 ReAct 循环积累上下文,任务结束即丢弃,能力固化进权重 | **防遗忘机制**=KL β=0(主动允许偏离 ref),无专门防遗忘【原文 §2.2, §3.1】。
- ⑦ 开源代码+框架/harness:https://github.com/THUDM/DeepDive(含 qa_synthetic/ KG 数据合成、assets;**仅含 KG 数据合成代码,不含 RL 训练代码**——训练在外部 slime 完成);RL 框架 **slime**(THUDM/slime,"LLM post-training framework for RL scaling",论文 §3.1 明确"using the open-source Slime framework");算法多轮 **GRPO**。本地已 clone(~2.9MB)。数据/模型公开(3,250 KG QA + 2,234 RL + 半自动 i.i.d.;DeepDive 模型)。【未完整复现需自行接入 slime】
- 💰 资源/成本与可扩展性:论文未给 GPU 卡时/训练成本(原文未说明);数据合成成本:KG 随机游走自动化(便宜),但难度过滤需 frontier 模型多次调用、实体模糊用 Gemini-2.5-Pro,i.i.d. 数据需 o3 辅助人工标注(较贵);推理期可 test-time scaling(增大工具调用预算/并行采样)换性能【原文 §3.3, §4】。
- 🎯 对"探索-巩固"对标:**弱相关(纯多轮工具 RL,作长 horizon agent RL 对照)** —— 一句判定:DeepDive 是"如何让 agent 在真实 web 环境多轮自主探索(reason-search-observe)并用稀疏最终奖励学会深搜"的范例,与"探索=发现有效行为/路径"在 agent 层面同构,但**不涉及巩固/回轨,也无 teacher 脚手架、无记忆/技能库**。依据:纯 on-policy 多轮 GRPO,奖励只看最终答案对错,无错误前缀注入、无路径恢复目标。可借组件:① redundancy penalty(Jaccard 抑制重复,Eq.7)可迁移为"路径多样性正则"防止巩固阶段塌缩到单一路径;②"最少工具调用选答案优于多数投票"(Fig.6c,置信→早停)这一推理期信号对"何时该停止探索/巩固"有启发。缺口:与本项目核心(MTP 前瞻、on-policy 自选恢复、防遗忘的记忆/技能固化)无直接交集,定位为外围 agent-RL 对照。
- 🔭 开放问题/未来方向:【原文】§4 把 function calling 扩展、多语言(中文网站)deep search 列为扩展方向;test-time scaling(工具调用预算、并行采样答案选择)值得继续挖。【推断】合成数据难度分布对泛化的影响隔离、把方法做到小模型(9B 增益小)、从外部 slime 解耦出可复现 RL 训练码、把多轮 agent 与持久记忆/技能库结合均未解。

---
RETURN: deepdive | 读到PDF? 是(16页全文,Eq.2-9 从 PDF 精确抄录) | L4(Agent/工具/多轮RL) | 弱相关(长horizon工具RL对照,无巩固/回轨/teacher脚手架);redundancy penalty 可借作路径多样性正则 | 残留待核 0(RL 期 KL β=0 显式去 KL;k∈[5,9]/d=3/dmin=4/dmax=8 是主实验参数)

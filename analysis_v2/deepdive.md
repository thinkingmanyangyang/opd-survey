deepdive | DeepDive: Advancing Deep Search Agents with Knowledge Graphs and Multi-Turn RL | 清华 / Z.AI / 东北大学(Rui Lu*, Zhenyu Hou*, Zihan Wang* 等;Jie Tang、Yuxiao Dong) | arXiv 2509.10446(v2 2025-10-14)·under review | 主题线 L4(Agent/工具/多轮自进化)·相关性 中

**原始论文**:https://arxiv.org/abs/2509.10446

## 一眼看懂
- 🟦 TL;DR:开源 LLM 当 deep-search agent(在数百网页里找"难找信息")表现差,原因是缺难数据 + 缺多轮 RL。DeepDive 两招:①从知识图谱(KG)随机游走抽长多跳路径 + 模糊属性,自动合成"hard-to-find"QA;②端到端多轮 GRPO 训练浏览 agent,加一个"冗余惩罚"(Jaccard 相似度惩罚重复查询)促多样高效搜索。DeepDive-32B 在 BrowseComp 达 15.3%(KG 数据),加半自动 i.i.d. 数据可达 22.2%;其数据被 GLM-4.5/4.6 采用【原文 Abstract, §1, Table 1】。
- 最巧的一步:**KG 难数据合成**。抽掉它(改用 HotpotQA 类数据),整套就退化为现有 deep-search agent 的水平——Table 2 消融显示用 HotpotQA 轨迹 SFT/RL 只有微弱增益,换成合成数据才在所有 benchmark 大涨。为什么:KG 的可验证性(实体-关系三元组可追溯)+ 可控多跳深度(随机游走步数 \(k\))+ 可控模糊度(选择性遮蔽属性造 blurry entity)三性质同时满足,才能造出"逼模型迭代搜索-验证而非走捷径"的难题【原文 §2.1, §3.4 Table 2】。

## 为什么做
- 研究背景:给 LLM 加浏览工具可成为 deep search agent,对数百在线来源推理检索以定位复杂难找信息(如 BrowseComp 题);但开源模型远落后于 OpenAI DeepResearch 等专有系统【原文 §1】。
- 解决的具体痛点:作者把差距归因于两点——①数据太简单:HotpotQA/2Wiki/Bamboogle/Musique 搜几个清晰实体即可解,不反映"hard-to-find";BrowseComp 含多个模糊实体(blurry entity),需长 horizon 推理 + deep search;②多轮 RL 训练缺失:如何把长 horizon 推理与 deep search 工具使用有效结合仍是开放问题,即便 DeepSeek-R1 也只做浅层工具调用且易幻觉【原文 §1】。
- 相关工作 & 各自不足(并行路线 + 精确差异):
  - **路线 ①——开源 search-RL agent(R1-Searcher / ReSearch / DeepResearcher / Search-o1)**:主要在 HotpotQA 类多跳 QA 上训练评测。**精确短板**:任务太简单(几个清晰实体直接搜即得),学不到"在数百模糊实体里长 horizon 深搜"。DeepDive 的差异:用 KG 随机游走 + 属性模糊**系统性造 blurry-entity 难题**(而非靠人标或简单多跳)。
  - **路线 ②——专有 DeepResearch(OpenAI / Grok-DeepResearch)**:遥遥领先(BrowseComp 51.5%)但闭源,配方不可得。DeepDive 定位为"开源可竞争结果"(虽绝对值仍低)。
  - **路线 ③——通用浏览 agent / WebSailor / WebDancer 等**:多为直接搜索任务设计。DeepDive 差异:**redundancy penalty 显式抑制重复查询**(Jaccard 正则),促搜索多样。
  - **算法底座**:多轮 RL 用标准 **GRPO(Shao 2024,DeepSeekMath)**,DeepDive 不改 RL 算法本身,只在 reward 上加冗余惩罚+严格格式。
- 动机链:现状(deep search agent 是趋势但开源落后)→缺陷(难数据稀缺 + 缺多轮 RL)→所以必须(KG 自动合成难数据 + 端到端多轮 GRPO + 冗余惩罚促多样)【原文 §1】。
- 与最近邻工作的Δ:相对 R1-Searcher/WebSailor 等,差在"用 KG 随机游走+属性模糊系统性造 blurry-entity 难题"+"redundancy penalty 显式抑制重复查询"。有用点:KG 数据可验证、可控难度、可规模化,直接被 GLM-4.5/4.6 采用以提升其 BrowseComp 表现。

## 怎么做 + 靠不靠谱

### A. KG 难数据合成(读完可复现,§2.1,Eq.2-5)
KG 形式化为有向图 \(\mathcal G=(V,E)\),实体 \(v_i\in V\) 带属性集 \(A(v)=\{a_i^0,a_i^1,\dots,a_i^t\}\)。流水线:
1. **随机游走抽长路径**:从 \(v_0\) 走 \(k\) 步得 \(P=[v_0,v_1,\dots,v_k]\),每步 \((v_i,v_{i+1})\in E\);取长路径(\(k>5\),主实验 \(k\in[5,9]\))提升推理复杂度。
2. **属性富化 + 模糊化**(造 blurry entity):把每节点与其属性拼成属性丰富路径
\(\displaystyle P_A=\big[v_0,(a_0^0,a_0^1,\dots),\;v_1,(a_1^0,a_1^1,\dots),\;\dots,\;v_k,(a_k^0,a_k^1,\dots)\big] \tag{2}\)
取终点 \(v_k\) 的某属性 \(a_k^i\) 作 ground-truth 答案,再用 LLM 沿整条路径**模糊化**信息(如把具体日期泛化成区间):
\(\displaystyle (q,a_k^i)=\text{LLM-obscure}(P_A) \tag{3}\)
3. **路径质量两约束**:(i)按出度区间过滤候选下一节点——出度过高=太流行答案可预测,过低=难扩路径:
\(\displaystyle \mathcal N(v_i)=\{u\mid (v_i,u)\in E\;\wedge\;d_{\min}\le d(u)\le d_{\max}\}\quad(d_{\min}=4,\,d_{\max}=8) \tag{4}\)
(ii)LLM 在候选里选**逻辑最连贯**的下一节点(\(d=3\) 为每步候选数设定):
\(\displaystyle v_{i+1}=\text{LLM-select}(P_i,\mathcal N(v_i)) \tag{5}\)
4. **难度过滤**:frontier 模型(GPT-4o + 基础搜索)对每题试 4 次,**任一次答对即丢弃**,只留 4 次全错的题 → 得 **3,250 QA**,切 **1,016 SFT + 2,234 RL**。

### B. agent 交互范式(ReAct 式多轮循环,§2)
agent 迭代执行 reason → tool execution → observe:在第 \(t\) 步先输出推理 \(c_t\)、再输出动作 \(a_t\)(工具调用)、然后观察网页内容 \(o_t\),循环直到判定信息足够给出最终答案。一条轨迹 \(T=[q,(c_1,a_1,o_1),\dots,c_{\text{ans}},a_{\text{eos}}]\)。工具:Serper API(search 返回 top-10 页)、Jina API(click/open)。

### C. 端到端多轮 GRPO + 复合奖励(§2.2,Eq.6-9)
- **多轮 GRPO 目标(Eq.6)**:对每题从当前策略 \(\pi_\theta\) 采 \(G\) 条**轨迹**(非单响应),按最终 reward 组内归一得优势 \(A_i=\big(r_i-\mathrm{mean}\{r_k\}_{k=1}^G\big)/\mathrm{std}\{r_k\}_{k=1}^G\),最大化 clip 代理目标:
\(\displaystyle \mathcal L(\theta)=\frac1G\sum_{i=1}^{G}\Big[\min\big(\rho_i A_i,\;\mathrm{clip}(\rho_i,1-\epsilon,1+\epsilon)A_i\big)-\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})\Big],\quad \rho_i=\frac{\pi_\theta(T)}{\pi_{\theta_{\mathrm{old}}}(T)} \tag{6}\)
注意重要性比 \(\rho_i\) 是**整条轨迹级** \(\pi_\theta(T)/\pi_{\theta_{\mathrm{old}}}(T)\);实际训练设 \(\beta=0\)(去 KL 促探索,见超参)。
- **冗余惩罚(Eq.7,促搜索多样)**:轨迹内查询集 \(Q=[q_1,\dots,q_T]\),每查询是关键词集 \(q_i=\{w_{i,1},\dots,w_{i,n_i}\}\);两查询 Jaccard 相似度 \(\mathrm{sim}(q_i,q_j)=|q_i\cap q_j|/|q_i\cup q_j|\);轨迹整体相似度:
\(\displaystyle S(T)=\frac{1}{T(T-1)}\sum_{i\ne j}\mathrm{sim}(q_i,q_j)\;\in[0,1] \tag{7}\)
\(S=1\) 时所有查询相同,\(S=0\) 时完全不重叠;越低=搜索越多样。
- **严格二值奖励(Eq.8,全有或全无)**:仅当**每一步格式都对**(reason \(c_i\)、action \(a_i\))**且**最终答案经 LLM judge 判对才给 +1:
\(\displaystyle r(T)=\begin{cases}1,&\big(\forall i,\;\text{Format}(c_i,a_i)\big)\wedge \text{Judge}(a_{\text{eos}},a^*)\\0,&\text{otherwise}\end{cases} \tag{8}\)
- **复合奖励(Eq.9)**:把冗余惩罚叠加:
\(\displaystyle r'(T)=r(T)-\lambda\cdot S(T),\qquad \lambda=0.1 \tag{9}\)

### D. 逐组件必要性(消融)
- **KG 难数据合成**:有消融(Table 2)。HotpotQA 轨迹 SFT/RL 仅微弱增益,合成数据在四 benchmark 大涨(尤其 BrowseComp-266)。没它→agent 学不到长 horizon 深搜【原文 §3.4 Table 2】。
- **多轮 GRPO**:有对照(Fig.1 右)。SFT-only 32B 在 BrowseComp 9.5%,RL 后 15.3%;四 benchmark 一致提升(+5.8/+6.7/+1.6/+3.3%)。注:9B 模型增益小(SFT-only 5.6%→RL 6.3%),作者归因其推理容量有限/易过拟合合成数据【原文 §3.2, Table 1】。
- **Redundancy Penalty**:有消融(Fig.7b/c)。后期精度提升约 20%、工具调用减约 14%。没它→重复查询浪费、搜索效率低【原文 §3.4】。
- **Strict Format Reward**:有消融(Fig.7a)。去掉它精度停在约 8.0 几乎不涨,加上稳定上升约高 2 个绝对点。没它→学习不稳【原文 §3.4】。

### E. 训练超参与流程(复现锚点)【原文 §3.1】
- 基模型 GLM-Z1-9B-0414 与 QwQ-32B。
- **SFT(cold-start)**:Claude-4-Sonnet-Thinking 拒绝采样得 858 高质量轨迹;3 epochs、global batch 32、lr 1e-5、max ctx 104,800。
- **RL**:**slime 框架**、全部 2,234 样本、rollout size 8、每 prompt 16 samples、global batch 128、temperature 1.0、max ctx 51,200、\(\lambda=0.1\)、**KL \(\beta=0\)(显式去 KL 促探索)**、lr 1e-6。
- 评测:BrowseComp/BrowseComp-ZH/Xbench-DeepSearch/SEAL-0,LLM-as-Judge(Llama-3.1-70B);训练期 checkpoint 选择用 BrowseComp-266 子集(turn ≤75),饱和后评全集 1,266(turn ≤128),每集评两次取均值。
- 结果(Table 1):DeepDive-32B BrowseComp 15.3%(开源最佳,超 WebSailor-32B 10.5%、Search-o1-32B 2.8%),仅次于 OpenAI DeepResearch(51.5%)。

### 靠不靠谱
- baseline 公平性:对 GLM-Z1-9B/QwQ-32B 也开了 function calling 做公平对照,合理;但"看着强但没回答"的点:**BrowseComp 绝对值仍很低(15.3%/22.2%),距专有 DeepResearch 巨大差距,任务远未解决**;训练期 checkpoint 选择用 266 子集存在调参-评测耦合风险【推断,据 §3.1 用 266 选 checkpoint 再评全集】。
- 假设与失效边界:【原文】KG 数据合成依赖高质量 KG(KILT/AMiner)的可验证三元组;难度过滤靠 frontier 模型(GPT-4o)4 次答不对;评测靠 LLM judge。【推断】合成难度分布对结果影响未隔离;RL 数据规模极小(2,234),多样性是否够支撑泛化未充分检验;9B 上增益小说明方法对模型容量敏感(小模型可能过拟合合成数据)。
- 祛魅总结:真贡献=KG-based 可验证可控难数据合成 pipeline(被 GLM-4.5/4.6 实际采用,这是最硬的外部背书)+ 冗余惩罚这一简单有效的搜索多样性正则 + 系统的 test-time scaling 分析(工具调用越多越准、"最少工具调用"选答案优于多数投票)。包装/高估处:"new open-source competitive result"成立但绝对值低(15.3%),标题强调 KG+多轮 RL,实际增益主要来自数据(Table 2)而非 RL 算法创新(就是标准多轮 GRPO);i.i.d. 数据达 22.2% 靠半自动人工标注(o3 辅助),并非全自动【推断,据 §4 semi-automated】。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=最终答案正确性的 strict binary reward(LLM judge)+ 冗余惩罚(Jaccard)| **改什么**=策略参数(多轮 on-policy 梯度更新)| **何时改**=RL 训练期(cold-start SFT 后)| **免梯度?**=否(GRPO 梯度训练;但 test-time scaling 部分是免训练的推理期增益)| **记忆-技能生命周期**=不涉及持久记忆/技能库;agent 在单轨迹内通过 ReAct 循环积累上下文,任务结束即丢弃,能力固化进权重 | **防遗忘机制**=KL β=0(主动允许偏离 ref),无专门防遗忘【原文 §2.2, §3.1】。
- ⑦ 开源代码+框架/harness:https://github.com/THUDM/DeepDive(含 qa_synthetic/ KG 数据合成、assets;**仅含 KG 数据合成代码,不含 RL 训练代码**——训练在外部 slime 完成);RL 框架 **slime**(THUDM/slime,"LLM post-training framework for RL scaling",论文 §3.1 明确"using the open-source Slime framework");算法多轮 **GRPO**。本地已 clone(~2.9MB)。数据/模型公开(3,250 KG QA + 2,234 RL + 半自动 i.i.d.;DeepDive 模型)。【未完整复现需自行接入 slime】
- 💰 资源/成本与可扩展性:论文未给 GPU 卡时/训练成本(原文未说明);数据合成成本:KG 随机游走自动化(便宜),但难度过滤需 frontier 模型多次调用、实体模糊用 Gemini-2.5-Pro,i.i.d. 数据需 o3 辅助人工标注(较贵);推理期可 test-time scaling(增大工具调用预算/并行采样)换性能【原文 §3.3, §4】。
- 🎯 对"探索-巩固"对标:**弱相关(纯多轮工具 RL,作长 horizon agent RL 对照)** —— 一句判定:DeepDive 是"如何让 agent 在真实 web 环境多轮自主探索(reason-search-observe)并用稀疏最终奖励学会深搜"的范例,与"探索=发现有效行为/路径"在 agent 层面同构,但**不涉及巩固/回轨,也无 teacher 脚手架、无记忆/技能库**。依据:纯 on-policy 多轮 GRPO,奖励只看最终答案对错,无错误前缀注入、无路径恢复目标。可借组件:① redundancy penalty(Jaccard 抑制重复,Eq.7)可迁移为"路径多样性正则"防止巩固阶段塌缩到单一路径;②"最少工具调用选答案优于多数投票"(Fig.6c,置信→早停)这一推理期信号对"何时该停止探索/巩固"有启发。缺口:与本项目核心(MTP 前瞻、on-policy 自选恢复、防遗忘的记忆/技能固化)无直接交集,定位为外围 agent-RL 对照。
- 🔭 开放问题/未来方向:【原文】§4 把 function calling 扩展、多语言(中文网站)deep search 列为扩展方向;test-time scaling(工具调用预算、并行采样答案选择)值得继续挖。【推断】合成数据难度分布对泛化的影响隔离、把方法做到小模型(9B 增益小)、从外部 slime 解耦出可复现 RL 训练码、把多轮 agent 与持久记忆/技能库结合均未解。

---
RETURN: deepdive | 读到PDF? 是(16页全文,Eq.2-9 从 PDF 精确抄录) | L4(Agent/工具/多轮RL) | 弱相关(长horizon工具RL对照,无巩固/回轨/teacher脚手架);redundancy penalty 可借作路径多样性正则 | 残留待核 0(RL 期 KL β=0 显式去 KL;k∈[5,9]/d=3/dmin=4/dmax=8 是主实验参数)

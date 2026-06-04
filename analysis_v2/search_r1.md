search_r1 | Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning | UIUC(Bowen Jin、Jiawei Han 等)+ UMass Amherst(Hansi Zeng、Hamed Zamani)+ Google Cloud AI Research(Jinsung Yoon、Sercan Arık) | 2025-03 首发 arXiv 2503.09516(v5 2025-08-05)·COLM 2025 | 主题线 L4(Agent/工具 RL)+L3(RLVR)·相关性 高

**原始论文**:https://arxiv.org/abs/2503.09516

## 一眼看懂
- 🟦 TL;DR:把"调用搜索引擎"嵌进 RL 训练循环——让 LLM 自己在 `<think>` 推理、需要时发 `<search>` query、检索结果包进 `<information>` 拼回上下文继续推理、最后 `<answer>` 给答案;训练时**对检索回来的 token 做 loss masking**(只对 LLM 自己生成的 token 算策略梯度),奖励只用最简的 **EM 结果奖励**(无过程奖励、无 format 奖励)。这样 LLM 在纯结果奖励下自发学会"何时检索、检索什么、如何结合检索继续推理",是 DeepSeek-R1-Zero 从"纯参数推理"到"搜索增强推理"的扩展。
- 最巧的一步:**检索 token 的 loss masking**。Table 4 直接证明:去掉 masking,7B Avg 从 **0.431 掉到 0.343**(掉 0.088,约 −20%)。原因(直觉):检索回来的 token 是"环境观测"不是"策略输出",对它求策略梯度等于让模型去优化它无法控制的外部内容,训练不稳。抽掉这一步,整个"搜索可稳定嵌入 RL"的卖点就垮了。【原文】§3.1、§5.4、Table 4

## 为什么做
- 研究背景:LLM 推理常需外部/时效知识。已有两类整合方式:(1) RAG——单轮"检索→生成",query=LLM 输入,多轮交互弱;(2) 搜索当工具——提示式(IRCoT/ReAct)泛化差,SFT 式(Toolformer)依赖难规模化的高质量标注轨迹,且搜索操作不可微、端到端梯度下降不可用。R1/o1 表明纯结果奖励 RL 能涌现推理(self-verification/self-correction),但用在"调搜索引擎"上还没被探索。【原文】§1、§2
- 解决的具体痛点:三个挑战——(1) RL Framework & Stability:怎么把搜索引擎稳定嵌入 RL(尤其检索 token 进入序列后);(2) Multi-Turn Interleaved:LLM 应能迭代地按问题复杂度动态调整检索;(3) Reward Design:简单结果奖励是否够,还是必须复杂过程奖励。【原文】§1
- 相关工作 & 各自不足(来龙去脉 + 并行路线 + 各自短板 + 站谁肩上):
  - **(路线 A)RAG 系**:经典 RAG 是单轮"检索→拼上下文→生成",query 直接取自 LLM 输入。**短板**【原文 §2.1】:易检索到无关信息、上下文一次性给满后不可动态补充、多轮交互弱;Self-RAG/IRCoT 等做了多轮检索但靠 prompt 编排。
  - **(路线 B)把搜索当工具的两条子线**:① **提示式**(IRCoT、ReAct、Search-o1)——靠 in-context 指令让模型交替"想-搜",**短板**:泛化差、行为不稳、不更新模型权重;② **SFT 式**(Toolformer、拒绝采样蒸馏)——监督模型何时插工具调用,**短板**:依赖难规模化的高质量标注/筛选轨迹,且**搜索操作不可微**,无法把检索整链端到端反传。Search-R1 站在这条线的肩上,但用 **RL 直接优化最终结果**绕开"标注轨迹"与"不可微"两个坎。
  - **(路线 C)直接偏好优化族 DPO/SimPO/RLOO**:可优化结果偏好。**短板**【原文 §2.2】:有 off-policy 问题(在固定数据上学偏好),效果不及在线纯 RL;且未针对"多轮含检索"的交织序列设计。
  - **(路线 D,最近邻/直接奠基)R1-Zero 式纯结果奖励 RL**:DeepSeek-R1-Zero(Guo 2025)证明从 base 起、仅规则奖励即可涌现 self-verification/self-correction。**短板/空白**:只在**纯参数推理**(模型脑内知识)上演示,"把真实搜索引擎当环境"这一扩展未被探索。Search-R1 正是这块的填补:把环境从"纯参数"扩展为"含真实检索",并发现 **GRPO 长训会 reward 坍缩**(§5.1)而改默认用 PPO。
  - 关键差别(与最近邻 R1-Zero / 与 Search-o1):vs R1-Zero——rollout 序列**交织 LLM 生成与真实检索**,需处理"检索 token 进梯度"问题(→ loss masking);vs Search-o1(prompt 式多轮检索)——**端到端训练**模型学检索策略而非靠 prompt。承重设计就是 **loss masking 让"检索 token 不进梯度"** + **结果奖励驱动的自发涌现**。
- 动机链:LLM 缺时效/领域知识且会幻觉 → 需边推理边检索 → prompt/SFT 式工具调用泛化差且依赖标注 → 用纯结果奖励 RL 让模型自学搜索行为 → 但检索 token 进序列会让 RL 不稳 → 用 loss masking 解决 + 最简 EM 奖励验证"够用"。

## 怎么做(到可复现粒度)
- **总目标(Eq.1)**【原文 §3.1】:把搜索引擎 \(R\) 写进 RL 目标,最大化结果奖励减 KL 锚定:
  \(\displaystyle \max_{\pi_\theta}\ \mathbb{E}_{x\sim D,\ y\sim\pi_\theta(\cdot|x;R)}\big[r_\phi(x,y)\big]\;-\;\beta\,D_{\mathrm{KL}}\!\big[\pi_\theta(y|x;R)\,\|\,\pi_{\mathrm{ref}}(y|x;R)\big]\)
  关键记号:策略 \(\pi_\theta(\cdot|x;R)=\pi_\theta(\cdot|x)\otimes R\),其中 \(\otimes\) 表示"检索-推理交织"——即 rollout 序列由 LLM 生成 token 与检索回来的 token **交错拼接**而成。这区别于普通 RL 的 \(\pi_\theta(\cdot|x)\)(只生成、无外部观测)。
- **奖励(Eq.4,极简规则奖励)**【原文 §3.3】:
  \(\displaystyle r_\phi(x,y)=\mathrm{EM}(a_{\mathrm{pred}},a_{\mathrm{gold}})\)
  \(a_{\mathrm{pred}}\)=从 \(y\) 抽取的最终答案,\(a_{\mathrm{gold}}\)=ground truth。**显式不用 format reward**(因学到的模型已有强格式遵从)、**不训神经奖励模型**(规避 reward hacking 与 RM 训练成本,follow Guo 2025)。
- **多轮交织 rollout(Algorithm 1,可复现的循环逻辑)**【原文 §3.2、Alg.1】:输入 query \(x\)、策略 \(\pi_\theta\)、搜索引擎 \(R\)、最大 action 预算 \(B\)。
  1. 初始化 rollout 序列 \(y\leftarrow\varnothing\)、action 计数 \(b\leftarrow 0\)。
  2. **while** \(b<B\):
     - 内层 while 逐 token 生成 \(y_t\sim\pi_\theta(\cdot|x,\ y+y_b)\),追加到当前 action 段 \(y_b\);**一旦 \(y_t\in\{\)`</search>`, `</answer>`, `<eos>`\(\}\) 即 break**。
     - 把 \(y_b\) 接到总序列 \(y\)。
     - **若检出 `<search>...</search>`**:抽 query \(q\leftarrow\mathrm{Parse}(y_b)\),调 \(R\) 检索 top-k,把结果包进 `<information>...</information>` 追加回 \(y\);\(b\leftarrow b+1\)。
     - **若检出 `<answer>...</answer>`**:返回 \(y\)。
     - 否则(既没搜也没答):提示 "My action is not correct. Let me rethink." 继续。
  3. 到 \(B\) 上限停。
- **结构化模板(Table 1,原文逐字)**【原文 §3.3】:
  > Answer the given question. You must conduct reasoning inside `<think>` and `</think>` first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by `<search>` query `</search>`, and it will return the top searched results between `<information>` and `</information>`. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside `<answer>` and `</answer>` ... Question: question.

  设计意图:**只约束格式、不注入内容偏置**(不强制反思、不强制搜索、不背书特定解法),好让 RL 的"自然学习动态"保持可观测、无偏(§3.3 原话)。
- **检索 token 的 loss masking(核心承重设计)**【原文 §3.1】:定义掩码 \(I(y_t)\):若 \(y_t\) 是 **LLM 生成 token** 则 \(I(y_t)=1\),若是**检索回来的 token** 则 \(I(y_t)=0\)。所有 token 级损失项与 **KL 损失项**都乘 \(I(y_t)\)。直觉:检索 token 是外部环境观测,模型无法控制其内容,对它求策略梯度会引入"优化无法控制内容"的不稳定动态。
- **优化器之一:PPO with Search(Eq.2)**【原文 §3.1】(actor-critic,默认算法):
  \(\displaystyle \mathcal{J}_{\mathrm{PPO}}(\theta)=\mathbb{E}_{x\sim D,\ y\sim\pi_{\mathrm{old}}(\cdot|x;R)}\!\left[\frac{1}{\sum_{t}I(y_t)}\sum_{t:\,I(y_t)=1}\min\!\Big(\rho_t A_t,\ \mathrm{clip}(\rho_t,1\!-\!\epsilon,1\!+\!\epsilon)A_t\Big)\right]\)
  其中比率 \(\rho_t=\dfrac{\pi_\theta(y_t|x,y_{<t};R)}{\pi_{\mathrm{old}}(y_t|x,y_{<t};R)}\),\(\epsilon\) 是 PPO 裁剪超参;**优势 \(A_t\) 用 GAE**(Schulman 2015)基于未来奖励 \(\{r_{\geq t}\}\) 与学习到的价值函数 \(V_\phi\) 估计。注意归一化分母 \(\sum_t I(y_t)\) 与求和下标 \(t:I(y_t)=1\) 都体现"只算 LLM token"。
- **优化器之二:GRPO with Search(Eq.3)**【原文 §3.1】(去 critic,组内相对优势):对每个 \(x\) 采一组 \(G\) 条响应 \(\{y_1,\dots,y_G\}\),
  \(\displaystyle \mathcal{J}_{\mathrm{GRPO}}(\theta)=\mathbb{E}_{x\sim D,\ \{y_i\}_{i=1}^{G}\sim\pi_{\mathrm{old}}}\!\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{\sum_t I(y_{i,t})}\sum_{t:\,I(y_{i,t})=1}\!\min\!\Big(\rho_{i,t}\hat{A}_{i,t},\ \mathrm{clip}(\rho_{i,t},1\!-\!\epsilon,1\!+\!\epsilon)\hat{A}_{i,t}\Big)-\beta D_{\mathrm{KL}}[\pi_\theta\|\pi_{\mathrm{ref}}]\right]\)
  \(\hat{A}_{i,t}\)=用组内相对奖励算的优势(无需 value 网络);GRPO 把 KL **直接加到 loss**(而非像 PPO 塞进 reward),且 **KL 计算同样施 masking**。
- 逐组件必要性:
  - **loss masking**:做了消融(Table 4),去掉掉 0.088,**必要性最强**。机制上对应 Eq.2/3 里的 \(I(y_t)\) 在求和与归一化两处同时生效。
  - **结果奖励够用**:无过程奖励仍跨 7 数据集涨点,且 Base 模型也学会(§4.4 obs 3);但"够用"是经验论断,未与"加过程奖励"直接对比来证明"过程奖励无必要"。【推断】
  - **PPO vs GRPO**(§5.1、Table 3):GRPO 收敛快但长训**reward 坍缩**,PPO 更稳;两者最终奖励相当 → 默认用 PPO 规避坍缩。**坍缩根因未给**,只描述现象。
  - **Base vs Instruct**(§5.2):Instruct 收敛快、起点高,但最终奖励两者相近 → RL 能弥合差距。【原文】Table 3/4、§5.1–5.4
- 关键超参默认值(汇总,§4.3)【原文】:**action 预算 \(B=4\)**;检索 **top-k=3**(line 1275 "B is set to 4, and we retrieve the top 3 passages by default");**默认 RL 算法 = PPO**(GRPO 作对照);结果奖励 = EM;知识源 = 2018 Wikipedia dump;检索器 = E5。模型 = Qwen2.5-3B/7B(Base 与 Instruct)。(其余学习率/batch 等附录 B,本轮未逐项核。)
- 数据流动小结:query → 模板包裹 → 策略生成 `<think>`/`<search>` → 检测到 `</search>` 暂停生成、调 \(R\) 取 top-3、包 `<information>` 回插 → 继续生成直到 `<answer>` 或耗尽 \(B\) → EM 判分 → PPO/GRPO 用带 masking 的 token 损失更新策略(检索 token 不进梯度)。

## 靠不靠谱
- 实验与证据:
  - **数据集**:训练=合并 **NQ + HotpotQA** 训练集。评测 7 个 QA——General(NQ†、TriviaQA⋆、PopQA⋆)+ Multi-Hop(HotpotQA†、2Wiki⋆、Musique⋆、Bamboogle⋆;†in-domain/⋆OOD)。知识源=2018 Wikipedia dump,检索器=**E5**,top-k=**3**。指标 **EM**。模型 Qwen2.5-3B/7B(Base 与 Instruct)。【原文】§4.1、§4.3
  - **关键数字**(Table 2):**7B Search-R1-base Avg=0.431 vs RAG 0.304**(相对 +42%);3B base=0.303。摘要写 24%(7B)/20%(3B),贡献处写 **41%/20%**——两口径都出现:Table 2 数值印证"41%"(0.431/0.304−1≈0.42),"24%"应是相对另一组 baseline 取法。Search-R1>R1(无检索 RL)、>拒绝采样、>RAG/IRCoT/Search-o1。【原文】Abstract、§1(贡献)、Table 2
  - **baseline 公平吗**:明确同检索器/同检索文档数/同语料/同训练集/同预训练 LLM(§4.2),R1 baseline 也用本文数据训练以公平对比——口径控制良好。
  - **看着强但没回答核心问题?**:"检索可稳定嵌入 RL"由 masking 消融 + 跨 7 集一致增益强力支撑;但 EM 奖励对"语义对但表述不同"会误判(reward hacking 空间),论文未深究;in-domain 仅 NQ/HotpotQA 两源,训练分布仍偏窄。【推断】
- 假设与失效边界:
  - 【原文】用 EM 作结果奖励(隐含假设答案可字符串匹配判正误);检索器+2018 静态 Wikipedia 是知识上界,时效/在线噪声未在主实验压测;主文仅 Qwen2.5(跨架构 Llama 结果在扩展版 2505.15117,COLM 主文不含)。
  - 【推断】GRPO 长训坍缩的失效边界只描述未解释,换数据/超参时何时坍缩不可预测;EM 的语义盲区在开放式/长答案任务上会更严重。
- 祛魅总结:真贡献=**第一个把"真实搜索引擎"稳定嵌入纯结果奖励 RL 的开源框架**,loss masking 这一极简却关键的设计 + "最简结果奖励够用"的经验结论,影响力大(被 veRL 官方、SkyRL、Thinking Machines Tinker-cookbook 等多方复现,印证可靠)。包装/瑕疵:(1) 摘要 24% 与贡献 41% 双口径并存,易致引用混淆;(2) "结果奖励足够"是经验主张,缺与过程奖励的正面对比;(3) GRPO 坍缩、EM reward hacking 都点到为止。【推断】

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=最终答案 EM 正确性(纯结果奖励,token 级 advantage 经 GAE/组内相对)｜**改什么**=policy LLM 参数(PPO/GRPO 更新),检索 token 不进梯度｜**何时改**=在线 RL,边 rollout 边更新｜**免梯度?**=否(RL 更新参数)｜**记忆-技能生命周期**=无显式记忆/技能库;外部知识全靠实时检索(非参数),检索策略本身固化进参数｜**防遗忘机制**=无专门设计;靠 \(\beta D_{\mathrm{KL}}\)(对 \(\pi_{\mathrm{ref}}\))约束防策略漂移。【原文】§3
- ⑦ 开源代码+框架/harness:https://github.com/PeterGriffinJin/Search-R1(Tier A,~8.3M,多方复现)。框架=**veRL**(仓库内置定制 fork,`VERL_README.md`=HybridFlow/veRL),rollout 用 vLLM;支持 PPO(默认)/GRPO/REINFORCE;检索支持本地 BM25/E5+ANN 与在线搜索引擎。核心脚本 `train_ppo.sh`/`train_grpo.sh`/`retrieval_launch.sh`,逻辑在 `search_r1/{llm_agent,search}`。代码可得性 Tier A。【原文】GitHub + 既有 analysis 核查
- 💰 资源/成本与可扩展性:训练数据=NQ+HotpotQA 训练集;检索 top-k=3、action 预算 \(B=4\)(§4.3 已核)。需在线起检索服务(`retrieval_launch.sh`)+ veRL RL,成本主要在 RL rollout + 检索调用。原文未给具体 GPU·小时。【原文】§4.3
- 🎯 对"探索-巩固"对标:**底座/对照型支撑,主要在 L4 的"自主探索有效工具调用"这一侧**。Search-R1 让模型**自主探索"何时/如何检索"并把有效检索策略固化进参数**——对应 idea 的"探索=发现有效行为/路径"+"巩固=固化进参数";on-policy 体现在 RL rollout 是策略自己生成。**可借组件**:(a) **检索 token loss masking**(Eq.2/3 的 \(I(y_t)\))是处理"多轮 agent 中外部观测 token"的标准做法,任何含工具调用/MTP 外部反馈的 OPD 训练都该借;(b) 多轮交织 rollout 的结构化标签 + 纯结果奖励是 agentic RL 的干净底座。**缺口/差异**:纯结果奖励无 token/段级稠密信用(idea 关心 token 信用与 path-recovery),无 teacher 脚手架(是 from-scratch RL 非蒸馏),无 MTP 前瞻,无"走偏后选恢复分支"的显式机制(只有 "Let me rethink" 的粗回退)。一句判定:**agentic OPD/工具调用的奠基底座与对照基线,loss masking 可直接复用,但缺稠密信用、teacher 脚手架与 MTP 前瞻,非 path-recovery 的精细实现**。【推断,依据 §3 vs idea】
- 🔭 开放问题/未来方向:【原文】更复杂的 reward 机制、基于不确定性的动态检索调整、结合多样工具与信息源、多模态推理(§6 Conclusion)。【推断】(1) 用语义奖励/LLM-judge 替 EM 以堵 reward hacking 与语义盲区;(2) 给 GRPO 坍缩找根因并修复;(3) 引入 token/段级稠密信用(可接 MTP 前瞻)替代纯结果奖励,提升长 horizon 信用分配;(4) 加"检测到检索失败后选择恢复分支"的显式 path-recovery 机制。

RETURN:search_r1 | 读PDF? 是(31页:全方法/模板Table1原文/Algorithm1全循环/Eq.1-4全式/Table2-4/§5全分析) | 加厚? 是(方法到可复现:Eq.1目标+Eq.2 PPO+Eq.3 GRPO+Eq.4奖励全转MathJax含ρ/clip/GAE/I(yt);Alg.1逐步;模板原文;相关工作四路线A-D精确短板;B=4/top3超参核出) | LaTeX公式条数 6 | 待核数 0(原待核 action预算/top-k 已从§4.3核出 B=4/k=3)

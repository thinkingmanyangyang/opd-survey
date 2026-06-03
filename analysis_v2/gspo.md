gspo | Group Sequence Policy Optimization (GSPO) | Qwen Team, Alibaba（通讯 Chujie Zheng / Bowen Yu）| arXiv 2507.18071v2, 2025-07-28 · 预印本（技术报告体例）| 主题线 L3（RLVR/GRPO 算法）·相关性 中

**原始论文**：https://arxiv.org/abs/2507.18071

## 一眼看懂
- 🟦 TL;DR：GRPO 在 **token 级**做重要性采样校正是「病态(ill-posed)」的——重要性采样的原理要求在行为分布上对**多样本(N≫1)** 取平均，权重 πtar/πbeh 才能完成分布校正；但 GRPO 在**每个 token 位置**用**单一样本**算比值，根本完不成校正，只是注入高方差噪声，噪声随序列变长累积、被 clip 放大，最终在大模型(尤其 MoE)上引发**常不可逆的崩溃**【原文 §3, 式4】。GSPO 改在**序列级**定义（长度归一化的）重要性比 si=(πθ(y)/πθold(y))^(1/|y|)，做序列级裁剪/奖励/优化，使「优化单位 = 奖励单位 = 整条序列」对齐；已用于训练最新 Qwen3【摘要, §4.1, §5】。
- 最巧的一步：**把重要性比的定义单位从 token 提到 sequence，并加 1/|y| 长度归一化**（式7）。抽掉它就是 GRPO。这一步的承重逻辑是「奖励授予整条序列，off-policy 校正就该在序列级做」；长度归一化则把不同长度响应的 si 收进统一数值区间、避免少数 token 的似然变化引起 si 剧烈抖动。**对 MoE 这步是救命的**：序列似然 πθ(y) 不像单 token 似然那样因专家路由抖动(每次更新~10% 激活专家变化)而剧烈波动，所以 GSPO 天然免疫 expert-activation volatility，无需 Routing Replay。

## 为什么做
- 研究背景：大规模 RL 是 scale 推理(o1/R1/Qwen3)的关键范式；continued RL 的前提是训练动态稳定，但 SOTA 的 GRPO 在巨型模型上有严重稳定性问题、常灾难性不可逆崩溃【原文 §1，引 Qwen3/MiniMax】。
- 解决的具体痛点：(1) GRPO token 级重要性比**误用**重要性采样原理——单样本无法做分布校正、只注入高方差噪声(§3)；(2) 噪声随序列变长累积、被 clip 放大 → 崩溃后**回滚 checkpoint / 调 clip / 改 query / 延长生成长度均无效**(§3，强陈述);(3) **MoE 雪上加霜**——48 层 Qwen3-30B-A3B 每次更新同一响应约 10% 激活专家变化、越深越显著，使 token 级比值剧烈抖动(§5.3)。
- 相关工作 & 各自不足：PPO（需同规模 value 模型，显存/算力重、value 可靠性与长序列扩展难）；GRPO（去 value 用组内相对优势，竞赛数学/代码 SOTA 基线，但超大模型不稳）；GRPO+Routing Replay（Qwen 自己之前的 MoE 补丁——缓存 πθold 激活专家在 πθ「重放」，能收敛但增内存/通信开销、且限制 MoE 实际容量，§5.3）。
- 动机链：GRPO 崩溃 → 根因是 token 级重要性比病态(单样本不能校正分布) → 「优化单位应与奖励单位匹配」 → 奖励是序列级的，故重要性比/裁剪/优化都应放到序列级 → GSPO。
- 与最近邻工作的Δ：vs **GRPO**——重要性比从 token→sequence + 长度归一化(式7 vs 式3)，梯度上 GSPO 对一条响应内所有 token **等权**(消除 GRPO 不等权累积的不稳定因子，§4.2)；vs **GMPO/几何平均**(并发)——GSPO 也在 log 空间对 token 比取均值再 exp，**与几何平均形式高度相通**(holderpo 把二者统一为 p→0 极限)，但 GSPO 用序列级单一裁剪、明确为「修重要性采样误用」立论，且**实战已上 Qwen3 全量训练**(工程验证更重);vs **Routing Replay**——GSPO 从根上免疫 MoE 路由抖动，直接删掉这套补丁。

## 怎么做 + 靠不靠谱
- 方法流水线：① 组内采 G 条响应、算组内相对优势 Âi(同 GRPO，免 value，整条序列共享 Âi)→ ② 对每条响应算序列级重要性比 si=(πθ(yi)/πθold(yi))^(1/|yi|)= exp((1/|yi|)Σ log(πθ/πθold))→ ③ **序列级裁剪**(min(si·Âi, clip(si,1−ε,1+ε)·Âi))→ ④ 序列级优化。裁剪窗因 si 定义不同与 GRPO **差一个数量级**(GSPO 用 3e-4/4e-4，GRPO 用 0.2/0.27)。
- 变体 **GSPO-token**(§4.3)：多轮 RL 等需 token 级优势微调时，用 si,t = sg[si]·(πθ/sg[πθ]) (sg=stop-gradient/detach)——**数值上 ≡ si，但允许 per-token 优势定制**；当所有 token 优势相同时与 GSPO 完全等价(目标/裁剪/梯度)。〔注：这正是 MEMORY 里「forward-hard/backward-soft、stop-gradient 解耦」签名的一个实例——前向取序列级数值、反向走 token 级梯度〕。
- 逐组件必要性：
  - **序列级 vs token 级比值**：核心。图1 训练曲线 GSPO 全程稳、GRPO(w/ Routing Replay)在同算力下 reward/AIME24/LCB/CodeForces 均更低。**无传统消融表**(技术报告体例)，靠训练曲线+机理论证。
  - **长度归一化 1/|y|**：§4.1 论证不归一会使 si 随长度剧烈波动、不同长度需不同裁剪窗——**未给独立消融数字**，标〔推断〕其必要性主要靠理论论证。
  - **免 Routing Replay**：图3 证明 GRPO 去掉 Routing Replay 即崩(reward 从 0.5 跌到 0.25);GSPO 无需它仍稳(图1)——这是 MoE 侧最硬的证据。
- 关键机制/公式（直觉）：梯度上(式10 vs 12)，GRPO 每 token 权重是各自的 πθ/πθold(范围可达 (0,1+ε] 或 [1−ε,+∞)，不可忽略、会累积出不可预测后果)；GSPO 把整条响应所有 token 权重压成同一个 si，等权处理、消除累积不稳定源。
- 实验与证据：base=Qwen3-30B-A3B-Base 冷启微调；评测 AIME'24(Pass@1 over 32)、LiveCodeBench(202410-202502, Pass@1 over 8)、CodeForces(Elo)。关键观察：(a) GSPO 同算力下三基准全面优于 GRPO+RoutingReplay(图1);(b) **反直觉现象**——GSPO 裁掉的 token 比例(0.15)比 GRPO(0.0013)高**两个数量级**，用更少 token 训练却效率更高 → 反证 GRPO token 级梯度本就 noisy、样本利用低效(§5.2);(c) 序列级似然对训练/推理引擎精度差更宽容，可直接用 inference engine 的 likelihood 省去 training engine 重算(利好 partial rollout/多轮/分离式框架，§5.4)。
- baseline 公平吗：GRPO 的裁剪窗「精心调过以保证公平对比」(§5.1)，且对照了 GRPO 有/无 Routing Replay。**但全文无具体最终分数表、无随机种子/方差、无标准消融**——属技术报告而非标准实证论文；「贡献于 Qwen3」是背书而非可复现实验。
- 假设与失效边界：【原文】(1) GSPO 用序列级似然，假设 MoE「始终维持语言建模能力故序列似然不剧烈波动」(§5.3)——此假设若被破坏(如极端分布漂移)未必成立;(2) 裁剪窗与 GRPO 差数量级，需重调。【推断】(3) 序列级等权处理 = **放弃 token 级信用分配的细粒度**——对「关键 token 应被区别对待」的任务(正是 hapo/holderpo 的出发点)，GSPO 反而把信号抹平，作者也因此补了 GSPO-token 变体;(4) 长度归一化对超长/超短响应的极端情形鲁棒性未实测;(5) 无 value、靠组内相对优势，继承 GRPO 在「全对/全错组无方差→无梯度」的难度匹配问题(本文未涉及)。
- 祛魅总结：真贡献=**(a) 一针见血指出 GRPO token 级重要性比的理论病灶(误用重要性采样)+ (b) 序列级重写一举免疫 MoE 路由抖动、删掉 Routing Replay 补丁 + (c) Qwen3 量级的工程验证**——影响力大、立论清晰、MoE 救场实用。需打折的：作为技术报告**缺标准消融/方差/最终分数表**，「优于 GRPO」主要靠训练曲线;且与 GMPO/几何平均**本质同源**(holderpo 已统一)，序列级是 token 级几何平均的极限情形，故「全新算法」更应理解为「把聚合算子推到序列极限 + 把 MoE 稳定性讲透」。【推断】对「探索-巩固」是 RL 优化层稳定器，与蒸馏/记忆/前瞻无直接关系。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=组内相对优势 Âi(可验证 0/1 奖励，序列级；整条响应共享)｜**改什么**=策略参数 θ(重要性比/裁剪单位 token→sequence)｜**何时改**=on-policy RL 每步(mini-batch off-policy 更新)｜**免梯度?**=否，policy gradient｜**记忆-技能生命周期**=无｜**防遗忘机制**=间接(序列级稳定性防 MoE 崩溃/不可逆漂移)，非显式防遗忘。
- ⑦ 开源代码+框架/harness：**无独立官方仓**——算法已被社区集成进 **veRL / TRL / ms-swift / ROLL** 等框架(均有 GSPO loss 实现)；论文实验用 **Qwen 内部 RL 设施**(训练 Megatron + 推理 SGLang/vLLM，§5.4)，未开源。CLAUDE 决策：从 PDF 读方法，无 clone 集。〔待核：各框架实现与原文式5/7 的等价性应以具体代码为准〕→ 实为**低风险待核**(多框架已收录，公认正确)。
- 💰 资源/成本与可扩展性：面向**巨型模型/MoE/长响应**的大 rollout batch 场景(§3 动机即此);相对 GRPO 无额外算力，反而**省**(删 Routing Replay 的内存/通信、可省 training-engine 重算 logprob)。已扩到 30B-A3B 及 Qwen3 全系。
- 🎯 对"探索-巩固"对标：**可借组件/竞品(RL 优化层)，非直接对标**。判定依据：① GSPO 让超大/MoE 模型 RL 不崩，是「巩固」阶段的**基础设施保障**——若 TSRD 在大模型上做 on-policy RL/蒸馏，GSPO 是稳定底座;② **GSPO-token 的 stop-gradient 构造(式14)与 MEMORY 记录的「forward-hard/backward-soft 解耦」同构**，是可直接借用的工程模式——可用于「前向走硬的路径选择/序列级量、反向流平滑梯度」;③ **张力**：GSPO 序列级等权会**抹平 token 级信号**，与 path-recovery「单点关键 token 接管」、MTP「关键步前瞻」的诉求方向相反——GSPO 想消除 token 异质带来的不稳定，TSRD 想利用 token 异质。缺口:无 teacher 脚手架、无选路/回轨、无记忆。
- 🔭 开放问题/未来方向：【原文】结尾仅愿景式「以 GSPO 为基石继续 scale RL」，未列具体 open problem。【推断】(1) 序列级 vs token 级的「稳定性 vs 细粒度信用」trade-off——holderpo/hapo 正是回应;(2) GSPO-token 在真多轮/agent RL 的实证缺失(§4.3 仅提动机);(3) 与显式 token 加权(关键步)如何调和而不重新引入不稳定;(4) 缺标准消融/方差，可复现性待社区补。

key|读到PDF?|L线|对标结论|残留待核数
gspo | 是(全文7页正文) | L3 | 可借RL优化层稳定底座(MoE防崩);GSPO-token的stop-gradient≡forward-hard/backward-soft可借;与token级信用(path-recovery/MTP)方向相反有张力 | 1(各框架实现等价性,低风险)

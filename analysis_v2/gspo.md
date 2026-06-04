gspo | Group Sequence Policy Optimization (GSPO) | Qwen Team, Alibaba（通讯 Chujie Zheng / Bowen Yu）| arXiv 2507.18071v2, 2025-07-28 · 预印本（技术报告体例）| 主题线 L3（RLVR/GRPO 算法）·相关性 中

**原始论文**：https://arxiv.org/abs/2507.18071 ｜ 无独立官方仓（算法已被 veRL/TRL/ms-swift/ROLL 集成；本分析的公式均从 PDF 抄准）

## 一眼看懂
> 一句话导读：GRPO 把「重要性采样」校正放在每个 token 上算，本质用错了——重要性采样要靠多样本平均才有意义，单样本只会注入噪声，模型一大(尤其 MoE)就崩。GSPO 把这步搬到整条序列上算，问题就消了。

- 🟦 TL;DR：先说病灶。GRPO 在 **token 级**做重要性采样校正是「病态(ill-posed)」的。原因：重要性采样的原理要求在行为分布上对**多样本(N≫1)** 取平均，权重 πtar/πbeh 才能完成「从行为分布校正到目标分布」；但 GRPO 在**每个 token 位置**只用**单一样本**算这个比值，根本完不成校正，反而注入高方差噪声。噪声随序列变长累积、又被 clip 放大，最终在大模型(尤其 MoE)上引发**常常不可逆的崩溃**【原文 §3, 式4】。
- GSPO 的修法：改在**序列级**定义重要性比——用长度归一化的 si=(πθ(y)/πθold(y))^(1/|y|)，并在序列级做裁剪/奖励/优化。这样「优化的单位 = 奖励的单位 = 整条序列」三者对齐。该算法已用于训练最新的 Qwen3【摘要, §4.1, §5】。
- 最巧的一步：**把重要性比的定义单位从 token 提到 sequence，并加一个 1/|y| 长度归一化**（式7）。抽掉它就退回 GRPO。
  - 承重逻辑：「奖励是授予整条序列的，那 off-policy 校正也该在序列级做」（§3 末句"the unit of optimization objective should match the unit of reward"）。
  - 长度归一化的作用：把不同长度响应的 si 收进统一数值区间，避免少数 token 的似然变化就让 si 剧烈抖动。
  - **对 MoE 这一步是救命的**：序列似然 πθ(y) 不像单 token 似然那样会因专家路由抖动(每次更新约 10% 的激活专家会变)而剧烈波动。所以 GSPO 天然免疫这种 expert-activation volatility，不再需要 Routing Replay 这类补丁。

## 为什么做
> 一句话导读：模型越大、越想长时间跑 RL，GRPO 的崩溃就越致命；本文要找出崩溃的根因(token 级重要性比用错了)并从根上修掉。

- 研究背景：大规模 RL 是把推理能力 scale 上去(o1/R1/Qwen3)的关键范式。持续跑 RL 的前提是训练动态稳定，但 SOTA 的 GRPO 在巨型模型上有严重稳定性问题、常常灾难性且不可逆地崩溃【原文 §1，引 Qwen3/MiniMax-M1】。
- 解决的具体痛点：
  - (1) GRPO 的 token 级重要性比**误用**了重要性采样原理——单样本根本做不了分布校正，只注入高方差噪声(§3)。
  - (2) 噪声随序列变长累积、又被 clip 放大；一旦崩溃，**回滚 checkpoint / 调 clip / 换 query / 加长生成长度全都无效**(§3，措辞很强)。
  - (3) **MoE 雪上加霜**——48 层的 Qwen3-30B-A3B，每次更新同一条响应约有 10% 的激活专家会变、越深越显著，使 token 级比值剧烈抖动(§5.3)。

- 相关工作 & 各自不足（§2 Preliminaries 给了 PPO/GRPO 的完整公式，下面补全语境）：
  1. **PPO**(Schulman 2017)：用 πθold 采样，靠 clip 把更新限制在 proximal(邻近)区域；目标见式1。
     - **短板**（§2 原话）：依赖一个与 policy 同规模的 **value 模型**——显存/算力都重；效果还取决于 value 估计是否可靠，扩到长响应/复杂任务时 value 更难训。
  2. **GRPO**(Shao 2024, DeepSeekMath；最近邻、被批判的对象)：去掉 value，改用**组内相对优势**（式3：\(\hat A_i=(r-\text{mean})/\text{std}\)，组内所有 token 共享同一个优势）。它是竞赛数学/代码的 SOTA 基线。
     - **短板**：超大模型上不稳、不可逆崩溃（§1/§3 的核心靶子）。
  3. **GRPO + Routing Replay**(Qwen 自己之前打的 MoE 补丁，§5.3)：缓存 πθold 那一步的激活专家，在 πθ 上「重放」同一套路由，使 token 级比值的分子分母走同一个子网络，从而能正常收敛（图3 证明缺了它就崩，reward 从 0.5 跌到 0.25）。
     - **短板**：增加内存/通信开销，且**限制了 MoE 的实际容量**。GSPO 从根上免疫路由抖动，于是可以直接删掉这套补丁。
  4. **用序列似然当信号的前作**：Zheng 2023（CLICK, sequence likelihood contrastive）——GSPO 的序列级重要性比定义就引自此。
  5. **几何平均/并发路线（本质同源）**：GMPO/几何平均的做法是把 token 比在 log 空间取均值、再 exp 回来——而 GSPO 的式7 `si=exp((1/|y|)Σlog(πθ/πθold))` 形式上**正是 token 比的几何平均**。holderpo 后来把 GRPO(算术均 p=1) 和 GSPO·GMPO(几何均 p→0) 统一成了 Hölder p-mean 的特例。
     - 差异：GSPO 用**序列级的单一裁剪**、明确以「修复重要性采样误用」立论，并且**实战上已经撑起 Qwen3 全量训练**（工程验证更硬）。

- 动机链（一步步推）：
  1. GRPO 会崩溃。
  2. 根因是 token 级重要性比病态(单样本无法校正分布)。
  3. 原则：「优化的单位应与奖励的单位匹配」。
  4. 既然奖励是序列级的，那重要性比/裁剪/优化都应放到序列级。
  5. 于是得到 GSPO。
- 与最近邻工作的精确Δ：
  - vs **GRPO**：重要性比从 token 提到 sequence、再加长度归一化(式7 vs 式3)；梯度上 GSPO 让一条响应内所有 token **等权**(消除了 GRPO 那种不等权累积的不稳定因子，§4.2)；裁剪窗因 si 定义不同，与 GRPO 差一个数量级。
  - vs **GMPO/几何平均**：形式同源(holderpo 已把它们统一为 p→0 极限)，但 GSPO 用序列级单裁剪 + Qwen3 工程验证。
  - vs **Routing Replay**：从根上免疫 MoE 路由抖动，直接删掉补丁。

## 怎么做 + 靠不靠谱
> 一句话导读：流程跟 GRPO 几乎一样，唯一变化在第 2 步——把每个 token 的比值先在 log 空间平均、再做长度归一，得到一条「序列级比值」si，然后所有裁剪/优化都对 si 做。

### 方法流水线（读完可复现，全部基于 §2/§4 + 式5/6/7）
**符号**：x=query；{yi}_{i=1}^G=对 x 采的 G 条响应（组大小 G）；πθold=rollout 时的旧策略；πθ=待优化策略；r(x,y)∈[0,1]=verifier 奖励；|y|=响应的 token 数。

1. **组采样 + 组相对优势**（同 GRPO，免 value，式6）：对每个 x 用 πθold 采 G 条响应，整条响应共享同一个优势
   \(\displaystyle \hat A_i=\frac{r(x,y_i)-\mathrm{mean}\big(\{r(x,y_j)\}_{j=1}^{G}\big)}{\mathrm{std}\big(\{r(x,y_j)\}_{j=1}^{G}\big)}.\)
2. **算序列级重要性比**（式7，承重公式 = 长度归一的 token-比几何平均）：
   \(\displaystyle s_i(\theta)=\Big(\frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{\text{old}}}(y_i\mid x)}\Big)^{1/|y_i|}=\exp\!\Big(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}\mid x,y_{i,<t})}\Big).\)
   直觉(它在量什么)：si 衡量整条响应从 πθold 走到 πθ 偏离了多远，**天然匹配序列级奖励**。1/|y| 的作用是把不同长度的 si 收进统一数值区间——否则少数 token 的似然变化就会让 si 剧烈抖动，而且不同长度还得用不同裁剪窗(§4.1)。
3. **序列级裁剪 + 优化**（式5，GSPO 主目标）：
   \(\displaystyle J_{\text{GSPO}}(\theta)=\mathbb{E}_{x\sim D,\,\{y_i\}\sim\pi_{\theta_{\text{old}}}}\Big[\frac{1}{G}\sum_{i=1}^{G}\min\big(s_i(\theta)\hat A_i,\ \mathrm{clip}(s_i(\theta),1-\varepsilon,1+\varepsilon)\hat A_i\big)\Big].\)
   裁剪是施加在**整条响应**上，而非单个 token。实战裁剪窗 ε：GSPO 用 **3e-4/4e-4**(左/右)，GRPO 用 **0.2/0.27**——因为 si 的定义不同，两者**差一个数量级**(§5.1)。
4. **mini-batch off-policy**：把大 rollout batch 切成 4 个 mini-batch 做梯度更新（§5.1）。因此 y 是采自 πθold(≠πθ)，clip 正是为这个 off-policy gap 而设。

### 关键梯度分析（式10 vs 式12，解释「为何稳」）
> 一句话导读：对比两个梯度就看一处——GSPO 让一条响应里所有 token 共用同一个权重(等权)，GRPO 则每个 token 带各自的比值，后者会累积出不可控的抖动。

- **GSPO 梯度**（式10，clip 略）：
  \(\displaystyle \nabla_\theta J_{\text{GSPO}}=\mathbb{E}\Big[\frac{1}{G}\sum_i \Big(\tfrac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{\text{old}}}(y_i\mid x)}\Big)^{1/|y_i|}\hat A_i\cdot\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\nabla_\theta\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})\Big].\)
  关键：整条响应里**所有 token 的权重被压成同一个 si**、彼此等权。
- **GRPO 梯度**（式12）：
  \(\displaystyle \nabla_\theta J_{\text{GRPO}}=\mathbb{E}\Big[\frac{1}{G}\sum_i \hat A_i\cdot\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}\mid x,y_{i,<t})}\nabla_\theta\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})\Big].\)
  关键：每个 token 带的是各自的 πθ/πθold（对 \(\hat A_i>0\) 落在 (0,1+ε]、对 \(\hat A_i<0\) 落在 [1−ε,+∞)）。这些权重**不可忽略、会累积出不可预测的后果**——这就是 GRPO 的不稳定源，而 GSPO 把它消掉了(§4.2)。

### GSPO-token 变体（§4.3，多轮/agent 需要 token 级优势时用）
> 一句话导读：当任务确实需要给每个 token 不同优势(如多轮/agent)，这个变体用 stop-gradient 做到「数值上还是序列级、但梯度走 token 级」——即「前向硬、反向软」的解耦。

- 构造（式14）：\(s_{i,t}(\theta)=\mathrm{sg}[s_i(\theta)]\cdot\dfrac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\mathrm{sg}[\pi_\theta(y_{i,t}\mid x,y_{i,<t})]}\)，其中 sg[·]=stop-gradient（即 PyTorch 的 detach）。
- 性质：
  - 第二项的数值恒等于 1，所以 \(s_{i,t}\) 在数值上 ≡ \(s_i\)（前向取的是序列级数值）；
  - 但梯度（式17）走的是 token 级 \(\nabla_\theta\log\pi_\theta(y_{i,t}\mid\cdot)\)，从而允许给每个 token 定制优势 \(\hat A_{i,t}\)；
  - 当所有 token 优势相同(\(\hat A_{i,t}=\hat A_i\))时，它与 GSPO 在目标/裁剪/梯度上**完全等价**。
- 〔注：这正是 MEMORY 里「forward-hard/backward-soft、stop-gradient 解耦」签名的一个实例——**前向取序列级数值、反向走 token 级梯度**〕。

### 逐组件必要性 + 证据
> 一句话导读：注意这是一篇技术报告，证据主要是训练曲线 + 机理论证，没有传统的消融表/方差，读时心里要留个折扣。

- **序列级 vs token 级比值**：核心改动。图1 的训练曲线显示 GSPO 全程稳，而 GRPO(带 Routing Replay)在同算力下 reward/AIME24/LCB/CodeForces 都更低。注意：**没有传统消融表**(技术报告体例)，靠训练曲线 + 机理论证支撑。
- **长度归一化 1/|y|**：§4.1 论证了不归一会使 si 随长度剧烈波动、且不同长度要用不同裁剪窗——但**没给独立的消融数字**，所以标〔推断〕：其必要性主要靠理论论证。
- **免 Routing Replay**：图3 证明 GRPO 去掉 Routing Replay 就崩(reward 0.5→0.25)；GSPO 不要它也稳(图1)——这是 MoE 侧最硬的一条证据。
- 实验与证据：
  - 配置：base=Qwen3-30B-A3B-Base 冷启微调；评测 AIME'24(Pass@1 over 32)、LiveCodeBench(202410-202502, Pass@1 over 8)、CodeForces(Elo)。
  - 关键观察：
    - (a) 同算力下，GSPO 在三个基准上全面优于 GRPO+RoutingReplay(图1)；
    - (b) **反直觉现象**——GSPO 裁掉的 token 比例(0.15)比 GRPO(0.0013)高**两个数量级**(图2)，却用更少 token 训出更高效率。这反过来证明 GRPO 的 token 级梯度本就 noisy、样本利用低效(§5.2)；
    - (c) 序列级似然对「训练引擎和推理引擎之间的精度差」更宽容，于是可以直接用 inference engine 算出的 likelihood、省掉 training engine 的重算(利好 partial rollout / 多轮 / 训推分离框架，§5.4)。
- baseline 公平吗：GRPO 的裁剪窗是「精心调过以保证公平对比」的(§5.1)，并且对照了 GRPO 有/无 Routing Replay 两种。**但全文没有具体的最终分数表、没有随机种子/方差、也没有标准消融**——它属于技术报告而非标准实证论文；「贡献于 Qwen3」是一种背书，而非可复现实验。
- 假设与失效边界：
  - 【原文】(1) GSPO 用序列级似然，前提假设是 MoE「始终维持语言建模能力、故序列似然不会剧烈波动」(§5.3)——这个假设若被破坏(如极端分布漂移)就未必成立；(2) 裁剪窗与 GRPO 差一个数量级，需要重调。
  - 【推断】(3) 序列级等权处理 = **放弃了 token 级信用分配的细粒度**。对那些「关键 token 应被区别对待」的任务(这正是 hapo/holderpo 的出发点)，GSPO 反而把信号抹平了——作者也因此补了 GSPO-token 变体。
  - 【推断】(4) 长度归一化在超长/超短响应等极端情形下的鲁棒性未实测。
  - 【推断】(5) 它无 value、靠组内相对优势，因而继承了 GRPO 的一个难题：当一组全对或全错时方差为 0、就没有梯度(本文未涉及)。
- 祛魅总结：
  - 真贡献=**(a)** 一针见血指出 GRPO token 级重要性比的理论病灶(误用重要性采样——式4 的 N≫1 前提被单样本破坏)；**(b)** 序列级重写一举免疫 MoE 路由抖动、删掉 Routing Replay 补丁；**(c)** Qwen3 量级的工程验证。整体影响力大、立论清晰、MoE 救场实用。
  - 需打折的：作为技术报告**缺标准消融/方差/最终分数表**，「优于 GRPO」主要靠训练曲线；而且它与 GMPO/几何平均**本质同源**(holderpo 已统一)——序列级只是 token 级几何平均的极限情形。所以「全新算法」更应理解为「把聚合算子推到序列极限 + 把 MoE 稳定性讲透」。
  - 【推断】对「探索-巩固」而言，它是 RL 优化层的稳定器，与蒸馏/记忆/前瞻没有直接关系。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**=组内相对优势 Âi(可验证的 0/1 奖励，序列级；整条响应共享)；
  - **改什么**=策略参数 θ(把重要性比/裁剪的单位从 token 提到 sequence)；
  - **何时改**=on-policy RL 每步(mini-batch off-policy 更新)；
  - **免梯度?**=否，是 policy gradient；
  - **记忆-技能生命周期**=无；
  - **防遗忘机制**=间接(用序列级稳定性防 MoE 崩溃/不可逆漂移)，并非显式的防遗忘。
- ⑦ 开源代码+框架/harness：**无独立官方仓**。
  - 算法已被社区集成进 **veRL / TRL / ms-swift / ROLL** 等框架(都有 GSPO loss 实现)；论文实验用的是 **Qwen 内部 RL 设施**(训练 Megatron + 推理 SGLang/vLLM，§5.4)，未开源。
  - CLAUDE 决策：从 PDF 读方法，无 clone 集。〔待核：各框架实现与原文式5/7 的等价性应以具体代码为准〕→ 实为**低风险待核**(多框架已收录，公认正确)。
- 💰 资源/成本与可扩展性：面向**巨型模型/MoE/长响应**的大 rollout batch 场景(§3 动机即此)。相对 GRPO 不增加算力，反而**省**(删掉 Routing Replay 的内存/通信、还可省掉 training-engine 重算 logprob)。已扩到 30B-A3B 及 Qwen3 全系。
- 🎯 对"探索-巩固"对标：**可借组件/竞品(RL 优化层)，非直接对标**。判定依据：
  - ① GSPO 让超大/MoE 模型跑 RL 不崩，是「巩固」阶段的**基础设施保障**——如果 TSRD 要在大模型上做 on-policy RL/蒸馏，GSPO 就是稳定底座。
  - ② **GSPO-token 的 stop-gradient 构造(式14)与 MEMORY 记录的「forward-hard/backward-soft 解耦」同构**，是可直接借用的工程模式——可用于「前向走硬的路径选择/序列级量、反向流平滑梯度」。
  - ③ **张力**：GSPO 序列级等权会**抹平 token 级信号**，与 path-recovery 的「单点关键 token 接管」、MTP 的「关键步前瞻」诉求方向相反——GSPO 想消除 token 异质带来的不稳定，TSRD 却想利用 token 异质。
  - 缺口：无 teacher 脚手架、无选路/回轨、无记忆。
- 🔭 开放问题/未来方向：
  - 【原文】结尾只有愿景式的「以 GSPO 为基石继续 scale RL」，没列具体 open problem。
  - 【推断】(1) 序列级 vs token 级的「稳定性 vs 细粒度信用」trade-off——holderpo/hapo 正是对此的回应；(2) GSPO-token 在真正的多轮/agent RL 上缺实证(§4.3 只提了动机)；(3) 怎么与显式 token 加权(关键步)调和而不重新引入不稳定；(4) 缺标准消融/方差，可复现性待社区补。

key|读到PDF?|L线|对标结论|残留待核数
gspo | 是(全文7页正文,全部公式抄准) | L3 | 可借RL优化层稳定底座(MoE防崩);GSPO-token的stop-gradient≡forward-hard/backward-soft可借;与token级信用(path-recovery/MTP)方向相反有张力 | 1(各框架实现等价性,低风险)

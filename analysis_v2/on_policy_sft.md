on_policy_sft | On-Policy Supervised Fine-Tuning for Efficient Reasoning | 东方理工(EIT,宁波)·香港理工·Paris Dauphine-PSL·上海交大·腾讯混元/AI Lab·LMU 慕尼黑(Anhao Zhao, Ziyang Chen, Junlong Tong, Yingqi Fan, Fanghua Ye, Shuhao Li, Yunpu Ma, Wenjie Li, Xiaoyu Shen*) | arXiv 2602.13407 v1 · Preprint Feb 17, 2026 | 主题线 L2(统一 SFT-RL/GFT 类)，兼 L1(on-policy 自蒸馏味)、L6(高效推理/token) · 相关性 高

**原始论文**：https://arxiv.org/abs/2602.13407 （代码 https://github.com/EIT-NLP/On-Policy-SFT）

## 一眼看懂
- 🟦 TL;DR：做"高效推理"(缩短 CoT 但保准确率)时，主流做法是把"长度惩罚多奖励"塞进 GRPO，结果训练不稳、精度-效率权衡次优。本文做**原理化简**:高效推理有两条 RLHF 没有的性质——正确性与长度**都直接可验证**、且本质是**多奖励**问题。据此论证 GRPO 里(1)**KL 正则冗余**(无学习型奖励模型⇒无 reward over-optimization 之忧)、(2)**组内归一化失配**(多奖励下会放大无信息样本梯度、并混淆不同奖励组成)。把这两项删掉、长度惩罚简化为**截断**(超长响应给零奖励),GRPO 目标就**退化为在自生成数据上做交叉熵 SFT**——该数据天然按"正确+简洁(未被截断)"过滤。取名 **on-policy SFT**。极简却在 5 个数学基准定义 accuracy-efficiency Pareto 前沿:CoT 缩短约 80%、精度持平,训练显存/墙钟降约 50%、收敛快约 70%。【原文 §Abstract / §3 / Fig.1 / Table 1】
- 最巧的一步：**论证三处化简后 PG 目标 ≡ 过滤式 SFT**(去 KL + 去组内归一化 + 截断奖励 → reward-free MLE,见下方 Eq.7→9 推导链)。抽掉这一步(即保留 KL/归一化/复杂长度奖励)就退回原版多奖励 GRPO 的病态——训练不稳、权衡次优。这一刀把"高效推理需要复杂 RL"祛魅为"其实是个在 on-policy 数据上的简单 SFT"。但**真正的功臣其实是 on-policy 数据本身**(§5.2 消融:同样的过滤式 SFT 目标,换成 off-policy 固定数据就大幅掉点)——所以"最巧的一步"在叙事上是化简、在机理上是"保持 on-policy 重采样"。【原文 §3 / §5.2 / Fig.6】

## 为什么做
- 研究背景：LRM 多用 GRPO 系 RL 训练(继承 DeepSeek-R1 配方,Guo 2025)并产生很长 CoT,推理时算力/显存开销大。"高效推理"方向涌现大量工作,在奖励里叠长度惩罚/简洁奖励(ThinkPrune、O1-Pruner、L1、LASER、ER-RL 等),且 \(R_{\text{Len}}(o\mid q)\) 与权重 \(\gamma(o\mid q)\) 的设计日趋复杂(按正确性/难度条件激活、用组内 mean/median/max 长度统计)。这些可统一写为多奖励形式(Eq.3,见下)。【原文 §1 / §2(Reward Shaping for Efficient Reasoning) / Appendix B】
- 解决的具体痛点：① RL-based 高效推理虽大幅缩 CoT,但常伴随随任务复杂度变化的**精度下降**,得到次优权衡;② 叠多奖励(正确+简洁)联合优化使**训练不稳、收敛慢、对超参敏感**(原文逐一引 Liu 2025a / Li 2025a / Arora&Zanette / Chen 2025b);③ 复杂奖励塑形被不加甄别地从 GRPO 沿用,与高效推理的内在结构未必对齐。【原文 §1 / §3】
- 相关工作 & 各自不足（来龙去脉 + 三条并行路线 + 各自短板 + 站谁肩上）：
  - **多奖励 GRPO 高效推理族**(O1-Pruner / L1(Aggarwal&Welleck) / ThinkPrune / LASER / ER-RL / Arora&Zanette 等)：在 GRPO 奖励里叠 \(R_{\text{Len}}\)。短板=复杂 \(R_{\text{Len}}+\gamma\) → 不稳、权衡次优、超参敏感。本文与之精确差异:**删掉 KL、删掉组内归一化、把 \(R_{\text{Len}}\) 退化为截断**,并证明该简化奖励族下 PG 目标 ≡ 过滤式 SFT。
  - **training-free 压缩**(CoD / DEER / CRST / TokenSkip / StepEntropy)：推理时直接裁(不训练)。短板=省事但压缩有限、且常牺牲精度(Table 1 显示 TokenSkip 的压缩率甚至 >100% 即更长)。
  - **SFT-based 高效推理**(在固定数据上 SFT 短 CoT)：与本文同是 reward-free SFT 目标。**最近邻**,精确差异在数据来源——它是 **off-policy 固定数据 ⇒ §5.2 证明会掉点**;本文坚持 **on-policy 重采样**。
  - **站谁肩上**：① GRPO(Shao 2024)的多奖励形式;② "优化目标设计 > 复杂奖励塑形"的近期发现(Liu 2025a「最简截断已够」、He 2025);③ on-policy 蒸馏(Lu&Lab 2025 等)启发了 Appendix K 的诊断工具。
  - **共性缺口**：没人指出"高效推理的复杂 RL 其实可化简成 on-policy 过滤式 SFT",也没人把增益正确归因到 on-policy 数据而非奖励塑形。【原文 §1 / §2 / §4(Related Work) / §5.2】
- 动机链：现状(高效推理=往 GRPO 塞复杂长度奖励)→ 质疑(把 GRPO 不加甄别套到高效推理上合理吗?)→ 原理分析(发现 KL 冗余 + 组内归一化失配两处错配)→ 化简(去两项 + 截断奖励 ⇒ PG 目标 ≡ 过滤式 SFT)→ 所以(直接做 on-policy SFT,更简单、更省、Pareto 更优)。【原文 §1 / §3】
- 与最近邻工作的 Δ：相对**多奖励 GRPO 族**——on-policy SFT 删掉 \(\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})\)、删掉组内归一化 \(\hat A_{i,t}=(R_i-\mathrm{mean})/\mathrm{std}\)、把长度奖励简化成截断,差在**用最简过滤式 SFT 替代复杂奖励塑形**,且证明二者在该简化奖励族下等价。相对 **SFT-based 高效推理**——同是 reward-free SFT 目标,差在**坚持 on-policy 重采样**(每轮用当前策略重新采),§5.2 证明这正是 SFT-baseline 掉点而 on-policy SFT 不掉点的根因。相对 **on-policy 蒸馏(opd_blog 等)**——本文**显式自我归入 on-policy 数据范式**(§5.2 / Appendix K 引 Lu & Lab 2025、Shenfeld 2026、Zhao 2026、Hübotter 2026),但**无 teacher**:它是"自己当自己的过滤器"的自蒸馏味 SFT,而非教师 KL 蒸馏。【原文 §3 / §5.2 / Appendix K】

## 怎么做 + 靠不靠谱
- 方法流水线（输入→输出,读完即可复现）：输入(当前策略 \(\pi_{\theta_{\text{old}}}\) + 数学题集 \(\mathcal D\)) → ① **Rollout**:对每题 \(q\) 采 \(G\) 条回答 \(\{o_i\}_{i=1}^G\)(温度严格 =1.0;脚本 ROLLOUT_N=32) → ② **评分+过滤**:verifier 判对错;**截断式奖励**令超 \(L\)(=max_response_length,脚本 3500)的响应得零奖励(即被当作"错误");`_data_filter` 按 uid 分组:**丢错误、留全部正确响应**;query 全错则整条丢;保留数对 dp_size 取整 → ③ **训练**:把 verl PPO actor 的 PG 损失换成交叉熵 \(-\log\pi_\theta\) 的 token-mean(无 advantage、无 clip、`use_kl_loss=False`),长度归一用 batch 内最长响应(见 length-bias correction) → ④ **同分布训练-评测**(训练与评测用同 prompt 模板,避免漂移混淆) → 迭代回 ①。【原文 §3.3 / §5.3 + v1 仓库核】
- 逐组件必要性（每模块干啥 / 怎么咬合 / 有无消融）：
  - **on-policy 重采样(真正的核心)**：每轮用当前策略重新 rollout,使训练分布 = 部署分布。没它(改 off-policy 固定数据)→ §5.2 消融显示"只温和缩短长度且精度大幅下降",而 on-policy SFT 快速缩短且保精度。**有消融(Fig.6,7B/MATH-500)**——全文最关键的对照。【原文 §5.2 / Fig.6】
  - **去 KL 正则**：在损失里删掉 \(-\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})\)(`use_kl_loss=False`)。没它→省维护 \(\pi_{\text{ref}}\) 的显存/算力 + 不过度约束更新;论证依据:高效推理无学习型奖励模型,KL 防 over-optimization 的本意失效(正确性/长度可验证,无分布漂移下奖励不可靠之忧)。【原文 §3.1(KL Divergence)】
  - **去组内归一化**：不再做 \(\hat A_{i,t}=(R_i-\mathrm{mean})/\mathrm{std}\),改用 baseline 直接优化。没它(保留)→ 多奖励下两病:(a) 组内奖励差异小(都对且都长)时分母 std 很小 → **放大无信息样本梯度**;(b) **混淆奖励组成**——原文举例:奖励向量 \((0,1)\) 与 \((0,0)\)(标量 1 与 0)归一化后得 advantage \((-0.7071, 0.7071)\),而**完全不同组成的** \((1,1)\) 与 \((0,0)\) 归一化后得**相同** advantage。【原文 §3.1(Group-wise Reward Normalization)】
  - **截断奖励(最简长度惩罚)**：\(\hat R_i=1\) 当且仅当响应在固定长度 \(L\) 内答对,否则 \(\hat R_i=0\)。据 Liu 2025a「最简截断已够」。它是"简洁性过滤"的**间接实现**——超长响应落入"错误"被 `_data_filter` 丢弃。**注意**:仓库 `_data_filter` 并不显式按 length 字段筛(length 被提取但代码注释 "not used"),简洁性靠截断奖励间接达成(v1 已核实)。【原文 §3.1(Truncation)/ v1 仓库核】
  - **length bias correction**：把梯度里的 \(1/|o_i|\) 归一项换成 \(1/\max_j|o_j|\)(batch 内最长),避免短(可能噪声)响应被赋过大梯度、把训练带偏。**有消融**(Appendix,with/without)——称其必要以稳长度控制。【原文 §3.2(Eq.9 推导)/ §5.3 / Appendix】
  - **rollout 温度 =1.0**：指南且有理论。温度 \(T\neq 1\) 会让 rollout 偏离 \(\pi_{\theta_{\text{old}}}\) 而**变成 off-policy**(Appendix J 证 Eq.42-43:仅 \(T=1\) 时温度缩放分布 \(=\pi_{\theta_{\text{old}}}\);\(T\neq1\) 即 \(\Pr_T\neq\pi_{\theta_{\text{old}}}\),Eq.9 期望变成对 \(\Pr_T\) 取);低温(0.3/0.6)过度缩短+掉点,高温(1.2/1.5)loss 单调升、不收敛。**有消融(Fig.7/Fig.13)**。【原文 §5.3 / Appendix J / Eq.42-43】
  - **rollout 数缩放 & max length**：\(T=1\) 固定下,增大 rollout 数 → 更短且保精度(Fig.8);max length 3k(最省但略掉点)/4k(最贵最准)/3.5k(平衡,匹配 4k 精度)——默认 3.5k(Fig.10)。【原文 §5.3 / Fig.8 / Fig.10】
- 关键机制/公式（真实形式 + 直觉,均从 PDF 抄准）：
  - **GRPO 组内优势(Eq.1)**:\(\displaystyle \hat A_{i,t}=\frac{R_i-\mathrm{mean}(\{R_i\}_{i=1}^G)}{\mathrm{std}(\{R_i\}_{i=1}^G)}.\) 这是被本文删掉的"组内归一化"——直觉:它为单奖励设计,多奖励下标准化丢信息。
  - **GRPO clipped 目标(Eq.2)**:\(\displaystyle J_{\text{GRPO}}(\theta)=\mathbb E_{(q,a)\sim\mathcal D,\,\{o_i\}\sim\pi_{\theta_{\text{old}}}}\!\Big[\tfrac1G\textstyle\sum_{i=1}^G\tfrac1{|o_i|}\textstyle\sum_{t=1}^{|o_i|}\min\big(r_{i,t}(\theta)\hat A_{i,t},\,\mathrm{clip}(r_{i,t}(\theta),1-\varepsilon,1+\varepsilon)\hat A_{i,t}\big)-\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})\Big],\) 其中重要性比 \(r_{i,t}(\theta)=\dfrac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q,o_{i,<t})}\)。本文删掉末项 \(-\beta\,\mathrm{KL}\)。
  - **多奖励统一形式(Eq.3)**:\(\displaystyle R_{\text{Eff}}(o\mid q)=R_{\text{Acc}}(o\mid q)+\gamma(o\mid q)\,R_{\text{Len}}(o\mid q),\) \(\gamma\) 控正确性-效率权衡。本文把 \(\gamma R_{\text{Len}}\) 退化成"超长判零分"的截断。
  - **化简到 SFT 的核心推导链(Eq.7→8→9,全文枢纽)**:先设 \(\hat R_{i,t}=R_i\)(exploitation 设定,§3.2)使 GRPO 梯度等价 REINFORCE(Appendix D);因奖励逐 token 共享,丢 \(t\) 得 \(\displaystyle \nabla_\theta J(\theta)=\mathbb E\Big[\tfrac1G\textstyle\sum_i \hat R_i\,\tfrac1{|o_i|}\textstyle\sum_t \nabla_\theta\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})\Big]\quad(\text{Eq.7});\) 代入二元截断奖励 \(\hat R_i\in\{0,1\}\)、用集合约束 \(\mathbb 1\{o_i\in C_L\}\)(\(C_L\)=对且在 \(L\) 内的响应集)得 \(\displaystyle \nabla_\theta J(\theta)=\mathbb E\Big[\tfrac1G\textstyle\sum_i \mathbb 1\{o_i\in C_L\}\,\tfrac1{|o_i|}\textstyle\sum_t \nabla_\theta\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})\Big]\quad(\text{Eq.8}),\) 此式与标准 SFT 的 MLE 梯度 \(\nabla_\theta J_{\text{SFT}}(\theta)=\mathbb E_{(q,o)\sim\mathcal D_{\text{SFT}}}\big[\tfrac1{|o|}\sum_t\nabla_\theta\log\pi_\theta(o_t\mid q,o_{<t})\big]\) **函数形式相同,唯一区别是数据分布**——\(\mathbb 1\{o_i\in C_L\}\) 充当"对+不超长"的选择器,且数据来自 on-policy 的 \(\pi_{\theta_{\text{old}}}\) 而非固定 \(\mathcal D_{\text{SFT}}\);最后把 \(1/|o_i|\) 换成 \(1/\max_j|o_j|\)(length-bias 校正)得**最终目标(Eq.9)**:\(\displaystyle J(\theta)=c_L\,\mathbb E_{o_i\sim \mathcal D_L^{+}}\Big[\tfrac1{\max_j|o_j|}\textstyle\sum_{t=1}^{|o_i|}\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})\Big],\) 其中 \(\mathcal D_L^{+}=\{o_i\mid (q,a)\sim\mathcal D,\ o_i\sim\pi_{\theta_{\text{old}}}(\cdot\mid q),\ o_i\in C_L\}\),常数 \(c_L=\mathbb E[\mathbb 1\{o_i\in C_L\}]\) 与 \(\theta\) 无关。**直觉**:删掉 KL/归一化、长度惩罚退化成"超长直接判零分"后,策略梯度目标只剩"在自己采的、且(对+不超长)的轨迹上做最大似然"——这正是 SFT。
  - **诚实归因(§5.2)**:论文明说真正 work 的不是"化简成 SFT"这个形式,而是"数据始终来自当前策略"——同样的 SFT 目标喂固定数据(off-policy)就崩。直觉:"复杂奖励是障眼法,on-policy 数据才是药效。"
  - **Appendix K 诊断工具(Eq.44)**:受 on-policy 蒸馏启发,造"低效 token 诊断"——从 base 采 CoT,把同前缀 \(x_{1:t}\) 喂给高效模型做 teacher forcing,在每位置量两者 next-token 分布的 token 级 KL \(\displaystyle D_t=\mathrm{KL}\big(p_{\text{orig}}(\cdot\mid x_{1:t})\,\big\|\,p_{\text{eff}}(\cdot\mid x_{1:t})\big)=\textstyle\sum_{v\in V}p_{\text{orig}}(v\mid x_{1:t})\big(\log p_{\text{orig}}(v\mid x_{1:t})-\log p_{\text{eff}}(v\mid x_{1:t})\big),\) \(D_t\) 大的 token 就是"把模型带向冗长推理的元凶"。Fig.14 词云显示 "Wait"/"Alternatively"/"Hmm"(犹豫/回溯)、以及第一人称 "I"/"me"(高效变体偏好复数 "we"/"us")divergence 最大。【原文 Appendix K / Eq.44 / Fig.14-18】
- 实验与证据：
  - **核心结果(Table 1 / Fig.1)**：1.5B(DeepSeek-R1-Distill-Qwen-1.5B)——Acc 59.9% / Pass@N 73.6%(略超原模型 59.0/73.5),平均生成长度 10,178→2,186 token(约 80% 缩短),Eff=2.74% > 最强 RL baseline 2.55%。7B——Eff=2.97%(最高),长度较原模型缩约 70%。不同长度预算下持续在 Pareto 前沿,优于 ThinkPrune/O1-Pruner/L1/LASER/ER-RL 等 10 个 baseline(training-free/SFT-based/RL-based 三类)。【原文 §Abstract / Table 1 / Fig.1-2】
  - **训练侧**：每步显存与墙钟约降 50%,收敛较 RL 快约 70%;长度控制更稳(多次生成长度方差更低,§5.1 用归一化标准差度量,Fig.5 vs ThinkPrune)。【原文 §Abstract / §1 / §5.1】
  - **机制消融(§5.2 / Fig.6)**：off-policy 变体(同采样/过滤、每题 8 响应、3500-token 限、匹配 50 步的总题量、固定数据训 1 epoch ×7 iter ≈350 步)只温和缩短且大幅掉点;on-policy SFT 快速收敛保精度 ⇒ **增益主因是 on-policy 数据**,与"RL 之效来自 on-policy 数据而非复杂算法"的近期发现一致(引 Chen 2025a / Shenfeld 2026 / Zhao 2026 / Hübotter 2026)。【原文 §5.2 / Fig.6】
  - baseline 公平吗：与 10 个 baseline 在同 5 基准(GSM8K/MATH-500/AMC23/AIME24/AIME25)对比,**同分布训练-评测**设计排除 prompt 不匹配混淆,较规范;Eff 综合指标定义见 Appendix。【原文 §4 / Table 1】
  - "看着强但没回答核心问题"：核心(化简成 SFT + Pareto 前沿 + on-policy 是主因)都给了证据;但 Eff 等综合指标定义对结论敏感,跨方法可比性需谨慎。
- 假设与失效边界：
  - 【原文 §3.1】"等价于 SFT"依赖一串简化前提(去 KL、去归一化、**截断式**长度奖励 + \(\hat R_{i,t}=R_i\) 的 exploitation 设定)——是"在某一特定简化奖励族下成立",非对任意长度奖励普适。
  - 【原文 §5.3 / Appendix J / Eq.42-43】温度必须 =1.0 才保持 on-policy;\(T\neq1\) 即变 off-policy,低温过度缩短掉点、高温不收敛。
  - 【推断】仅在**数学推理、可验证正确性**场景验证;对开放式/不可验证任务不适用(其"正确性直接可验证"前提失效 ⇒ KL 冗余论也失效)。依据:全部 5 基准为数学、reward 靠 verifier。
  - 【原文 §1 + 推断】"截断奖励"的 max length \(L\) 是关键超参——过紧牺牲难题精度(3k 即略掉点,Fig.10);论文承认精度随任务复杂度有边际下降风险。
  - 【推断/已核】仓库 as-shipped 与论文叙述的"简洁性过滤"实现方式有表述落差:仓库靠**截断奖励间接实现**(length 字段提取但 "not used"),而非显式 length 过滤器(读者易误以为有专门长度筛)。
- 祛魅总结【推断】：
  - 真贡献：**祛魅 + 极简 + 正确归因**三连。"两处错配"论证(KL 冗余、组内归一化在多奖励下歧义)清晰且有举例(\((0,1)/(0,0)\) vs \((1,1)/(0,0)\) 同 advantage)。把高效推理的复杂 RL 化简成过滤式 SFT(Eq.7→9)、并用 §5.2 把功劳**诚实地归给 on-policy 数据而非化简形式本身**——这种自我祛魅在 SFT-RL 统一线里很有价值。Appendix K 的"低效 token 诊断"(Eq.44)是有用的副产品。
  - 包装/被高估处："GRPO 退化为 SFT"易被读成普适结论,实则**绑定特定简化奖励族 + exploitation 设定**;"conciseness 过滤"在论文与代码间的实现落差容易误导;真正的 novelty 更多是"祛魅 + 归因",方法本身(过滤式 on-policy SFT)与 RFT / on-policy SFT 系工作高度同源(本文与 NFT/RFT/psft 等同属一族)。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：verifier 的正确性 + 截断式长度信号(超长判零);无奖励模型、无教师 logits。过滤后对保留轨迹做交叉熵(reward-free)。属"自蒸馏味的过滤式 on-policy SFT"。
  - **改什么**：只改训练目标(GRPO 的 PG loss → 交叉熵 Eq.9)+ 数据来源(坚持 on-policy 重采样);模型全量更新。
  - **何时改**：在线迭代——每轮用当前策略 rollout→过滤→交叉熵更新→进入下一轮。
  - **免梯度?**：损失侧无策略梯度/无 advantage/无 clip(就是 NLL),但仍是梯度优化;**去掉了 ref 模型**(省显存)。
  - **记忆-技能生命周期**：无外部记忆/技能库;"技能"=简洁正确的推理路径,固化进参数。无持续学习机制。
  - **防遗忘机制**：无显式防遗忘(且去掉 KL-to-ref);靠 on-policy 重采样维持分布不漂移。
- ⑦ 开源代码+框架/harness：https://github.com/EIT-NLP/On-Policy-SFT (`opsff/`→`opsft/` 子目录,已开源,本地已 clone)。**框架=veRL**(基于 verl 改造) + FSDP + vLLM rollout。recipe 在 `opsft/recipe/On_Policy_SFT/`:`dp_actor.py`(PG loss 换成交叉熵)、`on_policy_sft_trainer.py`(on-policy 流程 + `_data_filter`)、`main_on_policy_sft.py`、`fsdp_workers.py`、`config/on_policy_sft_trainer.yaml`。入口 `bash examples/On_Policy_SFT.sh`。关键超参(脚本):LR=1e-6、MAX_GEN_LENGTH=3500、ROLLOUT_N=32、temperature=1、batch=32、`use_kl_loss=False`、`loss_agg_mode=token-mean`、total_epochs=2、2×GPU。数据:内置 DeepScaleR/GSM8K/OpenThoughts3 train.parquet + 8 个评测 benchmark parquet。代码可得性高。【v1 仓库核查 + 本批 repo ls】
- 💰 资源/成本与可扩展性：核心卖点即成本——相对 RL 每步显存与墙钟降约 50%、收敛快约 70%;backbone R1-Distill-Qwen-1.5B/7B;脚本默认 2×GPU、LR=1e-6、total_epochs=2。【原文 §Abstract / §1 + v1 脚本】
- 🎯 对"探索-巩固"对标：**中-强支撑(巩固层:on-policy 数据是巩固之钥 + 低效 token 诊断接近"关键步")+ 可借诊断工具;非脚手架/蒸馏方法**。判定依据：① **巩固对标**——on-policy SFT 把"在自己采的、走通且简洁的轨迹上做 MLE(Eq.9)"作为巩固,与本课题"巩固=固化走通的有效路径"高度一致;§5.2"on-policy 数据是主因"为本课题"student on-policy 自选轨迹再固化"提供直接证据(off-policy 固定数据会掉点⇒必须 on-policy)。② **"低效 token 诊断工具"(Appendix K,Eq.44)≈关键步定位**——用 base 与高效模型在同前缀上的 next-token KL \(D_t\) 找"把模型带偏向冗长的元凶 token",这与本课题"识别关键步/path-recovery 的单点接管位置"在机制上同构(都是用分布差异定位关键 token)。③ 截断奖励=对"走偏成冗长路径"的惩罚,弱对应"巩固简洁路径、抑制冗余试探"。**可借组件**:(a) on-policy 重采样 + 正确性过滤的极简巩固管线(去 KL/去归一化省显存);(b) **Appendix K 的 token 级 KL 诊断(Eq.44)** 可直接迁移为"用 teacher 与 student 的 next-token 散度定位关键步/低效步",再对这些步做稀疏脚手架介入。**缺口**:① **无 teacher、无蒸馏**(自己当过滤器,非教师 KL);② 信号是**序列级正确性 + 长度**,过滤是粗粒度的"留对的整条",**非 token 级稀疏关键步介入**(虽然 Appendix K 有诊断,但训练 loss 仍是整条 NLL);③ **无 MTP/前瞻、无记忆/技能库**;④ 只针对"高效推理"(缩短),不针对"走通新难题"——按 opd_survey §7.1 的话,这类 correct-only on-policy 训练更像**压缩**而非**纠错**,与本课题"教 student 走通自己走不通的开头"目标不同。一句话:**它给"on-policy 自巩固"提供了最干净的证据与一个可借的关键步诊断工具,但缺 teacher 脚手架、稀疏接管与前瞻。**
- 🔭 开放问题/未来方向：
  - 【原文 §6 / §1 末】未来高效推理应重简单+原理而非堆算法复杂度;把化简思路推广到更多任务。
  - 【推断】把 Appendix K 的"token 级 KL 诊断(Eq.44)"从"找低效 token"升级为"找关键步并做 teacher 稀疏介入",把本文(自蒸馏、整条过滤)与 OPD(teacher、token 级 KL)缝合;研究截断 max length \(L\) 与任务难度的自适应,避免难题精度下滑;把"on-policy 是主因"与 MTP 前瞻结合,前瞻挑选最值得重采样的 prompt。

RETURN: on_policy_sft|读到PDF=是(§Abstract/§1-3全文+Eq.1-3/Eq.7-9化简链/Table1/Fig.1-2/§5.1-5.3+Fig.5-10/Appendix J Eq.42-43温度off-policy/Appendix K Eq.44诊断+Fig.14词云)|L线=L2(兼L1/L6)|对标=中-强支撑(on-policy数据是巩固主因的硬证据,AppdxK的token级KL诊断Eq.44≈关键步定位可借;但无teacher脚手架/粗粒度整条过滤非稀疏接管/无MTP,且correct-only更像压缩非纠错)|残留待核=0

hapo | Heterogeneous Adaptive Policy Optimization (HAPO): Tailoring Optimization to Every Token's Nature | 北京大学 / 上海 AI Lab / 北航（共一 Zheng Liu、Mengjie Liu；通讯 Wentao Zhang、Conghui He）| arXiv 2509.16591v2, 2026-04-29 · 预印本 | 主题线 L3/L6（token 异质性驱动 RLVR；token 信用）·相关性 中（token 级关键步识别与「选路/关键步」概念相关，但本身非蒸馏/无 teacher）

**原始论文**：https://arxiv.org/abs/2509.16591

## 一眼看懂
- 🟦 TL;DR：现有 RLVR(GRPO/DAPO)对所有 token **一视同仁**优化，违背语言生成的异质本质——关键推理决策 token 与常规格式 token 角色完全不同。已有用熵的方法只把熵当**离散过滤器或事后 bonus**，核心优化机制没变。HAPO 把 **token 熵升级为贯穿采样/优势/裁剪全流程的连续优化驱动量**，用四个组件对每个 token 做细粒度差异化处理，在 1.5B~14B 多模型、数学/代码/逻辑上一致优于 DAPO（7B 数学 50.04 vs DAPO 46.97）【原文 摘要, 表1, 表4】。
- 最巧的一步：**把熵归一化成一个连续调制信号 ˜h∈[−1,1]，再用「meta 方程 Parameter = Parameter_base·f(˜h)」把它注入采样温度、优势、裁剪界三处**（式5-7）——而非像前作那样只在某一处做离散分组。抽掉这个「连续注入」就退回到 DAPO+熵 bonus 的离散方案。但**承重墙其实是第三个组件 Differential Advantage Redistribution（差分优势再分配，组件C）**：它不只看熵、还**联合用重要性比 r 来判定 token 是否真有明确更新趋势**（高熵且 r 偏离中性区→放大优势；低熵且 r≈1→抑制）。这一步回应了作者最关键的实证发现——「相似熵的 token 优化需求可能截然不同，仅靠熵 reshape 优势是有缺陷的」(§3.3)——是 HAPO 区别于 Archer/Entropy-Adv/EDGE-GRPO 的核心。

## 为什么做
- 研究背景：RLHF/RLVR 是提升 LLM 推理的核心(o1/R1/Qwen3)，但主流算法对所有 token 统一优化、不区分「关键推理路径 token vs 常规模式 token」，与语言生成的异质本质冲突【原文 §1】。
- 解决的具体痛点（系统性实证，§3）：(1) **采样**——温度困境：低温保精度但压制稀有的高熵关键 token、高温增其出现却引噪声(图3);高熵关键 token 本就被系统性欠采样;(2) **优势**——序列级优势忽略序列内 token 异质;DAPO token-mean loss 下**长负样本主导梯度**(负样本显著比正样本长，图4);且发现 token 级优化需求与熵不完全对应(图5);(3) **裁剪**——低熵 token(多为格式符)易撞左界、阻止其降概率;高熵 token(关键推理)易撞右界、限制探索——统一裁剪「保护噪声、约束探索」(图6)。还发现 **「Dual-Entropy 现象」**：同词干 token 在高/低熵区有「孪生」(如 since/Since/(since))——故低熵 token 不能被一刀切丢弃(§3.1)。

- 相关工作 & 各自不足（§1 + §2，按「熵被怎么用」补全为两类）:
  1. **熵当离散过滤器**:**DAPO-with-forking-tokens**(Wang 2025b "Beyond the 80/20 rule")证明只有少数高熵 token 在引导优化、故只优化 top-ρ% 高熵 token(式4：用 \(\mathbb{1}[H_t\ge\tau_\rho]\) 硬门控);**Archer**(Wang 2025a)把 token 二分高/低熵两组、对高熵**放宽 clip 界**鼓励探索(二元离散)。短板:熵只当**离散过滤器**，核心优化机制没变;且 Archer 放宽高熵**左**界反而让高熵关键 token 在负优势主导下概率快速衰减→熵塌缩(HAPO §3.3 反证)。
  2. **熵当优势调节器/事后 bonus**:**Entropy-Adv**(Cheng 2025)把熵当 intrinsic reward 加进优势(式2：\(A^s_{i,t}=A_{i,t}+\psi(H_t)\))鼓励更长推理——但对**负优势的高熵错误 token，熵奖励抵消其负优势、削弱训练信号**(HAPO §3.3 点名的缺陷);**EDGE-GRPO**(Zhang 2025a)用**序列级**熵 reweight 优势偏向 confident 输出。短板:熵被限制为**辅助调节器/事后 bonus**，核心优化机制本身未变，且 EDGE 是序列级非 token 级。
  - 共性缺陷(§1 原话):熵"function as discrete filters" 或 "auxiliary regulator/post-hoc bonus"，"leaving the core optimization mechanics fundamentally unchanged"。HAPO 把熵嵌进采样/优势/裁剪**每一阶段**。

- 动机链：token 角色异质 → 现有统一优化不合理 → 已有用熵者只做离散过滤/事后 bonus → 把熵做成连续函数嵌入采样/优势/裁剪每一阶段 → 细粒度差异化 → 既保精度又保探索。
- 与最近邻工作的精确Δ：vs **DAPO/Archer/Forking**——从「离散二元分组」升级为「熵的连续函数调制」，且**四阶段全覆盖**(采样+优势+裁剪+优势再分配)而非单点;vs **Entropy-Adv**——不把熵当 bonus 加(避免抵消负优势)，而是**联合熵×重要性比**判定再分配(组件C);vs **EDGE-GRPO**——token 级而非序列级;vs **Archer 裁剪**——"反 Archer"：高熵升**右**界、低熵降**左**界(而非 Archer 对高熵放宽左界)。

## 怎么做 + 靠不靠谱
### 背景公式（§2，对照基线）
- **GRPO**(式1)：组内 G 条采样、序列级优势 \(A_{i,t}=\dfrac{R_i-\mathrm{mean}(\{R_j\})}{\mathrm{std}(\{R_j\})}\)，loss 按序列长度归一 `1/|o_i|`。
- **DAPO**(式3)：改 **token-mean loss**(总 token 数归一 `1/Σ|o_i|`，保长序列梯度) + **clip-higher** 非对称界 \(\epsilon_{\text{high}}>\epsilon_{\text{low}}\)(默认 0.28/0.2)。
- **重要性比**：\(r_{i,t}=\pi_\theta(a_{i,t}\mid s_{i,t})/\pi_{\text{ref}}(a_{i,t}\mid s_{i,t})\)——反映 token 当前优化状态(图5 显示高熵 token 比值分布更宽、偏离 1)。

### 熵的连续归一化（§4.1，全篇调制信号）
逐 token 熵先做对数平滑 + 减分位数 + 除 std（式5），再**非对称缩放到 [−1,1]**（式6）：
\[
h_{i,t}=\frac{\log(H_{i,t})-Q_\rho(\log H)}{\sigma(\log H)},\qquad
\tilde h_{i,t}=\begin{cases}h_{i,t}/h_{\max}, & h_{i,t}>0,\\ -h_{i,t}/|h_{\min}|, & h_{i,t}\le0,\end{cases}
\]
\(Q_\rho\)=ρ 分位(高/低熵分界，默认 ρ=80% 即关注 top 20% 高熵)。**meta 方程**(式7)：\(\text{Parameter}_{i,t}=\text{Parameter}_{\text{base}}\cdot f(\tilde h_{i,t})\)，把 ˜h 注入温度/优势/裁剪界。

### 四组件流水线（叠加在 DAPO 上，读完可复现）
1. **A — 自适应温度采样**(§4.2，式8)：rollout 逐 token 在线调温(用上一步 token 统计量算分位/方差)：
   \[
   T_{i,t}=T_{\text{base}}\cdot\Big(1+\frac{\log(H_{i,t})-\hat\rho_{\log H}}{\hat\sigma_{\log H}}\cdot\tau\Big),
   \]
   高熵升温(探索)、低熵降温(连贯)。默认 \(T_{\text{base}}=1.0,\ \tau=0.1\)。
2. **B — token 级组平均优势**(§4.3，式9)：把序列奖励先分到每个 token \(a_{i,t}=r_i\in\{0,1\}\)，再在组内**所有 token**上归一：
   \[
   A_{i,t}=\frac{a_{i,t}-\mu_{\text{tok}}}{\sigma_{\text{tok}}},\quad \mu_{\text{tok}}=\tfrac{1}{|\mathcal{T}|}\sum_{(i,t)\in\mathcal{T}}a_{i,t},\ \mathcal{T}=\{(i,t)\}.
   \]
   使 \(\sum_{(i,t)}\hat A_{i,t}=0\)，消除 token-mean loss 下长负样本梯度偏置，同时保留长序列梯度缩放(长序列贡献更大梯度)——兼得 GRPO 无偏 + DAPO token 粒度。
3. **C — 差分优势再分配**(§4.4，式10-11，承重墙)：定义中性区 \([\gamma_L,\gamma_U]\)(默认 \([1-\epsilon_L/2,\,1+\epsilon_R/2]\))，只在**有明确更新趋势**时改优势：
   \[
   \hat A_{i,t}=\begin{cases}A_{i,t}\cdot(1+\tilde h_{i,t}), & C(\tilde h_{i,t},r_{i,t})\ \text{为真},\\ A_{i,t}, & \text{否则},\end{cases}\qquad
   C(\tilde h_{i,t},r_{i,t})=\begin{cases}r_{i,t}\notin[\gamma_L,\gamma_U], & \tilde h_t>0\ (\text{高熵且比值出中性区→放大}),\\ r_{i,t}\in[\gamma_L,\gamma_U], & \tilde h_t\le0\ (\text{低熵且比值在中性区→抑制}).\end{cases}
   \]
   核心直觉：**熵告诉你 token 重不重要，重要性比 r 告诉你它当前更不更新得动/方向明不明确**——两者结合才精准分配(回应 §3.3「相似熵 token 优化需求可能截然不同」)。
4. **D — 非对称自适应裁剪**(§4.5，式12-13，"反 Archer")：
   \[
   \epsilon_L(i,t)=\begin{cases}\epsilon_L^{\text{base}}(1-\tilde h_{i,t}), & \tilde h_{i,t}\le0\ (\text{低熵→降左界,允许激进降概率压噪声}),\\ \epsilon_L^{\text{base}}, & \tilde h_{i,t}>0,\end{cases}\quad
   \epsilon_R(i,t)=\begin{cases}\epsilon_R^{\text{base}}, & \tilde h_{i,t}\le0,\\ \epsilon_R^{\text{base}}(1+\tilde h_{i,t}), & \tilde h_{i,t}>0\ (\text{高熵→升右界,关键决策点探索}).\end{cases}
   \]
   默认 \(\epsilon_L^{\text{base}}=0.2,\ \epsilon_R^{\text{base}}=0.28\)。全部是 ˜h 的平滑函数，熵在采样时已算，**几乎零额外开销**。

### 逐组件必要性（Table 4 消融，base DAPO 7B=46.97）
单开 **A 48.85 / B 48.56 / C 48.28 / D 48.02**(各 +1.0~1.9)；A+B 48.74；A+B+C 49.42；**A+B+C+D 全开 50.04**。证据扎实——每个组件单独有效、逐步叠加单调上升，无明显冗余。〔注：单组件中 A(温度) 增益最大、D(裁剪) 最小〕。

- 关键机制/公式（直觉）：核心是「熵告诉你 token **重不重要**，重要性比 r 告诉你它**当前更不更新得动/方向明不明确**」——两者结合才能精准分配优势(组件C 的洞察)。非对称裁剪是「反 Archer」：Archer 对高熵放宽左界，HAPO 发现这反而让高熵关键 token 在负优势主导下概率快速衰减→熵塌缩，故改为高熵升**右**界、低熵降**左**界。
- 实验与证据：base=Qwen2.5-Math-1.5B/7B、Qwen3-8B/14B、LLaMA3.2-3B/3.1-8B-Instruct;训练 DAPO-Math-17K，**32×A100(4 节点)**，verl+vLLM(+SGLang/mcore)，batch 512/mini 32，lr 1e-6;评测 AIME24/25、AMC、MATH500、Minerva、OlympiadBench(每题 8 样、T=0.5 平均);另在 LiveCodeBench + Logic-RL 验证泛化。关键数字：7B 数学 Avg **50.04 vs DAPO 46.97 vs Forking 47.43 vs Archer 45.63**;14B **62.09 vs DAPO 58.06**;LLaMA3.2-3B **27.42 vs DAPO 22.99**(弱模型增益更大);代码 LCB **34.02 vs Forking 31.46**、逻辑 Logic-RL **92.69 vs Forking 90.15**(均超 Forking)。训练动态：HAPO 维持**更长响应+更高熵+更高精度**(图7-10)——即「保探索的同时涨分」。
- baseline 公平吗：较公平——所有方法在**同一 DAPO 设置**(同 clip-higher 0.28/0.2、同 overlong shaping 10240 max + 4096 cache、同数据)下跑，固定超参跨所有模型(展示泛化);对比了 5 个 entropy-based 近作(GRPO/DAPO/Forking/Archer/Entropy-Adv/EDGE-GRPO)。**但**：评测 T=0.5、8 样平均(比贪心更稳)，AIME 仍仅 30 题，单基准 ±2-3 点可能差 1 题;LLaMA 上 AIME25 多为 0~8% 极低区间，增益解读需谨慎。
- 假设与失效边界：【原文】(1) ρ=80% 分位、Tbase=1.0、τ=0.1、中性区=[1−ϵL/2,1+ϵR/2] 等超参固定(附录D 有 ablation);(2) 「高熵=关键推理决策点」是贯穿全文的核心假设。【推断】(3) **「熵=重要性」是相关而非因果**——高熵也可能是模型在格式/无关处的真实不确定(作者自己的 Dual-Entropy 发现正说明熵与语义角色不一一对应)，HAPO 用 r 做二次过滤缓解但未根除;(4) 四组件叠加的超参耦合复杂(温度/中性区/双侧裁剪界)，跨任务迁移可能需重调(虽作者称固定超参，但都在数学/代码/逻辑这类可验证推理域内);(5) 升高熵 token 右界=放任其概率上涨，若该 token 实为错误方向，可能放大错误探索(原文未单独验证「被放大的高熵 token 是否真是有效关键步」);(6) 依赖 token 级熵/比值统计，多轮/agentic 长程下的稳定性未测(全在单轮推理)。
- 祛魅总结：真贡献=**(a) 系统性实证把「token 异质」在采样/优势/裁剪三处的具体病灶讲透(Dual-Entropy、负样本长度偏置、反 Archer 的裁剪洞察)+ (b) 用熵的连续函数(式5-7) + 熵×重要性比联合判据(组件C)，把异质处理嵌入全流程 + (c) 完整四组件消融 + 多规模/多域验证**——工程扎实、消融完整、增益稳定(尤其弱模型)。需打折的：(1) 仍是 **DAPO 之上的精细调参**，不引入新信号/能力，本质是「把 token 权重/温度/裁剪按熵重新分配」;(2) 「熵=关键推理 token」是相关性假设，缺「被放大的高熵 token 确为有效关键步」的因果验证;(3) 增益虽稳但多在 +2~3 点量级，四组件超参耦合是落地复杂度成本。【推断】对「探索-巩固」这是**「关键步识别」侧的相关工作**，但用熵(无 teacher/无前瞻)定位、且无巩固/记忆机制。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=组内相对优势(可验证奖励) + **token 熵**(贯穿采样/优势/裁剪) + **重要性比 r**(组件C 的二次判据)｜**改什么**=策略参数 θ + 采样温度(逐 token)｜**何时改**=on-policy RL 每步(熵在采样时算)｜**免梯度?**=否，policy gradient(DAPO 系)｜**记忆-技能生命周期**=无外部记忆/技能库｜**防遗忘机制**=无显式;但「保高熵关键 token 的探索/防熵塌缩」间接维持能力多样性(图7-10 更高熵更长响应)。
- ⑦ 开源代码+框架/harness：https://github.com/starriver030515/HAPO （已 clone ~8.6MB，README 已核）。框架=**verl + vLLM(+ SGLang/mcore)**，把 token 级策略**叠加进 DAPO**实现;含 recipe/、scripts/install_vllm_sglang_mcore.sh、HF 模型集合。〔注：README 里 arXiv badge 链接误填占位 1234.12345，实际为 2509.16591〕
- 💰 资源/成本与可扩展性：**32×A100(4 节点)**训练;max 10240 token + 4096 cache，batch 512/mini 32。**相对 DAPO 几乎零额外计算开销**(熵采样时已算)。规模验证 1.5B~14B + LLaMA，泛化到代码/逻辑。
- 🎯 对"探索-巩固"对标：**「探索/选路」侧相关、可借组件；「巩固」侧无贡献**。判定依据：① **支撑「探索/选路」**——HAPO 的核心命题「区分关键推理路径 token 与常规 token、在关键决策点鼓励探索」与本课题「探索=发现有效行为/路径」「关键步」直接同频;其「高熵=关键决策点、稀有、易被欠采样/误裁剪」的实证，为 TSRD「在关键步/路径分叉处做文章」提供了 token 级证据基础;② **可借组件**——(a) 「熵×重要性比联合判据」(组件C,式10-11)可借来在 OPD/蒸馏中**定位真正需要 teacher 接管的关键 token**(而非全 token 对齐);(b) 「非对称自适应裁剪」(式12-13)「token 级组平均优势消长度偏置」(式9)是即插即用的稳定性工具;(c) 自适应温度采样(式8)可用于「探索期在关键步升温找替代路径」。**竞品/张力 & 缺口**：(a) HAPO 用**熵(模型自身不确定)** 定位关键步，TSRD 用 **teacher 脚手架 + MTP 前瞻** 定位——前者是模型内省信号、后者是外部监督/前瞻信号，**定位机制不同且 HAPO 缺因果验证**(熵≠真关键步，Dual-Entropy 自证);(b) HAPO **完全无「巩固/回轨」机制**——它不固化探索成果进记忆/技能、不处理「走偏后恢复」，纯属 RL 优化层的 token 加权。一句判定：**HAPO 为「关键步存在且应差异化对待」提供了强 token 级实证与可借的熵×比值定位工具，但其定位靠熵(非 teacher/前瞻、缺因果)、且无任何巩固机制，只能作为 TSRD「探索/选路」侧的优化层补充与对照，不触及「巩固」**。
- 🔭 开放问题/未来方向：【原文】未设独立 open problem 章；隐含方向=超参自适应(附录D ablation)、更多域泛化(已做代码/逻辑)。【推断】(1) 「被 HAPO 放大的高熵 token 是否真是有效关键推理步」需因果/可解释验证——直接决定它能否当「选路」定位器用;(2) 把熵定位换成/融合 **teacher 信号或 MTP 前瞻**——在 teacher 标记的关键步处升温+升右界探索、在恢复分支处做差分优势放大，可把 HAPO 的「自适应」升级为 TSRD 式定向脚手架;(3) 增加「巩固」——把探索到的高价值轨迹/关键步固化进记忆或参数;(4) 多轮/agentic 长程下的 token 异质处理(当前仅单轮推理);(5) 与 holderpo 的 p<0「巩固替代路径」对照——两者都想善待「非主流/犹豫」token，可融合。

key|读到PDF?|L线|对标结论|残留待核数
hapo | 是(PyMuPDF全文24页;Eq.1-13全抄准) | L3/L6 | 「探索/选路」侧强相关:为关键步存在性提供token级实证,熵×重要性比定位(式10-11)+非对称裁剪(式12-13)可借;但靠熵定位(非teacher/前瞻、缺因果)、零巩固机制,仅作TSRD探索侧补充与对照 | 2(熵放大的是否真为有效关键步;AIME小样本统计稳健性)

cepo | CEPO: RLVR Self-Distillation using Contrastive Evidence Policy Optimization | MBZUAI + Linköping + ANU（Ahmed Heakl、Salman Khan 等） | 2026-05-19 · arXiv preprint v1 · cs.LG | 主题线 L3（RLVR/GRPO·token 信用）兼 L1（特权自蒸馏）·相关性 高

**原始论文**：https://arxiv.org/abs/2605.19436

## 一眼看懂
- 🟦 TL;DR：GRPO 给一条正确轨迹里**所有 token 同一个正优势**(错误轨迹同一负信号),把梯度浪费在 filler(连接词/格式/样板)上、低估真正决定成败的少数推理步【原文 §1 L76-87】。CEPO 在每个 token 问一个更尖锐的问题:不只"正确答案偏好它吗",而是"正确答案偏好它**且**错误答案反对它吗"——满足两者=真推理步,皆不满足=filler【原文 Abstract L21-26】。用对比比率 P⁺_T(y_t)/P⁻_T(y_t)(正确/错误两个 teacher,错误答案直接取组内已有的最低分 rejected rollout)在 stop-gradient 下调制 GRPO 优势的**幅度**,符号仍锚定 verifier【原文 §3.4 式(5)(7)】。
- 最巧的一步：**把分母从 P_S(学生先验)换成 P⁻_T(错误 teacher)**。这一换,学生先验 P_S 在 stop-gradient 下被完全消掉,RLSD 的"fluency confound"(分母反映 base-rate 流畅度而非语义相关,常见 token 无论多被 r⁺ 偏好都被压低)从构造上消失【原文 §3.4 L399-401】。抽掉这步就退回 RLSD,而 RLSD 无法区分"正反答案都同等支持的 filler"与"r⁺ 支持/r⁻ 反对的决定性步"(两者 P⁺_T/P_S ratio 相同时被同等加权)【原文 §3.3 L345-349】。论文自证 P⁻_T(y_t)=P_S(y_t) 时 CEPO 精确退回 RLSD【§3.4 Thm 1(iii)】。

## 为什么做
- 研究背景：RLVR 采 rollout、verifier 打分、更新策略;GRPO 去掉 value 网络、组内归一化得序列级优势,但代价是"too blunt"——整条轨迹均匀 credit【原文 §1 L70-77】。
- 解决的具体痛点：①GRPO 的 credit assignment 问题("哪些 token 真重要"完全未解);②"用 r⁺ 当 teacher 做分布匹配"会**信息泄漏**——RLSD 证明任何把 P⁺_T 当分布目标的散度目标,其梯度都含 vocab-wide 的 r-条件求和(式3),方差 ∝ I(Y_t;R⁺|X),训练后期主导、逼模型编码 x→r⁺ 伪相关,且不可约【原文 §1 L94-99, §3.2 式(3)】。
- 相关工作 & 各自不足【原文 §2, Table 1】：token 级方法要么靠 Monte Carlo 重模拟(VinePPO、SPO)、要么训练独立 PRM——都需昂贵重采样或辅助网络;特权自蒸馏 OPSD(per-token KL 到 P⁺_T)、SDPO(扩成 JSD + EMA teacher)、HDPO(专用于全错 prompt)——都泄漏(Priv ✓ 但 Leak-free ✗);RLSD 是直接前身——首个 Priv ✓ + Leak-free ✓,但信号质量有三缺陷(见下)。Table 1 用 Priv/Leak-free/Contr/No-Aux 四维定位 CEPO 是唯一四项全 ✓。
- 动机链：GRPO 钝→想用 r⁺ 当 dense teacher→但分布匹配会泄漏→RLSD 用 evidence ratio + stop-gradient 解泄漏→但 RLSD 信号有三缺陷[(1)fluency confound:分母 P_S 是流畅度非相关性;(2)asymmetric negative:对错误轨迹的惩罚缺乏对 r⁻ 的显式 grounding;(3)one-sided evidence:P⁺_T/P_S 区分不了 filler 与决定性步]→所以必须引入对比双参照 P⁺_T/P⁻_T。为什么不用更简单做法(对比 KL):脚注证明 ∇[D_KL(P⁺‖P_S)−D_KL(P⁻‖P_S)] 会产生与 OPSD 同构的 vocab-wide leakage,所以必须用 ratio + stop-gradient 而非两个 KL 相减【原文 §4 脚注1 L639-645】。
- 与最近邻工作的Δ：vs RLSD(最近邻)——唯一区别是把单参照升级为对比双参照,分母 P_S→P⁻_T;**关键有用点**:P⁻_T 打破 RLSD 的"平局"——错误答案主动反对的 token(P⁻_T<P_S)获得更小分母→更高 credit,恰在语义决定性位置;filler 处 P⁻_T≈P⁺_T≈P_S→改进消失(Proposition 1)【原文 §3.4 L596-604】。

## 怎么做 + 靠不靠谱
- 方法流水线【原文 §3.4, Algorithm 1 L352-383】：①采 G 条 rollout,按式(1)算组归一化序列优势 A,分正确组 G⁺/错误组 G⁻→②r⁻ ← argmin_{j∈G⁻} R 的最终答案(组内最低 reward rollout 的答案,无额外采样);G⁻=∅ 则 P⁻_T←P_S→③对每条轨迹每位置 t:∆^CE_t = sg(log P⁺_T(y_t) − log P⁻_T(y_t)),w^CE_t = exp(sign(A)·∆^CE_t) = (P⁺_T/P⁻_T)^{sign(A)},Â_t = A·[(1−λ) + λ·clip(w^CE_t, 1−ε_w, 1+ε_w)]→④用 Â_t 代入标准 PPO clipped surrogate 更新;λ 从 λ₀ 线性退火到 0。
- 逐组件必要性：
  - **对比分母 P⁻_T**：核心组件,消 fluency confound;Table 5 feedback source 消融——"GT r⁺ + peer answer-only r⁻"最优(43.43),用整条 peer rollout 当负参照次之(42.74),只 prefix/suffix 截断反而 <GRPO(40.47/40.60)【原文 Table 5】。
  - **stop-gradient**：保证 leakage-free(Thm 1(ii))+ 方向锚定(Thm 1(i)),无它就泄漏。
  - **λ 退火**：Figure 3——λ=0.5 与 25 步退火都 >GRPO,λ=1.0(恒定最大)反而引入噪声变差(尽管积分 CEPO 压力最大);10 步快退也接近 25 步,说明增益"front-loaded"(前 10-25 步贡献主要改进)【原文 §5.1 L799-804】。
  - **ε_w(evidence clip)**：Figure 3(c)——[0.4,0.5] 峰值(42.7),ε_w=0.1 退回 GRPO、ε_w≥0.8 失稳【原文 §5.1 L796-799】。
  - **teacher source**：Table 4——actor-policy teacher 最优(43.43)>每 25 步同步(42.74)>fixed reference(42.18);teacher 新鲜度/on-policy 对齐比拉大师生分布差更重要,且 actor 共享权重省显存【原文 §5.1 L719-736】。
- 关键机制/公式(直觉)：对比证据 delta 的贝叶斯解释(对两个 teacher 应用 RLSD Thm 4 相减,P_S 抵消):∆^CE_t = [token y_t 把 r⁺ 后验抬高多少] − [把 r⁻ 后验压低多少]【原文 §3.4 式(6)】。直觉:决定性步同时"支持正确答案 + 反对错误答案"→大正值;filler 对两个答案都中性→≈0。
- 实验与证据【原文 §4-5】：
  - **数据集/设置**:训练 Geo3K(3000 几何题,可验证数值答案);评测 5 个 held-out 多模态数学基准 DynaMath/LogicVista/MathVision-mini/MMMU/WeMath;模型 Qwen3-VL-2B/4B-Instruct,LoRA(rank16/α32),仅 50 步;CEPO 用 lr 5e-6(其余 1e-6),λ₀=0.5、25 步退火、ε_w=0.5【原文 §4 L619-632, Table 6/7】。
  - **主结果(Table 2)**:2B 平均 CEPO 43.43 vs GRPO 41.17(+2.26pp)、base 39.73、OPSD 34.96、SDPO 35.70;4B CEPO 60.56 vs GRPO 57.43(+3.13pp)、base 58.36。增益最大 LogicVista(4B +6.18 over GRPO)、MathVision-mini(2B +4.94),最小 MMMU(短链知识检索,2B +1.67)。
  - **最有力实证**:**OPSD/SDPO 跌破未训练 base**(2B 34.96/35.70 vs 39.73;4B 同样 56.23/55.75 vs 58.36)——直接验证 information leakage 理论预测,且不随规模消失【原文 §5 L692-699】。
  - **机理(Figure 4/5)**:训练中正 delta 占比上升、负 delta 占比下降;token 热图显示 CEPO 把 credit 锐化到关键代数推导(x+4=3x−6)与最终答案、把 filler 压到近 1,clip 率更低(49.5% vs RLSD 71.3%,即更宽有效动态范围)【原文 §5.2 L857-877】。
  - **baseline 公平吗**:同 LoRA rank、group size、训练步数,公平【原文 §4 Baselines L633-638】。Table 7 列出各方法精确超参,透明。
  - **看着强但没回答核心问题**：增益绝对值偏小(base→CEPO 仅 +2.2~3.7pp),只跑 50 步、单一数据集、小模型(2/4B)、LoRA——Figure 1 显示 CEPO 早期更快、约 step 40 差距最大、最终部分收敛;"长训是否保持优势"未答。
- 假设与失效边界：
  - 【原文】依赖组内同时有正确与错误 rollout 构造 r⁻;全对/全错 prompt 上对比信号退化(全错时退回 RLSD)【§3.4 Algorithm L364】。
  - 【原文】Prop.1 给出 CEPO > RLSD 的充要条件 P⁻_T(y_t)<P_S(y_t)——只在"错误答案相对学生先验更不喜欢该 token"时锐化;否则两者相同【§3.4 L580-595】。
  - 【推断】只在 Qwen3-VL + Geo3k + 50步 + LoRA 验证;纯文本/代码/大模型未知(作者自承 future work);更像"低预算快速收敛优势"而非已证的稳态优势。
  - 【推断】r⁻ 只取"最低 reward rollout 的最终答案",对负参照选择策略敏感(Table 5 prefix/suffix 反而有害)。
- 祛魅总结【推断】：真贡献是**理论 + 实证的闭环**——三条结构保证(方向锚定/leakage-free/RLSD 包含)+ Prop.1(锐度充要条件)+ token 热图 + OPSD/SDPO 退化反例,自洽性强。最有说服力的是"OPSD/SDPO 跌破 base"这个实证反例,把"结构安全是实践前提"从理论变成可观测现象。被高估的是泛化主张(规模/数据/任务都很窄)和绝对增益(+2.2~3.7pp 且部分收敛);被低估的可能是"对比双参照"作为通用 credit 工具的潜力(目前只在多模态数学验证)。本质是 RLSD 的一个干净增量,而非全新范式。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：可验证二元奖励(rule-based 数值 verifier)给序列优势 A;**特权信息**(正确答案 r⁺ + 组内最低分错误答案 r⁻)经两个 teacher forward 转成 token 级对比 delta,只调制优势幅度、不改符号。
  - **改什么**：参数(LoRA adapter 权重,经 PPO clipped surrogate)+ **logits 层的 credit**(token 级优势 Â_t 重加权)。
  - **何时改**：在线 per-step;λ 随步数退火。
  - **免梯度?**：否(policy-gradient RL);但 teacher 信号 stop-gradient,只当标量进入。
  - **记忆-技能生命周期**：无外部记忆/技能库;r⁻ 是 per-batch 即时从组内 rejected rollout 取(写入→用完即弃,无检索/遗忘/共享)。
  - **防遗忘机制**：无显式防遗忘;但 leakage-free 设计本身防"编码 x→r⁺ 伪相关"这种泛化退化。
- ⑦ 开源代码+框架/harness：https://github.com/ahmedheakl/CEPO (既有 analysis 记录已 clone 约 16MB,含复现脚本 `geo/cepo.sh` 等)。框架 **veRL,经由 EasyR1**(Zheng 2025,基于 verl 的多模态 RLVR 框架)+ FSDP + vLLM 加速 rollout【原文 §4 L621, Table 6, ref [28]】。
- 💰 资源/成本与可扩展性：CEPO 比 GRPO 多两次 teacher forward(r⁺/r⁻ 各一次),**无额外采样**;Geo3k 50 步 wall-clock GRPO 5h58m→CEPO 6h34m(多 ~36 分钟)【原文 §4 Table 3 L607-618, L631-632】。硬件 NVIDIA RTX6000 Pro Blackwell 100GB。actor-policy teacher 共享权重→无独立参数副本、省显存【原文 §5.1 L730】。
- 🎯 对"探索-巩固"对标：**可借组件 + 部分竞品**。CEPO 直击"token 级信用分配"——与本项目 L6(Token 信用/前瞻)和"path-selection 偏向自己能走通的开头"在精神上相关:它识别"哪些 token 真正决定成败"。**关键 Δ**:CEPO 是 RLVR 内的 credit 调制(改优势幅度),不是 logit-OPD,也无 path-recovery 的"走偏后接管"——它是"事后给 token 打分",非"过程中纠偏"。**最可借的是 leakage-free 的 stop-gradient 套路**(特权信息只作 stop-gradient 标量进入,不污染梯度方向)——与本项目 MEMORY 记录的"forward-hard/backward-soft 解耦"同精神,是把"特权/前瞻信号"安全注入而不破坏 on-policy 性的现成模板。一句判定:机制层高度相关(token 信用 + 特权信号安全注入),但属 RLVR 调制而非蒸馏/记忆,是"可借的技术零件",非整体竞品。
- 🔭 开放问题/未来方向：
  - 【原文 §6 L890-892】扩展到更大模型、纯文本推理、代码生成。
  - 【推断】全对/全错 prompt 上对比信号退化——可探索"无负参照时的替代信号"(当前粗暴退回 RLSD);负参照选择策略(目前固定取最低分答案)可学习化;长训稳态优势需验证(50 步太短);对比双参照能否迁移到非数学/多轮 agent 的 token 信用(与本项目 L4/L6 交叉)。

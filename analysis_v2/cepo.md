cepo | CEPO: RLVR Self-Distillation using Contrastive Evidence Policy Optimization | MBZUAI + Linköping + ANU（Ahmed Heakl、Abdelrahman Shaker、Fahad Shahbaz Khan、Salman Khan 等） | 2026-05-19 · arXiv preprint v1 · cs.LG | 主题线 L3（RLVR/GRPO·token 信用）兼 L1（特权自蒸馏）·相关性 高

**原始论文**：https://arxiv.org/abs/2605.19436

## 一眼看懂
- 🟦 TL;DR：GRPO 给一条正确轨迹里**所有 token 同一个正优势**（错误轨迹同一负信号），把梯度浪费在 filler（连接词/格式/样板）上、低估真正决定成败的少数推理步——"The credit assignment problem, which tokens actually mattered?, is left entirely unresolved"【原文 §1 L75-87】。CEPO 在每个 token 问一个更尖锐的问题：不只"正确答案偏好它吗"，而是"正确答案偏好它**且**错误答案反对它吗"——满足两者=真推理步，皆不满足=filler【原文 Abstract L21-26】。用对比比率 \(P^+_T(y_t)/P^-_T(y_t)\)（正确/错误两个 teacher，错误答案直接取组内已有的**最低分 rejected rollout**）在 stop-gradient 下调制 GRPO 优势的**幅度**，符号仍锚定 verifier【原文 §3.4 式(5)(7)】。
- 最巧的一步：**把分母从 \(P_S\)（学生先验）换成 \(P^-_T\)（错误 teacher）**。这一换，学生先验 \(P_S\) 在 stop-gradient 下被**完全消掉**（"The student prior \(P_S\) cancels entirely"），RLSD 的"fluency confound"（分母反映 base-rate 流畅度而非语义相关，常见 token 无论多被 \(r^+\) 偏好都被压低）从构造上消失【原文 §3.4 L399-401】。抽掉这步就退回 RLSD，而 RLSD 无法区分"正反答案都同等支持的 filler"与"\(r^+\) 支持/\(r^-\) 反对的决定性步"（两者 \(P^+_T/P_S\) ratio 相同时被同等加权）【原文 §3.3 L345-349】。论文自证：当 \(P^-_T(y_t)=P_S(y_t)\) 时 CEPO 精确退回 RLSD【§3.4 Thm 1(iii)】。

## 为什么做
- 研究背景：RLVR 采 rollout、verifier 打分、更新策略；GRPO（Shao/DeepSeekMath [17]）去掉 value 网络、组内归一化得序列级优势，DAPO（[22]）改进探索稳定性，但整族方法都"assign uniform sequence-level advantages"——整条轨迹均匀 credit【原文 §1 L70-77, §2 L217-228】。
- 解决的具体痛点：①GRPO 的 credit assignment 问题（"哪些 token 真重要"完全未解）；②"用 \(r^+\) 当 teacher 做分布匹配"会**信息泄漏**——RLSD（[21]）证明任何把 \(P^+_T\) 当分布目标的散度目标，其梯度都含 vocab-wide 的 \(r\)-条件求和（式3），harmful deviation \(\delta(\theta;r^+)\) 的方差 \(\propto I(Y_t;R^+\mid X)\)，训练后期主导、逼模型编码 \(x\to r^+\) 伪相关，且**不可约**（irreducible）【原文 §1 L94-99, §3.2 L308-310】。
- 相关工作 & 各自不足【原文 §2, Table 1（四维 Priv./Leak-free/Contr./No-Aux.）】：
  - **token 级但无特权信息**：VinePPO（[7]）/SPO（[5]）靠 **Monte Carlo 重模拟**估 token 价值；PRM（[9,16]）训**独立 process reward 网络**——都需昂贵重采样或辅助网络（Table 1 上块，Priv.✗）。
  - **特权自蒸馏但泄漏**：OPSD（[26]，per-token KL 到 \(P^+_T\)）、SDPO（[6]，扩成 JSD + EMA teacher 稳定）、HDPO（[3]，专用于全错 prompt）——Table 1 标 Priv.✓ 但 **Leak-free✗**，§5 实证它们跌破 base 来确认泄漏。
  - **对比方向但离线**：cDPO（[2]，DPO 族）用对比估计找 critical token，但"operates offline on fixed response pairs under a sequence-level implicit reward rather than within the RLVR loop"【§2 L249-252】。
  - **直接前身 RLSD（[21]）**：首个 Priv.✓ + Leak-free✓（停在采样 token + stop-gradient + 用 evidence ratio 只调幅度）；但信号质量有**三缺陷**（fluency confound / asymmetric negative / one-sided evidence，见下）。Table 1 中 CEPO 是唯一四项全 ✓。
- 动机链：GRPO 钝→想用 \(r^+\) 当 dense teacher→但分布匹配会泄漏→RLSD 用 evidence ratio + stop-gradient 解泄漏→但 RLSD 信号有三缺陷：
  - (1) **fluency confound**：分母 \(P_S(y_t)\) 是 base-rate 流畅度非语义相关，常见 token 无论 \(r^+\) 多偏好都被压低；
  - (2) **asymmetric negative**：对错误轨迹，权重 \(P_S/P^+_T\) 惩罚的是"\(r^+\) 本会支持的 token"，间接、缺乏对 \(r^-\) 预测的显式 grounding；
  - (3) **one-sided evidence**：\(P^+_T/P_S\) 区分不了 filler（正反答案都支持）与决定性步（\(r^+\) 支持/\(r^-\) 反对），两者 ratio 相同时被同等加权【§3.3 L339-349】。
  - →所以必须引入对比双参照 \(P^+_T/P^-_T\)。**为什么不用更简单的"对比 KL"**：脚注1 证明 \(\nabla_\theta[D_{KL}(P^+_T\Vert P_S)-D_{KL}(P^-_T\Vert P_S)]\) 会产生 vocab-wide 求和 \(-\sum_v[P^+_T(v)-P^-_T(v)]\nabla_\theta\log P_S(v)\)，与 OPSD 同构的 leakage 缺陷——所以必须用 ratio + stop-gradient 而非两个 KL 相减【原文 §4 脚注1 L639-645】。
- 与最近邻工作的Δ：vs RLSD（最近邻）——唯一区别是把单参照升级为对比双参照，分母 \(P_S\to P^-_T\)；**关键有用点**：\(P^-_T\) 打破 RLSD 的"平局"——错误答案主动反对的 token（\(P^-_T<P_S\)）获得更小分母→更高 credit，恰在语义决定性位置；filler 处 \(P^-_T\approx P^+_T\approx P_S\)→改进消失（Proposition 1）【原文 §3.4 L596-604】。

## 怎么做 + 靠不靠谱
- 方法流水线（读完可复现）【原文 §3.4, Algorithm 1 L352-383, Fig.2】：
  1. **三个共享参数 \(\theta\) 但不同上下文的下一-token 分布**（式2）：学生 \(P_S(y_t)\triangleq\pi_\theta(y_t\mid x,y_{<t})\)；正 teacher \(P^+_T(y_t)\triangleq\pi_\theta(y_t\mid x,r^+,y_{<t})\)（额外条件在正确答案 \(r^+\)）；负 teacher \(P^-_T(y_t)\triangleq\pi_\theta(y_t\mid x,r^-,y_{<t})\)（条件在错误答案 \(r^-\)）。三者**同一份权重**，仅 prompt 不同。
  2. **采 G 条 rollout，算组归一化序列优势**（式1）：\(A^{(i)}=\dfrac{R(x,y^{(i)})-\mu_G}{\sigma_G}\)，按 verifier 把组分成正确组 \(G^+\)、错误组 \(G^-\)。
  3. **构造 \(r^-\)**：\(r^-\leftarrow\text{answer}\bigl(\arg\min_{j\in G^-}R(y^{(j)})\bigr)\)，即**组内最低 reward rollout 的最终答案**（只取答案，不取整条轨迹）；若 \(G^-=\emptyset\) 则令 \(P^-_T\leftarrow P_S\)（退回 RLSD）【Alg.1 L4】。**无额外采样开销**——\(r^-\) 来自已有 rejected rollout。
  4. **逐 token 算对比证据 delta + 调制优势**：对每条轨迹每位置 \(t\)，先算 stop-gradient 的对比 delta（式5），再得 token 级调制优势（式7），代入标准 PPO clipped surrogate 更新 \(\theta\)。
  5. **\(\lambda\) 退火**：从 \(\lambda_0\) 线性降到 0，跨 \(T_{\text{warm}}\) 步（默认 \(\lambda_0=0.5\)、\(T_{\text{warm}}=25\)）。
- 关键公式（真实形式 + 直觉）：
  - **OPSD/SDPO 的泄漏梯度**（式3，问题之源）：
    \(\displaystyle \nabla_\theta\mathcal{L}_{\text{OPSD}}=-\sum_{v\in V}P^+_T(v\mid r^+)\,\nabla_\theta\log P_S(v)\)
    这个 vocab-wide 求和把 \(r^+\) 直接编进每个梯度方向【§3.2 L302-307】。
  - **RLSD 的修复**（式4）：\(w^{\text{RLSD}}_t=\exp\bigl(\text{sign}(A)\cdot\text{sg}(\log P^+_T(y_t)-\log P_S(y_t))\bigr)\)，\(\hat A^{(i)}_t=A^{(i)}\cdot[(1-\lambda)+\lambda\cdot\text{clip}(w^{\text{RLSD}}_t,1-\epsilon_w,1+\epsilon_w)]\)【§3.2 L317-333】。
  - **CEPO 对比证据 delta**（式5，核心）：
    \(\displaystyle \Delta^{CE}_t=\text{sg}\!\left(\log\frac{P^+_T(y_t)}{P^-_T(y_t)}\right)\)
  - **贝叶斯解释**（式6，对两个 teacher 用 RLSD Thm 4 相减、\(P_S\) 抵消）：
    \(\displaystyle \Delta^{CE}_t=\underbrace{\log\frac{P(r^+\mid x,y_{\le t})}{P(r^+\mid x,y_{<t})}}_{\text{belief update for }r^+}\;-\;\underbrace{\log\frac{P(r^-\mid x,y_{\le t})}{P(r^-\mid x,y_{<t})}}_{\text{belief update for }r^-}\)
    直觉：\(\Delta^{CE}_t\) = "token \(y_t\) 把 \(r^+\) 后验抬高多少" − "把 \(r^-\) 后验抬高多少"；决定性步同时支持正确答案 + 反对错误答案→大正值，filler 对两个答案都中性→\(\approx0\)【§3.4 L413-439】。
  - **对比权重 + 调制优势**（式7）：
    \(\displaystyle w^{CE}_t=\exp\bigl(\text{sign}(A)\cdot\Delta^{CE}_t\bigr)=\left(\frac{P^+_T(y_t)}{P^-_T(y_t)}\right)^{\text{sign}(A)},\qquad \hat A^{(i)}_t=A^{(i)}\cdot\bigl[(1-\lambda)+\lambda\cdot\text{clip}(w^{CE}_t,1-\epsilon_w,1+\epsilon_w)\bigr]\)
    注意符号由 \(A\) 锚定（verifier），对比比率只改**幅度**；\(\lambda\) 控对比信号占比，clip 把权重限在 \([1-\epsilon_w,1+\epsilon_w]\)【§3.4 L440-471】。
  - **理论保证**（Theorem 1，证明在附录 A）：(i) **方向锚定**——\(\text{sign}(\hat A_t)=\text{sign}(A)\) 对所有 \(t\)（因 \(w^{CE}_t>0\) 且凸组合保正），特权信息**不能翻转**任何 token 的更新方向；(ii) **leakage-free 梯度**——\(\nabla_\theta\mathcal{L}_{\text{CEPO}}\) 不含 vocab-wide \(r\)-条件求和，\(r^+/r^-\) 只作 stop-gradient 标量在采样 token 进入；(iii) **RLSD 包含**——令 \(P^-_T=P_S\) 精确退回 RLSD【§3.4 L472-481】。
  - **判别锐度**（Proposition 1）：对正确轨迹，\(w^{CE}_t>w^{\text{RLSD}}_t\) **当且仅当** \(P^-_T(y_t)<P_S(y_t)\)（错误答案相对学生先验更不喜欢该 token）；错误轨迹对称（\(P^-_T(y_t)>P_S(y_t)\)）；filler 处 \(P^-_T\approx P^+_T\approx P_S\)→\(w^{CE}_t\approx w^{\text{RLSD}}_t\approx1\)，CEPO 不在无信息位置引入噪声——"filler-token neutrality is therefore not a limitation but a correctness criterion"【§3.4 L580-604】。
- 逐组件必要性：
  - **对比分母 \(P^-_T\)**：核心组件，消 fluency confound；**Table 5 feedback source 消融**——"GT \(r^+\) + peer answer-only \(r^-\)"最优（43.43, +2.26）；"GT \(r^+\) + peer full rollout \(r^-\)"次之（42.74）；两侧都用 full peer rollout（41.99）；只 prefix/suffix 截断反而 <GRPO（40.47/40.60）——截断的推理轨迹给噪声对比信号【§5.1 L737-794】。
  - **stop-gradient**：保证 leakage-free（Thm 1(ii)）+ 方向锚定（Thm 1(i)），无它就泄漏。
  - **\(\lambda\) 退火**：**Figure 3(a)(b)**——常值 \(\lambda=0.5\) 峰值 41.40% >GRPO；\(\lambda=1.0\)（恒定最大、积分压力 50 units vs 25）反而引入噪声变差；线性退火 25 步（从 \(\lambda_0=1.0\)）匹配常值峰值（41.25%），10 步快退接近 25 步——增益"front-loaded"，前 10-25 步贡献主要改进【§5.1 L799-804】。
  - **\(\epsilon_w\)（evidence clip）**：**Figure 3(c)**——\([0.4,0.5]\) 峰值（42.7%，+1.5pp），\(\epsilon_w=0.1\) 太紧退回 GRPO、\(\epsilon_w\ge0.8\) 不约束权重→失稳；推荐 \(\epsilon_w=0.5\)【§5.1 L796-799】。
  - **teacher source**：**Table 4**——**actor-policy teacher 最优**（43.43）>每 25 步同步（42.74）>fixed reference（42.18）；结论：teacher **新鲜度/on-policy 对齐**比"拉大师生分布差"更重要，且 actor 共享权重→无独立参数副本、省显存【§5.1 L719-736】。
- 实验与证据【原文 §4-5】：
  - **数据集/设置**：训练 **Geo3K**（3000 几何题，可验证数值答案）；评测 5 个 held-out 多模态数学基准 DynaMath/LogicVista/MathVision-mini/MMMU/WeMath；模型 **Qwen3-VL-2B/4B-Instruct**，**LoRA（rank 16/α 32, dropout 0）**，仅 **50 步**；AdamW，lr \(1\times10^{-6}\)（CEPO \(5\times10^{-6}\)），cosine decay + 5 步线性 warmup，batch 32，group \(G=8\)，max len 2048，**PPO clip 高 \(\epsilon_{\text{high}}=0.28\)/低 \(\epsilon_{\text{low}}=0.20\)**，无 KL、无 entropy 正则，rule-based 数值 verifier；CEPO 用 \(\lambda_0=0.5\)、25 步退火、\(\epsilon_w=0.5\)【原文 §4 L619-632, Table 6/7】。评测用 lmms-eval，温度 1.0/top-p 1.0/top-k 40/presence penalty 2.0/max 32000 token。
  - **主结果（Table 2）**：2B 平均 CEPO **43.43** vs GRPO 41.17（+2.26pp）、base 39.73、RLSD 40.05、OPSD 34.96、SDPO 35.70；4B CEPO **60.56** vs GRPO 57.43（+3.13pp）、base 58.36、RLSD 58.51。增益最大 LogicVista（4B +6.18 over GRPO）、MathVision-mini（2B +4.94），最小 MMMU（短链知识检索/多选，2B +1.67）——印证"推理链短时对比信号杠杆小"。
  - **最有力实证**：**OPSD/SDPO 跌破未训练 base**（2B 34.96/35.70 vs 39.73；4B 56.23/55.75 vs 58.36，五基准中四个跌破）——直接验证 information leakage 理论预测，且不随规模消失，"structural safety is a practical prerequisite, not a theoretical nicety"【原文 §5 L692-699】。
  - **机理（Figure 4/5）**：训练中正 delta 占比上升、负 delta 占比下降（CEPO 越来越能识别支持正确推理的 token）；token 热图显示 CEPO 把 credit 锐化到关键代数推导（\(x+4=3x-6\)、isolation steps、最终答案）与定位错误的 angle-equality 推断，把 filler 压到近 1，**clip 率更低（49.5% vs RLSD 71.3%）**——更宽有效动态范围，与 Prop.1 一致【§5.2 L835-877】。
  - **baseline 公平吗**：同 LoRA rank、group size、训练步数（Table 7 列出各方法精确超参，含 OPSD \(D_{KL}\)/SDPO \(D_{JS}\)+EMA \(\beta=0.999\)），透明公平【原文 §4 Baselines L633-638】。
  - **看着强但没回答核心问题**：增益绝对值偏小（base→CEPO 仅 +2.2~3.7pp），只跑 **50 步**、单一数据集（Geo3K）、小模型（2/4B）、LoRA——Figure 1 显示 CEPO 早期更快、约 step 40 差距最大、最终**部分收敛**；"长训是否保持优势"未答。
- 假设与失效边界：
  - 【原文】依赖组内同时有正确与错误 rollout 构造 \(r^-\)；全对/全错 prompt 上对比信号退化（\(G^-=\emptyset\) 时退回 RLSD）【§3.4 Alg.1 L364】。
  - 【原文】Prop.1 给出 CEPO > RLSD 的充要条件 \(P^-_T(y_t)<P_S(y_t)\)——只在"错误答案相对学生先验更不喜欢该 token"时锐化；否则两者相同【§3.4 L580-595】。
  - 【推断】只在 Qwen3-VL + Geo3K + 50 步 + LoRA 验证；纯文本/代码/大模型未知（作者自承 future work）；更像"低预算快速收敛优势"而非已证的稳态优势。
  - 【推断】\(r^-\) 只取"最低 reward rollout 的最终答案"，对负参照选择策略敏感（Table 5 prefix/suffix 反而有害）。
- 祛魅总结【推断】：真贡献是**理论 + 实证的闭环**——三条结构保证（方向锚定/leakage-free/RLSD 包含）+ Prop.1（锐度充要条件）+ token 热图 + OPSD/SDPO 退化反例，自洽性强。最有说服力的是"OPSD/SDPO 跌破 base"这个实证反例，把"结构安全是实践前提"从理论变成可观测现象。被高估的是泛化主张（规模/数据/任务都很窄）和绝对增益（+2.2~3.7pp 且部分收敛）；被低估的可能是"对比双参照"作为通用 credit 工具的潜力（目前只在多模态数学验证）。本质是 RLSD 的一个干净增量，而非全新范式。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：可验证二元奖励（rule-based 数值 verifier）给序列优势 \(A\)；**特权信息**（正确答案 \(r^+\) + 组内最低分错误答案 \(r^-\)）经两个 teacher forward 转成 token 级对比 delta \(\Delta^{CE}_t\)，只调制优势幅度、不改符号。
  - **改什么**：参数（LoRA adapter 权重，经 PPO clipped surrogate）+ **logits 层的 credit**（token 级优势 \(\hat A_t\) 重加权）。
  - **何时改**：在线 per-step；\(\lambda\) 随步数线性退火（25 步）。
  - **免梯度?**：否（policy-gradient RL）；但 teacher 信号 stop-gradient，只当标量进入。
  - **记忆-技能生命周期**：无外部记忆/技能库；\(r^-\) 是 per-batch 即时从组内 rejected rollout 取（写入→用完即弃，无检索/遗忘/共享）。
  - **防遗忘机制**：无显式防遗忘；但 leakage-free 设计本身防"编码 \(x\to r^+\) 伪相关"这种泛化退化。
- ⑦ 开源代码+框架/harness：https://github.com/ahmedheakl/CEPO （既有记录已 clone 约 16MB，含复现脚本 `geo/cepo.sh` 等）。框架 **EasyR1（Zheng 2025 [28]，基于 veRL 的多模态 RLVR 框架）+ FSDP（[27]）+ vLLM（[8]）加速 rollout**【原文 §4 L621, Table 6】。teacher 与 actor 共享权重（仅换 prompt 加 \(r^+/r^-\)），实现上是同一模型三次 forward。
- 💰 资源/成本与可扩展性：CEPO 比 GRPO 多**两次 teacher forward**（\(r^+/r^-\) 各一次），**无额外采样**；Geo3K 50 步 wall-clock GRPO 5h58m → SDPO 6h14m → RLSD 6h15m → **CEPO 6h34m**（多 ~36 分钟，与 RLSD/SDPO 同量级）【原文 Table 3 L607-618, L631-632】。硬件 **NVIDIA RTX6000 Pro Blackwell 100GB**。actor-policy teacher 共享权重→无独立参数副本、省显存【§5.1 L730】。
- 🎯 对"探索-巩固"对标：**可借组件 + 部分竞品**。CEPO 直击"token 级信用分配"——与本项目 L6（Token 信用/前瞻）和"path-selection 偏向自己能走通的开头"在精神上相关：它识别"哪些 token 真正决定成败"。**关键 Δ**：CEPO 是 RLVR 内的 credit 调制（改优势幅度），不是 logit-OPD，也无 path-recovery 的"走偏后接管"——它是"事后给 token 打分"，非"过程中纠偏"。**最可借的是 leakage-free 的 stop-gradient 套路**（特权信息只作 stop-gradient 标量进入，不污染梯度方向）——与本项目 MEMORY 记录的"forward-hard/backward-soft 解耦"同精神，是把"特权/前瞻信号"安全注入而不破坏 on-policy 性的现成模板。另一可借点：**对比双参照**（"\(r^+\) 支持且 \(r^-\) 反对"才算决定性步）可迁移为"用一个负锚点过滤伪关键步"的通用判据。一句判定：机制层高度相关（token 信用 + 特权信号安全注入），但属 RLVR 调制而非蒸馏/记忆，是"可借的技术零件"，非整体竞品。
- 🔭 开放问题/未来方向：
  - 【原文 §6 L890-892】扩展到更大模型、纯文本推理、代码生成。
  - 【推断】全对/全错 prompt 上对比信号退化（当前粗暴退回 RLSD）——可探索"无负参照时的替代信号"；负参照选择策略（目前固定取最低分答案）可学习化；长训稳态优势需验证（50 步太短，且已观察到 step 40 后部分收敛）；对比双参照能否迁移到非数学/多轮 agent 的 token 信用（与本项目 L4/L6 交叉）。

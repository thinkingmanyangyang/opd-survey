avsd | AVSD: Adaptive-View Self-Distillation by Balancing Consensus and Teacher-Specific Privileged Signals | UNC Chapel Hill / Capital One / UT Austin（Duy Nguyen 等;Elias Stengel-Eskin、Mohit Bansal 谱系）| 2026-05（arXiv:2605.20643v1，Preprint，cs.LG）| 主题线 L1 OPD/自蒸馏·相关性 高

**原始论文**：https://arxiv.org/abs/2605.20643

## 一眼看懂
- 🟦 TL;DR：自蒸馏让**同一个模型**当 student 又当 teacher，teacher 偷看"特权信息"（完整解 / 部分推理 / 只给答案），在 student 自采轨迹上给 dense 的 token 级信号。问题是没有哪种特权视图永远最好（§1 Fig.1：Qwen3-4B 上不同数据集最优视图不同），且单视图会逼 student 编码它推理时根本看不到的关联（信息不对称→不可消除互信息 gap，引 Yang 2026）。AVSD 不挑也不固定某个视图,而是**同时上 M=3 个视图**，把它们诱导的 teacher 信号拆成"跨视图共识"（geometric，可靠方向）和"单视图残差"（arithmetic−geometric，可能有用也可能是特权 artifact），用一个**门控**只在"各视图方向一致 + 残差幅度不喧宾夺主"时才把残差加进去。
- 最巧的一步：**门控 \(\lambda_t(v)=C_t(v)\cdot R_t(v)\)（§2.3 Eq.2）以及它的有界性 \(\lambda_t(v)J_t(v)\le |A^G_t(v)|\)（Appendix B.3）**。抽掉它，方法要么退化成纯几何共识（太保守，§4.1 消融平均输 1.4%），要么退化成纯算术边际（被单个视图带跑、符号翻转，输 1.7%）。正是"共识定方向、残差只能调幅不能翻向"这个数学保证让它两头都不踩雷。

## 为什么做
- 研究背景：可验证任务（数学/代码）后训练主流是 RLVR（GRPO/DeepSeekMath，§1 引 Shao 2024、DeepSeek-AI 2025），但奖励**稀疏且只在 outcome 层**，失败轨迹几乎没学习信号、采样贵（§1 引 Tao 2025 "Hybrid Reinforcement"）。蒸馏（Hinton 2015）给 dense token 级信号，但**标准离线蒸馏有 train-test mismatch**——student 在自己分布外的轨迹上训（引 Xu 2024、Agarwal 2024）。**on-policy 蒸馏（OPD，Agarwal 2024 / MiniLLM Gu 2024）**在 student 自采 rollout 上用 teacher 概率给局部信号缓解了这点，但仍**依赖一个更强的外部 teacher**。**自蒸馏（self-distillation）**去掉外部 teacher，让同模型在特权信息（解/示范/反馈/答案）下当 teacher（引 Zhao 2026 OPSD、Shenfeld 2026 SDFT）。
- 解决的具体痛点：现有自蒸馏**一旦选定特权视图就全程固定**（§1 强假设，引 Shenfeld 2026 / Penaloza 2026），但 (1) **最优视图因任务而异、无先验判据**——给完整解 CoT 可能把 teacher 锁进与 student 不合的思路（"lock the teacher into a preexisting thought pattern"），只给答案更灵活但信息少（引 Penaloza 2026：不同特权形式在信息密度/任务效用/诱导分布漂移上各异）；(2) **信息不对称导致不可消除的互信息 gap**（引 Yang 2026），student 被迫去编码 test 时看不到的 view-specific 关联;(3) 最有信息量的特权形式反而**丢掉帮模型识别推理错误的信号**、伤 OOD 数学推理（引 Kim 2026 why_sd_degrade）。
- 相关工作 & 各自不足（§5 两块）：
  - **① Learning from Privileged Information（特权信息学习，Lopez-Paz 2016 奠基）**：RLVR 给稀疏可验证奖励;自举/自改进造监督（STaR-Zelikman 2022、Self-Instruct-Wang 2023）;用特权信息改**探索/课程**——POPE（Qu 2026）用 oracle 解前缀做 on-policy exploration、off-policy guidance（Yan 2026）、把特权轨迹转 abstract hint（Chen 2026a / Liao 2026 self-hinting / Wang 2026 skill-SD）、合成课程（Sundaram 2026 / Cog-drift Chen 2026b）。**短板**：它们只关心"怎么拿到更好轨迹/hint/课程"，**不解决"多视图同时可用时怎么构造 token 级信号"**。
  - **② On-Policy Self-Distillation & Teacher Aggregation**：OPSD（Zhao 2026 条件化数学参考解）、SDFT（Shenfeld 2026 蒸示范行为、还能 continual learning）、On-policy context distillation（Ye 2026）、RLSD（Hübotter 2026 / Song 2026 / He 2026 用 feedback/成功 rollout/reward 条件自改作特权上下文）;hybrid RL-distillation（HDPO-Ding 2026、sample routing-Li 2026a、KDRL-Xu 2025);ensemble 蒸馏平均 logits/votes。**短板**：**不刻画 teacher 信号里哪部分跨视图稳定、哪部分是特权专有**;ensemble 只做朴素平均无方向-幅度分解。
- 动机链：现状（自蒸馏靠特权信息当 dense 信号）→ 缺陷（单视图既不对称又难选）→ 中心问题（§1 原文："can we leverage multiple privileged views simultaneously to construct better token-level on-policy learning signals than any single privileged teacher?"）→ "同时用多视图 + 自适应组合"。为什么不用更简单做法？**直接算术平均**会被单个强 promote 的视图带翻方向（Fig.2 Token 1，符号翻错）;**直接取几何交集**又太保守丢互补信息（Fig.2 Token 2，consensus 太保守）——都不行，必须**分解 + 门控**。
- 与最近邻工作的Δ：
  - vs OPSD/SDFT（**单视图**自蒸馏）——AVSD 把"选哪个视图"变成"组合所有视图并按几何结构自适应加权";OPSD baseline 在本文里就是"同样的 reverse-KL 目标但只条件化单一视图"（数学=full solution、代码=reference implementation）。
  - vs **Yang 2026（也处理信息不对称）**——Yang 靠**可验证二值奖励锚定更新方向**、自蒸馏只调幅度;AVSD **不依赖二值奖励**（只在可验证设定才有），而是**从 reverse-KL 学习信号本身的几何**推出聚合结构，无奖励时也能用。关键点：用"跨视图一致性"代替"外部奖励"判断哪个更新可信。

## 怎么做 + 靠不靠谱

> 一句话设定（§2.1）：训练集 \((x,r)\sim\mathcal D\)，\(x\) 是 prompt、\(r\) 是训练时特权信息;student 只见 \(x\)，teacher 见 \(r\)。在线采 \(y=(y_1,\dots,y_T)\sim P^S_\theta(\cdot\mid x)\)，前缀记 \(h_t=(x,y_{<t})\)。

**【数据流 / 逐 step 流水线（§2 + Algorithm 1）——读完可复现】**

1. **采 on-policy rollout**：student 当前策略 \(P^S_\theta\) 对 prompt \(x\) 生成一条轨迹 \(y\)（温度 0.7，每 prompt 只采 1 条——见超参表）。**只此一处顺序生成，是主成本**。
2. **构造 M 个特权视图**：\(r^{(m)}=T_m(r)\)，变换 \(T_m\) 保留任务相关信息、只改"暴露给 teacher 的视图"。数学三视图 = full solution / partial solution（保留前若干步推理）/ final answer;代码三视图 = reference implementation / algorithmic hint / execution feedback（跑 student rollout 拿 pass/fail）。\(M=3\) 经验最佳（Algorithm 1 注）。
3. **逐 view 算 teacher 分布与 advantage**：对同一前缀 \(h_t\)，让**同一个模型**条件化在 \((h_t,r^{(m)})\) 上做一次 **prefill-only 前向**（可 batch），得 \(q^{(m)}_t(v):=\mathrm{sg}\big[P^T_\theta(Y_t=v\mid h_t,r^{(m)})\big]\)（\(\mathrm{sg}\)=停梯度）。per-view distillation advantage：
   \(\displaystyle \Delta^{(m)}_t(v):=\log q^{(m)}_t(v)-\log p_t(v),\qquad p_t(v):=P^S_\theta(Y_t=v\mid h_t).\)
   直觉:这是 **reverse-KL 的 token 级 advantage**——\(\Delta>0\) 表示该 view 的 teacher 给 token \(v\) 比 student 更高概率（应 promote），\(<0\) 则 suppress。
4. **两个池化目标 + 残差（§2.2）**：
   - **几何共识（intersection support，强调"所有视图都顶"的 token）**：未归一分数 \(\tilde q^G_t(v)=\big(\prod_m q^{(m)}_t(v)\big)^{1/M}=\exp(\frac1M\sum_m\log q^{(m)}_t(v))\)。对应 advantage 是 per-view advantage 的**算术均值**（归一化只差一个 token 无关常数,故可直接用未归一分数）：
     \(\displaystyle A^G_t(v):=\log\tilde q^G_t(v)-\log p_t(v)=\tfrac1M\textstyle\sum_{m=1}^M\Delta^{(m)}_t(v).\)
     几何均值会**下压任何一个视图给低概率的 token**，故保守。理论上 \(q^G_t\) 是**最小化"到各 teacher 的平均 reverse-KL"** \(\arg\min_q \frac1M\sum_m D_{KL}(q\|q_m)\) 的解（Appendix B.2，log-linear / product-of-experts pool）。
   - **算术边际（union support，强调"至少一个视图强顶"的 token）**：\(q^A_t(v)=\frac1M\sum_m q^{(m)}_t(v)\)，advantage \(A^A_t(v)=\log q^A_t(v)-\log p_t(v)\)。它是**最小化平均 forward-KL** \(\arg\min_q\frac1M\sum_m D_{KL}(q_m\|q)\) 的解（=随机均匀抽视图后的边际分布,Appendix B.2，linear pool）。permissive，但**对 view-specific artifact 敏感**（某视图因有参考解强 promote 某 token,另一视图毫无支持）。
   - **跨视图残差**：由 AM-GM 不等式 \(q^A_t\ge\tilde q^G_t\) 逐 token 成立，故
     \(\displaystyle A^A_t(v)=A^G_t(v)+J_t(v),\qquad J_t(v):=\log q^A_t(v)-\log\tilde q^G_t(v)\ \ge 0.\)
     \(J_t\) 量化"算术边际里有、但严格几何共识里没有"的那部分概率质量。\(J_t=0\) 表示所有 teacher 精确一致;\(J_t\) 大表示只有部分 teacher 支持（可能是有用互补,也可能是特权 artifact——单看残差分不清,所以**用共识定方向、残差只调幅度**）。
5. **门控重构（§2.3，核心）**：重构 advantage \(\hat A_t(v):=A^G_t(v)+\lambda_t(v)J_t(v)\)，\(\lambda_t(v)\in[0,1]\)。门控两分量：
   - **对齐分量**（views 方向冲突时压制残差）：
     \(\displaystyle C_t(v):=\frac{|A^G_t(v)|}{\frac1M\sum_{m}|\Delta^{(m)}_t(v)|+\epsilon}=\frac{\frac1M\big|\sum_m\Delta^{(m)}_t(v)\big|}{\frac1M\sum_m|\Delta^{(m)}_t(v)|+\epsilon}\in[0,1].\)
     各 view advantage **同号或互不矛盾时 \(C_t\to1\)**（分子≈分母）;正负互相抵消时 \(C_t\to0\)。即"平均后的净幅度 / 各自幅度之和"——区分"互补支持"与"冲突驱动的支持"。
   - **幅度分量**（残差盖过共识时压制）：
     \(\displaystyle R_t(v)=\frac{|A^G_t(v)|}{|A^G_t(v)|+J_t(v)+\epsilon}.\)
     \(J_t\gg|A^G_t|\) 时 \(R_t\to0\)，防残差反转共识方向（例如 \(A^G_t<0\) 但 \(J_t\) 巨大会让算术目标去 promote 一个均值上该被压的 token）。
   - 合成 \(\lambda_t(v)=C_t(v)R_t(v)\)（**Eq.2**）。**有界性证明（Appendix B.3）**：因 \(C_t\le1\) 且 \(\frac{J_t}{|A^G_t|+J_t+\epsilon}\le1\)，得
     \(\displaystyle 0\le \lambda_t(v)J_t(v)=C_t(v)\frac{|A^G_t(v)|\,J_t(v)}{|A^G_t(v)|+J_t(v)+\epsilon}\le |A^G_t(v)|.\)
     于是 \(\hat A_t\) **保号**:\(A^G_t>0\Rightarrow\hat A_t\ge0\)，\(A^G_t<0\Rightarrow\hat A_t\le0\)——残差**只能加强正共识或软化负共识，永远不能翻转方向**。这正是它"经验性抑制 Yang 2026 所指特权泄漏风险"的机制。
6. **重构目标 + 训练损失（§2.3 Eq.3）**：把几何共识目标按门控残差重加权再归一：
   \(\displaystyle q^\star_t(v)=\frac{\tilde q^G_t(v)\exp(\lambda_t(v)J_t(v))}{\sum_{u\in V}\tilde q^G_t(u)\exp(\lambda_t(u)J_t(u))}.\)
   \(\lambda_t\equiv0\) 退几何共识目标，\(\lambda_t\equiv1\) 退算术边际目标——逐 token 在二者间插值。最终在 \(q^\star_t\) 上跑**标准 per-token reverse-KL**:
   \(\displaystyle \mathcal L_{\text{AVSD}}(\theta)=\mathbb E_{(x,r)}\,\mathbb E_{y\sim P^S_\theta(\cdot\mid x)}\Big[\textstyle\sum_{t=1}^{|y|}D_{KL}\big(p_\theta(\cdot\mid h_t)\,\|\,\mathrm{sg}[q^\star_t(\cdot)]\big)\Big].\)
   梯度**只流过 student** \(p_\theta\);用 log-derivative 形式,采样 token 的更新就是 \(\hat A_t(y_t)\)（天然 on-policy）。
- **逐组件必要性（消融真值）**：
  - **几何共识 \(A^G\)（定方向）**——去掉就成纯算术边际,§4.1 Fig.3 平均**输 1.7%**（AIME25 增益有限）。
  - **算术残差 \(J\)（补互补信息）**——去掉就成纯共识,平均**输 1.4%**（HMMT25 上甚至不如单视图 OPSD,原文："slightly worse than OPSD on HMMT25"）。
  - **对齐分量 \(C_t\)**——没它则 views 冲突时残差仍乱加、符号翻转（Fig.2 Token 1 演示:full 给 +0.65、partial/answer 给 −0.45/−0.42,算术目标会把负共识翻成 +0.07 正号）。
  - **幅度分量 \(R_t\)**——没它则大残差能反转共识方向（Appendix B.3 证明 \(\lambda J\le|A^G|\) 正是靠它）。
  - 【待核/可补】原文**只整体消融了 consensus-only / arithmetic-only,未单独拆 \(C_t\) vs \(R_t\) 各自贡献**——属可补充项。但 §D Table 5 给了 gate-open 率（Qwen3-4B 24.6%、8B 21.1%、DS-7B 26.6%,稳定 21–27%），证明门控既不塌成纯共识也不塌成纯算术。
- **关键机制/公式（直觉浓缩）**：核心是"**reverse-KL 学习信号本身的几何**"——几何均值=最小化平均 reverse-KL=log-linear pool;算术均值=最小化平均 forward-KL=linear pool（Abbas 2009 的 opinion pooling 对应）。门控精髓：**残差只能在共识已定的方向上加力或减力**,把"哪个更新可信"从依赖外部奖励变成依赖"跨视图一致性"。非均匀权重也支持（Appendix B.2:给每 view 一个前缀级可靠性权重 \(w^{(m)}_t\) 即得加权 KL barycenter,主实验用均匀权重保持 parameter-free）。
- **实验与证据（§3-4）**：3 backbone（Qwen3-4B/8B、DeepSeek-R1-Distill-Qwen-7B）× 数学（训 OpenThoughts 数学子集,三视图 full/partial/answer;评 AIME24/25、HMMT25）+ 代码（训 Codeforces Python 5K,三视图 ref/hint/feedback;评留出 100 Codeforces 题 + LiveCodeBench v6）。指标 **Avg@8**。
  - **主表 Table 1**：Qwen3-4B 数学较 base +5.5、较最强基线 OPSD +2.2;Qwen3-8B 较最强基线 GRPO +3.1;DeepSeek-7B 较自蒸馏基线 +2.3;Qwen3-8B 代码较 base +4.9、较 OPSD +2.4。**全部 backbone 数学/代码平均均最佳**。
  - **token 级机制证据（§4.2 Fig.4/8）**：错误 rollout 上看 top-20 高影响 token 的 credit 符号——GRPO 在 **29.6%** 例子无学习信号（group 全错时 outcome reward 给不出信号）、OPSD 给 **12.3%** 高影响 token 错误正号、**AVSD 降到 2.9%**——直接证明信号更可靠。Fig.7 给了具体例子（盐水浓度题,full/partial/answer 三视图符号不一致,AVSD 输出 coherent 负 credit）。
  - **视图数 scaling（§4.3 Fig.5/6）**：数学 1→3 视图 AIME24 +2.9 / AIME25 +2.8,**第 4 视图（model attempt + reference）只 +0.7/+0.2**（递减）;代码 1→3 视图 Codeforces +2.0,第 4 视图仍有 +3.3/+0.5（代码递减更慢）。
  - **成本（Table 2）**：训练时间 41.5s→46.7s/step,约 **1.13× 开销**（因 rollout 顺序生成才是主成本,多视图 teacher 评估是 prefill-only 可 batch,不增 rollout）。
  - baseline 公平:SFT/GRPO/OPSD **同数据、同 reverse-KL 目标**,仅视图数不同。
- **假设与失效边界**：【原文】(§A Limitations) 只验证数学/代码,**tool-use agent 等更广设定未验**;依赖**构造视图的质量 + 底层数据质量**（数学三视图靠规则切分,噪声标注解会给不可靠 teacher 信号）;**仅实例化于 reverse-KL**——换 forward-KL/JS 时重构目标可用,但"advantage 解释与有界性保证需进一步研究"（Appendix B.5,因 forward-KL 期望在 \(q^\star\) 下、需全词表监督、丢失 sampled-token advantage 的 on-policy 性质）;训练时**禁用 Qwen3 thinking mode**（§C,因长 exploratory CoT 难学,留 open problem）。【推断】增益 2–3% 属中等而非压倒性,且高度集中于竞赛数学;门控两条件（同号 + 成比例）是**手工启发式**,缺最优性论证;\(M\) 增大收益递减。
- **祛魅总结**：【推断】真贡献是"**用跨视图一致性这一免奖励信号,把'哪个更新可信'量化成方向-幅度分解 + 有界门控**"——干净、可迁移,且 token 级错误正号率 12.3%→2.9% 是有说服力的机制证据。被适度包装的是:它声称缓解 Yang 2026 的"互信息 gap",但本质各视图 teacher 仍条件化于特权信息,门控只是**经验性抑制**危险 token,**并未从原理上消除** student 编码不可见关联的压力（§2.3 自己也只说 "addresses this risk by preventing residuals from freely determining direction"）;增益幅度多处用"相对最强基线 +2~3%"乐观呈现。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=token 级 reverse-KL advantage（跨 3 视图的几何共识方向 \(A^G\) + 门控残差幅度 \(\lambda J\)）｜**改什么**=student 参数（LoRA r=64/α=128,target 全 attn+MLP proj,logits 分布对齐）｜**何时改**=在线 per-step（on-policy rollout + 即时多视图 teacher 评估）｜**免梯度?**=否（标准 AdamW 梯度下降,reverse-KL 蒸馏）｜**记忆-技能生命周期**=不适用（无外部记忆/技能库）｜**防遗忘机制**=不适用（单任务后训练,未涉持续学习;但 SDFT 谱系 Shenfeld 2026 本身做 continual learning）
- ⑦ 开源代码+框架/harness：https://github.com/duykhuongnguyen/AVSD （旧 analysis 记已 clone 到 resource/repos/avsd 约 1.9M）。框架栈=**DeepSpeed ZeRO-2 + HF Accelerate + PEFT(LoRA) + vLLM**，trainer 沿用 OPSD（siyan-zhao/OPSD）→ 底层 **TRL**。【待核】具体依赖版本（trl 0.26 等）来自旧 analysis 对仓库的核查,本次基于 PDF 重写未重新进仓核对。核心实现:多视图 teacher 评估 + 门控重构目标 \(q^\star_t\)（对应 Algorithm 1 步 8-15）。**关键超参（Table 3,本次从 PDF 抄准）**:lr \(5\times10^{-6}\)、effective batch 16、grad clip 0.1、训练步 OPSD/AVSD 200（GRPO 500）、AVSD rollout 数=1/温度 0.7、评测温度 0.6/top-p 0.95/top-k 20/max 38912 tokens;多数实验 4×A6000,Qwen3-8B 与 GRPO baseline 用 4×A100。
- 💰 资源/成本与可扩展性：【原文】AVSD 相比单视图只多"对同一前缀做 M 个视图的 teacher 前向"（prefill-only,可 batch）,不需额外 rollout;Table 2:41.5s→46.7s/step,约 **1.13× 开销**。rollout（顺序生成）才是主成本。
- 🎯 对"探索-巩固"对标：**支撑 + 可借组件**。判定:AVSD 的"几何共识=可靠方向、门控残差=谨慎补充"与 TSRD"teacher 当稀疏脚手架教选路+回轨"同构——共识方向≈student 能走通的稳妥开头（探索/选路）,残差≈互补但有风险的分支（需谨慎采纳）。**可直接借的组件**:① 方向-幅度分解 + 有界门控（\(\lambda J\le|A^G|\),保证"脚手架只调力度不夺方向"）可移植到 path-recovery 的"单点接管"——让 teacher 提示只能微调而非覆盖 student 自选恢复分支;② 多视图里"partial solution"视图天然就是"给前缀/脚手架";③ 对齐分量 \(C_t\) 的"净幅度/绝对幅度之和"可作"该步是否值得 teacher 介入"的判据。**缺口**:无 MTP/前瞻;门控是 token 级即时计算,非 step/turn 级;不涉及把学到的东西固化进记忆/技能。
- 🔭 开放问题/未来方向：【原文】扩到 tool-use agent;换其他蒸馏损失（forward-KL/JS）时重建目标的理论保证（Appendix B.5）;视图质量/数据质量依赖;Qwen3 thinking mode 下的长 CoT 难学问题。【推断】门控启发式的最优性/可学习化（如把 \(w^{(m)}_t\) 学出来）;视图数 \(M\) 与质量的系统扫描;把"跨视图一致性"思想用到 multi-turn/agentic 的 turn 级信用分配;与 MTP 前瞻结合——把"未来 token 一致性"也纳入共识判据。

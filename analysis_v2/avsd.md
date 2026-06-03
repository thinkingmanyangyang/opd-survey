avsd | AVSD: Adaptive-View Self-Distillation by Balancing Consensus and Teacher-Specific Privileged Signals | UNC Chapel Hill / Capital One / UT Austin（Mohit Bansal、Elias Stengel-Eskin 谱系）| 2026-05（arXiv:2605.20643v1，Preprint，cs.LG）| 主题线 L1 OPD/自蒸馏·相关性 高

**原始论文**：https://arxiv.org/abs/2605.20643

## 一眼看懂
- 🟦 TL;DR：自蒸馏让同一个模型当 student 又当 teacher，teacher 偷看"特权信息"（完整解/部分推理/只给答案）。问题是没有哪种特权视图永远最好（§1 Fig.1：不同数据集最优视图不同），且单视图会逼 student 编码它推理时根本看不到的关联。AVSD 不挑也不固定某个视图，而是同时上 M=3 个视图，把它们诱导的 teacher 信号拆成"跨视图共识"（可靠方向）和"单视图残差"（可能有用也可能是特权 artifact），用一个门控只在"各视图方向一致 + 残差幅度不喧宾夺主"时才把残差加进去。
- 最巧的一步：**门控 λ_t=C_t·R_t（§2.3 Eq.2）以及它的有界性 λ_t·J_t ≤ |A^G_t|（Appendix B.3）**。抽掉它，方法要么退化成纯几何共识（太保守，消融里输 1.4%），要么退化成纯算术边际（被单个视图带跑、符号翻转，输 1.7%）。正是"共识定方向、残差只能调幅不能翻向"这个数学保证让它两头都不踩雷。

## 为什么做
- 研究背景：可验证任务（数学/代码）后训练主流是 RLVR（GRPO），但奖励**稀疏且只在 outcome 层**，失败轨迹几乎没学习信号、采样贵（§1，引 Tao 2025）。蒸馏给 dense 的 token 级信号，但离线蒸馏有 train-test mismatch；on-policy 蒸馏（OPD）在 student 自采轨迹上训练缓解了这点，但仍**依赖一个更强的外部 teacher**。自蒸馏（self-distillation）去掉外部 teacher，让同模型在特权信息（解/示范/反馈/答案）下当 teacher。
- 解决的具体痛点：现有自蒸馏**一旦选定特权视图就全程固定**（§1，引 Shenfeld 2026 / Penaloza 2026），但 (1) 信息不对称导致**不可消除的互信息 gap**（引 Yang 2026），student 被迫去编码 test 时看不到的 view-specific 关联；(2) **最优视图因任务而异、无先验判据**（引 Penaloza 2026 / Kim 2026），给完整解可能把 teacher 锁进与 student 不合的思路，只给答案更灵活但信息少。
- 相关工作 & 各自不足：① RLVR——奖励稀疏；② 用特权信息做更好轨迹/hint/课程（POPE 用 oracle 解前缀、off-policy guidance、把特权轨迹转 hint、合成课程）——**只关心怎么拿到更好轨迹，不解决"多视图同时可用时怎么构造 token 级信号"**（§5）；③ 自蒸馏/teacher 聚合（OPSD-Zhao 2026、SDFT-Shenfeld 2026、RLSD-Hübotter 2026）——**不刻画 teacher 信号里哪部分跨视图稳定、哪部分是特权专有**；④ ensemble 蒸馏只平均 logits/votes，无方向-幅度分解。
- 动机链：现状（自蒸馏靠特权信息当 dense 信号）→ 缺陷（单视图既不对称又难选）→ 所以要"同时用多视图 + 自适应组合"。为什么不用更简单做法？直接平均（算术）会被单个强 promote 的视图带翻方向；直接取交集（几何）又太保守丢互补信息——都不行，必须分解+门控。
- 与最近邻工作的Δ：vs OPSD/SDFT（单视图自蒸馏）——AVSD 把"选哪个视图"变成"组合所有视图并按几何结构自适应加权"；vs Yang 2026（也处理信息不对称，但靠**可验证奖励**锚定方向、自蒸馏只调幅度）——AVSD **不依赖二值奖励**（只在可验证设定才有），而是**从 reverse-KL 学习信号本身的几何**推出聚合结构，因此在无奖励时也能用。关键点：用"跨视图一致性"代替"外部奖励"来判断哪个更新可信。

## 怎么做 + 靠不靠谱
- 方法流水线（§2，每 step）：① student 在 prompt x 上 on-policy 采一条 rollout y；② 对同一前缀 h_t，用 M=3 个特权视图 r^(m)=T_m(r) 各算一个 teacher 分布 q_t^(m)（sg 停梯度），得 per-view advantage Δ_t^(m)=log q^(m)−log p；③ 几何共识 A^G_t=均值(Δ^(m))（交集支持，强调所有视图都顶的 token）+ 算术边际 A^A_t（并集支持）；残差 J_t=A^A−A^G≥0（AM-GM 不等式保证非负）；④ 门控 λ_t=C_t·R_t——C_t（对齐分量）= |平均 advantage| / 平均|advantage|，各视图同号时高、互相抵消时低；R_t（幅度分量）= |A^G|/(|A^G|+J_t)，防残差盖过共识；⑤ 重构 advantage Â_t=A^G_t+λ_t·J_t，对应重构目标 q*_t，在它上跑标准 per-token reverse-KL（Eq.3）；梯度只流过 student。
- 逐组件必要性：**几何共识**（定方向）——去掉就成算术边际，§4.1 消融输 1.7%；**算术残差**（补互补信息）——去掉就成纯共识，输 1.4%（HMMT25 上甚至不如单视图 OPSD）；**对齐分量 C_t**——没它则视图冲突时残差仍乱加、符号翻转（Fig.2 Token 1 演示）；**幅度分量 R_t**——没它则大残差能反转共识方向（Appendix B.3 证明 λJ≤|A^G| 正是靠它）。门控两分量都有 Fig.2 的 token 级图示+消融支撑。**未单独消融 C_t vs R_t 各自贡献**（只整体消融了 consensus-only/arithmetic-only），属可补充项。
- 关键机制/公式（直觉）：核心是 reverse-KL 的 token 级 advantage A_t(v)=log q−log p（>0 promote，<0 suppress；Eq.1，policy-gradient 形式 Appendix B.1）。几何均值=最小化"到各 teacher 的平均 reverse-KL"的解；算术均值=最小化平均 forward-KL 的解（对应 opinion pooling 的 log-linear vs linear pool，Appendix B.2）。门控的精髓：**残差只能在共识已定的方向上加力或减力，永远不能翻转方向**——这把 Yang 2026 指出的"特权信息泄漏"风险限制住了。
- 实验与证据：3 个 backbone（Qwen3-4B/8B、DeepSeek-R1-Distill-Qwen-7B）× 数学（AIME24/25、HMMT25，训练用 OpenThoughts 数学子集，三视图=full/partial(ratio0.5)/answer）+ 代码（训 Codeforces Python 5K，评 Codeforces 留出 100 题 + LiveCodeBench v6，三视图=reference/hint/execution-feedback）。指标 Avg@8。**核心数字（Table 1）**：Qwen3-4B 数学较 base +5.5、较最强基线 OPSD +2.2；Qwen3-8B 较最强基线 GRPO +3.1；DeepSeek-7B 较自蒸馏基线 +2.3；Qwen3-8B 代码较 OPSD +2.4。**token 级证据（§4.2 Fig.4/Fig.8）**：错误 rollout 上，GRPO 在 29.6% 例子无学习信号、OPSD 给 12.3% 高影响 token 错误正号、**AVSD 降到 2.9%**——直接证明信号更可靠。baseline 公平：SFT/GRPO/OPSD 同数据同 reverse-KL 目标，仅视图数不同，较公平。
- 假设与失效边界：【原文】(§A Limitations) 只验证数学/代码，tool-use agent 等更广设定未验；依赖**构造视图的质量**（数学三视图靠规则切分），噪声标注解会给不可靠 teacher 信号；仅实例化于 reverse-KL，换其他 distribution-matching 损失时其 advantage 解释与有界性保证"需进一步研究"（Appendix B.5）。【推断】增益 2–3% 属中等而非压倒性，且高度集中于竞赛数学；门控两条件（同号+成比例）是**手工启发式**，缺最优性论证；M 增大收益递减（§4.3：数学第 4 视图只 +0.7/+0.2）。
- 祛魅总结：【推断】真贡献是"**用跨视图一致性这一免奖励信号，把'哪个更新可信'量化成方向-幅度分解 + 有界门控**"——这是干净、可迁移的思路，且 token 级错误正号率从 12.3%→2.9% 是有说服力的机制证据。被适度包装的是：它声称缓解 Yang 2026 的"互信息 gap"，但本质各视图 teacher 仍条件化于特权信息，门控只是**经验性抑制**危险 token，**并未从原理上消除** student 编码不可见关联的压力（§2.3 自己也只说"addresses this risk by preventing residuals from freely determining direction"）。增益幅度被乐观呈现（多处用相对最强基线的 +2~3%）。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=token 级 reverse-KL advantage（跨 3 视图的几何共识方向 + 门控残差幅度）｜**改什么**=student 参数（LoRA，logits 分布对齐）｜**何时改**=在线 per-step（on-policy rollout + 即时多视图 teacher 评估）｜**免梯度?**=否（标准梯度下降，reverse-KL 蒸馏）｜**记忆-技能生命周期**=不适用（无外部记忆/技能库）｜**防遗忘机制**=不适用（单任务后训练，未涉持续学习）
- ⑦ 开源代码+框架/harness：https://github.com/duykhuongnguyen/AVSD （旧 analysis 记已 clone 到 resource/repos/avsd 约 1.9M）。框架栈=**DeepSpeed ZeRO-2 + HF Accelerate + PEFT(LoRA) + vLLM**，trainer 沿用 OPSD（siyan-zhao/OPSD）→ 底层 **TRL**。【待核】具体依赖版本（trl 0.26 等）来自旧 analysis 对仓库的核查，本次未重新进仓核对。核心实现 `src/avsd/common/multiview_distill.py::build_avsd_target`。
- 💰 资源/成本与可扩展性：【原文】AVSD 相比单视图只多"对同一前缀做 M 个视图的 teacher 前向"（prefill-only，可 batch），不需额外 rollout；Table 2/Appendix B.4：训练时间 41.5s→46.7s/step，约 **1.13× 开销**。因 rollout（顺序生成）才是主成本。
- 🎯 对"探索-巩固"对标：**支撑 + 可借组件**。判定：AVSD 的"几何共识=可靠方向、门控残差=谨慎补充"与 TSRD"teacher 当稀疏脚手架教选路+回轨"同构——共识方向≈student 能走通的稳妥开头（探索/选路），残差≈互补但有风险的分支（需谨慎采纳）。**可直接借的组件**：① 方向-幅度分解 + 有界门控（λJ≤|A^G|，保证"脚手架只调力度不夺方向"）可移植到 path-recovery 的"单点接管"——让 teacher 提示只能微调而非覆盖 student 自选恢复分支；② 多视图里"partial solution"视图天然就是"给前缀/脚手架"。**缺口**：无 MTP/前瞻；门控是 token 级即时计算，非 step/turn 级；不涉及把学到的东西固化进记忆/技能。
- 🔭 开放问题/未来方向：【原文】扩到 tool-use agent；换其他蒸馏损失时重建目标的理论保证；视图质量/数据质量依赖。【推断】门控启发式的最优性/可学习化；视图数 M 与质量的系统扫描；把"跨视图一致性"思想用到 multi-turn/agentic 的 turn 级信用分配；与 MTP 前瞻结合——把"未来 token 一致性"也纳入共识判据。

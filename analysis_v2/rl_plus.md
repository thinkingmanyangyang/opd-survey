rl_plus | RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization | 北京大学 + 阿里通义实验室 + University of Alberta（Yihong Dong、Xue Jiang 一作@通义实习；Zhi Jin、Ge Li、Yongbin Li 等） | arXiv 2508.00222（v5 2026-04-15；标注 Preprint July 2025）· cs.AI | 主题线 L2(统一SFT-RL)+L3(RLVR/GRPO)·相关性 较高

**原始论文**:https://arxiv.org/abs/2508.00222

## 一眼看懂
- 🟦 TL;DR:纯 on-policy RLVR(如 GRPO)会"收窄"基座能解的问题集——pass@1 升但 pass@128 反而低于基座(capability boundary collapse,因为巨大动作空间+稀疏奖励逼模型只做"向内利用"而非"向外探索")。RL-PLUS 用混合策略把外部数据(SFT 示范)稳定吸进 RL:① **Multiple Importance Sampling(MIS)** 把外部样本当作"旧策略 πθold 与未知外部策略 πω 的混合",分母用 πθ_old + πω(贝叶斯估计 πω≈½πθold+½均匀分布),πθold 充当"方差护栏"使比值有界,解决分布失配;② **Exploration-Based Advantage(focal 式)** 给"正确但当前策略概率低"的 token 放大优势(权重 (1−detach(πθ))^γ),逼模型关注低概率正确路径;③ **去掉 clip**(clip 会压制高信息低概率事件的梯度=正想学的新知识)。结果:6 个数学基准 SOTA、6 个 OOD 任务领先、pass@k 曲线全程高于基座(真突破天花板)。【Abstract,§3,图1】
- 最巧的一步:**MIS 的分母构造(式4:2πθ /(πω+πθold))**。抽掉它(消融"−MIS",Table4)平均从 53.4 暴跌到 45.5(=退回普通 GRPO 水平)——它是把"从外部 off-policy 数据稳定学习"变得可行的关键;没它,要么用 on-policy 代理(系统偏差 Lemma A.5)、要么用纯 off-policy 权重(支撑失配 A.6 + 高方差 A.7),都会让外部数据吸收失败。πθold 在分母里"即使 πω 很差比值也有界"(Theorem 3.1)是整个方法稳定性的支柱。

## 为什么做
- 研究背景:RLVR(o1/DeepSeek-R1/Kimi)靠可验证奖励驱动 LLM 延展 CoT、自发反思与探索,被视为通向更强 AI 的路径。【§1,L34-42】
- 解决的具体痛点:多项工作(Havrilla 2024、Shao 2024、Yue 2025a)指出现行 RLVR 不能让模型获得**新**推理能力,只复用基座已有模式——pass@1 升、pass@128 反低于基座(图1a),即可解问题集被收窄(**capability boundary collapse**);根因=解空间巨大+奖励稀疏,长链单步出错即清零整条轨迹奖励,模型被迫"向内利用"而非"向外探索";伴随 entropy collapse(熵坍缩,过度确定、丧失探索)。【§1,§2.2,L43-106】
- 相关工作 & 各自不足:① on-policy RLVR(GRPO/PRIME-Zero 隐式过程奖励/Oat-Zero 简化 advantage)——受限基座知识、boundary+entropy collapse;② 混合 SFT-RL:顺序 SFT→RL(InstructGPT)易**灾难性遗忘**+低效;统一/交错框架(ReLIFT 交替 RL 与难题在线微调、LUFFY 混合策略选择性模仿高质量外部轨迹、TAPO 注入抽象"思维模式"、SASR/SuperRL 自适应切换 SFT/RL)——常依赖复杂可能不稳的启发式;"GRPO w/ SFT Loss"简单相加反而掉点;UFT 统一 SFT-RL 加速收敛但未显式解决"稳定 off-policy 更新 + 同时导向新解探索"。【§2.2,L207-251】
- 动机链(用孔子"学而不思则罔,思而不学则殆"作隐喻):当前 RLVR="思而不学"(只想不学外部知识),SFT="学而不思"(只模仿不内化、遇新题脆)→需既能稳定从外部 off-policy 数据"学"、又能显式激励"思"出低概率但正确路径的混合策略。两大挑战:① 分布失配(标准 IS 不够:on-policy 代理有系统偏差,纯 off-policy 高方差+支撑失配,且 πω 未知);② 高效提取外部数据价值(模型天生偏好高概率 token,但新知识常藏在被忽略的低概率正确 token)。【§1,§2.2 Motivation】
- 与最近邻工作的Δ:vs LUFFY(并发,也混合策略用外部轨迹)——LUFFY 把外部策略概率近似为 1(当完美 oracle);RL-PLUS 用贝叶斯估计 πω≈½πθold+½U,消融(Table4 "πθ/πθω with Our Policy Estimation" 47.5 vs naive 变体)证明这比 oracle/πθold 近似各高 ~3 点。vs SFT+GRPO——RL-PLUS 平均 +5.2 点,且 OOD 不退化(SFT 系在 OOD 编程任务严重退化)。vs DAPO/Dr.GRPO 等纯 on-policy 改进——它们仍在基座内,RL-PLUS 靠外部数据+低概率探索突破 pass@k 天花板。关键 Δ:**用 MIS 把"稳定吸收外部数据"理论化(有界方差),并用 focal 式 advantage 显式奖励低概率正确路径**——同时解决"稳定学"和"激励思"。

## 怎么做 + 靠不靠谱
- 方法流水线:① 准备外部静态数据集 De(高质量示范/解题轨迹)+ 内部 on-policy rollout Do;② 内部项:标准 GRPO policy gradient(式1 的 ri,t·Ai,负责稳住并精炼已有能力);③ 外部项:对每个外部 token 算 MIS 比率 rm(式4,分母 πω+πθold,πω 用贝叶斯均值 ½πθold+½U 估计)× Exploration advantage Ac(式5-6,标准化 reward × focal 权重 (1−detach(πθ))^γ);④ 复合目标(式7)= 内部利用 + 外部探索;⑤ **去 clip**(让低概率高信息事件梯度不被压);⑥ γ=0.5 默认。【§3.1-3.3】
- 逐组件必要性(消融 Table4 充分):
  - **MIS**:去掉→53.4→45.5(回到 GRPO 水平),最大跌幅——稳定吸收外部知识的核心。【L985-986】
  - **Exploration-Based Advantage**:去掉→53.4→50.9,−2.5 点——高效探索低概率正确路径的贡献。【L983-984】
  - **πω 贝叶斯估计**:vs πθold 近似(41.7)、vs oracle≈1(LUFFY 式,44.6),用本文估计 47.5,各 +~3 点。【Table4,L987-992】
  - **去 clip**:论证(§3.3 + 附录梯度分析 L1630-1677:梯度 ∝ Ai·(1−pt)^γ,pt→0 权重→1、pt→1 权重→0,即聚焦低概率正确动作)——clip 会砍掉正想要的低概率事件梯度;无独立数值消融但有理论支撑。
  - **detach/stop-gradient**(式6):focal 权重里 detach(πθ) 防梯度经概率回传,增训练稳定性——工程性必要项。
  - **γ 超参**(§附录 图6):不敏感但 γ=0.5 峰值;任何 γ 都超 GRPO。
- 关键机制/公式(直觉):MIS 分母 = 旧策略 + 外部策略的混合,因为旧策略 πθold 被刻意保持接近 πθ,所以即便外部策略 πω 烂到天上,比值也有上界(方差护栏,Theorem 3.1)——这是把"理论正确但高方差的 off-policy"驯服成"可稳定训练"的关键。focal 权重 (1−πθ)^γ:模型对某正确外部 token 越没把握(πθ 小)权重越大→把优势信号放大到"被忽视的低概率正确区域"。
- 实验与证据:基座 Qwen2.5-Math-7B(主)+ LLaMA-3.1-8B / DeepSeek-Math-7B / Qwen2.5-Math-1.5B(泛化);ID 基准 AIME24/25、AMC、MATH-500、Minerva、Olympiad;OOD 基准 HumanEval/LeetCode/LiveCodeBench(编程)+ ARC-c/GPQA-diamond/MMLU-Pro(科学 QA)。**关键数字**:ID 平均 RL-PLUS 53.4 > LUFFY 50.1 > ReLIFT 49.4 > TAPO 49.5 > SFT+GRPO 48.2 > GRPO 45.5;OOD 平均 48.8 > GRPO 44.9(+3.9 over 次优);跨族相对增益最高 +69.2%。pass@k(图1b/图3):RL-PLUS 全程 > 基座(真突破天花板),而 GRPO 在大 k 处被基座反超。训练动态(图2):baseline 熵坍缩到≈0(丧失探索),naive 直接塞外部数据→"熵爆炸"(输出混乱),RL-PLUS 熵不归零(保留探索潜力)、response length 稳增。【Table1/2/3,图1-3】
- baseline 公平吗:同基座 Qwen2.5-Math-7B 横扫十余 baseline(含并发 LUFFY/ReLIFT/TAPO),且专设 "SFT/GRPO/GRPO w-SFT Loss/SFT+GRPO" 四个直接对照隔离"外部知识 vs 自探索 vs 二者结合",公平性强;pass@k 与熵动态正面回应了"是否真扩边界"的核心问题(不是只刷 pass@1)。
- 看着强但没回答核心问题:**外部数据 De 的质量/来源是前提**——论文核心收益来自"稳定吸收高质量外部示范",但对 De 的依赖与其构造(用什么 teacher/数据生成)在正文着墨少,这本质上把"能否突破边界"外包给了"外部数据有多好";"突破基座天花板"严格说是"突破基座 + 注入了外部数据所含的新知识",并非凭空产生新能力(与 rethink_opd"高分≠新知识、需 teacher 带新知识"异曲同工)。
- 假设与失效边界:【原文】① 需要静态外部数据集 De(含正确轨迹);② πω 未知,用 ½πθold+½U 贝叶斯估计(假设"具体代理 vs 最大不确定"各 ½ 先验);③ 主实验数学域 + Qwen2.5-Math-7B,泛化在 4 个模型上验证。【推断】MIS 方差护栏依赖"πθold 始终接近 πθ"——若训练中策略漂移过大(长训/大步去 clip),护栏可能松动;focal 权重对"正确但低概率"的判定依赖 reward 正确性,若 verifier 噪声大或外部数据含错误轨迹,会把错误低概率路径也放大;去 clip 在外部数据差时可能放大不稳(naive 塞外部数据已现"熵爆炸",RL-PLUS 靠 MIS 压住,但边界未充分探)。
- 祛魅总结:真贡献=① 把"capability boundary collapse"用 pass@k 清晰刻画并给出有理论支撑(有界方差 MIS)的混合策略解;② focal 式 exploration advantage 显式奖励低概率正确路径,配合去 clip,实证 pass@k 突破基座。【推断】被高估的可能是"突破天花板"的叙事——更准确说是"高效注入外部数据所含知识 + 保住探索熵",新能力来自外部数据而非算法本身凭空创造;被低估的是 MIS 的方差护栏这一可迁移的稳定性工具(比 LUFFY oracle 近似更 principled)。它是混合 SFT-RL 谱系里"理论 + 探索"结合较好的一篇,但收益与外部数据质量强绑定。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=verifier 二元 reward → GRPO 组相对 advantage(内部)+ MIS 加权的外部数据 advantage,focal 重加权低概率正确 token | **改什么**=策略参数 θ;复合目标 = 内部利用项 + 外部探索项 | **何时改**=在线 RLVR,内外数据同批次联合优化(无 clip) | **免梯度?**=否(policy gradient);但 focal 权重内用 detach/stop-gradient 阻断概率项回传 | **记忆-技能生命周期**=无显式记忆/技能库;外部知识=静态数据集 De 经 MIS 吸收进参数 | **防遗忘机制**=核心针对"capability boundary collapse(=一种探索能力遗忘)";靠"内部利用项稳住已有能力 + 不归零的熵 + MIS 稳定融合"避免顺序 SFT→RL 的灾难性遗忘
- ⑦ 开源代码+框架/harness:https://github.com/YihongDong/RL-PLUS(v1 元信息已核:含 rl_plus/{deepscaler,verl,scripts,setup.py}、exp_scripts/、eval_scripts/、data/);框架 **VeRL + DeepScaleR**。【§1 脚注 L49】
- 💰 资源/成本与可扩展性:原文未给 GPU 时/卡数;训练 600 步级别(图2 横轴);在 4 个模型族(1.5B-8B)验证泛化。可推断成本=常规 7B RLVR 量级 + 外部数据生成开销;去 clip + MIS 不增显著算力。【§4,Table3】
- 🎯 对"探索-巩固"对标:**强支撑(探索侧),直接同源于 path-recovery,可借组件多**。映射:RL-PLUS 的"capability boundary collapse + 向内利用 vs 向外探索"正是 TSRD"探索/选路"要对抗的失败模式;其 **focal 式 exploration advantage(放大"正确但低概率"路径)** 与 TSRD"path-recovery:走偏后自选恢复分支并固化低概率但正确的恢复路径"高度同源——都在显式奖励"模型自己不容易走到、但正确"的路径。**可借组件**:① **MIS 方差护栏(πθold 在分母兜底)**——若 TSRD 用 teacher/外部恢复轨迹做 path-recovery 监督,MIS 是比 oracle 近似更稳的融合工具(直接可用于"把 teacher 的恢复分支当外部 off-policy 数据稳定吸收");② **focal (1−πθ)^γ 重加权**——可用于 TSRD 给"低概率关键恢复 token"加权(与 rho1 excess-loss、entropy 的 token 异质性思路互补);③ **detach/stop-gradient 在权重项**——与项目 memory 记录的"forward-hard/backward-soft 解耦"signature 同构(权重值参与前向、梯度被截断),是稳定性可迁移技巧;④ pass@k 作为"是否真扩了可恢复路径集"的评测。**缺口/差异**:RL-PLUS 是 RLVR + 外部静态数据,无 teacher token 级 dense 监督(不是蒸馏)、无 on-policy 自蒸馏、无 MTP 前瞻、无记忆库;其"外部数据"是预备好的静态集,而 TSRD 想要 teacher 在线当稀疏脚手架动态给恢复分支。判定:**探索/path-recovery 侧的强方法参考 + 多个可直接移植的稳定性组件(MIS 护栏、focal 重加权、stop-grad)**。
- 🔭 开放问题/未来方向:【原文】Conclusion 强调 pass@k 与训练动态证明突破天花板;γ 仍有细调空间(附录 图6);未设详尽 Future Work。【推断】把静态外部数据 De 换成 teacher 在线生成的恢复分支(向 on-policy/OPD 靠拢)、把 MIS 护栏用于稳定 OPD/自蒸馏中的 off-policy 成分、把 focal exploration advantage 与 MTP 前瞻结合(用前瞻定位"将走偏需恢复"的低概率正确 token 优先放大)、研究"突破边界"中外部数据质量的下界与 verifier 噪声鲁棒性。

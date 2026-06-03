# opd_survey — A Survey of On-Policy Distillation for Large Language Models

> **一句话重点 (TL;DR)**：本课题(MTP+OPD / TSRD)的核心背景综述——把 On-Policy Distillation(OPD)统一刻画为"学生采样轨迹上的 f-散度最小化",沿三条设计轴(优化什么 / 信号从哪来 / 如何稳定)组织 >100 篇文献,并系统给出成功条件、失效模式与 OPD↔KL-约束 RL 的连接;提供统一分析词汇与"一方法一类别"分类。

**元信息**：arXiv 2604.00626（v3, 2026-05-18, cs.LG, 78 页）｜ 腾讯大语言模型部（Mingyang Song、Mao Zheng,China）｜ 2026 Preprint｜ 主题 T1/T2/T4 综述,High｜ 代码 github.com/nick7nlp/Awesome-LLM-On-Policy-Distillation（awesome-list / 论文清单类综述仓,CloneTier=B,**未 clone**;无独立训练代码）｜ 框架 n/a（综述）

> 说明:本条为 awesome-list 式综述(paper-only)。§6=分类框架(taxonomy),§7=失效/成功条件,§8/§9=覆盖范围(coverage)。无独立实验,数值多为转引各原始方法。

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/opd_survey/fig_02.png)

*Figure 2: Forward KL vs. Reverse KL divergence for fitting a student distribution PS to a bimodal teacher PT . (a) The teacher places equal probability mass on two valid modes (e.g., two correct answers or two stylistic variants). (b) Forward KL minimizes D KL ( PT ∥ PS ) and is mode-covering (zero-*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/opd_survey/fig_01.png)

*Figure 1: Taxonomy of On-Policy Distillation methods organized along three design axes: (1) Objective function design (§4), (2) Signal source and teacher architecture (§5), and (3) Training dynamics and efficiency (§6). Each leaf lists representative methods with a badge showing the method count. Me*

## 1. 相关工作与进展
知识蒸馏 [Hinton 2015] 从同架构压缩工具,演化为跨规模/跨架构迁移能力的通用机制;DeepSeek-R1 把 671B MoE 教师蒸到 1.5B–70B 稠密学生使之具体化。OPD 起点:GKD [Agarwal 2024] 与并发 MiniLLM [Gu 2024] 于 2023 年中把 on-policy 蒸馏带入 LLM,两年内扩展到散度设计、reward-guided、self-play、多教师辩论、agentic 轨迹蒸馏、跨模态等 >100 篇。OPD 已进入生产线:Qwen3、DeepSeek-V4、Gemma 2、MiMo-V2-Flash 均把它作核心训练成分(DeepSeek-V4 更以纯多教师 OPD 替换混合 RL 阶段做模型整合)。既有蒸馏综述 [Xu 2024] 仍用经典压缩框架、把 off/on-policy 当可互换变体。

## 2. 现有工作存在的问题
工业主流是 off-policy 静态模仿(学生在固定语料/教师预生成轨迹上匹配 next-token 分布,每步条件于完美教师前缀)。其结构性缺陷随任务变长、推理密集而加重:推理时学生从自身部分输出自回归生成,偏离即进入训练未覆盖状态——这对应交互式模仿学习的复合误差 O(εT²)(DAgger [Ross 2011])。文献分散在 KD/RLHF/imitation 三社区,记号、基准、失效分类各异,缺统一数学处理与白盒/黑盒/teacher-free 的系统比较。

## 3. Motivation
为爆发式增长(>100 篇)的 OPD 文献提供统一分析框架与设计中心分类;把"从 off-policy 到 on-policy"重述为序列决策问题,证明核心 OPD 算法都是"学生采样轨迹上的 f-散度最小化"(对应 DAgger 把 O(εT²) 降到 O(εT)),从而连接 KD、RLHF、imitation learning 三条线。

## 4. 主要灵感 / 核心直觉
统一直觉:改变"训练数据从哪来"(从静态语料转为学生自身演化策略)比改变"匹配什么"更关键。学生提议轨迹、填充部署时会访问的状态,教师在这些状态上给反馈;散度生成元 f 决定似然比的隐式加权(forward KL 覆盖模式/up-weight 学生低估处,reverse KL 寻峰/up-weight 高估处),πmix 控制 on-policy 探索程度。

## 5. 主要解决思路(一段话讲清核心)
统一目标 L_OPD(θ)=E_{y~πmix}[Σ_t D_f(p_T(·|x,y_<t), p_θ(·|x,y_<t))](Eq.8):f-散度家族(forward/reverse KL、JSD、α-divergence)× 采样混合 πmix × 散度内参数序。把三个奠基方法映入此空间:GKD(πmix=λp_θ+(1−λ)p_data,散度无关)、MiniLLM(reverse KL + REINFORCE)、DistiLLM(skew KLD + replay buffer + 自适应调度)。再沿三设计轴展开方法,并用 §7 统一解释成功/失效。

## 6. 方法详解(分类框架,通俗、分步骤)
三条对应顺序设计决策的轴(Fig.1 taxonomy,每方法归一主类):

- **§4 目标函数设计**:4.1 固定散度(GKD/MiniLLM/DistiLLM/DistiLLM-2/KETCHUP/vOPD/AntiSD);4.2 自适应散度(ToDi/AKL/EOPD/AOPD,按局部几何在 forward/reverse 间切换,优于固定);4.3 RL-增强目标(G-OPD/RLKD/KDRL/RLAD/AlignDistil 等——证 OPD 是 KD-约束 RL 特例,可超越教师上限)。
- **§5 信号源与教师架构**:5.1 白盒 logit(同族 / 跨族,后者处理词表失配 DSKD/ULD/TAID 等);5.2 黑盒/API 受限(标量奖励或成对偏好:Lion/GAD/LUFFY/ThinkTuning 等);5.3 自蒸馏(**最大且增长最快**:5.3.1 特权信息 OPSD/CRISP/OEL/**OPHSD**/COPSD 等;5.3.2 纯自蒸馏 SDFT/SPIN 等;5.3.3 外部反馈 SDPO/SD-ZERO/**OpenClaw-RL** 等)。
- **§6 训练效率与稳定**:6.1 token/样本加权(TIP/SCOPE/R-OPD 等);6.2 课程与难度自适应(PACED/Stable-OPD/CaOPD 等);6.3 计算优化(Lightning-OPD/SKD/FOPD 等)。
- 另含 §2.4 蒸馏 scaling law(Busbridge 2025:教师过强出现 capacity gap)与 §3.3 方法选择因素(把部署约束/算力映射到可用方法)。

## 7. 实验数据集
综述本身无实验。用 per-section 对比表(Tables 1, 3–9, 2)汇总各方法的类别、核心贡献、信号源、loss 粒度等。覆盖白盒/黑盒/teacher-free 三大设定及散度设计、reward-guided、self-play、multi-teacher debate、agentic 轨迹蒸馏、跨模态等分支。引用的具体数值(如 DeepSeek-R1 学生规模 scaling、Qwen3 "OPD 以 ~1/10 GPU 时超直接 RL")均为转引原文。

## 8. 实验结果与主要发现(综述的核心结论与覆盖)

- **成功条件(§7.1)**[Li 2026i]:① 师生需共享兼容推理模式(top-k token 高重叠;非思考教师蒸进思考学生会因初始重叠过低而失败);② 教师须提供超出学生已有的新能力(同数据同配方训出的师生分布趋同、无可迁移信号)。OPD 收益与"可利用的师生差距"成正比,过小过大都不行。另:Kim & Lee 2026 指出 OPSD 更像**压缩**(让模型更高效表达已知解)而非**纠错**(教会解更难题),推荐 SFT→RLVR→correct-only OPSD 流水线序。
- **失效模式(§7.2)**:flawed prefix trap(学生错误前缀使教师条件分布失准)、extrapolation cliff(λ>1 reward 外推超阈值致格式坍缩)、Rock Tokens(高频结构 token 持续高 loss 却无功能贡献,占大量梯度)、self-play saturation/Ouroboros(自蒸馏锁死自身幻觉)、precision-recall/diversity collapse(reverse KL 高 Pass@1 低 Pass@k)、calibration-capability gap(更强但更过自信)、agentic 多轮坍缩(teacher 硬拷贝重置致 KL 从 2.637 骤降 0.343、轨迹结构侵蚀、reward-hint runaway)。
- **统一理论(§7.3)**:散度选择本质是正则化决策;OPD≈稠密 KL-约束 RL,与 DPO/偏好优化同属"由散度选择与监督密度参数化"的目标族;Stable-OPD 加 reference 散度项 + rollout 混合可破坏长度膨胀自放大环(+7.2%)。
- **决策框架(§7.4)**:师生容量比 >10× 且推理浅时用纯 off-policy SFT;学生 >~7B / 多步推理误差复合 / off-policy loss 平台但 on-policy reward 仍升时切 OPD;否则用 hybrid(off-policy 预热 + on-policy 精修)。
- **工业/系统(§8)**:五种部署模式(两阶段蒸馏、模型整合如 DeepSeek-V4/KAT-Coder-V2、多预算推理 ORBIT、agentic 蒸馏 TCOD/MAD-OPD/Skill-SD/OpenClaw-RL、安全闭环 Safactory);系统侧需教师 co-hosting、logit-tensor 传输(70B 教师 8×H100 约 16GB/batch)、staleness 容忍,常用 OpenRLHF/veRL/SLIME 分离 rollout/scoring/update。
- **开放问题(§9)**:on-policy 蒸馏 scaling law(rollout 预算 R 为新轴)、uncertainty-aware 反馈、agent-level 蒸馏、KD 与 RL 的融合谱系。

## 9. 结果如何支撑其主张(覆盖与论证质量)
统一框架由 Eq.8 + 三奠基方法映射 + DAgger O(εT²)→O(εT) 论证支撑;成功/失效模式逐条引文献佐证并配 mitigation;OPD↔RL 等价由 G-OPD/Li 2025 的梯度分解(稠密 KD 项 + MC RL 项)支撑。作为综述其"贡献"是组织与统一而非新实验,论证密度高、交叉引用一致,但所有数值依赖原文可信度,综述未独立复现。

## 10. 逻辑自洽性(中性评估)
分类(一方法一主类)清晰、三轴正交且承接(目标→信号→稳定),失效模式按根因而非症状归类,理论节把分散现象收敛为少数原理,整体自洽。可质疑处:f-散度统一框架对 reward-guided/黑盒方法的覆盖偏形式化;"一方法一主类"对跨多轴方法有归类武断之嫌(作者已说明按最显著贡献归类);DAgger 界在 LLM 上的适用性作者自己也加了限定(教师在 OOD 前缀上可能失准)。

## 11. 残留问题 / 局限
综述自陈的开放问题即其局限边界:缺 on-policy 蒸馏的联合 scaling law(NT/NS/D/R 指数未定);uncertainty-aware 反馈、agent-level 蒸馏、长 horizon 下后段 token 监督质量退化等仍开放。作为 awesome-list 综述,覆盖随领域(2025–2026 高速增长)易过时;无统一基准复现,方法间数值不可直接横比;部分前沿引用为同期 preprint,结论稳定性待时间检验。

## 12. 开源代码与框架(链接+框架+代码可得性)
仓库 github.com/nick7nlp/Awesome-LLM-On-Policy-Distillation(awesome-list / curated paper list,CloneTier=B,**本次未 clone**)。无独立训练/实验代码,仅论文清单与分类。综述统一用"f-散度在学生采样轨迹上最小化"的框架刻画各 OPD 算法(框架本身 n/a)。

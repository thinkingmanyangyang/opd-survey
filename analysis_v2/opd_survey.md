opd_survey | A Survey of On-Policy Distillation for Large Language Models | 腾讯大语言模型部(Mingyang Song, Mao Zheng) | 2026-05-18 · arXiv 2604.00626 v3 · Preprint(cs.LG, 78 页) | 主题线 L1+L2+L4 综述(OPD/自蒸馏 + 统一 SFT-RL + agentic 蒸馏) · 相关性 高
> 类型说明:本条是**综述(survey)**,无独立实验。下文"方法/组件"栏写其 **taxonomy(分类框架)** 与覆盖面;数值多为转引各原始方法。

**原始论文**：https://arxiv.org/abs/2604.00626

## 一眼看懂
- 🟦 TL;DR：本课题(MTP+OPD / TSRD)的核心背景综述。把 On-Policy Distillation(OPD)统一刻画为"**学生采样轨迹上的 f-散度最小化**"(Eq.8),沿**三条设计轴**(优化什么 / 信号从哪来 / 如何稳定)组织 >100 篇文献;系统给出**成功条件、失效模式、以及 OPD↔KL-约束 RL 的等价**;并把"从 off-policy 到 on-policy"重述为序列决策问题——对应 DAgger 把复合误差从 \(O(\epsilon T^2)\) 降到 \(O(\epsilon T)\)。一方法归一主类,提供统一分析词汇。【原文 §Abstract / §1 / §3 / §6 / §7】
- 最巧的一步：**用 f-散度统一框架(Eq.8)+ DAgger 复合误差论证**把分散在 KD / RLHF / imitation 三社区的 OPD 文献收编成一棵树。抽掉这个统一框架,综述就退回"按表面相似度堆方法"的清单(它批评既有蒸馏综述 Xu 2024 正是如此)。这个框架让"散度选 forward/reverse"、"信号来自 white-box/black-box/self"、"如何稳"三件事正交化、可逐方法定位。【原文 §3 / §6 / §2.2-2.3】

## 为什么做
- 研究背景：KD 从同架构压缩工具演化为跨规模/跨架构迁移能力的通用机制(DeepSeek-R1 把 671B 教师蒸到 1.5B-70B 学生)。OPD 起点是 GKD(Agarwal 2024)与并发 MiniLLM(Gu 2024)于 2023 年中把 on-policy 蒸馏带入 LLM,两年内扩展到散度设计、reward-guided、self-play、多教师辩论、agentic 轨迹蒸馏、跨模态等 >100 篇。OPD 已进生产线:Qwen3、DeepSeek-V4、Gemma 2、KAT-Coder-V2 等均把它作核心训练成分。【原文 §1 / §Abstract】
- 解决的具体痛点：① 工业主流是 **off-policy 静态模仿**(学生在固定语料/教师预生成轨迹上匹配 next-token,每步条件于完美教师前缀)——其结构性缺陷随任务变长/推理密集而加重:推理时学生从自身部分输出自回归生成,偏离即进入训练未覆盖状态(对应交互式模仿学习的复合误差 \(O(\epsilon T^2)\),DAgger);② 文献分散在 KD / RLHF / imitation **三社区**,记号/基准/失效分类各异,**缺统一数学处理**与 white-box/black-box/teacher-free 的系统比较;③ 既有蒸馏综述(Xu 2024)仍用经典压缩框架、把 off/on-policy 当可互换变体。【原文 §Abstract / §1 / §2.2】
- 相关工作 & 各自不足：经典 KD 综述(Xu 2024,压缩框架、未区分 on/off-policy 的结构差异);分散的 OPD 单篇(各自记号/失效分类不通用)。共性缺口:无人给 OPD **统一框架 + 设计中心分类 + 成功/失效条件 + OPD↔RL 连接**的整体处理。【原文 §1】
- 动机链：现状(OPD 文献爆发 >100 篇但分散、无统一处理)→ 缺陷(off-policy 静态模仿的 \(O(\epsilon T^2)\) 复合误差 + 三社区割裂)→ 所以(把 OPD 重述为序列决策、证明核心算法都是"学生轨迹上的 f-散度最小化"、对应 DAgger \(O(\epsilon T^2)\to O(\epsilon T)\),连接 KD/RLHF/imitation)。【原文 §Abstract / §2.2 / §3】
- 与最近邻工作的 Δ：相对 **Xu 2024 蒸馏综述**——从"压缩框架/方法表面分类"转为"**序列决策 + f-散度统一框架 + 设计中心分类**",差在把 on-policy 当作"改训练数据从哪来"的范式转变(而非 off-policy 的可互换变体),并给出 DAgger 误差界与 OPD↔KL-约束 RL 的形式等价。【原文 §1 / §2.2 / §7.3】

## 怎么做 + 靠不靠谱
- 方法流水线（综述的组织框架,非实验）：① 用 **f-散度统一目标(Eq.8)**:\[L_{\text{OPD}}(\theta)=\mathbb E_{y\sim\pi_{\text{mix}}}\Big[\textstyle\sum_{t=1}^{|y|} D_f\big(p_T(\cdot\mid x,y_{<t}),\,p_\theta(\cdot\mid x,y_{<t})\big)\Big]\]——f-散度家族 × 采样混合 \(\pi_{\text{mix}}\) × 散度内参数序 → ② 把三个奠基方法映入此空间(GKD / MiniLLM / DistiLLM) → ③ 沿三轴展开 taxonomy(Fig.1,一方法归一主类) → ④ 用 §7 统一解释成功/失效 + OPD↔RL 连接 → ⑤ §8 工业部署模式 + §9 开放问题。【原文 §3 / §6 / §7 / §8 / §9】
- 逐组件必要性(此处=taxonomy 三轴 + 关键节)：
  - **§4 目标函数设计轴(优化什么)**：4.1 固定散度(GKD/MiniLLM/DistiLLM/DistiLLM-2/AntiSD 等);4.2 自适应散度(ToDi/AKL/EOPD/AOPD,按局部几何在 forward/reverse 间切换,优于固定);4.3 RL-增强目标(G-OPD/RLKD/KDRL/RLAD/AlignDistil——证 OPD 是 KD-约束 RL 特例,可超越教师上限)。【原文 §4】
  - **§5 信号源与教师架构轴(信号从哪来)**：5.1 白盒 logit(同族/跨族,后者处理词表失配 DSKD/ULD/TAID/Delta-KD/Veto/PromptKD);5.2 黑盒/API 受限(标量奖励或成对偏好:Lion/GAD/LUFFY/ThinkTuning/PRISM/ROPD 等,ROPD 用 rubric 替 logit 达 ~10× 样本效率、Qwen3-4B 学生 68.75% 超 GPT-5.2 教师 67.08%);5.3 **自蒸馏(早 2026 最大且增长最快的类)**:5.3.1 特权信息(OPSD/CRISP/GATES/OEL/OPHSD/COPSD)、5.3.2 纯自蒸馏(SDFT;游戏论 SPIN/IRIS 被明确排除到 §9)、5.3.3 外部反馈(SDPO/SD-ZERO/OpenClaw-RL)。【原文 §5 / §5.3 分布观察 / Table 7-8】
  - **§6 训练效率与稳定轴(如何稳)**：6.1 token/样本加权与自适应信任(TIP/SCOPE/R-OPD,如 top-k 近似 KL 基线减墙钟时间达 57.7%);6.2 课程与难度自适应(PACED/Stable-OPD/CaOPD);6.3 计算优化(Lightning-OPD/SKD/FOPD)。【原文 §6】
  - **§2.4 蒸馏 scaling law**(Busbridge 2025:教师过强出现 capacity gap)与 **§3.3 方法选择因素**(把部署约束/算力映射到可用方法)。【原文 §2.4 / §3.3】
  - 必要性视角:三轴**正交且承接**(目标→信号→稳定按顺序设计决策),缺任一轴则分类不完整;一方法归一主类(按最显著贡献)是为避免跨轴方法的归类爆炸——代价是对跨多轴方法略武断(作者已说明)。
- 关键机制/公式（真实形式 + 直觉,从 PDF 抄准）：
  - **统一 OPD 目标(Eq.8)**:见上。核心解耦了"**采样轨迹** \(y\sim\pi_{\text{mix}}\)"与"**局部匹配度量** \(D_f\)"两件事。
  - **f-散度定义(Eq.9)**:\[D_f(P\|Q)=\mathbb E_{y\sim Q}\Big[f\big(\tfrac{P(y)}{Q(y)}\big)\Big],\quad f:(0,\infty)\to\mathbb R\ \text{凸},\ f(1)=0.\] 生成元 \(f\) 决定似然比 \(p_T/p_\theta\) 的隐式加权:
    - Forward KL \(f(u)=u\log u\):mode-covering(zero-avoiding),覆盖教师所有模式,易在模式间幻觉;
    - Reverse KL \(f(u)=-\log u\):mode-seeking(zero-forcing),聚到教师单峰,高精度低多样;
    - JSD \(f(u)=u\log u-(u+1)\log\frac{u+1}{2}\):对称、有界、平滑插值;
    - α-散度:连续插值 forward(\(\alpha\to1\))↔reverse(\(\alpha\to0\))。
    关键性质:\(D_f(P_T\|P_\theta)=\mathbb E_{y\sim p_\theta}[f(p_T(y)/p_\theta(y))]\) 全部可写成**对学生策略 \(p_\theta\) 的期望**,故梯度可经重参数化样本回传,**无需重要性权重/off-policy 修正**(降方差)。
  - **πmix 与 GKD 映射**:GKD 取 \(\pi_{\text{mix}}=\lambda p_\theta+(1-\lambda)p_{\text{data}}\),\(\lambda\to1\) 纯 on-policy、\(\lambda=0\) 退回 off-policy KD;\(\lambda\) 是 on-policy 探索程度的旋钮(中间值 ~0.5 兼顾曝光与稳定)。【原文 §2.3 / Eq.8-9】
  - **DAgger 复合误差(§2.2)**:若策略以每步误差 \(\epsilon\) 模仿专家,在学生自身状态访问下,长度 \(T\) 轨迹的期望偏差 \(\sim O(\epsilon T^2)\)(off-policy);在学生自己访问到的状态上查询专家(on-policy)把它降到 \(O(\epsilon T)\)。直觉:off-policy 训练让学生没在自己会犯错的状态上被纠正,误差按 \(T^2\) 累积;on-policy 把它压成线性。**作者自加限定(§2.2 Remark)**:LLM 上教师在 OOD(学生错误)前缀可能失准,\(O(\epsilon T^2)\to O(\epsilon T)\) 的理论化简需谨慎(Jeong 2026 给反例式经验证据)。
  - **OPD↔KL-约束 RL 等价(§7.3,G-OPD,Eq.14)**:\[\max_\theta\ \mathbb E_{y\sim p_\theta}\Big[\textstyle\sum_{t=1}^{|y|}\alpha\log\tfrac{p_T(y_t\mid y_{<t})}{p_{\text{ref}}(y_t\mid y_{<t})}-D_{\text{KL}}\big(p_\theta(\cdot\mid y_{<t})\,\|\,p_{\text{ref}}(\cdot\mid y_{<t})\big)\Big].\] \(\alpha=1\) 即标准 reverse KL 蒸馏;\(\alpha>1\) 迫使学生**外推超出教师概率质量**(Reward Extrapolation),在 \(p_T(y\mid x)\) 低但 outcome reward 高的区域发现教师没走过的合理路径——多教师下 ExOPD 甚至产出超过所有同尺寸领域教师的统一学生,说明 **OPD 不必是有损压缩**。这把 GKD(token KL)/MiniLLM(序列 reverse KL)/G-OPD(reward-增强)统一为"散度选择 + 监督密度参数化的 KL-约束策略优化"族。【原文 §7.3 / Eq.14 / §4.3】
  - **梯度分解(KD+RL 互补)**:Li 2025 把 hybrid 梯度拆成 稠密 KD 项(token 级模仿,抑制 PG 高方差)+ MC RL 项(防学生塌到对下游次优的教师模式);REOPOLD 用"师生 log-likelihood ratio 作 token reward"实例化。【原文 §4.3 / §7.3】
  - 核心一句话:**"改训练数据从哪来(从静态语料转为学生自身演化策略)比改匹配什么更关键。"** 现代 OPD = 放松经典 KD 的**四条假设**(小师生差距、共享词表、相似容量、off-policy 数据足够)。【原文 §2.2-2.3 / §3】
- 实验与证据(综述=覆盖与论证质量)：
  - **统一框架支撑**：Eq.8 + 三奠基方法映射 + DAgger \(O(\epsilon T^2)\to O(\epsilon T)\) 论证;OPD↔RL 等价由 G-OPD 的梯度分解(稠密 KD 项 + MC RL 项)支撑。【原文 §3 / §7.3】
  - **成功条件(§7.1)**:① 师生需共享兼容推理模式(top-k token 高重叠;非思考教师蒸进思考学生会因初始重叠过低而失败);② 教师须提供超出学生已有的新能力(同数据同配方训出的师生分布趋同、无可迁移信号)。OPD 收益与"可利用的师生差距"成正比,过小过大都不行。另:OPSD 更像**压缩**(让模型更高效表达已知解)而非**纠错**(教更难题),推荐 SFT→RLVR→correct-only OPSD 流水线序。【原文 §7.1】
  - **失效模式(§7.2)**:flawed prefix trap(学生错误前缀使教师条件分布失准)、extrapolation cliff(reward 外推超阈致格式坍缩)、Rock Tokens(高频结构 token 持续高 loss 却无功能贡献、占大量梯度)、self-play saturation/Ouroboros(自蒸馏锁死自身幻觉)、precision-recall/diversity collapse(reverse KL 高 Pass@1 低 Pass@k)、calibration-capability gap(更强但更过自信)、agentic 多轮坍缩(teacher 硬拷贝重置致 KL 从 2.637 骤降到 0.343、轨迹结构侵蚀、reward-hint runaway)。【原文 §7.2 / L340-341】
  - **统一理论(§7.3)**:散度选择本质是正则化决策;OPD≈稠密 KL-约束 RL;Stable-OPD 加 reference 散度项 + rollout 混合可破坏长度膨胀自放大环。【原文 §7.3】
  - **决策框架(§7.4)**:师生容量比 >10× 且推理浅时用纯 off-policy SFT;学生 >~7B / 多步推理误差复合 / off-policy loss 平台但 on-policy reward 仍升时切 OPD;否则用 hybrid。【原文 §7.4】
  - **工业/系统(§8)**:多种部署模式(两阶段蒸馏、模型整合如 DeepSeek-V4/KAT-Coder-V2、多预算推理、agentic 蒸馏、安全闭环);系统侧需教师 co-hosting、logit-tensor 传输、staleness 容忍,常用 OpenRLHF/veRL/SLIME 分离 rollout/scoring/update。【原文 §8】
  - "看着强但没回答核心问题"：作为综述其贡献是组织与统一而非新实验,论证密度高、交叉引用一致,但**所有数值依赖原文可信度**,综述未独立复现。
- 假设与失效边界：
  - 【原文 §2.2 Remark / L330-338】**DAgger 界在 LLM 上的适用性作者自己加了限定**——教师在 OOD(学生错误)前缀上可能失准,\(O(\epsilon T^2)\to O(\epsilon T)\) 的理论化简需谨慎(Jeong 2026 经验反例)。
  - 【推断】**f-散度统一框架对 reward-guided/黑盒方法的覆盖偏形式化**——标量奖励/成对偏好硬塞进 f-散度形式有牵强处。依据:§5.2 黑盒方法与 Eq.8 的散度形式契合度低于白盒。
  - 【推断】**"一方法一主类"对跨多轴方法武断**(作者按最显著贡献归类,但许多方法同时改目标+信号+稳定)。依据:作者在 §6 末自述按主导贡献归类。
  - 【推断】作为 awesome-list 式综述(paper-only),**覆盖随领域高速增长(2025-2026)易过时**;部分前沿引用为同期 preprint,结论稳定性待时间检验;无统一基准复现,方法间数值不可直接横比。
- 祛魅总结【推断】：
  - 真贡献：**把 OPD 从"散落三社区的方法集"收敛成"少数原理 + 三正交轴 + 成功/失效条件 + OPD↔RL 等价"**,分类按根因(失效模式按机制而非症状)、理论节把分散现象收敛为少数原理,是本课题最有价值的"地图"。成功条件(§7.1)与失效模式(§7.2)对设计 TSRD 极有指导性。
  - 包装/被高估处：f-散度统一是漂亮的**形式化包装**,但对黑盒/reward-guided 的覆盖偏形式化、对跨轴方法归类武断;DAgger 界的 LLM 适用性作者自己打了折扣;数值全为转引、无复现。**它是地图不是实验**——读者不应把它的统一框架当成"已被实证的定律"。

## 结构化抽取
- 🎯 机制速览6轴(此处=综述如何刻画这 6 轴,作为 taxonomy 维度)：
  - **学什么信号**：综述把信号源作为**第二轴(§5)**系统分类——白盒 logit(同/跨族)、黑盒标量奖励/成对偏好/rubric、自蒸馏(特权信息/纯自/外部反馈)。覆盖全谱。
  - **改什么**：第一轴(§4)优化什么——固定散度/自适应散度/RL-增强目标(决定改 student 分布的方式与是否超越教师,Eq.14 的 \(\alpha>1\) 即超越)。
  - **何时改**：综述强调 \(\pi_{\text{mix}}\)(on-policy 探索程度,GKD 的 \(\lambda\))与课程/难度自适应(§6.2)——即"何时用学生数据、何时用教师数据、按难度调度"。
  - **免梯度?**：均为梯度优化;但 §7.3 揭示 OPD≈KL-约束 RL,可用策略梯度形式(reverse KL 作 advantage)或显式散度 loss 实现,两种都覆盖。
  - **记忆-技能生命周期**：综述未把"记忆/技能库"列为独立轴(OPD 主要是参数内蒸馏);agentic 蒸馏(§8)与自蒸馏触及"把行为固化进参数",但持续学习/技能库非其焦点——**对本课题 L5(记忆/技能库)覆盖薄弱**。
  - **防遗忘机制**：综述把稳定/防漂移作为**第三轴(§6)+ §7.2 失效模式 + §7.3 reference 散度项**处理(Stable-OPD 加 reference 散度破坏长度膨胀环);防"灾难性遗忘"非主轴,更多是防训练坍缩/长度膨胀。
- ⑦ 开源代码+框架/harness：仓库 github.com/nick7nlp/Awesome-LLM-On-Policy-Distillation(**awesome-list / curated paper list,CloneTier=B,本次未 clone**)——**无独立训练/实验代码**,仅论文清单与分类。综述统一用"f-散度在学生采样轨迹上最小化"框架(Eq.8)刻画各 OPD 算法(框架本身 n/a)。**未克隆原因:纯论文清单仓,无方法代码可核(见决策②B 层只记录不 clone);需手动访问该 GitHub 获取最新论文清单。** 【既有 analysis + §Abstract 脚注】
- 💰 资源/成本与可扩展性：综述本身无成本;但 §8 系统侧给出 OPD 部署成本要点——教师 co-hosting、logit-tensor 传输、staleness 容忍;§2.4/§9 提出 rollout 预算 \(R\) 是 on-policy 蒸馏独有的新 scaling 轴(质量 \(\propto N_T^\alpha N_S^\beta D^\gamma R^\delta\) 形式待定)。【原文 §8 / §2.4 / §9】
- 🎯 对"探索-巩固"对标：**强支撑(本课题的理论地图与设计指南);是定位 TSRD 的坐标系**。判定依据：① **直接给本课题提供框架坐标**——TSRD 的 OPD 一侧 = §4 目标轴(reverse/forward KL,Eq.9 生成元)× §5 信号轴(白盒同族 teacher)× §6 稳定轴;"teacher 当稀疏脚手架"可定位为 §6.1 token/样本加权 + 自适应信任的极端(只在关键 token 给 KL);"path-recovery 走偏回轨"正对 §7.2 的 **flawed prefix trap** 失效模式(综述指出这是 student 远离 teacher 时的主导失效)——本课题用"自选恢复分支"正是针对此 trap 的 mitigation。② **成功条件(§7.1)直接约束 TSRD 设计**:师生需 top-k 高重叠(否则蒸馏失败)——支撑本 idea"teacher 偏向 student 自己能走通的开头"(保证重叠);"OPD 更像压缩非纠错、推荐 SFT→RLVR→correct-only OPSD"——提醒本课题若要"教 student 走通走不通的题"(纠错)需超出纯 OPSD(对应 Eq.14 的 \(\alpha>1\) 外推路线)。③ **OPD↔KL-约束 RL 等价(§7.3,Eq.14)** 把本课题的 OPD 与 RLVR(L3)/GFT(L2)统一到同一目标族,是三线打通的理论支点。**可借组件(概念层)**:f-散度框架(选 reverse 还是自适应散度)、\(\pi_{\text{mix}}\) 设计、§6 的关键 token 加权(向稀疏脚手架靠拢)、§7.2 失效模式清单(设计时逐条规避)。**缺口/警示**:① 综述**对 L5(记忆/技能库&持续学习防遗忘)覆盖薄弱**——OPD 是参数内蒸馏,本课题"巩固进记忆/技能且不遗忘"需在 OPD 之外补;② **无 MTP/前瞻探针的专门讨论**(token 信用/前瞻属 L6,综述只在 §6 token 加权侧触及);③ **agent-level 蒸馏仍是开放问题**(§9)——本课题 L4(多轮自进化)在综述里尚无成熟方法可抄,是空白也是机会;④ 警示:综述自己说 DAgger 界在 LLM 打折、f-散度对黑盒偏形式化——别把它的统一框架当铁律。一句话:**它是把 TSRD 三线(OPD/GFT/RLVR)钉在同一坐标系上的地图,成功/失效条件直接指导设计;但记忆-技能、MTP 前瞻、agent 蒸馏三块对本课题最关键的增量恰是它标注的薄弱/开放区。**
- 🔭 开放问题/未来方向：
  - 【原文 §9】on-policy 蒸馏的**联合 scaling law**(\(N_T/N_S/D/R\) 指数未定,rollout 预算 \(R\) 为新轴);**uncertainty-aware 反馈**(按教师不确定性调监督);**agent-level 蒸馏**(多轮/工具轨迹);KD 与 RL 的融合谱系;长 horizon 下后段 token 监督质量退化;game-theoretic self-play(SPIN/IRIS)作为邻接范式。【原文 §9 / §2.4】
  - 【推断】把本课题"MTP 前瞻 + 稀疏关键步脚手架 + 自选回轨"填进综述标注的三块空白(agent-level 蒸馏 / uncertainty-aware 反馈对应"前瞻挑关键步" / 记忆-技能持续学习);用综述的 §7.2 失效模式清单作 TSRD 的"设计 checklist"逐条规避(尤其 flawed prefix trap 与 diversity collapse)。

RETURN: opd_survey|读到PDF=是(78页综述;核§Abstract/§1-3 f-散度框架Eq.8-9+DAgger O(εT²)→O(εT)/§4-6三轴taxonomy/§7.1-7.4成功失效条件+OPD↔KL约束RL Eq.14/§8工业/§9开放问题)|L线=L1+L2+L4综述|对标=强支撑(把TSRD三线钉在同一坐标系的地图,§7.1成功条件+§7.2失效模式(flawed prefix trap对应path-recovery)直接指导设计;但记忆-技能/MTP前瞻/agent蒸馏三块对本课题最关键的增量恰是其薄弱/开放区)|残留待核=0(未克隆:awesome-list论文清单仓无方法代码,见决策②B,需手动访问GitHub)

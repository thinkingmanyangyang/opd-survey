opd_survey | A Survey of On-Policy Distillation for Large Language Models | 腾讯大语言模型部(Mingyang Song, Mao Zheng) | 2026-05-18 · arXiv 2604.00626 v3 · Preprint(cs.LG, 78 页) | 主题线 L1+L2+L4 综述(OPD/自蒸馏 + 统一 SFT-RL + agentic 蒸馏) · 相关性 高
> 类型说明:本条是**综述(survey)**,无独立实验。下文"方法/组件"栏写其 **taxonomy(分类框架)** 与覆盖面;数值多为转引各原始方法。

**原始论文**：https://arxiv.org/abs/2604.00626

## 一眼看懂
> 一句话导读：这是本课题最重要的"地图"——一篇把 On-Policy Distillation(OPD,在线策略蒸馏)整个领域梳理清楚的综述。它的核心动作是:用一个统一的数学式子(学生在自己轨迹上最小化"f-散度",即教师与学生分布的某种距离)把上百篇分散在三个社区的论文收进同一棵分类树,再沿三个维度(优化什么 / 信号从哪来 / 怎么稳)逐个定位,并系统总结"什么时候有效、什么时候会崩"。

- 🟦 TL;DR：这是本课题(MTP+OPD / TSRD)的核心背景综述。它把 On-Policy Distillation(OPD)统一刻画为"**学生在自己采样的轨迹上最小化 f-散度**"(Eq.8;f-散度是一族用来度量两个分布差异的量,KL 是其特例),并沿**三条设计轴**——优化什么 / 信号从哪来 / 如何稳定——组织了 >100 篇文献。它系统给出了:
  - **成功条件、失效模式**;
  - **OPD 与 KL-约束 RL 之间的等价关系**。
  它还把"从 off-policy 到 on-policy"重述为一个序列决策问题——这对应 DAgger 把复合误差从 \(O(\epsilon T^2)\) 降到 \(O(\epsilon T)\)(直觉见下文)。每个方法只归入一个主类,从而提供一套统一的分析词汇。【原文 §Abstract / §1 / §3 / §6 / §7】
- 最巧的一步：**用 f-散度统一框架(Eq.8)+ DAgger 复合误差论证**,把分散在 KD / RLHF / imitation 三个社区的 OPD 文献收编成一棵树。如果抽掉这个统一框架,综述就退回成"按表面相似度堆方法"的清单(它批评既有的蒸馏综述 Xu 2024 正是如此)。这个框架的好处是,把三件事正交化、可以逐方法定位:
  - 散度选 forward 还是 reverse;
  - 信号来自 white-box(看得到教师 logit)、black-box(看不到)还是 self(自蒸馏);
  - 如何稳定训练。【原文 §3 / §6 / §2.2-2.3】

## 为什么做
> 一句话导读：工业界默认的训法是"让学生背教师的固定范文"(off-policy 静态模仿),但任务一长、推理一密集就出问题——学生一走偏就进入训练里从没见过的状态,误差越滚越大。同时这个领域两年里炸出上百篇论文、却散在三个互不通话的社区,谁也没把它们统一起来讲。这篇综述就来补这两块。

- 研究背景：知识蒸馏(KD)已从"同架构的压缩工具"演化为"跨规模、跨架构迁移能力"的通用机制(例如 DeepSeek-R1 把 671B 的教师蒸到 1.5B-70B 的学生)。OPD 的起点是 GKD(Agarwal 2024)与并发的 MiniLLM(Gu 2024)——它们在 2023 年中把 on-policy 蒸馏带进 LLM;此后两年里,这个方向扩展到散度设计、reward-guided、self-play、多教师辩论、agentic 轨迹蒸馏、跨模态等,共 >100 篇。OPD 已进入生产线:Qwen3、DeepSeek-V4、Gemma 2、KAT-Coder-V2 等都把它当作核心训练成分。【原文 §1 / §Abstract】
- 解决的具体痛点：
  - ① 工业主流是 **off-policy 静态模仿**:学生在固定语料、或教师预生成的轨迹上匹配 next-token,每一步都条件于教师的完美前缀。它的结构性缺陷会随任务变长、推理变密集而加重——推理时学生是从自己的部分输出自回归生成的,一旦偏离就进入训练里从未覆盖的状态(这对应交互式模仿学习的复合误差 \(O(\epsilon T^2)\),即 DAgger 分析的那个量);
  - ② 文献分散在 KD / RLHF / imitation **三个社区**,记号、基准、失效分类各不相同,**缺一套统一的数学处理**,也缺 white-box / black-box / teacher-free 之间的系统比较;
  - ③ 既有的蒸馏综述(Xu 2024)仍沿用经典的压缩框架,把 off-policy 和 on-policy 当成可互换的变体。【原文 §Abstract / §1 / §2.2】
- 相关工作 & 各自不足：经典的 KD 综述(Xu 2024,用压缩框架,未区分 on/off-policy 的结构差异);加上分散的 OPD 单篇(各自的记号/失效分类不通用)。共性缺口是:没人对 OPD 做过**统一框架 + 设计中心分类 + 成功/失效条件 + OPD↔RL 连接**这样一个整体处理。【原文 §1】
- 动机链（一步步推下来）：
  - 现状:OPD 文献爆发到 >100 篇,但分散、无统一处理;
  - 缺陷:off-policy 静态模仿有 \(O(\epsilon T^2)\) 的复合误差,加上三社区割裂;
  - 结论:于是把 OPD 重述为序列决策问题、证明核心算法都是"学生轨迹上的 f-散度最小化"、对应 DAgger 的 \(O(\epsilon T^2)\to O(\epsilon T)\),从而把 KD/RLHF/imitation 三者连起来。【原文 §Abstract / §2.2 / §3】
- 与最近邻工作的 Δ：相对 **Xu 2024 蒸馏综述**——从"压缩框架 / 方法表面分类"转向"**序列决策 + f-散度统一框架 + 设计中心分类**"。差别在于:它把 on-policy 当作"改训练数据从哪来"的范式转变(而不是 off-policy 的可互换变体),并给出了 DAgger 的误差界、以及 OPD 与 KL-约束 RL 之间的形式等价。【原文 §1 / §2.2 / §7.3】

## 怎么做 + 靠不靠谱
> 一句话导读：综述本身没有实验,它的"方法"就是那套组织框架——先用一个统一式子(Eq.8)把所有 OPD 写成同一个模板,再沿三条轴(§4 优化什么 / §5 信号从哪来 / §6 怎么稳)把上百篇论文逐个挂上去,最后用 §7 统一讲清成功/失效条件。靠不靠谱?它的论证密度高、交叉引用一致,但所有数字都是转引原文、综述自己没复现,所以"地图准不准"取决于被引论文。

- 方法流水线（这是综述的组织框架,不是实验）：分五步——
  - ① 用 **f-散度统一目标(Eq.8)**:\(\displaystyle L_{\text{OPD}}(\theta)=\mathbb E_{y\sim\pi_{\text{mix}}}\Big[\textstyle\sum_{t=1}^{|y|} D_f\big(p_T(\cdot\mid x,y_{<t}),\,p_\theta(\cdot\mid x,y_{<t})\big)\Big]\)——这个模板由三个自由度构成:f-散度家族、采样混合分布 \(\pi_{\text{mix}}\)、以及散度内的参数次序;
  - ② 把三个奠基方法(GKD / MiniLLM / DistiLLM)映射进这个空间;
  - ③ 沿三条轴展开 taxonomy(分类树,Fig.1;一个方法只归入一个主类);
  - ④ 用 §7 统一解释成功/失效,并打通 OPD↔RL 的连接;
  - ⑤ 用 §8 讲工业部署模式、§9 列开放问题。【原文 §3 / §6 / §7 / §8 / §9】
- 逐组件必要性(这里 = taxonomy 的三条轴 + 几个关键小节)：
  - **§4 目标函数设计轴(优化什么)**:
    - 4.1 固定散度(GKD / MiniLLM / DistiLLM / DistiLLM-2 / AntiSD 等);
    - 4.2 自适应散度(ToDi / AKL / EOPD / AOPD,会按局部几何在 forward/reverse 间切换,优于固定散度);
    - 4.3 RL-增强目标(G-OPD / RLKD / KDRL / RLAD / AlignDistil——它们证明 OPD 是 KD-约束 RL 的特例,且可超越教师的上限)。【原文 §4】
  - **§5 信号源与教师架构轴(信号从哪来)**:
    - 5.1 白盒 logit(分同族/跨族;跨族要处理词表失配,如 DSKD / ULD / TAID / Delta-KD / Veto / PromptKD);
    - 5.2 黑盒 / API 受限(只能拿到标量奖励或成对偏好:Lion / GAD / LUFFY / ThinkTuning / PRISM / ROPD 等;其中 ROPD 用 rubric 替代 logit,达到约 10× 的样本效率,Qwen3-4B 学生拿到 68.75%、超过 GPT-5.2 教师的 67.08%);
    - 5.3 **自蒸馏(截至 2026 年初,这是最大、增长最快的一类)**:又分 5.3.1 特权信息(OPSD / CRISP / GATES / OEL / OPHSD / COPSD)、5.3.2 纯自蒸馏(SDFT;博弈论的 SPIN/IRIS 被明确排除到 §9)、5.3.3 外部反馈(SDPO / SD-ZERO / OpenClaw-RL)。【原文 §5 / §5.3 分布观察 / Table 7-8】
  - **§6 训练效率与稳定轴(如何稳)**:
    - 6.1 token/样本加权与自适应信任(TIP / SCOPE / R-OPD,例如用 top-k 近似 KL 作基线,可减少墙钟时间达 57.7%);
    - 6.2 课程与难度自适应(PACED / Stable-OPD / CaOPD);
    - 6.3 计算优化(Lightning-OPD / SKD / FOPD)。【原文 §6】
  - **§2.4 蒸馏 scaling law**(Busbridge 2025:教师过强时会出现 capacity gap,即容量鸿沟)与 **§3.3 方法选择因素**(把部署约束/算力映射到可用的方法)。【原文 §2.4 / §3.3】
  - 必要性视角:这三条轴**正交且层层承接**(按"目标→信号→稳定"的顺序逐级做设计决策),缺任何一条轴分类就不完整;"一个方法只归一个主类"(按最显著的贡献来归)是为了避免跨轴方法导致归类爆炸——代价是对那些同时跨多轴的方法略显武断(作者已说明这点)。
- 关键机制/公式（真实形式 + 直觉,从 PDF 抄准）：
  - **统一 OPD 目标(Eq.8)**:见上。核心解耦了"**采样轨迹** \(y\sim\pi_{\text{mix}}\)"与"**局部匹配度量** \(D_f\)"两件事。
  - **f-散度定义(Eq.9)**:\(\displaystyle D_f(P\|Q)=\mathbb E_{y\sim Q}\Big[f\big(\tfrac{P(y)}{Q(y)}\big)\Big],\quad f:(0,\infty)\to\mathbb R\ \text{凸},\ f(1)=0.\) 这里的生成元 \(f\)(一个满足"凸 + \(f(1)=0\)"的函数)决定了对似然比 \(p_T/p_\theta\) 的隐式加权方式。换不同的 \(f\) 就得到不同的散度,各有性格:
    - Forward KL,\(f(u)=u\log u\):mode-covering(也叫 zero-avoiding),会去覆盖教师的所有模式,代价是容易在模式之间产生幻觉;
    - Reverse KL,\(f(u)=-\log u\):mode-seeking(也叫 zero-forcing),会聚到教师的单个峰,精度高但多样性低;
    - JSD,\(f(u)=u\log u-(u+1)\log\frac{u+1}{2}\):对称、有界,在两者之间平滑插值;
    - α-散度:可连续插值,\(\alpha\to1\) 趋向 forward、\(\alpha\to0\) 趋向 reverse。
    一个关键性质:\(D_f(P_T\|P_\theta)=\mathbb E_{y\sim p_\theta}[f(p_T(y)/p_\theta(y))]\),即它们全都能写成**对学生策略 \(p_\theta\) 的期望**。好处是梯度可以直接经重参数化样本回传,**不需要重要性权重、也不需要 off-policy 修正**(从而降低方差)。
  - **πmix 与 GKD 的映射**:GKD 取 \(\pi_{\text{mix}}=\lambda p_\theta+(1-\lambda)p_{\text{data}}\)。当 \(\lambda\to1\) 时是纯 on-policy,当 \(\lambda=0\) 时退回 off-policy KD;所以 \(\lambda\) 就是控制"on-policy 探索程度"的旋钮(取中间值约 0.5 时,可兼顾曝光与稳定)。【原文 §2.3 / Eq.8-9】
  - **DAgger 复合误差(§2.2)**:假设策略以每步误差 \(\epsilon\) 来模仿专家。在 off-policy 设定下(学生在专家状态上学、却要在自己状态上跑),长度为 \(T\) 的轨迹期望偏差约为 \(O(\epsilon T^2)\);而 on-policy 设定(直接在学生自己访问到的状态上查询专家)能把它降到 \(O(\epsilon T)\)。直觉是:off-policy 训练没让学生在"自己会犯错的状态"上被纠正过,误差就按 \(T^2\) 累积;on-policy 则把它压成线性。**注意作者自己加了限定(§2.2 Remark)**:在 LLM 上,教师在 OOD(学生犯错)的前缀上也可能失准,所以 \(O(\epsilon T^2)\to O(\epsilon T)\) 这个理论化简要谨慎看待(Jeong 2026 给出了反例式的经验证据)。
  - **OPD↔KL-约束 RL 的等价(§7.3,G-OPD,Eq.14)**:\(\displaystyle \max_\theta\ \mathbb E_{y\sim p_\theta}\Big[\textstyle\sum_{t=1}^{|y|}\alpha\log\tfrac{p_T(y_t\mid y_{<t})}{p_{\text{ref}}(y_t\mid y_{<t})}-D_{\text{KL}}\big(p_\theta(\cdot\mid y_{<t})\,\|\,p_{\text{ref}}(\cdot\mid y_{<t})\big)\Big].\) 这里 \(\alpha=1\) 就是标准的 reverse KL 蒸馏;而 \(\alpha>1\) 会迫使学生**外推、走出教师的概率质量之外**(称为 Reward Extrapolation),从而在"教师概率 \(p_T(y\mid x)\) 低、但 outcome reward 高"的区域,发现教师没走过的合理路径。在多教师设定下,ExOPD 甚至能产出一个超过所有同尺寸领域教师的统一学生——这说明 **OPD 不一定是有损压缩**。这一节也把 GKD(token KL)、MiniLLM(序列 reverse KL)、G-OPD(reward-增强)统一成同一族:"散度选择 + 监督密度参数化的 KL-约束策略优化"。【原文 §7.3 / Eq.14 / §4.3】
  - **梯度分解(KD 与 RL 互补)**:Li 2025 把 hybrid 梯度拆成两项——一项是稠密的 KD 项(做 token 级模仿,抑制 PG 的高方差);另一项是 MC 的 RL 项(防止学生塌到那些"对下游其实次优"的教师模式)。REOPOLD 把它实例化为"用师生的 log-likelihood ratio 当 token reward"。【原文 §4.3 / §7.3】
  - 核心一句话:**"改训练数据从哪来(从静态语料,转为学生自身不断演化的策略),比改'匹配什么'更关键。"** 换句话说,现代 OPD = 放松经典 KD 的**四条假设**:师生差距小、共享词表、容量相近、off-policy 数据足够。【原文 §2.2-2.3 / §3】
- 实验与证据(综述=覆盖与论证质量)：
  - **统一框架支撑**：Eq.8 + 三奠基方法映射 + DAgger \(O(\epsilon T^2)\to O(\epsilon T)\) 论证;OPD↔RL 等价由 G-OPD 的梯度分解(稠密 KD 项 + MC RL 项)支撑。【原文 §3 / §7.3】
  - **成功条件(§7.1)**:两个条件——
    - ① 师生要共享兼容的推理模式(表现为 top-k token 高度重叠;比如把一个"非思考型"教师蒸进一个"思考型"学生,就会因初始重叠太低而失败);
    - ② 教师必须提供超出学生现有能力的新东西(如果用同样的数据、同样的配方训出师生,二者分布会趋同、没有可迁移的信号)。
    所以 OPD 的收益与"可利用的师生差距"成正比,差距过小或过大都不行。另外:OPSD 更像是**压缩**(让模型更高效地表达它已会的解),而不是**纠错**(教它更难的题),因此推荐按 SFT→RLVR→correct-only OPSD 的流水线顺序来用。【原文 §7.1】
  - **失效模式(§7.2)**:综述列了一串(都按"根因"而非"症状"来命名)——
    - flawed prefix trap:学生的错误前缀让教师的条件分布也跟着失准;
    - extrapolation cliff:reward 外推一旦超过阈值,会导致格式坍缩;
    - Rock Tokens:高频的结构性 token 持续维持高 loss、却没有功能贡献,还占掉大量梯度;
    - self-play saturation / Ouroboros:自蒸馏把自身的幻觉锁死;
    - precision-recall / diversity collapse:reverse KL 带来高 Pass@1、低 Pass@k;
    - calibration-capability gap:模型变强了,却也更过度自信;
    - agentic 多轮坍缩:teacher 硬拷贝重置,导致 KL 从 2.637 骤降到 0.343、轨迹结构被侵蚀、出现 reward-hint runaway。【原文 §7.2 / L340-341】
  - **统一理论(§7.3)**:散度的选择本质上是一个正则化决策;OPD ≈ 稠密的 KL-约束 RL;Stable-OPD 通过加一个 reference 散度项 + 混合 rollout,可以打破"长度膨胀自我放大"的环。【原文 §7.3】
  - **决策框架(§7.4)**:给出何时用哪种训法——
    - 当师生容量比 >10× 且推理较浅时,用纯 off-policy SFT;
    - 当学生 >约 7B、或多步推理会让误差复合、或 off-policy loss 已到平台但 on-policy reward 仍在涨时,切换到 OPD;
    - 其余情况用 hybrid。【原文 §7.4】
  - **工业/系统(§8)**:列了多种部署模式——两阶段蒸馏、模型整合(如 DeepSeek-V4 / KAT-Coder-V2)、多预算推理、agentic 蒸馏、安全闭环。系统侧则需要教师 co-hosting、logit-tensor 传输、对 staleness(陈旧度)的容忍,常用 OpenRLHF / veRL / SLIME 来把 rollout、scoring、update 三件事分离。【原文 §8】
  - "看着强但没回答核心问题"：作为综述其贡献是组织与统一而非新实验,论证密度高、交叉引用一致,但**所有数值依赖原文可信度**,综述未独立复现。
- 假设与失效边界：
  - 【原文 §2.2 Remark / L330-338】**DAgger 这个误差界在 LLM 上是否适用,作者自己加了限定**——教师在 OOD(学生犯错)的前缀上可能失准,所以 \(O(\epsilon T^2)\to O(\epsilon T)\) 这个理论化简要谨慎(Jeong 2026 给了经验反例)。
  - 【推断】**f-散度统一框架对 reward-guided / 黑盒方法的覆盖偏形式化**——把标量奖励、成对偏好硬塞进 f-散度的形式,有牵强之处。依据是:§5.2 的黑盒方法与 Eq.8 的散度形式契合度,明显低于白盒方法。
  - 【推断】**"一个方法只归一个主类"对那些跨多轴的方法显得武断**(作者按最显著贡献来归类,但许多方法其实同时改了目标、信号和稳定三块)。依据是作者在 §6 末尾自述了"按主导贡献归类"。
  - 【推断】作为一篇 awesome-list 式的综述(只含论文、无代码),它**随领域高速增长(2025-2026)而容易过时**;部分前沿引用是同期的 preprint,结论的稳定性还要等时间检验;而且没有统一基准复现,方法之间的数值不能直接横比。
- 祛魅总结【推断】：
  - 真贡献:**把 OPD 从"散落在三社区的方法集",收敛成"少数原理 + 三条正交轴 + 成功/失效条件 + OPD↔RL 等价"**;分类是按根因来的(失效模式按机制命名、而非按症状),理论那几节把分散的现象收敛成了少数原理。这是本课题最有价值的"地图"。其中成功条件(§7.1)与失效模式(§7.2)对设计 TSRD 极有指导性。
  - 包装/被高估处:f-散度统一是一层漂亮的**形式化包装**,但它对黑盒 / reward-guided 的覆盖偏形式化、对跨轴方法的归类也武断;DAgger 界在 LLM 上的适用性,作者自己也打了折扣;数值全是转引、没有复现。**它是地图、不是实验**——读者不该把它的统一框架当成"已被实证的定律"。

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
- 🎯 对"探索-巩固"对标：**强支撑(它是本课题的理论地图与设计指南);是定位 TSRD 的坐标系**。判定依据如下:
  - ① **它直接给本课题提供了框架坐标**:TSRD 的 OPD 一侧 = §4 目标轴(reverse/forward KL,即 Eq.9 的生成元)× §5 信号轴(白盒同族 teacher)× §6 稳定轴。在这个坐标系里:
    - "teacher 当稀疏脚手架"可定位为 §6.1 "token/样本加权 + 自适应信任"的极端情形(只在关键 token 上给 KL);
    - "path-recovery 走偏回轨"正对应 §7.2 的 **flawed prefix trap** 失效模式(综述指出,这是 student 远离 teacher 时的主导失效)——本课题用的"自选恢复分支",正是针对这个 trap 的缓解手段。
  - ② **成功条件(§7.1)直接约束 TSRD 的设计**:
    - 师生需 top-k 高重叠(否则蒸馏失败)——这支撑了本 idea"teacher 偏向 student 自己能走通的开头"(以保证重叠);
    - "OPD 更像压缩、而非纠错,推荐 SFT→RLVR→correct-only OPSD"——这提醒本课题:若要"教 student 走通它原本走不通的题"(即纠错),就得超出纯 OPSD(对应 Eq.14 中 \(\alpha>1\) 的外推路线)。
  - ③ **OPD↔KL-约束 RL 等价(§7.3,Eq.14)** 把本课题的 OPD 与 RLVR(L3)、GFT(L2)统一到同一个目标族,是三线打通的理论支点。
  - **可借组件(概念层)**:f-散度框架(决定选 reverse 还是自适应散度)、\(\pi_{\text{mix}}\) 的设计、§6 的关键 token 加权(向稀疏脚手架靠拢)、以及 §7.2 的失效模式清单(设计时逐条规避)。
  - **缺口/警示**:
    - ① 综述**对 L5(记忆/技能库 & 持续学习防遗忘)覆盖薄弱**——OPD 是参数内蒸馏,本课题"把能力巩固进记忆/技能且不遗忘"这块,得在 OPD 之外另补;
    - ② **没有对 MTP/前瞻探针的专门讨论**(token 信用/前瞻属于 L6,综述只在 §6 的 token 加权那一侧略微触及);
    - ③ **agent-level 蒸馏仍是开放问题**(§9)——本课题的 L4(多轮自进化)在综述里还没有成熟方法可抄,既是空白也是机会;
    - ④ 警示:综述自己也承认 DAgger 界在 LLM 上要打折、f-散度对黑盒方法偏形式化——别把它的统一框架当成铁律。
  - 一句话:**它是把 TSRD 三线(OPD/GFT/RLVR)钉在同一坐标系上的地图,其成功/失效条件可直接指导设计;但对本课题最关键的三块增量——记忆-技能、MTP 前瞻、agent 蒸馏——恰恰是它自己标注的薄弱区/开放区。**
- 🔭 开放问题/未来方向：
  - 【原文 §9】综述列了一串:
    - on-policy 蒸馏的**联合 scaling law**(\(N_T/N_S/D/R\) 的指数都还没定,而 rollout 预算 \(R\) 是一条新增的轴);
    - **uncertainty-aware 反馈**(按教师的不确定性来调监督强度);
    - **agent-level 蒸馏**(多轮 / 工具轨迹);
    - KD 与 RL 的融合谱系;
    - 长 horizon 下,后段 token 监督质量会退化;
    - 博弈论式的 self-play(SPIN/IRIS)作为一种邻接范式。【原文 §9 / §2.4】
  - 【推断】两个动作:
    - 把本课题的"MTP 前瞻 + 稀疏关键步脚手架 + 自选回轨"填进综述标注的三块空白——agent-level 蒸馏、uncertainty-aware 反馈(对应"前瞻挑关键步")、记忆-技能持续学习;
    - 把综述 §7.2 的失效模式清单当作 TSRD 的"设计 checklist",逐条规避(尤其是 flawed prefix trap 与 diversity collapse)。

RETURN: opd_survey|读到PDF=是(78页综述;核§Abstract/§1-3 f-散度框架Eq.8-9+DAgger O(εT²)→O(εT)/§4-6三轴taxonomy/§7.1-7.4成功失效条件+OPD↔KL约束RL Eq.14/§8工业/§9开放问题)|L线=L1+L2+L4综述|对标=强支撑(把TSRD三线钉在同一坐标系的地图,§7.1成功条件+§7.2失效模式(flawed prefix trap对应path-recovery)直接指导设计;但记忆-技能/MTP前瞻/agent蒸馏三块对本课题最关键的增量恰是其薄弱/开放区)|残留待核=0(未克隆:awesome-list论文清单仓无方法代码,见决策②B,需手动访问GitHub)

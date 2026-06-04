prefix_rft | Prefix-RFT: Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling | 爱丁堡 ILCC + 复旦 + 阿里 Qwen + StepFun + UvA ILLC(Zeyu Huang 等;SFT-RL 统一谱系,与 LUFFY/UFT/ReLIFT 同期并列) | ICML 2026(PMLR 306)·arXiv·v3(2026-05-15) | 主题线 L2(统一 SFT-RL)·相关性 高

**原始论文**:https://arxiv.org/abs/2507.01679

## 一眼看懂
- 🟦 TL;DR:SFT(模仿示范,泛化差)和 RFT(奖励驱动,依赖初始策略、难突破上界)各有短板。本文从一条 demonstration \(y^*\) 里**采一段前缀 \(y^*_{<L}\)(off-policy)当起步提示,让当前策略续写 \(y_{\ge L}\)(on-policy)**,把"前缀+续写"拼成一条混合轨迹 \(y^{(N)}\),和普通 on-policy rollout 一起进 RFT 更新——用同一套 PPO advantage/clip 处理。等于在单阶段内把 SFT 的过程监督和 RFT 的目标导向探索揉到一起:沿可靠路径起步、但保留探索更优续写的自由。【原文 Abstract,§3】
- 最巧的一步:**让前缀 token 也参与梯度、但用整条混合轨迹的 advantage 来加权(而非固定 SFT 权重)**。Table 8 给了三组对照证明这是命门:① Freeze Prefix(前缀不回传梯度)→ 45.4,几乎等于纯 RFT(45.5),说明"显式从前缀学习"不可或缺;② Static Weight(给前缀固定 0.001 权重,类 UFT)→ 43.8,远逊;③ Ours(advantage 加权)→ 51.8。所以巧点不是"给个前缀",而是"前缀的学习强度由轨迹级 advantage 动态决定"——好前缀(让混合轨迹得高奖励)自动被强化。【原文 §A.3,Table 8】

## 为什么做
- 研究背景:LLM 后训练两范式——SFT(在 curated 标注数据上模仿,简单、能注入知识)与 RFT(奖励驱动,o1/DeepSeek-R1 证明能解竞赛数学/代码)。【原文 §1】
- 解决的具体痛点:① SFT 是 behavior cloning,泛化/鲁棒性差;② RFT 奖励稀疏、token 级信用分配难(易致 language mixing 等异常),且效果**强依赖初始策略**,被质疑只是"打磨已有能力"而非提升上界;③ 二者通常被当两个独立串行阶段,缺形式化统一框架。【原文 §1】
- 相关工作 & 各自不足(均作 baseline,精确差异):
  - **LUFFY**(Yan 2025)——把**整条** demonstration 混入 on-policy RFT(限制自主探索、pass@2048 未能显著推高上界);Prefix-RFT 只给前缀。
  - **UFT**(Liu 2025b)——先采前缀,对前缀用**固定小权重 SFT loss** + 续写用 RFT loss(静态权重);Prefix-RFT 证明动态 advantage 加权优于静态(Table 8:43.8→51.8)。
  - **ReLIFT**(Ma 2025)——分阶段交错 SFT/RFT,SFT 专攻 RFT 解不出的难题(需精细调参);Prefix-RFT 单阶段、免分阶段调参。
  - **RL w/ SFT Loss**——直接在 RFT 中混 SFT loss(本文证明 counterproductive,SFT 梯度支配优化,Table 1:40.1<45.5)。
  - **SFT+RFT(两阶段)**——先 SFT 再续 RFT(本文强基线 48.2);Prefix-RFT 单阶段且监督只施加在"最需要的地方"。
  - **Zero-RL 系(SimpleRL-Zero / Oat-Zero)**——纯 RFT 配方,作"我们 RFT 基线足够强"的旁证。【原文 §3+§4+App.A.1】
- 动机链:SFT 注入知识扩边界但泛化差 + RFT 提能力但被初始策略锁死 → 二者互补(§4 实测:SFT 在难题 AIME25 更强、RFT 在中等题 AMC/MATH500 更强)→ 中心问题:如何**形式化融合**两者?→ 先给统一视角(§2:SFT/PG/PPO 都是对 log-prob 加梯度,只差权重),再以"前缀采样"实例化。【原文 §1+§2】
- 与最近邻工作的Δ:vs LUFFY——只给**前缀**而非整条 demo,保留 RFT 解题目标与"受约束的自主性",实测唯一能推高 pass@2048 上界;vs UFT——前缀权重**动态(轨迹 advantage)**而非静态 0.001;vs SFT+RFT 两阶段——单阶段、且 SFT 监督只施加在难题/不确定 token。关键有用点:advantage 加权让"高价值前缀获更大梯度权重"(§3),把过程监督的强度与结果挂钩。【原文 §3+§5】

## 怎么做 + 靠不靠谱
- 方法流水线(§3,Fig.1):① 给定 prompt \(x\) 与 demonstration \(y^*\);② 标准 RFT 先用当前策略 \(\pi_{\theta_{\mathrm{old}}}\) 生成 \(N-1\) 条纯 on-policy rollout \(\{y^{(1)},\dots,y^{(N-1)}\}\);③ 第 \(N\) 条:把 \(y^*\) 截成前缀 \(y^*_{<L}\),用 \(\pi_{\theta_{\mathrm{old}}}\) 续写 \(y_{\ge L}\),拼成混合轨迹 \(y^{(N)}\)(**不额外加轨迹,rollout 预算与标准 RFT 一致**——一条普通 rollout 被替换为前缀引导的混合 rollout);④ 全部 \(N\) 条一起估 prefix-aware advantage \(\hat A_t\);⑤ actor 用带前缀分支的 PPO 更新(前缀 token 与续写 token 都用 \(W^{\mathrm{PPO}}_{i,t}\) 加权,Eq.3),叠加熵约束裁剪 + 前缀长度余弦衰减调度。
- 逐组件必要性(Table 8 / Fig.6,均在 Qwen2.5-Math-7B):
  - **前缀回传梯度(vs Freeze)**:有消融。Freeze=45.4≈RFT 45.5 → 不学前缀=白做。
  - **熵约束裁剪(只更 top-k% 高熵前缀 token)**:有消融。Update All(不裁剪)=45.7,仅边际增益且 **rollout 长度爆炸**(undesirable dynamics)→ 裁剪必要。原理:\(\pi_{\mathrm{off}}\) 远离当前策略,前缀 token 的 \(\pi_\theta\) 概率普遍低、梯度可比 on-policy 大几个数量级(Table 4),不裁会让模型只学示范表面特征(如长度)。【§3+§A.4】
  - **动态 advantage 加权(vs Static 0.001)**:有消融。Static=43.8 vs Ours=51.8 → 动态加权关键。
  - **"增益是否只来自裁剪技巧"对照**:把 top-20% 裁剪用到普通 RFT 的 on-policy rollout(RFT+On-policy Clip)=43.8≈naive RFT → 增益来自"从前缀学习",非裁剪本身。
  - **DR-PO 变体对照**(从前缀起 rollout 但不计入 loss、且 N=8 条都来自同一前缀)=33.8(差),∵ 训练(有引导)与推理(无引导)分布漂移严重。
  - **前缀长度余弦衰减调度(vs Uniform)**:有消融(Fig.6b)。缓解"只学开头 token"的位置偏置 + 内置课程(从"几乎给全 demo"过渡到"几乎纯 RFT",对应 SFT→RFT 配方)。默认 high=0.95、low 从 0.95 余弦衰减到近 0。【§6+§A.4】
- 关键机制/公式(真实符号,从 PDF 抄准 + 直觉):
  - **统一视角(§2)——三种更新都是"对 log-prob 加梯度,只差权重"**:
    \[ \nabla_\theta L_{\mathrm{SFT}}=-\sum_t \nabla_\theta\log\pi_\theta(y^*_t|x,y^*_{<t})\quad(\text{隐式视优势}\equiv1,\ \text{在专家序列}y^*\text{上}), \]
    \[ \nabla_\theta L_{\mathrm{PG}}=\sum_t \hat A_t\,\nabla_\theta\log\pi_\theta(y_t|x,y_{<t})\quad(\text{优势加权},\ \text{在自采轨迹上}), \]
    \[ \nabla_\theta L_{\mathrm{PPO}}=\sum_t \mathbb{I}_{\mathrm{clip}}(r_t,\hat A_t)\,\hat A_t\,r_t\,\nabla_\theta\log\pi_\theta(y_t|x,y_{<t}),\quad r_t=\frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{\theta_{\mathrm{old}}}(y_t|x,y_{<t})}, \]
    其中 clipping 指示子 \(\mathbb{I}_{\mathrm{clip}}(r_t,\hat A_t)=\mathbb{I}\big[\{\hat A_t>0\wedge r_t\le1+\epsilon\}\vee\{\hat A_t<0\wedge r_t\ge1-\epsilon\}\big]\)。
  - **混合梯度统一式**(Eq.2,把每条响应的 token 分成探索集 \(T^{(i)}_{\mathrm{exp}}\)(模型生成)与模仿集 \(T^{(i)}_{\mathrm{imit}}\)(示范)):
    \[ \nabla_\theta L_{\mathrm{Hybrid}}=-\frac{1}{N}\sum_{i=1}^{N}\!\sum_{t\in T^{(i)}_{\mathrm{exp}}}\!\!\alpha_{i,t}\nabla_\theta\log\pi_\theta(y^{(i)}_t|x,y^{(i)}_{<t})\;-\;\frac{1}{N}\sum_{i=1}^{N}\!\sum_{t\in T^{(i)}_{\mathrm{imit}}}\!\!\beta_{i,t}\nabla_\theta\log\pi_\theta(y^{(i)}_t|x,y^{(i)}_{<t}). \]
  - **Prefix-RFT 实例化**(Eq.3,**命门**:令 \(\alpha_{i,t}=\beta_{i,t}=W^{\mathrm{PPO}}_{i,t}=\mathbb{I}_{\mathrm{clip}}(r_t,\hat A_t)\hat A_t r_t\),即前缀 token 也用整条混合轨迹的 PPO 权重):
    \[ -\frac{1}{N}\!\left(\underbrace{\sum_{i=1}^{N-1}\sum_t W^{\mathrm{PPO}}_{i,t}\nabla\log\pi+\sum_{t\ge L} W^{\mathrm{PPO}}_{N,t}\nabla\log\pi}_{\text{Exploration: 标准 rollout + 续写}}\;+\;\underbrace{\sum_{t<L} W^{\mathrm{PPO}}_{N,t}\nabla\log\pi}_{\text{Imitation: 前缀引导}}\right). \]
    直觉:前缀 token(\(t<L\))虽来自 off-policy 专家,却用"整条混合轨迹估出的 advantage"强化——所以对"模型解不好的题",高质量前缀会拿到更高梯度权重(advantage 大),模型从这段部分示范获益更多;同时 PPO 的 ratio+clip 抑制示范数据带来的过大更新。
  - **熵约束裁剪的梯度直觉**(§A.4):示范 token 目标 logit 的梯度幅度 \(|\partial\ell/\partial z|=1-p_{a^*}\)——低熵 token 要么已匹配(\(p_{a^*}\!\to\!1\),梯度≈0)要么自信错配(尖锐覆写),只留高熵 token(模型不确定处);实现上把其余 token 的 advantage 直接置 0。
  - **前缀长度调度**:\(L=\lfloor l\cdot|y^*|\rfloor\),\(l\sim U(\text{low},\text{high})\),high 常数、low 从 high 余弦衰减到近 0——既消位置偏置又内置 SFT→RFT 课程。【原文 §2+§3+Eq.(2)(3)】
- 实验与证据:
  - 数据集/设置:训练=OpenR1-Math-220K 的长度过滤子集(沿用 LUFFY/Yan et al.)约 **46k 题**,每题配一条 **DeepSeek-R1 生成的 demonstration**(前缀来源)。主基座 **Qwen2.5-Math-7B**;另在 Qwen2.5-Math-1.5B / LLaMA-3.1-8B / Qwen3-1.7B-base 验证(Table 6)。评测 6 数学(AIME24/25、AMC、MATH-500、Minerva、Olympiad)+ 3 通用(ARC-c、GPQA*、MMLU-Pro);另用 **pass@2048(n=2048 采样)** 评估推理边界扩展。【§4】
  - 关键数字(Table 1,Qwen2.5-Math-7B):Prefix-RFT 数学均值 **51.8**,超 RFT(45.5)、SFT(44.1)、SFT+RFT(48.2)、UFT(44.2)、ReLIFT(47.8)、RL w/ SFT Loss(40.1),≈/略超 LUFFY(50.1);通用域均值 58.4 最高。**pass@2048**:唯一把上界推高的方法——AIME24/25 上较 base 提升 **+6.67 个百分点**;RFT 在 k=2048 几乎贴 base(无法提上界),SFT 在大 k 反而退化(过度约束解空间)。【Table 1+Fig.2+§5】
  - baseline 公平性:**强**——RFT/SFT/RFT w/ SFT Loss/SFT+RFT/UFT/ReLIFT/LUFFY 全部同基座同数据(46k+R1 demo),仅"如何用 off-policy 数据"不同;并提供显著性检验(Table 5)。
  - 看着强但没回答的:与 LUFFY 在均值上**仅持平**(51.8 vs 50.1),核心卖点其实是"pass@2048 上界"与"简单可插拔",而非均值碾压;且主战场是数学(可验证奖励),代码仅作迁移性补充(Table 3)。
- 假设与失效边界:【原文】实验主要限于"可靠 outcome reward 的可验证推理"(数学);代码生成只作迁移性证据;依赖外部 demonstration——无 demo 即退化为纯 RFT;多候选 demo 的系统化选择留作 future work(§8)。【推断】① 前缀质量直接决定上限——若 demo 本身次优/与任务错位,advantage 加权虽能下调其权重但仍受限;② top-k% 高熵阈值(默认 20%)与前缀长度区间为关键超参,默认值"在所测模型上鲁棒"但跨域最优值未系统给(§A.4);③ 开放式生成/噪声奖励场景未验证。
- 祛魅总结:【推断】真贡献=① 一个干净的统一视角(SFT=A≡1 的 PG 特例)+ ② 把它实例化成"前缀引导的混合轨迹 + advantage 加权 + 熵裁剪 + 余弦长度调度"这一**简单可插拔**配方,且 Table 8 五组对照把每个设计的必要性钉得很实。被高估处:相对 LUFFY 的均值优势很小,"超越所有 baseline"主要靠 pass@2048 上界这一指标(对 k/温度敏感);被低估处:"同 rollout 预算下融合 SFT/RFT"的工程简洁性(只改一条 rollout)其实是落地友好的真亮点。

## 结构化抽取
- 🎯 机制速览6轴:
  - **学什么信号**:结果奖励(verifier 0/1)经 GRPO/Dr.GRPO 组内归一的 advantage \(\hat A_t\);前缀(示范)token 与续写 token 共用此 advantage 加权(prefix-aware),无独立 teacher 软标签。
  - **改什么**:策略参数 \(\theta\)(端到端 PPO 梯度);前缀 token 只对 top-k% 高熵者回传梯度。
  - **何时改**:单阶段在线训练;前缀长度 low 边界随训练余弦衰减(课程:前期多给 demo→后期近纯 RFT)。
  - **免梯度?**:否,全程梯度更新。
  - **记忆-技能生命周期**:无外部记忆/技能库;"技能/路径"通过混合轨迹强化固化进参数;demonstration 是外部静态知识源(off-policy 池),非可写记忆。
  - **防遗忘机制**:无显式防遗忘;熵裁剪 + PPO clip 限制"被示范尖锐覆写"间接保护原策略;余弦衰减让后期回归 on-policy 减少对 demo 的过拟合。【推断:论文以"避免梯度支配/分布漂移"立论,非"防遗忘"】
- ⑦ 开源代码+框架/harness:仓库 https://github.com/ZeroYuHuang/prefix_rft(本地已克隆 ~9.1MB,完整)。**框架 = veRL**,全部实现集中在 `recipe/prefix_rft/`(已逐文件核验):`core_algos.py`(prefix-aware advantage:`compute_grpo_prefix_outcome_advantage` / `compute_dr_grpo_prefix_outcome_advantage` / `compute_dr_grpo_prefix_v2_outcome_advantage`)、`dp_actor.py`(熵 reshaper:`off_adv_reshaper` 做 entropy/entropy_low/random masking)、`ray_trainer.py`、`rl_dataset.py`、`templates.py`、`scheduler/`(cosine_decay 控制器)、`main.py`、`fsdp_workers.py`、单/多节点启动脚本。关键算法可逐行核验。【已核本地仓 recipe/prefix_rft 文件树+函数名】
- 💰 资源/成本与可扩展性:与标准 RFT **同 rollout 预算**(一条 rollout 被替换为混合 rollout,不增轨迹),故额外训练开销低——这是相对 ReLIFT(分阶段)/SFT+RFT(两阶段)的成本优势。分析实验:batch 128、5 epoch,SFT lr 5e-5、RFT/Prefix-RFT lr 1e-6。具体 GPU 卡数/时长正文未给(称见 Appendix A.2)〔原文未在主文给出绝对算力数字〕。
- 🎯 对"探索-巩固"对标:**强支撑,且机制非常贴近**。Prefix-RFT 的"前缀=可靠起步路径(off-policy 引导)+ 续写=自主探索(on-policy)"几乎就是 idea 里"teacher 当稀疏脚手架:偏向自己能走通的开头(探索/选路)+ 走偏后自选恢复(巩固/回轨)"的直接实例——余弦衰减的前缀长度=脚手架从"给大半路径"逐步撤到"几乎不给",对应"巩固后撤除"。可借组件:① **prefix-aware advantage**(前缀 token 用整条轨迹结果反推其价值,Eq.3)可直接迁移到 TSRD 的"选路偏好"建模;② **熵约束裁剪**(只在高熵/不确定处吃示范监督)与本项目记忆里"切高熵/低置信关键步、path-recovery 单点接管"高度同构,是现成的"在哪里施加脚手架监督"的工程答案;③ 余弦长度调度=课程化撤脚手架。缺口:前瞻信号是文本 demo 而非 MTP 概率探针;无记忆/技能库与防遗忘;前缀是静态外部 demo 而非"自己能走通的开头"(student 自生成)——若要对标 TSRD 的"on-policy 自选开头",需把 prefix 来源从外部 demo 换成 student 自采高分前缀。一句判定:**最贴近"探索-巩固"机制的直接基线/可借框架**(尤其熵裁剪选点 + advantage 加权选路),差距在"前瞻载体(文本 vs MTP)"与"前缀来源(外部 demo vs 自生成)"。
- 🔭 开放问题/未来方向:【原文】① 推广到开放式生成与噪声奖励设置;② 多候选 demonstration 的系统化选择(目前靠 advantage 隐式加权);③ 更广域(已试代码,Table 3)迁移(§8)。【推断】① 把前缀来源换成 student 自采的高分前缀(self-prefix),做真正"自选可走通开头"的 on-policy 版,直接对标 TSRD;② 熵裁剪的"高熵=该学的关键 token"假设可与 MTP 前瞻熵结合,做更精的 path-selection 信用分配;③ 前缀长度调度的课程可与难度自适应(难题给更长前缀)联动,论文已观察"难题从 demo 学得更多"但未做自适应长度。

— RETURN —
prefix_rft | 读PDF? 是(17页,§1-§8+App.A.1-A.4,Table 1/6/8 数字 + Eq.1/2/3 + 统一视角 SFT/PG/PPO 梯度逐字核对) | 加厚? 是(统一视角三梯度全转 MathJax + 混合 Eq.2 + Prefix-RFT Eq.3 全展开 + 熵裁剪梯度直觉 + 相关工作扩到 6 条精确对位 + 仓库三函数名核验) | LaTeX公式条数:6(SFT梯度 / PG梯度 / PPO梯度+clip指示子 / 混合Eq.2 / Prefix-RFT Eq.3 / 前缀长度调度) | 待核数:0

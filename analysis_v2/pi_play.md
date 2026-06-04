pi_play | π-Play: Multi-Agent Self-Play via Privileged Self-Distillation without External Data | 中科院自动化所+国科大+美团(Yaocheng Zhang, Yuanheng Zhu 等;自演化/深度搜索谱系,承接 Dr.Zero/SQLM 自博弈) | 2026-04 预印本·arXiv·v2(2026-05-25) | 主题线 L1(OPD/自蒸馏,叠加 L4 Agent 自进化)·相关性 高

**原始论文**:https://arxiv.org/abs/2604.14054

## 一眼看懂
- 🟦 TL;DR:深度搜索 agent 训练缺数据、奖励稀疏、信用分配差。本文发现——自博弈造题时会顺手产出一条"问题构造路径(QCP, question construction path)",即出题人从答案/证据**反向**把题目拼出来的全过程。把这条 QCP \(c\) 当作零成本的"内部特权信息"喂给一个同规模 teacher,让 teacher 据此对 student 做 token 级 reverse-KL 蒸馏,就把稀疏的结果奖励变成稠密逐 token 监督,实现完全无外部数据(data-free)的多智能体协同自演化。【原文 Abstract,§1】
- 最巧的一步:**把 QCP 当特权上下文喂给 teacher**。抽掉它(Table 3:用 ground-truth 答案代替 QCP)效果立刻退回"无蒸馏"水平(GT 264.1 ≈ w/o Distillation 265.5,而 QCP 达 277.2)——因为 GT 只给最终答案、不含"题是怎么从证据推出来的"逻辑,teacher 无从给出有效逐 token 指导。所以方法的命门不是"加个 teacher 蒸馏",而是"teacher 拿到的那条**构造逻辑轨迹** \(c\)"。【原文 §4.1,Table 3】

## 为什么做
- 研究背景:深度搜索 agent = LLM 推理 + 外部搜索引擎多轮检索;RL(RLVR)能显著提升其推理/搜索行为(Search-R1、ToolForge),但规模化训练被**数据**卡脖子。【原文 §1】
- 解决的具体痛点:(1) 传统自博弈只给 student **稀疏结果奖励**,多轮搜索任务学习低效、信用分配弱;(2) 自博弈实际产出的不止 \((q,o^\star)\),还有被忽视的 QCP \(c\)——逆向解题轨迹,但现有自博弈**无法直接拿它做 SFT**,白白浪费;(3) 另一条线"自蒸馏"能改善信用分配,但所需高质量特权信息通常要人类专家/更强模型构造、且依赖 curated QA,难规模化。【原文 §1】
- 相关工作 & 各自不足(站谁肩上 + 精确差异):
  - **监督 RL(Search-R1 / ToolForge)**——靠标注数据与昂贵专家轨迹;π-Play data-free 反超(8B 上 +15.4%)。
  - **自博弈(Dr.Zero / SQLM / SSP,Yue 2026 / Chen 2025b)**——同规模轮流当 examiner/student,但只有稀疏结果奖励;π-Play 多加"QCP-guided teacher"把稀疏奖励补成稠密 token 监督。examiner 的难度奖励 \(r_d\) 与 hop-grouped 优化即直接沿用 Dr.Zero(Yue 2026)。
  - **on-policy distillation(Agarwal 2024/Gu 等)**——依赖**更大的外部 teacher**;π-Play teacher 与 student **同规模**(实为 student 的 EMA),能力优势完全来自"看得到 QCP"而非更大模型。
  - **自蒸馏 + 特权信息(OPSD/SDPO,Hübotter 2026 / Penaloza 2026)**——同规模 teacher+特权信息,但特权信息获取代价高、且需 QA 训练数据;EMA-teacher 更新式(Eq.11)即引自 Hübotter/Penaloza 2026。π-Play 的差异:特权信息**不再外部构造**而是自博弈副产物(免费、可扩展)。【原文 §1+§2.4】
- 动机链:自博弈低效 ∵ 奖励稀疏 → 想用稠密监督(自蒸馏)但特权信息贵 → **关键观察:自博弈天然、低成本、可规模化地产出 QCP**,而 QCP 恰是一种内禀特权信息(记录题如何从证据构造)→ 所以让 teacher 条件于 QCP 做自蒸馏,既稠密又零额外数据。【原文 §1】
- 与最近邻工作的Δ:vs 传统自博弈(Dr.Zero)——多加一个"QCP-guided teacher";vs 经典自蒸馏——特权信息是自博弈副产物;vs on-policy distillation——teacher 与 student 同规模(EMA)。关键有用点:把"出题副产物"变成监督信号,这一步是零成本的。【原文 §1】

## 怎么做 + 靠不靠谱
- 方法流水线(三角色交替优化,Algorithm 1):① **Examiner** \(\pi^E_\phi\) 带搜索工具与引擎交互,造出三元组 \((q,c,o^\star)\),目标=造"中等难度+可验证"的题(难度奖励 Eq.4);② 用 examiner 策略生成一批 student 训练数据 \(D\);③ **Student** \(\pi^S_\theta\) 仅见 \(q\),采 \(G\) 条 rollout,用"结果奖励(GRPO)+ teacher 逐 token reverse-KL 蒸馏"联合目标(Eq.7)更新;④ 每个 student step 后,**Teacher** \(\pi^T_\psi\) 以 student 的 EMA 软更新(Eq.11);⑤ 共 \(M\) 轮(实验 \(M=3\)、每阶段 \(W=50\) 步)。三角色全从同一 base LLM 初始化、全靠搜索工具取外部知识。
- 逐组件必要性(标"有无消融"):
  - **QCP 作为特权上下文**:有消融(Table 3)。去掉→退化到无蒸馏;用 GT/GT+Hop 替代→几乎无增益;Partial QCP(随机砍掉一半)→次优但仍远超 GT。证据充分,是核心命门组件。
  - **蒸馏项本身(teacher guidance)**:有消融(w/o Distillation=265.5 vs QCP=277.2)。
  - **λ 衰减调度(distillation 系数随轮次衰减)**:有消融(Table 9),衰减优于固定 λ——前期靠 teacher 起步、后期放手让 student 自主探索。
  - **EMA teacher(Eq.11)**:τ=0.05(Table 7)。**论文未对"EMA teacher(Eq.11) vs 直接优化理想目标 Eq.2"做对照消融**——理想 teacher 目标 Eq.2 只写出来,实际用 EMA 近似,二者落差未实验量化。【推断:依据 §2.4"directly optimizing it would introduce additional computational overhead",仅给近似无对照】
  - **examiner 难度奖励 Eq.4(k=1 时最大、随正确数线性衰减)**:有 Table 4 验证 examiner 确实越训越难,但难度奖励形式本身无单独消融。
  - **K(每题采样数)**:有消融 Table 10。
- 关键机制/公式(真实符号,从 PDF 抄准 + 直觉):
  - **三角色总目标**(§2.1,examiner Eq.1 / teacher Eq.2 / student Eq.3):
    \(\displaystyle \max_\phi\ \mathbb{E}_{(q,c,o^\star)\sim\pi^E_\phi,\,\{y_k\}_{k=1}^n\sim\pi^S_\theta(\cdot|q)}\big[r_d(o^\star,\{o_k\}_{k=1}^n)\big], \tag{1}\)
    \(\displaystyle \max_\psi\ \mathbb{E}_{(q,c,o^\star)\sim\pi^E_\phi,\,y\sim\pi^T_\psi(\cdot|q,c)}\Big[\mathbf{1}(o=o^\star)-\beta D_{\mathrm{KL}}\big(\pi^T_\psi(\cdot|q,c)\,\|\,\mathrm{stopgrad}[\pi^S_\theta(\cdot|q)]\big)\Big], \tag{2}\)
    \(\displaystyle \max_\theta\ \mathbb{E}_{(q,c,o^\star)\sim\pi^E_\phi,\,y\sim\pi^S_\theta(\cdot|q)}\Big[\mathbf{1}(o=o^\star)-\lambda D_{\mathrm{Distill}}\big(\pi^S_\theta(\cdot|q)\,\|\,\mathrm{stopgrad}[\pi^T_\psi(\cdot|q,c)]\big)\Big]. \tag{3}\)
    注意 Eq.2 是 teacher 的**理想**目标(看得到 \(c\)、又不能离 student 太远),但 §2.4 因开销大改用 EMA 近似(Eq.11)。
  - **examiner 难度奖励**(Eq.4,\(k\) = \(n\) 次采样中答对数;k=1 时最大、随 \(k\) 线性衰减):
    \(\displaystyle r_d(o^\star,\{o_i\}_{i=1}^n)=\mathbf{1}(0<k<n)\,\frac{n-k}{n-1}+r_f,\qquad k=\sum_{i=1}^n\mathbf{1}(o_i=o^\star),\)
    \(r_f\) 为 format reward,鼓励 examiner 边推理边搜索→题目既有事实依据又附带 informative 的构造路径(即 QCP,作特权信息)。
  - **examiner 训练目标**(Eq.5,hop-grouped relative policy optimization,按 cross-hop 复杂度分组归一以降方差):
    \(\displaystyle J_{\mathrm{Examiner}}(\phi)=\mathbb{E}\Big[\tfrac{1}{N}\sum_{h\in H}\sum_{i\in I_h}\log\pi^E_\phi\,A_{i,h}-\beta D_{\mathrm{KL}}(\pi^E_\phi\,\|\,\pi_{\mathrm{ref}})\Big],\)
    hop-wise 优势(Eq.6):\(A_{i,h}=\dfrac{r^d_i-\mathbb{E}_{j\in I_h}[r^d_j]}{\sqrt{\mathrm{Var}_{j\in I_h}[r^d_j]+\delta}}\),\(I_h\) 是 hop 组 \(h\) 内的题集。
  - **student 联合目标**(Eq.7,GRPO 结果奖励 + KL-to-ref − λ·蒸馏):
    \(\displaystyle J_{\pi\text{-play}}(\theta)=\mathbb{E}\Big[\underbrace{\tfrac{1}{G}\sum_{i=1}^G\sum_{t=1}^{|y_i|}L_{i,t}-\beta D_{\mathrm{KL}}(\pi^S_\theta\,\|\,\pi_{\mathrm{ref}})}_{\text{Learning from outcome reward}}\;-\;\underbrace{\lambda D_{\mathrm{Distill}}(\pi^S_\theta\,\|\,\pi^T_\psi)}_{\text{Teacher guidance}}\Big],\)
    PPO-clip 项(Eq.8):\(L_{i,t}=\min\big(w_{i,t}A_i,\ \mathrm{clip}(w_{i,t},1-\epsilon,1+\epsilon)A_i\big)\);重要性权重与组归一优势(Eq.9):
    \(\displaystyle w_{i,t}=\frac{\pi^S_\theta(y_{i,t}|q,y_{i,<t})}{\pi^S_{\theta_{\mathrm{old}}}(y_{i,t}|q,y_{i,<t})},\qquad A_i=\frac{r^e_i-\mathbb{E}_{j\in G}[r^e_j]}{\sqrt{\mathrm{Var}_{j\in G}[r^e_j]+\delta}},\quad r^e(q,o_i)=\mathbf{1}(o_i=o^\star).\)
  - **蒸馏损失**(Eq.10,沿 student rollout 的逐 token reverse-KL,teacher 看得到 \(c\),stop-gradient):
    \(\displaystyle D_{\mathrm{Distill}}(\pi^S_\theta\,\|\,\pi^T_\psi)=\frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \mathrm{KL}\Big(\pi^S_\theta(\cdot\mid q,y_{i,<t})\ \big\|\ \mathrm{stopgrad}\big[\pi^T_\psi(\cdot\mid q,c,y_{i,<t})\big]\Big).\)
  - **teacher EMA 软更新**(Eq.11,τ=0.05,近似 Eq.2):\( \psi\leftarrow(1-\tau)\psi+\tau\theta,\ \tau\in(0,1). \)
  直觉:结果奖励无偏但高方差,teacher 逐 token 指导有偏但低方差(Schulman 2016 / Gu 2016 的 bias-variance 视角),二者组合改善信用分配。teacher 看得到 \(c\)、student 看不到,所以 teacher 的"下一 token 分布"对 student 是有信息的软标签。【原文 Eq.(1)–(11)+Fig.2】
- 实验与证据:
  - 数据集/设置:**完全 data-free**(无任何外部 demo/问题/答案)。基座 Qwen3-4B / Qwen3-4B-Instruct / Qwen3-8B。评测 3 个 General QA(NQ/TriviaQA/PopQA)+ 4 个 Multi-hop(HotpotQA/2WikiMQA/MuSiQue/Bamboogle),exact-match,检索器 E5-base,英文 Wikipedia dump。基线:ReAct(training-free)、Search-R1/ToolForge(监督 RL)、Dr.Zero/SQLM*(自博弈)。【§3.1】
  - 关键数字:平均较 **Search-R1 +6.3%/+4.2%/+15.4%**(4B/4B-Instruct/8B);Total 分如 Qwen3-8B π-Play **280.3** vs Dr.Zero 274.9 vs Search-R1 242.8(Table 1)。效率:首轮(Iter1)即追平/超过 Dr.Zero 三轮收敛(Table 2),号称 2–3× 演化效率(Fig.3 显示 reward/entropy 在 Iter3 收敛)。
  - baseline 公平性:统一检索器/语料/exact-match,基线含监督 RL 与自博弈两类,较公平;但报告的是"best-performing training iteration 的 checkpoint"(§3.1),略有选择性。
  - 看着强但没回答的:"2–3× 效率"依赖与 Dr.Zero 同设定可比;且全部局限于英文 Wikipedia QA + exact-match,未验证更开放的搜索任务。
- 假设与失效边界:【原文】data-free 设定严格(§2.1);teacher 用 EMA 近似 ∵ 直接优化 Eq.2 开销大(§2.4)。【推断】① QCP 有用的前提是"出题=逆向解题",即 examiner 必须真的边搜索边构造、QCP 含解题线索;若题目并非检索可解(如纯数学推导),QCP 的"特权"含金量存疑——本文只在检索式 QA 上验证。② teacher=student 的 EMA,其"特权能力"全来自条件于 \(c\);一旦 student 自己已很强、\(c\) 提供的边际信息变小,蒸馏增益会衰减(论文用 λ 衰减调度恰好顺应了这一点,但未单独验证大规模下的稳定性)。
- 祛魅总结:【推断】真贡献=指出"自博弈造题副产物 QCP 是免费特权信息"并把它接进自蒸馏,这一因果链清晰且新颖,Table 3 给了硬证据。包装/需注意:① "teacher"听起来像独立强模型,实则是 student 的 EMA(Eq.11),**没有独立训练的更强教师**,容易被高估为"师生异构蒸馏";② "black-box/data-free"成立,但增益规模(8B 上 +15.4% 较显著,4B-Instruct 仅 +4.2%)对基座敏感,小基座增益有限;③ 论文给出 Eq.2 理想 teacher 目标却不优化它,存在"理想 vs 实现"落差(论文坦承)。

## 结构化抽取
- 🎯 机制速览6轴:
  - **学什么信号**:teacher(看得到 QCP \(c\))的逐 token 软分布(reverse-KL,Eq.10),叠加结果正误的 0/1 outcome reward(GRPO,Eq.7-9)。
  - **改什么**:student 策略参数 \(\theta\)(端到端梯度更新);examiner \(\phi\) 也在演化(RL 训,Eq.5),teacher \(\psi\) 用 EMA(Eq.11)。
  - **何时改**:训练期在线、on-policy(student 自采 rollout 上算蒸馏);三角色交替(每轮 examiner W=50 步→生成数据→student W=50 步,期间 teacher 每 step EMA 软更新)。
  - **免梯度?**:否。student 与 examiner 均梯度更新;teacher 免梯度(EMA 参数滑动平均,τ=0.05)。
  - **记忆-技能生命周期**:无显式外部记忆/技能库;"技能"通过 RL+蒸馏固化进 student 参数;课程(难度)通过 examiner-student 协同动态演化(Table 4)。
  - **防遗忘机制**:无专门 continual-learning 防遗忘设计;靠 \(-\beta D_{\mathrm{KL}}(\pi^S_\theta\|\pi_{\mathrm{ref}})\) 锚定 base(Eq.7)与 λ 衰减软化后期蒸馏,间接限制漂移。【推断:论文未以"防遗忘"为名,但 KL-to-ref 起到此作用】
- ⑦ 开源代码+框架/harness:仓库 https://github.com/zhyaoch/pi-play(README 称 "Code will be released soon")。**实际克隆(2026-06-02 复核)只有 README.md + images/,TODO 中 "Release code" 仍未勾选,训练代码尚未发布,框架/搜索工具栈/reverse-KL 实现均无法核验**【待核:完整训练代码待官方释出】。论文正文未绑定具体开源训练框架;student 用 GRPO、examiner 用 hop-grouped relative policy optimization、teacher 用 EMA。
- 💰 资源/成本与可扩展性:训练极短——共 3 轮、每轮 examiner W=50 步 + student W=50 步 = **150 步**,远少于 Search-R1 等(§3.2/Algorithm 1)。声称演化效率较传统自博弈 2–3×。具体 GPU 卡数/时长正文未给(称见 Appendix C 训练成本)〔待核:Appendix C 具体数值未在主文抽到〕。
- 🎯 对"探索-巩固"对标:**强支撑**。π-Play 把"teacher 当稀疏脚手架教 student"具象化了——QCP 正是 teacher 的"前瞻/脚手架"信息,student on-policy 自采 rollout(自选路径)、teacher 逐 token 软指导(选路/回轨),λ 衰减=脚手架逐步撤除(从依赖 teacher 到自主),与"探索→巩固"节律同构。可借组件:① 把"构造路径/逆向轨迹"当免费特权前瞻信号的思路,可类比 MTP 的前瞻探针;② "teacher=student 的 EMA + 条件于特权上下文"(Eq.10-11)是低成本提供稳定脚手架的工程范式,可直接迁移到 TSRD 的 teacher 端;③ Eq.7 的"outcome-reward + λ·distill"联合目标 + λ 衰减调度,是"探索(RL)与巩固(蒸馏)按课程配比"的现成模板。缺口:无外部记忆/技能库、无显式防遗忘、仅检索式 QA 验证,与 mtp_opd 想要的"MTP 前瞻 + 参数化巩固"还隔一层(此处前瞻=文本 QCP 而非 MTP 概率探针)。一句判定:理念高度对标(脚手架+撤除),但载体是检索 QA 的文本特权信息,非 MTP/通用 agent,作可借范式而非直接基线。
- 🔭 开放问题/未来方向:【原文】扩展到更复杂搜索任务、更大规模。【推断】① EMA teacher vs 直接优化 Eq.2 在更大模型/更长训练下是否仍稳定占优——本文无对照,值得补;② QCP 在"非检索可解"任务(数学/代码)上是否仍是有效特权信息;③ 把 QCP 这类"逆向构造轨迹"与 MTP/前瞻概率信号结合,做更细粒度的 path-selection 监督;④ 多轮搜索的信用分配是否能用 QCP 做 step 级(而非全程 token 级)切分。

— RETURN —
pi_play | 读PDF? 是(23页,§1-§4+Appendix B/C/D,Eq.1-11 逐字核对) | 加厚? 是(Eq.1-11 全转 MathJax:理想 teacher Eq.2 / 难度奖励 Eq.4 / hop-grouped Eq.5-6 / student 联合 Eq.7-9 / 蒸馏 Eq.10 / EMA Eq.11;相关工作扩到 4 类精确对位;§-anchor 替换原行号锚) | LaTeX公式条数:11(三角色 Eq.1/2/3 + 难度奖励 Eq.4 + examiner Eq.5 + 优势 Eq.6 + student Eq.7 + PPO-clip Eq.8 + 权重/优势 Eq.9 + 蒸馏 Eq.10 + EMA Eq.11) | 待核数:2(Appendix C 具体 GPU 成本未抽到;完整训练代码未发布无法核验框架/实现)

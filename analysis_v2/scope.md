scope | SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting | 中科大 + 美团 LongCat Interaction + 南大 + 复旦 + 华科(Binbin Zheng/Xing Ma 共同一作,实习期间完成;通讯 Xing Ma/Benchang Zhu/Xunliang Cai) | 2026-05-30 v2 · arXiv preprint(cs.LG) | L1 OPD/自蒸馏(+L3 GRPO 元素) · 相关性中

**原始论文**:https://arxiv.org/abs/2604.10688

## 一眼看懂
- 🟦 TL;DR:标准 OPD 把 teacher 的稠密 token-level KL 监督**一视同仁**地施加到所有 rollout,忽视信号质量差异。SCOPE 按轨迹正确性把 on-policy rollout 路由成两条互补路径:**正确轨迹**走 student-PPL 加权的 MLE 自强化(把强化集中到能力边界的低置信/高 student-PPL 样本,避免过度强化已掌握的);**错误轨迹**走 teacher-PPL 加权的反向 KL 蒸馏(只信 teacher 真有把握即低 teacher-PPL 的纠错,过滤被坏前缀诱导的噪声)。两路均在同 prompt 组内做 PPL 归一化(group-level softmax)。【原文】Abstract、§3
- 最巧的一步:**按正确性的双路路由(outcome-driven branching)+ 两路用相反的 PPL 加权方向**。抽掉路由,退化成"全轨迹一个 loss",就回到被它批判的标准 OPD(消融 w/o DPAW:AIME25 Pass@32 50.9→45.7);抽掉"两路加权方向相反"(student 高 PPL 升权 vs teacher 高 PPL 降权),则 §2 两个经验发现(多样性退化、坏前缀陷阱)无法被分别针对——尤其**反转 teacher-guided weight 致 AIME24 Avg@32 从 42.7 暴跌到 38.6**(Fig.5)。这步是全文的支点。【原文】§3.1、Eq.4-5、Fig.5

## 为什么做
- **研究背景的来龙去脉**:RLVR(DeepSeek-R1/GRPO/DAPO)是推理对齐主流,用 deterministic outcome verifier 给无歧义信号、防 reward hacking;但稀疏 outcome-level 奖励只在长轨迹**末端**发放,使 token 级 credit assignment 极难、收敛慢,小模型尤甚。PRM(过程奖励模型)虽给 step-wise 反馈但需昂贵人工标注、跨域泛化差。这逼出"从强 teacher 蒸馏稠密 token 监督"这条路。OPD(on-policy distillation,Lu & Lab 2025)用 teacher 在 student 自采 rollout 上的 token-level reverse-KL 监督,兼顾分布一致与训练效率。【原文】§1、§5.1
- **并行技术路线 + 各自具体短板**:
  - **GRPO/RLVR**:纯 outcome reward、无 token 监督、收敛慢、credit assignment 难。
  - **off-policy KD**(Kim & Rush;Hinton):在 teacher 生成的静态序列上 SFT/logit 对齐 → exposure bias + 分布失配。
  - **标准 OPD**:on-policy + reverse-KL,缓解失配、收敛更强——但**均匀加权**,假设 teacher 稠密监督在所有 rollout 上一致可靠。
  - **RL-KD 混合**:KDRL(联合优化 reward + KL)、RLAD/REOPOLD(动态 reward shaping 注入 teacher 信号)——但**仍隐含"teacher 监督处处可靠"假设**,忽视 teacher 可能"自信地错"(confidently wrong),使无差别蒸馏成为 confirmation bias 的传播载体。SCOPE 直接以"OPD 均匀加权 / teacher 可靠性同质"为靶子。【原文】§5.1-5.2、§1
- **解决的具体痛点(§2 两个经验前置研究)**:OPD 假设 teacher 稠密监督处处可靠,实则两处出问题——
  - **(2.1) Pass@k Paradox(多样性退化)**:均匀强化正确轨迹会放大主导推理模式、边缘化稀有有效路径。实证:PSR(正样本强化)在 Qwen2.5-7B 上 Pass@32 从 93.7→84.9;OPD 在 1.5B 上 Pass@32 76.5→75.0(Fig.1a)——均匀优化牺牲多样性致 mode collapse。
  - **(2.2) Flawed Prefix Trap(纠错低效)**:on-policy 下 teacher 被 condition 在 student 的坏前缀上,若前缀逻辑已损,teacher 指导退化为高熵噪声。实证(2000 题、student=Distill-R1-Qwen-1.5B 生成轨迹、teacher=Skywork-OR1-7B 算 PPL 分桶、截断前缀让 teacher 续写):低 teacher-PPL(Q1,avg PPL=1.36)recovery 率显著高于高 PPL(Q4,avg PPL=2.38),差距 up to **+19.4%**;且截断比越高 recovery 越低(80% 截断时最好组也仅约 35%,Fig.1b)。【原文】§2.1、§2.2、Fig.1
- **动机链**:OPD 信号质量异质 → 标准 OPD 无"信号质量感知"→ §2 证两病灶 → **PPL 可作信号质量探针**(teacher-PPL↔坏前缀上 teacher 是否可信;student-PPL↔是否在能力边界)→ 按正确性分路 + 各自 PPL 加权(方向相反)。【原文】§2 结尾
- **与最近邻(标准 OPD)的精确差异**:相比标准 OPD 的均匀 token-level KL,关键差是"**用 PPL 探针做轨迹级信号质量加权 + 正确/错误分路用不同监督源(自身 MLE vs teacher KL)**"——把"什么时候该信 teacher、什么时候该强化自己"显式化。【原文】§3

## 怎么做 + 靠不靠谱
- **方法流水线(输入→输出,两阶段)**:
  1. **Stage 1 — Outcome-Driven Group Branching**(§3.1):on-policy 对每 prompt \(x\) 采 \(N\) 条 rollout \(Y^x=\{y_1,\dots,y_N\}\),verifier 给二元奖励 \(R_i\in\{0,1\}\) → 切成正确集 \(\Omega^x_c=\{y_i\mid R_i=1\}\) 与错误集 \(\Omega^x_w=\{y_i\mid R_i=0\}\)。两条轨迹的 surrogate(用 token 级 IS ratio \(\rho_{i,t}(\theta)=\frac{\pi_\theta(a_{i,t}\mid x,a_{i,<t})}{\pi_{\mathrm{old}}(a_{i,t}\mid x,a_{i,<t})}\) 做 off-policy 校正):
     - 正确路(Valid Trajectory Exploitation):MLE 自强化 \(\mathcal{L}_{\mathrm{MLE}}(x,y_i;\theta)=-\sum_{t=1}^{|y_i|}\rho_{i,t}(\theta)\)(Eq.2)。
     - 错误路(Flawed Trajectory Rectification):把 detached log-ratio 当负 advantage 的 reverse-KL 蒸馏 \(\mathcal{L}_{\mathrm{OPD}}(x,y_i;\theta)=\sum_{t=1}^{|y_i|}\rho_{i,t}(\theta)\big(\log\pi_{\bar\theta}(a_{i,t}\mid x,a_{i,<t})-\log\pi_T(a_{i,t}\mid x,a_{i,<t})\big)\)(Eq.3,\(\bar\theta\)=detach)。
     **输出**:两个带各自 loss 形式的轨迹子集。
  2. **Stage 2 — Dual-Path Adaptive Weighting(DPAW)**(§3.2):用序列 PPL \(\mathrm{PPL}(y_i\mid x)=\exp(-\frac{1}{|y_i|}\log\pi(y_i\mid x))\) 量化"轨迹意外度",在**各组内**做 group-relative softmax 得权重(方向相反):
     - **Student-guided weight(放大非常规有效路径)**,只作用于正确轨迹:
       \(\displaystyle w^{\mathrm{stu}}_i=\frac{\exp\!\big(-\frac{1}{\tau|y_i|}\log\pi_S(y_i\mid x)\big)}{\sum_{j\in\Omega^x_c}\exp\!\big(-\frac{1}{\tau|y_j|}\log\pi_S(y_j\mid x)\big)}=\frac{\mathrm{PPL}_S(y_i\mid x)^{1/\tau}}{\sum_{j\in\Omega^x_c}\mathrm{PPL}_S(y_j\mid x)^{1/\tau}},\quad\forall i\in\Omega^x_c.\)
       即 **student PPL 越高(越在能力边界)权重越高** → 缓解 §2.1 多样性退化。
     - **Teacher-guided weight(过滤坏前缀噪声)**,只作用于错误轨迹:
       \(\displaystyle w^{\mathrm{tea}}_i=\frac{\exp\!\big(\frac{1}{\tau|y_i|}\log\pi_T(y_i\mid x)\big)}{\sum_{j\in\Omega^x_w}\exp\!\big(\frac{1}{\tau|y_j|}\log\pi_T(y_j\mid x)\big)}=\frac{\mathrm{PPL}_T(y_i\mid x)^{-1/\tau}}{\sum_{j\in\Omega^x_w}\mathrm{PPL}_T(y_j\mid x)^{-1/\tau}},\quad\forall i\in\Omega^x_w.\)
       即 **teacher PPL 越高(对坏前缀越没把握)权重越低** → 应对 §2.2 坏前缀陷阱。
     **输出**:每条轨迹的自适应权重。
  3. **Stage 3 — 统一目标**(§3.3):
     \(\displaystyle \mathcal{J}_{\mathrm{SCOPE}}=\mathbb{E}_{x\sim\mathcal{D}}\Big[\sum_{i\in\Omega^x_c}w^{\mathrm{stu}}_i\cdot\mathcal{L}_{\mathrm{MLE}}(x,y_i)+\sum_{i\in\Omega^x_w}w^{\mathrm{tea}}_i\cdot\mathcal{L}_{\mathrm{OPD}}(x,y_i)\Big].\)
     group-level 归一化保证 \(\sum_{i\in\Omega}w_i=1,\ 0<w_i<1\)(Eq.18-19),应对 prompt 间难度方差。**输出**:加权后的统一梯度更新。
- **逐组件必要性(基于真实消融 Fig.5,AIME24/25)**:
  - **outcome-driven 分路 / 整个 DPAW**:w/o DPAW(退回均匀加权)→ AIME25 Pass@32 从 50.9 暴跌到 45.7,证均匀加权未能最优利用 rollout。【原文】§4.3、Fig.5
  - **Student Path**:w/o student-guided weight → AIME24 Pass@32 从 77.9 掉到 74.1(多样性受损);**反转方向**(高 student-PPL 反降权)也伤性能。【原文】§4.3、Fig.5
  - **Teacher Path**:w/o teacher-guided weight 伤整体精度;**反转方向**(高 teacher-PPL 反升权)致 AIME24 Avg@32 从 42.7 暴跌 38.6——证动态惩罚不可靠 teacher 指导确实过滤了坏前缀噪声。【原文】§4.3、Fig.5
  - **组内 PPL 归一化 + \(\tau\)**:有 τ 消融(Table 6)与 entropy/Avg@32 动力学图(Fig.3);但**两路各自的剥离消融(只 student-path / 只 teacher-path 单独跑)未在主文给出**,难判定各路独立贡献。【原文】§3.3、Fig.3
- **关键机制/公式(直觉 + 理论支撑)**:PPL=\(\exp(-\)平均 \(\log p)\),即"模型对这条轨迹有多意外"。Student 路用"学生越意外(越在能力边界)越该学";Teacher 路用"teacher 越意外(对坏前缀越没把握)越不该信"。两方向相反但都服务"挑高质量信号"。附录 A 给三段理论 motivation(非新数学,是直觉化论证):(A.1)由 Cauchy 不等式 \(\mathbb{E}[\|g_{\mathrm{OPD}}\|^2]\le|y|G^2\sum_t\mathbb{E}[A_t^2]\)(\(A_t=\log\pi_{\bar\theta}-\log\pi_T\))说明高 teacher-PPL(坏前缀)放大梯度噪声,故下调 \(w^{\mathrm{tea}}\);(A.2)\(w^{\mathrm{stu}}\propto\pi_S(y)^{-1/(\tau|y|)}\) 把加权梯度的 \(\pi_S\) 指数从 1 降到 \(1-\frac{1}{\tau|y|}\),部分抵消采样频率偏置、保稀有路径;(A.3)group-level softmax 经三角不等式稳定跨 prompt 更新尺度。【原文】§3.2、Appendix A.1-A.3、Eq.7-21
- **训推数据如何流动**:on-policy rollout(vLLM)→ verifier 二元打标 → 切 \(\Omega_c/\Omega_w\) → 正确组算 \(\mathrm{PPL}_S\)、错误组**调在线 teacher**(经 vLLM OpenAI 兼容 API,`api_interface.py`)算 \(\mathrm{PPL}_T\) 与 token 分布 → 组内 softmax 得权 → 合成 \(\mathcal{J}_{\mathrm{SCOPE}}\) 反传(带 IS ratio \(\rho\))。需额外在线 teacher 服务算 PPL,增推理开销。【原文】§3、§4.1 + 仓库
- **实验与证据**:两组师生(① teacher=Skywork-OR1-7B → student=DeepSeek-R1-Distill-Qwen-1.5B;② teacher=Qwen3-8B-Instruct → student=Qwen3-1.7B-Base),均在 DeepMath(He et al. 2025b)训练;global batch 256、max_prompt 4096、completion 12288、rollout temp 0.6、weight temp \(\tau=1.0\);lr \(5\times10^{-5}\)(GRPO/OPD/SCOPE 同,Table 3)。评测 6 数学 benchmark(AIME24/25、AMC23、MATH500、Minerva、OlympiadBench),报 Avg@32 / Pass@32(eval temp 0.6、top-p 0.95、max 32768)。
  - 摘要主口径:较竞争基线(GRPO/KD/OPD 平均)相对 **+11.42% Avg@32 / +7.30% Pass@32**。**仅相对标准 OPD**(配置①):Avg@32 47.9→52.3(OPD)→**55.2**(SCOPE),即 +5.54%;Pass@32 73.1→75.0,+2.60%(逐项 AIME24 A@32 40.2→42.7、Olympiad A@32 44.9→49.7 等);配置② +6.21%/+4.83%(其中 Minerva P@32 -2.77%、AIME24 P@32 +0.00%,有少数项不增或微降,如实报告)。【原文】§4.1-4.2、Table 1
  - 扩展(Table 2,代码):HumanEval/Codeforces/LiveCodeBench,相对 OPD Avg@32 +4.69%、相对 GRPO +4.21%,证跨域可用。【原文】§4.4、Table 2
  - 训练动力学(Fig.3):GRPO 熵持续衰减(Pass@k paradox 的驱动),OPD/SCOPE 维持健康熵;OPD 很快 plateau,SCOPE 持续更优。【原文】§4.2、Fig.3
  - baseline 公平性:GRPO/KD/OPD 同数据同基座,公平;但**未报多 seed 方差**,且相对 OPD 净增益(Pass@32 +2.60%)偏小,统计稳健性存疑。【推断】
- **假设与失效边界**:
  - 【原文】"低 teacher PPL ↔ 成功 error recovery"是**经验相关**(Fig.1b 分桶),非因果;论文未给反事实验证。
  - 【原文·Limitation】依赖自动可验证 outcome 做路由(数学/代码),开放/主观任务需额外 reward model;算力受限,未验证大模型/MoE。
  - 【推断】Student Path 隐患:"student 高 PPL = 能力边界有效路径" 与 "= 错得自信/噪声" 难区分;虽只作用于**正确**轨迹一定程度缓解,但仍可能放大偶然走对的噪声轨迹。
  - 【推断】teacher PPL 需 teacher 在线/可查(经 vLLM API),有部署成本。
- **祛魅总结**:
  - 真贡献:把"OPD 均匀加权"这一隐含假设显式拆成"正确/错误 × 该信谁"两个子问题,并用 PPL 探针给出轻量加权方案;§2 两个经验现象(尤其 Flawed Prefix Trap 的 +19.4% recovery gap)本身有诊断价值。【推断】
  - 包装/高估:摘要的 +11.42%/+7.30% 是相对"GRPO/KD/OPD 取平均"的口径,**相对最强基线 OPD 的净增益要小得多(+5.54%/+2.60%)**;无方差报告。【推断】
  - **代码-论文一致性风险(沿用 v1 仓库核查,本轮 PDF 仅确认公式方向)**:v1 对发布仓 `verl/trainer/ppo/ray_trainer.py:_compute_scope_dual_path_weights` 做过 softmax 方向数值验证,结论:两脚本均设 `STUDENT_PATH_PPL_POSITIVE=True / TEACHER_PATH_PPL_POSITIVE=True`,据 `sign=+1 if ppl_positive else −1` 推导——Teacher 路(=True)与 Eq.5 一致(正确),但 **Student 路(=True)实际让高 student-PPL 反被降权,与 Eq.4「放大能力边界样本」相反**;复现时应把 student_path_ppl_positive 改 False 才符合论文。另:论文 §4.1 写 max_prompt=4096,脚本实设 2048。〔待核:发布 HF 权重按哪组 sign 训练无法静态确认,建议跑权重对齐实验定性。〕本轮 PDF 内容与 v1 仓库结论无冲突。【推断·依据 v1 verify】

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号:轨迹级 PPL(\(\mathrm{PPL}_S\)=是否在能力边界;\(\mathrm{PPL}_T\)=teacher 在坏前缀上是否可信)+ 二元 outcome reward 做路由。
  - 改什么:OPD/MLE 的**样本/轨迹级权重**(\(w^{\mathrm{stu}}/w^{\mathrm{tea}}\),不改 token 选择,改整条轨迹的损失权重)。
  - 何时改:每步在线,按当前 batch 的 rollout 组内动态归一(group-relative softmax,非离线固定)。
  - 免梯度?否,梯度训练(正确路 MLE、错误路 reverse-KL OPD,均带 IS ratio \(\rho\))。
  - 记忆-技能生命周期:无外部记忆/技能库,纯参数内。
  - 防遗忘机制:无显式防遗忘;Student Path 的"放大非常规有效路径"间接缓解 RL 常见的多样性坍缩(mode collapse)。
- ⑦ 开源代码+框架/harness:https://github.com/machine981/SCOPE(v1 已 clone ~9.1MB);权重 HF `Machine981/SCOPE-Qwen3-1.7B`、`Machine981/SCOPE-Deepseek-R1-Distill-Qwen-1.5B`。框架=**veRL(volcengine/verl)**,蒸馏逻辑整合在仓库内 `verl/`;teacher 经 vLLM(`deploy_vllm.sh`)以 OpenAI 兼容 API 提供,verl 经 `verl/utils/api_interface.py` 调用(api-key 须一致),支持多节点 IP_POOL。入口 `run_experiment_distill_1_5b.sh` / `run_experiment_qwen_1_7b.sh`;双路开关 `USE_SCOPE_DUAL_PATH_WEIGHTING`。【原文】§4.1 + 仓库(v1 核查)
- 💰 资源/成本与可扩展性:global batch 256、completion 12288、eval max 32768、lr \(5\times10^{-5}\);需额外在线 teacher(vLLM 服务)算 PPL,增推理开销;具体 GPU 数论文未在主文明确。【原文】§4.1、Table 3(其余原文未说明)
- 🎯 对"探索-巩固"对标:**强对标(支撑+部分竞品)**。判定:SCOPE 的双路就是本项目"探索-巩固"二分的一个具体实例——**正确路(\(w^{\mathrm{stu}}\) 加权 MLE)≈ 巩固/固化"自己走得通的路径",且偏向能力边界(对应"偏向自己能走通的开头");错误路(\(w^{\mathrm{tea}}\) 加权 KL)≈ 走偏后由 teacher 接管纠错(对应 path-recovery)**。可借组件:用 PPL 做"信号质量/能力边界"探针的思路,可迁移到 TSRD 决定"哪条轨迹该自蒸馏巩固、哪条该让 teacher 接管";尤其 §2.2 "坏前缀越长 recovery 越低"对本项目"在哪个点接管"有直接参考价值(接管点越靠后、前缀越脏,teacher 越难救)。缺口/与本项目的差异:SCOPE 是**整条轨迹**级路由与加权(粗粒度),没有 token/step 级的"第一处走偏点"定位与"单点接管"(本项目核心),也无 MTP 前瞻;且 teacher 接管是 full-trajectory KL 而非"从出错后缀起"。依据:§3.1 分路设计 + Eq.4/5。【推断】
- 🔭 开放问题/未来方向:【原文·Limitation】更广任务普适性(已用代码 benchmark 初步扩展);大模型/MoE/更多可验证域。【推断】PPL→信号质量的因果验证;把轨迹级加权细化到 token/step 级(与"单点接管"结合);多 seed 方差与统计显著性;修复并验证 Student Path 代码方向。

RETURN: scope|读到PDF?是(18页,_txt 65k字,Eq.1-6 + Appendix A 理论 Eq.7-21 + Fig.5 消融全核)|L1(+L3)|对标=强对标:正确路=巩固/错误路=teacher 接管纠错 + 坏前缀越长越难救(接管点参考),但为轨迹级粗粒度、缺 token 级单点接管与 MTP|残留待核 1(发布 HF 权重实际训练所用 sign,需权重对齐实验)

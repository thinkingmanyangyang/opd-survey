scope | SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting | 中科大 + 美团 LongCat Interaction + 南大 + 复旦 + 华科(Binbin Zheng/Xing Ma 共同一作,实习期间完成;通讯 Xing Ma/Benchang Zhu/Xunliang Cai) | 2026-05-30 v2 · arXiv preprint(cs.LG) | L1 OPD/自蒸馏(+L3 GRPO 元素) · 相关性中

**原始论文**:https://arxiv.org/abs/2604.10688

## 一眼看懂
- 🟦 TL;DR:标准 OPD 把 teacher 的稠密 token-level KL 监督**一视同仁**地施加到所有 rollout,忽视信号质量差异。SCOPE 按轨迹正确性把 on-policy rollout 路由成两条互补路径:**正确轨迹**走 student-PPL 加权的 MLE 自强化(把强化集中到能力边界的低置信/高 student-PPL 样本,避免过度强化已掌握的);**错误轨迹**走 teacher-PPL 加权的反向 KL 蒸馏(只信 teacher 真有把握即低 PPL 的纠错,过滤被坏前缀诱导的噪声)。两路均在同 prompt 组内做 PPL 归一化(group-level softmax)。【原文】Abstract、§3
- 最巧的一步:**按正确性的双路路由(outcome-driven branching)+ 两路用相反的 PPL 加权方向**。抽掉路由,退化成"全轨迹一个 loss",就回到被它批判的标准 OPD;抽掉"两路加权方向相反"(student 高 PPL 升权 vs teacher 高 PPL 降权),则 §2 两个经验发现(多样性退化、坏前缀陷阱)无法被分别针对。这步是全文的支点。【原文】§3.1、Eq.4-5

## 为什么做
- 研究背景:on-policy RL(DeepSeek-R1/GRPO/DAPO)是推理对齐主流,但稀疏 outcome-level 奖励使 token 级 credit assignment 难、收敛慢;OPD 用 teacher 稠密 token-level KL 监督缓解,兼顾分布一致与训练效率。【原文】§1
- 解决的具体痛点:OPD 假设 teacher 稠密监督在所有 rollout 上一致可靠,实则两处出问题——(1) 错误轨迹上 teacher 若也不熟(高 teacher PPL),其 token 分布是不可靠信号;(2) 正确轨迹上严格匹配 teacher KL 会压制学生有效但非常规的推理路径,而等权 MLE 又过度强化已稳掌握样本、边缘化能力边界的低置信样本。【原文】§1、§2
- 相关工作 & 各自不足:GRPO(纯 outcome reward,无 token 监督、收敛慢);KD(离线 teacher 序列上 SFT,off-policy 分布失配);标准 OPD(Lu & Lab 2025,dense KL 但均匀加权)。SCOPE 直接以"OPD 均匀加权"为靶子。【原文】§1、§4.1
- 动机链:OPD 信号质量异质 → 标准 OPD 无"信号质量感知"→ §2 证两病灶(正确路多样性退化、错误路纠错低效)→ PPL 可作信号质量探针 → 按正确性分路 + 各自 PPL 加权。【原文】§2 结尾
- 与最近邻工作的Δ:相比标准 OPD,关键差是"**用 PPL 探针做 token/轨迹级信号质量加权 + 正确/错误分路用不同监督源(自身 MLE vs teacher KL)**"。有用之处:把"什么时候该信 teacher、什么时候该强化自己"显式化。【原文】§3

## 怎么做 + 靠不靠谱
- 方法流水线:① on-policy 对每 prompt 采 N 条 rollout,verifier 给二元奖励 → 按正确性切 Ω_c(正确)/ Ω_w(错误)→ ② 正确组算 student 序列 PPL、错误组算 teacher 序列 PPL → ③ 各组内做 group-relative softmax 得权重(Eq.4/5)→ ④ 合成统一目标 J_SCOPE = Σ_{Ω_c} w_i^stu·L_MLE + Σ_{Ω_w} w_i^tea·L_OPD(Eq.6),用 importance-sampling ratio ρ 做 off-policy 校正。【原文】§3.1-3.3
- 逐组件必要性:
  - **outcome-driven 分路**:必要,是框架骨架。无消融单独测,但两路目标天然互补。【原文】§3.1
  - **Student Path(Eq.4)**:w_i^stu ∝ PPL_S(y_i)^{1/τ} 组内归一 ⇒ **student PPL 越高权重越高**,放大能力边界非常规有效路径,缓解多样性退化(§2.1 Pass@k Paradox)。【原文】Eq.4
  - **Teacher Path(Eq.5)**:w_i^tea ∝ PPL_T(y_i)^{−1/τ} 组内归一 ⇒ **teacher PPL 越高权重越低**,过滤坏前缀诱导噪声(§2.2)。【原文】Eq.5
  - **组内 PPL 归一化**:应对 prompt 间难度方差;有 τ 消融(Table 6)与 entropy/Avg@32 动力学图(Fig.3),但**两路各自的剥离消融(只 student-path / 只 teacher-path)未在主文给出**,难判定各路独立贡献。【原文】§3.3、Fig.3
- 关键机制/公式(直觉):PPL=exp(−平均 logp),即"模型对这条轨迹有多意外"。Student 路用"学生越意外(越在能力边界)越该学";Teacher 路用"teacher 越意外(对坏前缀越没把握)越不该信"。两个方向相反但都服务"挑高质量信号"。【原文】§3.2
- 实验与证据:两组师生(① teacher=Skywork-OR1-7B → student=DeepSeek-R1-Distill-Qwen-1.5B;② teacher=Qwen3-8B-Instruct → student=Qwen3-1.7B-Base),均在 DeepMath(He et al. 2025b)训练;global batch 256、max_prompt 4096、completion 12288、rollout temp 0.6、weight temp τ=1.0。评测 6 数学 benchmark(AIME24/25、AMC23、MATH500、Minerva、OlympiadBench),报 Avg@32 / Pass@32(eval temp 0.6、top-p 0.95、max 32768)。摘要主口径:较竞争基线(GRPO/KD/OPD 平均)相对 +11.42% Avg@32 / +7.30% Pass@32。**仅相对标准 OPD**(配置①):Avg@32 47.9→52.3(OPD)→**55.2**(SCOPE),即 +5.54%;Pass@32 +2.60%(逐项 AIME24 A@32 40.2→42.7、Olympiad 44.9→49.7 等);配置② +6.21%/+4.83%(其中 Minerva Pass@32 -2.77%、AIME24 P@32 +0.00%,有少数项不增或微降,如实报告)。【原文】§4.1、Table 1
  - 经验前置研究(§2,从 DeepMath 采 2000 题、student=Distill-R1-Qwen-1.5B 生成轨迹、teacher=Skywork-OR1-7B 算 PPL 分桶):(a) Pass@k Paradox——PSR 在 Qwen2.5-7B 上 Pass@32 从 93.7→84.9;OPD 在 1.5B 上 Pass@32 76.5→75.0,证均匀优化牺牲多样性。(b) Flawed Prefix Trap——低 teacher PPL(Q1)的截断前缀 recovery 率显著高于高 PPL(Q4),差距 up to +19.4%;且截断比越高 recovery 越低(80% 截断时最好组也仅约 35%)。【原文】§2.1、§2.2、Fig.1
  - baseline 公平性:GRPO/KD/OPD 同数据同基座,公平;但**未报多 seed 方差**,且相对 OPD 净增益(Pass@32 +2.60%)偏小,统计稳健性存疑。【推断】
- 假设与失效边界:
  - 【原文】"低 teacher PPL ↔ 成功 error recovery"是**经验相关**(Fig.1b 分桶),非因果;论文未给反事实验证。
  - 【推断】Student Path 的隐患:"student 高 PPL = 能力边界有效路径" 与 "student 高 PPL = 错得自信/噪声" 难区分;虽只作用于**正确**轨迹一定程度缓解,但仍可能放大偶然走对的噪声轨迹。
  - 【推断】依赖可靠 verifier(二元奖励)做路由;无 verifier 的开放任务不适用。teacher PPL 需 teacher 在线/可查(经 vLLM OpenAI 兼容 API),有部署成本。
- 祛魅总结:
  - 真贡献:把"OPD 均匀加权"这一隐含假设显式拆成"正确/错误 × 该信谁"两个子问题,并用 PPL 探针给出轻量加权方案;§2 两个经验现象(尤其 Flawed Prefix Trap)本身有诊断价值。【推断】
  - 包装/高估:摘要的 +11.42%/+7.30% 是相对"GRPO/KD/OPD 取平均"的口径,**相对最强基线 OPD 的净增益要小得多(+5.54%/+2.60%)**;无方差报告。【推断】
  - **代码-论文一致性风险(沿用 v1 仓库核查,本轮 PDF 仅确认公式方向)**:v1 对发布仓 `verl/trainer/ppo/ray_trainer.py:_compute_scope_dual_path_weights` 做过 softmax 方向数值验证,结论:两脚本均设 `STUDENT_PATH_PPL_POSITIVE=True / TEACHER_PATH_PPL_POSITIVE=True`,据 `sign=+1 if ppl_positive else −1` 推导——Teacher 路(=True)与 Eq.5 一致(正确),但 **Student 路(=True)实际让高 student-PPL 反被降权,与 Eq.4「放大能力边界样本」相反**;复现时应把 student_path_ppl_positive 改 False 才符合论文。另:论文 §4.1 写 max_prompt=4096,脚本实设 2048。〔待核:发布 HF 权重按哪组 sign 训练无法静态确认,建议跑权重对齐实验定性。〕本轮 PDF 内容与 v1 仓库结论无冲突。【推断·依据 v1 verify】

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号:轨迹级 PPL(student PPL=是否在能力边界;teacher PPL=teacher 在坏前缀上是否可信)+ 二元 outcome reward 做路由。
  - 改什么:OPD/MLE 的**样本/轨迹级权重**(不改 token 选择,改整条轨迹的损失权重)。
  - 何时改:每步在线,按当前 batch 的 rollout 组内动态归一(group-relative,非离线固定)。
  - 免梯度?否,梯度训练(正确路 MLE、错误路 reverse-KL OPD,均带 IS ratio)。
  - 记忆-技能生命周期:无外部记忆/技能库,纯参数内。
  - 防遗忘机制:无显式防遗忘;Student Path 的"放大非常规有效路径"间接缓解 RL 常见的多样性坍缩(mode collapse)。
- ⑦ 开源代码+框架/harness:https://github.com/machine981/SCOPE(v1 已 clone ~9.1MB);权重 HF `Machine981/SCOPE-Qwen3-1.7B`、`Machine981/SCOPE-Deepseek-R1-Distill-Qwen-1.5B`。框架=**veRL(volcengine/verl)**,蒸馏逻辑整合在仓库内 `verl/`;teacher 经 vLLM(`deploy_vllm.sh`)以 OpenAI 兼容 API 提供,verl 经 `verl/utils/api_interface.py` 调用(api-key 须一致),支持多节点 IP_POOL。入口 `run_experiment_distill_1_5b.sh` / `run_experiment_qwen_1_7b.sh`;双路开关 `USE_SCOPE_DUAL_PATH_WEIGHTING`。【原文】§4.1 + 仓库(v1 核查)
- 💰 资源/成本与可扩展性:global batch 256、completion 12288、eval max 32768;需额外在线 teacher(vLLM 服务)算 PPL,增推理开销;具体 GPU 数论文未在主文明确。【原文】§4.1(其余原文未说明)
- 🎯 对"探索-巩固"对标:**强对标(支撑+部分竞品)**。判定:SCOPE 的双路就是本项目"探索-巩固"二分的一个具体实例——**正确路(student-PPL 加权 MLE)≈ 巩固/固化"自己走得通的路径",且偏向能力边界(对应"偏向自己能走通的开头");错误路(teacher-PPL 加权 KL)≈ 走偏后由 teacher 接管纠错(对应 path-recovery)**。可借组件:用 PPL 做"信号质量/能力边界"探针的思路,可迁移到 TSRD 决定"哪条轨迹该自蒸馏巩固、哪条该让 teacher 接管"。缺口/与本项目的差异:SCOPE 是**整条轨迹**级路由与加权(粗粒度),没有 token/step 级的"第一处走偏点"定位与"单点接管"(本项目核心),也无 MTP 前瞻;且 teacher 接管是 full-trajectory KL 而非"从出错后缀起"。依据:§3.1 分路设计 + Eq.4/5。【推断】
- 🔭 开放问题/未来方向:【原文】更广任务的普适性(已用代码 benchmark 初步扩展);τ 与归一化粒度的鲁棒性(部分消融)。 【推断】PPL→信号质量的因果验证;把轨迹级加权细化到 token/step 级(与"单点接管"结合);多 seed 方差与统计显著性;修复并验证 Student Path 代码方向。

RETURN: scope|读到PDF?是(18页,_txt 65k字)|L1(+L3)|对标=强对标:正确路=巩固/错误路=teacher 接管纠错,但为轨迹级粗粒度、缺 token 级单点接管与 MTP|残留待核 1(发布 HF 权重实际训练所用 sign,需权重对齐实验)

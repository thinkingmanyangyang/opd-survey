lightning_opd | Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation | NVIDIA(Yecheng Wu, Song Han, Han Cai;通讯 hcai@nvidia.com) | 2026-04-14 v1 / 2026-05-08 v2(据 v2)· arXiv 2604.13010 · preprint | 主题线 L1 OPD/自蒸馏 · 相关性 高(核心命中)

**原始论文**:https://arxiv.org/abs/2604.13010

## 一眼看懂
- 🟦 TL;DR:标准 on-policy distillation(OPD)要在整个训练期常驻一个大 teacher server,对学生每步新采的 rollout 在线查 teacher log-prob,GPU 被学生+教师共置而碎片化、贵、MoE 场景常 OOM。本文把 OPD **离线化**:训练前一次性用 π_ref 采 rollout、查一次 teacher 算好并缓存每 token log-prob,训练期只读缓存、在线只算学生 log-prob,**全程无 live teacher server**。核心理论贡献是识别并证明一个被忽视的前提——**teacher consistency**:SFT 阶段与 OPD 阶段必须用同一个 teacher,否则引入不可约梯度偏置(对在线/离线 OPD 都有害,离线更甚)。在该条件下,离线 OPD 与标准 OPD 共享最优点、梯度差有界(χ² 控制)、且固定 rollout 带来隐式 trust-region 正则。性能持平/略优,提速 3.6×(4B)/4.0×(8B),并让 30B MoE 从 OOM 变可行。
- 最巧的一步:**把"OPD 梯度"经重要性采样分解 \(\nabla J_{\mathrm{on}}=\mathbb E_{x\sim\pi_{\mathrm{ref}}}[w(x;\theta)\cdot f(x;\theta)]\)(\(w=\pi_\theta/\pi_{\mathrm{ref}}\)),离线版就是令 \(w\equiv 1\) 的特例**(§3.1),再配 **teacher consistency**(SFT 的 \(\pi_{\mathrm{ref}}\) 与 OPD 教师同源)。抽掉 teacher consistency:Theorem 3.8/3.9 显示梯度带持久偏置 \(G\sigma_\Delta\),Table 4 实测错配使 Lightning OPD 掉最多 6.8 分(8B)、且比标准 OPD 更敏感(标准 OPD 每步刷新 rollout 能部分恢复,离线版固定 rollout 把 \(\pi_{\mathrm{ref}}\) 的两个角色——参考分布 + rollout 源——同时污染)。这一条是"离线化为何能不掉点"的全部根基。

## 为什么做
- 研究背景与来龙去脉:大模型后训练有两大谱系。① **RLVR(可验证奖励 RL)**:行为只由稀疏 outcome reward 塑造,与 SFT 起点解耦。② **On-Policy Distillation(OPD)**:在学生**自生成 rollout** 上让学生对齐 teacher 分布,提供 **dense per-token 监督**——比起 RLVR 的稀疏信号信息量大得多,常在数学/代码上达到媲美甚至超越 RLVR 的效果(TML/Thinking Machines [9]、Qwen3 [22] 等),是本文站立的主路线。与 OPD 并列的还有 **标准离线 KD**([49,50]):用固定的 teacher 生成数据训学生,但**没有 train-inference 对齐**(学生从不在自己的分布上被纠),效果通常不如 on-policy。
- 三条路线各自的**具体**短板:标准 OPD([8,9,22])—— 要在**整个训练期常驻大 teacher server**,对学生每步新采的 rollout 在线查 teacher log-prob,学生+教师共置导致 GPU 碎片化、需多节点、成本高;MoE 学生(30B-A3B)+ 教师共置在单 8×H100 常 **OOM**。离线 KD —— 无 on-policy 对齐、效果差。ExOPD([10],一个 OPD baseline)—— 本文实测远超它(4B AIME24 68.1 vs 61.0;LCB v6 40.3 vs 29.0)。**朴素离线化**(预算缓存 teacher log-prob 复用)—— 看似显然但**无法可靠匹配标准 OPD**,作者把根因追溯到一个被忽视且现实中普遍违反的前提:**teacher consistency**(SFT 阶段生成数据的 teacher 必须 = OPD 阶段提供参考分布的 teacher)。论文点名 TML [9] 用 QwQ-32B 生成 OpenThoughts-3 做 SFT、却用 Qwen3-32B 当 OPD teacher,正是违反此前提的真实案例。
- 解决的具体痛点:① 标准 OPD 须常驻大 teacher server、GPU 碎片化、需多节点、成本高;② MoE 学生 + 教师共置单节点 OOM;③ 朴素离线化不匹配标准 OPD,根因=teacher inconsistency。
- 相关工作精确 Δ:vs 标准 OPD/ExOPD —— 共用 reverse-KL per-token advantage(Eq.2),但**把 rollout 分布固定为 \(\pi_{\mathrm{ref}}\) 并离线缓存 teacher log-prob**(Eq.4/7),靠 teacher consistency 保证不掉点;vs 离线 KD —— 离线 KD 缺乏 train-inference 对齐,而本文虽固定 rollout 却仍是"学生从 SFT 同源分布采样 + 在线算学生 log-prob 的优势加权",理论上(Thm 3.5/3.6/3.7)与在线 OPD 共享最优点、梯度差有界、并自带 trust-region;论文在 Appendix D 进一步把自己与"OOD value estimation"类工作区分(本文核心是 consistency 而非 OOD 估值)。首次给出"离线 OPD 何时可证等价标准 OPD"的理论 + 工程上彻底去掉 live teacher。
- 动机链:OPD 强但 live teacher 贵 + MoE OOM(现状/缺陷)→ 想离线化但朴素离线不匹配(障碍)→ 发现 teacher consistency 是关键前提,在其下离线 = 在线(理论)→ 去掉 teacher server、全 GPU 投学生、MoE 可行(收益)。
- 与最近邻工作的 Δ:vs 标准 OPD / ExOPD(最近邻):同样的 reverse-KL per-token advantage(Eq.2),但**把 rollout 分布固定为 \(\pi_{\mathrm{ref}}\) 并离线缓存 teacher log-prob**(Eq.4/7),用 teacher consistency 保证不掉点。关键有用点:同性能下省 3.6-4.0× GPU 时、单节点可跑、MoE 解锁,且固定 rollout 反而隐式稳训(免显式 KL 惩罚)。

## 怎么做 + 靠不靠谱
- 方法流水线(Algorithm 1,Fig.2,两阶段,具体到输入→输出):
  - **Stage 1 · SFT**:teacher \(\pi_T\) 在 prompt 集 \(\mathcal Q_{\mathrm{SFT}}\) 上生成轨迹得 \(\mathcal D_{\mathrm{SFT}}=\{(q_i,x_i)\mid x_i\sim\pi_T(\cdot\mid q_i)\}\)(Eq.5),base 模型 \(\pi_{\mathrm{base}}\) 在其上 MLE 微调得参考策略 \(\pi_{\mathrm{ref}}=\arg\max_\theta \mathbb E_{(q,x)\sim\mathcal D_{\mathrm{SFT}}}\big[\sum_t\log\pi_\theta(a_t\mid s_t)\big]\)(Eq.6)。**硬约束:\(\mathcal D_{\mathrm{SFT}}\) 必须由与 OPD 同一 \(\pi_T\) 生成**。
  - **Stage 2 · 离线 OPD**:① **预处理相**:对 \(\mathcal Q_{\mathrm{OPD}}\) 每 prompt 从 \(\pi_{\mathrm{ref}}\) 采 1 条 rollout \(x_j\),查**一次** teacher、存下每 token \(\log\pi_T(a^j_t\mid s^j_t)\),形成 \(\mathcal D_{\mathrm{OPD}}\)(Eq.7);② **训练相**:\(\pi_\theta\) 从 \(\pi_{\mathrm{ref}}\) 初始化,每步从 \(\mathcal D_{\mathrm{OPD}}\) 取 mini-batch,读缓存的 \(\log\pi_T\)、在线算 \(\log\pi_\theta\),得 advantage 并 clip 到 \([-\tau,\tau]\),按 \(\nabla J_{\mathrm{off}}\) 更新 \(\theta\leftarrow\theta+\eta\,\nabla J_{\mathrm{off}}(\theta)\)。**全程无 live teacher server**。
- 数据如何流动:teacher log-prob 是**一次性离线物化**到 \(\mathcal D_{\mathrm{OPD}}\) 的静态字段;训练时 teacher 项当常量读盘、学生项实时算——把"标准 OPD 每步查教师"的在线依赖彻底搬到预处理一次。
- 核心算法/损失(真实形式 + 直觉,符号从 PDF §3.1/§3.3 抄准):
  - **per-token reverse-KL advantage**(Eq.2):\(\ A_t(\theta)=\log\pi_T(a_t\mid s_t)-\log\pi_\theta(a_t\mid s_t)\)。直觉:teacher 比学生更自信处为正(该学)、否则为负。\(A_t\) 当**固定标量(stop-gradient)**,梯度只走 \(\nabla\log\pi_\theta\)。
  - **在线 vs 离线目标**(Eq.3/4,只差 rollout 分布):
    \[
    J_{\mathrm{on}}(\theta)=\mathbb E_{q\sim p,\,x\sim\pi_\theta}\!\Big[\textstyle\sum_{t=1}^{T}A_t(\theta)\Big],\qquad
    J_{\mathrm{off}}(\theta)=\mathbb E_{q\sim p,\,x\sim\pi_{\mathrm{ref}}}\!\Big[\textstyle\sum_{t=1}^{T}A_t(\theta)\Big].
    \]
  - **重要性采样分解(全篇支点)**:令每轨迹梯度 \(f(x;\theta)=\sum_t A_t(\theta)\nabla\log\pi_\theta(a_t\mid s_t)\)、IS 权重 \(w(x;\theta)=\pi_\theta(x)/\pi_{\mathrm{ref}}(x)\),则
    \[
    \nabla J_{\mathrm{on}}(\theta)=\mathbb E_{x\sim\pi_{\mathrm{ref}}}\big[w(x;\theta)\cdot f(x;\theta)\big],\qquad
    \nabla J_{\mathrm{off}}(\theta)=\mathbb E_{x\sim\pi_{\mathrm{ref}}}\big[f(x;\theta)\big],
    \]
    **离线版正是令 \(w\equiv 1\) 的特例**——这一句把"离线化"精确还原为"丢掉 IS 权重"。
  - **梯度差界**(Thm 3.5):\(\ \big\|\nabla J_{\mathrm{on}}(\theta)-\nabla J_{\mathrm{off}}(\theta)\big\|_2\le G\cdot\sigma_A\cdot\sqrt{\chi^2(\pi_\theta\,\|\,\pi_{\mathrm{ref}})}\),其中 \(\chi^2(\pi_\theta\|\pi_{\mathrm{ref}})=\mathbb E_{x\sim\pi_{\mathrm{ref}}}[w(x;\theta)^2]-1\)。初始 \(\pi_\theta=\pi_{\mathrm{ref}}\) 时 \(\chi^2=0\)、两梯度**严格相等**;drift 增大才偏离,但 KL 正则下保持小。
  - **共享零点**(Thm 3.6):\(\ J_{\mathrm{on}}(\theta)=-D_{\mathrm{KL}}(\pi_\theta\,\|\,\pi_T)\le 0\),全局最大在 \(\theta^\ast\in\arg\min_\theta D_{\mathrm{KL}}(\pi_\theta\|\pi_T)\);当 teacher 可表示(\(\pi_T\in\Pi_\Theta\))时 \(A_t(\theta^\ast)=0\) a.s.,\(\theta^\ast\) 是在线与离线更新的**共同零点**。
  - **协方差校正 = 隐式 trust-region**(Thm 3.7):\(\ \nabla J_{\mathrm{off}}(\theta)=\nabla J_{\mathrm{on}}(\theta)-\mathrm{Cov}_{\pi_{\mathrm{ref}}}\!\big[w(x;\theta),\,f(x;\theta)\big]\)。协方差项初始(\(w\equiv1\))为 0、随 drift 增长,经验上充当 trust-region 稳训(无需显式 KL 惩罚,Fig.3a)。
  - **teacher 不一致的偏置**(Thm 3.8/3.9):记 SFT/OPD 两 teacher 为 \(\pi^{\mathrm{SFT}}_T,\pi^{\mathrm{OPD}}_T\),其失配尺度 \(\sigma_\Delta\)。则离线梯度差界变成 \(\le G(\sigma_A+\sigma_\Delta)\sqrt{\chi^2(\pi_\theta\|\pi_{\mathrm{ref}})}\),且当 \(\sigma_\Delta>0\) 时离线梯度带一个**与 \(\chi^2\) 无关的持久偏置** \(\|\nabla J_{\mathrm{off}}-\nabla J^{\delta=0}_{\mathrm{off}}\|_2\le G\sigma_\Delta\)(Eq.9);标准 OPD 在初始点也有 \(\|\nabla J_{\mathrm{on}}(\theta_{\mathrm{ref}})-\nabla J^{\delta=0}_{\mathrm{on}}(\theta_{\mathrm{ref}})\|_2\le G\sigma_\Delta\)(Eq.10)。即 \(G\sigma_\Delta\) **同时害两范式**;\(\sigma_\Delta=0\)(teacher consistency)时 Thm 3.5/3.6 完全成立、离线匹配在线(Remark 3.10)。
- 关键超参与默认值:每 prompt 采 rollout 数 \(K=1\)(固定);advantage clip 阈 \(\tau\);学习率 \(\eta\);评测采样 temp 0.6 / top-p 0.95、数学 max 32768(32 解)、代码 max 40960(4 解);OPD ~150 步收敛。
- 模块如何咬合:Stage 1 的 \(\pi_{\mathrm{ref}}\) **身兼两职**——既是 OPD 的初始化与参考分布,又是离线 rollout 的采样源;teacher consistency 之所以对离线版更敏感,正因为一旦 SFT teacher 错配,\(\pi_{\mathrm{ref}}\) 这两个角色**同时被污染**(在线版每步刷新 rollout 还能部分恢复)。
- 逐组件必要性:
  - **teacher consistency**:有消融(§4.5,Table 4 完整 grid)。对角线(一致)永远最优;错配降精度,Lightning OPD 比标准 OPD 更敏感(8B 错配掉 6.8 vs 3.7)。这是全篇核心验证。
  - **离线缓存 / \(w\equiv 1\)**:Table 1 显示离线 OPD ≈ 标准 OPD(多处略超),验证 IS 分解中丢掉 \(w\) 的偏差在 teacher consistency 下可忽略;Table 2 量化省下的成本;Fig.3a 显示协方差项当隐式 trust-region。
  - **advantage clip \([-\tau,\tau]\)**:满足 Assumption 3.1(\(|A_t|\le\tau\Rightarrow\sigma_A\le T\tau\)),是 Theorem 3.5 界成立的工程条件;无单独"去 clip"消融,但理论上必要。
  - **stop-gradient on \(A_t\)**:遵循标准 OPD 实践,\(A_t\) 当固定标量(梯度只走 \(\nabla\log\pi_\theta\)),是 reverse-KL 优势加权策略梯度的标准写法。
- 实验与证据:
  - 模型:Qwen3-4B-Base(teacher Qwen3-8B)、Qwen3-8B-Base(teacher Qwen3-32B)、Qwen3-30B-A3B-Base MoE(teacher Qwen3-30B-A3B-Thinking-2507)。
  - 数据:SFT 用 OpenThoughts-3(teacher 生成回答);OPD 数学用 DAPO-Math-17k、代码用 EpiCoder-func-380k 采样 30K。每 prompt 采 1 条 + 预算 teacher logprob 一次。
  - benchmark:数学 AIME24/25、HMMT25(avg pass@1, 32 解, temp 0.6, top-p 0.95, max 32768);代码 LiveCodeBench v5/v6(4 解, max 40960)。
  - 主结果(Table 1):Lightning OPD 在 5 个 benchmark、两规模上**与标准 OPD 持平、多处略超**(8B:AIME24 69.9 vs OPD 68.5;LCB v5 49.5 vs 47.3)。远超 ExOPD(4B AIME24 68.1 vs 61.0;LCB v6 40.3 vs 29.0)。OPD 相对 SFT 增益大且一致(8B math avg 50.8→55.6/57.0)。
  - 成本(Table 2):4B 72→20 GPU 时(3.6×)、8B 120→30(4.0×)。分项:rollout 采集 10、teacher logprob 预算 2-4、OPD 训练本身 8-16——后两者都是一次性离线操作、无需专用基础设施;标准 OPD 需常驻多 GPU teacher server。
  - MoE(Table 3):标准 OPD 在单 8×H100 OOM(共置 30B 教师+30B 学生);**Lightning OPD 让其可行**,AIME24 71.0、LCB v5 60.8,称 open MoE 同规模 SOTA。
  - baseline 公平性:标准 OPD 与 Lightning OPD 在 OPD 阶段**完全相同设置**,只差 rollout 来源(在线当前学生 vs 离线 π_ref 预算)——干净对照。
  - "看着强但没答核心问题":主张是"离线≈在线",Table 1 多处 Lightning **略超**标准 OPD——作者归因于隐式 trust-region(固定 rollout 稳训),但"为何能超而不仅持平"未深究,可能含调参/随机性。
- 假设与失效边界:
  - 【原文】4 条假设:3.1 有界绝对 advantage(advantage clip 自动满足);3.2 支撑覆盖(π_θ 从 π_ref 初始化自然成立);3.3 有界 score function(标准);3.4 有界 teacher mismatch(仅 teacher consistency 结果需要,一致时 σ_Δ=0)。
  - 【原文】teacher consistency 是硬前提,违反则 Gσ_Δ 偏置;离线版比在线版更敏感(§4.5)。
  - 【推断】每 prompt 只采 1 条 rollout 并固定;若 π_ref 质量差或覆盖窄,固定 rollout 的偏差/天花板可能比每步刷新的在线 OPD 更受限(原文未系统扫 rollout 数 K>1 的影响)。
  - 【推断】依赖白盒 teacher(可取 per-token log-prob);黑盒 API teacher 不可用。χ² 界随 drift 增长,长训/大 drift 下离线-在线差是否仍小,实证只到 150 步收敛。
- 祛魅总结【推断】:
  - 真贡献:① 用 IS 分解 + χ²/协方差把"离线 OPD 何时等价在线 OPD"讲清楚,teacher consistency 是干净且实践上被普遍违反的洞见;② 工程收益实打实——3.6-4.0× 提速、单节点、MoE 从 OOM 到 SOTA,且配可复现代码。
  - 包装/可能高估:方法本体是"OPD + 把 rollout 离线缓存 + w≡1",概念上简单;"Lightning" 的提速主要来自去掉 teacher server 这一基础设施优化,而非新损失。Table 1 多处"略超"标准 OPD 的叙事略微夸大(差距在噪声量级);隐式 trust-region 的好处真实但定量证据偏 Fig.3a 单图。低估:teacher consistency 作为"任何 SFT→OPD 流水线都该遵守"的设计规范,影响面可能超出本文。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=per-token reverse-KL advantage A_t=log π_T − log π_θ(teacher 离线缓存)|**改什么**=学生参数(advantage 加权策略梯度,A_t stop-gradient)|**何时改**=OPD 训练每步,读缓存 teacher logprob|**免梯度?**=否(策略梯度;但 advantage 项 stop-grad,梯度只走学生 score function)|**记忆-技能生命周期**=无显式记忆/技能库;teacher 知识经 SFT(同教师)+ 离线 OPD 固化进学生权重|**防遗忘机制**=隐式——固定 π_ref rollout 产生的协方差校正项当 trust-region(Theorem 3.7 + Fig.3a),抑制 policy drift、免显式 KL 惩罚。
- ⑦ 开源代码+框架/harness:https://github.com/jet-ai-projects/Lightning-OPD (已 clone,~3MB,真实可用)。**框架(分阶段明确)**:SFT 阶段 = **LlamaFactory**(`sft/` configs + run script,conda env `llamafactory`);OPD 阶段 = **slime**(Megatron + Ray + SGLang,`train.py` import slime.ray/sglang;`slime_plugins/` 含 mbridge/megatron_bridge/rollout_buffer);teacher logprob 预算 = **vLLM**(`data_curation/pipeline.py`,多 GPU 分片,大教师设 TP_SIZE=4)。
- 💰 资源/成本与可扩展性:**8B 仅 30 GPU 时(单 8×H100 节点)**达 AIME24 69.9;4B 20 GPU 时。3.6-4.0× 提速来自去掉 live teacher server(基础设施从多节点教师+学生降到单节点训练)。MoE 30B-A3B 单节点可训(标准 OPD OOM)。OPD 训 150 步收敛。
- 🎯 对"探索-巩固"对标:**支撑(OPD 工程基座)+ 可借组件**。判定:这是本项目 OPD 主轴的**直接、核心命中**——它把"on-policy 学生自采 + teacher dense per-token 监督"做成低成本可复现的工程范式,正是 TSRD"on-policy 自选 + 教师当稀疏脚手架"的底座。可借组件:① **离线缓存 teacher log-prob** 可直接用于 TSRD/MTP 实验(去掉常驻教师、省算力、MoE 可跑);② **teacher consistency** 是 TSRD 设计的硬约束——若用 MTP 教师当脚手架,SFT 与蒸馏阶段须同源,否则引入 Gσ_Δ 偏置;③ **协方差项当隐式 trust-region** 的洞见可用于稳住 path-recovery 阶段的训练。竞品/缺口:Lightning OPD 是 **dense 全 token 监督、单点固定 rollout、无 step/path 级选择性、无 MTP 前瞻**;TSRD 要的是"稀疏脚手架(只在关键步/走偏处接管)+ MTP 探针"——即把这里的"每 token 都对齐 teacher"改成"只在选路/回轨的关键 token 上用教师信号",这是其留白与 TSRD 的差异化空间。
- 🔭 开放问题/未来方向:【原文】teacher consistency 作为设计规范;离线协方差当 trust-region 的进一步利用。【推断】rollout 数 K>1 / 周期性刷新缓存(半离线)对天花板的影响;把 dense per-token advantage 稀疏化到关键步(与高熵/低置信切分结合)以贴近"稀疏脚手架";结合 MTP——用未来 token 预测决定"哪些 token 该用 teacher 信号",在离线缓存框架下做选择性 OPD;长训/大 drift 下 χ² 界是否仍小的实证扩展。

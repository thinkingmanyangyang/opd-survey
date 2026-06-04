madopd | MAD-OPD: Breaking the Ceiling in On-Policy Distillation via Multi-Agent Debate | 华中科技大(Jianze Wang,阿里实习)+阿里巴巴(Yong Xie/Qianglong Chen 通讯) | Preprint 2026-05-02 v1;arXiv 2605.01347 | 主题线 L1(OPD/蒸馏)+L4(Agent/工具)·相关性 高

**原始论文**:https://arxiv.org/abs/2605.01347

## 一眼看懂
- 🟦 TL;DR:OPD(on-policy distillation,学生在自己轨迹上受 teacher 逐 token 监督)有两个老毛病——单 teacher 封死了学生上限(teacher 错学生跟着错),且基本没人在"会用工具的多步 agent"任务上做过。MAD-OPD 把**多智能体辩论(MAD)**搬进 OPD 训练环:\(K\) 个 teacher 就学生当前 on-policy 状态多轮辩论,辩论记录当作只有 teacher 能看的"privileged context",各 teacher 按辩论后自报的置信加权,合起来给学生逐 token 监督(§4)。再用 **OPAD**(step-level 采样)把 OPD 扩到 agentic 任务,并提出**任务自适应散度原则**:agentic 用有界 JSD、code 用 reverse KL(Remark 1)。
- 最巧的一步:**让 teacher 在学生的 on-policy 状态上"辩论"再监督**(而非各自独立打分求平均)。抽掉辩论(退回 MT-OPD 等权多 teacher),code 上反而**低于单 teacher OPD**——因为两个 teacher 的分布逐 token 平均会插值出"互不兼容的代码路径",产生不连贯监督(§5.2/§5.4,RQ3 消融:debate 比 MT-OPD **+4.6% Co-Avg**)。辩论让 teacher 先收敛到一致立场再监督,避开了这个"平均的陷阱"。

## 为什么做
- 研究背景:OPD 已是主流后训练配方(Qwen3 strong-to-weak、DeepSeek-V4 多 teacher OPD),提供 dense on-policy 信号,是 outcome-reward RL 与 off-policy 序列蒸馏的高效替代。数学上 OPD ≈ teacher-forcing 下的 dense token 级 RL:state \(s_t=(x,\hat y_{<t})\)、action \(a_t=\hat y_t\)、per-token 奖励 \(r_D(s_t)\triangleq-D(\pi^*(\cdot|x,c,\hat y_{<t})\,\|\,\pi_\theta(\cdot|x,\hat y_{<t}))\),故**选散度=选奖励**(§3.1,App.D.1)。
- 解决的具体痛点(三个 L):**L1 单 teacher 天花板**——学生被 teacher 上界封死,诊断工作显示更强 teacher 也未必提得动学生;**L2 任务覆盖窄**——OPD 几乎只在数学/常识做,agentic(多步工具+环境反馈)未探索,逐步误差跨长轨迹累积会 destabilize 训练;**L3 散度选择 ad-hoc**——散度直接塑造奖励,但现有稳定化是"一个失效模式打一个补丁",缺把散度与任务结构挂钩的原则。
- 相关工作 & 各自具体短板:
  - **多 teacher 蒸馏**:汇聚互补信号,但 **teacher 间零交互**(各自独立产出,学生学一个固定聚合)——逐 token 平均会插值出不兼容路径(尤其 code)。
  - **Multi-Agent Debate**:其涌现集体智能此前**只在推理期(inference-time)被消费**(辩论出更好答案直接用),没人把"辩论产生的 privileged 信息"搬进**训练环**当 token 级监督源。
  - **Privileged-info OPD**(Penaloza 等,§2 末):用的是别的 privileged 信号,且散度仍**任务无关**;本文的 privileged info 是辩论 transcript,且散度按任务结构选。
  - **agentic RL / outcome-reward RL**:与 MAD-OPD 正交(可叠加);但 outcome reward 稀疏,OPD 给 dense 信号更高效。
- 动机链:破单 teacher 上限需多模型协作 → 但现有多 teacher 是 off-policy 被动消费 → 把 MAD 的"辩论产生 privileged info"搬进 on-policy 训练环当 token 级监督源 → 同时用 OPAD 填 L2、用散度理论解 L3。
- 与最近邻工作的精确Δ:相对 MT-OPD,差在 **teacher 先辩论再监督 + 按辩论后置信加权**(而非独立等权);相对推理期 MAD,差在**把集体智能落到训练梯度**;相对普通 OPD,差在 **OPAD 让监督随学生实际 rollout/观察自适应** + 散度按任务选(JSD/reverse KL)。

## 怎么做 + 靠不靠谱
- 理论地基(§3,先建"散度=奖励"再推"该选哪个散度"):
  - **OPD 损失**(Eq.1):\(L_{\text{OPD}}(\theta)=\mathbb{E}_{x,\hat y\sim\pi_\theta}\big[\tfrac{1}{|\hat y|}\sum_{t=1}^{|\hat y|}D(\pi^*(\cdot|x,c,\hat y_{<t})\,\|\,\pi_\theta(\cdot|x,\hat y_{<t}))\big]\),\(c\)=仅 teacher 可见的 privileged 信息(此处即辩论 transcript)。记 \(p\triangleq\pi^*(\cdot|x,c,\hat y_{<t})\)、\(q\triangleq\pi_\theta(\cdot|x,\hat y_{<t})\),学生 logit \(z_i\),\(q(v)=\mathrm{softmax}(z)_v\)。
  - **privileged p–q gap**(§3.1):因 \(p\) 看了 \(c\)、\(q\) 没看,在学生采样的 token 上结构性不对称——(a) teacher 给学生采样 token 赋近零概率(\(p(v)\to0\) 而 \(q(v)>0\),**agentic 主导**);(b) teacher 把质量集中在多个有效 token(**code 主导**)。
  - 三个候选散度(Eq.2-4):forward KL \(D_{\mathrm{KL}}^\to(p\|q)=\sum_v p(v)\log\tfrac{p(v)}{q(v)}\);reverse KL \(D_{\mathrm{KL}}^\leftarrow(p\|q)\triangleq D_{\mathrm{KL}}(q\|p)=\sum_v q(v)\log\tfrac{q(v)}{p(v)}\);JSD \(\mathrm{JSD}_\beta(p\|q)=\beta D_{\mathrm{KL}}(p\|m)+(1-\beta)D_{\mathrm{KL}}(q\|m),\ m=\beta p+(1-\beta)q\)(全程 \(\beta=0.5\))。
  - **Lemma 1(JSD 双有界)**:loss \(\mathrm{JSD}_{0.5}\in[0,\log 2]\);**logit 梯度** \(\|\nabla_z\mathrm{JSD}_{0.5}(p\|q)\|_\infty\le 2\),**与 p,q 支撑重叠无关、无任何独立性/分布假设**(App.D.2)。
  - **Lemma 2(reverse KL mode 集中)**:若 \(p=\sum_{j=1}^J\alpha_j p_j\) 为不相交支撑的多 mode 混合,softmax 参数化下对 \(D_{\mathrm{KL}}(q\|p)\) 做梯度下降会收敛到只压在主导 mode \(p_{j^*}\)(\(\alpha_{j^*}=\max_j\alpha_j\))的驻点,代价 \(D_{\mathrm{KL}}(q^*\|p)=-\log\alpha_{j^*}\)(App.D.3)。
  - **Prop 1(agentic 梯度稳定)**:JSD 逐 token logit 梯度 \(\le 2\) 且与轨迹长 \(M\)、师生 gap 无关(Lemma 1.2 + 逐位置可加分解);reverse KL 的 \(\|\nabla_{z_t}D_{\mathrm{KL}}^\leftarrow\|_\infty\) **无界**(含 \(q(i)\log(q(i)/p(i))\) 项,在 \(p(i)\to0,q(i)>0\) 时发散,正是 privileged 监督制造的 regime);forward KL \(\|\nabla_{z_t}D_{\mathrm{KL}}^\to\|_\infty=\|q-p\|_\infty\le1\) 有界但 mode-covering 损 agentic 性能。
  - **Prop 2(code 连贯)**:对多 mode code 分布,reverse KL 的 \(q^*\) 收敛到单一主导实现路径(避免拼接);forward KL 覆盖所有 mode(平均不兼容代码结构);JSD 部分集中但保留次 mode 非零质量。
  - **Remark 1(任务自适应散度选择原则)**:**多步 agentic → \(D=\mathrm{JSD}_\beta\)**(梯度有界、长轨迹稳);**code → \(D=D_{\mathrm{KL}}^\leftarrow\)**(集中到单一连贯实现)。§5.4 经验验证(Table 4 / Fig.6)。
- 方法流水线(§4,Fig.2):
  - ① **辩论产 privileged info**(§4.1,Eq.5):每决策点状态 \(s_m\)(单轮=prompt \(x\),agentic=context \((x,\tau_{<m})\)),\(K\) teacher 辩论 \(R\) 轮——round 1 各自独立采样、round \(r\ge2\) 读全部历史并修订 \(h_r^k\sim p_{T_k}(\cdot|s_m,\{h_{r'}^j\}_{j,r'<r})\);全部历史 \(H_R^m=\{h_r^k\}\) 作 privileged context \(c_m\)(Def.1:teacher 可见、student 不可见)。
  - ② **置信加权**(§4.2,Eq.6-7):辩论后各 teacher 自报置信 \(c_k\in[0,100]\),归一 \(\tilde c_k=c_k/100\),softmax 得 \(w_k=\dfrac{\exp(\tilde c_k/\tau_{\text{conf}})}{\sum_j\exp(\tilde c_j/\tau_{\text{conf}})}\)(\(\tau_{\text{conf}}=1.0\))。因 \(c_k\) 在 R 轮后产生,权重反映**辩论后**的确定性(立场在辩论中被削弱者贡献更小)。
  - ③ **token 级蒸馏目标**(§4.3,Eq.8):teacher **带** \(H_R^m\) force-decode 学生 on-policy 样本 \(\hat y\)、学生**不带**,按置信加权多 teacher 散度求 token loss:
  \(\displaystyle L_{\text{MAD-OPD}}(\theta)=\mathbb{E}_{s_m\sim D,\hat y\sim\pi_\theta}\Big[\tfrac{1}{|\hat y|}\sum_{t=1}^{|\hat y|}\sum_{k=1}^{K}w_k\cdot D\big(p_{T_k}(\cdot|s_m,H_R^m,\hat y_{<t})\,\|\,p_S(\cdot|s_m,\hat y_{<t})\big)\Big].\)
  散度跨**全词表**;teacher logits 作固定目标,梯度只流学生。
  - ④ **OPAD**(§4.4,Eq.9-10):学生逐步 rollout 轨迹 \(\tau=(a_1,o_1,\dots,a_M,o_M)\),step \(m\) 在状态 \(s_m=(x,\tau_{<m})\) 采 \(a_m\sim\pi_\theta(\cdot|s_m)\)、环境返回 \(o_m\sim P^{\text{env}}(\cdot|s_m,a_m)\);每步 teacher 就 \(s_m\) 辩论、就**实际观察**force-decode \(a_m\),per-step loss
  \(\displaystyle L^{\text{opad}}_D(s_m)=\tfrac{1}{|a_m|}\sum_{t=1}^{|a_m|}\sum_{k=1}^{K}w_k\cdot D\big(p_{T_k}(\cdot|s_m,H_R^m,a_{m,<t})\,\|\,p_S(\cdot|s_m,a_{m,<t})\big),\)
  总损失 \(L_{\text{OPAD}}(\theta)=\mathbb{E}_{x\sim D,\tau\sim\pi_\theta}\big[\sum_{m=1}^{M}L^{\text{opad}}_D(s_m)\big]\)。关键自适应:辩论**逐步发生且条件于真实观察**(非假想),监督随学生实际轨迹走。
  - ⑤ 散度按 Remark 1 选:agentic→JSD_{0.5}、code→reverse KL;梯度只流学生。
- 逐组件必要性(均有消融,§5.4 RQ3 Fig.4a,全数 App.C.1):
  - **辩论(debate)**:破 L1 的核心。去掉→MT-OPD,code 上 **−4.6% Co-Avg**(且低于单 teacher OPD)。必要。
  - **置信加权**:去掉(等权)→**Ag-Avg −1.7%**。必要,但贡献小于 debate。
  - **多 teacher**:去掉→单 teacher OPD,被天花板封顶。必要。
  - **on-policy**:去掉→MT-SeqKD(off-policy),继承 exposure bias。必要。
  - **辩论轮数 R**:R=2(独立生成上加 1 轮互评)经验最优,R=3 因 prompt-context 膨胀过头反而变差(App.C.6 扫 \(R\in\{0,1,2,3\}\))。属调参,非"越多越好"。
  - **散度选择**:固定其余只换 D(§5.4 RQ4 Fig.4b/表4):JSD 在 agentic 领先、reverse KL 在 code 领先、forward KL **两者都落后**,与理论(Prop 1.3 + Prop 2.2)一致(mode-covering 两边都伤,mode 集中只对 code 的单连贯路径结构对路)。
- 关键机制一句话直觉:**agentic 怕梯度炸→选有界的 JSD(\(\le2\),与轨迹长无关);code 怕路径串味→选会挑一条 mode 的 reverse KL**。辩论的作用是先让多 teacher 收敛一致立场,再 force-decode,使监督"在 privileged p–q gap 最宽处最密"、比单 teacher argmax 解码更锐(Prop.1)。
- 实验与证据:训练 agentic 用 ToolACE(按 OPAD 拆 step-level,~16K)、code 用 OpenThoughts3 采 30K(两者独立 checkpoint)。评测 5 集(agentic:BFCL-v4、τ²-Bench、VitaBench;code:LiveCodeBench v6、MBPP+;**刻意排除数学**,因近期 OPD 已覆盖)。六组师生(Qwen3/Qwen3.5,师生比 4.7×–15.5×)。关键数字(14B+8B→4B,Table 1):MAD-OPD Avg **34.66** vs OPD 31.72 / MT-OPD 31.27 / MT-SeqKD 29.19 / Base 27.79;**6 配置 MAD-OPD 全部 overall Avg 第一**。Δ(Base) 随能力 scaling(Fig.3b):+5.3%(1.7B)/+6.9%(4B)/+8.0%(4B,32B+30B teacher)/+6.4%(8B)/+4.6%(14B)/+3.5%(Qwen3.5 27B+9B→4B,该配置 MT-OPD<OPD,辩论把分歧转成加权共识才挽回)。更强主张:4B 学生在 LCB-v6 **超其 14B teacher** +4.26% pass@1、+10.29% BoN@16(App.C.3)→ 论文据此称"瓶颈是 teacher-pool 多样性而非单 teacher 能力"。baseline 设计较公平(单 teacher 取较强者、MT-OPD 等权、MT-SeqKD 7:3 off-policy 混合)。
- 假设与失效边界:
  - 【原文 §6】需 teacher 的 **token 级分布**(排除纯 API black-box);假设**师生共享词表**(cross-vocab 是开放扩展);训练成本随 **\(K\times R\) teacher 前向 scale**;全程 **non-thinking 模式**;仅 agentic+code(排除数学),长程 agentic 留 future work。
  - 【推断】"4B 超 14B teacher"仅在 **LCB-v6 单基准 + BoN@16 放大**,普适性需谨慎(论文也只在此处给出)。置信由 teacher **自报**,可被操纵/不可靠的风险只靠 App.C.1 鲁棒性分析间接缓解。task-adaptive 原则的理论前提(code=disjoint-support 多 mode、agentic=\(p\to0\) gap)是**理想化建模**,真实分布未必严格满足。
  - 【推断】成本/收益偏温和:agentic +2.4%/code +3.7% 相对"在线跑 \(K\times R\) teacher 辩论+force-decode"的开销而言不大,且论文**未给相对单 teacher OPD 的端到端开销对比**(只给绝对 wall-clock:4B agentic≈16h、code≈32h)。
- 祛魅总结【推断】:真贡献有二——(1) **首次把 MAD 的集体智能落进 OPD 训练梯度**,并用辩论规避了"多 teacher 逐 token 平均产生不兼容监督"这一真实失效;(2) **罕见地给出 logit 梯度有界性证明**(Lemma 1.2 \(\le2\)),把"该用哪个散度"从超参升格为可推导的原则(理论-经验闭环较完整)。被适度高估的是"破天花板"的力度——绝对增益温和、最亮的"4B 超 14B"是单基准 BoN 个案,且全程 non-thinking、排除数学,适用面比标题窄。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=token 级散度监督(置信加权的多 teacher 分布 \(D(p_{T_k}\|p_S)\),全词表);**改什么**=学生策略参数;**何时改**=on-policy,每个决策点/每步(OPAD 随学生实际观察);**免梯度?**=否(梯度只流学生,teacher logits 作固定目标);**记忆-技能生命周期**=无记忆库/技能库,辩论 transcript 是**每决策点临时**生成的 privileged context,用完即弃;**防遗忘机制**=无显式防遗忘;靠 on-policy(在学生自己分布上学)+ 有界 JSD 防长轨迹失稳间接稳住训练。
- ⑦ 开源代码+框架/harness:https://github.com/chiefovoavicii/MAD-OPD(已克隆 ~1.9MB,含核心算法)。结构:`mad_opd/trainers/`(`mad_opd_core.py` 散度原语、`mad_opd_trainer.py`、`vllm_teacher_manager.py`)、`scripts/`(四算法脚本 + `launch_teachers.py` + `force_decode_server.py`)、`eval/`、`data/`。框架 **TRL(>=0.15,<0.25) + vLLM(>=0.6,出辩论文本)+ DeepSpeed ZeRO-3 + transformers 4.51.1 + liger-kernel**(LigerFusedLinearJSDLoss 可选)。〔核码已核:默认走全词表 chunked JSD,与论文 §B.2 "full vocabulary rather than top-k truncation" 一致;top-k 是默认关闭的省显存旋钮。〕可得性:真实可复现。
- 💰 资源/成本与可扩展性:8×NVIDIA H20 + ZeRO-3;每 teacher 由两个单卡进程服务(vLLM 出辩论文本 + sidecar 出 token 级 logit)。超参(§B.1):K=2、R=2、β=0.5、lr 1e-5 cosine+5% warmup、有效 batch 128、grad clip 1.0、agentic max len 4096 / code 16384、1 epoch。wall-clock:4B agentic≈16h、code≈32h。**成本随 \(K\times R\) teacher 前向线性增长**是主要扩展瓶颈(论文自承,未量化相对单 teacher 的额外开销)。
- 🎯 对"探索-巩固"对标:**强支撑(直接对标 idea 的 OPD 骨架)**——MAD-OPD 本身就是 on-policy distillation,与"teacher 当稀疏脚手架在学生自身轨迹上监督"高度同构;OPAD 的 step-level + 随实际观察自适应 ≈ idea 的"走偏后在实际状态上接管"。可借组件:(a) **任务自适应散度原则**(JSD 有界 \(\le2\) vs reverse KL mode 集中)——直接可用于决定 MTP/OPD 在"探索期"(怕梯度炸,用有界 JSD)vs"巩固期"(怕路径串味,用 reverse KL)分别该用什么散度;(b) **privileged p–q gap 的有界性分析**(Prop.1)给"何时监督会失稳"提供了可计算判据。缺口:MAD-OPD 的脚手架**不稀疏**(每决策点都辩论+全 token 监督,开销大),与 idea 的"稀疏单点接管/MTP 前瞻探针触发"相反;无记忆/技能固化、无 MTP 前瞻。判定依据:§3-4 + Remark 1 + 表1。
- 🔭 开放问题/未来方向:【原文 §6】长程 agentic(误差累积更重)、thinking 模式、cross-vocabulary 蒸馏、数学域。【推断】把"每决策点都辩论"改为"仅在学生走偏/低置信的关键步触发辩论监督"(稀疏化,直接对接 idea 与降本);用 MTP 前瞻探针判定"何时该触发 teacher 辩论";teacher 自报置信的可靠性校准(防操纵)。

— 残留待核:0(Eq.1–10、Lemma 1/2、Prop 1/2、Remark 1、消融 +4.6%/−1.7% 与 Δ(Base) scaling 均据 PDF §3–§5 + Fig.3/4 + Table 1 抄准。)

caopd | The Illusion of Certainty: Decoupling Capability and Calibration in On-Policy Distillation (CaOPD) | Salesforce AI Research（Jiaxin Zhang、Xiangyu Peng、Qinglin Chen、Qinyuan Ye、Caiming Xiong、Chien-Sheng Wu）| 2026-04（arXiv:2604.16830v1，Preprint，cs.LG）| 主题线 L1 OPD/自蒸馏·相关性 高

**原始论文**：https://arxiv.org/abs/2604.16830

## 一眼看懂

> 一句话导读：OPD 在提准确率的同时,会把模型训得越来越"嘴硬"（过度自信）。CaOPD 发现根因是"teacher 开卷、student 闭卷",于是把"答什么"和"多自信"拆开:只把监督里的置信片段换成学生自己估的成功率,其余照旧——几乎零成本,能力和校准两头都不丢。

- 🟦 TL;DR：on-policy distillation（OPD,这里是自蒸馏形式）在提升准确率的同时,会系统性地把模型推进"过度自信"区。作者把这个现象叫 **Scaling Law of Miscalibration（误校准的标度律）**——连 GPT-5.x/Claude/Gemini/DeepSeek/Kimi/Qwen3.5-397B 这些前沿模型都落在过自信区（Fig.1）。根因是**训练-部署的信息不对称**:teacher"开卷"（拿到特权上下文 \(z\)）能产生近乎确定性的轨迹,student"闭卷"（只看得到 \(x\)）去匹配它,就被迫把 logits 锐化、还顺带继承了那种"笃定"的风格。CaOPD 的解法是把"答什么（能力）"和"多确信（置信）"在监督目标上**解耦**——只把轨迹里的置信片段替换成 student 自己 rollout 估出来的经验成功率 \(\hat\mu(x)\)，其余部分照常做 reverse-KL 蒸馏。这样几乎零额外改动,就能同时拿到能力和校准。
- 最巧的一步：**target replacement（目标替换）——只改置信段的监督目标,reverse-KL 机制完全不动（§4.2 Eq.7）**。把它抽掉,就退回标准 OPD（置信段被逼着去模仿 teacher 的 ≈1.0）。精髓在于:reasoning 段的前缀和 teacher 内容都没变,所以在 reasoning 位置上它与标准 OPD 等价（能力克隆原样保留）;只有置信位置的目标从"teacher 的笃定"换成了"student-grounded 的 \(\hat\mu(x)\)"。于是熵坍缩和乐观偏置被直接掐掉,而且全程不与 RL 优化器对抗（这就避开了所谓的能力税）。

## 为什么做

> 一句话导读：OPD/自蒸馏迁移能力很在行,但全都不管置信度,结果把模型训得越来越过度自信;现有的校准办法（RL reward shaping）又会付出"能力税",test-time 多采样则太贵。

- 研究背景：后训练越来越依赖 OPD（Agarwal 2024）和自蒸馏（SDPO-Hübotter 2026、SDFT-Shenfeld 2026、OPSD-Zhao 2026、Sang 2026、why_sd_degrade-Kim 2026）。它们的共同套路是:让 teacher 在**特权上下文**下（比如 verifier 反馈、专家示范、ground-truth 解）生成高质量轨迹,再让 student 仅凭 prompt 去模仿。这类方法在迁移能力上很成功,但几乎都是 capability-centric（以能力为中心）,**完全不管置信度**（§1）。
- 解决的具体痛点,有三条:
  - ① **误校准的 Scaling Law**——实测连前沿大模型都落在过自信区,说明**能力变强并不会自动修正盲目乐观**（Fig.1;引 Kalai 2025 的 "test-taking mode"——模型被激励去输出听起来合理的假话,而不是认怂）。
  - ② **OPD 本身就加剧过自信**——它在提准确率的同时,把平均置信推向饱和（Tool Use 上 SDPO 达到 0.996）,OCG（过自信缺口）不降反升（§5 Table 1）。
  - ③ **RL reward shaping 有"能力税"**——RLCR/CAR（Damani 2025 / Xuan 2026 把 Brier/proper scoring 塞进 PPO）虽然能压低置信,但为了躲避罚项会变得过度保守,准确率明显往下掉（Askell 2021 的 capability tax）。
- 相关工作 & 各自不足:
  - ① **OPD/自蒸馏（SDFT/SDPO/OPSD）**——能力强,但不管置信;
  - ② **置信校准主流靠 RL + reward shaping（RLCR/CAR）**——有能力税,而且与优化器对抗;
  - ③ **test-time 不确定性估计（SelfCheckGPT/SAC³,靠多次采样）**——可靠,但推理成本是 \(O(K)\)。
- 动机链：
  - 起点是现状:OPD 迁移能力很成功,但全场过自信。
  - 追到根因:信息不对称——teacher 开卷、熵低,student 闭卷,于是被逼着锐化 logits、还继承了"成功轨迹"自带的那股笃定。
  - 得出结论:"**能力可以靠模仿可靠地迁移,但'该多确信'这件事,不能跨信息状态安全迁移**"。
  - 落到做法:把能力和置信在监督目标上拆开,而置信目标应当锚到 student 自己的部署成功率 \(\mu(x)\)。
  - 为什么不用 RL reward shaping？因为有能力税、且与优化器对抗（§4 原文 "rather than fighting the optimizer"）。
  - 为什么不用 test-time 多采样？因为贵,是 \(O(K)\)——但这笔开销可以**摊销**到训练阶段:训练时把 \(\hat\mu\) 算好、蒸进参数,部署时只需单次前向。
- 与最近邻工作的 Δ：
  - vs RLCR/CAR（用 reward shaping 做校准）——CaOPD **不改 reward、不加优化阶段**,纯粹在蒸馏管线内做 target replacement,因此没有能力税。
  - vs 标准 OPD/SDPO——只多一步置信替换,reasoning 段完全等价。
  - vs why_sd_degrade（Kim 2026,诊断自蒸馏为何会退化推理）——两者切的是同一个信息不对称现象的不同侧面:CaOPD 改监督目标来修过自信,why_sd_degrade 只做退化诊断。
  - 关键点:把校准当成"换一个监督目标",而不是"加一个奖励罚项"。

## 怎么做 + 靠不靠谱

> 一句话导读：先用三条信息论命题论证"为什么能力蒸馏一定会带来过自信",再给出极简解法——只把监督目标里的置信片段换成学生自估的成功率 \(\hat\mu(x)\)，reasoning 段一字不动。所以能力照旧、过自信被掐掉。

> 设定（§2.1）：生成格式带 **verbalized confidence（把置信度直接写出来的格式）**——每条生成 \(y=(a,c)\) 分两段:\(a\) 是 reasoning 轨迹（含最终答案）,\(c\) 是置信段（如 "Confidence: 0.85"）,\(\mathrm{val}(c)\in[0,1]\) 是从中解析出的标量置信。自蒸馏架构里,同一个模型扮演两个角色:**student** \(\pi_\theta(\cdot\mid x)\)（只看得到 prompt,模拟部署时的状态）;**teacher** \(\pi_\theta(\cdot\mid x,z)\)（额外条件化在特权上下文 \(z\sim Z(x)\) 上,\(z\) 可以是示范、反馈或 ground-truth 解）。一个结构性质:teacher 见到的证据严格多于部署时的 student。

**【理论:为何能力蒸馏会加剧过自信（§3,三个命题,Appendix A 给全证）】**
- **理想的校准目标（§2.2）**：部署成功率定义为 \(\mu(x):=\mathbb E_{a\sim\pi_\theta(\cdot\mid x)}[R(x,a)]=\Pr_{\pi_\theta}(R=1\mid x)\)，其中 \(R\) 是二值验证函数。校准的要求是 \(\mathbb E[\mathrm{val}(c_\theta(x))]=\mu(x)\)。所以 **\(\mu(x)\) 才是置信对齐真正该用的 ground-truth 目标**。
- **标准 OPD 目标的盲点（§2.3）**：用的是 per-token reverse-KL（先采 \(y\sim\pi_\theta(\cdot\mid x)\)，再让 student/teacher 沿这条轨迹各自算分布）——
  \(\displaystyle \mathcal L_{\text{OPD}}(\theta)=\mathbb E_{x,z\sim Z(x)}\,\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\Big[\sum_{t=1}^T D_{KL}\big(\pi_\theta(\cdot\mid y_{<t},x)\,\|\,\pi_\theta(\cdot\mid y_{<t},x,z)\big)\Big].\)
  问题在于:\(y=(a,c)\) 里含置信 token,所以这个损失也会作用在置信位置上。而 teacher 拿了 \(z\)（比如正确答案）,在置信位会输出 ≈1.0。结果就是 **OPD 在迁移能力的同时,也逼着 student 去模仿 teacher 那种无端的笃定**。
- **命题1（信息差 → 不可辨识）**：当条件互信息 \(I(R;Z\mid X)>0\)（即特权上下文 \(Z\) 携带了 \(X\) 之外、对正确性有用的信息）时,teacher 的条件成功率 \(\mu_T(X,Z)\) 对 \(X\) 是不可测的;并且在平方误差意义下,**仅凭 \(X\) 能做的最优预测恰好就是 student 的部署成功率 \(\mu(X)\)**,残差严格为正:
  \(\displaystyle \min_g \mathbb E\big[(\mu_T(X,Z)-g(X))^2\big]=\mathbb E_X\big[\mathrm{Var}(\mu_T(X,Z)\mid X)\big]>0.\)
  含义:拿 teacher 的笃定当置信目标,在信息论层面本身就是错的。
- **命题2（特权条件 → 熵坍缩）**：当 \(I(A;Z\mid X)>0\) 时,teacher 轨迹的期望熵严格低于"仅给 \(X\)"时的条件熵:\(\mathbb E_{X,Z}[H(\pi_\theta(A\mid X,Z))]<\mathbb E_X[H(A\mid X)]\)。去最小化对它的 per-token reverse-KL,就会**逼 student 的 logits 人为锐化**——因为表达自然的不确定性 \(H(A\mid X)\) 会被惩罚。
- **命题3（选择偏置 → 乐观）**：特权上下文 \(Z\) 往往取自成功/高质量样本（记作 \(\mathcal D_{\text{helpful}}\)）,这使得 \(\mathbb E_{Z\sim\mathcal D_{\text{helpful}}}[\mu_T(X,Z)\mid X]\ge\mu(X)\)（在正测度集上严格成立）。于是蒸进 student 的隐式目标,会对真实部署成功率**系统性上偏**:\(\mathbb E_{X,Z\sim\mathcal D_{\text{helpful}}}[\mu_T(X,Z)-\mu(X)]>0\)。

**【CaOPD 方法流水线（§4 Algorithm 1）——读完可复现】** 对每个输入 \(x\)：
1. **student-grounded 置信估计（§4.1）**：采 \(K\) 条独立的 \((a_k,c_k)\sim\pi_\theta(\cdot\mid x)\)，用 verifier 打分,算经验成功率——
   \(\displaystyle \hat\mu(x)=\frac1K\sum_{k=1}^K R(x,a_k).\)
   **关键的省成本之处**:在 SDPO 下,可以**直接复用基础训练循环里已经生成的 rollout** 来算 \(\hat\mu(x)\)，只额外多一次轻量的 verifier 评估;蒸馏轨迹是另采的,不增加额外成本。开放域里若没有 verifier,\(\hat\mu\) 就退化成用 **Teacher-Anchored Self-Consistency** 来估（Appendix B.6）。
2. **target replacement（§4.2,只改 completion 和 teacher context,reverse-KL 机制不动）**：给定 student 生成轨迹 \(y=(a,c)\)——
   - (i) **改 completion**:把置信段 \(c\) 换成 \(\hat\mu(x)\)，得到 \(\tilde y=(a,\hat\mu(x))\)。
   - (ii) **改 teacher context**:在特权上下文 \(z\) 里,把原本 ≈1.0 的置信改写成 \(\hat\mu(x)\)，得到 \(\tilde z\)。
3. **蒸馏（§4.2 Eq.7）**：让 student/teacher 各自条件化（分别在 \(x\) 与 \((x,\tilde z)\) 上）给 \(\tilde y\) 打分;token 位置分成两段——reasoning 段 \(I_a=\{1,\dots,T_a\}\)、置信段 \(I_c=\{T_a+1,\dots,T\}\)：
   \(\displaystyle \mathcal L_{\text{CaOPD}}(\theta)=\mathbb E_{x,\tilde z}\,\mathbb E_{\tilde y}\Big[\underbrace{\sum_{t\in I_a}D_{KL}\big(\pi_\theta(\cdot\mid\tilde y_{<t},x)\,\|\,\pi_\theta(\cdot\mid\tilde y_{<t},x,\tilde z)\big)}_{\text{Capability Cloning(保留)}}+\underbrace{\sum_{t\in I_c}D_{KL}\big(\pi_\theta(\cdot\mid\tilde y_{<t},x)\,\|\,\pi_\theta(\cdot\mid\tilde y_{<t},x,\tilde z)\big)}_{\text{Confidence Calibration(student-grounded)}}\Big].\)
   - **reasoning 位置 \(t\in I_a\)**:此时前缀 \(\tilde y_{<t}\) 只含 reasoning token（置信替换还没出现）,而且 \(\tilde z\) 的 reasoning 内容也没改。所以这里的 per-token KL **与标准 OPD（Eq.2）实质相同**——能力克隆原样保留。
   - **置信位置 \(t\in I_c\)**:此时打分的 token 编码的是 \(\hat\mu(x)\) 而非原来的 \(c\)，teacher context 反映的也是 \(\hat\mu(x)\) 而非 1.0。于是 student 被训练去产出 student-grounded 的经验成功率,**直接消解了熵坍缩（命题2）和乐观偏置（命题3）**。
   - 优化器用 AdamW。整套流程**不改 reward、不加额外优化阶段**。
- **逐组件必要性**：
  - **置信段替换**——核心,没它就是标准 OPD（过自信）。
  - **特权上下文里也改置信（步 (ii)）**——保证 teacher 在置信位置不再给≈1.0 的目标（否则 reverse-KL 仍把 student 拉向笃定）。
  - **\(K\) 次 rollout 估 \(\hat\mu\)**——§K 消融:准确率对 \(K\in\{1,\dots,32\}\) **平坦**（能力与置信采样方差解耦）,\(K\ge8\) 是校准甜点,K 太小目标过度量化反困在过自信。
  - reasoning 段不动是**设计而非消融对象**（理论上等价 OPD）。
- **关键机制/公式（直觉）**：三条命题把"为什么必须这样做"锚定住了——\(\mu(X)\) 是唯一一个仅凭 \(X\) 可测的无偏置信目标（命题1）;teacher 的低熵会逼着 student 锐化（命题2）;对成功样本的筛选会带来系统性上偏（命题3）。target replacement 的优雅之处在于:reasoning 前缀和 teacher 内容都不变,所以能力克隆等价于 OPD;只换置信段的目标,校准就被结构性地纠偏了。而且它不与优化器对抗——不像 RL 的罚项,要在 reward 里反复拉扯。至于 test-time 那笔昂贵的置信估计,则被**摊销**进了单次前向的模型参数里。
- **实验与证据（§5）**：覆盖两个域——Science Q&A（Chemistry,来自 SciKnowEval/Feng 2024）和 Tool Use（ToolAlpaca）;主模型是 Qwen3-8B、Olmo-3-7B-Instruct;scaling 实验覆盖 Qwen3 全家 0.6B→32B（Appendix C.1）。指标除了 Accuracy、ECE/Brier,还有两个自引的:
  - **OCG**（mean confidence − accuracy）——量化过自信的方向,值越正、越过自信;
  - **SPR**（Strict Pairwise Ranking,即 \(P(c^+>c^-)\)）——对"置信饱和/打平"会**给零分重罚**,比 AUROC 更严格。
  - **OPD 加剧过自信（Table 1）**：Qwen3-8B Science Q&A base OCG +58.7%,SDFT/SDPO 推到 +48.1%/+12.9%;Tool Use SDPO 平均置信 0.996（OCG +32.0%）、SDFT +32.4%。
  - **CaOPD 结构纠偏不掉能力（Table 1/2）**：Qwen3-8B Tool Use OCG 从 SDFT +32.4%→**−0.7%**;主表（vs SDFT）Acc 67.6→70.6、ECE 0.321→0.228、BS 0.320→0.242、**SPR 0.085→0.555**（SDFT 几乎丧失区分度,CaOPD 恢复）;Science Q&A vs SDFT Acc 49.1→50.0、ECE 0.486→0.266、SPR 0.387→0.599。Olmo 同向（Science Q&A ECE 0.429→0.176）。
  - **避开能力税**：vs SDPO 在 Qwen3-8B Tool Use Acc 66.2%→70.9% 同时 ECE 0.298→0.133,而 RLCR/CAR 校准好但准确率明显更低。
  - **成本（Fig.2）**：每步耗时与 SDPO 几乎一致。**OOD**（Tool Use→Chemistry）SDFT ECE 0.599、CaOPD 0.358。**持续学习(CT)** 解决"校准遗忘"（CT ECE 0.126、SPR 0.662）。
  - baseline 较公平（SDFT/SDPO/GRPO + 专门校准的 RLCR/CAR 都比）。
- **假设与失效边界**：【原文】(§Limitations) 依赖**可验证 verifier** 估 \(\hat\mu\)（开放域退 self-consistency,质量待考）;依赖**可解析的 verbalized confidence 格式**（测试偶发格式失败）;SFT 下不一定能像 SDPO 那样零成本复用 \(K\) 次 rollout;**能力上界受 base 约束,CaOPD 不提升推理能力只校准置信**;仅 utterance-level 置信,step-level 校准是 future work。【推断】校准收益几乎全来自置信段,准确率提升主要来自底层 OPD/SDPO 而非 CaOPD 机制本身。
- **祛魅总结**：
  - 【推断】真贡献是两块:一个**干净的诊断**（用三条信息论命题把"OPD → 过自信"讲清楚）+ 一个**极小工程改动的解法**（只换置信段的监督目标,reverse-KL 不动,既不改 reward 也不加额外阶段）。target replacement 这一招优雅且可复现。
  - 要祛魅的边界有三:
    - ① 方法**只对"显式 verbalized confidence"这种格式生效**,并不是给模型注入了某种内在的校准能力;
    - ② 对推理能力本身**零直接增益**（命题和实证都只动置信段）——标题 "decoupling capability and calibration" 准确,但容易被误读成"两头都提",其实能力提升完全靠底层 OPD;
    - ③ 命题1说"\(\mu(X)\) 是最优目标"是理论层面的论断,实证只能侧面印证（换上 \(\hat\mu\) 后 ECE/BS 确实下降）。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=reasoning 段 teacher 的 per-token reverse-KL（能力克隆）+ 置信段的 student-grounded 经验成功率 \(\hat\mu(x)\)（校准）｜**改什么**=参数（logits 分布,尤其置信 token 位置）+ **监督目标的置信片段**（target replacement,改 completion + teacher context）｜**何时改**=在线 per-prompt（先 \(K\) 次 rollout 估 \(\hat\mu\) → 替换 → 蒸馏）,训练时摊销 test-time 估计｜**免梯度?**=否（reverse-KL 蒸馏梯度下降）｜**记忆-技能生命周期**=不适用（无外部记忆/技能库）｜**防遗忘机制**=持续学习(CT)下解决"**校准遗忘**"（CT ECE 0.126、SPR 0.662）,但非参数级防遗忘机制,是 target 解耦带来的副产物
- ⑦ 开源代码+框架/harness：https://github.com/SalesforceAIResearch/CaOPD （旧 analysis 记已 clone 约 40M,含 `main.py`/`distil_trainer.py`/`distil_config.py`/`eval_science.py`/`eval_tooluse.py` + Tool Use 与 Chemistry 两域数据 + Salesforce 合规文件）。框架=**TRL 0.24 + vLLM 0.12 + DeepSpeed 0.18 + transformers 4.57**（自建 distil trainer,在 SDFT/SDPO 自蒸馏管线上做 target replacement）。【待核】依赖版本与目录细节复用旧 analysis 核查,本次基于 PDF 重写未重新进仓核对。代码可得性:训练+评测+数据齐全。
- 💰 资源/成本与可扩展性：【原文】每 prompt 需 \(K\) 次 rollout（K=8 够用,准确率对 K 平坦）;**SDPO 下复用已有 rollout,几乎零额外成本**（只多轻量 verifier 评估）,每步耗时与 SDPO 几乎一致（Fig.2）;SFT 下不一定零成本复用。scaling 0.6B→32B 均有效。
- 🎯 对"探索-巩固"对标：**支撑 + 警示（边界提供者）**。判定:CaOPD 揭示的"**信息不对称 → teacher 的笃定不能安全迁移给 student**"对 TSRD 是一个关键警示——teacher 当脚手架（开卷）产生的轨迹和置信,不能让 student（闭卷）盲目模仿,否则会继承不属于自己的确定性。这正对应"巩固"环节该固化的是 student **自己走得通**的东西（\(\hat\mu(x)\) 就是用 student 自采 rollout 估的）,而不是 teacher 的特权产物。
  - **可借组件**:① "student-grounded 目标"这个思想——TSRD 在巩固时,也应当用 student 自采的成功率 / 自选的恢复分支当目标,而不是直接抄 teacher;② target replacement 这种"只改某一段监督目标、底层机制不动"的低侵入手法,可以移植到 path-recovery（只替换接管点的目标,reverse-KL 保持不变）。
  - **缺口/竞品面**:它只解决置信校准、不碰推理能力;只到 utterance-level、没有 step-level;没有 MTP/前瞻;与"探索/选路"的关系也弱（它的主战场是巩固阶段的"该多确信"）。
- 🔭 开放问题/未来方向：【原文】long/agentic 多步推理的 **step-level 校准**是 future work;开放域无 verifier 时的 \(\hat\mu\) 近似质量（self-consistency）;verbalized confidence 格式鲁棒性。【推断】把"student-grounded target replacement"推广到置信之外的其他可解耦属性（如安全/风格）;与 step/turn 级信用分配结合做细粒度校准;与 MTP 前瞻结合预测"该多确信"。

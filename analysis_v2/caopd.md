caopd | The Illusion of Certainty: Decoupling Capability and Calibration in On-Policy Distillation (CaOPD) | Salesforce AI Research（Jiaxin Zhang、Xiangyu Peng、Qinglin Chen、Qinyuan Ye、Caiming Xiong、Chien-Sheng Wu）| 2026-04（arXiv:2604.16830v1，Preprint，cs.LG）| 主题线 L1 OPD/自蒸馏·相关性 高

**原始论文**：https://arxiv.org/abs/2604.16830

## 一眼看懂
- 🟦 TL;DR：on-policy distillation（OPD/自蒸馏）提准确率的同时会系统性把模型推进"过度自信"区（作者称 **Scaling Law of Miscalibration**,连 GPT-5.x/Claude/Gemini/DeepSeek/Kimi/Qwen3.5-397B 都落在过自信区,Fig.1）。根因是**训练-部署信息不对称**:teacher"开卷"（拿特权上下文 \(z\)）产生近确定性轨迹,student"闭卷"（只见 \(x\)）去匹配它就被迫锐化 logits + 继承"笃定"风格。CaOPD 把"答什么(能力)"和"多确信(置信)"在监督目标上**解耦**——只把轨迹里的置信片段替换成 student 自己 rollout 估的经验成功率 \(\hat\mu(x)\)，其余照常 reverse-KL 蒸馏,从而几乎零额外改动同时拿能力与校准。
- 最巧的一步：**target replacement——只改置信段的监督目标、reverse-KL 机制完全不动（§4.2 Eq.7）**。抽掉它就退回标准 OPD（置信段被逼模仿 teacher≈1.0）。精髓:reasoning 段前缀和 teacher 内容都没变 → 在 reasoning 位置等价标准 OPD（能力克隆原样保留）,只有置信位置的目标从"teacher 的笃定"换成"student-grounded 的 \(\hat\mu(x)\)",于是熵坍缩和乐观偏置被直接掐掉,且不与 RL 优化器对抗（避开能力税）。

## 为什么做
- 研究背景：后训练越来越依赖 OPD（Agarwal 2024）与自蒸馏（SDPO-Hübotter 2026、SDFT-Shenfeld 2026、OPSD-Zhao 2026、Sang 2026、why_sd_degrade-Kim 2026）,共同套路是让 teacher 在**特权上下文**（verifier 反馈/专家示范/ground-truth 解）下生成高质量轨迹、student 仅凭 prompt 模仿。这类方法迁移能力很成功,但几乎都 capability-centric,**不管置信度**（§1）。
- 解决的具体痛点：① **误校准的 Scaling Law**——实测连前沿大模型都落过自信区,**能力变强不自动修盲目乐观**（Fig.1,引 Kalai 2025"test-taking mode"——模型被激励输出 plausible falsehood 而非认怂）;② **OPD 本身加剧过自信**——提准确率同时把平均置信推向饱和（Tool Use SDPO 达 0.996）,OCG（过自信缺口）不降反升（§5 Table 1）;③ **RL reward shaping 有"能力税"**——RLCR/CAR（Damani 2025 / Xuan 2026 把 Brier/proper scoring 塞进 PPO）虽压低置信但为躲罚项变过度保守,准确率明显掉（Askell 2021 capability tax）。
- 相关工作 & 各自不足：① **OPD/自蒸馏（SDFT/SDPO/OPSD）**——能力强但不管置信;② **置信校准主流靠 RL + reward shaping（RLCR/CAR）**——有能力税且与优化器对抗;③ **test-time 不确定性估计（SelfCheckGPT/SAC³,多次采样）**——可靠但推理成本 \(O(K)\)。
- 动机链：现状（OPD 迁移能力成功但全场过自信）→ 根因（信息不对称:teacher 开卷低熵、student 闭卷被逼锐化 logits + 继承成功轨迹的笃定）→ 结论"**能力可靠模仿迁移,但'该多确信'不能跨信息状态安全迁移**"→ 把能力和置信在监督目标上拆开,且置信目标应锚到 student 部署成功率 \(\mu(x)\)。为什么不用 RL reward shaping？有能力税且与优化器对抗（§4 原文 "rather than fighting the optimizer"）。为什么不用 test-time 多采样？贵 \(O(K)\)——但可**摊销**到训练阶段（训练时算好 \(\hat\mu\) 蒸进参数,部署单次前向）。
- 与最近邻工作的Δ：vs RLCR/CAR（reward shaping 校准）——CaOPD **不改 reward、不加优化阶段**,纯在蒸馏管线内做 target replacement,因而无能力税;vs 标准 OPD/SDPO——只多一步置信替换,reasoning 段等价;vs why_sd_degrade（Kim 2026,诊断自蒸馏为何退化推理）——同一信息不对称现象的两种切法,此处改监督目标修过自信、彼处只做退化诊断。关键点:把校准当"换监督目标"而非"加奖励罚项"。

## 怎么做 + 靠不靠谱

> 设定（§2.1）：生成格式带 **verbalized confidence**——每条生成 \(y=(a,c)\) 分两段:\(a\)=reasoning 轨迹（含最终答案）,\(c\)=置信段（如 "Confidence: 0.85"）,\(\mathrm{val}(c)\in[0,1]\) 为解析出的标量置信。自蒸馏架构:同一模型两角色——**student** \(\pi_\theta(\cdot\mid x)\)（只见 prompt,模拟部署）;**teacher** \(\pi_\theta(\cdot\mid x,z)\)（额外条件化特权上下文 \(z\sim Z(x)\),可为示范/反馈/ground-truth 解）。结构性质:teacher 见的证据严格多于部署 student。

**【理论:为何能力蒸馏加剧过自信（§3,三命题,Appendix A 给全证）】**
- **理想校准目标（§2.2）**：部署成功率 \(\mu(x):=\mathbb E_{a\sim\pi_\theta(\cdot\mid x)}[R(x,a)]=\Pr_{\pi_\theta}(R=1\mid x)\)（\(R\) 是二值验证函数）。校准要求 \(\mathbb E[\mathrm{val}(c_\theta(x))]=\mu(x)\)，故 **\(\mu(x)\) 才是置信对齐的 ground-truth 目标**。
- **标准 OPD 目标的盲点（§2.3）**：per-token reverse-KL（先采 \(y\sim\pi_\theta(\cdot\mid x)\)，再让 student/teacher 沿该轨迹算分布）
  \(\displaystyle \mathcal L_{\text{OPD}}(\theta)=\mathbb E_{x,z\sim Z(x)}\,\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\Big[\sum_{t=1}^T D_{KL}\big(\pi_\theta(\cdot\mid y_{<t},x)\,\|\,\pi_\theta(\cdot\mid y_{<t},x,z)\big)\Big].\)
  因 \(y=(a,c)\) 含置信 token,该损失也作用在置信位置——而 teacher 拿了 \(z\)（如正确答案）会在置信位输出≈1.0 → **OPD 同时迁移能力 AND 逼 student 模仿 teacher 的无端笃定**。
- **命题1（信息差→不可辨识）**：当条件互信息 \(I(R;Z\mid X)>0\)（特权上下文对正确性有 \(X\) 之外的信息）,teacher 条件成功率 \(\mu_T(X,Z)\) 对 \(X\) 不可测,且平方误差下 **X-可测最优预测恰是 student 部署成功率 \(\mu(X)\)**,残差严格为正:
  \(\displaystyle \min_g \mathbb E\big[(\mu_T(X,Z)-g(X))^2\big]=\mathbb E_X\big[\mathrm{Var}(\mu_T(X,Z)\mid X)\big]>0.\)
  → 用 teacher 的笃定当置信目标在信息论上就是错的。
- **命题2（特权条件→熵坍缩）**：当 \(I(A;Z\mid X)>0\)，teacher 轨迹期望熵严格低于仅给 \(X\) 的条件熵:\(\mathbb E_{X,Z}[H(\pi_\theta(A\mid X,Z))]<\mathbb E_X[H(A\mid X)]\)。最小化对它的 per-token reverse-KL **逼 student logits 人为锐化**（被罚于表达自然不确定性 \(H(A\mid X)\)）。
- **命题3（选择偏置→乐观）**：特权上下文 \(Z\) 常取自成功/高质量样本（\(\mathcal D_{\text{helpful}}\)），使 \(\mathbb E_{Z\sim\mathcal D_{\text{helpful}}}[\mu_T(X,Z)\mid X]\ge\mu(X)\)（正测度集上严格）,于是蒸进 student 的隐式目标对真实部署成功率**系统性上偏**:\(\mathbb E_{X,Z\sim\mathcal D_{\text{helpful}}}[\mu_T(X,Z)-\mu(X)]>0\)。

**【CaOPD 方法流水线（§4 Algorithm 1）——读完可复现】** 对每输入 \(x\)：
1. **student-grounded 置信估计（§4.1）**：采 \(K\) 条独立 \((a_k,c_k)\sim\pi_\theta(\cdot\mid x)\)，verifier 打分,经验成功率
   \(\displaystyle \hat\mu(x)=\frac1K\sum_{k=1}^K R(x,a_k).\)
   **关键省成本**:SDPO 下**复用基础训练循环已生成的 rollout** 算 \(\hat\mu(x)\)，只多一次轻量 verifier 评估;蒸馏轨迹另采、不增额外成本。开放域无 verifier 时 \(\hat\mu\) 退化用 **Teacher-Anchored Self-Consistency**（Appendix B.6）。
2. **target replacement（§4.2,改 completion 与 teacher context、reverse-KL 机制不动）**：给 student 生成轨迹 \(y=(a,c)\)——
   - (i) **改 completion**:把置信段 \(c\) 换成 \(\hat\mu(x)\)，得 \(\tilde y=(a,\hat\mu(x))\)。
   - (ii) **改 teacher context**:在特权上下文 \(z\) 里把原≈1.0 的置信改写成 \(\hat\mu(x)\)，得 \(\tilde z\)。
3. **蒸馏（§4.2 Eq.7）**：student/teacher 各自条件化（\(x\) 与 \((x,\tilde z)\)）给 \(\tilde y\) 打分;token 位置分 reasoning \(I_a=\{1,\dots,T_a\}\) 与置信 \(I_c=\{T_a+1,\dots,T\}\)：
   \(\displaystyle \mathcal L_{\text{CaOPD}}(\theta)=\mathbb E_{x,\tilde z}\,\mathbb E_{\tilde y}\Big[\underbrace{\sum_{t\in I_a}D_{KL}\big(\pi_\theta(\cdot\mid\tilde y_{<t},x)\,\|\,\pi_\theta(\cdot\mid\tilde y_{<t},x,\tilde z)\big)}_{\text{Capability Cloning(保留)}}+\underbrace{\sum_{t\in I_c}D_{KL}\big(\pi_\theta(\cdot\mid\tilde y_{<t},x)\,\|\,\pi_\theta(\cdot\mid\tilde y_{<t},x,\tilde z)\big)}_{\text{Confidence Calibration(student-grounded)}}\Big].\)
   - **reasoning 位置 \(t\in I_a\)**:前缀 \(\tilde y_{<t}\) 只含 reasoning token（置信替换还没出现）、\(\tilde z\) 的 reasoning 内容未改 → **per-token KL 与标准 OPD（Eq.2）实质相同**（能力克隆原样保留）。
   - **置信位置 \(t\in I_c\)**:打分的 token 编码 \(\hat\mu(x)\) 而非原 \(c\)、teacher context 反映 \(\hat\mu(x)\) 而非 1.0 → student 被训去产出 student-grounded 经验成功率,**直接消解熵坍缩(命题2) + 乐观偏置(命题3)**。
   - AdamW 更新。整套**无 reward 改造、无额外优化阶段**。
- **逐组件必要性**：
  - **置信段替换**——核心,没它就是标准 OPD（过自信）。
  - **特权上下文里也改置信（步 (ii)）**——保证 teacher 在置信位置不再给≈1.0 的目标（否则 reverse-KL 仍把 student 拉向笃定）。
  - **\(K\) 次 rollout 估 \(\hat\mu\)**——§K 消融:准确率对 \(K\in\{1,\dots,32\}\) **平坦**（能力与置信采样方差解耦）,\(K\ge8\) 是校准甜点,K 太小目标过度量化反困在过自信。
  - reasoning 段不动是**设计而非消融对象**（理论上等价 OPD）。
- **关键机制/公式（直觉）**：三命题锚定"为什么必须这样"——\(\mu(X)\) 是唯一 X-可测的无偏置信目标（命题1）;teacher 低熵会逼锐化（命题2）;成功筛选带来上偏（命题3）。target replacement 的优雅在于:**reasoning 前缀与 teacher 内容不变 ⇒ 能力克隆等价 OPD;只换置信段目标 ⇒ 校准被结构性纠偏**,且不与优化器对抗（不像 RL 罚项要在 reward 里拉扯）。test-time 的昂贵置信估计被**摊销**进单次前向模型参数。
- **实验与证据（§5）**：两域 Science Q&A（Chemistry,SciKnowEval/Feng 2024）+ Tool Use（ToolAlpaca）;主模型 Qwen3-8B、Olmo-3-7B-Instruct;scaling 覆盖 Qwen3 全家 0.6B→32B（Appendix C.1）;指标 Accuracy + ECE/Brier + 自引 **OCG**（mean confidence − accuracy,量化过自信方向,正越大越过自信）+ **SPR**（Strict Pairwise Ranking,\(P(c^+>c^-)\)，对置信饱和/打平**零分重罚**,比 AUROC 更严）。
  - **OPD 加剧过自信（Table 1）**：Qwen3-8B Science Q&A base OCG +58.7%,SDFT/SDPO 推到 +48.1%/+12.9%;Tool Use SDPO 平均置信 0.996（OCG +32.0%）、SDFT +32.4%。
  - **CaOPD 结构纠偏不掉能力（Table 1/2）**：Qwen3-8B Tool Use OCG 从 SDFT +32.4%→**−0.7%**;主表（vs SDFT）Acc 67.6→70.6、ECE 0.321→0.228、BS 0.320→0.242、**SPR 0.085→0.555**（SDFT 几乎丧失区分度,CaOPD 恢复）;Science Q&A vs SDFT Acc 49.1→50.0、ECE 0.486→0.266、SPR 0.387→0.599。Olmo 同向（Science Q&A ECE 0.429→0.176）。
  - **避开能力税**：vs SDPO 在 Qwen3-8B Tool Use Acc 66.2%→70.9% 同时 ECE 0.298→0.133,而 RLCR/CAR 校准好但准确率明显更低。
  - **成本（Fig.2）**：每步耗时与 SDPO 几乎一致。**OOD**（Tool Use→Chemistry）SDFT ECE 0.599、CaOPD 0.358。**持续学习(CT)** 解决"校准遗忘"（CT ECE 0.126、SPR 0.662）。
  - baseline 较公平（SDFT/SDPO/GRPO + 专门校准的 RLCR/CAR 都比）。
- **假设与失效边界**：【原文】(§Limitations) 依赖**可验证 verifier** 估 \(\hat\mu\)（开放域退 self-consistency,质量待考）;依赖**可解析的 verbalized confidence 格式**（测试偶发格式失败）;SFT 下不一定能像 SDPO 那样零成本复用 \(K\) 次 rollout;**能力上界受 base 约束,CaOPD 不提升推理能力只校准置信**;仅 utterance-level 置信,step-level 校准是 future work。【推断】校准收益几乎全来自置信段,准确率提升主要来自底层 OPD/SDPO 而非 CaOPD 机制本身。
- **祛魅总结**：【推断】真贡献是 **干净的诊断（OPD→过自信的三命题信息论解释）+ 极小工程改动的解法（只换置信段监督目标,reverse-KL 不动,无 reward 改造无额外阶段）**——target replacement 是优雅且可复现的。要祛魅的边界:① **方法只对"显式 verbalized confidence"格式生效**,不是给模型注入某种内在校准能力;② **对推理能力本身零直接增益**（命题/实证都只动置信段）,标题"decoupling capability and calibration"准确但易被读成"两头都提"——其实能力提升全靠底层 OPD;③ 命题1的"\(\mu(X)\) 是最优目标"是理论层面,实证只能侧面印证（换 \(\hat\mu\) 后 ECE/BS 降）。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=reasoning 段 teacher 的 per-token reverse-KL（能力克隆）+ 置信段的 student-grounded 经验成功率 \(\hat\mu(x)\)（校准）｜**改什么**=参数（logits 分布,尤其置信 token 位置）+ **监督目标的置信片段**（target replacement,改 completion + teacher context）｜**何时改**=在线 per-prompt（先 \(K\) 次 rollout 估 \(\hat\mu\) → 替换 → 蒸馏）,训练时摊销 test-time 估计｜**免梯度?**=否（reverse-KL 蒸馏梯度下降）｜**记忆-技能生命周期**=不适用（无外部记忆/技能库）｜**防遗忘机制**=持续学习(CT)下解决"**校准遗忘**"（CT ECE 0.126、SPR 0.662）,但非参数级防遗忘机制,是 target 解耦带来的副产物
- ⑦ 开源代码+框架/harness：https://github.com/SalesforceAIResearch/CaOPD （旧 analysis 记已 clone 约 40M,含 `main.py`/`distil_trainer.py`/`distil_config.py`/`eval_science.py`/`eval_tooluse.py` + Tool Use 与 Chemistry 两域数据 + Salesforce 合规文件）。框架=**TRL 0.24 + vLLM 0.12 + DeepSpeed 0.18 + transformers 4.57**（自建 distil trainer,在 SDFT/SDPO 自蒸馏管线上做 target replacement）。【待核】依赖版本与目录细节复用旧 analysis 核查,本次基于 PDF 重写未重新进仓核对。代码可得性:训练+评测+数据齐全。
- 💰 资源/成本与可扩展性：【原文】每 prompt 需 \(K\) 次 rollout（K=8 够用,准确率对 K 平坦）;**SDPO 下复用已有 rollout,几乎零额外成本**（只多轻量 verifier 评估）,每步耗时与 SDPO 几乎一致（Fig.2）;SFT 下不一定零成本复用。scaling 0.6B→32B 均有效。
- 🎯 对"探索-巩固"对标：**支撑 + 警示（边界提供者）**。判定:CaOPD 揭示的"**信息不对称→teacher 笃定不能安全迁移给 student**"对 TSRD 是关键警示——teacher 当脚手架（开卷）产生的轨迹/置信不能让 student（闭卷）盲目模仿,否则继承不属于自己的确定性;这正对应"巩固"环节要固化的是 student **自己走得通**的东西（\(\hat\mu(x)\) 用 student 自采 rollout 估）,而非 teacher 的特权产物。**可借组件**:① "student-grounded 目标"思想——TSRD 巩固时也应用 student 自采成功率/自选恢复分支作目标,而非直接抄 teacher;② target replacement 这种"只改某段监督目标、底层机制不动"的低侵入手法,可移植到 path-recovery（只替换接管点的目标,reverse-KL 不动）。**缺口/竞品面**:只解决置信校准、不碰推理能力,仅 utterance-level、无 step-level;无 MTP/前瞻;与"探索/选路"关系弱（主战场是巩固阶段的"该多确信"）。
- 🔭 开放问题/未来方向：【原文】long/agentic 多步推理的 **step-level 校准**是 future work;开放域无 verifier 时的 \(\hat\mu\) 近似质量（self-consistency）;verbalized confidence 格式鲁棒性。【推断】把"student-grounded target replacement"推广到置信之外的其他可解耦属性（如安全/风格）;与 step/turn 级信用分配结合做细粒度校准;与 MTP 前瞻结合预测"该多确信"。

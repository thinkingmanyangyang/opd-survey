open_agentrl | RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System | Gen-Verse / Princeton（Yinjie Wang, Tianbao Xie, Ke Shen, Mengdi Wang, Ling Yang 通讯） | 2026-02-03·arXiv·v1（仓库 README 标 ICML 2026,**PDF 自身未声明任何 venue**——已核实正文/页眉无 ICML/NeurIPS 字样,保持已修正结论勿回退） | 主题线 L4(Agent/工具·多轮自进化)·相关性 中

**原始论文**:https://arxiv.org/abs/2602.02488 （代码 github.com/Gen-Verse/Open-AgentRL）

## 一眼看懂
> 一句话导读:训 agent 时通常只优化策略,但这篇把策略、给过程打分的奖励模型、出题的环境三样东西一起在线进化——出题的难度跟着策略能力自动调,奖励模型跟着策略轨迹一起练,三者互相喂反馈、一起变强。这个方向和蒸馏主线没有方法交集,主要作为"过程信号有价值 + 环境难度该自适应"的外部论据。

- 🟦 TL;DR:这是一个"完全动态"的闭环 RL 系统,**同时进化环境、策略、生成式奖励模型三者**。具体是:
  - 策略用 step-wise + outcome 融合的反馈来训练;
  - 奖励模型经一致性反馈做联合优化,产出可靠的逐步监督;
  - 环境则根据策略当前的能力,自适应地调整题目难度。
  它还论证了一个判断:**优化后的逐步奖励信号,优于人工写的 outcome 标签**。【原文 Abstract+§2】
- 最巧的一步:**把"环境难度自适应"用理论(Theorem 1&2)接进闭环**。看反证就清楚了:如果抽掉环境自适应,奖励模型的训练环境(它是由策略轨迹诱导出来的)在任务过难或过易时,会让 \(p_+,p_-\) 的重要性采样极不平衡,从而违反 reward-precision 的目标 \(\mu=p_++p_->1\)(Theorem 2);奖励模型质量一掉,策略也跟着学不好。而"当 acc 落在 \([0.2,0.8]\) 之外时就调难度,再加一道质量门控"这一招,正好把这条因果链闭合,使三者互相放大(Table 1 显示:每加一个动态组件,所有指标都一致提升)。【原文 §2.3-2.4 + Table 1】

## 为什么做
> 一句话导读:RLVR 在长轨迹任务上有个老毛病——答完一整条只给一个对/错,信号太稀疏。补救办法是请一个"生成式奖励模型"逐步打分,可这个奖励模型本身又难训。本文的观察是:这三件事(策略学不好、奖励模型难训、题目难度不合适)其实是同一个闭环里互相牵连的,所以干脆一起动态优化。

- 研究背景:RLVR 能提升 LLM 推理;但当策略与环境做长轨迹的迭代交互时,二值的 outcome 奖励监督就不够用了。补救的 step-wise 信号通常由生成式奖励模型给出(它借助语言模型的推理能力,效果优于标量奖励模型),但训练这类模型需要高质量、任务专属的监督。此外,环境质量(任务难度与策略能力是否匹配、任务是否多样)对 RL 的扩展同样关键。【原文 §1】
- 解决的具体痛点:
  - ① 长轨迹下,二元的 outcome 奖励信号不足;
  - ② 奖励模型与策略相互割裂,而且生成式奖励模型难以获得可靠的 step-wise 监督;
  - ③ 环境的任务难度与策略当前能力不匹配,会同时拖累策略和奖励模型的训练动态;
  - ④ 三个要素(环境 / 策略 / 奖励模型)通常被分开处理。【原文 §1】
- 相关工作 & 各自不足（来龙去脉 + 并行路线 + 精确差异）:
  - **RLVR 单 outcome 奖励**(可联合优化奖励模型 + 策略):短板是不能直接扩到多轮(缺 step-wise 监督)。
  - **调难度类**(Zeng 2025 的多难度引擎、Xue 2026 的可验证任务合成):它们能改善策略,但短板是**缺长 horizon 交互所需的 step-wise 信号**,而且常常是盲调难度。
  - **生成式过程奖励模型(PRM)**:借助 LM 推理来给逐步分,优于标量奖励模型;短板是训练它需要高质量、任务专属的监督。
  - **精确差异**:本文是首个把**环境 / 策略 / 奖励模型三者同时动态、互为反馈**的闭环,并用 Theorem 1&2 把"调难度"与"奖励模型精度"连成因果关系(即 targeted 有针对性、而非盲调)。它还论证 step-wise 的(奖励模型)信号可以**替代、甚至超过人工写的 outcome 标签**(§3.2.6),从而为大规模自进化 agent 去掉人工评测脚本铺路。【原文 §4.1-4.2 + §2 + §3.2.6】
- 动机链（一步步推下来）:
  - 单 outcome 在长轨迹下太稀疏,所以需要 step-wise 信号(来自生成式奖励模型);
  - 但奖励模型本身需要可靠监督、且与策略割裂;
  - 于是让策略的轨迹去充当奖励模型的训练环境(即 consistency feedback,一致性反馈);
  - 但奖励模型质量又取决于任务难度是否平衡(Theorem 1&2);
  - 所以环境也得根据策略能力动态调难度;
  - 最终三者形成闭环。【原文 §1+§2】
- 与最近邻工作的 Δ:
  - vs 标准 agentic RL/RLVR:它不是只优化策略,而是**让环境 / 策略 / 奖励模型三者同时动态、互为反馈**;
  - vs 调难度类工作:它的难度调整是由"奖励模型对失败步的诊断摘要"驱动的(有针对性),而非盲调;而且它有理论说明,为什么调难度会同时利好奖励模型。【原文 §2+§3.2.6】

## 怎么做 + 靠不靠谱
> 一句话导读:一轮就是"策略对每道题采几条轨迹→奖励模型给每一步打分→把分数同时拿去训策略和训奖励模型→再看这题的通过率,过高就出难一点、过低就出易一点(并过一道质量关)"。靠不靠谱?逐组件消融(每加一个动态件都涨)较公平,但 step reward 把 outcome 直接加进了每一步,所以"step-wise 优于 outcome"的对比要在这个耦合下解读;而且三个组件全由 LLM 自评/自改写,可能积累偏置。

- 方法流水线(输入→输出,对应 Algorithm 1,读完即可复现):一轮分五步——
  - ① 采一道任务 \(q\sim Q\);策略 \(\pi_\theta\) 采出一组轨迹 \(\mathbb T_q\),每条 \(\tau=(\tau_1,\dots,\tau_T)\),其 outcome \(O_\tau\in\{-1,1\}\);
  - ② 奖励模型 \(r_\phi\) 对每一步 \(\tau_i\) 独立采 \(m=3\) 次,得到评估推理 \(r_{\tau_i,j}\) 与最终分 \(S_{\tau_i,j}\in\{-1,1\}\);
  - ③ 算 **step reward(Eq.1)** \(R_{\tau_i}=O_\tau+\lambda\sum_{j=1}^m S_{\tau_i,j}/m\)(取 \(\lambda=1\));在同一步序号 \(i\) 上、跨各条轨迹 \(\tau\in\mathbb T_q\) 做标准化,得到 advantage \(A^{\pi_\theta}_{\tau_i}\),用来训策略;
  - ④ **训奖励模型(Eq.2)**:把 \(R_{\tau_i}\cdot S_{\tau_i,j}\) 跨 \(j\) 标准化,得到 \(A^{r_\phi}_{\tau_i,j}\),用来训它的评估推理 \(r_{\tau_i,j}\);
  - ⑤ **环境自适应**:根据这道题的 rollout 通过率来调难度——
    - 若 \(\mathrm{acc}(q)>\alpha_{\text{high}}=0.8\),就提议一道更难的 \(q'\leftarrow\text{harder}(q;s)\);
    - 若 \(\mathrm{acc}(q)<\alpha_{\text{low}}=0.2\),就提议一道更易的 \(q'\leftarrow\text{easier}(q;s)\);
    - 其中摘要 \(s\) 只由"失败步(即 \(S_{\tau_i,j}=-1\) 的那些 \(\tau_i\))的评估摘要"喂给一个 LM(Qwen3-4B)生成;
    - 还有一道**质量门控**:变难只在 \(\alpha_{\text{low}}<\mathrm{acc}(q')<\mathrm{acc}(q)\) 时才接受,变易只在 \(\mathrm{acc}(q)<\mathrm{acc}(q')<\alpha_{\text{high}}\) 时才接受;通过后才用 \(q'\) 替换 \(q\);
  - 然后循环 \(K\) 步。【原文 Algorithm 1 + §2.1-2.4】
- 逐组件必要性（每模块干啥 / 有无消融）:
  - **integrated feedback(step + outcome 融合,Eq.1)**:有消融(Fig.6a)。因为 outcome-only 在长轨迹下太稀疏,所以这个融合是必要的,尤其在长 horizon 下。
  - **联合优化奖励模型(Eq.2)**:Table 1 里,从 Policy 到 Policy+Reward 是一致提升的(而且 reward 的 process/outcome acc 都上升)——所以必要。
  - **环境自适应**:Table 1 里,从 Policy+Reward 到 +Env 进一步提升了所有指标(尤其是 OOD 与奖励模型 acc),Fig.4 的训练曲线也收敛得更高——所以必要,且 Theorem 2 给了它理论依据。
  - **\(\lambda\)(step 与 outcome 的权衡)**:有消融(Table 2,Alf World)。策略在 \(\lambda=1\) 时 acc 最优(54.1);\(\lambda\) 越大,奖励模型的 process acc 越高、但 outcome acc 略降——默认取 \(\lambda=1\)。
  - **质量门控 / 接受机制**:Fig.7a + §3.2.9。用更强的模型(Qwen3-VL-32B / Qwen3-32B)去验证,被接受任务的 pass-at-least-one 分别是 96.0% / 96.7% / 94.2%——证明它确实过滤掉了错误合成的任务。
- 关键机制/公式（真实形式 + 直觉,从 PDF 抄准）:
  - **step reward(Eq.1)**:\(\displaystyle R_{\tau_i}=O_\tau+\lambda\,\tfrac1m\textstyle\sum_{j=1}^m S_{\tau_i,j},\quad \lambda>0,\ i=1,\dots,T.\) 直觉:把序列级 outcome \(O_\tau\) 与该步多次评分的均值融合,长 horizon 下给每步稠密信号。
  - **reward-model 监督优势(Eq.2)**:策略优势 \(A^{\pi_\theta}_{\tau_i}=\mathrm{standardize}_{\tau\in\mathbb T_q}(R_{\tau_i})\);奖励模型优势 \(A^{r_\phi}_{\tau_i,j}=\mathrm{standardize}_j(R_{\tau_i}\cdot S_{\tau_i,j})\)——即奖励模型的"评估推理"被其与 step reward 一致性的信号驱动(consistency feedback)。
  - **reward precision(目标量)**:\(\displaystyle A=P\big(S_{\tau_i^+}>S_{\tau_i^-}\mid O_{\tau^+}=1,\,O_{\tau^-}=-1\big),\quad S_{\tau_i}=\tfrac1m\textstyle\sum_{j=1}^m S_{\tau_i,j}.\) 即奖励模型不仅要判单步对错,还要预测该步对最终 outcome 的影响(对成功轨迹的步给更高分)。
  - **Theorem 1**:记 \(\mu\triangleq p_++p_-\),其中 \(p_+=P(S_{\tau_i^+,j}=1)\)、\(p_-=P(S_{\tau_i^-,j}=-1)\)。结论是:当 \(m\to\infty\) 时 \(A\to1\) **当且仅当** \(\mu>1\);并且在 \(\mu>1\) 时有 \(A\ge 1-e^{-m(\mu-1)^2/4}\)。直觉:用来估计 \(p_+,p_-\) 的采样密度必须平衡,否则估计量会被其中一类主导、评估就有偏。
  - **Theorem 2**:奖励模型的 RL 目标可以写成 \(\mathbb E_{q,\tau\sim\pi_\theta}\mathbb E_{S_{\tau_i,j}\sim r_\phi}[RS_{\tau_i,j}]=4\,\mathbb E_q[\langle p_+,f_+\rangle+\langle p_-,f_-\rangle]+C\)(此处取 \(\lambda=1\),\(\langle\cdot,\cdot\rangle\) 是在 \(\tau\sim\pi_\theta(\cdot\mid q)\) 上的 \(L_2\) 内积)。其中重要性权重的范数比会走向两个极端:当 \(P(O_\tau=-1\mid q,\pi_\theta)\to1\)(任务过难)时 \(\|f_+\|/\|f_-\|\to0\);当 \(P(O_\tau=1\mid q,\pi_\theta)\to1\)(任务过易)时 \(\|f_+\|/\|f_-\|\to\infty\)。直觉:任务过难或过易,都会让重要性采样在 \(p_+,p_-\) 之间极不平衡,**从而破坏 Theorem 1 要求的 \(\mu>1\) 目标**。这正是为什么"环境自适应"(把难度维持在"有对有错"的平衡区)既利于策略、也利于奖励模型。【原文 §2.3 / Theorem 1&2 / §2.4】
- 实验与证据:在三个场景上做——
  - ① GUI(OSWorld):策略和奖励都用 Qwen3-VL-8B-Thinking,任务改写用 Qwen3-4B;RL rollout 最大 30 步、评测 50 步;每步采 12 任务 × 8 rollout,奖励模型每步评 3 次;训练用 OSWorld-verified,并排除 Multiple Apps/Chrome 留作 OOD。
  - ② Alf World:策略 Qwen2.5-7B-Instruct、奖励 Qwen2.5-14B-Instruct、改写 Qwen3-4B;RL rollout 最大 40 步、评测 60 步;每步采 16 任务 × 8 rollout。
  - ③ coding:模型组合同 Alf World,无交互环境;每步 64 任务 ×(32 个解 + 32 个单元测试)。
  - 关键数字(Table 1,Policy+Reward+Env 相对 Before):GUI 的 in-domain +11.7(40.4→52.1)、OOD +5.2;LLM agent 的 in-domain(40.4→…)、OOD **+18.7**;reward 的 process/outcome acc 同步上升。
  - abstract 口径:OSWorld **+9.1%**(Qwen3-VL-8B)、Alf World +18.7%、LiveBench +11.9%(Qwen2.5-7B)。
  - baseline 公平性:逐组件消融(每加一个动态组件)较公平;但 step reward 把 \(O_\tau\) 直接加进了每一步(Eq.1),所以 step-wise 与 outcome 之间存在部分耦合。【原文 Table 1+§3.2.4】
- 假设与失效边界:
  - 【原文 §3.2.6】用"仅靠奖励模型的 step-wise 监督、不给 outcome 脚本"来训练,效果甚至超过用 verifiable outcome——但这是在 GUI 这种"需要人工写评测脚本"的设定下成立的;它隐含的假设是奖励模型能可靠地评每一步。
  - 【推断】几点局限:
    - reward precision 的理论是在 \(m\to\infty\) 下的渐近结论,而有限的 \(m=3\) 下偏差有多大没有充分量化;
    - 环境改写依赖 LLM 的质量,以及阈值/门控的设定;
    - 奖励模型和环境改写都由 LLM 担任,存在"自评 / 自改写"的潜在偏置;
    - 评测只集中在 OSWorld / Alf World / coding 三个域。
    论文有 Conclusion,但 Limitations 写得较简。
- 祛魅总结:【推断】
  - 真贡献:
    - ① 把"环境-策略-奖励模型"三者纳入一个有理论支撑的闭环(Theorem 1&2 把 reward precision 与任务难度平衡连成了因果),这个 framing 较新、也自洽;
    - ② "优化后的 step-wise 信号可替代、甚至超过人工 outcome 标签"(§3.2.6)这一结论若稳健,对去掉人工评测、扩展 agentic 环境有实际意义。
  - 被适度高估之处:
    - step reward 把 outcome 直接加进了每一步(Eq.1),所以"step-wise 优于 outcome"的对比要在这个耦合下来解读;
    - 三个组件全由 LLM 自评/自改写,可能积累偏置;
    - 它与 OPD/蒸馏主线**没有直接关联**(本文不含任何蒸馏),只能作为"过程信号有价值 + 环境难度该自适应"的外部论据。

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号:策略学 step+outcome 融合优势(Eq.1);奖励模型学"评估推理"的一致性优势(Eq.2);环境学"据 acc 调难度"。
  - 改什么:策略 \(\pi_\theta\)、奖励模型 \(r_\phi\) 两套参数 + 环境任务集 \(Q\)(任务被改写替换)。
  - 何时改:每个 RL step 三者同步更新;环境仅当 acc 落在 \([0.2,0.8]\) 外且通过门控时改任务。
  - 免梯度?策略/奖励模型走梯度(GRPO 类 group-based);环境自适应是**免梯度**的 LLM 任务改写 + 阈值门控。
  - 记忆-技能生命周期:无显式记忆/技能库;"经验"以"奖励模型诊断摘要→改写任务"的形式驱动 active learning from experience;任务集 \(Q\) 随训练演进(接受任务近似线性增长)。
  - 防遗忘机制:原始训练任务保留在所有 setup(§3.1.4 为公平比较);无显式防遗忘机制(非持续学习设定)。
- ⑦ 开源代码+框架/harness:github.com/Gen-Verse/Open-AgentRL(已 clone,~38M,同仓库另含 DemyAgent;HF 有 Policy & Reward 模型 collection)。框架 **veRL**(仓库内置 verl/ 目录,GRPO 类 group-based RL);含 recipe/、train/、reward/、按任务的 alfworld_rl.py/coding_rl.py/osworld_rl.py 训练入口与 *_eval.py,以及 OSWorld-main/、alfworld_master/ 环境;requirements_rlanything.txt / requirements_sglang.txt / requirements-npu.txt。代码可得性高(完整训练/评测/环境脚本齐全)。【原文/仓库】
- 💰 资源/成本与可扩展性:多模型同时在线(策略+奖励模型+任务改写 LM Qwen3-4B,Alf World 奖励模型 14B 大于策略 7B);GUI 每步 12×8 rollout、奖励每步评 3 次,长 horizon(GUI 30 步/Alf 40 步 RL rollout);GUI 预造扰动版任务(含 evaluator/verifier 文件)。具体 GPU 数/卡时论文正文未明说。【原文 §3.1】
- 🎯 对"探索-巩固"对标:**间接支撑(在探索这一侧)+ 一个可借的环境调度组件**。
  - 一句判定:本文与 OPD/蒸馏主线没有方法交集,但它的"**环境难度自适应**"恰好是"探索 = 发现有效行为/路径"在环境侧的实现——把任务维持在"有对有错"的平衡区(\(\mu>1\),Theorem 2),正对应"偏向自己能走通的开头/边界"。依据是 Theorem 2 + §2.4 那套据 acc 调难度的做法。
  - 可借组件:
    - ① **据 rollout acc 的难度门控调度**(\(\alpha_{\text{low}}/\alpha_{\text{high}}\) + 质量门控):可直接用于"探索-巩固"的课程,把训练样本维持在学生能力的前沿;
    - ② **"奖励模型对失败步的诊断摘要 → 有针对性地改写任务"**(Fig.3):这与"识别走偏的关键步、再给针对性脚手架/hint"的思路相通,可看作 path-recovery 在环境侧的类比;
    - ③ step-wise 过程奖励对长 horizon 的必要性(Fig.6),支撑了"过程级信号优于纯结果"。
  - 缺口:
    - ① 没有任何蒸馏 / 特权信息 / on-policy 自蒸馏(它是纯 RL + 生成式奖励模型);
    - ② 没有 MTP/前瞻;
    - ③ 没有"把能力固化进参数/记忆、且不遗忘"的巩固机制(任务集演进 ≠ 参数级巩固);
    - ④ step-wise 信号来自一个外部奖励模型,而不是"看过答案的自己",所以与 teacher-scaffold 的思路不同源。
- 🔭 开放问题/未来方向:
  - 【原文 abstract+§3.2.6】enabling large-scale self-evolving agents in real-world environments(用奖励模型替人工评测脚本、扩环境);新环境任务线性扩展。
  - 【推断】两个方向:
    - 把"奖励模型诊断摘要 → 改写任务"这套有针对性的难度调度,与 OPD 的"teacher 稀疏脚手架"结合——用奖励模型定位策略走偏的关键步,既调环境难度、又触发 path-recovery 监督;
    - 把 MTP 前瞻接到奖励模型的"预测该步对 outcome 的影响"上(reward precision \(A\) 本质上就是一种前瞻),这可能让过程奖励更早、更准地定位到关键的分支步。

RETURN: open_agentrl|读到PDF=是(标题RLAnything;§1-2方法+Algorithm 1+Eq.1-2+Theorem 1&2+reward precision/§2.4环境自适应/§3.1-3.2.6实验Table1)|venue已核=PDF无ICML/NeurIPS字样(README标ICML2026),保持已修正结论|L线=L4|对标=间接支撑(探索侧:环境难度自适应=维持能走通的边界,可借难度门控+失败步诊断改写;但无蒸馏/MTP/参数级巩固,过程信号来自外部奖励模型非"看过答案的自己")|残留待核=0

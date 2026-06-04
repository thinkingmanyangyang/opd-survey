scaf_grpo | Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning | HKUST / CUHK / HKU(Xichen Zhang*, Sitong Wu*, …, Jiaya Jia 通讯;*共同一作) | 2026-02·ICLR 2026·v2(2026-02-28, cs.CL) | 主题线 L3(RLVR/GRPO 家族)+ L1(教师引导)·相关性 高

**原始论文**:https://arxiv.org/abs/2510.19807 （arXiv:2510.19807v2;ICLR 2026 接收）

## 一眼看懂

> 一句话导读:RLVR 训练遇到太难的题时,所有采样答案全错、组内优势归零、梯度消失,难题就成了训练"看不见"的死角(作者叫"学习悬崖")。Scaf-GRPO 的招是:只在卡死时,像老师搭脚手架那样,从抽象到具体一点点给学生提示,直到学生自己解出来,再把这条对的塞回去重启学习——而且提示只进题面、答案仍由学生自己产,所以全程还是 on-policy。

- 🟦 TL;DR:RLVR(可验证奖励的 RL,如 DeepSeek-R1 范式)有个"**学习悬崖**"问题,链条是:
  - 题目远超当前能力时,所有 rollout 全错;
  - 于是 GRPO 里这组 advantage \(\hat A_i=\frac{R(o_i)-\mu_G}{\sigma_G+\epsilon_{\mathrm{std}}}\) 因为 \(\mu_G=\sigma_G=0\) 而全部归零;
  - advantage 归零 → 梯度消失;
  - 这些难题对训练"隐形",卡在长尾。
- Scaf-GRPO 的招:只在**学习真正停滞**时,按"知识 → 规划 → 解答"三层、由抽象到具体地**增量注入 in-prompt 提示**(提示只写进题面),直到**当前策略 \(\pi_\theta\) 自己**在带提示的 prompt \(q\oplus h^*\) 上采出一条正确轨迹 \(o^*_h\);再用它**替换 rollout buffer 里某条失败轨迹**,从而恢复非零 advantage、让学习重新启动。
  - 因为这条成功轨迹仍是当前策略采的、重要性比率也是对"带提示的 prompt"算的,所以**全程 on-policy**(而不是 teacher 续写 golden 前缀那种 off-policy)。
- 最巧的一步:**抽掉"用 \(\pi_\theta\) 自采样的 \(o^*_h\) 替换失败轨迹、且比率对 \(q\oplus h^*\) 计算"这一步,整个方法就垮成 off-policy 前缀法**。
  - 这个方法并不改 GRPO 的损失形式(§3.2 明说"does not alter the mathematical form…only modifies the data");真正的魔法在于"提示只进 prompt、答案仍由学生自己产、比率对带提示的 prompt 算"。
  - 这一步同时保住了两样东西:on-policy 一致性(避免前缀法那种师生分布失配)+ 探索自主性(提示是路标,不是铁轨)。
  - 为什么是它:Eq.4 那个分段比率(\(o^*_h\) 的比率用 \(q\oplus h^*\) 而非 \(q\))正是论文反复强调的"on-policy integrity"所在;附录 F.5 还专门证了:若用 \(\pi_\theta(\cdot|q)/\pi_{\theta_{\mathrm{old}}}(\cdot|q\oplus h^*)\) 这种失配的比率,训练会崩。

## 为什么做

> 一句话导读:RLVR 在超出当前能力的难题上学不动(学习悬崖)。前人主流办法是给学生一段标准答案的前缀让它续写,但这会带来师生分布失配、还把模型逼上预定路径、扼杀探索。Scaf-GRPO 借教育学里的"脚手架"思想——只在卡死时给临时、最小、会撤除的分层提示,并让提示只指方向、不铺铁轨。

- **研究背景的来龙去脉**:DeepSeek-R1 确立了 RLVR——用外部 verifier 给 outcome-based(对 / 错)的奖励,模型就能自主习得推理策略,免去逐步人工标注(§1/§2)。后续分两条线:
  - (a) **稳定性 / 去偏**(Dr.GRPO、DAPO,改归一化 / clip);
  - (b) **更稠密 / 更省的奖励**(用长度惩罚抑制 overthinking,如 L1 / RECUT;token 级稠密反馈,如 Wang 2025a/b 的 dual-token / 高熵 minority token)。【原文】§2 RLVR 段
- **解决的具体痛点(学习悬崖,本文靶子)**:超出能力的题持续全错 → 零奖励 → GRPO 的 advantage 在全零时 \(\mu_G=\sigma_G=0\Rightarrow\hat A_i=0\Rightarrow\) 梯度消失,于是难题"隐形",形成长尾瓶颈(§1 + Fig.2 实证:Qwen2.5-Math-1.5B 上 vanilla GRPO 在零奖励题上 plateau)。这就挡住了模型用最难的样本继续往更高水平提升。【原文】§1、Fig.2
- **并行技术路线一(off-policy 教师引导 / 前缀续写)及各自具体短板**:主流应对学习悬崖的办法,是给 student 一段 "golden" 轨迹的**前缀**、让它续写——
  - **LUFFY**(Yan 2025):把整条专家轨迹和多条 model rollout **混在同一个 batch** 里;
  - **Huang 2025**:用 **cosine decay** 调引导前缀的长度,且混合 SFT-RL 目标;
  - **StepHint / Zhang 2025a**:给**多级不同长度**的 hint,从多个起点探索;
  - **BREAD / Zhang 2025b**:从 expert anchor 分支 rollout,桥接 SFT 与 RL。
  - 这类做法有两弊:
    - (1) teacher 前缀与 student 后缀**分布失配**(一条轨迹里混了两个分布),需要 policy-shaping / 混合 SFT-RL 之类的**复杂补丁**,引入偏置且不稳;
    - (2) "**on-rails**"(上铁轨)——把模型逼上预定的路径,扼杀了它对替代 / 更优策略的探索。【原文】§1、§2 off-policy 段
- **动机链**:RLVR 好,但难题学不动(悬崖)→ 前缀续写虽能给正奖励,却带来失配 + 扼杀探索 → 于是借教育学里的**脚手架**(scaffolding,Berk & Winsler 1995):只在停滞时给"会随能力提升而撤除的、临时最小的分层"引导,且让"题 + 提示"在**同一个统一策略**下处理(避免失配)、提示只指方向、不定死路径(保住探索)。
  - 为何不用更简单的现成做法:前缀法虽现成,却破坏 on-policy 一致性、需要复杂补丁(§3.2 + 附录 F.5 证失配比率会训崩)。【原文】§1、§3.2
- **与最近邻(LUFFY / StepHint 多级 hint)的精确差异**:Scaf-GRPO 最像它们,但**关键差一点**——它的提示**只进 prompt(in-prompt)、答案完全由当前策略自己采**,而且重要性比率是对"带提示 prompt"计算的 → 严格 on-policy;而前缀法是把 teacher 的 token 直接拼进轨迹。
  - 它的好处:既补上了悬崖处缺的正奖励信号,又指向"模型自己够得着的最高效路径",还不引入失配偏置(主结果相对 LUFFY +9.2%、GPQA 这个 OOD 上与 LUFFY 持平甚至超出,§4.2/Table 6)。【原文】§1、§4.2

## 怎么做 + 靠不靠谱

> 一句话导读:流程是——训练前期先纯探索、把"真难题"和"多练就会的假难题"区分开;遇到真难题且整组全错时,从最抽象的提示开始一点点加,直到学生自采出一条对的;用这条替换一条失败轨迹、重算优势,且对它的重要性比率用"带提示的题面"来算,保住 on-policy。要冷静看:消融做得扎实,但 hint 由更强的 DeepSeek-R1 生成(本质是教师知识下放,"零外部知识"不成立),且只验证了数学域、主结果报的是 best checkpoint。

- **GRPO 前置(§3.1)**:对 prompt \(q\),策略 \(\pi_\theta\) 采 \(N\) 条轨迹 \(G=\{o_1,\dots,o_N\}\),verifier 给终端奖励 \(R(o_i)\);归一化 advantage \(\hat A_i=\frac{R(o_i)-\mu_G}{\sigma_G+\epsilon_{\mathrm{std}}}\);clipped surrogate(裁剪代理目标)为
  \(\displaystyle \mathcal{J}_{\mathrm{GRPO}}(\theta)=\hat{\mathbb{E}}_{i,t}\Big[\min\big(r_{i,t}(\theta)\hat A_i,\ \mathrm{clip}(r_{i,t}(\theta),1-\epsilon,1+\epsilon)\hat A_i\big)\Big],\quad r_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}\mid o_{i,<t},q)}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid o_{i,<t},q)}.\)
  全零奖励时 \(\hat A_i\to0\)、梯度消失,这就是学习悬崖。【原文】§3.1、Eq.1
- **方法流水线(§3.2 + Fig.3,输入→输出)**:
  1. **输入**:query \(q\);预定义的三层提示 \(H=\{H_{\mathrm{knowledge}},H_{\mathrm{planning}},H_{\mathrm{solution}}\}\)(由 DeepSeek-R1 基于 ground-truth 解题步骤**离线一次性**生成,§4.1)。三层的语义分别是:\(H_{\mathrm{knowledge}}\) 指出关键概念 / 公式,\(H_{\mathrm{planning}}\) 给高层策略框架,\(H_{\mathrm{solution}}\) 给具体计算步。
  2. **Phase 1 — 诊断 true-hard(guidance exemption period,引导豁免期)**:训练前 **15% 步数**纯做 on-policy 探索;同时监控"零奖励 query 被解决的速率"。
     - 等这个速率停滞后,那些仍持续失败的题被判为 **true-hard**(真难题),区别于 **pseudo-hard**(假难题,即只是格式 / 早期技能不熟、多训就能解);只有 true-hard 才成为引导候选。
     - 这样保证"提示只留给真正的悬崖"。**输出**:true-hard 题集。
  3. **标准 GRPO 步**:对 \(q\) 采一组 \(N\) 条 \(G\);若其中 \(\ge1\) 条对 → 走正常 GRPO,不干预(Eq.5,此时 \(\mathcal{J}_{\mathrm{Scaf}}\equiv\mathcal{J}_{\mathrm{GRPO}}\))。
  4. **Phase 2 — 全错时触发分层引导**:做确定性搜索,从最抽象的层(Knowledge)开始、在层内**增量地给提示**(\(H_{\mathrm{knowledge}}\to H_{\mathrm{planning}}\to H_{\mathrm{solution}}\)),一旦 \(\pi_\theta\) 在 \(q\oplus h\) 上采出正确解就停 → 得到**最小有效提示 \(h^*\)**(算法见附录 D.1)。**输出**:\(h^*\) 和一条带提示的成功轨迹 \(o^*_h\sim\pi_\theta(\cdot\mid q\oplus h^*)\)。
  5. **on-policy batch 增强**:用 \(o^*_h\) **替换**掉一条随机的失败轨迹 \(o_j\) → \(G_{\mathrm{final}}=(G\setminus\{o_j\})\cup\{o^*_h\}\);再在 \(G_{\mathrm{final}}\) 上重算 advantage
     \(\displaystyle \hat A'_i=\frac{R(o'_i)-\mu_{G_{\mathrm{final}}}}{\sigma_{G_{\mathrm{final}}}+\epsilon_{\mathrm{std}}},\qquad o'_i\in G_{\mathrm{final}},\)
     从而恢复 \(\mu_{G_{\mathrm{final}}}>0\)、得到非零的 \(\hat A'_i\)。
     - 损失形式仍是 GRPO 的 clipped surrogate \(\mathcal{J}_{\mathrm{Scaf\text{-}GRPO}}(\theta)=\hat{\mathbb{E}}_{i,t}[\min(r'_{i,t}\hat A'_i,\mathrm{clip}(r'_{i,t},1-\epsilon,1+\epsilon)\hat A'_i)]\),但**比率分段**(Eq.4):
       \(\displaystyle r'_{i,t}(\theta)=\begin{cases}\dfrac{\pi_\theta(o'_{i,t}\mid o'_{i,<t},q)}{\pi_{\theta_{\mathrm{old}}}(o'_{i,t}\mid o'_{i,<t},q)},& o'_i\in G_{\mathrm{final}}\ \text{且}\ o'_i\ne o^*_h\\[10pt]\dfrac{\pi_\theta(o'_{i,t}\mid o'_{i,<t},q\oplus h^*)}{\pi_{\theta_{\mathrm{old}}}(o'_{i,t}\mid o'_{i,<t},q\oplus h^*)},& o'_i=o^*_h.\end{cases}\)
     - 即:普通轨迹的比率对 \(q\) 算,而 \(o^*_h\) 的比率对 \(q\oplus h^*\) 算——这是保住 on-policy 一致性的关键。
  6. KL penalty 设为 0(为了最大化探索);**输出**:一个能持续从 true-hard 题里学习的策略。
- **逐组件必要性(消融见 Table 2,Qwen2.5-Math-7B;基线 No Guidance = vanilla GRPO 50.9 vs 45.2;以下降幅是相对 full 配置的相对值)**:
  - **guidance exemption period(15%)**:作用是防"一开始就给提示 → 产生 hint 依赖"。消融充分——附录 G.1:从头就 scaffold 会掉 **9.2%**;且这个比例在 10%–40% 区间内表现稳定(plateau 在 49.5–50.9%),所以选了 15%。
  - **三层 KPS 分层**:每层都有独特功能。消融充分——逐层删除:w/o Knowledge(即 P→S)得 49.2、w/o Planning(K→S)得 48.6、**w/o Solution(K→P)得 48.0、降幅最大(−5.7%)**——证明三层互补、非冗余(抽象层促进高阶推理,具体层是必要的兜底)。
  - **progressive / hierarchical(抽象→具体)vs Solution-Only**:Solution-Only(直接给最具体的提示)得 48.4(−4.9%)——证明"先逼模型啃高阶推理"能促成更可迁移的技能。
  - **incremental chunking(层内增量给)vs Full Hint(一次把整层给)**:Full Hint 得 47.7(**−6.3%,是 Table 2 里降幅最大的一项**)——证明最小增量干预对"保自主性、防过度依赖"最为关键。
  - **hint 质量**:§4.3 + 附录 H,用 LLM-as-Judge 从 accuracy / minimality / clarity / coherence 四维评分,结论是 DeepSeek-R1 生成的 hint 质量高于 Qwen2.5-72B,带来学生 +4% 的相对增益。
  - **数据过滤(Too Easy 丢 / Too Hard 留 / Potentially Solvable 做 50% 子采样)**:Table 3 显示,过滤后的数据对 vanilla GRPO 只 +0.5%、但对 Scaf-GRPO +6.0%(7B);附录 G.2 / Table 12 进一步:此过滤比用全量数据 +10.2%。说明"难课程"只在配合 Scaf-GRPO 这种能消化难题的框架时才有效。【原文】§4.3、Table 2-3
- **关键机制 / 公式(直觉)**:
  - **为何仍是 on-policy**(§3.2 "on-policy integrity"):
    - off-policy 前缀法的比率是 \(\pi_\theta/\pi_\phi\)(从别的策略 \(\phi\) 导入轨迹),需要高方差的 importance 校正、且分布失配;
    - 而 Scaf-GRPO 的 \(o^*_h\) 直接从当前 \(\pi_\theta\) 采,比率只是"在改过的输入 \(q\oplus h^*\) 上的标准 on-policy 比率"——把当前策略和旧策略都 condition 在**同一个带提示的 prompt** 上,信号更稳;
    - 附录 F.5 实证:用 \(\pi_\theta(\cdot|q)/\pi_{\theta_{\mathrm{old}}}(\cdot|q\oplus h^*)\) 这种"分子分母 condition 不一致"的失配比率会训崩。
  - **"最小有效引导促内化"**:用尽可能抽象的提示把题解出来 → 逼模型内化推理技能、而非死记解法(§1/§4 Internalizing skills);Fig.5 展示了一道题的"毕业"轨迹——从"靠 Solution 提示才对"→"靠 Planning 对"→"靠 Knowledge 对"→最终"无提示自解"。【原文】§3.2、§4.4、Fig.5
- **训推数据如何流动**:训练数据派生自 DeepScaleR(40k),按每个模型 8 次采样做动态过滤(全对的 Too Easy 丢 / 全错的 Too Hard 留 / 对 1–7 条的做 50% 子采样,附录 B.1)→ 训练中走标准 rollout(verl + vLLM)→ 若整组全错则触发确定性的 hint 搜索(在 \(q\oplus h\) 上重采)→ 拿 \(o^*_h\) 替换失败轨迹 → 在 \(G_{\mathrm{final}}\) 上重算 advantage → 用分段比率做 GRPO 更新。hint 是一次性离线由 DeepSeek-R1 生成的(基于 ground-truth 步骤,附录 B.2/E)。【原文】§4.1、附录 B
- **实验与证据**:
  - 设置:评测 7 个基准(pass@1 greedy):AIME24 / AIME25 / AMC / Minerva / MATH-500 / OlympiadBench / GaoKao2023en;OOD 用 **GPQA-Diamond**。模型 5 个:Qwen2.5-Math-7B、Qwen2.5-Math-1.5B、Qwen2.5-7B、Llama-3.2-3B-Instruct(非 Qwen 系)、DeepSeek-R1-Distill-Qwen-1.5B(长 CoT)。框架用 **verl**;训 **10 epochs**;max response 2048(长 CoT 8192);**KL penalty = 0**;报 best checkpoint。baseline 设置:vanilla GRPO 用本文数据 + 超参,LUFFY 用本文数据 + 其原参,Simple-RL / Oat-Zero 用公开权重。
  - 关键数字:
    - 主结果(Qwen2.5-Math-7B,Table 1):整体相对 vanilla GRPO **+12.6%**(50.9 vs 45.2)、相对 LUFFY **+9.2%**;AIME24 pass@1 从 30.0→**43.3**(相对 +44.3%);相对 Simple-RL +19.5%、相对 Oat-Zero +9.5%;
    - **5 个模型全部有正增益**(Llama-3.2-3B 26.1→28.8、+10.3%;长 CoT 1.5B 50.6→53.6、+5.9%;Qwen2.5-7B 41.5→44.9);
    - OOD(Table 6,GPQA-Diamond):7B 32.3→37.3(相对 vanilla GRPO **+15.5%**、与 LUFFY 持平),Qwen2.5-7B-Base 33.3→35.8(**+7.5%**,超过 LUFFY 的 34.4);
    - skill-graduation(Table 4,前 300 步里"从 hint 依赖 → 自解"的事件数)显著多于 vanilla:1.5B 1123→2670(+137.8%)、7B 434→483(+11.3%)、Llama 577→986(+70.9%)。
  - baseline 公平:vanilla GRPO 同数据同超参(可比);LUFFY 同数据、用其原参(基本可比);Simple-RL / Oat-Zero 用公开权重(口径略有不同,作者标注为 "contextualize against optimized public benchmarks")。
  - "看着强但没回答核心问题":
    - 主结果多报的是 **best checkpoint** 而非固定步数(跨方法的早停一致性需要信任作者的实现);
    - 消融只做到**层级粒度**,**没做"训练题 vs 持出题"的记忆探针**来直接排除"学生只是在 hint 下记住了该题"〔推断:Fig.5 只是定性单例,Table 4 的 graduation 是间接证据,二者都不是记忆探针〕。【原文】§4.1-4.4、Table 1/3/4/6
- **假设与失效边界**:
  - 显式假设【原文】:
    - 有可验证答案(verifier,这是 RLVR 的前提);
    - 有一个**更强的教师(DeepSeek-R1)**来生成分层 hint(§4.1);
    - 题目能被三层提示"够到"(否则搜到 Solution 层仍失败 → fallback,论文未细述全失败的比例)。
  - 隐式假设【推断】:三层 hint 的语义粒度划分对不同领域都普适(论文只在数学上验证过);"零奖励解决速率停滞"能可靠区分 true / pseudo-hard(阈值 / 窗口都是经验设定)。
  - 何时失效:【原文·Limitations】依赖一套高质量的分层 hint 体系(构造起来非 trivial);主要面向有 verifier、结构化推理路径的任务(数学),对开放 / 主观域(如创作)不直接适用。【推断】非数学域(代码 / 通用推理)未验证(GPQA 只是当 OOD 评测,不是训练域)。
- **祛魅总结**【推断】:
  - 真贡献:用三件套——"in-prompt 分层提示 + 当前策略自采替换 + 比率对带提示 prompt 算"——**在不改 GRPO 损失的前提下、严格保住 on-policy 一致性地攻克了学习悬崖**;相对前缀法(LUFFY 等)是一个干净且概念清晰的改进;消融(逐层、progressive、incremental、exemption、过滤)做得相当扎实,跨 5 个模型一致增益,增强了普适性论断。
  - 包装 / 高估:"真零外部知识 / 自主"并不成立——hint 由更强的 DeepSeek-R1 生成,本质仍是教师知识下放;"internalization"靠的是 graduation 计数和单例图,缺记忆探针这种硬证据;再加上 best-checkpoint 报告 + 仅数学域,使"unlocking ability beyond reach"这个说法有夸张空间。
  - 低估之处:"数据难度需要配上能消化它的框架"(Table 3:难数据只对 Scaf-GRPO 有大增益)其实是对 RLVR 课程设计很有价值的观察;exemption 在 10–40% 区间都稳,也说明方法对该超参不敏感。
  - 〔待核·论文内部小笔误〕Table 1 标题把那个非 Qwen 模型写成了 "Llama-3.2-8B-Instruct",但 §4.1 的 Models、表内行、§4.2 正文都是 **Llama-3.2-3B-Instruct**——应以 3B 为准。

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号:可验证 **结果奖励 \(R\)(对/错)**经 GRPO 组内归一化 advantage \(\hat A'_i\)(轨迹级);提示本身不直接进损失,只改采样输入 \(q\to q\oplus h^*\)。
  - 改什么:**参数**(GRPO 策略更新);训练期还动态改 **prompt**(注入分层 hint)和 **rollout buffer**(\(o^*_h\) 替换失败轨迹)。
  - 何时改:**在线 per-step / 触发式**(仅在某 query 整组全错=学习悬崖时才触发引导;Qwen2.5-Math-7B 上仅 17.4% 样本触发)。
  - 免梯度?:否(标准 GRPO 梯度;引导只是数据增强,损失形式不变)。
  - 记忆-技能生命周期:**无持久记忆/技能库**;三层 hint 是**离线预生成的静态资源**(训练前由 DeepSeek-R1 造好,按需注入);成功轨迹 \(o^*_h\) 用完即弃。技能"固化"靠参数更新。
  - 防遗忘机制:**无显式防遗忘**;KL penalty=0(反而最大化探索)。仅"exemption period 防 hint 依赖"算一种弱意义的"防过拟合到提示"。
- ⑦ 开源代码 + 框架/harness:https://github.com/JIA-Lab-research/Scaf-GRPO (已 clone 约 7.2MB,Tier A,origin 已核)。框架 **veRL 0.4.1.dev**(仓库含 `verl/` 与 `hint_mix_grpo/{main_ppo.py, trainer/, dataset/, config/}`;rollout 用 vLLM;conda env `scaf-grpo`/py3.10)。入口 `train.sh` → `bash sh/hint_mix_grpo/bs256_6k_mix.sh`;baseline/评测在 `sh/{baseline,generation_eval}`。数据集 HuggingFace `hkuzxc/scaf-grpo-dataset`。
- 💰 资源/成本与可扩展性:Qwen2.5-Math-7B 训练 ~73GB peak（vanilla ~72GB,几乎无额外显存,Table 5）；**引导仅 17.4% 样本触发**故吞吐多用于标准生成；到最佳 checkpoint **~12h**(vanilla GRPO 需 ~13h 才到更低峰值 45.2%)——即更省总时间还更高分（50.9%）。max response 2048（长 CoT 8192）；hint 一次性离线生成（DeepSeek-R1）。可扩展性:方法与 GRPO 损失解耦、只改数据,原则上可挂在任意 GRPO 实现上。【原文】§4.4、Table 5
- 🎯 对"探索-巩固"对标:**强支撑(尤其"探索 / 选路"维度)+ 可借组件 / 部分竞品**。判定:
  - Scaf-GRPO 是本项目"探索-巩固"里**探索 / 选路**思路最贴近的实现之一——它的"路标 vs 铁轨"哲学、"提示只指方向、让 student 自己采出能走通的解",几乎就是本项目"在岔路口偏向自己能走通的开头 / 方向 + student 全程 on-policy 自选"的 RLVR 版落地。
  - 它同时也是 path-recovery 的一种范式:全错(走偏)时,用最小提示帮 student 自采一条对的、来"接着做对"。依据:§3.2 on-policy 替换 + Fig.5 毕业轨迹。
  - 可直接借的积木(4 件):
    - ① **触发式干预**(只在整组全错 / advantage 坍缩时介入、平时纯自主)——天然契合"只在岔路口 / 走偏时让 teacher 当稀疏脚手架";
    - ② **比率对带提示 prompt 计算**(Eq.4 分段比率)这一保 on-policy 的技术细节,对"student 自选恢复分支后仍要 on-policy 更新"直接可用;
    - ③ **exemption period** 这种防止过早依赖脚手架的课程设计;
    - ④ **三层抽象→具体 + 增量**,可作"最小有效脚手架"的粒度模板。
  - 部分竞品:它已经占住了"教师分层提示 + 保 on-policy + 保探索"这一块,与本项目 idea 重叠度高,所以需要在三点上做出区分——MTP 前瞻探针、student 自选恢复分支(而非确定性搜索 + 预设三层 hint)、自蒸馏固化。
  - 缺口:它的 hint 是**预生成的静态三层**(不是 student 自己发现的恢复分支),且**无 MTP / 前瞻**、**无防遗忘 / 技能库**。
- 🔭 开放问题/未来方向:
  - 【原文·Future Work】自动化 hint 生成提升可扩展性;**adaptive scaffolding**——引导随模型 proficiency 动态调整、个性化学习过程。
  - 【推断】扩到代码/通用推理域;补"训练题 vs 持出题"记忆探针以坐实"内化非记忆";用固定步数而非 best-checkpoint 复核增益;把"预生成静态三层 hint"换成"student 自采/自生成的恢复分支"以更接近真自进化;研究 exemption/温度/组大小等超参的系统鲁棒性。

RETURN: scaf_grpo|读到PDF?是(本轮新抽 29页 _txt 105k字,Eq.1-5 + Table 2/3/4/5/6 消融全核)|L3(+L1)|对标=强支撑(探索/选路)+可借触发式干预/Eq.4 分段比率保 on-policy/exemption/三层增量;Δ=预生成静态 hint(非自采恢复分支),无 MTP/无防遗忘|残留待核 0(Table 1 标题 Llama-3.2-8B 为笔误,正文以 3B 为准)

why_sd_degrade | Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs? | Microsoft Research + KAIST + Seoul National University（Jeonghye Kim 实习于 MSR；Xufang Luo 通讯） | 2026-05-20·arXiv preprint, under review·v3（cs.CL） | 主题线 L1（OPD/自蒸馏，失败模式诊断）+ L6（CoT/不确定性表达）·相关性 高

**原始论文**：https://arxiv.org/abs/2603.24472 （代码 https://github.com/beanie00/self-distillation-analysis ）

## 一眼看懂

> 一句话导读：同一个模型"看着答案当老师、不看答案当学生"地自蒸馏，在化学/工具/代码上能"答得又短又好"，可一搬到数学推理就翻车——本文不提新算法，只用受控实验把根因锁定在"wait/hmm 这类犹豫词被训没了"。

- 🟦 TL;DR：自蒸馏指同一模型在富 context（这里 context=正确解）下当 teacher，给无 context 的 student 打稠密信号。它在很多域（化学/工具/代码）能让 response 变短、性能变好。但搬到**数学推理**上会反常——response 照样变短、性能却大跌（AIME24 最高掉约 40%）。
- 本文不提新算法，而是用受控实证把根因锁定在一个现象上：**epistemic verbalization（认知性不确定表达，即 wait/hmm/maybe 这类标记"我还在犹豫"的词）被压制了**。机理是：teacher 拿到富信息（正确解）后，生成的轨迹简洁又自信、几乎不带不确定词；student 一模仿,就把这些"标记备选假设、支持渐进纠错"的信号丢光了。
- 危害是否真的显现,取决于**任务覆盖度**(任务种类宽不宽)：覆盖窄时,压制掉犹豫词反而更高效;覆盖越宽、越偏离训练分布(OOD)就越有害【原文 Abstract、§1、Takeaway 1-4】。
- 最巧的一步：**用条件互信息 \(I(y^*; c \mid x) = H(y^* \mid x) - H(y^* \mid x, c)\)（式2）把"context 信息丰富度"量化成一个可比的数**。它先构造 4 种 context \(c\)——unguided（\(\varnothing\)，啥都不给）/ 全解（\(s\)）/ 去掉思考过程的全解（\(s_{\setminus\text{think}}\)）/ 用全解再生成一遍（\(y_r\)），并论证四者的互信息（MI）排序为 \(I(y^*;\varnothing\mid x)=0 < I(y^*; s_{\setminus\text{think}}\mid x) \le I(y^*; y_r\mid x) \le I(y^*; s\mid x)\)（式3）。然后观测到 response 长度、得分、10 个 epistemic token 的计数都随 \(I\) 单调变化（Table 1）。抽掉这个 MI 框架,"信息越富→越自信→犹豫词越少"就只是定性观察,没法把"风格漂移"和"性能退化"挂到同一个可控变量上【原文 §3、式2-3】。

## 为什么做

> 一句话导读：自蒸馏在别的域屡试不爽,可没人问"它什么时候、为什么会退化";本文从"数学=不断更新信念的贝叶斯推理"这个视角切入,补上这一问。

- 研究背景：自蒸馏（Snell 2022）作为 LLM 后训练范式,配 RLVR（带可验证奖励的 RL）在 agentic/科学推理域高效提分,且常伴随"response 变短、性能变好"（SDPO Hübotter 2026、OPSD Zhao 2026 等）。本文把数学推理视作 **self-Bayesian 推理**——即每一步都在 \(x\) 与已生成前缀 \(y_{<t}\) 上更新对中间假设的信念（§3,引 Kim 2026）。
- 解决的具体痛点：把同一套自蒸馏（SDPO）用到数学时出现反常——response 仍变短,但性能显著跌（AIME24 掉约 40%）。既有自蒸馏工作只展示有效性,**未研究"何时/为何会退化"**,尤其是在"模型需完全自主解题、无外部环境交互"的数学场景【原文 §1、附录 A】。
- 相关工作 & 各自不足（来龙去脉/并行路线/精确差异）：
  - **① on-policy 自蒸馏配方族（本文的诊断对象）**：
    - SDPO（Hübotter 2026,RL via Self-Distillation）：条件于自身正确轨迹/特权环境反馈做 dense 信用分配；
    - OPSD（Zhao 2026）：用 ground-truth 解作 privileged 信息,蒸 student 自己生成的轨迹；
    - Ye 2026（EAFT 同源）：熵感知。
    - **共性短板**：都只展示"有效 + response 变短",不问退化条件。本文**直接建于 SDPO codebase（lasgroup/SDPO）之上**,改 `_remove_thinking_trace` 来跑诊断。
  - **② epistemic verbalization 理论（站其肩上）**：Kim 2026 证"不确定表达对鲁棒推理在信息上是必要的"（移除会降鲁棒性）。但它**答不出**两件事:这些表达何时被鼓励/压制?为何同一压制机制在化学有益、在数学有害?——本文正是补上这两个调制因子。
  - **③ 推理压缩线（平行路线,被本文反向警示）**：GFPO（长度惩罚 RL）、OPSDC、ConPress、Accordion、CEEH 等都以"缩短 CoT"为目标。**本文补充结论**：压制不确定表达的压缩,即便答案对,也在 OOD 上有害——压缩不能只看"短 + 对"。
  - **④ 与最近邻 Kim 2026 的精确Δ**：Kim 给的是"必要性"（黑白判断）；本文给的是"**两个连续调制因子**"——信息丰富度 \(I(y^*;c\mid x)\) + 任务覆盖度 \(|\mathcal D|\)。本文系统扫描 4 个模型族,把"同一压制机制为何化学涨、数学跌"统一进"覆盖度 × epistemic 保留"这个可操作框架。
- 动机链（逐步推进）：
  1. 自蒸馏在数学上反常退化；
  2. 追问"朝正确答案训为何反而更差"；
  3. 把数学看作 self-Bayesian 推理——不确定表达保留备选假设、支持纠错,过早自信会锁死错误假设（Fig.2a）；
  4. teacher 拿富 context 会生成低不确定、高自信的轨迹；
  5. student 模仿就丢掉了 epistemic 信号；
  6. 而标准训练目标不惩罚这种"风格漂移",于是 OOD 退化是"隐性"的【原文 §1-2】。

## 怎么做 + 靠不靠谱

> 一句话导读：这是一篇诊断论文,主线是"固定一个因子、只动另一个"地做受控实验——其中 §4 那组"两份数据都对、唯一差异是犹豫词密度"的 SFT 对照,是最干净的因果证据。

- 方法流水线（诊断而非算法,输入→证据→输出）：系统改变两个因子,沿四节逐步推进——
  - **(§3) 变 \(c\) 的信息丰富度**：输入 100 题（base 8-rollout 准确率落在 [0.125,0.5]）× 4 种条件 \(c\) → 输出 length / score / epistemic 计数随 \(I\) 的单调关系（Table 1）。
  - **(§4) off-policy SFT 因果对照**：输入两组各 800 条轨迹（**都正确**,但 epistemic 密度不同）→ 微调 → 输出 OOD math 准确率差（Table 2）。
  - **(§5) on-policy GRPO vs SDPO 多模型 + EMA 消融**：输入 DAPO-Math-17k,跑约 100+ step → 输出 score / length / OOD 曲线 + epistemic token 频移（Fig.3-6）。
  - **(§6) 变任务覆盖度 \(|\mathcal D|\)**：输入 \(|\mathcal D|\in\{1,8,64,128,512\}\) → 输出"窄覆盖时压制高效、宽覆盖时压制有害"（Fig.7-8）。
- 逐组件（这里"组件"=各支证据,逐一看其必要性/能否被驳,且写清数据流）：
  - **§3 信息丰富度受控对比（Table 1,强支撑 Takeaway 1）**：
    - base = DeepSeek-R1-Distill-Qwen-7B,从 DAPO-Math-17k 选 100 题。
    - 4 种 \(c\)：(1) unguided（\(c=\varnothing\)）；(2) full solution（\(c=s\),含 `<think>`）；(3) 去 think 的全解（\(c=s_{\setminus\text{think}}\),残留 640/13054 个 token）；(4) 用全解再生成（\(c=y_r\),\(y_r\sim\pi_\theta(\cdot\mid x,s)\)）。
    - epistemic 计数定义 \(E(y)=\sum_{t\in\mathcal T}\text{count}(t,y)\),\(\mathcal T\)=10 个词（见下）。
    - 结果（Table 1,format = score / length / \(E\)）：(1) 0.30 / 13,054 / 182.5；(2) 0.98 / 1,873 / 8.8；(3) 0.78 / 12,036 / 159.8；(4) 0.95 / 2,808 / 24.1。
    - 即 length 与 \(E\) 都随 \(I\) 单调降,长度排序 \(\mathbb E[L(y)]_{(1)}>\mathbb E[L(y)]_{(3)}>\mathbb E[L(y)]_{(4)}>\mathbb E[L(y)]_{(2)}\)（式4）,\(E[E(y)]\) 同序【原文 Table 1、式4】。
  - **§4 off-policy SFT 因果对照（Table 2,最干净的因果对照）**：在 800 条正确轨迹上微调——
    - \(\mathcal D_{ug}\)（unguided,高 \(E\)、约 12k token）几乎不掉：AIME24 54.79→51.04、AIME25 37.92→40.00、AMC23 89.06→87.66、MATH500 92.19→90.93；
    - \(\mathcal D_{sg}\)（solution-guided,低 \(E\)、约 2k token）**全面大跌**：AIME24 54.79→**20.21**、AIME25 37.92→12.71、AMC23 89.06→57.03、MATH500 92.19→65.52。
    - **两个数据集都是正确答案,唯一差异是 epistemic 密度,却造成大幅性能差**——证明压制是实质有害,而非纯风格问题【原文 Table 2、Takeaway 2】。
  - **§5 on-policy GRPO vs SDPO（Fig.3-5）**：DAPO-Math-17k 上约 100 step,teacher **固定为初始策略**。
    - DeepSeek-7B：GRPO 略升（AIME24 54.7→56.0、AMC23 89.3→91.1,伴随 \(\mathbb E[L]\) 微升）；
    - SDPO（\(c=s\)）初期 score/length 双崩,AIME24 跌约 40%、AMC23 约 15%；
    - \(c=s_{\setminus\text{think}}\) 能缓解但仍低于 base。
    - Fig.3d/4d 显示 SDPO 比 GRPO 更激进地压制 epistemic（尤以 wait 为甚——SDPO 在 AIME24 上 wait 计数变化 −60.8,GRPO 则 +28.5）。Qwen3-8B（think on/off）、Olmo3-7B 同向【原文 §5.1-5.3、Fig.3-5】。
  - **§5.4 EMA 消融（Fig.6,支撑反馈回路解释）**：teacher_update_rate=0.0（固定 teacher）优于 0.05（慢更新）——慢更新会形成"越自信→teacher 越自信"的正反馈回路,退化更大【原文 §5.4、README config】。
  - **§6 任务覆盖度 sweep（Fig.7-8,支撑 Takeaway 4）**：变 \(|\mathcal D|\in\{1,8,64,128,512\}\)（Qwen3-8B think off）。
    - \(|\mathcal D|\le128\) 时 SDPO 快速高分、长度减 8 倍（窄覆盖高效）；
    - \(|\mathcal D|=512\) 时 SDPO 反伤 score、OOD 全程低于 base,而 GRPO 随 \(|\mathcal D|\) 增大,靠**增加** epistemic 持续提升【原文 §6.2、Fig.7-8】。
  - **Fig.12 频移对照（排除混淆,关键）**：全词表平均每词 shift 的 \(|\Delta|<1\),而 10 个 epistemic token 的 shift 大 30-40 倍（SDPO 达 −11.9/−12.2）——证明训练是**特异性**地作用于 epistemic 表达,而非均匀的词表漂移,从而排除"通用风格漂移"这一替代解释【原文 附录 B.4、Fig.12】。
- 关键机制/公式（真实符号 + 直觉）：
  - **被诊断的自蒸馏目标（前向 KL,注意不是 reverse-KL）**：student 向"富 context teacher"对齐,token 级最小化 \(\mathrm{KL}\!\big(\pi_\theta(\cdot\mid x,\hat y_{<t})\,\big\|\,\mathrm{sg}[\pi_\theta(\cdot\mid x,c,\hat y_{<t})]\big)\),其中 \(\mathrm{sg}\)=stop-gradient（截断梯度,把括号内当常数）,\(c=s\) 为正确解。直觉：这等价于"学一种预设了推理时根本拿不到的信息 \(c\) 的自信风格"【原文 §1-2,符号据正文论证;具体目标式分散于 SDPO 引文,标记为本文诊断对象】。
  - **信息丰富度量化**：\(\displaystyle I(y^*; c \mid x) = H(y^* \mid x) - H(y^* \mid x, c),\) 其中 \(y^*\) 为理想正确响应的随机变量。直觉链:\(c\) 越富 → 条件熵 \(H(y^*\mid x,c)\) 越小 → 互信息 \(I\) 越大 → teacher 越能"照抄"提示 → 轨迹越简洁自信 → epistemic 越少（式2）。MI 排序由两点导出:"\(s_{\setminus\text{think}}\) 是 \(s\) 的信息子集" + "数据处理不等式(\(y_r\) 由 \(s\) 派生)"（式3）。
  - **epistemic token 代理**：\(\mathcal T=\{\textit{wait, hmm, perhaps, maybe, actually, alternatively, seems, might, likely, check}\}\),\(E(y)=\sum_{t\in\mathcal T}\text{count}(t,y)\)（式见 §3;附录 B.5 用 GPT-5.4 as judge 验证这 10 词与多 token 不确定短语共现）。
  - **覆盖度调制直觉**：\(|\mathcal D|\) 大时模型需容纳更多推理模式——GRPO 靠增加 epistemic 表达来适应,SDPO 却逼向简洁自信,于是在宽覆盖下受限【原文 §6.2】。
- 实验与证据：
  - 数据集/设置：**作者自跑的核心实验全在数学**。
    - 训练:DAPO-Math-17k（14,000 道 distinct 题;§6.1 Table 3 报"100 步内 25,600 次采样中 78% 不重复"）；
    - OOD 评测:AIME24/25、AMC23、MATH-500（题型不与训练重叠）。
    - 模型:Qwen3-1.7B/8B（think on/off）、DeepSeek-R1-Distill-Qwen-7B、Olmo3-7B-Instruct。
    - 关键超参（README）:train_batch_size=256、ppo_mini_batch_size=128/64、max_prompt_length=2048、max_response_length=20480、max_reprompt_len=22528、teacher_update_rate=0（主）/0.05（消融）、`remove_thinking_from_demonstration`=False(\(c=s\))/True(\(c=s_{\setminus\text{think}}\))。
  - **跨域比较有一半是引用而非重跑（审计事实）**：Fig.1a 的 Chemistry 曲线直接取自 SDPO（Hübotter 2026）的 W&B 日志；§6 用 LiveCodeBench v6 做覆盖度对比。三者口径差异大（Table 3）:Chemistry 2,400 题但仅 6 类题型、90-10 split；LiveCodeBench v6 仅 131 题且 train/eval 全重叠；而 DAPO-Math 14,000 题不重叠。**因此跨域结论是间接论证**【原文 §6.1、Table 3、Fig.1 脚注】。
  - baseline 公平吗：GRPO vs SDPO 在同数据/同模型/同 step 下比较,公平；off-policy SFT 对照（Table 2）控制极干净(两组都正确、唯一变量是 epistemic 密度)。问题只在跨域那部分有一半证据是引用 SDPO 日志、口径不完全一致。
  - 有无"看着强但没回答核心问题"：核心问题（why degrade）答得清楚；但"epistemic verbalization 因果驱动正确率"这一点,主要靠相关性 + 受控对比,**没做"强行注入/移除 epistemic token 看正确率"的反事实干预**——属强相关而非严格因果【推断,依据全文无 counterfactual injection 实验】。
- 假设与失效边界：
  - 显式假设【原文】：数学推理可视为 self-Bayesian 推理（§2,引 Kim 2026）；10 个固定词可作 epistemic 的实用代理（§3,附录 B.5 补充验证）。
  - 隐式假设【推断】：epistemic token 频率与"真实不确定推理"正相关——这 10 词的近似可能漏掉多 token 短语;作者在 prompt 里已显式排除数学条件句"if...then",但单 token 计数仍然偏粗。
  - 何时失效【推断/原文】：在窄覆盖、train-eval 高重叠的任务上,压制 epistemic 不仅无害还高效（Takeaway 4）——所以结论是"取决于覆盖度",而非"自蒸馏一律有害"；"何种程度的不确定才算恰当"没有可操作判据。
- 祛魅总结【推断】：真贡献是把一个反直觉现象（自蒸馏在数学上变短变差）做了扎实的现象学拆解——Table 2 的 off-policy 对照 + Fig.12 的频移排除混淆是两处硬证据,结论"后训练目标应显式保留不确定表达、不能只看答案正确"有分量。被适度高估的是因果强度:通篇无反事实干预,"epistemic→正确率"是强相关;且跨域(化学/代码)对比一半是引用 SDPO 日志、非重跑。定位准确:这是诊断/机理论文,不提修复算法(作者自承)。

## 结构化抽取
- 🎯 机制速览6轴（注：本文是"诊断",6 轴描述其分析的对象=自蒸馏训练,非提出新机制）：
  - **学什么信号**：被分析的自蒸馏目标 = student 向"富 context teacher"对齐的 **前向 KL** \(\mathrm{KL}(\pi_\theta(\cdot\mid x,\hat y_{<t})\,\|\,\mathrm{sg}[\pi_\theta(\cdot\mid x,c,\hat y_{<t})])\)（注：是 forward-KL,非 reverse-KL）；本文发现这等价于"学一种预设了推理时不可得信息的自信风格"【原文 §2】。
  - **改什么**：student 参数；本文关注训练如何**改变 reasoning 行为/风格**（epistemic token 密度、response 长度）,而非仅看答案正确率【原文 §1、§5】。
  - **何时改**：off-policy SFT（§4）与 on-policy（§5）两种；on-policy 中 teacher 固定为初始策略时最优（§5.4）。
  - **免梯度?**：否（分析对象是 SFT/GRPO/SDPO 这些基于梯度的后训练；本文自身是实证分析,无训练新增）。
  - **记忆-技能生命周期**：N/A（无记忆/技能库；论点是"巩固不当会侵蚀不确定推理这一能力"——与本项目"巩固不应遗忘"直接呼应）【推断】。
  - **防遗忘机制**：本文恰是**反例诊断**——指出自蒸馏会"隐性遗忘 epistemic 表达能力",而标准目标不惩罚,于是 OOD 退化;隐含建议是训练目标须显式保留不确定表达（未给具体机制）【原文 §1、§7】。
- ⑦ 开源代码+框架/harness：仓库 https://github.com/beanie00/self-distillation-analysis （v1 记 53M 已 clone,仓库实测含 `verl/`、`analyzing_reasoning_behavior/`、`experiments/math/`、`baseline_multiturn/`、`data/math/`、`figures/`、多份 requirements）。**框架：veRL/HybridFlow**——References 引 Sheng 2024 HybridFlow,仓库含 `verl/` 目录,改了 `verl/trainer/ppo/ray_trainer.py` 的 `_remove_thinking_trace` 来处理 DeepSeek-Distill 的 `<think>` 标签特例；built upon lasgroup/SDPO；用 transformers + vLLM/SGLang。硬件 4×B200（据 README/INSTALL.md;Dockerfile.gh200 表明也支持 GH200）。释出 Qwen3-8B(think on/off)/DeepSeek-Distill-7B checkpoint + W&B 日志【v1 analysis 已核仓库结构；本次据 PDF References 确认 HybridFlow=veRL + README 确认超参】。
- 💰 资源/成本与可扩展性：原文正文未给卡时/wall-clock；README 标 4×B200、max_response_length=20480、batch=256。诊断实验规模适中（每设置约 100-150 step、100 题/800 轨迹级）【原文未在正文说明完整成本,据仓库 README】。
- 🎯 对"探索-巩固"对标：**竞品/警示（强相关,反向支撑）**——一句判定:这是对"privileged-context 自蒸馏"做巩固时的**失效边界警告**,直接约束本项目"巩固/回轨"分支的设计。依据:本项目 idea 要 teacher 当稀疏脚手架教 student"走偏后自选恢复"——而本文证明:若 teacher（富 context）生成的恢复轨迹过于自信简洁、抹掉了 epistemic 表达（wait/hmm 这些正是"察觉走偏→回轨"的语言信号）,student 模仿后在宽覆盖/OOD 上反而丧失自主纠错能力（AIME 掉约 40%）。**可借组件**:
  1. 用 \(I(y^*;c\mid x)\) 度量脚手架信息量,避免脚手架"过强"压垮 student 自主性——这对 idea 里"稀疏脚手架"的"稀疏"二字给了量化抓手（脚手架 \(c\) 的 \(I\) 越高,越要警惕 epistemic 压制）；
  2. Fig.12 式的频移诊断（全词表 \(|\Delta|<1\) vs epistemic 30-40 倍）可用于监控巩固阶段是否在"均匀漂移之外"特异性地抹掉关键 token。
  - **缺口/张力**：本文用的是 forward-KL(student‖teacher) 自蒸馏,而本项目/vla_opd 倾向 reverse-KL；reverse-KL 是否也压制 epistemic 本文未测——这是该项目需自查的开放点（reverse-KL mode-seeking 锁主模,可能同样丢长尾的不确定表达）。与 survey-grpo-step-segmentation 记忆里"切高熵/低置信关键步"高度互补:epistemic token 往往出现在高熵处,本文佐证"不该无脑压低这些步的熵"。
- 🔭 开放问题/未来方向：
  - 【原文】后训练目标须兼顾答案正确性 + 保留/诱导 uncertainty-aware 推理行为；呼吁开发显式保留不确定表达的更鲁棒训练策略（§7）。
  - 【原文/隐含】"何种程度的不确定才算恰当"——需可操作判据（当前只有定性"过度压制有害"）。
  - 【推断】补反事实因果实验（注入/移除 epistemic token 测正确率）；测 reverse-KL 自蒸馏是否同样压制 epistemic；把覆盖度 \(|\mathcal D|\) 与 epistemic 保留写进一个自适应目标（按覆盖度动态调脚手架强度）——后者正是本项目可落地的设计方向。

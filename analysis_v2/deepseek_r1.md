deepseek_r1 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | DeepSeek-AI | arXiv 2501.12948(v2 2026-01-04，Nature 版)·2025-01 首发 | 主题线 L3(RLVR/GRPO)+ L1(离线序列级蒸馏对照)·相关性 高

**原始论文**:https://arxiv.org/abs/2501.12948

## 一眼看懂
> 一句话导读：这篇有两个主角——R1-Zero 证明"光靠 RL、不用任何 SFT"就能让 base 模型自己长出推理能力；R1 则在其上加一条四阶段管线修好可读性和通用性，并把成果离线蒸馏进小模型。

- 🟦 TL;DR：本文分两层。
  - **R1-Zero**：证明仅靠纯 GRPO + 规则奖励、不经任何 SFT，就能在 base 模型上自演化出推理能力——响应自然变长，涌现出反思与"aha moment"。AIME pass@1 从 15.6% 升到 77.9%。
  - **R1**：在 R1-Zero 之上用一条四阶段管线修好可读性与通用性。四阶段依次是——cold-start SFT；推理向 RL（加语言一致性奖励）；拒绝采样扩 SFT（约 800k 样本）；全域 RL（加偏好奖励）。
  - 蒸馏：把上述 800k 数据通过**离线 SFT** 灌进小模型，得出结论"大模型 RL 再蒸馏到小模型，优于小模型直接 RL"【原文 Abstract, §2, §3, Appendix F.1】。
- 最巧的一步：**规则奖励 + 只约束格式不约束内容**（这是 R1-Zero 的设计）。如果抽掉它、改用神经/过程奖励模型，或者规定推理步骤，就观察不到"在无人类示例约束下自演化出非人类式推理路径"这一核心现象。
  - 作者明确说了两个"刻意"：刻意不用神经奖励模型（避免 reward hacking），刻意跳过 SFT（因为假设人类定义的推理模式会限制探索）【原文 §2.2, §2.3】。
  - 为什么有效：奖励只看答案对错 + 格式，于是模型在 RL 压力下自由地延长并重组推理（thinking time 单调增长，Fig.1b），自发涌现出自检、反思、换路。反过来，一旦约束内容或引入可被 hack 的神经奖励，这种自由探索就会被压制。

## 为什么做
> 一句话导读：当时推理能力主要靠"人标 CoT + SFT"来教，但这会被人类示例的上限卡住、也难探索更优路径；本文想用最小的人工标注、靠纯 RL 让模型自己演化出推理。

- 研究背景：当时有几个事实——LLM 配合 CoT prompting 在推理上有成效，但严重依赖人工标注的 CoT 轨迹做 SFT；test-time scaling(o1) 显示拉长推理可显著提升能力；GRPO（源自 DeepSeekMath）提供了一种免 critic 的高效 RL【原文 §1, §2.1】。
- 解决的具体痛点（几条叠加）：
  - 依赖人工推理轨迹会带来三个问题——扩展性差、引入认知偏差、性能被人类示例的上限封顶，从而难以探索"非人类式"的更优推理路径。
  - 传统的 SFT-before-RL 流程本身可能限制探索。
  - 纯 RL 的 R1-Zero 虽强，但有可读性差、中英混杂(language mixing) 的毛病；而窄域 RL 又会导致通用能力弱【原文 §1, §3】。
- 相关工作 & 各自不足（来龙去脉 + 精确差异）：
  - **CoT prompting(Wei 2022)**：能激发推理，但**受人类示例上限约束**——只能复现人给的推理模式。
  - **SFT-on-human-CoT 范式**：引入认知偏差、性能被人类示例封顶，无法探索非人类式的更优路径。R1-Zero 对此做了反转——**完全跳过 SFT**（理由是假设人类推理模式会限制探索），直接在 base 上做纯 RL。
  - **神经/过程奖励模型(RM/PRM)**：信号更细，但**易 reward hacking**、且重训成本高。R1-Zero 据此**弃用神经 RM**，只用规则二值奖励(§2.2)。
  - **o1(OpenAI)**：是 test-time scaling 的先驱，但**闭源、细节未公开**。R1 的差异在于**完整公开四阶段配方与超参，并系统给出蒸馏 vs 直接 RL 的对照**。
  - **算法底座 GRPO(DeepSeekMath 2024)**：R1 直接采用(§2.1)。但要注意，**此 Nature 版的优势 \(A_i\) 是序列级的**（见下 Eq.1/3，没有逐 token 下标），与 DeepSeekMath 里逐 token 的 \(\hat A_{i,t}\) 表述不同。
- 动机链（一步步推下来）：现状是推理靠人标 CoT + SFT；缺陷是扩展性差、被人类示例封顶、无法探索非人类路径；所以必须改用最小人工标注、靠纯 RL 自演化来激励推理——具体地，R1-Zero 跳过 SFT 直接在 base 上 RL、奖励只看最终答案，R1 再用少量 cold-start 加多阶段来补可读性/通用性【原文 §1-3】。
- 与最近邻工作的差异：相对 o1 与 SFT-on-CoT 范式，本文差在三点——证明纯 RL 单独即可激励推理（不需要 SFT 引导）、完整公开四阶段配方、系统给出蒸馏 vs 直接 RL 的对照。最有用的一点：R1-Zero 把"RL 可独立涌现推理"这一假设证实了，由此改变了整个社区的 post-training 范式。

## 怎么做 + 靠不靠谱
> 一句话导读：底座是 GRPO（一种免 critic 的 RL，用一组采样的组内均值当 baseline）；R1-Zero 用纯规则奖励直接在 base 上跑；R1 在其上叠四阶段管线补可读性与通用性，最后把成果离线蒸馏到小模型。

### A. GRPO 框架(§2.1,Eq.1-3,Nature 版=序列级优势)
GRPO 是免 critic 的策略梯度：对每题 \(q\) 从旧策略采 \(G\) 个输出 \(\{o_1,\dots,o_G\}\)，用这一组回报的均值/方差当 baseline（省掉值网络）。优化目标如下——注意重要性比与优势都是**整条输出级**的，没有逐 token 的下标 \(t\)：
\(\displaystyle \mathcal J_{\mathrm{GRPO}}(\theta)=\mathbb E_{q\sim P(Q),\,\{o_i\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}(O|q)}\frac1G\sum_{i=1}^{G}\Big[\min\Big(\tfrac{\pi_\theta(o_i|q)}{\pi_{\theta_{\mathrm{old}}}(o_i|q)}A_i,\ \mathrm{clip}\big(\tfrac{\pi_\theta(o_i|q)}{\pi_{\theta_{\mathrm{old}}}(o_i|q)},1-\varepsilon,1+\varepsilon\big)A_i\Big)-\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})\Big] \tag{1}\)
\(\displaystyle D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})=\frac{\pi_{\mathrm{ref}}(o_i|q)}{\pi_\theta(o_i|q)}-\log\frac{\pi_{\mathrm{ref}}(o_i|q)}{\pi_\theta(o_i|q)}-1 \tag{2}\)
\(\displaystyle A_i=\frac{r_i-\mathrm{mean}(\{r_1,\dots,r_G\})}{\mathrm{std}(\{r_1,\dots,r_G\})} \tag{3}\)
其中 Eq.3 就是上面说的"组内标准化"——把每条回报减去组均值再除以组标准差，得到优势 \(A_i\)。

### B. R1-Zero 奖励设计(§2.2,Eq.4=规则奖励,纯 base 上跑)
设计原则是**不用任何神经/过程 RM**（避免 reward hacking），只用规则给的二值奖励：
\(\displaystyle \mathrm{Reward_{rule}}=\mathrm{Reward_{acc}}+\mathrm{Reward_{format}} \tag{4}\)
- **准确率奖励**：把答案放进指定格式（如 box），再用规则校验对错（数学做比对，代码用编译器 + 测试）。
- **格式奖励**：强制把推理与答案分别包进 `<think>...</think>` 与 `<answer>...</answer>`（模板见 Table 1）。关键点是**只约束结构、不约束内容**——这正是涌现非人类式推理的关键。
- 两者**等权**相加。

### C. R1 四阶段管线(§3,Fig.2,逐阶段输入→输出)
输入是 DeepSeek-V3-Base(660B MoE)，依次经过四个阶段：
1. **Cold Start SFT**：用数千条对话式、经人类对齐的长 CoT 数据做 SFT，给可读性打个起点。
2. **推理向 RL（第一阶段）**：奖励 = 规则奖励（准确率 + 格式）+ **语言一致性(LC)奖励**。超参为 lr 3e-6、KL 系数 0.001、**clip ratio \(\varepsilon=10\)**、temperature 1、每题 16 输出、max length 32768、batch 512、每 400 步换一次 reference。产出 Dev-2。
   - **LC 奖励(Eq.7)**就是 CoT 中目标语言词所占的比例，**直接加到最终奖励**上：
   \(\displaystyle \mathrm{Reward_{language}}=\frac{\mathrm{Num}(\mathrm{Words_{target}})}{\mathrm{Num}(\mathrm{Words})} \tag{7}\)
3. **拒绝采样 + SFT**：从第一阶段的 RL checkpoint 做拒绝采样生成推理轨迹，并用 V3 当生成式裁判去过滤掉混语/长段/代码块。最终得到约 600k 推理样本，加上 V3 管线生成的约 200k 非推理样本，合计**约 800k** 监督数据做 SFT。产出 Dev-3。
4. **全域 RL（第二阶段）**：混合推理数据（规则奖励）与通用数据；通用查询按数据集给 **safety / helpful 偏好奖励**（Eq.5/6，如 \(\mathrm{Reward_{safety}}=\mathrm{RM_{safety}}(\mathrm{Response})\)）。注意偏好奖励**仅在最后 400 步引入**——引入更多步会导致 reward hacking(Appendix B.5)。产出最终的 R1。
- 另外两个变体：**R1-Zero** = 直接在 V3-Base 上做纯 GRPO（不含上述任何 SFT）；**R1-Distill** = 把那 800k 数据通过**离线 SFT** 灌进 Qwen/Llama 小模型（一次性纯 SFT，无 RL）。

### 逐组件必要性(部分有消融)
- **R1-Zero 纯 RL**：核心对照。AIME 从 15.6% 升到 77.9%（cons 86.7%），证明 RL 单独就能激励推理。没有它，就无法验证"不需 SFT"这一核心假设【原文 §2.3】。
- **Cold Start SFT**：对照 Dev-1 vs R1-Zero——指令遵循(IF-Eval/ArenaHard) 大涨；但因为 cold-start 数据量小，AIME 推理略有下降【原文 §4 Table 3】。
- **LC 奖励**：有消融(Appendix B.6)。加上 LC 后，语言一致性变稳定；代价是基准性能**略降**（作者原话 "slight degradation"），但换来更可读、更合人类偏好【原文 §3.2.1, Fig.7】。
- **第四阶段全域 RL**：对照显示 AlpacaEval2.0 +25%、ArenaHard +17%（通用能力补齐）；但前面提过，偏好奖励只在末 400 步引入，更多步会导致 reward hacking(Appendix B.5)【原文 §3.2.2】。
- **GRPO clip \(\varepsilon\)**：作者强调 clip ratio 很关键——取得过低会截断大量 token 的梯度、损害性能，过高则训练不稳（第一阶段取 \(\varepsilon=10\)）【原文 §3.2.1】。

### D. 关键超参与成本(复现锚点)【原文 §2.1, §3.2, Appendix B.4】
- R1-Zero：lr 3e-6、KL 0.001、temperature 1、每题 16 输出、max length 32768（8.2k 步后增至 65536）、batch 512（每步 32 题）、每 400 步换一次 ref、共 10,400 步（约 1.6 epoch）。
- SFT（cold-start/拒绝采样）：2-3 epoch、cosine lr 5e-5→5e-6、ctx 32768、batch 128。蒸馏：对应 base 微调 2-3 epoch、ctx 32768、batch 64。
- 评测覆盖 MMLU/AIME2024/MATH-500/GPQA/LiveCodeBench/Codeforces/AlpacaEval/ArenaHard。R1-Distill 结果(Table 15)：Qwen-32B AIME 72.6、Llama-70B 70.0。

### 靠不靠谱
- baseline 公平性：核心对照是**蒸馏 vs 直接 RL(Appendix F.1, Table 16)**——R1-Distill-Qwen-32B(AIME 72.6) 显著超过同 base 直接 RL 的 Qwen2.5-32B-Zero(47.0)。但作者自承，小模型直接 RL 的算力预算与 R1 不对等（直接 RL "require enormous computational power"）。
  - 几个"看着强但没完全回答"的点：① "无约束 RL 优于人类示例"主要靠 R1-Zero 这一个单点对照支撑；② "aha moment" 属于观察性叙述（Table 2、Fig.9b 用 "wait" 的词频来说明），没有严格的涌现度量【原文 Appendix F.1；推断据其叙述性证据】。
- 假设与失效边界：
  - 【原文】R1-Zero 的可读性/语言混杂问题，需要 R1 管线里的人对齐数据来补救——所以"纯 RL 零人工"并不完全成立。
  - 【原文】偏好奖励模型在可验证任务上会 reward hacking（用 helpful RM 时，reward 升但 CodeForces 降，Appendix B.5）。
  - 【原文】规则奖励要求任务本身可验证（数学/代码/逻辑）。
  - 【推断】660B MoE base 的容量，可能是 R1-Zero 自演化成功的隐含前提；小 base 上做纯 RL 未必奏效（Appendix F.1 已暗示小模型直接 RL 不及蒸馏）。
  - 【推断】这里的蒸馏是离线、序列级的 SFT，与在线 / on-policy KD 不同。
- 祛魅总结：真贡献有三点——用大规模实证确立"纯 RL 可单独激励推理(R1-Zero)"这一范式级结论、完整公开四阶段配方、给出蒸馏 vs 直接 RL 的清晰对照（结论是蒸馏更经济有效，但"超越人类智能边界仍需更强 base + 更大 RL"）。要祛魅/留意高估的地方：
  - 【推断】Nature 版强调 "obviating human-labeled trajectories"，但 R1（非 Zero）仍用了 cold-start + 800k SFT，并非零人工。
  - 【推断】"self-evolution""aha moment" 的叙事偏拟人化，缺少定量的涌现指标。
  - 【推断】蒸馏"优于直接 RL"的那个对照里，直接 RL 的算力被有意压低，对照并不完全对等（作者已注明）。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**=R1-Zero 与推理用规则结果奖励（答案对错 + 格式）；通用任务用神经偏好奖励（helpful/harmless）；再叠 LC 语言一致性奖励。
  - **改什么**=R1 改策略全参数；R1-Zero 与蒸馏 SFT 改 base 全参数。
  - **何时改**=R1-Zero 纯 RL；R1 走四阶段（SFT→RL→SFT→RL）；蒸馏是一次性纯 SFT。
  - **免梯度?**=否，全程梯度（RL 与 SFT 都是梯度训练）。
  - **记忆-技能生命周期**=不涉及显式的记忆/技能库；推理能力固化进权重；"自演化"发现的推理模式被蒸馏进小模型，相当于技能跨模型转移。
  - **防遗忘机制**=GRPO 带 KL 惩罚（系数 0.001，对 ref 约束很弱）；多阶段管线靠混合数据缓解能力遗忘，没有专门的防遗忘算法【原文 §2-3】。
- ⑦ 开源代码+框架/harness：https://github.com/deepseek-ai/DeepSeek-R1 ；权重 https://huggingface.co/deepseek-ai 。
  - **该仓是模型/权重发布仓，不含训练代码**——没有 GRPO 训练脚本、没有四阶段管线实现、没有拒绝采样脚本。
  - 框架：训练基础设施在 Appendix B.1 描述为**自研的高性能 RL 框架**（不在仓内；A100 做预实验，再用 64×8 H800 训 660B）。
  - CloneTier=B，仅记录、未 clone。GRPO 算法后来被 veRL/TRL/OpenRLHF/ms-swift 广泛实现。
  - 【未开源训练码，复现依赖论文公式 + 超参表；本条为模型报告，无训练代码】
- 💰 资源/成本与可扩展性：**论文给出了明确成本(Appendix B.4.4, Table 7)**。
  - R1-Zero：64×8 H800 约 198h；R1 同卡约 80h（4 天）；SFT 数据 5K GPU 时。
  - 总计 **147K H800 GPU 时 ≈ \$294K**（拆开：R1-Zero 101K/\$202K + SFT 5K/\$10K + R1 41K/\$82K）。
  - 蒸馏是纯 SFT，成本远低于 RL；作者明示"小模型直接 RL 需要巨大算力，且可能不及蒸馏"【原文 Appendix B.4.4】。
- 🎯 对"探索-巩固"对标（**支撑探索侧的范式奠基 + 蒸馏是巩固的离线 baseline**）：
  - 一句判定：R1-Zero 是"探索 = 在 RL 压力下自演化、发现有效推理路径（反思/换路/自检）"的奠基性实证，直接定义了本项目"探索/选路"要利用的现象。
  - 而它那 800k 离线 SFT 蒸馏，是"把发现的推理能力固化进（小）模型"的最朴素巩固形式，也正是本项目 on-policy 蒸馏要超越的 off-policy baseline。
  - 依据：§2.3 的 self-evolution + Fig.1b 的长度增长（对应探索）；Appendix F 的离线 SFT 蒸馏（对应巩固/固化）。
  - 可借组件一：R1-Zero 的"规则奖励 + 只约束格式"(Eq.4)，是稀疏脚手架最干净的奖励设计参考。
  - 可借组件二：拒绝采样扩 SFT（从 RL checkpoint 采正确轨迹再 SFT），相当于一种"巩固自己走通路径"的离线版，可对比本项目的 on-policy 自选。
  - 缺口/竞品差异：R1 的蒸馏是**离线、序列级的 SFT(teacher-forced)，不是 on-policy KD**；没有路径恢复/回轨目标，没有 MTP 前瞻；teacher（R1）的轨迹一次性生成后就固定了。R1-Zero 则是纯探索、没有 teacher 脚手架。对比本项目"teacher 当稀疏脚手架、教 student 同时学会探索 + 巩固两种能力"，R1 是"先大模型自探索，再把结果离线灌给小模型"，师生交互弱、且非在线。
- 🔭 开放问题/未来方向：
  - 【原文】作者明示把"蒸馏小模型 + RL"留给社区（F："leaving the exploration of the RL stage to the broader research community"）；"超越人类智能边界仍需更强 base + 更大 RL"；偏好奖励的 reward hacking 待解(B.5)。
  - 【推断】以下都未涉及——用 on-policy/在线蒸馏替代离线 800k SFT；小 base 上纯 RL 的可行性；把自演化的探索与显式的路径恢复/巩固结合；用 MTP 等前瞻信号引导探索。

---
RETURN: deepseek_r1 | 读到PDF? 是(86页,读核心方法/蒸馏/成本,公式从正文.txt 精确抄录) | L3(纯RL激励推理)+L1(离线蒸馏对照) | 探索侧奠基性支撑(R1-Zero 自演化=探索范式);蒸馏=off-policy SFT,本项目 on-policy KD 要超越的 baseline;无路径恢复/MTP | 残留待核 0(此 v2 为 Nature 版,GRPO 优势 Eq.1/3 为序列级 \(A_i\),与 DeepSeekMath 逐 token \(\hat A_{i,t}\) 表述不同,已据 PDF 核实;为模型报告无训练代码)

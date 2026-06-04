latent_agents | Latent Agents: A Post-Training Procedure for Internalized Multi-Agent Debate | Boston University(John Seon Keun Yi, Aaron Mueller, Dokyun Lee) | 2026-04-27 · arXiv 2604.24881v1 · ACL 2026 Oral(据 README;PDF 内未见〔待核〕) | 主题线 L4 多智能体内化/蒸馏(兼 L2 SFT→RL 两阶段) · 相关性 中

**原始论文**:https://arxiv.org/abs/2604.24881

## 一眼看懂
> 一句话导读:把"几个模型来回辩论再投票"这套又贵又慢的外部流程,塞进一个模型的"脑内",让它直接出答案;靠两个随训练变化的奖励(逐渐不奖励啰嗦格式 + 逐渐收紧输出长度)把显式辩论挤进潜空间,token 省 5-16 倍还不掉点。

- 🟦 TL;DR:把"多个 agent 多轮辩论(multi-agent debate, MAD)"这套昂贵的外部推理过程,内化进**单个 LLM**(方法叫 IMAD)。两阶段微调:
  - ① SFT 在完整辩论 trace(辩论全过程记录,带 `<|Agent i|>`/`<|Round k|>`/`<|endofdebate|>` 这类结构标签)上学辩论结构。
  - ② GRPO 用两个动态奖励:格式奖励(权重 \(w_{\mathrm{fmt}}\) 随训练衰减到 \(\sim 0\))+ 正确性×长度裁剪奖励(长度上限 \(l\) 从 2000 退火到 500)。两者把显式、啰嗦(verbose)的辩论挤压进潜空间,让模型直接产答案。
  - 结果:用 Debate 6.3%-21.1% 的 token(5-16× 提效)匹配/超过显式辩论。
  - 附带两个发现:内化后存在**线性可分的"agent 子空间"**(可用 CAA steering 向量 \(v_i\) 激发出各 agent 的风格);并可用**负向 steering 抑制恶意 agent**而几乎不伤通用能力。
- 最巧的一步:**双动态奖励的"合谋挤压"——格式奖励衰减 \(w_{\mathrm{fmt}}\to 0\) + 长度上限退火 \(l_0\to l^\ast\)**(§2.3,Eq.1-2)。抽掉任一都不行:
  - 只衰减格式奖励、不缩长度 → 模型仍可啰嗦辩论。
  - 只缩长度、不衰减格式奖励 → 模型被逼在短输出里还硬凑结构标签,与正确性奖励冲突。
  - 两者合力,才让"把多视角分析转入潜空间、直接出答案"成为模型唯一可行策略。§2.5 实测:RL 阶段相对 SFT 再降 token 至多 66%,验证此调度。

## 为什么做
> 一句话导读:多模型辩论有效但贵,而前人那种"只蒸馏最终共识"的省钱做法又打不过原版辩论;本文要做的是既省 token 又不丢中间交互,顺带研究"内化后辩论结构还在不在、能不能控制"。

- 研究背景与来龙去脉:本文坐落在三条(加一条工具)并行路线的交叉口。
  - ① **多智能体辩论(MAD)**:Du 2023(improving factuality through debate)、Liang 2024(encouraging divergent thinking)用**多个模型实例多轮对话**互相质疑、收敛共识,实证降幻觉、提事实性与推理。代价:每答一题要跑完整条"多 agent × 多轮"的 transcript(对话记录),token 随 agent 数 × 轮数线性膨胀。
  - ② **辩论蒸馏**:DebateGPT(Subramaniam 2024)把 MAD 产物蒸进单模型,但**只蒸最终 consensus(共识)输出**——token 最省却普遍打不过显式辩论,因为丢掉了驱动增益的中间多视角交互。
  - ③ **隐式推理 / 长链内化**:本文压缩机制**明示借鉴 ThinkPrune(Hou et al. 2025)**——后者在训练中**渐进缩短长推理链**来内化显式 CoT(思维链);本文把"渐进缩短"从单链 CoT 搬到多 agent 辩论 trace。
  - ④ **激活引导(可解释/可控,工具)**:CAA(Contrastive Activation Addition,Rimsky 2024)与 difference-in-means(均值差,Marks & Tegmark 2023)提供"用对比激活差提取方向向量、推理时加到隐藏态"的工具;本文借它检验"内化后是否长出可定位的 agent 子空间"。
- 三条路线**具体**短板:
  - MAD —— 推理贵、多模型常驻、真并行 agent 间易**协调失败**(各说各话、不收敛)。
  - DebateGPT —— 只学最终 consensus、**丢中间交互**、普遍不如显式辩论。
  - ThinkPrune —— 只缩单链 CoT、**不涉及多 agent 结构**。
  - CAA/steering —— 是分析/控制工具、**本身不做内化**。
  - Δ(本文独有):**首次把 SFT→RL 两阶段范式(Shao 2024 / Guo 2025 DeepSeek 式流水线)用于内化 MAD**,且在**完整辩论 trace**(非仅共识)上 SFT 学结构,再用 length-annealing(长度退火)GRPO 压成潜推理,并附"内化后 agent 子空间线性可分 + 可负向 steering 抑制恶意 agent"的机制发现。
- 解决的具体痛点:
  - ① 显式辩论 token 开销巨大(多模型多轮 transcript 才给答案)。
  - ② 只蒸共识(DebateGPT)token 最省但性能普遍不如显式辩论——丢掉了驱动增益的中间交互。
  - ③ 缺乏对"内化后辩论结构是否仍保留、能否被控制"的机制理解。
- 动机链:显式辩论贵、只蒸共识又丢交互(现状/缺陷)→ 在完整 trace 上 SFT 学结构 + GRPO 双动态奖励把结构挤进潜空间(所以这样)→ 进一步追问内化后 agent 表示是否留痕、可否控制(衍生出 §3/§4)。
- 与最近邻工作的 Δ:
  - vs **DebateGPT**(最近邻):IMAD 在**完整辩论 trace**上学(保留中间多视角交互),DebateGPT 只学最终输出。关键有用点:IMAD 既省 token 又超过显式 Debate——作者解释为 SFT 模型在**每个 token 都能看到完整辩论历史**(单流自回归,而非真并行 agent),反而比真 MAD 更易学到 agent 间关系、缓解协调失败;DebateGPT token 最省却普遍最弱。
  - vs **ThinkPrune**:把"渐进缩短长链"从单链 CoT 推广到"把多 agent 辩论结构整体压入潜空间",并额外引入**格式奖励衰减**与长度退火配合(单缩长度不够,见下文双调度)。

## 怎么做 + 靠不靠谱
> 一句话导读:先用 GPT-3.5 跑出辩论数据并打上结构标签,SFT 让单模型学会"自己演一遍辩论",再用 GRPO 配双动态奖励把这场辩论压进潜空间;实验只在小规模(算术训练、LoRA、944 条)上做,泛化结论偏 promising 而非定论。

- 方法流水线(Fig.1,三阶段,具体到输入→输出):
  - **阶段 0 · 数据生成(外部 MAD 采样)**:用标准 MAD(GPT-3.5-turbo 当 agent,\(n=3\) agents、\(m=2\) rounds、majority vote 多数投票)在算术题(6 个两位数表达式)上跑辩论,采 944 条 \(\{\text{Question},\text{Trace},\text{Answer}\}\);给 trace 插入结构标签 `<|Agent i|>` / `<|Round k|>` / `<|endofdebate|>`,**丢弃无共识的样本**。输入=题;输出=带标签的完整辩论 trace + 最终答案。
  - **阶段 1 · SFT 学结构**:在**整条 trace**上做 next-token 交叉熵(LoRA),让单模型学会"给一道题→自回归生成完整的结构化辩论"。输入=Question;监督=整条 trace(含中间交互,非仅共识)。这步把外部多 agent 动态蒸进单 agent,但**不保证答案正确、也不保证 agent 间对齐/收敛**(作者实测 SFT 后偶有幻觉 / misalignment 不对齐)。
  - **阶段 2 · GRPO 内化(双动态奖励)**:从 SFT 模型 \(\pi_\theta\) 初始化,每步对题 \(x\) 采 \(k\) 个候选,按 reward 打分、对**奖励不同的输出对**构 on-the-fly(即时)偏好对更新(LoRA)。reward 见下 Eq.1。两个动态调度联手把显式辩论挤进潜空间:
    - 格式奖励权重 \(w_{\mathrm{fmt}}:1.0\to 0.05\) 衰减(撤掉"啰嗦辩论"的激励)。
    - 长度上限 \(l:2000\to 500\) 退火(使"既啰嗦又拿到正确性奖励"变不可能)。
    - 合起来 → 模型**唯一可行策略 = 把多视角分析放进潜空间、直接出答案**。
- 模块如何咬合:阶段 1 负责"学会辩论的形状",阶段 2 负责"把这个形状从显式 token 压成隐式计算并保正确"。两个奖励项**反向拉扯出唯一解**:
  - 只衰减 \(w_{\mathrm{fmt}}\)、不缩 \(l\) → 模型仍可啰嗦辩论。
  - 只缩 \(l\)、不衰减 \(w_{\mathrm{fmt}}\) → 模型被逼在短输出里硬凑结构标签,与正确性奖励冲突。
  - 两者合力才逼出"内化"。
- 核心算法/损失(真实形式 + 直觉,符号从 PDF §2.3/§3 抄准):
  - **复合奖励**(Eq.1):\(\ r(x,y)=w_{\mathrm{fmt}}R_{\mathrm{fmt}}+w_{\mathrm{clip}}R(y;l)\)。\(R_{\mathrm{fmt}}\)=结构标签 token-matching 命中给正分(随训练靠 \(w_{\mathrm{fmt}}\to 0\) 淡出);\(R(y;l)\)=长度裁剪正确性奖励。
  - **长度裁剪正确性奖励**(Eq.2,直觉=逼模型把正确答案**尽早**放出):
    \(\displaystyle R(y;l)=\begin{cases}1, & \text{若 } y^\ast\in \mathrm{clip}(y,l),\\[2pt] 0, & \text{否则},\end{cases}\)
    其中 \(\mathrm{clip}(y,l)\) 把输出 \(y\) 截到前 \(l\) 个 token,\(y^\ast\) 是正确终答;只有正确答案落在前 \(l\) token 内才给 1。length annealing \(l_0\to l_1\to\cdots\to l^\ast\) 防一开始就限死、扼杀对推理空间的探索。
  - **CAA / difference-in-means steering 向量**(Eq.3,§3,仅用于内化后分析/控制,**不参与训练**):对 agent \(i\),固定上下文(题 + 到该 agent 标签前的辩论历史),正激活=接该 agent 的原始回应、负激活=接其余两 agent 回应的平均;向量为均值差
    \(\displaystyle v_i=\frac{1}{|D|}\sum_{p,c\in D}\Big(h_\ell(p,c_i)-h_\ell(p,c_{\neg i})\Big),\)
    \(h_\ell\) 为第 \(\ell\) 层激活,\(D\) 为辩论数据集。推理时 \(\ h_\ell\leftarrow h_\ell+\alpha\cdot v_i\):\(\alpha>0\) 放大、\(\alpha<0\) 抑制该 agent 特质。向量**从 SFT 模型(RL 前)提取**,以隔离"学到的表示"与"RL 优化伪影"。
- 关键超参与默认值:\(n=3\) agents、\(m=2\) rounds;数据 944 trace;\(w_{\mathrm{fmt}}\) 1.0→0.05;\(l\) 2000→500;SFT 3–6 epoch + GRPO 2 epoch(LoRA);backbone LLaMA-3.1-8B / Qwen2.5-7B / Mistral-Nemo-12B。
- 逐组件必要性:
  - **结构标签**:无标签则 agent 难区分、内化复杂化,且 §3 子空间分离变弱(有间接证据,非独立准确率消融)【原文 §2.1+§3】。
  - **在完整 trace 上 SFT(vs 只学 consensus)**:Table 1 中 DebateGPT(只学 consensus)普遍弱于 IMAD/Debate,构成对照消融。
  - **格式奖励衰减 + 长度退火(双动态奖励)**:§2.5 报告 RL 相对 SFT 再降 token 至多 66% 且精度升,验证调度有效;但**未做"固定 \(w_{\mathrm{fmt}}\) / 固定 \(l\)"的严格 ablation**(标出:无该消融,结论靠机制论证 + 整体曲线)【推断缺口】。
  - **SFT 阶段必要性**:作者称 SFT 后 agent 已有幻觉/misalignment(少量),RL 修正正确性——SFT/SFT+RL 两行均报(Table 1),构成阶段消融。
- 实验与证据:
  - 数据集:训练只用算术;评测 GSM8K(多步数学)、MMLU-Pro(多域选择)、BBH(多样推理),各采 1000 题,3 次取均值 ± 标准误。Backbone:LLaMA-3.1-8B、Qwen2.5-7B、Mistral-Nemo-12B。
  - baseline:Single Agent(单 agent,zero-shot)、Debate(3 agent × 2 round + 多数投票)、DebateGPT(只蒸共识),以及自家的 SFT 与 SFT+RL 两阶段。
  - 主结果(Table 1):IMAD 用 Debate 的 **6.3%-21.1% token(5-16×)**匹配/超过 Debate。
    - LLaMA-3.1-8B 最适配——三 benchmark 全超 Debate 且方差低(GSM8K 85.20 vs Debate 83.03,token 644 vs 5758)。
    - Mistral 的 GSM8K 超 Debate 18.97 个百分点;Qwen 提升温和。
    - DebateGPT token 最省但普遍最弱。
  - 跨域泛化:只训算术却在 GSM8K/MMLU-Pro/BBH 一致提升 → 内化的是通用推理结构而非任务特定(Appendix J:多任务训练更强)。
  - §3 子空间(Fig.2):IMAD vs base 在相同 steering 向量下 ROUGE-L AUC +12.7%(全条件均值 +15.41%,Agent3 PoT 最显著 21-25%);α=0 时两者相同 → 特质只在 steering 下显现 = 内化造出可定位的 agent 子空间。
  - §4 控制(Fig.3):内化恶意 agent(evil/hallucination)后,IMAD 负向 steering 可把 evil 特质压到 0(α=-3~-5),base 即便 α=-5 仍残留 1.01;且对 GSM8K 通用能力伤害更小 → 内化使恶意特质更可定位、可干净移除。
- 假设与失效边界:
  - 【原文】数据只用算术题(短答案、重结构、轻长推理);作者明言选它是为了聚焦"结构学习"而非长推理。n=3/m=2 已够,更多 agent/轮次边际收益小。
  - 【原文】SFT 模型在每个 token 都可见完整历史(非真并行 agent),作者认为这反而有利于学 agent 关系——意味着内化的"辩论"与真 MAD 并不完全等价(是单流自回归的模拟)。
  - 【推断】概念验证级:944 条 trace、LoRA、算术训练;长上下文复杂推理、更大规模能否保持,未验证。
  - 【推断】steering 子空间分析是在 SFT checkpoint 上提向量(避开 RL 优化伪影),意味着"agent 子空间"主要是 SFT 学结构的产物;RL 内化后子空间是否仍线性可分,原文未直接量化。
- 祛魅总结【推断】:
  - 真贡献:① SFT→RL + 双动态奖励把显式 MAD 压成潜推理,token 省 5-16× 且常超显式辩论,工程上漂亮;② "内化后 agent 子空间线性可分 + 可负向 steering 抑制恶意 agent" 是有趣且可复现的机制发现(可解释性 + 安全)。
  - 包装/可能高估:"Latent Agents / 内化辩论" 叙事很吸睛,但实质是"在完整 trace 上 SFT + length-annealing GRPO"(借 ThinkPrune 的缩长链思路);训练域窄(算术)、规模小(LoRA、944 条),泛化结论偏 promising 而非定论;"agent 子空间" 增益(ROUGE AUC +12-15%)中等,且依赖 steering 才显现。低估:把"昂贵外部多 agent 过程内化进单模型 + 内化后可控"作为一类范式,价值可能超出本文的小规模验证。

## 结构化抽取
- 🎯 机制速览 6 轴:
  - **学什么信号**=SFT 学辩论 trace 结构(CE 交叉熵);RL 学 outcome 正确性 + 格式标签(format reward)+ 长度(length clip)。
  - **改什么**=单一 LLM 参数(LoRA adapter)。
  - **何时改**=两阶段(先 SFT 学结构,后 GRPO 内化,reward 权重/长度随训练调度)。
  - **免梯度?**=否(SFT CE + GRPO 策略梯度);steering 控制阶段是推理时加向量、免训练。
  - **记忆-技能生命周期**=无外部记忆;"多视角辩论技能"经 SFT→RL 固化进权重并压入潜空间,内化后留下线性可分的 agent 子空间。
  - **防遗忘机制**=隐式——格式奖励衰减 + 长度退火是渐进式内化(非突变),保留中间交互能力;负向 steering 移除恶意特质时对 GSM8K 伤害小(可定位移除 ≠ 全局遗忘)。
- ⑦ 开源代码+框架/harness:https://github.com/johnsk95/latent_agents (已 clone,真实可用)。
  - **框架=TRL**(GRPOConfig/GRPOTrainer,见 grpo.py L27/279/310)+ **PEFT/LoRA**(sft.py get_peft_model;grpo.py LoraConfig)+ **HF transformers**。
  - requirements:trl>=0.11、peft>=0.13、transformers>=4.46;ROUGE 用 evaluate/rouge_score;persona/judge 用 openai。
  - steering 代码在 `steering/` 目录。
- 💰 资源/成本与可扩展性:LoRA 两阶段(SFT 3-6 epoch + GRPO 2 epoch);长度从 2000 退火到 500、格式奖励权重 1.0→0.05;backbone 7B-12B。数据极小(944 + 500/100 trace)。具体 GPU 数/时长原文未给明确表(Appendix B 详设)。亮点:推理 token 省 5-16×。
- 🎯 对"探索-巩固"对标:**部分支撑(范式同构)+ 弱竞品**。判定:IMAD 的 "SFT 学外部过程结构 → RL 把显式过程压进潜空间/固化进参数" 与本项目"巩固(把发现的有效行为固化进参数)+ length-annealing 把显式长推理压成隐式"在**范式上同构**;但它内化的是"多 agent 辩论"而非"探索-选路/回轨"。可借组件:① **length-annealing + 格式奖励衰减的双调度**可直接平移为 TSRD"把教师脚手架下的显式恢复步骤渐进压成隐式自恢复"的训练 schedule;② **CAA difference-in-means steering 向量 + 负向抑制**可作"定位并固化/抑制特定推理分支(如错误路径 vs 恢复路径)"的工具,呼应 path-recovery 单点接管的可控性诉求。缺口/竞品性:无 teacher-student logit 蒸馏、**无 MTP/前瞻**、奖励是 outcome+format 非 path/step 级监督,且训练域窄(算术、LoRA、小规模);TSRD 的 on-policy 自选恢复分支 + MTP 探针是其完全留白。
- 🔭 开放问题/未来方向:【原文】更多 agent/轮次 + 长上下文复杂任务上训练(Appendix J 多任务已更强);内化后 agent 子空间的更深机制理解。【推断】把"内化辩论"换成"内化探索-巩固循环"(用类似双调度把教师脚手架的 path-recovery 压成隐式);RL 内化后子空间是否仍线性可分、能否用 steering 在线引导"换到能走通的推理分支";规模化(全参 + 更大模型 + 真推理数据)验证泛化是否仍成立。

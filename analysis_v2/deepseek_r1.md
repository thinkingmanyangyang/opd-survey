deepseek_r1 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | DeepSeek-AI | arXiv 2501.12948(v2 2026-01-04，Nature 版)·2025-01 首发 | 主题线 L3(RLVR/GRPO)+ L1(离线序列级蒸馏对照)·相关性 高

**原始论文**:https://arxiv.org/abs/2501.12948

## 一眼看懂
- 🟦 TL;DR:**R1-Zero** 证明仅靠纯 GRPO + 规则奖励(不经任何 SFT)就能在 base 模型上自演化出推理能力(响应自然变长、涌现反思/"aha moment"),AIME pass@1 从 15.6%→77.9%;**R1** 再用四阶段管线(cold-start SFT → 推理向 RL 加语言一致性奖励 → 拒绝采样扩 SFT(800k)→ 全域 RL 加偏好奖励)修好可读性与通用性;并把 800k 数据**离线 SFT** 蒸馏到小模型,得出"大模型 RL→小模型蒸馏 优于 小模型直接 RL"【原文 Abstract, §2, §3, Appendix F.1】。
- 最巧的一步:**规则奖励 + 只约束格式不约束内容**(R1-Zero 的设计)。抽掉它(改用神经/过程奖励模型,或规定推理步骤),就观察不到"无人类示例约束下自演化出非人类式推理路径"这个核心现象——作者明说刻意不用神经奖励模型(避免 reward hacking)、刻意跳过 SFT(假设人类定义的推理模式会限制探索)【原文 §2.2, §2.3】。为什么:奖励只看答案对错 + 格式,模型在 RL 压力下自由地延长并重组推理(thinking time 单调增长,Fig.1b),涌现自检/反思/换路;一旦约束内容或引入可被 hack 的神经奖励,这种自由探索就被压制。

## 为什么做
- 研究背景:LLM + CoT prompting 在推理上有成效,但严重依赖人工标注 CoT 轨迹做 SFT;test-time scaling(o1)显示拉长推理可显著提升能力;GRPO(源自 DeepSeekMath)提供免 critic 的高效 RL【原文 §1, §2.1】。
- 解决的具体痛点:依赖人工推理轨迹→扩展性差、引入认知偏差、性能被人类示例上限封顶,难探索"非人类式"更优推理路径;传统 SFT-before-RL 可能限制探索;且纯 RL 的 R1-Zero 虽强但可读性差、中英混杂(language mixing)、窄域 RL 致通用能力弱【原文 §1, §3】。
- 相关工作 & 各自不足:CoT prompting(Wei 2022)受人类示例上限约束;SFT-on-human-CoT 引入偏差且封顶;神经/过程奖励模型易 reward hacking 且重训成本高(故 R1-Zero 弃用)【原文 §1, §2.2】。
- 动机链:现状(推理靠人标 CoT+SFT)→缺陷(扩展性差、被人类示例封顶、无法探索非人类路径)→所以必须(用最小人工标注、靠纯 RL 自演化激励推理:R1-Zero 跳过 SFT 直接 base 上 RL、奖励只看最终答案;R1 再用少量 cold-start + 多阶段补可读性/通用性)【原文 §1-3】。
- 与最近邻工作的Δ:相对 o1(闭源、细节未公开)与 SFT-on-CoT 范式,差在"证明纯 RL 单独即可激励推理(不需 SFT 引导)"+"完整公开四阶段配方与超参"+"系统给出蒸馏 vs 直接 RL 的对照"。有用点:R1-Zero 把"RL 可独立涌现推理"这一假设证实,改变了整个社区的 post-training 范式(GRPO 后被 veRL/TRL/OpenRLHF 等广泛实现)。

## 怎么做 + 靠不靠谱
- 方法流水线(R1 四阶段,Fig.2):输入 DeepSeek-V3-Base(660B MoE)→ ①Cold Start:数千条对话式人类对齐长 CoT 数据 SFT → ②推理向 RL:规则奖励(准确率+格式)+ 语言一致性(LC)奖励 → Dev-2 → ③拒绝采样 + SFT:从第一阶段 RL checkpoint 拒绝采样生成推理轨迹、V3 当生成式裁判过滤(混语/长段/代码块)→ 约 600k 推理样本 + V3 管线生成约 200k 非推理样本 = 约 800k 监督数据 SFT → Dev-3 → ④全域 RL:混合推理(规则奖励)+ 通用(helpful/harmless 偏好奖励)数据 → 最终 R1。另:R1-Zero = V3-Base 纯 GRPO(无前述 SFT)【原文 §2.3, §3, Appendix B.3.3】。
- 逐组件必要性(部分有消融):
  - **R1-Zero 纯 RL**:核心对照。AIME 15.6%→77.9%(cons 86.7%),证明 RL 单独可激励推理。没它→无法验证"不需 SFT"这一核心假设【原文 §2.3】。
  - **Cold Start SFT**:对照 Dev-1 vs R1-Zero——指令遵循(IF-Eval/ArenaHard)大涨,但因 cold-start 数据量小,AIME 推理略降【原文 §4 Table 3】。
  - **语言一致性(LC)奖励**:有消融(Appendix B.6)。加 LC 后语言一致性稳定,代价是基准性能略降——作者明说"slight degradation"但更可读、合人类偏好【原文 §3.2.1, Fig.7】。
  - **第四阶段全域 RL**:对照——AlpacaEval2.0 +25%、ArenaHard +17%(通用能力补齐);但偏好奖励仅在最后 400 步引入,因更多步会致 reward hacking(Appendix B.5)【原文 §3.2.2】。
  - **GRPO clip ε**:作者强调 clip ratio 关键——过低截断大量 token 梯度损性能,过高致训练不稳(第一阶段 ε=10)【原文 §3.2.1】。
- 关键机制/公式(直觉):①GRPO(Eq.1-3):每问题采 G=16 个输出,组内 reward 均值/标准差归一得 advantage(无价值网络),KL 直接加损失(此 Nature 版优势 A_i 为**序列级**,非 DeepSeekMath 的逐 token A_{i,t});②规则奖励(Eq.4)= 准确率(box 答案规则校验/代码编译器+测试)+ 格式(强制 `<think></think><answer></answer>`)等权;③LC 奖励(Eq.7)= CoT 中目标语言词占比;④全域 RL 奖励(Eq.8)= reasoning(规则)+ general(偏好模型,helpful 只评 summary、harmless 评整段)+ language【原文 §2.1-2.2, §3.1-3.2】。
- 实验与证据:R1-Zero 超参:lr 3e-6、KL 系数 0.001、rollout temperature 1、每题 16 输出、最大长 32768(8.2k 步后增至 65536)、每步 32 题 batch 512、每 400 步更新 ref、共 10,400 步(约 1.6 epoch);第一阶段 RL:同上 + ε=10;第二阶段 RL:temperature 降至 0.7、共 1,700 步、偏好奖励仅末 400 步;SFT(cold-start/拒绝采样):2-3 epoch、cosine lr 5e-5→5e-6、ctx 32768、batch 128;蒸馏:对应 base 微调 2-3 epoch、ctx 32768、batch 64(各模型初值见 Table 6)。评测 MMLU/AIME2024/MATH-500/GPQA/LiveCodeBench/Codeforces/AlpacaEval/ArenaHard 等。R1-Distill 结果(Table 15):Qwen-32B AIME 72.6、Llama-70B 70.0【原文 §2.1, §3.2, Appendix B.4.3, §4】。baseline 公平性:**蒸馏 vs 直接 RL 对照(Appendix F.1, Table 16)** ——R1-Distill-Qwen-32B(AIME 72.6)显著超 Qwen2.5-32B-Zero(同 base 直接 RL,47.0),但作者自承小模型直接 RL 的算力预算与 R1 不对等(直接 RL "require enormous computational power")。"看着强但没回答"的点:"无约束 RL 优于人类示例"主要靠 R1-Zero 单点对照支撑;"aha moment"为观察性叙述(Table 2、Fig.9b 用"wait"词频),无严格涌现度量【原文 Appendix F.1; 推断据其叙述性证据】。
- 假设与失效边界:【原文】R1-Zero 的可读性/语言混杂问题需 R1 管线人对齐数据补救——故"纯 RL 零人工"不完全成立;偏好奖励模型在可验证任务上会 reward hacking(helpful RM:reward 升而 CodeForces 降,Appendix B.5);规则奖励要求任务可验证(数学/代码/逻辑)。【推断】660B MoE base 的容量是 R1-Zero 自演化成功的隐含前提,小 base 上纯 RL 可能不奏效(Appendix F.1 已暗示小模型直接 RL 性能不及蒸馏);蒸馏为离线序列级 SFT,与在线/on-policy KD 不同。
- 祛魅总结:真贡献=用大规模实证确立"纯 RL 可单独激励推理(R1-Zero)"这一范式级结论 + 完整公开四阶段配方 + 蒸馏 vs 直接 RL 的清晰对照(蒸馏更经济有效,但"超越人类智能边界仍需更强 base + 更大 RL")。包装/高估处:Nature 版强调"obviating human-labeled trajectories",但 R1(非 Zero)仍用了 cold-start + 800k SFT,并非零人工;"self-evolution""aha moment"叙事偏拟人化,缺定量涌现指标;蒸馏"优于直接 RL"的对照中,直接 RL 的算力被有意压低(对照不完全对等,作者已注明)【推断】。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=R1-Zero/推理:规则结果奖励(答案对错+格式);通用:神经偏好奖励(helpful/harmless);+ LC 语言一致性奖励 | **改什么**=策略全参数(R1)/ base 全参数(R1-Zero、蒸馏 SFT)| **何时改**=R1-Zero 纯 RL;R1 四阶段(SFT→RL→SFT→RL);蒸馏=纯 SFT 一次性 | **免梯度?**=否(全程梯度;RL+SFT 均梯度训练)| **记忆-技能生命周期**=不涉及显式记忆/技能库;推理能力固化进权重;"自演化"发现的推理模式被蒸馏进小模型(技能跨模型转移)| **防遗忘机制**=GRPO 带 KL 惩罚(系数 0.001,弱约束 ref);多阶段管线靠混合数据缓解能力遗忘,无专门防遗忘算法【原文 §2-3】。
- ⑦ 开源代码+框架/harness:https://github.com/deepseek-ai/DeepSeek-R1;权重 https://huggingface.co/deepseek-ai 。**该仓为模型/权重发布仓,不含训练代码**(无 GRPO 训练脚本、无四阶段管线实现、无拒绝采样脚本)。框架:训练基础设施在 Appendix B.1 描述为**自研高性能 RL 框架**(不在仓内,A100 预实验 → 64×8 H800 训 660B)。CloneTier=B,仅记录未 clone。GRPO 算法后被 veRL/TRL/OpenRLHF/ms-swift 广泛实现。【未开源训练码,复现依赖论文公式+超参表】
- 💰 资源/成本与可扩展性:**论文给出明确成本(Appendix B.4.4, Table 7)**:R1-Zero 64×8 H800 约 198h;R1 同卡约 80h(4 天);SFT 数据 5K GPU 时;总计 **147K H800 GPU 时 ≈ $294K**(R1-Zero 101K/$202K + SFT 5K/$10K + R1 41K/$82K)。蒸馏为纯 SFT,成本远低于 RL;作者明示"小模型直接 RL 需巨大算力且可能不及蒸馏"【原文 Appendix B.4.4】。
- 🎯 对"探索-巩固"对标:**支撑(探索侧的范式奠基) + 蒸馏是巩固的离线 baseline** —— 一句判定:R1-Zero 是"探索=在 RL 压力下自演化发现有效推理路径(反思/换路/自检)"的奠基性实证,直接定义了本项目"探索/选路"要利用的现象;而其 800k 离线 SFT 蒸馏是"把发现的推理能力固化进(小)模型"的最朴素巩固形式,正是本项目 on-policy 蒸馏要超越的 off-policy baseline。依据:§2.3 self-evolution + Fig.1b 长度增长(探索);Appendix F 离线 SFT 蒸馏(巩固/固化)。可借组件:① R1-Zero 的"规则奖励+只约束格式"是稀疏脚手架最干净的奖励设计参考;②拒绝采样扩 SFT(从 RL checkpoint 采正确轨迹再 SFT)= 一种"巩固自己走通路径"的离线版,可对比本项目 on-policy 自选。缺口/竞品差异:R1 的蒸馏是**离线序列级 SFT(teacher-forced),非 on-policy KD**,无路径恢复/回轨目标、无 MTP 前瞻、teacher(R1)轨迹一次性生成后固定;R1-Zero 是纯探索无 teacher 脚手架。与本项目"teacher 当稀疏脚手架教 student 探索+巩固两能力"相比,R1 是"先大模型自探索→再把结果离线灌给小模型",师生交互弱且非在线。
- 🔭 开放问题/未来方向:【原文】作者明示把"蒸馏小模型 + RL"留给社区(F:"leaving the exploration of the RL stage to the broader research community");"超越人类智能边界仍需更强 base + 更大 RL";偏好奖励的 reward hacking 待解(B.5)。【推断】on-policy/在线蒸馏替代离线 800k SFT、小 base 上纯 RL 的可行性、把自演化的探索与显式的路径恢复/巩固结合、用 MTP 等前瞻信号引导探索均未涉及。

---
RETURN: deepseek_r1 | 读到PDF? 是(86页,读核心方法/蒸馏/成本) | L3(纯RL激励推理)+L1(离线蒸馏对照) | 探索侧奠基性支撑(R1-Zero 自演化=探索范式);蒸馏=off-policy SFT,本项目 on-policy KD 要超越的 baseline;无路径恢复/MTP | 残留待核 0(注:此 v2 为 Nature 版,GRPO 优势为序列级 A_i,与 DeepSeekMath 逐 token A_{i,t} 表述略异;为模型报告无训练代码)

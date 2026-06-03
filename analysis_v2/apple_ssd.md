apple_ssd | (SSD) Embarrassingly Simple Self-Distillation Improves Code Generation | Apple（Ruixiang Zhang / Richard He Bai / Huangjie Zheng / Yizhe Zhang 共一） | 2026-04-01 arXiv v1 · preprint | L1 自蒸馏(off-policy 纯 SFT) · 相关性中（与 OPD 对照/边界案例）

**原始论文**：https://arxiv.org/abs/2604.01193

## 一眼看懂
- 🟦 TL;DR：不用任何 teacher/verifier/reward/RL/代码执行环境，只让模型在"调过温度 Ttrain≠1 + 截断 ρ"下采样自己的**原始、未验证**输出，再用标准交叉熵 SFT，评估时单独调一个解码温度 Teval——就把代码生成显著提升(Qwen3-30B-Instruct LCB v6 pass@1 42.4%→55.3%，§3.2 Table 2)，增益集中在中/难题，5 个模型(Qwen/Llama × 4B/8B/30B × instruct/thinking)全部提升。
- 最巧的一步：**训练时用"非 1.0 温度 + 截断"采样作为 SFT 目标**（§2 Eq.1，即 Ttrain≠1 + ρtrain）。抽掉它(Ttrain=1 无截断)就垮——SSD 的全部增益来自"在低温/截断下采样等于先把分布尾巴(含 lock 处 distractor)削掉再当目标"，于是 SFT 学到的新分布**按上下文**自动在 lock 处压 distractor、在 fork 处留多样性(§4)。这是固定解码温度永远做不到的(§3.3 证明 best-tuned base 仍差 SSD 11.8pp)。

## 为什么做
- 研究背景：编码任务越来越难，高质量监督信号成瓶颈——人工解贵；合成数据要么需更强 teacher、要么需对每题做执行验证(§1)。teacher 蒸馏继承 teacher 上限；RLVR 操作复杂可能不稳；无监督内禀奖励法(majority voting/熵最小化)有早期成效但面临 reward hacking 与长训崩溃。
- 解决的具体痛点：**能否让模型在完全不借助任何外部标注/验证下自我提升？**(§1 末)现有路径都需 teacher/verifier/reward/执行环境/人工解之一。
- 相关工作 & 各自不足（Table 1 范式对比）：SFT on External Data(需外部数据)；GRPO(需 verifier)；On-Policy Distillation(需 teacher)；On-Policy Self-Distillation(需 verifier 筛选);**SSD = 无 teacher/无 verifier/无 privileged info 的 off-policy 自蒸馏**。
- 动机链：现状(自提升都需某种外部信号)→缺陷(贵/继承上限/不稳/崩溃)→所以做最简、零外部依赖的自蒸馏并搞清它为何 work。为什么选代码域？§1、§3 解释——代码的任务结构(fork/lock 位置并存)让底层机制**特别可见**，便于分析。
- 与最近邻工作的Δ：相对 on-policy distillation/self-distillation，差在**完全 off-policy(从冻结 base 采样后 SFT，不在线更新采样分布)+ 不做任何正确性筛选**；相对熵最小化等无监督法，差在**用温度+截断的采样配置**作为重塑分布的杠杆而非内禀奖励，避免 reward hacking。

## 怎么做 + 靠不靠谱
- 方法流水线（§2，三步极简）：① **Sample**：冻结 base pθ，以温度 Ttrain(非 1.0) + 截断 ρtrain(top-k/top-p) 对每 prompt 采 N 个候选(N=1 即够，§3)，**不做任何验证/筛选**(仅删空响应/单行 stub) → ② **Fine-tune**：对这些原始输出做标准交叉熵 SFT(Eq.2) → ③ **Decode**：评估时用单独调过的 Teval/ρeval 解码(Eq.3)。
- 逐组件必要性：
  - **Ttrain≠1 + 截断采样**：核心。§3.4 grid search——no-trunc 下 Teff=Ttrain·Teval 主导性能(R²=0.75，峰值 Teff≈1.2)；加截断后天花板更高(best: Ttrain=2.0/Teval=1.1/top-k=10 → 49.7% pass@1)。无它则无重塑。
  - **训练 + 解码协同(非互换)**：§4.2 toy 证明——训练只让 lock 更稳、不解决 fork；解码调温只动 fork、不清 lock；两者**必须同时**才有增益。§3.3 进一步：best-tuned base(仅调 Teval) pass@1 spread 仅 2.2pp，仍差 SSD 11.8pp。
  - **未验证原始输出**：刻意不筛，论文用"增益来自分布重塑而非数据筛选"论证逻辑闭环(§4)。
- 关键机制/公式（直觉）：**precision-exploration conflict**(§4.1)——代码有两类位置：**fork**(多续写都合理、对应不同解法，需多样性/高温) vs **lock**(语法语义近乎唯一但有低概率 distractor 尾巴，需精度/低温)。温度缩放 `p_T(v)∝p(v)^{1/T}` 全局拉平/锐化整个分布，所以任何**全局固定温度**必是折中(帮 fork 的温度恰好让 lock 的 distractor 复活，Fig 4)。SSD 把训练净效果形式化为对 token 分布的 **support compression(支撑压缩)+ within-support reshaping(支撑内重塑)**(Eq.4)：lock 处最狠地削尾巴→主 token 更难被挤掉(变 spike)；fork 处尾巴削掉但多个高概率续写保留并更均匀(变 plateau)，Fig 5。
- 实验与证据：5 模型(Llama-3.1-8B-Instruct、Qwen3-4B/30B 各 Instruct/Thinking)；数据 rSTARcoder seed 子集去重~10K 竞赛题；主基准 LCB v6(按难度分)，辅 LCB v5(374 题)。关键数字(Table 2)：Qwen3-30B-Instruct LCB v6 pass@1 42.4→55.3(+12.9pp/+30.4%)；难题 pass@1 +15.3pp、**hard pass@5 31.1→54.1(+23.0pp)**(关键旁证：若只锐化单模 coverage 该降而非升)；5 模型全提升(pass@1 Llama +3.5/4B-Ins +7.5/4B-Think +3.3/30B-Think +2.1/30B-Ins +12.9)。三路机制证据互证：受控仿真(§4.2 toy 闭式复现 conflict)、真实模型分析(§4.2 Fig 6 累积质量上升更快=lock 侧;升温后存活续写更多更均=fork 侧)、理论(附录 B)。§C.3 跨域(数学/通用代码)30B 模型基本稳定。baseline 公平(用各模型官方推荐采样作 base)。
- 假设与失效边界：【原文】§1 机制在代码域验证(任务结构让机制可见);§C.3 跨域仅 30B"broadly stable"。【推断】"代码任务结构让机制可见"反过来意味着该机制在非代码域是否成立未定;训练用未验证输出对"错误解占比 vs 最终增益"无定量关系(靠"低温采样的解平均质量较高 + SFT 对高频正确模式更敏感"间接解释);长程多轮自蒸馏是否崩溃(类似熵最小化)未测;Ttrain/ρ/Teval 三超参选取对结果影响大。
- 祛魅总结：【推断】真贡献是 precision-exploration conflict 这一**清晰且可证的机制解释** + 极简方法 + 三路证据(仿真/真实/理论)互证 + hard pass@5 大涨这一难造假的旁证。**包装处**："embarrassingly simple"叙事淡化了 Ttrain/ρ/Teval 三超参的调参成本(§3.4 grid search 本身不简单);"无任何外部信号自提升"成立但机制限于代码域。作者**可能高估**了跨域普适性,**诚实地标注**了"为何用未验证输出仍提升"的解释偏间接。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=模型自身在 Ttrain≠1+截断下的原始未验证输出(无正确性信号) | **改什么**=模型参数(通过 SFT 重塑 token 分布:support compression + within-support reshaping) | **何时改**=离线(从冻结 base 一次性采样→SFT,纯 off-policy,采样分布不随训练更新) | **免梯度?**=否(标准交叉熵 SFT) | **记忆-技能生命周期**=不适用(无记忆/技能库) | **防遗忘机制**=无显式;§C.3 报跨域基本稳定是观察。
- ⑦ 开源代码+框架/harness：https://github.com/apple/ml-ssd （约 0.85MB，社区关注高）。**可得性部分**：含 `data_generation/`(采样流水线)+ `evaluation/`(LCB 评测工具);**SFT 训练本体不在仓库**——标准交叉熵,依赖 **Megatron-LM**(§3.1 脚注 NVIDIA/Megatron-LM)外部框架,README 注明"复现"。框架=采样(vLLM,128K 上下文)+ 标准 SFT(Megatron-LM,8×B200,EP=8 for MoE)+ 评估单独调温;无 RL/verifier/teacher/执行环境。预训练模型(4B/30B)在 HuggingFace。
- 💰 资源/成本与可扩展性：【原文】§3.1——训练 8×B200 GPU,AdamW cosine(peak LR 5e-6),global batch 32,seq len 65536,instruct 2500 iters/thinking 300 iters;采样 vLLM 128K。【推断】~10K prompt × N=1 采样成本低;主要成本在 30B MoE 的 SFT;无需 reward/执行环境,流水线比 RLVR 轻。
- 🎯 对"探索-巩固"对标：**对照/边界案例(而非直接竞品)**——SSD 是**纯 off-policy 自蒸馏**,与本项目 OPD"student 全程 on-policy 自选"范式相反(Table 1 把 SSD 与 On-Policy Distillation 并列对比),正好作 OPD 的**消融对照锚点**:它证明"即便不在线 on-policy、不用 teacher,只靠重塑自身分布也能提升"。**可借组件**:① **precision-exploration conflict / fork-lock 框架**——"fork=该探索的岔路口、lock=该精确锁定的关键步"与 TSRD"探索/选路 vs 巩固/回轨"在 token 几何上高度同构,可作 TSRD 选"在哪个 token 接管/前瞻"的理论语言;② support compression + within-support reshaping 可类比"巩固=压住错误分支、保留有用多样性"。**缺口**:无 teacher 脚手架、无 path-recovery、无 on-policy 自选、无 MTP 前瞻;增益来自分布重塑而非主动选路。一句判定：**对照价值 > 直接借鉴;其 fork/lock 机制语言可为'探索-巩固在 token 级如何定位'提供理论框架,但方法范式(off-policy 自蒸馏)是 OPD 的对立面**。依据：student 不在线自选、无 teacher、纯 SFT。
- 🔭 开放问题/未来方向：【原文】§1、§5——SSD 作为 RL/teacher 蒸馏之外的互补后训练方向;现有代码模型含固定解码未实现的能力。【推断】把 fork/lock 框架迁移到推理(数学)域验证;长程多轮自蒸馏的崩溃边界;与 on-policy/RLVR 结合(SSD 重塑分布作 RL 的更好起点);用 fork/lock 信号指导 TSRD 的"接管点/前瞻点"选择;三温度超参的自适应。

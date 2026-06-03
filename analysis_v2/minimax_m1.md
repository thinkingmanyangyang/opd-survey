minimax_m1 | MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention | MiniMax (MiniMax-AI) | 2025-06-16(arXiv 2506.13585, v1);技术报告(开源权重发布) | 主题线 L3 RLVR/GRPO(算法 CISPO)+ L6 长 CoT/前瞻·相关性 高

**原始论文**:https://arxiv.org/abs/2506.13585

## 一眼看懂
- 🟦 TL;DR:首个开源权重的**大规模混合注意力(MoE + lightning attention)推理模型**(456B 总参/45.9B 激活,原生 1M 上下文,生成 10 万 token 时 FLOPs 仅 DeepSeek-R1 的约 25%)。核心算法 **CISPO**:不像 PPO/GRPO 那样裁剪 token 更新,而是**裁剪重要性采样(IS)权重**,从而**保留所有 token(尤其低概率的"反思/分叉"token)的梯度贡献**,对 DAPO 实现约 2× 加速。全量 RL 在 512×H800 上 3 周完成,约 $53.5 万。【原文 Abstract, §3.1】
- 最巧的一步:**CISPO 把 clip 从"token 更新"移到"IS 权重"**(式4-5)。抽掉它(回到 PPO/GRPO 的 token clip),那些第一次 on-policy 更新后 IS 比值很大的低概率反思 token(However/Wait/Recheck/Aha——推理路径的"岔口")就会被裁掉、在后续 off-policy 更新中拿不到梯度,长 CoT 行为难以涌现(尤其在该混合架构 + 16 轮 off-policy 更新下)。

## 为什么做
- 研究背景:大推理模型(o1/R1)靠 RL 拉长 CoT 提升,但 softmax 注意力的二次复杂度让"持续加长推理"很贵;线性/稀疏注意力等虽多,但**几乎没在大规模推理模型上验证过**(唯一例外 Hunyuan-T1 用 Mamba 但闭源)。MiniMax 要开源一个能高效扩 test-time compute 的大推理模型。【原文 §1】
- 解决的具体痛点:(1) 架构侧——长生成下传统注意力 FLOPs 爆炸;(2) 算法侧——GRPO/DAPO 的 token clip 会**裁掉对稳定熵/可扩展 RL 至关重要的低概率 token**,在混合架构 + 多轮 off-policy 更新下尤其严重;(3) 工程侧——混合架构 RL 训练有训练/推理精度失配等独有坑。【原文 §3.1, §3.2】
- 相关工作 & 各自不足:GRPO(去 critic、组内归一优势)与 DAPO(提高 clip 上界 Clip-Higher)——都仍在 token 层 clip;DAPO 的提高上界在"16 轮 off-policy 更新"设定下不够用。CISPO 直接不丢 token。【原文 §3.1】
- 动机链:要高效长 CoT → 选混合注意力(省 FLOPs)→ 但该架构 RL 不稳 + token clip 丢关键反思 token → 提 CISPO(clip IS 权重、保全 token 梯度)+ 修精度失配/优化器/重复截断 → 配多域数据课程做大规模 RL。
- 与最近邻工作的Δ:相比 DAPO 在 token clip 框架内"放宽上界",CISPO 的 Δ 是**彻底改变 clip 的作用对象(IS 权重而非 token 更新)**,在保证不丢任何 token 梯度的同时仍约束方差;并首次在混合 MoE+lightning attention 大模型上跑通大规模 RL。

## 怎么做 + 靠不靠谱
- 方法流水线:① 持续预训练 MiniMax-Text-01 再 +7.5T token(STEM/代码/推理占 70%,四阶段把上下文扩到 1M) → ② 冷启动 SFT 注入反思型 CoT(数学/代码占 ~60%) → ③ **CISPO 大规模 RL**:多域数据(数学 50K、逻辑 SynLogic 53K、竞赛代码 30K、SWE 沙盒、通用域用 GenRM)+ 课程(先 rule-based 可验证、再混通用域,防灾难性遗忘)→ ④ 分阶段把生成长度 40K→80K(窗口逐级扩 48/56/64/72/80K)。【原文 §2-§5】
- CISPO 子机制(§3.1):基于带停梯度 IS 修正的 REINFORCE,`J=E[ (1/Σ|o_i|) Σ_i Σ_t sg(r̂_{i,t})·Â_{i,t}·log π_θ(o_{i,t}) ]`,其中 `r̂=clip(r, 1-ε_low^IS, 1+ε_high^IS)`;实操**不设 IS 下界、只调上界 ε_high^IS**;优势用 GRPO 组内归一;无 KL 惩罚。还给"统一公式"(式6-7):把 PPO 的 trust-region 写成 token mask,CISPO/GRPO/DAPO 都是它的特例。
- 逐组件必要性:
  - **CISPO(核心)**:消融对比(Fig.2,Qwen2.5-32B-base/AIME2024):同步数下 CISPO > DAPO > GRPO,且 **CISPO 用 50% 步数即追平 DAPO**(2× 加速)。
  - **LM head 升 FP32**(修训练/推理概率失配):把相关性从 ~0.9→0.99,**否则 reward 根本涨不起来**(Fig.3)——小 dense softmax 模型不出现,混合架构才有。
  - **AdamW 超参(β2=0.95, eps=1e-15)**:该模型梯度幅度跨 1e-18~1e-5、相邻迭代相关弱,默认配置会不收敛。
  - **重复检测早停**(连续 3000 token 概率>0.99 即截断):防病态重复的大梯度毁稳定。
  - **长度扩展三招**(重复早停 + sample-level/token-level 混合归一 + 降 clip 阈值与 ε_high^IS):修长生成下"负样本比正样本长得快→后段堆积负梯度→模式塌缩/乱码"。
  - 消融较充分(CISPO vs DAPO/GRPO、精度修复前后),但多为系统级而非逐项理论拆解。
- 关键机制/公式(直觉):token clip 的问题是"低概率但重要的 token 一旦 IS 比值超界就被整条裁掉、彻底无梯度";CISPO 改为只把**IS 权重**夹到 [.,1+ε_high],梯度方向仍朝该 token,**幅度受限但不归零**——所以反思/分叉 token 始终参与学习,熵更稳。代价:梯度有轻微偏置(作者承认),但换来全 token 贡献 + 降方差。
- 实验与证据:MiniMax-M1-40k/80k。Table 2:AIME2024 86.0、SWE-bench Verified 56.0、TAU-bench(airline)62.0(超 Gemini-2.5 Pro)、长上下文 MRCR/LongBench 多项超 o3/Claude4。整体超原版 DeepSeek-R1、Qwen3-235B;对 R1-0528 在数学/代码竞赛略逊但 agent/长上下文更强。CISPO 的核心消融在 Qwen2.5-32B 受控实验(Fig.2),公平。**"看着强但没回答核心问题"**:CISPO 的优势主要在受控小实验给出,大模型上 CISPO 与架构/数据/工程修复深度耦合,难单独归因 CISPO 贡献多少。
- 假设与失效边界:【原文】精度失配/优化器敏感等问题是混合 lightning-attention 架构特有,dense softmax 小模型不出现 → 这些 recipe 对标准架构未必必要。【推断】CISPO 的"保全 token 梯度"在**强反思先验 + 长 CoT** 设定下收益最大;短输出/无明显反思 token 的任务收益可能不明显;结论建立在超大规模工业算力上,小规模复现不一定重现 2× 加速;轻微梯度偏置的长期影响未充分分析。
- 祛魅总结:【推断】真贡献=**CISPO(clip IS 权重以保全所有 token 尤其低概率反思 token 的梯度)**——一个干净、可移植到任意策略梯度框架的算法点子,且坐实"低概率分叉 token 对长 CoT 涌现的重要性";叠加首个开源大混合注意力推理模型的工程闭环。被高估处:很多增益来自架构/长上下文/数据课程,CISPO 单独贡献被打包难拆;$53.5 万、512 H800 的门槛使"高效"是相对而非平民可及。被低估处:CISPO 的"用 sg(IS) 当系数、保全梯度"形式与本课题关注的"forward-hard/backward-soft 解耦"思想同构(见对标)。

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号 = 可验证奖励(数学/代码/逻辑 rule-based)+ GenRM(通用域);CISPO 改变的是**怎么用**这些奖励的梯度(保全低概率 token)。
  - 改什么 = 策略参数(MoE 全参 RL)。
  - 何时改 = on-policy RL 训练期,16 轮 off-policy 更新/批;课程式从可验证域逐步混入通用域。
  - 免梯度? = 否(策略梯度 + IS 权重裁剪)。
  - 记忆-技能生命周期 = 无外部记忆/技能库;能力固化进参数;靠课程 + 通用域混入防遗忘。
  - 防遗忘机制 = 数据课程(先可验证后通用,§4.3 明说"防灾难性遗忘");无显式蒸馏/正则项。
- ⑦ 开源代码+框架/harness:https://github.com/MiniMax-AI/MiniMax-M1(model-release/权重仓,**CISPO 训练代码不在仓内**,本地未 clone,Tier B paper-only;推理支持 vLLM/Transformers)。框架=**自研 RL 系统**(借 lightning attention 高效 rollout;提到 VeRL 的默认优化器配置不适用,即知其参照过 veRL)。【原文 §3.2 提 VeRL 默认配置 + 元信息】
- 💰 资源/成本与可扩展性:**全量 RL = 512×H800 × 3 周 ≈ $53.5 万**(原文明确给出);推理 FLOPs 在 100K 生成长度时 ≈ R1 的 25%(架构优势)。【原文 Abstract, §1, §3】
- 🎯 对"探索-巩固"对标:**中等支撑(探索侧机制 + 思想同构),非巩固/蒸馏直接竞品**。CISPO 对应"探索/选路"侧:**保护低概率分叉 token = 不过早扼杀'可能走通的另一条开头'**,与本课题"探索=发现有效路径、偏向自己能走通的开头"高度契合;Clip-Higher/熵稳定同理。**思想同构**:CISPO 用 `sg(IS权重)` 当系数、对 log π 求梯度——保留方向、限制幅度、不归零,正是 MEMORY 里"forward-hard/backward-soft 解耦(前向保留硬操作、反向流有界平滑梯度)"的一个实例,可直接借鉴到 MTP/OPD 的梯度设计。**缺口/区别**:它是纯 RLVR(无 teacher 蒸馏、无 path-recovery、无 MTP),且"保全 token"是全局均匀策略而非针对关键分叉步定位。一句判定:CISPO 是"探索侧不丢分叉 token"的优秀算法原件 + forward-hard/backward-soft 的现成范例,但缺巩固/teacher 脚手架/前瞻三件套。
- 🔭 开放问题/未来方向:【原文】§9 展望:更合适的 loss/优化算法、用模型自身推理轨迹 bootstrapping、扩到下一量级算力、tool-use/多模态/agent。【推断】(1) 把"保全低概率 token"从全局升级为**关键分叉步定向保全**(配高熵检测),省算力且贴 path-recovery;(2) 把 CISPO 的 sg(IS)-加权梯度形式迁移到 on-policy 蒸馏(teacher 概率比当系数),统一 RLVR 与 OPD;(3) 用 MTP 前瞻预测"该低概率 token 是否真是有效分叉"以决定保不保;(4) 轻微梯度偏置的理论刻画。

— 残留待核:0

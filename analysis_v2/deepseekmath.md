deepseekmath | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO 起源) | DeepSeek-AI(Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Daya Guo 等) | arXiv 2402.03300(v3 2024-04-27)·2024-02 首发·模型报告 | 主题线 L3(RLVR/GRPO 算法基石)·相关性 高

**原始论文**:https://arxiv.org/abs/2402.03300

## 一眼看懂
- 🟦 TL;DR:首次提出 **GRPO**——用"同题多输出的组内相对奖励"替代 PPO 的价值网络(critic),省去与策略同规模的 critic、大幅降显存算力;并给出一个统一梯度范式把 SFT/RFT/Online-RFT/DPO/PPO/GRPO 归一(差异落在数据源/奖励函数/梯度系数三点)。DeepSeekMath-RL 7B 在 GSM8K=88.2%、MATH=51.7%(纯 CoT、无工具无投票),且仅用 GSM8K/MATH 的 CoT 数据训练却在所有(含 OOD)基准超 DeepSeekMath-Instruct 7B【原文 Abstract, §4, §5.2】。
- 最巧的一步:**用组内多采样的均值/标准差归一作 advantage baseline**(取代价值网络)。抽掉它就退回 PPO,需要训练一个与策略同规模的 critic(显存/算力翻倍),GRPO 的"高效"卖点垮掉。为什么:奖励模型本质衡量的就是"同题不同答案孰优孰劣",所以组内相对奖励天然契合奖励模型的比较性质,可直接充当无偏 baseline,根本不需要再训独立价值网络【原文 §4.1.1】。

## 为什么做
- 研究背景:RL 已被证明能在 SFT 之后进一步提升 LLM 数学推理,该阶段广泛用 PPO(actor-critic);本文在大规模数学语料预训练(120B math tokens,DeepSeekMath-Base/Instruct 7B)基础上引入新 RL 算法【原文 §1, §4】。
- 解决的具体痛点:①PPO 价值模型开销大(需训一个与策略同规模的 critic,巨大显存算力);②稀疏奖励下 critic 难训(LLM 场景通常只在最后一个 token 由奖励模型打分,逐 token 精确价值函数难学);③缺乏对各类后训练方法(RFT/DPO/PPO/GRPO)的统一理论理解框架【原文 §4.1.1, §5.2】。
- 相关工作 & 各自不足:PPO(Schulman 2017)是 RLHF 主流但 critic 昂贵;RFT/DPO 等各成一派,缺统一视角;过程奖励(Wang 2023b/Lightman 2023 PRM800K)信号细但标注含约 20% 错误【原文 §4, §5.2.3】。
- 动机链:现状(SFT 后用 PPO 做 RL 提升数学)→缺陷(critic 贵、稀疏奖励难训、方法零散无统一理解)→所以必须(用组内相对奖励省掉 critic=GRPO;并提供统一范式分析在线/离线、结果/过程监督、单轮/迭代,理解 RL 为何有效)【原文 §4.1, §5.2】。
- 与最近邻工作的Δ:相对 PPO,差在"去掉价值网络、用组内 baseline + KL 直接加损失(而非加奖励)"。相对 RFT/Online-RFT,差在"GRPO 按奖励值差异化调整正负梯度系数(Online RFT 不惩罚错误、对所有正确响应等强度强化)"——这是 §5.2 实证 GRPO>Online RFT 的根因。有用点:GRPO 成为 R1、DAPO 及绝大多数 RLVR 工作的算法基石,被 veRL/TRL/OpenRLHF/ms-swift 广泛实现。

## 怎么做 + 靠不靠谱
- 方法流水线:输入 DeepSeekMath-Instruct 7B + 约 144K GSM8K/MATH CoT 题 → ①每题从旧策略采 G=64 个输出 → ②奖励模型(基于 Base 7B 训,lr 2e-5)打分 → ③组内 reward 均值/标准差归一得 advantage(无 critic)→ ④GRPO clip 代理目标 + KL 直接加损失(用 Schulman 无偏估计器)更新策略 → 输出 DeepSeekMath-RL 7B。可选过程监督(每步打分)/迭代 RL(随训练用策略采样重训奖励模型,replay 含 10% 历史)【原文 §4.1, Algorithm 1】。
- 逐组件必要性(§5.2 有系统对照):
  - **组内 baseline(去 critic)**:核心,GRPO 与 PPO 比省 critic 仍达 SOTA【原文 §4.1.1, §4.2】。
  - **在线 vs 离线**:有对照(Fig.5)。Online RFT 显著超 RFT(早期相近、后期拉开),证明在线采样优越——后期 actor 与 SFT 模型差异大,实时采样更有利【原文 §5.2.1】。
  - **差异化梯度系数**:有对照(Fig.5)。GRPO 超 Online RFT,因 GRPO 按奖励值差异化正负梯度(Online RFT 不惩罚错误、等强度强化所有正确)【原文 §5.2.1】。
  - **过程监督 vs 结果监督**:有对照(Fig.5)。GRPO+PS 超 GRPO+OS,证明细粒度 step-aware 梯度系数有益【原文 §5.2.1】。
  - **迭代 RL**:有对照(Fig.6)。两轮迭代显著提升(尤其第一轮)【原文 §4.1.4, §5.2.1】。
- 关键机制/公式(直觉):①GRPO 目标(Eq.3/21):对每题采 {o_1..o_G},逐 token 重要性比 × clip,优势 Â_{i,t} 由组内相对奖励估计(无价值函数);KL 直接加损失用无偏估计器(Eq.4)π_ref/π−log(π_ref/π)−1,保证非负、避免复杂化 Â;②结果监督(§4.1.2):整条输出所有 token 优势 = 组内标准化奖励 r̃_i;③过程监督(§4.1.3):每 token 优势 = 其后续各步标准化奖励之和 Â_{i,t}=Σ_{index(j)≥t} r̃_i^{index(j)};④统一范式(§5.2.3,Eq.5):所有方法 = 数据源 × 奖励函数 × 梯度系数 GC 的组合,差异仅在这三点【原文 §4.1, §5.2.3】。
- 实验与证据:RL 基于 DeepSeekMath-Instruct 7B;RL 数据 约 144K GSM8K/MATH CoT 题(故意排除其它 SFT 题以观察 RL 对缺数据基准的影响);奖励模型基于 Base 7B(lr 2e-5);GRPO 超参:策略 lr 1e-6、KL β=0.04、每题 G=64、max length 1024、batch 1024、每探索阶段策略仅更新一次。结果(Table 5):RL 7B GSM8K 88.2%、MATH 51.7%(CoT),超 7B-70B 全部开源 + 多数闭源;仅训 GSM8K/MATH CoT 却在所有(含 OOD 如 MGSM-zh/CMATH/工具集成)基准超 Instruct 7B【原文 §4.2】。baseline 公平性:GRPO/Online-RFT/RFT/PPO 在同 base 同数据对照(Fig.5),公平;**关键去魅证据(§5.2.2 Fig.7)**:RL 提升 Maj@K 但**不提升 Pass@K**——说明 RL 主要是让输出分布更鲁棒(把正确答案从 TopK 顶上来),而**非提升根本能力**;作者自承可能因只用 SFT 阶段题 + 朴素 nucleus 采样【原文 §5.2.2】。
- 假设与失效边界:【原文】"所有方法 fully TRUST 奖励信号"——但奖励不总可靠(连 PRM800K 都约 20% 标错),复杂任务下会失效,故提出需 weak-to-strong/抗噪奖励算法(§5.2.3);"RL 仅稳定分布而非提升根本能力"在奖励模型不能泛化到 OOD 时尤甚。【推断】结论集中在数学单域、7B 单规模;G=64、max length 1024 的设定在更长 CoT 下不适用(后续 R1/DAPO 用 32768+);"在线优于离线""GRPO+PS 更优"的可推广性需外部验证(事实上 DAPO 后来指出朴素 GRPO 的熵坍缩问题)。
- 祛魅总结:真贡献=GRPO(省 critic 的高效 RLVR 算法,成为整个领域基石)+ 统一梯度范式(数据源/奖励/梯度系数三分解,解释力强)+ 罕见的诚实去魅(明说 RL 只升 Maj@K 不升 Pass@K=分布锐化非能力提升)。包装/高估处:GRPO 被后续奉为"标准"但本文只在数学 7B 验证,熵坍缩等问题留给 DAPO 等修补;过程监督/迭代 RL 依赖额外过程/迭代奖励模型,工程成本与稳定性本文未深入。注:此模型报告的主要篇幅其实是预训练语料工程(120B math corpus、code training 益处、arXiv 无用),GRPO 只是后半【推断,据全文结构】。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=奖励模型/规则对完整输出(结果监督)或每推理步(过程监督)的标量奖励,组内归一 | **改什么**=策略全参数(on-policy 梯度)| **何时改**=RL 训练期,每探索阶段策略更新一次;可迭代(同时重训奖励模型)| **免梯度?**=否(梯度 RL)| **记忆-技能生命周期**=不涉及显式记忆/技能库,数学推理能力固化进权重 | **防遗忘机制**=GRPO KL 惩罚(β=0.04,直接加损失约束 ref);无专门防遗忘【原文 §4.1】。
- ⑦ 开源代码+框架/harness:https://github.com/deepseek-ai/DeepSeek-Math;模型权重 DeepSeekMath-Base/Instruct/RL 7B(HuggingFace)。**仓库主要为评测/推理脚本与模型发布,GRPO 训练代码不在仓内**(实际 RL 为 DeepSeek 内部代码)。框架 custom/none。GRPO 算法后被 veRL/TRL/OpenRLHF/ms-swift 广泛实现。【训练码未开源,复现依赖论文公式 + 第三方框架】
- 💰 资源/成本与可扩展性:论文未给 RL 阶段的 GPU 卡时(原文未说明);GRPO 的核心卖点正是降成本——省掉与策略同规模的价值网络,显著优化 PPO 的显存占用(Abstract"optimizing the memory usage of PPO")。预训练 120B math tokens(独立大成本,与 RL 分离)【原文 Abstract】。
- 🎯 对"探索-巩固"对标:**支撑(探索侧算法底座) + 一个关键警示** —— 一句判定:GRPO 是本项目几乎所有 on-policy RL/蒸馏实验要用的算法地基(探索=策略自采样多输出、按相对奖励强化走得通的路径),其统一范式(数据源/奖励/梯度系数)正好提供了思考"稀疏脚手架如何改梯度系数"的语言;依据:§4.1 组内相对优势 + §5.2.3 梯度系数分解。**关键警示(对本项目最有价值)**:§5.2.2 实证 RL 只升 Maj@K 不升 Pass@K——若本项目的"探索-巩固"也只是把已有正确路径顶上来而非真正扩展能力边界,就会落入同样陷阱,这正是 denoiserl 等"注入新状态/负样本"工作和本项目"teacher 脚手架扩探索空间"想要突破的点。可借组件:①过程监督的 step-aware 优势(每 token = 后续步奖励和)可迁移为"路径恢复/巩固的逐步信用分配";②迭代 RL(策略进化→奖励模型同步进化)对"持续学习不遗忘"有结构启发。缺口:GRPO 本身无 teacher 脚手架、无路径恢复、无 MTP、无记忆/技能库,纯结果/过程奖励驱动。
- 🔭 开放问题/未来方向:【原文】§5.2.3 三方向——数据源(OOD prompt + tree-search 等高级采样以突破 Pass@K 瓶颈)、算法(抗噪奖励、weak-to-strong 对齐)、奖励函数(泛化性、不确定性建模、高质量过程奖励模型);否则"RL 仅稳定分布而非提升根本能力"。【推断】更大规模/更长 CoT 下的熵坍缩(后被 DAPO 揭示)、把 GRPO 与 on-policy 蒸馏/路径恢复结合、用前瞻(MTP)信号改进探索效率均未涉及。

---
RETURN: deepseekmath | 读到PDF? 是(读GRPO §4 + 统一范式 §5.2,正文.txt) | L3(GRPO 算法基石) | 探索侧算法底座(本项目 RL/蒸馏实验地基);§5.2.2"RL只升Maj@K不升Pass@K"是对"探索-巩固是否真扩能力边界"的关键警示;无teacher脚手架/路径恢复/MTP | 残留待核 0(为模型报告,GRPO 训练代码未开源)

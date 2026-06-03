dapo | DAPO: An Open-Source LLM Reinforcement Learning System at Scale | ByteDance Seed + 清华 AIR(SIA-Lab) + 港大;通讯 Hao Zhou、Mingxuan Wang | arXiv 2503.14476(v1 2025-03-17，v2 2025-05-20)·技术报告 | 主题线 L3(RLVR/GRPO)·相关性 高

**原始论文**:https://arxiv.org/abs/2503.14476

## 一眼看懂
- 🟦 TL;DR:朴素 GRPO 在 Qwen2.5-32B base 上跑数学 RL 只到 30 分(AIME 2024),作者诊断出 4 个病灶——熵坍缩、无效梯度、长序列梯度失衡、超长样本奖励噪声——逐一开药(Clip-Higher / Dynamic Sampling / Token-Level Loss / Overlong Reward Shaping),组成 DAPO,把 AIME 提到 50 分且只用一半训练步就超过 DeepSeek-R1-Zero-Qwen-32B(47),并完整开源算法+代码(基于 verl)+数据【原文 Abstract, Fig.1, §1】。
- 最巧的一步:**Dynamic Sampling**(动态采样)。看 Table 1 消融,前面四项技术叠加只到 42,最后加上 Dynamic Sampling 一步直接跳到 50——这一步贡献最大(+8 分),抽掉它整个系统从 SOTA 掉回平庸【原文 §4.2 Table 1】。为什么:它过滤掉组内"全对/全错"(advantage=0、零梯度)的 prompt,保证每个 batch 都是有效梯度,避免随训练推进有效 prompt 数持续缩水(Fig.3b 显示 acc=1 的样本比例一路上升)【原文 §3.2】。

## 为什么做
- 研究背景:test-time scaling(o1/R1)靠长 CoT + 大规模 RL 撑起推理能力革命,RL 是核心引擎;但 SOTA 模型(o1、R1)的算法与配方细节被技术报告藏起来,社区难复现工业级结果【原文 §1】。
- 解决的具体痛点:作者自己在 Qwen2.5-32B base 上跑朴素 GRPO 只得 30 分(远低于 DeepSeek 报告的 47),诊断出朴素 GRPO 有熵坍缩、reward noise、训练不稳定三类问题;且 R1 论文省略了构建可复现大规模 RL 系统所需的工程细节【原文 §1】。
- 相关工作 & 各自不足:PPO(§2.1)需价值网络估 advantage,开销大;GRPO(§2.2,源自 DeepSeekMath)去掉 critic、用组内相对奖励,但在长 CoT 大规模场景暴露上述 4 病灶;RLHF 范式保留 KL 约束以不偏离初始模型,但长 CoT RL 下模型分布本就要大幅偏移,KL 反而是累赘(§2.3 故去掉);用奖励模型易 reward hacking,故改用规则二值奖励(§2.4)【原文 §2.1-2.4】。
- 动机链:现状(长 CoT RL 是提升推理的核心但闭源)→缺陷(朴素 GRPO 复现只到 30 分且不稳定,工程细节缺失)→所以必须(完全开源一套达 SOTA 的系统:算法+代码+数据,把 4 个关键技术公开)【原文 §1】。
- 与最近邻工作的Δ:相对 GRPO/R1-Zero,差在"把朴素 GRPO 的隐性失败点显式拆成 4 个可独立验证的工程改造并逐项消融",且彻底开源(R1 只放权重不放训练码)。关键有用点:这 4 项是即插即用的解耦补丁,后续被 verl 收为官方 recipe、被社区广泛复用。

## 怎么做 + 靠不靠谱
- 方法流水线:输入 Qwen2.5-32B base + DAPO-Math-17K → ①去 KL + 规则二值奖励(对+1/错−1)的极简 GRPO 底座 → ②叠加 4 改造(Clip-Higher 解耦上下裁剪 0.2/0.28;Dynamic Sampling 过滤全对全错 prompt 直到 batch 填满有效样本;Token-Level Loss 把 sample-level 损失改 token 级聚合;Overlong Reward Shaping = Overlong Filtering 屏蔽截断样本 loss + Soft Overlong Punishment 长度软惩罚)→ 输出在 AIME 2024 达 50 分的推理模型【原文 §3.1-3.4, Algorithm 1, §4.1】。
- 逐组件必要性(均有 Table 1 累积消融):
  - **Clip-Higher**(+2,36→38):负责给低概率"探索"token 留上行空间。证据:实测被上裁剪 token 概率<0.2(Fig.3a),解耦后熵回升(Fig.2b)。没它→熵坍缩、采样近乎同质【原文 §3.1, Fig.2/3a】。
  - **Dynamic Sampling**(+8,42→50,**最大单项**):保证 batch 全为有效梯度。没它→有效 prompt 数随训练缩水、梯度方差变大。注:作者特别澄清它不显著拖慢训练(长尾样本本就主导生成时间,且收敛更快,Fig.6)【原文 §3.2, §4.2, Fig.6】。
  - **Token-Level Loss**(+1,41→42,贡献最小):让长序列对梯度有合理贡献、抑制长样本中的乱码/重复。作者明说"性能增益较小但提升训练稳定性、让长度更健康增长"【原文 §3.3, §4.2, Fig.4】。
  - **Overlong Reward Shaping**(Overlong Filtering +6,30→36;Soft Punishment +3,38→41):降低截断样本带来的 reward 噪声。证据 Fig.5(加 filtering 后精度/熵更稳)【原文 §3.4, Fig.5】。
- 关键机制/公式(直觉):①裁剪上界对"已经高概率的 exploitation token"几乎不设限,却把"低概率 exploration token"卡死——解耦 ε_high 放宽上界即解锁探索;②组内全对/全错→advantage 归一后全为 0→零梯度,故必须过滤;③sample-level 归一让长序列每 token 权重被稀释,token 级聚合(对所有 token 求和再按总 token 数归一,Eq.12)纠正之;④截断样本若一律重罚,会把"推理正确只是太长"的样本错误惩罚,引入噪声,故先屏蔽再软惩罚【原文 §3.1-3.4】。
- 实验与证据:基座 Qwen2.5-32B base;baseline 为朴素 GRPO+组奖励归一。超参:AdamW、常数 lr 1e-6、20 rollout step 线性 warm-up;prompt batch 512、每 prompt 16 响应;train mini-batch 512(每 rollout 16 次梯度更新);期望最大长 16384 + 4096 soft punish buffer = 最大生成 20480;评测 temperature 1.0、top-p 0.7,AIME 重复 32 次报 avg@32。主结果:AIME 2024 约 0%→50%,半数步超 R1-Zero-Qwen-32B(47)。Table 1 给 4 技术逐项累积消融【原文 §4.1-4.2】。baseline 公平性:同基座同评测对照 R1-Zero-Qwen-32B,公平;但**主结果集中在单一基座(Qwen2.5-32B)+单一评测(AIME 2024)**,跨基座/跨任务论文内未验证。"看着强但没回答"的点:50 分是否泛化到非数学/更小模型,论文未答(声称"可迁移到其它任务"但未给证据)【推断,据 §4.1 仅做数学】。
- 假设与失效边界:【原文】去 KL 的前提是"长 CoT RL 下模型分布本应大幅偏离初始模型"(§2.3),故在需要贴近初始模型的对齐场景不适用;规则二值奖励要求任务可验证(数学/代码)。【推断】ε_high=0.28 等为经验取值无敏感性分析,换基座/任务可能需重调;在更小模型或更弱 base 上,Clip-Higher 放宽探索是否仍稳定未知。
- 祛魅总结:真贡献=把朴素 GRPO 的失败点工程化拆解为 4 个可复现补丁 + 彻底开源(算法+码+数据),复现价值极高,这是它被社区奉为标准 recipe 的根本原因。包装/高估处="key techniques that make large-scale RL a success"的普适性主要靠社区后续复现背书,论文内只在数学单域+单基座验证;Token-Level Loss 被列为四大技术之一但实测仅 +1 分,贡献被叙事略微抬高【推断,据 Table 1】。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=规则二值结果奖励(答案对+1/错−1,可验证任务)| **改什么**=策略参数(on-policy 梯度更新)| **何时改**=训练期,每 rollout step 16 次梯度更新 | **免梯度?**=否(全程梯度 RL)| **记忆-技能生命周期**=不涉及(无显式记忆/技能库,能力固化进权重)| **防遗忘机制**=去掉 KL 惩罚(反而主动允许偏离),无专门防遗忘设计【原文 §2.3, §4.1】。
- ⑦ 开源代码+框架/harness:https://github.com/BytedTsinghua-SIA/DAPO(recipe/eval/数据说明;**训练逻辑实现在 verl**:https://github.com/volcengine/verl,DAPO 作为 verl 的一个 recipe);框架 **veRL**(requirements 含 vllm==0.8.3、ray[serve])。本地已 clone(~4.1MB)。【原文 Abstract 脚注 a 明确"built on the verl framework"】
- 💰 资源/成本与可扩展性:论文未给出 GPU 卡时/训练成本的明确数字(原文未说明);仅知 rollout prompt batch 512×16 响应、生成上限 20480 token、Dynamic Sampling 会增加采样量但作者称总训练时间不显著增加(Fig.6 收敛更快)【原文 §4.1-4.2】。
- 🎯 对"探索-巩固"对标:**支撑(探索侧)** —— 一句判定:DAPO 的 Clip-Higher 与 Dynamic Sampling 是"如何在 on-policy RL 中保住探索、不让策略过早确定化"的高质量工程方案,与"探索=发现有效行为/路径"的诉求直接同构;依据:Clip-Higher 专门给低概率探索 token 解锁上行空间(§3.1)、Dynamic Sampling 保证有效梯度即保证持续探索信号(§3.2)。但它**不涉及"巩固/回轨"也不涉及 teacher 脚手架**——纯 on-policy 自演化,无错误前缀注入、无路径恢复目标(对比 denoiserl)。可借组件:去 KL + token 级聚合损失 + Soft Overlong Punishment 可直接搬进任何 GRPO 底座;缺口:无 path-recovery/巩固机制、无 MTP 前瞻、无防遗忘。
- 🔭 开放问题/未来方向:【原文】§4.3 讨论训练动态——长度并非单调上升、训练 reward 与验证精度弱相关(过拟合训练集),提示需结合长度+验证精度共同监控;监控熵需维持在合适区间。【推断】跨基座/跨任务泛化、超参敏感性、与过程奖励或路径级信用分配的结合、在更小模型上的可扩展性均未解。

---
RETURN: dapo | 读到PDF? 是(16页全文) | L3(RLVR/GRPO) | 探索侧强支撑(Clip-Higher/Dynamic Sampling 是保探索工程范式),无巩固/回轨/teacher 脚手架,纯 on-policy | 残留待核 0

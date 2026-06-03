# qwen3 — Qwen3 Technical Report

- **arXiv/链接**: arXiv:2505.09388 (2025-05-15)
- **机构/作者**: Qwen Team (Alibaba)
- **发表/时间**: 2025-05
- **主题/相关性**: T1(High)。**与本综述高度相关**:Qwen3 显式提出 Strong-to-Weak Distillation 的两阶段方案,其中第二阶段就是 **On-policy Distillation(学生自采样序列 + 与教师 logits 对齐最小化 KL)**,是大厂技报里少见的明确 on-policy KD 实践;并提供统一 SFT-RL 四阶段后训练管线作为对照。

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/qwen3/fig_02.png)

*Figure 2: Performance of Qwen3-235B-A22B with respect to the thinking budget.*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/qwen3/fig_01.png)

*Figure 1: Post-training pipeline of the Qwen3 series models.*

## 1. 开源代码链接
- 仓库:https://github.com/QwenLM/Qwen3 ;权重 https://huggingface.co/Qwen 、https://modelscope.cn/organization/qwen 。
- **该仓为模型发布/权重仓,不含后训练训练代码**(蒸馏/RL 管线均不开源;社区微调依赖 Axolotl/Unsloth/ms-swift/LLaMA-Factory)。CloneTier=B,仅记录不 clone。

## 2. 使用框架
- Reasoning RL / General RL:**GRPO**(引用 Shao et al. 2024);提到大 batch、多 rollout、off-policy 提升样本效率、通过控制熵稳定训练。报告未提 GSPO(GSPO 为后续单独工作)。
- Strong-to-Weak Distillation:**离线(off-policy)response distillation + 在线(on-policy)logit/KL 蒸馏**。
- **开源仓不含上述训练代码**。

## 3. 研究背景
- Qwen3 系列含 dense 与 MoE,参数 0.6B–235B(旗舰 Qwen3-235B-A22B)。核心创新:把 thinking(复杂多步推理)与 non-thinking(快速响应)统一进单一模型,支持动态模式切换与 thinking budget(推理 token 预算控制)。

## 4. 当前存在的问题
- 过去需为 chat 模型(如 GPT-4o)与专用推理模型(如 QwQ-32B)分别部署/训练;
- 为每个小模型独立跑完整四阶段后训练计算开销大、开发成本高。

## 5. Motivation
- 用统一框架免去多模型切换;用 thinking budget 自适应分配推理算力;
- 用旗舰大模型的知识(strong-to-weak 蒸馏)大幅降低轻量模型后训练成本——报告称蒸馏仅需四阶段训练约 **1/10 GPU 时**,且 Pass@1(即时性能)与 Pass@64(探索能力)均更优。

## 6. 主要方法
**旗舰模型四阶段后训练管线(图 1)**:
1. **Long-CoT Cold Start**:精选数学/代码/逻辑/STEM 可验证问题,用 QwQ-32B 生成 N 个候选并严格过滤,做少量 SFT,只植入基础推理模式、不追求即时性能(刻意少样本少步数,保留 RL 提升空间)。
2. **Reasoning RL**:筛 3,995 条 query-verifier 对(未用于 cold-start、对 cold-start 模型可学、尽量难、覆盖广),用 **GRPO** 训练;大 batch + 多 rollout + off-policy + 控熵稳定;Qwen3-235B-A22B 在 170 步内 AIME'24 由 70.1→85.1。
3. **Thinking Mode Fusion**:在 Reasoning RL 模型上做持续 SFT,融合 non-thinking 能力;thinking 数据由 Stage-2 模型对 Stage-1 query 拒绝采样生成,non-thinking 数据覆盖代码/数学/指令/多语/创作/QA/角色扮演;设计 `/think` `/no_think` chat 模板与空 think 块;thinking budget(达阈值插入停思指令)为自然涌现能力。
4. **General RL**:覆盖 20+ 任务的奖励系统(指令遵循、格式遵循、偏好对齐、Agent 工具调用含真实环境多轮反馈、RAG 等);三类奖励:规则奖励、带参考答案的模型奖励(Qwen2.5-72B-Instruct 打分)、无参考的偏好奖励模型。

**Strong-to-Weak Distillation(轻量模型:0.6B/1.7B/4B/8B/14B dense + 30B-A3B MoE)**:
- **(1) Off-policy 蒸馏**:用教师在 `/think` 与 `/no_think` 两种模式下的输出做 response 蒸馏,使学生获得基础推理与模式切换能力。
- **(2) On-policy 蒸馏**:**学生自己生成 on-policy 序列**(对采样到的 prompt 在 `/think` 或 `/no_think` 模式下产出响应),再**对齐学生与教师(Qwen3-32B 或 235B-A22B)的 logits、最小化 KL 散度** 进行微调。这是本综述关注的核心 on-policy KD 形式。

## 7. 实验数据集
- 训练:数学/代码/逻辑/STEM 可验证 prompt;cold-start 精选子集;Reasoning RL 3,995 对;General RL 20+ 任务。
- 评测:AIME'24/'25、LiveCodeBench、CodeForces、GPQA、MMLU、多语基准(Multi-IF 8 语、INCLUDE 44 语)等,thinking/non-thinking 双模式评测。

## 8. 怎么做的(训练/数据/流程)
- 数据构建两阶段过滤:query 过滤(Qwen2.5-72B-Instruct 剔除不可验证/无需 CoT 即可答对/多子问题的 query,并标注 domain 平衡)+ response 过滤(剔除错误答案、重复、明显猜测、思考与总结不一致、语言混杂、疑似验证集泄漏)。
- Reasoning RL 单次 RL run 内 reward 与验证性能持续提升,无需人工调超参,靠控熵稳态维持。
- 蒸馏先 off-policy 打底再 on-policy 精炼,显著降本(约 1/10 GPU 时)且兼顾即时性能与探索能力。

glm45 | GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models | Zhipu AI 智谱 & 清华（GLM-4.5 Team）| arXiv 2508.06471v1, 2025-08-08 · 技术报告 | 主题线 L1/L3/L4（自蒸馏统一 + GRPO-no-KL + agent RL）·相关性 高（多专家→统一的 self-distillation 与 RL 间迭代自蒸馏，对标本课题「巩固」）

**原始论文**：https://arxiv.org/abs/2508.06471

## 一眼看懂
- 🟦 TL;DR：开源 MoE（355B 总/32B 激活，另有 106B 的 Air）**混合推理**模型(thinking + direct 双模)。**这是模型技术报告，不是单一方法论文**。与本课题最相关的是其后训练范式——**两阶段**：① Expert Training（分别冷启 SFT + 专家 RL 训出 Reasoning/Agent/General-chat 三个专家）→ ② Unified Training（用 **self-distillation** 把多专家整合进一个通才）；agent RL 中因 RL 昂贵，还在 RL 轮次之间插入 **iterative self-distillation**（用 RL 后模型的输出替换冷启数据再 SFT，抬高起点后继续 RL）【原文 §3, §3.3.2】。
- 最巧的一步：**把「多专家各自 RL 强化 + 再 self-distillation 合并」作为统一通才的生产线**——抽掉 Unified-stage 的 self-distillation，三个专家就无法在不互相打架的前提下融进一个支持双模式的模型。其精髓：**先在各域用 RL 把能力推到边界（探索/强化），再用蒸馏把这些边界能力固化进一套参数（巩固）**；agent 侧的 iterative self-distillation 进一步把「RL 探索的成果」周期性地「固化成新 SFT 起点」，让昂贵的 RL 不必从头慢爬。这正是「探索→巩固」在工业级 pipeline 的体现。

## 为什么做
- 研究背景：LLM 从知识库走向通用问题求解器；目标是单个开源模型同时擅长 **Agentic / Reasoning / Coding (ARC)** 三类互联能力——此前 o1/o3、Claude 在单域强，但开源界缺一个三域全能模型【原文 §1】。
- 解决的具体痛点：(1) 三域能力如何在一个模型里共存且支持「深思 + 快答」双模式;(2) RL 中模型能力随训练演化、与**静态数据失配**——后期太简单(reward 全 1)、早期太难(reward 全 0)，都无 reward 方差→无梯度;(3) 多阶段渐增输出长度的 RL 会让模型在短长度阶段「遗忘」长上下文能力、造成**不可逆**性能下降;(4) agent RL 极耗时;(5) function call 参数含代码时 JSON 转义负担重。
- 相关工作 & 各自不足：对标 o1/o3、Claude Sonnet/Opus 4、DeepSeek-R1、Kimi K2、Qwen3-235B 等;架构借鉴 DeepSeek-V3/Kimi K2 的 MoE 但**减宽增深**(更少 routed experts、更多层)、2.5× 注意力头、QK-Norm、loss-free balance routing、partial RoPE。
- 动机链：要单模型全能 ARC → 分域训专家最易把单域推到极致 → 但要交付一个模型 → 用 self-distillation 把多专家合成通才 → agent RL 太慢 → 在 RL 间迭代自蒸馏抬起点 → 数据/课程/长度上各打补丁(难度课程、单阶段 64K、动态温度)防失配与遗忘。
- 与最近邻工作的Δ：vs **Qwen3/DeepSeek-R1 等同期报告**——同走「专家/分域 + RL」，但 GLM-4.5 把 **self-distillation 作为「多专家→统一双模通才」的核心粘合剂**讲得最显式;且明确主张 **单阶段 64K RL 优于多阶段渐增长度**(反对 Polaris 式做法)、**GRPO 去 KL**、难度课程二阶段只用「已验证正确答案池」。

## 怎么做 + 靠不靠谱
- 方法流水线（后训练）：① **Cold-start SFT**：少量长 CoT 数据给每个专家打底 → ② **Expert RL**：三专家(Reasoning/Agent/General)各自 RL 强化(基于 **GRPO 去 KL 项**)→ ③ **Overall/Unified SFT(self-distillation)**：从各专家采百万级样本(rejection sampling 多级过滤)、128K 上下文训 base，**把多专家能力蒸成一个混合推理通才**(混合「带思考」与「无思考」数据实现双模)→ ④ **agent RL + iterative self-distillation 交替**：RL 抬 agent 性能→plateau 后用 RL 模型输出替换冷启数据做 SFT(自蒸馏)得更强 SFT 模型→再 RL、逐步增难 → ⑤ **General RL**：rule + human(RLHF) + model(RLAIF) 多源反馈 holistic 提升 + pathology RL 专治语言混杂/重复/格式。
- 逐组件必要性（均在**小实验模型**上做受控消融，非 GLM-4.5 本体）：
  - **难度课程二阶段**：图5 证明能持续突破性能天花板；二阶段严格只用「已验证正确答案池」降噪。有消融。
  - **单阶段 64K RL vs 多阶段增长度**：图6 证明直接 64K 持续推进更好;多阶段会「unlearn」长上下文致**不可逆**下降。有消融。**这是「防遗忘」的关键经验**。
  - **token-weighted mean loss（code RL）**：图7 左证明比 sequence-mean 收敛更快、缓解长度偏置、抑制「base case」灌水。有消融。〔注：与 DAPO/HAPO 的 token-mean 取向一致〕。
  - **科学 RL 数据质量**：图7 右证明只用专家验证的高质量选择题 (65.8) 远超混杂数据 (62.9)。有消融。
  - **动态采样温度**：reward 稳定即判定收敛并升温促多样性，配 held-out 验证「升温不掉 1% 以上」的质控。有机制描述。
  - **self-distillation / iterative self-distillation**：**作为核心范式陈述，但本报告未给「有 vs 无自蒸馏」的对照消融数字**——属技术报告，靠最终基准背书。〔待核：self-distillation 相对「直接多任务 SFT/RL」的净增益无独立量化〕。
- 关键机制/公式（直觉）：agent RL 用 group-wise(GRPO 式去 KL)：优势 = r(x,yi) − 组内平均 r̄(x)，**只对模型生成 token 计 loss、环境反馈不计**;agent 用 outcome 监督 + 过程格式惩罚(tool 格式错则 trace 得 0)。iterative self-distillation 的直觉:RL 慢、但 RL 产出的好轨迹可当 SFT 数据「廉价复用」，把 RL 的探索成果周期性固化进参数当新起点。
- 实验与证据：GLM-4.5(355B/32B) 在 12 基准综合排第 3、agentic 第 2;关键数字 **TAU-Bench 70.1 / AIME24 91.0 / SWE-bench Verified 64.2 / BFCL v3 77.8 / GPQA 79.1 / BrowseComp 26.4**;参数远少于多数对手。**注意**：消融曲线都在「smaller experimental model」上做，**不在 GLM-4.5 本体**——方法有效性的微观证据来自代理模型，本体只有最终分数。
- baseline 公平吗：最终基准对比是同期主流模型横评(截至 2025-07-28)，属标准 leaderboard 式;但**方法层消融的 baseline 是自家小模型**，外部可比性弱;无随机种子/方差。
- 假设与失效边界：【原文】(1) 单阶段 64K RL 的前提是「SFT 已把模型 condition 到生成 64K 长响应」——若 SFT 未充分长上下文化则未必;(2) 难度课程二阶段依赖「已验证正确答案池」;(3) 消融在小模型上，承认「曲线基于较小实验模型」。【推断】(4) **self-distillation「整合多专家」可能有能力相消/折中**——把三个域专家压进一个模型，单域峰值能力大概率低于专家本身(报告未量化这层损失，只报通才最终分);(5) iterative self-distillation 反复「RL→蒸回 SFT」可能累积分布偏移或过拟合 RL 偏好，长期稳定性未讨论;(6) 作为**模型发布仓不含后训练训练脚本**，方法细节(λ、课程切点、自蒸馏数据配比)多未公开，复现性受限;(7) 含一层 MTP 层但**仅用于推理时 speculative decoding 加速**，非训练信号——与本课题「MTP 作前瞻训练探针」无直接交集(只是同名机制的不同用途)。
- 祛魅总结：真贡献=**(a) 工业级开源 ARC 全能 MoE + 双模混合推理的完整后训练配方 + (b) 「多专家 RL → self-distillation 统一」与「RL↔iterative self-distillation 交替」两套自蒸馏范式的工程化 + (c) 一批实用经验(单阶段 64K 防遗忘、GRPO 去 KL、难度课程、token-weighted loss、动态温度)**——经验密度高、对工程落地有参考价值。需打折的：(1) **是技术报告非方法论文**——self-distillation 的净增益无独立消融，方法证据多来自小代理模型;(2) 后训练脚本未开源(只发模型权重 + slime 框架)，关键超参缺失;(3) 「多专家→统一」的单域能力损失未量化;(4) MTP 仅加速、与前瞻训练无关，勿误读。【推断】对「探索-巩固」这是**工业级最贴近「巩固」范式的一篇**——但其「巩固」是粗粒度的「整模型蒸馏/重 SFT」，非 token/路径级精细固化。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=多源——专家输出(self-distillation 的 SFT 交叉熵)、可验证 outcome 奖励(reasoning/agent RL，GRPO 去 KL)、RLHF/RLAIF 偏好(general RL)｜**改什么**=策略参数 θ(SFT + RL 交替)｜**何时改**=分阶段(冷启 SFT→专家 RL→统一 SFT→agent RL↔iterative 自蒸馏→general RL)｜**免梯度?**=否，SFT + policy gradient｜**记忆-技能生命周期**=**把多专家技能蒸馏固化进单套参数**(无外部记忆/技能库);iterative self-distillation 把 RL 探索成果周期性固化为新 SFT 起点｜**防遗忘机制**=**显式且关键**——单阶段 64K RL 避免短长度阶段「unlearn」长上下文(图6 不可逆下降的反例);难度课程防 reward 失配;但跨专家融合的遗忘未量化。
- ⑦ 开源代码+框架/harness：模型/权重仓 https://github.com/zai-org/GLM-4.5 （**仅模型发布仓，不含后训练训练脚本**；含 function call 模板等推理侧实现）;RL 框架 **Slime** 单独开源 https://github.com/THUDM/slime 。框架=**Slime**(Training=Megatron 读 Data Buffer + 同步参数;Rollout=SGLang+Router 生成数据/奖励/verifier 输出写回 Buffer;Data Buffer 桥接) + **BF16 训练/FP8 推理**(在线 block-wise FP8 量化加速 rollout) + Ray 调度 + **agent 用全异步解耦**(rollout/training GPU 分离、Docker 高并发隔离环境)。**CloneTier=B：仅记录链接、不 clone 巨型权重仓**(决策①)。〔注：slime 是 RL 框架≠完整后训练脚本，社区微调多用 LLaMA-Factory/ms-swift〕
- 💰 资源/成本与可扩展性：355B/32B-MoE + 106B Air;预训练 23T tokens、128K 上下文;RL 在 64K 输出长度直接训。**原文未给具体 GPU 数/训练 token 成本明细**(技术报告未公开算力)。slime 支持 colocated 同步(reasoning/code) + disaggregated 异步(agent/SWE) 双模以省 GPU 空转。
- 🎯 对"探索-巩固"对标：**强对标（工业级「巩固」范式 + 可借 pipeline 思想），但粒度粗、机制不同**。判定依据：① **直接支撑「巩固」**——「多专家各自 RL 探索→self-distillation 合并固化进一套参数」就是本课题「巩固=固化进参数且不遗忘」的工业实现;**iterative self-distillation(RL↔蒸回 SFT) 是「探索-巩固」循环的最清晰工程案例**——RL 探索得到更好轨迹→蒸馏固化为新起点→再探索;② **可借组件**——(a) 「RL 探索成果周期性蒸回 SFT 起点」可直接用于 TSRD 的训练节律(student 探索出的成功路径→固化→再探索);(b) 「单阶段 64K 防 unlearn 长上下文」是防遗忘的硬经验;(c) GRPO 去 KL + 难度课程 + 动态温度是即插即用的 RL 配方;(d) 双模(thinking/direct)混合训练对「何时深思/何时快答」有参考。**竞品/张力 & 缺口**：(a) GLM-4.5 的「巩固」是**整模型粗粒度蒸馏/重 SFT**——把整个专家分布蒸进通才，**不区分关键步/路径分叉、无单点接管、无 teacher 稀疏脚手架**，而 TSRD 主张稀疏脚手架 + path-recovery 单点接管 + MTP 前瞻;(b) self-distillation 这里是「自己的强版本教自己」(teacher=RL 后的自身)，与 TSRD「teacher 当稀疏脚手架教探索/选路」的精细监督定位不同;(c) MTP 仅加速、非前瞻信号。一句判定：**GLM-4.5 提供了「探索(RL)→巩固(self-distillation/重 SFT)」循环的最强工业背书与可借 pipeline 节律(尤其 iterative self-distillation 与单阶段 64K 防遗忘)，但其巩固是整模型粗粒度蒸馏、无 teacher 脚手架/无路径级精细化/MTP 仅加速——是 TSRD 的宏观范式参照系而非微观机制竞品**。
- 🔭 开放问题/未来方向：【原文】未设独立 open problem 章(技术报告)；隐含=继续 scale RL、扩展更复杂 agent 环境(slime 异步设计为此铺路)。【推断】(1) **self-distillation 整合多专家的单域能力损失需量化**——能否做「无损/近无损」合并(正是 gopd/G-OPD 用 ExOPD 试图解决的方向，可与本报告对照);(2) iterative self-distillation 的收敛性/分布偏移累积上界;(3) 把粗粒度整模型蒸馏细化到**关键步/路径级**(TSRD 方向)——在 self-distillation 时只对关键决策 token 用更强监督;(4) MTP 从「推理加速」升级为「训练期前瞻探针」(本课题 mtp_opd 核心设想，GLM-4.5 未触及);(5) 后训练脚本未开源，社区复现需补全。

key|读到PDF?|L线|对标结论|残留待核数
glm45 | 是(PyMuPDF全文26页;空密码解密) | L1/L3/L4 | 强对标(宏观范式):「专家RL探索→self-distillation巩固」+「RL↔iterative自蒸馏」是探索-巩固的工业实现,单阶段64K防遗忘可借;但巩固是整模型粗粒度蒸馏、无teacher脚手架/无路径级精细化/MTP仅加速,是宏观参照非微观竞品 | 2(self-distillation净增益与多专家融合的单域能力损失均无独立量化)

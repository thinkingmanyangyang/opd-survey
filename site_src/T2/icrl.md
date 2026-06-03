# icrl — ICRL: Learning to Internalize Self-Critique with Reinforcement Learning

> **一句话重点 (TL;DR)**：让同一 backbone 用 role-specific prompt 同时当 solver 与 critic 联合 RL，把"有 critique 才能做对"内化为"无 critique 也能做对"——核心是一个把 critique-conditioned 修订轨迹按 token 级 re-weight 比率 `w_t=π(y_t|q)/π(y_t|q,c)` 迁移到 critique-free 分布的"分布校准"，外加逐角色组内归一化以稳定联合优化。

**元信息**：arXiv 2605.15224v1（2026-05-13）｜ 港科大(GZ)、南京大学、中山大学、NUS、NTU、SAP、Microsoft Research（多机构）｜ Preprint，2026-05｜ 主题 solver-critic 联合 RL / 自我改进，与 OPD/TSRD 间接相关（"critique-conditioned→critique-free 的 token 级 re-weight"与 OPD"教师条件分布→学生无条件分布"形式同构）；不涉及 MTP，非 teacher-student 蒸馏｜ 代码 https://github.com/brick-pid/ICRL ｜ 框架 slime (THUDM, SGLang-native RL) + AgentGym。

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/icrl/fig_01.png)

*Figure 1: Critique can turn failed trajectories into successful revisions, while training should internalize such revision behavior into the critique-free solver.*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/icrl/fig_02.png)

*Figure 2: Overview of the ICRL framework. (1) Rollout with critique alternates solver and critic; (2) Policy optimization jointly trains both solver and critic; (3) Internalizing critique into solver.*

## 1. 相关工作与进展
外部/自我 critique（Self-Refine、Reflexion、CRITIC）常能引导同一模型纠错成功；Critique-GRPO 等用冻结 critic 做 critique-based RL。GRPO 依赖同前缀同分布的组内比较。

## 2. 现有工作存在的问题

- 撤掉 critique 后模型在同一 query 上又失败——能力没被内化。
- 冻结 critic 无法随训练提升反馈质量，solver 进步后 critic 过时、产无关/冗余反馈。
- critique-guided 成功轨迹来自 critique-conditioned 分布 (y|q,c)，直接当 (y|q) 更新会产生有偏估计，强化"依赖 critique"。
- solver 初解、critic 批评、修订解的 prompt 前缀各异，奖励不可直接组内比较。

## 3. Motivation
把 critique 引发的成功转化为无 critique 的 solver 能力（内化），同时让 critic 与 solver 共享 backbone、协同进化使批评质量可学习，并解决联合优化的稳定性。

## 4. 主要灵感 / 核心直觉
修订轨迹虽在 critique 上下文下产生，但其中一部分 token 在 critique-free 分布下本就可信——只把这部分迁移给 solver、下调重度依赖 critique 上下文的 token，即可内化而不强化依赖。critic 的价值应由"实际有用的反馈"（带来奖励提升）而非"听起来合理的反馈"来定义。

## 5. 主要解决思路(一段话讲清核心)
solver 采初解，失败则 critic 产 critique、solver 据此产修订解（最多 K=2 轮）；对修订轨迹去掉 critique 前缀、重视作 (y|q) 的证据，引入 token 级 re-weight 比率 `w_t=π(y_t|q)/π(y_t|q,c)`（上界 w_max=2）选择性迁移可信 token；solver 组与 critic 组分别做组内归一化（role-wise group advantage）保留各自学习信号；critic 奖励取修订成功记 1 否则等于 solver 奖励的时序提升 r(τᵢ₊₁)−r(τᵢ)。

## 6. 方法详解(通俗、分步骤)

- **Self-Improving Workflow**：solver 采初解 τ₁；失败→critic 产 critique cᵢ→solver 产修订解 τᵢ₊₁；最多 K 轮（实现 K=2，论文 "set the iteration round to 2 to improve training efficiency"）。
- **奖励**：solver 用任务 outcome reward r(τ)∈[0,1]；critic reward=修订成功记 1，否则 r(τᵢ₊₁)−r(τᵢ)（仅 dense reward 下非零）。
- **Critique-Conditioned Distribution Calibration（核心）**：token 级 `w_t=π(y_t|q)/π(y_t|q,c)`，只迁移 critique-free 下本就可信的 token、下调依赖 critique 上下文的 token。
- **Role-wise Group Advantage（Eq.5）**：solver 组与 critic 组分别组内归一化。
- **目标（Eq.6）**：`J=E[ min(w_t,w_max)·min(ρ_t(θ)Â, clip(ρ_t,1∓ε)Â) ]`，仅 critique-guided 修订 solver 轨迹用 w_t（其余 w_t=1），**w_max=2** 防梯度方差爆炸。
- 〔已核-代码〕`icrl/icrl/rewards.py::_role_group_key/_role_sample_norm` 按 `(group_index, role)` 逐角色组内归一化实现 Eq.5；`generate.py:218-225` 对修订轨迹存 `critic_free_prompt_ids`、把 `exec_sample.tokens` 重绑为"critique-free 前缀+原 response"，w_t 在 slime 损失阶段重算。K=2、w_max=2、env_nums=32 在 `hydra_conf/` 确认。

## 7. 实验数据集
四类任务：(1) Text world ALFWorld；(2) Web 导航 WebShop；(3) 多跳 QA（RAG，HotpotQA/2WikiMultiHop/Bamboogle/MuSiQue，合称 SearchQA）；(4) 数学 MATH500/Minerva/OlympiadBench/AIME24/AMC23。Backbone：Qwen3-4B、Qwen3-8B。算力：agentic/Math 4B 用 2×H100、8B 4×H100；SearchQA 4B 4×H100、8B 8×H100。

## 8. 实验结果与主要发现

- agentic（avg.）：4B **57.0**（vs GRPO 49.2 **+7.8**、vs Critique-GRPO 55.9 **+1.1**）；8B **57.8**（vs GRPO 52.8 **+5.0**、vs Critique-GRPO 56.6 **+1.2**）。
- 数学 8B 平均 **75.3%**（vs GRPO 68.3 +7.0、vs Critique-GRPO 73.3 +2.0；AIME24 50.0→65.1）。
- 摘要 "6.4 points over GRPO on agentic / 7.0 on math" 为跨 backbone 平均口径；逐 backbone agentic 为 +7.8(4B)/+5.0(8B)。
- 消融：去 role-wise 归一化 69.8→68.4；去 re-weight ratio 69.8→67.8（均正贡献，增量适中）。
- test-time 多轮 refinement：ALFWorld 第三轮达 **98%**。critic-swap：学到的 8B critic 在 ALFWorld 约 57 token（WebShop 93.9 token）即匹配/超过 20B/32B 冻结 critic（后者数百 token）。

## 9. 结果如何支撑其主张
"内化"由 critique-free 评测下相对 GRPO 的提升支撑；"分布校准/role-wise 归一化有用"由两项消融的正贡献支撑；"critic 可学习且高效"由 critic-swap 实验（小 critic 用极少 token 匹配大冻结 critic）支撑。但相对最直接对手 Critique-GRPO 的领先仅 +1.1~1.2（agentic），优势有限。

## 10. 逻辑自洽性(中性评估)
方法、公式、代码三者一致，消融定向支撑各组件。但增益的主要来源需谨慎归因：相对 GRPO 的大幅提升中，很大一部分来自 critique-based 范式本身（Critique-GRPO 已拿到大半），ICRL 的两项增量（分布校准 + role-wise 归一化）贡献适中而非数量级。

## 11. 残留问题 / 局限

- 相对 Critique-GRPO 领先有限（agentic +1.1~1.2），核心增益与"是否用 critique 范式"耦合。
- 作者自陈：依赖 critique 做修订，长尾轨迹在同步 RL 下 rollout 慢、可能成吞吐瓶颈；异步训练未探索。
- critic 时序提升奖励仅在 dense reward 下非零，sparse 任务下 critic 信号弱。
- Preprint（2026-05），未评审。

## 12. 开源代码与框架(链接+框架+代码可得性)

- 链接：https://github.com/brick-pid/ICRL 。主体在 `ICRL/` 子目录，README 顶层称 "will host the official implementation"，但实际已含完整训练代码（`train.py`/`train_async.py`、`icrl/`、数据 `data/agentgym`、`data/math`、`data/criticgrpo`）。
- 框架：slime（THUDM SGLang-native RL，内置 `slime/`+`slime_plugins/`）；agentic 环境经 AgentGym 起多 server（默认 32 并行 rollout）；数学沿用 Critique-GRPO 设置；RL 原语 GRPO。
- 代码可得、关键实现（role 分组、critique-free 重绑、超参 hydra 配置）可对照。

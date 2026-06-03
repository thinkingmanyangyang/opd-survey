# webagent_r1 — WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning

> **一句话重点 (TL;DR)**：用端到端多轮 on-policy RL（M-GRPO + 动态上下文压缩 + 并行轨迹 rollout），仅靠二值任务成功奖励，把 web agent 从 prompting/BC 水平大幅提升；并系统说明 BC warm-up 不可或缺、long-CoT 帮 SFT 但限制 RL 探索、以及"增加交互轮数"这一新的 test-time scaling。

**元信息**：arXiv 2505.16421v2（cs.CL，2025-10-08）｜ UVA（Zhepei Wei，实习于 Amazon）+ Amazon + Georgia Tech ｜ 预印本 ｜ 主题 T2 web agent / 多轮 RL，与 mtp_opd 关系：纯 on-policy 多轮 RL（**非蒸馏、无 teacher 监督**），可作"无 teacher 的多轮 agent RL"对照；warm-up(BC)+端到端 on-policy 与 path-recovery/初始化策略相关 ｜ 代码 https://github.com/weizhepei/WebAgent-R1 （已 clone，约 150MB）｜ 框架 **verifiers**（willccbb/verifiers，GRPO-based）

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/webagent_r1/fig_01.png)


**② 方法 / 架构图**

![② 方法 / 架构图](../figures/webagent_r1/fig_02.png)

*Figure 2: ( Top ): Overview of the end-to-end multi-turn RL training framework used in WEBAGENT-R1. ( Bottom ): An input/output example of agent-web interaction at the k -th step. The interaction continues until either the maximum number of steps is reached or the agent generates an exit() action to*

## 1. 相关工作与进展
RL 显著提升单轮 LLM（DeepSeek-R1，数学）。早期 web agent 靠 prompting 或行为克隆(BC/SFT)。近期 RL 训 web agent 多为离线/迭代 off-policy（Filtered BC、AWR、DigiRL、WebRL），其中 WebRL 还需额外训 outcome reward model 标 GPT-4 生成的新数据。并发工作 RAGEN、SkyRL 把端到端 on-policy RL 用于模拟游戏/编码，但真实 web 环境仍欠探索。GUI agent 线（含截图多模态）与本文（纯 HTML 文本）正交。

## 2. 现有工作存在的问题
- prompting/BC 缺探索与试错能力，泛化差。
- 离线/off-policy RL 打断 agent 与环境的端到端交互，引入轨迹过滤、outcome reward model 训练、迭代优化等额外复杂度；off-policy 数据来自旧版 agent，在任务相互依赖的动态 web 中学不到关键行为（论文举例：先登出再编辑 profile——登出后失去访问权，旧数据无登出行为则生成无效动作）。
- 每步 HTML observation 可达数千 token，长 horizon 累积上下文带来巨大显存开销甚至 OOM，使多轮 RL 训练不可行。
- 现有 RL 实现（单轮 GRPO）不适配多轮场景。

## 3. Motivation
用端到端 on-policy 多轮 RL 直接从在线交互学习：训练对齐 agent 最新行为、更稳定（Schulman），免 replay buffer/过滤开销，让 agent 据自身过去决策自适应——在"早期决策显著影响后续步"的交互环境中是关键优势。

## 4. 主要灵感 / 核心直觉
把 GRPO 的 group-relative 优势思想搬到多轮：一个任务采 G 条完整轨迹、按 group 内 reward 归一化算优势，逐 token 做 PPO-clip 更新（M-GRPO）。再用工程手段（动态压缩 + 并行 rollout）解决多轮 RL 在 web 上的显存与采样瓶颈。直觉上，"thinking 格式"促使 agent 进行更多轮交互而非更长单轮输出，揭示了 web 任务的 test-time scaling 走"交互深度"而非"单步长度"。

## 5. 主要解决思路(一段话讲清核心)
先用专家演示做 BC warm-up 初始化（acquire 基本 web 操作技能），再在 WebArena 环境做端到端多轮 on-policy RL：用规则化二值 outcome reward 引导，M-GRPO 逐 token 优化（带 group-relative 优势 + KL 正则），并行 rollout G 个独立浏览器实例生成多样轨迹，动态上下文压缩把历史 observation 替换成"Simplified HTML"占位符（保留完整动作历史、相应更新 loss mask），交互直到达最大步数或 agent 输出 exit()。

## 6. 方法详解(通俗、分步骤)
1. **POMDP 形式化**：状态=当前页 HTML 文本，动作∈预定义动作空间（Click/Type/Select/Scroll/Search/Exit…），终态二值奖励 r∈{0,1}。
2. **BC warm-up**：在固定专家演示 D={(h_t,a_t)}（h_t=完整交互历史）上 SFT，L_BC=−E log πθ(a_t|h_t)。用公开 9,460 条轨迹。
3. **动态上下文压缩**：新 observation 到来时把旧 s_i 替换成短模板 s'_i（"Simplified HTML"），保留完整动作历史，防 OOM；同步更新 loss mask 使损失只算在动作 token 上。
4. **M-GRPO**：每任务采 {τ1..τG}，逐 token 优势 Ã_{i,j,t}=PPO-clip(重要性比·A_{i,j})，组相对优势 A_{i,j}=(r_i−mean(r))/std(r)，附 βKL 项。
5. **并行轨迹 rollout**：G 个独立浏览器实例（各自 cookie/上下文），同起始页、独立交互 → 多样历史。
6. **奖励**：直接用环境默认规则化二值奖励（String/URL Match、程序执行），**无 outcome reward model**。
7. **三变体研究初始化策略**：R1（标准 BC→RL）、R1-Zero（无 SFT 直接起 RL）、R1-CoT（用 long-CoT BC 数据 SFT 初始化）。

## 7. 实验数据集
- 训练/评测：**WebArena-Lite**（WebArena 的 human-verified 自托管沙箱版，跨 Reddit/GitLab/CMS/Map/Shopping 五站；非线上真实站点）。BC 用公开 **9,460** 条轨迹；RL 用 **647** 个任务训练、**165** 个 verified 任务评测；成功率由内置规则 rubric 判定。
- OOD 评测：**WebVoyager**（5 个域、每域随机 25 任务，域在 WebArena 中未见）。
- 模型：Qwen2.5-3B、Llama3.1-8B（finetune backbone）；对照含 GPT-4o/o3/o4-mini、QwQ-32B 等。

## 8. 实验结果与主要发现
- **主结果（Table 2，WebArena-Lite 平均 SR%）**：Qwen2.5-3B 6.1→33.9；Llama3.1-8B 8.5→44.8。均超同尺寸 SFT/Filtered BC/AWR/DigiRL/WebRL（8B 上 WebRL 42.4 < 本法 44.8）。
- **与专有模型对照**：OpenAI o3=39.4、o4-mini=36.9。**8B 版(44.8) 超 o3；但 3B 版(33.9) 仍低于 o3**——"超 o3"主要由 8B 支撑，非全尺寸成立。逐站表现不均（如 Map 上本法 23.1 弱于 WebRL 36.7）。
- **BC 必要性（Fig.4）**：R1-Zero（无 SFT）初始 6.1%，RL 后**反而略降**（动作残缺/格式错、罕得正奖、探索失败）。
- **long-CoT（Fig.4）**：long-CoT SFT 初值 24.5% > 标准 BC 20%，但 RL 增益更小（R1-CoT 24.5→30.3 vs R1 20→33.9）；作者假设确定性 CoT 模式约束了 RL 探索空间。
- **thinking prompting（Table 3）**：显著提升 SR（o4-mini 15.9→36.9）；单轮长度几乎不变（Qwen 139→142）但交互轮数大增（6→17），指向"交互深度"型 test-time scaling。
- **test-time scaling（Fig.5）**：放宽最大交互轮数，prompting/SFT/RL 三类成功率均单调提升。
- **OOD（Table 4，WebVoyager）**：WebAgent-R1 平均 32% > SFT 12% > prompting 8.8%，各域全面领先。
- **训练动态（Fig.3）**：reward/轨迹长/交互数呈三阶段（初始技能获取→探索精炼→策略稳定），3B/8B 模式相似。

## 9. 结果如何支撑其主张
- "端到端 on-policy 多轮 RL 有效"：Table 2 两 backbone 的大幅提升 + 超 off-policy 基线支撑。
- "BC warm-up 关键"：R1-Zero RL 后退化是强反例支撑。
- "test-time scaling 走交互深度"：Table 3（长度不变、交互↑）+ Fig.5（轮数↑→SR↑）双重支撑，较有说服力。
- "泛化"：Table 4 OOD 领先支撑（但每域仅 25 任务，样本小）。
- "超 SOTA / 超 o3"：成立但需限定——8B 上超 o3、3B 不超；且基线取"复现与文献中较高者"，比较口径需注意。

## 10. 逻辑自洽性(中性评估)
- 论证整体自洽：off-policy 痛点（登出例子）→ on-policy 设计动机 → M-GRPO/工程 → 结果，链条连贯。
- 几处 claim 偏 hypothesize：long-CoT 限制探索、R1-Zero 失败归因，均为假设性解释而非受控证据。
- "超 o3"的标题级主张需谨慎（仅 8B、且 o3 为零样本 prompting 对照，非微调，比较不完全对等）。
- M-GRPO 相对单轮 GRPO 的"多轮"实质主要是把整条多轮轨迹的 token 一并纳入优化 + 二值终态奖励，无中间步奖励塑形（作者列为 future work）。

## 11. 残留问题 / 局限
- 仅文本 HTML 输入，无截图/多模态（作者承认）。
- 依赖规则化可验证 outcome reward，开放式任务（如旅行规划）不适用。
- 固定预定义动作空间，遇需新操作的交互元素受限。
- 仅 WebArena-Lite（沙箱、5 站）；OOD 评测样本小（每域 25）。
- 多处归因为假设，未做消融验证；安全风险（CMS 误删等）需权限/确认机制。

## 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库：https://github.com/weizhepei/WebAgent-R1 （已 clone，约 150MB；含 `WebAgent-R1/{Train, Eval, WebArena-Env-Setup}`）。
- **框架：verifiers**（willccbb/verifiers，"RL with LLMs in Verifiable Environments"，GRPO-based，Accelerate + DeepSpeed ZeRO3）；`Train/` 下实证含 `verifiers/`、`VisualAgentBench`，在其上扩展为 M-GRPO + 并行 rollout。评测/环境为 WebArena（`Eval/browser_env`，WebArena-Lite）。
- 〔已核：原始 shortlist 标 framework=unknown 不准——据仓库实证为 **verifiers（非 veRL）**。〕
- 代码可得性：完整开源（含训练/评测/环境搭建）。

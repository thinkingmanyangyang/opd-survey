# tcod — TCOD: Exploring Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents

> **一句话重点 (TL;DR)**：在多轮 agent 场景下 vanilla OPD 会因跨轮误差累积出现"轨迹级 KL 不稳定"（KL 飙升、成功率坍塌），TCOD 用一条时间课程逐步扩大暴露给学生的轨迹深度（F2B 浅到深 / B2F 由 teacher 前缀导航后到前），在保留 OPD 稠密信号的同时稳住训练并最高 +18 分。

**元信息**：arXiv 2604.24005（v3, 2026-04-29，cs.LG）｜ Tongyi Lab, Alibaba Group + 香港中文大学（CUHK）｜ 预印本 2026-04 ｜ 主题 T1(On-Policy Distillation)+T2(多轮自主 agent)，与 mtp_opd 高度相关 ｜ 代码 https://github.com/kokolerk/TCOD （已 clone ~84MB，可运行配置齐全）｜ 框架 Trinity-RFT（Ray-based RFT，Alibaba）

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/tcod/fig_01.png)

*Figure 1: (left) In OPD for multi-turn agents, as the number of turns increases, the teacher assigns progressively lower probabilities to tokens in student-generated responses, indicating increasing KL divergence at each turn, rendering the supervision signal unreliable. (right) OPD uses all turns a*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/tcod/fig_06.png)

*Figure 3: Overview of our method TCOD-F2B/B2F. Comparison of vanilla on-policy distillation and TCOD. Left is the OPD, middle is the illustration of TCOD-F2B, and right is TCOD-B2F. k is the linear pacing control the trajectory length. The blue step is executed by the student, and the red step is ex*

## 1. 相关工作与进展

- **多轮 LLM agent**：ReAct（reason+act 交错）是主流范式，应用于具身规划、网页导航等交互环境；OpenClaw 等系统推动长程 agent。
- **On-Policy Distillation 及其改进**：OPD（Agarwal 2024 GKD；Lu & Thinking Machines Lab 2025）用稠密蒸馏信号替代稀疏标量奖励、提升样本效率；既有改进集中在目标设计（forward/backward KL 平衡，Jang 2026、Jin 2026 即 entropy-aware OPD）、优化启发（reward clipping，Ko 2026）、监督来源（context distillation、self-distillation）——但均面向**单轮静态**任务。
- **课程学习**（Bengio 2009）：近期被用于 LLM 预训练/后训练与 RL（GRPO），但多依赖**外部模型评估难度**；Lauffer 2025 只用 expert 后续纠错动作训练学生，破坏了 on-policy 设定。TCOD 以"递增轨迹深度"定义难度，只用学生自生成数据，避开外部难度评估器。

## 2. 现有工作存在的问题
作者在 ALFWorld 上实证出 **Trajectory-Level KL Instability（轨迹级 KL 不稳定）**：

- 观察1：训练中 KL 持续**升高**且成功率**坍塌至近零**（与单轮设置 KL 收敛递减相反；Fig.2a/2b，小模型 Qwen3-0.6/1.7B、Qwen2.5-0.5/1.5B 上显著）。
- 观察2：即使最终收敛，初始 KL 极高（~1000 量级 vs 收敛 ~60，Fig.2c）。
- 机理：**跨轮误差累积（compounding error）**——学生动作/观测追加到历史 ht，因果耦合使 per-turn KL 随轮次单调增大（Fig.2d），学生被推出 teacher 有效支撑（support），teacher 对学生 token 赋低概率、监督信号失真。Remark 1 区分：long-CoT 只增长同一环境状态下的响应长度，而多轮 agent 每轮更新环境状态，才真正放大累积误差。

## 3. Motivation
如何在保留 OPD 稠密信号优势的同时，避免长 horizon 累积误差导致的失稳？借鉴课程学习（先易后难）：**控制暴露给学生的轨迹深度**，让学生始终停留在 teacher 有效指导范围内，从短稳定前缀渐进扩展到完整多轮 rollout。

## 4. 主要灵感 / 核心直觉
误差累积是长 horizon 的固有属性；与其一开始就让学生端到端跑完整轨迹（必然进入 teacher 支撑外），不如沿时间维度做课程——要么从轨迹早段学起逐步加深（F2B），要么让 teacher 把环境导航到接近终止状态、学生从"成功门口"接管再逐步前移（B2F）。难度 = 轨迹深度 k，可由训练步线性 pacing 自动调度，无需外部难度评估器。

## 5. 主要解决思路(一段话讲清核心)
TCOD 用线性 pacing \(k = k_{\mathrm{start}} + \lfloor n/\eta \rfloor\)（n 当前步、N 总步、η 控制增长率）沿训练步控制轨迹深度 k，并给出两个仅需极小代码改动的变体：**F2B**（学生 rollout 至多 k 步，k 从小到大，目标只对前 k 步算 token-level KL）；**B2F**（teacher 用预采集成功轨迹 τ\* 执行前 L−k 步把环境导到中间状态、stop-gradient，学生从该状态接管后 k 步并对其算 KL，k 递增直到学生端到端）。学生执行步计算 KL 梯度，teacher 执行步停梯度。

## 6. 方法详解(通俗、分步骤)

- **前置定义**：状态 \(h_t = (o_0, a_0, \ldots, o_{t-1}, a_{t-1}, o_t)\) 为完整交互历史；OPD 目标 \(L_{\mathrm{OPD}} = \mathbb{E}_{\tau \sim \pi_\theta} \sum_t D_{\mathrm{KL}}\big(\pi_\phi(a_t \mid h_t) \,\|\, \pi_\theta(a_t \mid h_t)\big)\)（reverse-KL，teacher πϕ、student πθ）。
- **TCOD-F2B（Algorithm 1）**：每训练步 \(k \leftarrow \min(k_{\mathrm{start}} + \lfloor n/\eta \rfloor,\ T_{\max})\)；学生从 h0 起 rollout k 步；\(L = \sum_{t=0}^{k-1} D_{\mathrm{KL}}(\pi_\phi \,\|\, \pi_\theta)\)；更新 θ。先学早轮信号再端到端，防 horizon 诱发的 KL 坍塌；无需任何示范。
- **TCOD-B2F（Algorithm 2）**：预采集 teacher 成功轨迹 T\*（pass@10 采样，保留成功）；每步 teacher 执行 τ\* 的前 L−k 步（stop-gradient）把环境带到中间状态，学生从 t=L−k 接管到 L，仅对学生步算 KL；k 递增直到学生从初始状态端到端。论文用 Appendix D.5 验证：随训练步 teacher 前缀从 L−1 降到 0，测试集端到端成功率稳步上升，证明课程平滑过渡避免了 train-test 分布漂移。
- **异步训练稳定性细节**：异步 rollout（4×H20 actor）/训练（2×H20 learner）/teacher（2×H20），lock-free ring buffer；**staleness-aware 子轨迹经验回放**——把长度 n 的轨迹拆成 n 个递归前缀子轨迹入 buffer，交互历史封装进 prompt 作结构化上下文；用 staleness filter 丢弃 \(n_{\mathrm{current}} - n_{\mathrm{old}} > \Delta_{\max}\) 的经验，经验上 Δ_max=2 最优。

## 7. 实验数据集
三个多轮 agent benchmark（Table 1）：**ALFWorld**（具身，max 30 turns，含 seen/unseen/hard split）、**WebShop**（电商，max 15 turns，需 ~1TB 内存）、**ScienceWorld**（科学推理，max 30 turns，需 Java/jar）。Hard split = 121 个 teacher 在 train split 上 pass@10 仍失败的任务。师生对：ALFWorld 主实验 student=Qwen2.5-{3,7}B、teacher=GRPO 训练的 Qwen2.5-7B（域内 SR 85.71）；跨 benchmark student=Qwen3-{1.7,4}B、teacher=Qwen3-30B-A3B-Instruct（通用、域内偏弱）。8× H20(96GB)。

## 8. 实验结果与主要发现

- **Q1 缓解 KL 升高 + 提升性能（Table 2，Qwen2.5-7B teacher）**：Qwen2.5-3B 上 F2B 在 Valid Unseen 较 OPD **+18.74 SR**、Seen +15.71；7B 上 B2F Seen +11.06、Hard +7.44。平均减少约 2.97 个动作轮次；KL 曲线更平稳、advantage 收敛更快（Fig.4/5）。
- 跨 benchmark（Table 3，Qwen3-30B teacher）：对 Qwen3-1.7B，vanilla OPD 几乎全崩（avg 0.17），TCOD 把三 benchmark 平均拉到 ~18.6（**+18.4~18.7**，从近零恢复）；对 Qwen3-4B 提升温和（avg +0.3~+1.9，与 OPD 相当）。
- **Q2 超越 teacher 能力边界**：Unseen 上较 teacher 最多 +2.5；**Hard split 上 teacher SR 仅 6.61，B2F 反超最高 +14 分**（B2F 7B 达 20.66）——非单纯模仿。
- **Q3 鲁棒性与效率**：η∈{2,4,6} 下性能波动 <2%；总训练时间较 vanilla OPD 减少近 32%（F2B 比 B2F 更省，因 B2F 中间状态接管后仍多探索几步）。teacher 质量比规模更决定上界：域内强的 7B-RL teacher 下 B2F 可略超 teacher（+0.7），而通用 30B teacher 下 OPD/TCOD 均差 teacher ~2 分。

## 9. 结果如何支撑其主张

- "失效模式真实存在"：Fig.2（KL 升高 + 成功率坍塌 + per-turn KL 随轮次增）直接对应"轨迹级 KL 不稳定 + 误差累积"机理。
- "课程能稳训练"：Fig.4b KL 曲线 TCOD 明显更平、Fig.5 response length/pg_loss 平滑，支撑"控制轨迹深度即可稳住"。
- "超越 teacher"：Hard split（teacher pass@10 失败任务）上的正增益是较强证据，说明学生学到的是更鲁棒策略而非复制 teacher。
- 效率主张由 Fig.6 训练时间柱状图（OPD vs F2B vs B2F）支撑。

## 10. 逻辑自洽性(中性评估)
整体自洽：失效现象→机理（误差累积）→对策（控制轨迹深度的时间课程）→两变体（F2B 无需示范 / B2F 需 teacher 成功轨迹）逻辑闭环，且课程末端回到端到端、消解了 B2F 的 train-test mismatch（有 Appendix D.5 佐证）。一个张力点：观察机理中作者自己承认"KL 升高既可能是学生模仿不能、也可能是学生进入 OOD 致 teacher 不确定"二者难以区分，但两种解释指向同一对策，不影响方法有效性。Qwen3-4B 上 TCOD 仅与 OPD 相当（非显著提升），说明收益高度依赖"学生本会崩 + teacher 域内够强"的条件。

## 11. 残留问题 / 局限

- B2F 依赖预采集 teacher 成功轨迹，有额外采样开销（F2B 是无示范替代）。
- 固定课程 pacing 虽鲁棒，但最优节奏可能随环境/师生对变化；作者建议未来用 KL 的 EMA 做自适应 horizon，尚未实现。
- 仅在三个**文本**多轮 benchmark 验证，未覆盖多模态/物理具身。
- 收益条件性强：teacher 域内性能（而非规模）是上界主因；通用强 teacher 下 TCOD 无法超越 teacher。
- 〔待核〕各师生对完整数值、k_start/η 取值与超参表见论文 Appendix C/D（正文已给 k_start=1、η=2 默认，η∈{2,4,6} 消融）。

## 12. 开源代码与框架(链接+框架+代码可得性)

- 仓库 https://github.com/kokolerk/TCOD （已 clone，构建于 Trinity-RFT）。README 明确"What Is Implemented"：三环境各一套 multi-turn OPD + TCOD-f2b + TCOD-b2f workflow（`trinity/common/workflows/envs/TCOD/{alfworld,webshop,scienceworld}`）、基于师生 token-level logprob 的蒸馏信号、可运行示例配置 `TCOD_examples/<env>/{opd,tcod_b2f,tcod_f2b}.yaml`。
- 模型：ModelScope `wjqkoko/TCOD`；HuggingFace `kolerk/tcod`。
- 关键配置：`algorithm.advantage_fn: multi_turn_opd`；`rollout_args.logprobs` 必须开启以算蒸馏 gap；TCOD 配置用 `workflow_args.checkpoint_strategy: linear`（即线性 pacing）。
- 框架：Trinity-RFT（`trinity run --config <yaml>`，Ray 驱动），依赖 vLLM 类推理 + flash-attn 2.8.1；环境 ALFWorld（pip）/ WebShop（princeton-nlp/webshop，需 Java17+、~1TB 内存）/ ScienceWorld（allenai，jar）。代码可得性高、与论文方法对应清晰。

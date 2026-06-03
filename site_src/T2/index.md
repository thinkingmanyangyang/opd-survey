# T2 · Tool-Agent / 多轮

共 19 篇。

- [chain_of_agents — Chain-of-Agents: End-to-End Agent Foundation Models via Multi-Agent Distillation and Agentic RL](chain_of_agents.md) — 把"多智能体协作"压进单一模型——先用 multi-agent distillation 把 SOTA 多智能体系统（OAgents）的执行轨迹转成 CoA 格式做 agentic SFT 冷启动，再用 agentic RL（DAPO）在可
- [deepdive — DeepDive: Advancing Deep Search Agents with Knowledge Graphs and Multi-Turn RL](deepdive.md) — DeepDive 用知识图谱（KG）随机游走 + 属性模糊化自动合成"难找"deep-search QA，再用端到端多轮 GRPO（带 redundancy penalty 抑制重复查询）训练浏览 agent，DeepDive-32B 在 
- [distill_agent_tools — Distilling LLM Agents into Small Models with Retrieval and Code Tools (Agent Distillation)](distill_agent_tools.md) — 不只蒸馏教师的"推理"，而是把教师 agent 的完整"think + act（检索/代码工具）"任务求解行为蒸馏到小模型；配两项改进——用 first-thought prefix 提高教师轨迹质量、用 self-consistent a
- [gigpo — Group-in-Group Policy Optimization for LLM Agent Training (GiGPO)](gigpo.md) — 把 GRPO 扩展到长 horizon 多轮 agent——除了像 GRPO 那样按整条轨迹回报算 episode 级相对优势,再利用"组内轨迹常反复经过相同环境状态"这一观察,把同一状态下的不同动作聚成 step 级组算 micro 相对
- [icrl — ICRL: Learning to Internalize Self-Critique with Reinforcement Learning](icrl.md) — 让同一 backbone 用 role-specific prompt 同时当 solver 与 critic 联合 RL，把"有 critique 才能做对"内化为"无 critique 也能做对"——核心是一个把 critique-co
- [latent_agents — Latent Agents: A Post-Training Procedure for Internalized Multi-Agent Debate](latent_agents.md) — 用"SFT 学辩论结构 + GRPO 把显式辩论压进潜空间"的两阶段微调，把多个 agent 多轮辩论（multi-agent debate）内化进单个 LLM（IMAD），用 Debate 6.3%~21.1% 的 token（5–16×
- [meow_tea_taro — A Practitioner's Guide to Multi-turn Agentic Reinforcement Learning](meow_tea_taro.md) — 把多轮 agentic RL 的设计空间拆成 environment/reward/policy 三支柱做系统受控消融，得出一份可操作配方——"课程(由简到繁) + 稳定化偏置策略(PPO/GRPO 优于无偏 RLOO 与朴素 REINFO
- [open_agentrl — RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System (Open-AgentRL)](open_agentrl.md) — 一个完全动态的闭环 RL 系统,同时进化"环境、策略、生成式奖励模型"三者——策略用 step-wise+outcome 融合反馈训练,奖励模型经一致性反馈联合优化产出可靠 step-wise 监督,环境据策略当前能力自适应调难度;论证优化
- [openclaw_rl — OpenClaw-RL: Train Any Agent Simply by Talking](openclaw_rl.md) — 把每次 agent 交互产生的"next-state 信号"当作在线学习源回收,从中抽 evaluative(标量/更频繁)与 directive(token-level/更富信息但稀疏)两类信号,在一次 hybrid RL 更新中统一;并
- [pi_play — π-Play: Multi-Agent Self-Play via Privileged Self-Distillation without External Data](pi_play.md) — 自博弈在造题时天然产出一条 "问题构造路径(QCP)"，本文把它当作零成本的内禀特权信息，让同规模 teacher 据此对 student 做 token 级 reverse-KL 自蒸馏，从而把稀疏奖励自博弈变成稠密反馈的 data-fr
- [rstar2 — rStar2-Agent: Agentic Reasoning Technical Report](rstar2.md) — 用"高吞吐代码执行环境 + 抗噪的 GRPO-RoC（Resample-on-Correct）+ 短长度多阶段 RL recipe"，在 64×MI300X、510 步 / 一周内把 Qwen3-14B-Base 推到前沿数学推理，AIME
- [score — From Correction to Mastery: Reinforced Distillation of LLM Agents (SCoRe)](score.md) — 让小学生 agent **主导**轨迹生成、teacher **只纠正最早一步错误**,学生从"已验证前缀"续写并做短 horizon RL,把行为克隆的累积误差从 O(H²) 降到 O(H);RL 阶段从最早错误前的前缀起 rollout
- [search_dont_guess — Search, Do not Guess: Teaching Small Language Models to Be Effective Search Agents](search_dont_guess.md) — 小模型(SLM)做 search agent 时反而比大模型**更少搜索、更易幻觉**(under-searching),且"自适应搜索"在 SLM 上会掉点(Adaptive Search Trap);本文用 **Always-Searc
- [search_r1 — Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](search_r1.md) — 把搜索引擎调用嵌进 RL 训练循环——用结构化标签做多轮"推理↔检索"交织生成,对检索回来的 token 做 **loss masking**,仅用最简的 **EM 结果奖励**(无过程/格式奖励),即可让 LLM 自发学会"何时检索、检索
- [sod_stepwise — SOD: Step-wise On-policy Distillation for Small Language Model Agents](sod_stepwise.md) — 把 OPD 用于小模型 agent 的工具集成推理（TIR）会因工具错误触发的"加速分布漂移"而训练崩溃；SOD 按 step 级师生发散自适应重加权蒸馏强度（高发散区衰减、重对齐时回升），在对齐区保留 dense 监督，使 0.6B/1.
- [spear — SPEAR: Learn the Ropes, Then Trust the Wins — Self-imitation with Progressive Exploration for Agentic RL](spear.md) — 针对多轮 agent RL 中机械熵最大化易致不稳定的问题，SPEAR 用"课程化自模仿学习（SIL）+ 内在奖励塑形"在自身经验引导下渐进调节策略熵（早期广探索、后期收敛利用），在 ALFWorld/WebShop/Sokoban/AIM
- [sweet_rl — SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks](sweet_rl.md) — 在多轮 agent 任务上，用训练期才可见、actor 看不到的额外信息（最终结果 + 参考解）训练一个非对称的 turn-level critic 做 step 级信用分配；关键设计是"直接用动作 log-prob 参数化 advanta
- [tcod — TCOD: Exploring Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents](tcod.md) — 在多轮 agent 场景下 vanilla OPD 会因跨轮误差累积出现"轨迹级 KL 不稳定"（KL 飙升、成功率坍塌），TCOD 用一条时间课程逐步扩大暴露给学生的轨迹深度（F2B 浅到深 / B2F 由 teacher 前缀导航后到前
- [webagent_r1 — WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning](webagent_r1.md) — 用端到端多轮 on-policy RL（M-GRPO + 动态上下文压缩 + 并行轨迹 rollout），仅靠二值任务成功奖励，把 web agent 从 prompting/BC 水平大幅提升；并系统说明 BC warm-up 不可或缺、

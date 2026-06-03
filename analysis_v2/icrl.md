icrl | ICRL: Learning to Internalize Self-Critique with Reinforcement Learning | 港科大(GZ)·南大·中山大·NUS·NTU·SAP·Microsoft Research(brick-pid) | 2026-05-13 · arXiv 2605.15224v1 · preprint | 主题线 L4 Agent/工具/多轮自进化(兼 L3 GRPO 变体) · 相关性 高

**原始论文**:https://arxiv.org/abs/2605.15224

## 一眼看懂
- 🟦 TL;DR:同一个 backbone 用两套 role prompt 同时当 solver(解题)和 critic(失败后写自然语言批评),联合 RL。目标是把"有 critique 才能做对"内化成"无 critique 也能做对"。核心难点:critique-guided 成功修订轨迹来自 critique-conditioned 分布 (y|q,c),直接拿来更新 critique-free 策略 (y|q) 会有偏、反而强化"依赖 critique"。解法是一个 **token 级"分布校准"重加权比 `w_t = π(y_t|q, y<t) / π(y_t|q, c, y<t)`**(Eq.4):去掉 critique 后这个 token 本就高概率→强迁移给 solver;高度依赖 critique 上下文→下调。外加**逐角色组内归一化**(Eq.5,solver/critic 各自算 group-relative advantage,因为前缀和奖励语义不同不能混)和**utility-based critic 奖励**(Eq.3,critic 因"实际带来修订成功"而非"听起来合理"而获奖)。
- 最巧的一步:**token 级 reweight 比 w_t(Eq.4)**。抽掉它(把修订轨迹当普通 critique-free rollout 直接 GRPO 更新),就只是在强化"依赖 critique"的行为、无法内化——消融 Table 4 显示去掉 w_t 平均从 69.8 掉到 67.8(全 benchmark 一致下降)。为什么是支点:它是"把 critique 上下文产生的好行为安全搬到无 critique 策略"的唯一机制,直接对应论文标题的"internalize"。

## 为什么做
- 研究背景:外部/自我 critique(Self-Refine、Reflexion、CRITIC)在**推理时**能引导同一模型纠错;Critique-GRPO 用 critique 做 critique-based RL。GRPO 靠同前缀同分布的组内比较。
- 解决的具体痛点:① 撤掉 critique 后模型在同一 query 又失败——能力没内化(只改善了 critique-conditioned 推理时行为);② 冻结 critic 无法随训练提升,solver 进步后 critic 过时、产无关反馈;③ critique-guided 成功轨迹来自 (y|q,c),当 (y|q) 更新有偏;④ solver 初解 / critic 批评 / 修订解前缀各异、奖励不可直接组内比较。
- 相关工作 & 各自不足:Self-Refine/Reflexion/CRITIC(只在推理时给局部引导,不保证内化进 critique-free 策略);Critique-GRPO(用 critic 反馈做 RL,但 critic 静态/冻结);MATPO(planner+subagent 角色训练);ScalingInter(逐步增交互步数);GRPO/GSPO(单 agent)。Δ:都没研究"如何把 critique-guided 修订内化进 critique-free solver"。
- 动机链:critique 能在推理时救场但不内化(现状)→ 直接拿修订轨迹训会强化依赖 + 冻结 critic 退化 + 多前缀奖励不可比(缺陷)→ 共享 backbone 联合训 solver/critic + token 级分布校准 + 逐角色优势 + utility critic 奖励(所以必须这样)。
- 与最近邻工作的 Δ:vs Critique-GRPO(最近邻):ICRL 让 critic 与 solver **共享 backbone 协同进化**(critic 可学、不退化)+ 用 **w_t 把修订内化进无 critique 策略**(Critique-GRPO 静态 critic、不内化)。关键有用点:推理时不用 critique 也变强(Table 2 数学 + Table 1 agent 均超 Critique-GRPO),且 critic 学到"短而准"的可执行反馈。

## 怎么做 + 靠不靠谱
- 方法流水线(Algorithm 1):对每个 query q,跑 G 个独立 self-improving session,每个 session 至多 K 轮:① solver 采初解 τ1∼π^S(·|q),环境给 outcome reward → ② 若失败,critic 写 c_i∼π^C(·|q,τ_i) → ③ solver 采修订 τ_{i+1}∼π^S(·|q,c_i),成功或耗尽轮次则停。收集 solver 组 G^S 与 critic 组 G^C → ④ 逐角色算 advantage(Eq.5)→ ⑤ 对 critique-guided 修订 solver 轨迹算 token 级 w_t(Eq.4)→ ⑥ 最大化多角色 GRPO-style clipped 目标 J(θ)(Eq.6,w_t 与 wmax 双重界住方差),联合更新共享 θ。
- 逐组件必要性(均有消融,Table 4):
  - **Re-weight ratio w_t(分布校准)**:去掉 → 69.8→67.8(最大降幅)。证明修订轨迹不能当 critique-free rollout 直接优化。
  - **Role-wise advantage(逐角色组内归一)**:去掉 → 69.8→68.4。证明 solver/critic 异质奖励不能用单一 scale 归一。
  - **Utility-based critic 奖励(Eq.3)**:无独立"去掉"消融,但通过 critic-swap(Table 3)间接证明——用此奖励训出的 8B 共享 critic 用 57 tokens 即匹配 20B/32B 冻结 critic 的 921/526 tokens 效果【原文 §5.3】。
- 关键机制/公式(直觉):w_t<1 → 该 token 重度依赖 critique,保守更新(早期普遍 <1,Fig.4);w_t≈1 → 与 critique-free 分布兼容,强迁移;w_t>1 → solver 本就更可能产此 token,上调。wmax 防止 critique-free 概率远大于 critique-conditioned 时权重爆炸。critic 奖励 = 修订成功给 1,否则给"solver reward 的时间增量"(只在 dense reward 时非零)。
- 实验与证据:
  - 任务:agentic(ALFWorld 具身、WebShop 电商、HotpotQA/2Wiki/Bamboogle/MuSiQue 多跳 RAG)+ 数学(MATH500/Minerva/Olympiad/AMC23/AIME24)。Backbone:Qwen3-4B、Qwen3-8B。
  - baseline:Prompting(Qwen3 系 + Gemini-2.5/3-Flash)、GRPO、GSPO、ScalingInter、MATPO、**Critique-GRPO**(最强对手)。
  - agent(Table 1):Qwen3-4B ICRL avg 57.0(超 GRPO +7.8、超 Critique-GRPO +1.1);8B avg 57.8(超 GRPO +5.0、Critique-GRPO +1.2)。ALFWorld/WebShop 最佳。
  - 数学(Table 2,Qwen3-8B):ICRL avg 75.3(超 GRPO +7.0、Critique-GRPO +2.0),AIME24 50.0→65.1 提升明显;5 个里 4 个超 Critique-GRPO(仅 AMC23 例外)。
  - 内化证据(关键):① §5.1 test-time 多轮 refinement,ICRL 第一轮就更强且随轮次涨更多(ALFWorld 第3轮 98%);② §5.2 Fig.4,**w_t 随训练从 <1 稳步上升 → 越来越多修订 token 与 critique-free 分布兼容 = solver 在内化修订模式**;③ §5.3 critic-swap,8B 共享 critic 用极短 critique 匹配/超大模型冻结 critic(57 vs 921 tokens)。
  - baseline 公平性:同 backbone、同环境;Critique-GRPO 是同类最强对手,逐项可比,公平。
  - "看着强但没答核心问题":数学上 AMC23 不及 Critique-GRPO;多跳 QA 多数 benchmark 非一致最优(只 2Wiki 双 backbone 最佳)——增益不均匀,但平均稳超。
- 假设与失效边界:
  - 【原文】依赖 outcome reward 可得(r(τ)∈[0,1]、可验证成功);critic 的 dense 奖励项仅在环境给非二值 reward 时非零(Eq.3),否则退化为 0/1。
  - 【原文】§5.2 critic reward 曲线更低更抖,自承"学批评比学解题更难、与任务成功的关联更间接"。
  - 【推断】强依赖"失败→批评→修订"这个可迭代 self-improvement 循环 + 可执行环境(AgentGym),开放无环境/无明确成功信号场景难用。
  - 【推断】solver 与 critic 共享 backbone,二者能力耦合;若 critic 学崩可能拖累 solver(论文靠 role-wise advantage + wmax 稳住,但未给失败案例边界)。
- 祛魅总结【推断】:
  - 真贡献:① 明确提出并解决"critique-conditioned→critique-free 的分布偏移"这个被忽视问题,w_t 是干净可复用的校准件;② utility-based critic + 共享 backbone 让 critic 可学且"短而准";③ 内化证据链(test-time / w_t 上升 / critic-swap)较扎实。
  - 包装/可能高估:整体是 GRPO 的多角色 + token reweight 扩展,单项增益(尤其 vs Critique-GRPO 平均 +1~2 分)不算大;"internalize" 的强叙事主要靠 w_t 动态与 test-time 曲线支撑,部分 benchmark 并不一致领先。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=outcome reward(solver)+ critique 下游修订 utility(critic,Eq.3)+ token 级 critique-free/critique-conditioned 概率比 w_t |**改什么**=单一共享 backbone 参数(solver+critic 两 role 联合)|**何时改**=每 step,对 critique-guided 修订 token 施 w_t、逐角色归一优势|**免梯度?**=否,GRPO-style clipped 策略梯度|**记忆-技能生命周期**=无显式记忆库;"修订/纠错技能"经 w_t 从 critique-conditioned 行为内化进 critique-free 策略参数(随训练 w_t↑)|**防遗忘机制**=隐式——逐角色 advantage + wmax 稳住联合优化;Table 4 未测遗忘,但 test-time 第一轮(无 critique)持续走强暗示内化不抹除直解能力。
- ⑦ 开源代码+框架/harness:https://github.com/brick-pid/ICRL (已 clone,真实可用,含中英 README)。**框架=slime**(THUDM SGLang-native RL,repo 内置 `slime/`+`slime_plugins/`,`train.py`/`train_async.py`)+ **AgentGym**(WooooDyy/AgentGym,提供 ALFWorld/WebShop/SearchQA 训练环境,默认起 32 个 env server 并行 rollout)。注:README 称 "will host the official implementation",代码已在仓内但措辞偏"即将正式发布"〔待核-以仓内现有实现为准〕。
- 💰 资源/成本与可扩展性:并行 rollout 需 32 个环境 server(端口 36001-36032),env_nums=32;backbone 4B/8B。具体 GPU 数/训练时长原文未说明。亮点:共享 backbone 省一个独立 critic 模型,且 critic 学得"短 critique"降推理成本(57-94 tokens vs 大 critic 数百~近千 tokens)。
- 🎯 对"探索-巩固"对标:**强支撑(形式同构)**。判定:ICRL 的 "critique 引导 solver 走出失败→把其中 critique-free 也成立的 token 内化进 solver" 与本项目"巩固/回轨(走偏后自选恢复分支并固化进参数)"在机制上**高度同构**;w_t(Eq.4)= "教师/critique 条件分布 → 学生/无条件分布"的 token 级安全迁移,正是 TSRD 想要的"把 teacher 脚手架下学到的恢复行为固化、且不强化对脚手架的依赖"。可借组件:① **w_t 重加权比**可直接平移为 OPD/TSRD 中"教师条件下产生的好 token,只迁移其无教师也成立的部分"——治理"学生依赖教师 prefix"的偏置;② utility-based 信号("因实际带来恢复成功而非看起来合理而奖励")可作 path-recovery 质量度量。竞品/缺口:ICRL 是 solver-critic 同 backbone 联合 RL,**不涉及 MTP/前瞻、非 teacher-student logit 蒸馏、critique 是自然语言而非稀疏脚手架信号**;TSRD 的"MTP 前瞻探针 + 教师当稀疏脚手架"是 ICRL 的留白。
- 🔭 开放问题/未来方向:【原文】学批评比学解题更难、信号更间接(§5.2),critic 学习的稳定化仍有空间;多角色/多 agent RL 是更广方向。【推断】把"critique-free 也成立"的判定从 token 级概率比升级为前瞻式(MTP 预测未来若干 token 是否能自走通)以更早决定迁移;非二值/无环境任务上如何定义 outcome 与 utility;共享 backbone 下 solver↔critic 能力耦合的崩溃边界刻画。

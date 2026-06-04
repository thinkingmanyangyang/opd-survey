icrl | ICRL: Learning to Internalize Self-Critique with Reinforcement Learning | 港科大(GZ)·南大·中山大·NUS·NTU·SAP·Microsoft Research(brick-pid) | 2026-05-13 · arXiv 2605.15224v1 · preprint | 主题线 L4 Agent/工具/多轮自进化(兼 L3 GRPO 变体) · 相关性 高

**原始论文**:https://arxiv.org/abs/2605.15224

## 一眼看懂
> 一句话导读:让同一个模型既当"解题手"又当"批评家",目标是把"有人指点才会做"练成"没人指点也会做",关键靠一个 token 级的概率比把"指点下学到的好招"安全搬到"无指点"的策略里。

- 🟦 TL;DR:同一个 backbone(主干模型)套两套 role prompt(角色提示词),同时扮演 solver(解题手)和 critic(批评家——解题失败后写一段自然语言批评指出问题),两个角色联合做 RL。目标是把"有 critique(批评)才能做对"内化成"无 critique 也能做对"。
  - 核心难点:critique-guided(有批评引导)的成功修订轨迹,采样自"看着批评写答案"的条件分布 \(\pi^S_\theta(\cdot\mid q,c)\)。直接拿它去更新"不看批评"的 critique-free 策略 \(\pi^S_\theta(\cdot\mid q)\),会有偏,反而强化模型对 critique 的依赖。
  - 解法是一个 **token 级"分布校准"重加权比** \(w_t=\dfrac{\pi^S_{\theta_{\mathrm{rollout}}}(y_t\mid q,y_{<t})}{\pi^S_{\theta_{\mathrm{rollout}}}(y_t\mid q,c,y_{<t})}\)(Eq.4)。直觉:某个 token 在"去掉批评后"本来概率就高 → 强迁移给 solver;某个 token 高度依赖批评上下文才产生 → 下调它的更新权重。
  - 另配两件套:**逐角色组内归一化**(Eq.5,solver/critic 各自算 group-relative advantage 组内相对优势;因为两者的输入前缀和奖励语义不同,不能混在一组里归一)+ **utility-based(看实效的)critic 奖励**(Eq.3,critic 因为"批评实际带来了修订成功"而非"批评听起来合理"才得分)。
- 最巧的一步:**token 级 reweight 比 \(w_t\)(Eq.4)**。
  - 抽掉它(把修订轨迹当普通 critique-free rollout 直接 GRPO 更新),就只是在强化"依赖 critique"的行为、无法内化。
  - 消融 Table 4:去掉 \(w_t\) 后平均分从 69.8 掉到 67.8(全 benchmark 一致下降)。
  - 为什么是支点:它是"把批评上下文产生的好行为安全搬到无批评策略"的唯一机制,直接对应论文标题的 "internalize"(内化)。

## 为什么做
> 一句话导读:前人要么让模型"临场自我纠错但纠错能力不写进脑子",要么用一个"冻住、不再进步"的批评家来做 RL;本文要补的空白是——怎样把"被批评引导出来的修订能力"真正内化进模型,而又不让它越来越离不开批评。

- 研究背景与来龙去脉:大模型"先解题、失败后自我批评、再修订"的范式有三条并行技术路线。
  - ① **推理时 self-correction(自我纠错)**:Self-Refine(Madaan 2023)、Reflexion(Shinn 2023)、CRITIC(Gou 2024)让同一模型在推理时产生反馈并重试。增益**只活在当次推理的上下文里、不写回参数**,撤掉反馈就失效。
  - ② **把批评用作 RL 信号**:Critique-GRPO(Zhang 2025b)在 GRPO 组内额外注入自然语言 critique 当反馈源做 RL。但它的 critic 是**静态/冻结**的——solver 进步后 critic 不再随之更新,反馈逐渐过时、变得无关。
  - ③ **多角色/多 agent RL**:MATPO(planner+subagent 角色分工训练)、ScalingInter(逐步增大交互步数预算)、GRPO/GSPO(单 agent、同前缀组内归一)。
  - 本文站在 GRPO(Guo 2025,作为 RL primitive 基本算法)+ Critique-GRPO(最近邻,提供"用 critique 做 RL"的思路)肩上。
- 三条路线各自的**具体**短板:
  - Self-Refine / Reflexion / CRITIC —— 改的是"有批评条件下"的推理时行为,**不保证内化进无批评策略**(下次没人给批评又错)。
  - Critique-GRPO —— critic 冻结、不与 solver 协同进化;且把"批评引导下产生的成功修订轨迹"直接当普通 rollout 去优化无批评策略,会引入**分布偏移偏置**(详见下文 Eq.4 直觉)。
  - MATPO / ScalingInter —— 是多角色 / 交互预算层面的工程,不触及"从有批评到无批评的能力迁移"。
  - Δ(本文独有问题):**没有任何前作研究"如何把有批评引导的修订安全内化进无批评 solver、且不强化对批评的依赖"**——这正是标题 "Internalize" 的落点。
- 解决的具体痛点(对应方法四件套):
  - ① 撤掉 critique 后,模型在同一道题又失败——能力没内化 → 需要把修订行为迁移进无批评策略。
  - ② 冻结 critic 会退化 → 需要 critic **可学**(共享 backbone + utility 奖励)。
  - ③ 有批评的成功轨迹来自条件分布 \(\pi^S_{\theta_{\mathrm{rollout}}}(\cdot\mid q,c)\),直接拿去更新 \(\pi^S_\theta(\cdot\mid q)\) **有偏**(会强化"依赖批评")。
  - ④ solver 初解 / critic 批评 / 修订解三类样本,输入前缀各异、奖励语义不同,**不能直接放一组里归一**。
- 动机链:批评能在推理时救场但不内化(现状)→ 直接拿修订轨迹训会强化依赖 + 冻结 critic 退化 + 多前缀奖励不可比(缺陷)→ 共享 backbone 联合训 solver/critic(治②)+ token 级分布校准 \(w_t\)(治①③)+ 逐角色优势(治④)+ utility critic 奖励(治②的信号质量),所以必须这四件套。
- 与最近邻工作的 Δ:vs **Critique-GRPO**(最近邻),三点差异:
  - (a) ICRL 让 critic 与 solver **共享同一 backbone 参数 θ、协同进化**(critic 随 solver 一起被更新、不退化)。
  - (b) 用 token 级 \(w_t\)(Eq.4)把修订**内化进无批评策略**(Critique-GRPO 是静态 critic、不做这层校准)。
  - (c) critic 奖励改为 **utility-based(看实效)**(Eq.3,因实际带来修订成功而非"听起来合理"而得分)。
  - 关键有用点:推理时不给批评也变强(Table 2 数学 + Table 1 agent 均超 Critique-GRPO);且 critic 学到"短而准"的可执行反馈(critic-swap,Table 3:8B 共享 critic 用 57 tokens 匹配 20B/32B 冻结 critic 的数百~近千 tokens)。

## 怎么做 + 靠不靠谱
> 一句话导读:一道题跑多个"解题→失败→批评→再解"的小回合,把三类样本(初解、批评、修订)分角色各自归一,再用那个 \(w_t\) 比值把修订里"无批评也成立"的部分搬给 solver,最后一份参数同时更新两个角色;实验证据扎实但单项相对最强对手只领先一两分。

- 方法流水线(Algorithm 1,完整到可复现):同一 backbone θ 实例化两个角色——solver \(\pi^S_\theta\)(role prompt \(p_S\))与 critic \(\pi^C_\theta\)(role prompt \(p_C\))。对每个 query \(q\),跑 \(G\) 个**独立**的 self-improving session(自我改进回合),每个 session 至多 \(K\) 轮:
  - **① 初解**:solver 采样 \(\tau_1\sim\pi^S_{\theta_{\mathrm{rollout}}}(\cdot\mid q)\),环境给 outcome reward(结果奖励)\(r(\tau_1)\in[0,1]\),放入 solver 组 \(\mathcal G^S(q)\)。
  - **② 失败→批评**:若 \(r_S(\tau_i)\neq 1\)(没做对),critic 采样 \(c_i\sim\pi^C_{\theta_{\mathrm{rollout}}}(\cdot\mid q,\tau_i)\)——把失败轨迹 \(\tau_i\) 喂进 critic,让它写批评。
  - **③ 修订**:solver 采样 \(\tau_{i+1}\sim\pi^S_{\theta_{\mathrm{rollout}}}(\cdot\mid q,c_i)\)(**条件在批评 \(c_i\) 上**);做对(\(r_S=1\))或耗尽 \(K\) 轮则停。一个 session 形如 \(\mathcal S=(\tau_1,c_1,\tau_2,\dots,c_{k-1},\tau_k),\ k\le K\)。修订对 \((q,c_i,\tau_{i+1})\) 入 \(\mathcal G^S(q)\),批评对 \((q,\tau_i,c_i)\) 入 critic 组 \(\mathcal G^C(q)\)。
  - **④ 逐角色优势**(Eq.5):solver/critic 各自在自己的组内算 group-relative advantage(组内相对优势)。
  - **⑤ token 级 reweight**(Eq.4):只对"有批评引导的修订"solver 轨迹算 \(w_t\)(把批评从 prompt 里移除后重新条件化、求概率比)。
  - **⑥ 联合更新**:最大化多角色 GRPO-style clipped 目标 \(J(\theta)\)(Eq.6);\(w_t\) 与上界 \(w_{\max}\) 双重界住方差,**同一份 θ 同时更新两个角色**。
- 核心算法/损失(真实形式 + 直觉,符号从 PDF §2.2/§3 抄准):
  - **总目标**(Eq.1):\(\ J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\!\left[r(\tau)\right]\)。
  - **GRPO 组内优势 + clipped 目标**(Eq.2):令 \(\hat A_i=\dfrac{r(\tau_i)-\mathrm{mean}_j\,r(\tau_j)}{\mathrm{std}_j\,r(\tau_j)+\delta}\),重要性比 \(\rho_t(\theta)=\dfrac{\pi_\theta(y_t\mid q,y_{<t})}{\pi_{\theta_{\mathrm{old}}}(y_t\mid q,y_{<t})}\),则
    \(\displaystyle J_{\mathrm{GRPO}}(\theta)=\mathbb E_{i,t}\!\left[\min\!\Big(\rho_t(\theta)\,\hat A_i,\ \mathrm{clip}\big(\rho_t(\theta),1-\epsilon,1+\epsilon\big)\hat A_i\Big)\right].\)
  - **utility-based critic 奖励**(Eq.3,直觉=因"实际带来修订成功"而非"听起来合理"而得分):
    \(\displaystyle r(c_i)=\begin{cases}1, & \text{若 }\tau_{i+1}\text{ 成功},\\[2pt] r(\tau_{i+1})-r(\tau_i), & \text{否则}.\end{cases}\)
    第二支(时间增量)**只在环境给非二值 dense reward 时非零**,否则退化为 0/1。
  - **核心:token 级分布校准重加权比**(Eq.4,全篇支点)。对一个自改进轮 \((\tau_i,c_i,\tau_{i+1})\),其中 \(\tau_i\sim\pi^S_{\theta_{\mathrm{rollout}}}(\cdot\mid q)\)、\(\tau_{i+1}\sim\pi^S_{\theta_{\mathrm{rollout}}}(\cdot\mid q,c_i)\):
    \(\displaystyle w_t=\begin{cases}\dfrac{\pi^S_{\theta_{\mathrm{rollout}}}(y_t\mid q,y_{<t})}{\pi^S_{\theta_{\mathrm{rollout}}}(y_t\mid q,c,y_{<t})}, & \text{critique-guided solver token},\\[10pt] 1, & \text{otherwise}.\end{cases}\)
    分子=**去掉 critique** 后该 token 的概率,分母=**有 critique** 时的概率。比值衡量"该 critique-guided token 在无 critique 分布下本就有多 plausible"。
  - **逐角色优势**(Eq.5,治"混合前缀不可比"):对每个 \(q\) 与角色 \(g\in\{S,C\}\),在角色专属组 \(\mathcal G^g(q)\) 内单独算 \(\ \hat A^g_i=\dfrac{r(\tau^g_i)-\mathrm{mean}_j\,r(\tau^g_j)}{\mathrm{std}_j\,r(\tau^g_j)+\delta}\)。
  - **最终多角色目标**(Eq.6):
    \(\displaystyle J(\theta)=\mathbb E_{\tau,t}\!\left[\min\!\big(w_t,\,w_{\max}\big)\cdot\min\!\Big(\rho_t(\theta)\,\hat A(\tau),\ \mathrm{clip}\big(\rho_t(\theta),1-\epsilon,1+\epsilon\big)\hat A(\tau)\Big)\right],\)
    其中**只有 critique-guided 修订 solver token 拿 \(w_t\)**,初解 solver 与 critic token 一律 \(w_t=1\);solver 的 \(\rho_t\) 在**去掉 critique 后**的 \((q,y_{<t})\) 上算,critic 的 \(\rho_t\) 在原 \((q,\tau_i)\) 上算。\(w_{\max}\) 是上界,防止 critique-free 概率远大于 critique-conditioned 时 \(w_t\) 爆炸、界住梯度方差。
- 模块如何咬合:
  - **solver↔critic 经 session 数据流互锁**:solver 失败产生 critic 的输入,critic 的输出又改变 solver 的修订分布。
  - **Eq.5 与 Eq.4 分工互补**:Eq.5 解决"三类前缀不可同组归一"(横向的角色隔离),Eq.4 解决"修订轨迹来自条件分布、迁移到无条件策略有偏"(纵向的分布校准)。
  - 两者都嵌在同一个 Eq.6 里,对同一份 θ 做一次联合策略梯度。
- 关键超参与默认值:并行 env server(环境服务器)32 个(端口 36001–36032,`env_nums=32`);group size 组大小 \(G\)、最大轮数 \(K\)、clip 阈 \(\epsilon\)、上界 \(w_{\max}\) 为方法超参(具体取值原文正文未列表,以仓内 config 为准〔待核〕);backbone Qwen3-4B/8B。
- 逐组件必要性(均有消融,Table 4):
  - **Re-weight ratio \(w_t\)(分布校准)**:去掉 → 69.8→67.8(降幅最大,全 benchmark 一致降)。证明修订轨迹不能当无批评 rollout 直接优化。
  - **Role-wise advantage(逐角色组内归一)**:去掉 → 69.8→68.4。证明 solver/critic 的异质奖励不能用单一尺度归一。
  - **Utility-based critic 奖励(Eq.3)**:无独立"去掉"消融,但通过 critic-swap(Table 3)间接证明——用此奖励训出的 8B 共享 critic 用 57 tokens 即匹配 20B/32B 冻结 critic 的 921/526 tokens 效果【原文 §5.3】。
- 关键机制直觉(\(w_t\) 怎么影响更新):
  - \(w_t<1\) → 该 token 重度依赖批评,保守更新(训练早期普遍 \(<1\),Fig.4)。
  - \(w_t\approx 1\) → 与无批评分布兼容,强迁移。
  - \(w_t>1\) → solver 本就更可能产此 token,上调。
  - \(w_{\max}\) 防权重爆炸;critic 奖励 = 修订成功给 1,否则给"solver reward 的时间增量"(只在 dense reward 稠密奖励时非零)。
- 实验与证据:
  - 任务:agentic(智能体类)—— ALFWorld 具身、WebShop 电商、HotpotQA/2Wiki/Bamboogle/MuSiQue 多跳 RAG;数学 —— MATH500/Minerva/Olympiad/AMC23/AIME24。Backbone:Qwen3-4B、Qwen3-8B。
  - baseline:Prompting(Qwen3 系 + Gemini-2.5/3-Flash)、GRPO、GSPO、ScalingInter、MATPO、**Critique-GRPO**(最强对手)。
  - agent(Table 1):Qwen3-4B 的 ICRL 平均 57.0(超 GRPO +7.8、超 Critique-GRPO +1.1);8B 平均 57.8(超 GRPO +5.0、Critique-GRPO +1.2)。ALFWorld/WebShop 最佳。
  - 数学(Table 2,Qwen3-8B):ICRL 平均 75.3(超 GRPO +7.0、Critique-GRPO +2.0);AIME24 从 50.0→65.1 提升明显;5 个里有 4 个超 Critique-GRPO(仅 AMC23 例外)。
  - 内化证据(关键三条):
    - ① §5.1 test-time(推理时)多轮 refinement,ICRL 第一轮就更强,且随轮次涨得更多(ALFWorld 第 3 轮 98%)。
    - ② §5.2 Fig.4,**\(w_t\) 随训练从 <1 稳步上升** → 越来越多修订 token 与无批评分布兼容 = solver 在内化修订模式。
    - ③ §5.3 critic-swap,8B 共享 critic 用极短批评即匹配/超过大模型的冻结 critic(57 vs 921 tokens)。
  - baseline 公平性:同 backbone、同环境;Critique-GRPO 是同类最强对手,逐项可比,公平。
  - "看着强但没答核心问题":数学上 AMC23 不及 Critique-GRPO;多跳 QA 多数 benchmark 非一致最优(只 2Wiki 双 backbone 最佳)——增益不均匀,但平均稳超。
- 假设与失效边界:
  - 【原文】依赖 outcome reward 可得(\(r(\tau)\in[0,1]\)、成功可验证);critic 的 dense 奖励项仅在环境给非二值 reward 时非零(Eq.3),否则退化为 0/1。
  - 【原文】§5.2 critic reward 曲线更低更抖,作者自承"学批评比学解题更难、与任务成功的关联更间接"。
  - 【推断】强依赖"失败→批评→修订"这个可迭代的自我改进循环 + 可执行环境(AgentGym);在无环境 / 无明确成功信号的开放场景下难用。
  - 【推断】solver 与 critic 共享 backbone,二者能力耦合;若 critic 学崩可能拖累 solver(论文靠 role-wise advantage + \(w_{\max}\) 稳住,但未给失败案例边界)。
- 祛魅总结【推断】:
  - 真贡献:① 明确提出并解决"critique-conditioned→critique-free 的分布偏移"这个被忽视问题,w_t 是干净可复用的校准件;② utility-based critic + 共享 backbone 让 critic 可学且"短而准";③ 内化证据链(test-time / w_t 上升 / critic-swap)较扎实。
  - 包装/可能高估:整体是 GRPO 的多角色 + token reweight 扩展,单项增益(尤其 vs Critique-GRPO 平均 +1~2 分)不算大;"internalize" 的强叙事主要靠 w_t 动态与 test-time 曲线支撑,部分 benchmark 并不一致领先。

## 结构化抽取
- 🎯 机制速览 6 轴:
  - **学什么信号**=outcome reward(solver)+ 批评下游修订的 utility(critic,Eq.3)+ token 级"无批评/有批评"概率比 \(w_t\)。
  - **改什么**=单一共享 backbone 参数(solver+critic 两角色联合)。
  - **何时改**=每 step,对有批评引导的修订 token 施加 \(w_t\)、逐角色归一优势。
  - **免梯度?**=否,GRPO-style clipped 策略梯度。
  - **记忆-技能生命周期**=无显式记忆库;"修订/纠错技能"经 \(w_t\) 从有批评行为内化进无批评策略参数(随训练 \(w_t\)↑)。
  - **防遗忘机制**=隐式——逐角色 advantage + \(w_{\max}\) 稳住联合优化;Table 4 未测遗忘,但 test-time 第一轮(无批评)持续走强,暗示内化不抹除直接解题能力。
- ⑦ 开源代码+框架/harness:https://github.com/brick-pid/ICRL (已 clone,真实可用,含中英 README)。
  - **框架=slime**(THUDM SGLang-native RL,repo 内置 `slime/`+`slime_plugins/`,`train.py`/`train_async.py`)。
  - 加 **AgentGym**(WooooDyy/AgentGym,提供 ALFWorld/WebShop/SearchQA 训练环境,默认起 32 个 env server 并行 rollout)。
  - 注:README 称 "will host the official implementation",代码已在仓内但措辞偏"即将正式发布"〔待核-以仓内现有实现为准〕。
- 💰 资源/成本与可扩展性:并行 rollout 需 32 个环境 server(端口 36001-36032),env_nums=32;backbone 4B/8B。具体 GPU 数/训练时长原文未说明。亮点:共享 backbone 省一个独立 critic 模型,且 critic 学得"短 critique"降推理成本(57-94 tokens vs 大 critic 数百~近千 tokens)。
- 🎯 对"探索-巩固"对标:**强支撑(形式同构)**。判定:ICRL 的 "critique 引导 solver 走出失败→把其中 critique-free 也成立的 token 内化进 solver" 与本项目"巩固/回轨(走偏后自选恢复分支并固化进参数)"在机制上**高度同构**;w_t(Eq.4)= "教师/critique 条件分布 → 学生/无条件分布"的 token 级安全迁移,正是 TSRD 想要的"把 teacher 脚手架下学到的恢复行为固化、且不强化对脚手架的依赖"。可借组件:① **w_t 重加权比**可直接平移为 OPD/TSRD 中"教师条件下产生的好 token,只迁移其无教师也成立的部分"——治理"学生依赖教师 prefix"的偏置;② utility-based 信号("因实际带来恢复成功而非看起来合理而奖励")可作 path-recovery 质量度量。竞品/缺口:ICRL 是 solver-critic 同 backbone 联合 RL,**不涉及 MTP/前瞻、非 teacher-student logit 蒸馏、critique 是自然语言而非稀疏脚手架信号**;TSRD 的"MTP 前瞻探针 + 教师当稀疏脚手架"是 ICRL 的留白。
- 🔭 开放问题/未来方向:【原文】学批评比学解题更难、信号更间接(§5.2),critic 学习的稳定化仍有空间;多角色/多 agent RL 是更广方向。【推断】把"critique-free 也成立"的判定从 token 级概率比升级为前瞻式(MTP 预测未来若干 token 是否能自走通)以更早决定迁移;非二值/无环境任务上如何定义 outcome 与 utility;共享 backbone 下 solver↔critic 能力耦合的崩溃边界刻画。

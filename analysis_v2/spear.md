spear | SPEAR: Learn the Ropes, Then Trust the Wins — Self-imitation with Progressive Exploration for Agentic RL | 腾讯优图 Youtu-Agent Team(通讯 yuleiqin/arthurtan@tencent.com) | Date 2025-09-22·arXiv v4 2025-12-07(2509.22601)·预印本 | 主题线 L4(Agent/工具/多轮自进化)·相关性 高(旁及 L1 用优质轨迹引导)

**原始论文**:https://arxiv.org/abs/2509.22601

## 一眼看懂
- 🟦 TL;DR:多轮 agent RL 里"机械地最大化策略熵"促探索很脆弱(环境反馈的低概率 token 累积致分布漂移→mode collapse 或 runaway divergence)。SPEAR 用"**课程化自模仿学习(SIL)+ 内在奖励塑形**":早期靠 tool-call 奖励促频繁工具交互做 skill-level 广探索,后期强化对成功轨迹的自模仿做 action-level 利用/巩固,把策略熵维持在"动态但受控"区间。在 ALFWorld/WebShop/Sokoban/AIME 上稳定提升 GRPO/GiGPO/Dr.BoT,额外开销仅理论 10%-25%【原文 abstract+§1】。
- 最巧的一步:**课程调度(curriculum)联合调"内在奖励"与"自模仿"两路权重**。抽掉课程,SIL 会在少量 buffer 轨迹上早期过拟合→熵坍缩、探索萎缩(Fig.3);而 tool-call 奖励无课程则后期与 outcome 奖励竞争、交互过长致精度下降(Fig.4,reward hacking)。课程是把"先学规矩(广探索)、再信战果(收敛利用)"操作化、避免两个极端的关键。

## 为什么做
- 研究背景:RL 是磨炼 LLM 长程、稀疏奖励 agent 任务中策略性工具使用能力的主流范式(基于 ReAct,应用含机器人导航/移动助手/web 导航/deep search/GUI),核心难题是探索-利用权衡【原文§1-§2】。
- 解决的具体痛点:① 纯熵控制在多轮 agent 中脆弱——环境反馈低概率 token 累积致严重分布漂移、mode collapse;多轮交互不确定性致持续熵增(runaway divergence)或塌缩,训练不稳;② cold-start SFT / RL+SFT 混合虽提稳,却限制策略发现 SFT 语料之外的新策略;③ vanilla SIL 用 replay buffer 做 off-policy 更新时优势需重算、且 off-policy 数据带来不稳/熵坍缩【原文§1+§2.4+§4.1】。
- 相关工作 & 各自不足:熵正则/最大化(机械、脆弱);cold-start/RL+SFT(限制新策略发现);GiGPO(group+step-level advantage,被当基线);ARPO(监控熵动态自适应分支轨迹);RAGEN(实例过滤+梯度塑形提稳);vanilla SIL/SAIL/GSIL(自模仿,但对 agent RL 诱发熵坍缩)【原文§2.2-2.4】。
- 动机链:agent RL 探索-利用难平衡 → 熵最大化机械且脆弱(多轮分布漂移致 collapse/divergence)→ cold-start SFT 又限制新策略 → 能否在"策略自身经验"引导下平滑调度何时探索何时利用、把熵维持在动态受控区间 → 课程化 SIL(早期广探索、后期收敛利用)+ 内在奖励 + 优势重校准 + 正则化【原文§1 core research question】。
- 与最近邻工作(vanilla SIL)的Δ:vanilla SIL 维护 replay buffer 仅存回报超基线的好轨迹做 off-policy 更新;SPEAR 三处改造:① **课程调度**跨阶段联合调内在奖励塑形与自模仿权重(skill→action 探索);② **优势重校准**(用 FIFO 基线 buffer 的 P50 百分位作保守基线、去 std 项,避免重算优势、过滤过时经验);③ **策略更新正则化**(covariance clipping 剔除"log-prob 与 advantage 增益高相关"的过自信 token,稳熵、抑 reward hacking)。为什么有用:分别对应 agent RL 不稳的三个来源(探索-利用时序 / off-policy 偏差 / 熵-hacking),且全程靠"agent 自己的奖励经验"无需专家模仿【原文§1+§4.2-4.3】。

## 怎么做 + 靠不靠谱
- 方法流水线(基于 group-based RL 类 GRPO,§4):输入 task → ① agent 与环境交互生成一组多轮工具交互轨迹 → ② 内在奖励塑形(Ri=Ri_outcome + µ·Ri_tool-call + Ri_format,Eq.6)+ group-based 优势估计 + on-policy 更新 → ③ 过滤优质轨迹(Â>0)入 replay buffer → ④ buffer 经优势重校准(Ãi_t=Ri−P50(DR),Eq.2;同时满足 Â>0 & Ã>0 才用,Eq.3)+ covariance 正则化做 self-imitation off-policy 更新 → ⑤ 课程跨阶段调两路权重:warm-up γ 控 SIL 项(Eq.5 J_Total=J_GRPO+γ·J̃_SIL)、µ 控 tool-call 奖励且随 step 衰减(Eq.6)→ 输出。基于 verl-agent 实现多轮 rollout【原文§4.2-4.4 Eq.1-6】。
- 逐组件必要性(**有消融** §5.3 Table 3,SI=Self-Imitation、IR=Intrinsic Reward):
  - **Self-Imitation(SIL)**:负责 action-level 利用,沿好轨迹学新策略而非随机游走;无它则缺利用、长程稀疏奖励下学习慢(§4.2)。
  - **Intrinsic Reward(tool-call 奖励)**:负责 skill-level 探索;无它 agent 因坏代码负反馈快速放弃 coding 退化为纯文本推理(Fig.4)——证其必要。
  - **课程调度**:无它则 SIL 早期过拟合致熵坍缩(Fig.3)、tool-call 奖励后期与 outcome 竞争致过长交互/reward hacking(Fig.4)——证其必要。
  - **优势重校准(P50 基线、去 std)**:处理 off-policy 偏差、过滤过时经验、缓解 group-norm 难度偏置(Eq.2 三好处);Table 1 含 GiGPO w/std vs w/o std 对照印证去 std。
  - **covariance clipping**:剔除过自信 token 稳熵(Fig.3 caption)。
  - 〔评估改善,纠 v1〕v1 担心"各组件独立消融需查附录"——本轮确认正文 §5.3 Table 3 已含 SI/IR 等组件消融,归因证据比 v1 判断的更充分(但课程超参敏感性仍需结合附录)。
- 关键机制/公式(直觉):① SIL 目标(Eq.1/3)只对 Â>0 的好轨迹做 GRPO 式更新,等于"反复回放成功经验"。② 优势重校准(Eq.2):因策略在迭代中不断改进,旧轨迹的观测回报与当前策略渐失配;维护最近 NDR 条 intra-group 基线的 FIFO buffer,取 50 百分位 P50 作"保守稳健"基线(高方差 agent RL 下比均值稳),并去掉 std 项(随 Dr.GRPO)→ 既校准相对增益又免额外采样计算。③ 课程(Eq.5/6):γ 给 SIL 项 warm-up(早期少模仿、保探索),µ 给 tool-call 奖励且随 step 衰减(早期促工具学习、后期聚焦精度防 hacking)。
- 实验与证据:
  - 数据集/模型:ALFWorld(文本具身)、WebShop(网购)、Sokoban(视觉推箱子,Qwen2.5-VL-3B)、AIME24/25(带 code interpreter,Qwen2.5-32B);模型 Qwen2.5-1.5B/7B-Instruct、32B、VL-3B【原文§5+Table 1/5】。
  - 关键数字【原文 abstract+Table,本轮 PDF 直读确认】:对 GRPO/GiGPO/Dr.BoT 提升 ALFWorld 最高 16.1%/5.1%/8.6%、WebShop 20.7%/11.8%/13.9%;Sokoban GRPO 67.1→86.7(+19.6%)、Dr.BoT 76.0→85.4(+9.4%);AIME24/25 对 Dr.BoT +3.8%/+6.1%。开销理论 +10%~25%、运行时可忽略。
  - baseline 公平吗:**自建强工业基线 Dr.BoT**(融合 DAPO clip-higher、Dr.GRPO 去长度/难度偏置等 bag-of-tricks),SPEAR 在其上仍有正增益——说明增益非来自弱基线,外部效度较好。
  - 看着强但没回答核心:SPEAR 是"课程+优势重校准+正则化+内在奖励"多组件叠加,虽有 §5.3 消融,但**课程超参(γ/µ 的阶段调度曲线)的敏感性**与各组件交互效应仍需结合附录判断;增益归因到单一机制仍偏粗。
- 假设与失效边界:
  - 显式【原文】:优势重校准假设"策略在迭代中持续改进"(故旧回报需校准,§4.2);tool-call 奖励是"双刃剑"(Fig.4,需课程调度才不致 reward hacking)。
  - 隐式【推断】:内在奖励依赖"tool-call 奖励"的设计,跨环境(非 code/python 工具)可迁移性未充分验证;课程跨阶段权重依赖阶段划分超参,自动化/自适应程度有限;主要在 Qwen 族 + 特定 agent benchmark 上验证。
  - **venue 状态**【待核→已澄清】:本轮 PDF 全文核查——唯一 ICLR 出现处为 ReAct[4] 的 ICLR 2023 引用,**无任何 SPEAR 自身的会议接收声明**(无 accepted/published/under review 字样);它是 2025-09-22 起的 arXiv 预印本(v4 2025-12-07)。引用时**勿标注会议接收**。
- 祛魅总结【推断】:真贡献=把"先学规矩、再信战果"的探索-利用直觉,用课程调度落到"内在奖励塑形 + 自模仿"两路权重的跨阶段调节上,并配优势重校准(免重算、P50 稳健基线)与 covariance 正则化稳熵;且自建强基线 Dr.BoT 证增益非来自弱对手。包装/高估:多组件配方,单一机制贡献被叠加效应稀释;"plug-and-play"对不同 agent 环境的真泛化(超出 ALFWorld/WebShop/Sokoban/code)未充分展开。低估:Dr.BoT 本身作为"工业 bag-of-tricks 强基线"的工程价值,以及"优势重校准用 P50 而非均值"这一对高方差 agent RL 的稳健性改进。

## 结构化抽取
- 🎯 机制速览6轴:
  - **学什么信号**:复合奖励(outcome 准确 + µ·tool-call + format,Eq.6)→ group-based 优势(经 P50 重校准);自模仿信号=回放 Â>0 的成功轨迹;熵作隐式被控量。
  - **改什么**:agent 策略 πθ 全参数(GRPO 类更新)。
  - **何时改**:多轮 agent RL 全程;课程跨阶段调 γ(SIL 权重)/µ(tool-call 奖励),early→late 从 skill-level 探索转 action-level 利用。
  - **免梯度?**:否,梯度策略优化(on-policy GRPO + off-policy SIL 联合)。
  - **记忆-技能生命周期**:有**显式 replay buffer**(存好轨迹及其奖励/优势 D={(τj,Rj,Âj)})——一种 episodic 经验记忆;另有 FIFO 基线 buffer DR(存最近 NDR 条 intra-group 基线供 P50 重校准)。技能=成功 tactics 经自模仿固化进参数;buffer 经"Â>0 & Ã>0"双过滤淘汰过时经验。
  - **防遗忘机制**:无显式跨任务防遗忘;但 self-imitation + replay buffer 本质是"防止遗忘已发现的成功行为"(反复回放好轨迹巩固);优势重校准过滤过时经验防"固化已劣化的旧策略";covariance clipping + KL-to-ref(Eq.4 含 β·D_KL(πθ||πref))防熵塌缩/偏离。
- ⑦ 开源代码+框架/harness:https://github.com/TencentYoutuResearch/SPEAR (v1 验证已 clone 约 96MB,commit 0edca96)。结构含 `verl/` 与 `verl-agent/` 两子目录(在 **veRL + verl-agent** 之上实现);verl-agent 提供 GiGPO 等 group-based RL 的多轮 agent 扩展。README 明确为 curriculum-based SIL 框架,给出 Self-imitation 配置项(`enable_trajectory_replay` 是否启用 self-imitation loss、replay buffer 最大轨迹数等),与论文方法一致。基线含 GRPO、GiGPO、自建 Dr.BoT。HF 模型 collection `yolay/spear-…`。代码可得、配置可定位。
- 💰 资源/成本与可扩展性:额外开销理论复杂度仅 +10%~25%,实际每迭代运行时开销可忽略(abstract+§ 末);replay buffer(ND)与基线 buffer(NDR)带少量显存。具体 GPU 规模正文未集中列(实验跨 1.5B~32B + VL-3B)。
- 🎯 对"探索-巩固"对标:**支撑/竞品(思路相通,可借组件)**。一句判定:SPEAR 与本项目"探索-巩固"在**高层范式上高度同构**——它把"探索=发现有效行为/路径(skill-level 广探索 + 沿 promising decision path 学新策略)"与"巩固=固化成功 tactics 进参数(self-imitation 回放好轨迹)"用课程显式调度,且强调"维持熵在动态受控区间"避免坍缩(对应本项目"偏向自己能走通的开头但不过早收敛")。可借组件:① replay buffer + self-imitation 作为"巩固已发现成功路径"的记忆-回放机制(可对接本项目"固化进记忆/技能");② 课程调度 γ/µ 作为探索→利用的时序旋钮;③ 优势重校准用 P50 稳健基线应对高方差;④ covariance clipping 稳熵。竞品/差距:SPEAR 是 **pure self-RL(无 teacher)**——靠 agent 自己的奖励经验,**没有 teacher 脚手架/稀疏接管**,也无"走偏后由 teacher 指引恢复分支"的 path-recovery(它靠正则化与课程被动稳住,而非 teacher 主动救);无 MTP 前瞻;"探索"是 skill/action 二分层的课程,非"关键步单点接管"。依据:§4.2 "learns novel strategies along the promising decision path instead of random walk and bifurcation" + replay buffer 自模仿 + 课程调度。
- 🔭 开放问题/未来方向:【原文】SPEAR 是 plug-and-play、可与现有算法组合(abstract+§2.2);课程把 skill-based 渐转 action-based 探索(可继续细化阶段划分)。【推断】把课程的"阶段划分超参"自动化/自适应(如按熵或发散动态触发阶段转换,而非预设曲线);引入 teacher 脚手架在高不确定/走偏的关键步主动给恢复方向(把 SPEAR 的"被动稳熵"升级为本项目的"主动 path-recovery");用 MTP 前瞻预测"这条工具交互会不会走偏"以提前调 γ/µ;把 replay buffer 从"episodic 轨迹"升级为可检索的技能库(对接 L5 记忆/技能&持续学习)。

RETURN: spear | 读到PDF?是(45页/112135字,abstract+§1+§4.2-4.4 Eq.1-6+§5.3消融+Table1/5 全直读;Sokoban 67.1→86.7/Dr.BoT基线/优势重校准P50/covariance clipping 均PDF核实;venue 全文确认无会议接收声明) | L4(Agent/工具多轮自进化,旁及L1) | 对标=支撑/竞品,高层范式同构(探索=skill广探索+沿promising path学新策略;巩固=replay buffer自模仿固化),可借replay-buffer记忆/课程γ-µ旋钮/P50稳健基线;差距在pure self-RL无teacher脚手架、无主动path-recovery、无MTP | 残留待核0(venue 已澄清为预印本无会议接收)

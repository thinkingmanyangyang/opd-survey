spear | SPEAR: Learn the Ropes, Then Trust the Wins — Self-imitation with Progressive Exploration for Agentic RL | 腾讯优图 Youtu-Agent Team(通讯 yuleiqin/arthurtan@tencent.com) | Date 2025-09-22·arXiv v4 2025-12-07(2509.22601)·预印本 | 主题线 L4(Agent/工具/多轮自进化)·相关性 高(旁及 L1 用优质轨迹引导)

**原始论文**:https://arxiv.org/abs/2509.22601

## 一眼看懂
- 🟦 TL;DR:多轮 agent RL 里"机械地最大化策略熵"促探索很脆弱(环境反馈的低概率 token 累积致分布漂移→mode collapse 或 runaway divergence)。SPEAR 用"**课程化自模仿学习(SIL)+ 内在奖励塑形**":早期靠 tool-call 奖励促频繁工具交互做 skill-level 广探索,后期强化对成功轨迹的自模仿做 action-level 利用/巩固,把策略熵维持在"动态但受控"区间。在 ALFWorld/WebShop/Sokoban/AIME 上稳定提升 GRPO/GiGPO/Dr.BoT,额外开销仅理论 10%-25%【原文 abstract+§1】。
- 最巧的一步:**课程调度(curriculum)联合调"内在奖励"与"自模仿"两路权重**。抽掉课程,SIL 会在少量 buffer 轨迹上早期过拟合→熵坍缩、探索萎缩(Fig.3);而 tool-call 奖励无课程则后期与 outcome 奖励竞争、交互过长致精度下降(Fig.4,reward hacking)。课程是把"先学规矩(广探索)、再信战果(收敛利用)"操作化、避免两个极端的关键(γ 用 cosine 从 0 升到 1、µ 用 cosine 从 1 衰到 0)。

## 为什么做
- 研究背景:RL 是磨炼 LLM 长程、稀疏奖励 agent 任务中策略性工具使用能力的主流范式(基于 ReAct,应用含机器人导航/移动助手/web 导航/deep search/GUI),核心难题是探索-利用权衡【原文§1-§2】。
- 解决的具体痛点:① 纯熵控制在多轮 agent 中脆弱——环境反馈低概率 token 累积致严重分布漂移、mode collapse;多轮交互不确定性致持续熵增(runaway divergence)或塌缩,训练不稳;② cold-start SFT / RL+SFT 混合虽提稳,却限制策略发现 SFT 语料之外的新策略;③ vanilla SIL 用 replay buffer 做 off-policy 更新时优势需重算、且 off-policy 数据带来不稳/熵坍缩【原文§1+§2.4+§4.1】。
- 相关工作 & 各自不足(本轮按 §2 引文链补全):
  - **§2.1 RL 算法谱系**:PPO(actor-critic+clip+KL 罚)→ GRPO(以 group-wise baseline 替 critic)→ **DAPO**(dynamic sampling + clip-higher 促探索稳训练)→ **Dr.GRPO**(去长度偏置与难度偏置)。SPEAR 把 DAPO/Dr.GRPO 等**industrial bag-of-tricks 融合成强基线 Dr.BoT**(§4.4),指出"裸组合各 trick 会冲突/紧耦合"。
  - **§2.2 LLM agent 优化**:**RAGEN**(实例过滤+梯度塑形提多轮稳定性,SPEAR 沿用其低方差组过滤)、**GiGPO**(group-level 优势 + 额外 step-level 优势,被当基线)、**ARPO**(监控 rollout 熵动态自适应分支轨迹)。共性短板:或靠手工启发式、或仍靠 trajectory-level 奖励。
  - **§2.3 RL 探索**:好奇心驱动(预测误差/新颖性)、count-based(伪计数 bonus)、skill acquisition(最大化互信息发现 options)、熵正则。短板:传统探索技术在 agent LLM 上易致 divergence(多轮交互本就增不确定)。
  - **§2.4 经验回放/SIL 谱系(本轮重点补全)**:**SIL**(Oh et al.,回放回报超基线的好经验)→ **SAIL**(扩到 off-policy/action-value)→ Tang et al.(证 SIL 的 return-based 更新给 bias-variance 折中、加速学习)→ **SILfD**(同时用外部 demo + 自身经验)→ **GSIL**(离线对齐框架,在 demo 上自模仿)。共性短板:SIL 在 agent RL 上**诱发熵坍缩**——这正是 SPEAR 用课程+covariance 正则要修的。
  - 精确差异:vs vanilla SIL——SPEAR 三处改造(课程调度 / 优势重校准 / covariance 正则),分别治"探索-利用时序 / off-policy 偏差 / 熵-hacking",且全程靠 agent 自身奖励经验、**无专家模仿**。
- 动机链:agent RL 探索-利用难平衡 → 熵最大化机械且脆弱(多轮分布漂移致 collapse/divergence)→ cold-start SFT 又限制新策略 → 能否在"策略自身经验"引导下平滑调度何时探索何时利用、把熵维持在动态受控区间 → 课程化 SIL(早期广探索、后期收敛利用)+ 内在奖励 + 优势重校准 + 正则化【原文§1 core research question + 假设:早期增熵利于 skill 探索、后期收敛熵利于 action 利用】。

## 怎么做 + 靠不靠谱
- 方法流水线(基于 group-based RL 类 GRPO,§4,输入→输出):输入 task → ① agent 与环境交互生成一组多轮工具交互轨迹 → ② 内在奖励塑形(复合奖励 \(R_i=R^i_{\text{outcome}}+\mu\cdot R^i_{\text{tool-call}}+R^i_{\text{format}}\),Eq.6)+ group-based 优势估计 + on-policy 更新 → ③ 过滤优质轨迹(\(\hat A>0\))入 replay buffer \(D=\{(\tau_j,R_j,\hat A_j)\}\) → ④ buffer 经优势重校准(\(\tilde A^i_t=R_i-P50(D_R)\),Eq.2;同时满足 \(\hat A>0\ \&\ \tilde A>0\) 才用,Eq.3)+ covariance 正则化(Eq.18-21)做 self-imitation off-policy 更新 → ⑤ 课程跨阶段调两路权重:warm-up γ(cosine 0→1)控 SIL 项、µ(cosine 1→0,前 200 步衰完)控 tool-call 奖励 → 输出。基于 **verl-agent** 实现多轮 rollout【原文§4.2-4.4 Eq.1-6+Appendix A.5/A.6】。
- 逐组件必要性(**有消融** §5.3 Table 3,SI=Self-Imitation、IR=Intrinsic Reward):
  - **Self-Imitation(SIL)**:负责 action-level 利用,沿好轨迹学新策略而非随机游走;无它则缺利用、长程稀疏奖励下学习慢(§4.2)。
  - **Intrinsic Reward(tool-call 奖励)**:负责 skill-level 探索;无它 agent 因坏代码负反馈(缺 import/未定义变量/缩进错/忘 print)快速放弃 coding 退化为纯文本推理(Fig.4)——证其必要。
  - **课程调度**:无它则 SIL 早期过拟合致熵坍缩(Fig.3)、tool-call 奖励后期与 outcome 竞争致过长交互/reward hacking(Fig.4)——证其必要。
  - **优势重校准(P50 基线、去 std)**:处理 off-policy 偏差、过滤过时经验、缓解 group-norm 难度偏置(Eq.2 三好处);Table 1 含 GiGPO w/std vs w/o std 对照印证去 std。
  - **covariance clipping**:剔除"log-prob 与 advantage 增益高相关"的过自信 token 稳熵(Fig.3 caption)。
  - **低方差组过滤**(§4.4/A.8,本轮补):去掉 intra-group reward std 最低的**底部 25%** 样本,保更新信息量(高 intra-group 方差=行为多样,对比利于利用)。
  - 〔评估改善,纠 v1〕v1 担心"各组件独立消融需查附录"——本轮确认正文 §5.3 Table 3 已含 SI/IR 等组件消融,归因证据比 v1 判断的更充分(但课程超参敏感性仍需结合附录)。
- 关键机制/公式(本轮据 PDF 正文 Eq.1-6 + Appendix A.5/A.6/A.9 补全,真符号 MathJax):
  - **SIL 目标(Eq.1)**:只对好轨迹做 GRPO 式更新——\(J^{\text{SIL}}_{\text{GRPO}}(\pi_\theta)=\mathbb{E}_{\{\tau_j\}\sim\{\pi_{\theta_{\text{old}}}\}}\sum_{j=1}^{N_D}J^j_{\text{GRPO}}\cdot\mathbb{1}(\hat A_j>0)\)(buffer \(D=\{(\tau_j,R_j,\hat A_j)\}\),轨迹可来自最近几步的旧策略)。
  - **优势重校准(Eq.2,off-policy 核心)**:维护最近 \(N_{D_R}\) 条 intra-group 基线的 **FIFO buffer** \(D_R=\{\bar R_j\}\),取 **50 百分位** 作保守稳健基线、并去 std 项(随 Dr.GRPO):\(\tilde A^i_t=R_i-P50(D_R)\)。三好处:基线随策略改进而变 / 用 \(\hat A_j>0\,\&\,\tilde A_j>0\) 双过滤淘汰过时经验 / 缓解 group-norm 难度偏置。
  - **off-policy SIL 目标(Eq.3-4)**:\(\tilde J^{\text{SIL}}_{\text{GRPO}}=\mathbb{E}\sum_j\tilde J^j_{\text{GRPO}}\cdot\mathbb{1}(\hat A_j>0\,\&\,\tilde A_j>0)\),其中 \(\tilde J^i_{\text{GRPO}}=\tfrac1T\sum_t[\min(r^i_t\tilde A^i_t,\mathrm{clip}(r^i_t,1-\epsilon,1+\epsilon)\tilde A^i_t)-\beta D_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}})]\)。
  - **总目标(Eq.5)**:\(J_{\text{Total}}(\pi_\theta)=J_{\text{GRPO}}(\pi_\theta)+\gamma\cdot\tilde J^{\text{SIL}}_{\text{GRPO}}(\pi_\theta)\)——on-policy GRPO 主项 + γ 加权的 off-policy SIL 项。
  - **复合奖励(Eq.6/15)**:\(R_i=R^i_{\text{outcome}}+\mu\cdot R^i_{\text{tool-call}}+R^i_{\text{format}}\),outcome 为二值 \(\{+1,-1\}\)(Eq.15),tool-call 正比工具调用轮数(促多轮交互)。
  - **课程 cosine 调度(Appendix A.6,Eq.13/14,本轮新补)**:
    \(\displaystyle \gamma=\begin{cases}\tfrac12\big(1-\cos(\pi\,\tfrac{t_{\text{iter}}}{T_{\text{warm-up}}})\big),&t_{\text{iter}}\le T_{\text{warm-up}}\\ 1,&t_{\text{iter}}>T_{\text{warm-up}}\end{cases} \qquad \mu=\begin{cases}\tfrac12\big(\cos(\pi\,\tfrac{t_{\text{iter}}}{T_{\text{decay}}})+1\big),&t_{\text{iter}}\le T_{\text{decay}}\\ 0,&t_{\text{iter}}>T_{\text{decay}}\end{cases}\)
    γ **cosine 升**(0→1):早期少模仿保探索、后期强自模仿固化;µ **cosine 降**(1→0,前 200 步衰完):早期促工具学习、后期聚焦精度防 hacking。
  - **covariance 正则化(Appendix A.5,Eq.18-21,本轮新补)**:对带正则的 SIL 目标 \(\tilde J^{\text{SIL-R}}_{\text{GRPO}}\) 乘逐 token mask \(M_j\)(Eq.18-19):
    \(\displaystyle \mathrm{Cov}(\log\pi_\theta(a^i_t|x,s^i_t),\tilde A^i_t)=\Big(\log\pi_\theta(a^i_t|x,s^i_t)-\tfrac1G\textstyle\sum_j\log\pi_\theta(a^j_t|\cdot)\Big)\Big(\tilde A^i_t-\tfrac1G\textstyle\sum_j\tilde A^j_t\Big)\)
    把"log-prob 与 advantage 协方差落在高区间 \([\omega_{lb},\omega_{ub}]\)"的过自信 token **uniform 采样出 \(N_{\text{clip}}=\lambda N_i\) 个并 mask 出 loss**(Eq.20),防其为博 advantage 而激进改 log-prob、压低熵。ω 经验取 top 20%/top 0.02% 协方差的整数;此 masking 引入的随机性还利于 RL 收敛。
  - **Theorem 1(Appendix A.9,surrogate 改进下界,本轮新补)**:warm-up γ 从 0 升 1 实现"向好响应分布的约束投影",在策略变化受 clip 界、优势无偏两假设下,
    \(\displaystyle J(\pi_{\theta_{t+1}})-J(\pi_{\theta_t})\ge\underbrace{\mathbb{E}_{a\sim\pi_{\theta_t}}[\tilde r(a)A^{\pi_{\theta_t}}(a)]}_{\text{GRPO improvement}}+\gamma(t)\underbrace{\mathbb{E}_{j\sim D}[\mathbb{1}_j\log r(a_j)]}_{\text{SIL improvement}}-\epsilon R_{\max}\)
    保证 SIL 项带来单调改进(\(\mathbb{1}_j=\mathbb{1}(\hat A_j>0\&\tilde A_j>0)\))。
  - 直觉:① SIL=反复回放成功经验沿 promising decision path 学新策略(非随机游走/分叉);② P50 基线比均值在高方差 agent RL 下更稳;③ 课程把探索→利用做成平滑 cosine 旋钮。
- 实验与证据:
  - 数据集/模型:ALFWorld(文本具身)、WebShop(网购)、Sokoban(视觉推箱子,Qwen2.5-VL-3B)、AIME24/25(带 code interpreter,Qwen2.5-32B);模型 Qwen2.5-1.5B/7B-Instruct、32B、VL-3B【原文§5+Table 1/5】。
  - 关键数字【原文 abstract+Table,本轮 PDF 直读确认】:对 GRPO/GiGPO/Dr.BoT 提升 ALFWorld 最高 16.1%/5.1%/8.6%、WebShop 20.7%/11.8%/13.9%;Sokoban GRPO 67.1→86.7(+19.6%)、Dr.BoT 76.0→85.4(+9.4%);AIME24/25 对 Dr.BoT +3.8%/+6.1%。开销理论 +10%~25%、运行时可忽略。
  - baseline 公平吗:**自建强工业基线 Dr.BoT**(融合 DAPO clip-higher、Dr.GRPO 去长度/难度偏置等 bag-of-tricks),SPEAR 在其上仍有正增益——说明增益非来自弱基线,外部效度较好。
  - 看着强但没回答核心:SPEAR 是"课程+优势重校准+正则化+内在奖励"多组件叠加,虽有 §5.3 消融,但**课程超参(\(T_{\text{warm-up}}/T_{\text{decay}}\)、γ/µ 调度曲线)的敏感性**与各组件交互效应仍需结合附录判断;增益归因到单一机制仍偏粗。
- 假设与失效边界:
  - 显式【原文】:优势重校准假设"策略在迭代中持续改进"(故旧回报需校准,§4.2);tool-call 奖励是"双刃剑"(Fig.4,需课程调度才不致 reward hacking);Theorem 1 假设策略变化受 clip 界、优势无偏。
  - 隐式【推断】:内在奖励依赖"tool-call 奖励"的设计,跨环境(非 code/python 工具)可迁移性未充分验证;课程跨阶段权重依赖阶段划分超参(\(T_{\text{warm-up}}/T_{\text{decay}}\)),自动化/自适应程度有限;主要在 Qwen 族 + 特定 agent benchmark 上验证。
  - **venue 状态**【待核→已澄清】:本轮 PDF 全文核查——唯一 ICLR 出现处为 ReAct[4] 的 ICLR 2023 引用,**无任何 SPEAR 自身的会议接收声明**(无 accepted/published/under review 字样);它是 2025-09-22 起的 arXiv 预印本(v4 2025-12-07)。引用时**勿标注会议接收**。
- 祛魅总结【推断】:真贡献=把"先学规矩、再信战果"的探索-利用直觉,用课程调度(cosine γ/µ)落到"内在奖励塑形 + 自模仿"两路权重的跨阶段调节上,并配优势重校准(免重算、P50 稳健基线)与 covariance 正则化稳熵;且自建强基线 Dr.BoT 证增益非来自弱对手。包装/高估:多组件配方,单一机制贡献被叠加效应稀释;"plug-and-play"对不同 agent 环境的真泛化(超出 ALFWorld/WebShop/Sokoban/code)未充分展开;Theorem 1 的下界依赖较强假设。低估:Dr.BoT 本身作为"工业 bag-of-tricks 强基线"的工程价值,以及"优势重校准用 P50 而非均值"这一对高方差 agent RL 的稳健性改进。

## 结构化抽取
- 🎯 机制速览6轴:
  - **学什么信号**:复合奖励(outcome 准确 + µ·tool-call + format,Eq.6)→ group-based 优势(经 P50 重校准);自模仿信号=回放 \(\hat A>0\) 的成功轨迹;熵作隐式被控量(covariance clipping + 课程间接稳)。
  - **改什么**:agent 策略 \(\pi_\theta\) 全参数(GRPO 类更新)。
  - **何时改**:多轮 agent RL 全程;课程跨阶段调 γ(SIL 权重,cosine 0→1)/µ(tool-call 奖励,cosine 1→0),early→late 从 skill-level 探索转 action-level 利用。
  - **免梯度?**:否,梯度策略优化(on-policy GRPO + off-policy SIL 联合)。
  - **记忆-技能生命周期**:有**显式 replay buffer** \(D=\{(\tau_j,R_j,\hat A_j)\}\)(存好轨迹及其奖励/优势)——一种 episodic 经验记忆;另有 FIFO 基线 buffer \(D_R\)(存最近 \(N_{D_R}\) 条 intra-group 基线供 P50 重校准)。技能=成功 tactics 经自模仿固化进参数;buffer 经"\(\hat A>0\,\&\,\tilde A>0\)"双过滤淘汰过时经验。
  - **防遗忘机制**:无显式跨任务防遗忘;但 self-imitation + replay buffer 本质是"防止遗忘已发现的成功行为"(反复回放好轨迹巩固);优势重校准过滤过时经验防"固化已劣化的旧策略";covariance clipping + KL-to-ref(Eq.4 含 \(\beta D_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}})\))防熵塌缩/偏离。
- ⑦ 开源代码+框架/harness:https://github.com/TencentYoutuResearch/SPEAR (v1 验证已 clone 约 96MB,commit 0edca96)。结构含 `verl/` 与 `verl-agent/` 两子目录(在 **veRL + verl-agent** 之上实现);verl-agent 提供 GiGPO 等 group-based RL 的多轮 agent 扩展。README 明确为 curriculum-based SIL 框架,给出 Self-imitation 配置项(`enable_trajectory_replay` 是否启用 self-imitation loss、replay buffer 最大轨迹数等),与论文方法一致。基线含 GRPO、GiGPO、自建 Dr.BoT。HF 模型 collection `yolay/spear-…`。代码可得、配置可定位。
- 💰 资源/成本与可扩展性:额外开销理论复杂度仅 +10%~25%,实际每迭代运行时开销可忽略(abstract+§ 末);replay buffer(\(N_D\))与基线 buffer(\(N_{D_R}\))带少量显存。具体 GPU 规模正文未集中列(实验跨 1.5B~32B + VL-3B)。
- 🎯 对"探索-巩固"对标:**支撑/竞品(思路相通,可借组件)**。一句判定:SPEAR 与本项目"探索-巩固"在**高层范式上高度同构**——它把"探索=发现有效行为/路径(skill-level 广探索 + 沿 promising decision path 学新策略)"与"巩固=固化成功 tactics 进参数(self-imitation 回放好轨迹)"用课程显式调度,且强调"维持熵在动态受控区间"避免坍缩(对应本项目"偏向自己能走通的开头但不过早收敛")。可借组件:① replay buffer + self-imitation 作为"巩固已发现成功路径"的记忆-回放机制(可对接本项目"固化进记忆/技能");② 课程 cosine γ/µ 作为探索→利用的时序旋钮(比硬阶段切换更平滑);③ 优势重校准用 P50 稳健基线应对高方差;④ covariance clipping 稳熵。竞品/差距:SPEAR 是 **pure self-RL(无 teacher)**——靠 agent 自己的奖励经验,**没有 teacher 脚手架/稀疏接管**,也无"走偏后由 teacher 指引恢复分支"的 path-recovery(它靠正则化与课程被动稳住,而非 teacher 主动救);无 MTP 前瞻;"探索"是 skill/action 二分层的课程,非"关键步单点接管"。依据:§4.2 "learns novel strategies along the promising decision path instead of random walk and bifurcation" + replay buffer 自模仿 + 课程调度(Eq.13/14)。
- 🔭 开放问题/未来方向:【原文】SPEAR 是 plug-and-play、可与现有算法组合(abstract+§2.2);课程把 skill-based 渐转 action-based 探索(可继续细化阶段划分)。【推断】把课程的"阶段划分超参 \(T_{\text{warm-up}}/T_{\text{decay}}\)"自动化/自适应(如按熵或发散动态触发阶段转换,而非预设 cosine 曲线);引入 teacher 脚手架在高不确定/走偏的关键步主动给恢复方向(把 SPEAR 的"被动稳熵"升级为本项目的"主动 path-recovery");用 MTP 前瞻预测"这条工具交互会不会走偏"以提前调 γ/µ;把 replay buffer 从"episodic 轨迹"升级为可检索的技能库(对接 L5 记忆/技能&持续学习)。

RETURN: spear | 读PDF?是(45页/112135字,abstract+§1+§2.1-2.4 全谱系+§4.2-4.4 Eq.1-6+Appendix A.5/A.6/A.9 Eq.13/14/18-21/Thm1+§5.3消融+Table1/5 全直读;Sokoban 67.1→86.7/Dr.BoT基线/P50重校准/covariance clipping/cosine 调度 均PDF核实;venue 全文确认无会议接收) | 加厚?是(补全 Eq.1-6 全套真符号公式+课程 cosine γ/µ 调度 Eq.13/14+covariance clipping Eq.18-21 含 Cov 定义+Theorem 1 surrogate 下界+低方差组过滤 bottom-25%;related-work SIL 谱系 SAIL/SILfD/GSIL 与 agent RL RAGEN/GiGPO/ARPO 精确定位) | LaTeX 公式条数 7(SIL目标Eq1/优势重校准Eq2/off-policy SIL目标Eq3-4合/总目标Eq5/复合奖励Eq6/cosine调度Eq13-14/covariance Eq21+Thm1下界) | 待核 0(venue 已澄清为预印本无会议接收)

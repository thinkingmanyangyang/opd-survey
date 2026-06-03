openclaw_rl | OpenClaw-RL: Train Any Agent Simply by Talking | Gen-Verse / Princeton（Yinjie Wang*, Xuyang Chen*, Xiaolong Jin*, Mengdi Wang†, Ling Yang† 通讯） | 2026-05-11·arXiv·v2（Blog yinjjiew.github.io/projects/openclawrl1） | 主题线 L4(Agent/工具·多轮自进化)+L1(OPD)·相关性 高

**原始论文**:https://arxiv.org/abs/2603.10165 （代码 github.com/Gen-Verse/OpenClaw-RL）

## 一眼看懂
- 🟦 TL;DR:把 agent 每次交互产生的 "next-state 信号"(用户回复/工具输出/终端·GUI 状态变化)当作在线实时学习源回收,从中抽两类互补信号——**evaluative**(标量、更频繁)与 **directive**(token-level、更富信息但稀疏)——在**一次 hybrid RL 更新里统一**;并用 **overlap-guided hint selection + logprob-diff clip** 稳定 teacher-student 失配下的 OPD,使"agent 越被使用越变强"。【原文】abstract+§3
- 最巧的一步:**overlap-guided hint selection**——在候选纠正 hint 中,选其诱导的 teacher 分布与学生 top-k token 重叠最大者。抽掉它——OPD 的核心失败模式就是 teacher-student 分布失配:当 teacher 把概率质量放在学生近零密度的 token 上,off-policy importance ratio 爆炸、梯度不稳(Li 2026);而"选重叠最大的 hint"让 teacher 已在学生高概率区与之一致,蒸馏在学生自己的高密度区内把它推向 teacher,使 ρ_v≈1、更新稳。配合 Δ-clip(把每 token logprob 差 bound 在 C)双重 bound surrogate。

## 为什么做
- 研究背景:LLM agent 已广泛部署(终端/GUI/SWE/tool-call);交互数据被用于改进 framework、构建 memory(Mem0/Cognee/Letta)、产训练数据,但鲜有把它当**在线、实时**学习源。现有 agentic RL 基础设施(slime/OpenRLHF/AReaL 等)假设批量预采集数据。【原文】§1
- 解决的具体痛点:① 基础设施——RL server 需灵活对接用户多样且演进的 agent 框架,且优化须异步、不阻塞推理使用;② 方法学——RLVR 仅能用标量奖励、无法把 directive(纠正性)信号转成策略梯度;OPD/hindsight 虽能用结构化纠正但只在固定数据集上操作;OPD 还因 teacher-student 分布失配训练不稳/失效,低质量 hint 时更严重。【原文】§1
- 相关工作 & 各自不足:RLVR(只标量);on-policy 蒸馏(Agarwal 2024、SDPO Hübotter 2026、SDFT Shenfeld 2026,能用结构化纠正但固定数据集);hindsight relabeling(加纠正信息进 context 提升输出,但固定数据集);并发 Buening 2026 直接用 next-state 提示在线改策略,但纠正 hint 仍隐含在 prompt 中(非显式信号)。本文把 directive 信号显式抽成 token-level hint。【原文】§1+§5.4
- 动机链:每个 next state 既隐含评价(re-query=不满、通过测试=成功、错误 trace=失败)又常携带指令信息("你应该先检查文件"在 token 级指明该怎么改)→ RLVR 只能用前者、纯 OPD 只能用后者,两者都没吃满 s_{t+1} → 所以要回收两类互补信号、统一进一次在线更新,并稳定其中的 OPD 蒸馏。【原文】§1+§3.1
- 与最近邻工作的Δ:vs 纯 RLVR——多了 directive(token-level)路径;vs 纯 OPD——多了 evaluative 兜底且**在线实时**(非固定数据集)、并解决失配稳定性;vs memory 类(Mem0/Cognee)——是参数级在线优化而非外部记忆。关键点:① 首个统一 personal + 四类 general agent(terminal/GUI/SWE/tool-call)的开源 RL 框架;② hybrid 把"频率高的标量"与"信息密的 token 分布"在单 loss 里互补。【原文】abstract+Table 2

## 怎么做 + 靠不靠谱
- 方法流水线(输入→输出):
  - **基础设施**:RL server 把策略 π_θ 包成 stateless 推理 API;用户 agent 框架(个人设备/终端/云)经 HTTP 流式回传交互数据;请求分 main-line turn(可训)/side turn(只前向);独立异步 PRM/Judge server 从 (a_t, s_{t+1}) 抽 evaluative 分 + directive hint(不阻塞推理,可用更强模型多投票);slime 协调四解耦异步组件(环境 server / PRM / Megatron 训练 / SGLang serving),零 serving 中断、graceful weight update。
  - **方法**:① evaluative=PRM 投 m 次取多数 r_t∈{+1,−1,0};② directive=PRM 先判 s_{t+1} 是否含有意义纠正,有则蒸成 hint h(放 [HINT_START]…[HINT_END])、append 得 s^h_t、同模型在 hint-augmented prompt 下出 teacher 分布 π_T;③ overlap-guided 选 h*(序列级或 token 级,选 top-k 重叠最大);④ top-k OPD loss(Eq.1)限制在词表子集 S_i=学生 top-k,A_v=Δ_v·w_v(w_v 把优势集中到学生可能采样的 token,Δ_v=clip(ℓ_T−ℓ_old,−C,+C)),clipped-surrogate 形式;⑤ hybrid:L_i = w_RL·L_GRPO_i + w_OPD·L_OPD_i(默认各 1)。【原文】§2+§3.1-3.2
- 逐组件必要性:
  - **overlap-guided hint selection**:有消融(Table 4,sequence-optimal 12.5 / token-optimal 12.4 均优于 random 16.1)——必要,且序列级在 general agentic 更稳。
  - **logprob-difference clip**:§4.9 标题即"Log-Probability Difference Is Vital for Stability"——必要(bound 每 token 优势)。
  - **hybrid(vs 纯 RLVR / 纯 OPD)**:有对照(Table 3,Hybrid 10.3 < GRPO 14.1 < OPD 29.7;Fig.6 Retool/RLVR 上 Hybrid 优于 PRM+Outcome 与 Outcome)——必要,纯 OPD 反而最差(29.7),佐证单靠 directive 不稳、需 evaluative 兜底。
  - **step-wise process reward(general agentic)**:§4.5,tool-call/GUI 集成 outcome+process 达 0.25/0.33 vs outcome-only 0.19/0.31——必要(长 horizon 稀疏)。明确建立在 RLAnything(open_agentrl)之上,沿用 o+Σr_i/m。
  - **PRM 判"是否值得抽 hint"**:必要(强抽会产生低质量 hint 破坏训练)。
- 关键机制/公式(直觉):overlap O[h,i]=|S^q_i ∩ S^p_{i,h}|(学生 top-k 与 teacher top-k 交集大小);选 h* 使重叠最大 → teacher 与学生在词表 support 上已一致 → 蒸馏在学生高密度区内移动而非推向陌生 token。OPD loss(Eq.1)的 ρ_v≈1(因 overlap 选择)+ Δ-clip 同时 bound surrogate 两个因子。附录 C 证 OPD 目标即 token-level KL。直觉:directive 用 OPD 的 token-level KL 承载(teacher 条件于纠正 hint),evaluative 用 GRPO 标量,两者共轨迹共更新。【原文】§3.2+Appendix C
- 实验与证据:
  - **Personal**(§4.1):LLM 模拟不同职业用户(学生避免 AI 痕迹/助教要详细评分/教师要友好评语)在 GSM8K 任务上用 OpenClaw,度量"对齐各用户偏好所需最少 sessions"(连续 3 session 满足偏好即达标,第一条消息硬编码不透露偏好)。policy & reward 均 Qwen3-4B-Thinking-2507,用户用 Qwen3-32B 模拟;lr 1e-5、C=1、每 16 样本触发一步。关键数字(Table 3,5 trials 均值):joint 优化下 Hybrid **10.3** sessions,优于 GRPO 14.1、OPD 29.7、Mem0 14.5、Cognee 14.9;separate 优化下 Hybrid 15.0 仍最优。注:joint 优化放大 RL 增益而 memory 类几乎不变。
  - **General**(§4.2/4.5):四环境模型 terminal=Qwen3-8B、GUI=Qwen3VL-8B-Thinking、SWE=Qwen3-4B、tool-call=Qwen3-4B-SFT(Retool-4B);训练数据 SETA RL / OSWorld-Verified / SWE-Bench-Verified / DAPO RL;GUI 评在训练集(去 chrome 与 multi-apps),tool-call 评 AIME 2024,terminal/SWE 报窗口内平均 rollout-task acc;128/64/64/32 并行环境;tool-call/GUI 集成奖励 0.25/0.33 vs outcome-only 0.19/0.31。超参:general lr 1e-6、KL 0.01、clip 0.2/0.28、每步 GUI/SWE 采 8 任务、terminal 16、tool-call 32,各 8 samples;GUI/SWE/terminal 最大交互步 30/20/10。
  - baseline 公平性:personal 高度依赖 LLM 模拟用户与硬编码偏好检测器(粗体/列表/长度/暖词),度量"对齐效率"而非真实能力提升,生态效度有限;GUI 部分直接在训练集上评。【原文】Table 3+Table 4+§4.5
- 假设与失效边界:
  - 【原文】§3.1+§3.2:directive 信号只在"PRM 判有意义纠正"的子集触发(稀疏);teacher 由同模型条件 hint 得到,hint 质量直接控制失配大小;Δ-clip C 是关键超参(personal C=1,RLVR C=2)。
  - 【原文】§4.5:hosting PRM 需额外资源(相对 outcome-only)。
  - 【推断】personal 评测的生态效度有限(模拟用户+硬编码检测器,度量风格对齐而非能力);general 部分环境在训练集上评;是否在真实人类用户上验证未知。〔待核〕"HuggingFace Daily Papers #1"一说——已确认论文正文不含此声明(应出自仓库 README),保留为待核。
- 祛魅总结:【推断】真贡献=① 把"next-state 既含评价又含指令"这一观察做成**两路互补信号 + hybrid 单 loss**,且用 overlap-guided hint selection 漂亮地解决了 OPD 在线失配稳定性(这是把 OPD 真正用进 agentic 在线训练的关键工程+方法贡献);② 首个统一 personal + 四类 general agent 的开源异步 RL 基础设施(server-client、零中断),工程价值高。被适度高估:① "越用越强"由 personal 设置度量,但该设置用模拟用户+硬编码偏好检测器,生态效度有限,数字(10.3 sessions)是相对这套人造度量;② 纯 OPD 最差(29.7),hybrid 的增益主要来自"加了 evaluative 兜底",directive 单路并不稳——所以核心其实是"用 evaluative 稳住 directive";③ step-wise reward 部分直接沿用 RLAnything,非本文原创。

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号:evaluative(PRM 标量,序列级)+ directive(hint-conditioned teacher 的 token-level 分布,即 OPD 的 token-level KL);hybrid 混合。
  - 改什么:学生策略 π_θ(Megatron 训练);PRM 是独立 frozen/更强模型(不训)。
  - 何时改:在线、每 main-line turn;但 directive 仅在 PRM 判"有有意义纠正"的稀疏 turn 触发;weight update 在 well-defined 边界推给 serving。
  - 免梯度?否——hybrid clipped-surrogate 梯度优化;但 hint selection(overlap argmax)与 PRM 判定是免梯度的离散选择。
  - 记忆-技能生命周期:与 memory 类(Mem0/Cognee)对比——本文是**参数级在线巩固**(把偏好/纠正写进权重)而非外部记忆;集成社区 SDFT/SDPO 到 openclaw-opd/。
  - 防遗忘机制:KL 系数 0.01(general)作 reference 锚;Δ-clip + overlap 选择把更新限制在学生高密度区(隐式防过度偏移);多用户 joint 优化共享一个模型。
- ⑦ 开源代码+框架/harness:github.com/Gen-Verse/OpenClaw-RL(已 clone,~73M)。框架 **slime(THUDM/slime)异步 RL** 为核心,四解耦异步组件:环境 server、PRM/Judge(SGLang/API)、**Megatron**(策略训练)、**SGLang**(策略 serving);server-client 把模型包成 OpenAI 兼容 API(经 OpenClaw 插件),HTTP 流式回传在线训练,零中断、graceful weight update;也支持 Tinker 云端/本地 GPU 与 LoRA。三范式:Binary RL / OPD / Combine(Hybrid)。关键子目录:openclaw-opd/(OPD,含 topk 蒸馏 loss、qwen3/qwen35 launcher、集成 SDFT/SDPO)、openclaw-rl/、openclaw-combine/、openclaw-test/、openclaw-tinker/、openclaw-fireworks/;Track-2:gui-rl/、terminal-rl/、swe-rl/、toolcall-rl/;含 slime/、Megatron-LM/、extensions/。(注:仓库有 qwen3.5-4B 配置但非论文报告模型。)【原文/仓库】
- 💰 资源/成本与可扩展性:personal lr 1e-5、每 16 样本一步、Qwen3-4B-Thinking;general lr 1e-6、128/64/64/32 并行环境;PRM 作独立 server(可用更强模型多投票,但需额外资源);异步设计使 serving 零中断。规模上限实测 8B 级。【原文】§4.1+§4.2+§4.5
- 🎯 对"探索-巩固"对标:**强支撑(把 OPD 用进在线 agentic,且专治失配稳定性)**。一句判定:OpenClaw-RL 把 OPD 显式作为 hybrid RL 的 "directive 信号",并用 overlap-guided hint selection 解决"teacher-student 失配"——这正对应本课题"巩固/回轨"里"on-policy 自选 + 选能走通的恢复分支"的稳定化难题。依据:§3.2 失配诊断 + overlap 选择。可借组件:① **overlap-guided hint/分支选择**——"选诱导 teacher 与学生高概率区重叠最大的纠正"=直接可用于"path-recovery 时选学生自己能走通的恢复分支"(对应 idea 里"偏向自己能走通的开头");② **Δ-clip(logprob 差 bound)+ top-k 子集 OPD loss**——对应"forward-hard/backward-soft"式的 bounded 梯度,可迁到 MTP/OPD;③ **evaluative 稳住 directive 的 hybrid 思路**——稀疏关键 directive 信号需要稠密标量兜底,对"sparse_critical 单点接管"有借鉴(单点 path-recovery 也可能需要全局 reward 兜底);④ next-state 当在线学习源 = "探索=发现有效行为/路径"的在线回收。缺口:① 无 MTP/前瞻(hint 由 PRM 事后抽,非前瞻预测走偏);② directive hint 来自外部 PRM 对 next-state 的解读,而非"看过答案的自己"主动选路;③ 巩固靠在线 KL 软锚,无显式"防遗忘进技能库/记忆"机制(明确与 memory 类对立);④ personal 评测生态效度限制了"巩固=真能力提升"的证据强度。
- 🔭 开放问题/未来方向:
  - 【原文】§6/§4.5:扩到更多真实 agent 部署;PRM 资源开销优化;长 horizon 过程奖励的更好整合。
  - 【推断】把 overlap-guided hint selection 从"事后选 PRM 抽的 hint"升级为"用 MTP 前瞻预测学生将走偏、主动构造/选择能走通的恢复分支"(前瞻探针 + path-recovery);把"PRM 判是否值得抽 hint"做成"是否到了需接管的关键步"判据(sparse_critical);在真实人类用户而非模拟用户上验证"越用越强",以建立巩固=真能力提升的证据。

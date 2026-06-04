openclaw_rl | OpenClaw-RL: Train Any Agent Simply by Talking | Gen-Verse / Princeton（Yinjie Wang*, Xuyang Chen*, Xiaolong Jin*, Mengdi Wang†, Ling Yang† 通讯） | 2026-05-11·arXiv·v2（Blog yinjjiew.github.io/projects/openclawrl1） | 主题线 L4(Agent/工具·多轮自进化)+L1(OPD)·相关性 高

**原始论文**:https://arxiv.org/abs/2603.10165 （代码 github.com/Gen-Verse/OpenClaw-RL）
> 已核更正(勿回退):①论文报告模型为 **Qwen3-4B-Thinking-2507**(personal)与 Qwen3-8B/Qwen3VL-8B-Thinking/Qwen3-4B/Qwen3-4B-SFT(general)——**PDF 全文无 "Qwen3.5"/"qwen3.5"/"35-4B" 任何字样**,仓库里的 qwen3.5-4B 配置非论文报告模型(保持已修正"伪造模型规格"结论);②**PDF 全文无 "HuggingFace Daily Papers #1"** 字样,该说法出自仓库 README,保留为〔待核〕。

## 一眼看懂
> 一句话导读:agent 每跟用户/工具/终端交互一次,得到的"下一个状态"里其实藏着两种信息——既有"满不满意"的评价(用户重问=不满、测试通过=成功),也有"该怎么改"的具体指令("你应该先检查文件")。本文把这两种信号都在线回收、揉进一次 RL 更新;难点是后者要走 OPD(在线蒸馏),而 OPD 容易因师生分布对不上而训崩,所以本文用"挑重叠最大的 hint + 给 logprob 差设上限"把它稳住,从而做到"agent 越用越强"。

- 🟦 TL;DR:核心做法是,把 agent 每次交互产生的 "next-state 信号"(用户回复 / 工具输出 / 终端·GUI 状态变化)当作一个在线实时的学习源回收回来,从中抽出两类互补信号:
  - **evaluative**:标量、更频繁;
  - **directive**:token-level、信息更丰富、但稀疏。
  然后在**一次 hybrid RL 更新里把两者统一**;并用 **overlap-guided hint selection + logprob-diff clip** 这两招,稳住 teacher-student 失配下的 OPD,从而实现"agent 越被使用越变强"。【原文 Abstract+§3】
- 最巧的一步:**overlap-guided hint selection(用重叠度来挑纠正提示)**——在多个候选的纠正 hint 中,挑那个"它诱导出的 teacher 分布与学生 top-\(k\) token 重叠最大"的。为什么这一步关键?OPD 的核心失败模式就是 teacher-student 分布失配:当 teacher 把概率质量放在学生几乎零密度的 token 上时,off-policy 的 importance ratio 会爆炸、梯度不稳(Li 2026)。而"选重叠最大的 hint",意味着 teacher 在学生的高概率区就已经和学生一致了,于是蒸馏是在学生自己的高密度区内把它往 teacher 推,使 \(\rho_v\approx1\)、更新就稳。这一招再配合 \(\Delta\)-clip(把每个 token 的 logprob 差 bound 在 \(C\) 之内),就对 surrogate 形成了双重 bound。【原文 §3.2 + Eq.1】

## 为什么做
> 一句话导读:agent 跟人交互产生的数据,以前要么拿去改框架、要么存进记忆库,很少有人把它当成"在线、实时"的训练信号直接喂回模型。本文要做的就是这件事——而且专治一个老大难:用结构化纠正去做在线蒸馏时,师生分布一对不上就训崩。

- 研究背景:LLM agent 已经广泛部署(终端 / GUI / SWE / tool-call);它们产生的交互数据,被用来改进框架、构建 memory(Mem0/Cognee/Letta)、生成训练数据,但很少有人把它当作**在线、实时**的学习源。现有的 agentic RL 基础设施(slime / OpenRLHF / AReaL 等)都假设数据是批量预采集好的。【原文 §1】
- 解决的具体痛点:
  - ① 基础设施层面:RL server 需要能灵活对接用户那些多样且不断演进的 agent 框架,而且优化必须异步、不能阻塞推理的正常使用;
  - ② 方法学层面:RLVR 只能用标量奖励,没法把 directive(纠正性)信号转成策略梯度;OPD / hindsight 虽然能用结构化纠正,但只能在固定数据集上操作;而且 OPD 还会因 teacher-student 分布失配而训练不稳、甚至失效,hint 质量低时更严重。【原文 §1】
- 相关工作 & 各自不足（来龙去脉 + 并行路线 + 精确差异）:
  - **RLVR**:只能用标量 outcome。精确差异:本文多了一条 directive(token-level)的路径。
  - **on-policy 蒸馏**(Agarwal 2024、SDPO(Hübotter 2026)、SDFT(Shenfeld 2026)):能用结构化纠正,但**只在固定数据集上**;本文把它做成**在线实时**,并解决了失配稳定性。
  - **hindsight relabeling**:把纠正信息加进 context 来提升输出,但用的是固定数据集,且纠正是隐含在 prompt 里的(不是显式信号)。
  - **并发的 Buening 2026**:直接用 next-state 提示在线改策略,但它的纠正 hint 仍隐含在 prompt 中(不是显式信号);本文则把 directive 信号**显式抽成 token-level 的 hint**。
  - **memory 类(Mem0/Cognee/Letta)**:走的是外部记忆;本文是**参数级的在线优化**而非外部记忆,且 §4 的实验直接把 Mem0/Cognee 拿来作对照。
  - **站谁肩上**:OPD 的 token-level KL(Agarwal / Shenfeld / Li 2026 的 top-\(k\) 蒸馏)+ RLAnything(即 open_agentrl)的 step-wise PRM(§3.3 明确沿用了 \(o+\sum r_i/m\))+ slime 异步 RL 栈。它的关键贡献有两点:
    - ① 首个统一了 personal 与四类 general agent(terminal / GUI / SWE / tool-call)的开源 RL 框架;
    - ② hybrid 把"频率高的标量"和"信息密的 token 分布"在同一个 loss 里互补起来。【原文 §1+§3.3+Table 2】
- 动机链（一步步推下来）:
  - 每个 next state 既隐含着评价(用户重问 = 不满、通过测试 = 成功、错误 trace = 失败),又常携带指令信息("你应该先检查文件"就在 token 级指明了该怎么改);
  - 而 RLVR 只能用前者、纯 OPD 只能用后者,两者都没把 \(s_{t+1}\) 吃满;
  - 所以要把这两类互补信号都回收回来、统一进一次在线更新,并稳住其中的 OPD 蒸馏。【原文 §1+§3.1】
- 与最近邻工作的 Δ:
  - vs 纯 RLVR:多了一条 directive(token-level)路径;
  - vs 纯 OPD:多了 evaluative 兜底,且是**在线实时**(非固定数据集),还解决了失配稳定性;
  - vs memory 类(Mem0/Cognee):它是参数级的在线优化,而非外部记忆。【原文 Abstract+Table 2】

## 怎么做 + 靠不靠谱
> 一句话导读:系统分两块——基础设施把模型包成一个推理 API,让用户的 agent 框架在线把交互数据流式回传、异步训练而不打断推理;方法则是"PRM 从每次交互抽出标量评价和一条纠正 hint,挑重叠最大的 hint 做 OPD,再和 GRPO 混进一个 loss"。靠不靠谱?纯 OPD 在这里反而最差,hybrid 的增益主要来自"加了 evaluative 兜底",所以核心其实是"用 evaluative 稳住 directive";另外它的 personal 评测用的是模拟用户 + 硬编码偏好检测器,生态效度有限。

- 方法流水线（输入→输出,读完即可复现）:
  - **基础设施**:RL server 把策略 \(\pi_\theta\) 包成一个 stateless 的推理 API;用户的 agent 框架(可在个人设备 / 终端 / 云上)经 HTTP 把交互数据流式回传;请求分两类——main-line turn(可训)和 side turn(只前向);另有一个独立异步的 PRM/Judge server,从 \((a_t, s_{t+1})\) 中抽出 evaluative 分与 directive hint(它不阻塞推理,还可用更强的模型做多投票);整套由 slime 协调四个解耦的异步组件(环境 server / PRM / Megatron 训练 / SGLang serving),做到零 serving 中断、且能 graceful 地更新权重。
  - **方法**:分五步——
    - ① **evaluative**:PRM 投 \(m\) 次、取多数,得 \(r_t\in\{+1,-1,0\}\);
    - ② **directive**:PRM 先判断 \(s_{t+1}\) 里是否含有有意义的纠正,若有就把它蒸成一条 hint \(h\)(放在 `[HINT_START]…[HINT_END]` 之间)、append 进去得到 \(s^h_t\),再让同一个模型在这个 hint-augmented 的 prompt 下输出 teacher 分布 \(\pi_T\);
    - ③ **overlap-guided 选出 \(h^\star\)**(可序列级或 token 级,都是选 top-\(k\) 重叠最大的那个);
    - ④ **top-\(k\) OPD loss(Eq.1)**:把损失限制在词表子集 \(S_i\)(= 学生的 top-\(k\)),优势 \(A_v=\Delta_v\cdot w_v\)——其中 \(w_v\) 把优势集中到学生真会采样的 token,\(\Delta_v=\mathrm{clip}(\ell_T-\ell_{\text{old}},-C,+C)\);整体是 clipped-surrogate 形式;
    - ⑤ **hybrid**:\(L_i = w_{\text{RL}}L^{\text{GRPO}}_i + w_{\text{OPD}}L^{\text{OPD}}_i\)(默认两个权重各取 1)。【原文 §2+§3.1-3.2】
  - **step-wise reward(用于 general agentic,§3.3)**:长 horizon 下,outcome-only 只在最后一步给梯度。所以让 PRM 对每个 turn 都据 next-state 给分,step \(t\) 的 reward = \(o+\sum_{i=1}^m r_i/m\)(这里明确沿用了 RLAnything);advantage 则按"同一 step 序号分组、再标准化"来算(因为真实状态很难聚类,所以不按状态聚类)。【原文 §3.3.1-3.3.2】
- 逐组件必要性（每模块干啥 / 有无消融）:
  - **overlap-guided hint selection**:有消融(Table 4:sequence-optimal 12.5 / token-optimal 12.4,都优于 random 的 16.1)——所以必要,而且序列级在 general agentic 上更稳。【原文 §4 / Table 4】
  - **logprob-difference clip(\(\Delta\)-clip)**:§4.9 的标题直接就是 "Log-Probability Difference Is Vital for Stability"——所以必要(它把每个 token 的优势 \(\Delta_v\) bound 住)。【原文 §4.9】
  - **hybrid(对比纯 RLVR / 纯 OPD)**:有对照(Table 3 Average:Hybrid 10.3 < GRPO 14.1 < OPD 29.7;Fig.6 在 Retool/RLVR 上,Hybrid 也优于 PRM+Outcome 与 Outcome)——所以必要。值得注意的是纯 OPD 反而最差(29.7),这佐证了"单靠 directive 不稳、需要 evaluative 兜底"。(注:这里的指标越小越好,代表对齐所需 sessions 数。)【原文 §4 / Table 3 / Fig.6】
  - **step-wise process reward(用于 general agentic)**:§4.5。在 tool-call/GUI 上,集成 outcome+process 优于 outcome-only——所以必要(因为长 horizon 下信号稀疏)。它明确建立在 RLAnything(即 open_agentrl)之上。【原文 §4.5】
  - **PRM 判"是否值得抽 hint"**:必要(若不加判断、强行抽取,会产生低质量 hint、反而破坏训练)。【原文 §3.1】
- 关键机制/公式（真实形式 + 直觉,从 PDF 抄准）:
  - **overlap 信号**:\(S^q_i=\mathrm{top\text{-}}k\{\pi_{\text{old}}(\cdot\mid s_t,y_{<i})\}\)(学生 top-\(k\) 词表)、\(S^p_{i,h}=\mathrm{top\text{-}}k\{\pi_T(\cdot\mid s^h_t,y_{<i})\}\)(hint \(h\) 下教师 top-\(k\));\(\displaystyle O[h,i]=\big|S^q_i\cap S^p_{i,h}\big|.\)
  - **hint 选择**:\(\displaystyle h^\star(i)=\begin{cases}\arg\max_h \sum_i O[h,i], & \text{sequence-level}\\[2pt]\arg\max_h O[h,i], & \text{token-level}\end{cases}\) 两种粒度的区别是:序列级对每条轨迹只选一个 hint,token 级对每个位置可选不同的 hint(因为 OPD loss 本身就是 token 级的,见 Appendix C);二者性能相近,而序列级在 general agentic 上更稳。直觉:重叠高,就说明师生在词表的 support 上已经一致;那么蒸馏就是在学生的高密度区内移动,而不是把学生往陌生 token 上推。
  - **top-\(k\) OPD loss(Eq.1,clipped-surrogate)**:对 \(v\in S_i\)(默认 \(S_i=S^q_i\)),令 \(\ell_{\text{old}}(v)=\log\pi_{\text{old}}(v\mid s_t,y_{<i})\)、\(\ell_{T,h^\star}(v)=\log\pi_T(v\mid s^{h^\star}_t,y_{<i})\)、\(\ell_{\text{cur}}(v)=\log\pi_{\text{cur}}(v\mid s_t,y_{<i})\),并定义 \(\displaystyle w_v=\mathrm{softmax}_{v\in S_i}\,\ell_{\text{old}}(v),\quad \Delta_v=\mathrm{clip}\big(\ell_{T,h^\star}(v)-\ell_{\text{old}}(v),\,-C,\,+C\big),\quad A_v=\Delta_v\cdot w_v,\quad \rho_v=\exp\big(\ell_{\text{cur}}(v)-\ell_{\text{old}}(v)\big),\) 则 \(\displaystyle L^{\text{OPD}}_i=\sum_{v\in S_i}\max\big(-A_v\rho_v,\ -A_v\,\mathrm{clip}(\rho_v,\,1-\varepsilon_{\text{lo}},\,1+\varepsilon_{\text{hi}})\big),\quad \varepsilon_{\text{lo}}=0.2,\ \varepsilon_{\text{hi}}=0.28.\) 直觉分三点:\(w_v\) 把优势集中到学生真会采样的 token 上;\(\Delta\)-clip 把单个 token 的 logprob 差 bound 在 \(C\) 之内,这样即使 teacher 在局部远离学生也不会爆;overlap 选择则让 \(\rho_v\approx1\)。后两者合起来**同时 bound 住了 surrogate 的两个因子**(即 advantage 与比值),从而既稳定、又不丢失 hint 的方向信息。附录 C 还证明了 OPD 目标其实就等价于 token-level KL。【原文 §3.2 / Eq.1 / Appendix C】
  - **hybrid 目标**:\(L_i=w_{\text{RL}}L^{\text{GRPO}}_i+w_{\text{OPD}}L^{\text{OPD}}_i\)(默认 \(w_{\text{RL}}=w_{\text{OPD}}=1\)),两项共享同一轨迹与同一策略更新,互补"频率(evaluative)× 信息密度(directive)"。【原文 §3.1】
- 实验与证据:
  - **Personal**(§4.1):用 LLM 模拟不同职业的用户(学生要避免 AI 痕迹、助教要详细评分(<100 token 就算不足)、教师要用友好的暖词),在 GSM8K 任务上使用 OpenClaw;度量的是"对齐各用户偏好所需的最少 sessions"(连续 3 个 session 都满足偏好即算达标,且第一条消息被硬编码、不透露偏好)。
    - 配置:policy 与 reward 都用 **Qwen3-4B-Thinking-2507**,用户用 Qwen3-32B 模拟;lr \(1\times10^{-5}\)、\(C=1\)、每 16 个样本触发一步。
    - 关键数字(Table 3 的 **Average** 行,5 trials 均值,越小越好):joint 优化下 Hybrid 是 **10.3** sessions,优于 GRPO 14.1、OPD 29.7、Mem0 14.5、Cognee 14.9;separate 优化下 Hybrid 是 15.0(GRPO 21.1 / OPD 29.4 / Mem0 15.1 / Cognee 15.1),仍最优。
    - 注:joint 优化会放大 RL 的增益,而 memory 类几乎不变。
  - **General**(§4.2/4.5):
    - 四个环境各用一个模型:terminal=Qwen3-8B、GUI=Qwen3VL-8B-Thinking、SWE=Qwen3-4B、tool-call=Qwen3-4B-SFT(即 Zhu 2025 在 Feng 2025a 数据上 SFT 得到的);
    - 训练数据分别是 SETA RL / OSWorld-Verified / SWE-Bench-Verified / DAPO RL;
    - 评测:GUI 在训练集上评(去掉 chrome 与 multi-apps),tool-call 评 AIME 2024,terminal/SWE 报窗口内平均 rollout-task acc;PRM(GUI/tool-call)分别是 Qwen3VL-8B-Thinking / Qwen3-4B;
    - 超参:lr \(10^{-6}\)、KL 0.01、clip 0.2/0.28;每步采任务数为 GUI/SWE 各 8、terminal 16、tool-call 32,每个任务各 8 samples;GUI/SWE/terminal 的最大交互步分别是 30/20/10;在 tool-call/GUI 上,集成奖励优于 outcome-only。
  - **Hybrid RL Extension**(§4.3):
    - tool-call 用 Retool-4B(在 Retool 数据上 SFT)+ Qwen3-8B PRM;
    - RLVR 用 DeepSeek-R1-Distill-Qwen-1.5B 策略 + Qwen3-4B PRM,以 DAPO 训练、在 AIME 上评;
    - 超参:\(C=2\)、lr \(10^{-6}\)、KL 0.01、clip 0.2/0.28、32 任务 × 8 rollout。
  - baseline 公平性:personal 这部分高度依赖 LLM 模拟用户 + 硬编码的偏好检测器(检测粗体/列表/长度/暖词),度量的是"对齐效率"而非真实能力提升,生态效度有限;GUI 那部分则直接在训练集上评。【原文 Table 3+Table 4+§4.5】
- 假设与失效边界:
  - 【原文 §3.1+§3.2】directive 信号只在"PRM 判定有意义纠正"的那部分子集上触发(因而稀疏);teacher 是由同一个模型在条件 hint 下得到的,所以 hint 的质量直接决定了失配的大小;\(\Delta\)-clip 的 \(C\) 是关键超参(personal 取 \(C=1\),RLVR 取 \(C=2\))。
  - 【原文 §4.5】hosting PRM 需要额外资源(相对 outcome-only 而言)。
  - 【推断】几点:
    - personal 评测的生态效度有限(用模拟用户 + 硬编码检测器,度量的是风格对齐、而非能力);
    - general 中有部分环境是在训练集上评的;
    - 是否在真实人类用户上验证过,未知。
    - 〔待核〕"HuggingFace Daily Papers #1":已确认 PDF 正文里不含这个声明(应该出自仓库 README),保留为待核。
- 祛魅总结:【推断】
  - 真贡献:
    - ① 把"next-state 既含评价又含指令"这个观察,做成了**两路互补信号 + 单个 hybrid loss**,并用 overlap-guided hint selection 漂亮地解决了 OPD 的在线失配稳定性——这是把 OPD 真正用进 agentic 在线训练的关键贡献(既是工程也是方法);
    - ② 它是首个统一了 personal 与四类 general agent 的开源异步 RL 基础设施(server-client、零中断),工程价值高。
  - 被适度高估之处:
    - ① "越用越强"是由 personal 设置度量出来的,但该设置用的是模拟用户 + 硬编码偏好检测器,生态效度有限,所以那个数字(10.3 sessions)是相对这套人造度量而言的;
    - ② 纯 OPD 最差(29.7),hybrid 的增益主要来自"加了 evaluative 兜底"、而 directive 单路并不稳——所以它的核心其实是"用 evaluative 稳住 directive";
    - ③ step-wise reward 那部分直接沿用了 RLAnything,不是本文原创。

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号:evaluative(PRM 标量,序列级)+ directive(hint-conditioned teacher 的 token-level 分布,即 OPD 的 token-level KL);hybrid 混合。
  - 改什么:学生策略 \(\pi_\theta\)(Megatron 训练);PRM 是独立 frozen/更强模型(不训)。
  - 何时改:在线、每 main-line turn;但 directive 仅在 PRM 判"有有意义纠正"的稀疏 turn 触发;weight update 在 well-defined 边界推给 serving。
  - 免梯度?否——hybrid clipped-surrogate 梯度优化;但 hint selection(overlap argmax)与 PRM 判定是免梯度的离散选择。
  - 记忆-技能生命周期:与 memory 类(Mem0/Cognee)对比——本文是**参数级在线巩固**(把偏好/纠正写进权重)而非外部记忆;集成社区 SDFT/SDPO 到 openclaw-opd/。
  - 防遗忘机制:KL 系数 0.01(general)作 reference 锚;\(\Delta\)-clip + overlap 选择把更新限制在学生高密度区(隐式防过度偏移);多用户 joint 优化共享一个模型。
- ⑦ 开源代码+框架/harness:github.com/Gen-Verse/OpenClaw-RL(已 clone,~73M)。框架 **slime(THUDM/slime)异步 RL** 为核心,四解耦异步组件:环境 server、PRM/Judge(SGLang/API)、**Megatron**(策略训练)、**SGLang**(策略 serving);server-client 把模型包成 OpenAI 兼容 API(经 OpenClaw 插件),HTTP 流式回传在线训练,零中断、graceful weight update;也支持 Tinker 云端/本地 GPU 与 LoRA。三范式:Binary RL / OPD / Combine(Hybrid)。关键子目录:openclaw-opd/(OPD,含 top-\(k\) 蒸馏 loss、qwen3/qwen35 launcher、集成 SDFT/SDPO)、openclaw-rl/、openclaw-combine/、openclaw-test/、openclaw-tinker/、openclaw-fireworks/;Track-2:gui-rl/、terminal-rl/、swe-rl/、toolcall-rl/;含 slime/、Megatron-LM/、extensions/。**(注:仓库有 qwen3.5-4B 配置/launcher,但 PDF 报告模型是 Qwen3-4B-Thinking-2507,qwen3.5 非论文报告模型——勿据仓库回退成"用了 qwen3.5"。)** 【原文/仓库】
- 💰 资源/成本与可扩展性:personal lr \(10^{-5}\)、每 16 样本一步、Qwen3-4B-Thinking-2507;general lr \(10^{-6}\)、每步采 8/16/32 任务×8 samples;PRM 作独立 server(可用更强模型多投票,但需额外资源);异步设计使 serving 零中断。规模上限实测 8B 级。【原文 §4.1+§4.2+§4.5】
- 🎯 对"探索-巩固"对标:**强支撑(它把 OPD 用进了在线 agentic,而且专治失配稳定性)**。
  - 一句判定:OpenClaw-RL 把 OPD 显式当作 hybrid RL 里的 "directive 信号",并用 overlap-guided hint selection 解决"teacher-student 失配"——这正好对应本课题"巩固/回轨"里"on-policy 自选 + 选一个能走通的恢复分支"的那个稳定化难题。依据是 §3.2 的失配诊断 + overlap 选择。
  - 可借组件:
    - ① **overlap-guided 的 hint/分支选择**:"挑那个诱导 teacher 与学生高概率区重叠最大的纠正"(即对 \(O[h,i]=|S^q_i\cap S^p_{i,h}|\) 取 argmax)——这可直接用于"path-recovery 时,选一个学生自己能走通的恢复分支"(对应 idea 里"偏向自己能走通的开头");
    - ② **\(\Delta\)-clip(给 logprob 差设上限)+ top-\(k\) 子集的 OPD loss(Eq.1)**:这对应"forward-hard / backward-soft"式的 bounded 梯度,可迁移到 MTP/OPD;
    - ③ **"用 evaluative 稳住 directive"的 hybrid 思路**:稀疏的关键 directive 信号需要稠密的标量来兜底,这对"sparse_critical 单点接管"有借鉴意义(单点的 path-recovery 也可能需要一个全局 reward 兜底);
    - ④ 把 next-state 当作在线学习源,这本身就是"探索 = 发现有效行为/路径"的在线回收。
  - 缺口:
    - ① 没有 MTP/前瞻(它的 hint 是 PRM 事后抽的,不是前瞻地预测将要走偏);
    - ② directive hint 来自一个外部 PRM 对 next-state 的解读,而非"看过答案的自己"主动选路;
    - ③ 巩固靠的是在线 KL 软锚,没有显式的"防遗忘、写进技能库/记忆"机制(它明确与 memory 类对立);
    - ④ personal 评测的生态效度,限制了"巩固 = 真能力提升"这一结论的证据强度。
- 🔭 开放问题/未来方向:
  - 【原文 §6/§4.5】扩到更多真实 agent 部署;优化 PRM 的资源开销;更好地整合长 horizon 的过程奖励。
  - 【推断】三个方向:
    - 把 overlap-guided hint selection 从"事后挑 PRM 抽出的 hint",升级为"用 MTP 前瞻预测学生将要走偏、主动构造或选择一个能走通的恢复分支"(即前瞻探针 + path-recovery);
    - 把"PRM 判定是否值得抽 hint",改造成"是否已到了需要接管的关键步"这一判据(对应 sparse_critical);
    - 在真实人类用户(而非模拟用户)上验证"越用越强",以建立"巩固 = 真能力提升"的证据。

RETURN: openclaw_rl|读到PDF=是(标题OpenClaw-RL;§3.1 hybrid+§3.2 overlap+Eq.1 OPD loss+§3.3 step reward(沿用RLAnything)+§4.1-4.5实验Table3-4+超参)|模型规格已核=PDF无qwen3.5(报告Qwen3-4B-Thinking-2507等),Daily Papers不在PDF(出自README),两处更正保持勿回退|L线=L4+L1|对标=强支撑(把OPD用进在线agentic+overlap选恢复分支直接可借于path-recovery,Δ-clip≈forward-hard/backward-soft;但无MTP前瞻/hint来自外部PRM非"看过答案的自己"/无技能库防遗忘)|残留待核=1(HF Daily Papers #1出自README非PDF,保留待核)

# openclaw_rl — OpenClaw-RL: Train Any Agent Simply by Talking

> **一句话重点 (TL;DR)**：把每次 agent 交互产生的"next-state 信号"当作在线学习源回收,从中抽 evaluative(标量/更频繁)与 directive(token-level/更富信息但稀疏)两类信号,在一次 hybrid RL 更新中统一;并用 overlap-guided hint selection + logprob-diff clip 稳定 teacher–student 失配下的 OPD,使"agent 越被使用越变强"。

**元信息**：arXiv 2603.10165（v2, 2026-05-11, cs.CL）｜ Gen-Verse（Yinjie Wang*, Xuyang Chen*, Xiaolong Jin*, Mengdi Wang†, Ling Yang† 通讯;含 Princeton 等）｜ 2026 Preprint（Blog yinjjiew.github.io/projects/openclawrl1）｜ 主题 T2（agent RL/工具）,与 mtp_opd 高度相关——把 OPD 显式作为 hybrid RL 的 "directive signal",专门处理 teacher–student 失配下的 OPD 稳定性(overlap-guided hint selection + logprob-diff clip,对应 path-recovery/hint 选择)｜ 代码 github.com/Gen-Verse/OpenClaw-RL（已 clone,~73M）｜ 框架 slime 异步 RL（Megatron 训练 + SGLang serving + PRM）

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/openclaw_rl/fig_01.png)

*Figure 1 | OpenClaw-RL infrastructure overview. Interaction streams come from two agent types: Personal Agents (conversational, personalized), hosted on personal devices, and General Agents (terminal, GUI, SWE, and tool-call agents), hosted on cloud services. The collected samples flow into our RL s*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/openclaw_rl/fig_03.png)

*Figure 3 | Method Overview. For personal agents, we support both binary-reward optimization and on-policy distillation training. In our experiments, we find that their combination yields significant performance gains. For general agentic RL, in addition to standard RLVR, we provide integrated step-w*

## 1. 相关工作与进展
LLM agent 已广泛部署(终端/GUI/SWE/tool-call);交互数据被用于改进 framework、构建 memory(Mem0/Cognee/Letta)、产训练数据,但鲜有把它当**在线、实时**学习源。现有 agentic RL 基础设施(slime [Sheng 2025]、OpenRLHF [Hu 2024]、AReaL [Fu 2025] 等)假设批量预采集数据。方法侧:RLVR 仅能用标量奖励;OPD [Agarwal 2024]、SDPO/SDFT [Hübotter/Shenfeld 2026]、hindsight relabeling 能用结构化纠正信息但都在固定数据集上操作;并发 Buening 2026 直接用 next-state 提示在线改策略,但纠正 hint 仍隐含在 prompt 中。

## 2. 现有工作存在的问题

- **基础设施**:RL server 需灵活对接用户多样且演进的 agent 框架,且优化须异步、不阻塞推理使用。
- **方法学**:RLVR 无法把 directive(纠正性)信号转成策略梯度;OPD/hindsight 虽能用结构化纠正,但只在固定数据集上做。OPD 还因 teacher–student 分布失配 [Li 2026] 训练不稳/失效,低质量 hint 时更严重。

## 3. Motivation
每个 next state(用户回复、工具输出、终端/GUI 状态变化)既隐含评价(re-query=不满、通过测试=成功、错误 trace=失败),又常携带指令信息("你应该先检查文件"在 token 级指明该怎么改)。回收这两类互补信号、统一进一次在线更新,并稳定其中的 OPD 蒸馏。

## 4. 主要灵感 / 核心直觉
evaluative 与 directive 互补:directive 更富信息(token-level)但更稀疏,evaluative 更频繁(标量)。OPD 的 token-level KL 监督(teacher 条件于纠正 hint)恰能承载 directive。稳定 OPD 的关键直觉:若所选纠正 hint 诱导的 teacher 分布与学生在高概率区高度重叠,则蒸馏梯度更 informative、off-policy importance ratio 接近 1,更新更稳。

## 5. 主要解决思路(一段话讲清核心)
基础设施:把 RL 系统扩成 server–client——RL server 把策略包成推理 API,用户终端经 OpenClaw 把交互数据 HTTP 流式回传;独立异步 PRM/Judge server 从 next state 抽 evaluative + directive 信号(不阻塞推理);slime 协调环境 server / PRM / Megatron 训练 / SGLang serving,零 serving 中断、graceful weight update。方法:hybrid RL 目标在单次更新融合 evaluative(RLVR 标量)与 directive(OPD token-level KL,teacher 条件于纠正 hint)两类 loss,配 overlap-guided hint selection 与 logprob-diff clip 稳定。

## 6. 方法详解(通俗、分步骤)

1. **Hybrid RL objective**:directive=OPD(token-level KL)+ evaluative=RLVR(标量奖励),单更新融合(附录 C 证明 OPD 目标即 token-level KL)。
2. **Overlap-guided hint selection**:候选纠正 hint 中,选其诱导 teacher 分布与学生 **top-k token 重叠最大**者(可逐 token 或序列聚合)。
3. **Log-probability-difference clip**:对 token-level logprob 差做 clip,bound 每 token advantage(personal 设置默认 C=1)。
4. **Step-wise / process reward**:通用 agentic RL 整合 outcome + process reward(PRM),强调过程奖励对长 horizon 稀疏奖励 agent 任务的重要性。
- 集成了社区 SDFT、SDPO 等方法到 `openclaw-opd/`。

## 7. 实验数据集

- **Personal agents**:用 LLM 模拟不同职业用户(学生避免 AI 痕迹 / 助教要详细评分 / 教师要友好评语)在 GSM8K 任务上使用 OpenClaw,度量"对齐各用户偏好所需最少 sessions"(连续 3 session 满足偏好即达标)。policy & reward 均 Qwen3-4B-Thinking-2507;用户用 Qwen3-32B 模拟。
- **General agents(Track 2)**:四类真实部署环境,模型分别为 Terminal=Qwen3-8B、GUI=Qwen3VL-8B-Thinking、SWE=Qwen3-4B、Tool-call=Qwen3-4B-SFT(Retool-4B,源自 Zhu 2025);训练数据分别为 SETA RL data / OSWorld-Verified / SWE-Bench-Verified / DAPO RL data;GUI 评在训练集(去 chrome 与 multi-apps),tool-call 评 AIME 2024,terminal/SWE 报窗口内平均 rollout-task acc。声称是首个统一这四类 agent 的开源 RL 框架。
- (注:正文实验未涉及 Qwen3.5;仓库 `openclaw-opd/run_qwen35_4b_openclaw_opd.sh` 与 slime qwen3.5-4B 配置存在,但非论文报告模型。)

## 8. 实验结果与主要发现

- **Personal**(Table 3,最少 sessions↓,5 trials 均值):joint 优化下 Hybrid RL 平均 **10.3** sessions,优于 GRPO 14.1、OPD 单用 29.7、Mem0 14.5、Cognee 14.9;separate 优化下 Hybrid 15.0 仍最优。注:纯 OPD 远差(29.7),增益主要来自 hybrid 融合;joint 优化放大 RL 增益而 memory 类几乎不变。
- **General**:hybrid RL 在四环境均优于纯 RLVR 且更稳定;next-state 信号在长 horizon 稀疏奖励环境尤其有用。
- **消融**:overlap-guided hint selection 与 logprob-diff clip 对效率/稳定性均关键;另有 k 与 support set S_i、PRM、policy/reward 模型组合的消融(§4.8–4.11)。
- 超参:personal lr 1e-5、每 16 样本触发一步;general lr 1e-6、KL 系数 0.01、clip 0.2/0.28、每步 GUI/SWE 采 8 任务、terminal 16、tool-call 32,各 8 samples;GUI/SWE/terminal 最大交互步 30/20/10。

## 9. 结果如何支撑其主张
"越用越强"由 personal 设置的 session-to-align 指标直接度量(Hybrid 最少 sessions)。"directive+evaluative 互补、hybrid 更优"由 hybrid 全面超 GRPO/纯 OPD 支撑(纯 OPD 反而最差,佐证单靠 directive 不稳、需 evaluative 兜底)。"稳定性来自 hint 选择/clip"由 §4.8–4.9 消融支撑。统一框架主张由四环境实跑支撑。

## 10. 逻辑自洽性(中性评估)
infra(slime 异步、server–client、零中断)与 method(hybrid+hint selection+clip)两条创新线自洽,且附录 C 把 OPD 归约为 token-level KL、与 evaluative 路径并列合理。但 personal-agent 评测高度依赖 LLM 模拟用户与硬编码偏好检测器(粗体/列表/长度/暖词),度量"对齐效率"而非真实能力提升,生态效度有限;general-agent 部分环境直接在训练集上评(GUI),需谨慎。

## 11. 残留问题 / 局限
〔待核〕HuggingFace Daily Papers #1 一说——已确认论文正文不含此声明(应出自仓库 README),保留为待核。模拟用户/检测器的生态效度;directive 信号质量依赖 PRM/hint 抽取质量;具体超参与 support set S_i 细节见附录 A。是否在真实人类用户上验证未知。

## 12. 开源代码与框架(链接+框架+代码可得性)
github.com/Gen-Verse/OpenClaw-RL(已 clone,~73M)。框架 **slime(THUDM/slime)异步 RL** 为核心,四解耦异步组件:环境 server、PRM/Judge(SGLang/API)、**Megatron**(策略训练)、**SGLang**(策略 serving);server–client 把模型包成 OpenAI 兼容 API(经 OpenClaw 插件),HTTP 流式回传在线训练,零 serving 中断、graceful weight update;也支持 Tinker 云端/本地 GPU 与 LoRA。三范式:Binary RL / OPD / Combine(Hybrid)。关键子目录:`openclaw-opd/`(OPD,含 topk 蒸馏 loss、qwen3/qwen35 launcher、集成 SDFT/SDPO)、`openclaw-rl/`、`openclaw-combine/`、`openclaw-test/`、`openclaw-tinker/`、`openclaw-fireworks/`;Track-2:`gui-rl/`(及 terminal/swe/toolcall 相关);含 `slime/`、`Megatron-LM/`、`extensions/`。

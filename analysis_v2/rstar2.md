rstar2 | rStar2-Agent: Agentic Reasoning Technical Report | Microsoft Research(Ning Shang/Yifei Liu/Yi Zhu/Li Lyna Zhang 等共同一作;Li Lyna Zhang、Mao Yang 为 project leaders/通讯) | 2025-08-28 v1 · arXiv 技术报告(cs.CL) | L4 Agent/工具/多轮(+L3 GRPO) · 相关性高

**原始论文**:https://arxiv.org/abs/2508.20722

## 一眼看懂
- 🟦 TL;DR:让 14B 模型"想得更聪明而非更长"。用三件套——(i) 可靠高吞吐的 Python 代码执行环境(每步可处理 45K 并发工具调用、0.3s/call)+ 负载均衡 rollout 调度器;(ii) 抗噪的 **GRPO-RoC**(Resample-on-Correct:oversample 2G 再 downsample 到 G,正样本筛"工具错误/格式问题最少"的高质量、负样本均匀保多样);(iii) 低成本 recipe(先 non-reasoning SFT 打底,再用 GRPO-RoC 做 8K→12K→12K 三阶段短长度 RL)——在 64×MI300X、510 步 / 一周内把 Qwen3-14B-Base 推到前沿数学:AIME24=80.6 / AIME25=69.8 / HMMT25=52.7,AIME24 与 HMMT25 超过 671B 的 DeepSeek-R1。【原文】Abstract、§4.3、Table 5
- 最巧的一步:**GRPO-RoC 的非对称采样(正筛优 / 负保多样)**。抽掉它,在 answer-only outcome reward 下,模型会把"最终答案对但中间工具调用出错/格式乱"的冗长低质轨迹当正样本学,环境噪声放大、产出退化。论文明确选它而非"step-level reward / tool-error 惩罚",因为后两者易 reward-hacking 且早期阻碍探索。它是"用最小 outcome reward 抗环境噪声"的关键。【原文】§3.3、Eq.4-6

## 为什么做
- 研究背景:培养模型自主用工具(Python 编码+解释器)来推理/验证/从工具反馈学习,补足单纯 long CoT 内部自反思的不足;拓宽动作空间、支持探索替代解与验证中间步。【原文】§1
- 解决的具体痛点(扩展 agentic RL 两大挑战):
  - **环境噪声**:编码工具复杂,模型生成错误代码时环境反馈让其浪费 token 纠错;outcome-only 奖励下"最终答案对就给正奖励"会让模型把中间工具错误当可接受,产冗长低质轨迹(观察:工具错误率随训练步数上升)。【原文】§2、§3.3
  - **基础设施压力**:单 batch 可触发数万并发工具调用(实测每步 up to 45K),agentic rollout 放大标准 RL 的 rollout 低效,严重拖慢训练。【原文】§3、§3.1
- 相关工作 & 各自不足:主流堆 rollout 长度(16K→48K→80K,如 MiniMax/MiMo/Magistral,见 Table 2),成本高且冗长;reward 侧两路(step-level reward / tool-error 惩罚)引入复杂度且易 reward-hacking。rStar2 走"短长度 + 最小 outcome reward + 采样层抗噪"。【原文】§3.3、Table 2
- 动机链:agentic RL 受限于环境噪声 + 基础设施 → 不在 reward 里惩罚(易 hack)、而在**采样层**差异化处理正/负轨迹 + 造高吞吐环境 + 短长度 recipe → 有限 GPU 上小模型超大模型。【原文】§3、§4
- 与最近邻工作的Δ:相比 DAPO/ReTool 等单阶段长长度 RL,关键差是 **GRPO-RoC(采样层抗噪)+ non-reasoning SFT 冷启 + 多阶段短长度(8K→12K)**;Table 2 显式对比"是否 Reasoning SFT / RL 阶段数 / 总步数 / 最大长度 / 难度过滤"——rStar2 唯一同时做到"无 reasoning SFT + 短长度 + 仅 510 步"。【原文】Table 2

## 怎么做 + 靠不靠谱
- 方法流水线:① non-reasoning SFT(仅注入指令遵循/JSON/基础工具用,不增推理)→ ② RL 数据清洗(只留整数答案题,>100K 候选清洗到 42K)→ ③ 多阶段 GRPO-RoC:Stage-1 8K(42K 题,300 步,响应 1K→4K)→ Stage-2 12K(4K→6K)→ Stage-3 在 offline 过滤的 17.3K 难题上(reset 优化器 + 更新 reference,125 步,6K→8K),共 510 步。环境服务异步执行工具调用 + 负载均衡调度。【原文】§4.1-4.3
- 逐组件必要性:
  - **non-reasoning SFT**:打底不增推理。Table 1 证 SFT 后 tool/IF/chat 提升、数学持平于 base(MATH-500 62.0→57.4 略降,AIME 仍很低)——刻意保持初始响应短、不过拟合。无它则起点响应过长。【原文】§4.1、Table 1
  - **GRPO-RoC(RoC 采样)**:核心抗噪。负样本不过滤、取 ⌊|Oneg|/2⌋ 保失败多样;正样本按 ptotal=perr+pformat 反比采样(perr=错误工具调用/总调用,无工具调用默认 0.5 鼓励用工具;pformat 惩罚多/缺 <answer> 标签)。Fig.4 证 GRPO-RoC 下编码工具错误率下降、训练更稳。【原文】§3.3、Eq.4-6
  - **高吞吐环境服务 + 负载均衡调度器**:解基础设施压力。中心任务队列 + 32 send workers + 每 worker 节点 1024 执行 workers;调度器按可用 KV cache 动态分配 rollout、异步 dispatch 工具调用;还把 answer verification 异步 offload。无它则 GPU 大量 idle。【原文】§3.1-3.2
  - **多阶段短长度**:Stage-1 8K 时 clipping ratio 一度 >10% 仍坚持(逼模型高效推理,随后自调);Stage-3 因 Stage-2 末 >70% 题已满分而切到难题。无消融单独剥离,但与"短而聪明"动机一致。【原文】§4.3
- 关键机制/公式(直觉):GRPO-RoC = GRPO(group-relative advantage,Clip-Higher εlow=0.2/εhigh=0.28)+ 先 oversample 2G=32 再用 RoC 选 16。正轨迹"挑最干净的成功样本"作正监督,负轨迹"保多样失败模式"作负信号——比在 reward 里惩罚工具错误更稳、避免 reward-hacking。【原文】Eq.6-7、§4.3
- 实验与证据:基座 Qwen3-14B-Base。RL 数据 42K(17K DAPO 整数子集 + 93K AoPS via OpenMathReasoning + 937 Project Euler,经 Qwen3-32B 16 次生成、≥2 次匹配过滤)。lr 1e-6、batch 512 prompts、2G=32→16。数学评测 AIME24/25、MATH500、HMMT25;泛化 GPQA-Diamond、BFCL v3、IFEval、Arena-Hard。
  - 数学(pass@1):rStar2-Agent-14B **AIME24=80.6 / AIME25=69.8 / HMMT25=52.7**;对照 DeepSeek-R1(671B) 79.8/70.0/44.4、o3-mini(medium) 79.6/77.0/53.0、DeepSeek-R1-Zero 71.0/53.3/46.0、QWQ-32B 79.5/65.8/47.5。即 14B 在 AIME24、HMMT25 超 R1-671B,**AIME25 略低 0.2(70.0 vs 69.8,基本持平)**。【原文】Table 5
  - 效率:510 步 / 一周 / 64×MI300X;环境服务 45K 并发/步、0.3s/call。【原文】§3.1、§4.3
  - baseline 公平性:对照表为各家公布成绩(非同一 harness 重测),属技术报告惯例,严格可比性有限。【推断】
- 假设与失效边界:
  - 【原文】仅整数答案题(规则 verifier 难判代数等价,如 (a+b)(b+c)(c+a) vs 重排),限制任务多样性。
  - 【原文】Stage-3 性能在 125 步后 saturate 甚至下降,故停;开源迁移版(VERL v0.2→v0.5)尚未训出完整模型(作者注前 50 步差异极小但未给最终复现成绩)。
  - 【推断】单一基座(Qwen3-14B-Base)、单一硬件栈(MI300X);可靠性高度依赖自研环境服务工程实现,迁移到其他环境的稳定性未验证;AIME25 未严格超 R1 说明优势非跨所有 benchmark 一致。
- 祛魅总结:
  - 真贡献:证明"采样层抗噪(RoC)+ 短长度 + 高吞吐环境"可让 14B 在极少步数/算力下达前沿,挑战"必须堆长度"的范式;GRPO-RoC 的"正筛优/负保多样"是简洁有效的抗 reward-hacking 设计;基础设施工程(45K 并发、负载均衡)是 agentic RL 落地的硬贡献。【推断】
  - 包装/需冷静看:TL;DR/摘要的"超越 671B R1"以 AIME24/HMMT25 为主,**AIME25 实为略低/持平**(差 0.2);"前沿"对照表为各家公布数而非同 harness。整数答案约束 + 单基座单硬件,泛化与可复现性有门槛(开源版未训出完整模型)。【推断】

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号:answer-only 0/1 outcome reward(经 RoC 在采样层加质量筛选)+ group-relative advantage。
  - 改什么:策略参数(GRPO-RoC 策略梯度);不改 token 选择,改"哪些轨迹进 batch"。
  - 何时改:每步在线,oversample→RoC 筛选→更新;难度过滤为阶段性 offline。
  - 免梯度?否,RL 策略梯度(GRPO)。
  - 记忆-技能生命周期:无外部记忆库;"工具使用技能"靠 SFT 注入格式 + RL 内化,固化进参数。
  - 防遗忘机制:无显式防遗忘;Stage-3 reset 优化器 + 更新 reference 是稳定性手段而非防遗忘。
- ⑦ 开源代码+框架/harness:https://github.com/microsoft/rStar(main 分支为 rStar2-Agent;prior work 在 rStar-mutualreasoning/rStar-math 分支;v1 浅克隆 ~576K 未拉 submodule)。框架=**veRL v0.5(volcengine/verl)+ Code Judge(0xWJ/code-judge,开源版的工具执行服务,Redis+uvicorn+workers)**;原训练基于 VERL v0.2 + 自研多轮工具框架,开源版迁移到 v0.5。rollout/推理用 vLLM(`--enable-auto-tool-choice --tool-call-parser hermes`)。关键脚本 `examples/run_qwen3-14b_rstar2_agent_weave.sh`;RoC 配置 `do_down_sampling=True`、`down_sample_to_n=16`、`roc_error_ratio=True`、`roc_answer_format=True`、`min_zero/non_zero_reward_trace_num=2`。**未完整克隆说明**:verl/code-judge 为 git submodule(需 `git submodule init/update`),v1 浅克隆未拉取;且开源迁移版未训出完整模型,端到端复现有门槛——代码链接见上,需手动补 submodule。【原文】§3.1 + 仓库(v1 核查)
- 💰 资源/成本与可扩展性:**64×MI300X、510 步、一周**(刻意"最小算力达前沿"是核心卖点);环境服务分布在 64-GPU 集群 CPU 核上(32 send workers + 每节点 1024 执行 workers),45K 并发/步、0.3s/call;短长度(8K→12K)大幅降算力。【原文】§3.1、§4.3
- 🎯 对"探索-巩固"对标:**部分支撑(可借采样思想),非直接同构**。判定:GRPO-RoC 的"**正样本筛最干净的成功轨迹作正监督(巩固有效行为)+ 负样本保多样失败模式(暴露并学会避开走偏路径)**"与本项目"探索=发现有效路径、巩固=固化有效行为"在精神上一致——尤其"保留多样失败作负信号"对应"教学生识别/恢复走偏"。可借组件:用工具错误率/格式做轨迹质量打分、非对称采样,可迁移到 TSRD 的"挑高质量正样本巩固"。关键差异:rStar2 是**轨迹级 outcome RL**(无 token/step 级 path-recovery、无 teacher 脚手架、无自蒸馏、无 MTP),"巩固"靠 RL 内化而非 OPD;且失败轨迹只作"负 advantage"而非"教如何从该点恢复"。依据:§3.3。【推断】
- 🔭 开放问题/未来方向:【原文】§5.3 讨论性能饱和后下降(何时停);Unsuccessful Attempts(§4.3.1)的经验教训。 【推断】超越整数答案约束的可验证任务扩展;GRPO-RoC 迁移到非编码 agentic 环境;把"采样层质量筛选"与 token/step 级监督(path-recovery)结合;开源版训出完整模型以验证可复现性。

RETURN: rstar2|读到PDF?是(22页,_txt 87k字)|L4(+L3)|对标=部分支撑:RoC 正筛优/负保多样≈巩固有效+学会避坑,可借非对称采样;非同构(轨迹级 RL,无 token 级 path-recovery/无 MTP)|残留待核 0(v1 标〔待核〕的多阶段配比/步数/lr 本轮已从 §4.3 核到:510 步 lr1e-6 batch512 2G=32→16 ε0.2/0.28)

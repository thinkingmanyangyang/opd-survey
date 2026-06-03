opd_blog | On-Policy Distillation (博客 + tinker-cookbook 配方) | Thinking Machines Lab(Kevin Lu 等) | 2025 · 博客(非论文) | 主题线 L1(on-policy 蒸馏方法论)，兼 L4(多轮工具)、L6(token 级稠密信号) · 相关性 高

**原始论文**：https://thinkingmachines.ai/blog/on-policy-distillation/ (博客，无 arXiv)
> 来源说明:本条无 PDF。正文据既有 analysis/opd_blog.md(已核博客)+ 本批对 tinker-cookbook 仓 `distillation/train_on_policy.py` 等源码的二次核实撰写;数值为博客单点近似值，已标注。

## 一眼看懂
- 🟦 TL;DR：系统推广 **on-policy distillation**——学生在**自己采样的轨迹**上，以更强教师的**逐 token 分布(reverse KL)** 作为**唯一稠密监督**(环境不给任何 reward)，兼顾 on-policy 的分布真实性(解决 SFT 的训练-推理失配)与教师反馈的稠密性(解决 RL 的奖励稀疏)。博客在三场景演示:推理(数学)、个性化(指令遵循恢复)、多轮工具使用;配套 tinker-cookbook 给可复现 LoRA 配方。代表性曲线:OpenThoughts3 上 SFT 到 AIME'24 ≈55%，再在 DeepMath 上 on-policy 蒸馏**仅约 100 步**就升到 ≈65%。【既有 analysis + 博客】
- 最巧的一步：**把 reverse KL 当 per-token advantage 用策略梯度形式回传**(代码 `incorporate_kl_penalty`:`reverse_KL=log p_student − log q_teacher`，`advantage = −kl_penalty_coef·reverse_KL`)，而**不是**写成一个显式 KL 散度 loss 项。抽掉"在学生轨迹上算 + reverse KL"这一步——若改成在教师轨迹上算(off-policy)或用 outcome reward(RL)——就分别退回 SFT 的失配 / RL 的稀疏。这一步让"token 级稠密蒸馏"与"on-policy 采样"用同一套 importance-sampling loss 实现。【本批源码核 `train_on_policy.py`】

## 为什么做
- 研究背景：大模型蒸馏分两类——off-policy(在教师生成的固定数据上做 SFT)与 on-policy(学生采样自身轨迹、教师对学生轨迹逐 token 打分)。博客把 on-policy distillation 作为兼顾稠密监督与 on-policy 真实性的范式系统推广。【既有 analysis】
- 解决的具体痛点：① off-policy 蒸馏/SFT 易"复述"教师局部模式、存在训练-推理分布不匹配(exposure bias)，学生在**自己的推理轨迹上从未被纠正**，误差累积;② on-policy RL(GRPO)仅序列级稀疏奖励，token 学习效率低、采样成本高;③ 单纯模仿教师文本无法在"学生自己访问到的状态"上获得稠密反馈。【既有 analysis】
- 相关工作 & 各自不足：off-policy/SFT(失配、误差累积);on-policy RL/RLVR(稀疏、贵);GKD/MiniLLM 等学术 on-policy KD(范式同源但散度需手选、每步生成贵)。博客的定位是**把该范式工程化 + 配套 recipe** 并给三场景演示，配 off-policy 与 SDFT recipe 作对照。【既有 analysis + 仓 `train_off_policy.py`/`sdft.py`】
- 动机链：现状(SFT 失配 vs RL 稀疏，二选一)→ 洞见(让学生自己采样解决失配 + 让教师逐 token 打分解决稀疏 = 两全)→ 所以(on-policy distillation:学生轨迹 + 教师 token 分布的 reverse KL，无 reward)。【既有 analysis】
- 与最近邻工作的 Δ：相对 **off-policy SFT**——把数据源从"教师轨迹"换成"学生自身轨迹"，差在学生在自己会访问到的状态上被稠密纠正。相对 **on-policy RL(GRPO/RLVR)**——把"序列级 outcome reward"换成"token 级教师 KL"，差在每个 token 都有稠密信号、样本/步效率高得多(≈100 步即见效)。相对 **GKD/MiniLLM**——把它落成托管 SDK(Tinker)上的 LoRA 配方 + 多教师/多轮工具扩展。【既有 analysis】

## 怎么做 + 靠不靠谱
- 方法流水线(代码 `distillation/train_on_policy.py`)：① 学生采样轨迹，记录 token 级 `sampled_logprobs(p)` 与 mask → ② 对每条 datum 用其对应 teacher sampling client 算 `teacher_logprobs(q)` → ③ `reverse_KL=(log p − log q)·mask`;`per-token advantage = −kl_penalty_coef·mask·reverse_KL`(默认 coef=1.0) → ④ 可选 `kl_discount_factor`(默认 0.0)折扣未来 KL(博客称无明显增益) → ⑤ 把该 advantage 加进 datum.advantages，经 Tinker 的 importance-sampling loss 更新 → ⑥ **仅对学生生成 token 计 loss**(系统提示/用户消息/工具返回/assistant header 被 mask)。【本批源码核】
- 逐组件必要性：
  - **reverse KL 作 advantage(核心)**：没它就不是 on-policy 蒸馏。代码注释明确 `KL[p||q]=log p − log q`、`adjust advantages in-place as negative reverse KL`。【本批源码核】
  - **仅学生 token 计 loss(mask)**：没它会把 prompt/工具返回也算进去，污染信号。【本批源码核】
  - **零 reward(纯 KL)**：多轮工具用 `reward_fn=zero_reward` 覆盖默认 HarborReward——唯一信号是对教师的 KL。没它就混入 RL 奖励、不再是纯蒸馏。【既有 analysis + 仓 `harbor_multiturn.py`】
  - **多教师**：对每个数据集配 (teacher_model, groups_per_batch)，分别采样后拼接成批(`on_policy_multi_teacher.py`)。**博客未展示实验**，仅 recipe 提供。【既有 analysis】
  - **kl_discount_factor**：可选，默认 0.0;博客实验**无明显增益**——可视为"非必要"组件。【既有 analysis + 源码默认值】
- 关键机制/公式(直觉)：RL 的毛病是奖励稀疏(一整条轨迹只在最后给一个对/错)、SFT 的毛病是分布失配(只在教师的完美前缀上学，自己一走偏就没人管)。on-policy 蒸馏让学生**自己走**(on-policy 解决失配)，同时让教师在学生走到的**每一步**都给出"你这一步的分布该长什么样"(reverse KL 解决稀疏)。用 reverse KL(KL[p_student‖q_teacher]) 而非 forward KL，是 mode-seeking(让学生聚焦教师高概率的几个模式)而非 mode-covering——博客取这个但未深入论证取舍。教师 logprob 提供的稠密信号比序列级 outcome reward 的样本/步效率高得多，这是"≈100 步从 55%→65%"的直觉来源。【既有 analysis】
- 实验与证据：
  - **推理**：① OpenThoughts3-1.2M 上 SFT(rank-128 LoRA，约 3000 步)→ AIME'24 ≈55%;② 加载该 ckpt 在 DeepMath-103K 上 on-policy 蒸馏(教师 Qwen3-32B，groups_per_batch=512，约 100 步)→ AIME'24 ≈65%。即仅约 100 步从 55%→65%。【既有 analysis(博客近似值)】
  - **个性化**：内部文档 + 重采样 Tulu3(assistant 轮由 Qwen3-8B 重生成)做 SFT，Tulu3 prompts 做 on-policy 蒸馏 → IFEval 约 100 步恢复。【既有 analysis】
  - **多轮工具使用**：README 示例以 **Kimi-K2-Thinking 同时作学生与教师**(max_turns=10、group_size=4、groups_per_batch=8、lora_rank=8、kl_penalty_coef=1.0)，在 Harbor 沙箱以纯 KL 信号训练。【既有 analysis(已据 README 核)】
  - baseline 公平吗：**这是博客非受控论文**——多为单点曲线/示例，**缺多 seed/基线对照/统计显著性**;"55%→65%"为近似值;多教师在博客**未展示**。
  - "看着强但没回答核心问题"：直观展示了样本/步高效性，但严格性不足(无显著性、近似值)。
- 假设与失效边界：
  - 【源码注释】教师与学生**不同 renderer/tokenizer 时需手工对齐**(`train_on_policy.py` 注释:"if your teacher has a different renderer than the student, you may want to modify the full_sequence_inputs_D")——跨家族蒸馏未演示。
  - 【既有 analysis】on-policy 蒸馏需教师**在线 logprob 计算**，依赖 Tinker 托管服务——复现门槛绑定该 SDK。
  - 【推断】博客示例学生与教师**同族**(Qwen3-8B-Base 学生 / Qwen3-32B 教师，共享 renderer/tokenizer)，降低了分布漂移;跨家族通用性边界未系统评估。依据:recipe 默认与代码注释。
  - 【推断】reverse KL(mode-seeking)可能牺牲多样性(高 Pass@1 低 Pass@k)，博客未评 Pass@k;forward vs reverse 取舍未深入。依据:reverse KL 的已知性质 + 博客未讨论。
- 祛魅总结【推断】：
  - 真贡献：把 on-policy distillation 这一范式**清晰工程化**(reverse-KL-as-advantage 的实现、纯 KL 无 reward、仅学生 token 计 loss 均经代码核实)，并以三场景 + 可复现配方 + checkpoint 句柄降低了门槛，传播价值高(本课题 opd_survey 把它列为奠基性博客)。
  - 包装/被高估处：**非论文，无严格基线/统计**，数值为近似单点;多教师能力**只在 recipe、博客未展示**;reverse-KL 取舍、跨家族蒸馏、Pass@k 多样性均未评估;"两全其美"叙事在同族师生 + 数学/IF 场景成立，边界未充分探。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：教师在**学生轨迹**上的逐 token 分布(reverse KL=log p_student − log q_teacher);白盒(需教师 logprob);**无 reward**(既非正确性也非格式)。
  - **改什么**：改学生参数(LoRA);不改结构。
  - **何时改**：在线——学生采样→教师打分→更新，每个 token 都更新。
  - **免梯度?**：否，是梯度优化;但用**策略梯度形式**(reverse KL 作 advantage 经 importance-sampling loss)实现 token 级 KL 蒸馏，而非显式 KL loss 项。
  - **记忆-技能生命周期**：无外部记忆/技能库;"技能"=教师的逐 token 推理分布，蒸进学生 LoRA 参数。
  - **防遗忘机制**：无显式防遗忘;个性化场景用"SFT 恢复 + 蒸馏"组合来恢复被改动的能力，弱相关。
- ⑦ 开源代码+框架/harness：https://github.com/thinking-machines-lab/tinker-cookbook (已 clone，~14MB)。核心 `tinker_cookbook/distillation/`(`train_on_policy.py`、`train_off_policy.py`、`sdft.py`、`datasets.py`)与 `recipes/distillation/`(`on_policy_distillation.py`、`off_policy_reasoning.py`、`on_policy_multi_teacher.py`、`harbor_multiturn.py`、`on_policy_distillation_harbor_multi_turn.py` + `README.md`)。**框架=Tinker SDK**(基于 LoRA 的自研托管训练 SDK，蒸馏 loss/采样由 Tinker 服务侧执行)——**非 veRL/TRL/OpenRLHF**。启动如 `python -m tinker_cookbook.recipes.distillation.on_policy_distillation model_name=Qwen/Qwen3-8B-Base ... lora_rank=128`。【本批 repo 核】
- 💰 资源/成本与可扩展性：全程 LoRA(rank 8/32/128 句柄均提供);on-policy 蒸馏约 100 步即见效(样本/步高效是卖点);依赖 Tinker 托管(教师在线 logprob)。具体卡数博客未集中说明。关键超参:kl_penalty_coef=1.0、kl_discount_factor=0.0、group_size=4;多教师默认 DeepMath→Qwen3-32B、Tulu3→Qwen3-235B-A22B-Instruct-2507、学生 Qwen3-8B。【既有 analysis + 源码默认】
- 🎯 对"探索-巩固"对标：**强支撑(本课题 OPD 一侧的直接范式蓝本);是核心对标对象**。判定依据：① **范式同构**——"学生 on-policy 自采样 + 教师在学生轨迹上逐 token 稠密监督"正是本课题 TSRD 的 OPD 骨架:探索=学生自己走(on-policy)、巩固=教师把走到的每步分布拽向正确。② **可借组件(很实在)**:(a) `incorporate_kl_penalty` 的 **reverse-KL-as-advantage** 是现成的 token 级稠密信号实现，可直接作为"teacher 当稀疏脚手架"的底层信号(只需把"处处 KL"改成"仅关键步 KL");(b) 多轮工具 `reward_fn=zero_reward` 的纯 KL 设置 = 本课题"多轮 agent 自进化"的可复现起点;(c) 多教师拼批 = "多脚手架来源"的接口;(d) Tinker checkpoint 句柄便于复现。**缺口/与本 idea 的关键差异**:① 博客是 **token 级处处 reverse KL(稠密、全程介入)**，而本课题要 **teacher 当稀疏脚手架(只在关键步/走偏处单点接管)**——需把博客的"处处监督"稀疏化(这正是 TSRD 相对 vanilla OPD 的创新点);② **无 path-recovery 的"自选恢复分支"机制**(博客学生只是被教师逐 token 拽，不显式"走偏→自选回轨分支");③ **无 MTP 前瞻探针**;④ reverse KL 的 mode-seeking 可能压多样性，与本课题"探索=发现多条能走通的路径"略有张力。一句话:**它是 OPD 的可复现范式蓝本与底层信号实现(直接可借),本课题的增量在于把"处处稠密 KL"改造成"稀疏关键步脚手架 + 自选回轨 + MTP 前瞻"。**
- 🔭 开放问题/未来方向：
  - 【原文/博客】跨家族(不同 renderer/tokenizer)教师蒸馏;多教师的实证;kl_discount_factor 的适用场景(博客称无增益)。
  - 【推断】把"处处 reverse KL"稀疏化为"只在高熵/关键 token 用教师 KL 接管"(向本课题稀疏脚手架靠拢);评估 reverse vs forward KL 对 Pass@k 多样性的影响;把 on-policy 蒸馏与 path-recovery(走偏后让学生自选恢复分支、教师只在分支点稠密监督)结合;引入 MTP 前瞻来挑选"值得教师介入的关键步"。

RETURN: opd_blog|读到PDF=否(无PDF;据既有analysis(博客)+本批tinker-cookbook源码核`train_on_policy.py`的reverse-KL-as-advantage)|L线=L1(兼L4/L6)|对标=强支撑(OPD范式蓝本+reverse-KL-as-advantage底层信号直接可借,纯KL多轮工具是agent自进化起点;但处处稠密非稀疏脚手架/无自选回轨/无MTP,本课题增量正在此)|残留待核=0

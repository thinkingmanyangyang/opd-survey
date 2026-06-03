# opd_blog — On-Policy Distillation (Thinking Machines Lab 博客 + tinker-cookbook 配方)

> **一句话重点 (TL;DR)**：系统性推广"on-policy distillation"——学生在自身采样的轨迹上，以更强教师逐 token 的分布（reverse KL）作为唯一稠密监督，兼顾 on-policy 的分布真实性与教师反馈的稠密性；博客在推理(数学)、个性化(指令遵循恢复)、多轮工具使用三场景演示，配套 tinker-cookbook 给出可复现 LoRA 配方。

**元信息**：博客 https://thinkingmachines.ai/blog/on-policy-distillation/（无 arXiv，非论文）｜ Thinking Machines Lab（Kevin Lu 等）｜ 2025 ｜ 主题 T?/High（on-policy 蒸馏方法论）｜ 代码 https://github.com/thinking-machines-lab/tinker-cookbook（已 clone，14MB）｜ 框架 Tinker SDK（自研托管训练 SDK，基于 LoRA），非 veRL/TRL

## 1. 相关工作与进展
大模型蒸馏分两类：off-policy（在教师生成的固定数据上做 SFT）与 on-policy（学生采样自身轨迹、教师对学生轨迹逐 token 打分）。off-policy/SFT 易"复述"教师局部模式且存在训练-推理分布不匹配（exposure bias）；on-policy RL（如 RLVR）反馈稀疏、采样成本高。博客把 on-policy distillation 作为兼顾稠密监督与 on-policy 真实性的范式系统推广（配套 off-policy 与 SDFT recipe 作对照）。

## 2. 现有工作存在的问题

- off-policy 蒸馏/SFT：学生在自身推理轨迹上从未被纠正，误差累积；
- on-policy RL（GRPO）：仅序列级稀疏奖励，token 学习效率低；
- 单纯模仿教师文本无法在"学生自己访问到的状态"上获得稠密反馈。

## 3. Motivation
让学生在自身采样轨迹上，以教师逐 token 分布为稠密监督（KL），从而结合 on-policy 分布真实性与教师反馈稠密性；并展示该范式在推理、个性化、多轮工具使用三场景的有效性与效率。

## 4. 主要灵感 / 核心直觉
RL 的问题是奖励稀疏、SFT 的问题是分布失配；若让学生自己采样（解决失配），同时让教师对每个 token 打分（解决稀疏），就能两全。教师 logprob 提供的稠密信号比序列级 outcome reward 样本/步效率高得多。

## 5. 主要解决思路(一段话讲清核心)
学生采样响应，环境不提供任何 reward（既非正确性也非格式），唯一监督是最小化学生与教师在**学生轨迹**上的逐 token **reverse KL**（KL[p‖q]=log p_student − log q_teacher，仅在学生采样到的 token 上算）。代码上把 **−kl_penalty_coef·reverse_KL 作为 per-token advantage**，经 Tinker 的 importance-sampling loss 回传——即用策略梯度形式实现 token 级 KL 蒸馏，而非直接的 KL 散度损失项。全程 LoRA。

## 6. 方法详解(通俗、分步骤)
代码确认（`distillation/train_on_policy.py::incorporate_kl_penalty`）：

1. 学生采样轨迹，记录 token 级 sampled_logprobs(p) 与 mask。
2. 对每条 datum 用其对应 teacher sampling client 计算 teacher_logprobs(q)。
3. reverse_KL = (log p − log q)·mask；逐 token advantage = −kl_penalty_coef·mask·reverse_KL（默认 kl_penalty_coef=1.0）。
4. 可选 `kl_discount_factor`（默认 0.0）优化折扣未来 KL——博客实验称无明显增益。
5. 把该 advantage 加到 datum 的 advantages，经 importance-sampling loss 更新。
6. 仅对学生生成 token 计 loss（系统提示、用户消息、工具返回、assistant header 等被 mask）。
7. **多教师**：对每个数据集配 (teacher_model, groups_per_batch)，分别采样后拼接成批；trainer 从各配置采样再拼接。
8. **多轮工具使用**（`harbor_multiturn.py`）：在 Harbor sandbox 用 `reward_fn=zero_reward`（恒返回 0）覆盖默认 HarborReward，唯一信号为对教师的 KL；复用 `tool_use` 库 + `harbor_rl` recipe。

## 7. 实验数据集

- 推理：SFT 用 OpenThoughts3-1.2M；on-policy 蒸馏用 DeepMath-103K；评测 AIME'24。
- 个性化：内部文档 + 重采样 Tulu3（assistant 轮由 Qwen3-8B 重新生成）做 SFT，Tulu3 prompts 做 on-policy 蒸馏；评测 IFEval。
- 多轮工具使用：Harbor sandbox 任务（如 terminal-bench@2.0）。

## 8. 实验结果与主要发现

- **推理**：① OpenThoughts3 上 SFT（rank-128 LoRA，lr=1e-3 LoRA / 1e-4 full，batch=128，3000 步）→ AIME'24 约 55%；② 加载该 ckpt 在 DeepMath 上 on-policy 蒸馏（lr=1e-4 LoRA / 5e-5 full，groups_per_batch=512，rank-128，约 100 步）→ AIME'24 约 65%。即仅约 100 步蒸馏即从 55% 升到 65%。
- **个性化**：SFT 初始化后在 Tulu3 prompts 上蒸馏（lr=1e-4，groups_per_batch=64），IFEval 约 100 步恢复。
- **多轮工具使用**：README 示例以 **Kimi-K2-Thinking** 同时作学生与教师（max_turns=10、group_size=4、groups_per_batch=8、lora_rank=8、kl_penalty_coef=1.0），在 Harbor 沙箱中以纯 KL 信号训练〔原稿仅写 Qwen3，已据 README 补正为 Kimi-K2-Thinking 示例〕。
- 提供各 LoRA rank(8/32/128) 的 Tinker checkpoint 句柄复现。

## 9. 结果如何支撑其主张
"SFT 55% → on-policy 蒸馏约 100 步 65%"直观显示在学生自身轨迹上用教师稠密信号的样本/步高效性，对照 off-policy SFT 的瓶颈支撑"on-policy + 稠密 KL"的优势。三场景（推理/个性化/工具）覆盖支撑该范式的通用性。代码与 checkpoint 公开使主张可复现。

## 10. 逻辑自洽性(中性评估)
方法论清晰、代码与博客一致（reverse KL 作 advantage、纯 KL 无 reward、仅学生 token 计 loss 均经代码核实）。但需注意定位：这是**博客 + 配方**而非受控论文，多数结论以单点曲线/示例形式给出，缺乏严格的多 seed/基线对照与统计显著性；"约 55%→约 65%"为近似值。多教师在博客中**未展示**（仅 recipe 提供）。teacher 用 Qwen3-32B 推理、个性化用更强模型，学生与教师同族（共享 renderer/tokenizer）降低了分布漂移，跨家族教师需自行改 renderer（代码注释明确提示），通用性边界未系统评估。

## 11. 残留问题 / 局限

- 非论文，无严格基线/统计，数值为近似单点。
- on-policy 蒸馏需教师在线 logprob 计算，依赖 Tinker 托管服务，复现门槛绑定该 SDK。
- 教师与学生不同 renderer/tokenizer 时需手工对齐（代码留有 TODO 提示），跨家族蒸馏未演示。
- 多教师能力仅在 recipe 提供、博客未展示；reverse-KL（mode-seeking）相对 forward-KL 的取舍未深入讨论。
- kl_discount_factor 在实验中无增益，其适用场景不明。

## 12. 开源代码与框架(链接+框架+代码可得性)

- 代码：https://github.com/thinking-machines-lab/tinker-cookbook （已 clone，14MB）。核心：`tinker_cookbook/distillation/`（`train_on_policy.py`、`train_off_policy.py`、`sdft.py`、`datasets.py`）与 `tinker_cookbook/recipes/distillation/`（`on_policy_distillation.py`、`off_policy_reasoning.py`、`on_policy_multi_teacher.py`、`harbor_multiturn.py`、`on_policy_distillation_harbor_multi_turn.py`），含 `recipes/distillation/README.md` 复现脚本。
- 框架：Tinker SDK（基于 LoRA 的自研托管训练 SDK）；蒸馏 loss 与采样由 Tinker 服务侧执行，cookbook 仅给 recipe 与超参。启动如 `python -m tinker_cookbook.recipes.distillation.on_policy_distillation model_name=Qwen/Qwen3-8B-Base ... lora_rank=128`。
- 关键超参/默认：kl_penalty_coef=1.0、kl_discount_factor=0.0、group_size=4、单教师默认 teacher=Qwen3-8B、学生 Qwen3-8B-Base；多教师 recipe 默认 DeepMath→Qwen3-32B、Tulu3→Qwen3-235B-A22B-Instruct-2507、学生 Qwen3-8B，各 groups_per_batch=512〔已据 `on_policy_multi_teacher.py` 补正 Tulu3 教师为 Qwen3-235B〕。
- 配套提供各 LoRA rank(8/32/128) 的 Tinker checkpoint 句柄。

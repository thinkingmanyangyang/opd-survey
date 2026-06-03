ampo | AMPO: Adaptive Multi-Guidance Policy Optimization for Diverse Exploration | 同济 / 香港理工 / 上海AI Lab / 新加坡国立 / 电子科大（Xiaoyang Yuan 等，通讯 Yi Bin） | 2025-10-09 arXiv v2 | L3 RLVR/GRPO家族 + L1 Mixed-Policy蒸馏 · 相关性高（与"探索-巩固/按需脚手架"强相关）

**原始论文**：https://arxiv.org/abs/2510.02227 （arXiv 标题 "More Than One Teacher: Adaptive Multi-Guidance Policy Optimization for Diverse Exploration"；README 标题 "Many Small Teachers Beat One Giant"，同一 arXiv id）

## 一眼看懂
- 🟦 TL;DR：on-policy RLVR(GRPO) 的探索被困在模型自身知识边界内，难题持续全失败→稀疏奖励→训练不稳。AMPO 在 GRPO 之上：**只在某 query 整组采样全失败时**(guidance-on-demand)，才从一个**多教师池**里挑正确解替换掉失败样本；挑哪条不看"哪个教师最强"，而看**"哪条对学生最易吸收"**(学生顺着该教师推理生成正确答案的几何平均概率最高，§3.3 Eq.5)。用 4 个 7B 级同伴教师 + 仅 8.5k 数据，媲美单一更强教师(DeepSeek-R1)+46k 数据(§4.2)，且全程保持更高 entropy。
- 最巧的一步：**guidance-on-demand 的触发条件——只在 on-policy 整组全部 reward<τ 时才替换**（§3.2 Eq.3-4）。抽掉它(改成静态注入)就垮——会无差别灌入学生本可自己解出的题，扼杀自探索、降低 entropy；正是"仅在自探索失败才兜底"既保证每步都有正确解可学、又最大化自探索价值。其次关键是 Eq.5 的可理解度选择(易吸收 > 最正确)。

## 为什么做
- 研究背景：RLVR(可验证奖励 RL)是提升 LLM LongCoT 推理的有效范式(§1)，但 on-policy GRPO 的探索受限于自身知识边界(Yue/Gandhi 2025)，"capacity-difficulty mismatch"导致难题持续失败、稀疏奖励、训练不稳。社区做 Mixed-Policy RL 把强教师 off-policy 轨迹混入 on-policy（如 LUFFY）或交织 RL/SFT。
- 解决的具体痛点（§2 Mixed-Policy RL 段，两条）：①**多依赖单一教师**——限制学习多样性，把单教师固有偏置/风格带进学生；②**静态数据整合**——不看模型当前需求与"可理解度"，无差别注入外部解，可能注入学生根本吸收不了的解法，浪费且扰动训练。
- 相关工作 & 各自不足：单教师蒸馏 OpenR1/OpenThought/AM(SFT 促记忆而非理解，OOD 差)；LUFFY(单教师 off-policy 混入，固定方式)；prompt 式引导 / SFT 辅助损失(Fu/Zhang 2025)。AMPO 的Δ：用 Multi-Guidance Pool 替单教师 + 用自适应可理解度机制替静态注入。
- 动机链：现状(单教师 + 静态注入)→缺陷(多样性受限 + 注入学生吸收不了的解)→所以多教师(借 KD 多教师思想) + 按需(只在全失败触发) + 按可理解度选路。为什么不用单一更强教师 SFT？§2 论证 SFT 促记忆、OOD 差，且单教师限制探索广度。
- 与最近邻工作的Δ：相对 LUFFY（最近邻，单教师 Mixed-Policy，沿用其 shaping `f(x)=x/(x+0.1)`），差在**多教师池 + on-demand 触发 + 可理解度选路**三点；相对多教师 SFT 蒸馏，差在**保持 on-policy 自探索 + 仅兜底**而非全程灌入。

## 怎么做 + 靠不靠谱
- 方法流水线（§3，Fig 1）：① π_old 对 query 采 G 个解，rule-based verifier 评分 → ② 若**整组全部 reward<τ**(Eq.3 置 I=True) 触发替换 → ③ 从 Multi-Guidance Pool P_G(多教师正确解) 按可理解度 r_p 选 top-k，替换掉 k 个错误 on-policy 解，`k=min(k0, N_g)`(Eq.4)，否则不替换、纯 GRPO → ④ 在增广 batch G_aug 上统一归一化算 advantage(Eq.6) → ⑤ 混合目标 J_Mixed 更新(Eq.7)：**off-policy 用 sequence-level 聚合**(每条教师解等权)、**on-policy 用 token-level 聚合**(DAPO 式)。N_off=0 时无缝退化为 GRPO。
- 逐组件必要性：
  - **Adaptive Multi-Guidance Replacement(on-demand 触发)**：核心，§4 与静态注入/LUFFY 对比 + 保持高 entropy 佐证；没它退化为静态蒸馏。
  - **Comprehension-based Selection(r_p, Eq.5)**：选学生最易吸收的教师路径；论文做了选择策略消融(r_p vs 随机/最强)；平局取更短路径。
  - **多教师池**：§4 分析教师池组成的影响；"4 个同伴 ≈ 单一 R1+46k"是其数据效率主张的支撑。
  - **off-policy sequence-level 聚合**：§3.4 明确——不同教师序列长度不一，token 级等权会让长序列主导梯度，故用序列级保证每条教师解等权。代码 `compute_token_on_seq_off_policy_loss` 确证。
- 关键机制/公式（直觉）：r_p = `clip(exp(平均 log π_θ(正确答案 token | 教师推理 z_off)), 0, 1)`(Eq.5)。白话：顺着这条教师推理，学生自己几乎就能写出正确答案 = 最易吸收 = 脚手架搭在学生够得着的高度。J_Mixed(Eq.7) off-policy 重要性比 `r̂_j,t = π_θ/π_φj`，on-policy `r_i,t = π_θ/π_old`，均过 `f(x)=x/(x+0.1)` shaping(沿用 LUFFY)。
- 实验与证据：教师池 4 个 7B 级 LongCoT(AceReason-Nemotron-1.1-7B、DeepSeek-R1-Distill-Qwen-7B、OpenR1-Qwen-7B、Qwen3-8B-thinking)；训练数据 8.5k(从 OpenR1-Math-46k 多教师 curation)；基模主 Qwen2.5-7B-Ins，另 Qwen2.5-1.5B-Ins、LLaMA3.2-8B-Ins。6 数学 ID + 3 OOD(ARC-c/GPQA*/MMLU-Pro)，AIME/AMC 报 Avg@32 其余 Pass@1。关键数字(Table 1)：相比 GRPO 数学平均 **+4.3%**、OOD **+12.2%**；在 1.5B/Llama 上同样优于 GRPO(如 Llama3.2-8B OOD 38.9→56.9)。提升 Pass@k、训练全程更高 entropy(未坍塌)。**注意**：Qwen2.5-7B 主表上 AMPO ID Avg 与 SFT+GRPO/LUFFY 接近甚至略低(LUFFY 41.0、SFT+GRPO 42.7，AMPO 该行被截断未完整抄)，其核心卖点是**数据效率(8.5k vs 46k) + OOD + entropy**而非绝对刷分。baseline 含 SFT/GRPO/SFT+GRPO/LUFFY，公平。
- 假设与失效边界：【原文】§3.1 省略 KL 项(随 Yu/Yan 2025)；需现成多教师 + format reward 以提取答案。【推断】教师池需 4 个现成强教师，构建/采样成本不小；阈值 τ、替换数 k0 为超参，跨任务自适应性未验；主验证集中数学+少量 OOD。**关键实现细节**(已核真实代码 `mix_core_alg.py`)：off-policy 重要性比有两条路径——(a) 传 `target_probs` 时 `off_ratio=π_θ/π_target`(line 234，对应论文 Eq.7 `π_θ/π_φj`)；(b) 默认 `target_probs is None` 时 `off_ratio=π_θ` 再经 `p_div_p_0.1` 整形为 `π_θ/(π_θ+0.1)`(line 222-223)，**教师概率根本不进入 ratio**。而 `train_ampo.sh` line 33 设 `off_policy_reshape="p_div_p_0.1"` 且未传 target_probs → **默认实验跑的是 (b)，与论文 Eq.7 写的 (a) 数学不等价**。
- 祛魅总结：【推断】真贡献是把 KD 的"多教师 + 可理解度"思想干净地搬进 Mixed-Policy RLVR，且 on-demand 触发 + r_p 选择 + 数据效率对比(8.5k≈46k)说服力强。**包装/缺口**：①论文 Eq.7 的重要性校正 `π_θ/π_φj` 在默认脚本里并未执行(退化成 LUFFY 式学生概率 shaping)——读者据公式理解会与实际跑的代码错位，这是真实的理论-实现缺口；②"+4.3%/+12.2%"是相对 GRPO，相对 LUFFY/SFT+GRPO 的 ID 优势不明显，卖点实为数据效率与 OOD。作者在重要性采样的理论叙述上**高估**了与实现的一致性。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=多教师正确解轨迹(off-policy) + 自身 rollout(on-policy)，按可理解度 r_p 选择；优化用 verifiable reward 的 GRPO advantage | **改什么**=策略参数(混合策略梯度) | **何时改**=在线 per-step(每个 GRPO 更新步按需替换) | **免梯度?**=否(策略梯度) | **记忆-技能生命周期**=Multi-Guidance Pool 是静态预构建的教师解库(写入=离线 curation，检索=按 r_p top-k，无遗忘/共享机制)——近似"外部技能/示范库"但不在线更新 | **防遗忘机制**=无显式;靠 on-demand 仅兜底 + 保持高 entropy 间接维持探索多样性。
- ⑦ 开源代码+框架/harness：https://github.com/SII-Enigma/AMPO （**已克隆约 163M，完整**：含 data/、exp_scripts/(train_ampo.sh)、eval_scripts/、figures/、HF checkpoint）。框架=**verl**，GRPO 的 Mixed-Policy 扩展。核心改动 `ampo/verl/verl/adaptive_mix_src/`(`mix_core_alg.py` 含两条 off-policy ratio 路径 line 212-236、`mix_actor.py`、`mix_dapo_trainer.py`、`mix_fsdp_worker.py`)；FSDP 或 Megatron，vllm/sglang rollout。
- 💰 资源/成本与可扩展性：【原文】超参在 Appendix A.1.4。【推断】需预先用 4 个 7B 教师对 46k 题采样并 curation 出 8.5k 解池(一次性离线成本)，训练本体是 7B GRPO + 每步少量额外 off-policy 前向算 r_p；数据效率高(8.5k)但教师池构建非零成本。
- 🎯 对"探索-巩固"对标：**最强支撑/可直接借鉴者之一**——AMPO 的"on-demand 仅在自探索全失败时才注入 + 按可理解度选学生够得着的教师路径"几乎是 TSRD"**稀疏脚手架 + 探索(选路)+ 路径恢复**"的现成 RL 实现：替换失败样本 = path-recovery 兜底；r_p 选"学生差一点就能自己走到"的路径 = 脚手架搭在够得着的高度；保持高 entropy = 保探索。**可借组件**：① guidance-on-demand 触发逻辑；② Probability Reward r_p 作"可吸收性/接管点"度量；③ off-policy sequence-level 等权聚合。**缺口/差异**：AMPO 用**整条教师正确解替换整条失败解**(response 级),不是 TSRD 的"**单点接管 + 自选恢复分支**"(step 级 path-recovery);student 不"自选"恢复路径而是被给定 top-k 教师解；无 MTP 前瞻。一句判定：**强相关、机制可直接借鉴的 RLVR 落地，但接管粒度(整条 vs 单步)是与 TSRD 的关键差异**。依据：替换发生在 response 级、由 r_p 选定而非 student 自选恢复分支。
- 🔭 开放问题/未来方向：【原文】§1 末——引导替换数量 k 与教师池组成的影响是 future research 方向。【推断】把 response 级替换细化到 step 级单点接管(逼近 TSRD path-recovery);让 student 自选恢复分支而非给定 top-k;统一论文 Eq.7 与默认脚本的重要性采样;τ/k0 的任务自适应;教师池在线更新(向"技能库持续学习"靠拢)。

ampo | AMPO: Adaptive Multi-Guidance Policy Optimization for Diverse Exploration | 同济 / 香港理工 / 上海AI Lab / 新加坡国立 / 电子科大（Xiaoyang Yuan 等，通讯 Yi Bin） | 2025-10-09 arXiv v2 | L3 RLVR/GRPO家族 + L1 Mixed-Policy蒸馏 · 相关性高（与"探索-巩固/按需脚手架"强相关）

**原始论文**：https://arxiv.org/abs/2510.02227 （arXiv 标题 "More Than One Teacher: Adaptive Multi-Guidance Policy Optimization for Diverse Exploration"；README 标题 "Many Small Teachers Beat One Giant"，同一 arXiv id）

## 一眼看懂
- 🟦 TL;DR：on-policy RLVR(GRPO) 的探索被困在模型自身知识边界内，难题持续全失败→稀疏奖励→训练不稳。AMPO 在 GRPO 之上：**只在某 query 整组采样全失败时**(guidance-on-demand)，才从一个**多教师池**里挑正确解替换掉失败样本；挑哪条不看"哪个教师最强"，而看**"哪条对学生最易吸收"**(学生顺着该教师推理生成正确答案的几何平均概率最高，§3.3 Eq.5)。用 4 个 7B 级同伴教师 + 仅 8.5k 数据，媲美单一更强教师(DeepSeek-R1)+46k 数据(§4.2)，且全程保持更高 entropy。
- 最巧的一步：**guidance-on-demand 触发条件——只在 on-policy 整组全部 \(R(o_i)<\tau\) 时才替换**（§3.2 Eq.3-4）。抽掉它(改成静态注入)就垮——会无差别灌入学生本可自己解出的题，扼杀自探索、降低 entropy；正是"仅在自探索失败才兜底"既保证每步都有正确解可学、又最大化自探索价值。其次关键是 Eq.5 的可理解度选择(易吸收 > 最正确)。

## 为什么做
- 研究背景：RLVR(可验证奖励 RL)是提升 LLM LongCoT 推理的有效范式(§1)，比 SFT 更能在数学等复杂域学到稳健推理。但 on-policy GRPO 的探索受限于自身知识边界(Yue/Gandhi 2025)，能精炼已有技能却难获取远超初始能力的新知识；"capacity-difficulty mismatch"(Yu 2025a/Liu 2025e)导致难题持续失败、稀疏奖励、训练不稳。
- 解决的具体痛点（§2 Mixed-Policy RL 段，两条）：①**多依赖单一教师**——限制学习多样性，把单教师固有偏置/风格带进学生(Tian/Xu 2025)；②**静态数据整合**——不看模型当前需求与"可理解度"，无差别注入外部解，可能注入学生根本吸收不了的解法，浪费且扰动训练。
- 相关工作 & 并行技术路线（每条具体短板，§2 三段）：
  - **KD 蒸馏推理线**：OpenR1/OpenThought/AM 用 DeepSeek-R1 蒸大量示范数据 SFT 小模型；Tian/Xu 2025 指出单教师**限制学习视角、降解题多样性、束缚探索深广度**→提多教师聚合(Beyer 2022 的 KD 多教师思想)。但 KD via SFT 被批**促记忆而非真正理解**(Chu 2025)→训练分布外性能差。
  - **RLVR 线**：GRPO/RLVR 激励模型自主发展推理，但 on-policy 把模型困在固有知识边界、主要放大已有能力(Yue/Gandhi 2025) + capacity-difficulty mismatch。
  - **Mixed-Policy RL 线(最近邻)**：把强教师 off-policy 轨迹混入 on-policy——(a) **LUFFY**(Yan 2025，混专家数据进 rollout，沿用其 shaping `f(x)=x/(x+0.1)`)；(b) 交织 RL/SFT(Ma 2025)；(c) external guidance 作 prompt(Liu 2025b/Wu 2025)；(d) SFT 目标作 RL 辅助损失(Fu/Zhang 2025)。两大共性短板：**单教师 + 静态注入**。AMPO 的Δ：Multi-Guidance Pool 替单教师 + 自适应可理解度机制替静态注入。
- 动机链：现状(单教师 + 静态注入)→缺陷(多样性受限 + 注入学生吸收不了的解)→所以多教师(借 KD 多教师思想) + 按需(只在全失败触发) + 按可理解度选路。为什么不用单一更强教师 SFT？§2 论证 SFT 促记忆、OOD 差，且单教师限制探索广度。
- 与最近邻工作的精确差异：相对 LUFFY（最近邻，单教师 Mixed-Policy，沿用其 shaping），差在**多教师池 + on-demand 触发 + 可理解度选路**三点；相对多教师 SFT 蒸馏，差在**保持 on-policy 自探索 + 仅兜底**而非全程灌入。

## 怎么做 + 靠不靠谱
### 0. 基座 GRPO（§3.1）
\(\pi_{\theta_{\mathrm{old}}}\) 对 query \(q\) 采 \(G\) 个解，组内归一化算 advantage：
\(\displaystyle A_{i,t}=\frac{R(o_i)-\mathrm{mean}(\{R(o_i)\}_{i=1}^{G})}{\mathrm{std}(\{R(o_i)\}_{i=1}^{G})} \quad(\text{Eq.1})\)
\(R(\cdot)\) 是 rule-based verifier(二值正确性)。目标(本文实现)：
\(\displaystyle J_{\mathrm{GRPO}}(\pi_\theta)=\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\big[r_{i,t}A_{i,t},\,\mathrm{clip}(r_{i,t},1-\epsilon,1+\epsilon)A_{i,t}\big] \quad(\text{Eq.2})\)
其中重要性比 \(r_{i,t}=\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}|q,o_{i,<t})}\)。随 Yu/Yan 2025 **省略 KL 项**。

### 1. 自适应多教师替换（§3.2，guidance-on-demand）
- 预构建 **Multi-Guidance Pool** \(P_G\)：多个不同教师对题目的**正确** off-policy 解。
- 触发判据：\(\pi_{\theta_{\mathrm{old}}}\) 先采 \(G\) 个解；若**全部** reward 低于阈值 \(\tau\) 则置替换标志
  \(\displaystyle I=\begin{cases}\text{True}&\text{if }R(o_i)<\tau,\ \forall i\in\{1,\dots,G\}\\ \text{False}&\text{otherwise}\end{cases} \quad(\text{Eq.3})\)
  即"只有一条 on-policy 解都不对"时才兜底。
- 若 \(I=\text{True}\)：随机选 \(k\) 个错误 on-policy 解，用按可理解度选出的 top-\(k\) off-policy 解替换，\(k=\min(k_0,N_g)\)(\(k_0\)=目标替换数，\(N_g\)=池中可用解数)，形成增广 batch
  \(\displaystyle G_{\mathrm{aug}}=\begin{cases}\{o_i\sim\pi_{\theta_{\mathrm{old}}}\}_{i=1}^{N_{\mathrm{on}}}\cup\{o_j\in P_G\}_{j=1}^{N_{\mathrm{off}}}&\text{if }I=\text{True}\\ \{o_i\sim\pi_{\theta_{\mathrm{old}}}\}_{i=1}^{G}&\text{otherwise}\end{cases} \quad(\text{Eq.4})\)
  \(N_{\mathrm{on}}=G-k\)、\(N_{\mathrm{off}}=k\)。保证每步都有正确解可学，同时优先自探索路径。

### 2. 可理解度选路（§3.3，核心创新之二）
- 设池中一条 off-policy 解 \(o_{\mathrm{off}}=(z_{\mathrm{off}},y)\)(教师推理 \(z_{\mathrm{off}}\) + 答案 \(y\))；构造修正轨迹 \(o^*=(z_{\mathrm{off}},y^*)\)(把答案换成 ground-truth \(y^*\))。
- **Probability Reward \(r_p\)** = 学生顺着 \(z_{\mathrm{off}}\) 生成正确答案 token 的几何平均概率(用平均对数概率算并裁剪到 [0,1])：
  \(\displaystyle r_p(o_{\mathrm{off}})=\mathrm{clip}\Big(\exp\big(\tfrac{1}{|y^*|}\sum_{\tau_i\in y^*}\log\pi_\theta(\tau_i|z_{\mathrm{off}},y^*_{<i})\big),\,0,\,1\Big) \quad(\text{Eq.5})\)
- 直觉：\(r_p\) 高 = 教师推理与学生内部知识表征更对齐 = 更易吸收 = 脚手架搭在学生够得着的高度。按 \(r_p\) 排序取 top-\(k\)；并列时**取更短(更简洁)路径**当 tie-breaker。需配 format reward 以提取 \(y\)(§A.1.3)。

### 3. 混合策略优化（§3.4）
- 增广 batch \(G_{\mathrm{aug}}\) 内统一归一化算 advantage：
  \(\displaystyle \hat A_{i,t}=\frac{R(o_i)-\mathrm{mean}(\{R(o_i)|o_i\in G_{\mathrm{aug}}\})}{\mathrm{std}(\{R(o_i)|o_i\in G_{\mathrm{aug}}\})} \quad(\text{Eq.6})\)
- 混合目标(off-policy 教师策略记 \(\pi_{\phi_j}\))：
  \(\displaystyle J_{\mathrm{Mixed}}(\theta)=\underbrace{\frac{1}{N_{\mathrm{off}}}\sum_{j=1}^{N_{\mathrm{off}}}\frac{1}{|o_j|}\sum_{t=1}^{|o_j|}\mathrm{CLIP}\big(f(\hat r_{j,t}),\hat A_{j,t},\epsilon\big)}_{\text{off-policy（序列级聚合）}}+\underbrace{\frac{1}{T_{\mathrm{on}}}\sum_{i=1}^{N_{\mathrm{on}}}\sum_{t=1}^{|o_i|}\mathrm{CLIP}\big(r_{i,t},\hat A_{i,t},\epsilon\big)}_{\text{on-policy（token 级聚合）}} \quad(\text{Eq.7})\)
  其中 \(\mathrm{CLIP}(r,A,\epsilon)=\min[r A,\mathrm{clip}(r,1-\epsilon,1+\epsilon)A]\)；off-policy 重要性比 \(\hat r_{j,t}=\frac{\pi_\theta(o_{j,t}|q,o_{j,<t})}{\pi_{\phi_j}(o_{j,t}|q,o_{j,<t})}\)、on-policy \(r_{i,t}=\frac{\pi_\theta}{\pi_{\theta_{\mathrm{old}}}}\)；\(T_{\mathrm{on}}=\sum_i|o_i|\)；shaping \(f(x)=\frac{x}{x+0.1}\)(沿用 LUFFY)。
- **聚合粒度设计(§3.4 明确)**：off-policy 用**序列级**(每条教师解等权)，因不同教师序列长度不一，token 级等权会让长序列主导梯度引入 bias；on-policy 用 token 级(DAPO 式)。\(N_{\mathrm{off}}=0\) 时无缝退化为 GRPO。

### 4. 逐组件必要性
- **Adaptive Multi-Guidance Replacement(on-demand 触发)**：核心，§4 与静态注入/LUFFY 对比 + 保持高 entropy 佐证；没它退化为静态蒸馏。
- **Comprehension-based Selection(\(r_p\))**：选学生最易吸收的教师路径；论文做选择策略消融(\(r_p\) vs 随机/最强)；平局取更短路径。
- **多教师池**：§4 分析教师池组成影响；"4 同伴 ≈ 单 R1+46k"是数据效率主张支撑。
- **off-policy 序列级聚合**：§3.4 明确论证；代码 `compute_token_on_seq_off_policy_loss` 确证。

### 5. 实验与证据
- 教师池 4 个 7B 级 LongCoT(AceReason-Nemotron-1.1-7B、DeepSeek-R1-Distill-Qwen-7B、OpenR1-Qwen-7B、Qwen3-8B-thinking)；训练数据 8.5k(从 OpenR1-Math-46k 多教师 curation)；基模主 Qwen2.5-7B-Ins，另 Qwen2.5-1.5B-Ins、LLaMA3.2-8B-Ins。6 数学 ID + 3 OOD(ARC-c/GPQA*/MMLU-Pro)，AIME/AMC 报 Avg@32 其余 Pass@1，temp 0.6。
- 关键数字(Table 1)：相比 GRPO 数学平均 **+4.3%**、OOD **+12.2%**；在 1.5B/Llama 上同样优于 GRPO(Llama3.2-8B OOD 38.9→**56.9**、ID 16.6→24.5)。提升 Pass@k、训练全程更高 entropy(未坍塌)。
- **关键 caveat(已核 PDF Table 1)**：Qwen2.5-7B 主表上 AMPO **ID Avg 40.4** 略低于 SFT+GRPO(42.7)、LUFFY(41.0)，但 **OOD Avg 64.2 最高**(超 LUFFY 58.0、SFT+GRPO 62.7)。其核心卖点是**数据效率(8.5k vs 46k) + OOD + entropy**而非绝对刷 ID 分。baseline 含 SFT(32k)/GRPO(8.5k)/SFT+GRPO/LUFFY(46k)，公平。
- 假设与失效边界：【原文】§3.1 省略 KL 项(随 Yu/Yan 2025)；需现成多教师 + format reward 以提取答案。【推断】教师池需 4 个现成强教师，构建/采样成本不小；阈值 \(\tau\)、替换数 \(k_0\) 为超参，跨任务自适应性未验；主验证集中数学+少量 OOD。
- **理论-实现关系(已重新核真实代码，比 v1 更精确)**：`mix_core_alg.py` 的 off-policy 重要性比有两条路径——(a) 传入 `target_probs`(教师概率，dataset key `target_ds_qwen_7b_probs`)时 line 234 `off_ratio=exp(log_prob)/(target_probs+1e-6)=π_θ/π_φ`，**与 Eq.7 一致**；(b) `target_probs is None` 时 line 213+222 `off_ratio=exp(log_prob)` 经 `p_div_p_0.1` 整形为 `π_θ/(π_θ+0.1)`，**教师概率不进入 ratio**(退化成 LUFFY 式学生概率 shaping)。`mix_actor.py:223` 仅在 `'target_probs' in data` 时才传该字段；`rl_dataset_with_target.py:210-232` 显示数据若含该 key 则加载、否则 `target_probs` 全填 -1。`train_ampo.sh:33` 设 `off_policy_reshape="p_div_p_0.1"`，其 `TRAIN_FILE=openr1_math_4longcots.parquet` **未随仓库提供**(克隆 `data/` 仅含 validation parquet)，故**无法从克隆确认默认训练数据是否含 `target_ds_qwen_7b_probs`**：若含→走 (a) 与论文一致；若不含→走 (b) 与 Eq.7 不等价。【待核：默认训练 parquet 是否含教师概率字段】——v1 断言"默认必跑 (b)、与 Eq.7 矛盾"过强；准确表述应是"两条路径并存，取决于训练数据是否带教师概率，克隆无法判定默认走哪条"。
- 祛魅总结：【推断】真贡献是把 KD 的"多教师 + 可理解度"思想干净地搬进 Mixed-Policy RLVR，on-demand 触发 + \(r_p\) 选择 + 数据效率对比(8.5k≈46k)说服力强。**包装/缺口**：① Eq.7 重要性校正 `π_θ/π_φj` 是否真执行取决于训练数据是否预存教师概率，克隆未带该数据→读者按公式理解可能与实际跑的代码错位(真实的理论-实现可核性缺口)；②"+4.3%/+12.2%"是相对 GRPO，相对 LUFFY/SFT+GRPO 的 **ID 优势不明显甚至略低**，卖点实为数据效率与 OOD。作者在重要性采样叙述与默认配置的一致性上**留了可核性缺口**。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=多教师正确解轨迹(off-policy) + 自身 rollout(on-policy)，按可理解度 \(r_p\) 选择；优化用 verifiable reward 的 GRPO advantage | **改什么**=策略参数(混合策略梯度) | **何时改**=在线 per-step(每个 GRPO 更新步按需替换) | **免梯度?**=否(策略梯度) | **记忆-技能生命周期**=Multi-Guidance Pool 是静态预构建的教师解库(写入=离线 curation，检索=按 \(r_p\) top-k，无遗忘/共享)——近似"外部技能/示范库"但不在线更新 | **防遗忘机制**=无显式；靠 on-demand 仅兜底 + 保持高 entropy 间接维持探索多样性。
- ⑦ 开源代码+框架/harness：https://github.com/SII-Enigma/AMPO （**已克隆约 163M，完整**：含 data/(仅 validation parquet)、exp_scripts/(train_ampo.sh)、eval_scripts/、figures/）。框架=**verl**，GRPO 的 Mixed-Policy 扩展。核心改动 `ampo/verl/verl/adaptive_mix_src/`(`mix_core_alg.py` 含两条 off-policy ratio 路径 line 212-234、`mix_actor.py`、`mix_dapo_trainer.py`、`mix_fsdp_worker.py`、`rl_dataset_with_target.py`)；FSDP 或 Megatron，vllm/sglang rollout。**未完整**：默认训练数据 `openr1_math_4longcots.parquet` 未随仓库提供，需手动获取/重新 curation。
- 💰 资源/成本与可扩展性：【原文】超参在 Appendix A.1.4。【推断】需预先用 4 个 7B 教师对 46k 题采样并 curation 出 8.5k 解池(一次性离线成本，若走 Eq.7 (a) 路径还需预存教师 token 概率)；训练本体是 7B GRPO + 每步少量额外 off-policy 前向算 \(r_p\)；数据效率高(8.5k)但教师池构建非零成本。
- 🎯 对"探索-巩固"对标：**最强支撑/可直接借鉴者之一**——AMPO 的"on-demand 仅在自探索全失败时才注入 + 按可理解度选学生够得着的教师路径"几乎是 TSRD"**稀疏脚手架 + 探索(选路)+ 路径恢复**"的现成 RL 实现：替换失败样本 = path-recovery 兜底；\(r_p\) 选"学生差一点就能自己走到"的路径 = 脚手架搭在够得着的高度；保持高 entropy = 保探索。**可借组件**：① guidance-on-demand 触发逻辑；② Probability Reward \(r_p\) 作"可吸收性/接管点"度量；③ off-policy 序列级等权聚合。**缺口/差异**：AMPO 用**整条教师正确解替换整条失败解**(response 级)，不是 TSRD 的"**单点接管 + 自选恢复分支**"(step 级 path-recovery)；student 不"自选"恢复路径而是被给定 top-k 教师解；无 MTP 前瞻。一句判定：**强相关、机制可直接借鉴的 RLVR 落地，但接管粒度(整条 vs 单步)是与 TSRD 的关键差异**。依据：替换发生在 response 级、由 \(r_p\) 选定而非 student 自选恢复分支。
- 🔭 开放问题/未来方向：【原文】§1 末——引导替换数量 \(k\) 与教师池组成的影响是 future research 方向。【推断】把 response 级替换细化到 step 级单点接管(逼近 TSRD path-recovery)；让 student 自选恢复分支而非给定 top-k；统一/公开 Eq.7 与默认脚本的重要性采样配置(含教师概率数据)；\(\tau/k_0\) 的任务自适应；教师池在线更新(向"技能库持续学习"靠拢)。

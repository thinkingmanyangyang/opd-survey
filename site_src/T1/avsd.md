# avsd — AVSD: Adaptive-View Self-Distillation by Balancing Consensus and Teacher-Specific Privileged Signals

> **一句话重点 (TL;DR)**：同一个模型当 student 又当 teacher，teacher 额外看到三种"特权信息视图"（完整解 / 部分推理 / 仅答案）；AVSD 不固定用某一种视图，而是把多视图 teacher 信号拆成"跨视图共识"（可靠方向）和"单视图残差"（有用但有风险），用一个门控只在残差与共识方向一致且幅度相称时才加进去——从而比任何单视图自蒸馏都更稳更好。

**元信息**：arXiv:2605.20643v1（2026-05-20，cs.LG，Preprint）｜ UNC Chapel Hill / Capital One / UT Austin｜ 2026-05 预印本｜ 主题 T1（on-policy 自蒸馏）/ 相关性 Med（与 TSRD 的"特权信息 teacher / scaffold"高度相关）｜ 代码 https://github.com/duykhuongnguyen/AVSD （已 clone 到 resource/repos/avsd，约 1.9M；真实可用，但生成的 JSONL 训练数据被 Git 忽略需自构）｜ 框架 DeepSpeed ZeRO-2 + HF Accelerate + PEFT(LoRA) + vLLM，底层基于 OPSD（siyan-zhao/OPSD）→ TRL。

---

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/avsd/fig_01.png)

*Figure 1: Self-distillation leverages access to privileged ground truth information as a teacher to distill into the student. However, performance across datasets on Qwen3-4B varies substantially depending on the type of privileged ground truth information the teacher has access to (full solution, p*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/avsd/fig_02.png)

*Figure 2: (A) Bar graphs show teacher advantages on student tokens. (B) The geometric target demands consensus between teachers (conservative), while the arithmetic target allows individual teachers to contribute more strongly (permissive). The residual is their difference. (C) A twocomponent gate c*

## 1. 相关工作与进展

- **RLVR**（GRPO 等）是当前可验证任务（数学/代码）后训练主流，但监督**稀疏、只在 outcome 层**：失败时几乎没学习信号，采样昂贵。
- **蒸馏**提供 dense 的 token-level 指导，但标准**离线蒸馏**有 train-test mismatch（学生在自身分布外的轨迹上训练）。
- **on-policy 蒸馏 (OPD)**：在学生自采样轨迹上用 teacher 概率做局部监督，缓解 off-policy 问题，但仍**依赖一个更强的外部 teacher**。
- **自蒸馏 (self-distillation)**：去掉外部 teacher，让同一模型当 teacher，只是 teacher 额外条件化于学生推理时看不到的**特权信息 (privileged information)**——解、示范、反馈、最终答案等。本文即在自蒸馏这一支上做改进。

## 2. 现有工作存在的问题
作者明确指出单视图自蒸馏的两个核心缺陷（§1，引 Penaloza 2026 / Yang 2026 / Kim 2026）：

1. **不对称性 (asymmetry)**：teacher 依赖某种特权信息，而学生推理时无法访问。Yang et al. 2026 证明信息不对称下的分布匹配存在**不可消除的互信息 gap**——学生被迫去编码 test 时观测不到的 view-specific 关联。
2. **视图选择难**：最优特权信息类型**因任务而异**且无先验判据。给完整解可能把 teacher 锁进与学生不合的思维模式；只给答案更灵活但信息量低；Kim et al. 2026 还发现信息量最大的视图反而丢掉了帮助识别推理错误的信号、损害 OOD 泛化。而现有方法一旦选定视图就**全程固定**。

## 3. Motivation
能否**同时**利用多个特权视图，构造出比任何单视图 teacher 都更好的 token-level on-policy 学习信号？

## 4. 主要灵感 / 核心直觉
两条直觉（§1）：

- 若**不同视图诱导相似的 token 更新**（都想 promote 或都想 suppress），该信号大概率**稳定且任务相关**——因为各视图共享同一学生可见前缀，只是特权信息不同，跨视图一致的更新不太可能依赖某个 view-specific artifact。
- 若**某 token 仅被一个视图强烈偏好**，它可能捕捉互补信息（有用），但也可能是学生看不到的特权 artifact（危险）——需要"加，但要小心地加"。

## 5. 主要解决思路（一段话讲清核心）
把 M 个特权视图各自诱导的 teacher 分布拆成两部分：**共识 (consensus)** = 跨视图一致支持的部分，用**几何均值池化 (geometric consensus target)** 刻画（只有被所有视图共同支持的 token 才得高概率）；**残差 (residual)** = 额外的 view-specific 支持，用**算术均值池化 (arithmetic marginal target)** 减去共识得到（只要有一个视图强支持就保留）。AVSD **不直接蒸馏任一池化目标**，而是以共识为基准方向，用一个**门控 (gate)** 决定是否把残差加上去：仅当各视图在 promote/suppress 方向一致、且残差幅度与共识幅度相称时才加。这样既能吸收互补信息，又防止任一视图主导。

## 6. 方法详解（通俗、分步骤）
形式化建立在 **token-level reverse-KL advantage** 上（§2.1，完整推导见 Appendix B.1）：

- 学生在前缀 h_t=(x,y_<t) 的下一 token 分布 p_t(v)；teacher 在第 m 个视图下的分布 \(q_t^{(m)}(v) = \mathrm{sg}\!\left[P^{T}(v \mid h_t, r^{(m)})\right]\)（sg 为 stop-gradient）。
- **per-view 蒸馏 advantage**：\(\Delta_t^{(m)}(v) = \log q_t^{(m)}(v) - \log p_t(v)\)。>0 表示该视图想 promote 此 token，<0 想 suppress。reverse-KL 的负梯度写成 policy-gradient 形式即 \(\mathbb{E}_{v \sim p}\!\left[A_t(v) \nabla \log p_t(v)\right]\)。

分步：

1. **构造 M 个视图**：\(r^{(m)} = T_m(r)\)，保留任务相关信息、改变暴露给 teacher 的特权形式。本文数学/代码均用 **M=3** 个视图。
2. **几何共识目标**：对各视图概率取几何均值再归一化 → 强调"被所有视图共同支持"的 token（交集支持）。
3. **算术边际目标**：取算术均值 → 保留"至少被一个视图强支持"的 token（并集支持）。
4. **残差** = 算术边际 − 几何共识。
5. **门控加残差**（正文 Eq.2）：从共识出发，仅当 (a) 各视图方向一致 (b) 残差幅度 ∝ 共识幅度时，按比例 λ=C·R 加入残差。代码实现于 `src/avsd/common/multiview_distill.py::build_avsd_target`，`avsd` 模式即 \(\lambda = \frac{|\text{consensus\_adv}|}{\text{delta\_abs\_mean}} \times \frac{|A^{G}|}{|A^{G}| + J}\)，与正文 Eq.2 一致，且断言要求**各视图权重均匀**。
6. **消融**显示：consensus-only（只蒸馏几何共识）与 arithmetic-only（只蒸馏算术边际）变体均逊于完整 AVSD —— 证明"共识做骨架 + 门控加残差"两件事都不可或缺。

## 7. 实验数据集

- **数学**：训练用 **OpenThoughts**（正文为 "OpenThoughts" 而非旧分析里的 "OpenThought"，已核）数学子集；每题自动构造三视图（full solution / partial solution / final answer，partial_solution_ratio=0.5）；评测 **AIME24 / AIME25 / HMMT25**，指标 **Avg@8**。模型 Qwen3-8B、Qwen3-4B、DeepSeek-R1-Distill-Qwen-7B。
- **代码**：从 Codeforces（open-r1/codeforces）Python 子集采 5K 训练 + 留出 100 题做 in-domain 测试；评测 Codeforces、**LiveCodeBench v6**；视图为 reference / hint / feedback。模型 Qwen3-8B。
- **基线**：单视图自蒸馏（Zhao 2026 / Shenfeld 2026）+ GRPO。

## 8. 实验结果与主要发现

- Qwen3-4B（数学）：较 base **+5.5%**、较最强基线 **+2.2%** Avg@8。
- Qwen3-8B（数学）：取得最佳均分，较最强基线 **+3.1%**。
- DeepSeek-R1-Distill-Qwen-7B（数学）：较自蒸馏基线 **+2.3%**。
- Qwen3-8B（代码）：较单视图自蒸馏基线 **+2.4%** 均值。
- **关键发现**：Fig.1 显示"最佳单视图随数据集变化、full solution 并非总最优"——这正是 AVSD 自适应组合多视图的立足点；token-level 分析进一步显示 AVSD 同时保留了跨视图一致性并选择性加入门控残差。

## 9. 结果如何支撑其主张
主张是"多视图自适应组合 > 任一单视图"。证据链较完整：(a) 跨三个模型、两个领域均一致超过最强单视图基线与 GRPO；(b) 消融排除了"只用共识"或"只用边际"的简化解释；(c) Fig.1 的"无单一最优视图"现象直接 motivate 了方法。增益幅度（2–3% Avg@8）属中等而非压倒性，且基准集中在竞赛数学，需谨慎看待泛化。

## 10. 逻辑自洽性（中性评估）

- 自洽：从"不对称 + 视图选择难"两个问题，到"共识/残差分解 + 门控"的方法，再到消融，论证链条闭合；reverse-KL advantage 的形式化与代码实现（`build_avsd_target`）一致。
- 张力点：方法声称缓解"互信息 gap"（不对称问题），但本质仍是各视图 teacher 都条件化于特权信息，门控只是抑制"仅单视图支持"的危险 token，**并未从原理上消除**学生编码不可见关联的压力——更像是经验性缓解而非理论保证。

## 11. 残留问题 / 局限

- **增益中等**（2–3%），且评测高度集中于竞赛数学 + 少量代码，未见更广泛任务/通用指令的验证。
- **视图需人工/自动构造**（数学三视图靠规则切分，partial_ratio=0.5 是超参），视图质量与数量（M=3）对结果的敏感性未充分扫描。
- 门控的两个条件（方向一致 + 幅度相称）是**手工设计的启发式**，缺乏最优性论证。
- 不涉及 MTP；与 TSRD 的关联在于"特权信息 teacher / scaffold"的思路，而非具体 foresight 机制。

## 12. 开源代码与框架（链接+框架+代码可得性）

- 代码：https://github.com/duykhuongnguyen/AVSD （resource/repos/avsd，约 1.9M）。
- **框架栈**：DeepSpeed（ZeRO-2 + CPU optimizer offload）+ HuggingFace Accelerate + PEFT(LoRA) + vLLM；trainer/自蒸馏基础设施沿用 **OPSD**（siyan-zhao/OPSD），OPSD 又基于 **TRL**。environment.yml 关键依赖：torch 2.8.0、transformers 4.57.1、**trl 0.26.0**、deepspeed 0.18.2、accelerate 1.11.0、peft 0.17.1、vllm 0.11.0、flash-attn 2.8.3。
- **训练**：`accelerate launch` + `configs/accelerate.yaml`（bf16、ZeRO-2、CPU offload、默认 4 进程）；vLLM colocated rollout（默认 `--vllm_gpu_memory_utilization 0.4`、TP=1，周期性从 trainer 同步权重）。
- **复现入口**：数学 `scripts/math/train_avsd_qwen3_8b.sh` / `train_avsd_deepseek_r1_distill_qwen_7b.sh`（默认 OpenThoughts 数据，写到 `outputs/avsd/`），评测 `python -m avsd.math.evaluate --dataset {aime24|aime25|hmmt25}`；代码 `download_codeforces_cots_py.sh` → `python -m avsd.code.prepare_code_views` → `scripts/code/train_avsd_code_*.sh` → `avsd.code.evaluate_codeforces`（num-samples 8）。
- **已核超参**（train_avsd_qwen3_8b.sh / train_avsd_code_qwen3_8b.sh）：lr 5e-6、max_grad_norm 0.1、max_steps 500、per_device_bs 4 × grad_accum 2、max_completion_length 4096、max_length 16384、temperature 0.7 / top_p 0.95 / top_k 20、LoRA r64/α128、bf16+flash_attn2、`--use_tinker_loss`、`--multi_view_mode avsd`、`--avsd_gate_mode avsd`。代码结构 `src/avsd/{common,code,math}/`。
- **代码可得性**：方法核心（门控、多视图蒸馏）真实可跑，但**生成的 JSONL 训练数据被 .gitignore**，需自行下载/构造。

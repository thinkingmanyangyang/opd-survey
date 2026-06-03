# orz — Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model

> **一句话重点 (TL;DR)**：用最朴素的 vanilla PPO + GAE(λ=γ=1) + 二值规则奖励、完全不带 KL，就能在 base 模型上稳定地大规模 scale up reasoning RL；首个把代码/数据/各尺寸权重乃至 critic 权重全部开源的 "Reasoner-Zero" 实现。

**元信息**：arXiv 2503.24290（v2 2025-07-05）｜ StepFun + 清华（Jingcheng Hu 等，沈向洋）｜ 2025-03 预印本 ｜ 主题 T3（RLVR/zero-RL 系统），相关性 High ｜ 代码 github.com/Open-Reasoner-Zero/Open-Reasoner-Zero（全开源，本地已克隆约 92MB）｜ 框架 OpenRLHF + vLLM + DeepSpeed + Ray。

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/orz/fig_01.png)

*Figure 1: Evaluation of Open-Reasoner-Zero-{7B, 32B} on benchmarks (averaged on 16 responses) during training. Using the same base model, Qwen2.5-32B base, as DeepSeek-R1-Zero-Qwen32B, Open-Reasoner-Zero-32B achieves superior performance on AIME2024, MATH500, and GPQA Diamond-requiring only a tenth*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/orz/fig_02.png)

*Figure 2: Train-time Scale up on Train Reward and Response Length of Open-Reasoner-Zero (ORZ) - {0.5B, 1.5B, 7B, 32B}. Train Reward and Response Length increase steadily, demonstrating consistent scalability across model sizes. Interestingly, the ORZ-32B Response Length exhibits fluctuations without*

## 1. 相关工作与进展
o1、DeepSeek-R1-Zero 展示了大规模 RL 的 "训练时间 scaling"：随算力增大，benchmark 性能与响应长度同步持续增长、无饱和迹象，并伴随 "Aha moment"。R1-Zero 还证明可直接在 base 模型上启动 RL（跳过 SFT/蒸馏）。但 DeepSeek 仅简述其训练 pipeline，关键细节（数据、超参、value/advantage 估计、稳定化技巧）不公开。

## 2. 现有工作存在的问题
社区缺乏一个直接在 base 模型上做大规模 reasoning RL 的、全开源且稳定可扩展的实现；对 value/advantage 估计与训练不稳定（尤其 GRPO 因无 value network 易在重复模式上误判、坍缩）缺乏系统分析与可复现配方。

## 3. Motivation
提供首个鲁棒、可扩展、易跟随的大规模 reasoning-oriented RL on base model 实现，并 democratize 相关技术——完整释放代码、数据、各尺寸权重乃至 critic 权重，降低社区复现门槛。

## 4. 主要灵感 / 核心直觉
"少即是多"：大规模数据天然降低方差，使无偏配置（GAE λ=γ=1）可行；学习到的 critic 比无 value 的 GRPO 能更准地做 token 级 credit assignment、识别并 devalue 重复等劣化模式；去掉 KL 与 reference model 反而鼓励探索、省显存省调参。

## 5. 主要解决思路(一段话讲清核心)
直接在 Qwen2.5 base 上用 R1-Zero 风格 prompt 启动 RL，采用极简(minimalist)配方：vanilla PPO + GAE(λ=1,γ=1) + 仅检查 `<answer>` 与参考答案精确匹配的二值奖励，完全不加任何 KL 正则，并配合大规模、多样化数据，即可稳定 scale up 性能与响应长度。

## 6. 方法详解(通俗、分步骤)
- **选 PPO 而非 GRPO**：学习到的 critic 给出更准的 token 级 value 与 credit assignment；分析显示 PPO 对重复 token 赋更负的 advantage，能抑制坍缩。
- **GAE λ=1, γ=1**：无偏配置充分捕捉长程依赖；优势简化为 Â = R − V_φ(s_t)，value 目标 (V_φ(s_t) − R)²。
- **去掉 KL**：免去 reference model 的显存/计算与调参，鼓励探索。
- **极简 reward**：二值（1/0）精确匹配，无 format reward，reward hacking 空间最小；base 模型也能很快学会正确格式。
- **scale up data**：数据规模与多样性对持续提升至关重要。
- **采样/训练细节**：每步 128 prompt × 每 prompt 64 response，temperature/top-p=1.0；严格 on-policy；batch-level advantage 归一化。32B 末段加 100 步 annealing（13k 难题）。〔已核 repo playground/orz_32b_ppo.py：gamma=lambd=1.0、init_kl_coef=0、kl_loss_coef=0.0（use_kl_loss=True 但系数为 0，即 KL 实际关闭）、n_samples_per_prompt=64〕

## 7. 实验数据集
- 训练：ORZ 精选数据（正文表述 "tens of thousands of curated QA pairs"；开源含 orz_math_57k / orz_math_72k_extended / orz_math_13k_hard；来源 AIME(≤2023)、MATH、Numina-Math、Tulu3 MATH、OpenR1-Math-220k、AoPS 论坛 + 程序合成的逻辑/多步/反事实题）；排除证明题等难评测题，并用 LLM 过滤极端 pass rate。对照实验用 ORZ-57k vs MATH-train-7.5k。
- 评测：AIME2024、AIME2025、MATH500、GPQA Diamond（均 avg@16）；泛化 MMLU、MMLU_PRO。

## 8. 实验结果与主要发现
- 基座 Qwen2.5-{0.5,1.5,7,32}B base，直接大规模 RL、跳过 SFT。
- 主结果：ORZ-32B 在 AIME2024(48.1)、MATH500(92.2)、GPQA Dia.(55.5) 上超越或持平 DeepSeek-R1-Zero-Qwen-32B(47.0/91.6/55.0)，且**仅用约 1/10 训练步数**；MMLU/MMLU_PRO 超 Qwen2.5-Instruct-32B。〔已核 PDF Table，行 318-321/968-971〕
- 消融：GAE λ=1.0 优于 0.95；去 KL 优于 KL Loss/KL Penalty；ORZ-57k 优于 MATH-7.5k（后者早早 plateau）。
- 扩展：对蒸馏模型续做 ORZ（两阶段，类 R1）得 ORZ-R1-Distill-Qwen-14B，超过更大的 R1-Distill-Qwen-32B。

## 9. 结果如何支撑其主张
主张"极简配方即可稳定 scale up"由两类证据支撑：(a) 训练曲线显示性能与响应长度随步数同步增长无饱和；(b) 跨尺寸（0.5→32B）一致受益，且 32B 以 ~1/10 步数追平 R1-Zero。逐项消融（GAE/KL/数据规模）直接对应配方中每个设计选择，因果链较完整。

## 10. 逻辑自洽性(中性评估)
配方主张与消融基本一一对应，可信度较高。需注意：PPO 优于 GRPO 的论证主要依赖作者自身曲线与 "advantage on repeated token" 的定性分析，并非对所有任务/尺度的普适结论；"1/10 步数" 的对比依赖与 DeepSeek 复现条件的可比性（数据、prompt 不同），属同向但非严格控制比较。

## 11. 残留问题 / 局限
- 与 R1-Zero 的步数对比跨实现/跨数据，严格可比性有限。
- 仅评测数学+少量通用基准；对代码、agentic 等域的可迁移性未充分验证。
- "去 KL 总更好" 的结论在 base 模型起点成立，迁移到已对齐/已蒸馏起点时不一定（参见 ProRL 反向主张保留 KL）。
- 二值奖励对证明题、开放式任务不适用。

## 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库：https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero（本地已克隆约 92MB，含 orz/ 包、playground/ 各尺寸训练脚本、docker/）。
- 模型/数据：HuggingFace Open-Reasoner-Zero（ORZ-{0.5,1.5,7,32}B、ORZ-R1-Distill-Qwen-14B、critic 权重、ORZ 数据）。
- 框架：OpenRLHF（+ vLLM + DeepSpeed + Ray），实现 vanilla PPO + GAE 的大规模分布式训练。代码可得性：完整（含 critic 权重，开放程度在同类工作中最高）。

# rstar2 — rStar2-Agent: Agentic Reasoning Technical Report

> **一句话重点 (TL;DR)**：用"高吞吐代码执行环境 + 抗噪的 GRPO-RoC（Resample-on-Correct）+ 短长度多阶段 RL recipe"，在 64×MI300X、510 步 / 一周内把 Qwen3-14B-Base 推到前沿数学推理，AIME24=80.6/AIME25=69.8/HMMT25=52.7，并在 AIME24/HMMT25 上超过 671B 的 DeepSeek-R1（AIME25 基本持平）。

**元信息**：arXiv 2508.20722（v1, 2025-08-28, cs.CL）｜ Microsoft Research（Ning Shang、Yifei Liu、Yi Zhu、Li Lyna Zhang 等共同一作；Li Lyna Zhang、Mao Yang 为 project leaders / 通讯）｜ 技术报告, 2025-08 ｜ 主题 Agentic RL / 工具调用推理 / T2,T4，Relevance=High（GRPO-RoC 的"正样本筛优 + 负样本保多样"与本项目 path-selection / 高质量正样本+保留失败负样本相关）｜ 代码 https://github.com/microsoft/rStar（已克隆 ~576K，主仓为骨架 + submodule 引用，verl/code-judge 为 git submodule，浅克隆未拉取）｜ 框架 veRL v0.5 + Code Judge + vLLM

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/rstar2/fig_01.png)

*Figure 1: rStar2-Agent-14B reaches frontier-level math reasoning, comparing AIME24 accuracy versus RL training steps against DeepSeek-R1-Zero (671B).*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/rstar2/fig_04.png)

*Figure 4: Proportion of tool calls that contain errors within correctly answered trajectories. Under naive GRPO, the error rate initially decreases but soon plateaus at a significant level. In contrast, our GRPO-RoC continues to reduce tool-related errors with more training steps.*

## 1. 相关工作与进展
让模型"想得更聪明而非更长"需培养自主用工具来推理、验证、从工具反馈学习的能力。本工作以 Python 编码工具 + 解释器作 agentic RL 环境，拓宽动作空间、支持探索替代解与验证中间步骤，补足单纯 long CoT 的内部自反思不足。对比主流把 rollout 长度堆到 16K→48K 的做法，rStar2 走短长度路线。多分支仓：rStar2-Agent 在 main，prior work 在 `rStar-mutualreasoning`、`rStar-math` 分支。

## 2. 现有工作存在的问题
扩展 agentic RL 两大挑战：
- **环境噪声**：编码工具/解释器复杂，模型生成语法/逻辑错误代码时，环境反馈会让其浪费 token 纠错而非推进推理；现有 outcome-only 奖励的 RL 即便中间工具调用失败、只要最终答案对仍给正奖励，导致模型把错误当可接受、产出冗长低质轨迹（论文给出工具错误率随训练步数上升的观察，Qwen2.5-32B≈15%、Qwen3-14B≈10%）。
- **基础设施压力**：单 batch 可触发数万并发工具调用，难构建可靠高吞吐执行环境；agentic rollout 放大标准 RL 系统的 rollout 低效，严重拖慢训练。

## 3. Motivation
通过三项创新使 agentic RL 在规模上有效，从而在有限 GPU（64 张 MI300X）上把 14B 基座在 510 步 / 一周内推到前沿数学推理水平、并超越 671B DeepSeek-R1。

## 4. 主要灵感 / 核心直觉
- **非对称采样**：与其在 reward 里显式惩罚工具错误（易 reward-hacking、不稳），不如在采样层面对正/负轨迹差异化处理——正样本只留最干净的高质量成功轨迹作正监督，负样本均匀 downsample 以保留多样失败模式作负信号。
- **短而聪明**：不堆长度（避免 16K→48K 的高成本与冗长低质），用 8K→12K 短长度逼模型高效推理。
- **SFT 只打底不增推理**：先做 non-reasoning SFT 仅注入指令遵循/工具使用/格式，避免 SFT 过拟合、保持初始响应短。

## 5. 主要解决思路(一段话讲清核心)
三件套：(i) 可靠高吞吐 Python 代码执行环境（Code Judge + Redis + 隔离 worker）缓解高 rollout 成本；(ii) **GRPO-RoC**——在 GRPO 上加 Resample-on-Correct：先 oversample 较大 rollout 组再 downsample 到标准 batch，正轨迹按"工具错误/格式问题最少"筛优、负轨迹均匀下采样，用非对称采样抗稀疏 outcome-only 奖励下的环境噪声；(iii) 低成本训练 recipe——先 non-reasoning SFT，再用 GRPO-RoC 做短长度多阶段 RL（8K→12K→12K，共 3 阶段 510 步）。

## 6. 方法详解(通俗、分步骤)
- **(i) 高效 RL 基础设施**：Code Judge 作工具调用服务器执行模型生成的 Python（Redis + uvicorn + workers，多节点可扩展），应对单 batch 数万并发调用。
- **(ii) GRPO-RoC**：RoC 先 oversample 大组 rollout，再 downsample 到标准 batch；**正轨迹**保留工具错误/格式问题最少的最高质量样本，**负轨迹**均匀 downsample。这种非对称采样保留多样失败模式作负信号、强调高质量成功样本作正监督；相比在 reward 里惩罚工具错误更稳、避免 reward-hacking（从更干净的正样本学习）。
- **(iii) 训练 recipe**：① non-reasoning SFT——用 165K function-call 数据（117K ToolACE-11K + APIGen-MT-5K + Glaive-function-calling-v2-101k 等）仅注入工具格式与指令遵循，不增强推理；② 多阶段 GRPO-RoC：Stage-1 在 8K 长度做简洁训练（响应从约 1K 增长），Stage-2/3 提到 12K 并逐步加难，3 阶段共 510 步。

## 7. 实验数据集
- 基座 **Qwen3-14B-Base**。RL 训练数据（§4.1）：仅保留整数答案题以保 verifier 可靠，从三源收 >100K 候选——**17K** 来自 DAPO 训练集（整数答案子集）、**93K** 来自 AoPS 论坛（经 OpenMathReasoning）、**937** 来自 Project Euler——清洗后得 **42K** 高质量题-答对为最终 RL 训练集（开源 data_preprocess 以 DAPO-17k 为示例入口，论文实际用 42K 复合集）。
- 数学评测：AIME 2024 / 2025、MATH500、HMMT25。泛化评测：GPQA-Diamond（科学）、BFCL v3（agentic 工具）、IFEval、Arena-Hard。

## 8. 实验结果与主要发现
- 数学（pass@1）：rStar2-Agent-14B **AIME24=80.6 / AIME25=69.8 / HMMT25=52.7**。对照 DeepSeek-R1(671B) 79.8/70.0/44.4、o3-mini(medium) 79.6/77.0/53.0、DeepSeek-R1-Zero(671B) 71.0/53.3/46.0、Claude-Opus-4.0(Think) 76.0/69.2/-、QWQ-32B 79.5/65.8/47.5。即 14B 在 AIME24 与 HMMT25 上超 R1-671B，AIME25 基本持平（略低 0.2）。
- 效率：510 步 / 一周 / 64×MI300X 达前沿，响应显著更短。
- 泛化：GPQA-Diamond 超 DeepSeek-V3；BFCL v3、IFEval、Arena-Hard 均有竞争力。
- 消融/观察：GRPO-RoC 提升训练稳定性、避免 reward-hacking；从近零起步即被显著拉升。

## 9. 结果如何支撑其主张
"小模型超大模型 + 高效"由对照表（14B vs 671B/o3-mini）与 510 步/一周直接支撑。"抗噪有效"由 GRPO-RoC 提升稳定性、工具错误率观察与从近零拉升支撑。"短长度足够"由 8K→12K 设置 + 前沿成绩支撑（隐含对照主流 16K→48K）。泛化主张由跨任务结果支撑。需注意 AIME25 上并未严格超过 R1（持平/略低），TL;DR 的"超越"以 AIME24/HMMT25 为主。

## 10. 逻辑自洽性(中性评估)
三创新对应两挑战（基础设施↔环境压力、GRPO-RoC↔环境噪声、recipe↔成本），映射清晰。GRPO-RoC 的"正筛优/负保多样"与"避免 reward-hacking"动机一致。SFT 设为 non-reasoning 与"保持初始响应短/不过拟合"自洽。整体为工程+算法的技术报告，论证连贯。

## 11. 残留问题 / 局限
- 开源迁移版（VERL v0.2→v0.5）尚未完整训出模型（作者注前 50 步差异极小，但未给最终复现成绩）。
- 〔待核〕各阶段精确数据配比、步数划分、学习率等见 §4.3 Multi-Stage RL Training，本次未深读。
- 仅整数答案题以保 verifier，限制了任务多样性；AIME25 未严格超 R1 说明优势未必跨所有 benchmark 一致。
- 单一基座（Qwen3-14B-Base）、单一硬件栈（MI300X）；可靠性高度依赖 Code Judge 工程实现，迁移到其他环境的稳定性未验证。

## 12. 开源代码与框架(链接+框架+代码可得性)
- 链接：https://github.com/microsoft/rStar（main 分支为 rStar2-Agent；浅克隆未拉 submodule）。
- 框架：**veRL v0.5（volcengine/verl）+ Code Judge（0xWJ/code-judge）**；原训练基于 VERL v0.2 + 自研多轮工具框架，开源版迁移到 v0.5。Code Judge：Redis + uvicorn server（:8088, MAX_EXECUTION_TIME=4）+ workers。rollout/推理用 vLLM（`--enable-auto-tool-choice --tool-call-parser hermes`）。安装需 `torch<2.8`、`verl/requirements_sglang.txt`。
- 关键脚本：`data_preprocess/{aime2024,dapo}_rstar2_agent_loop.py`；训练 `examples/run_qwen3-14b_rstar2_agent_weave.sh`（8×A100/H100）；RoC 配置 `augmentation.do_down_sampling=True`、`down_sample_to_n=16`、`reject_equal_reward=True`、`roc_error_ratio=True`（按工具错误率重采正确轨迹）、`roc_answer_format=True`、`min_zero/non_zero_reward_trace_num=2`（保留最少正/负轨迹）；评测 `examples/{aime_eval,math500_eval}.sh`。
- 可得性：代码骨架 + 配置齐备，但 verl/code-judge 为 submodule（需 `git submodule init/update`），且迁移版未训出完整模型，端到端复现有门槛。

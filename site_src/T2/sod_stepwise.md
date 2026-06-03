# sod_stepwise — SOD: Step-wise On-policy Distillation for Small Language Model Agents

> **一句话重点 (TL;DR)**：把 OPD 用于小模型 agent 的工具集成推理（TIR）会因工具错误触发的"加速分布漂移"而训练崩溃；SOD 按 step 级师生发散自适应重加权蒸馏强度（高发散区衰减、重对齐时回升），在对齐区保留 dense 监督，使 0.6B/1.7B 学生相对最强基线 OPD 平均 +20.86%/+18.50%。

**元信息**：arXiv 2605.07725v1（2026-05-08 Preprint） ｜ 浙大 + 腾讯 LLM 部 + 中科大 + 新加坡国立（Qiyong Zhong、Mao Zheng、Mingyang Song 共同一作；Junfeng Fang、Houcheng Jiang 通讯） ｜ 主题 T1/T2 High（OPD + 小模型 agent TIR + step 级自适应重加权，与本项目 path-recovery / MTP foresight 探针直接呼应） ｜ 代码 https://github.com/YoungZ365/SOD （已克隆约 23MB，基于 verl，含 recipe/examples/docker） ｜ 框架 veRL fork + Open-AgentRL + ReTool agentic 组件

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/sod_stepwise/fig_01.png)

*Figure 1: The motivation of SOD . (a) Student-teacher divergence d k across reasoning steps, sampled from 800 trajectories: in TIR, erroneous tool calls cause divergence to accelerate sharply, unlike the gradual drift in text-only reasoning. (b) Teacher entropy statistics over 800 sampled trajectori*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/sod_stepwise/fig_02.png)

*Figure 2: The overview of SOD . (a) The student generates multi-step trajectories where erroneous tool calls propagate across steps, degrading teacher supervision reliability. (b) Student-teacher distributions drift apart as errors accumulate. (c) Step-level divergence d k quantifies this drift. (d)*

## 1. 相关工作与进展
Agentic 能力多依赖大模型，推理成本高；将其迁移到可端侧部署的 SLM 有实践价值。TIR 后训练主流基于 RL（GRPO），只给稀疏 outcome 级 reward。OPD（on-policy distillation）提供 dense token 级监督，缓解 credit assignment、提升样本效率与稳定性，是把大模型 agentic 能力蒸到小模型的自然候选。训练数据/框架沿用 Yu et al.[52]（"Demystifying RL in agentic reasoning"，对应仓库 `recipe/demystify/`）与 ReTool。

## 2. 现有工作存在的问题
小模型容量有限、探索弱，稀疏 outcome 监督会加剧探索失败、陷入 cold-start。但直接把 OPD 用于 SLM-based TIR 会出现严重训练不稳定/崩溃。根因：TIR 经工具交互引入"非连续状态跳变"——单次错误工具调用注入错误观测，使后续推理在被污染状态上展开；teacher 工具使用准而小模型弱，早期累积多次错误工具调用，状态分布迅速偏离 teacher。在这些 OOD 状态上 teacher 的 token 级监督变得不可靠甚至误导（Fig.1：错误轨迹上 teacher 熵均值与方差在后段急剧增大），错误沿后续推理步级联放大，放大师生发散并诱发不稳定梯度。

## 3. Motivation
区别于文本推理中渐进式的分布漂移，TIR 的发散是由工具错误触发的"加速"漂移。需要按 step 级发散自适应调节蒸馏强度：在对齐良好的状态保留 dense 监督，在高发散状态衰减可能误导的 teacher 信号。

## 4. 主要灵感 / 核心直觉
用相邻 step 的发散比（而非绝对发散值）来调权：发散单调上升时累乘比值 <1、自动压低被污染信号权重；一旦重新对齐（d 下降）则允许蒸馏强度回升。仅依赖比值意味着任何与真实师生发散 ∆k 单调一致的可观测代理 dk 皆可用（Appendix D.5）。

## 5. 主要解决思路(一段话讲清核心)
定义 step-level divergence score d_k 作为 teacher 监督可靠性的可观测代理；据相邻 step 发散比累乘得到每步权重 w_k（上界 1+δ）；把 w_k 作用于该 step 内所有 token 的蒸馏损失，从而在高发散区衰减、在良好对齐区保留 dense 指导。整体目标为 GRPO + 加权 OPD 的联合损失。

## 6. 方法详解(通俗、分步骤)

- d_k：实现为该 step 内 token 的 mean(|log π_θ − log π_teacher|)，作为 teacher 监督可靠性代理。
- 权重（Eq.7）：w_1=1；step k≥2 权重为 1..k−1 各相邻发散比 (d_u+ε)/(d_{u+1}+ε) 的**累乘**，并以 1+δ 为上界（δ=0.2，ε=1e−6）。发散单调上升 → 累乘 <1 抑制被污染信号（Appendix D.4 证加权二阶矩被压制 O((d1/dk)²)）；d 下降（重对齐）→ 蒸馏强度回升。
- 施加粒度：w_k 作用于该 step 全部 token（非逐 token 均匀施加）。
- 联合目标：L = L_GRPO + L_OPD（OPD 系基线均用此口径）；开销可忽略（d_k/w_k 复用 OPD 前向已有的师生 logprob，仅 O(K) 标量运算）。

## 7. 实验数据集

- 训练数据（沿用 Yu et al.[52]）：3k 高质量多轮推理 SFT 语料（s1-1k 1k + LeetCode 1k + ReTool 1k，后两者经 ReasonFlux-PRM 打分各取 top-1k）+ ~30k RL 数据（DAPO-Math 17k + Skywork-OR1 Math 4902 / Code 3586 + MegaScience 3k）。SFT 轨迹由 Qwen3-Coder-30B-A3B 在 SandBoxFusion 环境内端到端交互生成。
- 评测（均报 average@32，百分比；temp=1.0、top_p=0.6、每题 32 采样）：Math=AIME 2024/2025、Science=GPQA-Diamond、Code=LiveCodeBench（v6 近期窗）。
- 模型：teacher 为 Qwen3-4B 经 GRPO 在 RL 数据上进一步优化（默认 4B teacher）；student 为 Qwen3-0.6B 与 Qwen3-1.7B。

## 8. 实验结果与主要发现

- 主结果：SOD 在两个 student 上对第二好基线（OPD）相对平均提升 0.6B +20.86%、1.7B +18.50%；四任务均最高分。0.6B student 在 AIME 2025 达 26.13%（average@32），据称为首个达到该水平的 sub-billion 模型。
- 基线含 SFT、GRPO 及多种蒸馏方法（共 6 个，Appendix B.3）；SFT/GRPO 单独均显著弱于蒸馏系。
- 开销（Table 4）：d_k/w_k 仅 O(K) 标量运算，显存差 <0.5GB；0.6B 上 SOD 反而比 OPD 快 3.5%（1052.3s vs 1090.5s，因自适应重加权抑制了错误学习、失败重试更少），1.7B +4.9% 开销（1105.4s vs 1053.6s）。
- 稳定性：错误轨迹后段 teacher 熵均值/方差急剧增大（Fig.1），SOD 重加权压制该区监督，缓解崩溃。

## 9. 结果如何支撑其主张
"OPD 在 SLM-TIR 上不稳"由 Fig.1 的发散/熵证据支撑；"step 级重加权有效"由消融（Table 2 移除 step-wise 退化）与主结果（对 OPD 的 +20.86%/+18.50%）支撑；"开销可忽略"由 Table 4 支撑；理论上加权方差被压制有 Appendix D.4 证明。证据链较完整，5 个随机种子重复增强了可靠性。

## 10. 逻辑自洽性(中性评估)
方法-动机-理论-实验四者自洽：动机（工具错误触发加速漂移）→ 代理 d_k（师生 logprob 差）→ 比值累乘权重（Eq.7）→ 方差压制证明（D.4）→ 代理单调一致性证明（D.5）→ 消融。一个口径需注意：主卖点是"对最强基线 OPD 的相对平均提升"（百分比口径），绝对点数提升（如 0.6B 整体由约 21→24+ 区间）规模较小，相对数放大了观感。

## 11. 残留问题 / 局限

- 模型/工具单一：仅 Qwen3 单一模型族、python 解释器（SandBoxFusion）单一工具环境验证；作者亦将 web/API 等其他 agent 设置与其他模型族列为局限。
- 相对增益口径：+20.86%/+18.50% 为对 OPD 的相对百分比而非绝对点数，需结合绝对值理解。
- d_k 代理依赖师生 logprob 差，teacher 自身在 OOD 状态的 logprob 可靠性边界未充分刻画（高熵区 logprob 噪声大可能反噬 d_k 估计）。

## 12. 开源代码与框架(链接+框架+代码可得性)

- 仓库 https://github.com/YoungZ365/SOD （已克隆约 23MB）。框架：veRL[76] + Open-AgentRL[52]（`recipe/demystify/`，复用其 sandbox_fusion 工具配置）+ ReTool（`recipe/retool/`）；rollout 走 vLLM（TP=4），SandBoxFusion 作 python 解释器、最多 16 轮工具调用。
- **〔重要更正——核心算法已开源〕** 与早前"step-wise 重加权核心损失未在仓库定位到"的判断相反：核心实现确在已发布代码中。`verl/trainer/ppo/ray_trainer.py:363 compute_stepwise_opd_weights` 完整实现 Eq.6/7——按 response_mask 提取 step 边界、`d_k = mean(|log π_θ − log π_teacher|)`、`w_1=1`、`w_k = min(∏_{u=1}^{k-1}(d_u+ε)/(d_{u+1}+ε), 1+δ)`、再 broadcast 到该 step 全部 token；由 `ray_trainer.py:878 _apply_token_kl_regularizer`（line 1622 处调用）将其乘到 OPD 优势项 `weighted_opd = opd_coef * stepwise_weights * raw_local_adv`。配置见 `verl/trainer/config/algorithm.py` 的 `TokenKLRegConfig`（stepwise_enable/epsilon/delta/opd_coef），运行脚本 `examples/SOD/run_sod.sh` 显式传入 `+algorithm.token_kl_reg.stepwise_*`。代码与论文 Eq.7 精确一致，可复现。
- **〔次要差异〕** 配置 dataclass 默认 `stepwise_delta=0.5`，但 `run_sod.sh` 覆写为 0.2（论文值），复现须用脚本而非 dataclass 默认。
- 训练硬件：单节点 8×H20（96GB）；0.6B/1.7B 1 epoch 约 2–3 天，4B/14B teacher 约 5–6 天；统一超参（AdamW lr=1e-6、batch 64、mini-batch 16、prompt≤2560、response≤20480、训练每 prompt 采 16、验证 32）；所有实验 5 个随机种子重复。

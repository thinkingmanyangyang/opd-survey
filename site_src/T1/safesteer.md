# safesteer — SafeSteer: Localized On-Policy Distillation for Efficient Safety Alignment

> **一句话重点 (TL;DR)**：安全特征在输出分布中本就稀疏，对齐应是"局部修改而非全局权衡"；SafeSteer 用 activation steering 造安全 teacher，挑出稀疏安全 token 子集 S，仅在 S 上施加 reverse-KL OPD，仅用 100 条有害样本即在几乎不掉通用能力下显著降 ASR。

**元信息**：arXiv 2606.02530（v1, 2026-06-01, cs.AI；投 EMNLP 2026）｜ 北航 / 北理工 / 北邮 / 北大 / 中科院自动化所 / 上海AI Lab / BAAI（Hao Li*、Jingkun An*、Zijun Song* 共同一作；通讯 Lei Sha）｜ arXiv preprint ｜ 主题 LLM 安全对齐 / 与本项目关系外围（安全方向），但方法层面高度相关——token-localized on-policy distillation + reverse-KL，与"在稀疏/局部 token 上做选择性蒸馏"（cf. TSRD path-token 监督、选择性 KD）同源 ｜ 代码 https://github.com/Anjingkun/SafeSteer（已克隆 ~8.9MB，含 distil_trainer.py / distil_config.py / main.py / model_utils / scripts / data；RepoExists=YES）｜ 框架 自研轻量安全对齐框架（activation steering teacher + token 选择 + localized OPD）

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/safesteer/fig_01.png)

*Figure 1: Safety-capability trade-off on Qwen2.5-7BInstruct . Each point is a method, with the gray point marking the base model. Our SafeSteer achieves the highest safety score while preserving general capability.*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/safesteer/fig_02.png)

*Figure 2: SafeSteer pipeline: (1) construct a safety teacher π t via activation steering, (2) select safety tokens S by contrastive log probability from π t responses, and (3) distill π t into π s with a token-level localized reverse KL on S .*

## 1. 相关工作与进展
安全对齐常以牺牲通用能力为代价（alignment tax）。现有方法多把它当双目标优化：混入大量通用数据、把安全梯度正交投影到通用能力零空间（NSPO）、或训练辅助奖励模型（SafeRLHF/MoCAN）。标准 LLM OPD 又通常需外部更强 teacher；近期自-teacher 方案重度依赖模型 in-context learning 能力。论文借鉴 Arditi et al. (2024) 的 refusal direction（activation steering）思路构造无需更强外部模型的安全 teacher。

## 2. 现有工作存在的问题

- 双目标 trade-off 依赖海量通用数据或辅助 reward model，成本高。
- 标准 OPD 需外部更强 teacher；自-teacher 方案依赖 ICL 能力。
- 即便有好 teacher，标准 OPD 对**整个词表**施加惩罚，会波及通用能力 token——而安全 token 在输出分布中稀疏、且与通用任务 token 大体不相交，全局惩罚反伤通用能力。

## 3. Motivation
论点："安全特征本就稀疏，对齐需要的是**局部修改而非全局权衡**"。因此用 activation steering 得稳定安全信号的 teacher，再把 reverse-KL 蒸馏惩罚**限制到稀疏安全 token 子集 S**，在调整安全特征同时缓解遗忘。

## 4. 主要灵感 / 核心直觉

- **稀疏性 → 局部更新**：安全行为集中在少数 token（拒答触发词等），无须全词表惩罚。
- **activation steering 作 teacher**：注入 refusal direction 即可让模型对任意输入稳定拒答，作为"廉价但稳定"的安全信号源，无需更强外部模型。
- **对比 + 投票挑 token**：不是取单点最大 logit 差，而是用对比 log 概率 + 跨位置/样本投票聚合，得到对 refusal direction 最敏感的稳健稀疏子集。

## 5. 主要解决思路(一段话讲清核心)
三步：(1) 提取 refusal direction d，在某层 ℓ 用 forward pre-hook 把残差流 h_ℓ 替换为 h_ℓ+d 并在所有 token 位持续注入，得稳定拒答的安全 teacher πt；(2) 在 harmless 指令上用 πt 采拒答轨迹，对每个位置算 token v 的对比 log 概率 Δ=log[pt(v)/p0(v)]（teacher vs base），用投票聚合挑出稀疏安全 token 子集 S；(3) student πs 在有害指令上 on-policy 自采样轨迹，仅对 S 内 token 最小化 D_KL(πs‖πt)（reverse-KL），从而只改安全特征、保留通用能力。

## 6. 方法详解(通俗、分步骤)

- **安全 teacher 构造**：按 Arditi et al. (2024) 提取 refusal direction d；在层 ℓ 用 forward pre-hook 令 h⋆_ℓ = h_ℓ + d，对所有 token 位持续注入，得 πt——它对**有害与无害输入都一致拒答**（即设计上会 over-refuse，但目的是提供稳定拒答信号）。
- **安全 token 选择**：在 harmless 指令（Alpaca）上用 πt 采 N 条拒答轨迹（长度 H）；每条每步算 token v 的对比 log 概率 Δ=log[pt(v|·)/p0(v|·)]；每位置取 top-K′，再用指示函数跨 harmless 数据、N 条轨迹、各 step 做**投票聚合** vote(v)=Σ 1[v∈C(x,n,j)]，得对 refusal direction 最敏感的稀疏子集 S（比单纯最大 logit 差更稳）。
- **localized OPD 训练**：student πs 在有害指令（PKU-SafeRLHF，仅 100 条）上 on-policy 生成轨迹，仅对 S 内 token 最小化 reverse-KL D_KL(πs‖πt)。

## 7. 实验数据集

- 安全 benchmark（7 个，报 ASR%↓）：有害查询 AdvBench、PKU-SafeRLHF、HarmBench、JailbreakBench(JBB)、SORRY-Bench；红队查询 HarmfulQA、ALERT。
- 通用能力（5 个，↑）：MMLU(STEM)、AlpacaEval(Win Rate)、GSM8K、MATH、HumanEval。
- 训练数据：harmless=Alpaca（选 token）；harmful=PKU-SafeRLHF，**仅 100 条**（<1% 基线数据量）。
- 模型：Llama-3-8B-Instruct、Llama-3.2-3B-Instruct、Qwen2.5-7B-Instruct、Qwen3-4B-Instruct-2507。
- 判官：默认 **Llama-Guard-4-12B**（另报 LLM-as-judge 校正 regex 误判）；通用能力用 lm-eval。基线：DPO-Mix、MoCAN、W-DOOR、BFPO、NSPO。学习率 Llama 系 1e-6、Qwen 系 1e-5。

## 8. 实验结果与主要发现

- Qwen3-4B-Instruct（t=0）：ASR 平均 **2.87%→0.91%**（基线 MoCAN 1.13%、NSPO 1.18%），通用 Avg **71.68→71.39**（几乎无损）。
- Qwen2.5-7B、Llama 系亦取得最低/次低 ASR 且通用能力近乎保持；安全–能力权衡上整体优于现有方法，且仅用 100 条有害样本、无通用数据。
- **关键发现（Appendix D.2）**：teacher πt 几乎拒绝所有 harmless 指令（over-refuse），但训练后的 student πs 与 base π0 **均不 over-refuse**——token-localization 成功蒸出安全特征而不吸收 teacher 的 over-refusal；PCA（Fig.3）显示 πs 相对 π0 在通用表征上几乎无 shift。

## 9. 结果如何支撑其主张
"局部更新而非全局权衡"由 ASR↓ 同时通用 Avg 几乎不变 + PCA 无表征 shift 共同支撑，逻辑直接。"稀疏 token 足够"由仅惩罚 S 即降 ASR 支撑。"不吸收 teacher 缺陷"由 Appendix D.2 的 student 不 over-refuse 实验支撑（这点直接回应了"steering teacher 过度保守"的质疑）。"高效"由 100 条样本 + 无通用数据支撑。

## 10. 逻辑自洽性(中性评估)
"安全稀疏 → 局部蒸馏"的核心论点—方法—结果三者对齐：token 选择（对比+投票）、localized reverse-KL、over-refusal 不传递的实验环环相扣。用 activation-steering teacher 这一"会 over-refuse 但信号稳定"的设计，与 token-localization 的"只取安全相关子集"互补，自洽性较强。

## 11. 残留问题 / 局限

- 仅做安全对齐这一窄域，"安全稀疏→局部更新"在更复杂能力对齐（如价值观、长程行为）上的泛化未验证。
- 安全 token 子集 S 离线固定（基于 Alpaca harmless + πt 拒答轨迹），对新型越狱 / 分布外有害模式的覆盖性存疑——S 的静态性是主要风险点。
- 依赖能可靠提取 refusal direction 的模型与层 ℓ 选择；refusal direction 方法本身（Arditi 2024）的局限会传导。
- 判官（Llama-Guard-4-12B）与基线超参对 ASR 结果敏感；SORRY-Bench 上残留 ASR 仍偏高（如 Qwen3-4B 5.91%），说明并非所有有害类别都被压住。
- 〔旧分析"teacher over-refusal 可能只被部分消解"已修正〕：论文 Appendix D.2 明确显示 student 不 over-refuse，over-refusal 未传递；但该结论基于其自定 over-refusal 测试集，覆盖面有限。

## 12. 开源代码与框架(链接+框架+代码可得性)

- 链接：https://github.com/Anjingkun/SafeSteer（已克隆 ~8.9MB；项目页 https://anjingkun.github.io/SafeSteer/）。RepoExists=YES。
- 框架：自研轻量安全对齐框架——`distil_trainer.py`（localized reverse-KL OPD 训练）、`distil_config.py`（配置）、`main.py`（入口）、`model_utils`（steering hook / refusal direction）、`scripts`、`data`。
- 流程：先离线选 token S（harmless 上用 πt 采轨迹 + 对比 log 概率 + 投票）→ 在线 OPD 仅惩罚 S（harmful 上 student on-policy 采样 + reverse-KL）。
- 可得性：训练/配置/入口与数据齐备，方法轻量（无需大规模通用数据、仅 100 条有害样本），复现门槛低。

# rosd — ROSD: Reflective On-Policy Self-Distillation for Language Model Reasoning across Domains

> **一句话重点 (TL;DR)**：标准在线策略自蒸馏（OPSD/SDPO）把自教师条件于"完整正确解"会让学生模仿训练域参考轨迹、损害 OOD；ROSD 改为"纠错思路 e + 错误引语 q"引导的**错误后缀局部蒸馏**——只修第一处出错起的后缀、保留有效前缀，从而在保住域内的同时大幅改善跨域泛化（4B 上 OOD 平均 41.31% vs SDPO 2.88%）。

**元信息**：arXiv 2605.28014（v1, 2026-05-27）｜ 香港理工 / 百度 / 山东大学 / 莱顿大学（Ziqi Zhao、Xinyu Ma、Liu Yang、Yujie Feng、Daiting Shi、Jingzhou He、Xin Xin、Zhaochun Ren、Xiao-Ming Wu）｜ arXiv preprint ｜ 主题 在线策略自蒸馏（OPSD）改进 / 与本项目 mtp_opd（MTP+OPD、TSRD）高度相关——"反思引导 + 错误定位"正对应"教路径选择/路径修复"｜ 代码 https://github.com/ZiqiZhao1/ROSD（Apache-2.0，已克隆 ~53MB，含完整 verl + 实验脚本 + reward 函数）｜ 框架 verl（扩展自 SDPO/lasgroup/SDPO）+ FSDP actor + vLLM rollout

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/rosd/fig_02.png)

*Figure 3: In domain training dynamics on Material and Chemistry. We report the mean@16 test score, and the shaded regions indicate the variance over 16 sampled responses.*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/rosd/fig_01.png)

*Figure 1: Left: Pilot study. We use SDPO (Hübotter et al., 2026) as a representative OPSD method. All methods are trained on the Material dataset with Qwen3-8B, and evaluated on Material and ToolUse as the in-domain and out-of-domain benchmarks, respectively. Right: Comparison of post-training metho*

## 1. 相关工作与进展
后训练对提升 LLM 推理至关重要。SFT 类依赖数据集提供的 golden solution；RL 类（RLVR）只靠 verifier 信号。典型 RLVR（GRPO）用 outcome reward 算 response 级 advantage，无 token 级监督。OPSD 用"自教师"（self-teacher，与学生同权重）在 token 级提供稠密监督弥补：给学生 on-policy rollout，让自教师条件于一个正确解，再对学生/教师分布算 token 级 KL（SDPO 是其 RL 实例，用成功 rollout / 环境反馈作教师上下文）。本工作的 GRPO 基线已用强化实现（含非对称 clip、无偏归一化、off-policy 校正）。

## 2. 现有工作存在的问题
标准 OPSD 即使域内也不稳健：域内性能训练早期升、后期不稳甚至下降；域外（OOD）快速恶化。两点归因：(1) 把自教师条件于"已验证正确解"会鼓励学生模仿训练域参考轨迹（而非针对自身错误修正），内化域特定模式；(2) 对整条 response 做蒸馏会覆盖已正确的推理前缀、惩罚有效的替代前缀，并把训练域偏好注入正确推理，损害 OOD 泛化（自教师还会引入"according to the reference solution"等参考依赖措辞）。

## 3. Motivation
错误通常是局部的——应只修正"第一处出错"起的后缀，并用"纠错思路"而非完整参考解引导教师，把"参考解模仿"转为"针对性推理纠错"，在修复错误同时保留有效前缀，提升域内并显著改善 OOD。

## 4. 主要灵感 / 核心直觉

- **错误局部性**：有效前缀（绿）+ 错误后缀（红）的二分；只需修后缀。
- **特权信息 vs 域偏好分离**：自教师的增益应来自"它能看到正确解这一特权信息"，而非把训练域风格灌进学生；故只把"纠错关键思路 e"作上下文、用"错误引语 q"做定位，避免全解模仿。
- 学生/自教师/自反思器共享同一 base，性能提升纯来自特权信息条件，无需更大外部教师、无额外模型存储。

## 5. 主要解决思路(一段话讲清核心)
对每题采 G 条 rollout，分正确集 Y+ / 错误集 Y-。对每条错误 rollout y-，配同组**最短**的正确 rollout y*，让自反思器输出 `<error_quote>`（q，精确标出第一处出错子串）与 `<explanation>`（e，解释错因与如何修）；把 e 作 auxiliary context 注入自教师 prompt，用 q 在 y- 中定位 k=Locate(q,y-)，仅对 t≥k 的后缀做 token 级蒸馏（匹配失败回退全响应 k=0；正确 rollout 仅反思"为何有效"得 e、全 token mt=1）。教师分布取 stopgrad。

## 6. 方法详解(通俗、分步骤)

- **错误聚焦自反思（Error-Focused Self-Reflection）**：错误 rollout 与同组最短正确 rollout 配对（最短以减少冗余）；反思 prompt（Fig.2）要求严格输出 `<error_quote>`（从错误解中抽的精确子串）+ `<explanation>`（错因+修法+正确逻辑）。正确 rollout 只单独反思得 e（"为何有效"），不直接全解模仿。自反思器复用自教师权重。
- **引语定位自蒸馏（Quote-Localized Self-Distillation）**：掩码 m_t = 0 (t<k) / 1 (t≥k)，k=Locate(q,y-)；目标对掩码后 token 算师生散度，π_teacher(·|x,e,y<t) 取 stopgrad。
- **散度实现**：继承 SDPO 的 token 级 KL（Eq.3）；代码中由 `alpha` 配置——alpha=1 为 reverse KL（SDPO 默认）、alpha=0 为 forward KL、中间为 **Generalized Jensen-Shannon Divergence**（`torch.lerp(kl_student, kl_teacher, alpha)`，core_algos.py:1163）。〔修正：旧分析称"采用 JSD"——准确说 JSD 是可配置的一般情形，reverse KL 是其特例/继承自 SDPO 的基线散度。〕
- **训练设定**：标准 RLVR（无人工 golden、仅 verifier）；学生/自教师/自反思器同 base，提升源自"教师条件于特权信息"。

## 7. 实验数据集

- 训练分别在 **5 个数据集**上单独进行：SciKnowEval(L3) 推理子集的化学/物理/生物/材料 4 个本科级科学 QA + ToolAlpaca 的工具调用数据集（ToolUse，把 API 规范+用户请求映射到正确 tool call）。
- 额外用 **AIME2024** 评数学推理。每个数据集划分训练/测试，"训一域、评所有域"以同时考察域内与 OOD。
- 训练每 prompt 采 **8 条 on-policy rollout**；评估每题采 **16 条报 mean@16**（取训练中最大 mean@16）。
- Backbone：**Qwen3-4B、Qwen3-8B**。基线：强化版 GRPO（非对称 clip + 无偏归一化 + off-policy 校正）、SDPO。

## 8. 实验结果与主要发现

- 域内（Table 2）：Qwen3-4B 平均 **72.83%**（较 GRPO/SDPO **+2.97/+5.81** 分）；Qwen3-8B **73.45%**（较 GRPO/SDPO +1.46/+0.95 分，8B 上增益较小）。
- OOD：ROSD 在所有训练域、两个规模上均稳定超过 SDPO；跨域差异大时（如 ToolUse→科学QA/数学）SDPO 严重退化——4B 上 SDPO OOD 平均跌到 **2.88%**，ROSD 仍维持 **41.31%**。
- 主要发现：把"全解模仿"换成"错误后缀局部纠错"是稳住域内 + 救回 OOD 的关键；自教师的增益来自特权信息而非更大模型。

## 9. 结果如何支撑其主张
"标准 OPSD 损害 OOD"由 pilot study（SDPO 域外恶化）支撑；"局部纠错改善泛化"由 ROSD 在所有 OOD 设置稳超 SDPO、尤其极端跨域（2.88%→41.31%）支撑，逻辑直接。域内 +2.97/+5.81 支撑"不牺牲域内"。8B 上增益缩小也被如实报告，未夸大。

## 10. 逻辑自洽性(中性评估)
"错误局部 → 只修后缀 + 用纠错思路而非全解"的因果链与两点归因（参考解模仿、覆盖有效前缀）一一对应。掩码机制（k=Locate(q)）直接实现该思路；匹配失败回退全响应是合理兜底。共享 base 的设计排除了"更大教师"混淆变量，自洽性强。

## 11. 残留问题 / 局限

- q 的定位依赖自反思器抽出"精确子串"并能在 y- 中匹配；匹配失败回退全响应，失败率/对结果的影响〔待核：论文未报回退比例〕。
- 8B 上相对 SDPO 增益已很小（+0.95），方法收益随基座变强而衰减，规模化外推存疑。
- "第一处出错"假设单点错误；多处分散错误或错误前缀本身无效时，"保留前缀"可能保留了错误前提。
- 评测域偏窄（科学 QA + 单一工具集 + AIME），缺代码/对话等异构域；自反思器质量本身依赖同一 base 的能力，弱基座下反思可能不可靠。

## 12. 开源代码与框架(链接+框架+代码可得性)

- 链接：https://github.com/ZiqiZhao1/ROSD（Apache-2.0，已克隆 ~53MB）。
- 框架：基于 **verl**（verl-project/verl）扩展自 **SDPO**（lasgroup/SDPO）；FSDP actor 训练 + vLLM rollout，8×NVIDIA A800 80G。
- 关键文件：`verl/trainer/ppo/ray_trainer.py`（rollout/reward/reflection/distillation batch 构建）、`verl/workers/actor/dp_actor.py`（自蒸馏 loss 分发）、`verl/trainer/ppo/core_algos.py`（`compute_self_distillation_loss`，alpha 配置 KL/JSD，line 1085+）、`verl/trainer/config/sdpo.yaml`（Hydra 配置）。主入口 `experiments/local/run_sdpo_processreflection_ood_all.sh`。
- 可得性：完整 verl 源码 + 启动脚本 + reward 函数齐备，自蒸馏 loss 与反思/定位逻辑可在代码中核实，可复现性好。

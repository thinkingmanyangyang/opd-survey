dr_grpo | Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO) | Sea AI Lab · NUS · SMU(Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi 等) | 2025-10-06 · arXiv 2503.20783 v2 · COLM 2025(+ICML 2025 AI4Math Workshop Best Paper 荣誉提名) | 主题线 L3(RLVR/GRPO·训练机理+算法)，兼 L6(token 信用) · 相关性 高

**原始论文**：https://arxiv.org/abs/2503.20783

## 一眼看懂
- 🟦 TL;DR：批判性审视 R1-Zero 范式的两大成分。**(成分一 base 模型)**：Qwen2.5 base **不用对话模板**时性能反而暴涨约 60%(疑似预训练就喂了拼接的 question-answer 文本→"已类 SFT")；"Aha moment(自反思)"在多个 base(含 DeepSeek-V3-Base)中**RL 之前就已存在**，并非纯 RL 涌现，且自反思行为与最终准确率**不正相关**。**(成分二 RL 算法)**：GRPO 目标里的 **1/|o_i|(响应级长度偏置)** 与 **std(R)(题目级难度偏置)** 会人为推高响应长度(尤其推长**错误**响应)；去掉这两项得到无偏的 **Dr. GRPO**，在不掉推理性能下大幅缩短错误响应、提升 token 效率，并给出 7B 极简 SOTA 配方(AIME24 43.3%，8×A100/27h)。【原文 §Abstract/§2/§3.1/Fig.1-2】
- 最巧的一步：**从 GRPO 目标里删掉 1/|o_i| 与 std(R) 两个归一化项**(Fig.1 左)。抽掉这个"删除"就回到 vanilla GRPO 的病态——Fig.4 证明 GRPO 的有效优势 a_{i,t} 等价于无偏优势 Ã_{i,t}=R−mean(R) 被 1/|o_i| 与 1/std(R) **重加权**后的版本；对负优势(错误响应)，长响应因 |o_i| 大而被罚得轻→策略偏好"把错误答案拖长"以摊薄惩罚。这是把"RL 长度增长是 feature"重新诊断为"部分是优化 bug"的关键一刀。【原文 §3.1/Fig.4】

## 为什么做
- 研究背景：DeepSeek-R1-Zero 证明可不经 SFT、直接对 base 大规模 RL 提升推理，伴随 RL scaling(响应长度持续增长)与"Aha moment"。社区多用 Qwen2.5 + GRPO 复现(SimpleRL-Zero、ORZ、PRIME 等)。【原文 §1】
- 解决的具体痛点：厘清 R1-Zero 里**哪些现象是真涌现、哪些是预训练遗留 / 优化假象**；并修掉 GRPO 的优化偏置以提升 token 效率；给一个极简配方。【原文 §1 takeaways】
- 相关工作 & 各自不足：GRPO(Shao 2024，组内相对优势 + 长度归一化的免 critic PPO 变体)——其长度归一化引入偏置；前人(Liu 2025b、Yeo 2025)已疑"开源 R1 复现里 Aha 是 base 自带"，但**没测真正训出 R1-Zero 的 DeepSeek-V3-Base**——本文补上这一块。多个开源 PPO 实现(trl/OpenRLHF/verl/SimpleRL-Zero/ORZ)也都有 length bias。【原文 §2.3/§3.1 末】
- 动机链：现状(R1-Zero 把"长度增长+Aha"当 RL 涌现的证据)→ 质疑(① base 是否本就会做题/会自反思? ② 长度增长是真推理需要还是优化 bug?)→ 诊断(base 去模板性能反升+V3-Base 有 Aha → 归因部分站不住；Fig.4 显示 GRPO 优势被长度/std 重加权)→ 所以(去掉两个归一化得无偏 Dr. GRPO；并据 base 分析定极简配方)。【原文 §2-3】
- 与最近邻工作的 Δ：相对**原版 GRPO**——Dr. GRPO 把 token 级 loss 的归一化从"除以本响应长度"改为"除以常数(生成预算)"、把优势从"减均值再除组内 std"改为"只减均值不除 std"。差在恢复了**无偏的 PPO 风格优势**(蒙特卡洛回报 + 无偏 baseline)。相对其它 R1-Zero 复现——本文不追新算法，而是**批判性归因 + 极简化**。【原文 §3.1-3.2/Fig.1】

## 怎么做 + 靠不靠谱
- 方法流水线：输入(base 模型 + 合适模板 + 数学题集) → 每题采一组 G 个响应、用规则验证器(Math-Verify)给 0/1 outcome 奖励(β=0，**去掉 KL 项**，省 ref 模型显存且对 R1-Zero 可能更好) → 算优势 Ã=R−mean(R)(**不除 std**) → token 级 loss 用 masked_sum / 常数(生成预算)归一化(**不除 |o_i|**) → PPO 风格 clip 更新 → 输出。【原文 §3/Eq.1-3 + v1 代码核】
- 逐组件必要性：
  - **去 1/|o_i|(Modification 1)**：没它→错误响应被激励变长(overthinking)。代码 `train_zero_math.py` L288-290:`masked_sum(..., constant_normalizer=generate_max_length) if critic_type=="drgrpo" else masked_mean`(v1 核)。【原文 §3.1/§3.2】
  - **去 std(R)(Modification 2)**：没它→难/易题(std≈0，奖励几乎全 1 或全 0)被赋过高权重，引入难度偏置。代码 `compute_monte_carlo_advantages` L294-308:`advantages=rewards-values`，仅 grpo 才 `/=(std+1e-8)`(v1 核)。【原文 §3.1/§3.2】
  - 两处修正各有清晰数学动机(恢复无偏优势)，Fig.4 给出"为什么有偏"的图示，是本文最扎实的部分。
- 关键机制/公式(直觉)：GRPO 把每个 token 的优势 ˆA_{i,t}=(R_i−mean(R))/std(R)，再在 loss 里对每条响应除以 |o_i|。等价地(Fig.4)，每 token 实际权重 = Ã_{i,t}·(1/|o_i|)·(1/std(R))。1/|o_i| 让长响应每 token 梯度被稀释、短响应被放大——对正优势偏好简短、对负优势偏好冗长(拖长错误回答摊薄惩罚)；1/std(R) 让组内方差小的题(太难/太易)权重过高。去掉二者→优势回到纯蒙特卡洛回报减组均值(无偏 baseline)。直觉:"别让算法因为分母而偏爱长错答或极端难度题"。【原文 §3.1/Fig.4】
- 实验与证据：
  - **极简 SOTA 配方**：Dr. GRPO 对 Qwen2.5-Math-7B 在 MATH lv.3-5 + Qwen-Math 模板上 RL → AIME24 **43.3%**(Fig.2 Oat-Zero-7B Avg 51.4，超 SimpleRL/PRIME/OpenReasoner-Zero)，仅 8×A100、27h。【原文 §1/Fig.2】
  - **算法对比**：vanilla GRPO 与 Dr. GRPO 的 reward 曲线相近，但 Dr. GRPO **阻止响应长度失控、错误响应长度大幅下降**、token 效率更高(Fig.1 右)。【原文 §3.1/Fig.1】
  - **base 发现**：Table 1——Qwen2.5-Math-7B 在 No template 下 Avg 38.2，而 **R1 template 下 Avg 暴跌到 0.0**(模板-模型严重不匹配会先破坏能力)；No template 比 4-shot(23.8)高 ~60%。Fig.3——所有 base 都"可探索"(pass@8>0，Qwen2.5 最强)；DeepSeek-V3-Base 在 R1 模板下也产生相当多自反思("Aha"/"wait")。【原文 §2.1-2.3/Table 1/Fig.3】
  - **自反思≠更准**：在 R1-Zero 上自反思更频繁，但**与更高准确率不正相关**(§2.3 末/§F)——削弱"自反思→更强推理"的强主张。【原文 §2.3/§F】
  - baseline 公平吗：算法对比同 base(Qwen2.5-1.5B)、同 R1 模板、同 Math-Verify 0/1 奖励，较公平；base 分析用统一 500 MATH 题 + GPT-4o-mini 判"是否作答格式"。
  - "看着强但没回答核心问题"：算法侧(去 bias→省 token 不掉分)直接切题且证据强；base 侧多为观察性/相关性证据。
- 假设与失效边界：
  - 【原文 §2.2】"Qwen2.5 预训练用了 QA 拼接文本"是从"去模板性能反升 60%"**反推的假设**(Qwen2.5 数据未公开)，非直接数据审计。
  - 【原文 §3 / Hu et al.】假设 **β=0(去 KL)**——因用规则验证器无分布漂移之虞；在用学习型奖励模型时该假设不成立。
  - 【推断】**评测几乎全是数学**——去偏置结论在代码/通用推理上的迁移性未验证。依据:训练/评测集均为 MATH/AIME/AMC/Minerva/Olympiad。
  - 【推断】**去 std 的代价**:std 归一化本意是稳定不同难度题的梯度尺度，移除后在极端难度分布下是否引入新不稳定，论文着墨不多；常数归一化(generate_max_length)引入新超参，其取值影响梯度尺度、需配 lr 调。
- 祛魅总结【推断】：
  - 真贡献：**算法侧**(Fig.4 的偏置分解 + 两行去偏 + token 效率提升)是干净、可复核、影响广的结果，已被社区广泛引用/采纳；极简 7B 配方提供了强 baseline。**认知侧**祛了"长度增长=推理变强""Aha=RL 涌现""自反思→更准"三个被过度解读的叙事。
  - 包装/证据强度落差：part 部分批判性结论(尤其 base 归因:"已类 SFT""Aha 早已存在")的**证据强度低于其修辞强度**——多为相关性/定性识别，缺统一量化判据与数据审计。"competitive minimalist SOTA"依赖特定 base(Qwen2.5-Math)与数学域，普适性有限。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：规则验证器(Math-Verify)的 outcome 0/1 奖励 → 蒙特卡洛回报 → 组内减均值的无偏优势(不除 std)；属 RLVR。
  - **改什么**：改 **RL 目标的归一化**(去 1/|o_i| 与 std(R))；不改模型结构/数据，只改 advantage 与 loss 归一化。
  - **何时改**：RL 训练时(每步 rollout 后算优势、更新)。
  - **免梯度?**：否——这是策略梯度/PPO 风格 RL，核心就是梯度优化(但去掉 critic/value 模型、去掉 KL ref)。
  - **记忆-技能生命周期**：纯参数内 RL，无外部记忆/技能库；"技能"=推理能力，靠 RL 在 base 已有探索能力上放大。
  - **防遗忘机制**：无(且明确 β=0 去掉了 KL-to-ref 这一最常见的"防漂移"项)；本文关注 token 效率而非遗忘。
- ⑦ 开源代码+框架/harness：https://github.com/sail-sg/understand-r1-zero (v1 记已克隆 ~56MB)。训练入口 `train_zero_math.py`(含 Modification 1 masked_sum 常数归一化、Modification 2 MC 优势不除 std，v1 均核对)；算法包 `understand_r1_zero/`(含 `math_grader.py`)。**框架=自研 Oat**(sail-sg 的模块化 LLM online alignment/RL 框架，底层 vLLM + DeepSpeed)，奖励 Math-Verify。资产 HF `sail/Oat-Zero`。代码可得性高。【v1 仓库核查】
- 💰 资源/成本与可扩展性：极简配方 8×A100、27h 出 7B SOTA(强调低成本)；去 KL 省 ref 模型显存/算力；去 critic(GRPO 本就免 value 模型)。【原文 §1/§3】
- 🎯 对"探索-巩固"对标：**中-强支撑(机理诊断层)+ 直接可借的训练信号工具**。判定依据：① **token 效率/抑制 overthinking** 与本课题"巩固=固化有效路径而非冗长试错"目标一致——Dr. GRPO 证明"无偏优势能让模型用更短正确路径解题"，对"路径选择/巩固"有直接价值；② **base 已具探索能力(pass@8)+ 模板构造探索性 base policy** 的观察支撑本课题"探索=发现自己能走通的开头"——RL/OPD 的探索上限由 base 决定，"偏向自己能走通的开头"有据。**可借组件**:去偏的无偏优势 + 常数归一化，可作 MTP/OPD 的 RL 巩固阶段的稳定信用分配基座(避免长度/难度偏置污染"关键步"的信用)。**缺口**:①无 teacher 脚手架/无蒸馏(纯 self-play RL)；②无 path-recovery 单点接管、无显式"走偏→回轨"机制(只是统计上缩短错误响应)；③无 MTP/前瞻、无记忆/技能库；④"自反思≠更准"的发现对本课题是**警示**——别假设让 student 多自反思就更好，要看是否真提升解题。
- 🔭 开放问题/未来方向：
  - 【原文】把无偏优化推广到代码/通用推理；更系统地量化"自反思 vs 准确率"的关系；理解 base 预训练特性如何决定 RL 上限(§2.4 Llama-3.2-3B 数学续训抬高 RL 天花板)。
  - 【推断】把"去 bias 的信用分配"与"关键步/高熵分支的稀疏信用"结合，让 RL 只在真正影响成败的步上给信号(贴近本课题稀疏脚手架)；研究去 std 后对极端难度题的稳定性补丁。

RETURN: dr_grpo|读到PDF=是(§1-3.1全文+Eq.1-3/Table1/Fig.1-4/§2.3自反思≠更准)|L线=L3(兼L6)|对标=中-强支撑(无偏优势抑制overthinking利"巩固短正确路径"、base探索能力支撑"走通的开头";可借去偏信用分配;但无脚手架/回轨/MTP,自反思≠更准是警示)|残留待核=0

dr_grpo | Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO) | Sea AI Lab · NUS · SMU(Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi 等) | 2025-10-06 · arXiv 2503.20783 v2 · COLM 2025(+ICML 2025 AI4Math Workshop Best Paper 荣誉提名) | 主题线 L3(RLVR/GRPO·训练机理+算法)，兼 L6(token 信用) · 相关性 高

**原始论文**：https://arxiv.org/abs/2503.20783

## 一眼看懂
- 🟦 TL;DR：批判性审视 R1-Zero 范式的两大成分。**(成分一 base 模型)**:Qwen2.5 base **不用对话模板**时性能反而暴涨约 60%(疑似预训练就喂了拼接的 question-answer 文本→"已类 SFT");"Aha moment(自反思)"在多个 base(含 DeepSeek-V3-Base)中**RL 之前就已存在**,并非纯 RL 涌现,且自反思行为与最终准确率**不正相关**。**(成分二 RL 算法)**:GRPO 目标里的 **\(1/|o_i|\)(响应级长度偏置)** 与 **\(\mathrm{std}(R)\)(题目级难度偏置)** 会人为推高响应长度(尤其推长**错误**响应);去掉这两项得到无偏的 **Dr. GRPO**,在不掉推理性能下大幅缩短错误响应、提升 token 效率,并给出 7B 极简 SOTA 配方(AIME24 43.3%,8×A100/27h)。【原文 §Abstract / §2 / §3.1 / Fig.1–2】
- 最巧的一步：**从 GRPO 目标里删掉 \(1/|o_i|\) 与 \(\mathrm{std}(R)\) 两个归一化项**(Fig.1 左)。抽掉这个"删除"就回到 vanilla GRPO 的病态——Fig.4 证明 GRPO 的有效优势 \(a_{i,t}\) 等价于无偏优势 \(\tilde A_{i,t}=R-\mathrm{mean}(R)\) 被 \(1/|o_i|\) 与 \(1/\mathrm{std}(R)\) **重加权**后的版本;对负优势(错误响应),长响应因 \(|o_i|\) 大而被罚得轻→策略偏好"把错误答案拖长"以摊薄惩罚。这是把"RL 长度增长是 feature"重新诊断为"部分是优化 bug"的关键一刀。【原文 §3.1 / Fig.4】

## 为什么做
- 研究背景：DeepSeek-R1-Zero(Guo et al. 2025)证明可不经 SFT、直接对 base 大规模 RL 提升推理,伴随 RL scaling(响应长度持续增长)与"Aha moment"。社区多用 Qwen2.5 + GRPO 复现(SimpleRL-Zero、Open-Reasoner-Zero、PRIME 等)。【原文 §1】
- 解决的具体痛点：厘清 R1-Zero 里**哪些现象是真涌现、哪些是预训练遗留 / 优化假象**;并修掉 GRPO 的优化偏置以提升 token 效率;给一个极简配方。【原文 §1 takeaways】
- 相关工作 & 各自不足(原文 §2.3 / §3.1 末,逐条点名)：
  - **GRPO(Shao et al. 2024)**:组内相对优势 + 长度归一化的免 critic PPO 变体——其 \(1/|o_i|\) 长度归一化与 \(\mathrm{std}(R)\) 题目归一化引入偏置(本文核心批判对象)。
  - **前人对 Aha 的怀疑(Liu et al. 2025b、Yeo et al. 2025)**:已疑"开源 R1 复现里 Aha 是 base 自带",但**没测真正训出 R1-Zero 的 DeepSeek-V3-Base**——本文自托管 685B V3-Base 补上这一块,确认它也有 Aha。
  - **开源 PPO/GRPO 实现普遍有长度 bias(Table 2)**:trl(von Werra)、OpenRLHF(Hu)、verl(Sheng)、SimpleRL-Zero(Zeng)、Open-Reasoner-Zero(Hu)——**全部**用响应长度归一化 loss(`masked_mean`),与 PPO 原始目标 Eq.2 不符;作者推测这源自预训练阶段"按固定上下文长度归一化提数值稳定"的惯性,但 RL 阶段响应长度非常数,引入意外 length bias。【原文 §2.3 / §3.1 末 / Table 2 / Listing 1】
- 动机链：现状(R1-Zero 把"长度增长 + Aha"当 RL 涌现的证据)→ 质疑(① base 是否本就会做题/会自反思? ② 长度增长是真推理需要还是优化 bug?)→ 诊断(base 去模板性能反升 60% + V3-Base 有 Aha → 归因部分站不住;Fig.4 显示 GRPO 优势被长度/std 重加权)→ 所以(去掉两个归一化得无偏 Dr. GRPO;并据 base 分析定极简配方)。【原文 §2–3】
- 与最近邻工作的 Δ：相对**原版 GRPO**——Dr. GRPO 把 token 级 loss 的归一化从"除以本响应长度 \(|o_i|\)"改为"除以常数(生成预算)"、把优势从"减均值再除组内 std"改为"只减均值不除 std",恢复了**无偏的 PPO 风格优势**(蒙特卡洛回报 + 无偏 baseline)。相对其它 R1-Zero 复现——本文不追新算法,而是**批判性归因 + 极简化**。【原文 §3.1–3.2 / Fig.1】

## 怎么做 + 靠不靠谱
- **方法流水线(可复现级)**：输入(base 模型 + 合适模板 + 数学题集) → 每题采一组 \(G\) 个响应 → 用规则验证器(Math-Verify)给 0/1 outcome 奖励 \(R(q,o)=\mathbb{1}[o\text{ 含正确最终答案}]\)(取 \(\beta=0\),**去掉 KL 项**,省 ref 模型显存且对 R1-Zero 可能更好) → 算优势 \(\tilde A=R-\mathrm{mean}(R)\)(**不除 std**) → token 级 loss 用 `masked_sum / 常数(生成预算)` 归一化(**不除 \(|o_i|\)**) → PPO 风格 clip 更新 → 输出。【原文 §3 / Eq.1–3 + 本地代码核】
- **核心目标的真实形式 + 直觉**：
  - RL 目标(token 级 MDP,Eq.1):\(\mathcal{J}(\pi_\theta)=\mathbb{E}_{q\sim p_Q}\big[\mathbb{E}_{o\sim\pi_\theta(\cdot\mid q)}[R(q,o)]-\beta D_{\mathrm{KL}}[\pi_\theta(\cdot\mid q)\|\pi_{\mathrm{ref}}(\cdot\mid q)]\big]\),本文全程取 \(\beta=0\)(规则验证器无分布漂移之虞,可去 KL)。
  - PPO 代理(Eq.2):\(\mathcal{J}_{\mathrm{PPO}}=\mathbb{E}\sum_{t=1}^{|o|}\min\big(\tfrac{\pi_\theta(o_t\mid q,o_{<t})}{\pi_{\theta_{\mathrm{old}}}}\hat A_t,\;\mathrm{clip}(\tfrac{\pi_\theta}{\pi_{\theta_{\mathrm{old}}}},1-\epsilon,1+\epsilon)\hat A_t\big)\)。
  - **GRPO(有偏,Eq.3)**:\[\mathcal{J}_{\mathrm{GRPO}}=\mathbb{E}\,\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\!\big(\cdots\hat A_{i,t},\,\mathrm{clip}(\cdots)\hat A_{i,t}\big),\quad \hat A_{i,t}=\frac{R(q,o_i)-\mathrm{mean}(\{R(q,o_j)\})}{\mathrm{std}(\{R(q,o_j)\})}.\]
  - **Dr. GRPO(无偏,Fig.1 左)**:删掉 \(\tfrac{1}{|o_i|}\) 与 \(\mathrm{std}(\{R\})\),即 \(\hat A_{i,t}=R(q,o_i)-\mathrm{mean}(\{R(q,o_j)\})\),且 loss 用常数 `MAX_TOKENS`(生成预算)归一化。原文证明:这恰好**恢复 PPO 目标 Eq.2**,优势为"蒙特卡洛回报 + 无偏 baseline"(Sutton & Barto)。
  - **偏置分解(Fig.4,核心机制)**:GRPO 的每 token 有效优势 \(a_{i,t}=\tilde A_{i,t}\cdot\tfrac{1}{|o_i|}\cdot\tfrac{1}{\mathrm{std}(R)}\),其中 \(\tilde A_{i,t}=R(q,o_i)-\mathrm{mean}(R)\)。两个偏置:
    - **响应级长度偏置(来自 \(1/|o_i|\))**:正优势(正确)时短响应每 token 梯度被放大→偏好简短正确;**负优势(错误)时长响应因 \(|o_i|\) 大被罚得轻→偏好把错误答案拖长**(overthinking 的优化根源)。
    - **题目级难度偏置(来自 \(1/\mathrm{std}(R)\))**:std 小的题(太易/太难,奖励几乎全 1 或全 0)被赋过高权重;advantage normalization 本应跨整 batch 算,GRPO 却按题归一化→难度偏置。
  - 直觉:"别让算法因为分母而偏爱长错答或极端难度题"。【原文 §3 / Eq.1–3 / Fig.4】
- 逐组件必要性：
  - **去 \(1/|o_i|\)(Modification 1)**：没它→错误响应被激励变长(overthinking)。本地核 `train_zero_math.py` L288–290:`masked_sum(..., constant_normalizer=generate_max_length) if critic_type=="drgrpo" else masked_mean`。【原文 §3.1/§3.2】
  - **去 \(\mathrm{std}(R)\)(Modification 2)**：没它→难/易题被赋过高权重。本地核 `compute_monte_carlo_advantages` L294–308:`advantages = rewards - values`,仅 grpo 才 `/= (std + 1e-8)`。【原文 §3.1/§3.2】
  - 两处修正各有清晰数学动机(恢复无偏优势),Fig.4 给出"为什么有偏"的图示,是本文最扎实的部分。
- 实验与证据：
  - **极简 SOTA 配方**：Dr. GRPO 对 Qwen2.5-Math-7B 在 MATH lv.3-5 + Qwen-Math 模板上 RL → AIME24 **43.3%**(Fig.2 Oat-Zero-7B Avg 51.4,超 SimpleRL-Zero 38.2 / PRIME 48.0 / OpenReasoner-Zero 43.0),仅 8×A100、27h。【原文 §1 / Fig.2】
  - **算法对比(Fig.5)**：vanilla GRPO 与 Dr. GRPO 的 reward 曲线相近(Plot 1),但 Dr. GRPO **阻止响应长度失控**(Plot 2)、**错误响应长度大幅下降**(Plot 4)、token 效率更高;两者都呈"长度随 reward 增长"的 R1-Zero 趋势(但 GRPO 的增长部分是 bug)。【原文 §3.1 / Fig.5】
  - **base 发现**：Table 1——Qwen2.5-Math-7B 在 No template 下 Avg 38.2,而 **R1 template 下 Avg 暴跌到 0.0**(模板-模型严重不匹配会先破坏能力);No template(38.2)比 4-shot(23.8)高 ~60%。Fig.3:三轴(Question-Answering Ability / Exploration pass@8 / Self-Reflection),所有 base 都"可探索"(pass@8>0,Qwen2.5 最强,甚至超 DeepSeek-V3-Base);DeepSeek-V3-Base 在 R1 模板下也产生相当多自反思("Aha"/"wait",Fig.13)。原文据 pass@8 给出 RL 可行性判据:"if a base policy cannot even sample a single trajectory that leads to the correct final answer, it is impossible for RL to improve"。【原文 §2.1–2.3 / Table 1 / Fig.3】
  - **自反思≠更准**：自托管 DeepSeek-R1-Zero 分析同一批 MATH 题,自反思更频繁,但**与更高准确率不正相关**(§2.3 末 / §F)——削弱"自反思→更强推理"的强主张。【原文 §2.3 / §F】
  - **§2.4 补充**:Math pretraining on Llama-3.2-3B 抬高其 RL 天花板——说明 RL 上限由 base 预训练特性决定。【原文 §2.4 takeaway】
  - baseline 公平吗：算法对比同 base(Qwen2.5-1.5B)、同 R1 模板、同 Math-Verify 0/1 奖励——较公平;base 分析用统一 500 MATH 题 + GPT-4o-mini 判"是否作答格式"。
  - "看着强但没回答核心问题":算法侧(去 bias→省 token 不掉分)直接切题且证据强;base 侧多为观察性/相关性证据。
- 假设与失效边界：
  - 【原文 §2.2】"Qwen2.5 预训练用了 QA 拼接文本(maximize \(\log p_\theta(q;o)\))"是从"去模板性能反升 60%"**反推的假设**(Qwen2.5 数据未公开),非直接数据审计——作者措辞 "we hypothesize"。
  - 【原文 §3 / Hu et al.】假设 **\(\beta=0\)(去 KL)**——因用规则验证器无分布漂移之虞;在用学习型奖励模型时该假设不成立。
  - 【推断】**评测几乎全是数学**——去偏置结论在代码/通用推理上的迁移性未验证。依据:训练/评测集均为 MATH/AIME/AMC/Minerva/Olympiad。
  - 【推断】**去 std 的代价**:std 归一化本意是稳定不同难度题的梯度尺度,移除后在极端难度分布下是否引入新不稳定,论文着墨不多;常数归一化(`generate_max_length`)引入新超参,其取值影响梯度尺度、需配 lr 调(原文 Listing 1 注:"Other constants also work with differences in gradient norm")。
- 祛魅总结【推断】：
  - 真贡献:**算法侧**(Fig.4 的偏置分解 + 两行去偏 + token 效率提升)是干净、可复核、影响广的结果,已被社区广泛引用/采纳;极简 7B 配方提供了强 baseline。**认知侧**祛了"长度增长=推理变强""Aha=RL 涌现""自反思→更准"三个被过度解读的叙事。
  - 包装/证据强度落差:部分批判性结论(尤其 base 归因:"已类 SFT""Aha 早已存在")的**证据强度低于其修辞强度**——多为相关性/定性识别,缺统一量化判据与数据审计。"competitive minimalist SOTA"依赖特定 base(Qwen2.5-Math)与数学域,普适性有限。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：规则验证器(Math-Verify)的 outcome 0/1 奖励 → 蒙特卡洛回报 → 组内减均值的无偏优势 \(\tilde A_{i,t}=R-\mathrm{mean}(R)\)(不除 std);属 RLVR。
  - **改什么**：改 **RL 目标的归一化**(去 \(1/|o_i|\) 与 \(\mathrm{std}(R)\));不改模型结构/数据,只改 advantage 与 loss 归一化。
  - **何时改**：RL 训练时(每步 rollout 后算优势、更新)。
  - **免梯度?**：否——这是策略梯度/PPO 风格 RL,核心就是梯度优化(但去掉 critic/value 模型、去掉 KL ref)。
  - **记忆-技能生命周期**：纯参数内 RL,无外部记忆/技能库;"技能"=推理能力,靠 RL 在 base 已有探索能力上放大。
  - **防遗忘机制**：无(且明确 \(\beta=0\) 去掉了 KL-to-ref 这一最常见的"防漂移"项);本文关注 token 效率而非遗忘。
- ⑦ 开源代码+框架/harness：https://github.com/sail-sg/understand-r1-zero (本地已克隆 ~56MB)。训练入口 `train_zero_math.py`(含 Modification 1 masked_sum 常数归一化、Modification 2 MC 优势不除 std,**本地均核对**);算法包 `understand_r1_zero/`(含 `math_grader.py`)。**框架=自研 Oat**(sail-sg 的模块化 LLM online alignment/RL 框架,底层 vLLM + DeepSpeed),奖励 Math-Verify。资产 HF `sail/Oat-Zero`。代码可得性高。【本地仓库核查】
- 💰 资源/成本与可扩展性：极简配方 8×A100、27h 出 7B SOTA(强调低成本);去 KL 省 ref 模型显存/算力;去 critic(GRPO 本就免 value 模型)。【原文 §1 / §3】
- 🎯 对"探索-巩固"对标：**中-强支撑(机理诊断层)+ 直接可借的训练信号工具**。判定依据:① **token 效率/抑制 overthinking** 与本课题"巩固=固化有效路径而非冗长试错"目标一致——Dr. GRPO 证明"无偏优势能让模型用更短正确路径解题"(Fig.5 Plot 4 错误响应长度下降),对"路径选择/巩固"有直接价值;② **base 已具探索能力(pass@8>0)+ 模板构造探索性 base policy** 的观察支撑本课题"探索=发现自己能走通的开头"——RL/OPD 的探索上限由 base 决定,"偏向自己能走通的开头"有据。**可借组件**:去偏的无偏优势 \(\tilde A=R-\mathrm{mean}(R)\) + 常数归一化,可作 MTP/OPD 的 RL 巩固阶段的稳定信用分配基座(避免长度/难度偏置污染"关键步"的信用)。**缺口**:① 无 teacher 脚手架/无蒸馏(纯 self-play RL);② 无 path-recovery 单点接管、无显式"走偏→回轨"机制(只是统计上缩短错误响应);③ 无 MTP/前瞻、无记忆/技能库;④ "自反思≠更准"的发现对本课题是**警示**——别假设让 student 多自反思就更好,要看是否真提升解题。
- 🔭 开放问题/未来方向：
  - 【原文】把无偏优化推广到代码/通用推理;更系统地量化"自反思 vs 准确率"的关系;理解 base 预训练特性如何决定 RL 上限(§2.4 Llama-3.2-3B 数学续训抬高 RL 天花板)。
  - 【推断】把"去 bias 的信用分配"与"关键步/高熵分支的稀疏信用"结合,让 RL 只在真正影响成败的步上给信号(贴近本课题稀疏脚手架);研究去 std 后对极端难度题的稳定性补丁。

RETURN: dr_grpo | 读PDF=是(§1–3.2 全文 + Eq.1–3/Table1/Table2/Fig.1–5/§2.3 自反思≠更准/§2.4,并本地核 train_zero_math.py 两处修改) | 加厚=是(新增 RL 目标 Eq.1、PPO Eq.2、GRPO Eq.3、Dr.GRPO 优势、Fig.4 偏置分解 \(a_{i,t}=\tilde A/|o_i|/\mathrm{std}\) 全 MathJax;related work 补 Table2 五个有 bug 的开源实现 + Listing1;补 pass@8 RL 可行性判据) | LaTeX公式=7条(RL目标Eq.1、PPO Eq.2、GRPO Eq.3+优势、Dr.GRPO优势、偏置分解 a_{i,t}、奖励 R、内联多处) | 待核=0

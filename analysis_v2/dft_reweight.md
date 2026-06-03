dft_reweight | On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification (DFT / Dynamic Fine-Tuning) | 东南大学·UCLA·上海交大·南洋理工·UC Berkeley·武汉大学·UC Merced 等(Yongliang Wu, Yizhou Zhou 等) | 2026-02-27 · arXiv 2508.05629 v3 · ICLR 2026 | 主题线 L2(统一SFT-RL视角)，兼 L3 · 相关性 高

**原始论文**：https://arxiv.org/abs/2508.05629

## 一眼看懂
- 🟦 TL;DR：把标准 SFT 的梯度用重要性采样改写成"策略梯度"形式后会发现，它隐含的奖励是**稀疏的指示函数(只在精确匹配专家 token 时=1)、且被 1/π_θ(逆概率)加权**——当模型给专家 token 的概率很低时这个权重爆炸→梯度异常大→训练不稳、过拟合稀有精确匹配。修法极简：**给每个 token 的交叉熵损失乘上该 token 的预测概率 π_θ(detach 阻断梯度)**，刚好把 1/π_θ 抵消，使隐含奖励整平为常数 1(类似 RLVR 给所有正确样本统一奖励)。一行代码，在数学/代码/多模态推理上显著超过 SFT。【原文 §Abstract/§3.2-3.3/Eq.5-9】
- 最巧的一步：**乘 π_θ 并 stop-gradient(sg)**。抽掉 sg 就垮——若梯度流过这个概率系数，乘性项会引入额外的"提高自身概率"梯度，不再是"整平奖励"而变成别的目标；§Eq.7 明确 sg 保证"梯度不流过 reward scaling 项 w"。另一个不可抽的是**token 级**(而非句级)加权：整句概率连乘极小、loss 几乎无信息(Table 5 句级 15.75≈base 15.92，几何均值 17.21，token 级 31.58)，必须 token 级才有效——这点与 PPO 的 token 级重要性采样同源。【原文 Eq.7/Eq.9/§4.6 Table 5】

## 为什么做
- 研究背景：SFT(拟合专家 demonstration，类机器人 behavioral cloning)是 LLM 后训练标准范式，简单高效；但"SFT memorizes, RL generalizes"(Chu et al. 2024)——SFT 易过拟合、泛化弱于 RL。RL 泛化好但算力贵、需显式奖励、调参敏感，且在"只有正样本、无负样本/无奖励模型"时不可用。【原文 §1/§2】
- 解决的具体痛点：**SFT 本身能否被根本性改进?** ——尤其在数据只含正样本、无法上 RL 的场景。【原文 §1 末】
- 相关工作 & 各自不足：
  - **混合 SFT+RL(InstructGPT 式 / 交错式)**：丰富了 pipeline 但不改 SFT 本体；需奖励/偏好/负样本。【原文 §2】
  - **DPO / NFT(Chen 2025b 负样本隐式策略)**：仍依赖偏好对或负样本。【原文 §2】
  - **统一 SFT-RL 的理论线(Du et al. 2025 把 RLHF 看作 reward-weighted SFT；Wang 2025a SFT=带隐式奖励的 RL；Qin & Springenberg 2025 iw-SFT 把 SFT 看作 RL 下界并按数据生成策略做重要性加权；GOLD/MixCE)**：建立了 SFT↔RL 的加权联系，但**没给出 SFT 梯度与 offline 策略梯度的精确数学等价**，也没把症结落到"1/π_θ 这一项"。【原文 §2，作者明确这是其 Δ】
  - **Focal Loss 对照(关键洞察)**：DFT 的修改后 CE 形如 −p·log p，而 Focal Loss 是 −(1−p)^γ·log p——**两者加权哲学完全相反**：Focal 降"已分类好"样本权重以强调难例(欠拟合时代)，DFT 降"分类差(低概率)"样本权重以促泛化(过拟合时代)。这个对比是理解 DFT 本质的最佳锚点。【原文 §2 末】
- 动机链：现状(SFT 泛化弱、但很多场景只能用 SFT)→ 数学诊断(Eq.5 用重要性采样把 SFT 梯度写成 on-policy 期望，Eq.6 显出奖励=1[y=y⋆]/π_θ 形式)→ 症结(1/π_θ 在低概率专家 token 处爆炸→大梯度→不稳+过拟合精确匹配)→ 所以(乘回 π_θ 抵消畸变，整平奖励)。【原文 §3.2-3.3】
- 与最近邻工作的 Δ：相对并发的 **iw-SFT(Qin & Springenberg)**——同样从"SFT 是带重要性权重的 RL"出发，但 iw-SFT 按"数据生成策略"加权，DFT 直接定位到 1/π_θ 这一项并用 sg(π_θ) 抵消(附录 A.4 专门讨论与 iw-SFT 的区别)。差在把"加权"的来源精确归到逆概率项，给出更形式化的策略梯度等价(附录 A.2)。【原文 §2/§4.1 末 A.4】

## 怎么做 + 靠不靠谱
- 方法流水线：输入(专家 demonstration {(x,y⋆)}) → 标准前向得每个 target token 的 π_θ(y⋆_t|·) → 把逐 token CE 乘上 **sg(π_θ(y⋆_t|·))** → loss = −Σ_t sg(π_θ(y⋆_t|y⋆_<t,x))·log π_θ(y⋆_t|y⋆_<t,x) → 反向(sg 项视为常数) → 更新。输出：一个泛化更好的 SFT 模型。【原文 Eq.9】
- 逐组件必要性：
  - **乘 π_θ + sg(核心，唯一改动)**：抽掉就退回普通 SFT。理论必要性见 Eq.5→6→7 推导；实证必要性见全表 DFT≫SFT。
  - **token 级 vs 句级(Table 5 消融)**：句级概率连乘数值崩塌、信号极弱(15.75/17.21 ≈ base)，token 级才有用(31.58)。这是一个直接的必要性消融。【原文 §4.6/Table 5】
  - **加权策略对比(Table 5)**：还对比了 GSPO 风格的几何均值句级加权——仍弱。说明"必须 token 级 + 直接乘概率"。【原文 §4.6】
- 关键机制/公式(直觉)：CE 对低概率 token 给最大梯度(要把 π 从很低推高需大更新)；从 RL 视角，这等价于给这些 token 一个 1/π 的巨大"奖励权重"，是病态的。乘上 π_θ 后，有效奖励对所有专家 token 变成 1，相当于"所有正确路径同等对待"，不再对个别低概率参考 token 过度集中。直觉效果(Fig.2)：SFT 把所有概率一律往训练集推(尤其抬低概率 token)；DFT 呈**两极化/双峰**——抬一部分、压另一部分，被压的多是 the/let/逗号/句号这类语法功能词，相当于"别死磕连接词、聚焦实质语义"，是一种正则化。【原文 §3.3/§4.7/Fig.2】
- 实验与证据(四组)：
  - **① 数学 SFT(Table 1)**：NuminaMath-CoT 采 100k 训练，5 个 base(Qwen2.5-Math-1.5B/7B、LLaMA-3.2-3B、LLaMA-3.1-8B、DeepSeekMath-7B)。Qwen2.5-Math-1.5B：DFT 平均 31.58(+15.66 over base)，是 SFT 增益(+2.09)的 **5.9×**；Qwen2.5-Math-7B DFT 37.15(+15.90)是 SFT 的 ~3.8×。关键：SFT 常在难基准上**退化**(如 Qwen2.5-Math-7B AIME24 从 6.68 掉到 2.48、OlympiadBench 1.5B 从 15.88 掉到 12.63)，DFT 反而提升(AIME24→8.56、Olympiad→27.08)。Avg@16、temp=1.0、16 次解码。【原文 §4.1/Table 1】
  - **② offline RL(Table 2，Qwen2.5-Math-1.5B)**：自采 100k×4 response 用 math-verify 筛得 ~140k，构 100k 偏好对训 DPO。DFT 平均 **35.43**，超最佳 offline 基线 RFT(+11.46)、甚至超最强 online 的 GRPO(+3.43)。逐项 AMC23 DFT 48.44(比 GRPO +7.19、比 RFT +17.66)；Minerva 25.16(比 GRPO +6.23、PPO +9.75)。且**不需 reference model / 大 batch**。【原文 §4.2/Table 2】
  - **③ 代码(Table 3)**：UltraFeedback 采 10k 选最高分 response，评 HumanEval/HE+/MultiPL-E。Qwen2.5-Coder-7B DFT 比 SFT 在 HE +12.8、HE+ +11.0、MultiPL-E avg +4.7。【原文 §4.3/Table 3】
  - **④ 多模态(Table 4)**：WeThink 训 Qwen2.5-VL-3B，评 MathVerse/MathVision/WeMath，全面超 base 与 SFT。【原文 §4.4/Table 4】
  - baseline 公平吗：Table 2 各基线(DPO 用 ms-swift lr 1e-6、PPO/GRPO 用 verl lr 1e-6、GRPO n=4)超参与数据构造不同，跨方法平均分比较的可比性需谨慎(各自是否同等调参未完全交代)。【原文 §4.2 实现细节】
  - "看着强但没回答核心问题"：主张是"改进 SFT 的泛化"，证据(难基准上由退化转提升)较切题；但 offline RL 表(DFT>GRPO)更像额外卖点，其公平性弱于主实验。
- 假设与失效边界：
  - 【原文 §4.5 关键失效案例】在 **Natural Questions(事实知识)** 上 SFT 把准确率从 31.24% 提到 36.62%，DFT 反而降到 30.14%——因为 DFT 按模型自身置信度重加权，会**强化已有信念**；当模型缺乏该事实知识时，这种强化反而**阻碍学习**。即 DFT 适合"与模型先验能力对齐的任务(逻辑推理/结构化预测)"，不适合"吸收新事实"。
  - 【原文 §5 Limitations】对**难例/训练数据中欠表示的样本**也可能不利——因为它们初始概率低、被降权。评测范围限于数学/代码，更大模型/更广任务未测。
  - 【原文 §3.2】Eq.6 的"SFT=策略梯度"是**理论透镜(theoretical lens)**而非严格等价：依赖把专家分布当 Dirac、奖励当指示函数等假设；作者明确"RL-style characterization serves solely as a theoretical lens"。
- 祛魅总结【推断】：
  - 真贡献：用 RL 透镜给出一个**优雅且可一行实现**的 SFT 改进，并把症结精确定位到 1/π_θ；Focal Loss 反演的类比(−p log p vs −(1−p)^γ log p)极具说服力地把它放进"过拟合时代的目标设计"叙事。实证覆盖数学/代码/多模态 + offline RL，分量足。
  - 包装/等价视角：所谓"隐含奖励病态"本质等价于"加权 CE 的梯度对低概率 token 敏感"这一已知事实；DFT 可被无须诉诸 RL 地理解为"**按模型置信度重加权 SFT / focal-loss 的反向版**"。RL 透镜提供了漂亮动机，但机制有多重等价解释，"competitive with PPO/GRPO"在不同设定下未必稳健。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：每个 target token 的**预测概率 π_θ(y⋆_t|·)** 作为乘性权重(detach)；奖励隐式=常数 1。不用外部奖励/教师 logits/负样本。
  - **改什么**：只改**损失函数**(CE × sg(prob))；模型/数据/流程同标准 SFT。
  - **何时改**：训练时(每步用当前 π_θ 算系数)；推理无改动。
  - **免梯度?**：是 SFT 范式、无 RL rollout、无 reference model、无大 batch；理论上是 offline 策略梯度的"整流版"。
  - **记忆-技能生命周期**：纯参数内"巩固"，无外部记忆/技能库；偏"更稳地把推理技能写进参数"。
  - **防遗忘机制**：无显式防遗忘设计。注意与同族 EAFT 的对照——EAFT 用**熵**门控显式防遗忘，DFT 用**概率**加权主打泛化；且 EAFT 论文实测 DFT 的遗忘缓解效果常差于 SFT(因概率无法区分"该学的低概率"vs"冲突的低概率")。【推断，依据 eaft.md/Table1】
- ⑦ 开源代码+框架/harness：https://github.com/yongliang-wu/DFT (v1 记录已克隆 ~66MB)。**框架=veRL**(FSDP SFT 训练器 `verl/verl/trainer/fsdp_dft_trainer.py`，v1 核到第 369–371 行即 detach 概率重加权：`probs=softmax(logits); coeff=probs.gather(labels); loss=loss*coeff.detach()`)+ Liger + bf16；评测沿用 Qwen2.5-Math 官方 pipeline；多模态用 LLaMA-Factory + VLMEvalKit。生态：TRL、LLaMA-Factory(`examples/extras/dft`)、ms-swift(`examples/train/full/dft.sh`)均一行启用。代码可得性高。【原文 §4.1/§4.4 + v1 仓库核查】
- 💰 资源/成本与可扩展性：成本 ≈ 标准 SFT(只多一次 softmax+gather)；主实验最大 7B、100k 数据、cosine decay、lr 5e-5(LLaMA-3.1-8B 用 2e-5)、batch 256、max_len 2048、1 epoch。offline RL 对照含自采 140k。【原文 §4.1/§4.2】
- 🎯 对"探索-巩固"对标：**中等支撑(巩固侧的损失工具)+ 部分竞品(同为单行 SFT 改造)**。判定依据：DFT 是"巩固时按置信度重加权"的代表，与本课题"forward-hard/backward-soft、按某统计量重加权 CE"同构，**可借组件**=sg(prob) 的 token 级乘性门控可直接用在 MTP/OPD 的巩固损失里。但**关键缺口/反例**：§4.5 证明 DFT 会**强化已有信念**、在"需吸收新知识/走偏后该纠正"的场景反而有害——这恰好与本课题"巩固=固化新有效行为且不遗忘"的目标相冲突(DFT 降低低概率 token 权重 = 降低"模型当前不会、恰恰最该学/最该回轨"的 token 的学习)。因此 DFT 更像"探索-巩固"框架要**警惕/改造**的对象，而非直接拿来用；EAFT 用熵门控正是对 DFT 这一缺陷的修正(更贴近本课题)。它也**无 on-policy 自选/无 path-recovery/无 teacher 脚手架**。
- 🔭 开放问题/未来方向：
  - 【原文 §5】更大模型/更广任务的验证；探索**非均匀 / 质量感知的奖励分配**(给 demonstration 赋不同奖励而非一律=1)。
  - 【推断】与 on-policy 蒸馏结合：用 student 自选轨迹替代固定专家 demonstration，可缓解 §4.5 的"强化错误先验"问题；以及把"概率门控(DFT)"与"熵门控(EAFT)"组合，分别管"泛化"与"防遗忘"。

RETURN: dft_reweight|读到PDF=是(全文+Eq.5-9/Table1-5/§4.5失效案例/§4.7,核到sg与token级)|L线=L2(兼L3)|对标=中等支撑但含反例(sg(prob)门控可借;但DFT强化先验、降权该学token,与"巩固新行为"冲突,需警惕)|残留待核=0

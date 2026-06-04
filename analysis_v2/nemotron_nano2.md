nemotron_nano2 | NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model | NVIDIA(Nemotron 团队，谱系承 Nemotron-H / Minitron) | 2025-09-02 · arXiv 2508.14444 v4 · 工业技术报告 | 主题线 L1(蒸馏:forward-KL logit KD)，兼 L3(GRPO/DPO/RLHF 多分支对齐)、L4(on-policy DPO 工具调用) · 相关性 中

**原始论文**：https://arxiv.org/abs/2508.14444

## 一眼看懂
> 一句话导读:目标是把推理模型塞进单张 22GiB 显卡还能跑 128k 上下文——先训 12B 大模型,再用 Minitron(剪枝 + forward-KL logit 蒸馏)压到 9B 恢复精度,关键是它的"蒸馏"是固定教师 logits 的 off-policy KD,跟本课题的 on-policy 蒸馏不是一回事。

- 🟦 TL;DR:造一个能在**单张 A10G(22GiB)上做 128k 推理**的紧凑推理模型。路线分三步:
  - 预训练 12B 混合 Mamba-Transformer base(20T tokens, FP8);
  - 多阶段 SFT + 多分支对齐(IFeval RL / on-policy DPO 工具 / GRPO-RLHF + 模型合并)得对齐 12B;
  - 用 **Minitron(剪枝 + 仅 forward-KL 的 logit 蒸馏)** 压到 9B 恢复精度。
  - 结果:在 8k 输入/16k 输出等生成密集场景,吞吐比 Qwen3-8B 高 **3×-6×**,同时精度相当或更优。【原文 §Abstract/§1/§4.4/Fig.1】
- 最巧的一步(两个,都是"对症工程一刀"非新算法):
  - 从工程系统看:**"先训大(12B)再剪小(9B)再 forward-KL 蒸馏恢复"** 这条 Minitron 路线(§4.3:logit 蒸馏在精度恢复阶段优于普通微调);
  - 从"可控推理"看:**训练时混入"突然截断到 1-2k token 的推理轨迹"**(§3.4)——抽掉它,模型在短 thinking budget(思考预算)下就会停不下来、生成多余 `</think>`、well-formedness(格式良好率)暴跌(Fig.5a vs 5b)。【原文 §3.4/§4.3/Fig.5】

## 为什么做
> 一句话导读:推理模型又大又慢,12B 权重塞不进 22GiB 显卡,而且短思考预算下停不下来、几种对齐能力还互相打架——所以要压缩 + 控 budget + 多分支合并一起解决。

- 研究背景:推理模型要生成长 thinking 轨迹,对吞吐压力大。Nemotron-H 提出把多数自注意力层替换为 Mamba-2 的混合架构提升长序列生成速度;压缩侧承袭 Minitron(剪枝 + 知识蒸馏)。【原文 §1/§2/§4.1】
- 解决的具体痛点:
  - ① 12B bf16 权重 **22.9GiB > A10G 22GiB**,**必须压缩**才能单卡 128k 推理;
  - ② 推理模型在不同 thinking budget 下**鲁棒性差**(budget 截断后仍停在 thinking 模式、生成多余 `</think>`、well-formedness 下降);
  - ③ Stage-1 SFT 的 128k 拼接**损害工具调用学习**;
  - ④ 推理能力与 chat 能力存在**权衡**。【原文 §1/§3.1-3.4/Fig.5】
- 相关工作 & 各自不足(站谁肩上 + 精确差异):
  - **Nemotron-H**(混合 Mamba-Transformer):提供架构基座,但**未做推理模型的压缩 + budget 控制**。
  - **Minitron / Muralidharan 2024 / Sreenivas 2024 / Taghibakhshi 2025**(剪枝 + KD 框架):本文直接复用其重要性估计与 forward-KL 蒸馏配方,但原框架为压缩 **base 模型**、未含推理模型的 budget/RL 恢复;本文扩展到压缩**推理**模型、并在剪枝-蒸馏链里嵌 DPO/GRPO/RLHF + post-RL KD + 合并。与 Minitron 的精确差异:① 加 Mamba head 剪枝轴的消融(发现本工作压缩比小、剪 Mamba 收益有限故弃);② 把 KD 恢复与多分支对齐缝成一条链。
  - **Qwen3-8B**(对照模型):同精度下生成密集场景吞吐 3×-6×(混合 Mamba 架构之功)。
  - **各对齐算法 SFT/GRPO/DPO/RLHF**:各管一摊、彼此干扰(Fig.6)——本文用模型合并(Wortsman 2022 model soups)缓解。
  - 共性缺口:没人把"混合架构 + 推理模型压缩 + 可控 budget + 多能力不打架"在单卡内存约束下整合成一条产线。【原文 §1/§4.1】
- 动机链(逐步推):
  - 现状:推理模型又大又慢、budget 不可控、多能力相互干扰;
  - 约束:目标单 A10G 22GiB 跑 128k;
  - 所以:混合架构提吞吐 + Minitron 压到 9B 进内存 + 截断训练控 budget + 多分支独立对齐再合并缓解干扰。【原文 §1/§3/§4】
- 与最近邻工作的 Δ：
  - 相对 **Minitron**——把它从"压 base 模型"扩展到"压**推理**模型",并在剪枝-蒸馏链里**嵌入 DPO/GRPO/RLHF + post-RL KD 恢复 + 模型合并**(§4.3 步骤 1-7)。
  - 相对 **Qwen3-8B**——同精度下生成密集场景吞吐 3×-6×。
  - 差在"把工业级多能力对齐 + 激进内存压缩"做成可复现 recipe 并开源大量数据/权重。【原文 §4.3/§4.4/Fig.1】

## 怎么做(到可复现)
> 一句话导读:三大块——预训练 12B、多分支对齐成 12B、再 Minitron 压成 9B;压缩里剪枝只用前向传播估重要性,精度靠 forward-KL logit 蒸馏恢复,中途夹 DPO/GRPO/RLHF 并用 0.5 线性插值合并模型来防能力回退。

### 总体流水线
① **预训练** 12B 混合 Mamba-Transformer base(20T tokens,FP8,DeepSeek FP8 recipe;约 5% 数据含刻意截断推理轨迹) → ② **对齐**(Base→3 阶段 SFT→DPO/GRPO/RLHF 分支→Merged 12B) → ③ **Minitron 压缩**(剪枝→架构搜索→forward-KL logit KD 恢复→分阶段 DPO/GRPO/post-RL KD/RLHF/合并)得最终 9B。

### 压缩:剪枝重要性估计(§4.1,真实形式)
全程**只用前向传播**(梯度敏感性在 LLM 规模不实际)。
- **层重要性(MSE)**：迭代式——逐个临时移除候选层、算原模型 logits 与剪后 logits 的 MSE;**移除 MSE 最低(影响最小)的层**,重复到目标深度。
- **FFN/embedding 通道重要性**：FFN 形式 \(\text{FFN}(X)=\delta(X\cdot W_1^T)\cdot W_2\)(\(\delta\)=squared ReLU,\(W_1,W_2\in\mathbb{R}^{d_{ffn}\times d_{model}}\))。第一线性算子第 \(i\) 个神经元重要性按其输出在 batch×seq 维聚合:
\(\displaystyle F^{(i)}_{\text{neuron}}=\sum_{B,S}\delta\big(X(W_1^i)^T\big),\)
聚合函数用 mean(\(\frac1n\sum|S_i|\))与 l2-norm(\(\sqrt{\sum S_i^2}\)),校准集 1024 样本;embedding 通道同理(看 LayerNorm 输出)。
- **Mamba head 重要性**：按 Taghibakhshi 2025 的 nested activation 评分(1024 校准样本);**实测剪 Mamba head 导致严重精度损失**——因本工作压缩比小(深度剪后 <15%),剪 Mamba 收益有限,故**只剪 FFN+embedding+深度**。

### 压缩:架构搜索 + KD 恢复(§4.2-4.3)
1. **定深度**:固定 4 个 attention 层(attn:total≈7-8%),比较 52/54/56 层(各 6B token KD):52→44.92、54→47.35、**56→51.48** → 定 56 层。
2. **宽度剪枝 + 选候选**:在 56 层上做 60B token KD,再沿 embedding/FFN/Mamba 枚举满足 128k/bs1 内存预算的候选,按估计显存降序取 top-3,各做 **19B token 短 KD** + 测吞吐 → 选 **Candidate 2**(56 层/hidden 4480/FFN 15680/128 Mamba heads/8.89B,精度 63.02,吞吐 156.42)。
3. **forward-KL logit 蒸馏(精度恢复核心)**：**仅用 forward KL 散度损失**做 logit 蒸馏(教师=原 12B,固定 logits),优于普通微调;损失具体形式**沿用 Minitron 论文 §3**(本文未重列公式 → 标注:原文未在正文重列 KD 损失式)。本质是 \(\min_\theta \mathrm{KL}[p_{\text{teacher}}\|q_\theta^{\text{student}}]\),让剪小后的学生覆盖教师整个输出分布。数据配比消融(Table 11,~6B token KD,reasoning-SFT/pretraining):50/50→57.5、**70/30→58.5(最佳)**、90/10→57.2。

### 最终 9B 推理模型的 7 步分阶段恢复链(§4.3,带 token 数)
1. 深度剪到 56 层;KD ~60B token @ 8192 长度。
2. 宽度剪枝 + KD:~50B@8192、~25B@49152、~1B@262144(逐级增长序列长)。
3. DPO。
4. GRPO。
5. KD ~0.4B@262144(**恢复 post-RL 掉点**)。
6. RLHF(人类偏好对齐)。
7. **步骤 5 与 6 间模型合并**:0.5 线性插值。

### 多能力对齐的关键子机制(§3,逐组件必要性)
- **Stage-1 SFT**：全量数据 + 混约 10% 去推理轨迹的"空 trace"样本(支持 reasoning-off 直答);拼接成约 128k 长序列。
- **Stage-2 SFT(工具,不拼接)**：**修复 Stage-1 拼接对工具学习的破坏**。
- **Stage-3 SFT(截断训练)**：加入把推理轨迹**突然截断到 1-2k token**(保留最终答案)的样本——没它短 budget 下 well-formedness 暴跌(Fig.5a→5b)。
- **iterative on-policy DPO(工具)**：在 WorkBench 多步可验证工具环境,对每个 prompt 用**当前 checkpoint** 生成 on-policy **正样本(成功调用)/负样本(失败生成)** 迭代 DPO;BFCL v3 评测。
- **GRPO(RLHF,Qwen-based RM,HelpSteer3 英文)**：提升 instruction-following/对齐,生成带/不带 thinking 两种 rollout 评分;**暂时损害 MMLU-Pro**(post-GRPO KD 恢复)。
- **模型合并(checkpoint 插值)**：\((1-\alpha)\cdot w_{\text{model1}}+\alpha\cdot w_{\text{model2}}\),把推理强/chat 强两个 RL checkpoint 调和;\(\alpha\) 在 0.1-0.9 扫(步长 0.1),**\(\alpha\approx0.5\) 最佳折中**——缓解 RLHF 引入的回退(Fig.6)。

### budget 控制机制(§3.4,可复现细节)
推理时从生成 `<think>` 起计 token;到 budget 后**不立即插 `</think>`,而是让模型把当前句子写完、在下一个换行处插**;极端情况无换行则在 **(budget+500)** 处强插。两种失效模式:
- ① compensation(thinking 受限→在 final answer 里补)→ 截断训练消除;
- ② 强插后仍停在 thinking 模式(再吐一个 `</think>`)→ 用 "Well-Formedness"(只含单个闭合标签为良)度量,截断训练后短 budget 也稳定良形(Fig.5b)。
- **直觉一句话**:"forward-KL 拽精度、分支合并调能力、截断训练驯 budget。"

## 靠不靠谱
> 一句话导读:作为产线报告,吞吐/内存/budget 三目标都给了直接证据、消融到位;但完整训练代码闭源、外部难复现,且它的"蒸馏"是 off-policy KD,与本课题的 on-policy 范式相反。

- 实验与证据:
  - **吞吐(Fig.1/Table 5-6)**：Nemotron-Nano-9B-v2 在生成密集场景(8k 输入/16k 输出)吞吐比 Qwen3-8B 高 **3×-6×**;推理基准相当或更优。【原文 §4.4/Fig.1】
  - **12B 对齐模型(Table 8,reasoning ON)**：AIME-2024 85.42、AIME-2025 76.25、MATH-500 97.75、RULER@128k 83.36 等多项超 Qwen3-8B,但 SciCode/ArenaHard 略低(ArenaHard 74 < Qwen3-8B 78.4、Qwen3-14B 87.7)——chat 是相对弱项。
  - **逐阶段分析(Fig.6)**：DPO+GRPO 显著提升 BFCL v3/IFEval;GRPO 暂砸 MMLU-Pro→post-GRPO KD(步骤5)恢复;RLHF 涨 Arena-Hard 但引回退→模型合并(步骤7)恢复。
  - **截断训练(Fig.5)**：显著改善短 budget 下 well-formedness。
  - baseline 公平吗:与 Qwen3-8B 同场景对比吞吐/精度,工业报告标准;消融(Table 9 深度、Table 10 候选、Table 11 配比)到位。
  - "看着强但没回答核心问题":作为产线报告,吞吐/内存/budget 三目标都给了直接证据;但**完整训练代码不在发布仓内**,外部难独立复现。
- 假设与失效边界:
  - 【原文 §4.1/§4.2】剪枝**只剪 FFN+embedding+深度、不剪 Mamba head**——因本工作**压缩比小(12B→9B,深度剪后 <15%)**;【推断】激进压缩下该结论(及 forward-KL KD 的恢复力)未必外推。
  - 【原文 §3.4】thinking budget 控制依赖训练时截断样本 + 推理时 (budget+500) 强插 `</think>`——对极短 budget 仍有鲁棒性边界。
  - 【推断】**这是模型/数据发布而非可复现方法论**——完整训练代码(NeMo/Megatron 内部栈)闭源,超参/阶段多为工程经验选择,缺替代方案系统对照。
  - 【原文 §3.2/§4.3】模型合并固定 \(\alpha\approx0.5\)——为经验做法,虽扫了 0.1-0.9 但**缺合并方式(非线性/任务向量等)的系统对照**。
- 祛魅总结【推断】：
  - 真贡献：一条**完整、消融到位的工业级产线**(混合架构 + 推理模型压缩 + budget 控制 + 多分支对齐合并),且开源大量预/后训练数据与权重;Fig.6 把各对齐/恢复阶段作用拆开、Table 11 给数据配比,工程透明度高。
  - 与本课题(on-policy 蒸馏)的关系**被高估的风险**:
    - ① 它的"蒸馏"是 **forward-KL logit 蒸馏 + 固定教师 logits/数据**,本质是 **off-policy KD**(教师不在学生轨迹上打分),与 on-policy distillation **不是同一范式**;
    - ② 真正 on-policy 的只有"iterative on-policy DPO(工具)"那一支——且那是偏好优化不是蒸馏;
    - ③ reverse vs forward KL 的取舍、on-policy 蒸馏的分布真实性收益均未讨论。
    - 所以它对本课题更多是**背景/对照**(工业如何混用 SFT+RL+KD),而非方法直接对标。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：压缩阶段=教师(原 12B)的 logits 分布(forward KL,白盒);对齐阶段=SFT 监督 + DPO 偏好 + GRPO/RLHF 奖励(Qwen-based RM)。混合多信号。
  - **改什么**：改模型结构(剪枝:深度 62→56 层、FFN 20480→15680、embedding 5120→4480)+ 改参数(KD 恢复 + 多分支对齐 + 合并)。
  - **何时改**：分阶段——预训练→3 阶段 SFT→DPO/GRPO/RLHF 分支→剪枝→分阶段 KD(增长序列长)→post-RL KD→RLHF→合并。
  - **免梯度?**：否(全程梯度训练);压缩用 forward-KL 监督蒸馏,对齐含 RL(GRPO/DPO/RLHF)。
  - **记忆-技能生命周期**：无外部记忆/技能库;"能力"=推理/工具/对齐,靠多分支训练注入参数,用合并保留;budget 控制是一种"行为技能"(训练时截断样本习得)。
  - **防遗忘机制**：**模型合并(0.5 线性插值)** 是其防"对齐回退/能力相互覆盖"的主要手段;post-RL KD(步骤5)恢复 RL 造成的退步;无针对持续学习的显式防遗忘。
- ⑦ 开源代码+框架/harness：**无完整训练代码**(属模型/数据发布,CloneTier=B,本地未 clone)。权重 HF nvidia(Nemotron-Nano-9B-v2、12B-v2-Base、9B-v2-Base);数据 Nemotron-Post-Training-Dataset-v1、Nemotron-Personas、Nemotron-Pretraining-Code/SFT-v1、Aegis Content-Safety v2 等多数开源;训练栈 **NVIDIA NeMo / Megatron**(FP8 预训练);评测用 NeMo-Skills、lm-evaluation-harness、math-verify。**未完整克隆——见 CLAUDE.md 决策①(大厂技术报告仅记链接不 clone 巨型权重仓);需手动从 HF 获取权重/数据。** 【原文 §1.1 发布物清单 + v1 核】
- 💰 资源/成本与可扩展性：预训练 20T tokens(FP8);后训练总计约 90B tokens;压缩 KD 总量约 60B+50B+25B+1B+0.4B(推理模型)/120B+360B+2.5B(base);目标硬件单 A10G(22GiB, bf16)跑 128k。响应多由 DeepSeek-R1-0528 与 Qwen3-235B-A22B 生成(蒸馏式数据合成)。【原文 §2/§3/§4.3】
- 🎯 对"探索-巩固"对标：**弱-中支撑(仅工业混用 SFT+RL+KD 的背景对照 + on-policy DPO 一支);非 on-policy 蒸馏方法**。判定依据：
  - ① **唯一直接相关点是 iterative on-policy DPO**——在 WorkBench 多步工具环境用**当前 checkpoint** 生成正/负样本迭代 DPO,与本课题"student on-policy 自选轨迹 + 从成功/失败中学"在**数据来源上**同构(on-policy 正=成功调用、负=失败生成),弱对应"探索(尝试工具调用)+巩固(把成功固化)";
  - ② **模型合并防能力回退** 提供一个朴素"巩固不遗忘"工程范式;
  - ③ **截断训练教模型在 budget 内收尾** 与 MTP"前瞻/在有限步内规划"弱相关。
  - **缺口**:① **压缩蒸馏是 forward-KL off-policy KD**(固定教师 logits),**不是学生轨迹上的 on-policy 蒸馏**——与本课题核心范式相反;② **无 teacher 稀疏脚手架、无 path-recovery 单点接管、无 MTP 前瞻探针**;③ 是产线整合而非机制创新。
  - 一句话:**它是"工业如何把 SFT/RL/KD/合并拼成产线"的背景样本,其 on-policy DPO 一支可作"on-policy 正负样本"对照,但整体非 OPD 范式、无脚手架/前瞻可借。**
- 🔭 开放问题/未来方向：
  - 【原文】把 budget 控制做到更短预算仍鲁棒;更系统地选合并系数/方式;把压缩-对齐链推到更激进压缩比。
  - 【推断】把压缩阶段的 forward-KL **off-policy** KD 换成"学生剪枝后在自身轨迹上的 on-policy 蒸馏",看能否更省 token 地恢复精度(把本报告与 opd_blog/opd_survey 的 on-policy 范式缝合);把 on-policy DPO(工具)的"成功/失败正负样本"扩展为"走偏后自选恢复分支"的 path-recovery 数据。

RETURN: nemotron_nano2|读到PDF=是(§Abstract/§1-4全文+§3.4截断1-2k与budget+500/§4.1剪枝重要性MSE+FFN神经元式/§4.3 forward-KL KD步骤1-7/Table9-11/Fig.5-6/§3.3 on-policy DPO WorkBench/Table8)|L线=L1(兼L3/L4)|对标=弱-中支撑(仅on-policy DPO一支同构"on-policy正负样本"+模型合并防回退作背景;但压缩蒸馏是forward-KL off-policy KD非OPD范式,无teacher脚手架/回轨/MTP)|残留待核=0(未完整克隆:大厂模型/数据发布仓,无训练代码,见决策①,需HF手动获取;forward-KL KD损失式正文未重列,沿用Minitron论文§3)

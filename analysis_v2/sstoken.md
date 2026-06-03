sstoken | ssToken: Self-modulated and Semantic-aware Token Selection for LLM Fine-tuning | 上海交大 + 上海创智学院(Xiaohan Qin, Xiaoxing Wang…通讯 Junchi Yan) | arXiv 2510.18250 v1(2025-10-21, cs.AI)·ICLR 2026 | 主题线 L2(统一 SFT 视角的 token 级数据选择,偏 SFT 侧)·相关性 中

**原始论文**:https://arxiv.org/abs/2510.18250

## 一眼看懂
- 🟦 TL;DR:SFT 里"不是所有 token 都值得学"。现有 token 选择(RHO-1/TokenCleaning)有两个痛点:要额外训/借一个 reference model,且只看 loss。ssToken 两招:① 用"当前模型 vs 自己历史 checkpoint"的 loss 下降量(Retrospective Excess Loss, REL)当自调制信号,免掉 reference model;② 加一个基于注意力的"语义重要性"分(response token 对 prompt 的注意力总和)。两分归一后线性融合(γ=0.5),取 top-ρ token 算 loss、其余 mask。四个基座(3B~14B)平均分都最优。【原文 §3, Table 1】
- 最巧的一步:把 RHO-1 的 Excess Loss(对外部 reference)翻成 REL(对自己历史)——`REL=L_θhis−L_θ=log(P_θ/P_θhis)`【原文 Eq.3】。抽掉它就退回"需要 reference model"的老范式,卖点(免 reference)直接没了。语义分(注意力)是第二根支柱,抽掉它就退化成纯 loss 选择,QA 类任务增益大幅缩水。两根都要才有协同。【原文 §4.3, Fig.3a】

## 为什么做
- 研究背景:SFT 阶段"数据质量 > 数量"已成共识;但即便做过样本级清洗,高质量数据里仍有 **token 级噪声**(无关/冗余片段)。RHO-1 首次做 token 级选择并大幅超 full-data;TokenCleaning 在 SFT 场景进一步优化(fixed-model / self-evolving)。【原文 §1-2】
- 解决的具体痛点:现有 token 选择两大局限——(1) **需训练或访问额外 reference model**:直接借更强同 tokenizer 模型不总可行,单独训 reference 增成本,且 reference 质量显著影响选择效果(引 Pang 2025);(2) **只靠 loss 信息**:token loss 反映预测不确定性,不必然反映语境中的语义重要性;频繁但语义无信息的 token 可能与任务关键 token 有相近 excess loss,loss-only 易误删有信息内容。【原文 §1, lines 48-51 / 122-127】
- 相关工作 & 各自不足:RHO-1=token 级选择开山,但需 reference + 只看 loss;TokenCleaning=SFT 场景两种清洗策略,同样需 reference;样本级选择(DS²/Superfiltering 等)粒度太粗。【原文 §2】
- 动机链:数据质量决定 SFT → 样本级清洗后仍有 token 级噪声 → token 级选择有效但依赖 reference + 只看 loss → 用"自己的历史模型"当天然 teacher(免 reference)+ 用注意力补语义信号 → ssToken。
- 与最近邻工作的Δ:vs RHO-1/TokenCleaning,两点关键差异:① reference model → history model(EMA 可更新,提供更稳的长程指引,且零额外训练);② 单 loss 信号 → loss + 注意力语义双信号。差在"免 reference + 正交语义轴",有用是因为两信号互补、QA 任务上语义轴带来额外增益。【原文 §1 contributions, §4.2】

## 怎么做 + 靠不靠谱
- 方法流水线:输入一个 SFT 样本(prompt+response)→ ① 对每个 response token 算 REL(当前模型 loss − 历史模型 loss)→ ② forward 时用 hook 取目标层注意力,算该 token 对 prompt 的注意力总和 AttnScore → ③ REL 在样本内 min-max 归一到[0,1],AttnScore 天然∈[0,1] → ④ `Score=γ·Norm(REL)+(1−γ)·AttnScore`(γ=0.5)→ ⑤ 取 top-ρ token,只对它们算 NLL loss、其余 mask → 输出梯度。【原文 §3.2-3.4, Eq.3/7/8/9/10】
- 逐组件必要性:
  - **REL(自调制)**:负责"免 reference 地找可学 token"。没它就退回需 reference 的老路。消融 γ=1(纯 REL)已超 full-data。【原文 Fig.3a】
  - **AttnScore(语义感知)**:负责补 loss 看不到的语义/指令相关性。没它(γ=1)在 QA 任务上增益缩小;消融 γ=0(纯注意力)也超 full-data。【原文 §4.2, Fig.3a】
  - **融合 γ**:有消融(γ∈{0,0.25,0.5,0.75,1}),中间值最好、两极(纯一个信号)都次之 → 证明两信号互补且都必要。【原文 §4.3, Fig.3a】
  - **比例 ρ**:有消融(ρ∈{0,0.2,…,1}),3B/8B 峰值在 0.6,14B 峰值在 0.8(连同 Random/RHO-1/TokenCleaning 也都在 0.8 达峰)。【原文 §4.3 + lines 853-855】
  - **层选择**:Appendix B 消融早/中/深层,结论"深层更好"(语义抽象随层加深)——属经验消融。【原文 §3.3, lines 326-333】
  - **EMA 更新历史模型**:标注为"(Optional)",非必需;基线可直接用 SFT 前的 base 当历史模型。【原文 Eq.4】
- 关键机制/公式(直觉):REL 与 RHO-1 的 Excess Loss 方向相反——EL 学"未来 loss"(对一个更强 reference),REL 学"历史 loss"(对自己更弱的过去);若当前模型相对历史在某 token 上 loss 大降,该 token 更可能"可学且有信息"。AttnScore 直觉:SFT 里所有 response token 都注意固定长 prompt,而 prompt 编码任务/指令,故 response token 对 prompt 的注意力强度=任务相关性代理;用因果 mask 下"别的 token 对目标 token"的注意力会引入位置偏置,所以专取"response→prompt"方向。【原文 §3.2-3.3】
- 实验与证据:
  - 数据池:从 5 个常用 SFT 集(Flan v2 / OpenAssistant / Stanford Alpaca / Dolly / WizardLM,共 300k)采 50k;reference 基线在 DS² 筛出的 10k 高质子集上训。【原文 §4.1, lines 388-408】
  - 评测 10 个通用基准:TriviaQA / TruthfulQA / MMLU / ARC-C / ARC-E / TyDiQA / Winogrande / HellaSwag / LogiQA / AGIEval,用 lm-eval-harness。基座 4 个:LLaMA-3.2-3B、LLaMA-3.1-8B、Qwen-2.5-7B、Qwen-2.5-14B。【原文 Table 1】
  - 关键数字:ssToken 四基座平均分均最优,相对 full-data +4.3% / +3.4% / +1.3% / +2.1%(3B/8B/7B/14B);相对 prior token 方法最高 +2.8%【原文 §1, lines 165-167, 735-737】。增益集中在 TyDiQA / TriviaQA / AGIEval 等需指令遵循的 QA(归功于注意力分量);MMLU/ARC 等知识密集任务 token 选择基本无提升【原文 lines 743-755】。RHO-1/TokenCleaning 在 Qwen 系上仅与 full-data 持平甚至更差,而 ssToken 跨族稳定【原文 lines 737-742】。
  - baseline 公平吗:同一基座下所有方法用相同 ρ;reference 基线给了 10k 高质子集训 reference——公平。一个隐性不平:ssToken 多算注意力(虽轻量),且 REL 需维护历史模型副本(显存/状态成本未量化)。
  - "看着强但没回答核心":不省训练时间——被 mask 的 token 仍走 forward,故 token 选择类方法都不减训练时间(作者诚实承认,Fig.2)【原文 lines 798-804】;卖点是"省掉 reference 训练时间"。
- 假设与失效边界:
  - 【原文】训练早期 history=current,REL 近似随机(作者明说,lines 266-268)。
  - 【原文】只在通用指令 SFT 验证,未触及长 CoT 推理蒸馏。
  - 【推断】AttnScore 只取"response→prompt"总注意力,长 prompt / 多轮 / 工具调用场景的有效性未验证;层选择(深层更好)依赖经验消融,跨架构未必稳。
  - 【推断】"无 reference"非完全免费:EMA 历史模型需额外参数副本,显存成本论文未充分量化。
- 祛魅总结【推断】:真贡献=两个工程性改进(REL 替 reference + 注意力语义轴),组合后跨四基座稳定小幅提升、且免掉 reference 训练。包装上"自调制/语义感知"听着大,本质仍是 RHO-1 式"top-ρ token + loss mask"范式的增量补丁。高估:把它当"普适显著提升"——Qwen-7B 仅 +1.3%、单项(TruthfulQA)有时反低于 BASE/FULL。低估:免 reference 这点在工程上很实用,降低了 token 选择的落地门槛。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=token 级 REL(自己 vs 历史的 loss 下降)+ response→prompt 注意力强度 | **改什么**=哪些 token 进 SFT loss(top-ρ 选择 + mask) | **何时改**=每个训练 step 在线评估(REL 依赖当前/历史 checkpoint) | **免梯度?**=否,被选 token 走标准 NLL 梯度;选择本身免梯度(基于 loss/注意力打分) | **记忆-技能生命周期**=不涉及(纯单轮 SFT 数据加权) | **防遗忘机制**=无(单阶段 SFT)
- ⑦ 开源代码+框架/harness:https://github.com/jianke0604/ssToken(已克隆 ~2.8MB,代码完整可跑);框架=**自写训练脚本 + FSDP**(支持 LoRA)+ hook 重算注意力兼容 FlashAttention + EleutherAI lm-evaluation-harness 评测。关键文件 `scripts/{calculate_token_loss.py, finetune_with_hook.py, generate_token_label.py, finetune.py}`,代码中 `combined=ratio·diff_norm+(1−ratio)·resp2prompt_scores`(ratio=γ)、`data_prop`=ρ,与论文 Eq.9/10 一致。
- 💰 资源/成本与可扩展性:训练用 3B~14B 模型;**不减训练时间**(mask token 仍走 forward),卖点是省掉 reference 训练;注意力用 hook 重算"目标层",引入边际开销;EMA 历史模型副本的显存成本原文未说明。【原文 §4.2 + lines 798-812】
- 🎯 对"探索-巩固"对标:**可借组件(弱)**。一句判定:ssToken 是 SFT 侧的 token 选择,与本项目"teacher 稀疏脚手架 + on-policy 自选路径"不在一个范式,但其"REL=用自己历史当天然 teacher"思路与"探索阶段偏向自己能走通的开头"有微弱同构,且"注意力定位语义关键 token"可作"切关键步"的候选信号之一。依据:它是 off-policy SFT 数据加权,不含 on-policy rollout、不含 teacher 监督、不含记忆/技能,故只能作边缘借鉴而非竞品。【推断,依据范式差异】
- 🔭 开放问题/未来方向:【原文】ρ 非普适、依赖手调,作者建议把 ρ 的优化集成进算法、按训练进度/容量/数据质量自适应(lines 875-883)。【推断】把 REL+注意力双信号迁移到长 CoT / on-policy 蒸馏的 token 选择(与 TIP 的"熵×散度"两轴可对照);验证注意力语义轴在多轮/工具调用场景是否还成立;量化 EMA 历史模型的真实显存成本。

〔本篇与既有 analysis/sstoken.md 核对:v1 footnote 称"14B 用 0.8 是误读"——经 PDF §4.2(lines 731-732)与 §4.3(lines 853-855)核实,**14B 确实用 ρ=0.8**(且同基座下所有对照方法亦在 0.8 达峰),v1 footnote 有误,本篇已更正。〕

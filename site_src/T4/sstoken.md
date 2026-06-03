# sstoken — ssToken: Self-modulated and Semantic-aware Token Selection for LLM Fine-tuning

> **一句话重点 (TL;DR)**：SFT 的 token 级选择方法，用"当前模型 vs 其历史模型"的 Retrospective Excess Loss（REL）替代外部参考模型，再融合一个基于注意力的语义重要性分，按比例 ρ 保留 top-ρ token 计 loss；四基座平均分最优，但增量温和、主要在需指令遵循的 QA 任务见效。

**元信息**：arXiv 2510.18250（v1 2025-10-21）｜ 上海交大 & 上海创智学院（Xiaohan Qin, Xiaoxing Wang 等，通讯 Junchi Yan）｜ ICLR 2026 ｜ 主题 SFT token 级数据选择 / 相关性 中（与"不是所有 token 都值得学 / OPD token 选择"相关，但场景是通用指令微调而非推理蒸馏；卖点是"无需参考模型 + 注意力语义信号"）｜ 代码 https://github.com/jianke0604/ssToken （已克隆 ~2.8MB，可跑）｜ 框架 自写训练脚本 + FSDP（支持 LoRA）+ lm-evaluation-harness 评测

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/sstoken/fig_02.png)

*Figure 2: Average performance vs. total training time across different methods.*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/sstoken/fig_01.png)


## 1. 相关工作与进展
SFT 中"数据质量 > 数量"已成共识；即便做过样本级过滤，高质量数据仍含 **token 级噪声**（与任务无关的冗余/无信息片段）。Rho-1 首次提出 token 级选择并显著超 full-data；TokenCleaning 在 SFT 场景进一步优化（fixed-model / self-evolving 两种清洗）。本文沿用 TokenCleaning 的数据准备与 DS²-50k 数据池。

## 2. 现有工作存在的问题
现有 token 级选择（Rho-1、TokenCleaning）两大局限：(1) **需训练或访问额外参考模型**——直接用更强同 tokenizer 模型不总可行，单独训 reference 增成本，且 reference 质量显著影响选择效果；(2) **仅依赖 loss 信息**——token loss 反映预测不确定性，不必然反映语境中的语义重要性，频繁但语义无信息的 token 可能与任务关键 token 有相近 excess loss，loss-only 易误删有信息内容。

## 3. Motivation
(1) 把"当前模型自身"当天然 teacher：训练推进中，当前模型相对其历史状态的进步本身就是可靠选择信号，由此摆脱外部 reference。(2) 注意力矩阵天然编码语义，可作 loss 之外的正交补充信号。

## 4. 主要灵感 / 核心直觉
若某 token 相对历史模型 loss 显著下降，则它更可能是"可学的、有信息的"而非噪声/已掌握；而 response token 对 prompt 的注意力强度，可代理其"任务相关性/指令遵循重要性"。两信号正交，融合产生协同增益。

## 5. 主要解决思路(一段话讲清核心)
对每个 response token，算"REL（相对历史模型的 loss 下降）"与"对 prompt 的注意力分"，归一化后线性融合为 Score，按固定比例 ρ 选 top-ρ token 计 loss、其余 mask，做带 mask 的 SFT。

## 6. 方法详解(通俗、分步骤)

- **Self-modulated（自调制）选择**：**Retrospective Excess Loss (REL)** = \(L_{\theta_{\mathrm{his}}}(x_i) - L_\theta(x_i) = \log\!\left[ P_\theta / P_{\theta_{\mathrm{his}}} \right]\)（论文式(3)），即当前模型相对历史模型的 loss 下降（与 Rho-1 的 Excess Loss"学未来 loss"相对，REL"学历史 loss"）。历史模型可由 EMA 自适应更新（式(4)：\(\theta_{\mathrm{his}} = \alpha \cdot \theta_{\mathrm{his}} + (1 - \alpha) \cdot \theta\)，可选），比固定 reference 提供更稳长程指引。
- **Semantic-aware（语义感知）选择**：基于注意力的 token 重要性。利用 SFT 中所有 response token 都关注固定长度 prompt 这点，计算每个 response token 对 prompt token 的注意力之和（多头平均）作为相关性代理；用深层（deeper layer）注意力效果更好；用 hook 重算目标层注意力以兼容 FlashAttention。
- **融合**：REL 在样本内 min-max 归一到 [0,1]，注意力分天然 ∈[0,1]；最终 \(\text{Score} = \gamma \cdot \mathrm{Norm}(\text{REL}) + (1 - \gamma) \cdot \text{AttnScore}\)（默认 γ=0.5）。代码 `scripts/finetune.py`：\(\text{diff\_norm} = \frac{\text{diff} - \text{diff.min}()}{\text{diff.max}() - \text{diff.min}() + 10^{-8}}\)、\(\text{combined} = \text{ratio} \cdot \text{diff\_norm} + (1 - \text{ratio}) \cdot \text{resp2prompt\_scores}\)（与论文 Score 一致 ✓，`ratio`=γ）。按固定比例 ρ（默认 0.6）选 top-ρ token 计 loss，其余 mask（`data_prop`=ρ）。

## 7. 实验数据集
数据池：从 5 个常用 SFT 集（Flan v2、OpenAssistant、Stanford Alpaca、Dolly、WizardLM，共 300k）采 50k（DS²-50k）；reference 基线在 DS² 样本级筛出的 10k 高质子集上训。评测 10 个通用基准：TriviaQA、TruthfulQA、MMLU、ARC-C/E、TyDiQA、Winogrande、HellaSwag、LogiQA、AGIEval。基座：LLaMA-3.2-3B、LLaMA-3.1-8B、Qwen-2.5-7B、Qwen-2.5-14B（3B~14B）。

## 8. 实验结果与主要发现

- 四基座上 ssToken 平均分均最优，相对 full-data 提升 4.3% / 3.4% / 1.3% / 2.1%（3B/8B/7B/14B），相对 prior token 选择方法最高 +2.8%。
- TyDiQA、TriviaQA、AGIEval 等需指令遵循的 QA 任务增益最明显（归功于注意力分量）；MMLU/ARC 等知识密集任务 token 选择基本无提升。
- Rho-1/TokenCleaning 在 Qwen 系上仅与 full-data 持平甚至更差，而 ssToken 跨族稳定。

## 9. 结果如何支撑其主张
跨四基座一致最优 + 两信号独立消融均超 full-data，支撑"无 reference + 注意力语义"双改进有效。但单族增益不均衡（Qwen-7B 仅 +1.3%），削弱"普适提升"的强主张。

## 10. 逻辑自洽性(中性评估)
方法自洽：REL 与注意力分两正交信号 + 融合 + top-ρ mask。代码与论文 Score 公式、ρ=0.6 一致。注意：〔原稿"14B 用 0.8"为误读，论文中 ρ=0.8 是对照方法（Random/RHO-1/TokenCleaning）达各自峰值的比例（Appendix），非 ssToken 在 14B 的设定；已核实更正——论文明确 ρ=0.6 一般有效，同基座下各方法用相同 ρ 比较。〕

## 11. 残留问题 / 局限

- 增量温和：主体仍是 Rho-1 式"top-ρ token + loss mask"范式，创新在"REL 替换 reference"与"注意力语义分"两个工程性改进。
- "无 reference"非完全免费：训练早期 history=current 使 REL 近似随机；EMA 历史模型需维护额外参数副本（显存/状态成本未充分量化）。
- 注意力分仅取"response→prompt"总注意力，长 prompt / 多轮场景有效性未验证；层选择（deeper better）依赖经验消融。
- 仅通用指令 SFT、未触及长 CoT 推理蒸馏，对本项目推理场景可迁移性需另证。
- 增益不均衡：Qwen-7B 相对 full-data 仅 +1.3%，部分单项（如 TruthfulQA）反低于 BASE/FULL。

## 12. 开源代码与框架(链接+框架+代码可得性)

- https://github.com/jianke0604/ssToken （已克隆 ~2.8MB，含 `scripts/` 下 `calculate_token_loss.py`、`finetune_with_hook.py`、`generate_token_label.py`、`finetune.py`，及 bash_src、fsdp_configs、eval；代码完整可跑）。
- 框架：自写训练脚本（finetune_trainer.py / finetune_with_hook.py），用 FSDP 配置、支持 LoRA；注意力重算用 hook 兼容 FlashAttention；评测用 EleutherAI lm-evaluation-harness。
- 流程：算 token loss / REL（calculate_token_loss.py）+ 注意力分（finetune_with_hook.py 重算目标层）→ 融合打分选 top-ρ → 带 mask 的 SFT。默认 γ=0.5（run.sh/finetune.sh `ratio`）、ρ=0.6（eval_tydiqa.sh `data_prop`）；同基座下所有方法用相同 ρ。

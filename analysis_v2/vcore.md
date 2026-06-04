vcore | VCORE: Variance-Controlled Optimization-based Reweighting for Chain-of-Thought Supervision | 上海交通大学 & 香港中文大学（深圳）（Xuan Gong、Senmiao Wang、Hanbo Huang、Ruoyu Sun、Shiyu Liang 通讯） | 2026-04-18·arXiv 2510.27462·v2（cs.CL，ACL ARR） | 主题线 L2（统一 SFT-RL / token 加权）+ L6（CoT/token 信用）·相关性 中

**原始论文**：https://arxiv.org/abs/2510.27462 （代码 https://github.com/coder-gx/VCORE ）

## 一眼看懂
- 🟦 TL;DR：长 CoT SFT（在 teacher 蒸馏的长推理迹上做监督微调）里，标准交叉熵对所有 token 等权——但很多 token 要么过易要么过歧义、学习价值低，自动蒸馏的长 CoT（动辄 >1k token）还常含幻觉/错位的 spurious token，等权让噪声主导梯度。VCORE 把 token 加权形式化为一个约束优化：**在单步 SGD 下使期望 loss 下降最大、且加权分布 \(q\) 与均匀分布 \(u\) 的 \(\mathrm{KL}(q\|u)\le\delta\)**，闭式解是 Gibbs 分布 \(q^*(t)\propto\exp(\tau\,s_t)\)（\(s_t\)=token 梯度效用），再配一个 one-backward 探针估 \(s_t\) + 方差控制系数 \(\alpha\) 稳训练。不依赖 teacher 引导/置信阈值/熵过滤，在综合均值与中小模型上稳定优于 SFT/DFT/iw-SFT【原文 Abstract、§3-4】。
- 最巧的一步：**one-backward forward-mode probing trick**。抽掉它方法就不可行——梯度效用 \(s_t=\langle\nabla L,\nabla\ell_t\rangle\) 朴素估计需每个 token 一次 backward（长序列上算不动）。VCORE 另抽一个 mini-batch \(B'\) 算均匀权下降方向 \(\nabla L_{B'}\)，沿该方向做小扰动 \(\epsilon\) 测每 token loss 变化，其极限 = \(s_t\)（无偏），**仅 1 次 backward + 1 次 forward 就覆盖全部 token，无二阶梯度/hook**【原文 §4.1、line 367-395】。

## 为什么做
- 研究背景：长 CoT SFT（从 teacher 蒸馏推理迹）是增强 LLM 推理的轻量有效路线，常作后续 RL 的初始化；但多数工作重数据配方与工程，**优化算法层留白**【原文 §1】。
- 解决的具体痛点：均匀加权两大缺陷——
  1. 并非所有 token 都值得学，过易/过歧义的 next-token 预测梯度价值低、浪费更新拖慢收敛；
  2. 自动蒸馏的长 CoT 含幻觉/错位 spurious token，均匀加权让噪声主导梯度、损害泛化。
  已有 reweighting（DFT/iw-SFT）依赖 \(1/\pi_\theta\) 或 RL 目标下界的重要性加权，而非直接从训练时梯度信息出发【原文 §1-2】。
- 相关工作 & 各自不足（来龙去脉/并行路线/精确差异）：
  - **① CoT prompting / 长 CoT SFT 路线**（Wei 2022；Team 2025、Muennighoff 2025、Xu 2025 等）——从更强 teacher 蒸推理迹，操作轻量、泛化强、常作 RL 初始化（Yeo 2025）；**短板：均匀加权，优化层留白**。
  - **② DFT（Wu 2025，最近邻 A）**——把标准 SFT 重解读为含隐式 \(1/\pi_\theta\) 重要性因子的 policy-gradient，**过度加权低概率 token**，故乘 token 概率 \(\pi_\theta(y_t)\) 纠正。**RL-motivated**。
  - **③ iw-SFT（Qin & Springenberg 2025，最近邻 B）**——证 curated/filtered 数据上 SFT 优化某 RL 目标下界，按相对参考/当前策略的重要性加权收紧界。**RL-motivated**。
  - **④ 其它启发式**（Lin 2024 按估计的对终答正确性影响排序；置信/熵过滤）——**不利用训练时梯度信息**。
  - **本文精确Δ**：VCORE 同为 SFT 阶段 token reweighting，但**optimization-driven**（非 RL-motivated）——从单步 SGD 一阶下降动力学直接导权重（token 价值=其梯度与全局下降方向的对齐度 \(s_t=\langle\nabla L,\nabla\ell_t\rangle\)），并显式加方差控制。为什么有用：综合均值 VCORE 31.03 > DFT 29.99 > iw-SFT 28.35，且作为 RL 初始化 RL 后更强（Obs 7）【原文 §1-2、§3、line 169-183】。
- 动机链：长 CoT SFT 优化层留白→均匀加权浪费易 token + 被 spurious token 噪声主导→能否用优化驱动而非启发式给 token 加权→把加权写成"单步 SGD 下降最大 + \(\mathrm{KL}(q\|u)\le\delta\)"约束优化→闭式 Gibbs 解→朴素估 utility 太贵→one-backward 探针→Gibbs 重加权改变更新方差→方差控制 \(\alpha\) 对齐【原文 §3-4】。

## 怎么做 + 靠不靠谱
- 方法流水线（Algorithm 1，对每条轨迹/每 batch，输入→输出）：
  1. 一阶 Taylor 展开期望 loss 下降，定义 token 梯度效用 \(s_t=\langle\nabla L,\nabla\ell_t\rangle\)；
  2. 在单纯形 \(\Delta\) 上 \(\max_q\sum_t q(t)s_t\) s.t. \(\mathrm{KL}(q\|u)\le\delta\)，闭式解 Gibbs \(q^*(t)\propto\exp(\tau s_t)\)（\(\tau\to0\) 退均匀、\(\tau\to\infty\) 集中最高效用）；
  3. 另抽 mini-batch \(B'\)（\(|B'|=32\)）用 one-backward forward-mode 探针无偏估 \(s_t\)；
  4. 乘方差控制系数 \(\alpha=\sqrt{V_u/V_q}\) 把重加权更新方差对齐到均匀加权；
  5. 更新 \(\theta\leftarrow\theta-\eta\,\mathbb E_{(x,y),\,t\sim q^*}[\nabla_\theta(\alpha\,\ell_t)]\)【原文 §4.1-4.2、Algorithm 1、line 441-468】。
- 逐组件必要性（每组件 + 证据/默认值）：
  - **Gibbs 最优加权（§4.1）**：核心信号。由约束优化闭式导出（非启发式），消融把增益拆给它。温度 \(\tau\) 默认按约束设（实验 \(\lg\tau\approx4\) 即 \(\tau\approx10^4\) 附近最优，Fig.2b）【原文 §4.1】。
  - **one-backward 探针**：效率使能组件。无它则 \(s_t\) 估计需逐 token backward 不可行；代价是每步额外一个 batch \(B'\) 的一次前/后向（实际每步成本约翻倍，**论文未给端到端 wall-clock 对比**）。探测尺度 \(\epsilon\) 默认 \(\approx10^{-4}\)（Fig.2b，对 \(\tau/\epsilon\) 鲁棒）【原文 §4.1，推断：成本翻倍无计时】。
  - **Variance-Controlled 缩放 \(\alpha\)（§4.2）**：必要稳定器。**Fig.3 消融：无 scaling 时 loss 频繁尖峰（峰值 >20）、有 scaling 平滑收敛（~0.4-0.55）**（line 1099-1100 明确"variance control is not optional but essential"）。\(q\) 越尖/序列越长 \(\alpha\) 越小以稳训练，\(q\) 平衡时 \(\alpha\approx1\)【原文 §4.2、Fig.3】。
- 关键机制/公式（真实符号 + 直觉，从 PDF 抄准）：
  - **朴素 CoT 监督（均匀加权基线）**：mini-batch 梯度 \(\nabla_\theta\hat L_B(\theta)=\sum_{(x,y)\in B}\sum_{t\ge1}(1/|y|)\nabla_\theta\ell_t(\theta;x,y)\)，\(\ell_t=-\log p_\theta(y_t\mid x,y_{<t})\)。均匀权 \(1/|y|\) 是 population 梯度的无偏估计，故为自然默认【原文 §3】。
  - **自适应加权问题（式1，约束优化）**：\[ \min_q\ L(\theta^+(q))\quad\text{s.t.}\quad \mathrm{KL}\big(q(\cdot\mid x,y,\theta)\,\|\,u\big)\le\delta,\qquad u(t)=1/|y|. \]
  - **token 梯度效用（式2）**：一阶 Taylor \(L(\theta^+)-L(\theta)=-\eta\sum_{(x,y)\in B}\sum_t q_t s_t+O(\eta^2)\)，其中 \[ s_t(x,y,\theta)\triangleq\big\langle\nabla_\theta L(\theta),\ \nabla_\theta\ell_t(\theta;x,y)\big\rangle. \] 直觉：\(s_t\) 是"该 token 梯度与全局下降方向的对齐度"——对齐高的 token 最能降总 loss、应优先。
  - **Gibbs 闭式解**：\[ q^*(t\mid x,y,\theta)=\frac{\exp(\tau s_t(x,y,\theta))}{\sum_{j\ge1}\exp(\tau s_j(x,y,\theta))}, \] \(\tau>0\) 由约束 \(\delta\) 定（标准 exponential tilting 的唯一闭式解）。
  - **one-backward 无偏估计（核心 trick）**：抽 \(B'\sim P\)，沿均匀权下降方向 \(\nabla_\theta L_{B'}(\theta;u)\) 做小扰动 \[ \lim_{\epsilon\to0}\frac{\mathbb E_{B'}\big[\ell_t(\theta;x,y)-\ell_t(\theta-\epsilon\nabla_\theta L_{B'}(\theta;u);x,y)\big]}{\epsilon}=\big\langle\nabla_\theta L(\theta),\nabla_\theta\ell_t(\theta;x,y)\big\rangle=s_t. \] 把估全部 \(s_t\) 的成本从 \(|y|\) 次 backward 降到 1 次 backward（算 \(B'\) 下降方向）+ 1 次 forward（评扰动后 token loss），无二阶梯度/hook/额外模型查询。
  - **方差控制（式见 §4.2）**：定义 \(V_q\triangleq\mathrm{Var}[\sum_t q_t s_t]\)，\(V_u\triangleq\mathrm{Var}[\sum_t s_t/|y|]\)，选 \(\alpha\) 使 \(\alpha^2 V_q\approx V_u\Rightarrow\alpha=\sqrt{V_u/V_q}\)。直觉：若 \(s_t\) 不相关方差 \(\sigma^2\)，则 \(V_u\approx\sigma^2/|y|\)，尖峰 Gibbs 下 \(V_q\approx\sigma^2\Rightarrow\alpha=1/\sqrt{|y|}\)——\(q\) 尖/序列长则 \(\alpha\) 必缩小稳训练，\(q\) 平衡则 \(\alpha\approx1\) 恢复全步长【原文 §4.1-4.2】。
- 实验与证据：
  - 数据集/设置：训练（仅留 DeepSeek-R1 生成、过滤正确性的 CoT）——数学 OpenMathReasoning、代码 OpenCodeReasoning 的 C++ 子集；Qwen3 每域 3.2k（math CoT 均长 3155.01、code 2861.25），LLaMA 每域 32k（math 3007.79、code 2805.43）。评测 In-domain：AIME(24+25)、OlympiadBench-math、LiveCodeBench v6、OJBench；Out-of-domain：R-Bench-T、SuperGPQA-1k。基座 Qwen3-{4,8,32}B、LLaMA-3.1-8B-Instruct（弱模型补充 Qwen3-1.7B、Mistral-7B-Instruct-v0.3）。指标 Pass@1（greedy，max_gen 8192），vLLM v1。**全程 LoRA**（Qwen3 rank8/lr2e-5、LLaMA rank64/lr2e-4，alpha=2×rank，1 epoch，AdamW + cosine），硬件 4×RTX PRO 6000 Blackwell，batch=32【原文 §5、line 874-890】。
  - 关键数字：主结果（Table 1）平均分 **VCORE 31.03 > DFT 29.99 > SFT 28.97 > Random 28.78 > iw-SFT 28.35**。中小模型增益更明显：**LLaMA-3.1-8B ID/OOD 从 DFT 5.54/7.17 升到 11.38/9.59**；Qwen3-4B 从 32.49/32.49 升到 36.09/32.87。**作为 RL 初始化（Obs 7，Table 4）**：Qwen3-4B/8B 用 VCORE 初始化后 GRPO 200 步（BigMath 子集），RL 后均超 DFT（即便 RL 前略低；作者归因 DFT 降生成熵→限 RL 探索）。**简单任务退化（Obs 8，Table 5）**：长 CoT SFT 在 GSM8K（短链）略降（94.16 vs Original 95.15）、在 MATH500（深链）升（92.8 vs 90.4）【原文 Table 1/4/5、line 893-1213】。
  - baseline 公平吗：DFT/iw-SFT/SFT/Random（Random=随机丢 80% CoT token）在同训练集/同基座/同 LoRA 设置比较，公平。
  - 有无"看着强但没回答核心问题"：**并非全面碾压**——Qwen3-8B 上 VCORE OOD-avg 35.26 = DFT…实为 ID 35.80 < DFT…细看 Qwen3-8B VCORE ID 35.80/OOD 35.26 vs DFT 35.55/37.29（**OOD 输 DFT**）；Qwen3-32B 上 VCORE 41.29/45.93 vs DFT 39.79/49.57（**OOD 大输 DFT**）。VCORE 优势集中在 ID/OOD 综合平均与中小模型（作者归因：VCORE 按 population loss 重加权，模型已强或 CoT 与目标失配时增益有限）。"strongest overall" 的措辞需结合此细节理解【原文 §5、Table 1】。
- 假设与失效边界：
  - 显式假设【原文】：理论建立在单步一阶 Taylor + SGD 假设；one-backward 探针的无偏性依赖沿均匀下降方向的小扰动极限。
  - 隐式假设【推断】：实际用 AdamW 而非 SGD，理论与实现的一致性论文未严格讨论；reweighting 可能过度强调"泄露最终答案的 spurious 模式"、放大数据集 artifact/标注偏置（作者自陈 Limitations，建议加 dropout masking/answer prefix control）。
  - 何时失效【原文/推断】：模型已强（大模型 32B）或 CoT 与目标失配时增益有限（OOD 不及 DFT）；GSM8K 这类短链简单任务上长 CoT SFT 反而略降（Obs 8）；max_gen 限 8192 对长 CoT 可能偏短；全程 LoRA 未做全参 SFT 验证。
- 祛魅总结【推断】：真贡献是给"长 CoT SFT 的 token 加权"一个干净的优化理论推导（约束优化→Gibbs 闭式→one-backward 无偏估计→方差对齐），且 one-backward 探针确实把"每 token 梯度效用"做到了可算（这是工程亮点），方差控制的必要性有 Fig.3 硬证据。被适度包装的是"strongest overall"——大模型 OOD 上输给 DFT、优势集中在综合均值与中小模型；理论用 SGD 而实现用 AdamW 的 gap 未讨论；one-backward 每步成本约翻倍却无端到端计时。定位清晰：SFT 阶段的 token reweighting，与 DFT/iw-SFT 同类竞争。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：token 梯度效用 \(s_t=\langle\nabla L,\nabla\ell_t\rangle\)（该 token 梯度与全局下降方向的对齐度）——纯从训练时梯度信息读出，**不靠 teacher 引导/置信阈值/熵过滤**【原文 §4.1】。
  - **改什么**：SFT 阶段每 token 的 loss 权重（经 Gibbs \(q^*\) + 方差控制 \(\alpha\) 重加权交叉熵）；模型参数随之更新【原文 §4】。
  - **何时改**：SFT 训练阶段、每 batch 动态计算 \(q^*\) 与 \(\alpha\)（介入 loss）【原文 Algorithm 1】。
  - **免梯度?**：否（本质是梯度方法；且额外用 one-backward 探针估梯度效用）。
  - **记忆-技能生命周期**：N/A（无记忆/技能库；纯 SFT 损失重加权，知识固化进参数）【推断】。
  - **防遗忘机制**：无专门防遗忘设计；但 \(\mathrm{KL}(q\|u)\le\delta\) 约束防权重过度集中、方差控制防训练发散，间接稳化（非以防遗忘为目标）【原文 §4，推断】。
- ⑦ 开源代码+框架/harness：仓库 https://github.com/coder-gx/VCORE （v1 记 ~219MB；仓库实测含 `llama_factory/`、`transformers-4.52.4/`、`figures/`、`paper_pdf/`、`requirements.txt`、`README.md`、`LICENSE`）。**框架：LLaMA-Factory（Zheng 2024）+ 定制 transformers 4.52.4**——Gibbs 加权 + one-backward 探针 + 方差控制实现于定制 transformers 训练循环（reweighting 介入 SFT 阶段 loss）；训练 LoRA + AdamW + cosine，推理 vLLM v1；硬件 4×RTX PRO 6000 Blackwell【原文 line 874-890 确认 LLaMA-Factory；仓库实测含定制 transformers 与内嵌 paper_pdf】。
- 💰 资源/成本与可扩展性：硬件 4×RTX PRO 6000 Blackwell，1 epoch LoRA。**one-backward trick 使每步约多一个 batch \(B'\)(=32) 的前/后向（成本约翻倍），但论文未给端到端 wall-clock 对比**；对 \(\tau/\epsilon\)（Fig.2b）、batch/lr（Table 3）、训练集规模 4k→32k（Fig.2a）鲁棒（大集略降因质量/风格混杂）【原文 §5、line 874-890，端到端计时原文未说明】。
- 🎯 对"探索-巩固"对标：**可借组件（中等，偏巩固侧的信用分配）**——一句判定：VCORE 不做探索/回轨，但它把"哪些 token 值得固化"做成了无需 teacher 的可算信号，正对本项目"巩固/固化进参数"阶段的 token 信用分配缺口。依据：本项目 idea 要把"走通的有效路径"巩固进参数且不被噪声污染——VCORE 的梯度效用 \(s_t\) 提供了一个"对齐全局下降方向"的 token 重要性度量，可用来在 path-recovery 后**只重加权真正推动学习的恢复步、抑制 spurious token**（与 survey-grpo-step-segmentation 记忆里"切高熵/低置信关键步"互补：VCORE 是从梯度对齐而非熵来定关键 token）。**可借组件**：(1) one-backward forward-mode 探针——一次前/后向估全 token 梯度效用，可低成本嵌入任何 SFT/蒸馏阶段做 token 加权；(2) 约束优化→Gibbs 闭式 \(q^*\propto\exp(\tau s_t)\) 的推导范式可复用于"加权脚手架蒸馏"（把 teacher 信号按梯度效用重加权）；(3) 方差控制 \(\alpha=\sqrt{V_u/V_q}\) 防长序列重加权发散——对长 CoT 巩固尤其有用。**缺口**：(1) 是 off-policy SFT 阶段方法，无 on-policy 自选轨迹（与本项目 on-policy 核心不符）；(2) 无 teacher 脚手架（恰相反，去 teacher 化）；(3) 无 path-selection/recovery 显式机制、无 MTP 前瞻；(4) 大模型 OOD 上增益有限。
- 🔭 开放问题/未来方向：
  - 【原文】更多样数据/其它 reasoning 模型生成的 CoT（当前仅 DeepSeek 蒸馏的 OpenMath/OpenCode）；加 dropout masking / answer prefix control 正则防过度强调泄露答案的 spurious 模式；全参 SFT 验证（当前仅 LoRA）。
  - 【推断】把 one-backward 梯度效用从 off-policy SFT 推广到 on-policy 蒸馏（在 student 自采样轨迹上按梯度效用重加权脚手架信号），是与本项目 OPD 路线对接的直接方向；补端到端 wall-clock 计时；讨论 SGD 理论与 AdamW 实现的一致性。

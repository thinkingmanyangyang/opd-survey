csd | Distillation of Large Language Models via Concrete Score Matching (CSD) | KAIST + summary.ai（Yeongmin Kim、Donghyeok Shin、Mina Kang、Byeonghu Na、Il-Chul Moon） | v3 2026-05-30 · ICLR 2026 · cs.LG | 主题线 L1（蒸馏 loss 设计/离线 logit-KD）·相关性 中

**原始论文**：https://arxiv.org/abs/2509.25837

## 一眼看懂
- 🟦 TL;DR：知识蒸馏（KD）主流是每个 token 上做 **softmax 概率匹配**（KL/f-散度），但 softmax 把 teacher 的大 logit 差异"压平"——\([-1,-4,4]\) 与 \([1,-9,6]\) 经 softmax 后概率几乎一致、梯度几乎相同，淹没 logit 里编码的细粒度知识（GPT-2-1.5B 上仅 **0.0023%** 的 token 概率 >0.01）【原文 §1 L45-48, Fig.1a/b】。直接对齐 logit（DLD）又因不允许"logit 常数平移不变性（logit shift invariance）"而严重收窄解集【原文 §1 L63-69, Fig.1c】。CSD 提出 logit 级的"离散 concrete score 匹配"目标：匹配 student/teacher 在**所有词表对**上的相对 logit 差，既绕过 softmax 平滑、又对 logit 常数平移不变（解集是 DLD 的**真超集**），且给出 \(O(|V|)\) 线性时间梯度【原文 Abstract, §3】。
- 最巧的一步：**对 concrete score 取 log（式7→8）**。朴素 concrete score 是概率比 \(q_\theta(x)/q_\theta(y_t)\)，分母趋零会发散导致训练不稳（自回归 LLM 特有，因逐 token 算概率再取比，不像离散扩散模型 sθ 一次输出全部比值）【原文 §3.1 L231-237】。取 log 后（a）目标变成 logit 之间的 MSE \((f_\theta[x]-f_\theta[y_t]-f_T[x]+f_T[y_t])^2\)，避开比值计算保稳定，(b) 天然落到 logit 级、契合动机【原文 §3.1 L265-267】。抽掉这步：要么训练发散，要么退回概率比匹配、丢掉"logit 级"卖点。

## 为什么做
- 研究背景：KD（Hinton 2015）让小 student 继承大 teacher 以降推理成本；主流在每 token 做 softmax 概率匹配，从 KL 出发到各类 f-散度（Wen/TV、Gu/RKL、Agarwal/GJS）与平滑变体（Ko/SKL-SRKL、Shing/TAID）【原文 §1 L37-42, §2.1】。
- 解决的具体痛点：
  - ①**softmax 平滑抹掉 logit 级知识**——大 logit 差→近乎相同概率/梯度（Fig.1b 反例），大词表下分布极稀疏（Fig.1a），teacher 对多数次要 token 给几乎相同的梯度信号【§1 L43-48, §2.1 L144-147】；
  - ②**DLD 解集受限**——DLD 直接对齐 logit 绕过 softmax，但其最优解**不允许 logit 常数平移不变**（而 softmax 推理下 logits 整体加常数 \(C\) 等价），解集被严重收窄，teacher/student 容量差大时尤甚【§1 L63-69, §2.1 L155-160】。
- 相关工作 & 各自不足：
  - **概率匹配族**（KL/RKL/Sym-KL/Jeffrey/TV/GJS/SKL/SRKL/AB）——都受 softmax 平滑约束【原文 Table 1】；
  - **DLD**（Ba&Caruana 2014；Kim 2021）——logit MSE \(\frac12\sum_{y_t}w(y_t)(f_\theta[y_t]-f_T[y_t])^2\)（式4），有更好泛化/表征但**解集受限**、且在 LLM 上"not been thoroughly explored"；
  - **能量模型 score matching**（Hyvärinen 2005；Song&Kingma 2021）——避开 sum-to-one 归一化约束的连续变量 Stein score；其离散版 concrete score（Meng 2022）此前主要用于**离散扩散模型**（Lou 2024 直接参数化 concrete score 拟合数据分布），CSD 反过来用它设计**自回归 LLM 的蒸馏 loss**；
  - **on-policy KD**（ImitKD/GKD/DistiLLM）——改的是"用谁生成的数据"（student on-policy / 混合 / 自适应选择），与"散度目标本身"**正交**【原文 §2.1, §4.1 L602-608】。
- 动机链：大词表 LLM 的 logit 编码丰富知识→softmax 抹平 + DLD 解集受限两条路都不理想→借能量模型的 score matching（绕开归一化约束）+ 其离散版 concrete score→适配自回归 LLM（解稳定性 + \(O(|V|^2)\to O(|V|)\)）→得到"兼顾两者 + 带最优性保证（解集应是 DLD 超集）"的目标【原文 §1 L70-82】。
- 与最近邻工作的Δ：
  - vs DLD——CSD 对齐"词表对之间的**相对** logit 差"而非"同一 token 的**绝对** logit"，天然对常数平移不变→\(\Theta^*_{\text{CSD}}\supsetneq\Theta^*_{\text{DLD}}\)（Thm 2）；
  - vs KL——梯度结构相似（都是"student 大处降、teacher 大处升"），**唯一区别是 logit 系数的归一化方式**：KL 用 softmax（问题源），CSD 用 **centering 归一化**（直接捕获 teacher logit 信息）【§3.2 L411-471】。有用点：更大解集 + 不被 softmax 平滑，在容量差大时更能逼近 teacher。

## 怎么做 + 靠不靠谱
- 方法流水线（读完可复现）【原文 §3, Algorithm 1】：①对每位置 \((c,y_{<t})\)，用 student logits 算"当前 token \(y_t\) 换成词表任一其它 token \(x\)"的相对概率比（concrete score），teacher 同理→②取 log 稳定训练，得 logit 级 MSE 目标（式8）→③可分权重假设 \(w(y_t,x)=w_1(y_t)\cdot w_2(x)\) 下，梯度 \(O(|V|)\) 算出（Thm 3, Algorithm 1）→④即插：把 CSD(S,S) + DLD(S) 叠加进 ImitKD/GKD/DistiLLM 的数据策略。
- 关键公式（真实形式 + 直觉）：
  - **概率与 KD 通用目标**（式1-2）：\(q_\theta(y_t\mid c,y_{<t})=\dfrac{\exp(f_\theta[y_t])}{\sum_{x\in V}\exp(f_\theta[x])}\)；KD = \(\mathbb{E}_{(c,y)}\bigl[\frac1L\sum_t D(p_T\Vert q_\theta)\bigr]\)。
  - **concrete score**（Meng 2022，离散版 Stein score）：
    \(\displaystyle s_\theta(y)\triangleq\left[\frac{q_\theta(x)}{q_\theta(y)}\right]_{x\in V}\)
    刻画"从当前 token \(y\) 换到其它 token \(x\)"的相对概率变化，唯一确定分布且无需算配分函数 \(Z_\theta\)【§2.2 L192-200】。
  - **朴素 concrete score matching**（式6，不稳）：\(L_{\text{CSM}}=\frac12\sum_y\sum_x w(y,x)\bigl(\frac{q_\theta(x)}{q_\theta(y)}-\frac{p_{\text{data}}(x)}{p_{\text{data}}(y)}\bigr)^2\)。
  - **CSD 目标**（式7→式8，核心；取 log 后变 logit MSE）：
    \(\displaystyle L_{\text{CSD}}(\theta;p_T,w)=\frac12\sum_{y_t\in V}\sum_{x\in V}w(y_t,x)\Bigl(\log\tfrac{q_\theta(x\mid c,y_{<t})}{q_\theta(y_t\mid c,y_{<t})}-\log\tfrac{p_T(x\mid c,y_{<t})}{p_T(y_t\mid c,y_{<t})}\Bigr)^2 =\frac12\sum_{y_t\in V}\sum_{x\in V}w(y_t,x)\bigl(f_\theta[x]-f_\theta[y_t]-f_T[x]+f_T[y_t]\bigr)^2\)
    第二个等号是关键：log 概率比 = logit 差，所以目标 = 匹配"所有词表对 \((y_t,x)\) 的相对 logit 差"，对 \(f_\theta\) 整体加常数 \(C\) 不变【§3.1 L240-267】。
  - **高效梯度**（Thm 3，式9，可分权重 \(w(y_t,x)=w_1(y_t)w_2(x)\) 下 \(O(|V|)\)）：
    \(\displaystyle \nabla_\theta L_{\text{CSD}}(\theta;p_T,w)=\sum_{y_t\in V}\mathbf{w}(y_t)^{\!\top}\bigl(\tilde f_\theta[y_t]-\tilde f_T[y_t]\bigr)\nabla_\theta f_\theta[y_t]\)
    其中 \(\mathbf{w}(y_t)=(w_1(y_t),w_2(y_t))^{\!\top}\)，归一化（centering）logit \(\tilde f^w_\theta[y_t]=f_\theta[y_t]-\mathbb{E}_{w(x)}[f_\theta[x]]\)，teacher 同理。Algorithm 1 每步只需线性时间（算加权均值 logit→减去做 centering→组合）。不接受可分假设则退回 **Monte Carlo 估计**（Algorithm 2：按 \(w_1\) 采单个 \(y_t\)，可建模联合权重空间、不需独立性，但 batch 内方差更大、收敛略慢，Fig.5c）【§3.2 L370-410】。
  - **CSD vs KL 梯度对照**（取 uniform 权重 \(U\)，直觉关键）：
    \(\displaystyle \nabla_\theta D_{KL}(p_T\Vert q_\theta)=\sum_{y_t}\Bigl(\underbrace{\tfrac{\exp(f_\theta[y_t])}{\sum_x\exp(f_\theta[x])}}_{\text{softmax 归一 student}}-\underbrace{\tfrac{\exp(f_T[y_t])}{\sum_x\exp(f_T[x])}}_{\text{softmax 归一 teacher}}\Bigr)\nabla_\theta f_\theta[y_t]\)
    \(\displaystyle \nabla_\theta L_{\text{CSD}}(\theta;p_T,U)=\sum_{y_t}\frac{2}{|V|}\Bigl(\underbrace{f_\theta[y_t]-\tfrac{\sum_x f_\theta[x]}{|V|}}_{\text{centering 归一 student}}-\underbrace{\bigl(f_T[y_t]-\tfrac{\sum_x f_T[x]}{|V|}\bigr)}_{\text{centering 归一 teacher}}\Bigr)\nabla_\theta f_\theta[y_t]\)
    两者都"student 大处降、teacher 大处升"，**唯一差别是系数归一化**：KL 用 softmax（抹平 logit 信息），CSD 用 centering（保留）。\((w_1,w_2)\) 提供归一化设计空间：\(w_1\) 控梯度更新时词表加权、\(w_2\) 控系数归一化（角色再反序作用）【§3.2 L411-471】。
  - **三条定理**：
    - **Prop.1（一致性）**：\(|\Theta|\to\infty\)、任意 \(w>0\)，最优解处 \(L_{\text{CSD}}(\theta^*)=0\) 且 \(q_{\theta^*}=p_T\)——保证 student 收敛到 teacher【§3.1 L342-350】。
    - **Thm 2（解集真超集）**：\(\Theta^*_{\text{CSD}}\supsetneq\Theta^*_{\text{DLD}}\)——DLD 能得的解 CSD 都能得，且 CSD 对常数平移不变（\(f_\theta=f_T+C\) 时 CSD=0 但 DLD≠最优），容量差大时优势更明显【§3.1 L351-363】。
    - **Thm 3（O(|V|) 梯度）**：见上式9【§3.2 L370-378】。
- 逐组件必要性：
  - **log 变换**：稳定性 + logit 级，无它发散（见"最巧一步"）。
  - **可分权重假设**：换 \(O(|V|)\) 线性梯度；不接受则退回 MC（高方差、慢）——"高效"与"通用"二选一，论文有交代【§3.2 L402-410】。
  - **权重函数 \((w_1,w_2)\)**：Table 1/Fig.3a 消融——默认 **(S,S)**（detached student 概率）最高保真；换 (U,S) 或 (T,S) 逐渐增多样性（"(S,S) makes the model focus only on regions where the student already assigns high likelihood, limiting exploratory ability"）；CSD 的 trade-off 前沿**包络**了既有 loss【§4.1 L593-601】。
  - **mode-seeking / mode-covering 实例**：同框架内通过权重选择可实例化两类行为【原文 Abstract, §1 L81-82】。
- 实验与证据【原文 §4】：
  - **数据集/设置**：task-agnostic 指令跟随用 **databricks-dolly-15k**（沿用 DistiLLM）；先在 dolly 微调 teacher 再蒸 student；评测 Dolly Eval/Self-Instruct/Vicuna Eval/Super-NI/UnNI，**ROUGE-L 跨 5 seed 平均** + Self-BLEU 测多样性；task-specific（摘要/翻译/GSM8K 数学）+ general chat；backbone GPT-2(0.1/0.3/1.5B)、OpenLLaMA-7B/3B、Gemma-7B-IT、Qwen2.5-7B-IT、**Gemma2-9B-IT（最大 teacher 9B）**。
  - **loss 对比（Table 1）**：GPT-2-1.5B→0.1B，CSD 平均 **20.65**，胜过其余 9 个目标（KL 16.68/RKL 18.88/Sym-KL 17.37/Jeffrey 17.42/TV 19.97/GJS 18.87/SKL 19.60/SRKL 20.00/AB 19.78），5 基准中 **3 个第一**、1 个第二。
  - **on-policy 正交增益（Table 2）**：CSD 叠加到 ImitKD/GKD/DistiLLM（区别仅在数据策略 D：On/Mix/Ada），GPT-2-0.1B/0.3B 全设置平均 ROUGE-L 都升，且**纯 on-policy（ImitKD）下增益最强**（0.3B：ImitKD+Ours 23.12 vs ImitKD 19.23 vs ImitKD+DLD 22.32；0.1B：ImitKD+Ours 20.73 vs ImitKD 16.69）；OpenLLaMA-7B→3B 上 ImitKD+Ours 29.03 也超各 baseline。GPT-4 judge（Fig.4）CSD 最佳模型被判优于其它（teacher 0.61）。
  - **baseline 公平吗**：Table 1 注明"全部用自己实现、同 teacher、纯用蒸馏目标"（排除预训练 loss/SFT 初始化/on-policy）；SKL/AB 略低于原报告，作者归因其依赖预训练 loss/on-policy——**公平性透明，但也意味着 Table 1 是"剥离增强后的纯 loss 对比"，不直接等于各方法的最佳表现**【§4.1 L488-493】。
  - **看着强但没回答核心问题**：增益绝对值不大（0.1B 上 CSD 20.65 vs SRKL 20.00，仅 +0.65 ROUGE-L）；评测以 ROUGE-L/Self-BLEU 代理指标为主，模型规模偏中小（最大 9B teacher），未在大规模推理任务验证。
- 假设与失效边界：
  - 【原文 §2.1 L100-102】假设 teacher 与 student **共享词表/tokenizer**——跨族蒸馏不适用。
  - 【原文 §3.2】\(O(|V|)\) 仅在**可分权重**下成立；一般权重需 Monte Carlo（方差更大）。
  - 【推断】属**离线 logit-level KD**（目标本身正交于"用谁的数据"），非在线/RL/推理路径蒸馏；在大词表 + 大容量差场景理论优势最明显，但论文最大只到 9B teacher，大规模未验证。
- 祛魅总结【推断】：真贡献是**理论清晰的 loss 设计**——从"softmax 抹平 logit 差 + DLD 解集受限"两个具体缺陷出发，用 score matching 思想给出兼顾两者的目标 + 三条定理（Prop.1 一致性 / Thm.2 解集真超集 / Thm.3 \(O(|V|)\) 梯度），且与 on-policy 框架正交可叠加。被高估的是实证强度（代理指标 + 中小规模 + 增益较小）；Thm.3 的"高效"依赖可分权重假设，通用情形退回高方差 MC——"高效"与"通用"不可兼得。最大短板：**截至 v3，GitHub 仅 README 占位、训练代码未释出**（见⑦），复现性无法独立验证。这是一个"漂亮的目标 + 待补的代码"。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：teacher 的 **logit 级**信息（所有词表对的相对 logit 差 \(f_T[x]-f_T[y_t]\)），非 softmax 后概率。
  - **改什么**：参数（student 权重），通过 logit 级 MSE 梯度更新——本质改 student 的 **logits**。
  - **何时改**：离线（固定数据集蒸馏）为主；但与 on-policy 数据策略（ImitKD/GKD/DistiLLM）正交可组合（可变成 student on-policy/混合/自适应数据）。
  - **免梯度?**：否（标准梯度下降蒸馏）。
  - **记忆-技能生命周期**：无外部记忆/技能库；teacher 是固定分布目标（冻结）。
  - **防遗忘机制**：无（单 teacher→single student 蒸馏，不涉及持续学习/遗忘）。
- ⑦ 开源代码+框架/harness：https://github.com/aailab-kaist/CSD 。**【待核→已核实仍为占位】**：repo 已 clone（40 字节），README 仅一行 "# CSD / Official repo for CSD (ICLR 26)"，**无任何训练代码**（2026-06-04 本次复核 `csd/` 目录仅 `.git/` + `README.md`，`find` 仅返回一个 README 文件）。框架：非独立训练框架，而是一个**可替换 KL 的 logit-level 蒸馏目标**，设计为嵌入现有 KD 流程（ImitKD/GKD/DistiLLM）。**未完整可复现：代码未释出，需以论文公式（式8）+ Algorithm 1（含 centering 归一化 + 可分权重梯度）手动实现**，附代码链接供后续手动获取。
- 💰 资源/成本与可扩展性：训练时间/显存见原文 Table 9（§D，本次未展开）；核心成本主张是梯度 \(O(|V|)\) 线性（vs 朴素 \(O(|V|^2)\) 双重词表求和不可行），使大词表可行【原文 §3.2 Thm 3】。规模化证据：从 GPT-2(0.1B) 到 OpenLLaMA-7B→3B 都有增益，但最大 teacher 仅 9B。
- 🎯 对"探索-巩固"对标：**外围相关（loss 层面）**。CSD 只触及"蒸馏散度目标本身"，与本项目"探索-巩固"的核心（teacher 当稀疏脚手架教 path-selection/path-recovery、on-policy 自选、MTP 前瞻）**几乎不重叠**——它是离线 logit 匹配，既无 on-policy 自选轨迹的机制（虽可叠加 ImitKD/GKD），也无脚手架/纠偏/记忆/前瞻。**可借组件**：若本项目的"巩固"环节需要把某种 dense 信号蒸进参数，CSD 的"logit 级、常数平移不变、\(O(|V|)\) 高效"目标可作为**比 KL 更保真的蒸馏 loss 候选**（尤其 teacher/student 容量差大时）；其"centering 归一化 vs softmax 归一化"的梯度分析（两式只差归一化方式）也是理解 logit 蒸馏行为的有用视角。**缺口**：无在线、无 RL、无前瞻、无记忆——离本项目主轴（MTP+OPD）较远。一句判定：技术零件可借（更好的蒸馏 loss），但不解决探索/巩固的任何核心机制，且代码缺位降低即用性。
- 🔭 开放问题/未来方向：
  - 【原文 §4.1 L613-614】针对特定数据集 \(D\) 可能存在比 (S,S) 更优的 CSD 变体（权重 \(w_1/w_2\) 设计空间未穷尽），留待 future work；temperature 调节 trade-off 的更优操作点（Fig.3b）。
  - 【推断】跨 tokenizer/跨族蒸馏（当前假设共享词表）；一般（不可分）权重下避免 MC 高方差的高效估计；在大规模推理任务（数学/代码长链）上验证 logit 级保真是否真带来下游收益；**最紧迫是补全开源训练代码**以让社区独立复现 Thm.2/3 的实际收益。

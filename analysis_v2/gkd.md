gkd | On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD) | Google DeepMind（+Mila / U. Toronto；Rishabh Agarwal、Nino Vieillard 共同一作；Matthieu Geist、Olivier Bachem 等） | 2024-01-17 (arXiv:2306.13649 v3；ICLR 2024) | L1 OPD/自蒸馏 · 相关性高（本课题 OPD 直接源头/奠基）

**原始论文**：https://arxiv.org/abs/2306.13649

## 一眼看懂
- 🟦 TL;DR：自回归 LM 的知识蒸馏有个老毛病——训练时学生拟合的是"固定数据集（真值或教师生成）"的序列，但推理时学生从自己的部分输出自回归续写，会走到训练没见过的状态，早期 token 误差级联放大（train-inference 分布失配 / exposure bias）。GKD 把蒸馏看成"带交互式专家的模仿学习"：让学生在**自己生成**的序列上、用教师的 token 概率当专家标签来学；并把"on-policy 数据比例 \(\lambda\)"和"学生-教师散度 \(D\)（forward/reverse KL、广义 JSD(β)）"统一成两个可调旋钮，所有已有 KD 方法都是其特例【原文 Abstract, §3.1, Algorithm 1】。
- 最巧的一步：**on-policy 数据 + 对采样过程 stop-gradient**（Eq.4）。在学生自生成序列上学，是消除分布失配的关键（λ=0 退回监督 KD、掉到基线，Fig.8 显示 λ≥25% 后性能随 λ 升、λ=1 通常最好）；而"不反传穿过学生采样过程"让它比 MiniLLM 那种序列级 policy-gradient 方案稳得多、且省去多种稳定化技巧。抽掉"on-policy"退回普通 KD/SeqKD（分布失配重现）；抽掉"stop-gradient"则训练不稳/低效。这两点合起来是命门【原文 §3.1 Eq.4, §4, Fig.8】。

## 为什么做
- 研究背景：自回归 LM 靠扩数据+扩参数取得能力，但部署受推理成本/显存限制，故需蒸馏压缩。现有自回归 KD 要么先用教师生成一批固定序列再 SFT（SeqKD，贵），要么在固定数据集上让教师打 token 概率（监督 KD，最小化 forward KL）【原文 §1, §2】。
- 解决的具体痛点：① **train-inference 分布失配（exposure bias）**——固定序列上训、推理时自回归走到未见状态，误差级联（模仿学习里的经典问题）；② **容量失配下 forward KL 的弊端**——学生表达力有限时，最小化 forward KL 迫使学生覆盖教师分布整个支撑，可能把概率质量分散到教师几乎不产生的 token 上，导致幻觉/低质量生成【原文 §1, §3.1 Choice of Divergence】。
- 相关工作 & 各自不足（来龙去脉 + 精确短板）：
  - ① **监督 KD**（Hinton15 软标签、Sanh19 DistilBERT）——固定数据上最小化 forward KL，是 \(\lambda=0\) 的特例；分布失配 + 均值寻求弊端。
  - ② **SeqKD**（Kim&Rush16）——在教师生成序列上做 SFT（最大化教师高概率序列似然），仍是固定数据、贵。
  - ③ **ImitKD**（Lin20）——首次点出"蒸馏↔模仿学习"，混采学生与固定数据（≈ GKD 的 \(\lambda=0.5\) + forward KL），但**停在 token 级 forward KL、未走纯 on-policy、未结合 RL**。
  - ④ **f-distill**（Wen23）——把序列级 KD 表述为 f-散度（total variation），是 GKD 的 \(\lambda=0.5\) + TV 距离特例。
  - ⑤ **并行的 MiniLLM**（Gu23）——把蒸馏当 RL、序列级优化 **reverse KL** 用 policy gradient，但需多种稳定化技巧（梯度反传穿采样、需 baseline/teacher-mixed 采样等）。
  - GKD 用 \(\lambda\)+\(D\) 把 ①–④ 全部统一为特例，并以 stop-gradient 取代 ⑤ 的 policy-gradient 不稳。理论根基是模仿学习里的 **DAgger**（Ross 2011，on-policy 收集 + 专家标注 + 重训）【原文 §1, §5】。
- 动机链：固定数据→分布失配 → 借模仿学习 DAgger"教师作交互式专家"的思路 → 让学生在自生成序列上接受教师 token 反馈（消除失配，且学生进步后自生成数据质量也升，正反馈环）→ 同时放开散度选择，让有限容量学生聚焦教师分布关键区域。
- 与最近邻工作的Δ（精确差异）：最近邻是 ImitKD（混采但停在 token 级 forward KL）与 MiniLLM（RL 式 reverse KL）。Δ：GKD 用 \(\lambda\)（on-policy 比例）+ \(D\)（散度）两旋钮把它们都统一为特例，并新增"纯 on-policy + 任意散度 + stop-gradient + 可叠 RL"的组合。为什么有用：比 ImitKD 更彻底（可纯 on-policy）、比 MiniLLM 更稳（不反传穿采样、无需稳定化技巧），且能与 RLHF/RLAIF 无缝结合（仅需学生样本、几乎无额外超参）【原文 §1, §3.1 Remark, §3.2】。

## 怎么做（细到可复现）
### 0. 散度工具箱（§2）
- 温度 \(\gamma\) 的 softmax：\(p(y_n\mid x)=\dfrac{\exp(z_n/\gamma)}{\sum_i\exp(z_i/\gamma)}\)，学生训练 \(\gamma=1\)，评测用贪心(\(\gamma\to0\))或温度采样。
- KL：\(D_{\text{KL}}(P\Vert Q)=\sum_c P(c)\log\frac{P(c)}{Q(c)}\)，不对称——forward KL（=最大似然、均值寻求/覆盖全支撑）vs reverse KL（模式寻求/聚焦高概率区）。
- **广义 JSD(β)**（Eq.1，0<β<1 在 forward/reverse KL 间插值，且对不相交支撑有界）：
\(\displaystyle D_{\text{JSD}(\beta)}(P\Vert Q)=\beta\,D_{\text{KL}}\!\big(P\,\Vert\,\beta P+(1-\beta)Q\big)+(1-\beta)\,D_{\text{KL}}\!\big(Q\,\Vert\,\beta P+(1-\beta)Q\big).\)
\(\beta\to0\) 时梯度 ≈ forward KL，\(\beta\to1\) 时 ≈ reverse KL。

### 1. 序列级散度记号（§3 Eq.2）
对教师 \(p_T\)、学生 \(p^\theta_S\)（\(\theta\) 可微），定义 token 级散度按序列长归一：
\(\displaystyle D\big(p_T\Vert p^\theta_S\big)(y\mid x):=\frac{1}{L_y}\sum_{n=1}^{L_y} D\big(p_T(\cdot\mid y_{<n},x)\,\Vert\,p^\theta_S(\cdot\mid y_{<n},x)\big).\)
监督 KD = \(L_{\text{SD}}(\theta)=\mathbb{E}_{(x,y)\sim(X,Y)}[D_{\text{KL}}(p_T\Vert p^\theta_S)(y\mid x)]\)（固定数据上的 forward KL，Eq.3）。

### 2. on-policy KD（§3.1 Eq.4，核心）
学生自生成 \(y\sim p_S(\cdot\mid x)\)、在中间状态 \(y_{<n}\) 上模仿教师 token 分布：
\(\displaystyle L_{\text{OD}}(\theta):=\mathbb{E}_{x\sim X}\Big[\mathbb{E}_{y\sim p_S(\cdot\mid x)}\big[D_{\text{KL}}(p_T\Vert p^\theta_S)(y\mid x)\big]\Big],\quad\text{且不反传穿过 }p_S(\cdot\mid x)\text{ 的采样}.\)
"不反传穿采样"使训练稳定高效（区别于 MiniLLM 的 policy gradient）；温度 \(\gamma=1\) 鼓励学生生成多样性。

### 3. 统一目标 GKD（§3.1）
\(\displaystyle L_{\text{GKD}}(\theta):=(1-\lambda)\,\mathbb{E}_{(x,y)\sim(X,Y)}\big[D(p_T\Vert p^\theta_S)(y\mid x)\big]+\lambda\,\mathbb{E}_{x\sim X}\Big[\mathbb{E}_{y\sim p_S(\cdot\mid x)}\big[D(p_T\Vert p^\theta_S)(y\mid x)\big]\Big].\)
两旋钮：\(\lambda\in[0,1]\)（on-policy 学生数据比例）、\(D\)（任意散度）。监督 KD = \((\lambda{=}0,\text{forward KL})\)，on-policy KD = \((\lambda{=}1,\text{forward KL})\)。

### 4. 算法（Algorithm 1，6 步）
① 给定教师 \(p_T\)、学生 \(p^\theta_S\)、数据 \((X,Y)\)；② 每步抽 \(u\sim\text{Uniform}(0,1)\)；③ 若 \(u\le\lambda\) → 用学生当前策略采一批 on-policy 序列 \(y\sim p^\theta_S(\cdot\mid x)\)；否则 → 取固定数据集 batch；④ 更新 \(\theta\leftarrow\theta-\eta\frac1B\sum_{(x,y)\in B}\nabla_\theta D(p_T\Vert p^\theta_S)(y\mid x)\)（对采样过程 stop-gradient）；⑤ 重复 K 步。

### 5. 可选 RL + on-policy GKD（§3.2 Eq.5）
直接优化非可微的真实目标（reward \(r\)）同时向教师正则：
\(\displaystyle \mathbb{E}_{x\sim X}\Big[\underbrace{(1-\alpha)\,\mathbb{E}_{y\sim p^\theta_S(\cdot\mid x)}[r(y)]}_{\text{RL 目标}}-\underbrace{\alpha\,\mathbb{E}_{y\sim p_S(\cdot\mid x)}\big[D(p_T\Vert p^\theta_S)(y\mid x)\big]}_{\text{Generalized On-Policy Distillation}}\Big],\)
\(\alpha\in[0,1]\) 控蒸馏 vs RL 强度，\(\alpha=1\) 即纯蒸馏。可减"alignment tax"。Remark：与 RL 结合时推荐用 reverse KL 或 JSD(0.9)。

### 关键超参/起点假设
- 起点：学生与教师都是已 SFT 的同族模型（学生须能生成质量尚可序列，类两阶段 RLHF：先 SFT 再 online）。
- 学生训练温度固定 1；λ 试 {0, 0.5, 1}；散度试 {forward KL, reverse KL, JSD(0.1/0.5/0.9)}。

### 逐组件必要性
- **\(\lambda\)**：核心。Fig.8 显示 λ≥25% 后性能随 λ 升、λ=1 通常最好；λ=0 退回监督 KD（无失配修正）【§4, Fig.8】。
- **散度 \(D\)**：容量失配下聚焦关键区域。高温评测下模式寻求散度（JSD(0.5/0.9)/reverse KL）质量更好但多样性更低；贪心评测下散度影响很小；指令微调中 reverse KL 显著优于 forward KL（BBH/MMLU +2%/+1%）【§4, Fig.4/6/7/10】。
- **stop-gradient**：使训练稳定高效、区别于 MiniLLM。设计要点非可选【§3.1 Eq.4】。
- **RL 叠加（§3.2）**：可选，用 RLAIF（文本蕴含奖励）抑摘要幻觉同时蒸馏提质（Fig.5）。
- **起点 SFT 学生**：必要假设（非随机初始化）【§3.1 Remark】。

## 靠不靠谱
- 实验与证据：
  - **任务特定蒸馏**：XSum（摘要 ROUGE-2 贪心）、WMT14 en→de（翻译 BLEU beam）、GSM8K（4-shot CoT、外部计算器判对）。**任务无关蒸馏**：FLAN2021（536 万样本/62 任务），held-out 评 MMLU(57)/BBH(23)【§1, §4】。
  - **模型**：教师=SFT 的 T5-XL(~3B)；学生 T5-Small(77M)/Base(250M)/Large(800M)，小 38×/12×/3.8×【§1 Fig.1, §2】。
  - **关键数字（Fig.1）**：on-policy GKD 三任务×各学生规模一致超监督 KD 与 SeqKD；相对初始学生的提升约为基线 KD 的 **2.1×（摘要）/1.7×（翻译）/1.9×（推理）**；任务无关蒸馏 BBH/MMLU **+2%/+1%**（Fig.10）【§1, Fig.1/10】。
  - **数据效率（Fig.3）**：仅 5% 子集且无真值摘要的 on-policy GKD，胜过用全量真值的监督 KD/ImitKD。**RL 叠加（Fig.5）**：GKD+RLAIF 事实一致性超教师同时大幅提升摘要质量。**自蒸馏**：同架构同尺寸 self-distill 学生可超教师【§4】。
  - **baseline 公平吗**：公平且系统——同教师同学生族，对 SFT/监督 KD/SeqKD/ImitKD/f-distill 全面对比，且 λ×散度做系统消融。
  - **"看着强但没回答核心问题"**：散度最优值任务相关、需逐任务调（非普适设置）；on-policy 采样有额外算力（GSM8K 约 1.8–2.2×）。
- 假设与失效边界：
  - 【原文】教师须可查询 token 概率（白盒），黑盒教师不适用（§3 problem setup）。
  - 【原文】起点学生须已具备一定生成质量（非随机初始化），与 RLHF 两阶段范式绑定（§3.1 Remark）。
  - 【原文】on-policy 采样有计算开销，但作者论证相对服务成本可接受（§4）。
  - 【推断】实验局限在 T5 编码器-解码器(≤3B) 与生成式任务（摘要/翻译/算术/指令），未覆盖现代 decoder-only 大模型与长 CoT 推理（依据：§1/§4 全用 T5；GSM8K 仅 4-shot 短 CoT）。
  - 【推断】λ/散度最优值任务相关、无自适应选择准则，迁移到新任务需重新调（依据：§3.1 "optimal divergence task-dependent" + Fig.4/10）。
- 祛魅总结：
  - 真贡献【推断】：把"自回归 KD = 交互式专家模仿学习"这一视角讲透，并用 \(\lambda\)+\(D\) 两旋钮统一监督 KD/SeqKD/ImitKD/f-distill，给出 stop-gradient 的稳定实现 + 可叠 RL——是 on-policy 蒸馏的奠基性框架（被 GAD、Gemma2、Thinking Machines OPD 等广泛引用为源头）。
  - 包装/高估【推断】："self-distill 超教师"等亮点是特定设置下成立，作普适结论易高估；"2.1×/1.7×/1.9×"是相对初始学生增益的倍数（基线 KD 本身增益小时倍数易显大），绝对 ROUGE/BLEU/Acc 提升需看 Fig.1 原值。
  - 低估【推断】："λ≥25% 才见效、纯 on-policy 最好"这一定量规律，对后续所有 OPD 工作的"该掺多少 on-policy 数据"是很实用的先验，但常被引用者忽略。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=教师在学生自生成序列上的 token 级概率分布（白盒 logits），经散度 \(D\) 对齐｜**改什么**=学生全参（\(p^\theta_S\) 可微）｜**何时改**=训练期，每步以概率 \(\lambda\) 用 on-policy 学生序列、否则用固定数据，逐 token 算散度更新｜**免梯度?**=部分——对学生采样过程 stop-gradient（不反传穿采样），但仍是监督式梯度拟合 token 概率（非 policy gradient）｜**记忆-技能生命周期**=无显式记忆/技能库，知识固化进学生参数｜**防遗忘机制**=无专门防遗忘（可选 RL 叠加时把"向初始策略正则"改为"向教师策略正则"，间接锚定）。
- ⑦ 开源代码+框架/harness：原论文实验在 Google 内部 **JAX / T5** 栈，**无公开官方训练仓**。社区实现：HuggingFace **TRL 的 `GKDTrainer`**（https://huggingface.co/docs/trl/gkd_trainer，已作内置 trainer）。本地**未 clone**（仅文档/集成，CloneTier=B）——属"方法已被主流框架内置"，无需单独仓库即可复现。
- 💰 资源/成本与可扩展性：学生 77M–800M、教师 ~3B，规模小（2023 年工作）。on-policy 采样增算力：GSM8K 约 1.8–2.2×；作者论证相对服务成本可接受。与 RLHF 结合几乎无额外超参/算力开销【原文 §3.1 Remark, §4】。
- 🎯 对"探索-巩固"对标：**强支撑（OPD 源头）+ 可借组件**。一句判定：GKD 是本课题"on-policy 蒸馏"这一支的直接奠基——"让学生在自己会走到的状态上、由 teacher 提供 token 反馈"正是"teacher 当稀疏脚手架、student on-policy 自选"的最基础形态。可借组件：① **\(\lambda\) 旋钮**（on-policy 数据比例）可直接迁移为"何时让 student 自走、何时给 teacher 脚手架"的混合调度；② **散度选择（mode-seeking reverse KL）**对应"偏向学生自己能走通的路径"（模式寻求=聚焦学生可达的教师高概率区，正合"探索/选路偏向自己能走通的开头"）；③ **stop-gradient 前向硬/后向软** 的稳定化思路（与 memory 里 forward-hard/backward-soft 同构，可迁移到 MTP/OPD）；④ **Eq.5 的 RL+蒸馏统一**（reward 项 + 向 teacher 正则项）为"探索(RL)与脚手架(蒸馏)同目标"提供干净模板。缺口：GKD 是全序列 token 级蒸馏，**无 path-recovery 单点接管**（不区分"走偏点"专门接管）、**无 MTP 前瞻**、**无技能/记忆固化**——它只解决"在哪条分布上学"，不解决"探索到的好路径如何选择性固化而不遗忘"。依据：Eq.4 全 token 均匀蒸馏、§3.1 起点须 SFT。
- 🔭 开放问题/未来方向：【原文】最优散度任务相关，缺自适应选择（§3.1）；未覆盖 decoder-only 大模型与长 CoT。【推断】把"全 token 均匀蒸馏"改成"只在走偏/高熵关键步由 teacher 接管"（path-recovery 单点接管），即从 GKD 的稠密 OPD 走向稀疏脚手架 OPD；引入 MTP 让学生前瞻判断"自己会走到哪、是否需要 teacher"，把 \(\lambda\) 从随机比例变成"按前瞻信号触发"（依据：GKD 的 \(\lambda\)/散度框架 + 本课题 path-recovery/MTP 前瞻）；将 GKD+RL 的"向教师正则"扩展到带记忆/技能库的持续蒸馏以防遗忘。

读到PDF? 是（PyMuPDF 全文 18 页；正文 §1–3.2 + Algorithm 1 + Eq.1–5 散度/RL 公式全核，实验图 Fig.1/3/4/5/8/10 经正文转述，未逐图核原始数值）｜L线 L1（白盒 token 级 on-policy 蒸馏奠基；兼 L6 散度/信号）｜对标结论 强支撑 OPD 源头 + 可借组件（λ 混合调度、mode-seeking↔偏向可走通路径、stop-gradient 前向硬后向软、Eq.5 RL+蒸馏统一模板）；缺 path-recovery 单点接管/MTP/记忆固化｜残留待核 0

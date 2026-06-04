amft | AMFT: Aligning LLM Reasoners by Meta-Learning the Optimal Imitation-Exploration Balance | 清华大学电子工程系（Lixuan He, Jie Feng, Yong Li） | 2025-08-09 arXiv v1 · Preprint under review | L2 统一SFT-RL · 相关性高（与"探索-巩固"概念同构）

**原始论文**：https://arxiv.org/abs/2508.06944

## 一眼看懂

> 一句话导读：别再做"先 SFT 后 RL"两阶段，也别靠拍脑袋硬切换；AMFT 把模仿(SFT)和探索(RL)揉进一个循环，并把"两者各占多少"当成一个能学的旋钮，让它直接对准"最终验证集表现"来调。

- 🟦 TL;DR：AMFT 把 SFT 和 RL 揉进**单阶段单循环**，并把"模仿(SFT) vs 探索(RL) 的配比 \(\mu\in[0,1]\)"当成**可学习参数**。它用 meta-gradient(元梯度，即"对超参本身求梯度")以"最大化最终验证集表现"为元目标，**前瞻式**地学这个 \(\mu\)（§3.2 Eq.4-6）。在数学/视觉推理上自称 new SOTA、OOD 泛化更好(§4.2 Table 2 ID 61.3/OOD 63.3；§4.3 Table 3)。
- 最巧的一步：**把 \(\mu\) 当可学习参数，用 meta-gradient \(\nabla_\mu U(\theta)\) 前瞻优化**（Eq.5）。所谓"前瞻"，是指它回答的是"现在这样配比，对若干步之后的最终性能是否最优"，而不是只看当下含噪的信号。抽掉它(消融 w/o Meta-Gradient，Table 4)：数学 ID 61.3→57.0、OOD 63.3→60.5，退化成纯熵启发式(也就是"反应式")。这一点正是 AMFT 区别于 SRFT/SASR/DyME 等所有"反应式启发式"方法的唯一关键。

## 为什么做

> 一句话导读：两阶段 SFT→RL 最大的病是阶段切换时把前面学的结构知识忘了(灾难性遗忘)；而现有单阶段方法虽不分阶段，却都"看当下信号被动调"，没有前瞻。AMFT 想用一个原则化、动态、前瞻的方式找最优配比。

- 研究背景：LLM 推理后训练的主流是 SFT→RL 两阶段（§1、§2.3）。SFT 擅长模仿，但困于静态数据集、倾向记忆、OOD 差；RL 探索强、泛化好，但样本低效、稀疏奖励下不稳，且 on-policy 受基模上界限制。最关键的是**阶段间目标突变导致灾难性遗忘**——RL 会覆盖 SFT 学到的结构知识（§1、§2.2）。
- 解决的具体痛点：现有单阶段统一方法**都是反应式(reactive)**——依赖短期含噪的启发式做被动调整或硬切换，缺前瞻性（§1、Table 1 系统对比）。核心未答之问是：**如何用一个原则化、动态、前瞻的方式找到模仿-探索的最优配比，而不是去反应噪声局部信号？**
- 相关工作与并行技术路线（按三大领域分，逐条点短板，§2 + Table 1）：
  - **领域一 SFT(模仿锚定，§2.1)**：大规模 instruction tuning(FLAN)、Minerva(step-by-step exemplar 刷数学 SOTA)、LIMA("少量高质量样本即可对齐"，把 SFT 定位为 format teacher)。短板有三条——NLL 目标只学训练分布，导致记忆而非泛化、OOD 差(引[4,29])；无从自身错误学习/探索更优解的机制；理论上 SFT 是在优化一个隐式 reward，但**缺 KL 正则项**，会 policy drift(引[35])。
  - **领域二 RL(探索泛化，§2.2)**：RLHF(InstructGPT/ChatGPT，用人类偏好 reward)、RLVR(DeepSeek-R1/Kimi-1.5，用可验证程序奖励)。短板三条：
    - (a) **不稳/低效**——PPO 资源重、高方差；on-policy RLVR 多是放大已有能力而非创造新能力(引[45,42])；不加正则会 policy collapse(引[47])。
    - (b) **稀疏奖励**——多步全对才 +1，难拿到学习信号("advantage collapse"，引[20,22])。
    - (c) **灾难性遗忘**——RL 接在 SFT 后会覆盖结构知识(引[3,9])。
  - **领域三 混合范式(§2.3 + Table 1)**：早期是固定交织/静态损失组合(引[25,21,24])，短板是固定配比难达最优；新一波是**自适应反应式**：
    - **SRFT**[11]：单阶段连续加权，用 policy entropy；短板是熵未必与性能相关。
    - **SuperRL**[22]：adaptive switch，按 reward density；短板是二元开关，中等 reward density 不理想。
    - **SASR**[3]：按 gradient norm；短板是梯度范数只是学习稳定性的代理。
    - **DyME**[20]：按 generation correctness 二元切换；短板是为小模型/稀疏奖励定制。
    - **LUFFY**[39]：off-policy data mixing；短板是固定 off-policy 比例。
    - AMFT 是**首个用 meta-gradient 直接优化长期性能**的连续配比方法(Table 1 末行)，灵感源自 meta-learning 做超参优化(引[10])。
  - 其他对照基线(§4.1)：ReLIFT[26]（交织 RL + 在难题上在线 fine-tune）、TAPO[37]（把外部知识作"thought patterns"融进 GRPO）。
- 动机链：现状(两阶段/反应式单阶段) → 缺陷(遗忘 + 反应式不前瞻) → 把配比 \(\mu\) 升格为可学习参数、用 meta-gradient 前瞻优化。为什么不用固定/手调 \(\mu\) 或熵启发式？§3.2 论证固定 \(\mu\) 难最优；熵这类信号只是 short-term proxy，并不直接对齐"最终性能"这个目标。
- 与最近邻工作的精确差异：相对 SRFT（同为单阶段连续加权，但用熵），差在**用 meta-gradient 替换熵作为主控信号**(熵在 AMFT 里降级为辅助的短期纠偏，Eq.6 第三项)；相对 SFT→RL，差在**单循环混合 batch + 动态 \(\mu\) 平滑过渡**而非阶段突变。

## 怎么做 + 靠不靠谱

> 一句话导读：先把 SFT 和 RL 统一成"同一策略下两种互补 reward"，再看它怎么用双层优化学那个配比 \(\mu\)——长期信号靠 meta-gradient，短期稳定靠熵。消融三件套缺一不可，但代码 404、且"前瞻>反应"的因果归因偏间接，是最大折扣。

### 0. 统一视角：SFT 与 RL = 同一策略下的两种互补 reward（§3.1）
- **显式 outcome reward（来自 RL）**：可验证奖励 \(R_{\mathrm{explicit}}(\tau)\)(正确 +1 否则 0)，目标 \(J_{\mathrm{RL}}(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R_{\mathrm{explicit}}(\tau)]\)（Eq.1），实际用 **PPO-clip** 的策略梯度损失 \(L_{\mathrm{RL}}(\theta)\) 做上升。
- **隐式 path reward（来自 SFT）**：把标准 SFT 的 NLL 解释为优化一个隐式 reward \(R_{\mathrm{implicit}}(\tau)\)(忠实复现专家轨迹则高，引[35,7])：\(L_{\mathrm{SFT}}(\theta)=-\mathbb{E}_{(x,y_{\mathrm{demo}})\sim D_{\mathrm{SFT}}}[\log\pi_\theta(y_{\mathrm{demo}}|x)]\)（Eq.2）。
- **统一损失**：\(L_{\mathrm{total}}(\theta;\mu)=(1-\mu)\cdot L_{\mathrm{RL}}(\theta)+\mu\cdot L_{\mathrm{SFT}}(\theta)\)（Eq.3）。\(\mu_t\in[0,1]\) 是"模仿-探索旋钮"，控制每步探索 vs 模仿的相对影响。核心创新在于**怎么学 \(\mu_t\) 的调度**。

### 1. 自适应配比控制器（§3.2，全文核心，双层优化）
把配比升格为 bilevel optimization：内层固定 \(\mu\) 优化策略 \(\theta\)，外层优化 \(\mu\) 来提升长期性能。
- **长期信号 — meta-gradient**：把 utility 定义为留出验证集上的期望显式奖励 \(U(\theta)=\mathbb{E}_{(x,\tau)\sim\pi_\theta(\cdot|D_{\mathrm{val}})}[R_{\mathrm{explicit}}(\tau)]\)（Eq.4）。控制器周期性地算 \(U\) 对 \(\mu\) 的梯度(链式)：
  \(\displaystyle \nabla_\mu U(\theta_t)=\nabla_\theta U(\theta_t)\,\frac{\partial\theta_t}{\partial\mu} \quad(\text{Eq.5})\)
  完整的 Jacobian-vector 积 \(\partial\theta_t/\partial\mu\) 很贵，所以用 **one-step 近似**：对内层单步更新 \(\theta_t\approx\theta_{t-1}-\alpha\nabla_\theta L_{\mathrm{total}}(\theta_{t-1};\mu_t)\) 求导(meta-learning 常用技巧，引[10])。直觉就是问"\(\mu\) 往哪挪能让未来验证奖励更高"。
- **短期信号 — 熵启发式**：用 policy entropy \(H(\pi_\theta)\) 当稳定性代理——熵远高于目标 \(H^*\)(混沌探索)就增 \(\mu\) 强化 SFT 稳定项；熵远低于 \(H^*\)(将坍塌/过拟合)就减 \(\mu\) 鼓励探索。\(H^*\) 是超参，初始化为 warm-up SFT 策略的平均熵。
- **合一更新规则**：
  \(\displaystyle \mu_{t+1}=\mathrm{clip}\big(\mu_t+\eta_\mu\nabla_\mu U(\theta_t)+\eta_H(H^*-H(\pi_{\theta_t})),\,\mu_{\min},\,\mu_{\max}\big) \quad(\text{Eq.6})\)
  \(\eta_\mu\) 和 \(\eta_H\) 分别是长期 meta-gradient 与短期熵的学习率——长期管"找最优策略"，短期管"防即时失稳"。

### 2. 完整训练循环（§3.3，Algorithm 1，可复现级）
1. **初始化**：\(\mu\leftarrow\mu_{\mathrm{init}}\)，初始化 value function \(\phi\)（PPO 需 critic）。
2. **SFT warm-up**：在 \(D_{\mathrm{SFT}}\) 上用 \(L_{\mathrm{SFT}}\) 训 \(W\) 步，提供稳定的指令对齐初始化，同时给出 \(H^*\) 的初值来源。
3. **主循环 t=1..T**，每步做三件事：
   - 取 SFT batch \(\{(x_i,y_i)\}_{i=1}^m\sim D_{\mathrm{SFT}}\) 算 \(L_{\mathrm{SFT}}\)（行4）；在 env 采 RL rollouts \(\{\tau_j\}_{j=1}^n\sim\pi_\theta\)，用 PPO 目标算 \(L_{\mathrm{RL}}\)（行5）。
   - **更新 \(\mu\)**：周期性在 validation batch 上算 \(g_\mu\leftarrow\nabla_\mu U(\theta_t)\)（行7），每步算 \(g_H\leftarrow H^*-H(\pi_{\theta_t})\)（行8），再按 Eq.6 更新 \(\mu_{t+1}\)（行9）。
   - **更新模型**：\(L_{\mathrm{total}}\leftarrow(1-\mu_{t+1})L_{\mathrm{RL}}+\mu_{t+1}L_{\mathrm{SFT}}\)（行11），降 \(\nabla L_{\mathrm{total}}\) 更新策略 \(\theta_{t+1}\) 和 value function \(\phi_{t+1}\)（行12）。
- **数据流动**：每步是 SFT 静态数据 + on-policy rollout 的**混合 batch**；\(\mu\) 用当前 validation 性能反馈即时调整，模型梯度用最新的 \(\mu\)。warm-up 后从 SFT-dominant 平滑过渡到 RL-dominant(Fig 2 红线)。

### 3. 理论支撑（§3.4，intuitive sketch，完整推导在附录 A）
SFT 损失项 \(\mu\cdot L_{\mathrm{SFT}}\) 可解释为动态 KL 惩罚 \(D_{\mathrm{KL}}(\pi_{\mathrm{demo}}\|\pi_\theta)\) 的代理。推导思路：因为 \(\min_\theta -\mathbb{E}_{\tau\sim\pi_{\mathrm{demo}}}[\log\pi_\theta(\tau|x)]\) 等价于最大化示范数据的对数概率，而 \(\mathbb{E}_{\tau\sim\pi_{\mathrm{demo}}}[\log\pi_{\mathrm{demo}}]\) 对 \(\theta\) 是常数，所以整体等价于 \(\min_\theta D_{\mathrm{KL}}(\pi_{\mathrm{demo}}\|\pi_\theta)\)。于是 \(\mu\) 就是这个 KL 约束的**时变 Lagrange 乘子**——比 RLHF 那种固定 KL 惩罚(引[47])更原则化、更响应。

### 4. 逐组件必要性（消融 Table 4，三件套缺一不可，方向一致）
- **meta-gradient**(w/o → Entropy-only)：数学 ID 61.3→57.0、OOD 63.3→60.5；负责前瞻性与性能上限。
- **熵启发式**(w/o → Meta-only)：数学 61.3→55.0（掉最多），§4.2 称去掉后训练不稳——负责短期稳定/防坍塌。
- **SFT warm-up**(w/o)：61.3→56.2；负责稳定初始化、\(H^*\) 初值来源。
- 视觉任务上三件套同向掉点(如 GP-Visual ID 72.1→{68.1/65.4/62.3})，必要性论证较充分。

### 5. 实验与证据
- 数学基模用 **Qwen2.5-Math-7B**，数据 OpenR1-Math-46k-8192；视觉基模用 **LLaMA-3.2-Vision-11B**，数据 General Points / V-IRL。所有 RL 方法都训 500 步、用 PPO、8 rollouts/prompt（§4.1，做了公平控制）；推理 temp 0.6、max len 8192；训练 reward 用 Math-Verify、最终评测用 OAT-Grader。
- 数学(Table 2)：ID Avg **61.3**(超 SRFT 59.5)、OOD Avg **63.3**(超 SRFT 62.5)；AIME24 36.1、AMC 77.9、Olympiad 62.1 多项最高。视觉(Table 3)：GP-Visual ID 72.1/OOD-Rule 45.8/OOD-Visual 70.3、V-IRL ID 95.2/OOD-Rule 71.4/OOD-Visual 85.2，全面超 SFT/RL/RL-from-SFT/LUFFY。效率(Table 5)：达到 60% GP-Visual 只需 ~310 步 / ~15,840 RL rollouts，远少于 RL-from-scratch(~480 步/30,720)与 LUFFY(~400/22,400)。Fig 3 显示纯 RL 快速熵坍塌(policy collapse)、AMFT 维持高熵高奖励。baseline 含 SFT/RL-GRPO/两阶段 + SRFT/LUFFY/TAPO/ReLIFT/SimpleRL-Zero/PRIME-Zero/OpenReasoner-Zero，较公平。
- **关键勘误(已核 PDF)**：RL-only baseline 是 **GRPO**(Table 2 "RL_GRPO[32]")，但 AMFT 统一损失里的 \(L_{\mathrm{RL}}\) 用的是 **PPO-clip**(§3.1 明示 + Algorithm 1 行5/行12 维护 value function \(\phi\))——v1 曾误记，实为 PPO。
- 假设与失效边界：【原文】§3.4 的 "SFT≈KL 正则"是 intuitive sketch，完整推导在附录 A；Eq.5 用了 one-step 近似。【推断】one-step 近似假设单步展开足够代表长期影响，其方差/偏差未在正文量化；长期 meta-gradient 与短期熵两路信号可能冲突，\(\eta_\mu/\eta_H\) 的鲁棒性未知；"new SOTA"为 preprint 自述、未评审。
- 祛魅总结：【推断】真贡献是"SFT/RL = 同一目标下两种互补 reward 信号(隐式 path-level + 显式 outcome)"的统一视角，加上 **meta-gradient 学习配比**这一前瞻机制，消融扎实。包装处有三：①"前瞻>反应"主要靠与 SRFT 等的**端到端对比**间接支撑，缺一个把 meta-gradient 直接对位某反应式信号的 head-to-head 控制实验，因果归因强度中等；②"new SOTA"未评审；③ **代码 404 不可得**(已二次核验)，使 \(\eta_\mu/\eta_H\)、validation batch 构造、meta-gradient 的具体近似实现、稳定性等都无法独立核实，是最大的可信度折扣。作者**高估**了归因确定性，**低估**(或未充分讨论) meta-gradient 的计算开销与方差。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=两路——(a)隐式 path-level reward(SFT/专家轨迹) + (b)显式 outcome reward(RLVR 正确性)；外加 meta 层学"两者配比 \(\mu\)" | **改什么**=策略参数 \(\theta\)（通过动态加权统一损失）+ 一个标量超参 \(\mu\) | **何时改**=在线 per-step(\(\mu\) 每步更新，meta-gradient 周期性估) | **免梯度?**=否(双层梯度优化，含 meta-gradient) | **记忆-技能生命周期**=不适用(无外部记忆/技能库) | **防遗忘机制**=核心卖点——\(\mu\cdot L_{\mathrm{SFT}}\) 作动态 KL 锚定 \(D_{\mathrm{KL}}(\pi_{\mathrm{demo}}\|\pi_\theta)\) 把策略拉向专家分布，meta-controller 自动保留足够 SFT 引导防 RL 阶段灾难性遗忘(§4.2 明确归因 OOD 不掉点于此)。
- ⑦ 开源代码+框架/harness：论文声明 https://github.com/hlxtsyj/AMFT 。**可得性=404 不可得**(2026-06 二次核验：直连 + 代理 127.0.0.1:7897 `git ls-remote` 均 "Repository not found"，本地无 clone)。框架=【待核，仓库不可得】方法层面建立在 PPO-clip(RL) + SFT 加权损失 + meta-gradient 控制器 + value function critic；无法确认具体 RL 框架(verl/TRL/oat 等)。
- 💰 资源/成本与可扩展性：【原文】§4.1 所有 RL 方法训 500 步、8 rollouts/prompt；Table 5 报达标所需 steps/样本/rollouts(AMFT 更省)。【推断】meta-gradient 需额外 validation batch 前向 + 一阶近似计算，单步成本高于普通混合损失，论文未给具体 GPU·h；规模仅验证到 7B/11B。
- 🎯 对"探索-巩固"对标：**最强概念同构者**——AMFT 的"模仿(SFT/path-level) vs 探索(RL/outcome) 的动态配比"几乎是 TSRD"探索(选路)+巩固(回轨固化)"在**损失加权层**的直接对应：\(\mu\) 大=巩固/锚定专家路径，\(\mu\) 小=探索新解。**可借组件**：① meta-gradient 前瞻调权可迁移到"何时该让 student 自探索、何时该用 teacher 脚手架"的调度；② SFT≈动态 KL 锚定(§3.4)与本项目"巩固=固化进参数且不遗忘"同理。**缺口/差异**：AMFT 在**全局标量 \(\mu\)** 层调度，不是 TSRD 的"**岔路口/关键步**单点接管 + path-recovery"那种 token/step 级稀疏脚手架；teacher 只以静态 SFT 数据出现，无"会走通的 teacher 轨迹按需注入"。一句判定：**概念同构、强相关，是'探索-巩固'在统一损失视角下的最佳类比，但调度粒度(全局 \(\mu\) vs 单步脚手架)是关键差异，可借机制不可直接照搬**。依据：\(\mu\) 是序列级单标量，无 per-step 路径选择/恢复语义。
- 🔭 开放问题/未来方向：【原文】Appendix E 进一步研究 controller(正文未展开)。【推断】把全局 \(\mu\) 细化到 per-step/per-token 的前瞻配比(逼近 TSRD 单点接管)；meta-gradient 方差/开销的系统分析；引入"会走通的 teacher 轨迹"作为 path-level 信号源(而非仅静态 SFT)；代码开源以供复现(当前最大障碍)。

hpd | Hybrid Policy Distillation for LLMs (HPD) | 上海交大 / 上海创智学院 / 腾讯(Wenhong Zhu, Ruobing Xie, Rui Wang, Pengfei Liu) | 2026-04-23 Preprint·arXiv 2604.20244v1(README 称 ICML 2026,正文未见〔待核〕) | 主题线 L1 OPD/自蒸馏 · 相关性 高

**原始论文**:https://arxiv.org/abs/2604.20244

## 一眼看懂
- 🟦 TL;DR:白盒知识蒸馏(用 teacher logits 教小模型)长期被三个互相纠缠的旋钮卡住——散度方向(前向 KL vs 反向 KL)、优化方式(loss vs reward)、数据来源(on-policy vs off-policy)。本文先把 SFT/SeqKD/FKLD/RKLD/JSD 统一写成同一个"token 级重加权对数似然"目标(只是权重 w 不同,见 Eq.9 / Table 1),再提出 HPD:对每个 teacher token 用 K1 估计器(Schulman 2020 的廉价 KL 估计)逐 token 判断 student 是低估还是高估了这个 token,据此自适应在前向/反向 KL 之间切换,并把被抑制 token 的概率质量重新导回 teacher token。既保留 one-hot 监督的省算力,又兼容 off-policy 数据 + 轻量近似 on-policy 采样(在 offline 前缀下让 student 采一个非专家 token 来识别并压制"不合理行为")。
- 最巧的一步:**K1 估计器的符号当作"前向/反向 KL 的自动开关"**(Eq.11-12)。`k1>0`(student 低估 expert token)→ 加前向 KL 式增强权重 `p(a*|s)+k1`;`k1≤0`(已高估)→ 取负权重抑制(等价反向 KL 行为)。抽掉这一步(改回固定系数加权和 FKL+RKL),就回退成普通 weighted-sum 蒸馏,失去"逐 token、按需、无超参"的混合,训练动态会像 SFT 一样早熵崩(Fig.1a)。这一步是全篇支点,因为它同时解决了"方向选择"和"无额外超参"两个痛点。

## 为什么做
- 研究背景:压缩大模型靠 KD;白盒 KD 能用 teacher 的预测分布做分布级匹配(KLD on logits)。近年工作强调要选对散度(Cho & Hariharan 2019; Ko 2025),但作者指出"光选散度不够"(§1, 引 Zhang 2025)。
- 解决的具体痛点:三轴(散度方向 / 优化策略 / 数据 regime)被孤立选择、缺统一视角【原文 §1】;FKL 促 mode coverage 但过平滑(Gu 2023)、RKL 促 mode-seeking 但师生差距大时不稳(Lu & Lab 2025);OPD 避免 train-inference 失配但 teacher 侧对 student-generated 输出有分布漂移 + 开销大(Ko 2024)。

- 相关工作 & 各自不足（§1 + §8 + Table 1，按三轴补全）:
  1. **Off-policy / 黑盒蒸馏**:直接拿 teacher 生成当 SFT 数据(Guo 2025; Zhu 2026)。可访问 teacher 时用散度 loss 对齐分布。三类找更好目标者:**Wen 2023** 系统比较各种 f-散度(TV 距离、JSD)在自回归 LM 上的表现;**Wu 2025** 给自适应 KL 来平衡 FKLD/RKLD 早期行为;**ABKD (Wang 2025)** 用 α-β 散度，对 teacher/student 间概率质量分配做有原则控制。短板:**各自只动散度这一轴**，未统一三轴、无"逐 token 自适应方向"机制。
  2. **散度方向(FKLD vs RKLD)**:FKLD(Kim & Rush 2016 SeqKD)促 mode coverage、重罚 missing mode(Song 2020)，但 student 容量不足时**过平滑铺开所有 mode**(Gu 2023; Wang 2023);RKLD 促 mode-seeking、聚焦 teacher 高概率 mode，但低概率有效输出被欠表征(Wang 2025)，且师生失配时**无界 log-ratio 引高方差→不稳**(Ko 2024)。HPD 用 K1 符号逐 token 在二者间切换。
  3. **On-policy 蒸馏(最近邻,直接 baseline)**:**MiniLLM**(Gu 2023)强调 reverse-KL 抑制 student 高估 teacher 低概率区;**GKD**(Agarwal 2024)核心是用 on-policy student 序列;**DistiLLM-2**(Ko 2025)对比式蒸馏——同时升 teacher 序列似然、降 student 序列似然，兼用 off/on-policy 数据。短板:全分布 on-policy 监督**贵**(teacher 在线 forward)。HPD 用 one-hot 风格 + offline 前缀 + 单 token 采样**近似** on-policy 省算力。
  4. **优化策略(loss vs reward)**:K1 既可当 token 级 reward penalty(如 PPO, Schulman 2017)，也可当显式 loss 项(如 GRPO, Shao 2024);近期 OPD 框架(Lu & Lab 2025)即在 student 采样 token 上算 teacher log-prob 的负 K1 当 reward——无偏梯度、稳(Shah 2025)。HPD 统一到 loss 视角(Eq.9)。

- 动机链:现状(三轴孤立、单向散度各有缺陷、OPD 贵)→ 缺陷(无法兼得双向互补性 + one-hot 效率 + on/off 兼容)→ 所以必须把 KD 统一成重加权似然,并用一个廉价逐 token 估计器自动调方向 + 轻量采样近似 on-policy。
- 与最近邻工作的精确 Δ:相对 weighted-sum KL(固定系数混 FKL/RKL),HPD 用 **mask + K1 符号** 做逐 token、无超参的方向选择,并显式把被压制 token 的概率质量"重定向回 expert token"(Reinforce 操作,Eq.14)。相对 GKD/MiniLLM 的全分布 on-policy 监督(贵),HPD 用 one-hot 风格 + offline 前缀 + 单 token 采样近似,省算力。

## 怎么做 + 靠不靠谱
### 问题建模(§2，读懂符号)
把 next-token 生成当序贯决策:轨迹 \(\tau=(s_1,a_1^*,\dots,s_T,a_T^*)\) 来自 offline 数据 D，状态 \(s_t=a_{<t}^*\)(GT 前缀)、\(a_t^*\in V\)=该步专家 token。student 策略 \(q_\theta(a_t\mid s_t)\)，teacher 分布记 \(p\)。预训练即 teacher forcing 最小化 NLL(Eq.1)。痛点(§2):有限容量 student 在高度多模态的 teacher/数据分布上，会把概率质量**稀释到太多 mode**而损害生成质量。

### 统一视角(§4.1，全篇地基)
所有 KD(SFT/FKLD/RKLD)统一为 **token 级重加权对数似然**(Eq.9):
\[
\mathcal{L}(\theta)=\min_\theta\ -\mathbb{E}_{(s_t,a_t)\sim D_\pi}\big[w(a_t\mid s_t)\,\log q_\theta(a_t\mid s_t)\big],
\]
其中 \(D_\pi\)=数据源(on-policy 取自 \(D_{\pi_\theta}\)、off-policy 取自固定集 D 或 teacher \(D_{\pi_T}\))。权重 \(w\) 捕捉师生在 \(s_t\) 处对 \(a_t\) 的局部差异(Table 1):**SFT** \(w=\mathbb{1}[a_t=a_t^*]\)(常数 1)、**FKLD/SeqKD** \(w=p(a_t\mid s_t)\)、**RKLD** \(w=\log p(a_t\mid s_t)-\log q_\theta(a_t\mid s_t)\)、**JSD** \(w=\tfrac12 q\cdot(\log q-\log\tfrac{p+q}{2})\)。
关键洞察(Eq.10，跨整个词表的梯度):对采样 token \(a_t\)，
\[
-\frac{\partial \mathcal{L}(\theta)}{\partial z_v}\propto
\begin{cases}\hat w_t\,q_v(1-q_v), & v=a_t,\\[2pt]-\hat w_t\,q_{a_t}q_v, & v\neq a_t,\end{cases}
\]
\(z_v\)=token v 的 logit、\(\hat w_t=w(a_t\mid s_t)\)。即**正权重升该 token 似然、负权重抑制它并把概率质量按当前分布摊给其他 token**——这正是 reverse K1 的天然行为。

### KL 的 MC 估计(§3.3)
KLD 精确算需遍历全词表(Eq.4)不可行 ⇒ MC 近似。最简 K1 估计器(Eq.8):\(K_1\triangleq\tfrac1N\sum_i \log\dfrac{q_\theta(a_t^{(i)}\mid s_t)}{p(a_t^{(i)}\mid s_t)},\ a_t^{(i)}\sim q_\theta\)，是 \(D_{\mathrm{KL}}(q_\theta\|p)\) 的**无偏估计但高方差**(log-ratio 对大部分样本为负)。

### HPD 方法流水线(Algorithm 1，读完可复现)
对每个 offline \((s_t,a_t^*)\):
1. **算 expert token 的 reverse-K1 gap**(Eq.11):
   \[
   k_1=q_\theta(a_t^*\mid s_t)\big(\log p(a_t^*\mid s_t)-\log q_\theta(a_t^*\mid s_t)\big).
   \]
   \(k_1>0\)⇒student **低估**专家 token(给它太低概率)；\(k_1\le0\)⇒已高估。
2. **student 在同一 offline 前缀下采一个非专家 token** \(a_t\sim q_\theta(\cdot\mid s_t),\ a_t\neq a_t^*\)，算其 gap \(k_1'\)(对 \(a_t\) 套 Eq.11)。这是"轻量近似 on-policy"那一步。
3. **合成 expert token 权重**(Eq.14，三分支):
   \[
   w_t^*\leftarrow\begin{cases}2p(a_t^*\mid s_t)+k_1, & k_1>0\ \text{且}\ k_1'<0\quad(\text{强化:加倍前向KL}),\\ k_1, & k_1<0\quad(\text{抑制:负权重,等价反向KL}),\\ p(a_t^*\mid s_t)+k_1, & \text{其他}\quad(\text{常规前向KL}).\end{cases}
   \]
   直觉:`k1>0` 用前向 KL 强化专家 token；当**同时**采样到的非专家被高估(`k1'<0`)时，把该被抑制的概率质量**加倍导回**专家 token(这就是 Reinforce 操作)。
4. **合成采样 token 权重**(Eq.13 / Algorithm 1 line 15):\(w_t\leftarrow\mathbb{1}[a_t\neq a_t^*]\cdot\mathbb{1}[k_1'<0]\cdot k_1'\)。只在**被高估**的非专家 token 上取负权重压制；`k1'≥0` 则 mask 掉(置 0)以**不强化非专家 token**。
5. **HPD 损失**(Eq.15):
   \[
   \mathcal{L}_{\text{HPD}}=\min_\theta\ \mathbb{E}_{(s_t,a_t^*)\sim D,\ a_t\sim q_\theta(\cdot\mid s_t)}\big[-w_t^*\log q_\theta(a_t^*\mid s_t)-w_t\log q_\theta(a_t\mid s_t)\big].
   \]
   梯度更新 \(\theta\leftarrow\theta-\alpha\nabla_\theta\mathcal{L}_{\text{HPD}}\)。**Hybrid FKL/RKL 的 mask 直觉**:`k1≤0` 时屏蔽前向权重，因为 student 已高估专家 token，再加前向 KL 会与反向方向**梯度冲突**——这是无超参混合的关键。
6. **可与标准 OPD 叠加**(§5.4，HPD+OPD)——HPD 当 OPD 的强初始化。

### 逐组件必要性(§6 消融，作者强调零额外超参)
- **Student sampling(让 student 采自己的非专家 token)**:去掉则性能快速收敛后立刻 plateau(只朝 teacher 分布优化→限制探索、早收敛)。有消融【原文 Fig.3】。
- **Reinforce 操作(被压制时把概率质量导回 expert token,即 w* 的 `2p+k1` 分支)**:去掉则 KL 下降变慢、对齐更慢。有消融【原文 Fig.3】。
- **Hybrid FKL/RKL 的 mask(k1≤0 时屏蔽前向权重)**:防止前向/反向梯度方向冲突;未单独做 mask 的消融,但其必要性由 Eq.12 论证 + 整体训练动态(不熵崩)间接支撑【推断:依据 §4.2 + Fig.1a】。

- 实验与证据:
  - 模型:Qwen2.5(1.5B/3B/7B)、LLaMA3(1B/3B/8B),各取最大者当 teacher;Coder 任务用 Qwen2.5-Coder-7B/DeepSeek-Coder-6.7B 教 1.5B/1.3B。数据:OpenR1-Math-8192(推理)、UltraFeedback(个性化)、WizardCoder(代码)。
  - 推理 off-policy(Table 2):HPD 全面超 SFT/SeqKD/RKLD/JSD,带 ∗(p<0.01)。最亮点:**Qwen2.5-3B 提升 41.0%(28.25→39.83 avg)、LLaMA3-3B 提升 77.9%(19.43→34.56)**,让 3B 逼近大模型推理。
  - on-policy 推理(Table 5):"HPD alone(纯 off-policy)" 已超过 "SFT→OPD 两阶段 baseline";HPD+OPD 进一步最高(Qwen 7B→1.5B 33.41 vs SFT+OPD 29.56)。GPQA 为 OOD,也有提升 → 不只是过拟合 teacher 行为。
  - 训练动态(Fig.1):SFT 早熵崩、KL 停滞、性能不动;HPD 熵稳、KL 持续降、性能持续升,且 train/inference 熵一致(行为对齐)。
  - 个性化(Table 3)/ 代码(Table 4):HPD 平均最优,尤其多轮对话(MT-1T/MT-2T)保持力强;代码上单点不总最高但方差小、更稳。
  - 额外应用:HPD+DPO(Table 6，HPD 初始化使后续 DPO 涨幅最大 +11.92)、迭代自蒸馏(Table 7，HPD-iter 把 teacher 性能无损迁回 base)。
  - baseline 公平性:作者把所有 baseline 统一近似为"重加权似然的不同权重估计器"(SFT=常数 1、SeqKD=p、RKLD=q(log q−log p)、JSD=½q(log q−log((p+q)/2))),同框架同数据,公平【原文 §5 + App C/E】。
- 假设与失效边界:
  - 【原文】用 "teacher 分布 ≈ 其训练数据经验分布" 来省 teacher 在线 forward(§5.1.1:先在 offline 数据上训 teacher 再 GRPO),即 offline 数据当作 teacher 软标签的代理。
  - 【推断】依赖 teacher logits 可得(白盒 KD),黑盒 API teacher 不适用;K1 是 KL 的一阶 MC 近似,师生分布差极大时近似质量与方向判定可能不稳(原文未量化此边界)。
  - 【推断】"offline 前缀 + 单 token 采样近似 on-policy" 仍非真正 on-policy 多步 rollout,长程误差累积场景下逼近程度未知。
- 祛魅总结【推断】:
  - 真贡献:① 把 KD 三轴收进一个重加权似然框架(Table 1 的统一 + Eq.9/10)是清晰、可复用的视角;② K1 符号当方向开关 + 概率质量重定向,实现"逐 token、无超参"混合 KL,且经验上稳(治住 SFT 熵崩)。
  - 包装/可能高估:"hybrid policy / 轻量 on-policy" 名头大,实质是 offline 主导 + 单 token student 采样的近似,并非完整 OPD;增益主要来自方向自适应 + 防熵崩,而非真正 on-policy 探索。低估之处:作为 SFT 即插即用的 loss(LlamaFactory 一行开关 `use_hpd_loss`)实用价值可能被它"统一视角"的理论包装掩盖。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=逐 token K1 估计器(student 对 teacher token 的 under/over-estimate 符号)|**改什么**=student 参数(token 级重加权 NLL)|**何时改**=每 token、按 k1/k'1 分支自适应|**免梯度?**=否,标准反向传播梯度更新|**记忆-技能生命周期**=无显式记忆/技能库,知识固化进权重|**防遗忘机制**=隐式——student sampling + 不熵崩维持探索性,多轮对话保持力强(Table 3);无显式 anti-forgetting 模块。
- ⑦ 开源代码+框架/harness:https://github.com/zwhong714/Hybrid-Policy-Distillation (已 clone,~40MB,真实可用)。**双实现**:① **LlamaFactory**(SFT 路径,HPD loss 在 `LlamaFactory/src/llamafactory/train/hpd.py`,开关 `use_hpd_loss`,支持全参 + LoRA);② **veRL**(RL/post-training 路径,`verl/recipe/HPD/`,`hpd_trainer.py` 基于 `RayPPOTrainer`,FSDP/FSDP2 worker)。配套 HF 模型(Qwen2.5-1.5B-HPD)。
- 💰 资源/成本与可扩展性:卖点即"省"——保留 one-hot 监督效率、用 offline 数据 + 单 token 采样近似 on-policy,避免全分布 OPD 的 teacher 在线 forward 开销;student 训练 ~2k steps、batch 256(on-policy 为 64 prompts × 4 rollouts)。具体 GPU 数/时长原文未给出明确表格(原文未说明)。
- 🎯 对"探索-巩固"对标:**支撑 + 可借组件**。判定:HPD 的 "k1>0 强化 expert / k'1<0 压制非专家 + 概率质量重定向" 与本项目"探索(偏向能走通的开头)+ 巩固(走偏后压制错误分支、固化正确路径)"在**机制上高度同构**;依据:Eq.14 三分支 = 选路(强化 teacher token)+ 回轨(抑制 student 跑偏的非专家 token 并把质量导回)。可直接借:① 用廉价 KL 估计器的符号做"该模仿 teacher vs 该抑制自走偏"的逐 token 门控;② "被压制处把概率质量定向回正确 token" 的 Reinforce 写法(Eq.10/14),可作 path-recovery 单点接管的损失原型。竞品性:它是 offline 主导的逐 token 自蒸馏,**不含 MTP/前瞻**,也非多步 on-policy rollout,与 TSRD 的"on-policy 自选恢复分支"互补而非替代。
- 🔭 开放问题/未来方向:【原文】Roadmap 提到把 HPD 扩到 mid-training / 甚至 pre-training。【推断】K1 方向判定在师生差距极大时的稳健性边界;把"单 token 近似 on-policy"升级为真多步 rollout 后增益是否还在;与 MTP/前瞻信号结合,用未来 token 预测来更早判定"该强化/该回轨"的位置。

key|读到PDF?|L线|对标结论|残留待核数
hpd | 是(全文19页,Eq.1-15全抄准) | L1 | 支撑+可借:K1符号当前/反向KL逐token门控≈选路;Eq.14"压制非专家+质量导回expert"≈path-recovery单点接管损失原型。差异:offline主导、单token近似on-policy、无MTP/前瞻 | 1(README称ICML2026,正文未见)

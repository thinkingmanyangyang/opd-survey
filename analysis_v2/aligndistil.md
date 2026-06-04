aligndistil | AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation | 北京交通大学(交通大数据与AI教育部重点实验室) + 腾讯（Songming Zhang 实习于腾讯，通讯 Yufeng Chen / Jinan Xu） | 2025-07-23 arXiv v3 · ACL 2025 | L1 OPD/自蒸馏 + L2 统一对齐 · 相关性高

**原始论文**：https://arxiv.org/abs/2503.02832

## 一眼看懂
- 🟦 TL;DR：把"带 token-level reward 的 RLHF 目标"在理论上**等价改写成一个蒸馏问题**（§3 Theorem 1）——其 teacher 分布是 DPO 模型与 reference 模型 logit 的线性组合。于是 RLHF 对齐就变成"向一个自动合成的 teacher 分布做 token 级 KL 蒸馏"，无需训练显式 reward model，且 token 级监督让收敛更快(§6.5 比 sentence-level 快 >2×)。
- 最巧的一步：**逐 token 自适应外插权重 \(\alpha_t = D_{\mathrm{TVD}}(t)\cdot r + \epsilon\)**（§4.2 Eq.16-17）。抽掉它(改用常数外插)就垮——常数权重要么太小(欠优化)要么太大(过优化、回复暴长)，§6.2 Table 3 显示常数 \(\frac{\beta_0}{\beta}=2.0\) 虽 AE2 LC 25.17 高但回复长度从 2332 飙到 4434、\(D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{dpo}})\) 从 10.16 飙到 34.52；\(\alpha_t\) 用"两个 DPO 在该 token 分歧大→外插更激进"自动平衡，KL 22.95/长度 2424 拿到 LC 21.16。

## 为什么做
- 研究背景：LLM 对齐两条主线——
  - **RLHF**（两阶段）：先用 Bradley-Terry 训 response-level reward model \(L_{\mathrm{RM}}(\phi)=-\mathbb{E}_{(x,y_w,y_l)\sim D}[\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))]\)（Eq.1），再 PPO 优化策略并加 KL 约束 \(J_{\mathrm{RLHF}}(\theta)=\max_\theta\mathbb{E}_{x\sim D,\,y\sim\pi_\theta}[r_\phi(x,y)-\beta\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}]\)（Eq.2）。
  - **DPO**（直接偏好学习）：用策略自身参数化 reward \(r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}+\beta\log Z(x)\)（Eq.3），代入 BT 损失得 \(L_{\mathrm{DPO}}\)（Eq.4），省掉显式 RL。Rafailov 等(2024a)进一步证明 DPO 自带可 token 级分解的隐式 reward \(r_{\mathrm{dpo}}(x,y_{<t},y_t)=\beta\log\frac{\pi_{\mathrm{dpo}}(y_t|y_{<t},x)}{\pi_{\mathrm{ref}}(y_t|y_{<t},x)}\)（Eq.6）。
- 解决的具体痛点：现有对齐多用**稀疏的 response-level reward/偏好标注**优化整条回复里所有 token，**粒度太粗**——会错误惩罚好回复里的高质量 token、或鼓励差回复里的低质量 token，拖慢收敛、限制上限（§1，引 Yoon/Li/Xia/Yang 2024）。而 DPO 自带的 token-level reward 虽可分解，但**精度不如纯 reward model**（§4.1，引 Lin et al. 2024，并在 Table 2 复现：DPO test acc 69.53% < RM 71.19%）。
- 相关工作 & 并行技术路线（每条具体短板，§7 + §5.2）：
  - **细粒度对齐线（fine-grained alignment）**：(a) PRM/过程奖励——Lightman 2023 用 step 级人工标注，Math-Shepherd/Luo/Yuan 2024 自动收 step 奖励，短板=需额外 PRM 或采样成本；(b) span/token 级——Cao 2024 从 critique 抽 span 奖励、Chan 2024 用 reward model attention、Li 2024a 用中间 token 输出、Guo/Chen 2024b 用 edit distance，短板=多为**标量**信号；(c) 用 DPO token 奖励——TIS-DPO/Yang 2024c 进 DPO、RTO(Zhong 2024)进 PPO、Inverse-Q*(Xia 2024)新算法。AlignDistil 站在这条线上但差异是**用整个 reward 分布而非标量**。
  - **token 级 DPO 变体**：TDPO1/2(Zeng 2024)给 DPO 加 token 级 forward-KL 约束，短板：§5.3 实测该约束在小模型上**反而限制性能**；SimPO(Meng 2024)去 reference 简化，KTO(Ethayarajh 2024)用非配对数据——均仍 response 级。
  - **最强基线 RTO**(Zhong 2024)：把 token-level DPO reward 塞进 PPO，§5.3 称其强(超 PPO)，但仍是**标量** token reward，丢失分布信息、收敛慢(§6.5)。
  - **LLM 知识蒸馏线**(§7)：white-box KD(MiniLLM/DistiLLM/DSKD，对齐分布或中间特征)、black-box KD(Alpaca/Vicuna/Zephyr，只收 teacher 输出做 SFT)。AlignDistil 与它们的差异：teacher 不是现成大模型，而是**由 RLHF 目标推导出的合成分布**，目标是 token 级 reward 优化而非压缩。
- 动机链：想要 token-level 奖励优化 + 不想额外训昂贵 reward model → 注意到 DPO reward 可 token 级分解 → 设问能否把"带 DPO token reward 的 RLHF"直接转成蒸馏 → Theorem 1 证明可以 → 但 vanilla DPO reward 不准(Eq.12 加 reverse DPO) + 常数外插不稳(Eq.16 加自适应 \(\alpha_t\))。为什么不直接用 RTO？因为 RTO 用标量 token reward，丢失分布信息、收敛慢(§6.5 实测 >2×)。
- 与最近邻工作的精确差异：相对 RTO（DPO reward 进 PPO），差在**蒸馏整个 teacher 分布**(信息更全、收敛 >2× 快)；相对 TDPO，差在**外插越过 DPO 而非把策略约束在其附近**；相对 Liu et al. 2024b(decoding-time 外插)，差在**把外插搬进训练 + 逐 token 自适应**。

## 怎么做 + 靠不靠谱
### 0. 核心理论桥：RLHF ⇔ 蒸馏（§3 Theorem 1，全文地基）
- 第一步直觉：把 DPO reward \(r_{\mathrm{dpo}}(x,y)=\beta_0\log\frac{\pi_{\mathrm{dpo}}(y|x)}{\pi_{\mathrm{ref}}(y|x)}\)（Eq.5，省去与 \(y\) 无关的 \(Z(x)\)）代入 RLHF 目标 Eq.2，得
  \(\displaystyle \widetilde{J}_{\mathrm{RLHF}}(\theta)=\max_\theta\mathbb{E}_{x\sim D,\,y\sim\pi_\theta(\cdot|x)}\Big[\underbrace{\beta_0\log\tfrac{\pi_{\mathrm{dpo}}(y|x)}{\pi_{\mathrm{ref}}(y|x)}}_{\text{DPO reward}}-\underbrace{\beta\log\tfrac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}}_{\text{KL divergence}}\Big] \quad(\text{Eq.8})\)
  其中 \(\beta_0\) 是 DPO 训练时的系数(常数)，\(\beta\) 是当前 KL 系数。
- **Theorem 1**：上式（Eq.9 / Eq.8）严格等价于一个 token 级蒸馏目标
  \(\displaystyle \widetilde{J}_{\mathrm{RLHF}}(\theta)=\min_\theta\mathbb{E}_{x\sim D,\,y\sim\pi_\theta}\;\frac{\beta}{|y|}\sum_{t=1}^{|y|}D_{\mathrm{KL}}\big(\pi_\theta(\cdot|y_{<t},x)\,\|\,\pi^*(\cdot|y_{<t},x)\big) \quad(\text{Eq.10})\)
  其中 teacher 分布 \(\pi^*\) 是对一个**合成 logit** 做 softmax，合成 logit 为 DPO 与 reference logit 的线性组合：
  \(\displaystyle z_t^*=\frac{\beta_0}{\beta}\,z_t^{\mathrm{dpo}}+\Big(1-\frac{\beta_0}{\beta}\Big)z_t^{\mathrm{ref}} \quad(\text{Eq.11})\)
  直觉：把"对齐"诠释为"向一个比 DPO 更靠 reward 方向的合成 teacher 做 token 级反向 KL 蒸馏"。证明见附录 A（把 Eq.8 的期望逐 token 展开、配方成 KL）。

### 1. AlignDistil 两个工程设计（§4）
- **设计一：Contrastive DPO Reward（管 reward 准不准，§4.1）**
  - 问题：vanilla DPO reward 泛化差于纯 RM（Lin 2024 + Table 2 复现）。
  - 做法：用一对对比 DPO 模型——正常 DPO \(\pi_{\mathrm{dpo}}\) + **reverse DPO** \(\pi_{\mathrm{dpo}}^-\)（把训练数据的 chosen/rejected **对调**后训，专门捕捉低质量数据的负面特征）。对比 reward \(r_{\mathrm{ctr}}(x,y)=\beta_0\log\frac{\pi_{\mathrm{dpo}}(y|x)}{\pi_{\mathrm{dpo}}^-(y|x)}\)（Eq.12）。
  - 配套技巧：把 RLHF 的 reference 从初始模型**前移成 \(\pi_{\mathrm{dpo}}\)**（既省一个模型，又把参考基线推得更对齐）。于是目标变 Eq.13，合成 logit 变
    \(\displaystyle z_t^*=\Big(1+\tfrac{\beta_0}{\beta}\Big)z_t^{\mathrm{dpo}}-\tfrac{\beta_0}{\beta}z_t^{\mathrm{dpo}-}=\underbrace{z_t^{\mathrm{dpo}}}_{\text{DPO 分布}}+\underbrace{\tfrac{\beta_0}{\beta}\big(z_t^{\mathrm{dpo}}-z_t^{\mathrm{dpo}-}\big)}_{\text{reward 分布}} \quad(\text{Eq.14-15})\)
    因 \(\beta_0,\beta>0\)，这严格是"在 forward DPO 基础上沿 `forward − reverse` 差分方向**再外推一段**"——构造比单纯 DPO 更对齐的分布，推策略**越过** DPO。
- **设计二：Token Adaptive Logit Extrapolation（管稳不稳，§4.2）**
  - 问题：固定 \(\frac{\beta_0}{\beta}\) 难调——大 \(\beta\)(小 \(\frac{\beta_0}{\beta}\))欠优化，小 \(\beta\)(大 \(\frac{\beta_0}{\beta}\))过优化、回复暴长。
  - 做法：用两 DPO 分布的 **TVD** 算逐 token 权重
    \(\displaystyle \alpha_t=D_{\mathrm{TVD}}(t)\cdot r+\epsilon\in[\epsilon,\,r+\epsilon],\quad D_{\mathrm{TVD}}(t):=\tfrac{1}{2}\sum_{y_t\in V}\big|\pi_{\mathrm{dpo}}(y_t|y_{<t},x)-\pi_{\mathrm{dpo}}^-(y_t|y_{<t},x)\big| \quad(\text{Eq.16})\)
    其中 \(r\) 控外插上界、\(\epsilon=0.001\) 防 \(\alpha_t=0\)。选 TVD 因其对称、计算高效且值域 \([0,1]\)。直觉：两 DPO 分歧大的 token 对最终 reward 影响大→该位置用更强 teacher。
  - 用 \(\alpha_t\) 替换常数得逐 token teacher \(z_t^*=z_t^{\mathrm{dpo}}+\alpha_t(z_t^{\mathrm{dpo}}-z_t^{\mathrm{dpo}-})\)（Eq.17），对应 \(\beta_t=\frac{\beta_0}{\alpha_t}\) 也随之自适应。

### 2. 训练目标与数据流动（§4.3）
- **on-policy**（Eq.18）：用当前策略采样 \(\hat y\sim\pi_\theta(\cdot|x)\)，Monte-Carlo 估期望：
  \(\displaystyle L_{\mathrm{AD}}^{\mathrm{on}}=\frac{1}{|B|}\sum_{x\in B}\frac{\beta_t}{|\hat y|}\sum_{t=1}^{|\hat y|}D_{\mathrm{KL}}\big(\pi_\theta(\cdot|\hat y_{<t},x)\,\|\,\pi^*(\cdot|\hat y_{<t},x)\big)\)
- **off-policy**（Eq.19）：用现成 prompt-response 数据集 \(\{(x,y)\}\)，把 \(\hat y\) 换成数据集里的 \(y\)。
- 数据流动：UltraFeedback 的 prompt+response pair 先训 \(\pi_{\mathrm{dpo}}\) 与 \(\pi_{\mathrm{dpo}}^-\)；on-policy 训练只用 prompt(策略自采样)，off-policy 用 prompt+chosen response；每步前向 \(\pi_{\mathrm{dpo}}/\pi_{\mathrm{dpo}}^-\) 合成 teacher logit → 算 token 级反向 KL → 更新 \(\pi_\theta\)。
- **关键超参与默认值**(§5.1)：1 epoch、batch 128、lr 1e-6、warmup 0.1；\(\epsilon=0.001\)；§6.5 收敛实验 \(\beta=0.08\)；外插上界 \(r\) 与 \(\beta_2\) 见附录 C；8×A100-40G。

### 3. 逐组件必要性（消融）
- **RLHF⇔蒸馏等价(Theorem 1)**：理论地基；没它就没有"对齐=蒸馏"的合法性与 teacher 构造式 Eq.11/15。
- **contrastive DPO reward**：§6.1 Table 2，contrastive test acc 71.29% > vanilla DPO 69.53% > 甚至 > RM 71.19%；对应 on-policy AE2 LC 19.45 vs vanilla 16.51。机制：reverse DPO 捕捉负面特征 + 隐式翻倍可训练参数。没它 reward 不准。
- **token 自适应 \(\alpha_t\)**：§6.2 Table 3（off-policy 隔离数据影响），\(\alpha_t\) vs 各常数——\(\alpha_t\) 在 KL 22.95/长度 2424 拿到 LC 21.16，平衡最好；常数 1.0 欠优化(LC 18.40)、常数 2.0 过优化(长度 4434)。另：§5.3 用 `DPOβ=0.01`(等价 rescale \(\beta\) 做简单外插)对照，发现单纯 rescale 不稳定(Qwen2.5-1.5B 上不涨)，反证两设计必要。
- **on-policy vs off-policy**：§5.3 结论 3——on-policy 数据贴近策略(分布内)通常更好，off-policy 更高效且具竞争力；两版都给，灵活权衡。

### 4. 实验与证据
- 初始模型 Qwen2-1.5B-Instruct、Qwen2.5-1.5B-Instruct；§6.3 另加 Qwen2.5-7B(SFT on UltraChat-200k)、Llama3-8B(用 SimPO 开源 SFT ckpt)。数据 UltraFeedback(63K)，评测 AlpacaEval 2.0(LC WR/WR)、MT-Bench、Arena-Hard(WR/SC WR)，裁判 Qwen2.5-72B-Instruct(§表6 验证其判断与 GPT-4 相当但便宜)。
- 关键数字(Table 1)：on/off-policy 两版均显著超基线(DPO/KTO/SimPO/TDPO1-2/PPO/RTO 共 8 个)，AE2 LC 较 DPO **>6%**(Qwen2-1.5B DPO 6.42→on-policy 12.93、off-policy 11.79)；Qwen2.5-1.5B off-policy LC 21.16 vs DPO 14.35。§6.3 7B/8B 上同样大幅领先(Qwen2.5-7B on-policy LC 31.32 vs TDPO1 26.42)。§6.4 TL;DR win-rate 92.5%/92.8% 超 PPO 87.7%、RTO 90.8%。§6.5 收敛 >2× 快于 token 标量、远快于 sentence-level(Fig 2)，因蒸馏整个分布可在每 token 位置精确算 reward 期望。baseline 覆盖全面，公平。
- 假设与失效边界：【原文】Limitations——评测限于小模型(~1.5B)，更大模型未充分探索(虽 §6.3 补了 7B/8B)。【推断】"外插越过 DPO"假设 DPO 改进方向在更大步长上仍正确，过度外插会放大噪声(靠 \(\epsilon\) 下限 + \(\alpha_t\) 缓解，但无理论上界)；需训 3 个模型(DPO/reverse DPO/最终策略)，成本不低；评测偏对话/指令，未验证数学/代码推理域。
- 祛魅总结：【推断】真贡献是"RLHF + DPO reward ⇔ token 级蒸馏"这一干净的理论桥 + 两个有消融支撑的工程设计(contrastive reward、自适应外插)。包装较实在，理论(Eq.8-17)与代码(v1 已核对 aligndistil_trainer.py)一一对应。作者**可能高估**了泛化性(规模/域有限)；**低估**了三模型训练的工程成本与超参(\(\beta/\beta_2/r\))敏感性。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=合成 teacher 分布 \(\pi^*\)(DPO 与 reverse-DPO 外插)的 token 级 \(D_{\mathrm{KL}}\)；隐含 token 级 distributional reward | **改什么**=策略模型 logits/参数 | **何时改**=在线(on-policy，per-step 用当前策略采样)或离线(off-policy 用偏好数据) | **免梯度?**=否(梯度蒸馏) | **记忆-技能生命周期**=不适用 | **防遗忘机制**=KL 形蒸馏天然把策略约束在 teacher(含 DPO/ref 信息)附近，间接抑制偏离；无专门持续学习设计。
- ⑦ 开源代码+框架/harness：https://github.com/songmzhang/AlignDistil （v1 已克隆约 2.4M，**完整**，自带定制版 OpenRLHF 子目录）。框架=**OpenRLHF v0.5.2.post2**(on-policy 另需 vLLM)。核心 loss 在 `openrlhf/trainer/aligndistil_trainer.py`(`reward_boost_type=="aligndistil"`)，训练脚本 `train_scripts/ultrafeedback/qwen2.5-1.5b/`(dpo/reverse_dpo/aligndistil_off_policy/on_policy)。
- 💰 资源/成本与可扩展性：【原文】§5.1 全部实验在 8×A100-40G；1 epoch、batch 128、lr 1e-6。【推断】需依次训 DPO + reverse DPO + 最终策略共 3 段，外加 on-policy 采样开销，总成本约 PPO 的数倍但单段轻量。
- 🎯 对"探索-巩固"对标：**竞品/可借组件**——属 L1 on-policy distillation 正统(on-policy 版用当前策略采样向 teacher 蒸馏，与本项目 OPD 主轴**直接同族**)。可借：①"teacher = 两模型 logit 自适应外插"构造比单 teacher 更强的引导分布；② **逐 token 自适应权重 \(\alpha_t\)(用分歧度 TVD 调引导强度)** 可迁移到 TSRD"在岔路口/关键步动态调脚手架强度"。**缺口**——AlignDistil 是对齐(helpfulness)任务、teacher 由 DPO 合成而非"会走通的 teacher 轨迹"，无显式探索/路径恢复语义，无 MTP 前瞻。一句判定：**强相关、可直接借鉴的 OPD 实例 + 自适应引导强度机制**。依据：on-policy 蒸馏 + token 级自适应 teacher 强度正是 TSRD"稀疏脚手架按需调强"的现成数学载体。
- 🔭 开放问题/未来方向：【原文】Limitations——把方法扩到更大模型。【推断】把"RLHF⇔蒸馏 + 自适应外插"从对齐迁移到 RLVR/推理(teacher 换成会走通的强推理模型)；\(\alpha_t\) 的 TVD 信号可作"关键步识别"的 proxy；外插步长的理论上界/自适应 \(r\)；与显式 reward 蒸馏的混合。

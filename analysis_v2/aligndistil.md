aligndistil | AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation | 北京交通大学(交通大数据与AI教育部重点实验室) + 腾讯（Songming Zhang 实习于腾讯，通讯 Yufeng Chen / Jinan Xu） | 2025-07-23 arXiv v3 · ACL 2025 | L1 OPD/自蒸馏 + L2 统一对齐 · 相关性高

**原始论文**：https://arxiv.org/abs/2503.02832

## 一眼看懂
> 一句话导读：这篇把"做 RLHF 对齐"在数学上证明等于"向一个自动拼出来的 teacher 做 token 级蒸馏"，从而省掉显式奖励模型；真正的关键技巧是一个会随位置自动调强弱的外插权重。

- 🟦 TL;DR：本文把"带 token-level reward 的 RLHF 目标"在理论上**等价改写成一个蒸馏问题**（§3 Theorem 1）。这里的 teacher 分布并非现成大模型，而是 DPO 模型与 reference 模型的 logit 做线性组合后合成出来的。于是 RLHF 对齐就变成一件事：向这个自动合成的 teacher 分布做 token 级 KL 蒸馏。好处有二——无需训练显式 reward model；token 级监督让收敛更快(§6.5 比 sentence-level 快 >2×)。
- 最巧的一步：**逐 token 自适应外插权重 \(\alpha_t = D_{\mathrm{TVD}}(t)\cdot r + \epsilon\)**（§4.2 Eq.16-17）。这个权重控制"teacher 比 DPO 再往 reward 方向推多远"。抽掉它、改用一个常数外插就会垮：常数太小则欠优化，常数太大则过优化、回复暴长。
  - 证据(§6.2 Table 3)：常数 \(\frac{\beta_0}{\beta}=2.0\) 时，AE2 LC 25.17 看着高，但回复长度从 2332 飙到 4434、\(D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{dpo}})\) 从 10.16 飙到 34.52——明显过优化。
  - 自适应权重的逻辑是"两个 DPO 在某 token 上分歧越大，就在该位置外插得越激进"，借此自动平衡。最终 KL 22.95、长度 2424，拿到 LC 21.16。

## 为什么做
> 一句话导读：现有对齐的奖励信号太粗（一整条回复一个分数），而 DPO 虽能拆到 token 级却不够准；本文想要"token 级、又不用额外训奖励模型"，于是发现可以把 RLHF 整个转成蒸馏。

- 研究背景：LLM 对齐有两条主线。
  - **RLHF**（两阶段）：第一阶段用 Bradley-Terry 训一个 response-level reward model \(L_{\mathrm{RM}}(\phi)=-\mathbb{E}_{(x,y_w,y_l)\sim D}[\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))]\)（Eq.1）；第二阶段用 PPO 优化策略并加 KL 约束 \(J_{\mathrm{RLHF}}(\theta)=\max_\theta\mathbb{E}_{x\sim D,\,y\sim\pi_\theta}[r_\phi(x,y)-\beta\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}]\)（Eq.2）。
  - **DPO**（直接偏好学习）：思路是省掉显式 RL。它用策略自身参数化 reward \(r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}+\beta\log Z(x)\)（Eq.3），代入 BT 损失即得 \(L_{\mathrm{DPO}}\)（Eq.4）。Rafailov 等(2024a)进一步证明，DPO 自带一个可按 token 分解的隐式 reward \(r_{\mathrm{dpo}}(x,y_{<t},y_t)=\beta\log\frac{\pi_{\mathrm{dpo}}(y_t|y_{<t},x)}{\pi_{\mathrm{ref}}(y_t|y_{<t},x)}\)（Eq.6）——也就是说 DPO 隐含了 token 级信号。
- 解决的具体痛点（两个）：
  - 痛点一——粒度太粗。现有对齐多用**稀疏的 response-level reward/偏好标注**去优化整条回复里的所有 token。这会错误惩罚好回复里的高质量 token，或反过来鼓励差回复里的低质量 token，从而拖慢收敛、限制性能上限（§1，引 Yoon/Li/Xia/Yang 2024）。
  - 痛点二——DPO 的 token 信号不够准。DPO 自带的 token-level reward 虽然可分解，但**精度不如纯 reward model**（§4.1，引 Lin et al. 2024）。Table 2 复现了这一点：DPO test acc 69.53% < RM 71.19%。
- 相关工作 & 并行技术路线（逐条点出短板，§7 + §5.2）：
  - **细粒度对齐线（fine-grained alignment）**，分三支：
    - (a) PRM/过程奖励：Lightman 2023 用 step 级人工标注，Math-Shepherd/Luo/Yuan 2024 自动收 step 奖励。短板=需额外 PRM 或采样成本。
    - (b) span/token 级：Cao 2024 从 critique 抽 span 奖励，Chan 2024 用 reward model attention，Li 2024a 用中间 token 输出，Guo/Chen 2024b 用 edit distance。短板=这些信号多为**标量**。
    - (c) 用 DPO token 奖励：TIS-DPO/Yang 2024c 把它接进 DPO，RTO(Zhong 2024)接进 PPO，Inverse-Q*(Xia 2024)做成新算法。
    - AlignDistil 也站在这条线上，差异是**用整个 reward 分布而非标量**。
  - **token 级 DPO 变体**：TDPO1/2(Zeng 2024)给 DPO 加 token 级 forward-KL 约束，但 §5.3 实测该约束在小模型上**反而限制性能**；SimPO(Meng 2024)去掉 reference 做简化；KTO(Ethayarajh 2024)用非配对数据。这几个仍都是 response 级。
  - **最强基线 RTO**(Zhong 2024)：把 token-level DPO reward 塞进 PPO，§5.3 称其很强(超 PPO)。但它仍是**标量** token reward，丢失了分布信息、收敛慢(§6.5)。
  - **LLM 知识蒸馏线**(§7)：包括 white-box KD(MiniLLM/DistiLLM/DSKD，对齐分布或中间特征)与 black-box KD(Alpaca/Vicuna/Zephyr，只收 teacher 输出做 SFT)。AlignDistil 与它们的差异：teacher 不是现成大模型，而是**由 RLHF 目标推导出的合成分布**，目标也不是压缩，而是 token 级 reward 优化。
- 动机链（一步步推下来）：想要 token-level 奖励优化、又不想额外训昂贵的 reward model；注意到 DPO reward 可按 token 分解；于是设问——能否把"带 DPO token reward 的 RLHF"直接转成蒸馏；Theorem 1 证明可以；但还有两个坑要补——vanilla DPO reward 不准(于是 Eq.12 加 reverse DPO)、常数外插不稳(于是 Eq.16 加自适应 \(\alpha_t\))。至于为什么不直接用 RTO：因为 RTO 用标量 token reward，丢失分布信息、收敛慢(§6.5 实测 >2×)。
- 与最近邻工作的精确差异（逐个对比）：
  - 相对 RTO（DPO reward 进 PPO）：差在**蒸馏整个 teacher 分布**，信息更全、收敛 >2× 快。
  - 相对 TDPO：差在**外插越过 DPO，而非把策略约束在 DPO 附近**。
  - 相对 Liu et al. 2024b（decoding-time 外插）：差在**把外插搬进训练 + 逐 token 自适应**。

## 怎么做 + 靠不靠谱
> 一句话导读：先证一个等价定理把 RLHF 变成蒸馏（第 0 节），再补两块工程设计让信号更准、外插更稳（第 1 节），最后给 on/off-policy 两套训练目标（第 2 节）；后面用消融逐个验证哪块拿掉就垮。

### 0. 核心理论桥：RLHF ⇔ 蒸馏（§3 Theorem 1，全文地基）
- 第一步直觉：把 DPO reward 代入 RLHF 目标。DPO reward 写作 \(r_{\mathrm{dpo}}(x,y)=\beta_0\log\frac{\pi_{\mathrm{dpo}}(y|x)}{\pi_{\mathrm{ref}}(y|x)}\)（Eq.5，这里省去了与 \(y\) 无关的归一项 \(Z(x)\)）。把它代入 RLHF 目标 Eq.2，得
  \(\displaystyle \widetilde{J}_{\mathrm{RLHF}}(\theta)=\max_\theta\mathbb{E}_{x\sim D,\,y\sim\pi_\theta(\cdot|x)}\Big[\underbrace{\beta_0\log\tfrac{\pi_{\mathrm{dpo}}(y|x)}{\pi_{\mathrm{ref}}(y|x)}}_{\text{DPO reward}}-\underbrace{\beta\log\tfrac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}}_{\text{KL divergence}}\Big] \quad(\text{Eq.8})\)
  其中 \(\beta_0\) 是 DPO 训练时用的系数(常数)，\(\beta\) 是当前的 KL 系数。
- **Theorem 1**：上式（Eq.9 / Eq.8）严格等价于一个 token 级蒸馏目标
  \(\displaystyle \widetilde{J}_{\mathrm{RLHF}}(\theta)=\min_\theta\mathbb{E}_{x\sim D,\,y\sim\pi_\theta}\;\frac{\beta}{|y|}\sum_{t=1}^{|y|}D_{\mathrm{KL}}\big(\pi_\theta(\cdot|y_{<t},x)\,\|\,\pi^*(\cdot|y_{<t},x)\big) \quad(\text{Eq.10})\)
  这里的 teacher 分布 \(\pi^*\) 是对一个**合成 logit** 做 softmax 得到的；而合成 logit 就是 DPO 与 reference 两个 logit 的线性组合：
  \(\displaystyle z_t^*=\frac{\beta_0}{\beta}\,z_t^{\mathrm{dpo}}+\Big(1-\frac{\beta_0}{\beta}\Big)z_t^{\mathrm{ref}} \quad(\text{Eq.11})\)
  直觉：所谓"对齐"，被诠释成"向一个比 DPO 更靠 reward 方向的合成 teacher，做 token 级反向 KL 蒸馏"。这里 reverse-KL 指让学生分布去逼近 teacher，而非反向。证明见附录 A，做法是把 Eq.8 的期望逐 token 展开、再配方成 KL 形式。

### 1. AlignDistil 两个工程设计（§4）
- **设计一：Contrastive DPO Reward（管 reward 准不准，§4.1）**
  - 问题：vanilla DPO reward 的泛化能力差于纯 RM（Lin 2024，并经 Table 2 复现）。
  - 做法：用一对对比 DPO 模型。一个是正常 DPO \(\pi_{\mathrm{dpo}}\)，另一个是 **reverse DPO** \(\pi_{\mathrm{dpo}}^-\)。reverse DPO 的训法是把训练数据的 chosen/rejected **对调**后再训，专门用来捕捉低质量数据的负面特征。两者相减即对比 reward \(r_{\mathrm{ctr}}(x,y)=\beta_0\log\frac{\pi_{\mathrm{dpo}}(y|x)}{\pi_{\mathrm{dpo}}^-(y|x)}\)（Eq.12）。
  - 配套技巧：把 RLHF 的 reference 从初始模型**前移成 \(\pi_{\mathrm{dpo}}\)**。这一步一举两得——既省下一个模型，又把参考基线推得更对齐。于是目标变成 Eq.13，合成 logit 变为
    \(\displaystyle z_t^*=\Big(1+\tfrac{\beta_0}{\beta}\Big)z_t^{\mathrm{dpo}}-\tfrac{\beta_0}{\beta}z_t^{\mathrm{dpo}-}=\underbrace{z_t^{\mathrm{dpo}}}_{\text{DPO 分布}}+\underbrace{\tfrac{\beta_0}{\beta}\big(z_t^{\mathrm{dpo}}-z_t^{\mathrm{dpo}-}\big)}_{\text{reward 分布}} \quad(\text{Eq.14-15})\)
    因为 \(\beta_0,\beta>0\)，这一式严格表示"在 forward DPO 的基础上，沿 forward 减 reverse 的差分方向**再外推一段**"。其效果是构造出比单纯 DPO 更对齐的分布，把策略推得**越过** DPO。
- **设计二：Token Adaptive Logit Extrapolation（管稳不稳，§4.2）**
  - 问题：固定的 \(\frac{\beta_0}{\beta}\) 很难调。\(\beta\) 偏大（即 \(\frac{\beta_0}{\beta}\) 偏小）会欠优化；\(\beta\) 偏小（即 \(\frac{\beta_0}{\beta}\) 偏大）则过优化、回复暴长。
  - 做法：改用两个 DPO 分布之间的 **TVD（总变差距离，衡量两分布的差异）**来算逐 token 的权重：
    \(\displaystyle \alpha_t=D_{\mathrm{TVD}}(t)\cdot r+\epsilon\in[\epsilon,\,r+\epsilon],\quad D_{\mathrm{TVD}}(t):=\tfrac{1}{2}\sum_{y_t\in V}\big|\pi_{\mathrm{dpo}}(y_t|y_{<t},x)-\pi_{\mathrm{dpo}}^-(y_t|y_{<t},x)\big| \quad(\text{Eq.16})\)
    其中 \(r\) 控制外插上界，\(\epsilon=0.001\) 用来防止 \(\alpha_t=0\)。之所以选 TVD，是因为它对称、计算高效、且值域落在 \([0,1]\)。直觉：两个 DPO 分歧越大的 token，对最终 reward 影响越大，所以该位置就用更强的 teacher。
  - 用 \(\alpha_t\) 替换常数，得到逐 token 的 teacher \(z_t^*=z_t^{\mathrm{dpo}}+\alpha_t(z_t^{\mathrm{dpo}}-z_t^{\mathrm{dpo}-})\)（Eq.17）；对应的 \(\beta_t=\frac{\beta_0}{\alpha_t}\) 也随之逐 token 自适应。

### 2. 训练目标与数据流动（§4.3）
- **on-policy 版**（Eq.18）：用当前策略采样 \(\hat y\sim\pi_\theta(\cdot|x)\)，再用 Monte-Carlo 估期望：
  \(\displaystyle L_{\mathrm{AD}}^{\mathrm{on}}=\frac{1}{|B|}\sum_{x\in B}\frac{\beta_t}{|\hat y|}\sum_{t=1}^{|\hat y|}D_{\mathrm{KL}}\big(\pi_\theta(\cdot|\hat y_{<t},x)\,\|\,\pi^*(\cdot|\hat y_{<t},x)\big)\)
- **off-policy 版**（Eq.19）：改用现成 prompt-response 数据集 \(\{(x,y)\}\)，把上式里的 \(\hat y\) 换成数据集里的 \(y\)。
- 数据流动（一条流程）：先用 UltraFeedback 的 prompt+response pair 训出 \(\pi_{\mathrm{dpo}}\) 与 \(\pi_{\mathrm{dpo}}^-\)。训练时，on-policy 版只用 prompt（策略自己采样），off-policy 版用 prompt+chosen response。每一步：前向 \(\pi_{\mathrm{dpo}}/\pi_{\mathrm{dpo}}^-\) 合成 teacher logit，算 token 级反向 KL，更新 \(\pi_\theta\)。
- **关键超参与默认值**(§5.1)：1 epoch、batch 128、lr 1e-6、warmup 0.1；\(\epsilon=0.001\)；§6.5 收敛实验用 \(\beta=0.08\)；外插上界 \(r\) 与 \(\beta_2\) 见附录 C；硬件 8×A100-40G。

### 3. 逐组件必要性（消融）
- **RLHF⇔蒸馏等价(Theorem 1)**：这是理论地基。没有它，就没有"对齐=蒸馏"的合法性，也没有 teacher 的构造式 Eq.11/15。
- **contrastive DPO reward**：§6.1 Table 2 显示，contrastive 的 test acc 71.29%，高于 vanilla DPO 69.53%，甚至高于纯 RM 71.19%；对应 on-policy AE2 LC 19.45 vs vanilla 16.51。机制有两点——reverse DPO 捕捉负面特征，以及隐式地把可训练参数翻倍。没有它，reward 就不准。
- **token 自适应 \(\alpha_t\)**：§6.2 Table 3（用 off-policy 来隔离数据影响），把 \(\alpha_t\) 与各种常数对比。\(\alpha_t\) 在 KL 22.95、长度 2424 下拿到 LC 21.16，平衡最好；常数 1.0 欠优化(LC 18.40)，常数 2.0 过优化(长度 4434)。另一条对照在 §5.3：用 `DPOβ=0.01`（等价于 rescale \(\beta\) 做简单外插），发现单纯 rescale 不稳定（在 Qwen2.5-1.5B 上不涨），反证这两个设计确有必要。
- **on-policy vs off-policy**：§5.3 结论 3——on-policy 数据更贴近策略(分布内)，通常效果更好；off-policy 更高效，也有竞争力。两版都提供，供灵活权衡。

### 4. 实验与证据
- 实验设置：初始模型用 Qwen2-1.5B-Instruct、Qwen2.5-1.5B-Instruct；§6.3 另加 Qwen2.5-7B（在 UltraChat-200k 上 SFT）、Llama3-8B（用 SimPO 开源的 SFT ckpt）。数据用 UltraFeedback(63K)。评测覆盖 AlpacaEval 2.0(LC WR/WR)、MT-Bench、Arena-Hard(WR/SC WR)，裁判用 Qwen2.5-72B-Instruct（§表6 验证其判断与 GPT-4 相当，但更便宜）。
- 关键数字(Table 1)：on/off-policy 两版都显著超过 8 个基线(DPO/KTO/SimPO/TDPO1-2/PPO/RTO)。
  - AE2 LC 较 DPO 提升 **>6%**：Qwen2-1.5B 上 DPO 6.42 → on-policy 12.93、off-policy 11.79；Qwen2.5-1.5B off-policy LC 21.16 vs DPO 14.35。
  - 大模型同样领先：§6.3 中 Qwen2.5-7B on-policy LC 31.32 vs TDPO1 26.42。
  - §6.4 TL;DR 任务 win-rate 92.5%/92.8%，超过 PPO 87.7%、RTO 90.8%。
  - §6.5 收敛速度 >2× 快于 token 标量信号、远快于 sentence-level(Fig 2)。原因：蒸馏整个分布可以在每个 token 位置精确算出 reward 期望。
  - baseline 覆盖全面，比较公平。
- 假设与失效边界：
  - 【原文】Limitations——评测限于小模型(~1.5B)，更大模型未充分探索（虽然 §6.3 补了 7B/8B）。
  - 【推断】"外插越过 DPO"这一步，隐含假设 DPO 的改进方向在更大步长上仍然正确；过度外插会放大噪声（靠 \(\epsilon\) 下限 + \(\alpha_t\) 缓解，但没有理论上界）。
  - 【推断】需要训 3 个模型(DPO / reverse DPO / 最终策略)，成本不低。
  - 【推断】评测偏对话/指令，未验证数学/代码推理域。
- 祛魅总结：【推断】真正的贡献是"RLHF + DPO reward ⇔ token 级蒸馏"这一干净的理论桥，加上两个有消融支撑的工程设计(contrastive reward、自适应外插)。包装比较实在，理论(Eq.8-17)与代码(v1 已核对 aligndistil_trainer.py)能一一对应。要注意的是：作者**可能高估**了泛化性(规模/域都有限)，也**低估**了三模型训练的工程成本与超参(\(\beta/\beta_2/r\))的敏感性。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**=合成 teacher 分布 \(\pi^*\)（由 DPO 与 reverse-DPO 外插得到）的 token 级 \(D_{\mathrm{KL}}\)；其中隐含一个 token 级的分布型 reward。
  - **改什么**=策略模型的 logits/参数。
  - **何时改**=在线（on-policy，每步用当前策略采样）或离线（off-policy，用偏好数据）。
  - **免梯度?**=否，是梯度蒸馏。
  - **记忆-技能生命周期**=不适用。
  - **防遗忘机制**=KL 形蒸馏天然把策略约束在 teacher 附近（teacher 已含 DPO/ref 信息），间接抑制偏离；没有专门的持续学习设计。
- ⑦ 开源代码+框架/harness：https://github.com/songmzhang/AlignDistil 。
  - v1 已克隆约 2.4M，**完整**，自带一个定制版的 OpenRLHF 子目录。
  - 框架=**OpenRLHF v0.5.2.post2**；on-policy 另需 vLLM。
  - 核心 loss 在 `openrlhf/trainer/aligndistil_trainer.py`（走 `reward_boost_type=="aligndistil"` 分支）；训练脚本在 `train_scripts/ultrafeedback/qwen2.5-1.5b/`（含 dpo / reverse_dpo / aligndistil_off_policy / on_policy）。
- 💰 资源/成本与可扩展性：
  - 【原文】§5.1 全部实验在 8×A100-40G；1 epoch、batch 128、lr 1e-6。
  - 【推断】需依次训 DPO + reverse DPO + 最终策略共 3 段，再加上 on-policy 采样开销；总成本约为 PPO 的数倍，但每一段都较轻量。
- 🎯 对"探索-巩固"对标（**竞品/可借组件**）：
  - 定位：属于 L1 on-policy distillation 的正统做法（on-policy 版用当前策略采样向 teacher 蒸馏），与本项目 OPD 主轴**直接同族**。
  - 可借组件一："teacher = 两模型 logit 自适应外插"——用它构造比单一 teacher 更强的引导分布。
  - 可借组件二：**逐 token 自适应权重 \(\alpha_t\)（用分歧度 TVD 调引导强度）**，可迁移到 TSRD 的"在岔路口/关键步动态调脚手架强度"。
  - **缺口**：AlignDistil 做的是对齐(helpfulness)任务；teacher 由 DPO 合成，而非"会走通的 teacher 轨迹"；没有显式的探索/路径恢复语义，也没有 MTP 前瞻。
  - 一句判定：**强相关、可直接借鉴的 OPD 实例 + 自适应引导强度机制**。依据是——on-policy 蒸馏加上 token 级自适应 teacher 强度，正是 TSRD"稀疏脚手架按需调强"的现成数学载体。
- 🔭 开放问题/未来方向：
  - 【原文】Limitations——把方法扩到更大模型。
  - 【推断】把"RLHF⇔蒸馏 + 自适应外插"从对齐迁移到 RLVR/推理（teacher 换成会走通的强推理模型）。
  - 【推断】\(\alpha_t\) 的 TVD 信号可当作"关键步识别"的 proxy。
  - 【推断】外插步长的理论上界、以及自适应的 \(r\)。
  - 【推断】与显式 reward 蒸馏的混合。

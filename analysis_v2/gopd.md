gopd | Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation (G-OPD / ExOPD) | 中国人民大学高瓴 + 腾讯 LLM 部（Wenkai Yang 腾讯实习；通讯 Yankai Lin）| arXiv 2602.12125v2, 2026-02-26 · 预印本 | 主题线 L1（OPD/自蒸馏）·相关性 高

**原始论文**：https://arxiv.org/abs/2602.12125

## 一眼看懂
> 一句话导读：on-policy 蒸馏(OPD，学生边自己生成边对齐老师)本质上等价于一种特殊的 RL；作者把这层关系点破后，加一个标量旋钮 λ，就能让学生不止「追平老师」、还能「越过老师」。

- 🟦 TL;DR：本文先在理论上证明一件事——**标准 OPD 其实是一种 dense KL-约束 RL 的特例**。这里 dense 指奖励落在每个 token 上(而非只在结尾给一个总分)。它的隐藏约束是：reward 项与 KL 正则**永远 1:1 等权(β=1)**，且作为对照锚点的 reference 模型可以任选（§3.2 Remark, 式7）。
- 点破这层关系后，作者加两个旋钮：**灵活的 reference πref + reward 缩放因子 λ**。
  - λ∈(0,1) 是「插值」：学生落在 ref 与 teacher 之间。
  - **λ>1 是「外推」(ExOPD)**：让学生**越过 teacher 的能力边界**。
- 落地效果：在「把多个领域专家合并回 base」这个设定下，ExOPD 是唯一能让统一学生稳定**超过所有领域 teacher** 的方法（图1a, 表2）。
- 最巧的一步：**把 OPD 改写出一个第三方 reference πref**。这一改写暴露出两样东西——一个隐式奖励 reward=log(π*/πref)，以及一个显式的 λ 权重（式7→式11）。
  - 为什么关键：抽掉这步重写，OPD 就还是「用 reverse-KL 对齐 teacher」的黑盒，λ 和 reference 这两个旋钮根本无处插入。正是这个代数恒等变形（OPD ≡ 带特定 reward 的 dense RL）解锁了「外推超越 teacher」的可能。
  - 工程代价极小：它不改任何训练循环，只改 token 级 advantage 的计算式（式14）。

## 为什么做
> 一句话导读：OPD 实测好用但大家不清楚它为什么好用、能挖多深；标准版还把两个本可调的旋钮焊死了——既不能让学生「精确停在老师和初始之间某一点」，也不能「超过老师」。

- 研究背景：OPD 指学生采样自己的轨迹、然后在每个 token 上用 reverse KL 对齐 teacher 的 logit。它已被证明在经验上比两类对手更快更有效——一是 off-policy 蒸馏(即直接拿 teacher 输出做 SFT)，二是纯 RL。它有两大应用：
  - (a) 多任务后训练：把不同领域的 RL 变体能力(近乎)无损地合并回 base(Xiao 2026/MiMo-v2)；
  - (b) 强弱蒸馏：把大模型的能力蒸到小模型(MiniLLM/Qwen3)。【原文 §1, §2】
- 解决的具体痛点：
  - **机理不清**——OPD 经验有效，但大家对它的机制理解有限、潜力也没挖透（"a mechanistic understanding of OPD remains limited", §1）。
  - **旋钮被焊死**——标准 OPD 把 reward:KL 比例写死成 1:1、把 reference 写死为学生的初始策略。结果是**缺少一个调节「学生最终落在哪」的旋钮**：既不能精确「落在老师和初始的中间」(budget-controlled)，也无法「超越 teacher」。

- 相关工作 & 各自不足（§2 仅两段，但语境可补全到「四条平行路线」）：
  1. **Off-policy 蒸馏 / 黑盒 KD**：训练数据来自 **teacher 生成的轨迹**，而非学生自己跑出来的。具体两种做法——对 token logits 做 KL（Hinton 2015；Sanh 2019 DistilBERT；Kim & Rush 2016 SeqKD），或直接拿 teacher 的 token 做 cross-entropy SFT（Alpaca, Taori 2023；LIMA, Zhou 2023；OpenThoughts, Guha 2025）。
     - 短板：**本质是 off-policy**——学生只在模仿 teacher 的行为，并没有从「自己动作引出的奖励」里学，所以 **测试时遇到相似问题难以适配/泛化**（§3.1 原话）。这是 G-OPD 整篇的对立面。
  2. **标准 On-policy 蒸馏（最近邻、本文要推广的对象）**：代表作有 GKD（Agarwal 2024）、MiniLLM（Gu 2024，强调 reverse-KL）、Thinking Machines 的 OPD blog（Lu & Lab 2025）、Qwen3 用 OPD 做强弱蒸馏（Yang 2025a）、Xiao 2026（MiMo-v2 用 OPD 把多领域 RL 能力合并回 base）。
     - 短板：经验有效，但 **上限被 teacher 卡死**(OPD 的最优解就是 teacher 分布)、机理不清、reward:KL 写死 1:1。**G-OPD 正是把这一族放开 λ 与 ref**。
     - 另有三条 OPD 子方向被点名：跨模型家族 OPD（Patiño 2025）、黑盒 OPD（不需 teacher logits，Ye 2025a）、in-context 自蒸馏（把上下文里的信息蒸进参数：Yang 2025c, Hübotter 2026, Shenfeld 2026, Zhao 2026 SDR, Penaloza 2026）。
  3. **权重空间外推 ExPO**（Zheng 2025，直接 baseline）：先把多个对齐模型的权重平均，再沿「平均点指向 base」的反方向外推（外推因子 α∈{0.25,0.5}）。它 training-free(无需再训练)，但**不可控、不保证超过所有 teacher**（表2 多项掉分）。
     - **G-OPD 的差异**：在**输出分布/奖励空间**外推，而非权重空间；可控，且实证上是唯一全面超 teacher 的方法。
  4. **隐式奖励 RL（同形物）**：所谓隐式奖励，是指不用单独训一个奖励模型、而是用两个策略的对数概率之比当奖励。代表：DPO 的 r=β·log(πθ/πref)（Rafailov 2023）；PRIME/free process reward 用 log(πθ/πref) 当 dense reward（Cui 2025, Yuan 2024）；LASER 在最后一个 token 上自给奖励（Yang 2025d）。
     - **G-OPD 的 reward=log(π*/πref) 与它们形式相同**，但有个关键差异（§3.2(1)）：这里的 π* **不必**是「从 πref 出发跑 RL」得到的，**甚至可以和 πref 不同尺寸**——只要两者的 log-prob 之差仍是有意义的信号即可。

- 动机链（一步步推）：
  1. OPD 有效，但是个黑盒。
  2. 证明它 = dense RL 的特例(β=1, ref 任选, 式7)。
  3. 既然是 RL 特例，就能把焊死的 β(即 λ) 和 ref 放开。
  4. 放开后：λ>1 外推让学生越过 teacher(式12)；把 ref 换成 teacher 跑 RL 前的 base，可纠正奖励(式13)。
  5. 最终结果：多 teacher 合并时，它是唯一能全面超越各 teacher 的方法(表2)。
- 与最近邻工作的精确Δ：
  - vs **标准 OPD**：多了 λ 与灵活 ref。当 λ=1 且 ref=学生初态时，第二项 (λ−1)(…)=0，于是**精确退回原版 OPD**（式14）。而 λ>1 是质变——从「逼近 teacher」变成「超越 teacher」。
  - vs **ExPO（权重空间外推）**：G-OPD 在输出分布上外推，保证多 teacher 全面超越（表2 里 ExOPD 是唯一全绿的方法）；ExPO 是平均权重再外推，不可控（表2 多项打负号）。
  - vs **隐式奖励 RL(PRIME/Eurus/DPO)**：两者都用 log-ratio 当 dense reward，但 G-OPD 解绑了两条隐含假设——「π* 必须从 πref 跑 RL 得到」和「两者必须同尺寸」（§3.2(1)）。

## 怎么做 + 靠不靠谱
> 一句话导读：整套流程不改训练循环，只在算 token 级 advantage 时换一个公式(式14)；核心就是「原 OPD 项 + 一个由 λ 控制的外推位移」。下面的符号和步骤照着就能复现。

### 方法流水线（读完可复现，全部基于 §3 + 式7~14 + §4.1.1）
**符号**：D=输入分布；πθ=学生（待优化的策略）；π*=teacher；πref=参考模型（默认取学生初态 πθ 的冷启 checkpoint）；λ=reward 缩放因子（一个标量超参，默认在 {0,0.25,…,1.5} 里扫，多 teacher 设定固定为 1.25）。

1. **造 teacher**（同尺寸设定下）：拿学生 base（Qwen3-4B-Non-Thinking），分别在 math/code 数据上跑 GRPO（Shao 2024）。奖励是可验证的 0/1（math 看答案对不对、code 看是否过全部单测），得到领域 teacher `Qwen3-4B-Non-Thinking-RL-Math/Code`。强弱设定则直接用大模型 `Qwen3-30B-A3B-Instruct-2507` 当 teacher。
2. **选 πref**（三种情形）：
   - ① 多 teacher 合并→base：πref **自然就是原 base**，此时 reward=log(π*/π_base) 恰好是式10 那个「良定义的隐式奖励」。
   - ② 强弱蒸馏的默认：πref=**学生 base** πstudent_base（只需 π* 和学生 base 两个模型）。
   - ③ 强弱蒸馏 + reward correction：πref=**teacher 跑 RL 前的 base** πteacher_base（需要额外持有这个模型）。
3. **on-policy 采样**：学生自己采轨迹 y~πθ(·|x)（temperature 1.0, top-p 1.0, max_response 16384；评测时 math 每题 32 解、code 每题 4 解）。训练中对 GRPO 与 G-OPD 都做 **token-level rollout correction**（Liu 2025b "When speed kills stability"），用来缓解训练和推理引擎之间的 mismatch。
4. **逐 token 算 G-OPD advantage**（式14，承重公式）：
   \(\displaystyle A_t^{\text{G-OPD}}=\big[\log\pi_\theta(y_t\mid x,y_{<t})-\log\pi^*(y_t\mid x,y_{<t})\big]+(\lambda-1)\big[\log\pi_{\text{ref}}(y_t\mid x,y_{<t})-\log\pi^*(y_t\mid x,y_{<t})\big].\)
   直觉(两项分别干什么)：
   - 第一项 = 原版 OPD 的 token 级 advantage（学生 log-prob 减 teacher log-prob，衡量学生偏离 teacher 多少）。
   - 第二项 = 外推位移，把目标从「对齐 teacher」继续往前推到「teacher 之外、沿 (λ−1)·(teacher−ref) 方向」。
   - λ=1 时第二项消失，退回原 OPD。
5. **policy gradient 更新**（式13/14 的梯度形式）：
   \(\displaystyle \nabla_\theta J_{\text{G-OPD}}(\theta)=\mathbb{E}_{x\sim D,\,y\sim\pi_\theta(\cdot\mid x)}\Big[\textstyle\sum_{t=1}^{T}A_t^{\text{G-OPD}}\,\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})\Big].\)
   在 veRL 实现里：开关 `only_reverse_kl_advantages=True`、传 `lambda_vals=λ`。**注**：这里 discount factor=0（即只看 next-token 的优化，沿用 Lu & Lab 2025 / Xiao 2026）；也就是把式5 的双重求和近似成式6 的单 token 形式。

### 关键算法/损失的真实形式 + 直觉（全部从 §3.2 抄准）
> 一句话导读：这一串公式其实就一条主线——把 OPD 的 reverse-KL 目标(式4)恒等改写成「隐式奖励 − KL 约束」的 RL 形式(式7)，再塞进一个 λ(式11)，就得到「插值/外推/纠偏」三合一。读时盯住 λ 取值在哪、ref 选谁即可。

- **OPD 目标（出发点，式4）**：\(J_{\text{OPD}}(\theta)=\min_\theta \mathbb{E}_{x\sim D,\,y\sim\pi_\theta(\cdot\mid x)}\big[D_{\mathrm{KL}}(\pi_\theta(y\mid x)\,\|\,\pi^*(y\mid x))\big]\)。注意 y 由学生自己采 ⇒ 这是 on-policy；用的是 **reverse KL**（学生在前、teacher 在后）。
- **核心恒等变形（式7，全篇支点）**：引入第三方 πref，把 reverse KL 拆成「隐式奖励 − KL-to-ref」两块：
  \(\displaystyle J_{\text{OPD}}(\theta)=\max_\theta \mathbb{E}\Big[\underbrace{\log\tfrac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)}}_{r(x,y)}-D_{\mathrm{KL}}\big(\pi_\theta(y\mid x)\,\|\,\pi_{\text{ref}}(y\mid x)\big)\Big].\)
  **Remark（这步在说什么）**：上式正是式2 那个 KL-约束 RL，对应关系是——reward \(r=\log(\pi^*/\pi_{\text{ref}})\)、KL 加在 πθ 与 πref 之间、且 reward:KL 的比例**永远 1:1（β=1）**。πref 怎么选都不影响化简回式4，因为 log πref 项在两处会抵消。
- **token 级隐式奖励（式9）**：\(r_t^{\text{OPD}}=\log\dfrac{\pi^*(y_t\mid x,y_{<t})}{\pi_{\text{ref}}(y_t\mid x,y_{<t})}\)。它与 DPO 的隐式奖励（式10：\(r=\beta\log\tfrac{\pi_\theta}{\pi_{\text{ref}}}+\beta\log Z(x)\)）形式相同——因为 log Z(x) 只依赖 x，所以 log-ratio 是「真奖励」的一个良定义代理。
  - 一个对比：RL 的奖励是稀疏的（式8：只有末 token 有 outcome reward，其余为 0），而 OPD 是 **dense 的——每个 token 都有奖励**。
- **G-OPD 目标（式11，加上 λ）**：\(J_{\text{G-OPD}}(\theta)=\max_\theta \mathbb{E}\big[\lambda\log\tfrac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)}-D_{\mathrm{KL}}(\pi_\theta\,\|\,\pi_{\text{ref}})\big]\)，其中 \(\lambda=1/\beta\)（即把上面焊死的 β 松开成可调）。
- **最优解（式12，看懂插值/外推就靠它）**：
  \(\displaystyle \log\pi_\theta(y\mid x)=\lambda\log\pi^*(y\mid x)+(1-\lambda)\log\pi_{\text{ref}}(y\mid x)=\log\pi^*(y\mid x)+(\lambda-1)\big(\log\pi^*(y\mid x)-\log\pi_{\text{ref}}(y\mid x)\big).\)
  - \(0<\lambda<1\)（**reward interpolation，插值**）：学生 log-prob = teacher 与 ref 的**线性插值**（等价于把 reward 换成 λ·r+(1−λ)·0）。学生的行为（精度、长度）落在 base 与 teacher 之间，且随 λ 增大单调逼近 teacher——于是可以做 **budget-controlled reasoning**（按预算控制推理深浅，图2/3/4）。
  - \(\lambda>1\)（**reward extrapolation = ExOPD，外推**）：在匹配 teacher 之外**再加一段位移** (λ−1)(log π*−log πref)，把学生推到 teacher 分布之外。
- **reward correction 的等价改写（式13，强弱蒸馏的关键）**：
  \(\displaystyle J_{\text{G-OPD}}(\theta)=\max_\theta \mathbb{E}\Big[(\lambda-1)\log\tfrac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)}-D_{\mathrm{KL}}\big(\pi_\theta\,\|\,\pi^*\big)\Big].\)
  在同等 KL 强度下，选 \(\pi_{\text{ref}}=\pi^{\text{teacher}}_{\text{base}}\)（teacher 跑 RL 前的 base）更合理：
  - 此时 reward \(\log(\pi^*/\pi^{\text{teacher}}_{\text{base}})\) 恰好是 teacher 自身 RL 后训练所诱导的**良定义隐式奖励**（按式10）；
  - 而原来的 \(\log(\pi^*/\pi^{\text{student}}_{\text{base}})\)，因为师生两个 base 之间存在**内在的知识/容量鸿沟**，信号更 noisy。
  - 做法：给默认 reward 加上修正项 \(\log(\pi^{\text{student}}_{\text{base}}/\pi^{\text{teacher}}_{\text{base}})\)，就得到校正后的 reward。
  - 代价：需要额外持有 πteacher_base，并多算一次这个更大的 ref 的 logprob。

### 逐组件必要性（消融与失效边界）
> 一句话导读：λ 是唯一的核心旋钮，证据也最足；但它有个明确的失效点——λ 调到 1.5 学生就会去「钻奖励的空子」而崩。

- **λ（reward 缩放因子）**：核心旋钮。消融方式=扫 λ∈{0,0.25,…,1.5}（图2/3/4）。λ=0 退回学生初态、λ=1 退回 OPD。证据充分：
  - λ<1：性能/长度单调地介于 base 与 teacher 之间；
  - λ=1.25：一致优于 OPD 且超过 teacher；
  - **λ=1.5：反而不稳、退化**。作者的解释是学生在「hack 隐式奖励」——即激进地去拟合 log-ratio 的峰值，哪怕某些 token 因为 bias 有过大的 log-ratio（§4.1.2）。这是关键的失效边界证据。
- **灵活 reference / reward correction**：图6 消融——强弱蒸馏里把 ref 从学生 base 换成 teacher-pre-RL base，math/code 各 +0.6/+1.0%（1.7B 学生：math 28.1（无修正）vs 28.7（有修正）；code 51.3 vs 52.3）。**作者诚实标注了代价**：需额外持有 teacher-pre-RL 模型，并多算更大 ref 的 logprob。
- **多 teacher 合并 vs 单 teacher**：表2 证明，把 λ=1.25 固定不调，多 teacher 设定下 ExOPD 是唯一能全面超越两个 teacher 的方法（SFT/ExPO/OPD 都有项掉分）。
- 关键机制直觉补充（训练动态，图5）：ExOPD 比 OPD **训练奖励更高、响应更长、熵更高**；作者把熵升归因于「响应变长带来的多样性增加」。

- 实验与证据：
  - 配置：base/student=Qwen3-1.7B/4B-Non-Thinking；teacher=对 4B 做领域 GRPO 得到的 RL-Math/RL-Code，以及强弱设定下的 Qwen3-30B-A3B-Instruct-2507。
  - 数据：DeepMath(He 2025)过滤后 57K(难度≥6) + Eurus-RL-Code(Cui 2025) 25K；蒸馏数据与 RL 数据同源。
  - 评测：AIME24/25、HMMT25(Feb/Nov)、HumanEval+/MBPP+/LiveCodeBench(v6 only, 2025-02~05)；math 用 Math-Verify 验证。
  - 关键数字：
    - 单 teacher math：**ExOPD 48.0 vs teacher 46.0 vs OPD 46.5 vs ExPO 45.8**(表2)；
    - 强弱蒸馏 4B-student：**ExOPD 45.3 vs OPD 42.6 vs SFT 35.1**(表3)；
    - 1.7B-student：**25.4 vs OPD 23.1 vs SFT 13.5**(表3)；
    - 关键对照：ExOPD 只跑 50 步就拿 47.5 avg(表1 math 列实为 48.0)，已超过「teacher 再继续 RL 100 步」的 46.9(表1)——这就排除了「teacher 只是训得不够」的解释。
- baseline 公平吗：较公平。
  - OPD/SFT/ExPO 在同数据、同轨迹数、同步数下对比(§4.1.3、附录B)；SFT 的轨迹数与 OPD/ExOPD 对齐；多 teacher 设定下还把 math 数据下采样到与 code 等量。
  - 还专门做了两个对照：「teacher 继续 RL」(表1) 和「teacher 充分训练 1200 步」(附录C 表8，此时 ExOPD 增益缩到 +0.3~0.6%)。
  - **这点很关键且诚实**：附录C 说明，当 teacher 已经被 RL 充分榨干，ExOPD 能再超越它的空间就很小了。
- 假设与失效边界：
  - 【原文】(1) λ 过大(1.5)会去 hack 隐式奖励、导致不稳退化；(2) reward correction 需要拿得到 teacher-pre-RL + 额外算力；(3) 附录C 显示 teacher 充分训练后 ExOPD 优势大幅缩水。
  - 【推断】(4) 「超越 teacher」的本质是**沿 teacher 已学方向的外推**——只在 teacher 还没收敛/欠训、且外推方向恰好仍指向正确分布时才成立。它并不凭空创造新能力，而是「把 teacher 没走完的那段路替它走完」（依据：式12 的外推位移沿 (π*−πref) 方向 + 附录C 证据）。
  - 【推断】(5) 多 teacher 当前实现**只支持两个 teacher**(README)，更多/更异质的 teacher 未验证(作者列为 future work)。
  - 【推断】(6) ExOPD 有个副作用：响应变长(图4/5)。作者归因于隐式奖励的长度偏置(引 Yang 2025d LASER)——也就是说，长度膨胀可能换来的是虚高的分数。
- 祛魅总结：
  - 真贡献=**(a)** OPD ≡ dense-RL(β=1) 这座干净的理论桥；**(b)** 用 λ/ref 两个旋钮把「插值/外推/纠偏」统一进一个框架；**(c)** 多 teacher 合并能全面超 teacher 的实证。理论与实验都较扎实、对照也诚实。
  - 需打折的：「learning beyond teacher」这个标题略大——附录C 自己就证明了 teacher 充分训练后超越幅度趋近于零。所以「超越」其实是**欠训 teacher 上的外推红利**，不是凭空涨能力；而且增益普遍在 +1~3% 量级、还伴随响应变长。
  - 【推断】对「探索-巩固」这是**最贴近的 L1 工作之一**：多 teacher→统一学生，正是「把多路能力巩固进一套参数」；λ 旋钮则相当于「巩固强度」的连续控制器。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**=teacher 在学生轨迹上的 token 级 logit（即 reverse-KL/隐式奖励 log(π*/πref)，dense、每个 token 都有）；
  - **改什么**=学生策略参数 θ；
  - **何时改**=on-policy 蒸馏训练的每一步（学生采自己的轨迹）；
  - **免梯度?**=否，是 policy gradient；
  - **记忆-技能生命周期**=把多领域 teacher 的技能**合并固化进单套参数**（multi-teacher→unified student），没有外部记忆/技能库；
  - **防遗忘机制**=间接——OPD 本身被定位为「(近乎)无损地把各 RL 变体能力合回 base」的多任务范式(引 Xiao 2026)，KL-to-ref 约束抑制漂移；但 λ>1 外推反而**增加**偏离 ref 的风险(过大即 hack)。
- ⑦ 开源代码+框架/harness：https://github.com/RUCBM/G-OPD （已 clone ~33MB，Tier A，README 已核）。
  - 框架=**veRL v0.6.1**(volcengine/verl，HybridFlow，引 Sheng 2024) + vLLM/SGLang rollout；含 verl/、math_eval/、code_eval/(基于 Absolute-Zero-Reasoner/EvalPlus/LiveCodeBench)。训练数据已开源(HF: Keven16/G-OPD-Training-Data)。
  - **额外亮点**：README 2026-03 更新了一份**自研的 on-policy self/context-distillation (OPSD)** 代码(`run_qwen3-4b-context-distill.sh`、`ref_input_utils.py`)。它的 teacher 分布 = 学生自己在「额外上下文(GT 解/反馈/高层经验)」下的 in-context 分布；teacher 权重固定为学生初态、不做 EMA(避免崩溃)。
- 💰 资源/成本与可扩展性：8×A800/H 级 8 卡(n_gpus_per_node=8)；batch 1024、rollout n=1、max_response 16384、lr 1e-5。G-OPD 只需 50 步(同尺寸)/100 步(强弱)，**步数极少**(作者说再多步反而过拟合)。额外成本：λ≠1 要多算一次 πref 的 logprob；reward correction 因 ref 更大、成本更高。
- 🎯 对"探索-巩固"对标：**强对标(支撑+可借组件)**。判定依据：
  - ① **巩固侧直接命中**——multi-teacher→unified student，就是「把多条已走通的路径(领域专家)固化进一套参数、且尽量不互相遗忘」，λ 就是巩固强度旋钮。
  - ② **可借组件**：
    - (a) λ>1 外推 =「让学生沿 teacher 指引的方向再多走一步、越过脚手架」(式12 外推位移)，与 TSRD「teacher 当稀疏脚手架、最终要学生自己走得更远」的精神一致；
    - (b) README 里的 **OPSD(on-policy 自/上下文蒸馏)** 几乎是 TSRD 的现成骨架——teacher = 学生加了「GT 解/反馈/经验」后的自身分布，正好对应「探索(给上下文提示找到路)→ 巩固(蒸回无上下文的参数)」。
  - 竞品风险：G-OPD 是 teacher-driven 的 dense 全程对齐，而 TSRD 主张**稀疏脚手架 + 学生自选恢复分支**。G-OPD 不区分「关键步/非关键步」、不做单点接管，这正是差异化空间。
  - 缺口：无 MTP/前瞻、无「走偏后自选恢复」的显式机制、无路径选择信号。
- 🔭 开放问题/未来方向：
  - 【原文】(1) 在更大模型上验证 ExOPD 泛化性；(2) 更广、更异质的多 teacher 集合下的鲁棒性；(3) 跨模型家族(不同 family)的 OPD。
  - 【推断】(4) λ 当前是全局常数——可否做 **token/step 级自适应 λ**（关键推理步外推强、常规步保守）？这天然连接 hapo/holderpo 的「token 异质」与本课题的「关键步接管」。
  - 【推断】(5) 把 OPSD 的「上下文=高层经验」推向 agent 自进化（探索得到的成功轨迹经验 → 上下文 → 蒸回参数）。
  - 【推断】(6) 缓解 λ>1 的 reward-hacking 与长度膨胀(需要一个长度无关的隐式奖励)。

key|读到PDF?|L线|对标结论|残留待核数
gopd | 是(全文17页) | L1 | 强对标:multi-teacher→统一学生=巩固多路径进参数;λ外推=越过脚手架;README的OPSD自蒸馏近似TSRD骨架。差异:全程dense无稀疏脚手架/无单点回轨 | 0

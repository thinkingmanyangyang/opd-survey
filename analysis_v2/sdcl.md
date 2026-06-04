sdcl | Self-Distillation Enables Continual Learning (SDFT) | MIT + Improbable AI Lab + ETH Zurich(Idan Shenfeld, Mehul Damani, Jonas Hübotter, Pulkit Agrawal) | 2026-01·arXiv 预印本·v1(2026-01-27, cs.LG) | 主题线 L1(在线策略/自蒸馏)+ L5(持续学习·防遗忘)·相关性 高

**原始论文**:https://arxiv.org/abs/2601.19897 （arXiv:2601.19897v1;项目页 http://idanshenfeld.com/SDFT）

## 一眼看懂

> 一句话导读:想让模型学新东西又不忘旧的(持续学习),关键不是"用什么 teacher",而是"让模型在自己采样出来的轨迹上学"——这一步才是真正承重的。

- 🟦 TL;DR:目标是让模型**学新技能/新知识又不忘旧的**(持续学习)。两条已有路都有坑:
  - 在线策略 RL(让模型在自己生成的数据上学)遗忘少,但要 reward(奖励信号),现实里常没有;
  - 从示范学只能 SFT,而 SFT 是 off-policy(在别人给的固定数据上学),会灾难性遗忘。
- SDFT 的招:用**同一个模型**当自己的 teacher。具体分两种模式——teacher 模式 = 模型同时看到(query + 一条专家示范 c);student 模式 = 模型只看 query。然后在 student **自己采样**出来的 on-policy 轨迹上,做 token 级 KL 蒸馏(逐 token 让 student 的分布向 teacher 靠拢),等于把 off-policy 的示范"软化"成 on-policy 的学习信号。teacher 的权重默认取 student 参数的 **EMA**(指数滑动平均,即对历史参数做平滑、得到一个跟得上但更稳的副本)。
- 最巧的一步:**把"on-policy 采样"抽掉(改成在 teacher 生成的文本上离线蒸馏/SFT),增益就大半消失**。
  - §4.6 的关键消融(Fig.6)正是证这点:同一个 teacher 下,offline distillation < on-policy SDFT。
  - 为什么是它承重:论文反复强调"光有好 teacher 不够,必须在学生自己的轨迹分布上学,才能即时纠错 + 贴近预训练分布 → 少遗忘"(§3.2 第二条件 + §4.6)。所以真正承重的柱子是"on-policy",而不是"用 ICL(in-context learning,即靠上下文示例临时学会任务)当 teacher"这个 trick——后者 GKD/context-distillation 早就有了。

## 为什么做

> 一句话导读:大家都知道 on-policy 遗忘少,但这个好处一直锁在"有 reward 的 RL"里;现实常常只有示范、没 reward,本文就是把这个好处搬过来。

- 研究背景:基础模型部署后通常是**静态**的,不再更新参数去学新技能或内化新知识(§1)。已有共识是 on-policy 学习比 off-policy 少遗忘(Shenfeld 2025 "RL's Razor"、Chen 2025),但成熟的 on-policy 方案几乎都在 RL 里、都需要显式 reward。
- 解决的具体痛点:现实里多数场景只有**专家示范**、没有 reward。于是主流只能做 SFT,而 SFT 本质 off-policy,顺序学新任务时会严重遗忘旧能力(§1)。核心矛盾就是:"只有示范时,怎么拿到 on-policy 学习的好处?"
- 相关工作 & 各自不足(来龙去脉 + 并行路线 + 各自短板 + 站谁肩上):
  - **(路线 A)持续学习的经典派系**——三派:
    - 正则化(EWC/SI):惩罚重要权重的漂移;
    - 回放(rehearsal/replay):存一些旧样本混进来一起训;
    - 参数隔离(adapter/LoRA):给新任务单开一组新参数。
    这三派要么需要估"权重重要性"或存旧数据,要么靠扩参数,都没回答"只有新任务示范、且要 full-FT(全参微调)单一模型时怎么不忘"。SDFT 不属于这三派——它走的是"**让更新本身贴近预训练分布**"的隐式正则路径(§3.2 的 trust-region 视角),既不存旧数据、也不扩参数。
  - **(路线 B,本文直接对手)on-policy vs off-policy 的遗忘机理**:Shenfeld 2025("RL's Razor")与 Chen 2025 给出关键经验/理论——on-policy 更新比 off-policy(SFT)遗忘更少,因为它把更新约束在当前策略附近(从 KL 角度看)。**短板**:这些工作几乎都在**有 reward 的 RL 设定**里成立,没解决"只有示范、没 reward"这个常见场景。SDFT 正踩在这块肩上,把"on-policy 少遗忘"搬到无 reward 的示范学习。
  - **(路线 C)逆强化学习 IRL**:从 Ng & Russell 2000 起,思路是先从专家示范反推出 reward,再 on-policy RL 优化。理论上正好填"只有示范"的坑。**短板**:不 scale——max-entropy IRL、对抗式(GAIL 系)、偏好式各需不同的强结构先验,而且内层 RL + 外层 reward 学习是双循环,既贵又不稳(§2)。SDFT 的巧思:**不真去学 reward**,而是用模型自身的 ICL 能力,把"示范条件化分布 \(\pi(\cdot|x,c)\)"当作那条隐式最优策略的免费近似(§3.1),从而绕开 IRL 的双循环。
  - **(路线 D,最近邻)context distillation / GKD**:
    - context distillation(把"condition 在某 context 上的模型"当 teacher、蒸给"没有 context 的自己")早就有;
    - GKD(Agarwal 2024)给出 on-policy 蒸馏的标准框架(学生采样、teacher 在学生轨迹上给出分布)。
    **短板/差异**:① 经典 context distillation **通常是离线的**(在 teacher 自己生成的文本上蒸),而且 context 是**固定的全局 prompt**(比如 few-shot 示例池或一段行为准则);② GKD 是通用蒸馏框架,没有定位到持续学习。SDFT 与二者的精确差异有三点:
    - on-policy:在学生自己的轨迹上学,teacher 可以即时纠错;
    - context 是逐 query 选的具体示范 c:属于实例级条件(表达细粒度的任务意图,而非单一全局先验);
    - 定位到防遗忘的持续学习,并补了一条 IRL 等价解释(§2 Context Distillation 段、§3.1)。
- 动机链(一步步推下来):
  1. 要少遗忘 → 要 on-policy;
  2. 但只有示范、没 reward;
  3. IRL 又不 scale;
  4. 所以改用模型的 **in-context learning(ICL)** 能力——"condition 在示范上的模型"可看作"对该任务近似最优、且仍贴近预训练分布"的策略,直接拿它当 teacher,在学生 on-policy 轨迹上蒸馏。
  - 为什么不用更简单的 SFT/离线蒸馏:SFT 会把模型推离预训练分布(§3.2 实测:SFT 偏离 base \(1.26\) nats,而示范条件化 teacher 仅 \(0.68\) nats),而离预训练分布越远,遗忘越多。
- 与最近邻工作的Δ:最像 GKD(on-policy 蒸馏)和 context-distillation。**关键就差一点**:把"示范条件化的同一模型(EMA 副本)"当 teacher 来做 **on-policy** 蒸馏,并定位到**持续学习防遗忘**,还给了一条 IRL 等价解释(§3.1)。为什么有用:这个 teacher 既懂新任务(ICL 把示范软化进了分布),又仍锚在预训练分布附近——于是既学到新技能又不偏离原能力(§3.2 Fig.2 右:示范条件化 teacher 的输出分布比 SFT 目标更接近 base)。

## 怎么做(到可复现粒度)

- **角色与符号约定**【原文 §3,Fig.1】:同一个模型 \(\pi\) 扮两个角色。
  - **学生** \(P=\pi_\theta(\cdot|x)\):只 condition 在 query \(x\) 上,负责采样 on-policy 轨迹、被更新。
  - **教师** \(Q=\pi(\cdot|x,c)\):同一模型 condition 在 (query \(x\) + 一条专家示范 \(c\)) 上,权重用学生参数的 EMA(下记 \(\phi\));teacher 的分布在每一步都被当作**常数**(stop-gradient,即不让梯度流过它)。
  - 输入是示范数据集 \(\{(x_i,c_i)\}\),输出是持续更新的学生。
- **数据流(单 query 一轮:输入→输出)**:
  1. 取一条 \((x,c)\)。
  2. **学生采样**:\(y\sim\pi_\theta(\cdot|x)\),**只采单条** on-policy rollout(附录 A.1 实测多采样无益,见下)。
  3. **构造 teacher context**:用固定模板把示范注入——【原文 §3 原样模板】
     > `<Question>`
     > `This is an example for a response to the question:`
     > `<Demonstration>`
     > `Now answer with a response of your own, including the thinking process:`
     论文称此模板**足以避免逐字复制 \(c\)**,而是激发模型"理解示范意图后用自己的话(含思考过程)作答"(§3 + §3.2)。teacher 用 EMA 权重 \(\phi\),在该 context 上对采样到的 \(y\) 算 token 分布 \(\pi(\cdot|y_{<t},x,c)\)。
  4. **算 KL 损失 + 反传**(只对学生参数 \(\theta\)):见下"核心损失"。
  5. **EMA 同步**:\(\phi \leftarrow \alpha\theta + (1-\alpha)\phi\)。
  6. 回到第 1 步。(完整伪码 = 附录 Algorithm 1。)
- **核心目标(原文写的是 reverse-KL)**【原文 Eq.(1)】:
  \(\displaystyle \mathcal{L}(\theta)=D_{\mathrm{KL}}\!\big(\pi_\theta(\cdot|x)\,\|\,\pi(\cdot|x,c)\big)=\mathbb{E}_{y\sim\pi_\theta(y|x)}\!\left[\log\frac{\pi_\theta(y|x)}{\pi(y|x,c)}\right]\)
  直觉:让"只看 query 的学生"在它**自己**的轨迹分布上,逐 token 向"看了示范的教师"靠拢。reverse-KL(student 在外)是 mode-seeking(抓主峰),会鼓励学生抓住 teacher 的主模式。
- **token 级梯度估计器**【原文 Eq.(2)】(对自回归结构展开,teacher 视为固定;推导见 Tang & Munos 2025):
  \(\displaystyle \nabla_\theta\mathcal{L}(\theta)=\mathbb{E}_{y\sim\pi_\theta}\!\left[\sum_t\sum_{y_t\in V}\log\frac{\pi_\theta(y_t|y_{<t},x)}{\pi(y_t|y_{<t},x,c)}\,\nabla_\theta\log\pi_\theta(y_t|y_{<t},x)\right]\)
  其中 \(V\) 是词表。**注意**:这是把 KL 解析地在整个词表上展开的形式(下文叫"full analytic per-token"估计器)。
- **三种 KL 梯度估计器(附录 A.1,做了消融,实测选第二种)**:
  - **(i) Token-level(partial)估计器**:把序列 KL 拆成各 token 项独立微分:
    \(\displaystyle \sum_t \log\frac{\pi_\theta(y_t|y_{<t},x)}{\pi(y_t|y_{<t},x,c)}\,\nabla_\theta\log\pi_\theta(y_t|y_{<t},x)\)
    【原文】它"忽略早期 token 对后续 token 分布的影响",对真梯度**有偏**、方差高、KL 控制弱。
  - **(ii) Full analytic per-token 估计器(实测默认)**:每一步都在整个词表上解析求和(marginalize over \(V\)):
    \(\displaystyle \sum_t\sum_{v\in V}\pi_\theta(v|y_{<t},x)\,\log\frac{\pi_\theta(v|y_{<t},x)}{\pi(v|y_{<t},x,c)}\,\nabla_\theta\log\pi_\theta(v|y_{<t},x)\)
    【原文】严格低于 (i) 的方差,但**在序列层仍有偏**(没考虑 \(y_t\) 对未来 \(y_{>t}\) 的影响);因为复用了 forward pass 已产出的量,计算上很划算。**实测它最稳、下游最好**(§A.1)。
  - **(iii) Rao-Blackwellized 估计器**(Amini et al. 2025):对 next-token 分布解析积分、对前缀保留 MC 采样,得到 KL 及其梯度的**无偏**、且方差可证更低的估计:
    \(\displaystyle b_{\mathrm{grb}}=\sum_t k_\theta(y_{<t})\;,\quad k_\theta(y_{<t})=\mathrm{KL}\!\big(\pi_\theta(\cdot|y_{<t},x)\,\|\,\pi(\cdot|y_{<t},x,c)\big)\)
    【原文】更贵;实测相对它额外的复杂度**没有可测增益**,故弃用。
  - **采样数**:理论上每个 prompt 多采几条能降方差,但**实测多采样收益微乎其微、却显著加 compute**,所以全部主实验用单轨迹/prompt + (ii)(§A.1)。
- **⚠ KL 方向:论文正文 vs 实际实现的关键脱节**(本轮基于 PDF + 仓库 README 双重核实):
  - 【原文】§3 与 Eq.(1) 明写最小化 **reverse-KL** \(D_{\mathrm{KL}}(\pi_\theta(\cdot|x)\,\|\,\pi(\cdot|x,c))\);§3.1 整条"Self-Distillation as Inverse RL"等价推导(隐式 reward \(r=\log\pi(y|x,c)-\log\pi_k(y|x)\),In-Context Assumption \(\pi^*_{k+1}\approx\pi(\cdot|x,c)\))**只在 reverse-KL 下成立**。
  - 【原文·仓库 README errata 04/07/26】仓库明确声明:"**所有论文结果实际是用 on-policy 采样 + per-token forward KL loss(类似 GKD)产生的**,故这是仓库默认参数,我们将尽快更新 arXiv 澄清。" 也就是说:论文正文的 reverse-KL 叙事与 §3.1 的 inverse-RL 推导,与**真正跑出全部结果**所用的 forward-KL(mode-covering,即覆盖整个分布;且不再对应那条 inverse-RL 等价链)**方向相反**。
  - 旁证(同在论文内,但容易被忽略):附录 A.1 实际选用的估计器是 "**full analytic per-token estimator**"(逐 timestep 在词表上解析求和,即上式 (ii)),论文自承它"在 sequence 层有偏";而正是这种 per-token 解析形式,与 GKD 式 forward-KL 的实现一致——这与 README errata 自洽。
  - 〔推断·影响〕这是本方法当前**最大的自洽性瑕疵**:理论框架(inverse-RL)挂在 reverse-KL 上,实证结果却来自 forward-KL。复现/引用时应以"on-policy + per-token forward KL(GKD 式)"为准,把 §3.1 的 inverse-RL 解释视为**事后理论叙事而非实测对应**。
- 逐组件必要性:
  - **on-policy 采样**:核心承重柱。消融充分——§4.6 Fig.6 里,同 teacher 下 offline distillation 与 SFT-from-teacher 都 < on-policy SDFT;§4.2 Fig.5 右的 pass@k(k 到 128)全程领先,说明增益是真技能习得而非熵坍缩(即不是靠收窄分布刷分)。没它 → 退化成普通离线蒸馏,遗忘加重。
  - **示范条件化 teacher(text + answer)**:负责"既懂新任务又贴近 base"。消融充分——§3.2 实测在 ToolAlpaca 上,无示范的 base 仅 42%、给示范的 teacher 达 **100%**;附录 A.2 Fig.7 进一步拆开:teacher 同时 condition 文本+答案(89% strict)> 仅文本(75%)> 仅答案(37%)。
  - **EMA teacher**:稳定性关键。消融充分——附录 A.3:
    - 用 frozen base 当 teacher → 稳但偏弱(跟不上学生进步);
    - 用**当前 student 自己**当 teacher → 严重不稳("token 概率的小随机波动经 on-policy 反馈环被快速放大致训练发散",§A.3 原话);
    - EMA 折中(Fig.8:EMA teacher 既跟得上学生进步、又能平滑高方差更新)。
    **注意:这意味着方法不是字面上的"同一当前模型自蒸馏",而是用一个 EMA 副本当 teacher**。
  - **KL 梯度估计器选择**:有消融(见上 (i)/(ii)/(iii)),**实测选 full analytic per-token**;并发现"每 prompt 多采样"几乎无收益却显著加 compute,故定为**单轨迹/prompt**。
  - **mask 前几个 token 的损失**(§5 Learned Artifacts):防止学生继承 teacher 的 "Based on the text..." 这类口头禅(由模板诱发)。论文自承这是**启发式补丁**,不是原理性的解。
- 关键机制/IRL 等价推导(直觉 + 真实公式):
  - **起点:trust-region RL**(Schulman 2015)——第 \(k\!+\!1\) 步的策略更新约束在当前策略 \(\pi_k\) 附近:
    \(\displaystyle \pi_{k+1}=\arg\max_{\pi}\ \mathbb{E}_{y\sim\pi}[r(y,x)]-\beta\,D_{\mathrm{KL}}\!\big(\pi(\cdot|x)\,\|\,\pi_k(\cdot|x)\big)\quad(\text{Eq.3})\)
    它的最优解是一个 tilted 分布 \(\pi^*_{k+1}(y|x)\propto\pi_k(y|x)\exp\!\big(\tfrac{1}{\beta}r(y,x)\big)\)(Korbak 2022 / Rafailov 2023)。反解出 reward:
    \(\displaystyle r(y,x)=\beta\big(\log\pi^*_{k+1}(y|x)-\log\pi_k(y|x)\big)+C\)
  - **In-Context Assumption(Eq.4,核心假设)**:\(\pi^*_{k+1}(y|x)\approx\pi(y|x,c)\)——直觉是"观察一条示范引发的行为偏移,反映了专家的真实意图",所以 condition 在 \(c\) 上的模型 ≈ 该任务的最优下一步策略。代入得到**内蕴 reward(Eq.5,丢掉不影响最优策略的线性项 \(\beta,C\))**:
    \(\displaystyle r(y,x,c)=\log\pi(y|x,c)-\log\pi_k(y|x)\)
  - **分解到 token 级**:\(r_t(y_t|y_{<t},x,c)=\log\dfrac{\pi(y_t|y_{<t},x,c)}{\pi_k(y_t|y_{<t},x)}\),且 \(\sum_t r_t=r(y,x,c)\)。
  - **等价性(Eq.6 邻近)**:在当前策略 \(\pi_k\) 下,该 reward 的策略梯度 \(\nabla_\theta J(\pi_k)=\mathbb{E}_{y\sim\pi_k}[r(y,x,c)\nabla_\theta\log\pi_k(y|x)]\) "在期望意义上等于 reverse-KL \(D_{\mathrm{KL}}(\pi_k(\cdot|x)\,\|\,\pi(\cdot|x,c))\) 的梯度"(§3.1 原话)。**直觉**:把学生当前行为,和它"更聪明的、看了示范的自己"作比较,二者差异(log-prob 变化)就是免费的、贴近 base 的隐式奖励信号。【再次提醒:这条等价链只对 reverse-KL 成立,与实测的 forward-KL 脱节。】
  - **为何少遗忘**(§3.2 两条件):
    - ① **Optimality**(够好)——示范条件化策略的期望 reward ≈ 未知最优策略:\(\mathbb{E}_{y\sim\pi(\cdot|x,c)}[r]\approx\mathbb{E}_{y\sim\pi^*_{k+1}}[r]\);
    - ② **Minimal Deviation**(走得近)——trust-region 下的最优策略是"在达最优 reward 的策略里离 \(\pi_k\) 最近的那个",而示范条件化 teacher 恰好满足"高质量输出 + 贴近 base"(实测 \(0.68\) nats vs SFT \(1.26\) nats),朝它更新走得"近",所以少遗忘。这就是 anchoring(锚定)的直觉。
- 关键超参默认值(汇总,附录 B / README):base = Qwen2.5-7B-Instruct;full fine-tuning;EMA teacher;单 on-policy rollout/prompt;full analytic per-token KL;greedy 评测、3 seeds 报均值+95%CI;单卡 NVIDIA H200。
  - lr:README 训练示例用 **5e-5**(`--num_train_epochs 2`)〔待核:v1 分析曾记 main.py 默认 2e-5,本轮未重开源码核对,以 README 示例记录〕。
  - EMA 率 \(\alpha\):论文只给"base/self/EMA"三档定性对比,未给具体 \(\alpha\) 数值〔待核〕。

## 靠不靠谱

> 一句话导读:核心卖点(on-policy 少遗忘)有一串干净消融坐实,尤其 OOD 间接问题和"保住长 CoT"两个结果很硬;但理论框架因 KL 方向 errata 与实测脱节,且只在 200K-token 窄域 + 7B 上验证。

- 实验与证据:
  - 数据集/设置:
    - **Skill Learning** 三域——Science Q&A(SciKnowEval Chemistry L-3)、Tool Use(ToolAlpaca)、Medical(HuatuoGPT-o1,stage-1 训/stage-2 评);
    - **Knowledge Acquisition**——2025 自然灾害的 Wikipedia 语料(超出模型 cutoff,约 200K tokens)生成 QA + OOD"间接问题",当全文超 context 时,与 oracle-retriever RAG 对照;
    - **遗忘评测**——HellaSwag/TruthfulQA/MMLU/IFEval/Winogrande/HumanEval 的均值(用 EleutherAI lm-eval-harness 指定 commit);
    - **顺序学三技能**——Tooluse→Science→Medical。
    - 主 base = Qwen2.5-7B-Instruct;scaling 用 Qwen2.5 家族 3B/7B/14B;§4.5 reasoning 用 Olmo-3-7B-Think。
  - 支撑核心主张的关键数字:
    - Knowledge Acquisition(Table 1):SDFT strict **89**/lenient **100**/OOD **98**,远超 SFT(80/95/80)、逼近 Oracle RAG(91/100/100),而 CPT 仅 9/37/7 —— 其中 OOD 间接问题 98 vs 80,是"真内化 vs 死记"最有力的证据;
    - Skill Learning(Fig.4):三域 Pareto 全优(新任务 acc 与旧能力保持同时优于 SFT/DFT/Re-invoke);
    - 顺序学(Fig.3):SDFT 累积不退化,SFT 则学新忘旧、来回震荡;
    - 规模效应(Fig.5 左):**3B=−3.3 / 7B=+4.0 / 14B=+6.9**(增益随 ICL 能力单调增);
    - §4.5(Table 2):在 answer-only 的医疗数据上,SFT 掉点 31.2→23.5 且输出变短(4612→3273 tok),SDFT 反升到 **43.7**(4180 tok,保住了长 CoT)。
  - baseline 公平吗:
    - Skill Learning 与 SFT、DFT(用重要性采样把离线当 on-policy)、Re-invoke(SFT 后再用 base 在通用 prompt 上 on-policy 蒸馏以恢复能力)比;
    - Knowledge 与 CPT/SFT/oracle-RAG 比;
    - 每个 baseline 都做了超参 sweep 取最优验证 checkpoint —— 对照设计较公平。
  - "看着强但没回答核心问题":
    - 3B 反而不如 SFT(−3.3),暴露方法对 base ICL 能力的强依赖;
    - Knowledge 语料只有约 200K tokens、窄域(2025 自然灾害),**大规模知识注入未验证**;
    - §5 自承"难以把非推理模型改造成显式 CoT 模型"(方法不擅长那种需要"生成模式根本性改变"的适配)。
- 假设与失效边界:
  - 显式假设【原文】:ICL 假设(示范条件化 ≈ 最优策略,§3.1 两条件:Optimality + Minimal Deviation);base 模型有足够强的 ICL 能力;有(逐 query 可配的)专家示范。
  - 隐式假设【推断】:示范是"专家级"且与 query 一一匹配(实验里 Science 用 GPT-4o 生成、Tool 用数据集自带);teacher 的"软化"不会把错误意图也软化进来(对噪声/非专家示范未测)。
  - 何时失效:【原文】小模型 ICL 弱(3B −3.3,§4.4/§5);需要根本改变生成模式时(非推理→推理,§5)。【推断】示范噪声大/非专家时;teacher EMA 跟不上快速分布漂移时(§A.3 用 student 自身当 teacher 会发散,暗示反馈环存在稳定性边界)。
- 祛魅总结【推断】:
  - 真贡献:**把"on-policy 少遗忘"从 RL 搬到"只有示范"的场景**,并用一系列干净消融(on-policy vs offline、text+answer vs 单一、EMA vs base/self、pass@k 排除熵坍缩、3B/7B/14B 单调)系统坐实——尤其 OOD 间接问题和 answer-only 保住长 CoT 两个结果很有说服力。定位(continual learning from demonstrations,从示范做持续学习)是其最大增量。
  - 包装/高估:核心 trick(ICL 条件化 teacher + on-policy 蒸馏)在 GKD/context-distillation 早有,新意主要在定位与实验;§3.1 的 inverse-RL 理论框架因 **KL 方向 errata** 与实测脱节,属"包装性理论"成分偏高〔基于 README errata〕;"establishing on-policy distillation as a practical path"在 200K-token 窄域知识 + 7B 上成立,但规模/领域外推被高估。
  - 低估之处:EMA teacher 的"反馈环稳定性"其实是个普适且重要的发现(self-as-teacher 会发散),却被放进了附录;"单轨迹/prompt 就够(多采样无益)"对成本敏感的持续学习很实用,也没被突出。

## 结构化抽取

- 🎯 机制速览6轴:
  - 学什么信号:示范条件化 teacher(EMA)的 **token 级分布**(per-token forward KL,据 errata;论文正文写的是 reverse-KL)。
  - 改什么:**参数**(full fine-tuning 整个模型;改 logits 分布)。
  - 何时改:**在线 per-step**(每步现采 student rollout、现算 teacher 分布、现更新 + EMA 同步)。
  - 免梯度?:否。
  - 记忆-技能生命周期:**无显式记忆/技能库**;"新知识/新技能"直接写进**参数**(知识内化进权重,由 OOD 间接问题验证);示范 c 是 per-query 的临时 context,不做持久存储。共享/遗忘都靠参数本身。
  - 防遗忘机制:**核心卖点且实测**——on-policy 更新让学生保持贴近预训练分布(§3.2:KL \(0.68\) vs SFT \(1.26\) nats),从而在 6 个通用基准上保持(Fig.4 Pareto、Table 5 分项)+ 顺序学不退化(Fig.3)。这是"靠 on-policy + anchoring 隐式防遗忘",不是 EWC/replay 那种显式机制。
- ⑦ 开源代码 + 框架/harness:https://github.com/Continual-Intelligence/Self-Distillation (README 现给此 clone URL;v1 记 idanshen/Self-Distillation,疑为迁移/镜像;项目页 idanshenfeld.com/SDFT)。已 clone。框架 **TRL**(README 装 trl;`distil_trainer.py` 定义 DistilTrainer,用 `trl.extras.vllm_client.VLLMClient` 做 on-policy 生成;遗忘指标用 EleutherAI lm-eval-harness 指定 commit 03c44adc)。入口 `main.py`(README 示例 `--model_name Qwen/Qwen2.5-7B-Instruct --learning_rate 5e-5 --num_train_epochs 2`);配置 `distil_config.py`(alpha 控 forward/reverse/JSD、beta 控 KL 系数与是否载 reference、EMA ref-sync)。
  - **〔待核·关键〕KL 方向**:README errata(04/07/26)明言全部结果实为 **forward KL(GKD 式)**,且仓库默认即 forward KL;论文正文写 reverse KL。复现以仓库默认(forward KL)为准。
  - 〔待核〕lr 默认:README 训练示例用 **5e-5**;v1 分析曾记 main.py 默认 2e-5。本轮未重新打开 main.py 源码核对默认值,以 README 示例(5e-5)记录、标待核。
- 💰 资源/成本与可扩展性:单卡 **NVIDIA H200**(附录 B);full fine-tuning;**单 on-policy rollout/prompt**(附录 A.1 实测多采样无益)。
  - 相比 SFT 约 **2.5× FLOPs、约 4× wall-clock**(因为要 on-policy 生成,§5 Computational Costs);
  - 但若计入 Re-invoke 那种"先 SFT 再恢复"的多阶段流程,SDFT 反而可能更省总时间;
  - 比 GRPO 省(单生成 vs group 采样),且 token/logit 级的信用比 GRPO 的轨迹级 advantage 更稠密(§5)。
- 🎯 对"探索-巩固"对标:**强支撑(尤其"巩固/防遗忘"维度)+ 可借组件**。
  - 判定:SDFT 几乎是我们"巩固"分支的现成范式——"把成功经验/新技能固化进参数且不遗忘",且证明了**on-policy + 贴近预训练分布**是少遗忘的关键机制(正中我们 idea 里"巩固=固化进参数/记忆且不遗忘")。依据:Fig.3 顺序学不退化 + §3.2 的 anchoring 机制。
  - 可直接借的积木(逐条):
    - ① **EMA self-teacher** 作为稳定的"自蒸馏 teacher"(避免 self-as-teacher 发散,可移植到 student 自选恢复分支后的固化阶段);
    - ② "示范条件化 → on-policy 蒸馏":这是把任意 privileged context(我们的 teacher 脚手架/MTP 前瞻提示)转成 on-policy 信号的通用配方;
    - ③ pass@k 不坍缩 + OOD 间接问题,可作为"真内化 vs 死记"的探针。
  - 缺口:SDFT 偏"巩固",**探索/选路**几乎不涉及(它假设有专家示范、不做岔路口的路径选择);也没有 MTP/前瞻,没有 student 自选恢复分支。
- 🔭 开放问题/未来方向:
  - 【原文】(§5/§11)与 on-policy RL 结合(SDFT 当 RL 前的初始化,或同时混合示范+reward 信号);进一步降残余遗忘(on-policy 仍有少量退化);从**非专家/噪声示范**或非结构化数据(如用户对话)学习;更原理性地解决"继承 teacher 口头禅"(替代 mask 前几 token 的启发式);让方法能支持更激进的行为改变(如非推理→推理)。
  - 【推断】把 §3.1 理论与实测 KL 方向对齐(改写成 forward-KL 的解释,或换实现验证 reverse-KL 是否同样有效);大规模、跨多领域的知识注入可扩展性验证;对 EMA 率 \(\alpha\) 做稳定性-收敛的系统刻画(目前只给"base/self/EMA"三档定性)。

RETURN:sdcl | 读PDF? 是(21页全文+附录A.1三估计器/A.3 EMA消融/B设置) | 加厚? 是(方法到可复现:双角色数据流+模板原文+Eq.1-6全推导+三KL估计器全式+IRL等价链;相关工作四路线A-D精确短板) | LaTeX公式条数 14 | 待核数 2(KL方向errata、lr/α默认值)

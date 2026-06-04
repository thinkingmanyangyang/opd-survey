safesteer | SafeSteer: Localized On-Policy Distillation for Efficient Safety Alignment | 北航 + 北理工 + 北邮 + 北大 + 中科院自动化所 + 上海AI Lab + BAAI(Hao Li/Jingkun An/Zijun Song 共同一作;通讯 Lei Sha) | 2026-06-01 v1 · arXiv preprint(cs.AI,投 EMNLP 2026) | L1 OPD/自蒸馏(安全对齐应用) · 相关性中(方法层高、主题外围)

**原始论文**:https://arxiv.org/abs/2606.02530

## 一眼看懂

> 一句话导读:做安全对齐通常会连累通用能力(这叫 alignment tax)。本文的看法是:安全特征其实只占输出分布里很少一撮 token、跟通用 token 基本不重叠,所以对齐应该是"只动那一小撮"的局部修改,而不是"安全 vs 能力"的全局权衡。

- 🟦 TL;DR:先说现状。安全对齐常以牺牲通用能力(alignment tax,即"对齐税")为代价,现有方法靠混入海量通用数据、或借助辅助的 reward model 来做这种双目标权衡。
  - 本文论点:**安全特征在输出分布中本就稀疏、且与通用 token 大体不相交,所以对齐应当是"局部修改"而非"全局权衡"**。
  - SafeSteer 的做法:用 activation steering(激活引导,即往中间层注入一个 refusal direction \(d\)、即"拒答方向")造一个能稳定拒答的"安全 teacher" \(\pi_t\);再用"对比 log 概率 + 投票"挑出对该方向最敏感的那一小撮安全 token 子集 \(S\);最后只在 \(S\) 上施加 reverse-KL OPD。
  - 代价极小:只用 **100 条**有害样本、零通用数据,就能在几乎不掉通用能力的前提下显著降低 ASR(attack success rate,攻击成功率)。【原文】Abstract、§1、§3
- 最巧的一步:**把 reverse-KL 惩罚限制到稀疏安全 token 子集 \(S\)(Eq.6,只对 \(v\in S\) 求和)**。
  - 抽掉它(退回全词表 KL,即消融 w/o safety token)→ 通用能力 token 也被一起惩罚 → alignment tax 就回来了(Table 4:MATH + HumanEval 均值在所有模型 / 温度下都更低)。
  - 论文的核心论点"局部而非全局"正是靠这一步落地的。【原文】§3.4、Table 4

## 为什么做

> 一句话导读:现有安全对齐方法不是要海量通用数据、就是要辅助模型或正交投影,成本都高;标准 OPD 又得有个更强的外部 teacher、还会惩罚整个词表连累通用 token。SafeSteer 想同时解决"teacher 从哪来"和"别惩到通用能力"这两件事。

- **研究背景的来龙去脉**:LLM 被广泛部署到对话 / agent 中,但容易产出有害内容(偏见、犯罪建议、越狱),所以安全对齐很必要;然而主流方法把安全当成一个**双目标优化**问题,这就导致了 alignment tax(通用能力退化)。【原文】§1
- **并行技术路线 + 各自具体短板**:
  - **SFT 类**(Qi 2024 / SafeRLHF 的 SFT 分支):能力退化严重。
  - **偏好 / 约束优化类**:
    - BFPO(Zhang 2025)靠**混入海量通用数据**来防灾难性遗忘;
    - SafeRLHF / MoCAN(Huang 2024)在安全或能力约束下最大化期望奖励,需要**辅助 reward model**;
    - NSPO(Niu 2025)把安全梯度**正交投影到通用能力表征的零空间**,需要估计通用能力子空间;
    - DPO-Mix(helpfulness + safety 各 50/50)。
    - 共性短板:要么海量通用数据、要么辅助模型、要么正交投影,**成本都高**;而且 DPO-Mix 这类朴素混数据甚至会**升高 ASR**(§4.2 实证)。
  - **OPD 类**:标准 LLM OPD(Lu & Lab 2025)用 token 级 reverse-KL,比 outcome RL(GRPO)训练更高效、靠 on-policy rollout 缓解遗忘、靠 reverse-KL 的 mode-seeking(只追单一主峰)避免 mode-covering(摊平到多个峰)——但它**需要一个更强的外部 teacher**;近期的自-teacher 方案(Zhao 2026b / SDFT Shenfeld 2026)用专家示范构造自教师,却**重度依赖模型的 in-context learning(ICL,上下文学习)能力**。
  - **表征工程**(Arditi 2024:发现 refusal 由单一方向调控;以及 Circuit Breaker):证明了安全神经元是稀疏的、可用 activation 来操控行为——但 Circuit Breaker 这类方法仍需要大量通用数据。【原文】§2.1-2.3、§1
- **三个具体痛点(本文靶子)**:
  - (1) 把安全当双目标优化,成本高;
  - (2) 标准 OPD 需要外部强 teacher,而自-teacher 又依赖 ICL;
  - (3) **即便有好 teacher,标准 OPD 对整个词表 \(V\) 都施加惩罚**,会波及那些与安全大体不相交的通用能力 token。【原文】§1
- **动机链**:安全特征稀疏、且与通用 token 大体不相交 → 全局惩罚反而伤通用能力 → 应该只在安全相关的稀疏 token 上更新 → 这就需要两样东西:一个稳定的安全信号源(steering teacher,免训练、免 prompt engineering)+ 一个稳健的稀疏 token 挑选器(对比 + 投票,防止个别极端 logit 差主导排序)。【原文】§1、§3
- **本文站在谁肩上 + 与最近邻的精确差异**:站在 Arditi(2024)的 refusal-direction + 标准 OPD 的 reverse-KL 之上,改两点——
  - ① teacher 来源:用 **activation steering 自造**(无需更强的外部模型 / 无 prompt engineering / 无 ICL 依赖),解决"teacher 从哪来";
  - ② 惩罚范围:从**全词表 reverse-KL** 收窄到**只惩稀疏子集 \(S\)**(即 token-localized,token 局部化),解决"惩罚波及通用能力"。【原文】§1、§3

## 怎么做 + 靠不靠谱

> 一句话导读:三步——先给 base 模型硬注入拒答方向造一个"逢问必拒"的安全 teacher;再用 teacher 比 base 明显更想吐的那些 token 投票选出 50 个安全 token;最后只在这 50 个 token 上让学生向 teacher 看齐。要冷静看:增益主要体现在"几乎不掉通用能力",安全分本身没有大幅领先;且这 50 个 token 是离线静态选定的,对新型越狱的覆盖性存疑。

- **方法流水线(输入→输出,逐模块)**:
  1. **构造 steering 安全 teacher \(\pi_t\)**(§3.1):按 Arditi 2024 的做法,用 160 条有害 + 160 条无害指令对比隐表征,提取出 refusal direction \(d\in\mathbb{R}^{d_{\mathrm{model}}}\) 和注入层 \(\ell\)。
     - 在第 \(\ell\) 层注册一个 forward pre-hook,把进入该层的残差流 \(h_\ell\) 替换为
       \(\displaystyle h^\star_\ell=h_\ell+d,\)
     - 对**所有 token 位**持续注入 → \(\pi_t\) 对有害、无害都**稳定拒答**(它在设计上就会 over-refuse,即过度拒答)。
     - **输出**:一个免训练、稳定拒答的 teacher。
  2. **挑稀疏安全 token 子集 \(S\)**(§3.3):在 harmless 指令(Alpaca,160 条)上用 \(\pi_t\) 采 \(N\) 条拒答轨迹(每条长度 \(\le H\))。
     - 对第 \(n\) 条轨迹的每个有效 step \(j\)、每个 token \(v\in V\),算 teacher 相对 base 的**对比 log 概率**(意思是"teacher 比 base 多想吐这个 token 多少"):
       \(\displaystyle \Delta^{(x,n)}_j(v)=\log\frac{p_t(v\mid x,r^{(n)}_{<j})}{p_0(v\mid x,r^{(n)}_{<j})};\)
     - 每个位置取 top-\(K'\) 进候选集 \(C^{(x,n)}_j\),再跨 harmless 数据 / \(N\) 条轨迹 / 各 step 做**投票聚合**,选出得票最高的 \(K\) 个:
       \(\displaystyle \mathrm{vote}(v)=\sum_{x\in D_{\mathrm{harmless}}}\sum_{n=1}^{N}\sum_{j=1}^{H_{x,n}}\mathbb{1}\big[v\in C^{(x,n)}_j\big],\qquad S=\arg\max_{S'\subset V,\,|S'|=K}\sum_{v\in S'}\mathrm{vote}(v).\)
     - 取 \(|S|=50\)。**输出**:离线固定下来的 50 个安全 token。
     - **为何用投票、而非直接取单点最大的 \(\Delta\)**:离散投票 + 多条 rollout 能防止个别极端的 \(\Delta\) 主导整个排序。
  3. **token-localized reverse-KL OPD**(§3.4):student \(\pi_s\) 在有害指令(PKU-SafeRLHF,**仅 100 条**)上 on-policy 采 \(M=8\) 条响应 \(\{y^{(m)}\}\)。
     - 标准 OPD 在全词表上算 \(L^{(m)}_t(\theta_s)=\sum_{v\in V}p_s(v)\log\frac{p_s(v)}{p_t(v)}\)(Eq.5);而 SafeSteer **只对 \(S\) 求和**:
       \(\displaystyle L^{(m)}_t(\theta_s)=\sum_{v\in S}p_s(v)\log\frac{p_s(v)}{p_t(v)},\qquad L(\theta_s)=\mathbb{E}_{b,m}\Big[\frac{1}{T^{(b,m)}}\sum_{t=1}^{T^{(b,m)}}L^{(b,m)}_t(\theta_s)\Big].\)
     - 即按有效步平均、再对 batch 与 \(M\) 条 rollout 取期望。
     - **输出**:只在那 50 个安全 token 上更新过的学生 \(\pi_s\)。
- **逐组件必要性(基于真实消融)**:
  - **steering teacher(Eq.1)vs system-prompt teacher**:Table 3 证明 steering teacher 把平均 ASR 压到近 0.00%,而 prompt-based 版本的 ASR 显著更高——因为 prompt 难维持稳定拒答,而 steering 直接在表征空间诱发拒答,更稳。【原文】§4.3、Table 3
  - **localized reverse-KL(Eq.6)vs 全词表(w/o safety token)**:Table 4 证明去掉局部化后,在所有模型 / 温度下 MATH + HumanEval 的均值都更低(即通用能力受损)。【原文】§3.4、Table 4
  - **reverse-KL vs forward-KL**:reverse-KL 的 mode-seeking 更适合收敛到单一、明确的拒答模式;消融换成 forward KL 后,通用能力更低。【原文】§3.4、Table 4
  - **rollout model 必须用 \(\pi_t\)(§5.2)**:如果改用 base \(\pi_0\) 采轨迹来挑 token,Llama-3-8B-Instruct 在 JailbreakBench 上的 ASR 会从 1.0% **暴增到 75.0%**。
    - 原因:此时 \(S\) 被格式 token(如 `\n\n`)主导,模型只学会插段落、不学拒答;
    - 根因:Llama 是浅层对齐(拒答集中在前几个 token),\(\pi_0\) 在更深的位置无法维持拒答,导致对比 log 概率失效。
    - **这是"为何必须用 steering teacher、而不能直接拿 base 对比"的硬证据。**【原文】§5.2
  - **response length \(H\) 把"表层对齐"变成"语义安全"(§5.2)**:
    - \(H=1\) 时,\(S\) 只含初始拒答词(I / Sorry / Unfortunately,编码的是"如何开始拒");
    - \(H=7\) 时,\(S\) 移向语义安全概念(illegal / unethical / harmful,编码的是"为何拒")。
    - 适当的 \(H\) 是实现"深度安全对齐"的关键。【原文】§5.2、Fig.4
  - **不对概率切片重归一化(§5.2)**:Eq.6 用的是 \(S\) 上的原始概率切片 \(p_t(v),p_s(v)\),**不**做 softmax 重归一化成合法分布。
    - Table 5 证明:强制重归一会显著掉通用能力(Llama-3.2-3B 掉 4.76、Qwen2.5-7B 掉 3.04)。
    - 机理:不归一时,reverse-KL 仍在约束 \(p_s(v)\) 的**绝对概率质量**(起 magnitude anchor、即"量级锚"的作用);一旦归一,optimizer 就会激进地抬高 \(S\) 内的 logits、挤占其余 token 的概率、扭曲通用分布。
    - **这意味着 Eq.6 严格说是"对 \(S\) 上概率质量比的局部惩罚",并非严格的全分布散度——但恰恰是这个"不归一"特性才防住了 alignment tax。**【原文】§5.2、Table 5
- **关键机制(直觉)**:
  - 安全 teacher = base 的残差流被"硬塞"了拒答方向;
  - 安全 token = "teacher 比 base 明显更想吐出来"(对比 log 概率 \(\Delta\) 高、且跨轨迹得票高)的那些 token;
  - 训练 = 只在这些 token 上让学生向 teacher 看齐,且只约束其绝对概率质量。【原文】§3.1/3.4
- **训推数据如何流动**:
  - ① 离线:160 条 harmless × \(\pi_t\) rollout(每条 \(\le H\))→ 算 \(\Delta\) → 取 top-\(K'\) → 投票 → 得 \(S\)(50 个 token,**固定**)。
  - ② 在线:100 条 harmful × \(\pi_s\) on-policy 采 8 条 → 对每步在 \(S\) 上算 reverse-KL → 按有效步平均反传。
  - teacher 全程免训练(推理时注入 \(d\) 即可)。【原文】§3.2-3.4
- **实验与证据**:模型为 Llama-3-8B-Instruct、Llama-3.2-3B-Instruct、Qwen2.5-7B-Instruct、Qwen3-4B-Instruct-2507。
  - 安全 benchmark(7 个,ASR% 越低越好):AdvBench、PKU-SafeRLHF、HarmBench、JailbreakBench、SORRY-Bench(有害)+ HarmfulQA、ALERT(红队)。
  - 通用能力(5 个,越高越好):MMLU(STEM)、AlpacaEval、GSM8K、MATH、HumanEval。
  - 训练数据:harmless 用 Alpaca(用于选 token)、harmful 用 PKU-SafeRLHF 的 **仅 100 条**(不到基线量的 1%;对照 NSPO 用了 40% 全集、其余方法用全集)。
  - 其他设置:判官默认 Llama-Guard-4-12B(SORRY-Bench 用其原生 scorer);lr Llama 系 \(1\times10^{-6}\)、Qwen 系 \(1\times10^{-5}\);\(M=8\) rollout、\(|S|=50\)。
  - 关键数字(Table 1,t=0):
    - Qwen3-4B:ASR 平均 **2.87→0.91**(全场最低;对比 MoCAN 1.13、NSPO 1.18、BFPO 1.14),通用 Avg **71.68→71.39**(几乎无损);
    - Qwen2.5-7B:ASR **4.59→1.48**(最低,约 2.5× 优于最强基线 BFPO 的 3.75),通用 68.90→69.17(反而升了);
    - SORRY-Bench 上残留 ASR 偏高(Qwen3-4B 5.91、Qwen2.5-7B 7.27);
    - Llama 系上,SafeSteer 安全有竞争力、通用几乎不损;而对照方法 W-DOOR 把两个 Llama 的通用 Avg 砍到约一半(63.61→34.04 / 63.64→37.82)。【原文】§4.2、Table 1-2
  - 关键发现(PCA,Fig.3):\(\pi_t\) 在通用 prompt 上会 over-refuse、导致表征大幅 shift,但训练后的 \(\pi_s\) 相对 \(\pi_0\) 在通用表征上**几乎完全重叠、没有 shift**——说明 token-localization 成功蒸出了安全特征,却**没吸收 teacher 的 over-refusal**。【原文】§5.1、Fig.3
  - baseline 公平性:对比 DPO-Mix / MoCAN / W-DOOR / BFPO / NSPO,用同一批 benchmark;但判官与超参对 ASR 都敏感。【原文】Table 1
- **假设与失效边界**:
  - 【原文·Limitations】依赖 base 模型**本身已具备 refusal 能力**(instruction-tuned 模型成立,纯预训练 checkpoint 不行)——因为 \(\pi_t\) 是 steering base 得来的、\(S\) 也是从它的 rollout 里挖的。
  - 【原文·Limitations】所有实验都 ≤10B 参数,大模型上的有效性与超参未验证;且仅限 text-only 自回归 LLM,没覆盖 VLM / 扩散 LLM。
  - 【原文】安全 token 子集 \(S\) 是**离线固定**的(基于 Alpaca harmless + \(\pi_t\) 拒答轨迹)。
  - 【推断】\(S\) 的静态性是主要风险:对新型越狱 / 分布外有害模式的覆盖性存疑(SORRY-Bench 上残留 ASR 偏高就是旁证)。
- **祛魅总结**:
  - 真贡献:把"安全特征稀疏"这个观察转成了"token-localized OPD"的极简方案,用 steering 自造 teacher 解决了"teacher 从哪来",仅 100 条样本 + 零通用数据就达到了最优的安全–能力权衡;§5.2 的三组分析(rollout 必须用 \(\pi_t\)、\(H\) 把表层变语义、不重归一以保 magnitude anchor)再加 Fig.3 的 PCA("不吸收 over-refusal")构成了扎实的机制证据。【推断】
  - 包装 / 边界:增益主要在"几乎不掉通用能力",而非"安全分大幅领先"(部分 benchmark 的 ASR 已被基线压得很低);\(S\) 静态、领域窄(只针对二元安全),远不是通用对齐方案;Eq.6 严格说也只是"\(S\) 上概率质量比的局部惩罚",并非完整散度。【推断】

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号:对比 log 概率 \(\Delta\)(steering teacher vs base)+ 跨轨迹投票挑出的稀疏安全 token \(S\) + 这些 token 上的 reverse-KL。
  - 改什么:仅安全 token 子集 \(S\) 上的输出分布(token-localized 参数更新,且约束绝对概率质量)。
  - 何时改:\(S\) 离线一次性选定;训练阶段在线对 harmful rollout 做 reverse-KL(只惩 \(S\))。
  - 免梯度?否,梯度蒸馏(token-localized reverse-KL OPD)。teacher 本身免训练(activation steering,推理时注入 \(d\))。
  - 记忆-技能生命周期:无外部记忆/技能库;"安全 token 集 \(S\)"是离线提炼的静态知识。
  - 防遗忘机制:**核心即一种防遗忘**——把更新局部化到稀疏安全 token、不动通用 token(PCA 证通用表征无 shift),以极小代价避免 alignment tax;但非 replay/EWC,而是"空间隔离"。
- ⑦ 开源代码+框架/harness:https://github.com/Anjingkun/SafeSteer(v1 已克隆约 8.9MB;项目页 anjingkun.github.io/SafeSteer/)。框架 = 自研的轻量安全对齐框架,关键文件:`distil_trainer.py`(localized reverse-KL OPD)、`distil_config.py`、`main.py`、`model_utils`(steering hook / refusal direction)、`scripts`、`data`;通用能力评测用 lm-eval。【原文】Abstract + 仓库(v1 核查)
- 💰 资源/成本与可扩展性:**极轻量**——仅 100 条有害样本、零通用数据、无 reward model、无正交投影;teacher 免训练(推理时注入方向即可);\(M=8\) rollout、\(|S|=50\);适合快速部署。具体 GPU 数原文未说明(模型均 ≤10B)。【原文】Abstract、§1、§4.1
- 🎯 对"探索-巩固"对标:**可借组件(方法同源),非主题竞品**。判定:
  - 与本项目"在稀疏 / 局部 token 上做选择性监督"高度同源——SafeSteer 的"用对比 \(\Delta\) + 投票挑出该被 teacher 监督的稀疏 token 子集 \(S\),再只在该子集上做 reverse-KL"可直接迁移到 TSRD 的"挑选承担 path-recovery / 巩固作用的关键 token,并只在其上施加 teacher 监督"。
  - 它还示范了"局部更新 → 防遗忘(保住通用能力)"(对应本项目"巩固进参数且不遗忘");而且**"不重归一以保 magnitude anchor"是一个可借的工程教训**(做局部 KL 时别 softmax 重归一,否则会挤占其余 token)。
  - 缺口 / Δ:SafeSteer 的 \(S\) 是**离线静态、按 refusal direction 一刀切**的,而本项目要的是**随轨迹动态、按"第一处走偏点"在线定位**;且 SafeSteer 的 teacher 是 activation-steering 出来的、不是真正更强的 teacher,也不涉及探索 / 选路与 MTP 前瞻。
  - 依据:§3.3-3.4、§5.2。【推断】
- 🔭 开放问题/未来方向:【原文·Limitations】大模型有效性、VLM/扩散 LLM 迁移、对纯预训练 checkpoint 的适配。【推断】安全 token 集 \(S\) 的在线/动态更新以覆盖新越狱;把"局部 token 选择"推广到价值观/长程行为等非二元安全的复杂对齐;SORRY-Bench 等残留高 ASR 类别的针对性增强;把"对比 \(\Delta\)+投票选 token"用于通用 OPD 的选择性监督(对接本项目)。

RETURN: safesteer|读到PDF?是(19页,_txt 69k字,Eq.1-7 + Table 3/4/5 消融 + §5.2 三机制分析全核)|L1(安全应用)|对标=可借组件:对比+投票挑稀疏监督 token + 局部更新防遗忘 + 不重归一保 magnitude anchor→TSRD;Δ=S 离线静态,非动态单点接管,无 MTP|残留待核 0(关键数字/消融/PCA/renorm 均在 PDF 正文核到)

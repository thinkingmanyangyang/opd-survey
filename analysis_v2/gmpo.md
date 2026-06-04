gmpo | Geometric-Mean Policy Optimization (GMPO) | UCAS / CUHK / HKUST / Microsoft Research（多数实验在 MSR 实习完成） | arXiv 2507.20673v3, 2025-10-18 · ICLR 2026 接收 | 主题线 L3（RLVR/GRPO 稳定性）·相关性 中

**原始论文**：https://arxiv.org/abs/2507.20673

## 一眼看懂
- 🟦 TL;DR：GRPO 优化的是 token 级奖励的**算术平均**,对离群的重要性采样比 \(\rho_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})}\) 极敏感——某些 token 的 \(\rho_{i,t}\) 飙到极值,把整条梯度拽歪,引发激进更新→不稳定甚至崩溃【原文 §1, 图1】。GMPO 即插即用地把算术平均换成**几何平均**(在 log 空间做乘积+裁剪),几何平均天生抗离群、\(\rho\) 分布方差更低,于是能用**比 GRPO/DAPO 大得多的裁剪窗口** \((e^{-0.4},e^{0.4})\) 鼓励探索而仍稳定;7B 上五个数学基准平均 Pass@1 比 GRPO 高 4.1%【原文 摘要, 表1】。
- 最巧的一步：把聚合算子从「算术平均」换成「几何平均」(等价于 log 空间算术平均)。抽掉它就退化回 GRPO——单个极端 \(\rho_{i,t}\) 在乘积/对数和里被「摊薄」(\(1/|o_i|\) 次幂),不再单点主导梯度【原文 §3, 式3/6】。**几何平均这步是整篇的承重墙**：正因为聚合后方差小,才敢放开裁剪窗;裁剪放宽又是性能增益来源。两者是因果链而非独立 trick。

## 为什么做
- 研究背景：test-time scaling 把长 CoT-RL(GRPO 系)推成数学/代码推理主流范式;DeepSeek-R1([Guo 2025a])用可验证奖励证明了 RL 后训练能显著提升推理。但 **RL 训练稳定性本身**研究不足,而它是可靠、可 scale 后训练的前提(§2.1 末句:"the stability of RL for LLMs remains underexplored, yet it is essential")。
- 解决的具体痛点:
  - (1) GRPO 算术平均对离群 \(\rho_{i,t}\hat A_i\) 敏感→激进更新→进一步放大 \(\rho\) 方差→恶性循环、性能退化【§1, §3】;
  - (2) GRPO 用窄 clip \((0.8,1.2)\) 压制离群,副作用是**限制探索、过早确定性策略、熵快速塌缩、性能 plateau**【§1, 引 Yu 2025 DAPO】。两难:要么不稳,要么不敢探索。
- 相关工作 & 各自不足(§2.1 罗列一长串 GRPO 变体,可分几束):
  - **去 critic / 去 surrogate / 去 KL 这束**:GRPO([Shao 2024])去掉昂贵 value model;GPG([Chu 2025])进一步去 surrogate loss、critic、KL 约束;GVPO([Zhang 2025a])给解析 KL-约束权重;INTUITOR([Zhao 2025])用模型自置信度去掉外部 reward;PRIME([Cui 2025a])提供可扩展的隐式过程奖励 RL 框架。
  - **rollout 选择 / 偏置纠正这束**:SRPO([Zhang 2025c])历史重采样、DAPO([Yu 2025])动态采样 + clip-higher、Dr.GRPO([Liu 2025])去长度偏置、OPO([Hao 2025])最优 baseline 降梯度方差;PODS([Xu 2025])只训信息量大的子集、RePO([Li 2025])replay 拉回 off-policy 多样样本、RAFT([Xiong 2025b])只训正样本却能匹敌 GRPO。
  - **reward 塑形 / advantage 估计这束**:EMPO([Zhang 2025b])语义熵、AAPO([Xiong 2025a])advantage 动量、BNPO([Xiao 2025])Beta 分布自适应归一、Seed-GRPO([Chen 2025])按问题不确定性缩放更新、GRPO-lead([Zhang & Zuo 2025])长度依赖准确率/显式惩罚/难度重加权治稀疏奖励。
  - **效率 / 长度控制这束**:CPPO 剪低 advantage、S-GRPO 早退、Ada-GRPO 自适应推理格式、GRPO-λ 在长度惩罚/长度无关奖励间动态切换防崩。
  - **探索 / 熵这束**:80/20 规则([Wang 2025])强调高熵少数 token 主导、entropy-adv([Cheng 2025])熵视角 advantage 增强。
  - **数据这束**:Open-Reasoner-Zero([Hu 2025])129k 课程式数据、Eurus([Yuan 2024])大规模对齐数据 + 新 reward modeling。
  - 作者归纳:**多数变体聚焦采样/优势/奖励塑形,少有人正面攻「训练稳定性」这个根因**——这正是 GMPO 的切入口。
- 动机链：GRPO 强但不稳 → 不稳源于算术平均对离群敏感 → 几何平均对离群天然鲁棒、方差更低 → 用它替换即可既稳又能放开裁剪促探索 → 兼得稳定与探索。
- 本文站在谁肩上 & 与最近邻工作的精确差异：基线/setup 直接承袭 **Dr.GRPO**([Liu 2025])(同训练数据、同评测、忽略对 ref 的 KL 项)。最近邻是 **DeepSeek-R1** 的**序列级**乘积 + 序列级 clip(R1 最大化 \((\prod_t\rho_{i,t})\hat A_i\) 并在序列级 clip)。GMPO 关键差异两点——
  - (i) **token 级裁剪**而非序列级(序列级一旦触发把整条序列**所有** token 梯度清零、丢有效信号;且序列级 \(\rho\) 范围反而更大、更易制造极端梯度,图3 消融证实);
  - (ii) 几何平均带 **\(1/|o_i|\) 幂归一化**(DeepSeek-R1 序列级乘积**无此项**,长响应下 \(\rho\) 爆炸,附录 C 图6 证实)。这两点是「为什么 GMPO 比直接抄 R1 序列级更好用」的答案。

## 怎么做 + 靠不靠谱
- **GRPO 基线目标(§2.2,式1/2)**：对每题 \(q\) 从 \(\pi_{\theta_{\mathrm{old}}}\) 采一组 rollout \(\{o_1,\dots,o_G\}\)、算奖励 \(\{r_1,\dots,r_G\}\),最大化(式1):
  \(\displaystyle J_{\mathrm{GRPO}}(\pi_\theta)=\mathbb{E}_{q,\{o_i\}\sim\pi_{\theta_{\mathrm{old}}}}\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big\{\min\big(\rho_{i,t}(\theta)\hat A_i,\ \mathrm{clip}(\rho_{i,t}(\theta),\epsilon_{\mathrm{low}},\epsilon_{\mathrm{high}})\hat A_i\big)-\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})\Big\},\)
  其中 \(\rho_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})}\)、\(\hat A_i=\frac{r_i-\operatorname{mean}(\{r_1..r_G\})}{\operatorname{std}(\{r_1..r_G\})}\)。跟 Dr.GRPO 一样**忽略 \(D_{\mathrm{KL}}\) 项**省显存,简化为算术平均形式(式2):\(J^{*}_{\mathrm{GRPO}}=\mathbb{E}\big[\frac{1}{G}\sum_i\frac{1}{|o_i|}\sum_t\rho_{i,t}(\theta)\hat A_i\big]\)。
- **GMPO 目标(§3,式3/4)**：把算术平均换几何平均(式3):
  \(\displaystyle J^{*}_{\mathrm{GMPO}}(\pi_\theta)=\mathbb{E}_{q,\{o_i\}\sim\pi_{\theta_{\mathrm{old}}}}\frac{1}{G}\sum_{i=1}^{G}\Big(\prod_{t=1}^{|o_i|}\rho_{i,t}(\theta)\hat A_i\Big)^{\frac{1}{|o_i|}}\!\cdot\mathrm{sgn}(\hat A_i),\)
  \(\mathrm{sgn}(\hat A_i)\) 确保优化方向正确(\(\hat A_i>0\) 返 1、否则 −1;因为对负 advantage 取偶/奇次根会丢符号,需显式纠向)。加入 token 级 clip 后的完整目标(式4):
  \(\displaystyle J_{\mathrm{GMPO}}(\pi_\theta)=\mathbb{E}\frac{1}{G}\sum_{i=1}^{G}\Big[\prod_{t=1}^{|o_i|}\min\big(\rho_{i,t}(\theta)\hat A_i,\ \mathrm{clip}(\rho_{i,t}(\theta),\epsilon_{\mathrm{low}},\epsilon_{\mathrm{high}})\hat A_i\big)\Big]^{\frac{1}{|o_i|}}\!\cdot\mathrm{sgn}(\hat A_i).\)
  **值域收缩(§3 不等式,这是"为何更稳"的第一层证据)**:由 AM-GM,
  \(\displaystyle |J^{*}_{\mathrm{GMPO}}|=\mathbb{E}\Big|\tfrac{1}{G}\textstyle\sum_i\big(\prod_t\rho_{i,t}\hat A_i\big)^{1/|o_i|}\Big|\ \le\ \mathbb{E}\Big|\tfrac{1}{G}\textstyle\sum_i\tfrac{1}{|o_i|}\sum_t\rho_{i,t}\hat A_i\Big|=|J^{*}_{\mathrm{GRPO}}|,\)
  即 GMPO 目标值域更窄 → 训练方差更低 → 更稳。
- **方法流水线(逐步 输入→输出)**：① 采样组内 \(G\) 条 rollout、算组内相对优势 \(\hat A_i\)(同 GRPO,免 value)→ ② 每条序列内对 token 级 \(\rho_{i,t}\hat A_i\) 取**几何平均**(乘积后开 \(1/|o_i|\) 次幂、乘 \(\mathrm{sgn}(\hat A_i)\))→ ③ token 级裁剪到 \((e^{-0.4},e^{0.4})\)(比 GRPO/DAPO 宽)→ ④ 最大化该目标,按标准 policy gradient 更新。
- **真实实现:全在 log 空间(Algorithm 1,~10 行,复现关键)**：把概率取对数 `new_log_probs, old_log_probs = log(new_probs), log(old_probs)`;算 `sgn_A = +1/−1`;`sgn_A_log_probs_diff = sgn_A * (new_log - old_log)`;**先按 sgn 把对数差 clamp 到 \([-\epsilon,\epsilon]\)**(\(\epsilon=0.4\)),再 `min(原差, clamp 差)`、乘回 sgn_A 得 `log_probs_diff_min`;几何平均 = `exp( sum(log_probs_diff_min[mask]) / mask.sum() )`(即对数和除 token 数再指数 = 乘积开 \(1/|o_i|\) 次幂);`loss = -advantage * importance_sampling_ratio`。→ 乘积/裁剪都在 log 空间做,避免连乘数值下溢/上溢。
- 逐组件必要性(表4 消融,5 行对比,均 Qwen2.5-Math-7B):
  - **几何 vs 算术平均**(行1 GRPO 51.2 → 行5 GMPO 52.7,+1.5%):核心增益来源。
  - **\(1/|o_i|\) 幂归一化**(行4 去归一 52.0 vs 行5 52.7,−0.7%):附录 C 图6 给机理——长响应下序列级 \(\rho\) 随长度爆炸,归一压住。有消融。
  - **裁剪策略**(行2 不裁剪 52.3 / 行3 序列级裁剪 52.6 / 行5 token 级 52.7):性能差异其实**很小(0.1~0.4%)**,但 token 级裁剪让 \(\rho\) 范围最稳(图3)。〔推断〕裁剪方式对最终分数贡献边际,主要价值在稳定性而非精度。
  - **裁剪窗口大小**(表5:\((e^{-0.2},e^{0.2})\) 52.4 / \((e^{-0.4},e^{0.4})\) 52.7 / \((e^{-0.8},e^{0.8})\) 52.1 / \((-\infty,+\infty)\) 52.3):\((e^{-0.4},e^{0.4})\) 是甜点;过宽不稳、过窄抑探索。有消融。
- **关键机制/公式:梯度视角(§3 + 附录 A 引理1-3,这是「为何稳」的数学根)**：两目标的梯度都是"各 token policy gradient \(\hat A_i\nabla_\theta\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})\) 的加权和",差在权重——
  \(\displaystyle \nabla_\theta J^{*}_{\mathrm{GRPO}}\big|_{q,o_i}=\frac{1}{G|o_i|}\sum_{t=1}^{|o_i|}\rho_{i,t}(\theta)\cdot\hat A_i\cdot\nabla_\theta\log\pi_\theta(o_{i,t}\mid q,o_{i,<t}),\)
  \(\displaystyle \nabla_\theta J^{*}_{\mathrm{GMPO}}\big|_{q,o_i}=\frac{1}{G|o_i|}\sum_{t=1}^{|o_i|}\Big(\prod_{k=1}^{|o_i|}\rho_{i,k}(\theta)\Big)^{\frac{1}{|o_i|}}\!\cdot\hat A_i\cdot\nabla_\theta\log\pi_\theta(o_{i,t}\mid q,o_{i,<t}).\)
  GRPO 里每 token 权重含**它自己**的 \(\rho_{i,t}\)——一个极端值就让该 token 梯度爆/灭;GMPO 里权重换成**整条序列所有 \(\rho\) 的几何平均** \((\prod_k\rho_{i,k})^{1/|o_i|}\)——任何单个极端 \(\rho\) 经 \(1/|o_i|\) 次幂被强力压缩,给出更均衡的更新信号。(推导用到引理1 \(\nabla_\theta\rho_{i,t}=\rho_{i,t}\nabla_\theta\log\pi_\theta(o_{i,t})\)。)
- 实验与证据:base=Qwen2.5-Math-1.5B/7B、R1-Distill-Qwen-7B、Qwen3-32B(MoE)、Qwen2.5-VL-7B;训练数据 MATH L3-5(8523 题)/DeepScaleR/CountDown/Geometry3K;评测 AIME24/AMC/MATH500/Minerva/OlympiadBench + Geometry3K。关键数字:R1-Distill-7B 上 **63.4 vs GRPO 59.3 (+4.1%)**;MoE Qwen3-32B MATH500 **96.7 vs 94.6 (+2.1%)**;多模态 Geometry3K **54.7 vs 53.3 (+1.4%)**。稳定性证据扎实:图4 显示 GMPO 全程更高熵、更小 KL、更稳梯度;图5(e) CountDown 上 GRPO ~250 步崩溃、GMPO 不崩。
- baseline 公平吗:主对比 GRPO/Dr.GRPO 在**同一套 Dr.GRPO 设置**下跑(每问 8 rollouts、max 3000 token、每轮 1024 rollouts、更新 8 次 batch 128),较公平;表3 与 SOTA(SimpleRL/PRIME/OpenReasoner/Oat-Zero/GPG)对比,GMPO-7B 52.7 居前但与 Oat-Zero-7B(51.4)差距不大、R1-Distill 版 63.4 vs Oat-Zero 61.5。**注意各 SOTA 训练数据/设置不一,非严格同条件**,应谨慎解读「outperform SOTA」。
- 假设与失效边界:【原文】语言任务温度=0 贪心单样本评 Pass@1(§4.1,多模态用 temp 0.5 采 16)——只评了贪心精度,未报多样性。【推断】(1) 增益主要来自「放开裁剪→保熵→不早塌」,若任务本就不易熵塌(如短输出、易题),几何平均优势会缩水;(2) 几何平均对**优势符号**敏感,需 \(\mathrm{sgn}(\hat A_i)\) 显式纠向,隐含「序列级 \(\hat A_i\) 同号」——若未来用 token 级/混合符号优势,几何平均形式需重推;(3) 实测最佳窗口 \((e^{-0.4},e^{0.4})\) 可能依赖模型规模/数据难度,换 setting 需重调。
- 祛魅总结:真贡献=**用一行算子替换换来「稳定×探索」双赢的清晰机理(图4 熵/KL/梯度三件套证据 + 梯度推导)**,且即插即用、可移植到 MoE/多模态——这是扎实的。包装/需打折的:「+4.1%」是在最易熵塌的 R1-Distill 上取的最大值,1.5B/7B-Math 上只 +1.4~1.5%;与 Oat-Zero/Dr.GRPO 等强 baseline 的纯精度差其实**边际**,核心卖点应理解为**稳定性**(尤其 MoE 防崩)而非精度跃升。【推断】对「探索-巩固」而言这是 RL 优化层的稳定器,非蒸馏/记忆机制。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=组内相对优势 \(\hat A_i\)(可验证 0/1 奖励,token 级 \(\rho\) 加权)｜**改什么**=策略参数 \(\theta\)(聚合算子从算术→几何)｜**何时改**=on-policy RL 训练中每步｜**免梯度?**=否,标准 policy gradient｜**记忆-技能生命周期**=无(纯参数更新,不涉记忆/技能库)｜**防遗忘机制**=间接——更小 \(D_{\mathrm{KL}}\)-to-base + 更稳梯度抑制过拟合/漂移(图4c-d),但非显式防遗忘设计。
- ⑦ 开源代码+框架/harness：https://github.com/callsys/GMPO （已 clone ~57MB,Tier A,readme 已核）。框架=**专用仓基于 oat-llm 0.1.3.post1 + vLLM 0.8.4**(构建于 sail-sg/understand-r1-zero / Dr.GRPO 之上,readme 第49行致谢);**另已集成进 veRL**(`volcengine/verl/examples/gmpo_trainer`,readme 第27行)。仓内含 `train_zero_math_gmpo.py`、`train_zero_math_hmpo.py`、scripts/、understand_r1_zero_main/。
- 💰 资源/成本与可扩展性：7B 以下模型在 **8×A800** 训练(§4.1);每问 8 rollouts、max 3000 token、每轮 1024 rollouts、更新 8 次 batch 128。相对 GRPO **无额外计算开销**(仅改聚合算子,log 空间连乘成本可忽略)。已验证可扩到 32B MoE(Qwen3-32B,128 experts/8 激活)。
- 🎯 对"探索-巩固"对标：**可借组件(RL 优化层),非直接对标**。判定依据：GMPO 的「放开裁剪窗 + 几何平均保熵」直接服务于「探索」——若 TSRD 的 student 在 on-policy RL/蒸馏阶段需要更长时间保持探索(不早塌缩到自己已会的开头),GMPO 的稳定器可即插即用接在 OPD/GRPO 之上。但它**不涉及 teacher 脚手架、选路/回轨、记忆固化**,对「巩固」一侧无贡献。缺口:几何平均会摊薄单 token 信号,与 path-recovery「单点关键 token 接管」的诉求**张力相反**(GMPO 想压离群、TSRD 想放大关键点),二者需权衡。
- 🔭 开放问题/未来方向：【原文】结尾仅泛泛说「为更可靠、可 scale 的 RL 系统铺路」,未列具体 open problem。【推断】(1) 几何平均与「高熵关键 token 应被放大」的诉求冲突,能否做**自适应/混合算子**(关键 token 用算术、常规 token 用几何)——正是 holderpo/hapo 在做的方向;(2) 最佳裁剪窗随规模/难度漂移,缺自适应机制;(3) 只在贪心 Pass@1 验证,对多样性/Pass@k 的影响未知。

key|读PDF?|方法&相关工作已加厚?|LaTeX 公式条数|残留待核数
gmpo | 是(全文16页) | 是(GRPO/GMPO 目标 + 值域 AM-GM 不等式 + log 空间 Algorithm 1 逐行 + 梯度对比引理;相关工作把 GRPO 变体分 6 束 + 与 R1 序列级精确差异) | 8(式1 GRPO、式2 GRPO*、式3 GMPO*、式4 GMPO clip、值域不等式、式5 GRPO 梯度、式6 GMPO 梯度、\(\rho\) 定义) | 0

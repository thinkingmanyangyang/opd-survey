hpt_upge | Towards a Unified View of LLM Post-Training (UPGE / HPT) | 清华大学 C3I (TsinghuaC3I) + 上海 AI Lab + WeChat AI | 2025-09-05 · arXiv 2509.04419 (v2, 2026-01-20) · preprint | 主题线 L2 统一SFT-RL(GFT类) · 相关性 高

**原始论文**:https://arxiv.org/abs/2509.04419

## 一眼看懂
- 🟦 TL;DR:把 SFT 和各种 RL 后训练(PPO/GRPO/REINFORCE/CISPO/GSPO/SRFT/LUFFY)的策略梯度统一写成同一个表达式 **Unified Policy Gradient Estimator(UPGE)`grad_Uni = 𝟙_stable · (1/π_ref) · Â · ∇π_θ`**(四个可互换组件:稳定化掩码、参考策略分母、优势估计、似然梯度)。论证 SFT 与 RL 不是对立范式,而是**同一个共同目标(Eq.1:最大化期望奖励 + 不偏离示范分布的 KL 约束)在不同数据分布假设/bias-variance 权衡下的实例**。据此提出 **Hybrid Post-Training(HPT)**:对每个 question 采 n=8 条 on-policy rollout、用 rule-based verifier 算准确率 P,P>γ 走纯 on-policy RL(Dr.GRPO)、P≤γ 在外部 supervising trajectory 上走纯 SFT,混合损失 `L=αL_RL+βL_SFT`(α,β 由 P 二值开关给)。
- 最巧的一步:**用 on-policy rollout 准确率 P 做实例级(per-question)的 RL/SFT 二值切换门(Eq.10,γ)**。抽掉它,HPT 退化为固定系数 / 预设 schedule 的混合(就是 LUFFY/SRFT 那类),失去"按模型当前在该题上的能力自适应"——而正是这个自适应让 HPT 同时拿到 RL 的探索(能自解→放手探索)和 SFT 的兜底(解不出→给示范),并在 §4.1 拿到最高 large-k Pass@1024。为什么是支点:它把"统一视角"这个理论洞见变成一个**零额外网络、几乎零成本**的可执行算法。

## 为什么做
- 研究背景:RL 让模型在后训练自由探索推理空间(但 "Zero RL" 对弱模型/难任务常因探索不到奖励而失败,§1);SFT 高效拟合高质量示范(但抑制探索、易过拟合、OOD 差)。于是 "SFT→RL" 两阶段成事实标准,但**贵且要精调**。
- 解决的具体痛点:① 已有"SFT loss + RL loss 复合"工作(LUFFY/SRFT/Zhang 等)都把两者当**两个不同目标**,用固定系数/schedule/熵/可学参数去配比,缺"为什么二者能在一个优化过程里结合"的理论分析(§1 原话"a detailed analysis of why these two learning signals can be effectively combined ... remains largely unexplored");② 两阶段范式遗忘 + 僵化、无法按能力/难度自适应。

- 相关工作 & 各自不足(§1 + §2.3 + Table 1，UPGE 把它们都收为四组件取值特例):
  1. **纯 RL（各自只改 UPGE 某组件）**:**PPO**(Schulman 2017)π_ref=π_θold、Â=GAE、有 clip mask;**GRPO**(Shao 2024)π_ref=π_θold、Â=组内 unit-std 归一;**REINFORCE**(Ahmadian 2024)π_ref=π_θ、Â=±1，1/π_θ **无偏但高方差**;**CISPO**(Chen 2025)token-wise CIS-mask;**GSPO**(Zheng 2025)π_ref=序列级 (π_θold/π_θ)^{1/|τ|}、Seq-Clip mask;**Dr.GRPO/RLOO/REINFORCE++** 主张只 re-center 不除 std(除 std 引难度偏置)。短板:都是**纯 RL**、未统一 SFT、无动态选组件机制。
  2. **混合策略(最近邻 + 直接 baseline + 代码基座)**:**LUFFY**(Yan 2025)off-policy 混 on-policy、π_ref≡1、**固定**混合比 + policy-shaping \(f'_{\text{shape}}\);**SRFT**(Fu 2025)offline，Â 在 on+off 合集上归一、固定混合(官方代码未公开，作者自实现复现);**Zhang 2025** 等用熵/可学参数配比。短板:**把 SFT 与 RL 当两个不同目标**、固定/预设配比、缺统一理论 + 不能按能力自适应。
  3. **稳定化掩码的演化(§2.3)**:PPO clip 首引 stop-gradient;**DAPO**(Yu 2025)放宽 clip 上界(认为 PPO 丢掉的大更新 token 其实关键);**CISPO**(Chen 2025)token-wise mask 更细;**Clip-Cov**(Cui 2025b)再加裁剪治熵崩;**GSPO** 指出 PPO-style clip 噪声大、它裁更多 token 反而更高效。这些都进了 UPGE 的 \(\mathbb{1}_{\text{stable}}\) 组件。
  4. **token 级优势(§2.3 末)**:Wang 2025/Yang 2025/Sun 2025 把 Â 下沉到 token 级 \(\hat A_{i,j,t}\)，UPGE 的 Â 组件留了这个口子。
  5. **理论起点**:Generalized Advantage Estimator(GAE, Schulman 2015b)——UPGE 自承受其启发，把多个 noisy 梯度估计当"互补滤波"(complementary filter, Marantos 2015)加权平均。

- 动机链:现状(SFT/RL 被当对立、两阶段贵)→ 缺陷(无统一理论 + 无法自适应)→ 把所有后训练梯度统一成 UPGE 四组件 → 证明同源可联合优化 → 导出按 rollout 表现动态混合的 HPT。
- 与最近邻工作的精确 Δ:vs **LUFFY/SRFT**:它们用**固定**off/on-policy 混合比，HPT 用 **per-question rollout 准确率动态二值切换**;且 §4.4 实验证明 off-policy RL 其实非必需(SFT/ON 41.9 > Mix/ON 40.3 > OFF/ON 38.1)，用普通 SFT 学 offline 数据就够。这个 Δ 关键且有用:动态比固定更贴合"模型在不同题/不同阶段能力不同"(Fig.6 弱 1.5B 久居 SFT 区、强 7B 早切 RL)。

## 怎么做 + 靠不靠谱
### UPGE 的推导(§2.2，全篇地基)
- **共同目标**(Eq.1)：最大化期望成功率 + 不偏离示范(behavior)策略 \(\pi_\beta\)：
  \(\displaystyle J_\mu(\theta)=\mathbb{E}_{\tau\sim\pi_\theta(\cdot\mid q)}\big[r(\tau\mid q)\big]-\mu\,\mathrm{KL}\big(\pi_\beta(\cdot\mid q)\,\|\,\pi_\theta(\cdot\mid q)\big),\quad \mu\ge0.\)
- **求导**(Eq.2)：得两项之和——reward 项(采自 π_θ) + data-adherence/SFT 项(采自 π_β)：
  \(\displaystyle \nabla_\theta J_\mu(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\big[r(\tau\mid q)\nabla_\theta\log\pi_\theta(\tau\mid q)\big]+\mu\,\mathbb{E}_{\tau\sim\pi_\beta}\big[\nabla_\theta\log\pi_\theta(\tau\mid q)\big].\)
- **测度变换 + ∇logπ=(1/π)∇π**(Eq.3)，引入 reference 策略 \(\pi_{\text{ref}}\)：
  \(\displaystyle \nabla_\theta J_\mu(\theta)=\mathbb{E}_{\tau\sim\pi_{\text{ref}}(\cdot\mid q)}\Big[\frac{1}{\pi_{\text{ref}}(\tau\mid q)}\,\hat A_{\text{uni}}(\tau,q)\,\nabla_\theta\pi_\theta(\tau\mid q)\Big].\)
- **统一优势**(Eq.4，关键)：SFT 不过是优势退化为重要性比、且 off-policy 常令 \(\pi_{\text{ref}}=1\) 的特例：
  \(\displaystyle \hat A_{\text{uni}}(\tau,q)=\underbrace{r(\tau\mid q)}_{\hat A_{\text{RL}}}+\underbrace{\mu\,\frac{\pi_\beta(\tau\mid q)}{\pi_\theta(\tau\mid q)}}_{\hat A_{\text{SFT}}}.\)
- **插入稳定化掩码**(乘性、不改目标)得 **UPGE**(Eq.6)：
  \(\displaystyle \mathrm{grad}_{\text{uni}}=\mathbb{E}_{\tau\sim\pi_{\text{ref}}}\Big[\mathbb{1}_{\text{stable}}(\tau,q)\,\frac{1}{\pi_{\text{ref}}(\tau\mid q)}\,\hat A_{\text{uni}}(\tau,q)\,\nabla_\theta\pi_\theta(\tau\mid q)\Big]=\mathbb{1}_{\text{stable}}\,\frac{1}{\pi_{\text{ref}}}\,\hat A\,\nabla\pi_\theta.\)
  四组件:① \(\mathbb{1}_{\text{stable}}\) 稳定化掩码(PPO clip 的 stop-gradient)；② \(1/\pi_{\text{ref}}\) 参考策略分母(token 级重加权——概率小=更重要=权重更大；SFT 用 1/π_θ、PPO 用 1/π_θold、offline 用 π_ref=1)；③ \(\hat A\) 优势(GRPO 归一见 Eq.5)；④ \(\nabla\pi_\theta\) 似然梯度(跨所有算法不变)。**SFT/RL 同源 ⇒ 可在一个 loss 里联合**。

### HPT 算法流水线(Algorithm 1，读完可复现)
每步 t、对 question q：
1. 采 n 条 on-policy rollout \(\tau_i\sim\pi_\theta(\cdot\mid q)\)。
2. verifier 打 0/1(Eq.7：含正确答案=1 否则 0)，算准确率 \(P=\tfrac1n\sum_{i=1}^n v(\tau_i)\)(Eq.8)。
3. 按门限 γ 取二值系数(Eq.10)：\(\alpha=f(P)=\mathbb{1}[P>\gamma],\ \beta=g(P)=\mathbb{1}[P\le\gamma]\)。即**能自解(P>γ)→放手 RL 探索；解不出(P≤γ)→给 SFT 示范**。
4. 算两个损失：
   - on-policy RL(用 Dr.GRPO 形式，Eq.11)：\(\mathcal{L}_{\text{RL}}=-\tfrac1n\sum_{i=1}^n\sum_{t=1}^{|\tau_i|}\min\big(r_{i,t}A_{i,t},\ \mathrm{clip}(r_{i,t},1-\epsilon,1+\epsilon)A_{i,t}\big)\)，其中 \(r_{i,t}=\pi_\theta/\pi_{\theta_{\text{old}}}\)、\(A_{i,t}\equiv A_i=\dfrac{R(\tau_i)-\mathrm{mean}}{\mathrm{std}}\)。
   - SFT(在外部 supervising trajectory \(\tau^\star\) 上，Eq.12)：\(\mathcal{L}_{\text{SFT}}=-\tfrac{1}{|\tau^\star|}\sum_{t=1}^{|\tau^\star|}\log\pi_\theta(\tau_t^\star\mid q,\tau_{<t}^\star)\)。
5. 混合损失(Eq.13)：\(\mathcal{L}=\alpha\mathcal{L}_{\text{RL}}+\beta\mathcal{L}_{\text{SFT}}\)，更新 \(\theta\leftarrow\theta-\eta\nabla_\theta\mathcal{L}\)。
- **关键超参默认值**：γ=0(Qwen 家族，即只在**全错**时才走 SFT)/ 2/8(LLaMA)；n=8 rollout、temp 1.0；lr=5e-6 AdamW(constant)；max gen 8192；8×A800 80GB。

### 逐组件必要性 + 消融
- **动态门 γ(实例级切换)**:有消融(§4.5,Table 6)。γ=0 平均 41.9 最优,> γ=1/8(38.7)、γ=2/8(39.0)→ "盲目加更多 SFT 不一定更好",动态平衡才关键。
- **on-policy RL 分支(vs 用 off-policy RL 替代)**:有消融(§4.4,Table 5)。SFT/ON(HPT)41.9 > Mix/ON 40.3 > OFF/ON 38.1 → off-policy RL 非必需,SFT 当 offline 学习器已够。
- **UPGE 四组件(稳定化掩码 / π_ref 分母 / 优势 / 似然梯度)**:是理论分解(Table 1),非可拆消融模块;作者用 §2.3 的 bias-variance 讨论论证每个组件的作用(如 π_ref=1 引入偏置换数值稳定;clip 减方差但引偏置)。【推断:理论分解充分,但 HPT 只实例化了"二值开关"这一个最简实例,四组件的更优组合留作 future work】。

- 实验与证据:
  - 模型:Qwen2.5-Math-7B / 1.5B、LLaMA3.1-8B。6 个数学 benchmark(AIME24/25、AMC、MATH-500、Minerva、Olympiad)+ Qwen 时加 GPQA-Diamond、ARC-c(OOD)。avg@32(AIME/AMC, temp 0.6 top-p 0.95),max gen 8192,8×A800 80GB。
  - 主结果(Table 2,Qwen2.5-Math-7B):HPT avg 52.7(ID)/62.3(OOD),超 SFT(44.5)、GRPO(43.1)、SFT→GRPO(46.5)、LUFFY(49.8)、SRFT(44.2)。**AIME24 比最强 baseline 高 6.9 分(33.0 vs LUFFY 26.1),比 SRFT 高 14.6 分**。Table 3:LLaMA3.1-8B avg 18.2(vs LUFFY 13.2)、Qwen-1.5B avg 41.9(vs LUFFY 34.7)。
  - 探索证据(§4.1,Fig.2,Pass@k 到 1024,bootstrap from 2048 解):带 SFT 的方法 large-k Pass@k 高于纯 GRPO(Limit-of-RLVR 现象, Yue 2025);**HPT 在 large-k 上最高**(不只 Pass@1)——同时提升性能与探索边界。Table 4:HPT 比 GRPO/LUFFY 多解的题随难度递增,且基本不丢已会的题(缓解灾难性遗忘)。
  - 训练动态(§4.3):Fig.6 offline 数据占比随训练自动下降(弱模型久居 SFT、强模型早切 RL,最终稳在小而非零);Fig.7 HPT 熵持续高于 GRPO;**response length 在 SFT 占比 plateau 后不回退 → 模型把长推理"内化进 policy",RL 只精修不抹除**(anti-forgetting 直接证据)。
  - baseline 公平性:同 backbone、同数据(LUFFY 的 Openr1-Math-46k-8192)、同 8192 长度;承认 SFT→GRPO 比 HPT 更耗算力却仍被 HPT 超过(对自己不利的比较)。SRFT 因官方代码未公开是自实现复现(已标注)。
- 假设与失效边界:
  - 【原文】UPGE 推导假设 offline 数据"均匀覆盖整个 state-action rollout 空间"才能令 π_ref=1 的重要性比化简成立——作者自承现实数据严重受限、这一条(及 PPO 的 1/π_θold)"在 RL 视角下都不完全理论正当"(§2.3, 引 Kakade 2003)。
  - 【原文】γ 的最优值依赖 base model 与训练数据特性(Qwen 用 γ=0、LLaMA 用 2/8),非普适常数。
  - 【推断】HPT 需要每 question 既有 supervising trajectory(SFT 监督)又能在线 rollout + verifier——强依赖 **rule-based 可验证任务**(数学);开放域/无 verifier 任务不适用。
  - 【推断】只实例化了二值开关这一最简 HPT;连续 α(P)/β(P) 或更细 token 级混合是否更好未验证。
- 祛魅总结【推断】:
  - 真贡献:① UPGE 是干净、有解释力的统一框架(Table 1 把一堆算法收进四组件 + Eq.1-6 的推导),理论价值高且可复用;② HPT 用"rollout 准确率二值门"这一极简实例就刷出强结果,且给出 Pass@k 探索保持 + length 不回退两个有说服力的机制证据。
  - 包装/可能高估:HPT 本体非常简单(就是按对错切 RL/SFT),"Hybrid Post-Training" 名头大于算法新意;增益部分来自 Dr.GRPO + LUFFY 数据基座本身强。off-policy RL 被证非必需,也说明"统一"框架在实践里被压缩成了 SFT+on-policyRL 二选一。低估:UPGE 作为"分析任意后训练算法"的统一记法,其工具价值可能比 HPT 算法本身更持久。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=per-question on-policy rollout 准确率 P(rule-based 0/1 verifier)|**改什么**=单个 policy 模型参数(混合 RL+SFT 梯度)|**何时改**=每步、每 question 按 P vs γ 二值切换|**免梯度?**=否,标准策略梯度 + SFT NLL|**记忆-技能生命周期**=无显式记忆库;"技能"(长推理 routine)从 offline 数据 SFT 内化进 policy,RL 阶段精修|**防遗忘机制**=① 动态混合避免两阶段遗忘;② 实测 response length 在 SFT plateau 后不回退、且基本不丢已会题(Table 4 green counts 不变)——内化而非覆盖。
- ⑦ 开源代码+框架/harness:https://github.com/TsinghuaC3I/Unify-Post-Training (已 clone,~11MB)。**框架=veRL + LUFFY**(repo `hpt/verl` 主要 build upon LUFFY 的 mix_src,FSDP + vLLM rollout;reward 用 deepscaler 的 math_reward;训练脚本 `hpt/scripts/train`、消融 `scripts/ablations`)。数据用 LUFFY 的 Openr1-Math-46k-8192。
- 💰 资源/成本与可扩展性:8×NVIDIA A800 80GB;lr 5e-6 AdamW;每 question 8 条 rollout(temp 1.0),max gen 8192。明确比 SFT→GRPO 省(无独立 SFT 前阶段、GRPO↔SFT 切换时省掉部分 rollout)。
- 🎯 对"探索-巩固"对标:**强支撑 + 可借框架**。判定:HPT 的 "P>γ 放手 RL 探索 / P≤γ 给 SFT 示范" 正是本项目"探索(走自己能走通的)+ 巩固(走偏/解不出时教师脚手架接管)"的**最直接、最简洁的算法化身**,且 §4.1 的 large-k Pass@k 最高 + §4.3 length 不回退,实证了"探索能力不被巩固抹除"这一 TSRD 核心诉求。可借:① per-question rollout 准确率(Eq.8)当"该探索 vs 该被教"的门控信号,可平移成 TSRD 里"哪些题/哪些 prefix 该让 student 自走、哪些该 teacher 接管";② UPGE 四组件记法(Eq.6)可用来统一描述 TSRD 自身的混合损失。竞品/缺口:HPT 是 **question 级**二值切换、**无 teacher logit 蒸馏、无 token/step 级粒度、无 MTP/前瞻**;TSRD 要的是 step/path 级(切关键步、单点 path-recovery)更细粒度的"选路 + 回轨",这正是 HPT 的留白。
- 🔭 开放问题/未来方向:【原文】UPGE 四组件还能构造更优梯度估计(加权平均多个 noisy 估计 / complementary filter,§2.3);HPT 只是统一后训练的"初步尝试";off-policy RL 的角色待深挖。【推断】把 question 级门控下沉到 step/token 级(结合高熵关键步切分);连续 α(P)/β(P) 取代二值;用 MTP/前瞻预测"该题能否自解"以更早决定 RL/SFT,减少 rollout 开销;非可验证(开放域)任务上如何取 P。

key|读到PDF?|L线|对标结论|残留待核数
hpt_upge | 是(全文22页正文,Eq.1-13全抄准) | L2 | 强支撑+可借框架:per-question rollout准确率P做"探索vs被教"门控=TSRD选路/接管的最简化身;UPGE四组件可统一描述TSRD混合损失。差异:question级、无teacher logit蒸馏、无token/step粒度、无MTP | 0

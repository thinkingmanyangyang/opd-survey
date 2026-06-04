trapo | TRAPO: Trust-Region Adaptive Policy Optimization | 清华大学 CoAI 组 + Ant Group(Mingyu Su, Jian Guan, Yuxian Gu, Minlie Huang, Hongning Wang;通讯 黄民烈/王宏宁) | arXiv 2512.17636 v1(2025-12-19, cs.LG)·README/OpenReview 标注 ICLR 2026 | 主题线 L2(统一 SFT-RL,实例级)+ L4(前缀脚手架)·相关性 **高**

**原始论文**:https://arxiv.org/abs/2512.17636

## 一眼看懂
- 🟦 TL;DR:两阶段 SFT→RL 有根本矛盾——SFT 把模型锁进刻板模仿、抑制 RL 探索且引发遗忘。TRAPO 在**每个训练实例内**交织 SFT 与 RL:只对专家轨迹**前缀**做 SFT、其后由目标策略自行 rollout 补全做 RL。关键稳定器 **Trust-Region SFT(TrSFT)**:把标准 SFT 梯度权重 \(1/p_\theta\) 改成 \(1/\max(p_\theta,\alpha)\)——信赖域内信任标准 SFT(激进模仿),域外用常数 \(1/\alpha\) 压制梯度,把 forward-KL 的 mode-covering 转成 reverse-KL 式 mode-seeking。再用 **micro-group 采样**按累计回报自适应分配前缀长度。5 个数学 benchmark 均值超 SFT/RL/SFT-then-RL。【原文 Abstract, §2】
- 最巧的一步:**TrSFT 的 \(1/\max(p_\theta,\alpha)\) 信赖域裁剪**(Eq 3 + Prop.1)。抽掉它(=朴素地把标准 SFT loss 与 RL loss 直接相加)直接**灾难崩溃**——较纯 RL 低 18+ 分(消融 Table 2:micro-group + 标准 SFT loss = 32.3 vs 纯 RL 50.4)。原因:标准 SFT 的 mode-covering 给专家无支撑的"空洞区"分配概率,在实例级交织时立即产出退化 rollout 毒化探索。**⚠️ 但这个"最巧的一步"在公开代码里没找到实现(见⑦,重大复现存疑)。**

## 为什么做
- 研究背景:o1/R1 等里程碑后,主流后训练是两阶段 SFT→RL(SFT 先教模仿专家、RL 再 trial-and-error 锐化推理)。【原文 §1】
- 解决的具体痛点:两阶段根本不一致——(1) SFT 把模型锁进刻板模仿、抑制 RL 所需探索;(2) SFT 易致灾难遗忘,使 RL 难用预训练知识。更低 SFT loss ≠ 更好 RL 起点,过度 SFT 反把模型推出适合 RL 的区域且无即时信号。论文实证:**朴素地把 SFT loss + RL loss 直接相加→灾难崩溃**(较纯 RL 低 18+ 分)。根因:标准 SFT(min forward-KL)的 mode-covering 给专家无支撑空洞区赋概率→重复/退化解码→在实例级交织时立即毒化探索。【原文 §1, §2.2, §3.3】
- 相关工作 & 各自不足(把"RL for reasoning"与"SFT+RL 结合"两族系统铺开,定位 TRAPO 的理论卖点):
  - **RL for reasoning(纯 RL 一族)**:PPO / GRPO / DAPO / Dr.GRPO / VAPO——靠 verifiable reward 做 trial-and-error,但 Fig.6 显示纯 GRPO 在大 \(k\) 下 pass@k 被 base 反超(RL 只在既有解空间里筛、不扩展底层解空间),且无专家引导时难解 hard prompt。【原文 §1, §3.2】
  - **SFT+RL 直接加权两 loss 一路**:SRFT(按 token 熵调权)、AMFT(meta-gradient 学权)、HPT(按 rollout 二值选 SFT/RL)——都在"如何给两 loss 配比"上做文章;TRAPO 论证**朴素相加会崩**(Table 2 的 32.3),问题不在配比而在 SFT 自身的 mode-covering。
  - **RL 流水线内插 SFT 一路**:ReLIFT(跨 batch 交替 SFT 与 RL)、**LUFFY**(把 1 条专家轨迹混进 7 条 online、用重要性比 \(\mathrm{off\text{-}ratio}\) 校准 off-policy 偏差)——TRAPO 公开代码实际就基于 LUFFY 基座改造(见⑦)。
  - **最相近——Prefix-RFT(Huang 2025)**:采样专家**前缀**引导、用熵选专家 token 做 SFT。TRAPO 自称首个从**理论**研究"SFT+RL 目标结合"挑战并给解者。【原文 §1, §3.1】
  - 动机链:两阶段 SFT→RL 不一致(锁模仿 + 遗忘)→实例级交织(只 SFT 前缀、其后 RL 补全)→但朴素相加崩溃(mode-covering 致空洞区)→TrSFT 信赖域裁剪(转 mode-seeking)+ micro-group 按需给最小前缀→TRAPO。
  - 与最近邻工作的精确Δ:vs LUFFY(off-policy 重要性比混专家轨迹),关键差异=TrSFT 信赖域裁剪(理论上转 reverse-KL)+ 按累计回报自适应前缀长度;vs Prefix-RFT(固定前缀 + 熵选 token),关键差异=micro-group 按 prompt 难度动态分配前缀。【原文 §3.1】**(注:这两项 Δ 正是公开代码中缺失的部分——见⑦。)**

## 怎么做 + 靠不靠谱
- 方法流水线(每 prompt,输入→输出):① 先无引导自探索 rollout(\(L_1=0\),恒从 guidance-free RL 起)→ ② 若前面微组平均回报 < 阈值 \(t_i\),则注入长度比 \(L_i\) 的专家前缀再采 \(n_i\) 个补全(\(L\) 递增,\(L_N=1\) 可给完整专家路径)→ ③ 目标策略补全 → ④ 补全部分用标准 GRPO、专家前缀部分用 **TrSFT loss**,全轨迹联合优化。【原文 §2, Algorithm 1】
- 逐组件必要性(消融 Table 2):
  - **TrSFT**:核心稳定器。同一 micro-group 骨架下,TrSFT(56.6) vs LUFFY loss(53.6) vs 标准 SFT loss(32.3,崩溃)→ 把稳定结合的功劳归给 TrSFT。
  - **micro-group 采样**:单独已超 GRPO(52.7 vs 50.4)。
  - **联合优化(前缀 SFT + 补全 RL)**:整体框架。
  - **〔代码核对发现:TrSFT 与 micro-group 阈值机制在公开代码中均未见实现——消融结论无法从所见代码复现,见⑦〕**
- 关键机制/公式(直觉):
  - **标准 SFT 的爆炸权重(问题根源)**:LUFFY 式前缀 SFT 梯度可写为 \(\nabla_\theta L_{\mathrm{SFT}}\) 中带权重 \(\frac{1}{p^\theta_T(y^i_n\mid x^i,y^i_{<n})}\)(Eq 2 的 token 权重)。当专家 token \(y^i_n\) 落在远离当前策略模式处(如专家分布最右模式),该权重**爆炸**,把 \(p^\theta_T\) 先推进"空洞区"再慢修正(GMM pilot)。实例级交织时,任何分到空洞区的概率质量都立即产出退化 rollout。
  - **TrSFT(Eq 3)**:把权重的分母改为 \(\max(p^\theta_T,\alpha)\),得梯度
    \(\displaystyle \nabla_\theta L^\alpha_{\mathrm{TrSFT}}=-\frac{1}{N}\sum_{i=1}^{N}\sum_{n=1}^{|y^i|}\frac{1}{\max\big(p^\theta_T(y^i_n\mid x^i,y^i_{<n}),\,\alpha\big)}\,\nabla_\theta p^\theta_T(y^i_n\mid x^i,y^i_{<n}),\)
    其中 \(\alpha\in[0,1]\) 是信赖域边界。\(p^\theta_T\ge\alpha\) 用标准 SFT 激进模仿("近"的专家智慧、保留已有强项);\(p^\theta_T<\alpha\) 用常数 \(1/\alpha\) 压制梯度、只追专家主模式,避免大梯度把策略推进空洞区。
  - **Prop.1(最优解,KKT 推导)**:令 \(S(\lambda)=\{c\mid p_E(c)>\alpha\lambda,\,c\in C\}\)(\(C\)=词表),存在唯一 \(\lambda\in(0,1)\) 使 \(\lambda=\sum_{c\in S(\lambda)}p_E(c)\),且最优解为
    \(\displaystyle p^*_T(c)=\begin{cases}\dfrac{p_E(c)}{\lambda}, & \text{若 } p_E(c)>\alpha\lambda,\\[2mm] 0, & \text{否则,}\end{cases}\)
    即**剪掉专家低概率区**(\(p^*_T(c)=0\))、**对主模式重标定**(\(p^*_T(c)=p_E(c)/\lambda\))。直觉:把目标从 forward-KL 的 mode-covering 转向 reverse-KL 的 mode-seeking,逼策略聚焦专家核心技能、利于高回报 rollout。
  - **Micro-group(§2.3)**:每 prompt 顺序建 \(N\) 微组,各由 (前缀长度比 \(L_i\), 回报阈值 \(t_i\), 采样预算 \(n_i\)) 决定。\(0=L_1<L_2<\dots<L_N=1\):\(L_1=0\) 保证恒从无引导自探索起(配 \(t_1=-1\) 必触发无引导),\(L_N=1\) 可给完整专家路径。判定:对 \(g_i\) 先算前面所有微组样本的平均回报,若 < \(t_i\) 则给比例 \(L_i\) 前缀再采 \(n_i\) 补全,否则直接采 \(n_i\) 无引导 rollout。"仅在需要时给最小引导"。
  - 另一直觉(Fig.2 pilot):越长专家前缀稳步提升准确率并激发 backtracking / backward chaining 等高级推理行为。
- 实验与证据:
  - 训练:OpenR1-Math-46k-8192(DeepSeek-R1 生成、已验证的数学推理轨迹),额外为每题配一条 OpenR1-Math-200k 轨迹增多样性。基座 Qwen2.5-Math-7B(主);另在 Qwen2.5-7B-Instruct 验证通用性。pilot 用 Qwen2.5-3B-Instruct + DeepSeek-R1 前缀。【原文 §3.1】
  - 超参:GRPO **without KL penalty**;batch 128、恒定 lr 5e-6;组大小 8 划 4 微组 \(\{4,2,1,1\}\),\(L=(0,0.2,0.5,1.0)\),\(t=(-1,0.5,0.7,0.9)\),**\(\alpha=0.1\)**。【原文 §3.1, lines 409-418】
  - 评测:5 数学=AIME2024、AMC、MATH-500、Minerva、OlympiadBench(AIME/AMC 报 avg@32,其余 pass@1);2 通用=ARC-c、MMLU-Pro(pass@1)。
  - 关键数字(Table 1,Qwen2.5-Math-7B):5 数学均值 **TRAPO 56.6**,较 SFT(50.3) **+6.3**、纯 GRPO(50.4) **+6.2**、SFT-then-RL(54.3) **+2.3**,超 ReLIFT(53.4)/LUFFY(55.5)。通用域均值 68.3 居首。**注意单 benchmark 上 TRAPO 常非最优**(AIME2024 TRAPO 28.3 < Oat-Zero 33.4、SFT-then-RL 33.5),优势在综合均值。消融 Table 2:朴素 SFT loss 崩溃 32.3。Fig.4:TRAPO 全程更高 reward、早期快速增长生成长度、长期稳在较高 policy entropy(保留探索)。Fig.6 pass@k:纯 GRPO 大 \(k\) 下被 base 反超(RL 只筛既有解空间),TRAPO 随 \(k\) 强 scaling(扩展底层解空间)。【原文 §3.2】
  - baseline 公平吗:同基座、同数据,对比 Pure-RL 家族(GRPO/PRIME-Zero/SimpleRL-Zero/ORZ/Oat-Zero)+ External-Guidance 家族(SFT/SFT-then-RL/LUFFY/ReLIFT)——较公平。
  - "看着强但没回答核心":单 benchmark 上常非最优(综合均值才赢),"strong new paradigm"措辞需结合此细节看;且全部训练数据来自 R1 蒸馏轨迹,多样性受限。
- 假设与失效边界:
  - 【原文】仅在数学推理(+少量通用 QA)验证;训练数据全来自 DeepSeek-R1 蒸馏轨迹,多样性受限。
  - 【原文】\(\alpha\)、各微组 (\(L_i,t_i,n_i\)) 为手调超参。
  - 【推断/代码核对】正文 GRPO "without KL penalty",但仓库默认 `use_kl_loss:True, kl_loss_coef:0.001, low_var_kl`——**需脚本覆盖才与论文一致**(已核 `config/mix_ppo_trainer.yaml`)。
  - 【推断】TrSFT 的 \(\alpha\) 与信赖域机制在专家分布与目标策略差异极大时(如跨域)是否仍稳,未验证。
- 祛魅总结【推断】:**真贡献(在论文层面)**=(a) 把"朴素 SFT+RL 相加会崩"实证清楚 + 归因到 mode-covering;(b) TrSFT 的理论(Prop.1 forward→reverse KL 转变)与 micro-group 的"按需最小引导"设计。**但最大问题在代码层面**:两项签名贡献(TrSFT 信赖域裁剪、micro-group 回报阈值调度)在公开仓库中均**未见实现**——公开代码实际跑的是 LUFFY 式 off-policy GRPO + 固定/step-based 前缀。因此论文结论的可复现性存疑:**包装上"theoretically grounded trust-region"很重,但开源代码兑现不了**。高估:把它当"已开源可复现的新范式"——签名方法缺失。低估:TrSFT 的理论分析(mode-covering→空洞区→信赖域裁剪→reverse-KL)本身对"为何 SFT+RL 直接相加会崩"有真实解释力,即便代码没兑现。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=专家前缀(TrSFT loss,信赖域裁剪)+ 自补全(GRPO advantage) | **改什么**=策略参数,前缀段与补全段联合优化 | **何时改**=每个训练实例(per-prompt micro-group),按累计回报动态注入前缀 | **免梯度?**=否,前缀走 TrSFT 梯度、补全走 GRPO 梯度 | **记忆-技能生命周期**=不涉及外部记忆/技能库(纯参数) | **防遗忘机制**=显式动机——实例级交织避免两阶段 SFT 的灾难遗忘 + TrSFT 信赖域保留预训练强项(论文主张;Fig.4 较高稳态 entropy + 通用域不退化 68.3 佐证)
- ⑦ 开源代码+框架/harness:https://github.com/Su-my/TRAPO(已克隆 6.0MB,README 标注 ICLR 2026)。**框架=基于 LUFFY 改造的 verl GRPO**;`luffy/verl/verl/trapo_src/{mix_actor.py, mix_core_alg.py, mix_vllm_rollout.py, mix_trainer.py, config/mix_ppo_trainer.yaml}`;脚本 `exp_scripts/{train_on_policy.sh, train_trapo.sh}`。
  - **⚠️ 重大代码-论文不一致(已逐文件独立核实,非沿用 v1)**:
    1. **TrSFT 的 \(1/\max(p_\theta,\alpha)\) 信赖域裁剪未在任何 .py 中实现**——repo-wide grep "trust/trsft" 仅命中 README.md 散文,代码无对应。前缀 SFT loss 由 `mix_core_alg.py::compute_sft_pure_loss` 实现,即 `sft_losses=-log_prob`(标准 NLL/forward-KL SFT),再以 `sft_loss_coef` 加权与 GRPO 相加(`mix_actor.py` L118-148 `use_sft_multitask_loss` 分支)。
    2. **micro-group 的回报阈值 \(t_i\) 调度未实现**——无 `return_threshold`/`micro_group` 阈值逻辑;前缀由 `mix_vllm_rollout.py` 的 `prefix_strategy`(random/linear/linear_max/reverse_linear/fix,**step-based 窗口**)控制,非按累计回报。
    3. **`train_trapo.sh` 实际用 `use_off_policy_loss=True`**(LUFFY 式 `off_ratio=exp(log_prob)/clamp(exp(log_prob.detach()),min=0.1)`)+ 固定 `min/max_prefix_ratio=1.0`,既非 TrSFT 路径、也非 SFT-multitask 路径。
    4. KL 配置默认 `use_kl_loss:True/kl_loss_coef:0.001/low_var_kl`,与正文"without KL penalty"冲突,需脚本覆盖。
  - **结论:公开代码=LUFFY 基座(off-policy GRPO + 固定/step-based 前缀),TRAPO 的两项签名新增在所见已提交代码中缺失。复现需谨慎;签名方法可能在未公开分支或私有脚本中。** 代码链接见上供手动核查。
- 💰 资源/成本与可扩展性:Qwen2.5-Math-7B 量级;batch 128、组大小 8;具体卡数原文未在正文详列(Appendix C.2)。【原文 §3.1】
- 🎯 对"探索-巩固"对标:**强支撑(论文 idea 层面)+ 对照基线,但代码不可直接复用**。一句判定:TRAPO 的"只对专家前缀做 SFT 脚手架 + 其后学生 on-policy 自探索补全 + 按难度给最小前缀"正是本项目"teacher 稀疏脚手架 + path-selection/recovery"在数学推理上的实例级实现——**micro-group 的"无引导自探索→失败则递增专家前缀"对应 path-recovery 的渐进接管**,TrSFT 的"只追专家主模式、剪掉空洞区"对应"巩固时固化进参数而不被无支撑模式污染"。可借组件(理论):① TrSFT 的 \(1/\max(p_\theta,\alpha)\) 信赖域裁剪思路可移植为"巩固阶段稳定固化"的梯度处理;② "按累计回报自适应前缀长度"可作 path-recovery 的接管粒度调度。**Δ/缺口**:(a) 前缀是 teacher **固定**轨迹的前缀,非学生**自选**恢复分支(本项目要 on-policy 自选);(b) 切点是"前缀长度比"(粗,按 token 数),非 TIP 式关键步定位;(c) 无 MTP 前瞻;(d) **最关键——签名方法未开源,无法直接拿代码做 baseline**,只能按论文重新实现。【推断,依据 §2 机制与本项目 idea 的对应 + ⑦ 代码核实】
- 🔭 开放问题/未来方向:【原文】未明确列 future work 专节(结论强调"新范式")。【推断】① 把 teacher 固定前缀换成学生 on-policy 自选恢复分支(本项目方向);② 把"前缀长度比"切点细化为 TIP 式关键步/Q3 点定位;③ 引入 MTP 前瞻决定"何时注入前缀";④ **补齐并开源 TrSFT + micro-group 的真实实现**(当前最大复现障碍);⑤ 验证 \(\alpha\) 信赖域在跨域/非数学任务的稳定性。

〔本篇与既有 analysis/trapo.md 核对:核心结论(签名方法 TrSFT/micro-group 在公开代码缺失)保持不变,逐文件核实结论沿用。本次增强:4 条核心公式(标准 SFT 爆炸权重、TrSFT 梯度 Eq.3、Prop.1 最优解分段式、micro-group 阈值规则)转 MathJax 并从 PDF 抄准;相关工作铺开纯 RL 族(PPO/GRPO/DAPO/Dr.GRPO/VAPO)+ SFT+RL 三条结合路线(加权 SRFT/AMFT/HPT、内插 ReLIFT/LUFFY、Prefix-RFT)各自定位与精确Δ;方法机制补"标准 SFT 权重爆炸→空洞区→TrSFT 裁剪→Prop.1 mode-seeking"完整推导链。**未编造任何代码细节**;代码不一致四点全部保留。无新增待核项。〕

asft | ASFT: Anchored Supervised Fine-Tuning | 南方科技大学 / 北京大学 / 上海AI Lab（He Zhu, Junyou Su, Peng Lai 共一，通讯 Guanhua Chen） | 2026-02-01 arXiv v3 · ICLR 2026 | L2 统一SFT-RL·奖励微调(GFT类) · 相关性高

**原始论文**：https://arxiv.org/abs/2509.23753

## 一眼看懂
- 🟦 TL;DR：DFT(用 `sg[π_θ]` 给交叉熵重加权的 SFT 变体)在推理域好用、在知识域(如医疗)**不稳**。ASFT 用 **reward-weighted regression(RWR) 框架**证明 DFT 等价于一个特定 auxiliary 分布选择、给出比 SFT **可证更紧**的 RL 下界，但**缺分布锚定→渐进漂移**(KL 持续增大)才是其不稳根因；于是只加一项**轻量 forward-KL 锚定**(把策略约束在 base 附近)就同时保住紧界优势和稳定性(§4，Eq.6)。数学 100k 较 base +17.89(142%)、医疗 10k 较 SFT +8.28(24.8%)，仅需全 RL 约 3% 算力。
- 最巧的一步：**forward-KL 锚定项 `λ·D_KL(π_base‖π_θ)`**（Eq.6，base‖policy 方向）。抽掉它就退回 DFT——§5.2 Table 1 知识域 DFT 在 10k 反而较 base **掉 2.19 点**(MMLU 30.52→26.69)、ASFT 加锚定后反而 +10.65(MMLU→46.37)。且方向必须 forward(mode-covering)：§5.3 消融 forward > reverse，reverse(mode-seeking)会致坍塌。

## 为什么做
- 研究背景：后训练在 SFT(off-policy 模仿,高效但易记忆、泛化弱)与 RL(on-policy,泛化好但贵且不稳)间存在根本权衡(§1)。一批工作从 RL 视角重审 SFT——DFT(Wu et al. 2025a)指出标准 SFT 的隐式 reward `r_SFT=I[y=y*]/π_θ` 在 π_θ→0 时**方差无界**,用 `L_DFT=−sg[π_θ]·log π_θ`(Eq.3)概率重加权修复,推理域显著提升。
- 解决的具体痛点：①**DFT 效果域相关**——推理域好,知识密集任务(医疗 MedMCQA/MMLU)**不稳**,且启发式重加权**缺理论依据**(§1)。②作者用 RWR 诊断根因:DFT 对应特定 auxiliary 分布(Eq.4),给可证更紧下界(Theorem 1),但**缺分布锚定**→渐进漂移(q 越来越集中在高 π_θ 轨迹,形成正反馈)→下界越来越松、重要性权重方差越来越大→失稳(§4.1 Key Finding 3)。
- 相关工作 & 各自不足：SFT(稳但松的 RL 下界,Proposition 1,易记忆);RL(紧但贵不稳);importance-weighted SFT(Qin&Springenberg 2025)、proximal SFT(Zhu et al. 2025)、DFT——ASFT 的Δ:**把 auxiliary 分布本身锚定到 base**,在保留紧界的同时控方差。
- 动机链：现状(DFT 紧但漂移)→缺陷(知识域 KL 漂移失稳)→所以加轻量 KL 锚定(就是把 RL 的 KL 惩罚搬进 SFT)。为什么不用更简单的 SFT+KL？§5.2/Table 1 显示 SFT+KL 几乎不涨甚至掉点(它没有 DFT 的紧界重加权)——必须 **DFT 重加权(管紧) + KL 锚定(管稳)** 二者皆有。
- 与最近邻工作的Δ：相对 DFT(最近邻),差在**加 forward-KL 锚定 + RWR 理论解释**;相对 SFT+KL,差在**保留 DFT 的概率重加权紧界**(SFT+KL 无紧界);相对 AMFT 等动态 µ 方法,ASFT 是**静态轻量正则**而非 meta 学习配比。

## 怎么做 + 靠不靠谱
- 方法流水线（§4）：① RWR 框架下证 SFT 是 RL 下界(Prop.1)、auxiliary 分布 q 决定下界紧度与稳定性(Eq.2) → ② 证 DFT = 特定 q(Eq.4)、给严格更紧下界(Theorem 1,当 `Var(π_θ)>0` on D+) → ③ 诊断 DFT 漂移(不等式 `u≥1+log u` 仅 u=π_θ/q=1 即 π_θ 在 D+ 上恒定时取等,训练中 π_θ 越来越非均匀→下界越来越松) → ④ **ASFT = L_DFT + λ·E[D_KL(π_base‖π_θ)]**(Eq.6),token 级实现(序列权重按位置归一分摊)。
- 逐组件必要性：
  - **DFT 概率重加权(管紧)**：提供比 SFT 更紧的下界(Theorem 1),推理域增益主来源。
  - **forward-KL 锚定(管稳)**：§5.2 知识域必需(无它 DFT 掉点);§5.3 方向必须 forward(mode-covering 防坍塌、维持 base 广分布),reverse 会 mode-seeking 坍塌。理论上 KL 项不改下界结构(保紧)只控方差(防指数增长)。
  - **λ**：本 ICLR 版默认 **0.05**(§5.1,数学/医疗同);代码同时实现 sft/dft/sft+kl/asft 四 mode 便于消融。【版本差异:旧 train_v2 代码默认 0.1、bf16 推荐 0.03;本 PDF 统一用 0.05】
  - 消融(§5.3)还含 lr/batch 鲁棒性分析。
- 关键机制/公式（直觉）：DFT 重加权 `sg[π_θ]·(−log π_θ)`——模型对正确轨迹当前概率越高,该样本权重越大→把学习集中到"已有点会、再推一把就稳"的样本。KL 锚定 `D_KL(π_base‖π_θ)`——把策略拉回 base 的 trust region,阻止 q 退化到只盯少数高概率样本(防 effective sample size 萎缩)。紧界 + 锚定 = 既贴近最优又不跑偏(正是 RL 里 KL 惩罚的作用)。
- 实验与证据：基模 **LLaMA-2-7B**(医疗,刻意选它避数据污染) + **Qwen2.5-7B**(数学+医疗)。数据:数学 **NuminaMath CoT**(10k/30k/100k)、医疗 **MedMCQA**(10k/30k/100k);测 Math500/Minerva/Olympiad/AIME24/AMC23 与 MedQA/MMLU/MedMCQA。关键数字(Table 1):数学 100k ASFT 较 base **+17.89**(vs DFT +13.43),AMC23 36.72 vs DFT 27.19;医疗 10k ASFT 较 base +10.65(DFT 反 −2.19),且跨 10k/30k/100k 稳定(+10.65/+10.63/+8.60);效率仅全 RL ~3% 算力。Fig 1 直接展示:DFT 的 KL 持续飙升(漂移),ASFT KL 平稳且 ID/OOD 双高。baseline=SFT/SFT+KL/DFT,公平。证据链完整:理论(更紧界+漂移诊断)→Fig 1(DFT KL 飙升)→Table 1(医疗 DFT 失稳而 ASFT 稳)直接对应理论预测。
- 假设与失效边界：【原文】§3.2 假设 sparse reward `R=I[y=y*]` 且 `supp(π_θ)⊆supp(π_ref)`;Theorem 1 需 `Var(π_θ)>0` on D+(故推理域因变量大而更受益)。【推断】仅 2 个 7B 基模(LLaMA-2/Qwen2.5),规模/家族覆盖有限;λ 跨域/精度需手调(本版 0.05、旧版 0.1/0.03),"轻量锚定"并非完全免调;**摘要提 code generation 但 Table 1 只报数学+医疗**,代码域结果在本版正文未见(待核附录)。
- 祛魅总结：【推断】真贡献是 **RWR 统一框架** + "DFT 紧但缺锚定→漂移"的清晰诊断 + 一行 KL 修复,理论与实验(Fig 1 KL 飙升 vs 平稳)咬合扎实。**包装/局限**:ASFT 本质是"DFT + 已知的 KL 正则",机制新颖性更多在**理论解释**而非机制本身(SFT+KL 早存在,论文也列为基线);λ 敏感性说明"轻量"有限;仅 7B 两基模。作者**诚实**地把 SFT+KL 列为基线并证明其不足(凸显紧界+锚定缺一不可),未明显高估。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=专家示范的概率重加权交叉熵(`sg[π_θ]·log π_θ`,token 级)+ 对 base 的 KL 锚定 | **改什么**=策略参数(SFT 损失重塑) | **何时改**=离线(off-policy SFT,固定数据集,无在线采样) | **免梯度?**=否(梯度 SFT) | **记忆-技能生命周期**=不适用 | **防遗忘机制**=核心——`D_KL(π_base‖π_θ)` forward-KL(mode-covering)把策略锚在 base 的 trust region,显式防分布漂移/坍塌、维持 base 广分布(§5.3 理论 + Fig 1 经验)。
- ⑦ 开源代码+框架/harness：https://github.com/zhuchichi56/ASFT （v1 已克隆约 54MB,**完整**;数据 HF chichi56/ASFT）。框架=自定义 HF Trainer + DeepSpeed(ZeRO-2/3) + PEFT/LoRA;新增 **veRL(FSDP2) 分支**(数值更稳,README 推荐优先);**已合入 LLaMA-Factory 主分支**。核心 loss `train_v2.py`(mode∈{sft,dft,sft+kl,asft})。【注:本 ICLR 版 λ=0.05;旧代码默认 0.1、bf16 推荐 0.03,存在版本差异】
- 💰 资源/成本与可扩展性：【原文】§5.1——数学 max len 2048/batch 256/lr 5e-5/1 epoch,医疗 max len 512/batch 64/lr 2e-5/3 epoch,均 AdamW cosine warmup 0.1;ASFT 仅比 SFT 多一个 KL 惩罚项,§1 称只需全 RL **~3%** 算力。可扩展性:SFT 级效率 + RL 级泛化。
- 🎯 对"探索-巩固"对标：**巩固/防遗忘侧的强相关者(竞品+可借组件)**——ASFT 的 "DFT 重加权(把学习集中到学生快会的样本)+ KL 锚定(不跑偏 base)" 正对应 TSRD"**巩固=把成功经验固化进参数且不遗忘**":重加权≈"固化学生够得着的成功路径",forward-KL 锚定≈"不遗忘"。**可借组件**:① RWR 框架可作"SFT/重加权/RL 谁更紧"的统一分析语言;② forward-KL(mode-covering)防漂移机制可直接用于 OPD/TSRD 巩固阶段的防遗忘;③ `sg[π_θ]` 重加权(forward-hard/backward-soft 同族的 stop-gradient 用法)与本项目记忆里的 reusable technique 一致。**缺口**:纯 off-policy SFT,无探索/选路、无 path-recovery、无 on-policy 自选、无 teacher 脚手架。一句判定：**强相关于"巩固/防遗忘"这一维,KL 锚定与重加权可直接借入 OPD 巩固阶段;但缺探索侧机制,只覆盖'探索-巩固'的后半**。依据:方法无任何在线探索或路径选择,纯重加权 SFT + 静态锚定。
- 🔭 开放问题/未来方向：【原文】§5.3 后续——KL 方向/超参的更细分析;ASFT 作 DAPO/GRPO 更优初始化(v1 提及)。【推断】把静态 λ 升级为动态(逼近 AMFT 的可学习配比);RWR 框架扩到 on-policy/带探索的设置;代码域结果补全;更大规模/家族验证;与 on-policy 蒸馏结合作"巩固"模块。

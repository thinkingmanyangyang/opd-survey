psft | Proximal Supervised Fine-Tuning (PSFT) | 上海交大 + 上海创智学院 + 腾讯大模型部 + 澳门大学(Wenhong Zhu, Ruobing Xie, Pengfei Liu 等;SFT-RL 统一/改进版 SFT 谱系,与 iw-SFT/DFT/Prefix-RFT 同族) | ICLR 2026·arXiv·v2(2026-04-12) | 主题线 L2(统一 SFT-RL 视角下的"改进版 SFT")·相关性 高

**原始论文**:https://arxiv.org/abs/2508.17784

## 一眼看懂
- 🟦 TL;DR:普通 SFT 是 behavior cloning,泛化差、还会**熵坍缩**(约 150 步后熵骤降),把后续 RL 的探索空间压没。PSFT 把 SFT 重新看成"**advantage 恒为正(A=1)、采样自固定离线数据集**的策略梯度特例",于是可以借 TRPO/PPO 的信任域思想给它套上 PPO 式 clipped surrogate(用 r_t=π_θ/π_θold 的重要性比 + 非对称 clip),把每步更新限制在策略附近。代价是 in-domain 略低于普通 SFT,换来明显更强的 OOD 泛化、不熵坍缩、以及作为后续 RL/DPO 起点更优。【原文 Abstract 行 10-23,§3 行 147-200】
- 最巧的一步:**把 advantage 设为常数 A=1,再套 PPO 的非对称 clip**。这一步等价于"给每个 demo token 的最大似然推力封顶"——梯度分析(Eq.7)显示:当离线数据分布偏离模型分布很多(r_t>1+ε)时,这些 token **梯度直接置零**,从而不被"硬覆写"。抽掉 clip 就退回普通 SFT(熵坍缩 + 泛化垮);抽掉"A=1"它就不再是 SFT 而变成需要真 advantage 的 RL。两者缺一不可——clip 是机制,A=1 是把 SFT 纳入 PG 框架的桥。【原文 §2.1 行 106-108,§3 Eq.6-7 行 156-196】

## 为什么做
- 研究背景:RL(PPO/GRPO)在推理任务上产出高质量长 CoT 轨迹;社区大量用 SFT(一种蒸馏)把这些轨迹的推理能力注入模型,因其比 RL 简单高效。【原文 §1 行 26-34】
- 解决的具体痛点:① SFT=behavior cloning,数据次优或与预训练分布失配时会引发**过大的策略更新**、损害原有能力(泛化差);② SFT 易致**熵坍缩**,削弱探索、约束后续 RL(SFT 常作 RL 冷启动,但过度依赖会把探索能力磨没)。核心挑战:同时改善 SFT 模型的泛化与探索。【原文 §1 行 35-47】
- 相关工作 & 各自不足:**iw-SFT**(Qin & Springenberg 2025)——把标准 SFT 重释为 RL 目标的**松下界**,用重要性重加权收紧下界(偏好轨迹更高权);**DFT**(Wu et al. 2025)——把 SFT 看成**有缺陷的策略梯度**,用基于概率的重加权改善目标性能。二者都聚焦"提升 target 性能/收紧界",而 **PSFT 聚焦"防熵坍缩 + 保泛化 + 给后续 RL 留空间"**(行 1037-1039)。另有 SFT-KL(在 loss 加 KL 约束,系数 0.5)作直接对照基线。【原文 §6 行 1025-1039,§4.1 行 213-216】
- 动机链:SFT 伤泛化/垮探索 ∵ 对每个 demo token 施加**无界**最大似然推力 → 借 TRPO/PPO 信任域把这股推力用重要性比裁剪封顶 → 既保留模仿、又把更新限制在策略附近 → 不引入显式 KL/reference 也能保熵 + 保原有能力 + 给 RL 留空间。【原文 §1 行 47-51,§3 行 166-168】
- 与最近邻工作的Δ:vs 普通 SFT——加 clip 信任域(非对称 ε),A=1;vs SFT-KL——**靠 trust-region(裁剪)而非显式 KL/reference 锚定**(KL 把策略拴在初始 ref 上、限制可从离线数据学到的知识;裁剪只在 token 级封顶,π_θold 还能动态演化);vs iw-SFT/DFT——它们重加权所有 token 以收紧/修正 PG,PSFT 用 clip **置零**远偏 token 的梯度且强调"留 RL 空间/防熵坍缩"。关键有用点:Eq.7 的梯度门控(r>1+ε 置零)直接对应"不被次优 demo 硬覆写",这是泛化与防熵坍缩的来源。

## 怎么做 + 靠不靠谱
- 方法流水线(§3):① 把 SFT 改写为带重要性比裁剪的代理目标(Eq.6):L_PSFT=E_{(s,a)∼D}[min(r_t, clip(r_t, 1−ε, 1+ε))],advantage 恒正(A=1)、(s,a) 采自离线数据集 D;② **非对称 clip**:ε=0.2 或 0.28(实验主用 clip_ratio_high=0.28,与 DAPO clip-higher 同思路鼓励探索),use_kl=False、kl_coef=0(靠 trust-region 非显式 KL);③ **π_θold 动态更新**(每 4/8/16 步刷新一次,fixed 不更新会限制可学知识);④ 可选 **warm-up SFT** 先把 π_θold 对齐到 D 的分布(改善 in-domain);⑤ 效果:训练全程不熵坍缩、保持生成多样性。
- 逐组件必要性:
  - **clip 信任域(vs 普通 SFT / SFT-KL)**:有对照(Table 1 + Figure 1 熵曲线)。普通 SFT/SFT-KL 每 epoch 后熵骤降(过拟合),PSFT 熵曲线平滑。【行 224-230】
  - **A=1(SFT 纳入 PG 的桥)**:理论推导(§2.1 行 106-108),代码内联实现(adv=ones)——非实验消融,但与"SFT=A≡1 的 PG"一致,且 reproducibility 声明明说"给序列所有 advantage 赋常数即可"(行 1068-1069)。
  - **π_θold 动态更新频率**:有消融(Table 6,§5.3)。**no-upd(fixed ref)= 差**(AIME24 13.02,仅略超 base);**upd.16=21.56、upd.8=19.38、upd.4=22.50**;vs SFT=22.08。结论:动态更新是必要的;更频繁(每 4 步)in-domain 增益最大,但小 batch+高 lr 提 in-domain 的同时**因高波动牺牲泛化**——故主实验取 upd.8(in-domain 与 OOD 的折中)。【行 1005-1020,Table 6 行 969-1004】
  - **warm-up**:有对照(Finding 3,Table 1)。warm-up 让 in-domain 稳步提升、可反超普通 SFT(Qwen in-domain 均值 warm-up 48.17 > SFT 47.99 > PSFT 46.98);更长 warm-up 更高 in-domain。【行 426-429】
  - **非对称 clip(high>low)**:取 0.28,与 DAPO clip-higher 同思路(行 198-199)——但 high vs 对称的单独消融未单列〔推断:有 clip 值消融 Figure 8,但 high/low 非对称性本身未隔离对照〕。
- 关键机制/公式(直觉):SFT=A≡1、采样自固定 D 的 PG 特例(§2.1)。PSFT 把 PPO 的 soft trust region(clip r_t 在 1±ε)搬到监督设定——clip 不是为修正 off-policy 分布偏移(SFT 没有 advantage 估计问题),而是当**正则器限制 token 概率的剧烈变化 + 重加权梯度**(行 169-173 明确"PSFT 不是 PG-RL,clip 主要作正则")。Eq.7 梯度门控直觉:模型已同意的 token 推力小;模型强烈不同意(远偏)的 token 推力被置零,避免"次优 demo 把原能力硬覆写"——这正是保泛化 + 防熵坍缩的来源。
- 实验与证据:
  - 数据集/设置:SFT 阶段训练用 **OpenR1-Math-8192** long-CoT(§4.1.1 明示);RL 阶段用 **DAPO-MATH-17k**(DAPO clip-higher 0.28,称"GPPO 的稳定变体")。基座 **Qwen2.5-7B-Instruct、Llama3.1-8B-Instruct**。in-domain=AIME-24/25、AMC(avg@32)、MATH-500、Olympiad、Minerva(avg@8);OOD=GPQA、ARC-C、TruthfulQA、IFEval(avg@8)、MMLU-Pro、SuperGPQA、HeadQA(pass@1)。推理长度 10,240(IFEval 4,096),top-p 0.95,温度 0.7。另覆盖 human-value alignment(Qwen3-4B-Base + UltraFeedback,后接 DPO)与多模态(Qwen2.5-VL,后接 GRPO)。【§4.1-4.4 行 209-216,440】
  - 关键数字(Table 1,Qwen2.5-7B-Instruct,均已核对行号):**in-domain PSFT 略低于 SFT**——AIME-24 SFT 22.08 / PSFT 19.38 / warm-up 22.92(行 257-261);in-domain 6 项均值 SFT 47.99 / PSFT 46.98 / warm-up 48.17(行 323-327)。**OOD PSFT 明显更优**——OOD 7 项均值 SFT 57.90 / PSFT 61.26(行 412-416);**IFEval SFT 54.42 / PSFT 73.03**(行 401-406,SFT 严重损伤指令遵循,PSFT 基本保住 base 的 73.94);TruthfulQA 63.14→67.16。Llama3.1-8B 同向:OOD 均值 SFT 50.49 / PSFT 59.25(行 417-421)。**作为 RL 起点更优(Table 2)**:PSFT→GRPO 在 in-domain 与 OOD 均超 SFT→GRPO。普适性:PSFT→DPO 降 alignment tax(行 656)、PSFT→GRPO 多模态保泛化而 SFT 退化。
  - baseline 公平性:**强且诚实**——SFT、SFT-KL(KL 系数 0.5)、PSFT、PSFT-warm-up 同基座同数据;**论文如实报告 in-domain 略逊普通 SFT**,不做选择性汇报(这与"凭空领先"相反)。
  - 看着强但没回答的:in-domain 略降是真实代价(追求纯 in-domain 峰值者不利);"留 RL 空间"的优势(Table 2)虽明显,但 RL 阶段用 DAPO+DAPO-MATH-17k,与 SFT 阶段数据不同,起点-终点的归因略复杂。
- 假设与失效边界:【原文】① 离线数据所有 token 视为"correct"(A>0,简化 A=1)——若数据含错误/噪声 demo,A=1 的假设不再合理;② ε=0.2/0.28,更大 ε→大梯度(行 199);③ π_θold 需动态更新(fixed 差,§5.3);④ warm-up 是可选项,改善 in-domain。【推断】⑤ in-domain 拟合略逊普通 SFT,不适合"只要 in-domain 峰值"场景;⑥ 高频更新(每 4 步)提 in-domain 但伤泛化(高波动),最优频率任务相关、论文经验取 8;⑦ "信任域=保泛化"的因果在数学/对齐/多模态三域验证,但更工业级/更大模型未试(Conclusion 自陈)。
- 祛魅总结:【推断】真贡献=① 一个干净的理论联系(SFT=A≡1 的 PG)+ ② 据此把 PPO clip 搬进 SFT 这一**极简改动**(reproducibility 声明:主流 RL 框架"rollout 换成 demo + advantage 赋常数"即可),并**诚实报告 in-domain 略逊、OOD 全面提升 + 不熵坍缩 + RL 起点更优**的完整权衡。被高估处:与 iw-SFT/DFT 同属"SFT-as-PG 重加权/约束"一族,**理论新意有限**(A=1 视角 DFT/Prefix-RFT 都用过),PSFT 的差异主要在"用 clip 置零远偏 token 而非重加权"+"强调防熵坍缩/留 RL 空间"。被低估处:IFEval 上 54.42→73.03 这种"防 SFT 把指令遵循能力打没"的实用价值很大,且 Table 2 的"RL 起点更优"对 SFT→RL 流水线有直接工程意义。

## 结构化抽取
- 🎯 机制速览6轴:
  - **学什么信号**:离线 demonstration 的 ground-truth token(交叉熵/最大似然),但 advantage 恒=1、且 token 级梯度被信任域裁剪门控;无 teacher logits、无奖励。
  - **改什么**:policy 参数 θ(裁剪后的最大似然梯度,Eq.7);π_θold 动态滚动更新(copy 快照,非梯度)。
  - **何时改**:离线 SFT 阶段(非在线 RL);π_θold 每 4/8/16 步刷新;可选 warm-up 先对齐 π_θold。
  - **免梯度?**:否,θ 梯度更新(但远偏 token r>1+ε 梯度置零)。
  - **记忆-技能生命周期**:无外部记忆/技能库;离线数据 D 是静态知识源;"技能"通过受约束的模仿固化进参数,刻意**留出后续 RL 优化空间**(不把熵学没)。
  - **防遗忘机制**:**核心卖点即"防 SFT 遗忘原有能力"**——靠 trust-region 裁剪(Eq.7 置零远偏 token 梯度)限制策略漂移、防熵坍缩,从而保住预训练/通用能力(OOD 全面提升、IFEval 不被打垮即证据)。非跨任务持续学习式防遗忘,而是"单次 SFT 不灾难性覆写原分布"。
- ⑦ 开源代码+框架/harness:仓库 https://github.com/zwhong714/PSFT(本地已克隆 **23MB**,完整)。框架=**veRL**,实现在 `verl/recipe/psft/`(main_psft.py、psft_ray_trainer.py、config/psft_trainer.yaml、run_psft.sh / run_sft.sh);adv_estimator=psft 内联于 ray_trainer.py(adv=ones_like·mask)。模型权重在 HF(`wh-zhu/psft-*`)。reproducibility 声明:主流 RL 框架最小改动即可(rollout 换 demo + advantage 赋常数)。【已核本地仓 verl/recipe/psft 文件树 + du 23M;v1 已逐行核 adv 内联分支】
- 💰 资源/成本与可扩展性:SFT 阶段成本与普通 SFT 同量级(只是 loss 加裁剪 + 维护 π_θold 快照);batch 256、mini-batch 16/32(对应 upd.16/8)、lr 1e-6。绝对 GPU 卡时主文未给(见 Appendix C)〔原文未在主文给绝对算力〕。可扩展性:数学/对齐/多模态三域 + Qwen/Llama 双族验证,工业级/更大模型留作 future work。
- 🎯 对"探索-巩固"对标:**强支撑"巩固"一侧 + 保探索的前置条件**。PSFT 解决的正是 idea 里"巩固=固化进参数**且不遗忘**"的核心痛点——它给出"如何在 SFT 阶段把(教师)demonstration 固化进 student 而不灾难性覆写原能力/不熵坍缩"的极简答案,这对 TSRD 的"巩固/回轨固化"阶段直接可用。与"探索"的关系是**前置使能**:不熵坍缩=给后续 on-policy 探索留空间(Table 2 RL 起点更优)。可借组件:① **A=1 + PPO clip** 这一"受约束模仿"目标可直接替换 TSRD 巩固阶段的普通 SFT,防止教师脚手架把 student 原有可走通路径覆写掉;② Eq.7 的**梯度门控**(远偏 token 置零)与本项目"forward-hard/backward-soft 解耦"哲学相通(都用"有界/受限梯度"避免破坏性更新),可类比"巩固时只固化与现有策略不冲突的部分";③ "防熵坍缩=保探索空间"的指标可作 TSRD 巩固阶段的健康度监控。缺口:PSFT 是**纯离线模仿**,无 on-policy 自采、无 teacher 软标签(只有 ground-truth token)、无 path-selection/recovery、无前瞻——它只管"模仿得温和不伤身",不管"探索/选路"。一句判定:**"巩固不遗忘"这一半的强可借基线**(受约束模仿 + 梯度门控 + 防熵坍缩保探索空间),与本项目 forward-hard/backward-soft 哲学同构,但不覆盖"探索/选路/前瞻"另一半。
- 🔭 开放问题/未来方向:【原文】在更多样、工业级数据集与模型上验证 PSFT 的普适性(Conclusion 行 1062-1063)。【推断】① 把 A=1 放松为"教师/前瞻给的软 advantage",使巩固阶段也能区分 demo token 的重要性(向 on-policy 蒸馏过渡);② 把"远偏 token 置零"与"高熵/低置信关键步切分"结合,做选择性巩固(只固化关键步);③ 噪声/含错 demo 下 A=1 假设失效时的鲁棒变体;④ π_θold 更新频率的自适应(in-domain vs OOD 权衡)而非固定 8;⑤ 与 MTP 前瞻信号结合,把"信任域裁剪"扩展到"前瞻一致的 token 才放行"。

— RETURN —
psft | 读到PDF? 是(23页/66k字,§1-§7+§5.3消融+Appendix声明全核,Table 1/2/6 数字逐项核对) | L线 L2(SFT-RL统一/改进版SFT) | 对标结论:"巩固不遗忘"这一半的强可借基线——A=1+PPO clip受约束模仿+Eq.7梯度门控(远偏token置零)与本项目forward-hard/backward-soft哲学同构,防熵坍缩=保探索空间可作巩固健康度指标;但纯离线模仿、无on-policy/软标签/选路/前瞻,不覆盖"探索"另一半 | 残留待核数:0(v1"取update.8"已补正:Table 6 中 upd.4 in-domain 最高22.50,主实验取upd.8是in-domain/OOD折中;绝对GPU卡时主文未给见Appendix)

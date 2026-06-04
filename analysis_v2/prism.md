prism | Beyond SFT-to-RL: Pre-alignment via Black-Box On-Policy Distillation for Multimodal RL (PRISM) | HKUST(广州)+清华+南洋理工+人大+中科大+国科大(Sudong Wang, Weiquan Huang 等共一;Chengwei Qin/Yunjian Zhang 通讯;多模态 RLVR + OPD 谱系,承接 GKD/对抗式 logit-free 蒸馏/VOLD) | 2026-05 预印本·arXiv·v2(arXiv:2604.28123v2,2026-05-01,cs.CV) | 主题线 L1(OPD 新用法)+L3(多模态 RLVR)·相关性 高(对 mtp_opd 的"在 SFT 与 RL 之间插对齐阶段"有直接启发)

**原始论文**:https://arxiv.org/abs/2604.28123

## 一眼看懂
- 🟦 TL;DR:多模态后训练标准做法是 SFT→RLVR,但 SFT 会引入**分布漂移**——既没充分匹配示范分布,又丢掉模型原有强项;多模态下这种漂移还**异质**(视觉感知错误 vs 推理错误,模式不同,会在后续 RL 里复合放大)。PRISM 在 SFT 与 RLVR 之间插入一个独立"预对齐"阶段:让 policy 与一个含**感知专家+推理专家的 MoE 判别器**做 response 级对抗博弈(black-box,不需要 teacher logits),用判别器分当奖励、GRPO 式更新 policy,把分布拉回示范分布,从而给下游 RLVR 一个更好的初始化。【原文 Abstract 行 24-45,§1 行 151-194】
- 最巧的一步:**把 OPD 从"终点训练目标"重定位为"SFT 与 RLVR 之间的中转对齐阶段",并用 MoE 双专家给出解耦的纠正信号**。命门有二:① 抽掉对齐阶段(=退回标准 SFT→RLVR)→ **−4.4 avg**(Table 2);② 把 MoE 判别器换成等激活算力的单一 dense 判别器 → **−3.4 avg**(WeMath −6.0、MathVerse −4.9)。前者证明"对齐阶段"本身不可或缺,后者证明"感知/推理解耦"是关键——单标量判别器在"一轴改进、另一轴退化"时无法分离两效应、梯度信号变噪。【原文 §4.3 Table 2 行 719-784,行 785-809】

## 为什么做
- 研究背景:LMM 标准后训练=SFT(curated 示范上 bootstrap 能力)→RLVR(自动 verifier 精炼,largely determines final performance)。大量工作分别改进两阶段(SFT 端重加权/正则 next-token,引 iw-SFT Qin&Springenberg 2025、PSFT Zhu 2025;RL 端 GRPO 变体改 importance weighting/clipping/advantage/sequence-level,引 Yu/Zheng/Zhao 2025)。底层直觉:SFT 在参数里建"隐式推理先验",RLVR 在线激活+精炼它(引 Chu 2025、Yue 2025b)。【原文 §1 行 51-59,行 145-150】
- 解决的具体痛点:近期发现 SFT 会把模型置于"两不靠"的受损态——既不充分匹配示范策略分布,又丢失原有有利分布(Kang 2025、Zhang 2026a),即 **SFT 是漂移源而非纯改进**;原因是 SFT 用**统一 token 级目标**模仿示范轨迹、**不区分 process 与 outcome**,导致只学表面 + 偏离原分布;**模型越强,token 级模仿外部示范越易挤占其原生强项**(而非补充);多模态下更突出且**异质**——感知微小偏差会扭曲推理前提并在 RL 中放大,且感知漂移与推理漂移"qualitatively different,单一纠正目标无法兼顾"。核心问题(原文加粗,行 169-171):"**如何在模型进 RL 之前,修复 SFT 引入的(且对感知/推理异质的)分布漂移?**"【原文 §1 行 151-171】
- 相关工作 & 各自不足(§2 两子节,精确到机制):
  - **§2.1 多模态 RLVR**:文本域 DeepSeek-R1 证明纯 RLVR 能涌现 CoT,催生一批稳定化改进(redesigned clipping / advantage estimation / critic-free / sequence-level)。多模态域:cold-start 初始化、跨模态形式化、大规模规则 RL + 涌现反思、课程采样、自反思激励;近期认识到 vanilla RLVR **忽视视觉感知保真**,提出 perception-aware reward(judging LLM Xiao 2025、证据锚定双分支 Zhang 2025a、differential visual reasoning + visual triplets Gao 2026)。**短板:全都在 RL 阶段内做,不碰前置 SFT 留下的分布 gap**(PRISM 的靶点,行 226-228)。
  - **§2.2 On-Policy Distillation**:标准 KD 在 teacher 输出上 SFT(off-policy,有训练-推理分布失配);OPD 在自身生成上训练:**GKD**(Agarwal 2024,灵活散度目标)→ 各种散度(Gu 2024、Ko 2024)→ **logit-free 对抗式**(Ye 2025a);近期沿自蒸馏(Zhao 2026)、reward extrapolation(Yang 2026)、selective imitation(Zhang 2026c)、多模态表征迁移(LLaVA-KD Cai 2025)扩展。**短板:多把蒸馏当终点(checkpoint 即终模型)**,且用**单一无差别**判别器/散度信号(行 241-244)。
  - **最近邻 VOLD**(Bousselham 2025,arXiv:2510.23497,多模态):把 GRPO 与"从纯文本 teacher 的 **logit-based** OPD"合进**单一**训练目标。
- 动机链:SFT 引入异质分布漂移 → token 级再模仿只学表面、修不了 on-policy 生成下的 mismatch,且黑盒监督源(Gemini)无 logits 不能做散度蒸馏 → 所以把对齐建模成**只需监督池样本的 response 级对抗博弈**(OPD 思想,在自身 rollout 上训练以缓解 exposure bias)→ 再用 MoE 双专家给感知/推理**解耦**纠正信号 → 修完漂移再进 RLVR。【原文 §1 行 172-190】
- 与最近邻工作的Δ:vs 标准 OPD——① 把对齐**从 RL 中解耦成独立中转阶段**(不是终点);② **logit-free**(对抗判别,适配 Gemini 等黑盒);③ **解耦反馈**(感知专家 \(D_v\) + 推理专家 \(D_r\)),而非单一判别器。vs **VOLD**——VOLD 需 teacher logits 且合进 RL 单目标;PRISM 在三点上不同:**对齐独立成阶段、无 teacher logits(对抗判别)、解耦感知/推理反馈**(§2.2 行 248-251 明列)。关键有用点:Table 2 证明"解耦"(MoE −3.4 if removed)和"独立对齐阶段"(−4.4 if removed)各自贡献正增益。

## 怎么做 + 靠不靠谱
- 方法流水线(三阶段,§3,每阶段输入→输出):
  - **Stage1 冷启动 SFT(§3.1)**:**输入**=113K 自蒸馏语料(107K)+ 1.26M 公开示范,合计 ≈1.37M。**做什么**=在全量上 SFT **1 epoch**,得初始多模态推理 policy。要点:因同一监督源后续要在对齐阶段复用,每样本须含**正确答案 + 完整推理轨迹 + 准确视觉 grounding**(公开数据多答案短/推理不全/视觉描述不精,故需自蒸馏)。**输出**=SFT checkpoint(narrow 与监督分布的 gap,但仍漂移)。
  - **Stage2 分布对齐 / 对抗 OPD(§3.2,核心)**:**输入**=SFT checkpoint(policy)+ warm-started MoE 判别器 + 监督池 T。**做什么**=极小极大对抗博弈把 policy 分布拉向监督分布。逐组件:
    - **MoE 判别器(§3.2.2)**:每条响应 y 拆成**视觉描述 c** + **推理轨迹 t**;两专家——**感知专家 \(D_v\)** 评 c(视觉 grounding 程度)、**推理专家 \(D_r\)** 评 t(推理一致性/有效性);判别器分按 Eq.1 加权合并。架构=**Qwen3-VL-MoE**,由**四个 Qwen3-VL-2B 组装为专家模块、top-2 路由**;\(D_v,D_r\) 是把视觉描述与推理轨迹**分别**过这一个 MoE 模型得到,再经 Eq.1 合并(行 467-470)。
    - **初始化(§3.2.3)**:对抗前提是"policy 与判别器能力相当";未对齐的 LMM 离监督分布太远会被判别器**秒判可分→饱和→无信息梯度**。故 policy 用 SFT checkpoint 初始化;两专家**从同一预训练 backbone 初始化**,各在指定成分上 **warm-start**(\(D_v\) 在视觉描述偏好对、\(D_r\) 在推理轨迹偏好对),并加 **load-balancing 辅助损失**防专家坍缩。
    - **对抗更新(§3.2.4)**:每 prompt 采 N 条 policy rollout \(\{y_i^-\}\),判别器打分作奖励,转成组内归一 advantage(Eq.3)→ policy 用 **GRPO** 更新;两专家各用 **Bradley-Terry loss**(Eq.2)更新——**给参考响应高分、给 policy rollout 低分**,且**两专家与 policy 联合在线更新**(作为 on-policy 判别器持续跟随演化的 rollout 分布,**避免 reward staleness**,行 399-402)。整体是 \(\min_\theta\max_\phi\) 对抗(Eq.4)。**显式去掉 KL 正则**(锚定 SFT 会与"修 SFT 漂移"直接冲突,行 440-442)。**输出**=固定跑 **500 步**后的对齐 checkpoint。
  - **Stage3 RLVR(§3.3)**:**输入**=对齐 checkpoint。**做什么**=从预留 6K 里按难度筛 pass∈[0.2,0.8] 的 **2K** 题,奖励从"学到的 MoE 判别器"切到确定性可验证奖励 \(r_v=r_{acc}+r_{fmt}\)(Eq.5),沿用与对齐阶段相同的 GRPO 式目标;PRISM 对 RL 算法不可知,实例化 GRPO/DAPO/GSPO。**输出**=最终模型,跑 **1500 步**。
- 逐组件必要性(Table 2,Qwen3-VL-4B + GRPO;full=66.2):
  - **MoE 双专家解耦(vs Dense 4B disc.,等激活算力)**:有消融。Dense=**62.8**(**−3.4**,WeMath 76.9 vs 82.9=−6.0、MathVerse 63.7 vs 68.6=−4.9)。原理:单标量无法分离"一轴升一轴降"的异质漂移,梯度噪声大;§4.4 训练动态(两专家沿不同轨迹收敛)佐证。【行 729-737,785-793】
  - **判别器需 vision-language(vs text-only 同架构)**:有消融。Text-only=**62.3**(−3.9),退化集中在需忠实视觉感知的任务——"parrot alignment"(只学会"听起来像示范"但没真看图)。【行 738-746,810-819】
  - **对齐阶段本身(vs w/o Alignment=标准 SFT→RLVR)**:有消融。w/o Alignment=**61.8**(**−4.4**)→ RLVR 单独补不回 SFT 漂移。【行 757-765,797-809】
  - **SFT 冷启动(vs w/o SFT)**:有消融。w/o SFT=**49.4**(**−16.8**)→ 无冷启动时 policy 离监督分布太远,判别器秒判可分、policy 漂向退化模式,对抗训练失效。【行 748-756,802-806】
  - **SFT 数据规模(107K vs 1.37M)**:有消融。SFT-107K=**62.5**(**−3.7**),但仍 > 全量但无对齐的 SFT→RLVR(61.8)→ SFT(收窄初始 gap)与对齐(闭合残余 gap)**互补而非互替**。【行 766-784,820-827】
  - **专家异步收敛(支撑 MoE 设计,§4.4)**:感知专家**早 peak、快收敛**;推理专家**渐升、震荡更大后稳定**(推理对齐比视觉 grounding 更 nuanced);二者沿不同轨迹但收敛到**可比的均衡水平**(Figure 3,500 步 + 额外延到 900 步验证收敛稳定)。**α 取值本身无单独消融**(默认见 Appendix A)〔推断:主文未扫 α〕。
  - **去 KL(KL=0)**:论文给了理由(锚 SFT 与修漂移冲突,行 440-442)但**未做"带 KL vs 去 KL"的对照消融**〔待核:无 KL 消融数据〕。
- 关键机制/公式(真实形式 + 直觉):
  - **判别器分(Eq.1)**:\(r(x,y)=\alpha\cdot D_v(x,c)+(1-\alpha)\cdot D_r(x,t)\),\(\alpha\) 控感知/推理反馈权衡。
  - **每专家 Bradley-Terry(Eq.2)**:\(L_{D_k}=-\mathbb{E}_{(x,y^+,y^-)\sim T}\big[\log\sigma\big(D_k(x,y_k^+)-D_k(x,y_k^-)\big)\big],\ k\in\{v,r\}\);\(y_k^\pm\) 为参考/policy 响应的第 k 成分(v=视觉描述,r=推理轨迹),\(y^-\sim G(\cdot\mid x)\) 来自当前 policy。给参考高分、给 rollout 低分。
  - **policy 侧组内归一 advantage(Eq.3)**:\(A_i=\dfrac{r(x,y_i^-)-\operatorname{mean}(\{r(x,y_j^-)\}_{j=1}^N)}{\operatorname{std}(\{r(x,y_j^-)\}_{j=1}^N)}\),组内归一,提升"更像监督分布"的响应概率、压低同 prompt 劣 rollout。
  - **极小极大博弈(Eq.4)**:\(\displaystyle\min_\theta\max_\phi\ \mathbb{E}_{(x,y^+)\sim T,\,y^-\sim G_\theta(\cdot\mid x)}\big[r_\phi(x,y^+)-r_\phi(x,y^-)\big]\)(θ=policy,φ=判别器)。交替更新:policy 用 GRPO,两专家用各自 BT 损失。
  - **RLVR 奖励(Eq.5)**:\(r_v(x,y)=r_{acc}(x,y)+r_{fmt}(x,y)\)(答案正确 + 格式合规)。
  - **整体直觉**:"偏离监督分布的方向本身就是奖励"——不用静态 teacher-forced 目标,而让判别器告诉 policy "你这条 rollout 像不像高质量示范",且感知/推理分两路给信号,避免把异质漂移压成一个标量。
- 实验与证据(数字已对 Table 1/2 逐项核对):
  - 数据集/设置:监督语料=从 **Gemini 3 Flash** 蒸馏 **113K**(针对当代强模型零通过率最难题,密集 visual grounding + 逐步推理,经 LLM-based correctness 验证),其中 **107K 用于 SFT、6K 最高质留给对齐+RL**;补 **1.26M** 同 Gemini 家族公开示范,合计 **≈1.37M** SFT(行 268-277)。Backbone Qwen3-VL-4B/8B;判别器=Qwen3-VL-MoE(**4×Qwen3-VL-2B、Top-2 路由**)。训练:SFT 1 epoch、对齐 500 步、RLVR 1500 步(行 471-473)。评测数学(MathVista/MathVerse/MathVision/WeMath)+ 通用(MMMU/MMMU-Pro/HallusionBench)。【§4.1 行 465-478】
  - **关键数字(Table 1)**:**PRISM+GRPO 较 SFT→GRPO:4B 61.8→66.2(+4.4)、8B 63.3→69.3(+6.0)avg**;最大增益集中在 MathVision/WeMath(如 8B MathVision SFT→GRPO 37.1 → PRISM+GRPO 52.0)。DAPO/GSPO 类似增益(算法无关)。对齐 checkpoint(对齐后、RLVR 前)精度与 SFT **相当**(4B 57.2 vs SFT 56.8;8B 59.3 vs 58.1)——对齐改的是分布不是即时准确率(行 495-500)。**8B 上 SFT 把 Instruct 拉低更多**(63.3→58.1),标准 SFT→RLVR(GRPO/GSPO)勉强回到 Instruct,而 PRISM+GRPO **反超 Instruct 5+ 点**(69.3 vs 63.3,行 501-507)。
  - **§4.4 结构代理证据**:用"推理步数 / caption 描述条目数"两个可解释代理看分布演化(Figure 4):base 与监督差很多;SFT 后向监督靠拢但仍 mismatch,**caption 侧 overshoot(描述条目比监督多)**;**对齐阶段显著缩小两维 mismatch,且这一改善 persist through RLVR**(post-RLVR 分布仍跟随相似趋势)——支撑"对齐给了更好的初始化 shape"(行 931-948)。
  - baseline 公平性:**较强**——w/o alignment 基线额外补了等于对齐步数的 RLVR 步以匹配总训练预算(行 480-482);跨 3 种 RL 算法一致 +Δ 是有力证据;PRISM+GRPO 还用更少 tokens/response(Appendix A.3)。
  - 看着强但没回答的:① "black-box" 仅指无需 teacher logits,**监督仍来自 Gemini 3 Flash 蒸馏的 113K+1.26M**,并非 teacher-free;② +4.4/+6.0 为平均增益,单基准方差/负向案例披露有限;③ 判别器奖励本身可靠性(对抗稳定性/reward hacking)只靠 load-balancing + 固定 500 步缓解,未深究是否引入新偏置。
- 假设与失效边界:【原文】① MoE 解耦依赖**结构化响应格式**(显式视觉描述 c + 推理 trace t,才能分给两专家);不天然可分解的任务难直接适用;② 对齐前 policy 与判别器需能力相当,故必须 SFT 冷启动 + 判别器 warm-start(否则判别器秒饱和,§3.2.3 行 356-371);③ 分布对齐分析用结构代理(推理步数/caption 条目)而非 embedding 散度等直接测度(§4.4 行 883-891)。【推断】④ 强依赖 Gemini 高质量蒸馏语料,换弱监督源时"对齐目标"是否仍指向更好分布存疑;⑤ 4×2B 判别器需独立 warm-start,额外内存/算力开销;⑥ α 固定(默认见 Appendix A),感知/推理权重失衡任务下最优 α 未知。
- 祛魅总结:【推断】真贡献=① 把"SFT 引入(且多模态异质)分布漂移"这一现象明确化,并给出"在 SFT 与 RL 间插独立对齐阶段"的清晰方案;② MoE 双专家解耦 + logit-free 对抗这一组合,Table 2 五行消融把每个设计的必要性钉得很实(尤其 w/o SFT −16.8、w/o Align −4.4 对比鲜明)。被高估处:"black-box/teacher-free"——实为 logit-free,监督仍重度依赖 Gemini 蒸馏;"on-policy distillation"在此其实是**对抗式分布匹配**(GAN 式),与经典 logit-KL OPD 不是一回事,容易因名生误。被低估处:"把对齐当中转阶段而非终点"这一**流程位置创新**本身可迁移性强(不限多模态/不限对抗);§4.4 "对齐改善 persist through RLVR"是个少被强调但对"为何插对齐阶段有用"很关键的证据。

## 结构化抽取
- 🎯 机制速览6轴:
  - **学什么信号**:对齐阶段=MoE 判别器给的"像不像监督分布"标量(感知 \(D_v\)/推理 \(D_r\) 两路加权,Eq.1),非真值、非 teacher logits;RLVR 阶段=确定性可验证奖励 \(r_v=r_{acc}+r_{fmt}\)。
  - **改什么**:policy 参数 θ(GRPO 梯度);判别器两专家参数 φ(Bradley-Terry,与 policy 联合在线更新)。
  - **何时改**:三阶段串行;对齐阶段在线 on-policy(judge 持续跟随演化 rollout,避免 reward staleness)、固定 500 步;RLVR 1500 步。
  - **免梯度?**:否,policy 与判别器均梯度更新。
  - **记忆-技能生命周期**:无外部记忆/技能库;监督池(supervision pool)是静态参考分布,既作 SFT 基础又作对齐参照;"对齐后的分布"通过参数固化并在后续 RL 中持续(论文称 corrections persist through RL,§4.4/Conclusion 行 943-959)。
  - **防遗忘机制**:**反其道而行——刻意去 KL(=0)让 policy 自由漂向监督分布**(因目标恰是"修正 SFT 把原分布带偏",锚定反而有害,行 440-442)。无传统防遗忘设计;靠"对齐回监督分布"间接恢复被 SFT 挤占的原生强项。
- ⑦ 开源代码+框架/harness:仓库 https://github.com/XIAO4579/PRISM(MIT;本地已克隆 **133MB**,完整)。含 vendored **transformers-4.57.0/**、**verl/**、**moe/**(MoE 判别器)、scripts/、tools/、difference/。框架=**Stage1 SFT 用 LLaMA-Factory + Stage2 对齐用 verl+transformers-4.57.0+moe(`qwen3_vl_prism`)+ Stage3 RLVR 用 verl(`qwen3_vl_xxpo_after_prism`,GRPO/DAPO/GSPO)**;评测 lmms-eval;数据/checkpoints 在 HF(prism-vlm)。代码/数据/权重均公开(Abstract 行 44-45)。【已核本地仓目录树 + du 133M】
- 💰 资源/成本与可扩展性:对齐阶段虽仅 500 步,但需**同时维持 policy generator 与 MoE 判别器**(4×Qwen3-VL-2B 需独立 warm-start),内存/算力较标准 SFT→RLVR 增加;RLVR 1500 步、PRISM+GRPO 用更少 tokens/response(Appendix A.3)。具体 GPU 卡数/总时长 + α/温度/group size N 见 Appendix A〔主文未给绝对卡时与 α 值〕。
- 🎯 对"探索-巩固"对标:**中等支撑,偏"巩固/纠偏"一侧 + 流程位置启发**。PRISM 没做"探索/选路"(没有 student 自选可走通开头、没有 path-recovery 接管),它解决的是"SFT 把分布带偏后如何**巩固/纠偏回正确分布**"——更接近 idea 里"巩固=固化进参数且不(把原能力)遗忘"的那一半,但用的是对抗分布匹配而非脚手架蒸馏。对 mtp_opd 的真正启发点:**"在 SFT 与 RL 之间插一个 on-policy 的对齐/巩固阶段"这一流程位置**——可类比"先用 MTP/teacher 把 student 的分布对齐到能稳定走通的路径,再放它做 RLVR 探索"。可借组件:① "判别器/奖励来自'是否像可走通的监督分布'"可替代显式 teacher logits(适配黑盒 teacher);② "感知/推理解耦专家"提示——对 path-selection 与 path-recovery 也可用**解耦信号**(选路专家 vs 回轨专家)而非单一标量;③ §4.4 "对齐改善 persist through RLVR" 提示巩固阶段的分布改造能被后续探索保留,对"先巩固再探索"流程有利。缺口:无探索/自选路径、无前瞻(MTP)、无记忆/技能库、纯多模态对抗、依赖结构化响应格式与强 teacher 蒸馏数据。一句判定:**流程位置(SFT↔RL 间插 on-policy 对齐阶段)与"解耦纠正信号"两点可借**,但机制(GAN 式对抗、黑盒蒸馏)与"教师脚手架探索-巩固"差距较大,作启发非直接基线。
- 🔭 开放问题/未来方向:【原文】未单列开放问题节;Conclusion 强调对齐缩小 SFT 残余 gap 且 persist through RL。【推断】① 把"对抗分布匹配"换成"MTP 前瞻一致性"作对齐信号,验证非对抗版是否更稳;② 在纯文本/推理任务上验证"SFT↔RL 间插对齐阶段"是否同样有效(去掉多模态特异性);③ 解耦专家思路迁移到"选路 vs 回轨"双信号,做 TSRD 式 path-selection/recovery 的解耦奖励;④ 系统研究 α 与去 KL 的边界(论文均只取默认值,无消融);⑤ 推广到不天然可分解(无显式 caption/trace 结构)的任务,需自动/学习式的判别器反馈分解。

— RETURN —
prism | 读到PDF? 是(26页/82k字重抽清洗null字节后,§1-§5+Eq.1-5+Table 1/2全核,数据split 107K/6K/1.26M/1.37M 与消融−4.4/−3.4/−16.8/−3.9/−3.7 逐项核对) | L线 L1(OPD新用法)+L3(多模态RLVR) | 对标结论:流程位置(SFT↔RL间插on-policy对齐/巩固阶段)+"感知/推理解耦纠正信号"+"对齐改善persist through RLVR"三点可借鉴(可类比选路/回轨解耦),但机制为GAN式对抗+黑盒蒸馏、无探索/前瞻/记忆,偏"巩固纠偏"一侧,作启发非直接基线 | 残留待核数:2(α 未做扫描消融、去KL无对照消融;绝对GPU卡时与α默认值主文未给见Appendix A)

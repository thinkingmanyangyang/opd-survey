prism | Beyond SFT-to-RL: Pre-alignment via Black-Box On-Policy Distillation for Multimodal RL (PRISM) | HKUST(广州)+清华+南洋理工+人大+中科大+国科大(Sudong Wang, Weiquan Huang 等;多模态 RLVR + OPD 谱系,承接 GKD/对抗式 logit-free 蒸馏) | 2026-05 预印本·arXiv·v2(2026-05-01,cs.CV) | 主题线 L1(OPD 新用法)+L3(多模态 RLVR)·相关性 高(对 mtp_opd 的"在 SFT 与 RL 之间插对齐阶段"有直接启发)

**原始论文**:https://arxiv.org/abs/2604.28123

## 一眼看懂
- 🟦 TL;DR:多模态后训练标准做法是 SFT→RLVR,但 SFT 会引入**分布漂移**——既没充分对齐示范分布,又丢掉模型原有强项;多模态下这种漂移还**异质**(视觉感知错误 vs 推理错误,模式不同,会在后续 RL 里复合放大)。PRISM 在 SFT 与 RLVR 之间插入一个独立"预对齐"阶段:让 policy 与一个含**感知专家+推理专家的 MoE 判别器**做 response 级对抗博弈(black-box,不需要 teacher logits),用判别器分当奖励、GRPO 式更新 policy,把分布拉回示范分布,从而给下游 RLVR 一个更好的初始化。【原文 Abstract 行 23-45,§1 行 168-188】
- 最巧的一步:**把 OPD 从"终点训练目标"重定位为"SFT 与 RLVR 之间的中转对齐阶段",并用 MoE 双专家给出解耦的纠正信号**。命门有二:① 抽掉对齐阶段(=退回标准 SFT→RLVR)→ −4.4 avg(Table 2);② 把 MoE 判别器换成等算力的单一 dense 判别器 → −3.4 avg(WeMath −6.0)。前者证明"对齐阶段"本身不可或缺,后者证明"感知/推理解耦"是关键——单标量判别器在"一轴改进、另一轴退化"时梯度信号变噪,无法分离异质漂移。【原文 §4.3 Table 2 行 712-777,行 778-801】

## 为什么做
- 研究背景:LMM 标准后训练=SFT(curated 示范上 bootstrap 能力)→RLVR(自动 verifier 精炼,决定最终性能)。大量工作分别改进两阶段(SFT 端重加权/正则 next-token;RL 端 GRPO 变体改 importance weighting/clipping)。【原文 §1 行 48-59,§2.1 行 209-226】
- 解决的具体痛点:近期发现 SFT 会把模型置于"两不靠"的受损态——既不充分匹配示范策略分布,又丢失原有有利分布(Kang/Zhang 2026),即 SFT 是漂移源而非纯改进;**模型越强,token 级模仿外部示范越易挤占其原生强项**;多模态下更突出且**异质**(感知 vs 推理漂移模式不同,单一纠正目标无法兼顾,且感知微小偏差会扭曲推理前提并在 RL 中放大)。【原文 §1 行 150-170】
- 相关工作 & 各自不足:① 多模态 RLVR 改进(perception-aware reward、双分支、visual triplets)——全都**在 RL 阶段内**做,不碰前置 SFT 留下的分布 gap(PRISM 的靶点);② OPD 系(GKD/各种散度/logit-free 对抗 Ye 2025/自蒸馏/选择性模仿)——多把蒸馏当**终点**(checkpoint 即终模型),且用**单一无差别**判别器/散度;③ VOLD(多模态)——把 GRPO 与"从纯文本 teacher 的 logit-based OPD"合进单一目标。【原文 §2.1-2.2 行 209-248】
- 动机链:SFT 引入异质分布漂移 → token 级再模仿只学表面、修不了 on-policy 生成下的 mismatch,且黑盒监督源(Gemini)无 logits 不能做散度蒸馏 → 所以把对齐建模成**只需监督池样本的 response 级对抗博弈**(OPD 思想)→ 再用 MoE 双专家给感知/推理**解耦**纠正信号 → 修完漂移再进 RLVR。【原文 §1 行 168-188,§3.2.1 行 282-291】
- 与最近邻工作的Δ:vs 标准 OPD——① 把对齐**从 RL 中解耦成独立中转阶段**(不是终点);② **logit-free**(对抗判别,适配 Gemini 等黑盒);③ **解耦反馈**(感知专家 D_v + 推理专家 D_r),而非单一判别器(§2.2 行 245-248 明列三点差异)。vs VOLD——VOLD 需 teacher logits 且合进 RL 单目标;PRISM 无 logits、对齐独立。关键有用点:Table 2 证明"解耦"(MoE −3.4 if removed)和"独立对齐阶段"(−4.4 if removed)各自贡献正增益。

## 怎么做 + 靠不靠谱
- 方法流水线(三阶段,§3):① **Stage1 冷启动 SFT**——在 1.37M 高质量示范(107K 自蒸馏 + 1.26M 公开,同 Gemini 家族)上 SFT 1 epoch,得初始多模态推理 policy;② **Stage2 分布对齐(adversarial OPD)**——对每个 prompt 采 N 条 policy rollout,MoE 判别器(感知专家查 visual grounding + 推理专家查推理一致性,Eq.1 加权 α=0.5)打分作奖励,policy 用 GRPO 式组内归一 advantage(Eq.3)更新、两专家各用 Bradley-Terry loss(Eq.2)更新,交替进行成极小极大博弈(Eq.4);**显式去掉 KL 正则(KL 系数=0)**(锚定 SFT 会与"修 SFT 漂移"冲突);跑固定 500 步取 checkpoint;③ **Stage3 RLVR**——在对齐 checkpoint 上,从预留 6K 里筛 pass∈[0.2,0.8] 的 2K 题,用确定性可验证奖励(r=racc+rfmt,Eq.5),GRPO/DAPO/GSPO 任选,1500 步。
- 逐组件必要性(Table 2,Qwen3-VL-4B + GRPO):
  - **MoE 双专家解耦(vs 单 dense 等算力)**:有消融。Dense 4B disc.=62.8 vs full=66.2(**−3.4**,WeMath −6.0、MathVerse −4.9)。原理:单标量无法分离"一轴升一轴降"的异质漂移,梯度噪声大。【行 722-730,778-786】
  - **判别器需是 vision-language(vs text-only 同架构)**:有消融。Text-only=62.3(−3.9),退化集中在需忠实视觉感知的任务——"parrot alignment"(只学会"听起来像示范"但没真看图)。【行 731-739,802-811】
  - **对齐阶段本身(vs w/o Alignment=标准 SFT→RLVR)**:有消融。w/o Alignment=61.8(**−4.4**)→ RLVR 单独补不回 SFT 漂移。【行 750-758,789-794】
  - **SFT 冷启动(vs w/o SFT)**:有消融。w/o SFT=49.4(**−16.8**)→ 无冷启动时 policy 离监督分布太远,判别器秒判可分、policy 漂向退化模式,对抗训练失效。【行 741-749,794-801】
  - **SFT 数据规模(107K vs 1.37M)**:有消融。SFT-107K=62.5(−3.7),但仍 > 全量但无对齐的 SFT→RLVR(61.8)→ SFT(收窄初始 gap)与对齐(闭合残余 gap)互补而非互替。【行 759-777,812-819】
  - **专家异步收敛(支撑 MoE 设计)**:§4.4 分析——感知专家早收敛、推理专家后收敛,两者沿不同轨迹,佐证解耦设计(行 861-869);**α 取值本身无单独消融**(默认 α=0.5)〔推断:仅给默认值,未扫 α〕。
  - **去 KL(KL=0)**:论文给了理由(锚 SFT 与修漂移冲突,行 435-437/1255-1256)但**未做"带 KL vs 去 KL"的对照消融**〔待核:无 KL 消融数据〕。
- 关键机制/公式(直觉):① 判别器分 r=α·D_v(感知)+(1−α)·D_r(推理)(Eq.1);② 每专家用 Bradley-Terry(Eq.2):给参考响应高分、给 policy rollout 低分,**且两专家与 policy 联合在线更新**(on-policy 判别器,持续跟随演化的 rollout 分布,避免 reward staleness,行 394-397);③ policy 侧把判别器分转成组内归一 advantage(Eq.3)做 GRPO 更新;④ 整体是 min_θ max_φ 的对抗博弈(Eq.4)。直觉:"偏离监督分布的方向本身就是奖励"——不用静态 teacher-forced 目标,而让判别器告诉 policy "你这条 rollout 像不像高质量示范"。
- 实验与证据:
  - 数据集/设置:监督语料=从 **Gemini 3 Flash** 蒸馏 **113K**(针对当代强模型零通过率最难题,密集 visual grounding + 逐步推理,多级过滤),其中 **107K SFT、6K 最高质留给对齐+RL**;补 **1.26M** 同 Gemini 家族公开示范,合计 **≈1.37M** SFT。Backbone Qwen3-VL-4B/8B;判别器=Qwen3-VL-MoE(**4×Qwen3-VL-2B 组装为专家、Top-2 路由**,行 459-464)。评测数学(MathVista/MathVerse/MathVision/WeMath)+ 通用(MMMU/MMMU-Pro/HallusionBench),lmms-eval。【§4.1 行 459-478】
  - 关键数字:**PRISM+GRPO 较 SFT→GRPO:4B +4.4、8B +6.0 avg**(行 483-484);DAPO/GSPO 类似增益(算法无关)。对齐 checkpoint(对齐后、RLVR 前)精度与 SFT 相当(行 489-494)——对齐改的是分布不是即时准确率。8B 上 SFT 把 Instruct 拉低更多,PRISM+GRPO 反超 Instruct 5+ 点(行 495-501)。
  - baseline 公平性:**较强**——w/o alignment 基线额外补了等于对齐步数的 RLVR 步以匹配总训练预算(行 474-476);跨 3 种 RL 算法一致 +Δ 是有力证据。
  - 看着强但没回答的:① "black-box" 仅指无需 teacher logits,**监督仍来自 Gemini 3 Flash 蒸馏的 113K+1.26M**,并非 teacher-free;② +4.4/+6.0 为平均增益,单基准方差/负向案例披露有限;③ 判别器奖励本身的可靠性(对抗训练稳定性/reward hacking)只靠 load-balancing + 固定 500 步缓解,未深究是否引入新偏置。
- 假设与失效边界:【原文】① MoE 解耦依赖**结构化响应格式**(显式 <caption> 视觉描述 + <think> 推理 trace),不天然可分解的任务难直接适用(行 1480-1482);② 对齐前 policy 与判别器需能力相当,故必须 SFT 冷启动 + 判别器 warm-start(否则判别器秒饱和,行 352-357);③ 分布对齐分析用结构代理(推理步数/caption 条目)而非 embedding 散度等直接测度(行 1483-1485)。【推断】④ 强依赖 Gemini 高质量蒸馏语料,换弱监督源时"对齐目标"是否仍指向更好分布存疑;⑤ 4×2B 判别器需独立 warm-start,额外内存/算力开销(论文坦承,行 1477-1479);⑥ α=0.5 固定,感知/推理权重失衡任务下最优 α 未知。
- 祛魅总结:【推断】真贡献=① 把"SFT 引入(且多模态异质)分布漂移"这一现象明确化,并给出"在 SFT 与 RL 间插独立对齐阶段"的清晰方案;② MoE 双专家解耦 + logit-free 对抗这一组合,Table 2 五行消融把每个设计的必要性钉得很实(尤其 w/o SFT −16.8、w/o Align −4.4 对比鲜明)。被高估处:"black-box/teacher-free"——实为 logit-free,监督仍重度依赖 Gemini 蒸馏;"on-policy distillation"在此其实是**对抗式分布匹配**(GAN 式),与经典 logit-KL OPD 不是一回事,容易因名生误。被低估处:"把对齐当中转阶段而非终点"这一**流程位置创新**本身可迁移性强(不限多模态/不限对抗)。

## 结构化抽取
- 🎯 机制速览6轴:
  - **学什么信号**:对齐阶段=MoE 判别器给的"像不像监督分布"标量(感知/推理两路加权),非真值、非 teacher logits;RLVR 阶段=确定性可验证奖励(acc+format)。
  - **改什么**:policy 参数 θ(GRPO 梯度);判别器两专家参数 φ(Bradley-Terry,与 policy 联合在线更新)。
  - **何时改**:三阶段串行;对齐阶段在线 on-policy(judge 持续跟随演化 rollout)、固定 500 步;RLVR 1500 步。
  - **免梯度?**:否,policy 与判别器均梯度更新。
  - **记忆-技能生命周期**:无外部记忆/技能库;监督池(supervision pool)是静态参考分布,既作 SFT 基础又作对齐参照;"对齐后的分布"通过参数固化并在后续 RL 中持续(论文称 corrections persist through RL,行 947-949)。
  - **防遗忘机制**:**反其道而行——刻意去 KL(=0)让 policy 自由漂向监督分布**(因为目标恰是"修正 SFT 把原分布带偏",锚定反而有害,行 435-437)。无传统防遗忘设计;靠"对齐回监督分布"间接恢复被 SFT 挤占的原生强项。
- ⑦ 开源代码+框架/harness:仓库 https://github.com/XIAO4579/PRISM(MIT;本地已克隆 **133MB**,完整)。含 vendored **transformers-4.57.0/**、**verl/**、**moe/**(MoE 判别器)、scripts/、tools/、difference/。框架=**Stage1 SFT 用 LLaMA-Factory + Stage2 对齐用 verl+transformers-4.57.0+moe(`qwen3_vl_prism`)+ Stage3 RLVR 用 verl(`qwen3_vl_xxpo_after_prism`,GRPO/DAPO/GSPO)**;评测 lmms-eval;数据/checkpoints 在 HF(prism-vlm)。项目页 https://xiao4579.github.io/PRISM/。代码/数据/权重均公开(Abstract 行 44-45)。【已核本地仓目录树 + du 133M】
- 💰 资源/成本与可扩展性:对齐阶段虽仅 500 步,但需**同时维持 policy generator 与 MoE 判别器**,内存/算力较标准 SFT→RLVR 增加(论文坦承,行 1477-1479);判别器=4×Qwen3-VL-2B 需独立 warm-start。RLVR:global batch 32、N=16 rollout、温度 1.0、1500 步(行 1260-1261)。具体 GPU 卡数/总时长见 Appendix A Table 3〔主文未给绝对卡时〕;PRISM+GRPO 还用更少 tokens/response(Appendix A.3)。
- 🎯 对"探索-巩固"对标:**中等支撑,偏"巩固/纠偏"一侧 + 流程位置启发**。PRISM 没做"探索/选路"(没有 student 自选可走通开头、没有 path-recovery 接管),它解决的是"SFT 把分布带偏后如何**巩固/纠偏回正确分布**"——更接近 idea 里"巩固=固化进参数且不(把原能力)遗忘"的那一半,但用的是对抗分布匹配而非脚手架蒸馏。对 mtp_opd 的真正启发点:**"在 SFT 与 RL 之间插一个 on-policy 的对齐/巩固阶段"这一流程位置**——可类比"先用 MTP/teacher 把 student 的分布对齐到能稳定走通的路径,再放它做 RLVR 探索"。可借组件:① "判别器/奖励来自'是否像可走通的监督分布'"可替代显式 teacher logits(适配黑盒 teacher);② "感知/推理解耦专家"提示——对 path-selection 与 path-recovery 也可用**解耦信号**(选路专家 vs 回轨专家)而非单一标量。缺口:无探索/自选路径、无前瞻(MTP)、无记忆/技能库、纯多模态对抗、依赖结构化响应格式与强 teacher 蒸馏数据。一句判定:**流程位置(SFT↔RL 间插 on-policy 对齐阶段)与"解耦纠正信号"两点可借**,但机制(GAN 式对抗、黑盒蒸馏)与"教师脚手架探索-巩固"差距较大,作启发非直接基线。
- 🔭 开放问题/未来方向:【原文】① 推广到不天然可分解的任务(探索自动/学习式判别器反馈分解);② scale 到更大基座;③ 对齐阶段能否进一步缩短或在多个下游 RL 目标间摊销;④ 开发 model-agnostic 的对齐度量(替代结构代理)(§E 行 1486-1489)。【推断】① 把"对抗分布匹配"换成"MTP 前瞻一致性"作对齐信号,验证非对抗版是否更稳;② 在纯文本/推理任务上验证"SFT↔RL 间插对齐阶段"是否同样有效(去掉多模态特异性);③ 解耦专家思路迁移到"选路 vs 回轨"双信号,做 TSRD 式 path-selection/recovery 的解耦奖励;④ 系统研究 α 与去 KL 的边界(论文均只取默认值,无消融)。

— RETURN —
prism | 读到PDF? 是(26页/82k字,§1-§4+Appendix A/B/D/E 全核,Table 1/2 数字与 α/KL/N 超参核对) | L线 L1(OPD新用法)+L3(多模态RLVR) | 对标结论:流程位置(SFT↔RL间插on-policy对齐/巩固阶段)+"感知/推理解耦纠正信号"两点可借鉴(可类比选路/回轨解耦),但机制为GAN式对抗+黑盒蒸馏、无探索/前瞻/记忆,偏"巩固纠偏"一侧,作启发非直接基线 | 残留待核数:2(α 未做扫描消融、去KL无对照消融;绝对GPU卡时主文未给见Appendix A Table 3)

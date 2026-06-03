sed_sft | SED-SFT: Selectively Encouraging Diversity in Supervised Fine-Tuning | 腾讯 WeChat AI(Yijie Chen*、Yijin Liu*、Fandong Meng) | 2026-02-07 arXiv 2602.07464(v1, cs.CL)·预印本 | 主题线 L2(统一 SFT-RL/为 RL 保探索)+L6(token 信用)·相关性 中

**原始论文**:https://arxiv.org/abs/2602.07464

## 一眼看懂
- 🟦 TL;DR:标准 SFT 的交叉熵(CE)把概率全压向唯一正确路径→**mode collapse**(生成多样性塌缩),限制后续 RL 的探索效率。SED-SFT 在 CE 之上加一个**选择性的"多样性鼓励项"**:对"探索空间大"的 token(用 Top-k 累积概率判定),加一个把 ground-truth 概率往 0.5(最大熵点)推的二次惩罚;对"探索空间小"的 token(固定连接词/结构 token,本就高置信)用掩码跳过、不鼓励。结果 RL 后相对 CE 平均仅 +1.20(Qwen2.5-Math-7B)/+2.06(Llama-3.2-3B)。
- 最巧的一步:**Top-k 累积概率掩码 Mt**(选择性地只对高探索空间 token 施加多样性鼓励)。抽掉它(SED-SFT w/o mask),RL 后 Qwen 56.60(满配 57.20)、Llama 35.10(满配 37.71)——介于 CE 与满配之间,说明"选择性"贡献了约一半增量。核心 insight:**盲目鼓励多样性有害**(对结构 token 鼓励多样性既无必要又掉精度),掩码是把"多样性"与"精度"调平衡的关键。【原文】§2.2、§3、Table 1

## 为什么做
- 研究背景:LLM 后训练主流是 SFT→RL。SFT 用 CE 把概率推向 target label,隐式压制多样性(O'Mahony 2024);这种多样性塌缩严重限制后续 RL 的探索空间。已有改进要么只作用在 RL 阶段(不适用 SFT),要么 SFT 阶段的 trade-off 在 RL 后补不回来。【原文】§1
- 解决的具体痛点:RL 阶段的熵控制方法(Wang 2025 的 80/20 高熵 token、Cui 2025 熵机制)不适用 SFT——SFT 的 mode collapse 由 token 级 CE 拟合诱发,机制不同。而现有 SFT 改法各有缺陷(见下)。【原文】§1、§6
- 相关工作 & 各自不足:三条线——(1) **RL Policy Integration**:DFT(Wu 2025)用置信度重加权防过大梯度保探索;ASFT(Zhu 2025a)在 DFT 上修偏移但需引 reference model、成本高。此线调更新方向但**不从根本上增多样性、反而可能压探索空间**,限制后续 RL。(2) **Diversity Modeling**:GEM(Li 2024)在目标里加 reverse-KL+熵直接鼓励多样,但**忽视 token 间差异**,在高精度任务(数学)上掉点。(3) **Selective Gradient Updating**:CTF(Ruan 2025)需预先定重要更新位置,依赖先验/高成本算法,通用性差。【原文】§6
- 动机链:CE 致 mode collapse → 限制 RL 探索 → 既有改法要么不管 SFT 要么 trade-off 补不回 → 但"盲目鼓励多样性"也有害(case study:结构 token 探索空间小、置信度远高于均值)→ 所以应**选择性**地只对探索空间大的 token 鼓励多样性。
- 与最近邻工作的Δ:vs GEM——不再对所有 token 一视同仁鼓励多样性,而用 Top-k 掩码**选择性**施加(故在数学这类高精度任务上不掉点);vs DFT——DFT SFT 阶段强但压探索空间致 RL 后垫底,SED-SFT 反过来保探索空间换 RL 后增益;vs CTF——掩码用可观测的 Top-k 累积概率自动判定,不需高成本先验。关键差别:**用 Top-k 累积概率这个廉价代理来"选择性"地定位该不该鼓励多样性的 token**。

## 怎么做 + 靠不靠谱
- 方法流水线:仅替换 SFT 目标,其余 SFT/RL 全保持一致。
  1. **Top-k 掩码 Mt**:PTop-k(t)=前 k 个最高概率之和;Mt=1[PTop-k(t)<τ];阈值 τ=该批训练样本累积概率集合 P 的 (1−r) 分位数(r=掩码比例)。
  2. **多样性鼓励函数** LDE(p)=(p−1/2)²(受 CHORD 启发的二次惩罚,p=ground-truth token 概率);p=0.5 处最小、p=1 或 0 处最大,把概率往 0.5 推。
  3. **总损失** L=Σt[−log πθ(y*t)+λ·Mt·LDE(πθ(y*t))],所有实验 λ=1。
  4. 走标准 SFT→RL:SFT 后用 verl GRPO 做 RL。【原文】§3
- 逐组件必要性:
  - **掩码 Mt**:做了消融(SED-SFT w/o mask,Table 1)——去掉后 RL 增量减半,必要性成立。
  - **k>1**:Table 3 敏感性——k=1 时 34.68 < k=2 的 37.71,k=3 为 36.89;论文解释 k=1 区分度高但看不到替代路径,k 太大趋均匀失区分力,故 k=2/3。
  - **r>0.5**:Table 3——r=0.2 时 34.18 反低于 CE 35.65,r=0.5 时 35.73,r=0.7 最高 37.71,r=0.8 回落 35.68;故 r 须 >0.5,最佳 0.7。
  - 消融较完整(掩码 + k + r 都测了),但**只在 Llama 上做敏感性**、未给多 seed 方差。【原文】Table 1/3、§5.2
- 关键机制/公式(直觉):把 SFT 视为策略更新、y* 选择视为二值事件,则 ground-truth 概率 p 决定更新幅度;LDE 把 p 推向 0.5 等于在二值事件下推向最大熵,从而把概率质量分给替代路径。Top-k 累积概率作为"token 探索空间"的可观测代理(对语义等价分支鲁棒,Kuhn 2023)。【原文】§2.2、§2.3、§3
- 实验与证据:
  - **数据集**:SFT=Micomind 采 20,000 条(lr 2e-5、DeepSpeed stage-2,沿用 GEM);RL=MATH(Level 1)训练切分,每 prompt 用 Qwen2.5-Math-7B-Instruct 采 8 次、过滤全对/全错(类离线 DAPO),从 5,000 得 **2,069** 条;verl GRPO,batch 256,余默认。评测 8 个数学基准(遵循 Qwen2.5-Math 框架):AIME24/25、AMC23、GSM8K、MATH500、GAOKAO-en、OlympiadBench、College-MATH(前三 Avg@8 temp 0.7,余采样 1 次 temp 1.0)。骨干 Qwen2.5-Math-7B-Instruct、Llama-3.2-3B-Instruct;全程 8×H20。【原文】§4.1、§4.2
  - **关键数字**(Table 1,RL 后 8 基准均值):Qwen CE 56.00→SED-SFT **57.20(+1.20)**;Llama CE 35.65→**37.71(+2.06)**。w/o mask:Qwen 56.60、Llama 35.10(介于 CE 与满配)。**SFT 阶段自身**:SED-SFT 几乎不优于甚至略低于 CE(Qwen SFT 42.29 vs CE 43.33;Llama 22.04 vs CE 22.25);而 DFT SFT 阶段大幅领先(Qwen 54.24/Llama 25.43)但**RL 后反而垫底**(Qwen 55.36/Llama 30.95)。多样性(Self-BLEU↓,Table 2):SED-SFT 35.57 < GEM 38.53 < CE 43.12 < DFT 51.26。【原文】Table 1/2
  - **baseline 公平吗**:所有方法仅变 SFT 目标、其余训练评测全一致(§4.1),口径严格对齐,公平。
  - **看着强但没回答核心问题?**:核心主张"SFT 多样性→RL 收益"由"SED-SFT SFT 阶段 Self-BLEU 最低且 SFT 精度不掉、RL 后最高"+"DFT 反例(SFT 强 RL 弱)"双向支撑,逻辑链成立;但**绝对增量仅 1~2 点**,且 SFT 阶段自身几乎无收益,卖点完全押在"传导到 RL"这一间接链路。【推断】
- 假设与失效边界:
  - 【原文】方法定位为数学推理任务(§1 "focuses primarily on mathematical reasoning")。
  - 【推断】隐含假设"Top-k 累积概率高=探索空间小=不该鼓励多样性"——对语义等价分支多的 token 可能误判;λ=1 固定、未扫;只两个骨干、RL 仅 MATH-L1 单数据集、无多 seed 方差,跨域/跨规模稳健性未验证。
  - 【待核/已验证】**代码与论文实现差异**(来自既有 analysis 对 clone 仓库的核查):仓库 `utils/sed_triton_loss.py` 中 SEDLoss 默认 `entropy_penalty_scale=0.2`(即 λ=0.2,非论文 λ=1),掩码用固定 `cumsum_threshold=0.95` 而非论文的 (1−r) 分位数自适应阈值;分位数/动态掩码分支声明了但 forward 默认走固定阈值。复现论文配置需自行对齐 λ 与阈值策略。
- 祛魅总结:真贡献=**指出"盲目鼓励多样性有害"并用 Top-k 累积概率掩码做"选择性鼓励"**,case study + 消融 + 敏感性较扎实。包装/局限:(1) 本质是 **GEM/CHORD 思路 + Top-k 累积概率掩码的工程化变体**,核心新意仅在"选择性掩码";(2) 增量仅 1~2 点,SFT 阶段自身不如 CE,卖点押在间接链路;(3) 证据面窄(两骨干、单 RL 数据集、无方差);(4) 代码默认超参与论文不一致,易致复现偏差(见上)。【推断】

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=CE(拟合 ground-truth)+ 选择性的"把高探索空间 token 概率推向 0.5"的熵正则｜**改什么**=SFT 阶段策略参数(改的是 SFT 损失,非数据/架构)｜**何时改**=SFT 阶段(目标替换),下游 RL 不变｜**免梯度?**=否(SFT 梯度训练)｜**记忆-技能生命周期**=无记忆/技能库概念｜**防遗忘机制**=无显式防遗忘;但其"保多样性/保探索空间"目标间接缓解 SFT 对预训练分布的过度收窄(可视为弱"防塌缩")。【原文】§3
- ⑦ 开源代码+框架/harness:https://github.com/pppa2019/SED-SFT(已 clone ~15MB)。框架=**自带 torch/transformers/DeepSpeed ZeRO-2 SFT trainer**(`sft_trainer_v2.py`+`train.py`)+ **verl GRPO**(仓库内 `verl/` 子目录)做 RL。核心损失 `utils/sed_triton_loss.py`(SEDLoss,Triton 实现 CE+(p−0.5)²+Top-k 累积掩码),另含 `ce_triton_loss.py`/`gem_triton_loss.py` 对照;脚本在 `train_scripts/`。代码核心可定位可复现,但默认超参与论文不一致(见失效边界)。【原文】GitHub + 既有 analysis 对 clone 的核查
- 💰 资源/成本与可扩展性:论文称多样性鼓励项相对 CE 计算开销**几乎可忽略**(Triton 核实现);全程 8×H20;SFT 20k 样本、RL 仅 2,069 条,数据成本低。【原文】§3、§4.1
- 🎯 对"探索-巩固"对标:**弱-中支撑,落在"探索/保探索空间"这一侧,但不涉 teacher 蒸馏也不涉 path-recovery**。SED-SFT 关心"SFT 阶段别把探索空间压死、给后续 RL 留探索余地"——对应 idea 的"探索=保留有效路径/分支的可能性";它对"探索空间大的 token"选择性保多样性,与 idea"切关键步/在高熵处接管"有**机制同构**(都在做 token 级的选择性处理,只是这里鼓励多样性、idea 是做 path-recovery)。**可借组件**:(a) **Top-k 累积概率掩码**可作"识别该不该在某 token 上探索/接管"的廉价门控,直接可移植到 MTP/OPD 的 token 级信用或 path-selection;(b)"SFT 阶段保探索空间→不损害后续可塑性"这一原则支持 idea 的"巩固不应损害继续学习的能力(防遗忘的另一面)"。**缺口/差异**:无 teacher 脚手架、无 on-policy 自选、无 path-recovery、无 MTP 前瞻;它是纯 SFT 损失正则,与 idea 的"teacher 教两能力"关系间接。一句判定:**"保探索空间不损可塑性"的弱对标 + Top-k 掩码这个可借的 token 级门控,但本身与 OPD/path-recovery 关系间接,作机制借鉴而非直接竞品**。【推断,依据 §2.2/§3 vs idea】
- 🔭 开放问题/未来方向:【原文】结论未列明确 future work(仅总结贡献)。【推断】(1) 扩到数学外的任务域验证"选择性多样性"的普适性;(2) 把固定 λ=1 改为自适应(按 token 探索空间动态调权);(3) 对齐代码默认超参与论文配置;(4) 把 Top-k 掩码与 RL 阶段的熵控制(80/20、熵机制)联动,做 SFT-RL 全程一致的探索空间管理;(5) 报多 seed 方差以确证 1~2 点增量的显著性。

RETURN:sed_sft | 读到PDF? 是(8页全文+全表Table1-3+全相关工作+附录heatmap说明) | L2(+L6) | 对标=弱-中,"保探索空间不损可塑性"弱对标+Top-k掩码可借作token级门控,但与OPD/path-recovery关系间接 | 残留待核 1(代码默认λ=0.2/固定阈值与论文λ=1/分位数不一致,已记录待对齐)

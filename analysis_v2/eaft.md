eaft | Entropy-Adaptive Fine-Tuning: Resolving Confident Conflicts to Mitigate Forgetting | 北京邮电大学 PRIS-CV / 中关村学院（Muxi Diao、Lele Yang、Wuxuan Gong 共一；通讯 Zhanyu Ma） | 2026-01-05 · arXiv 预印本(曾居 HF Daily Paper #1) · v1 | 主题线 L5(记忆/持续学习·防遗忘)，兼 L2(SFT 改造) · 相关性 高

**原始论文**：https://arxiv.org/abs/2601.02151

## 一眼看懂

> 一句话导读：SFT 会损害通用能力,本文把矛头指向一类具体 token——模型自己很有把握、标签却逼它改口。办法是用"每个 token 的熵"当闸门:模型越笃定就越少更新它,越不确定才越像普通 SFT。一行 loss 改动,不碰 RL。

- 🟦 TL;DR：SFT 损害通用能力,主因不是"训练流程"本身,而是一类特定 token——**模型自己很笃定(低熵)、标签却逼它输出另一个 token(对该标签给的概率很低)**。论文给它起名叫"Confident Conflict(自信冲突)"。
  - EAFT 的做法:用**每个 token 的归一化熵 \(\tilde H_t\)** 当门控系数,去缩放交叉熵损失。熵在这里就是模型对下一 token 不确定度的度量——熵低=很笃定,熵高=拿不准。
  - 效果上:模型越笃定(熵越低),它的梯度越被压到接近 0;越不确定(熵越高),就越退化回普通 SFT。
  - 代价极小:只换一个损失函数,不引入 RL、不加额外模型,就把遗忘缩到约 SFT 的 1/4~1/5,同时目标任务不掉分。【原文 §Abstract / §3.3 / Eq.2–3】
- 最巧的一步：**把门控信号从"概率"换成"熵"**。
  - 抽掉这一步(退回到看概率,也就是 DFT / FLOW / TALR 那套)方法就垮。原因:低概率 token 其实有两种来源——
    - (a) 模型真不会、该学(epistemic uncertainty,认知不确定性,此时熵也高);
    - (b) 模型很有把握想输出别的、只是和标签相悖(Confident Conflict,此时熵低)。
  - 只看概率会把这两类一视同仁,甚至在 (b) 上**放大**破坏性梯度、加速遗忘(原文:"prior methods risk accelerating forgetting");而熵恰好能把"该学(高熵)"和"该抑制(低熵)"分开。【原文 §2 末 / §3.2 / §1】

## 为什么做

> 一句话导读：业界已反复观察到"同样适配目标任务,on-policy RL 比 SFT 更不伤通用能力",但只有现象、没有机理;本文要回答"SFT 到底坏在哪个 token 上",并据此改 SFT。

- 研究背景：SFT 是把通用 LLM 适配到数学/医疗/Agent 等垂直域的标准做法,但常伴随灾难性遗忘("Alignment Tax",Askell 2021 / Ouyang 2022)。近期对照工作反复观察到 **on-policy RL 适配目标任务时,对通用能力的损害远小于 SFT**,并给出四条机理线索:
  - ① Chu et al. 2025——"SFT memorizes, RL generalizes"(SFT 倾向记忆、牺牲泛化);
  - ② Wang et al. 2025——RL 能从**单个**训练样本受益而不严重过拟合;
  - ③ Mukherjee et al. 2025——RL 只更新更小、更有效的参数子网络;
  - ④ Razin et al. 2023——RL 的参数更新更局部、更有针对性。
  - 这条共识就是 EAFT 的出发台阶。【原文 §1 / §2】
- 解决的具体痛点："为什么 SFT 频繁损害通用能力、而 on-policy RL 能保住?"(原文斜体设问 "Why does SFT frequently degrade general abilities, while on-policy RL preserves them?")——论文要给一个机理级解释,并据此改 SFT。【原文 §1】
- 相关工作 & 各自不足(原文 §2,分两簇):
  - **后训练范式 SFT vs RL**:SFT 最大化 ground-truth 似然(off-policy),RL 基于自生成响应 + 奖励优化(on-policy)。新近研究指出二者的学习行为有根本二分(见"研究背景"四条)。本文的定位是**追根究底——SFT 为何不稳**,论点是"SFT 不加区分地强行拟合 confident conflicts(与模型先验相悖的低熵样本)"。【原文 §2 第一簇】
  - **灾难性遗忘**:经典思路是约束参数变化——EWC(Kirkpatrick 2017)、LwF(Li & Hoiem 2017)、GEM(Lopez-Paz & Ranzato 2017);在 LLM 上就表现为"Alignment Tax"。【原文 §2 第二簇,作远端背景】
  - **基于 token 级动态指标的近邻方法(本文最直接的竞品,逐条点名)**:
    - ① **TALR**(Lin et al. 2025)——按 token 置信度动态缩放**学习率**以加速收敛;
    - ② **DFT**(Wu et al. 2025)——按预测**概率**重加权 SFT loss;
    - ③ **RL's Razor**(Shenfeld et al. 2025)——用 **KL 散度**正则约束模型偏离 base;
    - ④ **FLOW**(Sanyal et al. 2025)——按学习动态**上采样易样本**以平滑优化。
    - **作者明确的 Δ**:"existing dynamic methods predominantly rely on probability or KL divergence as proxies … probability alone is an insufficient statistic"(现有动态方法多用概率或 KL 当代理,而概率单独不是充分统计量)。理由就是前面那句——低概率 token 既可能是 epistemic uncertainty(该学)、又可能是 confident conflict(该抑制),概率分不开二者;**EAFT 用熵当门控信号**正是补这个缺口。
    - 论文还在主表把 DFT/TALR 列为基线,实测它们的**遗忘缓解常差于、甚至不如普通 SFT**(GLM4-9B 上 DFT General -8.4 vs SFT -6.0)。【原文 §2 末 / Table 1】
- 动机链(诊断→归因→方法,一个闭环):现状(SFT 适配必伴遗忘,RL 不会)→ 诊断(系统统计训练数据每个 token 的概率 \(p_t\) 和熵 \(H_t\),发现 SFT 数据比 on-policy rollout 多出一簇"低熵 + 低概率"的 token,Fig.1b)→ 归因(pilot study:单纯把这些 token(熵与概率排名都在最低 15%)的 loss **屏蔽掉**,就几乎消除了遗忘,Fig.2a,原文:"nearly eliminated catastrophic forgetting")→ 结论(遗忘的主因是"在冲突样本上被迫做了大梯度更新",**而非 SFT 流程本身**)→ 于是用连续的熵门控替代硬屏蔽(硬屏蔽会丢数据、还依赖敏感阈值 \(\tau,\delta\))。【原文 §3.2–3.3】
- 与最近邻工作的 Δ：相对 **DFT**(同样是"在 CE 上乘一个 token 级系数"的单行改动)——DFT 乘的是**预测概率**(detach),EAFT 乘的是**归一化熵**。差别在:概率无法区分"低概率是因为不会"还是"低概率是因为相悖",而熵可以。【原文 §2 / Table 1】

## 怎么做 + 靠不靠谱

> 一句话导读：方法就一行——在交叉熵前乘一个"归一化熵"系数(Eq.2)。下面先给这一行的精确形式,再说明熵怎么算(只取 Top-20 近似)、为什么用线性而非别的曲线。

- **方法流水线(可复现级)**：输入是标准 SFT 数据 \(\{(x,y)\}\)。前向得到每个 target token 在词表上的预测分布 \(P_t(v)=P_\theta(v\mid x,y_{<t})\);取 **Top-20** 概率算熵 \(H^{\text{top-20}}_t\);归一化到 \(\tilde H_t\in[0,1]\);用 \(\tilde H_t\) 缩放该 token 的 CE;再反向更新。输出是目标域已适配、但通用能力几乎不退的模型。推理时无任何改动。【原文 Eq.2–3】
- **核心损失的真实形式 + 直觉**：
  - **两个 token 级度量(原文 §3.1)**:概率 \(p_t=P_\theta(y_t\mid x,y_{<t})\)(模型对 ground-truth token 的信心);预测熵 \(H_t=-\sum_{v\in V}P_t(v)\log P_t(v)\)(模型对整个词表的不确定度)。
  - **标准 CE(对照)**:\(\mathcal{L}_{\mathrm{CE}}(\theta)=-\sum_{t=1}^{T}\log P_\theta(y_t\mid x,y_{<t})\)(Eq.1)。它的缺陷是"uniform treatment of all tokens"——不管模型自己怎么想,对每个 token 都一视同仁地猛更新。
  - **EAFT 目标(Eq.2)**:\(\displaystyle \mathcal{L}_{\mathrm{EAFT}}(\theta)=-\sum_{t=1}^{T}\underbrace{\tilde H_t}_{\text{adaptive gating}}\cdot\underbrace{\log P_\theta(y_t\mid x,y_{<t})}_{\text{standard supervision}}.\)就是在 CE 前面挂了个熵闸门 \(\tilde H_t\)。
  - **归一化门控(Eq.3,含 Top-K 近似)**:\(\displaystyle \tilde H_t=\frac{H^{\text{top-}K}_t}{\ln(K)}\;\approx\;\frac{H^{\text{top-20}}_t}{3.0},\qquad K=20,\;\ln 20\approx3.0.\)这里 \(\ln(K)\) 是 K 个结果的最大熵,用作归一化因子。这样就造出一个自调机制:
    - **Conflict Suppression** \((\tilde H_t\to0)\)——模型笃定(低熵)时权重趋 0,相当于把冲突标签的破坏性梯度屏蔽掉;
    - **Knowledge Acquisition** \((\tilde H_t\to1)\)——模型不确定/在探索(高熵)时权重≈1,恢复成标准 SFT 去学新模式。
  - **机制直觉(Fig.3 梯度热图)**:CE 天然对低概率 token 给最大梯度(要把概率从很低拉高,得大幅更新参数);而在低熵处,这种大更新会**改写**承载通用能力的表征(Fig.3 左下角深紫=强优化压力)。乘上 \(\tilde H_t\) 后:低熵→系数≈0→这股大梯度被压灭(Fig.3 右图同区域变浅黄);高熵→系数≈1→正常学。一句话:"按模型自身的不确定度,决定它该不该被标签'掰过来'"。【原文 §3.2 Theoretical Insight / §4.3 / Fig.3–4】
- **门控函数形式的鲁棒性(§5.1,关键消融,目的是把结论从"魔数"里解放出来)**:把门控泛化成一个函数 \(f(\tilde H_t)\),测了四类变体:
  - 线性(默认):\(f(\tilde H_t)=\tilde H_t\);
  - 多项式:\(f(\tilde H_t)=(\tilde H_t)^p,\;p\in\{2,3\}\)(记作 EAFT2 / EAFT3,更狠地压笃定样本);
  - Sigmoid:\(f(\tilde H_t)=\sigma\big(\alpha(\tilde H_t-\beta)\big)\),实验取 \(\beta=0.17\)(对齐"最低 15% 熵"分位)、\(\alpha=30\)(记作 EAFTsig);
  - 硬屏蔽(Masked SFT,基线):\(f(\tilde H_t)=\mathbb{I}(\tilde H_t>\tau_{0.15})\),即把最低 15% 熵 token 的 loss 直接置 0(Eq.4)。
  - **两条核心发现**:
    - ① **Universality of Entropy Awareness(熵感知的普适性)**——所有"熵感知"变体(EAFT/2/3/sig)都比 SFT 保住更多通用能力,**说明增益来自"看熵"这个机制本身、和具体函数形式无关**(强消融,结论不依赖某个魔数);
    - ② **The Necessity of Soft Gating(软门控的必要性)**——硬屏蔽虽然防遗忘,但目标 Math Score 掉到 **65.60**(EAFT 是 **69.27**),说明"自信冲突"里其实**也含有必要的适配信号**,一刀切会连有用的一起丢;软门控只降权、不删数据,从而占住 Pareto 前沿(Fig.5)。
    - 原文还在附录 G.1 论证为何首选**线性**:Sigmoid 的 \(\alpha,\beta\) 太敏感(\(\beta\) 太高就退化成 SFT、太低就变成 Hard Mask,得对每个数据集/规模做昂贵 grid search),而线性是"parameter-free"的结构先验(loss 权重 ∝ 不确定度),开箱即用。【原文 §5.1 / Fig.5 / 附录 G.1】
- **Top-K 近似为什么够用(§5.2)**:目的是省下"对全词表(\(|V|>100\)k)算熵"的开销。Fig.6 给出定量结论:K 增大时,Top-K 熵与精确熵的 Pearson 相关迅速饱和,**K=20 时相关已达 0.999**,额外显存 **<0.4 KB**(只多一次 sort + log-sum-exp)——近似几乎无损。底层依据是 LLM 的概率质量高度稀疏、集中在头部少数 token。【原文 §5.2 / Fig.6】
- 实验与证据：
  - **数据**:prompt 取自 NuminaMath / BigMathVerified / Nemotron-CrossThink,用 **Qwen3-235B-A22B-Instruct-2507** 合成回答,随机选 **19k 条被验证正确**的样本作数学训练集;医疗用 Huatuo-O1,Agent 用 Nemotron-Agentic-Tool-Use-v1 的非思考子集随机采 **20k** 轨迹。【原文 §4.1 / 附录 B】
  - **模型**:Qwen3-4B-Instruct-2507、Qwen2.5-32B-Instruct、GLM4-9B-0414(覆盖 4B–32B、跨族跨规模);医疗用 Qwen3-4B-Thinking-2507。【原文 §4.1 / 附录 D】
  - **关键数字(Table 1,Qwen3-4B-Instruct 数学域)**:
    - SFT 把 Math Avg 从 68.3→69.4,但 General Avg 从 81.1→**76.5(-4.6)**;
    - EAFT 的 Math Avg 69.3(与 SFT 几乎并列),General Avg **80.1(仅 -1.0)**;其中 CLUEWSC 这一项:SFT 85.2→74.5(掉 10.7),EAFT 只掉到 83.7。
    - 跨规模:32B 上 SFT -3.2 vs EAFT -1.1;GLM4-9B 上 SFT -6.0 vs EAFT -3.9。
    - 跨域:医疗(Table 2)SFT -4.8 vs EAFT -1.6(且 EAFT 目标分 73.7 还略超 SFT 73.6,见 BFCL/MedMCQA 等);Agent(Table 3,BFCL v3)SFT -6.3 vs EAFT -3.6。【原文 Table 1/2/3】
  - baseline 公平吗:基线(SFT/SFTKL/FLOW/DFT/TALR)同数据同评测,且**三次独立 run 取均值**——较规范;SFTKL 的 KL 系数 \(\beta=0.5\)。值得注意的是 DFT/TALR 在多处的遗忘缓解效果还差于 SFT 本身,凸显"概率门控未必防遗忘"。【原文 Table 1 / 附录 F.2】
  - 有无"看着强但没回答核心问题":核心问题是"减遗忘",主要战果都体现在 General Avg 的下降幅度上(目标分基本与 SFT 持平,个别略低,如 Qwen3-4B 上 AIME24 EAFT 60.0 < SFT 63.3)。它**不主张**超越 SFT 的目标性能,定位就是 Pareto 改进。【原文 §7 Target Performance Trade-off】
- **关键超参默认值(附录 F,Tab.4,可直接照搬)**:lr **1e-5**、cosine scheduler、warmup ratio **0.03**、AdamW、batch size **64**、**10 epochs**、max seq len **16384**;框架用 LLaMA-Factory(SFTKL 例外,用 veRL);8×A100;checkpoint 选"全 benchmark 平均最优"那个。【原文 附录 F.1–F.2】
- 假设与失效边界：
  - 【原文 §7】明确给了三条:
    - ① **反事实/知识编辑场景**(比如教模型"天是绿的"、纠正过时事实)——此时恰恰需要覆盖先验,EAFT 反而会把这些必要更新当成冲突压掉,**适得其反**(原文:"the model's resistance to Confident Conflicts is undesirable");
    - ② EAFT **不以超越 SFT 目标分为目标**,如果纯追目标性能,SFT 仍可能更优;
    - ③ **依赖 base 模型的校准**——若 base"自信地错"(即幻觉),EAFT 会**保护这些错误先验**而不去纠(未来工作:引入不确定性校准来区分"真知识 vs 自信幻觉")。
  - 【推断】门控只看模型**自身**的熵,并不直接判断"标签是否真和模型相悖":那些低熵、但标签恰好一致的 token 也被同等降权,理论上可能轻微拖慢一部分正确知识的快速拟合(论文用 Fig.4"高熵 token 的 loss 下降与 SFT 同速"来间接回应,但没单独分析"低熵且一致"的 token)。依据:Eq.2 的门控只是 \(\tilde H_t\) 的函数,与 ground-truth 是否冲突无关。
- 祛魅总结【推断】：
  - 真贡献:把"SFT 伤通用能力"的归因,从模糊的"过拟合"收敛到一个**可操作、可视化、可消融**的具体对象(低熵 + 低概率 token);诊断(Fig.1/2)→机制(Fig.3/4)→方法(Eq.2)→鲁棒性(Fig.5/6)整条闭环完整;§5.1 的"所有熵感知变体都胜 SFT"是相当强的"机制非魔数"证据。
  - 包装/可能被高估:"用熵当门控"本质上是**置信度加权的另一种形式**——和 DFT 的"概率加权"同属一族(都是 forward-hard / backward-soft 式的"按某个统计量重加权 CE"),区别只在它选了"熵"这个更能分离冲突的统计量;所谓"自信冲突",很大程度上就是给"模型先验与标签强不一致的 token"这一已知现象起了个名字。它解决的是"适配时少忘",**并不**让 SFT 学得更多/更好(目标分不升)。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：每个 target token 的**预测熵**(Top-20 近似全词表熵)\(\tilde H_t\),作为"该不该被标签掰过来"的门控;不学奖励、不学教师 logits。
  - **改什么**：只改**损失函数**(CE 前乘 \(\tilde H_t\));模型/数据/优化器/流程与全参 SFT 完全一致。
  - **何时改**：训练时(每步前向算熵→当步缩放 loss);推理时无改动。
  - **免梯度?**：是 SFT 范式、无 RL rollout、无策略梯度。〔待核:门控系数 \(\tilde H_t\) 是否对其自身 stop-gradient(detach)。原文 Eq.2/Eq.3 与正文、README 的公式 \(\mathcal{L}_{EAFT}=-\sum\tilde H_t\log P_\theta\) 均**未显式写 detach**;从"熵由当前 \(P_\theta\) 算出、且是 loss 的乘性系数"看,若不 detach 则梯度会经熵项回流(等价于额外鼓励/抑制熵)。同族 DFT 明确 detach 概率系数;EAFT 论文未明示,需以集成实现(LlamaFactory-EAFT 的 `use_eaft_loss` 损失代码)为准——**本地 `resource/repos/eaft/` 下两个 submodule(LlamaFactory-EAFT、ms-swift-EAFT)仅有 `.gitmodules` 指针、目录为空,损失实现无法本地审计**,需到 fork 查看。〕
  - **记忆-技能生命周期**：属"**巩固时防遗忘**"——把新技能(目标域)写进参数的同时,靠抑制低熵冲突梯度来**保护旧表征**;无显式外部记忆/技能库,纯参数内做。
  - **防遗忘机制**：核心即此——熵门控对"低熵冲突"token 近零梯度(Fig.3)、对低熵 token loss 全程保持稳定不被推向 0(Fig.4,SFT 把它驱向 0 = 强迫记忆),从而不改写通用表征。与 KL 正则(约束输出分布)不同,它是**逐 token 门控梯度尺度**。
- ⑦ 开源代码+框架/harness：主仓库 https://github.com/PRIS-CV/EAFT (本地已克隆但**仅 README + assets**,无独立训练代码);项目页 ymxyll.github.io/EAFT/。**实现集成**进 LLaMA-Factory(官方合入,`use_eaft_loss` 启用;作者分支 github.com/ymxyll/LlamaFactory-EAFT)与 ms-swift(github.com/ymxyll/ms-swift-EAFT,支持 megatron/deepspeed)。框架=标准全参/SFT(LLaMA-Factory;SFTKL 基线用 veRL),仅替换损失,无 RL、无额外模型。【代码可得性中等:主仓无核心代码;**本地核查 `.gitmodules` 确认两个 submodule 目录为空(指针未拉取),损失实现无法本地审计——此为"未完整克隆 + 原因",链接已附**供手动获取。】
- 💰 资源/成本与可扩展性：训练成本 ≈ 标准 SFT(只多算一次 Top-20 熵);§5.2 实测 Top-20 额外显存 <0.4 KB、几乎无额外开销。规模验证到 32B、跨 Qwen3/Qwen2.5/GLM4。数据量小(数学 19k、Agent 20k)。明确对比 RL/RFT(附录 G.2):RL 要同时维护 Policy/Reference/Value 三模型、约 3× 显存 + rollout 阶段,EAFT 保持标准 SFT 的显存与计算图、无 reference model、无 rollout。【原文 §5.2 / 附录 G.2】
- 🎯 对"探索-巩固"对标：**强支撑(巩固/防遗忘侧)+ 弱竞品(探索侧无)**。判定依据:本课题的"巩固"= 把有效行为固化进参数且不遗忘——EAFT 正好是"巩固时不遗忘"的一个干净机制;而且它"熵=探索信号"的用法,与本课题"高熵=分支/选路点、低熵=笃定"的直觉**高度同构**。
  - 这点在附录 A 的词云里很直观,三类 token 各对应一类语言成分:
    - **高熵 token = 抽象动词 / 推理连接词 / 语义分支点**(如 vary/depends/reconstruct);
    - **低熵高概率 = 数学符号 / LaTeX 语法 / 固定模式**(如 \frac/\sin);
    - **自信冲突(低熵低概率)= 特定实体 / 长尾术语 / 噪声**(如 Jayden/modpacks)。
  - **可借组件**:用归一化 Top-K 熵做逐 token 门控,可直接迁到 MTP/OPD 的"巩固"损失上——对"模型已笃定走通的开头/路径"降权、对"高熵需探索处"保权,从而"固化新技能而不冲掉旧能力";而且 §5.1 已证明门控函数形式不敏感,迁移时不必精调。
  - **缺口**:
    - ① EAFT 是离线 SFT,没结合 on-policy / student 自选(而本课题要 on-policy 自选恢复分支);
    - ② 无"回轨 / path-recovery"探索机制,它只防遗忘、不主动选路;
    - ③ 门控只看自身熵、不看"是否真冲突",与"teacher 当稀疏脚手架"没有对接。
- 🔭 开放问题/未来方向：
  - 【原文 §7】结合**不确定性校准**以区分"真知识 vs 自信幻觉"(否则会保护错误先验);明确 EAFT 不适用于知识编辑/反事实训练。
  - 【推断】把熵门控嵌入 on-policy 蒸馏/RL 的巩固阶段(而非纯离线 SFT),让"探索(高熵处学)+ 巩固(低熵冲突处稳)"在同一回路里成立;以及对"低熵且标签一致"token 是否被过度降权做专门消融。

RETURN: eaft | 读PDF=是(全文 + 附录A词云/B数据/F超参Tab.4/G.1线性vs Sigmoid/G.2 vs RL,核 Eq.1–3/Table1–3/Fig.3–6,并本地核 .gitmodules 确认 submodule 空) | 加厚=是(新增Eq.1–3 MathJax+四类门控函数公式Eq.4+Top-K公式;related work逐条点名TALR/DFT/RL's Razor/FLOW与四条RL机理线索;补附录F全套超参与G.1/G.2) | LaTeX公式=8条(Eq.1、Eq.2、Eq.3、Eq.4硬屏蔽、Sigmoid/多项式门控、熵定义、内联 \(\tilde H_t\) 等) | 待核=1(门控系数是否detach,集成fork本地submodule为空、无法审计)

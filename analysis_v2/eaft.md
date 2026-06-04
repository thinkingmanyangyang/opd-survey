eaft | Entropy-Adaptive Fine-Tuning: Resolving Confident Conflicts to Mitigate Forgetting | 北京邮电大学 PRIS-CV / 中关村学院（Muxi Diao、Lele Yang、Wuxuan Gong 共一；通讯 Zhanyu Ma） | 2026-01-05 · arXiv 预印本(曾居 HF Daily Paper #1) · v1 | 主题线 L5(记忆/持续学习·防遗忘)，兼 L2(SFT 改造) · 相关性 高

**原始论文**：https://arxiv.org/abs/2601.02151

## 一眼看懂
- 🟦 TL;DR：SFT 之所以损害通用能力,主因不是"训练流程"本身,而是一类特定 token——**模型自己很笃定(低熵)、标签却逼它输出另一个 token(低概率)**,论文叫它"Confident Conflict(自信冲突)"。EAFT 用**每个 token 的归一化熵 \(\tilde H_t\)** 当门控系数去缩放交叉熵损失:模型越笃定(熵越低)梯度越被压到接近 0,越不确定(熵越高)越退化为普通 SFT。只换一个损失函数,不引入 RL、不加额外模型,就把遗忘缩到约 SFT 的 1/4~1/5,同时目标任务不掉分。【原文 §Abstract / §3.3 / Eq.2–3】
- 最巧的一步：**把门控信号从"概率"换成"熵"**。抽掉这一步(退回到看概率,即 DFT / FLOW / TALR 那套)方法就垮——因为低概率 token 有两种来源:(a) 模型真不会、该学(epistemic uncertainty,此时熵也高);(b) 模型很有把握想输出别的、标签相悖(Confident Conflict,此时熵低)。只看概率会把这两类一视同仁,甚至在 (b) 上**放大**破坏性梯度、加速遗忘(原文:"prior methods risk accelerating forgetting");熵恰好能把"该学(高熵)"和"该抑制(低熵)"分开。【原文 §2 末 / §3.2 / §1】

## 为什么做
- 研究背景：SFT 是把通用 LLM 适配到数学/医疗/Agent 垂直域的标准做法,但常伴随灾难性遗忘("Alignment Tax",Askell 2021 / Ouyang 2022);近期对照工作反复观察到 **on-policy RL 适配目标任务时对通用能力损害远小于 SFT**——并给出三条机理线索:① Chu et al. 2025"SFT memorizes, RL generalizes"(SFT 倾向记忆、牺牲泛化);② Wang et al. 2025(RL 能从**单个**训练样本受益而不严重过拟合);③ Mukherjee et al. 2025(RL 只更新更小、更有效的参数子网络);④ Razin et al. 2023(RL 参数更新更局部、更有针对性)。这条共识是 EAFT 的出发台阶。【原文 §1 / §2】
- 解决的具体痛点："为什么 SFT 频繁损害通用能力、而 on-policy RL 能保住?"(原文斜体设问 "Why does SFT frequently degrade general abilities, while on-policy RL preserves them?")——论文要给机理级解释并据此改 SFT。【原文 §1】
- 相关工作 & 各自不足(原文 §2,分两簇)：
  - **后训练范式 SFT vs RL**：SFT 最大化 ground-truth 似然(off-policy),RL 基于自生成响应 + 奖励优化(on-policy)。新近研究指出二者学习行为有根本二分(见"研究背景"四条)。本文定位:**追根究底——SFT 为何不稳**,论点是"SFT 不加区分地强行拟合 confident conflicts(与模型先验相悖的低熵样本)"。【原文 §2 第一簇】
  - **灾难性遗忘**：经典思路约束参数变化——EWC(Kirkpatrick 2017)、LwF(Li & Hoiem 2017)、GEM(Lopez-Paz & Ranzato 2017);在 LLM 上表现为"Alignment Tax"。【原文 §2 第二簇,作远端背景】
  - **基于 token 级动态指标的近邻方法(本文最直接的竞品,逐条点名)**：① **TALR**(Lin et al. 2025)——按 token 置信度动态缩放**学习率**以加速收敛;② **DFT**(Wu et al. 2025)——按预测**概率**重加权 SFT loss;③ **RL's Razor**(Shenfeld et al. 2025)——用 **KL 散度**正则约束模型偏离 base;④ **FLOW**(Sanyal et al. 2025)——按学习动态**上采样易样本**以平滑优化。**作者明确的 Δ**:"existing dynamic methods predominantly rely on probability or KL divergence as proxies … probability alone is an insufficient statistic"——低概率 token 既可能是 epistemic uncertainty(该学)又可能是 confident conflict(该抑制),概率分不开二者;**EAFT 用熵作门控信号**正是为补这个缺口。论文还在主表把 DFT/TALR 列为基线,实测它们**遗忘缓解常差于甚至不如普通 SFT**(GLM4-9B 上 DFT General -8.4 vs SFT -6.0)。【原文 §2 末 / Table 1】
- 动机链(诊断→归因→方法,闭环)：现状(SFT 适配必伴遗忘,RL 不会)→ 诊断(系统统计训练数据逐 token 概率 \(p_t\)×熵 \(H_t\),发现 SFT 数据比 on-policy rollout 多出一簇"低熵 + 低概率"token,Fig.1b)→ 归因(pilot study:单纯**屏蔽**这些 token(熵与概率排名均在最低 15%)的 loss 就几乎消除遗忘,Fig.2a,原文:"nearly eliminated catastrophic forgetting")→ 结论(遗忘主因是"在冲突样本上被迫做大梯度更新",**而非 SFT 流程本身**)→ 用连续熵门控替代硬屏蔽(后者丢数据 + 依赖敏感阈值 \(\tau,\delta\))。【原文 §3.2–3.3】
- 与最近邻工作的 Δ：相对 **DFT**(同为"乘一个 token 级系数到 CE 上"的单行改动)——DFT 乘的是**预测概率**(detach),EAFT 乘的是**归一化熵**。差在:概率无法区分"低概率因为不会"vs"低概率因为相悖",熵可以。【原文 §2 / Table 1】

## 怎么做 + 靠不靠谱
- **方法流水线(可复现级)**：输入(标准 SFT 数据 \(\{(x,y)\}\)) → 前向得每个 target token 在词表上的预测分布 \(P_t(v)=P_\theta(v\mid x,y_{<t})\) → 取 **Top-20** 概率算熵 \(H^{\text{top-20}}_t\) → 归一化 \(\tilde H_t\in[0,1]\) → 用 \(\tilde H_t\) 缩放该 token 的 CE → 反向更新。输出:目标域适配、但通用能力几乎不退的模型。推理时无改动。【原文 Eq.2–3】
- **核心损失的真实形式 + 直觉**：
  - 两个 token 级度量(原文 §3.1)：概率 \(p_t=P_\theta(y_t\mid x,y_{<t})\)(模型对 ground-truth token 的信心);预测熵 \(H_t=-\sum_{v\in V}P_t(v)\log P_t(v)\)(模型对全词表的不确定度)。
  - 标准 CE(对照):\(\mathcal{L}_{\mathrm{CE}}(\theta)=-\sum_{t=1}^{T}\log P_\theta(y_t\mid x,y_{<t})\)(Eq.1),其缺陷是"uniform treatment of all tokens"——不论模型先验,对每个 token 一视同仁地猛更新。
  - **EAFT 目标(Eq.2)**：\[\mathcal{L}_{\mathrm{EAFT}}(\theta)=-\sum_{t=1}^{T}\underbrace{\tilde H_t}_{\text{adaptive gating}}\cdot\underbrace{\log P_\theta(y_t\mid x,y_{<t})}_{\text{standard supervision}}.\]
  - **归一化门控(Eq.3,含 Top-K 近似)**：\[\tilde H_t=\frac{H^{\text{top-}K}_t}{\ln(K)}\;\approx\;\frac{H^{\text{top-20}}_t}{3.0},\qquad K=20,\;\ln 20\approx3.0.\]其中 \(\ln(K)\) 是 K 个结果的最大熵(归一化因子)。这造出自调机制:**Conflict Suppression** \((\tilde H_t\to0)\)——模型笃定(低熵)时权重趋 0,等效屏蔽冲突标签的破坏性梯度;**Knowledge Acquisition** \((\tilde H_t\to1)\)——模型不确定/探索(高熵)时权重≈1,恢复标准 SFT 学新模式。
  - **机制直觉(Fig.3 梯度热图)**：CE 对低概率 token 天然给最大梯度(把概率从很低拉高需大幅更新参数);在低熵处这种大更新会**改写**承载通用能力的表征(Fig.3 左下角深紫=强优化压力)。乘上 \(\tilde H_t\) 后,低熵→系数≈0→把这股大梯度压灭(Fig.3 右图同区域变浅黄);高熵→系数≈1→正常学。一句话:"按模型自身的不确定度决定该不该被标签'掰过来'"。【原文 §3.2 Theoretical Insight / §4.3 / Fig.3–4】
- **门控函数形式的鲁棒性(§5.1,关键消融,把结论从"魔数"解放)**：把门控泛化为 \(f(\tilde H_t)\),测四类变体:
  - 线性(默认):\(f(\tilde H_t)=\tilde H_t\);
  - 多项式:\(f(\tilde H_t)=(\tilde H_t)^p,\;p\in\{2,3\}\)(记 EAFT2 / EAFT3,更狠地压笃定样本);
  - Sigmoid:\(f(\tilde H_t)=\sigma\big(\alpha(\tilde H_t-\beta)\big)\),实验取 \(\beta=0.17\)(对齐"最低 15% 熵"分位)、\(\alpha=30\)(记 EAFTsig);
  - 硬屏蔽(Masked SFT,基线):\(f(\tilde H_t)=\mathbb{I}(\tilde H_t>\tau_{0.15})\),即把最低 15% 熵 token 的 loss 直接置 0(Eq.4)。
  - **两条核心发现**:① **Universality of Entropy Awareness**——所有"熵感知"变体(EAFT/2/3/sig)都比 SFT 保住更多通用能力,**说明增益来自"看熵"这一机制本身、与具体函数形式无关**(强消融,结论不依赖某个魔数);② **The Necessity of Soft Gating**——硬屏蔽虽防遗忘,但目标 Math Score 掉到 **65.60**(EAFT **69.27**),说明"自信冲突"里其实**也含必要的适配信号**,一刀切会连有用的也丢;软门控只降权不删,占住 Pareto 前沿(Fig.5)。原文还在附录 G.1 论证为何首选**线性**:Sigmoid 的 \(\alpha,\beta\) 敏感(\(\beta\) 太高退化成 SFT、太低变成 Hard Mask,需对每个数据集/规模做昂贵 grid search),而线性是"parameter-free"的结构先验(loss 权重 ∝ 不确定度),开箱即用。【原文 §5.1 / Fig.5 / 附录 G.1】
- **Top-K 近似的合理性(§5.2)**：为省全词表(\(|V|>100\)k)算熵的开销。Fig.6 定量:K 增大时 Top-K 与精确熵的 Pearson 相关迅速饱和,**K=20 时相关 0.999**,额外显存 **<0.4 KB**(只多一次 sort + log-sum-exp)——近似几乎无损。依据是 LLM 概率质量高度稀疏、集中在头部 token。【原文 §5.2 / Fig.6】
- 实验与证据：
  - **数据**：prompt 取自 NuminaMath / BigMathVerified / Nemotron-CrossThink,用 **Qwen3-235B-A22B-Instruct-2507** 合成回答,随机选 **19k 条被验证正确**的样本作数学训练集;医疗用 Huatuo-O1,Agent 用 Nemotron-Agentic-Tool-Use-v1 非思考子集随机采 **20k** 轨迹。【原文 §4.1 / 附录 B】
  - **模型**：Qwen3-4B-Instruct-2507、Qwen2.5-32B-Instruct、GLM4-9B-0414(4B–32B 跨族跨规模);医疗用 Qwen3-4B-Thinking-2507。【原文 §4.1 / 附录 D】
  - **关键数字(Table 1,Qwen3-4B-Instruct 数学域)**：SFT 把 Math Avg 68.3→69.4,但 General Avg 81.1→**76.5(-4.6)**;EAFT Math Avg 69.3(与 SFT 几乎并列)、General Avg **80.1(仅 -1.0)**;其中 CLUEWSC:SFT 85.2→74.5(-10.7),EAFT 仅到 83.7。32B 上 SFT -3.2 vs EAFT -1.1;GLM4-9B 上 SFT -6.0 vs EAFT -3.9。跨域:医疗(Table 2) SFT -4.8 vs EAFT -1.6(且 EAFT 目标 73.7 略超 SFT 73.6,BFCL/MedMCQA 等);Agent(Table 3,BFCL v3) SFT -6.3 vs EAFT -3.6。【原文 Table 1/2/3】
  - baseline 公平吗:基线(SFT/SFTKL/FLOW/DFT/TALR)同数据同评测,**三次独立 run 取均值**——较规范;SFTKL 的 KL 系数 \(\beta=0.5\)。值得注意 DFT/TALR 在多处遗忘缓解效果差于 SFT 本身,凸显"概率门控未必防遗忘"。【原文 Table 1 / 附录 F.2】
  - 有无"看着强但没回答核心问题":核心问题是"减遗忘",主战果都在 General Avg 的下降幅度上(目标分基本与 SFT 持平、个别略低如 Qwen3-4B AIME24 EAFT 60.0 < SFT 63.3)。它**不主张**超越 SFT 的目标性能,定位是 Pareto 改进。【原文 §7 Target Performance Trade-off】
- **关键超参默认值(附录 F,Tab.4,可直接照搬)**：lr **1e-5**、cosine scheduler、warmup ratio **0.03**、AdamW、batch size **64**、**10 epochs**、max seq len **16384**;框架 LLaMA-Factory(SFTKL 例外用 veRL);8×A100;checkpoint 选"全 benchmark 平均最优"。【原文 附录 F.1–F.2】
- 假设与失效边界：
  - 【原文 §7】明确三条:① **反事实/知识编辑场景**(教模型"天是绿的"、纠正过时事实)——此时正需覆盖先验,EAFT 会把这些必要更新当冲突压掉,**适得其反**(原文:"the model's resistance to Confident Conflicts is undesirable");② EAFT **不以超越 SFT 目标分为目标**,纯追目标性能时 SFT 仍可能更优;③ **依赖 base 模型校准**——若 base"自信地错"(幻觉),EAFT 会**保护这些错误**先验,不去纠(未来工作:引入不确定性校准区分"真知识 vs 自信幻觉")。
  - 【推断】门控只看模型**自身**的熵,并不直接判断"标签是否真与模型相悖":低熵但标签恰好一致的 token 也被同等降权,理论上可能轻微拖慢一部分正确知识的快速拟合(论文用 Fig.4"高熵 token loss 下降与 SFT 同速"间接回应,但未对"低熵且一致"的 token 单列分析)。依据:Eq.2 门控仅是 \(\tilde H_t\) 的函数,与 ground-truth 是否冲突无关。
- 祛魅总结【推断】：
  - 真贡献:把"SFT 伤通用能力"的归因从模糊的"过拟合"收敛到一个**可操作、可视化、可消融**的具体对象(低熵 + 低概率 token),诊断(Fig.1/2)→机制(Fig.3/4)→方法(Eq.2)→鲁棒性(Fig.5/6)闭环完整;§5.1 的"所有熵感知变体都胜 SFT"是相当强的"机制非魔数"证据。
  - 包装/可能被高估:"用熵当门控"本质是**置信度加权的另一种形式**——与 DFT 的"概率加权"同属一族(forward-hard / backward-soft 式的"按某统计量重加权 CE"),区别在选了"熵"这个更能分离冲突的统计量;所谓"自信冲突"很大程度等价于"模型先验与标签强不一致的 token"这一已知现象的命名化。它解决的是"适配时少忘",**并未**让 SFT 学得更多/更好(目标分不升)。

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
- 🎯 对"探索-巩固"对标：**强支撑(巩固/防遗忘侧)+ 弱竞品(探索侧无)**。判定依据:本课题的"巩固"= 把有效行为固化进参数且不遗忘——EAFT 正是"巩固时不遗忘"的一个干净机制;且它对"熵=探索信号"的用法与本课题"高熵=分支/选路点、低熵=笃定"的直觉**高度同构**(附录 A 词云:**高熵 token = 抽象动词/推理连接词/语义分支点**(vary/depends/reconstruct);**低熵高概率 = 数学符号/LaTeX 语法/固定模式**(\frac/\sin);**自信冲突(低熵低概率)= 特定实体/长尾术语/噪声**(Jayden/modpacks))。**可借组件**:用归一化 Top-K 熵做逐 token 门控,可直接迁移到 MTP/OPD 的"巩固"损失上,对"模型已笃定走通的开头/路径"降权、对"高熵需探索处"保权,实现"固化新技能而不冲掉旧能力";且 §5.1 证明门控函数形式不敏感,迁移时不必精调。**缺口**:① EAFT 是离线 SFT,未结合 on-policy / student 自选(本课题要 on-policy 自选恢复分支);② 无"回轨 / path-recovery"探索机制,只防遗忘不主动选路;③ 门控只看自身熵、不看"是否真冲突",与"teacher 当稀疏脚手架"无对接。
- 🔭 开放问题/未来方向：
  - 【原文 §7】结合**不确定性校准**以区分"真知识 vs 自信幻觉"(否则会保护错误先验);明确 EAFT 不适用于知识编辑/反事实训练。
  - 【推断】把熵门控嵌入 on-policy 蒸馏/RL 的巩固阶段(而非纯离线 SFT),让"探索(高熵处学)+ 巩固(低熵冲突处稳)"在同一回路里成立;以及对"低熵且标签一致"token 是否被过度降权做专门消融。

RETURN: eaft | 读PDF=是(全文 + 附录A词云/B数据/F超参Tab.4/G.1线性vs Sigmoid/G.2 vs RL,核 Eq.1–3/Table1–3/Fig.3–6,并本地核 .gitmodules 确认 submodule 空) | 加厚=是(新增Eq.1–3 MathJax+四类门控函数公式Eq.4+Top-K公式;related work逐条点名TALR/DFT/RL's Razor/FLOW与四条RL机理线索;补附录F全套超参与G.1/G.2) | LaTeX公式=8条(Eq.1、Eq.2、Eq.3、Eq.4硬屏蔽、Sigmoid/多项式门控、熵定义、内联 \(\tilde H_t\) 等) | 待核=1(门控系数是否detach,集成fork本地submodule为空、无法审计)

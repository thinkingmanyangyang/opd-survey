segment_attrib | Segment-Level Attribution for Selective Learning of Long Reasoning Traces | University of Southern California(Siyuan Wang、Yanchen Liu、Xiang Ren) | 2026-01-31 arXiv 2602.00425(v1)·ICLR 2026(PDF 页眉标注) | 主题线 L6(CoT/token 信用分配)+L2(选择性 SFT)·相关性 中-高

**原始论文**:https://arxiv.org/abs/2602.00425

## 一眼看懂
- 🟦 TL;DR:长 CoT 里只有一小部分 token 真正帮到答案,大量是重复/截断/废话;直接在全 CoT 上 SFT 会让模型连这些冗余一起学坏。本文用 **integrated gradients(IG,积分梯度)** 量化每个 token 对"预测正确答案"的贡献,聚合成段落级两指标——**强度**(贡献大小)+ **方向一致性**(段内 IG 是否同号)——挑出"高强度 + 中等一致性"的**反思性段落**做 loss-mask 选择性 SFT(只在重要段算交叉熵、其余 mask)。相对 full-CoT SFT,greedy 下准确率最高 +4.7%、长度最多 −18%。
- 最巧的一步:**"中等一致性"这个判据**(而非只看强度)。直觉:段内 token IG **几乎全正或全负(极高一致性)= 浅层/已定方向**(冗余澄清或严重错误探索);**正负混合(中等一致性)= 同时有支持与纠正的反思性推理**,更有学习价值。抽掉一致性维度(消融"只取高强度段",Table 2)→ Acc 仍 46.9 但长度 14715(满配 13506),即一致性主要贡献了**额外的 token 削减而不掉精度**。强度+一致性联合才是完整判据。【原文】§2.3、Fig.2、Table 2

## 为什么做
- 研究背景:大推理模型(o1/R1/Qwen)靠 test-time scaling 生成长 CoT,也成了 cold-start SFT 的监督资源(s1/Muennighoff 2025)。但长 CoT 常上千 token,只有一小部分真贡献,大量是冗余重复/截断(Fig.1 left);冗长无实质反而随长度增长损害推理,且 incorrect CoT 通常比 correct CoT 段更多/token 更多(Fig.1 right-top)。在这种监督上训会让模型学坏冗余行为。【原文】§1
- 解决的具体痛点:已有"识别长链重要部分"的方法各有缺陷——(1) token 级分析(TokenSkip)忽略语义完整性,不构成可解释推理单元;(2) 段落级 **perplexity**(Cui 2025a)/ **entropy**(Li 2025b)是**与真实重要性不完全一致的间接指标**,既有假阳(过度强调"让我们一步步算"这类脚手架桥接文本——删它破坏连贯但贡献甚微),也有假阴(漏掉独立验证/中间结论这类低熵、删除不影响流畅但显著提升正确答案概率的段)。【原文】§1
- 相关工作 & 各自不足(并行路线 + 各自具体短板 + 站谁肩上 + 与最近邻精确差异):
  - **(路线 A)基于剪枝的 CoT 压缩监督**(删冗余得更简洁监督):**短板**【原文 §2.3】——剪枝**往往掉精度**(本文 Table 2 实测 Pruned CoT SFT 43.9 < Full CoT SFT 44.8),因为物理删除破坏轨迹连贯。本文不删、改用 loss-mask 保连贯。
  - **(路线 B)token 级重要性**(TokenSkip 等):**短板**【原文 §1】——忽略语义完整性,单 token 不构成可解释推理单元;本文 Table 2 也证段级(46.9)> 高绝对 IG token(46.1)> 原始 IG token(45.2)。
  - **(路线 C,最近邻)段级间接代理:perplexity / entropy**:用语言不确定性间接代理"重要性"。**短板**【原文 §1、§3.1】——既有假阳(脚手架桥接文本 perplexity 高但贡献低)又有假阴(独立验证段低熵但显著提升答案概率);§3.1 实证**重要段的 perplexity/entropy 反而更低**,直接反打 entropy 度量的脸。本文差异:用 IG **直接归因到"对正确答案的影响"**而非语言不确定性间接代理。
  - **(路线 D,框架来源/站其肩上)selective SFT**:**Rho-1**(Lin 2024)提供"只在部分 token 算 loss、其余 mask"的训练框架——本文**直接沿用**这个框架(Eq.9),新意不在框架而在**段重要性度量**(IG 归因 + strength/consistency 两指标)。段切分则沿用 Lu 2025 的转折关键词法。
  - **与最近邻精确差异**:vs perplexity/entropy 度量——IG 直接归因(且 §3.1 证"低熵≠不重要");vs token 级 IG——以**可解释的段**为单位保语义完整;vs Rho-1——继承其 selective-SFT 框架,差在"IG 归因 + 强度/一致性双指标 + 用一致性区分反思性 vs 浅层"。
- 动机链:长 CoT 大量冗余 → 在其上全量 SFT 学坏冗余 → 需识别真正重要的段 → 但 perplexity/entropy 是间接指标有假阳假阴 → 用 IG 直接度量"段对正确答案预测的影响"(能捕捉直接+间接贡献)+ 用"中等一致性"操作化"反思性推理" → 只在重要段做选择性 SFT。核心实证:**30~40% 的段累计贡献 >80% 的总归因**(correct/incorrect CoT 皆然,Fig.1 CDF)。【原文】§1、§3

## 怎么做(到可复现粒度)
- **方法流水线(三阶段:切段 → IG 归因聚合选段 → 选择性 SFT)**。
- **第 1 步:切段**【原文 §2.2】:按 "\n\nWait"、"\n\nAlternatively" 等**转折关键词**把 CoT 切成 \(\{S_1,\dots,S_M\}\)(完整关键词表见附录 C.2)。每段是语义完整的推理单元。
- **第 2 步:IG token 归因(Eq.1-2)**【原文 §2.2】:给模型 \(F\)、输入 token embedding \(x\)、baseline \(x'\)(取 padding token 的 embedding),第 \(i\) 维的积分梯度:
  \(\displaystyle \mathrm{IG}_i(x)=(x_i-x'_i)\times\int_{\alpha=0}^{1}\frac{\partial F\big(x'+\alpha\cdot(x-x')\big)}{\partial x_i}\,d\alpha\)
  实际用 \(J\) 步插值近似(\(J\!=\!50\)):
  \(\displaystyle \mathrm{IG}_i(x)\approx(x_i-x'_i)\times\frac{1}{J}\sum_{j=1}^{J}\frac{\partial F\big(x'+\tfrac{j}{J}\cdot(x-x')\big)}{\partial x_i}\)
  其中 \(F\) 取"预测正确答案"的输出。**直觉**:沿 baseline→真实嵌入的直线路径累积梯度,既捕**直接**影响也捕**间接**影响(优于"顺序追加段看答案概率变化"或 leave-one-out——后两者低估间接贡献且在后文完整时对该段不敏感,§2.1)。token 级归因 \(\mathrm{IG}(x)=\sum_i \mathrm{IG}_i(x)\)。**正负号含义**:正 IG = 提升正确答案似然,负 IG = 降低;但**负 IG token 不丢**——它可能是"错误但必要的探索性推理",故下面用**绝对值**捕影响幅度。
- **第 3 步:段级两指标(Eq.3)**【原文 §2.2】:给段 \(S=\{o_1,\dots,o_N\}\),每 token 有归因 \(\mathrm{IG}(o_n)\),
  \(\displaystyle \mathrm{Strength}(S)=\frac{\sum_{o_n\in S}|\mathrm{IG}(o_n)|}{\sqrt{N}}\,,\qquad \mathrm{Consistency}(S)=\frac{\big|\sum_{o_n\in S}\mathrm{IG}(o_n)\big|}{\sum_{o_n\in S}|\mathrm{IG}(o_n)|}\)
  - **强度**:段内绝对 IG 之和,除以 \(\sqrt{N}\) 做**长度归一**(防偏向长段)。
  - **一致性**:段内**净 IG / 总 |IG|**,衡量方向是否统一——接近 1 = 几乎全正或全负(浅层/已定),居中 = 正负混合(反思性,同时有支持与纠正)。
- **第 4 步:CoT 内强度归一(Eq.4)**【原文】(让同一 CoT 内各段强度可比):
  \(\displaystyle \mathrm{Strength}'(S_m)=\frac{\mathrm{Strength}(S_m)}{\sum_{j=1}^{M}\mathrm{Strength}(S_j)}\)
- **第 5 步:重要段判据(Eq.5-7)**【原文 §2.3】:
  - 按归一化强度降序排,设排列 \(\pi\) 使 \(\mathrm{Strength}'(S_{\pi(1)})\geq\cdots\geq\mathrm{Strength}'(S_{\pi(M)})\)(Eq.5)。
  - 取**累计强度超阈值 \(\tau\)(如 80%)的最小 top-\(k^*\)**(Eq.6):
    \(\displaystyle k^*=\arg\min_{k\in\{1,\dots,M\}}\Big\{\textstyle\sum_{i=1}^{k}\mathrm{Strength}'(S_{\pi(i)})\geq\tau\Big\}\)
  - **重要段集**=top-\(k^*\) 里**一致性低于阈值 \(\beta\)(如 0.8)**者(Eq.7):
    \(\displaystyle S_{\mathrm{important}}=\big\{S_{\pi(i)}\,\big|\,i\leq k^*,\ \mathrm{Consistency}(S_{\pi(i)})\leq\beta\big\}\)
    即"高强度 **且** 中等一致性"——过滤掉那些强度高但极端一致(浅层澄清或严重错误探索)的段。
- **第 6 步:选择性 SFT(Eq.8→9)**【原文 §2.3,沿用 Rho-1】:标准 full-CoT SFT 是
  \(\displaystyle \mathcal{L}_{\mathrm{SFT}}(\theta)=-\frac{1}{T}\sum_{t=1}^{T}\log P(o_t|o_{<t},q;\theta)\)
  改为**只在重要段 token 上算 CE、其余 mask**(保留完整轨迹连贯,不物理删除):
  \(\displaystyle \mathcal{L}_{\text{Selective-SFT}}(\theta)=-\frac{1}{\sum_t I(o_t)}\sum_{t=1}^{T}I(o_t)\,\log P(o_t|o_{<t},q;\theta)\)
  其中 \(I(o_t)=1\) 当且仅当 \(o_t\) 属于某重要段 \(S_m\in S_{\mathrm{important}}\)(代码里非重要段 labels 置 \(-100\) 实现 mask)。归一化分母用重要 token 数 \(\sum_t I(o_t)\)。**作用机理**:让参数更新只被最关键推理部分驱动 + 隐式正则(防过拟合冗余),但保全轨迹连贯。
- 逐组件必要性(Table 2 消融,R1-Distill-Qwen-1.5B):
  - **段级 vs token 级**:Our(段级 strength+consistency)46.9/length 13506 > High-Abs-IG Tokens 46.1/14612 > High-Orig-IG Tokens 45.2/14747 → 段级保连贯更优,且绝对 IG 优于原始 IG。
  - **strength+consistency vs 只 strength**:只 High Strength Segments=46.9/14715,加一致性后 length 降到 13506(同 Acc)→ 一致性贡献额外 token 削减。
  - **vs 随机段**:Random Segments 45.1 < Our 46.9 → 归因有效,非随机即可。
  - **vs 剪枝**:Pruned CoT SFT 43.9 < Full CoT SFT 44.8 → 剪枝掉精度,选择性 SFT(保连贯)才涨。消融较完整。【原文】Table 2、§4.3
- 关键超参默认值(汇总,§4.1/§3.2)【原文】:\(J\)(IG 插值步)=**50**;\(\tau=0.7\)、\(\beta=0.8\)(由 1.5B 的 confidence-gain 贪心搜索定,Fig.4);baseline \(x'\)=padding token embedding;全参 SFT,max_seq=16384,max_len(评测)=32768;训练数据 LIMO 817 题(R1-Distill 用 provided CoT,Qwen2.5-7B-Inst 用自生成 32 候选取最短正确解);IG 归因模型与训练基座一致。基座 = R1-Distill-Qwen-1.5B/7B、Qwen2.5-7B-Instruct。

## 靠不靠谱
- 实验与证据:
  - **数据集**:训练=LIMO 数学 817 题(R1-Distill 系用 provided CoT,Qwen2.5-7B-Instruct 用自生成——32 候选取最短正确解)。评测——In-domain MATH500/AMC23/AIME24;OOD GPQA-Diamond/Minerva/OlympiadBench。指标 greedy 准确率 + 温度采样(T=0.6)pass@1/pass@6,报平均输出 token 数,max_len 32768。基座 R1-Distill-Qwen-1.5B/7B、Qwen2.5-7B-Instruct;IG 归因模型与训练基座一致,\(J\!=\!50\),全参 SFT,max_seq=16384。超参 \(\tau=0.7\)、\(\beta=0.8\)(贪心搜索,Fig.4)。【原文】§3.2、§4.1
  - **关键数字**(Table 1):**greedy**——1.5B 44.8→**46.9(+4.7%)**,length 16520→13506(**−18.2%**);7B 62.1→64.5(+3.9%)、length −12.3%;Qwen2.5-7B-Inst 44.2→45.6(+3.2%)。**温度采样**——1.5B pass@1 50.9→51.7(**仅 +1.6%**)、pass@6 +1.1%;7B pass@1 65.5→65.8(**仅 +0.5%**)。段分析(§3):重要段(高强度+中等一致性)带最大正确答案置信度增益(Fig.2);重要段 perplexity/entropy 反而更低(Fig.3);不重要段 BLEU 自相似更高、49% 被判截断 vs 重要段 26%。【原文】Table 1/2、§3.1
  - **baseline 公平吗**:与 full-CoT SFT 在相同数据/相同超参下比,且消融控制"选中内容量可比"(随机 33% 段、top 45% token 等),公平。
  - **看着强但没回答核心问题?**:**greedy 下增益明显但温度采样下显著缩水**(7B 仅 +0.5%)——论文归因于采样随机性"抹平训练优势"。这说明部分增益对解码策略敏感,真实可用性打折扣。【原文】§4.2 自述
- 假设与失效边界:
  - 【原文】§4.2 自述:温度采样的随机性会平滑/部分抵消训练优势,故 greedy 增益更明显。
  - 【推断】依赖 IG 假设成立(归因可靠);§3.1"重要段 perplexity/entropy 更低"意味着 strength 与 entropy 度量在某些段上结论相反,可解释性有赖 IG;\(\tau=0.7\)/\(\beta=0.8\) 由 1.5B 的 confidence-gain 贪心搜索确定后**直接套用所有模型/数据源**(论文称两源搜索结果相似,但论证较弱)。
  - 【推断】仅数学 LIMO 单一训练源、≤7B 规模;IG 需 \(J=50\) 步插值,成本不低,论文未量化归因阶段额外算力。
- 祛魅总结:真贡献=**用 IG 直接归因 + strength/consistency 两指标做段级重要性度量**,§3 三类证据(置信度增益、低 perplexity/entropy、低重复率)+ Table 2 消融(段级>token级、strength+consistency>单指标、>各 baseline 度量)较扎实,且"低熵≠不重要"对 entropy 度量是有力反驳。包装/局限:(1) **框架创新有限**——selective SFT + loss mask 直接来自 Rho-1,核心新意仅在度量;(2) **温度采样下增益缩水严重**(7B +0.5%),部分增益对解码敏感;(3) \(\tau\)/\(\beta\) 跨模型直接复用、IG 算力开销未量化;(4) 单数据源、≤7B,泛化证据有限。【推断】

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=token 对"正确答案预测"的 IG 归因,聚合成段级 strength+consistency,筛出重要段做 CE｜**改什么**=SFT 阶段参数(全参),改的是"在哪些 token 上算 loss"(loss-mask)｜**何时改**=SFT 阶段(数据/监督选择层),离线｜**免梯度?**=否(IG 归因需梯度,SFT 也需梯度)｜**记忆-技能生命周期**=无记忆/技能库｜**防遗忘机制**=无显式防遗忘;但选择性 SFT 作为隐式正则(只学重要段、防过拟合冗余),间接保护模型不被冗余 token 带偏。【原文】§2.3
- ⑦ 开源代码+框架/harness:https://github.com/SiyuanWangw/SegmentSelectiveSFT(已克隆 ~81MB)。三阶段脚本齐全:(1) Attribution——`segment_split.py`切段、`grad_analyze.py`算 IG、`get_important_segments.py`按 \(\tau\)/\(\beta\) 聚合选段、`cal_attribution.sh`串流程;(2) SelectiveSFT——`train_mask.py`(`--mask` 开关,非重要段 labels 置 −100 实现 loss-mask 全参 SFT;requirements 锁 **unsloth==2025.11.3 + trl==0.23.0 + transformers 4.57.1**),`run_train.sh`;(3) Eval——`evaluate.py`/`grader.py`+`latex2sympy/`。框架=**SFT 用 unsloth+trl,归因阶段自写 IG 脚本,评测含 latex2sympy**。代码与方法一致、loss-mask 已确认,复现门槛主要在 IG 算力。【原文】GitHub + 既有 analysis 对 clone 的核查
- 💰 资源/成本与可扩展性:IG 需 \(J=50\) 步插值(每 token 算 50 次前/反向),成本不低且论文未量化;全参 SFT,max_seq 16384。训练数据极省(LIMO 817 题)。【原文】§4.1、§2.2
- 🎯 对"探索-巩固"对标:**中-高支撑,直击"巩固时该固化哪些 token/段"这一信用分配问题**。本文的 strength/consistency 度量正是 idea 里"巩固=固化有效路径"时需要回答的"**哪些 token 值得固化进参数**"——而且"中等一致性=反思性推理(混合支持与纠正)= 更值得学",与 idea"path-recovery=走偏后自选恢复分支"高度呼应(纠正性 token 正是 recovery 信号)。论文末尾明说"important segment 识别可推广到 RL 中强调重要内容的策略梯度更新",直接指向 idea 的 token 级稠密信用。**可借组件**:(a) **IG 段级归因**可作"判定哪个段是 path-recovery 关键段"的离线工具,给 MTP/OPD 提供 token/段级稠密信用权重;(b)"strength+consistency"双指标可移植到 RL 的 advantage 加权(强调反思性段);(c)"低熵≠不重要"提醒——做 path-selection/recovery 时不能只用熵/置信度定关键步(与 score/sed 等用熵的做法形成互补/警示)。**缺口/差异**:无 teacher 脚手架、无 on-policy(归因基于已有 CoT)、无 MTP 前瞻;是离线 SFT 监督选择而非在线探索-巩固。一句判定:**"巩固时固化哪些 token"的强对标 + IG 段级归因这个可直接借的稠密信用工具,且"低熵≠不重要"对用熵定关键步的路线是重要警示**。【推断,依据 §2.3/§3.1 + 作者自述 vs idea】
- 🔭 开放问题/未来方向:【原文】important segment 识别可推广到其他场景,如在 RL 中对重要内容强调策略梯度更新(§1 末)。【推断】(1) 把 IG 段级归因接进 RL/OPD 做 advantage 加权,替代纯结果奖励的均匀信用;(2) 降 IG 算力(更少插值步或近似归因);(3) 把 \(\tau\)/\(\beta\) 改为按模型/数据自适应而非固定复用;(4) 解决温度采样下增益缩水——可能需训练目标显式对采样鲁棒;(5) 与 MTP 前瞻结合:用前瞻分布的变化作为另一种"段重要性"信号交叉验证 IG。

RETURN:segment_attrib | 读PDF? 是(16页:全方法/IG Eq.1-2/strength-consistency Eq.3-4/重要段判据 Eq.5-7/选择性SFT Eq.8-9/§3分析/Table1主结果含温度采样/Table2消融/setup) | 加厚? 是(方法到可复现:三阶段流水线+IG积分式+J步近似+双指标+CoT内归一+top-k*累计判据+一致性筛选+Eq.8/9选择性SFT全转MathJax,逐式给直觉与数据流;相关工作四路线A-D精确短板) | LaTeX公式条数 9 | 待核数 0

rho1 | RHO-1: Not All Tokens Are What You Need | 厦门大学 / 清华 / 上海AI Lab / Microsoft（Zhenghao Lin、Zhibin Gou 共一@MSRA 实习；Weizhu Chen 等） | NeurIPS 2024（Best Paper Runner-up）· v4 2025-01-08 | 主题线 L6(Token信用/选择)+L1(token级蒸馏先验)·相关性 中（机制同源高）

**原始论文**:https://arxiv.org/abs/2404.07965

## 一眼看懂
- 🟦 TL;DR:预训练惯例是对**所有** token 一视同仁施加 next-token 损失。本文主张"语料里并非所有 token 都同等重要",提出 Selective Language Modeling(SLM):先用高质量数据训一个**参考模型(RM)**,用它给每个 token 算 loss;训练目标模型时只在"excess loss = 训练模型 loss − RM loss 最高的 top-k% token"上算损失(其余 token 前向照常进、但不回传损失)。在 15B OpenWebMath 上持续预训练,RHO-1 在 9 个数学任务上少样本精度最高绝对提升 30%、达到 baseline 精度快 5-10×;RHO-1-1B/7B 微调后 MATH 达 40.6%/51.8%,**只用 DeepSeekMath 3% 的预训练 token 就追平它**。【Abstract,图1】
- 最巧的一步:**用参考模型算 excess loss 来选 token(式3-5)**。抽掉"参考模型打分"这一步,SLM 就退化成"按训练模型自身 loss 选高 loss token"——而高 loss 的往往是噪声/不可学的 hard token(§2.1 的 H→H 噪声类),会选错。正是"excess loss = 当前模型还没学会、但 RM(理想分布)觉得该学"这个**相对量**,才把"高 aleatoric 噪声 token"和"真正可学且对齐目标分布的 token"区分开(§2.2 intuition)。【§2.2】

## 为什么做
- 研究背景:扩规模持续提升 next-token 精度,但"训所有数据"并非最优;文档级数据过滤(启发式/分类器)已是关键,显著提质增效。【§1,L131-137】
- 解决的具体痛点:① 即便严格文档级过滤,语料在 **token 级**仍含噪声(幻觉、高度歧义、难预测 token)——删 token 会改语义、过严过滤丢有用数据且引偏置;② web 分布与下游理想分布不天然对齐;③ 对所有 token 同等施损→在非关键 token 上浪费算力、可能限制模型上限。【§1,L138-147】
- 相关工作的来龙去脉与精确短板(据附录B.2):
  - **文档级 reference-model 数据选择**:Selection-via-Proxy、**DoReMi**(worst-case excess loss 定域权重)——粒度在**文档/域**,非 token。
  - **RHO-LOSS**(Mindermann 2022,**最近邻**,数学上与 excess loss 等价):用 holdout 小模型选 in-batch 样本,目标=最小化可约 holdout loss 以降泛化误差。**精确差异**(站它肩上但三处不同):① focus——RHO-LOSS 推导"可约 holdout loss"理论,SLM 直接"选有用 token";② 代理模型语义——RHO-LOSS 训在**随机 holdout** 上(目的是泛化界),SLM 的 RM 训在**高质量数据**上(反映目标分布);③ 尺度/粒度——RHO-LOSS 是**样本级 + 小规模(1K-1M)+ 任务微调**(MNIST/SST-2),SLM 是 **token 级 + 大规模(80B token)+ 预训练**,且打分函数不限 excess loss(附录H 有其他)。
  - **BERT 系 token 级策略**:selective masking 聚焦重要 token、token dropping 省算力——但在 BERT/分类语境,非自回归 LLM 预训练。
  - **训练动力学**:Xia2022 认为 ppl 少变=已学会——本文**修正**:存在"hard token resist convergence"的谱系(高 aleatoric 噪声 token 持续高 loss、抗收敛)。
  - 作者称**自己是首个把 token 级数据选择用于 LLM 预训练**。【附录B.2,L1742-1772】
- 动机链:训所有 token 浪费且受噪声拖累→观察 token loss 训练动力学发现只有少数 token 真在有效下降(26% H→L)→既然多数 token 要么已学会(51% L→L)要么噪声不可学(11% H→H)→那就只在"该学且能学"的 token 上施损→需要一个理想分布的参照来判定→用 RM 算 excess loss 选 top-k%。
- 与最近邻工作的Δ:见上 RHO-LOSS 三处差异;vs CLM——SLM 只换损失掩码(top-k% excess-loss token 算损失),其余流程同标准 CLM。关键点:**把"数据选择"从文档级下沉到 token 级,并用相对参照(excess loss)规避"高 loss=噪声"陷阱**。

## 怎么做(到"读完能复现"的粒度)

### A. 动机实证:token loss 的四类训练动力学(§2.1)
持续预训练 TinyLlama-1B(15B OpenWebMath,每 1B token 存 checkpoint),在 ~320K token 验证集上按 loss 轨迹把 token 分四类(图3a):
- **L→L(51%)**:始终低 loss,**已学会**。
- **H→L(26%)**:loss 显著下降,**真正在有效学习的少数**。
- **H→H(11%)**:持续高 loss、抗收敛,多因**高 aleatoric 不确定性(噪声)**。
- **L→H(12%)**:训练中 loss 反升。
关键观察:大量 L→L、H→H token 的 loss 训练中**剧烈波动、抗收敛**(图3b/c),内容多为噪声(附录D.2)。→ 总 loss 平滑下降掩盖了 token 间复杂异质动力学;选对 token 可稳定轨迹、提数据效率。【§2.1,图3】

### B. SLM 三步流水线(§2.2,图4;数据流动:RM → token 打分 → 选择性回传)
- **Step 1 训参考模型 RM**:在反映"理想分布"的 curated 高质量语料上,用标准交叉熵训 RM。RM 的 token 概率定义参考 loss
\(\displaystyle \mathcal L_{\text{RM}}(x_i)=-\log P(x_i\mid x_{<i}).\)
- **Step 2 用 RM 给大语料每个 token 打分**:对预训练语料每个 token 算 \(\mathcal L_{\text{RM}}(x_i)\)(可离线一次算完缓存)。
- **Step 3 选择性训练目标模型**:标准 CLM 是 \(\mathcal L_{\text{CLM}}(\theta)=-\frac1N\sum_{i=1}^N\log P(x_i\mid x_{<i};\theta)\)。SLM 改为只在高 excess loss 的 token 上回传。**excess loss**:
\(\displaystyle \mathcal L_\Delta(x_i)=\mathcal L_\theta(x_i)-\mathcal L_{\text{RM}}(x_i),\)
即"当前训练模型还没学会(\(\mathcal L_\theta\) 高)但 RM(理想分布)认为该会(\(\mathcal L_{\text{RM}}\) 低)"的差。引入选择比例 \(k\%\),只对 batch 内 excess loss 排名 top-\(k\%\) 的 token 算交叉熵:
\(\displaystyle \mathcal L_{\text{SLM}}(\theta)=-\frac{1}{N\cdot k\%}\sum_{i=1}^{N}\mathbb I_{k\%}(x_i)\cdot\log P(x_i\mid x_{<i};\theta),\qquad \mathbb I_{k\%}(x_i)=\begin{cases}1,&x_i\ \text{在 top-}k\%\ \text{(按打分}\ S(x_i)\text{)}\\0,&\text{otherwise}\end{cases}\)
默认打分函数 \(S=\mathcal L_\Delta\)。**关键工程**:整段序列照常喂前向(保持上下文/注意力完整、不破坏语言连贯),只在**反向**掩掉非选 token 的损失——故"无额外预训练开销、易集成"(只是把梯度预算集中)。【§2.2 式1-5】

### C. 各组件必要性(消融)
- **参考模型/excess loss 必要但可降级**(§2.1 动机 + §3.4):去掉外部高质量 RM,用 **self-referencing**(先在目标语料上训个 RM 再选)仍提升(数学/通用平均 +3.3%,Table3);但纯按训练模型**自身 loss** 选会选到噪声(§2.1 的 H→H)——所以"相对参照"是必要的。
- **token 选择本身是因果有效**(图6/7):SLM 在"被选 token"上 loss 降更多且与下游性能呈幂律**正**相关;而**未被选 token 的 loss 与下游性能负相关**(图7)→ 实证"降所有 token 的 loss 并非必要、甚至有害"。
- **选择比例 \(k\%\)**(§3.4 图9 + 训练配置):TinyLlama-1.1B 用 **60%**、Mistral-7B 用 **70%** 最佳;太高≈CLM、太低丢有用 token。
- **token 选择的动态性**(图8/14):后期 checkpoint 选的 token 在后期 ppl 更高、前期更低 → 模型先优化"可学空间大"的 token;观察到被选 token 的 sample-wise "double descent"。
- **被选 token 内容**(附录G.1):多与数学相关 → SLM 确实在原始语料里聚焦了数学相关部分。

### D. 训练/数据/评测(复现锚点)
- **RM 数据**:数学 RM 用 0.5B 高质量数学 token(GPT 合成 + 人工 curated);通用 RM 用 1.9B(Tulu-v2 / OpenHermes-2.5)。RM 训 3 epoch,lr 5e-5(1B)/1e-5(7B),cosine decay,max seq 2048(1B)/4096(7B),**RM 与持续预训练模型用同一 base 初始化**。
- **持续预训练**:数学 15B OpenWebMath(TinyLlama-1.1B / Mistral-7B);通用 80B token(TinyLlama);batch 统一 1M token;1B 模型约 19 小时。
- **评测**:9 个数学任务(GSM8K/MATH/SVAMP/ASDiv/MAWPS/TAB/MQA/MMLU-STEM/SAT,少样本 CoT)+ 通用 15 任务。**关键数字(Table1)**:数学持续预训练 RHO-1-1B 平均 38.1(TinyLlama-CT baseline 21.6,GSM8K +23.4、MATH +11.6);RHO-1-7B(Mistral-CT 底,10.5B 被选 token)平均 66.2,Mistral-CT 55.8;微调后 RHO-1-7B MATH 51.8、1B 40.6(首个 >40% 的 1B,逼近早期 GPT-4 CoT 42.5%);用 15B token 追平用 500B 的 DeepSeekMath-7B;通用域 80B token 平均 +6.8%(代码/数学增益 >10%);self-reference +3.3%。token 量口径标注清楚(RHO-1 只算被选 token,故"3% token 追平")。【Table1/3,§3】

## 靠不靠谱
- baseline 公平吗:与 CLM(TinyLlama-CT/Mistral-CT,同语料同 token 量)直接对比公平;与 DeepSeekMath/LLemma/Minerva 等用 token 量对齐("3% token 追平")论据有力。
- 看着强但没回答核心问题(作者自承,§C 限制):① **泛化性**——纯 SLM 会快速收敛到 RM 聚焦的域、未选 token loss 显著上升,目前未见偏置等不良后果,但混入通用预训练损失可能防过拟合(Goodhart),建议未来扩 RM 语料;② **可扩展性**——只在 ≤7B 模型、<100B token 验证,很大模型/海量语料可能自然习得"压缩有用数据"的归纳偏置,SLM 增益是否仍在未知;③ **RM 必要性**——需高质量 RM(可 self-reference 降级)。
- 假设与失效边界:【原文】① 必须有反映"理想/目标分布"的高质量 RM(或可 self-reference);② 验证仅限 ≤7B、<100B token、数学/通用域;③ 纯 SLM 会让未选 token loss 上升、向 RM 域收敛(§C)。【推断】SLM 的 token 选择是**静态/离线**的(RM 固定,excess loss 单向 = ref−train),不像 OPD 那样在学生访问状态上动态对齐——当目标分布与下游任务不一致时,RM 选错 token 的风险被放大;\(k\%\) 是启发式定的,跨域可能需重调。
- 祛魅总结:真贡献=① 揭示 token 级 loss 训练动力学(四类 token、噪声 token 抗收敛),挑战"训所有 token"惯例;② 用 excess loss 做 token 级选择,在数学持续预训练上拿到极高 token 效率(3% token 追平 DeepSeekMath)。【推断】被高估的风险:增益可能强绑定"小模型 + 数学窄域 + 有理想 RM"这一甜区(作者自列限制);RHO-LOSS 已有 excess-loss 的数学形式,本文创新更多在"token 级 + 预训练规模 + 训练动力学动机"的落地而非全新理论。被低估的是其作为"token 异质性/token 级信用分配"先验的思想价值——后续 token 级蒸馏/加权工作的精神源头之一。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=参考模型给的 token excess loss(ref−train)→选 top-k% token | **改什么**=目标模型参数 θ(只在被选 token 上回传 CLM 损失) | **何时改**=持续预训练阶段;选择随 checkpoint 动态变(每个 batch 按当前 excess loss 排名) | **免梯度?**=N/A(本身是预训练监督学习,非 RL;无策略梯度) | **记忆-技能生命周期**=无显式记忆/技能库 | **防遗忘机制**=无专门机制;反而 §C 指出纯 SLM 会让未选 token loss 上升、向 RM 域收敛(有"窄化/遗忘通用能力"风险),建议混通用预训练损失缓解
- ⑦ 开源代码+框架/harness:https://github.com/microsoft/rho(v1 元信息已核-真实)。**重要边界**:该官方仓**只发布模型权重 + 评测代码**(rho-1/ 下仅 math-evaluation-harness 子模块[本地克隆为空/未初始化] + 复现输出 outputs.zip;README Quick Start 只给 evaluation run_eval.sh cot/tora)——**SLM 持续预训练/微调的训练代码并未开源**,社区需自行按论文 §2 实现 token-level excess-loss masking。故"框架"实质=权重+评测仓,核心训练栈不可核;模型在 HF: microsoft/rho-math-1b/7b-v0.1。约 6.4MB 已 clone。
- 💰 资源/成本与可扩展性:1B 模型 15B token 约 19 小时,batch 1M token;核心卖点即**token 效率**(达 baseline 快 5-10×;3% token 追平 DeepSeekMath-7B)。可扩展性作者明确未验证 >7B / >100B token(§C Scalability),是显式开放问题。【§3.1,§C】
- 🎯 对"探索-巩固"对标:**思想先验级支撑(token 异质性 + token 信用),非方法竞品**。映射:RHO-1 的核心信念"并非所有 token 都该学、要用参照分布选出值得学的 token"与 TSRD 的"token 异质性 / 教 path-selection(哪些 token/步骤值得被脚手架接管)"高度同源;excess loss = ref − train 可视为一种**静态、离线的 token 级蒸馏信号**——把"参考模型(=理想分布代言)"换成 OPD 里的 teacher,"excess loss 选 token"就对应 OPD 里"teacher 信号决定哪些 token 值得学"。**可借组件**:① excess-loss 式 token 打分作为"哪些 token 该被脚手架/teacher 接管"的离线先验(便宜、可与 OPD 在线 overlap 信号互补);② "前向喂全序列、反向只在被选 token 回传"的稀疏掩码工程——正合 TSRD"稀疏脚手架/单点接管"的实现范式(不破坏上下文连贯,只集中梯度);③ token loss 训练动力学四分类(H→H 噪声 / H→L 可学)可用于诊断 path-recovery 该针对哪类 token。**缺口/差异**:RHO-1 是**静态离线选择 + 纯预训练监督**,无 on-policy(不在模型自己采样的状态上选)、无 teacher 分布对齐、无 MTP 前瞻、无记忆——与 TSRD 的"on-policy 自选 + teacher 脚手架 + MTP 前瞻"差好几层;且其选择信号是单向 excess loss,缺 OPD 的分布匹配语义。判定:**经典先验工作,提供 token-level 选择/信用的思想与稀疏掩码工程,但非 on-policy/蒸馏直接对标**。
- 🔭 开放问题/未来方向:【原文】① 从 token 级视角改进 LLM 预训练值得深入(Conclusion);② 扩 RM 语料范围 + 扩预训练数据(§C);③ 验证 SLM 能否扩到很大模型/海量数据(§C Scalability);④ 打分函数不限 excess loss,可探索其他(附录H)。【推断】把静态 excess-loss 选择升级为 on-policy 动态选择(在学生采样状态上算,逼近 OPD)、用 teacher 分布替代 RM 做 token 级蒸馏选择、把 token 选择与 MTP 前瞻结合(用前瞻判定"将走偏"的关键 token 优先施损)、研究"选择窄化导致通用能力遗忘"的防护(与持续学习/防遗忘线对接)。

RETURN: rho1|读PDF?是(_txt 1000+行,核到§2.1四类token动力学+§2.2 SLM式1-5+§3训练配置/Table1)|加厚?是(方法扩为A-D四块+RM/训练超参,相关工作RHO-LOSS三处差异精确化,5条公式转MathJax)|LaTeX公式条数 4|待核数 0

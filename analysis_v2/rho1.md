rho1 | RHO-1: Not All Tokens Are What You Need | 厦门大学 / 清华 / 上海AI Lab / Microsoft（Zhenghao Lin、Zhibin Gou 共一@MSRA 实习；Weizhu Chen 等） | NeurIPS 2024（Best Paper Runner-up）· v4 2025-01-08 | 主题线 L6(Token信用/选择)+L1(token级蒸馏先验)·相关性 中（机制同源高）

**原始论文**:https://arxiv.org/abs/2404.07965

## 一眼看懂
- 🟦 TL;DR:预训练惯例是对**所有** token 一视同仁施加 next-token 损失。本文主张"语料里并非所有 token 都同等重要",提出 Selective Language Modeling(SLM):先用高质量数据训一个**参考模型(RM)**,用它给每个 token 算 loss;训练目标模型时只在"excess loss = 训练模型 loss − RM loss 最高的 top-k% token"上算损失(其余 token 前向照常进、但不回传损失)。在 15B OpenWebMath 上持续预训练,RHO-1 在 9 个数学任务上少样本精度最高绝对提升 30%、达到 baseline 精度快 5-10×;RHO-1-1B/7B 微调后 MATH 达 40.6%/51.8%,**只用 DeepSeekMath 3% 的预训练 token 就追平它**。【Abstract,图1】
- 最巧的一步:**用参考模型算 excess loss 来选 token(式3-5)**。抽掉"参考模型打分"这一步,SLM 就退化成"按训练模型自身 loss 选高 loss token"——而高 loss 的往往是噪声/不可学的 hard token(§2.1 的 H→H/L→L 噪声类),会选错。正是"excess loss = 当前模型还没学会、但 RM(理想分布)觉得该学"这个**相对量**,才把"高 aleatoric 噪声 token"和"真正可学且对齐目标分布的 token"区分开(§2.2 intuition)。

## 为什么做
- 研究背景:扩规模持续提升 next-token 精度,但"训所有数据"并非最优;文档级数据过滤(启发式/分类器)已是关键,显著提质增效。【§1,L131-137】
- 解决的具体痛点:① 即便严格文档级过滤,语料在 **token 级**仍含噪声(幻觉、高度歧义、难预测 token)——删 token 会改语义、过严过滤丢有用数据且引偏置;② web 分布与下游理想分布不天然对齐;③ 对所有 token 同等施损→在非关键 token 上浪费算力、可能限制模型上限。【§1,L138-147】
- 相关工作 & 各自不足:① 文档级 reference-model 数据选择(Selection-via-Proxy、DoReMi worst-case excess loss 定域权重)——粒度在文档/域,非 token;② **RHO-LOSS**(Mindermann 2022,数学上与 excess loss 等价,用 holdout 小模型选 in-batch 样本)——但 RHO-LOSS 是**样本级、小规模(1K-1M)、任务微调**(MNIST/SST-2),且代理模型训在随机 holdout 上(目的=可约 holdout loss 最小化泛化误差);SLM 是**token 级、大规模(80B token)、预训练**,RM 训在高质量数据上(反映目标分布),且打分函数不限于 excess loss(附录H 有其他);③ BERT 系 token 级策略(selective masking 聚焦重要 token、token dropping 省算力);④ 训练动力学研究(Xia2022 认为 ppl 少变=已学会)——本文修正:存在"hard token resist convergence"的谱系。**作者称自己是首个把 token 级数据选择用于 LLM 预训练**。【附录B.2,L1742-1772】
- 动机链:训所有 token 浪费且受噪声拖累→观察 token loss 训练动力学发现只有少数 token 真在有效下降(26% H→L)→既然多数 token 要么已学会(51% L→L)要么噪声不可学(11% H→H)→那就只在"该学且能学"的 token 上施损→需要一个理想分布的参照来判定→用 RM 算 excess loss 选 top-k%。
- 与最近邻工作的Δ:vs RHO-LOSS(最近邻)——见上,差在 focus(选有用 token vs 推导可约 holdout loss)、代理模型语义(高质量数据 vs 随机 holdout)、尺度粒度(80B token 预训练 vs 1K-1M 样本微调)。vs CLM——SLM 只换损失掩码(top-k% excess-loss token 算损失),其余流程同标准 CLM。关键点:**把"数据选择"从文档级下沉到 token 级,并用相对参照(excess loss)规避"高 loss=噪声"陷阱**。

## 怎么做 + 靠不靠谱
- 方法流水线(SLM 三步,图4):① 在 curated 高质量语料上用标准交叉熵训一个参考模型 RM;② 用 RM 对预训练大语料每个 token 算 reference loss L_RM(式1,=−log P(xi|x<i));③ 训练目标模型时,对每个 token 算 excess loss L∆=Lθ−L_RM(式3),只对 batch 内 excess loss 排名 top-k% 的 token 算交叉熵损失(式4-5,indicator I_k%),其余 token 仍输入前向但损失置零。【§2.2】
- 逐组件必要性(消融较扎实):
  - **参考模型/excess loss**(§2.1 动机 + §3.4):去掉外部高质量 RM→可用"self-referencing"(目标语料上先训个 RM 再选)仍有提升(数学/通用平均 +3.3%,Table3)——说明 RM 必要但可降级为自参照;但纯按自身 loss 选会选到噪声(§2.1)。
  - **token 选择本身**(图6/7 分析):SLM 在"被选 token"上 loss 降更多且与下游性能呈幂律正相关;**未被选 token 的 loss 与下游性能负相关**→证明"降所有 token 的 loss 并非必要"(图7),这是 SLM 有效性的因果证据。
  - **选择比例 k%**(§3.4 图9):~60%(Tinyllama-1.1B)/70%(更大模型)最佳;太高=接近 CLM,太低=丢有用 token。
  - **token 选择的动态性**(图8/14):后期 checkpoint 选的 token 在后期 ppl 更高、前期更低→模型先优化"可学空间大"的 token;观察到选中 token 的 sample-wise "double descent"。
  - 被选 token 内容(§G.1):多与数学相关→SLM 确实在原始语料里聚焦了数学相关部分。
- 关键机制/公式(直觉):excess loss = "训练模型还没学会(Lθ 高)但理想分布认为该会(L_RM 低→RM 学得会)"的差,正好筛出"可学 + 对齐目标分布"的 token;前向仍喂全序列(保持上下文/注意力完整),只在反向掩掉非选 token——所以不破坏语言连贯性,只是把梯度预算集中。
- 实验与证据:数据 15B OpenWebMath(数学持续预训练)/ 80B 通用 token(TinyLlama);模型 TinyLlama-1.1B、7B;评测 9 个数学任务(GSM8K/MATH/SVAMP/ASDiv/MAWPS/TAB/MQA/MMLU-STEM/SAT,少样本 CoT)+ 通用 15 任务。**关键数字**:数学持续预训练 RHO-1-1B 平均 38.1(Tinyllama-CT baseline 21.6,GSM8K +23.4、MATH +11.6);RHO-1-7B 微调后 MATH 51.8、1B 40.6(首个 >40% 的 1B,逼近早期 GPT-4 CoT 42.5%);用 15B token 追平用 500B 的 DeepSeekMath-7B;通用域 80B token 平均 +6.8%(代码/数学增益 >10%);self-reference +3.3%。【Table1/3,§3】
- baseline 公平吗:与 CLM(Tinyllama-CT,同语料同 token 量)直接对比公平;与 DeepSeekMath/LLemma/Minerva 等用 token 量对齐("3% token 追平")论据有力。token 量口径标注清楚(RHO-1 只算被选 token)。
- 看着强但没回答核心问题:作者自承(§C 限制)——① **泛化性**:纯 SLM 会快速收敛到 RM 聚焦的域、未选 token loss 显著上升,目前未见偏置等不良后果,但混入通用预训练损失可能防过拟合(Goodhart),作者建议未来扩 RM 语料;② **可扩展性**:只在 ≤7B 模型、<100B token 验证——很大模型/海量语料可能自然习得"压缩有用数据"的归纳偏置,SLM 增益是否仍在未知;③ RM 必要性——需高质量 RM(可用 self-reference 降级)。这些都是真实未答的核心问题。
- 假设与失效边界:【原文】① 必须有反映"理想/目标分布"的高质量 RM(或可 self-reference);② 验证仅限 ≤7B、<100B token、数学/通用域;③ 纯 SLM 会让未选 token loss 上升、向 RM 的域收敛(§C Generalizability)。【推断】SLM 的 token 选择是**静态/离线**的(RM 固定,excess loss 单向 = ref−train),不像 OPD 那样在学生访问状态上动态对齐——当目标分布与下游任务不一致时,RM 选错 token 的风险会被放大;k% 是启发式定的,跨域可能需重调。
- 祛魅总结:真贡献=① 揭示 token 级 loss 训练动力学(四类 token、噪声 token 抗收敛),挑战"训所有 token"惯例;② 用 excess loss 做 token 级选择,在数学持续预训练上拿到极高 token 效率(3% token 追平 DeepSeekMath)。【推断】被高估的风险:增益可能强绑定"小模型 + 数学窄域 + 有理想 RM"这一甜区(作者自己列为限制);RHO-LOSS 已有 excess-loss 的数学形式,本文创新更多在"token 级 + 预训练规模 + 训练动力学动机"的落地而非全新理论。被低估的是其作为"token 异质性/token 级信用分配"先验的思想价值——这正是后续 token 级蒸馏/加权工作的精神源头之一。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=参考模型给的 token excess loss(ref−train)→选 top-k% token | **改什么**=目标模型参数 θ(只在被选 token 上回传 CLM 损失) | **何时改**=持续预训练阶段;选择随 checkpoint 动态变(每个 batch 按当前 excess loss 排名) | **免梯度?**=N/A(本身是预训练监督学习,非 RL;无策略梯度) | **记忆-技能生命周期**=无显式记忆/技能库 | **防遗忘机制**=无专门机制;反而 §C 指出纯 SLM 会让未选 token loss 上升、向 RM 域收敛(有"窄化/遗忘通用能力"风险),建议混通用预训练损失缓解
- ⑦ 开源代码+框架/harness:https://github.com/microsoft/rho(v1 元信息已核-真实)。**重要边界**:该官方仓**只发布模型权重 + 评测代码**(rho-1/ 下仅 math-evaluation-harness 子模块[本地克隆为空/未初始化] + 复现输出 outputs.zip;README Quick Start 只给 evaluation run_eval.sh cot/tora)——**SLM 持续预训练/微调的训练代码并未开源**,社区需自行按论文 §2 实现 token-level excess-loss masking。故"框架"实质=权重+评测仓,核心训练栈不可核;模型在 HF: microsoft/rho-math-1b/7b-v0.1。约 6.4MB 已 clone。
- 💰 资源/成本与可扩展性:原文未给 GPU 时;核心卖点即**token 效率**(达 baseline 快 5-10×;3% token 追平 DeepSeekMath-7B)。可扩展性作者明确未验证 >7B / >100B token(§C Scalability),是显式开放问题。【图1,§C】
- 🎯 对"探索-巩固"对标:**思想先验级支撑(token 异质性 + token 信用),非方法竞品**。映射:RHO-1 的核心信念"并非所有 token 都该学、要用参照分布选出值得学的 token"与 TSRD 的"token 异质性 / 教 path-selection(哪些 token/步骤值得被脚手架接管)"高度同源;excess loss = ref − train 可视为一种**静态、离线的 token 级蒸馏信号**——把"参考模型(=理想分布的代言)"换成 OPD 里的 teacher,"excess loss 选 token"就对应 OPD 里"teacher 信号决定哪些 token 值得学"。**可借组件**:① excess-loss 式的 token 打分作为"哪些 token 该被脚手架/teacher 接管"的离线先验(便宜、可与 OPD 在线 overlap 信号互补);② "前向喂全序列、反向只在被选 token 回传"的稀疏掩码工程——正合 TSRD "稀疏脚手架/单点接管"的实现范式(不破坏上下文连贯,只集中梯度);③ token loss 训练动力学四分类(H→H 噪声 / H→L 可学)可用于诊断 path-recovery 该针对哪类 token。**缺口/差异**:RHO-1 是**静态离线选择 + 纯预训练监督**,无 on-policy(不在模型自己采样的状态上选)、无 teacher 分布对齐、无 MTP 前瞻、无记忆——与 TSRD 的"on-policy 自选 + teacher 脚手架 + MTP 前瞻"差好几层;且其选择信号是单向 excess loss,缺 OPD 的分布匹配语义。判定:**经典先验工作,提供 token-level 选择/信用的思想与稀疏掩码工程,但非 on-policy/蒸馏直接对标**。
- 🔭 开放问题/未来方向:【原文】① 从 token 级视角改进 LLM 预训练值得深入(Conclusion);② 扩 RM 语料范围 + 扩预训练数据(§C);③ 验证 SLM 能否扩到很大模型/海量数据(§C Scalability);④ 打分函数不限 excess loss,可探索其他(附录H)。【推断】把静态 excess-loss 选择升级为 on-policy 动态选择(在学生采样状态上算,逼近 OPD)、用 teacher 分布替代 RM 做 token 级蒸馏选择、把 token 选择与 MTP 前瞻结合(用前瞻判定"将走偏"的关键 token 优先施损)、研究"选择窄化导致通用能力遗忘"的防护(与持续学习/防遗忘线对接)。

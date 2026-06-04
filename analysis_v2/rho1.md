rho1 | RHO-1: Not All Tokens Are What You Need | 厦门大学 / 清华 / 上海AI Lab / Microsoft（Zhenghao Lin、Zhibin Gou 共一@MSRA 实习；Weizhu Chen 等） | NeurIPS 2024（Best Paper Runner-up）· v4 2025-01-08 | 主题线 L6(Token信用/选择)+L1(token级蒸馏先验)·相关性 中（机制同源高）

**原始论文**:https://arxiv.org/abs/2404.07965

## 一眼看懂
> 一句话导读:预训练时不要对每个 token 平均用力——先用高质量数据训一个"参照模型"当标尺,只在"目标模型还没学会、但标尺觉得该会"的少数 token 上算损失,数据效率能涨一个量级。

- 🟦 TL;DR:预训练的惯常做法是对**所有** token 一视同仁地施加 next-token 损失(预测下一个词的交叉熵)。本文主张"语料里并非所有 token 都同等重要",提出 Selective Language Modeling(SLM,选择性语言建模)。三步:
  1. 先用一批高质量数据训一个**参考模型(RM)**;
  2. 用 RM 给大语料里每个 token 都算一个 loss;
  3. 训练目标模型时,只对 "excess loss(超额损失)= 训练模型 loss − RM loss" 排名最高的 top-k% token 计算损失,其余 token 照常进前向、但不回传损失。
- 效果:在 15B token 的 OpenWebMath 上做持续预训练,RHO-1 在 9 个数学任务上少样本精度最高绝对提升 30%、达到 baseline 同等精度只需 baseline 5-10× 更少的步数;RHO-1-1B/7B 微调后 MATH 分别达 40.6%/51.8%,且**只用了 DeepSeekMath 3% 的预训练 token 就追平它**。【Abstract,图1】
- 最巧的一步:**用参考模型算 excess loss 来选 token(式3-5)**。这一步的价值,看抽掉它会怎样:如果不用 RM 打分,SLM 就退化成"按训练模型自己的 loss 来选高 loss token"——但训练模型自己觉得难(高 loss)的 token,往往是噪声、是根本学不会的 hard token(对应 §2.1 的 H→H 噪声类),选了反而有害。关键在于 excess loss 是一个**相对量**:它度量的是"当前模型还没学会(\(\mathcal L_\theta\) 高),但 RM 这个理想分布的代言觉得本该学会(\(\mathcal L_{\text{RM}}\) 低)"的那批 token。正是这个相对差,才能把"高 aleatoric 噪声 token(数据本身固有的、学不掉的随机性)"和"真正可学、且对齐目标分布的 token"区分开。【§2.2】

## 为什么做
> 一句话导读:扩规模能提精度,但"对所有 token 平均施损"在浪费算力、还被噪声 token 拖累;观察发现真正在有效学习的 token 只占约四分之一,于是改成只训"该学且能学"的那批。

- 研究背景:扩规模持续提升 next-token 精度,但"训所有数据"并非最优。文档级的数据过滤(启发式规则或分类器筛文档)已是公认关键手段,能显著提质增效。【§1,L131-137】
- 解决的具体痛点(三条):
  1. 即便做了严格的文档级过滤,语料在**更细的 token 级**仍含噪声(幻觉、高度歧义、难预测的 token);而直接删 token 会改变语义,过严过滤又会丢有用数据并引入偏置;
  2. web 数据的分布与下游任务的理想分布不天然对齐;
  3. 对所有 token 同等施损,会在非关键 token 上浪费算力,甚至可能限制模型上限。【§1,L138-147】
- 相关工作的来龙去脉与精确短板(据附录B.2):
  - **文档级 reference-model 数据选择**:如 Selection-via-Proxy、DoReMi(用最坏情况下的 excess loss 来定各领域的权重)——但粒度都在**文档/域**层面,不是 token 级。
  - **RHO-LOSS**(Mindermann 2022,本文**最近邻**,数学形式上与 excess loss 等价):用一个 holdout 小模型来挑选 batch 内的样本,目标是最小化"可约的 holdout loss"以降低泛化误差。本文站在它肩上,但有三处精确差异:
    - **focus 不同**:RHO-LOSS 推导的是"可约 holdout loss"的理论,SLM 直接奔着"选出有用 token"去;
    - **代理模型的语义不同**:RHO-LOSS 的小模型训在**随机 holdout** 数据上(目的是给出泛化界),SLM 的 RM 训在**高质量数据**上(目的是反映目标分布);
    - **尺度与粒度不同**:RHO-LOSS 是**样本级 + 小规模(1K-1M)+ 任务微调**(实验在 MNIST/SST-2),SLM 是 **token 级 + 大规模(80B token)+ 预训练**,且打分函数不限于 excess loss(附录H 给了其他选择)。
  - **BERT 系 token 级策略**:selective masking(聚焦重要 token)、token dropping(省算力)——但都在 BERT/分类的语境,不是自回归 LLM 预训练。
  - **训练动力学**:Xia2022 认为困惑度(ppl)变化小就代表"已学会";本文**修正**了这一点——存在一类"hard token resist convergence"(抗收敛的硬 token):因为数据本身固有的随机性(高 aleatoric 噪声),它们的 loss 会持续偏高、学不下去。
  - 作者称**自己是首个把 token 级数据选择用于 LLM 预训练**的工作。【附录B.2,L1742-1772】
- 动机链:训所有 token 既浪费又被噪声拖累 → 观察 token loss 的训练动力学,发现真正在有效下降的 token 只占约 26%(H→L) → 既然多数 token 要么早已学会(51% L→L)、要么是学不会的噪声(11% H→H) → 那就只在"该学且能学"的 token 上施损 → 这需要一个"理想分布"的参照来判定哪些该学 → 于是用 RM 算 excess loss、选 top-k%。
- 与最近邻工作的Δ:与 RHO-LOSS 的差异见上三点;与标准 CLM(因果语言建模,即常规自回归预训练)相比,SLM 只换了损失掩码(只对 top-k% excess-loss token 算损失),其余流程完全一样。关键点:**把"数据选择"从文档级下沉到 token 级,并用相对参照(excess loss)规避"高 loss = 噪声"这个陷阱**。

## 怎么做(到"读完能复现"的粒度)
> 一句话导读:先用一个观察实验证明"token 学习动力学高度异质"(只有少数 token 在有效学习),再据此设计三步流水线——训 RM、用 RM 打分、只对高 excess-loss 的 token 回传损失。

### A. 动机实证:token loss 的四类训练动力学(§2.1)
做法:持续预训练 TinyLlama-1B(15B OpenWebMath,每训 1B token 存一个 checkpoint),在约 320K token 的验证集上,按每个 token 的 loss 随训练的轨迹把 token 分成四类(图3a):
- **L→L(51%)**:始终低 loss,即**已学会**。
- **H→L(26%)**:loss 显著下降,即**真正在有效学习的那少数**。
- **H→H(11%)**:持续高 loss、抗收敛,多因数据固有的**高 aleatoric 不确定性(即噪声)**。
- **L→H(12%)**:训练中 loss 反而上升。

关键观察:大量 L→L、H→H token 的 loss 在训练中**剧烈波动、抗收敛**(图3b/c),且这些内容多为噪声(附录D.2)。结论:总 loss 的平滑下降,掩盖了 token 之间复杂的异质动力学;只要选对 token,就能稳定训练轨迹、提升数据效率。【§2.1,图3】

### B. SLM 三步流水线(§2.2,图4;数据流动:RM → token 打分 → 选择性回传)
- **Step 1 训参考模型 RM**:在反映"理想分布"的 curated 高质量语料上,用标准交叉熵训 RM。RM 的 token 概率定义参考 loss
\(\displaystyle \mathcal L_{\text{RM}}(x_i)=-\log P(x_i\mid x_{<i}).\)
- **Step 2 用 RM 给大语料每个 token 打分**:对预训练语料每个 token 算 \(\mathcal L_{\text{RM}}(x_i)\)(可离线一次算完缓存)。
- **Step 3 选择性训练目标模型**:标准 CLM 的目标是 \(\mathcal L_{\text{CLM}}(\theta)=-\frac1N\sum_{i=1}^N\log P(x_i\mid x_{<i};\theta)\)(对所有 token 取平均)。SLM 改为只在高 excess loss 的 token 上回传。**excess loss** 定义为:
\(\displaystyle \mathcal L_\Delta(x_i)=\mathcal L_\theta(x_i)-\mathcal L_{\text{RM}}(x_i),\)
含义即上文所说"当前训练模型还没学会(\(\mathcal L_\theta\) 高)、但 RM 这个理想分布认为该会(\(\mathcal L_{\text{RM}}\) 低)"的那个差。再引入选择比例 \(k\%\),只对一个 batch 内 excess loss 排名 top-\(k\%\) 的 token 算交叉熵:
\(\displaystyle \mathcal L_{\text{SLM}}(\theta)=-\frac{1}{N\cdot k\%}\sum_{i=1}^{N}\mathbb I_{k\%}(x_i)\cdot\log P(x_i\mid x_{<i};\theta),\qquad \mathbb I_{k\%}(x_i)=\begin{cases}1,&x_i\ \text{在 top-}k\%\ \text{(按打分}\ S(x_i)\text{)}\\0,&\text{otherwise}\end{cases}\)
默认打分函数 \(S=\mathcal L_\Delta\)。**关键工程**:整段序列仍照常喂进前向(保持上下文与注意力完整、不破坏语言连贯),只在**反向传播**时把非选中 token 的损失掩掉。所以它"无额外预训练开销、易集成"——本质只是把同样的梯度预算集中到更值得学的 token 上。【§2.2 式1-5】

### C. 各组件必要性(消融)
- **参考模型/excess loss 必要但可降级**(§2.1 动机 + §3.4):去掉外部高质量 RM、改用 **self-referencing**(先在目标语料本身上训一个 RM 再据此选 token),仍有提升(数学/通用平均 +3.3%,Table3);但若纯按训练模型**自身 loss** 来选,就会选到噪声(即 §2.1 的 H→H 类)。可见"相对参照"这一步是必要的。
- **token 选择本身是因果有效的**(图6/7):SLM 让"被选 token"的 loss 降得更多,且这个降幅与下游性能呈幂律**正**相关;反过来,**未被选 token 的 loss 与下游性能负相关**(图7)。这就实证了"压低所有 token 的 loss 并非必要、甚至有害"。
- **选择比例 \(k\%\)**(§3.4 图9 + 训练配置):TinyLlama-1.1B 取 **60%**、Mistral-7B 取 **70%** 最佳;比例太高就接近普通 CLM,太低又会丢掉有用 token。
- **token 选择是动态变化的**(图8/14):越后期的 checkpoint 所选的 token,在后期 ppl 更高、前期更低——说明模型会先去优化"可学空间大"的 token;还观察到被选 token 出现了 sample-wise 的"double descent"(双下降)现象。
- **被选 token 的内容**(附录G.1):多与数学相关——印证 SLM 确实在原始语料里自动聚焦到了数学相关的部分。

### D. 训练/数据/评测(复现锚点)
- **RM 数据**:数学 RM 用 0.5B 高质量数学 token(GPT 合成 + 人工 curated);通用 RM 用 1.9B(Tulu-v2 / OpenHermes-2.5)。RM 训 3 epoch,lr 5e-5(1B)/1e-5(7B),cosine decay,max seq 2048(1B)/4096(7B),**RM 与持续预训练模型用同一 base 初始化**。
- **持续预训练**:数学 15B OpenWebMath(TinyLlama-1.1B / Mistral-7B);通用 80B token(TinyLlama);batch 统一 1M token;1B 模型约 19 小时。
- **评测**:9 个数学任务(GSM8K/MATH/SVAMP/ASDiv/MAWPS/TAB/MQA/MMLU-STEM/SAT,少样本 CoT)+ 通用 15 任务。**关键数字(Table1)**:数学持续预训练 RHO-1-1B 平均 38.1(TinyLlama-CT baseline 21.6,GSM8K +23.4、MATH +11.6);RHO-1-7B(Mistral-CT 底,10.5B 被选 token)平均 66.2,Mistral-CT 55.8;微调后 RHO-1-7B MATH 51.8、1B 40.6(首个 >40% 的 1B,逼近早期 GPT-4 CoT 42.5%);用 15B token 追平用 500B 的 DeepSeekMath-7B;通用域 80B token 平均 +6.8%(代码/数学增益 >10%);self-reference +3.3%。token 量口径标注清楚(RHO-1 只算被选 token,故"3% token 追平")。【Table1/3,§3】

## 靠不靠谱
> 一句话导读:对照公平、token 选择的因果有效性也证得很干净;主要软肋是作者自己承认的——只在 ≤7B 小模型、数学窄域、且有理想 RM 的"甜区"里验证过,可扩展性与泛化性都还是开放问题。

- baseline 公平吗:与 CLM(即 TinyLlama-CT/Mistral-CT,同语料、同 token 量)直接对比,公平;与 DeepSeekMath/LLemma/Minerva 等比较时按 token 量对齐("3% token 追平"),论据有力。
- 看着强但没回答核心问题(作者自承,§C 限制),三点:
  1. **泛化性**——纯 SLM 会快速收敛到 RM 所聚焦的那个域,未选 token 的 loss 会显著上升;目前没观察到偏置等不良后果,但作者建议未来混入通用预训练损失以防过拟合(Goodhart 效应),并扩大 RM 语料;
  2. **可扩展性**——只在 ≤7B 模型、<100B token 上验证过;很大的模型/海量语料可能自然就习得了"压缩有用数据"的归纳偏置,届时 SLM 是否还有增益未知;
  3. **RM 必要性**——需要一个高质量 RM(虽可用 self-reference 降级,但仍是依赖)。
- 假设与失效边界:
  - 【原文】① 必须有一个反映"理想/目标分布"的高质量 RM(或退而用 self-reference);② 验证仅限 ≤7B、<100B token、数学/通用域;③ 纯 SLM 会让未选 token 的 loss 上升、向 RM 域收敛(§C)。
  - 【推断】SLM 的 token 选择是**静态/离线**的:RM 固定不变,excess loss 是单向的(ref − train)。这不像 OPD 那样在学生自己采样的状态上动态对齐——所以当目标分布与下游任务不一致时,RM 选错 token 的风险会被放大;另外 \(k\%\) 是启发式定的,跨域可能需要重新调。
- 祛魅总结:真贡献有二——① 揭示了 token 级 loss 的训练动力学(四类 token、噪声 token 抗收敛),挑战"训所有 token"的惯例;② 用 excess loss 做 token 级选择,在数学持续预训练上拿到极高的 token 效率(3% token 追平 DeepSeekMath)。
  - 【推断】可能被高估的:增益或许强绑定"小模型 + 数学窄域 + 有理想 RM"这个甜区(作者自己也列了这些限制);而且 excess loss 的数学形式 RHO-LOSS 早已有之,本文的创新更多在"token 级 + 预训练规模 + 训练动力学动机"的落地,而非全新理论。
  - 被低估的:它作为"token 异质性 / token 级信用分配"先验的思想价值——是后续众多 token 级蒸馏/加权工作的精神源头之一。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=参考模型给的 token excess loss(ref−train)→选 top-k% token | **改什么**=目标模型参数 θ(只在被选 token 上回传 CLM 损失) | **何时改**=持续预训练阶段;选择随 checkpoint 动态变(每个 batch 按当前 excess loss 排名) | **免梯度?**=N/A(本身是预训练监督学习,非 RL;无策略梯度) | **记忆-技能生命周期**=无显式记忆/技能库 | **防遗忘机制**=无专门机制;反而 §C 指出纯 SLM 会让未选 token loss 上升、向 RM 域收敛(有"窄化/遗忘通用能力"风险),建议混通用预训练损失缓解
- ⑦ 开源代码+框架/harness:https://github.com/microsoft/rho(v1 元信息已核-真实)。**重要边界**:该官方仓**只发布模型权重 + 评测代码**(rho-1/ 下仅 math-evaluation-harness 子模块[本地克隆为空/未初始化] + 复现输出 outputs.zip;README Quick Start 只给 evaluation run_eval.sh cot/tora)——**SLM 持续预训练/微调的训练代码并未开源**,社区需自行按论文 §2 实现 token-level excess-loss masking。故"框架"实质=权重+评测仓,核心训练栈不可核;模型在 HF: microsoft/rho-math-1b/7b-v0.1。约 6.4MB 已 clone。
- 💰 资源/成本与可扩展性:1B 模型 15B token 约 19 小时,batch 1M token;核心卖点即**token 效率**(达 baseline 快 5-10×;3% token 追平 DeepSeekMath-7B)。可扩展性作者明确未验证 >7B / >100B token(§C Scalability),是显式开放问题。【§3.1,§C】
- 🎯 对"探索-巩固"对标:**思想先验级支撑(token 异质性 + token 信用),非方法竞品**。
  - 映射:RHO-1 的核心信念("并非所有 token 都该学、要用参照分布选出值得学的 token")与 TSRD 的"token 异质性 / 教 path-selection(判断哪些 token 或步骤值得被脚手架接管)"高度同源。excess loss = ref − train 可视为一种**静态、离线的 token 级蒸馏信号**:把这里的"参考模型(理想分布的代言)"换成 OPD 里的 teacher,"用 excess loss 选 token"就对应 OPD 里"用 teacher 信号决定哪些 token 值得学"。
  - **可借组件**(三件):① excess-loss 式 token 打分,可作"哪些 token 该被脚手架/teacher 接管"的离线先验(便宜,且可与 OPD 的在线 overlap 信号互补);② "前向喂全序列、反向只在被选 token 回传"的稀疏掩码工程,正合 TSRD"稀疏脚手架/单点接管"的实现范式(不破坏上下文连贯,只集中梯度);③ token loss 训练动力学的四分类(H→H 噪声 / H→L 可学),可用于诊断 path-recovery 该针对哪类 token。
  - **缺口/差异**:RHO-1 是**静态离线选择 + 纯预训练监督**,既没有 on-policy(不在模型自己采样的状态上选)、也没有 teacher 分布对齐、没有 MTP 前瞻、没有记忆——与 TSRD 的"on-policy 自选 + teacher 脚手架 + MTP 前瞻"差好几层;且它的选择信号是单向的 excess loss,缺 OPD 那种"分布匹配"的语义。
  - 判定:**经典先验工作,提供了 token-level 选择/信用的思想与稀疏掩码工程,但不是 on-policy/蒸馏的直接对标**。
- 🔭 开放问题/未来方向:【原文】① 从 token 级视角改进 LLM 预训练值得深入(Conclusion);② 扩 RM 语料范围 + 扩预训练数据(§C);③ 验证 SLM 能否扩到很大模型/海量数据(§C Scalability);④ 打分函数不限 excess loss,可探索其他(附录H)。【推断】把静态 excess-loss 选择升级为 on-policy 动态选择(在学生采样状态上算,逼近 OPD)、用 teacher 分布替代 RM 做 token 级蒸馏选择、把 token 选择与 MTP 前瞻结合(用前瞻判定"将走偏"的关键 token 优先施损)、研究"选择窄化导致通用能力遗忘"的防护(与持续学习/防遗忘线对接)。

RETURN: rho1|读PDF?是(_txt 1000+行,核到§2.1四类token动力学+§2.2 SLM式1-5+§3训练配置/Table1)|加厚?是(方法扩为A-D四块+RM/训练超参,相关工作RHO-LOSS三处差异精确化,5条公式转MathJax)|LaTeX公式条数 4|待核数 0

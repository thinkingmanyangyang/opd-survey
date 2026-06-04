on_policy_sft | On-Policy Supervised Fine-Tuning for Efficient Reasoning | 东方理工(EIT,宁波)·香港理工·Paris Dauphine-PSL·上海交大·腾讯混元/AI Lab·LMU 慕尼黑(Anhao Zhao, Ziyang Chen, Junlong Tong, Yingqi Fan, Fanghua Ye, Shuhao Li, Yunpu Ma, Wenjie Li, Xiaoyu Shen*) | arXiv 2602.13407 v1 · Preprint Feb 17, 2026 | 主题线 L2(统一 SFT-RL/GFT 类)，兼 L1(on-policy 自蒸馏味)、L6(高效推理/token) · 相关性 高

**原始论文**：https://arxiv.org/abs/2602.13407 （代码 https://github.com/EIT-NLP/On-Policy-SFT）

## 一眼看懂
> 一句话导读：所谓"高效推理"就是让模型把思维链(CoT,模型解题时写出的中间推理步骤)写短一点又不掉准确率;本文发现,要做这件事根本不需要复杂的 RL,把 GRPO 里几个零件删掉后它就退化成一个普通的"挑好样本来学"的 SFT,而且效果还更好。

- 🟦 TL;DR：做"高效推理"(目标是缩短 CoT、同时保住准确率)时,主流做法是把"长度惩罚"做成额外奖励塞进 GRPO,结果训练不稳、精度-效率的权衡也不理想。本文做的是**原理化简**——它先指出高效推理有两条 RLHF 任务所没有的性质:
  - 正确性与长度**都能直接用规则验证**(不像 RLHF 要靠一个学出来的奖励模型);
  - 它本质是个**多奖励**问题(同时要"答对"和"够短")。
  据此本文论证 GRPO 里有两处零件其实是多余或失配的:
  - (1)**KL 正则冗余**:既然没有可学习的奖励模型,就不存在"模型钻奖励模型空子(reward over-optimization)"的风险,那道用来防钻空子的 KL 约束自然就没必要;
  - (2)**组内归一化失配**:在多奖励设定下,GRPO 的组内标准化会放大那些"没信息量"样本的梯度,还会混淆不同来源的奖励。
  把这两项删掉、并把长度惩罚简化成**截断**(响应一超长就直接给零奖励),GRPO 的训练目标就**退化为在模型自己生成的数据上做交叉熵 SFT**——而这批数据天然被"答对 + 没超长(即够简洁)"过滤过。作者把这个方法取名 **on-policy SFT**(on-policy 指训练数据始终来自当前策略自己采样,而非外部固定数据集)。它极简,却在 5 个数学基准上画出了 accuracy-efficiency 的 Pareto 前沿:CoT 缩短约 80%、精度持平,训练显存/墙钟降约 50%、收敛快约 70%。【原文 §Abstract / §3 / Fig.1 / Table 1】
- 最巧的一步：**论证"做完三处化简后,策略梯度(PG)目标 ≡ 过滤式 SFT"**。三处化简是:去 KL、去组内归一化、把长度惩罚换成截断奖励;化简后整个目标就变成 reward-free(不带奖励项)的最大似然(见下方 Eq.7→9 推导链)。如果抽掉这一刀(也就是保留 KL、保留归一化、保留复杂的长度奖励),就退回原版多奖励 GRPO 的病态——训练不稳、权衡次优。这一刀的意义在于把"高效推理需要复杂 RL"祛魅成"其实是个在 on-policy 数据上的简单 SFT"。
  但要注意:**真正起作用的并不是这个化简形式,而是 on-policy 数据本身**。§5.2 的消融说明了这点:同样的过滤式 SFT 目标,一旦把数据换成 off-policy 的固定数据,就大幅掉点。所以"最巧的一步"在叙事层面是化简,在机理层面其实是"坚持 on-policy 重采样"。【原文 §3 / §5.2 / Fig.6】

## 为什么做
> 一句话导读：模型现在解题动辄写一大段又臭又长的推理,推理时又费算力又费显存;大家想让它写短点,做法是往 RL 奖励里叠各种"长度惩罚",可这些奖励越设越复杂、训练越来越不稳。本文质疑:这套从 GRPO 照搬来的复杂奖励,跟"缩短推理"这件事的内在结构真的对得上吗?

- 研究背景：现在的大推理模型(LRM)大多用 GRPO 这类 RL 训练(沿用 DeepSeek-R1 的配方,Guo 2025),会产生很长的 CoT,推理时算力和显存开销都很大。于是"高效推理"方向涌现出大量工作,做法都是在奖励里叠加长度惩罚或简洁奖励(如 ThinkPrune、O1-Pruner、L1、LASER、ER-RL 等)。而且其中长度奖励项 \(R_{\text{Len}}(o\mid q)\) 与它的权重 \(\gamma(o\mid q)\) 设计得越来越复杂——比如按正确性或难度条件性地激活、用组内 mean/median/max 的长度统计量等。这些复杂设计都可以统一写成一个多奖励形式(即 Eq.3,见下)。【原文 §1 / §2(Reward Shaping for Efficient Reasoning) / Appendix B】
- 解决的具体痛点：
  - ① RL-based 高效推理虽然能大幅缩短 CoT,但常常伴随**精度下降**,而且下降幅度还随任务复杂度变化,导致权衡次优;
  - ② 把多个奖励(正确 + 简洁)放一起联合优化,会让**训练不稳、收敛慢、对超参敏感**(原文逐一引了 Liu 2025a / Li 2025a / Arora&Zanette / Chen 2025b);
  - ③ 这套复杂的奖励塑形是被不加甄别地从 GRPO 沿用过来的,与高效推理本身的内在结构未必对得上。【原文 §1 / §3】
- 相关工作 & 各自不足（来龙去脉 + 三条并行路线 + 各自短板 + 站谁肩上）：高效推理目前有三条并行路线,本文与每条的精确差异如下。
  - **多奖励 GRPO 高效推理族**(O1-Pruner / L1(Aggarwal&Welleck) / ThinkPrune / LASER / ER-RL / Arora&Zanette 等):做法是在 GRPO 奖励里叠一个长度项 \(R_{\text{Len}}\)。短板是复杂的 \(R_{\text{Len}}+\gamma\) 设计带来不稳、权衡次优、超参敏感。本文与它的精确差异:**删掉 KL、删掉组内归一化、把 \(R_{\text{Len}}\) 退化为截断**,并证明在这个简化奖励族下,PG 目标 ≡ 过滤式 SFT。
  - **training-free 压缩**(CoD / DEER / CRST / TokenSkip / StepEntropy):不训练,在推理时直接裁。短板是省事但压缩量有限,而且常牺牲精度(Table 1 里 TokenSkip 的压缩率甚至 >100%,也就是反而更长了)。
  - **SFT-based 高效推理**(在固定数据上对短 CoT 做 SFT):它和本文一样都是 reward-free 的 SFT 目标,是**最近邻**。精确差异在数据来源:它用的是 **off-policy 固定数据,§5.2 证明这样会掉点**;而本文坚持 **on-policy 重采样**。
  - **站谁肩上**:
    - ① GRPO(Shao 2024)的多奖励形式;
    - ② "优化目标的设计比复杂的奖励塑形更重要"这一近期发现(Liu 2025a 提出「最简截断已够」、He 2025);
    - ③ on-policy 蒸馏(Lu&Lab 2025 等)启发了本文 Appendix K 的诊断工具。
  - **共性缺口**:此前没人指出"高效推理的复杂 RL 其实可以化简成 on-policy 过滤式 SFT",也没人把增益正确归因到 on-policy 数据(而非奖励塑形)。【原文 §1 / §2 / §4(Related Work) / §5.2】
- 动机链（一步步推下来）：
  - 现状:高效推理 = 往 GRPO 里塞复杂的长度奖励;
  - 质疑:把 GRPO 不加甄别地套到高效推理上,真的合理吗?
  - 原理分析:发现两处错配——KL 冗余 + 组内归一化失配;
  - 化简:去掉这两项、再把长度奖励换成截断,于是 PG 目标 ≡ 过滤式 SFT;
  - 结论:那就直接做 on-policy SFT,更简单、更省、Pareto 更优。【原文 §1 / §3】
- 与最近邻工作的 Δ：
  - 相对**多奖励 GRPO 族**:on-policy SFT 删掉了 KL 项 \(\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})\)、删掉了组内归一化 \(\hat A_{i,t}=(R_i-\mathrm{mean})/\mathrm{std}\)、把长度奖励简化成截断;差别在于**用最简的过滤式 SFT 替代复杂的奖励塑形**,且证明二者在该简化奖励族下等价。
  - 相对 **SFT-based 高效推理**:两者同是 reward-free 的 SFT 目标,差别在于本文**坚持 on-policy 重采样**(每一轮都用当前策略重新采样);§5.2 证明这正是 SFT-baseline 掉点、而 on-policy SFT 不掉点的根因。
  - 相对 **on-policy 蒸馏(opd_blog 等)**:本文**显式把自己归入 on-policy 数据范式**(§5.2 / Appendix K 引 Lu & Lab 2025、Shenfeld 2026、Zhao 2026、Hübotter 2026),但**没有 teacher**——它是"自己当自己的过滤器"、带自蒸馏味的 SFT,而非借助教师分布做 KL 蒸馏。【原文 §3 / §5.2 / Appendix K】

## 怎么做 + 靠不靠谱
> 一句话导读：流程其实就一个循环——模型对每道题自己采一批答案,留下"答对且没超长"的那些,然后在这些答案上做普通的交叉熵学习(就是 SFT),下一轮再用更新后的模型重新采。关键的几个工程细节(温度必须=1、长度归一化用 batch 内最长、不要 KL)都是为了"让训练数据始终等于自己当前的分布"这一件事服务。

- 方法流水线（输入→输出,读完即可复现）：输入是当前策略 \(\pi_{\theta_{\text{old}}}\) 和一个数学题集 \(\mathcal D\),一轮分四步:
  - ① **Rollout(采样)**:对每道题 \(q\) 采 \(G\) 条回答 \(\{o_i\}_{i=1}^G\)(温度严格 =1.0;脚本里 ROLLOUT_N=32);
  - ② **评分 + 过滤**:用 verifier 判对错;**截断式奖励**让超过长度 \(L\) 的响应得零奖励(也就是被当成"错误";\(L\)=max_response_length,脚本里设 3500);接着 `_data_filter` 按 uid 把同题的回答分组,**丢掉错误的、留下全部正确的**;若某题全错就整条丢掉;最后把保留数量对 dp_size 取整;
  - ③ **训练**:把 verl PPO actor 里的 PG 损失换成交叉熵 \(-\log\pi_\theta\) 的 token-mean(没有 advantage、没有 clip、`use_kl_loss=False`);长度归一化用 batch 内最长的那条响应(见下文 length-bias correction);
  - ④ **同分布训练-评测**:训练和评测用同一套 prompt 模板,避免分布漂移造成的混淆;
  - 然后迭代回 ①。【原文 §3.3 / §5.3 + v1 仓库核】
- 逐组件必要性（每模块干啥 / 怎么咬合 / 有无消融）：
  - **on-policy 重采样(真正的核心)**:每一轮都用当前策略重新 rollout,让训练分布 = 部署分布。如果没有它(改用 off-policy 固定数据),§5.2 消融显示效果会变成"只温和缩短长度、精度却大幅下降";而 on-policy SFT 是快速缩短且保精度。**有消融(Fig.6,7B/MATH-500)**,这是全文最关键的对照。【原文 §5.2 / Fig.6】
  - **去 KL 正则**:在损失里删掉 \(-\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})\)(对应 `use_kl_loss=False`)。好处有两点:省去维护参考模型 \(\pi_{\text{ref}}\) 的显存与算力;不再过度约束更新。论证依据是:高效推理没有可学习的奖励模型,所以 KL "防钻空子(over-optimization)"的本意在这里失效了——正确性和长度都能直接验证,不存在"分布漂移导致奖励不可靠"的隐患。【原文 §3.1(KL Divergence)】
  - **去组内归一化**:不再做 \(\hat A_{i,t}=(R_i-\mathrm{mean})/\mathrm{std}\) 这个标准化,改成用 baseline 直接优化。如果保留它,在多奖励下会有两个毛病:
    - (a) 当组内奖励差异很小(比如都答对、且都很长)时,分母 std 很小,会**放大那些没信息量样本的梯度**;
    - (b) **混淆奖励的组成**——原文举的例子是:奖励向量 \((0,1)\) 与 \((0,0)\)(对应标量 1 与 0)归一化后得到 advantage \((-0.7071, 0.7071)\);而组成完全不同的 \((1,1)\) 与 \((0,0)\) 归一化后,竟得到**相同的** advantage。【原文 §3.1(Group-wise Reward Normalization)】
  - **截断奖励(最简的长度惩罚)**:规则是当且仅当响应在固定长度 \(L\) 内答对时 \(\hat R_i=1\),否则 \(\hat R_i=0\)。这沿用 Liu 2025a 的「最简截断已够」。它是"简洁性过滤"的**间接实现**:超长响应会被归入"错误"、然后被 `_data_filter` 丢掉。**注意一个实现落差**:仓库里的 `_data_filter` 并不显式按 length 字段去筛(length 字段被提取出来,但代码注释写着 "not used"),简洁性其实是靠截断奖励间接达成的(v1 已核实)。【原文 §3.1(Truncation)/ v1 仓库核】
  - **length bias correction(长度偏置校正)**:把梯度里的归一项 \(1/|o_i|\) 换成 \(1/\max_j|o_j|\)(即用 batch 内最长那条来归一)。目的是避免短响应(可能是噪声)被赋予过大梯度、把训练带偏。**有消融**(Appendix,with/without 对照),作者称它对稳定长度控制是必要的。【原文 §3.2(Eq.9 推导)/ §5.3 / Appendix】
  - **rollout 温度 =1.0**:这既是经验指南也有理论支撑。温度 \(T\neq 1\) 会让 rollout 偏离 \(\pi_{\theta_{\text{old}}}\),从而**变成 off-policy**——Appendix J 的 Eq.42-43 证明:只有当 \(T=1\) 时,温度缩放后的分布才等于 \(\pi_{\theta_{\text{old}}}\);只要 \(T\neq1\),就有 \(\Pr_T\neq\pi_{\theta_{\text{old}}}\),此时 Eq.9 的期望就变成对 \(\Pr_T\) 求,而非对当前策略求。实验上:低温(0.3/0.6)会过度缩短并掉点,高温(1.2/1.5)则 loss 单调上升、不收敛。**有消融(Fig.7/Fig.13)**。【原文 §5.3 / Appendix J / Eq.42-43】
  - **rollout 数缩放 & max length**:在 \(T=1\) 固定的前提下,增大 rollout 数会让结果更短且仍保精度(Fig.8);max length 的取值则有个权衡——3k 最省但略掉点,4k 最贵最准,3.5k 是平衡点(能匹配 4k 的精度),默认取 3.5k(Fig.10)。【原文 §5.3 / Fig.8 / Fig.10】
- 关键机制/公式（真实形式 + 直觉,均从 PDF 抄准）：
  - **GRPO 组内优势(Eq.1)**:\(\displaystyle \hat A_{i,t}=\frac{R_i-\mathrm{mean}(\{R_i\}_{i=1}^G)}{\mathrm{std}(\{R_i\}_{i=1}^G)}.\) 这是被本文删掉的"组内归一化"——直觉:它为单奖励设计,多奖励下标准化丢信息。
  - **GRPO clipped 目标(Eq.2)**:\(\displaystyle J_{\text{GRPO}}(\theta)=\mathbb E_{(q,a)\sim\mathcal D,\,\{o_i\}\sim\pi_{\theta_{\text{old}}}}\!\Big[\tfrac1G\textstyle\sum_{i=1}^G\tfrac1{|o_i|}\textstyle\sum_{t=1}^{|o_i|}\min\big(r_{i,t}(\theta)\hat A_{i,t},\,\mathrm{clip}(r_{i,t}(\theta),1-\varepsilon,1+\varepsilon)\hat A_{i,t}\big)-\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})\Big],\) 其中重要性比 \(r_{i,t}(\theta)=\dfrac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q,o_{i,<t})}\)。本文删掉末项 \(-\beta\,\mathrm{KL}\)。
  - **多奖励统一形式(Eq.3)**:\(\displaystyle R_{\text{Eff}}(o\mid q)=R_{\text{Acc}}(o\mid q)+\gamma(o\mid q)\,R_{\text{Len}}(o\mid q),\) \(\gamma\) 控正确性-效率权衡。本文把 \(\gamma R_{\text{Len}}\) 退化成"超长判零分"的截断。
  - **化简到 SFT 的核心推导链(Eq.7→8→9,全文枢纽)**:这条推导分三步。
    - 第一步:先设 \(\hat R_{i,t}=R_i\)(这是 §3.2 的 exploitation 设定),使 GRPO 梯度等价于 REINFORCE(Appendix D);又因为奖励是逐 token 共享的,把对 \(t\) 的依赖丢掉,得到 \(\displaystyle \nabla_\theta J(\theta)=\mathbb E\Big[\tfrac1G\textstyle\sum_i \hat R_i\,\tfrac1{|o_i|}\textstyle\sum_t \nabla_\theta\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})\Big]\quad(\text{Eq.7})。\)
    - 第二步:代入二元截断奖励 \(\hat R_i\in\{0,1\}\),用一个集合指示函数 \(\mathbb 1\{o_i\in C_L\}\) 来表达(其中 \(C_L\) 是"答对且长度在 \(L\) 内"的响应集合),得到 \(\displaystyle \nabla_\theta J(\theta)=\mathbb E\Big[\tfrac1G\textstyle\sum_i \mathbb 1\{o_i\in C_L\}\,\tfrac1{|o_i|}\textstyle\sum_t \nabla_\theta\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})\Big]\quad(\text{Eq.8})。\) 这个式子与标准 SFT 的 MLE 梯度 \(\nabla_\theta J_{\text{SFT}}(\theta)=\mathbb E_{(q,o)\sim\mathcal D_{\text{SFT}}}\big[\tfrac1{|o|}\sum_t\nabla_\theta\log\pi_\theta(o_t\mid q,o_{<t})\big]\) **函数形式完全相同**,唯一区别是数据分布:这里的指示函数 \(\mathbb 1\{o_i\in C_L\}\) 充当"答对 + 不超长"的选择器,而且数据来自 on-policy 的 \(\pi_{\theta_{\text{old}}}\),而不是某个固定数据集 \(\mathcal D_{\text{SFT}}\)。
    - 第三步:最后把归一项 \(1/|o_i|\) 换成 \(1/\max_j|o_j|\)(即前面说的 length-bias 校正),得到**最终目标(Eq.9)**:\(\displaystyle J(\theta)=c_L\,\mathbb E_{o_i\sim \mathcal D_L^{+}}\Big[\tfrac1{\max_j|o_j|}\textstyle\sum_{t=1}^{|o_i|}\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})\Big],\) 其中数据集 \(\mathcal D_L^{+}=\{o_i\mid (q,a)\sim\mathcal D,\ o_i\sim\pi_{\theta_{\text{old}}}(\cdot\mid q),\ o_i\in C_L\}\),常数 \(c_L=\mathbb E[\mathbb 1\{o_i\in C_L\}]\) 与 \(\theta\) 无关。
    - **直觉**:删掉 KL 和归一化、再把长度惩罚退化成"超长直接判零分"之后,策略梯度目标里就只剩下一件事——"在自己采的、且答对又不超长的轨迹上做最大似然"。这恰恰就是 SFT。
  - **诚实归因(§5.2)**:论文明说真正 work 的不是"化简成 SFT"这个形式,而是"数据始终来自当前策略"——同样的 SFT 目标喂固定数据(off-policy)就崩。直觉:"复杂奖励是障眼法,on-policy 数据才是药效。"
  - **Appendix K 诊断工具(Eq.44)**:这是受 on-policy 蒸馏启发做的"低效 token 诊断"。做法是:先从 base 模型采一条 CoT,再把同一个前缀 \(x_{1:t}\) 喂给高效模型做 teacher forcing(即强制它读这段前缀),然后在每个位置上量两者 next-token 分布之间的 token 级 KL:\(\displaystyle D_t=\mathrm{KL}\big(p_{\text{orig}}(\cdot\mid x_{1:t})\,\big\|\,p_{\text{eff}}(\cdot\mid x_{1:t})\big)=\textstyle\sum_{v\in V}p_{\text{orig}}(v\mid x_{1:t})\big(\log p_{\text{orig}}(v\mid x_{1:t})-\log p_{\text{eff}}(v\mid x_{1:t})\big)。\) \(D_t\) 大的那些 token,就是"把模型带向冗长推理的元凶"。Fig.14 的词云显示,divergence 最大的是 "Wait"/"Alternatively"/"Hmm" 这类犹豫/回溯词,以及第一人称 "I"/"me"(而高效变体更偏好复数的 "we"/"us")。【原文 Appendix K / Eq.44 / Fig.14-18】
- 实验与证据：
  - **核心结果(Table 1 / Fig.1)**:
    - 1.5B(DeepSeek-R1-Distill-Qwen-1.5B):Acc 59.9% / Pass@N 73.6%(略超原模型的 59.0/73.5),平均生成长度从 10,178 token 降到 2,186 token(约 80% 缩短),Eff=2.74%,高于最强 RL baseline 的 2.55%;
    - 7B:Eff=2.97%(全场最高),长度较原模型缩约 70%;
    - 在不同长度预算下都持续位于 Pareto 前沿,优于 ThinkPrune/O1-Pruner/L1/LASER/ER-RL 等 10 个 baseline(涵盖 training-free / SFT-based / RL-based 三类)。【原文 §Abstract / Table 1 / Fig.1-2】
  - **训练侧**:每步显存与墙钟约降 50%,收敛比 RL 快约 70%;长度控制也更稳(多次生成的长度方差更低——§5.1 用归一化标准差来度量,Fig.5 对比了 ThinkPrune)。【原文 §Abstract / §1 / §5.1】
  - **机制消融(§5.2 / Fig.6)**:对照的 off-policy 变体设置为——同样的采样/过滤、每题 8 条响应、3500-token 限、总题量匹配到 50 步、用固定数据训 1 epoch ×7 iter(约 350 步)。结果是它只温和缩短、且大幅掉点;而 on-policy SFT 快速收敛并保精度。由此说明**增益的主因是 on-policy 数据**,这与"RL 的效果来自 on-policy 数据而非复杂算法"的近期发现一致(引 Chen 2025a / Shenfeld 2026 / Zhao 2026 / Hübotter 2026)。【原文 §5.2 / Fig.6】
  - baseline 公平吗:与 10 个 baseline 在同样 5 个基准(GSM8K/MATH-500/AMC23/AIME24/AIME25)上对比,且用"同分布训练-评测"的设计排除了 prompt 不匹配的混淆,较规范;综合指标 Eff 的定义见 Appendix。【原文 §4 / Table 1】
  - "看着强但没回答核心问题":三个核心点(化简成 SFT、画出 Pareto 前沿、on-policy 是主因)都给了证据;但 Eff 这类综合指标的定义对结论比较敏感,跨方法的可比性需要谨慎。
- 假设与失效边界：
  - 【原文 §3.1】"等价于 SFT"依赖一串简化前提:去 KL、去归一化、用**截断式**长度奖励、再加上 \(\hat R_{i,t}=R_i\) 的 exploitation 设定。也就是说它只"在某一特定的简化奖励族下成立",并不对任意长度奖励都普适。
  - 【原文 §5.3 / Appendix J / Eq.42-43】温度必须 =1.0 才能保持 on-policy。一旦 \(T\neq1\) 就变成 off-policy:低温会过度缩短并掉点,高温则不收敛。
  - 【推断】只在**数学推理、且正确性可验证**的场景下验证过。对开放式/不可验证的任务并不适用——因为一旦"正确性可直接验证"这个前提失效,前面"KL 冗余"的论证也跟着失效。依据是:全部 5 个基准都是数学题、reward 都靠 verifier。
  - 【原文 §1 + 推断】截断奖励里的 max length \(L\) 是个关键超参:设得过紧会牺牲难题精度(3k 就已略掉点,Fig.10)。论文也承认,精度随任务复杂度有边际下降的风险。
  - 【推断/已核】仓库的实际代码与论文叙述的"简洁性过滤"在实现方式上有表述落差:仓库是靠**截断奖励间接实现**的(length 字段虽被提取,但代码注释写着 "not used"),并不是一个显式的长度过滤器——读者容易误以为有一个专门的长度筛。
- 祛魅总结【推断】：
  - 真贡献:可概括为**祛魅 + 极简 + 正确归因**三连。
    - "两处错配"的论证(KL 冗余、以及组内归一化在多奖励下的歧义)清晰且配有举例:\((0,1)/(0,0)\) 与 \((1,1)/(0,0)\) 两组组成完全不同,归一化后却得到相同 advantage;
    - 它把高效推理的复杂 RL 化简成过滤式 SFT(Eq.7→9),又在 §5.2 里**诚实地把功劳归给 on-policy 数据、而非化简形式本身**——这种自我祛魅在 SFT-RL 统一这条线里很有价值;
    - Appendix K 的"低效 token 诊断"(Eq.44)是个有用的副产品。
  - 包装/被高估处:
    - "GRPO 退化为 SFT"容易被读成一个普适结论,实则它**绑定了特定的简化奖励族 + exploitation 设定**;
    - "conciseness 过滤"在论文与代码之间的实现落差容易误导读者;
    - 真正的 novelty 更多在"祛魅 + 归因"这一层,方法本身(过滤式 on-policy SFT)与 RFT / on-policy SFT 这一系工作高度同源(本文与 NFT/RFT/psft 等同属一族)。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：verifier 的正确性 + 截断式长度信号(超长判零);无奖励模型、无教师 logits。过滤后对保留轨迹做交叉熵(reward-free)。属"自蒸馏味的过滤式 on-policy SFT"。
  - **改什么**：只改训练目标(GRPO 的 PG loss → 交叉熵 Eq.9)+ 数据来源(坚持 on-policy 重采样);模型全量更新。
  - **何时改**：在线迭代——每轮用当前策略 rollout→过滤→交叉熵更新→进入下一轮。
  - **免梯度?**：损失侧无策略梯度/无 advantage/无 clip(就是 NLL),但仍是梯度优化;**去掉了 ref 模型**(省显存)。
  - **记忆-技能生命周期**：无外部记忆/技能库;"技能"=简洁正确的推理路径,固化进参数。无持续学习机制。
  - **防遗忘机制**：无显式防遗忘(且去掉 KL-to-ref);靠 on-policy 重采样维持分布不漂移。
- ⑦ 开源代码+框架/harness：https://github.com/EIT-NLP/On-Policy-SFT (`opsff/`→`opsft/` 子目录,已开源,本地已 clone)。**框架=veRL**(基于 verl 改造) + FSDP + vLLM rollout。recipe 在 `opsft/recipe/On_Policy_SFT/`:`dp_actor.py`(PG loss 换成交叉熵)、`on_policy_sft_trainer.py`(on-policy 流程 + `_data_filter`)、`main_on_policy_sft.py`、`fsdp_workers.py`、`config/on_policy_sft_trainer.yaml`。入口 `bash examples/On_Policy_SFT.sh`。关键超参(脚本):LR=1e-6、MAX_GEN_LENGTH=3500、ROLLOUT_N=32、temperature=1、batch=32、`use_kl_loss=False`、`loss_agg_mode=token-mean`、total_epochs=2、2×GPU。数据:内置 DeepScaleR/GSM8K/OpenThoughts3 train.parquet + 8 个评测 benchmark parquet。代码可得性高。【v1 仓库核查 + 本批 repo ls】
- 💰 资源/成本与可扩展性：核心卖点即成本——相对 RL 每步显存与墙钟降约 50%、收敛快约 70%;backbone R1-Distill-Qwen-1.5B/7B;脚本默认 2×GPU、LR=1e-6、total_epochs=2。【原文 §Abstract / §1 + v1 脚本】
- 🎯 对"探索-巩固"对标：**中-强支撑(巩固层:on-policy 数据是巩固之钥 + 低效 token 诊断接近"关键步")+ 可借诊断工具;但它不是脚手架/蒸馏方法**。判定依据如下:
  - ① **巩固对标**:on-policy SFT 把"在自己采的、走通且简洁的轨迹上做 MLE(Eq.9)"当作巩固,这与本课题"巩固=固化走通的有效路径"高度一致;§5.2 "on-policy 数据是主因"更为本课题"student 在 on-policy 自选轨迹上再固化"提供了直接证据(off-policy 固定数据会掉点,所以必须 on-policy)。
  - ② **"低效 token 诊断工具"(Appendix K,Eq.44)≈ 关键步定位**:它用 base 与高效模型在同一前缀上的 next-token KL \(D_t\) 去找"把模型带偏向冗长的元凶 token",这与本课题"识别关键步、定位 path-recovery 单点接管位置"在机制上同构——都是用分布差异来定位关键 token。
  - ③ 截断奖励本质是对"走偏成冗长路径"的惩罚,弱对应于"巩固简洁路径、抑制冗余试探"。
  - **可借组件**:
    - (a) on-policy 重采样 + 正确性过滤组成的极简巩固管线(去 KL、去归一化以省显存);
    - (b) **Appendix K 的 token 级 KL 诊断(Eq.44)** 可直接迁移为"用 teacher 与 student 的 next-token 散度定位关键步/低效步",再对这些步做稀疏脚手架介入。
  - **缺口**:
    - ① **无 teacher、无蒸馏**(它是自己当过滤器,不是教师 KL);
    - ② 信号只是**序列级的正确性 + 长度**,过滤也是粗粒度的"留下答对的整条",**而非 token 级的稀疏关键步介入**(虽然 Appendix K 有诊断工具,但训练 loss 仍是整条 NLL);
    - ③ **无 MTP/前瞻、无记忆/技能库**;
    - ④ 它只针对"高效推理"(把推理缩短),不针对"走通新难题"——按 opd_survey §7.1 的说法,这类 correct-only 的 on-policy 训练更像**压缩**而非**纠错**,与本课题"教 student 走通自己原本走不通的开头"的目标不同。
  - 一句话:**它给"on-policy 自巩固"提供了最干净的证据,外加一个可借的关键步诊断工具;但它缺 teacher 脚手架、稀疏接管与前瞻。**
- 🔭 开放问题/未来方向：
  - 【原文 §6 / §1 末】未来的高效推理应当重视简单与原理,而非堆砌算法复杂度;并把这套化简思路推广到更多任务。
  - 【推断】有三个可做的方向:
    - 把 Appendix K 的"token 级 KL 诊断(Eq.44)"从"找低效 token"升级为"找关键步并让 teacher 做稀疏介入",从而把本文(自蒸馏、整条过滤)与 OPD(有 teacher、token 级 KL)缝合;
    - 研究截断 max length \(L\) 随任务难度自适应,以避免难题精度下滑;
    - 把"on-policy 是主因"与 MTP 前瞻结合,用前瞻去挑选最值得重采样的 prompt。

RETURN: on_policy_sft|读到PDF=是(§Abstract/§1-3全文+Eq.1-3/Eq.7-9化简链/Table1/Fig.1-2/§5.1-5.3+Fig.5-10/Appendix J Eq.42-43温度off-policy/Appendix K Eq.44诊断+Fig.14词云)|L线=L2(兼L1/L6)|对标=中-强支撑(on-policy数据是巩固主因的硬证据,AppdxK的token级KL诊断Eq.44≈关键步定位可借;但无teacher脚手架/粗粒度整条过滤非稀疏接管/无MTP,且correct-only更像压缩非纠错)|残留待核=0

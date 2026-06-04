rethink_opd | Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe | 清华 THUNLP + 上海科技大学 + UIUC + 人大（Yaxuan Li/Yuxin Zuo/Bingxiang He 共一；Zhiyuan Liu/Ning Ding 通讯） | 2026-04-15 v2 · arXiv preprint cs.LG | 主题线 L1(OPD/自蒸馏)+L6(Token信用)·相关性 高

**原始论文**:https://arxiv.org/abs/2604.13016

## 一眼看懂
- 🟦 TL;DR:OPD(让学生自己采样、用 teacher 的逐 token log-prob 当稠密 reward 来纠正)已被 Qwen3/MiMo/GLM-5 大规模采用,但其训练动力学是黑箱,且很脆——**更强的 teacher 有时完全教不动学生,更弱的 teacher 反而成功**。本文系统拆解:① 现象学——OPD 成败由两条件决定:(a)师生"思维模式"要兼容(top-k 分布 overlap 高),(b)光分高/思维一致还不够,teacher 必须有学生训练时没见过的"新知识";② 机制——成功的 OPD 表现为"在学生访问状态上,师生 top-k 高概率 token 的 overlap 渐进上升、熵差收窄",且**这批 overlap token 仅占少量却集中 97-99% 概率质量,单独训它们就够**;③ recipe——两招救活失败的 OPD:off-policy cold start(先用 teacher rollout 做 SFT 拉高初始 overlap)+ teacher-aligned prompt(用 teacher 后训练数据的题/模板)。④ 代价——稠密 reward 的可靠性随轨迹变长而退化,失稳从轨迹尾部向前传播,暗示 OPD 难直接扩到长程/agentic。【Abstract,§1,图1】
- 最巧的一步:**§4.2 "只训 overlap token 就等于全量 top-k"的因果消融**。抽掉它,全文就只剩"overlap 上升与成功相关"的相关性观察;正是把 top-k 拆成 overlap / non-overlap 分别训(Overlap≈Student-TopK,Non-Overlap 显著弱)+ "overlap 自增强(reverse-KL 把质量越堆越多、把竞争 token 挤出 top-k)"才把"高概率 token 对齐"确立为 OPD 的**因果作用点**而非副产品。【§4.2,图7】

## 为什么做
- 研究背景:OPD 已成 LLM 后训练核心技术(Qwen3/MiMo/GLM-5 都用,Thinking Machines Lab 以极低 RL 成本复现 Qwen3 recipe);与离策略蒸馏(在固定 teacher 序列上训、有 exposure bias)不同,OPD 在学生实际访问状态上用 teacher 逐 token log-prob 当 dense reward,并被扩到自蒸馏(单模型当自己 teacher)。【§1,L67-82】
- 解决的具体痛点:存在显著失败模式——更强 teacher 反而完全教不动,更弱却能成功;少有研究解释 teacher 的 token 级信号为何/何时把学生分布推向期望方向、何时失败。【§1,L83-87】
- 相关工作的来龙去脉与精确短板(本条比 v1 更细,据 §7 三段):
  - **经典 KD / 序列级 KD**:Hinton 2015 把大模型 soft 分布教给小模型;Kim&Rush 2016 扩到序列级(在 teacher 生成序列上训学生),奠定 off-policy 蒸馏基线(TinyBERT/DistilBERT/MiniLM 均此族)。**共同短板**=train-inference 分布失配,即 exposure bias(Bengio 2015):学生在 teacher/参考序列上优化,推理时却要从自己分布生成,误差沿长生成累积——这正是"把蒸馏搬到学生 on-policy 分布"的动机。【§7 L1464-1476】
  - **OPD 谱系(站谁肩上)**:① **MiniLLM**(Gu 2023)首次为 LLM 形式化 OPD,用 reverse-KL + policy gradient,论证 reverse-KL 的 mode-seeking 阻止学生把质量摊到 teacher 认为不可能的区域;② **GKD**(Agarwal 2024)给统一框架,在 on/off-policy 数据与多种散度间插值;③ **Yang2026b**(本文最近邻理论)把 OPD 形式化为 dense KL-约束 RL 的特例,证明 teacher 逐 token log-ratio = 隐式 reward,且**把该 reward 放大超过标准权重可把学生推过 teacher 的性能上界**。OPD 已被工业后训练(Qwen3/MiMo/GLM-5 等多篇 2026)采用,并扩到 self-distillation(单模型借特权信息——ground-truth 解/执行反馈——当自己 teacher,Hübotter/Shenfeld/Zhao 2026 等十余篇)。**共同短板**=都在"展示 OPD 好"(稠密 reward、缓解 exposure bias),跨各种 objective/任务/师生对,**没系统分析何时/为何失败**。【§7 L1477-1493】
  - **capacity gap / distillability(并行路线)**:Cho&Hariharan 2019 发现 teacher 远强于学生时蒸馏反伤;Mirzadeh 2020 用中等 TA assistant 桥接;Busbridge 2025 给"蒸馏 scaling law"——学生 loss 是 teacher 质量/学生尺寸/数据量的幂律,存在**U 型容量区**(teacher 过强反降蒸馏效率);Li2025 记录"learnability gap"——小模型学强推理 teacher 的长 CoT 反不如简单法。**共同短板**=这些分析几乎全在 **off-policy KD**,OPD 的 capacity gap/distillability **仍未探**。【§7 L1494-1508】
- 动机链:OPD 火但黑箱、且观察到"强 teacher 失败"反常 → 需找成败条件(现象学) → 条件如何在 token 级体现(机制) → 既然 thinking-pattern gap 可训练缩小 → 给出修复 recipe → 但稠密 reward 真免费吗?查 reward 本身的可靠性边界(代价)。
- 与最近邻工作的精确Δ:vs **Yang2026b**(把 OPD 当 dense KL-RL、证明放大 teacher reward 可越过 teacher 上界)——本文不提新 objective,而是**诊断成败条件 + 揭示 overlap-token 作用机制 + 给救活配方**;vs RL 阵营对 GRPO 的拆解——本文把"成功签名"定位到 token 级 overlap 动力学(overlap ratio↑、overlap-token advantage→0、熵差↓),并用 reverse distillation 干净地把"训练动力学"与"benchmark 分数"**完全解耦**(§3.3:R1-7B 分更高却把学生拖回与弱 1.5B teacher 相同的退化水平)。关键论断:**"OPD 本质是学 thinking pattern,而非吃分数差"**——最反直觉、最有信息量。【§3.3,L579-598】

## 怎么做(到"读完能复现"的粒度)
> 本文是**诊断范式**而非新算法:核心"方法"=一套可监控的 token 级动态指标 + 三类监督粒度的 OPD 实现 + 一组受控因果实验。下面把每一块写到输入→输出、咬合关系、默认超参。

### A. OPD 的三种监督粒度(数据流动 + 显存代价)
统一目标=**学生自采轨迹上的序列级 reverse-KL**:给 prompt \(x\sim\mathcal D_x\),学生 autoregressive 采样 \(\hat y=(\hat y_1,\dots,\hat y_T)\sim\pi_\theta(\cdot\mid x)\);在学生生成的前缀 \(\hat y_{<t}\) 上同时取学生/teacher 的下一 token 分布 \(p_t(v)\triangleq\pi_\theta(v\mid x,\hat y_{<t})\)、\(q_t(v)\triangleq\pi_T(v\mid x,\hat y_{<t})\)。序列目标
\[\mathcal L_{\text{OPD}}(\theta)=\mathbb E_{x\sim\mathcal D_x}\big[D_{\mathrm{KL}}\!\big(\pi_\theta(\cdot\mid x)\,\Vert\,\pi_T(\cdot\mid x)\big)\big],\]
按 autoregressive 因子分解为逐 token 之和(注意 KL 方向是 \(p\Vert q\) = reverse-KL):
\[\mathcal L_{\text{OPD}}(\theta)=\mathbb E_{x\sim\mathcal D_x,\ \hat y\sim\pi_\theta(\cdot\mid x)}\Big[\textstyle\sum_{t=1}^{T}D_{\mathrm{KL}}(p_t\Vert q_t)\Big].\]
三种实现只差"每个位置评多少 token"(输入都是同一批学生 rollout,输出都是回传给 \(\theta\) 的梯度):
1. **Sampled-token OPD(最常用、最轻)**:只评学生实际采到的那个 token \(\hat y_t\sim p_t\),逐 token 损失 \(\ell^{\text{sample}}_t\triangleq\log p_t(\hat y_t)-\log q_t(\hat y_t)\)。因 \(\mathbb E_{\hat y_t\sim p_t}[\ell^{\text{sample}}_t]=D_{\mathrm{KL}}(p_t\Vert q_t)\),它是逐 token reverse-KL 的**无偏单样本估计**;每位置只查 teacher 1 个 log-prob,代价最低。Thinking Machines blog/MiMo/Yang2026b 都用这个。【§2.2 式3】
2. **Full-vocabulary OPD**:每个前缀算全词表 KL,梯度最稠密,但显存 \(O(BTM)\)(\(B\) batch、\(T\) 序列长、\(M=|\mathcal V|\) 词表)——大词表下昂贵。【§2.2 式4】
3. **Top-\(k\) OPD(折中)**:把散度限制到学生 top-\(k\) 子集 \(S_t=\text{TopK}(p_t,k)\),在 \(S_t\) 上重归一化 \(\bar p^{(S_t)}_t,\bar q^{(S_t)}_t\) 后算子集 KL \(D_{\mathrm{KL}}(\bar p^{(S_t)}_t\Vert\bar q^{(S_t)}_t)\)。丢掉 \(S_t\) 外的质量、仍是 full 的近似,但 teacher 查询代价显著降,且保留学生高概率区的多 token 监督。**默认 \(k=16\)、Student Top-K 策略**。【§2.2 式5,Table 2】

### B. 三个诊断指标(全文监控曲线,都在学生 rollout 上算,top-\(k\) 默认 16)
设学生/teacher 各自 top-\(k\) 集 \(S^{(p)}_t=\text{TopK}(p_t,k)\)、\(S^{(q)}_t=\text{TopK}(q_t,k)\)。
- **Overlap Ratio**(师生候选空间对齐度):\(\;M_{\text{overlap}}\triangleq\mathbb E_t\big[\,|S^{(p)}_t\cap S^{(q)}_t|/k\,\big].\) 趋 1 = 学生已找到 teacher 的支撑区;低 = mode mismatch。【式6】
- **Overlap-Token Advantage**(交集内分布对齐质量):在交集上重归一后 \(A_t(v)\triangleq\bar p_t(v)\big(\log\bar q_t(v)-\log\bar p_t(v)\big)\),取均值 \(M_{\text{adv}}\triangleq\mathbb E_t\big[\frac{1}{|S^{(p)}_t\cap S^{(q)}_t|}\sum_{v}A_t(v)\big].\) **趋 0 为佳**(学生在 teacher 偏好 token 上质量与置信恰当);大负值 = 学生比 teacher 过自信(\(p_t\) 高但 \(q_t\) 低)。【式7】
- **Entropy Gap**:\(\Delta H_t=|H(q_t)-H(p_t)|\),状态级 mode 对齐指标,趋 0 = 学生匹配上 teacher 的不确定性轮廓。【式8】
- 附录补:**overlap mass** \(M^{(p)}_{\text{overlap-mass}}=\mathbb E_t[\sum_{v\in S^{(p)}_t\cap S^{(q)}_t}p_t(v)]\)(teacher 侧同理),实测师生 overlap token 全程占 97%-99% 概率质量(图18)——这是"只训 overlap 就够"的概率基础。【附录B.1 式9/10】

### C. 现象学:两条件的受控实验(每个条件一组对照)
- **条件1 思维模式一致**(§3.1):学生 Qwen3-1.7B-Base;两 teacher = Qwen3-4B(Non-thinking) vs Qwen3-4B-Base-GRPO(对 Qwen3-4B-Base 跑 zero-RL/GRPO 得到)。两者 benchmark 分相近(图3),但 base 学生与 GRPO teacher 思维更近 → **GRPO teacher 初始 overlap 更高 → 蒸馏更好**;两 overlap 曲线后期收敛但性能差持续 → **早期 thinking-pattern mismatch 造成的损失后期补不回**。【§3.1,图2/3】
- **条件2 新知识≠分数**(§3.2):两族对照,各比"同 pipeline teacher" vs "经额外 RL 的 teacher"。DeepSeek 族:学生 R1-Distill-1.5B,teacher R1-Distill-7B vs Skywork-OR1-Math-7B(对 7B 加 RL);Qwen 族:学生 Qwen3-1.7B(NT),teacher Qwen3-4B(NT) vs Qwen3-4B-NT-RL-Math(在 DeepMath 57K 子集上加 RL)。指标用 **gap recovery rate** \(=\dfrac{\text{Acc}_{\text{after OPD}}-\text{Acc}_{\text{before OPD}}}{\text{Acc}_{\text{teacher}}-\text{Acc}_{\text{before OPD}}}\)。结果:同 pipeline teacher 增益有限;经 RL 的 teacher recovery 显著更高(**Qwen3-RL-Math 58.6% vs Qwen3 15.6%;Skywork 16.9% vs DeepSeek 5.3%**),且 overlap 都已高 → 增益来自 teacher 经 RL 获得的**新能力**而非尺度。【§3.2,图4】
- **reverse distillation(一招同时验两条件,§3.3 关键)**:JustRL-1.5B = 对 R1-Distill-1.5B 做 RL 得到;现反向,用 JustRL-1.5B 当学生,蒸馏自它自己的 pre-RL checkpoint R1-Distill-1.5B,也用更大更强的 R1-Distill-7B 当 teacher。结果两条:① 蒸回自己 pre-RL checkpoint → 学生**几乎精确退回 pre-RL 性能**(RL 增益被抹掉)→ 证明 OPD 主动吸收 teacher thinking pattern 并**覆盖学生自有的**;② 换成更大更强的 R1-Distill-7B,训练轨迹**几乎不可区分、同样退化** → 因 reverse-KL 在学生访问状态上,两 teacher 诱导出几乎相同的局部目标分布。三条结论:思维模式主导、benchmark 分不预测 OPD 结果、高分≠新知识。【§3.3,图5】

### D. 机制:overlap-token 自增强(§4)
- **成功 vs 失败签名**(§4.1):同学生 R1-Distill-1.5B,teacher JustRL-1.5B(成功) vs R1-Distill-7B(失败,虽更强)。成功 run:overlap ratio 稳升、overlap-token advantage 升向 0、entropy gap 收窄(学生渐进定位 teacher 高概率区、校准其内质量、匹配局部置信);失败 run 三指标全停滞。两观察:overlap token 全程占 97-99% 质量(故 overlap 上升≠集合巧合);overlap-token advantage 改善说明 OPD 主优化信号在**交集区内重分配质量**,而非交集外。【§4.1,图6】
- **因果消融(§4.2)**:固定成功设置(JustRL-1.5B→R1-Distill-1.5B),只改"损失覆盖哪些 token":(i) Student Top-\(k\)(全 \(S^{(p)}_t\));(ii) Overlap Top-\(k\)(交集 \(S^{(p)}_t\cap S^{(q)}_t\));(iii) Non-Overlap Top-\(k\)(对称差 \(S^{(p)}_t\triangle S^{(q)}_t\)),\(k=16\)。结果:**Overlap Top-\(k\) 几乎等于 Student Top-\(k\)**(三 benchmark 都是),Non-Overlap 显著弱;且 Student/Overlap 的 overlap-token advantage 曲线几乎重合。→ **OPD 主增益来自共享高概率区的梯度**。【§4.2,图7】
- **自增强机制(直觉+真实形式)**:Student/Overlap Top-\(k\) 都把 overlap ratio 从 ~72% 稳升到 >91%,Non-Overlap 先降后部分恢复。机制=reverse-KL 的 mode-seeking:一旦某 token 进入共享高概率区且被 teacher 偏好,reverse-KL 更新就**给它堆更多质量、把竞争 non-overlap token 挤出学生 top-\(k\)** → overlap 区"因优化而扩大"形成良性循环。统一论断:OPD 主效应=**在学生访问状态上,把学生分布在 teacher 支撑的高概率 token 上逐步精修**。【§4.2 末】

### E. Recipe:救活失败 OPD 的两招(§5)
- **off-policy cold start(§5.1)**:两阶段——先用 teacher 生成的 rollout 对学生做 SFT(拉近思维模式),再接标准 OPD。设置:学生 Qwen3-1.7B-Base,teacher Qwen3-4B(NT),prompt 源 OpenThoughts3-1.2M 数学子集;teacher 生成 **200K** 响应做 SFT 得 Qwen3-1.7B-SFT,再用去重后约 **30K** prompt 接 OPD。对照=纯 OPD 直接从 Base 起。结果:SFT 初始化的学生**起始 overlap 更高、轨迹更稳、最终天花板也更高**,且性能差全程持续(cold start 既改善早期优化也抬高最终上限);纯 OPD 起点低、早期剧烈不稳。【§5.1,图8】
- **teacher-aligned prompt(§5.2)**,两粒度:
  - **模板**(teacher JustRL-1.5B,学生 R1-Distill-1.5B,题集同为 DAPO-Math-17K,只换模板):把标准 DAPO 模板换成 JustRL 后训练用的 "Please reason step by step, and put your final answer within \boxed{}" → 三 benchmark 精度与 overlap 增长都升(连模板这种微小改动都能让学生生成状态更兼容 teacher)。【§5.2,图9】
  - **内容**(teacher Qwen3-4B-Base-GRPO,学生 Qwen3-1.7B-Base,等量比 DAPO-Math-17K[与 teacher RL 数据对齐] vs 去重 DeepMath 子集):teacher-aligned 内容下游更好,但 **overlap ratio 反而更低、而学生在 overlap token 上的累计质量显著更高**(质量集中到更少但更强共享的 token)。代价:**显著压低学生熵** → 作者建议混入 OOD prompt 防熵坍缩。【§5.2,图10】

### F. 代价:稠密 reward 的可靠性边界(§6,最重要的冷水)
- **长度甜区(§6.1)**:学生 R1-Distill-1.5B vs teacher JustRL-1.5B,跑 6 个 max response length 各 200 步。0.5K/1K 监督 token 太少;**3K/7K 最强**;10K/15K plateau 或下降,且 10K/15K 后期 overlap 骤崩,伴学生熵/grad norm 尖峰。【§6.1,图11a/12】
- **失稳从尾部向前传(§6.1)**:15K 设置下,把学生熵按输出位置画热图,高熵**先出现在响应末尾,随训练向前传播**(back-to-front,图13);teacher 熵也有 suffix→prefix 趋势——一致于"teacher 在更深位置遇到越来越陌生的前缀、产出更噪的 reward 进而带崩学生"。【§6.1,图13】
- **teacher continuation 随前缀加深退化**:采 2K DAPO prompt 生成 >16K 的学生 rollout,在多位置截断后让 teacher 续写。teacher 的精度优势**从 1K 前缀的 +0.37 单调降到 16K 前缀的 +0.02**(图11b 给 +0.3659/+0.2709/+0.1522/+0.0237)。结论:dense reward 在中等长度有效,但可靠性随深度退化 → **OPD 可能难干净扩到长 CoT/agentic 多轮**。【§6.1,图11b】
- **全局有信息 ≠ 局部可利用(§6.2)**:对每条学生 rollout 算序列均值 reward \(\bar r(y)=\frac1T\sum_t[\log\pi_T(y_t\mid x,y_{<t})-\log\pi_\theta(y_t\mid x,y_{<t})]\),比较对/错 rollout 的分布。**失败的 7B teacher 与成功的 1.5B teacher 都让对的 rollout 拿更高 reward,AUROC 相当(0.73 vs 0.75)** → 失败不是信号质量差,而是局部优化几何问题。**各向异性梯度假说**(§6.2,作者明说**未直接验证**):7B teacher 的逐 token advantage 虽单点大但跨位置各向异性,聚合成梯度时部分相消 → 有效梯度小(实测 7B 的 overlap-token advantage 幅值更大但 grad norm 持续更小);JustRL-1.5B 因思维兼容,advantage 集中在更连贯的 token 子集,梯度方向一致、可被 reverse-KL 的 mode-seeking 放大。【§6.2,图14 + 附录B.2】
- **support size \(k\)(§6.3)**:学生 R1-Distill-1.5B、teacher JustRL-1.5B,比 sampled-token vs Top-\(k\) (\(k\in\{1,4,16,64\}\))。**sampled-token≈Top-{4,16,64};仅 Top-1 明显差**(图15:Top-4 0.473/0.331/0.793 vs Top-1 0.446/0.310/0.772 等)。Top-1 不稳因总取 argmax、小策略变动就翻转 rank-1 → reward 不稳;sampled-token 每步按学生分布抽不同 token,提供高概率区的无偏覆盖。→ **\(k\) 非关键设计,避开 Top-1 即可**。【§6.3,图15/16】

### 默认超参(可复现锚点,Table 1/2)
- **OPD 默认(Table 2)**:training temperature 1.0、global/mini batch 64、rollout number 4、LogProb top-\(K\)=16、Top-K 策略=Student Top-K、top-\(p\)=1.0、max prompt 1024 / max response 7168、lr 1e-6、epoch 1、**KL coeff = 0.0**。
- **teacher 的 GRPO(Table 1,造 Qwen3-4B-Base-GRPO)**:base Qwen3-4B-Base、rollout \(n=8\)、max prompt 1024 / max response 7168 / validation max 31744、lr 1e-6、temperature 1.0 top-\(p\) 1.0、关 KL、token-mean loss、1 epoch、8×A800-80G。
- **评测口径**:AIME2024/AIME2025/AMC2023,每题采 16 解,T=0.7、top-\(p\) 0.95、validation max 31744,主指标 avg@16。【§3.1,附录A.2 Table2】

## 靠不靠谱
- baseline 公平吗:对照设计极干净(同学生/同数据/同 recipe 只换 teacher 或只换模板),reverse distillation 用"同族不同 scale"隔离"分数 vs 思维模式",公平性强;诚实标注未验证的假设(§6.2 各向异性梯度假说"未直接验证,留 future work")。
- 看着强但没回答核心问题:全部实验在**数学域 + 中小模型(1.5B-7B)**,作者自承未验证 code/开放域(§8);"新知识"条件依赖预训练语料差异,但跨族蒸馏混淆了 tokenizer/架构,受控预训练消融太贵——"新知识"到底是什么仍未被干净隔离【§8 承认】;§6.2 "全局有信息≠局部可利用"的各向异性梯度解释只是 suggestive hypothesis,未证。
- 假设与失效边界:【原文】① 所有结论隐含"teacher 逐 token reward 在学生访问状态上可靠",而该假设随轨迹深度破裂(§6.1):reward 质量有甜区(3K-7K),太短监督不足、太长(10K/15K)后期崩;失稳 back-to-front 向前传 → OPD 难干净扩长 CoT/agentic。② sampled-token 够用、Top-1 不行。【推断】"思维模式一致"是必要条件意味着跨族/跨范式(thinking↔non-thinking)蒸馏天然吃亏;更大但思维迥异的 teacher 即便更强也可能 induce 一个在学生策略附近局部平坦的 reward landscape(§6.2 假说),token 梯度无效。
- 祛魅总结:真贡献=把 OPD 从"经验配方"提升为"有诊断指标(overlap/advantage/entropy-gap)+ 因果机制(overlap-token 自增强)+ 成败条件 + 救活 recipe + 明确代价(长程退化)"的系统理解,且 reverse distillation 的"分数与动力学解耦"极有冲击力。【推断】被高估的风险:"新知识"条件偏定性、未被干净量化,可能掩盖"预训练数据 + 容量 + 范式"的混淆;结论强绑定数学域中小模型。被低估的是 §6 的"长程退化"——对当前 OPD 乐观叙事最重要的冷水,却放在 Discussion。它**不是**提新方法,而是给整条 OPD 线划了边界与体检指标。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=teacher 逐 token log-prob(reverse-KL 隐式 reward),核心作用集中在师生 top-k overlap token | **改什么**=学生参数 θ;OPD 本质改写学生在访问状态上的高概率 token 分布(并覆盖学生原有思维模式) | **何时改**=on-policy 在线、学生自采轨迹上;recipe 建议先 off-policy SFT cold start 再 OPD | **免梯度?**=否(reverse-KL policy-gradient);但揭示"sampled-token 单 token 即够"=极轻量 | **记忆-技能生命周期**=无显式记忆/技能库;知识=teacher 的思维模式被吸收进参数 | **防遗忘机制**=本文反而揭示 OPD 会"覆盖/抹掉"学生既有能力(reverse distillation 把 RL 增益抹回 pre-RL),即 OPD 在师生思维不一致时本身是一种**遗忘风险**;recipe 的 cold start/混 OOD prompt 间接缓解熵坍缩
- ⑦ 开源代码+框架/harness:https://github.com/thunlp/OPD(v1 元信息已核:约 195MB,verl fork + LlamaFactory);框架 **verl v0.7.0(OPD+GRPO,rollout 用 vLLM,verifier math-verify)+ LlamaFactory v0.9.5(SFT cold start)**;OPD 经自定义 ADV_ESTIMATOR=token_reward_direct 实现(teacher 当 reward model 给 token reward);其 overlap 诊断指标(overlap_ratio / overlap_token_advantage)已合并入官方 verl(PR #6469)。评测复用 thunlp/JustRL pipeline。【§1,v1 元信息】
- 💰 资源/成本与可扩展性:原文未给总 GPU 时(给了 teacher GRPO 用 8×A800-80G、1 epoch);给了关键扩展性结论=**稠密监督有"长度天花板"**(3K-7K 甜区,>10K 退化),且 sampled-token(每位置 1 token)即可达 Top-k 水平→可显著省 teacher 查询成本;cold start 需额外生成 200K teacher rollout 做 SFT。【§5.1,§6.1,§6.3,附录A.1】
- 🎯 对"探索-巩固"对标:**理论基石级支撑,直接为 TSRD 提供机制依据,亦是边界警示**。强支撑:① "OPD 学的是 thinking pattern 而非分数"+"overlap token 是作用点"→ 为 TSRD"teacher 当稀疏脚手架教 path-selection"提供 token 级理论:真正起作用的本就是**少量高概率共享 token**,与 TSRD 想做的"稀疏化/单点接管"天然契合(§4.2 已证只训 overlap 就够,正是"稀疏脚手架"的实证支持)。② "思维模式一致是必要条件"→ TSRD 让 student 自选"能走通的开头/恢复分支"正是在保证师生兼容、抬高初始 overlap,与本文 §5(cold start / teacher-aligned prompt 拉 overlap)同向。③ "高分≠新知识"→ 提醒 TSRD 选 teacher 要找能提供 student 没见过路径的,而非单纯更强。**可借组件**:overlap_ratio / overlap_token_advantage / entropy_gap 三件套可直接拿来当 TSRD 训练的健康度探针;cold start 拉 overlap 的两阶段范式。**重大警示/缺口**:§6 的"长程 reward 退化 + 失稳从尾部向前传"正中 TSRD 痛点——TSRD 想用 MTP 做长程前瞻探针,但本文表明 teacher dense 信号在长轨迹尾部本就不可靠,**这恰恰是 TSRD 用 MTP 前瞻 + 稀疏关键步接管 想解决的问题**(把稠密 reward 退化的尾部交给 sparse_critical 单点接管,而非全程稠密)。判定:**强支撑 + 给 TSRD 划清了"为何要稀疏/前瞻"的动机边界**。
- 🔭 开放问题/未来方向:【原文 §8】① 数学外(code/开放域)是否同条件同机制未知;② 预训练语料对"新知识"条件的影响难隔离(跨族混淆 tokenizer/架构);③ 自蒸馏动力学(思维一致天然满足、新知识来自特权信息)是自然下一步;④ 长程/agentic:提倡 hybrid——短段用 dense token 监督 + 长程用 sparse outcome reward,以及"渐进延长监督 horizon"的 curriculum。【推断】把 overlap-token 自增强机制与 MTP 前瞻结合做"提前对齐将访问的高概率 token";验证 §6.2 各向异性梯度假说并设计能利用各向异性 reward 的 objective;把"只训 overlap token"推到极致=显式稀疏脚手架蒸馏(与 TSRD sparse_critical 直接对接)。

RETURN: rethink_opd|读PDF?是(_txt 2273行,核到§2-§8全文+附录A/B超参表)|加厚?是(方法扩为A-F六块+默认超参,相关工作三段精确化,9条公式转MathJax)|LaTeX公式条数 12|待核数 1(§6.2各向异性梯度假说作者未验证)

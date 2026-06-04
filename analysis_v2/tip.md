tip | TIP: Token Importance in On-Policy Distillation | Princeton(通讯 Yuanda Xu)+ 多名共同一作(Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang, Alborz Geramifard;部分工业界) | arXiv 2604.14084 v4(2026-05-21, cs.LG)·Preprint | 主题线 L6(Token 信用/前瞻)+ L1(OPD)·相关性 **高**(与 MTP/OPD 直接相关)

**原始论文**:https://arxiv.org/abs/2604.14084

## 一眼看懂

> 一句话导读:OPD 里"哪些 token 值得学"光看学生熵不够,会漏掉"学生很自信但其实错了"的 token;TIP 加一根师生散度轴,用免参数 Soft-OR 把两轴并起来,在 agentic 规划里只训 20% token 反超全量。

- 🟦 TL;DR:OPD(on-policy distillation)里"哪些 token 有学习信号"由两根轴决定:
  - 学生熵 \(h_t\):学生有多不确定;
  - 师生散度 \(\delta_t\):teacher 有多不同意。
- 光看熵是一个"有效但结构性不完整"的代理:它能抓住高熵区(Q1/Q2),但漏掉 **Q3=低熵+高散度=过度自信却错** 的 token——这些 token 因为低熵而与"已解决 token"(Q4)无法区分,却携带密集的纠错信号。
- TIP 用一个**免参数的 Soft-OR 评分** \(s_t=\hat h_t+\hat\delta_t-\hat h_t\cdot\hat\delta_t\)(任一轴非零即非零)做 top-\(\rho\) 选择:数学上稳超熵单选,且在 agentic 规划里"只训 20% 的 Q3 token"竟反超全 token OPD。【原文 Abstract, §4-7】
- 最巧的一步:**把 token 重要性放到 \((h_t,\delta_t)\) 两轴平面 + 证明熵单轴对 Q3 结构盲(Prop.2)**。
  - 抽掉"加入散度轴"这一步,就退回熵单选、Q3 盲区无法恢复——而 Q3 恰是 agentic 任务里最致命的(一个自信的错误承诺,比如订了已关闭的场地或超了预算,会作废整个计划);
  - 另一处精妙:**选择用 forward-KL、训练 loss 仍用 reverse-KL**——两个方向各司其职,不能混(见下)。【原文 §5, §7.3】

## 为什么做

> 一句话导读:OPD 逐 token 监督,但 token 不等价;熵是常用代理却只覆盖高熵区,而"低熵 + 高分歧"的 Q3 才携带密集纠错信号——TIP 想把这块盲区补回来。

- 研究背景:OPD(Agarwal 2024、Gu 2025)让学生跑自己的 rollout、逐 token 受 teacher 监督,避免 off-policy 的 train-test 失配。但不是所有 token 信号等价;而且 OPD 里 token 重要性由学生自身分布决定,**不能从 teacher 输出预先算好**,必须在线评估——这使"选哪些 token 训"与 off-policy 的样本选择有本质不同。【原文 §1-2】
- 解决的具体痛点:**熵视角结构性不完整**——只用学生熵筛 token,会漏掉"学生自信但与 teacher 严重分歧"的 Q3 位置(过度自信错误)。这类 token 因为低熵而与已解决 token(Q4)无法区分,但携带密集纠错信号。【原文 §1, Abstract】
- 相关工作 & 各自不足(把"按重要性选 token/sample"几条线铺开,定位 Q3 的新意):
  - **样本级重要性/课程一路**:Bengio 2009(课程学习)、Katharopoulos 2018(按梯度范数选 batch)、Ren 2018(元梯度学样本权重)——都在**样本级**,不落到 token。【原文 §2】
  - **RL 里的 token 异质性一路**:Wang 2025c 发现 high-entropy "forking tokens" 驱动大部分梯度;Cui 2025 研究 entropy collapse;SPINE 只更新"决策分叉点"——这些都用**熵/高熵**当信号,正落在 TIP 所说"熵单轴只覆盖 Q1/Q2"的范畴。【原文 §2】
  - **蒸馏里按散度切换/调权一路**:AdaSwitch 按散度在师生间切换、Entropy-Aware OPD(Jin 2026)按 teacher 熵调 loss。
  - **最相关——AdaKD 的 LATF(Xie 2026,divergence-only)**:按师生 **Hellinger 距离** top-r% 硬选 token。TIP 与它的精确切割:
    - ① 论文论证 LATF 单独只 +0.04 ROUGE-L,AdaKD 的增益主要来自**正交的温度模块 IDTS**,而非 token 选择本身;
    - ② **Q3 不是"大散度 token"的改名**——它是**低熵 ∧ 高分歧的合取**,两轴诱导不同选择(Table 3 vs Table 8);budget-matched 对照(Table 8)证明:纯按 \(\delta\) 选在等预算下不及基线、需要 5× token 才追上,**是低熵那个合取项让选择在紧预算下高效**。【原文 §2, §7.3, lines 180, 1204】
  - 动机链(逐步推):
    1. OPD 逐 token 监督但 token 不等价;
    2. 熵是常用代理,但只覆盖高熵区;
    3. 证明熵单调评分对 Q3 结构盲(Prop.2);
    4. 散度 \(\delta_t\) 本就是 per-token loss(零额外计算);
    5. 用免参数 Soft-OR 把两轴并起来恢复 Q3;
    6. 得到 TIP。
  - 与最近邻工作的精确 Δ:
    - vs AdaKD/LATF(divergence-only):Q3 由"低熵 + 高散度"合取定义,并用 budget-matched 对照证明纯散度选不够;
    - vs 熵单选(RL forking-token 一路):Soft-OR 恢复了 Q3 盲区。
    - 有用是因为两轴正交、各抓一类信号,组合后覆盖全部有信息象限。【原文 §2, §7.3】

## 怎么做 + 靠不靠谱

> 一句话导读:两个量都在标准 OPD 前向已算好(零额外计算),用 Soft-OR 把熵和散度并起来选 top-\(\rho\);消融极充分(Q3-only 20% 反超全量是杀手锏),省显存实在,但数学任务上提升量级有限。

- 两个基本量(都在标准 OPD 前向里已经算过,零额外计算):
  - **OPD 基础 loss**(每 token 的 reverse-KL):\(L=\frac{1}{m}\sum_{t=1}^{m}D_{KL}\big(P_S(\cdot\mid c_t)\,\|\,P_T(\cdot\mid c_t)\big)\)(Eq.1),其中 \(c_t=(x,y_{<t})\)。
  - **学生熵(归一)**:\(h_t=\frac{H(P_S(\cdot\mid c_t))}{\log|V|}\in[0,1]\)(Eq.2),\(h_t\) 低=学生自信。
  - **师生散度**:\(\delta_t=D_{KL}\big(P_S(\cdot\mid c_t)\,\|\,P_T(\cdot\mid c_t)\big)\)(Eq.3)——**它就是 per-token loss 本身,无额外计算**;\(\delta_t\) 高=teacher 不同意。
- 四象限分类(Table 1 / Fig.2,各象限占比高度不均衡):
  - Q1=高熵+高散度:纠错或巩固脆弱知识,**信号最密**;
  - Q2=高熵+低散度:稳定欠自信的预测;
  - Q3=低熵+高散度:**过度自信错误**,打破系统性的自信偏差;
  - Q4=低熵+低散度:已解决,可忽略。
  - 统计:Q4 约 **40-47%**,Q1+Q2 约 **40-52%**,**Q3 仅 3-15%** 却携带超比例的纠错信号。Q1/Q2 对熵法可见,**Q3 是熵盲区**。【原文 §4, lines 274-277】
- 方法流水线(OPD 训练中每个 token 位置,输入→输出):
  1. 算 \(h_t\)(Eq.2)与 \(\delta_t\)(Eq.3,复用 loss);
  2. 每个 batch 把熵的顶 2% clip 掉:\(h_t\leftarrow\min(h_t,h^{(0.98)})\),再 min-max 归一 \(\hat h_t=\frac{h_t-h_{min}}{h_{max}-h_{min}}\)(稳排序);\(\hat\delta_t\) 同样归一;
  3. 算 Soft-OR 分 \(s_t=\hat h_t+\hat\delta_t-\hat h_t\cdot\hat\delta_t=1-(1-\hat h_t)(1-\hat\delta_t)\)(Eq.7,免参数,任一轴非零即非零);
  4. 取 top-\(\rho\) 的 token(TopK);
  5. 只对选中的 token 算**标准 reverse-KL OPD loss**(Eq.1)→ 更新 \(\theta\)。额外成本仅 \(O(m\log m)\) 排序,可忽略。【原文 §6, Eq.7/8/9】
- 逐组件必要性(消融极充分,Table 2 把三条理论一一映射实验):
  - **熵轴(Q1/Q2 选择)**:Table 3——保留 50% 熵采样在多数 benchmark 上匹配/超全 token,峰值显存最多降 47%(Qwen3 72.0→38.1 GB)。去掉则无显存收益。
  - **散度轴(恢复 Q3)**:Table 4——只训 Q3(<10% token)近乎追平全 token(Qwen3 MATH 5.7K token 达 76.1 vs 76.7);Table 5 Soft-OR 稳超熵单选(Qwen3 MATH 50% 79.1 vs 熵 78.6)。去掉散度轴=熵单选,Q3 盲。
  - **Soft-OR 的免参数形式**:Remark 1 证明它能恢复 Q3(对 \(\hat h_t\approx0,\hat\delta_t>0\) 有 \(s_t\approx\hat\delta_t>0\))、同时压制 Q4(\(\hat h_t\approx0,\hat\delta_t\approx0\) 得 \(s_t\approx0\))、保留 Q1 最高分(\(s_t\approx1\)),且无需 oracle 量 \(\bar\phi_t,\bar M_t\)。
  - **熵 clip 顶 2% + min-max 归一**:稳排序(无独立消融,但作为实现细节明确给出)。
  - **forward-KL 检测器(Q3-only 实验)**:置信度 \(\mathrm{conf}_t=1-\hat h_t\),Q3 分 \(w^{Q3}_t=\delta^{fwd}_t\cdot\mathrm{conf}_t\)(\(w^{Q3}_t\) 高=学生极自信而 teacher 强烈反对);forward-KL 用于排序、reverse-KL 用于训练 loss(见下机制)。
- 关键机制/公式(直觉):
  - **信号-曲率视角**:对单个 token,把诊断 loss 在学生 logits \(z_t\) 处做二阶展开,权重步 \(\Delta\ell_t(w_t)\approx-\eta w_t\|g_t\|^2+\frac{\eta^2 w_t^2}{2}g_t^\top H_t g_t\)(Eq.5),局部最优 \(w^{local}_t=\frac{\|g_t\|^2}{\eta\,g_t^\top H_t g_t}\)(Eq.6)。
    - 即 token 重要性 = **信号/曲率**权衡:分子 \(\|g_t\|^2\) 是一阶纠错信号,分母是该 token 上挪动的二阶代价。
    - Q3 的分子可以很大(teacher 强烈反对自信预测),而近确定的学生分布给出小的 softmax 曲率(\(\mathrm{tr}(H)=1-\sum p^2\),近确定时小,Appendix A.1)→ 故重要;Q4 同样低曲率,但师生分歧小、分子近零 → 不重要。
    - 所以 **Q3 vs Q4 是信号差异、非熵差异**。oracle 权重 \(w^*_t=\bar\phi_t/(\eta\beta\bar M_t)\)(Prop.1)依赖 population 量、不可直接用,Soft-OR 是它的可计算代理。
  - **forward vs reverse KL 的方向区分(§7.3,关键且论证清楚)**:
    - 用于**排序**的散度是 forward-KL \(\delta^{fwd}_t=D_{KL}(P_T\|P_S)=\sum_v P_T(v)\log\frac{P_T(v)}{P_S(v)}\)。理由:学生在某 token 近确定于 \(v^*\) 时,reverse-KL 由"teacher 赋给学生所选 token 的概率"主导、对 teacher 偏好的其它候选不敏感;forward-KL 直接惩罚漏掉的 teacher 概率质量,给出更锐利的 Q3 排序。
    - **训练 loss 仍用标准 reverse-KL**(mode-seeking、数值稳定、前向已算)。不用 forward-KL 当 loss,是因为它的 mode-covering 偏置会让学生把概率摊到多个 teacher 候选(而 reverse-KL OPD 才是被对比的目标)。
    - Q3 的概念定义(低熵+高分歧)不依赖 KL 方向,forward-KL 只是该诊断实验的"算子级探测器"。【原文 §7.3, lines 629-645】
  - **四象限统计高度不均衡**:Q4 约 40-47%、Q1+Q2 约 40-52%、**Q3 仅 3-15%** 却携带超比例纠错信号。【原文 §4, lines 274-277】
- 实验与证据:
  - 数学:训练 prompt 来自 DAPO;评测 MATH-500(500 题)、AIME 2024/2025(各 30 题),用 mean@16。三组师生对:
    - **Qwen3-8B(GRPO)→4B**;
    - **Llama-3.3-70B-Instruct→Llama-3.1-8B-Instruct**;
    - **Qwen2.5-14B-Instruct-thinking→Qwen2.5-1.5B-Instruct**(~9× 容量差、reasoning teacher)。【原文 §7.1】
  - Agentic 长程规划:**DeepPlanning**(多日旅行 + 多商品购物,需主动信息获取 / 局部约束推理 / 全局约束优化);师生对 **Qwen3-{14B,32B}→Qwen3-1.7B**(均开 thinking,在 agentic 数据上训 15 epoch);80/20 划分,Avg@16,按个性化硬约束满足比例打分。统一 AdamW + cosine + reverse-KL,lr=1e-6(Qwen3/Qwen2.5)/3e-7(Llama)。
  - 关键数字:
    - Table 5 Soft-OR——Qwen3 MATH 50% 达 **79.1** vs 熵 78.6 vs 基线 76.7;AIME'24 25.7 vs 熵 23.8。
    - Table 4 Q3-only——Qwen3 MATH 5.7K token(<10%)达 76.1 ≈ 基线 76.7。
    - **Table 7 DeepPlanning 最亮点:Q3-only 20% 反超全 token OPD(14B:12.6 vs 11.7;32B:13.6 vs 12.8)**。
    - Table 6 top-50% vs bottom-50% by Soft-OR:bottom 信号显著弱(MATH 79.1 vs 72.3)。
    - teacher 熵几乎恒定、对选择无判别力(§7.4)。
  - baseline 公平吗:全 token OPD(100%)作基线,同师生对、同训练配置,只变 token 保留策略——公平;budget-matched 对照(Table 8)排除"Q3=大散度"的平凡解释,严谨。
  - "看着强但没回答核心":数学上 Soft-OR 提升量级有限(MATH 76.7→79.1@50%);真正的强证据是 Q3 盲区的存在性 + DeepPlanning Q3-only 20% 反超(这才是"两轴必要性"的杀手锏)。
- 假设与失效边界:
  - 【原文】作者自陈:最大 teacher 仅 70B、rollout 16K token;象限结构与 Q3 集中现象是否在万亿参数或极长 agentic tool-calling 流水线下持续,仍开放(§Limitations)。
  - 【原文】理论用 token-separable 近似(忽略跨 token 梯度协方差,Assumption 2)+ β-smoothness;oracle 权重 \(w^*_t\) 是 population 量、不可直接用(作者诚实标注,用 Soft-OR 作可计算代理)。
  - 【推断】Soft-OR 的 min-max 归一是 batch 内做的,batch 组成变化可能影响排序稳定性(故需 clip 顶 2%)。
  - 【推断】省显存卖点实在,但"无额外计算"指排序可复用已算 logits,并非零成本(仍有 clip + 归一 + top-k 排序)。
- 祛魅总结【推断】:真贡献=
  - (a) 一个清晰的两轴诊断框架 + 严格的"熵单轴对 Q3 结构盲"证明(Prop.2)+ Q3≠大散度的 budget-matched 排除;
  - (b) 一个免参数、零额外计算的 Soft-OR 选择规则。
  - 包装上"token importance"听着大,方法本身是对既有熵选择的**增量补丁**(Soft-OR 在数学上提升有限)。高估之处:把它当"大幅提升 OPD"——数学上多为 +1~3 点。低估之处:Q3 盲区的诊断 + DeepPlanning 上 20% Q3 反超全 OPD,对"前瞻探针该盯什么"有真实启发(见对标)。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=token 级 (学生熵 \(h_t\), 师生散度 \(\delta_t\)) 两轴,Soft-OR 组合分 | **改什么**=哪些 token 进 reverse-KL OPD loss(top-\(\rho\) 选择) | **何时改**=每个 OPD 训练 step 在线评估(token 重要性依赖学生当前分布,不能预计算) | **免梯度?**=否,被选 token 走标准 reverse-KL OPD 梯度;选择/排序本身免梯度(用熵+散度打分) | **记忆-技能生命周期**=不涉及 | **防遗忘机制**=无(纯 token 选择 + OPD)
- ⑦ 开源代码+框架/harness:https://github.com/HJSang/OPSD_OnPolicyDistillation(**已克隆,本地在 `resource/repos/tip/`**,git remote 已核实指向此仓;为 **TIP/PACED 等共享的通用 OPSD 仓库**,README 列两篇论文,TIP 为在此扩展实现)。**框架=verl**;关键文件 `src/opd/{losses.py, opd_worker.py, opd_trainer.py, batch_builder.py}` + `src/opd/config/opd_trainer.yaml`。`losses.py` 已核对:含 `compute_reverse_kl_loss/compute_forward_kl_loss/compute_jsd_loss`(`LOSS_FN_MAP` 切换训练散度)、`entropy_weighted_sample`(weight=entropy_norm^α·jsd_norm^γ,按权 top-ratio 多项式无放回采样)、`_compute_per_token_entropy_and_jsd`(含 JSD teacher top-K 截断)、`compute_teacher_token_stats`(区分学生 OOD vs 真分叉)。注意:**论文 Q3 实验用 forward-KL 检测器,而仓库通用选择器用归一化 JSD**,二者需对应留意(非"\(\delta_t\) 就是 reverse-KL")。`opd_worker.py` 做两阶段 teacher/student 显存调度 + chunk 化散度(V=152K 时单 (N,V) float32 ~2.3GB 故分块)+ remove-padding。
- 💰 资源/成本与可扩展性:卖点之一就是省显存——熵 50% 选择峰值显存最多降 47%(Qwen3 72→38.1 GB),更激进保留降至 58%;Q3-only 也显著降显存(Table 4)。支持"在有限 GPU 预算下蒸馏更大模型"(README)。额外计算仅 \(O(m\log m)\) 排序,可忽略。【原文 Abstract, Table 3-4】
- 🎯 对"探索-巩固"对标:**强支撑(idea 的直接理论弹药)+ 可借组件**。一句判定:TIP 的 **Q3=低熵高散度=过度自信错误** 恰是本项目"teacher 当稀疏脚手架、在学生'走偏但自信'处接管做 path-recovery"最该捕捉的目标——MTP 前瞻探针的一个明确候选职责就是"检测学生自信却将走错的位置"(Q3),而非只看高熵分叉(熵单轴会漏掉它)。可借组件:① 两轴 (\(h_t,\delta_t\)) 诊断 + Soft-OR 评分可直接作"切关键步/选接管点"的打分函数;② forward-KL 当排序信号、reverse-KL 当训练 loss 的方向区分,是做 token 级蒸馏 loss 时的现成工程经验;③ "Q3 在 agentic 任务里特别集中(单个自信错误作废整个计划)"佐证本项目把 scaffold 聚焦在关键决策点的合理性。Δ/缺口:TIP 只做**被动 token 选择 + 标准 OPD 监督**,不含"学生自选恢复分支"、不含 MTP 前瞻(它用当前 \(h_t/\delta_t\),非未来 token 预测)、不含记忆固化;本项目要把"检测 Q3"升级为"在 Q3 点让 teacher 给恢复脚手架 + 学生 on-policy 自选续写"。【推断,依据 §1/§7.3 Q3 定义与本项目 path-recovery + MTP foresight 的对应】
- 🔭 开放问题/未来方向:【原文】象限结构/Q3 集中在万亿参数或极长 agentic tool-calling 流水线下是否持续(§Limitations);两轴框架可推广到 process reward fine-tuning 与 speculative decoding(§8)。【推断】把"检测 Q3"与"在 Q3 点做 path-recovery 脚手架"结合(本项目方向);用 MTP 前瞻替代/补充当前的 \(\delta_t\) 作为"将走错"的预判信号;与 TCOD 的轨迹深度课程结合,在课程允许深度内再做 Q3 token 选择;研究 Q3 token 的时序动态(对照 EDIS 的熵动力学)。

〔本篇与既有 analysis/tip.md 核对:无事实冲突。本次增强:8 条核心公式(OPD loss Eq.1、熵 Eq.2、散度 Eq.3、二阶展开 Eq.5、局部最优 Eq.6、Soft-OR Eq.7、oracle 权重 Prop.1、forward-KL 检测器/定义)转 MathJax 并从 PDF 抄准;补四象限完整分类与统计占比、信号-曲率推导链(Eq.5→6 + tr(H) 直觉)、forward/reverse KL 方向区分的完整论证;相关工作铺开样本级重要性/RL forking-token/蒸馏散度调权/AdaKD-LATF 四条线 + Q3≠大散度的 budget-matched 排除。无新增待核项。〕

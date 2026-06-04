sod_stepwise | SOD: Step-wise On-policy Distillation for Small Language Model Agents | 浙大 + 腾讯 LLM 部 + 中科大 + 新加坡国立(Qiyong Zhong、Mao Zheng、Mingyang Song 共同一作;Junfeng Fang、Houcheng Jiang 通讯) | 2026-05-08 arXiv v1(2605.07725)·Preprint | 主题线 L1(OPD/自蒸馏)+L4(Agent/工具/多轮)·相关性 高

**原始论文**:https://arxiv.org/abs/2605.07725

## 一眼看懂
- 🟦 TL;DR:把 OPD(on-policy distillation,teacher 在 student 自己生成的轨迹上给 dense token 级监督)用到**小模型 agent 的工具集成推理(TIR)**会训练崩溃——因为小模型工具用得差,一次错误工具调用注入错误观测,后续推理在被污染状态上展开,**师生发散"加速"漂移**,teacher 在这些 OOD 状态上的监督变得不可靠甚至误导。SOD 按 **step 级师生发散自适应重加权**蒸馏强度(高发散区衰减、重新对齐时回升),在对齐区保留 dense 监督。0.6B/1.7B 学生相对最强基线 OPD 平均 +20.86%/+18.50%;0.6B 在 AIME2025 达 26.13%(avg@32)【原文 abstract+§5】。
- 最巧的一步:**用相邻 step 的发散比(而非绝对发散值)累乘成权重**(Eq.7)。抽掉它(改用绝对发散或均匀权重)就垮:① 比值累乘让"发散单调上升→权重<1 自动压低被污染信号";② 一旦师生重新对齐(d 下降)则比值>1、蒸馏强度回升(论文称 **recovery from earlier errors**);③ 只依赖比值意味着任何与真实师生发散 \(\Delta_k\) 单调一致的可观测代理都可用(Appendix D.5),理论上还能把加权二阶矩压到 \(O((d_1/d_k)^2)\) 恢复梯度 SNR(Prop.2/D.4)。

## 为什么做
- 研究背景:agentic 能力多依赖大模型、推理成本高,迁移到可端侧部署的 SLM 有实践价值;但让 SLM 获得稳定有效的 TIR 仍难。TIR 后训练主流基于 RL(GRPO),只给稀疏 outcome 级 reward。OPD 提供 dense token 级监督,缓解 credit assignment、提样本效率与稳定性,是把大模型 agentic 能力蒸到小模型的自然候选【原文§1+§3.3】。
- 解决的具体痛点:小模型容量有限、探索弱,稀疏 outcome 监督加剧探索失败、陷 cold-start;但直接把 OPD 用于 SLM-TIR 会严重训练不稳/崩溃【原文§1+§4.1】。
- 相关工作 & 各自不足(本轮按引文链补全):
  - **RL for Agents**:RLHF→PPO→GRPO 谱系;结构化推理范式 ReAct/Toolformer/FireAct 靠 demonstration 而非在线优化;RL 扩到 code/tool/GUI/web 轨迹,核心难点是稀疏延迟反馈下的 credit assignment(靠 trajectory-level 更新与 value-free 形式)。KL-regularized 策略优化在 agentic 设置因分布漂移与复合误差被放大。**共性短板:仍靠 trajectory-level 稀疏奖励,无 dense 监督**。
  - **On-policy Distillation(OPD)精确谱系**(§2,本轮补全):**Gu et al.**(MiniLLM 系)把 OPD 形式化为**student 分布下的反向 KL 最小化**(SOD 的 Eq.4 正是此);**Agarwal et al.**(GKD 系)**统一 on/off-policy 蒸馏**于不同散度目标;**Yang et al.** 把 OPD 解释为 **KL-正则 RL + 隐式 per-token 奖励**;**Li et al.** 证 OPD 主要**对齐 student-visited 状态上的局部 support**,依赖师生推理模式的兼容性;另有 OPSD 等自蒸馏(无外部 teacher)变体(SOD 拿 OPSD_gt/OPSD_hint 当基线)。**共性短板:都假设 teacher 监督在所有 student-visited 状态可靠**——本文指出这在 TIR 因工具引入"非连续状态跳变"被严重违反。
  - 训练数据/框架沿用 **Yu et al.[52]（"Demystifying RL in agentic reasoning",对应仓库 `recipe/demystify/`）与 ReTool**【原文§3.3+§4.1+元信息】。
  - 精确差异:vs vanilla OPD——SOD 把"teacher 信号何时可信"做成**可观测、零额外成本、有方差压制理论保证**的 step 级连续调节,而非均匀 dense 监督。
- 动机链:OPD dense 监督本是好东西 → 但 TIR 的发散不同于文本推理的"渐进漂移",是工具错误触发的"加速"漂移(Prop.1:单次错误工具观测使 \(\Delta_k\) 跳变 \(\Omega(m\cdot\eta_{\text{tool}})\),连续 j 次错误超线性复合)→ 在低重叠 OOD 状态上 OPD 梯度被高方差无信息项主导、SNR→0(Prop.2)→ vanilla OPD 均匀聚合 = 系统性高估被污染信号 → 必须按 step 级发散自适应调蒸馏强度【原文§4.1】。

## 怎么做 + 靠不靠谱
- 方法流水线(输入→输出,§4.2-4.3):输入 prompt → ① student \(\pi_\theta\) 在 SandBoxFusion(python 解释器)环境内自生成多步 TIR 轨迹(最多 **16 轮**工具调用)→ ② 按工具观测把轨迹切成 \(K+1\) 个 reasoning step(每 step=两次工具观测之间的模型响应,**工具观测 token 排除**)→ ③ 每 step 算发散分数 \(d_k\)(Eq.6,teacher 监督可靠性的可观测代理)→ ④ 算权重 \(w_k\)(Eq.7,w1=1;k≥2 用相邻发散比累乘并上界裁剪)→ ⑤ \(w_k\) broadcast 到该 step 全部 token 的 OPD 损失(Eq.9)→ ⑥ 联合目标 \(L=L_{\text{GRPO}}+L^{\text{step}}_{\text{OPD}}\)(Eq.10)反向 → 输出蒸馏后的 SLM【原文§4.2-4.3】。
- 逐组件必要性(有消融 Table 2,1.7B student,本轮补全 4 个重加权变体):
  - **Step-wise OPD 重加权(核心)**:消融 (2.2) `w/o Step-wise OPD`(去掉 \(L^{\text{step}}_{\text{OPD}}\))→ Avg 暴跌到 **25.39%**(=纯 GRPO);(1.1) **均匀权重**(\(w_k=1\))→ 34.70%;(1.2) **启发式衰减**(从首次工具错误 step \(k_{\text{err}}\) 起固定指数衰减 \(w_k=\gamma^{k-k_{\text{err}}},\gamma=0.9\))→ 37.14%(优于均匀但**无法捕捉非单调发散/recovery**);(1.3) **首错后硬 mask**(首次工具错误后全部 step 的 OPD 信号置零)→ 31.85%(**最差**,丢弃部分正确轨迹的信息、阻止 recovery);(1.4) **去上界裁剪 δ** → 38.10%(无界放大致不稳)。SOD 完整 = **42.98%**。证比值累乘+上界+recovery 缺一不可。
  - **\(L_{\text{GRPO}}\) 项**:消融 (2.1) `w/o GRPO` → 40.78%,说明稀疏 outcome 奖励与 dense 蒸馏**互补**(step-OPD 给主体细粒度指导,GRPO 拓展 teacher 分布外探索)。
  - **发散比 vs 绝对发散**:Eq.7 用比值而非绝对值——理论上(D.5)任何与 \(\Delta_k\) 单调一致的代理都可用,鲁棒性来源。
  - **上界 \(1+\delta\)(δ=0.2)**:防 recovery 时蒸馏强度暴涨,保证稳定优化【原文§4.2 末】。
- 关键机制/公式(本轮据 PDF 正文 Eq.1-10 + Prop.1/2 补全,真符号 MathJax):
  - **轨迹定义(Eq.1)**:\(\tau=(x,y_1,o_1,\dots,y_K,o_K,y_{K+1})\),\(\pi_\theta\) 只生成模型 token,观测 \(\{o_k\}\) 由环境给。
  - **GRPO 优势 & 目标(Eq.2-3)**:\(\hat A_i=\dfrac{r_i-\mathrm{mean}(\{r_j\})}{\mathrm{std}(\{r_j\})+\epsilon_A}\);\(L_{\text{GRPO}}=-\mathbb{E}\big[\tfrac1G\sum_i\tfrac{1}{|T_i|}\sum_{t\in T_i}\min(\rho_{i,t}\hat A_i,\mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i)\big]\),\(T_i\)=模型生成 token 位置。
  - **OPD 目标(Eq.4,反向 KL 采样估计)**:\(L_{\text{OPD}}=\mathbb{E}\big[\sum_{t\in T_i}(\log\pi_\theta(y_t|y_{<t})-\log\pi_{\text{teacher}}(y_t|y_{<t}))\big]\)——student-visited 状态上的 reverse KL。
  - **step 级真实不匹配(Eq.5)**:\(\Delta_k=\dfrac{1}{|I_k|}\sum_{t\in I_k}D_{\text{KL}}(\pi_\theta(\cdot|y_{<t})\|\pi_{\text{teacher}}(\cdot|y_{<t}))\)(贵,需全词表 KL)。
  - **Prop.1(非连续发散放大)**:文本推理 \(\Delta_{k+1}-\Delta_k=O(\eta)\);TIR 单次长度 m 的错误观测使 \(\Delta_{k+1}-\Delta_k=\Omega(m\cdot\eta_{\text{tool}})\) 且 \(\eta_{\text{tool}}\gg\eta\);连续 j 次错误**超线性复合** \(\Delta_{k+j}-\Delta_k=\Omega(\sum_{i=0}^{j-1}m_i\eta^{(i)}_{\text{tool}})\)。
  - **Prop.2(梯度 SNR 退化)**:定义 teacher-支持区 \(S^\epsilon_t=\{v:\pi_{\text{teacher}}(v|y_{<t})\ge\epsilon\}\) 与重叠 \(\rho_t=\sum_{v\in S^\epsilon_t}\pi_\theta(v|y_{<t})\);当 \(\rho_t\le\rho\),OPD 损失二阶矩 \(\mathbb{E}[\ell^2_t]\ge(1-\rho)\log^2(1/\epsilon)\),且 \(\mathrm{SNR}(g_t)\to0\) as \(\rho_t\to0\)。
  - **廉价代理(Eq.6)**:\(d_k=\dfrac{1}{|I_k|}\sum_{t\in I_k}|\log\pi_\theta(y_t|y_{<t})-\log\pi_{\text{teacher}}(y_t|y_{<t})|\)(用 OPD 前向**已有**的师生 logprob 之差绝对值均值,零额外成本;D.5 证与 \(\Delta_k\) 单调一致)。
  - **核心权重(Eq.7)**:
    \[
    w_k=\min\!\left(\prod_{u=1}^{k-1}\frac{d_u+\epsilon}{d_{u+1}+\epsilon},\ 1+\delta\right),\quad k\ge2,\quad w_1=1
    \]
    发散一路升(\(d_{u+1}>d_u\))→每个比值 \(<1\)→累乘越来越小→被污染区监督被压;某步重新对齐(\(d_{u+1}<d_u\))→该比值 \(>1\)→权重回升(**recovery from earlier errors**);上界 \(1+\delta\) 防暴涨。ε=1e-6、δ=0.2。
  - **token OPD 与 step-wise 目标(Eq.8-9)**:\(\ell_{\text{OPD}}(y_t)=\log\pi_\theta(y_t|y_{<t})-\log\pi_{\text{teacher}}(y_t|y_{<t})\);\(L^{\text{step}}_{\text{OPD}}=\mathbb{E}_{y\sim\pi_\theta}\big[\sum_{k=1}^{K+1}w_k\sum_{t\in I_k}\ell_{\text{OPD}}(y_t)\big]\)。
  - **总目标(Eq.10)**:\(L=L_{\text{GRPO}}+L^{\text{step}}_{\text{OPD}}\)——GRPO 给稀疏 outcome 驱动探索,step-OPD 给 dense 指导且按发散调权。
  - **方差压制(D.4)**:单调递增发散下,重加权把加权二阶矩压到 \(O((d_1/d_k)^2)\),在 OPD 崩溃的低重叠区恢复有界 SNR。
  - 直觉:teacher 在"学生走偏后"说的话越来越不可信就越少听;学生绕回正轨了就重新多听。
- 实验与证据:
  - 数据集/设置:训练数据沿用 Yu et al.[52]——3k 高质量多轮推理 SFT 语料(s1-1k + LeetCode 1k + ReTool 1k,后两者 ReasonFlux-PRM 打分各取 top-1k)+ ~30k RL 数据(DAPO-Math 17k + Skywork-OR1 Math 4902/Code 3586 + MegaScience 3k);SFT 轨迹由 Qwen3-Coder-30B-A3B 在 SandBoxFusion 内端到端交互生成。评测均报 **average@32**(temp=1.0、top_p=0.6、每题 32 采样,5 个随机种子):Math=AIME2024/2025、Science=GPQA-Diamond、Code=LiveCodeBench(v6)。teacher=**Qwen3-4B 经 GRPO 进一步优化**(GRPO avg 61.59);student=Qwen3-0.6B/1.7B【原文§5.1+Table 1】。
  - 关键数字【原文 abstract+§5 Table 1,本轮 PDF 直读确认】:SOD 四任务均最高分,相对最强基线 OPD 平均 +20.86%(0.6B)/+18.50%(1.7B);**绝对值**:0.6B SOD avg **24.22**(OPD 20.04、Vanilla 12.16、SFT 8.97、GRPO 11.32、OPSD_gt 15.93),AIME2025 达 **26.13%** avg@32(称首个达该水平的 sub-billion 模型);1.7B SOD avg **42.98**(OPD 36.27),AIME2024 50.83。**1.7B 学生恢复 teacher 性能 69.8%**(OPD 仅 58.9%)。开销:\(d_k/w_k\) 仅 \(O(K)\) 标量运算,显存差 <0.5GB;0.6B 上 SOD 反比 OPD 快 3.5%(1052.3s vs 1090.5s,因抑制错误学习、失败重试更少)。
  - baseline 公平吗:基线含 Vanilla/SFT/GRPO/OPD/OPSD_gt/OPSD_hint(共 6 个),统一 Qwen3 族、同训练数据、5 种子重复,公平;主卖点是"对最强基线 OPD 的相对平均提升"。
  - 看着强但没回答核心:相对增益口径(+20.86%/+18.50% 是相对百分比而非绝对点数;0.6B 绝对 20.04→24.22,绝对提升约 4 点,相对数放大观感)——需结合绝对值理解。
- 假设与失效边界:
  - 显式【原文】:\(d_k\) 作为 \(\Delta_k\) 的代理,前提是二者单调一致(D.5 给证明);Prop.1/2 刻画 TIR 的发散加速与 SNR 退化(Appendix D 证明)。
  - 隐式【推断】:仅 Qwen3 单一模型族、python 解释器(SandBoxFusion)单一工具环境验证(作者把 web/API 等其他 agent 设置与其他模型族列为局限);\(d_k\) 依赖师生 logprob 差,teacher 自身在 OOD 高熵状态的 logprob 可靠性边界未充分刻画(Fig.1(b) 显示错误轨迹上 teacher 熵均值与标准差都飙升,高熵区 logprob 噪声大可能反噬 \(d_k\) 估计)。
- 祛魅总结【推断】:真贡献=精准诊断"OPD 在 SLM-TIR 上因工具错误触发加速漂移而崩溃"(Fig.1 发散/熵证据 + Prop.1/2 理论)+ 用相邻发散比累乘权重做 step 级自适应重加权(含上界、recovery、零额外成本、方差压制证明)+ 完整开源含核心算法。包装/高估:相对增益口径放大观感(绝对点数提升约 4 点);单模型族单工具环境,外推待验。低估:其"step 级可靠性代理 + 比值累乘"机制本身的通用性(任何"师生信号随状态漂移而失真"的 dense 蒸馏场景都可借),论文聚焦 SLM-TIR 反而限定了它的卖点。

## 结构化抽取
- 🎯 机制速览6轴:
  - **学什么信号**:teacher 在 student 自生成轨迹上的 token 级分布(反向 KL 蒸馏,Eq.4)+ 稀疏 outcome 奖励(GRPO)+ step 级师生发散 \(d_k\)(调权信号)。
  - **改什么**:student SLM(Qwen3-0.6B/1.7B)全参数。
  - **何时改**:on-policy 训练全程;每个 reasoning step 按 \(d_k\) 累乘比动态调该 step 蒸馏权重。
  - **免梯度?**:否,梯度训练(\(L=L_{\text{GRPO}}+L^{\text{step}}_{\text{OPD}}\) 联合反向);\(d_k/w_k\) 计算免梯度(\(O(K)\) 标量,复用前向 logprob)。
  - **记忆-技能生命周期**:无显式记忆/技能库;agentic 技能固化进 student 参数;teacher 提供"如何正确用工具/推理"的 dense 示范。
  - **防遗忘机制**:无跨任务防遗忘;但 step 级重加权可视为"防止学生把 teacher 在被污染状态上的错误监督学进去"(防 negative transfer/防错误级联固化),与 recovery 机制配合保留对齐区的正确监督。
- ⑦ 开源代码+框架/harness:https://github.com/YoungZ365/SOD (v1 验证已克隆约 23MB)。框架=**veRL fork + Open-AgentRL[52]（`recipe/demystify/`,复用其 sandbox_fusion 工具配置）+ ReTool(`recipe/retool/`)**;rollout 走 vLLM(TP=4),SandBoxFusion 作 python 解释器、最多 16 轮工具调用。
  - **〔核心算法确已开源,承 v1 更正〕**:`verl/trainer/ppo/ray_trainer.py:363 compute_stepwise_opd_weights` 完整实现 Eq.6/7(按 response_mask 提取 step 边界、\(d_k=\mathrm{mean}|\log\pi_\theta-\log\pi_{\text{teacher}}|\)、w1=1、\(w_k=\min(\prod(d_u+\epsilon)/(d_{u+1}+\epsilon),1+\delta)\)、broadcast 到该 step 全 token);`ray_trainer.py:878 _apply_token_kl_regularizer` 将其乘到 OPD 优势项 `weighted_opd = opd_coef·stepwise_weights·raw_local_adv`。配置 `verl/trainer/config/algorithm.py` 的 `TokenKLRegConfig`(stepwise_enable/epsilon/delta/opd_coef),运行脚本 `examples/SOD/run_sod.sh`。代码与论文 Eq.7 精确一致,可复现。
  - 〔次要差异,承 v1〕config dataclass 默认 `stepwise_delta=0.5`,但 run_sod.sh 覆写为 0.2(论文值);复现须用脚本而非 dataclass 默认。本轮 PDF 确认论文侧 δ=0.2、ε=1e-6(§4.2)。
- 💰 资源/成本与可扩展性:单节点 8×H20(96GB);0.6B/1.7B 1 epoch 约 2-3 天,4B/14B teacher 约 5-6 天;统一超参(AdamW lr=1e-6、batch 64、mini-batch 16、prompt≤2560、response≤20480、训练每 prompt 采 16、验证 32);所有实验 5 种子重复。SOD 相对 OPD 几乎零额外开销(\(d_k/w_k\) 仅 \(O(K)\) 标量,显存差 <0.5GB,0.6B 上反快 3.5%)。
- 🎯 对"探索-巩固"对标:**最强支撑/直接对标(可借核心组件)**。一句判定:SOD 是本轮 6 篇里与本项目"探索-巩固/路径恢复"**最直接同构**的工作——它显式建模"学生走偏(工具错误致状态漂移)→ teacher 监督在偏离区失真 → 应衰减;学生绕回正轨(\(d_{u+1}<d_u\),论文原话 recovery from earlier errors)→ 应回升蒸馏强度",这正是"巩固/回轨:走偏后自选恢复分支并固化"的 loss 级实现;且它就是 on-policy(学生自生成轨迹)+ teacher 当 dense 监督。可借组件:① step 级师生发散 \(d_k\) 作为"该不该听 teacher"的可观测探针(与本项目 MTP foresight probe 思路互通——都在找"信号失真/路径分叉"的点);② 相邻发散比累乘 + 上界,把"高发散区衰减、重对齐回升"做成连续可微调度;③ 零额外成本(复用前向 logprob)。竞品/差距:SOD 是"**被动衰减**误导监督"而非"**主动**让 student 自选恢复分支"——本项目 idea 强调 student 自己选走得通的恢复路径并由 teacher 稀疏接管,SOD 没有"teacher 在关键步单点接管/给出恢复方向"的脚手架,只是降权;也无 MTP 前瞻。消融 (1.3) "首错后硬 mask 最差" 这一发现尤其重要:**直接对应本项目"走偏后不应放弃整条轨迹,而应保留可恢复段的监督"**。依据:§4.2 recovery + Eq.6/7 + on-policy 设定 + Table 2 (1.3) 反例。
- 🔭 开放问题/未来方向:【原文】扩展到 web/API 等其他 agent 工具环境与其他模型族(作者列为局限);更精细刻画 teacher 在 OOD 状态的可靠性边界(隐含,Fig.1b)。【推断】把"被动降权被污染监督"升级为"teacher 在高发散关键步主动给出恢复分支/单点接管"(对接本项目 path-recovery 脚手架);用 MTP 前瞻提前预测"下一步会不会走偏",在错误提交前就调权(把 SOD 的事后发散检测变成前瞻);把 \(d_k\) 代理与高熵/低置信的关键步切分结合,统一"在哪切、怎么调权"。

RETURN: sod_stepwise | 读PDF?是(32页/103612字,abstract+§2 OPD谱系+§3 Eq.1-4+§4.1 Prop1/2+§4.2-4.3 Eq.5-10+§5 Table1/2 全直读;+20.86%/+18.50%/26.13%/δ=0.2/recovery/4个消融变体数字 均PDF核实) | 加厚?是(补全 Eq.1-10 全套真符号公式+Prop.1/2 形式化+方差压制 O((d1/dk)^2);Table2 四个重加权变体数字 34.70/37.14/31.85/38.10 与"硬mask最差"对本项目的直接启示;related-work OPD 四篇谱系 Gu/Agarwal/Yang/Li 精确定位) | LaTeX 公式条数 10(轨迹/GRPO优势/GRPO目标/OPD反KL/Δk/Prop1/Prop2/dk代理/wk权重Eq7/step-OPD+总目标) | 待核 0

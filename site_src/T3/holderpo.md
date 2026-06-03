# holderpo — Hölder Policy Optimisation

> **一句话重点 (TL;DR)**：把 GRPO 系 RLVR 里"token 级重要性比如何聚合成序列级标量"统一为 Hölder p-mean（p=1→GRPO、p→0→GMPO/GSPO），并沿训练时间退火 p，以单参数平衡"梯度集中放大稀疏信号"与"梯度方差受控"这一无法被任何固定算子同时兼得的 trade-off。

**元信息**：arXiv 2605.12058v2（2026-05-21）｜ UCL / 上海交大 / 港科大(广州)，通讯 Jun Wang (UCL)｜ Preprint（未评审）｜ 主题 token 级聚合算子的统一框架，与 OPD/MTP 关联较弱（纯 RL 聚合层面，可借鉴 token 加权视角）｜ 代码 https://github.com/YihangChen9/HolderPO （已 clone，约 8.8MB，与论文一致）｜ 框架 oat (sail-sg/oat, oat-llm 0.1.3.post1) + vLLM 0.8.4。

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/holderpo/fig_01.png)

*Figure 1: HölderPO unifies token-level aggregation under a single parameter p . The objective at the top generalises GRPO by replacing its arithmetic mean over token-level importance ratios with the Hölder mean of order p ∈ R , recovering GRPO ( p = 1 ) and GMPO/GSPO ( p → 0 ) as special cases. The*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/holderpo/fig_02.png)

*Figure 2: Token-level importance ratio log ρ t ( θ ) during training. Left and Right track the per-step upper and lower envelopes respectively. As p decreases, the upper envelope drops and the lower envelope rises, tightening the gap monotonically. Our decaying schedule p : 2 →-2 (solid green) thus*

## 1. 相关工作与进展
GRPO（Shao et al. 2024）用组内采样轨迹估优势、无需 critic，推动了 DeepSeek-R1 等推理模型。把轨迹级优势映射到策略更新时，需将序列内 token 级重要性比 `r_{i,t}=π_θ/π_{θ_old}` 聚合成序列级标量：GRPO 用算术均值（p=1）、GMPO/GSPO（Zhao et al. 2025）用几何均值（p→0）。并发工作 PMPO（Zhao et al. 2026）也在调聚合算子。

## 2. 现有工作存在的问题
固定聚合算子施加静态优化 landscape，出现临界 trade-off：稠密信号任务（监督分散在大量 token，如 MATH）下 GRPO（p=1）过度放大微小 token 误差→高方差梯度→训练坍塌；稀疏信号任务（正确性集中在罕见高幅 token，如 AIME）下 GSPO（p→0）过度平滑、压制罕见"aha moment"。无单一静态 p 兼得两端——实测 AIME24 在 p=3 峰值、MATH500 在 p=−1 峰值。

## 3. Motivation
最优聚合是"任务信号密度 × 训练进程"的函数，因此需要一个能连续调控、并可随训练动态变化的聚合算子，而非在 GRPO 与 GSPO 间二选一。

## 4. 主要灵感 / 核心直觉
用 Hölder mean（p-范数）把所有均值型聚合统一为单参数 p∈ℝ 的连续谱，并把 p 扩到全实轴——发现 p<0 是一个先前未探索的"逆向集中 (inverse-concentration)"相位（把梯度权重集中到最小比 token，即模型"犹豫"处）。早期用高正 p 激进放大稀疏信号、后期退到负 p 收紧方差，即可在训练生命周期内动态走完 trade-off 两端。与 PMPO 的区别：(i) p 扩到全实轴含 p<0；(ii) 沿训练时间轴（跨 step）而非按轨迹自适应 p。

## 5. 主要解决思路(一段话讲清核心)
把 GRPO 目标中对 token 级重要性比的算术均值替换为 Hölder p-mean `ρ_{i,p}=((1/|y_i|)Σ_t r_{i,t}^p)^{1/p}`，套上 PPO 式序列级 clip 形成目标；p 是连续旋钮，p→0 取几何均值（极限）恢复 GSPO，p=1 恢复 GRPO。理论上证明大 p 集中梯度权重以放大稀疏信号（代价方差界变松）、小/负 p 严格收紧梯度方差（代价削弱稀疏响应）。再用一个沿训练从高正值退火到负值的调度，无额外计算开销地兼顾两端。

## 6. 方法详解(通俗、分步骤)
- **Hölder 聚合**：`ρ_{i,p}(θ)=((1/|y_i|)·Σ_t r_{i,t}^p)^{1/p}`（p≠0），p=0 取几何均值。目标用序列级 clip：`J=E[ min(ρ_{i,p}·Â_i, clip(ρ_{i,p},1−ε,1+ε)·Â_i) ]`，以控梯度方差。
- **梯度集中（Thm 1）**：per-token 梯度权重 `W_{i,t}(p)=r_{i,t}^p/Σ_k r_{i,k}^p` 构成概率分布；其 Shannon 熵在 p=0 取全局最大（均匀），|p| 增大严格下降；p→+∞ 集中到最大比 token（上向集中），p→−∞ 集中到最小比 token（下向集中，放大模型犹豫处的非常规有效决策点→促进多样性）。
- **方差界（Thm 2）**：给出 `‖Var(∇J)‖` 上界，刻画"集中度↑→方差↑"的风险。
- **动态退火**：p 从高正值（早期激进信号放大）线性/分段调度到负值（后期方差受控收敛）。

## 7. 实验数据集
- 数学（基模 **Qwen2.5-Math-7B**，另覆盖 1.5B–8B 多基模）：AIME、AMC、MATH500、Minerva、OlympiadBench 五基准；亦报 R1-Distill-Qwen-7B。
- 智能体：**ALFWorld**（开放世界 agentic，基模 Qwen2.5-Instruct-1.5B，agentic 分支）。

## 8. 实验结果与主要发现
- 五数学基准平均 **54.9%**（Qwen2.5-Math-7B，linear 2→−2 schedule），对 GRPO 51.2 相对 +7.2%，超并发 PMPO 54.2、超 GMPO 52.7；R1-Distill-Qwen-7B 上同 schedule 达 66.4 avg。
- 固定 p=3 即把 AIME 记录从 43.3% 推到 46.7%（先验证 p 的任务敏感性，再用退火统一两端）。
- ALFWorld 用 **1→−1** schedule 取得 **93.8%** 成功率，对 GRPO 72.8% 相对 **+28.8%**；而数学用的 2→−2 schedule 在此仅 87.5%——印证"退火端点须按基模成熟度/任务信号密度标定"（Qwen2.5-Instruct-1.5B 缺域内预训练故偏好保守上限）。

## 9. 结果如何支撑其主张
"无银弹"主张由 p 扫描曲线（AIME24 峰在 p=3、MATH500 峰在 p=−1）直接支撑；"动态优于静态"由退火达 54.9（超任何单一固定 p 的横向对照）支撑；ALFWorld 上换 schedule 才达最优、错配 schedule 掉点，反向印证"端点需标定"。理论 Thm 1/2 给出集中-方差 trade-off 的形式刻画，与经验现象自洽。

## 10. 逻辑自洽性(中性评估)
框架自洽：单参数 p 把已有算子作为特例统一，理论（熵单调、方差界）与经验（p 扫描、退火）相互印证，代码实现与公式一致。但"动态优于静态"的因果归因部分被超参选择稀释——见 §11。

## 11. 残留问题 / 局限
- 主结论建立在单一基模 Qwen2.5-Math-7B，跨架构/规模泛化未充分验证（虽宣称覆盖 1.5B–8B，核心对照集中在 7B）。
- "p<0 逆向集中促进多样性"理论叙事直观，但其经验收益与退火日程（起止值、schedule 形状）强耦合——数学用 2→−2、ALFWorld 用 1→−1，本质是需调的额外超参；论文将其包装为"无额外开销"略乐观。
- Preprint（2026-05），未经评审。
- 〔已核-代码〕`train_zero_math_holder.py` 含 `holder_p_schedule`(constant/linear/quad)、`holder_p_min/max`、`_get_current_holder_p`，loss 为 `ρ=((1/|y|)Σr_t^p)^{1/p}`、p→0 取几何均值、序列级 PPO clip，与 §6 一致。

## 12. 开源代码与框架(链接+框架+代码可得性)
- 链接：https://github.com/YihangChen9/HolderPO （已 clone，约 8.8MB；README 标题/作者/摘要与论文一致；另有 `agentic` 分支跑 ALFWorld）。
- 框架：oat (sail-sg/oat) + vLLM 0.8.4，内含 `understand_r1_zero_main`（Understanding-R1-Zero / Dr.GRPO 系）子包。
- 入口：`train_zero_math_holder.py`，启动脚本 `scripts/qwen2.5-math-7b-holder.sh`，p 调度经环境变量旋钮配置。代码可得、可复现性良好。

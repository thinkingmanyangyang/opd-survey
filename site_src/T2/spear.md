# spear — SPEAR: Learn the Ropes, Then Trust the Wins — Self-imitation with Progressive Exploration for Agentic RL

> **一句话重点 (TL;DR)**：针对多轮 agent RL 中机械熵最大化易致不稳定的问题，SPEAR 用"课程化自模仿学习（SIL）+ 内在奖励塑形"在自身经验引导下渐进调节策略熵（早期广探索、后期收敛利用），在 ALFWorld/WebShop/Sokoban/AIME 上稳定提升 GRPO/GiGPO/Dr.BoT，且额外开销仅理论 10%–25%。

**元信息**：arXiv 2509.22601v4（Date 标注 2025-09-22；v4 2025-12-07 cs.LG） ｜ 腾讯优图 Youtu-Agent Team（通讯 yuleiqin/arthurtan@tencent.com） ｜ venue 〔待核：早前分析称"ICLR 2026 接收"，但 PDF 正文/页眉/页脚均无接收声明，全文唯一 ICLR 出现处为 ReAct[4] 的 ICLR 2023 引用；会议接收状态未经一手来源证实，已删除该断言〕 ｜ 主题 T2 agentic RL（自模仿 + 课程化探索管理长程稀疏奖励，replay buffer 存好轨迹做 off-policy 更新——与 OPD"用优质轨迹引导学生"思路相通） ｜ 代码 https://github.com/TencentYoutuResearch/SPEAR （已 clone 约 96MB，含 verl/ + verl-agent/，HF 有模型 collection） ｜ 框架 veRL + verl-agent

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/spear/fig_01.png)

*Figure 1. Our SPEAR harmonizes the curriculum-scheduled self-imitation learning with intrinsic reward shaping for progressive exploration, improving policy performance across agentic tasks.*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/spear/fig_02.png)


## 1. 相关工作与进展
RL 是磨炼 LLM 长程、稀疏奖励 agent 任务中策略性工具使用能力的主流范式（基于 ReAct 范式，应用含机器人导航、移动助手、web 导航、deep search、GUI），核心难题是探索-利用权衡。已有工作多从策略熵视角刺激探索（熵最大化/正则化技术），也有用 cold-start SFT 或 RL+SFT 混合方案提稳定性。

## 2. 现有工作存在的问题

- 纯熵控制在多轮 agent 中脆弱：环境反馈带来的低概率 token 累积引发严重分布漂移，导致 mode collapse；agent 对多轮交互的不确定性会引起持续熵增长（runaway divergence）或塌缩，训练不稳。
- cold-start SFT / RL+SFT 混合虽提稳定，却限制策略发现 SFT 语料之外的新策略。
- 用 replay buffer 做 off-policy 更新时优势函数需重算、且 off-policy 数据带来不稳定。

## 3. Motivation
能否在策略自身经验引导下平滑调度"何时探索、何时利用"，把策略熵维持在"随时间演化、动态但受控"的区间——早期增大熵做 skill-level 广度探索，后期收敛熵做 action-level 利用/巩固——以实现渐进式探索-利用平衡，而非机械熵最大化。

## 4. 主要灵感 / 核心直觉
"先学规矩，再信战果"（Learn the Ropes, Then Trust the Wins）：前期靠辅助工具使用奖励促频繁工具交互、广探索；待对环境熟悉后再强化对成功轨迹的自模仿利用。用课程调度把这一直觉操作化为跨阶段的两路权重调整。

## 5. 主要解决思路(一段话讲清核心)
在 vanilla SIL（独立 replay buffer 仅存回报超基线的 state-action 对、做 off-policy 更新）基础上，用课程跨阶段联合调节"内在奖励塑形"与"自模仿"两路权重；对 buffer 经验做优势重校准以避免重算优势，并正则化策略更新以稳熵、抑制 reward hacking。训练 batch 同时含 on-policy 与 replay buffer 的 off-policy 数据。

## 6. 方法详解(通俗、分步骤)
基于 group-based RL（类 GRPO）：同一输入生成一组多轮工具交互轨迹 → episode 级奖励计算与优势估计 → 走两条路：

1. **课程调度**：跨阶段联合调"内在奖励塑形"与"自模仿"权重——前期辅助工具使用奖励促 skill-level 探索，后期强化 self-imitation 利用成功轨迹。
2. **优势重校准（advantage recalibration）**：处理 buffer 经验的 off-policy 性质，避免对历史经验重算优势。
3. **策略更新正则化**：稳定熵、缓解 reward hacking / 分布漂移。
数据流：(a) 内在奖励塑形 + 优势估计 + on-policy 更新（类 vanilla GRPO）；(b) 过滤优质轨迹入 replay buffer，经优势重校准 + 正则化做 self-imitation off-policy 更新；课程跨阶段调两路权重以控熵区间。基于 verl-agent 实现多轮 rollout。

## 7. 实验数据集

- agent benchmark：ALFWorld（文本具身交互）、WebShop（网购）。SPEAR 对 GRPO/GiGPO/Dr.BoT 分别带来最高 16.1%/5.1%/8.6%（ALFWorld）与 20.7%/11.8%/13.9%（WebShop）成功率提升。
- Sokoban 视觉推箱子（Qwen2.5-VL-3B-Instruct，Table 5）：SPEAR 把 GRPO 67.1→86.7（+19.6%）、Dr.BoT 76.0→85.4（+9.4%）。
- 数学推理 AIME24/AIME25（带 code interpreter）：SPEAR 对 Dr.BoT 分别 +3.8%/+6.1%。
- 模型：Qwen2.5-1.5B/7B-Instruct、Qwen2.5-32B（code interpreter）、Qwen2.5-VL-3B-Instruct（Sokoban）。
- 开销：理论复杂度仅 +10%~25%，实际每迭代运行时开销可忽略。

## 8. 实验结果与主要发现

- 跨四类任务（文本 agent、视觉 agent、数学+工具）对多个强基线一致提升，验证 plug-and-play 可扩展性。
- 构建并对比强工业基线 Dr.BoT（bag-of-tricks of industrial RL optimizations），SPEAR 在其上仍有正增益，说明增益非来自弱基线。
- 课程化 SIL 把熵维持在受控区间，避免熵塌缩与 runaway divergence（对应 Motivation）。

## 9. 结果如何支撑其主张
"渐进式探索-利用优于机械熵最大化"由跨基线（GRPO/GiGPO/Dr.BoT）一致正增益支撑；"非弱基线红利"由对自建强基线 Dr.BoT 仍有提升支撑；"plug-and-play、低开销"由 +10%~25% 理论复杂度 + 可忽略运行时开销支撑。证据覆盖四类任务，外部效度较好。

## 10. 逻辑自洽性(中性评估)
方法-动机自洽：针对"机械熵最大化不稳"提出"经验引导的受控熵调度"，三项改造（课程、优势重校准、正则化）分别对应稳定性的三个来源（探索-利用时序、off-policy 偏差、熵/hacking）。一个评估张力：SPEAR 由"课程调度 + 优势重校准 + 正则化 + 内在奖励塑形"多组件叠加，论文虽给跨基线增益，但各组件的独立消融与课程超参敏感性需结合附录核验方能判断增益归因。

## 11. 残留问题 / 局限

- venue 状态未经一手证实（见元信息），引用时勿标注会议接收。
- 多组件配方，增益归因到单一机制（课程 vs 优势重校准 vs 正则化）需更细消融支撑。
- 内在奖励塑形依赖"辅助工具使用奖励"的设计，跨环境可迁移性与 reward hacking 风险（虽有正则化缓解）需进一步验证。
- 课程跨阶段权重的调度依赖阶段划分超参，自动化/自适应程度有限。

## 12. 开源代码与框架(链接+框架+代码可得性)

- 仓库 https://github.com/TencentYoutuResearch/SPEAR （已 clone 约 96MB，最新 commit 0edca96）。
- 结构含 `verl/` 与 `verl-agent/` 两个子目录（在 veRL + verl-agent 之上实现）；verl-agent 提供 GiGPO 等 group-based RL 的多轮 agent 扩展。
- README 明确为 curriculum-based SIL 框架，给出 Self-imitation Learning 配置项（`enable_trajectory_replay` 是否启用 self-imitation loss、replay buffer 最大轨迹数等），与论文方法一致。基线含 GRPO、GiGPO、作者构建的强基线 Dr.BoT。HF 有发布模型 collection（yolay/spear-…）。代码可得、配置可定位。

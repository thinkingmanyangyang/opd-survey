# gigpo — Group-in-Group Policy Optimization for LLM Agent Training (GiGPO)

> **一句话重点 (TL;DR)**：把 GRPO 扩展到长 horizon 多轮 agent——除了像 GRPO 那样按整条轨迹回报算 episode 级相对优势,再利用"组内轨迹常反复经过相同环境状态"这一观察,把同一状态下的不同动作聚成 step 级组算 micro 相对优势,从而**无需额外 rollout、无 critic** 就实现细粒度 step 级信用分配,额外时间成本 <0.002%。

**元信息**：arXiv:2505.10978（v3, 2025-10-28, cs.LG；NeurIPS 2025）｜ 南洋理工 NTU / Skywork AI Singapore（Lang Feng、Zhenghai Xue、Tingcong Liu、Bo An）｜ 主题 多轮 LLM agent 的细粒度 credit assignment / 相关性中（step 级信用分配机制可作 path-level 监督的 critic-free 替代/baseline；但本文纯 RL,无 teacher 蒸馏、无 MTP）｜ 代码 github.com/langfengQ/verl-agent（Apache-2.0，本地已 clone）｜ 框架 verl-agent（veRL 扩展）。

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/gigpo/fig_03.png)

*Figure 3: Illustration of step-level grouping in WebShop. Both τ 1 and τ 2 encounter the same environment state multiple times: a search results page (highlighted by the red border). Top : τ 1 eventually succeeds. Bottom : τ 2 leads to failure.*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/gigpo/fig_01.png)


## 1. 相关工作与进展

- **group-based RL(RLOO、GRPO)**:在单轮任务(数学、代码)上很成功——奖励即时、信用分配简单、无需 critic、低显存、稳定。
- **多轮 agent RL**:LLM agent 在外部环境中是长 episode(数十步、数万 token)、奖励稀疏/延迟。
- **已有 step 级信用方案**:actor-critic(PPO、ArCHer、AgentQ)用 value 网络/MCTS;或对每个 state 额外 rollout 新动作。

## 2. 现有工作存在的问题

- **朴素套 group-based RL 到多轮**:把整条轨迹当一个 episode-level 响应(如 RAGEN),组内每步给**同一** advantage,丧失 step 级区分,长 horizon(如 ALFWorld 50 步)下扩展性差。
- **naive step 级信用**:对每个 state 额外 rollout 动作 → 计算代价爆炸。
- **actor-critic**:需额外 value 网络/MCTS,复杂、开销大,失去 group-based RL 的简洁。
- **核心矛盾**:能否**既保留** group-based RL 的优点(critic-free、低显存、稳定)**又引入**细粒度信用分配?

## 3. Motivation
长 horizon 任务里,一个动作的好坏可能很晚才显现,而 episode 级单一 advantage 无法告诉模型"是哪一步走对/走错了"。需要在不增加 rollout、不引入 critic 的前提下,给出 step 级的相对好坏信号。

## 4. 主要灵感 / 核心直觉
关键观察:**同任务同初始条件下,组内许多轨迹因无效动作或循环会反复遇到相同环境状态**(同一网页/房间/游戏场景)。这些共享状态天然就是"对照实验"——在同一状态下不同轨迹采取了不同动作并得到不同后续回报,于是**不用额外 rollout 就能直接比较这些动作的优劣**,做局部(step 级)信用分配。

## 5. 主要解决思路(一段话讲清核心)
GiGPO 嵌套两级相对优势(group-in-group):宏观上,对同一任务采 N 条完整轨迹,按总回报算 episode 级相对优势 A^E(同 vanilla GRPO);微观上,回溯找出组内反复出现的环境状态(anchor state),把"在同一 anchor state 下采取的不同动作"聚成一个 step 级组,组内按各动作的**折扣回报**算 step 级相对优势 A^S;最后线性合并 \(A = A^E + \omega \cdot A^S\)(ω 直接设 1,不调参)。全程 critic-free、无额外 rollout、与原 GRPO 同显存。

## 6. 方法详解(通俗、分步骤)

- **Episode 级(宏观)**:同任务同初始状态采 N 条轨迹,按总回报算 macro 相对优势 A^E,反映整条轨迹的任务完成质量。
- **Step 级(微观)—— Anchor State Grouping(核心创新)**:回溯识别组内跨轨迹/跨时间重复出现的环境状态(anchor state);把所有"在同一 anchor state 下采取的不同动作"聚成一个 step-level group,每个唯一状态构一组。组内对动作算 micro 相对优势 A^S。为捕捉长期影响,对每步关联**折扣回报** R_t(折扣因子 γ∈(0,1]),使早期次优动作得到比后期正确动作更低的折扣回报,产生清晰的偏好排序(例:1st Item > 2nd Item > Next Page)。
- **合并**:\(A = A^E + \omega \cdot A^S\),ω=1(代码 `step_advantage_w` 默认 1.0,未调参)。
- **变体**:GiGPO w/ std 与 w/o std(是否用标准差归一化);**similarity-based 变体**——状态难精确匹配(如 QA)时,用最长匹配子序列相似度阈值(默认 0.95)判定"是否同一状态"。

## 7. 实验数据集

- **长 horizon agent**:ALFWorld(具身家务规划)、WebShop(目标驱动 web 交互)。
- **多轮工具集成推理**:search-augmented QA(沿用 Search-R1 设置,E5 检索器,max turn=4)。
- **base model**:Qwen2.5-1.5B/3B/7B-Instruct。

## 8. 实验结果与主要发现

- **较 GRPO 一致大幅提升**:GiGPO w/o std 在 1.5B 上 ALFWorld +13.3%、WebShop +10.6%;7B 上 +12.6% / +9.1%;QA 3B 42.1%、7B 47.2%。w/ 与 w/o std 均稳超 GRPO 与 RLOO。
- **几乎零额外成本**:GiGPO 特有操作(anchor 分组 + 算 step 优势)单次迭代仅约 0.53s,占总训练时间 **<0.002%**,且与原 GRPO 同显存、同 rollout 预算。
- 报告观察到 emergent reasoning(附录 F)。
- 对照闭源(GPT-4o、Gemini-2.5-Pro)、prompting(ReAct、Reflexion)、RL(PPO/RLOO/GRPO);QA 另比 Search-R1、ZeroSearch、StepSearch 等。

## 9. 结果如何支撑其主张
主张是"在保持 group-based RL 效率的同时引入细粒度信用分配"。两点证据:(1)在三类任务、三种规模上一致超过 GRPO/RLOO,且 w/ 和 w/o std 都成立,说明增益稳健来自机制而非调参;(2)额外时间 <0.002%、同显存,直接支撑"不牺牲效率"。代码核对(`core_gigpo.py`)与论文一致:`compute_gigpo_outcome_advantage` 中 `scores = episode_advantages + step_advantage_w * step_advantages`(默认 1.0),anchor 分组、折扣回报、similarity 变体均实现到位。

## 10. 逻辑自洽性(中性评估)
机制自洽且实现可核(本地代码已审计,与论文 Eq.3/6/7/8 对应)。需保留的批判点:(1)**核心前提是"组内状态确实重复"**——若任务状态空间大、轨迹很少撞到同一状态,step 级组多为 size=1(代码 `summarize_group_size` 正反映这一动态),A^S 退化,增益将缩水;similarity 阈值近似缓解但引入新超参与误聚风险;(2)ω=1 未调参虽显鲁棒,但也意味着两级优势的相对权重未被优化,可能非最优;(3)折扣回报需要环境给出 per-step reward 或可定义,纯稀疏终局奖励下 step 级信号主要靠折扣传播,效果依赖 γ。

## 11. 残留问题 / 局限

- **状态重复假设**:在状态难精确匹配/极少重复的环境中机制退化,论文用相似度阈值兜底但未给失效边界。
- **纯 outcome/环境奖励驱动**:无外部监督或蒸馏,增量集中在 step 级信用分配机制本身,属 GRPO 家族增量改进而非新范式。
- **ω、γ、similarity_thresh** 等超参的系统敏感性分析有限(ω 固定为 1)。
- **评测局限于 ALFWorld/WebShop/QA**,对更开放的真实工具环境(代码执行、长程网页)未验证。

## 12. 开源代码与框架(链接+框架+代码可得性)

- 代码 github.com/langfengQ/verl-agent(Apache-2.0,含 HF 模型 collection);本地已 clone,核心算法在 `gigpo/core_gigpo.py`(已审计)。
- **框架**:`verl-agent` 是 **veRL** 的扩展(pyproject 包名仍为 `verl`)。核心改造是 **step-independent multi-turn rollout**——不简单拼接完整交互历史,而是允许逐步可定制的 per-step 输入结构、history 管理与 memory 模块,支持超长 horizon(ALFWorld 可达 50 步)。
- **配置**:base Qwen2.5-1.5B/3B/7B-Instruct;ALFWorld/WebShop 所有 RL 方法用完全相同超参,rollout group size N=8;QA 用 N=5、max turn=4、E5 检索器;ω=1 不调。

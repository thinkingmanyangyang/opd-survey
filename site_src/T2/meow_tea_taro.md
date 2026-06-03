# meow_tea_taro — A Practitioner's Guide to Multi-turn Agentic Reinforcement Learning

> **一句话重点 (TL;DR)**：把多轮 agentic RL 的设计空间拆成 environment/reward/policy 三支柱做系统受控消融，得出一份可操作配方——"课程(由简到繁) + 稳定化偏置策略(PPO/GRPO 优于无偏 RLOO 与朴素 REINFORCE++) + 验证型稠密奖励(单测通过率远胜模型评判)"。非新算法，是经验研究；框架封装 veRL。

**元信息**：arXiv 2510.01132 (v2, 2025-12-06) ｜ UC San Diego(+ NVIDIA)；Ruiyi Wang, Prithviraj Ammanabrolu ｜ Preprint, under review ｜ 主题 多轮 agentic RL 实证(非新算法)/与 OPD 主线弱相关 ｜ 代码 https://github.com/pearls-lab/meow-tea-taro（Apache-2.0，本地已 clone ~11MB）｜ 框架 封装/vendoring veRL

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/meow_tea_taro/fig_02.png)

*Figure 2: Training curves for Qwen-1.5B on TextWorld w2-o3-q4 with varying hyperparameters. Parameters shown: KL coefficient (kl coef), rollout temperature (t), actor/critic learning rates, and discount factor (gam).*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/meow_tea_taro/fig_01.png)

*Figure 1: Illustration of multi-turn agentic RL and the key research questions.*

## 1. 相关工作与进展
单轮 RL(PPO、RLOO、GRPO、DAPO)已为即时响应质量做了大量优化，但它们假设奖励直接跟随单个 action；多轮交互环境只在长序列交互后才揭示结果，打破了单轮方法依赖的 action-reward 耦合。现有多轮 RL 工作进展有限：有的把单轮 QA 交错工具/推理步骤伪装成"多轮"，有的在真交互环境里只用稀疏终局奖励、或把 turn-level advantage 均匀摊到所有 token(无细粒度 credit assignment)。仓库名 "meow-tea-taro" 是 "Multi-turn" 的谐音梗。

## 2. 现有工作存在的问题

- 多轮 RL 框架与定义碎片化，各家结果不可比，对"什么是真多轮 vs 伪多轮"存在混淆。
- 缺乏对 environment/reward/policy 三者如何共同决定多轮性能的系统理解。
- 固定预算下 SFT:RL 最优配比未知；reward 稀疏性 × RL 算法的交互缺系统消融。

## 3. Motivation
回答"让多轮 agentic RL 真正 work 的因素实际是什么"。把设计空间拆成三支柱逐一受控消融，在情境化文本域(TextWorld/ALFWorld)与软件工程(SWE-Gym)上导出可操作 recipe，并区分增益来自"多轮 formulation"本身还是算法启发式。

## 4. 主要灵感 / 核心直觉
把多轮 agentic 任务形式化为 POMDP；agentic 环境只在命令完成(<eos>)时执行并给奖励，故标量奖励 r_t 赋到该 turn 的 <eos> token、其余 action token 奖励为 0、state token 全 mask 不计 loss。对有 advantage 估计的算法(PPO)用 token-level credit assignment + GAE，使前序 token 经 value bootstrapping 获得非零 advantage。隔离"多轮 formulation"与"算法启发式"的方法：对比偏置(PPO/GRPO/REINFORCE++)与无偏(RLOO)策略梯度——若无偏的 RLOO 也涨，则增益来自 formulation 而非 PPO 启发式。

## 5. 主要解决思路(一段话讲清核心)
非新算法，是三支柱受控实证：(1) Environment——沿 world size/object/quest 三维扫复杂度、测简单→复杂泛化、测任务多样性；(2) Policy——SFT 先验对 RL 收敛的影响、固定预算下 SFT:RL 最优配比、偏置 vs 无偏算法对比；(3) Reward——稀疏 vs 稠密 turn-level 奖励、验证型 vs 模型型奖励、奖励粒度。最终汇成 recipe：课程 + 稳定化偏置策略 + 验证型稠密奖励。

## 6. 方法详解(通俗、分步骤)

- **POMDP 形式化**：history h_t=(u,s_0,a_0,…,s_t)，action 是自然语言 token 序列，env.step 返回 next state/reward/done，reward 赋到 <|im_end|>。
- **Multi-turn PPO**：token-level TD δ + GAE 估每 token advantage；Clipped Surrogate 覆盖全轨迹 token。
- **Environment 消融**：TextWorld 程序化生成 w/o/q 三维(如 w2-o3-q4)，ALFWorld 6 类家务，SWE-Gym 5 类(getmoto/pydantic/mypy/pandas/dvc)；改环境复杂度 + 任务多样性混合，固定总数据量公平比较。智能体须从观察自生成可执行 NL 命令(无 admissible action 提示)。
- **Policy 消融**：TextWorld 金标解作 SFT 示范(与 RL 数据不同 seed 防泄漏)；固定 1000 成本单位(假设 SFT 数据贵 RL 的 10×)扫 SFT:RL 配比；对比 PPO/GRPO/RLOO/REINFORCE++。
- **Reward 消融**：TextWorld 用内置函数造 sparse/dense(steps-per-reward)；SWE-Gym 比验证奖励(二元 vs 单测通过比例) vs 模型评判(CodeRM-8B/GPT-4.1)；长程编程用 GRPO 求稳。

## 7. 实验数据集

- **环境**：TextWorld、ALFWorld(文本版，train 训 / valid-unseen 评，6 类)、SWE-Gym(getmoto/pydantic/mypy/pandas/dvc，随机 90 训 25 评)。
- **模型**：Qwen2.5-1.5B-Instruct、Qwen2.5-7B-Instruct、Qwen3-8B；算法 PPO/GRPO/RLOO/REINFORCE++；rollout temp 0.7。
- **指标**：任务成功率(SWE-Gym 用测试套件通过比例)。

## 8. 实验结果与主要发现

- **(1) 随环境复杂度 scaling**(表 1)：base 17%→3%(同扩空间+物体)；PPO 增益从 base 环境 +88% 降到最复杂 +51%；物体复杂度比空间更难。
- **(2) 简→繁泛化**(表 5/6)：训简单环境对复杂环境有显著迁移(w8-o3-q4 训出的迁移最强，把 w8-o12-q4 提 48%，追平直接训该环境)；编程域同样(Easy→Medium +4.8%/Hard +3.6%)。
- **(3) 多任务训练**(表 7/8)：单类型训练即有 12%(ALFWorld)/7%(SWE-Gym)跨类型泛化；混合训练甚至对看似无关任务也增益(4 类混合超单类 pick&place 专家 +19%/全类 +21%)。
- **(4) SFT 先验**(表 9)：60 示范 + 400 RL episode 达 85%，逼近纯 RL 5000 episode 的 88%；跨域 SFT 先验反而有害(快速策略崩溃)。
- **(5) 最优 SFT:RL 配比**(表 9)：纯 SFT 域内强(w2-o3-q4 95%)但泛化差(w4-o6-q8 55%)；甜点是 60 SFT + 400 RL(域内 85%、复杂泛化 59%)。
- **(6) 算法对比**(表 10)：PPO 与 RLOO 均超 base(证增益来自 formulation 非 PPO 启发式)，但 PPO 最稳/样本效率最高(w2-o3-q4 88% vs RLOO 51%；w4-o6-q8 PPO 59% 而 RLOO/REINFORCE++/GRPO 在 1.5B 上崩溃)；REINFORCE++/GRPO 在 TextWorld 增益微弱。
- **(7) 奖励密度**(表 11)：稠密 turn-level 奖励加速训练，但最优密度依算法——PPO 受益于更频繁反馈(41%→58%)，RLOO 对密度不敏感(稳 55%)。
- **验证 vs 模型奖励**(表 12)：SWE-Gym 上稀疏二元验证仅 4.2%≈base，稠密单测比例验证 22%；模型评判 CodeRM-8B 7.2%/GPT-4.1 9.3%，远逊验证奖励。

## 9. 结果如何支撑其主张
七问对应七组受控实验(固定数据量/预算公平比较)，逐条支撑 recipe 三要素：跨复杂度/多样性的迁移表(表 5–8)支撑"课程 + 简→繁泛化"；PPO vs RLOO 的双双增益(表 10)支撑"增益源于多轮 formulation"，PPO/GRPO 优于 RLOO/REINFORCE++ 支撑"稳定化偏置策略"；表 11/12 支撑"验证型稠密奖励优于模型评判"。无偏 RLOO 作对照是隔离启发式贡献的关键设计。

## 10. 逻辑自洽性(中性评估)
作为经验研究内部自洽：每问受控变量、固定预算/数据量、用无偏 RLOO 做隔离对照，方法论谨慎。需注意：(1) 结论高度绑定所测三个文本域 + Qwen 1.5B/7B/8B，跨域可迁移性受限(作者也强调"非单轮简单外推")；(2) "PPO 优于 GRPO/RLOO"与多轮信用分配/value bootstrapping 强相关，GRPO 在 SWE-Gym(稀疏终局)反而有计算优势——即结论是情境相关的(论文已限定 PPO/GRPO 同为偏置方法仅在稀疏奖励下类比)；(3) 多处增益为小样本(SWE-Gym 90 训 25 评)，绝对成功率低(个位数～二十几%)，统计稳健性存疑;(4) SFT 贵 RL 10× 的成本假设是人为设定，影响"最优配比"结论。

## 11. 残留问题 / 局限

- **域窄**：仅 TextWorld/ALFWorld/SWE-Gym 三个文本域 + Qwen 系列；recipe 的跨域/跨模型族泛化未验证。
- **小样本**：SWE-Gym 仅 90 训/25 评、成功率个位数～22%，方差与显著性未充分报告。
- **情境相关结论**：算法优劣随 reward 稀疏度/环境翻转(PPO 在 dense、GRPO 在 SWE-Gym 稀疏更优)，难给单一普适结论。
- **跨域 SFT 先验有害**：换域初始化导致策略崩溃，限制了迁移学习路径。
- **成本假设主观**：SFT:RL 最优配比依赖"SFT 贵 10×"这一设定。
- **非算法贡献**：是"经验法则汇编"，不提供新的可证明算法。

## 12. 开源代码与框架(链接+框架+代码可得性)

- 仓库 https://github.com/pearls-lab/meow-tea-taro （Apache-2.0；本地已 clone ~11MB；★数 〔待核〕，离线无法确认）。结构：`meow_tea_gym/`(环境，含 `SWE-agent/`)、`meow_tea_train/`(训练)、`meow_tea_experiments/`、`recipes/`、`scripts/`、`docs/`(Read the Docs)。
- **框架 = 封装/vendoring veRL**：仓库 `meow_tea_train/verl/` 直接 vendoring veRL；advantage estimator(grpo/rloo/reinforce_plus_plus/reinforce_plus_plus_baseline/grpo_passk/rloo_vectorized/grpo_vectorized 等)在 `meow_tea_train/verl/trainer/ppo/core_algos.py`(`AdvantageEstimator` 枚举 + `register_adv_est`);docker 基于 `hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0`(README)/`...th2.6.0-cu126-vllm0.8.4...`(docs)。真正"自研"的是环境层(`meow_tea_gym/`)与 environment/reward/policy 三支柱的可配置封装；策略侧 PPO/GRPO/RLOO/REINFORCE++ 均经 veRL 实现。
- 数据/模型权重：论文 Reproducibility 声明承诺随发表开源全部框架/脚本/权重(注：正文为"will release"将来时)。

〔本轮核实/新出入〕

1. 〔修正-时间〕PDF 实为 **arXiv:2510.01132 v2，2025-12-06**(cs.LG)；原分析写"2025-10 preprint"，应更新为 v2/2025-12。
2. 〔确认〕原分析"封装 veRL、非从零自研"的核码订正经仓库再核为真：`meow_tea_train/verl/` 确为 vendored veRL，`core_algos.py` 含完整 advantage estimator 枚举；`meow_tea_gym/` 含 SWE-agent。docker base 与 README 一致(`hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0`)。
3. 〔补充〕原分析未列 REINFORCE++ 对照与表 9–12 具体数值，本轮补全(含 SFT:RL 甜点 60+400、验证 vs 模型奖励 22% vs 7.2%/9.3%)。
4. 〔待核〕原分析"83★"无法离线确认，标 〔待核〕。

sdcl | Self-Distillation Enables Continual Learning (SDFT) | MIT + Improbable AI Lab + ETH Zurich(Idan Shenfeld, Mehul Damani, Jonas Hübotter, Pulkit Agrawal) | 2026-01·arXiv 预印本·v1(2026-01-27, cs.LG) | 主题线 L1(在线策略/自蒸馏)+ L5(持续学习·防遗忘)·相关性 高

**原始论文**:https://arxiv.org/abs/2601.19897 （arXiv:2601.19897v1;项目页 http://idanshenfeld.com/SDFT）

## 一眼看懂
- 🟦 TL;DR:想让模型**学新技能/新知识又不忘旧的**(持续学习)。在线策略 RL 能少遗忘但要 reward(常没有);从示范学只能 SFT,而 SFT 是 off-policy、会灾难性遗忘。SDFT 的招:用**同一个模型**当自己的 teacher——teacher 模式 = 模型 condition 在(query + 一条专家示范 c)上,student 模式 = 模型只看 query;在 student **自己采样**的 on-policy 轨迹上做 token 级 KL 蒸馏,把 off-policy 的示范"软化"成 on-policy 学习信号。teacher 权重默认取 student 参数的 **EMA**(指数滑动平均)。
- 最巧的一步:**抽掉"on-policy 采样"(改成在 teacher 生成文本上离线蒸馏/SFT),增益就大半消失**。§4.6 的关键消融(Fig.6)正是证这点:同一个 teacher,offline distillation < on-policy SDFT。为什么是它:论文反复强调"光有好 teacher 不够,必须在学生自己的轨迹分布上学,才能即时纠错 + 贴近预训练分布 → 少遗忘"(§3.2 第二条件 + §4.6),所以"on-policy"才是真正承重的那根柱子,而非"用 ICL 当 teacher"这个 trick 本身(后者 GKD/context-distillation 早有)。

## 为什么做
- 研究背景:基础模型部署后**静态**,不更新参数去学新技能/内化新知识(§1)。已有共识:on-policy 学习比 off-policy 少遗忘(Shenfeld 2025 "RL's Razor"、Chen 2025),但成熟 on-policy 方案几乎都在 RL 里、需显式 reward。
- 解决的具体痛点:现实多数场景只有**专家示范**没有 reward → 主流是 SFT,而 SFT 本质 off-policy,顺序学新任务会严重遗忘旧能力(§1)。核心矛盾:"只有示范时,怎么拿到 on-policy 学习的好处?"
- 相关工作 & 各自不足:(a) IRL(先从示范学 reward 再 on-policy RL)理论优雅但**不 scale**、要强结构先验(max-entropy/对抗/偏好式各需不同假设)(§2);(b) context distillation(condition 模型当 teacher 蒸给无 context 的自己)——但**通常离线**、且 context 是固定全局 prompt(few-shot/行为准则),在 teacher 分布上监督。SDFT 两点不同:**on-policy**(学生自己轨迹上,teacher 可即时纠错)+ context 是**逐 query 选的具体示范**(实例级条件,能表达细粒度任务意图,而非单一全局先验)(§2 Context Distillation 段)。
- 动机链:要少遗忘 → 要 on-policy → 但只有示范没 reward → IRL 不 scale → 所以用模型的 **in-context learning(ICL)** 能力:condition 在示范上的模型可看作"对该任务近似最优、且仍贴近预训练分布"的策略,直接拿它当 teacher 在学生 on-policy 轨迹上蒸馏。为什么不用更简单的 SFT/离线蒸馏:SFT 把模型推离预训练分布(§3.2 实测 SFT 偏离 base 1.26 nats vs 示范条件化 teacher 仅 0.68 nats),离开预训练分布越远遗忘越多。
- 与最近邻工作的Δ:最像 GKD(on-policy 蒸馏)和 context-distillation。**关键差一点**:把"示范条件化的同一模型(EMA)"当 teacher 来做 **on-policy** 蒸馏,并定位到**持续学习防遗忘**;并给了一条 IRL 等价解释(§3.1)。为什么有用:teacher 既懂新任务(ICL 把示范软化进分布)、又仍锚在预训练分布附近 → 既学到新技能又不偏离原能力(§3.2 Fig.2 右:示范条件化 teacher 的输出分布比 SFT 目标更接近 base)。

## 怎么做 + 靠不靠谱
- 方法流水线(Algorithm 1 在附录,§3 正文):
  1. 输入:示范数据集 {(x_i, c_i)};一个基础模型 π;teacher EMA 率 α。初始 teacher 权重 = student 权重。
  2. 对每个 query x:**student 模式**只看 query,on-policy 采样一条 rollout y ∼ π_θ(·|x)。
  3. **teacher 模式**:同一模型(权重用 EMA)condition 在固定模板注入的(query + 示范 c)上,得 teacher 分布 π(·|x,c)。模板是"这是一个回答示例:<示范>;现在用你自己的回答(含思考过程)回答" —— 论文称此模板足以**避免逐字复制 c**,而是激发模型理解示范意图后自己作答(§3)。
  4. 在采样到的 token 上做 **token 级 KL** 蒸馏,更新 student 参数(teacher 分布视为常数)。
  5. 更新 teacher:ϕ ← αθ + (1−α)ϕ(EMA)。输出:持续更新的 student。
- **⚠ KL 方向:论文正文 vs 实际实现的关键脱节**(本轮基于 PDF + 仓库 README 双重核实):
  - 【原文】§3 与 Eq.(1) 明写最小化 **reverse-KL** D_KL(π_θ(·|x) ‖ π(·|x,c));§3.1 整条"Self-Distillation as Inverse RL"等价推导(隐式 reward r=log π(y|x,c)−log π_k(y|x),In-Context Assumption π*_{k+1}≈π(·|x,c))**只在 reverse-KL 下成立**。
  - 【原文·仓库 README errata 04/07/26】仓库明确声明:"**所有论文结果实际是用 on-policy 采样 + per-token forward KL loss(类似 GKD)产生的**,故这是仓库默认参数,我们将尽快更新 arXiv 澄清。" 即:论文正文的 reverse-KL 叙事与 §3.1 inverse-RL 推导,与**真正跑出全部结果**所用的 forward-KL(mode-covering、不再对应那条 inverse-RL 等价链)**方向相反**。
  - 旁证(同在论文内,但易被忽略):附录 A.1 的实际选用估计器是 "**full analytic per-token estimator**"(逐 timestep 在词表上解析求和),论文自承它"在 sequence 层有偏",而正是这种 per-token 解析形式与 GKD 式 forward-KL 实现一致——这与 README errata 自洽。
  - 〔推断·影响〕这是本方法当前**最大的自洽性瑕疵**:理论框架(inverse-RL)挂在 reverse-KL 上,实证结果却来自 forward-KL。复现/引用时应以"on-policy + per-token forward KL(GKD 式)"为准,把 §3.1 的 inverse-RL 解释视为**事后理论叙事而非实测对应**。
- 逐组件必要性:
  - **on-policy 采样**:核心承重柱。消融充分——§4.6 Fig.6,同 teacher 下 offline distillation 与 SFT-from-teacher 均 < on-policy SDFT;§4.2 Fig.5 右 pass@k(k 到 128)全程领先 → 增益是真技能习得而非熵坍缩。没它 → 退化成普通离线蒸馏,遗忘加重。
  - **示范条件化 teacher(text + answer)**:负责"既懂新任务又贴近 base"。消融充分——§3.2 实测 ToolAlpaca 上无示范 base 仅 42%、给示范 teacher 达 **100%**;附录 A.2 Fig.7:teacher 同时 condition 文本+答案(89% strict)> 仅文本(75%)> 仅答案(37%)。
  - **EMA teacher**:稳定性关键。消融充分——附录 A.3:用 frozen base 当 teacher → 稳但偏弱(跟不上学生进步);用**当前 student 自己**当 teacher → 严重不稳(token 概率小波动经 on-policy 反馈环放大致发散);EMA 折中(Fig.8)。**注意:这意味着方法不是字面"同一当前模型自蒸馏",而是 EMA 副本**。
  - **KL 梯度估计器选择**:有消融——附录 A.1 比较 token-level(有偏高方差)/ full analytic per-token / Rao-Blackwellized(无偏低方差但贵)三者,**实测选 full analytic per-token**(最稳、下游最好);并发现"每 prompt 多采样"几乎无收益却显著加 compute,故定为**单轨迹/prompt**。
  - **mask 前几个 token 的损失**(§5 Learned Artifacts):防学生继承 teacher 的 "Based on the text..." 这类口头禅。论文自承是**启发式补丁**,非原理性解。
- 关键机制/公式(直觉):
  - **ICL 假设**(§3.1 Eq.4):"给一条示范 c,模型 condition 在 c 上 ≈ 该任务的最优下一步策略 π*_{k+1}"。直觉:观察一个示例引发的行为偏移,反映了专家的真实意图——把这个偏移当成"免费的、贴近 base 的最优策略近似"。
  - **为何少遗忘**(§3.2 第二条件 + trust-region 视角):trust-region RL 的最优策略是"在所有达最优 reward 的策略里、离当前策略 KL 最近的那个";示范条件化 teacher 恰好满足"高质量输出 + 贴近 base"(0.68 nats vs SFT 1.26 nats),所以朝它更新走得"近"→ 少遗忘。这是 mode-covering/anchoring 的直觉。
- 实验与证据:
  - 数据集/设置:**Skill Learning** 三域——Science Q&A(SciKnowEval Chemistry L-3)、Tool Use(ToolAlpaca)、Medical(HuatuoGPT-o1 stage-1 训/stage-2 评);**Knowledge Acquisition**——2025 自然灾害 Wikipedia 语料(超 cutoff,约 200K tokens)生成 QA + OOD"间接问题" + 全文超 context 时与 oracle-retriever RAG 对照;**遗忘评测**——HellaSwag/TruthfulQA/MMLU/IFEval/Winogrande/HumanEval 均值(用 EleutherAI lm-eval-harness 指定 commit);**顺序学三技能** Tooluse→Science→Medical。主 base = Qwen2.5-7B-Instruct;scaling 用 Qwen2.5 家族 3B/7B/14B;§4.5 reasoning 用 Olmo-3-7B-Think。greedy 评测、3 seeds 报均值+95%CI(附录 B)。
  - 支撑核心主张的关键数字:Knowledge Acquisition(Table 1)SDFT strict **89**/lenient **100**/OOD **98**,远超 SFT(80/95/80)、逼近 Oracle RAG(91/100/100),CPT 仅 9/37/7 —— OOD 间接问题 98 vs 80 是"真内化 vs 死记"最有力证据;Skill Learning(Fig.4)三域 Pareto 全优(新任务 acc 与旧能力保持同时优于 SFT/DFT/Re-invoke);顺序学(Fig.3)SDFT 累积不退化、SFT 学新忘旧震荡;规模效应(Fig.5 左)**3B=−3.3 / 7B=+4.0 / 14B=+6.9**(增益随 ICL 能力单调增);§4.5(Table 2)answer-only 医疗数据上 SFT 掉点 31.2→23.5 且缩短输出(4612→3273 tok),SDFT 反升到 **43.7**(4180 tok,保住长 CoT)。
  - baseline 公平吗:Skill Learning 与 SFT/DFT(用重要性采样把离线当 on-policy)/Re-invoke(SFT 后再用 base 在通用 prompt 上 on-policy 蒸馏以恢复能力)比;Knowledge 与 CPT/SFT/oracle-RAG 比;每个 baseline 都做了超参 sweep 取最优验证 checkpoint —— 对照设计较公平。
  - "看着强但没回答核心问题":3B 反不如 SFT(−3.3),暴露方法对 base ICL 能力强依赖;Knowledge 语料仅约 200K tokens 窄域(2025 自然灾害),**大规模知识注入未验证**;§5 自承"难以把非推理模型改造成显式 CoT 模型"(方法不擅长需要"生成模式根本性改变"的适配)。
- 假设与失效边界:
  - 显式假设【原文】:ICL 假设(示范条件化 ≈ 最优策略,§3.1 两条件:Optimality + Minimal Deviation);base 模型有足够强 ICL 能力;有(逐 query 可配的)专家示范。
  - 隐式假设【推断】:示范是"专家级"且与 query 一一匹配(§实验里 Science 用 GPT-4o 生成、Tool 用数据集自带);teacher 的"软化"不会把错误意图也软化进来(对噪声/非专家示范未测)。
  - 何时失效:【原文】小模型 ICL 弱(3B −3.3,§4.4/§5);需要根本改变生成模式时(非推理→推理,§5)。【推断】示范噪声大/非专家时;teacher EMA 跟不上快速分布漂移时(§A.3 用 student 自身当 teacher 会发散,暗示反馈环稳定性边界)。
- 祛魅总结【推断】:
  - 真贡献:**把"on-policy 少遗忘"从 RL 搬到"只有示范"的场景**,并用一系列干净消融(on-policy vs offline、text+answer vs 单一、EMA vs base/self、pass@k 排除熵坍缩、3B/7B/14B 单调)系统坐实——尤其 OOD 间接问题和 answer-only 保住长 CoT 两个结果很有说服力。定位(continual learning from demonstrations)是其最大增量。
  - 包装/高估:核心 trick(ICL 条件化 teacher + on-policy 蒸馏)在 GKD/context-distillation 早有,新意主要在定位与实验;§3.1 inverse-RL 理论框架因 **KL 方向 errata** 与实测脱节,属"包装性理论"成分偏高〔基于 README errata〕;"establishing on-policy distillation as a practical path"在 200K-token 窄域知识 + 7B 上成立,但规模/领域外推被高估。
  - 低估之处:EMA teacher 的"反馈环稳定性"其实是个普适且重要的发现(self-as-teacher 会发散),但被放进附录;单轨迹/prompt 就够(多采样无益)对成本敏感的持续学习很实用,也未被突出。

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号:示范条件化 teacher(EMA)的 **token 级分布**(per-token forward KL,据 errata;论文正文写 reverse-KL)。
  - 改什么:**参数**(full fine-tuning 整个模型;改 logits 分布)。
  - 何时改:**在线 per-step**(每步现采 student rollout、现算 teacher 分布、现更新 + EMA 同步)。
  - 免梯度?:否。
  - 记忆-技能生命周期:**无显式记忆/技能库**;"新知识/新技能"直接写进**参数**(知识内化进权重,OOD 间接问题验证);示范 c 是 per-query 临时 context,不持久存储。共享/遗忘靠参数本身。
  - 防遗忘机制:**核心卖点且实测**——on-policy 更新使学生保持贴近预训练分布(§3.2 KL 0.68 vs SFT 1.26 nats),从而在 6 个通用基准上保持(Fig.4 Pareto、Table 5 分项)+ 顺序学不退化(Fig.3)。这是"靠 on-policy + anchoring 隐式防遗忘",非 EWC/replay 式显式机制。
- ⑦ 开源代码 + 框架/harness:https://github.com/Continual-Intelligence/Self-Distillation (README 现给此 clone URL;v1 记 idanshen/Self-Distillation,疑迁移/镜像;项目页 idanshenfeld.com/SDFT)。已 clone。框架 **TRL**(README 装 trl;`distil_trainer.py` 定义 DistilTrainer,用 `trl.extras.vllm_client.VLLMClient` 做 on-policy 生成;遗忘指标用 EleutherAI lm-eval-harness 指定 commit 03c44adc)。入口 `main.py`(README 示例 `--model_name Qwen/Qwen2.5-7B-Instruct --learning_rate 5e-5 --num_train_epochs 2`);配置 `distil_config.py`(alpha 控 forward/reverse/JSD、beta 控 KL 系数与是否载 reference、EMA ref-sync)。
  - **〔待核·关键〕KL 方向**:README errata(04/07/26)明言全部结果实为 **forward KL(GKD 式)**,且仓库默认即 forward KL;论文正文写 reverse KL。复现以仓库默认(forward KL)为准。
  - 〔待核〕lr 默认:README 训练示例用 **5e-5**;v1 分析曾记 main.py 默认 2e-5。本轮未重新打开 main.py 源码核对默认值,以 README 示例(5e-5)记录、标待核。
- 💰 资源/成本与可扩展性:单卡 **NVIDIA H200**(附录 B);full fine-tuning;**单 on-policy rollout/prompt**(附录 A.1 实测多采样无益);相比 SFT 约 **2.5× FLOPs、约 4× wall-clock**(因要 on-policy 生成,§5 Computational Costs),但若计入 Re-invoke 那种"先 SFT 再恢复"的多阶段流程,SDFT 反而可能更省总时间;比 GRPO 省(单生成 vs group 采样)且 token/logit 级信用比 GRPO 的轨迹级 advantage 更稠密(§5)。
- 🎯 对"探索-巩固"对标:**强支撑(尤其"巩固/防遗忘"维度)+ 可借组件**。判定:SDFT 几乎是我们"巩固"分支的现成范式——"把成功经验/新技能固化进参数且不遗忘",且证明了**on-policy + 贴近预训练分布**是少遗忘的关键机制(正中我们 idea 的"巩固=固化进参数/记忆且不遗忘")。依据:Fig.3 顺序学不退化 + §3.2 anchoring 机制。可直接借的积木:① **EMA self-teacher** 作稳定的"自蒸馏 teacher"(避免 self-as-teacher 发散,可移植到 student 自选恢复分支后的固化阶段);② "示范条件化 → on-policy 蒸馏"把任意 privileged context(我们的 teacher 脚手架/MTP 前瞻提示)转成 on-policy 信号的通用配方;③ pass@k 不坍缩 + OOD 间接问题作为"真内化 vs 死记"的探针。缺口:SDFT 偏"巩固",**探索/选路**几乎不涉及(它假设有专家示范、不做岔路口的路径选择);也无 MTP/前瞻、无 student 自选恢复分支。
- 🔭 开放问题/未来方向:
  - 【原文】(§5/§11)与 on-policy RL 结合(SDFT 当 RL 前的初始化,或同时混合示范+reward 信号);进一步降残余遗忘(on-policy 仍有少量退化);从**非专家/噪声示范**或非结构化数据(用户对话)学习;更原理性地解决"继承 teacher 口头禅"(替代 mask-前几 token 的启发式);使方法能支持更激进的行为改变(如非推理→推理)。
  - 【推断】把 §3.1 理论与实测 KL 方向对齐(改写为 forward-KL 的解释或换实现验证 reverse-KL 是否同样有效);大规模、跨多领域知识注入的可扩展性验证;EMA 率 α 对稳定性-收敛的系统刻画(目前只给"base/self/EMA"三档定性)。

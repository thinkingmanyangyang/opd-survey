glm45 | GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models | Zhipu AI 智谱 & 清华（GLM-4.5 Team）| arXiv 2508.06471v1, 2025-08-08 · 技术报告 | 主题线 L1/L3/L4（自蒸馏统一 + GRPO-no-KL + agent RL）·相关性 高（多专家→统一的 self-distillation 与 RL 间迭代自蒸馏，对标本课题「巩固」）

**原始论文**：https://arxiv.org/abs/2508.06471

## 一眼看懂
> 一句话导读：这是 GLM-4.5 的模型技术报告;对本课题最有用的是它的后训练范式——先分域用 RL 把各专家能力推到极致,再用 self-distillation 把多专家合成一个双模通才,agent 侧还周期性把 RL 成果蒸回 SFT 起点。

- 🟦 TL;DR：GLM-4.5 是开源 MoE(总参 355B / 激活 32B,另有 106B 的 Air)**混合推理**模型(支持 thinking + direct 双模,即深思与快答两种模式)。**注意它是模型技术报告,不是单一方法论文。**
  - 与本课题最相关的是它的后训练范式,分**两阶段**:
    - ① Expert Training:分别用冷启 SFT + 专家 RL,训出 Reasoning / Agent / General-chat 三个专家。
    - ② Unified Training:用 **self-distillation**(自蒸馏)把多个专家整合进一个通才。
  - agent RL 部分因为 RL 昂贵,还在 RL 轮次之间插入 **iterative self-distillation**——用 RL 后模型的输出替换冷启数据再做 SFT,把起点抬高后继续 RL【原文 §3, §3.3.2】。
- 最巧的一步：**把"多专家各自 RL 强化 + 再 self-distillation 合并"做成统一通才的生产线**。
  - 反过来验证:抽掉 Unified 阶段的 self-distillation,三个专家就无法在不互相打架的前提下融进一个支持双模的模型。
  - 精髓:**先在各域用 RL 把能力推到边界(探索 / 强化),再用蒸馏把这些边界能力固化进一套参数(巩固)**。
  - agent 侧的 iterative self-distillation 又进一步把"RL 探索的成果"周期性地"固化成新的 SFT 起点",让昂贵的 RL 不必每次从头慢爬。这正是"探索→巩固"在工业级 pipeline 里的体现【原文 §3, §3.3.2】。

## 为什么做
> 一句话导读：目标是单个开源模型同时擅长 Agentic/Reasoning/Coding 三域且支持双模;难点在三域如何共存、RL 数据如何不随能力演化而失配、多阶段训练如何不遗忘长上下文、agent RL 如何不那么慢——self-distillation 与一堆 RL 补丁就是答案。

- 研究背景：LLM 正从知识库走向通用问题求解器。目标是让单个开源模型同时擅长 **Agentic / Reasoning / Coding (ARC)** 三类互相关联的能力——此前 o1/o3、Claude 在单域很强,但开源界缺一个三域全能的模型【原文 §1】。
- 解决的具体痛点(5 个)：
  - (1) 三域能力如何在一个模型里共存,且支持"深思 + 快答"双模。
  - (2) RL 训练中,模型能力随训练演化,会与**静态数据失配**——后期题太简单(reward 全 1)、早期题太难(reward 全 0),两种情况下都没有 reward 方差,也就没有梯度。
  - (3) 多阶段渐增输出长度的 RL,会让模型在短长度阶段"遗忘"长上下文能力,造成**不可逆**的性能下降。
  - (4) agent RL 极其耗时。
  - (5) function call(函数调用)参数里含代码时,JSON 转义负担很重。
  【原文 §3.1, §3.2, §3.3】。
- 相关工作 & 各自不足（来龙去脉 + 精确差异）：
  - ① **闭源单 / 多域强模型**:OpenAI o1/o3、Anthropic Claude Sonnet/Opus 4——单域或综合很强但闭源,开源界缺三域全能。
  - ② **同期开源报告(同样走"专家 / 分域 + RL")**:DeepSeek-R1、Kimi K2、Qwen3-235B——架构借鉴 DeepSeek-V3 / Kimi K2 的 MoE,但 GLM-4.5 **减宽增深**(更少 routed experts、更多层),还用了 2.5× 注意力头、QK-Norm、loss-free balance routing、partial RoPE;后训练上,它把 **self-distillation 当作"多专家 → 统一双模通才"的核心粘合剂**,这点讲得最显式。
  - ③ **多阶段渐增长度 RL**(Polaris 式 [25])——GLM-4.5 §3.2 明确**反对**,主张单阶段 64K 直接训更好(多阶段会 unlearn 长上下文、造成不可逆掉点)。
  - ④ **GRPO [31] 等 group-based RL**——GLM-4.5 在其上**去掉 KL loss 项**,并加了难度课程、token-weighted loss、动态温度等一系列补丁。
  - 这些对比都服务于本报告的核心问题:如何把分域专家无损地融成一个双模通才【原文 §1, §2, §3.2】。
- 动机链：
  - 要单模型全能 ARC → 分域训专家最容易把单域推到极致。
  - 但最终要交付一个模型 → 用 self-distillation 把多专家合成通才。
  - agent RL 太慢 → 在 RL 之间做迭代自蒸馏抬起点。
  - 数据 / 课程 / 长度上各打补丁(难度课程、单阶段 64K、动态温度),防失配与遗忘。
- 与最近邻工作的Δ（精确差异）：相对 Qwen3 / DeepSeek-R1 等同期报告——大家都走"专家 / 分域 + RL",但 GLM-4.5 有 5 点不同:
  - (1) 把 **self-distillation 当作"多专家 → 统一双模通才"的显式核心**;
  - (2) 明确主张 **单阶段 64K RL 优于多阶段渐增长度**;
  - (3) **GRPO 去 KL**;
  - (4) 难度课程的二阶段只用"已验证正确答案池";
  - (5) agent 侧引入 **iterative self-distillation**(RL ↔ 蒸回 SFT 交替)。

## 怎么做（细到可复现的程度——技术报告，后训练脚本未开源）
> 一句话导读：流程是冷启 SFT → 各域专家 RL(GRPO 去 KL + 难度课程 + 单阶段 64K + token-weighted loss)→ 把专家蒸成统一通才(Overall SFT)→ agent RL 与 iterative self-distillation 交替 → general RL。下面逐步拆开。

### 0. 后训练总体两阶段（§3）
- Stage 1 Expert Training:构造 Reasoning / Agent / General-chat 三个专家。
- Stage 2 Unified Training:用 self-distillation 把三个专家整合成一个支持"深思 + 快答"双模的通才。
- 两个阶段的开头都先做 SFT。

### 1. Cold-start SFT（§3.1）
- 用少量长 CoT 数据给每个专家打底,确保进 RL 前已具备基础的 chat / 推理 / 工具使用能力。
- Rejection Sampling(拒绝采样)从专家采样时做多级过滤,顺序是:去重复 / 过短 / 截断 / 非法格式 → 客观题做正确性校验 → 主观题用 reward model 过滤 → 工具调用校验是否到达期望终态。
- prompt selection(prompt 筛选)有两招:① 去掉响应长度后 50% 的 prompt,数学 / 科学 +2~4%(只用一半数据);② 对难 prompt 生成 4 个响应,再 +1~2%。

### 2. Expert RL（§3.2，以 Reasoning RL 为例）
- **RL 目标**：build on GRPO **去掉 KL loss 项**。对每个问题 \(x\) 采 \(K\) 条轨迹 \(\{y_1,\dots,y_K\}\sim\pi_{\text{old}}\)，优化
\(\displaystyle L_{\text{RL}}(\theta)=\mathbb{E}_{x\sim D}\Big[\frac1K\sum_{i=1}^{K}\big(r(x,y_i)-\bar r(x)\big)\Big],\qquad \bar r(x)=\frac1K\sum_{i=1}^{K}r(x,y_i),\)
即优势 = 单条奖励 − 组内平均奖励(group-wise、critic-free)。**只对模型生成的 token 计 loss,环境反馈不计**(agent 设定)。〔注:报告写的目标式没有显式含 importance ratio / clip,呈最简的 group-mean-baseline REINFORCE 形式;推断实际实现仍带 PPO 式 clip〕。
- **难度课程二阶段**(Fig.5):随能力升级换数据;二阶段严格只用"已验证正确答案池"来降噪,从而持续突破天花板(AIME24 81.8% → 83.4%)。
- **单阶段 64K RL**(Fig.6,关键的防遗忘经验):直接在 64K 输出长度训,**不**走多阶段渐增长度。原因是 SFT 已经把模型 condition 到 64K,若再引入短长度 RL 阶段,会让模型 unlearn 长上下文、平均输出变短,造成**不可逆**掉点(多阶段 80.6% vs 单阶段 83.4%)。
- **token-weighted mean loss**(用于 code RL,Fig.7 左):比 sequence-mean 收敛更快、缓解长度偏置、抑制"base case"灌水(LiveCodeBench 46.5 vs 46.3,且更快)〔与 DAPO/HAPO 的 token-mean 取向一致〕。
- **科学 RL 数据质量**(Fig.7 右):只用专家验证的高质量选择题得 65.8,而混杂数据只有 62.9。
- **动态采样温度**:reward 稳定就判为收敛、升温来促多样性;并用 held-out 验证"升温不掉 >1%"做质控。

### 3. Overall/Unified SFT = self-distillation（§3.1 Overall SFT）
- 从各专家采**百万级**样本(覆盖 reasoning / general chat / agentic / long-context),用 128K 上下文训 base,把多专家能力蒸成一个混合推理通才。
- **混合"带思考"与"无思考"数据**来实现双模(chit-chat 等不需要长思考的域,用无思考数据)。
- function call 用 XML-like 的特殊 token 标签来包裹键值,大幅减少代码段的转义负担。

### 4. Agentic RL + Iterative Self-distillation（§3.3.2，交替）
- 数据:web-search 用多跳知识图谱自动合成 + human-in-loop 选择性混淆;SWE 用 GitHub PR/issue + 可执行单测,沙箱隔离。
- **Outcome 监督 + process format penalty**:web 搜索用最终答案的准确率当作整条 trace 的奖励;若工具格式错,就 halt(中止)该 trace 并给 **0 奖励**。
- **Iterative Distillation**:因为 agent RL 慢,采取循环——先 RL 抬 agent 性能 → 到一定步数 / plateau → 用 RL 模型的输出替换原冷启数据做 SFT,得到**更强的 SFT 模型** → 再 RL、逐步增难。本质是把"RL 探索成果"周期性地"廉价复用"为 SFT 起点。
- test-time scaling:agent 任务靠增加与环境的交互轮数来扩 test-time compute(Fig.8,BrowseComp 随 browsing effort 平滑上升)。

### 5. General RL（§3.4）
- 多源反馈:rule-based + human(即 RLHF,用 5000 prompt 跨 7/33/139 类训 reward model)+ model(即 RLAIF,按是否有 GT 分两套 rubric)。
- Instruction Following RL:7 大类 / 151 小类约束,用"确定性规则 + RM + critique"三件套来抗 reward hacking。
- Function Calling RL:分 step-wise rule-based 与 end-to-end multi-turn 两种。
- pathology RL:专治语言混杂 / 重复 / 格式问题。

### 逐组件必要性（均在**小实验模型**上做受控消融，非 GLM-4.5 本体）
- **难度课程二阶段**：Fig.5 证明持续突破天花板；二阶段只用已验证正确答案池。有消融。
- **单阶段 64K RL**：Fig.6 证明直接 64K 更好;多阶段 unlearn 长上下文致**不可逆**下降。有消融。**这是「防遗忘」的关键经验**。
- **token-weighted mean loss**：Fig.7 左证明收敛更快、缓解长度偏置。有消融。
- **科学 RL 数据质量**：Fig.7 右 65.8 vs 62.9。有消融。
- **动态采样温度**：有机制描述 + 质控阈值。
- **self-distillation / iterative self-distillation**：**作为核心范式陈述，但本报告未给「有 vs 无自蒸馏」的对照消融数字**——靠最终基准背书。〔待核：self-distillation 相对「直接多任务 SFT/RL」的净增益无独立量化〕。

## 靠不靠谱
> 一句话导读：最终基准很硬(12 基准综合第 3、agentic 第 2)且参数更省,但要打个折——所有方法层的消融都在自家小代理模型上做,GLM-4.5 本体只有最终分数,且 self-distillation 的净增益没有单独量化。

- 实验与证据：
  - GLM-4.5(355B/32B)在 12 个基准上综合排第 3、agentic 排第 2;参数远少于多数对手。
  - 关键数字:**TAU-Bench 70.1 / AIME24 91.0 / SWE-bench Verified 64.2 / BFCL v3 77.8 / GPQA 79.1 / BrowseComp 26.4**。
  - **注意**:消融曲线都在"smaller experimental model"(更小的实验模型)上做,**不在 GLM-4.5 本体**上——也就是说,方法有效性的微观证据来自代理模型,本体只有最终分数。
- baseline 公平吗：最终基准对比是同期主流模型的横评(截至 2025-07-28),属标准 leaderboard 式;但**方法层消融用的 baseline 是自家小模型**,外部可比性弱;且没有随机种子 / 方差。
- 假设与失效边界：
  - 【原文】(1) 单阶段 64K RL 的前提是"SFT 已把模型 condition 到生成 64K 长响应"——若 SFT 没充分做长上下文化,则未必成立;(2) 难度课程二阶段依赖"已验证正确答案池";(3) 消融在小模型上做,报告自己也承认"曲线基于较小的实验模型"。
  - 【推断】(4) **self-distillation"整合多专家"可能有能力相消 / 折中**——把三个域的专家压进一个模型,单域峰值能力大概率低于专家本身(报告没量化这层损失,只报通才的最终分);
  - 【推断】(5) iterative self-distillation 反复"RL → 蒸回 SFT",可能累积分布偏移、或过拟合 RL 偏好,长期稳定性没讨论;
  - 【推断】(6) 作为**模型发布仓,它不含后训练训练脚本**,方法细节(λ、课程切点、自蒸馏数据配比)多数未公开,复现性受限;
  - 【推断】(7) 模型含一层 MTP 层,但**只用于推理时的 speculative decoding 加速**(预训练 MTP loss 权重 λ=0.3→0.1,§2.4),不是训练前瞻信号——与本课题"MTP 作前瞻训练探针"无直接交集(只是同名机制的不同用途)。
- 祛魅总结：
  - 真贡献有三块:**(a) 工业级开源 ARC 全能 MoE + 双模混合推理的完整后训练配方;(b) "多专家 RL → self-distillation 统一"与"RL ↔ iterative self-distillation 交替"这两套自蒸馏范式的工程化;(c) 一批实用经验(单阶段 64K 防遗忘、GRPO 去 KL、难度课程、token-weighted loss、动态温度)。**经验密度高,对工程落地有参考价值。
  - 需打折的:(1) **它是技术报告而非方法论文**——self-distillation 的净增益没有独立消融,方法证据多来自小代理模型;(2) 后训练脚本没开源(只发模型权重 + slime 框架),关键超参缺失;(3) "多专家 → 统一"的单域能力损失没量化;(4) MTP 只是加速、与前瞻训练无关,别误读。
  - 【推断】对"探索-巩固"而言,这是**工业级最贴近"巩固"范式的一篇**——但它的"巩固"是粗粒度的"整模型蒸馏 / 重 SFT",不是 token / 路径级的精细固化。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号** = 多源——专家输出(self-distillation 的 SFT 交叉熵)、可验证 outcome 奖励(reasoning/agent RL,GRPO 去 KL,优势 = 单条奖励 − 组内均值)、RLHF/RLAIF 偏好(general RL)。
  - **改什么** = 策略参数 θ(SFT + RL 交替)。
  - **何时改** = 分阶段(冷启 SFT → 专家 RL → 统一 SFT → agent RL ↔ iterative 自蒸馏 → general RL)。
  - **免梯度?** = 否,SFT + policy gradient。
  - **记忆-技能生命周期** = **把多专家技能蒸馏固化进单套参数**(无外部记忆 / 技能库);iterative self-distillation 把 RL 探索成果周期性固化为新的 SFT 起点。
  - **防遗忘机制** = **显式且关键**——单阶段 64K RL 避免短长度阶段"unlearn"长上下文(图6 是不可逆下降的反例);难度课程防 reward 失配;但跨专家融合的遗忘没量化。
- ⑦ 开源代码+框架/harness：
  - 模型 / 权重仓 https://github.com/zai-org/GLM-4.5(**仅模型发布仓,不含后训练训练脚本**;含 function call 模板等推理侧实现)。
  - RL 框架 **Slime** 单独开源:https://github.com/THUDM/slime。
  - 框架是 **Slime**,组成:Training 用 Megatron 读 Data Buffer + 同步参数;Rollout 用 SGLang + Router 生成数据 / 奖励 / verifier 输出并写回 Buffer;Data Buffer 做桥接。再加 **BF16 训练 / FP8 推理**(在线 block-wise FP8 量化来加速 rollout)+ Ray 调度 + **agent 用全异步解耦**(rollout / training 的 GPU 分离、Docker 高并发隔离环境)。
  - **CloneTier=B:仅记录链接、不 clone 巨型权重仓**(决策①)。〔注:slime 是 RL 框架,不等于完整后训练脚本;社区微调多用 LLaMA-Factory / ms-swift〕
- 💰 资源/成本与可扩展性：
  - 规模 355B/32B-MoE + 106B Air。
  - 预训练:23T tokens、Muon optimizer、batch warmup 16M→64M tokens、lr 2.5e-4→2.5e-5、RoPE base 10k→1M(32K 起)、mid-training 扩到 128K;RL 在 64K 输出长度直接训。
  - **原文没给具体 GPU 数 / 训练 token 成本明细**(技术报告未公开算力)。
  - slime 支持 colocated 同步(reasoning/code)+ disaggregated 异步(agent/SWE)双模,以省 GPU 空转。
- 🎯 对"探索-巩固"对标：**强对标(工业级"巩固"范式 + 可借 pipeline 思想),但粒度粗、机制不同**。
  - ① 直接支撑"巩固":"多专家各自 RL 探索 → self-distillation 合并固化进一套参数"就是本课题"巩固 = 固化进参数且不遗忘"的工业实现。其中 **iterative self-distillation(RL ↔ 蒸回 SFT)是"探索-巩固"循环最清晰的工程案例**——RL 探索得到更好轨迹 → 蒸馏固化为新起点 → 再探索。
  - ② 可借组件(4 个):
    - (a) "RL 探索成果周期性蒸回 SFT 起点"可直接用于 TSRD 的训练节律(student 探索出的成功路径 → 固化 → 再探索);
    - (b) "单阶段 64K 防 unlearn 长上下文"是防遗忘的硬经验;
    - (c) GRPO 去 KL + 难度课程 + 动态温度是即插即用的 RL 配方;
    - (d) 双模(thinking/direct)混合训练对"何时深思 / 何时快答"有参考。
  - 竞品 / 张力 & 缺口:
    - (a) GLM-4.5 的"巩固"是**整模型粗粒度蒸馏 / 重 SFT**——把整个专家分布蒸进通才,**不区分关键步 / 路径分叉、无单点接管、无 teacher 稀疏脚手架**;而 TSRD 主张稀疏脚手架 + path-recovery 单点接管 + MTP 前瞻;
    - (b) 这里的 self-distillation 是"自己的强版本教自己"(teacher = RL 后的自身),与 TSRD"teacher 当稀疏脚手架教探索 / 选路"的精细监督定位不同;
    - (c) MTP 只是加速、不是前瞻信号。
  - 一句判定:**GLM-4.5 提供了"探索(RL)→ 巩固(self-distillation / 重 SFT)"循环的最强工业背书,以及可借的 pipeline 节律(尤其 iterative self-distillation 与单阶段 64K 防遗忘);但它的巩固是整模型粗粒度蒸馏、无 teacher 脚手架 / 无路径级精细化 / MTP 仅加速——是 TSRD 的宏观范式参照系,而非微观机制竞品。**
- 🔭 开放问题/未来方向：
  - 【原文】没设独立的 open problem 章(技术报告);隐含方向 = 继续 scale RL、扩展更复杂的 agent 环境(slime 的异步设计正为此铺路)。
  - 【推断】(1) **self-distillation 整合多专家的单域能力损失需要量化**——能否做"无损 / 近无损"合并(正是 gopd/G-OPD 用 ExOPD 试图解决的方向,可与本报告对照);
  - 【推断】(2) iterative self-distillation 的收敛性 / 分布偏移累积的上界;
  - 【推断】(3) 把粗粒度的整模型蒸馏细化到**关键步 / 路径级**(TSRD 方向)——在 self-distillation 时只对关键决策 token 用更强的监督;
  - 【推断】(4) MTP 从"推理加速"升级为"训练期前瞻探针"(本课题 mtp_opd 的核心设想,GLM-4.5 未触及);
  - 【推断】(5) 后训练脚本未开源,社区复现需补全。

读到PDF? 是（PyMuPDF 全文 26 页;空密码解密;§3 后训练 + GRPO 目标式 + 各 RL 消融图 Fig.5-9 全核）｜L线 L1/L3/L4｜对标结论 强对标(宏观范式):「专家RL探索→self-distillation巩固」+「RL↔iterative自蒸馏」是探索-巩固的工业实现,单阶段64K防遗忘可借;但巩固是整模型粗粒度蒸馏、无teacher脚手架/无路径级精细化/MTP仅加速,是宏观参照非微观竞品｜残留待核 2（self-distillation 净增益与多专家融合的单域能力损失均无独立量化）

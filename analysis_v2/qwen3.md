qwen3 | Qwen3 Technical Report | 阿里巴巴 Qwen Team(An Yang 等;大厂旗舰模型技术报告,Qwen2.5→Qwen3 谱系) | 2025-05 预印本·arXiv(2505.09388,v1 2025-05-15) | 主题线 L1(on-policy 蒸馏实践)+ L2(统一 SFT-RL 四阶段)+ L4(Agent/工具 RL)·相关性 高(大厂技报里少见的**明确 on-policy logit-KD** + 受控 OPD-vs-RL 对照)

**原始论文**:https://arxiv.org/abs/2505.09388

## 一眼看懂
- 🟦 TL;DR:Qwen3 是覆盖 0.6B–235B 的 dense+MoE 系列,核心创新=把 thinking(复杂多步推理)与 non-thinking(快速响应)统一进**单一模型**,支持 `/think`、`/no_think` 动态切换与 **thinking budget**(推理 token 预算控制)。旗舰模型走**四阶段后训练**(Long-CoT 冷启动 SFT → Reasoning RL(GRPO)→ Thinking Mode Fusion(继续 SFT 融合双模)→ General RL);轻量模型则用 **Strong-to-Weak Distillation**(先 off-policy response 蒸馏打底,再 **on-policy 蒸馏**:学生自采序列 + 与教师 logits 对齐最小化 KL),只需四阶段约 **1/10 GPU 时**,且 Pass@1(即时性能)与 Pass@64(探索能力)双双更优。【原文 §4 行 1074-1095,§4.5 行 1236-1249】
- 最巧的一步(对本综述而言):**on-policy logit 蒸馏这一步**,以及它被一个**受控对照实验(Table 21)**坐实。在同一个"off-policy 蒸馏后的 8B checkpoint"上,分别接 RL 与 on-policy 蒸馏:RL 花 17,920 GPU·hr 把 AIME'24 提到 67.6 但 **pass@64 纹丝不动(90.0→90.0)**;on-policy 蒸馏只花 **1,800 GPU·hr(~1/10)** 就把 AIME'24 提到 74.4 且 **pass@64 升到 93.3**。即:**蒸馏在更低成本下同时赢了 Pass@1 和 Pass@64,而 RL 提不动 pass@64(探索边界)**。抽掉"教师 logits"(改纯 RL),探索潜力就上不去——这是"强教师 logit 指引能扩展学生探索空间"的硬证据。【原文 §(On-Policy Distillation 节)行 3069-3114,Table 21】

## 为什么做
- 研究背景:推理模型(o1/DeepSeek-R1)靠 test-time scaling 大幅提升;社区需要既能 chat 又能深推理、且尺寸全覆盖的开源系列。【原文 §1 行 26-56】
- 解决的具体痛点:① 过去需为 chat 模型(如 GPT-4o)与专用推理模型(如 QwQ-32B)**分别部署/训练**;② 为每个小模型独立跑完整四阶段后训练**计算开销大、开发成本高**。【原文 §4 行 1080-1083,1087-1093】
- 相关工作 & 各自不足:报告引 DeepSeek-R1/o1 等推理模型与 GRPO(Shao et al.)作基础;隐含对照="每个小模型独立四阶段"(贵)与"纯 RL 提升小模型"(Table 21 证明其 pass@64 提不动)。报告未单列竞品综述(技报体例)。
- 动机链:① 用统一框架(thinking/non-thinking + thinking budget)免去多模型切换与重复部署 → ② 用 thinking budget 自适应分配推理算力 → ③ 用旗舰大模型知识(strong-to-weak 蒸馏)大幅降低轻量模型后训练成本(1/10 GPU 时)且兼顾即时性能与探索能力。【原文 §4 行 1075-1093】
- 与最近邻工作的Δ:vs "每小模型独立四阶段 RL"——strong-to-weak 蒸馏(尤其 on-policy logit-KD)更省(1/10)且 pass@64 更优;vs 纯 off-policy response 蒸馏——再加 on-policy 阶段(学生自采 + logit-KL)进一步提性能与探索(Table 21 三行递进:off-policy → +RL → +on-policy distill,后者最优)。关键有用点:**on-policy 蒸馏的 pass@64 提升是 RL 给不了的**(行 3079-3080)。

## 怎么做 + 靠不靠谱
- 方法流水线:
  - **旗舰四阶段(Figure 1,§4.1-4.4)**:① **Long-CoT Cold Start**——精选数学/代码/逻辑/STEM 可验证题,两阶段过滤(query 过滤:Qwen2.5-72B-Instruct 剔不可验证/无需 CoT 即对/多子问题;response 过滤:用 QwQ-32B 生成 N 候选,剔错答/重复/猜测/思考-总结不一致/语言混杂/疑似泄漏),少量 SFT 只植入基础推理模式、**刻意少样本少步数**保留 RL 空间;② **Reasoning RL**——筛 **3,995** 条 query-verifier 对(未用于冷启动、可学、尽量难、覆盖广),用 **GRPO**;大 batch + 多 rollout + **off-policy** 提样本效率、**控熵**稳态;Qwen3-235B-A22B 在 **170 步内 AIME'24 70.1→85.1**;③ **Thinking Mode Fusion**——在 Reasoning RL 模型上继续 SFT 融合 non-thinking,thinking 数据由 Stage-2 模型对 Stage-1 query **拒绝采样**生成(防破坏 Stage-2 性能),设计 `/think`、`/no_think` chat 模板 + 空 think 块,thinking budget 为自然涌现能力;④ **General RL**——覆盖 20+ 任务的奖励系统(指令遵循、格式遵循、偏好对齐、**Agent 工具调用含真实环境多轮反馈**、RAG),三类奖励(规则奖励 / 带参考答案的模型奖励(Qwen2.5-72B-Instruct 打分)/ 无参考偏好奖励模型)。
  - **Strong-to-Weak Distillation(§4.5,5 dense:0.6/1.7/4/8/14B + 1 MoE:30B-A3B)**:(1) **Off-policy 蒸馏**——用教师在 `/think` 与 `/no_think` 两模式下的输出做 response 蒸馏,打底基础推理 + 模式切换;(2) **On-policy 蒸馏**——**学生自采 on-policy 序列**(对采样 prompt 在 /think 或 /no_think 下产出响应),再**对齐学生与教师(Qwen3-32B 或 235B-A22B)logits、最小化 KL** 微调。
- 逐组件必要性:
  - **on-policy 蒸馏 vs RL(同起点)**:有**受控对照**(Table 21,Qwen3-8B,仅数学/代码)。三行:Off-policy 蒸馏(基线)→ +RL(17,920 GPU·hr,pass@64 不变)→ +On-policy 蒸馏(1,800 GPU·hr,pass@1 与 pass@64 双升)。**这是报告里最硬的消融式证据**。【行 3095-3114】
  - **冷启动"少样本少步数"**:理由是保留 RL 提升空间(行 1118-1122),无独立消融。
  - **Thinking Mode Fusion / General RL 的效果**:有 Table 22(Qwen3-32B 跨 Stage2/3/4)分析其对各能力的影响(行 3128)。
  - **整体蒸馏 vs 四阶段**:声称 1/10 GPU 时 + Pass@1/Pass@64 更优(行 1090-1093),但"1/10"的完整对照口径(哪些模型、哪些任务)主要由 Table 21(8B,数学/代码)支撑,**全尺寸的"1/10"为概述性结论**〔推断:Table 21 是 8B 单点,全系"1/10"系外推〕。
- 关键机制/公式(直觉):on-policy 蒸馏=学生先生成、再被教师 logits 拉(reverse-KL 式对齐),比 off-policy(直接模仿教师文本)更贴合学生自身分布、缓解 exposure bias,且教师 logits 提供"软标签/多峰"信息使学生**保留并扩展探索空间**(pass@64 升)——这与综述其它 OPD 工作(GKD、MiniLLM)的核心直觉一致。thinking budget=思考 token 达阈值时插入停思指令,为模型在融合双模后自然涌现的能力(行 1186)。
- 实验与证据:
  - 数据集/设置:训练=数学/代码/逻辑/STEM 可验证 prompt(冷启动精选子集;Reasoning RL 3,995 对;General RL 20+ 任务)。教师=QwQ-32B(冷启动数据生成)、Qwen3-32B / Qwen3-235B-A22B(蒸馏)。评测=AIME'24/'25、LiveCodeBench、CodeForces、GPQA、MMLU、多语(Multi-IF 8 语、INCLUDE 44 语)等,thinking/non-thinking 双模式。
  - 关键数字:**Table 21(8B,核心 OPD 证据)**:Off-policy 蒸馏 AIME'24 55.0(pass@64 90.0)/ AIME'25 42.8(83.3);+RL 67.6(90.0)/ 55.5(83.3),17,920 GPU·hr;+On-policy 蒸馏 74.4(**93.3**)/ 65.5(**86.7**),**1,800 GPU·hr**。Reasoning RL:Qwen3-235B-A22B AIME'24 170 步 70.1→85.1。【已核 Table 21 行 3095-3114、行 1154-1155】
  - baseline 公平性:Table 21 **同起点(同 off-policy 蒸馏 8B checkpoint)、同任务域**对比 RL vs OPD,公平性好(这是报告里最严谨的对照);但仅 8B 单点 + 仅数学/代码。
  - 看着强但没回答的:① 全系"1/10 GPU 时 + Pass@1/64 更优"为概述,受控证据只在 8B/数学代码;② thinking budget、模式切换的可控性多为定性 + 少量基准,未给系统化失效分析;③ 大量后训练细节(数据规模、超参、蒸馏温度/损失权重)以技报体例略写。
- 假设与失效边界:【原文】① 蒸馏依赖**已有强教师**(Qwen3-32B/235B,QwQ-32B)——强教师不可得时方法不适用;② 冷启动刻意"少样本少步数"以保留 RL 空间(行 1121-1122)。【推断】③ Table 21 的"OPD>RL 且省 10×"在数学/代码可验证域成立,**开放式/无强教师域是否同样**未验证;④ on-policy logit-KD 需教师与学生**同 tokenizer/词表**(同系模型),跨族蒸馏不适用;⑤ thinking budget 的"自然涌现"依赖双模融合质量,弱模型上可能不稳。
- 祛魅总结:【推断】真贡献(对本综述)=① **大厂技报里少见的、明确的 on-policy logit-KD 实践**,且用 Table 21 给出"OPD 在 1/10 成本下同时赢 Pass@1 与 Pass@64、而 RL 提不动 pass@64"的受控证据——这是"on-policy 蒸馏 > RL(在有强教师时)"的强论据,直接支撑 OPD 主线;② 统一 thinking/non-thinking + thinking budget 的工程范式。被高估处:作为"方法论文"它**不开源后训练代码、细节略写**,"1/10"全系结论靠 8B 单点外推,复现性受限(技报性质)。被低估处:Table 21 的"pass@64:OPD 升 / RL 不升"这一对比,对"RL 只放大、蒸馏能扩展探索"的论点是少见的厂级实证,价值高于其在综述里常被简化的"1/10 GPU 时"卖点。

## 结构化抽取
- 🎯 机制速览6轴:
  - **学什么信号**:On-policy 蒸馏=教师(Qwen3-32B/235B)的 **logits**(最小化 KL,软标签);Off-policy 蒸馏=教师 response 文本(SFT);Reasoning/General RL=可验证奖励 + 规则/模型奖励(GRPO)。
  - **改什么**:学生/旗舰 policy 参数(SFT/蒸馏/GRPO 梯度);蒸馏阶段对齐 logits。
  - **何时改**:旗舰四阶段串行(SFT→RL→SFT→RL);蒸馏两阶段(off-policy SFT → **on-policy** 自采+logit-KL,在线生成)。
  - **免梯度?**:否,全程梯度更新。
  - **记忆-技能生命周期**:无外部记忆/技能库;能力通过四阶段或蒸馏固化进单一统一模型;thinking budget 是推理期算力分配(非记忆)。
  - **防遗忘机制**:Thinking Mode Fusion 用**拒绝采样自 Stage-2 模型**生成 thinking 数据(防额外 SFT 损伤 Stage-2 性能,行 1167-1168)= 一种针对性防遗忘;On-policy 蒸馏对齐 logits 比硬模仿更温和地保留学生分布。无显式 KL-to-ref 防遗忘的细节披露。
- ⑦ 开源代码+框架/harness:**模型发布/权重仓,后训练训练代码不开源**。仓库 https://github.com/QwenLM/Qwen3;权重 https://huggingface.co/Qwen 、https://modelscope.cn/organization/qwen 。**蒸馏/RL 管线均不开源**(社区微调依赖 Axolotl/Unsloth/ms-swift/LLaMA-Factory)。报告提及算法=**GRPO**(Reasoning/General RL);未提 GSPO(后续单独工作)。**CloneTier=B,仅记录链接、不 clone 巨型权重仓**——未完整克隆原因:无训练代码可 clone,权重仓体积过大且按 B 层规则只记链接。【元信息复用 v1 analysis/qwen3.md;正文来自 PDF】
- 💰 资源/成本与可扩展性:**核心卖点即成本**——Table 21:on-policy 蒸馏 1,800 GPU·hr vs RL 17,920 GPU·hr(~1/10);声称全系蒸馏约四阶段 1/10 GPU 时(概述)。Reasoning RL:Qwen3-235B-A22B 170 步即显著提升。可扩展性:0.6B–235B 全覆盖、dense+MoE,蒸馏让轻量模型低成本获得旗舰能力(报告主张)。
- 🎯 对"探索-巩固"对标:**强支撑 OPD 主线 + 对"探索"提供厂级实证,但无脚手架/前瞻/记忆**。Qwen3 的 on-policy 蒸馏(学生自采 + 教师 logit-KL)正是 mtp_opd 关注的核心 OPD 形式,且 Table 21 直接证明"**强教师 logit 指引能扩展学生 pass@64(探索空间),而 RL 不能**"——这为 TSRD "teacher 当脚手架帮 student 探索/选路"提供了关键的**可行性 + 收益**证据(教师软分布 > 纯结果奖励 在扩展探索上)。可借组件:① **on-policy logit-KD + 1/10 成本 + pass@64↑** 作为 OPD 优于 RL 的标杆基线/对照协议,TSRD 可对标"我的脚手架蒸馏能否也提 pass@k";② **拒绝采样自较新模型生成数据以防遗忘**的工程技巧;③ thinking budget 思路可类比"前瞻/探索预算"控制。缺口:Qwen3 的教师是**异构更大模型**(非同规模 EMA、非 MTP 前瞻),蒸馏是**全序列 logit-KL**(非 path-selection/recovery 的稀疏单点接管),无 on-policy 自选"能走通的开头"、无回轨、无 MTP——它证明了"教师软标签 OPD 有效且能扩展探索"这一**大前提**,但不提供"稀疏脚手架/选路/回轨/前瞻"的具体机制。一句判定:**OPD 主线的厂级标杆与"教师指引能扩展探索"的关键实证(Table 21)**,作大前提支撑与对照协议,而非脚手架机制的直接来源。
- 🔭 开放问题/未来方向:【原文】报告聚焦发布,未单列开放问题;隐含=把统一 thinking 模式 + 蒸馏推向更多尺寸/任务。【推断】① 把"全序列 logit-KL"细化为"关键步/高熵 token 的稀疏蒸馏",看能否在更省的同时进一步提 pass@k(对标 TSRD path-selection);② Table 21 的"OPD>RL"在开放式/无强教师/跨族 tokenizer 域是否成立;③ 把 thinking budget 与"前瞻探索预算(MTP)"结合做自适应推理深度;④ on-policy 蒸馏中教师=异构大模型,能否换成"同规模 + 前瞻特权(如 MTP/QCP)"的脚手架教师(连向 pi_play/TSRD)。

— RETURN —
qwen3 | 读到PDF? 是(35页/116k字,§4 后训练全核 + Table 21/22 数字逐项核对;技报性质后训练代码不开源,正文来自PDF、元信息复用v1) | L线 L1(on-policy logit-KD)+L2(四阶段SFT-RL)+L4(Agent工具RL) | 对标结论:OPD主线的厂级标杆 + "教师logit指引能扩展学生探索(pass@64↑)而RL不能"的关键受控实证(Table 21,1/10 GPU时且Pass@1/64双赢),为TSRD"教师脚手架助探索"提供大前提支撑与对照协议;但教师为异构大模型、全序列logit-KL、无选路/回轨/前瞻/记忆,非脚手架机制直接来源 | 残留待核数:1(全系"1/10 GPU时+Pass更优"为概述,受控证据仅Table 21的8B/数学代码单点;后训练超参/数据规模技报略写、代码不开源不可核)

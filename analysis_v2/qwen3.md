qwen3 | Qwen3 Technical Report | 阿里巴巴 Qwen Team(An Yang 等;大厂旗舰模型技术报告,Qwen2.5→Qwen3 谱系) | 2025-05 预印本·arXiv(arXiv:2505.09388,v1 2025-05-15) | 主题线 L1(on-policy 蒸馏实践)+ L2(统一 SFT-RL 四阶段)+ L4(Agent/工具 RL)·相关性 高(大厂技报里少见的**明确 on-policy logit-KD** + 受控 OPD-vs-RL 对照)

**原始论文**:https://arxiv.org/abs/2505.09388

## 一眼看懂

> 一句话导读:Qwen3 把"会深推理"和"快速响应"两种模式统一进一个模型;对本综述最关键的是它的小模型蒸馏——尤其有一个干净对照:同一起点上,on-policy logit 蒸馏只花 RL 约 1/10 的算力,却同时把即时性能和探索上限(pass@64)都拉高,而 RL 提不动 pass@64。

- 🟦 TL;DR:Qwen3 是覆盖 0.6B–235B 的 dense + MoE 系列。
  - 核心创新=把 thinking(复杂多步推理)与 non-thinking(快速响应)统一进**单一模型**,支持 `/think`、`/no_think` 动态切换,以及 **thinking budget**(对推理用多少 token 做预算控制)。
  - 旗舰模型走**四阶段后训练**:Long-CoT 冷启动 SFT → Reasoning RL(GRPO)→ Thinking Mode Fusion(继续 SFT、把两种模式融合)→ General RL。
  - 轻量模型则用 **Strong-to-Weak Distillation(强到弱蒸馏)**:先用 off-policy response 蒸馏打底,再做 **on-policy 蒸馏**(学生自己采样序列 + 与教师 logits 对齐、最小化 KL)。代价只有四阶段的约 **1/10 GPU 时**,且 Pass@1(即时性能)与 Pass@64(探索能力)双双更优。【原文 §4 行 1084-1102,§4.5 行 1248-1260】
- 最巧的一步(对本综述而言):**on-policy logit 蒸馏这一步**,以及它被一个**受控对照实验(Table 21)**坐实。
  - 设置:在同一个"off-policy 蒸馏后的 8B checkpoint"上,分别接 RL 与 on-policy 蒸馏。
  - RL:花 **17,920 GPU·hr**,把 AIME'24 提到 67.6,但 **pass@64 纹丝不动(90.0→90.0)**;
  - on-policy 蒸馏:只花 **1,800 GPU·hr(约 1/10)**,就把 AIME'24 提到 74.4,且 **pass@64 升到 93.3**(AIME'25 同向:83.3→86.7)。
  - 结论:**蒸馏在更低成本下同时赢了 Pass@1 和 Pass@64,而 RL 提不动 pass@64(探索边界)**。原文明确归因:"distillation from teacher logits enables the student to **expand its exploration space**(从教师 logits 蒸馏能让学生扩展它的探索空间)……而 RL 不带来 pass@64 改进,凸显**更强教师 logit 指引**的优势"(行 3097-3101)。【原文 §(On-Policy Distillation 节)行 3088-3134,Table 21】

## 为什么做

> 一句话导读:过去既要为 chat 模型和推理模型分别训练部署,又要给每个小模型单独跑完整四阶段、成本高;Qwen3 用统一模型解决前者,用强到弱蒸馏解决后者。

- 研究背景:推理模型(o1 / DeepSeek-R1)靠 test-time scaling 大幅提升;社区需要的是一套既能 chat 又能深推理、且尺寸全覆盖的开源系列。【原文 §1】
- 解决的具体痛点:
  - ① 过去需要为 chat 模型(如 GPT-4o)与专用推理模型(如 QwQ-32B)**分别部署/训练**;
  - ② 为每个小模型独立跑完整四阶段后训练,**计算开销大、开发成本高**(行 1089-1092)。【原文 §4 行 1080-1092】
- 相关工作 & 各自不足:报告引 DeepSeek-R1/o1 等推理模型与 GRPO(Shao et al.)作基础。**隐含的对照**有三条:
  - (a) "每个小模型独立跑四阶段"——贵;
  - (b) "直接把 teacher logits 蒸进小模型"——更省,且能细粒度控制推理过程(§4 preliminary 行 1096-1102);
  - (c) "纯 RL 提升小模型"——Table 21 证明它 pass@64 提不动;
  - 报告未单列竞品综述(技报体例)。
- 动机链(为什么这样设计):
  - ① 用统一框架(thinking/non-thinking + thinking budget)免去多模型切换与重复部署;
  - ② 用 thinking budget 自适应分配推理算力;
  - ③ 用旗舰大模型的知识(strong-to-weak 蒸馏)大幅降低轻量模型的后训练成本(1/10 GPU 时),同时兼顾即时性能(Pass@1)与探索能力(Pass@64)。【原文 §4 行 1084-1102】
- 与最近邻工作的Δ:
  - vs "每个小模型独立跑四阶段 RL"——strong-to-weak 蒸馏(尤其 on-policy logit-KD)更省(1/10)且 pass@64 更优;
  - vs 纯 off-policy response 蒸馏——再加一个 on-policy 阶段(学生自采 + logit-KL)能进一步提性能与探索(Table 21 三行递进:off-policy → +RL → +on-policy distill,最后一个最优);
  - 关键有用点:**on-policy 蒸馏带来的 pass@64 提升,是 RL 给不了的**(行 3099-3100)。

## 怎么做 + 靠不靠谱

> 一句话导读:旗舰走四阶段(先用 SFT+RL 练好"思考"、再融合"非思考"+做通用 RL);小模型走两段蒸馏(先 off-policy 模仿教师文本打底,再 on-policy 自采 + 对齐教师 logits)。

- 方法流水线:
  - **旗舰四阶段(Figure 1,§4.1-4.4;前两阶段练"思考",后两阶段融合"非思考")**:
    - ① **Long-CoT Cold Start(§4.1)**——在精选的数学/代码/逻辑/STEM 可验证题上做**两阶段过滤**:
      - **query 过滤**:用 Qwen2.5-72B-Instruct 剔掉不可验证/含多子问题/需 general 生成的题,并剔掉"无需 CoT 就答对"的题(防表面猜测),同时标注 domain 保持平衡;
      - **response 过滤**:用 **QwQ-32B** 对每 query 生成 N 个候选;QwQ-32B 一致失败的题转人工评判;对 Pass@N>0 的题再剔掉六类——(1)错答,(2)大量重复,(3)无充分推理的猜测,(4)思考-总结不一致,(5)语言混杂/风格漂移,(6)疑似与验证集过近;
      - 这一阶段**刻意用少样本、少步数**的 SFT,只植入基础推理模式、保留 RL 空间(行 1127-1131)。
    - ② **Reasoning RL(§4.2)**——筛出 **3,995** 条 query-verifier 对(四准则:未用于冷启动、对冷启动模型可学、尽量难、覆盖广),用 **GRPO**;靠**大 batch + 多 rollout + off-policy** 提样本效率,并**控熵**(让熵稳步上升或保持稳定)以维持稳态;Qwen3-235B-A22B 在 **170 步内把 AIME'24 从 70.1 提到 85.1**(行 1163-1165)。
    - ③ **Thinking Mode Fusion(§4.3)**——在 Reasoning RL 模型上**继续 SFT**、融合 non-thinking 模式:
      - **thinking 数据由 Stage-2 模型对 Stage-1 query 做拒绝采样生成**(目的是防止额外 SFT 破坏 Stage-2 的性能,行 1177-1178);
      - non-thinking 数据覆盖多任务 + 用自动 checklist 评质;
      - 设计 `/think`、`/no_think` chat 模板(non-thinking 也保留一个**空 think 块**以保格式一致,Table 9);
      - thinking budget 是自然涌现的能力。
    - ④ **General RL(§4.4)**——覆盖 **20+ 任务**的奖励系统(指令遵循、格式遵循、偏好对齐、**Agent 工具调用含真实环境多轮反馈**、RAG),用三类奖励:
      - **规则奖励**;
      - **带参考答案的模型奖励**(Qwen2.5-72B-Instruct 打分);
      - **无参考的偏好奖励模型**(由人类偏好训得)。
  - **Strong-to-Weak Distillation(§4.5;5 个 dense:0.6/1.7/4/8/14B + 1 个 MoE:30B-A3B)**:
    - (1) **Off-policy 蒸馏**——把教师在 `/think` 与 `/no_think` 两种模式下的输出做 **response 蒸馏**,打底基础推理 + 模式切换,为下一阶段铺路。
    - (2) **On-policy 蒸馏**——**学生自己采 on-policy 序列**(对采样 prompt,在 /think 或 /no_think 下产出响应),再**对齐学生与教师(Qwen3-32B 或 235B-A22B)的 logits、最小化 KL** 来微调(行 1257-1260)。
      - 〔报告未给损失公式;按标准 on-policy KD 形式,目标是在学生自身分布下最小化序列 token 的反向 KL:\(L=\mathbb{E}_{x,\,y\sim\pi_\theta}\big[\sum_t D_{\text{KL}}\!\big(\pi_\theta(\cdot\mid x,y_{<t})\,\|\,\pi_{\text{teacher}}(\cdot\mid x,y_{<t})\big)\big]\)——【推断:报告只说 "align logits, minimize KL",没写散度方向/公式;此式是标准 GKD / on-policy KD 形式,非原文逐字】〕。
- 逐组件必要性:
  - **on-policy 蒸馏 vs RL(同起点)**:有**受控对照**(Table 21,Qwen3-8B,仅数学/代码)。三行递进:Off-policy 蒸馏(基线)→ +RL(17,920 GPU·hr,pass@64 不变)→ +On-policy 蒸馏(1,800 GPU·hr,pass@1 与 pass@64 双升)。**这是报告里最硬的消融式证据**。【行 3115-3134】
  - **冷启动"少样本少步数"**:理由是保留 RL 的提升空间(行 1127-1131),无独立消融。
  - **Thinking Mode Fusion / General RL 的效果**:有 Table 22 分析其对各能力的影响(Qwen3-32B 跨 Stage2/3/4 × CounterFactQA / LengthCtrl / ThinkFollow / ToolUse 等 in-house 基准,行 3149-3249)。
  - **整体蒸馏 vs 四阶段**:声称 1/10 GPU 时 + Pass@1/Pass@64 更优(行 1099-1102);但"1/10"的完整对照口径(到底哪些模型、哪些任务)主要由 Table 21(8B,数学/代码)支撑,**全尺寸的"1/10"只是概述性结论**〔推断:Table 21 是 8B 单点,全系"1/10"系外推〕。
- 关键机制/公式(直觉):
  - on-policy 蒸馏=学生先生成、再被教师 logits 拉回(reverse-KL 式对齐),比 off-policy(直接模仿教师文本)更贴合学生自身分布、缓解 exposure bias;而且教师 logits 提供了"软标签/多峰"信息,使学生**保留并扩展探索空间**(pass@64 升)——这与综述里其它 OPD 工作(GKD、MiniLLM)的核心直觉一致。
  - **thinking budget** 机制:当思考 token 长度达到用户设的阈值时,手动插入一个停思指令("Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>"),模型就根据已积累的推理产出最终答案。这个能力**不是显式训练出来的,而是由双模融合自然涌现**(行 1196-1206)。
- 实验与证据:
  - 数据集/设置:
    - 训练=数学/代码/逻辑/STEM 可验证 prompt(冷启动用精选子集;Reasoning RL 用 3,995 对;General RL 覆盖 20+ 任务);
    - 教师=**QwQ-32B**(生成冷启动数据)、**Qwen3-32B / Qwen3-235B-A22B**(蒸馏);query 过滤/打分的辅助模型=Qwen2.5-72B-Instruct;
    - 评测=AIME'24/'25、LiveCodeBench v5、MMLU-Redux、GPQA-Diamond、多语等,且分 thinking/non-thinking 双模式。
  - **关键数字(Table 21,8B,核心 OPD 证据;括号内为 pass@64)**:
    - Off-policy 蒸馏(基线):AIME'24 **55.0 (90.0)** / AIME'25 **42.8 (83.3)** / MATH500 92.4 / LCB 42.0 / MMLU-Redux 86.4 / GPQA 55.6(GPU 时 -)。
    - +RL:AIME'24 **67.6 (90.0)** / AIME'25 **55.5 (83.3)** / MATH500 94.8 / LCB 52.9 / GPQA 61.3,**17,920 GPU·hr**。
    - +On-policy 蒸馏:AIME'24 **74.4 (93.3)** / AIME'25 **65.5 (86.7)** / MATH500 97.0 / LCB 60.3 / MMLU-Redux 88.3 / GPQA 63.3,**1,800 GPU·hr**。
    - → 看 pass@64:RL 在 AIME'24/'25 上是 90.0→90.0、83.3→83.3,**完全不动**;OPD 则把它们提到 **93.3 / 86.7**。【已核 Table 21 行 3115-3134】
    - 另:Reasoning RL 上 Qwen3-235B-A22B 在 AIME'24 跑 170 步从 70.1→85.1(行 1163-1165)。
  - baseline 公平性:Table 21 是**同起点(同一个 off-policy 蒸馏后的 8B checkpoint)、同任务域(math + code)**下对比 RL vs OPD,公平性好(这是报告里最严谨的对照);但仅 8B 单点 + 仅数学/代码。
  - 看着强但没回答的:
    - ① 全系"1/10 GPU 时 + Pass@1/64 更优"是概述,受控证据只在 8B / 数学代码;
    - ② thinking budget、模式切换的可控性多为定性 + 少量基准,未给系统化的失效分析;
    - ③ 大量后训练细节(数据规模、超参、蒸馏温度/损失权重/KL 方向)以技报体例略写。
- 假设与失效边界:
  - 【原文】① 蒸馏依赖**已有强教师**(Qwen3-32B/235B、QwQ-32B)——强教师不可得时,方法不适用。
  - 【原文】② 冷启动刻意"少样本少步数",目的是保留 RL 空间(行 1130-1131)。
  - 【原文】③ Thinking Mode Fusion 的"防破坏 Stage-2"靠的是"从 Stage-2 自身做拒绝采样"(行 1177-1178)。
  - 【推断】④ Table 21 的"OPD>RL 且省 10×"在数学/代码这类可验证域成立,**开放式/无强教师域是否同样**未验证。
  - 【推断】⑤ on-policy logit-KD 需要教师与学生**同 tokenizer/词表**(即同系模型),跨族蒸馏不适用。
  - 【推断】⑥ thinking budget 的"自然涌现"依赖双模融合的质量,在弱模型上可能不稳。
- 祛魅总结:【推断】
  - 真贡献(对本综述)=① **大厂技报里少见的、明确的 on-policy logit-KD 实践**,且用 Table 21 给出了"OPD 在 1/10 成本下同时赢 Pass@1 与 Pass@64、而 RL 提不动 pass@64"的受控证据——这是"on-policy 蒸馏 > RL(在有强教师时)"的强论据,直接支撑 OPD 主线;② 统一 thinking/non-thinking + thinking budget 的工程范式。
  - 被高估处:作为一篇"方法论文",它**不开源后训练代码、细节略写**,"1/10"的全系结论靠 8B 单点外推,复现性受限(技报性质)。
  - 被低估处:Table 21 的"pass@64:OPD 升 / RL 不升"这一对比,对"RL 只放大、蒸馏能扩展探索"的论点是少见的厂级实证,价值高于它在综述里常被简化掉的"1/10 GPU 时"卖点。

## 结构化抽取
- 🎯 机制速览6轴:
  - **学什么信号**:On-policy 蒸馏=教师(Qwen3-32B/235B)的 **logits**(最小化 KL,软标签);Off-policy 蒸馏=教师 response 文本(SFT);Reasoning/General RL=可验证奖励 + 规则/模型奖励(GRPO)。
  - **改什么**:学生/旗舰 policy 参数(SFT/蒸馏/GRPO 梯度);蒸馏阶段对齐 logits。
  - **何时改**:旗舰四阶段串行(SFT→RL→SFT→RL);蒸馏两阶段(off-policy SFT → **on-policy** 自采+logit-KL,在线生成)。
  - **免梯度?**:否,全程梯度更新。
  - **记忆-技能生命周期**:无外部记忆/技能库;能力通过四阶段或蒸馏固化进单一统一模型;thinking budget 是推理期算力分配(非记忆)。
  - **防遗忘机制**:Thinking Mode Fusion 用**拒绝采样自 Stage-2 模型**生成 thinking 数据(防额外 SFT 损伤 Stage-2 性能,行 1177-1178)= 一种针对性防遗忘;On-policy 蒸馏对齐 logits 比硬模仿更温和地保留学生分布。无显式 KL-to-ref 防遗忘的细节披露。
- ⑦ 开源代码+框架/harness:**模型发布/权重仓,后训练训练代码不开源**。仓库 https://github.com/QwenLM/Qwen3;权重 https://huggingface.co/Qwen 、https://modelscope.cn/organization/qwen 。**蒸馏/RL 管线均不开源**(社区微调依赖 Axolotl/Unsloth/ms-swift/LLaMA-Factory)。报告提及算法=**GRPO**(Reasoning/General RL);未提 GSPO(后续单独工作)。**CloneTier=B,仅记录链接、不 clone 巨型权重仓**——未完整克隆原因:无训练代码可 clone,权重仓体积过大且按 B 层规则只记链接。【元信息复用 v1 analysis/qwen3.md;正文来自 PDF】
- 💰 资源/成本与可扩展性:**核心卖点即成本**——Table 21:on-policy 蒸馏 **1,800 GPU·hr** vs RL **17,920 GPU·hr**(~1/10);声称全系蒸馏约四阶段 1/10 GPU 时(概述)。Reasoning RL:Qwen3-235B-A22B 170 步即显著提升。可扩展性:0.6B–235B 全覆盖、dense+MoE,蒸馏让轻量模型低成本获得旗舰能力(报告主张)。
- 🎯 对"探索-巩固"对标:**强支撑 OPD 主线 + 对"探索"提供厂级实证,但无脚手架/前瞻/记忆**。
  - Qwen3 的 on-policy 蒸馏(学生自采 + 教师 logit-KL)正是 mtp_opd 关注的核心 OPD 形式,且 Table 21 直接证明"**强教师 logit 指引能扩展学生的 pass@64(探索空间),而 RL 不能**"——这为 TSRD"teacher 当脚手架帮 student 探索/选路"提供了关键的**可行性 + 收益**证据(在扩展探索这件事上,教师软分布 > 纯结果奖励)。
  - 可借组件:
    - ① **on-policy logit-KD + 1/10 成本 + pass@64↑**,可作为"OPD 优于 RL"的标杆基线/对照协议,TSRD 可对标"我的脚手架蒸馏能否也提 pass@k";
    - ② **从较新的模型做拒绝采样来生成数据以防遗忘**的工程技巧,可用于巩固阶段防止 SFT 破坏已有能力;
    - ③ thinking budget 的思路可类比"前瞻/探索预算"的控制。
  - 缺口:
    - Qwen3 的教师是**异构的、更大的模型**(不是同规模 EMA、也不是 MTP 前瞻);
    - 蒸馏是**全序列 logit-KL**(不是 path-selection/recovery 那种稀疏的单点接管);
    - 无 on-policy 自选"能走通的开头"、无回轨、无 MTP;
    - 即它证明了"教师软标签 OPD 有效且能扩展探索"这一**大前提**,但不提供"稀疏脚手架/选路/回轨/前瞻"的具体机制。
  - 一句判定:**它是 OPD 主线的厂级标杆,以及"教师指引能扩展探索"的关键实证(Table 21)**,作大前提支撑与对照协议用,而非脚手架机制的直接来源。
- 🔭 开放问题/未来方向:【原文】报告聚焦发布,未单列开放问题;隐含=把统一 thinking 模式 + 蒸馏推向更多尺寸/任务。【推断】① 把"全序列 logit-KL"细化为"关键步/高熵 token 的稀疏蒸馏",看能否在更省的同时进一步提 pass@k(对标 TSRD path-selection);② Table 21 的"OPD>RL"在开放式/无强教师/跨族 tokenizer 域是否成立;③ 把 thinking budget 与"前瞻探索预算(MTP)"结合做自适应推理深度;④ on-policy 蒸馏中教师=异构大模型,能否换成"同规模 + 前瞻特权(如 MTP/QCP)"的脚手架教师(连向 pi_play/TSRD)。

— RETURN —
qwen3 | 读到PDF? 是(35页/116k字重抽清洗null字节后,§4 后训练全核 + Table 21/22 数字逐项核对一致;技报性质后训练代码不开源,正文来自PDF、元信息复用v1) | L线 L1(on-policy logit-KD)+L2(四阶段SFT-RL)+L4(Agent工具RL) | 对标结论:OPD主线的厂级标杆 + "教师logit指引能扩展学生探索(pass@64↑:90.0→93.3)而RL不能(90.0→90.0)"的关键受控实证(Table 21,1/10 GPU时1800 vs 17920且Pass@1/64双赢),为TSRD"教师脚手架助探索"提供大前提支撑与对照协议;但教师为异构大模型、全序列logit-KL、无选路/回轨/前瞻/记忆,非脚手架机制直接来源 | 残留待核数:1(全系"1/10 GPU时+Pass更优"为概述,受控证据仅Table 21的8B/数学代码单点;on-policy蒸馏KL方向/损失公式技报未写、以标准GKD形式标注为推断;后训练超参/数据规模技报略写不可核)

simplerl_zoo | SimpleRL-Zoo: Investigating and Taming Zero RL for Open Base Models in the Wild | HKUST + TikTok + 美团(Weihao Zeng*、Yuzhen Huang*、Qian Liu*、Junxian He 等) | 2025-03-24 arXiv 2503.18892(v3 2025-08-06, cs.LG)·预印本+Notion 博客 | 主题线 L3(RLVR/zero RL 经验研究)·相关性 高

**原始论文**:https://arxiv.org/abs/2503.18892

## 一眼看懂
- 🟦 TL;DR:DeepSeek-R1 证明从 base 模型直接做规则奖励的纯 RL("zero RL")能自发涌现长 CoT 与自反思("aha moment"),但社区复现几乎都集中在 Qwen2.5(它本身已强,不代表"野生"base)。本文在 **10 个跨家族/尺寸的 base 模型**上做透明 zero RL(仅正确性二值奖励、**不用 format reward**、所有模型同一组超参),用**认知行为级监控**(GPT-4o 标 Backtracking/Verification 等)把"长度增长"与"真实推理涌现"解耦,系统拆解 zero RL 成败的关键因素。**核心是经验研究 + 配方/工具/模型开源,而非新算法**。
- 最巧的一步:**用 Reasoning Behavior Ratio(GPT-4o 标四类认知行为)而非"响应长度"来判断推理是否真涌现**。抽掉它,就退回"看长度涨=有 aha moment"的表层结论——而本文恰恰证明这是错的(Mistral-7B 长度涨但其实是 incoherent 乱码;多数 Qwen2.5 长度涨但认知行为频率没升)。这个监控维度是本文所有反直觉结论的基础。【原文】§2.2、§2.3、§2.4、Fig.4/5

## 为什么做
- 研究背景:R1/o1/Kimi-k1.5 靠长 CoT + 反思推理;R1 证明从 base 直接纯 RL(规则奖励)可自发涌现长 CoT 与 "aha moment"(zero RL)。但最初在 671B DeepSeek-V3 上演示,社区复现(Zeng 2025a、Yeo 2025、Xie 2025、Hu 2025、DAPO 2025)主要在 Qwen2.5。【原文】§1
- 解决的具体痛点:(1) Qwen2.5 base 因预训练含大量合成数据,**本身已具指令跟随 + backtracking/verification**,不代表"in the wild"的多样 base;(2) 现有分析停留在长度/准确率等**表层指标**,判断不了推理行为是否真变,也没澄清涌现机制;(3) 缺乏对"哪些因素决定 zero RL 成败"的系统研究。【原文】§1、§2
- 相关工作 & 各自不足:既有 zero RL 复现只追长度/准确率(Zeng/Hu/Yu 2025);有些用关键词追踪反思行为(Yeo/Xie 2025)——本文论证关键词与高层推理行为只**弱相关**,捕捉不到真实涌现。【原文】§2、§2.2
- 动机链:R1 zero RL 在 671B 上成功 → 社区只在 Qwen2.5 复现且只看表层 → Qwen2.5 不具代表性、表层指标不可靠 → 在 10 个多样 base 上做透明 zero RL + 认知行为监控 → 回答三问(推理如何演化/弱 base 是否也现 aha moment/成败关键因素)。
- 与最近邻工作的Δ:vs 既有 Qwen2.5-only 复现——本文**跨 10 个家族/尺寸**(隔离"模型族"变量)+ **认知行为监控**(而非长度);vs Shao 2024(认为 RL 只是重排)——本文用 pass@k gap 随 k 增大反而**扩大**(13→30 点,k 到 128)证明是**真正增强而非重排**。关键差别:**多样 base + 认知行为指标 + 透明监控**。

## 怎么做 + 靠不靠谱
- 方法流水线(经验研究的"配方"):
  1. **算法**:GRPO,直接从 base 起训(无 SFT cold start)。
  2. **奖励**:仅正确性二值(+1/0),**不用 format reward**(避免强格式约束惩罚探索)。
  3. **数据难度控制**:按难度分 Easy(GSM8K+MATH lv.1)/ Medium(lv.1–4)/ Hard(lv.3–5),各约 8K;难度须匹配模型内在探索能力。
  4. **统一超参**:所有模型同一组超参;弱指令跟随模型(Llama-3.1-8B、Mistral-7B、Qwen2.5-0.5B/1.5B)用更简单 prompt。
  5. **监控指标**:Reasoning Behavior Ratio(GPT-4o 标 Backtracking/Verification/Subgoal Setting/Enumeration)、Clip Ratio(截断比例)、Average Stopped Length(正常停止响应平均长度)、pass@k。【原文】§2.1、§2.2
- 逐组件必要性(本文是经验研究,"组件"=关键设计,均有消融/对比支撑):
  - **去 format reward**(§3.1):刚性 format reward(强制 \boxed{})惩罚探索、压低性能上限、诱发 overthinking,许多 base 初期跟不上格式约束 → 用消融(Fig.6)支撑"去掉它更好"。
  - **难度匹配**(§3.2):难度须与 base 探索能力匹配,否则 zero RL 失败(Mistral-7B 在 Hard 数据上崩溃)→ 实验支撑。
  - **无 SFT cold start**(§4):传统短 CoT SFT 作 cold start 会限制后续 RL 探索,SFT 步数越多对 enumeration/verification 损害越大 → 对比支撑。
  - **认知行为监控**:Reasoning Behavior Ratio 是诊断工具,论文核对了 GPT-4o 与人工标注一致性(附录 E)。三个关键设计都有对应证据。【原文】§3.1、§3.2、§4
- 关键机制/公式(直觉):GRPO 用组内相对奖励当 baseline(无 value 网络)。本文不引新公式,核心是"最简配方 + 多维监控"的实验设计:用统一超参 + 多样模型隔离模型族变量,用认知行为指标把"长度"与"推理质量"解耦。【原文】§2.1
- 实验与证据:
  - **数据集**:训练仅 GSM8K + MATH(规则奖励),三档难度各约 8K。模型(10 个):Mistral-7B-v0.1、Mistral-Small-24B、Llama-3.1-8B、DeepSeek-Math-7B、Qwen2.5-{0.5,1.5,7,14,32}B、Qwen2.5-Math-7B。评测:GSM8K、MATH500、Minerva、OlympiadBench、AIME24(Pass@1 与 Avg@32)、AMC23;泛化 IFEVAL、MMLU、GPQA-Diamond。【原文】§2.1、Table 1/2
  - **关键数字**(Table 1,Avg):Mistral-Small-24B 27.6→**49.6**;DeepSeek-Math-7B 11.3→**29.2**(约 3 倍,长度 ~300→1200+);Qwen2.5-32B 45.9→61.9;Qwen2.5-Math-7B 37.2→59.5。**泛化**(Table 2):仅 8K 数学训练却泛化——Mistral-24B IFEval/MMLU/GPQA Avg 25.0→55.3;Llama-3.1-8B 23.6→32.6。**pass@k**(Fig.3,Mistral-24B):iter0 vs iter100 gap 13→30 点(k 到 128),且 pass@1 与 pass@8 gap 随训练**扩大**(Fig.2)→ 真增强非重排(与 Shao 2024 相反)。**Mistral-7B 反例**:长度涨但 clip ratio 高、是 mixed-language gibberish(Fig.4)→ 长度增长可能"不健康"。【原文】Table 1/2、Fig.2/3/4
  - **baseline 公平吗**:所有模型同一组超参隔离模型族变量(刻意公平);base 用 greedy、SimpleRL-Zoo 用 temp=1.0/top-p=0.95 评测(口径差异已注明 Table 1)。
  - **看着强但没回答核心问题?**:"真增强非重排"由 pass@k gap 随 k 扩大较强支撑;但 "aha moment 涌现"部分结论建立在 **GPT-4o 自动行为标注**之上(分类一致性/偏差未深入校验,虽核对了人工一致性);"难度须匹配探索能力"是定性结论,缺可操作的量化判据。【推断】
- 假设与失效边界:
  - 【原文】仅数学域(GSM8K+MATH)训练;认知行为分类依赖 GPT-4o;难度匹配是定性结论。
  - 【推断】"SFT cold start 损害探索"与具体 SFT 数据/步数强相关(本文用传统短 CoT SFT),外推到**长 CoT SFT**(如 R1-Distill 数据)需谨慎——这点与"长 CoT 蒸馏可有效冷启动"的主流做法张力很大,失效边界在"SFT 数据类型";GPT-4o 行为分类器的偏差是"aha moment"结论的潜在软肋。
- 祛魅总结:真贡献=**第一个跨 10 个多样 base 的透明 zero RL 经验研究 + 认知行为监控工具 + 一批反直觉发现**(长度≠aha moment、Qwen 外小模型首现 verification 涌现、format reward 害探索、难度须匹配、pass@k 真增强、短 CoT SFT cold start 害探索)。包装/局限:(1) **无新算法**(GRPO + 极简配方,贡献是配方/模型/分析工具/系统经验);(2) 仅数学域、行为分类依赖 GPT-4o 有标注偏差风险;(3)"难度匹配"无量化判据;(4) SFT cold-start 结论外推到长 CoT 需谨慎。**作为透明经验研究价值高,作为方法贡献新意有限**。【推断】

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=最终答案正确性二值奖励(GRPO 组内相对 advantage),无 format/过程奖励｜**改什么**=base 模型参数(GRPO 更新,无 SFT)｜**何时改**=在线 zero RL(从 base 直接起训)｜**免梯度?**=否｜**记忆-技能生命周期**=无记忆/技能库;推理技能(认知行为)在 RL 中自发涌现并固化进参数｜**防遗忘机制**=无显式设计;反而发现 SFT cold start 会"占用"探索能力(一种负向的"塑性损失"),且本文证明纯 RL(无 SFT)泛化不降反升(Table 2),即 zero RL 路径本身规避了 SFT 致塑性损失的问题。【原文】§2.1、§4、Table 2
- ⑦ 开源代码+框架/harness:https://github.com/hkust-nlp/simpleRL-reason(已克隆 ~66MB,最新 commit cf1c785,当前版即论文 v1 配方)。框架=**veRL**(仓内自带 `verl/` 目录,pyproject name="verl"),RL 算法 **GRPO**。核心脚本 `train_grpo_math_tune_ray.sh`(Ray 启动)、`eval_math_nodes.sh`、`install.sh`、`launch_gradio.sh`。README 注明旧版 v0 用 OpenRLHF+PPO。模型=HF hkust-nlp/simplerl-zoo collection(10 个)。代码/配方/模型/分析工具齐全可复现。【原文】GitHub + 既有 analysis 对 clone 的核查
- 💰 资源/成本与可扩展性:仅 8K 训练样本即对全部模型显著提升(数据成本极低);GRPO 每 query rollout 8 条;统一超参跨 10 模型(0.5B–32B)。原文未给汇总 GPU·小时(附录 B 有设置)。【原文】§2.1、§2.3
- 🎯 对"探索-巩固"对标:**中支撑,提供"探索质量如何度量与保护"的方法论与多条警示**。本文核心就是"如何让 base 模型在 RL 中**有效探索并固化有效推理行为(认知行为涌现)**"——正对应 idea 的"探索=发现有效行为/路径"+"巩固=固化进参数"。最有价值的对标点:(1) **认知行为监控(Backtracking/Verification)**给 idea 提供了"如何量化 path-recovery/path-selection 是否真涌现"的现成工具——Backtracking/Verification 正是 path-recovery 的行为表征;(2) **"长度≠真推理"**警示——做 MTP/OPD 时不能用响应长度当探索质量代理;(3) **"format reward 害探索""短 CoT SFT cold start 害探索"**支撑 idea"巩固/约束不应损害继续探索的能力(防塑性损失)"——这与 sed_sft/segment_attrib"SFT 别压探索空间"一致,三者互相印证。**可借组件**:GPT-4o 认知行为标注框架 + Clip Ratio/Average Stopped Length 作训练动态监控。**缺口/差异**:是 from-scratch zero RL 非蒸馏(无 teacher 脚手架),无 on-policy 蒸馏的 token 分布信号,无 MTP 前瞻,无显式 path-recovery 机制(靠 RL 自发涌现 backtracking)。一句判定:**"探索质量度量与保护"的中度对标 + 认知行为监控这套可直接借的诊断工具,且"长度≠真推理""约束害探索"对 idea 的探索-巩固设计是重要方法论警示;但无 teacher/MTP,作底座与诊断工具而非直接方法**。【推断,依据 §2.2/§3/§4 vs idea】
- 🔭 开放问题/未来方向:【原文】(隐含)给"难度须匹配探索能力"找可操作的量化判据;更可靠的认知行为度量(减少对 GPT-4o 的依赖);把发现推广到数学外域。【推断】(1) 把认知行为监控接入训练 reward(对 verification/backtracking 行为加权)以主动诱导 path-recovery;(2) 重审"SFT cold start 害探索"在长 CoT 蒸馏下是否成立(对 idea 的"teacher 脚手架冷启动"关系重大);(3) 用 MTP 前瞻或 pass@k 动态作"探索能力-数据难度"匹配的在线判据。

RETURN:simplerl_zoo | 读到PDF? 是(37页:全设置/6发现/Table1-2全数值/§2.4认知行为/§3-4关键因素) | L3 | 对标=中,"探索质量度量与保护"对标+认知行为监控可直接借作诊断工具+"长度≠真推理/约束害探索"方法论警示;无teacher/MTP作底座 | 残留待核 0

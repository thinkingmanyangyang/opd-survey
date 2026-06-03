# ophsd — Training with Harnesses: On-Policy Harness Self-Distillation for Complex Reasoning

> **一句话重点 (TL;DR)**：把"推理时脚手架(harness)"从永久固件重定位为临时训练支架——训练时让模型在 harness 内 rollout、用 harness 诱导的轨迹作 teacher(自蒸馏,reverse-KL),把过程性推理能力永久内化进基座参数,推理时撤掉 harness;数学/文本分类上超过 OPSD/GRPO,且再接回 harness 不再增益甚至降分。

**元信息**：arXiv 2605.08741（v1, 2026-05-09, cs.CL）｜ Peking University（Zhengyang Zhao、Lu Ma 共一,Wentao Zhang 通讯）｜ 2026 Preprint｜ 主题 T1/T2（On-Policy Distillation / Agent-harness）,Relevance=Med（与 TSRD"teacher-scaffolded reasoning"高度同构——把脚手架当临时支架内化能力）｜ 代码 github.com/zzy1127/OPHSD-On-Policy-Harness-Self-Distillation（已 clone,~57M;训练数据 LFS 大文件以 SKIP_SMUDGE clone,指针占位未实下载）｜ 框架 veRL（子类化 PPO trainer,base 为 OPSD repo）

## 关键图示

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/ophsd/fig_01.png)

*Figure 1: Overview of the OPHSD framework. Left: The student policy simultaneously rollouts and interacts with the harness to generate trajectories, while the teacher generates supervisory signals based on harness trajectories and update the student by reverse KL. Top Right: Draft-verify Harness for*

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/ophsd/fig_02.png)


## 1. 相关工作与进展
On-Policy Distillation(OPD):学生自采样轨迹、教师逐 token 监督,分两支——Reward-based(把 reverse-KL 当 policy-gradient 奖励,高方差)与 Loss-based(直接还原可微 token-level 蒸馏损失,稠密低方差但需教师分布)。自蒸馏家族按特权上下文 X 划分(Table 1):OPSD/SDFT 用 verified 参考解、SDPO 用 rollout 中环境反馈、CRISP 用静态"be concise"指令、OEL 用过往轨迹经验知识。另一相关线是 Harness Engineering(NLAHs、Meta-Harness [Lee 2026]、AutoHarness)——优化推理时编排层本身。

## 2. 现有工作存在的问题
推理时 harness(检索增强、plan-solve、draft-verify 等外部脚手架)能显著提升复杂推理,但提升来自外部流程而非模型本身;一旦移除 harness 能力即失。这种"模型与 harness 分离"带来延迟、token 成本、工程复杂度与新失败模式,也难判断模型本身学到了什么。后训练方法不能直接弥合:SFT 模仿静态示范不教自适应流程;RL 监督稀疏难定位关键过程行为;现有自蒸馏方法的特权上下文 X 都是静态变量(参考解/指令串)——只告诉"好答案长什么样",不教"如何推导出来"(信息性 vs 过程性优势)。

## 3. Motivation
OPD 在学生自身轨迹上给稠密 token 监督,是"内化 harness 行为"的天然载体。核心问题:harness 诱导的逐步流程能否被吸收进模型参数?把 harness 从静态变量泛化为"程序化、由学生参数驱动的工作流"。

## 4. 主要灵感 / 核心直觉
受 Learning Using Privileged Information(LUPI)[Vapnik 2015] 启发:把特权输入 z(x) 泛化为任何只在训练可得的 oracle 信息;再用一个确定性、有状态的 harness 程序 H 主动编排 z(x) 与 x 的处理,而非被动拼接到 prompt。学生须从裸输入 x 复现 harness 诱导行为 → 自然划出"可蒸馏边界":结构性推理先验(分解/自验证)可内化,真正的实时外部访问(工具/检索内容)不可。

## 5. 主要解决思路(一段话讲清核心)
OPHSD:训练时学生在 harness 内 rollout(增强推理流程生成轨迹),把这些 harness 辅助轨迹的终端上下文 C[H_θ(x,z(x))] 喂给同一个 frozen base 模型 p_θ̃ 作 teacher,用 **reverse-KL**(KL(p_T‖p_S),Eq.3)沿学生直接 rollout ŷ~p_θ(·|x) 训练不带 harness 的学生;C[·] 作 stop-gradient target。harness 编排由 θ 驱动(随能力演进),logit 监督锚定在 θ̃(稳定先验)。

## 6. 方法详解(通俗、分步骤)

- **harness 形式化**(Eq.2):确定性有状态程序,最多 T 次模型调用,经状态转移 τ 与读出 π 产出最终答案,诱导条件分布 H_θ(y|x);注入 z(x) 得 H_θ(y|x,z(x))。
- **两类实例化**(均自 Qwen3-8B):
  ① **Draft-Verify**(在线文本分类,推理先验=case-based comparison):z(x)=在线 MemoryBank M_<x(x 之前流入的全部标注先例)。draft 步检索 top-kd=5 近邻作 in-context demo 生成草稿 ŷd;verify 步用 ŷd 再检索 k+=5 confirmers(同标签)、k−=5 challengers(异标签),组装含 x/ŷd/两检索集的 prompt 出最终答案。终端上下文 C=(x,ŷd,N+,N−)。Embedder=BAAI/bge-small-zh-v1.5;冷启动保护(bank<10 条时退化为单次前向);harness baseline 评测时 bank 仅从测试流重填(防泄漏)。
  ② **Plan-Solve**(数学,推理先验=结构分解):z(x)=参考解 y*。planner 用 (x,y*) 蒸出策略草图 s~p_θ(·|x,y*),solver 用 (x,s) 执行完整推导 y~p_θ(·|x,s);终端上下文 C=(x,s)——学生匹配的是"已见 plan 的 solver"信号,而非 OPSD 那样直接给 y* 原文(经 planner 中介,强迫内化推导结构)。plan 温度 0.3、solve 温度 0.6;harness baseline 评测时移除 y*。

- **超参**(均 Qwen3-8B / veRL):lr 1e-6、batch 64、max gen 8192、8×H100;GRPO group 8、KL 系数 0;OPSD/CRISP 用 reverse-KL,CRISP 每 50 步同步教师。文本分类训 300 步(每 15 步评)、数学训 150 步(每 10 步评,4 次运行平均)。

## 7. 实验数据集
全部 Qwen3-8B。文本分类两路独立训练(各采 10k,严格防污染):①法条罪名预测 CAIL-2018 训练→LawBench 评测(215 类,报 F1);②化学反应预测 USPTO-50k 训练→USPTO test 评测(10 类,报 acc)。数学:DeepMath 采 10k→AIME24/AIME25/OlympiadBench(取 10% 数据)/HMMT25 评测,报 pass@8(4 次平均)。基线:GRPO、OPSD、CRISP(仅数学)。

## 8. 实验结果与主要发现

- **文本分类**(Table 2):base→harness→GRPO→OPSD→OPHSD。LawBench F1:55.29→60.22→62.44→64.25→**69.51**;USPTO acc:30.07→79.02→90.01→88.01→**90.81**。OPHSD 双双最高,超 GRPO 7.07/0.80、超 OPSD 5.26/2.80。内化后 OPHSD+Harness 反降(LawBench −1.10、USPTO −7.19)。
- **数学**(Table 5,pass@8):OPHSD avg **69.50**(AIME24 79.17、AIME25 61.67、OlympiadBench 83.82、HMMT25 53.33),超 OPSD 2.82、GRPO 2.93、CRISP 10.75;HMMT25 较 OPSD **+10.83**、较 GRPO **+8.33**。base+harness 较纯 base 平均 +17.83(Table 3)。OPSD 早期与 OPHSD 相当但随后因生成长度坍缩(40 步后)而退化(Fig.5)。
- **内化分析**:文本侧 cite-rate(GPT-4o 判 CoT 是否自发引用先例)OPHSD 训练首阶段即 ≥75%、末期 ≥90%,GRPO/OPSD 始终 <10%(Table 4)——学到的是"案例比较"推理形状而非记忆 bank 内容。数学侧按"base vs harness 能力差"分组(Fig.4):harness 才能解的子集相对提升 +84.62%,原本两边都不解的额外解出 +12.54%,且不损原有能力。优势集中在最难分层(Fig.6:Math hard +22.9、LawBench hard +33.8、USPTO hard +90.4)。

## 9. 结果如何支撑其主张
"能内化、且 harness 可撤"由 OPHSD(无 harness)≥ OPHSD+Harness、且 cite-rate/分组分析支撑;case study(LawBench idx44 单向检索覆盖学生双向权衡 → harness 干扰;OlympiadBench idx649 plan 显式标注 ordered-pair 陷阱 → OPHSD 自纠、OPSD 出错)给出机制叙事。"过程性 vs 信息性优势"由"plan 经 planner 中介 vs OPSD 直给 y*"的对照支撑。

## 10. 逻辑自洽性(中性评估)
框架自洽且把 OPSD/SDFT 纳为"平凡静态 wrapper"特例,Appendix C 的三档可蒸馏性分类(Fully/Partially/Non-Distillable + 推理时 z(x) 消融启发式)给出清晰边界。但"OPHSD>OPSD"的部分增益可能源于 OPSD 在该设置下生成长度坍缩(超参/早停敏感),而非纯方法优越;数学最佳分由"取全程最高评估分(4 次平均)"得到,选择性报告需留意。

## 11. 残留问题 / 局限
作者自陈:仅验证两类代表性 harness,未做大规模 harness 设计扫描;Non-Distillable 类(实时工具/检索内容)无法内化,仅"调用脚手架的程序结构"或可蒸馏但本文未验。外部:全 Qwen3-8B 单一规模;数学评测波动大(故 4 次平均);超参(lr/步数)论文正文未尽列,需查 scripts。原"待核"(Table OCR 错位)已用 PDF 重抽数值订正,本次无新事实性出入。

## 12. 开源代码与框架(链接+框架+代码可得性)
github.com/zzy1127/OPHSD-On-Policy-Harness-Self-Distillation(已 clone,~57M;`data/deepmath10k/data_train_10k.json` 由 Git LFS 跟踪,本次 GIT_LFS_SKIP_SMUDGE 未实下载)。框架 **veRL**(README:"OPHSD subclasses verl PPO trainer",需 `pip install -e <verl>`;底层 RL trainer base 为 HJSang/OPSD_OnPolicyDistillation)。依赖 torch≥2.4、hydra-core、ray、vllm≥0.6、openai 客户端(harness 经此与 vLLM 通信)。代码:`ophsd_train/src/ophsd/`(trainer+worker,Hydra)、`src/rewards/`、`harnesses/`(每任务自包含子包,共享 `_api.py`、`_memory_bank.py`);三 launcher `train_ophsd_{math,lawbench,uspto}.sh`;LawBench/USPTO 需 `precompute_embeddings` 预算训练嵌入。

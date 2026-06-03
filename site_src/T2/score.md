# score — From Correction to Mastery: Reinforced Distillation of LLM Agents (SCoRe)

> **一句话重点 (TL;DR)**：让小学生 agent **主导**轨迹生成、teacher **只纠正最早一步错误**,学生从"已验证前缀"续写并做短 horizon RL,把行为克隆的累积误差从 O(H²) 降到 O(H);RL 阶段从最早错误前的前缀起 rollout 并用 key-step 稠密奖励缓解稀疏奖励。

**元信息**：arXiv 2509.14257 (v2, 2025-10-09) ｜ 中科大 USTC(Yuanjie Lyu、Tong Xu 通讯) + Independent Researcher(Chengyu Wang 通讯、Jun Huang;原阿里相关) ｜ arXiv 预印本 2025-09(v2 2025-10) ｜ 主题 T2(agent/tool 蒸馏),与 mtp_opd 核心 path-selection + path-recovery / 教师纠正最早错误高度契合 ｜ 代码 github.com/modelscope/easydistill(SCoRe 在 `projects/SCoRe/`,整库 ~221MB) ｜ 框架 自定义工具链(LLaMA-Factory + LangGraph + veRL,非单一框架)

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/score/fig_01.png)

*Figure 1: Comparison between imitation-based distillation and our SCoRe framework. (a) Prior methods clone entire teacher trajectories. (b) Our approach lets the student explore, with the teacher correcting only the earliest error. Correction-based SFT mitigates the compounding errors of pure imitat*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/score/fig_02.png)

*Figure 2: The SCoRe framework. (a) A student agent attempts a task, and the teacher provides a single-step correction at the first error, creating student-centric training data. (b) The student is initially trained to imitate full solution trajectories via supervised fine-tuning. (c) The student is*

## 1. 相关工作与进展
LLM agent 经 ReAct 式"推理-动作-观测"迭代 + 外部工具(代码解释器、搜索)解复杂任务,但依赖超大昂贵 backbone(GPT-4/Qwen2.5-72B),延迟成本高。Agent Distillation(Kang et al. 2025)把 teacher 行为拆成 [Thought, Action, Observation] 轨迹让小学生模仿。本文构建于 **CodeAct**(action=可执行代码)。

## 2. 现有工作存在的问题
"teacher-acts, student-clones"全轨迹模仿两大 gap:(1) **Reasoning Ability Gap**——小模型难复现 teacher 逻辑分解;(2) **Knowledge Capability Gap**——照搬计划也可能因知识不足无法执行复杂动作。二者源于 emergent abilities,不能完全迁移。更关键:BC 中任一步失败把学生推入 OOD 状态,误差按 **O(H²)** 随 horizon 复合增长(Ross et al. 2011 DAgger 分析)。

## 3. Motivation
让学生**主导**、teacher **最小干预**(只纠最早错误),从而:(a) **Capability Matching**——轨迹复杂度匹配学生当前能力,数据可学;(b) **Deficiency Localization**——"已验证前缀 + 关键步"结构精确暴露弱点。把 teacher-student 分布偏移限制在单步,打破长误差链,累积误差 O(H²)→**O(H)**(论文附录给出 per-step 不等式推导)。

## 4. 主要灵感 / 核心直觉
纠错信号应施加在学生"最早出错"的那一步:此前前缀由学生自己生成(on-distribution),此后从被纠正的步起续写,既保证正确性又把监督定位到学生真实能力边界。

## 5. 主要解决思路(一段话讲清核心)
SCoRe(Student-Centered one-step Reinforcement)三阶段:先用少量 teacher 轨迹冷启动 BC;再让学生独立解题、teacher 只纠最早错误、学生从纠正前缀续写(必要时重复),保留"最小干预"纠正轨迹;最后做短 horizon RL——**从已验证前缀(最早错误前)起 rollout**(缩短 horizon、降梯度方差)并用 **key-step 稠密奖励**。

## 6. 方法详解(通俗、分步骤)
1. **Cold-Start BC**:在少量高质量 teacher 轨迹上 SFT,bootstrap 基础推理-动作技能。
2. **Mentored Problem-Solving (MPS) + SCoRe-SFT**:学生独立做新任务 → teacher 检查并纠正**最早错误** → 学生从纠正后前缀续写;若再错则重复。最终任务成功隐式验证修正正确性。纠正轨迹用于 SFT。
3. **SCoRe-RL** 两创新:(a) 从已验证前缀起 rollout 而非任务开头;(b) **key-step 稠密奖励**:最终答案正确 reward=**1**;关键步等于 teacher 修正 reward=**0.5**;关键步等于初始错误 reward=**0**;都不等价 reward=**0.1**(GRPO 式按组算 advantage)。RL 奖励正误由**轻量验证器 Qwen2.5-7B-Instruct**判语义一致性(注:与评测时用的 72B judge 不同,见 §7)。

## 7. 实验数据集
12 个 benchmark 三类(论文 §4.1 / Table 1–2):
- **数学(4)**:AIME2024、AIME2025、MATH500、OlympiadMath;
- **事实/多跳 QA(4)**:HotpotQA、2WikiMultihopQA、Musique、Bamboogle(开放域 QA 用 token-level F1);
- **agentic 深度搜索(4)**:GAIA、WebWalker、HLE、xBench(WebThinker text-only split)。
- **评测 judge**:math + deep-search 正误由 **Qwen2.5-72B-Instruct** 作 LLM-as-judge(Zheng et al. 2023 范式)判定;QA 用 F1。
- **student**:Table 1(math+factual)= Qwen2.5-7B / Qwen2.5-3B / Llama3.1-8B;Table 2(deep-search)= **Qwen3-8B-Instruct**。评测集组织参考 ARPO,GRPO/ARPO 数值多取自 ARPO 原文。

## 8. 实验结果与主要发现
- **math+factual(Table 1,8 项 Avg)**:Qwen2.5-7B SCoRe-RL = **50.8**(BC=42.5、GRPO=48.4、ARPO=49.3),仅比 72B teacher(51.7)低 **0.9**;较 BC **+8.3**(50.8−42.5,已核)。Qwen2.5-3B SCoRe-RL=46.7(较 BC +8.4);Llama3.1-8B=47.5(较 BC +10.2)。
- **deep-search(Table 2,Qwen3-8B)**:SCoRe-RL Avg = **30.5**,+7.7 over BC、+8.3 over GRPO、超 TIR-Qwen2.5-72B +3.2,部分子项超 teacher;GAIA-Avg 从 27.2 升。
- **消融(Table 3)**:去短 horizon rollout / 去 key-step reward 均掉点,二者齐备最优。Figure 3:hard data 上 SCoRe-RL>SCoRe-SFT。
- 〔待核〕RL 超参(组大小、lr、KL 系数)未在正文给出,见附录 B。

## 9. 结果如何支撑其主张
"逼近 teacher"由 7B 仅低 0.9、部分 deep-search 子项超 teacher 支撑;"O(H²)→O(H) 有益"由短 horizon rollout 消融掉点支撑;"key-step 暴露弱点有益"由 key-step reward 消融掉点 + Figure 3 hard-data 增益支撑;跨 3 个 student 一致较 BC 大幅提升支撑普适性。链条较完整。

## 10. 逻辑自洽性(中性评估)
整体自洽,但有一处**论文内部数值不一致(本轮发现)**:正文 §5 prose 写 7B "+6.3 over GRPO",但 Table 1 GRPO=48.4、SCoRe-RL=50.8,实差仅 **+2.4**,prose 的 +6.3 与自身表格矛盾(上一轮分析直接抄了 prose 的 +6.3,本轮按表格更正为 +2.4)。其余如 +8.3 over BC、deep-search +8.3 over GRPO 与表格自洽。另外:"最终任务成功隐式验证 teacher 修正正确性"是弱验证——任务成功不等于每步修正都正确,可能引入噪声标签;论文未量化误纠率。

## 11. 残留问题 / 局限
- teacher 需 Qwen2.5-72B 级模型生成纠正,蒸馏成本不低;真"小成本"仅指部署期 student。
- RL 奖励验证器(7B)与评测 judge(72B)不同,存在 reward hacking / 训练-评测口径不一致的潜在风险,论文未交叉验证。
- "最早错误"的定位依赖 teacher 判断,错判会污染 SFT/RL 数据;无误纠率量化。
- 论文 prose 与 Table 的 GRPO 增益不一致(见 §10),建议以 Table 为准。
- 〔待核〕RL 完整超参在附录 B。

## 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库:https://github.com/modelscope/easydistill(SCoRe 在 `projects/SCoRe/{MPS, SFT, RL, inference}`;整库 clone ~221MB)。
- 框架(组合工具链,非单一):
  - Cold-start BC / SCoRe-SFT 用 **LLaMA-Factory**(`llamafactory-cli train/api`,README 明列 `pip install llamafactory langgraph`)+ **LangGraph**(agent 轨迹生成,`MPS/graph/{graph.py, graph_repair.py}`)。
  - SCoRe-RL 用 **veRL**(含 `verl/tools/search_tool.py` 工具调用)。
  - 推理 LLaMA-Factory api(`infer_backend: vllm|sglang|huggingface`)。
  - teacher 经 `QWEN25_72B_API_ADDRESS`/`_KEY` 走官方 API 或 `llamafactory-cli api --model_name_or_path Qwen/Qwen2.5-72B` 本地部署。
- 数据流水:`MPS/BC_init_data_gen.py` → `MPS/format/BC_init_llamafatory_format_convert.py` → BC 训练;部署学生 → `MPS/SCoRe_data_gen.py` 生成"学生中心、teacher 仅纠最早错"轨迹 → `SCoRe_SFT_llamafactory_format_convert.py` → SCoRe-SFT;`MPS/format/SCoRe_RL_convert.py` 过滤与 SFT 重叠样本 → verl parquet → `RL/.../run_qwen2.5-7b_agent_distill_tool_agent_mlflow.sh`。
- 数据:种子主要取自 **Tool-Star**(math: NuminaMath、Omni-Math;factual: HotpotQA/2Wiki/WebWalker),共 **35k QA 对**;**20%** 全程 teacher 标注作 BC,**80%** 经 MPS 生成后**对半分**(一半 correction-based SFT、一半 RL);RL 最大 rollout 步数=8。
- 致谢 verl、LLaMA-Factory、ARPO(评测集)、agent-distillation(prompt 灵感)。代码可得性 Tier A。

# 系统性文献调研 — 执行计划与进度状态 (canonical)

> 任务:围绕「On-Policy Distillation(含 +Tool/Agent)」为主轴,外加「GFT 类统一 SFT-RL 微调」与「思维链推理(CoT reasoning)」两条相关线,系统性调研近一年论文。
> 时间窗:**2025-06 ~ 2026-06 为主**(奠基性/高被引的更早工作可作为引用补充)。
> 自主模式:**全程自主执行,中途不提问、不 human-in-loop**。
> 工作目录:`D:\claude_code_workspace\mtp_opd`
> 启动日期:2026-06-02

---

## 1. 调研范围与纳入/排除标准

### 主题(纳入)
- **T1 On-Policy Distillation 核心**:on-policy KD / online distillation / policy distillation / GKD / OPD / OPSD / 自蒸馏 / 师生协同蒸馏。
- **T2 On-Policy + Tool / Agent**:agentic / tool-use / 多轮环境交互场景下的 on-policy 蒸馏或 on-policy 训练。
- **T3 GFT 类统一 SFT-RL 微调**:GFT、PSFT(2508.17784)、UFT、DFT、SRFT、"SFT as RL / 统一框架";拒绝采样自举 SFT(STaR/ReST/RFT/RAFT/ExIt);group-advantage RL(GRPO/DAPO/GSPO/Dr.GRPO/RLOO 等)。
- **T4 思维链推理训练**:reasoning RL / reasoning distillation / 关键 decision token / reflection-纠错-回溯 / prefix / early-token。

### 排除
- 纯 DPO 偏好对齐族(IPO/KTO/SimPO/ORPO 等),除非与 on-policy 蒸馏/推理直接相关。
- 与推理后训练无关的纯下游应用。

### 相关性分级
- **High**:直接做 on-policy 蒸馏 / 统一 SFT-RL / 推理训练,且方法与 TSRD 主题强相关。
- **Med**:相关但偏外围(诊断/分析/旁证)。
- **Low**:仅边缘相关,归档不深挖。

---

## 2. 来源清单
arXiv · ACL Anthology · OpenReview(ICLR/NeurIPS/COLM)· HuggingFace(papers/models)· 大厂(Meta、OpenAI、Anthropic、Google DeepMind、Qwen/阿里、DeepSeek、GLM/智谱、ByteDance Seed、腾讯混元、MiniMax 等)。

---

## 3. 执行流程(分波次)
- **P1 发现(并行子代理撒网)**:主题×来源,各代理写出候选表 → 合并去重到 `candidates_master.md`。目标 ≥100 候选。
- **P2 粗筛**:逐条判断主题相关性,打 High/Med/Low。
- **P3 代码核查**:确认是否有可用开源代码(README/GitHub/HF/paperswithcode)。**只有有代码者进入 P4**。
- **P4 深读分析**:对有代码论文抓全文+仓库,产出 8 字段 + 框架识别,`git clone --depth 1` 到 `resource/repos/`。每篇写 `analysis/<key>.md`。
- **P5 引用扩展**:从已分析论文的相关引用二次搜索,回灌候选表,迭代到饱和。
- **P6 汇总**:合成 `SURVEY_on_policy_distillation.md`。

### 下载落盘规则(硬性约束)
- **所有下载物必须放在 `resource/` 下**:
  - 论文 PDF → `resource/papers/`(沿用既有目录)
  - 开源代码(`git clone --depth 1`)→ `resource/repos/<repo>/`
- 不得把下载内容散落到其他位置。

### 每篇有代码论文汇报字段
1. 开源代码链接
2. 使用框架(TRL/veRL/OpenRLHF/LLaMA-Factory/ms-swift/slime/AReaL/NeMo-Aligner/OpenInstruct/Megatron/DeepSpeed… 或"无框架/自研",必须说明)
3. 研究背景
4. 当前存在的问题
5. Motivation
6. 主要方法
7. 实验数据集
8. 怎么做的(训练/数据/流程细节)

---

## 4. 进度状态(每波次更新)

| 阶段 | 状态 | 备注 |
|---|---|---|
| 脚手架 | ✅ 完成 | 2026-06-02 |
| P1 发现 wave1 | ✅ 完成 | 8 代理 → 合并去重 111 篇 |
| P2 粗筛 | ✅ 完成 | 相关性已分级(随合并) |
| P3 代码核查+验证 | ✅ 完成 | V1-V6,85 篇全核验为真(0 伪造) |
| P3.5 合并固化 | ✅ 完成 | 深读队列 66 + 剔除 19 |
| P4 深读 | ✅ 完成 | 66 篇 8 字段分析,PDF+clone 落盘 |
| P4.5 抽检 | ✅ 完成 | 抽检 10 篇,抓出并修正 psft 1 处幻觉 |
| P5 引用扩展 | ✅ 完成 | 新增 28 候选,9 篇有代码深读(总 75) |
| P6 汇总 | ✅ 完成 | SURVEY_on_policy_distillation.md(3655 行) |

### 最终计数(含 Round-2/3 引文扩展 + 全量二次审查)
- 检视候选:候选表 139 + 引文扩展 r2 78 + r3 175;附录 B 未深读候选池去重 ~550
- 深读(8 字段、有代码)篇数:**117**(代码受限 19:404/未发布/README-only/仅权重)
- PDF:137 · 克隆仓库:100(4.1 GB)· 剔除:20(含 comap 仓库404)
- 防幻觉:P4.5 修正 psft 1 处编造数值;**Phase-B 对全部 117 篇重读全文+核心代码二次审查,约半数做精度修正(数据集/超参/框架)+ 纠正若干杜撰会议/模型/任务,无重大结论颠覆**
- **加强轮**:全部 117 篇重写为 12 节通俗结构(TL;DR + 相关进展/痛点/Motivation/灵感/思路/方法/数据集/结果/证据支撑/逻辑自洽/残留问题/代码),并第三次重读全文+核心代码再核验(更正了前轮个别误判:scope 方向、sod_stepwise 代码位置、sdcl KL 方向)
- 最终文档:SURVEY_on_policy_distillation.md(**8482 行 / 1.1 MB**;含 117 行一句话速览表 + 附录 B 546 未深读候选)
- 最终文档:SURVEY_on_policy_distillation.md(4547 行)

> 注:最终流程与保证机制以根目录 `CLAUDE.md` 为准(已闭环,含 P3.5/P4.5 与防幻觉内容层)。

### 候选计数(wave1 后)
- 候选总数:111
- High 相关:70
- 有代码标记(Y):41;疑似(?):12
- 已深读:0

### ⚠ 已知风险
- 大量 2601–2606 arXiv id 由发现代理给出但未独立核实(尤其 A8 引用扩展产出、org="—" 者),疑有幻觉。
- **铁律:任何论文进入最终报告前,必须独立核实 arXiv/OpenReview/ACL 真实存在且标题相符 + 代码仓库真实可达(非 404)。验证不过的一律剔除到 `_state/verify/rejected`。**

---

## 5. 断点续跑指引
- 真值文件:本 `PLAN.md`(进度)+ `_state/candidates_master.md`(候选)+ `analysis/*.md`(已深读)。
- 恢复时:读本文件第 4 节进度 → 读 `candidates_master.md` 找未处理项 → 继续对应阶段。

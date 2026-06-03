# vla_opd — VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models via On-Policy Distillation

> **一句话重点 (TL;DR)**：把 VLA 后训练改写成"在 student 自采样轨迹上、用冻结 teacher 的 dense token-level 监督做 reverse-KL 蒸馏"的 on-policy RL，从而同时拿到 SFT 的快收敛、RL 的少演示/抗遗忘，并用 reverse-KL 的 bounded mode-seeking 避免 Forward-KL 熵爆炸与 Hard-CE 熵坍缩。

**元信息**：arXiv 2603.26666v1（cs.RO，2026-03-27）｜ HKUST(GZ)（Zhide Zhong, Haodong Yan, Junfeng Li, Junjie He, Tianran Zhang, Haoang Li）｜ 预印本 ｜ 主题 机器人 VLA 后训练 / on-policy distillation，与本项目（LLM reasoning OPD）**关系外围**——共享"on-policy 蒸馏 + reverse-KL mode-seeking"方法骨架，但落在机器人动作 token 而非语言推理 ｜ 代码 项目页 https://irpn-lab.github.io/VLA-OPD/ 标注 "Code (Coming Soon)"（**截至核查无可用仓库，纯论文可读**）｜ 框架 〔待核：代码未放出，下述基于论文〕基于 GRPO 式分组采样、teacher=SimpleVLA-RL、student=OpenVLA-OFT

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/vla_opd/fig_03.png)

*Figure 3: Seen-Unseen Trade-off for Forgetting Analysis. Each point corresponds to a checkpoint during fine-tuning. The x-axis is the success rate on seen (target) tasks, and the y-axis is the success rate on a held-out unseen task. Offline SFT exhibits a strong collapse on unseen tasks as seen-task*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/vla_opd/fig_01.png)

*Figure 1: Overview of VLA-OPD. Our framework unifies offline SFT and online RL through three phases. Phase 1 (Student Sampling): The student VLA policy interacts with the environment to collect on-policy trajectory rollouts ( O → A → O ). Phase 2 (Teacher Labeling): For each state visited by the stu*

## 1. 相关工作与进展
VLA 后训练当前两大范式：(1) 离线 SFT / 行为克隆（O'Neill 2024, Black 2024 π0），dense 监督、收敛快；(2) 在线 RL（SimpleVLA-RL、VLA-RL、RLinf-VLA 等），用 GRPO 做 group-relative 优势、免 critic，把策略对齐到自身诱导的状态分布以学恢复行为。交互式模仿学习（DAgger/HG-DAgger）在 student 诱导的 OOD 状态上收集专家标注。LLM 蒸馏侧的 MiniLLM(Gu 2023)、GKD(Tan 2023) 提出 reverse-KL / on-policy 蒸馏，本文将其迁移到动作预测。

## 2. 现有工作存在的问题
- 离线 SFT 是 off-policy：在专家状态训练却在 student 诱导状态评测，复合误差（exposure bias）使其无法从自致偏离状态恢复；且对 static、disjoint 数据集做激进参数更新 → 灾难性遗忘。
- 稀疏奖励在线 RL（GRPO）：机器人任务通常只有终态二值信号 R(τ)∈{0,1}，信用分配困难、方差高、样本效率极低。
- 简单把 SFT 改 on-policy（如 DAgger）用次优对齐目标：Forward-KL（soft 标签）mode-covering，在 teacher 高熵的 OOD 状态会模仿其犹豫 → 熵爆炸；Hard-CE（argmax 标签）丢弃 dark knowledge，在多模决策边界刚性追 argmax → 过早熵坍缩、丧失探索多样性。

## 3. Motivation
在 student 自生成轨迹上用冻结 teacher 提供 dense token-level 监督，把延迟稀疏奖励转成即时监督（加速收敛 + 注入"恢复先验"）；用 reverse-KL 的 bounded mode-seeking 性质，只要 student 动作落在 teacher 可接受概率质量内即不被惩罚，从而过滤 teacher 尾部不确定性、保留模内随机性；on-policy 更新把梯度锚定在 student 当前行为流形上，实现"温和对齐"缓解遗忘。论文同时声称：依赖高性能 teacher 的可得性已不再是瓶颈（开源 checkpoint / API / 易训单任务策略）。

## 4. 主要灵感 / 核心直觉
核心直觉是"散度方向决定 OOD 状态下的熵动力学"。teacher 在 OOD 状态往往呈现平坦高熵分布（epistemic uncertainty）；Forward-KL 在 teacher 样本上估梯度强制覆盖全支撑 → 继承高熵；reverse-KL 的 zero-forcing 让 student 只需对齐 teacher 的主模、忽略长尾，从而"果断但仍可探索"。把 reverse-KL 写成 token-level 内在奖励后即可挂进 group-based policy gradient。

## 5. 主要解决思路(一段话讲清核心)
三阶段闭环（Algorithm 1）：student πθ 在环境 on-policy rollout G 条轨迹（主动暴露 OOD 失败状态）→ 对 student 访问过的每个状态查询冻结 teacher πtea 的 action logits（不在环境执行）→ 用 token-level 负 reverse-KL 作内在奖励 r_t = −(log πθ(a_t|s_t) − log πtea(a_t|s_t))（对 student log-prob 项 stop_gradient），等价于在 student-visited states 上最小化对 teacher 的 reverse-KL；用 group 平均的 policy gradient 降方差。

## 6. 方法详解(通俗、分步骤)
1. **初始化**：student 由极少演示 SFT 得到（LIBERO 1-traj、RoboTwin2.0 1000-traj），是脆弱下界；teacher 为 RL 训得的鲁棒专家（SimpleVLA-RL），全程冻结。
2. **Phase 1 学生采样（探索）**：πθ 在环境中跑 G 条轨迹，频繁进入 OOD 失败态 serr，把"未知"区显式纳入训练分布。
3. **Phase 2 教师标注（纠正）**：对每个被访问状态 st，查询 teacher 得 qt(a)=πtea(a|st) 作 dense 引导，注入恢复先验；teacher 只打标不执行。
4. **Phase 3 模式寻优（更新）**：token-level 奖励 r^OPD_t = −log(πθ/πtea)（式 6，stop_gradient 作用于 student log-prob），等价最小化 reverse-KL（式 5）。梯度按 group 平均（式 7）；**与标准 GRPO 不同，不做 outcome reward 归一化，直接用 raw reverse-KL reward 作优势**。
5. **理论对比**：Forward-KL=mode-covering→熵爆炸；Hard-CE=丢 dark knowledge→熵坍缩；reverse-KL=zero-forcing bounded mode-seeking→健康熵。
6. **两个变体**：Ours(Distill) 仅蒸馏；Ours(Distill+GRPO) 蒸馏热启后再 GRPO 微调。主实验 batch=64、G=8（沿用 SimpleVLA-RL）；消融固定 batch=32。

## 7. 实验数据集
- **LIBERO**（单臂，四套件 Spatial/Object/Goal/Long）：极端数据稀缺，每任务 1-traj SFT 初始化。
- **RoboTwin2.0**（双臂协作，四代表任务 Pick dual bottles / Place Empty Cup / Handover Block / Stack Bowls Two，覆盖 short→long horizon）：每任务 1000-traj SFT 初始化。
- 遗忘评测：在 seen 任务微调，在 4 个 held-out unseen 任务（2 Object、2 Spatial）评估 seen–unseen 权衡。
- 全部为**仿真基准，无真机**。

## 8. 实验结果与主要发现
- **效率（Fig.2）**：LIBERO-Object 蒸馏 10 步内 >90%（"垂直起飞"）；LIBERO-Long 仅 ~50 步达基线 ~150 步水平（≈3× 加速），且曲线平滑（基线 GRPO 锯齿震荡）。
- **效能（Table 2，LIBERO 成功率%）**：student init 平均 48.9 → Distill 87.4（媲美/超过若干 50-traj 全量基线如 Octo 75.1、OpenVLA 76.5）→ Distill+GRPO 93.4（逼近 teacher SimpleVLA-RL 93.9）。
- **双臂（Table 3，RoboTwin2.0）**：student init 45.2 → Distill 71.1（近 teacher 74.0），超 π0(50.5)、RDT(32.0)；未报 Distill+GRPO。
- **抗遗忘（Fig.3）**：离线 SFT 在 unseen 上严重坍缩（Object 近零、Spatial 大跌）；on-policy 方法（RL 与本法）基本避免，VLA-OPD 多数轴上匹配或超过 RL。
- **消融（Fig.4，RoboTwin2.0 Beat Block Hammer）**：reverse-KL 稳步上升；Forward-KL 早期 >50% 性能谷且熵爆炸；Hard-CE 熵坍缩、平台最低。
- **group size（Fig.5，LIBERO-Object，batch=32）**：G=8 最高(~89%)，G∈{2,4} 仍 >80% 不坍缩，小 G 显著省 rollout/teacher 推理开销。

## 9. 结果如何支撑其主张
- "效率优于稀疏 RL"：Fig.2 收敛步数对比直接支撑（10/50 步 vs 150 步）。
- "效能逼近 teacher"：Table 2/3 终值（93.4 vs 93.9；71.1 vs 74.0）支撑。
- "reverse-KL 优于另两目标且对应熵行为"：Fig.4 性能曲线 + actor 熵曲线一一对应，是对核心主张最直接的证据。
- "抗遗忘"：Fig.3 seen–unseen 散点支撑离线 SFT 坍缩 / on-policy 保留的定性对比。
- 整体证据链自洽，但多为单基准/单任务曲线（如消融仅 1 个 RoboTwin 任务），统计显著性与多 seed 方差未报。

## 10. 逻辑自洽性(中性评估)
- 内在逻辑顺：把延迟稀疏奖励换成 dense token 监督、reverse-KL 解释熵两极，链条清晰。
- 但有两处张力：(1) "raw reverse-KL 作优势、不做归一化仍稳定"只有经验曲线支撑，无理论收敛保证，且 stop_gradient 后该梯度估计的方差/偏置性质未分析；(2) 与稀疏 RL 的"效率"对比未计入 teacher 自身训练成本——teacher 即由 RL(SimpleVLA-RL) 得到，"省掉 RL 探索成本"实质是把成本前移到 teacher 训练，公平性存疑。
- "近 teacher"也意味着方法本质受 teacher 性能上界约束（作者承认）。

## 11. 残留问题 / 局限
- 依赖高性能 teacher 的先验可得性（作者列为 future work 的主攻方向）。
- 评测仅 LIBERO/RoboTwin2.0 两仿真基准、**无真机**；泛化到真实物理/感知噪声未知。
- 不归一化的优势稳定性靠经验；无多 seed 方差、无显著性检验。
- 与稀疏 RL 的成本对比未控制 teacher 训练成本。
- 代码未放出，复现性受限。

## 12. 开源代码与框架(链接+框架+代码可得性)
- 项目页：https://irpn-lab.github.io/VLA-OPD/ ，标注 "Code (Coming Soon)"；论文未给 GitHub 链接。
- **代码可得性：截至核查日无可用仓库（RepoExists=NO），仅论文可读**。〔待核：后续是否放出〕
- 论文层面框架：student=OpenVLA-OFT，teacher=SimpleVLA-RL；分组采样沿用 SimpleVLA-RL 设置（batch=64, G=8）；优化为 group-based policy gradient（类 GRPO 但不做 outcome reward 归一化）。

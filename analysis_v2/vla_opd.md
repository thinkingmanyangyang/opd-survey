vla_opd | VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models via On-Policy Distillation | HKUST(GZ)（Zhide Zhong, Haodong Yan, Junfeng Li, Junjie He, Tianran Zhang, Haoang Li） | 2026-03-27·arXiv preprint·v1（cs.RO） | 主题线 L1（OPD/自蒸馏，机器人 VLA 域）·相关性 中

**原始论文**：https://arxiv.org/abs/2603.26666 （项目页 https://irpn-lab.github.io/VLA-OPD/ ，代码 Coming Soon）

## 一眼看懂

> 一句话导读：机器人模型微调里,离线模仿(SFT)收敛快但一偏就崩、在线 RL 能纠错却只有"成/败"二值奖励太慢;本文让学生在自己走出的轨迹上、用一个冻结的强老师逐 token 打分做 reverse-KL 蒸馏,把稀疏奖励换成稠密监督,于是两边的好处都拿到。

- 🟦 TL;DR：机器人 VLA 后训练有个老大难——
  - 离线 SFT（行为克隆）收敛快,但只在专家走过的状态上训,一旦执行偏离就崩,还会灾难性遗忘;
  - 在线 RL（GRPO）能在自己走出的状态上学纠错,却只有终态二值奖励,样本效率极低。
  - 本文把后训练改写成一种 on-policy RL:student 在自己采样的轨迹上,用一个冻结的强 teacher 逐 token 打 logits 做 reverse-KL 蒸馏,把稀疏延迟奖励换成稠密即时监督,于是同时拿到 SFT 的快收敛和 RL 的少演示/抗遗忘【原文 §1、Table 1】。
- 最巧的一步：**用 reverse-KL（而非 Forward-KL 或 Hard-CE）当对齐目标**。先解释这三个名词:KL 散度衡量两个分布的差异,方向不同行为也不同;Hard-CE 是只对 teacher 的 argmax(最可能动作)算交叉熵、丢掉软概率。抽掉 reverse-KL 这步就垮——因为 student 在 OOD 失败态(没见过的状态)采样时,teacher 在那些态往往是平坦高熵分布(自己也拿不准):
  - Forward-KL \(\mathrm{KL}(\pi_{\text{tea}}\|\pi_\theta)\) 是 mode-covering(逼 student 覆盖 teacher 的每个模式),会逼 student 复刻 teacher 的"犹豫"→熵爆炸;
  - Hard-CE 只学 argmax、丢掉 soft 概率(即所谓 dark knowledge),在多模决策边界上刚性追 argmax→过早熵坍缩、丧失探索多样性;
  - reverse-KL \(\mathrm{KL}(\pi_\theta\|\pi_{\text{tea}})\) 的 zero-forcing 性质让 student 只要落在 teacher 可接受的概率质量内就不被罚,于是"果断但仍能探索",熵保持健康【原文 §3.4、Fig.4(b) 三条熵曲线一一对应】。

## 为什么做

> 一句话导读：预训练 VLA 直接部署不可靠必须微调;但 SFT 太脆、稀疏 RL 太慢、简单把 SFT 改 on-policy 又用错了对齐目标——本文要同时治这三个病。

- 研究背景：预训练 VLA（视觉-语言-动作模型,把感知+规划+控制统一进一个 transformer 来输出机器人动作 token）泛化强,但直接部署到具体下游任务时执行不可靠,必须后训练【原文 §1】。当前两大范式:离线 SFT（行为克隆,dense 监督、收敛快）和在线 RL（GRPO,免 critic、用 group-relative 优势）。
- 解决的具体痛点：
  1. SFT 是 off-policy——在专家状态上训、在 student 诱导状态上评测,复合误差累积,无法从自致偏离恢复;且对 static/disjoint 数据集激进更新会导致灾难性遗忘【原文 §1、§2.2】。
  2. 稀疏奖励 RL——机器人任务通常只有终态 \(R(\tau)\in\{0,1\}\),信用分配难、方差高、样本效率极低【原文 §2.3】。
  3. 简单把 SFT 改成 on-policy（如 DAgger）却用了次优的对齐目标:Forward-KL 导致熵爆炸、Hard-CE 导致熵坍缩【原文 §1、§5.1】。
- 相关工作 & 各自不足（来龙去脉/并行路线/精确差异）：
  - **① 离线 SFT/BC**（O'Neill 2024、π0 Black 2024）——dense 监督、收敛快,但有 covariate shift、复合误差;OOD 态下策略未定义→灾难性失败;激进拟合 disjoint 分布→灾难性遗忘。
  - **② 在线稀疏奖励 RL**（SimpleVLA-RL、VLA-RL、RLinf-VLA 等,多用 GRPO）——能从自致偏离纠错,但十亿参数 VLA 上稀疏 outcome reward 的信用分配难、方差高、样本效率极低。
  - **③ 交互式模仿学习**（DAgger Ross 2011 / HG-DAgger Kelly 2019）——在 student 诱导的 OOD 态收专家标注(on-policy 状态分布对了),但用 Hard-CE/Forward-KL 对齐次优,引出熵爆炸/坍缩。
  - **④ LLM 蒸馏侧**（MiniLLM Gu 2023、GKD Agarwal/Tan 2023）——首提 reverse-KL / on-policy 蒸馏;本文**把它迁移到动作预测域**,并显式论证 OOD 态的熵动力学。
  - **与最近邻 SimpleVLA-RL（它正是本文的 teacher）的精确Δ**：
    - (a) 把稀疏 outcome reward 换成稠密 token-level reverse-KL 内在奖励;
    - (b) **不做 outcome reward 归一化**,直接用 raw reverse-KL reward 当优势（沿用 SimpleVLA-RL 的 group 采样框架 batch=64/G=8,但把奖励信号换掉）【原文 §3.4 末句、式7】。
- 动机链（逐步推进）：
  1. 预训练 VLA 需后训练;
  2. SFT 快但脆弱+遗忘、RL 鲁棒但太慢;
  3. 既要 dense 监督又要 on-policy 的状态分布;
  4. 于是在 student 自己的轨迹上用冻结 teacher 打稠密 token 标签(把稀疏奖励变即时监督);
  5. 但 OOD 态 teacher 高熵,朴素地全分布蒸馏有害;
  6. 改用 reverse-KL 的 bounded mode-seeking 来过滤 teacher 尾部的不确定性;
  7. 这种温和对齐把 student 锚在自己的流形上,缓解遗忘【原文 §1、§3】。

## 怎么做 + 靠不靠谱

> 一句话导读：三阶段闭环——学生先探(进 OOD 失败态)、老师再标(在那些态注入恢复动作)、再 mode-seeking 更新;最扎实的证据是 Fig.4 的三条熵曲线,直接对应 Forward-KL 爆炸/Hard-CE 坍缩/reverse-KL 健康。

- 方法流水线（三阶段闭环,Algorithm 1,逐 phase 看输入→输出）：student \(\pi_\theta\)（用极少演示 SFT 初始化）→
  - **Phase 1 on-policy 采样（Exploration）**：在环境里跑 \(G\) 条轨迹 \(\{\tau_1,\dots,\tau_G\}\sim\pi_\theta(\cdot\mid o)\),收集 \(\mathcal D_k=\{\tau\mid\tau=(s_0,a_0,\dots,s_T)\}\),其中 \(a_t\sim\pi_{\theta_k}(\cdot\mid s_t)\)、\(s_{t+1}\sim P(\cdot\mid s_t,a_t)\)（式3）。brittle student 会频繁进入 OOD 失败态 \(s_{\text{err}}\),**显式暴露能力边界**——把"unknown OOD 区域"转成"known 训练数据"。
  - **Phase 2 teacher 标注（Correction）**：对 student 访问过的每个状态 \(s_t\),查询冻结 teacher 的 action 分布 \(q_t(a)=\pi_{\text{tea}}(a\mid s_t)\)（式4）。teacher **只打标、不在环境里执行**,注入"optimal recovery prior"(最优恢复先验)。
  - **Phase 3 mode-seeking 更新（Update）**：算 token-level 内在奖励 \(r_t^{\text{OPD}}(s_t,a_t)=-\big(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\text{tea}}(a_t\mid s_t)\big)=-\log\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{tea}}(a_t\mid s_t)}\)（式6）,对 student log-prob 项加 stop_gradient,等价于在 student-visited 状态上最小化 reverse-KL;再用 group 平均 policy gradient 降方差（式7）→ 更新 \(\pi_\theta\)【原文 §3.1-3.4、Algorithm 1、式3-7】。
- 逐组件必要性（每组件 + 消融证据）：
  - **reverse-KL 对齐目标**：核心。Fig.4(a) 的消融（RoboTwin2.0 Beat Block Hammer）显示:Forward-KL 早期有 >50% 的性能谷、Hard-CE 平台最低、reverse-KL 稳步上升;Fig.4(b) 对应的熵曲线为 Forward-KL 爆炸 / Hard-CE 坍缩 / reverse-KL 健康。**有消融,证据最直接**【原文 §4.4-1】。
  - **on-policy 采样（vs 离线 SFT）**：这是抗遗忘机制的来源——梯度锚在 student 当前的行为流形 \(d^{\pi_{\theta_k}}\)、只在 student 自然访问的轨迹上蒸馏,避免 SFT 式拟合 disjoint 分布带来的激进参数漂移。Fig.3 的 seen-unseen 散点显示:离线 SFT 在 unseen 上坍缩(Object 近零),而 on-policy 法(RL 与本法)基本避免。**有对照**【原文 §3.2、§4.3】。
  - **dense teacher 监督（vs 稀疏 RL）**：这是效率的来源——把延迟稀疏奖励变成即时监督。Fig.2 给出收敛步数对比(10/50 步 vs 150 步)。**有对照**【原文 §3.3、§4.2】。
  - **group size \(G\)**：Fig.5（LIBERO-Object,batch=32）显示 \(G=8\) 最高(约 89%),\(G\in\{2,4\}\) 仍 >80% 不坍缩——小 \(G\) 省 rollout/teacher 推理。**有消融**【原文 §4.4-2】。
  - **stop_gradient 于 student log-prob / 不做 outcome 归一化**：未单独消融——"raw reverse-KL 作优势仍稳定"只有主实验的经验曲线支撑,没有单独 ablation【原文 §3.4,推断:无对照】。
- 关键机制/公式（真实符号 + 直觉,从 PDF 抄准）：
  - **离线 SFT 基线（off-policy 痛点来源）**：\(\displaystyle \mathcal L_{\text{SFT}}(\theta) = -\mathbb E_{(s,a)\sim\mathcal D_{\text{demo}}}\big[\log\pi_\theta(a\mid s)\big].\)（式1）在专家态 \(\mathcal D_{\text{demo}}\) 上训、在 student 诱导态上评→distribution shift。
  - **GRPO 稀疏奖励基线**：\(\displaystyle J_{\text{RL}}(\theta)=\mathbb E_{s\sim\mathcal D,\,\tau\sim\pi_{\theta_{\text{old}}}}\!\Big[\tfrac1G\sum_{i=1}^G\min\!\Big(\tfrac{\pi_\theta(\tau_i)}{\pi_{\theta_{\text{old}}}(\tau_i)}\hat A_i,\ \mathrm{clip}\big(\tfrac{\pi_\theta(\tau_i)}{\pi_{\theta_{\text{old}}}(\tau_i)},1-\epsilon,1+\epsilon\big)\hat A_i\Big)\Big],\)（式2）其中 \(\hat A_i\) 由稀疏 outcome reward 经 group-relative 归一化得——信用分配难。
  - **reverse-KL 作 dense reward（核心,式5-6）**：目标改为最大化负 reverse-KL \(\displaystyle \max_\theta J(\theta)=\mathbb E_{s\sim\pi_\theta}\big[-D_{\mathrm{KL}}\big(\pi_\theta(\cdot\mid s)\,\big\|\,\pi_{\text{tea}}(\cdot\mid s)\big)\big],\) token 级即得内在奖励 \(r_t^{\text{OPD}}=-\log\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{tea}}(a_t\mid s_t)}\)（式6）:student 动作匹配 teacher 时奖励→0,偏离则给大负惩罚。算梯度时对 \(\log\pi_\theta(a_t\mid s_t)\) 施 stop_gradient(把奖励当常数、梯度只走 score-function 项)。
  - **group-based 梯度估计（式7,降方差）**：\(\displaystyle \nabla J \approx \frac{1}{B\times G}\sum_{j}\sum_{i=1}^{G}\sum_{t}\nabla_\theta\log\pi_\theta(a_{t,i}\mid s_{t,i})\cdot r_t, \qquad \theta\leftarrow\theta+\alpha\nabla J.\)
  - **三种散度的优化行为（直觉）**：
    - Forward-KL \(D_{\mathrm{KL}}(\pi_{\text{tea}}\|\pi_\theta)\):梯度在 teacher 样本上估(\(\mathbb E_{a\sim\pi_{\text{tea}}}\))→mode-covering→OOD 态复刻 teacher 高熵犹豫→熵爆炸;
    - Hard-CE:弃 soft 概率、只追 argmax→多模边界上刚性振荡→熵坍缩;
    - reverse-KL \(D_{\mathrm{KL}}(\pi_\theta\|\pi_{\text{tea}})\):zero-forcing(teacher 概率≈0 处 student 必须≈0,否则惩罚趋于无穷),但 teacher 有质量的地方 student 可以只占主模、忽略长尾→bounded mode-seeking【原文 §3.4】。
- 实验与证据：
  - 数据集/设置：
    - **LIBERO**（单臂,四套件 Spatial/Object/Goal/Long,每任务仅 1-traj SFT 初始化,极端数据稀缺）;
    - **RoboTwin2.0**（双臂协作,四个代表任务覆盖 short→long horizon,每任务 1000-traj SFT 初始化）。
    - teacher=SimpleVLA-RL（性能 oracle）,student=OpenVLA-OFT;主实验 batch=64、\(G=8\)(沿用 SimpleVLA-RL),消融固定 batch=32。**全为仿真,无真机**【原文 §4.1】。
  - 关键数字：
    - LIBERO（Table 2,成功率%）:student init 平均 48.9 → Distill 87.4（媲美/超过 50-traj 全量基线 Octo 75.1、OpenVLA 76.5）→ Distill+GRPO 93.4（逼近 teacher 93.9）。
    - RoboTwin2.0（Table 3）:student init 45.2 → Distill 71.1（近 teacher 74.0）,超过 π0(50.5)、RDT(32.0),**未报 Distill+GRPO**。
    - 效率（Fig.2）:LIBERO-Object 蒸馏 10 步内 >90%,LIBERO-Long 50 步即达到基线 150 步的水平（约 3× 加速）【原文 §4.2、Fig.2】。
  - baseline 公平吗：与 GRPO 对照时设同 batch/\(G\),算公平;但与稀疏 RL 比"效率"时**未计入 teacher 自身的训练成本**——teacher 本身就是由 RL(SimpleVLA-RL) 训出来的,所谓"省掉 RL 探索成本"实质上是把成本前移到了 teacher 训练【推断:依据 §4.1 teacher 来源 + §5.2 "decouple RL exploration from policy optimization",作者未控制此项成本】。
  - 有无"看着强但没回答核心问题"：消融多为单基准/单任务曲线（Fig.4 仅 1 个 RoboTwin 任务）,**无多 seed 方差、无显著性检验**;"近 teacher"也意味着受 teacher 性能上界约束（作者承认 §6）。
- 假设与失效边界：
  - 显式假设【原文】：高性能 teacher 的先验可得（开源 checkpoint / API / 易训的单任务策略,§1）;teacher 全程冻结（§3.1）。
  - 隐式假设【推断】：OOD 态 teacher 呈"平坦高熵"是 reverse-KL 论证的前提——若 teacher 在 OOD 态反而过自信但错误,reverse-KL 会把 student 锁进错误主模(zero-forcing 的反面),论文未讨论此情形。
  - 何时失效【推断】：teacher 本身弱/有系统偏差时方法上界受限(作者承认);真机感知噪声/物理失配下未验证(仅仿真);"不做归一化仍稳定"无理论保证,换任务/换 backbone 未必成立。
- 祛魅总结【推断】：真贡献是把 LLM 侧成熟的"reverse-KL on-policy 蒸馏"干净迁移到 VLA 动作域,并用熵动力学把 Forward-KL/Hard-CE/reverse-KL 三者的优化行为讲清楚 + Fig.4 熵曲线作证(这点扎实)。被适度包装的是"bridging SFT and RL"的新颖性——本质是 SimpleVLA-RL 框架换了奖励信号(稀疏 outcome → 稠密 reverse-KL);且"样本效率碾压 RL"在不计 teacher 训练成本时才成立,公平性存疑。代码 Coming Soon、无真机、无多 seed,复现性与统计稳健性偏弱。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：teacher 在 student-visited(含 OOD)状态上的 action logits 分布;转成 token-level 负 reverse-KL 内在奖励 \(r_t^{\text{OPD}}=-\log\frac{\pi_\theta}{\pi_{\text{tea}}}\)【原文 §3.3-3.4、式6】。
  - **改什么**：student VLA 策略参数 \(\pi_\theta\)（on-policy policy gradient 更新）【原文 §3.1、式7】。
  - **何时改**：闭环每迭代——采样→teacher 标注→更新,直至收敛【原文 Algorithm 1】。
  - **免梯度?**：否。是基于梯度的 RL（group-based policy gradient）,但对 student log-prob 项用 stop_gradient(奖励侧不回传)【原文 §3.4】。
  - **记忆-技能生命周期**：无显式记忆库/技能库;"恢复先验(recovery behavior)"直接固化进 student 参数;on-policy 锚定缓解遗忘但无外部存储【原文 §3.2-3.3,推断:纯参数化巩固】。
  - **防遗忘机制**：on-policy 的"温和对齐"——梯度锚在 student 当前行为流形上、只在 student 自然访问的轨迹上蒸馏,避免 SFT 式拟合 disjoint 分布带来的激进参数漂移;Fig.3 的 seen-unseen 散点为证【原文 §3.2、§4.3】。
- ⑦ 开源代码+框架/harness：项目页 https://irpn-lab.github.io/VLA-OPD/ 标注 **"Code (Coming Soon)"**,截至核查无可用仓库（`resource/repos/` 下无 vla_opd）,论文未给 GitHub 链接——**未开源(待放出)。未完整克隆,原因:代码尚未发布;附项目页链接供后续手动获取**。论文层面:student=OpenVLA-OFT,teacher=SimpleVLA-RL,分组采样沿用 SimpleVLA-RL（batch=64, \(G\)=8）,优化为 group-based policy gradient(类 GRPO 但不做 outcome reward 归一化)【原文 §4.1-4.2,框架未明说具体训练库,待核】。
- 💰 资源/成本与可扩展性：原文未给 GPU 型号/卡时/wall-clock 数字。可观测的相对成本信号:Fig.2 的收敛步数(蒸馏 10-50 步 vs 稀疏 RL 150 步);Fig.5 显示小 \(G\)(=2) 即可大幅降低 rollout+teacher 推理开销。**未计入 teacher 训练成本**【原文 §4.2、§4.4-2】。
- 🎯 对"探索-巩固"对标：**支撑（域外平行证据）**——一句判定:VLA-OPD 是"teacher 当稀疏脚手架教 student 在自走轨迹上恢复"的机器人版直接实例,与本项目 idea 高度同构,但落在动作 token 而非语言推理。依据:
  - 探索 = Phase 1 student 主动进入 OOD 失败态（§3.2 "active correction"）;
  - 巩固/回轨 = Phase 2 teacher 在失败态注入"recovery prior" + Phase 3 固化进参数（§3.3 "optimal recovery behaviors"）;
  - on-policy 自选轨迹 + reverse-KL 让 student "偏向自己能走通的主模"(zero-forcing 只对齐主模)——这恰对应 idea 里"偏向自己能走通的开头/自选恢复分支"。
  - **可借组件**：token-level 负 reverse-KL 作内在奖励 \(r_t^{\text{OPD}}\) + stop_gradient 这一"forward 用 student 采样、backward 走 score-function"的写法,可直接挪到 LLM long-CoT 的 path-recovery 单点接管（与本项目 reusable-techniques 记忆里 forward-hard/backward-soft 的"有界梯度"同构）。
  - **缺口**：
    1. 是单 teacher 全程冻结,无 MTP 式前瞻、无多 teacher 一致性;
    2. 无显式记忆/技能库;
    3. 仅机器人短-中 horizon,未涉长链推理的稀疏关键步切分（与本项目 survey-grpo-step-segmentation 记忆里的"切高熵/低置信关键步"无对应机制）。
- 🔭 开放问题/未来方向：
  - 【原文】降低对特定 teacher 模型的依赖（作者列为主攻 future work,§6）;把"多 teacher 蒸进统一/升级的 student backbone"（§1 末提及）。
  - 【推断】把"raw reverse-KL 作优势、不归一化"补上理论收敛/方差分析;扩到真机验证泛化;与中间步奖励塑形结合（当前 dense 奖励纯来自 teacher logits,无任务结构信号）。
  - 【推断】OOD 态 teacher 若过自信而错误时的失效处理（zero-forcing 锁错模问题）——可引入 MTP 前瞻或多 teacher 一致性来探测 teacher 可靠性（呼应 unisd 的 agreement 门控）,正是本项目 idea 的延伸点;并自查 reverse-KL 是否如 why_sd_degrade 警示般压制 epistemic 表达。

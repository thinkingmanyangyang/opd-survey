rl_survey_lrm | A Survey of Reinforcement Learning for Large Reasoning Models | 清华 + 上海AI Lab + 上海交大 + 北大 + 中科大 + 哈工大 + 华盛顿大学 + 华科 + UCL(40+ 作者;Project Lead Kaiyan Zhang/Yuxin Zuo;通讯 Biqing Qi/Ning Ding/Bowen Zhou) | 2025-10-10 v3 · arXiv 综述(cs.CL) | L3 RLVR/GRPO(综述,横跨 L1-L6) · 相关性中(领域坐标系/背景)

**原始论文**:https://arxiv.org/abs/2509.08827

> 注:本条为**综述**(非方法论文)。下列"怎么做/实验/方法"等栏按综述视角填写其 **taxonomy 与覆盖范围**,而非单一方法。

## 一眼看懂
- 🟦 TL;DR:系统综述 DeepSeek-R1 以来"RL for 大型推理模型(LRM)"范式,用一张总图(Fig.1)组织成五层——**基础组件(§3)/ 五对开放争议(§4)/ 训练资源(§5)/ 下游应用(§6)/ 未来方向(§7)**;核心论断:**RLVR(可验证奖励 RL)是区别于 RLHF/DPO(人类对齐)的新 scaling 轴**,把 LLM 转成 LRM,且其向 ASI 的可扩展性是核心开放问题。配套 GitHub Awesome-list 维护文献清单。【原文】Abstract、Fig.1、§1
- 最巧的一步(对综述而言=组织骨架):**五对"X or Y"对立式开放问题(§4)**——Sharpening or Discovery、Generalize or Memorize、Weak or Strong prior、Tricks or Traps、Process or Outcome。抽掉它,综述就退化成"文献罗列";正是这五对争议把海量快速演进的工作结构化成可追踪的分歧坐标系。【原文】§4、Fig.1

## 为什么做
- 研究背景:RL 在 AlphaGo/AlphaZero 时代已证窄而明确奖励可驱动超人表现;LLM 时代先以 RLHF/DPO 做人类对齐(3H)。近期出现 **RL for LRMs**——不止对齐行为,而是直接激励推理本身(OpenAI o1、DeepSeek-R1 两里程碑用 RLVR 诱导长链推理:规划/反思/自纠错,且随训练与推理算力平滑提升)。【原文】§1、§2
- 解决的具体痛点(综述视角):领域(尤其 R1 后)爆发式发展,进一步扩展 RL for LRM 面临算力/算法设计/训练数据/基础设施的基础性约束,亟需重访轨迹、重估方法论、探索通向 ASI 的可扩展策略。【原文】Abstract、§1
- 相关工作 & 各自不足:§2.3 梳理已有相关综述以定位自身(强调本综述聚焦 R1 后、以"语言智能体与环境长期演化中的大规模交互"为核心视角)。【原文】§2.3
- 动机链:领域文献海量且演进极快 → 需坐标系 → 用"组件—争议—资源—应用—未来"五层 + 五对争议组织 → 便于读者建立坐标系并追踪。【原文】Fig.1、§1
- 与最近邻(其他 RL/LLM 综述)的Δ:明确把 **RLVR 从 RLHF/DPO 中切出(Fig.2)** 作为独立新范式,并以"scaling 向 ASI"为主线;覆盖到 agentic/多模态/多智能体/机器人/医疗等外溢应用。【原文】§1、§6

## 怎么做 + 靠不靠谱(=综述 taxonomy 与覆盖)
- 方法流水线(综述结构):§2 预备(RL 定义/o1 以来模型脉络/相关综述)→ §3 基础组件 → §4 五对争议 → §5 训练资源 → §6 应用 → §7 未来方向 → §8 结论;配 Awesome-list。【原文】Fig.1、目录
- 逐组件(各章覆盖内容):
  - **§3 基础组件**:§3.1 奖励设计(Verifiable / Generative / Dense / Unsupervised / Reward Shaping);§3.2 策略优化(Policy Gradient / Critic-Based 如 PPO / Critic-Free 如 GRPO·RLOO / Off-Policy / Regularization 目标);§3.3 采样策略(Dynamic/结构化采样、采样超参)。【原文】§3、Fig.1
  - **§4 五对开放争议**:§4.1 RL 的作用 Sharpening or Discovery(只锐化 base 已有分布 vs 发现新能力);§4.2 RL vs SFT Generalize or Memorize;§4.3 Model Prior Weak or Strong;§4.4 Training Recipes Tricks or Traps;§4.5 Reward Type Process or Outcome。【原文】§4
  - **§5 训练资源**:§5.1 静态语料(Math/Code/STEM/Agent/Mixture);§5.2 动态环境(Rule/Code/Game/Ensemble);§5.3 RL 基础设施(OpenRLHF / veRL / AReaL / slime / TRL)。【原文】§5、Fig.1
  - **§6 应用**:Coding、Agentic、Multimodal、Multi-Agent、Robotics、Medical。【原文】§6
  - **§7 未来方向(9 项)**:7.1 Continual RL、7.2 Memory-based RL、7.3 Model-based RL、7.4 Efficient Reasoning、7.5 Latent Space Reasoning、7.6 RL for Pre-training、7.7 RL for Diffusion LLMs、7.8 Scientific Discovery、7.9 Architecture-Algorithm Co-Design。【原文】§7 目录
- 关键机制/核心论断(直觉):RLVR ≠ RLHF/DPO——前者用可验证信号(数学答案对/代码单测过)激励推理能力本身,后者用偏好对齐人类;这是不同 scaling 范式(Fig.2)。五对争议尚无定论,是当前活跃前沿。【原文】§1、Fig.2、§4
- 实验与证据(综述无原创实验):§5 梳理训练资源(静态语料/动态环境)与复用性,§5.3 比较主流基础设施;论断为对现有文献的归纳。【原文】§5
- 假设与失效边界:
  - 【原文】综述聚焦 o1/DeepSeek-R1 发布以来工作(v3 2025-10),范围明确。
  - 【推断】结论依赖所引文献质量;领域演进极快,v3 后工作必然遗漏;五对争议多呈现分歧而少定论性裁决;"RL 通向 ASI"为前瞻论断,缺可证伪具体路径;§5.3 基础设施比较随框架迭代易过时。
- 祛魅总结:
  - 真贡献:为快速演进、文献海量的"RL for LRM"领域提供权威坐标系;五对争议提炼清晰,§7 未来方向(尤其 Continual/Memory-based/Model-based RL)指向性强;Awesome-list 可追踪。【推断】
  - 边界:作为综述不提供新实证;对本项目的价值是"定位与查文献",而非可复现方法。【推断】

## 结构化抽取
- 🎯 机制速览6轴(综述 taxonomy 对应):学什么信号=综述把奖励分为 Verifiable/Generative/Dense/Unsupervised(§3.1);改什么=策略优化谱系 PG/Critic-Based/Critic-Free/Off-Policy(§3.2);何时改=采样策略 Dynamic Sampling(§3.3);免梯度?=不适用(综述);记忆-技能生命周期=§7.1 Continual RL / §7.2 Memory-based RL 专章讨论(与本项目 L5 直接对应);防遗忘机制=§7.1 明确讨论 CRL 的 stability-plasticity 平衡、可塑性丧失(Dohare et al. 2024 指深度网络在持续学习中退化)。【原文】§3、§7.1-7.2
- ⑦ 开源代码+框架/harness:https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs(**awesome-list,仅论文/资源清单,无方法实现代码**,不可运行复现)。综述本身无独立训练框架;§5.3 梳理并比较主流 RL 基础设施 **OpenRLHF / veRL / AReaL / slime / TRL**。【原文】§5.3 + 仓库(v1 核查)
- 💰 资源/成本与可扩展性:综述本身无成本;但全文核心议题之一即"RL for LRM 的 scaling(算力/算法/数据/基础设施约束)向 ASI",§5 讨论训练资源复用性。【原文】Abstract、§5
- 🎯 对"探索-巩固"对标:**坐标系/背景支撑,非方法对标**。判定:它为本项目提供文献定位——(1) §4.1 Sharpening vs Discovery 直接关系本项目"探索=发现有效路径"是真发现还是锐化已有分布;(2) §4.5 Process vs Outcome 关系 token/step 级信用分配(本项目 path 监督);(3) **§7.1 Continual RL + §7.2 Memory-based RL 正对应本项目 L5(记忆/技能库 & 持续学习防遗忘)与"巩固/固化且不遗忘"**,可作该方向的权威背景与引文来源;(4) §7.5 Latent Space Reasoning、§7.4 Efficient Reasoning 关系 CoT/前瞻。可借:用其 taxonomy 给 SURVEY 的章节定位与术语统一;缺口:不含 OPD/自蒸馏的专门章节(OPD 散见于 §3.2 正则/off-policy 与应用),也无 MTP 前瞻专题。依据:Fig.1、§4、§7。【推断】
- 🔭 开放问题/未来方向:【原文】§4 五对未决争议 + §7 九项未来方向(Continual / Memory-based / Model-based RL、Efficient/Latent Reasoning、RL for Pre-training/Diffusion、Scientific Discovery、Arch-Algo Co-Design);下一阶段 scaling(含 open-ended RL)向 ASI 仍开放。【推断】综述未给五对争议的定论,留待读者按自身场景判断;v3 后(2025-10 至今)的新工作需另行补充。

RETURN: rl_survey_lrm|读到PDF?是(综述,_txt 420k字,核到 Fig.1 总图+§4 五争议+§7 九方向)|L3(综述,横跨 L1-L6)|对标=坐标系/背景:§7.1 Continual+§7.2 Memory-based RL 直对应本项目 L5 防遗忘/巩固,§4.1/§4.5 关系探索真伪与 token 信用;无 OPD/MTP 专章|残留待核 0

hpd | Hybrid Policy Distillation for LLMs (HPD) | 上海交大 / 上海创智学院 / 腾讯(Wenhong Zhu, Ruobing Xie, Rui Wang, Pengfei Liu) | 2026-04-23 Preprint·arXiv 2604.20244v1(README 称 ICML 2026,正文未见〔待核〕) | 主题线 L1 OPD/自蒸馏 · 相关性 高

**原始论文**:https://arxiv.org/abs/2604.20244

## 一眼看懂
- 🟦 TL;DR:白盒知识蒸馏(用 teacher logits 教小模型)长期被三个互相纠缠的旋钮卡住——散度方向(前向 KL vs 反向 KL)、优化方式(loss vs reward)、数据来源(on-policy vs off-policy)。本文先把 SFT/SeqKD/FKLD/RKLD/JSD 统一写成同一个"token 级重加权对数似然"目标(只是权重 w 不同,见 Eq.9 / Table 1),再提出 HPD:对每个 teacher token 用 K1 估计器(Schulman 2020 的廉价 KL 估计 `k1=q·(log p − log q)`)逐 token 判断 student 是低估还是高估了这个 token,据此自适应在前向/反向 KL 之间切换,并把被抑制 token 的概率质量重新导回 teacher token。既保留 one-hot 监督的省算力,又兼容 off-policy 数据 + 轻量近似 on-policy 采样(在 offline 前缀下让 student 采一个非专家 token 来识别并压制"不合理行为")。
- 最巧的一步:**K1 估计器的符号当作"前向/反向 KL 的自动开关"**(Eq.11-12)。`k1>0`(student 低估 expert token)→ 加前向 KL 式增强权重 `p(a*|s)+k1`;`k1≤0`(已高估)→ 取负权重抑制(等价反向 KL 行为)。抽掉这一步(改回固定系数加权和 FKL+RKL),就回退成普通 weighted-sum 蒸馏,失去"逐 token、按需、无超参"的混合,训练动态会像 SFT 一样早熵崩(Fig.1a)。这一步是全篇支点,因为它同时解决了"方向选择"和"无额外超参"两个痛点。

## 为什么做
- 研究背景:压缩大模型靠 KD;白盒 KD 能用 teacher 的预测分布做分布级匹配(KLD on logits)。近年工作强调要选对散度,但作者指出"光选散度不够"。
- 解决的具体痛点:三轴(散度方向 / 优化策略 / 数据 regime)被孤立选择、缺统一视角【原文 §1】;FKL 促 mode coverage 但过平滑、RKL 促 mode-seeking 但师生差距大时不稳;OPD 避免 train-inference 失配但开销大且 teacher 侧分布漂移。
- 相关工作 & 各自不足:GKD(Agarwal 2024,on-policy 学生序列)、MiniLLM(Gu 2023,强调反向 KL)、DistiLLM、SeqKD(Kim&Rush 2016)、JSD —— 各自只动一个旋钮,没把三轴统一,也没给"逐 token 自适应方向"的机制。
- 动机链:现状(三轴孤立、单向散度各有缺陷、OPD 贵)→ 缺陷(无法兼得双向互补性 + one-hot 效率 + on/off 兼容)→ 所以必须把 KD 统一成重加权似然,并用一个廉价逐 token 估计器自动调方向 + 轻量采样近似 on-policy。
- 与最近邻工作的 Δ:相对 weighted-sum KL(固定系数混 FKL/RKL),HPD 用 **mask + K1 符号** 做逐 token、无超参的方向选择,并显式把被压制 token 的概率质量"重定向回 expert token"(Reinforce 操作,Eq.14)。相对 GKD/MiniLLM 的全分布 on-policy 监督(贵),HPD 用 one-hot 风格 + offline 前缀 + 单 token 采样近似,省算力。关键好处:在算力受限下逼近 dense 蒸馏且训练稳(不熵崩)。

## 怎么做 + 靠不靠谱
- 方法流水线(Algorithm 1):① 从 offline 数据 D 采轨迹(teacher 生成或 ground-truth 近似 teacher 分布)→ ② 对每个 (s_t, a*_t) 算 expert token 的 `k1=q(a*|s)·(log p(a*|s) − log q(a*|s))` → ③ 在同一 offline 前缀下让 student 采一个非专家 token a_t∼q(·|s),算其 `k'1` → ④ 算 expert 权重 w*_t(三分支:k1>0 且 k'1<0 → `2p+k1` 加倍强化;k1<0 → `k1` 抑制;否则 `p+k1`)+ 采样 token 权重 w_t = 1[a≠a*]·1[k'1<0]·k'1(只压制被高估的非专家 token)→ ⑤ 合成损失 `L_HPD = −w*_t·log q(a*|s) − w_t·log q(a_t|s)`(Eq.15)→ ⑥ 梯度更新。可与标准 OPD 叠加(§5.4,HPD+OPD)。
- 逐组件必要性(都有消融,§6,作者强调 HPD 零额外超参):
  - **Student sampling(让 student 采自己的非专家 token)**:去掉则性能快速收敛后立刻 plateau(只朝 teacher 分布优化→限制探索、早收敛)。有消融【原文 Fig.3】。
  - **Reinforce 操作(被压制时把概率质量导回 expert token,即 w* 的 `2p+k1` 分支)**:去掉则 KL 下降变慢、对齐更慢。有消融【原文 Fig.3】。
  - **Hybrid FKL/RKL 的 mask(k1≤0 时屏蔽前向权重)**:防止前向/反向梯度方向冲突;未单独做 mask 的消融,但其必要性由 Eq.12 论证 + 整体训练动态(不熵崩)间接支撑【推断:依据 §4.2 + Fig.1a】。
- 关键机制/公式(直觉):统一目标 Eq.9 把所有 KD 写成 `−E[w(a|s)·log q(a|s)]`,权重 w 正则增大该 token 似然、负则抑制并按当前分布把概率质量摊给其他 token(Eq.10 梯度跨整个词表传播)。K1 的符号天然区分"低估/高估",于是充当前向/反向 KL 的自动选择器。
- 实验与证据:
  - 模型:Qwen2.5(1.5B/3B/7B)、LLaMA3(1B/3B/8B),各取最大者当 teacher;Coder 任务用 Qwen2.5-Coder-7B/DeepSeek-Coder-6.7B 教 1.5B/1.3B。数据:OpenR1-Math-8192(推理)、UltraFeedback(个性化)、WizardCoder(代码)。
  - 推理 off-policy(Table 2):HPD 全面超 SFT/SeqKD/RKLD/JSD,带 ∗(p<0.01)。最亮点:**Qwen2.5-3B 提升 41.0%(28.25→39.83 avg)、LLaMA3-3B 提升 77.9%(19.43→34.56)**,让 3B 逼近大模型推理。
  - on-policy 推理(Table 5):"HPD alone(纯 off-policy)" 已超过 "SFT→OPD 两阶段 baseline";HPD+OPD 进一步最高(Qwen 7B→1.5B 33.41 vs SFT+OPD 29.56)。GPQA 为 OOD,也有提升 → 不只是过拟合 teacher 行为。
  - 训练动态(Fig.1):SFT 早熵崩、KL 停滞、性能不动;HPD 熵稳、KL 持续降、性能持续升,且 train/inference 熵一致(行为对齐)。
  - 个性化(Table 3)/ 代码(Table 4):HPD 平均最优,尤其多轮对话(MT-1T/MT-2T)保持力强;代码上单点不总最高但方差小、更稳。
  - baseline 公平性:作者把所有 baseline 统一近似为"重加权似然的不同权重估计器"(SFT=常数 1、SeqKD=p、RKLD=q(log q−log p)、JSD=½q(log q−log((p+q)/2))),同框架同数据,公平【原文 §5 + App C/E】。
- 假设与失效边界:
  - 【原文】用 "teacher 分布 ≈ 其训练数据经验分布" 来省 teacher 在线 forward(§5.1.1:先在 offline 数据上训 teacher 再 GRPO),即 offline 数据当作 teacher 软标签的代理。
  - 【推断】依赖 teacher logits 可得(白盒 KD),黑盒 API teacher 不适用;K1 是 KL 的一阶近似,师生分布差极大时近似质量与方向判定可能不稳(原文未量化此边界)。
  - 【推断】"offline 前缀 + 单 token 采样近似 on-policy" 仍非真正 on-policy 多步 rollout,长程误差累积场景下逼近程度未知。
- 祛魅总结【推断】:
  - 真贡献:① 把 KD 三轴收进一个重加权似然框架(Table 1 的统一)是清晰、可复用的视角;② K1 符号当方向开关 + 概率质量重定向,实现"逐 token、无超参"混合 KL,且经验上稳(治住 SFT 熵崩)。
  - 包装/可能高估:"hybrid policy / 轻量 on-policy" 名头大,实质是 offline 主导 + 单 token student 采样的近似,并非完整 OPD;增益主要来自方向自适应 + 防熵崩,而非真正 on-policy 探索。低估之处:作为 SFT 即插即用的 loss(LlamaFactory 一行开关 `use_hpd_loss`)实用价值可能被它"统一视角"的理论包装掩盖。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=逐 token K1 估计器(student 对 teacher token 的 under/over-estimate 符号)|**改什么**=student 参数(token 级重加权 NLL)|**何时改**=每 token、按 k1/k'1 分支自适应|**免梯度?**=否,标准反向传播梯度更新|**记忆-技能生命周期**=无显式记忆/技能库,知识固化进权重|**防遗忘机制**=隐式——student sampling + 不熵崩维持探索性,多轮对话保持力强(Table 3);无显式 anti-forgetting 模块。
- ⑦ 开源代码+框架/harness:https://github.com/zwhong714/Hybrid-Policy-Distillation (已 clone,~40MB,真实可用)。**双实现**:① **LlamaFactory**(SFT 路径,HPD loss 在 `LlamaFactory/src/llamafactory/train/hpd.py`,开关 `use_hpd_loss`,支持全参 + LoRA);② **veRL**(RL/post-training 路径,`verl/recipe/HPD/`,`hpd_trainer.py` 基于 `RayPPOTrainer`,FSDP/FSDP2 worker)。配套 HF 模型(Qwen2.5-1.5B-HPD)。
- 💰 资源/成本与可扩展性:卖点即"省"——保留 one-hot 监督效率、用 offline 数据 + 单 token 采样近似 on-policy,避免全分布 OPD 的 teacher 在线 forward 开销;student 训练 ~2k steps、batch 256(on-policy 为 64 prompts × 4 rollouts)。具体 GPU 数/时长原文未给出明确表格(原文未说明)。
- 🎯 对"探索-巩固"对标:**支撑 + 可借组件**。判定:HPD 的 "k1>0 强化 expert / k'1<0 压制非专家 + 概率质量重定向" 与本项目"探索(偏向能走通的开头)+ 巩固(走偏后压制错误分支、固化正确路径)"在**机制上高度同构**;依据:Eq.14 三分支 = 选路(强化 teacher token)+ 回轨(抑制 student 跑偏的非专家 token 并把质量导回)。可直接借:① 用廉价 KL 估计器的符号做"该模仿 teacher vs 该抑制自走偏"的逐 token 门控;② "被压制处把概率质量定向回正确 token" 的 Reinforce 写法,可作 path-recovery 单点接管的损失原型。竞品性:它是 offline 主导的逐 token 自蒸馏,**不含 MTP/前瞻**,也非多步 on-policy rollout,与 TSRD 的"on-policy 自选恢复分支"互补而非替代。
- 🔭 开放问题/未来方向:【原文】Roadmap 提到发布可复现 Docker + 把 HPD 扩到 mid-training。【推断】K1 方向判定在师生差距极大时的稳健性边界;把"单 token 近似 on-policy"升级为真多步 rollout 后增益是否还在;与 MTP/前瞻信号结合,用未来 token 预测来更早判定"该强化/该回轨"的位置。

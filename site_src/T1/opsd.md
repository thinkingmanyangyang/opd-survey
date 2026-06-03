# opsd — Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models

> **一句话重点 (TL;DR)**：同一个 LLM 自任教师与学生——教师条件于"问题 + 参考解(特权信息)"、学生只看问题,沿学生自采样轨迹做逐 token 散度蒸馏,无需外部教师即可用 ground-truth 提供稠密 on-policy 监督,在数学推理上匹配/超过 GRPO 且 token 效率显著更高。

**元信息**：arXiv 2601.18734（v3, 2026-03-20, cs.LG）｜ UCLA / HKU / Meta Superintelligence Labs（Siyan Zhao 等,Feiyu Chen 与 Aditya Grover 共同指导）｜ 2026 Preprint（博客 siyan-zhao.github.io/blog/2026/opsd/）｜ 主题 T1/T4,High（自蒸馏家族原型,直接定义 OPSD 范式）｜ 代码 github.com/siyan-zhao/OPSD（已 clone,~330KB,核心 trainer 完整）｜ 框架 TRL（基于其 experimental GOLD/GKD trainer）

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/opsd/fig_02.png)

*Figure 3. Token Efficiency of OPSD. We compare OPSD and GRPO on Qwen3-1.7B under the same effective training batch size, reporting Avg@12 accuracy with training steps and total tokens generated. Generation is capped at 1024 tokens for OPSD and 16k for GRPO. At the same number of training steps, OPSD*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/opsd/fig_01.png)

*Figure 1. Overview of On-Policy Self-Distillation (OPSD): Given a reasoning dataset S = { ( x i , y ⋆ i ) } N i =1 , we instantiate two policies from the same LLM: a student policy p S ( · | x ) and a teacher policy p T ( · | x, y ⋆ ) . The student generates an on-policy response ˆ y ∼ p S ( · | x )*

## 1. 相关工作与进展
推理 post-training 三大路线:RLVR(GRPO 等)、在高质量 CoT 轨迹上 SFT、知识蒸馏。on-policy 蒸馏(GKD [Agarwal 2024]、MiniLLM [Gu 2024]、Thinking Machines 的 OPD 博客 [Lu & Lab 2025])让学生采样自身轨迹、教师逐 token 稠密监督,兼具 on-policy 分布真实性与稠密反馈。与 STaR/ReST(条件于 hint/答案生成 rationale → 拒绝采样 → SFT,硬蒸馏)、context distillation [Snell 2022](特权上下文教师 + SFT 学生,off-policy)、in-context editing [Qi 2025](on-policy 软蒸馏内化上下文知识)相关。并发的 SDPO [Hübotter 2026]、SDFT [Shenfeld 2026] 探索同构的自蒸馏。

## 2. 现有工作存在的问题
(1) RLVR/GRPO:每 prompt 采一组响应成本高、方差大;一组全对/全错时优势为 0、梯度消失;奖励稀疏且对所有 token 一刀切。(2) SFT:exposure bias、泛化弱。(3) 传统蒸馏:off-policy 分布不匹配(训练见完美前缀、推理自生成)。(4) on-policy 蒸馏需要一个独立、往往更大的教师模型,且未显式利用推理数据集中已有的 ground-truth 解。

## 3. Motivation
现代 LLM 已具强推理能力,问:模型能否通过自蒸馏当自己的教师?既能省掉外部教师,又能直接利用数据集里的参考解作为特权信息。

## 4. 主要灵感 / 核心直觉
受人类学习启发:做错题后看正确解能 rationalize 步骤、定位自己错在哪;且"评估比生成更易"[Naor 1996, Sun 2024],推测"对给定正确答案做 rationalization"也比从零生成更易。故让模型在见到 y* 后隐式 rationalize,以此监督只见问题的弱版自己。

## 5. 主要解决思路(一段话讲清核心)
从同一模型 p_θ 实例化两条策略:教师 p_T(·|x,y*)(条件含问题 + 参考解),学生 p_S(·|x)(仅问题)。学生采样 on-policy 轨迹 ŷ~p_S(·|x);loss 最小化沿学生轨迹的逐 token 散度 D(p_T‖p_S)(ŷ|x)= (1/|ŷ|)Σ_n D(p_T(·|x,y*,ŷ_<n)‖p_S(·|x,ŷ_<n))。梯度仅经学生 logits 回传;教师只一次前向(prefill)隐式 rationalize、不真正生成 token(prompt 中要求教师"看完参考解后用自己的方法解",见 Fig.2)。

## 6. 方法详解(通俗、分步骤)

- **两条策略**:同参数 θ、不同条件上下文;教师额外看 y*。
- **on-policy 采样**:学生生成 ŷ,两条策略在同一学生前缀上各自给 next-token 分布。
- **训练目标(两种实例化)**:① 全词表 logit 蒸馏(如 GKD,full softmax,逐 token f-散度;效果更好但峰值显存高,因每位置存词表大小 logits);② 采样 token 的策略梯度(如 Lu & Lab 2025:把 A_n=log p_T(ŷ_n|·)−log p_S(ŷ_n|·) 当 stop-gradient 优势,做 reverse-KL 风格 policy gradient;省显存)。主实验用 ①。
- **教师固定为初始 policy**(非在线更新的学生),作隐式正则、稳定训练。
- **Per-Token Pointwise KL Clipping**:对每个 token×词表项的散度贡献 min(ℓ,τ) 裁剪。因风格 token('wait'/'think' 等连接词)的逐 token KL 比数学 token 高 6–15×(Table 5),不裁剪会让风格 token 主导信号、训练崩溃(Fig.4)。
- **消融结论**:forward KL > reverse KL/JSD(Table 3,AIME25/Qwen3-1.7B:FKL 36.7→43.9@step50);TM-off 学生 + TM-on 教师在数学 token 上 KL 最大、下游最好;生成长 1024 vs 4096 无一致增益(早期 token 更关键);全词表 > 采样 token(Table 4,Qwen3-4B/2048-gen,pass@8:AIME25 84.1 vs 82.1、HMMT25 60.0 vs 57.3)。

## 7. 实验数据集
训练:OpenThoughts 数学推理子集(采 ≤30K 问题-解对,含 CoT)。评测:竞赛级数学 AIME 2024、AIME 2025、HMMT 2025。主表 Table 2 报 **Avg@12**(温度 1.0、thinking 模式、max gen 38k,按 Qwen3 博客配置);消融 Table 4 报 pass@8(数值与 Avg@12 不可直接比)。模型:Qwen3-1.7B/4B/8B(instruct)。

## 8. 实验结果与主要发现
Table 2(Avg@12,基→OPSD):Qwen3-1.7B 37.1→43.4(超 GRPO 37.7、SFT 35.8;分项 AIME24 51.5→57.2、AIME25 36.7→43.9、HMMT25 23.1→29.2);4B 61.2→63.6(>GRPO 62.7);8B 61.8→64.8(>GRPO 64.0)。OPSD 每问题仅 1 rollout、生成 1024 token、约 100 步内收敛(Qwen3-1.7B 在 4×H100 上约 15 分钟,A100/H100 + LoRA);GRPO 用 8 rollouts×16k token,且 100 步内过半 batch 组内 reward 标准差为 0、梯度消失(Fig.3)。SFT 在简洁参考解上微调反而缩短测试生成长度、性能退化。

## 9. 结果如何支撑其主张
"token 效率"主张由 Fig.3 直接支撑(同训练步数下 OPSD 用更少 token 却全面超 GRPO,且 GRPO 因 reward 多样性坍缩停滞)。"无需外部教师 + 利用 ground-truth"由方法构造与 Table 1 对比(OPSD 同时满足 on-policy/稠密信号/低采样成本/无外部教师四项)支撑。机制证据较弱:全词表>采样 token、forward KL 最优等均有消融,但缺对"为何 rationalization 更易"的直接度量。

## 10. 逻辑自洽性(中性评估)
框架自洽:教师固定为初始 policy 既是稳定手段也是隐式 KL 正则,与 GRPO 的 reference-KL 思路一致。但教师"只 prefill 不生成"意味着 OPSD 本质是"在学生轨迹上、用 y*-条件分布做逐 token 匹配",其能否提供"新能力"取决于 y*-条件是否真把分布推向更优——这点论文未理论保证,survey(§7.1)亦指出 OPSD 的有效区间介于"教师-学生差距过小/过大"之间,且后续 Kim & Lee 2026 指出 OPSD 更像"压缩(让模型更高效表达已知解)"而非"纠错(教会解更难题)"。

## 11. 残留问题 / 局限
作者自陈:实验上限 8B(>8B 是否持续未知);未利用答案正确性验证信号(可作额外目标);若问题超出模型理解阈值,即便给 y* 教师也无法提供有效监督,需课程学习。外部视角:主表用 Avg@12、消融用 pass@8 易混淆;仅数学单域;教师固定初始 policy 可能限制能力上探(只匹配、难超越教师)。

## 12. 开源代码与框架(链接+框架+代码可得性)
github.com/siyan-zhao/OPSD(已 clone,~330KB,代码完整可得)。核心:`opsd_trainer.py`(OPSDTrainer 自蒸馏核心,72KB)、`data_collator.py`、`opsd_train.py`(入口)、`sft_train.py`/`grpo_train.py`(基线)、`scripts/run_opsd*.sh`、`eval/evaluate_math.py`(vLLM)。框架 **TRL**:`environment.yml` 确认 trl==0.26.0、transformers==4.57.1、accelerate==1.11.0、deepspeed==0.18.2、peft==0.17.1、vllm==0.11.0、torch==2.8.0;README 明示"基于 TRL experimental GOLD trainer"。

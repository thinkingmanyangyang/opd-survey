# opsa — Reducing the Safety Tax in LLM Safety Alignment with On-Policy Self-Distillation

> **一句话重点 (TL;DR)**：把 OPSD 自蒸馏迁到安全对齐——学生在线 rollout、frozen 自身副本(条件于按 prompt 类型选的安全/有益特权上下文)沿轨迹给逐 token KL 监督;创新点是用 **teacher flip rate(TFR)** 离线挑"能把不安全回答翻成安全"的上下文,从而在更小的推理代价下降低 "safety tax"。

**元信息**：arXiv 2605.15239（v1, 2026-05-14, cs.LG）｜ UC Riverside / ICSI / Microsoft / Berkeley Lab（Yu Fu, Longxuan Yu, … Yue Dong 通讯）｜ 2026 Preprint｜ 主题 OPSD/特权上下文自蒸馏家族成员（与 copsd/sdcl 同构,此处特权上下文=安全上下文）,对 OPD 主线 Med 相关｜ 代码 github.com/FYYFU/OPSA（Apache-2.0,~98% Python,已 clone）｜ 框架 NVIDIA NeMo-RL 扩展 fork

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/opsa/fig_02.png)

*Figure 2 Safety correction is concentrated in specific positions and tokens. We measure per-token symmetric KL between a safety-prompted teacher and each student (Base, ThinkSafe , OPSA) on harmful rollouts from the base model ( n =500 , Qwen3-0.6B). Left: Mean KL by position, with an inset for the*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/opsa/fig_01.png)

*Figure 1 Overview of OPSA. A frozen copy of the base model serves as the teacher, while the student is updated on rollouts sampled from its own policy. The teacher receives the prompt together with a type-conditional privileged context: I h activates refusal behavior on harmful prompts, whereas I b*

## 1. 相关工作与进展
安全对齐"safety tax"[Huang 2025]:提升拒答鲁棒性常牺牲推理。既有方法多从数据侧改监督信号:SafeChain(外部教师 DeepSeek-R1-Distill-Llama-70B 蒸 40k CoT 安全轨迹)、STAR-1(1k policy-guided 轨迹 + LLM-as-judge 过滤)、SafeKey(aha-moment 前加 dual-path 安全头)、SafePath(注入短安全 primer 锚定 comply-or-refuse)、ThinkSafe [Lee 2026](去外部教师,用 refusal-steering prompt 自蒸馏 in-distribution 安全数据,并指出稠密 token 监督优于稀疏 GRPO 奖励)。本文建立在 OPSD [Zhao 2026] 之上,研究"how(off-policy SFT vs on-policy)"而非"what"。

## 2. 现有工作存在的问题
即便用 in-distribution 自蒸馏数据,SFT 仍是 off-policy:监督施加于固定示范而非模型自采样轨迹。作者主张这是 safety tax 的第二来源(第一来源是数据分布不匹配)。安全决策集中在早期窗口(前 ~10 token,30 后衰减)与少数 safety-critical token(compliance openers: Here/Sure/Certainly;结构标记 Title/**),而 SFT 对所有位置/token 一刀切求和,稀释了真正决定 comply-or-refuse 的信号(Fig.2 token-level KL 分析)。

## 3. Motivation
若部分 safety tax 来自 off-policy 监督,则把"序列级模仿"换成"on-policy 逐 token 蒸馏"应能在不付出同等推理代价下提升安全。需把监督集中到 refusal-decision 窗口、并施加在学生自采样轨迹上。

## 4. 主要灵感 / 核心直觉
安全本质是"在模型自身生成轨迹上的局部纠正",而非整序列均匀模仿;且基座 LLM 已有部分潜在拒答能力,安全更多是"激活潜在安全推理"而非教新能力。on-policy 逐 token KL 天然把更新集中到 teacher 与 student 发散的少数关键 token。

## 5. 主要解决思路(一段话讲清核心)
单一目标、type-conditional teacher:对 benign prompt x_b 给指令 I_b("此 prompt 安全,正常帮助、勿拒答")防过度拒答;对 harmful prompt x_h 给 I_h("此 prompt 有害,只能拒答")。student(trainable)在线 rollout,\(D_{\mathrm{KL}}(p_T \| p_S)\)。学生训练/推理都不带 c*,使安全行为内化进参数。

## 6. 方法详解(通俗、分步骤)

- **散度**:\(\alpha=0.5\),沿用 nemo-rl on-policy 蒸馏的默认(注:非纯 forward KL)。
- **TFR 选上下文**:直接估 Δsafety(Eq.3)不实际(依赖采样轨迹),改用 teacher flip rate(Eq.4)——frozen teacher 在加 c* 后把贪婪解从 unsafe 翻成 safe 的比例;\(c^* = \arg\max \mathrm{TFR}\)。候选池 C=K=30 个 refusal-steering 上下文,由 GPT-5.5 沿五轴(strength/length/framing/specificity/style)生成。
- **验证 TFR 有效**:跨 3 模型、3 上下文(flip rate 9%→78%),\(\rho = -1.00\),且不伴随过度拒答上升。
- 训练只用 prompt + 类型标签(无需预生成响应),按 ThinkSafe 设置:AdamW、lr 1e-5、cosine+10% warmup、batch 64、3 epochs、全参数 FT;≤1.7B 用 2×A100,8B 用 4×A100(FSDP)。
- 公式与 OPSD/COPSD 同构,创新在"按 prompt 类型条件化 + TFR 选上下文",属增量贡献。

## 7. 实验数据集
两 reasoning-model 家族、五个规模配置:Qwen3(0.6B/1.7B/8B)+ DeepSeek-R1-Distill(1.5B/8B)。Prompt 全取自 SafeChain 数据集(harmful Dh + benign Db)。评测三轴:Harmfulness↓(HarmBench/StrongReject/WildJailbreak,Llama-Guard 判)、Over-refusal↓(XSTest safe 子集 / WildBenign,WildGuard 判)、Reasoning↑(GSM8K/MATH500/GPQA 数学QA + HumanEval/MBPP 代码)。复合安全分 S=1−(5 项率均值)。自适应越狱:HarmBench 4 攻击族(HumanJailbreaks/Prefilling/PAP-top5/PAIR,159 behaviors)。

## 8. 实验结果与主要发现
排序 SafeChain < ThinkSafe < OPSA 在复合安全与推理上均成立(Table 1)。OPSA 相对 ThinkSafe(off-policy NLL):五配置平均 **安全 +4.00、推理 +3.04**;小模型增益最大——R1-Distill-1.5B 复合安全 +8.85、Qwen3-0.6B +5.49。OPSA 同时降 harmfulness 与 over-refusal(尤其自然分布 benign)。自适应越狱:在 Prefilling(攻击早期 token,正中机制)上增益最清晰——Qwen3-1.7B/8B 的 mean ASR 与 pass@N 双双归零;但 20 个 model×attack 格中 OPSA 仅 13/20(mean ASR)、14/20(pass@N)优于 ThinkSafe,PAIR(迭代攻击者-目标-裁判搜索)最难、多处回退。Token 级分析:更新集中在早期 compliance-decision token 附近(较有说服力的机制证据)。

## 9. 结果如何支撑其主张
"off-policy 是第二来源"主张靠对照实验隔离:ThinkSafe 与 OPSA 同用 SafeChain 自生成数据、仅训练目标不同(NLL vs 逐 token KL),OPSA 全面更优 → 差距归因于 on-policy/逐 token。机制由 Fig.2(KL 集中早期+词汇)与 Prefilling 攻击上的最大增益闭环呼应。TFR 作为选择准则由 Fig.3 单调关系支撑。

## 10. 逻辑自洽性(中性评估)
逻辑链(off-policy mismatch → 早期窗口失控 → on-policy 逐 token 纠正 → Prefilling 鲁棒)自洽且有 token 级与攻击级双重佐证。但安全无"ground-truth 特权信号",TFR 只是 unsafe→safe 翻转的代理,且 c* 由 GPT-5.5 生成、Llama-Guard 既做数据过滤又做评测(循环依赖,作者已在 Limitations 承认)。OPSA 相对 OPSD 是领域迁移+一个选择准则,创新增量有限。

## 11. 残留问题 / 局限
作者自陈:① 依赖基座保留可被 c* 激活的潜在安全能力,后训练若已严重覆写则增益小;② teacher 固定 frozen base,周期刷新为学生滑动平均是未来方向;③ harmfulness 度量依赖 Llama-Guard,结论相对该分类器、可能继承其失效模式。外部:对全自适应搜索(PAIR)鲁棒性仍是短板;只在 reasoning model 上验证。

## 12. 开源代码与框架(链接+框架+代码可得性)
github.com/FYYFU/OPSA(Apache-2.0,~98% Python,已 clone)。核心是 NVIDIA **NeMo-RL** 的扩展 fork(`nemo_rl/`);含 `train_opsd.sh`(on-policy self-distillation)、`train_sft.sh`(baseline)、`evaluation/`(HarmBench/XSTest/WildJailbreak 等)、`docker/`、`uv.lock`;`pyproject.toml`。全参数 FT、uv 安装,支持 Qwen3、R1-Distill。

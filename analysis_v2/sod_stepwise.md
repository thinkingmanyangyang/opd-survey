sod_stepwise | SOD: Step-wise On-policy Distillation for Small Language Model Agents | 浙大 + 腾讯 LLM 部 + 中科大 + 新加坡国立(Qiyong Zhong、Mao Zheng、Mingyang Song 共同一作;Junfeng Fang、Houcheng Jiang 通讯) | 2026-05-08 arXiv v1(2605.07725)·Preprint | 主题线 L1(OPD/自蒸馏)+L4(Agent/工具/多轮)·相关性 高

**原始论文**:https://arxiv.org/abs/2605.07725

## 一眼看懂
- 🟦 TL;DR:把 OPD(on-policy distillation,teacher 在 student 自己生成的轨迹上给 dense token 级监督)用到**小模型 agent 的工具集成推理(TIR)**会训练崩溃——因为小模型工具用得差,一次错误工具调用注入错误观测,后续推理在被污染状态上展开,**师生发散"加速"漂移**,teacher 在这些 OOD 状态上的监督变得不可靠甚至误导。SOD 按 **step 级师生发散自适应重加权**蒸馏强度(高发散区衰减、重新对齐时回升),在对齐区保留 dense 监督。0.6B/1.7B 学生相对最强基线 OPD 平均 +20.86%/+18.50%;0.6B 在 AIME2025 达 26.13%(avg@32)【原文 abstract+§5】。
- 最巧的一步:**用相邻 step 的发散比(而非绝对发散值)累乘成权重**(Eq.7)。抽掉它(改用绝对发散或均匀权重)就垮:① 比值累乘让"发散单调上升→权重<1 自动压低被污染信号";② 一旦师生重新对齐(d 下降)则比值>1、蒸馏强度回升(论文称 **recovery from earlier errors**);③ 只依赖比值意味着任何与真实师生发散 ∆k 单调一致的可观测代理都可用(Appendix D.5),理论上还能把加权二阶矩压到 O((d1/dk)²) 恢复梯度 SNR(Prop.2/D.4)。

## 为什么做
- 研究背景:agentic 能力多依赖大模型、推理成本高,迁移到可端侧部署的 SLM 有实践价值;但让 SLM 获得稳定有效的 TIR 仍难。TIR 后训练主流基于 RL(GRPO),只给稀疏 outcome 级 reward。OPD 提供 dense token 级监督,缓解 credit assignment、提样本效率与稳定性,是把大模型 agentic 能力蒸到小模型的自然候选【原文§1+§3.3】。
- 解决的具体痛点:小模型容量有限、探索弱,稀疏 outcome 监督加剧探索失败、陷 cold-start;但直接把 OPD 用于 SLM-TIR 会严重训练不稳/崩溃【原文§1+§4.1】。
- 相关工作 & 各自不足:RL/GRPO(稀疏奖励,SLM 上不稳)、vanilla OPD(假设 teacher 监督在所有 student-visited 状态都可靠——在 TIR 因工具引入的"非连续状态跳变"被严重违反)。训练数据/框架沿用 Yu et al.[52]("Demystifying RL in agentic reasoning",对应仓库 recipe/demystify/)与 ReTool【原文§3.3+§4.1+元信息】。
- 动机链:OPD dense 监督本是好东西 → 但 TIR 的发散不同于文本推理的"渐进漂移",是工具错误触发的"加速"漂移(Prop.1:单次错误工具观测使 ∆k 跳变 Ω(m·η_tool),连续 j 次错误超线性复合)→ 在低重叠 OOD 状态上 OPD 梯度被高方差无信息项主导、SNR→0(Prop.2)→ vanilla OPD 均匀聚合 = 系统性高估被污染信号 → 必须按 step 级发散自适应调蒸馏强度【原文§4.1】。
- 与最近邻工作(vanilla OPD)的Δ:vanilla OPD 对所有 step 均匀施加 token 级反向 KL 蒸馏(Eq.4);SOD 的Δ=**在每个 step 上乘一个由相邻发散比累乘得到的可靠性权重 wk**(Eq.7/9),高发散区衰减、对齐区保留、重对齐时回升。为什么有用:把"teacher 信号何时可信"做成可观测、零额外成本(复用 OPD 前向已有的师生 logprob)、且有方差压制理论保证的连续调节【原文§4.2-4.3+D.4】。

## 怎么做 + 靠不靠谱
- 方法流水线:输入 prompt → ① student πθ 在 SandBoxFusion(python 解释器)环境内自生成多步 TIR 轨迹(最多 16 轮工具调用)→ ② 按工具观测把轨迹切成 K+1 个 reasoning step(每 step=两次工具观测之间的模型响应,工具观测 token 排除)→ ③ 每 step 算发散分数 dk = mean(|log πθ − log πteacher|)(Eq.6,teacher 监督可靠性的可观测代理)→ ④ 算权重:w1=1;k≥2 时 wk=min(∏_{u=1}^{k−1}(du+ε)/(du+1+ε), 1+δ)(Eq.7)→ ⑤ wk broadcast 到该 step 全部 token 的 OPD 损失(Eq.9)→ ⑥ 联合目标 L=L_GRPO+L_step_OPD(Eq.10)反向 → 输出蒸馏后的 SLM【原文§4.2-4.3】。
- 逐组件必要性(有消融 Table 2):
  - **Step-wise OPD 重加权**:核心;消融 "w/o Step-wise OPD"(去掉 L_step_OPD)与"去掉上界裁剪 δ"都会退化——证重加权必要【原文§5 ablation,§4.2】。
  - **L_GRPO 项**:消融 "w/o GRPO"(去掉 L_GRPO)亦退化,说明稀疏 outcome 奖励与 dense 蒸馏互补【原文 Eq.10 ablation】。
  - **发散比 vs 绝对发散**:Eq.7 用比值而非绝对值——理论上(D.5)任何与 ∆k 单调一致的代理都可用,鲁棒性来源。
  - **上界 1+δ(δ=0.2)**:防 recovery 时蒸馏强度暴涨,保证稳定优化【原文§4.2 末】。
- 关键机制/公式(直觉):∆k(Eq.5)是 step k 上师生真实 KL 不匹配,但贵;dk(Eq.6)是其廉价代理(用 OPD 前向已有的师生 logprob 之差的绝对值均值)。权重 wk(Eq.7)是从 step 2 到 k 各相邻"前一步发散/后一步发散"比值的累乘:发散一路升→每个比值<1→累乘越来越小→被污染区监督被压;某步重新对齐(du+1<du)→该比值>1→权重回升(recovery)。直觉:teacher 在"学生走偏后"说的话越来越不可信,就越少听;学生绕回正轨了就重新多听。
- 实验与证据:
  - 数据集/设置:训练数据沿用 Yu et al.[52]——3k 高质量多轮推理 SFT 语料(s1-1k + LeetCode 1k + ReTool 1k,后两者 ReasonFlux-PRM 打分各取 top-1k)+ ~30k RL 数据(DAPO-Math 17k + Skywork-OR1 Math 4902/Code 3586 + MegaScience 3k);SFT 轨迹由 Qwen3-Coder-30B-A3B 在 SandBoxFusion 内端到端交互生成。评测均报 average@32(temp=1.0、top_p=0.6、每题 32 采样,5 个随机种子):Math=AIME2024/2025、Science=GPQA-Diamond、Code=LiveCodeBench(v6)。teacher=Qwen3-4B 经 GRPO 进一步优化;student=Qwen3-0.6B/1.7B【原文§5.1+Table 1】。
  - 关键数字【原文 abstract+§5,本轮 PDF 直读确认】:SOD 四任务均最高分,相对最强基线 OPD 平均 +20.86%(0.6B)/+18.50%(1.7B);0.6B 在 AIME2025 达 26.13% avg@32(称首个达该水平的 sub-billion 模型)。开销:dk/wk 仅 O(K) 标量运算,显存差 <0.5GB;0.6B 上 SOD 反比 OPD 快 3.5%(1052.3s vs 1090.5s,因抑制错误学习、失败重试更少)。
  - baseline 公平吗:基线含 SFT、GRPO 及多种蒸馏方法(共 6 个),统一 Qwen3 族、同训练数据、5 种子重复,公平;主卖点是"对最强基线 OPD 的相对平均提升"。
  - 看着强但没回答核心:相对增益口径(+20.86%/+18.50% 是相对百分比而非绝对点数;0.6B 整体绝对值约 21→24+ 区间,绝对提升较小,相对数放大观感)——这是需结合绝对值理解的口径问题。
- 假设与失效边界:
  - 显式【原文】:dk 作为 ∆k 的代理,前提是二者单调一致(D.5 给证明);Prop.1/2 刻画 TIR 的发散加速与 SNR 退化(Appendix D 证明)。
  - 隐式【推断】:仅 Qwen3 单一模型族、python 解释器(SandBoxFusion)单一工具环境验证(作者把 web/API 等其他 agent 设置与其他模型族列为局限);dk 依赖师生 logprob 差,teacher 自身在 OOD 高熵状态的 logprob 可靠性边界未充分刻画(高熵区 logprob 噪声大可能反噬 dk 估计)。
- 祛魅总结【推断】:真贡献=精准诊断"OPD 在 SLM-TIR 上因工具错误触发加速漂移而崩溃"(Fig.1 发散/熵证据 + Prop.1/2 理论)+ 用相邻发散比累乘权重做 step 级自适应重加权(含上界、recovery、零额外成本、方差压制证明)+ 完整开源含核心算法。包装/高估:相对增益口径放大观感(绝对点数提升较小);单模型族单工具环境,外推待验。低估:其"step 级可靠性代理 + 比值累乘"机制本身的通用性(任何"师生信号随状态漂移而失真"的 dense 蒸馏场景都可借),论文聚焦 SLM-TIR 反而限定了它的卖点。

## 结构化抽取
- 🎯 机制速览6轴:
  - **学什么信号**:teacher 在 student 自生成轨迹上的 token 级分布(反向 KL 蒸馏,Eq.4)+ 稀疏 outcome 奖励(GRPO)+ step 级师生发散 dk(调权信号)。
  - **改什么**:student SLM(Qwen3-0.6B/1.7B)全参数。
  - **何时改**:on-policy 训练全程;每个 reasoning step 按 dk 累乘比动态调该 step 蒸馏权重。
  - **免梯度?**:否,梯度训练(L=L_GRPO+L_step_OPD 联合反向);dk/wk 计算免梯度(O(K) 标量,复用前向 logprob)。
  - **记忆-技能生命周期**:无显式记忆/技能库;agentic 技能固化进 student 参数;teacher 提供"如何正确用工具/推理"的 dense 示范。
  - **防遗忘机制**:无跨任务防遗忘;但 step 级重加权可视为"防止学生把 teacher 在被污染状态上的错误监督学进去"(防 negative transfer/防错误级联固化),与 recovery 机制配合保留对齐区的正确监督。
- ⑦ 开源代码+框架/harness:https://github.com/YoungZ365/SOD (v1 验证已克隆约 23MB)。框架=**veRL fork + Open-AgentRL[52](recipe/demystify/,复用其 sandbox_fusion 工具配置)+ ReTool(recipe/retool/)**;rollout 走 vLLM(TP=4),SandBoxFusion 作 python 解释器、最多 16 轮工具调用。
  - **〔核心算法确已开源,承 v1 更正〕**:`verl/trainer/ppo/ray_trainer.py:363 compute_stepwise_opd_weights` 完整实现 Eq.6/7(按 response_mask 提取 step 边界、dk=mean|log πθ−log πteacher|、w1=1、wk=min(∏(du+ε)/(du+1+ε),1+δ)、broadcast 到该 step 全 token);`ray_trainer.py:878 _apply_token_kl_regularizer` 将其乘到 OPD 优势项 `weighted_opd = opd_coef·stepwise_weights·raw_local_adv`。配置 `verl/trainer/config/algorithm.py` 的 `TokenKLRegConfig`(stepwise_enable/epsilon/delta/opd_coef),运行脚本 `examples/SOD/run_sod.sh`。代码与论文 Eq.7 精确一致,可复现。
  - 〔次要差异,承 v1〕config dataclass 默认 `stepwise_delta=0.5`,但 run_sod.sh 覆写为 0.2(论文值);复现须用脚本而非 dataclass 默认。本轮 PDF 确认论文侧 δ=0.2、ε=1e-6(§4.2)。
- 💰 资源/成本与可扩展性:单节点 8×H20(96GB);0.6B/1.7B 1 epoch 约 2-3 天,4B/14B teacher 约 5-6 天;统一超参(AdamW lr=1e-6、batch 64、mini-batch 16、prompt≤2560、response≤20480、训练每 prompt 采 16、验证 32);所有实验 5 种子重复。SOD 相对 OPD 几乎零额外开销(dk/wk 仅 O(K) 标量,显存差 <0.5GB,0.6B 上反快 3.5%)。
- 🎯 对"探索-巩固"对标:**最强支撑/直接对标(可借核心组件)**。一句判定:SOD 是本轮 6 篇里与本项目"探索-巩固/路径恢复"**最直接同构**的工作——它显式建模"学生走偏(工具错误致状态漂移)→ teacher 监督在偏离区失真 → 应衰减;学生绕回正轨(du+1<du,论文原话 recovery from earlier errors)→ 应回升蒸馏强度",这正是"巩固/回轨:走偏后自选恢复分支并固化"的 loss 级实现;且它就是 on-policy(学生自生成轨迹)+ teacher 当 dense 监督。可借组件:① step 级师生发散 dk 作为"该不该听 teacher"的可观测探针(与本项目 MTP foresight probe 思路互通——都在找"信号失真/路径分叉"的点);② 相邻发散比累乘 + 上界,把"高发散区衰减、重对齐回升"做成连续可微调度;③ 零额外成本(复用前向 logprob)。竞品/差距:SOD 是"被动衰减误导监督"而非"主动让 student 自选恢复分支"——本项目 idea 强调 student **自己选**走得通的恢复路径并由 teacher 稀疏接管,SOD 没有"teacher 在关键步单点接管/给出恢复方向"的脚手架,只是降权;也无 MTP 前瞻。依据:§4.2 recovery from earlier errors + Eq.6/7 + on-policy 设定。
- 🔭 开放问题/未来方向:【原文】扩展到 web/API 等其他 agent 工具环境与其他模型族(作者列为局限);更精细刻画 teacher 在 OOD 状态的可靠性边界(隐含)。【推断】把"被动降权被污染监督"升级为"teacher 在高发散关键步主动给出恢复分支/单点接管"(对接本项目 path-recovery 脚手架);用 MTP 前瞻提前预测"下一步会不会走偏",在错误提交前就调权(把 SOD 的事后发散检测变成前瞻);把 dk 代理与高熵/低置信的关键步切分结合,统一"在哪切、怎么调权"。

RETURN: sod_stepwise | 读到PDF?是(32页/103612字,abstract+§4.1 Prop1/2+§4.2-4.3 Eq.5-10+§5 主结果/消融 全直读;+20.86%/+18.50%/26.13%/δ=0.2/recovery from earlier errors 均PDF核实) | L1(OPD/自蒸馏)+L4(Agent/工具多轮) | 对标=最强支撑/直接对标,显式建模"走偏降权+重对齐回升(recovery)"=path-recovery的loss级实现,可借step级发散探针+比值累乘调度+零成本;差距在无teacher单点接管/无student自选恢复分支/无MTP前瞻 | 残留待核0

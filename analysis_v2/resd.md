resd | Learning with Rare Success but Rich Feedback via Reflection-Enhanced Self-Distillation (RESD) | UC San Diego + Amazon + Georgia Tech(Yuwei Zhang 一作@实习;Sha Li/Changlong Yu/Qin Lu 等 Amazon;Jingbo Shang 通讯) | 2026-05-12 v1 · arXiv preprint(arXiv:2605.12741v1)cs.LG | 主题线 L1(OPD/自蒸馏)+L4(Agent多轮自进化)+L5(记忆/技能库)·相关性 高

**原始论文**:https://arxiv.org/abs/2605.12741

## 一眼看懂

> 一句话导读:模型自己当老师做蒸馏,但在"几乎没成功轨迹"时会失灵;RESD 的招是把失败反馈主动加工成"反思"、并沉淀进一个跨步骤的经验手册(playbook),让老师即使没见过成功示范也能给出可用的逐 token 监督。

- 🟦 TL;DR:先讲背景——在线策略自蒸馏(self-distillation,即让模型自己当老师)在 rare-success(几乎没成功轨迹)时会失灵。
  - 失灵的原因:现有方法(SDPO/OPSD)把环境反馈只当成**被动条件**喂给 teacher,且强依赖同一个 rollout batch 里偶然冒出来的"成功 peer 示范(同组里别的成功样本)"。
  - RESD 的做法是把失败反馈**主动加工**:
    - 对每条失败轨迹生成一段"反思"——诊断哪一步错了、本该怎么改;
    - 并把反复出现的教训沉淀进一个跨步骤、持久的 **playbook(经验手册)**。
  - 这些被加工/积累出来的上下文喂给 self-teacher,使它即便没有任何成功示范,也能给出可执行的 **token 级监督**。
  - 结果:每个 prompt 只采 1 条轨迹(N=1),早期就比用 8 倍采样(group size 8)的 GRPO 涨得快。【原文 Abstract 行 18-34,§1,§3.3,Figure 1/6】
- 最巧的一步:**retrospective reflection(回顾式的失败诊断反思)**。
  - 证据:消融(Table 3)显示,去掉 reflection 后,MANUFACTORIA-HAS 上的 per-test-case m@4 从 **76.95 暴跌到 51.41**(最大跌幅)。
  - 它的作用:把"原始失败信号(只告诉你失败了)"转换成"指令性纠错(说清哪步错、该怎么改)"——这是关键转换。没有它,playbook 里就没有高质量教训可沉淀,teacher 又退回到"被动喂反馈"的老路。【原文 §4.5 Table 3 行 750-781】

## 为什么做

> 一句话导读:自蒸馏的成败完全取决于"老师上下文里有什么";SDPO 那套上下文在 N=1、没成功示范时就废了,所以 RESD 把"怎么表征和积累失败反馈"单独提成一条设计轴。

- 研究背景:让 LLM 通过与环境交互**持续自我改进**是后训练的核心挑战。这条线的演进:
  - RL(PPO/GRPO)在多步任务上受困于**稀疏奖励**——成功轨迹极罕见时,缺少稠密监督去引导庞大的探索空间;
  - OPD 用**更强/有特权的 teacher** 给每个 token 算目标概率,把稀疏的轨迹级结果变成稠密的 token 级信号;但要维护一个单独的 teacher(算力贵 + 师生分布失配);
  - 自蒸馏变体(SDPO/OPSD)进一步用模型**自身**(current 或 moving-average 权重 θ_old)当 teacher、并条件于特权信息(环境反馈/参考解),从而同时消除外部 oracle 成本与师生失配。【原文 §1 行 49-75】
- 解决的具体痛点:自蒸馏的成败**完全系于 self-teacher 的监督质量**。作者做消融(§3.1,Figure 3)发现:
  - SDPO 的 self-teacher 特权上下文含两个互补信号=**环境反馈** + **同一 rollout batch 内的成功 peer 示范**;
  - 当 rollout batch size **N=1**(没有同组的成功 peer)时,环境反馈成了唯一监督源,SDPO 性能**大幅退化**——说明 teacher 很难仅凭原始失败反馈就构造出有效的纠错分布;
  - 在 rare-success(每次 rollout 成功率极低)下尤其严重,因为多数组里只有失败、没有正反例。【原文 §3.1 行 195-204】
- 相关工作 & 各自不足(§5,精确到机制):
  - **① 经典 KD**(Hinton 等[8,15,21,9]):学生匹配 teacher 的输出分布/表征,但是**固定数据 + 师生失配**(学生却是在自身生成的轨迹上被评估)。
  - **② on-policy distillation**(GKD/Qwen3/MiMo[12,23,25,5]):学生学的是 teacher 对**自身生成**的反馈,对齐了训练-推理分布、近似 online imitation;**但仍是 instance 级的孤立纠错**。
  - **③ self-distillation**(SDPO[6]、OPSD[30]、[27,2]):self-teacher 条件于环境反馈/执行信号或参考解/hint;**但把反馈当成静态被动条件、依赖成功示范**,且"局限于孤立的 instance 级纠错"(行 857-859)。
  - **④ 训练-free 持续学习**:Reflexion[19,13]（把言语反馈当 episodic memory,情景记忆）、self-discover[32,22]、ACE/演化经验池[29,28]（动态注入精炼策略）——**但与 on-policy 蒸馏脱节**(纯推理期、不更新参数)。
  - **RESD 的差异**:把"**跨样本反思 + 经验积累**"**直接整合进 on-policy 自蒸馏的回路**(既改 context、又更新参数),并把"怎么表征/积累反馈"提为一条**独立的设计轴**,而不是提一个新损失。【原文 §1 贡献 行 102-108,§5 行 858-865】
- 动机链(为什么这样设计):
  - RL 稀疏奖励;
  - → OPD 给稠密 token 信号;
  - → 但需要外部 teacher(贵 + 失配);
  - → SDPO 用自 teacher 解决;
  - → 但 rare-success 时没有成功 peer,teacher 只能看到原始失败;
  - → **原始失败是"诊断性"的、不是"指令性"的**;
  - → 所以必须把失败**主动解释成因果 + 跨轨迹沉淀**,才能当监督用。
- 与最近邻工作的Δ:
  - vs **SDPO**(本文直接基线)——SDPO 的 teacher 特权上下文 = 环境反馈 + 同组成功 peer 示范;RESD 把"被动喂反馈"换成 **reflection(主动诊断)+ playbook(持久经验)+ 可选 solution buffer**,使 **N=1 也能学**。
  - vs **ACE**(playbook 思想的来源)——ACE 是训练-free 的上下文演化;RESD 则把这套经验沉淀**接到了 token 级蒸馏的 teacher 上下文里**。
  - 关键点:**把"怎么表征/积累反馈"提为 on-policy 自蒸馏的一条独立设计轴**。

## 怎么做 + 靠不靠谱

> 一句话导读:核心目标是一个带 clip 的 f-散度自蒸馏(学生去匹配老师在"反思+经验"上下文下给的分布);流程上每步都是"先更新 context(反思→playbook→缓存解),再更新参数",老师靠 EMA 跟着学生走。

- 基础自蒸馏目标(§1 末,Eq.1):
  - self-teacher θ_old 条件于中间状态 \(s_t=(x_t,y_{<t})\) 与回顾反馈 \(c(x,y)\),产出一个纠正后的、feedback-informed(吸收了反馈的)分布 \(\pi_{\theta_{\text{old}}}(\cdot\mid x,y_{<t},c)\);学生 \(\pi_\theta\) 在每一步去匹配它。
  - 定义 token 似然比 \(\tau_v^t=\dfrac{\pi_{\theta_{\text{old}}}(v\mid x,y_{<t},c)}{\pi_\theta(v\mid x,y_{<t})}\),并 clip 到有界区间 \(\tilde\tau_v^t=\operatorname{clip}(\tau_v^t,\epsilon_{\min},\epsilon_{\max})\) 来防爆;统一用 f-散度:
  - \(\displaystyle L_{\text{SD}}(\theta)=\mathbb{E}_{x,y\sim\pi_\theta,\,f}\left[\sum_{t=1}^{T}\sum_{v\in V}\pi_\theta(v\mid x,y_{<t})\cdot f(\tilde\tau_v^t)\right]\)
  - 选不同的凸函数 f,可恢复出 forward-KL / reverse-KL / JSD(各有不同的优化性质,Appendix A)。因为轨迹采自 base policy \(\pi_\theta\),这个目标**天然是 on-policy** 的——能 "recover forgotten behaviors after midtraining(在 midtraining 之后找回被遗忘的行为)" 并防 exposure bias。
- 方法流水线(Algorithm 1,讲清输入→输出):
  - **Require(初始化)**:学生 \(\pi_\theta\)、teacher 权重 θ_old、playbook \(P\leftarrow\emptyset\)、solution buffer \(B\leftarrow\emptyset\)。
  - **每步 t 的流程**:
    - ① 学生采 1 条 rollout \(y\sim\pi_\theta(\cdot\mid x)\),拿到环境反馈 \(c(x,y)\) 与 reward \(R(x,y)\);
    - ② **CONCISE** 先给 playbook 剪枝(删掉已净有害/过期的条目);
    - ③ 分两种情况:
      - **若失败**(\(R<\tau\)):**REFLECT** 生成反思 \(r\)、并给现有 playbook 条目打 helpful/harmful 标签;**CURATE** 从反思里提炼出新的、非冗余的条目加进 playbook(\(P_{t+1}=P_t\cup\text{CURATE}(P_t,r;\theta)\));
      - **若成功**:把解存进 solution buffer \(B(x)\leftarrow y\),且 \(r=\emptyset\);
    - ④ teacher 用 **EMA** 同步学生的权重(rate 0.0001);
    - ⑤ 构造一个 **enriched(增强)上下文**——把 playbook \(P_{t+1}\) + 反思 \(r\) + 上次轨迹 \(y\) + 反馈 \(c\) + 缓存解 \(B(x)\) 一起喂给 teacher,得到 token 级监督 \(\pi_{\theta_{\text{old}}}(\cdot\mid x,y_{<t},c,r,P_{t+1},B(x))\);
    - ⑥ 学生最小化 \(L_{\text{SD}}\)(Eq.1),并按 batch 成功率加 per-sample 权重(Eq.2)。
  - **关键设计**:因为 \(P,B\) 是**跨步持久**的,即便当前 batch 里没有任何成功示范,teacher 可用的监督也能**随时间改善**(行 405-407)——这就是 N=1 也能学的根本。
  - **数据流动**:在线流式,每个样本至多见一次;每个 batch 内循环 **K=4** 次;**先更 context、再更参数**,teacher 用 EMA 同步。
- 逐组件必要性(消融 Table 3,per-test-case m@4 / b@4):
  - **reflection**:去掉 → MANUF. 76.95→**51.41**(最大跌幅);它负责短期的局部纠错。【行 750-752,776-777】
  - **playbook**:去掉 → MANUF. 76.95→**60.65**、BSIM-Easy 38.87→**32.95**;它负责跨步保存可复用的教训、**防回退**。【行 757-759】
  - **solution buffer**:去掉 → MANUF.→**60.24**、BSIM-Easy→**31.97**;它把"部分进展"转成"完整正确解"。【行 764-766】
  - **success-rate rebalancing(Eq.2)**:仅在 BSIM-Easy/FINER 上启用,用来缓解"batch 成功率极端时,多数派 outcome 主导梯度"的问题——属于工程组件,无独立消融。
  - **CONCISE 剪枝**:规则是 helpful/harmful 计数满足 \(d_j\ge h_j\)(即净有害)就删,且超过 budget \(M_{\max}\) 时逐出最久未标注的条目。作用是保证 playbook 紧凑、收敛到反复有用的条目;两种变体(prioritized/staleness)在 Appendix D。
  - **f-散度选择**(forward-KL / reverse-KL / JSD):Appendix A 给了直觉——reverse-KL 是 mode-seeking(寻峰),能抑制幻觉;JSD 在师生差距大/高方差时**有界、更稳**,所以作 robust default(稳健默认)。按任务选:
    - MANUF.=reverse-KL(α=1.0,且**例外地用上 clip,ε_max=5**);
    - BSIM-Med=reverse-KL(α=1.0,EMA rate 0.01);
    - BSIM-Easy/FINER=JSD(α=0.5)。【Appendix G 行 1324-1332】
- 关键机制/公式(直觉):
  - **REFLECT(Eq. 行 331)**:\(\{r_j\}=\text{REFLECT}(x,y,c,P_t;\theta)\)。即学生在诊断模式下输出反思——把失败连接到**具体的推理步骤**,并给现有 playbook 条目打 helpful/harmful 标签(这是个轻量的使用信号,供后续维护)。
  - **success-rate reweighting(Eq.2)**:设 batch 成功率 \(s=|\{i:R(x_i,y_i)\ge\tau\}|/B\),权重为 \(w_i=(1-s)^\alpha\)(若成功)、\(s^\beta\)(若失败),并归一化使 batch 均值=1(不改有效学习率)。直觉:当成功率极低或极高时,稀有的那一类(少数类)样本会被加权放大,防止多数类主导梯度。
  - **核心洞见(§4.3 Figure 5)**:RESD 的蒸馏 loss 反而**更高**——说明 reflection/playbook 增强后的 teacher 给出的分布**偏离学生当前行为更多**;而且 loss mass 集中在**"决策 token"**上(如 "create"、"solution"、"processes"、"The goal")。这些都是"选择下一步做什么/如何框定问题"的点,说明 teacher 纠正的是**推理策略**、而非表面 token。
- 实验与证据(数字已对 Table 2/3 + 附录逐项核对):
  - 数据=4 个任务(Table 1):
    - **MANUFACTORIA-HAS**:写 DSL 程序做模式匹配,742/132 题,用 Qwen3-4B-Thinking-2507;
    - **BOUNCINGSIM-Easy/Medium**:写 Python 模拟 2D 碰撞,640/100 与 320/100,用 4B / Qwen3-30B-A3B-Thinking;
    - **FINER**:SEC 文件的 XBRL 实体标注,1000/500,用 4B;
    - 前三个初始成功率近 0(是 rare-success 试验台,来自 RL-Grok benchmark),FINER 初始成功率较高(用来检验普适性);
    - 指标=mean@4/best@4 × Per-Task/Per-Test-Case Acc;checkpoint 用统一的 rank-based 规则选。【§4.1 行 443-471】
  - **关键数字(Table 2,Per-Task m@4/b@4)**:
    - MANUF.:SDPO+ss **0.57/2.27** → RESD **35.80/65.91**(巨大跃升;per-test-case 37.97/71.93→**76.95/96.53**);
    - BSIM-Easy:SDPO+ss 1.00/4.00 → RESD **4.25/8.00**;
    - BSIM-Med:2.00/6.00 → **7.25/10.00**;
    - FINER:50.25/59.12 → **53.66/61.12**(Per-Test-Case 76.15/82.21→78.98/84.32);
    - vs GRPO(Figure 6):RESD 用 N=1 早期就比 group-size-8 的 GRPO 涨得快(这是交互效率上的比较,不是等 rollout 预算)。【Table 2 行 503-576,§4.4】
  - **案例研究(Table 4,MANUF. "Accept if tape contains BRBR")**——4 个连续的 inner-loop step,逐步纠错:
    - step45 acc 0.00:引用了未声明的状态 → parse error;反思提炼出"所有分支目标都必须引用已声明的状态";
    - step46 acc 0.84:错误地 reject 了前导的非模式符号;playbook 记下"非模式符号应跳过、不要 reject";
    - step47 acc 0.96:过度纠正,导致空 tape 死循环;playbook 记下"非空时跳过、空 tape 在非终态应 reject";
    - step48 acc 1.00:缓存这个成功解;
    - 这个案例**展示了:失败反馈不是靠简单 append、而是靠组织成持久且 scoped(有作用域)的知识,才真正有用**(行 798-845)。
  - baseline 公平性:对 GRPO 明确声明这是"interaction-efficiency(交互效率)比较、而非等 rollout 预算"(GRPO 每 prompt 采 8 条,RESD 只采 1 条)——**作者诚实标注了不对等**;其余设置相同(lr 1e-6 / batch 32 / inner-loop K=4)。还**诚实报告了非单调现象**:FINER 在 step50 因 response 截断而掉点、BSIM-Easy 在 step60 后退化(进入高成功率区后,稠密自蒸馏不如奖励法稳)。【§4.4 行 722-728】
  - 看着强但没回答的:
    - ① 4 个任务规模偏小、且都是可执行反馈任务;
    - ② "决策 token 集中"是 case-level(个案级)观察,没有定量的 attribution;
    - ③ 高成功率区切换 GRPO 的混合策略只给了建议,未系统验证。
- 假设与失效边界:
  - 【原文】① 环境要提供 "rich execution feedback(丰富的执行反馈)"(代码可执行/测例反馈),哪怕奖励本身是稀疏的二元——方法吃的就是这个反馈来反思,**纯黑箱奖励下反思就无据可依**。
  - 【原文】② teacher=学生的 EMA,这默认了**学生本身有足够能力做出有意义的反思/诊断**(REFLECT/CURATE 都由 \(\pi_\theta\) 完成)。
  - 【原文】③ 进入高成功率区后,自蒸馏会变不稳;作者建议"RESD 先快速脱离 rare-success → 再切 GRPO"(§4.4)。
  - 【推断】④ playbook 是全局共享的自然语言条目;当任务跨度大、或条目相互冲突时,CONCISE 的 helpful/harmful 计数可能误删(依赖反思打标的质量)。
  - 【推断】⑤ 反思由(可能较弱的)学生生成;学生太弱时,"诊断错因"本身就不可靠——会放大错误的教训。
- 祛魅总结:
  - 真贡献=明确把"反馈的表征/积累"提为自蒸馏的一条独立设计轴,并给出"反思→playbook→token 级监督"的可插拔闭环,在 rare-success 下用 N=1 实现了样本高效的自举(MANUF. 0.57→35.80 极有说服力)。
  - 【推断】被高估的可能是"自蒸馏部分的新颖性"——损失/objective(Eq.1 的 f-散度自蒸馏)沿用了 SDPO,真正新的是 **context 工程**(reflection + playbook),而且 playbook 思想直接借自 ACE。
  - 【推断】被低估的是"防遗忘"证据(Appendix E:IFEval per-task m@4 训练后 82.50→83.75 基本不退,是个被埋在附录里的好结果)。
  - 总体看,它本质是"context-engineering ⊕ on-policy 蒸馏"的缝合,效力强烈依赖于"任务要有可解释的执行反馈"。

## 结构化抽取
- 🎯 机制速览6轴:
  - **学什么信号**=teacher(自身 EMA,条件于反思 + playbook + 反馈 + 缓存解)给的 token 级分布(**top-100 词表**),经 f-散度蒸馏;额外有 success-rate 重加权(Eq.2);
  - **改什么**=两轨更新——学生参数 θ(token 级)+ 持久的 playbook(自然语言记忆);
  - **何时改**=在线流式、每个 batch 内循环 K=4 次;**先更 context、再更参数**,teacher 用 EMA 同步(rate 0.0001);
  - **免梯度?**=否,核心是带梯度的 KL/f-散度蒸馏;但 reflection/playbook 的"学习"是免梯度的(纯文本积累);
  - **记忆-技能生命周期**=显式两类——playbook(经验条目,带 helpful/harmful 计数 \((h_j,d_j)\) + staleness,由 CONCISE 剪枝、budget \(M_{\max}\)≈120-200)+ solution buffer(缓存成功轨迹供 replay);
  - **防遗忘机制**=三条叠加——on-policy 自蒸馏天然抗遗忘(§1 称可 "recover forgotten behaviors after midtraining")+ playbook 持久保存教训防回退 + Appendix E 实测 IFEval 不退(82.50→83.75)。
- ⑦ 开源代码+框架/harness:GitHub(horizon-llm/RESD,v1 元信息已核:约 52MB,含 verl 源码 + selfevolve 模块 + docker + paper.pdf;本地 repos/resd 已克隆);框架 **veRL(volcengine/verl)+ SDPO(lasgroup/SDPO)**;rollout 用 **vLLM(TP=4)**,训练用 **FSDP**。【Appendix G 行 1303-1306】
- 💰 资源/成本与可扩展性:【原文】**单节点 8×H200,每个实验约 1 天 wall-clock**;每步延迟:SDPO ~**400s** < RESD(反思/curation 带来中等稳定开销)< GRPO >**1200s**(GRPO 因 group rollout 最慢)→ RESD 比 GRPO 省很多(Figure 8/Appendix F)。可扩展性卖点=**N=1 单 rollout**、交互预算受限场景。长上下文需求高(MANUF. prompt/resp 上限 49152/20480 token,BSIM 58368/25600)。【Appendix F/G 行 1290-1332】
- 🎯 对"探索-巩固"对标:**强支撑 + 高度同源,既是直接竞品、又有可借组件**。
  - 映射(为何同源):
    - RESD 的"失败诊断反思 + 自选纠错 + playbook 固化",几乎就是 TSRD"巩固/回轨(走偏后自选恢复分支并固化)"的一个完整实例化——失败轨迹 → 反思定位"哪一步走偏" → 生成纠错 → 沉淀进记忆且不遗忘;
    - "探索"侧:on-policy 自采(学生自己 rollout)对应"偏向自己能走通的开头";solution buffer 缓存成功开头供复用 ≈ "有效路径固化"。
  - **可借组件**:
    - ① playbook 作为"技能/经验库"的载体(带 helpful/harmful 生命周期管理 + CONCISE 剪枝),正好填上 TSRD 的 L5 记忆库缺口;
    - ② "反思把失败转成 token 级纠错监督"=path-recovery 的具体监督来源;
    - ③ reverse-KL/JSD 按师生差距来选散度的工程经验;
    - ④ §4.3 的"loss mass 集中在决策 token"提示:纠错信号天然落在关键决策点,与 TSRD 想在"高熵/关键步"接管的思路同向。
  - **缺口/差异**:
    - RESD 的"前瞻"靠的是反思的**事后诊断**,**没有 MTP 式的前瞻探针**(而 TSRD 想用 MTP 提前预判走偏点);
    - 且它的巩固走的是 EMA 自蒸馏的**全 token 稠密监督**,而非 TSRD 设想的"单点接管 / sparse_critical"稀疏脚手架——粒度比 TSRD 更密。
  - 判定:**最近邻强竞品**;idea 同构,但缺 MTP 前瞻 + 稀疏化。
- 🔭 开放问题/未来方向:【原文】① RESD 定位为"feedback-enhancement 可插拔模块",可接更强自蒸馏 objective(sample routing 决定何时蒸馏[CHORD]、reward-grounded objective 决定更新方向[self-distilled RLVR]),自身只负责提供结构化反馈上下文(§6);② 进入高成功率区后切换到奖励法(GRPO)更稳的混合策略(§4.4)。【推断】用 MTP/前瞻信号提前定位"将走偏"的决策 token 以替代事后反思、把 playbook 从全局共享升级为按任务/技能分桶检索、把稠密 KL 蒸馏稀疏化到 reflection 标出的关键决策 token(更贴 TSRD 脚手架)、研究学生太弱时反思不可靠的下界。

— RETURN —
resd | 读到PDF? 是(18页/56k字重抽清洗null字节后,§1-§6+Algorithm 1+Eq.1-2+Table 2/3/4+Appendix A/B/E/F/G全核,所有数字逐项核对一致) | L线 L1(OPD/自蒸馏)+L4(Agent自进化)+L5(记忆库) | 对标结论:最近邻强竞品+高度同源——playbook作L5技能/经验库载体(helpful/harmful生命周期+CONCISE剪枝)、"反思转token级纠错"=path-recovery监督来源、reverse-KL/JSD按师生差距选散度、loss集中决策token同向TSRD关键步接管;但前瞻靠事后反思(无MTP前瞻探针)、巩固走EMA全token稠密(非sparse_critical稀疏脚手架),idea同构但缺MTP前瞻+稀疏化 | 残留待核数:0(Table 2/3/4 + IFEval 82.50→83.75 + 延迟400s/1200s + 8×H200 + top-100 + 各任务散度/Mmax 全部逐项核对一致)

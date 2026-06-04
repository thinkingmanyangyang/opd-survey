ophsd | Training with Harnesses: On-Policy Harness Self-Distillation for Complex Reasoning | 北京大学 Peking University（Zhengyang Zhao、Lu Ma 共一;Wentao Zhang 通讯） | 2026-05-09·arXiv 预印本·v1 | 主题线 L1(OPD/自蒸馏)+L4(Agent/工具脚手架)·相关性 高

**原始论文**:https://arxiv.org/abs/2605.08741

## 一眼看懂
- 🟦 TL;DR:把"推理时脚手架(harness,即检索增强、plan-solve、draft-verify 等外部编排流程)"从永久固件改成临时训练支架——训练时让模型在 harness 里 rollout、用 harness 诱导的轨迹当 teacher 做自蒸馏(reverse-KL),把"过程性推理能力"永久写进基座参数;推理时撤掉 harness。数学/文本分类上超过 OPSD/GRPO,且再把 harness 接回去不再增益甚至降分(说明能力已内化)。【原文】abstract+§3.2
- 最巧的一步:**把特权上下文 \(X\) 从"静态变量"升级成"由学生参数 \(\theta\) 驱动的程序化 harness 工作流",而 logit 监督锚在 frozen base \(\tilde\theta\)**(Eq.3 的解耦)。抽掉这步——若像 OPSD 那样直接把参考解 \(y^\star\) 原文塞给 teacher,teacher 的优势就只是"信息性"(告诉好答案长啥样)而非"过程性"(教如何推导,原文 §3.2 原话 "informational rather than procedural … reveals what a good answer looks like, but not how it should be derived");正是"plan 经 planner 中介、学生匹配的是'已见 plan 的 solver'信号"这一招,强迫学生内化推导结构(case study idx649:OPHSD 自纠 ordered-pair 陷阱,OPSD 直接算错)。

## 为什么做
- 研究背景:LLM 不带 harness 时在"需证据累积/中间验证/长 horizon"的问题上脆弱(会忽略上下文、传播早期错误、不修正错误中间解);近期系统给 base 模型配推理时 harness(外部脚手架,控制信息如何被维护/检索/变换/呈现),实践中提鲁棒性,但增益来自外部流程而非模型本身。【原文】§1
- 解决的具体痛点:① model 与 harness 分离带来延迟、token 成本、工程复杂度、新失败模式;② 难判断"模型本身学到了什么"(harness 移除后能力即失);③ 后训练方法不能直接弥合——SFT 模仿静态示范不教自适应流程、RL 监督稀疏难定位关键过程行为、现有自蒸馏的特权上下文 \(X\) 都是静态变量(只告诉"好答案",不教"如何推导")。【原文】§1+§3.2
- 相关工作 & 各自不足(站谁肩上 + 精确差异):
  - **OPD 两支**——Reward-based(把 reverse-KL 当 policy-gradient reward,高方差、对 reward 设计敏感)与 Loss-based(直接还原可微 token-level 蒸馏损失,稠密低方差但需 teacher 分布、teacher 大时贵);OPHSD 走 loss-based、teacher=同模型 frozen 副本。
  - **自蒸馏家族(按特权上下文 \(X\) 分,Table 1)**——OPSD/SDFT 用 verified 参考解 \(y^\star\)、SDPO 用 rollout 中收集的环境反馈、CRISP 用静态"be concise"指令、OEL 用过往轨迹经验知识。OPHSD 的差异:\(X\) 不再是任何"静态变量",而是 harness 程序产出的动态终端上下文。
  - **LUPI(Learning Using Privileged Information,Vapnik)**——OPHSD 显式借其范式,把"训练期专属 oracle 信息"统一记为 \(z(x)\)(参考解 / 在线 memory bank 都是其实例)。
  - **Harness Engineering**——NLAHs(把 harness 行为外化为可编辑的自然语言规格、共享 runtime 执行)、Meta-Harness[Lee 2026]（在 harness 实现空间里搜索以提升下游 agent）、AutoHarness(harness 合成 + 模块化设计)。**精确差异**:这些方法优化"留在部署环里的外部系统",OPHSD 用 harness 产出监督、模型部署时**撤掉** harness——区别在"优化后留下的是什么:harness 本身 vs 用 harness 产出训练的模型"(§2.2 原话 "what persists after optimization")。【原文】§2.1+§2.2+Table 1
- 动机链:OPD 在学生自身轨迹上给稠密 token 监督=内化 harness 行为的天然载体 → 现有自蒸馏的 \(X\) 都是静态变量、只给信息性优势 → 而 harness 本质是"编排检索/中间调用/验证"的程序 → 所以要把特权上下文从静态变量泛化为"程序化、由学生参数驱动的工作流"。【原文】§3.2
- 与最近邻工作的Δ:vs OPSD/SDFT——把它们纳为"平凡静态 wrapper"特例(§3.2 原话 "trivial static wrapper"),改用动态 harness 给"过程性"信号;vs Harness Engineering——后者优化"留在部署环里的外部系统",OPHSD 模型部署时撤掉 harness。【原文】§2.2+§3.2

## 怎么做 + 靠不靠谱
- 方法流水线(输入→输出,Fig.1):① 取问题 \(x\) + 训练期专属 oracle 信息 \(z(x)\)(参考解 / 在线 memory bank)→ ② 学生 \(p_\theta(\cdot|x)\) 直接 rollout 一条 \(\hat y\)(不带 harness)→ ③ 同时用 harness \(H_\theta\)(由学生 \(\theta\) 驱动,确定性有状态程序,最多 \(T\) 次模型调用)把 \((x,z(x))\) 编排成终端上下文 \(C[H_\theta(x,z(x))]\) → ④ frozen base \(\tilde\theta\) 条件于该终端上下文,沿学生前缀 \(\hat y_{<n}\) 给 next-token 分布作 teacher → ⑤ reverse-KL(\(\mathrm{KL}(p_T\|p_S)\),Eq.3),\(C[\cdot]\) 当 stop-gradient target,梯度只回传学生 → 更新 \(\theta\)。推理时撤 harness。【原文】Eq.(1)(2)(3)+Fig.1
- 逐组件必要性(标"有无消融"):
  - **harness 驱动 \(\theta\) / 监督锚 \(\tilde\theta\) 的解耦**:无单独消融,但是设计核心(Eq.3 论证:轨迹质量随能力演进、监督信号锚在稳定先验)——结构必要。
  - **reverse-KL**:沿用 OPSD 设置(\(\mathrm{KL}(p_T\|p_S)\)),无对 forward/JSD 的额外消融(注:OPSD 主结论是 forward KL 更优,OPHSD 这里反取 reverse KL,但未在本文重做该消融)。【待核:OPHSD 取 reverse 而 OPSD 取 forward,原文未解释此切换原因】
  - **plan 经 planner 中介(vs OPSD 直给 \(y^\star\) verbatim)**:有 case study(idx649)+ 长度分析(Fig.5:OPSD 直给 \(y^\star\) 致 40 步后生成长度坍缩、随之 benchmark 退化;OPHSD 稳)——这是"过程性 vs 信息性"的关键对照,有据。
  - **draft-verify 两步检索(confirmers \(N^+\)/challengers \(N^-\))**:无单步消融,但 cite-rate 分析(Table 4)证学生内化了"案例比较"推理形状(OPHSD 末期 ≥90%,GRPO/OPSD <10%)。
  - **冷启动保护(bank<10 条退化为单次前向)、训练嵌入预算**:工程必要(防泄漏:harness baseline 评测时 bank 仅从测试流重填,§3.3 原文 "memory bank is reset and re-populated exclusively from the test stream")。
- 关键机制/公式(真实符号,从 PDF 抄准 + 直觉):
  - **OPD 基础目标**(Eq.1,本文站它肩上):
    \[ L_{\mathrm{OPD}}(\theta)=\mathbb{E}_{x\sim S}\,\mathbb{E}_{\hat y\sim p_S(\cdot|x)}\,\frac{1}{|\hat y|}\sum_{n=1}^{|\hat y|} D\!\Big(p_T(\cdot\mid x,\hat y_{<n})\,\big\|\,p_S(\cdot\mid x,\hat y_{<n})\Big), \]
    teacher 常由"同模型 frozen 副本 \(\tilde\theta\) + 特权上下文 \(X\)"构造:\(p_T(\cdot|x)\triangleq p_{\tilde\theta}(\cdot|x,X)\)。
  - **harness 诱导的条件分布**(Eq.2,把 harness 形式化为确定性有状态 LLM 程序):
    \[ H_\theta(y\mid x)=\sum_{s_{1:T},\,c_{1:T}}\Big[\prod_{t=1}^{T} p_\theta(c_t\mid s_{t-1},x)\,\tau(s_t\mid s_{t-1},c_t)\Big]\,\delta\big(y=\pi(s_T)\big), \]
    其中 \(c_t\) 是第 \(t\) 次模型调用、\(\tau\) 是确定性状态转移、\(\pi\) 是从终端状态读出答案的确定性读出函数;\(z(x)\) 通过注入初始状态得到 \(H_\theta(y\mid x,z(x))\)。
  - **OPHSD 训练目标**(Eq.3,核心——动态 harness 终端上下文当 teacher 前缀):
    \[ L_{\mathrm{OPHSD}}(\theta)=\mathbb{E}_{x\sim S}\,\mathbb{E}_{\hat y\sim p_\theta(\cdot|x)}\,\frac{1}{|\hat y|}\sum_{n=1}^{|\hat y|}\mathrm{KL}\!\Big(\underbrace{p_{\tilde\theta}\big(\cdot\mid C[H_\theta(x,z(x))],\hat y_{<n}\big)}_{p_T(\cdot|x,\hat y_{<n})}\;\Big\|\;\underbrace{p_\theta\big(\cdot\mid x,\hat y_{<n}\big)}_{p_S(\cdot|x,\hat y_{<n})}\Big), \]
    \(C[H_\theta(x,z(x))]\) 视为 **stop-gradient** target。直觉:harness 编排(由 \(\theta\) 驱动、随能力演进)与 logit 监督(锚在稳定 \(\tilde\theta\))解耦——轨迹越来越好,但监督信号不漂移。学生被迫"从裸输入 \(x\) 复现自己在 harness 里的增强行为",自然划出**可蒸馏边界**:结构性推理先验(分解/自验证)可内化进权重,真正实时外部访问(工具/检索内容)不可。
  - **两个 harness 实例的终端上下文**(决定 teacher 看到什么):
    - Plan–Solve(数学,\(z(x)=y^\star\)):planner 把 oracle 解蒸成策略草图 \(s\sim p_\theta(\cdot|x,y^\star)\),solver 据 \((x,s)\) 执行推导 \(y\sim p_\theta(\cdot|x,s)\);**终端上下文 \(C=(x,s)\)**——teacher 是"已看过 plan 的 solver"信号,而非 \(y^\star\) 原文。
    - Draft–Verify(分类,\(z(x)=M_{<x}=\{(x_i,y_i)\}_{i<t(x)}\) 在线 memory bank):draft 用 top-\(k\) 近邻 \(N_d(x)\) 生成 \(\hat y_d\);verify 用 \(\hat y_d\) 重检索 \(k^+\) 个 confirmers \(N^+\)(同标签)与 \(k^-\) 个 challengers \(N^-\)(异标签);**终端上下文 \(C=(x,\hat y_d,N^+(x),N^-(x))\)**。【原文】Eq.(2)(3)+§3.3
- 实验与证据:全部从 Qwen3-8B 起。文本分类两路独立训练(各采 10k,严格防污染):CAIL-2018→LawBench(215 类,F1);USPTO-50k→USPTO test(10 类,acc)。数学:DeepMath 采 10k→AIME24/AIME25/OlympiadBench(取10%)/HMMT25,报 pass@8(4 次平均)。关键数字:文本(Table 2)base→harness→GRPO→OPSD→OPHSD,LawBench F1 55.29→60.22→62.44→64.25→**69.51**;USPTO acc 30.07→79.02→90.01→88.01→**90.81**;内化后 OPHSD+Harness 反降(LawBench 68.41 −1.10、USPTO 83.62 −7.19,即把 harness 接回反而更差,证能力已内化)。数学(Table 5,pass@8)OPHSD avg **69.50**,超 OPSD 2.82、GRPO 2.93、CRISP 10.75;HMMT25 53.33 较 OPSD +10.83、较 GRPO +8.33。base+harness 较纯 base 平均 **+17.83**(Table 3,25.64→43.47)。内化分析:cite-rate(Table 4,GPT-4o 判)、按"base vs harness 能力差"分组(Fig.4:harness 才能解的子集 +84.62%、两边都不解的额外解出 +12.54%、原有能力 −1.90 几乎不损)、难度分层(Fig.6:Math hard +22.9、LawBench hard +33.8、USPTO hard +90.4)。baseline 公平性:同框架 verl、同 lr 1e-6/batch 64/max-gen 8192,GRPO group=8/KL=0,OPSD teacher=初始 policy——较公平。【原文】Table 2-5+Fig.4-6
- 假设与失效边界:
  - 【原文】Appendix C 三档可蒸馏性:Fully Distillable(过程先验如 plan-solve/分解/自验证)、Partially Distillable(依赖训练期专属动态内容如 memory bank——结构形状可迁移但具体检索内容不能内化,且依赖模型参数先验是否够支撑)、Non-Distillable(实时外部交互如 live web/代码执行/有状态环境——优势是信息性而非过程性,无法内化;"调用脚手架的程序结构"或可蒸馏但本文未验)。诊断启发式:推理时消融 \(z(x)\),若增益大部分保留则高度兼容。
  - 【原文】Limitations:仅验证两类代表性 harness,未做大规模 harness 设计扫描。
  - 【推断】"OPHSD>OPSD"的部分增益可能源于 OPSD 在该设置下生成长度坍缩(Fig.5,超参/早停敏感)而非纯方法优越;数学最佳分由"取全程最高评估分(4 次平均)"得到(§A.3),选择性报告需留意。全 Qwen3-8B 单一规模。
- 祛魅总结:【推断】真贡献=① 把"特权上下文=静态变量"升级为"程序化 harness 工作流",给出统一框架(OPSD/SDFT 是其平凡特例)+ 清晰的三档可蒸馏性边界(Appendix C,理论上很完整);② 用 cite-rate / 能力差分组 / 难度分层三重证据证明"内化的是推理形状而非记忆内容",机制叙事扎实;③ "harness 是临时训练支架、用完即弃"这一定位本身具启发性。被适度高估:增益数字部分依赖 OPSD 在该配置下的退化做对照、且最佳分选择性报告;Non-Distillable 类(真正工具/检索)明确无法处理——这恰是 agentic 自进化最难的部分,本文回避了。

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号:frozen base 在"harness 终端上下文"下的 next-token 分布与学生(裸输入)分布的**逐 token reverse-KL**——稠密、token 级、过程性。
  - 改什么:学生策略参数 \(\theta\)(verl/PPO trainer 子类),梯度只经学生。
  - 何时改:学生直接 rollout 的每一步 token 处给信号(on-policy)。
  - 免梯度?否——基于梯度的 reverse-KL 蒸馏。
  - 记忆-技能生命周期:harness 内的"在线 memory bank"是训练期专属 oracle(\(z(x)\)),推理时撤除;"技能"=harness 诱导的推理形状被写进参数(cite-rate ≥90% 证内化);teacher 锚在初始 base(稳定先验)。
  - 防遗忘机制:监督 logit 锚在 frozen base \(\tilde\theta\)(隐式正则,与 OPSD 同);Fig.4 显示"严格保留原有能力(−1.90)同时大幅提升 harness-only 子集"。
- ⑦ 开源代码+框架/harness:github.com/zzy1127/OPHSD-On-Policy-Harness-Self-Distillation(已 clone,~57M;`data/deepmath10k/data_train_10k.json` 由 Git LFS 跟踪,本次 SKIP_SMUDGE **未实下载**——需手动 `git lfs install` 后拉取真实 JSON)。框架 **veRL**(README:"OPHSD subclasses verl PPO trainer",需 `pip install -e <verl>`;底层 RL trainer 基于 HJSang/OPSD_OnPolicyDistillation repo)。harness 经 openai 客户端与 vLLM 通信。代码结构(已核仓库树):`ophsd_train/src/ophsd/`(trainer+worker,Hydra 配置)、`src/rewards/`(任务奖励)、`harnesses/`(每任务自包含子包 lawbench/uspto/math,共享 `_api.py`、`_memory_bank.py`);三 launcher `scripts/train_ophsd_{math,lawbench,uspto}.sh`;`data_prep/`(JSON→verl parquet 转换 + `precompute_embeddings.py`);LawBench/USPTO 需 precompute_embeddings(嵌入器 BAAI/bge-small-zh-v1.5)。【原文/仓库】
- 💰 资源/成本与可扩展性:8×H100;lr 1e-6、batch 64、max gen 8192;文本分类训 300 步(每 15 步评)、数学 150 步(每 10 步评,4 次平均);harness 温度——draft 0.1 / plan 0.3 / solve 0.6;draft-verify 每输入 2 次模型调用(\(k_d=k^+=k^-=5\))。规模仅验证 8B。【原文】§A
- 🎯 对"探索-巩固"对标:**最强结构同构(直接对标 TSRD)**。一句判定:OPHSD 就是"teacher 当稀疏脚手架教 student 把过程能力内化进参数、用完撤除"的现成实现,与本课题"巩固/回轨=固化进参数且不遗忘"几乎一一对应。依据:abstract"temporary training scaffolds whose benefits are permanently fed back into the base model" + Eq.3 解耦。可借组件:① **harness 驱动 \(\theta\) / 监督锚 \(\tilde\theta\) 的解耦**(轨迹随能力演进、信号锚稳定先验)——可直接用于"探索分支随能力变、监督锚不漂移";② **三档可蒸馏性边界**(Appendix C)——明确划出"哪些能内化进参数、哪些是真外部"的判据,对 Agent 自进化的"巩固"边界极有价值;③ **plan 经中介强迫内化推导结构**——对应"教选路/恢复分支的过程而非答案";④ cite-rate/能力差分组的**内化度量**方法。缺口/竞品差异:① 它做的是"全程逐 token 均匀监督",不显式建模"在关键步/高熵步接管"或"path-recovery 单点恢复"(本课题的 sparse_critical 设计可补);② 无 MTP/前瞻;③ "探索/选路偏向自己能走通的开头"未建模(harness 由固定程序编排,不挑学生的强开头);④ Non-Distillable 类(真工具/检索)被排除——而 Agent 自进化恰需处理这一类。
- 🔭 开放问题/未来方向:
  - 【原文】§5+Appendix C:蒸馏更广的 harness 结构、内化更复杂工作流;Non-Distillable 类里"调用脚手架的程序结构(何时调工具/如何 format query)是否可蒸馏"是明确未来方向。
  - 【推断】把"全程均匀 reverse-KL"改为"只在 harness 与裸学生发散最大的关键步接管"可更省更准(与本 survey 的 sparse_critical / forward-hard-backward-soft 同向);把 MTP 当"前瞻探针"判断学生是否在某步将走偏、据此触发 harness 监督,可把 OPHSD 的"被动全程蒸馏"升级为"主动关键步脚手架";针对 Partially/Non-Distillable 类,探索"蒸馏工具调用的决策结构(而非内容)"以触达真正的 agentic 自进化。

— RETURN —
ophsd | 读PDF? 是(18页,§1-§4+Appendix A/C 全核,Eq.1/2/3 + 两 harness 终端上下文逐字核对) | 加厚? 是(Eq.2 harness 诱导分布 + Eq.3 全展开 + plan-solve/draft-verify 终端上下文公式化 + LUPI/NLAHs/Meta-Harness 相关工作扩位) | LaTeX公式条数:5(OPD Eq.1 / harness分布Eq.2 / OPHSD目标Eq.3 / plan-solve / draft-verify) | 待核数:1(reverse vs forward KL 切换原因原文未解释)

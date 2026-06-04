rstar2 | rStar2-Agent: Agentic Reasoning Technical Report | Microsoft Research(Ning Shang/Yifei Liu/Yi Zhu/Li Lyna Zhang 等共同一作;Li Lyna Zhang、Mao Yang 为 project leaders/通讯) | 2025-08-28 v1 · arXiv 技术报告(cs.CL) | L4 Agent/工具/多轮(+L3 GRPO) · 相关性高

**原始论文**:https://arxiv.org/abs/2508.20722

## 一眼看懂

> 一句话导读:微软这篇技术报告想证明"14B 小模型靠想得更聪明、而非把推理写得更长,也能冲到前沿数学"。关键招是在采样层(而不是在奖励里)抗环境噪声——正样本只留最干净的、负样本保留多样失败,再配上能撑住数万并发工具调用的执行环境和短长度训练。

- 🟦 TL;DR:目标是让 14B 模型"想得更聪明而非更长"。靠三件套:
  - (i) 一个可靠、高吞吐的 Python 代码执行环境(每步可处理 45K 并发工具调用、单次调用 0.3s)+ 负载均衡的 rollout 调度器;
  - (ii) 抗噪的 **GRPO-RoC**(Resample-on-Correct,即"按正确性重采样":先 oversample 出 \(2G\) 条、再 downsample 到 \(G\) 条,其中正样本筛"工具错误 / 格式问题最少"的高质量轨迹、负样本则均匀采样以保留多样性);
  - (iii) 一套低成本 recipe(先做 non-reasoning SFT 打底,再用 GRPO-RoC 做 8K→12K→12K 的三阶段短长度 RL)。
  - 成绩:在 64×MI300X 上、510 步 / 一周内,把 Qwen3-14B-Base 推到前沿数学——AIME24 = 80.6 / AIME25 = 69.8 / HMMT25 = 52.7,其中 AIME24 与 HMMT25 超过了 671B 的 DeepSeek-R1。【原文】Abstract、§4.3、Table 5
- 最巧的一步:**GRPO-RoC 的非对称采样(正筛优 / 负保多样)**。
  - 抽掉它,在 answer-only outcome reward(只看最终答案对错)下,模型会把那种"最终答案对、但中间工具调用出错 / 格式乱"的冗长低质轨迹也当正样本来学(Fig.4:naive GRPO 下,正样本里的工具错误率降到一定值后就 plateau 在 ~10–15%),于是环境噪声被放大、产出退化。
  - 论文明确选这条路、而不选"step-level reward(逐步奖励)/ tool-error 惩罚",因为后两者会引入复杂度、需要人工调参或额外的 reward model,而且**早期阻碍探索、容易 reward-hacking**(钻奖励的空子)。
  - 它是"用最小的 outcome reward 抗环境噪声"的关键。【原文】§2.2.3、Eq.4-6、Fig.4

## 为什么做

> 一句话导读:纯靠"写更长的推理"在难题上会撞天花板,让模型自己调代码工具能拓宽动作空间;但这条 agentic 路线一上规模就撞两堵墙——一是工具报错带来的环境噪声、二是数万并发工具调用拖垮基础设施。前人要么在奖励里惩罚错误(复杂、易钻空子)、要么硬堆长度(贵),rStar2 都不走。

- **研究背景的来龙去脉**:RLVR(DeepSeek-R1 范式)用规则可验证的 outcome reward 让模型自主习得 long CoT(长思维链)推理。但纯 long CoT 对难题有内禀上限:模型只能靠"内部自反思"来纠错,动作空间窄。
  - **agentic reasoning**(智能体式推理)这条线让模型自主调用工具(Python 编码 + 解释器)来推理 / 验证 / 从工具反馈里学——拓宽了动作空间,支持探索替代解、验证中间步,补足单纯 long CoT 的不足。本报告就是沿这条 agentic RL 路线,把 14B 推到前沿。【原文】§1
- **GRPO 基线与"更多探索"的现成做法**:本文 RL 底座是 GRPO,目标
  \(\displaystyle \mathcal{J}_{\mathrm{GRPO}}(\theta)=\mathbb{E}\Big[\tfrac{1}{|o_i|}\sum_{t}\min\big(\rho_{i,t}A_{i,t},\,\mathrm{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)A_{i,t}\big)-\beta\, D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})\Big],\quad \rho_{i,t}=\tfrac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})}\)
  其中 group-relative advantage \(A_{i,t}=\dfrac{r_i-\mathrm{mean}(\{r_1,\dots,r_G\})}{\mathrm{std}(\{r_1,\dots,r_G\})}\)(Eq.1-2)。为了"探索超出预训练能力的上限",本文吸收了三项现成改动:
  - **移除 KL 惩罚**(\(\beta=0\)):放开对参考模型 ref 的约束,以便发现新的、用工具增强的推理模式;
  - **Clip-Higher**(把 \(\varepsilon_{\mathrm{high}}\) 从 0.2 放到 0.28):放开 clip 上界,以探索那些高熵、低概率的"forking"(分叉)token;
  - **移除 entropy loss**:防止熵失控、导致 collapse(坍缩)。【原文】§2.2.1
- **agentic RL 扩展的两大具体挑战(本文靶子)**:
  - **环境噪声**:编码工具很复杂,模型一旦生成错误代码,环境就返回一堆与推理无关的报错,让它白白浪费 token 去纠错。而且 answer-only outcome reward(Eq.3,\(r_i=\mathbb{1}[\text{is\_equivalent}(a,o_i)]\))**没法惩罚中间的错误行为**——"最终答案对就给正奖励"会让模型把中间的工具错误也当成可以接受的,于是产出冗长低质的轨迹(Fig.4 实测:工具错误率随训练 plateau 在一个显著的水平)。【原文】§2.2.2-2.2.3
  - **基础设施压力**:单个 batch 就可能触发数万次并发工具调用(实测每步 up to 45K)。agentic rollout 把标准 RL 本就低效的 rollout 进一步放大,严重拖慢训练。【原文】§3、§3.1
- **并行技术路线 + 各自具体短板**:
  - **reward 侧两条现成路**:
    - (i) step-level reward(Yue 2025a)→ 需要构造 reward model / 人工调参,且早期阻碍探索;
    - (ii) outcome reward + tool-error 惩罚(ToolRL / Kimi 等)→ 同样引入复杂度,且容易 reward-hacking。
    - rStar2 **两者都不用**,改在**采样层**对正 / 负轨迹做差异化处理(即 RoC)。【原文】§2.2.3
  - **长度侧主流做法**:堆 rollout 长度(16K→48K→80K,如 MiniMax / MiMo / Magistral,Table 2)→ 成本高、产出冗长。rStar2 反其道,走"短长度(8K→12K)+ 最小 outcome reward + 采样层抗噪"。【原文】Table 2
- **与最近邻工作的精确差异(Table 2 显式对比)**:相比 DAPO / ReTool 这类单阶段长长度 RL,关键差异是三点——**GRPO-RoC(采样层抗噪)+ non-reasoning SFT 冷启 + 多阶段短长度(8K→12K)**。Table 2 按几个维度对齐:是否做 Reasoning SFT / RL 阶段数 / 总步数 / 最大长度 / 难度过滤——其中 rStar2 是唯一同时做到"**无 reasoning SFT + 短长度 + 仅 510 步**"的。【原文】Table 2

## 怎么做 + 靠不靠谱

> 一句话导读:三阶段——先用不含推理的数据 SFT 打底、再清出 4.2 万道可验证的整数答案题、最后做多阶段短长度的 GRPO-RoC。RoC 的核心是 oversample 一倍再非对称下采样:正样本用"工具错误率 + 格式惩罚"软筛出最干净的、负样本不筛以保多样。可靠性上要冷静看:对照表是各家公布数(非同 harness 重测),整数答案约束 + 单基座单硬件 + 开源版尚未训出完整模型,复现有门槛。

- **方法流水线(输入→输出,逐阶段)**:
  1. **non-reasoning SFT**(§4.1):只用注入指令遵循 / JSON / 基础工具使用的数据做 SFT(**不增加推理能力**),得到初始策略。**输出**:一个会走 Thought-Code-Observation 多轮循环、但推理仍弱、响应短的起点模型。Table 1 证实:SFT 后 tool / IF / chat 都有提升,数学持平于 base(MATH-500 62.0→57.4 略降)。
  2. **RL 数据清洗**(§4.2):>100K 候选 → **只留整数答案的题**(这样规则 verifier 能判)→ 经 Qwen3-32B 生成 16 次、要求 ≥2 次匹配来过滤 → 得 42K 题(17K DAPO 整数子集 + 93K AoPS via OpenMathReasoning + 937 Project Euler)。**输出**:42K 道可验证的训练题。
  3. **多阶段 GRPO-RoC**(§4.3):Stage-1 用 8K 长度(42K 题,300 步,响应从 1K 涨到 4K)→ Stage-2 用 12K(4K→6K)→ Stage-3 在 offline 过滤出的 17.3K 难题上(**reset 优化器 + 更新 reference**,125 步,6K→8K),合计 **510 步**。期间环境服务异步执行工具调用 + 负载均衡调度。**输出**:rStar2-Agent-14B。
- **GRPO-RoC 的核心算法(真实形式 + 直觉)**:先 oversample 出 \(2G\) 条 \(\{o_i\}_{i=1}^{2G}\),记负样本集为 \(O_{\mathrm{neg}}\)、正样本集为 \(O_{\mathrm{pos}}\)(\(|O_{\mathrm{neg}}|+|O_{\mathrm{pos}}|=2G\)),然后非对称地下采样到 \(G\) 条:
  - **负样本(保多样)**:对 \(O_{\mathrm{neg}}\) **不做过滤**,按原分布采 \(\lfloor|O_{\mathrm{neg}}|/2\rfloor\) 条 → 这样能暴露多样的失败模式,让模型学会避开多种错误。
  - **正样本(筛优)**:对每条成功轨迹算两项惩罚。
    - 工具错误率(无工具调用时默认 0.5,目的是**鼓励用工具**):
      \(\displaystyle p_{\mathrm{err}}=\begin{cases}0.5,&\text{无工具调用}\\[2pt]\dfrac{\#\text{错误工具调用}}{\#\text{全部工具调用}},&\text{否则}\end{cases}\)
    - 格式惩罚(惩罚缺 `<answer>` 标签、或标签过多导致重复):
      \(\displaystyle p_{\mathrm{format}}=\begin{cases}1,&\text{无 <answer> 标签}\\[2pt]\min\!\big(1,\ \dfrac{\#\langle\text{answer}\rangle\text{标签}-1}{\#\text{turns}}\big),&\text{否则}\end{cases}\)
    - 总惩罚 \(p_{\mathrm{total}}=p_{\mathrm{err}}+p_{\mathrm{format}}\);再按**与 \(p_{\mathrm{total}}\) 成反比**的概率采出半数正样本(惩罚越低越容易被选中)。
  - **最终目标**(Eq.6-7):在 RoC 选出的 \(\{\hat o_i\}_{i=1}^{G}\) 上算
    \(\displaystyle \mathcal{J}_{\mathrm{GRPO\text{-}RoC}}(\theta)=\mathbb{E}\Big[\tfrac{1}{\sum_i|\hat o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|\hat o_i|}\min\big(\hat\rho_{i,t}\hat A_{i,t},\,\mathrm{clip}(\hat\rho_{i,t},1-\varepsilon_{\mathrm{low}},1+\varepsilon_{\mathrm{high}})\hat A_{i,t}\big)\Big],\quad \hat A_{i,t}=\tfrac{\hat r_i-\mathrm{mean}(\{\hat r\})}{\mathrm{std}(\{\hat r\})}\)
    其中 \(\varepsilon_{\mathrm{low}}=0.2,\ \varepsilon_{\mathrm{high}}=0.28\)(即 Clip-Higher)。
    - **直觉**:正轨迹"挑最干净的成功样本"做正监督,负轨迹"保留多样失败模式"做负信号——这比在 reward 里直接惩罚工具错误更稳、能避免 reward-hacking(因为报错惩罚在训练早期会误伤探索)。Fig.4 / Fig.9 证实:GRPO-RoC 下工具错误率持续下降、推理更强、响应更短。【原文】§2.2.3、Eq.4-7、Fig.4/9
- **逐组件必要性**:
  - **non-reasoning SFT**:只打底、不增推理(Table 1)——刻意让初始响应保持短、不过拟合。没有它,起点的响应会过长。【原文】§4.1、Table 1
  - **GRPO-RoC**:核心抗噪手段;Fig.9 给出了与两个 RoC 变体的消融对比,证明非对称采样更优。【原文】§2.2.3、Fig.9
  - **高吞吐环境服务 + 负载均衡调度器**:架构是中心任务队列 + 32 个 send workers + 每个 worker 节点 1024 个执行 workers。调度器按可用的 KV cache 动态分配 rollout、异步 dispatch 工具调用,answer verification 也异步 offload。没有它,GPU 会大量 idle。【原文】§3.1-3.2
  - **多阶段短长度**:Stage-1 用 8K 时 clipping ratio 一度 >10% 仍坚持(逼模型高效推理);Stage-3 是因为 Stage-2 末 >70% 题已满分,才切到 17.3K 难题 + reset 优化器 / 更新 reference。【原文】§4.3
- **训推数据如何流动**:RL 数据(42K)→ vLLM 多轮 rollout(`--enable-auto-tool-choice --tool-call-parser hermes`)→ 每步触发 up to 45K 并发工具调用,经环境服务异步执行(0.3s/call)→ 收集 \(2G=32\) 条 → RoC 选出 16 条 → 算 \(\hat A_{i,t}\) → GRPO-RoC 策略梯度更新。超参:lr \(1\times10^{-6}\)、batch 512 prompts、\(2G=32\to16\)。【原文】§4.1、§3.1
- **实验与证据**:基座 Qwen3-14B-Base。数学评测用 AIME24/25、MATH500、HMMT25;泛化用 GPQA-Diamond、BFCL v3、IFEval、Arena-Hard。
  - 数学(pass@1):rStar2-Agent-14B **AIME24 = 80.6 / AIME25 = 69.8 / HMMT25 = 52.7**;对照组——DeepSeek-R1(671B) 79.8 / 70.0 / 44.4、o3-mini(medium) 79.6 / 77.0 / 53.0、DeepSeek-R1-Zero 71.0 / 53.3 / 46.0、QWQ-32B 79.5 / 65.8 / 47.5。即 14B 在 AIME24、HMMT25 上超过 R1-671B,**但 AIME25 略低 0.2(70.0 vs 69.8,基本持平)**。【原文】Table 5
  - 效率:510 步 / 一周 / 64×MI300X;环境服务 45K 并发/步、0.3s/call。【原文】§3.1、§4.3
  - baseline 公平性:对照表是各家公布的成绩(不是同一 harness 重测),这属技术报告惯例,严格的可比性有限。【推断】
- **假设与失效边界**:
  - 【原文】只用整数答案题(因为规则 verifier 很难判代数等价,例如 \((a+b)(b+c)(c+a)\) 和它的重排),这限制了任务多样性。
  - 【原文·§5.x】Stage-3 的性能在 510 步后 saturate 甚至 collapse(policy 与 reward 同时崩),所以就停了;开源迁移版(VERL v0.2→v0.5)还没训出完整模型(作者注:前 50 步差异极小,但没给出最终复现成绩)。
  - 【原文】不采用 overlong filtering(过长过滤):被截断的 rollout 保留下来、给负奖励(而不是像 DAPO 那样丢弃 / 不计奖励),作者称这有助于 resample-on-correct。
  - 【推断】单一基座(Qwen3-14B-Base)、单一硬件栈(MI300X);可靠性高度依赖自研的环境服务工程实现,迁移稳定性未验证;AIME25 没有严格超过 R1,说明优势并非跨所有 benchmark 一致。
- **祛魅总结**:
  - 真贡献:证明了"采样层抗噪(RoC)+ 短长度 + 高吞吐环境"能让 14B 在极少步数 / 算力下达到前沿,挑战了"必须堆长度"的范式;GRPO-RoC 的"正筛优 / 负保多样"是个简洁有效的抗 reward-hacking 设计(精髓是用 \(p_{\mathrm{err}}/p_{\mathrm{format}}\) 在采样层做"软筛",而非在 reward 里硬惩罚);基础设施工程(45K 并发、负载均衡)是 agentic RL 落地的硬贡献。【推断】
  - 包装 / 需冷静看:TL;DR / 摘要里"超越 671B R1"主要靠 AIME24 / HMMT25,**AIME25 实际是略低 / 持平**(差 0.2);"前沿"对照表是各家公布数而非同 harness。整数答案约束 + 单基座单硬件,使泛化和可复现性都有门槛(开源版尚未训出完整模型)。【推断】

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号:answer-only 0/1 outcome reward(\(r_i=\mathbb{1}[\text{is\_equivalent}]\),经 RoC 在采样层加质量筛选)+ group-relative advantage \(\hat A_{i,t}\)。
  - 改什么:策略参数(GRPO-RoC 策略梯度);**不改 token 选择,改"哪些轨迹进 batch"**(\(2G\to G\) 非对称下采样)。
  - 何时改:每步在线,oversample→RoC 筛选→更新;难度过滤为阶段性 offline。
  - 免梯度?否,RL 策略梯度(GRPO),且 \(\beta=0\)(无 KL)、Clip-Higher(\(\varepsilon_{\mathrm{high}}=0.28\))、无 entropy loss。
  - 记忆-技能生命周期:无外部记忆库;"工具使用技能"靠 SFT 注入格式 + RL 内化,固化进参数。
  - 防遗忘机制:无显式防遗忘;Stage-3 reset 优化器 + 更新 reference 是稳定性手段而非防遗忘。
- ⑦ 开源代码+框架/harness:https://github.com/microsoft/rStar(main 分支是 rStar2-Agent;prior work 在 rStar-mutualreasoning / rStar-math 分支;v1 浅克隆约 576K,未拉 submodule)。
  - 框架 = **veRL v0.5(volcengine/verl)+ Code Judge(0xWJ/code-judge,即开源版的工具执行服务,Redis + uvicorn + workers)**。原训练基于 VERL v0.2 + 自研多轮工具框架,开源版迁移到了 v0.5。rollout / 推理用 vLLM(`--enable-auto-tool-choice --tool-call-parser hermes`)。
  - 关键脚本 `examples/run_qwen3-14b_rstar2_agent_weave.sh`;RoC 配置项:`do_down_sampling=True`、`down_sample_to_n=16`、`roc_error_ratio=True`、`roc_answer_format=True`、`min_zero/non_zero_reward_trace_num=2`。
  - **未完整克隆说明**:verl / code-judge 是 git submodule(需 `git submodule init/update`),v1 浅克隆没拉取;且开源迁移版未训出完整模型,端到端复现有门槛——代码链接见上,需手动补 submodule。【原文】§3.1 + 仓库(v1 核查)
- 💰 资源/成本与可扩展性:**64×MI300X、510 步、一周**(刻意"最小算力达前沿"是核心卖点);环境服务分布在 64-GPU 集群 CPU 核上(32 send workers + 每节点 1024 执行 workers),45K 并发/步、0.3s/call;短长度(8K→12K)大幅降算力;lr \(1\times10^{-6}\)、batch 512、\(2G=32\to16\)。【原文】§3.1、§4.3
- 🎯 对"探索-巩固"对标:**部分支撑(可借采样思想),非直接同构**。判定:
  - GRPO-RoC 的两条思路——"正样本筛最干净的成功轨迹作正监督(对应巩固有效行为)+ 负样本保多样失败模式(对应暴露并学会避开走偏路径)"——与本项目"探索 = 发现有效路径、巩固 = 固化有效行为"在精神上一致;尤其"保留多样失败作负信号"对应"教学生识别 / 恢复走偏"。
  - 可借组件:用工具错误率 \(p_{\mathrm{err}}\) / 格式 \(p_{\mathrm{format}}\) 给轨迹质量打分、按 \(1/p_{\mathrm{total}}\) 做非对称采样,这套可迁移到 TSRD 的"挑高质量正样本来巩固"。
  - 关键差异:rStar2 是**轨迹级 outcome RL**(没有 token / step 级的 path-recovery、没有 teacher 脚手架、没有自蒸馏、没有 MTP),它的"巩固"靠 RL 内化而非 OPD;而且失败轨迹只当"负 advantage",并不"教如何从该点恢复"。
  - 依据:§2.2.3。【推断】
- 🔭 开放问题/未来方向:【原文】讨论性能饱和后 collapse(何时停);Unsuccessful Attempts(§4.3.x)的经验教训(如不采用 overlong filtering)。【推断】超越整数答案约束的可验证任务扩展;GRPO-RoC 迁移到非编码 agentic 环境;把"采样层质量筛选"与 token/step 级监督(path-recovery)结合;开源版训出完整模型以验证可复现性。

RETURN: rstar2|读到PDF?是(22页,_txt 87k字,GRPO/Eq.1-7 + perr/pformat + Clip-Higher 超参全核)|L4(+L3)|对标=部分支撑:RoC 正筛优(p_err/p_format)/负保多样≈巩固有效+学会避坑,可借非对称采样;非同构(轨迹级 RL,无 token 级 path-recovery/无 MTP)|残留待核 0

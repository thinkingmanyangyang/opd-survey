rl_survey_lrm | A Survey of Reinforcement Learning for Large Reasoning Models | 清华 + 上海AI Lab + 上海交大 + 北大 + 中科大 + 哈工大 + 华盛顿大学 + 华科 + UCL(40+ 作者;Project Lead Kaiyan Zhang/Yuxin Zuo;通讯 Biqing Qi/Ning Ding/Bowen Zhou) | 2025-10-10 v3 · arXiv 综述(cs.CL) | L3 RLVR/GRPO(综述,横跨 L1-L6) · 相关性中(领域坐标系/背景)

**原始论文**:https://arxiv.org/abs/2509.08827

> 注:本条为**综述**(非方法论文)。下列"怎么做/实验/方法"等栏按综述视角填写其 **taxonomy 与覆盖范围**,而非单一方法。综述本身几乎无原创公式,仅在 §4.1 借 KL 框架辨析 RL 与 SFT 的本质区别,故 LaTeX 公式仅此两条(忠实抄自原文)。

## 一眼看懂
> 一句话导读:这是一篇 DeepSeek-R1 之后"用 RL 训大型推理模型"的领域综述;它用一张总图把领域切成五层,并提炼出五对"X 还是 Y"的开放争议作为坐标系,核心立论是 RLVR 是区别于 RLHF/DPO 的一条新 scaling 轴。

- 🟦 TL;DR:本文系统综述了 DeepSeek-R1 以来"RL for 大型推理模型(LRM)"这一范式,用一张总图(Fig.1)把它组织成五层——**基础组件(§3)/ 五对开放争议(§4)/ 训练资源(§5)/ 下游应用(§6)/ 未来方向(§7)**。核心论断:**RLVR(可验证奖励 RL)是一条区别于 RLHF/DPO(后者做的是人类对齐)的新 scaling 轴**,正是它把 LLM 变成了 LRM;而 RLVR 向 ASI 的可扩展性,则是核心开放问题。配套有一个 GitHub Awesome-list 维护文献清单。【原文】Abstract、Fig.1、§1
- 最巧的一步(对综述而言,就是它的组织骨架):**五对"X or Y"对立式的开放问题(§4)**——Sharpening or Discovery、Generalize or Memorize、Weak or Strong prior、Tricks or Traps、Process or Outcome。抽掉它,综述就退化成单纯的"文献罗列";正是这五对争议,把海量、快速演进的工作结构化成了一套可追踪的"分歧坐标系"。【原文】§4、Fig.1

## 为什么做
> 一句话导读:R1 之后这个领域爆炸式增长、文献多到难追踪,作者写这篇综述是为了给出一套坐标系——把 RLVR 从 RLHF/DPO 里切出来当独立新范式,并以"如何 scaling 向 ASI"为主线。

- 研究背景:早在 AlphaGo/AlphaZero 时代,RL 就证明了"窄而明确的奖励"能驱动超人表现;到了 LLM 时代,先是用 RLHF/DPO 做人类对齐(即 3H:helpful/honest/harmless)。近期则出现了 **RL for LRMs**——它不只对齐行为,而是直接激励推理本身(OpenAI o1、DeepSeek-R1 这两个里程碑用 RLVR 诱导出长链推理:规划、反思、自纠错,且性能随训练与推理算力平滑提升)。【原文】§1、§2
- 解决的具体痛点(综述视角):这个领域(尤其 R1 之后)爆发式发展,而要进一步扩展 RL for LRM,又面临算力、算法设计、训练数据、基础设施这些基础性约束,亟需重访发展轨迹、重估方法论、并探索通向 ASI 的可扩展策略。【原文】Abstract、§1
- 相关工作 & 各自不足:§2.3 梳理了已有的相关综述以定位自身——强调本综述聚焦 R1 之后,并以"语言智能体与环境在长期演化中的大规模交互"为核心视角。【原文】§2.3
- 动机链:领域文献海量、演进极快 → 需要一个坐标系 → 于是用"组件—争议—资源—应用—未来"这五层 + 五对争议来组织 → 便于读者建立坐标系并持续追踪。【原文】Fig.1、§1
- 与最近邻(其他 RL/LLM 综述)的Δ:本综述明确把 **RLVR 从 RLHF/DPO 中切出来(Fig.2)**、当作一个独立的新范式,并以"scaling 向 ASI"为主线;覆盖范围还延伸到了 agentic、多模态、多智能体、机器人、医疗等外溢应用。【原文】§1、§6

## 怎么做 + 靠不靠谱(=综述 taxonomy 与覆盖)
> 一句话导读:这一节按综述视角填写——它的"方法"其实就是那套分类法(taxonomy):§3 拆基础组件、§4 摆五对争议、§5 盘训练资源、§6 列应用、§7 指未来方向;对本项目最有用的是 §4.1 的 KL 框架(与 OPD 同源)和 §7.1/§7.2 的 Continual/Memory-based RL。

- 方法流水线(综述结构):§2 预备(RL 定义、o1 以来的模型脉络、相关综述)→ §3 基础组件 → §4 五对争议 → §5 训练资源 → §6 应用 → §7 未来方向 → §8 结论;并配一个 Awesome-list。【原文】Fig.1、目录
- 逐组件(各章覆盖内容,本版比 v1 把 §3/§4 细分项写全):
  - **§3 基础组件**:
    - §3.1 奖励设计:**Verifiable**(数学答案对/代码单测过)/ **Generative**(生成式奖励模型)/ **Dense**(逐 token/步骤稠密)/ **Unsupervised**(无监督,如置信度)/ **Reward Shaping**。
    - §3.2 策略优化:**Policy Gradient** / **Critic-Based**(PPO 类)/ **Critic-Free**(GRPO·RLOO 类)/ **Off-Policy** / **Regularization 目标**(KL 约束/熵)。
    - §3.3 采样策略:Dynamic/结构化采样、采样超参。【原文】§3、Fig.1
  - **§4 五对开放争议(每对的具体分歧)**:
    - §4.1 **Sharpening or Discovery(锐化还是发现)**:一派认为 RL 只是锐化 base 已有的分布(证据:Limit-of-RLVR/Yue 2025b 发现 Pass@K 在大 k 处被 base 反超、spurious reward 也能涨分、forking token 主导);另一派认为 RL 能发现新能力(证据:ProRL 长时间稳定 RL 能同时提升 Pass@1/Pass@K、Yuan 2025c 的技能组合)。综述用一个 **KL 视角来统一**二者:SFT 优化的是 forward KL \(D_{\mathrm{KL}}(p_{\text{data}}\Vert p_{\text{model}})\)(mode-covering,覆盖所有 mode),RL 优化的是 reverse KL \(D_{\mathrm{KL}}(p_{\text{model}}\Vert p_{\text{reward}})\)(mode-seeking,把质量集中到高 reward 区)。结论:这个争议应当转化为"在什么条件下,是 sharpening 主导、还是 discovery 主导"。
    - §4.2 **RL vs SFT: Generalize or Memorize(泛化还是记忆)**:
      - Chu 2025a 提出"SFT memorizes, RL generalizes";
      - 经验上,在数学上做 RL 能保留甚至增强 OOD/指令遵循能力,而在数学上做 SFT 常常负迁移/灾难性遗忘(用 latent-space PCA + KL 诊断);
      - 但 RL 并非万能(遇到强 overfit/分布突变就失效),而 SFT 加上 reweight/trust-region 也能泛化、且常常能更好地为 RL 热身;
      - 还有一类**统一/交替范式**:LUFFY 的 off-policy trace、单阶段 SFT+RL(用以克服 long-horizon 的样本复杂度)、branch rollout(从 expert anchor 衔接两阶段);其中 Ma 2025a 总结为"RL 擅长巩固已有能力、SFT 擅长引入新知识"。
    - §4.3 **Model Prior: Weak or Strong**(base 先验到底该弱还是该强)。
    - §4.4 **Training Recipes: Tricks or Traps**(各种 trick 究竟是真增益还是陷阱)。
    - §4.5 **Reward Type: Process or Outcome**(过程奖励 PRM vs 结果奖励 ORM,关乎 token/step 级的信用分配)。【原文】§4
  - **§5 训练资源**:§5.1 静态语料(Math/Code/STEM/Agent/Mixture);§5.2 动态环境(Rule/Code/Game/Ensemble);§5.3 RL 基础设施(OpenRLHF / veRL / AReaL / slime / TRL)。【原文】§5、Fig.1
  - **§6 应用**:Coding、Agentic、Multimodal、Multi-Agent、Robotics、Medical。【原文】§6
  - **§7 未来方向(9 项)**:7.1 Continual RL、7.2 Memory-based RL、7.3 Model-based RL、7.4 Efficient Reasoning、7.5 Latent Space Reasoning、7.6 RL for Pre-training、7.7 RL for Diffusion LLMs、7.8 Scientific Discovery、7.9 Architecture-Algorithm Co-Design。【原文】§7 目录
- 关键机制/核心论断(直觉):RLVR ≠ RLHF/DPO——前者用可验证信号激励推理能力本身,后者用偏好来对齐人类,二者是不同的 scaling 范式(Fig.2)。一个统一视角是:**RL 与 SFT 的本质区别可以归结到 KL 的方向**——RL 是 reverse-KL(mode-seeking),SFT 是 forward-KL(mode-covering)。这恰与 OPD 用 reverse-KL 蒸馏的机制同源(见 rethink_opd:OPD 本质 = dense KL 约束的 RL)。所以本综述虽然没有 OPD 专章,但它的 §3.2(正则/off-policy)+ §4.1(KL 框架)其实已经隐含了 OPD 的理论坐标。五对争议目前都还没有定论,是当前活跃的前沿。【原文】§1、Fig.2、§4.1
- 实验与证据(综述无原创实验):§5 梳理训练资源(静态语料/动态环境)与复用性,§5.3 比较主流基础设施;论断为对现有文献的归纳。【原文】§5
- 假设与失效边界:
  - 【原文】综述聚焦 o1/DeepSeek-R1 发布以来工作(v3 2025-10),范围明确。
  - 【推断】结论依赖所引文献质量;领域演进极快,v3 后工作必然遗漏;五对争议多呈现分歧而少定论性裁决;"RL 通向 ASI"为前瞻论断,缺可证伪具体路径;§5.3 基础设施比较随框架迭代易过时。
- 祛魅总结:
  - 真贡献:为快速演进、文献海量的"RL for LRM"领域提供权威坐标系;五对争议提炼清晰,§7 未来方向(尤其 Continual/Memory-based/Model-based RL)指向性强;Awesome-list 可追踪。【推断】
  - 边界:作为综述不提供新实证;对本项目的价值是"定位与查文献",而非可复现方法。【推断】

## 结构化抽取
- 🎯 机制速览6轴(综述 taxonomy 对应):学什么信号=综述把奖励分为 Verifiable/Generative/Dense/Unsupervised(§3.1);改什么=策略优化谱系 PG/Critic-Based/Critic-Free/Off-Policy/Regularization(§3.2);何时改=采样策略 Dynamic Sampling(§3.3);免梯度?=不适用(综述);记忆-技能生命周期=§7.1 Continual RL / §7.2 Memory-based RL 专章讨论(与本项目 L5 直接对应);防遗忘机制=§7.1 明确讨论 CRL 的 stability-plasticity 平衡、可塑性丧失(Dohare et al. 2024 指深度网络在持续学习中退化),且 §4.2 引 Shenfeld "RL's Razor"——online RL 比 SFT 更好地保留旧知识(=天然防遗忘)。【原文】§3、§4.2、§7.1-7.2
- ⑦ 开源代码+框架/harness:https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs(**awesome-list,仅论文/资源清单,无方法实现代码**,不可运行复现)。综述本身无独立训练框架;§5.3 梳理并比较主流 RL 基础设施 **OpenRLHF / veRL / AReaL / slime / TRL**。【原文】§5.3 + 仓库(v1 核查)
- 💰 资源/成本与可扩展性:综述本身无成本;但全文核心议题之一即"RL for LRM 的 scaling(算力/算法/数据/基础设施约束)向 ASI",§5 讨论训练资源复用性。【原文】Abstract、§5
- 🎯 对"探索-巩固"对标:**坐标系/背景支撑,而非方法对标**。判定:它为本项目提供的是文献定位——
  1. **§4.1 Sharpening vs Discovery** 直接关系到本项目的"探索 = 发现有效路径"究竟是真发现、还是锐化已有分布;且它的 reverse-KL mode-seeking 框架与 OPD/TSRD 的机制同源(可直接引为理论背景);
  2. **§4.2 Generalize or Memorize** 关系到 TSRD"把 SFT(引入新知识)与 RL(巩固/泛化)结合"的设计——综述明确列出"RL 擅巩固、SFT 擅引入新知识"(Ma 2025a)、SFT 热身能稳住 RL、branch rollout 从 expert anchor 衔接两阶段(与 TSRD 的 path-recovery 从锚点接管同向);
  3. **§4.5 Process vs Outcome** 关系到 token/step 级的信用分配(即本项目的 path 监督);
  4. **§7.1 Continual RL + §7.2 Memory-based RL** 正对应本项目的 L5(记忆/技能库 & 持续学习防遗忘)与"巩固/固化且不遗忘",可作该方向的权威背景与引文来源;
  5. §7.5 Latent Space Reasoning、§7.4 Efficient Reasoning 则关系到 CoT/前瞻。
  - 可借:用它的 taxonomy 给本 SURVEY 做章节定位与术语统一。
  - 缺口:它不含 OPD/自蒸馏的专门章节(OPD 散见于 §3.2 的正则/off-policy 与应用部分),也没有 MTP 前瞻的专题。依据:Fig.1、§4、§7。【推断】
- 🔭 开放问题/未来方向:【原文】§4 五对未决争议 + §7 九项未来方向(Continual / Memory-based / Model-based RL、Efficient/Latent Reasoning、RL for Pre-training/Diffusion、Scientific Discovery、Arch-Algo Co-Design);下一阶段 scaling(含 open-ended RL)向 ASI 仍开放。【推断】综述未给五对争议的定论,留待读者按自身场景判断;v3 后(2025-10 至今)的新工作需另行补充。

RETURN: rl_survey_lrm|读PDF?是(综述,_txt 420k字,核到 Fig.1 总图+§3.1/§3.2 细分类+§4 五争议含 §4.1 KL框架/§4.2 统一范式+§7 九方向)|加厚?是(§3/§4 各子项写全、§4.1 forward/reverse-KL 框架转 MathJax、对标补 §4.2 SFT-RL 统一与 path-recovery 同源)|LaTeX公式条数 2|待核数 0

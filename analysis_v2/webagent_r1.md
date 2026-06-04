webagent_r1 | WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning | UVA（Zhepei Wei，实习于 Amazon）+ Amazon + Georgia Tech | 2025-10-08·arXiv preprint·v2（cs.CL，初版 2505.16421） | 主题线 L4（Agent/工具/多轮 RL）·相关性 中

**原始论文**：https://arxiv.org/abs/2505.16421

## 一眼看懂

> 一句话导读：把单轮数学 RL 里好用的 GRPO 搬到"多轮 web agent"——agent 在真实 web 沙箱里点击/输入/滚动完成任务,只拿一个"成没成"的二值奖励;靠端到端多轮 on-policy RL + 动态上下文压缩,把成功率拉高一大截。

- 🟦 TL;DR：本文把单轮数学 RL 里成功的 GRPO 思想搬到"多轮 web agent"场景——agent 在真实 web 沙箱里点击/输入/滚动地完成任务,只拿一个终态二值奖励(成没成)。方法用端到端多轮 on-policy RL（M-GRPO + 动态上下文压缩 + 并行轨迹 rollout）把 web agent 从 prompting/行为克隆水平大幅拉高（Qwen2.5-3B 6.1%→33.9%,Llama3.1-8B 8.5%→44.8%）,并系统说明三件事:
  1. BC warm-up(行为克隆预热)不可省;
  2. long-CoT 帮 SFT 但会限制 RL 探索;
  3. web 任务的 test-time scaling 走的是"交互轮数"而非"单步长度"【原文 Abstract、§4】。
- 最巧的一步：**动态上下文压缩**。抽掉它训练就 OOM 跑不起来——每步的 HTML observation 动辄数千 token,长 horizon 累积上下文会爆显存。本文把历史 observation \(s_i\) 替换成短占位符 \(s_i'\)="Simplified HTML",只保留完整动作历史,并**同步更新 loss mask 让损失只算在动作 token 上**,这才让多轮 RL 在 web 上首次可行【原文 §2.3、line 386-413】。（M-GRPO 本身只是把 GRPO 推广到多轮,工程上这步压缩才是"能不能训"的开关。）

## 为什么做

> 一句话导读：单轮 RL 在数学上很成功,但多轮 web 这种长 horizon 场景还没跑通;早期靠 prompting/行为克隆缺探索,而 off-policy RL 又会断掉端到端交互、学不到任务依赖行为。

- 研究背景：RL 显著提升了单轮 LLM（DeepSeek-R1,数学）,但在多轮交互环境(尤其 web browsing 这类长 horizon、需领域技能的)仍欠探索。早期 web agent 靠 prompting 或行为克隆（BC/SFT）。任务形式化为 POMDP:状态=当前页 HTML 文本,动作 ∈ 预定义空间（Click/Type/Select/Scroll/Search/Exit…）,终态二值奖励 \(r\in\{0,1\}\)【原文 §1】。
- 解决的具体痛点：
  1. prompting/BC 缺探索与试错、泛化差;
  2. 离线/迭代 off-policy RL（Filtered BC、AWR、DigiRL、WebRL）会打断端到端交互,引入轨迹过滤/outcome reward model 训练/迭代优化等额外复杂度;且 off-policy 数据来自旧版 agent,在任务相互依赖的动态 web 中学不到关键行为(举例:先登出再编辑 profile——登出后失去访问权,旧数据里没有登出行为,就会生成无效动作);
  3. 每步 HTML 可达数千 token、长 horizon 累积上下文→OOM;
  4. 现有单轮 GRPO 实现不适配多轮【原文 §1、§2.3】。
- 相关工作 & 各自不足（来龙去脉/并行路线/精确差异,Table 1 四维对照）：
  - **① prompting**（Wang 2024 等）/ **BC/SFT**（Yin 2024、Hong 2024 等）——无 Trial-and-Error(不与环境交互学习)。
  - **② AWR**（Peng 2019）——有 Self-Sufficient,但 off-policy、需 replay buffer。
  - **③ DigiRL**（Bai 2024）——有 Trial-and-Error,但 off-policy + 需 replay buffer。
  - **④ WebRL**（Qi 2025,最近邻）——有 Trial-and-Error,但 off-policy + 需 replay buffer + **非 Self-Sufficient(额外训一个 outcome reward model 来标 GPT-4 生成的数据)**。
  - **⑤ 并发的端到端 on-policy RL**（RAGEN、SkyRL）——用于模拟游戏/编码,真实 web 仍欠探索;GUI agent 线(截图多模态)与本文(纯 HTML 文本)正交。
  - **本文（WEBAGENT-R1）的精确Δ（Table 1 中唯一全 ✓ 者）**：**唯一同时满足** Trial-and-Error + On-Policy + Replay-Buffer-Free + Self-Sufficient(直接用环境内置的规则化二值奖励、无 reward model、无 replay buffer)。相对 WebRL 省掉了 reward model;相对单轮 GRPO,把整条多轮轨迹的 token 一并纳入 + 二值终态奖励 + 动态上下文压缩。**注意**:M-GRPO 的"多轮"实质上主要是把整轨 token 纳入 + group-relative 优势,**无中间步奖励塑形**(作者列为 future work)【原文 §2.1-2.3、Table 1、§5】。
- 动机链（逐步推进）：
  1. 单轮 RL 成功;
  2. 想用于多轮 web;
  3. off-policy 会断交互、学不到依赖行为;
  4. 改成 end-to-end on-policy（对齐最新行为、免 replay/过滤、agent 据自身过去决策自适应）;
  5. 但多轮上下文爆显存 + 单轮 GRPO 不适配;
  6. 于是动态压缩 + M-GRPO + 并行 rollout【原文 §1、§2.3】。

## 怎么做 + 靠不靠谱

> 一句话导读：两阶段——先 BC warm-up 学会基本操作、再端到端多轮 on-policy RL;最扎实的两个洞见是"R1-Zero(不预热直接 RL)反而变差"和"加 thinking 后单步没变长、交互轮数大增"。

- 方法流水线（两阶段,输入→输出）：
  - **(1) BC warm-up**：在公开的 9,460 条专家演示 \(\mathcal D=\{(h_t,a_t)\}\) 上做 SFT（\(h_t=(s_1,a_1,\dots,s_t)\) 是完整交互历史）,获取基本 web 操作技能。
  - **(2) 端到端多轮 on-policy RL**：在 WebArena 环境里,每个任务并行 rollout \(G\) 条完整轨迹（\(G\) 个独立浏览器实例 \(\{E_1,\dots,E_G\}\)、各自的 cookie、同起始页独立交互得到多样历史）,动态压缩历史 observation 防 OOM,用 M-GRPO 逐 token 做 PPO-clip 更新(带 group-relative 优势 + \(\beta\)KL),交互直到达最大步数或 agent 输出 `exit()`【原文 §2、§2.3】。
- 逐组件必要性（每组件 + 证据）：
  - **BC warm-up**：必要。R1-Zero(无 SFT 直接 RL)初始 6.1%、RL 后**反而略降**（动作残缺/格式错、罕得正奖、探索失败）——**强反例消融**【原文 §3/Fig.4】。
  - **动态上下文压缩**：必要（否则 OOM 不可训）。把 \(h_t=(s_1',a_1,\dots,s_t)\) 中早期的 observation 压成短占位符 \(s_i'\)(仅几个 token);新 obs 到达时 \(h_{t+1}=(s_1',a_1,\dots,s_t',a_t,s_{t+1})\) 把 \(s_t\) 换成 \(s_t'\);同步更新 loss mask 让损失只算动作 token。论文作为核心工程机制提出,未给"关掉它"的对照,但其必要性来自显存物理约束【原文 §2.3,推断:无 on/off 消融但属硬约束】。
  - **M-GRPO（group-relative 多轮优势）**：核心优化器。主结果 Table 2 的大幅提升即其贡献,但"多轮 vs 单轮 GRPO"无并列消融【推断】。
  - **并行轨迹 rollout**：效率/多样性组件——\(G\) 个独立实例同起始页独立交互得到多样历史【原文 §2.3】。
  - **long-CoT 初始化（R1-CoT 变体）**：研究性消融。long-CoT SFT 初值 24.5% > 标准 BC 20%,但 RL 增益更小（R1-CoT 24.5→30.3 vs R1 20→33.9）——揭示"确定性 CoT 会约束 RL 探索"(假设性解释,非受控证据)【原文 §4/Fig.4,推断:归因偏 hypothesize】。
- 关键机制/公式（真实符号 + 直觉,从 PDF 抄准）：
  - **BC warm-up 损失**：\(\displaystyle \mathcal L_{\text{BC}} = -\mathbb E_{(h_t,a_t)\sim\mathcal D}\big[\log\pi_\theta(a_t\mid h_t)\big].\)
  - **M-GRPO 损失（多轮,逐 action / 逐 token）**：\(\displaystyle \mathcal L_{\text{M-GRPO}}(\theta) = -\frac1G\sum_{i=1}^{G}\frac{1}{|\tau_i|}\sum_{j=1}^{|\tau_i|}\frac{1}{|a_{i,j}|}\sum_{t=1}^{|a_{i,j}|}\Big[\tilde A_{i,j,t}-\beta\,D_{\mathrm{KL}}(\theta)\Big],\) 其中 \(\tau_i=\{a_{i,1},\dots,a_{i,|\tau_i|}\}\) 是第 \(i\) 条轨迹的动作序列。token 级裁剪优势 \(\displaystyle \tilde A_{i,j,t}=\min\!\big\{r_{i,j,t}(\theta)A_{i,j},\ \mathrm{clip}(r_{i,j,t}(\theta),1-\epsilon,1+\epsilon)A_{i,j}\big\},\) 重要性比 \(r_{i,j,t}(\theta)=\dfrac{\pi_\theta(a_{i,j,t}\mid q,a_{i,j,<t})}{\pi_{\text{old}}(a_{i,j,t}\mid q,a_{i,j,<t})}\)。**group-relative 优势** \(\displaystyle A_{i,j}=\frac{r_i-\mathrm{mean}(\mathbf r)}{\mathrm{std}(\mathbf r)},\qquad \mathbf r=\{r_1,\dots,r_G\}\ (\text{规则化二值奖励}).\) 直觉:整条轨迹的所有 token 共享同一个 group-relative 优势 \(A_{i,j}\)（即"比同组其它轨迹好多少"）,逐 token 做 PPO-clip + \(\beta\)KL 约束偏离参考策略——本质是把 GRPO 的"单轮序列"换成"多轮整轨",**无中间步 credit assignment**【原文 §2.3、line 414-458】。
  - **test-time scaling 直觉**：thinking 格式 `<think>...</think><answer>do(...)</answer>` 促使 agent 多轮交互而非更长单轮输出——Table 3 显示加 thinking 后单步长度几乎不变（Qwen 139→142）,但交互轮数大增（6→17）,揭示 web 任务的 test-time scaling 走的是"交互深度"。
- 实验与证据：
  - 数据集/设置：训练/评测主用 **WebArena-Lite**（WebArena 的 human-verified 自托管沙箱,跨 Reddit/GitLab/CMS/Map/Shopping 五站,**非线上真实站点**）。BC 用公开的 **9,460** 条轨迹;RL 用 **647** 个任务训练、**165** 个 verified 任务评测(内置规则 rubric 判成功率)。OOD:WebVoyager（5 域、每域随机 25 任务,域在 WebArena 未见）。模型 Qwen2.5-3B、Llama3.1-8B;对照含 GPT-4o/o3/o4-mini、QwQ-32B【原文 §4.1、line 626-630】。
  - 关键数字：
    - 主结果（Table 2,WebArena-Lite 平均 SR%）:Qwen2.5-3B 6.1→33.9;Llama3.1-8B 8.5→44.8,均超同尺寸的 SFT/Filtered BC/AWR/DigiRL/WebRL（8B 上 WebRL 42.4 < 44.8）。
    - 与专有模型:o3=39.4、o4-mini=36.9——**8B(44.8) 超 o3,但 3B(33.9) 仍低于 o3**。
    - thinking prompting（Table 3）显著提升 SR（o4-mini 15.9→36.9）,单轮长度几乎不变但交互轮数大增。
    - test-time scaling（Fig.5）:放宽最大轮数→三类成功率单调升。
    - OOD（Table 4）:WebAgent-R1 平均 32% > SFT 12% > prompting 8.8%【原文 §4 Table 2-4、Fig.5】。
  - baseline 公平吗：基线取"复现与文献中较高者";"超 o3"需限定——仅 8B 成立、3B 不成立,且 o3 是零样本 prompting 对照(非微调),比较不完全对等【原文 §4,推断】。
  - 有无"看着强但没回答核心问题"：核心(端到端多轮 on-policy RL 有效 + BC 必要 + 交互深度 scaling)回答清楚;但 OOD 评测样本小(每域 25),多处归因(long-CoT 限探索、R1-Zero 失败)是假设性解释,未做消融验证【原文 §4、§6】。
- 假设与失效边界：
  - 显式假设【原文】：依赖规则化可验证的 outcome reward（String/URL Match、程序执行）;固定的预定义动作空间;仅文本 HTML 输入【原文 §4.1、§6】。
  - 隐式假设【推断】：BC warm-up 的 9,460 条演示已覆盖基本技能空间——否则 RL 起不来(R1-Zero 即反例)。
  - 何时失效【原文/推断】：开放式任务(如旅行规划)无可验证奖励→不适用【原文 §6】;遇到需新操作的交互元素(动作空间外)受限;线上真实站点(非沙箱)、多模态/截图场景未验证。
- 祛魅总结【推断】：真贡献是把端到端 on-policy 多轮 RL 在真实 web 沙箱里跑通(动态上下文压缩是让它"可训"的关键工程突破),并贡献两个扎实洞见——BC warm-up 不可省(R1-Zero 强反例)、web 的 test-time scaling 走交互深度(Table 3 长度不变/交互↑ + Fig.5 双证据,较有说服力)。被适度包装的是标题级的"超 SOTA/超 o3"——仅 8B 成立、3B 不超,且 o3 为 prompting 对照不对等;"M-GRPO"作为新算法的成分有限(约等于 GRPO 整轨 token + 二值奖励,无中间步塑形)。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：环境内置规则化的**二值终态奖励** \(r\in\{0,1\}\)(无 teacher、无 reward model);经 group-relative 归一化成 token 优势 \(A_{i,j}\)【原文 §2.1、§2.3】。
  - **改什么**：web agent 策略参数 \(\pi_\theta\)（BC warm-up 改一次 + M-GRPO on-policy 持续改）【原文 §2.2-2.3】。
  - **何时改**：两阶段——先 BC warm-up（off-policy SFT）,再端到端 on-policy RL 循环【原文 §2】。
  - **免梯度?**：否（BC=SFT 梯度;RL=M-GRPO policy gradient）。
  - **记忆-技能生命周期**：无外部记忆/技能库;"基本 web 操作技能"由 BC warm-up 固化进参数,RL 再精炼;动态上下文压缩是运行时上下文管理(非长期记忆)【原文 §2.3,推断:纯参数化】。
  - **防遗忘机制**：无专门防遗忘设计;on-policy + \(\beta\)KL 正则隐式约束偏离(未作为抗遗忘主张)【推断】。
- ⑦ 开源代码+框架/harness：仓库 https://github.com/weizhepei/WebAgent-R1 （v1 记已 clone 约 150MB;仓库实测含 `WebAgent-R1/`、`README.md`、`LICENSE`,子目录含 `Train/`、`Eval/`、`WebArena-Env-Setup/`）。**框架：verifiers（willccbb/verifiers,"RL with LLMs in Verifiable Environments",GRPO-based,Accelerate + DeepSpeed ZeRO3）**——在其上扩展为 M-GRPO + 并行 rollout;评测/环境为 WebArena（`Eval/browser_env`,WebArena-Lite）。〔已核:原始 shortlist 标 framework=unknown 不准,据仓库实证为 verifiers 而非 veRL〕【v1 analysis 已核仓库;PDF 仅给 GitHub 链接（line 54-55）,框架细节来自仓库】。
- 💰 资源/成本与可扩展性：原文正文未给 GPU 卡时/wall-clock 数字。可观测:并行 \(G\) 个独立浏览器实例(rollout 并行化降墙钟);动态上下文压缩是降显存、使多轮 RL 可行的关键。仅 3B/8B 级 backbone（Qwen2.5-3B、Llama3.1-8B）【原文 §2.3,成本数字原文未说明】。
- 🎯 对"探索-巩固"对标：**竞品/参照（无 teacher 的多轮 agent RL）**——一句判定:这是"纯 on-policy 多轮 RL、无 teacher 监督"的对照系,与本项目"teacher 当稀疏脚手架"路线**正交互补**,但其 BC warm-up + 端到端 on-policy 结构与 idea 的"探索/巩固"有局部映射。依据:
  - 探索 = 并行 rollout + 试错(在 web 上多样交互发现有效路径);
  - 巩固 = BC warm-up 把基本技能固化、RL 把成功策略压进参数。
  - **可借组件**：
    1. 动态上下文压缩 + loss mask 只算动作 token——对多轮/长 horizon 的 agent 自进化训练是通用工程基建;
    2. "BC warm-up 不可省"(R1-Zero 反例)佐证本项目 idea 里"student 应先有一个能走通的初始策略,再 on-policy 自选恢复分支"的必要性;
    3. test-time scaling 走交互深度——提示巩固阶段应保留"多轮回轨"的能力,而非压成单步。
  - **缺口**：
    1. 无 teacher、无稠密 token 监督(与本项目核心的脚手架蒸馏相反,纯稀疏二值奖励);
    2. 无 path-recovery 的显式单点接管机制(靠 RL 隐式学);
    3. 无记忆/技能库、无 MTP 前瞻;
    4. long-CoT 在此**限制** RL 探索的发现,与本项目 long-CoT 重心形成张力——提示在 agent 多轮场景里 CoT 模式不宜过早确定化。
- 🔭 开放问题/未来方向：
  - 【原文】中间步奖励塑形（当前只有终态二值奖励 \(A_{i,j}\) 全 token 共享,作者列为 future work）;多模态/截图输入;更丰富/动态的动作空间;安全风险（CMS 误删等）需权限/确认机制【原文 §6】。
  - 【推断】把"long-CoT 限制探索"从假设升级为受控实验;把 group-relative 优势与中间步 credit assignment 结合（与本项目 token 信用分配/MTP 前瞻可对接）;从沙箱迁到线上真实站点的鲁棒性。

minimax_m1 | MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention | MiniMax (MiniMax-AI) | 2025-06-16(arXiv 2506.13585, v1);技术报告(开源权重发布) | 主题线 L3 RLVR/GRPO(算法 CISPO)+ L6 长 CoT/前瞻·相关性 高

**原始论文**:https://arxiv.org/abs/2506.13585

## 一眼看懂
> 一句话导读:首个开源的"大规模高效长思考"模型,核心算法 CISPO 把 RL 里的"裁剪"从 token 更新挪到重要性权重上,保住所有反思 token 的梯度,让长 CoT 行为能稳定涌现。

- 🟦 TL;DR:首个开源权重的**大规模混合注意力推理模型**。这里"混合注意力"指 MoE(混合专家)叠加 lightning attention(一种线性注意力变体)。规模为 456B 总参 / 45.9B 激活,原生 1M 上下文;生成 10 万 token 时 FLOPs 仅 DeepSeek-R1 的约 25%。
  - 核心算法 **CISPO**:不像 PPO/GRPO 那样裁剪 token 更新,而是**裁剪重要性采样(IS,衡量新旧策略概率比)权重**。
  - 好处:**保留所有 token(尤其低概率的"反思/分叉"token)的梯度贡献**,对 DAPO 实现约 2× 加速。
  - 全量 RL 在 512×H800 上 3 周完成,约 \$53.5 万。【原文 Abstract, §3.1】
- 最巧的一步:**CISPO 把 clip 从"token 更新"移到"IS 权重"**(式4-5)。设想抽掉它(回到 PPO/GRPO 的 token clip):
  - 那些第一次 on-policy 更新后 IS 比值很大的低概率反思 token(However/Wait/Recheck/Aha,即推理路径的"岔口")就会被裁掉;
  - 它们在后续 off-policy 更新中拿不到梯度;
  - 于是长 CoT 行为难以涌现(尤其在该混合架构 + 16 轮 off-policy 更新下)。

## 为什么做
> 一句话导读:要让模型"想得更长"就得让注意力别那么贵(选混合架构),但这种架构上跑 RL 时,GRPO/DAPO 那套 token 裁剪会把关键的反思 token 误杀,所以得换裁剪方式。

- 研究背景:大推理模型(o1/R1)靠 RL 拉长 CoT 提升,但 softmax 注意力的二次复杂度让"持续加长推理"很贵。线性/稀疏注意力等替代方案虽多,但**几乎没在大规模推理模型上验证过**(唯一例外 Hunyuan-T1 用 Mamba 但闭源)。MiniMax 要开源一个能高效扩 test-time compute 的大推理模型。【原文 §1】
- 解决的具体痛点(三层):
  - (1) 架构侧——长生成下传统注意力 FLOPs 爆炸;
  - (2) 算法侧——GRPO/DAPO 的 token clip 会**裁掉对稳定熵/可扩展 RL 至关重要的低概率 token**,在混合架构 + 多轮 off-policy 更新下尤其严重;
  - (3) 工程侧——混合架构 RL 训练有训练/推理精度失配等独有坑。【原文 §3.1, §3.2】
- 相关工作 & 各自不足(站谁肩上 + 精确差异):
  - **PPO**(Schulman 2017):用 value model 估优势 + token 级 clip,本文式(1)给出其标准形式;问题是需独立 critic,且 clip 丢低概率 token。
  - **GRPO**(Shao 2024):去 critic、用组内归一优势 \(\hat A_{i,t}=(R_i-\text{mean}\{R_j\})/\text{std}\{R_j\}\)(式2)——CISPO 沿用这个优势但**换掉 clip 对象**。
  - **DAPO**(Yu 2025):用 Clip-Higher(提高 clip 上界)缓解低概率 token 被裁。本文实测在"16 轮 off-policy 更新"设定下**不够用**(Fig.2:CISPO 半步追平 DAPO)。CISPO 的 Δ 是**彻底改变 clip 的作用对象**(IS 权重而非 token 更新),不丢任何 token 的梯度。本文还沿用 DAPO 的 dynamic sampling 与 length penalty、去 KL 项。
  - **熵稳定/低概率 token 重要性**(Cui 2025; Wang 2025):支撑"低概率分叉 token 对稳定熵和可扩展 RL 关键"——CISPO 把这一洞察落到算法。
- 动机链(逐步推):
  - 要高效长 CoT → 选混合注意力(省 FLOPs);
  - 但该架构 RL 不稳 + token clip 丢关键反思 token → 提 CISPO(clip IS 权重、保全 token 梯度);
  - 再修精度失配/优化器/重复截断;
  - 配多域数据课程做大规模 RL。
- 与最近邻工作的Δ:相比 DAPO 在 token clip 框架内"放宽上界",CISPO 的 Δ 是**彻底改变 clip 的作用对象(IS 权重而非 token 更新)**,在保证不丢任何 token 梯度的同时仍约束方差;并首次在混合 MoE+lightning attention 大模型上跑通大规模 RL。

## 怎么做(到可复现)
> 一句话导读:CISPO 不裁 token 更新、只裁 IS 权重,并把它当 stop-gradient 系数乘到 log-prob 梯度上——方向照旧、幅度有界、绝不归零;此外混合架构跑 RL 还得修四个工程坑。

### 总体流水线(§2-§5,输入→输出)
1. **持续预训练**:MiniMax-Text-01 再 +7.5T token(STEM/代码/推理占 70%,四阶段把上下文扩到 1M)。
2. **冷启动 SFT**:注入反思型 CoT 行为(数学/代码占 ~60%)。
3. **CISPO 大规模 RL**:多域数据(数学 ~50K、逻辑 SynLogic 53K、竞赛代码 30K、SWE 沙盒、通用域用 GenRM)+ 课程(先 rule-based 可验证、再混通用域,防灾难性遗忘)。
4. **分阶段扩生成长度**:40K→80K(窗口逐级扩 48/56/64/72/80K)。
5. 输出 MiniMax-M1-40k / 80k。

### CISPO 核心算法(真实形式 + 直觉)
- **起点:带 IS 修正的 PPO(式1)**:
\(\displaystyle J_{\text{PPO}}(\theta)=\mathbb{E}_{q\sim D,\,o_i\sim\pi_{\theta_{old}}}\Big[\tfrac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\big(r_{i,t}(\theta)\hat A_{i,t},\,\text{clip}(r_{i,t}(\theta),1{-}\epsilon,1{+}\epsilon)\hat A_{i,t}\big)-\beta D_{KL}(\pi_\theta\|\pi_{\text{ref}})\Big],\)
IS 权重 \(r_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})}\)。
- **诊断:token clip 的问题**——反思 token(However/Recheck/Wait/Aha,推理"岔口")在 base 概率低,更新后 \(r_{i,t}\) 很大,**第一次 on-policy 更新后就被 clip 掉、后续 off-policy 更新拿不到梯度**;混合架构 + 16 轮 off-policy 更新下尤其严重。DAPO 的 Clip-Higher 在此设定下不够用。
- **CISPO 的形式(式3→4→5)**:从带停梯度 IS 修正的 REINFORCE 出发——
\(\displaystyle J_{\text{REINFORCE}}(\theta)=\mathbb{E}\Big[\tfrac{1}{|o_i|}\sum_{t}\text{sg}(r_{i,t}(\theta))\,\hat A_{i,t}\,\log\pi_\theta(o_{i,t}|q,o_{i,<t})\Big],\)
**不裁剪 token 更新,改为裁剪 IS 权重**(并采 GRPO 组内优势 + token 级 loss):
\(\displaystyle J_{\text{CISPO}}(\theta)=\mathbb{E}\Big[\tfrac{1}{\sum_{i=1}^{G}|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\text{sg}(\hat r_{i,t}(\theta))\,\hat A_{i,t}\,\log\pi_\theta(o_{i,t}|q,o_{i,<t})\Big],\quad \hat r_{i,t}(\theta)=\text{clip}\big(r_{i,t}(\theta),\,1{-}\epsilon^{IS}_{low},\,1{+}\epsilon^{IS}_{high}\big).\)
  - **实操**:不设 IS 下界(把 \(\epsilon^{IS}_{low}\) 设得很大使下界失效),**只调上界 \(\epsilon^{IS}_{high}\)**;无 KL 项;无 IS 权重裁剪时 CISPO 退化为标准策略梯度。
  - **直觉**:\(\text{sg}(\hat r_{i,t})\)(sg = stop-gradient,停梯度)当系数乘到 \(\log\pi_\theta\) 的梯度上。梯度**方向仍朝该 token**,幅度被 \([\,\cdot\,,1+\epsilon^{IS}_{high}]\) 限住,但**不归零**;所以反思/分叉 token 始终参与学习、熵更稳。
  - **代价**:权重裁剪带来轻微梯度偏置(作者承认),换来全 token 贡献 + 降方差。
- **统一公式(式6-7,把 PPO/GRPO/DAPO/CISPO 写成一家)**:在 CISPO 上加 token-wise mask \(M_{i,t}\):
\(\displaystyle J_{\text{unify}}(\theta)=\mathbb{E}\Big[\tfrac{1}{\sum_i|o_i|}\sum_{i}\sum_{t}\text{sg}(\hat r_{i,t}(\theta))\,\hat A_{i,t}\,\log\pi_\theta(o_{i,t}|q,o_{i,<t})\,M_{i,t}\Big],\)
\(\displaystyle M_{i,t}=\begin{cases}0 & \hat A_{i,t}>0\ \text{且}\ r_{i,t}(\theta)>1+\epsilon_{high}\\ 0 & \hat A_{i,t}<0\ \text{且}\ r_{i,t}(\theta)<1-\epsilon_{low}\\ 1 & \text{otherwise}\end{cases}\)
PPO/GRPO 是 \(M_{i,t}\) 取 PPO trust-region mask 的特例,CISPO 是 \(M_{i,t}\equiv1\) 的特例。

### 混合架构 RL 的四个工程修复(逐项:症状→根因→处方,均带数字)
- **① LM head 升 FP32**(修训练/推理概率失配):
  - 症状:训练 vs 推理 kernel 精度差,反映在 rolled-out token 概率不一致,导致 **reward 涨不起来**;
  - 根因:逐层定位为 LM output head 的高幅激活;
  - 处方:升 FP32 后训练-推理概率相关性 ~0.9x→**0.99x**(Fig.3,Pearson 0.987→0.997)。**小 dense softmax 模型不出现**,混合架构才有。
- **② AdamW 超参**:梯度幅度跨 1e-18~1e-5(多数 <1e-14)、相邻迭代相关弱;VeRL 默认 \(\beta=(0.9,0.999),\epsilon=10^{-8}\) 会不收敛 → 改 **\(\beta_1=0.9,\beta_2=0.95,\epsilon=10^{-15}\)**。
- **③ 重复检测早停**:复杂 prompt 诱发病态长重复、大梯度毁稳定;一旦进入重复循环 token 概率飙高 → 规则:**连续 3000 token 概率均 >0.99 即截断生成**(防不稳 + 提吞吐)。
- **④ 长度扩展三招**:修长生成下"负样本比正样本长得快→后段堆积负梯度→模式塌缩/乱码"——重复早停 + sample-level/token-level 混合归一 + 降 clip 阈值与 \(\epsilon^{IS}_{high}\)。

### 数据流动一句话
混合架构高效 rollout 采 G 条 → GenRM/rule verifier 给奖励 → 组内归一成优势 → 对每 token 用 \(\text{sg}(\)裁剪 IS 权重\()\times\)优势加权 \(\log\pi_\theta\) 梯度(全 token 不丢)→ 16 轮 off-policy 更新 → 课程逐步混入通用域。

## 靠不靠谱
> 一句话导读:CISPO 的 2× 加速主要在受控小实验里干净给出,但大模型上它和架构/数据/工程修复深度耦合、难单独归因;\$53.5 万 + 512 卡的"高效"是相对而非平民可及。

- 逐组件证据:
  - **CISPO** 消融(Fig.2,Qwen2.5-32B-base/AIME2024 zero-RL):同步数下 CISPO > DAPO > GRPO,且 **CISPO 用 50% 步数即追平 DAPO**(2× 加速);
  - FP32 修复前后对比(Fig.3)清晰;
  - 优化器/重复早停为系统级处方(有现象佐证,非逐项理论拆解)。
- 实验与证据:MiniMax-M1-40k/80k。
  - Table 2:AIME2024 86.0、SWE-bench Verified 56.0、TAU-bench(airline)62.0(超 Gemini-2.5 Pro)、长上下文 MRCR/LongBench 多项超 o3/Claude4;
  - 整体超原版 DeepSeek-R1、Qwen3-235B;对 R1-0528 在数学/代码竞赛略逊但 agent/长上下文更强;
  - **"看着强但没回答核心问题"**:CISPO 优势主要在受控小实验给出,大模型上 CISPO 与架构/数据/工程修复深度耦合,难单独归因 CISPO 贡献多少。
- 假设与失效边界:
  - 【原文 §3.2】精度失配/优化器敏感等问题是混合 lightning-attention 架构**特有**,dense softmax 小模型不出现 → 这些 recipe 对标准架构未必必要。
  - 【推断】CISPO 的"保全 token 梯度"在**强反思先验 + 长 CoT** 设定下收益最大;短输出/无明显反思 token 任务收益可能不明显;
  - 【推断】结论建立在超大规模工业算力上,小规模复现不一定重现 2× 加速;轻微梯度偏置的长期影响未充分分析。
- 祛魅总结:
  - 【推断】真贡献=**CISPO(clip IS 权重以保全所有 token 尤其低概率反思 token 的梯度)**——一个干净、可移植到任意策略梯度框架的算法点子,且坐实"低概率分叉 token 对长 CoT 涌现的重要性";叠加首个开源大混合注意力推理模型的工程闭环。
  - 被高估处:很多增益来自架构/长上下文/数据课程,CISPO 单独贡献被打包难拆;\$53.5 万、512 H800 门槛使"高效"是相对而非平民可及。
  - 被低估处:CISPO 的"用 sg(IS) 当系数、保全梯度"形式与本课题关注的"forward-hard/backward-soft 解耦"思想同构(见对标)。

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号 = 可验证奖励(数学/代码/逻辑 rule-based)+ GenRM(通用域);CISPO 改变的是**怎么用**这些奖励的梯度(保全低概率 token)。
  - 改什么 = 策略参数(MoE 全参 RL)。
  - 何时改 = on-policy RL 训练期,16 轮 off-policy 更新/批;课程式从可验证域逐步混入通用域。
  - 免梯度? = 否(策略梯度 + IS 权重裁剪)。
  - 记忆-技能生命周期 = 无外部记忆/技能库;能力固化进参数;靠课程 + 通用域混入防遗忘。
  - 防遗忘机制 = 数据课程(先可验证后通用,§4 明说"防灾难性遗忘");无显式蒸馏/正则项。
- ⑦ 开源代码+框架/harness:https://github.com/MiniMax-AI/MiniMax-M1(model-release/权重仓,**CISPO 训练代码不在仓内**,本地未 clone,Tier B paper-only;推理支持 vLLM/Transformers)。框架=**自研 RL 系统**(借 lightning attention 高效 rollout;§3.2 明确提 VeRL 默认优化器配置不适用,即知其参照过 veRL)。**未完整克隆**:训练代码未开源,仅模型权重仓,需手动从 HF 获取权重;代码链接见上。【原文 §3.2 提 VeRL 默认配置 + 元信息】
- 💰 资源/成本与可扩展性:**全量 RL = 512×H800 × 3 周 ≈ \$53.5 万**(原文明确给出);推理 FLOPs 在 100K 生成长度时 ≈ R1 的 25%(架构优势)。【原文 Abstract, §1, §3】
- 🎯 对"探索-巩固"对标:**中等支撑(探索侧机制 + 思想同构),非巩固/蒸馏直接竞品**。
  - CISPO 对应"探索/选路"侧:**保护低概率分叉 token = 不过早扼杀'可能走通的另一条开头'**,与本课题"探索=发现有效路径、偏向自己能走通的开头"高度契合;Clip-Higher/熵稳定同理。
  - **思想同构**:CISPO 用 \(\text{sg}(\hat r_{i,t})\) 当系数、对 \(\log\pi_\theta\) 求梯度——保留方向、限制幅度、不归零,正是 MEMORY 里"forward-hard/backward-soft 解耦(前向保留硬操作、反向流有界平滑梯度)"的一个实例,可直接借鉴到 MTP/OPD 的梯度设计。
  - **缺口/区别**:它是纯 RLVR(无 teacher 蒸馏、无 path-recovery、无 MTP),且"保全 token"是全局均匀策略而非针对关键分叉步定位。
  - 一句判定:CISPO 是"探索侧不丢分叉 token"的优秀算法原件 + forward-hard/backward-soft 的现成范例,但缺巩固/teacher 脚手架/前瞻三件套。
- 🔭 开放问题/未来方向:
  - 【原文 §9】更合适的 loss/优化算法、用模型自身推理轨迹 bootstrapping、扩到下一量级算力、tool-use/多模态/agent。
  - 【推断】(1) 把"保全低概率 token"从全局升级为**关键分叉步定向保全**(配高熵检测),省算力且贴 path-recovery;(2) 把 CISPO 的 sg(IS)-加权梯度形式迁移到 on-policy 蒸馏(teacher 概率比当系数),统一 RLVR 与 OPD;(3) 用 MTP 前瞻预测"该低概率 token 是否真是有效分叉"以决定保不保;(4) 轻微梯度偏置的理论刻画。

— 残留待核:0

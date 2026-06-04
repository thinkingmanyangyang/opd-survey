one_shot_rlvr | Reinforcement Learning for Reasoning in LLMs with One Training Example | UW · USC · Microsoft · UC Santa Cruz · Georgia Tech(Yiping Wang*, Qing Yang, Zhiyuan Zeng, …, Jianfeng Gao, Weizhu Chen, Shuohang Wang*, Simon S. Du*, Yelong Shen*) | 2025-10-24 · arXiv 2504.20571 v3 · NeurIPS 2025 | 主题线 L3(RLVR/GRPO·数据效率+机理)，兼 L6(自反思/CoT) · 相关性 高

**原始论文**：https://arxiv.org/abs/2504.20571 （代码 https://github.com/ypwang61/One-Shot-RLVR）

## 一眼看懂
> 一句话导读：RLVR 是"用规则验证器给对错打 0/1 奖励来做 RL"的简称。本文的暴论是:做这种 RL 训练,训练集压到**只剩一道题**都几乎不掉点。由此推出一个观点——RL 并没有给模型灌入新能力,只是把 base 模型本来就埋着的推理能力"点着了"。

- 🟦 TL;DR：**只用一条训练样本**做 RLVR(用 GRPO 或 PPO),就能把 Qwen2.5-Math-1.5B 在 MATH500 上从 36.0% 提到 73.6%、6 个数学基准的均值从 17.6% 提到 35.7%。这几乎追平了包含该样本的 1.2k DSR 子集(它是 73.6/35.9);用 2 条样本(74.8/36.6)甚至略超 1.2k 子集、追平 7.5k 的 MATH 全集。配套还发现一串现象:
  - post-saturation generalization:训练准确率很早就饱和到 100%,但测试准确率仍在持续上涨;
  - 跨类泛化(在数学单题上训练,却能提升非数学任务);
  - 自反思词频与响应长度都上升;
  - 增益主要来自 policy gradient loss(这点区别于 grokking 现象);
  - 适度的 entropy loss(熵正则,鼓励输出更多样)有助于探索。
  这些一起强力支持一个判断:RLVR 主要是**激发**base 模型已有的推理潜能,而不是**注入**新能力。【原文 §Abstract / §1 / Fig.1-2 / §4】
- 最巧的一步：关键不在算法,而在**把训练集压到 1 条**这个极限实验设计本身(再配上 Historical Variance 选样)。如果不压到极限、仍用上千条数据,就观察不到 post-saturation generalization(那时训练准确率一直没饱和,测试反而先掉,见 Fig.2 右),也就没法把"数据量"这个变量从增益里剥离出来,更没法论证"能力本就埋在 base 里、RL 只是点火"。所以**正是这个极简化,才让机理结论变得可见**(类似 dr_grpo 用极简配方让偏置暴露出来)。【原文 §3.2.2 / Fig.2】

## 为什么做
> 一句话导读：大家都知道 RLVR 能让模型数学变强,但精力几乎都花在"怎么调稳算法"上,没人认真问"到底要多少数据"。本文反过来死磕数据侧——把训练集一路砍到只剩一条,看它什么时候才崩,顺便看清增益到底从哪来。

- 研究背景：RLVR(用规则化的二元 outcome reward——即只按最终答案对错给 0/1 奖励——来做 RL)已经大幅推进了 LLM 的数学推理(o1、DeepSeek-R1、Kimi-1.5),并伴随自反思等认知行为的涌现,以及跨任务泛化。但社区的精力大多放在**算法侧**(调稳 PPO/GRPO/DAPO),**数据侧**(到底要多少数据、什么数据最有效)相对被忽视。【原文 §1】
- 解决的具体痛点：
  - ① RLVR 的训练集究竟能压缩到什么极限,尚未有人探明;
  - ② 数据的质量/数量,如何关联到自反思、跨任务泛化等经验现象,也不清楚。
  最相关的前作 LIMR 用 LIM 分数把数据减到约 1/6 仍保性能,但没继续探"还能减到多狠"。【原文 §1】
- 相关工作 & 各自不足（来龙去脉 + 并行路线 + 各自短板 + 站谁肩上 + 最近邻精确差异）：
  - **数据效率/选样**(LIMR(Li 2025) 用 LIM 分数减约 1/6;LIMO 做少样本 SFT):这是**最近邻**。精确差异:LIMR 减到约 1/6 就停了,本文一路推到**1 条**,从而暴露出只在单样本下才看得见的 post-saturation 现象。
  - **"base 已具推理能力"系列**(Liu 2025 / Yue 2025 / Gandhi 2025):它们提出了"RLVR 是激发而非注入"的主张,但**没有用"单样本即可"这种强证据来佐证**;本文给出了最锋利的单点实证——1 条样本 ≈ 1.2k ≈ 7.5k。
  - **算法侧的大量工作**(精炼 PPO/GRPO/DAPO、调稳):它们与数据效率**正交**(讨论的是不同维度);本文表明,在固定的标准 GRPO 下,单样本就已足够,从而把焦点从算法转向数据。
  - **站谁肩上**:
    - ① GRPO(Shao 2024)/PPO(Schulman 2017)的标准实现(verl pipeline);
    - ② "奖励信号的方差对 RL 训练很关键"这一发现(Razin 等)——它启发了本文的 Historical Variance 选样;
    - ③ 自反思/认知行为涌现的观察(Gandhi 2025 等)。
  - **共性缺口**:RLVR 的**数据下限**究竟在哪、以及"激发 vs 注入"的机理,此前无人系统回答。【原文 §1 / §2】
- 动机链（一步步推下来）：
  - 现状:RLVR 靠大数据集,增益机理被"涌现"这套叙事笼统解释过去;
  - 问题:训练集到底能减到多少还不掉点?
  - 实验发现:减到 1 条,效果仍约等于全集;
  - 推断:既然单样本就能激发,说明能力本就在 base 里,于是 RLVR 偏向"激发/对齐"而非"灌输";
  - 结论:把研究重心从算法复杂度转向数据效率与机理。【原文 §1】
- 与最近邻工作的 Δ：
  - 相对 **LIMR**:把"减到约 1/6"推到"减到 1 条",差别在于它**揭示了数据量几乎可压到零**这一质变,并由此暴露出 post-saturation 等只在单样本下可见的现象。
  - 相对"base 已具能力"系列:本文给出了**最锋利的单点证据**(1 条 ≈ 1.2k ≈ 7.5k),而且跨模型(Qwen-Math-1.5B/7B、Llama-3.2-3B、R1-Distill)、跨算法(GRPO/PPO)、跨样本都一致。【原文 §1 / Fig.1 / §3.3】

## 怎么做 + 靠不靠谱
> 一句话导读：流程很短——先用全集训一遍、记录每道题"训练准确率忽高忽低的程度"(方差)挑出最值钱的几道题,把单道题复制成一整个 batch,然后就用最标准的 GRPO 去训。靠不靠谱的关键在于:很多题(不只挑出的那几道)单独训都能涨,说明这是普遍现象而非个例;但它强依赖 base 本就会做这类题。

- 方法流水线（输入→输出,读完即可复现）：输入是 base 模型 + 单条数学题 + 一个二元 verifier。流程分四步:
  - ① **选样**:先用全集(DSR-sub 共 1209 条,从 DeepScaleR-Preview 随机取)做 RLVR、训 \(E\) 个 epoch(实操训 500 步);对每个样本 \(i\) 记录一个"逐 epoch 的训练准确率列表" \(L_i=[s_{i,1},\dots,s_{i,E}]\);算它的**方差(即 Historical Variance Score,Eq.1)**;按方差降序排名(Eq.2),高方差样本如 \(\pi_1/\pi_{13}\);
  - ② **复制凑 batch**:把这单条样本复制成 128 份来凑够一个 batch(因为 verl 的 dataloader 设了 `drop_last=True`,要求数据量 ≥ batch);
  - ③ **标准 verl GRPO**:用二元 0/1 outcome reward、KL 系数 \(\beta=0.001\)、熵系数 \(\alpha=-0.001\)、rollout 温度 0.6(vLLM 采样)、batch = mini-batch = 128、每个 prompt 采 8 条(于是每个 rollout 步做 8 次梯度更新)、max prompt 1024 / response 3072(因为 Qwen2.5-Math 上下文是 4096);
  - ④ **评测**用 Qwen2.5-Math 的官方 pipeline。【原文 §2 / §3.1 / Appendix B.1】
- 逐组件必要性（每模块干啥 / 有无消融）：
  - **单样本(1-shot)本身**:这是核心对象。Tab.3 显示**很多样本**(高、中、低方差都有)单独训练都能涨 ≥30%(只有错标的 \(\pi_{1207}\)、以及超难的 \(\pi_{1208}\) 例外)——说明这是**普遍现象**,而非个别特例。【原文 §3.2.3 / Tab.3】
  - **Historical Variance Score 选样**:**既非必要、也非最优**——作者明确说许多中低方差的样本也照样行;它只在 Qwen2.5-Math-7B 上略胜随机选(Tab.4)。**有消融**(隐含在"很多样本都行"这个对照里)。【原文 §2 脚注 2 / §3.2.3 / Tab.4】
  - **policy gradient loss**:这是增益的**主因**。消融(Tab.5,以 π1 为例):Row1 不加任何 loss 是 39.8;Row2 只加 PG loss 就到 71.8/15.4,已接近完整 loss;再加 weight decay(Row3,71.4)或 KL(Row4)几乎没影响。【原文 §4.1 / Tab.5】
  - **entropy loss(适度)**:有助于 post-saturation——不加熵 loss 时,训练准确率到饱和点后几乎就不再涨了;加上熵 loss,MATH500/AIME24 又能继续涨(Tab.5 Row5)。但**系数过大反而不稳**(Row6,设为 −0.003)。**有消融**。【原文 §4.1 / Fig.5】
  - **复制凑 batch**:这是纯工程上的必要(因为 verl 设了 `drop_last=True`),不是方法贡献。【原文 §3.1 脚注 3】
- 关键机制/公式（真实形式 + 直觉,均从 PDF 抄准）：
  - **Historical Variance Score(Eq.1)**:对样本 \(i\),取它在全集 RLVR 中那串逐 epoch 训练准确率 \(L_i=[s_{i,1},\dots,s_{i,E}]\) 的方差 \(\displaystyle v_i := \mathrm{var}(s_{i,1},\dots,s_{i,E}).\) 直觉:如果某道题的训练准确率忽高忽低(方差大),说明它"刚好卡在模型会与不会的边界",会反复给出有用的对错信号。
  - **排名(Eq.2)**:定义一个排列 \(\pi:[N]\to[N]\),让它满足 \(v_{\pi(1)}\ge\cdots\ge v_{\pi(N)}\)(即把方差从高到低排),记 \(\displaystyle \pi_j := \pi(j)=\operatorname*{arg\,sort}_{j}\{v_l: l\in[N]\}.\) 作者反复强调这**不是关键**——关键是"几乎任何不太难、也没标错的题"都能点火。
  - **GRPO 总损失分解(Appendix B.1,Eq.3)**:\(\displaystyle L_{\text{GRPO}}(\theta)=\mathbb E_{q\sim P(Q),\,\{o_i\}\sim\pi_{\theta_{\text{old}}}}\big[L'_{\text{PG-GRPO}}+\beta\,L'_{\text{KL}}+\alpha\,L'_{\text{Entropy}}\big],\quad \beta>0,\ \alpha<0.\)
  - **policy gradient 项(Eq.4,序列级 clipped surrogate)**:\(\displaystyle L'_{\text{PG-GRPO}}=-\tfrac1G\textstyle\sum_{i=1}^G\min\!\Big(\tfrac{\pi_\theta(o_i\mid q)}{\pi_{\theta_{\text{old}}}(o_i\mid q)}A_i,\ \mathrm{clip}\big(\tfrac{\pi_\theta(o_i\mid q)}{\pi_{\theta_{\text{old}}}(o_i\mid q)},1-\varepsilon,1+\varepsilon\big)A_i\Big).\)
  - **KL 项(Eq.5,k3 近似)**:\(\displaystyle L'_{\text{KL}}=D_{\text{KL}}(\pi_\theta\|\pi_{\theta_{\text{ref}}})=\tfrac{\pi_{\theta_{\text{ref}}}(o_i\mid q)}{\pi_\theta(o_i\mid q)}-\log\tfrac{\pi_{\theta_{\text{ref}}}(o_i\mid q)}{\pi_\theta(o_i\mid q)}-1.\)
  - **组内归一化优势(Eq.6)**:\(\displaystyle A_i=\frac{r_i-\mathrm{mean}(\{r_1,\dots,r_G\})}{\mathrm{std}(\{r_1,\dots,r_G\})},\quad i\in[G],\) reward \(r_i\) 为 0-1 准确率(答对为 1)。
  - **机理直觉**:base 模型(尤其是 Qwen-Math)在预训练里就几乎会做 \(\pi_1\) 这道题了——Fig.3 显示,除了最后一步开立方,base 已经能完成所有关键步,且 128 次采样中有 57.8% 输出了 "12.7"/"12.70"。所以 RLVR 做的只是**把已有的正确推理路径的概率抬上来,再鼓励探索其它表达方式**。这就解释了为什么单样本够用、还能跨类泛化:被点火的是"通用的推理表达",而不是"这道题的具体知识"。
  - **post-saturation 的诡异之处**:训到 step1860 时,模型对**训练样本**的输出退化成"多语言乱码夹杂正确算式"(Fig.3 里 cyan 色标出了其中正确的过程),但它对**测试样本**的输出仍然正常可读、准确率仍有 74%。
    - **作者给出的机制**(注意:这与既有 v2 里"zero-mean advantage 抑制梯度"的框法相反):之所以饱和后还能继续涨,是因为训练准确率其实只到约 99.x%、并非 100%;偶发的错误会让该 batch 的 advantage 因组内方差很小而变大、于是梯度非零;再配合 entropy loss 持续探索。所以 post-saturation 靠的是"残余非零梯度 + 探索",而不是"梯度自动衰减"。【原文 §4.1(offset 117145 处原文:"non-zero gradients (advantage becomes large for batches with wrong responses due to small variance)") / §5 / Fig.3】〔此说法与既有分析的"zero-mean advantage anti-overfitting"提法不一致,以原文此句为准〕
- 实验与证据：
  - **核心并列(Fig.1 / Tab.3)**:1-shot 的 \(\{\pi_{13}\}\)=35.7%,约等于 1.2k DSR-sub 的 35.9%;2-shot 的 \(\{\pi_1,\pi_{13}\}\)=36.6%,略高于 DSR-sub、约等于 7.5k MATH 的 36.7%。在 MATH500 上:1-shot 的 \(\{\pi_1\}\)=74.0、\(\{\pi_{13}\}\)=74.4,对比 DSR-sub 的 75.2、MATH 的 75.4(基本追平)。【原文 Fig.1 / Tab.3】
  - **跨模型/算法/样本**:Qwen2.5-Math-1.5B/7B、Llama-3.2-3B-Instruct、R1-Distill-Qwen-1.5B 全都见到提升;GRPO 与 PPO 都行;Tab.3 里多个样本都涨;而且 \(\{\pi_1,\dots,\pi_{16}\}\) 优于 16 个随机样本。【原文 §3.3 / Abstract / Tab.3】
  - **非数学泛化(Tab.1)**:在数学单样本上训练,竟能提升 ARC-Easy/Challenge,而且**超过用全集做 RLVR**——比如 \(\{\pi_{13}\}\) 在 ARC-E 上是 55.8,而 MATH 全集只有 51.6、DSR-sub 只有 42.2。【原文 Tab.1】
  - **entropy-only(Tab.6)**:**只加熵 loss、完全不给 outcome reward**,也能涨(1.5B 在 M500 上 36.0→63.4),但弱于 format-reward baseline 的 65.6;Llama/7B 也是类似情况(只在前几步有效)。说明"鼓励探索"本身就带来(弱)增益。【原文 §4.2 / Tab.6】
  - **label 鲁棒性(Tab.5 Row11-13)**:把标签从 12.8 改成更精确的 12.7,性能基本不变;改成 "4"(一个可被猜中、可被过拟合的错标)反而最差;改成 "9292725"(一个完全猜不到的错标)则居中(约等于 entropy-only)。说明轻微的标签误差无害,但"可过拟合的错标"危害比"完全错的标"还大。【原文 §4.2 / Tab.5】
  - baseline 公平吗:用 format-reward baseline(只奖励"能解析出答案")来对照,以剥离掉"格式修正"带来的增益,较严谨;全集/子集/单样本都用同一 pipeline、同一模型对比,归因清晰。
  - "看着强但没回答核心问题":核心结论(数据可压到 1 条 + 激发而非注入)证据非常充分;但"为何输出成乱码却仍能泛化"只给了观察加上面那个"残余非零梯度"机制,没给完整理论。
- 假设与失效边界：
  - 【原文 §3.2.3 / Tab.3】**会失效的样本**:错标的(\(\pi_{1207}\))、以及超难到模型几乎采不到正确答案的(\(\pi_{1208}\))——一旦没有有效的 policy gradient 信号就会失效。
  - 【原文 §3.3 / Tab.1 / Appendix C.1】Llama-3.2-3B-Instruct 上增益相对有限,而且 RLVR 过程不稳——这暗示:**对于预训练较弱、或非数学专长的 base,"激发"这套叙事要打折扣**。
  - 【推断】现象集中在 **Qwen2.5-Math 系**(它的 base 已在数学语料上充分预训练)。"激发已有潜能"这一说法,在数学潜能本就很强的 base 上才最成立;对真正缺乏该能力的模型,单样本根本无从"激发"。依据是:跨模型实验里 Qwen-Math 增益最大、Llama 最小。
  - 【推断】**选样阶段并不省算力**:Historical Variance Score 必须先跑完全集 RLVR 500 步才能算出来,所以"1-shot 省下的"只是 RL 训练阶段的数据,而不是端到端的算力。依据是 §2/§3.1 的选样流程。
  - 【原文 §3.2.2 + 推断】post-saturation 下,训练样本的输出退化为多语言乱码却仍能泛化——作者只给了观察(并归因于残余非零梯度 + 探索),**机理并未充分解释**;而且过度训练(>1.4k 步)对超参(熵系数、步数)很敏感。
- 祛魅总结【推断】：
  - 真贡献:**数据效率的极限证据**(1 条样本 ≈ 全集)干净、可复核、冲击力强,且跨模型/算法/样本都稳健,已成为"base 已具能力、RLVR 是激发"这一观点的标志性实证。post-saturation、跨类泛化、自反思增多等现象也很丰富,而 entropy-only 与 label 鲁棒性这两组消融把功劳定位到了 policy gradient loss + 探索,而非数据量或正则。
  - 包装/被高估处:
    - ① **"1-shot 省算力"容易被误读**——它省的是 RL 训练数据,不是选样算力;
    - ② **强依赖 Qwen-Math 系**,通用性有边界;
    - ③ "self-reflection 增多"只是个相关性观察(而且 dr_grpo 已指出"自反思 ≠ 更准"),不宜过度解读成"学会了反思";
    - ④ "single example sufficient"这个措辞,掩盖了一串前提——题要"不太难、标注正确、且 base 已接近会做"(见 Tab.3 里那些失效样本)。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：规则验证器的二元 0/1 outcome reward(数学题答案对错);policy gradient(GRPO/PPO)的组内归一化优势 \(A_i\)(Eq.6)驱动。属 RLVR,无教师/无蒸馏。
  - **改什么**：改模型参数(全量 RL 微调);不改结构/不加记忆。本质是"把 base 已有正确推理路径的概率抬高"。
  - **何时改**：在线 RL——每步对单样本采 8 条→算优势→8 次更新;训练可持续到百万次 rollout(单样本采 1024 次/步)。
  - **免梯度?**：否,核心是策略梯度;但揭示"policy gradient loss 是主因、weight decay/KL 可去",且 entropy-only(无 reward)也有弱增益。
  - **记忆-技能生命周期**：无外部记忆/技能库;"技能"=被激发的推理表达,固化进参数。无防遗忘设计——但意外展示 RLVR 抗过拟合性(训练样本被采百万次仍不退化测试性能,机制见上方 post-saturation)。
  - **防遗忘机制**：无显式机制;post-saturation 下"残余非零梯度 + 探索"使其在过拟合训练样本表面形式后仍保持测试泛化。
- ⑦ 开源代码+框架/harness：https://github.com/ypwang61/One-Shot-RLVR (已开源,本地已 clone ~70MB)。**框架=veRL**(改编自 verl + rllm/DeepScaleR,仓库内含完整 `verl/` 目录),算法 GRPO(也验证 PPO);推理 vLLM 0.6.3;评测复用 Qwen2.5-Math 官方 pipeline(`Qwen2.5-Eval/`,latex2sympy)。模型/数据 HF ypwang61/one-shot-rlvr、ypwang61/One-Shot-RLVR-Datasets。关键超参:\(\beta=0.001\)、\(\alpha=-0.001\)、温度 0.6、batch=mini-batch=128、每 prompt 8 rollout、max prompt 1024 / response 3072。代码可得性高。【v1 仓库核查 + 本批 repo ls】
- 💰 资源/成本与可扩展性：单样本复制成 128 凑 batch;1.5B/7B/3B 规模(context 4096);相对全集省 RL **数据**(非选样算力);训练可跑到百万 rollout(单样本)。具体卡数原文未集中说明。【原文 §3.1】
- 🎯 对"探索-巩固"对标：**中-强支撑(机理层:base 本就蕴含可激发的能力 + 探索本身有价值),外加两条警示;但它不是脚手架/蒸馏方法**。判定依据如下:
  - ① **"探索 = 发现自己能走通的开头"获得了最强背书**:Fig.3 显示 base 已能完成 \(\pi_1\) 几乎全部关键步、只在最后一步发散,而 RLVR 做的是把"自己本就能走通的路径"的概率抬上来。这与本课题"teacher 偏向 student 自己能走通的开头、student 在 on-policy 上自选"的设计**正面共振**——可激发的上限由 base 决定。
  - ② **entropy loss 有助于 post-saturation**,支撑了"探索是巩固能持续生效的前提"——巩固阶段如果不留探索,增益会随训练饱和而停下。
  - ③ **post-saturation 的抗过拟合(靠残余非零梯度 + 探索)**,给"巩固而不灾难性遗忘"提供了一个朴素的机制提示。
  - **可借组件**:
    - 把 1-shot/few-shot 当**探索探针**——用极少几条"自己几乎能走通"的种子样本来点火,再配 MTP 前瞻去挑选种子;
    - Historical Variance(卡在会与不会的边界,Eq.1)这套选样思路,可迁移为"挑临界路径/关键步"。
  - **缺口/警示**:
    - ① **无 teacher、无蒸馏、无回轨的单点接管**(它是纯 self-play RL);
    - ② **无记忆/技能库、无 MTP**;
    - ③ **警示一**:增益强依赖 base 已具备的能力——对"student 走不通的开头"它无能为力,这反过来证明了本课题"teacher 脚手架"在 base 能力不足处的必要性;
    - ④ **警示二**:post-saturation 下训练样本会退化为乱码——这提醒我们,"过度巩固单一路径"会侵蚀该路径的可读性/结构,本课题"把能力固化进参数"时需要防范这类塌缩。
- 🔭 开放问题/未来方向：
  - 【原文 §5 / Appendix D.4】更好的数据选择/采集;把结论推广到其它领域与更弱 base;理解 post-saturation 乱码却泛化的机理;探索的更优诱导方式。
  - 【推断】两个方向:
    - 用 MTP 前瞻去识别"base 临界可走通"的种子题、拿来做 1-shot 点火,再用 teacher 稀疏脚手架补上"走不通的开头";
    - 把 post-saturation 的"残余非零梯度 + 探索"这一性质,与"巩固防遗忘"显式结合,研究"巩固到何种程度、路径开始塌缩"的边界。

RETURN: one_shot_rlvr|读到PDF=是(§Abstract/§1-5全文+Fig.1-3/Tab.1/Tab.3/Tab.5-6/§4.1-4.2机理+Appendix B.1 Eq.1-6损失分解)|L线=L3(兼L6)|对标=中-强支撑(base蕴含可激发能力背书"走通的开头"、探索助巩固;1-shot可作探索探针;但无teacher/回轨/MTP,且强依赖base能力+过巩固致路径塌缩是警示)|残留待核=1(post-saturation抗过拟合机制:原文为"残余非零梯度+探索",与既有v2"zero-mean advantage抑制梯度"提法相反,已据原文§4.1更正并标注)

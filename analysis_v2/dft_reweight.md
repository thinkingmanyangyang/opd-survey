dft_reweight | On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification (DFT / Dynamic Fine-Tuning) | 东南大学·UCLA·上海交大·南洋理工·UC Berkeley·武汉大学·UC Merced 等(Yongliang Wu, Yizhou Zhou 等) | 2026-02-27 · arXiv 2508.05629 v3 · ICLR 2026 | 主题线 L2(统一SFT-RL视角)，兼 L3 · 相关性 高

**原始论文**：https://arxiv.org/abs/2508.05629

## 一眼看懂

> 一句话导读：标准 SFT 其实暗藏一个毛病——它对"模型当前几乎不会的那个正确词"会给出超大的梯度,容易学崩、还过拟合。本文用一行代码把这个畸变抵消掉(给每个词的损失乘上它自己的预测概率、再 detach),效果就稳了、泛化也更好。

- 🟦 TL;DR：先把标准 SFT 的梯度用重要性采样改写成"策略梯度"的形式,改完会发现它隐含的奖励长这样——是一个**稀疏的指示函数(只有精确匹配专家 token 时才 =1),并且被 \(1/\pi_\theta\)(逆概率)加权**。问题就出在这个逆概率权重上:当模型给某个专家 token 的概率很低时,这个权重会爆炸,导致梯度异常大,进而训练不稳、过拟合到那些罕见的精确匹配。
  - 修法极简:**给每个 token 的交叉熵损失乘上该 token 的预测概率 \(\pi_\theta\)(并 detach、阻断梯度)**,刚好把 \(1/\pi_\theta\) 抵消掉,使隐含奖励被整平成常数 1(类似 RLVR 给所有正确样本一个统一奖励)。就这一行代码,在数学 / 代码 / 多模态推理上都显著超过 SFT。【原文 §Abstract / §3.2–3.3 / Eq.5–9】
- 最巧的一步:**乘 \(\pi_\theta\),并对它做 stop-gradient(sg)**。这里有两个细节都不能省:
  - 一是 **sg 不能抽**:如果让梯度流过这个概率系数,这个乘性项就会额外引入一个"提高自身概率"的梯度,目标就不再是"整平奖励"、而变成别的东西了;§Eq.7 明确指出 sg 是为了保证"梯度不流过 reward scaling 项 \(w\)"(原文 "ensuring that gradients do not flow through the reward scaling term \(w\)")。
  - 二是必须**在 token 级**加权、而非句级:整句的概率是逐 token 连乘、数值极小,loss 几乎没有信息(Table 5:句级 15.75 ≈ base 15.92,几何均值 17.21,而 token 级达 31.58),只有 token 级才有效——这一点和 PPO 的 token 级重要性采样同源(原文 Eq.9 的注释就援引了 "as was adopted in PPO")。【原文 Eq.7 / Eq.9 / §4.6 Table 5】

## 为什么做

> 一句话导读：业界都知道"SFT 死记、RL 泛化",可 RL 又贵又得有奖励/负样本,很多场景只能用 SFT。所以本文问的是一个很实在的问题——能不能不上 RL、只改 SFT 本身,就把它的泛化救回来?

- 研究背景:SFT(去拟合专家 demonstration,类似机器人里的 behavioral cloning,原文 §2 明确说 "analogous to behavioral cloning in robotics")是 LLM 后训练的标准范式,简单又高效;但有个老观察叫"SFT memorizes, RL generalizes"(Chu et al. 2024)——SFT 容易过拟合,泛化弱于 RL。而 RL 虽然泛化好,却算力贵、需要显式奖励、调参敏感,并且在"只有正样本、没有负样本 / 没有奖励模型"的场景下根本用不了(原文 §1 末:"SFT remains the only viable option when datasets contain only positive demonstrations")。【原文 §1 / §2】
- 解决的具体痛点:**SFT 本身能不能被根本性地改进?**(原文用斜体设问 "can SFT itself be fundamentally improved?")——尤其是在数据只含正样本、上不了 RL 的那些场景。【原文 §1 末】
- 相关工作 & 各自不足(原文 §2,按"来龙去脉"梳理)：
  - **混合 SFT+RL**：两类做法——① InstructGPT 式(Ouyang 2022),先 SFT、再用学到的奖励模型做 RL refine;② 交错式(interleave SFT/RL updates,Sheng 2025 / Liu 2025 UFT / Qiu 2025 Metis-RISE),用交替更新来提稳定性和性能。共性短板:**它们丰富的是 pipeline、并没改 SFT 本体**,而且都需要奖励 / 偏好 / 负样本。【原文 §2 第三段】
  - **DPO(Rafailov 2023)/ NFT(Chen 2025b)**：DPO 绕过奖励建模、直接在偏好数据上优化,把模仿和强化统一进一个 loss;NFT 用隐式的负策略来建模错误生成,无需显式反馈就能自我改进。短板:**它们仍然依赖偏好对或负样本**,解决不了"只有正样本"的原生 SFT 场景。【原文 §2 第三段末】
  - **统一 SFT-RL 的理论线(本文最近邻,原文 §2 第四段逐条点名)**,作者逐个点了名:
    - ① Du et al. 2025——把 RLHF 看作 reward-weighted SFT(仍依赖显式奖励);
    - ② Wang et al. 2025a("Implicit reward as the bridge")——把 SFT 看成带隐式奖励的 RL,建议用更小的 lr 来管理 vanishing KL;
    - ③ Abdolmaleki et al. 2025——分析正 / 负反馈的平衡如何影响收敛;
    - ④ Qin & Springenberg 2025(iw-SFT)——把 SFT 看作 RL 的**下界**,并按"数据生成策略"做重要性加权;
    - ⑤ MixCE(Zhang 2023,前向 + 反向 KL 混合)、GOLD(Pang & He 2021,带 demonstration 的 offline RL,引入了一个未知的 demonstration 分布 \(\pi_b\) 和一个受限的 \(1/N\) 假设)。
    - **作者明确的 Δ**:这些工作"establish connections between SFT and RL through weighting, **but they do not provide a precise mathematical equivalence between the SFT gradient and the offline policy gradient**",也没把症结精确落到"\(1/\pi_\theta\) 这一项"上。【原文 §2 第四段,作者明确这是其增量】
  - **Focal Loss 对照(一个关键洞察)**：DFT 改完之后的 CE 形如 \(-p\log p\),而 Focal Loss(Lin 2017)是 \(-(1-p)^\gamma\log p\)——**两者的加权哲学正好相反**。Focal 是降低"已经分类好(高 \(p\))"那些样本的权重、好去强调难例(那是欠拟合的时代);DFT 反过来,降低"分类差(低 \(p\))"那些样本的权重、以促进泛化(这是过拟合的时代)。原文点睛:"This inversion reflects a fundamental shift in the LLM era: while underfitting was once a central challenge, overfitting and memorization now dominate"。这个对比是理解 DFT 本质的最佳锚点。【原文 §2 末】
- 动机链:现状是 SFT 泛化弱、但很多场景只能用 SFT;接着做数学诊断(Eq.5 用重要性采样把 SFT 梯度写成 on-policy 期望,Eq.6 让奖励显出 \(=\mathbb{1}[y=y^\star]/\pi_\theta\) 的形式);症结就落在 \(1/\pi_\theta\) 上——它在低概率的专家 token 处爆炸,带来大梯度,既不稳又过拟合精确匹配;所以解法是乘回一个 \(\pi_\theta\) 把这个畸变抵消、整平奖励。【原文 §3.2–3.3】
- 与最近邻工作的 Δ(精确到附录 A.4 的实测):
  - 最近邻是并发的 **iw-SFT(Qin & Springenberg)**——它同样从"SFT 是带重要性权重的 RL"出发,但区别有两点:iw-SFT 按"数据生成策略"加权,且**需要一个单独的 reference model 来算重要性权重**;而 DFT 直接定位到 \(1/\pi_\theta\)、用 \(\mathrm{sg}(\pi_\theta)\) 抵消,**只用当前模型的一次前向**就够了。
  - 附录 A.4 给了数据:DFT 在 LLaMA-3.2-3B(+2.39)、LLaMA-3.1-8B(+4.15)、DeepSeekMath-7B(+3.34)、Qwen2.5-Math-1.5B(+1.30)上都胜过 iw-SFT;只有 Qwen2.5-Math-7B 上 iw-SFT 反超(+2.45),但它表现并不一致(iw-SFT 在 LLaMA-3.2-3B 的 Math500 上 5.13 < SFT 的 8.65、AMC23 上 2.03 < SFT 的 3.13,也就是**iw-SFT 在多处反而比普通 SFT 还差**,而 DFT 几乎处处 ≥ base 和 SFT)。offline 设定下 DFT 平均 35.43 vs iw-SFT 31.86(+3.57)。【原文 §2 / 附录 A.4 / Table 6–7】

## 怎么做 + 靠不靠谱

> 一句话导读：方法本身就一行——给每个词的交叉熵乘上它的预测概率(detach)。下面这串公式是在论证"为什么这一乘恰好抵消了 SFT 的病态权重":先把 SFT 梯度改写成策略梯度(显出 \(1/\pi_\theta\) 这个病灶),再乘 \(\pi_\theta\) 把它消掉;附录 A.3 进一步证明改完之后等价于"直接最大化概率、而非最大化对数概率"。

- **方法流水线(可复现级,一行改动)**：流程是——输入专家 demonstration \(\mathcal{D}=\{(x,y^\star)\}\),做标准前向得到每个 target token 的 logits,softmax 得到 \(\pi_\theta(y^\star_t\mid y^\star_{<t},x)\),把逐 token 的 CE 乘上 **\(\mathrm{sg}(\pi_\theta(y^\star_t\mid\cdot))\)**,再反向(sg 项被当成常数),最后 AdamW 更新。输出是一个泛化更好的 SFT 模型,推理时不做任何改动。【原文 Eq.9 + 仓库 `fsdp_dft_trainer.py` L369–371 核对】
- **核心损失的真实形式 + 直觉(从 SFT 一步步推到 DFT)**：
  - 标准 SFT(句级 CE)：\(\displaystyle \mathcal{L}_{\mathrm{SFT}}(\theta)=\mathbb{E}_{(x,y^\star)\sim\mathcal{D}}\big[-\log\pi_\theta(y^\star\mid x)\big],\quad \nabla_\theta\mathcal{L}_{\mathrm{SFT}}=\mathbb{E}_{(x,y^\star)\sim\mathcal{D}}\big[-\nabla_\theta\log\pi_\theta(y^\star\mid x)\big].\)(原文 Eq.1–2)
  - **重要性采样改写(关键诊断,Eq.5,推导见附录 A.2)**：这一步是把"对固定专家分布求期望"通过插入 \(\pi_\theta\) 改写成"对模型自身分布(on-policy)求期望"——也就是从"看标准答案"变成"看模型自己会采样出什么":\(\displaystyle \mathbb{E}_{(x,y^\star)\sim\mathcal{D}}\big[-\nabla_\theta\log\pi_\theta(y^\star\mid x)\big]=\mathbb{E}_{x\sim\mathcal{D}_x}\,\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}\Big[\underbrace{\tfrac{\mathbb{1}[y=y^\star]}{\pi_\theta(y\mid x)}}_{\text{resample + reweight}}\big(-\nabla_\theta\log\pi_\theta(y\mid x)\big)\Big].\)
  - **显出隐含奖励(Eq.6)**：定义重要性权重 \(w(y\mid x)=\tfrac{1}{\pi_\theta(y\mid x)}\) 与奖励 \(r(x,y)=\mathbb{1}[y=y^\star]\),则 \(\displaystyle \nabla_\theta\mathcal{L}_{\mathrm{SFT}}(\theta)=-\mathbb{E}_{x\sim\mathcal{D}_x,\,y\sim\pi_\theta(\cdot\mid x)}\big[w(y\mid x)\,\nabla_\theta\log\pi_\theta(y\mid x)\,r(x,y)\big].\)这与策略梯度 Eq.4 \(\nabla_\theta J(\theta)=\mathbb{E}[\nabla_\theta\log\pi_\theta(y\mid x)\,r(x,y)]\) 形式相同,**唯一多了 \(w=1/\pi_\theta\) 这个病态权重**。原文强调这是"theoretical lens"(理论透镜),非严格等价。
  - **整流(Eq.7→8→9)**：核心动作是乘上 \(\mathrm{sg}(1/w)=\mathrm{sg}(\pi_\theta)\) 来抵消那个病态权重 \(w\):\(\displaystyle \nabla_\theta\mathcal{L}_{\mathrm{DFT}}=-\mathbb{E}_{x,\,y\sim\pi_\theta}\big[\mathrm{sg}(\tfrac{1}{w})\cdot w(y\mid x)\,\nabla_\theta\log\pi_\theta(y\mid x)\,r(x,y)\big].\)由于 \(\mathrm{sg}\) 阻断了梯度、且指示函数在 \(y\ne y^\star\) 时取 0,这就落到一个句级 loss:\(\mathcal{L}_{\mathrm{DFT}}=\mathbb{E}_{(x,y^\star)}[-\mathrm{sg}(\pi_\theta(y^\star\mid x))\log\pi_\theta(y^\star\mid x)]\)(Eq.8)。但整句概率是逐 token 连乘、数值不稳,所以最终要落到 **token 级版本**:\(\displaystyle \boxed{\;\mathcal{L}_{\mathrm{DFT}}(\theta)=\mathbb{E}_{(x,y^\star)\sim\mathcal{D}}\Big[-\sum_{t=1}^{|y^\star|}\mathrm{sg}\big(\pi_\theta(y^\star_t\mid y^\star_{<t},x)\big)\,\log\pi_\theta(y^\star_t\mid y^\star_{<t},x)\Big].\;}\)(Eq.9)整流之后,奖励对所有专家 token **统一为 1**(原文:"the reward of this corrected SFT … now becomes 1 uniformly … akin to … RLVR")。
  - **梯度层面看 DFT 到底在干啥(附录 A.3,Eq.10–13,极重要的"读完能理解"补充)**：由于 \(\mathrm{sg}(\pi_\theta)\) 前向数值上 \(=\pi_\theta\),DFT 的梯度 \(\nabla_\theta\mathcal{L}_{\mathrm{DFT}}=-\big[\tfrac{\mathrm{sg}(\pi_\theta)}{\pi_\theta}\big]\nabla_\theta\pi_\theta=-\nabla_\theta\pi_\theta(y^\star\mid x)\)(Eq.11–13)。**即 DFT 在数学上等价于"直接最大化 target token 的概率 \(\pi_\theta\)",而非像 CE 那样最大化 \(\log\pi_\theta\)**。对照:\(\nabla_\theta\mathcal{L}_{\mathrm{CE}}=-\tfrac{1}{\pi_\theta}\nabla_\theta\pi_\theta\)——**CE 与 DFT 梯度方向相同,仅差一个尺度因子**:CE 对低概率 token 放大梯度(因子 \(1/\pi\)),DFT 施加统一因子 1。这是"为什么 DFT 更稳"的最干净解释。原文还把 DFT 与"从噪声 demonstration 学习"(Sasaki & Yamashina 2020 的加权 BC,权重来自旧策略置信度)类比:**DFT 用单一在线模型实时算置信度,而非固定旧策略**。【原文 §3.2–3.3 / 附录 A.2–A.3】
- 逐组件必要性：
  - **乘 \(\pi_\theta\) + sg(核心,也是唯一的改动)**：抽掉它就退回普通 SFT。理论上的必要性见 Eq.5→6→7 的推导加附录 A.3 的 Eq.13;实证上的必要性看全表 DFT≫SFT 即可。
  - **token 级 vs 句级(Table 5 消融)**：句级做法会让概率连乘后数值崩塌、信号极弱(Sentence-Level 15.75 / Geometric-Mean 17.21 ≈ base 15.92),只有 Token-Level 才真正有用(31.58)。原文给的动机是:"computing importance weights over the entire trajectory can induce numerical instability … apply importance sampling at the token level, as … in PPO"。【原文 §4.6 / Table 5】
  - **加权策略对比(Table 5)**：作者还对比了 GSPO 风格的几何均值句级加权(它会 rescale 句概率以避免数值崩溃),但结果仍然弱。这说明**必须是 token 级 + 直接乘概率**。【原文 §4.6】
- **关键超参默认值(可直接照搬)**：用 veRL 框架推荐的 SFT 超参——AdamW,lr **5e-5**(只有 LLaMA-3.1-8B-Base 例外,用 **2e-5**),mini-batch **256**,max input len **2048**,cosine decay + warmup ratio **0.1**,**1 epoch**;评测用 Avg@16、temperature **1.0**、max gen len **4096**,走官方 Qwen2.5-Math pipeline。LoRA 设定(附录 A.6):rank **8**、alpha **16**。附录 A.7 的超参消融显示:lr 1e-4 / 5e-5 最佳、batch 32–256 对结果不敏感,而且 DFT 在**所有** lr / batch 下都胜过 SFT(这就排除了"是不是 SFT 调参没调好"的质疑)。【原文 §4.1 / 附录 A.6–A.7】
- 关键机制直觉(从概率分布层看,Fig.2)：SFT 会把所有概率一律往训练集推(尤其是抬高那些低概率 token,呈单向);DFT 则呈**两极化 / 双峰**——抬一部分、同时压另一部分,形成一个 bimodal 分布。被压到最低 bin 的多是 the / let / 逗号 / 句号这类**语法功能词**,相当于在说"别死磕连接词、把注意力放到实质语义上",本身是一种正则化。DPO / GPPO / PPO 也呈同样方向、但幅度更弱。原文用人类教学打了个比方:"students are taught to focus on substantive concepts rather than perfecting … common connective words"。【原文 §4.7 / Fig.2】
- 实验与证据(四组 + 两组附录)：
  > 一句话导读：分四块测(数学 / offline RL / 代码 / 多模态)。最能说明问题的一点是——SFT 在难基准上常常越训越差,而 DFT 反而往上走;但要留意 offline RL 那张表(DFT>GRPO)各基线的框架和调参并不统一,可比性弱于主实验。
  - **① 数学 SFT(Table 1)**：在 NuminaMath-CoT 上采 100k 训练,跑了 5 个 base。Qwen2.5-Math-1.5B:DFT 平均 31.58(比 base +15.66),这个增益是 SFT 增益(+2.09)的 **5.9×**;Qwen2.5-Math-7B 上 DFT 37.15(+15.90),约是 SFT(+2.37)的 3.8×;其余 LLaMA-3.2-3B +3.46(SFT +2.05,1.4×)、LLaMA-3.1-8B +10.02(SFT +5.33,1.88×)、DeepSeekMath-7B +15.51(SFT +7.18,1.58×)。最关键的现象:SFT 常常在难基准上**退化**(Qwen2.5-Math-7B 的 AIME24 6.68→2.48、1.5B 的 Olympiad 15.88→12.63),而 DFT 反而提升(AIME24→8.56、Olympiad→27.08)。Fig.1 还显示:DFT 在前 120 步就达峰值,10–20 步就超过了 SFT 的最终值。【原文 §4.1 / Table 1 / Fig.1】
  - **② offline RL(Table 2,Qwen2.5-Math-1.5B)**：自采 100k×4 response 用 math-verify 筛得 ~140k;DPO 构 100k 偏好对(ms-swift,lr 1e-6,bs 128);PPO/GRPO 用 verl(lr 1e-6,bs 256,GRPO n=4)。DFT 平均 **35.43**,超最佳 offline 基线 RFT(23.97,+11.46)、甚至超最强 online 的 GRPO(32.00,+3.43)。逐项:AMC23 DFT 48.44(比 GRPO +7.19、比 RFT +17.66)、Minerva 25.16(比 GRPO +6.23、PPO +9.75)。且**不需 reference model / 大 batch**。【原文 §4.2 / Table 2】
  - **③ 代码(Table 3)**：UltraFeedback 采 10k 选最高分 response(1 epoch,lr 5e-5,warmup 0.05,bs 16)。Qwen2.5-Coder-7B DFT 比 SFT 在 HE +12.8、HE+ +11.0、MultiPL-E avg +4.7。注意 SFT 在 Coder-7B 上 HumanEval 反降(62.2→54.9),DFT 升到 67.7。【原文 §4.3 / Table 3】
  - **④ 多模态(Table 4)**：WeThink 训 Qwen2.5-VL-3B(LLaMA-Factory + VLMEvalKit),评 MathVerse/MathVision/WeMath,全面超 base 与 SFT(MathVerse 33.83→37.54)。【原文 §4.4 / Table 4】
  - **附录 A.5(OpenR1-Math-220k 高质量数据)**：3 epoch,DFT 在 SFT(+13.24)之上再 +9.03=总 +22.27,说明高质量数据上仍有效。**附录 A.4(iw-SFT 对比)**:见上"Δ"。
  - baseline 公平吗:Table 2 各基线超参与数据构造不同(DPO/PPO/GRPO 各用不同框架与 lr),跨方法平均分比较的可比性需谨慎(各自是否同等调参未完全交代)。【原文 §4.2 实现细节】
  - "看着强但没回答核心问题":主张"改进 SFT 的泛化",证据(难基准上由退化转提升)切题;offline RL 表(DFT>GRPO)更像额外卖点,公平性弱于主实验。
- 假设与失效边界：
  > 一句话导读：DFT 的本事是"按模型自己的置信度重加权",这反过来也是它的软肋——遇到模型本来就不会的新事实,降权恰好压制了最该学的东西,反而学不动。
  - 【原文 §4.5 关键失效案例】在 **Natural Questions(事实知识)** 上,SFT 把准确率从 31.24% 提到 36.62%,DFT 反而降到 30.14%。原因是 DFT 按模型自身的置信度重加权,会**强化已有的信念**(原文:"tends to reinforce the model's existing beliefs");当模型本来就缺这条事实知识时,这种强化反而**阻碍了学习**。换句话说,DFT 适合"与模型先验能力对齐的任务"(逻辑推理 / 结构化预测),不适合"吸收新事实"。
  - 【原文 §5 Limitations】对**难例、以及训练数据里欠表示的样本**也可能不利——因为它们初始概率低、会被降权。而且评测只限于数学 / 代码,更大模型 / 更广任务都没测。
  - 【原文 §3.2】Eq.6 的"SFT = 策略梯度"只是一个**理论透镜(theoretical lens)**、而非严格等价:它依赖把专家分布当成 Dirac、把奖励当成指示函数等假设;作者也明确说"this RL-style characterization serves solely as a theoretical lens"。
- 祛魅总结【推断】：
  - 真贡献:用 RL 透镜给出了一个**优雅、且一行就能实现**的 SFT 改进,把症结精确定位到 \(1/\pi_\theta\);附录 A.3 用 Eq.13 证明"DFT ≡ 最大化概率、而非 log 概率",这是机制级的硬证据;Focal Loss 反演的类比(\(-p\log p\) vs \(-(1-p)^\gamma\log p\))非常有说服力地把它放进了"过拟合时代的目标设计"这个叙事里。实证覆盖数学 / 代码 / 多模态 + offline RL,分量是足的。
  - 包装 / 等价视角:所谓"隐含奖励病态",本质上就等价于"CE 梯度对低概率 token 敏感(因子 \(1/\pi\))"这个已知事实;DFT 完全可以不诉诸 RL 来理解——它就是"**按模型置信度重加权的 SFT / focal-loss 的反向版 / 把目标从最大化对数概率改成最大化概率**"。RL 透镜提供了一个漂亮的动机,但机制本身有多种等价解释,而且"competitive with PPO/GRPO"这个结论在不同设定下未必稳健。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：每个 target token 的**预测概率 \(\pi_\theta(y^\star_t\mid\cdot)\)** 作为乘性权重(detach);奖励隐式=常数 1。不用外部奖励/教师 logits/负样本。
  - **改什么**：只改**损失函数**(CE × sg(prob));模型/数据/流程同标准 SFT。
  - **何时改**：训练时(每步用当前 \(\pi_\theta\) 算系数);推理无改动。
  - **免梯度?**：是 SFT 范式、无 RL rollout、无 reference model、无大 batch;理论上是 offline 策略梯度的"整流版"(梯度等价于 \(-\nabla_\theta\pi_\theta\))。
  - **记忆-技能生命周期**：纯参数内"巩固",无外部记忆/技能库;偏"更稳地把推理技能写进参数"。
  - **防遗忘机制**：无显式防遗忘设计。注意与同族 EAFT 的对照——EAFT 用**熵**门控显式防遗忘,DFT 用**概率**加权主打泛化;且 EAFT 论文实测 DFT 的遗忘缓解效果常差于 SFT(GLM4-9B 上 DFT General Avg -8.4 反比 SFT 的 -6.0 更糟,因概率无法区分"该学的低概率"vs"冲突的低概率")。【推断,依据 eaft.md/Table1】
- ⑦ 开源代码+框架/harness：https://github.com/yongliang-wu/DFT (本地已克隆 ~66MB)。**框架=veRL(HybridFlow,Sheng 2025)**:FSDP SFT 训练器 `verl/verl/trainer/fsdp_dft_trainer.py`,**本地核对 L369–371 即概率重加权三行**——`probs = torch.softmax(shift_logits, dim=-1)` → `prob_coefficients = probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)` → `loss = loss * prob_coefficients.detach()`,与论文 Eq.9 逐字对应(`.detach()` 即 sg);配 Liger kernel + bf16。训练入口 `verl/train_dft.sh`(torchrun,8 卡)。评测沿用 `math_evaluation/`(Qwen2.5-Math 官方 pipeline);多模态用 LLaMA-Factory + VLMEvalKit。生态:TRL、LLaMA-Factory(`examples/extras/dft`)、ms-swift(`examples/train/full/dft.sh`)均一行启用。代码可得性高。【原文 §4.1/§4.4 + 本地仓库核查】
- 💰 资源/成本与可扩展性：成本 ≈ 标准 SFT(只多一次 softmax + gather + detach);主实验最大 7B、100k 数据、1 epoch;offline RL 对照含自采 140k。LoRA(rank 8/alpha 16)亦验证有效(附录 A.6)。【原文 §4.1/§4.2/附录 A.6】
- 🎯 对"探索-巩固"对标：**中等支撑(巩固侧的损失工具)+ 部分竞品(同为单行 SFT 改造)**。判定依据:DFT 是"巩固时按置信度重加权"的代表,与本课题"forward-hard/backward-soft、按某统计量重加权 CE"同构,**可借组件**= \(\mathrm{sg}(\pi_\theta)\) 的 token 级乘性门控可直接用在 MTP/OPD 的巩固损失里(且附录 A.3 证明它等价于"最大化概率",可与 MTP 前瞻概率天然衔接)。但**关键缺口/反例**:§4.5 证明 DFT 会**强化已有信念**、在"需吸收新知识/走偏后该纠正"的场景反而有害——这恰与本课题"巩固=固化新有效行为且不遗忘"冲突(DFT 降低低概率 token 权重 = 降低"模型当前不会、恰恰最该学/最该回轨"的 token 的学习)。因此 DFT 更像"探索-巩固"框架要**警惕/改造**的对象;EAFT 用熵门控正是对 DFT 这一缺陷的修正(更贴近本课题)。它也**无 on-policy 自选/无 path-recovery/无 teacher 脚手架**。
- 🔭 开放问题/未来方向：
  - 【原文 §5】更大模型/更广任务的验证;探索**非均匀 / 质量感知的奖励分配**(给 demonstration 赋不同奖励而非一律 =1,原文 "non-uniform or quality-aware reward assignments")。
  - 【推断】与 on-policy 蒸馏结合:用 student 自选轨迹替代固定专家 demonstration,可缓解 §4.5 的"强化错误先验"问题;以及把"概率门控(DFT)"与"熵门控(EAFT)"组合,分别管"泛化"与"防遗忘"。

RETURN: dft_reweight | 读PDF=是(全文+Eq.1–9/附录A.2推导/A.3梯度分析Eq.10–13/A.4 iw-SFT Table6–7/Table1–5/§4.5失效/§4.7,并本地核 fsdp_dft_trainer.py L369–371) | 加厚=是(新增Eq.5–9与A.3梯度等价Eq.11–13的MathJax+直觉;related work逐条点名Du/Wang/Abdolmaleki/Qin/MixCE/GOLD;补iw-SFT精确差异与超参默认值) | LaTeX公式=12条(Eq.1–2合1、Eq.5、Eq.6、Eq.9 boxed、A.3 梯度3式、内联 \(1/\pi_\theta\) 等多处) | 待核=0

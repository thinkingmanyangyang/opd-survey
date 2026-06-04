distillm | DistiLLM: Towards Streamlined Distillation for Large Language Models | KAIST AI(Jongwoo Ko, Sungnyun Kim, Se-Young Yun)+ Microsoft(Tianyi Chen) | 2024-07-03 · arXiv 2402.03898 v2 · ICML 2024 | 主题线 L1(白盒蒸馏/on-policy SGO)，兼 L6(散度/梯度信用) · 相关性 高

**原始论文**：https://arxiv.org/abs/2402.03898

## 一眼看懂

> 一句话导读：白盒蒸馏(学生能看到教师 logits)长期有"loss 不稳"和"用学生自己生成的样本来训太慢"两个老毛病;本文用"偏斜 KL"治不稳、用"复用旧样本 + 自适应少生成"治慢,质量不掉的同时训练快 2.5–4.3 倍。

- 🟦 TL;DR：自回归 LM 的白盒蒸馏有两大老毛病。
  - **(1) 目标函数没标准、KLD/RKLD 不稳**。这里 KLD 指前向 KL 散度、RKLD 指反向 KL,是衡量两个概率分布差距的常用 loss。问题在于 KL 的梯度系数被 \(1/q_\theta\) 加权(\(q_\theta\) 是学生给某 token 的概率):学生概率趋零时这个系数爆炸→梯度炸;同时还有 mode-averaging(被迫覆盖教师所有可能)/mode-collapse(只学一个峰)的老问题。
  - **(2) on-policy 蒸馏又慢**。所谓 SGO(Student-Generated Output,学生自己生成的输出)做 on-policy 蒸馏,确实能减小"训练时喂教师数据、推理时学生自走"的失配;但**每步都让学生现场生成 SGO 极慢**——SGO 生成可占训练总时长 80%、整体慢到 5.5×。
  - DistiLLM 两件套对症:① **skew KLD(偏斜 KL)** 在教师/学生分布间插一刀,防止分母趋零→梯度稳,并有 L2 误差界(Thm.1);② **自适应 off-policy** 用 replay buffer(经验回放池)缓存历史 SGO,再配一个按验证 loss 自适应的 SGO 使用概率 \(\phi\),以及线性递减的生成频率 \(\lambda_R=\phi(1-t/T)\),把昂贵的现场生成压到最低。最终质量达 SOTA 同时训练提速 2.5–4.3×。【原文 §Abstract / §3.1–3.2 / Alg.1】
- 最巧的一步：**用插值 \(\alpha p+(1-\alpha)q\) 给 KL 的分母"托底"**。具体见 Eq.6——它把梯度系数从 KL 的 \(r_{p,q}\) 换成 \((1-\alpha)r_{p,\tilde q}\),其中 \(\tilde q\) 是教师/学生的混合分布。抽掉这一刀就退回 KLD 的梯度爆炸/不稳。
  - 配套的 Thm.1 + Remark 1 给出一个凸权衡:\(\alpha\) 越大,L2 误差越小,但梯度尺度也越小→两头拉扯,存在最优,实验定在 \(\alpha\approx0.1\)。
  - 注意分工:off-policy 那套只是效率组件,抽掉只是变慢、不是变错;**skew 才是质量与稳定性的命门**。【原文 Eq.5–7 / Thm.1 / Remark 1 / Fig.3】

## 为什么做

> 一句话导读：把大教师压成小学生本来很标准,但"选哪个散度做 loss"全靠经验、"用学生自己生成的样本"又慢又会被教师误判;本文要同时把这两件事——质量和效率——一起解决。

- 研究背景：把大教师 LM 蒸馏成小学生 LM,目的是降推理/显存成本。经典白盒 KD 的做法是用 KLD 在固定数据集上让学生匹配教师分布(Hinton 2015;Kim & Rush 2016 的 SeqKD)。【原文 §1 / §2.1】
- 解决的具体痛点(原文 §2.2 两条):
  - **① 目标函数缺标准**。前向 KLD 易 mode-averaging:当存在某个 \((x,y)\) 使教师概率 \(p(y\mid x)\gg0\) 而学生概率 \(q_\theta(y\mid x)\approx0\) 时,学生被迫覆盖教师整个 support(所有可能输出)→过度平滑。反向 RKLD 则相反,易 mode-collapse(只盯一个峰)。JSD 折中,但又要调权重 \(\beta\)。Agarwal et al. 2024 观察到"最优散度是任务相关的",所以选 loss 本身就很麻烦。
  - **② SGO 用法的两难**。on-policy 用 SGO 减失配确实有效,但有两个副作用:(a) 教师对自己不熟悉/不准确的 SGO 会给**误导反馈**——Fig.1 里教师给"短而错"的生成打了低 loss 0.0671、却给"长而对"的打了高 loss 2.0954;(b) 每步现场生成 SGO **极慢**,Fig.2 显示生成可占训练时长 80%,整条 on-policy 慢达 5.5×。【原文 §2.2 / Fig.1–2】
- 相关工作 & 各自不足(原文 §1/§2.2,逐条点名)：
  - **散度族**:
    - Wen et al. 2023——系统研究了 f-散度(含 TVD、JSD),但 TVD 等**不可分解**为 token 级,本文 §2.1 明确只研究"tractable(可处理)的 KLD"。
    - Agarwal et al. 2024(GKD)——用广义 JSD + on-policy SGO,但散度仍需手选、且**每步生成**导致样本效率低。
  - **策略优化族**:MiniLLM(Gu et al. 2024)——用策略梯度最小化反向 KL 并降方差,但**需教师每步生成**,因教师大而算力暴增(原文:"notably increases training computation due to the requirement of a large teacher model")。
  - **共性缺口(本文 Δ)**:没人**同时**解决"目标函数标准化"与"SGO 效率",也少有人把"减失配的正效应"与"噪声反馈的负效应"做自适应平衡。【原文 §1 / §2.2 末】
- 动机链:现状(白盒 KD 用 KLD + SGO)→ 缺陷(KL 不稳、无标准 + SGO 又慢又可能被教师误导)→ 所以需要一个**同时**兼顾质量(有理论保证的散度)与效率(最小化 SGO 生成频率、平衡正负效应)的框架。【原文 §2.2 末】
- 与最近邻工作的 Δ(三范式对比,Fig.4):
  - (a) 固定数据集 KD——高效但性能低;
  - (b) on-policy(GKD/MiniLLM)——性能高但低效(每步都得生成);
  - (c) **本文自适应 off-policy**——想两头兼得:靠 replay buffer + 线性递减的 replay ratio \(\zeta=(1-t/T)\),把 SGO 生成频率 \(\lambda_R=\phi(1-t/T)\) 全程压低。
  - 散度侧则把"用哪个散度"**收敛到有 L2 误差界的 skew KL**(Thm.1),给出"为什么是这个散度"的理论依据。【原文 §3 / Fig.4】

## 怎么做 + 靠不靠谱

> 一句话导读：方法就两块——把 loss 换成"偏斜 KL"(给分母托底,梯度不炸),再用一个调度器决定"这一步要不要现场生成 SGO、要不要从缓存里取",尽量少生成。下面公式多,核心就一句:让 KL 的分母别趋零。

- **方法流水线(Alg.1,可复现级)**：输入是教师 \(p\)、学生 \(q_{\theta_0}\)、初始 \(\phi\)、一个空的 replay buffer \(\mathcal{D}_R\)、训练/验证集。每一步 \(t\) 先抽一个随机数 \(u\sim\mathrm{Unif}(0,1)\),然后:
  1. **(生成 SGO 并更新 buffer)** 若 \(u<\lambda_R:=\phi(1-\tfrac{t}{T})\),就用当前学生 \(q_{\theta_t}\) 生成一批 SGO \(\{\tilde y_i\}\) 存进 \(\mathcal{D}_R\)(满了就替换最旧的);
  2. **(选数据源)** 若 \(u<\phi\) 从 \(\mathcal{D}_R\) 抽 SGO(off-policy),否则从固定数据集 \(\mathcal{D}\) 抽(即用教师真值);
  3. **(更新)** 用 \(\alpha\)-S(R)KL\((\alpha=0.1)\) 更新 \(\theta_t\);
  4. **(SGO Scheduler,验证点触发)** 算一次验证 loss \(\tilde L_t\),若它比上次还涨了一截(\(\tilde L_t>\tilde L_{t-1}+\varepsilon\))就把 \(\phi\) 调高一点 \(\phi\leftarrow\min(\phi+\tfrac{1}{N_\phi},1.0)\),否则不动。
  输出:蒸馏好的学生 \(q_{\theta_T}\)。主实验用 SRKL + off-policy,初始 \(\phi=0\),buffer 容量 1000,\(\alpha=0.1\)。【原文 Alg.1 / §3.2 / §4】
- **核心散度的真实形式 + 直觉**：
  - **先看普通 KLD**——序列级 KLD 可以精确拆成 token 级求和(Eq.1–3):\(\displaystyle D_{\mathrm{KL}}(p,q_\theta)=\mathbb{E}_x\,\mathbb{E}_{y\sim p(\cdot\mid x)}\Big[\log\tfrac{p(y\mid x)}{q_\theta(y\mid x)}\Big]\approx\tfrac{1}{|\mathcal{D}|}\sum_{(x,y)\in\mathcal{D}}\sum_{t}\sum_{y_t\in V}p(y_t\mid y_{<t},x)\log\tfrac{p(y_t\mid y_{<t},x)}{q_\theta(y_t\mid y_{<t},x)}.\)
  - **skew KL 的定义**——核心就一句:把 KL 里的目标分布换成"教师/学生的混合"。\(\;D^{(\alpha)}_{\mathrm{SKL}}(p,q_\theta)=D_{\mathrm{KL}}\big(p,\;\alpha p+(1-\alpha)q_\theta\big),\quad D^{(\alpha)}_{\mathrm{SRKL}}(p,q_\theta)=D_{\mathrm{KL}}\big(q_\theta,\;(1-\alpha)p+\alpha q_\theta\big).\)
  - **梯度为什么变稳(关键,Eq.5–7)**:
    - 普通 KLD 的梯度是 \(\nabla_\theta D_{\mathrm{KL}}(p,q_\theta)=-r_{p,q_\theta}\nabla_\theta q_\theta(y\mid x)\),其中 \(r_{p_1,p_2}\) 是两个分布的比值。当学生概率 \(q_\theta\approx0\) 时,这个比值 \(r_{p,q_\theta}=p/q_\theta\) 会爆炸→产生又大又噪声的更新。
    - skew 之后变成:\(\displaystyle \nabla_\theta D^{(\alpha)}_{\mathrm{SKL}}(p,q_\theta)=-\underbrace{(1-\alpha)\,r_{p,\tilde q_\theta}}_{\text{coefficient}}\nabla_\theta q_\theta(y\mid x),\qquad \tilde q_\theta=\alpha p+(1-\alpha)q_\theta.\)分母被 \(\alpha p\) 托底、不再趋零,梯度就有界了(Fig.3a)。
    - 反向版(SRKL)同理:\(\displaystyle \nabla_\theta D_{\mathrm{KL}}(q_\theta,p)=-(\log r_{q_\theta,p}+1)\nabla_\theta q_\theta,\quad \nabla_\theta D^{(\alpha)}_{\mathrm{SRKL}}=-\underbrace{(\log r_{q_\theta,\tilde p}+1-\alpha r_{q_\theta,\tilde p})}_{\text{coefficient}}\nabla_\theta q_\theta,\;\tilde p=(1-\alpha)p+\alpha q_\theta.\)
    - Fig.3a/b 实测:\(\alpha\) 越大,梯度系数越小(越稳)。一句话直觉:"用一点教师分布给学生分布托底,既防梯度炸,又不至于把目标改太多"。
  - **误差界与最优 \(\alpha\)(Thm.1 + Remark 1)**:
    - Thm.1 给出 \(\alpha\)-SKL 经验估计的 L2 误差上界 \(\mathbb{E}[|D^{(\alpha)}_{\mathrm{SKL}}(p^1_n,p^2_n)-D^{(\alpha)}_{\mathrm{SKL}}(p^1,p^2)|^2]\le \tfrac{c_1(\alpha)}{n^2}+\tfrac{c_2\log^2(\alpha n)}{n}+\tfrac{c_3\log^2(c_4 n)}{\alpha^2 n}\),其中 \(c_1(\alpha)=\min\{\tfrac{1}{\alpha^2},\tfrac{\chi^2(p^1,p^2)^2}{(1-\alpha)^2}\}\)——这说明**\(\alpha\) 越大,误差越小**。
    - 但 Remark 1 提醒:现代优化器会按梯度尺度 \(\tfrac{1}{1-\alpha}\) 做归一化,归一化后的 L2 范数对 \(\alpha\) 是**凸的**→所以存在一个最优值,实验定在 \(\alpha\approx0.1\)(Fig.3d)。
    - 这一分析还顺带分清了 S(R)KL 与 JSD 的根本差异:JSD \(D^{(\beta)}_{\mathrm{JSD}}=\beta D^{(\beta)}_{\mathrm{SKL}}(p,q_\theta)+(1-\beta)D^{(1-\beta)}_{\mathrm{SRKL}}(q_\theta,p)\) 没法对两项同时取温和的 skew。【原文 §3.1 / Thm.1 / Remark 1】
- 逐组件必要性(逐个问"抽掉会怎样"):
  - **skew KLD/SRKL(核心)**：抽掉就退回 KLD 的梯度爆炸 + mode 问题。证据:Fig.3(a)(b) 显示梯度系数随 \(\alpha\) 增大而变小(更稳);Tab.1 消融里 \(\alpha\approx0.1\) 的 SKL/SRKL 在 5 个指令基准上全面优于 KLD/RKLD/JSD(如 Super-Natural:KLD 20.68→SKL 26.26、SRKL 25.83;Unnatural:23.38→28.06/28.62);Fig.6 还收敛更快。【原文 §3.1 / Tab.1 / Fig.6】
  - **Adaptive SGO Scheduler(\(\phi\) 调度)**：抽掉就只能在两个极端里选——\(\phi\) 恒高会被噪声反馈淹没(Tab.2 里 On-policy 多处反而掉分),\(\phi\) 恒低又修不了失配。调度器按验证 loss 自适应,排出"先少用 SGO、后多用"的课程。证据:Tab.2 里 Adaptive 在多数基准上优于 On-policy/Mixed(SKL Dolly 25.90 vs On-policy 24.27)。【原文 §3.2 / Tab.2】
  - **off-policy + replay buffer**：抽掉(改成每步重新生成 SGO)就慢 5.5×。buffer 复用历史 SGO 提升样本效率,而且换成 off-policy 后性能仅微降(Tab.2)。【原文 §3.2 / Tab.2 / Fig.4(c)】
  - **线性递减 replay ratio \(\zeta=(1-t/T)\)**：这是对 off-policy bias 的针对性设计(off-policy RL 在新旧策略差异大时偏差大,Kumar 2019)。逻辑是:早期学生变化快→多用新鲜 SGO 来压 off-policy bias;后期接近收敛→多复用 buffer 提效率。【原文 Fig.4 caption / §3.2】
  - **与 SKL 的协同(原文明说)**:off-policy 之所以能成功,"stems from the fast convergence speed of S(R)KL"。换句话说,正是因为 SKL/SRKL 早期改进快(Fig.6),才能在 off-policy 这种高 bias 的设置下不被拖垮;换成别的 loss 再切 off-policy 就会掉分(Tab.4)。【原文 §3.2 Synergy with SKL】
- 实验与证据：
  - **任务/对**:
    - 指令遵循:数据 databricks-dolly-15k(14K 训、各 500 验测),教师→学生对包括 GPT-2 XL(1.5B)→GPT-2(0.1B)、OPT-2.7B→1.3B、OpenLLaMA2-7B→3B(7B 用 LoRA)。
    - 摘要:SAMSum(附录另含 XSum/CNN-DM),用 T5-XL→Base/Small。
    - 翻译:IWSLT2017 En-De,用 mT5。
    - 指标 ROUGE-L / GPT-4 feedback / BLEU;**额外加了 OpenWebText 的 LM 损失**(沿 Gu 2024,能提指令调性能)。5 seeds、4×A100。【原文 §4 / §4.1】
  - **关键数字**:训练相对近期 KD(MiniLLM/GKD)提速 **2.5–4.3×**;DistiLLM 只需朴素 KD 约 1.6× 的时间,而 MiniLLM/GKD 要 3–7×;on-policy 每步生成可达 5.5× 运行时、SGO 生成占比达 80%(Fig.2/Fig.7)。质量在指令/摘要/翻译三类任务上都达 SOTA。【原文 §Abstract / §2.2 / §3.2】
  - baseline 公平吗:与 SFT / KD(KLD)/ SeqKD / ImitKD(在 SGO 上做 KLD)/ MiniLLM / GKD 同设置对比,且统一在 4×A100 上计时——较规范。
  - "看着强但没回答核心问题":核心主张是"又好又快",质量(SOTA)与速度(2.5–4.3×)都给了证据;但模型偏小偏旧(最大 7B 且用 LoRA),迁到现代大模型的外推性还待验证。
- 假设与失效边界：
  - 【原文 §2.1】只研究**可分解为 token 级的 KLD**(TVD 等不可分解的被排除)。【推断】方法是**白盒**——需要教师 logits 且学生/教师同 tokenizer、同词表,不适用黑盒蒸馏。
  - 【推断】一堆超参都是经验值:\(\alpha=0.1\)、buffer 容量(~1000)、\(\phi\) 的 \(1/N_\phi\) 步进、\(\varepsilon\) 阈值。Thm.1 只给"单调/凸"这个**方向**,不是闭式最优;off-policy 的 bias-效率权衡靠一个启发式的线性 schedule 控制。依据:Remark 1 是上界的凸性陈述,实际 \(\alpha\) 仍由实验定。
  - 【推断】所谓"理论保证"更多是定性方向(梯度有界 + L2 误差界的数量级),不是强保证;skew 带来的稳定性收益,在概念上和温度平滑/标签平滑有重叠。
- 祛魅总结【推断】：
  - 真贡献:把"散度选择"从经验调参,提升为**有梯度/误差分析支撑的 skew KL**——Eq.5–7 的梯度系数 + Thm.1 的凸权衡都是机制级的;同时把 on-policy 蒸馏的**效率瓶颈(SGO 生成)** 用 off-policy + buffer + 自适应调度系统性解决。"质量×效率"两条线都给了机制级解释,工程价值高,是后续 DistiLLM-2 及许多蒸馏工作的 backbone(底座)。
  - 包装/被高估处:skew 本质就是"插值平滑",新颖性有限(真正贡献在系统化 + 理论方向 + 与 off-policy 的协同);"理论保证"是方向性的、不是强界;off-policy 调度是合理但启发式的工程组合。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：教师分布 \(p\) 的 logits(白盒),经 skew 混合后做 KL/RKL 匹配;可选 SGO(学生自生成序列)上的教师反馈。
  - **改什么**：改**目标函数**(KLD→skew KLD/SRKL)+ 改**数据采样策略**(on-policy→自适应 off-policy + replay buffer)。学生参数全更新(7B 用 LoRA)。
  - **何时改**：训练时;散度每步生效,SGO 生成按 \(\lambda_R=\phi(1-t/T)\) 概率触发,\(\phi\) 在验证点更新。
  - **免梯度?**：是监督式蒸馏(最小化散度的标准反传),无策略梯度/RL 奖励;off-policy 仅指数据复用,非 RL off-policy 算法。
  - **记忆-技能生命周期**：**replay buffer \(\mathcal{D}_R\)** 是一个轻量"经验记忆"——存学生历史 SGO 供复用,是本文唯一的"记忆"成分(为效率而非防遗忘);技能=教师分布知识压进学生参数。
  - **防遗忘机制**：无显式防遗忘;skew 的"教师托底"间接保留教师 support,replay 复用旧 SGO 缓解 off-policy 漂移,但都不针对灾难性遗忘。
- ⑦ 开源代码+框架/harness：https://github.com/jongwooko/distillm (本地已克隆 ~853KB)。**框架=自研**(基于 MiniLLM 代码基 + 定制 HF Transformers + DeepSpeed,**无外部 RL 框架**)。核心 `distillm/losses.py`(skewed_forward / reverse_kl,默认 lam=0.1)、`distillm/buffer.py`(deque replay buffer)、`sampler.py`;含 gpt2/opt/openllama2 的 sft/kd/seqkd/imitkd/minillm/gkd/distillm 全基线脚本。代码可得性高。【本地仓库核查】
- 💰 资源/成本与可扩展性：4×40G A100;最大模型 7B(LoRA)。核心卖点即成本——相对 on-policy KD 训练提速 2.5–4.3×,SGO 生成频率被 scheduler + buffer 压低。支持一阶段蒸馏(无需先 SFT 学生)、对学生初始化更鲁棒。【原文 §Abstract / §4】
- 🎯 对"探索-巩固"对标：**中等支撑(on-policy 数据效率 + 梯度稳定的方法学)**。判定依据:DistiLLM 正面回应了 OPD 的核心痛点——"on-policy/SGO 太贵"。
  - **可借组件**(对本课题 MTP+OPD 很实际):
    - ① **replay buffer + 线性递减生成频率 \(\lambda_R=\phi(1-t/T)\)**——可直接缓解"student 自选轨迹/恢复分支"采样开销大的问题(本课题靠 on-policy 自选,生成成本是真问题);
    - ② **skew KL 的"教师托底防梯度爆炸"**——可用于"走偏后回轨"时学生概率很低、梯度易炸的场景(Eq.6 的 \((1-\alpha)r_{p,\tilde q}\) 系数直接可借)。
  - **缺口**:
    - ① 纯分布匹配,没有"探索/选路"或"path-recovery 单点接管"机制;
    - ② skew 是 token 级的全分布匹配,与本课题"teacher 当稀疏脚手架(只在关键步介入)"正相反——它处处匹配、不稀疏;
    - ③ 无 MTP/前瞻、无技能库/防遗忘。
  - 总评:属于"把 OPD 做高效"的工程基座,不是探索-巩固的直接对标。
- 🔭 开放问题/未来方向：
  - 【原文】把 skew KL + 自适应 off-policy 推广到更大现代 LLM、更多任务;散度选择的进一步标准化。
  - 【推断】把"自适应 SGO 调度/replay"思想嫁接到 RLVR/OPD-agent 的轨迹采样上以省算力;把 skew 的"教师托底"用于稀疏/关键步蒸馏(只在关键 token 用 skew 托底),向本课题"稀疏脚手架"靠拢。

RETURN: distillm | 读PDF=是(§1–4 全文 + Eq.1–7/Thm.1/Remark1/Alg.1/Fig.1–7/Tab.1–2,核 SKL/SRKL 梯度系数) | 加厚=是(新增 Eq.1 分解、SKL/SRKL 定义、Eq.5–7 梯度系数、Thm.1 误差界、Alg.1 四步流水线全 MathJax;related work逐条点名 Wen/Agarwal/Gu 及各自短板;补 Tab.1/Tab.2 具体数字与"协同"机制) | LaTeX公式=9条(Eq.1分解、SKL/SRKL定义、KL梯度、SKL梯度Eq.6、SRKL梯度Eq.7、Thm.1界、JSD分解、λ_R/ζ、内联多处) | 待核=0

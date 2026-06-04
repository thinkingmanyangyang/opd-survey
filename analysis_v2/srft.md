srft | SRFT: A Single-Stage Method with Supervised and Reinforcement Fine-Tuning for Reasoning | 中科院自动化所/国科大 + 美团 + 上海交大(通讯 Dongbin Zhao 团队;Yuqian Fu、Tinghong Chen 共同一作) | 2025-06-24 arXiv v1·ICLR 2026·OpenReview n6E0r6kQWQ | 主题线 L2(统一SFT-RL/GFT类)·相关性 高

**原始论文**:https://arxiv.org/abs/2506.19767

## 一眼看懂
- 🟦 TL;DR:把 SFT 和 RL 塞进**单阶段同一个 loss** 一起训(不再先 SFT 再 RL 两段),用"当前策略熵"做开关自适应调两者权重——论文叙述为熵高(不确定)时压低 demonstration 模仿、保探索,正样本 RL 项熵高时加权维持探索。基于 LUFFY 的混合策略 off-policy RL 改造,Qwen2.5-Math-7B 上平均 59.1%(Table 2 取到 59.5),比 zero-RL 基线在 5 个数学基准 +9.0%、3 个 OOD 基准 +10.9%【原文 abstract+§5.2+结论】。
- 最巧的一步:**熵感知权重 + 单阶段融合**。抽掉熵权重退化为 LUFFY 式 SFT+RL 朴素相加(§3.2.2 的 \(L_{\text{SFT+RL}}=L_{\text{SFT}}+L_{\text{RL}}\)),论文主张正是"靠熵动态在模仿/探索间按需定权"。但需注意:**论文 Eq.8 给 SFT 权重 \(\exp(-H)\)(熵高减弱模仿),开源代码实际用 \(\exp(+H)\)(熵高加强模仿),方向相反**【待核,见结构化抽取】——所以"最巧"这步的方向性在论文与代码间未对齐,是该工作最大的自洽裂缝。

## 为什么做
- 研究背景:LLM 推理后训练里 SFT 与 RL 如何最优结合是核心未解问题;传统两阶段串行(SFT 做指令跟随→RL 做对齐/推理)把二者当独立阶段【原文§1】。
- 解决的具体痛点:① 两阶段串行——SFT 易记忆模式而非真推理、过拟合数据集(Chu et al. 2025;Chen et al. 2025a);RL 样本效率低、探索难、易 mode collapse(Gao/Dou/Schmied 2025;Cai et al. 2025)【原文§1】。② 集成不足→误差传播、限制 RL 提升;过度依赖 demonstration→过拟合、约束探索。如何在"SFT 知识蒸馏"与"RL 策略优化"之间定权重是痛点【原文§1 末】。
- 相关工作 & 各自不足(本轮按引文链补全):
  - **LUFFY**(Yan et al. 2025,off-policy RL/混合策略,**直接被 SRFT 当底座**):把 demonstration 拼进 on-policy rollout group 做混合策略 GRPO;SRFT 沿用其优势估计(Eq.6/7),但 LUFFY 本身无熵感知调权。
  - **ReLIFT**(Ma et al. 2025):交错 RL 与对**最难题**在线 fine-tune;Δ 在 SRFT 全程单阶段融合而非按题难度切换。
  - **TAPO**(Wu et al. 2025):在 GRPO 内融合**结构化外部知识**(5.5k MATH 样本);Δ 在 SRFT 用整段 demonstration 而非结构化知识。
  - **各类 zero-RL**:SimpleRL-Zero(Zeng et al. 2025,24k GSM8K+MATH)、OpenReasoner-Zero(Hu et al. 2025,129k 多源 PPO)、PRIME-Zero(Cui et al. 2025,150k NuminaMath + 隐式过程奖励)、Oat-Zero。论文把这些归为"或两阶段、或动态切换、或纯 RL",**未在熵视角下系统定权**【原文§1+§5.1 baseline 列表】。
  - 精确差异:SRFT 是上述里**唯一**用"当前策略熵"作连续旋钮、在单一 loss 内同时挂 demo-SFT/demo-RL/self-rollout-RL 四项并按熵动态定权者。
- 动机链:两阶段各有缺陷(SFT 过拟合/RL 探索难)→ 单阶段融合更优但难定权(集成不足 vs 过度依赖 demo)→ 从熵视角做机理分析(§3)发现三条 Key Findings:(1) SFT 全局粗调、RL 选择性细调(§3.1.1-3.1.2);(2) 单阶段融合训练效率优于串行(§3.2.2);(3) 熵是训练有效性指标(§3.2.1)→ 所以用熵感知权重在单阶段内自适应平衡【原文§1 Key Findings+§3】。
- 与最近邻工作(LUFFY)的Δ:LUFFY 把 demonstration 直接拼进 on-policy rollout group 做混合策略 GRPO(SRFT 的 Eq.6/7 优势估计沿用此),Δ 在于 SRFT **额外加了两个熵感知自适应权重**(demo 的 SFT 项 \(0.5\,\exp(-H)\)、self-rollout 正样本项 \(0.1\,\exp(+H)\))+ 显式把 self-rollout RL 按正/负样本拆分(Eq.11)。为什么有用:论文称熵动态揭示训练机制,据此定权能在"熵高保探索 / 熵低强模仿"间动态平衡,避免 demo 与当前策略分布失配导致的退化【原文§4.1-4.2】。

## 怎么做 + 靠不靠谱
- 方法流水线(输入→输出,§4.1-4.3):输入 prompt → ① 当前策略 \(\pi_\theta\) 生成 **8 条 on-policy rollout**(max 8192 token)→ ② 把 demonstration(DeepSeek-R1 生成的 OpenR1-Math 高质量解)拼进 rollout group 组成异构 batch \(G_{\text{aug}}=G_{\text{roll}}\cup G_{\text{demo}}\)(Eq.6),组内**统一**算 group-norm advantage(Eq.7)→ ③ 四路 loss 合成单阶段反向:demo-SFT(熵权 \(w_{\text{SFT}}=0.5\cdot\mathrm{sg}(\exp(-H))\),Eq.8)+ demo-RL(LUFFY 式 off-policy,设 \(\pi_\beta=1\)、去 clip,Eq.9)+ self-rollout 正样本(熵权 \(w_{\text{RL}}=0.1\cdot\mathrm{sg}(\exp(+H))\),Eq.12)+ self-rollout 负样本(likelihood 最小化,Eq.12)→ ④ 总 loss \(L_{\text{SRFT}}=L^{\text{demo}}_{\text{SFT}}+L^{\text{demo}}_{\text{RL}}+L^{\text{self-rollout}}_{\text{RL}}\)(Eq.13)反向 → 输出更新后的 \(\pi_\theta\)。每步 500 训练步全程动态调权【原文§4.1-4.3 Eq.5-13】。
- 逐组件必要性(有消融 Table 3):
  - **demo-SFT 熵权 \(0.5\exp(-H)\)**(Eq.8):负责"粗粒度行为策略逼近";Table 3 消融 `w/o w_SFT`(把权重设固定常数 1.0)→ Avg 从 59.1 降到 55.1(**−4.0**),证其必要。论文称没它则 demo 与当前策略分布失配会致性能退化。
  - **demo-RL(off-policy,\(\pi_\beta=1\) 去 clip)**(Eq.9-10):负责"细粒度行为策略学习";沿用 LUFFY,设 \(\pi_\beta=1\) 避免 tokenization 复杂度(免重算 behavior policy 概率)、去 clip 因 \(\pi_\beta=1\) 时 clip 失衡【原文§4.1】。
  - **self-rollout 正/负拆分 + 正样本熵权 \(0.1\exp(+H)\)**(Eq.11-12):论文观察 binary 奖励 \(\{1,-1\}\) 下 RL 目标可自然拆成正样本(似 SFT 的 max-likelihood)+ 负样本(likelihood 最小化);self-exploration 致熵快速下降损探索,故对**正样本项**用 \(\exp(+H)\) 维持探索多样性。Table 3 消融 `w/o w_RL` → Avg 降 **−2.9**。
  - **单阶段 vs 两阶段**:有对照(Table 1:SFT→RL=52.5 > RL=49.4 > SFT=47.3 > RL→SFT=37.4 > RL→SFT_KL=38.3;§3.2.2 Fig.5 单阶段 SFT+RL 训练效率优于 SFT→RL)——该对照支撑"单阶段更优"。机理:Fig.4(b) 显示 RL→SFT 时 SFT 致熵骤升再缓降(对应 Fig.4(a) 性能骤跌),且 RL 后模型 plasticity 受损;Fig.3 学习动力学显示 SFT→RL 的高性能区**反而更靠近 base model**,提示初始 SFT 过度偏移损后续 RL。
- 关键机制/公式(本轮据 PDF 正文 Eq.1-13 补全,真符号 MathJax):
  - **SFT 目标(Eq.1)**:\(L_{\text{SFT}}(\theta)=\mathbb{E}_{(x,y)\sim D}[-\log\pi_\theta(y\mid x)]\)。其梯度(Eq.4,推导见 Appendix D)揭示 SFT 为何"全局粗调":
    \(\displaystyle \nabla_\theta L_{\text{SFT}}=\mathbb{E}_{(x,y)\sim D}\!\left[\sum_{t=1}^{|y|}\sum_{v\in V}\big(\pi_\theta(v\mid x,y_{<t})-\mathbb{1}_{v=y_t}\big)\nabla_\theta\log\pi_\theta(v\mid x,y_{<t})\right]\)
    即对**全词表**每个 token 都施梯度(抬高目标 token、压低其余),故分布整体 sharpen——"sledgehammer"。
  - **GRPO 优势(Eq.2)**:\(\hat A_k=\dfrac{R(x,y_k)-\mathrm{mean}(\{R(x,y_k)\})}{\mathrm{std}(\{R(x,y_k)\})}\);GRPO 目标(Eq.3)为标准 clip 形式。RL 只动少数 token——"scalpel"。
  - **混合 batch 优势(Eq.7)**:对 \(G_{\text{aug}}\) 整体算 group-norm;因专家解奖励高,拼入后抬高整组优势,促 optimistic exploration。
  - **demo-SFT 熵权(Eq.8)**:\(L^{\text{demo}}_{\text{SFT}}(\theta)=w_{\text{SFT}}\cdot\mathbb{E}_{(x,y)\sim D_{\text{demo}}}[-\log\pi_\theta(y\mid x)]\),其中 \(w_{\text{SFT}}=0.5\cdot\mathrm{sg}(\exp(-H(\pi_\theta)))\),\(\mathrm{sg}\)=stop-grad。**论文直觉**:熵高(不确定)→ \(\exp(-H)\) 小 → 少模仿,缓解 demo 与当前策略失配。
  - **demo-RL off-policy(Eq.9-10)**:LUFFY 式,重要性比 \(r_{k,t}(\theta)=\dfrac{\pi_\theta(y_{k,t}\mid x_t)}{\pi_\beta(y_{k,t}\mid x_t)}\),令 \(\pi_\beta=1\) 且去 clip。
  - **self-rollout 拆分(Eq.11)**:在 binary 奖励下
    \(\displaystyle L^{\text{self-rollout}}_{\text{RL}}=\underbrace{\mathbb{E}_{y^+\sim\pi_\theta}[-\log\pi_\theta(y^+\mid x)]}_{\text{正样本}\approx\text{on-policy SFT}}+\underbrace{\mathbb{E}_{y^-\sim\pi_\theta}[\log\pi_\theta(y^-\mid x)]}_{\text{负样本}=\text{likelihood 最小化}}\)
    正样本目标结构上等同最大化正确响应似然(但 \(y^+\) 是**当前策略 on-policy 生成**而非来自 SFT 数据集),负样本压低错误响应概率质量。
  - **self-rollout 熵权(Eq.12)**:对正样本项乘 \(w_{\text{RL}}=0.1\cdot\mathrm{sg}(\exp(+H(\pi_\theta)))\),熵高→加强(维持探索),与 Eq.8 的 demo-SFT 方向**相反但目的互补**:\(L^{\text{self-rollout}}_{\text{RL}}=w_{\text{RL}}\,\mathbb{E}_{y^+}[-\log\pi_\theta(y^+|x)]+\mathbb{E}_{y^-}[\log\pi_\theta(y^-|x)]\)。
  - **stop-grad** 保证两个熵权只调幅度不回传梯度。
- 实验与证据:
  - 数据集/设置:训练用 **OpenR1-Math-46k-8192**(OpenR1-Math-220k 的 46k 子集,源自 NuminaMath 1.5,DeepSeek-R1 生成解,经 Math-Verify 过滤去掉 >8192 token 与不可验证项);基座 Qwen2.5-Math-7B;8 rollout/prompt、500 训练步;推理 temp=0.6、max_response=8192;验证用 Math-Verify、终评用 OAT-Grader;OOD 多选题**乱序选项**防泄漏【原文§5.1】。
  - 关键实验+数字【原文§5.2 Table 2,本轮 PDF 直读】:5 个数学基准(AIME24/AMC 用 avg@32,Minerva/Olympiad/MATH500 用 pass@1)SRFT 平均 **59.5**(abstract 报 59.1),较最佳 RL 基线 +9.0、较 SFT 方法 +4.8、较 SFT+RL 方法 +3.4;3 个 OOD 基准(ARC-C 85.3/GPQA-D 46.4/MMLU-Pro 55.9)平均 **62.5**,较最佳基线 +4.7。逐项:AIME24 35.3 / AMC 74.3 / MATH500 89.8 / Minerva 39.7 / Olympiad 58.3。对比 LUFFY(底座)Avg 55.5(ID)/57.8(OOD)。abstract/结论的"+9.0%/+10.9%"口径是**对 zero-RL 基线**(非对 Table 中最佳 SFT+RL 基线)【原文 abstract+结论】。
  - baseline 公平吗:含 LUFFY/ReLIFT/TAPO 等同期强基线,且 LUFFY/SRFT 用同一 46k 数据集,基座统一 Qwen2.5-Math-7B,较公平;但部分 zero-RL 基线(SimpleRL/ORZ/PRIME)用不同/更大训练集(24k~150k),"对 zero-RL +9.0/+10.9"的口径混入了数据规模差异。
  - 看着强但没回答核心:Table 3 只消融了"两个熵权设为常数 1.0"(各 −4.0/−2.9),**未独立消融熵权的方向(\(\exp(\pm H)\))与系数(0.5/0.1)**,故"增益来自熵感知"vs"来自单阶段融合(LUFFY 已有)"的归因不够干净。
- 假设与失效边界:
  - 显式【原文】:依赖高质量 demonstration(Limitations 明说"假设可获高质量 demo,imperfect demo 待研究");设 \(\pi_\beta=1\) 简化(off-the-shelf 数据无需重算 behavior policy 概率);熵利用为"basic exponential weighting",自承简单。
  - 隐式【推断】:仅 Qwen2.5-Math-7B 单基座 + OpenR1-46k 单训练集,跨模型族/跨数据未验证(依据:§5.1 仅此一配置)。
  - 一致性瑕疵【原文 vs 代码,关键】:论文 Eq.8 SFT 权重 \(\exp(-H)\) 与开源代码 \(\exp(+H)\) 方向相反(见结构化抽取),作者未澄清。
- 祛魅总结【推断】:真贡献=熵视角的 SFT/RL 机理分析(SFT 全局粗调 vs RL 选择性细调、单阶段优于串行、熵作指标)+ 在 LUFFY 上加两个熵自适应权重的工程实现。包装/高估:① abstract 的"+9.0/+10.9"是对 zero-RL 而非对最强基线(对最强 SFT+RL 仅 +3.4/+4.7),相对弱基线放大了观感;② "熵感知"机制的 SFT 权重方向在论文(\(\exp-H\))与开源代码(\(\exp+H\))间相反且作者未澄清,削弱"熵如何调度模仿"叙事的可信度。低估:单阶段融合的训练效率优势(Fig.5/6:同步训更快收敛、response 渐长、熵更稳)论证较扎实但 abstract 未突出。

## 结构化抽取
- 🎯 机制速览6轴:
  - **学什么信号**:demonstration 的 token 级 NLL(SFT 模仿)+ on-policy rollout 的 group-norm advantage(RL,二值奖励 \(\{1,-1\}\))+ 当前策略熵 \(H(\pi_\theta)\) 作为调权信号。
  - **改什么**:同一策略模型 \(\pi_\theta\) 全参数(单 loss 反向)。
  - **何时改**:单阶段训练全程,500 步;每步按当前熵动态调 SFT/RL 权重。
  - **免梯度?**:否——是梯度训练(SFT+RL 联合 loss);熵权用 stop_grad 仅阻断"权重自身"的梯度,主 loss 仍正常回传。
  - **记忆-技能生命周期**:无显式记忆/技能库;"技能"即固化进 \(\pi_\theta\) 参数;demonstration 提供外部专家轨迹做粗粒度逼近,self-rollout 做细粒度精修。
  - **防遗忘机制**:单阶段融合本身缓解"RL 阶段灾难性遗忘 SFT 知识"(§3.2.2 指出两阶段 SFT→RL 的 RL 段会遗忘 SFT 知识致 transient 退化);熵感知权重防 demo 过拟合/RL 过早熵坍缩。无显式 KL-to-SFT 正则(kl_loss_coef=0)。
- ⑦ 开源代码+框架/harness:https://github.com/fyqqyf/SRFT (Tier A,v1 验证已克隆约 2.4MB,代码完整)。框架 **veRL + vLLM**(rollout/评测),底座沿用 **LUFFY 的 mix_src 结构** 与 deepscaler 奖励。核心实现 `srft/verl/verl/mix_src/`(`mix_core_alg.py` 的 `compute_token_on_off_policy_loss`、`mix_actor.py` loss 组装)。模型权重 HF `Yuqian-Fu/SRFT`。
  - **〔待核/已 flagged 的 paper-code 符号差异,本轮据实际仓库代码再坐实〕**(承 v1):论文 Eq.8 写 SFT 权重 \(0.5\cdot\exp(-H)\),而开源代码 `mix_core_alg.py:162-163` 实为 `entropy_exp_coeff = entropy.exp().detach()` 然后 `sft_loss = (- entropy_exp_coeff * log_prob)...`,即 **\(\exp(+H)\)**,且**该行 SFT 系数上无可见的 0.5 因子**(0.5 通过运行脚本里的 `sft_loss_coef=-0.5` 在外层施加,与 paper 的 0.5 数值对得上,但 exp 内符号方向仍相反)。以代码为准则"熵高加强模仿",与论文叙述(熵高减弱模仿)矛盾,作者未澄清(可能论文笔误或代码 bug)。RL 侧 `mix_core_alg.py:134` 为 `pos_entropy_exp_coeff = 0.1 * (entropy).exp().detach()` = \(\exp(+H)\),与论文 Eq.12 **一致**(含 0.1 系数)。结论:**SFT 项 paper-code 方向矛盾(±H),RL 项一致**。
  - adaptive temperature(可学习 log_alpha 对齐 target entropy,类 SAC)是代码可选项、**默认关闭**(`use_adaptive_temperature: False`),非核心组件;论文正文未把它列入方法/贡献(Limitations 只说"future work 可探索 adaptive entropy scheduling")。
- 💰 资源/成本与可扩展性:训练 64×A100(脚本 n_gpus_per_node=8、nnodes=4),500 步;8 rollout/prompt,max_prompt=1024/max_response=8192,actor lr=1e-6,train_batch=128/ppo_mini_batch=64,kl_loss_coef=0、entropy_coeff=0.001,sft_loss_coef=-0.5,tp=2,use_dynamic_bsz【元信息+v1 脚本核】。仅单基座单数据集验证,跨规模可扩展性原文未说明。
- 🎯 对"探索-巩固"对标:**支撑/可借组件**。一句判定:SRFT 是"统一 SFT-RL"线里与本项目 idea 最贴近的一类——它用**熵**当信号在"模仿专家(巩固)"与"自探索(探索)"间动态定权,正是"探索-巩固"的一种 loss 级实现;可借组件=熵感知自适应权重(把 \(H\) 当探索/利用调度的连续旋钮),以及"正样本 RL≈on-policy SFT"的拆解直觉(Eq.11)对设计 teacher 脚手架的稀疏监督有参考(提示:teacher 在 student 走通的开头给监督,等价对"on-policy 正样本"加权)。依据:§4.1-4.2 熵权设计 + §3.1.1 "RL 在初始邻域做选择性细调"恰对应"偏向自己能走通的开头"的 on-policy 自选。缺口:SRFT 全程 dense 融合,无"稀疏脚手架/单点接管/走偏后自选恢复分支"机制;teacher 信号是整段 demonstration 而非关键步介入;无 MTP/前瞻;熵权对整段 demo **均匀**施加而非按关键步/高熵 token 选择性施加。
- 🔭 开放问题/未来方向:【原文】熵利用目前仅"basic exponential weighting",可探索 adaptive entropy scheduling / 多时间尺度熵分析;假设高质量 demo,可研究 imperfect demonstration 训练(Limitations)。【推断】解决 paper-code 的 \(\exp(\pm H)\) 符号矛盾(关乎"熵如何调度模仿"的正确方向,建议复现两版本对比);把熵感知权重从"整段 demo 均匀施加"细化到"按关键步/高熵 token 选择性施加",更接近稀疏脚手架;跨模型族/跨数据集泛化验证。

RETURN: srft | 读PDF?是(22页/85100字,abstract+§3.1.1 SFT梯度Eq.4+§4.1-4.3 Eq.1-13+§5.2 Table2/3 全直读;Eq.8 exp(−H)、Eq.12 exp(+H) 均PDF核实) | 加厚?是(补全 Eq.1-13 全套真符号公式含 SFT 梯度全词表形式、self-rollout 正负拆分、混合 batch 优势;related-work 引文链 LUFFY/ReLIFT/TAPO/各 zero-RL 精确差异;Table3 消融 −4.0/−2.9 数字;据**实际代码 mix_core_alg.py 行号**坐实 SFT 项 exp(+H) 且无内嵌 0.5、RL 项一致) | LaTeX 公式条数 8(SFT目标/SFT梯度Eq.4/GRPO优势/混合优势/demo-SFT权Eq.8/off-policy比/self-rollout拆分Eq.11/self-rollout权Eq.12) | 待核 1(paper Eq.8 \(\exp(-H)\) vs 代码 \(\exp(+H)\) 的 SFT 权重符号矛盾,作者未澄清;RL 侧一致)

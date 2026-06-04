opd_blog | On-Policy Distillation (博客 + tinker-cookbook 配方) | Thinking Machines Lab(Kevin Lu 等) | 2025 · 博客(非论文) | 主题线 L1(on-policy 蒸馏方法论)，兼 L4(多轮工具)、L6(token 级稠密信号) · 相关性 高

**原始论文**：https://thinkingmachines.ai/blog/on-policy-distillation/ (博客,无 arXiv)
> 来源说明:本条无 PDF。正文据既有 analysis/opd_blog.md(已核博客)+ 本批对 tinker-cookbook 仓 `distillation/train_on_policy.py`、`recipes/distillation/README.md` 等源码/文档的二次核实撰写;数值为博客单点近似值,已标注。本次增强=公式 LaTeX 化 + 在已核事实上加厚措辞,**未新增未见事实**。

## 一眼看懂
> 一句话导读：on-policy distillation(在线策略蒸馏)就是让小模型(学生)自己写答案,然后请一个更强的大模型(教师)逐字逐句告诉它"这一步你的概率分布该长什么样",学生照着改。它同时占了两个便宜:学生走的是自己的路(避免 SFT 那种"训练时看完美范文、自己一上手就翻车"的失配),而教师在每一步都给反馈(避免 RL 那种"答完一整题才给一个对错"的稀疏)。

- 🟦 TL;DR：本文系统推广了 **on-policy distillation**。它的核心是:学生在**自己采样出来的轨迹**上,以更强教师的**逐 token 分布(用 reverse KL 度量)** 作为**唯一的稠密监督**(环境不给任何 reward)。这样做兼顾了两件事——
  - on-policy 的分布真实性:解决了 SFT 的训练-推理失配;
  - 教师反馈的稠密性:解决了 RL 的奖励稀疏。
  博客在三个场景上做了演示:推理(数学)、个性化(恢复指令遵循能力)、多轮工具使用;并配套 tinker-cookbook 给出了可复现的 LoRA 配方。一条代表性曲线:在 OpenThoughts3 上 SFT 到 AIME'24 约 55%,再在 DeepMath 上做 on-policy 蒸馏,**仅约 100 步**就升到约 65%。【既有 analysis + 仓 README】
- 最巧的一步：**把 reverse KL 当作 per-token advantage、用策略梯度的形式回传**,而**不是**把它写成一个显式的 KL 散度 loss 项。具体到代码(`incorporate_kl_penalty`):先算 `reverse_KL = log p_student − log q_teacher`,再令 `advantage = −kl_penalty_coef · reverse_KL`。如果抽掉"在学生轨迹上算 + 用 reverse KL"这一步——比如改成在教师轨迹上算(那就成了 off-policy),或改用 outcome reward(那就成了 RL)——就会分别退回 SFT 的失配 / RL 的稀疏。正是这一步,让"token 级稠密蒸馏"和"on-policy 采样"能用同一套 importance-sampling loss 来实现。【本批源码核 `train_on_policy.py` L53-130】

## 为什么做
> 一句话导读：训小模型一直是"两害取其轻"——SFT 让它背范文,可一上手走自己的路就翻车;RL 让它走自己的路,可一整题才给一个对错、学得又慢又贵。本文要的是两头的好处都拿到:学生走自己的路、同时教师在每一步都纠正它。

- 研究背景：大模型蒸馏分两类:
  - off-policy:在教师生成的固定数据上做 SFT;
  - on-policy:学生采样自己的轨迹,教师再对学生的轨迹逐 token 打分。
  博客把 on-policy distillation 当作一个"兼顾稠密监督与 on-policy 真实性"的范式来系统推广。【既有 analysis】
- 解决的具体痛点：
  - ① off-policy 蒸馏/SFT 容易"复述"教师的局部模式,且存在训练-推理的分布不匹配(即 exposure bias,曝光偏差):学生在**自己的推理轨迹上从未被纠正过**,于是误差不断累积;
  - ② on-policy RL(GRPO)只有序列级的稀疏奖励,token 层面的学习效率低、采样成本高;
  - ③ 单纯模仿教师的文本,无法在"学生自己访问到的那些状态"上获得稠密反馈。【既有 analysis】
- 相关工作 & 各自不足（来龙去脉 + 并行路线 + 精确差异）：
  - **off-policy / SFT**(在教师轨迹的固定数据上匹配 next-token):短板是失配、误差累积;仓库里的 `train_off_policy.py` 提供它作对照。
  - **on-policy RL / RLVR(GRPO)**:短板是只有序列级的稀疏 outcome reward,token 学习效率低、且贵。
  - **学术界的 on-policy KD**(GKD(Agarwal 2024)、MiniLLM(Gu 2024)):与本文**范式同源**(都在学生轨迹上蒸馏)。精确差异:博客把它**工程化成一套托管 SDK(Tinker)上的 LoRA 配方**;而学术 KD 里散度要手动选、每步生成又贵。
  - **SDFT(自蒸馏 SFT)**:仓库里的 `sdft.py` 提供它作并行对照(用同一个模型重新生成 + SFT)。
  - **站谁肩上**:GKD/MiniLLM 的 on-policy KD 范式 + Tinker 托管训练栈 + Harbor 沙箱(用于多轮工具)。博客的定位是**把这个范式工程化,并配上 recipe、三场景演示和 checkpoint 句柄**。【既有 analysis + 仓 `train_off_policy.py`/`sdft.py`/README】
- 动机链（一步步推下来）：
  - 现状:SFT 失配 vs RL 稀疏,只能二选一;
  - 洞见:让学生自己采样可解决失配,让教师逐 token 打分可解决稀疏,二者合起来就是两全;
  - 结论:于是有了 on-policy distillation——学生轨迹 + 教师 token 分布的 reverse KL、且不给 reward。【既有 analysis】
- 与最近邻工作的 Δ：
  - 相对 **off-policy SFT**:把数据源从"教师轨迹"换成"学生自身轨迹",差别在于学生能在自己会访问到的状态上被稠密纠正。
  - 相对 **on-policy RL(GRPO/RLVR)**:把"序列级 outcome reward"换成"token 级的教师 KL",差别在于每个 token 都有稠密信号、样本/步效率因此高得多(约 100 步就见效)。
  - 相对 **GKD/MiniLLM**:把它落地成托管 SDK(Tinker)上的 LoRA 配方,并扩展出多教师 / 多轮工具的用法。【既有 analysis】

## 怎么做 + 靠不靠谱
> 一句话导读：一轮就是"学生写一段→教师对这段逐字给概率→把两者的差当 advantage 回传更新",而且只对学生自己写的 token 算 loss(prompt、工具返回这些都遮掉)。靠不靠谱?它直观、约 100 步就见效,但本质是博客而非论文——没有多 seed、没有基线对照、没有显著性检验,数字都是单点近似。

- 方法流水线（输入→输出,对应代码 `train_on_policy.py`,读完即可复现）：一轮分六步:
  - ① **学生采样**:对每一批,用当前的 sampling client 采轨迹(温度默认 1.0,对应 `do_group_rollout_and_filter_constant_reward`),并记录每个 token 的 `sampled_logprobs`(即 \(\log p\))和 `mask`(对应 `prepare_minibatch`);
  - ② **教师打分**:对每条 datum,用它对应的 teacher sampling client(每个 dataset 配一个)异步算出 `teacher_logprobs`(即 \(\log q\));实现上,把学生的 `model_input` 末尾 append 上 target token 得到 `full_sequence_inputs_D`,教师 logprob 取 `teacher_logprobs[1:]` 来对齐到 next-token;
  - ③ **reverse-KL-as-advantage**(对应 `incorporate_kl_penalty`):逐 token 算 \(\mathrm{KL}[p\|q]=\log p-\log q\),per-token advantage = \(-\)`kl_penalty_coef` \(\times\) `mask` \(\times\) reverse_KL,然后**就地(in-place)加进** `datum.advantages`;
  - ④ 可选地用 `kl_discount_factor`(默认 0.0)对未来的 KL 做折扣(对应 `discounted_future_sum_vectorized`;博客称这没有明显增益);
  - ⑤ 通过 Tinker 的 `importance_sampling` loss 做一次 `train_step` 更新;
  - ⑥ **只对学生生成的 token 计 loss**(系统提示、用户消息、工具返回、assistant header 都被 mask 掉);
  - 然后进入同步循环(对应 `do_sync_training`)。【本批源码核 `train_on_policy.py` L53-130/L174-369】
- 逐组件必要性（每模块干啥 / 有无消融 / 怎么咬合）：
  - **reverse KL 作 advantage(核心)**:没有它就不是 on-policy 蒸馏。代码注释写得很清楚:"KL[p||q]=log p − log q"、"adjust the advantages in-place as the negative reverse KL"。它的作用是把"教师的逐 token 监督"塞进策略梯度的 advantage 槽里,从而复用 RL 的 importance-sampling loss。【本批源码核 L62-63/L86-108】
  - **仅对学生 token 计 loss(用 mask)**:没有它,prompt 和工具返回也会被算进去、污染信号(代码里 `float_masks` 会逐 token 乘进 reverse_kl 与 advantage)。在多轮工具场景下,"system / user / tool 返回 / assistant header"全部被 mask。【本批源码核 L90-103 + README "Environment-provided tokens … are masked out"】
  - **零 reward(纯 KL)**:多轮工具场景用 `reward_fn=zero_reward`(恒返回 0.0)去覆盖默认的 `HarborReward`,这样唯一信号就只剩"对教师的 KL"。没有它就会混入 RL 奖励、不再是纯蒸馏。【既有 analysis + README + `harbor_multiturn.py`】
  - **多教师**:对每个数据集配一组 `(teacher_model, groups_per_batch)`,各自建立 teacher sampling client、分别采样,再由 `CompositeDataset` 拼成一个 batch(对应 `on_policy_multi_teacher.py`)。注意**博客没有展示这部分实验**,只在 recipe 里提供。【既有 analysis + README "multiple teachers" + 源码 L443-475】
  - **kl_discount_factor**:可选,默认 0.0;博客实验里**没有明显增益**——可视为"非必要"组件。【既有 analysis + 源码默认值 L147】
- 关键机制/公式（真实形式 + 直觉,据源码/博客抄准）：
  - **per-token reverse KL(代码逐 token 形式)**:对位置 \(t\) 学生采到的 token,\(\displaystyle \mathrm{rKL}_t=\big(\log p_\theta(o_t\mid x,o_{<t})-\log q_T(o_t\mid x,o_{<t})\big)\cdot m_t,\) 其中 \(m_t\) 为 mask(仅学生生成 token 为 1)。注意这是**单点采样估计**(只在学生实际采到的 token 上算 \(\log p-\log q\)),而非对整词表求和的全 KL。
  - **reverse-KL-as-advantage**:\(\displaystyle A_t \mathrel{+}= -\,c_{\text{KL}}\cdot m_t\cdot \mathrm{rKL}_t,\qquad c_{\text{KL}}=\text{kl\_penalty\_coef}\ (\text{默认}\ 1.0),\) 即把"教师对该 token 的偏好"写成负的 reverse-KL 优势,加进 advantage 后由 `importance_sampling` loss(clipped surrogate)更新。可选折扣:\(\tilde A_t=\sum_{k\ge t}\gamma^{k-t}A_k\)(`kl_discount_factor` \(\gamma\),默认 0)。
  - **目标的直觉**:学生在某条它**自己采出来**的轨迹 \(y\sim\pi_\theta\) 上,最小化沿途逐 token 的 reverse KL,即目标 \(\mathbb E_{y\sim\pi_\theta}\big[\sum_t \mathrm{KL}\big(\pi_\theta(\cdot\mid x,y_{<t})\,\|\,\pi_T(\cdot\mid x,y_{<t})\big)\big]\)。回顾两种老办法的毛病:RL 的毛病是奖励稀疏(一整条轨迹只在最后给一个对/错);SFT 的毛病是分布失配(只在教师的完美前缀上学,学生自己一走偏就没人管)。on-policy 蒸馏的做法是:让学生**自己走**(on-policy,解决失配),同时让教师在学生走到的**每一步**都给出"你这一步的分布该长什么样"(reverse KL,解决稀疏)。
  - **为何用 reverse 而非 forward**:reverse KL \(\mathrm{KL}(p_{\text{student}}\|q_{\text{teacher}})\) 是 mode-seeking 的(让学生去聚焦教师的几个高概率模式),而非 mode-covering(去覆盖教师的所有模式)——博客选了前者,但没深入论证这个取舍。另外,教师 logprob 提供的稠密信号,样本/步效率远高于序列级的 outcome reward,这正是"约 100 步就从 55% 升到 65%"的直觉来源。【既有 analysis + 源码】
- 实验与证据：
  - **推理**:分两步——
    - ① 先在 OpenThoughts3-1.2M 上做 SFT(对应 `off_policy_reasoning`,rank-128 LoRA,约 3000 步,LR 1e-3),得到 AIME'24 约 55%;
    - ② 加载这个 ckpt,在 DeepMath-103K 上做 on-policy 蒸馏(教师 Qwen3-32B,`groups_per_batch=512`,LR 1e-4,约 100 步),升到 AIME'24 约 65%。
    也就是仅约 100 步,就从 55% 升到 65%。【README + 既有 analysis(博客近似值)】
  - **个性化**:用内部文档加上重采样过的 Tulu3(assistant 轮由 Qwen3-8B 重新生成)做 SFT,再用 Tulu3 的 prompts 做 on-policy 蒸馏(`groups_per_batch=64`),在 IFEval 上约 100 步即恢复。【README + 既有 analysis】
  - **多轮工具使用**:对应 `on_policy_distillation_harbor_multi_turn`,让 **Kimi-K2-Thinking 同时充当学生与教师**,在 Harbor 沙箱里以 `reward_fn=zero_reward` 的纯 KL 信号训练。具体配置:`max_turns=10`、`group_size=4`、`groups_per_batch=8`、`lora_rank=8`、`kl_penalty_coef=1.0`、`max_tokens=2048`、`max_trajectory_tokens=24576`、温度 1.0。【README + 既有 analysis(已据 README 核)】
  - baseline 公平吗:**这是一篇博客、非受控论文**——大多是单点曲线或示例,**缺多 seed、缺基线对照、缺统计显著性**;"55%→65%"是近似值;多教师在博客里**没有展示**。
  - "看着强但没回答核心问题":它直观展示了样本/步上的高效,但严格性不足(没有显著性、数值为近似)。
- 假设与失效边界：
  - 【源码注释 L72-73】当教师与学生**用不同的 renderer/tokenizer 时,需要手工对齐**(注释原话:"if your teacher has a different renderer than the student, you may want to modify the full_sequence_inputs_D")——跨家族蒸馏没有演示。
  - 【既有 analysis + 源码】on-policy 蒸馏需要教师做**在线 logprob 计算**(`teacher_client.compute_logprobs_async`),依赖 Tinker 托管服务——所以复现门槛绑定在这个 SDK 上。
  - 【推断】博客示例里学生与教师是**同族的**(学生 Qwen3-8B-Base / 教师 Qwen3-32B,共享 renderer/tokenizer;多轮场景甚至用同一个模型),这降低了分布漂移;跨家族的通用性边界没有系统评估。依据是 recipe 的默认设置与代码注释。
  - 【推断】reverse KL 是 mode-seeking 的,可能牺牲多样性(高 Pass@1、低 Pass@k);但博客没评 Pass@k,forward vs reverse 的取舍也没深入。依据是 reverse KL 的已知性质,加上博客未作讨论。
- 祛魅总结【推断】：
  - 真贡献:把 on-policy distillation 这个范式**清晰地工程化了**——reverse-KL-as-advantage 的实现、纯 KL 无 reward、仅对学生 token 计 loss,三点都经过代码核实;再以三场景 + 可复现配方 + checkpoint 句柄降低了上手门槛,传播价值高(本课题的 opd_survey 把它列为奠基性博客)。
  - 包装/被高估处:
    - 它**是博客、不是论文,没有严格的基线/统计**,数值都是近似的单点;
    - 多教师能力**只在 recipe 里、博客未展示**;
    - reverse-KL 的取舍、跨家族蒸馏、Pass@k 多样性都未评估;
    - "两全其美"这套叙事在"同族师生 + 数学/IF 场景"下成立,但边界没有充分探。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：教师在**学生轨迹**上的逐 token 分布(reverse KL=\(\log p_{\text{student}}-\log q_{\text{teacher}}\));白盒(需教师 logprob);**无 reward**(既非正确性也非格式)。
  - **改什么**：改学生参数(LoRA);不改结构。
  - **何时改**：在线——学生采样→教师打分→更新,每个 token 都更新(in-place 写进 advantage)。
  - **免梯度?**：否,是梯度优化;但用**策略梯度形式**(reverse KL 作 advantage 经 importance-sampling loss)实现 token 级 KL 蒸馏,而非显式 KL loss 项。
  - **记忆-技能生命周期**：无外部记忆/技能库;"技能"=教师的逐 token 推理分布,蒸进学生 LoRA 参数。
  - **防遗忘机制**：无显式防遗忘;个性化场景用"SFT 恢复 + 蒸馏"组合来恢复被改动的能力,弱相关。
- ⑦ 开源代码+框架/harness：https://github.com/thinking-machines-lab/tinker-cookbook (已 clone,~14MB)。核心 `tinker_cookbook/distillation/`(`train_on_policy.py`、`train_off_policy.py`、`sdft.py`、`datasets.py`)与 `recipes/distillation/`(`on_policy_distillation.py`、`off_policy_reasoning.py`、`on_policy_multi_teacher.py`、`harbor_multiturn.py`、`on_policy_distillation_harbor_multi_turn.py` + `README.md`)。**框架=Tinker SDK**(基于 LoRA 的自研托管训练 SDK,蒸馏 loss=`importance_sampling`、采样/教师 logprob 由 Tinker 服务侧执行)——**非 veRL/TRL/OpenRLHF**。启动如 `python -m tinker_cookbook.recipes.distillation.on_policy_distillation model_name=Qwen/Qwen3-8B-Base load_checkpoint_path=tinker://… dataset=deepmath learning_rate=1e-4 groups_per_batch=512 lora_rank=128`。【本批 repo 核 + README】
- 💰 资源/成本与可扩展性：全程 LoRA(rank 8/32/128 句柄均提供);on-policy 蒸馏约 100 步即见效(样本/步高效是卖点,LR 1e-4 LoRA / 5e-5 full);依赖 Tinker 托管(教师在线 logprob)。具体卡数博客未集中说明。关键超参:`kl_penalty_coef=1.0`、`kl_discount_factor=0.0`、`temperature=1.0`、`eval_every=20`/`save_every=20`、`loss_fn=importance_sampling`;多教师默认 DeepMath→Qwen3-32B、Tulu3→Qwen3-235B-A22B-Instruct-2507、学生 Qwen3-8B。【既有 analysis + 源码/README 默认】
- 🎯 对"探索-巩固"对标：**强支撑(它是本课题 OPD 一侧的直接范式蓝本);是核心对标对象**。判定依据如下:
  - ① **范式同构**:"学生 on-policy 自采样 + 教师在学生轨迹上逐 token 稠密监督"正是本课题 TSRD 的 OPD 骨架——探索 = 学生自己走(on-policy);巩固 = 教师把走到的每一步分布拽向正确。
  - ② **可借组件(很实在)**:
    - (a) `incorporate_kl_penalty` 里的 **reverse-KL-as-advantage**,是现成的 token 级稠密信号实现,可直接拿来作"teacher 当稀疏脚手架"的底层信号(只需把"处处 KL"改成"仅在关键步给 KL");
    - (b) 多轮工具里 `reward_fn=zero_reward` 的纯 KL 设置,可作本课题"多轮 agent 自进化"的可复现起点;
    - (c) 多教师拼批,可作"多脚手架来源"的接口;
    - (d) Tinker 的 checkpoint 句柄便于复现。
  - **缺口 / 与本 idea 的关键差异**:
    - ① 博客是 **token 级处处用 reverse KL(稠密、全程介入)**,而本课题要的是 **teacher 当稀疏脚手架(只在关键步或走偏处单点接管)**——需要把博客的"处处监督"稀疏化(这正是 TSRD 相对 vanilla OPD 的创新点);
    - ② **没有 path-recovery 的"自选恢复分支"机制**(博客里学生只是被教师逐 token 拽着走,不会显式地"走偏后自选一个回轨分支");
    - ③ **没有 MTP 前瞻探针**;
    - ④ reverse KL 的 mode-seeking 性质可能压低多样性,这与本课题"探索 = 发现多条能走通的路径"略有张力。
  - 一句话:**它是 OPD 的可复现范式蓝本与底层信号实现(直接可借);本课题的增量,在于把"处处稠密 KL"改造成"稀疏关键步脚手架 + 自选回轨 + MTP 前瞻"。**
- 🔭 开放问题/未来方向：
  - 【博客/README】跨家族(不同 renderer/tokenizer)教师蒸馏;多教师的实证;`kl_discount_factor` 的适用场景(博客称无增益)。
  - 【推断】四个方向:
    - 把"处处 reverse KL"稀疏化为"只在高熵/关键 token 上用教师 KL 接管"(向本课题的稀疏脚手架靠拢);
    - 评估 reverse 与 forward KL 对 Pass@k 多样性的影响;
    - 把 on-policy 蒸馏与 path-recovery 结合(走偏后让学生自选恢复分支,教师只在分支点做稠密监督);
    - 引入 MTP 前瞻,用来挑选"值得教师介入的关键步"。

RETURN: opd_blog|读PDF=否(无PDF;据既有analysis(博客)+本批tinker-cookbook源码核`train_on_policy.py`L53-130 reverse-KL-as-advantage+README超参)|加厚=是(公式LaTeX化+源码行级锚点,未新增未见事实)|LaTeX公式条数=4(per-token rKL/advantage更新/折扣/目标期望)|待核数=0

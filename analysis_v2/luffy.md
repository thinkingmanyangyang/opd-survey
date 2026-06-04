luffy | Learning to Reason under Off-Policy Guidance (LUFFY) | 上海AI实验室·西湖大学·南京大学·港中文(Yafu Li 等;Project Lead Yafu Li) | 2025-04-21 v1 → 2025-06-22 v5;NeurIPS 2025;arXiv 2504.14945 | 主题线 L2(统一SFT-RL)+L1(off-policy指导蒸馏)·相关性 高

**原始论文**:https://arxiv.org/abs/2504.14945

## 一眼看懂
- 🟦 TL;DR:纯 on-policy RL(GRPO)只能在模型"自己已会"的范围里放大,弱模型/难题很快撞天花板;纯 SFT/蒸馏又是死记硬背(behavior cloning),泛化差、熵塌。LUFFY 的做法是把一条**更强 teacher(DeepSeek-R1)的现成正确推理轨迹**和**模型自己采样的 rollout** 塞进同一个 GRPO group 里一起做组内 advantage 归一化(Eq.4)——自己全错时 teacher 轨迹(奖励恒正)自然主导信号,自己能解时则保留自我探索;再加一个 **policy shaping** 函数 \(f(x)=x/(x+\gamma)\)(\(\gamma=0.1\))专门放大"低概率但关键"动作的梯度、压住熵坍塌(§3.1-3.2)。
- 最巧的一步:**把 off-policy teacher 轨迹混进同一个 group 做组内归一化**。抽掉它(退回普通 GRPO),弱模型上 advantage 全负、学不到任何东西就垮(§5.2:on-policy RL 只能在简化数据上训 LLaMA3.1-8B);抽掉 policy shaping(退回线性 shaping \(f(x)=x\)),则会因 IS 比率把关键低概率动作梯度压成 \(\pi_\theta(1-\pi_\theta)\)→**熵坍塌、过快收敛**(§3.2,Fig.2 左熵曲线/Fig.6)。两者一个负责"能不能学到外部能力",一个负责"学的时候别塌",缺一不可。

## 为什么做
- 研究背景:RLVR(可验证奖励 RL,数学答案对/代码过测才给奖励)让大推理模型涌现多步推理与自反思("aha moment",§1);奖励设计 \(R(\tau)=\mathbb{1}[\tau\text{ 输出对 }q\text{ 的正确答案}]\)(Eq.1),靠规则验证避免 reward hacking。但 RLVR 本质 **on-policy**——GRPO 只从 \(\pi_{\theta_{\text{old}}}\) 自采样里学(Eq.2-3),性能受基座自身上界约束(与 limit_rlvr 结论一致)。
- 解决的具体痛点:① on-policy RL 只能放大已有行为,无法引入真正新的认知能力;弱模型(LLaMA3.1-8B)缺必要基础认知行为,RL 下迅速进入平台期。② 纯 SFT/蒸馏是 behavior cloning,泛化差、易过拟合、致熵坍塌。③ 二者之间缺一个能在"模仿"与"探索"间**动态平衡**的机制。
- 相关工作 & 各自具体短板:
  - **RLVR(R1/o1/Kimi-1.5/OpenReasoner-Zero)**:受基座自采样上界封顶——只能强化模型已能采样到的行为,引入不了基座没有的认知模式。
  - **纯 SFT / 序列蒸馏(SeqKD,Kim&Rush 2016)**:能把外部 R1 轨迹的知识注入,但是 off-policy 整轨模仿→ **暴露偏差 + 僵化过拟合 + 熵坍塌**,OOD 泛化差(论文 Table 1 实测 SFT 的 OOD 均值 47.5 远低于 LUFFY 的 57.8)。
  - **简单拼 SFT+RL / "RL w/ SFT Loss"**:论文专门做对照——把 SFT 损失直接加进 RL(等权混)反而比纯 SFT 还差(OOD 46.0 vs 47.5),"SFT+RL"两段式虽好(OOD 44.8)但仍缺"按当前 rollout 成败动态切换"的机制。
  - **此前混合 / off-policy 方案**:缺少"在同一 advantage 归一化里让 teacher 信号按需主导/退居"的极简开关,且未处理 off-policy 注入引发的熵坍塌。
- 动机链:on-policy RL 撞基座天花板 → 想引入更强 teacher 轨迹当"认知脚手架" → 但简单拼 SFT+RL 会僵化/熵塌 → 所以要(a)混进同 group 让信号"按需主导"(§3.1)+(b)用 shaping 保护关键低概率动作(§3.2)。
- 与最近邻工作的精确Δ:相对纯 RLVR,差在**注入了基座外的轨迹**(能破自采样上界);相对纯 SFT/SeqKD,差在**轨迹只在 student 自己失败时主导、成功时退居其次**(组内归一化天然实现 on-policy 自适应开关,无需手调权重)+ policy shaping 防熵塌;相对"RL w/ SFT Loss"等权混合,差在用 advantage 而非固定权重决定 teacher 影响力。

## 怎么做 + 靠不靠谱
- 方法流水线:① 对每个 prompt \(q\),准备一条 off-policy teacher 轨迹(来自 R1,奖励恒正)+ 若干条 on-policy rollout;② 二者混入**同一** GRPO group 一起算组内 advantage(§3.1);③ off-policy 项用重要性采样比 \(\hat r_{j,t}=\pi_\theta/\pi_\phi\)、on-policy 项用标准 \(r_{i,t}=\pi_\theta/\pi_{\theta_{\text{old}}}\);④ 对 off-policy IS 比率施 policy shaping \(f(\cdot)\)(§3.2);⑤ actor 更新合并 off-policy shaping 项 + on-policy clip 项,梯度更新 student;为效率取 \(\pi_\phi=1\) 并对 off-policy 项**省掉 clip**。
- 逐组件必要性:
  - **Mixed-Policy group 归一化**:负责"破基座上界"。有消融——纯 on-policy RL 在弱模型/难数据上完全失败(§5.2),证明其必要。
  - **Policy shaping \(f(x)=x/(x+0.1)\)**:负责"防熵坍塌/防过快收敛"。有消融——Fig.6 显示 Mixed-Policy 无 shaping(即线性 \(f(x)=x\))时早期猛涨但随即熵塌、后期被反超;\(\gamma\) 敏感性在 App. E.4。
  - **\(\pi_\phi=1\) 计算简化**:论文为计算效率取 \(\pi_\phi=1\),故 \(\hat r_{j,t}=\exp(\log\pi_\theta)=\pi_\theta\)(代码 `mix_core_alg.py`)。好处:避开师生不同 tokenization、可直接吃现成数据集无需重算 \(\pi_\phi\)、且收敛保证仍成立(Theorem 1 对任意良定义 \(\pi_\phi\) 都给 \(O(1/\sqrt K)\) 速率)。属工程近似,论文承认 off-policy clip 在 \(\pi_\phi=1\) 时会失衡故省去;无单独消融。
- 关键机制/公式(真实符号 + 直觉):
  - **标准 GRPO**(§2,作为起点):组内归一优势 \(A_i=\dfrac{R(\tau_i)-\mathrm{mean}(\{R(\tau_i)\})}{\mathrm{std}(\{R(\tau_i)\})}\)(Eq.2);目标
  \[
  J_{\text{GRPO}}(\theta)=\frac{1}{\sum_i|\tau_i|}\sum_{i=1}^{N}\sum_{t=1}^{|\tau_i|}\mathrm{CLIP}\big(r_{i,t}(\theta),A_i,\epsilon\big)-\beta\, D_{\mathrm{KL}}[\pi_\theta\|\pi_{\text{ref}}],
  \]
  其中 \(\mathrm{CLIP}(r,A,\epsilon)=\min[\,r A,\ \mathrm{clip}(r;1-\epsilon,1+\epsilon)A\,]\)(Eq.3)。
  - **Mixed-Policy 优势**(§3.1,Eq.4):把 off-policy 组 \(G_{\text{off}}\) 与 on-policy 组 \(G_{\text{on}}\) 合并再归一,
  \[
  \hat A_i=\frac{R(\tau_i)-\mathrm{mean}(G_{\text{on}}\cup G_{\text{off}})}{\mathrm{std}(G_{\text{on}}\cup G_{\text{off}})}.
  \]
  直觉:off-policy 轨迹奖励高,当模型自己难解(on-policy 全错)时它在组里**自然拿到高优势主导更新**;一旦模型开始解对,on-policy 轨迹优势升高、teacher 退居其次——"会就探索、不会就模仿"的开关由归一化自动给出,无需手调。
  - **Mixed-Policy 目标**(§3.1,Eq.5):\(J_{\text{Mixed}}=\frac{1}{Z}\big(\sum_{j}\sum_t \mathrm{CLIP}(\hat r_{j,t},\hat A_j,\epsilon)+\sum_i\sum_t \mathrm{CLIP}(r_{i,t},\hat A_i,\epsilon)\big)\),off-policy IS 比 \(\hat r_{j,t}=\pi_\theta(\tau_{j,t}|\cdot)/\pi_\phi(\tau_{j,t}|\cdot)\),归一因子 \(Z=\sum_j|\tau_j|+\sum_i|\tau_i|\)。
  - **Policy shaping**(§3.2,Eq.6):用 \(f(\hat r_{j,t})\) 替换 off-policy IS 比并省 clip:\(J_{\text{SHAPING}}=\frac{1}{Z}\big(\sum_j\sum_t f(\hat r_{j,t})\hat A_j+\sum_i\sum_t \mathrm{CLIP}(r_{i,t},\hat A_i,\epsilon)\big)\)。off-policy 项梯度(Eq.7)\(\nabla_\theta J_{\text{SHAPING-OFF}}=\mathbb{E}_{\tau\sim\pi_\phi}\big[f'(\pi_\theta)\tfrac{\pi_\theta}{\pi_\phi}\nabla_\theta\log\pi_\theta\cdot\hat A_j\big]\),可见 \(f'(\pi_\theta)\) 是**逐 logit 的梯度权重**。
  - **为什么选 \(f(x)=x/(x+\gamma)\)**(§3.2 核心直觉,Eq.8):对线性 \(f(\pi)=\pi\)(即 vanilla mixed-policy),\(f'=1\),单 logit 梯度尺度上界 \(\le\pi_\theta(1-\pi_\theta)\),在 \(\pi_\theta\to0\)(低概率关键动作)与 \(\pi_\theta\to1\) 处都**趋零**——故"模型当前几乎不会、但 teacher 在走的关键动作"梯度被压没→熵塌。改用 \(f(x)=x/(x+\gamma)\),其导数 \(f'(x)=\gamma/(x+\gamma)^2\) 在 \(x\to0\) 处**放大**,把梯度重分给低概率动作,提对"陌生但有效决策"的学习(Fig.2 中/右)。App. B.2 还证 \(f(\cdot)\) 正则后 IS 权重方差更小→训练更稳。
  - **收敛保证**(Theorem 1,§3.1):IS 权重 \(w=\pi_\theta/\pi_\phi\) 被裁到 \([\underline w,\overline w]\)、目标 Lipschitz 光滑且梯度 \(\sigma\)-有界时,\(\min_k\mathbb{E}\|\nabla J(\theta_k)\|^2=O(1/\sqrt K)\) 收敛到驻点(证在 App. B.1)。
- 实验与证据:数据 OpenR1-Math-220k 默认子集(94k prompts)→ 过滤 >8192 token + Math-Verify 判错 → **45k prompts + R1 轨迹**(§4;资源表 Table 2 记 on/off 数据用量 \(64\text{K}\times7\,/\,64\text{K}\),即每 prompt 7 条 on-policy+1 条 off-policy 的有效规模)。RL 实践:\(\beta=0\) 去 KL,熵损失系数 0.01,遵 Dr.GRPO 去长度/标准差归一,\(\gamma=0.1\),rollout batch 128 / update batch 64,8 rollouts/prompt(本法 1 off + 7 on),temp 1.0,Math-Verify 为奖励、无格式/长度奖励。评测 6 数学竞赛集(AIME24/25 avg@32、AMC、MATH-500、Minerva、OlympiadBench)+ 3 OOD(ARC-c、GPQA-diamond、MMLU-Pro)。关键数字(Qwen2.5-Math-7B,Table 1):6 数学均值 **50.1**,较此前 RLVR(Oat-Zero 等)**+6.4**;较 On-Policy RL **+4.6**;新发布 AIME'25 上 **+8.1**;3 OOD 均值 **57.8**,超最佳 RLVR(OpenReasoner-Zero)**+6.2**。OOD 增益佐证"非过拟合模仿"。baseline 较公平(同基座对比 on-policy RLVR、SFT、SeqKD、RL w/ SFT Loss、SFT+RL)。
- 假设与失效边界:
  - 【原文】依赖能拿到**高质量外部 teacher 完整轨迹**(R1);\(\gamma=0.1\) 等超参敏感性在 App. E.4;主要在数学+少数 OOD 验证。
  - 【推断】能力上界实质由 **teacher 决定**——"破基座上界"换来"被 teacher 上界封顶";当 teacher 轨迹质量差/覆盖不到目标题型时应失效。这更接近"引导式蒸馏+RL 混合",而非 limit_rlvr 所指"RL 自主发现新能力"。
  - 【推断】\(\pi_\phi=1\) 在 rollout 与训练分布差距大时偏差会被放大(off-policy IS 比退化为 \(\pi_\theta\),失去对 teacher 行为分布的真实校准);长程/多轮场景未验证。
- 祛魅总结【推断】:真贡献是**用"同 group 归一化"这一极简机制实现了模仿/探索的自动切换**,且代码与公式逐项可核,工程可复现性强(这是相对很多混合方法的实打实优势)。被(社区/标题)略微高估的是"突破基座上界"的叙事——它确实突破了"基座自采样上界",但代价是引入了 teacher,本质是把 teacher 的能力蒸进来,而非凭空长出新能力;\(f(x)=x/(x+\gamma)\) 对 low-probability 关键动作的 shaping 思想是其可迁移的真内核。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=token 级(off-policy teacher 轨迹的 IS-加权策略梯度 + on-policy advantage);**改什么**=策略参数(actor)；**何时改**=每个 group 内按 rollout 成败动态(自己全错→teacher 主导);**免梯度?**=否,核心就是梯度(且专门 reshape 梯度幅度);**记忆-技能生命周期**=无显式记忆/技能库,teacher 轨迹是一次性脚手架(扩展版用 ExGRPO 做经验回放,属同仓后续工作);**防遗忘机制**=policy shaping 防熵塌即间接防"探索能力遗忘",但无显式防灾难遗忘模块。
- ⑦ 开源代码+框架/harness:https://github.com/ElliottYan/LUFFY(已克隆 ~36MB)。框架 **veRL**(volcengine/verl)底座,rollout 用 vLLM;核心改动在 `luffy/verl/verl/mix_src`(`mix_actor.py` / `mix_core_alg.py` / `mix_trainer.py` / `mix_vllm_rollout.py`);评测用 Math-Verify。仓内还含后续工作 ExGRPO。可得性:完整、真实、可复现,方法实现与公式(Eq.4-8)逐项可核(`mix_core_alg.py` 中 \(\pi_\phi=1\) 与 shaping \(f\) 均可见)。
- 💰 资源/成本与可扩展性:温度 1.0、max_response_length 8192、rollout batch 128 / update batch 64,8 rollouts/prompt(1 off + 7 on);\(\beta=0\)、熵损失 0.01。GPU 时(Table 2):LUFFY \(77\times8\),LUFFY†(110K 数据版)\(130\times8\)。需预先用 R1 生成 teacher 轨迹(离线一次性成本);训练侧与标准 GRPO 同量级。多基座(7B/1.5B/Instruct/LLaMA-8B)均验证。
- 🎯 对"探索-巩固"对标:**强支撑(且是最近邻竞品)**——LUFFY 几乎就是 idea 的一个已实现版本:teacher 轨迹=稀疏脚手架,"自己失败时模仿 teacher 选路"=探索/选路,"自己成功时保留自采样"=偏向自己走得通的开头;policy shaping 放大低概率关键动作 ≈ "走偏时定向保护恢复分支"。可借组件:(a) **同 group 混入 off-policy 轨迹做归一化**这一极简切换机制(Eq.4);(b) **\(f(x)=x/(x+\gamma)\) 对低概率关键动作的梯度放大**(可直接用于 MTP 前瞻处"关键步"的信用加权,且其 \(f'(x)=\gamma/(x+\gamma)^2\) 形式给出了"放大多少"的可调旋钮)。缺口:LUFFY 是**整条 teacher 轨迹**注入,非 idea 设想的"稀疏单点脚手架/局部接管";teacher 轨迹给的是 prefix 级,不含"前瞻探针"。判定依据:§3.1-3.2 + 代码 mix_src。
- 🔭 开放问题/未来方向:【原文】把 LUFFY 扩到更广领域/模态、进一步精化 policy shaping(§Conclusion);teacher 轨迹质量与覆盖的系统消融未做。【推断】把"整条轨迹注入"细化为"仅在 student 走偏的关键步注入单步 teacher 监督"(更接近稀疏脚手架,省 teacher 成本);用 MTP 前瞻信号判定"何时该让 teacher 接管";\(\pi_\phi=1\) 近似在长程多轮下的偏差修正。

— 残留待核:0(Eq.1–8、\(f(x)=x/(x+\gamma)\) 与梯度上界 \(\pi_\theta(1-\pi_\theta)\)、数据 94k→45k、主结果 50.1/57.8/+6.4/+6.2 均据 PDF §2–§5 + Table 1/2 抄准。)

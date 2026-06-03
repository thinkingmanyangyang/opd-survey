luffy | Learning to Reason under Off-Policy Guidance (LUFFY) | 上海AI实验室·西湖大学·南京大学·港中文(Yafu Li 等;Project Lead Yafu Li) | 2025-04-21 v1 → 2025-06-22 v5;NeurIPS 2025;arXiv 2504.14945 | 主题线 L2(统一SFT-RL)+L1(off-policy指导蒸馏)·相关性 高

**原始论文**:https://arxiv.org/abs/2504.14945

## 一眼看懂
- 🟦 TL;DR:纯 on-policy RL(GRPO)只能在模型"自己已会"的范围里放大,弱模型/难题很快撞天花板;纯 SFT/蒸馏又是死记硬背(behavior cloning),泛化差、熵塌。LUFFY 的做法是把一条**更强 teacher(DeepSeek-R1)的现成正确推理轨迹**和**模型自己采样的 rollout** 塞进同一个 GRPO group 里一起做组内 advantage 归一化——自己全错时 teacher 轨迹(奖励恒正)自然主导信号,自己能解时则保留自我探索;再加一个 policy shaping 函数 f(x)=x/(x+0.1) 专门放大"低概率但关键"动作的梯度、压住熵坍塌(§3.1-3.2)。
- 最巧的一步:**把 off-policy teacher 轨迹混进同一个 group 做组内归一化**。抽掉它(退回普通 GRPO),弱模型上 advantage 全负、学不到任何东西就垮(§实验:on-policy RL 在弱模型/难数据上失败);抽掉 policy shaping(退回线性 shaping),则会因 IS 比率压低关键低概率动作梯度而**熵坍塌、过快收敛**(§3.2,Fig.2/Fig.6)。两者一个负责"能不能学到外部能力",一个负责"学的时候别塌",缺一不可。

## 为什么做
- 研究背景:RLVR(可验证奖励 RL,数学答案对/代码过测才给奖励)让大推理模型涌现多步推理与自反思("aha moment",§摘要/p1)。但 RLVR 本质 **on-policy**——只从模型自己的采样里学,性能受基座自身上界约束(与 limit_rlvr 的结论一致)。
- 解决的具体痛点:① on-policy RL 只能放大已有行为,无法引入真正新的认知能力;弱模型(LLaMA3.1-8B)缺必要基础认知行为,RL 下迅速进入平台期。② 纯 SFT/蒸馏是 behavior cloning,泛化差、易过拟合、致熵坍塌。③ 二者之间缺一个能在"模仿"与"探索"间**动态平衡**的机制。
- 相关工作 & 各自不足:RLVR(R1/o1/Kimi-1.5)受基座封顶;SFT 蒸馏注入外部知识但僵化;此前混合方案缺少"按当前 rollout 成败动态切换"的机制。
- 动机链:on-policy RL 撞基座天花板 → 想引入更强 teacher 轨迹当"认知脚手架" → 但简单拼 SFT+RL 会僵化/熵塌 → 所以要(a)混进同 group 让信号"按需主导"+(b)用 shaping 保护关键动作。
- 与最近邻工作的Δ:相对纯 RLVR,差在**注入了基座外的轨迹**(能破上界);相对纯 SFT/SeqKD,差在**轨迹只在 student 自己失败时主导、成功时退居其次**(on-policy 自适应)+ policy shaping 防熵塌。关键有用点:advantage 组内归一化天然实现了"自己会就探索、不会就模仿"的开关,无需手调权重。

## 怎么做 + 靠不靠谱
- 方法流水线:① 对每个 prompt,准备一条 off-policy teacher 轨迹(带 prefix_mask,奖励恒正)+ 若干条 on-policy rollout;② 二者混入同一 GRPO group 一起算组内 advantage(§3.1,Eq.3 扩展);③ off-policy 项用重要性采样比、on-policy 项用标准策略梯度;④ 对 off-policy IS 比率施加 policy shaping f(x)=x/(x+γ),γ=0.1(§3.2,Eq.6-7);⑤ actor 更新合并 off_pg_loss + SFT 项(-log_prob masked mean)+ 可选 KL,梯度更新 student。
- 逐组件必要性:
  - **Mixed-Policy group 归一化**:负责"破基座上界"。有消融——纯 on-policy RL 在弱模型/难数据上完全失败(§实验),证明其必要。
  - **Policy shaping f(x)=x/(x+0.1)**:负责"防熵坍塌/防过快收敛"。有消融——Fig.6 显示 Mixed-Policy 无 shaping 时早期猛涨但随即熵塌、后期被反超(§3.2),线性 shaping(vanilla)即原始问题来源。
  - **π_old=1 计算简化**:论文为计算效率取 π_old=1,故 off_ratio=exp(log_prob)(代码 mix_core_alg.py)。属工程近似,论文承认可能引入偏差。无单独消融。
- 关键机制/公式(直觉):f(x)=x/(x+γ) 的导数 f′(x)∝1/(x+γ)²,在 x→0(低概率动作)时放大梯度——即"越是模型当前几乎不会、但 teacher 在走的关键动作,越要使劲学",这正是对抗"IS 比率把低概率动作梯度压没→熵塌"的解药(§3.2)。
- 实验与证据:数据 OpenR1-Math-220k 子集(~64k,teacher 轨迹来自 DeepSeek-R1);评测 6 数学竞赛集(AIME24/25、AMC、MATH-500、Minerva、OlympiadBench)+ 3 OOD(ARC-c、GPQA-diamond、MMLU-Pro)。关键数字(Qwen2.5-Math-7B,Table 1):6 数学均值 **50.1**,较此前 RLVR(Oat-Zero 数学 43.7)**+6.4**;3 OOD 均值 **57.8**,较此前最佳 **+6.2**。baseline 较公平(同基座对比 on-policy RLVR、SFT、SeqKD)。OOD 增益佐证"非过拟合模仿"——较好地回答了核心问题。
- 假设与失效边界:
  - 【原文】依赖能拿到**高质量外部 teacher 完整轨迹**(R1);γ=0.1 等超参敏感性在 App. E.4;主要在数学+少数 OOD 验证。
  - 【推断】能力上界实质由 **teacher 决定**——"破基座上界"换来"被 teacher 上界封顶";当 teacher 轨迹质量差/覆盖不到目标题型时应失效。这更接近"引导式蒸馏+RL 混合",而非 limit_rlvr 所指"RL 自主发现新能力"。
  - 【推断】π_old=1 在 rollout 与训练分布差距大时偏差会被放大,长程/多轮场景未验证。
- 祛魅总结【推断】:真贡献是**用"同 group 归一化"这一极简机制实现了模仿/探索的自动切换**,且代码与公式逐项可核,工程可复现性强(这是相对很多混合方法的实打实优势)。被(社区/标题)略微高估的是"突破基座上界"的叙事——它确实突破了"基座自采样上界",但代价是引入了 teacher,本质是把 teacher 的能力蒸进来,而非凭空长出新能力;low-probability 关键动作的 shaping 思想是其可迁移的真内核。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=token 级(off-policy teacher 轨迹的 IS-加权策略梯度 + on-policy advantage);**改什么**=策略参数(actor)；**何时改**=每个 group 内按 rollout 成败动态(自己全错→teacher 主导);**免梯度?**=否,核心就是梯度(且专门 reshape 梯度幅度);**记忆-技能生命周期**=无显式记忆/技能库,teacher 轨迹是一次性脚手架(扩展版用 ExGRPO 做经验回放,属同仓后续工作);**防遗忘机制**=policy shaping 防熵塌即间接防"探索能力遗忘",但无显式防灾难遗忘模块。
- ⑦ 开源代码+框架/harness:https://github.com/ElliottYan/LUFFY(已克隆 ~36MB)。框架 **veRL**(volcengine/verl)底座,rollout 用 vLLM;核心改动在 `luffy/verl/verl/mix_src`(mix_actor.py / mix_core_alg.py / mix_trainer.py / mix_vllm_rollout.py);评测用 Math-Verify。仓内还含后续工作 ExGRPO。可得性:完整、真实、可复现,方法实现与公式逐项可核。
- 💰 资源/成本与可扩展性:温度 1.0、max_response_length 8192、rollout batch 128 / update batch 64(Qwen2.5-Math-7B-Zero 设置,§实验)。需预先用 R1 生成 teacher 轨迹(离线一次性成本);训练侧与标准 GRPO 同量级。多基座(7B/1.5B/Instruct/LLaMA-8B)均验证。
- 🎯 对"探索-巩固"对标:**强支撑(且是最近邻竞品)**——LUFFY 几乎就是 idea 的一个已实现版本:teacher 轨迹=稀疏脚手架,"自己失败时模仿 teacher 选路"=探索/选路,"自己成功时保留自采样"=偏向自己走得通的开头;policy shaping 放大低概率关键动作 ≈ "走偏时定向保护恢复分支"。可借组件:(a) **同 group 混入 off-policy 轨迹做归一化**这一极简切换机制;(b) **f(x)=x/(x+γ) 对低概率关键动作的梯度放大**(可直接用于 MTP 前瞻处"关键步"的信用加权)。缺口:LUFFY 是**整条 teacher 轨迹**注入,非 idea 设想的"稀疏单点脚手架/局部接管";teacher 轨迹给的是 prefix 级,不含"前瞻探针"。判定依据:§3.1-3.2 + 代码 mix_src。
- 🔭 开放问题/未来方向:【原文】把 LUFFY 扩到更广领域/模态、进一步精化 policy shaping(§摘要末/Conclusion);teacher 轨迹质量与覆盖的系统消融未做。【推断】把"整条轨迹注入"细化为"仅在 student 走偏的关键步注入单步 teacher 监督"(更接近稀疏脚手架,省 teacher 成本);用 MTP 前瞻信号判定"何时该让 teacher 接管";π_old=1 近似在长程多轮下的偏差修正。

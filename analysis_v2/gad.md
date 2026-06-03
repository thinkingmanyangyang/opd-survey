gad | Black-Box On-Policy Distillation of Large Language Models (GAD) | 微软研究院（Tianzhu Ye、Li Dong 共同一作；Furu Wei 等，General AI 组） | 2026-01-08 (arXiv:2511.10643 v3，首发 2025-11) | L1 OPD/自蒸馏 · 相关性高

**原始论文**：https://arxiv.org/abs/2511.10643

## 一眼看懂
- 🟦 TL;DR：教师是只吐文本的闭源 API（如 GPT-5-Chat）时，学生拿不到 logits、做不了白盒/likelihood 蒸馏，连"在自己生成上学"（on-policy）都没法做——因为教师无法给学生自己的 rollout 打概率分。GAD 把学生当 GAN 的生成器 G，另训一个判别器 D 去区分"这段文本是学生还是教师写的"，G 努力骗过 D、D 努力分开二者，构成极大极小博弈。D 本质是一个"随学生一起进化的 on-policy 奖励模型"，于是在黑盒下也能做 on-policy 蒸馏，且因 D 一直追着学生当前行为更新，不会像冻结 reward model 那样被 reward hacking【原文 Abstract, §2.1, Table 1】。
- 最巧的一步：把判别器 D 解读成"在线奖励模型"（Table 1：RLHF 的 reward model 训完冻结、易被 hack；GAD 的 D 在 minimax 里持续共更）。抽掉"D 在线共更"这一步——把 D 冻结成普通 reward model——就退化成 off-policy 判别器，论文 Fig.6 实测约 300 步后即 reward hacking（响应暴涨到 ~1300 token）。所以"on-policy 判别器"是 GAD 既能学又不崩的命门【原文 §3.3 Fig.6】。

## 为什么做
- 研究背景：知识蒸馏用大教师造小学生。白盒蒸馏能拿教师 logits/隐状态，用 forward/reverse KLD 对齐（KR16、MiniLLM）；近期白盒研究强调 **on-policy** 的价值——学生学自己生成的响应（reverse KLD）能 mode-seeking、减小 exposure bias，优于纯 teacher-forcing【原文 §1, §4】。
- 解决的具体痛点：教师是闭源 API 时只能看到文本（黑盒），没有概率级监督，likelihood 目标失效；学生/教师 tokenizer 不兼容时 likelihood 也失效。黑盒标准做法只能对教师响应做 SeqKD（序列级 SFT/行为克隆），而它易过拟合教师局部 n-gram、OOD 泛化差【原文 §1, §4】。
- 相关工作 & 各自不足：① 白盒 KD（forward/reverse KLD、隐状态对齐）——依赖完整教师访问，闭源不可用；② SeqKD（在教师响应上 SFT）——黑盒下的标准，但记忆局部词汇、mode-covering、OOD 弱；③ RLHF 固定 reward model——预训后冻结，策略易 reward hacking【原文 §1, §4, Table 1】。
- 动机链：要黑盒 + 要 on-policy 的好处（mode-seeking、低 exposure bias、好 OOD）→ 但黑盒下教师无法评价学生自生成 → 所以需要一个"不靠教师概率、又能对学生自生成给反馈"的信号 → GAN 判别器恰好提供这种隐式、在线的反馈。
- 与最近邻工作的Δ：最近邻是 SeqKD（黑盒标准）与白盒 on-policy（MiniLLM/GKD）。Δ 在于 GAD 把"on-policy 学习"从白盒搬到黑盒：用判别器代替"教师概率"作为对学生自生成的反馈源。为什么有用——RL（GRPO 优化 D 给的 reward）比 SFT 泛化更好（CZY+25、WZZ+25），且在线判别器避免了固定 reward model 的 hacking【原文 §1, §3.2 Fig.6】。

## 怎么做 + 靠不靠谱
- 方法流水线（5 步）：① 构数据 T={(x,y_t)}：遍历 prompt x、采教师响应 y_t；② **Warmup（关键）**：对生成器用教师响应做 CE/SFT 一个 epoch、对判别器用同数据做 BT loss 一个 epoch（且 warmup 阶段先单独训 D 10 步），保证 G-D 平衡；③ GAD 循环：每 batch 采学生响应 G(x)，用 D(G(x)) 当 reward 经 **GRPO** 更新 G，再用 BT loss 更新 D；④ 判别器结构：由生成器参数初始化 + 一个标量预测头（末 token 隐状态投影为序列级分数）；⑤ 交替更新至收敛【原文 §2.1–2.3, Algorithm 1】。
- 逐组件必要性：
  - **Warmup（生成器侧）**：去掉 → 直接用未 SFT 的 instruct 模型当 G/D 初始化 → D 早期太容易区分二者、分布鸿沟大、对抗失效，掉点（Qwen2.5-7B：LMSYS 50.8→49.7）【原文 Table 3】。
  - **Warmup（判别器侧）**：去掉（D 用原始模型初始化、G 已 SFT）→ G-D 失衡、D 反馈信息量不足、G 几乎学不动（LMSYS 50.8→49.0、Others 50.0→47.7，掉得更狠）【原文 Table 3】。
  - **BT loss vs CE loss（判别器损失）**：默认 Bradley-Terry 优于二分类 CE（Qwen2.5-3B：LMSYS 48.9 vs 47.9）；有消融【原文 Table 4, Eq.4】。
  - **判别器尺寸**：D 与 G 同尺寸最优；增大 D（3B→7B、7B→14B）反而掉点。有消融【原文 Table 5】。
  - **on-policy vs off-policy 判别器**：核心消融，off-policy（冻结 D）约 300 步后 reward hacking，on-policy 稳定数千步【原文 Fig.6】。
- 关键机制/公式（直觉）：价值函数 max_G min_D V = E[−log σ(D(y_t)−D(G(x)))]（BT 偏好：教师分应高于学生）。G 因采样不可微，把 D(G(x)) 当 reward 用 GRPO 做策略梯度（组内 N=8 响应、按均值/标准差归一算 advantage，省略 KL 与 clip 表述但实现里 KL β=0.001）；D 把组内每个学生响应 y_s^i 与同一教师响应 y_t 配对，最小化组内平均 BT loss【原文 §2.2, Appendix A.1 Eq.5–8】。
- 实验与证据：
  - **数据/模型**：训练用 LMSYS-Chat-1M-Clean 采 200K prompt + 收 GPT-5-Chat 教师响应；学生 Qwen2.5-Instruct(3B/7B/14B)、Llama-3.2-3B / Llama-3.1-8B-Instruct；评测 LMSYS 测试集 500（主）+ OOD：Dolly(500)/SelfInst(252)/Vicuna(80)；GPT-4o 打分（先生成参考答案再比分）+ 人工评测【原文 §3.1】。
  - **关键数字**：全数据集×全尺寸 GAD>SeqKD（Table 2）。Qwen2.5-14B+GAD 在 LMSYS 得 **52.1**，超教师 GPT-5-Chat（**51.7**）；3B+GAD≈7B+SeqKD、7B+GAD≈14B+SeqKD（尺寸"升一级"）；OOD 上 SeqKD 增益微弱/为负而 GAD 稳健提升；人工评测对 before-distill / SeqKD 胜率多 >50%、负率 <30%【原文 §3.2, Table 2, Fig.1, Fig.3】。
  - **机制证据**：N-gram 重叠显示 SeqKD 过拟合教师局部词汇、GAD 学全局风格（Fig.4）；toy 高斯混合上 SeqKD mode-covering、GAD（REINFORCE）mode-seeking（Fig.5）；off-policy 判别器 reward hacking（Fig.6）；Table 6 显示 SeqKD 缩短响应贴合教师长度分布、GAD 保留学生长度分布【原文 §3.3, Table 6】。
  - **baseline 公平吗**：基本公平——同训练预算（3 epoch、batch 256、~2400 步）、对 GAD/SeqKD 都搜了 lr∈[1e-6,5e-6]。但**评测全靠 GPT-4o-as-judge**这一代理指标（GPT-4o 分 = 学生分/(学生分+参考分)），是主要软肋。
  - **"看着强但没回答核心问题"**：① "接近 GPT-5-Chat"仅在 14B+LMSYS+GPT-4o 评分下成立，是上限演示而非任务正确率；② 全程聊天/通用指令域，数学/代码强推理域未测。
- 假设与失效边界：
  - 【原文】教师只返回文本即可（黑盒），不需 logits；甚至 tokenizer 不兼容也行（Table 7 用 Qwen2.5-14B 教师蒸到 Llama，仍有效）。
  - 【原文】对抗训练稳定性靠 warmup 保证 G-D 平衡，但未给收敛性证明。
  - 【推断】依赖 GPT-4o judge 的偏好；若教师风格被 judge 系统性偏好，"超过教师"可能部分是评判偏差而非真超越（依据：所有自动指标都来自 GPT-4o，§3.1）。
  - 【推断】GAD 在强推理域（数学/代码可验证正确率）是否仍优于 SeqKD 未知——本文只在主观质量评分占主导的聊天域验证（依据：实验全在 LMSYS/Dolly/SelfInst/Vicuna，§3.1, §3.2）。
- 祛魅总结：
  - 真贡献【推断】：把"on-policy 蒸馏"成功迁移到黑盒，且给出"判别器=在线 reward model→抗 hacking"这一干净的 GAN↔RLHF 映射，机制分析（N-gram/toy/Fig.6）扎实。
  - 包装/高估【推断】："Qwen2.5-14B 比肩 GPT-5-Chat"是建立在 GPT-4o 评分上的上限演示，易被过度解读；GAD 比 SeqKD 重（要同时维护同尺寸 G+D，3 epoch≈2400 步，14B 蒸馏需 16×H100 约 30 小时），论文未强调成本权衡（依据：Appendix A.2）。
  - 低估【推断】：OOD 泛化优势（SeqKD 甚至为负、GAD 稳健）其实是比"接近教师"更稳的卖点，但被标题的"comparable to teacher"盖过。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=判别器给的序列级标量分 D(G(x))（"像不像教师写的"），不是 token 概率｜**改什么**=学生全参（policy）+判别器全参（reward model）｜**何时改**=训练期 on-policy，每 batch 交替更新 G 与 D｜**免梯度?**=否（G 走 GRPO 策略梯度、D 走 BT loss 梯度；采样不可微处用 RL 绕过）｜**记忆-技能生命周期**=无显式记忆/技能库，知识固化进学生参数｜**防遗忘机制**=无专门机制（warmup SFT 提供起点；靠教师文本对齐）。
- ⑦ 开源代码+框架/harness：项目页 ytianzhu.github.io/Generative-Adversarial-Distillation/；代码仓 **microsoft/LMOps 的 `gad/` 子目录**（aka.ms/GAD-github 重定向至此，本地已 clone 488KB），但**仅含环境/数据/启动脚本**（scripts/、tools/export_lmsys_parquet.py、local_setup.sh）；**算法核心在外部 fork** `github.com/YTianZHU/verl`（基于 **veRL**，README 明确"hack critic 模块当判别器"，含 seqkd/warmup/gad/eval 分支，SeqKD/warmup 的 SFT 在 dp_actor.py）。数据/模型在 HF（ytz20/LMSYS-Chat-GPT-5-Chat-Response、ytz20/gad-models）。推荐 docker czwin32768/verl2:v0.2.0-vllm085（py3.10/torch2.6/vllm0.8.5）。**未完整本地化**：核心算法不在本地 clone 的 LMOps/gad 内，需到 YTianZHU/verl fork 审计〔待核：fork 内 GAD 算法实现细节未逐行核对〕。
- 💰 资源/成本与可扩展性：蒸 Qwen2.5-14B（GPT-5-Chat 教师）约 **30 小时 / 16×H100**；3 epoch（1 warmup+2 GAD）≈2400 步，batch 256、PPO mini-batch 256，N=8，KL β=0.001，温度 0.8，lr 1e-6（GAD 阶段）。需同时维护 G+D（同尺寸最优），比纯 SeqKD 重【原文 §3.1, Appendix A.2】。
- 🎯 对"探索-巩固"对标：**竞品/可借组件**。一句判定：GAD 是"黑盒 on-policy 蒸馏"的强 baseline，提供两个可借组件——(a) **在线协同进化的奖励/判别器**思路可类比"teacher 当随学生演化的稀疏脚手架"（D 总针对学生当前行为给反馈，正对应"巩固/回轨"里 teacher 需感知学生当前状态）；(b) **GAN↔RLHF 映射→抗 reward hacking** 对"固化进参数而不崩坏"有借鉴。缺口：GAD 无路径/step 级监督、无 MTP 前瞻、无记忆/技能库，且依赖教师文本（非"学生自选可走通开头"）——与"探索/选路"这一支撑较弱，更贴"巩固"的对齐侧。依据：D 共更对应在线 teacher 信号，但 GAD 学的是整段文本风格而非分支选择（§2.1, §3.3）。
- 🔭 开放问题/未来方向：【原文】未验证数学/代码等强推理域的黑盒蒸馏（§4 仅提"近期工作在教师推理轨迹上 SFT"，GAD 本身未做）；GAN 收敛性无理论保证，靠 warmup 经验缓解（§2.2）。【推断】把 D 做成 step/段级判别器（而非整段 1 个标量分）或可引入更细的路径级信用，接近"探索-巩固"的分支级监督；用真实任务正确率替代 GPT-4o judge 以去除评判偏差；探索教师=开源大模型但 tokenizer 不兼容时（Table 7）的更系统验证。

读到PDF? 是（全文 15 页已读）｜L线 L1（黑盒 on-policy 蒸馏）｜对标结论 竞品+可借组件（在线协同进化奖励/判别器、GAN↔RLHF 抗 hacking；缺路径级/MTP/记忆）｜残留待核 1（YTianZHU/verl fork 内 GAD 算法实现细节未逐行核对）

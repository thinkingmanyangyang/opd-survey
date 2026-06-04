gad | Black-Box On-Policy Distillation of Large Language Models (GAD) | 微软研究院（Tianzhu Ye、Li Dong 共同一作；Furu Wei 等,General AI 组） | 2026-01-08 (arXiv:2511.10643 v3,首发 2025-11) | L1 OPD/自蒸馏 · 相关性高

**原始论文**：https://arxiv.org/abs/2511.10643

## 一眼看懂
> 一句话导读：教师是只能吐文本、看不到内部概率的闭源 API 时,常规蒸馏全失效;GAD 借 GAN 的思路另训一个判别器当"在线打分器",让学生在黑盒下也能做 on-policy 蒸馏。

- 🟦 TL;DR：先说困境。当教师是只吐文本的闭源 API(如 GPT-5-Chat)时,学生拿不到 logits(模型输出的原始概率分数),做不了白盒蒸馏,也做不了 likelihood 蒸馏(对齐似然/概率的蒸馏)。更麻烦的是,连"在自己生成上学"(on-policy)都没法做——因为教师无法给学生自己采出的 rollout(学生跑出来的一整条响应)打概率分。
  - GAD 的解法:把学生当成 GAN(生成对抗网络)里的生成器 \(G\),另外训一个判别器 \(D\),让 \(D\) 去区分"这段文本是学生写的还是教师写的"。\(G\) 努力骗过 \(D\)、\(D\) 努力把二者分开,构成一个极大极小博弈(minimax,两方目标相反、轮流对抗)。
  - 关键点:\(D\) 本质是一个"随学生一起进化的 on-policy 奖励模型"。于是黑盒下也能做 on-policy 蒸馏;而且 \(D\) 一直追着学生当前行为更新,不会像冻结的 reward model 那样被 reward hacking(策略钻奖励模型的空子、刷高分却没真本事)【原文 Abstract, §2.1, Table 1】。
- 最巧的一步：把判别器 \(D\) 解读成一个"在线奖励模型"。对照见 Table 1：RLHF 里的 reward model 训完就冻结、容易被 hack;而 GAD 的 \(D\) 在 minimax 博弈里和学生持续共同更新。
  - 反过来验证它的重要性:抽掉"\(D\) 在线共更"这一步——把 \(D\) 冻结成普通 reward model——GAD 就退化成 off-policy 判别器。论文 Fig.6 实测约 300 步后就出现 reward hacking(响应长度暴涨到 ~1300 token)。
  - 所以"on-policy 判别器"是 GAD 既能学到东西、又不会崩的命门【原文 §3.3 Fig.6】。

## 为什么做
> 一句话导读：白盒 on-policy 蒸馏效果好但要看到教师概率,黑盒标准做法 SeqKD 只能死记教师文本、泛化差;GAD 想把白盒的 on-policy 好处搬到黑盒下,缺口就用一个对抗判别器补上。

- 研究背景：知识蒸馏([HVD15] Hinton)用大教师造小学生。
  - 学生能访问教师**预测分布 / 隐状态**的设定叫**白盒蒸馏**。标准做法有两类:一是对齐输出分布(forward/reverse KLD,[SST+20] LightPAFF、[GDWH24] MiniLLM),二是对齐内部状态([JYS+20] TinyBERT、[SCGL19] patient KD、[WWD+20] MiniLM、[WBH+21] MiniLMv2)。
  - 但教师是闭源 API(如 GPT-5)时,白盒访问不现实,只能看到**教师生成的文本**。这就是更难的**黑盒蒸馏**:缺细粒度概率监督,基于 likelihood 的目标失效;师生 **tokenizer 不兼容**(分词器不同、token 编号对不上)时 likelihood 目标也失效。
- 解决的具体痛点 & 两条已有路线各自短板：
  - **白盒 on-policy 这条线**([GDWH24] MiniLLM、[AVZ+24] GKD、[LL25] Thinking Machines OPD 博客、[YLY+25] Qwen3)近期强调 **on-policy 学习**的价值:学生学**自己生成的响应**(reverse-KLD)能促 mode-seeking(只聚焦教师的高概率模式)、减 exposure bias(训练只见真值、推理却要续写自己输出导致的偏差),优于纯 teacher-forcing。**短板**:全靠完整教师访问,闭源 / API-only 教师下不可用。
  - **黑盒标准做法 SeqKD**([KR16] sequence-level KD、[TGZ+23] Alpaca、[CLL+23] Vicuna、[PLH+23] GPT-4 instruct、[ZLX+23] LIMA)只对**教师响应做 SFT / 行为克隆**。**短板**:会记忆教师的**局部 n-gram**(连续几个词的固定搭配)、是 mode-covering(铺开覆盖教师整个分布)、OOD 泛化弱(分布外、训练没见过的输入上表现差)。本文有两处实证:Fig.4 的 N-gram 重叠、Table 6 的长度分布。近期 [MYS+25] s1、[GMK+25] OpenThoughts、[YHX+25] LIMO、[GYZ+25] R1 把 SeqKD 扩到"在教师**推理轨迹**上 SFT"来提升推理,但仍是 SFT 范式。
  - **RLHF 固定 reward model**([OWJ+22] InstructGPT)预训后**冻结**,策略容易 **reward hacking**([SHKK22] reward gaming)。
- 动机链(把黑盒 + on-policy 两个诉求接起来)：
  - 想要黑盒,又想保住 on-policy 的好处(mode-seeking、低 exposure bias、好 OOD)。
  - 但黑盒下教师无法评价学生自生成(没有概率级监督)。
  - 所以需要一个信号:不靠教师概率,却能对学生自生成给反馈。
  - GAN 判别器恰好提供这种**隐式、在线**的反馈([GPAM+14] GAN、[YZWY17] SeqGAN、[HE16] GAIL 是直系前身——SeqGAN 早已用 policy gradient 训文本 GAN,GAIL 把判别器当 reward)。而判别器 = "在线 reward model",又天然规避固定 RM 的 hacking。
- 本文站在谁肩上 & 与最近邻工作的精确差异：最近邻是 **SeqKD**(黑盒标准)与**白盒 on-policy**(MiniLLM/GKD)。
  - Δ:GAD 把"on-policy 学习"从白盒搬到黑盒——用**判别器代替"教师概率"**,作为对学生自生成的反馈源。
  - 技术承袭 GAN/SeqGAN/GAIL 的对抗框架,但有两点不同:一是**目标用 Bradley-Terry 偏好**(一种成对偏好模型)而非二分类 CE(§3.4 消融证 BT 更好);二是**强调判别器在线共更**(对照 RLHF 固定 RM,Table 1)。
  - 为什么有用——RL(GRPO 优化 \(D\) 给的 reward)比 SFT 泛化更好([CZY+25] SFT memorizes RL generalizes、[WZZ+25] SFT 泛化的 RL 视角),且在线判别器避免固定 RM 的 hacking【原文 §1, §3.3 Fig.6】。

## 怎么做 + 靠不靠谱
> 一句话导读：先用 SeqKD 暖身给学生和判别器一个起点,再进入对抗循环——学生采一批响应、判别器打分当 reward、GRPO 更新学生、BT loss 更新判别器,推理时只留学生。消融逐项验证了暖身、BT、判别器同尺寸、on-policy 这几步缺一不可。

- **形式化与 minimax 价值函数(§2.1,Eq.1)**：这是条件文本生成任务,目标是让学生分布 \(q_\theta(y\mid x)\) 逼近教师分布 \(p(y\mid x)\)。
  - 先构数据 \(\mathcal{T}=\{(x,y_t)\}\):遍历每个 prompt \(x\),为它各采一条教师响应 \(y_t\)。
  - 生成器 \(G\) = 学生模型,产出 \(G(x)\);判别器 \(D\) 给 \([x,y]\) 打一个**序列级标量分** \(D([x,y])\)(下文简写 \(D(y)\))。
  - 两玩家做极大极小博弈:
  \(\displaystyle \max_{G}\ \min_{D}\ V(G,D)=\mathbb{E}_{(x,y_t)\sim\mathcal{T}}\big[-\log\sigma\big(D(y_t)-D(G(x))\big)\big],\)
  其中 \(\sigma(\cdot)\) 是 sigmoid。这里用 **Bradley-Terry 模型**([BT52])刻画一个成对偏好:教师响应的分应当高于学生响应的分。
- **判别器结构(§2.1,复现关键)**：\(D\) **由生成器参数初始化**,外加一个**标量预测头**。这个 head 把序列**最后一个 token 的末层隐状态**投影成一个标量,作为整条序列的分数。也就是说,\(D\) 与 \(G\) 同架构、同尺寸(消融 Table 5 证同尺寸最优)。
- **方法流水线(5 步,Algorithm 1 + §2.2),每步标出输入与输出**：
  - **Step 0 构数据**:遍历 prompt \(x\)、采教师响应 \(y_t\),得到 \(\mathcal{T}=\{(x,y_t)\}\)。实现上从 LMSYS-Chat-1M-Clean 采 200K prompt,收 GPT-5-Chat 响应。
  - **Step 1 Warmup(关键,§2.2)**:对**整个 \(\mathcal{T}\) 跑 1 个 epoch**。生成器 \(G\) 用教师响应 \(y_t\) 做 **CE/SFT**(即先做一遍 SeqKD 当起点);判别器 \(D\) 用同数据做 **BT loss**(Eq.3)。附录 A.2 补了一个细节:warmup 阶段**先单独训 \(D\) 10 步**,再开始联合训 \(G,D\),以保证 G 和 D 的能力平衡。输出是已 SFT 的 \(G\) 和已初步判别的 \(D\)。
  - **Step 2 GAD 循环(2 个 epoch)**:对每个 batch \((x,y_t)\) 做三件事——
    - (a) 用当前 \(G\) 对 \(x\) 采**一组** \(N\) 条学生响应 \(\{y_s^i\}_{i=1}^{N}\)(\(N=8\),temperature 0.8);
    - (b) 把 \(D(G(x))\) **当 reward**,用 **GRPO** 更新 \(G\)(见下方 Eq.5-7);
    - (c) 用 **BT loss** 更新 \(D\)(组内每个 \(y_s^i\) 与同一 \(y_t\) 配对,Eq.8)。
  - **Step 3 交替至收敛**,返回 \(G\)。
- **生成器 / 判别器的真实目标形式(§2.2 + 附录 A.1)**：
  - 生成器目标(Eq.2):\(\displaystyle \max_{G}\ \mathbb{E}_{(x,y_t)\sim\mathcal{T}}\big[D(G(x))\big]\)。由于 \(G(x)\) 的**采样操作对 \(\theta\) 不可微**,这里把 \(D(G(x))\) 当 reward,用 **policy gradient**([SMSM99])经 **GRPO**([SWZ+24])优化。
  - GRPO 实现(附录 A.1):对每个 \(x\) 采一组 \(N\) 条响应 \(\{y_s^i\}\),奖励 \(r_s^i=D(y_s^i)\)(Eq.5);组内归一算 advantage(Eq.6):
    \(\displaystyle A_i=\frac{r_s^i-\operatorname{mean}\big(\{r_s^j\}_{j=1}^{N}\big)}{\operatorname{std}\big(\{r_s^j\}_{j=1}^{N}\big)};\)
    目标(Eq.7,**省略了 KL 与 clip 的表述**,但实现里 KL \(\beta=0.001\)):\(\displaystyle \max_{G}\ \mathbb{E}_{x,\,\{y_s^i\}\sim q_G}\Big[\tfrac{1}{N}\sum_{i=1}^{N}A_i\Big]\)。
  - 判别器目标(Eq.3 / 组内 Eq.8):默认用 **Bradley-Terry**。把组内每条学生响应 \(y_s^i\) 与同一教师响应 \(y_t\) 配对,最小化组内平均
    \(\displaystyle \min_{D}\ \mathbb{E}_{x,\,\{y_s^i\}\sim q_G}\Big[\tfrac{1}{N}\sum_{i=1}^{N}-\log\sigma\big(D(y_t)-D(y_s^i)\big)\Big],\)
    其中 \(D(y_t)\) 在组内共享。**消融对照用的 CE loss**(Eq.4,二分类判别器损失)则是
    \(\displaystyle \min_{D}\ \mathbb{E}_{(x,y_t)\sim\mathcal{T}}\big[-\log\sigma(D(y_t))-\log\big(1-\sigma(D(G(x)))\big)\big].\)
- **训练-推理数据怎么流动**:
  - 训练时,教师响应是**离线一次性收好**的固定 \(y_t\)(只查一次 GPT-5),而学生响应每步**在线现采**。
  - reward 不来自教师,而来自**与学生共同更新的 \(D\)**。
  - 推理时**只留 \(G\)**(贪心、max response 1536),\(D\) 丢弃。
  - 这套流程就是"黑盒(教师只给文本)+ on-policy(学生学自生成,反馈来自在线 \(D\))"的实现。
- 逐组件必要性(消融均以 GPT-4o 评分衡量)：
  - **Warmup(生成器侧)**:去掉它(直接用未 SFT 的 instruct 模型当 \(G/D\) 初始化),则 \(D\) 早期太容易区分二者、分布鸿沟太大、对抗失效,掉点(Qwen2.5-7B:LMSYS 50.8→49.7)【Table 3】。
  - **Warmup(判别器侧)**:去掉它(\(D\) 用原始模型初始化、\(G\) 已 SFT),则 G 和 D 失衡、\(D\) 反馈信息量不足、\(G\) 几乎学不动(LMSYS 50.8→49.0、Others 50.0→47.7,掉得更狠)【Table 3】。
  - **BT loss vs CE loss**:默认的 BT 优于二分类 CE(Qwen2.5-3B:LMSYS 48.9 vs 47.9、Others 47.9 vs 46.4)【Table 4, Eq.3 vs Eq.4】。
  - **判别器尺寸**:\(D\) 与 \(G\) 同尺寸最优;把 \(D\) 增大(3B→7B:48.9→47.8;7B→14B:50.8→50.5)反而掉点【Table 5】。
  - **on-policy vs off-policy 判别器**:这是核心消融。off-policy 做法是——SeqKD 暖身后**冻结 \(G\)** 训 \(D\) 2 epoch,再拿冻结的 \(D\) 当 RM 去训 \(G\)。结果约 300 步后就 reward hacking(响应暴涨到 ~1300 token);而 on-policy 版能稳定跑数千步【Fig.6】。
- 实验与证据:
  - **数据 / 模型**:训练用 LMSYS-Chat-1M-Clean 采 200K prompt,再收 GPT-5-Chat 教师响应。学生用 Qwen2.5-Instruct(3B/7B/14B)、Llama-3.2-3B / Llama-3.1-8B-Instruct。评测主集是 LMSYS 测试集 500,OOD 集是 Dolly(500)/SelfInst(252)/Vicuna(80)。评分方式:**GPT-4o 打分**(先生成参考答案再比分,分 = 学生分 / (学生分 + 参考分))+ 人工评测【§3.1】。
  - **关键数字**:全数据集 × 全尺寸下 GAD 都 > SeqKD(Table 2)。亮点:Qwen2.5-14B + GAD 在 LMSYS 得 **52.1**,超过教师 GPT-5-Chat 的 **51.7**;3B+GAD ≈ 7B+SeqKD、7B+GAD ≈ 14B+SeqKD(相当于把尺寸"升了一级");OOD 上 SeqKD 增益微弱甚至为负,而 GAD 稳健提升;人工评测对 before-distill / SeqKD 的胜率多 >50%、负率 <30%【§3.2, Table 2, Fig.1, Fig.3】。
  - **机制证据**:N-gram 重叠显示 SeqKD 过拟合教师的局部词汇、GAD 学到的是全局风格(Fig.4);在 toy 高斯混合分布上,SeqKD 表现为 mode-covering、GAD(REINFORCE)表现为 mode-seeking(Fig.5);off-policy 判别器出现 reward hacking(Fig.6);Table 6 显示 SeqKD 会缩短响应去贴合教师的长度分布、GAD 则保留学生自己的长度分布【§3.3, Table 6】。
  - **baseline 公平吗**:基本公平——训练预算相同(3 epoch、batch 256、约 2400 步),对 GAD 和 SeqKD 都搜了 lr ∈ [1e-6, 5e-6]。但**评测全靠 GPT-4o-as-judge** 这一代理指标,是主要软肋。
  - **"看着强但没回答核心问题"**:① "接近 GPT-5-Chat"只在 14B + LMSYS + GPT-4o 评分这一组合下成立,是上限演示而非任务正确率;② 实验全程在聊天 / 通用指令域,数学 / 代码这类强推理域没测。
- 假设与失效边界:
  - 【原文】教师只返回文本即可(黑盒),不需 logits;甚至 tokenizer 不兼容也行(Table 7 用 Qwen2.5-14B 教师蒸到 Llama,仍有效)。
  - 【原文】对抗训练稳定性靠 warmup 保证 G-D 平衡,但未给收敛性证明。
  - 【推断】依赖 GPT-4o judge 的偏好;若教师风格被 judge 系统性偏好,"超过教师"可能部分是评判偏差而非真超越(依据:所有自动指标都来自 GPT-4o,§3.1)。
  - 【推断】GAD 在强推理域(数学/代码可验证正确率)是否仍优于 SeqKD 未知——本文只在主观质量评分占主导的聊天域验证(依据:实验全在 LMSYS/Dolly/SelfInst/Vicuna,§3.1, §3.2)。
- 祛魅总结:
  - 真贡献【推断】:把"on-policy 蒸馏"成功迁移到黑盒,并给出"判别器 = 在线 reward model → 抗 hacking"这一干净的 GAN↔RLHF 映射;机制分析(N-gram / toy / Fig.6)扎实。
  - 包装 / 高估【推断】:"Qwen2.5-14B 比肩 GPT-5-Chat"是建立在 GPT-4o 评分上的上限演示,容易被过度解读。而且 GAD 比 SeqKD 重——要同时维护同尺寸的 \(G\) 和 \(D\),3 epoch 约 2400 步,14B 蒸馏需 16×H100 约 30 小时;论文未强调这层成本权衡(依据:Appendix A.2)。
  - 低估【推断】:OOD 泛化优势(SeqKD 甚至为负、GAD 稳健)其实是比"接近教师"更稳的卖点,但被标题的"comparable to teacher"盖过了。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号** = 判别器给的序列级标量分 \(D(G(x))\)("像不像教师写的"),不是 token 概率。
  - **改什么** = 学生全参(policy)+ 判别器全参(reward model)。
  - **何时改** = 训练期 on-policy,每 batch 交替更新 \(G\) 与 \(D\)。
  - **免梯度?** = 否(\(G\) 走 GRPO 策略梯度、\(D\) 走 BT loss 梯度;采样不可微处用 RL 绕过)。
  - **记忆-技能生命周期** = 无显式记忆 / 技能库,知识固化进学生参数。
  - **防遗忘机制** = 无专门机制(warmup SFT 提供起点;靠教师文本对齐)。
- ⑦ 开源代码+框架/harness：
  - 项目页 ytianzhu.github.io/Generative-Adversarial-Distillation/。
  - 代码仓是 **microsoft/LMOps 的 `gad/` 子目录**(aka.ms/GAD-github 重定向至此,本地已 clone 488KB),但**只含环境 / 数据 / 启动脚本**(scripts/、tools/export_lmsys_parquet.py、local_setup.sh)。
  - **算法核心在外部 fork** `github.com/YTianZHU/verl`(基于 **veRL**)。该 README 明确"hack critic 模块当判别器",含 seqkd/warmup/gad/eval 分支,SeqKD/warmup 的 SFT 在 dp_actor.py。
  - 数据 / 模型在 HF(ytz20/LMSYS-Chat-GPT-5-Chat-Response、ytz20/gad-models);推荐 docker czwin32768/verl2:v0.2.0-vllm085(py3.10/torch2.6/vllm0.8.5)。
  - **未完整本地化**:核心算法不在本地 clone 的 LMOps/gad 内,需到 YTianZHU/verl fork 审计〔待核:fork 内 GAD 算法实现细节未逐行核对〕。
- 💰 资源/成本与可扩展性：
  - 蒸 Qwen2.5-14B(GPT-5-Chat 教师)约 **30 小时 / 16×H100**。
  - 3 epoch(1 warmup + 2 GAD)约 2400 步,batch 256、PPO mini-batch 256,\(N=8\),KL \(\beta=0.001\),温度 0.8。
  - lr:GPT-5 教师的 GAD 阶段用 1e-6;Qwen 教师的 warmup 用 5e-6、GAD 用 1e-6;SeqKD 用 5e-6。
  - max context:prompt 2048 / response 1536。需同时维护 \(G\) 和 \(D\)(同尺寸最优),比纯 SeqKD 重【§3.1, Appendix A.2】。
- 🎯 对"探索-巩固"对标：**竞品 / 可借组件**。
  - 一句判定:GAD 是"黑盒 on-policy 蒸馏"的强 baseline,提供两个可借组件。
  - 可借组件 (a):**在线协同进化的奖励 / 判别器**思路,可类比"teacher 当随学生演化的稀疏脚手架"——\(D\) 总是针对学生当前行为给反馈,正对应"巩固 / 回轨"里 teacher 需感知学生当前状态。
  - 可借组件 (b):**GAN↔RLHF 映射 → 抗 reward hacking**,对"固化进参数而不崩坏"有借鉴。
  - 缺口:GAD 没有路径 / step 级监督、没有 MTP 前瞻、没有记忆 / 技能库,且依赖教师文本(不是"学生自选可走通的开头")。所以它对"探索 / 选路"这一支撑较弱,更贴"巩固"的对齐侧。依据:\(D\) 共更对应在线 teacher 信号,但 GAD 学的是整段文本风格而非分支选择(§2.1, §3.3)。
- 🔭 开放问题/未来方向：【原文】未验证数学/代码等强推理域的黑盒蒸馏(§4 仅提"近期工作在教师推理轨迹上 SFT",GAD 本身未做);GAN 收敛性无理论保证,靠 warmup 经验缓解(§2.2)。【推断】把 \(D\) 做成 step/段级判别器(而非整段 1 个标量分)或可引入更细的路径级信用,接近"探索-巩固"的分支级监督;用真实任务正确率替代 GPT-4o judge 以去除评判偏差;探索教师=开源大模型但 tokenizer 不兼容时(Table 7)的更系统验证。

key|读PDF?|方法&相关工作已加厚?|LaTeX 公式条数|残留待核数
gad | 是(全文15页) | 是(方法 5-step 流水线 + 判别器结构/warmup 含先训D 10步 + GRPO advantage 与 BT/CE 真实形式 + 训推数据流;相关工作分白盒on-policy/SeqKD/固定RM 三线 + GAN/SeqGAN/GAIL 承袭) | 6(Eq.1 minimax 价值函数、Eq.2 生成器、Eq.3 BT、Eq.4 CE、Eq.6 advantage、Eq.8 组内 BT) | 1(YTianZHU/verl fork 内 GAD 算法实现细节未逐行核对)

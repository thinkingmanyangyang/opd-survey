opsd | Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models | UCLA / HKU / Meta Superintelligence Labs（Siyan Zhao 等;Feiyu Chen & Aditya Grover 共同指导） | 2026-03-20·arXiv 预印本·v3 | 主题线 L1(OPD/自蒸馏)·相关性 高

**原始论文**:https://arxiv.org/abs/2601.18734 （博客 siyan-zhao.github.io/blog/2026/opsd/）

## 一眼看懂
- 🟦 TL;DR:同一个 LLM 同时当老师和学生——老师能偷看"题目+标准答案 \(y^\star\)"(特权信息),学生只看题目;让学生先自己写一遍(on-policy 采样),然后沿学生写的这条轨迹,逐 token 让学生分布去贴近老师分布(KL 蒸馏)。不需要外部大老师模型,就能把数据集里现成的参考解变成稠密的逐 token 监督,在数学推理上追平/超过 GRPO 且省 token。【原文】abstract+§3.2
- 最巧的一步:**老师"只 prefill 不真正生成"**(§3.2,Fig.2 prompt 要求老师"看完参考解后用自己的方法解";原文明写 "the teacher won't be generating tokens—rationalization is done implicitly through one forward pass")。抽掉这一步——若老师真的把参考解抄出来当目标序列,就退化成普通 SFT/硬蒸馏,丢掉"在学生自己轨迹上做软分布匹配"的精髓;而正是"老师隐式 rationalize、给出 \(y^\star\)-条件下的 next-token 软分布"这一招,让监督既稠密又锚在学生的探索轨迹上。次关键是 **per-token pointwise KL clipping**(§3.2/§4.3.3):没它风格 token 会淹没数学 token、训练崩溃(Fig.4)。

## 为什么做
- 研究背景:推理 post-training 三条路——RLVR(GRPO 等)、在高质量 CoT 上 SFT、知识蒸馏。on-policy 蒸馏(GKD/MiniLLM/Thinking Machines 的 OPD 博客)让学生采样自己轨迹 \(\hat y\sim p_S(\cdot|x)\)、老师在其上逐 token 给稠密监督,兼具分布真实性与稠密反馈(原文把它连到 imitation learning / DAgger,§2.1)。【原文】§1+§2.1
- 解决的具体痛点:① RLVR/GRPO 每题采一组 \(G\) 个响应贵且高方差,一组全对/全错时组归一优势 \(A_i=0\)、梯度消失(Eq.4),奖励只在序列末端给、对所有 token 一刀切(原文称 "reward signal is sparse … same feedback across all tokens regardless of where errors occur");② SFT 有 exposure bias、且因 ground-truth 解风格简洁导致测试期生成变短、性能反降(§4.2 实测 SFT 全面降);③ 传统蒸馏 off-policy 分布失配;④ on-policy 蒸馏需一个独立、往往更大的老师,且没显式用上数据集里现成的 ground-truth 解;⑤ PRM 提供稠密 token 信号但标注代价高、难规模化(Lightman 2023)。【原文】§1+§2.2+§3.1
- 相关工作 & 各自不足(站谁肩上 + 精确差异):
  - **context distillation**(Snell 2022)——同一模型当师生、把 prompt/指令蒸进权重,但 **off-policy + SFT 硬蒸馏**(在固定数据上),OPSD 改成 on-policy 软分布匹配。
  - **STaR/ReST**(Zelikman 2022)——条件于 hint/答案生成 rationale → 拒绝采样保留正确 trace → SFT;原文 Appendix D 把它形式化为"序列级二值 reward 的 policy gradient,所有 token 同等 credit、错样本梯度=0",OPSD 反之"每个 token 位置都给反馈、不管最终答案对错"。
  - **on-policy 蒸馏 / GKD**(Agarwal 2024)——OPSD 的 full-vocabulary logit 蒸馏即沿用 GKD 的"全词表 \(D(p_T\|p_S)\)"思想,但 GKD 用独立(常更大)teacher,OPSD 去掉外部教师改用"同模型 + 特权上下文 \(y^\star\)"。
  - **Thinking Machines OPD 博客 / Tinker**(Lu & Lab 2025)——采样-token 的 reverse-KL policy-gradient 变体即沿用其设置(Eq.9),OPSD 实测全词表更优(Table 4)。
  - **in-context editing**(Qi 2025)——on-policy 软蒸馏但用途是知识编辑而非推理。
  - **并发同族**:SDPO(Hübotter 2026,特权信息=环境反馈)、SDFT(Shenfeld 2026,用于持续学习/防遗忘)、Tinker/Lu(on-policy 蒸馏配方);OPSD 的差异是"用数据集 ground-truth 解当特权信息做数学推理"。【原文】§5+Table 1
- 动机链:现代 LLM 已有强推理能力 → 那它能不能"自己当自己老师"?→ 借"评估比生成易"(Naor 1996, Sun 2024)推测"对给定正确答案做 rationalization"也比从零生成易 → 让模型看到 \(y^\star\) 后隐式 rationalize,以此监督只看题目的弱版自己。【原文】§1+§3.2
- 与最近邻工作的Δ:vs 普通 on-policy 蒸馏——**去掉外部教师**,改用"同模型+特权上下文 \(y^\star\)";vs context distillation/STaR——**on-policy 软分布匹配**(逐 token KL)而非 off-policy 硬蒸馏(生成 rationale 再 SFT)。关键点:把"利用数据集 ground-truth"和"在学生自身访问分布上稠密监督"两件事第一次合到一个目标里(Table 1 四项 On-Policy Data / Dense Signal / Low Sampling Cost / No External Teacher 全勾,而 SFT 缺 on-policy、GRPO 缺 dense+low-cost、OPD 缺 no-external-teacher)。【原文】Table 1+§5

## 怎么做 + 靠不靠谱
- 方法流水线(输入→输出,Algorithm 1 + Fig.2):
  1. **取题-解对** \((x,y^\star)\),\(y^\star\) 含 CoT;
  2. **学生只看题、on-policy 采样轨迹** \(\hat y=(\hat y_1,\dots,\hat y_{|\hat y|})\sim p_S(\cdot|x)\),其中 \(p_S(\cdot|x)\triangleq p_\theta(\cdot|x)\)(student prompt 只有 Problem+Answer:);
  3. **构造老师 prompt** \(p_T(\cdot|x,y^\star)\triangleq p_\theta(\cdot|x,y^\star)\):同一参数 \(\theta\),但上下文里塞入 "Here is a reference solution: …; After understanding the reference solution, please try to solve this problem using your own approach below"(Fig.2 原文 prompt)。老师**不真正生成 token**,只在学生前缀 \(\hat y_{<n}\) 上做一次前向、隐式 rationalize 得到 next-token 软分布;
  4. **沿学生轨迹逐 token 算散度**:两策略在同一前缀 \(\hat y_{<n}\) 上各得 \(p_S(y_n|x,\hat y_{<n})\) 与 \(p_T(y_n|x,y^\star,\hat y_{<n})\),求轨迹平均的 token-wise 散度;
  5. **梯度只回传学生** \(p_S\)(老师 \(p_T\) 当固定全分布 target,stop-gradient);
  6. **per-token pointwise clipping** 后更新 \(\theta\)(LoRA)。
  推理时学生只用 \(p_\theta(\cdot|x)\),不带 \(y^\star\)。【原文】Algorithm 1+Eq.(6)(8)+Fig.2

- 逐组件必要性(每条都标"有无消融"):
  - **教师固定为初始 policy**(非在线更新的学生):隐式正则/稳定训练、防偏离初始分布——【原文】§4.1 原话 "fix the teacher policy to be the initial policy … helps stabilize training and implicitly acts as regularization to prevent excessive deviation"。无独立消融,但与 GRPO 的 reference-KL 同理(也因此把能力上探锁在初始策略附近)。
  - **per-token pointwise KL clipping**:**有消融**(Fig.4,Qwen3-1.7B/AIME24:不裁剪后期崩溃、裁剪稳住)——必要。机制依据:风格 token('wait'/'think' 连接词)的逐 token KL 比数学 token 高数倍(Table 5),不裁会主导信号。【原文】§4.3.3+Table 5
  - **forward KL(而非 reverse/JSD)**:**有消融**(Table 3,AIME25/Qwen3-1.7B:FKL base 36.7→step50 **43.9**→step100 41.1;reverse KL 37.5/35.0、JSD(β=0.5) 36.9/39.0 几乎不涨甚至降)——FKL 是经验最优,全实验采用。【原文】§4.3.1+Table 3
  - **TM-off 学生 + TM-on 老师**(Thinking-Mode 组合):**有分析**(Table 5),该组合在数学 token 上 KL 最大、下游最好——保留。直觉:学生不思考直接答(暴露其裸推理),老师带 CoT 给出更尖锐的"看过答案"分布。【原文】§4.3.2
  - **full-vocabulary logit 蒸馏 vs 采样 token policy-gradient**:**有消融**(Table 4,Qwen3-4B/2048-gen,pass@8:AIME25 84.1 vs 82.1、HMMT25 60.0 vs 57.3),全词表更优但峰值显存高(需存词表大小 logits)——主实验用全词表。【原文】§4.3.5+Table 4
  - **生成长度 1024 vs 4096**:**有消融**(Fig.5),无一致增益——作者归因"早期 token 更关键、后期 token 在足够长前缀下对老师已可预测、惩罚自然变小"(与 Lu & Lab 2025 同观察)。【原文】§4.3.4

- 关键机制/公式(真实符号,从 PDF 抄准 + 直觉):
  - **师生条件分布**(同参 \(\theta\),仅上下文不同):
    \[ p_T(\cdot\mid x,y^\star)\triangleq p_\theta(\cdot\mid x,y^\star),\qquad p_S(\cdot\mid x)\triangleq p_\theta(\cdot\mid x). \]
  - **GRPO 组归一优势**(Eq.4,作为对照——它正是 OPSD 要替代的稀疏信号):
    \[ A_i=\frac{r_i-\operatorname{mean}(\{r_j\}_{j=1}^{G})}{\operatorname{std}(\{r_j\}_{j=1}^{G})},\qquad r_i\in\{0,1\}. \]
    原文用 value-function 视角解读:\(\operatorname{mean}(\{r_j\})\) 是 \(V(x)\) 的 \(G\)-样本 MC 估计,\(r_i\) 是 \((x,o_i)\) 的(无折扣)\(Q\) 值;一组内所有 token 共享同一 \(A_i\)。
  - **核心目标——轨迹平均的逐 token 散度**(Eq.6):
    \[ D\!\left(p_T\,\|\,p_S\right)(\hat y\mid x)\;\triangleq\;\frac{1}{|\hat y|}\sum_{n=1}^{|\hat y|} D\!\Big(p_T(\cdot\mid x,y^\star,\hat y_{<n})\,\big\|\,p_S(\cdot\mid x,\hat y_{<n})\Big), \]
    其中 \(\hat y_{<n}\triangleq(\hat y_1,\dots,\hat y_{n-1})\)。直觉:老师因偷看了答案,对"下一步该往哪个推理分支走"有更尖锐分布,学生在自己写出的每一步上被拉向"看过答案的自己"会怎么续写。
  - **散度可选 generalized JSD**(Eq.7):
    \[ \mathrm{JSD}_\beta(p_T\|p_S)=\beta D_{\mathrm{KL}}(p_T\|m)+(1-\beta)D_{\mathrm{KL}}(p_S\|m),\quad m=\beta p_T+(1-\beta)p_S, \]
    但消融后主实验取 forward KL(\(\beta\to1\) 极限的方向)。
  - **总损失**(Eq.8,期望在 on-policy 学生样本上,梯度只过 \(p_S\)):
    \[ L(\theta)=\mathbb{E}_{(x,y^\star)\sim S}\Big[\mathbb{E}_{\hat y\sim p_S(\cdot|x)}\big[D(p_T\|p_S)(\hat y\mid x)\big]\Big]. \]
  - **per-token pointwise clipping**(防风格 token 主导;\(D_f\) 为 \(f\)-散度):对每个位置 \(n\)、词表项 \(v\) 定义单点贡献 \(\ell^{(f)}_{n,v}=p_T(v|\cdot)\,f\!\big(\tfrac{p_S(v|\cdot)}{p_T(v|\cdot)}\big)\),再裁剪求和
    \[ D^{(f)}_{\mathrm{clip}}(p_T\|p_S)=\frac{1}{|\hat y|}\sum_{n=1}^{|\hat y|}\sum_{v\in V}\min\!\big(\ell^{(f)}_{n,v},\;\tau\big). \]
    直觉:把"少数高散度词表项"的贡献钳到上限 \(\tau\),让数学 token 不被风格 token 淹没。
  - **等价视角——稠密逐 token reward 的 policy gradient**(Eq.9):把 \(A_n(x,\hat y)=\log p_T(\hat y_n|x,y^\star,\hat y_{<n})-\log p_S(\hat y_n|x,\hat y_{<n})\) 当 stop-gradient 优势:
    \[ L(\theta)=-\,\mathbb{E}_{(x,y^\star)\sim S}\Big[\mathbb{E}_{\hat y\sim p_S(\cdot|x)}\Big(\tfrac{1}{|\hat y|}\sum_{n=1}^{|\hat y|}A_n(x,\hat y)\,\nabla_\theta\log p_S(\hat y_n|x,\hat y_{<n})\Big)\Big], \]
    即"看过答案的老师 log-prob 减学生 log-prob"作为每 token 的稠密奖励——这把 OPSD 与 RL 桥接,也解释了它为何能在 GRPO 失效(组内 reward 全同→\(A_i=0\))处仍有信号。【原文】Eq.(4)(6)(7)(8)(9)+clipping 段

- 实验与证据:训练用 OpenThoughts 数学子集(采 ≤30K 题-解对,含 CoT);评测 AIME24/AIME25/HMMT25,主表 Table 2 报 **Avg@12**(温度 1.0、thinking、max gen 38k);模型 Qwen3-1.7B/4B/8B instruct。关键数字(Table 2,基→OPSD,Avg):1.7B 37.1→**43.4**(>GRPO 37.7、SFT 35.8);4B 61.2→**63.6**(>GRPO 62.7、SFT 58.6);8B 61.8→**64.8**(>GRPO 64.0、SFT 59.8)。逐项:8B 上 AIME24 75.8→77.8、AIME25 65.6→70.8、HMMT25 43.9→45.8。token 效率(Fig.3):OPSD 每题 **1 rollout × 1024 token**、~100 步收敛(1.7B 在 4×H100 约 15 分钟,带 LoRA);GRPO 用 **8 rollouts × 16k token**,且 100 步内过半 batch 组内 reward 标准差=0、梯度消失("reward diversity collapse",Fig.3 最右)。SFT 全面降:作者归因 ground-truth 解风格简洁→测试期生成变短。baseline 公平性:SFT/GRPO 同数据集、同样本数——较公平;但 GRPO 报"500 步内峰值"、OPSD 报"100 步内每 20 步评估的最佳",评估口径不完全对齐(表注明确)。【原文】Table 2+Table 3-4+Fig.3+表注
- 假设与失效边界:
  - 【原文】§A:若问题超出模型理解阈值,即便给 \(y^\star\) 老师也无法提供有效监督 → 需课程学习;实验仅到 8B,>8B 是否持续未知;未利用答案正确性验证信号(只用 \(y^\star\) 作上下文,不校验学生最终答案对错)。
  - 【推断】教师"只 prefill 不生成"意味 OPSD 本质是"在学生轨迹上用 \(y^\star\)-条件分布做逐 token 匹配",能否给出"新能力"取决于 \(y^\star\)-条件是否真把分布推向更优——论文无理论保证。主表 Avg@12、消融 pass@8 口径混用易误读(依据:Table 2 vs Table 4)。
- 祛魅总结:【推断】真贡献=把"自蒸馏+特权上下文+on-policy 逐 token"三者干净地合一并给出可复现配方与关键稳定化技巧(clipping + forward KL + TM-off/on + 教师固定),且证明 token 效率显著优于 GRPO——这点扎实。被适度高估的是"超越 GRPO":4B/8B 上增益很小(+0.9/+0.8),主要价值在效率而非天花板;且后续工作(本 survey 中 Kim&Lee 视角)认为 OPSD 更像"压缩已知解"而非"教会更难题",其能力上探受"教师=初始 policy"限制。

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号:\(y^\star\)-条件老师分布与学生分布的**逐 token 散度**(forward KL,full-vocab,带 pointwise clip)——稠密、token 级。
  - 改什么:学生策略参数 \(\theta\)(LoRA),梯度只经学生 logits。
  - 何时改:学生**每一步生成 token**处都给信号(on-policy 轨迹全程),作者强调早期 token 更关键(Fig.5 归因)。
  - 免梯度?否——是基于梯度的分布匹配(可写成 dense-reward policy gradient 形式 Eq.9,但仍走梯度)。
  - 记忆-技能生命周期:无外部记忆/技能库;"技能"=参考解中的推理被内化进参数;教师锚在初始 policy(不在线滚动)。
  - 防遗忘机制:教师固定=初始 policy,隐式 KL 正则防偏离初始分布(等价 reference-KL)。【原文】§4.1
- ⑦ 开源代码+框架/harness:github.com/siyan-zhao/OPSD(已 clone,~330KB,核心完整)。框架 **TRL**(README 明示"基于 TRL experimental GOLD trainer";environment.yml:trl==0.26.0、transformers==4.57.1、vllm==0.11.0、peft==0.17.1、torch==2.8.0)。核心文件:`opsd_trainer.py`(72KB,实现 full-vocab divergence + pointwise clipping + 师生双 prompt 构造)、`data_collator.py`、`opsd_train.py`、`sft_train.py`/`grpo_train.py`(基线)、`eval/evaluate_math.py`(vLLM)。【原文/仓库】
- 💰 资源/成本与可扩展性:Qwen3-1.7B 在 4×H100 上 ~100 步、约 15 分钟收敛;A100/H100 + LoRA 即可;每题仅 1 rollout×1024 token(对比 GRPO 8×16k)。全词表蒸馏峰值显存高于采样-token 变体(需存词表大小 logits)。规模上限实测 8B。【原文】§4.2+§4.3.5+§A
- 🎯 对"探索-巩固"对标:**强支撑(原型级)+ 有缺口**。一句判定:OPSD 正是"teacher 当稀疏脚手架、student on-policy 自选轨迹、用 \(y^\star\) 当特权信息给稠密信号"这一思路的最干净原型,直接对应"巩固/回轨"——学生走偏后被拉向"看过答案的自己"。依据:§3.2 师生构造+Eq.6 on-policy 逐 token。可借组件:per-token KL clipping(防风格 token 主导,可直接迁到 MTP/OPD)、教师=初始 policy 的隐式正则、Eq.9 的"老师 logit 差当稠密 reward"视角(可与 MTP 前瞻打分结合)。缺口:① 它只"巩固"不显式"探索"(无路径选择/恢复分支的主动机制,teacher 给的是全程均匀软信号而非稀疏关键步接管);② 无 MTP/前瞻;③ "偏向自己能走通的开头"未建模(teacher 直接给 \(y^\star\)-条件分布,不挑学生的强开头)。
- 🔭 开放问题/未来方向:
  - 【原文】§A:引入答案正确性验证作额外目标;课程学习把题目维持在能力前沿;>8B 规模验证。
  - 【推断】把"全程逐 token 均匀监督"改成"只在高熵/关键分支步接管"可能更省更准(与本 survey 的 sparse_critical / forward-hard-backward-soft 同向);teacher 在线滚动(而非固定初始)以突破能力上探天花板(作者把"固定"列为稳定手段,但也因此限制超越教师);把 Eq.9 的稠密 token reward 与 MTP 前瞻熵相乘,做"关键分支步加权"的细粒度 path-selection。

— RETURN —
opsd | 读PDF? 是(15页,§1-§5+Table1-5/Fig2-5 全核,Eq.4/6/7/8/9+clipping 公式逐字核对) | 加厚? 是(相关工作扩到 6 条精确对位 + 方法流水线 6 步细化 + Eq.9 等价视角与直觉) | LaTeX公式条数:8(师生条件 / GRPO优势Eq.4 / 核心散度Eq.6 / JSD_β Eq.7 / 总损失Eq.8 / pointwise clip / dense-PG Eq.9) | 待核数:0

gopd | Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation (G-OPD / ExOPD) | 中国人民大学高瓴 + 腾讯 LLM 部（Wenkai Yang 腾讯实习；通讯 Yankai Lin）| arXiv 2602.12125v2, 2026-02-26 · 预印本 | 主题线 L1（OPD/自蒸馏）·相关性 高

**原始论文**：https://arxiv.org/abs/2602.12125

## 一眼看懂
- 🟦 TL;DR：先在理论上证明**标准 OPD 其实是一种 dense KL-约束 RL 的特例**——它把 reward 项与 KL 正则**永远 1:1 等权(β=1)**、且 reference 模型可任选（§3.2 Remark, 式7）。点破后加两个旋钮：**灵活 reference πref + reward 缩放因子 λ**。λ∈(0,1) 是「插值」（学生介于 ref 与 teacher 之间）；**λ>1 是「外推」(ExOPD)**，让学生**越过 teacher 的能力边界**；在「多领域专家合并回 base」设定下，ExOPD 是唯一能让统一学生稳定**超过所有领域 teacher** 的方法（图1a, 表2）。
- 最巧的一步：**把 OPD 改写出一个第三方 reference πref，从而暴露出 reward=log(π*/πref) 这个隐式奖励 + 显式 λ 权重**（式7→式11）。抽掉这步重写，OPD 就还是「reverse-KL 对齐 teacher」的黑盒，λ 和 reference 这两个旋钮无从插入——正是这个代数恒等变形（OPD ≡ 带特定 reward 的 dense RL）解锁了「外推超越 teacher」的可能。它不改任何训练循环、只改 token 级 advantage 的计算式（式14）。

## 为什么做
- 研究背景：OPD（学生采样自己的轨迹、在每个 token 上对齐 teacher logit 的 reverse KL）已被证明经验上比 off-policy 蒸馏(SFT)和 RL 更快更有效；两大应用——(a) 把不同领域 RL 变体能力(近)无损合并回 base 的多任务后训练范式(Xiao 2026/MiMo-v2)；(b) 大→小强弱蒸馏(MiniLLM/Qwen3)【原文 §1, §2】。
- 解决的具体痛点：OPD 经验有效但**机理理解有限、潜力没挖透**；标准 OPD 把 reward:KL 写死 1:1、reference 写死为学生初始策略，**缺少调节「学生相对 teacher 落点」的旋钮**——既不能精确「落中间」(budget-controlled)，也无法「超越 teacher」。
- 相关工作 & 各自不足：off-policy KD/SFT（学生只模仿、不从自身经验学，test 时泛化弱）；标准 OPD（Agarwal 2024 / MiniLLM Gu 2024，有效但 ceiling 被 teacher 卡住）；权重外推 ExPO（Zheng 2025，training-free 但不可控、不保证超过所有 teacher）；隐式奖励 implicit reward(DPO, Rafailov 2023)——G-OPD 的 reward=log(π*/πref) 正是它的同形物。
- 动机链：OPD 有效但黑盒 → 证明它=dense RL 特例(β=1, ref 任选) → 既然是 RL 特例就能放开 β(即 λ) 和 ref → λ>1 外推让学生越过 teacher、ref 换 teacher-pre-RL 可纠正 reward → 多 teacher 合并时唯一能全面超越各 teacher。
- 与最近邻工作的Δ：vs **标准 OPD**——多了 λ 与灵活 ref，λ=1+ref=学生初态即退回原 OPD；λ>1 是质变(从「逼近 teacher」到「超越 teacher」)。vs **ExPO（权重空间外推）**——G-OPD 在**输出分布/奖励空间**外推，可控且保证多 teacher 全面超越(表2)；ExPO 平均权重再外推，不可控。vs **隐式奖励 RL(PRIME/Eurus)**——同样用 log(πθ/πref) 当 dense reward，但 G-OPD 指出 π* 无需由 RL-from-πref 得到、甚至可不同尺寸，仍是有意义信号(§3.2(1))。

## 怎么做 + 靠不靠谱
- 方法流水线：① 选 reference πref（默认=学生初态；强弱蒸馏可选 teacher 的 pre-RL base 做 reward correction）→ ② 学生 on-policy 采样自身轨迹 y~πθ → ③ 每 token 算 G-OPD advantage：A_t^{G-OPD} = [logπθ−logπ*] + (λ−1)·[logπref−logπ*]（式14）→ ④ 按 policy gradient 更新（veRL 里 `only_reverse_kl_advantages=True`、`lambda_vals=λ`）。λ=1 时退回 OPD（第二项消失）。
- 逐组件必要性：
  - **λ（reward 缩放因子）**：核心旋钮。消融=扫 λ∈{0,0.25,…,1.5}（图2/3/4）。证据充分：λ<1 性能/长度单调介于 base 与 teacher 之间(可做 budget 控制)；λ=1.25 一致优于 OPD 且超 teacher；**λ=1.5 反而不稳退化**（作者解释为学生「hack 隐式奖励」、激进拟合 log-ratio 峰值，§4.1.2）——这是关键的失效边界证据。
  - **灵活 reference / reward correction**：图6 消融——强弱蒸馏中把 ref 从学生 base 换成 teacher-pre-RL base，math/code 各 +0.6/+1.0%。**作者诚实标注代价**：需额外持有 teacher-pre-RL 模型 + 算更大 ref 的 logprob 成本。
  - **多 teacher 合并 vs 单 teacher**：表2 证明 λ=1.25 固定不调，多 teacher 设定下 ExOPD 是唯一全面超越两个 teacher 的方法(SFT/ExPO/OPD 均有项掉分)。
- 关键机制/公式（直觉）：最优解 logπθ = λ·logπ* + (1−λ)·logπref（式12）。λ∈(0,1)→学生 log-prob 落在 teacher 与 ref 的线性插值上；λ>1→额外加一个 (λ−1)(logπ*−logπref) 的「外推位移」，把学生推到 teacher 之外的方向。强弱蒸馏的 reward correction：等价目标(式13)显示选 πref=teacher-pre-RL 时 reward=log(π*/π_base^teacher) 恰是 teacher RL 的「良定义隐式奖励」，比 log(π*/π_base^student)（跨模型有知识鸿沟、更 noisy）更准。
- 实验与证据：base/student=Qwen3-1.7B/4B-Non-Thinking；teacher=对 4B 做领域 GRPO 得的 RL-Math/RL-Code，及强弱设定下 Qwen3-30B-A3B-Instruct-2507；数据=DeepMath 过滤 57K(难度≥6)+Eurus-RL-Code 25K；评测 AIME24/25、HMMT25(Feb/Nov)、HumanEval+/MBPP+/LiveCodeBench。关键数字：单 teacher math ExOPD **48.0 vs teacher 46.0 vs OPD 46.5**(表2)；强弱蒸馏 4B-student **ExOPD 45.3 vs OPD 42.6 vs SFT 35.1**(表3)；且 ExOPD(50步) > teacher+继续 RL(100步)(表1)，排除「teacher 训得不够」的解释。
- baseline 公平吗：较公平——OPD/SFT/ExPO 同数据同轨迹数同步数对比(§4.1.3、附录B)，且专门做了「teacher 继续 RL」对照(表1)和「teacher 充分训练 1200 步」对照(附录C 表8，此时 ExOPD 增益缩到 +0.3~0.6%)。**这点很关键且诚实**：附录C 显示当 teacher 已被充分 RL 榨干，ExOPD 超越 teacher 的空间就很小了。
- 假设与失效边界：【原文】(1) λ 过大(1.5)→hack 隐式奖励→不稳退化；(2) reward correction 需 teacher-pre-RL 可得 + 额外算力；(3) 附录C：teacher 充分训练后 ExOPD 优势大幅缩水。【推断】(4) 「超越 teacher」本质是**沿 teacher 已学方向的外推**——只在 teacher 尚未收敛/欠训、且外推方向恰好仍指向正确分布时成立；它不创造新能力，是「把 teacher 没走完的那段路替它走完」。(5) 多 teacher 当前实现**只支持两 teacher**(README)，更多/更异质 teacher 未验证(作者列为 future work)。(6) ExOPD 副作用：响应变长(图4/5)，作者归因隐式奖励的长度偏置——长度膨胀可能换来虚高。
- 祛魅总结：真贡献=**(a) OPD≡dense-RL(β=1) 的干净理论桥 + (b) λ/ref 两旋钮把「插值/外推/纠偏」统一进一个框架 + (c) 多 teacher 合并能全面超 teacher 的实证**，理论与实验都较扎实、对照诚实。需打折的：「learning beyond teacher」标题略大——附录C 自证 teacher 充分训练后超越幅度→几乎为零，故「超越」是**欠训 teacher 上的外推红利**而非凭空涨能力；且增益普遍在 +1~3% 量级、伴随响应变长。【推断】对「探索-巩固」这是**最贴近的 L1 工作之一**：多 teacher→统一学生正是「巩固多路能力进一套参数」，λ 旋钮是「巩固强度」的连续控制器。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=teacher 在学生轨迹上的 token 级 logit（reverse-KL/隐式奖励 log(π*/πref)，dense 每 token）｜**改什么**=学生策略参数 θ｜**何时改**=on-policy 蒸馏训练每步（学生采自身轨迹）｜**免梯度?**=否，policy gradient｜**记忆-技能生命周期**=把多领域 teacher 的技能**合并固化进单套参数**（multi-teacher→unified student），无外部记忆/技能库｜**防遗忘机制**=间接——OPD 本身被定位为「(近)无损把各 RL 变体能力合回 base」的多任务范式(引 Xiao 2026)，KL-to-ref 约束抑制漂移；λ>1 外推则**增加**偏离 ref 的风险(过大即 hack)。
- ⑦ 开源代码+框架/harness：https://github.com/RUCBM/G-OPD （已 clone ~33MB，Tier A，README 已核）。框架=**veRL v0.6.1**(volcengine/verl，HybridFlow)+ vLLM/SGLang rollout；含 verl/、math_eval/、code_eval/(基于 Absolute-Zero-Reasoner/EvalPlus/LiveCodeBench)。训练数据已开源(HF: Keven16/G-OPD-Training-Data)。**额外亮点**：README 2026-03 更新了**自研 on-policy self/context-distillation (OPSD)** 代码(`run_qwen3-4b-context-distill.sh`、`ref_input_utils.py`)——teacher 分布=学生自己在「额外上下文(GT 解/反馈/高层经验)」下的 in-context 分布(固定 teacher 权重为学生初态、不做 EMA 以免崩溃)。
- 💰 资源/成本与可扩展性：8×A800/H 级 8 卡(n_gpus_per_node=8)；batch 1024、rollout n=1、max_response 16384、lr 1e-5；G-OPD 仅 50 步(同尺寸)/100 步(强弱)即可，**步数极少**(作者称再多步反而过拟合)。λ≠1 额外成本=多算一次 πref 的 logprob；reward correction 因 ref 更大成本更高。
- 🎯 对"探索-巩固"对标：**强对标(支撑+可借组件)**。判定依据：① **巩固侧直接命中**——multi-teacher→unified student 就是「把多条已走通的路径(领域专家)固化进一套参数且尽量不互相遗忘」，λ 是巩固强度旋钮；② **可借组件**——(a) λ>1 外推=「让学生沿 teacher 指引方向再多走一步、越过脚手架」，与 TSRD「teacher 当稀疏脚手架、最终要学生自己走得更远」精神一致；(b) README 的 **OPSD(on-policy 自/上下文蒸馏)** 几乎是 TSRD 的现成骨架——teacher=学生加了「GT 解/反馈/经验」后的自身分布，正对应「探索(给上下文提示找到路)→巩固(蒸回无上下文的参数)」。竞品风险：G-OPD 是 teacher-driven 的 dense 全程对齐，而 TSRD 主张**稀疏脚手架+学生自选恢复分支**——G-OPD 不区分「关键步/非关键步」、不做单点接管，这是差异化空间。缺口：无 MTP/前瞻、无「走偏后自选恢复」的显式机制、无路径选择信号。
- 🔭 开放问题/未来方向：【原文】(1) 在更大模型上验证 ExOPD 泛化性；(2) 更广更异质的多 teacher 集合下的鲁棒性；(3) 跨模型家族(不同 family)的 OPD。【推断】(4) λ 当前是全局常数——可否做**token/step 级自适应 λ**（关键推理步外推强、常规步保守），天然连接 hapo/holderpo 的「token 异质」与本课题「关键步接管」；(5) 把 OPSD 的「上下文=高层经验」推向 agent 自进化（探索得到的成功轨迹经验→上下文→蒸回参数）；(6) 缓解 λ>1 的 reward-hacking 与长度膨胀(需长度无关的隐式奖励)。

key|读到PDF?|L线|对标结论|残留待核数
gopd | 是(全文17页) | L1 | 强对标:multi-teacher→统一学生=巩固多路径进参数;λ外推=越过脚手架;README的OPSD自蒸馏近似TSRD骨架。差异:全程dense无稀疏脚手架/无单点回轨 | 0

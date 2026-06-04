brts | On-Policy Distillation with Best-of-N Teacher Rollout Selection (BRTS) | JHU + TikTok + UCSD + 复旦(Ke Zhang JHU/TikTok 实习;通讯 Di Fu, TikTok) | 2026-05·arXiv 预印本·v2(2026-05-13, 标 cs.CV) | 主题线 L1(在线策略蒸馏)·相关性 高

**原始论文**:https://arxiv.org/abs/2605.09725 （arXiv:2605.09725v2）

## 一眼看懂
- 🟦 TL;DR:标准在线策略蒸馏(OPD,即"学生用自己采样的轨迹、由 teacher 逐 token 打 log-prob 当监督")有两个隐患——监督算在"学生自己跑出来、可能跑歪"的噪声前缀上,而且每道题通常只用**一条随机抽的 teacher 轨迹**,方差极大,抽到错的/不匹配的就会被放大。BRTS 的做法:每道题先采 N 条 teacher 轨迹,按"先看答对没(correctness)、再看像不像学生当前会走的(alignment)"挑一条;若 N 条全错,就把 ground-truth 答案当"静默校验"偷偷塞进 prompt 让 teacher 重新自然推导出一条对的(path-recovery);把选中的这条轨迹作为一条额外的 **teacher-context 蒸馏分支**,和标准 student-context OPD 一起训(总损失 \(L_{\mathrm{total}} = L_{\text{stu-ctx}} + \lambda\, L_{\text{tea-ctx}}\),\(\lambda=10\))。
- 最巧的一步:**抽掉"correctness-first 选择 + Tier-2 ground-truth 恢复"这一层,方法就退化回普通 OPD**。因为 BRTS 真正新增的不是损失形式(teacher-context KL 本身很朴素),而是"在 OPD 内循环里把随机 teacher rollout 结构化成可靠监督"这个 curation 步骤——尤其是难题上全错时还能靠注答案"救"出一条对的轨迹(§3.2 Tier-2),保证最该给监督的地方分支不空转。为什么是它:论文全部增益(Table 1 候选池 0/2/4 递增、Table 2 Tier-2)都挂在"挑得准/救得回"上,损失形式不变(§3.3 明说 "BRTS does not replace OPD; it augments OPD")。

## 为什么做
- 研究背景:OPD 已成 LLM 后训练标准工具(§1 列 Qwen3[47]/MiMo-V2[44]/GLM-5[11] 等工业 pipeline 采用,以一小部分 RL 算力获可比增益,引 [29] Thinking Machines 的 OPD 博客)。配方很简单:学生从自己的策略 \(\pi_S\) 采 rollout,用 teacher \(\pi_T\) 在这些 rollout 上逐 token 的 log-prob 当稠密监督。相比在 teacher 文本上做 SFT 或序列级蒸馏(SeqKD[23]),OPD 因"监督定义在学生推理时真实访问的状态上"而更少 exposure bias([3] scheduled sampling、[33] DAgger、[45] imitation error bound 的经典论证)。
- 解决的具体痛点:① 监督算在"不完美学生生成、可能漂到噪声推理态"的前缀上(student context 噪声,§1 + Fig.1a):难题上学生前缀会漂入噪声推理态,此处 teacher 的局部反馈信息量低,且学生可能**从头到尾没见过一条完整正确解**;② 每 prompt 通常只依赖**单条随机 teacher rollout**,而 teacher 本身随机——同题不同采样在正确性/推理风格/与学生接近度上差异大(尤其难题),单样本是对 teacher 该题能力的**高方差估计**,抽到错的/不匹配的就放大噪声(§1)。
- 相关工作 & 各自不足(把这条线的来龙去脉摆清):
  - **(a) off-policy → on-policy 蒸馏的演化(§2 第一段)**。经典 KD([12] survey、[17] Hinton、[20] TinyBERT、[34] DistilBERT、[42] MiniLM)在 teacher 生成文本上做监督。对自回归 LM,在固定 teacher 输出上做 off-policy 蒸馏会引入 **distribution mismatch**:训练时条件在 teacher 诱导的上下文、推理时却要条件在自己的前缀([3,33,45])。在固定 teacher 输出上优化监督目标还会**把学生过度约束到它够不着的轨迹**([46] speculative KD、[54] capacity-gap law),并诱发灾难性遗忘([30] continual FT 实证、[36] RL's Razor)。OPD 用"学生生成轨迹 + teacher 监督学生真实访问态"缓解之:MiniLLM[13]、GKD[1] 用 reverse-KL 及相关散度形式化;Qwen3[47]/MiMo[44]/GLM-5[11] 把蒸馏与 SFT、outcome-reward RL([15] R1、[35] DeepSeekMath、[52] DAPO)组合。近期变体把 OPD 推到黑盒([50] = 本survey 的 gad)、context([51])、entropy-aware([21])、reward-extrapolated([49])等设定。
  - **(b) "理解 OPD 何时成功/失败"这条线(§2 第二段)**。[26]（Rethinking OPD,本 survey 的 rethink_opd）指出成功 OPD 需三条件:师生**思维模式兼容**、teacher 有**真正新的知识**、二者**高概率 token 集重叠随训练增大**;[10]（经验失败模式）、[19,21,22] 进一步把失败归因到噪声学生前缀、熵失配、不稳定 target、师生推理态对齐弱。这些发现指出 OPD 不只取决于散度形式,**还取决于施加监督的 context**。还牵出 token 级 loss 方向:经典 KD 强调 teacher-confident token([17]),而很多 OPD 实现监督的是 student 策略采样/高权重的 token([29,44])。小模型难模仿风格不匹配的强 reasoner([9] specializing small models、[14] OpenThoughts、[27] small models struggle)。
  - **(c) rollout 选择 / privileged hint / 数据过滤(§2 第三段)**。best-of-N、拒绝采样、自训练([6] GSM8K verifier、[8] RAFT、[28] step verifier、[39] ReST、[41] summarization-RLHF、[43] self-consistency、[53] STaR)做的是 **outer-loop 离线过滤**保留高质量样本供推理/训练;privileged 信息族([7] HDPO、[18] RL via self-distillation、[32] privileged info distillation、[38] self-distillation continual、[40] learning by distilling context、[48] self-distilled RLVR、[55] self-distilled reasoner)用 ground-truth/示范/反馈增强监督。
- 本文站在谁肩上 & 与最近邻工作的精确差异:BRTS 直接承接 [26][29] 的标准 OPD recipe(同样的 reverse-KL + top-K 近似),把上面 (c) 的"选 + 救"放进 **OPD 内循环**——选中轨迹**不是离线训练数据,而是为当前这步 OPD 更新定制的 teacher-context 信号**(§2 第三段原文:"used not as offline training data, but as a teacher-context signal tailored to the current OPD update")。与离线 best-of-N/拒绝采样的精确差:① 选择在训练步内、针对**当前学生分布**(alignment 用 student top-K 重叠);② 新增的是一条**反向条件**的 teacher-context 分支(在 teacher 自己可靠前缀上、用 teacher top-K 监督),而非把选中样本当 SFT 目标;③ 全错时还能用 ground-truth **静默校验**逼 teacher 自然推导,救回"最需要监督却没正确样本"的难题。为什么有用:teacher-context 分支让学生见到"完整、连贯、且选得贴近自己当前分布"的推理路径,补上 student-context 分支看不到的"完整正确解"(§3.3)。

## 怎么做 + 靠不靠谱
- **符号与 OPD 基线(§3.1)**:prompt \(x\)、ground-truth \(y^\star\);student/teacher 策略 \(\pi_S,\pi_T\),各自定义词表 \(V\) 上的自回归分布。轨迹 \(y=(y_1,\dots,y_T)\) 的概率分解 \(\pi(y\mid x)=\prod_{t=1}^{T}\pi(y_t\mid x,y_{<t})\)(Eq.1)。记学生 rollout \(\hat y_S\sim\pi_S(\cdot\mid x)\)、teacher rollout \(y_T\sim\pi_T(\cdot\mid x)\)。标准 OPD 最小化序列级 reverse-KL(Eq.2):
  \[
  L_{\mathrm{OPD}}(S)=\mathbb{E}_{x,\,\hat y_S\sim\pi_S}\!\left[\sum_{t=1}^{T} D_{\mathrm{KL}}\!\big(\pi_S(\cdot\mid x,\hat y_S^{<t})\,\big\|\,\pi_T(\cdot\mid x,\hat y_S^{<t})\big)\right].
  \]
  实现里每步 KL 用**采样的 / top-K token 集**近似,该集合**默认取自 student 当前前缀下 student 分布的 top-K**([29,44])——即"在学生真实访问的态上纠正学生"。BRTS 在此之上加一条 correctness-/alignment-aware 的 **teacher-context 分支**。论文分三块讲(§3.2 轨迹 curation、§3.3 teacher-context 监督、§3.4 各分支 top-K 方向),Algorithm 1 给出单训练步。

- **方法流水线(逐步 输入→输出,Algorithm 1 + §3.2~3.4,具体到可复现)**:
  - **输入**:prompt \(x\)、ground-truth \(y^\star\)、teacher \(\pi_T\)、student \(\pi_S\)、Tier-1 采样数 \(N\)、辅助权重 \(\lambda\)。
  - **Step 1 学生采样**:采 1 条 on-policy rollout \(\hat y_S\sim\pi_S(\cdot\mid x)\)(实现:temperature 1.0、repetition penalty 1.0、max prompt 1024 / max response 7168、vLLM;每 prompt **仅 1 条** student rollout,附录 A)。→ 输出学生轨迹 \(\hat y_S\)。
  - **Step 2 teacher 候选采样**:采 \(N\) 条 **unconditioned** teacher 轨迹 \(\{y_{T,i}\}_{i=1}^{N}\sim\pi_T(\cdot\mid x)\)(实现:temperature 0.7、top-p 0.95——附录 A 称这是"给 teacher 最高 pass 率"的配置;同 max response 7168)。→ 输出候选池。
  - **Step 3 打分(correctness)**:对每条候选用 `\boxed{...}` 抽末答案(附录 B:取**最后一个** `\boxed{}`,无可解析 boxed 视为错),判 \(\mathrm{answer}(y_{T,i})=y^\star\) 与否。→ 输出每条的对/错标签。
  - **Step 4 Tier-1 选择(correctness-first → alignment-second,§3.2)**:若 \(\ge 1\) 条对 → 在**对的子集**里选与 student top-K 候选集 **token 重叠最高**者作为 \(y'\)。直觉:在正确的前提下,选"落在学生高概率区、学生够得着"的轨迹,更适合蒸馏。→ 输出选中轨迹 \(y'\)(若此步成功则跳到 Step 7)。
  - **Step 5 Tier-2 ground-truth 恢复(全错时,§3.2 + 附录 B)**:构造改写 prompt \(x_{gt}\)——把 \(y^\star\) 作为"SILENT_VALIDATION_KEY"注入(注入位置在 assistant turn marker 如 `<|Assistant|>` **之前**),四条 STRICT RULES 严禁在 `<think>`/答案里提及该 key、严禁说"答案给了/根据提示",要求整条思维链只能源自题面观察,**只准在推完得出结论后把 key 当 final sanity-check**(不准当起点/捷径)。采 1 条 guided rollout \(y_{T,gt}\sim\pi_T(\cdot\mid x_{gt})\),**仅当其抽出答案 \(=y^\star\) 才保留** \(y'\gets y_{T,gt}\)。→ 输出一条"自然推导式"正确轨迹(或失败转 Step 6)。
  - **Step 6 Tier-3 fallback(仍无对的,§3.2)**:回退到 Tier-1 候选里 **student-overlap 最高**的那条(可能错,但避免引入"离学生很远的任意轨迹")。→ 输出 best-available 轨迹 \(y'\)。
  - **Step 7 两支损失 + 一步梯度(§3.3)**:在 \(\hat y_S\) 上算 student-context 损失(用 **student top-K**)、在选中 \(y'\) 上算 teacher-context 损失(用 **teacher top-K**),合成 \(L_{\mathrm{total}}=L_{\text{stu-ctx}}+\lambda L_{\text{tea-ctx}}\),对 \(\pi_S\) 参数走一步梯度。→ 输出更新后的学生。
  - 关键默认超参(附录 A,复现必备):AdamW(betas (0.9,0.999)、weight decay 0.01、grad clip 1.0)、**常数 lr 1e-6 无 warmup**、token-mean loss 聚合、mini-batch 64、PPO micro-batch 1/GPU 动态批、bf16;**关闭对 frozen reference 的 KL**(故目标里只剩 student-context 蒸馏 KL 与 teacher-context 辅助 KL 两项);**teacher top-K=16**;\(\lambda=10\) 全程固定;8×B200 单节点;验证每 10 步一次、k=4、temp 0.7、top-p 0.95、max validation response 31744。

- **两支损失的真实形式与直觉(§3.3,Eq.3/4/5)**:两支都在"匹配的条件 context"下比 teacher 与 student 分布。student-context 分支两个分布都条件在学生前缀 \(\hat y_S^{<t}\);teacher-context 分支两个分布都条件在选中 teacher 前缀 \(y'^{<t}\)。
  \[
  L_{\text{stu-ctx}}=\mathbb{E}\!\left[\sum_t D_{\mathrm{KL}}\!\big(\pi_S(\cdot\mid x,\hat y_S^{<t})\,\big\|\,\pi_T(\cdot\mid x,\hat y_S^{<t})\big)\right],
  \]
  \[
  L_{\text{tea-ctx}}=\mathbb{E}\!\left[\sum_t D_{\mathrm{KL}}\!\big(\pi_T(\cdot\mid x,y'^{<t})\,\big\|\,\pi_S(\cdot\mid x,y'^{<t})\big)\right],
  \qquad
  L_{\mathrm{total}}=L_{\text{stu-ctx}}+\lambda\,L_{\text{tea-ctx}}.
  \]
  - **两条 KL 方向相反**(本文一个被低调处理、但很重要的设计):student-context(Eq.3)是 **reverse 形式** \(D_{\mathrm{KL}}(\pi_S\|\pi_T)\)(\(\pi_S\) 在前),"在学生访问态上把学生拉向 teacher";teacher-context(Eq.4)是 **forward 形式** \(D_{\mathrm{KL}}(\pi_T\|\pi_S)\)(\(\pi_T\) 在前),"在 teacher 可靠前缀上把学生分布拉去覆盖 teacher"。直觉:reverse-KL 是 mode-seeking(让学生在自己态上别犯错),forward-KL 是 mode-covering(让学生别漏掉 teacher 在好路径上偏好的 token)。**论文未论证为何两支取相反方向**——逻辑链上一个未解释的设计选择〔推断〕。
  - **\(\lambda\) 的标度直觉(§3.3 原文)**:teacher-context 分支评在"通常比噪声学生 rollout 更连贯"的选中轨迹上,故其原始贡献在 \(\lambda=1\) 时太小;经验取 \(\lambda=10\) 给它"有意义的尺度且仍稳定",全实验用此值。**未给 \(\lambda\) 敏感性扫描**〔推断:仅单值,无 5/10/20 对照〕。

- **top-K 方向:student- vs teacher-confident(§3.4,"注入新能力"的载体)**:两支都实现为 top-K 聚合损失,差在候选 token 集怎么定义。student-context 分支候选集 = student 前缀下 **student 的 top-K**(监督学生自认为合理的 token,做定向纠错);teacher-context 分支候选集 = 选中 teacher 前缀下 **teacher 的 top-K**(反映 teacher 在高质量轨迹上的局部分布,**会引入学生当前 top 之外、teacher 偏好的 token**——这正是"注入超出学生探索范围新能力"的机制),同时仍把监督集中在紧凑 top-K 上。**未单独消融**"teacher-context 也改用 student top-K"〔推断〕。

- **逐组件必要性(消融证据 + 缺口)**:
  - **Tier-1 候选池 \(N\)**:降"抽到错轨迹"的方差。Table 1 候选 0/2/4 递增,AIME24 mean 0.3917 →(1cand)0.3750 →(2cand)0.3750 →(4cand)**0.4000**、majority 0.4146→**0.4306**;Fig.6(e) Tier-1 命中率 2 候选 52.73% → 4 候选 66.70%。没它就退化回单样本高方差。
  - **alignment(student top-K 重叠)选择**:选"学生够得着"的轨迹。**未单独消融** alignment vs 在正确子集里随机选——一处证据缺口〔推断〕。
  - **Tier-2 ground-truth 恢复**:难题上"全错→救出对的"。Table 2 加 Tier-2 后 AIME25 mean 0.2667→**0.3000**、best→0.4309、majority→0.3133;Fig.4a 在更难的 AIME25 上早期持续抬升;Fig.6(e) Tier-2 在 1 个 Tier-1 候选时多救 25.00%(总命中 43.36%→68.36%)、2 个候选时多救 16.80%(52.78%→69.53%)。
  - **prompt 扰动(附录 B)**:给两条 teacher rollout 之一追加 "Please reason step by step and rethink in detail before giving the final answer" 做去相关。Table 3 AMC23 majority step20 0.6663→**0.6764**;Fig.5 解释:未加多样性时 Tier-1 命中率远低于 i.i.d. 理论曲线 \(1-(1-p)^n\),说明 teacher 样本强相关,轻微扰动能部分去相关;因选择经 correctness 过滤,扰动只增多样性不伤可靠性。
  - **\(\lambda=10\)、teacher-context top-K 方向**:均未给敏感性/方向对照〔推断〕。

- 实验与证据:
  - 数据集/设置:训练 prompt = **DAPO-Math-17K**[52]（可验证短答案）;评测 **AIME 2024 / AIME 2025 / AMC 2023**,k=4 解、temp 0.7、top-p 0.95,报 mean/best/majority;主实验 teacher=**JustRL-DeepSeek-1.5B**[16]、student=**DeepSeek-R1-Distill-Qwen-1.5B**[15]（同尺度 1.5B,**共享 tokenizer** 以简化 top-K 对齐）;teacher-swap 换 **DeepSeek-R1-Distill-Qwen-7B**[15];8×B200。
  - 支撑核心主张的关键数字:Table 1 候选池递增 AIME24 mean 0.3917→0.4000(+约 0.8pt)、majority 0.4146→0.4306;Table 2 Tier-2 使 AIME25 mean→0.3000;Table 4 teacher-swap 到 7B 后早期 AIME24 mean 0.3167→**0.3667**、majority 0.3331→**0.3952**(换 teacher 仍有效)。
  - baseline 公平吗:基线是"2 条 student rollout",BRTS 是"1 student + 1 teacher(从 N 候选选)"——**总用于 loss 的轨迹数相同(都是 2 条)**,额外成本只在选择阶段采样(每多 1 候选 +约 59s/step,Table 6:1/2/3/4 候选 = 281/338/400/460 s/step)。这个对照控制了 loss 轨迹数,设计公平。
  - "看着强但没回答核心问题":AMC23 几乎无提升(Table 1 各设置 mean 0.67–0.68 ≈ 基线 0.6777),论文归因"student-only 基线已强";增益**绝对值小**(AIME24 mean 仅 +0.8pt),"significantly improves"措辞与量级有张力〔推断〕。主结果是同尺度 1.5B 师生——"teacher 产出超学生范围解"这一 OPD 成功前提在同尺度下能否成立,靠"teacher JustRL-1.5B 经强 RL 仍强于 distill student",但论文未直接量化师生能力差〔推断〕。
- 假设与失效边界:
  - 显式假设【原文】:teacher 能在某些题上产出正确自然解;师生**共享 tokenizer**(附录 A 明说否则 top-K 对齐失效);有可验证 ground-truth \(y^\star\)(Tier-2 依赖它)。
  - 隐式假设【推断】:师生推理模式兼容(否则 alignment 选出来的也不可用);top-K token 重叠是有效的"学生可达性"代理(论文只当 proxy,未验证它优于别的对齐度量)。
  - 何时失效【推断】:① 无可验证 ground-truth 的开放生成(Tier-2 失效,附录 D 自承需换 retrieval/verifier/更强模型);② 师生 tokenizer 不一致;③ teacher 本身在该领域很弱(N 条全错且 Tier-2 也救不回,只能 fallback 到错轨迹)。
- 祛魅总结【推断】:
  - 真贡献:把"correctness-first + student-alignment 选择"和"ground-truth 静默校验恢复"这两个**轻量 curation 步骤**塞进 OPD 内循环,且控制 loss 轨迹数做公平对照——干净、可复用的工程组件,尤其 Tier-2 的"自然推导式恢复"对"难题上分支不空转"有清晰价值。
  - 包装/高估:"significantly improves"在 AIME 上绝对增益其实很小(<1pt mean),AMC 基本持平;评测仅竞赛数学三个集,泛化覆盖窄;同尺度师生使"OPD 注入新能力"的叙事打折。
  - 低估之处:作者其实给了一个挺通用的框架视角(附录 D:Tier-2 可推广到 retrieval/tool-feedback/learned-verifier/更大 teacher 引导小 student),但正文实验没兑现,把一个"可推广的可靠监督构造范式"局限在了数学。

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号:teacher 在两种 context 下的 **token 级 top-K 分布 / log-prob**(student 前缀上的 reverse 向 KL + 选中 teacher 前缀上的 forward 向 KL)。
  - 改什么:**参数**(梯度更新 student;改的是 logits 分布对齐)。
  - 何时改:**在线 per-step**(每训练步现采 teacher 候选、现选、现算两支损失)。
  - 免梯度?:否(标准梯度下降训 student)。
  - 记忆-技能生命周期:**无显式记忆/技能库**;teacher 轨迹是 per-step 临时构造、用完即弃。
  - 防遗忘机制:间接——OPD 范式本身(监督在学生访问态、关掉对 ref 的 KL 但靠 teacher 分布锚定)被认为比 SFT 少遗忘;论文还提"早期达峰→可缩短训练→少遗忘"(§4.2 Computational Cost),但**未做遗忘基准评测**,防遗忘是继承自 OPD 的论断而非本文实测〔推断〕。
- ⑦ 开源代码 + 框架/harness:https://github.com/BWGZK-keke/BRTS (已 clone 约 14MB,Tier A 可跑);框架 **veRL**(`python -m verl.trainer.main_ppo`,`ADV_ESTIMATOR=token_reward_direct`,rollout 用 vLLM,teacher 当 reward_model 提供 token 级 reward,swanlab 记录);与 thunlp/OPD(rethink_opd)同源 fork。脚本:on_policy_distillation.sh / grpo.sh / forward.sh(Tier-1 only)/ forward_tier2.sh(含 Tier-2)。
  - **〔待核·文-码不一致〕**:论文 §3.3 与附录 A 均明确 \(\lambda=10\);但 v1 分析记录仓库脚本默认 `AUX_TEACHER_CTX_KD_COEF=0.05`。复现需对齐(本轮未重新打开脚本核对,沿用 v1 记录,标待核)。
- 💰 资源/成本与可扩展性:8×B200 单节点;每多 1 个 teacher 候选 +约 59s/step(Table 6:1/2/3/4 候选 = 281/338/400/460 s/step),近似线性增长;**不增加** loss 用的轨迹数(选中后只用 1 条);teacher 解通常更短更直接故 forward 开销可控;max response 7168(训练)/31744(验证)。可扩展性:论文称可加大候选池、用多条选中轨迹做损失(附录 D),但要权衡采样成本 vs 质量。
- 🎯 对"探索-巩固"对标:**强支撑 + 可借组件**。判定:BRTS 的"Tier-2 ground-truth 静默校验恢复"几乎就是我们"巩固/回轨"里"走偏后从恢复分支自选一条能接着做对的"的一种 teacher-scaffolded 实现——只不过它是 teacher 端恢复、再蒸给 student;"alignment-second 选最像学生的轨迹"对应"偏向自己能走通的开头/方向"。依据:§3.2 三层 curation + 附录 B 静默校验模板;可直接借的积木是"correctness-first→student-alignment-second 的轨迹优先级规则"和"把 ground-truth 当静默 sanity-check 逼出自然推导(而非看答案倒推)"这两个机制。缺口:它是 teacher 外部恢复 + 同尺度,而我们要 student **全程 on-policy 自选**恢复分支;BRTS 不涉及 MTP/前瞻。
- 🔭 开放问题/未来方向:
  - 【原文】(附录 D)推广到无 ground-truth 的开放生成(Tier-2 改用 retrieval-hint / tool-feedback / learned verifier / 更强模型如 32B 引导 1.5B);更有表现力的师生兼容性度量(替代 top-K 重叠);把 OPD 做成**课程式**——自动识别题目对当前学生是 easy/learnable/too-hard 并自适应调 teacher 监督("像人类老师先识别学生能力再传授");自适应选 \(\lambda\)、候选数、选中轨迹条数。
  - 【推断】两支 KL 方向相反的理论依据值得补;同尺度师生下"注入新能力"vs"仅降方差"需解耦;应补遗忘基准、补"alignment 选择 vs 随机选"和"teacher-context top-K 方向"的消融以坐实各组件归因。

key|读PDF?|方法&相关工作已加厚?|LaTeX 公式条数|残留待核数
brts | 是(全文16页) | 是(方法 7-step 流水线 + 两支 KL 真实形式与方向直觉;相关工作分 a/b/c 三线+精确差异) | 5(Eq.1 分解、Eq.2 OPD reverse-KL、Eq.3 stu-ctx、Eq.4 tea-ctx、Eq.5 total;另含 \(1-(1-p)^n\)、\(\lambda\)) | 1(λ 文-码不一致 AUX_TEACHER_CTX_KD_COEF=0.05 vs λ=10 未重核脚本)

negative_reinforce | The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning | University of Virginia + Princeton PLI(Xinyu Zhu/Mengzhou Xia/Danqi Chen/Yu Meng 等) | 2025-06 首发,v2 2025-10-25;NeurIPS 2025 | 主题线 L3 RLVR/GRPO(机理+算法变体)·相关性 高

**原始论文**:https://arxiv.org/abs/2506.01347

## 一眼看懂
> 一句话导读:把 RLVR 的奖励拆成"只奖对的"和"只罚错的"两半,结果发现单用"只罚错的"在大采样数下反而更强——因为它压错答案时是按模型自己的先验比例把概率匀给其它候选,既不抹掉已有知识、改对了就停手。

- 🟦 TL;DR:把 RLVR(可验证奖励 RL,即用对/错二元信号训练)的学习信号(对 +1 / 错 -1)解析地拆成两条独立范式:
  - **PSR(Positive Sample Reinforcement,只奖励对的)**;
  - **NSR(Negative Sample Reinforcement,只惩罚错的)**。
  - 惊人发现:**只惩罚错的 NSR**,在整个 Pass@k 谱(k 到 256;Pass@k = 采 k 次至少对一次的概率)上一致超过基座,常追平甚至超过 PPO/GRPO;而只奖对的 PSR 提升 Pass@1 但因多样性塌缩,在大 k 处掉到基座以下。
  - token 级梯度分析给出解释:NSR 压低错 token、把概率**按模型自身先验比例**重分配给其它候选,所以"精修已有知识、不抹掉先验、改对就停手"。
  - 据此提出把正奖励降权 λ=0.1 的 **W-REINFORCE**。【原文 Abstract, §3.2, §4】
- 最巧的一步:**NSR 的 token 级梯度把未采样 token 的 logit 抬升量正比于其当前概率 \(\pi_v\)**(式8)。抽掉这个"按先验比例"性质(比如换成均匀 entropy bonus,即均匀加熵奖励),就不能保住模型先验、不能定向探索到"自己原本就觉得对"的候选,NSR 的多样性保持与 Pass@k 优势就垮了(附录 B 用梯度分析证明 entropy bonus 做不到这点)。

## 为什么做
> 一句话导读:RLVR 有效但大家不清楚"对样本和错样本各自起什么作用",而且多数研究只看 Pass@1、忽视了 RL 对大采样数表现(推理边界)的影响,本文要把这两半拆开看清。

- 研究背景:RLVR 用二元奖励训推理模型(DeepSeek-R1/Kimi-K1.5),既防 reward hacking(刷奖励)又省人工标注。但它**到底怎么用对/错样本、机理为何有效,仍未被理解**。【原文 §1, §6】
- 解决的具体痛点:多数工作只看 Pass@1/greedy 准确率,忽视了 RL 对模型行为的深层改变(尤其推理时扩展性 Pass@k)。"对样本和错样本各自起什么作用"被纠缠在一起、无法解释。【原文 §1, §2.2】
- 相关工作 & 各自不足(站谁肩上 + 精确差异):
  - **质疑 RLVR 的批评性工作**(Yue 2025):发现 RL 模型在大 k 处反不如基座、RLVR 只是把分布挪向高奖励而非赋予新能力。本文与之**一致并给出成因**(PSR 的塌缩 = 大 k 退化的根因),但 Δ 是给出可操作的 PSR/NSR 解耦 + 梯度级机理 + 一个能同时保 Pass@1/Pass@k 的简单算法。
  - **unlikelihood training**(Welleck 2020 等):同样抑制不良生成,但本文对比指出其**无 NSR 的"按先验比例重分配 + 改对即停"良性性质**(附录 B/C 对照)。
  - **SFT 多样性塌缩**(Chu 2025):平行现象(SFT 侧),本文聚焦 RL 侧、且 PSR 与 SFT 同构(都是抬高观测正样本似然)。
  - **GRPO/PPO**(Shao 2024 / Schulman 2017):用组内归一/clip/KL 稳定训练。§4.3 证明这三项只改梯度幅度不改"PSR 塌缩、NSR 按先验重分配"的方向性结论,故对它们同样适用(把分析从 REINFORCE 推广到 PPO/GRPO)。
  - **去 KL/熵正则的近期实践**(DAPO 等[6,62,69]):支持本文"KL 系数对推理任务可忽略甚至去掉更好"的设定。
- 动机链(逐步推):
  - RLVR 有效但机理黑箱;
  - 把目标解析拆成 PSR+NSR 两个子目标(式2-4);
  - 各自单独训练 + 全 Pass@k 谱评测;
  - 发现 NSR 反直觉地强;
  - token 级梯度分析找成因;
  - 据此设计 W-REINFORCE(降权正奖励)。
- 与最近邻工作的Δ:相比"RL 没赋予新能力"的批评性工作,本文 Δ 是**给出可操作的正/负信号解耦 + 梯度级机理 + 一个能同时保住 Pass@1 和 Pass@k 的简单算法(W-REINFORCE)**,把"负向信号被严重低估"这一点量化坐实。

## 怎么做(到可复现)
> 一句话导读:把期望奖励目标按对/错分组,得到 PSR(像 SFT 抬正样本似然)和 NSR(像 likelihood-minimization 压错样本似然);再用 token 级梯度证明 NSR 的三个良性性质;最后只把正奖励乘 λ=0.1 就得到强基线 W-REINFORCE。

### 形式化拆解(真实形式)
- **RLVR 期望奖励目标(式1)**:\(L_{\text{RLVR}}(\theta)=-\mathbb{E}_{x\sim D,\,y\sim\pi_\theta(\cdot|x)}[r(x,y)]\),\(r\in\{-1,+1\}\),同一响应所有 token 共享同一奖励;PPO/GRPO 实践里再做 batch 内零均值归一。
- **拆成 PSR+NSR(式2-4)**:把期望写成 reward-weighted 似然后按对错分组:
\(\displaystyle L_{\text{RLVR}}(\theta)=\underbrace{-\mathbb{E}_{x\sim D}\!\!\sum_{y:r=1}\!\pi_\theta(y|x)}_{L_{\text{PSR}}(\theta)}\;\underbrace{-\mathbb{E}_{x\sim D}\!\!\sum_{y:r=-1}\!\!\big(-\pi_\theta(y|x)\big)}_{L_{\text{NSR}}(\theta)},\qquad L_{\text{RLVR}}=L_{\text{PSR}}+L_{\text{NSR}}.\)
PSR 类似 SFT(抬高正确响应似然),NSR 类似 likelihood-minimization(压低错误响应似然);二者均 **on-policy**(响应从当前模型自采样)。
- **训练协议**:对每个 prompt 选择性地**只用对的(PSR)/只用错的(NSR)** 响应更新策略;因此 PSR/NSR 单 batch 有效样本比 PPO/GRPO 少(作者已说明)。

### token 级梯度分析(机理核心 + 直觉)
统一损失形式(式6):\(L(\theta)=-R\cdot\frac1T\sum_t \pi_\theta(y_t|x,y_{<t})=-R\cdot\frac1T\sum_t\frac{\exp(z_{y_t})}{\sum_{v'}\exp(z_{v'})}\),\(R\in\{-1,+1\}\),\(z_v\) 为 token v 的 logit。对 logit 求梯度(\(\pi_v=\pi_\theta(v|x,y_{<t})\)):
- **PSR(式7)**:
\(\displaystyle -\frac{\partial L_{\text{PSR}}}{\partial z_v}\propto\begin{cases}\pi_v(1-\pi_v) & v=y_t\ (\text{采样 token})\\[2pt] -\pi_{y_t}\pi_v & v\neq y_t\ (\text{未采样})\end{cases}\)
即**抬采样到的对 token、压所有其它**(含别的正确候选),分布越来越尖、熵塌缩、过拟合到"早期采到的对解"(Fig.5b/Fig.6 左)。
- **NSR(式8)**:
\(\displaystyle -\frac{\partial L_{\text{NSR}}}{\partial z_v}\propto\begin{cases}-\pi_v(1-\pi_v) & v=y_t\ (\text{采样 token})\\[2pt] \pi_{y_t}\pi_v & v\neq y_t\ (\text{未采样})\end{cases}\)
即**压采样到的错 token、把概率按 \(\pi_v\) 比例抬给其它候选**(Fig.6 右)。三个良性性质:
  1. **保高置信先验**:\(\pi_{y_t}\to1\) 时负梯度被 \((1-\pi_{y_t})\) 缩小，即便错误里含高置信的语法/常识 token,也几乎不动，于是不抹掉预训练学到的高置信先验;
  2. **按先验比例软重排**:抬未采样 token 正比于其当前概率 \(\pi_v\)，等于定向探索"模型本就觉得可能对"的路径;
  3. **改对即停**:NSR 仅在生成错误时更新,一旦不再犯该错就停止更新该样本，相当于隐式正则、**"锁住已掌握的成功经验"不再过拟合**(原文 "locking in successful experiences")。
- **推广到 PPO/GRPO(§4.3)**:PPO/GRPO 在式6 基础上加 (1) clip (2) KL (3) advantage 替代 raw reward。结论:
  - ① clip 只限幅度不改方向;
  - ② KL 在推理任务系数极小或去掉、影响可忽略;
  - ③ advantage \(A_i=\frac{r_i-\text{mean}(r)}{\text{std}(r)}\) 只是按符号缩放梯度,保留 raw reward 的符号。
  - 故"PSR 塌缩、NSR 按先验重分配"的方向性分析对 PPO/GRPO 仍成立。

### W-REINFORCE(式9,据机理设计的算法)
对 PSR 项的奖励幅度乘降权因子 \(\lambda\) 再与 NSR 合并:
\(\displaystyle L_{\text{W-REINFORCE}}(\theta)=\underbrace{-\mathbb{E}_{x\sim D}\!\!\sum_{y:r=1}\!\lambda\,\pi_\theta(y|x)}_{\lambda\cdot L_{\text{PSR}}}\;\underbrace{-\mathbb{E}_{x\sim D}\!\!\sum_{y:r=-1}\!\!\big(-\pi_\theta(y|x)\big)}_{L_{\text{NSR}}}.\)
\(\lambda=1\) 退化为 vanilla REINFORCE;实验用 **\(\lambda=0.1\)**(强烈偏向 NSR)。直觉:保留少量正向信号撑住小 k 的 Pass@1,主体靠 NSR 维持高熵/大 k 的 Pass@k。

### 评测协议(Pass@k 无偏估计,式5)
为降方差,每题采 \(n\ge k\) 个样本、数对的 \(c\) 个,无偏估计 \(\text{Pass@}k=\mathbb{E}_{x\sim D}\big[1-\binom{n-c}{k}/\binom{n}{k}\big]\)。Pass@1≈贪心准确率(exploitation,利用),大 k≈多样化正确生成能力(exploration/推理边界,探索)。
- **关键复现配置(§3.1)**:数据=MATH 训练集(7500 题),verl 框架;prompt batch 1024、每 prompt 8 rollouts、训练温度 1.0、max ctx 4096(Qwen2.5-Math-7B)/32768(Qwen3-4B);mini-batch 256、lr 1e-6。评测:Qwen2.5-Math-7B 采 256 样本(T=0.6,top-p=0.95),Qwen3-4B 采 64 样本(T=0.7,top-p=0.8,top-k=20)。

## 靠不靠谱
> 一句话导读:结论靠"全 Pass@k 谱"和 token 级梯度撑得很硬,但高度依赖 Qwen 这种强先验骨干——换成 Llama 几乎失效,本质更像"精修已有先验"而非"教会新推理"。

- 逐组件必要性:
  - ① **PSR/NSR 解耦**(核心分析工具)——无它回到纠缠的 RLVR,看不到相反效应;
  - ② **全 Pass@k 谱**(k 到 256)——若只看 Pass@1 会误判 PSR≈最好、完全错过 NSR 的大-k 优势(Fig.2-4);
  - ③ **W-REINFORCE 的 λ**——λ=1 即差的 vanilla REINFORCE,λ=0.1 取最佳折中(附录 E 有 λ 消融);
  - ④ entropy-bonus / unlikelihood 对照(附录 B/C)证明 NSR 的"按先验重分配"不可被简单替代。
- 实验与证据(Qwen2.5-Math-7B,Table 1,关键数):
  - AIME2025 Pass@256——base 46.7、PPO 43.3、GRPO 50.0、PSR 43.3、**NSR 53.3**、**W-REINFORCE 56.7**(最高);
  - MATH Pass@1——PPO/W-REINFORCE 并列最高 76.6、NSR 75.7、PSR 仅 74.1;
  - AMC23 Pass@256——NSR 与 base 并列 100.0、PSR/REINFORCE 仅 92.5;
  - 熵动态(Fig.5b):NSR 全程维持接近基座的高熵,PSR 熵快速塌缩,PPO/GRPO 居中(故其大 k 仍低于 base);
  - **强骨干依赖**:Qwen3-4B 非思考模式下 NSR/GRPO 能激活潜在思考、PSR 完全失败(Fig.3);**Llama-3.1-8B 上所有 RL 都使大-k 退化,NSR 只是退化最少**(Fig.4)——结论高度依赖强先验骨干;
  - baseline 公平(同数据/同框架/同评测;PSR/NSR 有效样本更少已说明)。
- 假设与失效边界:
  - 【原文 §3.2 末、Fig.4】结论强烈依赖"骨干本身已编码强推理先验"(Qwen 成立,Llama 基本失效);NSR 在小 k 处可能略弱(§5)。
  - 【推断】仅在**可验证二元对错**任务验证(数学),开放式/带 reward model 任务未测;
  - 【推断】只在 7B 级及以下、MATH 单数据集训练;长 CoT 的逐步信用分配未涉及(仍序列级二元奖励)。
- 祛魅总结:
  - 【推断】真贡献=**把"负向信号(惩罚错误)"从被忽视提升为 RLVR 多样性与 Pass@k 的主要驱动力,并给出干净的 token 级梯度机理 + 一行 λ 降权的强基线**。
  - 被高估处:W-REINFORCE 的普适性被"Qwen 强先验"撑着,换弱骨干就不灵——本质更像"**精修已有先验**"而非"教会新推理"。
  - 被低估处:"改对即停 + 按先验比例重分配"是一个非常干净的"巩固而不遗忘"机制范式,价值超出本文的数学设定。

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号 = 可验证二元对错奖励;但**只用负向(惩罚错误)**那一半(NSR)。
  - 改什么 = 策略模型参数(全参 RL);效果上是 token 级 logit 的定向重分配。
  - 何时改 = on-policy RL 训练期,逐 batch;NSR 仅在生成错误时更新,改对后对该样本停更。
  - 免梯度? = 否,策略梯度/REINFORCE 族。
  - 记忆-技能生命周期 = 无外部记忆;"成功经验"以"改对即停、不再扰动高置信先验"的方式隐式固化进参数。
  - 防遗忘机制 = NSR 内禀性质:负梯度被 \((1-\pi_{y_t})\) 缩小 → 不抹掉预训练高置信先验;改对即停 → 不过拟合已掌握样本(防遗忘/防塌缩)。
- ⑦ 开源代码+框架/harness:https://github.com/TianHongZXY/RLVR-Decomposed(本地已 clone,~15MB,Tier A)。框架=**veRL(vendored)**。【原文 §3.1 "using the verl framework" + 元信息】
- 💰 资源/成本与可扩展性:原文未给 GPU 时/成本;训练规模可推(7B 模型 + MATH 7.5K + prompt batch 1024×8 rollout、mini-batch 256、lr 1e-6)。【原文 §3.1;算力数字未说明】
- 🎯 对"探索-巩固"对标:**强支撑 + 直接可借组件**。NSR 三性质几乎是本课题"巩固/回轨"想要的:
  - ① "改对即停、锁住成功经验" = 巩固而不过拟合;
  - ② "按先验比例重分配、定向探索自己觉得可能对的候选" = 探索/选路偏向"自己能走通的开头";
  - ③ "不抹掉高置信先验" = 防遗忘。
  - **可借**:把 W-REINFORCE 的"正奖励降权 λ"思想与 on-policy 蒸馏结合——在走偏(错)处用强负向信号驱动模型自选恢复分支,在已走通处降权/停手以固化。
  - **缺口/竞品面**:纯参数级、序列级二元奖励,**无 teacher 脚手架、无关键步定位、无 MTP 前瞻**;且强依赖 Qwen 先验。
  - 一句判定:NSR 提供"巩固=改对即停 + 探索=按先验重分配"的极简机制原型,本课题需补"teacher 稀疏接管 + 高熵分叉点定位 + MTP 探针"。
- 🔭 开放问题/未来方向:
  - 【原文 §F】更深理解正负信号平衡、扩展到更多任务/骨干。
  - 【推断】(1) 把序列级二元奖励升级为**关键步/分叉点级**负向信号(只在高熵错误步惩罚),贴合 path-recovery 单点接管;(2) 验证弱先验骨干上能否靠 teacher 蒸馏补"先验不足"这一失效边界;(3) 与 reverse-KL on-policy 蒸馏(MiniLLM 族)融合:用教师分布定义"按谁的先验重分配",而非纯学生先验;(4) λ 自适应调度(随掌握程度动态降权正奖励)。

— 残留待核:0

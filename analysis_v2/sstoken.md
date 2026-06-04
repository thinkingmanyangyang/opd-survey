sstoken | ssToken: Self-modulated and Semantic-aware Token Selection for LLM Fine-tuning | 上海交大 + 上海创智学院(Xiaohan Qin, Xiaoxing Wang…通讯 Junchi Yan) | arXiv 2510.18250 v1(2025-10-21, cs.AI)·ICLR 2026 | 主题线 L2(统一 SFT 视角的 token 级数据选择,偏 SFT 侧)·相关性 中

**原始论文**:https://arxiv.org/abs/2510.18250

## 一眼看懂

> 一句话导读:SFT 时不必每个 token 都学;ssToken 不再借外部 reference 模型,而是拿"模型自己的旧版本"当参照、再加一路注意力语义分,挑出最值得学的 token。

- 🟦 TL;DR:SFT(监督微调,用标注数据继续训模型)里有个观察——"不是所有 token 都值得学"。现有的 token 选择方法(RHO-1、TokenCleaning)有两个痛点:
  - 要额外训练、或借用一个 **reference model**(参照模型,一个更强的同类模型,用来对比 token 难易);
  - 只看 loss(预测误差)这一个信号。
- ssToken 的两招:
  - ① 用"当前模型 vs 它自己历史 checkpoint"的 loss 下降量(Retrospective Excess Loss, REL)当自调制信号,从而免掉 reference model;
  - ② 加一路基于注意力的"语义重要性"分(response token 对 prompt 的注意力总和)。
- 两个分数各自归一化后线性融合(\(\gamma=0.5\)),取分数最高的 top-\(\rho\) 比例 token 来算 loss、其余 mask 掉。四个基座(3B~14B)上平均分都最优。【原文 §3, Table 1】
- 最巧的一步:把 RHO-1 的 Excess Loss(对外部 reference 算的)改写成 REL(对自己历史算的)——\(\mathrm{REL}(x_i)=L_{\theta_{his}}(x_i)-L_\theta(x_i)=\log\frac{P_\theta(x_i\mid x_{<i})}{P_{\theta_{his}}(x_i\mid x_{<i})}\)【原文 Eq.3】。
  - 抽掉它,就退回"需要 reference model"的老范式,卖点(免 reference)直接没了;
  - 语义分(注意力)是第二根支柱,抽掉它就退化成纯 loss 选择,QA 类任务上的增益大幅缩水;
  - 两根支柱都要,才有协同。【原文 §4.3, Fig.3a】

## 为什么做

> 一句话导读:即便样本级清洗做完,留下的"好样本"里仍藏着 token 级噪声;而现有 token 选择要么依赖外部模型、要么只看 loss——ssToken 想两个都补上。

- 研究背景:SFT 阶段"数据质量 > 数量"已成共识。但即便做过样本级清洗,高质量数据里仍有 **token 级噪声**(无关或冗余的片段)。
  - RHO-1 首次做 token 级选择,大幅超过用全部数据训练;
  - TokenCleaning 在 SFT 场景进一步优化(给出 fixed-model / self-evolving 两种策略)。【原文 §1-2】
- 解决的具体痛点——现有 token 选择有两大局限:
  - (1) **需训练或访问额外 reference model**:直接借一个更强、同 tokenizer 的模型并不总可行;单独训一个 reference 又增加成本;而且 reference 的质量会显著影响选择效果(引 Pang 2025)。
  - (2) **只靠 loss 信息**:token 的 loss 反映的是"预测有多不确定",不一定反映它在语境里的语义重要性。频繁但语义无信息的 token,可能和任务关键 token 有相近的 excess loss;只看 loss 容易误删有信息的内容。【原文 §1, lines 48-51 / 122-127】
- 相关工作 & 各自不足(梳理这条线的来龙去脉):
  - **样本级数据选择(粒度太粗这一路)**:包括 LIMA、DEITA、DS²(用 GPT 打分或梯度影响选高质样本)、Superfiltering(用小代理模型的困惑度筛样本)、Cherry-LLM。这一路能"用 <10% 数据近乎无损地微调",但都停在**样本粒度**,清不掉一个保留样本内部的 token 级噪声。ssToken 正是承认"样本级清完了仍有 token 噪声"才接着做 token 级。【原文 §2, lines 184-191】
  - **token 级选择开山——RHO-1(Lin 2024)**:它把"继续训练时 token loss 的变化轨迹"分成四类——persistent high(高→高)、increasing(低→高)、decreasing(高→低)、consistently low(低→低)。策略是优先学 H→L(能稳定学会的任务相关 token),避开已掌握的(L→L)与持续难/噪声的(H→H、L→H)。它靠在高质任务子集上**再训一个 reference model**,算 Excess Loss \(\mathrm{EL}(x_i)=L_\theta(x_i)-L_{\theta_{ref}}(x_i)\) 来给 token 打分。短板就是:需 reference + 只看 loss。【原文 §2.1-preliminaries, lines 213-230】
  - **TokenCleaning(Pang 2025)**:把 token 级清洗专门做到 SFT 场景,给出 fixed-model 与 self-evolving 两种策略;并实证 **reference model 的容量会显著影响选择质量**。这正是 ssToken 拿来当"为什么要免/换 reference"的直接论据——大规模数据下 reference 难持续更新,而 history model 可以经 checkpoint 或 EMA 迭代,提供更稳的长程指引。【原文 §3.2, lines 253-257】
  - **注意力做重要性这一路**:Luo 2020、Adnan 2024(都用于 KV-cache 压缩)用"其它 token 对目标 token 的注意力之和"来估 token 重要性。但在因果 mask 的 next-token 范式下,这会引入**位置偏置**——不同位置的 token 收到的注意力极不均。ssToken 反其道而行,取"response→prompt"方向来规避这个偏置(见下)。又因为这些方法常需输出完整的注意力矩阵、与 FlashAttention 不兼容,ssToken 改用 hook + 单层重算来解决。【原文 §3.3, lines 282-285, 334-340】
  - 动机链(逐步推):
    1. 数据质量决定 SFT 效果;
    2. 样本级清洗后,仍残留 token 级噪声;
    3. token 级选择有效,但依赖 reference、且只看 loss;
    4. 于是用"模型自己的历史版本"当天然 teacher(免 reference),再用注意力补语义信号;
    5. 得到 ssToken。
  - 与最近邻工作的精确 Δ:相比 RHO-1 / TokenCleaning,有两点关键差异——
    - ① **reference model → history model**:历史模型可经 checkpoint/EMA 持续更新,提供更稳的长程指引,且零额外训练;
    - ② **单 loss 信号 → loss + 注意力双信号**:注意力轴与 loss 轴正交。
    - 差在"免 reference + 正交语义轴";有用是因为两信号互补、且 QA 任务上语义轴带来额外增益。【原文 §1 contributions, §4.2】

## 怎么做 + 靠不靠谱

> 一句话导读:对每个 response token 同时算两个分(自己 vs 历史的 loss 降幅、对 prompt 的注意力强度),归一融合后取 top-\(\rho\) 学;消融显示两个分都不可少,但它不省训练时间,卖点是省掉 reference 的训练。

- 方法流水线(输入→输出,逐模块):
  1. **输入**:一个 SFT 样本 \(x=\{x_1,\dots,x_L\}\),分成 prompt 段 \(I_{prompt}\) 与 response 段 \(I_{resp}\)。只对 response token 做选择和算 loss,prompt 仅作条件。
  2. **REL 分(自调制 loss 轴)**:对每个 response token \(x_i\),同时跑当前模型 \(\theta\) 与历史模型 \(\theta_{his}\) 各一次 forward,取各自的 per-token NLL(负对数似然),作差得到 \(\mathrm{REL}(x_i)=\log\frac{P_\theta(x_i\mid x_{<i})}{P_{\theta_{his}}(x_i\mid x_{<i})}\)(Eq.3)。直觉:当前模型相比历史在某 token 上 loss 大降,说明该 token"还可学、且有信息";已掌握或噪声 token 的 REL 接近 0 或为负。
  3. **AttnScore 分(语义轴)**:forward 时用 **hook 把目标层的 hidden state 存下来、再单独重算该层**,得到注意力矩阵(这样就避开输出完整注意力,兼容 FlashAttention)。取子矩阵 \(A^{(h)}_{resp\to prompt}:=A^{(h)}[I_{resp},I_{prompt}]\in\mathbb{R}^{L_{resp}\times L_{prompt}}\)(Eq.5),沿 prompt 维求和 \(\mathrm{AttnScore}^{(h)}=A^{(h)}_{resp\to prompt}\cdot\mathbf{1}_{prompt}\)(Eq.6),再对所有 head 求平均:
     \(\displaystyle \mathrm{AttnScore}(x_i)=\frac{1}{H}\sum_{h=1}^{H}\mathbf{1}^{\top}_{prompt}\cdot\mathrm{softmax}\!\left(\frac{q^{(h)}_i K^{(h)\top}+M_i}{\sqrt{d_k}}\right),\)
     其中 \(M_i\) 是因果 mask(Eq.7)。该分天然 \(\in[0,1]\)。
  4. **归一 + 融合**:REL 在样本内做 min-max 归一 \(\mathrm{Normalize}(\mathrm{REL}(x_i))=\frac{\mathrm{REL}(x_i)-\min_j \mathrm{REL}(x_j)}{\max_j \mathrm{REL}(x_j)-\min_j \mathrm{REL}(x_j)}\)(Eq.8),映到 [0,1];AttnScore 本就在 [0,1]。两者线性融合 \(\mathrm{Score}(x_i)=\gamma\cdot\mathrm{Normalize}(\mathrm{REL}(x_i))+(1-\gamma)\cdot\mathrm{AttnScore}(x_i)\)(Eq.9,默认 \(\gamma=0.5\))。
  5. **top-\(\rho\) 选择 + mask loss**:按 Score 取每个样本 response token 的前 \(\rho\) 比例,用指示函数 \(I_\rho(x_i)=\mathbb{1}[x_i\in\text{top-}\rho\text{ by }\mathrm{Score}]\) 标记;loss 只对选中的 token 算,并按选中数归一:
     \(\displaystyle L_\theta(x)=-\frac{1}{L_{resp}\cdot\rho}\sum_i I_\rho(x_i)\,\log P_\theta(x_i\mid x_{<i}),\)
     (Eq.10/11,默认 \(\rho=0.6\))。输出梯度,未选中的 token 不进 loss。
- 逐组件必要性(每条都有消融支撑):
  - **REL(自调制)**:负责"免 reference 地找可学 token"。没它就退回需 reference 的老路。消融 \(\gamma=1\)(纯 REL)已超过用全部数据训练。【原文 Fig.3a】
  - **AttnScore(语义感知)**:负责补上 loss 看不到的语义/指令相关性。没它(\(\gamma=1\))时,QA 任务上增益缩小;消融 \(\gamma=0\)(纯注意力)也超过全数据。【原文 §4.2, Fig.3a】
  - **融合 \(\gamma\)**:有消融(\(\gamma\in\{0,0.25,0.5,0.75,1\}\)),中间值最好、两极(只用一个信号)都次之——证明两信号互补且都必要。【原文 §4.3, Fig.3a】
  - **比例 \(\rho\)**:有消融(\(\rho\in\{0,0.2,\dots,1\}\)),3B/8B 峰值在 0.6,14B 峰值在 0.8(连同 Random/RHO-1/TokenCleaning 也都在 0.8 达峰)。【原文 §4.3 + lines 853-855】
  - **层选择**:Appendix B 消融了早/中/深层,结论"深层更好"——浅层抓句法、局部结构、位置关系,深层抓语义抽象、高层概念、任务相关的全局信息,更利于指令遵循。属经验消融。【原文 §3.3, lines 326-333】
  - **EMA 更新历史模型**:标注为"(Optional)",非必需;基线可直接用 SFT 前的 base 当历史模型,公式为 \(\theta^t_{his}=\alpha\,\theta^{t-1}_{his}+(1-\alpha)\,\theta^t\)(Eq.4,\(\alpha\) 为平滑系数)。【原文 Eq.4】
- 关键机制/公式(直觉):
  - REL 与 RHO-1 的 Excess Loss **方向相反**:
    - EL \(=L_\theta-L_{\theta_{ref}}\),学的是"未来 loss"——对一个更强的 reference,衡量未来训练还能进步多少;
    - REL \(=L_{\theta_{his}}-L_\theta\),学的是"历史 loss"——对自己更弱的过去,衡量已经进步了多少。
    - 若当前模型相比历史在某 token 上 loss 大降,该 token 更可能"可学且有信息"。
  - AttnScore 直觉:SFT 里所有 response token 都注意同一段固定长度的 prompt,而 prompt 编码了任务/指令,所以"response token 对 prompt 的注意力强度"可作任务相关性的代理。之所以专取"response→prompt"方向:因果 mask 下"别的 token 对目标 token"的注意力会引入位置偏置,换方向可规避。【原文 §3.2-3.3】
- 实验与证据:
  - 数据池:从 5 个常用 SFT 集(Flan v2、OpenAssistant、Stanford Alpaca、Dolly、WizardLM,共 300k)采 50k;reference 基线在 DS² 筛出的 10k 高质子集上训。【原文 §4.1, lines 388-408】
  - 评测用 10 个通用基准:TriviaQA、TruthfulQA、MMLU、ARC-C、ARC-E、TyDiQA、Winogrande、HellaSwag、LogiQA、AGIEval,用 lm-eval-harness。基座 4 个:LLaMA-3.2-3B、LLaMA-3.1-8B、Qwen-2.5-7B、Qwen-2.5-14B。【原文 Table 1】
  - 关键数字:ssToken 在四个基座上平均分均最优,相对全数据 +4.3% / +3.4% / +1.3% / +2.1%(对应 3B/8B/7B/14B);相对此前 token 方法最高 +2.8%【原文 §1, lines 165-167, 735-737】。增益集中在 TyDiQA、TriviaQA、AGIEval 等需要指令遵循的 QA(归功于注意力分量);MMLU、ARC 等知识密集任务上 token 选择基本无提升【原文 lines 743-755】。RHO-1/TokenCleaning 在 Qwen 系上仅与全数据持平甚至更差,而 ssToken 跨族稳定【原文 lines 737-742】。
  - baseline 公平吗:同一基座下所有方法用相同 \(\rho\);reference 基线给了 10k 高质子集训 reference——公平。一个隐性不平:ssToken 多算了注意力(虽轻量),且 REL 需维护一份历史模型副本(显存/状态成本未量化)。
  - "看着强但没回答核心":不省训练时间——被 mask 的 token 仍要走 forward,所以 token 选择类方法都不减训练时间(作者诚实承认,Fig.2)【原文 lines 798-804】;它的卖点是"省掉 reference 的训练时间"。
- 假设与失效边界:
  - 【原文】训练早期 history 等于 current,REL 近似随机(作者明说,lines 266-268)。
  - 【原文】只在通用指令 SFT 上验证,未触及长 CoT 推理蒸馏。
  - 【推断】AttnScore 只取"response→prompt"的总注意力,在长 prompt / 多轮 / 工具调用场景下是否有效未验证;层选择(深层更好)依赖经验消融,跨架构未必稳。
  - 【推断】"无 reference"并非完全免费:EMA 历史模型需额外的参数副本,显存成本论文未充分量化。
- 祛魅总结【推断】:真贡献=两个工程性改进(REL 替 reference + 注意力语义轴),组合后跨四基座稳定地小幅提升,且免掉了 reference 训练。包装上"自调制/语义感知"听着大,本质仍是 RHO-1 式"top-\(\rho\) token + loss mask"范式的增量补丁。高估之处:把它当成"普适显著提升"——Qwen-7B 仅 +1.3%,单项(TruthfulQA)有时反低于 BASE/FULL。低估之处:免 reference 这点在工程上很实用,降低了 token 选择的落地门槛。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=token 级 REL(自己 vs 历史的 loss 下降)+ response→prompt 注意力强度 | **改什么**=哪些 token 进 SFT loss(top-\(\rho\) 选择 + mask) | **何时改**=每个训练 step 在线评估(REL 依赖当前/历史 checkpoint) | **免梯度?**=否,被选 token 走标准 NLL 梯度;选择本身免梯度(基于 loss/注意力打分) | **记忆-技能生命周期**=不涉及(纯单轮 SFT 数据加权) | **防遗忘机制**=无(单阶段 SFT)
- ⑦ 开源代码+框架/harness:https://github.com/jianke0604/ssToken(已克隆 ~2.8MB,代码完整可跑);框架=**自写训练脚本 + FSDP**(支持 LoRA)+ hook 重算注意力兼容 FlashAttention + EleutherAI lm-evaluation-harness 评测。关键文件 `scripts/{calculate_token_loss.py, finetune_with_hook.py, generate_token_label.py, finetune.py}`,代码中 `combined=ratio·diff_norm+(1−ratio)·resp2prompt_scores`(ratio=\(\gamma\))、`data_prop`=\(\rho\),与论文 Eq.9/10 一致。
- 💰 资源/成本与可扩展性:训练用 3B~14B 模型;**不减训练时间**(mask token 仍走 forward),卖点是省掉 reference 训练;注意力用 hook 重算"目标层",引入边际开销;EMA 历史模型副本的显存成本原文未说明。【原文 §4.2 + lines 798-812】
- 🎯 对"探索-巩固"对标:**可借组件(弱)**。一句判定:ssToken 是 SFT 侧的 token 选择,与本项目"teacher 稀疏脚手架 + on-policy 自选路径"不在一个范式,但其"REL=用自己历史当天然 teacher"思路与"探索阶段偏向自己能走通的开头"有微弱同构,且"注意力定位语义关键 token"可作"切关键步"的候选信号之一。依据:它是 off-policy SFT 数据加权,不含 on-policy rollout、不含 teacher 监督、不含记忆/技能,故只能作边缘借鉴而非竞品。【推断,依据范式差异】
- 🔭 开放问题/未来方向:【原文】\(\rho\) 非普适、依赖手调,作者建议把 \(\rho\) 的优化集成进算法、按训练进度/容量/数据质量自适应(lines 875-883)。【推断】把 REL+注意力双信号迁移到长 CoT / on-policy 蒸馏的 token 选择(与 TIP 的"熵×散度"两轴可对照,本质都是"loss/不确定性轴 + 第二根正交轴");验证注意力语义轴在多轮/工具调用场景是否还成立;量化 EMA 历史模型的真实显存成本。

〔本篇与既有 analysis/sstoken.md 核对:v1 footnote 称"14B 用 0.8 是误读"——经 PDF §4.2(lines 731-732)与 §4.3(lines 853-855)核实,**14B 确实用 \(\rho=0.8\)**(且同基座下所有对照方法亦在 0.8 达峰),v1 footnote 有误,本篇已更正。本次增强:5 条核心公式(Eq.3/7/8/9/10)转 MathJax 并从 PDF 抄准;相关工作补全样本级选择族(LIMA/DEITA/DS²/Superfiltering)、RHO-1 四类 loss 轨迹与 EL 定义、TokenCleaning 的 reference 容量论据、注意力重要性两路(Luo/Adnan)与位置偏置缘由;方法流水线细化到 hook 单层重算与每模块输入→输出。无新增待核项。〕

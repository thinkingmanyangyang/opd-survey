distillm2 | DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs | KAIST AI(Jongwoo Ko, Sungnyun Kim, Se-Young Yun)+ Microsoft(Tianyi Chen 等) | 2025-05-30 · arXiv 2503.07067 v2 · ICML 2025 Oral(top 1%) | 主题线 L1(白盒蒸馏·对比式)，兼 L6 · 相关性 高

**原始论文**：https://arxiv.org/abs/2503.07067

## 一眼看懂
- 🟦 TL;DR：以往蒸馏对"教师生成的数据(TGO)"和"学生生成的数据(SGO)"用**同一个 loss**,忽略了"loss 形式 × 数据类型"的协同。本文先观察到 KL 与 RKL 的**非对称行为**——KL 在教师概率 \(p\) 高的"头部"会把学生概率 \(q\) 往上抬(pulling-up)、RKL 在 \(p\) 低的"尾部"会把 \(q\) 往下压(pushing-down)——于是设计**对比式蒸馏 loss(CALD)**:对教师响应用 SKL(抬高其概率)、对学生响应用 SRKL(压低其概率),并配 \(\alpha\) 课程与 \(\beta\) 线性递增。在指令/数学/代码/VLM 全面超过 GKD、DistiLLM、Speculative KD。【原文 §Abstract / §3.1 / Eq.2,5】
- 最巧的一步：**"教师响应→SKL、学生响应→SRKL"的差异化配对**(而非对两类数据用同一散度)。抽掉这个配对就垮——论文明确:在**同一类**响应上对 SKL/SRKL 做任意 \(\gamma\) 插值 \(\gamma D_{\mathrm{SKL}}+(1-\gamma)D_{\mathrm{SRKL}}\) 都**不优于**单用其一(引自 DistiLLM 的发现);只有"不同类型响应用不同 loss"才显著提升(Fig.1c)。另一不可抽的是**用线性而非 log-sigmoid 形式**(Remark 1)——直接把 DPO 搬到 KD(DPKD)会 reward hacking、把 \(q(y_s)\) 压崩(NLL 飙到 91.25)。【原文 §3.1.2 末 / Eq.4 / Remark 1 / Fig.1b】

## 为什么做
- 研究背景：把大教师蒸成小学生(sLM,如 Llama-3.2 / Gemma-2 都靠 KD);研究分两条线——**改 loss**(KL 失真→SKL 等替代,Wen 2023 / Gu 2024 / Ko 2024)与**策展数据**(纯 TGO 有训练-推理失配→引入 SGO,Lin 2020 / Xu 2024b)。本文站在前作 DistiLLM(SKL/SRKL + 自适应 off-policy,Ko 2024)之上。【原文 §1 / §2.2】
- 解决的具体痛点：① 现有蒸馏对 TGO/SGO **施加相同 loss**,错失"loss×数据"协同→提升受限;② 想借 DPO 式对比思想到 KD,但**直接套 DPO(DPKD,Li 2024b,把参考模型换成教师)易 reward hacking**——因 \(p(y_s\mid x)\) 本身极小,DPKD 只顾拉大 \(\tfrac{q(y_t)}{p(y_t)}\) 与 \(\tfrac{q(y_s)}{p(y_s)}\) 的差距,会**过度压低 \(q(y_s)\)**(NLL 91.25),学生丢失预训练信息而非拟合教师(教师响应 NLL 仅 20.29,严重失衡)。【原文 §1 / §3.1.1 / Eq.4 / Fig.1b】
- 相关工作 & 各自不足(原文 §2.1,逐条点名)：ImitKD(Lin 2020,首用 SGO 作蒸馏数据)、GKD(Agarwal 2024,on-policy + RKL/JSD,每步生成)、Wen 2023(f-散度,含 TVD/JSD)、MiniLLM(Gu 2024,策略梯度降 RL 方差)、Speculative KD(Xu 2024b,投机解码生成数据)、DistiLLM(Ko 2024,SKL + 自适应 off-policy,前作 SOTA)。共性:要么对两类数据同 loss、要么把 DPO 简单搬过来会崩。**少有人把"对比/差异化处理"系统扩展到 KD**。【原文 §2.1 / 附录 A】
- 动机链：现状(蒸馏对 TGO/SGO 同 loss;DPO 在偏好对齐有效)→ 缺陷(同 loss 错失协同;DPKD 直接套会 reward hacking)→ 观察(KL 抬头部、RKL 压尾部的非对称性,Fig.1a 长尾 toy 数据上 KL 把 \(q\) 在 \(p\) 大处抬、RKL 在 \(p\) 小处压)→ 所以(用 SKL 处理"该抬的教师响应"、SRKL 处理"该压的学生响应",并改成线性形式以正则化、避免压崩)。【原文 §3.1】
- 与最近邻工作的 Δ：相对**前作 DistiLLM(SKL + off-policy,对所有数据用同一 skew 散度)**——DistiLLM-2 把"用哪个散度"与"数据是 TGO 还是 SGO"**绑定**起来(对比式)。相对 **DPKD/DPO**——CALD 是线性、可 token 级分解、且靠 \(\tilde q\) 与 \(p\) 的线性依赖正则化 \(q(y_s)\) 的下降,从而不 reward hacking。数据侧:把前作的逐步 off-policy 改为**每 epoch 前 vLLM 批量采样(batch on-policy,沿 Rosset 2024)**,兼顾 vLLM 加速与"adaptive off-policy 哲学"。差在"差异化配对 + 线性对比"这两点。【原文 §3.1.2 Remark 1 / §2.2 Summary】

## 怎么做 + 靠不靠谱
- **方法流水线(Alg.1,可复现级)**：输入(教师 \(p\)、学生 \(q_{\theta_0}\)、prompt 集、初始 skew 系数 \(\alpha_0\)) → 每个 epoch \(e\):
  1. **(批量采样)** 用 vLLM 从教师 \(p(\cdot\mid x)\) 与学生 \(q_{\theta_{e-1}}(\cdot\mid x)\) 各采一批响应 \(y_t,y_s\),构成本 epoch 训练集 \(\mathcal{D}_t=\{(x,y_t,y_s)\}\);
  2. 每个 mini-batch \(\tau\):
     - **(α 课程)** 按一致性闭式更新 \(\alpha_t\leftarrow1-(1-\alpha_0)\cdot\tfrac{m}{p(y_s\mid x)-q_\theta(y_s\mid x)}\),\(\alpha_s\leftarrow1-(1-\alpha_0)\cdot\tfrac{m}{p(y_t\mid x)-q_\theta(y_t\mid x)}\);
     - **(β 调度)** \(\beta\leftarrow\mathrm{clip}\big(\tfrac{e}{E}+\tfrac{\tau}{T},\;\beta_0,\;1\big)\)(随训练线性递增);
     - **(更新)** 最小化下面的 CALD/DistiLLM-2 loss。
  输出:蒸馏好的学生 \(q_{\theta_E}\)。【原文 Eq.2 / Alg.1】
- **核心损失的真实形式 + 直觉**：
  - 序列级 KL 的 token 分解(Eq.1):\(D_{\mathrm{KL}}(x,y;p\|q_\theta)=\sum_{t=1}^{T}p(y_t\mid y_{<t},x)\log\tfrac{p(y_t\mid y_{<t},x)}{q_\theta(y_t\mid y_{<t},x)}\);反向 KL \(D_{\mathrm{RKL}}(x,y;p\|q_\theta)=D_{\mathrm{KL}}(x,y;q_\theta\|p)\);skew 版同 DistiLLM。
  - **DPO(对照,Eq.3)**:\(-\log\sigma\big(\lambda\log\tfrac{q_\theta(y_w\mid x)}{q_{\mathrm{ref}}(y_w\mid x)}-\lambda\log\tfrac{q_\theta(y_l\mid x)}{q_{\mathrm{ref}}(y_l\mid x)}\big)\),抬 \(q(y_w)\)、压 \(q(y_l)\)。
  - **DPKD(失败的朴素搬运,Eq.4)**:把 \(q_{\mathrm{ref}}\) 换成教师 \(p\):\(-\log\sigma\big(\lambda\log\tfrac{q_\theta(y_t\mid x)}{p(y_t\mid x)}-\lambda\log\tfrac{q_\theta(y_s\mid x)}{p(y_s\mid x)}\big)\);因 \(p(y_s\mid x)\) 本就极小,优化只顾拉大差距→**过度压低 \(q(y_s)\)**(NLL 91.25)。
  - **CALD(本文核心,Eq.5)**:\[\mathcal{L}_{\mathrm{CALD}}=\frac{1}{2|\mathcal{D}|}\sum_{(x,y_t,y_s)\sim\mathcal{D}}\Big[D^{(\alpha)}_{\mathrm{SKL}}(x,y_t)+D^{(\alpha)}_{\mathrm{SRKL}}(x,y_s)\Big].\]即对**教师响应 \(y_t\)**(多数 \(p(y_t)\gg0\))用 SKL(抬),对**学生响应 \(y_s\)**(多数 \(p(y_s)\approx0\))用 SRKL(压)。
  - **完整目标(Eq.2,加 β 平衡)**:\[\mathcal{L}_{\text{DistiLLM-2}}=\frac{1}{2|\mathcal{B}|}\sum_{(x,y_t,y_s)\in\mathcal{B}}\Big[(1-\beta)\,D^{(\alpha_t)}_{\mathrm{SKL}}(x,y_t)+\beta\,D^{(\alpha_s)}_{\mathrm{SRKL}}(x,y_s)\Big].\]
  - **与 DPO 的形式桥接(Remark 1,Eq.6)**:Eq.5 可改写成类 DPO 的**线性**式 \(-\mathbb{E}_{y_t\sim p,\,y_s\sim q_\theta}\big[\tfrac{1}{\lambda}\big(\lambda\log\tfrac{\tilde q_\theta(y_t\mid x)}{p(y_t\mid x)}-\lambda\log\tfrac{q_\theta(y_s\mid x)}{\tilde p(y_s\mid x)}\big)\big]\),其中 \(\tilde q_\theta=\alpha p+(1-\alpha)q_\theta\)、\(\tilde p=\alpha q_\theta+(1-\alpha)p\)。**两点关键差异**:① 用**线性**而非 log-sigmoid→允许 token 级分解、且显式按 \(p(y_t)\) 或 \(q(y_s)\) 加权(如 Eq.1);② 因 \(\tilde q_\theta\) 与 \(p\) 的**线性依赖**,自动**正则化 \(q(y_s)\) 的过度下降**,解决 DPKD 的 reward hacking。
  - **机制直觉**:\(\mathrm{KL}=\sum p\log\tfrac{p}{q}\),为压低加权平均会在 \(p\) 大处把 \(q\) 抬上去(pulling-up,适合"该学的教师响应");\(\mathrm{RKL}=\sum q\log\tfrac{q}{p}\) 会在 \(p\) 小处把 \(q\) 压下去(pushing-down,适合"该抑制的学生响应")。CALD 就是"按数据该被抬还是该被压,分派对应方向的散度"。【原文 §3.1.2 / Fig.1a / Remark 1 / 附录 B】
- 逐组件必要性：
  - **CALD 对比配对(核心)**：没它(同 loss/同类插值)就回到 DistiLLM 水平,Fig.1c 显示 CALD(SKL) 收敛更快、ROUGE-L 更高(且 CALD 用朴素 KL/RKL 也有效,但 SKL/SRKL 为骨架更强)。【原文 §3.1.2 / Fig.1c】
  - **线性形式而非 log-sigmoid(Remark 1)**：没它就退化成 DPKD→reward hacking(\(q(y_s)\) NLL 91.25)。线性使 \(\tilde q_\theta\) 与 \(p\) 线性依赖,正则化 \(q(y_s)\) 过度下降。【原文 Remark 1 / Eq.6 / Fig.1b】
  - **\(\alpha\) 课程**：易样本(\(\bar p\approx\bar q\),分母大)用小 \(\alpha\)、难样本用大 \(\alpha\);自适应调散度强度。【原文 Alg.1 line 11】
  - **\(\beta\) 线性递增**：前期主拟合教师(SKL 权重 \(1-\beta\) 大)、后期主用 SGO 反馈减失配(SRKL 权重 \(\beta\) 大)。【原文 Alg.1 line 13 / §3.3】
  - **数据策展(§3.2)**：对 SKL 用纯教师生成、对 SRKL 用纯学生生成最优;反直觉发现——speculative/更强 LLM 的"高质量"响应反而更差,说明**教师响应的高 log-prob 比"高质量"更关键**。【原文 §3.2】
- 实验与证据：
  - 教师→学生对:指令 Qwen2-7B-Inst→Qwen2-1.5B、Mistral-7B-Inst→Danube2-1.8B、Gemma-2-9B→2B;数学 Qwen2(.5)-Math-7B→1.5B;代码 DS-Coder-6.7B/Qwen2.5-Coder-7B→1.3B/1.5B;VLM LLaVA-1.5-7B→TinyLLaVA-1.4B。
  - 数据:指令 UltraChat200k(50K prompts,评 AlpacaEval/Evol-Instruct/UltraFeedback);数学 MetaMathQA(50K,评 GSM8K/MATH);代码 WizardCoder(评 HumanEval/MBPP);VLM RLAIF-V(83K,评 OK-VQA/TextVQA)。
  - 关键数字:toy(Fig.1a,长尾数据)显示 KL 抬头部、RKL 压尾部;LLM 实验(Fig.1b,Mistral-7B→Danube2-1.8B)——DPKD 学生响应 NLL 飙到 **91.25**(reward hacking,教师响应 NLL 20.29)、CALD 保持正常(教师 1.61/学生 1.54 量级);CALD(SKL) 比单 SKL/SRKL 收敛更快、ROUGE-L 更高(Fig.1c)。指令/数学/代码/VLM 全面 SOTA(优于 GKD、DistiLLM、Speculative KD)。【原文 Fig.1 / §4】
  - baseline 公平吗:与 GKD/DistiLLM/SpecKD/DPKD/DPO 同设置,批量生成用 vLLM 统一加速——较规范。
  - "看着强但没回答核心问题":核心是"对比配对带来协同增益",Fig.1c + 全任务 SOTA + DPKD 反例(NLL 91.25)较好支撑;"高质量数据反而更差"是有趣但属相关性观察(无因果隔离)。
- 假设与失效边界：
  - 【推断】**白盒 + 同词表前提**:需教师 logits;Qwen2.5 跨尺度还需 `resize_embedding.py` 对齐分类头,限制适用范围。依据:Eq.1 token 级 KL 需共享词表。
  - 【推断】超参较多(\(\alpha_0\)、\(\beta\) schedule、\((1-\beta)/\beta\) 配比),最优区间无闭式依据;每 epoch 重生成 TGO/SGO 仍有额外开销(虽 vLLM 批量加速)。
  - 【原文 §3.2】"高质量响应反而更差"缺因果证据——可能与分布匹配度而非 log-prob 本身耦合;VLM 仅单一对(LLaVA→TinyLLaVA),外推性有限。
- 祛魅总结【推断】：
  - 真贡献:把"散度方向"与"数据类型"绑定的视角是干净且有用的洞察(KL 抬/RKL 压),并通过线性化巧妙避开 DPKD 的 reward hacking——Remark 1 提供了与偏好优化的形式桥接(Eq.6 把"对比"落到可分解的线性式),机制叙事与代码高度一致(tea_pos_kl/ref_pos_kl + detach)。ICML Oral 的分量主要在这个"协同"洞察 + 全面 SOTA。
  - 包装/被高估处:"对比"一词更多指 TGO vs SGO 的**差异化处理**,而非 DPO 式成对 margin;CALD 本质是"对两类数据分别选散度方向"的工程组合。其有效性高度依赖前作 DistiLLM 的 SKL/SRKL backbone(本文承认 KL/RKL 也能用、但 skew 更好)。

## 结构化抽取
- 🎯 机制速览6轴：
  - **学什么信号**：教师 logits(白盒);对**教师响应**用 SKL 学"该抬的头部"、对**学生响应**用 SRKL 学"该压的尾部"。
  - **改什么**：改**目标函数**(单 loss→TGO/SGO 差异化的对比 loss CALD)+ 数据策展(TGO 纯教师生成、SGO 纯学生生成)+ \(\alpha/\beta\) 调度。
  - **何时改**：训练时;每 epoch 前 vLLM 批量生成 TGO/SGO,每步按课程更新 \(\alpha\)、按 schedule 递增 \(\beta\)。
  - **免梯度?**：是监督式蒸馏(最小化散度),无策略梯度/RL 奖励;"on-policy"仅指每 epoch 用学生当前分布批量采样。
  - **记忆-技能生命周期**：无外部记忆/技能库(off-policy buffer 是前作 DistiLLM 的,本作改为 batch-per-epoch 生成);技能=教师生成行为压进学生参数。
  - **防遗忘机制**：CALD 的线性正则化**防止 \(q(y_s)\) 被过度压低**(避免学生"丢失预训练信息",§3.1.1)——这是一种针对 reward hacking 的"防退化",可视为弱防遗忘;但非针对持续学习的灾难性遗忘。
- ⑦ 开源代码+框架/harness：https://github.com/jongwooko/distillm-2 (本地已克隆 ~5.9MB,2025-06 发布)。**框架=HF alignment-handbook + Accelerate + DeepSpeed ZeRO-3 + vLLM(0.5.4)生成 + FlashAttention-2**(无 veRL/TRL)。核心 `src/distillm_trainer.py`(CALD loss,含 tea_pos_kl/ref_pos_kl、\(\alpha\) clip 课程、gradual \(\beta\),学生分支 detach)、`src/run_distillm.py`/`run_distivlm.py`/`run_sft.py`;生成 `generate/generate_vllm.py`。代码可得性高。【本地仓库核查】
- 💰 资源/成本与可扩展性：每 epoch 用 vLLM 批量生成 TGO/SGO(比每步 on-policy 省,§2.2/App D.1)；能更好处理教师-学生 capacity gap(教师增大单调提升)。规模到 9B 教师/1.4–2B 学生。【原文 §2.2 / §4】
- 🎯 对"探索-巩固"对标：**中等支撑(差异化处理 + 防退化的损失设计)**。判定依据:CALD 的"对不同来源数据施加不同方向的学习压力"与本课题"探索(发现自己能走通的开头,抬高)vs 巩固(走偏处压制/纠正)"在**直觉上同构**——SKL 抬"该学的"、SRKL 压"该抑制的",可类比"巩固有效路径、抑制走偏分支"。**可借组件**:① 线性对比 loss 的"防过度压低(正则 \(q(y_s)\))"机制(Eq.6 的线性依赖)可防止 OPD 在压制学生错误轨迹时把模型压崩;② "教师响应高 log-prob 比高质量更关键"的发现对 OPD 选 teacher 轨迹有参考。**缺口**:① 仍是处处 token 级散度匹配,非"稀疏关键步脚手架";② "对比"是数据类型层面(TGO vs SGO)而非"选路/回轨"层面,无 path-recovery 单点接管;③ 无 on-policy 自选恢复分支、无 MTP/前瞻、无技能库。属"把白盒蒸馏做精"的方向,与探索-巩固有直觉共鸣但机制不直接对应。
- 🔭 开放问题/未来方向：
  - 【原文 §6】把对比式 KD 扩展到偏好对齐(用更好的 reference model)与 VLM;
  - 【推断】把"差异化散度方向"从"TGO vs SGO"细化到"关键步 vs 普通步"(向稀疏脚手架靠拢);为"高质量≠高 log-prob"补因果隔离实验;与 RLVR/on-policy 自选恢复分支结合。

RETURN: distillm2 | 读PDF=是(§1–3.2 全文 + Eq.1–6/Remark1/Alg.1/Fig.1,核 CALD/DPKD/α课程/β调度) | 加厚=是(新增 Eq.1 分解、DPO Eq.3、DPKD Eq.4、CALD Eq.5、完整目标 Eq.2、Remark1 线性桥接 Eq.6、α/β 闭式更新全 MathJax;related work逐条点名 ImitKD/GKD/Wen/MiniLLM/SpecKD/DistiLLM;补 Alg.1 两步流水线) | LaTeX公式=9条(Eq.1分解、DPO Eq.3、DPKD Eq.4、CALD Eq.5、目标Eq.2、Remark1 Eq.6、α课程、β调度、内联多处) | 待核=0

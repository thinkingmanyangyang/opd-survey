fest | FEST: Boosting RLVR via Randomly Selected Few-Shot Guidance | UIUC（Kai Yan、Alexander G. Schwing、Yu-Xiong Wang） | 2026-05-14 (arXiv:2605.15012 v1，标注 Preprint / Ongoing Work) | L2 统一SFT-RL(GFT类) · 相关性中

**原始论文**：https://arxiv.org/abs/2605.15012

## 一眼看懂

> 一句话导读：RL 解不出难题时常拿 SFT 示范来补,但精选示范太贵;本文发现只要 128 条随机示范就够,关键是用 semi-online DPO 这个 loss——它的梯度天生同时带"监督 + on-policy + 自动衰减"三样东西,刚好防住在极少数据上过拟合。

- 🟦 TL;DR：demonstration-guided RLVR(在 RL 采样失败时,拿 SFT 示范来补充外部知识)很有效,但代价是要数 K~50K 条精选的 SFT 数据,太贵。
  - FEST 证明:只用 **128 条随机**(非精选)抽自 SFT 数据集的示范,就能显著超过纯 RLVR。
  - 关键是把"用好极少示范"拆成三要素:监督信号 + on-policy 信号 + 防过拟合的衰减权重。
  - 而 **semi-online DPO 的梯度恰好天然同时含这三项**(做法:把示范当 preferred \(y^+\)、把 agent rollout 当 non-preferred \(y^-\))。【原文 Abstract, §1, §3.1, Eq.4】
- 最巧的一步：**梯度分解 Eq.4**——\(\nabla_\theta L_E=-\beta\,\mathbb{E}\big[\sigma(\beta(r^--r^+))\cdot(\nabla\log\pi_\theta(y^+\!\mid x)-\nabla\log\pi_\theta(y^-\!\mid x))\big]\),式中三项依次对应监督学习、on-policy、衰减权重。
  - 抽掉这一步会怎样:不用 semi-online DPO 的这种特殊梯度结构,而是朴素地把 RL 也加到 gold few-shot 上做 SFT+RL(即所谓"-G"变体)→ HPT-G 平均只有 32.02(vs HPT 38.75)、ReLIFT-G 38.19(vs ReLIFT 40.51),而且训练中途会骤崩(Appendix D.2)。
  - 所以命门是:"用 DPO 梯度天然提供的衰减权重,来防止在极少 gold 数据上过拟合"。【原文 §3.2, §4.1 Table 2】

## 为什么做

> 一句话导读：RL 在难题上常常一批 rollout 全错、拿不到学习信号,得靠 SFT 示范补;但高质量示范贵得离谱,本文要把它压到 128 条随机数据还能稳定提分。

- 研究背景：RLVR(可验证奖励 RL)在数学/代码上很成功,但难题上样本效率低——一批 rollout 全错→advantage=0→没有学习信号;DAPO/CISPO 靠重复采样直到成功,平均要多花 3× 算力。于是 demonstration-guided RL / unified post-training(LUFFY/SRFT/HPT/ReLIFT/MIFO/CHORD 等)的思路是:在 RL 失败时引入 SFT,提供外部知识【原文 §1】。
- 解决的具体痛点：SFT 数据太贵——高质量的长链推理示范需要精心策划(例如 HLE 那 2500 题动用了 1000 名博士),从模型蒸馏又涉及合法性/API 成本/model collapse 风险;相比之下"只有答案、没有推理过程"的 RL 数据很易得(可从在线论坛挖)。而现有方法都用满量精选数据(SuperRL~50K、LUFFY/SRFT 46K、HPT 10K、ReLIFT 8.6K、MIFO 6.4K、CHORD 5K、SASR 2K),且多数还非随机【原文 §1, Table 1】。
- 相关工作 & 各自不足(来龙去脉 + 精确短板):
  - ① **多目标融合(SFT+RL 同时优化)**:LUFFY(NeurIPS25,把 off-policy expert 轨迹混进 GRPO)、SRFT(单阶段融合 SFT 与 RL)、CHORD(动态加权 SFT 项)——它们 SFT+RL 同步优化,但**数据需求大**(46K/5K);FEST 把这压到 128。
  - ② **RL-SFT 切换**:HPT(Hybrid Post-Training,RL 失败时切到 SFT、末期把 SFT 比例手动降到 2%)、ReLIFT(RL 与 SFT 交替)、MIFO、SuperRL、DyME——仍需数 K 精选数据,且假设"可以按需对失败题补示范"(在 few-shot 场景里不成立);FEST 的"衰减权重"则把 HPT 那个手动降 SFT 比例的动作内生化了。
  - ③ **专门 few-shot SFT**:LIMOv2(靠精心策划的 pipeline 选少量高价值样本)——FEST 不假设这种精选 pipeline,它的 \(D_E\) 可以就是随机一批。
  - ④ **底层范式**(FEST 缝合的几块):
    - semi-online DPO(Guo/Lanchantin,介于 offline 与 online DPO 之间,preferred 固定、non-preferred 在线采);
    - SPIN(self-play,把"专家 vs 自生成"做成对抗);
    - 负奖励 RL(Zhu et al.,证明纯负奖励能把概率质量重分配到其他可行解、从而防过拟合)。
    - FEST 把这些缝合成一个"few-shot 专用"的目标。
  - 共同不足:现有 demonstration-guided 方法都**没法用"随机 + 极少(128)"的数据**稳定提分;而且本文 §4.1 还指出,pure RL 在 lr 调优后其实已经是很强的 baseline(与全量 ReLIFT 持平),这削弱了部分方法的卖点【原文 §1, §3.1, Table 1】。
- 动机链:SFT 数据贵 → 想把它压到 128 条且随机 → 但 few-shot 这种极少数据有三大挑战(没有按需数据 / 语义覆盖有限 / 反复多 epoch 会过拟合)→ 据此拆出"用好 few-shot"必须同时满足的三要素 → 再找一个天然同时具备这三者的损失(就是 semi-online DPO)。
- 与最近邻工作的Δ(精确差异):最近邻是 HPT/ReLIFT(RL-SFT 切换式的统一后训练)。Δ 有三点:
  - (1) 数据量从数 K 降到 128、且随机;
  - (2) 用 semi-online DPO 替代"SFT 分支",因为它的梯度自带衰减权重→不会像 HPT-G/ReLIFT-G 那样在极少 gold 数据上做 RL 时崩;
  - (3) 进一步证明 semi-online DPO 在形式上等价于"REINFORCE(负奖励)+ 加权 SFT",从而把 DPO 纳入 HPT 的统一框架(Remark 3.3,算是对 HPT 的扩展)。

## 怎么做（细到可复现）

> 一句话导读：整套训练用两路数据(128 条示范 + 大批只有答案的 RL 题)、两支 loss(示范走 semi-online DPO、答案题走 GRPO)。重点看 §2 那个梯度公式——它一式拆出三件套,正是本文敢用 128 条的全部底气;§3/§4 是把它做稳的两个补丁。

### 0. 问题设定（§3.1）
两套数据并用:few-shot SFT 集 \(D_E\)(128 条含专家长链推理的轨迹)+ 大规模 answer-only RL 集 \(D_I\)(只有答案,供 verifier 给奖励)。对应"三大挑战",有三要素必须同时满足:
- **监督学习**——这是 RLVR 二值奖励之外唯一的外部知识来源;
- **on-policy 学习**——让模型拿自己的 rollout 去对比示范,缓解 exposure bias、扩大极少题的学习面;
- **自适应衰减权重**——早期重学 \(D_E\),之后随 \(D_I\) 的 RLVR 信号变主导而自动降权,防过拟合。

### 1. 总损失与两支（§3.2，Eq.3）
总损失就是两支加权相加:\(\displaystyle L=c\cdot L_E+L_I,\qquad c>0\ \text{为常数系数}.\)
- **\(D_E\) 支:semi-online DPO**(把示范 \(y^+\) 当 preferred,把当前 rollout \(y^-\sim\pi_{\theta_{\text{old}}}\) 当 non-preferred):
\(\displaystyle L_E=-\,\mathbb{E}_{(x,y^+)\sim D_E,\ y^-\sim\pi_{\theta_{\text{old}}}(\cdot\mid x)}\big[\log\sigma(\beta r^+-\beta r^-)\big],\)
其中 \(r^+=\log\dfrac{\pi_\theta(y^+\mid x)}{\pi_{\text{ref}}(y^+\mid x)},\ r^-=\log\dfrac{\pi_\theta(y^-\mid x)}{\pi_{\text{ref}}(y^-\mid x)}\),\(\sigma\) 是 sigmoid。
- **\(D_I\) 支:GRPO**(沿 HPT/Dr.GRPO 省掉 KL 与 advantage std,带 DAPO 式的非对称 clip \(1-\epsilon_1,1+\epsilon_2\)):
\(\displaystyle L_I=\mathbb{E}_{x\sim D_I,\ y\sim\pi_{\theta_{\text{old}}}}\!\Big[-\tfrac{1}{nM}\textstyle\sum_{i=1}^{n}\sum_{j=1}^{|y_i|}\min\big(\rho_{i,j}A_i,\ \mathrm{clip}(\rho_{i,j},1-\epsilon_1,1+\epsilon_2)A_i\big)\Big],\quad \rho_{i,j}=\tfrac{\pi_\theta(y_{i,j}\mid x,y_{i,<j})}{\pi_{\theta_{\text{old}}}(y_{i,j}\mid x,y_{i,<j})}.\)

### 2. 为什么选 semi-online DPO：梯度三要素（§3.2，Eq.4）
把 \(L_E\) 的梯度展开,正好一式拆出三件套:
\(\displaystyle \nabla_\theta L_E=-\beta\,\mathbb{E}_{(x,y^+)\sim D_E,\ y^-\sim\pi_{\theta_{\text{old}}}}\Big[\underbrace{\sigma(\beta(r^--r^+))}_{\text{衰减权重}}\cdot\big(\underbrace{\nabla\log\pi_\theta(y^+\mid x)}_{\text{监督}}-\underbrace{\nabla\log\pi_\theta(y^-\mid x)}_{\text{on-policy}}\big)\Big].\)
关键在那个衰减权重 \(\sigma(\beta(r^--r^+))\):随着训练把示范概率学高(即 \(r^+\) 上升),它会自动趋小→自然降低对 few-shot 的学习强度(HPT 是靠手动降到 2%,FEST 是梯度内生)。另外 SPIN 已证明这种范式 ≈ 对抗训练(判别器去分 \(r^+/r^-\)、策略当生成器,有闭式解,见 Appendix B.2)。

### 3. 自适应 β：按可解性三档（§3.2，Eq.5）
对一 batch 内的 \(n\) 个 rollout,按二值奖励把每个 pair \((x,y^-_i)\) 的 \(\beta\) 分三档设:
\(\displaystyle \beta(x,y^-_i)=\begin{cases}\beta_1,&\forall j,\ r(x,y^-_j)=0\ (\text{全错}),\\ \beta_2,&r(x,y^-_i)=0\ \text{且}\ \exists j,\ r(x,y^-_j)=1\ (\text{RLVR-可解但本条错}),\\ \beta_3,&r(x,y^-_i)=1\ (\text{本条正确}).\end{cases}\)
直觉:全错的难题最该强学示范(\(\beta_1\) 给最大引导),已经做对的题就该容忍它偏离示范。\(\beta_1,\beta_2,\beta_3\) 都是常数(启发式,Appendix D.3 调参)。Remark 3.2 提醒:长链推理需 \(\beta\in[0.001,0.1]\),远小于标准 DPO 的 0.1–0.2——因为序列长、log-ratio 差异大。

### 4. FEST-GRPO 变体：治梯度幅度失配（§3.3）
- **问题**:\(L_E\)(DPO) 是**序列级**的(log-sigmoid 内是整条响应的联合概率),\(L_I\)(GRPO) 是**token 级**的(逐 token clip),两者梯度幅度差很大,得做 exhaustive 调 \(c\)。
- **解法**:把 Eq.4 里的"衰减权重 + on-policy 项" \(\mathbb{E}[\beta\sigma(\beta(r^--r^+))\nabla\log\pi_\theta(y^-\mid x)]\) 拿去对照 REINFORCE 梯度,发现它 ≡ "负奖励 REINFORCE"(奖励是 \(-\beta\sigma(\beta(r^--r^+))<0\));而监督项 ≡ "正权重 \(\beta\sigma(\beta(r^--r^+))>0\) 的加权 SFT"。于是得到等价关系:
\(\displaystyle \textbf{Semi-online DPO}\ \approx\ \textbf{REINFORCE(负奖励)}\ +\ \textbf{加权 SFT}.\)
- **据此改**:**把其中的 REINFORCE 部分换成 GRPO** → 就是 FEST-GRPO(保留 \(L_I\),把 DPO 式的 \(L_E\) 换成"加权 SFT + 对 \(D_E\) 的 GRPO")。这既消除了失配,这一等价又把 DPO 纳入了 HPT 的统一框架(Remark 3.3)。负奖励 RL 的作用(Zhu et al.[108]):把概率质量重分配到其他可行解、防过拟合、促鲁棒探索(Remark 3.4)。

### 数据流动 / 关键超参（§4 Training Recipe）
每步:从 \(D_E\) 取 128 题(含 expert \(y^+\) + 在线采 \(y^-\))→ 算 \(L_E\);从 \(D_I\) 取 128 题 × \(n=8\) rollout → 算 \(L_I\);合并 \(L=cL_E+L_I\) 更新。其余设置:模型 Qwen2.5-Math-1.5B;数据 OpenR1-Math-46K-8192(随机抽 128 作 \(D_E\)、其余作 \(D_I\));600 步、2×GH200(96GB);温度 1.0、max len 8192;AdamW、cosine lr 1e-5→5e-6;global batch 128 题(\(D_E,D_I\) 各 128)、mini-batch 512 rollouts;汇报第 600 步的结果(沿 ReLIFT)。

### 逐组件必要性
- **监督信号（\(L_E\) 的 \(y^+\) 项）**：去掉→退回纯 RL（39.79 vs FEST-DPO 41.98）【§3.1, §4.1】。
- **on-policy 信号（\(y^-\) 项）**：缓解 exposure bias、起对抗训练作用、扩大极少题学习面【§3.1, §3.2】。
- **衰减权重 \(\sigma(\beta(r^--r^+))\)**：防过拟合。反向消融：去掉它（-G 变体直接 SFT+RL）→ HPT-G/ReLIFT-G 中途骤降【§4.1, App D.2】。
- **自适应 β（Eq.5）**：细粒度控不同来源数据学习强度；启发式，App D.3 称对 \(\beta\) 较鲁棒、最佳 \(\beta\in[0.001,0.1]\)〔待核：D.3 未逐页核〕。
- **FEST-GRPO（§3.3）**：治序列级 vs token 级梯度失配；消融 42.36 略高于 FEST-DPO 41.98【§3.3, Table 2】。

## 靠不靠谱
- 实验与证据：
  - **关键数字(Table 2,128-shot,Avg@8)**:FEST-DPO **41.98**、FEST-GRPO **42.36**,都超过 vanilla RL(39.79)、RL-G(40.55),以及所有 128-shot 基线(LUFFY 37.90、CHORD-φ 37.56、HPT 38.75、ReLIFT 40.51);还匹配/超过了用**全量数据**的 SRFT(35.05)。它是这种稀疏数据条件下**唯一**显著超过纯 RL 的方法。对照之下 -G 变体直接崩:HPT-G 32.02、ReLIFT-G 38.19【§4.1, Table 2】。
  - **Pass@8(Table 3,看探索潜力)**:FEST-DPO 60.08 / FEST-GRPO 61.06,对比 RL 59.67、RL-G 54.84。注意 RL-G 虽然 nominal 准确率不低,但 Pass@8 最低(说明它过拟合、多样性差、上限低);而 FEST 保持了最高的探索潜力【§4.1, Table 3】。
  - **shots 缩放(Fig.2)**:64/128/256/512 shots 都能工作;其中 FEST-GRPO 在极少(64)时更稳,FEST-DPO 则随数据增长扩展性更好(512 时已追平用全量 46K 的 HPT)【§4.2】。
  - **跨 \(D_E\) 鲁棒(Table 4)**:换两个额外的随机 128-split,以及 LIMOv2-8192(257 例),都能稳定提分【§4.3】。
  - **baseline 公平吗**:较公平——都在同样的 128-shot 约束下,多数基线还是自行复现的;但 SRFT 用了它的官方 ckpt(非开源、不兼容 HPT 代码)、MIFO 直接取了论文数字(未开源),这两项可比性偏弱。论文也自承"纯 RL 在 lr 调优(1e-6→5e-6)后已是强 baseline、与全量 ReLIFT 持平"。
  - **"看着强但没回答核心问题"**:绝对增益其实不大(比 RL 高 +2.2~2.6 分),真正卖点是"数据 128 vs 数 K"的差距;而且"128"只是沿用前作的 batch size(一个 epoch 恰好一步),并非对"最少需要多少示范"做系统搜索(§4.2 的 shots 缩放也有限)。
- 假设与失效边界：
  - 【原文】单轮交互、序列级奖励（§2 脚注 1）；省略 KL 与 advantage std（沿 HPT/Dr.GRPO，§2）。
  - 【原文】Remark 3.2：长链推理需 \(\beta=0.001\text{–}0.1\)（远小于标准 DPO）。
  - 【原文】Remark 3.1：承认 DPO "难翻转偏好/拒答主导"的批评，但辩称本场景是"向 expert 轨迹正则"（类 online TD3+BC），故无害——偏定性辩护。
  - 【推断】仅单模型 Qwen2.5-Math-1.5B、单数据源、纯数学，规模小（自标 Ongoing Work），跨模型/跨域泛化未知（依据：§4 全部实验在 1.5B + OpenR1-Math）。
  - 【推断】自适应 β 三档是启发式；FEST-DPO 仍需调系数 \(c\)（FEST-GRPO 才缓解 gradient mismatch）（依据：§3.2 \(c\) 为常数、§3.3 称需 exhaustive tuning \(c\)）。
- 祛魅总结：
  - 真贡献【推断】：梯度分解（Eq.4 三要素）+ "semi-online DPO ≈ 负奖励 REINFORCE + 加权 SFT"的形式等价，是论文最扎实部分——把"为什么 128 条随机就够"从经验巧合提升为可解释机制，并把 DPO 纳入 HPT 统一框架。
  - 包装/高估【推断】：标题强调"randomly selected few-shot"很抓眼，但绝对分数增益有限（~+2.5），"128"非系统最优；"匹配全量数据"主要因对比对象 SRFT（35.05）本身偏弱。
  - 低估【推断】：Pass@8 视角（FEST 保持高上限、RL-G 上限塌缩）揭示了"在少量 gold 数据上做硬 SFT/RL 会牺牲未来 RL 潜力"这一更普适的训练学，但被"数据效率"叙事盖过。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=两路——\(D_I\) 的可验证奖励 advantage（GRPO）+ \(D_E\) 的 DPO 偏好信号（示范 \(y^+\!>\!\)rollout \(y^-\)，经 \(\sigma(\beta(r^--r^+))\) 加权）｜**改什么**=策略全参（无 critic，semi-online DPO 用 ref policy 算 log-ratio）｜**何时改**=训练期在线，每步同时算 \(L_E\)（few-shot）+\(L_I\)（answer-only），衰减权重随训练自动降 few-shot 影响｜**免梯度?**=否（GRPO+DPO 都是梯度法）｜**记忆-技能生命周期**=无显式记忆/技能库（128 条 few-shot 是固定外部示范集，知识固化进参数）｜**防遗忘机制**=衰减权重（\(\sigma(\beta(r^--r^+))\) 内生 + adaptive \(\beta\) 三档）防对 few-shot 过拟合、保探索多样性（Pass@8 不塌）。
- ⑦ 开源代码+框架/harness：github.com/KaiYan289/FEST。核心 `ternary_dpo/`（基于 **VeRL**，含 verl、setup.py）、`examples/math-1.5b-v3`、`dataset/`、`utils/`、`eval_by_question_results/`。框架=**VeRL**；GRPO 为主 RL 框架，few-shot 用 semi-online DPO；FEST-GRPO 变体把 DPO 的 online 部分换成 GRPO。多数 baseline 自行实现（SRFT/MIFO 例外）。
- 💰 资源/成本与可扩展性：**2×NVIDIA GH200(96GB)，600 步**；\(n=8\) rollout/题、温度 1.0、max len 8192；AdamW、cosine lr 1e-5→5e-6；global batch 128 题、mini-batch 512 rollouts。few-shot 仅 128 条（核心卖点：SFT 数据成本极低）。仅 1.5B 规模验证，更大模型成本未测【原文 §4 Training Recipe】。
- 🎯 对"探索-巩固"对标：**可借组件 + 部分竞品**。一句判定:FEST 的"衰减权重防过拟合 + Pass@8 不塌"直接对应"巩固/固化时不损害未来探索";它"少量 gold 示范当 preferred、自生成当 non-preferred"的设计,也松散对应"teacher 稀疏脚手架 + on-policy 自选"。
  - 可借组件:
    - ① **\(\sigma(\beta(r^--r^+))\) 内生衰减权重**——把"固化进参数"做成随训练自动降权,正是"巩固但不遗忘/不过拟合"的一种轻量实现;
    - ② **adaptive β 按可解性分档**——对应"在学生走不通的题上更强地接管 teacher 示范"(\(\beta_1\) 在全错时给最强引导),与 path-recovery 单点接管的精神相通;
    - ③ **Pass@8 监控**——可当作"巩固是否损害了探索上限"的诊断指标;
    - ④ "DPO ≈ 负奖励 REINFORCE + 加权 SFT"的等价——为"把示范监督与 RL 统一在一个梯度里"提供了干净模板。
  - 缺口:无 MTP/前瞻、无路径级 step 信用;而且示范是固定的外部集、而非"学生自选可走通的开头"(\(y^+\) 是 expert 轨迹,不是学生自己的成功开头),所以对"探索/选路"的支撑较弱。依据:§3.1 三要素、§4.1 Pass@8 对比。
- 🔭 开放问题/未来方向：【原文】shots 缩放只做了有限搜索（§4.2），"最少需多少示范"未系统回答；附录 D 给完整超参分析〔待核〕。【推断】把"固定 128 expert 示范"换成"学生自己历史里走通过的成功开头"作 \(y^+\)，即可从"教师示范引导"转向"自蒸馏/自巩固"，更贴"探索-巩固"（依据：当前 \(y^+\)=expert 轨迹是外部的，可替换为 on-policy 成功轨迹）；自适应 β 可与"高熵关键步/path-recovery 接管点"结合，在关键步动态增强示范权重；跨模型规模与非数学域的验证（当前仅 1.5B 数学）。

读到PDF? 是（PyMuPDF 全文 25 页；正文 §1–4 + Eq.3/4/5 + Table 2/3/4 全核，附录 B/C/D 理论与超参未逐页核）｜L线 L2（统一 SFT-RL，GFT 类）｜对标结论 可借组件（内生衰减权重防过拟合、adaptive β 分档接管、Pass@8 诊断、DPO≈负奖励REINFORCE+加权SFT 模板）+ 部分竞品；缺 MTP/路径级、\(y^+\) 为外部示范非自选开头｜残留待核 1（附录 D.3 β 超参分析未逐页核）

fest | FEST: Boosting RLVR via Randomly Selected Few-Shot Guidance | UIUC（Kai Yan、Alexander G. Schwing、Yu-Xiong Wang） | 2026-05-14 (arXiv:2605.15012 v1，标注 Preprint / Ongoing Work) | L2 统一SFT-RL(GFT类) · 相关性中

**原始论文**：https://arxiv.org/abs/2605.15012

## 一眼看懂
- 🟦 TL;DR：demonstration-guided RLVR（RL 采样失败时拿 SFT 示范补外部知识）很有效，但要数 K~50K 条精选 SFT 数据，太贵。FEST 证明：只用 **128 条随机**（非精选）抽自 SFT 数据集的示范就能显著超过纯 RLVR——关键是把"用好极少示范"拆成三要素（监督信号 + on-policy 信号 + 防过拟合的衰减权重），而 **semi-online DPO 的梯度恰好天然同时含这三项**（把示范当 preferred y+、agent rollout 当 non-preferred y−）【原文 Abstract, §1, §3.1, Eq.4】。
- 最巧的一步：**梯度分解 Eq.4**——∇L_E = −β·E[σ(β(r⁻−r⁺))·(∇logπ(y⁺)−∇logπ(y⁻))]，三项依次对应监督学习、on-policy、衰减权重。抽掉这一步（即不用 semi-online DPO 的特殊梯度结构，而朴素地把 RL 也加到 gold few-shot 上做 SFT+RL，即"-G"变体）→ HPT-G/ReLIFT-G 在 128-shot 下骤降（HPT-G 平均仅 32.02 vs HPT 38.75，且训练中途崩）。所以"用 DPO 梯度天然提供衰减权重来防极少 gold 数据上的过拟合"是命门【原文 §3.2, §4.1 Table 2】。

## 为什么做
- 研究背景：RLVR（可验证奖励 RL）在数学/代码上很成功，但难题上样本效率低（一批 rollout 全错→advantage=0→无学习信号；DAPO/CISPO 靠重复采样直到成功，平均增 3× 算力）。demonstration-guided RL / unified post-training（LUFFY/SRFT/HPT/ReLIFT/MIFO/CHORD 等）在 RL 失败时引入 SFT 提供外部知识【原文 §1】。
- 解决的具体痛点：SFT 数据昂贵——高质量长链推理示范需精心策划（如 HLE 2500 题动用 1000 名博士），从模型蒸馏又涉合法性/API 成本/model collapse 风险；而"只有答案、无推理过程"的 RL 数据易得（在线论坛挖）。现有方法都用满量精选数据（SuperRL~50K、LUFFY/SRFT 46K、HPT 10K、ReLIFT 8.6K、MIFO 6.4K、CHORD 5K、SASR 2K），且多非随机【原文 §1, Table 1】。
- 相关工作 & 各自不足：① 多目标融合（LUFFY/SRFT/CHORD）——SFT+RL 同时优化，数据需求大；② RL-SFT 切换（HPT/ReLIFT/MIFO/SuperRL）——RL 失败时切 SFT，仍需数 K 精选数据且假设可"按需"补示范；③ 专门 few-shot（LIMOv2）——靠精心策划 pipeline，FEST 不假设这一点。共同不足：都不能用"随机 + 极少"数据【原文 §1, §3.1, Table 1】。
- 动机链：SFT 数据贵 → 想把它压到 128 条且随机 → 但 few-shot 极少数据有三大挑战（无按需数据 / 语义覆盖有限 / 反复多 epoch 过拟合）→ 拆出"用好 few-shot"必须同时满足的三要素 → 找一个天然同时具备三者的损失（semi-online DPO）。
- 与最近邻工作的Δ：最近邻是 HPT/ReLIFT（RL-SFT 切换的统一后训练）。Δ 在于 (1) 数据量从数 K 降到 128 且随机；(2) 用 semi-online DPO 替代"SFT 分支"，因其梯度自带衰减权重→不会像 HPT-G/ReLIFT-G 那样在极少 gold 数据上做 RL 时崩。为什么有用：Eq.4 的 σ(β(r⁻−r⁺)) 权重随训练自动降低对 few-shot 的学习强度，防过拟合（HPT 是手动把 SFT 比例降到 2%，FEST 是梯度内生）【原文 §3.1, §3.2, Remark 3.1】。

## 怎么做 + 靠不靠谱
- 方法流水线（4 步）：① 数据切分：OpenR1-Math-46K 随机抽 128 题作 few-shot D_E（含 expert 推理轨迹），其余作答案-only D_I；② D_I 上跑标准 GRPO（L_I，沿用 HPT/Dr.GRPO 省 KL 与 std）；③ D_E 上跑 **semi-online DPO**（L_E）：示范=preferred y⁺、当前 rollout=non-preferred y⁻；④ 总损失 L = c·L_E + L_I，c 为常数系数【原文 §3.2, Eq.3】。
- 逐组件必要性：
  - **监督信号（L_E 的 y⁺ 项）**：RLVR 二值奖励之外唯一的外部知识来源。去掉→退回纯 RL（39.79，FEST-DPO 41.98 更高）【原文 §3.1, §4.1】。
  - **on-policy 信号（L_E 的 y⁻ 项）**：让模型拿自己 rollout 与示范对比，缓解 exposure bias、起对抗训练作用、扩大极少题目的学习面。SPIN 证明此范式等价对抗训练【原文 §3.1, §3.2】。
  - **衰减权重（σ(β(r⁻−r⁺)) 项）**：防过拟合。反向消融：去掉它（-G 变体直接 SFT+RL）→ HPT-G/ReLIFT-G 中途骤降（§4.1 训练曲线，Appendix D.2）【原文 §3.1, §4.1】。
  - **自适应 β（Eq.5，三档）**：按可解性分 β1（一组全错）/β2（RLVR-可解但本条错）/β3（本条正确），细粒度控制不同来源数据学习强度。是启发式，有 Appendix D.3 超参分析〔待核：附录 D.3 未逐页核〕。
  - **FEST-GRPO 变体（§3.3）**：治 DPO(序列级) vs GRPO(token 级)的 gradient magnitude mismatch。论文证明 semi-online DPO ≈ "REINFORCE(负奖励) + 加权 SFT"，于是把 REINFORCE 部分换成 GRPO 消除失配；这一等价也把 DPO 纳入 HPT 的 unified 框架（Appendix B.3）。有消融：FEST-GRPO 42.36 略高于 FEST-DPO 41.98【原文 §3.3, Table 2, Remark 3.3】。
- 关键机制/公式（直觉）：semi-online DPO 的妙处不在 DPO 本身，而在它对参数求梯度后**正好分解出三要素**（Eq.4）；其中 on-policy + 衰减权重部分功能等价于"负奖励 REINFORCE"（−β·σ(β(r⁻−r⁺))<0），监督部分等价于"正权重加权 SFT"。负奖励 RL 的作用（Zhu et al. [108]）：把概率质量重分配到其他可行解，防过拟合、促鲁棒探索【原文 §3.2, §3.3, Remark 3.4】。
- 实验与证据：
  - **数据/模型**：Qwen2.5-Math-1.5B，OpenR1-Math-46K-8192，128 随机 D_E、其余 D_I；600 步、2×GH200(96GB)；n=8 rollout/题、温度 1.0、max len 8192；AdamW、cosine lr 1e-5→5e-6；global batch 128 题、mini-batch 512 rollouts；报第 600 步（沿用 ReLIFT）。评测 6 个数学基准：AIME25/AMC23/AIME24/MATH-500/OlympiadBench/Minerva，报 Avg@8(均±标准差) 与 Pass@8【原文 §4 Training Recipe, Eval】。
  - **关键数字（Table 2，128-shot）**：FEST-DPO 平均 **41.98**、FEST-GRPO **42.36**，均超 vanilla RL（39.79）、RL-G（40.55）及所有 128-shot 基线（LUFFY 37.90、CHORD-φ 37.56、HPT 38.75、ReLIFT 40.51），且匹配/超用**全量数据**的 SRFT（35.05）；是该稀疏数据条件下**唯一**显著超过纯 RL 的方法。-G 变体崩：HPT-G 32.02、ReLIFT-G 38.19【原文 §4.1, Table 2】。
  - **Pass@8（Table 3，探索潜力）**：FEST-DPO 60.08 / FEST-GRPO 更高 vs RL 59.67 vs RL-G 54.84——证明 RL-G 虽 nominal 准确率不低但 Pass@8 低（过拟合、多样性差、上限低），FEST 保持高探索潜力【原文 §4.1, Table 3】。
  - **baseline 公平吗**：较公平——同 128-shot 约束、多数基线自行复现；但 SRFT 用其官方 ckpt（非开源、不兼容 HPT 代码）、MIFO 直接取其论文数字（未开源），这两项可比性弱。注意：论文自承"纯 RL 在 lr 调优(1e-6→5e-6)后是强 baseline、与全量 ReLIFT 持平"，削弱了部分 demonstration-guided 方法的卖点。
  - **"看着强但没回答核心问题"**：绝对增益不大（+2.2~2.6 分 over RL），真正卖点是"数据 128 vs 数 K"。且"128"这数字是沿用前作 batch size（一 epoch 恰好一步），非对"最少需多少示范"的系统搜索（§4.2 有 shots 缩放但有限）。
- 假设与失效边界：
  - 【原文】单轮交互、序列级奖励（§2 脚注 1）；省略 KL 与 advantage std（沿 HPT/Dr.GRPO，§2）。
  - 【原文】Remark 3.2：长链推理需 β=0.001–0.1（远小于标准 DPO 的 0.1–0.2），因序列长、log-ratio 差异大。
  - 【原文】Remark 3.1：承认 DPO "难翻转偏好/拒答主导"的批评，但辩称本场景是"向 expert 轨迹正则"（类 online TD3+BC），故无害——偏定性辩护。
  - 【推断】仅单模型 Qwen2.5-Math-1.5B、单数据源、纯数学，规模小（自标 Ongoing Work），跨模型/跨域泛化未知（依据：§4 全部实验在 1.5B + OpenR1-Math）。
  - 【推断】自适应 β 三档划分是启发式、依赖经验调参；FEST-DPO 仍需调系数 c（FEST-GRPO 才缓解 gradient mismatch）（依据：§3.2 c 为常数、§3.3 称需 exhaustive tuning c）。
- 祛魅总结：
  - 真贡献【推断】：梯度分解（Eq.4 三要素）+ "semi-online DPO ≈ 负奖励 REINFORCE + 加权 SFT"的形式等价，是论文最扎实部分——把"为什么 128 条随机就够"从经验巧合提升为可解释机制，并把 DPO 纳入 HPT 统一框架。
  - 包装/高估【推断】：标题强调"randomly selected few-shot"很抓眼，但绝对分数增益有限（~+2.5），且"128"非系统最优；"匹配全量数据"主要是对比对象 SRFT（35.05）本身偏弱。
  - 低估【推断】：Pass@8 视角（FEST 保持高探索上限、RL-G 上限塌缩）其实揭示了"在少量 gold 数据上做硬 SFT/RL 会牺牲未来 RL 潜力"这一更普适的训练学，但被"数据效率"叙事盖过。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=两路——D_I 的可验证奖励 advantage（GRPO）+ D_E 的 DPO 偏好信号（示范 y⁺ > rollout y⁻，经 σ(β(r⁻−r⁺)) 加权）｜**改什么**=策略全参（无 critic，semi-online DPO 用 ref policy 算 log-ratio）｜**何时改**=训练期在线，每步同时算 L_E（few-shot）+L_I（answer-only），衰减权重随训练自动降 few-shot 影响｜**免梯度?**=否（GRPO+DPO 都是梯度法）｜**记忆-技能生命周期**=无显式记忆/技能库（128 条 few-shot 是固定外部示范集，知识固化进参数）｜**防遗忘机制**=衰减权重（σ(β(r⁻−r⁺)) 内生 + adaptive β 三档）防对 few-shot 过拟合、保探索多样性（Pass@8 不塌）。
- ⑦ 开源代码+框架/harness：github.com/KaiYan289/FEST。核心 `ternary_dpo/`（基于 **VeRL**，含 verl、setup.py）、`examples/math-1.5b-v3`、`dataset/`、`utils/`、`eval_by_question_results/`。框架=**VeRL**；GRPO 为主 RL 框架，few-shot 用 semi-online DPO；FEST-GRPO 变体把 DPO 的 online 部分换成 GRPO。多数 baseline 自行实现（SRFT/MIFO 例外）。
- 💰 资源/成本与可扩展性：**2×NVIDIA GH200(96GB)，600 步**；n=8 rollout/题、温度 1.0、max len 8192；AdamW、cosine lr 1e-5→5e-6；global batch 128 题、mini-batch 512 rollouts。few-shot 仅 128 条（核心卖点：SFT 数据成本极低）。仅 1.5B 规模验证，更大模型成本未测【原文 §4 Training Recipe】。
- 🎯 对"探索-巩固"对标：**可借组件 + 部分竞品**。一句判定：FEST 的"衰减权重防过拟合 + Pass@8 不塌"直接对应"巩固/固化时不损害未来探索"，其"少量 gold 示范当 preferred、自生成当 non-preferred"也松散对应"teacher 稀疏脚手架 + on-policy 自选"。可借组件：① **σ(β(r⁻−r⁺)) 内生衰减权重**——把"固化进参数"做成随训练自动降权，正是"巩固但不遗忘/不过拟合"的轻量实现；② **adaptive β 按可解性分档**对应"在学生走不通的题上更强地接管 teacher 示范"（β1 全错时最强引导），与 path-recovery 单点接管的精神相通；③ **Pass@8 监控**可作"巩固是否损害探索上限"的诊断指标。缺口：无 MTP/前瞻、无路径级 step 信用、示范是固定外部集而非"学生自选可走通开头"（y⁺ 是 expert 轨迹，非学生自己的成功开头），与"探索/选路"支撑较弱。依据：§3.1 三要素、§4.1 Pass@8 对比。
- 🔭 开放问题/未来方向：【原文】shots 缩放只做了有限搜索（§4.2），"最少需多少示范"未系统回答；附录 D 给完整超参分析〔待核〕。【推断】把"固定 128 expert 示范"换成"学生自己历史里走通过的成功开头"作 y⁺，即可从"教师示范引导"转向"自蒸馏/自巩固"，更贴"探索-巩固"（依据：当前 y⁺=expert 轨迹是外部的，可替换为 on-policy 成功轨迹）；自适应 β 可与"高熵关键步/path-recovery 接管点"结合，在关键步动态增强示范权重；跨模型规模与非数学域的验证（当前仅 1.5B 数学）。

读到PDF? 是（正文 §1–4 + Table 2/3 已读，附录 B/C/D 理论与超参未逐页核）｜L线 L2（统一 SFT-RL，GFT 类）｜对标结论 可借组件（内生衰减权重防过拟合、adaptive β 分档接管、Pass@8 诊断）+ 部分竞品；缺 MTP/路径级、y⁺ 为外部示范非自选开头｜残留待核 2（附录 D.3 超参分析、附录 B 理论等价细节未逐页核）

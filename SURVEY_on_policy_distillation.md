# On-Policy Distillation 及相关后训练技术综述（mtp_opd / TSRD 项目文献基座）

> 本文档由 117 篇逐篇深读分析（`analysis/*.md`）全量合成、二次审查并加强重写而成。围绕项目核心 **TSRD（Teacher-Scaffolded Reasoning Distillation）= 教路径选择(path-selection) + 教路径恢复(path-recovery)，并以 MTP 作为前瞻探针(foresight probe)；mtp_opd = MTP + OPD** 这一主线组织。
> 最近更新：2026-06-03。语气保持中立、客观；所有原始逐篇内容以 `cat` 内联、未作转述改写。

---

## 0. 概述

### 0.1 任务与三线主题

本综述服务于 mtp_opd / TSRD 项目，目标是把"On-Policy Distillation（OPD，在线策略蒸馏）"放回它所处的后训练技术坐标系中系统梳理。围绕项目主线，文献被组织为四条相互交织的主题线（一篇可跨多线，归入其最主要的一线）：

- **T1 — OPD 核心 / 自蒸馏**：白盒/黑盒 token 级在线策略蒸馏、自蒸馏（OPSD/SDPO 家族）、特权上下文蒸馏、蒸馏目标函数（KL/RKL/skew-KL/对比式）、teacher consistency、置信度校准、token 级动力学等。这是项目最直接命中的一线。
- **T2 — Tool-Agent / 多轮**：工具调用 / 检索 / 深度搜索 agent 的蒸馏与多轮 RL、step 级信用分配、自模仿与课程化探索。对应 TSRD 在 agentic 场景下的落地与对照。
- **T3 — GFT 统一 SFT-RL & GRPO-RLVR**：把 SFT 与 RL 统一到同一策略梯度视角的 GFT 类工作（CHORD/HPT/LUFFY/Prefix-RFT/SRFT/PSFT/ASFT/DFT 等），以及 GRPO/RLVR 算法谱系与训练机理（DAPO/Dr.GRPO/GSPO/GMPO/熵机制/能力边界批判等）。OPD 在理论上即一种 dense KL-约束 RL，故此线是其算法语境。
- **T4 — 思维链-Token 级 / MTP**：多 token 预测（MTP）、token 级数据选择/重加权（Rho-1/ssToken/SegmentSelectiveSFT/VCORE/Beyond-LogLik）、测试时前瞻干预等。对应项目的 MTP 前瞻探针与 token 异质性监督。

时间窗：覆盖 2023-06（GKD/MiniLLM 等奠基工作）至 2026-06（最新一批早期预印本，arXiv id 落在 2601–2606 区间）。来源：OPD 综述（opd_survey / Awesome-LLM-On-Policy-Distillation）、若干 awesome-list（Awesome-RL-for-LRMs、Awesome-Credit-Assignment）、引文图谱三轮扩展，以及对候选仓库的逐一可达性核验。

### 0.2 方法（如何得到这 117 篇）

1. **发现**：从 OPD 综述与 OPD blog 出发，结合多个 awesome-list 与关键词检索，汇出第一波候选（candidates_master，139 行级别）。
2. **验证**：逐条核验 arXiv id 真实性、代码链接可达性（剔除伪造 id / 404 / 与主题不符 / 重复，见附录 A）。
3. **深读**：对通过核验且与主题强相关、且代码可得（或机理价值足够高）的论文做逐篇深读，产出 12 节结构化分析。
4. **三轮引文图谱扩展**：在已深读论文的引用 / 被引邻域做 r2、r3 两轮扩展（r2 新增 78、r3 新增 175），把新命中的强相关 + 有代码者补入深读。
5. **全量二次审查**：对全部 117 篇重读全文 + 核心代码。
6. **加强重写**：在二次审查基础上，统一为带 TL;DR 的 12 节可读格式并加强表述。

### 0.3 防幻觉双层核验，与本轮如实勘误记录

本项目对"论文真实性 + 代码真实性"采取双层核验：第一层核 arXiv id / 标题 / 作者 / 会议；第二层 clone 仓库、读核心代码、核对损失/超参与论文是否一致。审查中如实发现并修正了下列问题（**无重大结论颠覆**）：

- **P4.5 修正**：`psft` 分析此前**编造了数值**，已据论文原文更正。
- **Phase-B / 加强轮**：对全部 117 篇重读全文 + 核心代码后，**修正了大量数据集 / 超参 / 框架精度问题**（如框架版本号、训练器入口、损失实现位置），并更正了**若干杜撰的会议 / 模型 / 任务**断言（典型：删除了若干"声称 ICLR/ACL 接收"但 PDF 正文无据的断言，如 `spear` 的 ICLR 2026 接收声明、对 `latent_agents`/`hpd` 等 venue 标注降级为〔待核〕）。
- **本轮甚至更正了前一轮审查的个别误判**：例如 `scope` 的双路 scope 方向曾被写反（"正确路 vs 错误路"的强化/纠错对象）；`sod_stepwise` 的 step 级重加权代码位置此前定位有误；`sdcl` 的 KL 方向（forward vs reverse）此前勘误方向反了，本轮据代码再校正。

所有未能由一手来源证实的断言，均保留 **〔待核〕** 标记，请读者以此为风险提示。

### 0.4 代码受限论文清单（训练代码不可直接获取）

下列论文虽通过真实性核验、且主题相关，但其**核心训练代码不可从所引来源直接获取**，审计深度受限（损失/超参常以论文正文为准）。按受限性质分组（共约 29 篇）：

- **404 / 不可得**：`amft`（仓库 404）、`llm_future_mtp`（apple/ml-mtp 返回 404）。
- **Coming-soon / 未释出**：`pi_play`（"Code will be released soon"）、`vla_opd`（项目页 "Code Coming Soon"）、`csd`（README 占位，训练代码尚未释出）。
- **Weights-only / 模型权重发布仓（无训练代码）**：`magistral`、`minimax_m1`（CISPO 训练代码不在仓内）、`prorl`、`deepseek_r1`、`deepseekmath`（仅评测/推理脚本）、`gemma2`、`qwen3`、`glm45`（后训练脚本不开源，RL 框架 Slime 另开源）、`nemotron_cascade2`、`nemotron_nano2`。
- **README-only / 仓库仅占位**：`adaspec`（仅 README/LICENSE/图，核心 loss 仅见论文附录 Listing 2）。
- **仅数据/评测，缺核心训练逻辑**：`deepdive`（仅 KG 数据合成，不含 RL 训练代码）。
- **无独立官方仓（集成于上游框架）**：`gkd`（TRL `GKDTrainer`）、`gspo`（集成于 veRL/TRL/ms-swift/ROLL）、`lite_ppo`（recipe in alibaba/ROLL）。
- **综述 / awesome-list（无方法实现代码，符合预期）**：`opd_survey`、`rl_survey_lrm`、`ca_survey`。
- **借道核验受限（训练代码实际可得但本仓不直接给）**：`apple_ssd`（部分：数据生成+评测，SFT 本体不在仓）、`behavior_priming`（RL 训练在另一仓 `cxcscmu/verl-agent-deepresearch`）、`eaft`（实现合入 LLaMA-Factory/ms-swift，主仓仅 README+assets，submodule 未拉取）、`lp_reg`（主仓占位，实现在 dev 仓 `Lp-Reg-dev`）、`ophsd`（已 clone 但训练数据 LFS 大文件以 SKIP_SMUDGE 占位未下载）、`rho1`（仅权重+评测，SLM 持续预训练代码未开源）。

### 0.5 如何阅读本文档

- **只想快速扫一遍 117 篇要点** → 直接看 **§1 的"一句话速览表"**：按主题分组，每篇一行 TL;DR + 框架 + 代码情况，一眼扫完。
- **想按主题精读** → §2（T1 OPD 核心）、§3（T2 Tool-Agent）、§4（T3 GFT/GRPO-RLVR）、§5（T4 CoT-Token/MTP）。每篇内联完整 12 节分析（标题已降一级，以 `######` 起），篇间以分隔线隔开。各篇 12 节结构为：核心贡献 / 方法 / 与本项目关系 / 关键实验 / 代码与框架 / 局限 / 可借鉴点 等（具体小节随原分析而定）。
- **想找"主题相关但未深读"的扩展候选** → **附录 B（§7）**，546 篇按主题分列简表（标题 | arXiv | 主题标签）。
- **想知道哪些被剔除、为什么** → **附录 A（§6）**。
- **产物 / 中间状态文件** → **§8 产物目录索引**。
- 遇到 **〔待核〕** 标记：表示该断言未经一手来源证实，使用时请自行复核。

---

## 1. 统计总览

### 1.1 数字一览

- **检视候选规模**：第一波 `candidates_master` 约 139 条（去重后 master 表内含 250 行级别条目，含后续合并）；r2 引文扩展新增 **78** 篇；r3 引文图谱扩展新增 **175** 篇。
- **深读论文**：**117 篇**（本文档 §2–§5 全文内联）。
- **代码受限**：约 **29 篇**（详见 §0.4 分组清单）。
- **主题分布（按主要一线归类，合计 117）**：T1 OPD 核心 = **41**；T2 Tool-Agent = **19**；T3 GFT/GRPO-RLVR = **50**；T4 CoT-Token/MTP = **7**。

### 1.2 框架分布

按 117 篇深读论文的"主训练框架"统计（一篇若用多框架则多线计入；keyword 命中以元信息/§2 为准）：

| 框架 | 命中篇数（约） | 说明 |
|---|---|---|
| **veRL / verl 家族**（含 VeRL、EasyR1、verl-agent、各定制 fork） | **~75** | 绝对主导。GRPO/RLVR 与多数 OPD-as-RL 实现的事实标准底座（FSDP/Megatron + vLLM rollout）。 |
| **TRL**（含 GKDTrainer/GRPOTrainer/SFTTrainer） | ~19 | 自蒸馏（OPSD/SDCL/CaOPD/UniSD 等）与轻量 KD 的主力；HF 生态。 |
| **DeepSpeed**（ZeRO-2/3，多作为训练后端） | ~13 | 常与 TRL/LLaMA-Factory/自研 trainer 搭配。 |
| **LLaMA-Factory**（含 LlamaFactory） | ~11 | SFT / 冷启动阶段主力（DASD/PRISM/HPD/VCORE/behavior_priming 等）。 |
| **vLLM**（rollout/生成后端，跨框架） | ~26 | 几乎所有 on-policy 采样的推理后端。 |
| **slime / Slime**（SGLang-native 异步 RL） | ~7 | GLM-4.5、Lightning-OPD、ICRL、OpenClaw-RL 等。 |
| **OpenRLHF** | ~6 | ORZ、spurious_rewards、sweet_rl、AlignDistil（定制版）等。 |
| **Oat**（sail-sg） | ~4 | Dr.GRPO、GMPO、HolderPO 等"机理派"工作。 |
| **Trinity-RFT**（Alibaba，veRL 后端 + Ray） | ~2 | CHORD、TCOD。 |
| **ROLL**（Alibaba） | ~2 | lite_ppo、（部分对照）。 |
| **NeMo / Nemo-RL / Megatron**（NVIDIA 栈） | ~5 | Nemotron 系、OPSA。 |
| **自研 / 其他**（KDFlow、Tinker、verifiers、smolagents、自写脚本等） | ~10 | DistiLLM/MiniLLM/Rock-Token/opd_blog/SafeSteer/LightReasoner 等。 |

要点：**veRL 家族是事实标准**（~75/117 篇涉及），TRL 在纯自蒸馏/轻量 KD 场景为第二极，slime 在大厂异步 agent RL 中崛起；机理批判派（Dr.GRPO/GMPO/HolderPO）偏好 Oat。

### 1.3 一句话速览表（重点）

下表抽取每篇 analysis 的 **TL;DR 行**，按四大主题分组，附主训练框架与代码情况，供一眼扫完 117 篇要点。表格中 `|` 已转义为 `/`。

#### T1 OPD 核心 / 自蒸馏

| Key | 一句话重点 (TL;DR) | 框架 | 代码情况 |
|---|---|---|---|
| `adaspec` | 投机解码里给小 draft 模型做蒸馏时，AdaSPEC 不再"所有 token 一视同仁地对齐大模型"，而是先用一个参考模型估出"哪些 token 对 draft 真的学得动"，只在这部分 easy token 上蒸馏，把有限容量花在刀刃上，从而把 token 接受率(acceptance rate)最高提升约 15%。值得看的点：它把"蒸馏目标(最小化全 token KL)与真实目标(最大化接受率)错位"这件事讲得很清楚，并给出一个极简的选择性过滤解法。 | HuggingFace transformers/TRL/Accelerate/DeepSpeed（两阶段选择性 KD） | 受限：README-only |
| `aligndistil` | 把"带 token-level reward 的 RLHF 目标"在理论上**等价改写成一个蒸馏问题**——其 teacher 分布是 DPO 模型与 reference 模型 logit 的线性组合，于是 RLHF 对齐就变成"向一个自动合成的 teacher 分布做 token 级蒸馏"。值得看的点：它给"DPO 的 token-level reward 分解"找到了一个干净的蒸馏对应，并用一个 reverse-DPO 对比 + 逐 token 自适应外插权重把这个对应做得更稳更准。 | OpenRLHF v0.5.2.post2（+ vLLM for on-policy） | 完整/已克隆 |
| `apple_ssd` | 不用任何 teacher / verifier / reward / RL / 代码执行环境，只让模型在"调过温度 + 截断"的设置下采样自己的**原始未验证**输出，再用标准交叉熵 SFT，最后评估时单独调一个解码温度——就能把代码生成显著提升(Qwen3-30B-Instruct LiveCodeBench v6 pass@1 42.4%→55.3%)。值得看的点：它给"自蒸馏为何起作用"提出一个清晰机制——代码解码存在 **precision-exploration conflict**，SSD 通过上下文相关地重塑 token 分布(该压的地方压、该留多样性的地方留)拿到固定解码拿不到的增益。 | 仅需 采样 + 标准 SFT(交叉熵) + 评估时单独调温；无 RL/verifier/teacher/执行环境 | 部分 |
| `avsd` | 同一个模型当 student 又当 teacher，teacher 额外看到三种"特权信息视图"（完整解 / 部分推理 / 仅答案）；AVSD 不固定用某一种视图，而是把多视图 teacher 信号拆成"跨视图共识"（可靠方向）和"单视图残差"（有用但有风险），用一个门控只在残差与共识方向一致且幅度相称时才加进去——从而比任何单视图自蒸馏都更稳更好。 | DeepSpeed ZeRO-2 + HF Accelerate + PEFT(LoRA) + vLLM，底层基于 OP | 完整/已克隆 |
| `brts` | 标准 OPD 每个 prompt 只用一条随机 teacher rollout，方差大、错了还会被放大。BRTS 对每 prompt 采 N 条 teacher 轨迹，按"先正确、再与学生对齐"挑一条；全错时用注入 ground-truth 的提示让 teacher 重新自然推导；选中的轨迹作为额外的 teacher-context 蒸馏分支，与标准 student-context OPD 一起训练。 | veRL（与 thunlp/OPD 同源 fork） | 完整/已克隆 |
| `caopd` | 标准 on-policy distillation（OPD/自蒸馏）在提升准确率的同时会系统性地把模型推入"过度自信"区；CaOPD 把"答什么（能力）"和"多确信（置信）"在监督目标上解耦——只把轨迹里的置信片段替换成学生自己 rollout 估出的经验成功率，其余照常做 reverse-KL 蒸馏，从而在几乎零额外改动下同时获得能力与校准。 | TRL 0.24 + vLLM 0.12 + DeepSpeed 0.18 + transformers 4.57（自建 | 完整/已克隆 |
| `copsd` | 低资源语言（尤其非洲语）数学推理差，是因为模型"有 latent 能力但调不出来"。COPSD 把 OPSD 的特权上下文 self-distillation 搬到跨语言场景：student 只看译成低资源语的题、必须用目标语推理；teacher 是同一个模型但额外拿到英文题 + 英文参考解，从而诱导出更可靠分布；在 student 自己的 rollout 上做逐 token reverse-KL 蒸馏。无需外部 teacher、无需目标语 rationale。 | HuggingFace TRL + Accelerate/DeepSpeed，LoRA，A100/H200 | 完整/已克隆 |
| `csd` | CSD 提出一种 **logit 级**的离线知识蒸馏目标，用"离散 concrete score 匹配"替代传统的 softmax 概率匹配（KL/f-散度），既避免 softmax 把 teacher 的大 logit 差异"压平"，又比直接对齐 logit（DLD）拥有更大的最优解集（对 logit 常数平移保持不变），并给出 O(/V/) 线性时间梯度。 | 即插式 KD 目标，可嵌入 ImitKD/GKD/DistiLLM | 受限：未释出/coming-soon |
| `dasd` | DASD 从"分布对齐"视角改造序列级蒸馏（即在 teacher 响应上做 SFT），用三件套（温度调度学习、散度感知采样、混合策略蒸馏）补回缺失的师生交互，仅 **448K** 样本就让 Qwen3-4B 在推理基准上达到同量级 SOTA，部分基准超过若干 32B 模型。 | LLaMA-Factory + DeepSpeed ZeRO-3 + Liger-Kernel | 见正文 |
| `distillm` | 把自回归 LM 蒸馏的两大痛点分别治理——用有理论保证的 **skew KLD** 替代不稳定的 KLD/RKLD 作目标函数，用带 replay buffer 的**自适应 off-policy 策略**把昂贵的学生生成输出（SGO）生成频率压到最低；在保持/超过 SOTA 蒸馏质量的同时把训练提速 2.5–4.3×。 | 自研（沿用 MiniLLM 代码基 + 定制 HF Transformers + DeepSpeed，无外部 RL 框架 | 完整/已克隆 |
| `distillm2` | 观察到 KL 与 RKL 的非对称行为——KL 在教师生成数据（TGO）上"抬高"、RKL 在学生生成数据（SGO）上"压低"——于是设计对比式蒸馏 loss（CALD）：对教师响应用 SKL、对学生响应用 SRKL，并配 α 课程与 β 线性递增，在指令/数学/代码/VLM 全面超过 GKD、DistiLLM、Speculative KD。 | HF alignment-handbook + Accelerate + DeepSpeed ZeRO-3 + vLLM | 完整/已克隆 |
| `gad` | 当教师是只返回文本的闭源 API(如 GPT-5)时无法做白盒/likelihood 蒸馏；GAD 把学生当生成器、训练一个判别器去区分学生与教师文本，构成 GAN 式极大极小博弈——判别器即"随学生共同演化的 on-policy reward model"，从而在黑盒下实现 on-policy 蒸馏，避免固定 reward model 的 reward hacking。 | veRL（GRPO，hack critic 当判别器） | 完整/已克隆 |
| `gemma2` | Gemma 2 的 2B/9B 用**知识蒸馏(逐 token 软标签的交叉熵)替代 next-token 预测**来预训练小模型；后训练 SFT 阶段明确"在学生自身分布上从教师蒸馏(on-policy KD，引 GKD/MiniLLM)"——这是它与本综述(on-policy 蒸馏)最直接的连接点。 | 内部 Google JAX + TPU 栈 | 受限：权重/无训练代码 |
| `gkd` | 把自回归 LM 的知识蒸馏当成"交互式专家模仿学习"——让学生在**自己生成**的序列上、用教师的 token 概率作监督，从而消除训练/推理的分布失配；并把散度选择（forward/reverse KL、JSD）与 on-policy 数据比例统一成一个可调框架 GKD。 | T5/JAX（论文）、TRL（社区集成） | 见正文 |
| `glm45` | 开源 MoE（355B 总 / 32B 激活）混合推理模型（thinking + direct 双模式）。后训练分两阶段——先分域训三个专家（Reasoning/Agent/General-chat，各 cold-start SFT + 专家 RL），再用 **self-distillation** 把多专家统一进一个通才；agent RL 中用 **iterative (self-)distillation** 在昂贵 RL 之间快速抬升起点。 | **Slime** 单独开源 https://github.com/THUDM/slime | 受限：权重/无训练代码 |
| `gopd` | 先在理论上证明 OPD 是"reward 与 KL 永远等权(β=1)、reference 可任选"的 dense KL-约束 RL 特例；再加两个旋钮——**灵活 reference πref** 与 **reward 缩放因子 λ**。λ∈(0,1) 是插值（学生介于 ref 与 teacher 之间），**λ>1 是外推(ExOPD)**，能让学生**越过 teacher 边界**；在多领域专家合并设定下 ExOPD 是唯一能让统一学生稳定超过所有领域 teacher 的方法。 | **veRL v0.6.1** | 完整/已克隆 |
| `hpd` | 把 SFT、FKLD、RKLD 统一为 token 级 reweighted log-likelihood 目标，用 K1 估计器（Schulman 2020）在每个 token 上判 student 对 expert token 的 under/over-estimate，据此自适应混合 forward/reverse-KL 并重分配概率质量——既保留 one-hot 监督的计算效率，又兼容 off-policy 数据 + 轻量近似 on-policy 采样，从而以更省算力逼近 dense 蒸馏。 | LlamaFactory（SFT 路径）+ veRL（RL 路径） | 完整/已克隆 |
| `lightning_opd` | 把 on-policy distillation (OPD) 改造为离线版——预先一次性算好并缓存每 token 教师 log-prob 复用，去掉训练期常驻教师 server；关键贡献是识别并证明被忽视的 **teacher consistency**（SFT 与 OPD 必须同一教师）条件，在该条件下离线 OPD 与标准 OPD 共享最优点、梯度差有界且带隐式正则，从而在性能持平/更优的前提下提速 3.6×–4.0×（含让 MoE 教师场景从 OOM 变可行）。 | slime + slime_plugins（SFT 用 LLaMA-Factory 风格 YAML，教师 log-pro | 完整/已克隆 |
| `lightreasoner` | 用"专家(大)模型 vs 业余(小)模型"的下一-token 分布分歧定位高价值推理时刻，把对比 log-prob 差(contrastive-decoding 信号)蒸成软标签反过来微调专家模型；卖点是极省资源、无需 ground-truth，但精度增益常不及标准 SFT、对已对齐模型几乎无效。 | 自实现轻量 Python 流水线(非 verl/TRL) | 完整/已克隆 |
| `madopd` | 把多智能体辩论(MAD)搬进 OPD 训练环，让 K 个 teacher 就 student 的 on-policy 状态多轮辩论、辩论 transcript 作为 privileged context 产生 token 级监督(按辩论后置信加权)，以突破单 teacher 天花板；并以 OPAD(step-level 采样)把 OPD 扩到 agentic 任务，配 task-adaptive 散度原则(agentic 用有界 JSD、代码用 reverse KL)。 | TRL(0.15–0.25) + vLLM + DeepSpeed ZeRO-3 + transformers 4.51 | 完整/已克隆 |
| `minillm` | 把标准 KD 的 forward KLD 换成 reverse KLD(让小学生 mode-seeking、只学教师主要模式而非长尾)，并用策略梯度做 on-policy 优化；配三项稳定化(单步分解降方差、教师混合采样防 reward hacking、长度归一去短句偏好)，在 120M–13B 跨族一致优于 SFT/word-KD/SeqKD。 | 自研(改版 HF Transformers + DeepSpeed + Accelerate，类 RLHF pipeli | 完整/已克隆 |
| `nemotron_cascade2` | 在按领域顺序的 Cascade RL 中插入一个"多领域 on-policy 蒸馏(MOPD)"稳定化阶段——用各领域最强中间 checkpoint 作教师、以 token 级 reverse-KL 蒸馏优势恢复 RL 造成的回退；30B/3B-激活 MoE 由此在数学/代码达到接近前沿、IMO/IOI 金牌级，但以牺牲通用知识/STEM(MMLU-Pro、GPQA) 为代价。 | Nemo-RL | 受限：权重/无训练代码 |
| `nemotron_nano2` | 基于 Nemotron-H 的混合 Mamba-Transformer 推理模型，先预训练 12B base(20T tokens, FP8)，多分支对齐(SFT+GRPO+DPO+RLHF+模型合并)后用 Minitron 剪枝 + 仅 forward-KL 的 logit 蒸馏压到 9B，目标在单张 A10G(22GiB)上做 128k 推理，较 Qwen3-8B 同精度下吞吐高 3×–6×。 | NeMo / Megatron | 受限：权重/无训练代码 |
| `opd_blog` | 系统性推广"on-policy distillation"——学生在自身采样的轨迹上，以更强教师逐 token 的分布（reverse KL）作为唯一稠密监督，兼顾 on-policy 的分布真实性与教师反馈的稠密性；博客在推理(数学)、个性化(指令遵循恢复)、多轮工具使用三场景演示，配套 tinker-cookbook 给出可复现 LoRA 配方。 | Tinker SDK（自研托管训练 SDK，基于 LoRA），非 veRL/TRL | 完整/已克隆 |
| `opd_survey` | 本课题(MTP+OPD / TSRD)的核心背景综述——把 On-Policy Distillation(OPD)统一刻画为"学生采样轨迹上的 f-散度最小化",沿三条设计轴(优化什么 / 信号从哪来 / 如何稳定)组织 >100 篇文献,并系统给出成功条件、失效模式与 OPD↔KL-约束 RL 的连接;提供统一分析词汇与"一方法一类别"分类。 | n/a（综述） | 受限：权重/无训练代码 |
| `ophsd` | 把"推理时脚手架(harness)"从永久固件重定位为临时训练支架——训练时让模型在 harness 内 rollout、用 harness 诱导的轨迹作 teacher(自蒸馏,reverse-KL),把过程性推理能力永久内化进基座参数,推理时撤掉 harness;数学/文本分类上超过 OPSD/GRPO,且再接回 harness 不再增益甚至降分。 | veRL（子类化 PPO trainer,base 为 OPSD repo） | 受限：README-only |
| `opsa` | 把 OPSD 自蒸馏迁到安全对齐——学生在线 rollout、frozen 自身副本(条件于按 prompt 类型选的安全/有益特权上下文)沿轨迹给逐 token KL 监督;创新点是用 **teacher flip rate(TFR)** 离线挑"能把不安全回答翻成安全"的上下文,从而在更小的推理代价下降低 "safety tax"。 | NVIDIA NeMo-RL 扩展 fork | 完整/已克隆 |
| `opsd` | 同一个 LLM 自任教师与学生——教师条件于"问题 + 参考解(特权信息)"、学生只看问题,沿学生自采样轨迹做逐 token 散度蒸馏,无需外部教师即可用 ground-truth 提供稠密 on-policy 监督,在数学推理上匹配/超过 GRPO 且 token 效率显著更高。 | TRL（基于其 experimental GOLD/GKD trainer） | 完整/已克隆 |
| `prism` | 在 SFT 与 RLVR 之间插入一个独立 "预对齐" 阶段，用 black-box（无需教师 logits）的对抗式 OPD——policy 对一个含感知/推理双专家的 MoE 判别器做极小极大博弈——修复 SFT 引入的（且对感知/推理异质的）分布漂移，为下游多模态 RLVR 提供更好初始化。 | 三阶段：LLaMA-Factory（SFT）+ verl（对齐/RLVR）+ vendored transformers | 完整/已克隆 |
| `qwen3` |  | - Reasoning RL / General RL:**GRPO**(引用 Shao et al. 2024);提到 | 受限：权重/无训练代码 |
| `resd` |  | 基于 veRL（volcengine/verl）与 SDPO（lasgroup/SDPO）。维护两个持久化上下文：**p | 完整/已克隆 |
| `rethink_opd` |  | 主体基于 **verl (v0.7.0)** 做 OPD 与 RL(GRPO);**LlamaFactory (v0.9 | 完整/已克隆 |
| `rock_tokens` | OPD 训练表观饱和后仍有约 6% 词表、占输出 18% 频次的 token 持续高 KL loss（"Rock Tokens"，多为结构/话语脚手架），它们贡献了不成比例的梯度但对实际推理性能功能贡献可忽略；从训练起冻结其梯度可在不掉点下精简对齐（约 1.4× 加速）。 | 自定义 KDFlow（SGLang+Ray+FSDP2+bf16） | 完整/已克隆 |
| `rosd` | 标准在线策略自蒸馏（OPSD/SDPO）把自教师条件于"完整正确解"会让学生模仿训练域参考轨迹、损害 OOD；ROSD 改为"纠错思路 e + 错误引语 q"引导的**错误后缀局部蒸馏**——只修第一处出错起的后缀、保留有效前缀，从而在保住域内的同时大幅改善跨域泛化（4B 上 OOD 平均 41.31% vs SDPO 2.88%）。 | verl（扩展自 SDPO/lasgroup/SDPO）+ FSDP actor + vLLM rollout | 完整/已克隆 |
| `safesteer` | 安全特征在输出分布中本就稀疏，对齐应是"局部修改而非全局权衡"；SafeSteer 用 activation steering 造安全 teacher，挑出稀疏安全 token 子集 S，仅在 S 上施加 reverse-KL OPD，仅用 100 条有害样本即在几乎不掉通用能力下显著降 ASR。 | 自研轻量安全对齐框架（activation steering teacher + token 选择 + localize | 完整/已克隆 |
| `scope` | 标准 OPD 把 teacher 的稠密 token-level 监督一视同仁地用在所有 rollout 上,忽视信号质量差异;SCOPE 按轨迹正确性双路路由——正确轨迹用 **student-PPL 加权 MLE 自强化**(放大能力边界处低置信样本),错误轨迹用 **teacher-PPL 加权 KL 蒸馏**(优先 teacher 真有纠错能力即低 PPL 的实例),并在组内做 perplexity 归一化。 | veRL(volcengine/verl) | 完整/已克隆 |
| `sdcl` | 把"示范条件化的同一模型(EMA)"当自己的 teacher,在 student(只看 query)的 on-policy rollout 上做 token 级 KL 蒸馏,把 off-policy 的专家示范转成 on-policy 学习信号,从而在学新技能/新知识时显著少遗忘——是 privileged-context self-distillation 家族用于持续学习的实例。 | TRL 0.24.0 | 见正文 |
| `tip` | OPD 中"哪些 token 有学习信号"由两轴决定——学生熵 ht 与师生散度 δt；熵单轴是有效但结构性不完整的代理，会漏掉"低熵高散度=过度自信错误"的 Q3 盲区；用免参数 Soft-OR 评分把两轴并起来做 top-k token 选择，可在大幅省显存的同时匹配/超过全 token OPD。 | verl | 完整/已克隆 |
| `unisd` | 把"无外部强 teacher 的自蒸馏"建模为 on-policy 轨迹上的可靠性感知自纠错，沿监督可靠性/表示对齐/训练稳定性三轴整合五个互补组件（多 teacher 一致性、EMA teacher、token 级对比、特征匹配、散度裁剪）做系统消融；整合版 UniSD\* 较 base +5.4、较最强基线 GKD +2.8。 | + 组件级消融，OPD 相关性中等（偏经验综述/工程整合而非单一新机制） | 完整/已克隆 |
| `vla_opd` | 把 VLA 后训练改写成"在 student 自采样轨迹上、用冻结 teacher 的 dense token-level 监督做 reverse-KL 蒸馏"的 on-policy RL，从而同时拿到 SFT 的快收敛、RL 的少演示/抗遗忘，并用 reverse-KL 的 bounded mode-seeking 避免 Forward-KL 熵爆炸与 Hard-CE 熵坍缩。 | 〔待核：代码未放出，下述基于论文〕基于 GRPO 式分组采样、teacher=SimpleVLA-RL、student= | 受限：未释出/coming-soon |
| `why_sd_degrade` | 自蒸馏在数学推理上会"response 变短但性能反降（最高 ~40%）"，根因是 teacher 在富 context 下生成的自信轨迹压制了 epistemic verbalization（wait/hmm 等不确定性表达）；该压制由 conditioning 信息丰富度驱动，其危害随任务覆盖度增大而显现于 OOD。 | veRL（仓库含 `verl/` 目录、megatron/sglang 脚本） | 完整/已克隆 |

#### T2 Tool-Agent / 多轮

| Key | 一句话重点 (TL;DR) | 框架 | 代码情况 |
|---|---|---|---|
| `chain_of_agents` | 把"多智能体协作"压进单一模型——先用 multi-agent distillation 把 SOTA 多智能体系统（OAgents）的执行轨迹转成 CoA 格式做 agentic SFT 冷启动，再用 agentic RL（DAPO）在可验证任务上优化，得到能原生动态激活不同 tool/role agent 的 Agent Foundation Model（AFM）。本质是组合既有组件的大体量工程 recipe，与 token 级 OPD 不同源。 | SFT 用 LLaMA-Factory、RL 用 veRL 跑 DAPO | 见正文 |
| `deepdive` | DeepDive 用知识图谱（KG）随机游走 + 属性模糊化自动合成"难找"deep-search QA，再用端到端多轮 GRPO（带 redundancy penalty 抑制重复查询）训练浏览 agent，DeepDive-32B 在 BrowseComp 上达 15.3%（KG 数据），加半自动 i.i.d. 数据可达 22.2%。 | slime（RL）+ 多轮 GRPO | 受限：权重/无训练代码 |
| `distill_agent_tools` | 不只蒸馏教师的"推理"，而是把教师 agent 的完整"think + act（检索/代码工具）"任务求解行为蒸馏到小模型；配两项改进——用 first-thought prefix 提高教师轨迹质量、用 self-consistent action generation 提高学生测试时鲁棒性——使小模型 agent 能匹配甚至超过比它大 2–4× 的 CoT 蒸馏模型。 | smolagents v1.13.0.dev0 + TRL SFT trainer + LoRA；检索环境沿用 Sear | 完整/已克隆 |
| `gigpo` | 把 GRPO 扩展到长 horizon 多轮 agent——除了像 GRPO 那样按整条轨迹回报算 episode 级相对优势,再利用"组内轨迹常反复经过相同环境状态"这一观察,把同一状态下的不同动作聚成 step 级组算 micro 相对优势,从而**无需额外 rollout、无 critic** 就实现细粒度 step 级信用分配,额外时间成本 <0.002%。 | verl-agent（veRL 扩展） | 完整/已克隆 |
| `icrl` | 让同一 backbone 用 role-specific prompt 同时当 solver 与 critic 联合 RL，把"有 critique 才能做对"内化为"无 critique 也能做对"——核心是一个把 critique-conditioned 修订轨迹按 token 级 re-weight 比率 `w_t=π(y_t/q)/π(y_t/q,c)` 迁移到 critique-free 分布的"分布校准"，外加逐角色组内归一化以稳定联合优化。 | slime (THUDM, SGLang-native RL) + AgentGym | 见正文 |
| `latent_agents` | 用"SFT 学辩论结构 + GRPO 把显式辩论压进潜空间"的两阶段微调，把多个 agent 多轮辩论（multi-agent debate）内化进单个 LLM（IMAD），用 Debate 6.3%~21.1% 的 token（5–16× 提效）匹配/超过显式辩论；并发现内化后存在线性可分的"agent 子空间"，可做行为控制。属概念验证级（944 条算术 trace、LoRA）。 | TRL (GRPOTrainer) + PEFT/LoRA | 见正文 |
| `meow_tea_taro` | 把多轮 agentic RL 的设计空间拆成 environment/reward/policy 三支柱做系统受控消融，得出一份可操作配方——"课程(由简到繁) + 稳定化偏置策略(PPO/GRPO 优于无偏 RLOO 与朴素 REINFORCE++) + 验证型稠密奖励(单测通过率远胜模型评判)"。非新算法，是经验研究；框架封装 veRL。 | 封装/vendoring veRL | 完整/已克隆 |
| `open_agentrl` | 一个完全动态的闭环 RL 系统,同时进化"环境、策略、生成式奖励模型"三者——策略用 step-wise+outcome 融合反馈训练,奖励模型经一致性反馈联合优化产出可靠 step-wise 监督,环境据策略当前能力自适应调难度;论证优化后的 step-wise 信号优于人工 outcome 标签。 | veRL（仓库内置 `verl/`） | 完整/已克隆 |
| `openclaw_rl` | 把每次 agent 交互产生的"next-state 信号"当作在线学习源回收,从中抽 evaluative(标量/更频繁)与 directive(token-level/更富信息但稀疏)两类信号,在一次 hybrid RL 更新中统一;并用 overlap-guided hint selection + logprob-diff clip 稳定 teacher–student 失配下的 OPD,使"agent 越被使用越变强"。 | slime 异步 RL（Megatron 训练 + SGLang serving + PRM） | 完整/已克隆 |
| `pi_play` | 自博弈在造题时天然产出一条 "问题构造路径(QCP)"，本文把它当作零成本的内禀特权信息，让同规模 teacher 据此对 student 做 token 级 reverse-KL 自蒸馏，从而把稀疏奖励自博弈变成稠密反馈的 data-free 自演化。 | 论文未绑定特定开源框架，student 用 GRPO | 受限：未释出/coming-soon |
| `rstar2` | 用"高吞吐代码执行环境 + 抗噪的 GRPO-RoC（Resample-on-Correct）+ 短长度多阶段 RL recipe"，在 64×MI300X、510 步 / 一周内把 Qwen3-14B-Base 推到前沿数学推理，AIME24=80.6/AIME25=69.8/HMMT25=52.7，并在 AIME24/HMMT25 上超过 671B 的 DeepSeek-R1（AIME25 基本持平）。 | veRL v0.5 + Code Judge + vLLM | 完整/已克隆 |
| `score` | 让小学生 agent **主导**轨迹生成、teacher **只纠正最早一步错误**,学生从"已验证前缀"续写并做短 horizon RL,把行为克隆的累积误差从 O(H²) 降到 O(H);RL 阶段从最早错误前的前缀起 rollout 并用 key-step 稠密奖励缓解稀疏奖励。 | 自定义工具链(LLaMA-Factory + LangGraph + veRL,非单一框架) | 见正文 |
| `search_dont_guess` | 小模型(SLM)做 search agent 时反而比大模型**更少搜索、更易幻觉**(under-searching),且"自适应搜索"在 SLM 上会掉点(Adaptive Search Trap);本文用 **Always-Search Policy(ASP)** 显式约束"总是搜索、别猜",经 SFT/OPD/Mixed 三种实现把搜索行为蒸进 SLM,使 1.7B 逼近甚至局部超过 8B。 | 检索 E5+BM25 / Search-o1 式迭代 agent / 训练 SFT+OPD+RFT | 见正文 |
| `search_r1` | 把搜索引擎调用嵌进 RL 训练循环——用结构化标签做多轮"推理↔检索"交织生成,对检索回来的 token 做 **loss masking**,仅用最简的 **EM 结果奖励**(无过程/格式奖励),即可让 LLM 自发学会"何时检索、检索什么、如何结合检索继续推理"。 | veRL 定制 fork | 完整/已克隆 |
| `sod_stepwise` | 把 OPD 用于小模型 agent 的工具集成推理（TIR）会因工具错误触发的"加速分布漂移"而训练崩溃；SOD 按 step 级师生发散自适应重加权蒸馏强度（高发散区衰减、重对齐时回升），在对齐区保留 dense 监督，使 0.6B/1.7B 学生相对最强基线 OPD 平均 +20.86%/+18.50%。 | veRL fork + Open-AgentRL + ReTool agentic 组件 | 完整/已克隆 |
| `spear` | 针对多轮 agent RL 中机械熵最大化易致不稳定的问题，SPEAR 用"课程化自模仿学习（SIL）+ 内在奖励塑形"在自身经验引导下渐进调节策略熵（早期广探索、后期收敛利用），在 ALFWorld/WebShop/Sokoban/AIME 上稳定提升 GRPO/GiGPO/Dr.BoT，且额外开销仅理论 10%–25%。 | veRL + verl-agent | 完整/已克隆 |
| `sweet_rl` | 在多轮 agent 任务上，用训练期才可见、actor 看不到的额外信息（最终结果 + 参考解）训练一个非对称的 turn-level critic 做 step 级信用分配；关键设计是"直接用动作 log-prob 参数化 advantage 函数 + Bradley-Terry 目标"，避开在 LLM 上训 value head 损泛化的问题；并配套开源多轮协作 benchmark ColBench。 | OpenRLHF 定制 fork（`YifeiZhou02/collab_openrlhf`，支持 multi-turn | 见正文 |
| `tcod` | 在多轮 agent 场景下 vanilla OPD 会因跨轮误差累积出现"轨迹级 KL 不稳定"（KL 飙升、成功率坍塌），TCOD 用一条时间课程逐步扩大暴露给学生的轨迹深度（F2B 浅到深 / B2F 由 teacher 前缀导航后到前），在保留 OPD 稠密信号的同时稳住训练并最高 +18 分。 | Trinity-RFT（Ray-based RFT，Alibaba） | 完整/已克隆 |
| `webagent_r1` | 用端到端多轮 on-policy RL（M-GRPO + 动态上下文压缩 + 并行轨迹 rollout），仅靠二值任务成功奖励，把 web agent 从 prompting/BC 水平大幅提升；并系统说明 BC warm-up 不可或缺、long-CoT 帮 SFT 但限制 RL 探索、以及"增加交互轮数"这一新的 test-time scaling。 | **verifiers**（willccbb/verifiers，GRPO-based） | 完整/已克隆 |

#### T3 GFT 统一 SFT-RL & GRPO-RLVR

| Key | 一句话重点 (TL;DR) | 框架 | 代码情况 |
|---|---|---|---|
| `amft` | 与其用"SFT→RL"两阶段或靠启发式硬切换，AMFT 把 SFT 和 RL 揉进**单阶段单循环**，并把"模仿(SFT)与探索(RL)的配比 µ"当成一个**可学习参数**，用 meta-gradient 以"最大化最终任务表现"为元目标前瞻式地学这个 µ。值得看的点：它把 SFT 形式化为"优化专家示范里隐含的隐式 reward"的特殊 RL，从而让 SFT/RL 在同一目标下统一——这与 OPD"在线策略蒸馏里如何调度模仿 vs 探索权重"的命题同构。**注意：代码仓库当前 404 不可得，方法细节无法独立核实。** | 〔待核，仓库不可得〕方法层面建立在 GRPO(RLVR)+SFT 加权损失之上 | 受限：404/不可得 |
| `ampo` | 当 on-policy RLVR 在某道难题上**整组采样全失败**(稀疏奖励、学不动)时，AMPO 才**按需**从一个**多教师池**里挑入正确解替换失败样本；挑哪条教师路径不看"哪个教师最强"，而看"哪条对学生最容易吸收"(学生在该路径下生成正确答案的概率最高)。值得看的点：用 4 个同量级"同伴"教师 + 仅 8.5k 数据，就媲美了用单一更强教师(DeepSeek-R1)+46k 数据的方法，并全程保持更高 entropy(探索性)。与 TSRD 的"按需脚手架 + 路径恢复"强相关。 | verl(GRPO 的 Mixed-Policy 扩展, FSDP 或 Megatron) | 完整/已克隆 |
| `asft` | DFT(用 token 概率给交叉熵重加权的 SFT 变体)在推理域好用、在知识域(如医疗)不稳，原因是它"界紧但会漂移"(KL 持续增大)。ASFT 只加一项**轻量 KL 锚定**把策略约束在 base 模型附近，就同时保住 DFT 的紧界优势和稳定性。值得看的点：它用 **RWR(reward-weighted regression)框架**统一解释 SFT/DFT，证明 DFT 给出比 SFT 可证更紧的 RL 下界，并指出其缺分布锚定才是不稳定根因——理论诊断 + 一行 KL 修复。 | 自定义 HF Trainer + DeepSpeed(ZeRO-2/3) + PEFT/LoRA；新增 veRL(FSD | 完整/已克隆 |
| `bapo` | 异策略（off-policy）RL 训练 LLM 时，数据越陈旧（staleness 越大）越容易梯度爆炸、熵崩溃。BAPO 找到两个根因——负优势样本主导梯度、固定对称裁剪系统性挡掉"增熵更新"——并据此**每个 batch 动态调裁剪上下界**，让"正 token 贡献占比"达到目标 ρ0，从而既防爆炸又保熵。 | veRL（GRPO 为基础算法） | 完整/已克隆 |
| `behavior_priming` | 先用 LLM pipeline 找出让"搜索 agent"成功的四种推理行为（验证、权威评估、自适应搜索、纠错），再用 SFT 把这些行为"种"进模型、之后再 RL。关键证据：用"展现这些行为但答案错误"的轨迹做 SFT，效果 ≈ 用答案正确的轨迹——说明**解锁 RL 的关键是推理行为（path）而非 outcome 正确性**。 | SFT 用 LLaMA-Factory、RL 用 GRPO（veRL 系） | 受限：404/不可得 |
| `ca_survey` | 一篇 survey + awesome-list，以"信用分配 (credit assignment, CA)"为中心透镜重审 LLM RL。核心产物是把 2024–2026 初的 **47 篇方法**（41 篇 core + 6 篇 adjacent）按**粒度 × 方法学二维分类**，并配 reporting checklist、benchmark 协议与方法选择决策树。判断：reasoning CA 正趋成熟（PRM + critic-free group comparison），agentic CA 催生真正新方法（hindsight counterfactual、privileged asymmetric critic、turn-level MDP）。 | ，可用于定位、选 baseline、找相邻工作 | 受限：404/不可得 |
| `cbrl` | 把 few-shot 示范当成 RLVR 训练时的"临时脚手架"——早期以高概率把示范前置到 prompt 里帮模型产生成功 rollout（拿到学习信号），再按课程把注入概率线性退火到 0，逼模型把推理模式"内化"而非依赖；只改训练输入分布、不动 RL 目标，因此算法无关、推理零开销。 | verl 0.3.0.post2 + vLLM 0.8.5 + PyTorch 2.6.0，Hydra 配置 | 完整/已克隆 |
| `cepo` | GRPO 给一条轨迹里所有 token 同一个优势，浪费在 filler 上、低估决定性步。CEPO 在每个 token 问"正确答案偏好它**且**错误答案反对它吗？"——用对比比率 P⁺_T(y_t)/P⁻_T(y_t)（正确/错误答案两个 teacher，错误答案取自组内已有的 rejected rollout）在 stop-gradient 下调制 GRPO 优势幅度，符号仍由 verifier 锚定；由于学生先验 P_S 被消掉，从构造上消除了 RLSD 的 fluency confound，在决定性 token 处锐化、filler 处恰好失效。 | veRL（经由 EasyR1）+ FSDP + vLLM | 完整/已克隆 |
| `chord` | 别再把 SFT 当 RL 前面的独立阶段（那会经历"漂移—再适应—过拟合"并破坏 on-policy 探索）。CHORD 把 SFT 重构成 on-policy RL 里一个**动态加权的辅助目标**：全局系数 μ（带 warmup 的余弦衰减）控制专家信号占比从"以模仿为主"平滑过渡到"以探索为主"；token 级权重 φ(p)=p(1−p) 对那些已很可能或极不可能的专家 token 下调学习信号，缓解熵坍缩与干扰。 | Trinity-RFT（基于 veRL 后端 + Ray） | 见正文 |
| `dapo` | DAPO 把朴素 GRPO 拆解出 4 个关键改造（Clip-Higher、动态采样、token 级损失、超长奖励整形），并完整开源算法+数据+代码，在 Qwen2.5-32B base 上把 AIME 2024 从约 0 提到 **50 分**，以约一半训练步数超越 DeepSeek-R1-Zero-Qwen-32B（47 分）。 | veRL（vllm==0.8.3、ray[serve]） | 见正文 |
| `deepseek_r1` | R1-Zero 证明仅靠纯 GRPO + 规则奖励（不经 SFT）即可在 base 模型上自演化出推理能力（含"aha moment"），R1 再用"cold-start SFT → 推理 RL → 拒绝采样 SFT → 全域 RL"四阶段管线修好可读性与通用性；并把 800k 数据离线 SFT 蒸馏到小模型，得出"大模型 RL→小模型蒸馏 优于 小模型直接 RL"。 | 自研高性能 RL 框架（附录 B.1） | 受限：权重/无训练代码 |
| `deepseekmath` | DeepSeekMath 首次提出 **GRPO**——用"同题多输出的组内相对奖励"替代 PPO 的价值网络，省去与策略同规模的 critic、大幅降显存算力，并给出统一梯度范式分析 SFT/RFT/DPO/PPO/GRPO 的异同；DeepSeekMath-RL 7B 在 GSM8K=88.2%、MATH=51.7%。 | custom/none（DeepSeek 内部实现） | 受限：权重/无训练代码 |
| `denoiserl` | 把弱模型生成的错误推理前缀当作"结构化噪声"注入策略的 rollout，用 RL 训练策略从错误中间状态"去噪并恢复"到正确答案，从而在不引入更强教师、不构造难数据的前提下把"自纠错"从涌现行为变成显式训练目标。 | VeRL，RL backbone 用 GRPO / DAPO | 完整/已克隆 |
| `dft_reweight` | 把标准 SFT 梯度还原成"策略梯度 + 隐含奖励"形式后发现其奖励被 1/πθ（逆概率）加权而病态；只需把每个 token 的交叉熵损失乘以该 token 的预测概率（detach 阻断梯度），就能把隐含奖励整平为常数 1，得到更稳定、更接近 RL 风格的更新，从而显著改善 SFT 的泛化——核心改动只有一行代码。 | veRL（FSDP SFT 训练器 + Liger），评测沿用 Qwen2.5-Math 仓库 | 完整/已克隆 |
| `dr_grpo` | 批判性审视 R1-Zero 范式的两大成分——base 模型与 RL 算法——指出 Qwen2.5 base 已"类 SFT"、"Aha moment"在 base 中早已存在；并发现 GRPO 目标里的 1//o_i/（响应级长度偏置）与 std(R)（题目级难度偏置）会人为推高（尤其错误）响应长度；去掉这两项得到无偏的 **Dr. GRPO**，在不损推理性能下大幅缩短错误响应、提升 token 效率，并给出 7B 极简 SOTA 配方（AIME24 43.3%，8×A100/27h）。 | Oat） | 完整/已克隆 |
| `eaft` | SFT 之所以损害通用能力，主要源于一类"模型很自信、却被强迫去学相悖标签"的 token（Confident Conflicts）；EAFT 用逐 token 的归一化熵作为门控系数去缩放交叉熵——模型自信(低熵)就压低梯度、不确定(高熵)就正常学习，从而在几乎不损失目标任务的同时显著缓解遗忘。 | LLaMA-Factory（主）+ ms-swift | 受限：README-only |
| `entropy_mechanism` | RLVR 训练里策略熵会快速坍缩、把性能死死锁在一个可预测的低上限上；本文给出熵坍缩的经验定律 R=−a·exp(H)+b 与其动力学根因(熵变正比于"动作概率与 logit 变化的协方差")，并提出 Clip-Cov / KL-Cov 两个只动"高协方差 token"的简单干预来持续维持探索、突破瓶颈。 | veRL（fork 自 DAPO recipe） | 完整/已克隆 |
| `fest` | 只用 **128 条随机**(非精选)抽自 SFT 数据集的示范，就能显著提升 RLVR；关键是 semi-online DPO 的梯度天然同时含"监督 + on-policy + 自带衰减权重"三要素，把少量示范作正样本、agent rollout 作负样本即可。 | VeRL（GRPO 为主，few-shot 用 semi-online DPO） | 见正文 |
| `gmpo` | GRPO 优化 token 级奖励的**算术平均**，对离群重要性比敏感、易引发激进更新与不稳定。GMPO 即插即用地换成**几何平均**（在 log 空间做乘积与裁剪），天然抗离群、重要性比方差更低，从而能用**更大的裁剪窗口**鼓励探索而仍保持稳定。 | 专用仓基于 **Oat + vLLM 0.8.4**（构建于 understand-r1-zero / Dr.GRPO  | 完整/已克隆 |
| `gspo` | GRPO 在 **token 级**做重要性采样校正是"病态"的——单样本权重无法完成分布校正，反而注入随序列累积、被 clip 放大的高方差噪声，导致大模型（尤其 MoE）不可逆崩溃。GSPO 改在 **序列级**定义（长度归一化的）重要性比并做序列级裁剪/优化，使优化单位与奖励单位（整条序列）对齐，训练更稳更高效，已用于 Qwen3。 | Qwen 内部 RL（训练 Megatron + 推理 SGLang/vLLM） | 见正文 |
| `hapo` | 现有 RLVR 对所有 token 一视同仁，违背语言生成的异质本质。HAPO 把 **token 熵当作贯穿采样/优势/裁剪全流程的连续优化驱动量**（而非离散过滤或事后正则），用四个组件对每个 token 做细粒度差异化处理，在多模型规模上一致优于 DAPO。 | **verl + vLLM** | 完整/已克隆 |
| `holderpo` | 把 GRPO 系 RLVR 里"token 级重要性比如何聚合成序列级标量"统一为 Hölder p-mean（p=1→GRPO、p→0→GMPO/GSPO），并沿训练时间退火 p，以单参数平衡"梯度集中放大稀疏信号"与"梯度方差受控"这一无法被任何固定算子同时兼得的 trade-off。 | ，与 OPD/MTP 关联较弱（纯 RL 聚合层面，可借鉴 token 加权视角） | 完整/已克隆 |
| `hpt_upge` | 提出 Unified Policy Gradient Estimator (UPGE)，把 SFT 与各类 RL 后训练的策略梯度分解为四个可互换组件，论证 SFT 与 RL 是同一优化过程在不同数据分布假设/bias-variance 权衡下的实例；据此提出 Hybrid Post-Training (HPT)——按 on-policy rollout 准确率在实例级二值切换"纯 RL"与"纯 SFT"信号。 | veRL + LUFFY 的 mix_src 扩展（FSDP + vLLM rollout） | 见正文 |
| `limit_rlvr` | 用大 k 的 pass@k 系统度量 base vs RLVR 模型的推理"能力边界"，发现 RLVR 只在小 k 占优、大 k 被 base 反超，且边界随训练收窄——结论是当前 RLVR 主要提升采样效率、并未引入超越 base 的新推理能力，而蒸馏可以。 | pass@k 评测(vLLM 多采样)，被评模型来自 SimpleRL-Zoo/Oat-Zero/DAPO/Code-R | 完整/已克隆 |
| `lite_ppo` | 在同一框架(ROLL)/同一模型/同一数据下逐一隔离评估 RL4LLM 常用 trick，发现只需"优势归一化(group 均值 + batch 标准差) + token-level loss 聚合"两项极简组合(Lite PPO)即可稳定超过堆砌组件的 DAPO/GRPO——"简单胜过复杂"。 | ROLL，统一 baseline = PPO loss + REINFORCE 优势(critic-free) | 见正文 |
| `lp_reg` | 把低概率探索 token 称为"Reasoning Sparks"(如 wait/however/perhaps)，通过构造去噪代理分布 + 前向 KL 的三重门控正则，定向保护这些 spark 不被 GRPO 过度惩罚消除，从而对抗熵崩溃、在长训练区间持续 scaling 不崩;五数学基准均值 60.17%、较此前 +2.66%。 | verl，GRPO 基础 | 受限：README-only |
| `luffy` | 把更强策略(DeepSeek-R1)的 off-policy 推理轨迹与 on-policy rollout 混进同一 GRPO group 做组内归一化(Mixed-Policy GRPO)，再用 policy shaping f(x)=x/(x+0.1) 放大低概率关键动作的梯度、抑制熵坍塌;在弱模型/难数据上突破基座上界，六数学基准均值较此前 +6.4、三 OOD +6.2。 | veRL，rollout 用 vLLM | 完整/已克隆 |
| `magistral` | Mistral 自底向上、不依赖任何蒸馏轨迹，纯 RL(改造版 GRPO + 四维奖励整形)把 Mistral Medium 3 训成推理模型 Magistral Medium(AIME-24 pass@1 +近 50%)；并给出强制推理语言一致、纯文本 RL 不损甚至提升多模态/指令/函数调用的实证与失败实验。 | custom(自研异步在线 RL 系统) | 受限：权重/无训练代码 |
| `minimax_m1` | 首个开源权重的大规模混合注意力(MoE+lightning attention)推理模型，原生支持 1M 上下文、长生成 FLOPs 仅 DeepSeek-R1 的约 25%；核心算法创新 CISPO——裁剪重要性采样权重而非裁剪 token 更新，保留全部 token(尤其反思类低概率 token)的梯度，对 DAPO 实现 2× 加速。 | 自研 RL(借 lightning attention 高效 rollout)，推理支持 vLLM/Transforme | 受限：权重/无训练代码 |
| `negative_reinforce` | 把 RLVR 的二元学习信号拆成"奖励正确(PSR)/惩罚错误(NSR)"两条独立范式，发现仅惩罚错误的 NSR 在整个 Pass@k 谱(k 到 256)上一致超过 base、常追平甚至超过 PPO/GRPO；据此提出把正奖励下调权重 λ=0.1 的 W-REINFORCE。结论高度依赖强先验 backbone(Qwen)。 | veRL(vendored) | 完整/已克隆 |
| `nft` | 通过 Bayes 规则把生成策略拆为正/负策略，并用同一目标网络隐式参数化负策略，NFT 让纯监督学习也能从负样本中"自我反思"；理论上证明 on-policy 时 NFT 与 GRPO 梯度完全等价，实证上 7B 略超 DAPO、32B 与 DAPO 基本持平。 | VeRL（fork 自官方 DAPO 环境）+ FSDP + Ray | 完整/已克隆 |
| `on_policy_sft` | 把"高效推理"的带长度惩罚 RL 目标做原理性化简——去掉 KL 正则、去掉组内归一化、用最简的截断式长度惩罚——可证明 GRPO 目标退化为对"按正确性与简洁性过滤的自生成数据"做交叉熵 SFT；该极简 on-policy SFT 在五个数学基准上定义了 accuracy–efficiency Pareto 前沿。 | veRL + FSDP + vLLM | 完整/已克隆 |
| `one_shot_rlvr` | 仅用一条（甚至一条以内）训练样本做 RLVR(GRPO/PPO)，就能把 Qwen2.5-Math-1.5B 在 MATH500 从 36.0% 拉到 73.6%、六基准均值从 17.6% 升到 35.7%，基本匹配含该样本的 1.2k 子集；揭示 post-saturation generalization、跨类泛化、自反思增多等现象，支持"RLVR 主要是激发而非注入推理能力"。 | veRL(GRPO/PPO) + vLLM 0.6.3 + Qwen2.5-Math 评测 pipeline | 完整/已克隆 |
| `orz` | 用最朴素的 vanilla PPO + GAE(λ=γ=1) + 二值规则奖励、完全不带 KL，就能在 base 模型上稳定地大规模 scale up reasoning RL；首个把代码/数据/各尺寸权重乃至 critic 权重全部开源的 "Reasoner-Zero" 实现。 | OpenRLHF + vLLM + DeepSpeed + Ray | 完整/已克隆 |
| `prefix_rft` | 从 demonstration 采一段前缀作为 off-policy 引导、让当前策略续写作为 on-policy 探索，整条混合轨迹一起进 RFT；用前缀感知 advantage、熵约束裁剪与余弦衰减的前缀长度调度，把 SFT 的过程监督与 RFT 的目标导向优化在单阶段内融合。 | veRL（recipe/prefix_rft） | 完整/已克隆 |
| `prorl` | 用 KL 正则 + 周期性参考策略硬重置 + DAPO 解耦 clip/动态采样稳定住长程 RL，让 GRPO 训练能跑 2k+ 步而不熵坍缩，从而在 base 模型即使大量采样也无法触及的任务上发现新推理策略——主张 RL 确能扩展（而非仅放大）推理边界。 | veRL | 受限：权重/无训练代码 |
| `psft` | 把 SFT 视为 "advantage 恒为正(A=1)、采样自固定离线数据集" 的策略梯度特例，给它套上 PPO/TRPO 式信任域裁剪约束策略漂移；代价是 in-domain 略低于普通 SFT，换来更强 OOD 泛化、不熵坍缩、以及作为后续 RL 起点更优。 | veRL（recipe/psft） | 完整/已克隆 |
| `raft_reinforce_rej` |  | veRL(仓库内置 verl 源码,入口 `verl.trainer.main_ppo`)。FSDP + vLLM。提供 | 见正文 |
| `revisit_entropy` |  | veRL（Sheng et al. 2025），GRPO 训练。 | 完整/已克隆 |
| `rl_plus` | 针对"纯 on-policy RLVR 反而收窄基座可解问题集（capability boundary collapse）"现象，RL-PLUS 用 Multiple Importance Sampling 稳定吸收外部 off-policy 数据 + focal-style 探索优势放大低概率正确路径，并去掉 clip，使 Pass@k 突破基座天花板。 | VeRL + DeepScaleR | 完整/已克隆 |
| `rl_survey_lrm` | 系统综述 DeepSeek-R1 以来"RL for 大型推理模型(LRM)"的范式，提出"基础组件 / 五对开放争议 / 训练资源 / 下游应用 / 未来方向"的分类体系，明确 RLVR 是区别于 RLHF/DPO 的新 scaling 轴，并把 RL 推向 LRM/ASI 的可扩展性列为核心开放问题。 | 综述仓，无独立训练框架（梳理 OpenRLHF/veRL/AReaL/slime/TRL） | 受限：权重/无训练代码 |
| `scaf_grpo` | 针对 RLVR 的"学习悬崖"(难题持续零奖励→GRPO advantage 坍缩→梯度消失),只在学习停滞时按"知识→规划→解答"三层、增量注入 in-prompt 提示,让**当前策略自己**采样出成功轨迹来替换失败轨迹,从而在保持 on-policy 一致性、不破坏探索自主性的前提下攻克长尾难题。 | veRL 0.4.1.dev(vLLM rollout) | 完整/已克隆 |
| `sed_sft` | 在 SFT 的交叉熵之上，对"探索空间大"的 token 选择性加一个把 ground-truth 概率往 0.5 推的二次惩罚（被 Top-k 累积概率掩码门控），以保留生成多样性、为后续 RL 留探索空间；RL 后相对 CE 基线平均仅 +1.2~+2.06 点，增量有限且证据面窄。 | 自带 torch/transformers/DeepSpeed ZeRO-2 SFT trainer + verl GR | 完整/已克隆 |
| `simplerl_zoo` | 在 10 个跨家族/尺寸的 base 模型上做透明的 zero RL（仅正确性二值奖励、不用 format reward），系统拆解成败关键因素，并首次在 Qwen 家族外的小模型上观察到 verification 等认知行为涌现；核心是经验研究与配方/工具开源，而非新算法。 | veRL（仓内自带 verl/，GRPO），旧 v0 用 OpenRLHF+PPO（README 注明） | 完整/已克隆 |
| `skywork_or1` | 面向"已蒸馏的长 CoT 模型"的高效可扩展 RL 配方（基于改造版 GRPO，命名 MAGIC），系统消融各组件并深入研究过早熵坍缩，论证"缓解过早熵坍缩对提升测试性能至关重要"；32B/7B 在 AIME+LiveCodeBench 平均分别 +15.0/+13.9，全开源。 | veRL 定制 fork（仓内自带 verl/ + or1_scripts/ + or1_data/，含 Math/Co | 完整/已克隆 |
| `spurious_rewards` | 在 Qwen2.5-Math 上，即使用随机/格式/错误标签等"虚假奖励"做 RLVR 也能大幅涨分（随机奖励 MATH-500 +21.4%，接近真值的 +29.1%），原因是 GRPO 的裁剪偏置放大了 base model 已有的高先验行为（如 code reasoning），而非注入新能力；该效应在 Llama3/OLMo2 等模型族上几乎消失。 | **OpenRLHF**（经 PRIME-RL 的 TTRL 改编；论文正文亦自述"沿用流行 RL 框架 OpenRLH | 完整/已克隆 |
| `srft` | 用熵感知（entropy-aware）的自适应权重，在单阶段内同时对同一模型施加 SFT（demonstration）与 RL（自探索 rollout），按当前策略熵动态平衡模仿与探索；建立在 LUFFY 之上，平均 59.1%、较 zero-RL 在 5 个数学基准 +9.0%、3 个 OOD 基准 +10.9%。 | veRL + vLLM（底座沿用 LUFFY 的 mix_src 结构与 deepscaler 奖励） | 完整/已克隆 |
| `superrl` | 在 RLVR 流程内做实例级自适应回退——某 prompt 的所有 rollout 都拿到零奖励（无 PG 梯度信号）时，就在该 prompt 上回退到高质量离线示范（tagged_answer）做 SFT，否则走标准 GRPO/PPO；二选一、不做损失融合，专治稀疏奖励下 rollout 全失败的"无梯度"困境。 | veRL（volcengine，v0.5.0）+ FSDP + 梯度检查点 | 见正文 |
| `trapo` | 在每个训练实例内细粒度交织 SFT 与 RL——只对专家轨迹**前缀**做 SFT、其后由目标策略自行 rollout 补全做 RL；用 Trust-Region SFT（把 SFT 权重 1/p_θ 改为 1/max(p_θ,α)）把 forward-KL 的 mode-covering 转成 reverse-KL 式 mode-seeking、稳住 RL 起点，再用 micro-group 采样按累计回报自适应分配前缀长度。 | verl 的 GRPO（去 KL penalty）+ 自定义 TrSFT | 完整/已克隆 |
| `trl_v1` | TRL v1.0 是 HuggingFace 的统一 LLM 后训练库，用"稳定/实验分级 + 低抽象"哲学把 SFT/RM/偏好优化/RLVR/蒸馏 75+ 方法收进一个库；其 on-policy distillation trainer（GKD/GOLD/MiniLLM/SDFT/SDPO，位于 `trl.experimental.*`）是本调研多篇 OPD 论文（OPSD、AVSD、distillm 等）的底层基础设施。 | TRL 本身即框架（基于 Transformers/Accelerate/PEFT，可选 vLLM/DeepSpeed） | 完整/已克隆 |

#### T4 思维链-Token 级 / MTP

| Key | 一句话重点 (TL;DR) | 框架 | 代码情况 |
|---|---|---|---|
| `beyond_loglik` | SFT 默认用 NLL（−log p），但它在"从零训练分类"才最优；后训练时基座已有先验。本文把 NLL 推广成参数族 f_α(p)=(1−p^α)/α，并提出一个统一刻画——**模型能力连续谱**：基座先验强（如数学）时，**下调低概率 token 的 prior-leaning 目标**（如 −p）持续胜过 NLL；基座先验弱（如 figfont 谜题）时 NLL 主导；中间区两者难分。 | VeRL（`main_verl`） | 见正文 |
| `llm_future_mtp` | 实证发现自回归 LLM 给 prompt 附占位 token 后，正确的未来 token 已落在 top-200 logits 内；用 mask token + gated LoRA + sampler 把这种隐知识显式化为并行多 token 生成，代码/数学近 5×、对话/知识近 2.5× 加速且无质量损失。注意是纯**推理加速(speculative)**工作，非推理质量/蒸馏。 | gated LoRA + 两层 MLP sampler，基模 Tulu3-8B，微调用 Tulu3 数据集 | 受限：404/不可得 |
| `rho1` |  | - **重要更正(已核)**:官方 microsoft/rho 仓 **只发布模型权重 + 评测代码**——`rho-1 | 受限：404/不可得 |
| `segment_attrib` | 用 integrated gradients 直接量化每个 token 对"正确答案预测"的贡献，聚合为段落级"强度 + 方向一致性"两指标，挑出"高强度但中等一致性"的反思性段落做 loss-mask 选择性 SFT；相对 full-CoT SFT 准确率最高 +4.7%、长度最多 −18%，但温度采样下增益明显收窄，框架本身沿用 Rho-1 selective SFT。 | SFT 用 unsloth+trl，归因阶段自写 IG 脚本，评测含 latex2sympy | 完整/已克隆 |
| `srgen` | 零训练的测试时方法，在解码过程中用动态熵阈值识别"高熵 critical token"，在该处短暂暂停、在线优化一个瞬态修正向量 δ 注入 hidden state 再发射下一 token，实现"主动错误预防"；数学/AIME 类任务增益显著，但在 AMC/GPQA/EvalPlus 上增益常仅 +0.2~+2.8pp。 | + evaluator + server，可跑） | 完整/已克隆 |
| `sstoken` | SFT 的 token 级选择方法，用"当前模型 vs 其历史模型"的 Retrospective Excess Loss（REL）替代外部参考模型，再融合一个基于注意力的语义重要性分，按比例 ρ 保留 top-ρ token 计 loss；四基座平均分最优，但增量温和、主要在需指令遵循的 QA 任务见效。 | 自写训练脚本 + FSDP（支持 LoRA）+ lm-evaluation-harness 评测 | 完整/已克隆 |
| `vcore` | 把长 CoT SFT 的 token 加权形式化为"单步 SGD 下使期望 loss 下降最大、且加权分布与均匀分布 KL ≤ δ"的约束优化，闭式解为 Gibbs 分布 q\*∝exp(τ·gradient-utility)，配一个 one-backward 探针估 utility + 方差控制系数 α 稳训练；不依赖 teacher 引导/置信阈值/熵过滤，在中小模型与综合均值上稳定优于 SFT/DFT/iw-SFT。 | LLaMA-Factory + 定制 transformers 4.52.4 | 完整/已克隆 |

### 1.4 深读论文总表

| Key | 标题 | arXiv | 机构 | 主题 | 框架 | 代码情况 |
|---|---|---|---|---|---|---|
| `adaspec` | AdaSPEC: Selective Knowledge Distillation for Efficient Speculative Decoders | arXiv 2510.19779 | UC Berkeley、清华、Georgia Tech | T1 | HuggingFace transformers/TRL/Accelerate/ | 见正文 |
| `aligndistil` | AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation | arXiv 2503.02832v3 | 北京交通大学 + 腾讯 | T1 | OpenRLHF v0.5.2.post2（+ vLLM for on-poli | 完整 |
| `apple_ssd` | SSD: Embarrassingly Simple Self-Distillation Improves Code Generation | arXiv 2604.01193v1 | Apple | T1 | 仅需 采样 + 标准 SFT(交叉熵) + 评估时单独调温；无 RL/verif | 部分 |
| `avsd` | AVSD: Adaptive-View Self-Distillation by Balancing Consensus and Teacher-Specific Privileged Signals | arXiv 2605.20643v1 | — | T1 | DeepSpeed ZeRO-2 + HF Accelerate + PEFT( | 完整 |
| `brts` | On-Policy Distillation with Best-of-N Teacher Rollout Selection (BRTS) | arXiv 2605.09725v2 | JHU + TikTok + UCSD + 复旦 | T1 | veRL（与 thunlp/OPD 同源 fork） | 完整 |
| `caopd` | The Illusion of Certainty: Decoupling Capability and Calibration in On-Policy Distillation (CaOPD) | arXiv 2604.16830 | Salesforce AI Research | T1 | TRL 0.24 + vLLM 0.12 + DeepSpeed 0.18 +  | 完整 |
| `copsd` | Crosslingual On-Policy Self-Distillation for Multilingual Reasoning (COPSD) | arXiv 2605.09548 | — | T1 | HuggingFace TRL + Accelerate/DeepSpeed，L | 完整 |
| `csd` | Distillation of Large Language Models via Concrete Score Matching (Concrete Score Distillation, CSD) | arXiv 2509.25837 | KAIST + summary.ai | T1 | 即插式 KD 目标，可嵌入 ImitKD/GKD/DistiLLM | 受限:未释出 |
| `dasd` | Distribution-Aligned Sequence Distillation (DASD-4B-Thinking) | arXiv 2601.09088v1 | 阿里云 | T1 | LLaMA-Factory + DeepSpeed ZeRO-3 + Liger | 见正文 |
| `distillm` | DistiLLM: Towards Streamlined Distillation for Large Language Models | arXiv 2402.03898 | KAIST AI+ Microsoft | T1 | 自研（沿用 MiniLLM 代码基 + 定制 HF Transformers + | 完整 |
| `distillm2` | DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs | arXiv 2503.07067 | KAIST AI+ Microsoft | T1 | HF alignment-handbook + Accelerate + Dee | 完整 |
| `gad` | Black-Box On-Policy Distillation of Large Language Models (GAD) | arXiv 2511.10643 | — | T1 | veRL（GRPO，hack critic 当判别器） | 完整 |
| `gemma2` | Gemma 2: Improving Open Language Models at a Practical Size | arXiv 2408.00118 | Gemma Team, Google DeepMind | T1 | 内部 Google JAX + TPU 栈 | 受限:权重 |
| `gkd` | On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD) | arXiv 2306.13649 | Google DeepMind | T1 | T5/JAX（论文）、TRL（社区集成） | 见正文 |
| `glm45` | GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models | arXiv 2508.06471 | Zhipu AI 智谱 & 清华 | T1 | **Slime** 单独开源 https://github.com/THUDM/ | 受限:权重 |
| `gopd` | Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation (G-OPD / ExOPD) | arXiv 2602.12125 | 中国人民大学高瓴 + 腾讯 LLM 部 | T1 | **veRL v0.6.1** | 完整 |
| `hpd` | Hybrid Policy Distillation for LLMs (HPD) | arXiv 2604.20244v1 | 上海交大 / 上海创智学院 / 腾讯，通讯 Rui Wang、Ruobing X | T1 | LlamaFactory（SFT 路径）+ veRL（RL 路径） | 完整 |
| `lightning_opd` | Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation | arXiv 2604.13010 | NVIDIA | T1 | slime + slime_plugins（SFT 用 LLaMA-Factor | 完整 |
| `lightreasoner` | LightReasoner: Can Small Language Models Teach Large Language Models Reasoning? | arXiv 2510.07962 | 香港大学 / 芝加哥大学 | T1 | 自实现轻量 Python 流水线(非 verl/TRL) | 完整 |
| `madopd` | MAD-OPD: Breaking the Ceiling in On-Policy Distillation via Multi-Agent Debate | arXiv 2605.01347 | 华中科技大 + 阿里巴巴；Yong Xie、Qianglong Chen通讯 | T1 | TRL(0.15–0.25) + vLLM + DeepSpeed ZeRO-3 | 完整 |
| `minillm` | MiniLLM: On-Policy Knowledge Distillation of Large Language Models | arXiv 2306.08543 | 清华 CoAI + 微软研究院，Minlie Huang 通讯 | T1 | 自研(改版 HF Transformers + DeepSpeed + Acce | 完整 |
| `nemotron_cascade2` | Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation | arXiv 2603.19220 | NVIDIA | T1 | Nemo-RL | 受限:权重 |
| `nemotron_nano2` | NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model | arXiv 2508.14444 | NVIDIA | T1 | NeMo / Megatron | 受限:权重 |
| `opd_blog` | On-Policy Distillation (Thinking Machines Lab 博客 + tinker-cookbook 配方) | — | Thinking Machines Lab | T1 | Tinker SDK（自研托管训练 SDK，基于 LoRA），非 veRL/TR | 完整 |
| `opd_survey` | A Survey of On-Policy Distillation for Large Language Models | arXiv 2604.00626 | 腾讯大语言模型部 | T1 | n/a（综述） | 受限:权重 |
| `ophsd` | Training with Harnesses: On-Policy Harness Self-Distillation for Complex Reasoning | arXiv 2605.08741 | Peking University | T1 | veRL（子类化 PPO trainer,base 为 OPSD repo） | 受限:README |
| `opsa` | Reducing the Safety Tax in LLM Safety Alignment with On-Policy Self-Distillation | arXiv 2605.15239 | UC Riverside / ICSI / Microsoft / Berkel | T1 | NVIDIA NeMo-RL 扩展 fork | 完整 |
| `opsd` | Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models | arXiv 2601.18734 | UCLA / HKU / Meta Superintelligence Labs | T1 | TRL（基于其 experimental GOLD/GKD trainer） | 完整 |
| `prism` | Beyond SFT-to-RL: Pre-alignment via Black-Box On-Policy Distillation for Multimodal RL (PRISM) | arXiv 2604.28123 | HKUST + 清华 + 南洋理工 + 人大 + 中科大 + 国科大 | T1 | 三阶段：LLaMA-Factory（SFT）+ verl（对齐/RLVR）+ v | 完整 |
| `qwen3` | Qwen3 Technical Report | arXiv 2505.09388 | # qwen3 — Qwen3 Technical Report

- **ar | T1 | 〔见正文〕 | 受限:权重 |
| `resd` | Learning with Rare Success but Rich Feedback via Reflection-Enhanced Self-Distillation (RESD) | arXiv 2605.12741v1 | — | T1 | 〔见正文〕 | 部分 |
| `rethink_opd` | Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe | arXiv 2604.13016 | # rethink_opd — Rethinking On-Policy Dis | T1 | 〔见正文〕 | 完整 |
| `rock_tokens` | Cornerstones or Stumbling Blocks? Deciphering the Rock Tokens in On-Policy Distillation | arXiv 2605.09253 | — | T1 | 自定义 KDFlow（SGLang+Ray+FSDP2+bf16） | 完整 |
| `rosd` | ROSD: Reflective On-Policy Self-Distillation for Language Model Reasoning across Domains | arXiv 2605.28014 | 香港理工 / 百度 / 山东大学 / 莱顿大学 | T1 | verl（扩展自 SDPO/lasgroup/SDPO）+ FSDP actor | 完整 |
| `safesteer` | SafeSteer: Localized On-Policy Distillation for Efficient Safety Alignment | arXiv 2606.02530 | **元信息**：arXiv 2606.02530 | T1 | 自研轻量安全对齐框架（activation steering teacher + | 完整 |
| `scope` | SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting | arXiv 2604.10688 | 中科大/美团 LongCat/南大/复旦/华科 | T1 | veRL(volcengine/verl) | 完整 |
| `sdcl` | Self-Distillation Enables Continual Learning (SDFT) | arXiv 2601.19897 | MIT + Improbable AI Lab + ETH Zurich | T1 | TRL 0.24.0 | 见正文 |
| `tip` | TIP: Token Importance in On-Policy Distillation | arXiv 2604.14084 | — | T1 | verl | 部分 |
| `unisd` | UniSD: Towards a Unified Self-Distillation Framework for Large Language Models | arXiv 2605.06597 | — | T1 | + 组件级消融，OPD 相关性中等（偏经验综述/工程整合而非单一新机制） | 完整 |
| `vla_opd` | VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models via On-Policy Distillation | arXiv 2603.26666v1 | — | T1 | 〔待核：代码未放出，下述基于论文〕基于 GRPO 式分组采样、teacher=S | 受限:未释出 |
| `why_sd_degrade` | Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs? | arXiv 2603.24472v3 | Microsoft Research + KAIST + SNU | T1 | veRL（仓库含 `verl/` 目录、megatron/sglang 脚本） | 完整 |
| `chain_of_agents` | Chain-of-Agents: End-to-End Agent Foundation Models via Multi-Agent Distillation and Agentic RL | arXiv 2508.13167 | OPPO AI Agent Team | T2 | SFT 用 LLaMA-Factory、RL 用 veRL 跑 DAPO | 见正文 |
| `deepdive` | DeepDive: Advancing Deep Search Agents with Knowledge Graphs and Multi-Turn RL | arXiv 2509.10446 | 清华 / Z.AI / 东北大学 | T2 | slime（RL）+ 多轮 GRPO | 见正文 |
| `distill_agent_tools` | Distilling LLM Agents into Small Models with Retrieval and Code Tools (Agent Distillation) | arXiv 2505.17612 | KAIST / DeepAuto.ai、UW-Madison、KRAFTON | T2 | smolagents v1.13.0.dev0 + TRL SFT traine | 完整 |
| `gigpo` | Group-in-Group Policy Optimization for LLM Agent Training (GiGPO) | arXiv 2505.10978 | 南洋理工 NTU / Skywork AI Singapore | T2 | verl-agent（veRL 扩展） | 完整 |
| `icrl` | ICRL: Learning to Internalize Self-Critique with Reinforcement Learning | arXiv 2605.15224v1 | 港科大、南京大学、中山大学、NUS、NTU、SAP、Microsoft Rese | T2 | slime (THUDM, SGLang-native RL) + AgentG | 见正文 |
| `latent_agents` | Latent Agents: A Post-Training Procedure for Internalized Multi-Agent Debate | arXiv 2604.24881v1 | Boston University | T2 | TRL (GRPOTrainer) + PEFT/LoRA | 见正文 |
| `meow_tea_taro` | A Practitioner's Guide to Multi-turn Agentic Reinforcement Learning | arXiv 2510.01132 | UC San Diego；Ruiyi Wang, Prithviraj Amma | T2 | 封装/vendoring veRL | 完整 |
| `open_agentrl` | RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System (Open-AgentRL) | arXiv 2602.02488 | — | T2 | veRL（仓库内置 `verl/`） | 完整 |
| `openclaw_rl` | OpenClaw-RL: Train Any Agent Simply by Talking | arXiv 2603.10165 | — | T2 | slime 异步 RL（Megatron 训练 + SGLang serving | 完整 |
| `pi_play` | π-Play: Multi-Agent Self-Play via Privileged Self-Distillation without External Data | arXiv 2604.14054 | — | T2 | 论文未绑定特定开源框架，student 用 GRPO | 受限:未释出 |
| `rstar2` | rStar2-Agent: Agentic Reasoning Technical Report | arXiv 2508.20722 | Microsoft Research | T2 | veRL v0.5 + Code Judge + vLLM | 完整 |
| `score` | From Correction to Mastery: Reinforced Distillation of LLM Agents (SCoRe) | arXiv 2509.14257 | 中科大 USTC + Independent Researcher | T2 | 自定义工具链(LLaMA-Factory + LangGraph + veRL, | 见正文 |
| `search_dont_guess` | Search, Do not Guess: Teaching Small Language Models to Be Effective Search Agents | arXiv 2604.04651 | **元信息**：arXiv 2604.04651 | T2 | 检索 E5+BM25 / Search-o1 式迭代 agent / 训练 SF | 见正文 |
| `search_r1` | Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning | arXiv 2503.09516 | UIUC+ UMass Amherst+ Google Cloud AI Res | T2 | veRL 定制 fork | 完整 |
| `sod_stepwise` | SOD: Step-wise On-policy Distillation for Small Language Model Agents | arXiv 2605.07725v1 | 浙大 + 腾讯 LLM 部 + 中科大 + 新加坡国立 | T2 | veRL fork + Open-AgentRL + ReTool agenti | 完整 |
| `spear` | SPEAR: Learn the Ropes, Then Trust the Wins — Self-imitation with Progressive Exploration for Agentic RL | arXiv 2509.22601v4 | 腾讯优图 Youtu-Agent Team | T2 | veRL + verl-agent | 完整 |
| `sweet_rl` | SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks | arXiv 2503.15478 | FAIR at Meta + UC Berkeley | T2 | OpenRLHF 定制 fork（`YifeiZhou02/collab_ope | 见正文 |
| `tcod` | TCOD: Exploring Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents | arXiv 2604.24005 | Tongyi Lab, Alibaba Group + 香港中文大学 | T2 | Trinity-RFT（Ray-based RFT，Alibaba） | 完整 |
| `webagent_r1` | WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning | arXiv 2505.16421v2 | — | T2 | **verifiers**（willccbb/verifiers，GRPO-ba | 完整 |
| `amft` | AMFT: Aligning LLM Reasoners by Meta-Learning the Optimal Imitation-Exploration Balance | arXiv 2508.06944v1 | 清华大学电子工程系 | T3 | 〔待核，仓库不可得〕方法层面建立在 GRPO(RLVR)+SFT 加权损失之上 | 受限:404 |
| `ampo` | AMPO: Adaptive Multi-Guidance Policy Optimization for Diverse Exploration | arXiv 2510.02227v2 | 同济、香港理工、上海 AI Lab、新加坡国立、电子科技大 | T3 | verl(GRPO 的 Mixed-Policy 扩展, FSDP 或 Mega | 完整 |
| `asft` | ASFT: Anchored Supervised Fine-Tuning | arXiv 2509.23753v3 | 南方科技大学、北京大学、上海 AI Lab | T3 | 自定义 HF Trainer + DeepSpeed(ZeRO-2/3) + P | 完整 |
| `bapo` | BAPO: Stabilizing Off-Policy Reinforcement Learning for LLMs via Balanced Policy Optimization with Adaptive Clipping | arXiv 2510.18927v1 | 复旦 FudanNLP / 上海稷迹智锋 / 上海创新研究院 | T3 | veRL（GRPO 为基础算法） | 完整 |
| `behavior_priming` | Beneficial Reasoning Behaviors in Agentic Search and Effective Post-training to Obtain Them | arXiv 2510.06534v3 | **元信息**：arXiv:2510.06534v3 | T3 | SFT 用 LLaMA-Factory、RL 用 GRPO（veRL 系） | 受限:404 |
| `ca_survey` | From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models | arXiv 2604.09459v2 | — | T3 | ，可用于定位、选 baseline、找相邻工作 | 受限:404 |
| `cbrl` | Context Bootstrapped Reinforcement Learning (CBRL) | arXiv 2603.18953 | UC Santa Barbara + Cisco Research | T3 | verl 0.3.0.post2 + vLLM 0.8.5 + PyTorch  | 完整 |
| `cepo` | CEPO: RLVR Self-Distillation using Contrastive Evidence Policy Optimization | arXiv 2605.19436 | MBZUAI + Linköping University + Australi | T3 | veRL（经由 EasyR1）+ FSDP + vLLM | 完整 |
| `chord` | On-Policy RL Meets Off-Policy Experts: Harmonizing SFT and RL via Dynamic Weighting (CHORD) | arXiv 2508.11408 | 阿里巴巴集团 | T3 | Trinity-RFT（基于 veRL 后端 + Ray） | 见正文 |
| `dapo` | DAPO: An Open-Source LLM Reinforcement Learning System at Scale | arXiv 2503.14476 | ByteDance Seed + 清华 AIR+ 港大；通讯 Hao Zhou、 | T3 | veRL（vllm==0.8.3、ray[serve]） | 见正文 |
| `deepseek_r1` | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | arXiv 2501.12948 | DeepSeek-AI | T3 | 自研高性能 RL 框架（附录 B.1） | 受限:权重 |
| `deepseekmath` | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models（GRPO 起源） | arXiv 2402.03300 | DeepSeek-AI | T3 | custom/none（DeepSeek 内部实现） | 受限:权重 |
| `denoiserl` | DenoiseRL: Bootstrapping Reasoning Models to Recover from Noisy Prefixes | arXiv 2605.28421 | 复旦大学 / 上海创智学院 ；Caijun Xu, Changyi Xiao,  | T3 | VeRL，RL backbone 用 GRPO / DAPO | 完整 |
| `dft_reweight` | On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification (DFT / Dynamic Fine-Tuning) | arXiv 2508.05629 | 东南大学、UCLA、上海交大、南洋理工、UC Berkeley、武汉大学、UC  | T3 | veRL（FSDP SFT 训练器 + Liger），评测沿用 Qwen2.5- | 完整 |
| `dr_grpo` | Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO) | arXiv 2503.20783 | Sea AI Lab、新加坡国立大学、新加坡管理大学；Zichen Liu, C | T3 | Oat） | 完整 |
| `eaft` | Entropy-Adaptive Fine-Tuning: Resolving Confident Conflicts to Mitigate Forgetting (EAFT) | arXiv 2601.02151 | 北京邮电大学 PRIS-CV / 中关村学院 | T3 | LLaMA-Factory（主）+ ms-swift | 受限:README |
| `entropy_mechanism` | The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models | arXiv 2505.22617 | 上海 AI Lab / 清华 / UIUC / 北大 / 南大 / CUHK | T3 | veRL（fork 自 DAPO recipe） | 完整 |
| `fest` | FEST: Boosting RLVR via Randomly Selected Few-Shot Guidance | arXiv 2605.15012 | — | T3 | VeRL（GRPO 为主，few-shot 用 semi-online DPO） | 见正文 |
| `gmpo` | Geometric-Mean Policy Optimization (GMPO) | arXiv 2507.20673 | UCAS / CUHK / HKUST / Microsoft Research | T3 | 专用仓基于 **Oat + vLLM 0.8.4**（构建于 understan | 部分 |
| `gspo` | Group Sequence Policy Optimization (GSPO) | arXiv 2507.18071 | Qwen Team, Alibaba | T3 | Qwen 内部 RL（训练 Megatron + 推理 SGLang/vLLM） | 见正文 |
| `hapo` | Heterogeneous Adaptive Policy Optimization: Tailoring Optimization to Every Token's Nature | arXiv 2509.16591 | 北京大学 / 上海 AI Lab / 北航 | T3 | **verl + vLLM** | 完整 |
| `holderpo` | Hölder Policy Optimisation | arXiv 2605.12058v2 | — | T3 | ，与 OPD/MTP 关联较弱（纯 RL 聚合层面，可借鉴 token 加权视角 | 完整 |
| `hpt_upge` | Towards a Unified View of Large Language Model Post-Training (UPGE / HPT) | arXiv 2509.04419 | 清华大学 C3I | T3 | veRL + LUFFY 的 mix_src 扩展（FSDP + vLLM ro | 见正文 |
| `limit_rlvr` | Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? | arXiv 2504.13837 | 清华大学 LeapLab、上海交通大学 | T3 | pass@k 评测(vLLM 多采样)，被评模型来自 SimpleRL-Zoo/ | 完整 |
| `lite_ppo` | Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning (Lite PPO) | arXiv 2508.08221 | Alibaba 联合 北交大、HKUST、南大、北大、Mila 等 | T3 | ROLL，统一 baseline = PPO loss + REINFORCE  | 见正文 |
| `lp_reg` | Low-probability Tokens Sustain Exploration in Reinforcement Learning with Verifiable Reward (Lp-Reg) | arXiv 2510.03222 | 腾讯 LLM Department | T3 | verl，GRPO 基础 | 受限:README |
| `luffy` | Learning to Reason under Off-Policy Guidance (LUFFY) | arXiv 2504.14945 | 上海 AI 实验室、西湖大学、南京大学、香港中文大学 | T3 | veRL，rollout 用 vLLM | 完整 |
| `magistral` | Magistral（Mistral 首个推理模型与自建可扩展 RLVR 流水线） | arXiv 2506.10910 | Mistral AI | T3 | custom(自研异步在线 RL 系统) | 受限:权重 |
| `minimax_m1` | MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention | arXiv 2506.13585 | MiniMax | T3 | 自研 RL(借 lightning attention 高效 rollout)， | 受限:权重 |
| `negative_reinforce` | The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning | arXiv 2506.01347 | University of Virginia / Princeton Langu | T3 | veRL(vendored) | 完整 |
| `nft` | NFT: Bridging Supervised Learning and Reinforcement Learning in Math Reasoning | arXiv 2505.18116 | Tsinghua / NVIDIA / UIUC / Stanford | T3 | VeRL（fork 自官方 DAPO 环境）+ FSDP + Ray | 完整 |
| `on_policy_sft` | On-Policy Supervised Fine-Tuning for Efficient Reasoning | arXiv 2602.13407 | 东方理工 + 香港理工 / Paris Dauphine-PSL / 上海交大  | T3 | veRL + FSDP + vLLM | 完整 |
| `one_shot_rlvr` | Reinforcement Learning for Reasoning in Large Language Models with One Training Example | arXiv 2504.20571 | UW / USC / Microsoft / UC Santa Cruz / G | T3 | veRL(GRPO/PPO) + vLLM 0.6.3 + Qwen2.5-Ma | 完整 |
| `orz` | Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model | arXiv 2503.24290 | StepFun + 清华 | T3 | OpenRLHF + vLLM + DeepSpeed + Ray | 完整 |
| `prefix_rft` | Prefix-RFT: Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling | arXiv 2507.01679 | 爱丁堡 ILCC + 复旦 + 阿里 Qwen + StepFun + UvA  | T3 | veRL（recipe/prefix_rft） | 完整 |
| `prorl` | ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in LLMs | arXiv 2505.24864 | NVIDIA | T3 | veRL | 受限:权重 |
| `psft` | Proximal Supervised Fine-Tuning (PSFT) | arXiv 2508.17784 | 上海交大 + 上海创智学院 + 腾讯大模型部 + 澳门大学 | T3 | veRL（recipe/psft） | 完整 |
| `raft_reinforce_rej` | A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce (RAFT / RAFT++ / Reinforce-rej) | arXiv 2504.11343 | # raft_reinforce_rej — A Minimalist Appr | T3 | 〔见正文〕 | 见正文 |
| `revisit_entropy` | Revisiting Entropy in Reinforcement Learning for Large Reasoning Models | arXiv 2511.05993 | # revisit_entropy — Revisiting Entropy i | T3 | 〔见正文〕 | 完整 |
| `rl_plus` | RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization | arXiv 2508.00222 | 北京大学 / 阿里通义实验室 / University of Alberta | T3 | VeRL + DeepScaleR | 完整 |
| `rl_survey_lrm` | A Survey of Reinforcement Learning for Large Reasoning Models | arXiv 2509.08827 | 清华 / 上海AI Lab / 上海交大 / 北大 / 中科大 / 哈工大 /  | T3 | 综述仓，无独立训练框架（梳理 OpenRLHF/veRL/AReaL/slime | 受限:权重 |
| `scaf_grpo` | Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning | arXiv 2510.19807 | 代码 github.com/JIA-Lab-research/Scaf-GRPO | T3 | veRL 0.4.1.dev(vLLM rollout) | 完整 |
| `sed_sft` | SED-SFT: Selectively Encouraging Diversity in Supervised Fine-Tuning | arXiv 2602.07464v1 | 腾讯 WeChat AI | T3 | 自带 torch/transformers/DeepSpeed ZeRO-2 S | 完整 |
| `simplerl_zoo` | SimpleRL-Zoo: Investigating and Taming Zero RL for Open Base Models in the Wild | arXiv 2503.18892v3 | — | T3 | veRL（仓内自带 verl/，GRPO），旧 v0 用 OpenRLHF+PP | 完整 |
| `skywork_or1` | Skywork Open Reasoner 1 Technical Report | arXiv 2505.22312v2 | Skywork AI，Jujie He*、Jiacai Liu* 等，通讯 Ju | T3 | veRL 定制 fork（仓内自带 verl/ + or1_scripts/ + | 完整 |
| `spurious_rewards` | Spurious Rewards: Rethinking Training Signals in RLVR | arXiv 2506.10947 | University of Washington / AI2 / UC Berk | T3 | **OpenRLHF**（经 PRIME-RL 的 TTRL 改编；论文正文亦自 | 完整 |
| `srft` | SRFT: A Single-Stage Method with Supervised and Reinforcement Fine-Tuning for Reasoning | arXiv 2506.19767 | — | T3 | veRL + vLLM（底座沿用 LUFFY 的 mix_src 结构与 dee | 完整 |
| `superrl` | SuperRL: Reinforcement Learning with Supervision to Boost Language Model Reasoning | arXiv 2506.01096 | 北京大学 / UIUC / 微软 | T3 | veRL（volcengine，v0.5.0）+ FSDP + 梯度检查点 | 见正文 |
| `trapo` | TRAPO: Trust-Region Adaptive Policy Optimization | arXiv 2512.17636v1 | 清华大学 CoAI 组 + Ant Group | T3 | verl 的 GRPO（去 KL penalty）+ 自定义 TrSFT | 完整 |
| `trl_v1` | TRL v1.0: 统一后训练栈（SFT, RM, DPO, GRPO, GKD, MiniLLM 等） | — | — | T3 | TRL 本身即框架（基于 Transformers/Accelerate/PEF | 完整 |
| `beyond_loglik` | Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum | arXiv 2510.00526v3 | — | T4 | VeRL（`main_verl`） | 见正文 |
| `llm_future_mtp` | Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential | arXiv 2507.11851v1 | Apple | T4 | gated LoRA + 两层 MLP sampler，基模 Tulu3-8B， | 受限:404 |
| `rho1` | Rho-1: Not All Tokens Are What You Need | arXiv 2404.07965v4 | # rho1 — Rho-1: Not All Tokens Are What  | T4 | 〔见正文〕 | 受限:404 |
| `segment_attrib` | Segment-Level Attribution for Selective Learning of Long Reasoning Traces | arXiv 2602.00425v1 | University of Southern California | T4 | SFT 用 unsloth+trl，归因阶段自写 IG 脚本，评测含 latex | 完整 |
| `srgen` | Self-Reflective Generation at Test Time (SRGen) | arXiv 2510.02919 | 框架 HuggingFace Transformers 即插即用 | T4 | + evaluator + server，可跑） | 完整 |
| `sstoken` | ssToken: Self-modulated and Semantic-aware Token Selection for LLM Fine-tuning | arXiv 2510.18250 | 上海交大 & 上海创智学院 | T4 | 自写训练脚本 + FSDP（支持 LoRA）+ lm-evaluation-ha | 完整 |
| `vcore` | VCORE: Variance-Controlled Optimization-based Reweighting for Chain-of-Thought Supervision | arXiv 2510.27462 | 上海交通大学 & 香港中文大学 | T4 | LLaMA-Factory + 定制 transformers 4.52.4 | 完整 |

---

## 2. T1 — OPD 核心 / 自蒸馏（41 篇）

本部分逐篇内联与 OPD 主线最直接相关的工作：白盒/黑盒 token 级在线策略蒸馏（GKD/MiniLLM/DistiLLM/DistiLLM-2/GAD）、自蒸馏 OPSD/SDPO 家族（opsd/copsd/opsa/sdcl/rosd/resd/scope/ophsd/avsd/why_sd_degrade）、OPD 机理与配方（rethink_opd/lightning_opd/rock_tokens/tip/madopd/hpd/caopd）、外推与多专家融合（gopd）、对齐视角（aligndistil/safesteer）、离线 logit KD（csd/adaspec）、大厂技报中的蒸馏实践（qwen3/gemma2/glm45/nemotron_*）、对照（apple_ssd/dasd/lightreasoner/brts/prism/unisd/vla_opd），以及方法论博客与综述（opd_blog/opd_survey）。每篇为完整 12 节，标题已降级。

### adaspec — AdaSPEC: Selective Knowledge Distillation for Efficient Speculative Decoders

> **一句话重点 (TL;DR)**：投机解码里给小 draft 模型做蒸馏时，AdaSPEC 不再"所有 token 一视同仁地对齐大模型"，而是先用一个参考模型估出"哪些 token 对 draft 真的学得动"，只在这部分 easy token 上蒸馏，把有限容量花在刀刃上，从而把 token 接受率(acceptance rate)最高提升约 15%。值得看的点：它把"蒸馏目标(最小化全 token KL)与真实目标(最大化接受率)错位"这件事讲得很清楚，并给出一个极简的选择性过滤解法。

**元信息**：arXiv 2510.19779 ｜ UC Berkeley、清华、Georgia Tech（Yuezhou Hu*、Jiaxin Guo* 共一，实习于 Georgia Tech 完成）｜ NeurIPS 2025 (Spotlight)，arXiv v1 2025-10-22 ｜ 主题 投机解码 draft 蒸馏 / 与本项目**外围相关** ｜ 代码 https://github.com/yuezhouhu/adaspec（**受限**：仓库仅含 README/LICENSE/图，无任何训练代码，核心 loss 仅见论文附录 Listing 2）｜ 框架 HuggingFace transformers/TRL/Accelerate/DeepSpeed（两阶段选择性 KD）

#### 1. 相关工作与进展(这条线现在做到哪一步)
投机解码(speculative decoding, SD)是当下主流的无损推理加速范式：用一个小 draft 模型"投机"地一次性生成多个 token，再用大 target 模型并行验证、接受或回退。加速倍率直接取决于 draft 与 target 的"对齐度"——draft 提的 token 越常被接受，跳得越多。为提升对齐，社区普遍给 draft 做知识蒸馏(KD)，代表方法是 **DistillSpec**（forward-KL，让 draft 分布逼近 target）。更先进的结构化方案如 **EAGLE** 改进了 draft 的特征复用。AdaSPEC 站在 DistillSpec 这条"用 KD 增强 draft-target 对齐"的线上，并声称可与 EAGLE 等叠加。

#### 2. 现有工作存在的问题(本文针对的痛点)
- **目标错位**：常规 KD 在**所有 token** 上最小化 draft↔target 的 KL，但 SD 的真实目标是**最大化接受率**。低 KL 不等于高接受率。
- **容量浪费**：draft 模型容量很小（论文做到最高 64× 容量差），强行去拟合"难学、且本来也很难被接受"的 hard token，会挤占 easy token 的学习预算，导致整体接受率次优，有时 loss 还不收敛。

#### 3. Motivation(为什么做这件事)
作者观察到 KD 中各 token 学习难度差异巨大：强行压低 hard token 的 loss，往往是以抬高 easy token 的 loss 为代价。既然 draft 只需要在"足够易提议"的 token 子集上预测对就能拿到接受，那就应该**主动剔除 hard token**，把有限容量集中到 easy token 上，在容量约束内最大化与 target 的对齐。

#### 4. 主要灵感 / 核心直觉
核心直觉：**"对 draft 而言有优化潜力的 token"≠"loss 大的 token"**。一个 token 即使当前 loss 大，如果连一个专门蒸馏过的参考模型都学不好，那它就是 draft 学不动的硬骨头，不值得投入。真正该学的，是"draft 现在差、但参考模型证明它学得动"的 token。于是引入一个参考模型作为"难度探针"。

#### 5. 主要解决思路(一段话把核心机制讲清)
两阶段：先训一个与 draft 同初始化的**参考模型(reference)**，用 DistillSpec 从 target 蒸馏，让它成为"token 难度分析器"；然后对每个 token 比较"draft 的蒸馏损失"与"参考模型的蒸馏损失"，差值越大说明这个 token 对 draft **越有可学空间(优化潜力大)**，取损失差最大的 top-k% token 组成子集 S，**draft 只在 S 上做蒸馏**，其余 token 直接忽略。

#### 6. 方法详解(通俗、分步骤;关键公式用白话解释,必要时给伪代码)
**Step 1 — 参考模型蒸馏 + 难度估计**：reference 初始化为 draft 的拷贝，用 DistillSpec(forward-KL) 从 target 蒸馏。它扮演"如果充分蒸馏，draft 这个规模最多能学成什么样"的上界探针。

**Step 2 — 选择性 draft 蒸馏**：对每个 token w，
- 算 draft 的损失 `L_draft(w)=KL(target‖draft)`、参考模型的损失 `L_ref(w)=KL(target‖ref)`；
- 算损失差 `ΔL(w)=L_draft(w) − L_ref(w)`。白话：ΔL 大 = "draft 现在比已学好的参考差得多" = 这个 token **还有很大可学空间**；ΔL 小 = draft 已接近参考能达到的极限，再练也榨不出多少。
- 选 ΔL 最大的 top-k% token 组成子集 S（默认 **k=0.4**），draft **仅对 S 内 token** 求蒸馏损失。

**工程实现**（据论文 Appendix A.4 Listing 2，约 100 行，override `transformers.Trainer.compute_loss`；**仓库内无可运行代码**）：对每 token 用 `KLDivLoss(reduction='none')`，以 target softmax 为 P，分别算 `actual=KL(P‖draft)`、`ref=KL(P‖ref)`，按 `delta=actual−ref` 取掩码 `delta >= torch.quantile(delta, 1−k)`，只对 masked token 求和(除以 num_items_in_batch)或求均值。可与 EAGLE 叠加。

#### 7. 实验数据集
- 任务/数据：GSM8K(算术推理)、Alpaca(指令跟随)、MBPP(代码)、CNN/DailyMail 与 XSUM(摘要)。
- 模型对：Pythia 31M→1.4B、CodeGen-350M→Phi-2(≈2.7B)；声称在最高 **64× 容量差**下仍有效。
- 主指标：acceptance rate α；辅以 block efficiency、wall-time speed-up。
- 超参：lr 多为 3e-4（MBPP/部分任务 1e-5~1e-4），filter fraction **k=0.4**，分 "3-epoch" 与 "optimal-epoch" 两套设置。

#### 8. 实验结果与主要发现(关键数字)
一致超过 SOTA 基线 DistillSpec，acceptance rate 在所有任务最高提升约 **15%**：
- GSM8K，Pythia(3-epoch)：57.58% → **62.63%**；
- CodeGen→Phi-2：79.49% → **82.79%**；
- MBPP(optimal-epoch)：49.88% → **65.12%**（提升最大的一档）。
（以上数值已对照论文 _txt 第 287-309 行核实。）

#### 9. 结果如何支撑其主张(证据链是否到位)
主张"选择性过滤 > 全 token 蒸馏"由跨任务、跨模型对的 acceptance rate 一致提升支撑，证据较扎实。但需注意：主表汇报的是**接受率这个代理指标**，而非端到端 wall-time 加速——后者的增益还依赖 cost coefficient c（draft 单步与 target 单步的相对成本），论文以 α 为主指标，端到端加速的提升幅度相对弱化。

#### 10. 逻辑自洽性(中性评估:哪里站得住、哪里牵强)
- **站得住**："用参考模型的损失作基准来定义可学性(ΔL)，而不是用绝对 loss"这一点逻辑清晰，直接对应"容量约束下的最优分配"直觉，且实验一致支持。
- **牵强/可质疑**：(1) ΔL 大确实代表"差距大"，但是否一定代表"draft 真能补上这个差距"，论文用经验结果回答，缺乏更强的理论保证；(2) 多训一个参考模型本身是额外开销，"省 draft 容量"与"多一阶段训练成本"的净收益论文未充分量化。

#### 11. 残留问题 / 局限
- **仅限同族/同词表 draft-target 对**（KL 要求词表对齐）。
- **多一阶段开销**：参考模型需额外完整蒸馏一遍。
- **k 为固定超参**（默认 0.4），跨任务未做自适应，最优 k 可能随任务漂移。
- **代理指标**：提升的是接受率，端到端 wall-time 加速依赖成本系数，主表未以墙钟时间为主。
- **代码不可得**：方法核对只能依据论文附录 Listing 2，无可运行实现可审。
- 属推理加速场景，与 reasoning RL 后训练目标不同；与本项目相关性在于"token-level 选择性蒸馏"这一思想谱系。

#### 12. 开源代码与框架(链接 + 框架 + 代码可得性)
- 链接：https://github.com/yuezhouhu/adaspec （OpenReview zNLlglSOwD）。
- **可得性受限**：已 clone 的 main 分支(与 origin/main 一致)仅含 `README.md / LICENSE / .gitignore / adaspec.png`，**无 train.py / utils.py / run.sh 等训练代码**。README 称"实验配置分散在各 git 分支"，但 origin 上并未实际推送这些分支。核心 `compute_loss` 仅见于**论文 Appendix A.4 Listing 2**。
- 框架：HuggingFace transformers / TRL / Accelerate / DeepSpeed（两阶段选择性 KD）。


---


### aligndistil — AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation

> **一句话重点 (TL;DR)**：把"带 token-level reward 的 RLHF 目标"在理论上**等价改写成一个蒸馏问题**——其 teacher 分布是 DPO 模型与 reference 模型 logit 的线性组合，于是 RLHF 对齐就变成"向一个自动合成的 teacher 分布做 token 级蒸馏"。值得看的点：它给"DPO 的 token-level reward 分解"找到了一个干净的蒸馏对应，并用一个 reverse-DPO 对比 + 逐 token 自适应外插权重把这个对应做得更稳更准。

**元信息**：arXiv 2503.02832v3（2025-07-23，cs.CL）｜ 北京交通大学(交通大数据与人工智能教育部重点实验室、计算机学院) + 腾讯（Songming Zhang 实习于腾讯完成；通讯 Yufeng Chen、Jinan Xu）｜ ACL 2025 ｜ 主题 T1/T3 On-Policy Distillation / RLHF 对齐，**相关性 High** ｜ 代码 https://github.com/songmzhang/AlignDistil（**完整**，含定制版 OpenRLHF）｜ 框架 OpenRLHF v0.5.2.post2（+ vLLM for on-policy）

#### 1. 相关工作与进展(这条线现在做到哪一步)
LLM 对齐主流两条线：(1) **RLHF**——先训 response-level reward model，再用 PPO 等优化策略并约束不偏离初始模型；(2) **直接偏好学习(DPO 等)**——用策略模型自身参数化 reward，直接在偏好数据上训练，省掉显式 RL。近期出现 token-level 的偏好优化(如 TDPO)与把 RLHF 视为蒸馏的视角(如 RTO)。AlignDistil 处在"把 token-level reward 优化与蒸馏统一起来"的交叉点。

#### 2. 现有工作存在的问题(本文针对的痛点)
- 现有对齐多用**稀疏的 response-level reward / 偏好标注**去优化一整条回复里的所有 token，**粒度太粗**：无法反映每个 token 的个体贡献，可能错误惩罚一条好回复里的高质量 token、或鼓励差回复里的低质量 token，拖慢收敛、限制上限。
- DPO 自带的 token-level reward 虽可分解，但**精度不如纯 reward model**，且在不同 token 上存在欠优化/过优化不均衡。

#### 3. Motivation(为什么做这件事)
作者想要 token-level 的奖励优化，又不想额外训练一个昂贵的 reward model。注意到 **DPO reward 本身可在 token 级分解**，于是设问：能否把"带 DPO token-level reward 的 RLHF 目标"直接转成一个**蒸馏**过程？如果能，token-level 奖励优化就等价于"向某个 teacher 分布蒸馏"，实现简单且稳定。

#### 4. 主要灵感 / 核心直觉
两个直觉：(1) **RLHF + DPO-reward ⇔ 蒸馏**：理论上可证 sequence-level RLHF 目标等价于一个 token-level 蒸馏目标，其 teacher 分布 = DPO 模型与 reference 模型 logit 的线性组合(本质是"沿着 DPO 改进方向外插")。(2) **正反两个 DPO 更有判别力**：再训一个把 chosen/rejected 交换的 **reverse DPO**，让它捕捉"低质量数据的负面特征"，与正常 DPO 组成对比，reward 更准、可训练参数隐式翻倍。

#### 5. 主要解决思路(一段话把核心机制讲清)
从 RLHF 目标出发引入 DPO reward，证明该目标等价于"把当前策略向一个合成 teacher 分布做 reverse-KL 蒸馏"；teacher 分布由 **forward DPO 模型**与 **reverse DPO 模型**(充当 reference)的 logit **外插(extrapolation)** 而成，推动策略**越过** DPO 模型；再用一个**逐 token 自适应外插权重 γ_t** 给每个 token 位置定制 teacher 强度，缓解欠/过优化。可在 on-policy(采样数据，效果更好)与 off-policy(偏好数据，更高效)间切换。

#### 6. 方法详解(通俗、分步骤;关键公式用白话解释,必要时给伪代码)
三步流程：**① 训正常 DPO** → **② 训 reverse DPO**(交换 chosen/rejected) → **③ AlignDistil**：用两者 logit 合成 teacher 分布，蒸馏到当前策略。

关键公式（已逐行核对 `aligndistil_trainer.py`，`reward_boost_type=="aligndistil"` 分支，代码注释明确 **teacher = forward DPO 模型，reference = reverse DPO 模型**）：
- **逐 token 外插权重**：`tvd = |p_tea − p_ref|.sum`（forward-DPO 与 reverse-DPO 概率分布的全变差距离 TVD），`weight = tvd·β2 + 1e-3`。白话：两个 DPO 模型在某 token 上分歧越大(TVD 大)，外插越激进。
- **合成 teacher logits**：`final_tea_logits = weight·(teacher_logits − reference_logits) + teacher_logits`。即在 forward DPO 的基础上，再沿"forward−reverse"差分方向外推一段，让 teacher 比单纯 DPO 更"对齐"。
- **蒸馏损失**(reverse-KL 形)：`rlhf_loss = (stu_probs·(stu_lprobs − final_tea_lprobs)).sum × beta2`，其中 `beta2 = β/weight`。
- 对照分支：`theorem1` 用 `weight·teacher + (1−weight)·reference` 的常数线性组合；`theorem1_contrast` 用差分形式但常数权重；`*_adaptive` / `aligndistil` 则把权重换成逐 token 的 TVD 自适应——消融正是为隔离这一项。

#### 7. 实验数据集
- 初始模型：Qwen2-1.5B-Instruct、Qwen2.5-1.5B-Instruct。
- 偏好/训练数据：UltraFeedback。
- 评测：AlpacaEval 2.0(length-controlled win rate)、MT-Bench、Arena-Hard；裁判用 Qwen2.5-72B-Instruct(作者测得与 GPT-4 判断相当但更便宜)。

#### 8. 实验结果与主要发现(关键数字)
- on/off-policy 两版 AlignDistil 均显著超基线；AlpacaEval 2.0 LC win rate 较 DPO **提升 >6%**；优于 TDPO1/2、RTO、PPO、DPO(β=0.01) 等。
- Table 2：对比 DPO reward 的 reward accuracy(UltraFeedback 训练/测试各 1000 样本)优于 vanilla DPO reward，甚至优于显式 reward model。
- Table 3：token 自适应外插优于常数外插(为隔离数据影响，用 off-policy 设置)。
- 〔待核〕Table 1 各 benchmark 精确数值未逐一抄录。

#### 9. 结果如何支撑其主张(证据链是否到位)
证据链较完整：理论(RLHF⇔蒸馏的等价推导) + Table 2(对比 reward 更准) + Table 3(自适应外插 > 常数外插) + 主表(三 benchmark 一致超基线)，分别对应方法的三个主张(蒸馏等价、对比 reward、token 自适应)。on-policy 优于 off-policy 也与"采样数据更贴近策略分布"的预期一致。

#### 10. 逻辑自洽性(中性评估:哪里站得住、哪里牵强)
- **站得住**：等价性推导 + 代码实现一一对应(外插公式、TVD 权重、reverse-KL 损失均已核实)；"用 reverse DPO 作 reference 而非固定初始模型"是一个有依据的设计，对比实验支持。
- **牵强/可质疑**：(1) "外插越过 DPO"假设 DPO 方向在更大步长上仍正确，过度外插可能放大噪声，论文靠 `1e-3` 下限和 TVD 自适应缓解但无理论上界；(2) 规模仅 1.5B、评测偏对话/指令，未验证更大模型或推理域；(3) 需训三个模型(DPO、reverse DPO、最终策略)，成本不低。

#### 11. 残留问题 / 局限
- 实验规模有限(1.5B)，未覆盖大模型与数学/代码等推理任务。
- 三阶段训练(DPO→reverse DPO→蒸馏)成本与超参(β、β2、kd_temperature)敏感性需更多消融。
- 外插权重虽自适应但仍依赖人工设定的 β/β2 与 `1e-3` 下限。
- 〔待核〕主表精确数值未抄录。

#### 12. 开源代码与框架(链接 + 框架 + 代码可得性)
- 链接：https://github.com/songmzhang/AlignDistil （已 clone 到 resource/repos/aligndistil，约 2.4M，**完整**，自带定制版 OpenRLHF 子目录）。
- 框架：**OpenRLHF v0.5.2.post2**（`OpenRLHF/version.txt` 实测；安装 `cd AlignDistil/OpenRLHF && pip install -e ./`，on-policy 版另需 vLLM）。
- 训练脚本：`train_scripts/ultrafeedback/qwen2.5-1.5b/`（dpo / reverse_dpo / aligndistil_off_policy / aligndistil_on_policy）。核心 loss 在 `openrlhf/trainer/aligndistil_trainer.py`（`reward_boost_type=="aligndistil"`）。


---


### apple_ssd — SSD: Embarrassingly Simple Self-Distillation Improves Code Generation

> **一句话重点 (TL;DR)**：不用任何 teacher / verifier / reward / RL / 代码执行环境，只让模型在"调过温度 + 截断"的设置下采样自己的**原始未验证**输出，再用标准交叉熵 SFT，最后评估时单独调一个解码温度——就能把代码生成显著提升(Qwen3-30B-Instruct LiveCodeBench v6 pass@1 42.4%→55.3%)。值得看的点：它给"自蒸馏为何起作用"提出一个清晰机制——代码解码存在 **precision-exploration conflict**，SSD 通过上下文相关地重塑 token 分布(该压的地方压、该留多样性的地方留)拿到固定解码拿不到的增益。

**元信息**：arXiv 2604.01193v1 ｜ Apple（Ruixiang Zhang、Yizhe Zhang 等共一）｜ 2026-04 arXiv preprint ｜ 主题 自蒸馏(off-policy 纯 SFT) / 与 OPD **对照** ｜ 代码 https://github.com/apple/ml-ssd（**部分**：含数据生成 + 评测，SFT 训练本体不在仓库，README 注明为"复现")｜ 框架 仅需 采样 + 标准 SFT(交叉熵) + 评估时单独调温；无 RL/verifier/teacher/执行环境

#### 1. 相关工作与进展(这条线现在做到哪一步)
LLM 编码任务越来越难，高质量监督信号成瓶颈：人工解贵；合成数据要么需要更强 teacher、要么需对每题做基于执行的验证。提升路径主要有：(1) **teacher 蒸馏**——继承 teacher 上限；(2) **RLVR**——操作复杂、可能不稳；(3) **基于内禀奖励的无监督自提升**(majority voting/熵最小化)——有早期成效但面临 reward hacking 与长训崩溃。SSD 处在"无外部信号自提升"这条线，但走的是最朴素的 off-policy 自蒸馏。

#### 2. 现有工作存在的问题(本文针对的痛点)
核心问题：**能否让模型在完全不借助任何外部标注或验证的情况下自我提升？** 现有路径都需要 teacher / verifier / reward / 执行环境 / 人工标注解之一；无监督内禀奖励法又易 reward hacking、长训崩溃。

#### 3. Motivation(为什么做这件事)
作者想要一个**最简、无任何外部依赖**的后训练方法，并搞清"自蒸馏到底为何 work"。他们把目光放到代码生成——因为代码的任务结构让底层机制**特别可见**，便于分析。

#### 4. 主要灵感 / 核心直觉
关键直觉是 **precision-exploration conflict(精度-探索冲突)**：代码里同时存在两类位置——
- **fork 位置**：多个续写都合理(对应不同解法)，需要**多样性**(高温有利)；
- **lock 位置**：语法语义近乎唯一，但仍有一条低概率 **distractor(干扰)尾巴**，需要**精度**(低温有利、压住 distractor)。

两类位置对解码温度 Teval 的需求**相反**，所以任何**全局固定温度**都必然是折中。而 SSD 在"温度偏移 + 截断"样本上训练，能**按上下文**重塑分布：在 lock 处最狠地压 distractor，同时在 fork 处保留有用多样性——这是单纯改解码温度无法恢复的增益。

#### 5. 主要解决思路(一段话把核心机制讲清)
三步：从冻结的 base 模型用非单位温度 Ttrain + 截断 τ 采样若干候选解；对这些**原始、未验证**的输出做标准交叉熵 SFT；评估时用一个**单独调过的**温度 Teval 解码。训练的净效果被形式化为对 token 分布的 **support compression(支撑压缩)** 与 **within-support reshaping(支撑内重塑)**(论文 Eq.4)，并用受控仿真 + 真实模型分析 + 理论(附录 B)支撑这一机制解释。

#### 6. 方法详解(通俗、分步骤;关键公式用白话解释,必要时给伪代码)
**SSD 三步**：
1. **Sample**：冻结 base，以温度 Ttrain(非 1.0) + 截断配置 τ(top-k/top-p)对每个 prompt 采样 N 个候选解。
2. **Fine-tune**：对这 N 个**未经任何验证**的原始输出做标准交叉熵 SFT。
3. **Decode**：评估时用单独调过的温度 Teval 解码。

**机制白话**：在低温 + 截断下采样，等于把分布尾巴(包括 lock 处的 distractor)先削掉再当训练目标，于是 SFT 学到的新分布在 lock 处**自动**压住 distractor；而 fork 处因为本来就有多个高概率续写，截断削不掉它们，多样性得以保留。这就是"按上下文重塑"——比全局调温更精细。

#### 7. 实验数据集
- 主基准：**LiveCodeBench v6**(按 Easy/Medium/Hard 分难度报告 pass@1 与 pass@5/coverage)；另含 **LCB v5**(374 题)。
- 模型(2 家族 × 3 规模 × instruct/thinking，共 5 个)：Qwen3-4B-Instruct、Qwen3-30B-Instruct、Qwen3-4B-Thinking、Qwen3-30B-Thinking、Llama-3.1-8B-Instruct。

#### 8. 实验结果与主要发现(关键数字)
- **Qwen3-30B-Instruct**：LCB v6 pass@1 42.4% → **55.3%**(+12.9pp，相对 +30.4%)；LCB v5 45.8% → **54.3%**(+8.5pp)。
- **增益集中在中/难题**：hard pass@5 从 31.1% → **54.1%**——说明保留了跨解法分支的探索，而非只锐化单一主模。
- **5 个模型全部提升**(pass@1)：Llama-8B +3.5pp、4B-Instruct +7.5pp、4B-Thinking +3.3pp、30B-Thinking +2.1pp、30B-Instruct +12.9pp；跨家族/规模/变体均泛化。
（数值已对照论文 _txt 第 96、296 行核实。）

#### 9. 结果如何支撑其主张(证据链是否到位)
- "极简自蒸馏有效"——5 模型一致提升，证据强。
- "机制是 precision-exploration conflict"——用三路证据互证：**受控仿真**(玩具分布上复现 support compression/reshaping)、**真实模型分析**(Section 4.2)、**理论**(附录 B)。hard pass@5 大涨(31.1→54.1)是关键旁证：若 SSD 只是锐化单模，coverage 不该升反该降。
- "固定解码无法恢复增益"——论文论证 Teval 调温的天花板低于 SSD，支撑"重塑分布"而非"换个温度"的主张。

#### 10. 逻辑自洽性(中性评估:哪里站得住、哪里牵强)
- **站得住**：机制解释与 coverage 上升的经验现象自洽；用未验证原始输出仍提升，说明增益来自分布重塑而非数据筛选，逻辑闭环。
- **牵强/需注意**：(1) 对"未验证输出里必然混入错误解"为何不损害性能，靠"低温采样的解平均质量较高 + SFT 对高频正确模式更敏感"间接解释，但缺乏对错误解占比与最终增益的定量关系；(2) 机制主要在代码域验证，"代码任务结构让机制可见"反过来也意味着该机制在非代码域是否成立尚不确定；(3) Ttrain/τ/Teval 三个温度/截断超参的选取对结果影响大，调参成本被"embarrassingly simple"的叙事淡化。

#### 11. 残留问题 / 局限
- 仅在代码生成域验证机制；跨域(数学/通用推理)是否成立未知。
- 训练用未验证输出，存在引入错误模式的风险，长程多轮自蒸馏是否会崩溃(类似熵最小化的崩溃)未测。
- 三组解码超参(Ttrain、τ、Teval)需调，最优组合的可迁移性待考。
- SFT 训练本体不在仓库，复现需自备 Megatron-LM 等外部框架。

#### 12. 开源代码与框架(链接 + 框架 + 代码可得性)
- 链接：https://github.com/apple/ml-ssd （Apple 许可；约 0.85MB，社区关注度高约 772 stars）。预训练模型(4B/30B)在 HuggingFace。
- **可得性部分**：仓库含 `data_generation/`(采样流水线 `generate.py` + 模板)与 `evaluation/`(LiveCodeBench 评测工具)；**SFT 训练本体不在仓库**——标准交叉熵，依赖 Megatron-LM 等外部框架，README 注明为"复现"。
- 框架：方法极简，仅需 采样 + 标准 SFT + 评估时单独调温；无 RL、无 verifier、无 teacher、无执行环境。


---


### avsd — AVSD: Adaptive-View Self-Distillation by Balancing Consensus and Teacher-Specific Privileged Signals

> **一句话重点 (TL;DR)**：同一个模型当 student 又当 teacher，teacher 额外看到三种"特权信息视图"（完整解 / 部分推理 / 仅答案）；AVSD 不固定用某一种视图，而是把多视图 teacher 信号拆成"跨视图共识"（可靠方向）和"单视图残差"（有用但有风险），用一个门控只在残差与共识方向一致且幅度相称时才加进去——从而比任何单视图自蒸馏都更稳更好。

**元信息**：arXiv:2605.20643v1（2026-05-20，cs.LG，Preprint）｜ UNC Chapel Hill / Capital One / UT Austin｜ 2026-05 预印本｜ 主题 T1（on-policy 自蒸馏）/ 相关性 Med（与 TSRD 的"特权信息 teacher / scaffold"高度相关）｜ 代码 https://github.com/duykhuongnguyen/AVSD （已 clone 到 resource/repos/avsd，约 1.9M；真实可用，但生成的 JSONL 训练数据被 Git 忽略需自构）｜ 框架 DeepSpeed ZeRO-2 + HF Accelerate + PEFT(LoRA) + vLLM，底层基于 OPSD（siyan-zhao/OPSD）→ TRL。

---

#### 1. 相关工作与进展
- **RLVR**（GRPO 等）是当前可验证任务（数学/代码）后训练主流，但监督**稀疏、只在 outcome 层**：失败时几乎没学习信号，采样昂贵。
- **蒸馏**提供 dense 的 token-level 指导，但标准**离线蒸馏**有 train-test mismatch（学生在自身分布外的轨迹上训练）。
- **on-policy 蒸馏 (OPD)**：在学生自采样轨迹上用 teacher 概率做局部监督，缓解 off-policy 问题，但仍**依赖一个更强的外部 teacher**。
- **自蒸馏 (self-distillation)**：去掉外部 teacher，让同一模型当 teacher，只是 teacher 额外条件化于学生推理时看不到的**特权信息 (privileged information)**——解、示范、反馈、最终答案等。本文即在自蒸馏这一支上做改进。

#### 2. 现有工作存在的问题
作者明确指出单视图自蒸馏的两个核心缺陷（§1，引 Penaloza 2026 / Yang 2026 / Kim 2026）：
1. **不对称性 (asymmetry)**：teacher 依赖某种特权信息，而学生推理时无法访问。Yang et al. 2026 证明信息不对称下的分布匹配存在**不可消除的互信息 gap**——学生被迫去编码 test 时观测不到的 view-specific 关联。
2. **视图选择难**：最优特权信息类型**因任务而异**且无先验判据。给完整解可能把 teacher 锁进与学生不合的思维模式；只给答案更灵活但信息量低；Kim et al. 2026 还发现信息量最大的视图反而丢掉了帮助识别推理错误的信号、损害 OOD 泛化。而现有方法一旦选定视图就**全程固定**。

#### 3. Motivation
能否**同时**利用多个特权视图，构造出比任何单视图 teacher 都更好的 token-level on-policy 学习信号？

#### 4. 主要灵感 / 核心直觉
两条直觉（§1）：
- 若**不同视图诱导相似的 token 更新**（都想 promote 或都想 suppress），该信号大概率**稳定且任务相关**——因为各视图共享同一学生可见前缀，只是特权信息不同，跨视图一致的更新不太可能依赖某个 view-specific artifact。
- 若**某 token 仅被一个视图强烈偏好**，它可能捕捉互补信息（有用），但也可能是学生看不到的特权 artifact（危险）——需要"加，但要小心地加"。

#### 5. 主要解决思路（一段话讲清核心）
把 M 个特权视图各自诱导的 teacher 分布拆成两部分：**共识 (consensus)** = 跨视图一致支持的部分，用**几何均值池化 (geometric consensus target)** 刻画（只有被所有视图共同支持的 token 才得高概率）；**残差 (residual)** = 额外的 view-specific 支持，用**算术均值池化 (arithmetic marginal target)** 减去共识得到（只要有一个视图强支持就保留）。AVSD **不直接蒸馏任一池化目标**，而是以共识为基准方向，用一个**门控 (gate)** 决定是否把残差加上去：仅当各视图在 promote/suppress 方向一致、且残差幅度与共识幅度相称时才加。这样既能吸收互补信息，又防止任一视图主导。

#### 6. 方法详解（通俗、分步骤）
形式化建立在 **token-level reverse-KL advantage** 上（§2.1，完整推导见 Appendix B.1）：
- 学生在前缀 h_t=(x,y_<t) 的下一 token 分布 p_t(v)；teacher 在第 m 个视图下的分布 q_t^(m)(v)=sg[P^T(v|h_t,r^(m))]（sg 为 stop-gradient）。
- **per-view 蒸馏 advantage**：Δ_t^(m)(v) = log q_t^(m)(v) − log p_t(v)。>0 表示该视图想 promote 此 token，<0 想 suppress。reverse-KL 的负梯度写成 policy-gradient 形式即 E_{v∼p}[A_t(v)∇log p_t(v)]。

分步：
1. **构造 M 个视图**：r^(m)=T_m(r)，保留任务相关信息、改变暴露给 teacher 的特权形式。本文数学/代码均用 **M=3** 个视图。
2. **几何共识目标**：对各视图概率取几何均值再归一化 → 强调"被所有视图共同支持"的 token（交集支持）。
3. **算术边际目标**：取算术均值 → 保留"至少被一个视图强支持"的 token（并集支持）。
4. **残差** = 算术边际 − 几何共识。
5. **门控加残差**（正文 Eq.2）：从共识出发，仅当 (a) 各视图方向一致 (b) 残差幅度 ∝ 共识幅度时，按比例 λ=C·R 加入残差。代码实现于 `src/avsd/common/multiview_distill.py::build_avsd_target`，`avsd` 模式即 λ = `consensus_adv.abs()/delta_abs_mean × |A^G|/(|A^G|+J)`，与正文 Eq.2 一致，且断言要求**各视图权重均匀**。
6. **消融**显示：consensus-only（只蒸馏几何共识）与 arithmetic-only（只蒸馏算术边际）变体均逊于完整 AVSD —— 证明"共识做骨架 + 门控加残差"两件事都不可或缺。

#### 7. 实验数据集
- **数学**：训练用 **OpenThoughts**（正文为 "OpenThoughts" 而非旧分析里的 "OpenThought"，已核）数学子集；每题自动构造三视图（full solution / partial solution / final answer，partial_solution_ratio=0.5）；评测 **AIME24 / AIME25 / HMMT25**，指标 **Avg@8**。模型 Qwen3-8B、Qwen3-4B、DeepSeek-R1-Distill-Qwen-7B。
- **代码**：从 Codeforces（open-r1/codeforces）Python 子集采 5K 训练 + 留出 100 题做 in-domain 测试；评测 Codeforces、**LiveCodeBench v6**；视图为 reference / hint / feedback。模型 Qwen3-8B。
- **基线**：单视图自蒸馏（Zhao 2026 / Shenfeld 2026）+ GRPO。

#### 8. 实验结果与主要发现
- Qwen3-4B（数学）：较 base **+5.5%**、较最强基线 **+2.2%** Avg@8。
- Qwen3-8B（数学）：取得最佳均分，较最强基线 **+3.1%**。
- DeepSeek-R1-Distill-Qwen-7B（数学）：较自蒸馏基线 **+2.3%**。
- Qwen3-8B（代码）：较单视图自蒸馏基线 **+2.4%** 均值。
- **关键发现**：Fig.1 显示"最佳单视图随数据集变化、full solution 并非总最优"——这正是 AVSD 自适应组合多视图的立足点；token-level 分析进一步显示 AVSD 同时保留了跨视图一致性并选择性加入门控残差。

#### 9. 结果如何支撑其主张
主张是"多视图自适应组合 > 任一单视图"。证据链较完整：(a) 跨三个模型、两个领域均一致超过最强单视图基线与 GRPO；(b) 消融排除了"只用共识"或"只用边际"的简化解释；(c) Fig.1 的"无单一最优视图"现象直接 motivate 了方法。增益幅度（2–3% Avg@8）属中等而非压倒性，且基准集中在竞赛数学，需谨慎看待泛化。

#### 10. 逻辑自洽性（中性评估）
- 自洽：从"不对称 + 视图选择难"两个问题，到"共识/残差分解 + 门控"的方法，再到消融，论证链条闭合；reverse-KL advantage 的形式化与代码实现（`build_avsd_target`）一致。
- 张力点：方法声称缓解"互信息 gap"（不对称问题），但本质仍是各视图 teacher 都条件化于特权信息，门控只是抑制"仅单视图支持"的危险 token，**并未从原理上消除**学生编码不可见关联的压力——更像是经验性缓解而非理论保证。

#### 11. 残留问题 / 局限
- **增益中等**（2–3%），且评测高度集中于竞赛数学 + 少量代码，未见更广泛任务/通用指令的验证。
- **视图需人工/自动构造**（数学三视图靠规则切分，partial_ratio=0.5 是超参），视图质量与数量（M=3）对结果的敏感性未充分扫描。
- 门控的两个条件（方向一致 + 幅度相称）是**手工设计的启发式**，缺乏最优性论证。
- 不涉及 MTP；与 TSRD 的关联在于"特权信息 teacher / scaffold"的思路，而非具体 foresight 机制。

#### 12. 开源代码与框架（链接+框架+代码可得性）
- 代码：https://github.com/duykhuongnguyen/AVSD （resource/repos/avsd，约 1.9M）。
- **框架栈**：DeepSpeed（ZeRO-2 + CPU optimizer offload）+ HuggingFace Accelerate + PEFT(LoRA) + vLLM；trainer/自蒸馏基础设施沿用 **OPSD**（siyan-zhao/OPSD），OPSD 又基于 **TRL**。environment.yml 关键依赖：torch 2.8.0、transformers 4.57.1、**trl 0.26.0**、deepspeed 0.18.2、accelerate 1.11.0、peft 0.17.1、vllm 0.11.0、flash-attn 2.8.3。
- **训练**：`accelerate launch` + `configs/accelerate.yaml`（bf16、ZeRO-2、CPU offload、默认 4 进程）；vLLM colocated rollout（默认 `--vllm_gpu_memory_utilization 0.4`、TP=1，周期性从 trainer 同步权重）。
- **复现入口**：数学 `scripts/math/train_avsd_qwen3_8b.sh` / `train_avsd_deepseek_r1_distill_qwen_7b.sh`（默认 OpenThoughts 数据，写到 `outputs/avsd/`），评测 `python -m avsd.math.evaluate --dataset {aime24|aime25|hmmt25}`；代码 `download_codeforces_cots_py.sh` → `python -m avsd.code.prepare_code_views` → `scripts/code/train_avsd_code_*.sh` → `avsd.code.evaluate_codeforces`（num-samples 8）。
- **已核超参**（train_avsd_qwen3_8b.sh / train_avsd_code_qwen3_8b.sh）：lr 5e-6、max_grad_norm 0.1、max_steps 500、per_device_bs 4 × grad_accum 2、max_completion_length 4096、max_length 16384、temperature 0.7 / top_p 0.95 / top_k 20、LoRA r64/α128、bf16+flash_attn2、`--use_tinker_loss`、`--multi_view_mode avsd`、`--avsd_gate_mode avsd`。代码结构 `src/avsd/{common,code,math}/`。
- **代码可得性**：方法核心（门控、多视图蒸馏）真实可跑，但**生成的 JSONL 训练数据被 .gitignore**，需自行下载/构造。


---


### brts — On-Policy Distillation with Best-of-N Teacher Rollout Selection (BRTS)

> **一句话重点 (TL;DR)**：标准 OPD 每个 prompt 只用一条随机 teacher rollout，方差大、错了还会被放大。BRTS 对每 prompt 采 N 条 teacher 轨迹，按"先正确、再与学生对齐"挑一条；全错时用注入 ground-truth 的提示让 teacher 重新自然推导；选中的轨迹作为额外的 teacher-context 蒸馏分支，与标准 student-context OPD 一起训练。

**元信息**：arXiv:2605.09725v2（2026-05-13，cs.CV）｜ JHU + TikTok + UCSD + 复旦（Ke Zhang JHU/TikTok 实习期间完成；通讯 Di Fu, TikTok）｜ 2026-05 预印本｜ 主题 T1/T2/T4（High）——针对 OPD"单次随机 teacher rollout 高方差/错误/不匹配"，提出 Best-of-N 选择 + ground-truth 引导恢复 + teacher-context 辅助损失，**与 TSRD 的 path-recovery、teacher-scaffolded 思路高度一致**｜ 代码 https://github.com/BWGZK-keke/BRTS （已 clone 约 14MB，基于 verl fork）｜ 框架 veRL（与 thunlp/OPD 同源 fork）。

---

#### 1. 相关工作与进展
- **OPD 已成 LLM 后训练标准工具**：学生自采样 rollout，用 teacher 逐 token log-prob 作 dense 监督；比 SFT/序列级蒸馏更不易受 exposure bias（监督定义在学生推理时真实访问的状态上）。工业 pipeline（Qwen3、MiMo、GLM-5）已采用，且报告以 RL 一小部分算力获得可比增益。
- 近期工作开始解析 OPD 成败条件（[26] 等）：师生需共享**兼容推理模式**，且 teacher 要能产出**超出学生当前探索范围的、自然推导的正确解**，才能迁移新能力；小模型难以模仿风格不匹配的强 reasoner。
- BRTS 与 best-of-N / rejection sampling / 特权信息（ground-truth、demonstration）方法相关，但**把选择放进 OPD 内循环**而非离线过滤。

#### 2. 现有工作存在的问题
- 标准 OPD 在"不完美学生生成、可能漂移到噪声推理状态"的**前缀**上计算 teacher 监督（学生 context 噪声）。
- 且通常每 prompt 只依赖**单次随机 teacher rollout**。teacher 本身随机：同一 prompt 的采样轨迹在正确性、推理风格、与学生接近度上差异大（尤其难题）。**单样本是对 teacher 该 prompt 能力/对齐的高方差估计**；当该样本错误或不匹配时，会产生噪声指导并被放大（Fig.1a）。

#### 3. Motivation
不从任意 teacher rollout 学，而是**先识别一条可靠 teacher 轨迹再蒸馏**，作为额外的 teacher-context 监督：既补充标准 student-context 蒸馏（修正学生访问状态），又让学生见到完整可靠的 teacher 推理路径；该分支同时是 **correctness-aware**（防错误样本放大）与 **alignment-aware**（选最接近学生当前分布的轨迹）。

#### 4. 主要灵感 / 核心直觉
"与其学一条随机轨迹，不如从一小池里挑最可靠且最像学生能走的那条"——把随机 teacher rollout 变成**结构化监督**。难题上全错时，给 teacher 偷偷看答案逼它产出自然推导（path-recovery），保证最需要监督处分支仍活跃。

#### 5. 主要解决思路（一段话讲清核心）
BRTS = **Best-of-N Rollout Teacher Selection**：每 prompt 采 N 条 teacher 轨迹 → 按"correctness first, student-alignment second"选一条（全错则注入 ground-truth 重采恢复，仍不行则回退最相似）；在标准 student-context OPD 损失外，加一条在选中轨迹上的 teacher-context 蒸馏损失，权重 λ 控制。

#### 6. 方法详解（通俗、分步骤）
**(A) teacher 轨迹 curation（三层，Algorithm 1）**
1. **Tier-1**：采 N 条 unconditioned teacher 样本，按答案判对错。
2. 若 ≥1 条正确 → 在正确轨迹中选与学生 **top-K overlap 最高**者（alignment）。
3. 若全错 → **Tier-2 ground-truth 引导恢复**：构造修改 prompt x_gt（把 ground-truth 答案作为"静默校验信号"，要求 teacher 给自然推导），采一条 guided rollout，**仅当其答案正确才保留**。
4. 若仍无正确 → fallback 到 Tier-1 中 student-overlap 最高者（虽可能错，但避免引入离学生很远的任意轨迹）。

**(B) teacher-context 监督（Eq.3–5）**
- 保留 student-context 损失（学生前缀上的标准 OPD，**reverse-KL**）：`L_stu-ctx = E[Σ_t D_KL(π_S(·|x,ŷ_<t) ‖ π_T(·|x,ŷ_<t))]`（Eq.3）。
- 新增 teacher-context 损失（在选中 teacher 前缀 y'_<t 上）：`L_tea-ctx = E[Σ_t D_KL(π_T(·|x,y'_<t) ‖ π_S(·|x,y'_<t))]`（Eq.4）。**〔已核，旧分析未指出〕注意这两条 KL 方向相反**——student-context 是 reverse-KL（π_S 在前），teacher-context 是 forward 方向（π_T 在前），即把学生分布往 teacher 沿可靠路径的局部分布上拉。
- 总损失 `L_total = L_stu-ctx + λ·L_tea-ctx`（Eq.5）。

**(C) top-K 方向（§3.4）**
- student-context 分支用**学生** top-K 候选（监督学生认为合理的 token）。
- teacher-context 分支用**teacher** 在选中前缀下的 top-K 候选，引入学生当前 top 之外的 teacher-preferred token。

#### 7. 实验数据集
- **训练 prompt**：DAPO-Math-17K（teacher-aligned 处理，`dapo-math-17k-processed.parquet`）。
- **评测**：AIME 2024、AIME 2025、AMC 2023，报 mean / best / majority accuracy；默认每题 k=4 解、temp 0.7、top-p 0.95。
- **模型**：主实验 teacher = **JustRL-1.5B**，student = 同尺度 **DeepSeek-R1-Distill-Qwen-1.5B**（论文称 "DeepSeek-1.5B"）；§4.2 teacher-swap 换成 DeepSeek-R1-Distill-Qwen-7B（student 仍 1.5B）。硬件 8×B200。

#### 8. 实验结果与主要发现
- **teacher-context 有用**（Table 1）：基线（2 条 student rollout）AIME24 mean 0.3917；BRTS 用 1 student + 1 teacher / **4 候选**达 AIME24 mean **0.400**、best 0.599、majority **0.4306**——**候选池越大、监督越有信息**。
- 难基准增益最大（AIME > AMC）；AMC23 上增益不明显甚至持平（Table 1 AMC23 各设置 mean 0.67–0.68，与基线 0.6777 接近）。
- **Tier-2 ground-truth 恢复**（Table 2）在 Tier-1 解不掉的难 prompt 上进一步补监督。
- Fig.3：BRTS 在训练早期达到比 student-only 基线更高的 majority 峰值。

#### 9. 结果如何支撑其主张
主张"选可靠 teacher 轨迹 + 恢复机制能改善 OPD（尤其难题）"。支撑：候选池 0/2/4 的递增对比显示选择确有增益；Tier-2 隔离实验显示恢复在难 prompt 上有效。但增益**绝对值小**（AIME24 mean 0.3917→0.400，约 +0.8 个百分点），且 AMC23 几乎无提升——支撑"难题更受益"的方向性，但量级有限。

#### 10. 逻辑自洽性（中性评估）
- 自洽：从"单样本高方差"问题，到"多采样+优先级选择+恢复"的方法，再到候选池/Tier-2 消融，链条清晰；Algorithm 1 与脚本环境变量（n_rollouts、tier1_only、fallback 等）一致。
- 张力点：(a) 增益量级小且基准窄（仅竞赛数学）；(b) teacher-context 用 forward-KL 而 student-context 用 reverse-KL，论文未深入论证方向选择的理由；(c) 主实验师生**同尺度 1.5B**，"teacher 产出超出学生范围的正确解"这一 OPD 成功前提在同尺度下是否成立存疑（teacher JustRL-1.5B 经强 RL，故仍可能强于 distill student）。

#### 11. 残留问题 / 局限
- 评测仅 AIME/AMC 竞赛数学，**泛化覆盖窄**；cs.CV 分类标签与实际数学任务不符（疑似投稿分类，待留意）。
- 增益绝对值小，AMC23 基本持平。
- 每 prompt 额外采 N 条 teacher 轨迹 + 可能的 Tier-2 重采，**计算开销**未充分量化。
- **λ 取值存在文-码不一致〔已核〕**：论文 §3.3 明确"λ=10 across all experiments"，而仓库脚本 `on_policy_distillation.sh` 默认 `AUX_TEACHER_CTX_KD_COEF=0.05`；复现需对齐（旧分析仅写 0.05、未指出论文用 10，此为本次新发现的不一致）。

#### 12. 开源代码与框架（链接+框架+代码可得性）
- 代码：https://github.com/BWGZK-keke/BRTS （resource/repos/brts，约 14MB；脚本 on_policy_distillation.sh / grpo.sh / forward.sh / forward_tier2.sh）。
- 框架：**veRL**（`python -m verl.trainer.main_ppo`；`ADV_ESTIMATOR=token_reward_direct`；rollout 用 vLLM；reward_model 即 teacher 提供 token 级 reward；swanlab 记录）。与 thunlp/OPD（rethink_opd）同源 fork（env 变量、top_k_strategy、reward_weight_mode 一致）。
- **已核脚本默认值**：N_RESPONSES=2、MAX_RESP_LENGTH=7168、LOG_PROB_TOP_K=16、TOP_K_STRATEGY=only_stu、REWARD_WEIGHT_MODE=student_p、lr 1e-6、FSDP + dynamic batch；teacher_rollout enable=True、n_rollouts=2、fallback=most_similar、tier1_only=False、n_rollouts_hint=1；**AUX_TEACHER_CTX_KD_COEF=0.05（脚本默认；注意论文用 λ=10）**；对负优势无特殊处理。forward.sh = Tier-1 only，forward_tier2.sh = 含 Tier-2。另有 prompt 扰动消融（对第二条 teacher rollout 追加 "rethink" 提示）。
- 代码可得性：方法实现完整可跑，需注意 λ 等超参与论文对齐。


---


### caopd — The Illusion of Certainty: Decoupling Capability and Calibration in On-Policy Distillation (CaOPD)

> **一句话重点 (TL;DR)**：标准 on-policy distillation（OPD/自蒸馏）在提升准确率的同时会系统性地把模型推入"过度自信"区；CaOPD 把"答什么（能力）"和"多确信（置信）"在监督目标上解耦——只把轨迹里的置信片段替换成学生自己 rollout 估出的经验成功率，其余照常做 reverse-KL 蒸馏，从而在几乎零额外改动下同时获得能力与校准。

**元信息**：arXiv 2604.16830（v1，2026-04-18）｜ Salesforce AI Research（Jiaxin Zhang、Xiangyu Peng、Qinglin Chen、Qinyuan Ye、Caiming Xiong、Chien-Sheng Wu）｜ Preprint，2026-04 ｜ 主题：OPD 的置信度校准（与 OPD 主线直接相关，高相关）｜ 代码 https://github.com/SalesforceAIResearch/CaOPD （已 clone，含完整训练/评测代码与两域数据）｜ 框架 TRL 0.24 + vLLM 0.12 + DeepSpeed 0.18 + transformers 4.57（自建 distil trainer，在 SDFT/SDPO 自蒸馏管线上做 target replacement）

#### 1. 相关工作与进展
- **OPD / 自蒸馏后训练**：近年后训练越来越依赖 on-policy distillation（Agarwal 2024；Lu & TML 2025）与自蒸馏框架。代表作 SDPO（Hübotter 2026）、SDFT（Shenfeld 2026）、OPSD（Zhao 2026）的共同套路是：让 teacher 在"特权上下文（privileged context）"——verifier 反馈、专家示范、或 ground-truth 解——下生成高质量轨迹，student 仅凭 prompt 模仿。这类方法在迁移推理能力上很成功，但几乎都是 capability-centric，不关心置信度。
- **置信度校准**：过度自信问题已被广泛记录。主流修法是 RL + reward shaping，把 Brier/proper scoring rule 罚项塞进 PPO/RL 目标（RLCR、Rewarding Doubt、CAR、Taming Overconfidence）。
- **Test-time 不确定性估计**：SelfCheckGPT、SAC³ 等用多次采样的一致性来估不确定性，可靠但推理成本 O(K)。

#### 2. 现有工作存在的问题
1. **"误校准的 Scaling Law"**：作者实测发现，不仅小模型，连前沿大模型（GPT、Claude、Gemini、DeepSeek、Kimi、Qwen3.5-397B 等）普遍落在"过自信区"——能力变强（横轴准确率右移）并不能自动修复盲目乐观。
2. **OPD 本身会加剧过自信**：在 OPD 提升准确率的同时，把平均置信推向饱和（Tool Use 上达到 0.996），过自信缺口（OCG）不降反升。
3. **RL reward shaping 有"能力税"**：RLCR、CAR 等虽能压低绝对置信，但为了躲避罚项让模型变得过度保守，准确率明显掉队（Qwen3-8B Science Q&A 上 RLCR 65.8%、CAR 61.6%，远低于 GRPO 74.5% / SDPO 80.6%）。

#### 3. Motivation
根因被归结为 **训练-部署的信息不对称（information asymmetry）**：teacher"开卷"（拿着特权上下文）产生低熵、近确定性的轨迹；student"闭卷"（只有 prompt）。当 student 去最小化对这个特权分布的 per-token reverse KL 时，被迫人为锐化 logits（熵坍缩），并继承成功轨迹那种"笃定"的表述风格（乐观偏置）。结论是：**能力可以靠模仿迁移，但"该有多确信"不能跨信息状态安全迁移**。因此应把能力和置信在监督目标上拆开。

#### 4. 主要灵感 / 核心直觉
- "答什么" vs "多确信"是两个正交目标，OPD 的标准 loss 把它们纠缠在一起：reasoning 段学到了能力，confidence 段却被逼着模仿 teacher 那个≈1.0 的笃定。
- 真正该作为置信目标的，是 **student 在部署条件下的成功率 µ(x)**——这恰好是 X-可测的最优预测（见命题 1）。而 µ(x) 可以用 student 自己的多次 rollout + verifier 经验估出来 µ̂(x)。
- 多采样估不确定性本身很贵（test-time O(K)），但可以把这笔账"摊销"到训练阶段：训练时算好 µ̂(x) 蒸进参数，部署时单次前向就能输出校准过的置信。

#### 5. 主要解决思路（一段话讲清核心）
CaOPD = **目标解耦 + target replacement**。对每个输入 x，先用 student 采 K 条 rollout、用 verifier 算经验成功率 µ̂(x)=ΣR/K；再做一次"目标替换"：把待蒸馏轨迹里的置信片段 c 改写成 µ̂(x)，同时把 teacher 特权上下文里那个原本≈1.0 的置信也改写成 µ̂(x)。然后照常跑原来的 per-token reverse-KL OPD loss——reasoning 段因为前缀和 teacher 内容都没变，等价于标准 OPD（能力克隆原样保留）；confidence 段则因为监督目标变成了 student-grounded 的 µ̂(x)，把熵坍缩和乐观偏置直接掐掉。reverse-KL 机制完全不动，无 reward 改造、无额外优化阶段。在 SDPO 下，µ̂(x) 直接复用基础训练循环已经生成的 rollout，几乎零额外成本。

#### 6. 方法详解（通俗、分步骤）
**生成格式约定**：每条生成 y=(a, c) 切成两段——推理段 a（含最终答案）+ 置信段 c（形如 "Confidence: 0.85" 的 verbalized confidence）。val(c)∈[0,1] 是解析出的标量置信。

**理论三命题（Appendix A 给全证明）**：
- **命题 1（信息差 → 不可辨识）**：当 teacher 特权上下文 Z 对正确性 R 有超出 X 的信息（条件互信息 I(R;Z|X)>0）时，teacher 条件成功率 µT(X,Z) 对 X 不可测（不存在 g(X)=µT 几乎处处成立）；且在平方误差下，X-可测的最优预测恰好是 student 部署成功率 µ(X)，残差严格为正。→ 用 teacher 的笃定当置信目标，从信息论上就是错的。
- **命题 2（特权条件 → 熵坍缩）**：当 I(A;Z|X)>0，teacher 轨迹分布的期望熵严格低于仅给 X 的条件熵；最小化对该分布的 reverse KL，会逼 student 内部 logits 人为锐化。
- **命题 3（选择偏置 → 乐观）**：特权上下文通常取自成功/高质量样本（Dhelpful），使 teacher 期望正确率 ≥ student 边际能力，于是蒸进 student 的隐式目标是对真实部署成功率的**系统性上偏**估计。

**算法（Algorithm 1）**：
1. **学生 grounded 置信估计**：对 x 采 K 条 (a_k,c_k)~π_θ(·|x)，verifier 打分，µ̂(x)=1/K·Σ R(x,a_k)。（SDPO 下复用已有 rollout，只多一次轻量 verifier 评估。）
2. **目标替换**：另采一条待蒸馏轨迹 y=(a,c)；(i) 把 c 换成 µ̂(x) 得 ỹ=(a, µ̂(x))；(ii) 构造特权上下文 z，把其中原≈1.0 的置信改写成 µ̂(x) 得 z̃。
3. **蒸馏**：在修订轨迹 ỹ 上算 per-token reverse KL（式 7）：推理位置 t∈Ia 保留"能力克隆"（与标准 OPD 等价），置信位置 t∈Ic 变成"student-grounded 校准"。AdamW 更新。

开放域无 verifier 时，µ̂(x) 可退化用 Teacher-Anchored Self-Consistency 近似（Appendix B.6）。

#### 7. 实验数据集
- **两域**：Science Q&A（Chemistry，SciKnowEval）与 Tool Use（ToolAlpaca）。
- **主模型**：Qwen3-8B、Olmo-3-7B-Instruct；scaling 分析覆盖 Qwen3 全家族 0.6B→32B。
- **对比前沿模型校准**：GPT-5.x、Claude、Gemini、DeepSeek-V3.1、Kimi-K2.5、Qwen3.5-397B 等（仅用于画"误校准 Scaling Law"散点）。
- **设置**：覆盖 SDFT / SDPO 两种 OPD 范式，外加 OOD 迁移与持续学习（CT）。
- **指标**：能力用 Accuracy；校准用 ECE、Brier Score；另引入 OCG（Overconfidence Gap = 平均置信 − 准确率）量化过自信方向，SPR（Strict Pairwise Ranking，对正确答案严格高于错误答案的概率，对置信饱和重罚）量化区分度。

#### 8. 实验结果与主要发现
- **OPD 确实加剧过自信**：Qwen3-8B Science Q&A 上，base OCG 已 +58.7%；SDFT/SDPO 进一步推到 +48.1%/+12.9%，Tool Use 上 SDFT 平均置信达 0.996（OCG +32.0%）。
- **CaOPD 结构性纠偏且不掉能力**：Qwen3-8B Tool Use OCG 从 SDFT 的 +32.0% 收到 −0.7%；主表（vs SDFT）上 ECE/BS 大幅下降、SPR 大幅回升（如 Tool Use SDFT 的 SPR 0.085 几乎丧失区分度，CaOPD 恢复到 0.555），准确率持平或略升。
- **避开 RL 的能力税**：vs SDPO 基线，CaOPD 在 Qwen3-8B Tool Use 上准确率 66.2%→70.9%，同时 ECE 0.298→0.133、BS 0.303→0.164；而 RLCR/CAR 校准虽好但准确率明显更低。
- **每步耗时与 SDPO 几乎一致**（Figure 2 右），准确率曲线与 SDPO 重合（左），校准 loss 快速收敛（中）。
- **泛化/持续学习**：OOD（Tool Use→Chemistry）下 SDFT 校准崩（ECE 0.599），CaOPD ECE 0.358（相对降约 40%）；CT 下 CaOPD 解决"校准遗忘"，CT ECE 0.126、SPR 0.662。
- **Scaling**：0.6B→32B 下 SDFT 平均置信恒为接近 1.0 的水平线；CaOPD 让置信随真实准确率动态对齐，在 Reliability(1-BS) 与 SPR 的 Pareto 前沿上全程占优；使 8B 模型校准质量媲美前沿大模型。
- **K 消融**：准确率对 K∈{1..32} 平坦（能力与置信采样方差解耦）；K≥8 是校准的高性价比甜点，K 太小目标过度量化反而困在过自信。

#### 9. 结果如何支撑其主张
- "OPD 加剧过自信"由 Table 1 的 OCG 红色扩大直接支撑，且与命题 2/3 的方向一致。
- "能力-校准可解耦"由两条曲线（准确率与 SDPO 重合、校准 loss 独立收敛）+ K 消融（准确率对 K 平坦）共同支撑，逻辑自洽。
- "避开能力税"由 vs RLCR/CAR 的准确率对照支撑，明确归因于"target replacement 不与 RL 优化器对抗"。
- "命题 1 的目标选择正确性"是理论层面给的，实证上由置信目标换成 µ̂(x) 后 ECE/BS 下降侧面印证。

#### 10. 逻辑自洽性（中性评估）
整体自洽、诊断清晰、工程改动极小（只改置信片段的监督目标，reverse-KL 机制不动）。理论命题与实证现象方向吻合。值得注意的边界条件：方法只对"verbalized confidence"这一显式格式生效；校准收益几乎全部来自置信段，对推理能力本身没有直接增益（准确率提升主要来自底层 OPD/SDPO，而非 CaOPD 的校准机制）。

#### 11. 残留问题 / 局限
- **依赖可验证 verifier 估 µ̂(x)**：开放域只能退到 self-consistency 近似，质量与覆盖面待考。
- **依赖可解析的置信格式**：测试时偶发格式失败（与所有 verbalized uncertainty 方法共有的问题）。
- **训练成本**：每 prompt 需 K 次 rollout（K=8 够用），SDFT 下不一定能像 SDPO 那样零成本复用。
- **能力上界受 base 模型约束**：CaOPD 不提升推理能力，只校准置信。
- **仅 utterance-level 置信**：长程/agentic 多步推理的 step-level 校准是 future work。
- 与 why_sd_degrade（Kim 2026，"自蒸馏为何退化推理"）是同一信息不对称现象的两种切法——此处改监督目标修过自信，彼处只做退化诊断。

#### 12. 开源代码与框架（链接 + 框架 + 代码可得性）
- 仓库：https://github.com/SalesforceAIResearch/CaOPD （已 clone，约 40M）。含 `main.py`、`distil_trainer.py`、`distil_config.py`、`eval_science.py`/`eval_tooluse.py`、Tool Use 与 Chemistry Q&A 两域数据、`scripts/`、`figures/`、Salesforce 标准合规文件（LICENSE、SECURITY 等）。
- 框架：TRL 0.24 + vLLM 0.12 + DeepSpeed 0.18 + transformers 4.57；在 SDFT/SDPO 自蒸馏管线上以"采样估 µ̂ → 替换置信 token → 原 reverse-KL 训练"的方式插入，核心改动集中在 distil trainer 的 target replacement。代码可得性：训练 + 评测 + 数据齐全，可复现性较好。


---


### copsd — Crosslingual On-Policy Self-Distillation for Multilingual Reasoning (COPSD)

> **一句话重点 (TL;DR)**：低资源语言（尤其非洲语）数学推理差，是因为模型"有 latent 能力但调不出来"。COPSD 把 OPSD 的特权上下文 self-distillation 搬到跨语言场景：student 只看译成低资源语的题、必须用目标语推理；teacher 是同一个模型但额外拿到英文题 + 英文参考解，从而诱导出更可靠分布；在 student 自己的 rollout 上做逐 token reverse-KL 蒸馏。无需外部 teacher、无需目标语 rationale。

**元信息**：arXiv 2605.09548（v1，2026-05-10，cs.CL）｜ LMU Munich (CIS) + MCML（Yihong Liu*、Raoyuan Zhao*、Michael A. Hedderich、Hinrich Schütze）｜ Preprint ｜ 主题：OPSD 的跨语言变体（与本项目 OPD 主线直接同源——把 siyan-zhao/OPSD 那套特权上下文自蒸馏换个输入条件搬到低资源语言数学推理）｜ 代码 https://github.com/cisnlp/COPSD （README 声明 fork 自原始 OPSD 代码库 siyan-zhao/OPSD，已 clone）｜ 框架 HuggingFace TRL + Accelerate/DeepSpeed，LoRA，A100/H200

#### 1. 相关工作与进展
- **On-Policy Distillation**：结合 student 自生成轨迹的 on-policy 监督与 dense token 级 teacher 反馈，缓解 train-inference mismatch、避免稀疏序列奖励。OPSD（Zhao 2026b）/ Sang 2026 / Zhang 2026a 进一步用**同一模型**在不同条件下分饰 student/teacher，无需外部 teacher。
- **多语言推理**：LLM 跨语言性能差距大，低资源语言尤甚，且常产生语言混杂或不一致的推理轨迹。现有解法：translate-and-test、SFT（机翻 rationale）、self-training、RL——多需翻译过的推理 rationale 或稀疏 outcome 奖励。

#### 2. 现有工作存在的问题
- **机翻 reasoning trace 噪声大、off-policy**：数学表达/数量/逻辑依赖易在翻译中出错，且与模型自身推理行为不匹配，造成 train-inference 分布失配。
- **outcome-only RL（GRPO）稀疏、不稳**：低资源语言下模型很少产生正确答案，二元奖励几乎无信号、样本低效。
- 模型有"latent ability"，但在低资源语言下调不出来。

#### 3. Motivation
让模型把自己"用英文时"可及的推理行为迁移到低资源语言：student 只看低资源问题、必须用目标语推理（匹配推理时条件）；teacher 是同一个模型，但额外获得 privileged 英文上下文（英文问题 + 英文参考解），从而诱导出更可靠的分布。这样既无需外部 teacher、也无需目标语 rationale，且 dense token 级监督比稀疏 outcome 奖励信号强得多。

#### 4. 主要灵感 / 核心直觉
- OPSD 公式不变，只把 teacher 的特权信息从"参考解"扩成"英文问题 + 英文解"，输入端从原语言换成低资源语言——这是 OPSD 的领域迁移应用，增量有限。
- 用 prompt-hacking 强制 student 和 teacher 都用目标语推理：在 `<think>` 后插入语言特定前缀（否则模型会切回英文推理）。

#### 5. 主要解决思路（一段话讲清核心）
两个 policy 来自同一 LLM p_θ。Student p_S(·|x_L) 只看低资源问题 x_L；Teacher p_T(·|x_L, x_H, y*) 额外条件英文问题 x_H 与英文参考解 y*。Student 在线生成目标语 rollout ŷ；两者在同一前缀上算逐 token 分布，最小化轨迹平均的 token-level 散度 D(p_T‖p_S)，论文实例化为 **reverse KL + full-vocabulary logit distillation**；梯度只过 student，teacher 作为固定（frozen）分布目标。本质 = OPSD 公式不变，只是把特权信息扩成"英文题+英文解"、输入换成低资源语言。

#### 6. 方法详解（通俗、分步骤）
1) 取英文数学题 x_H + 英文参考解 y*；2) 用 Gemini-3-Flash 把问题机翻到目标低资源语 x_L；3) student 看 x_L 在线生成目标语 rollout ŷ（max 2048 token），用 prompt-hacking 强制目标语推理；4) teacher（frozen 同模型）以英文特权上下文评估同一 ŷ；5) 逐 token reverse-KL（full-vocab）蒸馏，LoRA 更新 student。

**〔代码核查〕实现细节与论文表述的出入**：
- **散度**：`generalized_jsd_loss` 支持 forward/reverse KL 与广义 JSD，由 `beta` 选择（beta=0 即 reverse KL）；发布的 4B 训练脚本 `--beta 0`，与论文 reverse-KL 一致。
- **"full-vocabulary" 与发布脚本不符**：4B 脚本实际用 `--top_k 20`（只在 teacher top-20 token 上算散度并重归一化，而非论文正文所述 full-vocab），并加 `--jsd_token_clip 0.05`（逐 token 散度截断，抑制 style token 主导梯度）——与论文"full-vocabulary"表述略有出入。
- **teacher 固定方式**：`fixed_teacher` 通过在 teacher forward 时 `disable_adapter()` 实现——即用 base 模型（无 LoRA adapter）当 teacher，梯度只过带 adapter 的 student（同一份基座权重）。4B 脚本用 `--fixed_teacher`。
- **LoRA**：4B 脚本 `lora_r=64`、`lora_alpha=128`，目标模块 q/k/v/o/gate/up/down_proj，lr 5e-6。
- **代码额外能力（论文未强调）**：trainer 还实现了 EMA teacher（`use_ema_teacher`，与 fixed_teacher 互斥）、`reason_first` 模式（teacher 先对英文参考解推理再评估 student），以及 thinking-machines RL 式 reverse-KL（仅在采样 token 上的 policy-gradient `advantage=(teacher_logp − student_logp).detach()`）——发布主实验用的是 JSD/KL 全分布路径 + fixed_teacher。

#### 7. 实验数据集
- **训练**：OpenThoughts（Guha 2025）采样 0.5K 数学题，问题用 Gemini-3-Flash 译成 17 种 AfriMGSM 非洲语；英文问题 + 英文解作 teacher 特权信息。
- **评测**：AfriMGSM（Adelani 2025，17 语 × 250 题，Pass@12，主表 4096-token budget）；PolyMath（更难，8 语 × 125 题，8192-token budget）。
- **模型**：Qwen3-1.7B / 4B / 8B。
- 指标：Pass@12（每题采 12，至少一条正确即算对），Math-Verify 抽 \boxed{} 比对。

#### 8. 实验结果与主要发现
- **AfriMGSM 主表（Table 1，Pass@12，4096-token）**：1.7B 平均 9.11→15.53、4B 19.20→20.61、8B 19.41→23.55，全面超 base 与 GRPO；GRPO 几乎无提升（1.7B 仅 9.11→9.18，多语言上偶尔还低于 base）——印证低资源下 outcome RL 信号太稀疏。1.7B 相对提升超 70%，在类型学/正字法多样的语言上普遍受益。
- **训练动态（Figure 3）**：COPSD 早期快速提升 Pass@12 与 format rate；4B/8B 几步内达峰后缓降（可能因目标语生成能力弱、teacher 信号有限、继续训练过拟合不完美 teacher 信号）；GRPO 无明显上升趋势。
- **format adherence**：format rate 与 Pass@12 强正相关（mean Pearson 0.628/0.838/0.728），说明低资源失败部分源于"在有限 token 预算内产不出规定格式答案"。
- **test-time scaling（Table 3）**：大模型更稳定受益于更长生成预算；8B COPSD 从 1024→4096 token 提升 30.0%（GRPO 仅 13.8%），即 COPSD 强化了利用更长目标语推理轨迹的能力。
- **repeat rate（Figure 5）**：COPSD 显著降低 4-gram 重复率，缓解多语言推理常见的重复退化。
- **PolyMath 泛化（Figure 6）**：跨难度普遍超 base，**低资源语言增益最大**（medium 难度 Swahili +32.0、Telugu +32.8、Bengali +15.2；high 难度 Swahili +18.4、Telugu +16.8），高资源语（日/中/俄/西）增益小——支撑"模型已有 latent 能力但难经低资源语表达"的假设。

#### 9. 结果如何支撑其主张
- "dense 跨语言监督 > 稀疏 outcome RL" 由 COPSD 全面超 GRPO（GRPO 几乎不动）直接支撑。
- "迁移 latent 英文推理能力" 由 PolyMath 上低资源语增益远大于高资源语支撑（高资源语本就被 base 覆盖、无可迁移空间）。
- 但需在背景下解读：**teacher 拿到了英文参考解 y* 本身**，等于把答案信息蒸进 student——这是"特权"但也意味着 teacher 分布很接近"直接给答案"，提升幅度应在此背景下看待。

#### 10. 逻辑自洽性（中性评估）
方法自洽、主张与证据对齐，是 OPSD 的清晰跨语言迁移。增量有限——OPSD 公式不变，新意在"用英文上下文当跨语言特权信息"的设定与 17 语系统验证。需注意两点诚实暴露的张力：(1) 论文称 full-vocab，发布脚本实为 top_k=20 + token clip，表述与实现有出入；(2) teacher 见参考解，使"特权"接近"漏答案"，绝对增益（尤其 4B 仅 19.2→20.61）应保守解读。4B/8B 几步达峰后下降也说明 teacher 信号在弱目标语上很快饱和。

#### 11. 残留问题 / 局限
- **依赖英文参考解**：高质量英文监督不可得、或另一高资源语更合适时受限。
- **训练题靠机翻**：题面翻译 artifact 仍可能影响训练质量（虽不需翻译 rationale）。
- **同模型当 teacher**：目标语能力弱时 teacher 分布仍不完美，学习信号易快速饱和甚至继续训练后退化（4B/8B 已观察到）。
- 论文"full-vocabulary"表述与发布脚本（top_k=20 + jsd_token_clip）不符，复现需以脚本为准。

#### 12. 开源代码与框架（链接 + 框架 + 代码可得性）
- 仓库：https://github.com/cisnlp/COPSD （README 明确 fork 自 siyan-zhao/OPSD，已 clone）。含 `multilingual_opsd_trainer.py`（self-distillation loss，OPSDTrainer 继承 TRL SFTTrainer，含 fixed_teacher/EMA/reason_first/JSD 与 thinking-machines reverse-KL 两条 loss 路径）、`multilingual_data_collator.py`（student/teacher prompt 构造）、`multilingual_grpo_train.py`（GRPO baseline）、`language_config.py`、17 非洲语 + PolyMath 评测脚本、`multilingual_scripts/run_all_opsd_4b_3000.sh`（4B 复现脚本）。数据集放在 HF。
- 框架：HuggingFace TRL + Accelerate/DeepSpeed（含 vLLM rollout、ZeRO-3 支持），LoRA 训练（4B：r=64/α=128，lr 5e-6，beta=0 即 reverse KL，top_k=20，jsd_token_clip=0.05，fixed_teacher）。A100/H200。代码可得性：训练 + 评测脚本齐全，但复现需注意脚本超参与论文正文的若干出入。


---


### csd — Distillation of Large Language Models via Concrete Score Matching (Concrete Score Distillation, CSD)

> **一句话重点 (TL;DR)**：CSD 提出一种 **logit 级**的离线知识蒸馏目标，用"离散 concrete score 匹配"替代传统的 softmax 概率匹配（KL/f-散度），既避免 softmax 把 teacher 的大 logit 差异"压平"，又比直接对齐 logit（DLD）拥有更大的最优解集（对 logit 常数平移保持不变），并给出 O(|V|) 线性时间梯度。

**元信息**：arXiv 2509.25837 ｜ KAIST + summary.ai（Yeongmin Kim, Donghyeok Shin, Mina Kang, Byeonghu Na, Il-Chul Moon）｜ v3 2026-05-30（ICLR 2026 接收，cs.LG）｜ 主题 离线 logit-level KD（与本项目仅在"蒸馏 loss 设计"层面外围相关，非在线/RL 推理蒸馏）｜ 代码 https://github.com/aailab-kaist/CSD（仅 README 占位，**训练代码尚未释出**）｜ 框架 即插式 KD 目标，可嵌入 ImitKD/GKD/DistiLLM

#### 1. 相关工作与进展
知识蒸馏（KD）让小 student 继承大 teacher，以降低推理成本。主流 KD 在每个 token 上做 softmax 概率匹配——最常见是前向/反向 KL，以及各类 f-散度与平滑变体。近期 on-policy KD（ImitKD、GKD、DistiLLM）改进的是"用谁生成的数据训练"（student on-policy / 混合 / 自适应选择），而 CSD 改进的是与之正交的"散度目标本身"。CSD 借鉴能量模型中的 score matching（Hyvärinen 2005）与离散变量的 concrete score（Meng et al. 2022），把后者适配到自回归 LLM 蒸馏。

#### 2. 现有工作存在的问题
- **softmax 平滑抹掉 logit 级知识**：当 teacher 的 logit 差异很大时（如 [−1,−4,4] vs [1,−9,6]），经 softmax 后两者概率几乎一致、梯度近乎相同，淹没了 teacher logit 里编码的细粒度知识。在大词表下概率分布极度稀疏——论文统计 GPT-2-1.5B 上仅 **0.0023%** 的 token 概率大于 0.01。
- **Direct Logit Distillation (DLD) 解集受限**：DLD 直接对齐 logit、绕过 softmax，但其最优解**不允许 logit 的常数平移不变性**（而 softmax 推理下 logits 整体加常数等价）。这严重收窄解集，在 teacher/student 容量差异大时尤甚。

#### 3. Motivation
能否设计一个 logit 级目标，**同时**克服"softmax 平滑"与"DLD 解集受限"，并带可证明的最优性保证（解集应是 DLD 解集的超集）？

#### 4. 主要灵感 / 核心直觉
score matching 在能量模型中可绕开 sum-to-one 归一化约束。把它的离散版本（concrete score，刻画"换到另一 token"的相对概率变化）搬到 LLM 蒸馏上：只要 student/teacher 在**所有词表对**上的相对 logit 差对齐即可——这天然对 logit 常数平移不变，从而比 DLD 多出一整族等价解。

#### 5. 主要解决思路(一段话讲清核心)
定义 concrete score sθ(y)=[qθ(x)/qθ(y)]_{x∈V}，把蒸馏目标设为匹配 student 与 teacher 的 concrete score。为适配 LLM，做两处工程处理：(a) 概率比 qθ(x)/qθ(yt) 易发散导致训练不稳，改用其 **log 变换**形式；(b) 朴素双重词表求和是 O(|V|²)，在可分权重假设下降到 O(|V|)。最终目标归约为"匹配所有词表对上的相对 logit 差，权重可调"，并可在同一框架内实例化 mode-seeking 与 mode-covering 两类行为。

#### 6. 方法详解(通俗、分步骤)
1. **构造 concrete score**：对每个位置，用 student logits 算出"当前 token 换成词表中任一其它 token"的相对概率比，对 teacher 同理。
2. **log 变换稳定训练**：直接用概率比会发散，改对齐 log 形式（论文 §"adopt the logarithm"），得到 CSD 目标 L_CSD。
3. **理论保证**：
   - **Proposition 1（一致性）**：模型容量趋于无穷时，匹配 log-concrete-score 可使 student 收敛到 teacher。
   - **Theorem 2（解集超集）**：Θ*_CSD ⊋ Θ*_DLD——DLD 能达到的解 CSD 都能达到，且 CSD 因对 logit 常数平移不变而拥有更多解。
   - **Theorem 3（高效梯度）**：在权重可分假设 w(yt,x)=w1(yt)·w2(x) 下，梯度可在 **O(|V|)** 线性时间算出（Algorithm 1）。
4. **更一般权重的退路**：若不接受可分假设，可用 **Monte Carlo 估计**梯度（不需独立性假设，但方差更大、收敛略慢）。
5. **即插使用**：把 CSD(S,S) 与 DLD(S) 损失叠加进 ImitKD/GKD/DistiLLM——这三者的差异在于训练数据来源（ImitKD 纯 student on-policy、GKD 混合、DistiLLM 按验证损失自适应选择）。

#### 7. 实验数据集
- 蒸馏数据：databricks-dolly-15k（沿用 DistiLLM 设置）；评测集 Dolly Eval、Self-Instruct 等。
- 任务：task-agnostic 指令跟随、task-specific（摘要/数学/翻译）、通用 chat 蒸馏。
- backbone/teacher：GPT-2(0.1B/0.3B/1.5B)、OpenLLaMA-7B、Gemma-7B-IT、Qwen2.5-7B-IT、**Gemma2-9B-IT**（最大 teacher 9B）。

#### 8. 实验结果与主要发现
- 先在 dolly 上微调 teacher，再蒸馏 student；ROUGE-L 跨 5 个随机种子取平均，并用 Self-BLEU 衡量多样性。
- 基线覆盖 KL/RKL、各类 f-散度、DLD（及 DLD-mean 中心化变体）、ImitKD、GKD、DistiLLM。
- 结论：CSD 一致优于近期概率匹配目标与 DLD，并位于"多样性–保真"前沿；与 on-policy 技术（ImitKD/GKD/DistiLLM）结合时呈互补增益。

#### 9. 结果如何支撑其主张
理论侧（Prop.1 / Thm.2 / Thm.3）直接支撑"解集更大 + 可线性计算"的核心卖点；实验侧通过相对概率匹配目标与 DLD 的对比、以及与三种 on-policy 框架叠加的增益，支撑"logit 级知识被更好保留"的主张。多样性/保真前沿图支撑"不牺牲多样性"的说法。

#### 10. 逻辑自洽性(中性评估)
内在逻辑自洽：从"softmax 抹平 logit 差 + DLD 解集受限"两个具体缺陷出发，给出兼顾两者的目标并配理论。但需注意 Thm.3 的 O(|V|) 依赖**权重可分假设**，更一般情形退回方差更大的 Monte Carlo——即"高效"与"通用"二者不可兼得，这一权衡论文有交代。

#### 11. 残留问题 / 局限
- 假设 teacher 与 student **共享词表/tokenizer**，跨族蒸馏不适用。
- O(|V|) 仅在可分权重下成立；一般权重需 Monte Carlo（方差更大）。
- 评测以 ROUGE-L/Self-BLEU 等**代理指标**为主，模型规模偏中小（最大 9B teacher），未在大规模推理任务上验证。
- 属**离线 logit-level KD**，与本项目关注的在线/RL/推理路径蒸馏只在 loss 设计层面相关。
- 〔待核：GitHub 代码尚未填充，复现性无法独立验证〕

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库：https://github.com/aailab-kaist/CSD （已 clone，~83KB；**当前仅 README 占位**"Official repo for CSD (ICLR 26)"，无训练代码）。RepoExists=YES 但内容近乎空。
- 框架：非独立训练框架，而是一个可替换 KL 的 **logit-level 蒸馏目标**，设计为嵌入现有 KD 流程（ImitKD/GKD/DistiLLM）。


---


### dasd — Distribution-Aligned Sequence Distillation (DASD-4B-Thinking)

> **一句话重点 (TL;DR)**：DASD 从"分布对齐"视角改造序列级蒸馏（即在 teacher 响应上做 SFT），用三件套（温度调度学习、散度感知采样、混合策略蒸馏）补回缺失的师生交互，仅 **448K** 样本就让 Qwen3-4B 在推理基准上达到同量级 SOTA，部分基准超过若干 32B 模型。

**元信息**：arXiv 2601.09088v1（2026-01-14 提交，技术报告）｜ 阿里云（Shaotian Yan*, Kaiyuan Liu*, Chen Shen*†, Bing Wang*, Sinan Fan* 等）｜ 主题 序列级蒸馏改进、Long-CoT 推理（与 OPD/distillation 强相关；含一个轻量 mixed-policy 阶段直接缓解 exposure bias，呼应 prefix-OPD / 路径恢复思路）｜ **纯 SFT 式蒸馏，不含 RL** ｜ 代码 https://github.com/D2I-ai/dasd-thinking ｜ 框架 LLaMA-Factory + DeepSpeed ZeRO-3 + Liger-Kernel

#### 1. 相关工作与进展
DeepSeek-R1 首次证明"从强 teacher 蒸馏"可大幅赋能小模型推理，引发社区大量复刻（OpenR1、OpenThoughts、a-m-team、AceReason、LIMO、s1、Light-R1 等）。主流范式即 **SFT on teacher-generated responses = 序列级蒸馏（Kim & Rush 2016）**：简单高效、不限师生架构、无需 token-level logits。另一范式是 logit 蒸馏（Qwen3/Gemma 的 on-policy 变体、Thinking Machines Lab 实现），但需访问 logits 且跨 tokenizer 难对齐。

#### 2. 现有工作存在的问题
现有序列级蒸馏多停留在 SFT 视角，只设计启发式数据过滤规则，**忽视蒸馏本质——让 student 学到 teacher 完整输出分布以继承泛化能力**。三大缺陷：
- (i) **teacher 序列级分布表征不充分**：随机采样 + 质量过滤难覆盖分布全支撑，模式覆盖差，或过度表征低概率/噪声序列；
- (ii) **teacher 输出分布与 student 学习能力错配**：SFT 只抬高 ground-truth token 概率，会产生误导梯度（对 teacher 低概率但 student 已高概率的 token 仍继续抬高）；
- (iii) **exposure bias**：teacher-forced 训练与自回归（free-running）推理不一致。
根因是全程缺乏显式师生交互。

#### 3. Motivation
不改用 logit 蒸馏（保留序列级蒸馏简单/高效/无 logit 依赖的优点），而是从"分布对齐"角度补回师生交互：让 student 更好覆盖并对齐 teacher 序列级分布、找到更利于 student 学习的目标分布、再缓解 exposure bias。

#### 4. 主要灵感 / 核心直觉
温度控制覆盖–一致性的权衡（低温样本一致易学但覆盖窄，高温覆盖广但难学）；师生预测概率在候选响应上的差异可归为若干典型模式，其中"teacher 高置信 + student 低概率"这一高散度模式与测试性能提升正相关；exposure bias 可用"student 生成前缀 + teacher 续写"的少量混合数据廉价修正。

#### 5. 主要解决思路(一段话讲清核心)
在两阶段 off-policy SFT 的基础上，用**温度调度**先学一致模式再扩覆盖、用**散度感知采样**优先喂"高散度"实例以对齐 student 学习能力、再加一个轻量**混合策略蒸馏**阶段缓解 exposure bias，构成一条增强的序列级蒸馏 pipeline。

#### 6. 方法详解(通俗、分步骤)
1. **Temperature-scheduled Learning（温度调度学习）**：低温样本模式一致、易学但覆盖窄；高温样本覆盖 teacher 更多模式但多样大、学习效率低。故两阶段——先在低温高置信样本上训抓一致模式，再渐进加入高温样本扩覆盖，优于单一温度的单阶段训练。
2. **Divergence-aware Sampling（散度感知采样）**：提出分布分解框架，分析师生预测概率在候选响应上的差异，归纳四种典型分布模式；发现"**teacher 高置信 + student 低概率**"的高散度模式与测试性能提升一致相关，故引导 student 优先学习此类实例，自然缓解误导梯度。〔该思路与 ICLR 2026 论文 arXiv:2512.20908 "Where Did This Sentence Come From? Tracing Provenance…" 相呼应。〕
3. **Mixed-policy Distillation（混合策略蒸馏）**：在初始 off-policy SFT 后加一个轻量阶段——随机选小子集，让训练后的 student 生成完整响应，随机截断前缀，再由 **teacher 从截断点续写**，仅保留通过质量过滤的续写用于 student SFT。少量数据/少量步即缓解 exposure bias 并使输出更简洁。

#### 7. 实验数据集
- teacher **gpt-oss-120b**；student **Qwen3-4B**（MoE 变体用作 DASD-30B-A3B）。师生在规模/架构/词表/tokenizer/预训练语料上差异巨大，验证跨族兼容性。
- 训练集仅 **448K** 样本（比多数开源工作少一个数量级），跨数学/代码/科学推理/复杂指令多域；温度消融用 50K 数学响应（gpt-oss-120b，高/低温）。内部名 Apsara-Reason-v1-SFT-stage1/stage2（YAML），对外发布名 Superior-Reasoning-SFT-gpt-oss-120b（README，stage1 低温 105K + stage2 高温 330K）。
- 评测：AIME24、AIME25、LiveCodeBench v5、GPQA-Diamond 等。

#### 8. 实验结果与主要发现
- 两阶段 SFT（stage1.yaml / stage2.yaml，DeepSpeed ZeRO-3）：均为 full SFT、cutoff_len=65536、packing=true、per_device_bs=1、grad_accum=4、lr=5e-5（cosine_with_min_lr，min_lr=1e-5）、warmup_ratio=0.1、weight_decay=0.1、num_train_epochs=6、bf16。Stage-2 从 stage1_checkpoints 续训。
- 两阶段数据均经 divergence-aware sampling 生成以对齐分布与能力，并沿用严格质控（过滤截断/重复）；最后加 mixed-policy 阶段。DASD-30B-A3B-Preview 因时间限制仅用 Stage-1 数据训练。
- 结果：DASD-4B-Thinking 在同量级开源模型中达 SOTA，并在关键基准上超过若干 32B 级模型（论文柱状图对照 AM-Think-v1-32B、Qwen3-32B、GLM-Z1-32B 等）。

#### 9. 结果如何支撑其主张
温度消融（50K 数学）支撑"调度优于单温度"；散度感知采样的性能相关性分析支撑"高散度模式对齐能力"；mixed-policy 的输出长度/一致性改善支撑"缓解 exposure bias"。仅 448K 样本达到/超越更大模型，支撑"分布对齐比堆数据更重要"的核心论点。

#### 10. 逻辑自洽性(中性评估)
三件套各自对应 §2 的三个缺陷，逻辑闭环清晰。但论文是**技术报告**，三个机制多为联合呈现，缺少严格的逐项可加性消融与统计显著性；"高散度模式与性能正相关"为观测性结论，因果关系未做隔离验证。

#### 11. 残留问题 / 局限
- **论文文本与发布代码不一致**：正文多处写 student 为 **Qwen3-4B-Instruct-2507**（§lines 160/287/319/324），但 train/stage1.yaml 的 `model_name_or_path` 实为 **Qwen3-4B-Thinking-2507**，README 性能表也以 Thinking-2507 为同族基线对照——**以代码为准应为 Thinking 变体**。〔已逐一核对，结论维持。〕
- 纯 SFT 式蒸馏，未与 RL 结合，无 token-level 对齐。
- 技术报告体例，消融与显著性较弱。
- 散度感知采样依赖能拿到师生 logprob（发布了 -Logprob 数据集），跨任务可迁移性待验证。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库：https://github.com/D2I-ai/dasd-thinking （已 clone，~5.2MB；含 train/stage1.yaml、stage2.yaml、deepspeed/ds_z3_config.json、assets/、dasd_technical_report.pdf）。**含训练配置但不含数据生成/采样代码**。
- 框架：**LLaMA-Factory**（README 明确"We utilize LLaMA-Factory framework for training"）+ DeepSpeed ZeRO-3 + Liger-Kernel（enable_liger_kernel=true）；纯 SFT/序列级蒸馏栈，非 RL 框架。
- 模型/数据开源于 HF & ModelScope：DASD-4B-Thinking、DASD-30B-A3B-Thinking-Preview、数据集 Superior-Reasoning-SFT-gpt-oss-120b（及 -Logprob 版）。


---


### distillm — DistiLLM: Towards Streamlined Distillation for Large Language Models

> **一句话重点 (TL;DR)**：把自回归 LM 蒸馏的两大痛点分别治理——用有理论保证的 **skew KLD** 替代不稳定的 KLD/RKLD 作目标函数，用带 replay buffer 的**自适应 off-policy 策略**把昂贵的学生生成输出（SGO）生成频率压到最低；在保持/超过 SOTA 蒸馏质量的同时把训练提速 2.5–4.3×。

**元信息**：arXiv 2402.03898 (v2, 2024-07-03) ｜ KAIST AI（Jongwoo Ko, Sungnyun Kim, Se-Young Yun）+ Microsoft（Tianyi Chen）｜ ICML 2024（2024-02 首发）｜ 主题 T1 / High（白盒 LM 蒸馏目标函数 + 训练效率）｜ 代码 https://github.com/jongwooko/distillm（已克隆约 853KB）｜ 框架 自研（沿用 MiniLLM 代码基 + 定制 HF Transformers + DeepSpeed，无外部 RL 框架）。

#### 1. 相关工作与进展
- **白盒 KD**：用 KLD（前向）或 RKLD（反向）让学生匹配教师分布。
- **on-policy / SGO 类**：GKD、MiniLLM 等用**学生生成输出（SGO）**做蒸馏，以缓解训练-推理分布不匹配；但 SGO 生成极慢。
- 散度选择上各方法莫衷一是（KLD vs RKLD vs JSD），缺统一标准。

#### 2. 现有工作存在的问题
- **目标函数缺理论支撑**：最优散度任务相关（GKD 观察），KLD 可能梯度爆炸或泛化/收敛性不足。
- **朴素用 SGO 的两大问题**：(i) 教师对不熟悉/不准确的 SGO 给出**误导性反馈**（给短而错的低 loss、给长而对的高 loss）；(ii) **每步都生成 SGO 极低效**——SGO 生成可占训练总时长很大比例（论文报告近期 on-policy 方法慢 3–7×，甚至单独 SGO 生成占 80% 训练时间）。

#### 3. Motivation
需要一个**同时**兼顾蒸馏质量与训练效率的框架：用有理论保证的散度修掉目标函数的不稳定，用自适应 off-policy 调度在最小化 SGO 生成频率的前提下，平衡"减小训练-推理不匹配的正效应"与"噪声反馈的负效应"。

#### 4. 主要灵感 / 核心直觉
- **混合分布稳梯度**：在教师/学生概率间插值，防止 KL 中分母趋零导致的梯度爆炸——即 skew（偏斜）KLD。
- **SGO 可缓存复用**：训练-推理不匹配的修正不必每步重生成 SGO；早期多用新鲜 SGO 降 bias，后期多复用 buffer 提效率，并随验证 loss 自适应决定生成强度。

#### 5. 主要解决思路（一段话讲清核心）
两大组件协同：(1) skew KLD/SRKL——把目标函数从 KL(p,q) 改成 KL(p, αp+(1-α)q)（及反向版），用插值避免分母趋零、提供更稳的梯度，理论上更大 α 降低经验估计的 L2 范数误差，权衡后最优 α≈0.1；(2) 自适应 off-policy——用一个 SGO Scheduler 按验证 loss 自适应增大 SGO 使用概率 φ，并用 replay buffer 存历史 SGO、以线性递减的 replay ratio 控制重新生成的频率。SKL 的快速收敛使 off-policy 能在低 bias 下生效。

#### 6. 方法详解（通俗、分步骤）
1. **Skew KLD（SKL / SRKL）**：
   - 前向 D^(α)_SKL(p,q)=KL(p, αp+(1-α)q)；反向 D^(α)_SRKL(p,q)=KL(q, (1-α)p+αq)。
   - 代码（`distillm/losses.py` 第 66–95 行，已核对）：`skewed_forward_kl` 用 `mixed = lam*teacher + (1-lam)*student`、`skewed_reverse_kl` 用 `mixed = (1-lam)*teacher + lam*student`，默认 `lam=0.1`。
   - 性质（Thm.1）：更大 α 降低经验估计的 L2 范数误差；权衡后最优 α≈0.1，优于 KLD/RKLD/JSD。
2. **自适应 off-policy 方法**：
   - **Adaptive SGO Scheduler**：SGO 使用概率 φ 从低（实验初始 φ=0）起，依验证 loss 自适应增大（验证 loss 上升则增 φ）。
   - **Off-policy + replay buffer**：`distillm/buffer.py` 用 `deque(maxlen=capacity)` 存 SGO（实验 capacity≈1000），按线性递减 replay ratio λ_R=φ(1-t/T) 控制生成频率——早期多用当前 SGO 降 bias，后期多复用 buffer 提效率。
3. **协同**：SKL 快速收敛使 off-policy 在低 bias 下生效；默认配置 = SRKL + off-policy + α=0.1。

#### 7. 实验数据集
- **指令遵循**：databricks-dolly-15k（14K 训练 + 各 500 验证/测试），评 Dolly/Self-Instruct/Vicuna/Super-Natural/Unnatural；加 OpenWebText 语言建模辅助 loss。
- **文本摘要**：SAMSum（另 XSum、CNN/DM 在附录）。
- **机器翻译**：IWSLT 2017 En-De。
- 指标：ROUGE-L、GPT-4 feedback、BLEU（翻译）。

#### 8. 实验结果与主要发现
- **教师→学生对**：GPT-2 XL(1.5B)→GPT-2(0.1B)、OPT-2.7B→1.3B、OpenLLaMA2-7B→3B（7B 用 LoRA）；摘要/翻译用 T5-XL/mT5-XL→T5/mT5-Base/Small。4×40G A100 训练。
- **效率**：相比近期 KD，训练提速 2.5–4.3×；DistiLLM 仅需朴素 KD 的约 1.6× 时间，而 MiniLLM/GKD 需 3–7×。
- **质量**：在指令遵循/摘要/翻译上达到 SOTA 蒸馏性能。
- **额外性质**：支持一阶段蒸馏（无需先 SFT 学生），对学生初始化更鲁棒。

#### 9. 结果如何支撑其主张
- "目标函数更稳更好" → 消融中 SKL/SRKL（α≈0.1）优于 KLD/RKLD/JSD，与 Thm.1 的 L2 误差界方向一致。
- "高效" → 训练墙钟时间相对 MiniLLM/GKD 显著下降（2.5–4.3×），由 SGO 生成频率被 scheduler+buffer 压低直接解释。
- "鲁棒/可一阶段" → 跨多组教师-学生对、多任务的稳定增益支撑该主张。

#### 10. 逻辑自洽性（中性评估）
两组件目标清晰、与代码一一对应（skew 混合系数、replay deque、φ 调度），机制叙事自洽。值得注意：skew KLD 本质是"插值平滑"，其稳定性收益与温度平滑/标签平滑有概念重叠；"最优 α≈0.1"是经验权衡点，Thm.1 给的是单调方向而非闭式最优。off-policy 的 bias-效率权衡由线性 schedule（而非自适应最优）控制，属工程启发式。整体证据支撑主张，但"理论保证"更多是定性方向性而非强保证。

#### 11. 残留问题 / 局限
- **模型规模偏小且偏旧**：主力实验在 GPT-2/OPT/OpenLLaMA2 等老模型，最大 7B 且用 LoRA，现代大模型上的结论需外推。
- **α、φ、capacity 等为经验超参**：最优 α≈0.1、buffer≈1000、φ schedule 均为调出来的，缺自适应最优依据。
- **skew 的概念新颖性有限**：插值平滑与既有平滑技巧重叠，贡献更多在"系统化 + 理论方向 + 与 off-policy 协同"。
- **白盒前提**：需教师 logits（同 tokenizer/词表），不适用黑盒蒸馏。

#### 12. 开源代码与框架（链接+框架+代码可得性）
- 仓库 https://github.com/jongwooko/distillm（已克隆，约 853KB）。核心实现在 `distillm/`：`losses.py`（skewed_forward/reverse_kl，lam=0.1）、`buffer.py`（replay deque）、`sampler.py`、`__init__.py`；另有 `minillm/`（沿用的 MiniLLM 代码基）、`scripts/`（gpt2/opt/openllama2 的 sft/kd/seqkd/imitkd/minillm/gkd/distillm 各基线）、`train_minillm.py`、`finetune.py`、`generate.py`、`install.sh`。
- 框架：自研，基于 MiniLLM 代码基 + 定制 HF Transformers（README 注明"based on this commit of HF Transformers by following MiniLLM"，24.08.12 起已解除旧版依赖）+ DeepSpeed；无外部 RL 框架。
- 代码可得性：高（含全部基线复现脚本与核心 loss/buffer 实现，可直接复现）。


---


### distillm2 — DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs

> **一句话重点 (TL;DR)**：观察到 KL 与 RKL 的非对称行为——KL 在教师生成数据（TGO）上"抬高"、RKL 在学生生成数据（SGO）上"压低"——于是设计对比式蒸馏 loss（CALD）：对教师响应用 SKL、对学生响应用 SRKL，并配 α 课程与 β 线性递增，在指令/数学/代码/VLM 全面超过 GKD、DistiLLM、Speculative KD。

**元信息**：arXiv 2503.07067 (v2, 2025-05-30) ｜ KAIST AI（Jongwoo Ko, Sungnyun Kim, Se-Young Yun）+ Microsoft（Tianyi Chen, Tianyu Ding, Luming Liang, Ilya Zharkov）｜ ICML 2025 Oral（top 1%）｜ 主题 T1 / High ｜ 代码 https://github.com/jongwooko/distillm-2（已克隆约 5.9MB，2025-06 正式发布）｜ 框架 HF alignment-handbook + Accelerate + DeepSpeed ZeRO-3 + vLLM 生成 + FlashAttention-2（无 veRL/TRL）。

#### 1. 相关工作与进展
- **白盒 KD**：DistiLLM（前作）提出 skew KLD（SKL/SRKL）作稳定目标。
- **对比/偏好方法**：DPO 等通过对 chosen/rejected 两类响应施加不同学习策略，在偏好对齐/推理中高效，但少有人扩展到 LLM 蒸馏。
- 此前蒸馏多对 TGO 与 SGO **施加相同 loss**，忽略了"loss 形式 × 数据类型"的协同。

#### 2. 现有工作存在的问题
- **单一 loss 难兼顾 TGO/SGO**：用一种散度同时处理两类数据，性能受限。
- **直接把 DPO 套到 KD（DPKD，把参考模型换成教师）易 reward hacking**：因 p(y_s|x) 本身很小，会过度压低 q(y_s|x)（NLL 飙到 91.25），学生丢失预训练信息而非拟合教师。

#### 3. Motivation
设计**可扩展的对比式蒸馏**：利用 KL 与 RKL 的非对称行为，对不同类型的响应数据施加不同的 loss，从而捕捉"loss 与数据"的协同，并避免 DPKD 式的 reward hacking。

#### 4. 主要灵感 / 核心直觉
- **KL 抬头部、RKL 压尾部**：KL 对 TGO 有 "pulling-up"（在教师概率 p 高的头部抬高学生 q），RKL 对 SGO 有 "pushing-down"（在 p 低的尾部压低 q）。
- **用 SKL/SRKL 而非纯 KL/RKL**：以 DistiLLM 的 skew 版本作 backbone，插值稳梯度。
- **线性而非 log-sigmoid 的对比**：可写成类 DPO 的线性形式，支持 token 级分解与显式加权，并通过混合分布与 p 的线性依赖正则化 q(y_s) 的过度下降，避开 DPKD 的 reward hacking。

#### 5. 主要解决思路（一段话讲清核心）
CALD loss：L = 1/(2|D|) Σ [ (1-β)·D^(α_t)_SKL(教师响应 y_t) + β·D^(α_s)_SRKL(学生响应 y_s) ]，即对教师响应用 SKL（抬高其概率）、对学生响应用 SRKL（压低其概率）。两项增强：(a) α 课程——按一致性闭式更新 α（易样本小 α、难样本大 α）；(b) SRKL 系数 β 线性递增——前期主拟合教师、后期主用 SGO 反馈减小训练-推理不匹配。数据策展上对 SKL 用纯教师生成、对 SRKL 用纯学生生成最优。

#### 6. 方法详解（通俗、分步骤）
1. **对比 loss（CALD）**：对教师响应施 SKL（`tea_pos_kl`），对学生响应施 SRKL（`ref_pos_kl`），分别乘 (1-β)、β 求和。
   - 代码（`src/distillm_trainer.py` ~1141–1210，已核对）：teacher 项 mix = α₁·teacher + (1-α₁)·student → `tea_pos_kl = Σ p·(log p − log mix)`（SKL，抬高）；student 项 mix = (1-α₂)·teacher + α₂·student.detach() → `ref_pos_kl = Σ q·(log q − log mix)`（SRKL，压低，**学生分支 detach**）。
   - Remark 1 证明 CALD 可改写为类 DPO 的线性形式（增大 \tilde q(y_t)、减小 q(y_s)），但用**线性**而非 log-sigmoid，支持 token 级分解与显式加权，并借 \tilde q 与 p 的线性依赖正则化 q(y_s) 的过度下降，避免 DPKD reward hacking。
2. **α 课程**（代码 `update_alpha`）：`anchor=(1-base_α)·(logp−logq)`，`α=clip(1 − anchor/(p̄−q̄), min=1e-2, max=base_α)`，`base_α₁=base_α₂=0.1`——易样本（p̄≈q̄）小 α、难样本大 α。
3. **β 线性递增**（代码 `gradual_beta`）：`β` 随训练步从小到大（如 1.0→1.5），前期主拟合教师、后期主用 SGO 反馈减小训练-推理不匹配。
4. **数据策展**：对 SKL 用纯教师生成、对 SRKL 用纯学生生成最优——speculative decoding/更强 LLM 的"高质量"响应反而更差，说明**教师响应的高 log-prob 比"高质量"更关键**。每个 epoch 前用 vLLM 批量采集 TGO/SGO（batch on-policy）。

#### 7. 实验数据集
- **指令遵循**：UltraChat200k（采 50K prompts），评 AlpacaEval/Evol-Instruct/UltraFeedback（LLM-as-Judge，GPT-4o/4o-mini）。
- **数学**：MetaMathQA（50K）训练，评 GSM8K/MATH。
- **代码**：WizardCoder（Evol-Instruct code）训练，评 HumanEval/MBPP。
- **偏好对齐**：把 SFT 替换为 KD 后接 DPO。
- **VLM**：RLAIF-V-Dataset（83K），评 OK-VQA/TextVQA。

#### 8. 实验结果与主要发现
- **教师→学生对**：Qwen2-7B-Inst→Qwen2-1.5B、Mistral-7B-Inst→Danube2-1.8B、Gemma-2-9B-Inst→Gemma-2-2B（指令）；数学用 Qwen2(.5)-Math-7B-Inst→1.5B；代码用 DS-Coder-6.7B/Qwen2.5-Coder-7B→1.3B/1.5B；VLM 用 LLaVA-1.5-7B→TinyLLaVA-1.4B（Qwen2.5 不同尺度需 `resize_embedding.py` 对齐分类头）。
- **结果**：指令/数学/代码/VLM 全面 SOTA，优于 GKD、DistiLLM、Speculative KD。
- **消融**：对比 loss、β 递增、α 课程三组件逐项增益。
- **capacity gap**：能更好处理教师-学生容量差（教师增大单调提升）。

#### 9. 结果如何支撑其主张
- "loss×数据协同" → 消融显示对教师用 SKL、对学生用 SRKL 的搭配优于单一 loss 与对称搭配。
- "避免 reward hacking" → 相对 DPKD（NLL 飙到 91.25）CALD 保持正常 NLL，由线性形式 + \tilde q-p 线性依赖的正则化解释。
- "高质量≠高 log-prob" → 数据策展实验中"更强 LLM/speculative"响应反而更差，支撑"教师响应高 log-prob 比质量更关键"的论断。

#### 10. 逻辑自洽性（中性评估）
机制叙事与代码高度一致（teacher→SKL/pull-up、student→SRKL/push-down + detach、α clip 课程、gradual β）。Remark 1 的"类 DPO 线性形式"提供了与偏好优化的桥接，但 CALD 本质更接近"对两类数据分别选散度方向"的工程组合，"对比"一词更多指 TGO vs SGO 的差异化处理而非 DPO 式成对 margin。"高质量响应反而更差"是个有趣且反直觉的发现，但其解释（log-prob 主导）属相关性观察，未给因果隔离实验。整体证据扎实、消融完整，结论自洽。

#### 11. 残留问题 / 局限
- **白盒 + 同词表前提**：需教师 logits，Qwen2.5 跨尺度还需 resize 分类头，限制适用范围。
- **超参较多**：α 课程的 base_α、β 的递增 schedule、(1-β)/β 配比均需调，最优区间未给闭式依据。
- **"高质量数据反而更差"缺因果证据**：仅相关性观察，可能与分布匹配度而非 log-prob 本身耦合。
- **每 epoch 重生成 TGO/SGO 的成本**：虽用 vLLM 批量加速，但相对纯离线 KD 仍有额外生成开销。
- **VLM 迁移仅单一对**（LLaVA→TinyLLaVA），多模态结论外推性有限。

#### 12. 开源代码与框架（链接+框架+代码可得性）
- 仓库 https://github.com/jongwooko/distillm-2（已克隆，约 5.9MB）。核心：`src/distillm_trainer.py`（CALD loss，含 tea_pos_kl/ref_pos_kl、α clip 课程、gradual β，已核对）、`src/run_distillm.py`（LLM）、`src/run_distivlm.py`（VLM）、`src/run_sft.py`、`src/alignment/`；生成 `generate/generate_vllm.py` + `reformat.py`；配置在 `training_configs/`、`accelerate_configs/`。
- 框架：基于 HF **alignment-handbook**（setup.py 注明改编）+ Accelerate + DeepSpeed ZeRO-3 + vLLM(0.5.4) 生成 + FlashAttention-2；启动 `accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml src/run_distillm.py ...`。无 veRL/TRL。
- 代码可得性：高（trainer、生成、SFT、VLM 入口与配置齐全，可复现各任务）。


---


### gad — Black-Box On-Policy Distillation of Large Language Models (GAD)

> **一句话重点 (TL;DR)**：当教师是只返回文本的闭源 API(如 GPT-5)时无法做白盒/likelihood 蒸馏；GAD 把学生当生成器、训练一个判别器去区分学生与教师文本，构成 GAN 式极大极小博弈——判别器即"随学生共同演化的 on-policy reward model"，从而在黑盒下实现 on-policy 蒸馏，避免固定 reward model 的 reward hacking。

**元信息**：arXiv:2511.10643（v3, 2026-01-08；首发 2025-11）｜ 微软研究院（Tianzhu Ye、Li Dong 共同一作；Furu Wei 等）｜ 主题 T1（黑盒 on-policy 蒸馏）/ 相关性高 ｜ 代码 microsoft/LMOps `gad/` 子目录（aka.ms/GAD-github 重定向至此；本地已 clone 488KB）｜ 框架 veRL（GRPO，hack critic 当判别器）。

#### 1. 相关工作与进展
- **白盒蒸馏**：能拿到教师 logits/隐状态，用 forward/reverse KLD 对齐分布(KR16、GKD/MiniLLM 等)。
- **on-policy 蒸馏的价值**：白盒研究表明让学生学**自己生成的响应**(reverse KLD)能 mode-seeking、减小 exposure bias，优于纯 teacher-forcing。
- **黑盒蒸馏**：教师只返回文本(代理 API 教师)，标准做法是对教师响应做 **SeqKD**(序列级 SFT/行为克隆)。

#### 2. 现有工作存在的问题
- **黑盒下 on-policy 不可行**：学生生成自身响应后，教师无法给出任何概率级监督来评价/纠正它，标准 likelihood-based on-policy 蒸馏失效；学生与教师 tokenizer 不兼容时 likelihood 目标也失效。
- **SeqKD 的弱点**：易过拟合教师的局部 n-gram 模式、OOD 泛化差(只学到表层词汇而非全局风格)。
- **RLHF 固定 reward model 的弱点**：reward model 预训练后冻结，策略易 **reward hacking**(钻奖励空子，如响应暴长)。

#### 3. Motivation
要在"只能看到教师文本"的黑盒约束下，仍然获得 on-policy 学习的好处(mode-seeking、低 exposure bias、好的 OOD 泛化)，就需要一种**不依赖教师概率**、又能对学生自身生成给出反馈的监督信号。

#### 4. 主要灵感 / 核心直觉
把蒸馏看成 GAN：学生=生成器 G，再训一个判别器 D 去区分"这是学生还是教师的文本"。G 努力骗过 D(让自己的文本被打高分)，D 努力分开二者，形成极大极小博弈。D 本质是一个**随学生策略共同演化的 reward model**——它始终针对学生当前行为给反馈，因此不像固定 reward model 那样被 hack。

#### 5. 主要解决思路(一段话讲清核心)
价值函数 max_G min_D V = E_{(x,y_t)}[ −log σ(D(y_t) − D(G(x))) ](Bradley-Terry 偏好，教师分高于学生)。判别器由生成器参数初始化、加一个标量预测头(取末 token 隐状态投影为序列级分数)，用 BT loss 在线更新；生成器目标 max_G E[D(G(x))]，因采样不可微，把 D(G(x)) 当 reward 用 **GRPO** 做策略梯度优化。二者交替更新、co-evolve。

#### 6. 方法详解(通俗、分步骤)
1. **构数据**：遍历 prompt x，采教师响应 y_t，得训练集 T={(x, y_t)}。
2. **Warmup(关键)**：GAD 正式训练前，对生成器用教师响应做 CE/SFT 一个 epoch、对判别器用同数据做 BT loss 一个 epoch，保证生成器-判别器平衡(消融显示二者 warmup 均关键)。
3. **GAD 训练循环**：每个 batch——(a) 采学生响应 G(x)；(b) 用 D(G(x)) 作 reward、GRPO 更新生成器；(c) 用 BT loss(教师分应高于学生)更新判别器。
4. **RL 框架映射**：Policy=学生、Reward Model=判别器、reward=D(G(x))；与 RLHF 的唯一区别是 reward model **在线共更**而非冻结。

#### 7. 实验数据集
- **训练**：LMSYS-Chat-1M-Clean(从中采 200K prompt，收集 GPT-5-Chat 教师响应)。
- **评测**：LMSYS-Chat 测试集 500 样本(主)；OOD 用 Dolly(500)、SelfInst(252)、Vicuna(80)。打分用 GPT-4o(先生成参考答案再对比打分)+ 人工评测。
- **教师/学生**：教师 GPT-5-Chat(闭源)；学生 Qwen2.5-Instruct(3B/7B/14B)、Llama-3.2-3B / Llama-3.1-8B-Instruct。

#### 8. 实验结果与主要发现
- **GAD 全面超过 SeqKD**(Table 2，所有数据集×模型尺寸)：如 Qwen2.5-14B+GAD 在 LMSYS 得 52.1，**接近 GPT-5-Chat 教师(51.7)**；3B+GAD≈7B+SeqKD(尺寸"升一级")。
- **OOD 上差距更大**：Dolly/SelfInst/Vicuna 上 SeqKD 增益微弱甚至为负，GAD 仍稳健提升——归因于 RL 比 SFT 泛化更好。
- **人工评测**：GAD 对 before-distill 与 SeqKD 的胜率基本 >50%、负率 <30%。
- **机制证据**：(i) N-gram 重叠显示 SeqKD 过拟合教师局部词汇、GAD 学全局风格(Fig.4)；(ii) toy data 上 SeqKD mode-covering、GAD mode-seeking(Fig.5)；(iii) **on-policy 判别器避免了 off-policy 判别器约 300+ 步后的 reward hacking(响应暴长)**(Fig.6)。
- **消融**：BT loss 优于 CE loss；判别器与生成器同尺寸最优(增大判别器无益)。

#### 9. 结果如何支撑其主张
主张是"黑盒下也能 on-policy 蒸馏且优于 SeqKD"。自动+人工双评测 + 五个模型一致超 SeqKD 支撑"更优"；OOD 显著优势 + N-gram/toy/reward-hacking 三项机制分析支撑"为什么更优(RL 泛化、mode-seeking、on-policy 抗 hacking)"——主张与证据对应紧密。"接近教师"仅在 14B + LMSYS 评测下成立，属上限演示而非普遍结论。

#### 10. 逻辑自洽性(中性评估)
逻辑闭环良好：GAN↔RLHF 的映射(判别器=on-policy reward model)既自然又能解释"为何不 reward hacking"。需保留的批判点：(1)所有自动评测靠 GPT-4o 打分，存在评判模型偏好(偏向某种风格)的系统性风险，人工评测样本量(每对 25/题级)也有限；(2)"接近 GPT-5-Chat"的判定建立在 GPT-4o 评分这一代理指标上，并非任务正确率；(3)对抗训练本身的稳定性(GAN 难训)论文靠 warmup 缓解，但未给收敛性保证。

#### 11. 残留问题 / 局限
- **评测依赖 LLM-as-judge**：GPT-4o 打分可能与真实质量/人类偏好有偏差，且教师/裁判都来自相近生态。
- **任务域偏聊天/通用指令**：未验证数学、代码等强推理域的黑盒蒸馏效果。
- **训练成本**：需同时维护生成器+判别器(同尺寸)、3 epoch≈2400 步，比单纯 SeqKD 重。
- **核心算法实现在外部 fork**(YTianZHU/verl)而非本地 clone 的 LMOps/gad(后者仅环境/数据/启动脚本)，算法细节需到该 fork 审计。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 代码仓库为 **microsoft/LMOps 的 `gad/` 子目录**(aka.ms/GAD-github 重定向至此)；本地已 clone `resource/repos/gad`(488KB)。LMOps/gad 仅含环境/数据/启动脚本(`scripts/`、`tools/export_lmsys_parquet.py`、`local_setup.sh`)。
- **算法核心实现在外部 fork** `github.com/YTianZHU/verl`(基于 VeRL，含 seqkd/warmup/gad/eval 四分支)；数据/模型在 HF(ytz20/LMSYS-Chat-GPT-5-Chat-Response、ytz20/gad-models)。项目页 ytianzhu.github.io/Generative-Adversarial-Distillation/。
- **框架**：veRL(GRPO-based)，README 明确"hack critic 模块当判别器"，生成器=policy、判别器=在线 reward model；推荐 docker `czwin32768/verl2:v0.2.0-vllm085`(py3.10/torch2.6/vllm0.8.5)。SeqKD/warmup 的 SFT 也在 fork 内(`dp_actor.py`)。
- **训练**：3 epochs、batch 256、约 2400 步(PPO mini-batch 256)；prompt≤2048、响应≤1536；温度 0.8；每 50 步存 ckpt，选 GPT-4o 分最高且响应长度合理者。


---


### gemma2 — Gemma 2: Improving Open Language Models at a Practical Size

> **一句话重点 (TL;DR)**：Gemma 2 的 2B/9B 用**知识蒸馏(逐 token 软标签的交叉熵)替代 next-token 预测**来预训练小模型；后训练 SFT 阶段明确"在学生自身分布上从教师蒸馏(on-policy KD，引 GKD/MiniLLM)"——这是它与本综述(on-policy 蒸馏)最直接的连接点。

**元信息**：arXiv:2408.00118（技术报告，原始发布 2024-06-27）｜ Gemma Team, Google DeepMind ｜ 主题 T1（蒸馏训练小模型）/ 相关性中 ｜ 代码 仅发布权重(huggingface.co/google/gemma-2-9b 等)，**无训练/蒸馏代码**（CloneTier=B，不 clone）｜ 框架 内部 Google JAX + TPU 栈。

#### 1. 相关工作与进展
- **知识蒸馏**(Hinton et al. 2015)：用大模型(教师)的软标签指导小模型,比硬标签含更多信息。
- **on-policy 蒸馏**(Agarwal et al. 2024 GKD、Gu et al. 2024 MiniLLM)：让学生在**自己生成的分布**上接受教师监督,缓解 train/inference 分布不一致。
- **架构改进**:局部滑窗 + 全局注意力交替、GQA、logit soft-capping、pre+post RMSNorm 等(沿 Gemini 1.5 的蒸馏经验)。

#### 2. 现有工作存在的问题
- 小模型若用常规 next-token 预训练,在固定参数/算力预算下性能受限,难以逼近 2–3× 更大的模型。
- 单纯靠扩数据对小模型边际收益递减(数据效率瓶颈)。

#### 3. Motivation
在"实用规模"(2B/9B/27B,可在消费级/单机部署)下榨出同尺寸最佳性能。思路:用大教师的知识蒸馏训练小模型,相当于以远超 token 数的"虚拟"信息量训练,缓解小模型的数据效率问题。

#### 4. 主要灵感 / 核心直觉
小模型的瓶颈不在算法而在"每个 token 能学到多少信息"。教师对每个位置给出的**完整概率分布**比一个 one-hot 硬标签信息量大得多;用它当软目标,等于把训练信号的密度大幅提高。后训练再让蒸馏发生在学生自己的输出分布上(on-policy),进一步对齐推理时的真实分布。

#### 5. 主要解决思路(一段话讲清核心)
预训练阶段(2B/9B):不再用 next-token 预测,而是逐 token 最小化"教师概率 P_T 与学生概率 P_S 的交叉熵"(软标签蒸馏);27B 因没有更大的同族教师,仍用标准 next-token。后训练阶段:SFT 做行为克隆(响应主要由更大教师合成)并**在学生分布上从教师蒸馏(on-policy KD)**,再叠加 RLHF,最后对各阶段模型做**模型平均**。

#### 6. 方法详解(通俗、分步骤)
- **预训练知识蒸馏(§3.2)**:给定大教师,逐 token 用其概率分布 P_T(x|x_c) 作软目标,最小化 min_{P_S} Σ_x −P_T(x|x_c) log P_S(x|x_c)。2B、9B 用此蒸馏训练;27B 用标准 next-token(无更大同族教师)。
- **后训练(§4)**:
  - **SFT**:在合成+真实 prompt 上行为克隆,响应主要由更大教师合成;并**在学生自身分布上从教师蒸馏(on-policy distillation,引 GKD Agarwal 2024 / MiniLLM Gu 2024)**。
  - **RLHF**:沿用 Gemma 1.1 类似算法,但 reward model 大一个数量级,策略基于与 SFT 相同的 prompt,英文偏好数据训 reward model。
  - **模型平均(model merging/averaging)**:对各阶段后得到的模型做平均提升整体性能(WARP/WARM 思路)。
- **架构要点**:局部滑窗注意力(窗口 4096)与全局注意力(8192)逐层交替、GQA(num_groups=2)、logit soft-capping(注意力层 50.0、最终层 30.0,经 tanh 软截断:logits←soft_cap·tanh(logits/soft_cap))、pre+post RMSNorm。
- **数据**:后训练用 LMSYS-chat-1M 的 **prompt(不用其答案)**;过滤聚焦提升 helpfulness、降低 safety/hallucination 危害、去评测集污染、降 recitation 风险。

#### 7. 实验数据集
- **预训练**:27B 用 13T tokens、9B 用 8T、2B 用 2T(以英文为主的 web/code/science),SentencePiece 256k 词表。
- **评测**:MMLU、GSM8K、ARC、HellaSwag、HumanEval、MBPP、AGIEval、BBH、WinoGrande 等标准基准;指令模型用 Chatbot Arena/人类评测。

#### 8. 实验结果与主要发现
- **蒸馏 vs 从头训(Table 6/7)**:在固定 token 预算下,蒸馏训练的小模型显著优于从头 next-token 训练;且蒸馏的收益随模型变小而更明显——支撑"蒸馏对小模型尤其有效"。
- 2B/9B(蒸馏)相比同 token 数的 Gemma 1 大幅提质;9B/27B 在同尺寸开源模型中达到当时领先水平。
- **滑窗大小可调(Table 10)**:推理时调整滑窗大小对质量影响有限,提供部署灵活性。

#### 9. 结果如何支撑其主张
主张是"蒸馏能在实用规模下让小模型更强"。Table 6/7 的"蒸馏 vs from-scratch"对照在控制 token 预算下直接验证蒸馏增益,且呈现"模型越小蒸馏越有用"的趋势,与 motivation(缓解小模型数据效率瓶颈)一致。但报告为工程技术报告,蒸馏部分的消融相对粗略、缺乏对 on-policy KD 单独贡献的隔离实验。

#### 10. 逻辑自洽性(中性评估)
整体自洽,蒸馏(预训练)+on-policy KD(后训练)+模型平均构成一条连贯的"用大模型造好小模型"流水线。需注意:(1)作为技术报告,许多关键设置(教师规模、蒸馏温度、on-policy KD 的具体配方、模型平均权重)未充分披露,**不可复现**;(2)"蒸馏=以虚拟信息量训练"是直觉性说法而非可量化论证;(3)on-policy KD 虽被点名,但论文未给它相对 off-policy KD 的对照,故其独立贡献无法从本文判断。

#### 11. 残留问题 / 局限
- **完全闭源训练**:无蒸馏/后训练代码,教师模型、数据混合、超参均未公开,本综述只能据论文记录其方法立场。
- **on-policy KD 细节缺失**:仅引用 GKD/MiniLLM 并称"在学生分布上蒸馏",未给损失形式、采样策略、占比等。
- **27B 未蒸馏**:同族无更大教师,故"蒸馏更优"的结论严格说只在 2B/9B 上成立。
- **评测污染/recitation** 虽做过滤,但标准基准上的分数仍受数据混合影响,跨模型对比需谨慎。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- **权重**:huggingface.co/google/gemma-2-9b(及 2b、27b 及对应 -it 指令版)。
- **代码可得性**:**仅发布权重,无训练/蒸馏代码**(n/a)。CloneTier=B,仅记录不 clone。
- **框架**:预训练用逐 token 软标签蒸馏(min CE between P_T 与 P_S);后训练 SFT(行为克隆 + on-policy KD,引 GKD/MiniLLM)+ RLHF + 模型平均;内部 Google JAX + TPU 栈(2B 用 512 TPUv5e、9B 用 4096 TPUv4、27B 用 6144 TPUv5p),GSPMD 分片 + Pathways。


---


### gkd — On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD)

> **一句话重点 (TL;DR)**：把自回归 LM 的知识蒸馏当成"交互式专家模仿学习"——让学生在**自己生成**的序列上、用教师的 token 概率作监督，从而消除训练/推理的分布失配；并把散度选择（forward/reverse KL、JSD）与 on-policy 数据比例统一成一个可调框架 GKD。

**元信息**：arXiv 2306.13649（v3, 2024-01-17）｜ Google DeepMind（+Mila / U. Toronto）｜ ICLR 2024 ｜ 主题 T1/T3，相关性 High（白盒 token 级 OPD 的奠基工作之一，本课题 OPD 直接源头）｜ 代码 TRL `GKDTrainer`（无独立官方训练仓，原实验在 Google JAX 栈）｜ 框架 T5/JAX（论文）、TRL（社区集成）。

#### 1. 相关工作与进展
- **监督 KD**（Hinton 2015；DistilBERT/Sanh 2019）：学生在固定数据集上拟合教师的 token 级概率分布（最小化 forward KL）。
- **序列级 KD（SeqKD，Kim & Rush 2016）**：先用教师生成高概率序列，再对这些序列做监督微调，等价于"在教师输出上做 SFT"。
- **ImitKD（Lin 2020）**：首次点出蒸馏与模仿学习的联系，混采学生与固定数据，但停在 token 级 forward KL，未走纯 on-policy，也未结合 RL。
- **f-distill（Wen 2023）**：把序列级 KD 表述为 f-散度最小化（用 total variation）。
- **并行工作 MiniLLM（Gu 2023）**：把蒸馏当 RL 问题，序列级优化 reverse KL，用 policy gradient，但需多种稳定化技巧。

#### 2. 现有工作存在的问题
1. **训练-推理分布失配（暴露偏差）**：监督 KD / SeqKD 都在固定序列上训练；推理时学生从自身部分输出自回归生成，会进入训练未见状态，早期 token 误差级联放大。
2. **容量失配下 forward KL 的弊端**：学生表达力有限时，最小化 forward KL 迫使学生覆盖教师分布的整个支撑，可能把概率质量分散到教师几乎不产生的 token 上，导致幻觉/低质量生成。

#### 3. Motivation
把"在固定数据上学"换成"在学生自己会走到的状态上学"：借鉴模仿学习中 DAgger（Ross 2011）"教师作为交互式专家"的思路，让学生在 **on-policy（自生成）序列**上接受教师 token 概率的反馈；同时放开散度选择，使有限容量的学生能聚焦教师分布的关键区域。

#### 4. 主要灵感 / 核心直觉
- 自回归蒸馏 ≈ 带交互式专家的模仿学习；on-policy 数据收集能消除暴露偏差，且学生进步后自己生成的数据质量也随之提升（正反馈环）。
- forward KL（均值寻求）会在高温采样下覆盖教师不产生的 token；reverse KL / 偏 1 的 JSD（模式寻求）能避免低质量生成，但牺牲多样性——最优散度**与任务相关**。

#### 5. 主要解决思路(一段话讲清核心)
GKD 用两个旋钮统一所有自回归 KD：**(a) on-policy 数据比例 λ**（每步以概率 λ 用学生自采样序列，否则用固定数据/真值），**(b) 学生-教师 token 分布之间的散度 D**（forward KL / reverse KL / 广义 JSD(β)）。在所选序列上对每个 token 计算 D(教师‖学生) 作为损失（不反传穿过学生采样过程，故稳定且高效）。λ=0、D=forward KL 即退化为监督 KD；λ=1 即纯 on-policy 蒸馏。GKD 还能直接与 RL 微调叠加（式 5），把"向初始策略正则"改为"向教师策略正则"。

#### 6. 方法详解(通俗、分步骤)
1. **起点**：学生与教师都是已 SFT 的同族模型（论文用 T5 系；学生须能生成质量尚可的序列）。
2. **每训练步（Algorithm 1）**：抽 u~Uniform(0,1)；若 u≤λ → 用学生当前策略以温度 γ=1 采样一批 on-policy 序列；否则 → 用固定数据集（真值或教师生成）。
3. **计算损失**：对该批序列，逐 token 计算所选散度 D(教师 token 分布 ‖ 学生 token 分布)，对序列长度归一化；**对学生采样过程做 stop-gradient**（只学概率拟合，不学采样）。
4. **更新学生**，重复。
5. **可选 RL 叠加（§3.2）**：目标 = (1−α)·E[r(y)]（RL 奖励）− α·on-policy GKD 散度。α=1 即纯蒸馏。论文用 RLAIF（文本蕴含奖励）抑制摘要幻觉，同时蒸馏提升下游质量。与 RL 集成时建议用 reverse KL 或 JSD(0.9)。
6. **配方**：WMT 用 JSD(0.1)，其余任务用 forward KL；指令微调任务 reverse KL 最佳。

#### 7. 实验数据集
- **任务特定蒸馏**：XSum（摘要，ROUGE-2，贪心评测）；WMT14 en→de（翻译，BLEU，beam search）；GSM8K（带 4-shot CoT 的算术推理，外部计算器判对）。
- **任务无关蒸馏（指令微调）**：FLAN2021（536 万样本 / 62 任务），held-out 评测于 MMLU（57 任务）、BBH（23 任务）。
- **模型**：教师 = SFT 的 T5-XL（≈3B）；学生 = T5-Small(77M)/Base(250M)/Large(800M)，分别比教师小 38×/12×/3.8×。

#### 8. 实验结果与主要发现
- **总体（Fig.1）**：on-policy GKD 在三任务、各学生规模上一致超过监督 KD 与 SeqKD；相对初始学生的提升约为基线 KD 的 2.1×（摘要）/1.7×（翻译）/1.9×（推理）。
- **散度选择（Fig.4/6/7）**：高温评测下模式寻求散度（JSD(0.5/0.9)、reverse KL）质量更好但多样性更低；贪心评测下散度选择影响很小。指令微调中 reverse KL 显著优于 forward KL（MMLU +2%、BBH +1%）。
- **on-policy 比例（Fig.8）**：当 on-policy 数据占比 ≥25% 后，性能随其增加而提升。
- **数据效率（Fig.3）**：仅用 5% 子集且无真值摘要的 on-policy GKD，胜过用全量真值的监督 KD / ImitKD。
- **RL 叠加（Fig.5）**：GKD+RLAIF 在事实一致性上超过教师，同时大幅提升摘要质量。
- **自蒸馏（Fig.A.11）**：同架构同尺寸 self-distill，学生可超过教师。

#### 9. 结果如何支撑其主张
- "消除分布失配"由 on-policy（λ 越大越好）与数据效率结果直接支撑；"散度任务相关"由质量-多样性权衡曲线与指令微调对照支撑；"可与 RL 无缝结合"由 RLAIF 实验支撑。三类任务一致优于 SeqKD/监督 KD，证据链较完整。

#### 10. 逻辑自洽性(中性评估)
- 框架自洽：监督 KD/SeqKD/ImitKD/f-distill 都被纳为 GKD 的特例，理论与实验都给出。
- stop-gradient 的"不反传穿过采样"使其比 MiniLLM 等 policy-gradient 方案更稳，作者论证清楚。
- 主要消融（散度×λ）系统完整。

#### 11. 残留问题 / 局限
- 实验局限在 **T5 编码器-解码器（≤3B）** 与生成式任务，未覆盖现代 decoder-only 大模型与长 CoT 推理；λ/散度的最优值需逐任务调。
- on-policy 采样有计算开销（GSM8K 约 1.8×–2.2×），尽管作者论证相对服务成本可接受。
- 需要起点学生已具备一定生成质量（非随机初始化），与 RLHF 两阶段范式绑定。
- 教师须可查询 token 概率（白盒），黑盒教师不适用。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 原论文实验在 Google 内部 JAX / T5 栈，**无公开官方训练仓**。
- 社区实现：HuggingFace TRL 的 `GKDTrainer`（https://huggingface.co/docs/trl/gkd_trainer）——已作为内置 trainer。本地**未 clone**（仅文档/集成，CloneTier=B）。


---


### glm45 — GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models

> **一句话重点 (TL;DR)**：开源 MoE（355B 总 / 32B 激活）混合推理模型（thinking + direct 双模式）。后训练分两阶段——先分域训三个专家（Reasoning/Agent/General-chat，各 cold-start SFT + 专家 RL），再用 **self-distillation** 把多专家统一进一个通才；agent RL 中用 **iterative (self-)distillation** 在昂贵 RL 之间快速抬升起点。

**元信息**：arXiv 2508.06471（v1, 2025-08-08）｜ Zhipu AI 智谱 & 清华 ｜ 2025-08 ｜ 主题 T1/T3，相关性 High（多专家→统一的 self-distillation/迭代蒸馏范式；去 KL 的 GRPO + 难度课程）｜ 代码 https://github.com/zai-org/GLM-4.5（模型/权重发布仓，**不含后训练训练脚本**）；RL 框架 **Slime** 单独开源 https://github.com/THUDM/slime ｜ 框架 Slime（Megatron 训练 + SGLang/Router rollout + Data Buffer）。

> 注：这是**模型技术报告**而非单一方法论文；slime 是其 RL 框架（不等于完整后训练脚本，社区微调多用 LLaMA-Factory/ms-swift）。CloneTier=B，仅记录不 clone。

#### 1. 相关工作与进展
- 对标 o1/o3、Claude Sonnet 4、DeepSeek-R1、Kimi K2、Qwen3-235B 等；目标是单个开源模型同时擅长 Agentic / Reasoning / Coding（ARC）。
- 架构借鉴 DeepSeek-V3 / Kimi K2 的 MoE，但**减宽增深**（更少 routed experts、更多层），并用 2.5× 注意力头、QK-Norm、loss-free balance routing、partial RoPE。
- 含一层 **MTP（Multi-Token Prediction）MoE 层**用于推理时 speculative decoding（与本课题 MTP 主题有接触点，但此处仅用于加速解码，非 foresight 训练信号）。

#### 2. 现有工作存在的问题
- RL 中模型能力随训练演化，与**静态训练数据失配**：后期太简单(reward 全 1)、早期太难(reward 全 0)，都缺 reward 方差→无梯度信号。
- 此前主张"多阶段渐增输出长度"的 RL，会让模型在短长度阶段"遗忘"长上下文能力，造成**不可逆**性能下降。
- Agent 任务 RL 耗时；function call 参数含代码时 JSON 转义负担重。

#### 3. Motivation
- 用专家模型迭代（分域训练）+ self-distillation 统一，得到既能深思又能快答的混合推理通才；
- 用难度课程、单阶段长输出 RL、动态采样温度等技巧稳定高效地 scale reasoning RL；
- 用迭代自蒸馏在昂贵的 agent RL 之间快速抬升起点。

#### 4. 主要灵感 / 核心直觉
- "先专后通"：单独训练的领域专家上限更高，再蒸成一个统一模型可兼得各域能力 + 双响应模式。
- RL 数据应随模型能力滚动调难度（课程），始终保持 reward 方差；长输出能力一旦在短阶段退化就难恢复，故宁可直接在目标长度训。

#### 5. 主要解决思路(一段话讲清核心)
**Stage 1 专家训练**：分别构建 Reasoning / Agent / General-chat 专家，每个先 cold-start SFT（少量 extended-CoT）再专家 RL。**Stage 2 统一训练**：Overall SFT 收集各专家数百万样本（数学/代码/科学/chat/agentic/长上下文，ctx 最长 128K），平衡"带完整推理"与"无显式思考"数据 → 通过 **self-distillation** 得到 reflective + immediate 双模式混合推理模型。RL 全程基于 **去 KL 项的 GRPO**。

#### 6. 方法详解(通俗、分步骤)
**Reasoning RL 技巧**：
- **两阶段难度课程**：第二阶段切到极难题（pass@8=0 但 pass@512>0，且仅取有验证正确答案的题），持续突破上限。
- **单阶段 64K 长输出 RL**：直接在 64K 目标长度做 RL，优于渐增长度的多阶段（后者不可逆掉点）。
- **动态采样温度**：reward 收敛时升温增多样性；用 held-out 验证集做质控（取性能跌幅<1% 的最大温度）。
- **Code/Science RL**：code RL 用 **token-weighted mean loss**（优于 sequence-mean，收敛更快、缓解长度偏置）；science RL 仅用少量专家验证的高质量选择题效果最佳（GPQA 65.8% vs 混合数据 62.9%）。

**Agent RL**：
- group-wise policy optimization（仅 model-generated token 入 loss，忽略环境反馈）；
- **process format penalty**：工具格式错则中止 trace 并给零奖励；
- **Iterative Distillation**：RL 训到 plateau 后，用 RL 模型输出替换 cold-start 数据生成更强 SFT 模型，再继续渐增难度 RL，循环抬升；
- 通过增加交互轮数实现 test-time scaling（BrowseComp 随轮数平滑提升）。

**General RL**：Holistic（rule + RLHF + RLAIF 混合反馈）、Instruction-Following、Function-Calling（step-wise rule-based + end-to-end multi-turn）、Pathology RL（针对语言混杂/重复/格式错）。
**function call 模板**：用 XML-like 特殊 token 包裹 key/value，大幅减少代码段转义负担（不损调用执行性能）。

#### 7. 实验数据集
- **预训练 23T tokens**（前 15T 后调整数据权重；MTP loss 权重 0.3→0.1，bias 更新率 0.001→0），多阶段把序列长 4K→32K→128K。
- **后训练**：百万级 SFT 样本（数学/代码/科学/chat/agentic/长上下文）；RL 用规则可验证数学/代码/科学题（难度筛选）、agent 轨迹。
- **评测**：12 ARC 基准——TAU-Bench、BFCL V3、BrowseComp、AIME24、MATH-500、GPQA、HLE、LCB、SWE-bench Verified、Terminal-Bench、MMLU-Pro、SciCode。

#### 8. 实验结果与主要发现
- TAU-Bench 70.1%、AIME24 91.0%、SWE-bench Verified 64.2%、BFCL V3 77.8%、GPQA 79.1%、BrowseComp 26.4%。
- 12 基准总排名第 3、agentic 第 2、coding 第 3，且参数远少于竞品（DeepSeek-R1 的一半、Kimi K2 的 1/3），位于 SWE-bench/参数 Pareto 前沿。
- **关键消融（小实验模型，非 GLM-4.5 本体）**：难度课程 AIME24 81.8%→83.4%；单阶段 64K（83.4%）优于多阶段（80.6%，且早期不可逆掉点）；code RL token-weighted mean 收敛更快。

#### 9. 结果如何支撑其主张
- "专家迭代 + self-distillation 统一"由 ARC 全面强表现与高参数效率（Pareto 前沿）支撑；具体 RL 技巧由小模型对照消融支撑（难度课程、单阶段长输出、loss 计算、science 数据质量）。

#### 10. 逻辑自洽性(中性评估)
- 工程报告型，技巧与动机对应清晰，关键设计都有消融。
- **重要保留**：几乎所有消融与对比曲线**在"smaller experimental model"上做**，非 GLM-4.5 本体，结论能否完全外推到 355B 本体未直接验证（作者已注明）。
- self-distillation 的具体损失形式（logit KL vs SFT 交叉熵）、数据配比等细节披露有限。

#### 11. 残留问题 / 局限
- 后训练**训练代码不开源**（仅放权重 + 评测工具 + 通用 RL 框架 Slime），方法复现门槛高、细节缺失。
- 消融在小模型上，规模外推性存疑；多专家 self-distillation 的统一是否引入能力折损（相对各专家峰值）未量化。
- "去 KL 项的 GRPO" 在 355B 规模的稳定性细节、与 GSPO 等序列级算法的对比未给出。
- 报告自报 benchmark 为主，部分用 LLM 自动判分（HLE 用 GPT-4o、reasoning 用 LLM 验证），存在评测偏差风险。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 模型/权重：https://github.com/zai-org/GLM-4.5 与 https://huggingface.co/zai-org/GLM-4.5（355B + Air 106B）；评测工具 https://github.com/zai-org/glm-simple-evals。
- RL 框架 **Slime**（自研开源）：https://github.com/THUDM/slime——三模块 Training(Megatron) / Rollout(SGLang+Router) / Data Buffer，支持 colocated 同步 与 disaggregated 异步、BF16 训练 + FP8 推理。
- **后训练训练脚本不在主仓**；Slime 是通用 RL 框架，不等于 GLM-4.5 完整后训练流程。CloneTier=B，仅记录不 clone。


---


### gopd — Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation (G-OPD / ExOPD)

> **一句话重点 (TL;DR)**：先在理论上证明 OPD 是"reward 与 KL 永远等权(β=1)、reference 可任选"的 dense KL-约束 RL 特例；再加两个旋钮——**灵活 reference πref** 与 **reward 缩放因子 λ**。λ∈(0,1) 是插值（学生介于 ref 与 teacher 之间），**λ>1 是外推(ExOPD)**，能让学生**越过 teacher 边界**；在多领域专家合并设定下 ExOPD 是唯一能让统一学生稳定超过所有领域 teacher 的方法。

**元信息**：arXiv 2602.12125（v2, 2026-02-26）｜ 中国人民大学高瓴 + 腾讯 LLM 部（Wenkai Yang 腾讯实习完成；通讯 Yankai Lin）｜ 2026-02 预印本 ｜ 主题 T1/T4，相关性 High（为"学生超越 teacher""多 teacher 知识合并"提供可控旋钮，与本课题 OPD+多专家融合高度相关）｜ 代码 https://github.com/RUCBM/G-OPD（已 clone ~33MB，含 verl + math_eval + code_eval）｜ 框架 **veRL v0.6.1**。

#### 1. 相关工作与进展
- **离策略蒸馏（off-policy KD）**：在 teacher 生成轨迹上训学生（logit KL 或 SFT 交叉熵），有效但学生只模仿、不从自身经验学，测试时泛化弱。
- **on-policy 蒸馏（OPD，Agarwal 2024 / MiniLLM Gu 2024）**：学生采样自身轨迹、在每个 token 上对齐 teacher logit（reverse KL），实现 dense on-policy 学习；经验上比离策略蒸馏与 RL 更快更有效（Yang 2025a；Thinking Machines Lab 2025）。
- **应用**：OPD 已被用于(近)无损地把不同领域 RL 变体能力合回原 base（多任务后训练，Xiao 2026），也能大→小蒸馏。
- **相关**：implicit reward（Rafailov 2023, DPO）、权重外推 ExPO（Zheng 2025）。

#### 2. 现有工作存在的问题
- OPD 经验有效，但**机理理解有限**，潜力未充分发掘。
- 标准 OPD 把 reward 项与 KL 正则**强制等权(β=1)**、reference **固定为学生初始策略**，缺少调节学生相对 teacher/reference 行为的旋钮（无法控制"落在中间"或"超越 teacher"）。

#### 3. Motivation
建立 OPD 与 dense RL 的理论联系，并把标准 OPD 推广为带"灵活 reference + reward 缩放因子"的通用框架，从而能精确控制学生落在 reference 与 teacher 之间、甚至超越 teacher。

#### 4. 主要灵感 / 核心直觉
- **关键推导（式 7）**：引入第三方 πref 后，OPD 目标 = max E[ log(π∗/πref) − D_KL(πθ‖πref) ]，恰是 KL-约束 RL（式 2）在 reward r=log(π∗/πref)、β=1 时的特例。
- 该 token 级 reward 与 DPO 的 implicit reward 同形（式 10）；它捕捉从 ref 到 teacher 的对数概率位移，且 π∗ 与 πref 可不同规模。
- 既然 OPD 只是 β=1 的特例，那就把 1/β 暴露成可调 λ：λ>1 等于把 reward 权重"外推"出 teacher。

#### 5. 主要解决思路(一段话讲清核心)
G-OPD 目标（式 11）：max E[ **λ**·log(π∗/πref) − D_KL(πθ‖πref) ]，其中 λ=1/β。最优解满足 logπθ = logπ∗ + (λ−1)(logπ∗ − logπref)（式 12）：λ∈(0,1) 为 **reward interpolation**（学生行为/长度介于 ref 与 teacher）；**λ>1 为 reward extrapolation（ExOPD）**，学生额外拟合 (λ−1)(logπ∗−logπref) 这一外推项，可越过 teacher。reference 的选择在 λ≠1 时影响目标：**强→弱蒸馏**里把 πref 从学生 base 换成 teacher 的 pre-RL base（**reward correction**，式 13），reward log(π∗/π^teacher_base) 才是 teacher RL 诱导的良定义 implicit reward，比 log(π∗/π^student_base) 噪声更小。

#### 6. 方法详解(通俗、分步骤)
1. **造领域 teacher**：对同一 base（Qwen3-4B-Non-Thinking）分别在 math/code 数据上做 GRPO，得 -RL-Math / -RL-Code 专家。
2. **跑 G-OPD**：在原 student 上扫 λ∈{0,0.25,0.5,0.75,1.0,1.25,1.5}（λ=0 即初始态，λ=1 即标准 OPD）；此设定 reference 自然固定为 student base。
3. **梯度（式 14）**：token 级优势 A_t = (logπθ−logπ∗) + (λ−1)(logπref−logπ∗)；用 discount=0 的 next-token 近似。
4. **多 teacher 合并**：把 math/code 两专家用 ExOPD（固定 **λ=1.25**，不再单独调）合回原 base，得统一学生。
5. **强→弱蒸馏**：teacher = Qwen3-30B-A3B-Instruct-2507，student = Qwen3-1.7B/4B；默认 πref=student base；若有 teacher pre-RL base 则用 reward correction 进一步提升。
6. GRPO 与 G-OPD 都启用 token-level rollout correction 缓解训推失配；基于 veRL。〔核对：代码侧 G-OPD 经 `actor_rollout_ref.actor.policy_loss.lambda_vals`（ExOPD=1.25）在 verl(v0.6.1) `dp_actor.py` 实现，多/单 teacher 分支携带 teacher logits；脚本见 `verl/examples/g_opd/`。〕

#### 7. 实验数据集
- **训练**：DeepMath 过滤难度≥6 的 57K（math RL）、Eurus-RL-Code 25K（code RL）；蒸馏数据 = teacher RL 同源数据。
- **math 评测（每题 32 解，avg）**：AIME24、AIME25、HMMT25(February)、HMMT25(November)。
- **code 评测（每题 4 解）**：HumanEval+、MBPP+、LiveCodeBench(v6, 2025-02~05)。Math-Verify 规则验证；温度 1.0、max len 16384。
- **模型**：student=Qwen3-4B/1.7B-Non-Thinking；teacher 为对应 RL 专家或 Qwen3-30B-A3B-Instruct-2507。基线含 SFT（离策略）、标准 OPD、权重外推 ExPO。

#### 8. 实验结果与主要发现
- **单 teacher（Fig.2/3/4, Table 2）**：标准 OPD 几乎完全复刻 teacher 的精度与回复长度；插值 λ∈(0,1) 性能/长度随 λ 单调增、介于 base 与 teacher（可做预算可控推理）；**ExOPD λ=1.25 在所有设定一致超过 OPD 与领域 teacher**；λ=1.5 过度外推会失稳掉点（学生 hack implicit reward）。
- **比"teacher 多训"更强（Table 1）**：teacher 续训 100 步 RL 仅 +0.9（46.0→46.9）；ExOPD 仅 50 步达 48.0（+2.0），证明增益非源于 teacher 训得不够。
- **多 teacher 合并（Table 2）**：SFT 次优、OPD 受 teacher 上限封顶、ExPO 不可控；**唯有 ExOPD 产出在所有基准超过两个领域 teacher 的统一学生**（math 47.7、code 62.0）。
- **强→弱（Table 3）**：4B 学生 ExOPD 45.3 vs OPD 42.6（+2.7）、SFT 35.1；1.7B 学生 25.4 vs 23.1。
- **reward correction（Fig.6）**：用 teacher pre-RL base 作 ref 进一步提升（如 28.1→28.7）。
- **训练动态（Fig.5）**：ExOPD 训练 reward 更高、回复更长、熵更高。

#### 9. 结果如何支撑其主张
- "OPD=dense RL 特例"由式 7 推导直接给出；"λ>1 超越 teacher"由单/多 teacher 与强→弱三组实验一致支撑；Table 1 排除了"teacher 欠训"的混淆；reward correction 由式 13 推导 + Fig.6 实证。理论-实验闭环较完整。

#### 10. 逻辑自洽性(中性评估)
- 理论自洽：式 7→11→12 一以贯之，把 OPD/RL/DPO implicit reward 统一在一个框架。
- 实现与理论对齐（lambda_vals 旋钮、verl 分支）。
- 一处张力：ExOPD 增益伴随**回复变长**（implicit reward 的长度偏置，作者自承），即部分增益可能来自"写更长"而非纯能力提升；λ=1.5 失稳也佐证外推的脆弱性。

#### 11. 残留问题 / 局限
- 规模有限（主在 4B/1.7B；30B teacher 但无其 pre-RL 变体，reward correction 用 4B 代理验证）；作者列为 future work：更大模型、更多样领域 teacher、跨模型族。
- ExOPD 最优 λ（1.25）由小范围扫得，跨任务/规模的普适性未验证；过度外推风险需逐设定调。
- reward correction 需访问 teacher 的 pre-RL 变体并多算一份大 reference 的 log-prob，成本与可得性受限。
- 长度偏置可能虚高部分指标，缺长度归一化对照。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- https://github.com/RUCBM/G-OPD（已 clone ~33MB，含 `verl/` + `math_eval/` + `code_eval/`）。
- 框架 **veRL v0.6.1**；G-OPD/ExOPD 脚本在 `verl/examples/g_opd/`（`run_qwen3-4b-g-opd*.sh`，`lambda_vals` 控 λ）；math 评测用 Math-Verify。GRPO 与 G-OPD 共享同一 veRL 流程，超参见论文 Appendix B。


---


### hpd — Hybrid Policy Distillation for LLMs (HPD)

> **一句话重点 (TL;DR)**：把 SFT、FKLD、RKLD 统一为 token 级 reweighted log-likelihood 目标，用 K1 估计器（Schulman 2020）在每个 token 上判 student 对 expert token 的 under/over-estimate，据此自适应混合 forward/reverse-KL 并重分配概率质量——既保留 one-hot 监督的计算效率，又兼容 off-policy 数据 + 轻量近似 on-policy 采样，从而以更省算力逼近 dense 蒸馏。

**元信息**：arXiv 2604.20244v1（2026-04-22）｜ 上海交大 / 上海创智学院 / 腾讯，通讯 Rui Wang(SJTU)、Ruobing Xie(Tencent)｜ Preprint（README 称 ICML 2026，但论文正文无该字样，〔待核〕）｜ 主题 知识蒸馏统一视角 + 混合 KL/混合 on-off-policy，与 OPD 直接相关（把 OPD 视为数据 regime 之一并用轻量采样降其开销）｜ 代码 https://github.com/zwhong714/Hybrid-Policy-Distillation （已 clone，约 40MB）｜ 框架 LlamaFactory（SFT 路径）+ veRL（RL 路径）。

#### 1. 相关工作与进展
白盒 KD 可用 teacher logits 做分布级匹配（KLD）。FKL 促 mode coverage 但过平滑；RKL 促 mode-seeking 但在师生差距大时不稳。OPD（on-policy distillation）避免 train–inference 失配但开销大。既有 GKD/DistiLLM 系工作分别处理散度方向、优化策略（loss vs reward）、数据 regime（on/off-policy）。

#### 2. 现有工作存在的问题
散度方向、优化策略、数据 regime 三轴被孤立选择、缺乏统一视角；单向散度各有缺陷；OPD 虽避失配但 teacher 侧对 student 输出有分布漂移且算力开销大。

#### 3. Motivation
希望同时拿到双向散度的互补性与 one-hot 监督的计算效率，并天然兼容 off-policy 与轻量 on-policy，从而在算力受限场景务实地逼近 dense 蒸馏效果。

#### 4. 主要灵感 / 核心直觉
把 SFT/FKLD/RKLD 看作同一 token 级 reweighted log-likelihood 目标的不同权重特例：FKL/RKL 用 teacher 全分布给 dense 监督但丢了 one-hot 的效率。用 K1 估计器在每个 token 上廉价判断"student 是否低估了 expert token"，据此决定走 FKL 还是 RKL，并把被抑制 token 的概率质量重分配回 expert token。

#### 5. 主要解决思路(一段话讲清核心)
对每个 offline expert token 算 `k1=qθ(a*|s)·(log p(a*|s)−log qθ(a*|s))`：k1>0（student 低估 expert）触发 forward-KL 式增强、k1≤0（高估）取负值抑制；同时让学生在 offline 前缀下采一个替代 token、对其算 k'1，仅当 k'1<0（高估非 expert token）才以负权重抑制；当 k1>0 且 k'1<0 同时成立则把 expert 权重加倍，将从被抑制 token 释放的概率质量定向回流给 expert token。整套在保留 one-hot 效率的同时混合 FKL/RKL，且无额外超参。

#### 6. 方法详解(通俗、分步骤)
- **expert 权重 w\*（Eq.12/14）**：`k1>0` → `p(a*|s)+k1`（forward-KL）；`k1≤0` → 取 `k1`（负，抑制）。
- **sampled token 权重 wₜ（Eq.13）**：学生在 offline 前缀下采 `aₜ≠a*`，算 `k'1`；仅 `k'1<0` 保留为负权重，`k'1≥0` 置 0。
- **reinforce 加倍（Eq.14）**：`k1>0 且 k'1<0` → expert 权重升为 `2p(a*|s)+k1`，把释放的概率质量重分配回 expert token。
- **效率/兼容性**：保留 one-hot 监督，天然兼容 off-policy 数据 + 轻量近似 on-policy 采样。
- 〔已核-代码〕`LlamaFactory/src/llamafactory/train/hpd.py::compute_hpd_loss` 与 Eq.11-15/Algorithm 1 逐式吻合：`k1_gt_raw=(teacher_nll−student_nll)·exp(student_nll)`、`mask3=mask1&mask2` 时 `adv1+=exp(teacher_nll)` 实现加倍；loss=`−student_nll·adv1 − adv2·sampled_student_nll·(labels≠sampled)`。论文消融明确 "HPD introduces no additional hyperparameters"（已核-PDF）。

#### 7. 实验数据集
- **训练（蒸馏源）**：数学长 CoT 用 **OpenR1-Math-8192**（已核-PDF）；个性化/对话用 Ultrafeedback prompt；代码用 WizardCoder prompt。
- **评测**：数学 AIME24/AIME25/AMC/MATH/OlympiadBench/GPQA(OOD)；对话 AlpacaEval2(LC/WR)、Arena-Hard、MT-Bench；代码 HumanEval/MBPP（EvalPlus pass@1）。
- **模型/规模**：student=Qwen2.5(1.5B/3B, teacher 7B) 与 LLaMA3(1B/3B, teacher 8B)；代码 Qwen2.5-Coder(7B→1.5B)、DeepSeek-Coder(6.7B→1.3B)。teacher **非现成 instruct，而是先在 offline 数据 SFT 再用 GRPO 精调**（PSFT+RL，论文 §7 明确 "select Qwen2.5-7B-Base model as the teacher … SFT all the base models then …"）。

#### 8. 实验结果与主要发现
- 数学（off-policy, avg.）：Qwen2.5-3B 28.25→**39.83**（+41%）、LLaMA3-3B 19.43→**34.56**（+77.9%），均显著超 SFT/SeqKD/RKLD/JSD。
- on-policy：HPD 单独即超 "SFT→OPD" 两阶段；HPD 作 OPD 初始化（HPD+OPD）再获最高分（Qwen2.5-1.5B **30.24→33.41**）。
- 另演示 HPD+DPO、迭代自蒸馏。

#### 9. 结果如何支撑其主张
跨两个模型族、多规模、多任务（数学/对话/代码）的一致增益支撑"统一视角 + K1 混合"的有效性；HPD 单独超两阶段 SFT→OPD、且作 OPD 初始化再涨，支撑"高效逼近/补充 OPD"的卖点；消融逐项移除 K1 组件验证各部件贡献。

#### 10. 逻辑自洽性(中性评估)
代码与公式逐式吻合，消融自洽。但"reweighted log-likelihood 统一"基本是对既有 GKD/DistiLLM 系工作的重述式归纳，真正新颖性在 K1-based 混合与质量重分配规则；"无额外超参"成立，但 teacher 经 SFT+GRPO 加工，增益与 teacher 质量耦合，统一框架的解释力被这一工程依赖部分稀释。

#### 11. 残留问题 / 局限
- "approximate on-policy" 实为在 **offline ground-truth 前缀**下让学生采单个替代 token（非从学生 rollout 采整条轨迹），严格说是 off-policy 框架内的单步偏离纠正，称谓略宽松。
- 数学实验 student 仅 1B–3B、teacher 仅 7B/8B，长链推理增益是否随规模保持未充分验证。
- teacher 本身经 SFT+GRPO，蒸馏增益与 teacher 质量耦合。
- Preprint（2026-04），README 称 ICML 2026 但论文正文无此字样、未见正式接收证据，〔待核〕。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 链接：https://github.com/zwhong714/Hybrid-Policy-Distillation （已 clone，约 40MB）。含 `LlamaFactory/`（SFT 式蒸馏 + HPD loss）、`verl/`（RL 式后训练）、`evaluation/`；提供 Qwen2.5-1.5B（从 Qwen2.5-7B-PSFT-RL teacher 蒸出，README 命名为 Qwen2.5-7B-Thinking）checkpoint。
- 框架：双后端 LlamaFactory（SFT 路径，全参/LoRA）+ veRL（RL 路径）。代码可得、核心 loss 可逐式对照。


---


### lightning_opd — Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation

> **一句话重点 (TL;DR)**：把 on-policy distillation (OPD) 改造为离线版——预先一次性算好并缓存每 token 教师 log-prob 复用，去掉训练期常驻教师 server；关键贡献是识别并证明被忽视的 **teacher consistency**（SFT 与 OPD 必须同一教师）条件，在该条件下离线 OPD 与标准 OPD 共享最优点、梯度差有界且带隐式正则，从而在性能持平/更优的前提下提速 3.6×–4.0×（含让 MoE 教师场景从 OOM 变可行）。

**元信息**：arXiv 2604.13010（v1 2026-04-14，当前 PDF v2 2026-05-08，据 v2 核对）｜ NVIDIA（Yecheng Wu, Song Han, Han Cai；通讯 hcai@nvidia.com）｜ Preprint｜ 主题 OPD，**与本项目 OPD 直接相关、核心命中**｜ 代码 https://github.com/jet-ai-projects/Lightning-OPD （已 clone，约 3.0MB）｜ 框架 slime + slime_plugins（SFT 用 LLaMA-Factory 风格 YAML，教师 log-prob 离线生成用 vLLM）。

#### 1. 相关工作与进展
OPD 是有效的 LLM 后训练范式：在学生自生成 rollout 上让学生对齐教师分布（dense per-token 监督），常比离线 KD 收益更强（Thinking Machines Lab、Qwen3 等）。

#### 2. 现有工作存在的问题
- 标准 OPD 须在整个训练过程常驻大教师 server，GPU 被学生+教师共置而碎片化、成本高、需多节点；MoE 学生（如 30B-A3B）+教师共置常 OOM、不可行。
- 一个自然想法是离线化（训练前一次性预算并缓存教师 log-prob 复用），但**朴素离线化无法可靠匹配标准 OPD**。论文追溯根因为被忽视的 **teacher consistency**：SFT 阶段与 OPD 阶段必须同一教师，违反则引入不可约梯度偏置（对在线/离线 OPD 都有害、离线更甚）。现实常被违反——如 TML 用 QwQ-32B 生成的 OpenThoughts-3 做 SFT，却用 Qwen3-32B 作 OPD 教师。

#### 3. Motivation
若强制 teacher consistency，离线 OPD 在理论上可等价标准 OPD，从而彻底去掉常驻教师 server、把全部 GPU 投入学生训练以大幅提速，并让此前 OOM 的 MoE 教师场景变可行。

#### 4. 主要灵感 / 核心直觉
标准 OPD 的梯度可经重要性采样分解为 `∇J_on=E_{x∼π_ref}[w(x;θ)·f(x;θ)]`（w=π_θ/π_ref）；离线版相当于令 w≡1 的特例。在 teacher consistency 下（SFT 得到的 π_ref 与 OPD 教师同源），丢掉重要性权重所引入的偏差有界，且固定 rollout 分布 π_ref 反而带来一种隐式正则、抑制 policy drift——于是离线化既省又稳。

#### 5. 主要解决思路(一段话讲清核心)
两阶段：Stage-1 SFT 在与 OPD 同一教师生成的轨迹上 MLE 得 π_ref（teacher consistency）；Stage-2 离线 OPD——从 π_ref 采 rollout 并对教师只查询一次、预算缓存每 token 教师 log-prob 形成 D_OPD，训练阶段全程复用该缓存，无需 live teacher。OPD advantage `A_t=log π_T(a_t|s_t)−log π_θ(a_t|s_t)`（stop-gradient，等价对 reverse-KL 做 dense per-token 监督），训练时 clip 到 [−τ,τ]。

#### 6. 方法详解(通俗、分步骤)
设教师 π_T（固定）、学生 π_θ、SFT 参考策略 π_ref。
- **per-token OPD advantage**：`A_t(θ)=log π_T(a_t|s_t)−log π_θ(a_t|s_t)`（教师比学生更自信处为正，反之为负；视作 stop-gradient 标量）。
- **标准 OPD**：`J_on=E_{x∼π_θ}[Σ_t A_t]`（rollout 来自当前学生，需实时教师）。
- **Lightning OPD（离线）**：`J_off=E_{x∼π_ref}[Σ_t A_t]`（rollout 分布固定为 π_ref）。二者共用 advantage、仅响应分布不同；IS 分解显示 `∇J_off` 是 w≡1 特例，teacher consistency 下偏差有界 + 隐式正则。
- **流程**：Stage-1 SFT（D_SFT 须由 OPD 同一教师生成）→ Stage-2 预处理（从 π_ref 采 rollout、对教师只查询一次缓存 log-prob）→ 训练（复用缓存，无 live teacher）。`A_t` clip 到 [−τ,τ]（Alg.1 第13行）。OPD 阶段训 **150 step**（论文称足够收敛）。
- 〔已核-代码〕`slime/backends/megatron_utils/loss.py`（`advantage_estimator=="on_policy_distillation"`）：`advantages=teacher_log_prob−student_log_prob` 逐 token、与 Eq.2 一致、`returns=advantages`。`slime/rollout/on_policy_distillation.py` 用 `is_lightning_opd`/`is_offline_opd` 区分离线缓存 vs 在线查询，离线分支直接读预存 `teacher_log_probs`。SFT=LLaMA-Factory（configs/sft YAML）、OPD=slime（configs/opd 为标准 OPD 含 `deploy_teacher_model`+`--rm-url`；configs/lightning_opd 为离线版，含 4B/8B/30B-A3B 三套）。

#### 7. 实验数据集
- 学生/教师配对：Qwen3-4B-Base–Qwen3-8B、Qwen3-8B-Base–Qwen3-32B、Qwen3-30B-A3B-Base（MoE）；均 SFT 初始化。
- SFT 数据：从 **OpenThoughts3-1.2M** 抽 prompt 采 **300K**（〔已核-代码〕`prepare_sft_prompts.py` 默认 `--num-samples 300000`），由各自教师生成响应。
- OPD prompt：两域——数学 **DAPO-Math-17k**（17K 竞赛题）；代码 **EpiCoder-func-380k 的 30K 子集**（function-level）。每 prompt 从 π_ref 仅采单条 response 并一次性预算教师 log-prob。
- 评测：数学 AIME24/AIME25/HMMT2025（每题 32 解，avg pass@1，max len 32768），代码 LiveCodeBench v5/v6（每题 4 解，max len 40960）。temperature 0.6、top-p 0.95。

#### 8. 实验结果与主要发现
- 与标准 OPD 在所有基准×规模组合上持平或更优，但 **3.6×–4.0× 提速**——4B 从 72→20 GPU·h（3.6×），8B 从 120→30 GPU·h（4.0×）。
- 8B 在 30 GPU·h 达 AIME24 **69.9%**（SOTA 量级）。
- MoE Qwen3-30B-A3B 在单 8×H100 节点达 AIME24 **71.0%** / LiveCodeBench v5 **60.8%**——而标准 OPD 此设置 **OOM 不可行**（论文表中标 ✗）。

#### 9. 结果如何支撑其主张
"离线≈在线"由全基准×规模上持平/更优支撑；"省"由 3.6×–4.0× GPU·h 削减支撑；"让 MoE 可行"由 30B-A3B 单节点跑通（标准 OPD OOM）直接支撑；teacher consistency 的必要性由违反该条件时离线化掉点的对照实验支撑（理论上证明共享最优点 + 梯度差有界 + 隐式正则）。

#### 10. 逻辑自洽性(中性评估)
理论（IS 分解、有界偏差、隐式正则）、代码（loss 与 Eq.2 一致、离线/在线分支明确）、实验（持平+提速+MoE 可行）三者闭环自洽。teacher consistency 的提出既有理论刻画又有反例对照，是本文最扎实的贡献点。

#### 11. 残留问题 / 局限
- teacher consistency 是硬约束：要求 SFT 与 OPD 同教师，限制了复用第三方 SFT 数据（如直接用 OpenThoughts-3）的灵活性。
- 离线缓存固定 π_ref 的 rollout，OPD 阶段学生若漂移较远，w≡1 近似的偏差是否仍可忽略，依赖 150 step 短训练 + clip + 隐式正则共同保证；更长训练/更大师生差距下的稳健性未充分探索。
- 每 prompt 仅采单条 response，rollout 多样性受限。
- 主实验限于 Qwen3 族数学+代码两域，跨族/跨域泛化未展开。
- Preprint（v2 2026-05），未评审。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 链接：https://github.com/jet-ai-projects/Lightning-OPD （已 clone，约 3.0MB；HF 组织 Lightning-OPD）。含 `slime/`、`slime_plugins/`、`configs/`（sft/opd/lightning_opd/models）、`data_curation/`（pipeline.py、prepare_lightning_opd.py）、`scripts/`（precompute_teacher_logprobs_*.sh、collect_rollouts.sh、serve_teacher_*.sh、generate_sft_data.sh）、`train.py`。
- 框架：slime + slime_plugins；SFT 用 LLaMA-Factory 风格 YAML（含 dataset_info.json）；OPD 配置为 Python（configs/opd 标准 OPD、configs/lightning_opd 离线版含 4B/8B/30B-A3B 三套）；教师 log-prob 离线生成用 vLLM（data_curation/pipeline.py）。代码可得、关键 loss 与离线分支可逐处对照。


---


### lightreasoner — LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?

> **一句话重点 (TL;DR)**：用"专家(大)模型 vs 业余(小)模型"的下一-token 分布分歧定位高价值推理时刻，把对比 log-prob 差(contrastive-decoding 信号)蒸成软标签反过来微调专家模型；卖点是极省资源、无需 ground-truth，但精度增益常不及标准 SFT、对已对齐模型几乎无效。

**元信息**：arXiv 2510.07962 ｜ 香港大学 / 芝加哥大学(Jingyuan Wang、Yankai Chen 共一，Chao Huang 通讯) ｜ ACL 2026 / 2025-10 ｜ 主题 对比解码式自蒸馏·无标签推理增强(与 OPD/蒸馏强相关但方向反转，相关性高) ｜ 代码 https://github.com/HKUDS/LightReasoner (已克隆，真实可用，中英双语 README) ｜ 框架 自实现轻量 Python 流水线(非 verl/TRL)

#### 1. 相关工作与进展
LLM 推理增强主流依赖 SFT，常配 rejection sampling：生成多候选 → 用 ground-truth 过滤留正确轨迹 → 对全部 token 统一微调。CoT 类工作早已表明推理能力在预训练中潜伏、可被激发。本文方法谱系属 contrastive decoding(对比解码，Li et al. 2023)的"训练化"——原 CD 在推理时用 expert−amateur log-prob 差重排序，本文把同一对比信号转成微调监督。

#### 2. 现有工作存在的问题
- SFT/拒绝采样**资源密集**：需大规模 curated 数据、生成多候选、依赖 ground-truth 过滤。
- 对轨迹**所有 token 统一优化**，把琐碎步骤与关键推理步骤同等对待——而只有一小部分 token 真正承载学习价值。

#### 3. Motivation
反直觉命题：小模型(SLM)能否"教"大模型？作者认为，在 expert(LLM)与 amateur(SLM)预测**分歧大**的位置，恰是专家推理强项所在；据此可构造监督信号、无需 ground-truth 标签，且只在少量高价值位置构样以省资源。

#### 4. 主要灵感 / 核心直觉
关键直觉(§正文 L1683 起)：expert 与 amateur 的 log-prob 差刻画了"专家偏离业余倾向"的程度——分歧越大，越是专家独特推理能力发挥之处。把这种差当作"教学信号"，等于让专家朝"远离 amateur 式倾向"的方向自我强化。

#### 5. 主要解决思路(一段话讲清核心)
两阶段：先在推理轨迹的每个位置比较 expert/amateur 下一-token 分布，用 KL 散度筛出"信息性步骤"，在选中步上以对比分数(log π_E − log π_A，经 plausibility 掩码 + softmax)构造软标签；再用 LoRA 微调专家、最小化该软标签与专家分布的 KL，从而放大专家推理强项。全程只在高价值位置构样、无需 ground-truth。

#### 6. 方法详解(通俗、分步骤)
1. **Sampling 阶段**：对每个位置比较 expert/amateur 的下一-token 分布，用 **D_KL(π_E‖π_A) > β** 做"信息性步骤筛选"(§2.3.1，代码实现为 full-vocab KL，`kl_div < beta` 时跳过)；在选中步上用 **plausibility 掩码**(只保留 π_E(a) ≥ α·max π_E 的 token)后，以**对比分数 v'_C(a)=log π_E(a)−log π_A(a)**，softmax 归一化为软标签 v_C(§2.3.2)。〔已核-代码 `LightR_sampling.py` 默认 **α=0.2、β=0.4**(L30-31)，amateur 固定为 **Qwen2.5-0.5B**(论文 L684)〕
2. **Fine-tuning 阶段**：对选中步，Expert 最小化 **D_KL(v_C‖π_E)**(等价于以 v_C 为软目标的交叉熵，§2.3.3)，用 LoRA 微调，放大其推理强项。约 1K 问题级样本即可。

#### 7. 实验数据集
- 7 个基准：GSM8K、MATH、Minerva Math、OlympiadBench、SVAMP、ASDiv、MMLU-STEM(MMLU-STEM 为 5-shot，其余 zero-shot)。
- 基模：Qwen2.5-Math-1.5B、Qwen2.5-Math-7B、Qwen2.5-Math-1.5B-Instruct、Qwen2.5-Math-7B-Instruct、DeepSeek-R1-Distill-Qwen-1.5B。amateur 固定 Qwen2.5-0.5B。
- 主指标：zero-shot **pass@1**(Qwen2.5-Math 工具包评测)。〔已核-§3.1〕

#### 8. 实验结果与主要发现
- 作者自报(Abstract)：7 基准精度最高 **+28.1%**；时间 **−90%**、采样问题 **−80%**、tuned token **−99%**，全程无 ground-truth。
- 效率对比(Table 2，已核)：Qwen2.5-Math-1.5B LightR +7.7%(0.5h/1K 题/0.02M token) vs SFT +11.8%(4h/4K 题)；7B +4.5%(0.75h) vs SFT +4.7%(9.5h);DeepSeek-R1-Distill-1.5B LightR +5.6% vs SFT +3.0%(此例 LightR 反超 SFT);Qwen2.5-Math-1.5B-Instruct **LightR +0.1% = SFT +0.1%**(均近 0)。
- Fig.6：Expert-Amateur 专业差距越窄，增益越小。

#### 9. 结果如何支撑其主张
"极省资源换接近 SFT 增益"这一**效率主张**被 Table 2 充分支撑(时间/token 数量级下降属实)。但"教会推理 / 最高 +28.1%"的**精度主张**支撑较弱：+28.1% 是跨模型×数据集的单点峰值，多数主力配置(1.5B/7B)增益不及 SFT;唯 DeepSeek-Distill 一例反超。故主张应理解为"效率向"而非"精度超越向"。

#### 10. 逻辑自洽性(中性评估)
方法内部自洽：把 contrastive decoding 的推理时信号搬到训练时，KL 筛选 + plausibility 掩码 + softmax 软标签链条清晰，代码与公式(2/4/5/6)一致。"无 ground-truth 仍能改进"的可行性由 amateur 提供对照信号支撑，逻辑成立。

#### 11. 残留问题 / 局限
- (1)精度增益常不及 SFT(1.5B +7.7% vs +11.8%)，卖点是效率而非精度;标题与"+28.1%"有挑选最优 case 之嫌，需看全表。
- (2)对已对齐/Instruct 模型增益趋近 0(1.5B-Instruct +0.1%)——只修复"未充分激发"的基模，对饱和模型边际收益极小;且增益随 Expert-Amateur 差距收窄而衰减(Fig.6)。
- (3)本质是 contrastive-decoding 信号训练化，creativity 在"反向用弱模型作对照"而非全新机制。
- (4)amateur 选取(多弱、是否同族)对分歧信号质量影响大，论文固定 Qwen2.5-0.5B，鲁棒性边界未充分给出。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 链接：https://github.com/HKUDS/LightReasoner (约 67MB，已 clone;含 `LightR_sampling.py`/`LightR_finetuning.py`/`data_prep.py`/`merge.py`/`evaluation/`/`LRsamples`;README 中英双语，arXiv badge 2510.07962 一致)。HF 有模型 collection。
- 框架：自实现轻量 Python 流水线(requirements Python 3.10+)，非 verl/TRL 重框架;微调用 LoRA;基线评测走 Qwen2.5-Math 工具包。
- 可得性：完整、真实、可复现。


---


### madopd — MAD-OPD: Breaking the Ceiling in On-Policy Distillation via Multi-Agent Debate

> **一句话重点 (TL;DR)**：把多智能体辩论(MAD)搬进 OPD 训练环，让 K 个 teacher 就 student 的 on-policy 状态多轮辩论、辩论 transcript 作为 privileged context 产生 token 级监督(按辩论后置信加权)，以突破单 teacher 天花板；并以 OPAD(step-level 采样)把 OPD 扩到 agentic 任务，配 task-adaptive 散度原则(agentic 用有界 JSD、代码用 reverse KL)。

**元信息**：arXiv 2605.01347 (v1, 2026-05-02) ｜ 华中科技大(Jianze Wang，阿里实习) + 阿里巴巴；Yong Xie(HUST)、Qianglong Chen(Alibaba)通讯 ｜ Preprint 2026-05 ｜ 主题 OPD/distillation(直接相关) ｜ 代码 https://github.com/chiefovoavicii/MAD-OPD（本地已 clone ~1.9M，含核心算法）｜ 框架 TRL(0.15–0.25) + vLLM + DeepSpeed ZeRO-3 + transformers 4.51.1 + liger-kernel

#### 1. 相关工作与进展
OPD(on-policy distillation)已成主流后训练配方：student 在自身轨迹上受 token 级 teacher 监督，提供 dense on-policy 信号，是 outcome-reward RL 与 off-policy 序列级蒸馏的高效替代；已用于 Qwen3 strong-to-weak、DeepSeek-V4 多 teacher OPD。近期工作扩展 OPD 以利用 privileged info、reward extrapolation、verifiable reward、context internalization；并有诊断工作研究 OPD 何时失败、reverse-KL 目标的稳定化重构。多智能体侧：multi-agent debate(MAD)经迭代论辩产生超越单体的涌现集体智能，confidence-weighted consensus 可匹配/超过最强单体。多 teacher 蒸馏汇聚互补信号但 teacher 间无交互(独立产出，student 学固定聚合)。

#### 2. 现有工作存在的问题(三个 L)
- **L1 单 teacher 能力天花板**：现有 OPD 从单 teacher 蒸馏，student 被该 teacher 上界限制；诊断工作显示即便更强 teacher 也未必能提 student。
- **L2 任务覆盖窄**：OPD 研究集中在数学/常识推理，agentic(多步工具调用 + 环境反馈)几乎未探索；逐步错误跨长轨迹累积、destabilize 训练。
- **L3 散度选择 ad-hoc**：OPD 数学上等价于 dense token 级 RL，散度 D 直接塑造奖励信号；但现有稳定化各 patch 一个失效模式，缺把 D 与任务结构挂钩的原则，散度被当任务无关超参。

#### 3. Motivation
突破单 teacher 上限需多模型协作；但现有多 teacher 蒸馏是 off-policy(student 被动消费预算好的信号)，缺对自身 rollout 的实时反应。MAD 的涌现集体智能此前只在推理期消费——把它搬进 on-policy 训练环作 token 级监督源，可同时破 L1，并以专门方法填补 L2、以理论原则解 L3。

#### 4. 主要灵感 / 核心直觉
OPD = teacher-forcing 下 dense token 级 RL，per-token 奖励 r_D(s_t)=−D(p‖q)，选散度即选奖励。privileged p–q gap(teacher 见辩论 transcript c、student 不见)在 student 访问态造成结构性不对称：(a) teacher 对 student 采样 token 赋近零概率(p→0 而 q>0，主导 agentic)；(b) teacher 集中在多个有效 token(主导代码)。这恰对应两种散度需求：agentic 需 logit 梯度有界(JSD，Lemma 1.2 worst-case ∥∇_z JSD_0.5∥_∞≤2，与轨迹长 M 和师生 gap 无关；reverse KL 含 q log(q/p) 项在 p→0 时无界、forward KL 有界但 mode-covering 损 agentic)；代码需 mode concentration(reverse KL，Lemma 2 收敛到主导 mode，避免拼接不兼容实现；forward KL/JSD 会拼接)。

#### 5. 主要解决思路(一段话讲清核心)
MAD-OPD 在 OPD 训练环每个决策点让 K teacher 就 student on-policy 状态做 R 轮辩论(round 1 独立、之后读全部历史修订)，辩论历史 H_R^m 作 privileged context c；各 teacher 辩论后自报置信 c_k∈[0,100]，softmax(温度 τ_conf=1.0)归一成权重 w_k；teacher 带 c force-decode student 的 on-policy 样本、student 不带 c，按 w_k 加权的散度 D(p_Tk‖p_S) 求 token 级 loss(式 8)，梯度只流 student。按 Remark 1：agentic 用 JSD_β、代码用 reverse KL。OPAD(式 9/10)给 agentic 加 step-level 采样：student 逐步 rollout、环境返回观察、teacher 每步就实际观察辩论 force-decode，监督随 student 实际轨迹自适应。

#### 6. 方法详解(通俗、分步骤)
1. **辩论生成 privileged info(§4.1)**：K teacher 对状态 s_m 辩论 R 轮(式 5)，得 H_R^m 作 c(只对 teacher 可见)。
2. **置信加权(§4.2)**：辩论后各 teacher 自报 c_k，式 7 softmax 归一为 w_k；反映 deliberation 后确定性(辩论中立场被削弱者贡献小)。
3. **token 级目标(§4.3，式 8)**：teacher 带 H_R^m force-decode、student 不带；Σ_k w_k·D(p_Tk‖p_S)；**散度跨全词表(full vocabulary)**，teacher logits 作固定目标。
4. **OPAD(§4.4)**：agentic 走 step-level——s_m=(x,τ<m)，student 采 a_m，env 返回 o_m；每步辩论 force-decode a_m，per-step loss(式 9)求和成轨迹 loss(式 10)；条件于实际观察使监督自适应。
5. **散度选择(Remark 1)**：agentic→JSD_β(β=0.5)，代码→reverse KL。

#### 7. 实验数据集
- **训练**：agentic 用 ToolACE 按 OPAD 协议拆 step-level(∼16K 实例)；code 用 OpenThoughts3 采 30K 题；agentic/code 训练独立 checkpoint、不共享数据。
- **评测**(五 benchmark，体现 agentic/code 划分)：agentic BFCL-v4、τ²-Bench、VitaBench；code LiveCodeBench v6、MBPP+。排除数学(已被近期 OPD 工作覆盖)。
- **配置**：六组师生(Qwen3 与 Qwen3.5；student 1.7B–14B，teacher 8B–32B，师生比 4.7×–15.5×)；两 Qwen3 对(14B+8B、32B+30B-A3B，30B-A3B 为 Qwen3-30B-A3B-Instruct-2507)+ 一 Qwen3.5 对(27B+9B)。基线：Base、单 teacher OPD(取较强 teacher)、MT-OPD(等权多 teacher 无辩论)、MT-SeqKD(off-policy 7:3 强弱混)。全程 non-thinking 模式；K=2, R=2, β=0.5。

#### 8. 实验结果与主要发现
- **(RQ1) 六配置全部 overall Avg 第一**(表 1)。如 14B+8B→4B：MAD-OPD Avg 34.66 vs OPD 31.72 / MT-OPD 31.27 / MT-SeqKD 29.19 / Base 27.79；Ag-Avg 25.69(OPD 23.26)、Co-Avg 48.12(OPD 44.41)，即 agentic +2.4%、code +3.7% 超更强单 teacher OPD。
- **4B 超 14B teacher**(App. C.3)：14B+8B→4B 在 LCB-v6 超其 14B teacher +4.26% pass@1、+10.29% BoN@16，token 成本可比 → 瓶颈是 teacher-pool 多样性而非单 teacher 能力。
- **基线各有结构性失效**：单 teacher OPD 被 teacher 上限封顶；MT-OPD 的 per-token 梯度冲突在 code 上低于单 teacher OPD(两 teacher 分布逐 token 平均会插值不兼容代码路径产生不连贯监督)；MT-SeqKD 继承 off-policy exposure bias。
- **(RQ2) scaling**(图 3)：跨族跨配置泛化，增益随师生能力 scale；Δ(Base) Avg +3.5%~+8.0%。
- **(RQ3) 组件消融**(图 4a)：debate 比 MT-OPD +4.6% Co-Avg；confidence weighting +1.7% Ag-Avg；R=2 为经验最优(R=3 因 prompt-context 膨胀过头)。
- **(RQ4) 散度选择**(图 4b/表 4)：JSD 在 agentic 领先、reverse KL 在 code 领先、forward KL 两者都落后，与 Prop 1.3/2.2 一致；训练动态(图 6)佐证。

#### 9. 结果如何支撑其主张
表 1 六配置一致第一直接支撑"破单 teacher 天花板";4B 超 14B teacher 支撑"瓶颈是 teacher-pool 多样性"这一更强主张;组件消融(去 debate/置信/multi-T/on-policy)逐项隔离贡献,支撑各设计非冗余;散度消融(固定其余只换 D)沿 agentic/code 划分翻转,与理论(Prop 1/2、Lemma 1/2)预测一致,理论-经验闭环较完整;BoN@16(App. C.4)补足 pass@1 之外的采样表现。

#### 10. 逻辑自洽性(中性评估)
理论(散度有界性/mode 几何)→原则(Remark 1)→方法(辩论+置信+OPAD)→受控消融,链条自洽且有理论支撑(罕见地给出 logit 梯度有界性证明)。需注意:(1) 增益相对成本偏温和——14B+8B→4B 仅 agentic +2.4%/code +3.7%,而训练需在线跑 K×R teacher 辩论 + force-decode,论文也承认"训练成本随 K×R teacher 前向 scale",但未给单 teacher OPD 的端到端开销对比(只给绝对 wall-clock:4B agentic ≈16h、code ≈32h);(2) "4B 超 14B teacher"仅在 LCB-v6 这一基准且为 BoN@16 放大,普适性需谨慎;(3) 置信由 teacher 自报,其可靠性/可被操纵性靠 App. C.1 的鲁棒性分析支撑,属间接;(4) task-adaptive 原则的理论假设(代码=disjoint-support 多 mode、agentic=p→0 gap)是理想化建模。

#### 11. 残留问题 / 局限(论文 §6 Limitations)
- **需 teacher token 级分布**:排除 black-box(仅 API)模型。
- **假设师生共享词表**:cross-vocabulary 蒸馏是开放扩展。
- **训练成本随 K×R teacher 前向 scale**:在线辩论 + force-decode 工程/计算开销显著,论文未充分量化相对单 teacher OPD 的额外成本。
- **增益温和**:相对该成本,+2.4/+3.7% 偏小;"4B 超 14B"是单基准 BoN 个案。
- **范围**:仅 agentic + code(排除数学),长程 agentic(误差累积更重)留待 future work;全程 non-thinking 模式。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库 https://github.com/chiefovoavicii/MAD-OPD （本地已 clone ~1.9M）。结构:`mad_opd/trainers/`(`mad_opd_core.py` 核心散度原语、`mad_opd_trainer.py`、`vllm_teacher_manager.py`)、`scripts/`(四算法 `run_opd.sh`/`run_mt_opd.sh`/`run_mad_opd.sh`/`run_mt_seqkd.sh` + `launch_teachers.py` + `force_decode_server.py`)、`eval/`(BFCL/LCB/τ²/Vita/MBPP+)、`data/`。
- **框架**:TRL(`trl>=0.15,<0.25`) + vLLM(`>=0.6`，多 teacher debate 文本服务) + DeepSpeed ZeRO-3 + transformers 4.51.1 + liger-kernel(LigerFusedLinearJSDLoss 可选)。
- **基础设施**(§B.2):8×NVIDIA H20 + ZeRO-3;每 teacher 经两个单卡进程服务——(i) vLLM 出辩论文本,(ii) sidecar 出 token 级 logit 分布。超参(§B.1):K=2,R=2,β=0.5,AdamW(β1=0.9,β2=0.999,wd 0.01),lr 1e-5 cosine + 5% warmup,有效 batch 128,grad clip 1.0,agentic max len 4096 / code 16384,训练 1 epoch,每 10 步存 ckpt。

〔本轮核码——修正前一轮的"差异"判定〕
- **前一轮记的〔核码差异〕(论文称全词表而仓库 JSD 是 top-k 截断)经再核为不成立/被高估**:`mad_opd/trainers/mad_opd_core.py` 的 `generalized_jsd_loss_single` 的 `top_k: Optional[int] = None`,仅当 `top_k is not None and top_k>0 and top_k<vocab` 时才截断(L201);`mad_opd_trainer.py` 中 `self.jsd_top_k = getattr(args,'jsd_top_k',None)` 默认 None,且 `sidecar_top_k=self.jsd_top_k` 也随之为 None → **默认走全词表 chunked JSD,与论文 §B.2"full vocabulary distribution rather than top-k truncation"一致**。`_chunked_jsd` 的 "chunk" 只是对全词表分块以省显存,非 top-k 截断;top-k 是可选 memory 旋钮(默认关)。核心模块注释亦写 "JSD is the default",reverse_kl 经 `divergence='reverse_kl'` 切换。
- 结论:论文与默认实现一致,**无实质描述-实现出入**;前一轮的差异标注应撤销。其余事实(K=2/R=2/β=0.5、8×H20 ZeRO-3、TRL 框架、四算法脚本、16h/32h、师生比、表 1 数值)经再核一致。


---


### minillm — MiniLLM: On-Policy Knowledge Distillation of Large Language Models

> **一句话重点 (TL;DR)**：把标准 KD 的 forward KLD 换成 reverse KLD(让小学生 mode-seeking、只学教师主要模式而非长尾)，并用策略梯度做 on-policy 优化；配三项稳定化(单步分解降方差、教师混合采样防 reward hacking、长度归一去短句偏好)，在 120M–13B 跨族一致优于 SFT/word-KD/SeqKD。

**元信息**：arXiv 2306.08543 (v6, 2026-01-31) ｜ 清华 CoAI(Yuxian Gu，微软实习) + 微软研究院(Li Dong、Furu Wei)，Minlie Huang 通讯 ｜ ICLR 2024(2023-06 首发，持续更新至 2026-01) ｜ 主题 T1/High ｜ 代码 microsoft/LMOps 子目录 minillm（本地已 clone 整仓 348MB，Tier 保留）｜ 框架 自研(改版 HF Transformers + DeepSpeed + Accelerate，类 RLHF pipeline)

#### 1. 相关工作与进展
KD 分两类：black-box(仅教师生成文本可见，近期在 LLM API 蒸馏小模型上很流行)与 white-box(教师输出分布/隐状态可见)。white-box KD 此前主要用于 <1B 的语言理解/分类模型(模仿 logits/隐状态/注意力)；文本生成的标准 KD 是近似最小化 forward KLD(word-level 用教师每步输出作监督，sequence-level/SeqKD 直接训练在教师生成文本上)。分布散度度量(forward KLD、TVD、最优传输、reverse KLD)对文本生成训练影响显著，已有并行工作探索。

#### 2. 现有工作存在的问题
随着开源 LLM 兴起，white-box KD 价值上升，但面向生成式 LLM 的 white-box KD 尚未充分探索。标准 KD 本质最小化 forward KLD(KL[p‖q_θ])，迫使学生覆盖教师所有模式。对开放式文本生成，教师分布 p 的模式数远超低容量学生 q_θ 所能表达；最小化 forward KLD 会让学生把概率质量放到 p 的零概率/空白区(zero-forcing 的反面，overestimate void regions)，自由生成时产出低质量、不可能的样本。

#### 3. Motivation
改用 reverse KLD(KL[q_θ‖p])做蒸馏目标，让学生 mode-seeking——聚焦教师主要模式、对空白区赋低概率，避免学习长尾变体，更适合需要真实性/可靠性的生成场景；并推导其策略梯度形式做 on-policy 优化，使学生在自身能力内生成"教师偏好"的样本，而非死记教师的全部采样。

#### 4. 主要灵感 / 核心直觉
计算机视觉与 RL 中 reverse KLD 的 mode-seeking 性质(toy 高斯混合实验，图 2)；可用策略梯度定理把 min KL[q_θ‖p] 转成 on-policy 优化(R_t = 累计 log p/q 作每步生成质量奖励)。另有逆强化学习(IRL)视角的等价理解(附录 A.1)。On-policy 采样天然缓解 teacher-forcing 带来的 exposure bias。

#### 5. 主要解决思路(一段话讲清核心)
目标 min_θ KL[q_θ‖p]，用 Policy Gradient Theorem 推梯度(式 2，∇L = −E[Σ(R_t−1)∇log q_θ])；因策略梯度高方差、reward hacking、偏好短句，引入三项稳定化：单步分解(把单步 r_t 从 R_t 分离、直接对词表求和算 E[r_t] 降方差)、教师混合采样(ep=α·p+(1−α)·q_θ，α=0.2，抑制退化句、缓解 reward hacking，并用重要性采样修正、近似为单步 importance weight 降方差)、长度归一(对 R_{t+1} 归一消除短句偏好)。最终梯度含 PPO 式 clipping，并叠加 D_PT 上的语言建模 loss 保基准能力；整体流程类 RLHF：先 SFT 初始化，再 on-policy 训练。

#### 6. 方法详解(通俗、分步骤)
1. **初始化**：在任务数据 D(Dolly)上 SFT 学生，取最低验证 loss 的 checkpoint。
2. **采样**：从混合分布 ep=α·p+(1−α)·q_θ 采 response(α=0.2)，抑制退化。
3. **算梯度**：(∇L)_Single 直接对整个词表求和算单步质量(降方差)；(∇L)_Long^Norm 用长度归一的 R_{t+1} 加 importance weight w_t≈q_θ/ep(近似单步以避免连乘方差累积)与 clip(式 7、算法 1)。
4. **保基准能力**：加 ∇L_PT = −∇log q_θ(d)(d 采自预训练语料 D_PT)。
5. **更新**：θ ← θ − η·[(∇L)_Single + (∇L)_Long^Norm + ∇L_PT]，按验证集 Rouge-L 选 checkpoint。

#### 7. 实验数据集
- **训练**：databricks-dolly-15K(过滤超长后约 12.5K 训练 / 1K 验证 / 0.5K 测试)；D_PT：GPT-2 族用 OpenWebText，其余用 RoBERTa 语料。
- **评测**(5 个指令遵循集)：DollyEval(500)、SelfInst(252)、VicunaEval(80)、S-NI(SuperNaturalInstructions，用 [11,+∞] 长子集)、UnNI(从 UnnaturalInstructions 采 10K)。指标：Rouge-L、GPT-4 feedback(仅 Dolly/SelfInst/Vicuna)、人工评测(SelfInst)；另测校准(ECE，SST2/BoolQ)、exposure bias(ExAccErr)、多样性(distinct-4gram + 测试集 LM loss)。temp=1，每 prompt 取 5 次生成均值。
- **模型**：GPT-2(120M/340M/760M，教师 GPT-2-1.5B)、OPT(1.3B/2.7B/6.7B，教师 OPT-13B)、LLaMA-7B(教师 13B)；附录另用 GPT-J 6B 作教师。

#### 8. 实验结果与主要发现
- **全范围一致超基线**：120M–13B、三族、5 集、Rouge-L 与 GPT-4 两指标下 MiniLLM 几乎全胜 SFT/word-KD/SeqKD；非 Dolly 集上优势更大(OOD 泛化好)。如 GPT-2-1.5B→120M 在 DollyEval GPT4：MiniLLM 44.7 vs SeqKD 41.2 / KD 40.3 / SFT 38.6。
- **学生有时超教师 Rouge-L**(Vicuna/S-NI/UnNI)：归因于教师 teacher-forcing 的 exposure bias，而 MiniLLM 的 on-policy 采样缓解之。
- **人工评测**(LLaMA-7B←13B)：MiniLLM 人偏好优于所有基线，逼近教师。
- **附加性质**：更低 exposure bias(ExAccErr)、更好校准、长文本表现更优；多样性(distinct-4gram、LM loss)几乎无损。
- **消融**(GPT-2-1.5B→125M)：教师混合采样 + 长度归一对稳定训练关键(去掉则模型快速 reward hacking 生成重复/短/无意义串)；单步分解显著降方差、提升验证/测试分。

#### 9. 结果如何支撑其主张
跨 3 族 × 多尺度 × 5 集 × Rouge-L/GPT-4/人工三指标的一致增益支撑"reverse-KLD on-policy KD 更优、可扩展"的核心主张；exposure bias(ExAccErr)、校准(ECE)、长文本与多样性的专门分析分别佐证"mode-seeking 提升正确性/可靠性而不显著损多样性"；三项稳定化的消融直接支撑各自必要性。toy 高斯实验(图 2)与梯度/IRL 推导给出 reverse-KLD 选择的理论动机。

#### 10. 逻辑自洽性(中性评估)
方法链(reverse-KLD 动机 → 策略梯度 → 三项降方差/防 hacking/去偏 → 类 RLHF 训练)自洽，消融能逐项验证稳定化项的作用。需注意：(1) 三项稳定化均为近似(单步 importance weight 近似、单步分解、长度归一)，是为可训练性做的工程权衡，可能引入偏差，论文以经验稳定性而非无偏性论证；(2) "学生超教师 Rouge-L"基于 Rouge-L 这一表面重叠度量 + 教师也经 teacher-forcing 微调，结论需谨慎(并非学生能力真超教师)；(3) reverse-KLD 理论上 mode-dropping，作者用 distinct-4gram/LM loss 说明多样性"几乎无损"，但承认对需多样输出的场景这是权衡。

#### 11. 残留问题 / 局限
- **无专门 Limitations 章节**：局限需从正文推断。
- **mode-seeking 的多样性代价**：reverse-KLD 本性丢模式；论文以"多数 NLP 应用一个正确响应即够"为由淡化，但对需高覆盖/多样生成的任务并不适用。
- **依赖 white-box 教师**：需教师完整输出分布(全词表 logits)，无法用于仅 API 可见的 black-box 教师；且需教师与学生 tokenizer/词表兼容。
- **稳定化的脆弱性**：去掉教师混合/长度归一即 reward hacking，说明方法对超参(尤其 α=0.2)与稳定化技巧依赖较强。
- **任务/规模范围**：仅指令遵循 + ≤13B，未验证更大模型、推理(数学/代码)或 agentic 任务；评测以 Rouge-L 为主指标，受其表面度量局限。
- **算力开销**：on-policy 需训练中持续采样 + 教师前向算分布，比离线 SeqKD 重。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库 https://github.com/microsoft/LMOps （子目录 `minillm/`）；本地已 clone 整个 LMOps(348MB，<500MB 保留)，MiniLLM 代码在 `resource/repos/minillm/minillm/`，核心算法在 `minillm/minillm/`：`losses.py`(含单步正则 `single_step_reg`、`length_norm`、reverse-KLD 奖励/优势计算)、`trainer.py`、`sampler.py`、`pipelines.py`、`reward.py`、`storages.py`。已被 HuggingFace TRL 收录(`trl/experimental/minillm`)。
- **框架 = 自研(custom)**：基于改版 HF Transformers(`t1101675/transformers@minillm` 分支，加 model/tensor parallel 与 teacher-mixed sampling)+ DeepSpeed + Accelerate；`install.sh` 装定制 transformers/deepspeed/accelerate/peft；`train_minillm.py` + `scripts/` 经 deepspeed 启动。无 veRL/TRL/OpenRLHF 等外部 RL 框架，训练 pipeline 自研(类 RLHF)。
- 训练资源：16×32G V100(小模型可减)，大模型用 tensor parallel(size=4)。基线 SFT / word-level KD / SeqKD。α=0.2 全程固定，按验证 Rouge-L 搜超参。

〔核实结论〕原分析与论文/仓库一致(单步分解/教师混合/长度归一三项、α=0.2、跨族 120M–13B 均核对无误)；本轮补全：论文无独立 Limitations 章(局限改为从 reverse-KLD mode-dropping 与方法假设推断)、消融具体现象(去稳定化即 reward hacking)、white-box 教师依赖等中性局限。无新事实出入。


---


### nemotron_cascade2 — Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation

> **一句话重点 (TL;DR)**：在按领域顺序的 Cascade RL 中插入一个"多领域 on-policy 蒸馏(MOPD)"稳定化阶段——用各领域最强中间 checkpoint 作教师、以 token 级 reverse-KL 蒸馏优势恢复 RL 造成的回退；30B/3B-激活 MoE 由此在数学/代码达到接近前沿、IMO/IOI 金牌级，但以牺牲通用知识/STEM(MMLU-Pro、GPQA) 为代价。

**元信息**：arXiv 2603.19220 ｜ NVIDIA（Zhuolin Yang*, Zihan Liu*, Yang Chen*, Wenliang Dai*, Boxin Wang*, …, 通讯 Wei Ping*†）｜ 2026-03-16 ｜ 主题 T?/High（多领域/多教师 OPD + RL 交织的工业级范例，与本课题直接相关）｜ 代码 无独立训练代码仓（NVIDIA 模型/数据发布；依托开源 Nemo-RL）｜ 框架 Nemo-RL

#### 1. 相关工作与进展
将前沿推理与 agentic 能力压入紧凑模型是后训练核心目标。前作 Nemotron-Cascade 1 提出 **Cascade RL**——按领域顺序逐域 RL 的框架，简化多领域 RL 编排。Cascade 2 基于 Nemotron-3-Nano-30B-A3B-Base(30B MoE, 3B 激活)，引用 on-policy distillation 系谱（Agarwal 2024、Gu 2024、Lu & Lab 2025/Thinking Machines、Xiao 2026、Zeng 2026 等）。

#### 2. 现有工作存在的问题
即使 Cascade RL 比任意顺序的顺序 RL 大幅缓解灾难性遗忘，随训练环境增多仍存在能力漂移：某些 RLVR 训练降低熵、缩短推理轨迹从而损害数学；RLHF 优化部分牺牲指令遵循。需要一个额外阶段在 Cascade 过程中**重新平衡**各域能力。

#### 3. Motivation
在 Cascade RL 中引入 MOPD 作为稳定化阶段：用各领域最强中间教师在 RL 过程中蒸馏学生，高效恢复 benchmark 回退、巩固能力——以 30B/3B-激活 的紧凑规模逼近前沿开源模型（约少 20× 参数）。

#### 4. 主要灵感 / 核心直觉
MOPD 在此设定下有三点吸引力：(1) 教师 checkpoint 直接从 Cascade RL pipeline 里按各 benchmark 类别选最强验证 checkpoint，无需引入外部模型家族即可组成能力多样的教师池；(2) 这些教师源自同一 SFT 初始化，共享 tokenizer/词表，降低分布漂移、避免跨家族对齐；(3) MOPD 提供**稠密 token 级**训练优势，比 GRPO 稀疏序列级 outcome reward 样本/步更高效。

#### 5. 主要解决思路(一段话讲清核心)
全程用 GRPO + 严格 on-policy（每轮采一组 rollout 后只做单次梯度更新，IS ratio 恒为 1，并完全移除 KL 项，使 GRPO 退化为 group-normalized REINFORCE + token-level loss）。在 Cascade RL 的特定位置插入 MOPD：学生在 inference 引擎用 π_inf 采样响应，为该样本选一个域教师 π_domain，定义 token 级蒸馏优势 a_t = log π_domain(y_t|s_t) − log π_train(y_t|s_t)（教师比当前策略给该 token 更高概率时为正，训练中收敛到 0），仅在学生采样 token 上计算；因采样/优化策略不一致，加截断重要性权重 w_t = sg[r_t]·1[ε_low≤r_t≤ε_high]（ε_low=0.5, ε_high=2.0），优化 surrogate 损失 L_MOPD（Eq.4）。

#### 6. 方法详解(通俗、分步骤)
**总体流程（Figure 2）**：Base → SFT → **IF-RL** → **Multi-domain RL** → **MOPD** → **RLHF** → **Long-context RL** → **Code RL** → **SWE RL**。
- **SFT**：把所有样本打包到 ≤256K token 序列，单阶段训练，约 **1.5 epoch** 达最优。数学含 1.8M tool-calling + 2.6M 非 TIR 样本（响应由 DeepSeek-V3.2/V3.2-Speciale、GPT-OSS-120B 生成）、816K 证明样本；代码约 165K 去重 prompt，教师 GPT-OSS-120B，按测试用例正确性过滤。
- **IF-RL（首阶段）**：用 Nano-v3 的可验证指令数据 + 动态过滤(去掉全对/全错) + overlong penalty；仅 thinking mode、无奖励模型；IFBench 达 83.13%。置于首位原因：IF-RL 会损害 ArenaHard 而后续 RLHF 几乎不损 IF；且早期 IF-RL 产出的强指令遵循模型可作后续 MOPD 的教师。
- **Multi-domain RL**：增强工具调用、STEM 推理、格式遵循。
- **MOPD**：见 §5。三个域教师——**math 教师=初始 SFT checkpoint**（精心策划 SFT 数据使其数学已强）、**RLHF 教师=从 SFT 经 RLHF 的 checkpoint**、**multi-domain 教师=IF-RL + Multi-domain RL 后的 checkpoint**；prompt 从 RLHF/IF-RL/Multi-domain 训练池 + AceReason-Math(数学) 采样。
- **RLHF**：生成式奖励模型；**Long-context RL / Code RL / SWE RL**（含基于执行的 agentic SWE scaffold RL）依次。

**MOPD 超参**：rollout 4 + 每次 128 prompts（有效 batch 512 responses）；或 512 prompts × rollout 1（更稳、结果相近）；lr=2×10⁻⁶，前 30 步从 2×10⁻⁷ 线性 warm-up（warm-up 对稳定关键，初期 grad norm 大）；通常 40–50 步收敛。

#### 7. 实验数据集
评测：IMO 2025 / IMO-AnswerBench / IMO-ProofBench、AIME 2025/2026、HMMT Feb25（数学）；IOI 2025、ICPC World Finals 2025、LiveCodeBench v6 / LiveCodeBenchPro 25Q2（代码）；SciCode、MMLU-Redux/Pro、GPQA-Diamond、HLE（知识/STEM）；ArenaHard v2、SWE Verified(OpenHands)（对齐/agentic）。基线：Nemotron-3-Nano-30B-A3B、Nemotron-3-Super-120B-A12B、Qwen3.5-35B-A3B 等。训练数据：已公开 SFT-Data 与 RL-Data 集合。

#### 8. 实验结果与主要发现
- **数学/代码（Table 1）**：AIME 2025 92.4 (TIR 98.6)、HMMT Feb25 94.6、IMO-AnswerBench 79.3、IMO-ProofBench 72.9；LiveCodeBench v6 87.2、LCBPro 25Q2 Easy 87.0/Med 27.6；**IMO 2025 35 pts 金牌级、IOI 2025 439.28 金牌级、ICPC WF 2025 解出 10/12**。继 DeepSeek-V3.2-Speciale-671B 之后第二个达 IMO/IOI/ICPC 金牌级的开源权重模型，参数约少 20×。
- **MOPD 效率（核心证据）**：AIME25（Fig.3c），math-only 下 GRPO 25 步 89.9→91.0，**MOPD 30 步达 92.0 并恢复到教师水平**；ArenaHard v2（Table 3），MOPD **52 步** 把 Hard Prompt 71.5→85.5、Creative Writing 40.6→71.0，而 RLHF 需 **160 步** 才到 80.7/71.2。
- **代价（限制性发现）**：在知识/STEM 上**弱于** Qwen3.5-35B-A3B——MMLU-Redux 86.3 vs 93.3、MMLU-Pro 79.8 vs 85.3、GPQA-Diamond 76.1 vs 84.2、HLE 17.7 vs 22.4；SciCode 36.4 也低于 Super-120B(42.1)。即该 pipeline 把能力预算重压在推理/agentic，牺牲了通用知识广度。

#### 9. 结果如何支撑其主张
"MOPD 更省步达教师水平"由 AIME25 与 ArenaHard v2 的步数–分数对照直接支撑（dense token 信号 vs 稀疏 outcome reward）；金牌级 IMO/IOI/ICPC 结果支撑"高 intelligence density"。但"恢复回退、维持各域强度"的主张主要由 MOPD 单阶段的局部对照支撑，缺乏"有/无 MOPD 的完整 pipeline 端到端消融"。

#### 10. 逻辑自洽性(中性评估)
方法描述与公式自洽，MOPD 的 reverse-KL 优势 + 截断重要性权重设计清晰、超参透明。但有几点需中性看待：(1) 这是技术报告 + 模型发布，**无端到端训练代码**，外部不可完整复现；(2) Cascade RL 的阶段顺序自承"非普适常数、依模型行为动态决定"，本质是经验工程而非可迁移原则；(3) MOPD 与 abstract 措辞"throughout the Cascade RL process"略有出入——Figure 2 实际把 MOPD 作为 Multi-domain RL 之后的**单个稳定化阶段**，而非贯穿全程的并行机制〔原稿"贯穿 Cascade 全程"已据 Figure 2 与 §4.4 修正为定位于特定阶段〕；(4) 知识/STEM 的明显落后说明"接近前沿"仅限数学/代码维度，整体能力画像并不均衡。

#### 11. 残留问题 / 局限
- 知识广度(MMLU/GPQA/HLE)显著落后同级 Qwen3.5，pipeline 存在明显能力取舍。
- 缺乏"完整 pipeline 有无 MOPD"的端到端消融，MOPD 贡献多以局部 matched-checkpoint 对照展示。
- 阶段顺序高度经验化、依赖大量内部数据与教师 checkpoint，迁移到其他 base/数据未知。
- 训练代码闭源，仅放模型与数据集，复现门槛高。
- 教师全部源自同一 SFT 初始化，限制了"教师比学生强多少"的上限（无外部更强教师注入）。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 无独立训练代码仓（CloneTier=B，未 clone）。NVIDIA 发布：HF Nemotron-Cascade-2-30B-A3B（后训练模型）、Nemotron-Cascade-2-SFT-Data、Nemotron-Cascade-2-RL-Data。
- 框架：**Nemo-RL**（NVIDIA 开源后训练库，§4.1.2 明确"using the Nemo-RL repository"；环境用 Nemo Gym）。RL 算法 GRPO + 严格 on-policy（单次更新、IS ratio=1、完全移除 KL，退化为 group-normalized REINFORCE + token-level loss）〔原稿曾误记 NeMo-Aligner，已据 §4.1.2 确认为 Nemo-RL〕。
- 论文未随附本工作专用的端到端训练脚本；项目页 https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/ 。


---


### nemotron_nano2 — NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model

> **一句话重点 (TL;DR)**：基于 Nemotron-H 的混合 Mamba-Transformer 推理模型，先预训练 12B base(20T tokens, FP8)，多分支对齐(SFT+GRPO+DPO+RLHF+模型合并)后用 Minitron 剪枝 + 仅 forward-KL 的 logit 蒸馏压到 9B，目标在单张 A10G(22GiB)上做 128k 推理，较 Qwen3-8B 同精度下吞吐高 3×–6×。

**元信息**：arXiv 2508.14444 (v4, 2025-09-02) ｜ NVIDIA ｜ 2025-09 ｜ 主题 T?/中等相关（模型压缩/蒸馏 + 多算法对齐；含 iterative on-policy DPO 与 forward-KL 蒸馏，非通用 on-policy KD 方法论）｜ 代码 无完整训练代码（NVIDIA 模型/数据发布）｜ 框架 NeMo / Megatron

#### 1. 相关工作与进展
推理模型需生成长 thinking 轨迹，对吞吐压力大。Nemotron-H 提出把多数自注意力层替换为 Mamba-2 的混合架构以提升长序列生成速度。压缩侧承袭 Minitron（剪枝 + 知识蒸馏），Mamba 重要性估计沿用 Taghibakhshi 2025。对齐侧组合 SFT/GRPO/DPO/RLHF 等成熟算法。

#### 2. 现有工作存在的问题
- 12B bf16 权重 22.9GiB > A10G 22GiB，必须压缩才能单卡 128k 推理；
- 推理模型在不同 thinking budget 下鲁棒性差：budget 截断后仍停留 thinking 模式、生成多余 `</think>`，well-formedness 下降；
- Stage-1 SFT 的 128k 拼接会损害工具调用学习；
- 推理能力与 chat 能力存在权衡。

#### 3. Motivation
在硬件内存约束（单 A10G）下，造一个既快（混合架构）又准、且支持 128k 上下文与可控 thinking budget 的紧凑推理模型；通过多分支对齐 + 合并缓解能力权衡，通过截断训练提升 budget 鲁棒性，通过 Minitron 剪枝 + 蒸馏在精度损失可控下压缩。

#### 4. 主要灵感 / 核心直觉
- 用 Mamba-2 替换大部分自注意力层，把长生成的吞吐瓶颈打开；
- 压缩本质是"先训大再剪小再蒸馏恢复"，logit 蒸馏（forward KL）比普通微调更能恢复精度；
- 不同能力（IFEval/工具/Arena-Hard）彼此干扰，分支独立优化后做 checkpoint 插值合并是低成本折中；
- 训练时混入"突然截断的推理轨迹"，模型学会在 budget 内收尾。

#### 5. 主要解决思路(一段话讲清核心)
预训练 12B 混合 Mamba-Transformer base(20T tokens, FP8) → 多阶段 SFT + 多分支 RL/DPO/RLHF 对齐并合并得对齐 12B → 用 Minitron(剪枝 + forward-KL logit 蒸馏)压到 9B，得最终 Nano 2 推理/base 模型。

#### 6. 方法详解(通俗、分步骤)
**对齐（Base → 3 阶段 SFT → DPO/GRPO/RLHF 分支 → Merged）**：
- **Stage-1 SFT**：全量数据，混入约 10% 去除推理轨迹的"空 trace"样本以支持 reasoning-off 直答；拼接成长序列（约 128k）。
- **Stage-2 SFT**：专攻工具调用，**不拼接**（修复 Stage-1 拼接对工具学习的破坏）。
- **Stage-3 SFT**：强化长上下文，并加入把推理轨迹**突然截断到 1–2k token**（保留最终答案）的增强样本，提升 thinking budget 鲁棒性〔原稿误记为截断到 12k，已据正文 §3.x 改为 1–2k〕。
- **IFeval RL**、**iterative on-policy DPO**（工具调用）、**RLHF(GRPO)**（HelpSteer3，Qwen-based 奖励模型）分支独立优化。
- **DPO 保持 on-policy**：在 WorkBench 多步可验证工具调用环境中，对每个 checkpoint 生成 on-policy 正样本(成功调用)/负样本(失败生成)，迭代 DPO，保证在线性；BFCL v3 评测。
- **模型合并**：对推理强/chat 强 checkpoint 做插值得 Merged 12B。

**剪枝 + 蒸馏（Minitron，压 12B→9B 推理模型分阶段进行）**：
1. 深度剪枝到 56 层；KD 约 60B tokens @8192 序列长。
2. 宽度剪枝 + KD：约 50B @8192、约 25B @49152、约 1B @262144。
3. DPO → 4. GRPO → 5. KD 约 0.4B @262144 恢复 post-RL 回退 → 6. RLHF → 7. 在 step 5/6 间做 0.5 线性插值合并。
- **重要性估计（仅前向）**：层重要性用逐层临时移除后与原 logits 的 MSE；FFN 神经元/embedding 通道用 1024 样本校准聚合；Mamba head 按 Taghibakhshi 2025（本工作压缩比小，剪 Mamba head 收益有限，故只剪 FFN+embedding 维度 + 深度）。
- **架构搜索**：按 128k/bs1 内存排候选，top-3 各做短 KD 后选 **Candidate 2（精度 63.02）**。
- **重训蒸馏**：对剪枝模型做 **logit-based 蒸馏，仅用 forward KL 散度损失**（accuracy recovery 阶段专用，优于普通微调）；消融(Table 11, ~6B tokens KD)：reasoning-SFT 数据占比 50/50→57.5、70/30→58.5、90/10→57.2，**70%/30% 最佳**。

#### 7. 实验数据集
- 预训练 20T tokens(FP8)；约 5% 预训练数据含刻意截断的推理轨迹，用于推理时 budget 控制。
- 后训练总计约 **90B tokens**（多为单轮 prompt-response），SFT 域分布(Table 7)：Math 1.5M、Coding 1.1M、Science 2.0M、Tool-calling 400K、Conversational 1.5M、Safety 2K、Multilingual(全域) 5.0M。响应多由 DeepSeek-R1-0528 与 Qwen3-235B-A22B 生成〔原稿"SFT ~80B"已据 abstract 改为后训练 ~90B〕。
- 评测：AIME-2024/2025、MATH-500、GPQA-Diamond、LiveCodeBench、SciCode、HLE、IFEval、BFCL v3、Arena-Hard、Global-MMLU-Lite、MGSM 等。

#### 8. 实验结果与主要发现
- Nemotron-Nano-9B-v2 在推理基准上与 Qwen3-8B 相当或更优，在 8k 输入/16k 输出等生成密集场景吞吐高约 3×–6×（最高 6×）。
- 阶段分析（Figure 6）：DPO 与 GRPO 显著提升 function-calling(BFCL v3) 与 instruction-following(IFEval)，但 GRPO 暂时损害 MMLU-Pro（post-GRPO KD 恢复）；RLHF 提升 Arena-Hard 对齐但引入回退，经模型合并恢复。
- 截断训练显著改善短 budget 下的 well-formedness（Figure 5a→5b）。
- 成功在单 A10G(22GiB, bf16)实现 128k 推理。

#### 9. 结果如何支撑其主张
吞吐对比直接支撑"混合 Mamba-Transformer 提升长生成速度"；单卡 128k 推理达成支撑"压缩到内存约束内"；Figure 6 的逐阶段曲线把各对齐/恢复步骤的作用拆开，支撑多分支 + 合并的设计；Table 11 的数据配比消融支撑 forward-KL KD 的数据选择。

#### 10. 逻辑自洽性(中性评估)
作为工业级技术报告，pipeline 描述详尽、消融到位，主张与证据基本对齐。但局限明显：(1) 这是模型/数据发布而非可复现的方法论论文，完整训练代码不在发布仓内，外部难以独立复现；(2) "on-policy DPO"与本课题关心的 on-policy KD 关系较弱，蒸馏部分用的是 forward-KL logit 蒸馏（教师固定数据/logits，本质 off-policy KD），与 on-policy distillation 不是同一范式；(3) 大量超参/阶段为工程经验选择，缺乏对替代方案的系统对照。

#### 11. 残留问题 / 局限
- 训练代码闭源（NeMo/Megatron 内部栈），仅放模型与多数数据集，方法可复现性受限。
- forward-KL KD + 固定教师 logits，无 on-policy 蒸馏的分布真实性收益；reverse vs forward KL 取舍未讨论。
- 多分支合并(0.5 线性插值)为经验做法，缺乏对合并系数/方式的系统消融。
- thinking budget 控制依赖训练时截断样本与推理时强制插入 `</think>`，对极短 budget 的鲁棒性仍有边界。
- 压缩比较小(<15%)使其"剪枝 + 蒸馏"结论未必外推到激进压缩。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 权重：HF nvidia（Nemotron-Nano-9B-v2、12B-v2-Base、9B-v2-Base）。
- 数据：Nemotron-Post-Training-Dataset-v1、Nemotron-Personas、Aegis Content-Safety v2 等多数预/后训练数据集开源。
- 训练栈：NVIDIA NeMo / Megatron（FP8 预训练）；评测用 lm-evaluation-harness、math-verify。**完整训练代码不在发布仓内**（属模型/数据发布，记录不 clone）。
- 对齐算法组合：3 阶段 SFT + IFeval RL + GRPO(RLHF, Qwen-based RM) + iterative on-policy DPO(工具) + 模型合并；压缩：Minitron 剪枝 + forward-KL logit KD。


---


### opd_blog — On-Policy Distillation (Thinking Machines Lab 博客 + tinker-cookbook 配方)

> **一句话重点 (TL;DR)**：系统性推广"on-policy distillation"——学生在自身采样的轨迹上，以更强教师逐 token 的分布（reverse KL）作为唯一稠密监督，兼顾 on-policy 的分布真实性与教师反馈的稠密性；博客在推理(数学)、个性化(指令遵循恢复)、多轮工具使用三场景演示，配套 tinker-cookbook 给出可复现 LoRA 配方。

**元信息**：博客 https://thinkingmachines.ai/blog/on-policy-distillation/（无 arXiv，非论文）｜ Thinking Machines Lab（Kevin Lu 等）｜ 2025 ｜ 主题 T?/High（on-policy 蒸馏方法论）｜ 代码 https://github.com/thinking-machines-lab/tinker-cookbook（已 clone，14MB）｜ 框架 Tinker SDK（自研托管训练 SDK，基于 LoRA），非 veRL/TRL

#### 1. 相关工作与进展
大模型蒸馏分两类：off-policy（在教师生成的固定数据上做 SFT）与 on-policy（学生采样自身轨迹、教师对学生轨迹逐 token 打分）。off-policy/SFT 易"复述"教师局部模式且存在训练-推理分布不匹配（exposure bias）；on-policy RL（如 RLVR）反馈稀疏、采样成本高。博客把 on-policy distillation 作为兼顾稠密监督与 on-policy 真实性的范式系统推广（配套 off-policy 与 SDFT recipe 作对照）。

#### 2. 现有工作存在的问题
- off-policy 蒸馏/SFT：学生在自身推理轨迹上从未被纠正，误差累积；
- on-policy RL（GRPO）：仅序列级稀疏奖励，token 学习效率低；
- 单纯模仿教师文本无法在"学生自己访问到的状态"上获得稠密反馈。

#### 3. Motivation
让学生在自身采样轨迹上，以教师逐 token 分布为稠密监督（KL），从而结合 on-policy 分布真实性与教师反馈稠密性；并展示该范式在推理、个性化、多轮工具使用三场景的有效性与效率。

#### 4. 主要灵感 / 核心直觉
RL 的问题是奖励稀疏、SFT 的问题是分布失配；若让学生自己采样（解决失配），同时让教师对每个 token 打分（解决稀疏），就能两全。教师 logprob 提供的稠密信号比序列级 outcome reward 样本/步效率高得多。

#### 5. 主要解决思路(一段话讲清核心)
学生采样响应，环境不提供任何 reward（既非正确性也非格式），唯一监督是最小化学生与教师在**学生轨迹**上的逐 token **reverse KL**（KL[p‖q]=log p_student − log q_teacher，仅在学生采样到的 token 上算）。代码上把 **−kl_penalty_coef·reverse_KL 作为 per-token advantage**，经 Tinker 的 importance-sampling loss 回传——即用策略梯度形式实现 token 级 KL 蒸馏，而非直接的 KL 散度损失项。全程 LoRA。

#### 6. 方法详解(通俗、分步骤)
代码确认（`distillation/train_on_policy.py::incorporate_kl_penalty`）：
1. 学生采样轨迹，记录 token 级 sampled_logprobs(p) 与 mask。
2. 对每条 datum 用其对应 teacher sampling client 计算 teacher_logprobs(q)。
3. reverse_KL = (log p − log q)·mask；逐 token advantage = −kl_penalty_coef·mask·reverse_KL（默认 kl_penalty_coef=1.0）。
4. 可选 `kl_discount_factor`（默认 0.0）优化折扣未来 KL——博客实验称无明显增益。
5. 把该 advantage 加到 datum 的 advantages，经 importance-sampling loss 更新。
6. 仅对学生生成 token 计 loss（系统提示、用户消息、工具返回、assistant header 等被 mask）。
7. **多教师**：对每个数据集配 (teacher_model, groups_per_batch)，分别采样后拼接成批；trainer 从各配置采样再拼接。
8. **多轮工具使用**（`harbor_multiturn.py`）：在 Harbor sandbox 用 `reward_fn=zero_reward`（恒返回 0）覆盖默认 HarborReward，唯一信号为对教师的 KL；复用 `tool_use` 库 + `harbor_rl` recipe。

#### 7. 实验数据集
- 推理：SFT 用 OpenThoughts3-1.2M；on-policy 蒸馏用 DeepMath-103K；评测 AIME'24。
- 个性化：内部文档 + 重采样 Tulu3（assistant 轮由 Qwen3-8B 重新生成）做 SFT，Tulu3 prompts 做 on-policy 蒸馏；评测 IFEval。
- 多轮工具使用：Harbor sandbox 任务（如 terminal-bench@2.0）。

#### 8. 实验结果与主要发现
- **推理**：① OpenThoughts3 上 SFT（rank-128 LoRA，lr=1e-3 LoRA / 1e-4 full，batch=128，3000 步）→ AIME'24 约 55%；② 加载该 ckpt 在 DeepMath 上 on-policy 蒸馏（lr=1e-4 LoRA / 5e-5 full，groups_per_batch=512，rank-128，约 100 步）→ AIME'24 约 65%。即仅约 100 步蒸馏即从 55% 升到 65%。
- **个性化**：SFT 初始化后在 Tulu3 prompts 上蒸馏（lr=1e-4，groups_per_batch=64），IFEval 约 100 步恢复。
- **多轮工具使用**：README 示例以 **Kimi-K2-Thinking** 同时作学生与教师（max_turns=10、group_size=4、groups_per_batch=8、lora_rank=8、kl_penalty_coef=1.0），在 Harbor 沙箱中以纯 KL 信号训练〔原稿仅写 Qwen3，已据 README 补正为 Kimi-K2-Thinking 示例〕。
- 提供各 LoRA rank(8/32/128) 的 Tinker checkpoint 句柄复现。

#### 9. 结果如何支撑其主张
"SFT 55% → on-policy 蒸馏约 100 步 65%"直观显示在学生自身轨迹上用教师稠密信号的样本/步高效性，对照 off-policy SFT 的瓶颈支撑"on-policy + 稠密 KL"的优势。三场景（推理/个性化/工具）覆盖支撑该范式的通用性。代码与 checkpoint 公开使主张可复现。

#### 10. 逻辑自洽性(中性评估)
方法论清晰、代码与博客一致（reverse KL 作 advantage、纯 KL 无 reward、仅学生 token 计 loss 均经代码核实）。但需注意定位：这是**博客 + 配方**而非受控论文，多数结论以单点曲线/示例形式给出，缺乏严格的多 seed/基线对照与统计显著性；"约 55%→约 65%"为近似值。多教师在博客中**未展示**（仅 recipe 提供）。teacher 用 Qwen3-32B 推理、个性化用更强模型，学生与教师同族（共享 renderer/tokenizer）降低了分布漂移，跨家族教师需自行改 renderer（代码注释明确提示），通用性边界未系统评估。

#### 11. 残留问题 / 局限
- 非论文，无严格基线/统计，数值为近似单点。
- on-policy 蒸馏需教师在线 logprob 计算，依赖 Tinker 托管服务，复现门槛绑定该 SDK。
- 教师与学生不同 renderer/tokenizer 时需手工对齐（代码留有 TODO 提示），跨家族蒸馏未演示。
- 多教师能力仅在 recipe 提供、博客未展示；reverse-KL（mode-seeking）相对 forward-KL 的取舍未深入讨论。
- kl_discount_factor 在实验中无增益，其适用场景不明。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 代码：https://github.com/thinking-machines-lab/tinker-cookbook （已 clone，14MB）。核心：`tinker_cookbook/distillation/`（`train_on_policy.py`、`train_off_policy.py`、`sdft.py`、`datasets.py`）与 `tinker_cookbook/recipes/distillation/`（`on_policy_distillation.py`、`off_policy_reasoning.py`、`on_policy_multi_teacher.py`、`harbor_multiturn.py`、`on_policy_distillation_harbor_multi_turn.py`），含 `recipes/distillation/README.md` 复现脚本。
- 框架：Tinker SDK（基于 LoRA 的自研托管训练 SDK）；蒸馏 loss 与采样由 Tinker 服务侧执行，cookbook 仅给 recipe 与超参。启动如 `python -m tinker_cookbook.recipes.distillation.on_policy_distillation model_name=Qwen/Qwen3-8B-Base ... lora_rank=128`。
- 关键超参/默认：kl_penalty_coef=1.0、kl_discount_factor=0.0、group_size=4、单教师默认 teacher=Qwen3-8B、学生 Qwen3-8B-Base；多教师 recipe 默认 DeepMath→Qwen3-32B、Tulu3→Qwen3-235B-A22B-Instruct-2507、学生 Qwen3-8B，各 groups_per_batch=512〔已据 `on_policy_multi_teacher.py` 补正 Tulu3 教师为 Qwen3-235B〕。
- 配套提供各 LoRA rank(8/32/128) 的 Tinker checkpoint 句柄。


---


### opd_survey — A Survey of On-Policy Distillation for Large Language Models

> **一句话重点 (TL;DR)**：本课题(MTP+OPD / TSRD)的核心背景综述——把 On-Policy Distillation(OPD)统一刻画为"学生采样轨迹上的 f-散度最小化",沿三条设计轴(优化什么 / 信号从哪来 / 如何稳定)组织 >100 篇文献,并系统给出成功条件、失效模式与 OPD↔KL-约束 RL 的连接;提供统一分析词汇与"一方法一类别"分类。

**元信息**：arXiv 2604.00626（v3, 2026-05-18, cs.LG, 78 页）｜ 腾讯大语言模型部（Mingyang Song、Mao Zheng,China）｜ 2026 Preprint｜ 主题 T1/T2/T4 综述,High｜ 代码 github.com/nick7nlp/Awesome-LLM-On-Policy-Distillation（awesome-list / 论文清单类综述仓,CloneTier=B,**未 clone**;无独立训练代码）｜ 框架 n/a（综述）

> 说明:本条为 awesome-list 式综述(paper-only)。§6=分类框架(taxonomy),§7=失效/成功条件,§8/§9=覆盖范围(coverage)。无独立实验,数值多为转引各原始方法。

#### 1. 相关工作与进展
知识蒸馏 [Hinton 2015] 从同架构压缩工具,演化为跨规模/跨架构迁移能力的通用机制;DeepSeek-R1 把 671B MoE 教师蒸到 1.5B–70B 稠密学生使之具体化。OPD 起点:GKD [Agarwal 2024] 与并发 MiniLLM [Gu 2024] 于 2023 年中把 on-policy 蒸馏带入 LLM,两年内扩展到散度设计、reward-guided、self-play、多教师辩论、agentic 轨迹蒸馏、跨模态等 >100 篇。OPD 已进入生产线:Qwen3、DeepSeek-V4、Gemma 2、MiMo-V2-Flash 均把它作核心训练成分(DeepSeek-V4 更以纯多教师 OPD 替换混合 RL 阶段做模型整合)。既有蒸馏综述 [Xu 2024] 仍用经典压缩框架、把 off/on-policy 当可互换变体。

#### 2. 现有工作存在的问题
工业主流是 off-policy 静态模仿(学生在固定语料/教师预生成轨迹上匹配 next-token 分布,每步条件于完美教师前缀)。其结构性缺陷随任务变长、推理密集而加重:推理时学生从自身部分输出自回归生成,偏离即进入训练未覆盖状态——这对应交互式模仿学习的复合误差 O(εT²)(DAgger [Ross 2011])。文献分散在 KD/RLHF/imitation 三社区,记号、基准、失效分类各异,缺统一数学处理与白盒/黑盒/teacher-free 的系统比较。

#### 3. Motivation
为爆发式增长(>100 篇)的 OPD 文献提供统一分析框架与设计中心分类;把"从 off-policy 到 on-policy"重述为序列决策问题,证明核心 OPD 算法都是"学生采样轨迹上的 f-散度最小化"(对应 DAgger 把 O(εT²) 降到 O(εT)),从而连接 KD、RLHF、imitation learning 三条线。

#### 4. 主要灵感 / 核心直觉
统一直觉:改变"训练数据从哪来"(从静态语料转为学生自身演化策略)比改变"匹配什么"更关键。学生提议轨迹、填充部署时会访问的状态,教师在这些状态上给反馈;散度生成元 f 决定似然比的隐式加权(forward KL 覆盖模式/up-weight 学生低估处,reverse KL 寻峰/up-weight 高估处),πmix 控制 on-policy 探索程度。

#### 5. 主要解决思路(一段话讲清核心)
统一目标 L_OPD(θ)=E_{y~πmix}[Σ_t D_f(p_T(·|x,y_<t), p_θ(·|x,y_<t))](Eq.8):f-散度家族(forward/reverse KL、JSD、α-divergence)× 采样混合 πmix × 散度内参数序。把三个奠基方法映入此空间:GKD(πmix=λp_θ+(1−λ)p_data,散度无关)、MiniLLM(reverse KL + REINFORCE)、DistiLLM(skew KLD + replay buffer + 自适应调度)。再沿三设计轴展开方法,并用 §7 统一解释成功/失效。

#### 6. 方法详解(分类框架,通俗、分步骤)
三条对应顺序设计决策的轴(Fig.1 taxonomy,每方法归一主类):
- **§4 目标函数设计**:4.1 固定散度(GKD/MiniLLM/DistiLLM/DistiLLM-2/KETCHUP/vOPD/AntiSD);4.2 自适应散度(ToDi/AKL/EOPD/AOPD,按局部几何在 forward/reverse 间切换,优于固定);4.3 RL-增强目标(G-OPD/RLKD/KDRL/RLAD/AlignDistil 等——证 OPD 是 KD-约束 RL 特例,可超越教师上限)。
- **§5 信号源与教师架构**:5.1 白盒 logit(同族 / 跨族,后者处理词表失配 DSKD/ULD/TAID 等);5.2 黑盒/API 受限(标量奖励或成对偏好:Lion/GAD/LUFFY/ThinkTuning 等);5.3 自蒸馏(**最大且增长最快**:5.3.1 特权信息 OPSD/CRISP/OEL/**OPHSD**/COPSD 等;5.3.2 纯自蒸馏 SDFT/SPIN 等;5.3.3 外部反馈 SDPO/SD-ZERO/**OpenClaw-RL** 等)。
- **§6 训练效率与稳定**:6.1 token/样本加权(TIP/SCOPE/R-OPD 等);6.2 课程与难度自适应(PACED/Stable-OPD/CaOPD 等);6.3 计算优化(Lightning-OPD/SKD/FOPD 等)。
- 另含 §2.4 蒸馏 scaling law(Busbridge 2025:教师过强出现 capacity gap)与 §3.3 方法选择因素(把部署约束/算力映射到可用方法)。

#### 7. 实验数据集
综述本身无实验。用 per-section 对比表(Tables 1, 3–9, 2)汇总各方法的类别、核心贡献、信号源、loss 粒度等。覆盖白盒/黑盒/teacher-free 三大设定及散度设计、reward-guided、self-play、multi-teacher debate、agentic 轨迹蒸馏、跨模态等分支。引用的具体数值(如 DeepSeek-R1 学生规模 scaling、Qwen3 "OPD 以 ~1/10 GPU 时超直接 RL")均为转引原文。

#### 8. 实验结果与主要发现(综述的核心结论与覆盖)
- **成功条件(§7.1)**[Li 2026i]:① 师生需共享兼容推理模式(top-k token 高重叠;非思考教师蒸进思考学生会因初始重叠过低而失败);② 教师须提供超出学生已有的新能力(同数据同配方训出的师生分布趋同、无可迁移信号)。OPD 收益与"可利用的师生差距"成正比,过小过大都不行。另:Kim & Lee 2026 指出 OPSD 更像**压缩**(让模型更高效表达已知解)而非**纠错**(教会解更难题),推荐 SFT→RLVR→correct-only OPSD 流水线序。
- **失效模式(§7.2)**:flawed prefix trap(学生错误前缀使教师条件分布失准)、extrapolation cliff(λ>1 reward 外推超阈值致格式坍缩)、Rock Tokens(高频结构 token 持续高 loss 却无功能贡献,占大量梯度)、self-play saturation/Ouroboros(自蒸馏锁死自身幻觉)、precision-recall/diversity collapse(reverse KL 高 Pass@1 低 Pass@k)、calibration-capability gap(更强但更过自信)、agentic 多轮坍缩(teacher 硬拷贝重置致 KL 从 2.637 骤降 0.343、轨迹结构侵蚀、reward-hint runaway)。
- **统一理论(§7.3)**:散度选择本质是正则化决策;OPD≈稠密 KL-约束 RL,与 DPO/偏好优化同属"由散度选择与监督密度参数化"的目标族;Stable-OPD 加 reference 散度项 + rollout 混合可破坏长度膨胀自放大环(+7.2%)。
- **决策框架(§7.4)**:师生容量比 >10× 且推理浅时用纯 off-policy SFT;学生 >~7B / 多步推理误差复合 / off-policy loss 平台但 on-policy reward 仍升时切 OPD;否则用 hybrid(off-policy 预热 + on-policy 精修)。
- **工业/系统(§8)**:五种部署模式(两阶段蒸馏、模型整合如 DeepSeek-V4/KAT-Coder-V2、多预算推理 ORBIT、agentic 蒸馏 TCOD/MAD-OPD/Skill-SD/OpenClaw-RL、安全闭环 Safactory);系统侧需教师 co-hosting、logit-tensor 传输(70B 教师 8×H100 约 16GB/batch)、staleness 容忍,常用 OpenRLHF/veRL/SLIME 分离 rollout/scoring/update。
- **开放问题(§9)**:on-policy 蒸馏 scaling law(rollout 预算 R 为新轴)、uncertainty-aware 反馈、agent-level 蒸馏、KD 与 RL 的融合谱系。

#### 9. 结果如何支撑其主张(覆盖与论证质量)
统一框架由 Eq.8 + 三奠基方法映射 + DAgger O(εT²)→O(εT) 论证支撑;成功/失效模式逐条引文献佐证并配 mitigation;OPD↔RL 等价由 G-OPD/Li 2025 的梯度分解(稠密 KD 项 + MC RL 项)支撑。作为综述其"贡献"是组织与统一而非新实验,论证密度高、交叉引用一致,但所有数值依赖原文可信度,综述未独立复现。

#### 10. 逻辑自洽性(中性评估)
分类(一方法一主类)清晰、三轴正交且承接(目标→信号→稳定),失效模式按根因而非症状归类,理论节把分散现象收敛为少数原理,整体自洽。可质疑处:f-散度统一框架对 reward-guided/黑盒方法的覆盖偏形式化;"一方法一主类"对跨多轴方法有归类武断之嫌(作者已说明按最显著贡献归类);DAgger 界在 LLM 上的适用性作者自己也加了限定(教师在 OOD 前缀上可能失准)。

#### 11. 残留问题 / 局限
综述自陈的开放问题即其局限边界:缺 on-policy 蒸馏的联合 scaling law(NT/NS/D/R 指数未定);uncertainty-aware 反馈、agent-level 蒸馏、长 horizon 下后段 token 监督质量退化等仍开放。作为 awesome-list 综述,覆盖随领域(2025–2026 高速增长)易过时;无统一基准复现,方法间数值不可直接横比;部分前沿引用为同期 preprint,结论稳定性待时间检验。

#### 12. 开源代码与框架(链接+框架+代码可得性)
仓库 github.com/nick7nlp/Awesome-LLM-On-Policy-Distillation(awesome-list / curated paper list,CloneTier=B,**本次未 clone**)。无独立训练/实验代码,仅论文清单与分类。综述统一用"f-散度在学生采样轨迹上最小化"的框架刻画各 OPD 算法(框架本身 n/a)。


---


### ophsd — Training with Harnesses: On-Policy Harness Self-Distillation for Complex Reasoning

> **一句话重点 (TL;DR)**：把"推理时脚手架(harness)"从永久固件重定位为临时训练支架——训练时让模型在 harness 内 rollout、用 harness 诱导的轨迹作 teacher(自蒸馏,reverse-KL),把过程性推理能力永久内化进基座参数,推理时撤掉 harness;数学/文本分类上超过 OPSD/GRPO,且再接回 harness 不再增益甚至降分。

**元信息**：arXiv 2605.08741（v1, 2026-05-09, cs.CL）｜ Peking University（Zhengyang Zhao、Lu Ma 共一,Wentao Zhang 通讯）｜ 2026 Preprint｜ 主题 T1/T2（On-Policy Distillation / Agent-harness）,Relevance=Med（与 TSRD"teacher-scaffolded reasoning"高度同构——把脚手架当临时支架内化能力）｜ 代码 github.com/zzy1127/OPHSD-On-Policy-Harness-Self-Distillation（已 clone,~57M;训练数据 LFS 大文件以 SKIP_SMUDGE clone,指针占位未实下载）｜ 框架 veRL（子类化 PPO trainer,base 为 OPSD repo）

#### 1. 相关工作与进展
On-Policy Distillation(OPD):学生自采样轨迹、教师逐 token 监督,分两支——Reward-based(把 reverse-KL 当 policy-gradient 奖励,高方差)与 Loss-based(直接还原可微 token-level 蒸馏损失,稠密低方差但需教师分布)。自蒸馏家族按特权上下文 X 划分(Table 1):OPSD/SDFT 用 verified 参考解、SDPO 用 rollout 中环境反馈、CRISP 用静态"be concise"指令、OEL 用过往轨迹经验知识。另一相关线是 Harness Engineering(NLAHs、Meta-Harness [Lee 2026]、AutoHarness)——优化推理时编排层本身。

#### 2. 现有工作存在的问题
推理时 harness(检索增强、plan-solve、draft-verify 等外部脚手架)能显著提升复杂推理,但提升来自外部流程而非模型本身;一旦移除 harness 能力即失。这种"模型与 harness 分离"带来延迟、token 成本、工程复杂度与新失败模式,也难判断模型本身学到了什么。后训练方法不能直接弥合:SFT 模仿静态示范不教自适应流程;RL 监督稀疏难定位关键过程行为;现有自蒸馏方法的特权上下文 X 都是静态变量(参考解/指令串)——只告诉"好答案长什么样",不教"如何推导出来"(信息性 vs 过程性优势)。

#### 3. Motivation
OPD 在学生自身轨迹上给稠密 token 监督,是"内化 harness 行为"的天然载体。核心问题:harness 诱导的逐步流程能否被吸收进模型参数?把 harness 从静态变量泛化为"程序化、由学生参数驱动的工作流"。

#### 4. 主要灵感 / 核心直觉
受 Learning Using Privileged Information(LUPI)[Vapnik 2015] 启发:把特权输入 z(x) 泛化为任何只在训练可得的 oracle 信息;再用一个确定性、有状态的 harness 程序 H 主动编排 z(x) 与 x 的处理,而非被动拼接到 prompt。学生须从裸输入 x 复现 harness 诱导行为 → 自然划出"可蒸馏边界":结构性推理先验(分解/自验证)可内化,真正的实时外部访问(工具/检索内容)不可。

#### 5. 主要解决思路(一段话讲清核心)
OPHSD:训练时学生在 harness 内 rollout(增强推理流程生成轨迹),把这些 harness 辅助轨迹的终端上下文 C[H_θ(x,z(x))] 喂给同一个 frozen base 模型 p_θ̃ 作 teacher,用 **reverse-KL**(KL(p_T‖p_S),Eq.3)沿学生直接 rollout ŷ~p_θ(·|x) 训练不带 harness 的学生;C[·] 作 stop-gradient target。harness 编排由 θ 驱动(随能力演进),logit 监督锚定在 θ̃(稳定先验)。

#### 6. 方法详解(通俗、分步骤)
- **harness 形式化**(Eq.2):确定性有状态程序,最多 T 次模型调用,经状态转移 τ 与读出 π 产出最终答案,诱导条件分布 H_θ(y|x);注入 z(x) 得 H_θ(y|x,z(x))。
- **两类实例化**(均自 Qwen3-8B):
  ① **Draft-Verify**(在线文本分类,推理先验=case-based comparison):z(x)=在线 MemoryBank M_<x(x 之前流入的全部标注先例)。draft 步检索 top-kd=5 近邻作 in-context demo 生成草稿 ŷd;verify 步用 ŷd 再检索 k+=5 confirmers(同标签)、k−=5 challengers(异标签),组装含 x/ŷd/两检索集的 prompt 出最终答案。终端上下文 C=(x,ŷd,N+,N−)。Embedder=BAAI/bge-small-zh-v1.5;冷启动保护(bank<10 条时退化为单次前向);harness baseline 评测时 bank 仅从测试流重填(防泄漏)。
  ② **Plan-Solve**(数学,推理先验=结构分解):z(x)=参考解 y*。planner 用 (x,y*) 蒸出策略草图 s~p_θ(·|x,y*),solver 用 (x,s) 执行完整推导 y~p_θ(·|x,s);终端上下文 C=(x,s)——学生匹配的是"已见 plan 的 solver"信号,而非 OPSD 那样直接给 y* 原文(经 planner 中介,强迫内化推导结构)。plan 温度 0.3、solve 温度 0.6;harness baseline 评测时移除 y*。
- **超参**(均 Qwen3-8B / veRL):lr 1e-6、batch 64、max gen 8192、8×H100;GRPO group 8、KL 系数 0;OPSD/CRISP 用 reverse-KL,CRISP 每 50 步同步教师。文本分类训 300 步(每 15 步评)、数学训 150 步(每 10 步评,4 次运行平均)。

#### 7. 实验数据集
全部 Qwen3-8B。文本分类两路独立训练(各采 10k,严格防污染):①法条罪名预测 CAIL-2018 训练→LawBench 评测(215 类,报 F1);②化学反应预测 USPTO-50k 训练→USPTO test 评测(10 类,报 acc)。数学:DeepMath 采 10k→AIME24/AIME25/OlympiadBench(取 10% 数据)/HMMT25 评测,报 pass@8(4 次平均)。基线:GRPO、OPSD、CRISP(仅数学)。

#### 8. 实验结果与主要发现
- **文本分类**(Table 2):base→harness→GRPO→OPSD→OPHSD。LawBench F1:55.29→60.22→62.44→64.25→**69.51**;USPTO acc:30.07→79.02→90.01→88.01→**90.81**。OPHSD 双双最高,超 GRPO 7.07/0.80、超 OPSD 5.26/2.80。内化后 OPHSD+Harness 反降(LawBench −1.10、USPTO −7.19)。
- **数学**(Table 5,pass@8):OPHSD avg **69.50**(AIME24 79.17、AIME25 61.67、OlympiadBench 83.82、HMMT25 53.33),超 OPSD 2.82、GRPO 2.93、CRISP 10.75;HMMT25 较 OPSD **+10.83**、较 GRPO **+8.33**。base+harness 较纯 base 平均 +17.83(Table 3)。OPSD 早期与 OPHSD 相当但随后因生成长度坍缩(40 步后)而退化(Fig.5)。
- **内化分析**:文本侧 cite-rate(GPT-4o 判 CoT 是否自发引用先例)OPHSD 训练首阶段即 ≥75%、末期 ≥90%,GRPO/OPSD 始终 <10%(Table 4)——学到的是"案例比较"推理形状而非记忆 bank 内容。数学侧按"base vs harness 能力差"分组(Fig.4):harness 才能解的子集相对提升 +84.62%,原本两边都不解的额外解出 +12.54%,且不损原有能力。优势集中在最难分层(Fig.6:Math hard +22.9、LawBench hard +33.8、USPTO hard +90.4)。

#### 9. 结果如何支撑其主张
"能内化、且 harness 可撤"由 OPHSD(无 harness)≥ OPHSD+Harness、且 cite-rate/分组分析支撑;case study(LawBench idx44 单向检索覆盖学生双向权衡 → harness 干扰;OlympiadBench idx649 plan 显式标注 ordered-pair 陷阱 → OPHSD 自纠、OPSD 出错)给出机制叙事。"过程性 vs 信息性优势"由"plan 经 planner 中介 vs OPSD 直给 y*"的对照支撑。

#### 10. 逻辑自洽性(中性评估)
框架自洽且把 OPSD/SDFT 纳为"平凡静态 wrapper"特例,Appendix C 的三档可蒸馏性分类(Fully/Partially/Non-Distillable + 推理时 z(x) 消融启发式)给出清晰边界。但"OPHSD>OPSD"的部分增益可能源于 OPSD 在该设置下生成长度坍缩(超参/早停敏感),而非纯方法优越;数学最佳分由"取全程最高评估分(4 次平均)"得到,选择性报告需留意。

#### 11. 残留问题 / 局限
作者自陈:仅验证两类代表性 harness,未做大规模 harness 设计扫描;Non-Distillable 类(实时工具/检索内容)无法内化,仅"调用脚手架的程序结构"或可蒸馏但本文未验。外部:全 Qwen3-8B 单一规模;数学评测波动大(故 4 次平均);超参(lr/步数)论文正文未尽列,需查 scripts。原"待核"(Table OCR 错位)已用 PDF 重抽数值订正,本次无新事实性出入。

#### 12. 开源代码与框架(链接+框架+代码可得性)
github.com/zzy1127/OPHSD-On-Policy-Harness-Self-Distillation(已 clone,~57M;`data/deepmath10k/data_train_10k.json` 由 Git LFS 跟踪,本次 GIT_LFS_SKIP_SMUDGE 未实下载)。框架 **veRL**(README:"OPHSD subclasses verl PPO trainer",需 `pip install -e <verl>`;底层 RL trainer base 为 HJSang/OPSD_OnPolicyDistillation)。依赖 torch≥2.4、hydra-core、ray、vllm≥0.6、openai 客户端(harness 经此与 vLLM 通信)。代码:`ophsd_train/src/ophsd/`(trainer+worker,Hydra)、`src/rewards/`、`harnesses/`(每任务自包含子包,共享 `_api.py`、`_memory_bank.py`);三 launcher `train_ophsd_{math,lawbench,uspto}.sh`;LawBench/USPTO 需 `precompute_embeddings` 预算训练嵌入。


---


### opsa — Reducing the Safety Tax in LLM Safety Alignment with On-Policy Self-Distillation

> **一句话重点 (TL;DR)**：把 OPSD 自蒸馏迁到安全对齐——学生在线 rollout、frozen 自身副本(条件于按 prompt 类型选的安全/有益特权上下文)沿轨迹给逐 token KL 监督;创新点是用 **teacher flip rate(TFR)** 离线挑"能把不安全回答翻成安全"的上下文,从而在更小的推理代价下降低 "safety tax"。

**元信息**：arXiv 2605.15239（v1, 2026-05-14, cs.LG）｜ UC Riverside / ICSI / Microsoft / Berkeley Lab（Yu Fu, Longxuan Yu, … Yue Dong 通讯）｜ 2026 Preprint｜ 主题 OPSD/特权上下文自蒸馏家族成员（与 copsd/sdcl 同构,此处特权上下文=安全上下文）,对 OPD 主线 Med 相关｜ 代码 github.com/FYYFU/OPSA（Apache-2.0,~98% Python,已 clone）｜ 框架 NVIDIA NeMo-RL 扩展 fork

#### 1. 相关工作与进展
安全对齐"safety tax"[Huang 2025]:提升拒答鲁棒性常牺牲推理。既有方法多从数据侧改监督信号:SafeChain(外部教师 DeepSeek-R1-Distill-Llama-70B 蒸 40k CoT 安全轨迹)、STAR-1(1k policy-guided 轨迹 + LLM-as-judge 过滤)、SafeKey(aha-moment 前加 dual-path 安全头)、SafePath(注入短安全 primer 锚定 comply-or-refuse)、ThinkSafe [Lee 2026](去外部教师,用 refusal-steering prompt 自蒸馏 in-distribution 安全数据,并指出稠密 token 监督优于稀疏 GRPO 奖励)。本文建立在 OPSD [Zhao 2026] 之上,研究"how(off-policy SFT vs on-policy)"而非"what"。

#### 2. 现有工作存在的问题
即便用 in-distribution 自蒸馏数据,SFT 仍是 off-policy:监督施加于固定示范而非模型自采样轨迹。作者主张这是 safety tax 的第二来源(第一来源是数据分布不匹配)。安全决策集中在早期窗口(前 ~10 token,30 后衰减)与少数 safety-critical token(compliance openers: Here/Sure/Certainly;结构标记 Title/**),而 SFT 对所有位置/token 一刀切求和,稀释了真正决定 comply-or-refuse 的信号(Fig.2 token-level KL 分析)。

#### 3. Motivation
若部分 safety tax 来自 off-policy 监督,则把"序列级模仿"换成"on-policy 逐 token 蒸馏"应能在不付出同等推理代价下提升安全。需把监督集中到 refusal-decision 窗口、并施加在学生自采样轨迹上。

#### 4. 主要灵感 / 核心直觉
安全本质是"在模型自身生成轨迹上的局部纠正",而非整序列均匀模仿;且基座 LLM 已有部分潜在拒答能力,安全更多是"激活潜在安全推理"而非教新能力。on-policy 逐 token KL 天然把更新集中到 teacher 与 student 发散的少数关键 token。

#### 5. 主要解决思路(一段话讲清核心)
单一目标、type-conditional teacher:对 benign prompt x_b 给指令 I_b("此 prompt 安全,正常帮助、勿拒答")防过度拒答;对 harmful prompt x_h 给 I_h("此 prompt 有害,只能拒答")。student(trainable)在线 rollout,teacher(frozen 同模型 p_θ0 + 对应特权上下文 c*)沿 rollout 给逐 token D_KL(p_T‖p_S)(L_OPSA,Eq.2)。学生训练/推理都不带 c*,使安全行为内化进参数。

#### 6. 方法详解(通俗、分步骤)
- **散度**:用 forward+reverse KL 的对称混合(α=0.5),沿用 nemo-rl on-policy 蒸馏的默认(注:非纯 forward KL)。
- **TFR 选上下文**:直接估 Δsafety(Eq.3)不实际(依赖采样轨迹),改用 teacher flip rate(Eq.4)——frozen teacher 在加 c* 后把贪婪解从 unsafe 翻成 safe 的比例;选 c*=argmax TFR(Eq.5)。候选池 C=K=30 个 refusal-steering 上下文,由 GPT-5.5 沿五轴(strength/length/framing/specificity/style)生成。
- **验证 TFR 有效**:跨 3 模型、3 上下文(flip rate 9%→78%),训练后 harmfulness 随 TFR 单调下降(Spearman ρ=−1.00 within model,Fig.3),且不伴随过度拒答上升。
- 训练只用 prompt + 类型标签(无需预生成响应),按 ThinkSafe 设置:AdamW、lr 1e-5、cosine+10% warmup、batch 64、3 epochs、全参数 FT;≤1.7B 用 2×A100,8B 用 4×A100(FSDP)。
- 公式与 OPSD/COPSD 同构,创新在"按 prompt 类型条件化 + TFR 选上下文",属增量贡献。

#### 7. 实验数据集
两 reasoning-model 家族、五个规模配置:Qwen3(0.6B/1.7B/8B)+ DeepSeek-R1-Distill(1.5B/8B)。Prompt 全取自 SafeChain 数据集(harmful Dh + benign Db)。评测三轴:Harmfulness↓(HarmBench/StrongReject/WildJailbreak,Llama-Guard 判)、Over-refusal↓(XSTest safe 子集 / WildBenign,WildGuard 判)、Reasoning↑(GSM8K/MATH500/GPQA 数学QA + HumanEval/MBPP 代码)。复合安全分 S=1−(5 项率均值)。自适应越狱:HarmBench 4 攻击族(HumanJailbreaks/Prefilling/PAP-top5/PAIR,159 behaviors)。

#### 8. 实验结果与主要发现
排序 SafeChain < ThinkSafe < OPSA 在复合安全与推理上均成立(Table 1)。OPSA 相对 ThinkSafe(off-policy NLL):五配置平均 **安全 +4.00、推理 +3.04**;小模型增益最大——R1-Distill-1.5B 复合安全 +8.85、Qwen3-0.6B +5.49。OPSA 同时降 harmfulness 与 over-refusal(尤其自然分布 benign)。自适应越狱:在 Prefilling(攻击早期 token,正中机制)上增益最清晰——Qwen3-1.7B/8B 的 mean ASR 与 pass@N 双双归零;但 20 个 model×attack 格中 OPSA 仅 13/20(mean ASR)、14/20(pass@N)优于 ThinkSafe,PAIR(迭代攻击者-目标-裁判搜索)最难、多处回退。Token 级分析:更新集中在早期 compliance-decision token 附近(较有说服力的机制证据)。

#### 9. 结果如何支撑其主张
"off-policy 是第二来源"主张靠对照实验隔离:ThinkSafe 与 OPSA 同用 SafeChain 自生成数据、仅训练目标不同(NLL vs 逐 token KL),OPSA 全面更优 → 差距归因于 on-policy/逐 token。机制由 Fig.2(KL 集中早期+词汇)与 Prefilling 攻击上的最大增益闭环呼应。TFR 作为选择准则由 Fig.3 单调关系支撑。

#### 10. 逻辑自洽性(中性评估)
逻辑链(off-policy mismatch → 早期窗口失控 → on-policy 逐 token 纠正 → Prefilling 鲁棒)自洽且有 token 级与攻击级双重佐证。但安全无"ground-truth 特权信号",TFR 只是 unsafe→safe 翻转的代理,且 c* 由 GPT-5.5 生成、Llama-Guard 既做数据过滤又做评测(循环依赖,作者已在 Limitations 承认)。OPSA 相对 OPSD 是领域迁移+一个选择准则,创新增量有限。

#### 11. 残留问题 / 局限
作者自陈:① 依赖基座保留可被 c* 激活的潜在安全能力,后训练若已严重覆写则增益小;② teacher 固定 frozen base,周期刷新为学生滑动平均是未来方向;③ harmfulness 度量依赖 Llama-Guard,结论相对该分类器、可能继承其失效模式。外部:对全自适应搜索(PAIR)鲁棒性仍是短板;只在 reasoning model 上验证。

#### 12. 开源代码与框架(链接+框架+代码可得性)
github.com/FYYFU/OPSA(Apache-2.0,~98% Python,已 clone)。核心是 NVIDIA **NeMo-RL** 的扩展 fork(`nemo_rl/`);含 `train_opsd.sh`(on-policy self-distillation)、`train_sft.sh`(baseline)、`evaluation/`(HarmBench/XSTest/WildJailbreak 等)、`docker/`、`uv.lock`;`pyproject.toml`。全参数 FT、uv 安装,支持 Qwen3、R1-Distill。


---


### opsd — Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models

> **一句话重点 (TL;DR)**：同一个 LLM 自任教师与学生——教师条件于"问题 + 参考解(特权信息)"、学生只看问题,沿学生自采样轨迹做逐 token 散度蒸馏,无需外部教师即可用 ground-truth 提供稠密 on-policy 监督,在数学推理上匹配/超过 GRPO 且 token 效率显著更高。

**元信息**：arXiv 2601.18734（v3, 2026-03-20, cs.LG）｜ UCLA / HKU / Meta Superintelligence Labs（Siyan Zhao 等,Feiyu Chen 与 Aditya Grover 共同指导）｜ 2026 Preprint（博客 siyan-zhao.github.io/blog/2026/opsd/）｜ 主题 T1/T4,High（自蒸馏家族原型,直接定义 OPSD 范式）｜ 代码 github.com/siyan-zhao/OPSD（已 clone,~330KB,核心 trainer 完整）｜ 框架 TRL（基于其 experimental GOLD/GKD trainer）

#### 1. 相关工作与进展
推理 post-training 三大路线:RLVR(GRPO 等)、在高质量 CoT 轨迹上 SFT、知识蒸馏。on-policy 蒸馏(GKD [Agarwal 2024]、MiniLLM [Gu 2024]、Thinking Machines 的 OPD 博客 [Lu & Lab 2025])让学生采样自身轨迹、教师逐 token 稠密监督,兼具 on-policy 分布真实性与稠密反馈。与 STaR/ReST(条件于 hint/答案生成 rationale → 拒绝采样 → SFT,硬蒸馏)、context distillation [Snell 2022](特权上下文教师 + SFT 学生,off-policy)、in-context editing [Qi 2025](on-policy 软蒸馏内化上下文知识)相关。并发的 SDPO [Hübotter 2026]、SDFT [Shenfeld 2026] 探索同构的自蒸馏。

#### 2. 现有工作存在的问题
(1) RLVR/GRPO:每 prompt 采一组响应成本高、方差大;一组全对/全错时优势为 0、梯度消失;奖励稀疏且对所有 token 一刀切。(2) SFT:exposure bias、泛化弱。(3) 传统蒸馏:off-policy 分布不匹配(训练见完美前缀、推理自生成)。(4) on-policy 蒸馏需要一个独立、往往更大的教师模型,且未显式利用推理数据集中已有的 ground-truth 解。

#### 3. Motivation
现代 LLM 已具强推理能力,问:模型能否通过自蒸馏当自己的教师?既能省掉外部教师,又能直接利用数据集里的参考解作为特权信息。

#### 4. 主要灵感 / 核心直觉
受人类学习启发:做错题后看正确解能 rationalize 步骤、定位自己错在哪;且"评估比生成更易"[Naor 1996, Sun 2024],推测"对给定正确答案做 rationalization"也比从零生成更易。故让模型在见到 y* 后隐式 rationalize,以此监督只见问题的弱版自己。

#### 5. 主要解决思路(一段话讲清核心)
从同一模型 p_θ 实例化两条策略:教师 p_T(·|x,y*)(条件含问题 + 参考解),学生 p_S(·|x)(仅问题)。学生采样 on-policy 轨迹 ŷ~p_S(·|x);loss 最小化沿学生轨迹的逐 token 散度 D(p_T‖p_S)(ŷ|x)= (1/|ŷ|)Σ_n D(p_T(·|x,y*,ŷ_<n)‖p_S(·|x,ŷ_<n))。梯度仅经学生 logits 回传;教师只一次前向(prefill)隐式 rationalize、不真正生成 token(prompt 中要求教师"看完参考解后用自己的方法解",见 Fig.2)。

#### 6. 方法详解(通俗、分步骤)
- **两条策略**:同参数 θ、不同条件上下文;教师额外看 y*。
- **on-policy 采样**:学生生成 ŷ,两条策略在同一学生前缀上各自给 next-token 分布。
- **训练目标(两种实例化)**:① 全词表 logit 蒸馏(如 GKD,full softmax,逐 token f-散度;效果更好但峰值显存高,因每位置存词表大小 logits);② 采样 token 的策略梯度(如 Lu & Lab 2025:把 A_n=log p_T(ŷ_n|·)−log p_S(ŷ_n|·) 当 stop-gradient 优势,做 reverse-KL 风格 policy gradient;省显存)。主实验用 ①。
- **教师固定为初始 policy**(非在线更新的学生),作隐式正则、稳定训练。
- **Per-Token Pointwise KL Clipping**:对每个 token×词表项的散度贡献 min(ℓ,τ) 裁剪。因风格 token('wait'/'think' 等连接词)的逐 token KL 比数学 token 高 6–15×(Table 5),不裁剪会让风格 token 主导信号、训练崩溃(Fig.4)。
- **消融结论**:forward KL > reverse KL/JSD(Table 3,AIME25/Qwen3-1.7B:FKL 36.7→43.9@step50);TM-off 学生 + TM-on 教师在数学 token 上 KL 最大、下游最好;生成长 1024 vs 4096 无一致增益(早期 token 更关键);全词表 > 采样 token(Table 4,Qwen3-4B/2048-gen,pass@8:AIME25 84.1 vs 82.1、HMMT25 60.0 vs 57.3)。

#### 7. 实验数据集
训练:OpenThoughts 数学推理子集(采 ≤30K 问题-解对,含 CoT)。评测:竞赛级数学 AIME 2024、AIME 2025、HMMT 2025。主表 Table 2 报 **Avg@12**(温度 1.0、thinking 模式、max gen 38k,按 Qwen3 博客配置);消融 Table 4 报 pass@8(数值与 Avg@12 不可直接比)。模型:Qwen3-1.7B/4B/8B(instruct)。

#### 8. 实验结果与主要发现
Table 2(Avg@12,基→OPSD):Qwen3-1.7B 37.1→43.4(超 GRPO 37.7、SFT 35.8;分项 AIME24 51.5→57.2、AIME25 36.7→43.9、HMMT25 23.1→29.2);4B 61.2→63.6(>GRPO 62.7);8B 61.8→64.8(>GRPO 64.0)。OPSD 每问题仅 1 rollout、生成 1024 token、约 100 步内收敛(Qwen3-1.7B 在 4×H100 上约 15 分钟,A100/H100 + LoRA);GRPO 用 8 rollouts×16k token,且 100 步内过半 batch 组内 reward 标准差为 0、梯度消失(Fig.3)。SFT 在简洁参考解上微调反而缩短测试生成长度、性能退化。

#### 9. 结果如何支撑其主张
"token 效率"主张由 Fig.3 直接支撑(同训练步数下 OPSD 用更少 token 却全面超 GRPO,且 GRPO 因 reward 多样性坍缩停滞)。"无需外部教师 + 利用 ground-truth"由方法构造与 Table 1 对比(OPSD 同时满足 on-policy/稠密信号/低采样成本/无外部教师四项)支撑。机制证据较弱:全词表>采样 token、forward KL 最优等均有消融,但缺对"为何 rationalization 更易"的直接度量。

#### 10. 逻辑自洽性(中性评估)
框架自洽:教师固定为初始 policy 既是稳定手段也是隐式 KL 正则,与 GRPO 的 reference-KL 思路一致。但教师"只 prefill 不生成"意味着 OPSD 本质是"在学生轨迹上、用 y*-条件分布做逐 token 匹配",其能否提供"新能力"取决于 y*-条件是否真把分布推向更优——这点论文未理论保证,survey(§7.1)亦指出 OPSD 的有效区间介于"教师-学生差距过小/过大"之间,且后续 Kim & Lee 2026 指出 OPSD 更像"压缩(让模型更高效表达已知解)"而非"纠错(教会解更难题)"。

#### 11. 残留问题 / 局限
作者自陈:实验上限 8B(>8B 是否持续未知);未利用答案正确性验证信号(可作额外目标);若问题超出模型理解阈值,即便给 y* 教师也无法提供有效监督,需课程学习。外部视角:主表用 Avg@12、消融用 pass@8 易混淆;仅数学单域;教师固定初始 policy 可能限制能力上探(只匹配、难超越教师)。

#### 12. 开源代码与框架(链接+框架+代码可得性)
github.com/siyan-zhao/OPSD(已 clone,~330KB,代码完整可得)。核心:`opsd_trainer.py`(OPSDTrainer 自蒸馏核心,72KB)、`data_collator.py`、`opsd_train.py`(入口)、`sft_train.py`/`grpo_train.py`(基线)、`scripts/run_opsd*.sh`、`eval/evaluate_math.py`(vLLM)。框架 **TRL**:`environment.yml` 确认 trl==0.26.0、transformers==4.57.1、accelerate==1.11.0、deepspeed==0.18.2、peft==0.17.1、vllm==0.11.0、torch==2.8.0;README 明示"基于 TRL experimental GOLD trainer"。


---


### prism — Beyond SFT-to-RL: Pre-alignment via Black-Box On-Policy Distillation for Multimodal RL (PRISM)

> **一句话重点 (TL;DR)**：在 SFT 与 RLVR 之间插入一个独立 "预对齐" 阶段，用 black-box（无需教师 logits）的对抗式 OPD——policy 对一个含感知/推理双专家的 MoE 判别器做极小极大博弈——修复 SFT 引入的（且对感知/推理异质的）分布漂移，为下游多模态 RLVR 提供更好初始化。

**元信息**：arXiv 2604.28123（v2 2026-05-01）｜ HKUST(广州) + 清华 + 南洋理工 + 人大 + 中科大 + 国科大 ｜ 2026-05 预印本 ｜ 主题 OPD 新用法（SFT/RLVR 之间的预对齐），与 mtp_opd 相关 ｜ 代码 github.com/XIAO4579/PRISM（MIT，已克隆约 133MB，完整）｜ 框架 三阶段：LLaMA-Factory（SFT）+ verl（对齐/RLVR）+ vendored transformers-4.57.0 + MoE 判别器。

#### 1. 相关工作与进展
LMM 标准后训练为两阶段：先在 curated 示范上做 SFT（能力 bootstrap），再用 RLVR 精炼（决定最终性能）。一系列工作分别改进两阶段的有效性与稳定性，包括重设计 importance weighting/clipping 的 GRPO 变体。OPD 表明从自身 on-policy 生成学习可缓解 exposure bias。

#### 2. 现有工作存在的问题
近期发现 SFT 会引入**分布漂移**：既不充分匹配示范策略分布，又丢失模型原有有利分布——SFT 成为漂移源而非纯改进；模型越强，token 级模仿外部示范越易挤占其原生强项。多模态下更突出且**异质**：感知（visual grounding）错误与推理失败的漂移模式不同，会在后续 RL 中复合放大，单一纠正目标无法兼顾。

#### 3. Motivation
如何在进入 RL 前修复 SFT 引入的、且对感知/推理异质的分布漂移？据 OPD 思想把对齐重定位为 SFT 与 RLVR 之间的独立预对齐阶段；并提出 logit-free（black-box）形式以摆脱外部教师依赖（适配 Gemini 等无法访问 logits 的黑盒监督源）；用 MoE 判别器为感知/推理提供解耦纠正信号。

#### 4. 主要灵感 / 核心直觉
"偏离监督分布的方向本身就是奖励信号"：与其用静态 teacher-forced 目标，不如让一个判别器去分辨 policy rollout 与监督池，并把 "像监督" 的程度当奖励；再把这个判别器拆成感知/推理两个专家，分别给出解耦的纠正信号，避免单一标量把异质漂移混为一谈。

#### 5. 主要解决思路(一段话讲清核心)
三阶段流水线：SFT 冷启动 → PRISM 对齐（policy 与 MoE 判别器做 response 级对抗博弈，判别器分作奖励、GRPO 式更新 policy，跑固定步数得 checkpoint）→ 在对齐 checkpoint 上做标准结果型 RLVR（GRPO/DAPO/GSPO）。对齐阶段仅需监督池样本、无需教师 logits。

#### 6. 方法详解(通俗、分步骤)
- **Stage1 冷启动 SFT**：高质量示范上得到初始多模态推理策略。
- **Stage2 分布对齐（adversarial OPD）**：建模为 policy 与 MoE 判别器的 response 级极小极大博弈。判别器含**感知专家 D_v**（查 visual grounding/物体）与**推理专家 D_r**（查推理一致性），判别分为两专家加权组合（Eq.1）。架构沿用 Qwen3-VL-MoE，由**四个 Qwen3-VL-2B 组装为专家、Top-2 路由**（已核 §4 行 467 "four Qwen3-VL-2B"、行 327 "Top-2"）；两专家用 Bradley-Terry loss 学习区分 policy rollout 与 supervision pool，从同一预训练 backbone 初始化并各在视觉描述/推理偏好对上 warm-start，带 load-balancing 辅助损失防专家坍塌。〔已核 PDF §3.2、Fig.2、行 439-443〕
- **对齐优化方式（已核 §3.2.3 / 行 439-443、1272）**：交替进行——policy 用 GRPO 式目标（组内归一化、奖励为 MoE 判别器分）更新，判别器两专家用各自 BT loss 更新；**显式去掉 KL 正则**（KL 系数=0，锚定 SFT 初始化会与 "修正 SFT 漂移" 冲突）；跑固定步数后 checkpoint 作 RLVR 初始化，对感知/推理漂移给出解耦奖励。
- **Stage3 RLVR**：在对齐后策略上做结果型 RLVR（GRPO/DAPO/GSPO）。

#### 7. 实验数据集
- 监督语料：从 **Gemini 3 Flash** 蒸馏的 **113K** 高质量多模态推理语料（针对当前 LMM 零通过率最难题，含密集视觉 grounding 与逐步推理）；其中 **107K 用于 SFT、6K 最高质量留给对齐与 RL**，并补充同 Gemini 系 **1.26M** 公开示范，合计约 **1.37M** SFT 语料。〔已核行 268-277〕
- Backbone：Qwen3-VL-4B/8B。
- 评测：多种多模态基准（用 lmms-eval）。

#### 8. 实验结果与主要发现
- 流程：SFT（1.37M）→ PRISM 对齐（MoE 判别器对抗式 OPD，repair 漂移）→ RLVR；监督池同时作 SFT 基础与对齐参考。
- 主结论（Qwen3-VL）：PRISM 在 GRPO/DAPO/GSPO 多算法、多基准上持续提升下游 RLVR；**PRISM+GRPO 较 SFT→GRPO 基线在 4B/8B 上平均 +4.4 / +6.0 点**（已核行 43、206），DAPO/GSPO 有类似增益。
- 分析：对齐 checkpoint（SFT 后、RLVR 前）精度与 SFT checkpoint 相当但显著缩小了与监督分布的间隙；SFT 单独会在两个尺度上平均拉低 Instruct 起点，PRISM 对齐缓解之。

#### 9. 结果如何支撑其主张
"对齐能改善下游 RL" 由跨三种 RL 算法（GRPO/DAPO/GSPO）一致 +Δ 支撑；"修复漂移" 由对齐前后分布间隙缩小的分析支撑；"感知/推理异质" 由双专家解耦奖励设计 + 相应分析支撑。多算法一致性是较强证据。

#### 10. 逻辑自洽性(中性评估)
对抗式 OPD + 双专家判别器的设计与 "异质漂移需解耦纠正" 的动机自洽；去 KL 的理由（与修漂移目标冲突）合理。但 "black-box" 仅指无需教师 logits，监督仍来自 Gemini 蒸馏语料，因此并非完全无外部强模型依赖；判别器奖励的可靠性依赖 BT 训练质量，存在对抗训练常见的不稳定/奖励 hacking 风险（论文用 load-balancing 与固定步数缓解）。

#### 11. 残留问题 / 局限
- 监督语料强依赖 Gemini 3 Flash 蒸馏（含 113K + 1.26M），方法的 "data-free/teacher-free" 程度有限，仅 logit-free。
- 对抗博弈引入额外不稳定性与算力（4×Qwen3-VL-2B 判别器需独立 warm-start）。
- 评测仅多模态基准；判别器奖励 vs 真值 RLVR 奖励的相互作用、是否引入新偏置未深究。
- +4.4/+6.0 为平均增益，单基准方差与负向案例披露有限。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库：https://github.com/XIAO4579/PRISM（MIT，vendored 上游 Apache-2.0；本地约 133MB，含 vendored transformers-4.57.0、verl、moe（MoE 判别器）、scripts、tools）。数据/checkpoints 在 HuggingFace（prism-vlm）。项目页 https://xiao4579.github.io/PRISM/。
- 框架：Stage1 SFT 用 LLaMA-Factory；Stage2 对齐用 `qwen3_vl_prism`（verl + transformers-4.57.0 + moe）；Stage3 RLVR 用 `qwen3_vl_xxpo_after_prism`（GRPO/DAPO/GSPO）；评测 lmms-eval。
- 代码可得性：完整开源（含判别器与三阶段脚本）。


---


### qwen3 — Qwen3 Technical Report

- **arXiv/链接**: arXiv:2505.09388 (2025-05-15)
- **机构/作者**: Qwen Team (Alibaba)
- **发表/时间**: 2025-05
- **主题/相关性**: T1(High)。**与本综述高度相关**:Qwen3 显式提出 Strong-to-Weak Distillation 的两阶段方案,其中第二阶段就是 **On-policy Distillation(学生自采样序列 + 与教师 logits 对齐最小化 KL)**,是大厂技报里少见的明确 on-policy KD 实践;并提供统一 SFT-RL 四阶段后训练管线作为对照。

#### 1. 开源代码链接
- 仓库:https://github.com/QwenLM/Qwen3 ;权重 https://huggingface.co/Qwen 、https://modelscope.cn/organization/qwen 。
- **该仓为模型发布/权重仓,不含后训练训练代码**(蒸馏/RL 管线均不开源;社区微调依赖 Axolotl/Unsloth/ms-swift/LLaMA-Factory)。CloneTier=B,仅记录不 clone。

#### 2. 使用框架
- Reasoning RL / General RL:**GRPO**(引用 Shao et al. 2024);提到大 batch、多 rollout、off-policy 提升样本效率、通过控制熵稳定训练。报告未提 GSPO(GSPO 为后续单独工作)。
- Strong-to-Weak Distillation:**离线(off-policy)response distillation + 在线(on-policy)logit/KL 蒸馏**。
- **开源仓不含上述训练代码**。

#### 3. 研究背景
- Qwen3 系列含 dense 与 MoE,参数 0.6B–235B(旗舰 Qwen3-235B-A22B)。核心创新:把 thinking(复杂多步推理)与 non-thinking(快速响应)统一进单一模型,支持动态模式切换与 thinking budget(推理 token 预算控制)。

#### 4. 当前存在的问题
- 过去需为 chat 模型(如 GPT-4o)与专用推理模型(如 QwQ-32B)分别部署/训练;
- 为每个小模型独立跑完整四阶段后训练计算开销大、开发成本高。

#### 5. Motivation
- 用统一框架免去多模型切换;用 thinking budget 自适应分配推理算力;
- 用旗舰大模型的知识(strong-to-weak 蒸馏)大幅降低轻量模型后训练成本——报告称蒸馏仅需四阶段训练约 **1/10 GPU 时**,且 Pass@1(即时性能)与 Pass@64(探索能力)均更优。

#### 6. 主要方法
**旗舰模型四阶段后训练管线(图 1)**:
1. **Long-CoT Cold Start**:精选数学/代码/逻辑/STEM 可验证问题,用 QwQ-32B 生成 N 个候选并严格过滤,做少量 SFT,只植入基础推理模式、不追求即时性能(刻意少样本少步数,保留 RL 提升空间)。
2. **Reasoning RL**:筛 3,995 条 query-verifier 对(未用于 cold-start、对 cold-start 模型可学、尽量难、覆盖广),用 **GRPO** 训练;大 batch + 多 rollout + off-policy + 控熵稳定;Qwen3-235B-A22B 在 170 步内 AIME'24 由 70.1→85.1。
3. **Thinking Mode Fusion**:在 Reasoning RL 模型上做持续 SFT,融合 non-thinking 能力;thinking 数据由 Stage-2 模型对 Stage-1 query 拒绝采样生成,non-thinking 数据覆盖代码/数学/指令/多语/创作/QA/角色扮演;设计 `/think` `/no_think` chat 模板与空 think 块;thinking budget(达阈值插入停思指令)为自然涌现能力。
4. **General RL**:覆盖 20+ 任务的奖励系统(指令遵循、格式遵循、偏好对齐、Agent 工具调用含真实环境多轮反馈、RAG 等);三类奖励:规则奖励、带参考答案的模型奖励(Qwen2.5-72B-Instruct 打分)、无参考的偏好奖励模型。

**Strong-to-Weak Distillation(轻量模型:0.6B/1.7B/4B/8B/14B dense + 30B-A3B MoE)**:
- **(1) Off-policy 蒸馏**:用教师在 `/think` 与 `/no_think` 两种模式下的输出做 response 蒸馏,使学生获得基础推理与模式切换能力。
- **(2) On-policy 蒸馏**:**学生自己生成 on-policy 序列**(对采样到的 prompt 在 `/think` 或 `/no_think` 模式下产出响应),再**对齐学生与教师(Qwen3-32B 或 235B-A22B)的 logits、最小化 KL 散度** 进行微调。这是本综述关注的核心 on-policy KD 形式。

#### 7. 实验数据集
- 训练:数学/代码/逻辑/STEM 可验证 prompt;cold-start 精选子集;Reasoning RL 3,995 对;General RL 20+ 任务。
- 评测:AIME'24/'25、LiveCodeBench、CodeForces、GPQA、MMLU、多语基准(Multi-IF 8 语、INCLUDE 44 语)等,thinking/non-thinking 双模式评测。

#### 8. 怎么做的(训练/数据/流程)
- 数据构建两阶段过滤:query 过滤(Qwen2.5-72B-Instruct 剔除不可验证/无需 CoT 即可答对/多子问题的 query,并标注 domain 平衡)+ response 过滤(剔除错误答案、重复、明显猜测、思考与总结不一致、语言混杂、疑似验证集泄漏)。
- Reasoning RL 单次 RL run 内 reward 与验证性能持续提升,无需人工调超参,靠控熵稳态维持。
- 蒸馏先 off-policy 打底再 on-policy 精炼,显著降本(约 1/10 GPU 时)且兼顾即时性能与探索能力。


---


### resd — Learning with Rare Success but Rich Feedback via Reflection-Enhanced Self-Distillation (RESD)

- **arXiv/链接**: arXiv:2605.12741v1 — https://arxiv.org/abs/2605.12741 ；代码 https://github.com/horizon-llm/RESD
- **机构/作者**: Yuwei Zhang、Sha Li、Changlong Yu、Qin Lu、Shuowei Jin、Chengyu Dong、Haoran Liu、Ilgee Hong、Xintong Li、Zhenyu Shi、Bing Yin、Jingbo Shang。机构：UC San Diego、Amazon、Georgia Tech（部分为 Amazon 实习期间完成）。
- **发表/时间**: 2026-05-12（arXiv preprint）
- **主题/相关性**: 在线策略自蒸馏（OPSD/SDPO）在稀疏成功（rare-success）下的改进。与 mtp_opd/TSRD 高度相关——把失败反馈转为可复用的纠错监督（"路径修复"思想），且提出"反思+playbook"为可插拔模块。

#### 1. 开源代码链接
https://github.com/horizon-llm/RESD （已克隆约 52MB；含 verl 源码、selfevolve 模块、docker、environment.yml；含 paper.pdf）。Notion 博客与项目页见 README。

#### 2. 使用框架
基于 veRL（volcengine/verl）与 SDPO（lasgroup/SDPO）。维护两个持久化上下文：**playbook**（受 ACE 思想启发，存储从失败中提炼的可复用经验条目）与**可选的 solution buffer**（缓存成功轨迹）。核心代码在 `selfevolve/` 与 `verl/`。

#### 3. 研究背景
让 LLM 通过与环境交互持续自我改进是后训练核心挑战。RL（PPO/GRPO）在多步任务上受困于稀疏奖励；成功轨迹极少时缺乏稠密监督来引导探索。OPD/自蒸馏（SDPO）用同规模"自教师"条件于特权信息，把稀疏的轨迹级结果转为稠密 token 级信号，且消除外部教师与分布失配。

#### 4. 当前存在的问题
现有自蒸馏把环境反馈当作"被动条件变量"，严重依赖成功示范。作者通过消融发现：当 rollout batch size N=1（无同组成功 peer 示范）时，SDPO 性能大幅退化——说明教师难以仅凭原始失败反馈构造有效纠错分布。在 rare-success（每 rollout 成功率极低）下尤其严重，因为多数 rollout 组只含失败、无正反例。

#### 5. Motivation
问题不在"缺反馈"而在"反馈如何被表示和使用"。原始失败信号往往是诊断性而非指导性的（只说明失败、不说明哪一步导致失败）。需从"被动暴露反馈"转向"主动理解反馈"：(a) 回溯性解释失败轨迹，把最终反馈连到导致失败的中间推理；(b) 跨轨迹保留有用经验（很多失败是同一隐藏规则的重复表现）。

#### 6. 主要方法
RESD 在自蒸馏循环中算子化"主动反馈理解"：
- **回溯反思（REFLECT）**：对失败 rollout 生成自然语言反思 r，定位失败的可能原因与本应采取的修正；同时对已有 playbook 条目打 helpful/harmful/neutral 标签。
- **playbook 策展（CURATE/CONCISE）**：playbook 是持久化的自然语言经验条目集；CURATE 由反思生成非冗余新条目，每条维护 (helpful, harmful) 计数；CONCISE 在每次更新前剪枝（删除净 harmful 条目、超预算时驱逐最久未标记条目）。
- **记忆增强自蒸馏**：teacher 经 EMA 与 student 同步，条件于"富化上下文"（playbook P、反思 r、上次 trial y、反馈 c、buffer 解 B(x)）产出 token 级监督；学生最小化统一的 f-散度自蒸馏目标 L_SD（可取 forward/reverse KL 或 JSD），并对成功样本按 batch 成功率施加 per-sample 加权。因 P、B 跨步持久，即使当前 batch 无成功示范，监督也能随训练改善。

#### 7. 实验数据集
4 个面向"持续学习/分布外"的任务（带丰富执行反馈但二值稀疏奖励）：MANUFACTORIA-HAS（写 DSL 程序判断输入带是否含目标模式，742 训/132 测）、BOUNCINGSIM-EASY（写 Python 模拟 2-D 多物体弹跳动力学-易，640 训/100 测）、BOUNCINGSIM-MEDIUM（同上-中，320 训/100 测）、FINER（用 XBRL 标签标注 SEC 财报中的金融命名实体，1000 训/500 测）〔已核-论文 Table〕。各任务初始成功率差异大（如 BOUNCINGSIM-MEDIUM/部分任务初始成功率极低、近稀疏成功 regime，其余较高），以考察两种 regime。骨干：Qwen3-4B（多数任务）/ Qwen3-30B（BOUNCINGSIM-MEDIUM，见 selfevolve/ 脚本）。

#### 8. 怎么做的(训练/数据/流程)
在线流式（online streaming）协议。训练循环（Algorithm 1）每步：采样 rollout y~π(·|x)→取环境反馈 c 与奖励 R→上下文更新阶段（先 CONCISE 剪枝 P；失败则 REFLECT+CURATE 扩展 playbook，成功则更新 solution buffer 并置 r=∅）→teacher EMA 同步→teacher 条件于富化上下文产出 log-prob→学生最小化 L_SD 更新。实验表明：RESD 显著超过标准自蒸馏基线；仅用单 rollout/prompt 即可比用 8× 样本的 GRPO 更快实现早期提升（交互效率更高）；SDPO+Ref(N=1) 可恢复并超过 SDPO(N=8)，主要靠减少"完全错误"的 prompt 数量。结构化反馈支持从失败中样本高效 bootstrap，奖励优化在成功 rollout 足够后作为互补工具。


---


### rethink_opd — Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe
- **arXiv/链接**: arXiv:2604.13016 (v2, 2026-04-15);代码 https://github.com/thunlp/OPD
- **机构/作者**: 清华大学(THUNLP)+ 上海科技大学 + UIUC + 中国人民大学。Yaxuan Li、Yuxin Zuo、Bingxiang He(共同一作)、Jinqian Zhang、Chaojun Xiao、Cheng Qian、Tianyu Yu、Huan-ang Gao、Wenkai Yang、Zhiyuan Liu、Ning Ding(通讯)。
- **发表/时间**: arXiv 预印本,2026-04;已被 ICML 2026 FoGen Workshop 接收。
- **主题/相关性**: T1, T4(High)。直接研究 OPD 的训练动力学/成败机理,与本项目 OPD 主线高度相关;其"思维模式一致性""高分≠新知识""token 级 overlap 机制""off-policy cold start"等结论可直接为 TSRD 中 path-selection / path-recovery 设计提供理论依据。

#### 1. 开源代码链接
https://github.com/thunlp/OPD(已克隆,约 195MB,含 verl fork + LlamaFactory)。该仓库的 top-k OPD overlap 诊断指标(`distillation/overlap_ratio`、`distillation/overlap_token_advantage`)已合并入官方 verl(PR #6469)。

#### 2. 使用框架
主体基于 **verl (v0.7.0)** 做 OPD 与 RL(GRPO);**LlamaFactory (v0.9.5)** 做 SFT(cold start)。rollout 用 vLLM,verifier 用 math-verify。评测复用 thunlp/JustRL 的 pipeline。OPD 通过自定义 `ADV_ESTIMATOR=token_reward_direct` 实现,teacher 作为 reward model 提供 token 级 reward。

#### 3. 研究背景
OPD 已成为 LLM 后训练的核心技术(Qwen3、MiMo、GLM-5 均采用),Thinking Machines Lab 也以极低 RL 计算成本复现了 Qwen3 OPD recipe。与离策略蒸馏(在固定 teacher 序列上训练、有 exposure bias)不同,OPD 让学生自采样 rollout,用 teacher 的逐 token log-prob 作为 dense reward,在学生实际访问的状态上修正其行为。但 OPD 的训练动力学仍缺乏理解。

#### 4. 当前存在的问题
存在显著失败模式:更强的 teacher 反而可能完全无法提升学生,而更弱的 teacher 却能成功。少有研究解释 teacher 的 token 级信号为何/何时能把学生分布推向期望方向,以及失败的条件。

#### 5. Motivation
系统性刻画 OPD 的成败条件(现象学)→ token 级机制 → 可落地的修复 recipe,并揭示 dense 监督的代价(长程/agentic 设定下的可扩展性)。

#### 6. 主要方法
分三层递进:
- **现象学(§3)**:提出两条支配 OPD 成败的经验条件——(i) 思维模式一致性(师生 top-k 分布的 overlap ratio 要高,即使 teacher 分数更高,模式不匹配导致初始 overlap 低则训练无法挽回);(ii) 高分≠新知识(若师生用同数据/recipe 训练会收敛到同尺度的相似分布,teacher 缺少可迁移信号;只有 teacher 携带学生未见过的知识时 OPD 才有大增益)。通过 weak-to-strong 反向蒸馏验证:同族 1.5B 与 7B teacher 从学生视角分布上"不可区分",证明 OPD 本质学的是思维模式而非分数。
- **机制(§4)**:定义 Overlap Ratio、Overlap-Token Advantage、Entropy / Entropy Gap 等动态指标。成功 OPD 的签名是 student-visited states 上分布渐进对齐:高概率 token 的 overlap ratio 从约 72% 升到 91%,熵差收窄,共享 top-k token 集中了 97%–99% 的概率质量。失败 run 则 overlap 停滞、熵差持续。进一步证明仅用 overlap token 监督即可匹配 full top-k 性能,说明 overlap 集是 OPD 梯度信号的主要来源。给出三种监督粒度的统一刻画:sampled-token OPD(单样本无偏估计逐 token reverse KL)、full-vocabulary OPD、top-k OPD(在学生 top-k 子集上重归一化后算子集 KL)。
- **Recipe(§5)**:两个互补修复策略——(i) **off-policy cold start**:OPD 前先在 teacher 生成的 rollout 上做 SFT warmup,抬高初始 overlap ratio;(ii) **teacher-aligned prompt selection**:用取自 teacher 后训练数据的 prompt 锐化高概率 token 对齐,但会显著降低学生熵,需要混入 OOD prompt。两者恢复的 run 都呈现与天然成功 run 相同的动态签名。
- **代价(§6)**:reward 质量随轨迹深度系统性退化,不稳定起于靠后 token 并向前传播;即便失败 teacher 其 reward 仍与 rollout 正确性全局相关——说明失败不是信号质量问题,而是局部优化几何(更大 teacher 在学生策略附近诱导出局部平坦的 reward landscape),揭示监督密度与监督可靠性的根本张力,指向当前 OPD 在长程推理/agentic 上的局限。

#### 7. 实验数据集
- 训练:DAPO-Math-17K(主)、DeepMath-103K(对比/cold-start 互补 prompt,需对 DAPO-Math-17K 去重)、OpenThoughts3-1.2M 的 math 子集(用于 teacher rollout → 学生 SFT cold start,发布为 OpenThought3-Qwen3-4B 数据集)。
- 评测:AIME 2024、AIME 2025、AMC 2023(math 竞赛级,avg@16)。
- 模型:学生 Qwen3-1.7B / DeepSeek-Distill-1.5B(DS-1.5B);teacher 含 Qwen3-4B(Non-thinking)、Qwen3-4B-Math、DS-7B、JustRL-1.5B(=对 DS-1.5B 做 RL)、Skywork-OR1-Math-7B(SW-7B=对 DS-7B 做 RL)。发布 Qwen3-1.7B-SFT、Qwen3-4B-Base-GRPO 等 checkpoint。

#### 8. 怎么做的(训练/数据/流程)
- OPD:`bash on_policy_distillation.sh`,`ADV_ESTIMATOR=token_reward_direct`,teacher 作为 REWARD_MODEL 提供 token 级 reward;关键超参 N_RESPONSES=4、MAX_RESP_LENGTH=7168、`LOG_PROB_TOP_K=16`(置 0 退化为 sampled-token OPD)、`TOP_K_STRATEGY=only_stu`(可选 only_tch/intersection/union/union-intersection)、`REWARD_WEIGHT_MODE=student_p`(可选 teacher_p/none)。
- SFT(cold start):用 `scripts/infer/vllm_rollout.py` 让 teacher(如 Qwen3-4B Non-thinking)对 OpenThoughts3 math prompt 做带 rejection sampling 的 rollout,再用 LlamaFactory 对学生(如 Qwen3-1.7B-Base)做 full SFT。
- RL 对照:GRPO,设 `ADV_ESTIMATOR=grpo` 且 `LOG_PROB_TOP_K=0`(`grpo.sh`)。
- 评测:复用 JustRL 的 gen_vllm.py + grade.py(可开 LLM verifier)。
- 硬件:8×NVIDIA A800 80GB。
- 注意:作者指出 verl v0.7.0 内置 validation 会低估性能 5–7 个百分点,建议 `test_freq=-1` 关闭在训验证、`MAX_VAL_RESP_LENGTH=MAX_RESP_LENGTH`,改用 scripts/val/ 单独评测(v0.8.0 已修)。


---


### rock_tokens — Cornerstones or Stumbling Blocks? Deciphering the Rock Tokens in On-Policy Distillation

> **一句话重点 (TL;DR)**：OPD 训练表观饱和后仍有约 6% 词表、占输出 18% 频次的 token 持续高 KL loss（"Rock Tokens"，多为结构/话语脚手架），它们贡献了不成比例的梯度但对实际推理性能功能贡献可忽略；从训练起冻结其梯度可在不掉点下精简对齐（约 1.4× 加速）。

**元信息**：arXiv 2605.09253（v2, 2026-05-29）｜ UMBC + Case Western Reserve + Arizona State + VU Amsterdam（Yuxuan Jiang、Runchao Li、Shubhashis Roy Dipta 共同一作；Dawei Li；通讯 Zhao Yang）｜ Preprint, 2026-05 ｜ 主题 OPD token 级动力学 / T1(Med)（与本项目 token 级监督/梯度分配、path-selection 分析互补）｜ 代码 https://github.com/YuxuanJiang1/Rock-Token（已克隆 ~7.4MB，三目录 `KDFlow_localopd`/`rock_detection`/`stumbling`）｜ 框架 自定义 KDFlow（SGLang+Ray+FSDP2+bf16）

#### 1. 相关工作与进展
On-Policy Distillation (OPD) 已成现代 LLM 后训练基石（DeepSeek-V4、MiMo、Qwen-3 等在 SFT/RLVR 之外用它进一步榨取推理性能）。RLVR 侧研究已揭示 token 非等价：少数 critical/高熵"forking tokens"不成比例地驱动推理增益。但 OPD 的 token 级理解仍少被探索——尽管 OPD 本质依赖 dense 全 token 监督。

#### 2. 现有工作存在的问题
OPD 的 per-token KL loss 中，high-loss token 是师生失配最直接的信号，按既有理解应随训练收敛而减少。作者实证发现相反现象：即便训练到表观饱和，仍有一批 token 持续高 loss（Rock Tokens），约占词表 **6%**、却占输出 token 频次的多达 **18%**。两大悖论：(1) 因高频，它们贡献了不成比例大份额的总梯度范数，自身却在训练中停滞、抵抗 teacher 修正；(2) 因果干预（token knock-out）显示其对实际推理性能功能贡献可忽略。大量优化带宽花在学生学不会也不必学的结构/话语残差上。

#### 3. Motivation
回答标题之问：Rock Tokens 究竟是策略对齐不可或缺的"基石（Pillars）"还是制造冗余的"绊脚石（Stumbling Blocks）"，并据此挑战 OPD 均匀 token 加权的必要性，给出更高效的蒸馏范式。

#### 4. 主要灵感 / 核心直觉
- 借鉴 RLVR 的"token 非等价"思路反向追问 OPD：高 loss ≠ 高价值。
- **path dependency 假设**：学生对结构脚手架 token 形成强解码依赖以维持推理流，从而顽固抵抗 teacher 修正——这是一种内部优化偏置，而非缺乏建设性学习信号。
- 用 token knock-out 把"梯度成本"与"功能贡献"解耦验证。

#### 5. 主要解决思路(一段话讲清核心)
分 What / Why / How 三阶段拆解 Rock Tokens：先用 Rock Score 在 MATH-500 轨迹上识别它们的身份（What），再用 token knock-out + path dependency 解释其持久性来源（Why），最后从训练一开始就冻结这些 token 的梯度（gradient sparsification）来隔离其真实功能必要性并实现加速（How）。围绕三个研究问题 RQ1（训练中是否仍提供有用信号）、RQ2（持久性是否源于学生 path dependency）、RQ3（从一开始排除会怎样）。

#### 6. 方法详解(通俗、分步骤)
- **Rock Score 识别（What）**：per-token KL b_{ℓv} = E[D_KL(πθ‖πT) | x_t=v]（用学生 rollout 估计），在 N=500 MATH-500 轨迹上计算。结合 KL 覆盖（蓝）与跨样本选择稳定性（红）的交点定 **top-K=100**——该边界覆盖约 60% 的语料级 KL 蒸馏负担，且在 n∈[50,400] 下稳定（更大 cutoff 反而退化）。识别出的 Rock Tokens 主要是句法/结构脚手架：格式分隔符、空白符号、高频话语标记（如 "So"、"Wait"）；rare token 则是噪声主导。
- **机制检验（Why）**：用 token knock-out（推理时屏蔽该 token）测功能贡献，按 Δaccuracy 把 token 分为 **Pillar / Neutral / Stumbling**（|Δ|<ε 为 Neutral）。结果绝大多数落在 Neutral（如 MATH-500：7 Pillar / 0 Stumbling / 193 Neutral；IFEval：3 Pillar / 0 Stumbling / 197 Neutral），证明它们既非不可或缺也非有害，而是"冗余"。结论：持久性源于学生主动保护这些 token 以维持推理流的 path dependency。
- **利用（How）**：从训练起冻结这些高成本 token 的梯度（gradient sparsification）。对 30% 高成本 token 做 freeze-weighting 取得 **1.4× wall-clock 加速**且性能持平（performance parity）；对照 Random 冻结则 ΔKL 呈对称噪声、无净变化。

#### 7. 实验数据集
- 训练（两阶段蒸馏）：Stage 1 离策略用 **OpenThoughts3 的 20k** teacher 生成解；Stage 2 在线策略从另外 **10k** prompt（位置 20k–30k 切片）采样、分 7 个 checkpoint。
- 评测（LM-Eval-Harness，zero-shot，Pass@1 取 5 次独立运行平均）：竞赛数学 **AIME 24 / AIME 25 / HMMT 25-Feb**（来自 MathArena，各 30 题、合 90 题为主指标）；扩展到 MATH-500、IFEval 以获大样本 token 级统计。
- 模型：teacher = **Qwen3-30B-A3B-Instruct-2507**（MoE，3B active）；student = **Qwen3-4B-Instruct-2507**；师生均关闭 thinking 模式。

#### 8. 实验结果与主要发现
- **What**：Rock Score 能从频率-KL 平面上分离出真正的 Rock Tokens（高频高 KL），K=100 是覆盖/稳定的最佳交点（≈60% KL 负担）。
- **Why**：knockout 显示 Rock Tokens 几乎全为 Neutral（极少 Pillar、零 Stumbling），说明它们对推理正确性无关键贡献；学生对其的固守是 path dependency 而非有用信号。
- **How (RQ3)**：从训练起冻结其梯度，ΔKL 分布精确集中于零（Panel d），性能与原 OPD 持平（Fig.5：Original OPD ≈ Ours，Random 控制无效），并获 1.4× 加速。

#### 9. 结果如何支撑其主张
"基石 vs 绊脚石"之问由 knockout 的 Pillar/Stumbling/Neutral 分类直接回答（绝大多数 Neutral → 既非基石亦非绊脚石，是冗余）。"挑战均匀加权"由从训练起冻结仍保持 performance parity + 1.4× 加速支撑，逻辑闭合。Random 对照排除了"冻结任意 token 都无害"的平凡解释。

#### 10. 逻辑自洽性(中性评估)
三阶段 What→Why→How 因果链清晰；Rock Score 的选择稳定性分析与 knockout 因果证据、freeze 实验三者相互印证。Neutral 主导的结论与"冻结不掉点"自洽。论证为"诊断—解释—利用"标准范式，结论稳健。

#### 11. 残留问题 / 局限
- 仅在单一师生对（Qwen3-30B-A3B → Qwen3-4B）、关闭 thinking 模式下验证；对开启 thinking、不同 tokenizer/词表、更大跨度师生的泛化未验证。
- knockout 是推理时屏蔽，与"训练时冻结梯度"是两种不同干预；"功能贡献可忽略"主要基于数学/IFEval 任务的准确率指标，对更依赖格式/话语连贯的任务（长文档、对话）未必成立。
- Rock 检测依赖 MATH-500 轨迹与最终 checkpoint 的 KL，跨任务 Rock 集合的迁移性〔待核〕。
- 1.4× 加速来自对 30% 高成本 token 的 freeze-weighting，与"top-K=100 Rock Tokens"两个口径需注意区分（前者为成本驱动的更宽集合）。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 链接：https://github.com/YuxuanJiang1/Rock-Token（已克隆 ~7.4MB）。三目录：`KDFlow_localopd`（训练框架）、`rock_detection`（Rock token 检测/logit 计算）、`stumbling`（冻结梯度实验）。
- 框架：自定义 **KDFlow**（songmzhang/KDFlow，arXiv:2603.01875）；后端 **SGLang + Ray + FSDP2 + bf16**（requirements: sglang、sglang_router、ray≥2.0、torch≥2.4、ring_flash_attn、peft；docker SGLang 0.5.9 / torch 2.9.1 / CUDA 12.8）；评测 LM-Evaluation-Harness。KDFlow_localopd README 致谢明确：模型封装/分布式抽象沿用 **OpenRLHF**（`model.py` 注 "modified from OpenRLHF/openrlhf/models/actor.py"），on-policy KD 的 Ray placement-group 初始化与 SGLang 权重更新借鉴 **slime**（多处 "following slime" 注释），rollout 用 SGLang。即 OpenRLHF(训练抽象)+slime(on-policy KD)+SGLang(rollout)。
- 可得性：训练/检测/冻结三套实验代码齐备；硬件 4×H100(80GB)，FSDP2+bf16+gradient checkpointing，超参沿用 KDFlow 默认（Appendix C）。


---


### rosd — ROSD: Reflective On-Policy Self-Distillation for Language Model Reasoning across Domains

> **一句话重点 (TL;DR)**：标准在线策略自蒸馏（OPSD/SDPO）把自教师条件于"完整正确解"会让学生模仿训练域参考轨迹、损害 OOD；ROSD 改为"纠错思路 e + 错误引语 q"引导的**错误后缀局部蒸馏**——只修第一处出错起的后缀、保留有效前缀，从而在保住域内的同时大幅改善跨域泛化（4B 上 OOD 平均 41.31% vs SDPO 2.88%）。

**元信息**：arXiv 2605.28014（v1, 2026-05-27）｜ 香港理工 / 百度 / 山东大学 / 莱顿大学（Ziqi Zhao、Xinyu Ma、Liu Yang、Yujie Feng、Daiting Shi、Jingzhou He、Xin Xin、Zhaochun Ren、Xiao-Ming Wu）｜ arXiv preprint ｜ 主题 在线策略自蒸馏（OPSD）改进 / 与本项目 mtp_opd（MTP+OPD、TSRD）高度相关——"反思引导 + 错误定位"正对应"教路径选择/路径修复"｜ 代码 https://github.com/ZiqiZhao1/ROSD（Apache-2.0，已克隆 ~53MB，含完整 verl + 实验脚本 + reward 函数）｜ 框架 verl（扩展自 SDPO/lasgroup/SDPO）+ FSDP actor + vLLM rollout

#### 1. 相关工作与进展
后训练对提升 LLM 推理至关重要。SFT 类依赖数据集提供的 golden solution；RL 类（RLVR）只靠 verifier 信号。典型 RLVR（GRPO）用 outcome reward 算 response 级 advantage，无 token 级监督。OPSD 用"自教师"（self-teacher，与学生同权重）在 token 级提供稠密监督弥补：给学生 on-policy rollout，让自教师条件于一个正确解，再对学生/教师分布算 token 级 KL（SDPO 是其 RL 实例，用成功 rollout / 环境反馈作教师上下文）。本工作的 GRPO 基线已用强化实现（含非对称 clip、无偏归一化、off-policy 校正）。

#### 2. 现有工作存在的问题
标准 OPSD 即使域内也不稳健：域内性能训练早期升、后期不稳甚至下降；域外（OOD）快速恶化。两点归因：(1) 把自教师条件于"已验证正确解"会鼓励学生模仿训练域参考轨迹（而非针对自身错误修正），内化域特定模式；(2) 对整条 response 做蒸馏会覆盖已正确的推理前缀、惩罚有效的替代前缀，并把训练域偏好注入正确推理，损害 OOD 泛化（自教师还会引入"according to the reference solution"等参考依赖措辞）。

#### 3. Motivation
错误通常是局部的——应只修正"第一处出错"起的后缀，并用"纠错思路"而非完整参考解引导教师，把"参考解模仿"转为"针对性推理纠错"，在修复错误同时保留有效前缀，提升域内并显著改善 OOD。

#### 4. 主要灵感 / 核心直觉
- **错误局部性**：有效前缀（绿）+ 错误后缀（红）的二分；只需修后缀。
- **特权信息 vs 域偏好分离**：自教师的增益应来自"它能看到正确解这一特权信息"，而非把训练域风格灌进学生；故只把"纠错关键思路 e"作上下文、用"错误引语 q"做定位，避免全解模仿。
- 学生/自教师/自反思器共享同一 base，性能提升纯来自特权信息条件，无需更大外部教师、无额外模型存储。

#### 5. 主要解决思路(一段话讲清核心)
对每题采 G 条 rollout，分正确集 Y+ / 错误集 Y-。对每条错误 rollout y-，配同组**最短**的正确 rollout y*，让自反思器输出 `<error_quote>`（q，精确标出第一处出错子串）与 `<explanation>`（e，解释错因与如何修）；把 e 作 auxiliary context 注入自教师 prompt，用 q 在 y- 中定位 k=Locate(q,y-)，仅对 t≥k 的后缀做 token 级蒸馏（匹配失败回退全响应 k=0；正确 rollout 仅反思"为何有效"得 e、全 token mt=1）。教师分布取 stopgrad。

#### 6. 方法详解(通俗、分步骤)
- **错误聚焦自反思（Error-Focused Self-Reflection）**：错误 rollout 与同组最短正确 rollout 配对（最短以减少冗余）；反思 prompt（Fig.2）要求严格输出 `<error_quote>`（从错误解中抽的精确子串）+ `<explanation>`（错因+修法+正确逻辑）。正确 rollout 只单独反思得 e（"为何有效"），不直接全解模仿。自反思器复用自教师权重。
- **引语定位自蒸馏（Quote-Localized Self-Distillation）**：掩码 m_t = 0 (t<k) / 1 (t≥k)，k=Locate(q,y-)；目标对掩码后 token 算师生散度，π_teacher(·|x,e,y<t) 取 stopgrad。
- **散度实现**：继承 SDPO 的 token 级 KL（Eq.3）；代码中由 `alpha` 配置——alpha=1 为 reverse KL（SDPO 默认）、alpha=0 为 forward KL、中间为 **Generalized Jensen-Shannon Divergence**（`torch.lerp(kl_student, kl_teacher, alpha)`，core_algos.py:1163）。〔修正：旧分析称"采用 JSD"——准确说 JSD 是可配置的一般情形，reverse KL 是其特例/继承自 SDPO 的基线散度。〕
- **训练设定**：标准 RLVR（无人工 golden、仅 verifier）；学生/自教师/自反思器同 base，提升源自"教师条件于特权信息"。

#### 7. 实验数据集
- 训练分别在 **5 个数据集**上单独进行：SciKnowEval(L3) 推理子集的化学/物理/生物/材料 4 个本科级科学 QA + ToolAlpaca 的工具调用数据集（ToolUse，把 API 规范+用户请求映射到正确 tool call）。
- 额外用 **AIME2024** 评数学推理。每个数据集划分训练/测试，"训一域、评所有域"以同时考察域内与 OOD。
- 训练每 prompt 采 **8 条 on-policy rollout**；评估每题采 **16 条报 mean@16**（取训练中最大 mean@16）。
- Backbone：**Qwen3-4B、Qwen3-8B**。基线：强化版 GRPO（非对称 clip + 无偏归一化 + off-policy 校正）、SDPO。

#### 8. 实验结果与主要发现
- 域内（Table 2）：Qwen3-4B 平均 **72.83%**（较 GRPO/SDPO **+2.97/+5.81** 分）；Qwen3-8B **73.45%**（较 GRPO/SDPO +1.46/+0.95 分，8B 上增益较小）。
- OOD：ROSD 在所有训练域、两个规模上均稳定超过 SDPO；跨域差异大时（如 ToolUse→科学QA/数学）SDPO 严重退化——4B 上 SDPO OOD 平均跌到 **2.88%**，ROSD 仍维持 **41.31%**。
- 主要发现：把"全解模仿"换成"错误后缀局部纠错"是稳住域内 + 救回 OOD 的关键；自教师的增益来自特权信息而非更大模型。

#### 9. 结果如何支撑其主张
"标准 OPSD 损害 OOD"由 pilot study（SDPO 域外恶化）支撑；"局部纠错改善泛化"由 ROSD 在所有 OOD 设置稳超 SDPO、尤其极端跨域（2.88%→41.31%）支撑，逻辑直接。域内 +2.97/+5.81 支撑"不牺牲域内"。8B 上增益缩小也被如实报告，未夸大。

#### 10. 逻辑自洽性(中性评估)
"错误局部 → 只修后缀 + 用纠错思路而非全解"的因果链与两点归因（参考解模仿、覆盖有效前缀）一一对应。掩码机制（k=Locate(q)）直接实现该思路；匹配失败回退全响应是合理兜底。共享 base 的设计排除了"更大教师"混淆变量，自洽性强。

#### 11. 残留问题 / 局限
- q 的定位依赖自反思器抽出"精确子串"并能在 y- 中匹配；匹配失败回退全响应，失败率/对结果的影响〔待核：论文未报回退比例〕。
- 8B 上相对 SDPO 增益已很小（+0.95），方法收益随基座变强而衰减，规模化外推存疑。
- "第一处出错"假设单点错误；多处分散错误或错误前缀本身无效时，"保留前缀"可能保留了错误前提。
- 评测域偏窄（科学 QA + 单一工具集 + AIME），缺代码/对话等异构域；自反思器质量本身依赖同一 base 的能力，弱基座下反思可能不可靠。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 链接：https://github.com/ZiqiZhao1/ROSD（Apache-2.0，已克隆 ~53MB）。
- 框架：基于 **verl**（verl-project/verl）扩展自 **SDPO**（lasgroup/SDPO）；FSDP actor 训练 + vLLM rollout，8×NVIDIA A800 80G。
- 关键文件：`verl/trainer/ppo/ray_trainer.py`（rollout/reward/reflection/distillation batch 构建）、`verl/workers/actor/dp_actor.py`（自蒸馏 loss 分发）、`verl/trainer/ppo/core_algos.py`（`compute_self_distillation_loss`，alpha 配置 KL/JSD，line 1085+）、`verl/trainer/config/sdpo.yaml`（Hydra 配置）。主入口 `experiments/local/run_sdpo_processreflection_ood_all.sh`。
- 可得性：完整 verl 源码 + 启动脚本 + reward 函数齐备，自蒸馏 loss 与反思/定位逻辑可在代码中核实，可复现性好。


---


### safesteer — SafeSteer: Localized On-Policy Distillation for Efficient Safety Alignment

> **一句话重点 (TL;DR)**：安全特征在输出分布中本就稀疏，对齐应是"局部修改而非全局权衡"；SafeSteer 用 activation steering 造安全 teacher，挑出稀疏安全 token 子集 S，仅在 S 上施加 reverse-KL OPD，仅用 100 条有害样本即在几乎不掉通用能力下显著降 ASR。

**元信息**：arXiv 2606.02530（v1, 2026-06-01, cs.AI；投 EMNLP 2026）｜ 北航 / 北理工 / 北邮 / 北大 / 中科院自动化所 / 上海AI Lab / BAAI（Hao Li*、Jingkun An*、Zijun Song* 共同一作；通讯 Lei Sha）｜ arXiv preprint ｜ 主题 LLM 安全对齐 / 与本项目关系外围（安全方向），但方法层面高度相关——token-localized on-policy distillation + reverse-KL，与"在稀疏/局部 token 上做选择性蒸馏"（cf. TSRD path-token 监督、选择性 KD）同源 ｜ 代码 https://github.com/Anjingkun/SafeSteer（已克隆 ~8.9MB，含 distil_trainer.py / distil_config.py / main.py / model_utils / scripts / data；RepoExists=YES）｜ 框架 自研轻量安全对齐框架（activation steering teacher + token 选择 + localized OPD）

#### 1. 相关工作与进展
安全对齐常以牺牲通用能力为代价（alignment tax）。现有方法多把它当双目标优化：混入大量通用数据、把安全梯度正交投影到通用能力零空间（NSPO）、或训练辅助奖励模型（SafeRLHF/MoCAN）。标准 LLM OPD 又通常需外部更强 teacher；近期自-teacher 方案重度依赖模型 in-context learning 能力。论文借鉴 Arditi et al. (2024) 的 refusal direction（activation steering）思路构造无需更强外部模型的安全 teacher。

#### 2. 现有工作存在的问题
- 双目标 trade-off 依赖海量通用数据或辅助 reward model，成本高。
- 标准 OPD 需外部更强 teacher；自-teacher 方案依赖 ICL 能力。
- 即便有好 teacher，标准 OPD 对**整个词表**施加惩罚，会波及通用能力 token——而安全 token 在输出分布中稀疏、且与通用任务 token 大体不相交，全局惩罚反伤通用能力。

#### 3. Motivation
论点："安全特征本就稀疏，对齐需要的是**局部修改而非全局权衡**"。因此用 activation steering 得稳定安全信号的 teacher，再把 reverse-KL 蒸馏惩罚**限制到稀疏安全 token 子集 S**，在调整安全特征同时缓解遗忘。

#### 4. 主要灵感 / 核心直觉
- **稀疏性 → 局部更新**：安全行为集中在少数 token（拒答触发词等），无须全词表惩罚。
- **activation steering 作 teacher**：注入 refusal direction 即可让模型对任意输入稳定拒答，作为"廉价但稳定"的安全信号源，无需更强外部模型。
- **对比 + 投票挑 token**：不是取单点最大 logit 差，而是用对比 log 概率 + 跨位置/样本投票聚合，得到对 refusal direction 最敏感的稳健稀疏子集。

#### 5. 主要解决思路(一段话讲清核心)
三步：(1) 提取 refusal direction d，在某层 ℓ 用 forward pre-hook 把残差流 h_ℓ 替换为 h_ℓ+d 并在所有 token 位持续注入，得稳定拒答的安全 teacher πt；(2) 在 harmless 指令上用 πt 采拒答轨迹，对每个位置算 token v 的对比 log 概率 Δ=log[pt(v)/p0(v)]（teacher vs base），用投票聚合挑出稀疏安全 token 子集 S；(3) student πs 在有害指令上 on-policy 自采样轨迹，仅对 S 内 token 最小化 D_KL(πs‖πt)（reverse-KL），从而只改安全特征、保留通用能力。

#### 6. 方法详解(通俗、分步骤)
- **安全 teacher 构造**：按 Arditi et al. (2024) 提取 refusal direction d；在层 ℓ 用 forward pre-hook 令 h⋆_ℓ = h_ℓ + d，对所有 token 位持续注入，得 πt——它对**有害与无害输入都一致拒答**（即设计上会 over-refuse，但目的是提供稳定拒答信号）。
- **安全 token 选择**：在 harmless 指令（Alpaca）上用 πt 采 N 条拒答轨迹（长度 H）；每条每步算 token v 的对比 log 概率 Δ=log[pt(v|·)/p0(v|·)]；每位置取 top-K′，再用指示函数跨 harmless 数据、N 条轨迹、各 step 做**投票聚合** vote(v)=Σ 1[v∈C(x,n,j)]，得对 refusal direction 最敏感的稀疏子集 S（比单纯最大 logit 差更稳）。
- **localized OPD 训练**：student πs 在有害指令（PKU-SafeRLHF，仅 100 条）上 on-policy 生成轨迹，仅对 S 内 token 最小化 reverse-KL D_KL(πs‖πt)。

#### 7. 实验数据集
- 安全 benchmark（7 个，报 ASR%↓）：有害查询 AdvBench、PKU-SafeRLHF、HarmBench、JailbreakBench(JBB)、SORRY-Bench；红队查询 HarmfulQA、ALERT。
- 通用能力（5 个，↑）：MMLU(STEM)、AlpacaEval(Win Rate)、GSM8K、MATH、HumanEval。
- 训练数据：harmless=Alpaca（选 token）；harmful=PKU-SafeRLHF，**仅 100 条**（<1% 基线数据量）。
- 模型：Llama-3-8B-Instruct、Llama-3.2-3B-Instruct、Qwen2.5-7B-Instruct、Qwen3-4B-Instruct-2507。
- 判官：默认 **Llama-Guard-4-12B**（另报 LLM-as-judge 校正 regex 误判）；通用能力用 lm-eval。基线：DPO-Mix、MoCAN、W-DOOR、BFPO、NSPO。学习率 Llama 系 1e-6、Qwen 系 1e-5。

#### 8. 实验结果与主要发现
- Qwen3-4B-Instruct（t=0）：ASR 平均 **2.87%→0.91%**（基线 MoCAN 1.13%、NSPO 1.18%），通用 Avg **71.68→71.39**（几乎无损）。
- Qwen2.5-7B、Llama 系亦取得最低/次低 ASR 且通用能力近乎保持；安全–能力权衡上整体优于现有方法，且仅用 100 条有害样本、无通用数据。
- **关键发现（Appendix D.2）**：teacher πt 几乎拒绝所有 harmless 指令（over-refuse），但训练后的 student πs 与 base π0 **均不 over-refuse**——token-localization 成功蒸出安全特征而不吸收 teacher 的 over-refusal；PCA（Fig.3）显示 πs 相对 π0 在通用表征上几乎无 shift。

#### 9. 结果如何支撑其主张
"局部更新而非全局权衡"由 ASR↓ 同时通用 Avg 几乎不变 + PCA 无表征 shift 共同支撑，逻辑直接。"稀疏 token 足够"由仅惩罚 S 即降 ASR 支撑。"不吸收 teacher 缺陷"由 Appendix D.2 的 student 不 over-refuse 实验支撑（这点直接回应了"steering teacher 过度保守"的质疑）。"高效"由 100 条样本 + 无通用数据支撑。

#### 10. 逻辑自洽性(中性评估)
"安全稀疏 → 局部蒸馏"的核心论点—方法—结果三者对齐：token 选择（对比+投票）、localized reverse-KL、over-refusal 不传递的实验环环相扣。用 activation-steering teacher 这一"会 over-refuse 但信号稳定"的设计，与 token-localization 的"只取安全相关子集"互补，自洽性较强。

#### 11. 残留问题 / 局限
- 仅做安全对齐这一窄域，"安全稀疏→局部更新"在更复杂能力对齐（如价值观、长程行为）上的泛化未验证。
- 安全 token 子集 S 离线固定（基于 Alpaca harmless + πt 拒答轨迹），对新型越狱 / 分布外有害模式的覆盖性存疑——S 的静态性是主要风险点。
- 依赖能可靠提取 refusal direction 的模型与层 ℓ 选择；refusal direction 方法本身（Arditi 2024）的局限会传导。
- 判官（Llama-Guard-4-12B）与基线超参对 ASR 结果敏感；SORRY-Bench 上残留 ASR 仍偏高（如 Qwen3-4B 5.91%），说明并非所有有害类别都被压住。
- 〔旧分析"teacher over-refusal 可能只被部分消解"已修正〕：论文 Appendix D.2 明确显示 student 不 over-refuse，over-refusal 未传递；但该结论基于其自定 over-refusal 测试集，覆盖面有限。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 链接：https://github.com/Anjingkun/SafeSteer（已克隆 ~8.9MB；项目页 https://anjingkun.github.io/SafeSteer/）。RepoExists=YES。
- 框架：自研轻量安全对齐框架——`distil_trainer.py`（localized reverse-KL OPD 训练）、`distil_config.py`（配置）、`main.py`（入口）、`model_utils`（steering hook / refusal direction）、`scripts`、`data`。
- 流程：先离线选 token S（harmless 上用 πt 采轨迹 + 对比 log 概率 + 投票）→ 在线 OPD 仅惩罚 S（harmful 上 student on-policy 采样 + reverse-KL）。
- 可得性：训练/配置/入口与数据齐备，方法轻量（无需大规模通用数据、仅 100 条有害样本），复现门槛低。


---


### scope — SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting

> **一句话重点 (TL;DR)**：标准 OPD 把 teacher 的稠密 token-level 监督一视同仁地用在所有 rollout 上,忽视信号质量差异;SCOPE 按轨迹正确性双路路由——正确轨迹用 **student-PPL 加权 MLE 自强化**(放大能力边界处低置信样本),错误轨迹用 **teacher-PPL 加权 KL 蒸馏**(优先 teacher 真有纠错能力即低 PPL 的实例),并在组内做 perplexity 归一化。

**元信息**：arXiv 2604.10688 (v2, 2026-05-30, cs.LG) ｜ 中科大/美团 LongCat/南大/复旦/华科(Binbin Zheng¹²、Xing Ma²、Yiheng Liang²³、Jingqing Ruan²、Xiaoliang Fu⁴、Kepeng Lin⁵、Benchang Zhu²、Ke Zeng²、Xunliang Cai² 通讯;实习期间完成) ｜ arXiv preprint, 2026-04(v2 2026-05) ｜ 主题 T1(On-Policy Distillation),Relevance=Med(双路"正确路强化 student 自身、错误路用 teacher 纠错"与 TSRD path-selection/path-recovery 强相关) ｜ 代码 github.com/machine981/SCOPE(已克隆 ~9.1M;HF 已放 SCOPE-Qwen3-1.7B、SCOPE-Deepseek-R1-Distill-Qwen-1.5B) ｜ 框架 veRL(volcengine/verl)

#### 1. 相关工作与进展
On-policy RL(DeepSeek-R1/GRPO/DAPO 范式)已成 LLM 推理对齐主流,但稀疏 outcome-level 奖励使 token-level credit assignment 困难、收敛慢。**OPD**(On-Policy Distillation)在 student 自采样 rollout 上引入 teacher 的稠密 token-level KL 监督来缓解,兼顾分布一致与训练效率。SCOPE 直接以"标准 OPD 一视同仁"为靶子改进加权。

#### 2. 现有工作存在的问题
现有 OPD 假设 teacher 的 dense 监督在**所有 rollout 上一致可靠**,忽视信号质量根本差异,带来两个问题:(1) **多样性退化**——对正确路径一律等权强化,抑制能力边界处有效但非常规的推理路径,过度强化已掌握样本;(2) **纠错低效**——对错误轨迹,当 teacher 自身也不熟悉(高 teacher PPL)时其 token 分布是不可靠信号(论文称 context 诱导噪声),会误导。

#### 3. Motivation
经验分析(论文 §2 PSR/recovery 实验,从 DeepMath 采 2,000 题用 student 生成轨迹、用 teacher 计算 PPL 分桶):**错误轨迹上低 teacher PPL 强相关于成功 error recovery**(低 PPL=teacher 有真纠错把握,可作"真正纠错能力"代理);**正确轨迹上应按 student PPL 自适应加权**,把强化集中到能力边界的低置信(高 student PPL)样本而非已掌握样本。故应按 rollout 正确性路由到不同监督路径并各自加权。

#### 4. 主要灵感 / 核心直觉
信号质量可由 PPL 探针预测:teacher PPL 衡量"teacher 在此轨迹上是否可信",student PPL 衡量"该样本是否处于 student 能力边界"。把"按正确性分路 + 按 PPL 自适应加权"组合成单一目标。

#### 5. 主要解决思路(一段话讲清核心)
双路自适应框架 DPAW:按正确性把 on-policy rollout 路由到两条互补监督路径——**Student Path(正确轨迹 Ω_c)** 做 student-PPL 加权 MLE 自强化,**Teacher Path(错误轨迹 Ω_w)** 做 teacher-PPL 加权 KL 蒸馏;两路均在同 prompt 轨迹组内做 perplexity 归一化(group-level softmax),以应对 prompt 间难度方差。总目标 L_SCOPE = Σ_{i∈Ω_c} w_i^stu·L_MLE + Σ_{i∈Ω_w} w_i^tea·L_OPD。

#### 6. 方法详解(通俗、分步骤)
- **Student Path(Eq.4)**:w_i^stu = softmax_{j∈Ω_c}(−1/(τ|y_i|)·logπ_S(y_i|x)) = PPL_S(y_i|x)^{1/τ} 归一化 ⇒ **student PPL 越高权重越高**,放大能力边界非常规有效路径。
- **Teacher Path(Eq.5)**:w_i^tea = softmax_{j∈Ω_w}(+1/(τ|y_i|)·logπ_T(y_i|x)) = PPL_T(y_i|x)^{−1/τ} 归一化 ⇒ **teacher PPL 越高权重越低**,过滤 teacher 不可信(高 PPL)的错误轨迹噪声。
- **组内归一化**:同一 prompt 的正确组、错误组分别做 softmax,权重乘以组大小(均值≈1),自适应校准。
- **流程**:① `pip install -r requirements.txt` + `pip install -e .`(装 verl);② `bash deploy_vllm.sh` 部署 teacher(默认 Skywork-OR1-7B,served_model_name/api-key 须与 `verl/utils/api_interface.py` 一致,支持多节点 IP_POOL);③ 在 `run_experiment_distill_1_5b.sh` 设 TEACHER_MODEL_NAME、IP_POOL、POLICY_MODEL_PATH;④ 训练。双路开关在 `verl/trainer/ppo/ray_trainer.py:_compute_scope_dual_path_weights`:USE_SCOPE_DUAL_PATH_WEIGHTING=True、SCOPE_TAU=1、SCOPE_USE_SEQ_WEIGHTS=True、USE_STUDENT/TEACHER_PATH_WEIGHTS=True。

#### 7. 实验数据集
- 两组 teacher–student(论文 §4.1):主配置 teacher=**Skywork-OR1-7B**(脚本/论文均写 `Skywork-OR1-7B`;**README 结果标题误写 `Skywork-OR1-Math-7B`,不一致**)→ student=DeepSeek-R1-Distill-Qwen-1.5B;第二配置 teacher=Qwen3-8B-Instruct → student=Qwen3-1.7B-Base。均在 **DeepMath**(DeepMath-103k)训练。
- 评测 6 数学 benchmark:AIME24、AIME25、AMC23、MATH500、Minerva、OlympiadBench,报 Avg@32 / Pass@32(rollout temp=0.6、top-p=0.95、max_response=32,768)。
- 扩展:3 代码 benchmark(HumanEval、Codeforces、LiveCodeBench)证跨可验证任务普适。

#### 8. 实验结果与主要发现
- 摘要主口径:较竞争基线(GRPO/KD/OPD 取平均)平均相对提升 **Avg@32 +11.42%、Pass@32 +7.30%**。
- 仅相对标准 OPD(主配置 1.5B,Table 1):平均 Avg@32 **+5.54%**、Pass@32 **+2.60%**(逐项如 +0.90/+0.20、+8.31/+3.96、+10.69/+2.31 等)。
- 〔待核〕6 benchmark 各自精确 baseline 数值见 Table 1(如主配置 SCOPE Avg@32 AIME24 42.7 / Olympiad 49.7),τ 消融见 Table 6,§2 的 PSR/recovery 经验图具体数据未细抄。

#### 9. 结果如何支撑其主张
"按信号质量加权优于一视同仁"由相对标准 OPD 的 +5.54%/+2.60% 直接支撑;"PPL 可预测信号质量"由 §2 的 recovery 分桶相关性支撑;"普适性"由代码任务扩展支撑。支撑成立,但相对 OPD 的净增益(尤其 Pass@32 +2.60%)偏小,且未报多 seed 方差,统计稳健性存疑。

#### 10. 逻辑自洽性(中性评估)
方法公式(Eq.4/5)与"放大边界/过滤不可信"叙事自洽。但需中性看待:(1) student-PPL 高=能力边界 与 student-PPL 高=该轨迹只是噪声/错得自信,二者难区分,Student Path 仅作用于**正确**轨迹一定程度缓解但未完全排除;(2) teacher PPL 作"纠错能力"代理是经验相关而非因果,论文未给反事实验证;(3) 净增益较小且无方差报告。**最关键:代码实现的加权方向与论文公式存在系统性不一致(见 §11),若按发布脚本运行,Student Path 实际加权方向与 Eq.4 相反。**

#### 11. 残留问题 / 局限
- 相对标准 OPD 增益小、无多 seed 方差;τ、归一化粒度的鲁棒性仅部分消融。
- 经验代理(PPL→信号质量)缺因果证据。
- **〔代码-论文差异,本轮重新严格推导,修正了上一轮结论〕** `_compute_scope_dual_path_weights` 用 `logits = sign·mean_logp/τ` 做 softmax,其中 `sign = +1 if ppl_positive else −1`,而 PPL=exp(−mean_logp):
  - 经数值验证:**`ppl_positive=True` ⇒ 高 PPL 得低权重;`ppl_positive=False` ⇒ 高 PPL 得高权重**(代码注释 line 214-215 把二者写反,注释本身有误)。
  - 论文要求:Student 高 PPL→高权重(Eq.4,需 `ppl_positive=False`);Teacher 高 PPL→低权重(Eq.5,需 `ppl_positive=True`)。
  - 两个 SCOPE 训练脚本均设 `STUDENT_PATH_PPL_POSITIVE=True`、`TEACHER_PATH_PPL_POSITIVE=True`(distill_1_5b.sh:40-41、qwen_1_7b.sh:43-44)。据上推导:**Teacher Path(=True)实际与论文 Eq.5 一致(正确)**;而 **Student Path(=True)与论文 Eq.4 相反(错误)——高 student PPL 反被降权,与"放大能力边界样本"主张相悖**。
  - 这与上一轮分析的结论**相反**(上一轮认为 teacher=True 错、应改 False;实为 student=True 错、应改 False)。根因是上一轮误读了代码注释而未做 softmax 方向验证。函数默认值 `student=True, teacher=False`:据上推导 default 下 student 仍错、teacher 也与 Eq.5 相反——即**纯默认值两路都偏离论文**;唯有脚本里 teacher=True 把 teacher 路救回正确。复现时应将 **student_path_ppl_positive 改为 False** 才符合论文。〔待核:发布的 HF 权重究竟按哪组 sign 训练,无法从仓库静态确认;建议跑一次权重对齐实验定性。〕
- **〔代码-论文差异〕** 训练长度:论文 §4.1/Table 写 max_prompt=**4,096**;但脚本 `run_experiment_distill_1_5b.sh:16`、`_qwen_1_7b.sh` 实设 `MAX_PROMPT_LENGTH=2048`(completion=12,288 两者一致)。
- 上一轮称仓库含 `verl-distillation-ori` 定制分支——**本轮核查不存在该目录**,蒸馏逻辑直接整合进仓库内 `verl/`(已更正)。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库:https://github.com/machine981/SCOPE(已 clone ~9.1M);权重 HF `Machine981/SCOPE-Qwen3-1.7B`、`Machine981/SCOPE-Deepseek-R1-Distill-Qwen-1.5B`。
- 框架:**veRL(volcengine/verl)**,蒸馏逻辑整合在仓库内 `verl/`(非独立分支)。安装 `pip install -r requirements.txt` + `pip install -e .`。teacher 经 vLLM(`deploy_vllm.sh`)以 OpenAI 兼容 API 提供,verl 侧经 `verl/utils/api_interface.py` 调用(api-key 须一致),支持多节点 IP_POOL。`recipe/` 含 dapo/drgrpo/prime/r1/sppo 等参考配方。
- 入口脚本:`run_experiment_distill_1_5b.sh`(SCOPE 主)、`run_experiment_qwen_1_7b.sh`;GRPO 基线脚本(`*_grpo.sh`,teacher=False 且不启用 dual-path)。
- 代码可得性:Tier A(完整可跑),但上述加权方向 bug 需复现者警惕。


---


### sdcl — Self-Distillation Enables Continual Learning (SDFT)

> **一句话重点 (TL;DR)**：把"示范条件化的同一模型(EMA)"当自己的 teacher,在 student(只看 query)的 on-policy rollout 上做 token 级 KL 蒸馏,把 off-policy 的专家示范转成 on-policy 学习信号,从而在学新技能/新知识时显著少遗忘——是 privileged-context self-distillation 家族用于持续学习的实例。

**元信息**：arXiv 2601.19897 (v1, 2026-01-27, cs.LG) ｜ MIT + Improbable AI Lab + ETH Zurich(Idan Shenfeld、Mehul Damani、Jonas Hübotter、Pulkit Agrawal) ｜ arXiv 预印本 ｜ 主题 T1(On-Policy Distillation / 持续学习),与本项目 OPD 主线高度同源(privileged 信息=专家示范,与 copsd/opsa 同族) ｜ 代码 github.com/Continual-Intelligence/Self-Distillation(README 现给该 clone URL;主页 idanshenfeld.com/SDFT) ｜ 框架 TRL 0.24.0

#### 1. 相关工作与进展
基础模型部署后静态、难增量学。已有路线:on-policy RL 可减遗忘但需显式 reward(常不可得);从示范学习的主流是 **SFT**(本质 off-policy);IRL(先学 reward 再 on-policy RL)不 scale。本文方法与 GKD、context-distillation/RL-from-distilling-context 同源——核心 trick(条件化 teacher + on-policy 蒸馏)此前已出现,贡献在"用于 continual learning 防遗忘"的定位与系统实验。

#### 2. 现有工作存在的问题
- SFT off-policy → 顺序学新任务时旧能力崩(灾难性遗忘)。
- RL 需要 reward,纯示范学习场景没有。
- IRL 类(先学 reward 再 RL)不 scale。

#### 3. Motivation
关键观察(图 2 右):把模型 **condition 在专家示范上**得到的 teacher 分布,比直接 SFT 目标分布更接近 base 模型分布——而越接近预训练分布的更新遗忘越少。于是用"示范条件化的同一模型"作 teacher,在 student(只看 query)的 on-policy rollout 上蒸馏,把 off-policy 示范转成 on-policy 信号。

#### 4. 主要灵感 / 核心直觉
ICL 把示范"软化"进分布,得到一个既懂新任务、又仍贴近预训练分布的 teacher;在 student 自采样轨迹上对齐到该 teacher,既学到新技能又不偏离原能力。

#### 5. 主要解决思路(一段话讲清核心)
对每个 query x:student P=π_θ(·|x);teacher Q=π(·|x, c),c 为用固定 in-context 模板注入的专家示范(作"一个示例",避免逐字复制)。next token 从 P 采样(on-policy),最小化 student 与 teacher 的 token 级 KL。teacher 权重默认取 student 参数的 **EMA**(§3/§4.6 消融,非字面"同一当前模型")。论文形式化论证其等价于 on-policy RL,隐式 reward r(y,x,c)=logπ(y|x,c) − logπ_k(y|x)(§3.1 "Self-Distillation as Inverse RL",ICL 假设 π*_{k+1}≈π(·|x,c))。

#### 6. 方法详解(通俗、分步骤)
1. student 只看 query 生成 on-policy rollout;
2. 同一模型(EMA)condition 在 query + 示范上得 teacher 分布;
3. 在 rollout token 上做 token 级 KL,更新 student。
4. **KL 方向(重要修正,见 §10/§11)**:论文正文写**反向 KL(reverse KL)**;但仓库 README(04/07/26 errata)声明**论文全部结果实为 forward KL(GKD 式) per-token 损失**,且仓库默认即 forward KL。配置 `distil_config.py`:`alpha`(0=forward KL〔默认〕、1=reverse KL、中间=JSD)、`beta`(KL 系数,默认 0 即不载 reference model)。`main.py` 默认 base = `Qwen/Qwen2.5-7B-Instruct`,默认 1 epoch、lr 2e-5,**未暴露 alpha CLI 参数 → 走 config 默认 forward KL**。

#### 7. 实验数据集
- **Skill Learning(3 域)**:Science Q&A(SciKnowEval Chemistry L-3)、Tool Use(ToolAlpaca)、Medical(HuatuoGPT-o1 stage-1 训练 / stage-2 评测)。
- **Knowledge Acquisition**:2025 年自然灾害 Wikipedia 语料(超模型 cutoff)生成 QA;额外测 OOD"间接问题"考察知识是否真内化;全文超 context window 时与 oracle-retriever RAG 对照。
- **遗忘评测**:HellaSwag/TruthfulQA/MMLU/IFEval/Winogrande/HumanEval 均值。
- **顺序学三技能**:Tooluse→Science→Medical。
- 主实验 base = Qwen2.5-7B-Instruct;scaling 用 Qwen2.5 家族 3B/7B/14B;§4.5 reasoning 用 Olmo-3-7B-Think。

#### 8. 实验结果与主要发现
- **Knowledge Acquisition(Table 1)**:SDFT strict **89** / lenient **100** / OOD **98**,远超 SFT(80/95/80),逼近 Oracle RAG(91/100/100);CPT 仅 9/37/7。OOD 间接问题上优势更明显,说明 SDFT 真正内化而非记忆。
- **Skill Learning**:SDFT 在 new-task accuracy 与 prior-task 保持上同时优于 SFT/DFT(Pareto 更优);顺序学三技能不退化。
- **规模效应**(图):**3B = −3.3**(ICL 弱反不如 SFT)、7B = **+4.0**、14B = **+6.9**——增益随 base 模型 ICL 能力增强。
- **§4.5 reasoning(Olmo-3-7B-Think, Table 2)**:无 CoT 标注的 medical answer-only 数据上,SFT 掉点(31.2→23.5)且缩短输出(4612→3273 tokens),SDFT 反升到 **43.7**(4180 tokens)。
- **代价**:约 **2.5× FLOPs**、**~4× wall-clock**(需 on-policy rollout)。
- **图 7 消融**:teacher 同时条件化于"文章+答案"(89% strict)显著优于仅条件化文章(75%)或仅答案(37%)。

#### 9. 结果如何支撑其主张
"on-policy 蒸馏防遗忘"由 Skill Learning Pareto 优、顺序学不退化支撑;"真正内化知识"由 OOD 间接问题 98 vs SFT 80 支撑;"依赖 base ICL 能力"由 3B/7B/14B 单调趋势支撑;"对 reasoning 模型保持长 CoT"由 Olmo 实验支撑。支撑较充分,但下述 KL 方向 errata 削弱了"reverse-KL/inverse-RL"理论框架与实证的对应关系。

#### 10. 逻辑自洽性(中性评估)
理论(§3.1 把 reverse-KL 蒸馏等价于 on-policy RL/inverse-RL)与方法叙事自洽——**但仅在 reverse KL 假设下成立**。仓库 errata 表明实际跑的是 forward KL,这与 §3.1 的等价性推导**前提不符**:forward KL(GKD 式)是 mode-covering、不再对应论文给出的那条 inverse-RL 等价链。即论文的理论分析与其报告结果所用损失存在方向性脱节(作者已承认并承诺更新 arXiv)。这是本方法当前最大的自洽性瑕疵。增量层面也偏小:核心 trick 此前已在 GKD/context-distillation 出现,贡献主要在定位与实验。

#### 11. 残留问题 / 局限
- **〔代码-论文差异,本轮发现,关键〕** 论文正文称 **reverse KL**,但仓库 README(04/07/26)errata 明言"所有论文结果实为 **forward KL(GKD 式)per-token 损失**",且 `distil_config.py` 的 `alpha` 默认 0.0=forward KL、`main.py` 不暴露该参数。⇒ 论文方法描述与实际实现/结果的 KL 方向相反,§3.1 inverse-RL 等价推导的前提(reverse KL)与实测不一致。上一轮分析按论文写"reverse KL",需据 errata 更正。
- teacher 权重为 EMA 而非字面"同一当前模型";`beta=0` 默认不载 reference,KL 约束仅来自 teacher 分布本身。
- 小模型(3B)因 ICL 能力弱反而**劣于 SFT**(−3.3),方法对 base ICL 强依赖,适用范围受限。
- 2.5× FLOPs / 4× wall-clock 的训练成本显著高于 SFT。
- Knowledge Acquisition 语料仅 ~200K tokens 的窄域(2025 自然灾害),泛化到大规模知识注入未验证。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库:README 现给 clone URL `https://github.com/Continual-Intelligence/Self-Distillation`(上一轮记 `github.com/idanshen/Self-Distillation`,可能已迁移/镜像;主页 idanshenfeld.com/SDFT)。本地已 clone。
- 框架:**TRL 0.24.0**(`requirements.txt` 钉死 `trl==0.24.0`)。`distil_trainer.py:157` 定义 `DistilTrainer(BaseTrainer)`,用 `trl.extras.vllm_client.VLLMClient` 做 on-policy 生成。
- 文件:`distil_trainer.py`、`distil_config.py`(alpha/beta/EMA ref-sync 参数)、`main.py`(入口,默认 Qwen2.5-7B-Instruct / lr 2e-5 / 1 epoch)、`eval_science.py`/`eval_tooluse.py`、`data/`。遗忘指标用 EleutherAI lm-evaluation-harness(指定 commit)。
- 代码可得性:Tier A(可跑),覆盖 tooluse/science 训练+评测;但 KL 方向 errata 需复现者注意(默认 forward KL,非论文所写 reverse KL)。


---


### tip — TIP: Token Importance in On-Policy Distillation

> **一句话重点 (TL;DR)**：OPD 中"哪些 token 有学习信号"由两轴决定——学生熵 ht 与师生散度 δt；熵单轴是有效但结构性不完整的代理，会漏掉"低熵高散度=过度自信错误"的 Q3 盲区；用免参数 Soft-OR 评分把两轴并起来做 top-k token 选择，可在大幅省显存的同时匹配/超过全 token OPD。

**元信息**：arXiv 2604.14084（v4, 2026-05-21，cs.LG）｜ Princeton（通讯 Yuanda Xu）+ 多名共同一作（Hejian Sang、Zhengze Zhou、Ran He、Zhipeng Wang、Alborz Geramifard，部分工业界）｜ Preprint 2026-04 ｜ 主题 OPD 中 token 级重要性选择，与 MTP/OPD 直接相关（"低熵+高散度=过度自信错误"恰是 MTP 前瞻探针可能想捕捉的对象）｜ 代码 https://github.com/HJSang/OPSD_OnPolicyDistillation （已 clone ~390K，**通用 OPSD 仓库**，同挂 TIP/PACED/Sparse-to-Dense，非 TIP 专属，TIP 为在此扩展实现）｜ 框架 verl

#### 1. 相关工作与进展
- **课程学习 / 重要性采样**：Bengio 2009 课程、Kumar 2010 self-paced 按难度排序/加权样本；Katharopoulos 2018 按梯度范数选 mini-batch、Ren 2018 元梯度学样本权重——均在**样本级**。TIP 把粒度推到**序列内单 token**，轴是学生不确定性与师生分歧而非标量难度。
- **off- vs on-policy KD**：序列级 KD（Kim&Rush 2016）训 teacher 生成序列（off-policy）；OPD（Agarwal 2024、Gu 2025）让学生跑自己 rollout、逐 token 监督，避免 train-test 失配。关键：OPD 中 token 重要性由学生自身分布决定，**不能从 teacher 输出预计算**，必须在线评估。
- **token 级重要性**：RL 中 high-entropy "forking tokens" 驱动多数梯度（Wang 2025c）、entropy collapse（Cui 2025）、SPINE 只更新决策分叉点；蒸馏中 AdaSwitch 按散度切换师生引导、Entropy-Aware OPD（Jin 2026）按 teacher 熵调 loss、AdaKD 的 LATF 按 teacher-student Hellinger 距离 top-r% 选择。**最相关是 AdaKD**：但其 LATF 单独只 +0.04 ROUGE-L，主要增益来自正交的温度模块 IDTS；TIP 论证 Q3 不是"大散度 token"的改名，而是低熵∧高分歧的合取（两轴诱导不同选择）。

#### 2. 现有工作存在的问题
熵视角结构性不完整：只用学生熵筛 token，会**漏掉"学生自信但与 teacher 严重分歧"的 Q3 位置（低熵高散度，过度自信错误）**——这类 token 因低熵而与"已解决 token"无法区分，但携带密集纠错信号。论文证明（Prop.2）：任何非递减、f(0)=0 的熵单调评分都对 Q3 近乎赋零权。

#### 3. Motivation
把 token 重要性放到 (学生熵 ht, 师生散度 δt) 两轴平面系统化，理论证明熵单轴的结构盲区，并给出**免参数、无额外计算**（两轴量在标准 OPD 训练中已算）的补救选择规则。

#### 4. 主要灵感 / 核心直觉
信息性 token 来自两个区域：(1) 高学生熵（学生不确定、仍在成形）——熵可检测；(2) 低学生熵但高师生散度（学生自信却错）——熵不可见。从信号-曲率视角看：token 重要性 ∝ ‖梯度‖²/(梯度·Hessian·梯度)，Q3 的 numerator 可大（teacher 强烈反对自信预测），而近确定的学生分布给出小 softmax 曲率，故重要——但 Q4（自信且对）numerator 近零。Q3 vs Q4 是信号差异、非熵差异。

#### 5. 主要解决思路(一段话讲清核心)
TIP 是诊断 + 选择规则：把每个 token 按 (ht, δt) 分到四象限（Q1 高熵高散度=最密纠错信号；Q2 高熵低散度=稳定欠自信；Q3 低熵高散度=过度自信盲区；Q4 低熵低散度=已解决可丢）。用**免参数 Soft-OR 评分** st=ĥt+δ̂t−ĥt·δ̂t（两轴 min-max 归一后，任一轴非零即非零）做 top-ρ 选择，只对选中 token 计标准 reverse-KL OPD loss。

#### 6. 方法详解(通俗、分步骤)
- **两个轴**（论文 Eq.2/3）：学生熵 ht=H(P_S)/log|V|∈[0,1]；师生散度 δt=D_KL(P_S‖P_T)，即 per-token 训练 loss 本身（无额外计算）。
- **四象限统计高度不均衡**（§4）：Q4 约 40–47% token，Q1+Q2 合计约 40–52%，**Q3 仅 3–15%** 但携带超比例纠错信号。
- **理论三结论**：Prop.1 oracle 权重 w\*_t=φ̄_t/(ηβ M̄_t)（φ̄_t=⟨∇L,μ̄_t⟩，M̄_t=E‖g_t‖²）压制 Q4、对 Q1/Q3 给正权；Prop.2 熵单调评分对 Q3 结构盲；Remark 1 Soft-OR 恢复 Q3 覆盖同时压制 Q4。（Prop.1/2、Remark 1 编号已在论文 §5 + Table 2 逐字核到。）
- **选择 = 训练两种 KL 方向的精确区分（论文 §7.3，关键）**：用于 token **排序/选择**的散度是 **forward KL** δ^fwd_t=D_KL(P_T‖P_S)（理由：学生在某 token 近乎确定时 reverse-KL 对 teacher 偏好的其它候选不敏感，forward KL 直接惩罚漏掉的 teacher 概率质量、给出更锐利的 Q3 排序）；**训练 loss 仍是标准 reverse-KL** D_KL(P_S‖P_T)（Eq.1，mode-seeking、数值稳定、前向已算）。Q3-only 检测器 w^Q3_t=δ^fwd_t·(1−ĥt)。
- **仓库实现的散度轴**：通用 Soft-OR 选择器（`entropy_weighted_sample`/`_compute_per_token_entropy_and_jsd`）用的是**归一化 JSD**（weight=entropy_norm^α · jsd_norm^γ，JSD 支持 teacher top-K 截断减噪），三种**训练**散度 reverse_kl/forward_kl/jsd 经 `LOSS_FN_MAP` 切换；另有 `compute_teacher_token_stats` 区分"学生 OOD（teacher 熵高且 p_T(yt) 低）vs 真分叉点"。即论文 Q3 实验用 forward-KL 检测器、通用选择器用 JSD，两者都不是"δt 就是 reverse-KL"。
- **落地（§6）**：给定保留比 ρ，按 st top-K 选 token；选前对每 batch 熵 clip 顶 2% 再 min-max 归一，稳排序；额外成本仅 O(m log m) 排序，可忽略。

#### 7. 实验数据集
- 数学推理：训练 prompt 来自 DAPO；评测 MATH-500（500 题）、AIME 2024/2025（各 30 题），mean@16。三组师生对：**Qwen3-8B(GRPO)→4B**、**Llama-3.3-70B-Instruct→Llama-3.1-8B-Instruct**、**Qwen2.5-14B-Instruct-thinking→Qwen2.5-1.5B-Instruct**（~9× 容量差、reasoning teacher）。
- Agentic 长程规划：DeepPlanning（多日旅行 + 多商品购物，需主动信息获取/局部约束推理/全局约束优化）；师生对 **Qwen3-{14B,32B}→Qwen3-1.7B**（均开 thinking，agentic 规划数据训练，15 epoch）；80/20 划分，Avg@16，按个性化硬约束满足比例打分。
- 训练统一 AdamW + cosine + reverse-KL，lr=1e-6（Qwen3/Qwen2.5）/3e-7（Llama）。

#### 8. 实验结果与主要发现
- **Q1/Q2（熵选择，Table 3）**：保留 50% token 的熵采样匹配/超过全 token 基线于多数 benchmark（Qwen3 MATH 76.7→78.6，Llama 71.0→74.0），峰值显存最多降 47%（Qwen3 72.0→38.1 GB）；更激进保留时性能常掉到基线下（印证有信号留在被丢的低熵区）。
- **Q3-only（Table 4）**：只训低熵高散度 token（<10% 全 token），近乎追平全 token 基线（Qwen3 MATH 5.7K token 达 76.1 vs 76.7）。Q3 不是"大散度"——纯按 δt 选在等预算下不及基线、需 5× token 才追上（Appendix B.2 Table 8）。
- **Soft-OR（Table 5）**：数学上稳定优于熵单选（Qwen3 MATH 50% 达 79.1 vs 熵 78.6；AIME'24 25.7 vs 23.8）。top-50% vs bottom-50%（Table 6）证 bottom 信号显著更弱。
- **DeepPlanning（Table 7）**：熵 50% 匹配/超全 token OPD（14B：12.1 vs 11.7）；**最亮点 Q3-only 20% 反超全 token OPD（14B：12.6 vs 11.7；32B：13.6 vs 12.8）**——agentic 任务里单个自信错误（订已关场地/超预算）可作废整个计划，故 Q3 纠错特别集中。
- teacher 熵几乎恒定、对选择无判别力（§7.4）。

#### 9. 结果如何支撑其主张
论文用 Table 2 把三条理论结论一一映射到实验：Prop.1→§7.2（Q1/Q2 携带最多信号、去 Q4 提效）、Prop.2→§7.3（熵漏 Q3）、Remark 1→§7.4（组合分恢复 Q3 且超熵单选）。三结论在 3 模型族 × 2 任务域上均被支持；最锐利的是 Q3 盲区——<10% 专门选过度自信的 token 近乎追平全训。DeepPlanning 上 Q3-only 20% 反超全 OPD 是最强的"两轴必要性"证据。

#### 10. 逻辑自洽性(中性评估)
逻辑自洽且理论-实验对应工整：两轴诊断→四象限→证明熵盲区→免参数 Soft-OR 补救→分任务验证。forward-KL（选择）vs reverse-KL（训练）的方向区分论证清楚、动机正确，未混淆。oracle 权重是 population 量、不可直接用，作者诚实标注并用 Soft-OR 作可计算代理。Q3 与"大散度"的区分通过 budget-matched 对照（Table 8）切实排除了平凡解释。

#### 11. 残留问题 / 局限
- 核心贡献是"诊断 + 选择规则"，方法本身是对既有熵选择的增量补丁（Soft-OR 在数学上提升量级有限，如 MATH-500 76.7→79.1@50%）。
- 共享 OPSD 仓库（同挂 TIP/PACED/Sparse-to-Dense）使复现归属略含糊；论文 Q3 检测用 forward-KL，而仓库通用选择器用 JSD，二者需对应留意。
- 作者自陈局限：最大 teacher 仅 70B、rollout 16K token；象限结构与 Q3 集中是否在万亿参数或极长 agentic tool-calling 流水线下持续，仍开放。
- 省显存卖点实在，但"无额外计算"指排序量可复用已算 logits，非零成本（仍有 clip+归一+top-k 排序）。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库 https://github.com/HJSang/OPSD_OnPolicyDistillation （已 clone，`src/opd/{losses.py,opd_worker.py,opd_trainer.py,batch_builder.py}`）。基于 **verl**：`opd_worker.py` 两阶段 teacher/student 显存调度、chunk 化散度（V=152K 时单 (N,V) float32 ~2.3GB，故分块）、remove-padding；GRPO 与 OPD 并存。
- `losses.py` 已核对：提供 `compute_reverse_kl_loss`/`compute_forward_kl_loss`/`compute_jsd_loss`（`LOSS_FN_MAP` 切换训练散度）、`entropy_weighted_sample`（weight=entropy_norm^α·jsd_norm^γ，按权 top-ratio 多项式无放回采样）、`_compute_per_token_entropy_and_jsd`（含 JSD teacher top-K 截断）、`compute_teacher_token_stats`（teacher 熵 + p_T(yt) 区分 OOD/分叉）。
- 框架栈：torch 2.9 / transformers 4.57 / ray / hydra（`config/opd_trainer.yaml`）。代码与论文核心（两轴选择 + 三种散度 + 显存优化）对应清晰，TIP 为在 OPSD 上的扩展实现。


---


### unisd — UniSD: Towards a Unified Self-Distillation Framework for Large Language Models

> **一句话重点 (TL;DR)**：把"无外部强 teacher 的自蒸馏"建模为 on-policy 轨迹上的可靠性感知自纠错，沿监督可靠性/表示对齐/训练稳定性三轴整合五个互补组件（多 teacher 一致性、EMA teacher、token 级对比、特征匹配、散度裁剪）做系统消融；整合版 UniSD\* 较 base +5.4、较最强基线 GKD +2.8。

**元信息**：arXiv 2605.06597（v2, 2026-05-21，cs.CL）｜ Georgia Tech（共同一作 Yiqiao Jin、Yiyang Wang）+ UCLA + CMU + W&M（通讯 Jindong Wang@W&M、Srijan Kumar@GT）｜ Preprint 2026-05 ｜ 主题 自蒸馏统一框架 + 组件级消融，OPD 相关性中等（偏经验综述/工程整合而非单一新机制）｜ 代码 https://github.com/Ahren09/UniSD （已 clone ~883K，项目页 unifiedsd.github.io）｜ 框架 TRL 1.4 + vLLM 0.20.2 + transformers 5.8 + torch 2.11(cu128)

#### 1. 相关工作与进展
- **持续学习与 on-policy 学习**：catastrophic forgetting 是核心挑战；标准 SFT 是 off-policy（训固定示范、有 train-inference 失配），on-policy 学习（GKD 减 exposure bias；MiniLLM/DistiLLM 用稳定 KL 目标改进分布匹配）缓解失配。
- **KD 与自蒸馏**：经典 KD 匹配预测/logits/隐状态/输出/推理迹；近期 on-policy 变体 VLA-OPD、SCOPE、StableOPD 用专家 teacher 或自适应稳定监督学生轨迹，但**依赖外部 teacher**。自蒸馏从模型自身导监督（SDFT 用 demonstration-conditioned base 当 teacher；OPSD 在学生轨迹上稠密监督；SDPO 用特权环境反馈）。UniSD 区别于"研究单个自蒸馏配方"，做统一可扩展框架。

#### 2. 现有工作存在的问题
自回归 LLM 自蒸馏三难：(1) **开放式生成**——自生成轨迹自由、正确性任务相关、前缀改变后续条件；(2) **自监督不可靠/不稳定**——on-policy 暴露自身错误，过自信预测与稀有高散度 token 会被跨步强化；(3) **缺乏系统理解**——既有方法孤立研究单个设计选择，不清楚谁起作用、如何交互。

#### 3. Motivation
中心问题：LLM 能否仅靠 self-derived 监督改进（避开外部强 teacher 的成本、访问/许可限制与偏置传播风险）？把自蒸馏建模为"on-policy 轨迹上的可靠性感知自纠错"，并沿三条轴组织既有机制以便系统消融：监督可靠性、表示对齐、训练稳定性。

#### 4. 主要灵感 / 核心直觉
自蒸馏的成败取决于三件事：用什么信号（可靠性）、匹配什么表示（对齐）、每步更新多强（稳定）。这三者可由互补组件分别处理——agreement 决定当前步信任哪些信号、EMA 平滑 teacher 跨步漂移、对比学习区分有效监督与貌似合理的错误、特征匹配把表示对齐拉到输出分布之外、散度裁剪防稀有高散度 token 主导。把它们放进同一 on-policy 训练环即可受控消融"谁起作用、如何交互"。

#### 5. 主要解决思路(一段话讲清核心)
统一目标 L=E[Σ_t m_t w_t D(πθ‖π_teacher) + λ_aux L_aux]（m_t token 掩码、w_t 可靠性权重、D token 级散度）。在此框架下逐组件开关做大规模消融，再把五组件拼成整合版 UniSD\*：agreement+对比选可靠信号、特征匹配传表示、EMA+裁剪稳优化，全部在同一 on-policy loop 内。

#### 6. 方法详解(通俗、分步骤)
- **(a) Multi-Teacher Agreement**：用多个 task-preserving 上下文视角（retrieved / random few-shot / induced 指令）的同一 teacher 重打分学生轨迹，token 级 δt=A({ℓ^k_t})、序列级 δ_seq=A({L^k}) 估不一致（A 为方差/极差等变率统计），转成可靠性权重 w_t；所有视角共享一个 teacher、批处理，不额外复制 teacher。
- **(b) EMA Teacher**：θ̄_n=βθ̄_{n-1}+(1−β)θ_n，用 EMA teacher 替代主 teacher 做时间平滑目标，防 teacher 跨步漂移传播瞬时错误/过自信。
- **(c) Token-Level Contrastive Learning**：margin 目标 L_aux=Σ m_t max(0, γ+d^+_t−d^-_t)，d^±_t=|ℓθ_t−ℓ^±_t| 为学生到正/负条件 teacher 信号的距离；负例 y^- 由 LLM 生成貌似合理错误、腐化推理或 WordNet/PPDB/TextAttack 词法扰动构造。
- **(d) Feature Matching**：L_feat=Σ m_t‖f^θ_t−f^*_t‖²，实现里匹配末层隐状态。
- **(e) Divergence Clipping**：先算加权 JSD D^(α)_t（α∈(0,1) 插值 forward/reverse KL，也支持纯 forward/reverse 端点），再 cap eD_t=min(D^(α)_t, κ)；与 agreement 权 w_t 组成 L_clip=Σ m_t w_t eD_t / Σ m_t w_t。
- **UniSD\***：上述全开（Algorithm 1）；从监督/表示/优化三视角组合 signal selection + 表示对齐 + 时间平滑 + loss 稳定。

#### 7. 实验数据集
六 benchmark（四任务类）：**ScienceQA、GPQA**（科学，GPQA 仅测）、**CoS-E**（常识）、**MBPP、HumanEval**（代码，HumanEval 仅测）、**ToolAlpaca**（工具）。SCIENCEQA→GPQA、MBPP→HumanEval 作 OOD 泛化。六模型 / 三族（已核对论文 §3.1 + requirements）：**Qwen2.5-Instruct 四规模 {0.5,1.5,3,7}B**（7B 为主实验 base）+ **Llama-3.1-8B-Instruct** + **gemma-3-4b-it**（后两者用于跨族泛化）。〔确认：论文中并无 InternLM，第三族实为 Gemma-3；原稿曾误列已更正。〕

#### 8. 实验结果与主要发现
- 主表（Table 1，Qwen2.5-7B，retrieved 上下文）：Raw 67.9；baseline SFT 68.3 / SDFT 70.1 / **GKD 70.5（最强基线）** / SSD 67.3 / OPSD 68.2；单组件 Agree(Tok.)72.2、Agree(Seq.)72.5、EMA 72.5、Contrast 71.9、Match(Joint)72.1、Clip 70.3；**UniSD\* 73.3**（较 Raw **+5.4**、较 GKD **+2.8**）。
- 关键发现：SFT 仅在格式向任务（ToolAlpaca +4.4）有效、在 ScienceQA/GPQA/MBPP/HumanEval 退化（mean-seeking 不适合多样推理路径）；on-policy baseline 更强。EMA 是最强单组件（ToolAlpaca 77.9，+16.1 over Raw）；Contrast 最均匀正（六 benchmark 全升）；Clip 最保守、最省时省显存（轻量稳定器而非主信号）。
- 跨族泛化（§3.4，Fig.7）：UniSD\* 在 Qwen2.5/Llama-3.1/Gemma-3 上较 base +5.4/+3.1/+2.2，18 个 model-dataset 对中 15 升 2 平 1 退（仅 1 个 OOD 退化），均超 GKD。
- 分布保持（§3.5）：自蒸馏大幅降 gold-completion PPL（Qwen2.5-7B 20.74→5.7–6.1）；SFT 致 base-distribution 漂移（retention PPL 1.14→1.68），EMA 较 SFT 降 retention PPL 33.9%；UniSD\* 把 mean token-level JSD-to-base 从 SFT 的 0.054 降到 0.041。
- agreement 敏感性（§3.3）：性能随 teacher 数 K 非单调；retrieval 上下文在语义相似有用时最强、coding/开放生成上未必；γ 大→更鲁棒但峰值低（stability-adaptivity 权衡）。

#### 9. 结果如何支撑其主张
- "自蒸馏可不靠外部 teacher 改进"：UniSD\* 全程 self-derived 监督仍 +5.4/+2.8，且跨三族稳定（15/18 升）。
- "三轴组件互补"：Table 1 单组件 + Fig.5 组件有效性显示无单一 benchmark/组件主导增益（EMA 强于 ToolAlpaca、Agreement/UniSD\* 强于 ScienceQA/HumanEval、UniSD\* 最强于 MBPP/GPQA）。
- "稳定/保持"：retention PPL 与 token-level JSD-to-base 的下降支撑"自蒸馏避免 SFT 式灾难性漂移"。

#### 10. 逻辑自洽性(中性评估)
框架组织自洽（三轴 → 五组件 → 整合），消融充分、结论（谁起作用、如何交互）有数据支撑且诚实报告了组件的条件性（如 retrieval 非一致最优、K 非单调、Clip 仅轻量稳定）。但本质是"kitchen-sink 式整合 + 消融贡献"：五组件单看都非首创（EMA teacher、对比学习、feature matching、散度裁剪、多视角一致性均为已有思路），价值在统一接口与交互结论而非新机制。

#### 11. 残留问题 / 局限
- +2.8/+5.4 绝对增益不大；benchmark 多为短生成（代码/QA/工具），未覆盖长链数学推理，与本项目 long-CoT/OPD 场景相关性较弱。
- agreement 计算昂贵（每补全要多上下文重打分；Qwen2.5-7B seq-level ~100 min vs SFT 18.6 min），作者自提应做"按可靠性预算分配"的自适应自蒸馏（未实现）。
- 资源/碳排放估算（Table 3）为基于固定假设的相对估计、非实测。
- 组件最优配置（K、γ、上下文构造）随任务/粒度变化，需调参。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库 https://github.com/Ahren09/UniSD （已 clone）。`src/` 含完整实现：`trainers/unisd_trainer.py`、`train/train_unisd.py`、`teacher/{auxiliary_context, instruction_induction, negative_demonstrations}.py`（对应 agreement 上下文构造 + 对比负例）、`eval/{eval_code, eval_gsm8k, eval_mcqa, eval_retention, eval_tooluse}.py`、`config/`、`prompts/`、`analysis/`（含资源消耗测量）。代码与五组件 + 消融脚本对应清晰，可得性高。
- 框架（已核对 requirements.txt，CUDA 12.8/Python 3.12）：torch 2.11.0+cu128、vllm 0.20.2、transformers 5.8.0、**trl 1.4.0**、accelerate 1.13、peft 0.19、deepspeed 0.19、flash_attn 2.8.3。（README 徽章显示的 torch 2.9/transformers 4.57/vLLM 0.12 为粗略版本，以 requirements.txt 为准。）


---


### vla_opd — VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models via On-Policy Distillation

> **一句话重点 (TL;DR)**：把 VLA 后训练改写成"在 student 自采样轨迹上、用冻结 teacher 的 dense token-level 监督做 reverse-KL 蒸馏"的 on-policy RL，从而同时拿到 SFT 的快收敛、RL 的少演示/抗遗忘，并用 reverse-KL 的 bounded mode-seeking 避免 Forward-KL 熵爆炸与 Hard-CE 熵坍缩。

**元信息**：arXiv 2603.26666v1（cs.RO，2026-03-27）｜ HKUST(GZ)（Zhide Zhong, Haodong Yan, Junfeng Li, Junjie He, Tianran Zhang, Haoang Li）｜ 预印本 ｜ 主题 机器人 VLA 后训练 / on-policy distillation，与本项目（LLM reasoning OPD）**关系外围**——共享"on-policy 蒸馏 + reverse-KL mode-seeking"方法骨架，但落在机器人动作 token 而非语言推理 ｜ 代码 项目页 https://irpn-lab.github.io/VLA-OPD/ 标注 "Code (Coming Soon)"（**截至核查无可用仓库，纯论文可读**）｜ 框架 〔待核：代码未放出，下述基于论文〕基于 GRPO 式分组采样、teacher=SimpleVLA-RL、student=OpenVLA-OFT

#### 1. 相关工作与进展
VLA 后训练当前两大范式：(1) 离线 SFT / 行为克隆（O'Neill 2024, Black 2024 π0），dense 监督、收敛快；(2) 在线 RL（SimpleVLA-RL、VLA-RL、RLinf-VLA 等），用 GRPO 做 group-relative 优势、免 critic，把策略对齐到自身诱导的状态分布以学恢复行为。交互式模仿学习（DAgger/HG-DAgger）在 student 诱导的 OOD 状态上收集专家标注。LLM 蒸馏侧的 MiniLLM(Gu 2023)、GKD(Tan 2023) 提出 reverse-KL / on-policy 蒸馏，本文将其迁移到动作预测。

#### 2. 现有工作存在的问题
- 离线 SFT 是 off-policy：在专家状态训练却在 student 诱导状态评测，复合误差（exposure bias）使其无法从自致偏离状态恢复；且对 static、disjoint 数据集做激进参数更新 → 灾难性遗忘。
- 稀疏奖励在线 RL（GRPO）：机器人任务通常只有终态二值信号 R(τ)∈{0,1}，信用分配困难、方差高、样本效率极低。
- 简单把 SFT 改 on-policy（如 DAgger）用次优对齐目标：Forward-KL（soft 标签）mode-covering，在 teacher 高熵的 OOD 状态会模仿其犹豫 → 熵爆炸；Hard-CE（argmax 标签）丢弃 dark knowledge，在多模决策边界刚性追 argmax → 过早熵坍缩、丧失探索多样性。

#### 3. Motivation
在 student 自生成轨迹上用冻结 teacher 提供 dense token-level 监督，把延迟稀疏奖励转成即时监督（加速收敛 + 注入"恢复先验"）；用 reverse-KL 的 bounded mode-seeking 性质，只要 student 动作落在 teacher 可接受概率质量内即不被惩罚，从而过滤 teacher 尾部不确定性、保留模内随机性；on-policy 更新把梯度锚定在 student 当前行为流形上，实现"温和对齐"缓解遗忘。论文同时声称：依赖高性能 teacher 的可得性已不再是瓶颈（开源 checkpoint / API / 易训单任务策略）。

#### 4. 主要灵感 / 核心直觉
核心直觉是"散度方向决定 OOD 状态下的熵动力学"。teacher 在 OOD 状态往往呈现平坦高熵分布（epistemic uncertainty）；Forward-KL 在 teacher 样本上估梯度强制覆盖全支撑 → 继承高熵；reverse-KL 的 zero-forcing 让 student 只需对齐 teacher 的主模、忽略长尾，从而"果断但仍可探索"。把 reverse-KL 写成 token-level 内在奖励后即可挂进 group-based policy gradient。

#### 5. 主要解决思路(一段话讲清核心)
三阶段闭环（Algorithm 1）：student πθ 在环境 on-policy rollout G 条轨迹（主动暴露 OOD 失败状态）→ 对 student 访问过的每个状态查询冻结 teacher πtea 的 action logits（不在环境执行）→ 用 token-level 负 reverse-KL 作内在奖励 r_t = −(log πθ(a_t|s_t) − log πtea(a_t|s_t))（对 student log-prob 项 stop_gradient），等价于在 student-visited states 上最小化对 teacher 的 reverse-KL；用 group 平均的 policy gradient 降方差。

#### 6. 方法详解(通俗、分步骤)
1. **初始化**：student 由极少演示 SFT 得到（LIBERO 1-traj、RoboTwin2.0 1000-traj），是脆弱下界；teacher 为 RL 训得的鲁棒专家（SimpleVLA-RL），全程冻结。
2. **Phase 1 学生采样（探索）**：πθ 在环境中跑 G 条轨迹，频繁进入 OOD 失败态 serr，把"未知"区显式纳入训练分布。
3. **Phase 2 教师标注（纠正）**：对每个被访问状态 st，查询 teacher 得 qt(a)=πtea(a|st) 作 dense 引导，注入恢复先验；teacher 只打标不执行。
4. **Phase 3 模式寻优（更新）**：token-level 奖励 r^OPD_t = −log(πθ/πtea)（式 6，stop_gradient 作用于 student log-prob），等价最小化 reverse-KL（式 5）。梯度按 group 平均（式 7）；**与标准 GRPO 不同，不做 outcome reward 归一化，直接用 raw reverse-KL reward 作优势**。
5. **理论对比**：Forward-KL=mode-covering→熵爆炸；Hard-CE=丢 dark knowledge→熵坍缩；reverse-KL=zero-forcing bounded mode-seeking→健康熵。
6. **两个变体**：Ours(Distill) 仅蒸馏；Ours(Distill+GRPO) 蒸馏热启后再 GRPO 微调。主实验 batch=64、G=8（沿用 SimpleVLA-RL）；消融固定 batch=32。

#### 7. 实验数据集
- **LIBERO**（单臂，四套件 Spatial/Object/Goal/Long）：极端数据稀缺，每任务 1-traj SFT 初始化。
- **RoboTwin2.0**（双臂协作，四代表任务 Pick dual bottles / Place Empty Cup / Handover Block / Stack Bowls Two，覆盖 short→long horizon）：每任务 1000-traj SFT 初始化。
- 遗忘评测：在 seen 任务微调，在 4 个 held-out unseen 任务（2 Object、2 Spatial）评估 seen–unseen 权衡。
- 全部为**仿真基准，无真机**。

#### 8. 实验结果与主要发现
- **效率（Fig.2）**：LIBERO-Object 蒸馏 10 步内 >90%（"垂直起飞"）；LIBERO-Long 仅 ~50 步达基线 ~150 步水平（≈3× 加速），且曲线平滑（基线 GRPO 锯齿震荡）。
- **效能（Table 2，LIBERO 成功率%）**：student init 平均 48.9 → Distill 87.4（媲美/超过若干 50-traj 全量基线如 Octo 75.1、OpenVLA 76.5）→ Distill+GRPO 93.4（逼近 teacher SimpleVLA-RL 93.9）。
- **双臂（Table 3，RoboTwin2.0）**：student init 45.2 → Distill 71.1（近 teacher 74.0），超 π0(50.5)、RDT(32.0)；未报 Distill+GRPO。
- **抗遗忘（Fig.3）**：离线 SFT 在 unseen 上严重坍缩（Object 近零、Spatial 大跌）；on-policy 方法（RL 与本法）基本避免，VLA-OPD 多数轴上匹配或超过 RL。
- **消融（Fig.4，RoboTwin2.0 Beat Block Hammer）**：reverse-KL 稳步上升；Forward-KL 早期 >50% 性能谷且熵爆炸；Hard-CE 熵坍缩、平台最低。
- **group size（Fig.5，LIBERO-Object，batch=32）**：G=8 最高(~89%)，G∈{2,4} 仍 >80% 不坍缩，小 G 显著省 rollout/teacher 推理开销。

#### 9. 结果如何支撑其主张
- "效率优于稀疏 RL"：Fig.2 收敛步数对比直接支撑（10/50 步 vs 150 步）。
- "效能逼近 teacher"：Table 2/3 终值（93.4 vs 93.9；71.1 vs 74.0）支撑。
- "reverse-KL 优于另两目标且对应熵行为"：Fig.4 性能曲线 + actor 熵曲线一一对应，是对核心主张最直接的证据。
- "抗遗忘"：Fig.3 seen–unseen 散点支撑离线 SFT 坍缩 / on-policy 保留的定性对比。
- 整体证据链自洽，但多为单基准/单任务曲线（如消融仅 1 个 RoboTwin 任务），统计显著性与多 seed 方差未报。

#### 10. 逻辑自洽性(中性评估)
- 内在逻辑顺：把延迟稀疏奖励换成 dense token 监督、reverse-KL 解释熵两极，链条清晰。
- 但有两处张力：(1) "raw reverse-KL 作优势、不做归一化仍稳定"只有经验曲线支撑，无理论收敛保证，且 stop_gradient 后该梯度估计的方差/偏置性质未分析；(2) 与稀疏 RL 的"效率"对比未计入 teacher 自身训练成本——teacher 即由 RL(SimpleVLA-RL) 得到，"省掉 RL 探索成本"实质是把成本前移到 teacher 训练，公平性存疑。
- "近 teacher"也意味着方法本质受 teacher 性能上界约束（作者承认）。

#### 11. 残留问题 / 局限
- 依赖高性能 teacher 的先验可得性（作者列为 future work 的主攻方向）。
- 评测仅 LIBERO/RoboTwin2.0 两仿真基准、**无真机**；泛化到真实物理/感知噪声未知。
- 不归一化的优势稳定性靠经验；无多 seed 方差、无显著性检验。
- 与稀疏 RL 的成本对比未控制 teacher 训练成本。
- 代码未放出，复现性受限。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 项目页：https://irpn-lab.github.io/VLA-OPD/ ，标注 "Code (Coming Soon)"；论文未给 GitHub 链接。
- **代码可得性：截至核查日无可用仓库（RepoExists=NO），仅论文可读**。〔待核：后续是否放出〕
- 论文层面框架：student=OpenVLA-OFT，teacher=SimpleVLA-RL；分组采样沿用 SimpleVLA-RL 设置（batch=64, G=8）；优化为 group-based policy gradient（类 GRPO 但不做 outcome reward 归一化）。


---


### why_sd_degrade — Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?

> **一句话重点 (TL;DR)**：自蒸馏在数学推理上会"response 变短但性能反降（最高 ~40%）"，根因是 teacher 在富 context 下生成的自信轨迹压制了 epistemic verbalization（wait/hmm 等不确定性表达）；该压制由 conditioning 信息丰富度驱动，其危害随任务覆盖度增大而显现于 OOD。

**元信息**：arXiv 2603.24472v3（cs.CL，2026-05-20）｜ Microsoft Research + KAIST + SNU（Jeonghye Kim 实习于 MSR；Xufang Luo 通讯）｜ Preprint, under review ｜ 主题 自蒸馏/OPD 失败模式分析（**诊断/机理论文，无新算法**），与 OPD 直接相关：解释 privileged-context 自蒸馏为何在数学上"变短变差" ｜ 代码 https://github.com/beanie00/self-distillation-analysis （53M，已 clone，built upon lasgroup/SDPO）｜ 框架 veRL（仓库含 `verl/` 目录、megatron/sglang 脚本）

#### 1. 相关工作与进展
自蒸馏（Snell 2022）：同一模型在 privileged context（如 ground-truth 解、环境反馈）下当 teacher，给无 context 的 student 提供 dense 信号。SDPO(Hübotter 2026) 条件于自身正确轨迹/环境反馈；OPSD(Zhao 2026) 用 ground-truth 解作 privileged 信息。这类方法在 agentic、科学推理等域高效提分，且常"response 变短、性能变好"。相关线：epistemic verbalization 框架(Kim 2026)；推理压缩(GFPO/OPSDC/ConPress/Accordion/CEEH)。本文建于 SDPO codebase 之上，专门追问"何时/为何自蒸馏会退化"。

#### 2. 现有工作存在的问题
既有自蒸馏工作展示了有效性，但**未研究何时/为何会退化**——尤其在"模型需完全自主解题、无外部环境交互"的数学推理场景。把同一套自蒸馏(SDPO)用到数学时出现反常：response 仍随训练变短，但性能显著下降（最高掉约 40%）。"为何朝正确答案训练反而退化？"

#### 3. Motivation
追因到 epistemic verbalization 被压制。把数学推理视为 self-Bayesian 推理：逐步在已生成 token 上更新对中间假设的信念；不确定性表达不是冗余，而是保留备选假设、支持渐进纠错的信号，过早自信会锁死错误假设无从恢复（Fig.2a）。teacher 拿到富 context（c=解）后生成低不确定性的自信轨迹，student 模仿即丢掉这些 epistemic 信号。

#### 4. 主要灵感 / 核心直觉
用条件互信息 I(y*;c|x)=H(y*|x)−H(y*|x,c) 量化 context c 对正确答案 y* 的信息量。直觉：c 越富 → teacher 轨迹越简洁自信、epistemic token 越少；这在窄覆盖任务上加速 in-domain 收敛，但在宽覆盖/OOD 上因丢失不确定性信号而退化。标准训练目标不惩罚这种"风格漂移"，故 OOD 受损是"隐性"的。

#### 5. 主要解决思路(一段话讲清核心)
不是提新算法，而是受控实证：固定/系统改变两个因子——(1) conditioning context 信息丰富度（用 I(y*;c|x) 形式化），(2) 任务覆盖度（训练题数 |D|）——观测 response length、score、10 个 epistemic token（wait/hmm/perhaps/maybe/actually/alternatively/seems/might/likely/check）频率随之如何变化，并对比 GRPO vs SDPO 的 in-domain 与 OOD 表现，从而定位"epistemic 压制 ↔ OOD 退化"的关联。

#### 6. 方法详解(通俗、分步骤)
1. **自蒸馏目标（Eq.1）**：L=Σ_t KL(πθ(·|x,y_<t) ‖ stopgrad πθ(·|x,c,y_<t))，即 **student 向 teacher 对齐的标准 KL(student‖teacher)，是前向方向的自蒸馏目标**。〔已核并保留：此前 analysis 一度写"reverse-KL"不准——论文 Eq.1 是 forward-direction KL(student‖teacher)。〕
2. **§3 信息丰富度受控对比**：DeepSeek-R1-Distill-Qwen-7B 在 DAPO-Math-17k 选 100 题（base 8-rollout 准确率∈[0.125,0.5]），比较 4 种 c：(1)unguided c=∅、(2)solution c=s、(3)c=s\think、(4)regeneration c=yr。MI 排序 (1)<(3)≤(4)≤(2)。**Table 1**：随 I 增大，length 与 epistemic 计数单调下降——(1) score 0.30 / len 13,054 / E 182.5；(2) 0.98 / 1,873 / 8.8；(3) 0.78 / 12,036 / 159.8；(4) 0.95 / 2,808 / 24.1。
3. **§4 off-policy SFT 对照（Table 2）**：在 800 条正确轨迹上微调 DeepSeek-7B——D_ug（unguided，高 E、~12k tok）几乎不掉；D_sg（solution-guided，低 E、~2k tok）全面大跌（AIME24 54.79→20.21、AIME25 37.92→12.71、AMC23 89.06→57.03、MATH500 92.19→65.52）。证明"即便全是正确答案，过度压制 epistemic 也实质损害推理"。
4. **§5 on-policy 自蒸馏**：GRPO vs SDPO，DAPO-Math-17k 上跑 ~100+ step，比较 c=s 与 c=s\think。teacher **固定为初始策略（EMA rate 0.0）**优于移动 target。DeepSeek-7B：SDPO(c=s) 使 AIME24 ~40%、AMC23 ~15% 跌；c=s\think 缓解但仍低于 base。Qwen3-8B(think on/off)、Olmo3-7B 同向。Fig.3d/4d：SDPO 比 GRPO 更激进压制 epistemic token（尤以 wait 为甚）。
5. **§5.4 EMA 消融**：EMA 0.05（慢更新 teacher）比固定 teacher 退化更大——形成"越自信→teacher 越自信"的反馈回路。
6. **§6 任务覆盖度**：变 |D|∈{1,8,64,128,512}（Qwen3-8B think off）。|D|≤128 时 SDPO 快速高分且 len 减 8×（窄覆盖高效）；|D|=512 时 SDPO 反伤 score、且 OOD 全程低于 base，而 GRPO 随 |D| 增大靠增 epistemic 表达持续提升。

#### 7. 实验数据集
- **作者自跑核心实验在数学**：DAPO-Math-17k（训练，14,000 distinct 题）；AIME24/25、AMC23、MATH-500（OOD 评测，题型不与训练重叠）。
- **Chemistry(ScienceQ&A) 与 LiveCodeBench v6 并非作者重跑**：Fig.1a 的 Chemistry 曲线直接取自 SDPO(Hübotter 2026) 的 W&B 日志，与 LiveCodeBench v6 一起仅用于 §6 的**任务覆盖度对比**（Table 3：Chemistry 共 2,400 题但仅 6 类题型/90-10 split；LiveCodeBench v6 仅 131 题且 train/eval 全重叠；vs DAPO-Math 14,000 题不重叠题型）。〔已核并保留此审计事实。〕
- 模型：Qwen3-1.7B/8B（think on/off）、DeepSeek-R1-Distill-Qwen-7B、Olmo3-7B-Instruct。

#### 8. 实验结果与主要发现
- **Takeaway 1**：c 越富 → 越自信、epistemic 越少（Table 1 单调）。
- **Takeaway 2**：即便训练全是正确轨迹，过度压制 epistemic 也大幅降推理（Table 2 D_sg 全面跌）。
- **Takeaway 3**：on-policy 自蒸馏随 c 变富压制 epistemic、缩短 response，幅度依 base 原有不确定性水平而定。
- **Takeaway 4**：epistemic 的价值随泛化需求增长——窄覆盖近冗余可删，覆盖越宽越重要。
- 关键现象：math 上 SDPO 持续低于 GRPO 与 base（AIME24 ~40% 跌）；c=s\think 缓解但不消除；EMA 加剧；epistemic 频移（Fig.12）远大于全词表均移（30–40×），说明训练**专门**作用于 epistemic 表达而非均匀漂移。

#### 9. 结果如何支撑其主张
- "信息丰富度→epistemic 压制"：Table 1 + 逐 token Fig.9 强支撑（单调、且 wait/maybe/perhaps 变化最大）。
- "压制实质有害（非纯风格）"：Table 2 off-policy SFT 是较干净的因果对照（同为正确轨迹，仅 epistemic 密度不同→大幅差异）。
- "覆盖度调制危害"：§6 |D| sweep + GRPO/SDPO 反向趋势支撑。
- "为何 chem 涨 math 跌"：Table 3 用覆盖度差异（题型数/train-eval 重叠）解释，但 chem 数据为**引用**而非重跑，属间接论证。
- 频移对照（Fig.12）排除"通用词表漂移"混淆，强化"训练特异性作用于 epistemic"。

#### 10. 逻辑自洽性(中性评估)
- 机理叙事自洽：I(y*;c|x)→epistemic 压制→OOD 退化，由 MI 排序、Table 1/2、|D| sweep 串起。
- 但"epistemic verbalization 因果驱动正确率"主要靠相关性 + 受控对比，**尚非严格因果**（未做"强行注入/移除 epistemic token 看正确率"的反事实干预）。
- 跨域比较一半证据为引用 SDPO 日志，口径不完全一致；epistemic 用 10 个固定词近似（LLM-as-Judge 在 Appendix 作补充验证），多词短语未被单 token 计数捕获。
- 结论稳健但偏现象学，主张落在"后训练目标应显式保留不确定性表达"。

#### 11. 残留问题 / 局限
- 缺反事实因果实验，"epistemic→正确率"为强相关而非证明。
- Chemistry/LiveCodeBench 为引用结果，跨域对比间接。
- 仅数学 OOD 基准；"何种程度的不确定性是恰当的"无可操作判据。
- 未给出修复算法（自承为分析论文）；与 caopd 同源（privileged-context 信息不对称→过自信），但 caopd 修 verbalized confidence 校准、本文关注 reasoning 链内 epistemic token 对正确率影响，二者互补。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库：https://github.com/beanie00/self-distillation-analysis （53M，已 clone）。含 `analyzing_reasoning_behavior`、`experiments/math`（GRPO/SDPO 训练脚本）、`eval`、`data/math`（§6.2 缩减题集 + 评测集）、Docker、HF↔mcore 转换。
- **框架：veRL**（仓库含 `verl/` 目录；修改 `verl/trainer/ppo/ray_trainer.py` 的 `_remove_thinking_trace`）；代码 built upon lasgroup/SDPO，环境按 SDPO README 安装；transformers + vLLM/SGLang。
- 关键超参（README）：train_batch_size=256、max_response_length=20480、teacher_update_rate=0（主）/0.05（消融）、remove_thinking_from_demonstration 区分 c=s 与 c=s\think；训练在 4×B200。
- 代码可得性：完整开源 + HF 释出 Qwen3-8B(think on/off)/DeepSeek-Distill-7B checkpoint + W&B 日志。


---

## 3. T2 — Tool-Agent / 多轮（19 篇）

本部分逐篇内联工具调用 / 检索 / 深度搜索 agent 的蒸馏与多轮 RL 工作：agent 轨迹蒸馏（chain_of_agents/distill_agent_tools/score/sod_stepwise/search_dont_guess/pi_play）、多轮 agentic RL 与 step 级信用分配（search_r1/deepdive/webagent_r1/gigpo/sweet_rl/rstar2/spear/open_agentrl/openclaw_rl/tcod/meow_tea_taro）、内化外部过程（icrl/latent_agents）。这些对应 TSRD 在 agentic 场景下的落地与无-teacher 对照。每篇为完整 12 节，标题已降级。

### chain_of_agents — Chain-of-Agents: End-to-End Agent Foundation Models via Multi-Agent Distillation and Agentic RL

> **一句话重点 (TL;DR)**：把"多智能体协作"压进单一模型——先用 multi-agent distillation 把 SOTA 多智能体系统（OAgents）的执行轨迹转成 CoA 格式做 agentic SFT 冷启动，再用 agentic RL（DAPO）在可验证任务上优化，得到能原生动态激活不同 tool/role agent 的 Agent Foundation Model（AFM）。本质是组合既有组件的大体量工程 recipe，与 token 级 OPD 不同源。

**元信息**：arXiv 2508.13167（v1，2025-08-06；论文标注 Date: 2025-08-20）｜ OPPO AI Agent Team（30+ 作者，通讯 Wangchunshu Zhou）｜ Preprint ｜ 主题：agent foundation model 训练 recipe（与 OPD/distillation 主线关联较弱——这里的 "distillation" 是 agent-level/sequence-level 轨迹 SFT，不是 token-level on-policy 蒸馏）｜ 代码 https://github.com/OPPO-PersonalAI/Agent_Foundation_Models （Apache-2.0，~61MB，模型权重/数据/训练评测代码全开源）｜ 框架 SFT 用 LLaMA-Factory、RL 用 veRL 跑 DAPO

#### 1. 相关工作与进展
- **多智能体系统（MAS）**：靠多个角色/工具的 agent 协作解复杂任务（deep research、vibe coding），性能强但靠人工 prompt/workflow 工程。
- **Tool-Integrated Reasoning（TIR）**：把工具调用显式编进推理（Search-R1、WebThinker），让 LLM 端到端支持 ReAct 式 "think-action-observation"。实证显示 TIR 训练优于纯 prompt 工程的 ReAct agent。
- 但 TIR 只能表达 ReAct 单一轨迹模式，无法端到端训练 LLM 去支持完整 MAS。CoA 想填这个 gap。

#### 2. 现有工作存在的问题
1. 手工多智能体 workflow 计算低效（agent 间冗余通信）、难泛化（换域要重做 prompt/workflow 工程）、不可端到端 data-centric 学习。
2. MAS 里的 backbone LLM 通常静态、没针对 agentic（多轮/多工具/多角色）用法训练。
3. TIR 只支持 ReAct（think-action-observation），轨迹表达能力有限。

#### 3. Motivation
把多智能体协作建模进**单一模型**：让它端到端、原生地动态激活不同 tool agent（search/crawl/code）与 role-playing agent（think/plan/reflect/verify），像 MAS 那样做多轮多工具求解，同时可被 SFT + RL 直接优化；并大幅削减 MAS 的 agent 间通信 token 开销。

#### 4. 主要灵感 / 核心直觉
- 既然 MAS 优于 ReAct，就把 MAS 的"动态角色编排"能力蒸进一个模型，用 system prompt 定义多个 agent、由 Thinking Agent 通过状态转移 S_t = f_θ(S_{t-1}, ϕ_{t-1}, o_{t-1}) 动态激活角色 ϕ_t。
- 蒸馏层级是 **agent-level / sequence-level**（学 MAS 的序列决策模式），而非 word-level 分布——这是它与 token 级 KD/OPD 的本质区别。

#### 5. 主要解决思路（一段话讲清核心）
两阶段 recipe：**(I) Multi-Agent Distillation（agentic SFT 冷启动）**——沿用 Shi et al. 的 agentic 任务生成+过滤，让 SOTA 多智能体系统 OAgents 执行这些任务，把成功的多智能体协作过程录成 CoA 兼容轨迹，经四阶段质量过滤后用 LLaMA-Factory 做 SFT（带 observation masking 防环境噪声反传）。**(II) Agentic RL**——在 SFT 未用过的 QA 上做 tool-aware rollout，用 veRL 跑 DAPO，outcome-driven 二元奖励，只选难题训练。

#### 6. 方法详解（通俗、分步骤）
**CoA 范式**：role-playing agents（Thinking 编排、Plan 分解、Reflection 自评、Verification 校验）+ tool agents（Search、Crawl、Code Generate）；单次解码过程内由 Thinking Agent 动态编排，保持上下文连续、省去 MAS 的 agent 间通信开销。
**阶段 I 数据生成与四阶段渐进质量过滤**：① 复杂度（<5 agent-tool 交互剔除）② 质量（剔错答/冗余/脏数据）③ reflection enrichment（缺反思的下采样）④ error-correction 上采样（search/QA 中带 `<double_check>` agent 纠错的轨迹）。轨迹格式带 observation masking（O 不计入 loss）。
**阶段 II Agentic RL**：tool-aware rollout + DAPO；reward 用 outcome-driven 二元信号（web agent 用 LLM judge M_j 二元判断、免格式奖励；code agent reward = score_answer · score_format）；RL 数据选难题（rq≤0.3）。
**backbone**：Qwen2.5-3B/7B/32B-Instruct。整体是组合既有组件（OAgents 蒸馏 + LLaMA-Factory + veRL/DAPO）的工程 recipe，方法学创新点有限。

#### 7. 实验数据集
近 20 个 agent benchmark。Web/QA：NQ、TriviaQA、PopQA、HotpotQA、2Wiki、MuSiQue、Bamboogle、GAIA、WebWalker、BrowseComp、HLE 等。Code agent：LiveCodeBench、CodeContests 等。数学：AIME25。RL 数据 setting 同 Search-R1。

#### 8. 实验结果与主要发现
- **AFM-32B 多 benchmark SOTA**（Figure 1）：GAIA 55.3%、BrowseComp 11.1%、HLE 18.0%（web agent）；LiveCodeBench v5 47.9%、CodeContests 32.7%（code agent）；AIME2025 59.8%（数学，较此前最佳 TIR 如 ReTool/SimpleTIR 绝对 +10.5%+）。
- **效率**：相比传统多智能体系统，token 消耗减少 84.6%，性能仍有竞争力。
- **7B 版** web agent 上接近用更强 QwQ-32B backbone 的 WebThinker-RL。
- 全开源（模型权重 + 数据 + 训练/评测代码）。

#### 9. 结果如何支撑其主张
- "单模型能像 MAS 一样工作" 由跨 web/code/math 近 20 基准的 SOTA 与 84.6% token 削减支撑。
- "端到端可训练" 由两阶段 recipe 本身 + RL 进一步提升支撑。
- **但 SOTA 主张需谨慎**：多数 SOTA 是在**同 backbone（Qwen2.5）受控对比**下成立；跨 backbone 比较（如对比用更强 QwQ-32B 的 WebThinker-RL）应保留。

#### 10. 逻辑自洽性（中性评估）
作为系统/工程论文整体自洽、结果扎实、开源完整。方法创新有限——核心是把 OAgents 蒸馏成 CoA 轨迹 + LLaMA-Factory SFT + veRL/DAPO 的组合 recipe，每个组件都是既有的。"distillation"一词在此是 agent-level 轨迹 SFT，与本项目 token-level on-policy 蒸馏不同源，关联较弱。

#### 11. 残留问题 / 局限
- SOTA 在跨 backbone 比较下不一定稳健（受控对比多在同 Qwen2.5 backbone）。
- 严重依赖教师 MAS（OAgents）的质量与四阶段过滤的人工启发式（阈值如 <5 交互、rq≤0.3 等）。
- 蒸馏是 sequence-level，无 token 级信用分配；reward 是 outcome 二元信号，过程监督缺失。
- 对本项目（MTP/OPD 方向）参考价值主要在 agentic 数据构造与 observation masking，而非蒸馏机制本身。

#### 12. 开源代码与框架（链接 + 框架 + 代码可得性）
- 仓库：https://github.com/OPPO-PersonalAI/Agent_Foundation_Models （Apache-2.0，~61MB clone）。结构：`AFM/`（data / evaluation / models / tool_servers / train）、`LLaMA-Factory/`（SFT）、`verl/`（RL）、tool servers（web search/crawl + code sandbox）。
- 框架：SFT 用 **LLaMA-Factory**；RL 用 **veRL** 跑 **DAPO**；backbone Qwen2.5-3B/7B/32B-Instruct。代码可得性：模型权重、训练数据、训练 + 评测代码全开源，复现条件好（但需自建 tool servers 与较大算力）。


---


### deepdive — DeepDive: Advancing Deep Search Agents with Knowledge Graphs and Multi-Turn RL

> **一句话重点 (TL;DR)**：DeepDive 用知识图谱（KG）随机游走 + 属性模糊化自动合成"难找"deep-search QA，再用端到端多轮 GRPO（带 redundancy penalty 抑制重复查询）训练浏览 agent，DeepDive-32B 在 BrowseComp 上达 15.3%（KG 数据），加半自动 i.i.d. 数据可达 22.2%。

**元信息**：arXiv 2509.10446（v2 2025-10-14，under review）｜ 清华 / Z.AI / 东北大学（Rui Lu*, Zhenyu Hou*, Zihan Wang* 等；Jie Tang、Yuxiao Dong）｜ 主题 T2（deep search agent / 多轮 RL），与 mtp_opd 外围相关（纯多轮 RL 非蒸馏，可作长 horizon 工具调用 agent RL 对照；redundancy penalty / test-time scaling 对 path 多样性有参考价值）｜ 代码 https://github.com/THUDM/DeepDive（仅 KG 数据合成，**不含 RL 训练代码**）｜ 框架 slime（RL）+ 多轮 GRPO

#### 1. 相关工作与进展
给 LLM 加浏览工具可成为 deep search agent，对数百在线来源推理检索以定位复杂、难找信息（如 BrowseComp 题目）。R1-Searcher / ReSearch / DeepResearcher 等开源工作主要在 HotpotQA 类多跳 QA 上训练评测；OpenAI DeepResearch 等专有系统遥遥领先。GLM-4.5/4.6 采用了 DeepDive 的数据并贡献其 BrowseComp 强表现。

#### 2. 现有工作存在的问题
- **数据太简单**：HotpotQA、2Wiki、Bamboogle、Musique 等靠搜几个清晰实体即可解，不反映真正"hard-to-find"案例；而 BrowseComp 含多个模糊实体（blurry entity）、需长 horizon 推理 + deep search。
- **训练是开放问题**：如何把长 horizon 推理与 deep search 工具使用有效结合仍未解决；即便 DeepSeek-R1 也只做浅层工具调用且易幻觉。

#### 3. Motivation
(1) 自动从开放 **知识图谱** 合成复杂、难找问题，补足难数据；(2) 用端到端多轮 RL 提升"长 horizon 推理 + deep search"；(3) 用 redundancy penalty 鼓励多样、减少冗余重复查询。

#### 4. 主要灵感 / 核心直觉
KG 天然支持多跳连接、每实体多属性——沿 KG 随机游走可抽出长多跳路径，故意模糊部分属性即可制造"blurry entity"，从而构造刺激长 horizon 推理与 deep search 的难题；同时 deep search 中重复相似查询是浪费，可用查询相似度直接惩罚。

#### 5. 主要解决思路(一段话讲清核心)
KG 难数据合成提供训练信号，端到端多轮 GRPO 让 LLM 在真实 web 环境中边搜边推、按最终答案给奖励，redundancy penalty 用 Jaccard 相似度惩罚重复查询以提升搜索多样性与效率。

#### 6. 方法详解(通俗、分步骤)
1. **KG 难数据合成**：先按出度区间 [d_min, d_max] 过滤候选节点（出度过高→答案过于流行可预测；过低→难扩展路径）；在 KG 上随机游走抽长多跳路径；再用 LLM 进一步混淆关键线索形成 blurry entity。i.i.d. side study 随机游走参数：k∈[5,9]、d=3、d_min=4、d_max=8（核自 §4），实体混淆用 Gemini-2.5-Pro。
2. **端到端多轮 RL（multi-turn GRPO）**：LLM 与 web 环境交互，按构造 QA 的最终答案给奖励；为促探索保留 KL penalty。
3. **Redundancy penalty**：用 Jaccard 相似度度量任意两次查询的重复程度并惩罚（系数 λ=0.1），鼓励多样高效搜索（图1左：显著减少 RL 训练中工具调用冗余计数）。

#### 7. 实验数据集
- 评测：**BrowseComp、BrowseComp-ZH、SEAL-0、Xbench-DeepSearch**（四个 deep search benchmark）。
- 训练数据：摘要/引言称构造 **3,090** 条 KG 自动合成 deep search QA；§实验细则述随机游走流程实际产出 **3,250** 条，随机切分为 **1,016 SFT + 2,234 RL**（RL 用全部 2,234）。另含半自动 i.i.d. 合成数据（side study，用 Gemini-2.5-Pro 混淆实体）。〔3,090 vs 3,250 为论文内部叙述口径差异，已核。〕

#### 8. 实验结果与主要发现
- 基模型：GLM-Z1-9B-0414 与 QwQ-32B。SFT 阶段 3 epochs、global batch 32、lr=1e-5、max context 104,800；RL 用 slime、全部 2,234 样本，rollout size=8、每 prompt 16 samples（GRPO），redundancy penalty λ=0.1。训练期 checkpoint 选择用 BrowseComp-266（从 1,266 题随机预采样子集），turn limit 提至 128。
- 结果：DeepDive-32B 在 BrowseComp 达 **15.3%**（KG 数据），超过 WebSailor、Search-o1、DeepSeek-R1-Browse；加半自动 i.i.d. 数据进一步到 **22.2%**。多轮 RL 在四个 benchmark 一致提升（+5.8/+6.7/+1.6/+3.3% 等）。展示工具调用与并行采样的 **test-time scaling**（增大 max tool calls 提升成功率）。做了 n-gram 污染分析（表4）。其数据被 GLM-4.5/4.6 采用。

#### 9. 结果如何支撑其主张
四个 benchmark 一致正增益支撑"多轮 RL 有效"；redundancy penalty 的工具调用计数下降图（图1左）支撑"减少冗余"；test-time scaling 曲线支撑"更多工具调用→更高成功率"；n-gram 污染分析回应数据泄漏质疑，间接支撑结果可信度。

#### 10. 逻辑自洽性(中性评估)
数据合成→RL 训练→多样性惩罚三段逻辑连贯，benchmark 选择（BrowseComp 系）契合"hard-to-find"主张。但部分增益（如 Xbench +1.6%）幅度较小，且 BrowseComp 绝对值仍偏低（15.3%/22.2%），说明任务远未解决；checkpoint 选择用 266 子集存在一定调参–评测耦合风险。

#### 11. 残留问题 / 局限
- 训练数据规模小（RL 仅 2,234 样本），合成数据多样性与难度分布对结果影响未充分隔离。
- BrowseComp 绝对准确率仍低，距专有 DeepResearch 有差距。
- RL 训练代码不在仓内（依赖外部 slime），独立复现需自行接入。
- 3,090 / 3,250 的数据计数口径在论文内不一致（摘要 vs 实验细则），需以实验节为准。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库：https://github.com/THUDM/DeepDive （已 clone，~2.9MB；含 qa_synthetic/（KG 数据合成）、assets/）。**仅含 KG 数据合成代码，不含 RL 训练代码**（训练在外部框架完成）。
- RL 框架：**slime**（THUDM/slime，"LLM post-training framework for RL scaling"，Zhu et al. 2025）；论文 §实验设置明确"we conduct training using the open-source Slime framework with all 2,234 data samples"。多轮 **GRPO** 算法。
- 数据/模型公开：3,090（KG 自动）+ 2,234（RL）+ 半自动 i.i.d. 合成数据；DeepDive 模型。


---


### distill_agent_tools — Distilling LLM Agents into Small Models with Retrieval and Code Tools (Agent Distillation)

> **一句话重点 (TL;DR)**：不只蒸馏教师的"推理"，而是把教师 agent 的完整"think + act（检索/代码工具）"任务求解行为蒸馏到小模型；配两项改进——用 first-thought prefix 提高教师轨迹质量、用 self-consistent action generation 提高学生测试时鲁棒性——使小模型 agent 能匹配甚至超过比它大 2–4× 的 CoT 蒸馏模型。

**元信息**：arXiv 2505.17612 (v2, 2025-11-05) ｜ KAIST / DeepAuto.ai（Minki Kang*, Seanie Lee, Sung Ju Hwang）、UW-Madison（Jongwon Jeong*）、KRAFTON（Jaewoong Cho）｜ NeurIPS 2025 ｜ 主题 T2（agent/tool 蒸馏，离线 SFT 式，非 on-policy，可作 OPD-agent 对照基线）｜ 代码 https://github.com/Nardien/agent-distillation（已克隆约 7.6MB）｜ 框架 smolagents v1.13.0.dev0 + TRL SFT trainer + LoRA；检索环境沿用 Search-R1（pyserini + faiss）。

#### 1. 相关工作与进展
- **CoT 蒸馏**：用教师 LLM 的 Chain-of-Thought trace 通过 next-token 蒸馏把推理能力迁移到小模型（sLM）。
- **检索增强（RAG）**：在蒸馏与推理时引入外部知识，作为公平对比基线。
- **CodeAct 范式**：每步由 Thought–Action（Python 代码）–Observation 组成，本文 agent 形式化沿用之。
- **prefix-attack（jailbreak）**：往模型回答前缀注入内容来引导生成，是 first-thought prefix 的直接灵感来源。
- **majority voting / self-consistency**：对多次采样结果投票提升鲁棒性。

#### 2. 现有工作存在的问题
- 纯 CoT 蒸馏在需要**罕见事实知识**或**精确计算**时失效——小模型容易幻觉、算错（如"$100 投资 Apple 2010→2020 值多少"需查历史股价、拆股，再精确计算）。
- 静态推理 trace 模仿无法在测试时获取新知识或保证计算正确。
- 直接让 instruction-tuned 教师（Qwen2.5-32B-Instruct）当 agent 时，其**初始推理质量不佳**，生成的轨迹不够好。
- 小蒸馏 agent 常生成**无效/不可解析的动作**（代码报错、库函数误用），阻碍与环境交互。

#### 3. Motivation
迁移的不应只是"推理"，而是 LLM agent 的**完整任务求解行为**（think + act）：让小模型学会用检索工具获取事实、用代码工具精确计算，从而对幻觉更鲁棒、对 OOD 任务泛化更强。核心问题：如何在更小模型中保留 LLM 级问题求解能力。

#### 4. 主要灵感 / 核心直觉
- **教师轨迹质量瓶颈在"第一步思考"**：instruction-tuned 教师直接做 agent 时开头推理弱；借 prefix-attack 思路，先用 CoT prompt 诱导教师生成 step-by-step 推理，把这"第一步思考"作为前缀注入教师 agent → 抬高整条轨迹质量。学生推理时不需要该前缀。
- **学生测试时靠采样+投票纠错**：对每步采多条 thought-action（高温 nucleus 采样增多样性），无效动作可由其报错 observation 引导自纠，并对结果 observation 做 majority voting，降低单次无效动作的影响。

#### 5. 主要解决思路（一段话讲清核心）
用 first-thought prefix 增强后的教师（Qwen2.5-32B-Instruct）在 smolagents 中对训练问题各采一条"Thought-Action(code/search)-Observation"轨迹，过滤掉答错的，得到约 2,000 条轨迹；用 TRL 的 SFT trainer + LoRA 把这些 agent 轨迹蒸馏进 Qwen2.5-Instruct 小模型（0.5B/1.5B/3B/7B）；测试时小 agent 配 self-consistent action generation（每步采 N=8、温度 0.4，投票），得到实用的 tool-using 小 agent。

#### 6. 方法详解（通俗、分步骤）
**Agent Distillation 框架，两条互补改进轴：**
1. **First-thought prefix (ftp)**：用 CoT prompt 诱导教师产生 step-by-step 推理，截取其作为前缀注入教师 agent 的"第一步思考"，再让教师在 smolagents 中续走 reason-act-observe。只用于改善教师轨迹采集，学生推理时不需要。
2. **Self-consistent action generation (sag)**：测试时不用 greedy，对每一步用 nucleus 采样以高温采 N 条 thought-action（主实验 N=8、温度 0.4）；无效动作的报错 observation 反馈回去供后续步自纠；对各条得到的 observation 做 majority voting 选最终动作结果。

**流程**：教师=Qwen2.5-32B-Instruct；学生=Qwen2.5-Instruct 四个尺寸（0.5B/1.5B/3B/7B，蒸馏前已 instruction-tuned）。每题从教师采 1 条轨迹并过滤错误轨迹 → 约 2,000 条训练数据 → LoRA（rank 64，所有线性层）SFT 学生（2 epoch、batch 8、lr 2e-4、4×A100 80GB）→ 测试时学生 agent 配 sag、max steps=5、greedy 主解码。

#### 7. 实验数据集
- **训练数据**：1,000 HotPotQA + 2,000 MATH examples（每题采 1 条教师轨迹、过滤错误后约 2,000 条用于蒸馏）。〔已核：训练源含 HotPotQA 与 MATH 两部分，非仅 MATH〕
- **检索环境**：Wikipedia 2018 作知识库（agent 与 RAG 共用），e5-base-v2 做 doc/query embedding；沿用 Search-R1 的 retriever。
- **评测（8 任务，每集限 500 例）**：
  - 事实多跳 QA：HotpotQA（in-domain）、MuSiQue、Bamboogle、2WikiMultiHopQA（OOD）。
  - 数学：MATH500（in-domain）、GSM-Hard、AIME、OlympiadMATH（OOD）。
  - 指标：数学用 exact match；事实用 LLM-as-judge（gpt-4o-mini）。

#### 8. 实验结果与主要发现
- **agent 蒸馏全面优于 CoT 蒸馏**，尤其在 OOD 任务上。蒸馏前除 7B 外多数尺寸靠 prompting 无法产生有效 agentic 输出（常出不可解析代码）。
- **跨档匹配**（论文主张）：0.5B agent ≈ 1.5B CoT 蒸馏；1.5B agent ≈ 3B CoT；3B agent 超过 7B CoT；7B agent 甚至超过 32B CoT 教师。
- **逐项数值（Table 2 摘选，Avg. 为 8 任务均值）**：
  - 教师 32B：CoT Prompting 39.54；Agent Prompting 46.00。
  - 7B 学生：CoT Distill 33.54 → Agent Distill 39.85 → +ftp 42.26 → +sag 41.86 → +ftpsag 42.68。
  - 3B 学生：CoT Distill 27.72 → Agent Distill 33.60 → +ftpsag 36.60。
- ftp 与 sag 在 0.5B–7B 全尺寸、跨两域基本稳定提升（但 3B+sag 在 AIME 上掉到 0.0，说明 sag 单用并非处处增益）。

#### 9. 结果如何支撑其主张
- "迁移工具使用行为提升泛化" → agent 蒸馏在 4 个 OOD 任务上相对 CoT 蒸馏增益最明显，与"测试时可查事实/算数→对幻觉更鲁棒"的机制一致。
- "小模型匹配更大 CoT 模型" → 表中 7B Agent(ftpsag) 42.68 > 32B CoT 39.54，3B Agent(ftpsag) 36.60 > 7B CoT 33.54，支撑跨档主张。
- ftp/sag 的增量贡献由"Distill → +ftp/+sag/+ftpsag"逐行递增体现，但并非每个单元格都单调（sag 在个别难任务上可能反伤），说明组件增益是平均意义上的。

#### 10. 逻辑自洽性（中性评估）
框架自洽、实现干净：ftp 只动教师数据采集、sag 只动学生测试，互不耦合，消融清晰。但"跨档匹配"这种比较高度依赖平均分聚合（8 个异质任务直接平均），单任务上结论不稳（如 AIME 这类小样本基准方差大、出现 0.0/15.6 的剧烈跳变）。sag 用 majority voting 实质是 test-time 多次采样，提升来自额外推理算力而非模型本身能力——与"高效小 agent"的卖点存在一定张力（推理成本被转移到 test-time）。整体是合理的工程组合，机制解释成立但增益的统计稳健性有限。

#### 11. 残留问题 / 局限
- **离线 SFT、非 on-policy**：纯模仿教师轨迹，未做 on-policy/RL 矫正，学生在工具调用上的分布漂移未被训练时纠正。
- **sag 的 test-time 成本**：每步采 N=8 条，推理开销显著上升，"小模型省算力"的优势部分被抵消。
- **小样本基准方差大**：AIME/OlympiadMATH 等任务样本少，Avg. 聚合掩盖了单任务的剧烈波动（如 3B+sag AIME=0.0）。
- **依赖外部判分**：事实任务用 gpt-4o-mini 做 LLM-as-judge，引入评测器偏差。
- **教师/检索环境固定**：仅 Wikipedia 2018 + 单一 retriever，真实开放检索下的表现未验证。

#### 12. 开源代码与框架（链接+框架+代码可得性）
- 仓库 https://github.com/Nardien/agent-distillation（已克隆，约 7.6MB）。顶层含 `src/smolagents/`（fork 的 smolagents）、`data_processor/{math_dataset,qa_dataset}`、`exps_research/first_thought_prefix`、`scripts/{training,inference}`、`search/`、`serve_vllm.py`、`e2b.toml`。
- 框架：基于 **smolagents v1.13.0.dev0**（logging→可训练轨迹 / training / benchmarking）；训练用 **TRL** SFT trainer + **LoRA**（rank 64）；检索沿用 **Search-R1**（pyserini + faiss-gpu）；代码工具走本地或 e2b 沙箱。
- 资产：HF `agent-distillation/*` 提供教师轨迹（2k baseline 与 first-thought-prefix 两版）与 agent-distilled Qwen2.5-1.5B-Instruct 等模型。
- 代码可得性：高（框架、数据处理、训练/推理脚本、轨迹与模型权重齐全；README TODO 标注 ftp 详细说明待补）。


---


### gigpo — Group-in-Group Policy Optimization for LLM Agent Training (GiGPO)

> **一句话重点 (TL;DR)**：把 GRPO 扩展到长 horizon 多轮 agent——除了像 GRPO 那样按整条轨迹回报算 episode 级相对优势,再利用"组内轨迹常反复经过相同环境状态"这一观察,把同一状态下的不同动作聚成 step 级组算 micro 相对优势,从而**无需额外 rollout、无 critic** 就实现细粒度 step 级信用分配,额外时间成本 <0.002%。

**元信息**：arXiv:2505.10978（v3, 2025-10-28, cs.LG；NeurIPS 2025）｜ 南洋理工 NTU / Skywork AI Singapore（Lang Feng、Zhenghai Xue、Tingcong Liu、Bo An）｜ 主题 多轮 LLM agent 的细粒度 credit assignment / 相关性中（step 级信用分配机制可作 path-level 监督的 critic-free 替代/baseline；但本文纯 RL,无 teacher 蒸馏、无 MTP）｜ 代码 github.com/langfengQ/verl-agent（Apache-2.0，本地已 clone）｜ 框架 verl-agent（veRL 扩展）。

#### 1. 相关工作与进展
- **group-based RL(RLOO、GRPO)**:在单轮任务(数学、代码)上很成功——奖励即时、信用分配简单、无需 critic、低显存、稳定。
- **多轮 agent RL**:LLM agent 在外部环境中是长 episode(数十步、数万 token)、奖励稀疏/延迟。
- **已有 step 级信用方案**:actor-critic(PPO、ArCHer、AgentQ)用 value 网络/MCTS;或对每个 state 额外 rollout 新动作。

#### 2. 现有工作存在的问题
- **朴素套 group-based RL 到多轮**:把整条轨迹当一个 episode-level 响应(如 RAGEN),组内每步给**同一** advantage,丧失 step 级区分,长 horizon(如 ALFWorld 50 步)下扩展性差。
- **naive step 级信用**:对每个 state 额外 rollout 动作 → 计算代价爆炸。
- **actor-critic**:需额外 value 网络/MCTS,复杂、开销大,失去 group-based RL 的简洁。
- **核心矛盾**:能否**既保留** group-based RL 的优点(critic-free、低显存、稳定)**又引入**细粒度信用分配?

#### 3. Motivation
长 horizon 任务里,一个动作的好坏可能很晚才显现,而 episode 级单一 advantage 无法告诉模型"是哪一步走对/走错了"。需要在不增加 rollout、不引入 critic 的前提下,给出 step 级的相对好坏信号。

#### 4. 主要灵感 / 核心直觉
关键观察:**同任务同初始条件下,组内许多轨迹因无效动作或循环会反复遇到相同环境状态**(同一网页/房间/游戏场景)。这些共享状态天然就是"对照实验"——在同一状态下不同轨迹采取了不同动作并得到不同后续回报,于是**不用额外 rollout 就能直接比较这些动作的优劣**,做局部(step 级)信用分配。

#### 5. 主要解决思路(一段话讲清核心)
GiGPO 嵌套两级相对优势(group-in-group):宏观上,对同一任务采 N 条完整轨迹,按总回报算 episode 级相对优势 A^E(同 vanilla GRPO);微观上,回溯找出组内反复出现的环境状态(anchor state),把"在同一 anchor state 下采取的不同动作"聚成一个 step 级组,组内按各动作的**折扣回报**算 step 级相对优势 A^S;最后线性合并 A = A^E + ω·A^S(ω 直接设 1,不调参)。全程 critic-free、无额外 rollout、与原 GRPO 同显存。

#### 6. 方法详解(通俗、分步骤)
- **Episode 级(宏观)**:同任务同初始状态采 N 条轨迹,按总回报算 macro 相对优势 A^E,反映整条轨迹的任务完成质量。
- **Step 级(微观)—— Anchor State Grouping(核心创新)**:回溯识别组内跨轨迹/跨时间重复出现的环境状态(anchor state);把所有"在同一 anchor state 下采取的不同动作"聚成一个 step-level group,每个唯一状态构一组。组内对动作算 micro 相对优势 A^S。为捕捉长期影响,对每步关联**折扣回报** R_t(折扣因子 γ∈(0,1]),使早期次优动作得到比后期正确动作更低的折扣回报,产生清晰的偏好排序(例:1st Item > 2nd Item > Next Page)。
- **合并**:A = A^E + ω·A^S,ω=1(代码 `step_advantage_w` 默认 1.0,未调参)。
- **变体**:GiGPO w/ std 与 w/o std(是否用标准差归一化);**similarity-based 变体**——状态难精确匹配(如 QA)时,用最长匹配子序列相似度阈值(默认 0.95)判定"是否同一状态"。

#### 7. 实验数据集
- **长 horizon agent**:ALFWorld(具身家务规划)、WebShop(目标驱动 web 交互)。
- **多轮工具集成推理**:search-augmented QA(沿用 Search-R1 设置,E5 检索器,max turn=4)。
- **base model**:Qwen2.5-1.5B/3B/7B-Instruct。

#### 8. 实验结果与主要发现
- **较 GRPO 一致大幅提升**:GiGPO w/o std 在 1.5B 上 ALFWorld +13.3%、WebShop +10.6%;7B 上 +12.6% / +9.1%;QA 3B 42.1%、7B 47.2%。w/ 与 w/o std 均稳超 GRPO 与 RLOO。
- **几乎零额外成本**:GiGPO 特有操作(anchor 分组 + 算 step 优势)单次迭代仅约 0.53s,占总训练时间 **<0.002%**,且与原 GRPO 同显存、同 rollout 预算。
- 报告观察到 emergent reasoning(附录 F)。
- 对照闭源(GPT-4o、Gemini-2.5-Pro)、prompting(ReAct、Reflexion)、RL(PPO/RLOO/GRPO);QA 另比 Search-R1、ZeroSearch、StepSearch 等。

#### 9. 结果如何支撑其主张
主张是"在保持 group-based RL 效率的同时引入细粒度信用分配"。两点证据:(1)在三类任务、三种规模上一致超过 GRPO/RLOO,且 w/ 和 w/o std 都成立,说明增益稳健来自机制而非调参;(2)额外时间 <0.002%、同显存,直接支撑"不牺牲效率"。代码核对(`core_gigpo.py`)与论文一致:`compute_gigpo_outcome_advantage` 中 `scores = episode_advantages + step_advantage_w * step_advantages`(默认 1.0),anchor 分组、折扣回报、similarity 变体均实现到位。

#### 10. 逻辑自洽性(中性评估)
机制自洽且实现可核(本地代码已审计,与论文 Eq.3/6/7/8 对应)。需保留的批判点:(1)**核心前提是"组内状态确实重复"**——若任务状态空间大、轨迹很少撞到同一状态,step 级组多为 size=1(代码 `summarize_group_size` 正反映这一动态),A^S 退化,增益将缩水;similarity 阈值近似缓解但引入新超参与误聚风险;(2)ω=1 未调参虽显鲁棒,但也意味着两级优势的相对权重未被优化,可能非最优;(3)折扣回报需要环境给出 per-step reward 或可定义,纯稀疏终局奖励下 step 级信号主要靠折扣传播,效果依赖 γ。

#### 11. 残留问题 / 局限
- **状态重复假设**:在状态难精确匹配/极少重复的环境中机制退化,论文用相似度阈值兜底但未给失效边界。
- **纯 outcome/环境奖励驱动**:无外部监督或蒸馏,增量集中在 step 级信用分配机制本身,属 GRPO 家族增量改进而非新范式。
- **ω、γ、similarity_thresh** 等超参的系统敏感性分析有限(ω 固定为 1)。
- **评测局限于 ALFWorld/WebShop/QA**,对更开放的真实工具环境(代码执行、长程网页)未验证。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 代码 github.com/langfengQ/verl-agent(Apache-2.0,含 HF 模型 collection);本地已 clone,核心算法在 `gigpo/core_gigpo.py`(已审计)。
- **框架**:`verl-agent` 是 **veRL** 的扩展(pyproject 包名仍为 `verl`)。核心改造是 **step-independent multi-turn rollout**——不简单拼接完整交互历史,而是允许逐步可定制的 per-step 输入结构、history 管理与 memory 模块,支持超长 horizon(ALFWorld 可达 50 步)。
- **配置**:base Qwen2.5-1.5B/3B/7B-Instruct;ALFWorld/WebShop 所有 RL 方法用完全相同超参,rollout group size N=8;QA 用 N=5、max turn=4、E5 检索器;ω=1 不调。


---


### icrl — ICRL: Learning to Internalize Self-Critique with Reinforcement Learning

> **一句话重点 (TL;DR)**：让同一 backbone 用 role-specific prompt 同时当 solver 与 critic 联合 RL，把"有 critique 才能做对"内化为"无 critique 也能做对"——核心是一个把 critique-conditioned 修订轨迹按 token 级 re-weight 比率 `w_t=π(y_t|q)/π(y_t|q,c)` 迁移到 critique-free 分布的"分布校准"，外加逐角色组内归一化以稳定联合优化。

**元信息**：arXiv 2605.15224v1（2026-05-13）｜ 港科大(GZ)、南京大学、中山大学、NUS、NTU、SAP、Microsoft Research（多机构）｜ Preprint，2026-05｜ 主题 solver-critic 联合 RL / 自我改进，与 OPD/TSRD 间接相关（"critique-conditioned→critique-free 的 token 级 re-weight"与 OPD"教师条件分布→学生无条件分布"形式同构）；不涉及 MTP，非 teacher-student 蒸馏｜ 代码 https://github.com/brick-pid/ICRL ｜ 框架 slime (THUDM, SGLang-native RL) + AgentGym。

#### 1. 相关工作与进展
外部/自我 critique（Self-Refine、Reflexion、CRITIC）常能引导同一模型纠错成功；Critique-GRPO 等用冻结 critic 做 critique-based RL。GRPO 依赖同前缀同分布的组内比较。

#### 2. 现有工作存在的问题
- 撤掉 critique 后模型在同一 query 上又失败——能力没被内化。
- 冻结 critic 无法随训练提升反馈质量，solver 进步后 critic 过时、产无关/冗余反馈。
- critique-guided 成功轨迹来自 critique-conditioned 分布 (y|q,c)，直接当 (y|q) 更新会产生有偏估计，强化"依赖 critique"。
- solver 初解、critic 批评、修订解的 prompt 前缀各异，奖励不可直接组内比较。

#### 3. Motivation
把 critique 引发的成功转化为无 critique 的 solver 能力（内化），同时让 critic 与 solver 共享 backbone、协同进化使批评质量可学习，并解决联合优化的稳定性。

#### 4. 主要灵感 / 核心直觉
修订轨迹虽在 critique 上下文下产生，但其中一部分 token 在 critique-free 分布下本就可信——只把这部分迁移给 solver、下调重度依赖 critique 上下文的 token，即可内化而不强化依赖。critic 的价值应由"实际有用的反馈"（带来奖励提升）而非"听起来合理的反馈"来定义。

#### 5. 主要解决思路(一段话讲清核心)
solver 采初解，失败则 critic 产 critique、solver 据此产修订解（最多 K=2 轮）；对修订轨迹去掉 critique 前缀、重视作 (y|q) 的证据，引入 token 级 re-weight 比率 `w_t=π(y_t|q)/π(y_t|q,c)`（上界 w_max=2）选择性迁移可信 token；solver 组与 critic 组分别做组内归一化（role-wise group advantage）保留各自学习信号；critic 奖励取修订成功记 1 否则等于 solver 奖励的时序提升 r(τᵢ₊₁)−r(τᵢ)。

#### 6. 方法详解(通俗、分步骤)
- **Self-Improving Workflow**：solver 采初解 τ₁；失败→critic 产 critique cᵢ→solver 产修订解 τᵢ₊₁；最多 K 轮（实现 K=2，论文 "set the iteration round to 2 to improve training efficiency"）。
- **奖励**：solver 用任务 outcome reward r(τ)∈[0,1]；critic reward=修订成功记 1，否则 r(τᵢ₊₁)−r(τᵢ)（仅 dense reward 下非零）。
- **Critique-Conditioned Distribution Calibration（核心）**：token 级 `w_t=π(y_t|q)/π(y_t|q,c)`，只迁移 critique-free 下本就可信的 token、下调依赖 critique 上下文的 token。
- **Role-wise Group Advantage（Eq.5）**：solver 组与 critic 组分别组内归一化。
- **目标（Eq.6）**：`J=E[ min(w_t,w_max)·min(ρ_t(θ)Â, clip(ρ_t,1∓ε)Â) ]`，仅 critique-guided 修订 solver 轨迹用 w_t（其余 w_t=1），**w_max=2** 防梯度方差爆炸。
- 〔已核-代码〕`icrl/icrl/rewards.py::_role_group_key/_role_sample_norm` 按 `(group_index, role)` 逐角色组内归一化实现 Eq.5；`generate.py:218-225` 对修订轨迹存 `critic_free_prompt_ids`、把 `exec_sample.tokens` 重绑为"critique-free 前缀+原 response"，w_t 在 slime 损失阶段重算。K=2、w_max=2、env_nums=32 在 `hydra_conf/` 确认。

#### 7. 实验数据集
四类任务：(1) Text world ALFWorld；(2) Web 导航 WebShop；(3) 多跳 QA（RAG，HotpotQA/2WikiMultiHop/Bamboogle/MuSiQue，合称 SearchQA）；(4) 数学 MATH500/Minerva/OlympiadBench/AIME24/AMC23。Backbone：Qwen3-4B、Qwen3-8B。算力：agentic/Math 4B 用 2×H100、8B 4×H100；SearchQA 4B 4×H100、8B 8×H100。

#### 8. 实验结果与主要发现
- agentic（avg.）：4B **57.0**（vs GRPO 49.2 **+7.8**、vs Critique-GRPO 55.9 **+1.1**）；8B **57.8**（vs GRPO 52.8 **+5.0**、vs Critique-GRPO 56.6 **+1.2**）。
- 数学 8B 平均 **75.3%**（vs GRPO 68.3 +7.0、vs Critique-GRPO 73.3 +2.0；AIME24 50.0→65.1）。
- 摘要 "6.4 points over GRPO on agentic / 7.0 on math" 为跨 backbone 平均口径；逐 backbone agentic 为 +7.8(4B)/+5.0(8B)。
- 消融：去 role-wise 归一化 69.8→68.4；去 re-weight ratio 69.8→67.8（均正贡献，增量适中）。
- test-time 多轮 refinement：ALFWorld 第三轮达 **98%**。critic-swap：学到的 8B critic 在 ALFWorld 约 57 token（WebShop 93.9 token）即匹配/超过 20B/32B 冻结 critic（后者数百 token）。

#### 9. 结果如何支撑其主张
"内化"由 critique-free 评测下相对 GRPO 的提升支撑；"分布校准/role-wise 归一化有用"由两项消融的正贡献支撑；"critic 可学习且高效"由 critic-swap 实验（小 critic 用极少 token 匹配大冻结 critic）支撑。但相对最直接对手 Critique-GRPO 的领先仅 +1.1~1.2（agentic），优势有限。

#### 10. 逻辑自洽性(中性评估)
方法、公式、代码三者一致，消融定向支撑各组件。但增益的主要来源需谨慎归因：相对 GRPO 的大幅提升中，很大一部分来自 critique-based 范式本身（Critique-GRPO 已拿到大半），ICRL 的两项增量（分布校准 + role-wise 归一化）贡献适中而非数量级。

#### 11. 残留问题 / 局限
- 相对 Critique-GRPO 领先有限（agentic +1.1~1.2），核心增益与"是否用 critique 范式"耦合。
- 作者自陈：依赖 critique 做修订，长尾轨迹在同步 RL 下 rollout 慢、可能成吞吐瓶颈；异步训练未探索。
- critic 时序提升奖励仅在 dense reward 下非零，sparse 任务下 critic 信号弱。
- Preprint（2026-05），未评审。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 链接：https://github.com/brick-pid/ICRL 。主体在 `ICRL/` 子目录，README 顶层称 "will host the official implementation"，但实际已含完整训练代码（`train.py`/`train_async.py`、`icrl/`、数据 `data/agentgym`、`data/math`、`data/criticgrpo`）。
- 框架：slime（THUDM SGLang-native RL，内置 `slime/`+`slime_plugins/`）；agentic 环境经 AgentGym 起多 server（默认 32 并行 rollout）；数学沿用 Critique-GRPO 设置；RL 原语 GRPO。
- 代码可得、关键实现（role 分组、critique-free 重绑、超参 hydra 配置）可对照。


---


### latent_agents — Latent Agents: A Post-Training Procedure for Internalized Multi-Agent Debate

> **一句话重点 (TL;DR)**：用"SFT 学辩论结构 + GRPO 把显式辩论压进潜空间"的两阶段微调，把多个 agent 多轮辩论（multi-agent debate）内化进单个 LLM（IMAD），用 Debate 6.3%~21.1% 的 token（5–16× 提效）匹配/超过显式辩论；并发现内化后存在线性可分的"agent 子空间"，可做行为控制。属概念验证级（944 条算术 trace、LoRA）。

**元信息**：arXiv 2604.24881v1（2026-04-27）｜ Boston University（John Seon Keun Yi, Aaron Mueller, Dokyun Lee）｜ ACL 2026 Oral（据 README；论文 PDF 内未见该字样，〔待核-以 README 为准〕）｜ 主题 多智能体辩论的内化/蒸馏，与 TSRD/OPD 间接相关（"内化昂贵外部过程进单模型"的 SFT→RL 两阶段范式、length-annealing 把显式长推理压成隐式潜推理）；不涉及 MTP，非 teacher-student logit 蒸馏，奖励为 outcome+format 非 path 监督｜ 代码 https://github.com/johnsk95/latent_agents ｜ 框架 TRL (GRPOTrainer) + PEFT/LoRA。

#### 1. 相关工作与进展
Multi-agent debate（Du et al. 2023; Liang et al. 2024）通过多模型多轮对话降幻觉、提升事实准确性。DebateGPT（Subramaniam et al. 2024）仅用最终 consensus 输出蒸馏。length-pruning 借鉴 ThinkPrune（Hou et al. 2025，代码注释明示）。

#### 2. 现有工作存在的问题
- 显式辩论 token 开销巨大（多模型多轮 transcript 才给答案）。
- 仅蒸馏最终 consensus（DebateGPT）token 最省但性能普遍不如显式辩论，因丢掉了驱动增益的中间交互。
- 缺乏对"内化后辩论结构是否仍保留、能否被控制"的机制性理解。

#### 3. Motivation
用一次性微调投入，换单模型的效率 + 多智能体辩论的推理能力；并探究内化后模型是否学到可恢复的"每个 agent 的表示"，进而用于行为控制（如抑制恶意 agent）。

#### 4. 主要灵感 / 核心直觉
辩论的增益来自中间多视角交互而非仅最终结论，所以要在完整 trace 上学结构；学会结构后，用"格式奖励衰减 + 长度上限收缩"逼模型把多视角分析从显式文本转入潜空间，直接产答案。内化后不同 agent 的"声音"应在表示空间留下线性可分的痕迹。

#### 5. 主要解决思路(一段话讲清核心)
三阶段 IMAD：(1) 用标准 multi-agent debate（n=3 agents, m=2 rounds, GPT-3.5-turbo 当 agent）在算术题上生成 944 条带结构标签的辩论 trace；(2) 在完整 trace 上做 next-token CE 学辩论格式（SFT）；(3) GRPO 内化，奖励 `r=w_fmt·R_fmt+w_clip·R(y;l)`，R_fmt 为结构标签匹配的格式奖励（权重 w_fmt 随训练 1.0→0.05 衰减），R(y;l) 为"正确答案出现在前 l token 内记 1"的长度裁剪奖励（l 随训练 2000→500 退火），两者协同迫使模型把分析压进潜空间。

#### 6. 方法详解(通俗、分步骤)
1. **数据收集**：标准 debate（n=3, m=2, GPT-3.5-turbo）在 6 个两位数表达式算术题上生成 transcript；过滤无 majority consensus 的；加结构标签 `<|Agent 1|>`/`<|Round 1|>`/`<|Consensus|>`/`<|endofdebate|>`；共 **944** 条 {Question, Trace, Answer}。
2. **Debate Structure Learning（SFT）**：在完整辩论 trace（非仅最终输出）上做自回归 CE。
3. **RL for Internalization（GRPO）**：`r=w_fmt·R_fmt+w_clip·R(y;l)`；w_fmt 1.0→0.05 衰减、l 2000→500 退火，把多视角分析转入潜空间直接产答案。
4. **机制分析**：用 difference-in-means 提取 agent-specific steering vector，发现内化产生线性可分的 agent 子空间；对恶意 agent 子空间做 negative steering 可在保任务性能下抑制有害行为，且内化后比直接 steer base model 更有效。
- 两阶段均用 LoRA；GRPO 阶段从 SFT 的 LoRA checkpoint 起再叠一层 LoRA。

#### 7. 实验数据集
- 训练：仅 944 条算术题辩论 trace。
- 评测：GSM8K（多步数学）、MMLU-Pro（多领域多选）、BigBench Hard（多样推理）；各随机采 1000 题，三次运行报均值±标准误。训练仅算术、评测跨域跨格式，测泛化。

#### 8. 实验结果与主要发现
- **效率**：所有模型上 IMAD 仅用 Debate 的 **6.3%~21.1%** token（5–16× 提效）。
- LLaMA-3.1-8B-Instruct 上三个 benchmark 全面超 Debate；Mistral-Nemo-12B 在 GSM8K 上超 Debate **18.97** 个百分点；Qwen2.5-7B 增益温和。
- 仅算术训练却能跨域泛化（附录另做多任务扩展数据集，性能更强）。
- DebateGPT（仅最终输出蒸馏）token 最省但性能不及 IMAD，印证保留中间交互的价值。

#### 9. 结果如何支撑其主张
"内化保性能+提效"由三 benchmark 上 IMAD 以极少 token 匹配/超 Debate 支撑；"中间交互重要"由 IMAD 超 DebateGPT 支撑；"agent 子空间可控"由 difference-in-means steering 实验（negative steering 抑制恶意 agent、内化后比 base 更有效）支撑。

#### 10. 逻辑自洽性(中性评估)
两阶段范式与奖励设计逻辑清晰，机制实验为"内化"提供了表示层证据。但整体属概念验证：训练数据极小（944 条、仅算术、n=3/m=2 最小辩论），增益跨 backbone 差异大（Qwen 仅"温和"），机制结论主要靠 steering 实验单一证据链支撑，泛化主张依赖附录扩展数据。

#### 11. 残留问题 / 局限
- 训练数据与规模都很小（944 条算术 trace、LoRA、n=3/m=2），结论可扩展性存疑。
- 增益在不同 backbone 间差异显著（LLaMA/Mistral 强、Qwen 弱），未充分解释。
- 教师辩论用 GPT-3.5-turbo，质量天花板受限；机制结论（agent 子空间可控）证据链较窄。
- ACL 2026 Oral 据 README，论文 PDF 内未见该字样，〔待核〕。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 链接：https://github.com/johnsk95/latent_agents （README 写 ACL 2026 Oral）。含 `sft.py`、`grpo.py`、`grpo_persona.py`、`steering/`、`eval/`、`data/`、`utils/generate_arithmetic*.py`。
- 框架：TRL（`GRPOConfig`/`GRPOTrainer`，trl>=0.11）+ PEFT/LoRA（peft>=0.13）+ transformers>=4.46 + accelerate；length-pruning 注释借鉴 ThinkPrune。
- Backbone：LLaMA-3.1-8B-Instruct、Qwen2.5-7B、Mistral-Nemo-12B。SFT 3–6 epoch、GRPO 2 epoch。代码可得、流程可复现。


---


### meow_tea_taro — A Practitioner's Guide to Multi-turn Agentic Reinforcement Learning

> **一句话重点 (TL;DR)**：把多轮 agentic RL 的设计空间拆成 environment/reward/policy 三支柱做系统受控消融，得出一份可操作配方——"课程(由简到繁) + 稳定化偏置策略(PPO/GRPO 优于无偏 RLOO 与朴素 REINFORCE++) + 验证型稠密奖励(单测通过率远胜模型评判)"。非新算法，是经验研究；框架封装 veRL。

**元信息**：arXiv 2510.01132 (v2, 2025-12-06) ｜ UC San Diego(+ NVIDIA)；Ruiyi Wang, Prithviraj Ammanabrolu ｜ Preprint, under review ｜ 主题 多轮 agentic RL 实证(非新算法)/与 OPD 主线弱相关 ｜ 代码 https://github.com/pearls-lab/meow-tea-taro（Apache-2.0，本地已 clone ~11MB）｜ 框架 封装/vendoring veRL

#### 1. 相关工作与进展
单轮 RL(PPO、RLOO、GRPO、DAPO)已为即时响应质量做了大量优化，但它们假设奖励直接跟随单个 action；多轮交互环境只在长序列交互后才揭示结果，打破了单轮方法依赖的 action-reward 耦合。现有多轮 RL 工作进展有限：有的把单轮 QA 交错工具/推理步骤伪装成"多轮"，有的在真交互环境里只用稀疏终局奖励、或把 turn-level advantage 均匀摊到所有 token(无细粒度 credit assignment)。仓库名 "meow-tea-taro" 是 "Multi-turn" 的谐音梗。

#### 2. 现有工作存在的问题
- 多轮 RL 框架与定义碎片化，各家结果不可比，对"什么是真多轮 vs 伪多轮"存在混淆。
- 缺乏对 environment/reward/policy 三者如何共同决定多轮性能的系统理解。
- 固定预算下 SFT:RL 最优配比未知；reward 稀疏性 × RL 算法的交互缺系统消融。

#### 3. Motivation
回答"让多轮 agentic RL 真正 work 的因素实际是什么"。把设计空间拆成三支柱逐一受控消融，在情境化文本域(TextWorld/ALFWorld)与软件工程(SWE-Gym)上导出可操作 recipe，并区分增益来自"多轮 formulation"本身还是算法启发式。

#### 4. 主要灵感 / 核心直觉
把多轮 agentic 任务形式化为 POMDP；agentic 环境只在命令完成(<eos>)时执行并给奖励，故标量奖励 r_t 赋到该 turn 的 <eos> token、其余 action token 奖励为 0、state token 全 mask 不计 loss。对有 advantage 估计的算法(PPO)用 token-level credit assignment + GAE，使前序 token 经 value bootstrapping 获得非零 advantage。隔离"多轮 formulation"与"算法启发式"的方法：对比偏置(PPO/GRPO/REINFORCE++)与无偏(RLOO)策略梯度——若无偏的 RLOO 也涨，则增益来自 formulation 而非 PPO 启发式。

#### 5. 主要解决思路(一段话讲清核心)
非新算法，是三支柱受控实证：(1) Environment——沿 world size/object/quest 三维扫复杂度、测简单→复杂泛化、测任务多样性；(2) Policy——SFT 先验对 RL 收敛的影响、固定预算下 SFT:RL 最优配比、偏置 vs 无偏算法对比；(3) Reward——稀疏 vs 稠密 turn-level 奖励、验证型 vs 模型型奖励、奖励粒度。最终汇成 recipe：课程 + 稳定化偏置策略 + 验证型稠密奖励。

#### 6. 方法详解(通俗、分步骤)
- **POMDP 形式化**：history h_t=(u,s_0,a_0,…,s_t)，action 是自然语言 token 序列，env.step 返回 next state/reward/done，reward 赋到 <|im_end|>。
- **Multi-turn PPO**：token-level TD δ + GAE 估每 token advantage；Clipped Surrogate 覆盖全轨迹 token。
- **Environment 消融**：TextWorld 程序化生成 w/o/q 三维(如 w2-o3-q4)，ALFWorld 6 类家务，SWE-Gym 5 类(getmoto/pydantic/mypy/pandas/dvc)；改环境复杂度 + 任务多样性混合，固定总数据量公平比较。智能体须从观察自生成可执行 NL 命令(无 admissible action 提示)。
- **Policy 消融**：TextWorld 金标解作 SFT 示范(与 RL 数据不同 seed 防泄漏)；固定 1000 成本单位(假设 SFT 数据贵 RL 的 10×)扫 SFT:RL 配比；对比 PPO/GRPO/RLOO/REINFORCE++。
- **Reward 消融**：TextWorld 用内置函数造 sparse/dense(steps-per-reward)；SWE-Gym 比验证奖励(二元 vs 单测通过比例) vs 模型评判(CodeRM-8B/GPT-4.1)；长程编程用 GRPO 求稳。

#### 7. 实验数据集
- **环境**：TextWorld、ALFWorld(文本版，train 训 / valid-unseen 评，6 类)、SWE-Gym(getmoto/pydantic/mypy/pandas/dvc，随机 90 训 25 评)。
- **模型**：Qwen2.5-1.5B-Instruct、Qwen2.5-7B-Instruct、Qwen3-8B；算法 PPO/GRPO/RLOO/REINFORCE++；rollout temp 0.7。
- **指标**：任务成功率(SWE-Gym 用测试套件通过比例)。

#### 8. 实验结果与主要发现
- **(1) 随环境复杂度 scaling**(表 1)：base 17%→3%(同扩空间+物体)；PPO 增益从 base 环境 +88% 降到最复杂 +51%；物体复杂度比空间更难。
- **(2) 简→繁泛化**(表 5/6)：训简单环境对复杂环境有显著迁移(w8-o3-q4 训出的迁移最强，把 w8-o12-q4 提 48%，追平直接训该环境)；编程域同样(Easy→Medium +4.8%/Hard +3.6%)。
- **(3) 多任务训练**(表 7/8)：单类型训练即有 12%(ALFWorld)/7%(SWE-Gym)跨类型泛化；混合训练甚至对看似无关任务也增益(4 类混合超单类 pick&place 专家 +19%/全类 +21%)。
- **(4) SFT 先验**(表 9)：60 示范 + 400 RL episode 达 85%，逼近纯 RL 5000 episode 的 88%；跨域 SFT 先验反而有害(快速策略崩溃)。
- **(5) 最优 SFT:RL 配比**(表 9)：纯 SFT 域内强(w2-o3-q4 95%)但泛化差(w4-o6-q8 55%)；甜点是 60 SFT + 400 RL(域内 85%、复杂泛化 59%)。
- **(6) 算法对比**(表 10)：PPO 与 RLOO 均超 base(证增益来自 formulation 非 PPO 启发式)，但 PPO 最稳/样本效率最高(w2-o3-q4 88% vs RLOO 51%；w4-o6-q8 PPO 59% 而 RLOO/REINFORCE++/GRPO 在 1.5B 上崩溃)；REINFORCE++/GRPO 在 TextWorld 增益微弱。
- **(7) 奖励密度**(表 11)：稠密 turn-level 奖励加速训练，但最优密度依算法——PPO 受益于更频繁反馈(41%→58%)，RLOO 对密度不敏感(稳 55%)。
- **验证 vs 模型奖励**(表 12)：SWE-Gym 上稀疏二元验证仅 4.2%≈base，稠密单测比例验证 22%；模型评判 CodeRM-8B 7.2%/GPT-4.1 9.3%，远逊验证奖励。

#### 9. 结果如何支撑其主张
七问对应七组受控实验(固定数据量/预算公平比较)，逐条支撑 recipe 三要素：跨复杂度/多样性的迁移表(表 5–8)支撑"课程 + 简→繁泛化"；PPO vs RLOO 的双双增益(表 10)支撑"增益源于多轮 formulation"，PPO/GRPO 优于 RLOO/REINFORCE++ 支撑"稳定化偏置策略"；表 11/12 支撑"验证型稠密奖励优于模型评判"。无偏 RLOO 作对照是隔离启发式贡献的关键设计。

#### 10. 逻辑自洽性(中性评估)
作为经验研究内部自洽：每问受控变量、固定预算/数据量、用无偏 RLOO 做隔离对照，方法论谨慎。需注意：(1) 结论高度绑定所测三个文本域 + Qwen 1.5B/7B/8B，跨域可迁移性受限(作者也强调"非单轮简单外推")；(2) "PPO 优于 GRPO/RLOO"与多轮信用分配/value bootstrapping 强相关，GRPO 在 SWE-Gym(稀疏终局)反而有计算优势——即结论是情境相关的(论文已限定 PPO/GRPO 同为偏置方法仅在稀疏奖励下类比)；(3) 多处增益为小样本(SWE-Gym 90 训 25 评)，绝对成功率低(个位数～二十几%)，统计稳健性存疑;(4) SFT 贵 RL 10× 的成本假设是人为设定，影响"最优配比"结论。

#### 11. 残留问题 / 局限
- **域窄**：仅 TextWorld/ALFWorld/SWE-Gym 三个文本域 + Qwen 系列；recipe 的跨域/跨模型族泛化未验证。
- **小样本**：SWE-Gym 仅 90 训/25 评、成功率个位数～22%，方差与显著性未充分报告。
- **情境相关结论**：算法优劣随 reward 稀疏度/环境翻转(PPO 在 dense、GRPO 在 SWE-Gym 稀疏更优)，难给单一普适结论。
- **跨域 SFT 先验有害**：换域初始化导致策略崩溃，限制了迁移学习路径。
- **成本假设主观**：SFT:RL 最优配比依赖"SFT 贵 10×"这一设定。
- **非算法贡献**：是"经验法则汇编"，不提供新的可证明算法。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库 https://github.com/pearls-lab/meow-tea-taro （Apache-2.0；本地已 clone ~11MB；★数 〔待核〕，离线无法确认）。结构：`meow_tea_gym/`(环境，含 `SWE-agent/`)、`meow_tea_train/`(训练)、`meow_tea_experiments/`、`recipes/`、`scripts/`、`docs/`(Read the Docs)。
- **框架 = 封装/vendoring veRL**：仓库 `meow_tea_train/verl/` 直接 vendoring veRL；advantage estimator(grpo/rloo/reinforce_plus_plus/reinforce_plus_plus_baseline/grpo_passk/rloo_vectorized/grpo_vectorized 等)在 `meow_tea_train/verl/trainer/ppo/core_algos.py`(`AdvantageEstimator` 枚举 + `register_adv_est`);docker 基于 `hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0`(README)/`...th2.6.0-cu126-vllm0.8.4...`(docs)。真正"自研"的是环境层(`meow_tea_gym/`)与 environment/reward/policy 三支柱的可配置封装；策略侧 PPO/GRPO/RLOO/REINFORCE++ 均经 veRL 实现。
- 数据/模型权重：论文 Reproducibility 声明承诺随发表开源全部框架/脚本/权重(注：正文为"will release"将来时)。

〔本轮核实/新出入〕
1. 〔修正-时间〕PDF 实为 **arXiv:2510.01132 v2，2025-12-06**(cs.LG)；原分析写"2025-10 preprint"，应更新为 v2/2025-12。
2. 〔确认〕原分析"封装 veRL、非从零自研"的核码订正经仓库再核为真：`meow_tea_train/verl/` 确为 vendored veRL，`core_algos.py` 含完整 advantage estimator 枚举；`meow_tea_gym/` 含 SWE-agent。docker base 与 README 一致(`hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0`)。
3. 〔补充〕原分析未列 REINFORCE++ 对照与表 9–12 具体数值，本轮补全(含 SFT:RL 甜点 60+400、验证 vs 模型奖励 22% vs 7.2%/9.3%)。
4. 〔待核〕原分析"83★"无法离线确认，标 〔待核〕。


---


### open_agentrl — RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System (Open-AgentRL)

> **一句话重点 (TL;DR)**：一个完全动态的闭环 RL 系统,同时进化"环境、策略、生成式奖励模型"三者——策略用 step-wise+outcome 融合反馈训练,奖励模型经一致性反馈联合优化产出可靠 step-wise 监督,环境据策略当前能力自适应调难度;论证优化后的 step-wise 信号优于人工 outcome 标签。

**元信息**：arXiv 2602.02488（v1, 2026-02-02, cs.LG）｜ Gen-Verse（Yinjie Wang, Tianbao Xie, Ke Shen, Mengdi Wang, Ling Yang 通讯）｜ **ICML 2026**（仓库 README 标注"RLAnything (ICML 2026)";论文 PDF 自身未声明 venue）｜ 主题 T2（agentic RL）;**注:本文无 OPD 内容**,与 mtp_opd 仅间接相关——为"agentic 场景下过程信号优于纯结果"的论据,可与 OPD 的过程级反馈思路互补｜ 代码 github.com/Gen-Verse/Open-AgentRL（已 clone,~38M,同仓库另含 DemyAgent;HF 有 Policy & Reward 模型 collection）｜ 框架 veRL（仓库内置 `verl/`）

#### 1. 相关工作与进展
RLVR 提升 LLM 推理;但长轨迹下二元 outcome 奖励监督不足。step-wise 信号常由生成式奖励模型给出(借语言模型推理能力,优于标量奖励模型),但训练这类模型需高质量任务专属监督。环境质量(任务难度与策略能力匹配)对 RL 扩展同样关键,RLVR 中已证训练时调难度可改善策略。三要素——环境、策略、奖励模型——通常被分开处理。

#### 2. 现有工作存在的问题
(1) 长轨迹二元 outcome 奖励信号不足;(2) 奖励模型与策略割裂,且生成式奖励模型难获得可靠 step-wise 监督;(3) 环境任务难度与策略当前能力不匹配,影响策略与奖励模型双方训练动态。

#### 3. Motivation
是否存在一个同时优化环境、策略、奖励模型、以放大学习信号并强化整个系统的 RL 框架?构建完全动态、闭环优化的系统,让三者互相提供反馈、协同进化,适配任意 LLM/agentic 场景。

#### 4. 主要灵感 / 核心直觉
理论驱动:奖励模型质量不仅取决于单步逻辑正确,还取决于其预测该步未来影响的能力(reward precision A=P(S_τ+>S_τ−|...))。Theorem 1:A→1 当且仅当 μ=p++p−>1;Theorem 2:任务过难/过易会使 p+、p− 的重要性采样极不平衡、违反 μ 目标。故"调节任务难度"既利策略也利奖励模型训练——这是把"环境自适应"纳入闭环的理论动机。

#### 5. 主要解决思路(一段话讲清核心)
RLAnything 三组件闭环 forge(Algorithm 1):策略用 integrated feedback 训练;奖励模型把策略轨迹当训练环境、经 consistency feedback 联合优化;环境据策略 rollout 准确率(落在阈值 α_low=0.2 / α_high=0.8 外时)调难度。三者互为反馈、迭代。

#### 6. 方法详解(通俗、分步骤)
1. **策略(Integration Feedback,Eq.1)**:对第 i 步 τ_i,奖励模型独立查询 m 次得 S_τi,j∈{−1,1};step reward R_τi = O_τ + (λ/m)Σ_j S_τi,j(默认 λ=1),融合 outcome 与 step-wise;在同一步 index 上跨轨迹标准化得优势,训策略。
2. **奖励模型(Consistency Feedback,Eq.2)**:第 j 次评估的监督信号 RS_τi,j = R_τi · S_τi,j(R_τi 反映该步整体质量,与单次评估的一致性即监督);跨 j 标准化得优势,训奖励模型的评估推理 r_τi,j。Section 2.3 证此目标提升奖励模型预测未来 outcome 的精度。
3. **环境(Critic Feedback,Eq./§2.4)**:把奖励模型对失败步(S_τi,j=−1)的评估摘要喂给一个 LM(Qwen3-4B 做任务改写),据 acc 提议更难/更易任务 q',并质量门控(变难仅当 α_low<acc(q')<acc(q);变易仅当 acc(q)<acc(q')<α_high)后替换。

#### 7. 实验数据集
三类场景:① 计算机使用 agent — OSWorld(策略/奖励 Qwen3-VL-8B-Thinking,任务改写 Qwen3-4B,最大交互步 30;230 in-domain + 139 OOD);② 文本交互游戏 — ALFWorld(策略 Qwen2.5-7B-Instruct,官方 3.5k 训练 / 140 in-domain / 134 OOD,最大步 60);③ coding LLM — LiveCodeBench-V2、CodeContests、LiveBench(同 ALFWorld 的模型组合,无交互环境;奖励模型生成 32 个单元测试判正误,每步采 64 任务×32 解)。

#### 8. 实验结果与主要发现
RLAnything 跨三场景每加一个动态组件都一致改善整体并提升 OOD。关键数字:Qwen3-VL-8B-Thinking 在 OSWorld **+9.1%**;Qwen2.5-7B-Instruct 在 ALFWorld **+18.7%**、LiveBench **+11.9%**。核心发现:① 联合优化奖励模型+环境反过来抬高策略收敛精度;② 优化后的 step-wise 奖励信号**优于人工标注 outcome 信号**,且 integrated feedback 对长轨迹任务至关重要;③ 新环境任务可线性扩展,奖励模型在"评估当前步正确性"与"预测 outcome 影响"两方面都变强。

#### 9. 结果如何支撑其主张
"三组件协同放大信号"由逐组件消融(每加一个动态组件均提升)支撑;"step-wise 优于人工 outcome"由直接对比支撑,且与 Theorem 1/2(平衡难度→更高 reward precision)呼应;环境自适应的有效性由 Fig.3 中 acc 改善样例(如 0.0→0.25)与 λ 消融佐证。理论(两定理)为"为何要调难度"提供了闭环依据,逻辑链较完整。

#### 10. 逻辑自洽性(中性评估)
框架自洽:两定理把"奖励精度"与"任务难度平衡"连成因果,支撑环境自适应进入闭环。但 step reward R_τi 直接把 outcome O_τ 加进每步(Eq.1),使 step-wise 与 outcome 部分耦合,"step-wise 优于 outcome"的对比需在此耦合下解读;奖励模型与环境改写均由 LLM 担任,存在自评/自改写的潜在偏置。与 OPD 主线无直接关联,仅作"过程信号价值"的外部论据。

#### 11. 残留问题 / 局限
论文未见独立 Limitations 节(待核附录)。环境改写依赖 LLM 质量与阈值/门控设定;reward precision 理论在 m→∞ 渐近,有限 m 下偏差未充分量化;评测集中在 OSWorld/ALFWorld/coding 三域,泛化到其他 agentic 任务未验。

#### 12. 开源代码与框架(链接+框架+代码可得性)
github.com/Gen-Verse/Open-AgentRL(已 clone,~38M,同仓库含 DemyAgent;HF Policy & Reward collection)。框架 **veRL**(仓库内置 `verl/` 目录),GRPO 类 group-based RL;含 `recipe/`、`train/`、`reward/`、按任务的 `alfworld_rl.py`/`coding_rl.py`/`osworld_rl.py` 训练入口与对应 `*_eval.py`,以及 `OSWorld-main/`、`alfworld_master/` 环境;`requirements_rlanything.txt`、`requirements_sglang.txt`、`requirements-npu.txt`。代码可得性高(完整训练/评测/环境脚本齐全)。


---


### openclaw_rl — OpenClaw-RL: Train Any Agent Simply by Talking

> **一句话重点 (TL;DR)**：把每次 agent 交互产生的"next-state 信号"当作在线学习源回收,从中抽 evaluative(标量/更频繁)与 directive(token-level/更富信息但稀疏)两类信号,在一次 hybrid RL 更新中统一;并用 overlap-guided hint selection + logprob-diff clip 稳定 teacher–student 失配下的 OPD,使"agent 越被使用越变强"。

**元信息**：arXiv 2603.10165（v2, 2026-05-11, cs.CL）｜ Gen-Verse（Yinjie Wang*, Xuyang Chen*, Xiaolong Jin*, Mengdi Wang†, Ling Yang† 通讯;含 Princeton 等）｜ 2026 Preprint（Blog yinjjiew.github.io/projects/openclawrl1）｜ 主题 T2（agent RL/工具）,与 mtp_opd 高度相关——把 OPD 显式作为 hybrid RL 的 "directive signal",专门处理 teacher–student 失配下的 OPD 稳定性(overlap-guided hint selection + logprob-diff clip,对应 path-recovery/hint 选择)｜ 代码 github.com/Gen-Verse/OpenClaw-RL（已 clone,~73M）｜ 框架 slime 异步 RL（Megatron 训练 + SGLang serving + PRM）

#### 1. 相关工作与进展
LLM agent 已广泛部署(终端/GUI/SWE/tool-call);交互数据被用于改进 framework、构建 memory(Mem0/Cognee/Letta)、产训练数据,但鲜有把它当**在线、实时**学习源。现有 agentic RL 基础设施(slime [Sheng 2025]、OpenRLHF [Hu 2024]、AReaL [Fu 2025] 等)假设批量预采集数据。方法侧:RLVR 仅能用标量奖励;OPD [Agarwal 2024]、SDPO/SDFT [Hübotter/Shenfeld 2026]、hindsight relabeling 能用结构化纠正信息但都在固定数据集上操作;并发 Buening 2026 直接用 next-state 提示在线改策略,但纠正 hint 仍隐含在 prompt 中。

#### 2. 现有工作存在的问题
- **基础设施**:RL server 需灵活对接用户多样且演进的 agent 框架,且优化须异步、不阻塞推理使用。
- **方法学**:RLVR 无法把 directive(纠正性)信号转成策略梯度;OPD/hindsight 虽能用结构化纠正,但只在固定数据集上做。OPD 还因 teacher–student 分布失配 [Li 2026] 训练不稳/失效,低质量 hint 时更严重。

#### 3. Motivation
每个 next state(用户回复、工具输出、终端/GUI 状态变化)既隐含评价(re-query=不满、通过测试=成功、错误 trace=失败),又常携带指令信息("你应该先检查文件"在 token 级指明该怎么改)。回收这两类互补信号、统一进一次在线更新,并稳定其中的 OPD 蒸馏。

#### 4. 主要灵感 / 核心直觉
evaluative 与 directive 互补:directive 更富信息(token-level)但更稀疏,evaluative 更频繁(标量)。OPD 的 token-level KL 监督(teacher 条件于纠正 hint)恰能承载 directive。稳定 OPD 的关键直觉:若所选纠正 hint 诱导的 teacher 分布与学生在高概率区高度重叠,则蒸馏梯度更 informative、off-policy importance ratio 接近 1,更新更稳。

#### 5. 主要解决思路(一段话讲清核心)
基础设施:把 RL 系统扩成 server–client——RL server 把策略包成推理 API,用户终端经 OpenClaw 把交互数据 HTTP 流式回传;独立异步 PRM/Judge server 从 next state 抽 evaluative + directive 信号(不阻塞推理);slime 协调环境 server / PRM / Megatron 训练 / SGLang serving,零 serving 中断、graceful weight update。方法:hybrid RL 目标在单次更新融合 evaluative(RLVR 标量)与 directive(OPD token-level KL,teacher 条件于纠正 hint)两类 loss,配 overlap-guided hint selection 与 logprob-diff clip 稳定。

#### 6. 方法详解(通俗、分步骤)
1. **Hybrid RL objective**:directive=OPD(token-level KL)+ evaluative=RLVR(标量奖励),单更新融合(附录 C 证明 OPD 目标即 token-level KL)。
2. **Overlap-guided hint selection**:候选纠正 hint 中,选其诱导 teacher 分布与学生 **top-k token 重叠最大**者(可逐 token 或序列聚合)。
3. **Log-probability-difference clip**:对 token-level logprob 差做 clip,bound 每 token advantage(personal 设置默认 C=1)。
4. **Step-wise / process reward**:通用 agentic RL 整合 outcome + process reward(PRM),强调过程奖励对长 horizon 稀疏奖励 agent 任务的重要性。
- 集成了社区 SDFT、SDPO 等方法到 `openclaw-opd/`。

#### 7. 实验数据集
- **Personal agents**:用 LLM 模拟不同职业用户(学生避免 AI 痕迹 / 助教要详细评分 / 教师要友好评语)在 GSM8K 任务上使用 OpenClaw,度量"对齐各用户偏好所需最少 sessions"(连续 3 session 满足偏好即达标)。policy & reward 均 Qwen3-4B-Thinking-2507;用户用 Qwen3-32B 模拟。
- **General agents(Track 2)**:四类真实部署环境,模型分别为 Terminal=Qwen3-8B、GUI=Qwen3VL-8B-Thinking、SWE=Qwen3-4B、Tool-call=Qwen3-4B-SFT(Retool-4B,源自 Zhu 2025);训练数据分别为 SETA RL data / OSWorld-Verified / SWE-Bench-Verified / DAPO RL data;GUI 评在训练集(去 chrome 与 multi-apps),tool-call 评 AIME 2024,terminal/SWE 报窗口内平均 rollout-task acc。声称是首个统一这四类 agent 的开源 RL 框架。
- (注:正文实验未涉及 Qwen3.5;仓库 `openclaw-opd/run_qwen35_4b_openclaw_opd.sh` 与 slime qwen3.5-4B 配置存在,但非论文报告模型。)

#### 8. 实验结果与主要发现
- **Personal**(Table 3,最少 sessions↓,5 trials 均值):joint 优化下 Hybrid RL 平均 **10.3** sessions,优于 GRPO 14.1、OPD 单用 29.7、Mem0 14.5、Cognee 14.9;separate 优化下 Hybrid 15.0 仍最优。注:纯 OPD 远差(29.7),增益主要来自 hybrid 融合;joint 优化放大 RL 增益而 memory 类几乎不变。
- **General**:hybrid RL 在四环境均优于纯 RLVR 且更稳定;next-state 信号在长 horizon 稀疏奖励环境尤其有用。
- **消融**:overlap-guided hint selection 与 logprob-diff clip 对效率/稳定性均关键;另有 k 与 support set S_i、PRM、policy/reward 模型组合的消融(§4.8–4.11)。
- 超参:personal lr 1e-5、每 16 样本触发一步;general lr 1e-6、KL 系数 0.01、clip 0.2/0.28、每步 GUI/SWE 采 8 任务、terminal 16、tool-call 32,各 8 samples;GUI/SWE/terminal 最大交互步 30/20/10。

#### 9. 结果如何支撑其主张
"越用越强"由 personal 设置的 session-to-align 指标直接度量(Hybrid 最少 sessions)。"directive+evaluative 互补、hybrid 更优"由 hybrid 全面超 GRPO/纯 OPD 支撑(纯 OPD 反而最差,佐证单靠 directive 不稳、需 evaluative 兜底)。"稳定性来自 hint 选择/clip"由 §4.8–4.9 消融支撑。统一框架主张由四环境实跑支撑。

#### 10. 逻辑自洽性(中性评估)
infra(slime 异步、server–client、零中断)与 method(hybrid+hint selection+clip)两条创新线自洽,且附录 C 把 OPD 归约为 token-level KL、与 evaluative 路径并列合理。但 personal-agent 评测高度依赖 LLM 模拟用户与硬编码偏好检测器(粗体/列表/长度/暖词),度量"对齐效率"而非真实能力提升,生态效度有限;general-agent 部分环境直接在训练集上评(GUI),需谨慎。

#### 11. 残留问题 / 局限
〔待核〕HuggingFace Daily Papers #1 一说——已确认论文正文不含此声明(应出自仓库 README),保留为待核。模拟用户/检测器的生态效度;directive 信号质量依赖 PRM/hint 抽取质量;具体超参与 support set S_i 细节见附录 A。是否在真实人类用户上验证未知。

#### 12. 开源代码与框架(链接+框架+代码可得性)
github.com/Gen-Verse/OpenClaw-RL(已 clone,~73M)。框架 **slime(THUDM/slime)异步 RL** 为核心,四解耦异步组件:环境 server、PRM/Judge(SGLang/API)、**Megatron**(策略训练)、**SGLang**(策略 serving);server–client 把模型包成 OpenAI 兼容 API(经 OpenClaw 插件),HTTP 流式回传在线训练,零 serving 中断、graceful weight update;也支持 Tinker 云端/本地 GPU 与 LoRA。三范式:Binary RL / OPD / Combine(Hybrid)。关键子目录:`openclaw-opd/`(OPD,含 topk 蒸馏 loss、qwen3/qwen35 launcher、集成 SDFT/SDPO)、`openclaw-rl/`、`openclaw-combine/`、`openclaw-test/`、`openclaw-tinker/`、`openclaw-fireworks/`;Track-2:`gui-rl/`(及 terminal/swe/toolcall 相关);含 `slime/`、`Megatron-LM/`、`extensions/`。


---


### pi_play — π-Play: Multi-Agent Self-Play via Privileged Self-Distillation without External Data

> **一句话重点 (TL;DR)**：自博弈在造题时天然产出一条 "问题构造路径(QCP)"，本文把它当作零成本的内禀特权信息，让同规模 teacher 据此对 student 做 token 级 reverse-KL 自蒸馏，从而把稀疏奖励自博弈变成稠密反馈的 data-free 自演化。

**元信息**：arXiv 2604.14054（v2 2026-05-25）｜ 中科院自动化所 + 国科大 + 美团 ｜ 2026-04 预印本 ｜ 主题 self-play + self-distillation for 深度搜索 agent，与 TSRD（QCP=foresight/教师脚手架）相关 ｜ 代码 github.com/zhyaoch/pi-play（README 注明 "Code will be released soon"，当前仅 README+images，**训练代码未发布**）｜ 框架 论文未绑定特定开源框架，student 用 GRPO。

#### 1. 相关工作与进展
深度搜索 agent 结合 LLM 推理与外部搜索引擎做多轮检索分析；RL 能显著提升其推理与搜索行为（Search-R1、ToolForge 等监督 RL）。自博弈（Dr.Zero、SQLM、SSP）让同规模模型轮流当 examiner/student 以降低数据依赖；自蒸馏可改善 credit assignment，但需高质量特权信息。

#### 2. 现有工作存在的问题
(1) 自博弈只给学生**稀疏结果奖励**，多轮搜索任务学习低效、信用分配弱；(2) 自博弈实际产出的不止最终 QA 对 (q,o)，还有被忽视的中间产物——**QCP(c)**，记录从答案反向构造问题的逆向解题过程，但现有自博弈无法直接拿它做 SFT；(3) 自蒸馏所需高质量特权信息获取代价高，且常依赖人类专家/更强模型与 curated QA 数据，难规模化。

#### 3. Motivation
关键观察：自博弈天然、低成本、可规模化地产出 QCP，而 QCP 记录了问题如何从事实证据构造而来，正是一种**内禀特权信息**——可让同规模 teacher 在条件于 c 时生成比仅看 q 的 student 更准确的 rollout。由此把自博弈产生的 QCP 直接喂给自蒸馏，无需人类反馈或人工特权数据。

#### 4. 主要灵感 / 核心直觉
"造题过程本身就是答案的脚手架"：既然 examiner 是从事实/答案反向构造问题，那条构造轨迹 c 天然包含解题所需的特权线索；用它做 teacher 的额外上下文，就能把稀疏的结果奖励转成稠密的逐 token 监督，而不引入任何外部数据。

#### 5. 主要解决思路(一段话讲清核心)
三角色（examiner / teacher / student，均由同一 base LLM 初始化、均为带搜索工具的搜索 agent）交替优化：examiner 造出 (q,c,o)，teacher 条件于特权上下文 c、student 仅见 q；student 在结果奖励 + teacher 的逐 token reverse-KL 蒸馏联合作用下学习，形成 data-free 协同自演化闭环。

#### 6. 方法详解(通俗、分步骤)
- **Examiner**：带搜索工具与搜索引擎交互获取事实，生成三元组 (q, c, o)；目标为造出多样且有难度的题（含难度奖励，并 −β·KL 约束；难度奖励随正确预测数线性衰减）。
- **Teacher**：理想目标 Eq.2 兼顾 "生成更准 rollout" 与 "不过度偏离 student"，但**实际实现并不直接优化该目标**，而是把 teacher 参数取为 student 的 **EMA 软更新**：ψ←(1−τ)ψ+τθ（Eq.11，soft-update 权重 **τ=0.05**，已核 Table 7），低成本提供稳定且随 student 缓慢演化的监督。
- **Student**：在结果奖励 I(ô=o)（GRPO）与教师指导（−per-token reverse-KL D(πS_θ(·|q) ‖ stopgrad[πT_ψ(·|q,c)])）联合下学习。
- **蒸馏权重衰减调度**：用衰减的 **λ**（distillation 系数）逐步弱化教师指导——Qwen3-4B/8B 取 0.03/0.003/0.002，Qwen3-4B-Instruct 取 0.1/0.03/0.03。〔修正：原分析在 §6/§8 误用 "β" 指代蒸馏衰减系数；论文中蒸馏权重为 λ，β 是 examiner 难度奖励/KL 系数——已统一改为 λ〕

#### 7. 实验数据集
论文为 data-free 自演化，**不依赖任何外部训练数据**（无人工 demo/问题/答案）。〔已核 PDF §3.1〕
- 基座：Qwen3-4B、Qwen3-4B-Instruct、Qwen3-8B。
- 评测：3 个 one-hop/General QA（NQ、TriviaQA、PopQA）+ 4 个 multi-hop QA（HotpotQA、2WikiMQA、MuSiQue、Bamboogle）；统一 exact-match，检索器 E5-base，语料英文 Wikipedia dump。
- 基线：training-free（ReAct）、监督 RL（Search-R1、ToolForge）、自博弈（Dr.Zero、SQLM*）。

#### 8. 实验结果与主要发现
- 流程（Algorithm 1）：每轮先训 examiner 50 步 → 生成学生训练数据 → 训 student 50 步（GRPO + teacher reverse-KL），每个 student step 后 teacher 做 EMA 软更新；共 3 轮 = **150 步**（远少于 Search-R1 等基线）。examiner 默认造 1/2/3/4-hop 比例 4:3:2:1。
- 主结论：data-free 的 π-Play 超过全监督搜索 agent——平均较 Search-R1 **+6.3% / +4.2% / +15.4%**（Qwen3-4B / 4B-Instruct / 8B）；演化效率较传统自博弈提升 **2–3×**（首轮即可媲美 Dr.Zero 三轮收敛值）。〔已核行 838、43/251、1057-1058〕
- 消融：QCP 优于其他特权信息形式（含 ground-truth）；衰减 λ 调度优于固定 λ。

#### 9. 结果如何支撑其主张
"QCP=有用特权信息" 由 Table 3 消融支撑（QCP > Partial QCP > 无/替代特权信息）；"高效率" 由首轮即追平 Dr.Zero 三轮收敛、150 步胜全监督基线两点支撑。主张与证据方向一致。

#### 10. 逻辑自洽性(中性评估)
内在逻辑自洽：把被忽视的造题副产物变成监督信号是清晰且新颖的因果链。但需注意 teacher 实际只是 student 的 EMA，"教师" 的额外能力完全来自条件于 c 的特权上下文而非独立训练，这与 §2.4 "理想 teacher 目标" 之间存在 "理想 vs 实现" 落差，论文已坦承。

#### 11. 残留问题 / 局限
- **代码未发布**：框架/实现细节（搜索工具栈、reverse-KL 具体实现、examiner 难度奖励工程）无法核验，复现性受限。〔待核：完整训练代码待官方释出〕
- 评测局限于英文 Wikipedia QA（one/multi-hop）+ exact-match，未覆盖更开放的搜索任务；检索器/语料固定。
- "2–3× 效率" 的对比依赖与 Dr.Zero 等的同设定可比性。
- EMA teacher 是否在更大规模/更长训练下仍稳定优于直接优化 Eq.2，未做对照。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库：https://github.com/zhyaoch/pi-play（本地已克隆约 1.5MB，**仅含 README 与 images/，代码尚未发布**；TODO：paper 已发布、code 未发布）。
- 框架：论文未在正文绑定特定开源训练框架；student 用 GRPO，teacher 为 EMA 软更新，三角色交替优化。〔待核：框架实现细节待 code 发布确认〕
- 代码可得性：paper-only（"coming soon"）。


---


### rstar2 — rStar2-Agent: Agentic Reasoning Technical Report

> **一句话重点 (TL;DR)**：用"高吞吐代码执行环境 + 抗噪的 GRPO-RoC（Resample-on-Correct）+ 短长度多阶段 RL recipe"，在 64×MI300X、510 步 / 一周内把 Qwen3-14B-Base 推到前沿数学推理，AIME24=80.6/AIME25=69.8/HMMT25=52.7，并在 AIME24/HMMT25 上超过 671B 的 DeepSeek-R1（AIME25 基本持平）。

**元信息**：arXiv 2508.20722（v1, 2025-08-28, cs.CL）｜ Microsoft Research（Ning Shang、Yifei Liu、Yi Zhu、Li Lyna Zhang 等共同一作；Li Lyna Zhang、Mao Yang 为 project leaders / 通讯）｜ 技术报告, 2025-08 ｜ 主题 Agentic RL / 工具调用推理 / T2,T4，Relevance=High（GRPO-RoC 的"正样本筛优 + 负样本保多样"与本项目 path-selection / 高质量正样本+保留失败负样本相关）｜ 代码 https://github.com/microsoft/rStar（已克隆 ~576K，主仓为骨架 + submodule 引用，verl/code-judge 为 git submodule，浅克隆未拉取）｜ 框架 veRL v0.5 + Code Judge + vLLM

#### 1. 相关工作与进展
让模型"想得更聪明而非更长"需培养自主用工具来推理、验证、从工具反馈学习的能力。本工作以 Python 编码工具 + 解释器作 agentic RL 环境，拓宽动作空间、支持探索替代解与验证中间步骤，补足单纯 long CoT 的内部自反思不足。对比主流把 rollout 长度堆到 16K→48K 的做法，rStar2 走短长度路线。多分支仓：rStar2-Agent 在 main，prior work 在 `rStar-mutualreasoning`、`rStar-math` 分支。

#### 2. 现有工作存在的问题
扩展 agentic RL 两大挑战：
- **环境噪声**：编码工具/解释器复杂，模型生成语法/逻辑错误代码时，环境反馈会让其浪费 token 纠错而非推进推理；现有 outcome-only 奖励的 RL 即便中间工具调用失败、只要最终答案对仍给正奖励，导致模型把错误当可接受、产出冗长低质轨迹（论文给出工具错误率随训练步数上升的观察，Qwen2.5-32B≈15%、Qwen3-14B≈10%）。
- **基础设施压力**：单 batch 可触发数万并发工具调用，难构建可靠高吞吐执行环境；agentic rollout 放大标准 RL 系统的 rollout 低效，严重拖慢训练。

#### 3. Motivation
通过三项创新使 agentic RL 在规模上有效，从而在有限 GPU（64 张 MI300X）上把 14B 基座在 510 步 / 一周内推到前沿数学推理水平、并超越 671B DeepSeek-R1。

#### 4. 主要灵感 / 核心直觉
- **非对称采样**：与其在 reward 里显式惩罚工具错误（易 reward-hacking、不稳），不如在采样层面对正/负轨迹差异化处理——正样本只留最干净的高质量成功轨迹作正监督，负样本均匀 downsample 以保留多样失败模式作负信号。
- **短而聪明**：不堆长度（避免 16K→48K 的高成本与冗长低质），用 8K→12K 短长度逼模型高效推理。
- **SFT 只打底不增推理**：先做 non-reasoning SFT 仅注入指令遵循/工具使用/格式，避免 SFT 过拟合、保持初始响应短。

#### 5. 主要解决思路(一段话讲清核心)
三件套：(i) 可靠高吞吐 Python 代码执行环境（Code Judge + Redis + 隔离 worker）缓解高 rollout 成本；(ii) **GRPO-RoC**——在 GRPO 上加 Resample-on-Correct：先 oversample 较大 rollout 组再 downsample 到标准 batch，正轨迹按"工具错误/格式问题最少"筛优、负轨迹均匀下采样，用非对称采样抗稀疏 outcome-only 奖励下的环境噪声；(iii) 低成本训练 recipe——先 non-reasoning SFT，再用 GRPO-RoC 做短长度多阶段 RL（8K→12K→12K，共 3 阶段 510 步）。

#### 6. 方法详解(通俗、分步骤)
- **(i) 高效 RL 基础设施**：Code Judge 作工具调用服务器执行模型生成的 Python（Redis + uvicorn + workers，多节点可扩展），应对单 batch 数万并发调用。
- **(ii) GRPO-RoC**：RoC 先 oversample 大组 rollout，再 downsample 到标准 batch；**正轨迹**保留工具错误/格式问题最少的最高质量样本，**负轨迹**均匀 downsample。这种非对称采样保留多样失败模式作负信号、强调高质量成功样本作正监督；相比在 reward 里惩罚工具错误更稳、避免 reward-hacking（从更干净的正样本学习）。
- **(iii) 训练 recipe**：① non-reasoning SFT——用 165K function-call 数据（117K ToolACE-11K + APIGen-MT-5K + Glaive-function-calling-v2-101k 等）仅注入工具格式与指令遵循，不增强推理；② 多阶段 GRPO-RoC：Stage-1 在 8K 长度做简洁训练（响应从约 1K 增长），Stage-2/3 提到 12K 并逐步加难，3 阶段共 510 步。

#### 7. 实验数据集
- 基座 **Qwen3-14B-Base**。RL 训练数据（§4.1）：仅保留整数答案题以保 verifier 可靠，从三源收 >100K 候选——**17K** 来自 DAPO 训练集（整数答案子集）、**93K** 来自 AoPS 论坛（经 OpenMathReasoning）、**937** 来自 Project Euler——清洗后得 **42K** 高质量题-答对为最终 RL 训练集（开源 data_preprocess 以 DAPO-17k 为示例入口，论文实际用 42K 复合集）。
- 数学评测：AIME 2024 / 2025、MATH500、HMMT25。泛化评测：GPQA-Diamond（科学）、BFCL v3（agentic 工具）、IFEval、Arena-Hard。

#### 8. 实验结果与主要发现
- 数学（pass@1）：rStar2-Agent-14B **AIME24=80.6 / AIME25=69.8 / HMMT25=52.7**。对照 DeepSeek-R1(671B) 79.8/70.0/44.4、o3-mini(medium) 79.6/77.0/53.0、DeepSeek-R1-Zero(671B) 71.0/53.3/46.0、Claude-Opus-4.0(Think) 76.0/69.2/-、QWQ-32B 79.5/65.8/47.5。即 14B 在 AIME24 与 HMMT25 上超 R1-671B，AIME25 基本持平（略低 0.2）。
- 效率：510 步 / 一周 / 64×MI300X 达前沿，响应显著更短。
- 泛化：GPQA-Diamond 超 DeepSeek-V3；BFCL v3、IFEval、Arena-Hard 均有竞争力。
- 消融/观察：GRPO-RoC 提升训练稳定性、避免 reward-hacking；从近零起步即被显著拉升。

#### 9. 结果如何支撑其主张
"小模型超大模型 + 高效"由对照表（14B vs 671B/o3-mini）与 510 步/一周直接支撑。"抗噪有效"由 GRPO-RoC 提升稳定性、工具错误率观察与从近零拉升支撑。"短长度足够"由 8K→12K 设置 + 前沿成绩支撑（隐含对照主流 16K→48K）。泛化主张由跨任务结果支撑。需注意 AIME25 上并未严格超过 R1（持平/略低），TL;DR 的"超越"以 AIME24/HMMT25 为主。

#### 10. 逻辑自洽性(中性评估)
三创新对应两挑战（基础设施↔环境压力、GRPO-RoC↔环境噪声、recipe↔成本），映射清晰。GRPO-RoC 的"正筛优/负保多样"与"避免 reward-hacking"动机一致。SFT 设为 non-reasoning 与"保持初始响应短/不过拟合"自洽。整体为工程+算法的技术报告，论证连贯。

#### 11. 残留问题 / 局限
- 开源迁移版（VERL v0.2→v0.5）尚未完整训出模型（作者注前 50 步差异极小，但未给最终复现成绩）。
- 〔待核〕各阶段精确数据配比、步数划分、学习率等见 §4.3 Multi-Stage RL Training，本次未深读。
- 仅整数答案题以保 verifier，限制了任务多样性；AIME25 未严格超 R1 说明优势未必跨所有 benchmark 一致。
- 单一基座（Qwen3-14B-Base）、单一硬件栈（MI300X）；可靠性高度依赖 Code Judge 工程实现，迁移到其他环境的稳定性未验证。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 链接：https://github.com/microsoft/rStar（main 分支为 rStar2-Agent；浅克隆未拉 submodule）。
- 框架：**veRL v0.5（volcengine/verl）+ Code Judge（0xWJ/code-judge）**；原训练基于 VERL v0.2 + 自研多轮工具框架，开源版迁移到 v0.5。Code Judge：Redis + uvicorn server（:8088, MAX_EXECUTION_TIME=4）+ workers。rollout/推理用 vLLM（`--enable-auto-tool-choice --tool-call-parser hermes`）。安装需 `torch<2.8`、`verl/requirements_sglang.txt`。
- 关键脚本：`data_preprocess/{aime2024,dapo}_rstar2_agent_loop.py`；训练 `examples/run_qwen3-14b_rstar2_agent_weave.sh`（8×A100/H100）；RoC 配置 `augmentation.do_down_sampling=True`、`down_sample_to_n=16`、`reject_equal_reward=True`、`roc_error_ratio=True`（按工具错误率重采正确轨迹）、`roc_answer_format=True`、`min_zero/non_zero_reward_trace_num=2`（保留最少正/负轨迹）；评测 `examples/{aime_eval,math500_eval}.sh`。
- 可得性：代码骨架 + 配置齐备，但 verl/code-judge 为 submodule（需 `git submodule init/update`），且迁移版未训出完整模型，端到端复现有门槛。


---


### score — From Correction to Mastery: Reinforced Distillation of LLM Agents (SCoRe)

> **一句话重点 (TL;DR)**：让小学生 agent **主导**轨迹生成、teacher **只纠正最早一步错误**,学生从"已验证前缀"续写并做短 horizon RL,把行为克隆的累积误差从 O(H²) 降到 O(H);RL 阶段从最早错误前的前缀起 rollout 并用 key-step 稠密奖励缓解稀疏奖励。

**元信息**：arXiv 2509.14257 (v2, 2025-10-09) ｜ 中科大 USTC(Yuanjie Lyu、Tong Xu 通讯) + Independent Researcher(Chengyu Wang 通讯、Jun Huang;原阿里相关) ｜ arXiv 预印本 2025-09(v2 2025-10) ｜ 主题 T2(agent/tool 蒸馏),与 mtp_opd 核心 path-selection + path-recovery / 教师纠正最早错误高度契合 ｜ 代码 github.com/modelscope/easydistill(SCoRe 在 `projects/SCoRe/`,整库 ~221MB) ｜ 框架 自定义工具链(LLaMA-Factory + LangGraph + veRL,非单一框架)

#### 1. 相关工作与进展
LLM agent 经 ReAct 式"推理-动作-观测"迭代 + 外部工具(代码解释器、搜索)解复杂任务,但依赖超大昂贵 backbone(GPT-4/Qwen2.5-72B),延迟成本高。Agent Distillation(Kang et al. 2025)把 teacher 行为拆成 [Thought, Action, Observation] 轨迹让小学生模仿。本文构建于 **CodeAct**(action=可执行代码)。

#### 2. 现有工作存在的问题
"teacher-acts, student-clones"全轨迹模仿两大 gap:(1) **Reasoning Ability Gap**——小模型难复现 teacher 逻辑分解;(2) **Knowledge Capability Gap**——照搬计划也可能因知识不足无法执行复杂动作。二者源于 emergent abilities,不能完全迁移。更关键:BC 中任一步失败把学生推入 OOD 状态,误差按 **O(H²)** 随 horizon 复合增长(Ross et al. 2011 DAgger 分析)。

#### 3. Motivation
让学生**主导**、teacher **最小干预**(只纠最早错误),从而:(a) **Capability Matching**——轨迹复杂度匹配学生当前能力,数据可学;(b) **Deficiency Localization**——"已验证前缀 + 关键步"结构精确暴露弱点。把 teacher-student 分布偏移限制在单步,打破长误差链,累积误差 O(H²)→**O(H)**(论文附录给出 per-step 不等式推导)。

#### 4. 主要灵感 / 核心直觉
纠错信号应施加在学生"最早出错"的那一步:此前前缀由学生自己生成(on-distribution),此后从被纠正的步起续写,既保证正确性又把监督定位到学生真实能力边界。

#### 5. 主要解决思路(一段话讲清核心)
SCoRe(Student-Centered one-step Reinforcement)三阶段:先用少量 teacher 轨迹冷启动 BC;再让学生独立解题、teacher 只纠最早错误、学生从纠正前缀续写(必要时重复),保留"最小干预"纠正轨迹;最后做短 horizon RL——**从已验证前缀(最早错误前)起 rollout**(缩短 horizon、降梯度方差)并用 **key-step 稠密奖励**。

#### 6. 方法详解(通俗、分步骤)
1. **Cold-Start BC**:在少量高质量 teacher 轨迹上 SFT,bootstrap 基础推理-动作技能。
2. **Mentored Problem-Solving (MPS) + SCoRe-SFT**:学生独立做新任务 → teacher 检查并纠正**最早错误** → 学生从纠正后前缀续写;若再错则重复。最终任务成功隐式验证修正正确性。纠正轨迹用于 SFT。
3. **SCoRe-RL** 两创新:(a) 从已验证前缀起 rollout 而非任务开头;(b) **key-step 稠密奖励**:最终答案正确 reward=**1**;关键步等于 teacher 修正 reward=**0.5**;关键步等于初始错误 reward=**0**;都不等价 reward=**0.1**(GRPO 式按组算 advantage)。RL 奖励正误由**轻量验证器 Qwen2.5-7B-Instruct**判语义一致性(注:与评测时用的 72B judge 不同,见 §7)。

#### 7. 实验数据集
12 个 benchmark 三类(论文 §4.1 / Table 1–2):
- **数学(4)**:AIME2024、AIME2025、MATH500、OlympiadMath;
- **事实/多跳 QA(4)**:HotpotQA、2WikiMultihopQA、Musique、Bamboogle(开放域 QA 用 token-level F1);
- **agentic 深度搜索(4)**:GAIA、WebWalker、HLE、xBench(WebThinker text-only split)。
- **评测 judge**:math + deep-search 正误由 **Qwen2.5-72B-Instruct** 作 LLM-as-judge(Zheng et al. 2023 范式)判定;QA 用 F1。
- **student**:Table 1(math+factual)= Qwen2.5-7B / Qwen2.5-3B / Llama3.1-8B;Table 2(deep-search)= **Qwen3-8B-Instruct**。评测集组织参考 ARPO,GRPO/ARPO 数值多取自 ARPO 原文。

#### 8. 实验结果与主要发现
- **math+factual(Table 1,8 项 Avg)**:Qwen2.5-7B SCoRe-RL = **50.8**(BC=42.5、GRPO=48.4、ARPO=49.3),仅比 72B teacher(51.7)低 **0.9**;较 BC **+8.3**(50.8−42.5,已核)。Qwen2.5-3B SCoRe-RL=46.7(较 BC +8.4);Llama3.1-8B=47.5(较 BC +10.2)。
- **deep-search(Table 2,Qwen3-8B)**:SCoRe-RL Avg = **30.5**,+7.7 over BC、+8.3 over GRPO、超 TIR-Qwen2.5-72B +3.2,部分子项超 teacher;GAIA-Avg 从 27.2 升。
- **消融(Table 3)**:去短 horizon rollout / 去 key-step reward 均掉点,二者齐备最优。Figure 3:hard data 上 SCoRe-RL>SCoRe-SFT。
- 〔待核〕RL 超参(组大小、lr、KL 系数)未在正文给出,见附录 B。

#### 9. 结果如何支撑其主张
"逼近 teacher"由 7B 仅低 0.9、部分 deep-search 子项超 teacher 支撑;"O(H²)→O(H) 有益"由短 horizon rollout 消融掉点支撑;"key-step 暴露弱点有益"由 key-step reward 消融掉点 + Figure 3 hard-data 增益支撑;跨 3 个 student 一致较 BC 大幅提升支撑普适性。链条较完整。

#### 10. 逻辑自洽性(中性评估)
整体自洽,但有一处**论文内部数值不一致(本轮发现)**:正文 §5 prose 写 7B "+6.3 over GRPO",但 Table 1 GRPO=48.4、SCoRe-RL=50.8,实差仅 **+2.4**,prose 的 +6.3 与自身表格矛盾(上一轮分析直接抄了 prose 的 +6.3,本轮按表格更正为 +2.4)。其余如 +8.3 over BC、deep-search +8.3 over GRPO 与表格自洽。另外:"最终任务成功隐式验证 teacher 修正正确性"是弱验证——任务成功不等于每步修正都正确,可能引入噪声标签;论文未量化误纠率。

#### 11. 残留问题 / 局限
- teacher 需 Qwen2.5-72B 级模型生成纠正,蒸馏成本不低;真"小成本"仅指部署期 student。
- RL 奖励验证器(7B)与评测 judge(72B)不同,存在 reward hacking / 训练-评测口径不一致的潜在风险,论文未交叉验证。
- "最早错误"的定位依赖 teacher 判断,错判会污染 SFT/RL 数据;无误纠率量化。
- 论文 prose 与 Table 的 GRPO 增益不一致(见 §10),建议以 Table 为准。
- 〔待核〕RL 完整超参在附录 B。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库:https://github.com/modelscope/easydistill(SCoRe 在 `projects/SCoRe/{MPS, SFT, RL, inference}`;整库 clone ~221MB)。
- 框架(组合工具链,非单一):
  - Cold-start BC / SCoRe-SFT 用 **LLaMA-Factory**(`llamafactory-cli train/api`,README 明列 `pip install llamafactory langgraph`)+ **LangGraph**(agent 轨迹生成,`MPS/graph/{graph.py, graph_repair.py}`)。
  - SCoRe-RL 用 **veRL**(含 `verl/tools/search_tool.py` 工具调用)。
  - 推理 LLaMA-Factory api(`infer_backend: vllm|sglang|huggingface`)。
  - teacher 经 `QWEN25_72B_API_ADDRESS`/`_KEY` 走官方 API 或 `llamafactory-cli api --model_name_or_path Qwen/Qwen2.5-72B` 本地部署。
- 数据流水:`MPS/BC_init_data_gen.py` → `MPS/format/BC_init_llamafatory_format_convert.py` → BC 训练;部署学生 → `MPS/SCoRe_data_gen.py` 生成"学生中心、teacher 仅纠最早错"轨迹 → `SCoRe_SFT_llamafactory_format_convert.py` → SCoRe-SFT;`MPS/format/SCoRe_RL_convert.py` 过滤与 SFT 重叠样本 → verl parquet → `RL/.../run_qwen2.5-7b_agent_distill_tool_agent_mlflow.sh`。
- 数据:种子主要取自 **Tool-Star**(math: NuminaMath、Omni-Math;factual: HotpotQA/2Wiki/WebWalker),共 **35k QA 对**;**20%** 全程 teacher 标注作 BC,**80%** 经 MPS 生成后**对半分**(一半 correction-based SFT、一半 RL);RL 最大 rollout 步数=8。
- 致谢 verl、LLaMA-Factory、ARPO(评测集)、agent-distillation(prompt 灵感)。代码可得性 Tier A。


---


### search_dont_guess — Search, Do not Guess: Teaching Small Language Models to Be Effective Search Agents

> **一句话重点 (TL;DR)**：小模型(SLM)做 search agent 时反而比大模型**更少搜索、更易幻觉**(under-searching),且"自适应搜索"在 SLM 上会掉点(Adaptive Search Trap);本文用 **Always-Search Policy(ASP)** 显式约束"总是搜索、别猜",经 SFT/OPD/Mixed 三种实现把搜索行为蒸进 SLM,使 1.7B 逼近甚至局部超过 8B。

**元信息**：arXiv 2604.04651 (v1, 2026-04-06, cs.AI) ｜ NYU + UIUC + NTU(Yizhou Liu*、Qi Sun*、Yulin Chen、Siyue Zhang、Chen Zhao;*共同一作) ｜ arXiv 预印本 ｜ 主题 agentic RAG / SLM 蒸馏,Relevance=Med(把 **OPD** 作为三训练变体之一与 SFT、Mixed 实测对比,提供 OPD 在"让小模型坚持搜索"上的具体实证) ｜ 代码 github.com/yizhou0409/Agentic-Rag(100% Python) ｜ 框架 检索 E5+BM25 / Search-o1 式迭代 agent / 训练 SFT+OPD+RFT

#### 1. 相关工作与进展
带搜索工具的 agent 是知识密集任务的有效方案,但依赖 ≥7B LLM,部署成本高。近期工作把 agentic 行为蒸到 <4B SLM(Agent Distillation 路线)。本文在此之上做全面评测并提出针对 SLM 的训练 recipe。

#### 2. 现有工作存在的问题
全面评测发现:尽管参数知识更少,SLM 反而**更少调用搜索、更易幻觉**(under-searching,Figure 2)。直接蒸 LLM 轨迹效果差——LLM 轨迹常含"I remember…"式不搜索直接答的行为,被 SLM 学坏。更关键:让模型自己决定要不要搜的**自适应搜索在 SLM 上反而掉点**("Adaptive Search Trap")。

#### 3. Motivation
作者用 **confidence probe** 检验 SLM 内部知识可靠性:SLM 高置信 query 远少于 teacher。⇒ SLM 不该依赖内部知识"猜",应"总是搜索、别猜"。需要一种训练显式约束搜索行为、强制 grounded 生成。

#### 4. 主要灵感 / 核心直觉
对 SLM 而言,"一致地总是搜索"优于"自适应地按置信度决定是否搜索"——后者把不可靠的内部置信引入决策,放大幻觉。把这一先验直接编码进训练目标。

#### 5. 主要解决思路(一段话讲清核心)
提出 **Always-Search Policy(ASP)**:训练时显式约束搜索行为,三种实现——(1) **SFT**:只保留"始终用搜索工具获取信息"(剔除"I remember"类)的高质量轨迹做监督;(2) **OPD(on-policy distillation)**:不显式过滤轨迹,而是用 system prompt 要求模型总是搜索,期望 teacher 的 log-prob 分布在 student 自身 rollout 上调控其搜索行为;(3) **Mixed**:先 ASP-SFT 再 OPD 强化。下游再加 **RFT(Rejection Fine-Tuning)** 选择性强化高质量 agentic 行为。整体是工程化 recipe,非新算法。

#### 6. 方法详解(通俗、分步骤)
- 推理:Search-o1 式迭代 search agent,检索 = E5(embedding)+ BM25(keyword)双检索,summarizer = Qwen3-32B。
- **ASP-SFT**:从 HotpotQA 采 **18,000** 条 Qwen3-32B teacher 轨迹(String-F1>0.65 保留;且只留"始终搜索"轨迹),**3 epochs、lr 1e-5** 蒸馏。
- **ASP-OPD**:**3,000** 个 HotpotQA 训练题,每题学生采样 **8** 条轨迹,teacher 给 token 分布、最小化 KL,**4 epochs、lr 2e-6**;靠 system prompt 强制搜索而非显式过滤。
- **RFT**:学生自生成 **10,000** 条轨迹(与蒸馏用不同题),**2 epochs、lr 5e-6**,拒绝采样保留高质量行为。

#### 7. 实验数据集
- **训练仅用 HotpotQA 训练集**(teacher = **Qwen3-32B**)。
- 评测:结构化多跳 HotpotQA / 2WikiMultiHopQA / Bamboogle / MuSiQue;agentic 信息检索 BrowseComp-plus / Frames / LongSeAL。指标 **String-F1**(也报 EM)。
- 检索器 e5-large-v2 + fullwiki-20210620 语料(BrowseComp-Plus 用 Qwen3-Embedding-8B + 自带语料)。
- 模型:Qwen3-0.6B/1.7B/4B/8B/32B、Llama-3.2-1B/3B。

#### 8. 实验结果与主要发现
- **逼近大模型**(Table 1,String-F1):三种 ASP 法均使 1.7B 逼近 Qwen3-8B——HotpotQA 上 Mixed-1.7B = **58.2** ≈ 8B(58.2);**2Wiki 上 OPD-1.7B = 62.9,超过 8B(58.1)**。(注:上一轮已更正"OPD 56.2→62.9"的误读——56.2 是 OPD 的 HotpotQA 分、62.9 是其 2Wiki 分;本轮再核 Table 1 OPD-1.7B 行 = 56.2/62.9/61.4/… 属实。)
- **泛化**:只在 HotpotQA 训练却泛化到 OOD(BrowseComp/Frames/LongSeAL)。
- **搜索频率**(§4.2):vanilla 1.72 → ASP-SFT 2.47 → ASP-OPD 2.84 搜索/题(Vanilla-8B 亦 2.84)。OPD 在多个表上搜索频率最高,是本文 OPD 有效性的具体实证点。
- **噪声鲁棒**(10% 检索失败):vanilla SLM / 普通蒸馏掉 **12.1** 分,ASP 仅掉 **2.3 / 1.7**,显示更强恢复能力。
- **Adaptive Search Trap**:Adaptive Distill(Top-5/10/20% confidence)在 Bamboogle 等上劣于一致搜索的 ASP。

#### 9. 结果如何支撑其主张
"SLM under-search"由 Figure 2 + confidence probe 支撑;"一致搜索优于自适应"由 Adaptive Distill 各档掉点支撑;"ASP 让 SLM 逼近 LLM"由 1.7B≈8B、2Wiki OPD>8B 支撑;"更 grounded/抗噪"由噪声实验 2.3/1.7 vs 12.1 支撑。支撑较直接。但"训练仅用 HotpotQA"既是泛化卖点也是局限——单源单任务,泛化结论的可靠性受训练分布单一性限制。

#### 10. 逻辑自洽性(中性评估)
recipe 与诊断对齐,叙事自洽。中性看待:(1) 三种 ASP 实现孰优缺乏统一胜者(SFT/OPD/Mixed 在不同 benchmark 互有高低),论文未给清晰选择准则;OPD 的优势主要体现在搜索频率与个别 benchmark,而非全面占优;(2) "总是搜索"在 query 本可由内部知识快速回答时会增加延迟/成本(Table 5 显示 OPD-1.7B 端到端 ~3.1s),论文承认但未量化"过度搜索"代价;(3) 仅 Qwen3 系评测,跨家族结论保守(论文亦自述局限)。

#### 11. 残留问题 / 局限
- 训练单一(仅 HotpotQA 单跳/多跳混合),跨域/跨任务训练分布缺失。
- "总是搜索"对简单 query 引入不必要搜索开销,无最优搜索预算分析。
- 检索噪声处理仅做 10% 失败注入的鲁棒性测试,缺更系统的对抗检索机制(论文列为 future work)。
- 评测集中于 Qwen3 家族(Llama-3.2 仅部分),跨架构普适性未充分验证。
- 仓库以语料/检索/推理脚手架为主,SFT/OPD/RFT 训练实现细节相对简略(在 appendix B)。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库:https://github.com/yizhou0409/Agentic-Rag(100% Python)。含 `build_corpus/`(`extract_wiki.py`、`build_e5_corpus.py`、`merge_e5_splits.py`:Wikipedia 语料构建/索引)、`e5_retriever.py` + `bm25_retriever.py`(双检索)、`main.py`(推理 pipeline)、`prompts/`(`default_QA.yaml`、`default_retrieval_summary.yaml`、`free_QA.yaml`)、`utils.py`。
- 框架:检索 E5(embedding)+ BM25(keyword);推理 Search-o1 式迭代 search agent;训练含 SFT、OPD、RFT 三阶段(trainer 实现见 appendix,仓库内偏脚手架)。
- 代码可得性:Tier B——语料/检索/推理可跑,但训练(SFT/OPD/RFT)脚本不完整,复现需补 appendix B 细节。


---


### search_r1 — Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning

> **一句话重点 (TL;DR)**：把搜索引擎调用嵌进 RL 训练循环——用结构化标签做多轮"推理↔检索"交织生成,对检索回来的 token 做 **loss masking**,仅用最简的 **EM 结果奖励**(无过程/格式奖励),即可让 LLM 自发学会"何时检索、检索什么、如何结合检索继续推理"。

**元信息**：arXiv 2503.09516 (v5, 2025-08-05;COLM 2025 会议论文);实证扩展见 arXiv 2505.15117 ｜ UIUC(Bowen Jin、Zhenrui Yue、Dong Wang、Jiawei Han)+ UMass Amherst(Hansi Zeng、Hamed Zamani)+ Google Cloud AI Research(Jinsung Yoon、Sercan Ö. Arık) ｜ 2025-03 首发,COLM 2025 ｜ 主题 T2(agentic/工具调用 RL),与本课题相关:多轮"推理-检索"交织端到端 RL、检索 token loss masking、纯结果奖励——可作 agentic OPD 的对照/底座 ｜ 代码 github.com/PeterGriffinJin/Search-R1(Tier A,~8.3M) ｜ 框架 veRL 定制 fork

#### 1. 相关工作与进展
LLM 推理时常需外部知识/时效信息。已有两类:(1) **RAG**——单轮"检索→生成",query 质量与多轮交互受限;(2) **搜索当工具**——提示式/SFT 式工具调用泛化差。R1/o1 表明纯结果奖励 RL 能自发涌现推理能力,本文将其扩展到"会调用搜索引擎"的场景。已被 veRL 官方、SkyRL、Thinking Machines Tinker-cookbook 集成复现。

#### 2. 现有工作存在的问题
(1) 如何把搜索引擎**稳定地**嵌入 RL 训练循环——检索回来的 token 不应被当作策略生成 token 优化(否则训练不稳);(2) 是否需要复杂过程奖励,还是简单结果奖励即可;(3) 多轮交织检索-推理的结构化建模缺失。

#### 3. Motivation
让 LLM 在 RL(规则化结果奖励)中自主学会"何时检索、检索什么、如何结合检索结果继续推理",无需过程监督或检索标注,得到完全开源、可替代 OpenAI DeepResearch 的工具增强推理 RL 方案。

#### 4. 主要灵感 / 核心直觉
检索内容是"环境观测"而非策略输出,故须从策略梯度中屏蔽;而最终答案正确与否(EM)已是足够的学习信号,无需精心设计过程奖励——把复杂性留给模型自己在 RL 中涌现。

#### 5. 主要解决思路(一段话讲清核心)
用结构化标签把"多轮推理-检索"显式建模,生成遇 `</search>` 即抽 query 调检索、把结果包进 `<information>` 拼回上下文继续生成;在 PPO/GRPO 的 token 级损失上对**检索回来的 token 做 masking**(只对 LLM 生成 token 计损),仅用 EM 结果奖励 + 对 π_ref 的 KL 约束做策略更新。

#### 6. 方法详解(通俗、分步骤)
- **多轮交织生成**:结构化标签 `<think></think>`(推理)、`<search></search>`(触发检索的 query)、`<information></information>`(检索返回内容)、`<answer></answer>`(最终答案)。
- **检索 token 的 loss masking**:对 PPO/GRPO token 级损失,只对 LLM 生成 token 计损(I(y_t)=1),检索 token 置 0,避免对外部 token 求梯度致训练不稳。
- **奖励设计**:仅用规则化结果奖励(基于答案 Exact Match),不用过程奖励或格式奖励(论文论证最简奖励已足够)。
- **目标函数**:带 KL 约束(对 π_ref)的策略梯度,兼容 PPO 与 GRPO;**默认 PPO**("Unless stated otherwise, PPO is used as the default RL method")。

#### 7. 实验数据集
- **训练**:合并 **NQ + HotpotQA** 训练集。
- **评测(7 QA)**:单跳 NQ、TriviaQA、PopQA;多跳 HotpotQA、2WikiMultiHopQA、Musique、Bamboogle(in-domain=NQ/HotpotQA,余 5 项 OOD)。
- **知识源**:2018 Wikipedia dump;检索器 **E5**(稠密);检索文档数主实验固定 **top-k=3**(另做 k=1/3/5 消融)。指标 **Exact Match(EM)**。
- **模型**:Qwen2.5-3B/7B(Base 与 Instruct);Llama3.2 见后续实证扩展版(2505.15117),COLM 主文仅用 Qwen2.5。

#### 8. 实验结果与主要发现
- **主结果(EM,Table 2)**:Qwen2.5-7B Search-R1-base Avg = **0.431** vs RAG **0.304**,相对提升约 24%;Qwen2.5-3B 约 20%(同检索器/语料/训练集/预训练模型设定)。
- **口径差异(论文内)**:摘要写"24%(7B)/20%(3B)";贡献处另给"**41% / 20%**"(同一 setup 下不同 baseline 取法)。两者并存,摘要为主口径。
- **PPO vs GRPO(§5.1)**:GRPO 收敛更快但长训后**奖励坍缩**,PPO 更稳定;两者最终奖励相当。
- **奖励/标签消融**:简单 EM 结果奖励已足够;Base 与 Instruct 模型均能自发学会多轮检索-推理行为。

#### 9. 结果如何支撑其主张
"搜索可稳定嵌入 RL"由 loss masking 下训练稳定 + 跨 7 数据集一致增益支撑;"简单结果奖励足够"由无过程奖励仍涨点支撑;"自发涌现检索-推理"由 Base 模型也学会多轮行为支撑。支撑充分。中性提示:24% vs 41% 两口径并存易致引用混淆,需注明 baseline 取法。

#### 10. 逻辑自洽性(中性评估)
方法-实验-结论自洽,是该方向的奠基工作之一(被广泛复现佐证其可靠性)。中性看待:(1) EM 作奖励对"语义正确但表述不同"会误判,可能压低真实能力上限并诱导格式拟合(论文未深究 reward hacking);(2) in-domain 仅 NQ/HotpotQA 两源,OOD 增益虽存在但训练分布仍偏窄;(3) 24%/41% 双口径属表述瑕疵,非逻辑硬伤。

#### 11. 残留问题 / 局限
- EM 结果奖励的语义盲区(同义/格式差异误判),可能限制上限并引入 reward hacking 空间。
- 检索质量(E5 + 2018 Wikipedia 静态语料)是上界约束;时效性、在线搜索噪声未在主实验充分压测。
- GRPO 长训坍缩问题只描述未给根因/修复方案(留作默认 PPO 规避)。
- 主文仅 Qwen2.5;跨架构(Llama)结果在扩展版(2505.15117),COLM 主文不含。
- 摘要(24%/20%)与贡献(41%/20%)口径不一,引用时须注明。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库:https://github.com/PeterGriffinJin/Search-R1(origin 已核,Tier A,~8.3M)。含仓库内自带 `verl/`(基于 veRL 的定制 fork,见 `VERL_README.md`=HybridFlow/veRL)、`train_ppo.sh`、`train_grpo.sh`、`retrieval_launch.sh`、`search_r1/{llm_agent, search}`(核心逻辑)、`scripts/data_process`(NQ/HotpotQA 数据处理)、`infer.py`。
- 框架:**veRL**(仓库内置定制版);rollout 用 vLLM(0.5.x/0.6.x);支持 PPO、GRPO、REINFORCE 等(默认 PPO);检索支持本地稀疏(BM25)/稠密(E5+ANN)与在线搜索引擎。
- 复现:启动检索服务(`retrieval_launch.sh`)→ NQ+HotpotQA 构造带结构化标签 prompt → veRL 跑 PPO/GRPO rollout(遇 `<search>` 调检索、`<information>` 拼回)→ 仅对生成 token 计损(检索 token masked)→ EM 结果奖励 + KL 约束更新。
- 代码可得性:Tier A(完整可跑,社区多方复现)。


---


### sod_stepwise — SOD: Step-wise On-policy Distillation for Small Language Model Agents

> **一句话重点 (TL;DR)**：把 OPD 用于小模型 agent 的工具集成推理（TIR）会因工具错误触发的"加速分布漂移"而训练崩溃；SOD 按 step 级师生发散自适应重加权蒸馏强度（高发散区衰减、重对齐时回升），在对齐区保留 dense 监督，使 0.6B/1.7B 学生相对最强基线 OPD 平均 +20.86%/+18.50%。

**元信息**：arXiv 2605.07725v1（2026-05-08 Preprint） ｜ 浙大 + 腾讯 LLM 部 + 中科大 + 新加坡国立（Qiyong Zhong、Mao Zheng、Mingyang Song 共同一作；Junfeng Fang、Houcheng Jiang 通讯） ｜ 主题 T1/T2 High（OPD + 小模型 agent TIR + step 级自适应重加权，与本项目 path-recovery / MTP foresight 探针直接呼应） ｜ 代码 https://github.com/YoungZ365/SOD （已克隆约 23MB，基于 verl，含 recipe/examples/docker） ｜ 框架 veRL fork + Open-AgentRL + ReTool agentic 组件

#### 1. 相关工作与进展
Agentic 能力多依赖大模型，推理成本高；将其迁移到可端侧部署的 SLM 有实践价值。TIR 后训练主流基于 RL（GRPO），只给稀疏 outcome 级 reward。OPD（on-policy distillation）提供 dense token 级监督，缓解 credit assignment、提升样本效率与稳定性，是把大模型 agentic 能力蒸到小模型的自然候选。训练数据/框架沿用 Yu et al.[52]（"Demystifying RL in agentic reasoning"，对应仓库 `recipe/demystify/`）与 ReTool。

#### 2. 现有工作存在的问题
小模型容量有限、探索弱，稀疏 outcome 监督会加剧探索失败、陷入 cold-start。但直接把 OPD 用于 SLM-based TIR 会出现严重训练不稳定/崩溃。根因：TIR 经工具交互引入"非连续状态跳变"——单次错误工具调用注入错误观测，使后续推理在被污染状态上展开；teacher 工具使用准而小模型弱，早期累积多次错误工具调用，状态分布迅速偏离 teacher。在这些 OOD 状态上 teacher 的 token 级监督变得不可靠甚至误导（Fig.1：错误轨迹上 teacher 熵均值与方差在后段急剧增大），错误沿后续推理步级联放大，放大师生发散并诱发不稳定梯度。

#### 3. Motivation
区别于文本推理中渐进式的分布漂移，TIR 的发散是由工具错误触发的"加速"漂移。需要按 step 级发散自适应调节蒸馏强度：在对齐良好的状态保留 dense 监督，在高发散状态衰减可能误导的 teacher 信号。

#### 4. 主要灵感 / 核心直觉
用相邻 step 的发散比（而非绝对发散值）来调权：发散单调上升时累乘比值 <1、自动压低被污染信号权重；一旦重新对齐（d 下降）则允许蒸馏强度回升。仅依赖比值意味着任何与真实师生发散 ∆k 单调一致的可观测代理 dk 皆可用（Appendix D.5）。

#### 5. 主要解决思路(一段话讲清核心)
定义 step-level divergence score d_k 作为 teacher 监督可靠性的可观测代理；据相邻 step 发散比累乘得到每步权重 w_k（上界 1+δ）；把 w_k 作用于该 step 内所有 token 的蒸馏损失，从而在高发散区衰减、在良好对齐区保留 dense 指导。整体目标为 GRPO + 加权 OPD 的联合损失。

#### 6. 方法详解(通俗、分步骤)
- d_k：实现为该 step 内 token 的 mean(|log π_θ − log π_teacher|)，作为 teacher 监督可靠性代理。
- 权重（Eq.7）：w_1=1；step k≥2 权重为 1..k−1 各相邻发散比 (d_u+ε)/(d_{u+1}+ε) 的**累乘**，并以 1+δ 为上界（δ=0.2，ε=1e−6）。发散单调上升 → 累乘 <1 抑制被污染信号（Appendix D.4 证加权二阶矩被压制 O((d1/dk)²)）；d 下降（重对齐）→ 蒸馏强度回升。
- 施加粒度：w_k 作用于该 step 全部 token（非逐 token 均匀施加）。
- 联合目标：L = L_GRPO + L_OPD（OPD 系基线均用此口径）；开销可忽略（d_k/w_k 复用 OPD 前向已有的师生 logprob，仅 O(K) 标量运算）。

#### 7. 实验数据集
- 训练数据（沿用 Yu et al.[52]）：3k 高质量多轮推理 SFT 语料（s1-1k 1k + LeetCode 1k + ReTool 1k，后两者经 ReasonFlux-PRM 打分各取 top-1k）+ ~30k RL 数据（DAPO-Math 17k + Skywork-OR1 Math 4902 / Code 3586 + MegaScience 3k）。SFT 轨迹由 Qwen3-Coder-30B-A3B 在 SandBoxFusion 环境内端到端交互生成。
- 评测（均报 average@32，百分比；temp=1.0、top_p=0.6、每题 32 采样）：Math=AIME 2024/2025、Science=GPQA-Diamond、Code=LiveCodeBench（v6 近期窗）。
- 模型：teacher 为 Qwen3-4B 经 GRPO 在 RL 数据上进一步优化（默认 4B teacher）；student 为 Qwen3-0.6B 与 Qwen3-1.7B。

#### 8. 实验结果与主要发现
- 主结果：SOD 在两个 student 上对第二好基线（OPD）相对平均提升 0.6B +20.86%、1.7B +18.50%；四任务均最高分。0.6B student 在 AIME 2025 达 26.13%（average@32），据称为首个达到该水平的 sub-billion 模型。
- 基线含 SFT、GRPO 及多种蒸馏方法（共 6 个，Appendix B.3）；SFT/GRPO 单独均显著弱于蒸馏系。
- 开销（Table 4）：d_k/w_k 仅 O(K) 标量运算，显存差 <0.5GB；0.6B 上 SOD 反而比 OPD 快 3.5%（1052.3s vs 1090.5s，因自适应重加权抑制了错误学习、失败重试更少），1.7B +4.9% 开销（1105.4s vs 1053.6s）。
- 稳定性：错误轨迹后段 teacher 熵均值/方差急剧增大（Fig.1），SOD 重加权压制该区监督，缓解崩溃。

#### 9. 结果如何支撑其主张
"OPD 在 SLM-TIR 上不稳"由 Fig.1 的发散/熵证据支撑；"step 级重加权有效"由消融（Table 2 移除 step-wise 退化）与主结果（对 OPD 的 +20.86%/+18.50%）支撑；"开销可忽略"由 Table 4 支撑；理论上加权方差被压制有 Appendix D.4 证明。证据链较完整，5 个随机种子重复增强了可靠性。

#### 10. 逻辑自洽性(中性评估)
方法-动机-理论-实验四者自洽：动机（工具错误触发加速漂移）→ 代理 d_k（师生 logprob 差）→ 比值累乘权重（Eq.7）→ 方差压制证明（D.4）→ 代理单调一致性证明（D.5）→ 消融。一个口径需注意：主卖点是"对最强基线 OPD 的相对平均提升"（百分比口径），绝对点数提升（如 0.6B 整体由约 21→24+ 区间）规模较小，相对数放大了观感。

#### 11. 残留问题 / 局限
- 模型/工具单一：仅 Qwen3 单一模型族、python 解释器（SandBoxFusion）单一工具环境验证；作者亦将 web/API 等其他 agent 设置与其他模型族列为局限。
- 相对增益口径：+20.86%/+18.50% 为对 OPD 的相对百分比而非绝对点数，需结合绝对值理解。
- d_k 代理依赖师生 logprob 差，teacher 自身在 OOD 状态的 logprob 可靠性边界未充分刻画（高熵区 logprob 噪声大可能反噬 d_k 估计）。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库 https://github.com/YoungZ365/SOD （已克隆约 23MB）。框架：veRL[76] + Open-AgentRL[52]（`recipe/demystify/`，复用其 sandbox_fusion 工具配置）+ ReTool（`recipe/retool/`）；rollout 走 vLLM（TP=4），SandBoxFusion 作 python 解释器、最多 16 轮工具调用。
- **〔重要更正——核心算法已开源〕** 与早前"step-wise 重加权核心损失未在仓库定位到"的判断相反：核心实现确在已发布代码中。`verl/trainer/ppo/ray_trainer.py:363 compute_stepwise_opd_weights` 完整实现 Eq.6/7——按 response_mask 提取 step 边界、`d_k = mean(|log π_θ − log π_teacher|)`、`w_1=1`、`w_k = min(∏_{u=1}^{k-1}(d_u+ε)/(d_{u+1}+ε), 1+δ)`、再 broadcast 到该 step 全部 token；由 `ray_trainer.py:878 _apply_token_kl_regularizer`（line 1622 处调用）将其乘到 OPD 优势项 `weighted_opd = opd_coef * stepwise_weights * raw_local_adv`。配置见 `verl/trainer/config/algorithm.py` 的 `TokenKLRegConfig`（stepwise_enable/epsilon/delta/opd_coef），运行脚本 `examples/SOD/run_sod.sh` 显式传入 `+algorithm.token_kl_reg.stepwise_*`。代码与论文 Eq.7 精确一致，可复现。
- **〔次要差异〕** 配置 dataclass 默认 `stepwise_delta=0.5`，但 `run_sod.sh` 覆写为 0.2（论文值），复现须用脚本而非 dataclass 默认。
- 训练硬件：单节点 8×H20（96GB）；0.6B/1.7B 1 epoch 约 2–3 天，4B/14B teacher 约 5–6 天；统一超参（AdamW lr=1e-6、batch 64、mini-batch 16、prompt≤2560、response≤20480、训练每 prompt 采 16、验证 32）；所有实验 5 个随机种子重复。


---


### spear — SPEAR: Learn the Ropes, Then Trust the Wins — Self-imitation with Progressive Exploration for Agentic RL

> **一句话重点 (TL;DR)**：针对多轮 agent RL 中机械熵最大化易致不稳定的问题，SPEAR 用"课程化自模仿学习（SIL）+ 内在奖励塑形"在自身经验引导下渐进调节策略熵（早期广探索、后期收敛利用），在 ALFWorld/WebShop/Sokoban/AIME 上稳定提升 GRPO/GiGPO/Dr.BoT，且额外开销仅理论 10%–25%。

**元信息**：arXiv 2509.22601v4（Date 标注 2025-09-22；v4 2025-12-07 cs.LG） ｜ 腾讯优图 Youtu-Agent Team（通讯 yuleiqin/arthurtan@tencent.com） ｜ venue 〔待核：早前分析称"ICLR 2026 接收"，但 PDF 正文/页眉/页脚均无接收声明，全文唯一 ICLR 出现处为 ReAct[4] 的 ICLR 2023 引用；会议接收状态未经一手来源证实，已删除该断言〕 ｜ 主题 T2 agentic RL（自模仿 + 课程化探索管理长程稀疏奖励，replay buffer 存好轨迹做 off-policy 更新——与 OPD"用优质轨迹引导学生"思路相通） ｜ 代码 https://github.com/TencentYoutuResearch/SPEAR （已 clone 约 96MB，含 verl/ + verl-agent/，HF 有模型 collection） ｜ 框架 veRL + verl-agent

#### 1. 相关工作与进展
RL 是磨炼 LLM 长程、稀疏奖励 agent 任务中策略性工具使用能力的主流范式（基于 ReAct 范式，应用含机器人导航、移动助手、web 导航、deep search、GUI），核心难题是探索-利用权衡。已有工作多从策略熵视角刺激探索（熵最大化/正则化技术），也有用 cold-start SFT 或 RL+SFT 混合方案提稳定性。

#### 2. 现有工作存在的问题
- 纯熵控制在多轮 agent 中脆弱：环境反馈带来的低概率 token 累积引发严重分布漂移，导致 mode collapse；agent 对多轮交互的不确定性会引起持续熵增长（runaway divergence）或塌缩，训练不稳。
- cold-start SFT / RL+SFT 混合虽提稳定，却限制策略发现 SFT 语料之外的新策略。
- 用 replay buffer 做 off-policy 更新时优势函数需重算、且 off-policy 数据带来不稳定。

#### 3. Motivation
能否在策略自身经验引导下平滑调度"何时探索、何时利用"，把策略熵维持在"随时间演化、动态但受控"的区间——早期增大熵做 skill-level 广度探索，后期收敛熵做 action-level 利用/巩固——以实现渐进式探索-利用平衡，而非机械熵最大化。

#### 4. 主要灵感 / 核心直觉
"先学规矩，再信战果"（Learn the Ropes, Then Trust the Wins）：前期靠辅助工具使用奖励促频繁工具交互、广探索；待对环境熟悉后再强化对成功轨迹的自模仿利用。用课程调度把这一直觉操作化为跨阶段的两路权重调整。

#### 5. 主要解决思路(一段话讲清核心)
在 vanilla SIL（独立 replay buffer 仅存回报超基线的 state-action 对、做 off-policy 更新）基础上，用课程跨阶段联合调节"内在奖励塑形"与"自模仿"两路权重；对 buffer 经验做优势重校准以避免重算优势，并正则化策略更新以稳熵、抑制 reward hacking。训练 batch 同时含 on-policy 与 replay buffer 的 off-policy 数据。

#### 6. 方法详解(通俗、分步骤)
基于 group-based RL（类 GRPO）：同一输入生成一组多轮工具交互轨迹 → episode 级奖励计算与优势估计 → 走两条路：
1. **课程调度**：跨阶段联合调"内在奖励塑形"与"自模仿"权重——前期辅助工具使用奖励促 skill-level 探索，后期强化 self-imitation 利用成功轨迹。
2. **优势重校准（advantage recalibration）**：处理 buffer 经验的 off-policy 性质，避免对历史经验重算优势。
3. **策略更新正则化**：稳定熵、缓解 reward hacking / 分布漂移。
数据流：(a) 内在奖励塑形 + 优势估计 + on-policy 更新（类 vanilla GRPO）；(b) 过滤优质轨迹入 replay buffer，经优势重校准 + 正则化做 self-imitation off-policy 更新；课程跨阶段调两路权重以控熵区间。基于 verl-agent 实现多轮 rollout。

#### 7. 实验数据集
- agent benchmark：ALFWorld（文本具身交互）、WebShop（网购）。SPEAR 对 GRPO/GiGPO/Dr.BoT 分别带来最高 16.1%/5.1%/8.6%（ALFWorld）与 20.7%/11.8%/13.9%（WebShop）成功率提升。
- Sokoban 视觉推箱子（Qwen2.5-VL-3B-Instruct，Table 5）：SPEAR 把 GRPO 67.1→86.7（+19.6%）、Dr.BoT 76.0→85.4（+9.4%）。
- 数学推理 AIME24/AIME25（带 code interpreter）：SPEAR 对 Dr.BoT 分别 +3.8%/+6.1%。
- 模型：Qwen2.5-1.5B/7B-Instruct、Qwen2.5-32B（code interpreter）、Qwen2.5-VL-3B-Instruct（Sokoban）。
- 开销：理论复杂度仅 +10%~25%，实际每迭代运行时开销可忽略。

#### 8. 实验结果与主要发现
- 跨四类任务（文本 agent、视觉 agent、数学+工具）对多个强基线一致提升，验证 plug-and-play 可扩展性。
- 构建并对比强工业基线 Dr.BoT（bag-of-tricks of industrial RL optimizations），SPEAR 在其上仍有正增益，说明增益非来自弱基线。
- 课程化 SIL 把熵维持在受控区间，避免熵塌缩与 runaway divergence（对应 Motivation）。

#### 9. 结果如何支撑其主张
"渐进式探索-利用优于机械熵最大化"由跨基线（GRPO/GiGPO/Dr.BoT）一致正增益支撑；"非弱基线红利"由对自建强基线 Dr.BoT 仍有提升支撑；"plug-and-play、低开销"由 +10%~25% 理论复杂度 + 可忽略运行时开销支撑。证据覆盖四类任务，外部效度较好。

#### 10. 逻辑自洽性(中性评估)
方法-动机自洽：针对"机械熵最大化不稳"提出"经验引导的受控熵调度"，三项改造（课程、优势重校准、正则化）分别对应稳定性的三个来源（探索-利用时序、off-policy 偏差、熵/hacking）。一个评估张力：SPEAR 由"课程调度 + 优势重校准 + 正则化 + 内在奖励塑形"多组件叠加，论文虽给跨基线增益，但各组件的独立消融与课程超参敏感性需结合附录核验方能判断增益归因。

#### 11. 残留问题 / 局限
- venue 状态未经一手证实（见元信息），引用时勿标注会议接收。
- 多组件配方，增益归因到单一机制（课程 vs 优势重校准 vs 正则化）需更细消融支撑。
- 内在奖励塑形依赖"辅助工具使用奖励"的设计，跨环境可迁移性与 reward hacking 风险（虽有正则化缓解）需进一步验证。
- 课程跨阶段权重的调度依赖阶段划分超参，自动化/自适应程度有限。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库 https://github.com/TencentYoutuResearch/SPEAR （已 clone 约 96MB，最新 commit 0edca96）。
- 结构含 `verl/` 与 `verl-agent/` 两个子目录（在 veRL + verl-agent 之上实现）；verl-agent 提供 GiGPO 等 group-based RL 的多轮 agent 扩展。
- README 明确为 curriculum-based SIL 框架，给出 Self-imitation Learning 配置项（`enable_trajectory_replay` 是否启用 self-imitation loss、replay buffer 最大轨迹数等），与论文方法一致。基线含 GRPO、GiGPO、作者构建的强基线 Dr.BoT。HF 有发布模型 collection（yolay/spear-…）。代码可得、配置可定位。


---


### sweet_rl — SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks

> **一句话重点 (TL;DR)**：在多轮 agent 任务上，用训练期才可见、actor 看不到的额外信息（最终结果 + 参考解）训练一个非对称的 turn-level critic 做 step 级信用分配；关键设计是"直接用动作 log-prob 参数化 advantage 函数 + Bradley-Terry 目标"，避开在 LLM 上训 value head 损泛化的问题；并配套开源多轮协作 benchmark ColBench。

**元信息**：arXiv 2503.15478（v1, 2025-03-19, cs.LG）｜ FAIR at Meta + UC Berkeley（Yifei Zhou, Song Jiang, Yuandong Tian, Jason Weston, Sergey Levine, Sainbayar Sukhbaatar\*, Xian Li\*）｜ arXiv 2025-03 ｜ 主题 多轮 agent step 级信用分配 + 新 benchmark / 相关性 中高（非对称 critic 用 teacher 特权信息做信用分配，与 TSRD 把 teacher 参考路径/foresight 用于 scaffold/信用分配同构，可作 asymmetric-critic 路线代表 baseline；不涉及 MTP）｜ 代码 https://github.com/facebookresearch/sweet_rl ｜ 框架 OpenRLHF 定制 fork（`YifeiZhou02/collab_openrlhf`，支持 multi-turn DPO + length normalization）

#### 1. 相关工作与进展
LLM agent 需在真实任务中做多轮交互；要在序列决策任务上拿到最好性能，需直接优化多轮目标（如成功率），比 next-token 预训练只模仿每轮最可能动作更难。现有多轮 RL 路线包括 RAFT/DPO/PPO 系单轮方法外推、value-function/TD-learning、Path Consistency 等。

#### 2. 现有工作存在的问题
- 把单轮 RLHF 方法（RAFT/DPO/PPO）直接用于多轮、不做跨轮显式信用分配，长 horizon 下高方差、样本复杂度差。
- value function 学习（TD-learning）需在 LLM 表征上训任务专用 value head，微调数据有限时泛化差（与 next-token 预训练目标差异大）。
- 缺乏同时满足三条件的 benchmark：(1) 任务多样性避免过拟合；(2) 足够复杂挑战推理/泛化；(3) 最小工程开销便于快速原型。现有 benchmark 无一兼具。

#### 3. Motivation
利用一个常被忽视的事实：训练期可拿到 actor 推理时拿不到的额外信息（最终 outcome、reference solution）。这种非对称信息可为"信息搜寻型"动作的信用分配提供捷径。

#### 4. 主要灵感 / 核心直觉
直接拿训练期额外信息训标量 value 会偏离 next-token 目标、损泛化；故改为**直接学 advantage 函数**（用每轮动作平均 log-prob 参数化）并用偏好（Bradley-Terry）目标对齐 LLM——这比"在 hidden state 上训 value head"更契合预训练 LLM。

#### 5. 主要解决思路(一段话讲清核心)
两阶段：先用训练期额外信息 c（actor 不可见）训一个 turn-level critic，直接建模 turn-wise advantage（用动作 log-prob 参数化、BT 轨迹级目标），再把该 advantage 当每轮 reward model 对 actor 做 RLHF 式优化。配套提出 ColBench 多轮协作 benchmark 做评测。

#### 6. 方法详解(通俗、分步骤)
两部分贡献：
1. **ColBench（Collaborative Agent Benchmark）**：聚焦 artifact creation 的多轮人机协作任务，含 Backend Programming（写 Python 函数）与 Frontend Design（做网页）。用 LLM 当 human simulator 且给其 reference artifact 以保证忠实模拟；用 functional evaluator（单元测试 / 图像相似度）客观度量产出与参考的相似度。>10k 程序化生成任务，可扩展、可调难度。
2. **SWEET-RL（RL with Step-WisE Evaluation from Training-time information）——两阶段**：
   - 阶段一（训 critic/advantage）：用训练期额外信息 c（如 reference solution，actor 不可见）训 turn-level critic，利用 critic 与 actor 的**非对称观测空间**。**直接建模 turn-wise advantage 函数**（用每轮动作平均 log-prob 参数化），不先学 value 再求 advantage。同任务下取两条轨迹按累积奖励标 chosen/rejected，在轨迹级用 **Bradley-Terry 目标**训该 advantage 函数。
   - 阶段二（policy improvement）：把训好的 advantage 函数当每轮 reward model，对 actor 做 RLHF 式 per-step 优化。

#### 7. 实验数据集
ColBench 两任务：Backend Programming、Frontend Design。Backend 数据生成：用 Llama-3.1-70B-Instruct 从 DCLM 抽取片段生成 Python 函数 + 高层描述 + 单元测试，仅保留能过单测的；train 10k、test 1k（test 经作者人工检查）。15k 离线训练轨迹由 Llama-3.1-8B-Instruct 当 agent、Llama-3.1-70B-Instruct 当 human simulator 零样本生成。backbone：Llama-3.1-8B（agent）。

#### 8. 实验结果与主要发现
- 相较其他 SOTA 多轮 RL，ColBench 成功率与 win rate 绝对 +6%。
- 使 Llama-3.1-8B 在协作内容创作上匹配/超过 GPT-4o（部分超 o1-mini）。
- baselines 含 RAFT/DPO/PPO 系、value-function/Bellman bootstrapping/Path Consistency 路线，及闭源 GPT-4o、o1-mini。

#### 9. 结果如何支撑其主张
在自建 ColBench 上对一组 SOTA 多轮 RL 的一致 +6% 绝对提升，加上匹配 GPT-4o 的端到端表现，支撑"非对称 critic + 直接学 advantage + BT 目标"做 step 级信用分配的有效性。但提升幅度属稳健而非数量级，且评测局限于自家 benchmark。

#### 10. 逻辑自洽性(中性评估)
设计自洽：非对称信息→直接学 advantage（避 value head 泛化问题）→BT 目标对齐 LLM→当 reward model 优化 policy。代码框架（OpenRLHF fork 的 multi-turn DPO）与方法的偏好学习目标一致。最大假设依赖在于"训练期可得 reference solution"——这是方法成立的前提，但也是其外推到真实场景的主要限制。

#### 11. 残留问题 / 局限
- 增量是"非对称 critic + 直接学 advantage + BT 目标"的组合，依赖训练期可得 reference solution 这一较强假设（真实场景未必有）。
- ColBench 的"人类"用 LLM simulator 近似，且 simulator 依赖 reference artifact，模拟忠实度受 simulator 能力上限约束。
- 提升幅度（+6% 绝对）稳健但非数量级；仅在自建 benchmark 验证，跨 benchmark 泛化未证。
- 与 TSRD 是思路同构（特权信息做信用分配）而非直接可用方案；最大可复用资产是开源 benchmark + 数据。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- https://github.com/facebookresearch/sweet_rl （ColBench + SWEET-RL 官方实现）；数据集 https://huggingface.co/datasets/facebook/collaborative_agent_bench 。
- 框架：基于 **OpenRLHF 的定制 fork**（`YifeiZhou02/collab_openrlhf`），改造以支持 multi-turn DPO 与 length normalization（README 训练用 `openrlhf.cli.train_dpo`）。Frontend Design 评测需额外装 GeckoDriver + Firefox（渲染 HTML）。
- 流程：生成任务与离线轨迹 → 构 chosen/rejected 偏好对 → BT 目标训 turn-wise advantage（critic 用训练期信息 c）→ 用 advantage 当 per-turn reward 优化 policy。human simulator 用 Llama-3.1-70B-Instruct。


---


### tcod — TCOD: Exploring Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents

> **一句话重点 (TL;DR)**：在多轮 agent 场景下 vanilla OPD 会因跨轮误差累积出现"轨迹级 KL 不稳定"（KL 飙升、成功率坍塌），TCOD 用一条时间课程逐步扩大暴露给学生的轨迹深度（F2B 浅到深 / B2F 由 teacher 前缀导航后到前），在保留 OPD 稠密信号的同时稳住训练并最高 +18 分。

**元信息**：arXiv 2604.24005（v3, 2026-04-29，cs.LG）｜ Tongyi Lab, Alibaba Group + 香港中文大学（CUHK）｜ 预印本 2026-04 ｜ 主题 T1(On-Policy Distillation)+T2(多轮自主 agent)，与 mtp_opd 高度相关 ｜ 代码 https://github.com/kokolerk/TCOD （已 clone ~84MB，可运行配置齐全）｜ 框架 Trinity-RFT（Ray-based RFT，Alibaba）

#### 1. 相关工作与进展
- **多轮 LLM agent**：ReAct（reason+act 交错）是主流范式，应用于具身规划、网页导航等交互环境；OpenClaw 等系统推动长程 agent。
- **On-Policy Distillation 及其改进**：OPD（Agarwal 2024 GKD；Lu & Thinking Machines Lab 2025）用稠密蒸馏信号替代稀疏标量奖励、提升样本效率；既有改进集中在目标设计（forward/backward KL 平衡，Jang 2026、Jin 2026 即 entropy-aware OPD）、优化启发（reward clipping，Ko 2026）、监督来源（context distillation、self-distillation）——但均面向**单轮静态**任务。
- **课程学习**（Bengio 2009）：近期被用于 LLM 预训练/后训练与 RL（GRPO），但多依赖**外部模型评估难度**；Lauffer 2025 只用 expert 后续纠错动作训练学生，破坏了 on-policy 设定。TCOD 以"递增轨迹深度"定义难度，只用学生自生成数据，避开外部难度评估器。

#### 2. 现有工作存在的问题
作者在 ALFWorld 上实证出 **Trajectory-Level KL Instability（轨迹级 KL 不稳定）**：
- 观察1：训练中 KL 持续**升高**且成功率**坍塌至近零**（与单轮设置 KL 收敛递减相反；Fig.2a/2b，小模型 Qwen3-0.6/1.7B、Qwen2.5-0.5/1.5B 上显著）。
- 观察2：即使最终收敛，初始 KL 极高（~1000 量级 vs 收敛 ~60，Fig.2c）。
- 机理：**跨轮误差累积（compounding error）**——学生动作/观测追加到历史 ht，因果耦合使 per-turn KL 随轮次单调增大（Fig.2d），学生被推出 teacher 有效支撑（support），teacher 对学生 token 赋低概率、监督信号失真。Remark 1 区分：long-CoT 只增长同一环境状态下的响应长度，而多轮 agent 每轮更新环境状态，才真正放大累积误差。

#### 3. Motivation
如何在保留 OPD 稠密信号优势的同时，避免长 horizon 累积误差导致的失稳？借鉴课程学习（先易后难）：**控制暴露给学生的轨迹深度**，让学生始终停留在 teacher 有效指导范围内，从短稳定前缀渐进扩展到完整多轮 rollout。

#### 4. 主要灵感 / 核心直觉
误差累积是长 horizon 的固有属性；与其一开始就让学生端到端跑完整轨迹（必然进入 teacher 支撑外），不如沿时间维度做课程——要么从轨迹早段学起逐步加深（F2B），要么让 teacher 把环境导航到接近终止状态、学生从"成功门口"接管再逐步前移（B2F）。难度 = 轨迹深度 k，可由训练步线性 pacing 自动调度，无需外部难度评估器。

#### 5. 主要解决思路(一段话讲清核心)
TCOD 用线性 pacing `k = k_start + ⌊n/η⌋`（n 当前步、N 总步、η 控制增长率）沿训练步控制轨迹深度 k，并给出两个仅需极小代码改动的变体：**F2B**（学生 rollout 至多 k 步，k 从小到大，目标只对前 k 步算 token-level KL）；**B2F**（teacher 用预采集成功轨迹 τ\* 执行前 L−k 步把环境导到中间状态、stop-gradient，学生从该状态接管后 k 步并对其算 KL，k 递增直到学生端到端）。学生执行步计算 KL 梯度，teacher 执行步停梯度。

#### 6. 方法详解(通俗、分步骤)
- **前置定义**：状态 ht=(o0,a0,…,o_{t-1},a_{t-1},ot) 为完整交互历史；OPD 目标 L_OPD=E_{τ~πθ} Σ_t D_KL(πϕ(at|ht)‖πθ(at|ht))（reverse-KL，teacher πϕ、student πθ）。
- **TCOD-F2B（Algorithm 1）**：每训练步 k←min(k_start+⌊n/η⌋, T_max)；学生从 h0 起 rollout k 步；L=Σ_{t=0}^{k-1} D_KL(πϕ‖πθ)；更新 θ。先学早轮信号再端到端，防 horizon 诱发的 KL 坍塌；无需任何示范。
- **TCOD-B2F（Algorithm 2）**：预采集 teacher 成功轨迹 T\*（pass@10 采样，保留成功）；每步 teacher 执行 τ\* 的前 L−k 步（stop-gradient）把环境带到中间状态，学生从 t=L−k 接管到 L，仅对学生步算 KL；k 递增直到学生从初始状态端到端。论文用 Appendix D.5 验证：随训练步 teacher 前缀从 L−1 降到 0，测试集端到端成功率稳步上升，证明课程平滑过渡避免了 train-test 分布漂移。
- **异步训练稳定性细节**：异步 rollout（4×H20 actor）/训练（2×H20 learner）/teacher（2×H20），lock-free ring buffer；**staleness-aware 子轨迹经验回放**——把长度 n 的轨迹拆成 n 个递归前缀子轨迹入 buffer，交互历史封装进 prompt 作结构化上下文；用 staleness filter 丢弃 n_current−n_old>Δ_max 的经验，经验上 Δ_max=2 最优。

#### 7. 实验数据集
三个多轮 agent benchmark（Table 1）：**ALFWorld**（具身，max 30 turns，含 seen/unseen/hard split）、**WebShop**（电商，max 15 turns，需 ~1TB 内存）、**ScienceWorld**（科学推理，max 30 turns，需 Java/jar）。Hard split = 121 个 teacher 在 train split 上 pass@10 仍失败的任务。师生对：ALFWorld 主实验 student=Qwen2.5-{3,7}B、teacher=GRPO 训练的 Qwen2.5-7B（域内 SR 85.71）；跨 benchmark student=Qwen3-{1.7,4}B、teacher=Qwen3-30B-A3B-Instruct（通用、域内偏弱）。8× H20(96GB)。

#### 8. 实验结果与主要发现
- **Q1 缓解 KL 升高 + 提升性能（Table 2，Qwen2.5-7B teacher）**：Qwen2.5-3B 上 F2B 在 Valid Unseen 较 OPD **+18.74 SR**、Seen +15.71；7B 上 B2F Seen +11.06、Hard +7.44。平均减少约 2.97 个动作轮次；KL 曲线更平稳、advantage 收敛更快（Fig.4/5）。
- 跨 benchmark（Table 3，Qwen3-30B teacher）：对 Qwen3-1.7B，vanilla OPD 几乎全崩（avg 0.17），TCOD 把三 benchmark 平均拉到 ~18.6（**+18.4~18.7**，从近零恢复）；对 Qwen3-4B 提升温和（avg +0.3~+1.9，与 OPD 相当）。
- **Q2 超越 teacher 能力边界**：Unseen 上较 teacher 最多 +2.5；**Hard split 上 teacher SR 仅 6.61，B2F 反超最高 +14 分**（B2F 7B 达 20.66）——非单纯模仿。
- **Q3 鲁棒性与效率**：η∈{2,4,6} 下性能波动 <2%；总训练时间较 vanilla OPD 减少近 32%（F2B 比 B2F 更省，因 B2F 中间状态接管后仍多探索几步）。teacher 质量比规模更决定上界：域内强的 7B-RL teacher 下 B2F 可略超 teacher（+0.7），而通用 30B teacher 下 OPD/TCOD 均差 teacher ~2 分。

#### 9. 结果如何支撑其主张
- "失效模式真实存在"：Fig.2（KL 升高 + 成功率坍塌 + per-turn KL 随轮次增）直接对应"轨迹级 KL 不稳定 + 误差累积"机理。
- "课程能稳训练"：Fig.4b KL 曲线 TCOD 明显更平、Fig.5 response length/pg_loss 平滑，支撑"控制轨迹深度即可稳住"。
- "超越 teacher"：Hard split（teacher pass@10 失败任务）上的正增益是较强证据，说明学生学到的是更鲁棒策略而非复制 teacher。
- 效率主张由 Fig.6 训练时间柱状图（OPD vs F2B vs B2F）支撑。

#### 10. 逻辑自洽性(中性评估)
整体自洽：失效现象→机理（误差累积）→对策（控制轨迹深度的时间课程）→两变体（F2B 无需示范 / B2F 需 teacher 成功轨迹）逻辑闭环，且课程末端回到端到端、消解了 B2F 的 train-test mismatch（有 Appendix D.5 佐证）。一个张力点：观察机理中作者自己承认"KL 升高既可能是学生模仿不能、也可能是学生进入 OOD 致 teacher 不确定"二者难以区分，但两种解释指向同一对策，不影响方法有效性。Qwen3-4B 上 TCOD 仅与 OPD 相当（非显著提升），说明收益高度依赖"学生本会崩 + teacher 域内够强"的条件。

#### 11. 残留问题 / 局限
- B2F 依赖预采集 teacher 成功轨迹，有额外采样开销（F2B 是无示范替代）。
- 固定课程 pacing 虽鲁棒，但最优节奏可能随环境/师生对变化；作者建议未来用 KL 的 EMA 做自适应 horizon，尚未实现。
- 仅在三个**文本**多轮 benchmark 验证，未覆盖多模态/物理具身。
- 收益条件性强：teacher 域内性能（而非规模）是上界主因；通用强 teacher 下 TCOD 无法超越 teacher。
- 〔待核〕各师生对完整数值、k_start/η 取值与超参表见论文 Appendix C/D（正文已给 k_start=1、η=2 默认，η∈{2,4,6} 消融）。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库 https://github.com/kokolerk/TCOD （已 clone，构建于 Trinity-RFT）。README 明确"What Is Implemented"：三环境各一套 multi-turn OPD + TCOD-f2b + TCOD-b2f workflow（`trinity/common/workflows/envs/TCOD/{alfworld,webshop,scienceworld}`）、基于师生 token-level logprob 的蒸馏信号、可运行示例配置 `TCOD_examples/<env>/{opd,tcod_b2f,tcod_f2b}.yaml`。
- 模型：ModelScope `wjqkoko/TCOD`；HuggingFace `kolerk/tcod`。
- 关键配置：`algorithm.advantage_fn: multi_turn_opd`；`rollout_args.logprobs` 必须开启以算蒸馏 gap；TCOD 配置用 `workflow_args.checkpoint_strategy: linear`（即线性 pacing）。
- 框架：Trinity-RFT（`trinity run --config <yaml>`，Ray 驱动），依赖 vLLM 类推理 + flash-attn 2.8.1；环境 ALFWorld（pip）/ WebShop（princeton-nlp/webshop，需 Java17+、~1TB 内存）/ ScienceWorld（allenai，jar）。代码可得性高、与论文方法对应清晰。


---


### webagent_r1 — WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning

> **一句话重点 (TL;DR)**：用端到端多轮 on-policy RL（M-GRPO + 动态上下文压缩 + 并行轨迹 rollout），仅靠二值任务成功奖励，把 web agent 从 prompting/BC 水平大幅提升；并系统说明 BC warm-up 不可或缺、long-CoT 帮 SFT 但限制 RL 探索、以及"增加交互轮数"这一新的 test-time scaling。

**元信息**：arXiv 2505.16421v2（cs.CL，2025-10-08）｜ UVA（Zhepei Wei，实习于 Amazon）+ Amazon + Georgia Tech ｜ 预印本 ｜ 主题 T2 web agent / 多轮 RL，与 mtp_opd 关系：纯 on-policy 多轮 RL（**非蒸馏、无 teacher 监督**），可作"无 teacher 的多轮 agent RL"对照；warm-up(BC)+端到端 on-policy 与 path-recovery/初始化策略相关 ｜ 代码 https://github.com/weizhepei/WebAgent-R1 （已 clone，约 150MB）｜ 框架 **verifiers**（willccbb/verifiers，GRPO-based）

#### 1. 相关工作与进展
RL 显著提升单轮 LLM（DeepSeek-R1，数学）。早期 web agent 靠 prompting 或行为克隆(BC/SFT)。近期 RL 训 web agent 多为离线/迭代 off-policy（Filtered BC、AWR、DigiRL、WebRL），其中 WebRL 还需额外训 outcome reward model 标 GPT-4 生成的新数据。并发工作 RAGEN、SkyRL 把端到端 on-policy RL 用于模拟游戏/编码，但真实 web 环境仍欠探索。GUI agent 线（含截图多模态）与本文（纯 HTML 文本）正交。

#### 2. 现有工作存在的问题
- prompting/BC 缺探索与试错能力，泛化差。
- 离线/off-policy RL 打断 agent 与环境的端到端交互，引入轨迹过滤、outcome reward model 训练、迭代优化等额外复杂度；off-policy 数据来自旧版 agent，在任务相互依赖的动态 web 中学不到关键行为（论文举例：先登出再编辑 profile——登出后失去访问权，旧数据无登出行为则生成无效动作）。
- 每步 HTML observation 可达数千 token，长 horizon 累积上下文带来巨大显存开销甚至 OOM，使多轮 RL 训练不可行。
- 现有 RL 实现（单轮 GRPO）不适配多轮场景。

#### 3. Motivation
用端到端 on-policy 多轮 RL 直接从在线交互学习：训练对齐 agent 最新行为、更稳定（Schulman），免 replay buffer/过滤开销，让 agent 据自身过去决策自适应——在"早期决策显著影响后续步"的交互环境中是关键优势。

#### 4. 主要灵感 / 核心直觉
把 GRPO 的 group-relative 优势思想搬到多轮：一个任务采 G 条完整轨迹、按 group 内 reward 归一化算优势，逐 token 做 PPO-clip 更新（M-GRPO）。再用工程手段（动态压缩 + 并行 rollout）解决多轮 RL 在 web 上的显存与采样瓶颈。直觉上，"thinking 格式"促使 agent 进行更多轮交互而非更长单轮输出，揭示了 web 任务的 test-time scaling 走"交互深度"而非"单步长度"。

#### 5. 主要解决思路(一段话讲清核心)
先用专家演示做 BC warm-up 初始化（acquire 基本 web 操作技能），再在 WebArena 环境做端到端多轮 on-policy RL：用规则化二值 outcome reward 引导，M-GRPO 逐 token 优化（带 group-relative 优势 + KL 正则），并行 rollout G 个独立浏览器实例生成多样轨迹，动态上下文压缩把历史 observation 替换成"Simplified HTML"占位符（保留完整动作历史、相应更新 loss mask），交互直到达最大步数或 agent 输出 exit()。

#### 6. 方法详解(通俗、分步骤)
1. **POMDP 形式化**：状态=当前页 HTML 文本，动作∈预定义动作空间（Click/Type/Select/Scroll/Search/Exit…），终态二值奖励 r∈{0,1}。
2. **BC warm-up**：在固定专家演示 D={(h_t,a_t)}（h_t=完整交互历史）上 SFT，L_BC=−E log πθ(a_t|h_t)。用公开 9,460 条轨迹。
3. **动态上下文压缩**：新 observation 到来时把旧 s_i 替换成短模板 s'_i（"Simplified HTML"），保留完整动作历史，防 OOM；同步更新 loss mask 使损失只算在动作 token 上。
4. **M-GRPO**：每任务采 {τ1..τG}，逐 token 优势 Ã_{i,j,t}=PPO-clip(重要性比·A_{i,j})，组相对优势 A_{i,j}=(r_i−mean(r))/std(r)，附 βKL 项。
5. **并行轨迹 rollout**：G 个独立浏览器实例（各自 cookie/上下文），同起始页、独立交互 → 多样历史。
6. **奖励**：直接用环境默认规则化二值奖励（String/URL Match、程序执行），**无 outcome reward model**。
7. **三变体研究初始化策略**：R1（标准 BC→RL）、R1-Zero（无 SFT 直接起 RL）、R1-CoT（用 long-CoT BC 数据 SFT 初始化）。

#### 7. 实验数据集
- 训练/评测：**WebArena-Lite**（WebArena 的 human-verified 自托管沙箱版，跨 Reddit/GitLab/CMS/Map/Shopping 五站；非线上真实站点）。BC 用公开 **9,460** 条轨迹；RL 用 **647** 个任务训练、**165** 个 verified 任务评测；成功率由内置规则 rubric 判定。
- OOD 评测：**WebVoyager**（5 个域、每域随机 25 任务，域在 WebArena 中未见）。
- 模型：Qwen2.5-3B、Llama3.1-8B（finetune backbone）；对照含 GPT-4o/o3/o4-mini、QwQ-32B 等。

#### 8. 实验结果与主要发现
- **主结果（Table 2，WebArena-Lite 平均 SR%）**：Qwen2.5-3B 6.1→33.9；Llama3.1-8B 8.5→44.8。均超同尺寸 SFT/Filtered BC/AWR/DigiRL/WebRL（8B 上 WebRL 42.4 < 本法 44.8）。
- **与专有模型对照**：OpenAI o3=39.4、o4-mini=36.9。**8B 版(44.8) 超 o3；但 3B 版(33.9) 仍低于 o3**——"超 o3"主要由 8B 支撑，非全尺寸成立。逐站表现不均（如 Map 上本法 23.1 弱于 WebRL 36.7）。
- **BC 必要性（Fig.4）**：R1-Zero（无 SFT）初始 6.1%，RL 后**反而略降**（动作残缺/格式错、罕得正奖、探索失败）。
- **long-CoT（Fig.4）**：long-CoT SFT 初值 24.5% > 标准 BC 20%，但 RL 增益更小（R1-CoT 24.5→30.3 vs R1 20→33.9）；作者假设确定性 CoT 模式约束了 RL 探索空间。
- **thinking prompting（Table 3）**：显著提升 SR（o4-mini 15.9→36.9）；单轮长度几乎不变（Qwen 139→142）但交互轮数大增（6→17），指向"交互深度"型 test-time scaling。
- **test-time scaling（Fig.5）**：放宽最大交互轮数，prompting/SFT/RL 三类成功率均单调提升。
- **OOD（Table 4，WebVoyager）**：WebAgent-R1 平均 32% > SFT 12% > prompting 8.8%，各域全面领先。
- **训练动态（Fig.3）**：reward/轨迹长/交互数呈三阶段（初始技能获取→探索精炼→策略稳定），3B/8B 模式相似。

#### 9. 结果如何支撑其主张
- "端到端 on-policy 多轮 RL 有效"：Table 2 两 backbone 的大幅提升 + 超 off-policy 基线支撑。
- "BC warm-up 关键"：R1-Zero RL 后退化是强反例支撑。
- "test-time scaling 走交互深度"：Table 3（长度不变、交互↑）+ Fig.5（轮数↑→SR↑）双重支撑，较有说服力。
- "泛化"：Table 4 OOD 领先支撑（但每域仅 25 任务，样本小）。
- "超 SOTA / 超 o3"：成立但需限定——8B 上超 o3、3B 不超；且基线取"复现与文献中较高者"，比较口径需注意。

#### 10. 逻辑自洽性(中性评估)
- 论证整体自洽：off-policy 痛点（登出例子）→ on-policy 设计动机 → M-GRPO/工程 → 结果，链条连贯。
- 几处 claim 偏 hypothesize：long-CoT 限制探索、R1-Zero 失败归因，均为假设性解释而非受控证据。
- "超 o3"的标题级主张需谨慎（仅 8B、且 o3 为零样本 prompting 对照，非微调，比较不完全对等）。
- M-GRPO 相对单轮 GRPO 的"多轮"实质主要是把整条多轮轨迹的 token 一并纳入优化 + 二值终态奖励，无中间步奖励塑形（作者列为 future work）。

#### 11. 残留问题 / 局限
- 仅文本 HTML 输入，无截图/多模态（作者承认）。
- 依赖规则化可验证 outcome reward，开放式任务（如旅行规划）不适用。
- 固定预定义动作空间，遇需新操作的交互元素受限。
- 仅 WebArena-Lite（沙箱、5 站）；OOD 评测样本小（每域 25）。
- 多处归因为假设，未做消融验证；安全风险（CMS 误删等）需权限/确认机制。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库：https://github.com/weizhepei/WebAgent-R1 （已 clone，约 150MB；含 `WebAgent-R1/{Train, Eval, WebArena-Env-Setup}`）。
- **框架：verifiers**（willccbb/verifiers，"RL with LLMs in Verifiable Environments"，GRPO-based，Accelerate + DeepSpeed ZeRO3）；`Train/` 下实证含 `verifiers/`、`VisualAgentBench`，在其上扩展为 M-GRPO + 并行 rollout。评测/环境为 WebArena（`Eval/browser_env`，WebArena-Lite）。
- 〔已核：原始 shortlist 标 framework=unknown 不准——据仓库实证为 **verifiers（非 veRL）**。〕
- 代码可得性：完整开源（含训练/评测/环境搭建）。


---

## 4. T3 — GFT 统一 SFT-RL & GRPO-RLVR（50 篇）

本部分逐篇内联两类工作：(a) **GFT 类 / SFT-RL 统一**——把 SFT 重构为 RL 的辅助目标或同一策略梯度的实例（chord/hpt_upge/luffy/prefix_rft/srft/psft/asft/dft_reweight/superrl/raft_reinforce_rej/on_policy_sft/trapo/amft/eaft/sed_sft/rl_plus/fest/cbrl/scaf_grpo/behavior_priming/denoiserl/cepo/ampo）；(b) **GRPO/RLVR 算法谱系与训练机理**——GRPO 起源与系统（deepseekmath/dapo/dr_grpo/orz/simplerl_zoo/skywork_or1）、稳定性与聚合算子（gspo/gmpo/holderpo/hapo/bapo/lp_reg/lite_ppo）、熵机制（entropy_mechanism/revisit_entropy）、能力边界与奖励机理批判（limit_rlvr/spurious_rewards/negative_reinforce/nft/one_shot_rlvr/prorl）、大模型推理 RL 技报（deepseek_r1/magistral/minimax_m1）、综述与基础设施（rl_survey_lrm/ca_survey/trl_v1）。OPD 在理论上即 dense KL-约束 RL，本线提供其算法语境。每篇为完整 12 节，标题已降级。

### amft — AMFT: Aligning LLM Reasoners by Meta-Learning the Optimal Imitation-Exploration Balance

> **一句话重点 (TL;DR)**：与其用"SFT→RL"两阶段或靠启发式硬切换，AMFT 把 SFT 和 RL 揉进**单阶段单循环**，并把"模仿(SFT)与探索(RL)的配比 µ"当成一个**可学习参数**，用 meta-gradient 以"最大化最终任务表现"为元目标前瞻式地学这个 µ。值得看的点：它把 SFT 形式化为"优化专家示范里隐含的隐式 reward"的特殊 RL，从而让 SFT/RL 在同一目标下统一——这与 OPD"在线策略蒸馏里如何调度模仿 vs 探索权重"的命题同构。**注意：代码仓库当前 404 不可得，方法细节无法独立核实。**

**元信息**：arXiv 2508.06944v1 ｜ 清华大学电子工程系（Lixuan He, Jie Feng, Yong Li）｜ Preprint, under review, 2025-08-09 ｜ 主题 单阶段统一 SFT+RL 后训练 / 与 OPD **真实相关** ｜ 代码 https://github.com/hlxtsyj/AMFT（**不可得**：该仓库与 TSYJ-He/AMFT 均 404，已二次核验仍 "Repository not found"）｜ 框架 〔待核，仓库不可得〕方法层面建立在 GRPO(RLVR)+SFT 加权损失之上

#### 1. 相关工作与进展(这条线现在做到哪一步)
LLM 推理后训练主流是 **"SFT→RL"两阶段流水线**。近期出现一批**单阶段统一 SFT+RL** 的尝试：SRFT 用 policy entropy 调权；SuperRL/SASR/DyME 分别用 reward density、gradient norm、生成正确性做"硬切换"在 SFT 与 RL 间跳转。AMFT 处在"单阶段统一"这条线，但主张前面这些都是**反应式(reactive)** 启发式，缺一个原则化、前瞻性的配比机制。

#### 2. 现有工作存在的问题(本文针对的痛点)
- **两阶段流水线**：SFT 擅长模仿专家轨迹但局限于静态数据集、倾向记忆而非泛化(OOD 差)；RL 探索强、泛化好但样本低效、稀疏奖励下不稳，且 on-policy 受基模能力上界限制。两阶段目标突变易导致**灾难性遗忘**(RL 阶段覆盖 SFT 学到的结构知识)。
- **现有单阶段方法**都是反应式：依赖短期、含噪的启发式信号(熵、reward density、梯度范数等)做被动调整或硬切换，**缺乏前瞻性**——不回答"现在这样调配，对最终性能是否最优"。

#### 3. Motivation(为什么做这件事)
作者要的是一个**原则化、动态、前瞻**的模仿-探索平衡机制：模仿/探索的最优配比不该靠启发式被动反应，而应作为**可学习参数**，用"最大化最终任务表现"的元目标直接优化(forward-looking)。

#### 4. 主要灵感 / 核心直觉
核心灵感是**"隐式奖励"统一视角**：SFT 不只是分布匹配，可形式化为"优化专家示范中隐含的隐式 reward"的一种特殊 RL。于是 SFT(path-based 隐式 reward)与 RL(outcome-based 显式 reward)是**同一目标下互补的两种 reward 信号**，二者的配比 µ 就成了一个可被元学习优化的量。

#### 5. 主要解决思路(一段话把核心机制讲清)
用一个动态权重 µ_t∈[0,1] 把 SFT 损失与 RL 损失线性组合成统一损失；µ 由一个 **meta-gradient 自适应控制器**驱动：长期看，把 µ 当可学习参数，在 validation batch 上估"调 µ 对未来最终性能的影响"做前瞻式全局控制；短期看，用 policy entropy 偏离目标值的程度做快速纠偏。每个 update step 先更新 µ，再用新 µ 算统一损失更新策略。

#### 6. 方法详解(通俗、分步骤;关键公式用白话解释,必要时给伪代码)
**统一损失**：`L_total = µ_t · L_SFT(path-based 隐式奖励) + (1−µ_t) · L_RL(outcome-based 显式奖励)`，RL 部分用 GRPO(RLVR)。

**µ 控制器(双机制)**：
- **长期 meta-gradient**：把 µ 当可学习参数，周期性在 validation batch 上估 `g_µ = ∇_µ U(θ_t)`（U 为长期效用/任务表现）。白话："现在把 µ 往哪个方向挪，能让若干步之后的最终性能更好"——这是前瞻式全局控制。
- **短期 entropy 启发式**：`g_H = H* − H(π_θ)`。熵过高(太发散)→增大 µ 收敛到模仿；熵过低/坍塌→减小 µ 鼓励探索。目标熵 H* 由 warm-up 阶段的平均熵初始化。
- **更新**：`µ_{t+1} = µ_t + η_µ·g_µ + η_H·g_H`。

**单步流程伪代码**：
```
每个 update step:
  1. 取 SFT 数据 + on-policy rollout 的混合 batch
  2. 控制器更新 µ_t  (meta-gradient g_µ + 熵启发式 g_H)
  3. 用新 µ_t 算 L_total = µ·L_SFT + (1−µ)·L_RL(GRPO)
  4. 更新策略 θ
```
〔待核〕η_µ/η_H 取值、meta-gradient 用的 validation batch 如何构造、稳定性细节——因仓库 404 无法核实。

#### 7. 实验数据集
- **数学**：训练集 **OpenR1-Math-46k-8192**（论文明确命名，HF: Elliott/Openr1-Math-46k-8192）。ID 评测 **5 个** benchmark：AIME24、AMC、MATH500、Minerva、**OlympiadBench**；OOD 泛化用 **3 个**通用推理 benchmark：ARC-C、GPQA-D、MMLU-Pro。基模 **Qwen2.5-Math-7B**。
- **多模态**：**General Points**(算术推理纸牌)、**V-IRL**(视觉-语言导航)，各设 Rule-Variation 与 Visual-Variation 两类 ID/OOD split。基模 **LLaMA-3.2-Vision-11B**。
- 效率目标指标示例：General Points 60% win-rate、V-IRL 70% success-rate(作为"达标所需步数/样本数"的衡量点)。

#### 8. 实验结果与主要发现(关键数字)
- 作者自述在数学、抽象视觉推理(General Points)、视觉导航(V-IRL)三类任务上"建立 new SOTA"，并强调 **OOD 泛化更好**、达标所需训练步数/样本更少(计算与样本效率)。
- **消融(Table 4)** 确认三件套缺一不可：去掉 meta-gradient、去掉熵启发式、去掉 SFT warm-up 均显著掉点；去掉熵启发式还会**训练不稳**。
- 对比基线含 RL-from-SFT(两阶段)、RL-only(GRPO from scratch)、LUFFY、SRFT 等。
- 〔保留意见〕"new SOTA"为 preprint 自述、未经评审；精确数值因仓库 404 与本审计未逐一抄录主表，待核。

#### 9. 结果如何支撑其主张(证据链是否到位)
主张是"前瞻式可学习配比 > 反应式启发式"。消融把三个组件逐一拿掉证明各自必要性(尤其熵启发式关乎稳定、meta-gradient 关乎性能)，这条证据链对"三件套都有用"较扎实。但"前瞻式优于反应式"这一**因果归因**主要靠与 SRFT 等基线的端到端对比间接支撑，缺少把 meta-gradient 直接对位某个反应式信号的 head-to-head 控制实验，归因强度中等。

#### 10. 逻辑自洽性(中性评估:哪里站得住、哪里牵强)
- **站得住**：把 SFT 视为隐式 reward RL 的统一视角是合理且有文献基础的；单循环混合 batch + 动态 µ 的工程设计自洽；消融支持各组件必要性。
- **牵强/可质疑**：(1) meta-gradient 估 `∇_µ U` 通常需要展开多步或近似(隐式微分/有限差分)，论文层面对其方差与计算开销的讨论待核，仓库不可得使其无法验证；(2) "long-term meta-gradient + short-term entropy"两个信号可能冲突，加权 η_µ/η_H 的鲁棒性未知；(3) new SOTA 系自述、未评审。

#### 11. 残留问题 / 局限
- **代码 404 不可得**——本分析仅基于 34 页 PDF，方法实现、超参、稳定性均无法独立核实，这是最大局限。
- preprint 未经评审，"new SOTA"待第三方复现。
- meta-gradient 的计算成本/方差、双学习率敏感性、validation batch 构造细节缺失。
- 多模态任务仅两个(General Points / V-IRL)，泛化结论的覆盖面有限。

#### 12. 开源代码与框架(链接 + 框架 + 代码可得性)
- 链接：论文声明 https://github.com/hlxtsyj/AMFT （正文首页脚注）。
- **可得性：404 不可得**。`hlxtsyj/AMFT` 与任务清单给出的 `TSYJ-He/AMFT` 经二次 `git ls-remote` 核验均返回 "Repository not found"，api.github.com 同样 404。仓库当前不公开/已下线。
- 框架：〔待核〕无法确认具体 RL 框架(verl/TRL/oat 等)；方法层面建立在 GRPO(RLVR)+SFT 加权损失之上，视觉任务用 GRPO 作 RL 基线。


---


### ampo — AMPO: Adaptive Multi-Guidance Policy Optimization for Diverse Exploration

> **一句话重点 (TL;DR)**：当 on-policy RLVR 在某道难题上**整组采样全失败**(稀疏奖励、学不动)时，AMPO 才**按需**从一个**多教师池**里挑入正确解替换失败样本；挑哪条教师路径不看"哪个教师最强"，而看"哪条对学生最容易吸收"(学生在该路径下生成正确答案的概率最高)。值得看的点：用 4 个同量级"同伴"教师 + 仅 8.5k 数据，就媲美了用单一更强教师(DeepSeek-R1)+46k 数据的方法，并全程保持更高 entropy(探索性)。与 TSRD 的"按需脚手架 + 路径恢复"强相关。

**元信息**：arXiv 2510.02227v2 ｜ 同济、香港理工、上海 AI Lab、新加坡国立、电子科技大（Xiaoyang Yuan 等，通讯 Yi Bin）｜ 2025-10-09 (v2) ｜ 主题 多教师 Mixed-Policy RLVR / 与 TSRD **强相关** ｜ 代码 https://github.com/SII-Enigma/AMPO（**完整**，已克隆约 163M，含数据/脚本/checkpoint）｜ 框架 verl(GRPO 的 Mixed-Policy 扩展, FSDP 或 Megatron)
> 标题两版同指一文(arXiv id 一致)：arXiv 版 "More Than One Teacher: Adaptive Multi-Guidance Policy Optimization for Diverse Exploration"；README 版 "Many Small Teachers Beat One Giant: Adaptive Multi-Peer Policy Optimization for LLM Reasoning"。

#### 1. 相关工作与进展(这条线现在做到哪一步)
RLVR(可验证奖励 RL)是提升 LLM LongCoT 推理的有效范式，但 on-policy GRPO 的探索被困在模型自身知识边界内，难学到超出初始能力的新策略，并受 "capacity-difficulty mismatch" 困扰(难题持续失败→稀疏奖励→训练不稳)。为突破边界，社区做 **Mixed-Policy RL**：把强教师的 off-policy 轨迹混入 on-policy，或交织 RL 与 SFT(代表如 LUFFY)。AMPO 处在 Mixed-Policy 这条线，主打"多教师 + 按需 + 可理解度选择"。

#### 2. 现有工作存在的问题(本文针对的痛点)
现有 Mixed-Policy RL 有两个关键局限：
1. **多依赖单一教师**——限制学习多样性，且把单个教师的固有偏置/风格带进学生。
2. **静态数据整合**——不看模型当前需求与"可理解度"，无差别注入外部解，可能注入学生根本吸收不了的解法，浪费且扰动训练。

#### 3. Motivation(为什么做这件事)
借鉴知识蒸馏的**多教师**思想：用多个**同量级"同伴"教师**的集体智慧替代单一更强教师。两条原则：(1) **guidance-on-demand**——只在学生自己解不出时才用外部引导替换失败样本，最大化自探索价值；(2) **按可理解度选路径**——挑学生最容易吸收的那条教师推理，平衡探索与利用。

#### 4. 主要灵感 / 核心直觉
- **"多个同伴 > 一个巨人"**：多个 7B-级教师风格各异，集体覆盖的解法多样性比单一强教师更广，且不把单一模型偏置灌给学生。
- **"易吸收 > 最正确"**：一条教师推理即使正确，若离学生当前内部表征太远，学生也学不动；应选学生"差一点就能自己走到"的那条——这正好对应"脚手架"该搭在学生够得着的高度。

#### 5. 主要解决思路(一段话把核心机制讲清)
在 GRPO 之上：对每个 query 先由 π_old 采 G 个解；若整组奖励都低于阈值(稀疏奖励、全失败)，就触发**自适应多引导替换**——从多教师正确解池里按"可理解度(Probability Reward)"选 top-k 条，替换掉 k 个错误的 on-policy 解；否则不替换，优先自探索。然后在增广 batch 上统一算 advantage，用**混合目标**(off-policy 序列级聚合 + on-policy token 级聚合)更新策略。N_off=0 时无缝退化为 GRPO。

#### 6. 方法详解(通俗、分步骤;关键公式用白话解释,必要时给伪代码)
三部分：

**(1) Adaptive Multi-Guidance Replacement(自适应多引导替换)**：构建 Multi-Guidance Pool `P_G`(多教师的正确 off-policy 解)。π_old 对 query 采 G 个解；若全部奖励 < 阈值 τ(置替换标志 I=True)，随机选 k 个错误 on-policy 解，用 P_G 中按可理解度选出的 top-k off-policy 解替换，`k=min(k_0, N_g)`；否则不替换。保证每步都有正确解可学，但优先自探索。

**(2) Comprehension-based Guidance Selection(基于可理解度的选择)**：定义 **Probability Reward r_p**：把教师推理路径 z_off 与 ground-truth 答案 y* 拼成 o*=(z_off, y*)，r_p = 学生 π_θ 在给定 z_off 下生成正确答案 token 的**几何平均概率**(log-prob 平均后 exp，clip 到 [0,1])。白话：r_p 高 = "顺着这条教师推理，学生自己几乎就能写出正确答案" = 最易吸收。按 r_p 取 top-k，平局取**更短(更简洁)** 路径。

**(3) Policy Optimization with Multi-Guidance**：在增广 batch G_aug 上统一归一化算 advantage(GRPO 式)。`J_Mixed = off-policy 目标 + on-policy 目标` 加权和：
- **off-policy 用 sequence-level 聚合**——避免长教师序列主导梯度，保证每条教师解等权(`compute_token_on_seq_off_policy_loss`：对每条教师序列先 masked_mean，再跨序列 `.mean()`)。
- **on-policy 用 token-level 聚合**(DAPO 式)。

**off-policy 重要性采样比的两条实现路径**(已核 `mix_core_alg.py`)：
- **路径(a) 论文正文形式**：传入 `target_probs` 时，`off_ratio = π_θ / π_target`（教师概率显式出现，对应论文 Eq.7 `r̂=π_θ/π_φj`）。
- **路径(b) 默认发布脚本**：`train_ampo.sh` 用 `off_policy_reshape="p_div_p_0.1"` 走 reshape 分支，`off_ratio = f(π_θ) = π_θ/(π_θ+0.1)`——**教师概率不显式出现**，shaping 直接作用于学生概率，与 LUFFY(Yan et al. 2025)实现一致。两者数学上不同：默认实验跑的是(b)，论文公式写的是(a)。`f(x)=x/(x+0.1)` 为沿用 LUFFY 的 shaping 函数。

#### 7. 实验数据集
- **教师池(4 个同量级 LongCoT 教师)**：AceReason-Nemotron-1.1-7B、DeepSeek-R1-Distill-Qwen-7B、OpenR1-Qwen-7B、Qwen3-8B(thinking)。
- **训练数据**：基于公开 OpenR1-Math-46k-8192 经多教师 curation 筛出的 **8.5k 高质量样本**。
- **基座**：主 Qwen2.5-7B-Instruct；另用 Qwen2.5-1.5B-Instruct、LLaMA3.2-8B-Instruct 验证。
- **评测**：6 个数学 ID 基准(AIME2024、AIME2025、AMC、Minerva、OlympiadBench、MATH500) + 3 个 OOD(ARC-c、GPQA*、MMLU-Pro)。

#### 8. 实验结果与主要发现(关键数字)
- 相比 GRPO：数学基准**平均 +4.3%**、**OOD +12.2%**。
- 提升 **Pass@k**，训练全程保持**更高 entropy**(探索性更强、未坍塌)。
- **数据效率**：用 4 个同量级教师 + 仅 **8.5k** 数据，即可媲美用单一更强教师(DeepSeek-R1) + **46k** 数据的方法。
- 论文另分析引导替换数量 k 与教师池组成的影响。

#### 9. 结果如何支撑其主张(证据链是否到位)
三个主张各有证据：(1) "多教师 > 单教师"——与单教师/单一强教师基线对比 + 教师池组成消融；(2) "按需 > 静态"——guidance-on-demand 仅在全失败时触发，对比静态注入；(3) "可理解度选择有效"——r_p 选择 vs 随机/最强教师。"更高 entropy + Pass@k 提升"共同支撑"保留了探索而非塌成单模"。证据链较完整，数据效率对比尤其有说服力。

#### 10. 逻辑自洽性(中性评估:哪里站得住、哪里牵强)
- **站得住**：on-demand 替换的逻辑(只在自探索失败时才注入)直接对应"最大化自探索 + 兜底正确解"，自洽；sequence-level 等权聚合避免长 CoT 主导梯度，工程上合理且代码确证。
- **牵强/需注意**：**论文公式(路径 a)与默认实验(路径 b)的重要性采样比不一致**——这是实现与理论之间的真实缺口。路径(b)里教师概率根本不进入 ratio，意味着"off-policy 重要性校正"在默认实验中并未按 Eq.7 执行，而是退化成 LUFFY 式的学生概率 shaping。读者若据论文公式理解机制，会与实际跑的代码错位。

#### 11. 残留问题 / 局限
- **公式-实现缺口**(见 §10)：默认脚本未用论文 Eq.7 的 `π_θ/π_target`，而用 `π_θ/(π_θ+0.1)`，二者数学不等价。
- 教师池需 4 个现成强教师，构建/采样成本不小；"同量级同伴"在更大/更小规模下是否仍优于单一强教师未充分验证。
- 阈值 τ、替换数 k_0 为超参，跨任务自适应性未知。
- 主验证集中在数学推理 + 少量 OOD，覆盖面有限。

#### 12. 开源代码与框架(链接 + 框架 + 代码可得性)
- 链接：https://github.com/SII-Enigma/AMPO （已克隆约 163M，**完整**）。核心改动在 `ampo/verl/verl/adaptive_mix_src/`(`mix_core_alg.py` 含两条 off-policy ratio 路径、`compute_token_on_seq_off_policy_loss` 序列级聚合)；含 `data/`、`exp_scripts/`(含 `train_ampo.sh`)、`eval_scripts/`、`examples/`、`figures/`。HF 组织 SII-Enigma 提供 checkpoint。
- 框架：基于 **verl**，实现 GRPO 的 Mixed-Policy 扩展；conda py3.10，支持 FSDP 或 Megatron(vllm/sglang/mcore)。


---


### asft — ASFT: Anchored Supervised Fine-Tuning

> **一句话重点 (TL;DR)**：DFT(用 token 概率给交叉熵重加权的 SFT 变体)在推理域好用、在知识域(如医疗)不稳，原因是它"界紧但会漂移"(KL 持续增大)。ASFT 只加一项**轻量 KL 锚定**把策略约束在 base 模型附近，就同时保住 DFT 的紧界优势和稳定性。值得看的点：它用 **RWR(reward-weighted regression)框架**统一解释 SFT/DFT，证明 DFT 给出比 SFT 可证更紧的 RL 下界，并指出其缺分布锚定才是不稳定根因——理论诊断 + 一行 KL 修复。

**元信息**：arXiv 2509.23753v3（2026-02-01）｜ 南方科技大学、北京大学、上海 AI Lab（He Zhu、Junyou Su 共一，通讯 Guanhua Chen）｜ ICLR 2026（2026-01-30 接收）｜ 主题 T3 GFT/SFT-RL 统一类，**相关性 High** ｜ 代码 https://github.com/zhuchichi56/ASFT（**完整**，约 54MB；已合入 LLaMA-Factory 主分支）｜ 框架 自定义 HF Trainer + DeepSpeed(ZeRO-2/3) + PEFT/LoRA；新增 veRL(FSDP2) 分支(README 推荐优先)

#### 1. 相关工作与进展(这条线现在做到哪一步)
后训练在 **SFT**(高效模仿但易记忆、泛化弱)与 **RL**(泛化好但贵且不稳)之间存在根本权衡。**DFT**(Dynamic/Discriminative Fine-Tuning)通过 token 概率重加权成为有前景的中间地带，在推理域取得改进。ASFT 处在"用统一理论框架解释 SFT-RL 中间地带、并修复其稳定性"这条线，是 **DFT 的轻量扩展**。

#### 2. 现有工作存在的问题(本文针对的痛点)
- **DFT 效果域相关**：推理密集型域好，但在知识密集型任务(如医疗 MedQA 等)**不稳定**，且其启发式重加权设计**缺乏理论依据**。
- 作者用 RWR 框架诊断出根因：DFT 对应一种特定 auxiliary distribution 构造，能给出比 SFT **可证更紧**的 RL 下界；但它**缺乏分布锚定**，导致**渐进式漂移**(KL 持续增大)——下界越来越松、重要性权重方差越来越大，从而训练不稳。

#### 3. Motivation(为什么做这件事)
既然 DFT 的问题是"紧但会漂移"，那么只要给它**加一个轻量 KL 锚定项**把策略约束在 reference(base 模型)附近，就能在保留 DFT 紧界优势的同时获得稳定性——兼得 SFT 的效率与 RL 的泛化。

#### 4. 主要灵感 / 核心直觉
核心直觉来自 **RWR 视角**：把 SFT/DFT/RL 都看成"用某个 auxiliary distribution 加权的回归"。DFT 选了一个能让 RL 下界更紧的加权(故推理域更好)，但这个加权没有任何力量把策略拉回 reference，于是越训越偏。**紧界 + 锚定 = 既贴近最优又不跑偏**——这正是 RL 里 KL 惩罚的作用，ASFT 把它搬进 SFT。

#### 5. 主要解决思路(一段话把核心机制讲清)
ASFT = **DFT 概率重加权 + KL 锚定**。在 DFT 的 token 级重加权交叉熵上，加一项把当前策略 π_θ 拉向 reference(base/原模型，LoRA 时禁用 adapter 即得)的 per-token KL，用一个小权重 kl_weight 平衡。重加权负责"紧"，KL 锚定负责"稳"。

#### 6. 方法详解(通俗、分步骤;关键公式用白话解释,必要时给伪代码)
代码(`train_v2.py`, `mode="asft"`)逐行核对(已确认与论文一致)：
- **DFT 重加权**：`weights = softmax(logits).gather(label).detach()`；`dft_losses = token_losses · weights`。白话：模型对正确 token 当前预测概率越高，这个 token 的 loss 权重越大——把学习集中到"模型已经有点会、再推一把就稳"的 token 上。
- **KL 锚定**：`kl_div = KL( log_softmax(π_θ) ‖ softmax(ref) )`(per-token，对 vocab 求和)；ref 为 base/原模型(`disable_adapter()` 取 reference，或单独加载 `original_model` 并冻结)。
- **最终损失**：`weighted_losses = dft_losses + kl_weight · kl_div`，再按 valid_mask(非 -100 的回复 token)归一。
- **kl_weight**：代码默认 **0.1**(train_v2.py / .sh)，但 README 与示例命令在**混合精度(bf16/fp16)下推荐 0.03**——过大会放大精度噪声致失稳。
- 代码同时实现 `sft / dft / sft+kl / asft` 四种 mode，便于消融对照。
- **理论**：在 RWR 框架下证明 DFT 给出比 SFT 严格更紧的 RL 下界(论文 _txt 确认 "provably tighter bound than SFT")，KL 锚定控制方差/漂移。

#### 7. 实验数据集
- 三类域：**数学推理**(100k 训练样本)、**医疗知识 grounding**(MedQA/MMLU/MedMCQA，10k)、**代码生成**。
- 基座：**LLaMA-2-7B 与 Qwen2.5-7B** 两个 7B 模型(论文 §5.1；选 LLaMA-2-7B 是为规避数据污染)。
- 对比：SFT、DFT、iw-SFT、SFT+KL；并验证 ASFT 作为 DAPO/GRPO 的更优初始化。

#### 8. 实验结果与主要发现(关键数字)
- **数学(100k)**：较 DFT **+4.85 点(18.6%)**，较基座 **+17.89 点(142%)**。
- **医疗(10k)**：较 SFT **+8.28 点(24.8%)**，较基座 **+10.65 点(33.9%)**。
- **效率**：仅需全 RL 约 **3%** 的训练算力。
- 跨推理/知识两类域均稳定优于 SFT、DFT；泛化接近 RL 而保持 SFT 级效率。
（数值已对照论文 _txt 第 131-134 行核实。）

#### 9. 结果如何支撑其主张(证据链是否到位)
证据链完整：理论(RWR 框架证 DFT 更紧界 + KL 锚定控漂移) → 诊断(DFT 在知识域因 KL 漂移失稳) → 修复(加 KL 锚定) → 实验(数学 vs 医疗两类域均稳定超 DFT/SFT)。"医疗域 DFT 不稳而 ASFT 稳"直接对应理论诊断的预测，归因较扎实；"3% 算力达近 RL 泛化"支撑"兼得效率与泛化"的主张。

#### 10. 逻辑自洽性(中性评估:哪里站得住、哪里牵强)
- **站得住**：RWR 统一框架 + "紧界但缺锚定→漂移"的诊断逻辑清晰；KL 锚定就是把 RL 的 KL 惩罚搬进 SFT，理论动机干净；代码与公式逐行吻合(DFT 项 + KL 项)。
- **牵强/需注意**：(1) ASFT 本质是"DFT + 已知的 KL 正则"，新颖性更多在**理论解释**而非机制本身(SFT+KL 早已存在，论文也把它列为基线)；(2) kl_weight 在不同精度/域下需手调(0.1 vs 0.03)，说明稳定性对该超参敏感，"轻量锚定"并非完全免调；(3) 仅两个 7B 基座，规模与家族覆盖有限。

#### 11. 残留问题 / 局限
- 〔待核〕RWR 框架下"更紧下界"的形式化定理与 kl_weight 在不同域的敏感性曲线需对照正文与附录细读。
- 仅 LLaMA-2-7B / Qwen2.5-7B 两个 7B 模型，未验证更大规模或更多家族。
- kl_weight 需按精度/任务调参(默认 0.1，bf16/fp16 推荐 0.03)，DeepSpeed 下有精度问题(故新增 veRL/FSDP2 分支)。
- 相对 DFT 的增量主要是一项 KL 锚定 + 理论解释，机制创新有限。

#### 12. 开源代码与框架(链接 + 框架 + 代码可得性)
- 链接：https://github.com/zhuchichi56/ASFT （已克隆约 54MB，**完整**）。数据 HF: `chichi56/ASFT`。
- 实现：主分支 `train_v2.py`(loss 实现，mode ∈ {sft, dft, sft+kl, asft}) + `scripts/ds_zero*.json`；新增 **veRL 分支**(2026-05-08，FSDP2，数值更稳，README 推荐优先)；已合入 **LLaMA-Factory 主分支**(2026-02-12, commit #10174)。
- 框架：自定义 HF Trainer + DeepSpeed(ZeRO-2/3) + PEFT/LoRA；新增 veRL(FSDP2)；已并入 LLaMA-Factory。（注：repo 与论文均未提及 ms-swift。）
- 推荐配置：full SFT 用 lr=2e-5、3 epoch、kl_weight=0.03；LoRA 推荐 r=8/alpha=16/dropout=0.05/lr=5e-4(医疗任务最优)。


---


### bapo — BAPO: Stabilizing Off-Policy Reinforcement Learning for LLMs via Balanced Policy Optimization with Adaptive Clipping

> **一句话重点 (TL;DR)**：异策略（off-policy）RL 训练 LLM 时，数据越陈旧（staleness 越大）越容易梯度爆炸、熵崩溃。BAPO 找到两个根因——负优势样本主导梯度、固定对称裁剪系统性挡掉"增熵更新"——并据此**每个 batch 动态调裁剪上下界**，让"正 token 贡献占比"达到目标 ρ0，从而既防爆炸又保熵。

**元信息**：arXiv:2510.18927v1（2025-10-21，cs.LG）｜ 复旦 FudanNLP / 上海稷迹智锋 / 上海创新研究院（共一 Zhiheng Xi、Xin Guo；通讯 Tao Gui、Qi Zhang）｜ 2025-10 预印本｜ 主题：off-policy RL 稳定化（自适应裁剪 + 熵保持），与 OPD **直接关联弱**，但其 Entropy-Clip Rule 对理解 partial rollout / 经验回放等异策略训练的熵崩溃有参考价值｜ 代码 https://github.com/WooooDyy/BAPO （已 clone 约 5.2MB，基于 veRL，方法在 `recipe/bapo`，真实可用）｜ 框架 veRL（GRPO 为基础算法）。

---

#### 1. 相关工作与进展
- RL 已成对齐/强化 LLM 的核心范式（推理、代码、agentic）。
- **off-policy RL**（rollout 行为策略 ≠ 训练目标策略，使用过期数据）样本效率高、契合现代基础设施（**partial rollout**、经验复用），适合超长/难任务。
- 与之相关的"非对称裁剪"家族已有多项：Clip-Higher（DAPO）、KL-Cov、CE-GPPO、80/20 等，思路都是"放宽对低概率正 token 的裁剪"。

#### 2. 现有工作存在的问题
直接套用 off-policy RL 到 LLM：随 staleness 增大（Fig.2），出现**优化不稳、梯度爆炸、甚至崩溃**，同时**策略熵急剧下降**（探索退化、过度利用）。已有非对称裁剪方法**手动固定阈值**，僵硬、缺乏自适应。

#### 3. Motivation
作者通过理论+实证锁定两个根因：
- **(i) 优化失衡**：负优势样本在数量与对策略梯度损失贡献上都占主导，过度惩罚会压制有用/中性行为；且低概率负 token（log 项 → −∞）易引发梯度爆炸。难题/训练早期会进一步加剧负样本占比。
- **(ii) Entropy-Clip Rule（理论推导）**：PPO 类目标的**固定对称裁剪**会系统性阻断"增熵更新"——把大量低概率正 token 挡在更新之外，同时过度惩罚低概率负 token，使分布锐化、熵崩溃。
据此目标：平衡正负贡献 + 防梯度爆炸 + 保熵维持探索。验证实验表明非对称裁剪（增大上界 c_high 纳入更多低概率正 token）能提升性能并抑制熵下降，但固定阈值不够灵活——故引出自适应版本。

#### 4. 主要灵感 / 核心直觉
既然问题是"正 token 被裁太多、负 token 罚太狠"，那就**用'正优势信号对策略梯度损失的贡献占比'作为可观测的调节目标**，每个 batch 自适应地扩裁剪窗，直到正 token 贡献达到目标占比 ρ0——免去手动调阈值。

#### 5. 主要解决思路（一段话讲清核心）
BAPO（Balanced Policy Optimization with Adaptive Clipping）在 GRPO/PPO 代理目标上**动态调整裁剪上下界 (c_high, c_low)**：每个 batch 搜索一对界，使正优势信号对策略梯度损失的贡献占比 ≥ 目标 ρ0；扩 c_high 纳入更多低概率正 token（增熵），扩 c_low 过滤过多低概率负 token（防爆炸），设 ρ0 又能防止熵无控增长。

#### 6. 方法详解（通俗、分步骤）
- 代理目标仍是 PPO/GRPO 的 min(r·A, clip(r,1−ε,1+ε)·A)，但裁剪界不再固定。
- **论文 Algorithm 1 的搜索顺序**：从 c_low=a−、c_high=a+ 起，while 循环内**先增 c_high**（步长 δ1，优先纳入更多低概率正 token）至 b+；不够再增 c_low（步长 δ2，过滤低概率负 token）——即"先 c_high 后 c_low"，单 while 内交替。
- **〔code 不一致，已核〕**：开源实现 `recipe/bapo/policy_loss.py::compute_policy_loss_bapo` 顺序与论文**相反**——先 "adjust lower clip range first"（增 c_low 到 ratio_lower_max），不满足再 "increase upper bound"（增 c_high），且为**两段顺序循环**而非论文的单 while 交替。差异不改方法本质（都为满足正 token 贡献占比而扩裁剪窗），但与 Algorithm 1 表述不符，**复现以代码实际行为为准**。
- **〔code 额外细节，已核〕**：代码对负优势样本另含 **dual-clip**（clip_ratio_c 默认 3.0），论文 Eq.8 正文未强调。
- 净效果：增大正 token 贡献、防负 token 主导与梯度爆炸；纳入低概率正 token + 过滤低概率负 token 来保熵；ρ0 防熵失控与尾部退化。免去 DAPO/手动非对称裁剪的繁琐调参。

#### 7. 实验数据集
- **RL 训练数据**：SkyWork-OR1-RL-Data。
- **评测**：AIME 2024、AIME 2025（report 16 rollouts 平均）。
- **骨干**：DeepSeek-R1-Distill-Qwen-7B/32B、OctoThinker-Llama3.2-3B-Long-Zero；并自训两个 SFT 模型 BP-Math-7B/32B（由 Qwen2.5-Math 微调而来）。

#### 8. 实验结果与主要发现
- **BP-Math-7B(BAPO)**：AIME24/25 = **70.8 / 62.5**，超 SkyWork-OR1-7B（70.2 / 54.6，AIME25 +7.9）。
- **BP-Math-32B(BAPO)**：**87.1 / 80.0**，同规模 SOTA，并超 o3-mini-medium、Gemini-2.5-Flash-Thinking、DeepSeek-R1(671B)。
- **Llama**（GRPO→BAPO）：AIME24 2.5%→5.4%、MATH 58.4%→66.0%。
- partial rollout 与不同 staleness 下均比 GRPO 更稳（Fig.2 复现的崩溃在 BAPO 下消失）。

#### 9. 结果如何支撑其主张
主张是"自适应裁剪能稳定 off-policy RL 并保熵"。支撑：(a) staleness 扫描显示 GRPO 崩溃而 BAPO 稳定（直接对应 motivation）；(b) 主基准上超 SkyWork-OR1 与多个专有系统。但支撑链有缺口——见 §10/§11。

#### 10. 逻辑自洽性（中性评估）
- 理论贡献 **Entropy-Clip Rule** 较清晰，把"裁剪→挡增熵更新→熵崩溃"链条讲通，且与 motivating 实验一致。
- 但**主结果泛化覆盖窄**：核心评测只在 AIME24/25 两个小测试集，Llama 才补 MATH。
- 与 SkyWork-OR1 的对比部分依赖**自训的强 SFT 起点 BP-Math**；BAPO 相对自家 GRPO 的增益在 32B 上较小（AIME24 84.6→87.1），削弱了"方法本身"的归因强度。
- 仓库默认 config/example 是示例值而非论文实验值，复现需手动改超参（见 §12）——这本身不影响逻辑，但增加复现摩擦。

#### 11. 残留问题 / 局限
- 评测基准窄（AIME 为主），泛化性证据不足。
- 自适应搜索的目标占比 ρ0、可移动区间、步长仍是超参（论文称"未精调"），其鲁棒性未系统扫描。
- **代码与论文 Algorithm 1 搜索顺序相反**，方法描述与实现存在不一致（已核，见 §6）。
- 与既有非对称裁剪家族（Clip-Higher/KL-Cov/CE-GPPO/80-20）同源，**创新点集中在"用正 token 贡献占比作自适应目标"+ Entropy-Clip Rule 理论**，算法增量有限。

#### 12. 开源代码与框架（链接+框架+代码可得性）
- 代码：https://github.com/WooooDyy/BAPO （resource/repos/bapo，约 5.2MB；含 verl 子目录 + `recipe/bapo`）。依赖 hydra-core、liger-kernel、accelerate 等。
- 框架：**veRL**，以 **GRPO** 为基础算法。`python -m verl.trainer.main_ppo` 主循环。
- 训练设定：预备/验证实验用 R1-Distill-Qwen-7B，max len 8k、lr 2e-6、temp 0.6（staleness 实验用 SkyWork-OR1-RL、max len 32k）；BP-Math 主实验 max len 64k。通过 ppo_epoch 经验复用与 partial rollout 引入 staleness。
- **论文 BAPO 超参（未精调）**：ρ0=0.4，可移动区间 a−=0.6/b−=0.9、a+=1.2/b+=3.0，步长 δ1=0.05/δ2=0.02。
- **〔复现注意，已核〕**：仓库默认 config（`recipe/bapo/config/bapo_trainer.yaml`）与 `run_bapo_example.sh` 给的是**示例值非论文值**——config 默认 adv_ratio_target=1、ratio_lower 0.6→0.8 step0.05、ratio_upper 1.2→2.0 step0.05；example.sh 用 ppo_epochs=2、low_max=0.95、upper_step 0.1。复现论文数字需手动改回上述论文超参。
- 代码可得性：方法实现完整可跑，仅需对齐超参。


---


### behavior_priming — Beneficial Reasoning Behaviors in Agentic Search and Effective Post-training to Obtain Them

> **一句话重点 (TL;DR)**：先用 LLM pipeline 找出让"搜索 agent"成功的四种推理行为（验证、权威评估、自适应搜索、纠错），再用 SFT 把这些行为"种"进模型、之后再 RL。关键证据：用"展现这些行为但答案错误"的轨迹做 SFT，效果 ≈ 用答案正确的轨迹——说明**解锁 RL 的关键是推理行为（path）而非 outcome 正确性**。

**元信息**：arXiv:2510.06534v3（2026-01-16，cs.AI）｜ CMU LTI（Jiahe Jin、Abhijay Paladugu、Chenyan Xiong）｜ 2026-01 预印本｜ 主题：SFT-then-RL 的"行为先验"版（agentic search），**与 TSRD 高度相关**——"RL 前先种入特定推理行为/path 比 outcome 正确性更重要"正对应 TSRD 的"teacher-scaffolded→内化→RL"，且其错误轨迹消融为"path 监督 > outcome 监督"提供直接证据；本文**不涉及 MTP**，行为靠 LLM-judge 识别而非 logit 蒸馏｜ 代码 https://github.com/cxcscmu/Behavior-Priming-for-Agentic-Search （旧 KEYS 的 `cxcscmu/Behavior-Priming` 为 404，此为正确仓库名，已验证 200；RL 训练在另一仓库 `cxcscmu/verl-agent-deepresearch`）｜ 框架 SFT 用 LLaMA-Factory、RL 用 GRPO（veRL 系）。

---

#### 1. 相关工作与进展
- **Agentic search**：LLM 多步检索解复杂信息需求（分解任务、多步搜索、综合结果）。商用如 ChatGPT Deep Research、Google AI Mode；开源侧 RL 训练 agentic search 进展快（Search-R1、R1-searcher、DeepResearcher）。
- 训练范式分两类：早期靠强模型轨迹**蒸馏/SFT**；近期主流是端到端 **online RL**；不少工作采 **SFT-then-RL** 两段式先初始化工具使用与推理能力。
- 数学/代码领域已有发现：**RL 是否成功高度依赖 base model 是否已具备特定推理模式**（如 verification / backtracking）——但这类研究**几乎只在数学**。

#### 2. 现有工作存在的问题
- 对 agentic search，"**哪些具体推理行为有益、如何系统培养**"仍不清楚。
- agentic search 有独特挑战：海量结果中识别有用信息、解决来源冲突、长轨迹中保持目标聚焦。
- 关键观察：non-primed 模型在 RL 过程中**无法内生地**发展出这些行为——所以需要显式行为先验。

#### 3. Motivation
先搞清"什么推理行为让搜索 agent 成功"，再可靠地把它们植入模型，为 RL 建立稳健的探索与 test-time scaling 基础。

#### 4. 主要灵感 / 核心直觉
把数学领域"RL 成功取决于 base 已有推理模式"的发现迁移到 agentic search：与其指望 RL 自己长出好行为，不如**事先用 SFT 把成功轨迹里反复出现的行为种进去**，让 RL 在此基础上精炼。

#### 5. 主要解决思路（一段话讲清核心）
两步：(1) 用 LLM-based 三阶段 pipeline 对比"强模型成功 vs 弱模型失败"的轨迹，提炼出四种普适有益行为；(2) **Behavior Priming** = 先在"同时展现全部四种行为"的轨迹上做 step 级 SFT，再在 primed 模型上跑 GRPO（outcome 二值奖励）。

#### 6. 方法详解（通俗、分步骤）
**第一部分：识别有益行为**
- 先建标准 agentic search 框架：每步输出 reasoning + action；action ∈ {search, answer, **summary**}（summary 压缩历史以管理上下文长度）。
- 配对轨迹：Gemini 2.5 Flash（强）成功而 Qwen3-1.7B（弱）失败的 **500 题**。
- 三阶段 LLM pipeline：①轨迹比较 → ②行为抽取（这两步用 Gemini 2.5 Flash）→ ③行为合并去重（Gemini 2.5 Pro），再人工复核。
- 得到**四种行为**：**Information Verification**（跨源验证并引证）、**Authority Evaluation**（识别冲突、评估来源可信度）、**Adaptive Search**（动态调整搜索策略）、**Error Recovery**（识别并纠正先前错误）。跨 Gemini/DeepSeek-R1/Llama/Qwen 多模型测得"行为频率与性能强相关"（Fig.2），证其普适。

**第二部分：Behavior Priming（SFT-then-RL）**
- **SFT**：用 Gemini 2.5 Flash 每题采 10 条轨迹，筛出"同时展现全部四种行为"的轨迹；把每条轨迹的**每一步当独立训练样本**（D_SFT = {(x_k, y_k)}）。
- **RL**：在 behavior-primed 模型上跑 **GRPO**，按 step 聚合更新；**outcome-based 二值奖励**（LLM-judge 判最终答案对/错，1/0），该奖励与 advantage 对轨迹内所有 step 恒定（无 step 级信用分配）。
- **核心消融**：对比"展现目标行为但答案错误 (Behavior Prime Incorrect)" vs "答案正确 (Correct)"两组轨迹做 prime——用**错误轨迹** prime 的模型性能 ≈ 用正确轨迹的（Table 1 显示两组 SFT 数据行为频率均 100%、accuracy 分别 0% / 100%）。结论：**推理行为而非 outcome 正确性才是解锁 RL 潜力的关键**。
- **训练动态**：SFT 后行为频率、pass@8、平均步数同升；RL 中 primed 模型维持更高 policy entropy，而 direct RL 熵骤降、过早收敛。

#### 7. 实验数据集
- **backbone**：Qwen3-1.7B、Llama3.2-3B-Instruct。
- **评测**：3 个 web agent benchmark + 7 个 multi-hop QA benchmark。
- **数据来源**：SFT 数据问题/答案取自 Li et al. 2025c；RL 用 Li et al. 2025c 的 web agent RL 集 + Zheng et al. 2025 的 multi-hop QA 集。行为识别的配对轨迹来自上述 500 题。

#### 8. 实验结果与主要发现
- 相对 **Direct RL**：web benchmark **+37.2%**、multi-hop QA **+6.2%**（相对提升）；并一致超过两个 SFT-then-RL 基线（Distillation 随机采样 / Outcome-driven 只筛正确轨迹）。
- 增益在 **web 任务上明显**，QA 上较温和（+6.2%）。
- non-primed 模型 RL 中**不会自发**长出四种行为——印证显式先验的必要性。
- §5.3 数据规模消融（5k/10k/20k）、§5.4 单行为 (IV-only) vs 复合四行为对比。

#### 9. 结果如何支撑其主张
主张"行为先验 > outcome 正确性"。支撑很直接：错误轨迹 prime ≈ 正确轨迹 prime（控制了行为频率，只变 outcome），且超过 Outcome-driven 基线（专门筛正确轨迹）。训练动态（熵、pass@8、步数）从机制上解释了为何 primed 能解锁 RL。证据链对该主张闭合度较高。

#### 10. 逻辑自洽性（中性评估）
- 自洽：识别→培养→验证三段闭合，错误轨迹消融是干净的因果隔离实验。
- 张力点：四种行为由 **LLM-judge 识别与筛选**，整套结论依赖判别 prompt 的质量与 Gemini 的判断一致性，存在循环依赖（用 LLM 定义"好行为"、又用 LLM 评估是否展现）；"行为频率与性能相关"是相关而非因果证据，真正因果靠错误轨迹消融补足。

#### 11. 残留问题 / 局限
- **规模偏小**（1.7B / 3B），是否在更大模型上同样需要显式先验未知。
- RL 仍是 **outcome 级 GRPO**，无 step/turn 级信用分配——与 path 监督的主张存在张力（SFT 阶段是 path，RL 阶段又退回 outcome）。
- 行为定义/识别依赖特定 LLM-judge 与 prompt，可迁移性与稳健性未充分检验。
- 不涉及 MTP；对 TSRD 的价值在于"**path 监督 > outcome 监督**"这一可迁移结论的强证据，而非具体机制。

#### 12. 开源代码与框架（链接+框架+代码可得性）
- 主仓库：https://github.com/cxcscmu/Behavior-Priming-for-Agentic-Search （含 DeepResearch agent scaffold、行为分析 pipeline `behaviour_analysis/`、SFT recipe、agent 主循环 `main_parallel.py`、评测套件 `evaluation/short/`）。
- **SFT 框架**：**LLaMA-Factory**（`scripts/train.sh` → `llamafactory-cli train train/sft/sft.yaml`，仓库内 vendored LLaMA-Factory）。
- **agent scaffold / rollout / 评测**：vLLM 服务底层模型（Qwen3 开内置 thinking，无内置推理的模型加 `use_explicit_thinking`）。
- **RL 框架**：**GRPO**，实现在**独立仓库** https://github.com/cxcscmu/verl-agent-deepresearch （veRL 系，verl-agent 的 deep-research 变体）；本主仓库**不含 RL 训练代码**。
- 代码可得性：行为分析 + SFT + agent/评测可跑；RL 需到第二仓库。


---


### ca_survey — From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models

> **一句话重点 (TL;DR)**：一篇 survey + awesome-list，以"信用分配 (credit assignment, CA)"为中心透镜重审 LLM RL。核心产物是把 2024–2026 初的 **47 篇方法**（41 篇 core + 6 篇 adjacent）按**粒度 × 方法学二维分类**，并配 reporting checklist、benchmark 协议与方法选择决策树。判断：reasoning CA 正趋成熟（PRM + critic-free group comparison），agentic CA 催生真正新方法（hindsight counterfactual、privileged asymmetric critic、turn-level MDP）。

**元信息**：arXiv:2604.09459v2（2026-04-13，cs.CL）｜ Chenchen Zhang（独立研究者，**单作者** survey）｜ 2026-04 预印本｜ 主题：LLM RL 的 credit assignment 综述（**非新方法论文**）。与 OPD/TSRD/MTP 不直接相关，但 CA（把稀疏 outcome 奖励分配到 token/step/turn）正是 TSRD 中"path-selection / path-recovery 细粒度信用分配"的上位框架，可用于定位、选 baseline、找相邻工作｜ 代码/列表 https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL （旧 KEYS 的 `xxzcc/Awesome-Credit-Assignment` 为 404，此为正确仓库名，已验证 200）｜ 框架 不适用（资源整理 + `gen_figures.py` 生成分类图）。

> 说明：按设定，本分析重点放在 §6（分类法）与 §8/§9（覆盖范围 & gaps）。

---

#### 1. 相关工作与进展
- LLM RL 两波：**reasoning RL**（DeepSeek-R1/o1，单条 CoT 500–30K+ token，纯 outcome 奖励）→ **agentic RL**（多轮环境交互，10–100+ turn，100K–1M token，稀疏 terminal 奖励）。
- 经典 RL CA 工具箱（TD/GAE、return decomposition RUDDER、hindsight HCA、counterfactual/difference reward）是 LLM 方法的基础。
- 现有相邻综述：Pignatelli 2023（经典 RL CA，前 LLM 时代）、Zhang 2025a（agentic RL 100 页综述，把 CA 当子话题）——**无人系统覆盖 reasoning + agentic 两个 regime 的 CA**。

#### 2. 现有工作存在的问题（survey 要解决的 gap）
- **Episode 级方法**（GRPO、REINFORCE）给每个 token 同一 advantage，trajectory 一长就失效（turn 3 的错误 tool-call 与后续几十个正确动作受同样惩罚；"echo trap" 现象）。
- agentic 设定使 CA **双层级化**：先判哪个 turn 关键，再判 turn 内哪些 token 重要；并有随机转移、部分可观测、超长 horizon。
- 既有综述要么只覆盖经典 RL，要么把 CA 当 agentic 的子话题，**缺乏专门、跨两 regime 的 CA 综述**。

#### 3. Motivation
以 credit assignment 为中心透镜重审 LLM RL，给出 reasoning→agentic 演进叙事（Classical RL → Reasoning RL → Agentic RL → Future Multi-Agent），并产出可复用资源（清单、checklist、benchmark 协议）。

#### 4. 主要灵感 / 核心直觉
"PRM 本质就是 CA"——一个给每步打分的 Process Reward Model，其实是在做 step 级的 terminal-reward 分解；PRM 文献与 CA 文献是同一问题的两个视角。把所有方法都拉到"如何把稀疏奖励分配到动作"这一统一问题下。

#### 5. 主要解决思路（一段话讲清核心）
做一篇专门的 CA 综述，用"粒度 × 方法学"二维分类组织 47 篇方法，配三类标准化产物（机器可读清单、reporting checklist、benchmark 协议 + 决策树），并明确论证"为何 agentic 比 reasoning 的 CA 更难、催生哪些新技术"。

#### 6. 方法详解（= 分类法 taxonomy，本文核心贡献）
覆盖 2024–2026 初 **47 篇方法**（**41 篇 core CA + 6 篇 adjacent enabler**）。**二维分类法**（§2.4，Fig.2）：
- **粒度轴 (granularity)**：Token / Segment / Step-Turn / Multi-Agent。
- **方法学轴 (methodology)**：Monte Carlo（VinePPO）/ Temporal Difference + value baseline（GAE；ArCHer 用 off-policy critic、AgentPRM 用 TD+GAE 学 turn-level value）/ **Model-based / LLM-as-Critic**（〔已核更正〕LLM-as-Critic 是论文 Fig.2 方法学轴**正文已列**的一列、且论文强调它"无经典 RL 对应"，并非 repo 后加；repo 后扩的是 uncertainty-control、verifiable-feedback shaping 等新类）/ Game-theoretic（Shapley、counterfactual）/ Information-theoretic（信息增益、熵）。
- **经典→LLM 映射**：return decomposition → reward redistribution（RED、SPA-RL）；hindsight → retrospective（HCAPO，基于 generative verification）；counterfactual → leave-one-out 与 Shapley（C3、CCPO 做 turn 级 LOO，SCAR 用 Shapley value）；TD/GAE → learned critics（ArCHer、AgentPRM）。
- **LLM-as-Critic 是新轴**：LLM 自身作 critic 给中间状态自然语言评价（CAPO 等），经典 RL 无对应。
- **综合判断**：reasoning CA 正收敛到 **process reward model + critic-free group comparison**（成熟化）；agentic CA 催生真正新方法 —— **hindsight counterfactual、privileged asymmetric critic、turn-level MDP 重构**（无 reasoning RL 先例）。
- **三类可复用产物**：(1) 机器可读的 47 方法清单（taxonomy 标签 / baseline family / evidence level / 主 benchmark，将以 CSV/JSON 发布，§B）；(2) 面向未来 CA 论文的 **reporting checklist**（§C，已对现有文献验证以找系统性 gap）；(3) **benchmark 协议规范**（任务族、元数据要求、controlled bifurcation tasks）+ 方法选择**决策树**（Fig.4 / Table 8，§9）。

#### 7. 实验数据集
不适用（无自有实验）。但 §7 做 **systematic comparison**：按计算成本、是否需辅助模型、适用场景、实证性能比较各方法，含 GRPO-family meta-comparison；§9 给 benchmark 协议建议（任务族 + controlled bifurcation tasks）。

#### 8. 实验结果与主要发现（= 文献覆盖与组织方式）
- **覆盖范围**：2024-01 至 2026-04 的 LLM RL CA 方法；arXiv/Semantic Scholar/Google Scholar 关键词检索（"credit assignment / process reward / reward decomposition / turn-level reward" × LLM/RL），并从 VinePPO/ArCHer/GRPO/DeepSeek-R1 前后向引用追踪，监控 NeurIPS/ICML/ICLR/ACL 2025 与 HF Daily Papers。
- **分类口径**：**core**（主算法贡献是新的稀疏奖励再分配，如 VinePPO/HCAPO/CARL）vs **adjacent enabler**（基础设施/reward shaping/agent 框架，如 Agent Lightning/RAGEN/PRS）；边界 case 在 §9.4 讨论。
- **章节组织**：§2 背景+形式化（reasoning = token-level MDP；agentic = turn-level POMDP）+ 分类法；§3 reasoning CA；§4 为何 agentic 更难；§5 agentic 专属 CA；§6 multi-agent CA；§7 系统比较；§8 把 CA 放进 agentic RL 训练 pipeline；§9 开放问题（multi-agent credit、超长 horizon、exploration-credit 互作用）+ roadmap；§10 结论。
- **覆盖判断**：reasoning RL 的 CA 已"成熟化"（token/segment/step 方法密集，PRM 主流）；step/turn 级是 LLM agent 自然粒度、聚类最密；multi-agent 与超长 horizon 是最薄弱处。

#### 9. 结果如何支撑其主张（= 覆盖范围 & gaps）
- **主张**："reasoning→agentic 的演进复杂化并重塑了 CA 版图。" 支撑方式：分类网格 Fig.2/Fig.3 直观展示方法在 reasoning（左上密集）→ agentic（右下）的迁移；§4 论证 SNR 随 turn 数恶化（T=100 时单动作信噪比约差 100×）；agentic 专属新方法（HCAPO/C3/CCPO/SWEET-RL）无 reasoning 先例。
- **自陈 gaps（§9.4）**：单作者 survey，覆盖可能有缺口；multi-agent CA、超长 horizon、exploration 与 credit 的互作用是开放前沿。
- 作为 survey，其"支撑"靠系统检索协议 + 分类一致性 + checklist 对现有文献的验证，而非实验数据。

#### 10. 逻辑自洽性（中性评估）
- 自洽：分类二维轴正交清晰，经典→LLM 映射明确，PRM=CA 的概念澄清有说服力。
- 张力点：(a) **单作者**，覆盖与分类标签难免主观（作者已自陈）；(b) 47 篇含 6 篇 adjacent，"core vs adjacent" 边界（如 RAGEN、Agent Lightning）有判断空间；(c) 部分被引方法（2026 年的 HCAPO/C3/CCPO 等）极新，evidence level 参差，survey 的"成熟化"判断对早期方法可能偏乐观。

#### 11. 残留问题 / 局限
- 单作者覆盖缺口（§9.4 自陈）。
- 是**索引 + 分类 + 标准化提案**，无新方法、无新实验；CSV/JSON 清单"将于发表时发布"，截至 snapshot 未必齐全。
- 与 TSRD 关系：可作定位/找 baseline 的索引，但本身不涉及蒸馏/MTP/path-recovery 的具体机制。

#### 12. 开源代码与框架（链接+框架+代码可得性）
- 列表：https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL （MIT，PRs welcome，**living list**：已超出论文 snapshot，2026.05 又加了 agentic/coding-agent/uncertainty-control/multi-agent 等新论文）。
- 内容：`README.md`（带分类表）、`gen_figures.py`（生成 taxonomy 网格图与树图）、`assets/`。
- 框架：**不适用**——仓库是论文清单 + 分类可视化脚本，非训练代码。
- 代码可得性：脚本可跑出分类图；机器可读 CSV/JSON 清单论文称"发表时发布"，需留意是否已上传。


---


### cbrl — Context Bootstrapped Reinforcement Learning (CBRL)

> **一句话重点 (TL;DR)**：把 few-shot 示范当成 RLVR 训练时的"临时脚手架"——早期以高概率把示范前置到 prompt 里帮模型产生成功 rollout（拿到学习信号），再按课程把注入概率线性退火到 0，逼模型把推理模式"内化"而非依赖；只改训练输入分布、不动 RL 目标，因此算法无关、推理零开销。

**元信息**：arXiv 2603.18953（v1，2026-03-19，cs.LG）｜ UC Santa Barbara + Cisco Research（Saaket Agashe、Jayanth Srinivasa、Gaowen Liu、Ramana Kompella、Xin Eric Wang）｜ Preprint ｜ 主题：RLVR 探索效率（与 TSRD"教师脚手架/路径引导后撤除"直觉高度同构，作为 prompt 层面的极简对照）｜ 代码 https://github.com/context-bootstrapped-rl/cbrl （项目页 https://context-bootstrapped-rl.github.io，已 clone，约 1.3MB，研究原型）｜ 框架 verl 0.3.0.post2 + vLLM 0.8.5 + PyTorch 2.6.0，Hydra 配置

#### 1. 相关工作与进展
- **RLVR**（Tülu3/Lambert 2024；DeepSeek-R1 2025）已是推理后训练主流，靠确定性 verifier 给二元奖励，在数学、代码、工具调用上推动显著进展；GRPO（Shao 2024）用组内归一化去掉 value 网络，成事实标准算法。
- **探索低效（exploration inefficiency）的现有解法**：(i) 混合策略训练——RL 中混入/替换 off-policy 轨迹（LUFFY 用正则化重要性采样）；(ii) 部分监督/提示——给 ground-truth 解片段救活失败 rollout（HINT、GHPO、BREAD）；(iii) 课程方法——E2H（易到难）、Absolute Zero（自演化课程）。

#### 2. 现有工作存在的问题
- 零样本 RLVR 在"新推理模式 / 领域知识缺失"场景下无法 bootstrap：当一组 rollout 全错时，GRPO 组优势坍缩为 0、没有梯度，模型靠自身探索越不过 reward 平台期（领域欠表示如小众编程语言尤其严重）。
- 持续提供示范又会让模型**依赖**示范、测试时（无示范）无法独立工作。
- 现有解法的代价：mixed-policy 易把更新引向不可泛化的 off-policy 解路径；partial supervision 会在 rollout 中途干预；课程方法需要难度估计或自生成题目等额外基础设施。

#### 3. Motivation
利用 LLM 自带的 in-context learning 能力当**临时**脚手架：早期高频注入 few-shot 引导模型产生成功 rollout 从而拿到学习信号；随训练推进把注入概率退火到 0，迫使模型把示范中的推理模式内化、最终在无示范下独立成功。相比 mixed-policy，CBRL 保持完全 on-policy——示范只是"上下文"而非要模仿的轨迹；相比 partial supervision，不在 rollout 中途干预；相比课程方法，在固定分布上用一个简单退火即构成"从被引导到独立"的隐式课程，无需外部基础设施。

#### 4. 主要灵感 / 核心直觉
- few-shot 示范提供的不是要照抄的目标，而是"怎么开始尝试"的引导，所以保持 on-policy 探索动力学不被破坏。
- 批内同时存在"带示范"与"不带示范"两种样本：若每条 prompt 都带示范，模型会学成依赖示范才会做题；随机不给，强制它即便早期也尝试独立解题。
- 退火 = 隐式课程：早期弱→多给脚手架，后期强→撤掉，"内化而非依赖"。

#### 5. 主要解决思路（一段话讲清核心）
CBRL 三个组件：**(1) Few-Shot Example Bank ℬ**——每条含问题 q、可选推理轨迹 r、答案 a，来源可为专家示范、更强模型解或手工构造；**(2) 随机上下文注入**——每步以概率 p_i 用 Bernoulli 决定每个 prompt 是否前置 k 条示范（作为对话对），reward 只对生成响应计算；**(3) 课程退火**——p_i = p_start + (t−1)/(T−1)·(p_end − p_start)，线性从 p_start（典型 0.5~1.0）退火到 p_end（典型 0）。关键设计：只修改训练输入分布，不改 RL 目标 / 损失 / 优化过程，故与任意 policy-gradient 算法兼容（GRPO、RLOO 皆可），推理时不注入、无额外开销（Algorithm 1）。

#### 6. 方法详解（通俗、分步骤）
每个训练步 t：
1. 取 mini-batch {q_i}；设当前注入概率 p←p_i。
2. 对每个 q_i：从 ℬ 采 k 条示范 E_i；掷 Bernoulli(p) 得 b_i；按 b_i 决定是否把 E_i 前置组成输入 x_i（带示范的样本把示范当作前面的 user-assistant 对话轮）。
3. 用 π_θ 对 {x_i} rollout，得经验 𝒟_t；做策略更新（GRPO 或 RLOO）。
4. 推理时 p=0，不注入。

实现细节（Appendix）：
- Reasoning Gym：bank 每任务 20 题，程序化求解得 ground-truth，GPT-5.2 生成 step-by-step 推理轨迹；注入时均匀随机采样，k=2，p_start=0.5→p_end=0。
- Q 编程：bank 取 50 条验证过的代码示例（只有代码、无推理标注）；注入时按 tag（Array / Dynamic Programming 等）过滤后再采，k=2。
- 训练超参（Reasoning Gym）：GRPO/RLOO，500 步，lr 1e-6，batch 32，4×A6000，FSDP，温度 1.0；format reward +0.2、答案 +1.0。Q 编程：QQwen-7B-Pretrain，batch 64、group 8，按通过测例比例给分 + 全过 +2 bonus，4×GH200。

#### 7. 实验数据集
- **Reasoning Gym** 5 个程序化生成任务（可控 seed、无限数据）：ARC-1D、Manipulate Matrix、Spell Backward、Word Sorting、Puzzle-24。
- **Q Programming**（Hogan 2025，morganstanley/sft-python-q-problems）：时序数据库 DSL，右到左求值、隐式类型、简洁数组语法，偏离主流语言、预训练语料少见；542 训练 / 136 测试。
- **骨干**：Reasoning Gym 用 Qwen2.5-3B-Instruct 与 Llama-3.2-3B-Instruct；Q 用 QQwen-7B-Pretrain。
- 评测：Reasoning Gym 程序化 verifier、100 题/环境、3 次平均；Q 用 5 个单测/题、5 次平均。

#### 8. 实验结果与主要发现
- **主结果（Table 1）**：在全部 10 个 model–environment 对上均超 GRPO-only 基线，增益 +1.3%（Spell Backward, Llama）~ +22.3%（Word Sorting, Qwen）。Qwen2.5-3B 最大增益在 Word Sorting（+22.34）与 Puzzle-24（+12.67）；Llama 在 ARC-1D（+8.0）与 Manipulate Matrix（+5.0）。
- **Q 编程（Table 2）**：test-pass 27.3%→43.0%，Pass@1 5.0%→26.3%。注：CBRL 的 Valid Q 率反而略低于 GRPO（80.9% vs 89.1%），但功能正确率大幅更高——说明 GRPO 主要学"写出合法 Q 语法"，CBRL 进一步学"用这些构造真正解题"。
- **算法无关（Table 3，RLOO）**：Word Sorting 20.3%→67.3%、Puzzle-24 23.0%→66.0%、Spell Backward 63.7%→89.7%（RLOO 下增益甚至大于 GRPO，可能因 RLOO 高方差梯度让早期更需引导）；但 ARC-1D（−2.3）与 Manipulate Matrix（−7.0）反而掉了。
- **训练动态（Figure 4）**：CBRL 早期 mean reward 显著更高（探索效率改善），且注入退火到 0 后性能不崩——支撑"内化而非依赖"。
- **注入概率消融（Figure 3）**：ARC-1D 上呈倒 V，p_i=0.5 峰值（0.31），p_i=0（0.26）与 p_i=1.0（0.23）都更低——过高压制自身探索、过低脚手架不足。

#### 9. 结果如何支撑其主张
- "缓解探索低效"由训练曲线早期 reward 更高 + 全 10 对增益直接支撑。
- "内化而非依赖"由退火到 0 后性能不崩支撑，逻辑成立。
- "算法无关"由 GRPO 与 RLOO 两套结果支撑——但 RLOO 下 ARC-1D/Manipulate Matrix 退化，说明该主张有条件（依赖示范与任务结构的匹配）。

#### 10. 逻辑自洽性（中性评估）
方法极简、主张与证据基本对齐。新意主要在"用 ICL 当临时脚手架 + 退火内化"的组合与系统验证，而非算法创新（本质 = 带退火课程的 few-shot prompt 注入）。"内化"是行为层证据（退火后不崩 + 定性例子显示 CBRL 模型模仿示范的 step-by-step ASCII 推理），未给参数/表征层的内化证明。RLOO 上两个任务的退化诚实地暴露了脚手架与任务结构需匹配。

#### 11. 残留问题 / 局限
- **评测面窄**：集中在 Reasoning Gym 合成任务 + 单一 DSL（Q），未在主流数学/代码大基准（AIME/MATH/LiveCodeBench）上验证，泛化与规模化证据有限。
- **增益区间极宽（+1.3%~+22.3%）**：效果强依赖任务，对已能较好探索的任务收益小。
- **示范来源/选择是开放问题**：固定 bank + 均匀或 tag 过滤采样；学习式检索、自适应注入调度（按 reward 趋势调 p_i）、长程/agentic 设置都是 future work。
- 代码为小型研究原型（公开仓、规模小、commit 少）。

#### 12. 开源代码与框架（链接 + 框架 + 代码可得性）
- 仓库：https://github.com/context-bootstrapped-rl/cbrl （项目页确有指向此仓的 Code 链接，非 404；已 clone 约 1.3MB）。含 `cbrl/`（trainers、utils）、`config/`（Hydra）、`reasoning_gym/`、`evaluations/`、`data_preprocess/`、`q_reward.py`、`scripts/`。
- 框架：verl 0.3.0.post2 + vLLM 0.8.5 + PyTorch 2.6.0（pyproject 声明 torch==2.6.0、vllm==0.8.5、verl），Hydra 配置，Flash-Attention，自带 `cbrl/trainers`。代码可得性：公开可用，属研究原型，复现脚本与超参（Appendix A/B）齐全。


---


### cepo — CEPO: RLVR Self-Distillation using Contrastive Evidence Policy Optimization

> **一句话重点 (TL;DR)**：GRPO 给一条轨迹里所有 token 同一个优势，浪费在 filler 上、低估决定性步。CEPO 在每个 token 问"正确答案偏好它**且**错误答案反对它吗？"——用对比比率 P⁺_T(y_t)/P⁻_T(y_t)（正确/错误答案两个 teacher，错误答案取自组内已有的 rejected rollout）在 stop-gradient 下调制 GRPO 优势幅度，符号仍由 verifier 锚定；由于学生先验 P_S 被消掉，从构造上消除了 RLSD 的 fluency confound，在决定性 token 处锐化、filler 处恰好失效。

**元信息**：arXiv 2605.19436（v1，2026-05-19，cs.LG）｜ MBZUAI + Linköping University + Australian National University（Ahmed Heakl、Abdelrahman M. Shaker、Youssef Mohamed、Rania Elbadry、Omar Fetouh、Fahad Shahbaz Khan、Salman Khan）｜ Preprint，2026-05 ｜ 主题：RLVR self-distillation 的 token 级 credit assignment（多模态数学推理；与本项目 token 级监督/path-selection 相关，但偏 RLVR 而非纯 logit-OPD）｜ 代码 https://github.com/ahmedheakl/CEPO （已 clone，约 16MB）｜ 框架 veRL（经由 EasyR1）+ FSDP + vLLM

#### 1. 相关工作与进展
- **RLVR / credit assignment**：RLVR 采样 rollout、verifier 打分、更新策略；GRPO 去掉 value 网络但赋予整条轨迹**均匀**的序列级优势。token 级方法要么靠 Monte Carlo 重模拟（VinePPO、SPO），要么训练独立的 PRM——都需昂贵重采样或辅助网络。
- **带特权信息的 on-policy 自蒸馏**：用正确答案 r⁺ 当 teacher 产生 dense token 级信号，无需辅助网络。OPSD（Zhao 2026）最小化 P⁺_T 与 student 的 per-token KL；SDPO（Hübotter 2026）扩成 JSD + EMA teacher；HDPO 专门用于全错的 prompt。
- **RLSD**（Yang 2026，CEPO 的直接前身）：在 stop-gradient 下只在采样 token 上算 evidence ratio P⁺_T(y_t)/P_S(y_t)，仅用于调制 GRPO 优势幅度、符号锚定 verifier——做到既用特权信息又"leakage-free"（梯度里无 vocab-wide r-条件求和）。
- Table 1 用 Priv./Leak-free/Contr./No-Aux 四维定位 CEPO 是唯一四项全 ✓ 的方法。

#### 2. 现有工作存在的问题
- **GRPO 太钝**：正确轨迹每 token 同正优势、错误轨迹每 token 同负信号；数学推理里单步算错或单个正确推断即可决定整条 CoT 成败，均匀 credit 把梯度浪费在 filler（连接词、格式、样板）上、低估少数决定性 token。
- **"用 r⁺ 当 teacher 做分布匹配"会泄漏**：RLSD 证明任何把 P⁺_T 当分布目标的 divergence 目标，其梯度都含 vocab-wide 的 r-条件求和（式 3），方差 ∝ I(Y_t;R⁺|X)，训练后期主导，逼模型编码 x→r⁺ 的伪相关（information leakage，与实现无关、不可约）。
- **RLSD 虽解决泄漏，但信号质量有三缺陷**：(1) **fluency confound**——分母 P_S 反映 base-rate 流畅度而非语义相关，常见 token 无论 r⁺ 多偏好它都被压低；(2) **asymmetric negative**——对错误轨迹的惩罚缺乏对 r⁻ 的显式 grounding；(3) **one-sided evidence**——P⁺_T/P_S 无法区分"正反答案都同等支持的 filler"与"r⁺ 支持 / r⁻ 反对的决定性步"，两者 ratio 相同时被同等加权。

#### 3. Motivation
在每个 token 问更尖锐的问题：不仅"正确答案是否偏好该 token"，而是"正确答案偏好**且**错误答案反对吗"。两者皆满足→真正推理步；皆不满足→filler。且 wrong-answer teacher 可从 batch 中已有的 rejected rollout 构造，无额外采样开销。

#### 4. 主要灵感 / 核心直觉
- 把单参照（只看 r⁺）升级为对比双参照（r⁺ vs r⁻）。用对比比率 P⁺_T/P⁻_T 后，**学生先验 P_S 在 stop-gradient 下完全抵消**，fluency confound 从构造上消失。
- 对比证据 delta 有清晰的贝叶斯解释（应用 RLSD Theorem 4 于两个 teacher 相减）：∆^CE_t = [r⁺ 的信念更新] − [r⁻ 的信念更新]，即"token y_t 同时把 r⁺ 后验抬高、把 r⁻ 后验压低多少"。决定性步得大正值，filler 趋近 0。
- filler 处改进恰好消失不是缺陷而是**正确性准则**——在信息中性位置放大梯度只会引入噪声。

#### 5. 主要解决思路（一段话讲清核心）
CEPO = Contrastive Evidence Policy Optimization：定义三个共享参数 θ 但条件不同的分布——student P_S(y_t)=π_θ(y_t|x,y<t)、正确 teacher P⁺_T（条件 r⁺）、错误 teacher P⁻_T（条件 r⁻）。r⁻ 取组内最低 reward 的 rejected rollout 的最终答案（无额外采样）。对比证据 delta ∆^CE_t = sg(log P⁺_T(y_t) − log P⁻_T(y_t))；对比权重 w^CE_t = exp(sign(A)·∆^CE_t) = (P⁺_T/P⁻_T)^{sign(A)}；token 级优势 Â_t = A·[(1−λ) + λ·clip(w^CE_t, 1−ε_w, 1+ε_w)]，λ 从 λ₀ 线性退火到 0。把 Â_t 代入标准 PPO clipped surrogate 更新。G⁻=∅ 时令 P⁻_T=P_S，精确退回 RLSD。

#### 6. 方法详解（通俗、分步骤）
**算法 1**：每次迭代、每个 (x,r⁺)：(1) 采 G 条 rollout，按式(1) 算组归一化序列优势 A，分成正确组 G⁺/错误组 G⁻；(2) r⁻ ← argmin_{j∈G⁻} R 的最终答案，G⁻=∅ 则 P⁻_T←P_S；(3) 对每条轨迹每个位置 t：∆_t ← sg(log P⁺_T(y_t) − log P⁻_T(y_t))，Â_t ← A·[(1−λ)+λ·clip(e^{sign(A)∆_t}, 1−ε_w, 1+ε_w)]；(4) 用 Â_t 做 PPO clipped surrogate 更新。

**理论保证（Theorem 1，证明见 Appendix A）**：(i) 方向锚定 sign(Â_t)=sign(A)（特权信息不能翻转任何 token 更新方向）；(ii) leakage-free 梯度（无 vocab-wide r-条件求和，r⁺/r⁻ 只作为采样 token 处的 stop-gradient 标量进入）；(iii) RLSD 包含性（P⁻_T=P_S 时精确退回 RLSD）。
**Proposition 1（区分锐度）**：正确轨迹下 w^CE_t > w^RLSD_t 当且仅当 P⁻_T(y_t) < P_S(y_t)（即错误答案相对学生先验更不喜欢该 token，恰是决定性位置）；错误轨迹对称；filler 处三者皆近 1、改进消失。

**成本**：CEPO 比 RLSD 多一次 teacher forward（r⁺/r⁻ 各一次），即比 GRPO 多两次 forward；无额外采样。Geo3k 50 步 wall-clock：GRPO 5h58m、SDPO 6h14m、RLSD 6h15m、CEPO 6h34m（多约 36 分钟）。

#### 7. 实验数据集
- **训练**：Geo3K（3,000 道带可验证数值答案的几何题）。
- **评测**：5 个 held-out 多模态数学推理基准——DynaMath、LogicVista、MathVision-mini、MMMU、WeMath（不含 MathVista）。
- **模型**：Qwen3-VL-2B-Instruct、Qwen3-VL-4B-Instruct，LoRA（rank 16 / α 32）微调，50 步。
- **关键超参（Table 6/7，以论文为准）**：AdamW，lr 1e-6（**CEPO 用 5e-6**），cosine decay + 5 步 warmup，batch 32 prompts，group G=8，温度 1.0，max len 2048，PPO clip 0.20/0.28，无 KL/熵正则，rule-based 数值 verifier；CEPO 默认 λ₀=0.5、25 步线性退火到 0、ε_w=0.5。评测用 lmms-eval（温度 1.0、top-p 1.0、top-k 40、presence penalty 2.0、max 32k token）。
- 硬件：NVIDIA RTX6000 Pro Blackwell 100GB。
- 〔已核〕仓库脚本 `geo/cepo.sh`（2B）历史上有 lr 5e-6、cepo_lambda_init=0.5、warmup 25、eps_w 等设置，与论文超参基本一致（个别脚本数值可能与论文表略有出入，以论文 Table 6/7 为准）。

#### 8. 实验结果与主要发现
- **主结果（Table 2）**：2B 五基准平均 CEPO 43.43% vs GRPO 41.17%（+2.26pp）、base 39.73%（+3.70pp）、OPSD 34.96%、SDPO 35.70%；4B CEPO 60.56% vs GRPO 57.43%（+3.13pp）、base 58.36%（+2.20pp）、OPSD 56.23%。增益最大在 LogicVista（4B 上 +6.18 over GRPO）与 MathVision-mini（2B 上 +4.94），最小在 MMMU（短链知识检索，2B 上 +1.67）。
- **OPSD/SDPO 跌破未训练 base**（2B 上 34.96/35.70 vs 39.73；4B 同样），实证印证 information leakage 理论预测，且不随规模消失。
- **消融**：teacher source（Table 4）actor-policy teacher 最优 43.43%（fixed reference 42.18、每 25 步同步 42.74）——teacher 新鲜度/on-policy 对齐比拉大师生分布差更重要，且 actor 共享权重省显存。feedback source（Table 5）"ground-truth r⁺ + peer answer-only r⁻"最优 43.43%；用整条 peer rollout 当负参照次之（42.74）；只 prefix/suffix 截断反而 < GRPO。超参（Figure 3）λ=0.5 与 25 步退火都 > GRPO，λ=1.0 反而引入噪声变差；ε_w 在 [0.4,0.5] 峰值，过小退回 GRPO、过大失稳。
- **机理分析**：训练中正 delta 占比上升、负 delta 占比下降（Figure 4）；token 权重热图（Figure 5）显示 CEPO 把 credit 锐化到关键代数推导与最终答案、把 filler 压到近 1，clip 率更低（49.5% vs RLSD 71.3%，即更宽的有效动态范围）。

#### 9. 结果如何支撑其主张
- "CEPO > GRPO/RLSD" 由 Table 2 在两个规模、五基准上的一致增益支撑。
- "结构安全是实践前提" 由 OPSD/SDPO 跌破 base 这一反例强支撑——这也是论文最有说服力的实证。
- "在决定性 token 锐化、filler 处失效" 由 Prop.1 + token 热图 + delta 占比演化共同支撑，理论-实证闭环较好。
- "对比比率 ≠ 对比 KL"：脚注证明 ∇[D_KL(P⁺‖P_S) − D_KL(P⁻‖P_S)] 会产生与 OPSD 同构的 vocab-wide leakage，凸显 CEPO 用 ratio + stop-gradient 的必要性。

#### 10. 逻辑自洽性（中性评估）
理论（三保证 + Prop.1）与实证（OPSD/SDPO 退化、热图、delta 演化）方向一致，自洽性强。但**增益绝对值偏小**（base→CEPO 仅 +2.2~3.7pp），且训练只跑 50 步、单一数据集（Geo3k）、小模型（2/4B）、LoRA——更像"低预算下的快速收敛优势"（Figure 1 显示 CEPO 早期更快、约 step 40 差距最大、最终部分收敛）。强主张（"结构安全是实践前提"）证据扎实，但泛化主张需更大规模验证。

#### 11. 残留问题 / 局限
- 仅在 Qwen3-VL 2/4B + Geo3k + 50 步 + LoRA 上验证；更大模型、纯文本推理、代码生成是 future work（作者自承）。
- 依赖组内同时有正确与错误 rollout 来构造 r⁻；全对/全错的 prompt 上对比信号退化（全错时退回 RLSD）。
- 增益小且部分收敛，长训是否保持优势未知。
- r⁻ 只取"最低 reward rollout 的最终答案"，对负参照的选择策略敏感（feedback source 消融显示 prefix/suffix 反而有害）。

#### 12. 开源代码与框架（链接 + 框架 + 代码可得性）
- 仓库：https://github.com/ahmedheakl/CEPO （已 clone，约 16MB）。含 `examples/`、`experiments/`、`scripts/`（含 `geo/cepo.sh` 等复现脚本）、`verl/`（内置 veRL）、`Dockerfile`、`requirements.txt`、`setup.py`。
- 框架：veRL，经由 **EasyR1**（Zheng 2025，基于 verl 的多模态 RLVR 框架）+ FSDP + vLLM 加速 rollout。代码可得性：训练 + 评测脚本齐全，依赖明确，可复现性较好（个别脚本超参与论文表略有出入，以论文为准）。


---


### chord — On-Policy RL Meets Off-Policy Experts: Harmonizing SFT and RL via Dynamic Weighting (CHORD)

> **一句话重点 (TL;DR)**：别再把 SFT 当 RL 前面的独立阶段（那会经历"漂移—再适应—过拟合"并破坏 on-policy 探索）。CHORD 把 SFT 重构成 on-policy RL 里一个**动态加权的辅助目标**：全局系数 μ（带 warmup 的余弦衰减）控制专家信号占比从"以模仿为主"平滑过渡到"以探索为主"；token 级权重 φ(p)=p(1−p) 对那些已很可能或极不可能的专家 token 下调学习信号，缓解熵坍缩与干扰。

**元信息**：arXiv 2508.11408（v3，2026-03-17；ICLR 2026 会议论文）｜ 阿里巴巴集团（Wenhao Zhang、Yuexiang Xie、Yuchang Sun、Yanxi Chen、Guoyin Wang、Yaliang Li(通讯)、Bolin Ding、Jingren Zhou）｜ ICLR 2026 ｜ 主题：统一 SFT 与 RL 的 off-policy/on-policy 视角，把 SFT 重构为 RL 中动态加权的辅助目标（GFT-class 代表作）｜ 代码 https://github.com/modelscope/Trinity-RFT （示例 `examples/mix_chord/`，损失 `trinity/algorithm/policy_loss_fn/chord_policy_loss.py`；另有 ms-swift 集成）｜ 框架 Trinity-RFT（基于 veRL 后端 + Ray）

#### 1. 相关工作与进展
- 现代后训练用两类数据：on-policy（模型自生成 rollout，RL 用）与 off-policy（专家/其他模型示范，SFT 用）。常规做法是 SFT→RL 两阶段。
- 把 off-policy 专家数据并入 on-policy RL 的现有思路：直接 dataset 混合（SimpleMix）、把专家轨迹混进 rollout 组（LUFFY、SRFT）、用专家数据引导生成（UFT、BREAD）、按预定/自适应调度交错 RL 与 SFT 步（SASR、Ma et al.）。SRFT 提出 sample-level SFT loss 的统一框架。
- 作者强调本文聚焦"已建立自身回答模式的 instruct 模型"，比微调 base 模型更难也更实用。

#### 2. 现有工作存在的问题
- 作者实测 SFT-then-RL 的学习曲线呈 **"shift–readapt–overfit"（漂移—再适应—过拟合）**：在 MATH-500 上先因突然的策略漂移掉点（exposure bias 加剧），再重新适应专家模式回升，最终过拟合到有限静态样本、丧失输出多样性与探索能力。
- SFT-then-RL 的 SFT→RL 转换时机高度任务相关（数学上 SFT-best+RL 最好、工具调用上 SFT-light+RL 更好），需大量调参，且两阶段分离本身就可能次优——Figure 1 显示 SFT-then-RL 不一定胜过纯 RL。
- 离线专家数据与当前 on-policy 分布有差异，直接大权重模仿会破坏探索、导致熵坍缩或被极端不可能 token 干扰。

#### 3. Motivation
从 off-policy vs on-policy 的统一视角，把 SFT 重构为 RL 过程中**一个动态加权的辅助目标**而非独立阶段。核心原则：稳定地融入 off-policy 专家信号需要对其学习信号做"下调权重"，尤其是对那些已经很可能（避免熵坍缩）或极不可能（避免干扰）的专家 token。

#### 4. 主要灵感 / 核心直觉
- SFT-then-RL 的二元开关（μ=1→0）太僵硬；改成可衰减的 μ schedule 可在过拟合前平滑退出专家影响（与 scheduled sampling 缓解 exposure bias 同理）。
- 但全局 μ 缺乏精度：CHORD-µ 会让模型整体照搬专家的冗长风格、覆盖自身简洁性（case study 证实）。所以需要 token 级 φ：把专家信号集中在"模型还不确定"（p≈0.5）的 token 上，对已确定（p→0 或 1）的 token 几乎不学——pt(1−pt) 恰是对"生成该 token 这一二元事件"的策略不确定性度量，制造"learning sweet spot"。

#### 5. 主要解决思路（一段话讲清核心）
CHORD = Controllable Harmonization of On- and Off-Policy RL via Dynamic Weighting。统一损失 **L = (1 − μ)·L_GRPO + μ·L_SFT**（代码 `MIXCHORDPolicyLossFn` 确认）。数据用 `expert_mask` 区分：非专家（on-policy）走 GRPO 损失（`PPOPolicyLossFn`），专家（off-policy 示范）走 SFT 损失。两层动态加权：(1) 全局 μ 随训练步用带 warmup 的余弦衰减从 mu_peak 降到 mu_valley；(2) token 级 φ(y*_t)=p_t(1−p_t)（p_t = π_θ(y*_t|x,y*<t)）这条抛物线在 p_t=0.5 取峰、p_t→0/1 衰减到 0，对已很可能或极不可能的专家 token 下调学习信号。论文给两个实例：**CHORD-µ**（只用全局 μ 调度、关闭 φ）与 **CHORD-φ**（启用 token 级 φ、μ 取较小常值）。

#### 6. 方法详解（通俗、分步骤）
**损失实现（代码确认）**：
- 全局 μ：`mu_schedule_function(step, mu_warmup_steps, mu_decay_steps, mu_peak, mu_valley)`——warmup 段线性从 0 升到 mu_peak，之后余弦衰减到 mu_valley。
- token 级 SFT：`SFTPhiLossFn` 实现 `-logprob · φ(p).detach()`（φ=p(1−p)，token-mean 聚合，可选 cutoff_prob 截断 logprob）；另提供 `SFTISLossFn`（重要性采样，权重 p.detach()，对应论文式 4 的 IS——把分母假设为 1）与可关闭 φ 的标准 `SFTLossFn`。
- 凸组合：`loss = (1-μ)·grpo_loss + μ·sft_loss`，且非专家/专家两部分各按 batch 内数量与 `train_batch_size_usual/expert` 归一化（`per_micro_batch_weight_*`）。

**为何要 φ 而非纯 IS**：Figure 5 显示无 IS 混入 off-policy 数据→熵暴涨（established pattern 被破坏）；但 IS 又会让熵急剧坍缩（过度强化高概率 token、忽视低概率新 token，过度自信）。φ=p(1−p) 同时下调两端，兼顾"不坍缩 + 不破坏"。

**两个实例的超参（代码 yaml 注释确认）**：CHORD-µ：mu_warmup=0、mu_decay=200、mu_peak≈0.9→mu_valley≈0.05、phi off；CHORD-φ：mu_warmup=0、mu_decay=0、mu_peak=mu_valley≈0.1（常值）、phi on。仓库 `mix_chord.yaml` 默认是 µ+φ 混合示例（warmup 200 / decay 400 / peak 0.5 / valley 0.02 / phi on），`expert_data_ratio=0.20`、`train_batch_size_expert=64`。作者明示不存在跨所有任务/数据/模型都最优的单一 φ，但 p(1−p) 是有效鲁棒的实例。

#### 7. 实验数据集
- **数学推理**：OpenR1-Math-220k，采样 5k 做 SFT、20k 做 RL（无重叠）；policy model 为 Qwen2.5-7B-Instruct（回答模式与专家 DeepSeek-R1 差异显著）；in-domain 评测 AIME24/AIME25/AMC，用 MMLU-Pro 监控通用推理变化。
- **工具调用**：ToolACE 单轮实例，5k RL / 500 SFT，专家为 DeepSeek-R1，评测 BFCL（Live/Non-live）；policy model 为 LLaMA3.2-3B-Instruct。
- 仓库示例默认 policy model 为 Qwen2.5-1.5B-Instruct（Figure 1 也在该模型 + Open-R1 上验证 SFT-then-RL 现象）。
- 基线：Original Model、SFT-light、SFT-best、SFT-light+RL、SFT-best+RL、SASR、GRPO(Pure RL)、LUFFY。

#### 8. 实验结果与主要发现
- **主表（Table 1）**：CHORD-µ 在数学上超强基线 SFT-best+RL（AMC +2.4、AIME24 +1.0、AIME25 +1.6），工具调用整体也更好；CHORD-φ 进一步在数学 + 工具调用上全面最优（如 MMLU-Pro 56.2 显著高于各基线、BFCL Overall 78.5）。
- **响应长度（Table 2）**：专家 DeepSeek-R1 远长于原模型（数学 6132 vs 659 token）；CHORD-µ 会被拉到专家级冗长（数学 6081），而 CHORD-φ 取得更细致平衡（数学 2444、工具调用 120）——token 级加权使模型按任务选择性吸收专家模式。
- **μ 消融（Figure 7）**：固定 μ 一律差于动态 μ，甚至可能不及纯 RL；小固定 μ（0.02）能减损但提升不显著。衰减 μ 才能平滑化解 on/off-policy 冲突。
- **φ 训练动态（Figure 8/9）**：CHORD-φ（固定 μ=0.1）既防熵过早坍缩、又避免熵暴涨，reward 稳定持续上升、显著优于纯 RL。用了 φ 后不再需要复杂 μ schedule（对 μ 选择鲁棒）。
- **进一步分析**：换专家源（DeepSeek-R1 vs 风格更接近的 Qwen2.5-72B）CHORD 均超基线，且偏模仿的方法（SFT+RL、CHORD-µ）在专家风格相近时增益更大；扩展到非可验证域（RaR-Medicine）CHORD 仍超纯 RL；弱模型（Qwen2.5-3B）上 naive 模仿/SFT+RL 不稳，CHORD-φ 更鲁棒。

#### 9. 结果如何支撑其主张
- "SFT-then-RL 次优"由 Figure 1（不胜纯 RL）+ Figure 2（shift-readapt-overfit 曲线）直接支撑。
- "动态 μ > 固定 μ/两阶段"由 Figure 7 + Table 1 支撑。
- "φ 缓解熵坍缩/干扰、选择性吸收"由 Figure 8/9 的熵-reward 曲线 + Table 2 的任务自适应长度 + 代码实现共同支撑，理论（p(1−p) 的不确定性解释）与实现一致。

#### 10. 逻辑自洽性（中性评估）
论文-代码高度一致（损失公式、μ schedule、φ、expert_mask 均在 `chord_policy_loss.py` 复现），自洽性强。诚实承认 φ 非唯一最优、配置随 setup 变（也试了 entropy-based/clipping/focal 变体）。核心贡献是"把 SFT 当 RL 动态加权辅助目标"这一统一视角 + 一个简单鲁棒实例，而非全新算法。

#### 11. 残留问题 / 局限
- μ、φ 的配置跨任务/数据/模型会变，仍需调参（自适应 reward-aware μ 可行但调参重）；无单一普适最优设计。
- 模式漂移分析主要是经验性的，缺乏对"不同 CoT 模式如何影响学习"的机理理论。
- 主实验规模有限（7B/3B/1.5B policy），更大模型与异构专家混合是 future work。
- 假设专家数据分母 IS 比为 1（把专家当 ground-truth 分布），是常见但近似的处理。

#### 12. 开源代码与框架（链接 + 框架 + 代码可得性）
- 仓库：https://github.com/modelscope/Trinity-RFT 。示例 `examples/mix_chord/`（`mix_chord.yaml` 数学、`mix_chord_toolace.yaml` 工具、`get_openr1_data.py`、`README.md`）；损失实现 `trinity/algorithm/policy_loss_fn/chord_policy_loss.py`（`MIXCHORDPolicyLossFn` + `SFTPhiLossFn`/`SFTISLossFn`/`SFTLossFn` + `mu_schedule_function`）。另有 ms-swift 集成。
- 框架：Trinity-RFT（modelscope，基于 veRL 后端 + Ray）。算法注册名 `mix_chord`（`algorithm_type: mix_chord`），policy loss 为 `MIXCHORDPolicyLossFn`，用 `expert_mask` + `expert_data_ratio`（示例 0.20）区分专家/非专家数据。代码可得性：损失实现 + 复现脚本 + 超参齐全，可复现性好。


---


### dapo — DAPO: An Open-Source LLM Reinforcement Learning System at Scale

> **一句话重点 (TL;DR)**：DAPO 把朴素 GRPO 拆解出 4 个关键改造（Clip-Higher、动态采样、token 级损失、超长奖励整形），并完整开源算法+数据+代码，在 Qwen2.5-32B base 上把 AIME 2024 从约 0 提到 **50 分**，以约一半训练步数超越 DeepSeek-R1-Zero-Qwen-32B（47 分）。

**元信息**：arXiv 2503.14476（v1 2025-03-17，v2 2025-05-20）｜ ByteDance Seed + 清华 AIR（SIA-Lab）+ 港大；通讯 Hao Zhou、Mingxuan Wang ｜ 主题 T3（RLVR / RL 算法系统），相关性 High ｜ 代码 https://github.com/BytedTsinghua-SIA/DAPO（recipe/eval/数据说明，训练逻辑在 veRL）｜ 框架 veRL（vllm==0.8.3、ray[serve]）

#### 1. 相关工作与进展
o1/R1 引领 test-time scaling，长 CoT + 大规模 RL 成为提升推理能力的核心技术路线。但 SOTA 推理模型（OpenAI o1、DeepSeek R1）的关键训练细节被隐藏，社区难以复现工业级结果。GRPO（源自 DeepSeekMath）成为主流的免 critic RLVR 算法基石。

#### 2. 现有工作存在的问题
作者在 Qwen2.5-32B base 上用朴素 GRPO 起步，AIME 2024 仅约 30 分（远低于 DeepSeek 报告的 47 分）。复现中暴露的关键问题：
- **熵坍缩（entropy collapse）**：策略熵迅速下降、探索受限、过早确定化。
- **reward noise** 与训练不稳定。
- **R1 论文省略了构建可复现大规模 RL 系统所需的工程细节**。

#### 3. Motivation
完全开源一套达到 SOTA 的大规模 LLM RL 系统（算法 + 代码 + 数据），把"让大规模长 CoT RL 成功"的关键技术公开，democratize 工业级 RL 结果。

#### 4. 主要灵感 / 核心直觉
熵坍缩的一个直接诱因是裁剪上界压制了"低概率探索 token"的提升空间；无效梯度（整组全对或全错的 prompt）浪费 batch；长序列在 sample-level 归一下贡献被稀释；超长截断样本引入奖励噪声。逐一对症即可在不加 KL 惩罚、仅用规则二值奖励的极简设定下稳定训练。

#### 5. 主要解决思路(一段话讲清核心)
在"去掉 KL 惩罚 + 规则化二值奖励（正确 +1 / 否则 −1）"的极简 GRPO 之上，叠加 4 个解耦改造来分别缓解熵坍缩、无效梯度、长序列梯度失衡和超长奖励噪声，组合成 **DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）**。

#### 6. 方法详解(通俗、分步骤)
1. **Clip-Higher**：解耦上下裁剪范围 ε_low / ε_high（取 **0.2 / 0.28**）。提高 ε_high 给低概率"探索"token 留出上行空间（实测被上裁剪的 token 概率 <0.2），缓解熵坍缩。
2. **Dynamic Sampling（动态采样）**：过采样并过滤掉准确率为 0 或 1 的 prompt（其组内优势全为 0、无梯度），保证每个 batch 都含有效梯度，稳定训练效率。
3. **Token-Level Policy Gradient Loss**：把 GRPO 的 sample-level 损失改为 token 级聚合（对所有 token 求和后按总 token 数归一），让长序列对梯度有更合理贡献，抑制长样本中的乱码/重复。
4. **Overlong Reward Shaping**：先做 Overlong Filtering（屏蔽超长截断样本的 loss），再用 Soft Overlong Punishment（在长度缓冲区内长度越长惩罚越大），降低 reward 噪声。

#### 7. 实验数据集
- 训练：**DAPO-Math-17K**（17K 数学题，答案转为整数便于规则解析；来源网络抓取 + 竞赛官网 + 人工标注）。
- 评测：AIME 2024（重复评测 32 次，报告 **avg@32**）。

#### 8. 实验结果与主要发现
- 基座 Qwen2.5-32B base，基线为朴素 GRPO + group reward normalization。
- 优化器 AdamW，常数 lr 1e-6，20 rollout step 线性 warm-up。
- Rollout：prompt batch 512，每 prompt 采 16 个响应；训练 mini-batch 512（每 rollout 做 16 次梯度更新）。
- 长度：期望最大 16384 token + 4096 soft punish 缓冲 → 最大生成 20480 token。
- 评测推理：temperature 1.0、top-p 0.7。
- 结果：AIME 2024 从约 0% → **50 分**，以约 50% 训练步数超越 DeepSeek-R1-Zero-Qwen-32B（47 分）。论文 Table 1 给出 4 项技术逐项消融的累积贡献。

#### 9. 结果如何支撑其主张
逐项消融表（Table 1）显示每项技术对 AIME avg@32 的边际贡献，支撑"这 4 项改造是大规模长 CoT RL 成功关键"的主张；熵曲线对比（w/ vs w/o Clip-Higher）支撑熵坍缩缓解；以更少步数超越同基座 R1-Zero 支撑系统级有效性与可复现性。

#### 10. 逻辑自洽性(中性评估)
论证链条自洽：每个问题（熵坍缩/无效梯度/长序列/超长噪声）对应一个改造并有消融支撑。需注意主结果集中在单一基座（Qwen2.5-32B）与单一评测（AIME 2024），普适性主要靠社区后续复现而非论文内多任务验证。

#### 11. 残留问题 / 局限
- 主评测窄（AIME 2024 + Qwen2.5-32B base），跨基座/跨任务的稳健性论文内验证有限。
- ε_high=0.28 等超参为经验取值，缺少敏感性分析。
- 去 KL + 二值奖励的极简设定在更长/更难任务上的可扩展性需外部验证。
- 训练逻辑不在 DAPO 仓内，需配合 veRL 才能复现。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库：https://github.com/BytedTsinghua-SIA/DAPO （本地已 clone，~4.1MB；含 recipe/eval/数据说明，**训练逻辑实现在 veRL**：https://github.com/volcengine/verl）。
- 框架：veRL（Volcano Engine RL），requirements 含 vllm==0.8.3、ray[serve]；DAPO 作为 verl 的一个 recipe。
- 数据/权重：DAPO-Math-17K（HF: BytedTsinghua-SIA/DAPO-Math-17k）；DAPO-Qwen-32B。代码可得性高（开源系统）。


---


### deepseek_r1 — DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

> **一句话重点 (TL;DR)**：R1-Zero 证明仅靠纯 GRPO + 规则奖励（不经 SFT）即可在 base 模型上自演化出推理能力（含"aha moment"），R1 再用"cold-start SFT → 推理 RL → 拒绝采样 SFT → 全域 RL"四阶段管线修好可读性与通用性；并把 800k 数据离线 SFT 蒸馏到小模型，得出"大模型 RL→小模型蒸馏 优于 小模型直接 RL"。

**元信息**：arXiv 2501.12948（v2 2026-01-04）｜ DeepSeek-AI ｜ 2025-01 首发 ｜ 主题 T1/T3（High）：纯 RL 激励推理（GRPO）、多阶段 SFT-RL 管线、SFT-on-trajectory 离线蒸馏（可作 OPD 对照基线）｜ 代码 https://github.com/deepseek-ai/DeepSeek-R1（**模型/权重发布仓，不含训练代码**）｜ 框架 自研高性能 RL 框架（附录 B.1）

#### 1. 相关工作与进展
此前 LLM 推理严重依赖人工标注 CoT 轨迹 + SFT；test-time scaling（o1 等）显示拉长推理可显著提升能力。GRPO（源自 DeepSeekMath）提供免 critic 的高效 RL。R1 在 DeepSeek-V3-Base（660B 级 MoE）上把"用最少人工标注、靠 RL 自演化激励推理"推到极致。

#### 2. 现有工作存在的问题
- 依赖人工推理轨迹 → 扩展性差、引入认知偏差、性能被人类示例上限封顶，难探索"非人类式"更优推理路径。
- 传统 SFT-before-RL 范式可能限制模型探索。
- 纯 RL 得到的 R1-Zero 虽强，但**可读性差、中英混杂（language mixing）**，且窄域 RL 导致写作/开放问答等通用能力弱。

#### 3. Motivation
用最小人工标注、通过 RL 自演化激励推理；假设人类定义的推理模式会限制探索，无约束 RL 更能激发新能力。故 R1-Zero 跳过 SFT 直接在 base 上 RL，奖励仅基于最终答案正确性（可验证任务），不约束推理过程，以观察自然涌现的自验证、反思、动态策略调整。R1 再用少量 cold-start 数据 + 多阶段管线解决可读性与通用性。

#### 4. 主要灵感 / 核心直觉
若只对"答案对不对"给奖励、不规定"怎么想"，模型会在 RL 压力下自行延长并重组推理过程（响应长度自然增长），涌现反思/自检等行为；而可读性/通用性这类"非可验证"问题，再用少量人对齐数据与偏好奖励模型补齐即可。

#### 5. 主要解决思路(一段话讲清核心)
两条线：R1-Zero = base 上纯 GRPO + 规则奖励（准确率 + 格式），验证 RL 可单独激励推理；R1 = 在其上叠加四阶段管线（cold-start SFT → 推理向 RL（加语言一致性奖励）→ 拒绝采样扩充 SFT → 混合推理/通用数据的全域 RL），兼顾推理强度、可读性与通用对话能力。

#### 6. 方法详解(通俗、分步骤)
- **算法 GRPO**：每问题从旧策略采一组（R1-Zero 用 16 个）输出，用组内 reward 均值/标准差归一得 advantage，无价值网络；带 clip 与 KL 惩罚。
- **R1-Zero**：DeepSeek-V3-Base 纯 GRPO。奖励 = 准确率奖励（数学 box 答案规则校验 / 代码用编译器+测试用例）+ 格式奖励（强制 `<think>…</think><answer>…</answer>`），等权相加。**刻意不用神经/过程奖励模型**（避免 reward hacking、降复杂度）。AIME2024 pass@1 由 15.6% → 77.9%。
- **R1 四阶段管线（图2）**：
  1. **Cold Start**：收集数千条对话式、人类对齐的长 CoT 数据做 SFT。
  2. **推理向 RL**：规则奖励 + **语言一致性（LC）奖励**（缓解语言混杂），得 Dev-2。
  3. **拒绝采样 + SFT**：从第一阶段 RL checkpoint 拒绝采样生成推理轨迹，用 DeepSeek-V3 作生成式裁判扩充，过滤混语/长段/代码块 → 约 600k 推理样本；另用 V3 管线生成约 200k 非推理样本（写作/事实QA/翻译/软工）；共约 **800k** 监督数据做 SFT，得 Dev-3。
  4. **全域 RL**：混合推理与通用数据，推理用规则奖励、通用用 helpful/harmless 偏好奖励 → 最终 R1。AlpacaEval2.0 +25%、ArenaHard +17%。
- **奖励模型（3.1）**：通用数据用偏好奖励模型（helpful 只评最终 summary，harmless 评整段）；推理任务坚持规则奖励。
- **蒸馏（附录 F / B.4.3）**：把 800k 数据直接 SFT 到更小 base 模型（**离线/off-policy 序列级蒸馏，非 on-policy KD**），得 R1-Distill-Qwen-{1.5B,7B,14B,32B}、R1-Distill-Llama-{8B,70B}。结论：蒸馏后小模型超过其原 instruct 版，且"大模型 RL→小模型蒸馏"优于"小模型直接 RL"。

#### 7. 实验数据集
- 训练：数学/代码/STEM/逻辑等可验证推理 prompt；cold-start 数千条；800k SFT（600k 推理 + 200k 非推理）。
- 评测：AIME 2024、MATH-500、CodeForces、LiveCodeBench、GPQA Diamond、MMLU、AlpacaEval 2.0、ArenaHard、Aider-Polyglot 等。

#### 8. 实验结果与主要发现
- R1-Zero 超参：lr 3e-6，KL 系数 0.001，rollout 温度 1；每题采 16 个输出，最大长度 32768（8.2k 步后增至 65536）；每步 32 题、batch 512；每 400 步更新参考模型；共 10,400 步（约 1.6 epoch）。
- SFT 超参（cold-start 与第二阶段）：2–3 epoch，cosine lr 5e-5→5e-6，ctx 32768，batch 128。
- 蒸馏超参：对应 base 微调 2–3 epoch，cosine lr 衰减到初值 1/10（各模型初值见 Table 6），ctx 32768，batch 64。
- 训练成本：R1-Zero 用 64×8 H800 约 198h；R1 约 80h（4 天）；SFT 数据 5K GPU 时；总计约 147K H800 GPU 时（约 $294K）。
- 关键发现：响应长度随训练自然增长、涌现"aha moment"；helpful 奖励模型出现 reward hacking（reward 升而 CodeForces 性能降）；LC 奖励消融显示其稳定语言一致性，代价是代码基准略降。

#### 9. 结果如何支撑其主张
R1-Zero 在不经 SFT 下 AIME pass@1 大幅提升 + 响应长度/反思行为自然增长，支撑"RL 可单独激励推理"；R1 四阶段后通用基准（AlpacaEval/ArenaHard）大涨支撑"管线修好可读性/通用性"；蒸馏小模型超越其 instruct 版、且优于小模型直接 RL，支撑蒸馏结论。

#### 10. 逻辑自洽性(中性评估)
主张与证据基本对应。但"无约束 RL 优于人类示例"主要靠 R1-Zero 单点对照支撑；蒸馏"优于直接 RL"的对照规模有限（小模型直接 RL 的算力预算与 R1 不对等）。"aha moment"为观察性叙述，缺乏严格涌现度量。

#### 11. 残留问题 / 局限
- R1-Zero 可读性/语言混杂问题需 R1 管线人对齐数据补救，纯 RL 并非"零人工"。
- helpful 奖励模型的 reward hacking 暴露偏好奖励在可验证任务上的风险。
- **训练代码与 RL 框架未开源**（仅模型/权重），GRPO 实现、四阶段管线、拒绝采样脚本均不可得，复现依赖论文公式与超参表。
- 蒸馏为离线序列级，与本项目关注的 on-policy / 在线蒸馏不同。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库：https://github.com/deepseek-ai/DeepSeek-R1 ；权重 https://huggingface.co/deepseek-ai 。
- **该仓为模型发布/权重仓，不含训练代码**（无 GRPO 训练脚本、无四阶段管线实现）。CloneTier=B，仅记录，未 clone。
- 框架：训练基础设施在附录 B.1 描述为自研高性能 RL 框架（不在仓内）。GRPO 算法后被 veRL/TRL/OpenRLHF/ms-swift 等广泛实现。


---


### deepseekmath — DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models（GRPO 起源）

> **一句话重点 (TL;DR)**：DeepSeekMath 首次提出 **GRPO**——用"同题多输出的组内相对奖励"替代 PPO 的价值网络，省去与策略同规模的 critic、大幅降显存算力，并给出统一梯度范式分析 SFT/RFT/DPO/PPO/GRPO 的异同；DeepSeekMath-RL 7B 在 GSM8K=88.2%、MATH=51.7%。

**元信息**：arXiv 2402.03300 ｜ DeepSeek-AI（Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Daya Guo 等）｜ 2024-02 ｜ 主题 T3（RL 算法）：GRPO 是 DeepSeek-R1 及绝大多数 RLVR 工作的算法基石，与本项目 RL/post-training 直接相关 ｜ 代码 https://github.com/deepseek-ai/DeepSeek-Math（**评测/推理脚本 + 模型发布，GRPO 训练代码不在仓内**）｜ 框架 custom/none（DeepSeek 内部实现）

#### 1. 相关工作与进展
RL 已被证明能在 SFT 之后进一步提升 LLM 数学推理。该阶段广泛使用 PPO（actor-critic）。GRPO 提出后被 veRL/TRL/OpenRLHF/ms-swift 等框架广泛实现，成为后续 R1、DAPO 等工作的算法基石。本文在大规模数学语料预训练 + 指令微调（DeepSeekMath-Base/Instruct 7B）基础上引入新的高效 RL 算法。

#### 2. 现有工作存在的问题
- **PPO 价值模型开销大**：需训练一个与策略模型规模相当的 critic，带来巨大显存与算力负担。
- **稀疏奖励下 critic 难训**：LLM 场景通常只在最后一个 token 由奖励模型打分，使训练逐 token 精确的价值函数变得困难。
- 缺乏对各类后训练方法（RFT、DPO、PPO、GRPO 等）的**统一理论理解框架**。

#### 3. Motivation
- 用"同一问题采样多个输出的平均奖励"作为 baseline 替代价值模型——这与奖励模型本质上做的"同题多输出相对比较"天然契合，从而省去价值函数、大幅降低训练资源。
- 提供统一范式来分析在线/离线、结果监督 vs 过程监督、单轮 vs 迭代 RL，理解 RL 为何有效并据此设计更优 RL。

#### 4. 主要灵感 / 核心直觉
既然奖励模型衡量的是"同题不同答案孰优孰劣"，那么用组内多个采样的平均/标准化奖励就能充当无偏的 advantage baseline，根本不需要再训一个独立的价值网络。

#### 5. 主要解决思路(一段话讲清核心)
对每个问题从旧策略采一组 G 个输出，用组内 reward 的均值/标准差归一化得到 advantage（无 critic），KL 正则直接加在损失里（而非奖励里），即为 GRPO；并把它与其它后训练方法纳入同一梯度形式，差异落在"数据来源、奖励函数、梯度系数"三点上。

#### 6. 方法详解(通俗、分步骤)
- **从 PPO 到 GRPO**：PPO 目标（式1）逐 token 用重要性比 πθ/πθold 加 clip，优势 A_t 由 GAE + 价值函数 V_ψ 估计，并在奖励里加逐 token KL（式2）。
- **GRPO 目标（式3）**：对每问题 q 采一组 {o_1,…,o_G}：
  J_GRPO = E[ (1/G)Σ_i (1/|o_i|)Σ_t { min( (πθ/πθold)·Â_{i,t}, clip(πθ/πθold,1−ε,1+ε)·Â_{i,t} ) − β·D_KL(πθ‖π_ref) } ]
  - **去掉价值模型**，用组内相对奖励估计优势 Â_{i,t}。
  - KL 正则**直接加在损失里**，用无偏估计器（式4，Schulman 2020）：D_KL = π_ref/πθ − log(π_ref/πθ) − 1，保证非负，避免复杂化优势计算。
- **两种优势估计**：
  - **结果监督（§4.1.2）**：奖励模型对每个完整输出打分得 {r_i}，组内标准化 r̃_i=(r_i−mean)/std，输出内所有 token 优势 Â_{i,t}=r̃_i。
  - **过程监督（§4.1.3）**：过程奖励模型对每个推理步末 token 打分，组内标准化后，每 token 优势为其后续各步标准化奖励之和 Â_{i,t}=Σ_{index(j)≥t} r̃_i^{index(j)}。
- **迭代 RL（§4.1.4 / Algorithm 1）**：随训练推进旧奖励模型不足以监督新策略，故用策略采样结果构造奖励模型新训练集，用回放机制（含 10% 历史数据）持续训练奖励模型；同时把参考模型设为当前策略，用新奖励模型继续训练策略。
- **统一范式（§5.2.1）**：把 SFT/RFT/DPO/PPO/GRPO 统一写成同一梯度形式，差异在于 (1) 数据来源（在线采样 vs 离线）、(2) 奖励函数（Rule vs Model）、(3) **梯度系数 GC**（由数据 + 奖励信号决定每个样本/token 的梯度权重）。

#### 7. 实验数据集
- RL 训练数据：来自 SFT 数据中 GSM8K、MATH 相关的 CoT 格式题，约 **144K** 题（故意排除其它 SFT 题以观察 RL 对缺数据基准的影响）。
- 评测基准：GSM8K、MATH（in-domain CoT）；MGSM-zh、CMATH（中文，out-of-domain）；以及工具集成推理（Tool-Integrated Reasoning）设置。

#### 8. 实验结果与主要发现
- 在 **DeepSeekMath-Instruct 7B** 上做 GRPO RL。
- 奖励模型：基于 DeepSeekMath-Base 7B 训练，lr 2e-5（按 Wang et al. 2023b 构造数据）。
- GRPO 超参：策略 lr 1e-6；KL 系数 β=0.04；每题采 **G=64** 个输出；最大长度 1024；训练 batch size 1024；每个探索阶段后策略仅更新一次。
- 结果：DeepSeekMath-RL 7B GSM8K=88.2%、MATH=51.7%（CoT），超过 7B–70B 全部开源模型及多数闭源模型；仅用 GSM8K/MATH 的 CoT 指令数据训练，却在**所有**基准（含 out-of-domain）上超越 DeepSeekMath-Instruct 7B。
- 讨论（§5.2）：在线采样优于离线；统一范式下不同方法核心区别在梯度系数与数据/奖励来源。

#### 9. 结果如何支撑其主张
"仅训 GSM8K/MATH 却全基准（含 OOD）提升"支撑 RL 的泛化有效性；GRPO 与 PPO 相比省 critic 仍达 SOTA，支撑"组内相对奖励可替代价值网络"；统一范式 + GRPO+PS 优于其它方法的对比，支撑"梯度系数差异解释方法优劣"的分析框架。

#### 10. 逻辑自洽性(中性评估)
GRPO 的动机（critic 开销 + 稀疏奖励难训）与解法（组内 baseline）对应清晰，统一范式提供了较有解释力的分析视角。局限是消融主要在数学单域、7B 单规模；"在线优于离线""GRPO+PS 更优"等结论的可推广性需后续工作验证（事实上社区后续如 DAPO 也指出朴素 GRPO 的熵坍缩等问题）。

#### 11. 残留问题 / 局限
- GRPO 在更大规模/更长 CoT 下暴露熵坍缩等问题（后续 DAPO/Dr.GRPO 等修补）。
- 实验集中在数学单域、7B 单规模，跨域跨规模稳健性论文内未充分覆盖。
- 过程监督与迭代 RL 依赖额外的过程/迭代奖励模型，工程成本与稳定性未深入分析。
- **GRPO 训练代码未开源**，算法仅以论文公式给出，复现依赖第三方框架实现。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库：https://github.com/deepseek-ai/DeepSeek-Math ；模型权重 DeepSeekMath-Base/Instruct/RL 7B（HuggingFace）。
- **仓库主要为评测/推理脚本与模型发布，GRPO 训练代码并不在该仓库**（实际 RL 训练为 DeepSeek 内部代码）。框架 custom/none。
- GRPO 算法本身后被 veRL/TRL/OpenRLHF/ms-swift 等框架广泛实现。


---


### denoiserl — DenoiseRL: Bootstrapping Reasoning Models to Recover from Noisy Prefixes

> **一句话重点 (TL;DR)**：把弱模型生成的错误推理前缀当作"结构化噪声"注入策略的 rollout，用 RL 训练策略从错误中间状态"去噪并恢复"到正确答案，从而在不引入更强教师、不构造难数据的前提下把"自纠错"从涌现行为变成显式训练目标。

**元信息**：arXiv 2605.28421 (v1, 2026-05-27) ｜ 复旦大学 / 上海创智学院 (Shanghai Innovation Institute)；Caijun Xu, Changyi Xiao, Zhongyuan Peng, Yixin Cao ｜ 2026-05 预印本 ｜ 主题 path-recovery / prefix 注入 / 负样本利用（与本项目高相关）｜ 代码 https://github.com/ALEX-nlp/DenoiseRL（已克隆，VeRL fork + recipe/denoise，含 1.7B/4B/8B 的 GRPO 与 DAPO 启动脚本）｜ 框架 VeRL，RL backbone 用 GRPO / DAPO。

#### 1. 相关工作与进展
- **On-policy RL 引导推理**：GRPO、DAPO 等结果/过程驱动 RL 已取代 SFT 成为扩展推理能力的主范式。
- **Weak-to-Strong Generalization (W2SG)**：用弱模型监督更强学生，以期突破能力瓶颈。
- **Prefix 条件 / off-policy 探索**：LUFFY 把 off-policy 推理轨迹混入 on-policy RL；PrefixRL 在**成功**的 off-policy 前缀上条件化并优化其后续续写；更广义地，专家解、oracle 提示、成功轨迹等被用来让稀疏奖励问题更可达。

#### 2. 现有工作存在的问题
- **W2SG 受弱教师上限约束**：学生被优化去模仿伪标签，天花板被弱监督者的噪声与有限能力卡住。
- **难数据构造成本高**：难题合成、对抗样本、长轨迹构造依赖复杂流水线、过滤验证与大量人工。
- **标准 on-policy RL 的探索瓶颈**：策略受限于自生成状态分布；一旦饱和，多产出正确 rollout 或狭窄失败模式，信息量大的失败样本太稀缺，难以提供有意义的梯度更新。

#### 3. Motivation
当没有足够强的现成教师时，能力提升越来越难——"如何在不依赖更强模型作监督者的前提下获得强模型？"作者想**统一** W2SG 与难度驱动的数据合成：不让弱模型合成难数据或提供学习信号，而是把它当作**结构化扰动的生成器**，在不产生任何新数据的前提下自动抬高训练难度。同时把推理 RL 重述为**去噪问题**（呼应去噪自编码器 / BART 式预训练）。

#### 4. 主要灵感 / 核心直觉
- **前缀决定后续轨迹**：前缀对随后的推理状态有不成比例的影响；既然好前缀能把策略导向更有利状态（PrefixRL），那么**反过来注入错误前缀**就能控制起始状态、强迫策略从损坏的中间状态恢复。
- **两个耦合效应**：(i) 噪声前缀跨越的失败模式空间远比正确轨迹宽，极大扩展训练状态多样性，把策略暴露到标准 on-policy RL 很少遇到的 off-policy 语境；(ii) 直接强化一项被低估的能力——**从错误中恢复**，把自纠错从涌现行为提升为直接训练目标。

#### 5. 主要解决思路（一段话讲清核心）
离线先用一个弱模型对训练集每题采样若干次，保留 verifier 判错的轨迹构成固定"噪声池"。RL 训练时，每题除标准 on-policy rollout 外，额外采若干 **denoise rollout**：取某条错误轨迹的前缀（固定比例 ρ）作为已写好的 assistant 消息，让策略从这个 off-policy 错误前缀续写到答案；verifier 对"前缀+续写"的完整折叠响应打 0/1 奖励，但**只对策略自己生成的续写部分计算梯度**（前缀被 mask）。两类 rollout 共享同一题内 GRPO advantage baseline，按 N:K 加权进同一目标联合优化。

#### 6. 方法详解（通俗、分步骤）
1. **离线噪声采集**：弱模型 πw（Qwen2.5-1.5B-Instruct）对训练集每题采 M 次（实验 M=8），过滤出 verifier 判错的轨迹，构成池 W(q)。这是一次性预处理，训练中固定、**每步零额外成本**。若某题 M 次都没产生格式良好的错误答案，则其 denoise 槽用额外的标准 main rollout 顶替。
2. **每步采样**：每题采 N 个 main rollout（标准 on-policy，y∼πθ(·|q)）+ K 个 denoise rollout。denoise rollout 取错误轨迹 w 的前 p=max(1, ⌈ρ|w|⌉) 个 token 作前缀 w₁:ₚ，策略续写 y>p∼πθ(·|q, w₁:ₚ)。
3. **预算折叠 (output budget & folding)**：两类 rollout 共享同一响应窗口宽度 R 以保证公平；前缀已占 p token，续写折叠进剩余预算，保留长度 L=min(T_{y>p}, R−p)，超出 R 的尾部 token 丢弃。verifier 对完整折叠响应 ỹ=(前缀, 续写) 打奖励。
4. **只更新 on-policy 续写**：训练只对续写部分 y_{p+1:p+L} 算梯度；off-policy 前缀被 mask，避免 PPO 对重 off-policy token 的不稳定。
5. **token 级 GRPO**：同题 N+K 条轨迹共享同一 advantage baseline（μ_q、σ_q 在 N+K 上统计，Eq.5），用 PPO clip 代理目标（ε_low=ε_high）。联合目标 J = N/(N+K)·J_main + K/(N+K)·J_denoise（Eq.8）。
6. **两条设计经验**：(i) 噪声不能过强——过长错误前缀会把模型推向 overthinking（更长自纠错循环、更高不确定性）；(ii) 不要更新 off-policy 前缀——否则训练不稳，与"PPO 式目标对重 off-policy token 敏感"的近期观察一致。

#### 7. 实验数据集
- **噪声采集 & 训练**：均在 MATH-7.5K。弱模型每题 8 rollout 取错误样本。
- **策略模型**：Qwen3-4B-Base、Qwen3-8B-Base（仓库另含 1.7B 脚本，但论文正文主表只报 4B/8B）。
- **评测**：MATH500、AMC23、AIME2024、AIME2025、BBEH。AMC23/AIME24/AIME25 报 AVG@16，MATH500/BBEH 报 AVG@1。验证解码 temperature=0.6, top-p=0.95。

#### 8. 实验结果与主要发现
关键超参：N=12 main + K=4 denoise，response 长度 4096，ρ=0.2，prompt batch 16，lr 1e-6，无 KL/length loss，PPO clip 0.2，训练采样 temperature=1.0, top-p=1.0。主表（Table 1，平均分）：
- **Qwen3-4B-Base**：Base 26.6 → GRPO 39.6 / DAPO 39.8；**DenoiseRL-GRPO 42.0**（最佳平均）、DenoiseRL-DAPO 41.5（AMC23、BBEH 上最佳）。
- **Qwen3-8B-Base**：Base 29.3 → GRPO 43.0 / DAPO 42.8；DenoiseRL-GRPO 43.3、**DenoiseRL-DAPO 44.8**（每个 benchmark 均最佳）。
两个模型尺度、两个 RL backbone 上 DenoiseRL 都一致改进了对应基线，说明它不绑定特定模型大小或优化后端。

#### 9. 结果如何支撑其主张
- "复用弱模型作扰动器即可提升"——主表上每个 (模型×backbone) 组合的 DenoiseRL 版本平均分都 ≥ 基线，支撑"无需更强教师即可改进"的核心主张。
- "提供互补训练信号、对更难基准更有效"——增益在 AIME/BBEH 等更难基准上较明显，与"denoise rollout 补充负样本→正样本携带有效学习信号"的论证方向一致。
- "强化自纠错"——论文称恢复能力随训练难度增强，但这一结论主要靠定性观察（图示与行为描述），缺定量的"恢复率"曲线作硬证据。

#### 10. 逻辑自洽性（中性评估）
机制叙事自洽且实现干净：把 prefix 注入从"注入好前缀"（LUFFY/PrefixRL）反转为"注入弱模型错误前缀"，并通过"只更新 on-policy 续写 + 共享 baseline + 预算折叠"在工程上回避了重 off-policy 优化的不稳定。GRPO/折叠/联合目标的数学表述前后一致。主要张力在于：把"恢复"建模为价值，但训练只在续写段算梯度，等于把 off-policy 前缀当成"免费的难初始状态"而非真正去优化 off-policy 分布——这绕开了真正的 off-policy 优化难题，因此"去噪/恢复"更多是被**条件化**出来而非被**显式优化**出来。

#### 11. 残留问题 / 局限
- **增益偏小**：4B 约 +2.2~+2.4，8B 上 GRPO 仅 +0.3（DAPO +2.0）；提升幅度有限。
- **验证面窄**：仅 MATH-7.5K 单一数据源、仅 2 个策略模型（4B/8B）、仅数学+BBEH，泛化性未充分检验。
- **回避真正 off-policy 优化**：mask 掉前缀只是规避了不稳定，没有正面解决从重 off-policy 数据学习的问题。
- **"恢复能力随难度增强"缺定量证据**：主要靠定性观察，未给出可复核的恢复率/纠错率指标。〔待核〕仓库提供 1.7B 脚本但论文未报其数值。
- 噪声强度（ρ）与 overthinking 的权衡只给了单点 ρ=0.2，缺系统扫描曲线（正文称有此现象但主表未附消融表）。

#### 12. 开源代码与框架（链接+框架+代码可得性）
- 仓库 https://github.com/ALEX-nlp/DenoiseRL，本地已克隆。基于 **VeRL** fork（含完整 `verl/` 包）。
- 核心配方在 `recipe/denoise/`：`dapo_ray_trainer.py`（denoise rollout + 折叠 + GRPO/DAPO 联合目标）、`data_prepare.py`、`verifier.py`、`main_dapo.py`，以及 `denoise_qwen3-{1.7b,4b,8b}_v1.0.sh` 和 `dapo_denoise_qwen3-{1.7b,4b,8b}_v1.0.sh` 启动脚本，配置在 `config/`。另含 `data/`、`paper/`。
- 代码可得性：高（配方、脚本、数据预处理、verifier 齐全，可直接复现 4B/8B 实验）。


---


### dft_reweight — On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification (DFT / Dynamic Fine-Tuning)

> **一句话重点 (TL;DR)**：把标准 SFT 梯度还原成"策略梯度 + 隐含奖励"形式后发现其奖励被 1/πθ（逆概率）加权而病态；只需把每个 token 的交叉熵损失乘以该 token 的预测概率（detach 阻断梯度），就能把隐含奖励整平为常数 1，得到更稳定、更接近 RL 风格的更新，从而显著改善 SFT 的泛化——核心改动只有一行代码。

**元信息**：arXiv 2508.05629 (v3, 2026-02-27) ｜ 东南大学、UCLA、上海交大、南洋理工、UC Berkeley、武汉大学、UC Merced 等；Yongliang Wu, Yizhou Zhou 等 ｜ ICLR 2026 ｜ 主题 T3（High，"用 RL 视角理解并改进 SFT"的代表性单行改动法，ASFT 的直接前置工作）｜ 代码 https://github.com/yongliang-wu/DFT（已克隆约 66MB，已被 TRL / LLaMA-Factory / ms-swift 收录）｜ 框架 veRL（FSDP SFT 训练器 + Liger），评测沿用 Qwen2.5-Math 仓库。

#### 1. 相关工作与进展
- **SFT vs RL 的权衡**是 LLM 对齐的核心议题。SFT 模仿专家 demonstration、简单高效（类机器人 behavioral cloning），但"SFT memorizes, RL generalizes"——SFT 易过拟合、泛化弱于 RL。
- **混合范式**：SFT 预训练 + RL 精修（InstructGPT 式）是最常见策略；近期也有交错式方法。
- **偏好/拒绝采样类**：DPO、RFT/RAFT 等可视为缓解稀疏奖励问题。
- 已有少量工作引入基于数据生成策略的重要性加权，但未把它落到"修正 SFT 隐含奖励"这一点上。

#### 2. 现有工作存在的问题
- RL 泛化好但算力贵、需显式奖励、调参敏感，且在"只有正样本 demonstration、无负样本/无奖励模型"时不可用——此时 SFT 是唯一可行选项。
- 因此核心未解问题是：**SFT 本身能否被根本性改进？** 作者通过数学分析定位症结：把 SFT 梯度解释为策略梯度时，其隐含奖励是 (i) 稀疏的（只在精确匹配专家 token 时非零），且 (ii) 被一个 1/πθ 的重要性权重缩放——当模型给专家动作的概率很低时，权重爆炸→梯度异常大→优化不稳、过拟合稀有精确匹配样本，限制泛化。

#### 3. Motivation
既然 SFT 的隐含奖励因 1/πθ 逆概率加权而病态，那么就用一个"校正性逆比"——即乘以策略概率 πθ——来抵消该畸变，把隐含奖励整平为均匀常数。这样不引入采样、不引入奖励模型、不需要 reference model 或大 batch，就能把 SFT 梯度从有偏不稳定估计变成更接近 RL 风格的稳定更新。

#### 4. 主要灵感 / 核心直觉
- **把 SFT 写成 RL**：SFT 的梯度可解释为 on-policy 策略梯度，其奖励是匹配专家轨迹的稀疏指示函数 r(x,y)=1[y=y⋆]，但被重要性权重 1/πθ 偏置（论文 eq.6）。
- **病态来源是 1/πθ**：低概率专家 token → 巨大权重 → 不成比例的大梯度。
- **校正即乘回 πθ**：乘以 sg(πθ(y⋆|x)) 后，隐含奖励对所有专家轨迹均匀变为 1，类似 RLVR 给所有正确样本统一奖励，避免对低概率参考 token 的过度集中。

#### 5. 主要解决思路（一段话讲清核心）
在标准 SFT 的逐 token 交叉熵损失上，乘以该 token 的模型预测概率（用 stop-gradient/detach 使该系数在反向传播中视为常数）。这一步把 SFT 隐含奖励里的 1/πθ 因子抵消掉，使每个 token 的有效奖励变为常数 1。整个方法仍以纯 SFT 形式实现（无需采样、无 reference model、无奖励模型），只是 loss 多乘一个 detach 后的概率权重。

#### 6. 方法详解（通俗、分步骤）
1. **理论刻画**（eq.6）：SFT 梯度 ≈ E[w(y|x)·∇log πθ(y|x)·r(x,y)]，其中 r 是稀疏指示函数，w=1/πθ 是重要性权重；这说明标准 SFT 是带病态奖励的特殊策略梯度。
2. **奖励整流**（eq.7）：在期望内乘以 sg(1/w)=sg(πθ(y⋆|x))，抵消 1/πθ；sg 保证梯度不流过该缩放项。
3. **轨迹级 DFT loss**（eq.8）：L = E[ −sg(πθ(y⋆|x))·log πθ(y⋆|x) ]。
4. **token 级稳定化**（eq.9，实际使用版本）：因整轨迹重要性权重会数值不稳，仿 PPO 在 **token 级**做重要性采样：L = −Σ_t sg(πθ(y⋆_t|y⋆_{<t},x))·log πθ(y⋆_t|...)。
5. **一行代码实现**（README/仓库 `fsdp_dft_trainer.py` 第 369–371 行，已核对）：
   ```python
   probs = torch.softmax(shift_logits, dim=-1)
   prob_coefficients = probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
   loss = loss * prob_coefficients.detach()
   ```
6. **效果直觉**（Fig.2 分析）：标准 SFT 把所有概率一律推向训练集；DFT 选择性地提高一部分、压低另一部分，弱拟合 token 的比例上升，相当于一种正则化。

#### 7. 实验数据集
- **数学 SFT 训练**：从 NuminaMath-CoT 随机采 100k 条。基座 Qwen2.5-Math-1.5B/7B 等多模型多规模。
- **数学评测**：MATH-500、OlympiadBench、AIME 2024、AMC 2023、Minerva Math 等。
- **offline RL 设定**（Qwen2.5-Math-1.5B）：构造 100k 正负偏好对训 DPO；对比 DPO、RFT/RAFT（offline）与 PPO、GRPO（online，n=4 for GRPO）。
- **跨域**：代码生成（Table 3）、多模态推理数学（Table 4）。

#### 8. 实验结果与主要发现
- **数学主实验**：在 Qwen2.5-Math 上 DFT 比标准 SFT 增益数倍；标准 SFT 常在 OlympiadBench/AIME/AMC 上**退化**，而 DFT 持续改善并提升泛化；增益跨模型、规模、数据量稳定。示例超参：train_batch=256、max_length=2048、lr=5e-5、1 epoch、use_liger=True、bf16。
- **offline RL（Table 2，Qwen2.5-Math-1.5B）**：DFT 平均 **35.43**，比最佳 RL 基线 GRPO 高 +3.43。逐项：MATH500 64.71（GRPO 62.86、PPO 56.10、RFT 48.23）；AMC23 48.44（比 GRPO +7.19、比 RFT +17.66）；Minerva 25.16（比 GRPO +6.23、PPO +9.75）。即 DFT 同时优于 DPO/RFT（offline）与 PPO/GRPO（online），且不需要 reference model 或大 batch。
- **跨域**：在代码生成与多模态推理上同样改善。
- **局限发现**：DFT 在"非确定性多解轨迹"任务（数学/复杂代码 CoT、信息丰富的多模态 CoT）上强，在"单一确定答案、低熵约束 CoT"任务上偏弱。

#### 9. 结果如何支撑其主张
- "1/πθ 是泛化症结" → 校正后（DFT）相对 SFT 在多个困难基准上由退化转为持续提升，且 Fig.2 显示概率分布从"一律拉高"变为"选择性调整"，与理论预测的"整平病态奖励→更好正则化"一致。
- "可与 RL 竞争" → Table 2 上 DFT 平均分超过 PPO/GRPO，且不需奖励模型/reference model，支撑"streamlined alternative"主张。
- 但严格说，eq.6 的"SFT=策略梯度"刻画依赖若干假设（把数据生成策略当成 πθ、指示函数奖励），是一种**理论透镜**而非严格等价；论文也明确承认"RL-style characterization serves solely as a theoretical lens"。

#### 10. 逻辑自洽性（中性评估）
推导链条（eq.5→6→7→8→9）清晰且与代码一一对应（detach 概率系数 = sg(πθ)），实现极简、可复核。最大概念张力在于：所谓"隐含奖励"是把 SFT 强行套进策略梯度框架后的产物，其"病态"本质上等价于"加权交叉熵的梯度对低概率 token 敏感"这一已知事实；DFT 的整流可被看作一种**置信度加权 / focal-loss 反向版**（降低对难拟合 token 的权重）。换言之，RL 透镜提供了优雅动机，但方法本身也能在不诉诸 RL 的情况下被理解为"按模型置信度重加权 SFT"。结论与证据自洽，只是机制解释存在多重等价视角。

#### 11. 残留问题 / 局限
- **任务依赖**：在低熵/单一确定答案、强约束 CoT 任务上增益弱甚至可能不利（作者自陈）。
- **理论刻画的假设性**：eq.6 的策略梯度等价依赖把专家分布视作 Dirac、奖励视作指示函数等假设，是透镜而非严格证明。
- **置信度加权的双刃**：降低低概率 token 权重虽稳定训练，但也可能弱化对"模型当前不会、恰恰最该学"的 token 的学习；论文未充分讨论这一潜在反作用。
- **offline RL 对比的公平性**：DPO/RFT/PPO/GRPO 各自超参与数据构造不同，跨方法平均分比较的可比性需谨慎看待。〔待核〕Table 2 各基线是否经过同等调参未完全交代。

#### 12. 开源代码与框架（链接+框架+代码可得性）
- 仓库 https://github.com/yongliang-wu/DFT（已克隆，约 66MB）。顶层含 `verl/`（训练）与 `math_evaluation/`（评测）。
- 训练器：`verl/verl/trainer/fsdp_dft_trainer.py`（FSDP SFT 训练器，第 369–371 行即 detach 概率重加权，已核对）；FSDP + Liger + bf16。
- 评测沿用 Qwen2.5-Math 仓库（latex 答案匹配）。
- 生态收录：TRL、LLaMA-Factory（`examples/extras/dft`）、ms-swift（`examples/train/full/dft.sh`）均一行启用；模型权重在 HF collection `Liang0223/dft-...`。
- 代码可得性：高（核心改动一行，多框架可直接复现）。


---


### dr_grpo — Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)

> **一句话重点 (TL;DR)**：批判性审视 R1-Zero 范式的两大成分——base 模型与 RL 算法——指出 Qwen2.5 base 已"类 SFT"、"Aha moment"在 base 中早已存在；并发现 GRPO 目标里的 1/|o_i|（响应级长度偏置）与 std(R)（题目级难度偏置）会人为推高（尤其错误）响应长度；去掉这两项得到无偏的 **Dr. GRPO**，在不损推理性能下大幅缩短错误响应、提升 token 效率，并给出 7B 极简 SOTA 配方（AIME24 43.3%，8×A100/27h）。

**元信息**：arXiv 2503.20783（v1 2025-03-21，v2 2025-10-06）｜ Sea AI Lab、新加坡国立大学(NUS)、新加坡管理大学(SMU)；Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi 等 ｜ COLM 2025；ICML 2025 AI4Math Workshop Best Paper Honorable Mention ｜ 主题 T3（RLVR / R1-Zero 训练机理 + RL 算法，High）｜ 代码 https://github.com/sail-sg/understand-r1-zero（已克隆约 56MB，基于自研 RL 框架 Oat）。

#### 1. 相关工作与进展
- **R1-Zero 范式**：DeepSeek-R1-Zero 证明可不经 SFT、直接对 base 模型大规模 RL 提升推理，并伴随 RL scaling（响应长度持续增长）与 "Aha moment"（自反思涌现）。
- **社区复现**：多用 Qwen2.5 系列 + GRPO（SimpleRL-Zero、ORZ 等）。
- **GRPO**（DeepSeekMath）：组内相对优势 + 长度归一化的 PPO 变体。

#### 2. 现有工作存在的问题
- **base 模型的预训练偏置被忽视**：Qwen2.5 base 不用对话模板时性能反而提升约 60%（疑似预训练用了拼接的 question-answer 文本），使其"已类 SFT"；"Aha moment" 在 base 模型（含 DeepSeek-V3-Base）中早已存在，并非纯 RL 涌现。许多"RL 涌现"的归因因此站不住。
- **GRPO 的优化偏置**：目标中 1/|o_i|（响应级长度归一化）和 std(R)（题目级难度归一化）会人为推高响应长度，尤其推高**错误响应**的长度；trl/OpenRLHF/verl/SimpleRL-Zero/ORZ 等多个开源 PPO 实现也存在 length bias。

#### 3. Motivation
从 base 模型与 RL 两个核心成分批判性审视 R1-Zero：厘清哪些现象是真涌现、哪些是预训练遗留；去除 GRPO 的优化偏置以提升 token 效率；并给出一个极简（minimalist）的 R1-Zero 配方。

#### 4. 主要灵感 / 核心直觉
- **GRPO 的长度增长可能是 bug 不是 feature**：1/|o_i| 让长响应每 token 梯度被稀释、短响应被放大，对负优势样本会**奖励变长**（拖长错误回答以摊薄惩罚）；std(R) 归一化让难/易题权重失衡。去掉它们即恢复无偏 PPO 风格优势（蒙特卡洛回报 + 无偏 baseline）。
- **模板与 base 的匹配至关重要**：模板-模型不匹配会先破坏能力、再由 RL 重建；领域预训练能抬高 RL 上限。

#### 5. 主要解决思路（一段话讲清核心）
Dr. GRPO（GRPO Done Right）= 在 GRPO 目标中移除 1/|o_i| 长度归一化项与 std(R) 难度归一化项：把 token 级 loss 的 mask 归一化从"除以本响应长度"改为"除以一个常数（生成预算 MAX_TOKENS）"，并把优势计算改为只减组均值、**不除以组内 std**。其余沿用 PPO 风格。这样在保持推理性能的同时显著抑制响应长度无意义增长、大幅缩短错误响应、缓解 overthinking。

#### 6. 方法详解（通俗、分步骤）
1. **移除长度偏置（Modification 1）**：把 masked_mean（除以 mask.sum=本响应长度）换成 masked_sum 除以常数。
   - 代码（`train_zero_math.py` 第 288–290，已核对）：`masked_sum(..., constant_normalizer=args.generate_max_length) if critic_type=="drgrpo" else masked_mean`。
2. **移除难度偏置（Modification 2）**：优势只减组均值、不除以 std。
   - 代码（`compute_monte_carlo_advantages`，第 294–308，已核对）：`advantages = rewards - values`；仅当 `critic_type=="grpo"` 时才 `advantages /= (std_grouped_rewards + 1e-8)`，drgrpo 不除。
3. **效果**：响应长度不再失控增长，错误响应长度大幅下降，token 效率更高；两者最终 reward 相近。
4. **配套分析**：模板对 base 作答至关重要；模板-模型不匹配先破坏能力再由 RL 重建；领域（数学）预训练提升 RL 上限（Llama-3.2-3B + FineMath/NuminaQA 续训后 RL 更强）。

#### 7. 实验数据集
- **训练**：MATH（Hendrycks）level 3–5（主配方）；分析中另用 ORZ-57k / MATH-12k / GSM-8k / ASDiv-2k 研究"模板 × 题集覆盖度"交互。
- **评测**：AIME 2024、AMC、MATH500、Minerva Math、OlympiadBench。
- **base 模型分析**：Qwen2.5-Math-1.5B/7B、Qwen2.5-7B、Llama-3.1-8B、DeepSeek-Math-7B、DeepSeek-V3-Base-685B。

#### 8. 实验结果与主要发现
- **极简 SOTA 配方**：用 Dr. GRPO 对 Qwen2.5-Math-7B 在 MATH lv.3-5 + Qwen-Math 模板上 RL，达 AIME 2024 **43.3%** 的 7B SOTA，仅 8×A100、27 小时。
- **算法对比**（Qwen2.5-1.5B base + R1 模板，奖励 Math-Verify 0/1）：vanilla GRPO 与 Dr. GRPO 的 reward 曲线相近，但 Dr. GRPO **阻止响应长度失控、错误响应长度大幅下降**，token 效率更高。
- **base 发现**：Qwen2.5 base 去模板性能反升约 60%；多个 base（含 DeepSeek-V3-Base）已有 "Aha moment"。
- **模板/领域**：模板-模型不匹配先掉能力再由 RL 恢复；数学领域续训提升 RL 上限。

#### 9. 结果如何支撑其主张
- "GRPO 有长度偏置" → 去掉 1/|o_i| 后错误响应长度显著下降而 reward/准确率不降，直接验证"长度增长部分来自优化偏置而非真实推理需要"。
- "Aha 非纯涌现" → 在未 RL 的 base 模型（含 V3-Base）上即观测到自反思 token，削弱"RL 涌现自反思"的强主张。
- "base 已类 SFT" → 去模板性能反升的对照支撑 Qwen2.5 base 预训练已含 QA 拼接的推断（间接证据，非直接数据审计）。

#### 10. 逻辑自洽性（中性评估）
两处算法修正有清晰的数学动机（恢复无偏 PPO 优势）且与代码精确对应（masked_sum 常数归一化、优势不除 std），是该工作最扎实的部分。base 分析多为观察性/相关性证据："Qwen2.5 预训练用了 QA 拼接"是从"去模板性能反升"反推的合理猜测而非确证；"Aha moment 早已存在"依赖对 base 生成的定性识别，缺统一的量化指标。因此"批判 R1-Zero 范式"的算法侧结论强、归因侧结论偏推测。整体论证自洽，但部分批判性结论的证据强度低于其修辞强度。

#### 11. 残留问题 / 局限
- **base 归因偏推测**：Qwen2.5 预训练数据未公开，"已类 SFT"为间接推断；"Aha moment 早已存在"缺量化判据。
- **去 std 的代价未充分讨论**：std 归一化本意是稳定不同难度题的梯度尺度，移除后在极端难度分布下是否引入新不稳定，论文着墨不多。
- **验证集中于数学**：评测与训练几乎全是数学推理，去偏置结论在代码/通用推理上的迁移性未验证。
- **常数归一化引入新超参**：用 generate_max_length 作常数，其取值对梯度尺度有影响，需配合 lr 调整。

#### 12. 开源代码与框架（链接+框架+代码可得性）
- 仓库 https://github.com/sail-sg/understand-r1-zero（已克隆，约 56MB）。训练入口 `train_zero_math.py`（含 Modification 1：masked_sum 常数归一化；Modification 2：MC 优势不除 std，均已核对）；算法包 `understand_r1_zero/`（含 `math_grader.py` 答案判定）；另有 `analysis/`、`datasets/`、`examples/`、`deploy_dpsk/`、`evaluate_model.py`。
- 框架：自研 **Oat**（sail-sg 的模块化 LLM online alignment / RL 框架；底层 vLLM + DeepSpeed），依赖 math-verify、pylatexenc、fire 等；奖励用 Math-Verify。
- 资产：HF `sail/Oat-Zero` collection（Oat-Zero-7B 等）。
- 代码可得性：高（训练入口、算法、判分、评测齐全，可复现极简 7B 配方与 1.5B 对比实验）。


---


### eaft — Entropy-Adaptive Fine-Tuning: Resolving Confident Conflicts to Mitigate Forgetting (EAFT)

> **一句话重点 (TL;DR)**：SFT 之所以损害通用能力，主要源于一类"模型很自信、却被强迫去学相悖标签"的 token（Confident Conflicts）；EAFT 用逐 token 的归一化熵作为门控系数去缩放交叉熵——模型自信(低熵)就压低梯度、不确定(高熵)就正常学习，从而在几乎不损失目标任务的同时显著缓解遗忘。

**元信息**：arXiv:2601.02151（2026-01-07，预印本，曾居 HuggingFace Daily Paper #1）｜ 北京邮电大学 PRIS-CV / 中关村学院（Muxi Diao、Lele Yang、Wuxuan Gong 共同一作；通讯 Zhanyu Ma）｜ 主题 T3（SFT 改进 / 缓解遗忘）/ 相关性中-高 ｜ 代码 github.com/PRIS-CV/EAFT（主仓库仅 README+assets，实现已合入 LLaMA-Factory 与 ms-swift；本地两个 submodule 指针未拉取，**无法直接审计损失实现**）｜ 框架 LLaMA-Factory（主）+ ms-swift。

#### 1. 相关工作与进展
- **领域适配的主流范式是 SFT**：行为克隆地拟合外部监督数据，简单高效，是把通用模型适配到数学、医学、Agent 等垂直域的标准做法。
- **缓解遗忘的已有思路**：加 KL 正则（SFT-KL，约束新模型不偏离原模型）、对 token 按预测概率重加权（DFT、FLOW、TALR 等，思路是"模型已会的少学、不会的多学"）。
- **RL 与 SFT 的对照观察**：近期工作（Chu et al. 2025 等）发现 on-policy RL 在适配目标任务时对通用能力的损害远小于 SFT，因为 RL 对齐的是模型"自己的信念"，而 SFT 强迫拟合"外部标签"。本文正是从这一对照出发。

#### 2. 现有工作存在的问题
- **KL 正则**：粗粒度约束整个分布，既会抑制有害更新也会抑制有益学习，往往得不偿失。
- **基于概率的重加权（DFT 等）**：只看预测概率不足以判断该不该学。低概率 token 有两种截然不同的来源——(a) **epistemic uncertainty**：模型确实不会、应该学的有效知识；(b) **Confident Conflict**：模型其实很有把握(自己想输出别的)，标签却要它输出一个相悖 token。概率重加权在后者上反而**放大**破坏性梯度，加速遗忘。

#### 3. Motivation
作者系统统计训练数据的逐 token 概率与熵，刻画出 SFT 与 on-policy rollout 之间的"**分布差异 (distributional gap)**"，并定位出最有害的一类样本：**低概率 + 低熵**的 token——模型对自己的预测高度确信(低熵)，却被 ground truth 强行拉向另一个 token(低概率)。论文称之为 **Confident Conflicts**，并通过 pilot study 验证：单纯**屏蔽**这些 token 就能明显减小通用能力下降，说明遗忘主要来自"在冲突样本上的破坏性梯度"，而非 SFT 流程本身。

#### 4. 主要灵感 / 核心直觉
区分"该学"与"该抑制"的信号不是概率、而是**熵**：
- 熵高 = 模型在该位置不确定/在探索 → 这是真该学的新知识，保持高学习权重；
- 熵低 = 模型很笃定 → 若此时标签相悖即为 Confident Conflict，应压低梯度，避免破坏已有的通用表征。
用一个连续的熵门控替代 pilot study 里的硬屏蔽(后者会丢数据、且依赖敏感阈值 τ、δ)，得到一个软的、自调节的学习信号。

#### 5. 主要解决思路(一段话讲清核心)
在标准 SFT 的逐 token 交叉熵损失前，乘上一个**由模型当前预测熵算出的归一化门控系数** H̃_t∈[0,1]：模型在该 token 越自信(熵越低)系数越接近 0、梯度被抑制；越不确定(熵越高)系数越接近 1、退化为普通 SFT。整个改动**只替换损失函数**，不引入 RL rollout、不引入额外模型、流程与标准全参微调完全一致。

#### 6. 方法详解(通俗、分步骤)
**EAFT 损失(论文 Eq.2)**：

L_EAFT(θ) = − Σ_{t=1}^{T} H̃_t · log P_θ(y_t | x, y_{<t})

其中前一项 H̃_t 是"自适应门控信号"，后一项是标准监督项。门控信号(Eq.3)用 **Top-K 熵**近似全词表熵以省算力：

H̃_t = H_top-K_t / ln(K) ≈ H_top-20_t / 3.0  (K=20，ln(20)≈3.0 为 K 个结果的最大熵，归一化到 [0,1])

H_top-K_t 是在该 token 的 Top-20 概率分布上算的熵。自调节机制(论文原文)：
- **Conflict Suppression（H̃_t→0）**：模型固执(低熵)时权重趋零，屏蔽冲突标签的破坏性梯度；
- **Knowledge Acquisition（H̃_t→1）**：模型不确定/探索(高熵)时权重接近 1，恢复标准 SFT 目标以学习新模式。

与 DFT/FLOW/TALR 的关键区别：用**熵而非概率**作门控，从而避免在 Confident Conflict 上放大梯度。
〔待核：门控系数是否对其本身停止梯度(detach)——论文正文只说"用熵缩放监督"，未显式写 stop-gradient；实现细节需以 LlamaFactory-EAFT 损失代码为准，本地 submodule 为空无法核对。〕

#### 7. 实验数据集
- **训练数据构造**：prompt 取自 NuminaMath 与 Nemotron-CrossThink，用 **Qwen3-235B-A22B-Instruct** 对 prompt 合成回答，随机选取 **19k 条被验证为正确**的实例作为数学域训练集(医学/Agent 域各自另构)。
- **Backbone**：Qwen3 与 GLM-4 系列，4B–32B(如 Qwen3-4B-Instruct、Qwen3-4B-Thinking、GLM-4 系列、32B 级)。
- **目标任务评测**：数学(AIME24/AIME25/GSM8K)、医学(MedMCQA/MedQA/PubMedQA)、Agent(BFCL v3)。
- **通用能力(遗忘)评测**：MMLU、IFEval、CLUEWSC。
- **基线**：SFT、SFT-KL(β=0.5)、FLOW、DFT、TALR。

#### 8. 实验结果与主要发现
- 以 Qwen3-4B-Instruct 数学域为例(Table 1)：目标任务上 EAFT 的 Math Avg 与标准 SFT 相当(SFT 把 Math Avg 从 68.3 提到 69.4)；**通用域 General Avg 几乎不掉(EAFT 80.1，仅 -1.0；而 SFT 76.5，掉 -4.6)**——即在不牺牲目标提升的前提下把遗忘量级缩到约 1/4~1/5。
- pilot study(硬屏蔽 Confident Conflict)即可减小通用能力下降，验证了"遗忘主因是冲突样本"的假设；EAFT 的软门控比硬屏蔽更好(不丢数据、无敏感阈值)。
- 跨域(医学、Agent)与跨模型(4B–32B)上趋势一致，支撑方法的通用性主张。

#### 9. 结果如何支撑其主张
论文的核心因果链是"Confident Conflict → 破坏性梯度 → 遗忘"。pilot study 的屏蔽实验直接验证了"去掉这些 token 就能减小遗忘"，为因果方向提供了证据；主表则证明软门控版本(EAFT)能同时保住目标任务收益与通用能力——两者结合较好地支撑了主张。增益主要体现在**遗忘量(General Avg 下降幅度)**而非目标任务绝对分(目标分与 SFT 基本持平)。

#### 10. 逻辑自洽性(中性评估)
机制清晰且自洽：熵门控的两端行为(低熵抑制、高熵学习)与"区分 epistemic uncertainty vs Confident Conflict"的动机直接对应；用 Top-20 熵近似全词表熵、ln(K) 归一化都合理。一个需注意的内在张力：门控只看模型**自身**的熵，并不直接判断"标签是否真的与模型相悖"——低熵但标签恰好一致的 token 也会被同等压权重，这在理论上可能轻微拖慢部分正确知识的学习；论文用"目标任务不掉分"的实验结果间接回应了这一担忧。

#### 11. 残留问题 / 局限
- **实现不可直接审计**：核心贡献(熵门控损失)在两个集成分支里，本地 submodule 为空；detach/数值稳定/Top-K 取法等细节只能据论文+README 推断。
- **熵门控可能抑制部分有益学习**：低熵≠冲突，门控对"低熵且标签一致"的 token 也降权，是否最优未深入消融。
- **超参 K 与归一化**：K=20、ln(K) 归一化为经验选择，未见对 K 的系统敏感性分析(正文只给计算开销分析 Sec 5.2)。
- **评测以中英文通用基准为主**：遗忘的衡量依赖 MMLU/IFEval/CLUEWSC 三项，覆盖面有限。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 主仓库 github.com/PRIS-CV/EAFT(README+assets，无独立训练代码)；项目页 ymxyll.github.io/EAFT/。
- **实现集成**：LLaMA-Factory(已合入官方，经 `use_eaft_loss` 启用；作者分支 github.com/ymxyll/LlamaFactory-EAFT)与 ms-swift(github.com/ymxyll/ms-swift-EAFT，支持 megatron/deepspeed)。
- **框架**：标准全参/SFT 流程(DeepSpeed/Megatron/FSDP2/LoRA)，EAFT 仅替换损失，无 RL、无额外模型。
- **代码可得性**：〔本地 clone 的 `resource/repos/eaft/` 下 `LlamaFactory-EAFT/`、`ms-swift-EAFT/` 两个子目录为空(submodule 指针未拉取)，损失实现需到上述 fork 仓库查看，本地无法审计。〕


---


### entropy_mechanism — The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models

> **一句话重点 (TL;DR)**：RLVR 训练里策略熵会快速坍缩、把性能死死锁在一个可预测的低上限上；本文给出熵坍缩的经验定律 R=−a·exp(H)+b 与其动力学根因(熵变正比于"动作概率与 logit 变化的协方差")，并提出 Clip-Cov / KL-Cov 两个只动"高协方差 token"的简单干预来持续维持探索、突破瓶颈。

**元信息**：arXiv:2505.22617（v1, 2025-05-28，预印本）｜ 上海 AI Lab / 清华 / UIUC / 北大 / 南大 / CUHK（PRIME-RL 团队，Ganqu Cui 等）｜ 主题 T3（RLVR 训练机理 + 算法）/ 相关性高 ｜ 代码 github.com/PRIME-RL/Entropy-Mechanism-of-RL（本地已 clone ~5.9MB，Tier A；已合入官方 veRL PR #1830）｜ 框架 veRL（fork 自 DAPO recipe）。

#### 1. 相关工作与进展
- **RLVR(可验证奖励 RL)被视作后训练算力的下一增长极**：在数学、代码等有客观奖励的任务上能持续提升推理能力。
- **探索-利用的经典张力**：RL 要靠探索找到更优策略，策略熵是探索能力的直接度量。
- **已有的熵控制手段**：传统做法是加熵正则项或 KL 正则；但本文证明这些朴素手段在 LLM RLVR 上**无法有效缓解熵坍缩**。

#### 2. 现有工作存在的问题
- **缺乏对 LLM 策略熵典型行为的系统刻画**：大家都观察到训练后期模型"变得过度自信、探索枯竭、性能饱和"，但没有定量规律。
- **熵坍缩(entropy collapse)**：无干预时策略熵在训练早期就急剧降到接近 0，性能随之触顶。这意味着即便继续投入算力，边际收益也趋近于零——直接限制了"扩 RL 算力"的价值。
- **朴素熵/KL 正则失效**：实验显示它们要么压不住坍缩，要么过度干扰优化。

#### 3. Motivation
既然熵坍缩可预测地锁死性能上限，那就应当(1)给它建立一条可外推的经验定律(像 Scaling Law 那样用早期/小模型预测终态)，(2)从优化动力学层面找到坍缩的根因，(3)据此设计一个能持续注入探索、突破熵瓶颈的可扩展方法。

#### 4. 主要灵感 / 核心直觉
熵的变化不是均匀来自所有 token，而是**集中在少数"高协方差" token**上：那些"模型已经高概率、又恰好拿到高 advantage"的动作会被进一步强化、迅速降熵。只要**专门限制这一小撮 token 的更新步长**，就能在不破坏整体优化的前提下稳住策略熵——这比对全局加熵正则精准得多。

#### 5. 主要解决思路(一段话讲清核心)
先用经验定律 **R = −a·exp(H) + b** 把"验证性能 R"与"策略熵 H"绑定，说明性能是用熵"换"来的、上限(H=0 时 −a+b)完全可预测；再从理论上证明相邻两步的熵变 ≈ 动作 log 概率与其 logit 变化的**协方差**，而在策略梯度类算法下 logit 变化又正比于 advantage——于是高概率×高 advantage 的动作降熵、稀有×高 advantage 的动作升熵，且训练全程协方差为正使熵单调下降。最后提出两个干预——**Clip-Cov**(对一小撮正协方差 token 停梯度)与 **KL-Cov**(对协方差最大的一批 token 加 KL 惩罚)——直接替换 surrogate loss 里的 clip / PPO-KL，从而主动调控熵。

#### 6. 方法详解(通俗、分步骤)
**(a) 经验定律**：R = −a·exp(H) + b(a、b 为拟合系数)。求导得 dR/dH = −a·exp(H)，性能随熵下降而提升、在 H=0 触顶。该定律在多家族模型上都成立，可用早期/小模型外推终态。

**(b) 熵动力学理论**：对 softmax 策略，两步间熵变 ∝ Cov(log π(a), Δlogit(a))。在 Policy Gradient / Natural PG 下 logit 差 ∝ advantage。实证显示协方差项与实测熵差**精确吻合**，且训练全程协方差为正，解释了熵为何单调坍缩。

**(c) 两个干预**(都只作用于"高协方差" token)：
- **Clip-Cov**：随机选取一小部分**正协方差** token，detach 其梯度(停止更新)。
- **KL-Cov**：对协方差 **Top-k%** token 改用带 KL 惩罚的 loss。
二者分别替换 surrogate loss 里的 clip 与 PPO-KL，通过阈值参数主动把策略熵维持在更高水平，逃离低熵陷阱。

#### 7. 实验数据集
- **任务**：数学 + 代码(均可验证奖励)。数学评测集 MATH500、AIME24、AIME25、AMC、OlympiadBench、OMNI-MATH(主表 Table 2 重点报 AIME24/AIME25/AMC)；代码用 Eurus-2-RL-Code 与 KodCode 的测试划分。评测时 AIME/AMC 用温度 0.6 rollout、其余数学题用 greedy(Appendix A)。
- **熵-性能拟合(覆盖广)**：Qwen2.5 家族、Mistral 家族(7B-v0.3/Nemo/Small-3.1-24B)、LLaMA 家族(3.2-3B/3.1-8B)、DeepSeek-Math-7B-Base(附录给出不同模型/数据/Instruct 的拟合)。
- **Clip-Cov/KL-Cov 主实验**：Qwen2.5-7B 与 Qwen2.5-32B(Zero 设置，从 base 起 RL；Table 2、Fig.11)。
- 〔注：先前分析提到"data_source 代码硬编码 aime/aime25/amc"不成立——本地 fork 的 `verl/utils/reward_score/__init__.py` 相关分支为注释，未见硬编码。〕

#### 8. 实验结果与主要发现
- **经验定律普适**：R=−a·exp(H)+b 在多家族模型上都拟合良好，性能上限可由早期训练预测——意味着"无干预的 RLVR 收益本就有限"。
- **协方差理论被实证支持**：协方差项与熵差精确匹配、且全程为正，定量解释了坍缩。
- **朴素正则失效、Cov 干预有效**：传统熵/KL 正则压不住坍缩；Clip-Cov / KL-Cov 能持续维持探索并提升下游数学推理性能(7B/32B 上均优于无干预与朴素正则)。

#### 9. 结果如何支撑其主张
三层证据互相咬合：经验定律说明"熵决定性能上限"(why it matters)，动力学理论给出"为什么熵会坍缩"(协方差为正)，干预实验证明"按理论限制高协方差 token 确实能维持熵并提分"(理论→方法→效果的闭环)。这条链条比单纯报告一个 trick 更有说服力。

#### 10. 逻辑自洽性(中性评估)
内部自洽性强：定律、理论、干预三者由同一个"协方差"概念贯穿。需保留的批判点：(1)经验定律 R=−a·exp(H)+b 是拟合关系而非严格因果，外推到很大算力/很强模型时是否仍成立未知；(2)协方差理论是对 softmax + 策略梯度的局部一阶分析，对加了大量工程 trick 的实际 RLVR 是近似；(3)Clip-Cov/KL-Cov 引入新阈值超参(选多少比例、KL 系数多大)，等于把"调熵正则"换成"调协方差阈值"，并未消除调参负担。

#### 11. 残留问题 / 局限
- **干预带来新超参**：`k_percent`/`ppo_kl_coef`(KL-Cov)、`clip_ratio`/`clip_cov_lb`/`clip_cov_ub`(Clip-Cov)需调，最优值任务相关。
- **维持熵 ≠ 一定更好**：过度维持熵可能反而拖慢收敛；论文给的是"在测试任务上更好"，但何时该停止维持探索缺乏自适应准则。
- **主干预实验集中在 Qwen2.5-7B/32B 的 Zero 设置**，对 Instruct 起点、更大模型、代码任务的干预效果验证相对薄弱。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库 github.com/PRIME-RL/Entropy-Mechanism-of-RL(本地已 clone，Tier A)；方法已合入官方 **veRL**(PR #1830)，可经 `loss_mode=clip_cov`/`kl_cov` 直接使用(上游 `recipe/entropy/`)。
- **框架**：fork 自 veRL，在 **DAPO recipe**(`recipe/dapo/`)上构建；conda env `entropy`(`environment.yaml`)。
- **代码核对(已审计)**：`core_algos.py:compute_policy_loss_clip_cov` 在正协方差(lb~ub 间)token 上随机选 `clip_ratio` 比例，把梯度校正系数 corr 置 0(停更)；`compute_policy_loss_kl_cov` 对协方差 Top-k% token 改用带 KL 惩罚的 loss——与 §6 描述一致。
- **运行**：单节点 `bash recipe/dapo/7b_kl_cov.sh`(KL-Cov 训 Qwen2.5-7B)；多节点 `recipe/dapo/32b_*.sh`(Qwen2.5-32B)。


---


### fest — FEST: Boosting RLVR via Randomly Selected Few-Shot Guidance

> **一句话重点 (TL;DR)**：只用 **128 条随机**(非精选)抽自 SFT 数据集的示范，就能显著提升 RLVR；关键是 semi-online DPO 的梯度天然同时含"监督 + on-policy + 自带衰减权重"三要素，把少量示范作正样本、agent rollout 作负样本即可。

**元信息**：arXiv:2605.15012（v1, 2026-05-14，标注 Ongoing Work）｜ UIUC（Kai Yan、Alexander G. Schwing、Yu-Xiong Wang）｜ 主题 few-shot demonstration-guided RLVR / 相关性中（与"少量教师示范引导 + 负样本 + SFT/RL 统一"相关）｜ 代码 github.com/KaiYan289/FEST ｜ 框架 VeRL（GRPO 为主，few-shot 用 semi-online DPO）。

#### 1. 相关工作与进展
- **RLVR**(可验证奖励 RL)在数学/编码上很成功，但在难题上**样本效率低**。
- **demonstration-guided RL / unified post-training**(LUFFY、SRFT、HPT、ReLIFT、MIFO、SuperRL、CHORD 等)：在 RL 采样失败(全错、advantage=0、无学习信号)时引入 SFT 提供外部知识。
- 这条线已较成熟，但代价是 **SFT 数据需求大**(数 K~50K)。

#### 2. 现有工作存在的问题
- **SFT 数据昂贵**：高质量长链推理示范需精心策划(如 HLE 2500 题动用 1000 名博士)；从已有模型蒸馏又涉及合法性、API 成本、model collapse 风险。相比之下"只有答案、无推理过程"的 RL 数据易得。
- **现有方法不能用随机少量数据**：LUFFY/SRFT 用满 46K、HPT 10K、ReLIFT 8.6K、MIFO 6.4K，且多需精心策划、非随机。
- **few-shot 极少数据的三大挑战**：(i) 无按需数据(专家不能对任意失败题现场补示范)、(ii) 语义覆盖有限、(iii) 反复多 epoch 训练**过拟合风险高**。

#### 3. Motivation
能不能把 SFT 数据从数 K 降到只有 **128 条、且随机抽取**，仍显著超过纯 RLVR？为此先拆出"用好极少示范"必须同时满足的三个要素，再找一个天然同时具备三者的损失。

#### 4. 主要灵感 / 核心直觉
三个必要要素：(i) **监督信号**(外部知识，RLVR 的二值奖励之外的唯一来源)、(ii) **on-policy 信号**(让模型拿自己的 rollout 与示范对比，缓解 exposure bias、起对抗训练作用、扩大极少题目的学习面)、(iii) **decaying weight**(随训练推进降低对少量数据的权重，防过拟合)。关键观察：**semi-online DPO** 的梯度恰好天然分解出这三项。

#### 5. 主要解决思路(一段话讲清核心)
总损失 L = c·L_E + L_I：在大规模"答案-only"数据 D_I 上跑标准 GRPO(L_I，省 KL 与 std)；在 128 条 few-shot D_E 上跑 **semi-online DPO**(L_E)——把 SFT 示范当 preferred 样本 y+、把 agent 当前 rollout 当 non-preferred y−。该 DPO 损失对参数求梯度后正好得到"监督项 − on-policy 项"再乘以一个 σ(β(r−−r+)) 的自带衰减权重，三要素一次满足。

#### 6. 方法详解(通俗、分步骤)
**(a) L_I(答案-only 数据)**：标准 GRPO，按组内相对奖励算 advantage，沿用 HPT/Dr.GRPO 省去 KL 与 std。

**(b) L_E(few-shot semi-online DPO)**：L_E = −E[ log σ(β·r+ − β·r−) ]，r+=log(π_θ(y+)/π_ref(y+))、r−=log(π_θ(y−)/π_ref(y−))，y+ 为示范、y− 为 rollout。其梯度(Eq.4)= −β·E[ σ(β(r−−r+))·(∇log π(y+) − ∇log π(y−)) ]，三项依次对应监督学习、on-policy、衰减权重。

**(c) 自适应 β(Eq.5)**：按题目可解性分三档——一组全错用 β1、RLVR-可解但本条错用 β2、本条正确用 β3，细粒度控制不同来源数据的学习强度。β 取 **0.001–0.1**(远小于标准 DPO 的 0.1–0.2，因长链推理序列长、log-ratio 差异大)。

**(d) FEST-GRPO 变体(Sec.3.3，治 gradient mismatch)**：DPO 是 sequence-level、GRPO 是 token-level，二者梯度量级不匹配、需繁琐调 c。论文证明 **semi-online DPO ≈ "REINFORCE(负奖励) + 加权 SFT"**，于是把其中的 REINFORCE 部分换成 GRPO，消除量级失配；这一等价关系也把 DPO 纳入了 HPT 的 unified post-training 框架(Appendix B.3)。

#### 7. 实验数据集
训练用 OpenR1-Math-46K-8192，随机抽 128 题作 D_E、其余作答案-only D_I；模型 Qwen2.5-Math-1.5B。评测 6 个数学基准：AIME25、AMC23、AIME24、MATH-500、OlympiadBench、Minerva，报 Avg@8(均值±标准差)与 Pass@8。基线 GRPO、SRFT、LUFFY、CHORD-φ、MIFO、HPT、ReLIFT(及其 -G 变体)。

#### 8. 实验结果与主要发现
- **128-shot 下 FEST 最优**：FEST-DPO 平均 **41.98**、FEST-GRPO **42.36**，均超 vanilla RL(39.79)及所有 128-shot 基线，甚至匹配用**全量数据**的 SRFT(35.05)/MIFO；是该稀疏数据条件下**唯一**显著超过纯 RL 的方法。
- **朴素把 RL 也加到 gold few-shot 上(HPT-G、ReLIFT-G)会显著掉点**(如 HPT-G 仅 32.02)：训练曲线显示中途骤降——说明在极少 gold 数据上做 RL 不稳定。
- 增益相对 RL 基线约 +2.2~+2.6 分(绝对值)，主要卖点是"数据从数 K 降到 128"。

#### 9. 结果如何支撑其主张
主张是"128 条随机示范足以提升 RLVR"。Table 2 在同一 128-shot 约束下与多个强基线对比、并显示 FEST 匹配全量数据方法，直接支撑数据效率主张；梯度分解(Eq.4)与 REINFORCE 等价关系(Sec.3.3)从理论上解释了"为什么 semi-online DPO 恰好够用"，让结果不只是经验巧合。但绝对性能增益不大、卖点在数据量而非分数。

#### 10. 逻辑自洽性(中性评估)
形式化清晰、自洽：把"用好 few-shot"拆成三要素，再证明 semi-online DPO 的梯度天然含这三项、并与"REINFORCE 负奖励 + 加权 SFT"等价，把 DPO 纳入 HPT 框架——这一分解是论文最扎实的部分。可质疑处：(1)自适应 β 三档划分是启发式，β 取值范围(0.001–0.1)依赖经验调参;(2)Remark 3.1 对"DPO 难翻转偏好/拒答主导"的辩护(称在本场景无害)偏定性。

#### 11. 残留问题 / 局限
- **验证面窄**：仅单模型 Qwen2.5-Math-1.5B、单一数据源(OpenR1-Math)、纯数学；规模小(论文自标 Ongoing Work)。
- **超参敏感**：自适应 β 三档 + 系数 c(FEST-DPO 仍需调，FEST-GRPO 才缓解)依赖经验。
- **128 这一数字的来源**是沿用前作 batch size(一个 epoch 恰好一步)，并非对"最少需多少示范"的系统搜索(Sec.4.2 有 shots 缩放但仍有限)。
- **txt 仅捕获到前 6 页**，附录(B/C/D 的理论与超参分析)未在本地全文核对〔待核：附录细节〕。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 代码 github.com/KaiYan289/FEST。核心 `ternary_dpo/`(基于 VeRL，含 verl、setup.py)、`examples/math-1.5b-v3`、`dataset/`、`utils/`、`eval_by_question_results/`。
- **框架**：VeRL；GRPO 为主 RL 框架，few-shot 用 semi-online DPO。
- **训练配置**：2×NVIDIA GH200(96GB)，600 步；n=8 rollout/题，温度 1.0，max len 8192；AdamW，cosine lr 1e-5→5e-6；global batch 128 题、mini-batch 512 rollouts。报告取第 600 步结果(沿用 ReLIFT)。


---


### gmpo — Geometric-Mean Policy Optimization (GMPO)

> **一句话重点 (TL;DR)**：GRPO 优化 token 级奖励的**算术平均**，对离群重要性比敏感、易引发激进更新与不稳定。GMPO 即插即用地换成**几何平均**（在 log 空间做乘积与裁剪），天然抗离群、重要性比方差更低，从而能用**更大的裁剪窗口**鼓励探索而仍保持稳定。

**元信息**：arXiv 2507.20673（v3, 2025-10-18）｜ UCAS / CUHK / HKUST / Microsoft Research（部分作者 MSR 实习完成）｜ ICLR 2026 接收 ｜ 主题 T3（RLVR 稳定性），相关性 High（对本课题 RL 阶段稳定性与更大裁剪窗口下探索设计有直接参考）｜ 代码 https://github.com/callsys/GMPO（已 clone ~57MB，Tier A）｜ 框架 专用仓基于 **Oat + vLLM 0.8.4**（构建于 understand-r1-zero / Dr.GRPO 之上）；另集成入 **veRL**（`examples/gmpo_trainer`）。

#### 1. 相关工作与进展
- **GRPO（Shao 2024）**：组内相对优势，免 value 模型，数学/代码/QA 上强。
- 大量 GRPO 变体（论文 §2.1 罗列）：DAPO（动态采样 + clip-higher）、Dr.GRPO（去长度偏置）、GPG（去 surrogate/critic/KL）、OPO（最优 baseline 降方差）、80/20 规则（高熵少数 token 主导）、entropy-based advantage 等。
- 多数变体聚焦采样/优势/奖励塑形；**RL 训练稳定性本身仍欠探索**。

#### 2. 现有工作存在的问题
- GRPO 目标是 token 级奖励的**算术平均**，对离群值敏感：当某些 token 的重要性比 ρt(θ)=πθ/πθold 达到极值时，importance-weighted reward ρt·Â 出现离群，驱动**激进策略更新**并进一步放大 ρt 方差，导致不稳定甚至退化。
- GRPO 用 clip 限制 ρt 偏离，但过窄的 clip **压制探索、过早收敛到确定性策略**，熵快速塌缩、性能 plateau。

#### 3. Motivation
用对离群值天然更鲁棒、重要性比分布方差更低的**几何平均**替换算术平均；在维持稳定的同时**放开裁剪范围**以促进探索，兼得稳定与探索。

#### 4. 主要灵感 / 核心直觉
- 几何平均 = log 空间的算术平均；单个极端 ρt 在乘积/对数和里被"摊薄"，不会单点主导梯度。
- 更稳的目标 → 可用更宽 clip（如 e^±0.4，远宽于 GRPO 的 0.8/1.2、DAPO 的 0.8/1.28）→ 更高熵、更强探索 → 更好性能。

#### 5. 主要解决思路(一段话讲清核心)
把 GRPO 目标里 token 级奖励的算术平均（式 2）换成几何平均（式 3）：对每条 rollout，取 token 重要性比乘积的 1/|o| 次方（含 sgn(Â) 保证优化方向），等价于在 **log 空间**求 token 对数比的均值再 exp，乘积与裁剪都在 log 空间执行（Algorithm 1）。理论上 GMPO 目标值域更窄（训练方差更小）、梯度对 ρt 离群更鲁棒；实测训练中保持更小的对 ref 模型 KL 与更高 token 熵。裁剪采用 **token 级**（而非序列级）。

#### 6. 方法详解(通俗、分步骤)
1. 与 GRPO 同样采样 G 个 rollout、组内标准化得优势 Âi。
2. **几何平均目标（式 3/4）**：loss = −Â · exp( Σ_t clip(sgn(Â)·(logπθ−logπθold), −ε, ε)·sgn(Â) / |o| )（伪代码见 Algorithm 1，全程 log 空间）。
3. **token 级裁剪（关键设计 i）**：对每个 token 的对数比做 clip，而非对整条序列乘积做 clip。理由：(1) token 级 clip 的重要性比范围更小、更稳（Fig.3）；(2) 序列级 clip 一旦触发会把整条序列所有 token 梯度清零，过于激进、丢弃有用信号。
4. **放宽裁剪窗口（关键设计 ii）**：设 (ε_low, ε_high)=(e^−0.4, e^0.4)，显著宽于 GRPO/DAPO，兼顾稳定与探索；过宽（如 −∞,+∞）反而不稳。
5. 沿用 Dr.GRPO 设置，忽略显式 KL 正则项以省显存。

#### 7. 实验数据集
- **语言侧（沿用 Dr.GRPO）**：训练用 MATH Level 3-5（8523 题，<7B 模型）、MoE 用 DeepScaleR + CountDown；评测五个数学基准 AIME24(30)、AMC(83)、MATH500(500)、Minerva(272)、OlympiadBench(675)。
- **多模态侧（沿用 EasyR1）**：训练/评测 Geometry3K(601)。
- **模型**：Qwen2.5-Math-1.5B/7B、DeepSeek-R1-Distill-Qwen-7B、Qwen3-32B(MoE)；多模态 Qwen2.5-VL-Instruct-7B。8×A800。

#### 8. 实验结果与主要发现
- **主结果（Table 1）**：GMPO-7B(R1-Distill) 五数学基准均值 **63.4% vs GRPO 59.3%（+4.1%）**；Qwen2.5-Math-7B +1.5%、1.5B +1.4%。MoE Qwen3-32B 上 MATH500 **96.7% vs 94.6%（+2.1%）**。多模态 Geometry3K **54.7% vs 53.3%（+1.4%）**。
- **消融（Table 4）**：GRPO 51.2% → GMPO 52.7%（+1.5%）；去归一化项 1/|o| 掉 0.7%；去 clip（−∞,+∞）掉 0.4%；seq-clip 与 token-clip 性能相近但前者重要性比范围更大，故选 token-clip。
- **裁剪阈值（Table 5）**：(e^−0.4,e^0.4) 最优（52.7%）。
- **训练动态（Fig.4/5）**：GMPO 全程更高熵、更稳梯度、更小 KL；CountDown 上 GRPO 约 250 步后崩溃，GMPO 稳定。

#### 9. 结果如何支撑其主张
- "更稳"由更小 KL、更稳梯度、MoE/CountDown 不崩溃支撑；"更强探索"由更高熵 + 更宽可用 clip 支撑；理论侧给出目标值域更窄（不等式）与梯度鲁棒性推导（Appendix A）。证据链较完整。

#### 10. 逻辑自洽性(中性评估)
- 理论与实现一致（log 空间几何平均 = 对数比均值），梯度推导（Lemmas 1-3）严谨。
- 自洽点：把"几何平均抗离群"→"可放宽 clip"→"更高熵/更好性能"串成闭环，并用熵/KL/梯度曲线佐证。
- 注意：增益幅度任务相关（消融里 +1.5%，主表 R1-Distill 上 +4.1%），4.1% 这一最大增益来自特定基座，不宜泛化为普遍幅度。

#### 11. 残留问题 / 局限
- 评测局限于数学/几何推理，未覆盖代码、通用对话、长 CoT agentic 等；与 GRPO 对比共享 Dr.GRPO 设置但 clip 范围本就不同，公平性部分依赖调参。
- 几何平均隐含"序列内 token 同等重要"的假设，对极端长序列或含大量格式 token 的响应是否最优未深究。
- 与 GSPO（序列级长度归一化重要性比）思想高度相通，论文未与之直接对照。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 规范仓：https://github.com/callsys/GMPO（本地已 clone ~57MB，Tier A）。
- 专用实现：**Oat**(`oat-llm==0.1.3.post1`) + **vLLM 0.8.4**，构建于 `understand_r1_zero_main/`（Dr.GRPO）；入口 `bash scripts/qwen2.5-math-7b-gmpo.sh`（另含 hmpo 与 ablation 脚本）。
- 另已集成入官方 **veRL** `examples/gmpo_trainer`。〔核对：专用实现框架确为 Oat/Dr.GRPO，veRL 仅作额外集成入口。〕


---


### gspo — Group Sequence Policy Optimization (GSPO)

> **一句话重点 (TL;DR)**：GRPO 在 **token 级**做重要性采样校正是"病态"的——单样本权重无法完成分布校正，反而注入随序列累积、被 clip 放大的高方差噪声，导致大模型（尤其 MoE）不可逆崩溃。GSPO 改在 **序列级**定义（长度归一化的）重要性比并做序列级裁剪/优化，使优化单位与奖励单位（整条序列）对齐，训练更稳更高效，已用于 Qwen3。

**元信息**：arXiv 2507.18071（v2, 2025-07-28）｜ Qwen Team, Alibaba（通讯 Chujie Zheng / Bowen Yu）｜ 2025-07 预印本 ｜ 主题 T3（RL 算法），相关性 High（影响本课题 RL 阶段稳定性与算法选型）｜ 代码 无独立官方仓（集成于 veRL/TRL/ms-swift/ROLL）｜ 框架 Qwen 内部 RL（训练 Megatron + 推理 SGLang/vLLM）。

#### 1. 相关工作与进展
- **PPO**：依赖与策略同规模的 value 模型，显存/计算重，且 value 估计可靠性与长序列扩展性是难点。
- **GRPO（DeepSeekMath, Shao 2024）**：用组内相对优势替代 value 模型，token 级重要性比 + 裁剪，成为竞赛数学/代码 RL 的 SOTA 基线，但在超大模型上不稳。
- **背景**：RL 已是扩展 LLM 深度长链推理的关键范式（o1、DeepSeek-R1、Qwen3）；继续 scale 的前提是训练动态稳定。

#### 2. 现有工作存在的问题
作者诊断 GRPO 目标"病态(ill-posed)"，根因是**重要性采样权重的误用**：
- 重要性采样要在行为分布上对**多样本(N≫1)** 取平均，权重 πtar/πbeh 才能完成分布校正（式 4）。
- GRPO 在**每个 token 位置**用单一样本 yi,t 算比值 πθ/πθold，无法完成校正，反而注入**高方差噪声**；噪声随序列变长累积、被 clip 放大，最终引发**常不可逆的模型崩溃**（回滚 checkpoint、调 clip、改 query 均无效）。
- **MoE 雪上加霜**：一次梯度更新后同一响应约 10% 激活专家变化（越深越显著），使 token 级比值剧烈抖动、进一步失效。
- 核心原则：**优化目标的单位应与奖励的单位匹配**——奖励授予整条序列，却在 token 级做 off-policy 校正，是问题所在。

#### 3. Motivation
token 级重要性权重既病态又与序列级奖励错配；而序列级重要性比 πθ(y|x)/πθold(y|x) 有清晰理论含义（反映采样响应偏离当前策略的程度），与序列级奖励天然对齐，也能作为 clip 的有意义指标。故放弃 token 级目标，直接在序列级做加权、裁剪与优化。

#### 4. 主要灵感 / 核心直觉
- "奖励是整条序列给的，优化和裁剪也应以整条序列为单位。"
- 对序列似然比做**长度归一化（几何平均）**，把 si 控制在统一数值范围、降低方差，避免少数 token 似然变化造成序列比剧烈波动，也省去为不同长度响应设不同 clip 范围。

#### 5. 主要解决思路(一段话讲清核心)
对一个 query 的一组 G 个响应，逐响应计算**序列级重要性比** si(θ)=(πθ(yi|x)/πθold(yi|x))^{1/|yi|}（即 token 对数比均值的 exp）；优势 Âi 用组内标准化（与 GRPO 同）；目标对整条响应做 min/clip（式 5）。本质区别（梯度分析 §4.2）：GRPO 用各 token 各自不等的重要性权重加权 log 似然梯度，累积致不稳；GSPO 对一条响应内**所有 token 等权**，消除该不稳定因素。

#### 6. 方法详解(通俗、分步骤)
1. 对 query x 从 πθold 采样 G 个响应；verifier 给每条 [0,1] 奖励。
2. 组内标准化得优势 Âi=(ri−mean{r})/std{r}（式 6，组内所有 token 共享同一优势）。
3. 算序列级比 si(θ)（式 7，含 1/|yi| 长度归一化）。
4. 按式 5 做 `min(si·Âi, clip(si,1−ε,1+ε)·Âi)`，对**整条响应**裁剪以排除过度 off-policy 样本。注意 clip 范围与 GRPO 差一个数量级（实验左右取 3e-4/4e-4，GRPO 为 0.2/0.27）。
5. **GSPO-token 变体（§4.3，式 13-14）**：用于多轮 RL 等需逐 token 调优势的场景，`si,t=sg[si]·πθ(yi,t)/sg[πθ(yi,t)]`。当所有 token 优势相同时，与 GSPO 在目标、clip 条件、理论梯度上**数值完全等价**，但允许逐 token 自定义优势。

#### 7. 实验数据集
- 评测：AIME'24（32 采样 avg Pass@1）、LiveCodeBench（2024-10~2025-02，8 采样 avg Pass@1）、CodeForces（Elo）。
- 训练基座：Qwen3-30B-A3B-Base 冷启动微调模型；每批 rollout 切 4 个 mini-batch 做梯度更新。
- 实战：GSPO 已用于最新 Qwen3 系列的 RL 训练。

#### 8. 实验结果与主要发现
- **稳定+高效（Fig.1）**：GSPO 全程稳定；同算力/同消耗 query 下训练精度与基准表现优于精调的 GRPO；可靠地随增算力/更新 query/延长生成长度持续提升。
- **MoE（§5.3）**：GRPO 须依赖 **Routing Replay**（缓存并重放 πθold 的激活专家）才能收敛，带来额外显存/通信开销并限制 MoE 容量；GSPO 只看序列似然、对单 token 似然不敏感，**无需 Routing Replay** 即可稳定收敛。
- **裁剪悖论（§5.2）**：GSPO 被裁 token 比例约 0.15，比 GRPO（≈0.0013）高两个数量级，但训练效率反而更高——反证 GRPO token 级梯度估计噪声大、样本利用低效。
- **基础设施（§5.4）**：仅用序列级似然，对训练/推理引擎精度差异更鲁棒，可直接用推理引擎(SGLang/vLLM)返回的似然优化，省去用训练引擎(Megatron)重算 πθold，利好 partial rollout、多轮 RL、训推分离框架。

#### 9. 结果如何支撑其主张
- "token 级病态"由崩溃现象 + 梯度分析 + 裁剪悖论三方面支撑；"序列级更稳"由 MoE 免 Routing Replay 与稳定训练曲线支撑。论证既有理论（重要性采样原理、梯度推导）又有大规模实证（Qwen3）。

#### 10. 逻辑自洽性(中性评估)
- 推理链清晰：从重要性采样基本原理 → token 级误用 → 序列级修正，理论自洽。
- GSPO 与 GSPO-token 的数值等价性给出严格推导。
- 隐患：论文给的训练曲线横轴是抽象"Training Compute"、纵轴含归一化，缺绝对数值与方差带；"崩溃不可逆"为定性描述，未给可复现的崩溃判据。

#### 11. 残留问题 / 局限
- 实验主要在 Qwen3-30B-A3B-Base 一个基座族；与 GRPO 的对比依赖各自精调的 clip 范围，公平性依赖调参。
- 序列级长度归一化对极端长/短响应、混合长度分布的鲁棒性未深入。
- 无公开代码与脚本随论文发布，复现需到第三方框架对照实现，细节（如奖励、数据筛选）披露有限。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- **无独立官方仓**；论文为纯算法贡献。
- 已被多个开源 RL 框架集成：veRL、TRL、ms-swift、ROLL（代码实现需到这些框架查阅）。原始实验在 Qwen 内部基础设施（Megatron 训练 + SGLang/vLLM 推理）。


---


### hapo — Heterogeneous Adaptive Policy Optimization: Tailoring Optimization to Every Token's Nature

> **一句话重点 (TL;DR)**：现有 RLVR 对所有 token 一视同仁，违背语言生成的异质本质。HAPO 把 **token 熵当作贯穿采样/优势/裁剪全流程的连续优化驱动量**（而非离散过滤或事后正则），用四个组件对每个 token 做细粒度差异化处理，在多模型规模上一致优于 DAPO。

**元信息**：arXiv 2509.16591（v2, 2026-04-29）｜ 北京大学 / 上海 AI Lab / 北航（共一 Zheng Liu、Mengjie Liu；通讯 Wentao Zhang，README 另标 Lijun Wu）｜ Preprint ｜ 主题 token 异质性驱动的 RLVR（on-policy RL，对标 DAPO），与 OPD 关联**间接**（"token 级熵信号做细粒度加权"思路可迁移到蒸馏的 token 级加权，但本身非蒸馏）｜ 代码 https://github.com/starriver030515/HAPO（已 clone ~8.6MB）｜ 框架 **verl + vLLM**。

#### 1. 相关工作与进展
- RLHF/RLVR 是提升 LLM 推理的核心（o1、DeepSeek-R1、Qwen3）。
- 用熵区分 token 角色的近期工作：DAPO-with-forking-tokens（少数高熵 token 引导优化）、Archer（高/低熵两组，对高熵放宽 clip）、Entropy-Adv / EDGE-GRPO（用熵调优势）、80/20 规则（高熵少数 token 驱动 RL）。

#### 2. 现有工作存在的问题
现有算法对所有 token **统一优化**，无法区分关键推理路径 token 与常规模式 token；已有用熵的方法只把熵当**离散过滤器或事后 bonus**，核心优化机制本身未变。三处具体痛点：
- **采样**：温度困境——低温保精度但压制关键(高熵)token，高温增其出现却引入噪声；高熵关键 token 本就稀少。
- **优势**：序列级优势忽略序列内 token 异质；DAPO token-mean loss 下长负样本主导梯度。
- **裁剪**：低熵 token（多为格式符）易撞左界、阻止其降概率；高熵 token（关键推理）易撞右界、限制探索——统一裁剪"保护噪声、约束探索"。

#### 3. Motivation
把 entropy 从"辅助调节器"提升为"核心优化驱动量"：将采样/优势/裁剪的优化参数表达为 **token 熵的连续函数**，在每一阶段嵌入 token 级细粒度处理，实现连续（而非离散分组）的异质优化。

#### 4. 主要灵感 / 核心直觉
- 语言生成本质异质：少数高熵 token 是"决策分叉点"（关键推理），多数低熵 token 是"既定模式"（格式/连接词）。
- 二者应被反向对待：高熵处放开探索、低熵处允许激进降噪——统一优化恰好做反了。

#### 5. 主要解决思路(一段话讲清核心)
以 token 熵为连续信号，在 RLVR 全流程嵌入四个组件：(1) 采样时按熵实时调温；(2) 优势用 token 级组平均（全局均值/方差归一）；(3) 用熵+重要性比对优势做差分重分配；(4) 反向非对称自适应裁剪（低熵扩左界、高熵扩右界）。先做系统实证（熵-频率 landscape、熵与训练动态、温度-精度曲线）论证三痛点，再逐阶段嵌入。

#### 6. 方法详解(通俗、分步骤)
1. **Adaptive Temperature Sampling**：按 token 熵实时调采样温度（高熵升温促探索、低熵降温保连贯），解采样阶段精度-探索权衡。〔code: `rollout.adaptive_temperature=True`, `temperature_tau=0.05`〕
2. **Token-Level Group Average Advantage**：组内按 token 级估优势 A_{i,t}=(a_{i,t}−µ_glob)/σ_glob，兼顾序列长度（保留 token-mean 对长序列的好处）且对正负样本无偏。〔code: `adaptive_advantage=True`〕
3. **Differential Advantage Redistribution**：用熵+重要性比调制优势——对高熵且比值极端的 token 放大优势、对低熵接近 1 的抑制，做细粒度信号归因。
4. **Asymmetric Adaptive Clipping**：反向非对称裁剪界——低熵 token 扩左界（允许激进降噪声概率），高熵 token 扩右界（关键决策点放开探索）。〔code: `adaptive_clip=True`, `clip_alpha=1.0`, `entropy_pivot=0.8`〕

#### 7. 实验数据集
- **任务**：数学、代码、逻辑三类，跨多模型规模。
- **数学基准（Table 1/3）**：AIME24(30)、AIME25(30)、AMC(83)、Math(500)、OlympiadBench(675)、Minerva(272)，六基准均值；模型 Qwen2.5-Math-1.5B/7B、Qwen3-8B（recipe 另含 qwen3_14b、llama3.1-8B-Instruct、llama3.2-3B-Instruct）。
- **代码/逻辑**：Logic RL(4–7 ppl 子集)、LiveCodeBench(V5: 24/8-25/2; V6: 25/2-25/5)，模型 Qwen2.5-7B-Instruct-1M。训练数据 dapo-math-17k。
- **基线**：Vanilla GRPO/DAPO、DAPO w/ Forking Tokens、Archer、Entropy Env、EDGE-GRPO；主对照 DAPO 系。

#### 8. 实验结果与主要发现
- **主结果（均值）**：Qwen2.5-Math-7B HAPO **50.04 vs DAPO 46.97**；Qwen2.5-Math-1.5B **40.62 vs 38.34**；Qwen3-8B 亦优于各基线，跨规模一致优于 DAPO。
- **逐组件消融（Table 4，A=自适应温度采样 / B=token 级组平均 / C=差分优势重分配 / D=非对称裁剪）**：单加 A 增益最大（46.97→48.85），B/C/D 单加各达 48.56/48.28/48.02，四件齐备 50.04；作者称 A 最关键（主导 token 分布与熵）。

#### 9. 结果如何支撑其主张
- "异质优化优于统一优化"由跨规模一致超 DAPO 支撑；"熵作核心驱动量"由消融中四组件各有正贡献、且自适应温度采样(A)主导支撑；三处痛点先有实证 landscape 再被对应组件解决，论证结构清晰。

#### 10. 逻辑自洽性(中性评估)
- 四组件都以 token 熵为统一信号，主线自洽；代码 recipe 与论文四组件一一对应（adaptive_temperature / adaptive_advantage / adaptive_clip / entropy_pivot）。
- 保留意见：(1) 四组件叠加相对 A+B+C(49.42) 再增约 0.6，单组件间差异不大，需注意是否含调参收益；(2) "连续 entropy 函数"相比 Archer 离散分组的本质增益、迁移到非 DAPO 基线的稳健性，论文未单独验证；(3) token 级超参遵循 80/20，熵分位 ρ=80%（entropy_pivot=0.8，取 top-20% 高熵 token）。

#### 11. 残留问题 / 局限
- 仅在 DAPO 基线上验证四组件协同；对其他 RL 基线（GRPO/GSPO 等）的可迁移性未测。
- 引入多个 token 级超参（温度 τ、clip_alpha、entropy_pivot 等），调参面变大，部分增益可能来自调参而非机制本身。
- 与 OPD/蒸馏仅间接相关——其 token 级熵加权思想可借鉴，但本文是 on-policy RLVR，不涉及 teacher 监督。
- **工程瑕疵**：README 的 arXiv badge/链接仍是占位符 `1234.12345`，badge 文字写 2507.21848，但 README 内 BibTeX 与 HF collection 均指向真实 ID **2509.16591**（与本标头一致）。〔已核〕

#### 12. 开源代码与框架(链接+框架+代码可得性)
- https://github.com/starriver030515/HAPO（已 clone ~8.6MB；含 `verl/`、`recipe/`、`vllm/`、`pyproject.toml`）。
- 框架 **verl + vLLM**；四组件作为 verl 训练流程的配置项接入（见 `recipe/qwen2.5_math_7b.sh` 等）。HF 有对应模型 collection。


---


### holderpo — Hölder Policy Optimisation

> **一句话重点 (TL;DR)**：把 GRPO 系 RLVR 里"token 级重要性比如何聚合成序列级标量"统一为 Hölder p-mean（p=1→GRPO、p→0→GMPO/GSPO），并沿训练时间退火 p，以单参数平衡"梯度集中放大稀疏信号"与"梯度方差受控"这一无法被任何固定算子同时兼得的 trade-off。

**元信息**：arXiv 2605.12058v2（2026-05-21）｜ UCL / 上海交大 / 港科大(广州)，通讯 Jun Wang (UCL)｜ Preprint（未评审）｜ 主题 token 级聚合算子的统一框架，与 OPD/MTP 关联较弱（纯 RL 聚合层面，可借鉴 token 加权视角）｜ 代码 https://github.com/YihangChen9/HolderPO （已 clone，约 8.8MB，与论文一致）｜ 框架 oat (sail-sg/oat, oat-llm 0.1.3.post1) + vLLM 0.8.4。

#### 1. 相关工作与进展
GRPO（Shao et al. 2024）用组内采样轨迹估优势、无需 critic，推动了 DeepSeek-R1 等推理模型。把轨迹级优势映射到策略更新时，需将序列内 token 级重要性比 `r_{i,t}=π_θ/π_{θ_old}` 聚合成序列级标量：GRPO 用算术均值（p=1）、GMPO/GSPO（Zhao et al. 2025）用几何均值（p→0）。并发工作 PMPO（Zhao et al. 2026）也在调聚合算子。

#### 2. 现有工作存在的问题
固定聚合算子施加静态优化 landscape，出现临界 trade-off：稠密信号任务（监督分散在大量 token，如 MATH）下 GRPO（p=1）过度放大微小 token 误差→高方差梯度→训练坍塌；稀疏信号任务（正确性集中在罕见高幅 token，如 AIME）下 GSPO（p→0）过度平滑、压制罕见"aha moment"。无单一静态 p 兼得两端——实测 AIME24 在 p=3 峰值、MATH500 在 p=−1 峰值。

#### 3. Motivation
最优聚合是"任务信号密度 × 训练进程"的函数，因此需要一个能连续调控、并可随训练动态变化的聚合算子，而非在 GRPO 与 GSPO 间二选一。

#### 4. 主要灵感 / 核心直觉
用 Hölder mean（p-范数）把所有均值型聚合统一为单参数 p∈ℝ 的连续谱，并把 p 扩到全实轴——发现 p<0 是一个先前未探索的"逆向集中 (inverse-concentration)"相位（把梯度权重集中到最小比 token，即模型"犹豫"处）。早期用高正 p 激进放大稀疏信号、后期退到负 p 收紧方差，即可在训练生命周期内动态走完 trade-off 两端。与 PMPO 的区别：(i) p 扩到全实轴含 p<0；(ii) 沿训练时间轴（跨 step）而非按轨迹自适应 p。

#### 5. 主要解决思路(一段话讲清核心)
把 GRPO 目标中对 token 级重要性比的算术均值替换为 Hölder p-mean `ρ_{i,p}=((1/|y_i|)Σ_t r_{i,t}^p)^{1/p}`，套上 PPO 式序列级 clip 形成目标；p 是连续旋钮，p→0 取几何均值（极限）恢复 GSPO，p=1 恢复 GRPO。理论上证明大 p 集中梯度权重以放大稀疏信号（代价方差界变松）、小/负 p 严格收紧梯度方差（代价削弱稀疏响应）。再用一个沿训练从高正值退火到负值的调度，无额外计算开销地兼顾两端。

#### 6. 方法详解(通俗、分步骤)
- **Hölder 聚合**：`ρ_{i,p}(θ)=((1/|y_i|)·Σ_t r_{i,t}^p)^{1/p}`（p≠0），p=0 取几何均值。目标用序列级 clip：`J=E[ min(ρ_{i,p}·Â_i, clip(ρ_{i,p},1−ε,1+ε)·Â_i) ]`，以控梯度方差。
- **梯度集中（Thm 1）**：per-token 梯度权重 `W_{i,t}(p)=r_{i,t}^p/Σ_k r_{i,k}^p` 构成概率分布；其 Shannon 熵在 p=0 取全局最大（均匀），|p| 增大严格下降；p→+∞ 集中到最大比 token（上向集中），p→−∞ 集中到最小比 token（下向集中，放大模型犹豫处的非常规有效决策点→促进多样性）。
- **方差界（Thm 2）**：给出 `‖Var(∇J)‖` 上界，刻画"集中度↑→方差↑"的风险。
- **动态退火**：p 从高正值（早期激进信号放大）线性/分段调度到负值（后期方差受控收敛）。

#### 7. 实验数据集
- 数学（基模 **Qwen2.5-Math-7B**，另覆盖 1.5B–8B 多基模）：AIME、AMC、MATH500、Minerva、OlympiadBench 五基准；亦报 R1-Distill-Qwen-7B。
- 智能体：**ALFWorld**（开放世界 agentic，基模 Qwen2.5-Instruct-1.5B，agentic 分支）。

#### 8. 实验结果与主要发现
- 五数学基准平均 **54.9%**（Qwen2.5-Math-7B，linear 2→−2 schedule），对 GRPO 51.2 相对 +7.2%，超并发 PMPO 54.2、超 GMPO 52.7；R1-Distill-Qwen-7B 上同 schedule 达 66.4 avg。
- 固定 p=3 即把 AIME 记录从 43.3% 推到 46.7%（先验证 p 的任务敏感性，再用退火统一两端）。
- ALFWorld 用 **1→−1** schedule 取得 **93.8%** 成功率，对 GRPO 72.8% 相对 **+28.8%**；而数学用的 2→−2 schedule 在此仅 87.5%——印证"退火端点须按基模成熟度/任务信号密度标定"（Qwen2.5-Instruct-1.5B 缺域内预训练故偏好保守上限）。

#### 9. 结果如何支撑其主张
"无银弹"主张由 p 扫描曲线（AIME24 峰在 p=3、MATH500 峰在 p=−1）直接支撑；"动态优于静态"由退火达 54.9（超任何单一固定 p 的横向对照）支撑；ALFWorld 上换 schedule 才达最优、错配 schedule 掉点，反向印证"端点需标定"。理论 Thm 1/2 给出集中-方差 trade-off 的形式刻画，与经验现象自洽。

#### 10. 逻辑自洽性(中性评估)
框架自洽：单参数 p 把已有算子作为特例统一，理论（熵单调、方差界）与经验（p 扫描、退火）相互印证，代码实现与公式一致。但"动态优于静态"的因果归因部分被超参选择稀释——见 §11。

#### 11. 残留问题 / 局限
- 主结论建立在单一基模 Qwen2.5-Math-7B，跨架构/规模泛化未充分验证（虽宣称覆盖 1.5B–8B，核心对照集中在 7B）。
- "p<0 逆向集中促进多样性"理论叙事直观，但其经验收益与退火日程（起止值、schedule 形状）强耦合——数学用 2→−2、ALFWorld 用 1→−1，本质是需调的额外超参；论文将其包装为"无额外开销"略乐观。
- Preprint（2026-05），未经评审。
- 〔已核-代码〕`train_zero_math_holder.py` 含 `holder_p_schedule`(constant/linear/quad)、`holder_p_min/max`、`_get_current_holder_p`，loss 为 `ρ=((1/|y|)Σr_t^p)^{1/p}`、p→0 取几何均值、序列级 PPO clip，与 §6 一致。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 链接：https://github.com/YihangChen9/HolderPO （已 clone，约 8.8MB；README 标题/作者/摘要与论文一致；另有 `agentic` 分支跑 ALFWorld）。
- 框架：oat (sail-sg/oat) + vLLM 0.8.4，内含 `understand_r1_zero_main`（Understanding-R1-Zero / Dr.GRPO 系）子包。
- 入口：`train_zero_math_holder.py`，启动脚本 `scripts/qwen2.5-math-7b-holder.sh`，p 调度经环境变量旋钮配置。代码可得、可复现性良好。


---


### hpt_upge — Towards a Unified View of Large Language Model Post-Training (UPGE / HPT)

> **一句话重点 (TL;DR)**：提出 Unified Policy Gradient Estimator (UPGE)，把 SFT 与各类 RL 后训练的策略梯度分解为四个可互换组件，论证 SFT 与 RL 是同一优化过程在不同数据分布假设/bias-variance 权衡下的实例；据此提出 Hybrid Post-Training (HPT)——按 on-policy rollout 准确率在实例级二值切换"纯 RL"与"纯 SFT"信号。

**元信息**：arXiv 2509.04419（2025-09-05）｜ 清华大学 C3I (TsinghuaC3I)｜ arXiv 预印本，2025-09｜ 主题 T3：统一 SFT-RL 视角的理论 + 算法（GFT-class 代表作）｜ 代码 https://github.com/TsinghuaC3I/Unify-Post-Training （含 `hpt/verl/verl/mix_src` 核心实现，约 11MB）｜ 框架 veRL + LUFFY 的 mix_src 扩展（FSDP + vLLM rollout）。

#### 1. 相关工作与进展
现代 LLM 后训练有两类数据来源：on-policy（模型自生成 rollout）与 off-policy（人类/其他模型示范）。RL 善探索、SFT 善高效利用示范，二者通常被视为对立范式，常以 SFT→RL 两阶段串联使用。LUFFY 提供了 on/off-policy 混合 rollout（prefix_mask 区分）的工程基础，本工作在其 mix_src 上扩展。

#### 2. 现有工作存在的问题
长期缺乏把 SFT 与 RL 统一理解的理论框架；SFT→RL 两阶段范式存在遗忘与效率问题，且无法按模型能力/数据难度自适应地选择训练信号。

#### 3. Motivation
若能把各类后训练算法的梯度统一表达，就能在理论上厘清 SFT 与 RL 的关系，并导出一个按模型当前表现自适应混合两种信号的算法，避免两阶段范式的遗忘与僵化。

#### 4. 主要灵感 / 核心直觉
核心论断：**SFT 与 RL 并非对立，而是同一优化过程在不同数据分布假设与不同 bias-variance 权衡下的实例**。SFT 可看作优势估计 Â 退化、且 off-policy 算法常令参考策略分母 π_ref(τ)=1 的特例。既然同源，就能按"模型在某问题上是否已能自解"动态决定该用 RL（能自解→探索）还是 SFT（解不出→示范）。

#### 5. 主要解决思路(一段话讲清核心)
把广泛后训练算法的策略梯度写成 `grad_Uni = 𝟙_stable · (1/π_ref) · Â · ∇π_θ` 四组件（稳定化掩码、参考策略分母、优势估计、似然梯度），不同算法对应不同取值；据此设计 HPT：对每个 question 采 n=8 条 on-policy rollout，用 rule-based verifier 得准确率 P，P>γ 走纯 on-policy RL（Dr. GRPO）、P≤γ 在外部 supervising trajectory 上走纯 SFT，混合损失 `L=αL_RL+βL_SFT`（α,β 为二值开关）。

#### 6. 方法详解(通俗、分步骤)
**UPGE 四组件**：(1) Stabilization mask 𝟙_stable（如 PPO clip）；(2) Reference-Policy denominator 1/π_ref（off-policy SFT 类常令 π_ref(τ)=1）；(3) Advantage estimate Â（SFT 为 Â 退化特例）；(4) Likelihood gradient ∇π_θ。
**HPT 二值开关（Algorithm 1, Eq.10-13）**：每 question 采 n=8 rollout 得 P；`P>γ→(α,β)=(1,0)` 纯 RL；`P≤γ→(α,β)=(0,1)` 在 τ⋆ 上纯 SFT。gate γ：Qwen 系固定 **γ=0**（仅 8 条全错才转 SFT），LLaMA 系 **γ=2/8**。RL 原语论文正文明确为 **Dr. GRPO**（不除组内 std、去 token-mean 长度偏置）。
〔已核-代码〕工程实现 `select_on_off_ada_balance` 按每 prompt 正确 rollout 数 `on_solve_num`：`≤switch_gate`(默认0)→移除 on-policy、注入示范走 SFT；`switch_gate<…≤switch_gate_off`→过渡；更高→纯 on-policy RL。actor 端（`mix_actor.py`）对 off-policy 部分用 `compute_sft_pure_loss`，以 `sft_loss_coef`（默认1.0，Qwen2.5-Math-1.5B 用0.3）与 pg_loss 相加：`loss=sft_loss·sft_loss_coef+pg_loss`。仓库另含 `off_policy`/`off_sft`/`switch_off_sft`/`srft`(`sft_coef=0.5·exp(-H_coef)`) 等变体。仓库 `adv_estimator` 默认写 `grpo`，与正文 Dr. GRPO 通过 `loss_remove_token_mean`/`loss_remove_clip` 旋钮区分。

#### 7. 实验数据集
- 训练：遵循 LUFFY，使用 OpenR1-Math 数据（脚本默认 `openr1.parquet`）；Qwen2.5-Math-7B 需 rope_theta 重设 40000、max_position_embeddings 设 16384。
- Backbone：Qwen 与 LLaMA 多规模（主结果 Qwen2.5-Math-7B，另含较小/较弱模型）。
- 评测：In-Distribution（AIME24/AIME25/AMC/MATH-500/Minerva/Olympiad）+ Out-of-Distribution（ARC-c、GPQA）。
- 基线：SFT、GRPO、SFT→GRPO、LUFFY、SRFT 等。

#### 8. 实验结果与主要发现
- HPT 在 Qwen2.5-Math-7B 上超过 SFT→GRPO 与 LUFFY，相对最强基线在整体上提升约 **7 个点**（论文摘要 "a 7-point gain over our strongest baseline"）。
- gate 消融（Qwen2.5-Math-7B）：γ=0 → 平均 **41.9**，优于 γ=1/8 的 38.7 与 γ=2/8 的 39.0——即 Qwen 上"仅在全错时才转 SFT"最优。
- 在较小/较弱模型上也有显著提升。

#### 9. 结果如何支撑其主张
"SFT/RL 同源"由 UPGE 的梯度分解（把已有算法表为四组件特例）从理论支撑；"自适应混合优于两阶段"由 HPT 超过 SFT→GRPO/LUFFY 支撑；gate 消融显示切换阈值确实影响结果且存在最优值，间接印证"按模型表现选信号"的有效性。

#### 10. 逻辑自洽性(中性评估)
理论与算法衔接自洽，代码与 Algorithm 1 一致（二值开关 + sft_loss_coef 加权）。需注意：论文正文 HPT 是二值开关而非连续混合，"hybrid"更多体现在实例级在两种信号间切换，而非单样本上加权融合；UPGE 四组件框架与既有统一视角工作（如本批 hpd 的 reweighted log-likelihood）思路相近，新颖性主要在四组件的明确分解 + 基于准确率的自适应 gate。

#### 11. 残留问题 / 局限
- gate γ 需按模型族手调（Qwen 0、LLaMA 2/8），`sft_loss_coef` 亦随模型变（1.0 vs 0.3），自适应性仍含人工先验。
- n=8 rollout 估准确率带来额外采样开销；γ 离散且粒度粗（0/1/8/2/8）。
- 主结果集中在数学推理 + Qwen2.5-Math-7B，跨域泛化以 ARC-c/GPQA 两个 OOD 点为主。
- arXiv 预印本（2025-09），未见正式接收。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 链接：https://github.com/TsinghuaC3I/Unify-Post-Training （约 11MB）。核心在 `hpt/verl/verl/mix_src/`（`mix_actor.py`、`mix_core_alg.py`、`mix_trainer.py`），训练入口 `verl.mix_src.main_mix_ppo`。
- 框架：veRL + LUFFY mix_src 扩展，FSDP + vLLM rollout，prefix_mask 区分 on/off-policy。
- 复现：提供 `train.sh`、`train_luffy.sh`、`train_srft.sh`、`train_llama.sh` 及数据准备脚本 `data/prepare_train_sft_rl.py`、`hpt/scripts/data/prepare_openr1_data*.py`。代码可得、可对照。


---


### limit_rlvr — Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?

> **一句话重点 (TL;DR)**：用大 k 的 pass@k 系统度量 base vs RLVR 模型的推理"能力边界"，发现 RLVR 只在小 k 占优、大 k 被 base 反超，且边界随训练收窄——结论是当前 RLVR 主要提升采样效率、并未引入超越 base 的新推理能力，而蒸馏可以。

**元信息**：arXiv 2504.13837 ｜ 清华大学 LeapLab、上海交通大学(Yang Yue 乐洋/Zhiqi Chen 共一，Gao Huang 黄高通讯) ｜ 2025-04-21 v1，2025-11-24 v5(新增 DAPO/DeepScaler、熵/温度分析) ｜ 主题 T3 RLVR 能力边界批判性分析(相关性 High) ｜ 代码 https://github.com/LeapLabTHU/limit-of-RLVR (已克隆 ~83MB，评测/分析为主) ｜ 框架 pass@k 评测(vLLM 多采样)，被评模型来自 SimpleRL-Zoo/Oat-Zero/DAPO/Code-R1，VeRL

#### 1. 相关工作与进展
o1、DeepSeek-R1、Kimi-1.5 等推理模型核心驱动是大规模 RLVR(可验证奖励：数学答案正确 / 代码通过单测)。业界类比 Atari/Go 中 RL 能自主发现超越人类的新策略，普遍相信 RLVR 同样能让 LLM 自主发展出超越 base 的新推理模式(枚举、自反思、迭代精炼)，被视为持续自进化路径。

#### 2. 现有工作存在的问题
RLVR 的根本有效性尚未被充分检验。传统 greedy/nucleus 的平均分(pass@1)只反映平均情形、会低估模型潜力，无法刻画推理能力**边界**。核心追问：**当前 RLVR 是让 LLM 获得了 base 没有的新推理能力，还是只是更高效地利用了 base 已有的推理模式？**

#### 3. Motivation
用 pass@k(尤其大 k)严格度量 base 与 RLVR 模型的推理能力边界，判断 RLVR 是否带来"超越性"能力；并把 base 当作上界，量化现有 RLVR 算法距最优有多远，以及 RLVR 与蒸馏的本质差异。

#### 4. 主要灵感 / 核心直觉
pass@k(k 个采样中任一正确即解出)在大 k 时逼近"模型潜在可解题集合"，因此是推理边界的代理。若 RLVR 真扩展能力，其 pass@k 曲线应在所有 k 上不低于 base；若只是把概率质量集中到 base 已能解的路径上，则会在小 k 占优、大 k 被 base 反超——这正是作者用来证伪"RLVR 引入新能力"的判别工具。

#### 5. 主要解决思路(一段话讲清核心)
不提出新训练方法，而是把 base 当上界，系统比较 base 与其多种 RLVR 后版本在多家族/多尺寸/多算法/多 benchmark 上的 pass@k 曲线；辅以覆盖度与 perplexity 分析验证 RLVR 路径是否已在 base 采样分布内；定义采样效率差距 ΔSE 量化各算法逼近最优的程度；并对比蒸馏以区分二者本质。

#### 6. 方法详解(通俗、分步骤)
- **评测指标**：pass@k(无偏低方差估计器)；主张 pass@k(而非 Best-of-N / 多数投票)才反映"边界"。数学对 guessing 风险题人工核查 CoT 正确性，代码用编译器+单测。
- **量化指标 ΔSE**：sampling efficiency gap = RL 模型 pass@1 与 base 模型 pass@k(k=256 作上界代理)之差，衡量算法逼近最优程度(§Fig.8 top)。
- **覆盖度 + perplexity 分析**：验证 RLVR 生成的推理路径是否已存在于 base 的采样分布中。
- **算法对比**：六种主流 RLVR 算法 = **PPO、GRPO、Reinforce++、RLOO、ReMax、DAPO**(均在 VeRL 上、以 Qwen2.5-Math-7B 为 base、Omni-Math-Rule 训练，依 DAPO/Oat-Zero 去掉 KL；§4.3)。〔已核-Table 1/§4.3〕

#### 7. 实验数据集
- 数学(zero-RL，base 起点)：LLaMA-3.1-8B、Qwen2.5-7B/14B/32B-Base、Qwen2.5-Math-7B;benchmark GSM8K、MATH500、Minerva、Olympiad、AIME24、AMC23。
- 代码(instruct 起点)：Qwen2.5-7B-Instruct、DeepSeek-R1-Distill-Qwen-14B，用 Code-R1。
- 视觉推理：另有设置。
- v5 新增：评测 Oat-Zero-7B、DAPO-32B(Fig.11)、DeepScaleR，及熵/温度匹配分析。

#### 8. 实验结果与主要发现
1. **小 k 时 RLVR 优于 base，但大 k 时 base 持续反超**——所有 benchmark 与家族无例外，说明当前 RLVR 不扩展、甚至缩小可解题目范围(推理边界变窄)；随训练进行 pass@256 覆盖度下降(Fig 中 GRPO-step150/300/450 递减)。
2. RLVR 生成的推理路径已存在于 base 输出分布(perplexity 分析佐证)——RLVR 只是更高效采样 base 已能解的题，能力受 base 上界约束。
3. 六种 RLVR 算法 ΔSE 一致偏大(in-domain 从 GRPO 43.9 到 RLOO 最优 42.6，差异小)，均远未最优;DAPO 在 k=256 处显著下滑。
4. **RLVR 与蒸馏本质不同**：蒸馏能从更强 teacher 引入新推理模式、真正扩展边界；RLVR 不能。
5. v5：提高 RLVR 模型生成温度至匹配 base 熵后，pass@k 差距缩小但 base 仍占优——熵下降是边界收窄的部分(非全部)原因。

#### 9. 结果如何支撑其主张
跨多家族×尺寸×算法×benchmark 的 pass@k 反超现象高度一致，强力支撑"RLVR 不扩展边界"的核心主张;perplexity/覆盖度分析对"路径已在 base 分布内"提供机制证据;蒸馏对照实验支撑"蒸馏≠RLVR"。论据系统、可重复，是该批判的最强支撑。

#### 10. 逻辑自洽性(中性评估)
论证自洽：pass@k 作为边界代理的合理性、k=256 上界代理的选择、无偏估计器、人工核查 guessing 题，环环相扣。潜在张力在于"大 k 反超"对采样预算敏感(k=256 是工程代理而非真上界)，作者已用熵匹配等鲁棒性检查回应，结论稳健。

#### 11. 残留问题 / 局限
- pass@k 边界以有限 k(≤256)代理"真实潜力"，极大 k 或更优解码下结论是否变化存开放性。
- 结论针对"当前 RLVR 算法/数据规模"；论文自身呼吁更好探索机制、更大数据 curation、细粒度 process signal、多轮 agent 交互——即不排除未来 RLVR 范式突破边界。
- 蒸馏"扩展边界"的对照较简略，蒸馏引入的新能力是否只是 teacher 能力转移(而非 RL 式自主发现)未深究。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 链接：https://github.com/LeapLabTHU/limit-of-RLVR (已克隆 ~83MB;项目页 https://limit-of-RLVR.github.io)。含 `code/`(DeepCoder)、`math/`(eval_math_nodes.sh、pass@k.py、examples、install.sh、requirements.txt)。
- 框架：评测/分析为主，核心是 vLLM 多次采样的 pass@k 评测;被评 RLVR 模型来自 SimpleRL-Zoo、Oat-Zero、DAPO(数学)、Code-R1(代码)，训练侧统一在 VeRL，并非自研新训练框架。
- 可得性：评测脚本完整、可复现;不含新方法实现(本就无新方法)。


---


### lite_ppo — Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning (Lite PPO)

> **一句话重点 (TL;DR)**：在同一框架(ROLL)/同一模型/同一数据下逐一隔离评估 RL4LLM 常用 trick，发现只需"优势归一化(group 均值 + batch 标准差) + token-level loss 聚合"两项极简组合(Lite PPO)即可稳定超过堆砌组件的 DAPO/GRPO——"简单胜过复杂"。

**元信息**：arXiv 2508.08221 (v3, 2025-10-27) ｜ Alibaba(ROLL/Future Life Lab) 联合 北交大、HKUST、南大、北大、Mila 等(Weixun Wang 通讯) ｜ 2025-08(v3 2025-10) ｜ 主题 T3 RL 算法/技巧实证(对选 RL 配方/规避无效组件有直接参考) ｜ 代码 **recipe in alibaba/ROLL**(论文为基于 ROLL 的系统复现，无独立方法仓库) ｜ 框架 ROLL，统一 baseline = PPO loss + REINFORCE 优势(critic-free)

#### 1. 相关工作与进展
2025 年 RL4LLM 爆发，数学/代码任务上 RL 把 LLM 推到预训练之上。但各路工作对同一问题给出**互相矛盾**结论：GRPO 主张 group-level 归一化、REINFORCE++ 主张 batch-level;GRPO 含方差归一化、Dr.GRPO 主张去掉方差;GRPO 用 response-level loss、DAPO 用 token-level loss。trick(Normalization/Clip/Overlong Filtering 等)看似正交、数量繁多。

#### 2. 现有工作存在的问题
- 各类 trick 数量多、看似正交，从业者难在特定场景挑出有效组合。
- 实验设置(训练数据/初始化/规模)不一致导致结论冲突，难判断每个 trick 的真实贡献与适用范围。
- 缺乏标准化使用指南、机制理解碎片化。

#### 3. Motivation
核心追问：现有技巧分别适用什么场景？是否存在简单且可泛化的组合来增强策略优化？遵循经典 RL 机制分析方法论，在**同一开源框架(ROLL)、同一策略模型、同一数据**下复现并隔离评估每个 trick，覆盖不同难度数据、不同模型规模与类型，给出明确选型指南；并验证"简单胜过复杂"。

#### 4. 主要灵感 / 核心直觉
矛盾结论多源于实验条件不可比；只要控制变量、逐一消融，trick 的真实效用会显现，且大概率只有少数核心 trick(归一化、loss 聚合)真正起作用——其余多为针对特定设置的局部修补。

#### 5. 主要解决思路(一段话讲清核心)
以 vanilla PPO loss + 无 critic(REINFORCE 优势)为统一 baseline，在三档难度数据、两种规模(4B/8B)、Base 与 aligned 两类模型上逐一隔离评估 Normalization/Clip/Loss 聚合/Overlong filtering;据此提炼选型指南，并把最稳健的两项(group 均值 + batch 标准差归一化、token-level loss 聚合)整合为 Lite PPO，验证其优于 GRPO/DAPO。

#### 6. 方法详解(通俗、分步骤) —— Lite PPO(两技巧极简组合)
在 **vanilla PPO loss + 无 critic** 基础上仅组合两项：
1. **优势归一化：group-level 均值 + batch-level 标准差(§4.1.2)**。优势减去 group 内均值、再除以整个 batch 的标准差("混合归一化")，比纯 group-std 或纯 batch-std 都更稳健。〔已核-Takeaway②〕
2. **token-level loss 聚合(§4.3.1)**。对 base/非对齐模型尤为有效。
主要实证 Takeaways：①Group-level 归一化在各奖励设置下稳健;②group 均值 + batch 标准差最稳健、batch-std 在大规模奖励下更稳;Clip-Higher 对已对齐模型促进高质量探索，小模型上 clip 上界与性能存在"scaling law"(§4.2);token-level 聚合对 **base** 有效、对已对齐模型提升有限;Overlong filtering 对短-中长推理提升准确/清晰但对长尾收益有限、且限制小模型生成复杂长尾输出(故 Lite PPO 取消它)。

#### 7. 实验数据集
- **训练数据(仅开源)**：SimpleRL-Zoo-Data、DeepMath(-103k)。按 GPT-4o 评估难度分三档各 5,000 条——Easy(SimpleRL-Zoo-Data-Easy，GSM8K/MATH-500-level1)、Medium(DeepMath-103k 最易 5000)、Hard(DeepMath-103k 按难度概率采样)。去除过多 True/False 样本以避免"假阳性"(错误推理链却得对答案)噪声。
- **评测(6 个数学集)**：MATH-500、OlympiadBench、MinervaMath、AIME24、AIME25、AMC23。
- **基座**：Qwen3-4B、Qwen3-8B，各含 Base(非对齐)与 aligned。

#### 8. 实验结果与主要发现
- 统一超参：global batch 1024(rollout batch 128 × 每 prompt 8 响应)，max 响应长度 8192，lr 1e-6;生成 top_p=0.99、top_k=100、temperature=0.99。〔已核-§实验设置〕
- 训练动态：对齐模型初始准确率高、响应更长但后续仅约 +2% 提升。
- 逐一隔离评估 Normalization/Clip/Loss 聚合/Overlong filtering，分别在 Base/aligned、4B/8B 上观察。
- 最终结果：Lite PPO 在六个数学基准上稳定**超越组件繁多的 DAPO**(含 group 归一化、Clip-Higher、Overlong Reward Shaping、token-level loss、Dynamic Sampling)以及广泛使用的 **GRPO**;其他策略常在峰值后崩溃，Lite PPO 持续上升。

#### 9. 结果如何支撑其主张
"简单胜过复杂"由 Lite PPO(2 项) vs DAPO(5 项)/GRPO 的稳定超越直接支撑;各 Takeaway 由对应消融图表支撑，且区分了 Base/aligned 与规模，选型指南有据。统一框架/模型/数据使 trick 归因可信，是该工作的实证强项。

#### 10. 逻辑自洽性(中性评估)
方法论自洽：控制变量消融 + 统一 baseline，结论与消融一致。但"普适选型指南"的外推性受限于单一 LLM family(见局限)，"两项即够"是在 Qwen3+这些数据档位下的结论，非绝对定律——论文表述基本克制。

#### 11. 残留问题 / 局限
- 为公平统一仅用 **Qwen3 系列**初始化，结论可能因 LLM family(预训练/架构差异)而变化——作者自陈。
- 训练数据仅用开源数学集、规模(每档 5000)有限，奖励均为可验证数学奖励;对代码/通用/更大规模是否成立未验证。
- Lite PPO 无独立方法仓库，只作 ROLL recipe;部分 trick 结论(如 Clip 的 scaling law)依赖小模型观察，样本面较窄。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 链接：recipe in https://github.com/alibaba/ROLL。Lite PPO 作为 ROLL 框架内的 recipe 提供;论文本身是基于 ROLL 的系统性复现与实证研究，**无独立方法仓库**。〔注：lite_ppo = recipe in ROLL，paper-only〕
- 框架：所有实验在开源 RL 框架 ROLL(Alibaba LLM RL 平台)上完成;统一 baseline 采用 PPO loss + REINFORCE 优势(critic-free)。
- 可得性：方法以 recipe 形式可得，无单独 release;复现需在 ROLL 内按论文超参配置。


---


### lp_reg — Low-probability Tokens Sustain Exploration in Reinforcement Learning with Verifiable Reward (Lp-Reg)

> **一句话重点 (TL;DR)**：把低概率探索 token 称为"Reasoning Sparks"(如 wait/however/perhaps)，通过构造去噪代理分布 + 前向 KL 的三重门控正则，定向保护这些 spark 不被 GRPO 过度惩罚消除，从而对抗熵崩溃、在长训练区间持续 scaling 不崩;五数学基准均值 60.17%、较此前 +2.66%。

**元信息**：arXiv 2510.03222 (v2, 2025-11-07) ｜ 腾讯 LLM Department(Guanhua Huang、Tingqiang Xu 共一;含清华/北大/港中文实习生，Bo Zhou 通讯) ｜ 2025-10 arXiv，cs.LG ｜ 主题 RLVR 熵崩溃/探索保持(与 OPD 间接相关) ｜ 代码 主仓 https://github.com/CarlanLark/Lp-Reg(占位/README-only)；实现在 dev 仓 https://github.com/CarlanLark/Lp-Reg-dev(已克隆 ~4.6MB，verl 底座，含 recipe/lp_reg 与 recipe/dapo) ｜ 框架 verl，GRPO 基础

#### 1. 相关工作与进展
RLVR 推动 LLM 复杂推理，但训练常在性能平台期崩溃，伴随策略熵快速衰减、探索丧失。已有方法多通过"维持高整体熵"应对：自适应熵正则、高熵变化阻断(Cui et al. 2025)、选择性 token 更新(Wang et al. 2025)。

#### 2. 现有工作存在的问题
依赖"整体熵"是间接且不精确的工具：无差别最大化随机性会放大噪声、加速崩溃(GRPO+Entropy Loss 比 baseline 崩得更快)。真正问题在于有价值的低概率探索 token 被系统性消除——这一根因未被现有熵方法精准命中。

#### 3. Motivation
作者将低概率探索 token 称为 **Reasoning Sparks**(如 "wait"、"however"、"perhaps"，常作逻辑连接/不确定性表达，开启多样推理路径)。预训练模型中这类 token 丰富，但 GRPO 因过度惩罚把它们系统性"扼杀"。目标：保护 reasoning sparks 而不放大无关噪声。

#### 4. 主要灵感 / 核心直觉
关键统计观察：在低概率区间内，有意义的探索 token 的平均概率**一贯高于**无关噪声 token(例：spark "Wait" p=0.03 vs noise "cost" 更低)。这一可分性使得"先用概率阈值滤掉噪声、再保护剩余低概率 token"成为可行——区分 spark 与 noise 而非无差别提熵。

#### 5. 主要解决思路(一段话讲清核心)
构造一个"去噪代理分布 π_proxy"：丢弃概率低于阈值的 token(presumed noise)、把质量重归一化到剩余 token 上，从而放大 reasoning sparks 的相对概率;再在 GRPO 目标上加一个前向 KL 正则项 D_KL(π_proxy‖π_θ)，仅对"低概率 ∩ 非噪声 ∩ 负优势"的 token 触发，定向阻止这些 spark 被消除，又不强制策略完全匹配 proxy。

#### 6. 方法详解(通俗、分步骤)
Low-probability Regularization(Lp-Reg)，集成进 GRPO：
- **代理分布 π_proxy**：(1)过滤噪声——丢弃概率 < 阈值 τ 的 token(τ 可用固定值如 0.02，或 **min-p**：τ=κ·max π，主实验用 min-p、κ=0.02，自适应分布锐度);(2)概率重归一化——把丢弃 token 的质量重分配到剩余 token，得放大 spark 相对概率的"去噪"参考分布。
- **目标函数**：第一项为 GRPO 策略梯度，但**去掉裁剪下界**(避免裁掉低概率探索动作)、加一个大上界 U(数值稳定);第二项为 Lp-Reg 惩罚——仅对**同时满足三条件**的 token 触发：①π_θ 低于批内最低 ρ 分位阈值 δ_B^ρ(低概率)、②在 π_proxy 中概率>0(非噪声)、③优势 A<0(负样本)，施加**前向 KL** D_KL(π_proxy‖π_θ)。前向 KL 在 π_θ→0 而 proxy 非零时给大惩罚，定向防 token 被消除，又不强制完全匹配 proxy。〔实现以 Lp-Reg-dev 为准〕

#### 7. 实验数据集
- RL 训练：Dapo-Math-17K，max 响应长度 8,192，global batch 256。
- 评测(5 个数学基准)：AIME24、AIME25、MATH-500、OlympiadBench、Minerva Math。AIME24/25 采样 16 次(temp 0.6)、其余 greedy。
- 骨干：Qwen3-14B-Base(主)、Qwen2.5-32B-Base。

#### 8. 实验结果与主要发现
- verl 上 GRPO，lr=1e-6 常数无 warmup，group=8，IS 比率上界 U=10。off-policy mini-batch 32(每 rollout 8 次梯度更新)。Lp-Reg：ρ=0.5%(32B)/1%(14B)，β=1.0，min-p κ=0.02。
- 算力：14B 训约 1000 步(8000 GPU·h/32×H20)，32B 约 800 步(16000 GPU·h/64×H20);崩溃(准确率掉>10%)则早停。稳定性测试：Qwen2.5-32B 训 3000 步、81,204 GPU·h。
- 基线：GRPO、GRPO+Entropy Loss、Clip-Higher、80/20 高熵训练、KL-Cov、GSPO。
- 结果：Lp-Reg 在 Qwen3-14B-Base 五基准均值 **60.17%**，较此前方法 **+2.66%**;能在基线崩溃的长训练区间(3000 步)持续 scaling 不崩。

#### 9. 结果如何支撑其主张
"保护 spark 维持探索"由长训练(3000 步)不崩 + 持续 scaling 直接支撑;"整体熵不如定向保护"由 GRPO+Entropy Loss 崩得更快的对照支撑;spark vs noise 可分性由概率统计分析支撑。论证链(观察→机制→对照)较完整。但绝对分数主张较弱——见局限。

#### 10. 逻辑自洽性(中性评估)
自洽：从"整体熵"下沉到"低概率 token 的语义筛选(spark vs noise)"，用前向 KL + 三重门控做定向保护，比无差别熵 bonus 更有针对性，机制叙事与消融一致。前向 KL 的方向选择(proxy‖π_θ)与"防 token 消除"目标匹配，数学上合理。

#### 11. 残留问题 / 局限
- 核心增益 **+2.66%** 属中等增量，且大量算力(8 万 GPU·h)主要用于证明"长训练不崩"而非绝对分数跃升。
- 引入多个阈值超参(τ/κ、ρ、β、U)，调参负担不小且对 14B/32B 取值不同，泛化稳健性需更多骨干验证。
- "spark vs noise"以平均概率统计区分，个例上二者可能重叠，门控误判的代价未量化;仅在数学域验证。
- 主仓为占位、需用 dev 仓复现，是工程可得性上的不便。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 链接：主仓 https://github.com/CarlanLark/Lp-Reg 当前仅含 README(占位，称正整合进最新 veRL);复现代码在开发仓 https://github.com/CarlanLark/Lp-Reg-dev(已 clone ~4.6MB，verl 底座，含 `recipe/lp_reg` 与 `recipe/dapo`)。〔实际实现以 Lp-Reg-dev 为准〕
- 框架：verl(Sheng et al. 2024)，GRPO 基础。
- 可得性：dev 仓代码真实可用、可复现;主仓为占位，需自行切到 dev 仓。


---


### luffy — Learning to Reason under Off-Policy Guidance (LUFFY)

> **一句话重点 (TL;DR)**：把更强策略(DeepSeek-R1)的 off-policy 推理轨迹与 on-policy rollout 混进同一 GRPO group 做组内归一化(Mixed-Policy GRPO)，再用 policy shaping f(x)=x/(x+0.1) 放大低概率关键动作的梯度、抑制熵坍塌;在弱模型/难数据上突破基座上界，六数学基准均值较此前 +6.4、三 OOD +6.2。

**元信息**：arXiv 2504.14945 (v5, 2025-06-22) ｜ 上海 AI 实验室、西湖大学、南京大学、香港中文大学(Yafu Li 等;Project Lead Yafu Li，实习期间完成) ｜ NeurIPS 2025(2025-09-19 接收) ｜ 主题 T3/T4 统一 SFT-RL / off-policy 指导的代表作(相关性 High;SRFT、Prefix-RFT 均建立其上、沿用 mix_src) ｜ 代码 https://github.com/ElliottYan/LUFFY(已克隆 ~36MB) ｜ 框架 veRL，rollout 用 vLLM

#### 1. 相关工作与进展
RLVR(DeepSeek-R1、o1、Kimi-1.5)能让大推理模型涌现多步推理与自我反思("aha moment")。但 RLVR 本质 **on-policy**——只能从模型自身采样中学习，性能受基座自身能力上界约束(与 limit-of-RLVR 观察一致)。纯模仿(SFT/蒸馏)能注入外部知识但属 behavior cloning，泛化差、易过拟合、致熵坍塌。

#### 2. 现有工作存在的问题
- on-policy RL 只能放大已有行为，无法引入真正新的认知能力;弱模型(如 LLaMA3.1-8B)在 RL 下很快进入平台期，因缺乏必要的基础认知行为。
- 纯模仿(SFT)是 behavior cloning，泛化差、易过拟合、熵坍塌。
- 二者之间缺乏在 imitation 与 exploration 间动态平衡的机制。

#### 3. Motivation
引入来自更强策略(如 DeepSeek-R1)的 off-policy 推理轨迹作为"认知脚手架"，让模型在自身 rollout 失败时选择性模仿高质量轨迹、成功时保留自我探索，从而突破基座能力上界。

#### 4. 主要灵感 / 核心直觉
把"外部强 teacher 轨迹"与"自身 rollout"放进同一 group 一起做组内 advantage 归一化——这样当自身全错(advantage 退化)时，恒为正奖励的 teacher 轨迹自然主导学习信号;当自身能解时则保留探索。再观察到 off-policy IS 比率会让"低概率但关键"动作梯度被压制、引发过快收敛/熵坍塌，故用 shaping 函数放大这些动作权重。

#### 5. 主要解决思路(一段话讲清核心)
Mixed-Policy GRPO：在 GRPO 的 advantage 计算中，把 off-policy 教师轨迹(恒为正奖励)与 on-policy rollout 混入同一 group 一起做组内归一化;off-policy 项用重要性采样比、on-policy 项用标准策略梯度;并对 off-policy IS 比率施加 policy shaping f(x)=x/(x+γ)(γ=0.1)放大低概率关键动作梯度、抑制熵坍塌。

#### 6. 方法详解(通俗、分步骤)
- **Mixed-Policy GRPO**：off-policy 教师 prefix 与 on-policy rollout 同 group 组内归一化。off-policy 项 IS 比率，代码 `compute_token_on_off_policy_loss`，off_ratio 仅作用于 prefix_mask;on-policy 项按 advantage 正负标准处理。
- **Policy Shaping via Regularized Importance Sampling**：论文 Eq.6/7 显式给出 **f(x)=x/(x+γ)，γ=0.1**(代码 `off_policy_reshape="p_div_p_0.1"` → `off_ratio/(off_ratio+0.1)`，train.sh 默认即此)，作用于 off-policy IS 比率，放大"低概率但关键"动作的梯度权重(f′(x)∝1/(x+γ)²)。注：论文为计算效率取 **π_old=1**，故 off_ratio=exp(log_prob)(`mix_core_alg.py` L165/186)。〔已核-Eq.6-8 + 代码〕
- **SFT loss 项**：代码 `compute_sft_pure_loss` 实为 **-log_prob 的 masked mean**(对 prefix 的纯 NLL，`mix_core_alg.py` L7-9)，经 `sft_loss_coef` 与 off_pg_loss、可选 KL loss 组合(`mix_actor.py`)。〔已核-代码确为纯 -log_prob〕
- **off-policy IS 默认不裁剪**：train.sh 仅设 `use_off_policy_loss=True` 与 `off_policy_reshape="p_div_p_0.1"`，未设 off_max_clip/off_min_clip(默认 None，与论文"对 off-policy rollout 省略 clip"一致)。〔已核〕

#### 7. 实验数据集
- 训练：OpenR1-Math-220k 子集(约 64k，off-policy 教师轨迹来自 DeepSeek-R1);扩展版用 110k;NuminaMath-CoT。
- 数学评测(6 个竞赛级基准)：AIME 2024/2025、AMC、MATH-500、Minerva、OlympiadBench。
- OOD：ARC-c、GPQA-diamond、MMLU-Pro。
- 基座：Qwen2.5-Math-7B、Qwen2.5-Math-1.5B、Qwen2.5-7B-Instruct、LLaMA3.1-8B。

#### 8. 实验结果与主要发现
- 基于 veRL GRPO 训练器扩展为 mixed-policy：每个 prompt 同时准备教师 off-policy 轨迹(带 prefix_mask)和模型 on-policy rollout，一起算组内 advantage;actor 更新对 on/off token 分别用上述 loss 组合并施加 policy shaping。Qwen2.5-Math-7B-Zero 设置：temperature=1.0、max_response_length=8192、vLLM rollout;rollout batch 128、update batch 64。
- 结果(Qwen2.5-Math-7B，Table 1)：6 数学基准平均 **50.1**，较此前 RLVR(如 Oat-Zero 数学均值 43.7)**+6.4**;3 OOD 基准(ARC-c/GPQA-diamond/MMLU-Pro)平均 **57.8**，较此前最佳 **+6.2**;在 on-policy RL 完全失败的弱模型/难数据场景仍可训练成功。〔已核-Table 1〕

#### 9. 结果如何支撑其主张
"突破基座上界"由弱模型/难数据上 on-policy 失败而 LUFFY 成功直接支撑;"imitation+exploration 平衡优于纯 SFT/纯 RL"由 +6.4 数学/+6.2 OOD 的对照支撑;OOD 增益佐证非过拟合模仿。policy shaping 的抑熵坍塌作用有 Fig.2 与方差分析支撑。证据与主张吻合。

#### 10. 逻辑自洽性(中性评估)
自洽：混入同 group 归一化使 teacher 轨迹在自身失败时主导信号、成功时退居次位，机制与"动态平衡"叙事一致;f(x)=x/(x+0.1) 的 f′∝1/(x+γ)² 确实在 x→0 处放大梯度，与"保护低概率关键动作"目标一致;代码(off_ratio、SFT=-log_prob、不裁剪)与论文公式逐一对应。论文 Theorem 1 给出 importance-weighted 梯度的收敛/方差分析，理论自洽性较强。

#### 11. 残留问题 / 局限
- 依赖高质量外部 teacher(DeepSeek-R1)轨迹，能力上界实质由 teacher 决定——这是"突破基座"的代价，与"RL 自主发现新能力"不同(更接近引导式蒸馏+RL 混合)。
- off-policy 取 π_old=1 是计算简化，可能引入偏差;γ=0.1 等超参的敏感性见 Appendix E.4，跨基座稳健性主要在数学域验证。
- 主要在数学+少数 OOD 验证;teacher 轨迹质量/覆盖对结果的影响未做系统消融。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 链接：https://github.com/ElliottYan/LUFFY(CloneTier=A，已克隆 ~36MB)。核心改动在 `luffy/verl/verl/mix_src`(mix_actor.py / mix_core_alg.py / mix_trainer.py / mix_vllm_rollout.py 等);仓库还内含后续工作 `ExGRPO`(ICLR 2026，用模型自身 off-policy 经验回放，无需外部指导)。模型权重 HF `Elliott/LUFFY-Qwen-Math-7B-Zero` 等。
- 框架：veRL(volcengine/verl)为底座，rollout 用 vLLM;数据/脚本沿用 deepscaler(rllm);其他 off-policy baseline 的 SFT 走 OpenRLHF;评测用 Math-Verify。
- 可得性：完整、真实、可复现，方法实现与论文公式逐项可核。


---


### magistral — Magistral（Mistral 首个推理模型与自建可扩展 RLVR 流水线）

> **一句话重点 (TL;DR)**：Mistral 自底向上、不依赖任何蒸馏轨迹，纯 RL(改造版 GRPO + 四维奖励整形)把 Mistral Medium 3 训成推理模型 Magistral Medium(AIME-24 pass@1 +近 50%)；并给出强制推理语言一致、纯文本 RL 不损甚至提升多模态/指令/函数调用的实证与失败实验。

**元信息**：arXiv 2506.10910 (v1, 2025-06-12) ｜ Mistral AI ｜ 2025-06 技术报告 ｜ 主题 T3/T4(RL 算法+推理) ｜ 代码 仅开源 Magistral Small (24B, Apache 2.0) 权重，无训练代码（weights-only，paper-only）｜ 框架 custom(自研异步在线 RL 系统)

#### 1. 相关工作与进展
提升 LLM 推理是前沿方向；o1 等推理模型靠更长 CoT 提升复杂任务表现。DeepSeek-R1 给出了 RLVR(可验证奖励 RL)大规模造推理模型的关键配方。Magistral 建立在 GRPO(Shao et al. 2024，去 critic、用组内平均奖励作 baseline)之上，并吸收 DAPO/Dr.GRPO 等近期对 GRPO 的改造(Clip-Higher、loss/advantage 归一)。分布式在线 RL 架构借鉴 IMPALA 与近期 trainer-generator-verifier 三角系统(OpenRLHF、verl 等)。

#### 2. 现有工作存在的问题
- 多数推理模型依赖从已有推理模型蒸馏的 RL 轨迹/冷启动；缺乏完全自底向上、仅靠自有模型与基础设施的纯 RL 实践记录。
- 标准 GRPO 在长程推理 RL 中存在 KL 开销(需维护参考模型)、组内长度偏置、熵坍缩、无效(全对/全错)组等问题。
- 数学/代码 RL 常导致响应语言混杂(英/中/俄混用)，体验差。
- "小模型上 RL 能否超过蒸馏 SFT 基线""文本 RL 是否损害多模态/指令/函数调用能力"等问题文献结论不一(DeepSeek-R1 称小模型纯 RL 不及蒸馏)。

#### 3. Motivation
自底向上构建可扩展 RL 栈，探索 LLM 纯 RL(无蒸馏冷启动)的极限；提出简单方法强制模型推理语言与用户一致；验证仅用文本数据 RL 能保持(甚至提升)多模态理解、指令遵循、函数调用；并诚实分享失败实验。两个核心问题：(i) 大 base 上纯 RL 能走多远？(ii) 给定强教师，如何训出最强轻量学生？

#### 4. 主要灵感 / 核心直觉
在线策略本就大幅偏离参考，维护参考模型算 KL 不值 → 直接移除 KL。Clip-Higher 给低概率 token 增长空间以抗熵坍缩、促探索。强制 (problem, thoughts, answer) 三段语言一致即可消 code-switching。系统层面：generator 是在线 RL 独有且最重的负载，异步、不打断地频繁更新 generator(NCCL GPU→GPU 广播，单次更新 <5s)可在效率与 on-policy 性间取平衡(in-flight 序列不刷新 KV-cache，靠 loss 的 off-policy 修正容忍轻微过时)。

#### 5. 主要解决思路(一段话讲清核心)
以 GRPO 为基座做多项稳定性改造(去 KL、按组总长归一 loss、advantage = r−μ 再做 minibatch 归一、Clip-Higher 调高 ε_high=0.26–0.28、过滤零优势组)，配四维奖励整形(格式/正确性/长度惩罚/语言一致)，在自研异步在线 RL 系统(trainer/generator/verifier 三类 worker)上大规模训练；数据经两阶段难度过滤取"goldilocks"难度。Magistral Medium 在 Mistral Medium 3 上纯 RL；Magistral Small 用 Medium 生成的 traces 做 SFT 冷启动后再 RL。

#### 6. 方法详解(通俗、分步骤)
- **算法(改造 GRPO，最终式见论文红字)**：(1) 去 KL 惩罚；(2) loss 归一——组内所有 token/生成累加后除以组内总长 Σ|o_i|，消长度偏置；(3) advantage Â=r−μ，再在 minibatch 内按序列归一 Â^norm=(Â−Â_mean)/Â_std；(4) Clip-Higher——用 clip(·,1−ε_low,1+ε_high)，调高 ε_high 给低概率 token 空间(训练中 0.26–0.28 细调稳组熵；Small RL 用 0.3)；(5) 剔除全对/全错(零优势)组(约束 ∃ r_m≠r_n)。
- **奖励整形(四维)**：格式——须恰含一对 <think></think>，数学答案放 \boxed{}、代码须带语言标注 markdown 块；不满足→reward=0 不评分，满足→0.1 进评分。正确性——数学用多 parser+SymPy 归一对比，正确 +0.9(总 1.0)；代码(C++ 用 C++20、预编译 bits/stdc++.h、10s 编译、随机 20 测试、每测 4s/300MB)全通过 +0.9。长度惩罚——软惩罚(式 1，l_max/l_cache 两阈值，最多 −0.1)。语言一致——译 10% 英文题为法/西/意/德/中/俄，用 fastText 判 (problem,thoughts,answer) 三段(去 LaTeX/代码后)是否同语言，一致 +0.1。System prompt 指定格式与语言("Be as casual and as long as you want"提熵促探索)。
- **基础设施**：三类 worker；异步生成不打断、NCCL 广播权重；batch 按完成数(非 token 数)定义；贪心拼包减 19% padding。
- **数据(§4)**：数学 700k→格式过滤 501k→两阶段难度过滤 38k(先 Mistral Large 2 每题采 16 解去过易/不可解，训一个 24B RL 打分模型，再用它重判全量、并剔除"多数样本一致但与 ground-truth 不符"的疑似错标)；代码多源汇集，执行所有解过滤一致性不足/修正/生成测试，按需复制 Python/C++ 两版，得 35k。
- **训练分阶段**：Medium 纯 RL 多阶段(逐步加难、l_max−l_cache 16k→24k→32k、batch 8k→4k→2k 控 KV-cache)；Small 先 SFT 冷启动(Medium traces + OpenThoughts/OpenR1 提示，含 10% 通用指令保非推理能力，Mistral Small 3 训 4 epoch 按 AIME24 选 ckpt)，再 RL(batch 2048、l_max−l_cache 32k、temp 1.0、ε_high 0.3)。

#### 7. 实验数据集
- **训练**：数学 38k(过滤后)，代码 35k(均可验证)。
- **评测**：数学 AIME'24/'25、MATH-500；代码 LiveCodeBench v5/v6、Aider Polyglot；STEM GPQA；Humanity's Last Exam(文本子集 2500 题)；多语言 AIME'24(法/西/德/意/俄/中)；多模态 MathVista/MMMU/MMMU-Pro；函数调用(内部 bench)、指令遵循(内部 IFEval)。评测 temp=0.7，数学/GPQA top-p=1.0、代码 0.95，max len AIME/LCB 40k 其余 32k。

#### 8. 实验结果与主要发现
- **Magistral Medium 纯 RL**(表 2)：AIME'24 pass@1 26.8→73.6(maj@64 90.0)，AIME'25 21.2→64.9，MATH-500 91.0→94.3，GPQA 59.6→70.8，LiveCodeBench v5 29.1→59.4，v6 30.0→50.3，HLE 4.4→9.0。即 AIME'24 pass@1 提升近 50%。
- **小模型纯 RL 可超蒸馏**(§6.2，与 DeepSeek-R1 结论相反)：Mistral Small 3 纯 RL 在 AIME'24 ≈ 蒸馏版，MATH/GPQA 还更高，仅代码略低；SFT+RL(=Magistral Small) 最佳(表 3：AIME'24 pass@1 SFT 65.4 / RL-only 65.8 / SFT+RL 70.7)。
- **跨域泛化**(§6.1)：math-only RL 也提 LCB(+15.6)，code-only RL 也提 AIME(+17.5)。
- **多模态免费午餐**(§7.2)：纯文本 RL 不损反提多模态推理(MMMU +5%→70%、MMMU-Pro-Std +4.4%、MMMU-Pro-Vision +12%)。
- **其他能力**(§7.3)：函数调用 87.2→87.4、IFEval 86.8→87.4，基本不降甚至小升。
- **多语言**(表 4)：多语言 AIME'24 比英文低 4.3–9.9%(约 1–3 题)，与 base 退化相当。
- **权重轨迹**(§7.1)：RL 在低维空间移动权重，存在明显"length 方向"，raw reward 随输出长度对数 scaling。
- **失败实验**(§7.4)：代码按测试通过率给比例奖励→训练更快但 LCB 终值低 2%、长度增长慢；entropy bonus 不稳(math-only 熵降、math+code 熵爆)，改用调 ε_high 更稳；KL 项主要妨碍训练。
- **§8**：先在 OSS 推理 traces(OpenThoughts+OpenR1，约 1.3M 生成，含 R1 traces)SFT 再 RL，可达与 R1 相当(但 Magistral Medium 未走此路线)。

#### 9. 结果如何支撑其主张
表 2 与 DeepSeek RL-from-scratch 数据对齐比较，直接支撑"自建 RL 栈纯 RL 有效"；表 3/图 5 三范式(SFT/RL-only/SFT+RL)对照支撑"小模型纯 RL 可比蒸馏、且 RL 叠加蒸馏更好"这一与 R1 相反的主张；跨域(表 5)、多模态(图 10)、函数/指令(表 6)分别支撑泛化与"文本 RL 不损其他能力"；多语言(表 4)支撑语言一致奖励有效；PCA(§7.1)与失败实验(§7.4)增强透明度与因果解释(length 是主要增益资源、ε_high 优于 entropy bonus)。

#### 10. 逻辑自洽性(中性评估)
报告型论文，工程细节充分、消融到位、罕见地公开失败实验，自洽性较强。需注意：(1) 多项关键评测(函数调用 internal bench、IFEval internal 版)为内部基准，"分数与公开不可比"，外部不可复现；(2) 与 DeepSeek-R1 的对照是引用其论文数据点而非同条件复跑，base 模型/数据不同，"小模型纯 RL 超蒸馏"的对比并非严格 apples-to-apples；(3) 无训练代码与数据开源，所有结论仅能 paper-only 信任；(4) "AIME'24 +50%"是相对初始 checkpoint 的相对提升表述，绝对值见表 2。

#### 11. 残留问题 / 局限
- **不可复现**：自建 RLVR 流水线完全未开源，仅放 Magistral Small 权重(Apache 2.0)，Medium 不开源；数据(38k 数学/35k 代码)也未公开。
- **内部基准**：函数调用/指令遵循用内部 bench，无法横向对比。
- **多语言推理有代价**：约束推理语言使多语言 AIME 低 4.3–9.9%，作者归因于语言约束。
- **GPQA 在 RL 后回退**(§8 OSS-traces 实验)：72.9%→71.0%，提示 RL 对某些能力可能小幅负迁移。
- **方法依赖经验调参**：ε_high 需在 0.26–0.30 间细调稳熵；entropy bonus 在不同数据集行为相反、不稳；这些"靠手调"的稳定性手段泛化性存疑。
- **范围**：限于可验证(数学数值/代码测试)任务，未及开放式/agentic；length 作为主要增益资源也意味着推理成本随之上升。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- **仅开源权重**：Magistral Small (24B, Apache 2.0) https://huggingface.co/mistralai/Magistral-Small-2506 ；Magistral Medium 不开源。
- **无训练代码仓库**：自建可扩展 RLVR 流水线(改造 GRPO + 四维奖励 + 异步在线 RL 系统：trainer/generator/verifier 三类 worker，NCCL 权重广播)完全未开源。
- **框架 = custom**：不依赖现成 RL 实现或前人蒸馏轨迹，完全基于自有模型与基础设施。**weights-only / paper-only**——本分析基于论文(2506.10910)，无仓库可核码。

〔核实结论〕原分析与论文一致，本轮按论文补全：表 2/表 3 具体数值、§6.1 跨域、§6.2 小模型纯 RL vs 蒸馏、§7 PCA/多模态/函数调用、§7.4 两项失败实验、§8 OSS-traces+RL 达 R1 水平且 GPQA 回退、数据过滤 700k→501k→38k 与代码 35k、多语言退化幅度。无新事实出入(weights-only，无代码可核)。


---


### minimax_m1 — MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention

> **一句话重点 (TL;DR)**：首个开源权重的大规模混合注意力(MoE+lightning attention)推理模型，原生支持 1M 上下文、长生成 FLOPs 仅 DeepSeek-R1 的约 25%；核心算法创新 CISPO——裁剪重要性采样权重而非裁剪 token 更新，保留全部 token(尤其反思类低概率 token)的梯度，对 DAPO 实现 2× 加速。

**元信息**：arXiv 2506.13585 (v1, 2025-06-16) ｜ MiniMax (MiniMax-AI) ｜ 2025-06 技术报告 ｜ 主题 T3/T4(reasoning-RL 算法 CISPO + 长上下文推理后训练)/High ｜ 代码 https://github.com/MiniMax-AI/MiniMax-M1（model-release/权重仓，**CISPO 训练代码不在仓内**，本地未 clone，Tier B paper-only）｜ 框架 自研 RL(借 lightning attention 高效 rollout)，推理支持 vLLM/Transformers

#### 1. 相关工作与进展
大推理模型(o1、DeepSeek-R1)靠大规模 RL 延长推理链取得成功，test-time compute 成为新 scaling 维度。但传统 softmax 注意力二次复杂度限制了推理链持续延长。学界提出稀疏注意力、线性注意力、状态空间模型、线性 RNN 等高效替代，但几乎未在大规模推理模型上充分验证(例外 Hunyuan-T1 用 Mamba，但未开源、细节少)。RL 算法侧 PPO→GRPO(去 critic、组相对 advantage)→DAPO(提高上裁剪界、dynamic sampling、length penalty)是主线。

#### 2. 现有工作存在的问题
- 大规模 RL 训练中 PPO/GRPO/DAPO 的**裁剪丢弃大更新 token 的梯度**：反思类 token(However/Recheck/Wait/Aha，常是推理"分叉点")在 base 模型下概率低，更新时 r_{i,t} 高，首次 on-policy 更新后即被裁掉，无法参与后续 off-policy 梯度；在 16 轮 off-policy 更新/批 的设置下尤甚，而这些 token 对稳熵/可扩展 RL 关键。DAPO 的提高上界做法在此设置下不够有效。
- 混合注意力架构下 RL scaling 出现**训练/推理 kernel 精度不匹配**，导致 reward 无法增长。
- 长生成 RL 中负样本长度增长快于正样本，序列后段累积过大负梯度→pattern collapse(后段退化为乱码)。

#### 3. Motivation
构建并开源一个能高效 scale test-time compute、与 SOTA 推理模型竞争的大推理模型；提出不丢弃 token(即便大更新)同时维持合理熵的 RL 算法(CISPO)，并利用 lightning attention 天然高效 rollout 高效 scale RL；统一连续预训练→SFT cold-start→大规模 RL 的推理后训练管线。

#### 4. 主要灵感 / 核心直觉
PPO 的 IS 权重本是为 off-policy 修正，裁掉 token 更新会连带丢掉该 token 的梯度贡献；改为**只裁 IS 权重、保留 log π 梯度项**即可既稳训练又不丢任何 token(尤其长响应中的低概率反思 token)。训练-推理概率本应相同，偏差源于 LM head 高幅激活 → 提升其精度到 FP32 即可对齐。

#### 5. 主要解决思路(一段话讲清核心)
从带 offline 修正的 REINFORCE 目标出发，CISPO(Clipped IS-weight Policy Optimization)沿用 GRPO 的组相对 advantage 与 token-level loss，但对 IS 权重 r̂_{i,t}=clip(r_{i,t},1−ε^IS_low,1+ε^IS_high) 做裁剪(实验中不设下界、只调 ε^IS_high)，无 KL 项；梯度因权重裁剪略有偏差但保留全部 token 梯度。再给统一公式(引入 token-wise mask M_{i,t}，可表示 PPO 信任域隐式 mask 等不同裁剪策略)。配合连续预训练(7.5T)、SFT cold-start、curriculum RL(先规则可验证、渐混入模型奖励通用任务)，并解决混合架构特有工程问题(FP32 LM head、AdamW 超参、重复检测早停)。

#### 6. 方法详解(通俗、分步骤)
1. **连续预训练**(§2.1)：在 MiniMax-Text-01(456B 总参、单 token 激活 45.9B、32 experts；每 7 个 lightning-attention transnormer 块后跟 1 个 softmax 块)上续训 7.5T tokens(STEM/code/book/reasoning 占 70%，避合成数据、语义去重)；lr 8e-5 训 2.5T 再衰减到 8e-6 over 5T；长上下文分四阶段从 32K 扩到 1M(避免梯度爆炸)。
2. **SFT cold-start**(§2.2)：注入反思式长 CoT，math+code 约 60%。
3. **CISPO**(§3.1，式 4–5)：裁 IS 权重而非 token 更新；用 dynamic sampling + length penalty；无 KL。
4. **混合架构工程**(§3.2)：(a) 训练/推理精度失配 → LM head 提到 FP32，相关性约 0.9→0.987(后续训练稳定在 0.997)；(b) AdamW 用 β1=0.9, β2=0.95, eps=1e-15(梯度幅度跨 1e-18~1e-5、多数 <1e-14，相邻迭代相关弱)；(c) 重复检测早停：连续 3000 token 概率 >0.99 即截断。
5. **数据/奖励**(§4)：规则可验证——数学 pass@10∈(0,0.9) 筛得近 50K(与 SFT 严格不重叠、n-gram+embedding 去污染)、逻辑 SynLogic 合成 41 类约 53K、竞赛编程 30K、SE 基于 SWE-bench 沙箱执行型奖励数千；模型奖励通用任务 25K(有 GT 的用 GenRM 五级打分；无 GT 的用 pairwise −1/0/1 比对参考答案)。
6. **GenRM 长度偏置**(§4.2.2)：GenRM 偏好长输出诱发 reward hacking；离线手段不足 → 在线监控长度偏置、触发即重校准 GenRM，配 reward shaping/value clipping/normalization。
7. **curriculum**(§4.3)：先规则可验证任务、渐混入通用任务，防灾难性遗忘。
8. **延长思考**(§5)：40K→48K→56K→64K→72K→80K 分阶段扩窗(看 perplexity 收敛与 99 分位长度判定);后期 pattern collapse 根因是负样本更快触顶累积大负梯度，三招解决——重复检测早停、sample-level + token-level 归一缓解正负失衡、降梯度裁剪阈值与 ε^IS_high。
9. **资源**：完整 RL run 512×H800、约 3 周、约 $0.53M。

#### 7. 实验数据集
- **训练**：连续预训练 7.5T；RL 数据近 50K 数学 + 53K 逻辑 + 30K 竞赛编程 + 数千 SE + 25K 通用(curriculum 组织)。
- **评测**(temp 1.0, top-p 0.95)：数学 MATH-500/AIME 2024/2025(AIME 采 32 取均)；编程 LiveCodeBench/FullStackBench(16 采均)；推理知识 GPQA-Diamond(32 采)/MMLU-Pro/HLE(无工具)；SWE-bench Verified；agentic TAU-Bench；长上下文 MRCR 等。

#### 8. 实验结果与主要发现
- **CISPO 受控对比**(图 2，Qwen2.5-32B-base zero-RL，AIME 2024)：同步数显著超 DAPO/GRPO，用 50% 步数即达 DAPO 水平(对 DAPO 2× 加速)。
- **核心基准**(表 2，M1-80k)：AIME 2024 86.0、AIME 2025 76.9、MATH-500 96.8、LiveCodeBench 65.0、FullStackBench 68.3、GPQA-Diamond 70.0。
- **定位**：整体超原 DeepSeek-R1 与 Qwen3-235B；对最新 DeepSeek-R1-0528，数学/编程竞赛落后，但工具使用/长上下文相当或更优；TAU-Bench 超 Gemini 2.5 Pro，长上下文超 o3 与 Claude 4 Opus。
- **test-time scaling**：M1-80k > M1-40k(数学/代码)，验证延长思考收益。
- **效率**：100K 生成长度 FLOPs 仅 DeepSeek-R1 约 25%；原生 1M 上下文(R1 的 8 倍)。
- **工程发现**：FP32 LM head 使训练-推理概率相关性 0.9→0.987(关键，否则 reward 不增长)。

#### 9. 结果如何支撑其主张
图 2 的 zero-RL 受控对比(同 backbone/数据/步数)直接支撑"CISPO 比 DAPO/GRPO 更高效"这一算法主张(2× 加速、用全 token 梯度);表 2 横向对标多个开/闭权模型支撑"开源权重 SOTA 推理"主张,且在 agentic/长上下文上的相对优势与"lightning attention 高效长生成"的架构卖点一致;图 3 训练-推理概率相关性前后对比支撑 FP32 修复的因果性;M1-80k vs 40k 支撑 test-time scaling。

#### 10. 逻辑自洽性(中性评估)
算法动机(token clipping 丢反思 token)→CISPO(裁 IS 权重)→统一 mask 公式→受控验证,逻辑链清晰;工程问题(精度失配/优化器/重复/pattern collapse)均给出根因+解法,自洽性较强。需注意:(1) CISPO 的核心比较(图 2)在 Qwen2.5-32B 而非 M1 本体上做,M1 全量训练并无 CISPO-vs-DAPO 的同条件消融,"2× 加速"结论的外推到 456B 混合架构属间接证据;(2) GenRM/通用任务奖励高度依赖内部 GenRM 与人标基准,长度偏置靠"在线监控+重校准"这类经验闭环处理,难以复现/量化;(3) 多个核心创新(CISPO 代码、GenRM、沙箱、数据)均未开源,paper-only 信任;(4) 部分基准(HLE 等)带 ∗ 标注(自测/口径差异),横向比较需谨慎。

#### 11. 残留问题 / 局限
- **核心算法不可复现**:CISPO 训练实现、GenRM、SE 沙箱、RL 数据均未开源;公开仓为权重/推理仓。
- **CISPO 的 M1 本体证据间接**:主算法对比在 32B dense 上,缺 M1(456B 混合)上的同条件消融。
- **数学/编程竞赛落后最新 R1**:相对 DeepSeek-R1-0528 在 AIME/LiveCodeBench 上落后,优势集中在工具/长上下文。
- **CISPO 梯度有偏**:权重裁剪引入偏差(作者承认),长期/不同设置下的影响未充分刻画。
- **混合架构脆弱性**:精度失配、优化器超参、长生成 pattern collapse 等问题需大量针对性补丁,泛化到其他架构/规模未知;合成推理数据会破坏长上下文 RL 稳定性(被下采样)。
- **GenRM 长度偏置/reward hacking** 是反复出现的隐患,靠在线监控缓解而非根治。
- **成本高**:512×H800×3 周(~$0.53M),门槛极高。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库 https://github.com/MiniMax-AI/MiniMax-M1 ;权重在 HuggingFace。支持 vLLM / Transformers 推理,有部署指南;另有商用 API(minimax.io)。
- **该仓为模型发布/权重仓;CISPO 训练代码、GenRM、SE 沙箱、RL 数据均不在仓内**。本地未 clone(CloneTier=B,仅记录)。
- **框架 = 自研 RL**(借 lightning attention 高效 rollout);注:论文提到 VeRL 默认 AdamW 配置会导致不收敛(故改 β2=0.95, eps=1e-15),说明其设置与 verl 相关但训练实现未公开。**model-release / paper-only**——CISPO 不在 repo,本分析基于论文(2506.13585)。

〔核实结论〕原分析与论文一致(CISPO 裁 IS 权重、无 KL、统一 mask 公式、FP32 LM head 0.9→0.987、数据规模近 50K/53K/30K/25K、512×H800×3 周×$0.53M、2× DAPO 加速 均核对无误);本轮按论文补全:表 2 核心基准数值、§4.2.2 GenRM 长度偏置在线监控、§5 长思考分阶段扩窗与 pattern collapse 三招、AdamW/重复检测早停细节、与 R1-0528 的相对定位。无新事实出入(CISPO 不在 repo,无代码可核)。


---


### negative_reinforce — The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning

> **一句话重点 (TL;DR)**：把 RLVR 的二元学习信号拆成"奖励正确(PSR)/惩罚错误(NSR)"两条独立范式，发现仅惩罚错误的 NSR 在整个 Pass@k 谱(k 到 256)上一致超过 base、常追平甚至超过 PPO/GRPO；据此提出把正奖励下调权重 λ=0.1 的 W-REINFORCE。结论高度依赖强先验 backbone(Qwen)。

**元信息**：arXiv 2506.01347 (v2, 2025-10-25) ｜ University of Virginia / Princeton Language and Intelligence (PLI)；Xinyu Zhu、Mengzhou Xia、Zhepei Wei、Wei-Lin Chen、Danqi Chen、Yu Meng ｜ NeurIPS 2025 ｜ 主题 T3(RLVR 机理+算法变体)/相关性 High ｜ 代码 https://github.com/TianHongZXY/RLVR-Decomposed（本地已 clone，~15MB，Tier A）｜ 框架 veRL(vendored)

#### 1. 相关工作与进展
RLVR(可验证奖励强化学习)已成为提升 LLM 推理的关键技术：用确定性验证函数给二元奖励(+1/−1)，既缓解 reward hacking 又免去人工标注与奖励模型；DeepSeek-R1、Kimi K1.5 证明其可诱发长 CoT 与自反思等涌现推理行为。inference-time scaling 方向另起一支：通过多候选采样或更长推理 trace 提升命中率，并用 Pass@k 而非 Pass@1/贪心衡量模型能力边界。近期工作(如 Yue et al. [72])质疑"RL 训练模型是否真比 base 更强"，发现 RLVR 主要把分布推向高奖励响应而非引入新能力，在大 k 处反不如 base。

#### 2. 现有工作存在的问题
RLVR 同时用正确与错误样本经 policy gradient 更新，但其精确机理(尤其如何分别利用正确/错误样本)被低估、研究不足；多数工作只看 Pass@1 或贪心解码，忽视模型行为(inference-scaling/Pass@k 多样性)层面的变化。正负信号在标准 RLVR 中纠缠在一起，难以解释模型到底从成功还是失败中学到什么。

#### 3. Motivation
把 RLVR 学习信号拆成两条独立、可隔离分析的范式——只奖励正确样本(PSR)与只惩罚错误样本(NSR)，回答"PSR/NSR 各自如何塑造模型行为与泛化"，并据此设计更好平衡精度与多样性的目标。

#### 4. 主要灵感 / 核心直觉
RLVR 的二元奖励使奖励符号天然绑定到序列正确性(同一序列所有 token 同奖励，batch 均值始终落在 [−1,1]，归一化后保号)，因此可干净地按符号拆成 PSR/NSR(§C 论证这点是 RLVR 区别于带奖励模型 RL 的关键)。直觉：惩罚错误时按"其他 token 当前概率"成比例地把概率质量重分配回去，等于按模型先验做软重排，既纠错又保留探索性。

#### 5. 主要解决思路(一段话讲清核心)
将 RLVR 目标 L = L_PSR + L_NSR 形式化分解(式 2–4)：PSR 像 SFT，提升正确响应似然；NSR 像 likelihood minimization，压低错误响应概率。分别独立训练后用全 Pass@k 谱评测，发现 NSR 单独训练异常有效；再用 token 级梯度分析解释机理；最后提出 W-REINFORCE——在 REINFORCE 目标上把正奖励贡献按 λ 缩小(λ=1 即 REINFORCE，推荐 λ=0.1)，在 PSR 的高 Pass@1 与 NSR 的高多样性之间取得平衡。

#### 6. 方法详解(通俗、分步骤)
- **分解**：L_PSR 只在 r=+1 样本上更新(增大正确似然)，L_NSR 只在 r=−1 样本上更新(减小错误似然)；二者均 on-policy(响应采自当前模型)。
- **梯度分析(式 7/8)**：对 token logit 求导。PSR 抬高被采样(正确)token logit、压低其余 → 持续 sharpening，熵下降、过拟合。NSR 压低被采样(错误)token、按其余 token 当前概率 π_v 成比例抬高它们的 logit；且被采样 token 的负梯度被 (1−π_yt) 缩放 → 对高置信 token 更新很小，从而(1)保护高置信先验、(2)按先验做概率重分配促探索、(3)一旦不再犯错即自动停止更新(隐式正则)。
- **与熵正则/unlikelihood 对比(§B)**：熵正则会无差别压高概率 token、抬低概率 token，可能违背先验；unlikelihood 用 −log(1−π) 惩罚，缺少 (1−π_v) 阻尼会侵蚀先验；NSR 因阻尼项更温和。
- **W-REINFORCE(式 9)**：L = λ·L_PSR + L_NSR，λ=0.1。

#### 7. 实验数据集
- **训练**：MATH(7,500 题)。
- **评测**：MATH、AIME 2025、AMC23 的测试集，报告完整 Pass@k 谱(用 [5] 的无偏估计量)。Qwen2.5-Math-7B/Llama 采 256 样本(temp 0.6, top-p 0.95)，Qwen3-4B 采 64 样本(temp 0.7, top-p 0.8, top-k 20)。
- **模型**：Qwen2.5-Math-7B、Qwen3-4B(非思考模式训练/推理)、Llama-3.1-8B-Instruct。

#### 8. 实验结果与主要发现
- **NSR 单独训练出人意料地有效**：全 Pass@k 谱一致优于 base；在 Qwen2.5-Math-7B 上 k=256 处超过 PPO/GRPO/PSR。表 1(MATH)：base Pass@1=63.2，NSR=75.7，PPO=76.6，GRPO=76.3；但 k=256 处 NSR=96.9(=base)、PPO=96.3、GRPO=95.5。AIME2025 k=256：NSR=53.3 vs PPO 43.3/GRPO 50.0；W-REINFORCE=56.7(最佳)。
- **PSR 提精度损多样性**：Pass@1 上升快但 k>8 后跌破 base，熵急剧下降、过拟合。
- **Qwen3-4B 非思考模式**：PSR 无法激活潜在(思考模式)能力甚至损 MATH/AMC23；NSR/GRPO 能逼近思考模式(NSR Pass@1=94.0/Pass@64=98.0 ≈ 思考模式 94.5/97.8)。
- **Llama 上 RL 普遍损 inference-scaling**：所有方法 Pass@256 都跌破 base，NSR 损失最小 → backbone 先验强弱决定 RL 能否获益。
- **训练动态(图 5)**：NSR 全程维持接近 base 的高熵；PSR 熵骤降；PPO/GRPO 居中。
- **W-REINFORCE** 在多数 k 上稳超 PPO/GRPO；vanilla REINFORCE 反而欠佳。λ 消融(§E)：λ≤0.2 稳定，λ=1 时 Pass@256 大跌。

#### 9. 结果如何支撑其主张
全 Pass@k 谱 + 训练熵/正确样本比/全解比等多维动态，直接支撑"NSR 保多样性、PSR 损多样性"的主张；token 级梯度推导(式 7/8 含完整 §A 推导)给出机理解释并能外推到 PPO/GRPO(§4.3 论证 clip 只限幅不改方向、KL 系数通常极小或移除、GRPO advantage 只是保号重标定)，逻辑链较完整。W-REINFORCE 在三基准多 k 上的稳定增益支撑"简单下调正奖励即可平衡"。

#### 10. 逻辑自洽性(中性评估)
内部自洽性强：分解—独立实验—梯度推导—外推—简单变体，环环相扣。需注意：(1) §4.3 把分析外推到 PPO/GRPO 属"定性不变"论证，未对 PPO critic 的细粒度 credit assignment 做严格分析(论文自己也观察到 PPO 后期熵回弹这一 GRPO 没有的现象)；(2) NSR/PSR 因只用半数样本，每 batch 有效样本少于 PPO/GRPO，比较并非等样本量(论文如实指出)；(3) 强先验依赖是反复出现的前提(Qwen 有效、Llama 普遍退化)，把结论限定在"模型先验强"时。

#### 11. 残留问题 / 局限
- **NSR 长训不稳**(§F)：上百步后性能明显下滑，提示其隐式护先验机制不足以长期稳定；W-REINFORCE 无此问题，但作者也承认这与 GRPO 等的长训崩溃同类，可能需引入一定 PSR。
- **仅限稀疏二元奖励**：未验证 dense/连续/过程奖励或主观任务下 PSR/NSR/W-REINFORCE 的表现。
- **backbone 依赖**：Llama 系列上 RL 普遍损 inference-scaling，方法增益主要在强先验模型上成立，泛化性受限。
- **数据/任务窄**：训练仅 MATH 7.5K，评测均为数学竞赛类，未及代码/agentic 等。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库 https://github.com/TianHongZXY/RLVR-Decomposed（本地已 clone，~15MB，Tier A）；模型集合 HuggingFace `TianHongZXY/rlvr-decomposed`。
- **框架 = veRL**：仓库内 vendoring `verl/`(advantage/clip 逻辑在 `verl/trainer/ppo/core_algos.py`、`ray_trainer.py`)；推荐用 verl 官方 docker，Qwen3 需 vllm 0.8.5 + transformers 4.52.2。
- 训练入口 `run_qwen2.5-math-7b_psr_nsr.sh` / `run_qwen3-4b_psr_nsr.sh`(脚本内指定 advantage 为 PSR/NSR/W-REINFORCE，W-REINFORCE 设 `positive_advantage_weight`=λ=0.1)；另有 `_ppo.sh`/`_grpo.sh`。评测 `eval.sh` + `calculate_metrics.py`(算 Pass@k)、`grader.py`。
- 超参(§D.1)：prompt batch=1024，每 prompt 8 rollout，temp 1.0，mini-batch=256，lr=1e-6，clip ε=0.2，PPO/GRPO KL 系数 1e-3、PSR/NSR 不用 KL；熵 bonus 1e-4；8×H200 单节点。max ctx：Qwen2.5-Math-7B/Llama 4096，Qwen3-4B 32768。注：PSR/NSR 因不用 KL，advantage 即原始奖励，故禁用 veRL 的 advantage 归一化(否则归一后为 0、失去信号)。

〔核实结论〕原分析与论文/仓库一致，无新出入；本轮补全限制(§F 长训不稳、仅稀疏二元奖励)、Pass@k 具体数值、§4.3 外推论证与等样本量/backbone 依赖等中性评注。


---


### nft — NFT: Bridging Supervised Learning and Reinforcement Learning in Math Reasoning

> **一句话重点 (TL;DR)**：通过 Bayes 规则把生成策略拆为正/负策略，并用同一目标网络隐式参数化负策略，NFT 让纯监督学习也能从负样本中"自我反思"；理论上证明 on-policy 时 NFT 与 GRPO 梯度完全等价，实证上 7B 略超 DAPO、32B 与 DAPO 基本持平。

**元信息**：arXiv 2505.18116 (v3, 2026-03-01) ｜ Tsinghua / NVIDIA / UIUC / Stanford（Huayu Chen, Kaiwen Zheng, Qinsheng Zhang, Ganqu Cui, Lifan Yuan, Yin Cui, Haotian Ye, Tsung-Yi Lin, Ming-Yu Liu, Jun Zhu*, Haoxiang Wang）｜ ICLR 2026 会议论文 ｜ 主题 T?/中等相关（RLVR 数学推理 + 负样本利用，非蒸馏）｜ 代码 https://github.com/NVlabs/NFT（已开源，含 7B/32B 训练与评测脚本，权重 HF nvidia/NFT-7B、NFT-32B）｜ 框架 VeRL（fork 自官方 DAPO 环境）+ FSDP + Ray

#### 1. 相关工作与进展
RLVR（PPO、GRPO、DAPO 等）以 ground-truth verifier 的二元信号驱动 LLM 数学推理自我改进，相比依赖奖励模型模拟人类反馈的传统 RLHF 更可靠、相比 DPO 等偏好学习更省内存（无需成对偏好数据）。监督学习一侧，RFT（拒绝采样微调，Dong et al. 2023）已证明在正样本上微调有效，但被普遍认为只能"记忆正样本"。本文方法上与 DPO 的"用策略网络隐式参数化奖励模型"以及视觉生成中"用生成网络隐式定义条件/残差模型"在思想上同源——都强调通过隐式定义的模型做直接优化。

#### 2. 现有工作存在的问题
- RFT 类 SL 方法完全丢弃负样本，被普遍视为其落后于 RL 的根因；
- "自我反思是 RL 专属、SL 天生无法从错误中学习"这一论断缺乏理论澄清；
- SL 与 RL（尤其 GRPO）之间究竟是何种关系并不清楚。

#### 3. Motivation
质疑"verification-driven 自我改进是 RL 专属"的论断：能否在纯 SL 范式内同样实现从负样本中改进？若能，则 SL 与 RL 的差距主要源于负样本利用能力，而非 RL 本身的优越性。

#### 4. 主要灵感 / 核心直觉
由 Bayes 规则，生成策略 π 可分解为正策略 π+ 与负策略 π−，并满足耦合关系 rq·π+ + (1−rq)·π− = π_old（rq 为该 prompt 的正确率）。因此只要 π 与 rq 已知，从负样本学 π− 等价于在塑造目标正策略 π+——负样本里同样蕴含可监督的信息。

#### 5. 主要解决思路(一段话讲清核心)
把负策略**隐式重参数化**为目标正策略：π−_θ = (π_old − rq·π+_θ)/(1−rq)。于是在负样本上做最大似然训练就直接优化了 π+_θ（Theorem 3.1：理想容量下最优解 π+_θ* = π+）。结合正样本的常规 MLE，得到统一的 token 级损失（Eq.9/10）：正样本走似然比对数，负样本走隐式负似然比，并用 straight-through max 算子裁剪保证对数参数为正、梯度可回传。全程只维护单一模型，内存开销极小。

#### 6. 方法详解(通俗、分步骤)
在线迭代（Algorithm 1）：
1. **数据收集**：当前 LLM π 对每个 prompt q 采 K 个答案，verifier 判二元正误 r₁:K，估计正确率 r̂q=mean{r₁:K} 并记录 token 级 πold 似然。
2. **prompt 过滤**：只保留 0<rq<1 的 prompt（全对/全错无梯度信息）。
3. **构造似然比**：正样本 Rt_θ=π+_θ/πold；负样本用隐式负似然比 Rt_θ=(1−r̂q·Rt_θ)/(1−r̂q)，再经 straight-through max 裁剪下界 ϵ。
4. **最大似然更新**：θ ← θ + λ∇Σt log Rt_θ；prompt 加权 ω(q) 侧重难题。
5. π ← π+_θ，进入下一轮。

理论分析（Sec.4）：(a) 仅二元奖励下 GRPO 的损失梯度可改写；(b) GRPO 的 group normalization（advantage 标准化）已隐含在 NFT 损失中；(c) Theorem——令 ϵ≤1，on-policy 时 ∇L_NFT = ∇L_GRPO 完全等价；二者唯一差异在 off-policy 的梯度裁剪策略（GRPO 硬置零，NFT 软衰减）。

#### 7. 实验数据集
- 训练：DAPO-Math-17k（仅含整数答案的数学题）。
- 模型：Qwen2.5-Math-7B、Qwen2.5-32B（base）。
- 评测：6 个数学基准——AIME24、AIME25、AMC23（报 avg@32）、MATH500、OlympiadBench、Minerva（报 avg@1），取平均。
- 训练规模：约 5,000 梯度步、batch 512、生成温度 1.0。

#### 8. 实验结果与主要发现
Table 1（均值）：
- **7B**：base 31.6 → NFT 51.7，超过 GRPO(49.5)、Dr.GRPO(49.8)、DAPO(51.2)，远超 RFT(48.3)、DPO(48.9)。
- **32B**：base 29.6 → NFT 59.2，与 DAPO(59.9)基本持平（略低 0.7），远超 RFT(52.8)。
主要发现：(1) 纯 SL（NFT）无需外部教师即可显著提升数学推理，匹配甚至超过 SOTA RL；(2) RFT 与 RL 在线训练的差距主要源于 SL 过去无法利用负样本，而非 RL 固有优越——NFT 通过负样本利用大幅弥合该差距；(3) on-policy 下 NFT 与 GRPO 实证一致，印证理论等价。

#### 9. 结果如何支撑其主张
所有算法用相同训练数据、基础设施与通用超参对比，使"SL 能否匹敌 RL"的结论归因清晰。7B 上 NFT 超 DAPO、32B 持平，支撑"SL 可达 RL 水平"；NFT 显著超 RFT 支撑"负样本利用是关键差距来源"；理论等价 + on-policy 实证一致互相印证，构成自洽的"SL-RL 桥接"叙事。

#### 10. 逻辑自洽性(中性评估)
理论链条（Bayes 分解 → 隐式重参数化 → Theorem 3.1 → on-policy 等价）清晰且有解释力，是本文最强部分。实验受控对比严谨。需注意：所谓"等价"严格成立仅限 on-policy；off-policy 行为仍依赖经验性的软衰减裁剪，缺乏对其最优性的理论保证。"自我反思"一词用得偏宣传——本质是"在负样本上做带符号的似然更新"，与人们直觉的"反思"语义不完全对应。

#### 11. 残留问题 / 局限
- 实证增益有限：32B 仅与 DAPO 持平（甚至略低 0.7 点），核心价值是"统一视角"而非性能突破。
- 隐式负似然比需裁剪下界 ϵ 保数值稳定，off-policy 收敛性无理论保证。
- 仅在数学推理 + 整数答案可验证场景验证，对开放式/非二元奖励任务是否成立未知。
- 等价性证明依赖"无限数据与模型容量"理想假设。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 代码：https://github.com/NVlabs/NFT （完整开源）。核心文件 `main_nft.py`、`ray_nft_trainer.py`、`experience_maker.py`、`verifier.py`；脚本 `train_7B.sh`/`train_32B.sh`、`eval_local_7B.sh`/`eval_local_32B.sh`；数据下载 `download_data.sh`。
- 框架：VeRL（volcengine/verl，fork 自官方 DAPO 环境，固定 commit），FSDP + Ray 分布式；NFT 继承 DAPO 绝大多数超参与设计。
- 关键超参（README/脚本）：`neg_weight`（1.0=NFT，0.0=RFT，-1.0=DAPO；32B 建议试 0.5）、`normalize`（1=Dr.GRPO 对齐、2=标准 GRPO 对齐）、`ratio_type`（token/sequence）、`clamp_negative`/`clamp_positive`（隐式负/正似然比的 straight-through 裁剪下界，默认 1.0/0.0）。
- 权重：HF nvidia/NFT-7B、nvidia/NFT-32B；验证集 HF ChenDRAG/VeRL_math_validation。7B 用 4×8 H100，32B 用 16×8 H100。


---


### on_policy_sft — On-Policy Supervised Fine-Tuning for Efficient Reasoning

> **一句话重点 (TL;DR)**：把"高效推理"的带长度惩罚 RL 目标做原理性化简——去掉 KL 正则、去掉组内归一化、用最简的截断式长度惩罚——可证明 GRPO 目标退化为对"按正确性与简洁性过滤的自生成数据"做交叉熵 SFT；该极简 on-policy SFT 在五个数学基准上定义了 accuracy–efficiency Pareto 前沿。

**元信息**：arXiv 2602.13407 (v1, 标注 2026-02-13；脚注 Preprint Feb 17, 2026) ｜ 东方理工(EIT, 宁波) + 香港理工 / Paris Dauphine-PSL / 上海交大 / 腾讯混元AI Lab / LMU 慕尼黑（Anhao Zhao, Ziyang Chen, Junlong Tong, Yingqi Fan, Fanghua Ye, Shuhao Li, Yunpu Ma, Wenjie Li, Xiaoyu Shen*）｜ 预印本 2026-02 ｜ 主题 T?/相关（统一 SFT-RL 视角下的高效推理，含 on-policy 自蒸馏味道）｜ 代码 https://github.com/EIT-NLP/On-Policy-SFT（已开源，`opsft/` 子目录）｜ 框架 veRL + FSDP + vLLM

#### 1. 相关工作与进展
LRM 多用 RL（GRPO 系）训练并产生很长 CoT，带来显著推理开销。为缓解，"高效推理"方向涌现大量工作，在奖励中叠加长度惩罚/简洁性奖励（ThinkPrune、O1-Pruner、L1、LASER 等），且 RLen 与权重 γ(o|q) 的设计日趋复杂（按正确性/难度条件激活、用组内 mean/median/max 长度统计）。这些方法可统一写为 REff = RAcc + γ(o|q)·RLen。

#### 2. 现有工作存在的问题
- RL-based 高效推理虽大幅缩短 CoT，但常伴随随任务复杂度变化的精度下降，得到次优的精度–效率权衡；
- 叠加多奖励（正确性 + 简洁性）联合优化使训练不稳定；
- 复杂的奖励塑形被不加甄别地沿用，与高效推理问题的内在结构未必对齐。

#### 3. Motivation
重新审视：把 GRPO 等复杂目标直接套用于高效推理是否合理？作者通过原理分析指出该范式存在两处根本性错配（KL 冗余、组内归一化失配），并据此推导出能否用更简单、有原理依据的方案达到同样甚至更好的 Pareto 前沿。

#### 4. 主要灵感 / 核心直觉
高效推理有两条与 RLHF 不同的关键性质：正确性与长度都**直接可验证**（无需学习奖励模型），且本质是**多奖励**问题。基于此：(1) KL 正则在 RLHF 中用于防止 reward over-optimization，但此处奖励可靠、无分布漂移之忧，故冗余；(2) 组内奖励归一化在多奖励下会放大无信息样本的梯度、并混淆不同奖励组成（举例：奖励向量 (0,1)/(0,0) 与 (1,1)/(0,0) 归一化后得到相同 advantage），引入歧义。去掉这两项 + 采用最简截断奖励（超长响应给零奖励），策略梯度目标即退化为 reward-free 的最大似然 SFT。

#### 5. 主要解决思路(一段话讲清核心)
在带长度相关奖励 γ(o|q) 的 RL 目标下，移除 KL 正则、移除组内归一化、把长度惩罚简化为"截断"（超过固定长度的响应奖励为零），原始策略梯度目标就**等价于在自生成数据上做监督微调**，而该数据天然按"正确性 + 简洁性"过滤（正确且未被截断者得正奖励）。由此得极简 recipe——on-policy SFT：用当前策略自采样、过滤、对保留轨迹做交叉熵，无需 PPO/GRPO 机制与奖励塑形。

#### 6. 方法详解(通俗、分步骤)
1. **Rollout**：用当前策略对每个 prompt 采 N=32 条回答（温度 1.0）。
2. **评分与过滤**：reward_fn(verifier) 判正误；截断式奖励使超长响应得零奖励。代码 `_data_filter` 按 uid 分组，**丢弃错误回答、保留全部正确回答（score>0）**；query 全错则整条丢弃；保留数对 dp_size 取整。注意"简洁性过滤"在 as-shipped 仓库中**并非由 `_data_filter` 显式按长度筛选**（length 字段被提取但代码注释明确"not used"），而是通过截断奖励——超 `max_response_length` 的响应被截断/得零奖励从而落入"错误"被丢弃——间接实现〔原 〔待核〕 已解决〕。
3. **训练**：把 verl PPO actor 的 PG 损失替换为交叉熵——`cross_entropy_loss = agg_loss(loss_mat=-log_prob, loss_mask=response_mask, loss_agg_mode="token-mean")`，即对 response token 的 NLL，无 advantage、无 clip、`use_kl_loss=False`。
4. **同分布训练–评测**：训练与评测用相同 prompt 模板，避免分布漂移混淆增益归因。
5. 实践指南：rollout 温度、每输入 rollout 数、length bias correction、最大输出长度等，附机制解释以稳定优化。

#### 7. 实验数据集
- 训练：DeepScaleR(DSR) 为主；仓库另含 OpenThoughts3-1.2M(math-only)、GSM8K 训练集。
- Backbone：DeepSeek-R1-Distill-Qwen-1.5B 与 -7B（脚本默认 1.5B，LR=1e-6，max_gen=3500，batch=32，rollout_n=32，total_epochs=2）。
- 评测：五个数学基准 GSM8K、MATH-500、AMC23、AIME24、AIME25，报 Acc、Pass@N、平均 token 数(Tok)、压缩率(CR) 与综合效率分 Eff。
- 对比 10 个 baseline，跨 training-free / SFT-based / RL-based。

#### 8. 实验结果与主要发现
- **1.5B**：整体 Acc 59.9%、Pass@N 73.6%，略超原模型(59.0%/73.5%)；平均生成长度从 10,178 → 2,186 token（约 80% 缩短）；Eff=2.74%，超最强 RL baseline(2.55%)。
- **7B**：Eff=2.97%（最高），长度较原模型缩短约 70%。
- 在不同生成长度预算下持续位于 accuracy–efficiency Pareto 前沿，优于 ThinkPrune、O1-Pruner、L1、LASER 等。
- 训练侧：每步显存与墙钟约降 50%，收敛较 RL 加速约 70%；长度控制更稳（多次生成的长度方差更低）。
- 机制分析：增益主要来自**使用 on-policy 数据**这一点本身（而非奖励设计）。

#### 9. 结果如何支撑其主张
"GRPO 退化为过滤式 SFT"的理论推导 + 实证 Pareto 前沿构成主张闭环：既然去掉 KL/归一化后目标等价于过滤式 SFT，那么直接做该 SFT 应不劣于复杂 RL——实验在五基准上证实其定义 Pareto 前沿且训练更省。同分布训练–评测设计排除了 prompt 不匹配的混淆。消融"on-policy 数据是增益主因"进一步把功劳归于 on-policy 性而非奖励塑形。

#### 10. 逻辑自洽性(中性评估)
"两处错配"的论证清晰且有举例支撑，化简逻辑成立。但需注意：所谓"等价于 SFT"依赖一系列简化前提（去 KL、去归一化、截断奖励），并非对任意长度奖励普适——它本质是"在某一特定的简化奖励族下成立"。"conciseness 过滤"在论文叙述与仓库实现间存在表述落差：仓库靠截断奖励间接实现，而非显式长度筛选，读者易误以为有专门的长度过滤器。Eff 等综合指标的定义对结论敏感，跨方法可比性需谨慎。

#### 11. 残留问题 / 局限
- 仅在数学推理、可验证正确性场景验证；对开放式/不可验证任务不适用。
- "截断奖励"的最大长度是关键超参，过紧会牺牲难题精度——论文也承认精度随任务复杂度有边际下降风险。
- 等价性结论绑定特定简化奖励族，非对一般高效推理目标的普适证明。
- 仓库 as-shipped 与论文描述的简洁性过滤实现方式不一致（已核实为靠截断奖励间接实现）。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 代码：https://github.com/EIT-NLP/On-Policy-SFT （`opsft/` 子目录，已开源）。
- 框架：veRL（基于 verl 改造），FSDP + vLLM rollout。recipe 位于 `opsft/recipe/On_Policy_SFT/`：`dp_actor.py`（PG 损失换成交叉熵）、`on_policy_sft_trainer.py`（on-policy 训练流程 + `_data_filter` 数据过滤）、`main_on_policy_sft.py`、`fsdp_workers.py`、`config/on_policy_sft_trainer.yaml`。
- 入口：`bash examples/On_Policy_SFT.sh`。关键超参（脚本确认）：LR=1e-6、MAX_GEN_LENGTH=3500、ROLLOUT_N=32、temperature=1、train_batch_size=32、use_kl_loss=False、loss_agg_mode=token-mean、total_epochs=2、2×GPU。
- 数据：仓库内置 DeepScaleR/GSM8K/OpenThoughts3 train.parquet 与八个评测 benchmark parquet。


---


### one_shot_rlvr — Reinforcement Learning for Reasoning in Large Language Models with One Training Example

> **一句话重点 (TL;DR)**：仅用一条（甚至一条以内）训练样本做 RLVR(GRPO/PPO)，就能把 Qwen2.5-Math-1.5B 在 MATH500 从 36.0% 拉到 73.6%、六基准均值从 17.6% 升到 35.7%，基本匹配含该样本的 1.2k 子集；揭示 post-saturation generalization、跨类泛化、自反思增多等现象，支持"RLVR 主要是激发而非注入推理能力"。

**元信息**：arXiv 2504.20571 (v3, 2025-10-24) ｜ UW / USC / Microsoft / UC Santa Cruz / Georgia Tech（Yiping Wang*, Qing Yang, Zhiyuan Zeng, Liliang Ren, Liyuan Liu, … Jianfeng Gao, Weizhu Chen, Shuohang Wang*, Simon Shaolei Du*, Yelong Shen*；Yiping 于 MSR 实习完成）｜ NeurIPS 2025 ｜ 主题 T?/High（RLVR 数据效率与机理）｜ 代码 https://github.com/ypwang61/One-Shot-RLVR（已开源，本地已克隆 Tier A）｜ 框架 veRL(GRPO/PPO) + vLLM 0.6.3 + Qwen2.5-Math 评测 pipeline

#### 1. 相关工作与进展
RLVR（以规则化二元 outcome reward 做 RL）已大幅推进 LLM 数学推理（o1、DeepSeek-R1、Kimi-1.5）。但研究多聚焦算法侧（PPO/GRPO/DAPO），数据侧（需要多少、什么数据最有效）相对被忽视。最相关前作 LIMR 用 LIM 分数把训练数据减少约 6 倍仍保持性能。

#### 2. 现有工作存在的问题
- RLVR 训练集究竟能压缩到何种极限尚未探明；
- 数据质量/数量如何关联 self-reflection、跨任务泛化等经验现象不清楚。

#### 3. Motivation
回答一个极限问题：在保持与全量数据相当性能的前提下，RLVR 训练集最少能减到多少？并借此考察 RLVR 增益究竟来自"注入新能力"还是"激发已有潜能"。

#### 4. 主要灵感 / 核心直觉
若单条样本即可激发出接近全量训练的推理能力，则强烈暗示 base model 已蕴含大量推理潜能，RLVR 的作用更偏"激发/对齐"而非"灌输知识"。这把研究重心从算法复杂度转向数据效率与机理。

#### 5. 主要解决思路(一段话讲清核心)
提出 **1-shot RLVR**：仅用单条（或两条）数学样本，按标准 verl GRPO（也验证 PPO）流程在线训练。配套**历史方差分数(Historical Variance Score)**做数据选择——先用全集 RLVR 训 E 个 epoch，记录每条样本逐 epoch 训练准确率列表，按其方差降序排名，高方差样本（如 π₁、π₁₃）在 1-shot 下表现好；但作者强调该准则非最优，许多中低方差样本单独训练也能涨分，说明这是普遍现象。

#### 6. 方法详解(通俗、分步骤)
1. 用全集 RLVR 训 E epoch，逐样本记录逐 epoch 训练准确率 → 算方差 → 排名（Eqn.1/2）。
2. 取排名靠前样本（如 π₁、π₁₃）作 1-shot/2-shot 训练集。
3. 按 verl pipeline 做 GRPO：binary 0-1 outcome reward；KL 系数 β=0.001、entropy 系数 α=−0.001；rollout 温度 0.6（vLLM）；batch=mini-batch=128，每 prompt 采 8 条 → 每 rollout 步 8 次梯度更新；max prompt 1024 / response 3072（上下文 4096）。
4. 评测用 Qwen2.5-Math 官方脚本；AIME/AMC 因题量小重复测试集 8 次、温度 0.6 报 avg@8，其余 3 基准温度 0。

#### 7. 实验数据集
- 训练实例池：DeepScaleR-Preview 的 1209 条子集（DSR-sub）；对比用 MATH 训练集(7500 条)。
- 评测：6 数学基准（MATH-500、AIME24、AMC23、Minerva、OlympiadBench、AIME25）；非数学泛化 ARC-Easy/Challenge。
- 模型：Qwen2.5-Math-1.5B/7B、Llama-3.2-3B-Instruct、DeepSeek-R1-Distill-Qwen-1.5B（附录另含 Qwen2.5-1.5B 等）。

#### 8. 实验结果与主要发现
- **1-shot（Qwen2.5-Math-1.5B）**：MATH500 36.0→73.6%（超 format-correction 增益 +8.6%），6 基准均值 17.6→35.7%（非格式增益 +7.0%），基本匹配含该样本的 1.2k DSR-sub（73.6/35.9）。
- **2-shot（π₁+π₁₃）**：均值 36.6%、MATH500 74.8%，略超 1.2k 子集、与 7.5k MATH 全集相当。
- 现象：(1) **post-saturation generalization**——训练准确率快速逼近 100% 后测试准确率仍持续上升，过拟合到约 1.4k 步才出现，且过拟合后训练样本输出退化为多语言乱码，但测试输出仍可读、性能仍强；(2) 跨类泛化（单类样本提升其他类）；(3) 自反思词频与响应长度上升；(4) 增益主要源自 policy gradient loss，区别于依赖 weight decay 的 grokking；(5) 适当系数 entropy loss 促进探索可进一步提升；(6) 仅用 entropy loss、无 outcome reward 也有（弱于 format-reward 的）提升。
- 跨模型/算法(GRPO、PPO)/不同样本均观察到类似大幅提升。

#### 9. 结果如何支撑其主张
"单样本≈1.2k 子集≈全集"的并列对比直接支撑"RLVR 训练集可极限压缩"；跨模型、跨算法、跨样本的一致性排除了偶然性；post-saturation、跨类泛化、自反思增多等现象共同支撑"激发而非注入"的机理解读。消融（policy gradient loss 为主因、entropy 单独亦有效）把功劳定位到具体损失成分而非数据量。

#### 10. 逻辑自洽性(中性评估)
实证非常扎实、现象丰富，多角度交叉验证使核心结论可信。但有边界需注意：(1) 现象集中在 Qwen2.5-Math 系，其 base 已在数学语料上充分预训练，"激发"叙事对预训练较弱的模型未必成立（Llama-3.2-3B 增益相对有限）；(2) 历史方差分数需先跑全集训练才能算，故"数据选择"本身并不省算力，1-shot 的省是指 RL 阶段；(3) 过拟合后训练样本输出退化为乱码却仍泛化，这一现象机理论文给出观察但未给出完整理论解释。

#### 11. 残留问题 / 局限
- 主要局限于数学、可验证奖励、Qwen-Math 系 base，泛化到其他领域/弱 base 待验。
- 历史方差分数非最优选择准则，且依赖全集预训练，选择阶段不省算力。
- post-saturation 下训练输出退化为多语言乱码的机理未充分解释。
- 单样本训练对超参（entropy 系数、训练步数）较敏感，过训会过拟合。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 代码：https://github.com/ypwang61/One-Shot-RLVR （已开源，本地已克隆，Tier A，约 70MB）。
- 框架：veRL（改编自 verl + rllm/DeepScaleR，仓库内含完整 `verl/` 目录），算法 GRPO（也验证 PPO）；推理 vLLM 0.6.3；评测复用 Qwen2.5-Math 官方 pipeline（`Qwen2.5-Eval/`，latex2sympy）。
- 模型/数据：HF 集合 ypwang61/one-shot-rlvr；数据集 ypwang61/One-Shot-RLVR-Datasets。
- 关键超参：β=0.001、α=−0.001、rollout 温度 0.6、batch=mini-batch=128、每 prompt 8 rollout、max prompt 1024 / response 3072。


---


### orz — Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model

> **一句话重点 (TL;DR)**：用最朴素的 vanilla PPO + GAE(λ=γ=1) + 二值规则奖励、完全不带 KL，就能在 base 模型上稳定地大规模 scale up reasoning RL；首个把代码/数据/各尺寸权重乃至 critic 权重全部开源的 "Reasoner-Zero" 实现。

**元信息**：arXiv 2503.24290（v2 2025-07-05）｜ StepFun + 清华（Jingcheng Hu 等，沈向洋）｜ 2025-03 预印本 ｜ 主题 T3（RLVR/zero-RL 系统），相关性 High ｜ 代码 github.com/Open-Reasoner-Zero/Open-Reasoner-Zero（全开源，本地已克隆约 92MB）｜ 框架 OpenRLHF + vLLM + DeepSpeed + Ray。

#### 1. 相关工作与进展
o1、DeepSeek-R1-Zero 展示了大规模 RL 的 "训练时间 scaling"：随算力增大，benchmark 性能与响应长度同步持续增长、无饱和迹象，并伴随 "Aha moment"。R1-Zero 还证明可直接在 base 模型上启动 RL（跳过 SFT/蒸馏）。但 DeepSeek 仅简述其训练 pipeline，关键细节（数据、超参、value/advantage 估计、稳定化技巧）不公开。

#### 2. 现有工作存在的问题
社区缺乏一个直接在 base 模型上做大规模 reasoning RL 的、全开源且稳定可扩展的实现；对 value/advantage 估计与训练不稳定（尤其 GRPO 因无 value network 易在重复模式上误判、坍缩）缺乏系统分析与可复现配方。

#### 3. Motivation
提供首个鲁棒、可扩展、易跟随的大规模 reasoning-oriented RL on base model 实现，并 democratize 相关技术——完整释放代码、数据、各尺寸权重乃至 critic 权重，降低社区复现门槛。

#### 4. 主要灵感 / 核心直觉
"少即是多"：大规模数据天然降低方差，使无偏配置（GAE λ=γ=1）可行；学习到的 critic 比无 value 的 GRPO 能更准地做 token 级 credit assignment、识别并 devalue 重复等劣化模式；去掉 KL 与 reference model 反而鼓励探索、省显存省调参。

#### 5. 主要解决思路(一段话讲清核心)
直接在 Qwen2.5 base 上用 R1-Zero 风格 prompt 启动 RL，采用极简(minimalist)配方：vanilla PPO + GAE(λ=1,γ=1) + 仅检查 `<answer>` 与参考答案精确匹配的二值奖励，完全不加任何 KL 正则，并配合大规模、多样化数据，即可稳定 scale up 性能与响应长度。

#### 6. 方法详解(通俗、分步骤)
- **选 PPO 而非 GRPO**：学习到的 critic 给出更准的 token 级 value 与 credit assignment；分析显示 PPO 对重复 token 赋更负的 advantage，能抑制坍缩。
- **GAE λ=1, γ=1**：无偏配置充分捕捉长程依赖；优势简化为 Â = R − V_φ(s_t)，value 目标 (V_φ(s_t) − R)²。
- **去掉 KL**：免去 reference model 的显存/计算与调参，鼓励探索。
- **极简 reward**：二值（1/0）精确匹配，无 format reward，reward hacking 空间最小；base 模型也能很快学会正确格式。
- **scale up data**：数据规模与多样性对持续提升至关重要。
- **采样/训练细节**：每步 128 prompt × 每 prompt 64 response，temperature/top-p=1.0；严格 on-policy；batch-level advantage 归一化。32B 末段加 100 步 annealing（13k 难题）。〔已核 repo playground/orz_32b_ppo.py：gamma=lambd=1.0、init_kl_coef=0、kl_loss_coef=0.0（use_kl_loss=True 但系数为 0，即 KL 实际关闭）、n_samples_per_prompt=64〕

#### 7. 实验数据集
- 训练：ORZ 精选数据（正文表述 "tens of thousands of curated QA pairs"；开源含 orz_math_57k / orz_math_72k_extended / orz_math_13k_hard；来源 AIME(≤2023)、MATH、Numina-Math、Tulu3 MATH、OpenR1-Math-220k、AoPS 论坛 + 程序合成的逻辑/多步/反事实题）；排除证明题等难评测题，并用 LLM 过滤极端 pass rate。对照实验用 ORZ-57k vs MATH-train-7.5k。
- 评测：AIME2024、AIME2025、MATH500、GPQA Diamond（均 avg@16）；泛化 MMLU、MMLU_PRO。

#### 8. 实验结果与主要发现
- 基座 Qwen2.5-{0.5,1.5,7,32}B base，直接大规模 RL、跳过 SFT。
- 主结果：ORZ-32B 在 AIME2024(48.1)、MATH500(92.2)、GPQA Dia.(55.5) 上超越或持平 DeepSeek-R1-Zero-Qwen-32B(47.0/91.6/55.0)，且**仅用约 1/10 训练步数**；MMLU/MMLU_PRO 超 Qwen2.5-Instruct-32B。〔已核 PDF Table，行 318-321/968-971〕
- 消融：GAE λ=1.0 优于 0.95；去 KL 优于 KL Loss/KL Penalty；ORZ-57k 优于 MATH-7.5k（后者早早 plateau）。
- 扩展：对蒸馏模型续做 ORZ（两阶段，类 R1）得 ORZ-R1-Distill-Qwen-14B，超过更大的 R1-Distill-Qwen-32B。

#### 9. 结果如何支撑其主张
主张"极简配方即可稳定 scale up"由两类证据支撑：(a) 训练曲线显示性能与响应长度随步数同步增长无饱和；(b) 跨尺寸（0.5→32B）一致受益，且 32B 以 ~1/10 步数追平 R1-Zero。逐项消融（GAE/KL/数据规模）直接对应配方中每个设计选择，因果链较完整。

#### 10. 逻辑自洽性(中性评估)
配方主张与消融基本一一对应，可信度较高。需注意：PPO 优于 GRPO 的论证主要依赖作者自身曲线与 "advantage on repeated token" 的定性分析，并非对所有任务/尺度的普适结论；"1/10 步数" 的对比依赖与 DeepSeek 复现条件的可比性（数据、prompt 不同），属同向但非严格控制比较。

#### 11. 残留问题 / 局限
- 与 R1-Zero 的步数对比跨实现/跨数据，严格可比性有限。
- 仅评测数学+少量通用基准；对代码、agentic 等域的可迁移性未充分验证。
- "去 KL 总更好" 的结论在 base 模型起点成立，迁移到已对齐/已蒸馏起点时不一定（参见 ProRL 反向主张保留 KL）。
- 二值奖励对证明题、开放式任务不适用。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库：https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero（本地已克隆约 92MB，含 orz/ 包、playground/ 各尺寸训练脚本、docker/）。
- 模型/数据：HuggingFace Open-Reasoner-Zero（ORZ-{0.5,1.5,7,32}B、ORZ-R1-Distill-Qwen-14B、critic 权重、ORZ 数据）。
- 框架：OpenRLHF（+ vLLM + DeepSpeed + Ray），实现 vanilla PPO + GAE 的大规模分布式训练。代码可得性：完整（含 critic 权重，开放程度在同类工作中最高）。


---


### prefix_rft — Prefix-RFT: Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling

> **一句话重点 (TL;DR)**：从 demonstration 采一段前缀作为 off-policy 引导、让当前策略续写作为 on-policy 探索，整条混合轨迹一起进 RFT；用前缀感知 advantage、熵约束裁剪与余弦衰减的前缀长度调度，把 SFT 的过程监督与 RFT 的目标导向优化在单阶段内融合。

**元信息**：arXiv 2507.01679（v3 2026-05-15）｜ 爱丁堡 ILCC + 复旦 + 阿里 Qwen + StepFun + UvA ILLC ｜ ICML 2026 ｜ 主题 T3/T4（SFT-RL 统一），相关性 High（前缀=教师脚手架、续写=自主路径，对应路径选择/恢复）｜ 代码 github.com/ZeroYuHuang/prefix_rft（已克隆约 9.1MB，完整）｜ 框架 veRL（recipe/prefix_rft）。

#### 1. 相关工作与进展
LLM 后训练两大范式：SFT（模仿 demonstration，简单、注入知识）与 RFT（奖励驱动，提升能力但依赖初始策略）。近期出现并行融合方法 ReLIFT、UFT、LUFFY（LUFFY 把整条 demonstration 混入 on-policy RFT），以及 SimpleRL-Zero、Oat-Zero 等 Zero-RL。

#### 2. 现有工作存在的问题
- SFT 是 behavior cloning，泛化/鲁棒性差。
- RFT 奖励稀疏、难做 token 级 credit assignment（易致 language mixing 等异常），效果强依赖初始策略，被质疑只是 "打磨已有能力" 而非提升上界。
- 二者通常被当作两个独立串行阶段，缺乏形式化统一框架；LUFFY 等混入整条 demo 又会限制自主探索、难突破性能上界。

#### 3. Motivation
提出 SFT 与 RFT 的统一视角：二者核心更新动力学一致（SFT 相当于把 advantage 隐式置 1 的策略梯度；PG 用 advantage 加权；PPO 再加 per-token clip）。核心问题：如何把 SFT 的过程监督与 RFT 的目标导向优化形式化融合，既起步可靠又保留探索？

#### 4. 主要灵感 / 核心直觉
(1) 高质量 prefix 是比纯 RFT 更强的探索引导——若混合轨迹得高奖励，则该 prefix 被强化；(2) 相比给整条 demo 的 SFT，只给前缀保留了 RFT 的解题目标与 "受约束的自主性"：沿可靠路径起步、但仍可探索更优续写。

#### 5. 主要解决思路(一段话讲清核心)
从 demonstration 采样一段 prefix，让当前策略续写得到 continuation；组合序列 = off-policy prefix + on-policy continuation，作为一条 trajectory 与标准 on-policy rollout 一起参与 RFT advantage 估计与 PPO 更新。

#### 6. 方法详解(通俗、分步骤)
- **前缀感知 advantage**：`compute_grpo_prefix_outcome_advantage` / `compute_dr_grpo_prefix_outcome_advantage`——prefix token 与 continuation token 分别按 prefix 分组归一化，prefix advantage 经 `/num_rollouts_per_prefix` 缩放后用 `prefix_mask` 写回。〔已核 recipe/prefix_rft/core_algos.py:162-215, 262-307〕
- **策略损失**：on-policy 部分用 PPO dual-clip（clip_ratio_low/high，clip_ratio_c=3.0），off-policy(prefix) 部分用独立的 `cliprange_low_off/high_off` 裁剪（`enable_clip` 控制），按 prefix_mask 合并。
- **熵约束裁剪（核心组件）**：因 π_off 可能远离当前策略、prefix token 的 π_θ 概率普遍偏低，其梯度易压制 RFT 梯度；故只对 **top-k% 高熵 prefix token** 计入梯度，其余 prefix token advantage 置零（保守 off-policy 滤波：低熵 token 要么已被匹配信号小，要么是会引发尖锐覆写的 "自信错配"）。代码以 dp_actor.py 的 entropy "reshaper"（off_adv_reshaper：entropy/entropy_low/random masking）实现。〔已核 recipe/prefix_rft/dp_actor.py:64-86〕
- **前缀长度余弦衰减调度**：L=⌊l·|y\*|⌋，l~U(low, high)；high 为常数，low 全程从 high 余弦衰减到近零——既缓解 "只学开头 token" 的位置偏置，又内置课程学习（由 "几乎给全 demo" 过渡到 "几乎纯 RFT"，对应 SFT→RFT 配方）。代码 scheduler/global_step.py 的 `cosine_decay` controller；avg_score.py 另可按平均分调度。〔已核〕
- **默认超参（config/prefix_rft_trainer.yaml，已核）**：clip_ratio_low=high=0.2、clip_ratio_c=3.0、entropy_coeff=0.001；yaml 中 adv_estimator=gae 为上游模板残留，prefix recipe 运行时改用 prefix-aware GRPO/Dr.GRPO。

#### 7. 实验数据集
- 训练（已核 PDF §4）：OpenR1-Math-220K 的长度过滤子集（沿用 LUFFY/Yan et al.）约 **46k 题**，每题配一条 **DeepSeek-R1 生成的 demonstration**（即 prefix 来源）。
- 主基座：**Qwen2.5-Math-7B**（Table 1 主实验）；另在 Qwen2.5-Math-1.5B、LLaMA-3.1-8B、Qwen3-1.7B-base 上验证（Tab.6）。
- 评测：6 个数学基准（AIME24、AIME25、AMC、MATH-500、Minerva、OlympiadBench）+ 3 个通用域（ARC-c、GPQA*、MMLU-Pro）；另用 **AIME pass@2024**（大 k 采样）评估推理能力边界扩展。
- 基线：Zero-RL（SimpleRL-Zero、Oat-Zero）、同基座同数据的 RFT/SFT/RFT w/ SFT-Loss/SFT+RFT，及并行混合方法 ReLIFT、UFT、LUFFY。

#### 8. 实验结果与主要发现
- veRL recipe：每 prompt N 条序列中 N−1 条纯 on-policy rollout、第 N 条为 "prefix(off-policy)+continuation(on-policy)" 混合轨迹（**非额外增加一条，rollout 预算与标准 RFT 一致**），全部共同估计 prefix-aware advantage，actor 用带 off-policy 分支的 dual-clip PPO 更新。〔已核〕
- 主结论：Prefix-RFT 超越纯 SFT、纯 RFT、SFT→RFT 两阶段，以及并行 mixed-policy 方法（LUFFY 等）；对 demonstration 质量/数量鲁棒；能扩展推理能力边界（AIME pass@2024 提升），而 LUFFY 未能显著推高上界。
- 消融：熵约束裁剪、前缀长度调度均被验证为有效组件。

#### 9. 结果如何支撑其主张
"统一且更优" 由与 SFT/RFT/两阶段/并行混合四类基线的全面对比支撑；"扩展能力上界" 由 pass@2024（大 k）相对 LUFFY 的提升支撑；"鲁棒" 由 demonstration 数量/质量消融支撑。证据与主张对应较紧密。

#### 10. 逻辑自洽性(中性评估)
统一视角（SFT=advantage≡1 的 PG 特例）推导清晰，方法是该视角的自然实例化。熵约束裁剪与长度调度有合理直觉且有消融背书。需注意：yaml 默认 adv_estimator=gae 与正文 GRPO/Dr.GRPO 不一致，属模板残留（运行时覆盖），易误读但不影响结论。

#### 11. 残留问题 / 局限
- 实验集中于数学推理 + 单一 DeepSeek-R1 demonstration 来源；跨域（代码、通用 agent）泛化未充分展开。
- top-k% 高熵阈值、前缀长度 (low,high) 区间为关键超参，对其敏感性的系统分析有限。
- num_rollouts_per_prefix 缩放的默认取值随 run 脚本设定，正文未给统一最优值。
- 依赖高质量 demonstration 可得；无 demo 场景退化为纯 RFT。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库：https://github.com/ZeroYuHuang/prefix_rft（CloneTier=A，本地约 9.1MB，完整）。核心：recipe/prefix_rft/（core_algos.py 多种 prefix advantage、dp_actor.py 熵 reshaper、ray_trainer.py、rl_dataset.py、templates.py、scheduler/）。
- 框架：veRL（全部 prefix-rft 代码集中在 recipe/prefix_rft）。
- 代码可得性：完整开源，关键算法（前缀 advantage、熵裁剪、余弦调度）均可逐行核验。


---


### prorl — ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in LLMs

> **一句话重点 (TL;DR)**：用 KL 正则 + 周期性参考策略硬重置 + DAPO 解耦 clip/动态采样稳定住长程 RL，让 GRPO 训练能跑 2k+ 步而不熵坍缩，从而在 base 模型即使大量采样也无法触及的任务上发现新推理策略——主张 RL 确能扩展（而非仅放大）推理边界。

**元信息**：arXiv 2505.24864（v1 2025-05-30）｜ NVIDIA（Mingjie Liu 等，含 Yejin Choi、Jan Kautz）｜ 2025-05 预印本（under review）｜ 主题 T3（RL 算法/训练），与长程 RL、探索-稳定性直接相关 ｜ 代码 **仅发布模型权重**（无训练代码仓库）｜ 框架 veRL。

#### 1. 相关工作与进展
o1、DeepSeek-R1 等推理模型通过 test-time scaling（长 CoT、探索/验证/回溯）在数学、代码等任务上大幅提升，RL（针对可验证奖励 RLVR）是核心驱动并能缓解 reward hacking。

#### 2. 现有工作存在的问题
- 流行观点认为 RL 不带来超越 base 的新能力（base 即使大量采样也做不出的题，RL 后仍不行），只是放大已潜在的高奖励输出。
- 长程 RL 面临**熵坍缩**与不稳定：输出分布过早变尖、熵骤降、探索受限；GRPO 依赖多样采样估相对优势，熵坍缩使更新偏置、训练停滞。
- 单纯提高采样温度只能延缓而非阻止熵坍缩。

#### 3. Motivation
挑战 "RL 不扩展能力" 的假设：主张通过**延长 RL（ProRL）**配合恰当稳定化，能发现 base 即使大量采样也触及不到的**新推理策略**，并验证推理边界提升与 base 任务能力、训练时长强相关——说明 RL 能随时间探索并填充新解空间。

#### 4. 主要灵感 / 核心直觉
"边界扩展需要持续的探索预算"：只要能压住熵坍缩、让训练长期保持探索且不偏离稳定参考太远，RL 就能在 2k+ 步内持续涌现新颖（与预训练语料重叠低、Creativity Index 高）的解法，而非早早收敛到 base 分布内的高奖励 mode。

#### 5. 主要解决思路(一段话讲清核心)
以 GRPO 为基座（去 critic、组内标准化优势），叠加 DAPO 的解耦 clip + 动态采样、KL 正则、以及**周期性把参考策略硬重置为近期在线快照**，从而维持熵、防偏离、避免 KL 项随训练主导损失而停滞，支撑 "prolonged" 训练。

#### 6. 方法详解(通俗、分步骤)
基座 **GRPO**：A(τ)=(R−mean)/std。叠加：
1. **DAPO 两组件**：
   - **Decoupled Clip（clip-higher）**：把 PPO 上下 clip 界拆成独立 ε_low/ε_high，调高 ε_high 提升低概率 token、鼓励探索、保留熵、减少过早 mode collapse。
   - **Dynamic Sampling**：过滤一贯全对(acc=1)/全错(acc=0) 的 prompt，聚焦中等难度、维持学习信号。
2. **KL 正则 + 参考策略重置（关键创新）**：
   - L = L_GRPO − β·D_KL(πθ‖π_ref)，维持熵并防偏离稳定参考、抑制对 spurious reward 过拟合。
   - 反 "去 KL" 主流观点：因本工作从已能产生连贯 CoT 的良好起点（DeepSeek-R1-Distill-Qwen-1.5B）出发，保留 KL 有益于稳定与持续熵。
   - **Reference Policy Reset**：训练推进后 KL 项渐主导、更新变小；故周期性把 π_ref **硬重置**为近期在线快照并重置优化器状态，在保留 KL 收益的同时持续提升、避免过早收敛。

#### 7. 实验数据集
- 训练：自构 **136K** 可验证问题，覆盖数学、代码、STEM、逻辑谜题（Reasoning Gym）、指令遵循，每类配清晰奖励（二值或连续）。
- 评测：跨域 pass@k；与 DeepSeek-R1-Distill-Qwen-1.5B 及领域专用基线对比；用 **Creativity Index** 衡量推理轨迹与预训练语料的重叠（越低越新颖）。

#### 8. 实验结果与主要发现
- 框架 veRL；基座 DeepSeek-R1-Distill-Qwen-1.5B；产出 **Nemotron-Research-Reasoning-Qwen-1.5B**（号称当时最强 1.5B 推理模型）。
- 超参（已核 §3）：GRPO+DAPO 解耦 clip ε_low=0.2、ε_high=0.4；动态采样过滤 acc=0/1；每 prompt n=16；高采样温度 **1.2**；batch 256、mini-batch 64（每 rollout 步 4 次更新）；AdamW 常数 lr 2×10⁻⁶；约 16k GPU·小时（4×8 H100）；多数训练 response 上限 8k，末段约 200 步提至 16k。
- 主结果（相对 base，已核 §3 行 201-202/305-308）：数学 **+15.7%**、代码 **+14.4%**、STEM **+25.9%**、指令遵循 **+22.0%**、逻辑谜题 **+54.8%**；并超领域专用基线（数学 +4.6%、代码 +6.5%）。
- 延长训练得到更高 Creativity Index（新颖模式涌现）；RL 模型在 base 完全失败（任意尝试次数）的场景仍优于 base，支撑 "RL 能扩展推理边界、无需额外数据"。
- 〔NEW 待核〕**论文内部数值不一致**：摘要（行 104-105）报数学 +14.7%、代码 +13.9%、STEM +25.1%、指令 +18.1%、逻辑 +54.8%；而 §3/详细结果（行 201-202、305-308）报 +15.7%/+14.4%/+25.9%/+22.0%/+54.8%。本分析采用 §3/结果节一致的后一组（与 Table 数值对应），但摘要存在不同口径，原因（不同采样/口径）论文未说明。

#### 9. 结果如何支撑其主张
"RL 扩展边界" 由两点支撑：(a) base 任意采样次数仍失败的题上 RL 模型有正确解；(b) Creativity Index 随训练上升（与预训练语料重叠下降）。"延长有效" 由 2k+ 步持续提升曲线支撑。稳定化各组件由消融/曲线说明对熵的维持作用。

#### 10. 逻辑自洽性(中性评估)
内在逻辑自洽且与 ORZ "去 KL" 的对立主张可调和——ProRL 明确把适用前提限定在 "良好初始化（已蒸馏）起点"。但 "扩展边界" 的核心证据（pass@k 在大 k 下 base 仍 0）对 k 的取值与采样温度敏感，结论强度依赖该测度的稳健性。摘要与正文数值不一致削弱了表面严谨性（虽不改变定性结论）。

#### 11. 残留问题 / 局限
- 仅 1.5B 单一规模、单一 base（R1-Distill）；更大模型/不同起点的可迁移性未验证。
- "扩展边界 vs 放大分布" 的判定依赖 Creativity Index 与大 k pass@k，两者皆有测度争议。
- 摘要 vs 正文增益数值不一致（见 §8 NEW）。
- 无独立训练代码，复现依赖对 veRL + 描述超参的自行拼装。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仅发布**模型权重**：https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B 。无独立训练代码仓库。
- 框架：veRL（verl-based）；算法 = GRPO + DAPO（解耦 clip + 动态采样）+ KL 正则 + 参考策略重置。
- 代码可得性：paper-only（weights-only，无训练代码，需依论文超参在 veRL 上复现）。


---


### psft — Proximal Supervised Fine-Tuning (PSFT)

> **一句话重点 (TL;DR)**：把 SFT 视为 "advantage 恒为正(A=1)、采样自固定离线数据集" 的策略梯度特例，给它套上 PPO/TRPO 式信任域裁剪约束策略漂移；代价是 in-domain 略低于普通 SFT，换来更强 OOD 泛化、不熵坍缩、以及作为后续 RL 起点更优。

**元信息**：arXiv 2508.17784（v2 2026-04-12）｜ 上海交大 + 上海创智学院 + 腾讯大模型部 + 澳门大学 ｜ ICLR 2026 ｜ 主题 T3（SFT-RL 统一视角下的 "改进版 SFT"），相关性 High ｜ 代码 github.com/zwhong714/PSFT（CloneTier=A，已克隆约 23MB，完整）｜ 框架 veRL（recipe/psft）。

#### 1. 相关工作与进展
社区大量利用强推理模型产出的高质量轨迹，通过 SFT（一种蒸馏）注入推理能力，因其相比 RL 简单高效。SFT-RL 统一视角（如 Prefix-RFT 等）逐渐兴起。

#### 2. 现有工作存在的问题
- SFT 本质是 behavior cloning，泛化差：数据次优或与预训练分布失配时会引发过大的策略更新，损害原有能力。
- SFT 易致**熵坍缩**（作者图示约 150 step 后熵骤降至近零），削弱探索，约束后续 RL。
- 挑战：如何同时改善 SFT 模型的泛化与探索能力。

#### 3. Motivation
建立 SFT 与 RL 的理论联系：SFT 是 **advantage 恒为正(A=1)、采样自固定离线数据集**的策略梯度特例。借鉴 TRPO/PPO 的信任域思想，给 SFT 加 PPO 式 clipped surrogate，约束策略漂移，避免死记硬背并防熵坍缩。

#### 4. 主要灵感 / 核心直觉
"SFT 之所以伤泛化、垮探索，是因为它对每个 demo token 都施加无界的最大似然推力"；若把这股推力用重要性比裁剪封顶（信任域），既保留模仿、又把更新限制在策略附近，就能在不引入显式 KL/reference 的情况下保住熵与原有能力。

#### 5. 主要解决思路(一段话讲清核心)
把 demonstration 当作 "trajectory"、advantage 设为常数正值(A=1)，用 r_t=π_θ/π_old 的重要性比 + PPO 式非对称 clip(1−ε_low, 1+ε_high) 构造信任域约束的代理目标，替代普通 SFT 的最大似然，靠 trust-region（非显式 KL）约束漂移。

#### 6. 方法详解(通俗、分步骤)
- 将 SFT 改写为带重要性比裁剪的代理目标；advantage 恒正(A=1)。
- 非对称 clip：run 脚本 clip_ratio_low=0.2、clip_ratio_high=0.28（high>low，与 DAPO clip-higher 同思路，鼓励探索）；use_kl_in_reward=False、use_kl_loss=False、kl_coef=0（靠 trust-region 而非显式 KL）。〔已核 verl/recipe/psft/run_psft.sh:8-16,80（clip_ratio_c=10.0）〕
- **adv_estimator=psft 的实现**：并非独立 `@register_adv_est` 函数，而是在 ray_trainer.py 的 `compute_advantage` 中内联分支——当 adv_estimator==PSFT 时直接令 `advantages = returns = ones_like(response_mask) * response_mask`，即每个 response token 赋恒定 advantage=1（被 response_mask 掩到回复 token）。随后 actor 用 PPO 非对称 clip surrogate 更新。〔已核 verl/verl/trainer/ppo/ray_trainer.py:274-276、core_algos.py:104 枚举 PSFT="psft"〕
- π_θold 动态更新（每 4/8/16 步刷新一次，过快或不更新都损害效果，论文取 update.8）；可选 warm-up SFT 先对齐 π_θold。
- 效果：训练全程不熵坍缩，保持生成多样性。

#### 7. 实验数据集
- 数学推理（主实验，SFT 阶段）：训练用 **OpenR1-Math-8192** long-CoT 数据集（HF Open-R1，§4.1.1 明示）；RL 阶段用 **DAPO-MATH-17k**（§4.2，clip-higher 0.28）。〔注：开源 run_psft.sh 默认 TRAIN_FILE 指向 `wh-zhu/train_openr1_4k`（4k 变体），与正文主实验 8192 版略有出入，属仓库默认脚本配置差异，已核〕
- 基座：Qwen2.5-7B-Instruct、Llama3.1-8B-Instruct。
- 评测：in-domain 数学 = AIME-24/25、AMC（avg@32）、MATH-500、OlympiadBench、Minerva（avg@8）；OOD = GPQA、ARC-C、TruthfulQA、IFEval（avg@8）、MMLU-Pro、SuperGPQA、HeadQA（pass@1）。推理长度 10,240 token（IFEval 4,096），top-p 0.95，temperature 0.7。
- 普适性另覆盖：human-value alignment（Qwen3-4B-Base + UltraFeedback，后接 DPO）与多模态（Qwen2.5-7B-VL-Instruct，文本 OpenR1-Math-4096、几何 Geometry-3k，后接 GRPO）。

#### 8. 实验结果与主要发现
（Table 1，Qwen2.5-7B-Instruct，全部已核对 PDF 行号）：
- **in-domain：PSFT 略低于标准 SFT**，并非更高。AIME-24 SFT 22.08 / PSFT 19.38 / PSFT-warm-up 22.92（行 261-265）；in-domain 6 项均值 SFT 47.99 / PSFT 46.98 / warm-up 48.17（行 328-331）。即 "持平、warm-up 后可反超"，绝非凭空领先。
- **OOD 泛化 PSFT 明显更优**：OOD 7 项均值 SFT 57.90 / PSFT 61.26（行 417-420）；GPQA 32.89 / 33.21；TruthfulQA SFT 63.14 / PSFT 67.16（行 362-365）；IFEval SFT 54.42 / PSFT 73.03（行 406-409，SFT 严重损伤指令遵循，PSFT 基本保住）。
- Llama3.1-8B-Instruct 同向：OOD 均值 SFT 50.49 / PSFT 59.25（行 422-425）。
- **作为 RL 起点更优（Table 2）**：PSFT→GRPO 在 in-domain 与 OOD 均超 SFT→GRPO——Qwen in-domain 均值 53.31 vs 52.40、OOD 均值 64.06 vs 59.90（行 535-538、608-611）；PSFT 起步虽慢但因保留更高熵、探索空间大，RL 后反超。
- 普适性：human-value alignment（PSFT→DPO 在 AlpacaEval2/Arena-Hard/MT-Bench 优于 SFT→DPO 并降 alignment tax）与多模态（PSFT→GRPO 在 MMMU/MMMU-Pro 等保持泛化而 SFT 显著退化）。

#### 9. 结果如何支撑其主张
主张是 "牺牲少量 in-domain 拟合换取泛化+探索+更好 RL 起点"，证据链与之**完全对齐且不夸大**：(a) in-domain 略降被如实报告；(b) OOD 全面提升（IFEval/TruthfulQA 提升尤显）；(c) 熵曲线不坍缩；(d) Table 2 下游 RL 反超。证据方向一致、内部统一。

#### 10. 逻辑自洽性(中性评估)
理论联系（SFT=A≡1 的 PG）清晰，方法是该视角的直接实例化，代码实现与正文 "A=1" 严格一致（已逐行核）。关键优点：论文坦诚 in-domain 略逊，不做选择性汇报——这与上一轮被修正的 "凭空领先" 表述相反，本次复核确认现有 §8 数字与论文 Table 1/2 **逐项吻合，无新捏造**。clip_ratio_high>low 与 DAPO 同思路，自洽。

#### 11. 残留问题 / 局限
- in-domain 拟合略逊普通 SFT，对追求纯 in-domain 峰值的场景不利。
- 依赖 π_θold 更新频率（4/8/16）这一敏感超参，论文经验性取 8。
- 〔注〕Table 2 表头标 SFT→GRPO/PSFT→GRPO，而 §4.2 RL setup 文字用 "DAPO(clip-higher 0.28)+DAPO-MATH-17k"；论文 GRPO/DAPO 表述并存（DAPO 即带 clip-higher 的 GRPO 变体），本文 §7 用 DAPO、§8 沿用 Table 2 标签 GRPO，均忠实原文。
- 仓库默认脚本数据（4k）与正文主实验（8192）不一致，可能影响直接复现峰值。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库：https://github.com/zwhong714/PSFT（CloneTier=A，本地约 23MB，完整）。实现：verl/recipe/psft/（main_psft.py、psft_ray_trainer.py、config/psft_trainer.yaml、run_psft.sh / run_sft.sh）；adv_estimator=psft 内联于 ray_trainer.py。模型权重：HF collection `wh-zhu/psft-...`。
- 框架：veRL（recipe 内实现）；torch2.6.0+cu124+vllm0.8.5。
- 代码可得性：完整开源，PSFT 的核心（A=1 内联分支 + 非对称 clip）可逐行核验。


---


### raft_reinforce_rej — A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce (RAFT / RAFT++ / Reinforce-rej)

- **arXiv/链接**: arXiv:2504.11343 (v2, 2025-06-12); https://arxiv.org/abs/2504.11343
- **机构/作者**: Salesforce AI Research + UIUC。Wei Xiong, Jiarui Yao, Yuhui Xu, Bo Pang, Lei Wang, Doyen Sahoo, Junnan Li, Nan Jiang, Tong Zhang, Caiming Xiong, Hanze Dong(共同通讯)。
- **发表/时间**: arXiv 预印本,2025-04(v2 2025-06)。
- **主题/相关性**: T3。从拒绝采样(RAFT,本质是在自生成正例上的 SFT)到 Reinforce/GRPO 的统一最小化视角,剖析当前 RL 实践成功的关键因子。属 GFT-class / SFT-RL 统一谱系的关键分析性工作。

#### 1. 开源代码链接
https://github.com/RLHFlow/Minimal-RL 。

#### 2. 使用框架
veRL(仓库内置 verl 源码,入口 `verl.trainer.main_ppo`)。FSDP + vLLM。提供脚本 `scripts/run_raft.sh`、`run_raftpp.sh`、`run_reinforce_rej.sh`、`run_grpo.sh`、`run_ppo.sh`(无独立 vanilla `run_reinforce.sh`;vanilla RAFT/Reinforce 通过 `policy_loss=vanilla` 切换,RAFT++ 用 `policy_loss=plusplus`)。〔已核-脚本目录〕

#### 3. 研究背景
RLVR(可验证奖励的 RL)已成为提升 LLM 数学推理的主流。GRPO 因 DeepSeek-R1 的成功被广泛采用,但其算法细节缺乏文档,且不清楚其优势究竟来自算法本身还是延续性惯例。作者系统重审一组算法以找出当前 RL 实践成功的关键因子。

#### 4. 当前存在的问题
- GRPO 相对 vanilla Reinforce 的增益来源不明(奖励归一化 vs 隐式过滤)。
- 仅用正例训练(RAFT/拒绝采样)虽收敛快但会过早熵坍缩、性能停滞。
- 对所有采样回答全错的 prompt 训练会显著损害 on-policy 方法性能。

#### 5. Motivation
以"从拒绝采样到 Reinforce"的连续谱系作为最小化基线,逐项消融,搞清楚:正/负样本各自的作用、prompt 过滤的作用、奖励归一化的作用,从而提出一个简洁却接近 GRPO 的新变体。

#### 6. 主要方法
重审并实现三类算法 + 一个新变体(advantage 估计在 `verl/trainer/ppo/core_algos.py` 中确认):

1. **RAFT(拒绝采样 = 在正例上的 SFT)** —— `compute_raft_outcome_advantage`:对每条回答取 outcome reward 之和,**将 <0 的分数截断为 0**(`scores[scores<0]=0`),即只对正确/正奖励样本产生梯度。配 `policy_loss='vanilla'`(`compute_policy_loss_vanilla`),其 `pg_losses1 = -advantages · log_prob`,**无重要性采样比、无 clip**,本质是对正例的加权对数似然(SFT)。

2. **RAFT++** —— 在 vanilla RAFT 基础上加入 **重要性采样 + clipping**(`policy_loss='plusplus'` → `compute_policy_loss`,PPO 风格 ratio = π_θ/π_θold,带 dual-clip)。损失:L_Reinforce(θ) = (1/|D|) Σ min(ratio·Â, clip(ratio)·Â)。

3. **Vanilla Reinforce / GRPO** —— Reinforce 为去掉 critic 的 PPO 简化版;GRPO(`compute_grpo_outcome_advantage`)对每 prompt 采 n 条回答,用组内 mean/std 归一化得相对优势。

4. **新变体 Reinforce-rej** —— `compute_reinforce_rej_outcome_advantage`:对每个 prompt(按 `index` 分组)计算组内分数标准差,**若 std>0 保留该组的优势信号,若 std==0(即全对或全错)则把优势置 0**(`scores[i]-scores[i]`),即选择性过滤掉"全对或全错"的 prompt。

#### 7. 实验数据集
- 训练:NuminaMath / MATH(脚本默认 `data=numina_math`,也支持 `math`;数据预处理脚本在 `scripts/` 下:`math_dataset.py`、`numina_math.py`、`gsm8k.py`)。
- Backbone:Qwen2.5-Math-7B-base 与 LLaMA-3.2-3B-instruct(用各自默认 chat template;脚本示例 Qwen2.5-Math-1.5B,n=4,lr=1e-6,use_kl_loss=True)。
- 评测:MATH500、Minerva Math、Olympiad Bench(报告 average pass@16 等;不含 AIME2024,因仅 30 题趋势不稳)。
- 结果(Qwen2.5-Math-7B-base):Base 23.6 → RAFT 52.3 → RAFT++ 56.1(三基准均值);RAFT++ 早期收敛更快、接近 GRPO/PPO;Reinforce-rej 终态性能与 GRPO 相当且 KL 效率更优。

#### 8. 怎么做的(训练/数据/流程)
1. 对每个 prompt 用当前策略采样 n 条回答,verifier 给可验证奖励。
2. 按所选 `adv_estimator` 计算优势:RAFT 截断负值→只学正例;Reinforce-rej 过滤全对/全错 prompt;GRPO 组内归一化。
3. actor 用对应 `policy_loss`:`vanilla`(无 IS/clip,等价正例 SFT)或 `plusplus`(PPO 式 IS+clip)更新;可选 KL loss。
4. 通过消融对比得出主要结论:GRPO 优于 Reinforce 主要来自对"全错 prompt"的隐式过滤,而非 mean/std 奖励归一化;仅正例训练加速收敛但致熵坍缩,负样本对维持探索/防分布坍缩至关重要;由此 Reinforce-rej 兼顾性能与 KL 效率。


---


### revisit_entropy — Revisiting Entropy in Reinforcement Learning for Large Reasoning Models

- **arXiv/链接**: https://arxiv.org/abs/2511.05993 (v3, 2026-04-19；v1 2025-11-08)
- **机构/作者**: Renren Jin、Pengzhi Gao、Yuqi Ren 等；天津大学 TJUNLP Lab、天津师范大学、独立研究者。通讯：Deyi Xiong（熊德意）
- **发表/时间**: 2025-11（arXiv），cs.CL
- **主题/相关性**: RLVR 中熵动态的系统性实证研究 + 一个轻量熵调控方法。属于"熵机制/熵崩溃"研究线，对理解 RL 阶段探索-利用权衡有参考意义；与 OPD 关联间接。

#### 1. 开源代码链接
https://github.com/cordercorder/EntropyRL （已 clone，约 7.8MB；以 verl 为底，`train_scripts/` 提供 ada_ent_reg.sh、clip_cov.sh、kl_cov.sh、entropy_adv.sh、pos_adv_reweight.sh、rand_pos_clip.sh 及 reward 脚本）。代码真实可用；README 说明 Clip-Higher、Adv≤0/≥0 等基线只需对 veRL 做小改、未单独提供脚本。

#### 2. 使用框架
veRL（Sheng et al. 2025），GRPO 训练。

#### 3. 研究背景
RLVR 是提升 LLM 推理能力的主流范式，但训练中策略熵常崩溃，导致过早收敛到次优局部最优、阻碍进一步提升。已有缓解方法（熵正则、Clip-Higher、Clip-Cov/KL-Cov、CE-GPPO、80/20 等）众多，但对 RLVR 中熵本身缺乏系统研究。

#### 4. 当前存在的问题
三个未被充分探讨的问题：(1) RLVR 训练中 LLM 熵与性能如何关联？(2) 什么因素从理论与实证上支配熵动态？(3) 如何有效调控熵以提升性能？

#### 5. Motivation
通过大规模实证回答上述三问，并基于"正优势 token 是熵崩溃主因"这一理论+实证结论，设计可精确调控熵的轻量方法。

#### 6. 主要方法
核心实证发现：
- 熵与响应多样性强正相关；训练中 in-domain prompt 熵下降快于 out-of-domain；prompt 熵与准确率仅弱相关。
- 性能可在熵不被牺牲的情况下持续提升（自适应熵正则把熵维持在训练前水平时，AIME24 准确率反而更高）。
- 熵与性能相关性高度依赖任务与指标（如仅用数学数据训练时，LiveCodeBench 的 Avg@64 与熵强负相关，其他基准弱相关）。
- 熵崩溃伴随 miscalibration（过度自信）；崩溃越重、校准越差。
- 三个影响熵动态的因素：裁剪阈值、off-policy 更新次数、训练数据多样性（约 600 条样本可达约 17k 样本的性能）。
- 理论+实证证明：正优势 token 是熵崩溃主驱动。
方法 Positive-Advantage Reweighting（Pos-Adv-Reweight）：用超参 λ 动态调节正优势 token 的损失权重，三个变体——(a) Stage-based：前半段 λ=0（只用非正优势 token），后半段 λ 从 0 线性升到 1；(b) Epoch-wise：λ 按 epoch 线性 (e−1)/(E−1)；(c) Entropy-guided：当熵>阈值 δ 时 λ+Δ（抑熵）、否则 λ−Δ（增熵），把熵稳在 δ 附近（δ=0.2、Δ=0.05、λ0=0）。

#### 7. 实验数据集
- 训练：DAPO-Math-17K，Qwen2.5-Math-7B + GRPO。
- in-domain 评测：AIME24/25、MATH500、AMC2023、Minerva Math；out-of-domain：LiveCodeBench（代码）、IF-Eval（指令遵循）。指标 Avg@64 / Pass@64。

#### 8. 怎么做的(训练/数据/流程)
veRL 上用 GRPO 训 Qwen2.5-Math-7B（DAPO-Math-17K，主实验）。对比 GRPO、Clip-Lower、Clip-Free、Ada-Ent-Reg(δ=0.2/0.3657)、Adv≤0、Rand-Pos-Clip 及三个 Pos-Adv-Reweight 变体。**另在 Llama-3.1-8B-Instruct 上做泛化验证（同样 DAPO-Math-17K，附录 F）**：熵崩溃/校准/正优势 token 主导等核心发现与 Pos-Adv-Reweight(Entropy-guided) 的有效性在该骨干上同样成立〔已核-论文附录 F〕。
结果：仅在 Adv≤0 上训练虽能抑制熵崩溃但平均 Avg@64 偏低；Stage-based 与 Epoch-wise 在 7 基准中的 5 个（AIME24/25、MATH500、Minerva、IF-Eval）超 GRPO，平均与其他熵正则方法相当；三个变体平均 Avg@64 均超 Clip-Higher，其中 Entropy-guided 在 7 个中的 6 个上超 Clip-Higher 且能精确把熵控在目标值。
**批判性评价**：本文最大价值在系统性实证与 ablation，方法本身（按优势符号重加权损失）是渐进式改良，与 Clip-Cov/KL-Cov、80/20 等"针对高协方差/正优势 token 限更新"思路同源，性能增益有限（多与现有熵正则"相当"，而非显著超越），且训练集单一（仅 DAPO-Math-17K）、骨干覆盖也有限（主验 Qwen2.5-Math-7B，附录补充 Llama-3.1-8B-Instruct 泛化）。"熵不一定是性能可靠代理"的结论（相关性任务依赖）是有用的负面/澄清性结论。


---


### rl_plus — RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization

> **一句话重点 (TL;DR)**：针对"纯 on-policy RLVR 反而收窄基座可解问题集（capability boundary collapse）"现象，RL-PLUS 用 Multiple Importance Sampling 稳定吸收外部 off-policy 数据 + focal-style 探索优势放大低概率正确路径，并去掉 clip，使 Pass@k 突破基座天花板。

**元信息**：arXiv 2508.00222（v5, 2026-04-15；标注 Preprint July 2025）｜ 北京大学 / 阿里通义实验室 / University of Alberta（Yihong Dong、Xue Jiang、Ge Li、Zhi Jin、Yongbin Li 等，一作实习于通义）｜ Preprint ｜ 主题 混合策略 RLVR / 相关性较高（外部+教师轨迹引导、低概率正确路径利用，与 TSRD 的 path-recovery 同源）｜ 代码 https://github.com/YihongDong/RL-PLUS（已克隆，含 `rl_plus/`{deepscaler,verl,scripts,setup.py}、`exp_scripts/`、`eval_scripts/`、`data/`）｜ 框架 VeRL + DeepScaleR

#### 1. 相关工作与进展
RLVR（OpenAI o1、DeepSeek-R1、Kimi 等）通过可验证奖励（数学答案正确、代码单测通过）驱动 LLM 延展 CoT、自发出现反思与探索行为，被视为通向更强 AI 的有希望路径。论文从两条线定位自身：(1) on-policy RLVR（GRPO 及其改进，如 PRIME-Zero 用隐式过程奖励）；(2) 引入外部知识以突破纯 RL 的知识上限的方法（SFT→RL、GRPO w/ SFT Loss，以及并发工作 LUFFY、ReLIFT）。

#### 2. 现有工作存在的问题
- 多篇工作（Havrilla 2024、Shao 2024、Yue 2025a）指出现行 RLVR 不能让模型获得**新**推理能力，只是复用基座已有模式：Pass@1 升、Pass@128 反而低于基座，说明可解问题集被收窄（capability boundary collapse）。
- 根因：解空间巨大 + 奖励稀疏，长链推理中单步出错即清零整条轨迹奖励，模型被迫做"向内利用（inward exploitation）"而非"向外探索（outward exploration）"。
- 混合 SFT-RL 方案各有缺陷：顺序 SFT→RL 易遗忘；"GRPO w/ SFT Loss"简单相加反而掉点。
- 用 IS 吸收外部数据存在两难：on-policy 代理（分母用 πθold）有系统偏差（Lemma A.5）；理论正确的 off-policy 权重 πθ/πω 有支撑失配（A.6）与策略发散时的高方差（A.7），且 πω 通常未知不可直接算。

#### 3. Motivation
以"学而不思则罔，思而不学则殆"作隐喻：当前 RLVR 是"思而不学"（只想不学外部知识），SFT 是"学而不思"（只模仿不内化、遇新题脆弱）。需要一种既能稳定从外部 off-policy 数据"学"、又能显式激励"思"出低概率但正确推理路径的混合策略方法，从而真正扩展能力边界。

#### 4. 主要灵感 / 核心直觉
- **混合策略视角**：把外部样本看作来自 πθold 与外部策略 πω 的**混合**，用 Multiple Importance Sampling 而非单一 IS，使分母中的 πθold 充当"方差护栏"——即使 πω 很差，比值也有界（Theorem 3.1：只要行为池中存在一个≈πθ 的策略，方差就低）。
- **focal loss 直觉**：新知识藏在模型自认低概率的正确 token 上；借 focal loss 思想，对低概率正确 token 加大优势权重，迫使模型关注被忽视区域。
- **去 clip**：clip 会压制高信息量、低概率事件的梯度，正是要吸收的新知识，故移除。

#### 5. 主要解决思路(一段话讲清核心)
在 RLVR 训练中同时混入内部 on-policy rollout（Do，用标准 PG）与静态外部数据（De）。对 De，用 token 级 MIS 比 r^m = 2πθ / (πω + πθold) 校正分布失配，其中未知 πω 用 Bayes 最优估计 ½πθold + ½U（U 为均匀分布，Theorem 3.2）；再乘以 focal-style 探索优势 C = (1−detach(πθ(e_t)))^γ 放大低概率正确 token 的信号。复合目标（Eq.7）= 内部利用项（标准 PG）+ 外部探索项（MIS·探索优势），且全程**不做 clip**。

#### 6. 方法详解(通俗、分步骤)
1. **MIS 比（Eq.4）**：r^m_{i,t} = 2πθ(e_{i,t}) / [πω(e_{i,t}) + πθold(e_{i,t})]。把外部 token 当作混合策略产物，分母的 πθold（被刻意保持接近 πθ）使比值有界，把"坏代理/支撑失配"的爆炸性偏差换成有界的可控失真（Remarks A.8/A.9）。
2. **估计 πω（Theorem 3.2）**：在"具体代理 πθold"与"非信息均匀策略 U=1/V"之间按无差别原则各赋 ½ 先验，最小化 Bayes 风险得贝叶斯模型平均 π̂ω = ½πθold + ½U。
3. **探索优势（Eq.5–6）**：A^c_{i,t} = 组内标准化优势 (R_i − mean)/std · C_{i,t}，其中 C_{i,t} = (1−detach(πθ(e_{i,t})))^γ。正确 token 概率越低权重越大；detach 阻断梯度回传以稳训练；γ 为超参。
4. **复合目标（Eq.7）+ 去 clip**：内部项用标准 PG（稳定+精炼已有能力），外部项用 MIS×探索优势（驱动外部探索），移除 clip 让模型在遇到外部高价值信息时迈更大优化步。

#### 7. 实验数据集
- 主基座 **Qwen2.5-Math-7B**（Table 1 全部方法同一基座）；额外报 Qwen2.5-Math-7B-Instruct、LLaMA-3.1-8B 验证跨模型族（Table 3 类）。
- **6 个数学推理 benchmark**：AIME 24、AIME 25、AMC、MATH-500、Minerva、OlympiadBench（SOTA 比较）。
- **6 个 OOD 任务**：编程 HumanEval / LiveCodeBench / LeetCode + 科学 QA ARC-c / GPQA-diamond / MMLU-Pro（Table 2）。
- 基线：SFT、GRPO、SFT+GRPO、GRPO w/ SFT Loss、LUFFY、ReLIFT。

#### 8. 实验结果与主要发现
- Table 1：RL-PLUS 在 6 个数学 benchmark 全面 SOTA；较 "SFT+GRPO" 平均 **+5.2 分**；优于并发的 LUFFY、ReLIFT。
- 跨模型族：在 Qwen2.5-Math-7B 等多基座一致提升，GRPO 平均相对提升最高 **69.2%**；在 LLaMA-3.1-8B（GRPO 等基线表现差）上亦稳定改进。
- Table 2（OOD）：在编程与科学 QA 上一致超过 GRPO 与 SFT+GRPO；论文称 SFT 在 OOD 上常劣于 RL 类，而 RL-PLUS 同时在 in-domain 与 OOD 占优。
- Pass@k 曲线（Fig.3）：RL-PLUS 的 Pass@k 高于基座，**解决了能力边界坍塌**。
- 消融：去掉探索优势→Pass@k 上界明显下降；去掉 MIS→吸收外部数据不稳。

#### 9. 结果如何支撑其主张
"突破边界坍塌"的主张直接由 Pass@k 曲线（RL-PLUS > Base，而 GRPO < Base）支撑，这是 capability boundary 的标准度量，逻辑闭合。"稳定吸收外部数据"由 MIS 消融 + 理论方差界（Thm 3.1）双重支撑。"探索低概率路径"由 focal-style 权重消融支撑。跨模型族与 OOD 结果支撑泛化性主张。

#### 10. 逻辑自洽性(中性评估)
组件动机—理论—消融三者基本对齐：MIS 的方差护栏与 Bayes 估计有清晰推导（Thm 3.1/3.2 + 附录引理），探索权重有 focal loss 类比。去 clip 与"吸收低概率新知识"动机一致，且其稳定性显式依赖 MIS 分母约束，论证自洽。

#### 11. 残留问题 / 局限
- πω ≈ ½πθold + ½U 的估计较粗糙；Thm 3.1 的"低方差"依赖"行为池中存在 ≈πθ 的策略"这一假设，外部数据质量差时是否仍成立缺乏针对性实证。
- 去 clip 的稳定性完全寄托于 MIS 分母，未给出 MIS 失效时的兜底分析。
- "泛化"证据主要在数学相邻的 OOD 推理任务（编程/科学 QA），未涉及真正异构域（对话、长文本生成等）。
- 外部数据 De 的来源/规模/质量对结果的敏感性〔待核：附录是否给出 De 构造细节〕。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 链接：https://github.com/YihongDong/RL-PLUS（已克隆，含 README.md）。
- 框架：基于 **VeRL + DeepScaleR**；目录 `rl_plus/`（deepscaler、verl、scripts、setup.py）、`exp_scripts/`（训练）、`eval_scripts/`（评测）、`data/`。
- 可得性：训练/评测脚本与数据齐备，可复现性较好（核心改动在 advantage/IS 比与目标函数，落在 verl 训练 loop 内）。


---


### rl_survey_lrm — A Survey of Reinforcement Learning for Large Reasoning Models

> **一句话重点 (TL;DR)**：系统综述 DeepSeek-R1 以来"RL for 大型推理模型(LRM)"的范式，提出"基础组件 / 五对开放争议 / 训练资源 / 下游应用 / 未来方向"的分类体系，明确 RLVR 是区别于 RLHF/DPO 的新 scaling 轴，并把 RL 推向 LRM/ASI 的可扩展性列为核心开放问题。

**元信息**：arXiv 2509.08827（v3, 2025-10-09/10）｜ 清华 / 上海AI Lab / 上海交大 / 北大 / 中科大 / 哈工大 / 华盛顿大学 / 华科 / UCL（40+ 作者；负责人 Kaiyan Zhang、Yuxin Zuo；通讯 Biqing Qi、Ning Ding、Bowen Zhou）｜ 2025-09（v3 2025-10）｜ 主题 T3 RL for LRM 综述 / 为本项目 RL/post-training 的全局定位与文献坐标系提供权威综述 ｜ 代码 https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs（awesome-list，仅论文/资源清单，**无方法实现代码**）｜ 框架 综述仓，无独立训练框架（梳理 OpenRLHF/veRL/AReaL/slime/TRL）

> 注：本条为综述（非方法论文，paper-only），下列 12 节按"综述视角"映射填写（如"实验数据集/结果"对应综述梳理的训练资源与核心论断）。

#### 1. 相关工作与进展
RL 在 AlphaGo/AlphaZero 时代已证窄而明确的奖励可驱动超人表现。LLM 时代 RL 先以 RLHF/DPO 做人类对齐（3H：helpful/honest/harmless）。近期出现新趋势 **RL for LRMs**——不止对齐行为，而是直接激励推理本身。OpenAI o1 与 DeepSeek-R1 两个里程碑证明：用**可验证奖励的 RL（RLVR）**（数学答案正确、代码单测通过）可让模型产生长链推理（规划、反思、自我纠错），且性能随训练与推理时算力平滑提升，开辟预训练之外的新 scaling 轴。DeepSeek-R1 用 GRPO + 规则奖励，证明大规模 RL 甚至能在 base 模型上诱导复杂推理。综述 §2.3 梳理了相关已有综述以定位自身。

#### 2. 现有工作存在的问题
综述把进一步扩展 RL for LRM 的基础性约束（算力、算法设计、训练数据、基础设施）归为五对开放/争议问题（§4）：
- **§4.1 RL's Role: Sharpening or Discovery**——RL 仅锐化 base 已有分布，还是发现新能力。
- **§4.2 RL vs SFT: Generalize or Memorize**——RL 泛化 vs SFT 记忆。
- **§4.3 Model Prior: Weak and Strong**——弱/强 base 先验对 RL 的影响。
- **§4.4 Training Recipes: Tricks or Traps**——各类 trick 有效还是陷阱。
- **§4.5 Reward Type: Process or Outcome**——过程奖励 vs 结果奖励。

#### 3. Motivation
随领域（尤其 DeepSeek-R1 后）爆发式发展，亟需重访其轨迹、重估方法论、探索把 RL 推向 ASI 的可扩展策略。综述以"语言智能体与环境在长期演化中的大规模交互"为核心视角，系统化整理基础组件、开放问题、训练资源与应用，识别未来机会。

#### 4. 主要灵感 / 核心直觉
- 把 RLVR 明确区分于 RLHF/DPO（Figure 2）：前者激励推理能力本身，后者对齐人类偏好，是不同范式。
- 以"组件—争议—资源—应用—未来"五层结构组织一个快速演进、文献海量的领域，便于读者建立坐标系并追踪。

#### 5. 主要解决思路(一段话讲清核心)
全文围绕一张总图（Figure 1）展开：先给 RL 在 LRM 语境下的预备定义（§2.1）、o1 以来前沿模型脉络（§2.2）与相关综述（§2.3）；再分别综述基础组件（§3）、五对开放争议（§4）、训练资源（§5）、下游应用（§6），最后给未来方向（§7）与结论（§8）。配套 GitHub Awesome-list 维护论文/资源清单。

#### 6. 方法详解(综述分类体系 / Taxonomy)
- **§3 基础组件**：
  - §3.1 奖励设计：可验证(Verifiable)、生成式(Generative)、稠密(Dense)、无监督(Unsupervised)、奖励整形(Reward Shaping)。
  - §3.2 策略优化：策略梯度目标、基于 critic（如 PPO）、无 critic（Critic-Free，如 GRPO/RLOO）、off-policy 优化、正则化目标。
  - §3.3 采样策略：动态/结构化采样、采样超参。
- **§4 五对争议**（见 §2）。
- **§5 训练资源**：§5.1 静态语料（Math/Code/STEM/Agent/Mixture）、§5.2 动态环境（Rule/Code/Game/Ensemble）、§5.3 RL 基础设施（OpenRLHF / veRL / AReaL / slime / TRL）。
- **§6 应用**：§6.1 编码、§6.2 智能体(Agentic)、§6.3 多模态、§6.4 多智能体、§6.5 机器人、§6.6 医疗。
- **§7 未来方向**：持续 RL（Continual）、记忆型 RL（Memory-based）、模型型 RL（Model-based）、高效推理、潜空间推理（Latent Space Reasoning）、RL 用于预训练、RL 用于扩散式 LLM、科学发现、架构-算法协同设计。

#### 7. 实验数据集（综述梳理的训练资源，§5）
综述本身不做实验。§5.1/§5.2 梳理 RL 训练资源：静态语料（数学/代码/STEM/Agent/混合）与动态环境（规则、代码、游戏、ensemble 等），并讨论其复用性；§5.3 比较主流基础设施 OpenRLHF / veRL / AReaL / slime / TRL。

#### 8. 实验结果与主要发现（综述的核心论断）
- RLVR 是区别于 RLHF/DPO（人类对齐）的新范式（Figure 2），显著提升复杂任务求解能力。
- 五对争议尚无定论（如 §4.1 sharpening vs discovery、§4.5 process vs outcome），是当前活跃前沿。
- 下一阶段 scaling（含 open-ended RL）仍开放，是迈向 ASI 的关键且具挑战的方向。
- §6 显示 RL for LRM 已外溢到编码、智能体、多模态、多智能体、机器人、医疗等多领域。

#### 9. 结果如何支撑其主张（综述覆盖范围与组织如何支撑论点）
聚焦 o1 / DeepSeek-R1 发布以来的工作，覆盖范围明确；用 Figure 1 总图 + 五层分类把海量文献结构化，支撑"RLVR 为新范式、scaling 为核心开放问题"的论点。Awesome-list 提供可追踪的论文/资源清单，支撑其作为领域坐标系的实用价值。

#### 10. 逻辑自洽性(中性评估)
分类体系（组件/争议/资源/应用/未来）层次清晰、互不重叠；五对争议以"X or Y"对立式提炼便于读者把握分歧。作为综述其论断为对现有文献的归纳而非新实证，自洽性体现在分类的完备性与一致性上，整体组织连贯。

#### 11. 残留问题 / 局限
- 综述无原创实验/方法，结论依赖所引文献质量；领域演进极快，v3（2025-10）后的工作必然遗漏。
- 五对争议虽提炼清晰，但综述多呈现分歧而较少给出定论性裁决，读者仍需自行判断。
- "RL 通向 ASI"为前瞻性论断，缺乏可证伪的具体路径。
- §6 应用覆盖广但每域深度有限；基础设施比较（§5.3）随框架快速迭代易过时。
- 〔注〕本任务模板曾设想 §6/§8 为 taxonomy，本综述实际把分类体系放在 §3–§6、§8 为结论，已按实际结构记录。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 链接：https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs（awesome-list，配套论文/资源清单）。
- 框架：综述本身无独立训练框架；§5.3 梳理并比较主流 RL 基础设施 **OpenRLHF / veRL / AReaL / slime / TRL**。
- 可得性：**无方法实现代码**，仅维护精选清单，便于追踪快速演进的文献；不可"运行复现"。


---


### scaf_grpo — Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning

> **一句话重点 (TL;DR)**：针对 RLVR 的"学习悬崖"(难题持续零奖励→GRPO advantage 坍缩→梯度消失),只在学习停滞时按"知识→规划→解答"三层、增量注入 in-prompt 提示,让**当前策略自己**采样出成功轨迹来替换失败轨迹,从而在保持 on-policy 一致性、不破坏探索自主性的前提下攻克长尾难题。

**元信息**：arXiv 2510.19807 (v2, 2026-02-28) ｜ HKUST / CUHK / HKU(Xichen Zhang*、Sitong Wu*、Yinghao Zhu、Haoru Tan、Shaozuo Yu、Ziyi He、Jiaya Jia 通讯;*共同一作) ｜ ICLR 2026 接收 ｜ 主题 T3(RLVR 算法/引导式探索),与 T1 教师引导相关,Relevance=High(其"教师按需提供分层最小提示、保 on-policy、保探索"与本项目 TSRD 路径选择思路高度契合) ｜ 代码 github.com/JIA-Lab-research/Scaf-GRPO(已克隆 ~7.2MB,Tier A,origin 已核) ｜ 框架 veRL 0.4.1.dev(vLLM rollout)

#### 1. 相关工作与进展
RLVR(DeepSeek-R1 范式)用稀疏 binary 结果奖励即可让模型自主习得推理策略,免去逐步人工标注。后续工作沿两条线推进:(a) 算法稳定性/去偏(Dr.GRPO、DAPO 等);(b) 更稠密/更有信息量的奖励(长度惩罚抑制 overthinking、token 级稠密反馈)。针对"学习悬崖",已有一类 **off-policy 教师引导**工作:LUFFY(整条专家轨迹与多条 rollout 混批)、cosine-decay 前缀长度调度(Huang et al.)、多级不同长度 hint(Zhang et al.)等,主流形式是"golden 解前缀续写"。

#### 2. 现有工作存在的问题
- **学习悬崖(learning cliff)**:面对远超当前能力的题目,所有探索尝试持续失败 → 持续零奖励 → GRPO 中 advantage 坍缩为 0 → 梯度消失,这些难题对策略更新"不可见",形成训练停滞的长尾瓶颈(论文 Figure 2 实证 Qwen2.5-Math-1.5B 上 vanilla GRPO 在零奖励题上停滞)。
- 现有"前缀续写"应对带来两弊:(1) 教师前缀与学生后缀**分布失配**,需 policy shaping / 混合 SFT-RL 等补丁,引入偏置与不稳定;(2) **"on-rails"**把模型逼上预定路径,扼杀对替代/更优策略的探索。

#### 3. Motivation
借鉴教育学**脚手架(Scaffolding, Berk & Winsler 1995)**理论:仅在学习停滞时提供"会随能力提升而撤除"的临时、最小、分层引导,以"路标(signpost)"而非"铁轨(railroad)"方式引导。两条设计目标:(a) **policy consistency**——同一统一策略同时处理"题目+提示",避免前缀法的分布失配;(b) **保留探索灵活性**——提示只指方向不定路径。

#### 4. 主要灵感 / 核心直觉
"最小有效引导"原则:奖励模型**用尽可能抽象的提示**解出问题,促其内化推理技能而非记忆解法。提示分三层(抽象→具体)、增量给出而非一次性给全,既精确诊断学生缺口,又最大化技能迁移。

#### 5. 主要解决思路(一段话讲清核心)
两阶段:先用"豁免期"区分真难题与伪难题,只对**真难题**触发引导;引导时按"知识→规划→解答"三层、由抽象到具体增量注入 in-prompt 提示,直到当前策略 π_θ **自己**采样出正确解 o*_h,用该成功轨迹替换 rollout buffer 中某条失败轨迹。因为成功轨迹仍是当前策略在 hint-augmented prompt 上采样所得、重要性比率也对该 prompt 计算,所以整个过程仍是 **on-policy**(而非教师 off-policy 续写),既给出有意义学习信号又指向模型可达的最高效推理路径。

#### 6. 方法详解(通俗、分步骤)
1. **Phase 1 — 诊断 true-hard 问题(guidance exemption period)**:训练前 **15% 步数**纯 on-policy 探索。监控"零奖励 query 被解决的速率",一旦该速率停滞,仍持续失败的题判为 **true-hard**(区别于因格式/早期技能不熟而"多训即可解"的 pseudo-hard),才成为引导候选。
2. **Phase 2 — 分层 hint 引导**:预定义三层提示 H = {H_knowledge(知识)、H_planning(规划)、H_solution(解答)},抽象→具体。框架做确定性搜索:从最抽象层起、层内增量给提示,一旦模型生成正确解即终止,得到**最小有效引导**。提示为 in-prompt 注入(不破坏 on-policy)。
3. **训练机制**:在 veRL 上保持 GRPO on-policy 性质,仅在停滞时**策略性扩充 rollout buffer**(用 π_θ 自采样的 o*_h 替换一条失败轨迹)。KL penalty=0 以最大化探索;hint-guided 探索仅对 17.4% 样本触发(Qwen2.5-Math-7B)。

#### 7. 实验数据集
- **训练**:派生自 DeepScaleR-Preview-Dataset,按每模型初始能力动态过滤(Too Easy 丢弃 / Too Hard 保留 / Potentially Solvable 50% 子采样);三层 hint 由 prompt DeepSeek-R1 基于 ground-truth 解题步骤生成。
- **评测**(7 基准,pass@1 greedy decoding):AIME24、AIME25、AMC、Minerva、MATH-500、OlympiadBench、GaoKao2023en;OOD 泛化用 **GPQA-Diamond**(Table 6)。
- **模型**(Table 1,5 个):Qwen2.5-Math-7B、Qwen2.5-Math-1.5B、Qwen2.5-7B、Llama-3.2-3B-Instruct(非 Qwen 架构)、DeepSeek-R1-Distill-Qwen-1.5B(长 CoT)。跨架构/规模(1.5B–7B)/专长验证。

#### 8. 实验结果与主要发现
- **主结果**(Qwen2.5-Math-7B):整体相对 vanilla GRPO **+12.6%**、相对 LUFFY **+9.2%**;AIME24 pass@1 较 vanilla GRPO 相对 **+44.3%**;均值 0.509,较 Simple-RL +19.5%、较 Oat-Zero +9.5%(Figure 1 / Table 1)。
- **OOD(GPQA-Diamond, Table 6)**:Qwen2.5-Math-7B 较 vanilla GRPO **+15.5%**(且与 LUFFY 持平);Qwen2.5-7B-Base **+7.5%**(且超 LUFFY)。
- **消融**:去 Solution 层掉 5.7%(三层中最严重),证三层 KPS 互补;增量给提示优于一次性 "Full Hint"(后者掉 6.3%)。
- **效率**:Scaf-GRPO 约 12h 达最佳 checkpoint(50.9% avg.),vanilla GRPO 需 13h 才达更低峰值;计算开销小(仅 17.4% 样本触发引导)。

#### 9. 结果如何支撑其主张
"攻克学习悬崖"由 Figure 2 训练动态(持续解掉零奖励题)与 7 基准一致增益共同支撑;"跨架构/规模/专长普适"由 5 个模型(含 Llama、长 CoT 1.5B)的一致提升支撑;"内化而非记忆"由"去 Solution 层降幅最大 + 增量优于 Full Hint"两个消融支撑;"保 on-policy 优于前缀法"由相对 LUFFY +9.2% 与 GPQA OOD 增益支撑。逻辑链较完整。

#### 10. 逻辑自洽性(中性评估)
方法叙事与实验对齐,核心机制(自采样替换→保 on-policy)在概念上成立。但有几点需中性看待:(1) true-hard/pseudo-hard 的判定靠"零奖励解决速率停滞",阈值与窗口是经验设定,论文未给敏感性分析;(2) "最小有效引导促内化"是合理叙事,但缺直接证据排除"学生在 hint 下记住了该题"——消融只在层级粒度,未做"训练题 vs 持出题"的记忆探针;(3) 主结果多基于 best checkpoint 报告,存在 checkpoint 挑选偏差风险(基线同口径,影响相对可比)。

#### 11. 残留问题 / 局限
- 引导依赖一个更强教师(DeepSeek-R1)生成分层 hint,真"零外部知识"并不成立;hint 质量上限受教师约束。
- 仅在数学域验证;代码/通用推理是否同样有效未测(GPQA 仅作 OOD 评测,非训练域)。
- 15% 豁免期、KL=0、温度等超参的鲁棒性未系统消融;〔待核〕组大小、lr 等完整超参在附录(正文未全列)。
- 报告 best checkpoint 而非固定步数,跨方法早停策略一致性需信任作者实现。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库:https://github.com/JIA-Lab-research/Scaf-GRPO(本地已克隆 ~7.2MB,Tier A;origin 已核为该地址)。
- 框架:**veRL 0.4.1.dev**(仓库含 `verl/` 与 `hint_mix_grpo/{main_ppo.py, trainer/, dataset/, config/}`);rollout 用 vLLM。README 安装步骤列出 SGLang、Megatron-Core,但**明确注明本实现禁用、不需要**(原分析称其为"核心依赖"略夸大,已更正)。conda env `scaf-grpo`(python 3.10)。
- 入口:`train.sh`(conda activate `scaf-grpo` → `bash sh/hint_mix_grpo/bs256_6k_mix.sh`);提示混合逻辑在 `hint_mix_grpo/`,baseline/评测脚本在 `sh/{baseline,generation_eval}`。
- 数据集:HuggingFace `hkuzxc/scaf-grpo-dataset`。


---


### sed_sft — SED-SFT: Selectively Encouraging Diversity in Supervised Fine-Tuning

> **一句话重点 (TL;DR)**：在 SFT 的交叉熵之上，对"探索空间大"的 token 选择性加一个把 ground-truth 概率往 0.5 推的二次惩罚（被 Top-k 累积概率掩码门控），以保留生成多样性、为后续 RL 留探索空间；RL 后相对 CE 基线平均仅 +1.2~+2.06 点，增量有限且证据面窄。

**元信息**：arXiv 2602.07464v1 ｜ 腾讯 WeChat AI（Yijie Chen*、Yijin Liu*、Fandong Meng） ｜ 2026-02-07 预印本（cs.CL） ｜ 主题 SFT 损失正则 / 为 RL 保探索（与 OPD/蒸馏"SFT 不损害后续可塑性"相关，但本文不涉 teacher distillation） ｜ 代码 https://github.com/pppa2019/SED-SFT （真实可用，含自带 trainer + verl 子目录 + Triton 核） ｜ 框架 自带 torch/transformers/DeepSpeed ZeRO-2 SFT trainer + verl GRPO（RL）

#### 1. 相关工作与进展
后训练主流范式为 SFT→RL。围绕"SFT 阶段如何不抑制多样性"已有三条线：(1) RL Policy Integration——DFT（Wu 2025）把 RL 策略更新思想引入 SFT，用置信度重加权抑制过大梯度，ASFT（Zhu 2025a）在 DFT 上修偏移但需引入 reference model、成本高；(2) Diversity Modeling——GEM（Li 2024）在目标中显式加 reverse-KL + 熵以鼓励多样生成；(3) Selective Gradient Updating——CTF（Ruan 2025）等需预先确定重要更新位置，依赖先验或高成本算法。另有大量工作（Wang 2025 的 80/20 高熵少数 token、Cui 2025 熵机制）在 RL 阶段做基于熵的探索控制，不直接适用于 SFT。

#### 2. 现有工作存在的问题
- RL 阶段的熵控制方法不适用于 SFT（SFT 的 mode collapse 由 token 级 CE 拟合诱发）。
- DFT 在 SFT 阶段精度高但压缩探索空间，后续 RL 难再提升；GEM 鼓励整体多样性却忽视 token 间差异，在高精度任务（数学）上反而掉点。
- 盲目鼓励多样性有害：作者 case study（Appendix A heatmap）发现固定连接词、结构 token、特定词汇这类 token 探索空间本就很小、置信度远高于均值，对其鼓励多样性既无必要又损精度。

#### 3. Motivation
多样性鼓励应"选择性"施加于探索空间大的 token，跳过探索空间小的 token，从而在多样性与精度间取得平衡。

#### 4. 主要灵感 / 核心直觉
把 SFT 视为策略更新、把 y* 的选择视为二值事件，则 ground-truth 概率决定更新幅度。用累积 Top-k 概率作为量化"token 探索空间"的可观测代理：对 100 道数学题统计发现 k=1 最具区分度但信息不足（看不到替代路径），k 增大趋于均匀失去区分力，故取 k=2 或 3。

#### 5. 主要解决思路(一段话讲清核心)
在标准 CE 损失上叠加一个受二值掩码门控的"多样性鼓励项"：掩码用 Top-k 累积概率判定 token 是否落在"探索空间大"区域；鼓励项是一个把 ground-truth token 概率往 0.5（最大熵点）推的二次惩罚。其余 SFT/RL 设置全部保持一致，仅替换 SFT 目标。

#### 6. 方法详解(通俗、分步骤)
- **Top-k 掩码 Mt**：P_Top-k(t)=Σ_{j∈Kt} π(y_j|·)（前 k 个最高概率之和），Mt=1[P_Top-k(t)<τ]。阈值 τ 取该批训练样本累积概率集合 P={P_Top-k(t)} 的 (1−r) 分位数（r 为掩码比例）。
- **多样性鼓励函数** L_DE(p)=(p−1/2)²（受 CHORD 启发的二次惩罚，p=π(y*t|·) 为 ground-truth token 概率），p=0.5 处最小、p=1 或 0 处最大，把概率往 0.5 推。
- **总损失** L = Σ_t [ −log π(y*t|·) + λ·Mt·L_DE(π(y*t|·)) ]，论文所有实验 λ=1。
- 超参：k=2 或 3；r>0.5 时稳定优于 CE，最佳 r=0.7（敏感性见 Table 3：r=0.2 时 34.18 反低于 CE 35.65，r=0.7/k=2 最高 37.71）。计算开销相对 CE 几乎可忽略。

#### 7. 实验数据集
- SFT：Micomind 数据集采样 20,000 条；lr=2e-5、DeepSpeed stage-2（沿用 GEM 设置）。
- RL：MATH（Level 1）训练切分（DigitalLearningGmbH/MATH-lighteval），每 prompt 用 Qwen2.5-Math-7B-Instruct 采 8 次、过滤全对/全错（类离线 DAPO），从 5,000 得 2,069 条；verl GRPO，batch 256，其余默认。
- 评测（8 个数学基准，遵循 Qwen2.5-Math 框架）：AIME24、AIME25、AMC23、GSM8K、MATH500、GAOKAO-en、OlympiadBench、College-MATH。AIME24/25/AMC23 报 Avg@8（temp 0.7），其余采样 1 次（temp 1.0）。
- 骨干：Qwen2.5-Math-7B-Instruct、Llama-3.2-3B-Instruct。全程 8×H20。

#### 8. 实验结果与主要发现
- **RL 后**（八基准均值，Table 1）：Qwen2.5-Math-7B CE 56.00 → SED-SFT 57.20（+1.20）；Llama-3.2-3B CE 35.65 → SED-SFT 37.71（+2.06）。SED-SFT w/o mask 在 Qwen 上 56.60、Llama 上 35.10，介于两者间，验证掩码的增量贡献。
- **SFT 阶段自身**：SED-SFT 几乎不优于甚至略低于 CE（Qwen SFT 均值 SED 42.29 vs CE 43.33；Llama SED 22.04 vs CE 22.25），而 DFT 在 SFT 阶段大幅领先（Qwen 54.24、Llama 25.43）但 RL 后反而垫底（Qwen 55.36、Llama 30.95）。
- **多样性**（Self-BLEU↓，Llama/AIME，Table 2）：SED-SFT 35.57 < GEM 38.53 < CE 43.12 < DFT 51.26，SED-SFT 多样性最高。

#### 9. 结果如何支撑其主张
核心主张"SFT 多样性→RL 收益"由两点支撑：SED-SFT 在 SFT 阶段 Self-BLEU 最低（多样性最高）且 SFT 精度不掉，RL 后又取得最高均值；DFT 的反例（SFT 强、RL 弱）反向印证"SFT 阶段压缩探索空间会限制 RL 上限"。掩码消融（w/o mask 介于 CE 与 SED 之间）支撑"选择性"的必要性。逻辑链成立，但绝对增量仅 1~2 点。

#### 10. 逻辑自洽性(中性评估)
方法-动机-实验自洽：动机（选择性鼓励）→ Top-k 掩码机制 → 消融验证掩码有效。L_DE 把概率推向 0.5 而非更高熵点，是工程化简化（二值事件下 0.5=最大熵），与"鼓励多样性"一致。主要张力在于卖点完全押在"SFT 多样性传导到 RL 收益"这一间接链路上，而 SFT 阶段自身几乎无收益。

#### 11. 残留问题 / 局限
- 增量有限：RL 后绝对提升仅 1~2 点；SFT 阶段 SED-SFT 不如 CE。
- 证据面窄：仅两个骨干、RL 仅 MATH-L1 单数据集、未报多 seed 方差，结论稳健性存疑。
- 新意有限：本质是 GEM/CHORD 思路加 Top-k 累积概率掩码的工程化变体，核心新意在"选择性掩码"这一机制。
- 〔待核/新发现〕**代码与论文的实现差异**：仓库 `utils/sed_triton_loss.py` 中 SEDLoss 默认 `entropy_penalty_scale=0.2`（即 λ=0.2，非论文所述 λ=1），且掩码用固定 `cumsum_threshold=0.95` 而非论文的 (1−r) 分位数自适应阈值 τ；分位数/动态掩码等分支（use_low_topk_cumsum_ratio 等）在类中声明但 forward 默认走固定阈值路径。复现论文配置需自行对齐 λ 与阈值策略。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库 https://github.com/pppa2019/SED-SFT （已 clone，约 15MB；最新 commit 6a89ea3 "Update README"）。
- 核心损失：`utils/sed_triton_loss.py`（SEDLoss，Triton 实现 CE + (p−0.5)² 惩罚 + Top-k 累积掩码），另含 `ce_triton_loss.py`、`gem_triton_loss.py` 作对照；trainer 为 `sft_trainer_v2.py` + `train.py`，脚本在 `train_scripts/`（grpo_math-cumsum.sh、train_cumsum_numina.sh、tokenize_data.sh）。
- RL 阶段：仓库内打包的 `verl/` 子目录上跑 GRPO。
- 代码核心算法可定位、可复现，但默认超参与论文不完全一致（见 §11）。


---


### simplerl_zoo — SimpleRL-Zoo: Investigating and Taming Zero RL for Open Base Models in the Wild

> **一句话重点 (TL;DR)**：在 10 个跨家族/尺寸的 base 模型上做透明的 zero RL（仅正确性二值奖励、不用 format reward），系统拆解成败关键因素，并首次在 Qwen 家族外的小模型上观察到 verification 等认知行为涌现；核心是经验研究与配方/工具开源，而非新算法。

**元信息**：arXiv 2503.18892v3（v1 2025-03-24，v3 2025-08-06 cs.LG） ｜ HKUST + TikTok + 美团（Weihao Zeng*、Yuzhen Huang*、Qian Liu*、Junxian He 等） ｜ 预印本 + Notion 博客 ｜ 主题 T3 RLVR / zero RL 经验研究，相关性 High ｜ 代码 https://github.com/hkust-nlp/simpleRL-reason （已克隆约 66MB，当前版即 v1；HF hkust-nlp/simplerl-zoo collection 10 个模型） ｜ 框架 veRL（仓内自带 verl/，GRPO），旧 v0 用 OpenRLHF+PPO（README 注明）

#### 1. 相关工作与进展
DeepSeek-R1 表明从 base 模型直接做规则奖励的纯 RL（"zero RL training"）可自发涌现长 CoT 与自反思（"aha moment"）。但该成功最初在 671B 的 DeepSeek-V3 上演示，社区复现（Zeng 2025a、Yeo 2025、Xie 2025、Hu 2025、Yu/DAPO 2025）主要集中在 Qwen2.5 系列。

#### 2. 现有工作存在的问题
- Qwen2.5 base 因预训练含大量合成数据，本身已具较强指令跟随与 backtracking/verification 行为，**不能代表"in the wild"的多样 base 模型**；
- 现有分析多停留在响应长度、准确率等表层指标，无法判断推理行为是否真正改变，也未澄清推理涌现机制；
- 缺乏对"哪些关键因素决定 zero RL 成败"的系统研究。

#### 3. Motivation
在 10 个不同家族/尺寸 base 模型上做透明 zero RL，回答三问：(1) 不同模型推理能力如何演化；(2) 初始缺乏指令跟随/自验证能力的 base 是否仍现 "aha moment"；(3) 跨多样 base 成功 zero RL 的关键因素是什么。

#### 4. 主要灵感 / 核心直觉
表层指标（长度、准确率）不足以刻画"推理是否真改变"，需引入认知行为级监控：用 GPT-4o 识别 Backtracking / Verification / Subgoal Setting / Enumeration 四类行为的频率（Reasoning Behavior Ratio），并辅以 Clip Ratio（截断比例）、Average Stopped Length（正常停止响应平均长度），把"长度增长"与"认知行为涌现"解耦观察。

#### 5. 主要解决思路(一段话讲清核心)
用最简配方——GRPO + 仅正确性二值奖励（+1/0，**不加 format reward**）+ 全部模型相同超参——在 10 个 base 模型上从零起训，配合两项关键设计（避免刚性格式约束、按模型内在探索能力匹配数据难度），并用认知行为指标透明监控训练动态，归纳成败因素。

#### 6. 方法详解(通俗、分步骤)
- **算法**：GRPO；奖励仅判正确性（二值），不用 format reward（避免强格式约束惩罚探索）。
- **数据难度控制**：按难度分 Easy（GSM8K + MATH lv.1）/ Medium（lv.1–4）/ Hard（lv.3–5），各约 8K 题；难度须匹配模型内在探索能力。
- **统一超参**：所有模型同一组超参，弱指令跟随模型用更简单 prompt；均从 base 直接 RL（无 SFT cold start）。
- **监控指标**：Reasoning Behavior Ratio（GPT-4o 标四类认知行为）、Clip Ratio、Average Stopped Length、pass@k。

#### 7. 实验数据集
- 训练：仅 GSM8K + MATH 训练集（规则奖励），三档难度各约 8K。
- 模型（10 个）：Mistral-7B-v0.1、Mistral-Small-24B、Llama-3.1-8B、DeepSeek-Math-7B、Qwen2.5-{0.5,1.5,7,14,32}B、Qwen2.5-Math-7B。
- 评测：GSM8K、MATH500、Minerva Math、OlympiadBench、AIME24（Pass@1 与 Avg@32）、AMC23；泛化 IFEVAL、MMLU、GPQA-Diamond。

#### 8. 实验结果与主要发现
1. 响应长度增长并不总对应 "aha moment"——多数 Qwen2.5 模型长度涨但认知行为（自反思）频率未升。
2. 首次在 Qwen 家族外的小模型（Llama3-8B、DeepSeek-Math-7B）观察到 verification 等认知行为显著增长。
3. 刚性 format reward（如强制 \boxed{}）会惩罚探索、压低性能上限、诱发 overthinking；许多 base 初期跟不上格式约束，format reward 反而惩罚正确探索（Fig.6 对比）。
4. 训练数据难度须与 base 探索能力匹配，否则 zero RL 失败（如 Mistral-7B 在 Hard 数据上崩溃）。
5. zero RL 提升 pass@k 达 10–30 个绝对点，且 base 与 RL 后模型的 pass@k 差距随 k 增大反而**扩大**（pass@1 与 pass@8 gap 随训练加宽，Fig.3）→ 证明不仅是重排，而是真正增强（与 Shao 2024 观察相反）。
6. 传统短 CoT SFT 作为 cold start 会限制后续 RL 探索，SFT 步数越多对 enumeration/verification 等行为损害越大。
- 总体：仅 8K 样本即对全部模型显著提升（DeepSeek-Math-7B 约三倍增长，长度约 300→1200+ token）；Mistral-Small-24B + SimpleRL-Zoo 平均 27.6→49.6。

#### 9. 结果如何支撑其主张
三问均有对应证据：Q1/Q2 由 Reasoning Behavior Ratio 跨模型曲线 + Llama3/DeepSeek-Math 的认知行为涌现支撑；Q3 由 format reward 消融（Fig.6）、难度匹配实验（Mistral 崩溃）、pass@k 扩大（Fig.3）共同支撑"format、难度、是否 SFT cold start"为关键因素。"真正增强而非重排"的主张由 pass@k gap 随 k 扩大这一反直觉证据较强地支撑。

#### 10. 逻辑自洽性(中性评估)
作为经验研究自洽性较好：用统一超参 + 多样模型隔离"模型族"变量，用认知行为指标把"长度"与"推理质量"解耦。一个需注意点：Reasoning Behavior Ratio 依赖 GPT-4o 作行为分类器，分类一致性/偏差未深入校验；"aha moment 涌现"部分结论建立在该自动标注之上。

#### 11. 残留问题 / 局限
- 无新算法：贡献是配方、模型、分析工具与系统经验，方法学新意有限。
- 仅数学域（GSM8K+MATH）训练，认知行为分类器依赖 GPT-4o，存在标注偏差风险。
- "难度须匹配探索能力"为定性结论，缺乏可操作的难度-能力量化判据。
- 部分结论（如 SFT cold start 损害探索）与具体 SFT 数据/步数强相关，外推到长 CoT SFT 需谨慎。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库 https://github.com/hkust-nlp/simpleRL-reason （已克隆约 66MB，最新 commit cf1c785；当前版即论文 v1 配方）。
- 框架：veRL（仓内自带 `verl/` 目录，pyproject name="verl"），RL 算法 GRPO。核心脚本 `train_grpo_math_tune_ray.sh`（Ray 启动）、`eval_math_nodes.sh`、`install.sh`、`launch_gradio.sh`。README 注明旧版（v0）用 OpenRLHF + PPO。
- 模型：HF hkust-nlp/simplerl-zoo collection（10 个）。代码与配方可得、可复现。


---


### skywork_or1 — Skywork Open Reasoner 1 Technical Report

> **一句话重点 (TL;DR)**：面向"已蒸馏的长 CoT 模型"的高效可扩展 RL 配方（基于改造版 GRPO，命名 MAGIC），系统消融各组件并深入研究过早熵坍缩，论证"缓解过早熵坍缩对提升测试性能至关重要"；32B/7B 在 AIME+LiveCodeBench 平均分别 +15.0/+13.9，全开源。

**元信息**：arXiv 2505.22312v2（2025-05-29 cs.LG） ｜ Skywork AI（昆仑万维 Kunlun Inc），Jujie He*、Jiacai Liu* 等，通讯 Jujie He ｜ 技术报告 + Notion 博客 ｜ 主题 T3 RLVR / 长 CoT 模型 RL，相关性 High ｜ 代码 https://github.com/SkyworkAI/Skywork-OR1 （已克隆约 11MB；数据 HF Skywork/Skywork-OR1-RL-Data；权重 OR1-{Math-7B,7B,32B} 及 Preview） ｜ 框架 veRL 定制 fork（仓内自带 verl/ + or1_scripts/ + or1_data/，含 Math/Code 验证器）

#### 1. 相关工作与进展
DeepSeek-R1 证明 online RL + 简单规则奖励即可大幅提升 base 模型推理。R1-Distill 系列生成的 CoT 在 AIME24 上平均超 10K token，远长于 Qwen2.5/Llama3.1。已有复现（Logic-RL、ORZ、DAPO、VAPO）多聚焦对 base 模型 RL；DeepScaleR、Light-R1、DeepCoder 等开始探索对长 CoT 模型 RL，但未系统拆解组件贡献。

#### 2. 现有工作存在的问题
- 如何高效、可扩展地用 RL 提升**已做过 SFT 的长 CoT 模型**仍不清楚；
- DeepScaleR、Light-R1、DeepCoder 等虽有初步进展，但未系统拆解各算法组件在 RL 训练中的独立贡献；
- 训练中普遍出现 premature entropy collapse（过早熵坍缩、过度 exploitation），其成因与缓解缺乏系统研究。

#### 3. Motivation
提出针对长 CoT 模型的高效可扩展 RL 配方，系统消融各组件；深入研究熵坍缩现象，证明缓解过早熵坍缩对提升测试性能至关重要；全开源代码、数据、权重。

#### 4. 主要灵感 / 核心直觉
把"维持探索能力"操作化为"用 target-entropy 给策略熵托一个动态下界"：当熵被自适应机制下界托住时，模型保持探索能力与高学习可塑性，测试性能稳步提升；过早熵坍缩则普遍对应更差性能（§4.2）。

#### 5. 主要解决思路(一段话讲清核心)
在改造版 GRPO（命名 **MAGIC = Multi-stage Adaptive entropy scheduling for GRPO In Convergence**）上，从 数据收集 / 训练策略 / 损失函数 三方面组合多项设计，并通过逐组件消融验证；其中以"自适应熵控制 + 多阶段长度调度 + on-policy + 高温采样 + 去 KL"为核心。

#### 6. 方法详解(通俗、分步骤)
- **数据收集**：严格预处理 + 更准验证器；离线+在线过滤（去掉 base 正确率为 0/1 的题、每阶段开始丢弃上阶段已全对的题）；Rejection Sampling（batch 仅保留含非零 advantage 的组）；model-aware 难度估计。
- **训练策略**：(1) Multi-Stage Training——逐阶段增大 context 长度 T（借鉴 DeepScaleR），先短后长省算力又保 scaling；(2) **不采用 advantage mask**——实验（§3.2.3）证明对截断响应赋负 advantage 反而提升 token 效率且不损后期 scaling，故最终不用任何 advantage mask；(3) High-Temperature Sampling（τ=1）保探索；(4) On-Policy Training（7B/32B 严格 on-policy，显著减缓熵坍缩；Math-7B 用两步梯度更新，非严格 on-policy）。
- **损失函数**：去掉 1/|y_ij| 长度归一项 → token 级 policy loss（全 batch token 平均，缓解长度偏置）；(1) Adaptive Entropy Control——引入 target-entropy 超参，按当前熵与目标熵之差动态调 entropy loss 系数，使熵被 target 下界托住；(2) No KL Loss（KL 项在多阶段后期反而妨碍提升）。

#### 7. 实验数据集
- 训练：自建数据 mixture（严格难度过滤 + 质量控制，含从 NuminaMath-1.5 过滤的 hard 题），对照 DeepScaleR mixture（AIME/AMC/Omni-MATH/STILL）；含数学与代码题，配 Math Verifiers + Code Sandboxes。
- 评测：AIME24、AIME25、LiveCodeBench（2024-08~2025-02）；主指标用 Avg@K 而非 Pass@1。
- 基座：DeepSeek-R1-Distill 系列（7B/32B 等已蒸馏长 CoT 模型）。

#### 8. 实验结果与主要发现
- Skywork-OR1-32B：AIME24/AIME25/LiveCodeBench 平均 57.8%→72.8%（+15.0），AIME 上超 DeepSeek-R1 与 Qwen3-32B、LiveCodeBench 持平；7B 43.6%→57.5%（+13.9）。
- 多阶段效率：Stage I T=8K → Stage II T=16K（step 540 切换）→ Stage III；同样最终精度但省约 100 训练小时/1000 步，token 效率显著更高（平均长度约 12.5K→5.4K）。
- 熵动态关键发现：熵坍缩越快测试性能越差；增大 batch/group 对熵动态影响小，而提高采样温度影响显著；off-policy 更新（增大 mini-batch / 数据复用 N_SGD）加速熵坍缩并劣化性能；通过自适应 entropy 系数或 clip-higher 可减缓熵坍缩。

#### 9. 结果如何支撑其主张
"配方高效可扩展"由 32B/7B 的 +15/+13.9 与多阶段省算力（100 小时/1000 步、12.5K→5.4K token）支撑；"组件各有贡献"由逐组件消融（§3.2.1–3.2.6）支撑；"缓解过早熵坍缩至关重要"由 §4 一系列对照（熵坍缩速度 vs 性能、N_SGD/off-policy 影响、自适应熵/clip-higher 缓解效果）支撑，证据较充分。

#### 10. 逻辑自洽性(中性评估)
作为技术报告自洽性好：MAGIC 各组件均有独立消融，熵坍缩论证有多角度对照。需注意：诸多结论（如"高温影响大、batch/group 影响小"）建立在自建数据 mixture + DeepSeek-R1-Distill 基座这一特定设置上；advantage mask 的"反而提升 token 效率"结论与部分截断惩罚直觉相反，依赖其多阶段长度调度场景，外推到非多阶段设置需谨慎。

#### 11. 残留问题 / 局限
- 强工程报告属性：贡献是配方组合 + 系统消融 + 开源，单个组件多借鉴已有工作（DeepScaleR 多阶段、clip-higher 等）。
- target-entropy 等关键超参的设定缺乏跨基座/任务的可迁移性论证。
- 评测面集中在 AIME + LiveCodeBench，泛化基准较窄。
- Math-7B 采用非严格 on-policy（两步梯度更新），与 7B/32B 的严格 on-policy 不一致，削弱了"on-policy 减缓熵坍缩"结论在 Math-7B 上的纯净性。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库 https://github.com/SkyworkAI/Skywork-OR1 （已克隆约 11MB，最新 commit 64e96af）。
- 框架：veRL 定制 fork（仓内自带 `verl/` + `or1_scripts/` + `or1_data/`，含 Math/Code 验证器与 code sandbox）；RL 算法为改造版 GRPO（MAGIC）。
- 数据：HF Skywork/Skywork-OR1-RL-Data；权重：Skywork-OR1-{Math-7B,7B,32B}（及 Preview 版）。代码、数据、权重全开源，可复现性高。


---


### spurious_rewards — Spurious Rewards: Rethinking Training Signals in RLVR

> **一句话重点 (TL;DR)**：在 Qwen2.5-Math 上，即使用随机/格式/错误标签等"虚假奖励"做 RLVR 也能大幅涨分（随机奖励 MATH-500 +21.4%，接近真值的 +29.1%），原因是 GRPO 的裁剪偏置放大了 base model 已有的高先验行为（如 code reasoning），而非注入新能力；该效应在 Llama3/OLMo2 等模型族上几乎消失。

**元信息**：arXiv 2506.10947（v2, 2026-02-25）｜ University of Washington / AI2 / UC Berkeley（Rulin Shao\*、Shuyue Stella Li\*、Rui Xin\*、Scott Geng\* 等，… Nathan Lambert、Sewon Min、Pang Wei Koh、Luke Zettlemoyer）｜ Preprint（RLVR 训练信号机理分析）｜ 主题 T3（RLVR / Zero-RL 机理）/ 相关性 High（直击"RL 是否注入新能力 vs. 仅放大先验"，与本项目教师脚手架蒸馏的核心追问对齐）｜ 代码 https://github.com/ruixin31/Rethink_RLVR （本地已克隆 ~259MB，Tier A）｜ 框架 **OpenRLHF**（经 PRIME-RL 的 TTRL 改编；论文正文亦自述"沿用流行 RL 框架 OpenRLHF 的默认评测设置"）

#### 1. 相关工作与进展
RLVR（可验证奖励强化学习）已成为提升 LLM 数学推理的主流后训练范式，开源社区高度依赖 Qwen2.5-Math 系列作为事实标准 base model，大量"zero-RL"结论均基于该单一模型族。相关探索包括：TTRL（用多数投票估计伪标签做无监督 RL）、各类弱监督/自监督奖励设计。本文把这些"奖励信号是否必须准确"的隐含假设系统化地推到极端来检验。

#### 2. 现有工作存在的问题
多数 RLVR 方法仅在 Qwen 上验证、缺乏跨模型族确认；社区默认把"Qwen 上的增益"等同于"真实推理能力提升"，忽视预训练先验对 RL 动力学的塑造作用，从而可能高估了 RL 算法本身的贡献。

#### 3. Motivation
用一组从弱到虚假递进的奖励作诊断探针，检验 RLVR 究竟需要多少有效监督信号，并解释为何 Qwen 上即使无信息奖励也能涨分。

#### 4. 主要灵感 / 核心直觉
若涨分主要来自"放大 base model 已有高先验行为"，那么奖励是否携带正确信号就不再是必要条件——只要更新方向系统性偏向高先验 token 即可。GRPO 的 clip 机制恰好提供了这种与奖励无关的偏置来源。

#### 5. 主要解决思路(一段话讲清核心)
设计奖励梯度链（Ground Truth → Majority Vote → Format → Random → Incorrect Label），只替换奖励函数、固定其余超参，在多模型族上跑 GRPO，观测增益差异；再用裁剪偏置的理论推导 + code reasoning 行为分析解释"为何无信息奖励在 Qwen 上仍涨分"，并用 prompt-based / RL-based 干预验证机理。

#### 6. 方法详解(通俗、分步骤)
- **奖励梯度链**：Ground Truth → Majority Vote（64 rollout 多数投票伪标签）→ Format（含非空 `\boxed{}` 即给 1）→ Random（概率 γ 随机给 1，主实验 γ=0.5）→ Incorrect Label（只奖励多数投票得到的错误答案）。
- **主结果（Qwen2.5-Math-7B, MATH-500 绝对涨幅）**：GT +29.1、Majority +27.1、Format +13.8/+15.5、Incorrect Label +24.1、Random +21.4，仅个别（如某弱奖励变体）为 −6.4。即随机奖励涨幅已接近真值。
- **跨族对比**：上述效应在 Llama3.1/3.2、OLMo2、Qwen2.5（非 Math）上几乎无效甚至掉点——说明依赖 base 先验。
- **机理（GRPO 裁剪偏置 clipping bias）**：clip 项使更新系统性放大 base model 中高先验概率的 token/行为，即使奖励无信息也会强化"高先验行为"；附录给出完整推导（随机奖励诱导的梯度正比于策略相对行为策略的概率偏离）。
- **关键案例行为 code reasoning**：Qwen2.5-Math-7B 含 Python 代码（无执行器，纯文本推理）的回答正确率 60.9% vs 无代码 28.0%；虚假奖励下 code 频率 15 步内升至 ~90%、随机奖励最高 95.6%，紧跟准确率上升。模型按 No-Code / Bad-Code / 有效 code 分类，解释跨族差异。
- **验证**：用 prompt-based 与 RL-based 干预显式提升 code reasoning 与词汇重复，均显著抬升 Qwen 表现，反向印证机理。

#### 7. 实验数据集
- 训练：DeepScaleR；及其多数投票伪标注 / 错误标注派生集（如 `DeepScaleR_mv_labeled_llama3.2_3b_instruct_incorrect`）。
- 评测：MATH-500（pass@1）、AMC（avg@8）、AIME 2024/2025。
- 模型：Qwen2.5-Math-7B/1.5B、Qwen2.5-7B/1.5B、Llama3.1-8B(-Instruct)、Llama3.2-3B(-Instruct)、OLMo2-7B 及 OLMo2-7B-SFT。

#### 8. 实验结果与主要发现
- 随机奖励即可使 Qwen2.5-Math-7B MATH-500 +21.4%（vs GT +29.1%）；错误标签奖励 +24.1%。
- 同样的虚假奖励在 Llama3/OLMo2 等族上失效，强力支持"先验依赖"假说。
- code reasoning 频率与准确率高度耦合（升至 90%+，随机奖励 95.6%），是 Qwen 涨分的主要可观测中介行为。

#### 9. 结果如何支撑其主张
跨族对照（Qwen 涨 / 他族不涨）+ 机理推导（裁剪偏置）+ 干预验证（显式抬升 code reasoning 即涨分）三条证据互相印证，较有说服力地支撑"RLVR 在开源算力规模下主要是激发/放大已有潜在能力，而非教会新能力"。

#### 10. 逻辑自洽性(中性评估)
论证链条自洽：诊断探针—跨族对照—理论解释—干预复现闭环。裁剪偏置推导与 code reasoning 行为分析互为佐证。需注意：结论的适用范围被作者自限于"当前开源后训练算力规模"，并非否定 RLVR 在更大规模下的能力增益。

#### 11. 残留问题 / 局限
- 结论强绑定 Qwen2.5-Math 这一特殊（预训练即含大量含代码数学解）base model，外推到通用模型/更大算力需谨慎。
- "虚假奖励涨分"仅是分析工具，不构成可用训练方案（作者明确不推荐）。
- 裁剪偏置推导基于若干简化假设；code reasoning 作为中介行为是相关性证据，未必穷尽全部机理。
- README 文末示例把 R1-style 奖励后缀写作 `_r1_only`，与实际代码/脚本一致使用的 `_r1_style` 不符（README 笔误，不影响复现，以代码为准）。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 规范仓库：https://github.com/ruixin31/Rethink_RLVR （本地已克隆 ~259MB，Tier A，代码完整可跑）。模型集合：HuggingFace `stellalisy/spurious-rewards`；复现页见 Notion / W&B。
- **框架：OpenRLHF**（README 自述代码库改编自 PRIME-RL 的 TTRL，TTRL 本身为 OpenRLHF 衍生；`code/ttrl/` 多处引用 OpenRLHF issue，训练脚本用 OpenRLHF 风格参数 `micro_train_batch_size`/`n_samples_per_prompt`/`--advantage_estimator group_norm`/`--normalize_reward`）。算法主体 **GRPO**（advantage 用 `group_norm`，见 `ttrl/trainer/experience_maker.py`）。环境：python 3.10 + flash_attn 2.7.0.post2，`pip install -e .`。〔原稿曾误记 veRL，现据 README、`ttrl/` 代码与论文正文确认为 OpenRLHF。〕
- 关键超参（`scripts/rlvr_deepscaler_grpo_qwen_*.sh`）：训练 300 步、binary 0-1 奖励、`train_batch_size=128`、`n_samples_per_prompt=16`、`micro_train_batch_size=4`、`gamma=1.0`；仅改 `TASK`/`REWARD`。奖励函数实现见 `ttrl/verifier/qwen/qwen_eval.py` 与 `ttrl/verifier/auto_verify.py`（`random_reward_fn(rate=0.5)`、`inverse_qwen_reward_fn`、`box_only_format_reward_fn` 等）。无 chat template 模型加 `_r1_style` 后缀（脚本 `--verify_task "${REWARD}_r1_style"` 已确认）。


---


### srft — SRFT: A Single-Stage Method with Supervised and Reinforcement Fine-Tuning for Reasoning

> **一句话重点 (TL;DR)**：用熵感知（entropy-aware）的自适应权重，在单阶段内同时对同一模型施加 SFT（demonstration）与 RL（自探索 rollout），按当前策略熵动态平衡模仿与探索；建立在 LUFFY 之上，平均 59.1%、较 zero-RL 在 5 个数学基准 +9.0%、3 个 OOD 基准 +10.9%。

**元信息**：arXiv 2506.19767（v1, 2025-06-24，OpenReview n6E0r6kQWQ）｜ 中科院自动化所 / 国科大 / 美团 / 上海交大（Yuqian Fu, Tinghong Chen, Jiajun Chai 等，通讯 Dongbin Zhao 团队）｜ ICLR 2026 ｜ 主题 T3/T4（High，GFT 类 / SFT-RL 单阶段统一）｜ 代码 https://github.com/fyqqyf/SRFT （Tier A，已克隆 ~2.4MB，可跑）｜ 框架 veRL + vLLM（底座沿用 LUFFY 的 mix_src 结构与 deepscaler 奖励）

#### 1. 相关工作与进展
SFT 与 RL 的结合是后训练的核心问题。传统做法两阶段串行（SFT 做 instruction-following → RL 做 alignment/reasoning），二者被视为独立阶段。近期 GFT/统一后训练方向（如 LUFFY 的 off-policy RL、混合策略训练）尝试把 demonstration 与 rollout 信号融合。本文直接建立在 LUFFY 之上（README 致谢，代码沿用 `mix_src`），创新点是熵感知的 SFT/RL 自适应加权。

#### 2. 现有工作存在的问题
- 两阶段串行：SFT 易记忆模式而非习得真实推理、易过拟合；RL 样本效率低、探索困难、易 mode collapse。
- 集成不足导致误差传播、限制 RL 提升；过度依赖 demonstration 又过拟合、约束探索。如何在 SFT 的知识蒸馏与 RL 的策略优化之间定权重，是核心痛点。

#### 3. Motivation
从熵视角对 SFT/RL 做机理分析，据此在两个范式间做"按需平衡"的权重调度，使单阶段统一训练既不过拟合 demonstration 又保持探索。

#### 4. 主要灵感 / 核心直觉
作者三条关键发现：(1) SFT 对策略分布做"粗粒度全局"改变，RL 做"细粒度选择性"修改；(2) 单阶段集成优于串行；(3) 熵动态是训练有效性的关键指标，可据此在两范式间平衡加权——熵高时减弱模仿、保持探索，熵低时加强模仿。

#### 5. 主要解决思路(一段话讲清核心)
单阶段同时对同一模型施加 SFT（demonstration）与 RL（on-policy rollout），用 entropy-aware weighting 把两者合成为单一 loss 反向：以当前策略熵自适应调节 demonstration 模仿强度与 RL 探索强度。

#### 6. 方法详解(通俗、分步骤)
**SRFT（Supervised Reinforcement Fine-Tuning）**：
- 论文（Sec.4）两个熵感知权重解析式：SFT 权重 `w_SFT = 0.5·stop_grad(exp(−H(π_θ)))`（论文叙述：熵高时减弱模仿、熵低时加强）；RL 正样本目标权重 `w_RL = 0.1·stop_grad(exp(H(π_θ)))`（熵高时维持探索）。
- 代码层面（`mix_src/mix_core_alg.py` 的 `compute_token_on_off_policy_loss`）：
  - 正 advantage 项乘 `pos_entropy_exp_coeff = 0.1 * (entropy).exp().detach()`（L134，与论文 w_RL 一致 ✓）。
  - `entropy_exp_coeff = entropy.exp().detach()`（L162），`sft_loss = -entropy_exp_coeff * log_prob`（L163）；再在 `mix_actor.py` 中以 `policy_loss = policy_loss - sft_loss·sft_loss_coef`、训练脚本 `sft_loss_coef=-0.5`（train.sh L63）合成，即等效 `+0.5·exp(H)·(−log_prob)`。
  - ⚠️ **paper-code 符号差异（保留 flagged）**：论文写 SFT 权重 `exp(−H)`，代码实际用 `exp(+H)`，**符号相反**——以代码为准时 SFT 权重随熵升高而增大（与论文叙述方向相反）；0.5 系数则一致。RL 侧 `exp(+H)` 则 paper-code 一致。该差异已在论文与代码间双向核对确认。
- `mix_actor.py` 同时含 `policy_loss = pg_loss - entropy_loss·entropy_coeff`（`entropy_coeff=0.001`，train.sh L59）。
- **adaptive temperature**（可学习 `log_alpha` 对齐 target entropy，类 SAC 温度自适应）是代码中的**可选项、默认关闭**：config `use_adaptive_temperature: False`、`adaptive_temperature_target_entropy: 1.0`（`mix_ppo_trainer.yaml` L78/L81），主训练脚本未启用——非核心方法必备组件。

#### 7. 实验数据集
- 训练：OpenR1-Math-46k-8192（openr1.parquet；OpenR1-Math-220k 的 46k 子集，源自 NuminaMath 1.5，带高质量推理 demonstration）+ on-policy rollout。
- 评测：5 个数学推理基准 + 3 个 OOD 基准（AIME24/AMC 用 avg@32；推理 temperature=0.6、max_gen=8192）。基座 Qwen2.5-Math-7B(-16k-think)。训练 64×A100（脚本 n_gpus_per_node=8、nnodes=4）。

#### 8. 实验结果与主要发现
平均准确率 59.1%，较 zero-RL 方法在 5 个数学基准 +9.0%、3 个 OOD 基准 +10.9%（三项与 arXiv v1 摘要一致，已核）。熵感知调度使单阶段统一训练在数学与 OOD 上均稳定优于两阶段及纯 SFT/RL 基线。

#### 9. 结果如何支撑其主张
跨数学 + OOD 双类基准的一致增益，配合熵动态分析（SFT 全局/RL 选择性、单阶段优于串行），支撑"用熵调度统一 SFT-RL"的主张。但增益相对 LUFFY 等强基线的具体边际、以及熵权重的消融贡献需看正文逐表（见局限）。

#### 10. 逻辑自洽性(中性评估)
方法叙事自洽：熵机理分析 → 熵感知权重 → 单阶段统一。最大隐患是上述 paper-code 符号差异——论文用 `exp(−H)` 论证"熵高减弱模仿"，而开源代码实为 `exp(+H)`（熵高加强模仿），二者机制方向相反。若以代码为准，则论文对 w_SFT 的直觉解释不成立；这是使用方必须注意的自洽性裂缝（可能是论文笔误或代码 bug，作者未澄清）。

#### 11. 残留问题 / 局限
- **paper-code 符号矛盾未解释**（见 §6/§10），影响对"熵如何调度模仿"的理解。
- adaptive temperature 默认关闭，论文若将其计入贡献叙事需谨慎。
- 仅 Qwen2.5-Math-7B 单基座、单训练集（OpenR1-46k），跨模型族/跨数据泛化未验证。
- 〔待核〕anonymous 项目页与 OpenReview 版本的逐表数字、以及熵权重各项的消融未逐一交叉核对。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- https://github.com/fyqqyf/SRFT （Tier A，已克隆 ~2.4MB，代码完整）。核心实现：`srft/verl/verl/mix_src/`（`mix_core_alg.py` 的 `compute_token_on_off_policy_loss`、`mix_actor.py` 的 loss 组装、`mix_trainer.py`）。模型权重：HuggingFace `Yuqian-Fu/SRFT`。
- 框架 **veRL + vLLM**（rollout/评测），底座沿用 LUFFY 的 mix_src 与 deepscaler 奖励。
- 关键超参（exp_scripts/train.sh）：train_batch_size=128、ppo_mini_batch=64、max_prompt=1024 / max_response=8192、actor lr=1e-6、temperature=1.0、val_temperature=0.6、kl_loss_coef=0、kl_loss_type=low_var_kl、entropy_coeff=0.001、sft_loss_coef=-0.5、tp=2、use_dynamic_bsz；adaptive temperature 默认关闭。


---


### superrl — SuperRL: Reinforcement Learning with Supervision to Boost Language Model Reasoning

> **一句话重点 (TL;DR)**：在 RLVR 流程内做实例级自适应回退——某 prompt 的所有 rollout 都拿到零奖励（无 PG 梯度信号）时，就在该 prompt 上回退到高质量离线示范（tagged_answer）做 SFT，否则走标准 GRPO/PPO；二选一、不做损失融合，专治稀疏奖励下 rollout 全失败的"无梯度"困境。

**元信息**：arXiv 2506.01096（v2, 2025-08-08）｜ 北京大学 / UIUC / 微软（Yihao Liu, Shuocheng Li, Lang Cao, Yuhang Xie 共同一作，通讯 Mengyu Zhou）｜ Preprint, under review, 2025 ｜ 主题 T3（统一 SFT-RL，GFT 类）/ 相关性 中高（按 reward 信号在 RL 与 SFT 间切换，与本项目"信号不足时回退教师监督"思路一致）｜ 代码 https://github.com/microsoft/SuperRL （仅核心组件，需集成进官方 verl v0.5.0+）｜ 框架 veRL（volcengine，v0.5.0）+ FSDP + 梯度检查点

#### 1. 相关工作与进展
LLM 推理任务常有大量高质量离线数据（专家标注 / 蒸馏轨迹）。在线 RL（PPO/GRPO）是 on-policy，只能从当前策略采样轨迹学习，难利用分布外离线数据；传统两阶段 SFT→RL（RLHF 范式）则在 RL 阶段易灾难性遗忘 SFT 知识。SuperRL 属"在 RL 流程内细粒度统一 SFT/RL"的方向。

#### 2. 现有工作存在的问题
- 纯 SFT 只记忆正例，缺乏从错误/负例学习的机制。
- 纯在线 RL 在稀疏奖励下因 rollout 全失败而拿不到梯度信号。
- 两阶段 SFT→RL 在 RL 阶段灾难性遗忘 SFT 知识，样本/算力效率低；过拟合离线轨迹与狭窄 RL 目标又损泛化。

#### 3. Motivation
将 SFT 与 RL 在更紧粒度（实例级 interleave/unify）统一，让模型在在线信号不足时回退到高质量离线监督，从而在稠密与稀疏奖励两种 regime 下都稳定有效学习。

#### 4. 主要灵感 / 核心直觉
稀疏奖励下"全 rollout 失败"恰是 SFT 最该介入的时刻——此时 RL 无任何梯度，而离线示范能提供确定的学习信号。把"是否有有效 PG 信号"作为实例级的自动开关，无需手工阶段切换。

#### 5. 主要解决思路(一段话讲清核心)
对每个 prompt 采样多条 rollout 算 reward：只要存在非零 reward（有 PG 信号）就走标准 policy gradient；若该 prompt 所有 rollout reward 全为 0（无梯度）则回退到在该 prompt 上对离线示范做交叉熵 SFT。二选一，不融合。

#### 6. 方法详解(通俗、分步骤)
**核心方法（主推，`SuperRLActor`）：实例级自适应回退（adaptive switching / fallback）。**
- `SuperRLActor.update_policy`：用 `advantages*response_mask` 与 `token_level_rewards*response_mask` 的**绝对均值是否 > eps** 双重判定信号有效性（`pg_signal_eps=1e-8`、`reward_eps=1e-8`）；
- 有有效信号 → `_ppo_update`（复用父类 GRPO/PPO）；两者皆 ~0（无梯度）→ `_sft_update`：对 `extra_info["tagged_answer"]` 做 `F.cross_entropy(..., reduction="mean")`。

论文给出两个变体（代码也提供）：
- **Hybrid-Adv-Gated**（`HybridAdvGatedActor`）：有效 PG 信号→PPO，否则→SFT，同样不融合（比 SuperRLActor 少一层 reward 检查，仅判 advantage）。
- **Hybrid-Log-Sigma**（`HybridLogSigmaActor`）：用学习到的不确定性权重软融合：`L = exp(−2·σ_pg)·L_ppo + exp(−2·σ_sft)·L_sft + (σ_pg + σ_sft)`，`log_sigma_pg/sft` 为可学习参数加入优化器（init 0 / 1）；并带 σ 随步衰减（`sigma_decay_rate=0.99`、`min_log_sigma=−2.0`）。论文指出变体虽有改进但需额外调参/开销，主推简洁的实例级回退。

#### 7. 实验数据集
覆盖稠密奖励任务（GSM8K、MetaMathQA）与稀疏奖励任务（OpenR1-Math-220k、PRM12K/MixChain-Z-PRM12K），另含 LIMO、AIME24/AIME25、HiTab（层级表格 QA）。Backbone 跨三族：Qwen2.5（0.5B/1.5B/3B/7B）、LLaMA 3.x（3.2 1B/3B、3.1 8B）、DeepSeek-R1-Distilled。SFT/SFT+RL 基线用相同数据/学习率/上下文长度；actor 与 critic 默认从同一预训练 checkpoint 初始化。

#### 8. 实验结果与主要发现
SuperRL 在 GSM8K/Metamath/PRM12K/LIMO/OpenR1/AIME 上全面优于 RL、SFT、SFT+RL 基线（如 GSM8K 78.86 vs RL 72.29；已核实表中数字）。在稀疏奖励任务上的相对优势尤为关键，印证"无梯度时回退 SFT"的设计。

#### 9. 结果如何支撑其主张
跨稠密/稀疏两类任务、三模型族的一致优于基线，支撑"实例级回退在两种 regime 下都稳健"的核心主张。但增益幅度（如 GSM8K +6.6pp）属稳健改进而非数量级。

#### 10. 逻辑自洽性(中性评估)
方法逻辑清晰自洽：判信号→二选一更新。代码与论文描述一致（双 eps 判定、cross_entropy 回退、Log-Sigma 软融合公式）。需注意：阈值 eps=1e-8 极小，意味着"几乎任何非零 advantage"都判为有效信号、回退仅在严格全零时触发——回退频率高度依赖奖励稀疏度，论文未给出回退触发率的统计。

#### 11. 残留问题 / 局限
- 仓库仅核心组件（actor/dataset/reward/预处理），**非完整训练框架**，需拷入并集成进官方 verl v0.5.0+ 对应目录（改 `fsdp_workers.py`、`main_ppo.py`、`ray_trainer.py`）才能运行——复现门槛较高。
- 回退依赖每个 prompt 都备有高质量 `tagged_answer`（离线示范），真实稀疏场景未必都有。
- 主方法是硬切换，对"部分有信号"的中间情形无细粒度调节；软融合变体则需额外调参。
- 回退触发率、SFT 与 RL 更新步占比等关键运行时统计未充分报告。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- https://github.com/microsoft/SuperRL 。仓库提供 `actor/`（`SuperRLActor.py`、`HybridAdvGatedActor.py`、`HybridLogSigmaActor.py`）、`dataset/`（`HybridDataset`）、`reward/`（`superrl.py` 统一数学奖励）、`data_preprocess/`；需集成进官方 verl。
- 框架：veRL（v0.5.0）；三 actor 均继承 `verl.workers.actor.dp_actor.DataParallelPPOActor`，通过 `actor_type`（`superrl`/`hybrid_adv_gated`/`hybrid_log_sigma`/`default`）在 `fsdp_workers.py` 选用；adv_estimator 用 grpo。FSDP + 梯度检查点。
- 流程：`HybridDataset` 加载 prompt + `tagged_answer`（预处理脚本将 GSM8K/MetaMath/OpenR1/PRM12K/LIMO/HiTab 转 parquet）→ verl GRPO 流程采样 rollout → `reward/superrl.py` 打分 → 自定义 actor 的 `update_policy` 按信号有效性实例级切换（主方法）或软融合（Log-Sigma 变体）。集成方式见 README。


---


### trapo — TRAPO: Trust-Region Adaptive Policy Optimization

> **一句话重点 (TL;DR)**：在每个训练实例内细粒度交织 SFT 与 RL——只对专家轨迹**前缀**做 SFT、其后由目标策略自行 rollout 补全做 RL；用 Trust-Region SFT（把 SFT 权重 1/p_θ 改为 1/max(p_θ,α)）把 forward-KL 的 mode-covering 转成 reverse-KL 式 mode-seeking、稳住 RL 起点，再用 micro-group 采样按累计回报自适应分配前缀长度。

**元信息**：arXiv 2512.17636v1（2025-12-19，cs.LG）｜ 清华大学 CoAI 组 + Ant Group（Mingyu Su、Jian Guan、Yuxian Gu、Minlie Huang、Hongning Wang；通讯黄民烈、王宏宁）｜ README/OpenReview 标注 ICLR 2026 ｜ 主题 SFT+RL 实例级统一后训练，与 TSRD 高度相关（专家前缀脚手架 + 自探索补全 = path-selection/recovery 的一种实现；与 mtp_opd 的 MTP foresight 无直接关系，但前缀-脚手架是可借鉴的对照基线）｜ 代码 https://github.com/Su-my/TRAPO （已 clone ~6MB，基于 LUFFY/verl）｜ 框架 verl 的 GRPO（去 KL penalty）+ 自定义 TrSFT

#### 1. 相关工作与进展
- **RL for LLM reasoning**：o1/R1/Kimi-1.5 等里程碑；后续从经验分析（Yue 2025a：RL 精炼既有解轨迹而非扩展能力）、数据中心（R3 反向课程、ADARFT/Logic-R1 动态难度）、优化方法（PPO/GRPO/DAPO/Dr.GRPO/VAPO）三向推进。
- **SFT 与 RL 结合**：直接加权两 loss（SRFT 按 token 熵动态调权；AMFT 用 meta-gradient 学权重；HPT 按 rollout 表现二值选 SFT/RL）；或在 RL 流水线内插 SFT（ReLIFT 收集 RL 中差样本入 buffer 后做 SFT；LUFFY 把 1 条专家轨迹当 offline 数据混进 7 条 online、用重要性比校准分布漂移）。**最相近 Prefix-RFT**（Huang 2025）采样前缀作引导、用熵选专家 token 做 SFT。TRAPO 自称首个**理论**研究 SFT+RL 目标结合挑战并给解。

#### 2. 现有工作存在的问题
两阶段 SFT→RL 范式根本不一致：(1) SFT 把模型锁进刻板模仿、抑制 RL 所需探索；(2) SFT 易致灾难性遗忘，使 RL 难用预训练知识。更低 SFT loss 并不意味更好 RL 起点，过度 SFT 反把模型推出适合 RL 的区域且无即时信号。论文实证：**朴素地把 SFT loss 与 RL loss 直接相加导致灾难性崩溃**（较纯 RL 低 18 分以上，Table 2 验证）。根因：标准 SFT(最小 forward KL) 的 mode-covering 特性给专家无支撑的"空洞区"分配概率，引发重复/退化解码——在 SFT/RL 逐实例交织时立即毒化探索。

#### 3. Motivation
把 SFT 与 RL 在**每个训练实例内细粒度交织**：只对专家轨迹前缀做 SFT，其后目标策略自行 rollout 补全做 RL，既吸收专家蒸馏收益、又不损探索与预训练知识。需解决两挑战：(1) 如何有效内化前缀知识（学习目标）；(2) 如何为每 prompt 选最优前缀长度（引导选择）。

#### 4. 主要灵感 / 核心直觉
GMM pilot 实验揭示 SFT 的 distribution-blending：标准 SFT 梯度里 token 权重 1/p_θ(yn|·) 在专家 token 落在远离当前策略模式时会爆炸，先把策略推进"空洞区"再慢慢修正。在交织设定下任何分到空洞区的概率质量都会立即产出退化 rollout。直觉：建立一个"信赖域"，区域内信任标准 SFT 梯度、激进模仿；区域外用常数权重 1/α 抑制梯度，只追专家主模式——即把 mode-covering 转成 mode-seeking。另一直觉（Fig.2 pilot）：越长的专家前缀稳步提升准确率并激发 backtracking / backward chaining 等高级推理行为，故"按需给最小前缀"是合理脚手架。

#### 5. 主要解决思路(一段话讲清核心)
对每 prompt：先无引导自探索 rollout→若回报不足则按递增阈值注入越来越长的专家前缀→目标策略补全→对补全部分用标准 GRPO、对专家前缀部分用 TrSFT loss，全轨迹联合优化。TrSFT 把 SFT 梯度权重从 1/p_θ 改为 1/max(p_θ,α)（α 信赖域边界）；micro-group 采样按前面微组的平均回报与阈值决定本微组的前缀长度比。

#### 6. 方法详解(通俗、分步骤)
- **Trust-Region SFT (TrSFT)**：标准 SFT 梯度 ∇L_SFT 含权重 1/p_θ(yn|·)；TrSFT 改为 1/max(p_θ(yn|·), α)（α∈[0,1]）。p_θ≥α 时用标准 SFT 激进模仿；p_θ<α 时用常数 1/α 抑制梯度。**Prop.1**（已核对论文 §A.2 KKT 推导）：其最优解 p\*_T(c)=p_E(c)/λ（若 p_E(c)>αλ）否则 0，其中 λ=Σ_{c∈S(λ)}p_E(c)——即**剪掉专家低概率区、对主模式重标定，把目标从 forward-KL 的 mode-covering 转向 reverse-KL 的 mode-seeking**，给 RL 稳定起点。
- **Micro-group Sampling（自适应前缀选择）**：每 prompt 顺序建 N 个微组，各组由 (前缀长度比 L_i、回报阈值 t_i、采样预算 n_i) 决定。先做无引导自探索（L_1=0、t_1=−1 保证恒触发）；若前面微组平均回报 < t_i，则给长度比 L_i 的专家前缀再采 n_i 个补全；0=L_1<L_2<…<L_N=1（L_N=1 可给完整专家路径）。"仅在需要时给最小引导"。
- **联合更新**：补全用标准 GRPO 目标、专家前缀用 TrSFT loss，全轨迹统一优化（完整流程 Appendix B Algorithm 1）。

#### 7. 实验数据集
- 训练：OpenR1-Math-46k-8192（DeepSeek-R1 生成、已验证的数学推理轨迹），并额外为每题配一条来自 OpenR1-Math-200k 的轨迹增多样性。
- 基座：Qwen2.5-Math-7B（主）；另在 Qwen2.5-7B-Instruct 验证通用性。pilot 用 Qwen2.5-3B-Instruct + DeepSeek-R1 前缀（MATH-500）。
- 评测：**5 个数学** benchmark = AIME2024、AMC、MATH-500、Minerva、OlympiadBench（AIME/AMC 样本少故报 avg@32，其余 pass@1）；**2 个通用** = ARC-c、MMLU-Pro（pass@1）。

#### 8. 实验结果与主要发现
- 主结果（Table 1，Qwen2.5-Math-7B 基座）：5 数学 benchmark 均值 **TRAPO 56.6**，较 standalone SFT(50.3) **+6.3**、纯 GRPO(50.4) **+6.2**、强基线 SFT-then-RL(54.3) **+2.3**，也超 ReLIFT(53.4)/LUFFY(55.5)。通用域均值 68.3 居首。注意单 benchmark 上 TRAPO 不一定最高（如 AIME2024 TRAPO 28.3 < Oat-Zero 33.4、SFT-then-RL 33.5），优势在综合均值。
- 消融（Table 2）：micro-group 单独已超 GRPO（52.7 vs 50.4）；+ 标准 SFT loss 崩溃（32.3）；+ LUFFY loss 仅小增（53.6）；**+ TrSFT 达 56.6**——证 TrSFT 是稳定结合的关键。
- 训练动态（Fig.4）：TRAPO 全程更高 reward、早期快速增长生成长度（快速内化专家长推理）、长期稳在较高 policy entropy（保留探索）。
- Test-time scaling（Fig.6 pass@k on AIME2024）：纯 GRPO 在大 k 下被 base 反超（RL 只筛选既有解空间），TRAPO 与 SFT 类一样随 k 强 scaling（扩展了底层解空间）。
- 通用模型（Fig.5，Qwen2.5-7B-Instruct）：TRAPO 5 数学均值 45.20 > GRPO 40.60 > Base 39.72 > SFT 32.98。

#### 9. 结果如何支撑其主张
- "朴素相加会崩"：Table 2 中 micro-group+标准 SFT loss 掉到 32.3（较纯 RL 低 18+ 分）直接验证。
- "TrSFT 是关键"：同一 micro-group 骨架下，换 TrSFT(56.6) vs LUFFY loss(53.6) vs 标准 SFT(32.3) 的对比，把增益归因到 TrSFT。
- "扩展解空间而非筛选"：pass@k（Fig.6）TRAPO 不被 base 反超，支撑 test-time scaling 主张。
- "保留探索/不灾难遗忘"：Fig.4 较高稳态 entropy + 通用域不退化（68.3）佐证。

#### 10. 逻辑自洽性(中性评估)
理论-方法链条自洽：诊断（mode-covering 致空洞区）→TrSFT（信赖域裁剪→Prop.1 等价 mode-seeking）→micro-group（按需最小引导）→联合优化。Prop.1 的 KKT 推导完整、forward→reverse KL 的转变有理论支撑。一处需留意的张力：正文 GRPO"without KL penalty"，但仓库默认 `use_kl_loss: True, kl_loss_coef: 0.001`，需脚本覆盖才与论文一致。另：单 benchmark 上 TRAPO 常非最优（综合均值才赢），"strong new paradigm" 的措辞需结合此细节看。

#### 11. 残留问题 / 局限
- **代码核对（已读 trapo_src 核心）：仓库存在专用 `luffy/verl/verl/trapo_src/` 目录，但其 SFT-前缀 loss 由 `mix_core_alg.py::compute_sft_pure_loss` 实现，即 `sft_losses = -log_prob`（标准 NLL/forward-KL SFT），再以 `sft_loss_coef` 加权与 GRPO 相加（`mix_actor.py` L118-148 `use_sft_multitask_loss` 分支）；另有 LUFFY 式 off-policy 重要性比 `off_ratio = exp(log_prob)/(target_probs)`（`use_off_policy_loss` 分支）。未在已读文件中找到 TrSFT 的 `1/max(p_θ,α)` 信赖域裁剪的独立实现；前缀窗口提供 random/linear/fix 三种（`mix_vllm_rollout.py`），但按累计回报递增前缀的 micro-group 阈值 t_i 逻辑未见清晰落地。结论：仓库偏向 LUFFY 基座 + 标准 SFT 加权，TRAPO 的**签名 TrSFT 与完整 micro-group 阈值机制在所见已提交代码中缺失/不完整**（可能在未读分支或脚本参数中）。〔TrSFT/micro-group-t_i 的确切代码落点待核〕**
- 仅在数学推理（+少量通用 QA）验证；训练数据全来自 DeepSeek-R1 蒸馏轨迹，多样性受限。
- α、各微组 (L_i,t_i,n_i) 为手调超参（默认 α=0.1，组大小 8 划 4 微组 {4,2,1,1}，L=(0,0.2,0.5,1.0)，t=(−1,0.5,0.7,0.9)）。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库 https://github.com/Su-my/TRAPO （已 clone，README 确认 ICLR 2026）。结构：`luffy/`（含 verl + `luffy_src/` 与 `trapo_src/` 两套）、`data/`、`exp_scripts/{train_on_policy.sh, train_trapo.sh}`、`eval_scripts/`、`eval.sh`、`figures/`。
- 框架：基于 LUFFY 改造的 verl GRPO（正文"without KL penalty"；但 `trapo_src/config/mix_ppo_trainer.yaml` 默认 `use_kl_loss:True, kl_loss_coef:0.001, low_var_kl`，需脚本覆盖）；batch 128、恒定 lr 5e-6。
- 代码可得性：框架级骨架（前缀 rollout、SFT+GRPO 联合、off-policy 比）齐全且可运行，但 TRAPO 相对 LUFFY 的两项核心新增（TrSFT 信赖域裁剪、micro-group 回报阈值调度）在已读代码中未见完整对应——方法实现完整性存疑，复现需谨慎。


---


### trl_v1 — TRL v1.0: 统一后训练栈（SFT, RM, DPO, GRPO, GKD, MiniLLM 等）

> **一句话重点 (TL;DR)**：TRL v1.0 是 HuggingFace 的统一 LLM 后训练库，用"稳定/实验分级 + 低抽象"哲学把 SFT/RM/偏好优化/RLVR/蒸馏 75+ 方法收进一个库；其 on-policy distillation trainer（GKD/GOLD/MiniLLM/SDFT/SDPO，位于 `trl.experimental.*`）是本调研多篇 OPD 论文（OPSD、AVSD、distillm 等）的底层基础设施。

**元信息**：无 arXiv PDF（基于 release blog + 仓库核对）｜ Hugging Face（blog 署 Quentin Gallouédec、Steven Liu、Pedro Cuenca、Sergio Paniego 等 47+ 贡献者）｜ TRL v1.0 release 2026-03-31（blog published date）｜ 主题 T1,T3 统一后训练栈（含 on-policy distillation trainer），相关性 Med（OPD 生态基础设施基准）｜ 代码 https://github.com/huggingface/trl （已 clone ~11MB，本地 VERSION=**1.6.0.dev0**，已超 v1.0 里程碑）｜ 框架 TRL 本身即框架（基于 Transformers/Accelerate/PEFT，可选 vLLM/DeepSpeed）

#### 1. 相关工作与进展
TRL 处于 LLM 后训练方法爆发的生态位：SFT、偏好优化（DPO/KTO/ORPO/CPO/IPO）、RLVR（PPO/GRPO/RLOO/GSPO）、on-policy 蒸馏（GKD/GOLD/MiniLLM）、self-distillation（SDFT/SDPO）等。本调研中 OPSD、AVSD、distillm/distillm2、aligndistil 等多篇均直接或间接构建于 TRL 之上，TRL 是这些工作的底座框架与对照实现来源。

#### 2. 现有工作存在的问题
方法繁多导致两难：抽象膨胀与破坏性变更风险 vs 生产系统需要稳定 API、研究需要快速迭代。单库内若用大量共享基类/泛化层级，会让研究者难以读懂与改造单个方法。

#### 3. Motivation
提供"统一后训练栈"：在单一库内覆盖 SFT、RM、偏好优化、RL、蒸馏全流程，单 GPU/标准栈即可运行，深度集成 HF Hub，并逐步让训练"对 agent 可读"（结构化诊断信号）。TRL 月下载约 300 万次，社区需要可维护、稳定且 agent 友好的栈。

#### 4. 主要灵感 / 核心直觉
设计哲学："显式有限抽象、偏好重复而非泛化层级、独立实现而非共享基类"（more explicit, more adaptable）——让每个方法尽量自包含、易读易改。配合**稳定/实验分级**：稳定 trainer（`from trl import SFTTrainer`）遵循语义化版本（semver）；实验性方法置于 `trl.experimental.*`（`from trl.experimental.orpo import ORPOTrainer`），允许快速迭代 API 而不破坏生产。

#### 5. 主要解决思路(一段话讲清核心)
用 v1.0 的"稳定核心 + 实验隔离"双层结构解决稳定性与迭代速度的矛盾：稳定核心（SFT/DPO/Reward Modeling/RLOO/GRPO）遵循 semver 给生产用；实验方法（含全部蒸馏 trainer）隔离在 `trl.experimental/` 下快速演进，成熟后再"转正"。每方法一子包、独立实现，便于读改。

#### 6. 方法详解(通俗、分步骤)
- **稳定核心**：SFTTrainer、DPO、Reward Modeling、RLOO、GRPO（顶层导入）。
- **实验性方法**（已核对本地 `trl/experimental/` 目录，确含）：`gkd`（GKD，Generalized Knowledge Distillation，on-policy distillation）、`gold`、`minillm`、`sdft`、`sdpo`、`distillation`、`ssd`、`async_grpo`、`gspo_token`、`gfpo`、`papo`、`dppo`、`ppo`、`prm`、`bco`、`cpo`、`kto`、`orpo`、`nash_md`、`online_dpo`、`xpo`、`tpo`、`openenv`、`openreward`、`grpo_with_replay_buffer`、`bema_for_ref_model` 等——即 shortlist 标题中的 GKD/MiniLLM 确实存在（experimental 子模块）。
- **on-policy distillation 实现路径**：经 GKD/GOLD/MiniLLM trainer（student 自生成轨迹 + teacher token-level 监督）。
- **其它特性**：VLM 支持（SFT/DPO/GRPO）、GRPO 的 `environment_factory` 接口（工具使用 + verification-based reward，面向 agent）。路线图：异步 GRPO、KTO 与蒸馏 trainer 转正、增强 MoE/专家并行、训练对 agent 可读。

#### 7. 实验数据集
N/A——TRL 是库/框架而非单篇实验论文，release blog 未指定特定 benchmark/数据集，重点在方法实现而非评测。

#### 8. 实验结果与主要发现
N/A（非实验论文）。可观测事实：本地 clone VERSION=1.6.0.dev0（开发快照，已越过 v1.0 tag）；`trl/experimental/` 目录确含上述蒸馏与 RL 方法子包，印证"75+ 方法 + 稳定/实验分级"的结构性主张。

#### 9. 结果如何支撑其主张
"统一栈"主张由仓库结构直接支撑：`trl/trainer/`（稳定 trainer）+ `trl/experimental/<method>/`（实验方法各一子包）的双层布局与 blog 描述一致；蒸馏 trainer（gkd/gold/minillm/sdft/sdpo/distillation/ssd）齐备，支撑"覆盖 on-policy distillation 全家族"的主张。

#### 10. 逻辑自洽性(中性评估)
作为基础设施，"低抽象 + 稳定/实验分级"哲学与其目标（生产稳定 + 研究迭代）自洽，且仓库实际目录布局与文档一致。无实验主张需检验。唯一需谨慎的是版本号：blog 称 v1.0，本地是 1.6.0.dev0 开发快照，精确特性清单应以官方 blog/CHANGELOG 为准。

#### 11. 残留问题 / 局限
- 蒸馏 trainer 全在 `experimental`，API 不受 semver 保护、可能变动。
- 〔待核〕blog 给出的 release 日期 2026-03-31 与"75+ 方法"数为 WebFetch 小模型摘要，未逐字核对原 blog；本地 clone 版本为 1.6.0.dev0（非恰好 v1.0 tag），v1.0 精确特性清单以官方 blog/CHANGELOG 为准。
- 作为框架本身不产出新方法/新发现，对 mtp_opd 的价值是"可复用的 OPD 基线实现与训练栈"。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库 https://github.com/huggingface/trl （已 clone 到 resource/repos/trl_v1，~11MB，<500MB 故保留；含 `trl/`、`trl/experimental/`、`examples/`、`docs/`、`tests/`、VERSION=1.6.0.dev0）。
- 框架：TRL 本身即框架，基于 HuggingFace Transformers / Accelerate / PEFT，可选 vLLM（GRPO rollout）、DeepSpeed（多卡）。
- 用法：`pip install trl`，稳定方法从 `trl` 顶层导入、实验方法从 `trl.experimental.<method>` 导入；配 `accelerate`/DeepSpeed 多卡；GRPO 类可选 vLLM rollout 与 `environment_factory` 做工具/验证奖励。on-policy distillation 经 GKD/GOLD/MiniLLM trainer 实现。代码可得性满分（活跃维护的大型开源库）。


---

## 5. T4 — 思维链-Token 级 / MTP（7 篇）

本部分逐篇内联与项目 MTP 前瞻探针、token 异质性监督最相关的工作：多 token 预测（llm_future_mtp）、token / 段落级数据选择与 loss 重加权（rho1/sstoken/segment_attrib/vcore/beyond_loglik）、测试时前瞻干预（srgen）。每篇为完整 12 节，标题已降级。

### beyond_loglik — Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum

> **一句话重点 (TL;DR)**：SFT 默认用 NLL（−log p），但它在"从零训练分类"才最优；后训练时基座已有先验。本文把 NLL 推广成参数族 f_α(p)=(1−p^α)/α，并提出一个统一刻画——**模型能力连续谱**：基座先验强（如数学）时，**下调低概率 token 的 prior-leaning 目标**（如 −p）持续胜过 NLL；基座先验弱（如 figfont 谜题）时 NLL 主导；中间区两者难分。

**元信息**：arXiv:2510.00526v3（2026-05-22）｜ UIUC（共一 Gaotang Li、Ruizhong Qiu、Xiusi Chen；Heng Ji、Hanghang Tong）｜ **ICML 2026 Spotlight**（PMLR 306, 2026）｜ 主题：系统研究 SFT 训练目标（不再默认 NLL），**与 OPD/SFT 损失设计、token 加权高度相关**｜ 代码 https://github.com/GaotangLi/Beyond-Log-Likelihood ｜ 框架 VeRL（`main_verl`）。

---

#### 1. 相关工作与进展
- SFT 是 LLM 后训练标准做法，但**泛化常受限**，普遍归因于"模仿学习范式本身"。
- 从 RL 视角改 SFT 的一批工作：把 SFT/DPO 看作隐式 reward learning（小 lr + 散度目标）、把 importance sampling 引入 SFT、PPO-style clip 约束 drift、**均匀重加权梯度系数（Wu et al. 2026，等价于本文 −p 目标）**。
- 其他 SFT 损失：MSE、focal loss、Huber loss、entropic distribution matching 等——本文框架可统一解释为 prior-leaning。

#### 2. 现有工作存在的问题
- 把 SFT 泛化差归咎于"模仿范式"，**忽略了默认目标 NLL 本身**。NLL 在从零训练小规模分类上经典最优，但后训练范式不同：基座已编码任务先验、监督序列长且可能含噪——**要求预训练模型逐 token 复刻冗长 CoT 会损害泛化**。
- 上述 RL 启发的改进各自在某些域有效，但**缺乏"何时用哪种目标"的统一刻画**。

#### 3. Motivation
把 NLL 推广为参数族 **f_α(p) = (1−p^α)/α**（α→0 退化为 NLL；α=1 即 −p，对应最大化期望平均预测准确率）。实证发现 α=1、α=10 在数学上比 NLL 提升高达 **+15.75 / +14.50**（Fig.1）。由此系统性追问：何种场景适合 NLL、何种适合其他目标——**不主张单一万能损失**。

#### 4. 主要灵感 / 核心直觉
一个目标对"correct logit"的梯度权重 **W_f(p) = −f'(p)·p·(1−p)** 决定它强调哪类 token：
- **凸目标**（−log p）：W_f 峰值在 [0,0.5] → 强调**低概率 token**（prior-averse，先验排斥）。
- **凹目标**（−p、−p^10）：W_f 峰值在 [0.5,1] → 强调**高概率 token**（prior-leaning，先验倚靠）。
凸凹相当于"对模型先验尊重程度"的代理；f_α 族在 prior-averse↔prior-leaning 间平滑过渡。

#### 5. 主要解决思路（一段话讲清核心）
把 SFT 目标统一写成 L_f = E[f(p_θ(y|x))]（f 可微非增）；用 W_f 的凸凹分析把各种损失归到 prior-leaning / prior-averse 两端；再提出**模型能力连续谱**：损失的好坏取决于基座先验强度——先验强用 prior-leaning，先验弱用 prior-averse（NLL），中间无单一最优。

#### 6. 方法详解（通俗、分步骤）
- **统一框架**（§3）：Lemma 3.1 给出 correct-logit 梯度 = W_f(p)；Prop 3.2 证明凹目标的 W_f 峰值落 [0.5,1]、凸目标落 [0,0.5]。f_α 族 W_f(p)=p^α(1−p)：α→0 得 (1−p)（重低概率），α≥1 时低概率信号迅速衰减。
- **能力连续谱三段**：
  - **Model-Strong (MS)**（基座先验强，如数学）：prior-leaning（−p、阈值化 −log p·1{p≥0.2}）**持续优于** NLL。
  - **Model-Weak (MW)**（无相关预训练，如 figfont 谜题）：**NLL 主导**——逼模型从所有 token（尤其低概率/错误处）广泛学习。
  - **Model-Intermediate (MI)**（如医疗推理）：两者**难分伯仲**。
- **能力位置可量化**：用"训练集平均预测概率"——数学 0.76–0.81（Qwen2.5-Math-7B 0.81、LLaMA-3.1-8B 0.76）、医疗 ~0.50、figfont ~0.01。
- **理论**：梯度流下给出充分条件——MS 端 −p 损失下降更大、MW 端 NLL 更大。
- **消融工具**：hard-thresholding 变体 L_HT(I) 只在概率区间 I 内更新，隔离特定概率段 token 的贡献。

#### 7. 实验数据集
- **8 个 backbone**：LLaMA-3.1-3B/8B、LLaMA-3.2-3B、DeepSeekMath-7B、Qwen2.5-Math-1.5B/7B、Qwen2.5-1.5B~32B、Qwen2.5-Coder-7B。
- **27 个 benchmark / 7 个域**。锚点域：MS 用 **NuminaMath**；MI 用 **m23k**（医疗）；MW 用 **Reasoning Gym figfont**。另含通用指令微调（AlpacaEval2 胜率）、编码、低资源多语。

#### 8. 实验结果与主要发现
- **MS（数学，Table 1）**：−p 与阈值化 −log p 在所有模型/数据集上一致超 −log p。例：Qwen2.5-Math-1.5B 平均 17.00（NLL）→ 32.75（−p），+15.75。
- **MI（医疗，Table 2）**：−p 与 −log p 平均分几乎相同（差异在统计波动内）。
- **MW（figfont，Table 3）**：−log p 一致大幅超 −p（−p 的 Exact Match 常为 0）。
- **通用指令微调（Fig.4）**：固定数据/协议、只变 backbone 规模（3B→14B），偏好从 NLL 平滑转向 −p——同一设定内复现连续谱。
- 编码、低资源多语分别复现 MS 端（偏 −p）、MW 端（偏 NLL）。

#### 9. 结果如何支撑其主张
主张"无单一最优损失，取决于基座能力"。支撑很强：三锚点域分别落到谱两端与中间且与理论一致；规模扫描在单一设定内复现转变；"训练集平均概率"作为能力代理与定性预期吻合（0.81/0.50/0.01）。多 backbone × 多 benchmark 的实证规模是其最大说服力来源。

#### 10. 逻辑自洽性（中性评估）
- 高度自洽：W_f 凸凹分析 → 谱位置预测 → 实证三段，理论与实验互相印证，且把既有散点（−p 等价均匀重加权等）纳入同一谱系。
- 张力点：能力位置（训练集似然）需**事后估计**，并非先验可操作；MI 区"无单一最优"意味着实践指导有限。

#### 11. 残留问题 / 局限
- **缺乏先验可操作的目标选择准则**：连续谱位置要训练后用训练集似然量化，落地时仍需试。
- **MI 区无定论**：在最常见的"中等先验"任务上反而给不出明确建议（作者称应转向数据/监督质量）。
- 提出的**不是新损失**，而是"何时用何损失"的认识论框架——价值在解释而非新算法。
- 与本项目 OPD 的关系：其"prior-leaning 下调低概率 token"思想与 token 加权/重要性采样工作相通，可指导 OPD/SFT 损失设计，但需注意 OPD 是 student-on-policy + teacher 分布监督，与本文的固定 ground-truth SFT 设定不同。

#### 12. 开源代码与框架（链接+框架+代码可得性）
- 代码：https://github.com/GaotangLi/Beyond-Log-Likelihood 。
- 框架：**VeRL**（`main_verl`）。目录含 `scripts/{training,evaluation,ablation,one_click}`、`data`、`evaluations`，评测覆盖 27 benchmark。
- 代码可得性：固定数据/评测协议、仅改 SFT 目标（−log p / −p / 阈值化 −log p·1{p≥0.2}）即可复现；hard-thresholding 变体用于消融。


---


### llm_future_mtp — Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential

> **一句话重点 (TL;DR)**：实证发现自回归 LLM 给 prompt 附占位 token 后，正确的未来 token 已落在 top-200 logits 内；用 mask token + gated LoRA + sampler 把这种隐知识显式化为并行多 token 生成，代码/数学近 5×、对话/知识近 2.5× 加速且无质量损失。注意是纯**推理加速(speculative)**工作，非推理质量/蒸馏。

**元信息**：arXiv 2507.11851v1 ｜ Apple(Mohammad Samragh、Arnav Kundu、David Harrison 共一，Mehrdad Farajtabar 等) ｜ 2025-07-16，ICLR 2026 Workshop on Latent & Implicit Thinking ｜ 主题 多 token 预测(MTP)+推理加速(与本项目 MTP 主线高度相关) ｜ 代码 https://github.com/apple/ml-mtp **返回 404**(官方代码未公开释出，paper-only/代码受限) ｜ 框架 gated LoRA + 两层 MLP sampler，基模 Tulu3-8B，微调用 Tulu3 数据集

#### 1. 相关工作与进展
自回归 LM 推理严格串行(每步一 token)，在生成后段(方向/语义已较确定时)尤其浪费。speculative decoding(Leviathan 2022)用 draft+verifier 加速但本质仍依赖自回归。已有用额外 token 做 speculative/MTP 的工作(Gerontopoulos 2025、Chen 2024、Liu & Zhu 2024、Xiao 2024)，但本文区别在于"训练模型把 mask token 填成未来 token"。

#### 2. 现有工作存在的问题
现有 MTP/并行生成要么需重建全新建模与训练流水线(diffusion LM)，要么用额外 head 但牺牲生成质量。如何"对现有自回归训练/推理设置做最小改动"实现高效多 token 生成、且不掉质量，是开放问题。

#### 3. Motivation
关键观察(Fig.1 左)：给预训练模型 prompt 后附冗余(占位)token、检查输出 logits，**正确的未来 token 序列已出现在 top-200 logits 内**——说明模型已隐式"知道未来"。于是可引导模型把这种隐知识结构化。

#### 4. 主要灵感 / 核心直觉
模型已隐式编码未来 token，只是没被"读出"。加 mask token 训练直接预测可把正确 token 推到 **top-10**(Fig.1 中);再加 sampler 头进一步精化(Fig.1 右)。因此无需重建训练范式，轻量 SFT 即可激活隐含的多 token 预测能力。

#### 5. 主要解决思路(一段话讲清核心)
在序列尾附 k 个唯一可学习 mask token，从同一前缀联合预测 k 个未来 token;用 gated LoRA 把 NTP/MTP 分成两条功能路径(NTP 路径行为与原模型完全不变、无质量退化，只让 MTP 路径学多 token);加轻量 sampler 头顺序精化 MTP token，配合 consistency 等辅助损失;推理时用 speculative 策略让 token 在未来"二次方扩展"同时保真。微调只更新 LoRA + sampler，冻结原 decoder。

#### 6. 方法详解(通俗、分步骤)
- **Masked-input 表述**：序列尾附 k 个唯一 mask token m_1..m_k(嵌入为可学习随机向量)，从同一前缀联合预测 k 个未来 token。NTP=标准下一 token 预测，MTP=对 mask 位的预测。
- **Gated LoRA**：对 decoder 层加 gated LoRA，用二值 mask(已知每个位置是否为 mask token)区分 NTP/MTP 两条路径——保证 NTP token 行为与原模型**完全不变**(无质量退化)，只让 MTP 路径学多 token。微调只更新 LoRA + sampler，冻结原权重。
- **Sampler head**：轻量两层 MLP，顺序生成 MTP token，每步条件于当前 latent 与上一已采样 token，产出连贯序列。
- **辅助损失**：含 latent consistency loss 等，提升联合生成 token 的连贯与准确。
- **Speculative generation**：让 token 在未来"二次方扩展"同时保真的推测式生成;每步验证 k 个 speculative token、保留已验证部分并继续。
- **训练效率技巧**：对长 n 序列，改输入/position id/label/attention bias，在一次前向里并行模拟多个"前 i token + k mask"子 prompt;NTP token 只 attend 之前 NTP token(保 NTP 输出不变)，MTP token attend 同 block 内 NTP+MTP 但不跨更早 MTP block。

#### 7. 实验数据集
- 基模 Tulu3-8B，微调预测 **8 个额外 token**;微调数据 Tulu3 数据集(开源，近 100 万样本，跨问答/数学/编码/对话/科学)。
- 评测基准(加速比 + 质量是否退化为主指标)：知识=MMLU、PopQA、TruthfulQA;数学=GSM8k;编码=HumanEval;对话=AlpacaEval、IFEval;安全=XSTest、HarmBench;另用 ARC-Challenge(Harness 库)验证 NTP 路径质量不变。〔已核-正文/Fig.6a〕

#### 8. 实验结果与主要发现
- 加速比单调随 token 数上升：代码/数学近 **5×**，通用对话/知识近 **2.5×**(知识约 2.4× 收敛)，且"无质量损失"。
- NTP 路径质量验证：ARC-Challenge zero-shot 准确率在加 gated LoRA 后不掉(普通 LoRA 会掉，Fig.6a)——证明 gated 设计的必要性。
- "无质量损失"是**结构性保证**(NTP 路径冻结、行为不变)，而非经验偶然。

#### 9. 结果如何支撑其主张
"模型隐式知道未来"由 top-200/top-10 logits 排名直接支撑;"无质量损失"由 gated LoRA 的冻结-NTP 路径结构 + ARC-Challenge 不退化实证支撑;加速比由各任务 Table/Fig 给出。主张("加速、不掉质量")与证据吻合较好。

#### 10. 逻辑自洽性(中性评估)
内部自洽：gated LoRA 用二值 mask 隔离 NTP/MTP，理论上 NTP 输出与原模型逐 token 相同，故"无质量损失"是可证而非偶然，这是相对其他 MTP 头方法的实质强项。但需注意主张严格限定在"知道未来 token(speculative 加速)"，并未主张"知道未来推理结论/提升正确率"。

#### 11. 残留问题 / 局限
- **纯加速向**工作：MTP 输出不直接用于提升推理正确率;把它当"foresight 提升推理质量"的证据时需谨慎——论文只主张模型隐式知道未来 token，未主张知道未来推理结论。
- 官方代码缺位(repo 404)：sampler 结构、consistency loss 形式、speculative 二次扩展的具体算法细节无法独立复核。〔代码受限〕
- 仅在 Tulu3-8B 单一基模上验证;k=8 的选择、对更大模型/更长上下文的扩展性未充分给出。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 链接：任务清单给出的 https://github.com/apple/ml-mtp **返回 404**(api.github.com 与 git clone 均报 "Repository not found")。Apple ML Research 论文页(machinelearning.apple.com/research/prediction-potential)未挂 GitHub 链接，仅指向 arXiv。结论：**官方代码当前未公开释出**(或仓库名有误/已下线)，本分析仅基于 PDF(14 页)。
- (社区相关但非官方实现：jwkirchenbauer/mtp-lm、Xiaohao-Liu/L-MTP 等为他人 MTP 工作，勿混淆。)
- 框架：无官方仓库。论文实现层面在预训练 decoder 上加 gated LoRA + 两层 MLP sampler head 做 SFT(只更新 LoRA 与 sampler，冻结原 decoder)。
- 可得性：**受限**，仅论文描述可用。


---


### rho1 — Rho-1: Not All Tokens Are What You Need

- **arXiv/链接**: arXiv:2404.07965v4 (https://arxiv.org/abs/2404.07965)
- **机构/作者**: 厦门大学 / 清华 / 上海AI Lab / Microsoft。Zhenghao Lin, Zhibin Gou(共一)等,Weizhu Chen 等
- **发表/时间**: NeurIPS 2024(Best Paper Runner-up);v4 2025-01-08
- **主题/相关性**: token 级数据选择 / 选择性预训练。提出 Selective Language Modeling(SLM):用参考模型给 token 打分,只在"高 excess loss"的有用 token 上算损失。**与 OPD/蒸馏中"token 级加权/选择"高度同源**——其"参考模型打分→选 token"机制,与 OPD 里"teacher 信号决定哪些 token 值得学"在思想上一致(rho1 用 excess loss = ref − train,可视为一种静态、离线的 token 级蒸馏信号),是 TSRD"教 path-selection、token 异质性"的经典先验工作。

#### 1. 开源代码链接
- https://github.com/microsoft/rho 〔已核-真实〕(README 标题/作者/HF 模型链接与论文一致;约 6.4MB,已 clone)。模型在 HF: microsoft/rho-math-1b/7b-v0.1 等。

#### 2. 使用框架
- **重要更正(已核)**:官方 microsoft/rho 仓 **只发布模型权重 + 评测代码**——`rho-1/` 目录下仅含 `math-evaluation-harness`(git 子模块,本地克隆为空/未初始化)与复现输出 `outputs.zip`;README"Quick Start"只给 evaluation(`run_eval.sh cot/tora`)。**SLM 持续预训练/微调的训练代码并未开源**(社区需自行按论文实现 token-level excess-loss masking)。故 CloneTier 实质为"权重+评测仓",核心训练栈不可核。SLM 损失原理(对 top-k% excess-loss token 做 loss mask、其余流程同标准 CLM)以论文 §2/§3 为准。

#### 3. 研究背景
LM 预训练惯例:对所有训练 token 统一施加 next-token 预测损失。数据过滤已成关键(文档级启发式/分类器),但即使经过严格文档级过滤,高质量语料在 **token 级**仍含噪声(幻觉、高度歧义、难预测 token)。

#### 4. 当前存在的问题
- 文档级过滤粒度太粗:删 token 可能改变语义,过严过滤又丢有用数据并引入偏置。
- web 数据分布与下游理想分布不天然对齐。
- 对所有 token 同等施损失→在非关键 token 上浪费算力。
- token 级训练动态分析发现:显著 loss 下降只发生在一小撮 token 上;很多是已学会的"easy token",一些是 loss 波动、难收敛的"hard token",二者带来大量无效梯度更新。

#### 5. Motivation
"语料中并非所有 token 对 LM 训练同等重要"。应聚焦于"与目标(理想)分布对齐的有用 token",把损失集中到真正有信息密度的 token(ρ = 信息密度,故名 Rho)。

#### 6. 主要方法
**Selective Language Modeling (SLM)** 三步:
1. 在高质量语料上训练一个**参考模型(reference model)**,建立对齐目标分布的 token 效用度量。
2. 用参考模型对待训语料每个 token 用其 loss 打分。
3. 训练目标模型时,只在"参考模型与训练模型之间 **excess loss**(差值)较高"的 token 上算损失(输入完整序列、选择性 mask 掉不需要的 token 的 loss),即选择性学习最利于下游的 token。

#### 7. 实验数据集
- 数学持续预训练:**15B OpenWebMath** 语料;评测 9 个数学任务(含 GSM8K、MATH)。
- 微调后:MATH 数据集 SOTA(Rho-1-1B 40.6%、7B 51.8%,仅用 DeepSeekMath 3% 的预训练 token 量)。
- 通用持续预训练:**80B 通用 token**;15 个多样任务平均 +6.8%。

#### 8. 怎么做的(训练/数据/流程)
先用高质量小语料训 reference model;再对大语料逐 token 打分;持续预训练时按 excess-loss 选 token、只反传选中 token 的损失(token-level loss masking),其余流程同标准 CLM。
- **结果**:15B OpenWebMath 持续预训练下数学 few-shot 精度绝对提升最高 +30%(9 任务);达到 baseline 性能快 5–10×(token 效率显著)。
- 〔评注/局限〕(1)SLM 的 token 选择是**静态、离线**的(参考模型一次性打分),与在线策略蒸馏(OPD)里 teacher 随 student 演化的动态信号不同——迁移到 OPD 时这是关键差异点;(2)依赖一个"对齐目标分布"的高质量参考模型,该参考模型质量直接决定 token 选择质量,存在循环依赖;(3)excess-loss 选择本质是"选 student 还学不好但 ref 学得好"的 token,与 OPD/distill 中"匹配 teacher 分布"目标相近但非同一;(4)工作面向**预训练/持续预训练**,非推理后训练 RL,定位需区分。


---


### segment_attrib — Segment-Level Attribution for Selective Learning of Long Reasoning Traces

> **一句话重点 (TL;DR)**：用 integrated gradients 直接量化每个 token 对"正确答案预测"的贡献，聚合为段落级"强度 + 方向一致性"两指标，挑出"高强度但中等一致性"的反思性段落做 loss-mask 选择性 SFT；相对 full-CoT SFT 准确率最高 +4.7%、长度最多 −18%，但温度采样下增益明显收窄，框架本身沿用 Rho-1 selective SFT。

**元信息**：arXiv 2602.00425v1（2026-01-31 提交） ｜ University of Southern California（Siyuan Wang, Yanchen Liu, Xiang Ren） ｜ PDF 页眉标注 "Published as a conference paper at ICLR 2026" ｜ 主题 长 CoT 的 SFT 段落级选择性学习 / credit-assignment（与 OPD/蒸馏"不是所有 token 都值得学"直接相关） ｜ 代码 https://github.com/SiyuanWangw/SegmentSelectiveSFT （已克隆，约 81MB，Attribution/SelectiveSFT/Eval 三阶段齐全） ｜ 框架 SFT 用 unsloth+trl，归因阶段自写 IG 脚本，评测含 latex2sympy

#### 1. 相关工作与进展
长 CoT（o1/R1/Qwen 系）靠 test-time scaling 提升推理，也成为 cold-start SFT 的监督资源（s1/Muennighoff 2025）。围绕"识别长链中重要部分以构造压缩监督"已有：token 级分析（TokenSkip/Xia 2025b）、段落级 perplexity（Cui 2025a）、段落级 entropy（Li 2025b）。selective SFT（Rho-1，Lin 2024）提供"只在部分 token 上算 loss、其余 mask"的框架。段落切分沿用 Lu 2025 的转折关键词法。

#### 2. 现有工作存在的问题
- token 级度量忽略语义完整性，不构成可解释推理单元。
- 段落级 perplexity/entropy 是与真实重要性"不完全一致"的间接指标：既有假阳（过度强调"让我们一步步算"这类脚手架桥接文本——删它会破坏后文连贯但贡献甚微），也有假阴（漏掉独立验证/中间结论这类低熵、删除不影响流畅度、但显著提升正确答案概率的段落）。
- 基于剪枝的压缩监督方法（删冗余）往往掉精度。

#### 3. Motivation
需要一个直接度量"段落对正确答案预测的影响"的指标（能同时捕捉直接与间接贡献），并以可解释的段落为单位，从而比 perplexity/entropy 更准地区分真正重要段落与各类冗余。核心实证：30~40% 的段落累计贡献 >80% 的总归因（correct/incorrect CoT 皆然，Fig.1 right-bottom CDF），说明长 CoT 存在大量冗余。

#### 4. 主要灵感 / 核心直觉
- 用 IG（Sundararajan 2017）沿 baseline→真实嵌入的直线路径积分梯度，捕捉 token 的直接+间接影响，优于"顺序追加段落看答案概率变化"或 leave-one-out（后两者会低估间接贡献且对后文完整时不敏感）。
- 方向一致性的直觉：极高一致性（token IG 几乎全正或全负）= 浅层/已定方向（冗余澄清或严重错误探索）；中等一致性 = 混合支持与纠正的反思性推理，更有学习价值。

#### 5. 主要解决思路(一段话讲清核心)
按转折关键词切段 → 对每个 token 算 IG 归因 → 聚合为段落级"强度"与"方向一致性"两指标 → 按强度降序取累计达阈值 τ 的 top-k 段落、再用一致性阈值 β 过滤掉极高一致性者，得到"重要段落集" → 只在重要段落 token 上算交叉熵、其余 mask，做全参选择性 SFT。

#### 6. 方法详解(通俗、分步骤)
- **切段**：按 "\n\nWait"、"\n\nAlternatively" 等转折关键词将 CoT 切为 {S1..Sn}（完整关键词表见 Appendix C.2）。
- **IG 归因**：IGi(x)=(xi−x'i)·∫∂F/∂xi dα，J 步插值近似，baseline x' 取 padding token embedding；token 归因 IG(x)=Σ_i IGi(x)。用绝对值捕捉影响幅度（负 IG 可能是必要的探索性推理，不应丢弃）。
- **两个段落指标**（Eq.3）：Strength(S)=Σ|IG(on)|/√N（√N 长度归一防偏长段），再在 CoT 内跨段归一化（Eq.4）；Consistency(S)=|Σ IG(on)| ÷ Σ|IG(on)|。
- **重要段落判据**（Eq.5-7）：按归一化 strength 降序，取累计 ≥ τ 的最小 top-k* 段落集，其中 Consistency ≤ β 者为"重要"。
- **超参**：贪心搜索（最大化重要/不重要段落的"正确答案置信度变化 ∆"之差）定 τ=0.7、β=0.8；τ=0.7 时平均 ~33% 段落被判重要、占 CoT 中 ~45% token（重要段落偏长）。
- **选择性 SFT**（Eq.9）：L=−(1/Σ I(ot))Σ_t I(ot)·log P(ot|·)，I(ot) 标记 token 是否属重要段落；mask 其余、保留完整轨迹连贯性。

#### 7. 实验数据集
- 训练：LIMO 数学数据集 817 题（用 provided CoT 或 R1-Distill-Qwen-7B 自生成、从 32 候选取最短正确/错误解）。
- 评测——In-domain：MATH500、AMC23、AIME24；Out-of-domain：GPQA-Diamond、Minerva、OlympiadBench。指标：greedy 准确率，或温度采样 T=0.6、max_len 32768 下 pass@1/pass@6，并报平均输出 token 数。
- 基座：R1-Distill-Qwen-1.5B/7B、Qwen2.5-7B-Instruct。IG 归因模型与训练基座一致（1.5B/7B），IG_STEPS=50。

#### 8. 实验结果与主要发现
- **段落分析**（§3）：高强度+中等一致性段落带来最大的正确答案置信度增益（Fig.2）；重要段落 perplexity/entropy 反而更低（Fig.3）；不重要段落 BLEU 自相似更高（重复），49% 被判截断 vs 重要段落仅 26%。
- **主结果**（Table 1，greedy）：R1-Distill-Qwen-1.5B overall 44.8→46.9（+4.7%），长度 16520→13506（−18.2%）；7B 62.1→（部分基准已见 MATH500 91.2→95.2、AIME24 50.0→56.7）。
- 消融：段落级优于 token 级 IG 选择、优于随机段落、优于仅取高 strength；对比 First-Correct-Solution / Confidence-Gain / Perplexity / Entropy 等度量均更优。

#### 9. 结果如何支撑其主张
"IG 段落归因能识别真正重要段落"由 §3 三类证据支撑（置信度增益、低 perplexity/entropy、低重复率），且消融证明两指标（strength+consistency）联合优于单指标与各 baseline 度量；"选择性学习提升精度+效率"由 Table 1 的 +4.7%/−18% 支撑。逻辑链完整，但 greedy 下的增益在温度采样下显著缩水。

#### 10. 逻辑自洽性(中性评估)
方法-动机自洽：用 IG 直接归因解决"间接指标不一致"的痛点，用"中等一致性"操作化"反思性推理"。一个张力点：§3 称重要段落 perplexity/entropy 更低，恰说明"低熵≠不重要"，反向支撑其相对 entropy 度量的优势，但也意味着 strength 与 entropy 度量在某些段落上结论相反，可解释性有赖 IG 假设成立。

#### 11. 残留问题 / 局限
- 框架创新有限："selective SFT + loss mask"直接来自 Rho-1（Lin 2024），本文核心新意在"IG 归因 + strength/consistency 两指标"的段落重要性度量。
- 解码敏感：温度采样下增益明显收窄（1.5B pass@1 仅 +1.6%、7B 仅 +0.5%），作者归因于采样随机性抹平训练优势，说明部分增益对解码策略敏感。
- 算力开销：IG 需 J=50 步插值，成本不低，论文未量化归因阶段的额外算力。
- 泛化证据有限：仅数学 LIMO 单一训练源、≤7B 规模。
- 超参可迁移性：τ/β 由 1.5B 的 confidence-gain 贪心搜索确定后直接套用到所有模型/数据源，论证较弱。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库 https://github.com/SiyuanWangw/SegmentSelectiveSFT （已克隆，约 81MB）。
- 三阶段脚本齐全且核心可定位：(1) Attribution——`segment_split.py` 切段、`grad_analyze.py` 算 IG、`get_important_segments.py` 按 τ/β 聚合选段、`cal_attribution.sh` 串流程；(2) SelectiveSFT——`train_mask.py`（`--mask` 开关，对非重要段落把 labels 置 −100 实现 loss-mask 全参 SFT；requirements 锁定 unsloth==2025.11.3 + trl==0.23.0 + transformers 4.57.1），`run_train.sh` 启动；(3) Eval——`evaluate.py`/`grader.py` + `latex2sympy/` 子目录，`CoT_generation.sh`/`run_eval` 评测。
- 代码与论文方法一致，loss-mask 实现已确认；归因脚本可运行，复现门槛主要在 IG 算力。


---


### srgen — Self-Reflective Generation at Test Time (SRGen)

> **一句话重点 (TL;DR)**：零训练的测试时方法，在解码过程中用动态熵阈值识别"高熵 critical token"，在该处短暂暂停、在线优化一个瞬态修正向量 δ 注入 hidden state 再发射下一 token，实现"主动错误预防"；数学/AIME 类任务增益显著，但在 AMC/GPQA/EvalPlus 上增益常仅 +0.2~+2.8pp。

**元信息**：arXiv 2510.02919（v1 2025-10-03；v2 2026-05-29，预印本未注明会议）｜ 港科大（广州）/ 南洋理工 / 爱丁堡 / 香港城大 / 港中深（Jian Mu, Qixin Zhang 等，通讯 Yao Shu）｜ 主题 测试时推理增强（零训练）/ 相关性 中（与 MTP foresight probe / 路径恢复概念相关——都在生成中识别不确定点并干预，但实现是推理时 hidden-state 修正而非训练时蒸馏，属正交可叠加方向）｜ 代码 https://github.com/2020-qqtcg/SRGen （已克隆 ~18MB，含框架 + evaluator + server，可跑）｜ 框架 HuggingFace Transformers 即插即用（含 vLLM/OpenAI 兼容 server）

#### 1. 相关工作与进展
LLM 靠长 CoT 解决复杂推理。已有纠错分两类：(1) post-hoc 迭代精修（如 Self-Refine：对完整草稿批判重写，延迟/算力高）；(2) 训练内生自纠（如 RL：需昂贵训练，且只能在错误产生后干预）。另有 SLOT、MI-Peak 等测试时方法作对照。

#### 2. 现有工作存在的问题
前向自回归解码"只能向前、无法回改"，早期 token 错误会级联放大、毁掉整条轨迹。现有自反思方法本质都是 **reactive（被动）**——只在错误已发生后才纠正；"主动错误预防"（错误被提交前就把模型引开）仍是空白。post-hoc 成本随全序列长度线性增长；RL 纠错需先产出错误片段才能介入。

#### 3. Motivation
能否在**单次解码过程中**实时识别并在潜在错误点干预，以最小额外成本提升推理可靠性？关键前提：token 信息量不同，"critical token"可由高预测熵识别。

#### 4. 主要灵感 / 核心直觉
不在错误之后纠正，而在"风险时刻"（高熵点）之前介入——短暂暂停、在线优化一个小修正向量 δ 注入 hidden state，再发射下一 token。干预局部、瞬态、不需额外完整前向草稿。

#### 5. 主要解决思路(一段话讲清核心)
两阶段 monitor-reflect-optimize 循环：用滑窗熵统计的动态阈值识别 critical token；触发时暂停解码，在投影头前的 hidden state 上优化一个瞬态 δ（最小化当前步熵同时保真历史前缀），用优化后的 logits 生成该 token 后即丢弃 δ。

#### 6. 方法详解(通俗、分步骤)
- **Stage 1 动态不确定性监测**：每步算 next-token 分布熵 H_t；维护大小 N 的滑窗，算均值 μ、标准差 σ；当 H_t > μ + k·σ 时触发反思（动态阈值适配不同模型的熵分布，避免固定阈值失效；代码另含 `minimal_threshold` 下限）。
- **Stage 2 自反思优化**：触发时暂停解码，优化瞬态向量 δ∈R^d（初始化 0），加到投影头前 hidden state：logits' = W(h_{t−1}+δ)。混合损失 L = (1−λ)·L_CE + λ·L_AEM：
  - **L_CE（回溯上下文损失）**：对已生成前缀施加同一 δ，惩罚破坏既有上下文预测的修正（保真度）；
  - **L_AEM（前瞻熵最小化）**：最小化当前步 next-token 分布熵（让决策更果断）。
  内层优化几步得 δ*，生成 y_t 后丢弃（每次干预局部化）。Theorem 1：该混合损失等价于"min L_AEM s.t. L_CE ≤ ε"约束优化的 Lagrangian，λ 隐式决定保真容忍 ε。

#### 7. 实验数据集
数学推理：AIME2024、AIME2025、HMMT2025、AMC；通用推理：GPQA；代码：EvalPlus；效率分析在 MATH500。基座：Qwen2.5-Math-7B、DeepSeek-R1-Distill-Qwen-7B、DeepSeek-R1-Distill-Llama-8B、Qwen3-32B（覆盖两架构族、7B~32B、distill/SFT/RL 多种后训练）。

#### 8. 实验结果与主要发现
- 超参（论文正文）：内层步 T=3、lr η=0.01、熵窗 N=25、std 系数 k=4；解码 T=0.6 / top-p=0.95（Qwen2.5-Math-7B 另报 T=0）；max_gen 4096（Qwen2.5-Math-7B）/32768（其余）；准确率取 5 次 pass@1 均值。
- 数学增益显著：AIME2024 上 DS-R1-Qwen-7B +12.0pp、Qwen2.5-Math-7B +7.4pp、Qwen3-32B +6.0pp，普遍优于 Self-Refine。
- 效率（MATH500/Qwen2.5-Math-7B，100 题）：wall-clock 1025s→1198s、token 7.2w→8.0w，远低于 Self-Refine（2316s）与 MI-Peak（1744s）；可与 SLOT 等叠加。
- 在 AMC、GPQA、EvalPlus 上增益往往仅 +0.2~+2.8pp，部分接近噪声。

#### 9. 结果如何支撑其主张
数学任务的稳定大增益 + 远低于 Self-Refine 的开销，支撑"低成本主动纠错"主张。但非数学任务的微弱/噪声级增益削弱了"通用可靠性提升"的泛化论断。

#### 10. 逻辑自洽性(中性评估)
方法机制自洽（高熵→触发→局部 δ 优化→丢弃）。Theorem 1 只是把加权和重述为 Lagrangian，理论新意有限，不保证 δ 优化得到更"正确"的 token，仅更"自信且保真"。代码 argparse 默认 N=20、K=2、lr=0.1，与论文报告 N=25、k=4、lr=0.01 不同；但 README 的推荐命令与 config 示例正是 N=25/K=4/lr=0.01，与论文一致——即默认值与论文实验设置不同、推荐设置一致（以论文/README 推荐为准）。代码确含 `--adaptive_entropy`/`--minimal_threshold` 开关印证动态熵阈值机制。

#### 11. 残留问题 / 局限
- 增益高度集中在"早期 slip 易翻盘"的数学/AIME 类任务，对任务类型挑剔。
- "零训练"但有真实推理开销：每个触发点 T 步反向传播优化 δ，且 L_CE 需对整段前缀重算，长前缀下单次干预成本随已生成长度增长（与论文"只随干预次数缩放"的表述存在张力，触发频繁或前缀很长时开销不可忽略）。
- 熵阈值法对已低熵的强 RL 模型可能很少触发，非数学领域泛化证据弱。
- 与本项目 MTP 仅是松散概念类比（都关注不确定点），实现路径完全不同；可借鉴价值在"动态熵阈值 + 局部修正向量"这一测试时机制本身。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- https://github.com/2020-qqtcg/SRGen （已克隆 ~18MB，含 `SRGen/` 框架与 aime/gsm8k/math/gpqa evaluator、`analysis/`、`scripts/`、OpenAI 兼容 `srgen_server.py`，代码完整可跑）。
- 框架：基于 HuggingFace Transformers 的即插即用推理框架（任意 HF 模型可用）；含 vLLM/OpenAI 兼容 server；evaluator 覆盖 AIME/GSM8K/MATH/GPQA。硬件 NVIDIA A800-80G。
- 关键实现：`SRGen/tnot_decorator.py`（熵滑窗阈值 `mean_history + K·std_history`、触发逻辑）、`SRGen/base_evaluator.py`（argparse 超参，默认 N=20/K=2/lr=0.1，推荐用论文 N=25/K=4/lr=0.01）。


---


### sstoken — ssToken: Self-modulated and Semantic-aware Token Selection for LLM Fine-tuning

> **一句话重点 (TL;DR)**：SFT 的 token 级选择方法，用"当前模型 vs 其历史模型"的 Retrospective Excess Loss（REL）替代外部参考模型，再融合一个基于注意力的语义重要性分，按比例 ρ 保留 top-ρ token 计 loss；四基座平均分最优，但增量温和、主要在需指令遵循的 QA 任务见效。

**元信息**：arXiv 2510.18250（v1 2025-10-21）｜ 上海交大 & 上海创智学院（Xiaohan Qin, Xiaoxing Wang 等，通讯 Junchi Yan）｜ ICLR 2026 ｜ 主题 SFT token 级数据选择 / 相关性 中（与"不是所有 token 都值得学 / OPD token 选择"相关，但场景是通用指令微调而非推理蒸馏；卖点是"无需参考模型 + 注意力语义信号"）｜ 代码 https://github.com/jianke0604/ssToken （已克隆 ~2.8MB，可跑）｜ 框架 自写训练脚本 + FSDP（支持 LoRA）+ lm-evaluation-harness 评测

#### 1. 相关工作与进展
SFT 中"数据质量 > 数量"已成共识；即便做过样本级过滤，高质量数据仍含 **token 级噪声**（与任务无关的冗余/无信息片段）。Rho-1 首次提出 token 级选择并显著超 full-data；TokenCleaning 在 SFT 场景进一步优化（fixed-model / self-evolving 两种清洗）。本文沿用 TokenCleaning 的数据准备与 DS²-50k 数据池。

#### 2. 现有工作存在的问题
现有 token 级选择（Rho-1、TokenCleaning）两大局限：(1) **需训练或访问额外参考模型**——直接用更强同 tokenizer 模型不总可行，单独训 reference 增成本，且 reference 质量显著影响选择效果；(2) **仅依赖 loss 信息**——token loss 反映预测不确定性，不必然反映语境中的语义重要性，频繁但语义无信息的 token 可能与任务关键 token 有相近 excess loss，loss-only 易误删有信息内容。

#### 3. Motivation
(1) 把"当前模型自身"当天然 teacher：训练推进中，当前模型相对其历史状态的进步本身就是可靠选择信号，由此摆脱外部 reference。(2) 注意力矩阵天然编码语义，可作 loss 之外的正交补充信号。

#### 4. 主要灵感 / 核心直觉
若某 token 相对历史模型 loss 显著下降，则它更可能是"可学的、有信息的"而非噪声/已掌握；而 response token 对 prompt 的注意力强度，可代理其"任务相关性/指令遵循重要性"。两信号正交，融合产生协同增益。

#### 5. 主要解决思路(一段话讲清核心)
对每个 response token，算"REL（相对历史模型的 loss 下降）"与"对 prompt 的注意力分"，归一化后线性融合为 Score，按固定比例 ρ 选 top-ρ token 计 loss、其余 mask，做带 mask 的 SFT。

#### 6. 方法详解(通俗、分步骤)
- **Self-modulated（自调制）选择**：**Retrospective Excess Loss (REL)** = L_θhis(x_i) − L_θ(x_i) = log[P_θ / P_θhis]（论文式(3)），即当前模型相对历史模型的 loss 下降（与 Rho-1 的 Excess Loss"学未来 loss"相对，REL"学历史 loss"）。历史模型可由 EMA 自适应更新（式(4)：θ_his = α·θ_his + (1−α)·θ，可选），比固定 reference 提供更稳长程指引。
- **Semantic-aware（语义感知）选择**：基于注意力的 token 重要性。利用 SFT 中所有 response token 都关注固定长度 prompt 这点，计算每个 response token 对 prompt token 的注意力之和（多头平均）作为相关性代理；用深层（deeper layer）注意力效果更好；用 hook 重算目标层注意力以兼容 FlashAttention。
- **融合**：REL 在样本内 min-max 归一到 [0,1]，注意力分天然 ∈[0,1]；最终 `Score = γ·Norm(REL) + (1−γ)·AttnScore`（默认 γ=0.5）。代码 `scripts/finetune.py`：`diff_norm = (diff-diff.min())/(diff.max()-diff.min()+1e-8)`、`combined = ratio·diff_norm + (1−ratio)·resp2prompt_scores`（与论文 Score 一致 ✓，`ratio`=γ）。按固定比例 ρ（默认 0.6）选 top-ρ token 计 loss，其余 mask（`data_prop`=ρ）。

#### 7. 实验数据集
数据池：从 5 个常用 SFT 集（Flan v2、OpenAssistant、Stanford Alpaca、Dolly、WizardLM，共 300k）采 50k（DS²-50k）；reference 基线在 DS² 样本级筛出的 10k 高质子集上训。评测 10 个通用基准：TriviaQA、TruthfulQA、MMLU、ARC-C/E、TyDiQA、Winogrande、HellaSwag、LogiQA、AGIEval。基座：LLaMA-3.2-3B、LLaMA-3.1-8B、Qwen-2.5-7B、Qwen-2.5-14B（3B~14B）。

#### 8. 实验结果与主要发现
- 四基座上 ssToken 平均分均最优，相对 full-data 提升 4.3% / 3.4% / 1.3% / 2.1%（3B/8B/7B/14B），相对 prior token 选择方法最高 +2.8%。
- TyDiQA、TriviaQA、AGIEval 等需指令遵循的 QA 任务增益最明显（归功于注意力分量）；MMLU/ARC 等知识密集任务 token 选择基本无提升。
- Rho-1/TokenCleaning 在 Qwen 系上仅与 full-data 持平甚至更差，而 ssToken 跨族稳定。

#### 9. 结果如何支撑其主张
跨四基座一致最优 + 两信号独立消融均超 full-data，支撑"无 reference + 注意力语义"双改进有效。但单族增益不均衡（Qwen-7B 仅 +1.3%），削弱"普适提升"的强主张。

#### 10. 逻辑自洽性(中性评估)
方法自洽：REL 与注意力分两正交信号 + 融合 + top-ρ mask。代码与论文 Score 公式、ρ=0.6 一致。注意：〔原稿"14B 用 0.8"为误读，论文中 ρ=0.8 是对照方法（Random/RHO-1/TokenCleaning）达各自峰值的比例（Appendix），非 ssToken 在 14B 的设定；已核实更正——论文明确 ρ=0.6 一般有效，同基座下各方法用相同 ρ 比较。〕

#### 11. 残留问题 / 局限
- 增量温和：主体仍是 Rho-1 式"top-ρ token + loss mask"范式，创新在"REL 替换 reference"与"注意力语义分"两个工程性改进。
- "无 reference"非完全免费：训练早期 history=current 使 REL 近似随机；EMA 历史模型需维护额外参数副本（显存/状态成本未充分量化）。
- 注意力分仅取"response→prompt"总注意力，长 prompt / 多轮场景有效性未验证；层选择（deeper better）依赖经验消融。
- 仅通用指令 SFT、未触及长 CoT 推理蒸馏，对本项目推理场景可迁移性需另证。
- 增益不均衡：Qwen-7B 相对 full-data 仅 +1.3%，部分单项（如 TruthfulQA）反低于 BASE/FULL。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- https://github.com/jianke0604/ssToken （已克隆 ~2.8MB，含 `scripts/` 下 `calculate_token_loss.py`、`finetune_with_hook.py`、`generate_token_label.py`、`finetune.py`，及 bash_src、fsdp_configs、eval；代码完整可跑）。
- 框架：自写训练脚本（finetune_trainer.py / finetune_with_hook.py），用 FSDP 配置、支持 LoRA；注意力重算用 hook 兼容 FlashAttention；评测用 EleutherAI lm-evaluation-harness。
- 流程：算 token loss / REL（calculate_token_loss.py）+ 注意力分（finetune_with_hook.py 重算目标层）→ 融合打分选 top-ρ → 带 mask 的 SFT。默认 γ=0.5（run.sh/finetune.sh `ratio`）、ρ=0.6（eval_tydiqa.sh `data_prop`）；同基座下所有方法用相同 ρ。


---


### vcore — VCORE: Variance-Controlled Optimization-based Reweighting for Chain-of-Thought Supervision

> **一句话重点 (TL;DR)**：把长 CoT SFT 的 token 加权形式化为"单步 SGD 下使期望 loss 下降最大、且加权分布与均匀分布 KL ≤ δ"的约束优化，闭式解为 Gibbs 分布 q\*∝exp(τ·gradient-utility)，配一个 one-backward 探针估 utility + 方差控制系数 α 稳训练；不依赖 teacher 引导/置信阈值/熵过滤，在中小模型与综合均值上稳定优于 SFT/DFT/iw-SFT。

**元信息**：arXiv 2510.27462（v1 2025-10，v2 2026-04-18，cs.CL）｜ 上海交通大学 & 香港中文大学（深圳）（Xuan Gong、Senmiao Wang、Hanbo Huang、Ruoyu Sun、Shiyu Liang 通讯）｜ ACL 2026 Main ｜ 主题 长 CoT SFT 阶段 token 级 loss reweighting（与 DFT/iw-SFT 同类），与本项目蒸馏/OPD 的 token 加权/credit assignment 直接相关，但路线是"从训练信号自身导权重"、无需 teacher 引导 ｜ 代码 https://github.com/coder-gx/VCORE （已 clone ~219MB，含 LLaMA-Factory + 改过的 transformers-4.52.4 + paper_pdf）｜ 框架 LLaMA-Factory + 定制 transformers 4.52.4

#### 1. 相关工作与进展
- **SFT for reasoning**：长 CoT SFT（从 teacher 蒸馏推理迹）是轻量有效路线，常作后续 RL 初始化；但多数工作重工程与数据配方，**优化算法层留白**。
- **SFT 中的 token reweighting**：DFT（Wu 2025）把标准 SFT 重解读为含隐式 1/π_θ 重要性因子的 policy-gradient，过度加权低概率 token，故乘上模型对目标 token 的概率来纠正；iw-SFT（Qin&Springenberg 2025）证 curated 数据上 SFT 优化某 RL 目标下界，按相对参考/当前策略的重要性加权 log-likelihood 收紧界。二者均 **RL-motivated**。VCORE 区别：**optimization-driven**，从单步 SGD 一阶下降动力学直接导权重 + 显式方差控制。

#### 2. 现有工作存在的问题
均匀加权（标准交叉熵对所有 token 等权）两大缺陷：(1) 并非所有 token 都值得学——很多 next-token 预测要么过易要么过歧义，梯度学习价值低、浪费更新拖慢收敛；(2) 自动蒸馏的 CoT（动辄 >1k token）常含幻觉/错位的 spurious token，均匀加权让噪声主导梯度、损害泛化。已有 reweighting（DFT/iw-SFT）依赖 1/π_θ 或 RL 目标下界的重要性加权，而非直接从训练时梯度信息出发。

#### 3. Motivation
中心问题：能否用**优化驱动**而非启发式的方式给 token 加权？把 token 加权形式化为"在单步 SGD 下使期望 loss 下降最大、且加权分布 q 与均匀分布 u 的 KL(q‖u)≤δ"的约束优化，从一阶下降动力学直接导出权重，不依赖 teacher 引导、置信度阈值或熵过滤。

#### 4. 主要灵感 / 核心直觉
"信用分配=梯度该去哪"。把 token 的价值定义为其梯度与全局下降方向的对齐度（gradient utility s_t=⟨∇L,∇ℓ_t⟩）——对齐度高的 token 最能降总 loss、应优先；约束 KL(q‖u)≤δ 防过度集中致不稳。这样既不靠 teacher/熵/阈值等启发式，又把"哪些 token 重要"直接从训练信号读出。

#### 5. 主要解决思路(一段话讲清核心)
对每条轨迹：(1) 一阶 Taylor 展开期望 loss 下降，定义 token 梯度效用 s_t=⟨∇L,∇ℓ_t⟩；(2) 在单纯形上 max Σq(t)s_t s.t. KL(q‖u)≤δ，闭式解 Gibbs 分布 q\*(t)∝exp(τ s_t)（τ→0 退均匀、τ→∞ 集中最高效用）；(3) 用 one-backward 探针无偏估 s_t；(4) 乘方差控制系数 α=√(V_u/V_q) 把重加权更新方差对齐到均匀加权方差。

#### 6. 方法详解(通俗、分步骤)
- **最优加权（Gibbs，§4.1）**：对 loss 一阶 Taylor，L(θ+)−L(θ)=−η Σ q_t s_t + O(η²)，s_t=⟨∇L,∇ℓ_t⟩；约束优化闭式解 q\*(t)=exp(τs_t)/Σ_j exp(τs_j)。
- **One-backward probing trick（关键效率点）**：朴素估 s_t 需每 token 一次 backward（长序列不可行）。VCORE 另抽 mini-batch B' 算均匀权下降方向 ∇L_{B'}(θ;u)，沿该方向做小扰动 ϵ 测 token loss 变化 lim_{ϵ→0}[ℓ_t(θ)−ℓ_t(θ−ϵ∇L_{B'})]/ϵ = s_t（无偏）——仅 1 次 backward + 1 次 forward 覆盖全部 |y| token，无二阶梯度/hook。
- **Variance-Controlled 缩放（§4.2）**：Gibbs 重加权改变更新方差 V_q；引入 α=√(V_u/V_q)（V_u 为均匀权方差），把重加权更新方差对齐到均匀加权（q 越尖/序列越长 α 越小以稳训练，q 平衡时 α≈1）；无额外架构改动。
- **Algorithm 1**：每 batch 抽 B'→算均匀下降方向→估 s_t→得 q\*→算 α→θ←θ−η E_{(x,y),t~q\*}[∇(α·ℓ_t)]。

#### 7. 实验数据集
- 训练（仅留 DeepSeek-R1 生成、过滤正确性的 CoT）：数学 OpenMathReasoning，代码 OpenCodeReasoning 的 C++ 子集；Qwen3 每域 3.2k（math CoT 均长 3155、code 2861），LLaMA 每域 32k。
- 评测——In-domain：AIME(24+25)、OlympiadBench-math（math），LiveCodeBench v6、OJBench（code）；Out-of-domain：R-Bench-T、SuperGPQA-1k。基座 Qwen3-{4,8,32}B、LLaMA-3.1-8B-Instruct（弱模型补充：Qwen3-1.7B、Mistral-7B-Instruct-v0.3）。指标 Pass@1（greedy，max_gen 8192），vLLM v1。

#### 8. 实验结果与主要发现
- **主结果（Table 1）平均分**：VCORE **31.03** > DFT 29.99 > SFT 28.97 > Random 28.78 > iw-SFT 28.35（ID/OOD "Avg." 列在四模型上的平均）。
- **中小模型增益更明显**：LLaMA-3.1-8B ID/OOD 从 DFT 的 5.54/7.17 升到 11.38/9.59；Qwen3-4B 从 32.49/32.49 升到 36.09/32.87；弱模型表（Table 2）Qwen3-1.7B Olympiad 53.41→55.64、Mistral-7B Olympiad 3.41→9.94。规模 8B→32B 相对 base 增益由 +4.12 升到 +4.70。
- **并非全面碾压**：Qwen3-8B 上 VCORE OOD-avg 35.26 不及 DFT 37.29；Qwen3-32B 上 VCORE 45.93 < DFT 49.57——大模型上 DFT 更强，VCORE 优势集中在 ID/OOD 综合平均与中小模型（作者归因：VCORE 按 population loss 重加权，模型已强或 CoT 与目标失配时增益有限）。
- 组件/鲁棒：方差控制必要（Fig.3 无 scaling 时 loss 频繁尖峰、有 scaling 平滑收敛）；对 τ/ϵ（Fig.2b）、batch size/lr（Table 3）鲁棒；训练集 4k→32k 上稳定优于 DFT（Fig.2a，但大集略降因质量/风格混杂）。
- **作为 RL 初始化（Obs 7，Table 4）**：Qwen3-4B/8B 用 VCORE 初始化后 GRPO 200 步，RL 后均超 DFT（即便 RL 前略低）；推测 DFT 降生成熵限制 RL 探索。
- **简单任务退化（Obs 8，Table 5）**：长 CoT SFT 在 GSM8K（短链）略降、在 MATH500（深链）升——长 CoT 监督主要利于推理密集任务。

#### 9. 结果如何支撑其主张
- "optimization-driven 优于启发式"：综合均值 VCORE > DFT/iw-SFT 支撑核心主张，且消融把增益拆给 Gibbs 加权 + 方差控制。
- "方差控制必要"：Fig.3 有/无 scaling 的 loss 曲线对比直接验证。
- "更好 RL 初始化"：Table 4 RL 后 VCORE > DFT（尽管 RL 前略低）支撑"更高 RL 天花板"。
- "鲁棒性"：τ/ϵ（Fig.2b）、bs/lr（Table 3）、训练集规模（Fig.2a）多组消融支撑超参不敏感。

#### 10. 逻辑自洽性(中性评估)
理论链条自洽：约束优化 → Gibbs 闭式解 → one-backward 无偏估计 → 方差对齐，且诚实报告了"非全面碾压""简单任务退化""大模型不及 DFT"等反例。两处需留意的假设张力：(1) 理论建立在单步一阶 Taylor + SGD 假设，实际用 AdamW，二者一致性未严格讨论；(2) one-backward trick 依赖额外 batch B' 的一次前/后向，实际每步成本约翻倍，论文未给端到端 wall-clock 开销对比。"strongest overall" 的措辞需结合"优势主要在综合均值/中小模型、大模型 DFT 更强"这一细节理解。

#### 11. 残留问题 / 局限
- 全程仅 LoRA（Qwen3 rank8/lr2e-5、LLaMA rank64/lr2e-4，alpha=2×rank，1 epoch），未做全参 SFT 验证；max_gen 限 8192 对长 CoT 推理可能偏短。
- 训练语料仅 OpenMathReasoning + OpenCodeReasoning（均 DeepSeek 蒸馏），未探更多样数据/其它 reasoning 模型生成的 CoT（作者自陈）。
- 潜在失效模式：reweighting 可能过度强调泄露最终答案的 spurious 模式、放大数据集 artifact/标注偏置（作者自陈，建议加 dropout masking / answer prefix control 正则）。
- τ/ϵ 需逐模型调（虽鲁棒但有量级敏感性，Fig.2b）；one-backward 每步成本约翻倍但无端到端计时。

#### 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库 https://github.com/coder-gx/VCORE （已 clone ~219MB），含基于 **LLaMA-Factory** 的 `llama_factory/`、定制 `transformers-4.52.4`、`figures/`、`data/`、`examples/`、`docker/`，以及仓库内 `paper_pdf/ACL_ARR_OCT_preprint.pdf`。
- 框架：LLaMA-Factory（Zheng 2024）+ 定制 transformers 4.52.4；训练 LoRA + AdamW + cosine，推理 vLLM v1；硬件 4×RTX PRO 6000 Blackwell。
- 代码可得性：完整训练栈 + 论文 PDF 内置，VCORE 的 Gibbs 加权 + one-backward 探针 + 方差控制实现于定制 transformers 训练循环中（reweighting 介入 SFT 阶段 loss）。可得性高。


---

## 6. 附录 A：被剔除论文（rejected）

下列条目在核验阶段被剔除（伪造/查无 arXiv id、代码死链 404、与主题不符、重复，或属"主题相关但相关度 Med/Low 不进入深读范围"），记录原因以备追溯，不进入正文深读。

来源：`_state/rejected.md`。另注：`comap` 一条因代码链接 404 在更早阶段即被排除。

| Key | 标题（声称） | 声称 id/链接 | 剔除原因 |
|---|---|---|---|
| vold | VOLD: Reasoning Transfer from LLMs to VLMs via On-Policy Distillation | arXiv:2510.23497 | no code (project page only; GitHub link is placeholder "your_VOLD_repo", no repo released) |
| entropy_aware_opd | Entropy-Aware On-Policy Distillation of Language Models | arXiv:2603.07079 | no code (input "code URL" was OpenReview forum WSRQ37tzk1, not a repo; no repo exists) |
| gkd_framework | GKD: A General Knowledge Distillation Framework for Large-scale Pre-trained Language Model | arXiv:2306.06629 | no code (ACL 2023 industry; no repo found) |
| high_entropy_tokens | Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RL for LLM Reasoning | arXiv:2506.01939 | no code (master "repo" URL is a Nerfies project-page website, web assets only; no dedicated code repo) |
| scale_rl | The Art of Scaling Reinforcement Learning Compute for LLMs | arXiv:2510.13786 | no code (ScaleRL recipe described; no dedicated open repo) |
| gtpo_grpos | GTPO and GRPO-S: Token and Sequence-Level Reward Shaping with Policy Entropy | arXiv:2508.04349 | no code (referenced only; no official repo, only third-party write-ups) |
| olmo3 | Olmo 3 (Base/Think/Instruct/RL-Zero) | https://allenai.org/papers/olmo3 | out of deep-read scope (Med/Low T2-T4) (T3,T4 Med) |
| deepscaler | DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL | https://github.com/agentica-project/rllm | out of deep-read scope (Med/Low T2-T4) (T3 Med) |
| rlvr_implicit | RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs | arXiv:2506.14245 | out of deep-read scope (Med/Low T2-T4) (T3 Med) |
| reason_exploration | Reasoning with Exploration: An Entropy Perspective on RL for LLMs | arXiv:2506.14758 | out of deep-read scope (Med/Low T2-T4) (T3 Med) |
| ngrpo | NGRPO: Negative-enhanced Group Relative Policy Optimization | arXiv:2509.18851 | out of deep-read scope (Med/Low T2-T4) (T3 Med) |
| dont_waste_mistakes | Don't Waste Mistakes: Leveraging Negative RL-Groups via Confidence Reweighting | arXiv:2510.08696 | out of deep-read scope (Med/Low T2-T4) (T3 Med) |
| grpo_lambda | GRPO-λ: Credit Assignment improves LLM Reasoning | arXiv:2510.00194 | out of deep-read scope (Med/Low T2-T4) (T3 Med) |
| raft_tmlr | RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment | arXiv:2304.06767 | out of deep-read scope (Med/Low T2-T4) (T3 Med) |
| agentic_rl_survey | The Landscape of Agentic Reinforcement Learning for LLMs: A Survey | arXiv:2509.02547 | out of deep-read scope (Med/Low T2-T4) (T2 Med) |
| hypereyes | HyperEyes: Dual-Grained Efficiency-Aware RL for Parallel Multimodal Search Agents | arXiv:2605.07177 | out of deep-read scope (Med/Low T2-T4) (T2 Med) |
| webrl | WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum RL | arXiv:2411.02337 | out of deep-read scope (Med/Low T2-T4) (T2 Med) |
| prm_survey | A Survey of Process Reward Models | arXiv:2510.08049 | out of deep-read scope (Med/Low T2-T4) (T4 Med) |
| demystify_longcot | Demystifying Long Chain-of-Thought Reasoning in LLMs | arXiv:2502.03373 | out of deep-read scope (Med/Low T2-T4) (T4 Med) |
| comap | （早期阶段，主题相关但代码链接 404） | — | 代码死链 404 |

## 7. 附录 B：主题相关但无代码 / 未深读候选

本附录合并 `candidates_master`（CodeFlag=N，即无可用代码链接者）+ `r2_merged` + `r3_merged` 三处候选，去重（按 arXiv id 与归一化标题），并**剔除已在 §2–§5 深读的 117 篇**，按主要主题分列简表。这些是项目的"扩展阅读 / 后续可补深读"池。

去重后共 **546 篇**：T1 = 144，T2 = 81，T3 = 175，T4 = 117，其他/相邻（SD/PC/XT/SPEC/MM 等标签为主、未归四线）= 29。表列 `标题 | arXiv/链接 | 主题标签`，主题标签保留原始多标签。

### T1 OPD 核心 / 自蒸馏（144 篇）

| 标题 | arXiv/链接 | 主题标签 |
|---|---|---|
| A Brief Overview: On-Policy Self-Distillation In LLMs | https://arxiv.org/abs/2605.18141 | T1 |
| A Note on Hybrid Online RL and Imitation Learning for LLMs | https://arxiv.org/abs/2512.23097 | T1,T3 |
| A Predictive Law for On-Policy Self-Distillation From World Feedback | https://arxiv.org/abs/2605.30070 | T1 |
| AdaSwitch: Balancing Exploration and Guidance in KD via Adaptive Switching | https://arxiv.org/abs/2510.07842 | T1 |
| ADWIN: Adaptive Windows for Horizon-Aware OPD | https://arxiv.org/abs/2605.28396 | T1,T2 |
| AllMem: A Memory-centric Recipe for Efficient Long-context Modeling | https://arxiv.org/abs/2602.13680 | T1,PC |
| AMiD: KD for LLMs with alpha-mixture Assistant Distribution | https://arxiv.org/abs/2510.15982 | T1 |
| AMR-SD: Asymmetric Meta-Reflective Self-Distillation for Token-Level Credit | https://arxiv.org/abs/2605.18529 | T1 |
| Anti-Self-Distillation for Reasoning RL via Pointwise Mutual Information | https://arxiv.org/abs/2605.11609 | T1,T4 |
| AOPD: Asymmetric On-Policy Distillation (token-level exploit/imitate) | https://arxiv.org/abs/2605.06387 | T1 |
| Are Full Rollouts Necessary for On-Policy Distillation? | https://arxiv.org/abs/2605.31490 | T1 |
| ATESD: Adaptive Teacher Exposure for Self-Distillation | https://arxiv.org/abs/2605.11458 | T1 |
| Backtracking When It Strays (Motab): Mitigating Dual Exposure Biases in Reasoning Distillation | https://arxiv.org/abs/2605.19433 | T1,T4 |
| Beyond GRPO and On-Policy Distillation | https://arxiv.org/abs/2605.12483 | T1,T3 |
| Bridging Reasoning Trajectories in On-Policy Distillation via Near-Future Guidance | https://arxiv.org/abs/2606.00305 | T1,T4 |
| Canonical-Context On-Policy Distillation | https://arxiv.org/abs/2605.30251 | T1,T2 |
| Co-Evolving Policy Distillation (CoPD) | https://arxiv.org/abs/2604.27083 | T1,T3 |
| CoDistill-GRPO: A Co-Distillation Recipe for Efficient GRPO | https://arxiv.org/abs/2605.08873 | T1,T3 |
| Combining On-Policy Optimization and Distillation for Long-Context Reasoning | https://arxiv.org/abs/2605.12227 | T1 |
| CORD: Audio-Text Reasoning via Weighted On-policy Cross-modal Distillation | https://arxiv.org/abs/2601.16547 | T1,MM |
| Counteraction-Aware Multi-Teacher On-Policy Distillation (CaMOPD) | https://arxiv.org/abs/2605.27115 | T1 |
| CREDIT: From Generic Correlation to Input-Specific Credit | https://arxiv.org/abs/2605.11613 | T1 |
| CTPD: Cross Tokenizer Preference Distillation | https://arxiv.org/abs/2601.11865 | T1,XT |
| Decomposed On-Policy Distillation for Vision-Language Reasoning (Visual Gradient Steering) | https://arxiv.org/abs/2606.00564 | T1,T4,MM |
| Decoupling KL and Trajectories: A Unified Perspective for SFT, DAgger, Offline RL, and OPD | https://arxiv.org/abs/2605.16826 | T1,T3 |
| Demystifying OPD: Length Inflation and Stabilization Strategies | https://arxiv.org/abs/2604.08527 | T1,T4 |
| Didactic to Constructive: Turning Expert Solutions into Learnable Reasoning | https://arxiv.org/abs/2602.02405 | T1,T4 |
| DiffusionOPD: A Unified Perspective | https://arxiv.org/abs/2605.15055 | T1 |
| Distillation Traps and Guards: A Calibration Knob for LLM Distillability | https://arxiv.org/abs/2604.18963 | T1,T4 |
| Distilling LLM Feedback for Lean Theorem Proving | https://arxiv.org/abs/2605.30861 | T1,T4,SD |
| DP-OPD: Differentially Private On-Policy Distillation for Language Models | https://arxiv.org/abs/2604.04461 | T1 |
| Draft-OPD: On-Policy Distillation for Speculative Draft Models | https://arxiv.org/abs/2605.29343 | T1,SPEC |
| EDGE-OPD: Internalizing Privileged Context with Evidence Guided On-Policy Distillation | https://arxiv.org/abs/2605.23493 | T1 |
| EffOPD: Learning to Foresee | https://arxiv.org/abs/2605.11739 | T1 |
| Expanding the Capabilities of Reinforcement Learning via Text Feedback | https://arxiv.org/abs/2602.02482 | T1,T3 |
| Exploratory Memory-Augmented LLM Agent via Hybrid On-/Off-Policy Optimization | https://arxiv.org/abs/2602.23008 | T1,T2 |
| f-OPD: Stabilizing Long-Horizon On-Policy Distillation with Freshness-Aware Control | https://arxiv.org/abs/2605.17862 | T1 |
| Fast and Effective On-policy Distillation from Reasoning Prefixes | https://arxiv.org/abs/2602.15260 | T1,T4 |
| Find Your Optimal Teacher: Router-Guided Multi-Teacher Distillation | https://arxiv.org/abs/2510.10925 | T1 |
| Found in Conversation: LLMs Teach Themselves | https://arxiv.org/abs/2605.24432 | T1,T2 |
| From Deferral to Learning: Online In-Context KD for LLM Cascades | https://arxiv.org/abs/2509.22984 | T1,T2 |
| GAPD: Gold-Action Policy Distillation for Agentic RL in KBQA | https://arxiv.org/abs/2605.29584 | T1,T2 |
| HDPO: Hybrid Distillation Policy Optimization via Privileged Self-Distillation | https://arxiv.org/abs/2603.23871 | T1 |
| HINT-SD: Targeted Hindsight Self-Distillation for Long-Horizon Agents | https://arxiv.org/abs/2605.17873 | T1,T2 |
| HY-MT1.5 Technical Report | https://arxiv.org/abs/2512.24092 | T1 |
| Internalize the Temperature: On-Policy Self-Distillation as Policy Reheater for RL | https://arxiv.org/abs/2606.00755 | T1,T3 |
| IRIS: Interpolative Rényi Iterative Self-play for LLM Fine-Tuning | https://arxiv.org/abs/2604.20933 | T1,T3,SD |
| It Takes Two: Complementary Self-Distillation for Contextual Integrity in LLMs | https://arxiv.org/abs/2605.20258 | T1 |
| KDRL: Post-Training Reasoning LLMs via Unified Knowledge Distillation and RL | https://arxiv.org/abs/2506.02208 | T1,T3 |
| KEPO: Knowledge-Enhanced Preference Optimization (Medical VQA) | https://arxiv.org/abs/2602.00400 | T1,MM |
| KETCHUP: K-Step Return Estimation for Sequential Knowledge Distillation | https://arxiv.org/abs/2504.19024 | T1 |
| Learn where to Click from Yourself: On-Policy Self-Distillation for GUI Grounding | https://arxiv.org/abs/2605.00642 | T1,T2 |
| Less is More: Early Stopping Rollout for On-Policy Distillation (ESR) | https://arxiv.org/abs/2605.27028 | T1 |
| LLM-Oriented Token-Adaptive Knowledge Distillation | https://arxiv.org/abs/2510.11615 | T1 |
| MAIGO: Mitigating Lost-in-Conversation | https://arxiv.org/abs/2605.27186 | T1,T2 |
| MixSD: Mixed Contextual Self-Distillation | https://arxiv.org/abs/2605.16865 | T1 |
| MT-BKD: Multi-Teacher KD via Teacher-Informed Mixture Priors | https://arxiv.org/abs/2605.27967 | T1,T4 |
| Multi-Rollout On-Policy Distillation via Peer Successes and Failures (MOPD) | https://arxiv.org/abs/2605.12652 | T1,T4 |
| Multi-Token Prediction via Self-Distillation | https://arxiv.org/abs/2602.06019 | T1,T4 |
| Not All Disagreement Is Learnable | https://arxiv.org/abs/2605.26844 | T1 |
| NPD: Near-Policy Distillation | https://arxiv.org/abs/2605.05940 | T1 |
| OGLS-SD: OP Self-Distillation with Outcome-Guided Logit Steering | https://arxiv.org/abs/2605.12400 | T1,T4 |
| OISD: On-Policy Internal Self-Distillation of Language Models | https://arxiv.org/abs/2605.29089 | T1 |
| OmniOPD: Logit-Free On-Policy Distillation via Speculative Verification | https://arxiv.org/abs/2606.01476 | T1 |
| On-Policy Context Distillation for Language Models (OPCD) | https://arxiv.org/abs/2602.12275 | T1 |
| On-Policy Distillation of LMs for AV Motion Planning | https://arxiv.org/abs/2604.07944 | T1,MM |
| On-Policy Self-Distillation for Reasoning Compression / CRISP | https://arxiv.org/abs/2603.05433 | T1,T4 |
| Online Experiential Learning for Language Models | https://arxiv.org/abs/2603.16856 | T1 |
| Online Knowledge Distillation with Reward Guidance | https://arxiv.org/abs/2505.18952 | T1,T3 |
| OPCT: On-Policy Consistency Training Improves LLM Safety with Minimal Capability Degradation | https://arxiv.org/abs/2605.21834 | T1 |
| OPD+: Rethinking the Advantage Design for On-Policy Distillation | https://arxiv.org/abs/2606.01039 | T1 |
| OPPO: Bayesian Value Recursion for Token-Level Credit Assignment in LLM Reasoning | https://arxiv.org/abs/2605.21851 | T1,T3,T4 |
| OPSD Compresses What RLVR Teaches: A Post-RL Compaction Stage for Reasoning Models | https://arxiv.org/abs/2605.06188 | T1,T4 |
| OPSDL: On-Policy Self-Distillation for Long-Context Language Models | https://arxiv.org/abs/2604.17535 | T1 |
| ORBIT: On-policy Exploration-Exploitation for Controllable Multi-Budget Reasoning | https://arxiv.org/abs/2601.08310 | T1,T4 |
| ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture Distillation | https://arxiv.org/abs/2509.25100 | T1 |
| OVD: On-policy Verbal Distillation | https://arxiv.org/abs/2601.21968 | T1,T2 |
| PACED: Distillation and On-Policy Self-Distillation at the Frontier of Student Competence | https://arxiv.org/abs/2603.11178 | T1,T4 |
| PAINT: Partial-Solution Adaptive Interpolated Training for Self-Distilled Reasoners | https://arxiv.org/abs/2604.26573 | T1,T4 |
| Positive-Unlabeled RL Distillation for On-Premise Small Models | https://arxiv.org/abs/2601.20687 | T1,T3 |
| Post-Training is About States, Not Tokens: A State Distribution View of SFT/RL/OPD | https://arxiv.org/abs/2605.22731 | T1,T3 |
| Preference-Based Self-Distillation: Beyond KL Matching via Reward Regularization | https://arxiv.org/abs/2605.05040 | T1 |
| Prefix Teach, Suffix Fade: Local Teachability Collapse | https://arxiv.org/abs/2605.13643 | T1 |
| Prune-OPD: Efficient and Reliable On-Policy Distillation | https://arxiv.org/abs/2605.07804 | T1 |
| Rabtriever: On-policy Distillation from Generative Rerankers based on JEPA | https://arxiv.org/abs/2604.23336 | T1 |
| RAFT: Data Refinement and Adaptive Distillation with Alleviated Forgetting | https://arxiv.org/abs/2606.00147 | T1,T3 |
| Reasoning Compression with Mixed-Policy Distillation (MPD) | https://arxiv.org/abs/2605.08776 | T1,T4 |
| Reinforcement Learning via Self-Distillation | https://arxiv.org/abs/2601.20802 | T1,T4 |
| Reinforcement Learning vs. Distillation: Accuracy and Capability | https://arxiv.org/abs/2505.14216 | T1,T4 |
| Reinforcement-aware Knowledge Distillation for LLM Reasoning | https://arxiv.org/abs/2602.22495 | T1,T3 |
| Respecting Self-Uncertainty in OP Self-Distillation (EGRSD/CL-EGRSD) | https://arxiv.org/abs/2605.13255 | T1 |
| Restoring the Sweet Spot: Pass-Rate Weighted Self-Distillation (SDPO variant) | https://arxiv.org/abs/2605.27765 | T1,T4 |
| Retaining by Doing: Role of On-Policy Data in Mitigating Forgetting | https://arxiv.org/abs/2510.18874 | T1,T3 |
| Rethinking LLM Distillation: A Constrained Markov Decision Process Perspective | https://arxiv.org/abs/2509.22921 | T1 |
| Rethinking LLM Distillation: A Constrained MDP Perspective | https://openreview.net/forum?id=TBJIf2M23q | T1 |
| Rethinking Selective Knowledge Distillation | https://arxiv.org/abs/2602.01395 | T1 |
| Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes | https://arxiv.org/abs/2603.25562 | T1 |
| Reward-Weighted OPD for NL-to-SVA | https://arxiv.org/abs/2605.13501 | T1 |
| RL Squeezes, SFT Expands: A Comparative Study of Reasoning LLMs | https://arxiv.org/abs/2509.21128 | T1,T3,T4 |
| RL's Razor: Why Online RL Forgets Less | https://arxiv.org/abs/2509.04259 | T1,T3 |
| RLKD: Distilling LLMs' Reasoning via Reinforcement Learning | https://arxiv.org/abs/2505.16142 | T1,T4 |
| RLRT / Rebellious Student | https://arxiv.org/abs/2605.10781 | T1,T4 |
| Rubric-based On-policy Distillation (ROPD) | https://arxiv.org/abs/2605.07396 | T1 |
| Scaling Reasoning Efficiently via Relaxed On-Policy Distillation (REOPOLD) | https://arxiv.org/abs/2603.11137 | T1,T4 |
| SD-Search: On-Policy Hindsight Self-Distillation for Search-Augmented Reasoning | https://arxiv.org/abs/2605.18299 | T1,T2 |
| Search-E1: Self-Distillation Drives Self-Evolution | https://arxiv.org/abs/2605.22511 | T1,T2 |
| Self-Distillation for Multi-Token Prediction | https://arxiv.org/abs/2603.23911 | T1,T4 |
| Self-Distilled Reinforcement Learning for Co-Evolving Agentic Recommender Systems | https://arxiv.org/abs/2604.10029 | T1,T2 |
| Self-Distilled RLVR (RLSD) | https://arxiv.org/abs/2604.03128 | T1 |
| Self-Policy Distillation via Capability-Selective Subspace Projection | https://arxiv.org/abs/2605.22675 | T1 |
| Self-Supervised On-Policy Distillation for Reasoning Language Models | https://arxiv.org/abs/2605.17497 | T1 |
| Self-Verified Distillation: Your LM Is Secretly Its Own Synthetic Data Pipeline | https://arxiv.org/abs/2605.26132 | T1,T4 |
| SimCT: Recovering Lost Supervision (cross-tokenizer) | https://arxiv.org/abs/2605.07711 | T1 |
| Skill-Conditioned Gated Self-Distillation for LLM Reasoning | https://arxiv.org/abs/2605.28791 | T1 |
| SODA: Semi On-Policy Black-Box Distillation for LLMs | https://arxiv.org/abs/2604.03873 | T1 |
| SpecKD: Speculative Decoding for Effective KD of LLMs | https://arxiv.org/abs/2510.24021 | T1 |
| Speculative Knowledge Distillation: Interleaved Sampling (SKD) | https://arxiv.org/abs/2410.11325 | T1 |
| SRR-Judge: Step-Level Rating and Refinement for Search-Integrated Reasoning | https://arxiv.org/abs/2602.07773 | T1,T2 |
| Stable On-Policy Distillation through Adaptive Target Reformulation | https://arxiv.org/abs/2601.07155 | T1 |
| StepOPSD: Step-Aware Online Preference Distillation for Agent RL | https://arxiv.org/abs/2605.27140 | T1,T2 |
| Surgical Post-Training (SPOT): Proximal On-Policy Distillation with Knowledge Retention | https://arxiv.org/abs/2603.01683 | T1,T4 |
| Tailoring Teaching to Aptitude: Direction-Adaptive Self-Distillation (DASD) | https://arxiv.org/abs/2605.22263 | T1,T4 |
| TGPO: Teacher-Guided Policy Optimization for LLM Distillation | https://arxiv.org/abs/2605.13230 | T1 |
| The Extrapolation Cliff in OPD of Near-Deterministic Structured Outputs | https://arxiv.org/abs/2605.08737 | T1 |
| The Many Faces of On-Policy Distillation | https://arxiv.org/abs/2605.11182 | T1 |
| TRACE: Distilling Where It Matters via Token-Routed Self On-Policy Alignment | https://arxiv.org/abs/2605.10194 | T1,T4 |
| Trust Region On-Policy Distillation | https://arxiv.org/abs/2606.01249 | T1 |
| Trust-Region Behavior Blending for On-Policy Distillation (TRB) | https://arxiv.org/abs/2605.31159 | T1 |
| Typhoon-S: Minimal Open Post-Training for Sovereign LLMs (SFT+OPD+RFT) | https://arxiv.org/abs/2601.18129 | T1 |
| Uni-OPD: Unifying On-Policy Distillation | https://arxiv.org/abs/2605.03677 | T1 |
| Unifying Group-Relative and Self-Distillation Policy Optimization via Sample Routing (SRPO) | https://arxiv.org/abs/2604.02288 | T1,T3 |
| Unmasking On-Policy Distillation: Where It Helps, Where It Hurts | https://arxiv.org/abs/2605.10889 | T1 |
| Video-OPD: On-Policy Distillation for Temporal Video Grounding | https://arxiv.org/abs/2602.02994 | T1,MM |
| VISD: Enhancing Video Reasoning via Structured Self-Distillation | https://arxiv.org/abs/2605.06094 | T1 |
| Vision-OPD: Learning to See Fine Details for Multimodal LLMs via OP Self-Distillation | https://arxiv.org/abs/2605.18740 | T1 |
| Visual-Advantage On-Policy Distillation for Vision-Language Models | https://arxiv.org/abs/2605.21924 | T1 |
| vOPD: On-Policy Distillation with a Control Variate Baseline | https://arxiv.org/abs/2605.07865 | T1 |
| Warmup-Distill: Bridge the Distribution Mismatch before KD | https://arxiv.org/abs/2502.11766 | T1 |
| Weak Critics Make Strong Learners: On-Policy Critique Distillation for Scalable Oversight | https://arxiv.org/abs/2606.00424 | T1,T4 |
| What and When to Distill: Selective Hindsight Distillation for Multi-Turn Agents | https://arxiv.org/abs/2605.19447 | T1,T2 |
| When Are Teacher Tokens Reliable? Position-Weighted OP Self-Distillation (PW-OPSD) | https://arxiv.org/abs/2605.21606 | T1 |
| X-KD: General Experiential Knowledge Distillation for LLMs | https://arxiv.org/abs/2602.12674 | T1 |
| X-OPD: Cross-Modal On-Policy Distillation for Speech LLMs | https://arxiv.org/abs/2603.24596 | T1,MM |
| Your Teacher Can't Help You Here | https://arxiv.org/abs/2605.30833 | T1 |

### T2 Tool-Agent / 多轮（81 篇）

| 标题 | arXiv/链接 | 主题标签 |
|---|---|---|
| A2FM: Adaptive Agent Foundation Model for Tool-Aware Hybrid Reasoning | https://arxiv.org/abs/2510.12838 | T2 |
| AgentArk: Distilling Multi-Agent Intelligence into a Single LLM Agent | https://arxiv.org/abs/2602.03955 | T2 |
| AgentEvolver: Towards Efficient Self-Evolving Agent System | https://arxiv.org/abs/2511.10395 | T2 |
| Agentic Reasoning and Tool Integration for LLMs via RL | https://arxiv.org/abs/2505.01441 | T2 |
| Agentic Reinforced Policy Optimization (ARPO) | https://arxiv.org/abs/2507.19849 | T2 |
| AgentPRM: Process Reward Models for LLM Agents via Step-Wise Promise and Progress | https://arxiv.org/abs/2511.08325 | T2,T4 |
| Aligning Language Models from User Interactions | https://arxiv.org/abs/2603.12273 | T2 |
| AT^2PO: Agentic Turn-based Policy Optimization via Tree Search | https://arxiv.org/abs/2601.04767 | T2 |
| Budget-Aware Tool-Use Enables Effective Agent Scaling | https://arxiv.org/abs/2511.17006 | T2 |
| Co-Evolution of Policy and Internal Reward for Language Agents | https://arxiv.org/abs/2604.03098 | T2 |
| Co-Evolving LLM Coder and Unit Tester via RL (CURE) | https://arxiv.org/abs/2506.03136 | T2 |
| COMAP: Co-Evolving World Models and Agent Policies for LLM Agents | https://arxiv.org/abs/2606.02372 | T2 |
| Complementary Reinforcement Learning | https://arxiv.org/abs/2603.17621 | T2 |
| CriticSearch: Fine-Grained Credit Assignment for Search Agents via Retrospective Critic | https://arxiv.org/abs/2511.12159 | T2 |
| DeepResearcher: Scaling Deep Research via RL in Real-world Environments | https://arxiv.org/abs/2504.03160 | T2 |
| Demystifying Reinforcement Learning in Agentic Reasoning | https://arxiv.org/abs/2510.11701 | T2 |
| Demystifying RL for Long-Horizon Tool-Using Agents: A Comprehensive Recipe | https://arxiv.org/abs/2603.21972 | T2 |
| Don't Just Fine-tune the Agent, Tune the Environment | https://arxiv.org/abs/2510.10197 | T2 |
| Dr. MAS: Stable RL for Multi-Agent LLM Systems | https://arxiv.org/abs/2602.08847 | T2 |
| Dynamic Dual-Granularity Skill Bank for Agentic RL (D2Skill) | https://arxiv.org/abs/2603.28716 | T2 |
| Efficient Multi-turn RL for GUI Agents via Decoupled Training + Adaptive Curation | https://arxiv.org/abs/2509.23866 | T2 |
| Experiential Reinforcement Learning | https://arxiv.org/abs/2602.13949 | T2 |
| From Web Search towards Agentic Deep ReSearch (survey/position) | https://arxiv.org/abs/2506.18959 | T2 |
| GEAR: Granularity-Adaptive Advantage Reweighting | https://arxiv.org/abs/2605.11853 | T2 |
| GRAFT: Graph-Tokenized LLMs for Tool Planning | https://arxiv.org/abs/2605.11706 | T2 |
| GrepSeek: Training Search Agents for Direct Corpus Interaction | https://arxiv.org/abs/2605.29307 | T2 |
| Healthcare AI GYM for Medical Agents | https://arxiv.org/abs/2605.02943 | T2 |
| High-Level Schedulers with Execution-Feedback RL for Long-Horizon GUI Automation | https://arxiv.org/abs/2511.22235 | T2 |
| Hindsight Credit Assignment for Long-Horizon LLM Agents (HCAPO) | https://arxiv.org/abs/2603.08754 | T2 |
| Learn from Weaknesses: Automated Domain Specialization for Small Computer-Use Agents | https://arxiv.org/abs/2605.28775 | T2 |
| Learning Agentic Policy from Action Guidance | https://arxiv.org/abs/2605.12004 | T2,T3 |
| LiteGUI: Distilling Compact GUI Agents with RL (Guided OPD + dual-level GRPO) | https://arxiv.org/abs/2605.07505 | T2 |
| LiteResearcher: Scalable Agentic RL Training Framework for Deep Research | https://arxiv.org/abs/2604.17931 | T2 |
| LongTraceRL: Long-Context Reasoning from Search Agent Trajectories with Rubric Rewards | https://arxiv.org/abs/2605.31584 | T2 |
| MemCollab: Cross-Model Memory Collaboration via Contrastive Trajectory Distillation | https://arxiv.org/abs/2603.23234 | T2 |
| Meta-RL with Self-Reflection for Agentic Search (MR-Search) | https://arxiv.org/abs/2603.11327 | T2 |
| MUA-RL: Multi-turn User-interacting Agent RL for agentic tool use | https://arxiv.org/abs/2508.18669 | T2 |
| Multi-Turn RL for Tool-Calling Agents with Iterative Reward Calibration (MT-GRPO/GTPO) | https://arxiv.org/abs/2604.02869 | T2 |
| Nemotron-Research-Tool-N1: Tool-Using LMs with Reinforced Reasoning | https://arxiv.org/abs/2505.00024 | T2 |
| O-Researcher: Open-Ended Deep Research via Multi-Agent Distillation + Agentic RL | https://arxiv.org/abs/2601.03743 | T2 |
| On Effectiveness and Efficiency of Agentic Tool-calling and RL Training | https://arxiv.org/abs/2606.00135 | T2 |
| One-Way Policy Optimization for Self-Evolving LLMs | https://arxiv.org/abs/2605.22156 | T2,T3 |
| PivotRL: High Accuracy Agentic Post-Training at Low Compute Cost | https://arxiv.org/abs/2603.21383 | T2 |
| Plan Before Search: Search Agents Need Plan | https://arxiv.org/abs/2605.28354 | T2 |
| R-Search: Empowering LLM Reasoning with Search via Multi-Reward RL | https://arxiv.org/abs/2506.04185 | T2 |
| R3L: Reflect-then-Retry RL with Pivotal Credit and Positive Amplification | https://arxiv.org/abs/2601.03715 | T2 |
| RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn RL | https://arxiv.org/abs/2504.20073 | T2 |
| Reinforcing Multi-Turn Reasoning via Turn-Level Reward Design | https://arxiv.org/abs/2505.11821 | T2 |
| ReSearch: Learning to Reason with Search for LLMs via RL | https://arxiv.org/abs/2503.19470 | T2 |
| Rethinking Agentic Reinforcement Learning In Large Language Models | https://arxiv.org/abs/2604.27859 | T2 |
| ReTool: Reinforcement Learning for Strategic Tool Use in LLMs | https://arxiv.org/abs/2504.11536 | T2 |
| Retrieval, Reward, and Training Protocols: What Matters in Training Search Agents? | https://arxiv.org/abs/2605.27881 | T2 |
| Revisiting DAgger for LLM-Agents | https://arxiv.org/abs/2605.12913 | T2 |
| RewardFlow: Topology-Aware Reward Propagation on State Graphs for Agentic RL | https://arxiv.org/abs/2603.18859 | T2,T4 |
| RL for Long-Horizon Interactive LLM Agents (LOOP) | https://arxiv.org/abs/2502.01600 | T2 |
| SAAS: Self-Aware RL for Over-Search Mitigation in Agentic Search | https://arxiv.org/abs/2605.29796 | T2 |
| Scaling Agent Learning via Experience Synthesis (DreamGym) | https://arxiv.org/abs/2511.03773 | T2 |
| Search More, Think Less: Rethinking Long-Horizon Agentic Search | https://arxiv.org/abs/2602.22675 | T2 |
| Search-R2: Search-Integrated Reasoning via Actor-Refiner Collaboration | https://arxiv.org/abs/2602.03647 | T2 |
| SEARL: Joint Optimization of Policy and Tool Graph Memory for Self-Evolving Agents | https://arxiv.org/abs/2604.07791 | T2 |
| Self-Distilled Agentic Reinforcement Learning (SDAR) | https://arxiv.org/abs/2605.15155 | T2 |
| SHARP: Shapley Credit-based Optimization for Multi-Agent Systems | https://arxiv.org/abs/2602.08335 | T2 |
| Signal Reshaping for GRPO in Weak-Feedback Agentic Code Repair | https://arxiv.org/abs/2605.07276 | T2 |
| Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents | https://arxiv.org/abs/2604.10674 | T2 |
| Stabilizing Off-Policy Training for Long-Horizon LLM Agent (SORL) | https://arxiv.org/abs/2511.20718 | T2 |
| StepSearch: Igniting LLM Search Ability via Step-Wise PPO | https://arxiv.org/abs/2505.15107 | T2 |
| Structured Agent Distillation for Large Language Model Agents | https://arxiv.org/abs/2505.13820 | T2 |
| Structured Distillation of Web Agent Capabilities Enables Generalization | https://arxiv.org/abs/2604.07776 | T2 |
| Subliminal Transfer of Unsafe Behaviors in AI Agent Distillation | https://arxiv.org/abs/2604.15559 | T2 |
| SWE-RL: Advancing LLM Reasoning via RL on Open Software Evolution | https://arxiv.org/abs/2502.18449 | T2 |
| Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use (SWiRL) | https://arxiv.org/abs/2504.04736 | T2 |
| Teaching Thinking Models to Reason with Tools: A Full-Pipeline TIR Recipe | https://arxiv.org/abs/2605.06326 | T2 |
| Thinker: Hierarchical Thinking for Deep Search via Multi-Turn Interaction | https://arxiv.org/abs/2511.07943 | T2 |
| Tool-Light: Tool-Integrated Reasoning via Self-Evolved Preference Learning | https://arxiv.org/abs/2509.23285 | T2 |
| Tool-Star: Multi-Tool Reasoner via RL | https://arxiv.org/abs/2505.16410 | T2 |
| ToolRL: Reward is All Tool Learning Needs | https://arxiv.org/abs/2504.13958 | T2 |
| Training Long-Context, Multi-Turn Software Engineering Agents with RL | https://arxiv.org/abs/2508.03501 | T2 |
| Training Task Reasoning LLM Agents for Multi-turn Planning via Single-turn RL | https://arxiv.org/abs/2509.20616 | T2 |
| UI-Voyager: A Self-Evolving GUI Agent Learning via Failed Experience | https://arxiv.org/abs/2603.24533 | T2 |
| Verified Critical Step Optimization for LLM Agents | https://arxiv.org/abs/2602.03412 | T2,T4 |
| When Agents Look the Same: Quantifying Distillation-Induced Similarity in Tool-Use | https://arxiv.org/abs/2604.21255 | T2 |

### T3 GFT 统一 SFT-RL & GRPO-RLVR（175 篇）

| 标题 | arXiv/链接 | 主题标签 |
|---|---|---|
| A Survey of RL for LLMs under Data Scarcity: Challenges and Solutions | https://arxiv.org/abs/2604.17312 | T3 |
| A2D: Adaptive Ability Decomposing for Effective RLVR | https://arxiv.org/abs/2602.00759 | T3 |
| Adaptive Negative Reinforcement for LLM Reasoning (RLVR) | https://arxiv.org/abs/2605.07137 | T3 |
| Addressing Performance Saturation for LLM RL via Precise Entropy Curve Control | https://arxiv.org/abs/2604.26326 | T3 |
| Advantage Collapse in GRPO: Diagnosis and Mitigation (ACR) | https://arxiv.org/abs/2605.21125 | T3 |
| Advantage Shaping as Surrogate Reward Maximization: Unifying Pass@K Policy Gradients | https://arxiv.org/abs/2510.23049 | T3 |
| AEM: Adaptive Entropy Modulation for Multi-Turn Agentic RL | https://arxiv.org/abs/2605.00425 | T3 |
| AIPO: Learning to Reason from Active Interaction | https://arxiv.org/abs/2605.08401 | T3 |
| Asymmetric Advantage Modulation Calibrates Entropy Dynamics in RLVR | https://arxiv.org/abs/2604.04894 | T3 |
| Asymmetric REINFORCE for off-Policy RL: Balancing positive and negative rewards (AsymRE) | https://arxiv.org/abs/2506.20520 | T3 |
| Balanced Aggregation: Understanding and Fixing Aggregation Bias in GRPO | https://arxiv.org/abs/2605.04077 | T3 |
| BandPO: Bridging Trust Regions and Ratio Clipping via Probability-Aware Bounds | https://arxiv.org/abs/2603.04918 | T3 |
| Beyond Human Data: Scaling Self-Training for Problem-Solving (ReST-EM) | https://arxiv.org/abs/2312.06585 | T3 |
| Beyond Imitation: Recovering Dense Rewards from Demonstrations | https://arxiv.org/abs/2510.02493 | T3 |
| Beyond Mode Collapse: Distribution Matching for Diverse Reasoning | https://arxiv.org/abs/2605.19461 | T3 |
| Beyond Negative Rollouts: Positive-Only Policy Optimization with Implicit Negative Gradients | https://arxiv.org/abs/2605.06650 | T3 |
| Beyond Trajectory-Level Attribution: Graph-Based Credit Assignment for Agentic RL | https://arxiv.org/abs/2605.26684 | T3 |
| Beyond Two-Stage Training: Cooperative SFT and RL (BRIDGE) | https://arxiv.org/abs/2509.06948 | T3,T4 |
| Beyond Uniform Credit Assignment: Selective Eligibility Traces for RLVR | https://arxiv.org/abs/2605.05965 | T3 |
| BREAD: Branched Rollouts from Expert Anchors Bridge SFT & RL | https://arxiv.org/abs/2506.17211 | T3 |
| Bridging Offline and Online Reinforcement Learning for LLMs | https://arxiv.org/abs/2506.21495 | T3 |
| Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning | https://arxiv.org/abs/2604.08926 | T3 |
| Buffer Matters: Off-Policy RL in LLM Reasoning (BAPO off-policy) | https://arxiv.org/abs/2602.20722 | T3 |
| CAST: Non-Privileged Clipped Asymmetric Self-Teaching with Advantage Flipping | https://arxiv.org/abs/2606.00172 | T3 |
| CE-GPPO: Controlling Entropy via Gradient-Preserving Clipping | https://arxiv.org/abs/2509.20712 | T3 |
| CLIPO: Contrastive Learning in Policy Optimization Generalizes RLVR | https://arxiv.org/abs/2603.10101 | T3 |
| Clipping Bottleneck: Stabilizing RLVR via Stochastic Recovery of Near-Boundary Signals | https://arxiv.org/abs/2605.22703 | T3 |
| CorR-PO: Entropy-Gradient Inversion / Correlation-Regularized Group PO | https://arxiv.org/abs/2605.17770 | T3,T4 |
| CurveRL: Principled Distribution-Aware Context Reweighting for LLM Reasoning | https://arxiv.org/abs/2605.24331 | T3 |
| Decouple before Integration: Test-time Synthesis of SFT and RLVR Task Vectors (DoTS) | https://arxiv.org/abs/2605.00610 | T3 |
| Deep Dense Exploration for LLM RL via Pivot-Driven Resampling | https://arxiv.org/abs/2602.14169 | T3 |
| DelTA: Discriminative Token Credit Assignment for RLVR | https://arxiv.org/abs/2605.21467 | T3,T4 |
| Demystifying GRPO: Its Policy Gradient is a U-Statistic | https://arxiv.org/abs/2603.01162 | T3 |
| DeReason: Difficulty-Aware Curriculum for Decoupled SFT-then-RL | https://arxiv.org/abs/2603.11193 | T3,T4 |
| DGPO: Decoupled Gradient Policy Optimization (Taming Divergence in Soft Clipping) | https://arxiv.org/abs/2603.14389 | T3 |
| DGPO: Distribution Guided Policy Optimization for Fine-Grained Credit | https://arxiv.org/abs/2605.03327 | T3 |
| Diagnosing Training-Inference Mismatch in LLM RL | https://arxiv.org/abs/2605.14220 | T3 |
| DISA: Decoupled Importance-Sampled Anchoring for Distribution-Matching RL | https://arxiv.org/abs/2605.17295 | T3 |
| Distribution-Aligned Hint Synthesis + Backward Hint Annealing for Math RLVR | https://arxiv.org/abs/2604.07747 | T3,T4 |
| DRIFT: Decoupled Rollouts + Importance-Weighted Fine-Tuning for Multi-Turn | https://arxiv.org/abs/2605.31455 | T3 |
| DSDR: Dual-Scale Diversity Regularization for Exploration | https://arxiv.org/abs/2602.19895 | T3 |
| DVAO: Dynamic Variance-adaptive Advantage Optimization | https://arxiv.org/abs/2605.25604 | T3 |
| EAD: Exploratory Annealed Decoding for Verifiable RL | https://arxiv.org/abs/2510.05251 | T3 |
| EBPO: Empirical Bayes Shrinkage for Stabilizing GRPO | https://arxiv.org/abs/2602.05165 | T3 |
| EchoRL: Reinforcement Learning via Rollout Echoing | https://arxiv.org/abs/2605.31228 | T3 |
| Entropy Polarity in Reinforcement Fine-Tuning | https://arxiv.org/abs/2605.11775 | T3 |
| Entropy-Gated Selective Policy Optimization (EG-SPO) | https://arxiv.org/abs/2602.03309 | T3 |
| Entropy-KL Divergence-based Token Masking: Selective Fine-tuning of LLMs | https://arxiv.org/abs/2605.29303 | T3 |
| Entropy-Preserving Reinforcement Learning | https://arxiv.org/abs/2603.11682 | T3 |
| ESPO: Early-Stopping Proximal Policy Optimization | https://arxiv.org/abs/2605.29860 | T3 |
| Exploration-Driven Optimization for Test-Time LLM Reasoning (EDO) | https://arxiv.org/abs/2605.09853 | T3 |
| FBOS-RL: Feedback-Driven Bi-Objective Synergistic RL | https://arxiv.org/abs/2605.20256 | T3 |
| FIPO: Eliciting Deep Reasoning with Future-KL Influenced Policy Optimization | https://arxiv.org/abs/2603.19835 | T3,T4 |
| Flexible Entropy Control in RLVR with Gradient-Preserving Perspective | https://arxiv.org/abs/2602.09782 | T3 |
| FSPO: Clip Your Sequences Fairly (Length Fairness for Sequence-Level RL) | https://arxiv.org/abs/2509.09177 | T3 |
| GAC: Noise-Aware Adaptive Mixing for Hybrid SFT-RL Post-Training | https://arxiv.org/abs/2605.26184 | T3 |
| GCPO: Cooperative Policy Optimization Improves Diverse Reasoning | https://arxiv.org/abs/2605.11461 | T3 |
| Getting Your LLMs Ready for RL with Lightweight SFT | https://openreview.net/forum?id=yezWGJmODg | T3 |
| GFT: From Imitation to Reward Fine-Tuning (Group Advantages + Dynamic Coeff Rectification) | https://arxiv.org/abs/2604.14258 | T3 |
| GIFT: Reconciling Post-Training Objectives via Finite-Temperature Gibbs Initialization | https://arxiv.org/abs/2601.09233 | T3,T4 |
| Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning | https://arxiv.org/abs/2602.01058 | T3 |
| Gradients Must Earn Their Influence: Unifying SFT with Generalized Entropic Objectives | https://arxiv.org/abs/2602.11424 | T3 |
| Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm | https://arxiv.org/abs/2509.24203 | T3 |
| Guidance Contrastive Token Credit Assignment for Discrete Policy Optimization | https://arxiv.org/abs/2605.29198 | T3,T4 |
| GXPO: Gradient Extrapolation-Based Policy Optimization | https://arxiv.org/abs/2605.06755 | T3 |
| Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks | https://arxiv.org/abs/2602.22817 | T3 |
| HiLL: Learning to Hint for Reinforcement Learning | https://arxiv.org/abs/2604.00698 | T3 |
| Hindsight-Anchored Policy Optimization (HAPO) | https://arxiv.org/abs/2603.11321 | T3 |
| HINT: Helping Ineffective Rollouts Navigate Towards Effectiveness | https://arxiv.org/abs/2510.09388 | T3 |
| Hista and Numca: Estimate State Value for LLM RL | https://arxiv.org/abs/2605.29782 | T3 |
| How Much Backtracking is Enough? Interplay of SFT and RL | https://arxiv.org/abs/2505.24273 | T3,T4 |
| How Off-Policy Can GRPO Be? Mu-GRPO | https://arxiv.org/abs/2605.17570 | T3 |
| How You Begin is How You Reason: Driving Exploration in RLVR via Prefix-Tuned Priors | https://arxiv.org/abs/2605.08817 | T3,T4 |
| HTPO: Exploration-Exploitation Balanced PO via Hierarchical Token-level Objective Control | https://arxiv.org/abs/2605.08283 | T3,T4 |
| Implicit Reward as the Bridge: A Unified View of SFT and DPO Connections | https://arxiv.org/abs/2507.00018 | T3 |
| IN-RIL: Interleaved Reinforcement and Imitation Learning for Policy Fine-Tuning | https://arxiv.org/abs/2505.10442 | T3 |
| InfoSFT: Learn More and Forget Less with Information-Aware Token Weighting | https://arxiv.org/abs/2605.14967 | T3 |
| It Takes Two: Your GRPO Is Secretly DPO | https://arxiv.org/abs/2510.00977 | T3 |
| It's Not You, It's Clipping: A Soft Trust-Region via Probability Smoothing | https://arxiv.org/abs/2509.21282 | T3 |
| LAD: Learning Advantage Distribution for Reasoning | https://arxiv.org/abs/2602.20132 | T3 |
| LambdaPO: Lambda Style Policy Optimization | https://arxiv.org/abs/2605.19416 | T3 |
| LamPO: A Lambda-Style Policy Optimization for Reasoning Language Models | https://arxiv.org/abs/2605.21235 | T3 |
| LLM Post-Training: A Unified View of Off-Policy and On-Policy Learning | https://arxiv.org/abs/2604.07941 | T3 |
| Long Live The Balance: Information-Bottleneck-Driven Tree-based Policy Optimization | https://arxiv.org/abs/2605.28109 | T3 |
| LoPE: Lorem Perturbation for Exploration | https://arxiv.org/abs/2605.05566 | T3 |
| MCPO: Mastery-Consolidated Policy Optimization | https://arxiv.org/abs/2604.16972 | T3 |
| Mechanistically Interpreting the Role of Sample Difficulty in RLVR for LLMs | https://arxiv.org/abs/2605.28388 | T3 |
| Missing Old Logits in Asynchronous Agentic RL: Off-Policy Repair | https://arxiv.org/abs/2605.12070 | T3 |
| Mitigating Forgetting Between SFT and RL Yields Stronger Reasoners (MIFO) | https://arxiv.org/abs/2510.04454 | T3 |
| Mono-Anchored Advantage Normalization for Multi-Source Visual Reasoning | https://arxiv.org/abs/2605.25437 | T3,MM |
| Multi-Step Likelihood-Ratio Correction for RL with Verifiable Rewards | https://arxiv.org/abs/2605.20865 | T3 |
| Near-Future Policy Optimization (NPO) | https://arxiv.org/abs/2604.20733 | T3 |
| Not Every Rubric Teaches Equally: Policy-Aware Rubric Rewards for RLVR | https://arxiv.org/abs/2605.20164 | T3 |
| Not Only Where, But When: Temporal Scheduling for RLVR | https://arxiv.org/abs/2605.25381 | T3 |
| Off-Policy Learning to Reason Works Because It Is More Pessimistic | https://arxiv.org/abs/2605.28150 | T3 |
| OGER: Robust Offline-Guided Exploration Reward for Hybrid RL | https://arxiv.org/abs/2604.18530 | T3 |
| On the Direction of RLVR Updates: Identification and Exploitation | https://arxiv.org/abs/2603.22117 | T3 |
| On the Implicit Reward Overfitting and Low-rank Dynamics in RLVR | https://arxiv.org/abs/2605.06523 | T3 |
| On the Non-decoupling of SFT and RL in Post-training | https://arxiv.org/abs/2601.07389 | T3 |
| One Ring to Rule Them All: Unifying Group-Based RL via Power-Mean Geometry | https://arxiv.org/abs/2601.22521 | T3 |
| One-Token Rollout (OTR): Guiding SFT with Policy Gradient | https://arxiv.org/abs/2509.26313 | T3 |
| Online Causal Kalman Filtering for Stable Policy Optimization | https://arxiv.org/abs/2602.10609 | T3 |
| Online SFT for LLM Reasoning: Self-Tuning without Rewards | https://arxiv.org/abs/2510.18814 | T3 |
| OpenVLThinker: Vision-Language Reasoning via Iterative SFT-RL | https://openreview.net/pdf/62ceb097c643e0416c764c187ebf4f4d6d1ba9c3.pdf | T3 |
| PEPO: Perception-Exploration Policy Optimization for Multimodal CoT | https://arxiv.org/abs/2603.22847 | T3 |
| PR2: Predictive Routing Replay for MoE-Based LLM RL | https://arxiv.org/abs/2606.00395 | T3 |
| Probability-Entropy Calibration: An Elastic Indicator for Adaptive Fine-tuning (PEC) | https://arxiv.org/abs/2602.01745 | T3 |
| Probing to Refine: Reinforcement Distillation via Explanatory Inversion (ExGRPO) | https://arxiv.org/abs/2603.19266 | T3 |
| Quagmires in SFT-RL Post-Training: When High SFT Scores Mislead | https://arxiv.org/abs/2510.01624 | T3 |
| R2VPO: Ratio-Variance Regularized Policy Optimization | https://arxiv.org/abs/2605.26784 | T3 |
| Recycling Failures: Salvaging Exploration in RLVR via Fine-Grained Off-Policy Guidance (SCOPE) | https://arxiv.org/abs/2602.24110 | T3,T4 |
| Reference-Sampled Boltzmann Projection for KL-Regularized RLVR (weighted SFT) | https://arxiv.org/abs/2605.02469 | T3 |
| RePO: Bridging On-Policy Learning and Off-Policy Knowledge through Rephrasing Policy Optimization | https://arxiv.org/abs/2602.10819 | T3 |
| RePO: Replay-Enhanced Policy Optimization | https://arxiv.org/abs/2506.09340 | T3 |
| Resolving Action Bottleneck: Agentic RL Informed by Token-Level Energy | https://arxiv.org/abs/2605.14558 | T3 |
| Rethinking Entropy Interventions in RLVR (STEER) | https://arxiv.org/abs/2510.10150 | T3 |
| Rethinking Expert Trajectory Utilization in LLM Post-training for Mathematical Reasoning | https://arxiv.org/abs/2512.11470 | T3 |
| Rethinking Generalization in Reasoning SFT: Conditional Analysis | https://arxiv.org/abs/2604.06628 | T3 |
| Rethinking Muon Beyond Pretraining: Spectral Failures for VLA and RLVR | https://arxiv.org/abs/2605.19282 | T3 |
| Rethinking RL for LLM Reasoning: It's Sparse Policy Selection, Not Capability Learning | https://arxiv.org/abs/2605.06241 | T3 |
| Rethinking Sample Polarity in RLVR | https://arxiv.org/abs/2512.21625 | T3 |
| Reuse your FLOPs: Scaling RL on Hard Problems by Conditioning on Very Off-Policy Prefixes (PrefixRL) | https://arxiv.org/abs/2601.18795 | T3,T4 |
| RIFT: Repurposing Negative Samples via Reward-Informed Fine-Tuning | https://arxiv.org/abs/2601.09253 | T3 |
| RL Fine-Tuning Heals OOD Forgetting in SFT | https://arxiv.org/abs/2509.12235 | T3 |
| RL Is Neither a Panacea Nor a Mirage: SFT vs RL Fine-Tuning | https://arxiv.org/abs/2508.16546 | T3 |
| RL-finetuning LLMs from on- and off-policy data with a single algorithm (AGRO) | https://arxiv.org/abs/2503.19612 | T3 |
| RLHF in an SFT Way: From Optimal Solution to Reward-Weighted Alignment | https://arxiv.org/abs/2502.11026 | T3 |
| RLoop: Self-Improving Framework with Iterative Policy Initialization | https://arxiv.org/abs/2511.04285 | T3 |
| Sample More to Think Less: Group Filtered Policy Optimization (GFPO) | https://arxiv.org/abs/2508.09726 | T3 |
| Scalpel vs. Hammer: GRPO Amplifies Capabilities, SFT Replaces Them | https://arxiv.org/abs/2507.10616 | T3 |
| SCRL: Subproblem Curriculum RL Enables Credit Assignment | https://arxiv.org/abs/2605.22074 | T3 |
| Selective Expert Guidance for Exploration in RL of LLMs (MENTOR) | https://arxiv.org/abs/2510.04140 | T3 |
| Selective Off-Policy Reference Tuning with Plan Guidance (SORT) | https://arxiv.org/abs/2605.11505 | T3 |
| Self-Hinting Language Models Enhance RL (SAGE) | https://arxiv.org/abs/2602.03143 | T3 |
| SFT versus RL: A Study of Post-Training Methods for LLMs | https://arxiv.org/abs/2603.13985 | T3 |
| SIMPLEMIX: Simple Mixing of Off- and On-policy Data in Preference Learning | https://arxiv.org/abs/2505.02363 | T3 |
| Single-stream Policy Optimization (SPO) | https://arxiv.org/abs/2509.13232 | T3 |
| SIOP: Self-Induced Outcome Potential for Turn-Level Credit without Verifiers | https://arxiv.org/abs/2605.04984 | T3 |
| Smaller Models are Natural Explorers for Policy-Level Diversity in GRPO | https://arxiv.org/abs/2605.30789 | T3 |
| Soft Adaptive Policy Optimization (SAPO) | https://arxiv.org/abs/2511.20347 | T3 |
| SPINE: Token-Selective Test-Time RL with Entropy-Band Regularization | https://arxiv.org/abs/2511.17938 | T3,T4 |
| Stabilizing LLM Supervised Fine-Tuning via Explicit Distributional Control (Anchored-SFT) | https://arxiv.org/abs/2605.04468 | T3 |
| Stabilizing MoE RL by Aligning Training and Inference Routers | https://arxiv.org/abs/2510.11370 | T3 |
| STAPO: Stabilizing RL by Silencing Rare Spurious Tokens | https://arxiv.org/abs/2602.15620 | T3,T4 |
| STaR: Bootstrapping Reasoning With Reasoning | https://arxiv.org/abs/2203.14465 | T3 |
| Step-wise Adaptive Integration of SFT and RL (SASR) | https://arxiv.org/abs/2505.13026 | T3 |
| StraTA: Strategic Trajectory Abstraction for Agentic RL | https://arxiv.org/abs/2605.06642 | T3 |
| Supervised Fine Tuning on Curated Data is Reinforcement Learning (and can be improved) (iw-SFT) | https://arxiv.org/abs/2507.12856 | T3 |
| Supervised Fine-Tuning Needs to Unlock the Potential of Token Priority | https://arxiv.org/abs/2602.01227 | T3 |
| Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning (SRL) | https://arxiv.org/abs/2510.25992 | T3,T4 |
| T-STAR: Tree-structured Self-Taught Agent Rectification | https://arxiv.org/abs/2604.07165 | T3 |
| T2PO: Uncertainty-Guided Exploration Control for Multi-Turn Agentic RL | https://arxiv.org/abs/2605.02178 | T3 |
| Tapered Off-Policy REINFORCE (TOPR): Stable and efficient RL for LLMs | https://arxiv.org/abs/2503.14286 | T3 |
| The Cancellation Hypothesis in Critic-Free RL: From Outcome Rewards to Token Credits | https://arxiv.org/abs/2605.08666 | T3,T4 |
| The Unlearnability Phenomenon in RLVR for Language Models | https://arxiv.org/abs/2605.16787 | T3 |
| TMS: Trajectory-Mixed Supervision for Reward-Free, On-Policy SFT | https://arxiv.org/abs/2602.03073 | T3 |
| Token Hidden Reward: Steering Exploration-Exploitation in GRPO | https://arxiv.org/abs/2510.03669 | T3,T4 |
| Token-Level Policy Optimization: Linking Group-Level Rewards to Token-Level Aggregation via Markov Likelihood (TEPO) | https://arxiv.org/abs/2510.09369 | T3,T4 |
| Towards On-Policy SFT: Distribution Discriminant Theory (IDFT/Hinted Decoding) | https://arxiv.org/abs/2602.12222 | T3 |
| Training a Scientific Reasoning Model for Chemistry | https://openreview.net/pdf?id=Mr5Pyb3MLk | T3 |
| Trust the Batch, On- or Off-Policy: Adaptive PO via Effective Sample Size | https://arxiv.org/abs/2605.12380 | T3 |
| Two is Better than One: Collapse-free Multi-Reward RLIF | https://arxiv.org/abs/2605.22620 | T3 |
| UCPO: Uniform-Correct Policy Optimization | https://arxiv.org/abs/2605.00365 | T3 |
| UFT: Unifying Fine-Tuning of SFT and RLHF/DPO/UNA via Implicit Reward | https://arxiv.org/abs/2410.21438 | T3 |
| UFT: Unifying Supervised and Reinforcement Fine-Tuning | https://arxiv.org/abs/2505.16984 | T3,T4 |
| UniAPL: A Unified Adversarial Preference Learning Framework for Instruct-Following | https://arxiv.org/abs/2509.25148 | T3 |
| VAPO: Efficient and Reliable RL for Advanced Reasoning | https://arxiv.org/abs/2504.05118 | T3 |
| VeriGate: Verifier-Gated Step-Level Supervision for GRPO | https://arxiv.org/abs/2605.30451 | T3 |
| VI-CuRL: Verifier-Independent RL via Confidence-Guided Variance Reduction | https://arxiv.org/abs/2602.12579 | T3 |
| Weak-to-Strong Elicitation via Mismatched Wrong Drafts | https://arxiv.org/abs/2605.17314 | T3 |
| When Can LLMs Learn to Reason with Weak Supervision? | https://arxiv.org/abs/2604.18574 | T3 |
| When RL Suppresses Its Own Vocabulary: Recovering Reasoning Diversity in Puzzle-to-Math Transfer | https://arxiv.org/abs/2605.29190 | T3 |
| When to Stop Reusing: Dynamic Gradient Gating for Sample-Efficient RLVR | https://arxiv.org/abs/2605.19425 | T3 |
| Where Rollouts Begin: Low-Load, High-Leverage First-Token Diversification for RLVR | https://arxiv.org/abs/2605.28295 | T3 |
| Where to Spend Rollouts: Hit-Utility-Optimal Rollout Allocation for Group-Based RLVR | https://arxiv.org/abs/2605.07114 | T3 |

### T4 思维链-Token 级 / MTP（117 篇）

| 标题 | arXiv/链接 | 主题标签 |
|---|---|---|
| AIR: Post-training Data Selection for Reasoning via Attention Head Influence | https://arxiv.org/abs/2512.13279 | T4 |
| Arbitrage: Efficient Reasoning via Advantage-Aware Speculation | https://arxiv.org/abs/2512.05033 | T4 |
| Attention Illuminates LLM Reasoning: Preplan-and-Anchor Rhythm | https://arxiv.org/abs/2510.13554 | T4 |
| Beyond High-Entropy: Correctness-Aware Low-Entropy Segment Advantage Shaping | https://arxiv.org/abs/2512.00908 | T4 |
| Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries (FSP) | https://arxiv.org/abs/2510.14751 | T4 |
| Beyond Uniform Credit: Causal Credit Assignment for Policy Optimization | https://arxiv.org/abs/2602.09331 | T4 |
| Breaking the Exploration Bottleneck: Rubric-Scaffolded RL | https://arxiv.org/abs/2508.16949 | T4 |
| COMPACT: Compatibility-Aware Multi-Teacher CoT Distillation | https://arxiv.org/abs/2601.13992 | T4 |
| Confidence-Guided Stepwise Model Routing for Cost-Efficient Reasoning | https://arxiv.org/abs/2511.06190 | T4 |
| Critical Tokens Matter: Token-Level Contrastive Estimation | https://arxiv.org/abs/2411.19943 | T4 |
| Curriculum Learning for Efficient CoT Distillation via Structure-Aware Masking+GRPO | https://arxiv.org/abs/2602.17686 | T4 |
| D-CoT: Disciplined Chain-of-Thought Learning for Efficient Reasoning in SLMs | https://arxiv.org/abs/2602.21786 | T4 |
| DeAR: Dual-Stage Document Reranking with Reasoning Agents | https://aclanthology.org/2025.findings-emnlp.306/ | T4 |
| Deliberative Alignment: Reasoning Enables Safer Language Models | https://arxiv.org/abs/2412.16339 | T4 |
| Distilling Reasoning into Student LLMs: Local Naturalness for Teacher Data | https://arxiv.org/abs/2510.03988 | T4 |
| Distilling the Essence: Efficient Reasoning Distillation via Sequence Truncation | https://arxiv.org/abs/2512.21002 | T4 |
| Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Distillation | https://arxiv.org/abs/2602.20816 | T4 |
| Dynamic Thinking-Token Selection for Efficient Reasoning in Large Reasoning Models | https://arxiv.org/abs/2601.18383 | T4 |
| Dynamics Within Latent CoT: Empirical Study of Causal Structure | https://arxiv.org/abs/2602.08783 | T4 |
| Effectiveness of Chain-of-Thought in Distilling Reasoning Capability | https://arxiv.org/abs/2511.05184 | T4 |
| Emergent Search and Backtracking in Latent Reasoning Models | https://arxiv.org/abs/2602.08100 | T4 |
| Enhancing Generalization in Chain of Thought Reasoning for Smaller Models | https://arxiv.org/abs/2501.09804 | T4 |
| Enhancing LLM Reasoning via Non-Human-Like Reasoning Path Preference Optimization | https://arxiv.org/abs/2510.11104 | T4 |
| Enhancing LLM Reasoning via Selective Critical Token Fine-Tuning (CFT) | https://arxiv.org/abs/2510.10974 | T4 |
| Enhancing Long-Chain Reasoning Distillation through Error-Aware Self-Reflection | https://arxiv.org/abs/2505.22131 | T4 |
| ERPO: Token-Level Entropy-Regulated Policy Optimization | https://arxiv.org/abs/2603.28204 | T4 |
| ESPO: Entropy Importance Sampling Policy Optimization | https://arxiv.org/abs/2512.00499 | T4 |
| EST-PRM: Stress-Testing Process Reward Models Before They Become Load-Bearing | https://arxiv.org/abs/2606.00437 | T4 |
| Explain in Your Own Words: Token-Selective Dual Knowledge Distillation (TSD-KD) | https://arxiv.org/abs/2603.13260 | T4 |
| Fast and Expressive Multi-Token Prediction with Probabilistic Circuits (MTPC) | https://arxiv.org/abs/2511.11346 | T4 |
| FastMTP: Accelerating LLM Inference with Enhanced Multi-Token Prediction | https://arxiv.org/abs/2509.18362 | T4 |
| Finding the Cracks: Paraphrastic Probing and Consistency Verification | https://arxiv.org/abs/2602.11361 | T4 |
| From Emergence to Control: Probing and Modulating Self-Reflection | https://arxiv.org/abs/2506.12217 | T4 |
| Gained in Translation: Privileged Pairwise Judges Enhance Multilingual Reasoning | https://arxiv.org/abs/2601.18722 | PC,T4 |
| GCPO: Group Critical-token Policy Optimization for Autoregressive Image Generation | https://arxiv.org/abs/2509.22485 | T4,MM |
| GradAlign: Gradient-Aligned Data Selection for LLM RL | https://arxiv.org/abs/2602.21492 | T4 |
| Gumbel Distillation for Parallel Text Generation | https://arxiv.org/abs/2603.22216 | T4 |
| How Does Prefix Matter in Reasoning Model Tuning? | https://arxiv.org/abs/2601.01624 | T4 |
| How Transformers Learn to Plan via Multi-Token Prediction | https://arxiv.org/abs/2604.11912 | T4 |
| Ignore the KL Penalty! Boosting Exploration on Critical Tokens to Enhance RL Fine-Tuning | https://arxiv.org/abs/2502.06533 | T4 |
| In-Token Rationality Optimization (InTRO): Concise Reasoning via Self-Feedback | https://arxiv.org/abs/2511.09865 | T4,SD |
| Keypoint-based Progressive Chain-of-Thought Distillation for LLMs | https://arxiv.org/abs/2405.16064 | T4 |
| Knapsack RL: Optimizing Exploration Budget Allocation | https://arxiv.org/abs/2509.25849 | T4 |
| Latent Chain-of-Thought as Planning: Decoupling Reasoning from Verbalization | https://arxiv.org/abs/2601.21358 | T4 |
| Learning Concepts, Not Tokens: Self-Supervised Semantic Alignment | https://arxiv.org/abs/2603.29123 | T4 |
| Learning Reasoning Rewards from Expert Demonstrations with Inverse RL | https://arxiv.org/abs/2510.01857 | T4 |
| Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning | https://arxiv.org/abs/2506.07851 | T4 |
| Long-Chain Reasoning Distillation via Adaptive Prefix Alignment | https://arxiv.org/abs/2601.10064 | T4 |
| LongPerceptualThoughts: Distilling System-2 Reasoning for System-1 Perception | https://colmweb.org/2025/AcceptedPapers.html | T4 |
| LookaheadKV: KV Cache Eviction by Glimpsing into the Future | https://arxiv.org/abs/2603.10899 | T4 |
| MiCoTA: Bridging the Learnability Gap with Intermediate CoT and Teacher Assistants | https://arxiv.org/abs/2507.01887 | T4 |
| Multi-chain Graph Refinement and Selection for Reliable Reasoning | https://arxiv.org/abs/2511.23136 | T4 |
| Multi-Stream LLMs: Parallel Streams of Thoughts/Inputs/Outputs | https://arxiv.org/abs/2605.12460 | T4 |
| Nanbeige4.1-3B: Small General Model that Reasons, Aligns, Acts | https://arxiv.org/abs/2602.13367 | T4 |
| NaturalThoughts: Selecting and Distilling Reasoning Traces for General Reasoning Tasks | https://arxiv.org/abs/2507.01921 | T4 |
| Next-Latent Prediction Transformers Learn Compact World Models (NextLat) | https://arxiv.org/abs/2511.05963 | T4 |
| NITP: Next Implicit Token Prediction for LLM Pre-training | https://arxiv.org/abs/2605.24956 | T4 |
| On Multi-Token Prediction for Efficient LLM Inference | https://arxiv.org/abs/2502.09419 | T4 |
| On the Step Length Confounding in LLM Reasoning Data Selection | https://arxiv.org/abs/2604.06834 | T4 |
| Outcome-Grounded Advantage Reshaping for Fine-Grained Credit Assignment | https://arxiv.org/abs/2601.07408 | T4 |
| Pair-In, Pair-Out (PIPO): Latent Multi-Token Prediction | https://arxiv.org/abs/2605.27255 | T4 |
| Parallel Token Prediction for Language Models (PTP) | https://arxiv.org/abs/2512.21323 | T4 |
| Path-Lock Expert: Separating Reasoning Mode in Hybrid Thinking | https://arxiv.org/abs/2604.27201 | T4 |
| Pre-Training Curriculum for Multi-Token Prediction in Language Models | https://arxiv.org/abs/2505.22757 | T4 |
| Pre-training under Infinite Compute | https://arxiv.org/abs/2509.14786 | T4 |
| Process Reward Models That Think (ThinkPRM) | https://arxiv.org/abs/2504.16828 | T4 |
| ProFit: High-Value Signals in SFT via Probability-Guided Token Selection | https://arxiv.org/abs/2601.09195 | T4 |
| Project Aletheia: Verifier-Guided Distillation of Backtracking for Small Language Models | https://arxiv.org/abs/2601.14290 | T4 |
| R-PRM: Reasoning-Driven Process Reward Modeling | https://arxiv.org/abs/2503.21295 | T4 |
| Reasoning Can Be Restored by Correcting a Few Decision Tokens | https://arxiv.org/abs/2605.16874 | T4 |
| ReasonXL: Shifting LLM Reasoning Language Without Sacrificing Performance | https://arxiv.org/abs/2604.12378 | T4 |
| RelayLLM: Efficient Reasoning via Collaborative Decoding | https://arxiv.org/abs/2601.05167 | T4 |
| Revisiting the Capacity Gap in Chain-of-Thought Distillation from a Practical Perspective | https://arxiv.org/abs/2604.08880 | T4 |
| RL for Reasoning by Adaptively Revealing Rationales (AdaBack) | https://arxiv.org/abs/2506.18110 | T4 |
| Search-Based Correction of Reasoning Chains for Language Models | https://arxiv.org/abs/2505.11824 | T4 |
| Seed1.5-Thinking: Advancing Superb Reasoning Models with RL | https://arxiv.org/abs/2504.13914 | T4 |
| Select to Think: Unlocking SLM Potential with Local Sufficiency | https://arxiv.org/abs/2604.26940 | T4 |
| Selective Latent Thinking: Adaptive Compression of LLM Reasoning Chains | https://arxiv.org/abs/2605.25745 | T4 |
| Self-Correction Bench: Revealing the Self-Correction Blind Spot | https://arxiv.org/abs/2507.02778 | T4 |
| Self-Taught Self-Correction for Small Language Models (STaSC) | https://arxiv.org/abs/2503.08681 | T4 |
| Shorthand for Thought: Compressing LLM Reasoning via Entropy-Guided Supertokens | https://arxiv.org/abs/2604.26355 | T4 |
| Skill-Targeted Adaptive Training (STAT) | https://arxiv.org/abs/2510.10023 | T4 |
| Skip-Thinking: Chunk-wise CoT Distillation | https://aclanthology.org/2025.emnlp-main.610/ | T4 |
| Small Models Struggle to Learn from Strong Reasoners | https://arxiv.org/abs/2502.12143 | T4 |
| Sparse but Critical: Token-Level Analysis of Distributional Shifts in RLVR | https://arxiv.org/abs/2603.22446 | T4 |
| Spotlight on Token Perception for Multimodal RL | https://arxiv.org/abs/2510.09285 | T4,MM |
| Stateful Reasoning via Insight Replay | https://arxiv.org/abs/2605.14457 | T4 |
| Staying in the Sweet Spot: Capability-Adaptive Hint Scaffolding | https://arxiv.org/abs/2509.06923 | T4 |
| Step Back to Leap Forward: Self-Backtracking for Boosting Reasoning | https://arxiv.org/abs/2502.04404 | T4 |
| Step Back: Prefix Importance Ratio Stabilizes Policy Optimization (VPPO-related) | https://arxiv.org/abs/2601.22718 | T4 |
| Structure Enables Effective Self-Localization of Errors (Thought-ICS) | https://arxiv.org/abs/2602.02416 | T4 |
| Student-in-the-Loop CoT Distillation via Generation-Time Selection | https://arxiv.org/abs/2604.02819 | T4 |
| Synthetic Error Injection Fails to Elicit Self-Correction | https://arxiv.org/abs/2512.02389 | T4 |
| Teach Small Models to Reason by Curriculum Distillation | https://aclanthology.org/2025.emnlp-main.376.pdf | T4 |
| Teacher-Student Cooperation to Synthesize Student-Consistent SFT Data | https://arxiv.org/abs/2604.14164 | T4,SD |
| Teaching Large Reasoning Models Effective Reflection | https://arxiv.org/abs/2601.12720 | T4 |
| The Conductor and the Engine: Co-Designed Reasoning (CEPO) | https://arxiv.org/abs/2509.19762 | T4 |
| The Master Key Hypothesis: Cross-Model Capability Transfer via Subspace Alignment | https://arxiv.org/abs/2604.06377 | T4 |
| The Molecular Structure of Thought: Topology of Long CoT | https://arxiv.org/abs/2601.06002 | T4 |
| Think-at-Hard: Selective Latent Iterations to Improve Reasoning LMs | https://arxiv.org/abs/2511.08577 | T4 |
| Thinking into the Future: Latent Lookahead Training for Transformers | https://arxiv.org/abs/2603.20219 | T4 |
| ThinkSLM: Towards Reasoning in Small Language Models | https://aclanthology.org/2025.emnlp-main.1659/ | T4 |
| Token-weighted Direct Preference Optimization with Attention | https://arxiv.org/abs/2605.21883 | T4 |
| Toward Consistent World Models with Multi-Token Prediction + Latent Semantic Enh. | https://arxiv.org/abs/2604.06155 | T4 |
| Towards Efficient CoT Distillation: Self-Guided Rationale Selector | https://aclanthology.org/2025.findings-emnlp.413/ | T4 |
| Training-Free Multi-Token Prediction via Embedding-Space Probing (ESP) | https://arxiv.org/abs/2603.17942 | T4 |
| Training-Trajectory-Aware Token Selection (for distillation) | https://arxiv.org/abs/2601.10348 | T4 |
| TRIM: Hybrid Inference via Targeted Stepwise Routing | https://arxiv.org/abs/2601.10245 | T4 |
| TS-PEFT: Token-Level Redundancy in Parameter-Efficient Fine-Tuning | https://arxiv.org/abs/2511.16147 | T4 |
| TwT: Thinking without Tokens by Habitual Reasoning Distillation | https://aclanthology.org/2025.findings-emnlp.894/ | T4 |
| Understanding and Enhancing the Planning Capability of LMs via Multi-Token Prediction | https://arxiv.org/abs/2509.23186 | T4 |
| Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning | https://arxiv.org/abs/2502.18001 | T4 |
| Well Begun, Half Done: RL with Prefix Optimization (PPPO) | https://arxiv.org/abs/2512.15274 | T4 |
| Where Did This Sentence Come From? Tracing Provenance in LLM Reasoning Distillation | https://arxiv.org/abs/2512.20908 | T4 |
| Where Hindsight Credit Can Reside: Signed-Capacity View of Token Updates in RLVR | https://arxiv.org/abs/2604.11056 | T4 |
| Where Reasoning Breaks: Logic-Aware Path Selection via Logical Connectives | https://arxiv.org/abs/2604.20564 | T4 |
| Which Reasoning Trajectories Teach Students to Reason Better? (Rank-Surprisal Ratio) | https://arxiv.org/abs/2601.14249 | T4 |

### 其他/未标注主题（29 篇）

| 标题 | arXiv/链接 | 主题标签 |
|---|---|---|
| Adversarial Dual On-Policy Distillation from Expressive Flow-based Teacher | https://arxiv.org/abs/2605.27095 | MM |
| AnyFlow: Any-Step Video Diffusion with On-Policy Flow Map Distillation | https://arxiv.org/abs/2605.13724 | MM |
| Aurora: When RL Meets Adaptive Speculative Training | https://arxiv.org/abs/2602.06932 | SPEC |
| CollectionLoRA: Collecting 50 Effects in 1 LoRA via Multi-Teacher OPD | https://arxiv.org/abs/2605.25378 | MM |
| Cross-Tokenizer Likelihood Scoring Algorithms for LM Distillation | https://arxiv.org/abs/2512.14954 | XT |
| Cross-Tokenizer LLM Distillation through a Byte-Level Interface | https://arxiv.org/abs/2604.07466 | XT |
| D-OPSD: On-Policy Self-Distillation for Step-Distilled Diffusion Models | https://arxiv.org/abs/2605.05204 | SD,MM |
| Data-Efficient On-Policy Distillation for ASR | https://arxiv.org/abs/2605.28139 | MM |
| DWA-KD: Dual-Space Weighting and Time-Warped Alignment for Cross-Tokenizer KD | https://arxiv.org/abs/2602.21669 | XT |
| Enhancing Cross-Tokenizer KD with Contextual Dynamical Mapping (CDM) | https://arxiv.org/abs/2502.11104 | XT |
| Flatter Tokens are More Valuable for Speculative Draft Model Training | https://arxiv.org/abs/2601.18902 | SPEC |
| Flow-OPD: On-Policy Distillation for Flow Matching Models | https://arxiv.org/abs/2605.08063 | MM |
| GDSD: RL as Guided Denoiser Self-Distillation for Diffusion LMs | https://arxiv.org/abs/2605.29398 | SD,MM |
| LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding | https://arxiv.org/abs/2602.23881 | SPEC |
| Mirror Speculative Decoding | https://arxiv.org/abs/2510.13161 | SPEC |
| Multilingual Safety Alignment via Self-Distillation | https://arxiv.org/abs/2605.02971 | SD |
| Post-Trained MoE Can Skip Half Experts via Self-Distillation | https://arxiv.org/abs/2605.18643 | SD |
| Reinforced Attention Learning | https://arxiv.org/abs/2602.04884 | MM |
| Reinforcing Human Behavior Simulation via Verbal Feedback (Ditto) | https://arxiv.org/abs/2605.20506 | SD |
| Self-Distillation as a Performance Recovery Mechanism for LLMs | https://arxiv.org/abs/2604.15794 | SD |
| Self-Trained Verification for Train- and Test-Time Self-Improvement | https://arxiv.org/abs/2605.30290 | SD |
| SGS: Scaling Self-Play with Self-Guidance | https://arxiv.org/abs/2604.20209 | SD |
| SHRED: Retain-Set-Free Unlearning via Self-Distillation with Logit Demotion | https://arxiv.org/abs/2605.07482 | SD |
| SpecBlock: Block-Iterative Speculative Decoding with Dynamic Tree Drafting | https://arxiv.org/abs/2605.07243 | SPEC |
| Speculative Speculative Decoding | https://arxiv.org/abs/2603.03251 | SPEC |
| TAD: Temporal-Aware Trajectory Self-Distillation for Diffusion LLM | https://arxiv.org/abs/2605.09536 | SD,MM |
| THINKSAFE: Self-Generated Safety Alignment for Reasoning Models | https://arxiv.org/abs/2601.23143 | SD |
| Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching | https://arxiv.org/abs/2503.20083 | XT |
| Why Fine-Tuning Encourages Hallucinations and How to Fix It | https://arxiv.org/abs/2604.15574 | SD |

## 8. 产物目录索引

本综述及其支撑材料位于 `D:/claude_code_workspace/mtp_opd/resource/research/survey/`：

- **`SURVEY_on_policy_distillation.md`** —— 本文档（最终合成综述）。
- **`analysis/*.md`** —— 117 篇逐篇深读分析（每篇 12 节 + TL;DR + 元信息），本文 §2–§5 的来源。
- **`_state/`** —— 过程与中间状态：
  - `candidates_master.md` —— 第一波候选主表（含 CodeFlag / Topic / Relevance / Verify? 标注）。
  - `P5a_candidates.md`、`P5b_candidates.md` —— 阶段性候选补充。
  - `r2_E1..E6.md`、`r2_merged.md`、`r2_codelist.md` —— 第二轮引文扩展（分片 + 合并 + 有代码清单）。
  - `r3_E1..E6.md`、`r3_merged.md`、`r3_codelist.md`、`r3_candidates_raw.json`、`r3_fetch.py`/`r3_filter.py`/`r3_score.py`、`r3_cache/` —— 第三轮引文图谱扩展（抓取/过滤/打分脚本 + 缓存）。
  - `rejected.md` —— 被剔除条目（附录 A 来源）。
  - `verified_shortlist.md` —— 核验后的深读候选短名单。
  - `_known_ids.txt`、`_known_titles.txt` —— 去重用的已知 id / 标题索引。
  - `verify/`、`wave1/` —— 早期核验与第一波材料。
  - 本次合成的中间产物：`buckets.txt`（主题分桶）、`speedtable_out.md`、`mastertable_out.md`、`appendixb_out.md`、`frontmatter.md`、`sec_intros.md`（已并入本文，可清理）。
- **`resource/repos/`**（项目根下）—— 已克隆的可审计代码仓库（如 ampo/brts/gad/gopd/opsd/rosd/scope/... 等，详见各篇 §12）。

> 文档结束。所有 〔待核〕 标记表示该断言未经一手来源证实，使用时请自行复核。

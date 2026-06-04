orz | Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model | StepFun + 清华（Jingcheng Hu 等;沈向洋 Heung-Yeung Shum） | 2025-07-05·arXiv 预印本·v2 | 主题线 L3(RLVR/GRPO·zero-RL 系统)·相关性 低(背景/系统基线)

**原始论文**:https://arxiv.org/abs/2503.24290

## 一眼看懂
- 🟦 TL;DR:不用任何花活——直接在 base 模型上跑最朴素的 vanilla PPO + GAE(\(\lambda=\gamma=1\)) + 只看答案对错的二值规则奖励、完全不加 KL 正则——就能稳定地大规模 scale up reasoning RL,性能与响应长度随训练步同步上涨且不饱和,复现了 DeepSeek-R1-Zero 的 scaling 现象;且是首个把代码/数据/各尺寸权重乃至 critic 权重全部开源的 "Reasoner-Zero" 实现。【原文】abstract+§1
- 最巧的一步:**用 PPO(带学到的 critic)而非 GRPO**(§2.2+§3.3)。抽掉 critic——GRPO 无 value network,分不清"真正答对"与"陷在重复 loop 里偶然答对"(原文 §2.2 原话 "struggles to distinguish genuinely correct responses from those occurring within negative patterns"),会误导强化、训练崩溃;而 PPO 学到的 critic 能给重复 token 赋更负的 advantage(Fig.5 定量证明),提供更稳健的 credit assignment。这是全文"为何稳"的核心机制证据。次关键:大规模数据天然降方差,使无偏的 GAE(\(\lambda=\gamma=1\))配置可行(优势简化为 \(\hat A=R-V_\phi\))。

## 为什么做
- 研究背景:o1、DeepSeek-R1-Zero 展示了"训练时间 scaling"——随算力增大,benchmark 性能与响应长度同步持续增长无饱和,并伴随 reflection/"aha"行为;R1-Zero 还证明可直接在 base 上启动 RL(跳过 SFT/蒸馏)。【原文】§1
- 解决的具体痛点:社区缺一个直接在 base 上做大规模 reasoning RL 的、全开源且稳定可扩展的实现;DeepSeek 只简述 pipeline,关键细节(数据、超参、value/advantage 估计、稳定化技巧)不公开;且对"GRPO 因无 value network 易在重复模式上误判、坍缩"缺系统分析与可复现配方。【原文】§1+§2.2
- 相关工作 & 各自不足(站谁肩上 + 精确差异):
  - **DeepSeek-R1-Zero**[2]——直接 base+RL 启动、用 GRPO,但配方黑箱、不开源关键细节;ORZ 全开源 + 改用 PPO + 去 KL + 约 1/10 步追平。
  - **DAPO**[5]——并发 base-RL 系统,匹配 ORZ 的 AIME 但用约 5× 迭代、其他基准更差(Table 1:DAPO* 在 MATH500 仅 71.8、GPQA 16.0,远逊 ORZ 92.2/55.5)。
  - **VAPO**——AIME 更强但 scale 效率低、同迭代预算只到 ORZ 约 60%(value learning 难题拖累)。
  - **Logic-RL / SimpleRL-Zoo / Understanding-R1-Zero**——pilot 配方,验证 base-RL 可行但非全开源大规模。
  - **另一线(先 SFT 蒸馏冷启动再 RL)**——在"已增强推理模型"上做 RL;ORZ 反其道直接 base 起步。
  - **RLHF 主流 / Reasoner 模型**[6,7,2]——默认保留 KL 正则;ORZ 显式去 KL(主张 KL 限制探索),是与主流相反的设计选择。【原文】§1+§4+§2.2
- 动机链:R1-Zero 现象惊人但配方黑箱 → 社区复现 GRPO 频繁崩溃 → 需要一个鲁棒、可扩展、易跟随的 base-model 大规模 RL 实现 + 把 value/advantage 估计讲清楚 → 全开源 democratize。【原文】§1
- 与最近邻工作的Δ:vs DeepSeek-R1-Zero——**全开源 + 用 PPO 而非 GRPO + 去 KL + 仅 1/10 训练步**追平/超过;vs DAPO/VAPO——更简单的算法设计、避开 value learning 难题、scale 效率更高。关键点:证明"极简即最优"(minimal reward 无 format reward → 无 reward hacking 空间)。【原文】Table 1+§4

## 怎么做 + 靠不靠谱
- 方法流水线(输入→输出):① Qwen2.5-{0.5,1.5,7,32}B base + R1-Zero 风格 prompt(要求 `<think>…</think><answer>…</answer>`)→ ② 每步 128 prompt × 每 prompt 64 response(temp/top-p=1.0)→ ③ 仅检查 `<answer>` 与参考答案精确匹配给二值 reward(1/0)→ ④ vanilla PPO + GAE(\(\lambda=\gamma=1\)),严格 on-policy(每次生成对应恰好一次 policy 更新),batch-level advantage 归一化,无 KL → ⑤ 32B 末段加 100 步 annealing(13k 难题,lr 线性衰减到 3e-7)。【原文】§2.3+§B
- 逐组件必要性(标"有无消融"):
  - **PPO over GRPO**:有定量分析(§3.3+Fig.5,PPO 对重复 token 赋更负 advantage;Fig.7 PPO 比 GRPO 稳)——核心,有据。
  - **GAE λ=1(vs 0.95)**:有消融(Fig.3 左),λ=1 reward 快增且稳,0.95 慢且长度坍缩——必要。【原文】§3.2
  - **去 KL(vs KL Loss / KL Reward Shaping)**:有消融(Fig.3 中),去 KL 最优且省显存/省调参(无需加载 reference model、不算其 log-prob)——必要。【原文】§3.2
  - **极简二值 reward(无 format reward)**:有论证(Fig.4 左,base 模型很快学会格式,Correct Fmt Ratio 快速到 1)——必要,且杜绝 reward hacking("minimal design leaves no room for reward hacking")。【原文】§2.2+§3.1
  - **scale up data(ORZ-57k vs MATH-7.5k)**:有消融(Fig.3 右),小数据集早早 plateau——必要。【原文】§3.2
  - **γ=1**:有论证(低 γ 会诱导模型过早终止生成抢即时奖励)——保留。【原文】§2.2
- 关键机制/公式(真实符号,从 PDF 抄准 + 直觉):
  - **轨迹与稀疏奖励**:每个响应 \(o_i\) 是轨迹 \(\tau_i=(s_0,a_0,\dots,s_{T_i-1},a_{T_i-1})\),\(s_t\)=prompt+已生成 token,\(a_t\)=第 \(t\) 个 token;只在末端给二值 reward \(R_i\in\{0,1\}\)(\(r_t=0\) for \(t<T_i-1\),\(r_{T_i-1}=R_i\))。
  - **GAE 一般式**(Eq.1):
    \[ \hat A^{\mathrm{GAE}(\gamma,\lambda)}_t=\sum_{k=0}^{T-t-1}(\gamma\lambda)^k\,\delta_{t+k},\qquad \delta_{t+k}=r_{t+k}+\gamma V_\phi(s_{t+k+1})-V_\phi(s_{t+k}). \]
  - **PPO 策略目标**(Eq.2,clipped surrogate;\(\epsilon=0.2\)):
    \[ J_{\mathrm{PPO}}(\theta)=\mathbb{E}_{\tau\sim\pi_{\theta_{\mathrm{old}}}}\Big[\sum_{t=0}^{T-1}\min\big(\rho_t(\theta)\hat A_t,\;\mathrm{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\big)\Big],\quad \rho_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}. \]
  - **value 目标**(Eq.3):\( J_{\mathrm{value}}(\phi)=\tfrac{1}{2}\mathbb{E}_{\tau\sim\pi_{\theta_{\mathrm{old}}}}\big[\sum_{t=0}^{T-1}(V_\phi(s_t)-V^{\mathrm{target}}_t)^2\big]\),其中 \(V^{\mathrm{target}}_t=G_t=\hat A^{\mathrm{GAE}(\gamma,\lambda)}_t+V_\phi(s_t)\)。
  - **λ=γ=1 下的关键简化**(Eq.4-5——全文"为何稳"的算法核心):
    \[ \hat A^{\mathrm{GAE}(\gamma=1,\lambda=1)}_t=R-V_\phi(s_t),\qquad J_{\mathrm{value}}(\phi)=\tfrac{1}{2}\mathbb{E}_{\tau\sim\pi_{\theta_{\mathrm{old}}}}\Big[\sum_{t=0}^{T-1}(V_\phi(s_t)-R)^2\Big], \]
    \(R\) 为单一末端 reward。直觉:这就是"无偏 Monte-Carlo 回报 \(R\) 减 critic 基线 \(V_\phi\)"——本来最高方差(\(\lambda=1\)),但大数据量自然压方差使该无偏配置可行,从而吃满长程依赖;critic 负责把"重复/退化片段"识别出来打低分,在 token 级精确扣分。【原文】Eq.(1)(2)(3)(4)(5)+§3.3
- 实验与证据:base Qwen2.5-{0.5,1.5,7,32}B,直接大规模 RL 跳过 SFT。主结果(Table 1):ORZ-32B AIME2024 **48.1** / AIME2025 36.0 / MATH500 **92.2** / GPQA Dia. **55.5**,vs DeepSeek-R1-Zero-Qwen-32B 47.0/-/91.6/55.0(且仅 ~1/10 步);Table 4 各尺寸单调上升(0.5B→32B);Table 2 泛化 MMLU 84.9 / MMLU_PRO 74.4 超 Qwen2.5-Instruct-32B;Table 3 对蒸馏模型续做(ORZ-R1-Distill-Qwen-14B)超更大的 R1-Distill-Qwen-32B。baseline 公平性:DAPO* 用作者自己 metric 在 released ckpt 上重测(较公平);但"1/10 步"对比 DeepSeek 跨实现/跨数据,严格可比性有限。【原文】Table 1-4+§3.4
- 假设与失效边界:
  - 【原文】§5 未来工作隐含其边界:当前仅数学+少量通用基准;二值 reward 排除证明题等难评测题(数据 curation 显式过滤 proof-oriented problems,并用 LLM 滤极端 pass-rate 样本保持难度平衡)。
  - 【推断】"去 KL 总更好"的结论在 base 模型起点成立,迁移到已对齐/已蒸馏起点时不一定(参见 ProRL 等保留 KL 的反向主张);"PPO>GRPO"的论证主要靠作者自身曲线 + "repeated token 的 advantage"定性/定量分析,非对所有任务/尺度的普适结论;"1/10 步"依赖与 DeepSeek 复现条件可比性(数据、prompt 不同),属同向但非严格控制比较。
- 祛魅总结:【推断】真贡献=最高开放度的 Reasoner-Zero 可复现配方(连 critic 权重都放)+ 对"为何 PPO 比 GRPO 稳"给出 critic-credit-assignment 的定量证据,这是扎实的工程与分析贡献。被适度高估的是"极简即最优"的普适性:它是"在 base 起点 + 大规模数据 + 数学/可验证答案"这一特定配方下成立的结论,换起点/换任务需重验。与本课题(探索-巩固/OPD/MTP)几乎无直接方法关联,仅作 RLVR 系统基线与"训练-时 scaling/reflection 现象"的背景参照。

## 结构化抽取
- 🎯 机制速览6轴:
  - 学什么信号:序列末端二值正确性 reward(规则匹配 `<answer>`)+ critic 学到的 token 级 value \(V_\phi(s_t)\)。
  - 改什么:policy \(\pi_\theta\) 与 critic \(V_\phi\) 两套独立网络参数(不共享权重)。
  - 何时改:每次 rollout 后做一次 on-policy PPO 更新(critic 12 mini-batch/iter)。
  - 免梯度?否——标准 PPO 梯度优化。
  - 记忆-技能生命周期:无外部记忆/技能库;推理"技能"靠 RL 直接在参数里涌现(reflection 模式自发增长,Fig.4 右)。
  - 防遗忘机制:**显式去掉 KL/reference model**(与主流相反),理由是 KL 限制探索;不靠 KL 防遗忘,靠大数据+critic 稳定。【原文】§2.2
- ⑦ 开源代码+框架/harness:github.com/Open-Reasoner-Zero/Open-Reasoner-Zero(本地已克隆 ~92MB,含 `orz/` 包、`playground/` 各尺寸训练脚本如 `orz_32b_ppo.py`、`docker/`、`ORZ_paper.pdf`)。框架 **OpenRLHF + vLLM + DeepSpeed + Ray**(实现 vanilla PPO+GAE 的大规模分布式训练)。HF 放 ORZ-{0.5,1.5,7,32}B + ORZ-R1-Distill-Qwen-14B + critic 权重 + ORZ 数据。代码可得性:完整(同类工作中最高,含 critic 权重)。【原文/仓库】
- 💰 资源/成本与可扩展性:每步 128 prompt×64 response;policy lr 1e-6 / critic lr 5e-6(AdamW,β=[0.9,0.95],无 weight decay,constant+50 步 warmup);32B + annealing 100 步(13k 难题,lr→3e-7);StepFun 提供算力(具体 GPU 数/总卡时论文未明说)。核心卖点即"~1/10 训练步追平 R1-Zero"。【原文】§B
- 🎯 对"探索-巩固"对标:**竞品/背景,非直接组件**。一句判定:ORZ 走的是"纯结果奖励 RLVR + 去 KL 鼓励探索"路线,与本课题"teacher 稀疏脚手架 + on-policy 蒸馏稠密信号"是对立思路(它恰恰主张去掉 reference/KL),只能作为"探索由 RL 自发驱动"的对照与系统基线。依据:§2.2 去 KL 鼓励探索。可借组件:① critic 对"重复/退化片段"的 token 级 devalue(Fig.5)——这与"识别走偏/需回轨的关键步"思路相通,可借来定位 path-recovery 的接管点;② annealing 阶段挖"64 次只对<4 次"的难题做课程——可用于"探索-巩固"的难度调度。缺口:无任何蒸馏/特权信息/MTP,纯 outcome 信号、无过程级巩固。
- 🔭 开放问题/未来方向:
  - 【原文】§5:Data/Model/Test-Time/Scenario 四类 scaling;test-time 提到 multi-turn、value model 评估推理轨迹、multi-agent——可通向 agentic。
  - 【推断】把 ORZ 的"learned critic 定位退化/重复"显式接到 path-recovery(在 critic 打低分的 token 处触发回轨/重选分支),可能把纯 outcome RL 与过程级巩固桥接;"去 KL"结论需在已蒸馏/已对齐起点重验(本课题多从 instruct 模型起步,可能反而需要保留 reference 锚点)。

— RETURN —
orz | 读PDF? 是(21页,§1-§3+Table1-4/Fig2-5 全核,Eq.1/2/3/4/5 逐字核对) | 加厚? 是(GAE Eq.1 + PPO Eq.2 + value Eq.3 + λ=γ=1 简化 Eq.4/5 全列 + 相关工作扩到 7 条精确对位 + 轨迹/稀疏奖励形式化) | LaTeX公式条数:5(GAE Eq.1 / PPO Eq.2 / value Eq.3 / 简化优势Eq.4 / 简化value Eq.5) | 待核数:0

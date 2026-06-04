gigpo | Group-in-Group Policy Optimization for LLM Agent Training (GiGPO) | 南洋理工 NTU / Skywork AI Singapore（Lang Feng、Zhenghai Xue、Tingcong Liu、Bo An） | 2025-10-28 (arXiv:2505.10978 v3，cs.LG；NeurIPS 2025) | L4 Agent/工具/多轮自进化 · 相关性中（兼 L3 GRPO 家族）

**原始论文**：https://arxiv.org/abs/2505.10978

## 一眼看懂
- 🟦 TL;DR：GRPO 在单轮数学/代码上很强（即时奖励、信用分配简单、critic-free），但搬到长 horizon 多轮 agent（如 ALFWorld 一集可达 50 步、20k+ token、奖励稀疏/延迟）就会"把整条轨迹当一个响应、组内每步给同一 advantage"，丧失 step 级区分。GiGPO 嵌套两级相对优势：宏观按整条轨迹总回报算 episode 级 \(A_E\)（同 GRPO），微观利用"组内轨迹常反复经过相同环境状态"这一观察，把同一状态下的不同动作聚成 step 级组算 \(A_S\)；二者线性合并 \(A=A_E+\omega\cdot A_S\)。**无额外 rollout、无 critic、同显存**，额外时间 <0.002%【原文 Abstract, §4, §5.6】。
- 最巧的一步：**anchor state grouping**（§4.2）——回溯识别组内跨轨迹/跨时间重复出现的环境状态当 anchor，用 hashmap 把"在同一 anchor state 下的不同动作"聚成 step 级组，纯离线、零额外前向。抽掉它（即去掉 \(A_S\)、只剩 \(A_E\)）就退回 vanilla GRPO，论文 §5.4 消融显示在 Cool/Pick2/WebShop 等复杂任务上显著掉点。所以"用天然重复状态当免费对照实验"是命门——它把"step 级信用"从需额外 rollout（中间方案，代价爆炸）变成几乎免费【原文 §4.2, Fig.1, §5.4】。

## 为什么做
- 研究背景：LLM agent 从静态问答跃到多轮感知-推理-行动（具身助手、web 导航、游戏）。group-based RL（RLOO/GRPO）用组内相对优势替代 value 网络，低显存、critic-free、可扩展，但成功主要限于单轮（奖励即时、信用简单）【原文 §1, §2】。
- 解决的具体痛点：agent 是长 episode（数十步、数万 token）、奖励稀疏/延迟，单个动作好坏可能很晚才显现。朴素套 group-based RL（如 RAGEN 把整条轨迹拼成 episode 级响应）丧失 step 级区分，长 horizon 扩展性差【原文 §1, §2, §4】。
- 相关工作 & 各自不足（来龙去脉 + 精确短板）：
  - ① **actor-critic agent RL**：PPO（需 value 网络）、ArCHer（分层 critic）、AgentQ（MCTS+DPO）——需额外 value/搜索，复杂、开销大，失去 group-based RL 的简洁与低显存。
  - ② **naive step 级信用**（对每个 state 额外 rollout 新动作再比较）——计算代价爆炸（Fig.1 middle），horizon 越长越不可行。
  - ③ **trajectory-level GRPO**：RAGEN（把整条交互历史 state/reasoning/action 拼成一个 episode 级响应、给统一 advantage）——长 horizon（ALFWorld 50 步）扩展性差、无 step 区分；这是 GiGPO 最直接的最近邻。
  - ④ **LOOP（RLOO+PPO 混合）**——agent RL SOTA 之一，但仍带 PPO 式更新与 value 估计。
  - ⑤ **search-augmented QA 的多轮 RL**：Search-R1、ZeroSearch、StepSearch——只优化轨迹级或简单 step 信号，GiGPO 在其上叠 anchor-state step 信用。
  - 共同不足：要么放弃 critic-free 简洁（①②），要么没有细粒度 step 信用（③④⑤）【原文 §2】。
- 动机链：长 horizon 里 episode 级单一 advantage 无法告诉模型"是哪一步走对/走错" → 需 step 级相对信号 → 但额外 rollout 太贵、critic 太重 → 关键观察：组内轨迹因无效动作/循环反复撞到相同状态 → 这些共享状态天然是"对照实验"，不用额外 rollout 就能比较动作优劣 → anchor state grouping。
- 与最近邻工作的Δ（精确差异）：最近邻是 RAGEN（trajectory-level GRPO）与各 actor-critic agent RL。Δ：GiGPO 在保持 GRPO 的 critic-free/低显存/同 rollout 预算下，**额外引入 step 级信用**，且核心是"利用已有 rollout 里的重复状态"而非额外采样或 value 网络。为什么有用：把细粒度信用的成本降到 <0.002%，且 hierarchical core 与现有 group-based RL 正交、可叠加【原文 §1, §2, §5.6】。

## 怎么做（细到可复现）
### 0. 数据采集（§4.1）
同任务 \(x\)、同初始状态下用 \(\pi_{\theta_{\text{old}}}\) 采 \(N\) 条完整轨迹 \(\{\tau_i\}\)，\(\tau_i=\{(s^{(i)}_1,a^{(i)}_1,r^{(i)}_1),\dots,(s^{(i)}_T,a^{(i)}_T,r^{(i)}_T)\}\)，初始状态全相同 \(s^{(1)}_1=\dots=s^{(N)}_1\)。每步动作格式 `<think>…</think> <action>…</action>`。

### 1. Episode 级（宏观，§4.1）
- 总回报 \(R(\tau_i)=\sum_t r^{(i)}_t\)（二值终局奖励时即 0/1）。组成 episode 组 \(G_E=\{(\tau_i,R(\tau_i))\}_{i=1}^N\)（Eq.2）。
- **episode 相对优势**（Eq.3）：
\(\displaystyle A_E(\tau_i)=\frac{R(\tau_i)-\mathrm{mean}\{R(\tau_j)\}_{j=1}^N}{F_{\text{norm}}\{R(\tau_j)\}_{j=1}^N}.\)
\(F_{\text{norm}}=\text{std}\)（默认，同 GRPO）或 \(F_{\text{norm}}=1\)（给无偏 Leave-One-Out 估计、缓解 difficulty bias，难任务更稳）。

### 2. Step 级（微观，anchor state grouping，§4.2）
- 令 \(\mathcal{U}=\{\tilde s_1,\dots,\tilde s_U\}\) 为轨迹组里出现的所有**不同**环境状态。每个 \(\tilde s\) 当 anchor，用 hashmap 把所有"状态恰为 \(\tilde s\)"的 (动作,奖励) 聚成 step 组（Eq.4）：
\(\displaystyle G_S(\tilde s)=\big\{(a^{(i)}_t,r^{(i)}_t)\ \big|\ s^{(i)}_t=\tilde s,\ 1\le i\le N,\ 1\le t\le T\big\}.\)
**零额外 rollout、纯离线、只做 hashmap key 聚合。**
- **折扣回报**（Eq.5，把稀疏即时奖励变成捕捉长期影响的信号）：\(R^{(i)}_t=\sum_{k=t}^{T}\gamma^{k-t}r^{(i)}_k\)，\(\gamma\in(0,1]\)；组更新为 \(G_S(\tilde s)=\{(a^{(i)}_t,R^{(i)}_t)\mid s^{(i)}_t=\tilde s\}\)（Eq.6）。
- **step 相对优势**（Eq.7，对同 anchor 状态下的动作组内归一）：
\(\displaystyle A_S(a^{(i)}_t)=\frac{R^{(i)}_t-\mathrm{mean}\{R^{(j)}_t\mid (a^{(j)}_t,R^{(j)}_t)\in G_S(\tilde s)\}}{F_{\text{norm}}\{R^{(j)}_t\mid \cdots\in G_S(\tilde s)\}}.\)
- **直觉（Fig.3，WebShop）**：同一搜索结果页(anchor)下，τ1 先点"2nd Item"(错)→返回→点"1st Item"(对)成功，τ2 点"Next Page"失败；因时间折扣，早期次优动作折扣回报更低，于是组内得到清晰排序 \(A_S(\text{1st Item})>A_S(\text{2nd Item})>A_S(\text{Next Page})\)——这正是 episode 级单一 advantage 给不出的细粒度信用。

### 3. 合并 + 优化目标（§4.3）
- **group-in-group 优势**（Eq.8）：\(A(a^{(i)}_t)=A_E(\tau_i)+\omega\cdot A_S(a^{(i)}_t)\)，\(\omega\ge0\) 平衡两级（默认 **\(\omega=1\) 不调**）。
- **clipped 目标**（Eq.9）：
\(\displaystyle J_{\text{GiGPO}}(\theta)=\mathbb{E}\Big[\tfrac{1}{NT}\textstyle\sum_{i,t}\min\big(\rho_\theta(a^{(i)}_t)A(a^{(i)}_t),\ \mathrm{clip}(\rho_\theta(a^{(i)}_t),1\pm\epsilon)A(a^{(i)}_t)\big)\Big]-\beta D_{\text{KL}}\big(\pi_\theta(\cdot\mid x)\Vert\pi_{\text{ref}}(\cdot\mid x)\big),\)
其中 \(\rho_\theta(a^{(i)}_t)=\dfrac{\pi_\theta(a^{(i)}_t\mid s^{(i)}_t,x)}{\pi_{\theta_{\text{old}}}(a^{(i)}_t\mid s^{(i)}_t,x)}\) 为重要性比，\(\beta\) 控 KL 向 ref 正则。
- **similarity-based 变体**：状态难精确匹配时（如 QA），用最长匹配子序列相似度 >0.9 判同状态。

### 数据流动小结
采 \(N\) 轨迹 → 算各轨迹总回报→ Eq.3 得 \(A_E\)；同时按状态 key hashmap 聚组 → Eq.5 折扣回报 → Eq.7 得 \(A_S\)；Eq.8 合并 → 代入 Eq.9 clipped PG 更新策略全参（无 critic）。所有 step 级计算都是对**已执行**动作的事后离线聚合。

### 逐组件必要性
- **\(A_E\)**：去掉（w/o \(A_E\)）→ 全任务大幅掉点，失去 trajectory-wide 稳定信号【§5.4 Fig.4】。
- **\(A_S\)（anchor grouping 核心）**：去掉（w/o \(A_S\)）→ 退回 GRPO，在 Cool/Pick2/WebShop 等显著掉点【§5.4 Fig.4】。
- **折扣 \(\gamma\)（Eq.5）**：使早期次优动作得更低折扣回报、产生清晰偏好排序（WebShop 例）。未见单独 \(\gamma\) 消融【§4.2 Fig.3】。
- **\(F_{\text{norm}}\)（std vs 1）**：w/ vs w/o std 差距远小于结构消融（\(A_E/A_S\)）——增益主要来自两级合并而非归一化选择；\(F_{\text{norm}}=1\) 在难场景更稳但非普适更优【§4.1, §5.2 末, §5.4】。
- **\(\omega\)**：直接设 1、未做敏感性扫描【§5.1】。

## 靠不靠谱
- 实验与证据：
  - **基准/模型**：ALFWorld（3827 任务×6 类家务）、WebShop（1.1M 商品、12k 指令）；search-augmented QA（单跳 NQ/TriviaQA/PopQA + 多跳 HotpotQA/2Wiki/MuSiQue/Bamboogle，沿 Search-R1，E5 检索，max turn=4）；base Qwen2.5-1.5B/3B/7B-Instruct【§5.1】。
  - **关键数字（Table 1）**：闭源 Gemini-2.5-Pro 仅 60.3%(ALFWorld)/35.9%(WebShop)、GPT-4o 48.0/23.7；GiGPO w/o std 较 GRPO：1.5B 上 ALFWorld 86.1 vs 72.8（**+13.3%**）、WebShop succ. 67.4 vs 56.8（**+10.6%**）；7B 上 90.2 vs 77.6（**+12.6%**）、75.2 vs 66.1（**+9.1%**）；QA（Table 2）3B **42.1%**、7B **47.2%**（超 Search-R1/StepSearch）。w/ 与 w/o std 均稳超 GRPO 与 RLOO【§5.2, Table 1/2】。
  - **成本（Fig.6，每 iter 时间）**：anchor grouping（hashmap）**0.01s**、step-adv（算术）**0.53s**，vs 共享操作 362.83s/iter（rollout 221.97 + old/ref prob 24.99+25.96 + update 89.86），GiGPO 特有部分占 **<0.002%**，同显存、同 rollout【§5.6, Fig.6】。
  - **step-group 动态（Fig.5）**：size=1 的组全程 <35%（即 >65% 状态跨轨迹重复）；早期(iter 10) size≥10 占 >20%（循环/无效动作多），iter 75 时 10≤size<50 从 16.2%→12.1%、size≥50 从 5.6%→3.1%，iter 140 收敛到紧凑分布，与成功率 plateau(>80%) 吻合——直接证明"状态重复"前提在这些任务上成立且随训练演化【§5.5】。
  - **tool 效率（§5.3）**：7B QA 单跳平均 ~0.9 次调用、多跳 ~1.6 次（匹配 OTC 的 ~1.0/~1.7），说明能抑制冗余查询。
  - **baseline 公平吗**：公平——ALFWorld/WebShop 所有 RL 方法用完全相同超参；对照覆盖闭源(GPT-4o/Gemini-2.5-Pro)、prompting(ReAct/Reflexion)、actor-critic(PPO)、group-based(RLOO/GRPO)；QA 另比 Search-R1/ZeroSearch/StepSearch。结果对 3 随机种子平均。
  - **"看着强但没回答核心问题"**：增益稳健且近零成本，但本质是 GRPO 家族的 step 级信用增量，无外部监督/蒸馏；emergent reasoning 仅附录 F 提及，未深究。
- 假设与失效边界：
  - 【原文】核心前提："组内状态确实重复"。极端无状态重复（\(A_S=0\)）→ **自然退化为 GRPO**，保留其有效性与稳定性（强 lower bound）【§6】。
  - 【原文】高复杂环境中相同状态因噪声/细微差异难检测，是明确 limitation；similarity-based grouping 部分缓解但引入相似度阈值这一新超参【§6, §5.1】。
  - 【原文】\(F_{\text{norm}}\) 是任务相关：难任务（Look/Pick2/WebShop）std 会放大难样本梯度、损稳定，\(F_{\text{norm}}=1\) 更好；其他任务两者相近【§5.2 末】。
  - 【推断】状态空间大、轨迹很少撞同一状态时，step 组多为 size=1（Fig.5），\(A_S\) 退化、增益缩水（依据：§5.5 size 分布 + §6 状态匹配 limitation）。
  - 【推断】\(\omega=1\) 未调参虽显鲁棒，但两级权重未被优化，可能非最优；similarity 阈值 0.9 与 \(\gamma\) 缺系统敏感性分析（依据：§5.1 固定、无扫描）。
- 祛魅总结：
  - 真贡献【推断】：用"已有 rollout 里的重复状态当免费对照"这一简洁洞见，在 critic-free/同显存/<0.002% 成本下实现 step 级信用，且给出"无重复则退化为 GRPO"的安全下界——工程上很实用、机制可核（代码已审计与 Eq.3/6/7/8 对应）。
  - 包装/高估【推断】：是 GRPO 家族的增量改进而非新范式；"超过 Gemini-2.5-Pro/GPT-4o"部分是因为闭源模型在这些专门 agent 任务上本就 zero-shot 偏弱，非通用能力超越。
  - 低估【推断】：Fig.5 的"step-group 分布随训练从大组收敛到 size~N"其实是一个很有信息量的"策略成熟度"诊断信号，可独立用于监控 agent 训练，但论文只当辅助分析。

## 结构化抽取
- 🎯 机制速览6轴：**学什么信号**=两级相对优势——\(A_E\)(整条轨迹总回报的组内相对值) + \(A_S\)(同 anchor state 下动作的折扣回报组内相对值)，纯 outcome/环境奖励驱动｜**改什么**=策略全参（critic-free，无 value 网络）｜**何时改**=训练期在线 RL，每 iter 采 \(N\) 轨迹后回溯聚组算优势再更新｜**免梯度?**=否（clipped policy gradient，Eq.9）｜**记忆-技能生命周期**=框架支持 per-step history/memory 模块管理（verl-agent 的 step-independent rollout），但 GiGPO 算法本身不固化技能/记忆，知识进策略参数｜**防遗忘机制**=无专门防遗忘（KL 向 ref 正则提供基本稳定）。
- ⑦ 开源代码+框架/harness：github.com/langfengQ/verl-agent（Apache-2.0，含 HF 模型 collection；本地已 clone）。核心算法 `gigpo/core_gigpo.py`（已审计：`compute_gigpo_outcome_advantage` 中 `scores = episode_advantages + step_advantage_w * step_advantages`，step_advantage_w 默认 1.0；anchor 分组、折扣回报、similarity 变体均实现，与 Eq.7/8 一致）。框架=**verl-agent**（**veRL** 扩展，pyproject 包名仍为 verl），核心改造是 **step-independent multi-turn rollout**——不简单拼接完整交互历史，而是允许逐步可定制的 per-step 输入结构 + history 管理 + memory 模块，支持超长 horizon（ALFWorld 可达 50 步）。
- 💰 资源/成本与可扩展性：与 GRPO **同 GPU 显存、同 rollout 预算**；GiGPO 特有操作 <0.002% 总时间（anchor 0.01s + step-adv 0.53s vs 362.83s/iter）。base Qwen2.5-1.5B/3B/7B；ALFWorld/WebShop 所有 RL 方法同超参、N=8；QA N=5、max turn=4、E5 检索；ω=1 不调。结果对 3 随机种子平均【原文 §5.1, §5.6, Table 1】。
- 🎯 对"探索-巩固"对标：**竞品 / 可借组件（信用分配侧）**。一句判定：GiGPO 是"path/step 级信用分配"的 critic-free 强 baseline，其 anchor state grouping 给"探索-巩固"提供一个可借的**免梯度 step 级信用**组件——可用作 path-recovery 监督的对照式替代（同状态下"走偏动作 vs 恢复动作"天然成组，\(A_S\) 自动给出偏好排序，正对应"走偏后自选恢复分支"的相对评价）。但 GiGPO 是纯 RL：无 teacher 蒸馏、无 MTP 前瞻、无技能/记忆固化，增量集中在信用机制本身。缺口：① 无"teacher 当稀疏脚手架"——它靠组内自对照而非外部示范；② 无前瞻（MTP），\(A_S\) 仅事后比较已执行动作；③ 不解决"探索到的好行为如何固化进参数/记忆而不遗忘"。依据：§6 纯 outcome 驱动、退化为 GRPO 的下界，无外部监督。
- 🔭 开放问题/未来方向：【原文】更鲁棒的状态匹配（embedding 表示、domain-specific 结构等价）以应对复杂环境状态难精确匹配的 limitation（§6）；emergent reasoning 现象（附录 F）未深究。【推断】把 anchor-state 内"恢复动作 vs 走偏动作"的 \(A_S\) 排序与 teacher 的 path-recovery 示范结合，可在"自对照信用"上叠加"稀疏脚手架"，更贴"巩固/回轨"；引入 MTP 前瞻作 anchor 内动作的提前评估，减少对实际重复状态的依赖（依据：Fig.3 同状态动作排序 + 本课题 MTP 前瞻探针）。

读到PDF? 是（PyMuPDF 全文 27 页；正文 §1–6 + Eq.2–9 + Table 1/2 + Fig.3-6 消融/成本/动态全核，附录未逐页核）｜L线 L4（多轮 agent 训练；兼 L3 GRPO 家族 step 信用）｜对标结论 竞品+可借组件（anchor-state 免梯度 step 信用↔path-recovery 对照式监督）；缺 teacher 蒸馏/MTP/记忆固化｜残留待核 0

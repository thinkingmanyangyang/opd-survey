denoiserl | DenoiseRL: Bootstrapping Reasoning Models to Recover from Noisy Prefixes | 复旦大学 / 上海创智学院(Caijun Xu, Changyi Xiao, Zhongyuan Peng, Yixin Cao) | arXiv 2605.28421(v1 2026-05-27)·预印本 | 主题线 L1(自蒸馏/前缀注入)+ L3(GRPO/DAPO)·相关性 高(与本项目核心 path-recovery 最近邻)

**原始论文**:https://arxiv.org/abs/2605.28421

## 一眼看懂
- 🟦 TL;DR:把**弱模型生成的错误推理前缀**当作"结构化噪声"注入策略的 rollout,用 RL 训练策略从错误中间状态"去噪并恢复"到正确答案——从而在不引入更强 teacher、不构造难数据的前提下,把"自纠错/从错误中恢复"从涌现行为提升为**显式训练目标**。在 Qwen3-4B/8B-Base 上一致改进 GRPO/DAPO 基线(4B 平均 39.6→42.0,8B-DAPO 42.8→44.8)【原文 Abstract, §1, Table 1】。
- 最巧的一步:**只对策略自己生成的续写部分算梯度、把 off-policy 错误前缀 mask 掉**。抽掉它(也更新前缀),训练直接崩——§4.4 Fig.5 实证:更新 off-policy 前缀时验证精度 step 80 峰值 34.7% 后急降,step 400 在所有 benchmark 塌到 0。为什么:offline 噪声前缀在当前策略下与产生它的行为策略 log-prob 分布差异巨大,PPO 重要性比作用到这些重 off-policy token 上注入高方差噪声梯度,摧毁推理质量与长度控制【原文 §4.4, Fig.5】。

## 为什么做
- 研究背景:RL(GRPO/DAPO)已成 LLM 推理 post-training 主范式,但 SOTA 系统常依赖更强模型的监督/指导;当没有足够强的现成 teacher 时,能力提升越来越难,引出根本问题:**如何在不依赖更强模型作监督者的前提下获得强模型?**【原文 §1】。
- 解决的具体痛点:①W2SG(weak-to-strong)用弱模型监督强学生,但学生天花板被弱监督者的噪声与有限能力卡住(被优化去模仿伪标签);②难数据构造(难题合成/对抗样本/长轨迹)依赖复杂流水线、过滤验证与大量人工;③标准 on-policy RL 受限于自生成状态分布——策略一旦饱和就多产正确 rollout 或狭窄失败模式,**信息量大的失败样本太稀缺**,难提供有意义梯度【原文 §1, §2】。
- 相关工作 & 各自不足(三条并行路线 + 精确差异,DenoiseRL 站谁肩上):
  - **路线 ①——weak-to-strong generalization(W2SG)**【原文 §2】:用弱模型当 imperfect oracle 监督强学生(学生学伪标签)。**精确短板**:学生天花板被弱监督者噪声/能力封顶。DenoiseRL 的反转:**不把弱模型当 oracle,而当"扰动器"**——弱模型只负责产生错误中间态,不提供学习目标。
  - **路线 ②——prefix/trajectory 注入类**:
    - **LUFFY [36]**:把 off-policy 推理轨迹**混入** on-policy RL(整条轨迹参与优化)。DenoiseRL 差异:**只更新 on-policy 续写、前缀 mask 不优化**(规避重 off-policy 不稳定)。
    - **PrefixRL [25]**:在**成功**前缀上条件化并优化续写(让稀疏奖励更可达)。DenoiseRL 差异:**注入弱模型错误前缀**(失败状态),反转为"从死胡同恢复"而非"从好起点起飞"。
    - **更广的 expert-solution / oracle-hint / 成功轨迹类 [14,23,29]**:都用"特权信息/示范"降低稀疏奖励难度。DenoiseRL 差异:错误前缀**既非示范也非特权提示**,而是"误导推理状态"让策略从中恢复。
  - **路线 ③——去噪自编码视角(BART / denoising autoencoder)**:DenoiseRL 把推理 RL **重述为去噪问题**(错误前缀=结构化噪声,策略学重建有效解路径),把自纠错从涌现行为升为显式训练目标(§1, §3.1)。这是它独有的概念框架。
- 动机链:现状(推理 RL 依赖强 teacher/难数据)→缺陷(W2SG 受弱 teacher 封顶、难数据贵、标准 RL 失败样本稀缺)→所以必须(统一 W2SG 与难度驱动合成:不让弱模型合成难数据/给学习信号,而把它当**结构化扰动生成器**,在不产生任何新数据下自动抬高难度;并把推理 RL 重述为去噪问题)【原文 §1, §3.1】。
- 与最近邻工作的Δ:相对 PrefixRL(注入成功前缀)——**DenoiseRL 反转为注入弱模型错误前缀**,控制起始状态、强迫从损坏中间态恢复;相对 LUFFY(混 off-policy 轨迹)——DenoiseRL 只更新 on-policy 续写、前缀 mask。关键有用点:前缀对后续轨迹有不成比例影响,注入错误前缀既极大扩展失败模式状态多样性(暴露标准 on-policy RL 罕见的 off-policy 语境),又直接强化"从错误恢复"这一被低估的能力。

## 怎么做 + 靠不靠谱

### 0. 任务定义:denoising reasoning(§3.1)
把弱模型的错误中间推理态当**结构化噪声**:一个错误部分解被**前置(prepend)**到策略生成之前,策略被训练从这个损坏态续写到正确答案。错误前缀同时扮演两角色:weak-to-strong 扰动信号 + 低成本难度提升器。

### 1. 离线噪声采集(一次性预处理,§3.1)
弱模型 \(\pi_w\)(Qwen2.5-1.5B-Instruct)对训练集 \(\mathcal D\)(MATH-7.5K)每题采 \(M=8\) 次,verifier 判错的轨迹构成**固定池** \(\{\mathcal W(q)\}_{q\in\mathcal D}\)——训练全程固定、**每步零额外成本**。**退化处理**:若某题 8 次都无格式良好的错误答案(\(\mathcal W(q)=\varnothing\)),则其 denoise 槽用额外 main rollout 顶替(保证 batch 形状不变)。

### 2. 每步双类 rollout(§3.2,Eq.1-3)
对每题 \(q\sim\mathcal D\):
- **Main rollouts(\(N=12\)/题)**:标准 on-policy
\(\displaystyle y\sim\pi_\theta(\cdot\mid q) \tag{1}\)
- **Denoise rollouts(\(K=4\)/题)**:抽错误解 \(w\sim\mathcal W(q)\),按**固定前缀比** \(\rho\in(0,1]\) 保留其前 \(p\) 个 token 作 assistant 前缀
\(\displaystyle p=\max\big(1,\ \lceil \rho\,|w|\rceil\big)\quad(\rho=0.2) \tag{2}\)
策略从这个 **off-policy 前缀**续写:
\(\displaystyle y_{>p}\sim\pi_\theta(\cdot\mid q,\,w_{1:p}) \tag{3}\)

### 3. 预算折叠(length-fair,§3.2,Eq.4)
两类 rollout 共享同宽响应窗 \(R=4096\)。因前缀已占 \(p\) token,denoise rollout 被折叠成可见响应
\(\displaystyle \tilde y=\big(\underbrace{w_{1:p}}_{\text{prefix}},\ \underbrace{y_{p+1:p+L}}_{\text{continuation}}\big),\qquad p+L\le R,\qquad L=\min(T_{y>p},\,R-p) \tag{4}\)
超出 length-fair 预算的尾部 token 被丢弃。verifier 对**完整折叠响应** \(\tilde y\) 打 0/1 奖励 \(r(\tilde y;q)\)(看是否在条件化错误前缀后到达正确答案)。**训练时只更新 on-policy 续写段 \(y_{p+1:p+L}\)**——这是防崩溃的核心设计。

### 4. token 级 GRPO 目标(§3.2,Eq.5-7)
- **共享 baseline**:同题 \(N+K\) 条 rollout 共用一个 advantage baseline(\(\mathcal G(q)=\{1,\dots,N+K\}\),终端奖励 \(r_i\in\{0,1\}\)):
\(\displaystyle A_i=\frac{r_i-\mu_q}{\sigma_q+\varepsilon},\quad \mu_q=\frac{1}{N+K}\sum_{j\in\mathcal G(q)}r_j,\quad \sigma_q^2=\frac{1}{N+K}\sum_{j\in\mathcal G(q)}(r_j-\mu_q)^2 \tag{5}\)
**直觉(关键)**:denoise rollout 对易题提供**负样本**(错误前缀拉低成功率),使该题的正样本携带有效学习信号——否则易题组内全对、advantage 归零无梯度。
- **token 级重要性比**(context 对 main 是 \(q\)、对 denoise 是 \((q,w_{1:p_i})\)):
\(\displaystyle r_{i,t}(\theta)=\frac{\pi_\theta(y_{i,t}\mid c_{i,t},y_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid c_{i,t},y_{i,<t})} \tag{6}\)
- **clip 代理目标**(\(\varepsilon_{\mathrm{low}}=\varepsilon_{\mathrm{high}}=0.2\)):
\(\displaystyle \mathcal L_i^{\mathrm{PPO}}(\theta)=\frac{1}{|\mathcal T_i|}\sum_{t\in\mathcal T_i}\min\Big(r_{i,t}(\theta)\hat A_{i,t},\ \mathrm{clip}\big(r_{i,t}(\theta),1-\varepsilon_{\mathrm{low}},1+\varepsilon_{\mathrm{high}}\big)\hat A_{i,t}\Big) \tag{7}\)
（\(\mathcal T_i\) 对 denoise rollout 只含续写段索引——前缀不进 \(\mathcal T_i\),即 mask。）

### 5. 联合目标(§3.2,Eq.8-11)
两条 rollout 分布的混合期望(\(\pi_\theta^{\mathrm{main}}(\cdot|q)=\pi_\theta(\cdot|q)\),\(\pi_\theta^{\mathrm{denoise}}(\cdot|q,w)=\pi_\theta(\cdot|q,w_{1:p})\)):
\(\displaystyle \mathcal J(\theta)=\frac{N}{N+K}\,\mathcal J_{\mathrm{main}}(\theta)+\frac{K}{N+K}\,\mathcal J_{\mathrm{denoise}}(\theta) \tag{8}\)
\(\displaystyle \mathcal J_{\mathrm{main}}(\theta)=\mathbb E_{q\sim\mathcal D,\;y\sim\pi^{\mathrm{main}}_{\theta_{\mathrm{old}}}}\big[\mathcal L^{\mathrm{PPO}}(\theta;q,y)\big],\qquad \mathcal J_{\mathrm{denoise}}(\theta)=\mathbb E_{q\sim\mathcal D,\,w\sim\mathcal W(q),\;y\sim\pi^{\mathrm{denoise}}_{\theta_{\mathrm{old}}}}\big[\mathcal L^{\mathrm{PPO}}(\theta;q,w_{1:p},y)\big] \tag{9,10}\)
实际每步优化的 Monte-Carlo 估计(\(B\)=每 batch 题数,\(\mathcal M(q_b)\)=\(N\) main,\(\mathcal S(q_b)\)=\(K\) denoise):
\(\displaystyle \hat{\mathcal J}(\theta)=\frac{1}{B(N+K)}\sum_{b=1}^{B}\Big[\sum_{i\in\mathcal M(q_b)}\mathcal L_i^{\mathrm{PPO}}(\theta)+\sum_{i\in\mathcal S(q_b)}\mathcal L_i^{\mathrm{PPO}}(\theta)\Big] \tag{11}\)
即:两类 rollout 按 \(N/(N+K)\) 与 \(K/(N+K)\) 加权、共享同题单一 baseline、且**只在策略自己生成的 token 上更新**。

### 逐组件必要性(消融较齐全)
- **denoise rollout(注入错误前缀)**:核心。Table 1 两尺度两 backbone 全部 ≥ 基线。没它=退回标准 GRPO/DAPO【原文 §4.2 Table 1】。
- **只更新 on-policy 续写(前缀 mask)**:**有硬消融(§4.4 Fig.5)**——不 mask 则 step 400 塌到 0。没它=训练崩溃【原文 §4.4】。
- **length-fair 折叠(\(p+L\le R\))**:**有消融(Table 2)**——不设上限 42.0→40.2(降 1.8 点),不公平的长恢复预算鼓励冗长但不可靠推理【原文 §4.5 Table 2】。
- **噪声强度 \(\rho\)**:**有扫描(§4.3 Fig.2/3)**——\(\rho\in\{0.2,0.5,0.8\}\);\(\rho=0.2\) 长度紧凑(末 100 步均 1.38K),\(\rho=0.8\) 升至 2.26K 并触顶,\(\rho=0.5\) 反而最不稳(3.87K);\(\rho\) 越大 overthinking 越重(无休止自我怀疑/反复验证)。\(\rho=0.2\) 为最佳权衡【原文 §4.3】。
- **denoise rollout 数 \(K\)**:**有扫描(§4.3 Fig.4)**——\(K\in\{1,4,8\}\) 固定 \(\rho=0.2\);\(K=1\) 信号太稀(+14.9%),\(K=8\) 占半数 rollout 分散主目标(+11.9%),**\(K=4\) 最佳(+16.3%,AIME24 +16.5/AIME25 +16.9)**【原文 §4.3】。

### 实验与证据 / 靠不靠谱
- 配置:弱模型 Qwen2.5-1.5B-Instruct,策略 Qwen3-4B-Base / Qwen3-8B-Base(仓库另含 1.7B 脚本但正文主表只报 4B/8B),训练 MATH-7.5K;\(N=12\)+\(K=4\)、\(R=4096\)、\(\rho=0.2\)、prompt batch 16、lr 1e-6、**无 KL/无 length loss**、PPO clip 0.2、训练 temperature 1.0/top-p 1.0;评测 MATH500/AMC23/AIME2024/AIME2025/BBEH(AMC23/AIME24/AIME25 报 AVG@16,MATH500/BBEH 报 AVG@1),验证解码 temperature 0.6/top-p 0.95。
- 主表(Table 1 平均):4B Base 26.6→GRPO 39.6/DAPO 39.8→**DenoiseRL-GRPO 42.0**(最佳平均)/DenoiseRL-DAPO 41.5;8B Base 29.3→GRPO 43.0/DAPO 42.8→DenoiseRL-GRPO 43.3/**DenoiseRL-DAPO 44.8**(每个 benchmark 均最佳)。
- baseline 公平性:与同 base 同 backbone 的 GRPO/DAPO 直接对照,且 length-fair 折叠保证 denoise 与 main rollout 共享同响应预算,公平。"看着强但没回答"的点:**增益偏小**(8B 上 GRPO 仅 +0.3,DAPO +2.0);且"恢复能力随难度增强"的核心叙事**主要靠定性案例(Table 4/5/6 + Fig.3 overthinking)而非定量恢复率曲线**——没有可复核的"恢复率/纠错率随训练难度变化"硬指标【原文 §4.7, §5; 推断】。
- 假设与失效边界:【原文 Limitations】①效果依赖弱模型行为——若弱模型产出过于平凡/重复/不真实的错误,恢复信号价值有限;②增大 corruption 长度会放大 overthinking(更长轨迹、更高推理成本、更低解码效率)。【推断】仅 MATH-7.5K 单一数据源 + 仅 2 个策略模型(4B/8B)+ 仅数学+BBEH,跨域泛化未充分检验;mask 掉前缀只规避了重 off-policy 优化的不稳定,**没有正面解决从重 off-policy 数据学习的问题**——"去噪/恢复"更多是被**条件化**出来而非被**显式优化**出来(训练只在续写段算梯度,off-policy 前缀实为"免费的难初始状态")。
- 祛魅总结:真贡献=把 prefix 注入从"注入好前缀"反转为"注入弱模型错误前缀",实现干净(只更新 on-policy 续写 + 共享 baseline + 预算折叠),在工程上回避了重 off-policy 优化的不稳定,且把"从错误恢复"显式化为训练目标;消融较扎实(\(\rho\)/\(K\) 扫描、前缀 mask 崩溃、length-fair、训练时间均有)。包装/高估处:"recover from mistakes / denoising"叙事很吸引人,但本质是"用错误前缀当难初始状态做条件化续写 RL",**并未真正优化 off-policy 分布**(前缀被 mask);增益规模有限(8B-GRPO +0.3);"self-correction emerges / 随难度增强"缺定量证据,靠 case study 支撑【推断】。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=verifier 对"错误前缀+策略续写"完整折叠响应的 0/1 结果奖励(组内 N+K 共享 baseline 归一)| **改什么**=策略全参数,但**只在 on-policy 续写 token 上算梯度**(off-policy 前缀 mask)| **何时改**=RL 训练期,每步 N main + K denoise rollout 联合优化 | **免梯度?**=否(token 级 GRPO 梯度训练)| **记忆-技能生命周期**=不涉及显式记忆/技能库;弱模型错误轨迹池 \(\mathcal W(q)\) 一次性离线构建后固定(可视为静态"负样本记忆"),自纠错能力固化进权重 | **防遗忘机制**=无 KL loss(明确"no KL loss"),无专门防遗忘;靠 \(N{:}K=12{:}4\) 的 main:denoise 配比防止偏离主目标(K=8 会损主目标)【原文 §3.2, §4.1, §4.3】。
- ⑦ 开源代码+框架/harness:https://github.com/ALEX-nlp/DenoiseRL(本地已 clone,完整无 LFS/体积问题)。基于 **VeRL** fork(仓内含完整 verl/ 包:base_config.py、interactions/、model_merger/ 等)。核心配方在 **recipe/denoise/**:`dapo_ray_trainer.py`(denoise rollout + 折叠 + GRPO/DAPO 联合目标)、`data_prepare.py`、`verifier.py`、`main_dapo.py`,启动脚本 `denoise_qwen3-{1.7b,4b,8b}_v1.0.sh` 与 `dapo_denoise_qwen3-{1.7b,4b,8b}_v1.0.sh`,配置在 `config/`;另含 `data/`、`paper/`、`img/`(均已本地核验存在)。框架 **VeRL**,RL backbone GRPO/DAPO。代码可得性高(配方/脚本/数据预处理/verifier 齐全,可直接复现 4B/8B)。
- 💰 资源/成本与可扩展性:**有具体数据(Table 3)**:Qwen3-4B-Base、MATH-7.5K、batch 16、**4×H100**;GRPO 基线 16 on-policy rollout = 43.8 s/step,DenoiseRL(12+4)= 49.7 s/step(略慢,因 denoise rollout 让续写 token 多 1.27× 需更长采样/反传,但同 cost 量级)。离线噪声采集为一次性预处理、训练中零额外成本【原文 §4.6, §3.1】。
- 🎯 对"探索-巩固"对标:**强支撑(巩固/回轨侧最近邻竞品 + 探索侧扩状态多样性)** —— 一句判定:DenoiseRL 与本项目"巩固/回轨=走偏后自选恢复分支并固化"几乎正面对标——它把"从错误中间态恢复"显式化为 RL 训练目标,且通过注入弱模型错误前缀扩展失败模式状态多样性(对应"探索=暴露到罕见 off-policy 语境");依据:Fig.1 四步(错误前缀→强制绕道→学去噪→到达答案)、§1"elevates self-correction from emergent behavior to direct training target"。**与本项目的关键区别(竞品 vs 差异化)**:①DenoiseRL 用**弱模型外部错误**前缀(off-policy 注入,Eq.3),本项目设想 student **自己走偏**(on-policy 自选恢复分支)——前者前缀来源 off-policy 且 mask 掉不优化(Eq.7 的 \(\mathcal T_i\) 只含续写),后者强调 on-policy 自选;②DenoiseRL 的 teacher 角色被"弱模型扰动器"替代(无强 teacher 脚手架),本项目是"强 teacher 当稀疏脚手架";③DenoiseRL **无 MTP 前瞻、无技能/记忆固化的持续学习视角**。可借组件:**直接可借**——recipe/denoise 的"前缀注入(Eq.2-3)+折叠(Eq.4)+只更新续写(Eq.7 mask)+共享 baseline(Eq.5)"是 path-recovery 的现成干净实现,本项目可在此 fork 上把"弱模型错误前缀"替换为"student 自生成的走偏前缀"、把"固定 \(\rho\)"替换为"在高熵/低置信关键步切分"(呼应 MEMORY 中 survey-grpo-step-segmentation 的结论:切关键步而非固定比例);缺口:on-policy 自选恢复分支、MTP 前瞻探针、防遗忘的技能固化均需本项目补。**这是 ASSIGNED 六篇里与本项目核心 idea 最直接对标的一篇,建议作为方法实现基线 fork。**
- 🔭 开放问题/未来方向:【原文】Limitations——降低对弱模型错误质量的依赖(过平凡错误无价值)、平衡恢复监督强度与推理效率(抑制 overthinking)。【推断】把 off-policy 错误前缀升级为 on-policy 自选走偏前缀(正面解决 off-policy 优化而非 mask 规避)、给"恢复率"定量度量、把关键步切分(高熵/低置信)替代固定 \(\rho\)、引入 MTP 前瞻判断"何时该恢复"、把恢复能力固化进技能库做持续学习均未涉及。

---
RETURN: denoiserl | 读到PDF? 是(17页全文,Eq.1-11 从 PDF 精确抄录) | L1(前缀注入/自蒸馏)+L3(GRPO/DAPO) | 强支撑且最近邻竞品:path-recovery 显式化的现成干净实现(recipe/denoise 已本地核验),建议作本项目方法基线 fork;但用弱模型 off-policy 错误前缀(mask 不优化)而非 on-policy 自选,无 MTP/无技能固化——正是本项目差异化空间 | 残留待核 1("恢复能力随难度增强"只有定性 case study/overthinking 图,无定量恢复率曲线)

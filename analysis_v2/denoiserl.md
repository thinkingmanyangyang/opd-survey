denoiserl | DenoiseRL: Bootstrapping Reasoning Models to Recover from Noisy Prefixes | 复旦大学 / 上海创智学院(Caijun Xu, Changyi Xiao, Zhongyuan Peng, Yixin Cao) | arXiv 2605.28421(v1 2026-05-27)·预印本 | 主题线 L1(自蒸馏/前缀注入)+ L3(GRPO/DAPO)·相关性 高(与本项目核心 path-recovery 最近邻)

**原始论文**:https://arxiv.org/abs/2605.28421

## 一眼看懂

> 一句话导读:不找更强的老师、也不费劲造难题,而是故意拿一个弱模型写出的错误开头塞给策略,逼它从"半道走错的状态"里把题做对——相当于把"自我纠错"从碰运气涌现出来的能力,变成了一个明确要训练的目标。与本项目的 path-recovery 核心想法最接近的一篇。

- 🟦 TL;DR:把**弱模型生成的错误推理前缀**当成一种"结构化噪声",注入到策略的 rollout 里,再用 RL 训练策略从这个错误的中间状态"去噪并恢复"到正确答案。这样做的好处是:不引入更强的 teacher、也不构造难数据,就把"自纠错 / 从错误中恢复"从一种涌现行为,提升为**显式的训练目标**。在 Qwen3-4B/8B-Base 上一致改进了 GRPO/DAPO 基线(4B 平均 39.6→42.0,8B-DAPO 42.8→44.8)【原文 Abstract, §1, Table 1】。
- 最巧的一步:**只对策略自己生成的续写部分算梯度,把那段 off-policy 的错误前缀 mask 掉(不参与优化)**。
  - 为什么必须这样:这段离线的噪声前缀,在当前策略看来,和当初生成它的那个行为策略的 log-prob 分布差异巨大;一旦 PPO 的重要性比作用到这些"重 off-policy"的 token 上,就会注入高方差的噪声梯度,把推理质量和长度控制都摧毁。
  - 反证:抽掉它(连前缀也一起更新),训练直接崩——§4.4 Fig.5 实证,更新 off-policy 前缀时,验证精度在 step 80 冲到 34.7% 的峰值后急转直下,到 step 400 在所有 benchmark 上塌到 0【原文 §4.4, Fig.5】。

## 为什么做

> 一句话导读:推理 RL 现在很依赖"更强的老师"或"精心造的难题",可两者都不好得。本文换个角度——既然弱模型只会犯错,那就专门用它的错误来给强学生制造"从错误中恢复"的训练机会,等于把弱模型从"老师"改造成"出难题的捣乱者"。

- 研究背景:RL(GRPO/DAPO)已经是 LLM 推理 post-training 的主范式,但 SOTA 系统往往依赖更强模型的监督 / 指导。一旦没有足够强的现成 teacher,能力就越来越难提升,于是引出一个根本问题:**如何在不依赖更强模型当监督者的前提下,得到一个更强的模型?**【原文 §1】。
- 解决的具体痛点,三条:
  - ① **W2SG(weak-to-strong)**用弱模型监督强学生,但学生的天花板会被弱监督者的噪声和有限能力卡死(因为它被优化去模仿那些伪标签);
  - ② **难数据构造**(难题合成 / 对抗样本 / 长轨迹)依赖复杂的流水线、过滤验证以及大量人工;
  - ③ **标准 on-policy RL** 受限于自己生成的状态分布——策略一旦饱和,要么大量产出正确 rollout、要么只剩很窄的失败模式,导致**信息量大的失败样本太稀缺**,提供不了有意义的梯度【原文 §1, §2】。
- 相关工作 & 各自不足(三条并行路线,以及 DenoiseRL 站在谁肩上):
  - **路线 ①——weak-to-strong generalization(W2SG)**【原文 §2】:把弱模型当一个不完美的 oracle 来监督强学生(学生去学伪标签)。**精确短板**:学生天花板被弱监督者的噪声 / 能力封顶。DenoiseRL 的反转是:**不把弱模型当 oracle,而把它当"扰动器"**——弱模型只负责产生错误的中间态,不提供任何学习目标。
  - **路线 ②——prefix / trajectory 注入类**,有两个代表:
    - **LUFFY [36]**:把 off-policy 的推理轨迹**混入** on-policy RL(整条轨迹都参与优化)。DenoiseRL 的差异是**只更新 on-policy 续写、把前缀 mask 掉不优化**(以规避重 off-policy 带来的不稳定)。
    - **PrefixRL [25]**:在**成功**前缀上做条件化并优化续写(让稀疏奖励更容易拿到)。DenoiseRL 的差异是**注入的是弱模型的错误前缀**(失败状态),把任务从"从好起点起飞"反转为"从死胡同里恢复"。
    - 更广的 **expert-solution / oracle-hint / 成功轨迹类 [14,23,29]**:都靠"特权信息 / 示范"来降低稀疏奖励的难度。DenoiseRL 的差异是:错误前缀**既不是示范、也不是特权提示**,而是一个"误导性的推理状态",让策略从中恢复。
  - **路线 ③——去噪自编码视角(BART / denoising autoencoder)**:DenoiseRL 把推理 RL **重新表述为一个去噪问题**(错误前缀就是结构化噪声,策略要学会重建出有效的解题路径),从而把自纠错从涌现行为升为显式训练目标(§1, §3.1)。这是它独有的概念框架。
- 动机链:现状是推理 RL 依赖强 teacher / 难数据;缺陷在于 W2SG 受弱 teacher 封顶、难数据贵、标准 RL 失败样本稀缺;所以本文的做法是把 W2SG 与"难度驱动的数据合成"统一起来——不让弱模型去合成难数据、也不让它给学习信号,而是把它当成一个**结构化的扰动生成器**,在不产生任何新数据的情况下自动抬高难度,并把推理 RL 重述为去噪问题【原文 §1, §3.1】。
- 与最近邻工作的Δ:相对 PrefixRL(注入成功前缀)——**DenoiseRL 反转为注入弱模型的错误前缀**,以此控制起始状态、强迫从损坏的中间态恢复;相对 LUFFY(混入 off-policy 轨迹)——DenoiseRL 只更新 on-policy 续写、前缀 mask。关键有用点:前缀对后续轨迹的影响是不成比例的,因此注入错误前缀既能极大扩展失败模式的状态多样性(暴露到标准 on-policy RL 很少见到的 off-policy 语境),又能直接强化"从错误恢复"这一被低估的能力。

## 怎么做 + 靠不靠谱

> 一句话导读:整套流程——先一次性收集一堆弱模型的错误开头当"噪声库"(0、1);每步训练里,一部分 rollout 正常走,另一部分被强行接在某个错误开头后面续写(2);为了公平比较,两类共用同一个长度预算(3);最后用 GRPO 优化,但只在策略自己写的那段算梯度(4、5)。

### 0. 任务定义:denoising reasoning(§3.1)
把弱模型的错误中间推理态当成一种**结构化噪声**:取一个错误的部分解,**前置(prepend)**到策略生成之前,然后训练策略从这个损坏的状态续写到正确答案。这个错误前缀同时扮演两个角色——既是 weak-to-strong 的扰动信号,又是一个低成本的难度提升器。

### 1. 离线噪声采集(一次性预处理,§3.1)
用弱模型 \(\pi_w\)(Qwen2.5-1.5B-Instruct)对训练集 \(\mathcal D\)(MATH-7.5K)的每道题采 \(M=8\) 次,把 verifier 判错的轨迹收集成一个**固定的池** \(\{\mathcal W(q)\}_{q\in\mathcal D}\)。这个池在整个训练过程中固定不变,因此**每一步都不产生额外成本**。**退化处理**:如果某题 8 次都没采到一个格式良好的错误答案(即 \(\mathcal W(q)=\varnothing\)),就用一条额外的 main rollout 顶替它的 denoise 槽(以保证 batch 的形状不变)。

### 2. 每步双类 rollout(§3.2,Eq.1-3)
对每道题 \(q\sim\mathcal D\),同时产生两类 rollout:
- **Main rollouts(每题 \(N=12\) 条)**:标准的 on-policy 采样
\(\displaystyle y\sim\pi_\theta(\cdot\mid q) \tag{1}\)
- **Denoise rollouts(每题 \(K=4\) 条)**:先抽一个错误解 \(w\sim\mathcal W(q)\),按**固定的前缀比** \(\rho\in(0,1]\) 保留它前面 \(p\) 个 token 当作 assistant 前缀
\(\displaystyle p=\max\big(1,\ \lceil \rho\,|w|\rceil\big)\quad(\rho=0.2) \tag{2}\)
然后策略从这个 **off-policy 前缀**接着往下续写:
\(\displaystyle y_{>p}\sim\pi_\theta(\cdot\mid q,\,w_{1:p}) \tag{3}\)

### 3. 预算折叠(length-fair,§3.2,Eq.4)
两类 rollout 共用同样宽的响应窗 \(R=4096\)。由于前缀已经占掉了 \(p\) 个 token,denoise rollout 会被折叠成下面这样一条可见响应:
\(\displaystyle \tilde y=\big(\underbrace{w_{1:p}}_{\text{prefix}},\ \underbrace{y_{p+1:p+L}}_{\text{continuation}}\big),\qquad p+L\le R,\qquad L=\min(T_{y>p},\,R-p) \tag{4}\)
超出这个 length-fair 预算的尾部 token 会被丢弃。verifier 对**完整折叠后的响应** \(\tilde y\) 打一个 0/1 奖励 \(r(\tilde y;q)\)(看它在被错误前缀条件化之后,有没有最终到达正确答案)。**训练时只更新 on-policy 的续写段 \(y_{p+1:p+L}\)**——这是防止训练崩溃的核心设计。

### 4. token 级 GRPO 目标(§3.2,Eq.5-7)
> 一句话导读:同一道题的 main 和 denoise rollout 共用一个组内 baseline——这点很关键,因为错误前缀会在"本来很容易、全对"的题上制造出失败样本(负样本),让正样本重新带上有效梯度。
- **共享 baseline**:同一道题的 \(N+K\) 条 rollout 共用一个 advantage baseline(记 \(\mathcal G(q)=\{1,\dots,N+K\}\),终端奖励 \(r_i\in\{0,1\}\)):
\(\displaystyle A_i=\frac{r_i-\mu_q}{\sigma_q+\varepsilon},\quad \mu_q=\frac{1}{N+K}\sum_{j\in\mathcal G(q)}r_j,\quad \sigma_q^2=\frac{1}{N+K}\sum_{j\in\mathcal G(q)}(r_j-\mu_q)^2 \tag{5}\)
**这里的关键直觉**:对那些本来很容易的题,denoise rollout 会贡献**负样本**(错误前缀把成功率拉低),从而让该题的正样本重新携带有效的学习信号;否则易题组内全对、advantage 归零,就没有梯度了。
- **token 级重要性比**(context 对 main rollout 是 \(q\),对 denoise rollout 是 \((q,w_{1:p_i})\)):
\(\displaystyle r_{i,t}(\theta)=\frac{\pi_\theta(y_{i,t}\mid c_{i,t},y_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid c_{i,t},y_{i,<t})} \tag{6}\)
- **clip 代理目标**(取 \(\varepsilon_{\mathrm{low}}=\varepsilon_{\mathrm{high}}=0.2\)):
\(\displaystyle \mathcal L_i^{\mathrm{PPO}}(\theta)=\frac{1}{|\mathcal T_i|}\sum_{t\in\mathcal T_i}\min\Big(r_{i,t}(\theta)\hat A_{i,t},\ \mathrm{clip}\big(r_{i,t}(\theta),1-\varepsilon_{\mathrm{low}},1+\varepsilon_{\mathrm{high}}\big)\hat A_{i,t}\Big) \tag{7}\)
（注意:\(\mathcal T_i\) 对 denoise rollout 只包含续写段的索引——前缀不进 \(\mathcal T_i\),也就是被 mask 掉了。）

### 5. 联合目标(§3.2,Eq.8-11)
最终目标是两条 rollout 分布的混合期望(其中 \(\pi_\theta^{\mathrm{main}}(\cdot|q)=\pi_\theta(\cdot|q)\),\(\pi_\theta^{\mathrm{denoise}}(\cdot|q,w)=\pi_\theta(\cdot|q,w_{1:p})\)):
\(\displaystyle \mathcal J(\theta)=\frac{N}{N+K}\,\mathcal J_{\mathrm{main}}(\theta)+\frac{K}{N+K}\,\mathcal J_{\mathrm{denoise}}(\theta) \tag{8}\)
\(\displaystyle \mathcal J_{\mathrm{main}}(\theta)=\mathbb E_{q\sim\mathcal D,\;y\sim\pi^{\mathrm{main}}_{\theta_{\mathrm{old}}}}\big[\mathcal L^{\mathrm{PPO}}(\theta;q,y)\big],\qquad \mathcal J_{\mathrm{denoise}}(\theta)=\mathbb E_{q\sim\mathcal D,\,w\sim\mathcal W(q),\;y\sim\pi^{\mathrm{denoise}}_{\theta_{\mathrm{old}}}}\big[\mathcal L^{\mathrm{PPO}}(\theta;q,w_{1:p},y)\big] \tag{9,10}\)
下面是实际每步优化时用的 Monte-Carlo 估计(其中 \(B\) 是每个 batch 的题数,\(\mathcal M(q_b)\) 是这道题的 \(N\) 条 main rollout,\(\mathcal S(q_b)\) 是它的 \(K\) 条 denoise rollout):
\(\displaystyle \hat{\mathcal J}(\theta)=\frac{1}{B(N+K)}\sum_{b=1}^{B}\Big[\sum_{i\in\mathcal M(q_b)}\mathcal L_i^{\mathrm{PPO}}(\theta)+\sum_{i\in\mathcal S(q_b)}\mathcal L_i^{\mathrm{PPO}}(\theta)\Big] \tag{11}\)
归纳起来就三点:两类 rollout 按 \(N/(N+K)\) 与 \(K/(N+K)\) 加权;共享同一道题的单一 baseline;并且**只在策略自己生成的 token 上更新**。

### 逐组件必要性(消融较齐全)
> 一句话导读:四个开关里,"只更新续写(前缀 mask)"是生死攸关的(去掉就直接崩);其余三个(注入错误前缀、长度公平折叠、噪声强度 ρ、denoise 数量 K)都有对应的扫描或消融,最佳点是 ρ=0.2、K=4。
- **denoise rollout(注入错误前缀)**:这是核心。Table 1 里两个尺度、两个 backbone 全部 ≥ 基线;没它就退回标准 GRPO/DAPO【原文 §4.2 Table 1】。
- **只更新 on-policy 续写(前缀 mask)**:**有硬消融(§4.4 Fig.5)**——不 mask 的话,到 step 400 就塌到 0;没它训练直接崩溃【原文 §4.4】。
- **length-fair 折叠(约束 \(p+L\le R\))**:**有消融(Table 2)**——不设这个上限,平均分从 42.0 降到 40.2(降 1.8 点);因为不公平的"长恢复预算"会鼓励冗长但不可靠的推理【原文 §4.5 Table 2】。
- **噪声强度 \(\rho\)**:**有扫描(§4.3 Fig.2/3)**,扫了 \(\rho\in\{0.2,0.5,0.8\}\)。\(\rho=0.2\) 时长度紧凑(末 100 步平均 1.38K);\(\rho=0.8\) 时升到 2.26K 并触顶;\(\rho=0.5\) 反而最不稳(3.87K)。总体看,\(\rho\) 越大 overthinking 越严重(无休止地自我怀疑、反复验证)。所以 \(\rho=0.2\) 是最佳权衡【原文 §4.3】。
- **denoise rollout 数 \(K\)**:**有扫描(§4.3 Fig.4)**,固定 \(\rho=0.2\) 扫 \(K\in\{1,4,8\}\)。\(K=1\) 信号太稀(+14.9%);\(K=8\) 占了一半 rollout、分散了主目标(+11.9%);**\(K=4\) 最佳(+16.3%,AIME24 +16.5 / AIME25 +16.9)**【原文 §4.3】。

### 实验与证据 / 靠不靠谱
> 一句话导读:方法干净、消融扎实,确实把"从错误恢复"做成了训练目标;但要打折看——增益规模偏小,而且"恢复能力随难度增强"这个核心卖点主要靠案例撑着、缺少定量曲线。还有个本质点:前缀被 mask 掉,所以它其实没真正"优化"off-policy 分布,只是把错误前缀当成"免费的难初始状态"做条件化续写。
- 配置:弱模型用 Qwen2.5-1.5B-Instruct,策略用 Qwen3-4B-Base / Qwen3-8B-Base(仓库另含 1.7B 脚本,但正文主表只报 4B/8B),训练数据为 MATH-7.5K。关键超参:\(N=12\)+\(K=4\)、\(R=4096\)、\(\rho=0.2\)、prompt batch 16、lr 1e-6、**无 KL、无 length loss**、PPO clip 0.2、训练 temperature 1.0 / top-p 1.0。评测集为 MATH500 / AMC23 / AIME2024 / AIME2025 / BBEH(其中 AMC23 / AIME24 / AIME25 报 AVG@16,MATH500 / BBEH 报 AVG@1),验证解码用 temperature 0.6 / top-p 0.95。
- 主表(Table 1 平均):4B Base 26.6 → GRPO 39.6 / DAPO 39.8 → **DenoiseRL-GRPO 42.0**(最佳平均)/ DenoiseRL-DAPO 41.5;8B Base 29.3 → GRPO 43.0 / DAPO 42.8 → DenoiseRL-GRPO 43.3 / **DenoiseRL-DAPO 44.8**(每个 benchmark 都最佳)。
- baseline 公平性:与同 base、同 backbone 的 GRPO/DAPO 直接对照,而且 length-fair 折叠保证 denoise 与 main rollout 共享同样的响应预算,这点公平。但有个"看着强、其实没真正回答"的地方:**增益偏小**(8B 上 GRPO 只 +0.3,DAPO +2.0);而且"恢复能力随难度增强"这个核心叙事**主要靠定性案例(Table 4/5/6 + Fig.3 的 overthinking)、而不是定量的恢复率曲线**——没有可复核的"恢复率 / 纠错率随训练难度变化"的硬指标【原文 §4.7, §5;推断】。
- 假设与失效边界:
  - 【原文 Limitations】① 效果依赖弱模型的行为——如果弱模型产出的错误太平凡、太重复或不真实,恢复信号的价值就有限;② 增大 corruption 长度会放大 overthinking(轨迹更长、推理成本更高、解码效率更低)。
  - 【推断】只用了 MATH-7.5K 这一个数据源 + 仅 2 个策略模型(4B/8B)+ 仅数学和 BBEH,跨域泛化没充分检验;而且 mask 掉前缀只是规避了重 off-policy 优化的不稳定,**并没有正面解决"从重 off-policy 数据学习"这个问题**——"去噪 / 恢复"更多是被**条件化**出来的,而不是被**显式优化**出来的(训练只在续写段算梯度,那段 off-policy 前缀其实就是"免费的难初始状态")。
- 祛魅总结:
  - 真贡献:把 prefix 注入从"注入好前缀"反转成"注入弱模型的错误前缀",实现得很干净(只更新 on-policy 续写 + 共享 baseline + 预算折叠),在工程上回避了重 off-policy 优化的不稳定,并把"从错误恢复"显式化为训练目标;消融也较扎实(\(\rho\) / \(K\) 扫描、前缀 mask 崩溃、length-fair、训练时间都做了)。
  - 包装 / 高估处:"recover from mistakes / denoising"这个叙事很吸引人,但本质上就是"用错误前缀当难初始状态、做条件化续写 RL",**并没有真正优化 off-policy 分布**(前缀被 mask);增益规模有限(8B-GRPO 仅 +0.3);"self-correction emerges / 随难度增强"缺定量证据,靠 case study 支撑【推断】。

## 结构化抽取
- 🎯 机制速览6轴:**学什么信号**=verifier 对"错误前缀+策略续写"完整折叠响应的 0/1 结果奖励(组内 N+K 共享 baseline 归一)| **改什么**=策略全参数,但**只在 on-policy 续写 token 上算梯度**(off-policy 前缀 mask)| **何时改**=RL 训练期,每步 N main + K denoise rollout 联合优化 | **免梯度?**=否(token 级 GRPO 梯度训练)| **记忆-技能生命周期**=不涉及显式记忆/技能库;弱模型错误轨迹池 \(\mathcal W(q)\) 一次性离线构建后固定(可视为静态"负样本记忆"),自纠错能力固化进权重 | **防遗忘机制**=无 KL loss(明确"no KL loss"),无专门防遗忘;靠 \(N{:}K=12{:}4\) 的 main:denoise 配比防止偏离主目标(K=8 会损主目标)【原文 §3.2, §4.1, §4.3】。
- ⑦ 开源代码+框架/harness:https://github.com/ALEX-nlp/DenoiseRL(本地已 clone,完整无 LFS/体积问题)。基于 **VeRL** fork(仓内含完整 verl/ 包:base_config.py、interactions/、model_merger/ 等)。核心配方在 **recipe/denoise/**:`dapo_ray_trainer.py`(denoise rollout + 折叠 + GRPO/DAPO 联合目标)、`data_prepare.py`、`verifier.py`、`main_dapo.py`,启动脚本 `denoise_qwen3-{1.7b,4b,8b}_v1.0.sh` 与 `dapo_denoise_qwen3-{1.7b,4b,8b}_v1.0.sh`,配置在 `config/`;另含 `data/`、`paper/`、`img/`(均已本地核验存在)。框架 **VeRL**,RL backbone GRPO/DAPO。代码可得性高(配方/脚本/数据预处理/verifier 齐全,可直接复现 4B/8B)。
- 💰 资源/成本与可扩展性:**有具体数据(Table 3)**:Qwen3-4B-Base、MATH-7.5K、batch 16、**4×H100**;GRPO 基线 16 on-policy rollout = 43.8 s/step,DenoiseRL(12+4)= 49.7 s/step(略慢,因 denoise rollout 让续写 token 多 1.27× 需更长采样/反传,但同 cost 量级)。离线噪声采集为一次性预处理、训练中零额外成本【原文 §4.6, §3.1】。
- 🎯 对"探索-巩固"对标:**强支撑(巩固/回轨侧最近邻竞品 + 探索侧扩状态多样性)** —— 一句判定:DenoiseRL 与本项目"巩固/回轨=走偏后自选恢复分支并固化"几乎正面对标——它把"从错误中间态恢复"显式化为 RL 训练目标,且通过注入弱模型错误前缀扩展失败模式状态多样性(对应"探索=暴露到罕见 off-policy 语境");依据:Fig.1 四步(错误前缀→强制绕道→学去噪→到达答案)、§1"elevates self-correction from emergent behavior to direct training target"。**与本项目的关键区别(竞品 vs 差异化)**:①DenoiseRL 用**弱模型外部错误**前缀(off-policy 注入,Eq.3),本项目设想 student **自己走偏**(on-policy 自选恢复分支)——前者前缀来源 off-policy 且 mask 掉不优化(Eq.7 的 \(\mathcal T_i\) 只含续写),后者强调 on-policy 自选;②DenoiseRL 的 teacher 角色被"弱模型扰动器"替代(无强 teacher 脚手架),本项目是"强 teacher 当稀疏脚手架";③DenoiseRL **无 MTP 前瞻、无技能/记忆固化的持续学习视角**。可借组件:**直接可借**——recipe/denoise 的"前缀注入(Eq.2-3)+折叠(Eq.4)+只更新续写(Eq.7 mask)+共享 baseline(Eq.5)"是 path-recovery 的现成干净实现,本项目可在此 fork 上把"弱模型错误前缀"替换为"student 自生成的走偏前缀"、把"固定 \(\rho\)"替换为"在高熵/低置信关键步切分"(呼应 MEMORY 中 survey-grpo-step-segmentation 的结论:切关键步而非固定比例);缺口:on-policy 自选恢复分支、MTP 前瞻探针、防遗忘的技能固化均需本项目补。**这是 ASSIGNED 六篇里与本项目核心 idea 最直接对标的一篇,建议作为方法实现基线 fork。**
- 🔭 开放问题/未来方向:【原文】Limitations——降低对弱模型错误质量的依赖(过平凡错误无价值)、平衡恢复监督强度与推理效率(抑制 overthinking)。【推断】把 off-policy 错误前缀升级为 on-policy 自选走偏前缀(正面解决 off-policy 优化而非 mask 规避)、给"恢复率"定量度量、把关键步切分(高熵/低置信)替代固定 \(\rho\)、引入 MTP 前瞻判断"何时该恢复"、把恢复能力固化进技能库做持续学习均未涉及。

---
RETURN: denoiserl | 读到PDF? 是(17页全文,Eq.1-11 从 PDF 精确抄录) | L1(前缀注入/自蒸馏)+L3(GRPO/DAPO) | 强支撑且最近邻竞品:path-recovery 显式化的现成干净实现(recipe/denoise 已本地核验),建议作本项目方法基线 fork;但用弱模型 off-policy 错误前缀(mask 不优化)而非 on-policy 自选,无 MTP/无技能固化——正是本项目差异化空间 | 残留待核 1("恢复能力随难度增强"只有定性 case study/overthinking 图,无定量恢复率曲线)

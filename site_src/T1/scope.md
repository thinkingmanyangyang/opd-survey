# scope — SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting

> **一句话重点 (TL;DR)**：标准 OPD 把 teacher 的稠密 token-level 监督一视同仁地用在所有 rollout 上,忽视信号质量差异;SCOPE 按轨迹正确性双路路由——正确轨迹用 **student-PPL 加权 MLE 自强化**(放大能力边界处低置信样本),错误轨迹用 **teacher-PPL 加权 KL 蒸馏**(优先 teacher 真有纠错能力即低 PPL 的实例),并在组内做 perplexity 归一化。

**元信息**：arXiv 2604.10688 (v2, 2026-05-30, cs.LG) ｜ 中科大/美团 LongCat/南大/复旦/华科(Binbin Zheng¹²、Xing Ma²、Yiheng Liang²³、Jingqing Ruan²、Xiaoliang Fu⁴、Kepeng Lin⁵、Benchang Zhu²、Ke Zeng²、Xunliang Cai² 通讯;实习期间完成) ｜ arXiv preprint, 2026-04(v2 2026-05) ｜ 主题 T1(On-Policy Distillation),Relevance=Med(双路"正确路强化 student 自身、错误路用 teacher 纠错"与 TSRD path-selection/path-recovery 强相关) ｜ 代码 github.com/machine981/SCOPE(已克隆 ~9.1M;HF 已放 SCOPE-Qwen3-1.7B、SCOPE-Deepseek-R1-Distill-Qwen-1.5B) ｜ 框架 veRL(volcengine/verl)

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/scope/fig_01.png)

*Figure 1: (a) Performance changes on the AIME24 benchmark before and after training. Both PSR and OPD training enhance pass@1 at the expense of pass@32, highlighting a clear trade-off between accuracy and reasoning diversity. (b) Recovery rate of the teacher model across varying truncation levels, c*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/scope/fig_02.png)

*Figure 2: (a) Standard OPD applies uniform supervision to all samples. (b) Our SCOPE framework refines the learning process by first dividing trajectories into correct Ω C and incorrect Ω W sets, applying dual-path perplexity-based weighting, and finally optimizing the weighted branches via a unifie*

## 1. 相关工作与进展
On-policy RL(DeepSeek-R1/GRPO/DAPO 范式)已成 LLM 推理对齐主流,但稀疏 outcome-level 奖励使 token-level credit assignment 困难、收敛慢。**OPD**(On-Policy Distillation)在 student 自采样 rollout 上引入 teacher 的稠密 token-level KL 监督来缓解,兼顾分布一致与训练效率。SCOPE 直接以"标准 OPD 一视同仁"为靶子改进加权。

## 2. 现有工作存在的问题
现有 OPD 假设 teacher 的 dense 监督在**所有 rollout 上一致可靠**,忽视信号质量根本差异,带来两个问题:(1) **多样性退化**——对正确路径一律等权强化,抑制能力边界处有效但非常规的推理路径,过度强化已掌握样本;(2) **纠错低效**——对错误轨迹,当 teacher 自身也不熟悉(高 teacher PPL)时其 token 分布是不可靠信号(论文称 context 诱导噪声),会误导。

## 3. Motivation
经验分析(论文 §2 PSR/recovery 实验,从 DeepMath 采 2,000 题用 student 生成轨迹、用 teacher 计算 PPL 分桶):**错误轨迹上低 teacher PPL 强相关于成功 error recovery**(低 PPL=teacher 有真纠错把握,可作"真正纠错能力"代理);**正确轨迹上应按 student PPL 自适应加权**,把强化集中到能力边界的低置信(高 student PPL)样本而非已掌握样本。故应按 rollout 正确性路由到不同监督路径并各自加权。

## 4. 主要灵感 / 核心直觉
信号质量可由 PPL 探针预测:teacher PPL 衡量"teacher 在此轨迹上是否可信",student PPL 衡量"该样本是否处于 student 能力边界"。把"按正确性分路 + 按 PPL 自适应加权"组合成单一目标。

## 5. 主要解决思路(一段话讲清核心)
双路自适应框架 DPAW:按正确性把 on-policy rollout 路由到两条互补监督路径——**Student Path(正确轨迹 Ω_c)** 做 student-PPL 加权 MLE 自强化,**Teacher Path(错误轨迹 Ω_w)** 做 teacher-PPL 加权 KL 蒸馏;两路均在同 prompt 轨迹组内做 perplexity 归一化(group-level softmax),以应对 prompt 间难度方差。总目标 L_SCOPE = Σ_{i∈Ω_c} w_i^stu·L_MLE + Σ_{i∈Ω_w} w_i^tea·L_OPD。

## 6. 方法详解(通俗、分步骤)
- **Student Path(Eq.4)**:w_i^stu = softmax_{j∈Ω_c}(−1/(τ|y_i|)·logπ_S(y_i|x)) = PPL_S(y_i|x)^{1/τ} 归一化 ⇒ **student PPL 越高权重越高**,放大能力边界非常规有效路径。
- **Teacher Path(Eq.5)**:w_i^tea = softmax_{j∈Ω_w}(+1/(τ|y_i|)·logπ_T(y_i|x)) = PPL_T(y_i|x)^{−1/τ} 归一化 ⇒ **teacher PPL 越高权重越低**,过滤 teacher 不可信(高 PPL)的错误轨迹噪声。
- **组内归一化**:同一 prompt 的正确组、错误组分别做 softmax,权重乘以组大小(均值≈1),自适应校准。
- **流程**:① `pip install -r requirements.txt` + `pip install -e .`(装 verl);② `bash deploy_vllm.sh` 部署 teacher(默认 Skywork-OR1-7B,served_model_name/api-key 须与 `verl/utils/api_interface.py` 一致,支持多节点 IP_POOL);③ 在 `run_experiment_distill_1_5b.sh` 设 TEACHER_MODEL_NAME、IP_POOL、POLICY_MODEL_PATH;④ 训练。双路开关在 `verl/trainer/ppo/ray_trainer.py:_compute_scope_dual_path_weights`:USE_SCOPE_DUAL_PATH_WEIGHTING=True、SCOPE_TAU=1、SCOPE_USE_SEQ_WEIGHTS=True、USE_STUDENT/TEACHER_PATH_WEIGHTS=True。

## 7. 实验数据集
- 两组 teacher–student(论文 §4.1):主配置 teacher=**Skywork-OR1-7B**(脚本/论文均写 `Skywork-OR1-7B`;**README 结果标题误写 `Skywork-OR1-Math-7B`,不一致**)→ student=DeepSeek-R1-Distill-Qwen-1.5B;第二配置 teacher=Qwen3-8B-Instruct → student=Qwen3-1.7B-Base。均在 **DeepMath**(DeepMath-103k)训练。
- 评测 6 数学 benchmark:AIME24、AIME25、AMC23、MATH500、Minerva、OlympiadBench,报 Avg@32 / Pass@32(rollout temp=0.6、top-p=0.95、max_response=32,768)。
- 扩展:3 代码 benchmark(HumanEval、Codeforces、LiveCodeBench)证跨可验证任务普适。

## 8. 实验结果与主要发现
- 摘要主口径:较竞争基线(GRPO/KD/OPD 取平均)平均相对提升 **Avg@32 +11.42%、Pass@32 +7.30%**。
- 仅相对标准 OPD(主配置 1.5B,Table 1):平均 Avg@32 **+5.54%**、Pass@32 **+2.60%**(逐项如 +0.90/+0.20、+8.31/+3.96、+10.69/+2.31 等)。
- 〔待核〕6 benchmark 各自精确 baseline 数值见 Table 1(如主配置 SCOPE Avg@32 AIME24 42.7 / Olympiad 49.7),τ 消融见 Table 6,§2 的 PSR/recovery 经验图具体数据未细抄。

## 9. 结果如何支撑其主张
"按信号质量加权优于一视同仁"由相对标准 OPD 的 +5.54%/+2.60% 直接支撑;"PPL 可预测信号质量"由 §2 的 recovery 分桶相关性支撑;"普适性"由代码任务扩展支撑。支撑成立,但相对 OPD 的净增益(尤其 Pass@32 +2.60%)偏小,且未报多 seed 方差,统计稳健性存疑。

## 10. 逻辑自洽性(中性评估)
方法公式(Eq.4/5)与"放大边界/过滤不可信"叙事自洽。但需中性看待:(1) student-PPL 高=能力边界 与 student-PPL 高=该轨迹只是噪声/错得自信,二者难区分,Student Path 仅作用于**正确**轨迹一定程度缓解但未完全排除;(2) teacher PPL 作"纠错能力"代理是经验相关而非因果,论文未给反事实验证;(3) 净增益较小且无方差报告。**最关键:代码实现的加权方向与论文公式存在系统性不一致(见 §11),若按发布脚本运行,Student Path 实际加权方向与 Eq.4 相反。**

## 11. 残留问题 / 局限
- 相对标准 OPD 增益小、无多 seed 方差;τ、归一化粒度的鲁棒性仅部分消融。
- 经验代理(PPL→信号质量)缺因果证据。
- **〔代码-论文差异,本轮重新严格推导,修正了上一轮结论〕** `_compute_scope_dual_path_weights` 用 `logits = sign·mean_logp/τ` 做 softmax,其中 `sign = +1 if ppl_positive else −1`,而 PPL=exp(−mean_logp):
  - 经数值验证:**`ppl_positive=True` ⇒ 高 PPL 得低权重;`ppl_positive=False` ⇒ 高 PPL 得高权重**(代码注释 line 214-215 把二者写反,注释本身有误)。
  - 论文要求:Student 高 PPL→高权重(Eq.4,需 `ppl_positive=False`);Teacher 高 PPL→低权重(Eq.5,需 `ppl_positive=True`)。
  - 两个 SCOPE 训练脚本均设 `STUDENT_PATH_PPL_POSITIVE=True`、`TEACHER_PATH_PPL_POSITIVE=True`(distill_1_5b.sh:40-41、qwen_1_7b.sh:43-44)。据上推导:**Teacher Path(=True)实际与论文 Eq.5 一致(正确)**;而 **Student Path(=True)与论文 Eq.4 相反(错误)——高 student PPL 反被降权,与"放大能力边界样本"主张相悖**。
  - 这与上一轮分析的结论**相反**(上一轮认为 teacher=True 错、应改 False;实为 student=True 错、应改 False)。根因是上一轮误读了代码注释而未做 softmax 方向验证。函数默认值 `student=True, teacher=False`:据上推导 default 下 student 仍错、teacher 也与 Eq.5 相反——即**纯默认值两路都偏离论文**;唯有脚本里 teacher=True 把 teacher 路救回正确。复现时应将 **student_path_ppl_positive 改为 False** 才符合论文。〔待核:发布的 HF 权重究竟按哪组 sign 训练,无法从仓库静态确认;建议跑一次权重对齐实验定性。〕
- **〔代码-论文差异〕** 训练长度:论文 §4.1/Table 写 max_prompt=**4,096**;但脚本 `run_experiment_distill_1_5b.sh:16`、`_qwen_1_7b.sh` 实设 `MAX_PROMPT_LENGTH=2048`(completion=12,288 两者一致)。
- 上一轮称仓库含 `verl-distillation-ori` 定制分支——**本轮核查不存在该目录**,蒸馏逻辑直接整合进仓库内 `verl/`(已更正)。

## 12. 开源代码与框架(链接+框架+代码可得性)
- 仓库:https://github.com/machine981/SCOPE(已 clone ~9.1M);权重 HF `Machine981/SCOPE-Qwen3-1.7B`、`Machine981/SCOPE-Deepseek-R1-Distill-Qwen-1.5B`。
- 框架:**veRL(volcengine/verl)**,蒸馏逻辑整合在仓库内 `verl/`(非独立分支)。安装 `pip install -r requirements.txt` + `pip install -e .`。teacher 经 vLLM(`deploy_vllm.sh`)以 OpenAI 兼容 API 提供,verl 侧经 `verl/utils/api_interface.py` 调用(api-key 须一致),支持多节点 IP_POOL。`recipe/` 含 dapo/drgrpo/prime/r1/sppo 等参考配方。
- 入口脚本:`run_experiment_distill_1_5b.sh`(SCOPE 主)、`run_experiment_qwen_1_7b.sh`;GRPO 基线脚本(`*_grpo.sh`,teacher=False 且不启用 dual-path)。
- 代码可得性:Tier A(完整可跑),但上述加权方向 bug 需复现者警惕。

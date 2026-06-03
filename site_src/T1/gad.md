# gad — Black-Box On-Policy Distillation of Large Language Models (GAD)

> **一句话重点 (TL;DR)**：当教师是只返回文本的闭源 API(如 GPT-5)时无法做白盒/likelihood 蒸馏；GAD 把学生当生成器、训练一个判别器去区分学生与教师文本，构成 GAN 式极大极小博弈——判别器即"随学生共同演化的 on-policy reward model"，从而在黑盒下实现 on-policy 蒸馏，避免固定 reward model 的 reward hacking。

**元信息**：arXiv:2511.10643（v3, 2026-01-08；首发 2025-11）｜ 微软研究院（Tianzhu Ye、Li Dong 共同一作；Furu Wei 等）｜ 主题 T1（黑盒 on-policy 蒸馏）/ 相关性高 ｜ 代码 microsoft/LMOps `gad/` 子目录（aka.ms/GAD-github 重定向至此；本地已 clone 488KB）｜ 框架 veRL（GRPO，hack critic 当判别器）。

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/gad/fig_01.png)

*Figure 1: Comparison between GAD and sequence-level knowledge distillation (SeqKD; KR16) trained on LMSYS-Chat [ZCS + 24] dataset, evaluated by averaged GPT-4o scores. Left : Results on the LMSYS-Chat test set. Right : Average performance across Dolly [Dat23], SelfInst [WKM + 23], and Vicuna [CLL +*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/gad/fig_02.png)

*Figure 2: Training procedure of GAD. The student (generator) learns to generate responses that maximize the score assigned by the discriminator. The discriminator is trained with Bradley-Terry loss to assign a lower score to the student than the teacher, learning to distinguish between them. Togethe*

## 1. 相关工作与进展
- **白盒蒸馏**：能拿到教师 logits/隐状态，用 forward/reverse KLD 对齐分布(KR16、GKD/MiniLLM 等)。
- **on-policy 蒸馏的价值**：白盒研究表明让学生学**自己生成的响应**(reverse KLD)能 mode-seeking、减小 exposure bias，优于纯 teacher-forcing。
- **黑盒蒸馏**：教师只返回文本(代理 API 教师)，标准做法是对教师响应做 **SeqKD**(序列级 SFT/行为克隆)。

## 2. 现有工作存在的问题
- **黑盒下 on-policy 不可行**：学生生成自身响应后，教师无法给出任何概率级监督来评价/纠正它，标准 likelihood-based on-policy 蒸馏失效；学生与教师 tokenizer 不兼容时 likelihood 目标也失效。
- **SeqKD 的弱点**：易过拟合教师的局部 n-gram 模式、OOD 泛化差(只学到表层词汇而非全局风格)。
- **RLHF 固定 reward model 的弱点**：reward model 预训练后冻结，策略易 **reward hacking**(钻奖励空子，如响应暴长)。

## 3. Motivation
要在"只能看到教师文本"的黑盒约束下，仍然获得 on-policy 学习的好处(mode-seeking、低 exposure bias、好的 OOD 泛化)，就需要一种**不依赖教师概率**、又能对学生自身生成给出反馈的监督信号。

## 4. 主要灵感 / 核心直觉
把蒸馏看成 GAN：学生=生成器 G，再训一个判别器 D 去区分"这是学生还是教师的文本"。G 努力骗过 D(让自己的文本被打高分)，D 努力分开二者，形成极大极小博弈。D 本质是一个**随学生策略共同演化的 reward model**——它始终针对学生当前行为给反馈，因此不像固定 reward model 那样被 hack。

## 5. 主要解决思路(一段话讲清核心)
价值函数 max_G min_D V = E_{(x,y_t)}[ −log σ(D(y_t) − D(G(x))) ](Bradley-Terry 偏好，教师分高于学生)。判别器由生成器参数初始化、加一个标量预测头(取末 token 隐状态投影为序列级分数)，用 BT loss 在线更新；生成器目标 max_G E[D(G(x))]，因采样不可微，把 D(G(x)) 当 reward 用 **GRPO** 做策略梯度优化。二者交替更新、co-evolve。

## 6. 方法详解(通俗、分步骤)
1. **构数据**：遍历 prompt x，采教师响应 y_t，得训练集 T={(x, y_t)}。
2. **Warmup(关键)**：GAD 正式训练前，对生成器用教师响应做 CE/SFT 一个 epoch、对判别器用同数据做 BT loss 一个 epoch，保证生成器-判别器平衡(消融显示二者 warmup 均关键)。
3. **GAD 训练循环**：每个 batch——(a) 采学生响应 G(x)；(b) 用 D(G(x)) 作 reward、GRPO 更新生成器；(c) 用 BT loss(教师分应高于学生)更新判别器。
4. **RL 框架映射**：Policy=学生、Reward Model=判别器、reward=D(G(x))；与 RLHF 的唯一区别是 reward model **在线共更**而非冻结。

## 7. 实验数据集
- **训练**：LMSYS-Chat-1M-Clean(从中采 200K prompt，收集 GPT-5-Chat 教师响应)。
- **评测**：LMSYS-Chat 测试集 500 样本(主)；OOD 用 Dolly(500)、SelfInst(252)、Vicuna(80)。打分用 GPT-4o(先生成参考答案再对比打分)+ 人工评测。
- **教师/学生**：教师 GPT-5-Chat(闭源)；学生 Qwen2.5-Instruct(3B/7B/14B)、Llama-3.2-3B / Llama-3.1-8B-Instruct。

## 8. 实验结果与主要发现
- **GAD 全面超过 SeqKD**(Table 2，所有数据集×模型尺寸)：如 Qwen2.5-14B+GAD 在 LMSYS 得 52.1，**接近 GPT-5-Chat 教师(51.7)**；3B+GAD≈7B+SeqKD(尺寸"升一级")。
- **OOD 上差距更大**：Dolly/SelfInst/Vicuna 上 SeqKD 增益微弱甚至为负，GAD 仍稳健提升——归因于 RL 比 SFT 泛化更好。
- **人工评测**：GAD 对 before-distill 与 SeqKD 的胜率基本 >50%、负率 <30%。
- **机制证据**：(i) N-gram 重叠显示 SeqKD 过拟合教师局部词汇、GAD 学全局风格(Fig.4)；(ii) toy data 上 SeqKD mode-covering、GAD mode-seeking(Fig.5)；(iii) **on-policy 判别器避免了 off-policy 判别器约 300+ 步后的 reward hacking(响应暴长)**(Fig.6)。
- **消融**：BT loss 优于 CE loss；判别器与生成器同尺寸最优(增大判别器无益)。

## 9. 结果如何支撑其主张
主张是"黑盒下也能 on-policy 蒸馏且优于 SeqKD"。自动+人工双评测 + 五个模型一致超 SeqKD 支撑"更优"；OOD 显著优势 + N-gram/toy/reward-hacking 三项机制分析支撑"为什么更优(RL 泛化、mode-seeking、on-policy 抗 hacking)"——主张与证据对应紧密。"接近教师"仅在 14B + LMSYS 评测下成立，属上限演示而非普遍结论。

## 10. 逻辑自洽性(中性评估)
逻辑闭环良好：GAN↔RLHF 的映射(判别器=on-policy reward model)既自然又能解释"为何不 reward hacking"。需保留的批判点：(1)所有自动评测靠 GPT-4o 打分，存在评判模型偏好(偏向某种风格)的系统性风险，人工评测样本量(每对 25/题级)也有限；(2)"接近 GPT-5-Chat"的判定建立在 GPT-4o 评分这一代理指标上，并非任务正确率；(3)对抗训练本身的稳定性(GAN 难训)论文靠 warmup 缓解，但未给收敛性保证。

## 11. 残留问题 / 局限
- **评测依赖 LLM-as-judge**：GPT-4o 打分可能与真实质量/人类偏好有偏差，且教师/裁判都来自相近生态。
- **任务域偏聊天/通用指令**：未验证数学、代码等强推理域的黑盒蒸馏效果。
- **训练成本**：需同时维护生成器+判别器(同尺寸)、3 epoch≈2400 步，比单纯 SeqKD 重。
- **核心算法实现在外部 fork**(YTianZHU/verl)而非本地 clone 的 LMOps/gad(后者仅环境/数据/启动脚本)，算法细节需到该 fork 审计。

## 12. 开源代码与框架(链接+框架+代码可得性)
- 代码仓库为 **microsoft/LMOps 的 `gad/` 子目录**(aka.ms/GAD-github 重定向至此)；本地已 clone `resource/repos/gad`(488KB)。LMOps/gad 仅含环境/数据/启动脚本(`scripts/`、`tools/export_lmsys_parquet.py`、`local_setup.sh`)。
- **算法核心实现在外部 fork** `github.com/YTianZHU/verl`(基于 VeRL，含 seqkd/warmup/gad/eval 四分支)；数据/模型在 HF(ytz20/LMSYS-Chat-GPT-5-Chat-Response、ytz20/gad-models)。项目页 ytianzhu.github.io/Generative-Adversarial-Distillation/。
- **框架**：veRL(GRPO-based)，README 明确"hack critic 模块当判别器"，生成器=policy、判别器=在线 reward model；推荐 docker `czwin32768/verl2:v0.2.0-vllm085`(py3.10/torch2.6/vllm0.8.5)。SeqKD/warmup 的 SFT 也在 fork 内(`dp_actor.py`)。
- **训练**：3 epochs、batch 256、约 2400 步(PPO mini-batch 256)；prompt≤2048、响应≤1536；温度 0.8；每 50 步存 ckpt，选 GPT-4o 分最高且响应长度合理者。

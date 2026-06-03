# search_r1 — Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning

> **一句话重点 (TL;DR)**：把搜索引擎调用嵌进 RL 训练循环——用结构化标签做多轮"推理↔检索"交织生成,对检索回来的 token 做 **loss masking**,仅用最简的 **EM 结果奖励**(无过程/格式奖励),即可让 LLM 自发学会"何时检索、检索什么、如何结合检索继续推理"。

**元信息**：arXiv 2503.09516 (v5, 2025-08-05;COLM 2025 会议论文);实证扩展见 arXiv 2505.15117 ｜ UIUC(Bowen Jin、Zhenrui Yue、Dong Wang、Jiawei Han)+ UMass Amherst(Hansi Zeng、Hamed Zamani)+ Google Cloud AI Research(Jinsung Yoon、Sercan Ö. Arık) ｜ 2025-03 首发,COLM 2025 ｜ 主题 T2(agentic/工具调用 RL),与本课题相关:多轮"推理-检索"交织端到端 RL、检索 token loss masking、纯结果奖励——可作 agentic OPD 的对照/底座 ｜ 代码 github.com/PeterGriffinJin/Search-R1(Tier A,~8.3M) ｜ 框架 veRL 定制 fork

## 关键图示

**① Motivation / 对比图**

![① Motivation / 对比图](../figures/search_r1/fig_02.png)

*Figure 2: (a) PPO vs. GRPO: GRPO generally converges faster but may exhibit instability after trained for a number of steps, whereas PPO provides more stable optimization but converges at a slower rate. (b) Base vs. Instruct LLM study: Instruction-tuned LLMs converge faster, but the final performanc*

**② 方法 / 架构图**

![② 方法 / 架构图](../figures/search_r1/fig_01.png)

*Figure 1: Demonstration of PPO and GRPO training with the search engine (SEARCH-R1). During the rollout, LLMs can conduct multi-turn interactions with the search engine.*

## 1. 相关工作与进展
LLM 推理时常需外部知识/时效信息。已有两类:(1) **RAG**——单轮"检索→生成",query 质量与多轮交互受限;(2) **搜索当工具**——提示式/SFT 式工具调用泛化差。R1/o1 表明纯结果奖励 RL 能自发涌现推理能力,本文将其扩展到"会调用搜索引擎"的场景。已被 veRL 官方、SkyRL、Thinking Machines Tinker-cookbook 集成复现。

## 2. 现有工作存在的问题
(1) 如何把搜索引擎**稳定地**嵌入 RL 训练循环——检索回来的 token 不应被当作策略生成 token 优化(否则训练不稳);(2) 是否需要复杂过程奖励,还是简单结果奖励即可;(3) 多轮交织检索-推理的结构化建模缺失。

## 3. Motivation
让 LLM 在 RL(规则化结果奖励)中自主学会"何时检索、检索什么、如何结合检索结果继续推理",无需过程监督或检索标注,得到完全开源、可替代 OpenAI DeepResearch 的工具增强推理 RL 方案。

## 4. 主要灵感 / 核心直觉
检索内容是"环境观测"而非策略输出,故须从策略梯度中屏蔽;而最终答案正确与否(EM)已是足够的学习信号,无需精心设计过程奖励——把复杂性留给模型自己在 RL 中涌现。

## 5. 主要解决思路(一段话讲清核心)
用结构化标签把"多轮推理-检索"显式建模,生成遇 `</search>` 即抽 query 调检索、把结果包进 `<information>` 拼回上下文继续生成;在 PPO/GRPO 的 token 级损失上对**检索回来的 token 做 masking**(只对 LLM 生成 token 计损),仅用 EM 结果奖励 + 对 π_ref 的 KL 约束做策略更新。

## 6. 方法详解(通俗、分步骤)

- **多轮交织生成**:结构化标签 `<think></think>`(推理)、`<search></search>`(触发检索的 query)、`<information></information>`(检索返回内容)、`<answer></answer>`(最终答案)。
- **检索 token 的 loss masking**:对 PPO/GRPO token 级损失,只对 LLM 生成 token 计损(I(y_t)=1),检索 token 置 0,避免对外部 token 求梯度致训练不稳。
- **奖励设计**:仅用规则化结果奖励(基于答案 Exact Match),不用过程奖励或格式奖励(论文论证最简奖励已足够)。
- **目标函数**:带 KL 约束(对 π_ref)的策略梯度,兼容 PPO 与 GRPO;**默认 PPO**("Unless stated otherwise, PPO is used as the default RL method")。

## 7. 实验数据集

- **训练**:合并 **NQ + HotpotQA** 训练集。
- **评测(7 QA)**:单跳 NQ、TriviaQA、PopQA;多跳 HotpotQA、2WikiMultiHopQA、Musique、Bamboogle(in-domain=NQ/HotpotQA,余 5 项 OOD)。
- **知识源**:2018 Wikipedia dump;检索器 **E5**(稠密);检索文档数主实验固定 **top-k=3**(另做 k=1/3/5 消融)。指标 **Exact Match(EM)**。
- **模型**:Qwen2.5-3B/7B(Base 与 Instruct);Llama3.2 见后续实证扩展版(2505.15117),COLM 主文仅用 Qwen2.5。

## 8. 实验结果与主要发现

- **主结果(EM,Table 2)**:Qwen2.5-7B Search-R1-base Avg = **0.431** vs RAG **0.304**,相对提升约 24%;Qwen2.5-3B 约 20%(同检索器/语料/训练集/预训练模型设定)。
- **口径差异(论文内)**:摘要写"24%(7B)/20%(3B)";贡献处另给"**41% / 20%**"(同一 setup 下不同 baseline 取法)。两者并存,摘要为主口径。
- **PPO vs GRPO(§5.1)**:GRPO 收敛更快但长训后**奖励坍缩**,PPO 更稳定;两者最终奖励相当。
- **奖励/标签消融**:简单 EM 结果奖励已足够;Base 与 Instruct 模型均能自发学会多轮检索-推理行为。

## 9. 结果如何支撑其主张
"搜索可稳定嵌入 RL"由 loss masking 下训练稳定 + 跨 7 数据集一致增益支撑;"简单结果奖励足够"由无过程奖励仍涨点支撑;"自发涌现检索-推理"由 Base 模型也学会多轮行为支撑。支撑充分。中性提示:24% vs 41% 两口径并存易致引用混淆,需注明 baseline 取法。

## 10. 逻辑自洽性(中性评估)
方法-实验-结论自洽,是该方向的奠基工作之一(被广泛复现佐证其可靠性)。中性看待:(1) EM 作奖励对"语义正确但表述不同"会误判,可能压低真实能力上限并诱导格式拟合(论文未深究 reward hacking);(2) in-domain 仅 NQ/HotpotQA 两源,OOD 增益虽存在但训练分布仍偏窄;(3) 24%/41% 双口径属表述瑕疵,非逻辑硬伤。

## 11. 残留问题 / 局限

- EM 结果奖励的语义盲区(同义/格式差异误判),可能限制上限并引入 reward hacking 空间。
- 检索质量(E5 + 2018 Wikipedia 静态语料)是上界约束;时效性、在线搜索噪声未在主实验充分压测。
- GRPO 长训坍缩问题只描述未给根因/修复方案(留作默认 PPO 规避)。
- 主文仅 Qwen2.5;跨架构(Llama)结果在扩展版(2505.15117),COLM 主文不含。
- 摘要(24%/20%)与贡献(41%/20%)口径不一,引用时须注明。

## 12. 开源代码与框架(链接+框架+代码可得性)

- 仓库:https://github.com/PeterGriffinJin/Search-R1(origin 已核,Tier A,~8.3M)。含仓库内自带 `verl/`(基于 veRL 的定制 fork,见 `VERL_README.md`=HybridFlow/veRL)、`train_ppo.sh`、`train_grpo.sh`、`retrieval_launch.sh`、`search_r1/{llm_agent, search}`(核心逻辑)、`scripts/data_process`(NQ/HotpotQA 数据处理)、`infer.py`。
- 框架:**veRL**(仓库内置定制版);rollout 用 vLLM(0.5.x/0.6.x);支持 PPO、GRPO、REINFORCE 等(默认 PPO);检索支持本地稀疏(BM25)/稠密(E5+ANN)与在线搜索引擎。
- 复现:启动检索服务(`retrieval_launch.sh`)→ NQ+HotpotQA 构造带结构化标签 prompt → veRL 跑 PPO/GRPO rollout(遇 `<search>` 调检索、`<information>` 拼回)→ 仅对生成 token 计损(检索 token masked)→ EM 结果奖励 + KL 约束更新。
- 代码可得性:Tier A(完整可跑,社区多方复现)。

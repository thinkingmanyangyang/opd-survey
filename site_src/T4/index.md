# T4 · 思维链-Token 级 / MTP

共 7 篇。

- [beyond_loglik — Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum](beyond_loglik.md) — SFT 默认用 NLL（−log p），但它在"从零训练分类"才最优；后训练时基座已有先验。本文把 NLL 推广成参数族 f_α(p)=(1−p^α)/α，并提出一个统一刻画——**模型能力连续谱**：基座先验强（如数学）时，**下调低概率
- [llm_future_mtp — Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential](llm_future_mtp.md) — 实证发现自回归 LLM 给 prompt 附占位 token 后，正确的未来 token 已落在 top-200 logits 内；用 mask token + gated LoRA + sampler 把这种隐知识显式化为并行多 toke
- [rho1 — Rho-1: Not All Tokens Are What You Need](rho1.md) — 
- [segment_attrib — Segment-Level Attribution for Selective Learning of Long Reasoning Traces](segment_attrib.md) — 用 integrated gradients 直接量化每个 token 对"正确答案预测"的贡献，聚合为段落级"强度 + 方向一致性"两指标，挑出"高强度但中等一致性"的反思性段落做 loss-mask 选择性 SFT；相对 full-Co
- [srgen — Self-Reflective Generation at Test Time (SRGen)](srgen.md) — 零训练的测试时方法，在解码过程中用动态熵阈值识别"高熵 critical token"，在该处短暂暂停、在线优化一个瞬态修正向量 δ 注入 hidden state 再发射下一 token，实现"主动错误预防"；数学/AIME 类任务增益显
- [sstoken — ssToken: Self-modulated and Semantic-aware Token Selection for LLM Fine-tuning](sstoken.md) — SFT 的 token 级选择方法，用"当前模型 vs 其历史模型"的 Retrospective Excess Loss（REL）替代外部参考模型，再融合一个基于注意力的语义重要性分，按比例 ρ 保留 top-ρ token 计 loss
- [vcore — VCORE: Variance-Controlled Optimization-based Reweighting for Chain-of-Thought Supervision](vcore.md) — 把长 CoT SFT 的 token 加权形式化为"单步 SGD 下使期望 loss 下降最大、且加权分布与均匀分布 KL ≤ δ"的约束优化，闭式解为 Gibbs 分布 q\*∝exp(τ·gradient-utility)，配一个 on

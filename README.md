# 自进化 / 持续学习 LLM Agent · 论文精读手册

围绕 **探索-巩固(explore-consolidate)** 视角的系统性论文精读:On-Policy Distillation(含 +Tool/Agent)、统一 SFT-RL(GFT 类)、RLVR/GRPO、Agent 自进化、持续学习/防遗忘、思维链/Token 信用与前瞻(MTP)。时间窗以 **2025-06 ~ 2026-06** 为主。

- **117 篇有代码深读论文**,按 **L1–L6** 六条主题线组织;每篇含:
  一眼看懂(TL;DR + 最巧的一步)→ 为什么做(背景/痛点/相关工作/动机链/与最近邻Δ)→ 怎么做+靠不靠谱(流水线/逐组件必要性+消融核查/机制直觉/实验证据/假设与失效边界/祛魅)→ 结构化抽取(机制6轴 / 开源代码+框架 / 成本 / 对"探索-巩固"对标 / 开放问题)。
- 每篇附 **📄 原始论文链接** 与 **2 张关键图**(motivation + 方法/架构)。
- **防幻觉**:严格读真实 PDF 正文,三标注分离事实与判断 —— **【原文】** 直接来自论文 / **【推断】** 有据判断 / **【待核】** 拿不准项;关键论断带 §/图/表/页 锚点。

## 三种阅读方式
1. **在线(推荐)**:GitHub Pages 站点(本仓库 Actions 自动部署),手机/电脑浏览器打开,带全文搜索、L1–L6 导航、图片放大、暗色、公式 MathJax 渲染。
2. **离线单文件**:直接打开 `survey.html`(自包含,图与 MathJax 已内嵌,无需联网)。
3. **本地预览**:`pip install mkdocs-material mkdocs-glightbox && mkdocs serve` → `http://127.0.0.1:8000`。

## 部署到 GitHub Pages
```bash
# 已是 git 仓库且已提交;关联远端并推送:
git remote add origin https://github.com/<你的用户名>/opd-survey.git
git push -u origin main
# 仓库 Settings → Pages → Source 选 "GitHub Actions";1~2 分钟后:
# https://<你的用户名>.github.io/opd-survey/
```
(代理:`git config --global http.proxy http://127.0.0.1:7897` + `https.proxy` 同。)

## 目录
- `analysis_v2/` — 117 篇精读源(探索-巩固模板);`site_src/` — MkDocs 文档源(L1–L6 + 图);`mkdocs.yml` — 站点配置。
- `survey.html` — 离线单文件版。
- `_build/` — 构建脚本(build_site_v2.py / make_single_html.py / 抽图与选图脚本)。

> 内容为对公开论文的中立、批判性解读,事实尽量对照原文与开源代码核验,不确定处标 〔待核〕。

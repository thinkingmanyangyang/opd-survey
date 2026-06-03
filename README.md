# On-Policy Distillation 调研手册

围绕 **On-Policy Distillation(含 +Tool/Agent)** 为主轴,外加 **GFT 类统一 SFT-RL 微调** 与 **思维链推理(CoT/MTP)** 两条相关线的系统性文献调研。时间窗以 **2025-06 ~ 2026-06** 为主。

- **深读(有代码)论文:117 篇**,每篇 12 节解析 + 2 张关键图(① motivation ② 方法/架构)。
- 检视候选合计约 **660+** 篇(深读 117 + 附录候选池 ~550)。
- 每篇均经过**两层防幻觉核验**(元数据 WebSearch 交叉验证 + 内容层对照 PDF 全文与核心代码),并做过多轮二次审查纠错。

## 三种阅读方式

1. **在线(推荐)**:部署到 GitHub Pages 后,手机/电脑浏览器打开站点 URL,带全文搜索、分主题导航、图片点击放大、暗色模式。
2. **离线单文件**:直接打开 `survey.html`(自包含,229 张图已内嵌),手机浏览器即可看,无需联网。
3. **本地预览**:`pip install mkdocs-material mkdocs-glightbox && mkdocs serve` 然后访问 `http://127.0.0.1:8000`。

## 部署到 GitHub Pages(只需 3 步)

```bash
# 1) 在 github.com 新建一个空的【公开】仓库,例如 opd-survey(不要勾选 README)
# 2) 关联并推送(本目录已是 git 仓库、已提交)
git remote add origin https://github.com/<你的用户名>/opd-survey.git
git push -u origin main
# 3) 仓库 Settings → Pages → Build and deployment → Source 选 "GitHub Actions"
```
推送后 `.github/workflows/deploy.yml` 会自动用 mkdocs-material 构建并发布,
站点地址形如 `https://<你的用户名>.github.io/opd-survey/`。

## 目录

- `site_src/` — MkDocs 文档源(`index.md` 速览表 + `T1`–`T4` 四主题 × 每篇一页,含已选关键图)。
- `mkdocs.yml` — 站点配置(Material 主题、搜索、glightbox 图片放大、KaTeX 公式)。
- `survey.html` — 离线单文件版(图内嵌)。
- `SURVEY_on_policy_distillation.md` — 全部内容的单一长文 markdown(打印/导出用)。
- `PLAN.md` — 调研全流程与进度记录。

> 内容为对公开论文的中立、批判性解读;方法/数据集/数字均尽量对照原文与开源代码核验,不确定处标注 `〔待核〕`。

# 在线阅读与发布维护

本项目采用 **GitHub Pages 在线阅读站 + GitHub 仓库内容源**。GitBook 可继续连接同一份内容；GitHub Wiki 用作阅读入口。

- 在线阅读地址：https://honggui.github.io/sgd-net-science/
- 源文件仓库：https://github.com/honggui/sgd-net-science
- 发布记录：[Publish reading site](https://github.com/honggui/sgd-net-science/actions/workflows/pages.yml)

## 为什么采用这个方案

GitHub Wiki 的页面存储在独立 Git 仓库，不能把 GitBook 网站直接嵌入为 Wiki 的渲染界面。维护整套 Wiki 副本会增加章节链接、公式和目录同步的工作。本项目用 Pages 提供侧栏、站内搜索、深浅色和移动端阅读；Wiki 仅链接到阅读站。[GitHub Wiki 文档](https://docs.github.com/en/communities/documenting-your-project-with-wikis/adding-or-editing-wiki-pages)、[GitHub Pages 文档](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)

## 单一内容源

正文继续维护在 README.md、docs/、guide/ 和 architecture/。SUMMARY.md 同时提供 GitBook 与 Pages 的章节顺序，长文分章和既有锚点继续使用。Pages 的构建临时适配行内数学分隔符、相邻公式块间距，以及表格中竖线的等价 TeX 写法；不改写源文件。自动检查逐式核对源公式与生成页面。

Pages 使用 MkDocs Material 构建静态 HTML，MathJax 渲染公式，Mermaid 渲染图示。公式和图示的浏览器脚本使用固定版本的 jsDelivr CDN，因此首次阅读需要能访问该 CDN。站内搜索在浏览器运行，不上传查询。[数学渲染配置](https://squidfunk.github.io/mkdocs-material/reference/math/)

## 自动发布

推送到 main 后，`.github/workflows/pages.yml` 安装文档依赖、执行严格构建，再发布生成的 site/。GitHub 仓库 Settings → Pages 的 Source 应为 **GitHub Actions**。构建或部署失败时查看 Actions 日志；不得把失败的构建称为已上线。[GitHub Actions 发布 Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)

本地预览（Python 3.12 或兼容版本）：

```bash
python -m venv .venv-docs
# 激活虚拟环境后：
pip install -r requirements-docs.txt
python scripts/build_site.py
python scripts/check_site.py
python -m http.server 8000 --directory site
```

访问 `http://localhost:8000/`。检查[显示测试页](rendering-check.md)、长公式、复杂图示、章节前后页和搜索。site/ 与临时构建目录不提交。修改文件后运行 `python scripts/update_manifest.py` 更新发布清单；提交前检查 `git status` 中的文件均属于公开发布范围。

## Wiki 入口

Wiki 功能已启用时，如果 `.wiki.git` 仍不可克隆，需要仓库所有者在 GitHub Wiki 网页创建并保存第一张页面。GitHub 没有用于创建 Wiki 页面的常规 REST 接口。

可在 [Wiki 首页](https://github.com/honggui/sgd-net-science/wiki) 使用仓库中 [Home.md 入口模板](https://github.com/honggui/sgd-net-science/blob/main/wiki-entry/Home.md) 的内容。初始化后，可克隆 `https://github.com/honggui/sgd-net-science.wiki.git`，通过普通 Git 更新 Home.md。入口模板本身不表示 Wiki 已发布。

## GitBook 与许可

现有 `.gitbook.yaml` 和 SUMMARY.md 保留，连接方法见 [GitBook 接入与维护](../GITBOOK_SETUP.md)。Pages 上线不等于 GitBook 账号已连接。

这次增加的是文档构建工具，不是 SGD-Net 的训练或推理实现。研究状态及内容许可仍分别以[研究状态](status.md)和[NOTICE](../NOTICE.md)为准。

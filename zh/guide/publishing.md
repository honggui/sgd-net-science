# 在线阅读与发布维护

本项目采用 GitHub Pages 双语阅读站，以 GitHub 仓库为内容源。GitBook 可连接同一份内容，GitHub Wiki 用作入口。

- [在线阅读首页](https://honggui.github.io/sgd-net-science/)
- [源文件仓库](https://github.com/honggui/sgd-net-science)
- [自动发布记录](https://github.com/honggui/sgd-net-science/actions/workflows/pages.yml)

## 1. 发布方式

GitHub Wiki 使用独立 Git 仓库，不直接承载 GitBook 的网站界面。本项目用 Pages 提供章节侧栏、搜索、语言切换、深浅色和移动端阅读；Wiki 只保存入口，避免维护整套副本。[GitHub Wiki 文档](https://docs.github.com/en/communities/documenting-your-project-with-wikis/adding-or-editing-wiki-pages)、[GitHub Pages 文档](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)

## 2. 内容源与渲染

zh/ 与 en/ 分别保存完整正文；两者使用相同编号路径。每个语言目录的 SUMMARY.md 决定本语言目录顺序。共享 architecture/ JSON、scripts/ 构建工具和 site-assets/ 样式只维护一份。

Pages 使用 MkDocs Material。构建时临时适配数学分隔符、相邻公式块和表格中的竖线，保持数学含义与源文件不变。MathJax 和 Mermaid 的浏览器脚本来自固定版本的 jsDelivr CDN，首次阅读需要能访问该 CDN。搜索在浏览器中运行。[数学渲染配置](https://squidfunk.github.io/mkdocs-material/reference/math/)

## 3. 自动发布与本地检查

推送到 main 后，Actions 先检查双语对应、公式和译文版本，再构建两种语言并检查生成链接，最后发布 site/。Settings → Pages 的 Source 应为 GitHub Actions。失败的构建不会部署。[GitHub Actions 发布 Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)

本地使用 Python 3.12 或兼容版本：

```bash
python -m venv .venv-docs
# 激活虚拟环境后执行：
pip install -r requirements-docs.txt
python scripts/check_bilingual.py
python scripts/build_site.py
python scripts/check_site.py
python -m http.server 8000 --directory site
```

打开 `http://localhost:8000/`，检查两种语言的显示测试页、目录、搜索和对应章节切换。site/、虚拟环境与临时目录不提交。提交前执行 `python scripts/update_manifest.py` 并检查 git status。

## 4. 旧链接与 Wiki

i18n/legacy-paths.json 保存旧中文路径到新编号路径的映射。网站构建生成旧地址跳转，并保留浏览器中的章节锚点。仓库源文件已迁移到 zh/ 与 en/，新引用应使用新路径。

Wiki 未初始化时，需在 [Wiki 网页](https://github.com/honggui/sgd-net-science/wiki) 保存首张页面，之后才能克隆独立 Wiki 仓库。可使用 [Home.md 模板](https://github.com/honggui/sgd-net-science/blob/main/wiki-entry/Home.md)。模板存在不表示 Wiki 已完成发布。

## 5. GitBook 与研究状态

GitBook 的单空间双语目录和多空间语言变体见 [GitBook 接入](../GITBOOK_SETUP.md)。Pages 上线不表示 GitBook 账号已绑定。

这里新增的是文档发布工具，不是 SGD-Net 训练或推理实现。研究状态、许可和双语同步分别见[研究状态](status.md)、[NOTICE](../NOTICE.md)与[双语维护](bilingual.md)。

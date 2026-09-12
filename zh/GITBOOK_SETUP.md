# GitBook 接入与维护

**本页目录**

- [1. 仓库与双语结构](#section-001)
- [2. 连接 GitBook](#section-002)
- [3. 阅读入口](#section-003)
- [4. 公式、图示与目录](#section-004)
- [5. 后续更新](#section-005)

---

## 1. 仓库与双语结构 <a href="#section-001" id="section-001"></a>

仓库为 [honggui/sgd-net-science](https://github.com/honggui/sgd-net-science)，使用 main 分支。zh/ 与 en/ 分别保存完整中文和英文版，编号路径一一对应。根 README.md 是双语入口，根 SUMMARY.md 同时收录两种语言；每个语言目录还有自己的 README.md、SUMMARY.md 和 `.gitbook.yaml`。[GitBook 内容配置](https://gitbook.com/docs/docs-as-code/git-sync/content-configuration)

## 2. 连接 GitBook <a href="#section-002" id="section-002"></a>

有两种可选配置：

1. **一个空间展示双语目录**：Git Sync 连接仓库根目录，使用根 `.gitbook.yaml`。访客从中文与 English 两组目录进入。
2. **两个空间作为语言变体**：分别以 zh/ 和 en/ 为内容目录连接两个空间，再在同一文档站点中设置语言变体。每个空间使用本语言目录中的 SUMMARY.md。此方式适合希望使用 GitBook 原生语言切换的站点。

首次同步以 GitHub 为来源，避免空空间覆盖仓库内容。不要同时重复设置相同的根目录前缀；根据 GitBook 当前界面检查 Project directory 和 Content mapping 的最终路径。[GitHub Sync 官方步骤](https://gitbook.com/docs/docs-as-code/git-sync/enabling-github-sync)、[GitBook 语言变体](https://gitbook.com/docs/guides/content-organization-and-localization/localize-your-docs-with-variants-in-gitbook)

GitBook 账号连接和云端发布需要在对应账号中完成。本仓库保留可同步配置，但 Pages 发布不表示 GitBook 云端已绑定。

## 3. 阅读入口 <a href="#section-003" id="section-003"></a>

[在线阅读首页](https://honggui.github.io/sgd-net-science/) 提供中英文入口，当前主要阅读站由 GitHub Pages 发布。若另行发布 GitBook，使用其实际生成的网址补充入口。部署与 Wiki 入口见[发布维护](guide/publishing.md)。

## 4. 公式、图示与目录 <a href="#section-004" id="section-004"></a>

两种语言均保留 GitBook 的双美元数学标记、稳定的显式锚点和 Mermaid 代码块。英文翻译保留公式、组件 ID 和参考链接；图示中的显示文字翻译为英文。首次导入检查[显示测试页](guide/rendering-check.md)，再抽查长公式与复杂图。[数学格式](https://gitbook.com/docs/create-content/blocks/math-and-tex)、[Mermaid 格式](https://gitbook.com/docs/create-content/blocks/mermaid-blocks)

13、36 的编号页面是章节导读，正文在 docs/chapters/。每种语言的 SUMMARY.md 只登记本语言页面一次。

## 5. 后续更新 <a href="#section-005" id="section-005"></a>

正文、目录和对应译文一起维护。GitBook 双向同步时处理合并差异；不要用旧版长文覆盖分章入口。更新双语校验记录与发布清单的方法见[双语维护](guide/bilingual.md)。

共享组件 JSON 位于仓库根 architecture/。阅读目录中的组件表是对应语言的可读解释。对于按语言分开的 GitBook 空间，跨空间或共享资源链接可能需要根据实际空间地址配置；Pages 构建会自动处理共享 JSON 资源。

不要提交本地内部审查记录、ZIP、虚拟环境或构建产物。许可状态见 NOTICE；本地检查不替代 GitBook 云端预览。

---

[← 上一页](NOTICE.md) · [全书目录](SUMMARY.md)

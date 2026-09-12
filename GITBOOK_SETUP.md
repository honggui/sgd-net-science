# GitBook 接入与维护

**本页目录**

- [1. 放入 GitHub 项目](#section-001)
- [2. 连接 GitBook](#section-002)
- [3. 给 GitHub 访问者提供入口](#section-003)
- [4. 公式、图示与目录](#section-004)
- [5. 后续更新](#section-005)

---

## 1. 放入 GitHub 项目 <a href="#section-001" id="section-001"></a>

目标仓库为 [honggui/sgd-net-science](https://github.com/honggui/sgd-net-science)，文档位于仓库根目录，使用 `main` 分支。解压包中的文件可用于更新该仓库。若已有 README、LICENSE 或其他文档，先合并差异，不直接覆盖。必须包含以点开头的 `.gitbook.yaml`。

本配置使用当前目录作为内容根：README.md 是首页，SUMMARY.md 是侧栏。若整包放进已有项目的子目录，例如 `book/`，在 GitBook 同步设置中把项目/空间内容目录映射到 `book/`；不要把根目录和内容映射重复叠加。[GitBook 内容配置](https://gitbook.com/docs/docs-as-code/git-sync/content-configuration)

## 2. 连接 GitBook <a href="#section-002" id="section-002"></a>

1. 在 GitBook 新建用于此项目的文档站点与空间，打开 Git Sync。
2. 连接有权访问目标仓库的 GitHub 账号，选择仓库和分支。
3. 初次同步以 **GitHub → GitBook** 为来源；目标空间若已有内容，先确认应合并还是替换。
4. 本项目使用仓库根目录：Project directory 留空或选根目录；对应空间 Content mapping 选 `./`。`.gitbook.yaml` 的 `root` 已设为 `./`。
5. 执行同步，预览目录、公式与图示后，使用 GitBook 的发布功能生成公开地址。

账户连接与站点发布需要你的 GitBook/GitHub 登录。本包没有访问令牌，也没有完成云端绑定。[GitHub Sync 官方步骤](https://gitbook.com/docs/docs-as-code/git-sync/enabling-github-sync)

## 3. 给 GitHub 访问者提供入口 <a href="#section-003" id="section-003"></a>

发布成功后，将实际站点地址填写到 GitHub 项目的 About → Website，并在仓库 README 的阅读入口加入“在线阅读”链接。站点尚未创建时不要使用猜测的网址。本包的 README 和 SUMMARY 已能在 GitHub 内直接导航。

## 4. 公式、图示与目录 <a href="#section-004" id="section-004"></a>

本包将正文公式分隔符规范为 GitBook 的双美元标记，公式内容不改写。Mermaid 使用带 `mermaid` 语言标记的代码块；当前官方文档说明其为原生图块。先检查[显示测试页](guide/rendering-check.md)，再抽查论文与架构中的长公式和复杂图。[数学格式](https://gitbook.com/docs/create-content/blocks/math-and-tex)、[Mermaid 格式](https://gitbook.com/docs/create-content/blocks/mermaid-blocks)

章节小标题使用稳定的显式锚点；目录只登记每个文件一次。13、36 的原路径是导读入口，正文位于 `docs/chapters/`，其他文章链接仍可跳到对应入口。每章有返回全书和前后页导航。

## 5. 后续更新 <a href="#section-005" id="section-005"></a>

以本包的 Markdown 为持续维护源，正文和 SUMMARY 的变更一起提交。GitBook 与 GitHub 双向编辑时需处理合并，避免把旧版完整长文覆盖分章入口。修改 JSON 组件登记后，同步更新可阅读组件索引。

不要上传本地发布过程的内部审查报告、原始 ZIP 或历史备份。本包未选择新许可证，许可状态沿用 NOTICE。首次云端预览仍需检查实际渲染，本地结构检查不替代 GitBook 服务端验收。

---

[← 上一页](NOTICE.md) · [全书目录](SUMMARY.md)

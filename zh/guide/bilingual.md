# 双语文档结构与维护

中文版是当前研究文档的来源版本，英文版是完整译文。两种语言应表达同一组假设、方法、限制和验证计划。翻译与格式检查不等于独立学术评审，也不增加已实现能力。

## 1. 目录职责

| 路径 | 职责 |
|---|---|
| README.md / SUMMARY.md | GitHub 与单空间 GitBook 的双语入口 |
| zh/ | 中文正文、目录、指南和可读组件表 |
| en/ | 与中文编号路径一一对应的英文版 |
| architecture/ | 两种语言共用的组件登记 JSON |
| i18n/ | 旧地址映射、逐页译文对应记录 |
| scripts/ / site-assets/ | 共用的构建、校验与阅读界面 |
| wiki-entry/ | Wiki 首页入口模板 |

例如 `zh/docs/36.md` 与 `en/docs/36.md` 对应同一架构导读；`docs/chapters/36/section-20.md` 在两种语言下均表示同一章节。标题可翻译、修订，编号路径和显式锚点保持稳定。

## 2. 翻译约定

完整翻译正文、表格、图示文字和注释，不用摘要代替全文。保留 TeX 数学载荷、组件 ID、代码标识符、引用地址和显式锚点。图中的显示文字可以翻译，边连接和计算关系不能改变。

区分 research proposal、conditional guarantee 和 validated result；不把候选方法写成已验证能力。equivariance 表示等变，invariance 表示不变；SSM 内部状态递推不等于 implicit solve。生长、细化、观测创新、稳定性条件等术语与[术语表](glossary.md)一致。

## 3. 更新一对文档

1. 修改中文页面，并同步修改对应英文全文。
2. 若新增页面，同时更新两种语言的 SUMMARY.md；新页面使用相同相对路径。
3. 检查语义，运行结构检查。检查通过不能替代专业内容审阅。
4. 显式登记这对页面的译文版本，然后更新发布清单。

```bash
python scripts/check_bilingual.py --skip-record
python scripts/record_translations.py docs/36.md
python scripts/update_manifest.py
python scripts/check_bilingual.py
python scripts/build_site.py
python scripts/check_site.py
```

`record_translations.py` 的路径相对 zh/ 和 en/。只有确认译文同步后才执行；整版审阅可用 `--all`。i18n/translations.json 保存两种语言的文件哈希，CI 会拒绝未登记的变更，避免中文更新后英文长期静默过期。发布清单的更新不会自动认可译文同步。

## 4. 校验范围与局限

自动检查包括页面覆盖、公式载荷、标题层级数量、锚点、引用、代码块语言、中文残留与本地链接。构建后再核对公式与生成链接。浏览器检查覆盖公式、Mermaid、搜索、移动端和语言切换。

这些检查可以发现结构性遗漏，不能证明翻译的科学含义完全正确。重要术语和论证应继续接受领域专家审阅；修正时同步两种语言。

## 5. 发布与阅读

[在线阅读首页](https://honggui.github.io/sgd-net-science/) 提供中英文入口。两种语言各自有独立目录与搜索；页面语言切换保持相同编号章节。共享登记中的 ID 不因语言变化而更名。部署见[发布维护](publishing.md)，GitBook 配置见[接入说明](../GITBOOK_SETUP.md)。

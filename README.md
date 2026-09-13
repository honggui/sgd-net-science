# SGD-Net Science — Architecture Research for Scientific and Physical AI

**Graph structure, temporal memory, geometric constraints, and adaptive computation.**

**[Read in English](https://honggui.github.io/sgd-net-science/en/) · [阅读中文版](https://honggui.github.io/sgd-net-science/zh/)**

SGD-Net Science publishes research papers, a model component catalog, mathematical analyses, and proposals for scientific and physical AI. This is a research design project: no accompanying trained model, weights, or validated performance results are provided.

本项目公开 SGD-Net 的论文、模型组件、数学分析与科学/物理 AI 构建方案。当前属于研究设计，未附训练模型、权重或已验证的性能成果。

| English | 中文 |
|---|---|
| [Start here](en/guide/start-here.md) | [从这里开始](zh/guide/start-here.md) |
| [Full contents](en/SUMMARY.md) | [全书目录](zh/SUMMARY.md) |
| [Paper and research questions](en/docs/01.md) | [论文与研究问题](zh/docs/01.md) |
| [Unified architecture and components](en/docs/36.md) | [统一架构与组件全集](zh/docs/36.md) |
| [Transformer comparison](en/guide/transformer-comparison.md) | [Transformer 算子与架构对比](zh/guide/transformer-comparison.md) |
| [Research status](en/guide/status.md) | [研究状态](zh/guide/status.md) |
| [Component registry](en/architecture/README.md) | [组件登记](zh/architecture/README.md) |

The two complete editions share stable document IDs, equations, and citations. The 33 research documents include long works divided into chapters. The component registry contains 72 top-level units, 27 internal interfaces, and 56 domain units; these are design counts, not implementation claims.

中英文版保留一致的文档编号、公式与引用；33 篇研究文档中的长文已分章。组件登记包含 72 个顶层单元、27 个内部接口与 56 个领域单元，这些数量不代表已实现能力。

## Repository layout / 目录结构

```text
zh/               Chinese edition / 中文版
en/               English edition / 英文版
architecture/     Shared component JSON / 共享组件登记
i18n/             Translation records and legacy URL map / 译文记录与旧地址映射
scripts/          Build and validation tools / 构建与检查工具
site-assets/      Reading interface / 阅读界面
wiki-entry/       Wiki gateway template / Wiki 入口模板
```

Edit corresponding zh/ and en/ pages together. CI checks translation versions, equations, and links before publishing both editions to GitHub Pages. Existing Chinese website URLs redirect to the new numbered paths. GitBook configurations remain available for a combined bilingual space or separate language variants.

修改正文时同步对应译文；CI 检查译文版本、公式和链接后发布双语 Pages 站点。旧中文网页地址会跳转到新编号路径。GitBook 可使用双语合并目录，也可配置独立语言变体。

- [Bilingual maintenance](en/guide/bilingual.md) · [双语维护](zh/guide/bilingual.md)
- [Publishing and Wiki](en/guide/publishing.md) · [发布与 Wiki](zh/guide/publishing.md)
- [GitBook setup](en/GITBOOK_SETUP.md) · [GitBook 接入](zh/GITBOOK_SETUP.md)
- [Licensing notice](en/NOTICE.md) · [许可说明](zh/NOTICE.md)

Translations are AI-assisted and structurally checked; this does not constitute independent scientific peer review. No new open-source, open-content, or patent license is granted by this publication.

译文经过 AI 辅助翻译与结构检查，不等同于独立学术评审。本次发布未新增开源、开放内容或专利许可。

# SGD-Net Science — Architecture Research for Scientific and Physical AI

**图结构、时间记忆、几何约束与自适应计算的模型架构研究。**

[GitHub 项目](https://github.com/honggui/sgd-net-science) · [开始阅读](guide/start-here.md) · [全书目录](SUMMARY.md)

这里公开 SGD-Net 的论文、模型组件、数学分析和领域构建方案。当前是研究设计文档，未附训练代码、模型权重或已验证的性能成绩。

**本页目录**

- [开始阅读](#section-001)
- [你可以在这里找到什么](#section-002)
- [阅读方式](#section-003)
- [附录与维护](#section-004)

---

## 开始阅读 <a href="#section-001" id="section-001"></a>

- [首次阅读：选择适合你的路线](guide/start-here.md)
- [全书目录](SUMMARY.md)
- [术语与阅读约定](guide/glossary.md)
- [研究状态与贡献边界](guide/status.md)

## 你可以在这里找到什么 <a href="#section-002" id="section-002"></a>

| 内容 | 推荐入口 |
|---|---|
| 研究问题与论文方法 | [论文与研究问题](docs/01-SGD-Net学术论文.md) |
| 统一组件与模型构建 | [统一架构与组件全集](docs/36-SGD-Net统一基础模型架构与组件单元全集20260912.md) |
| 层、算子与实现细节 | [层与算子详解](docs/13-SGD-Net模型架构与层算子详解.md) |
| 表达能力与误差分析 | [函数空间与数值误差](docs/48-SGD-Net函数空间算子逼近与约束解空间评估20260912.md) |
| 动态生长与持续学习 | [动态生长与持续演化](docs/46-SGD-Net先验后验驱动的动态生长与持续演化20260912.md) |
| 科学与物理 AI 应用 | [跨领域扩展路线](docs/35-SGD-Net跨领域扩展路线与科研任务映射20260912.md) |
| 可证伪实验与任务卡 | [任务卡与评估路线](docs/49-SGD-Net能力优先路线与数学评估任务卡20260912.md) |

## 阅读方式 <a href="#section-003" id="section-003"></a>

GitBook 侧栏按主题组织章节，13 与 36 两篇长文拆为独立章节，页面内提供目录和前后页导航。GitHub 用户可从本页或 SUMMARY.md 直接阅读同一份文档。

完整研究内容来自 33 篇公开文档；组件登记包含 72 个顶层单元、27 个内部接口与七领域的 56 个单元。配置与计数不代表已实现能力，也不要求每个任务同时启用全部组件。

## 附录与维护 <a href="#section-004" id="section-004"></a>

- [可阅读的组件登记](architecture/README.md)
- [GitBook 接入与维护](GITBOOK_SETUP.md)
- [公式与图示显示检查](guide/rendering-check.md)
- [许可说明](NOTICE.md)

文中原始文献用于界定已有研究与候选贡献；首次导入 GitBook 后请预览公式和图示，再发布站点。本包本身是可同步的内容源，不是已经上线的网址。

---

[全书目录](SUMMARY.md) · [下一页 →](guide/start-here.md)

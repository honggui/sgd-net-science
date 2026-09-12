# 公式与图示显示检查

本页用于发布 Pages 或接入 GitBook 后快速确认公式、图示和表格显示；不代表模型实验。

**本页目录**

- [数学显示](#section-001)
- [Mermaid 图示](#section-002)
- [表格显示](#section-003)

---

## 数学显示 <a href="#section-001" id="section-001"></a>

行内公式：$$z_{t+1}=Az_t+Bu_t$$。

$$
V(e)=e^TPe,\qquad P\succ0.
$$

如果看到美元符号原文而非公式，请按维护说明核对导入方式。本包采用 GitBook 当前文档描述的数学标记，TeX 内容保留原稿。

## Mermaid 图示 <a href="#section-002" id="section-002"></a>

```mermaid
flowchart LR
    A["阅读任务定义"] --> B["选择基础模型"]
    B --> C["检查假设与数据"]
    C --> D["设计独立验证"]
```

该图只描述阅读和研究步骤。正文原有 Mermaid 图的语法与字体可能受服务端版本影响，首次发布应检查论文、架构和实现章节。

## 表格显示 <a href="#section-003" id="section-003"></a>

| 变量 | 解释 |
|---|---|
| $$z_t$$ | 第 t 个时刻的状态 |
| $$V$$ | 指定误差的候选函数 |

[返回首页](../README.md)

---

[← 上一页](../architecture/README.md) · [全书目录](../SUMMARY.md) · [下一页 →](../NOTICE.md)

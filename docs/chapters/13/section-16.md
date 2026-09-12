# 14. Layer 9：任务输出头

**本页目录**

- [14.1 功能](#section-001)
- [14.2 节点级输出](#section-002)
- [14.3 图级输出](#section-003)
- [14.4 路径级输出](#section-004)

---

## 14.1 功能 <a href="#section-001" id="section-001"></a>

输出头将 SGD-Net 的最终图状态映射为任务结果。

## 14.2 节点级输出 <a href="#section-002" id="section-002"></a>

$$
\hat{y}_i=f_{head}(x_i)
$$

适合节点分类、局部物理量预测、残基风险预测。

## 14.3 图级输出 <a href="#section-003" id="section-003"></a>

$$
\hat{y}=f_{head}(\operatorname{Readout}(X))
$$

Readout 可为：

- mean pooling；
- sum pooling；
- attention pooling；
- set transformer；
- graph-level token。

## 14.4 路径级输出 <a href="#section-004" id="section-004"></a>

Safe LLM / Agent 场景中，可输出路径：

$$
path=(v_1,e_{12},v_2,\ldots,v_k)
$$

并给出解释：

```text
answer
supporting_path
confidence
blocked_paths
refinement_events
```


---

[← 上一页](section-15.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-17.md)

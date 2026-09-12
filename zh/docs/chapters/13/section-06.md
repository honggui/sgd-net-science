# 6. Layer 1：图构造层

**本页目录**

- [6.1 功能](#section-001)
- [6.2 边构造策略](#section-002)
- [6.3 kNN 图构造算子](#section-003)
- [6.4 半径图构造算子](#section-004)
- [6.5 事实可达图构造算子](#section-005)
- [6.6 工程接口](#section-006)

---

## 6.1 功能 <a href="#section-001" id="section-001"></a>

图构造层将编码对象变成图：

$$
\operatorname{GraphBuild}(Z_0) \rightarrow (X_0,E_0,A_0)
$$

图节点和边的定义由任务决定。

## 6.2 边构造策略 <a href="#section-002" id="section-002"></a>

常见边类型：

| 边类型 | 用途 |
|---|---|
| 几何邻近边 | PDE 网格、分子空间邻近、材料晶格 |
| 物理约束边 | 边界条件、守恒关系、接触关系 |
| 化学键边 | 分子、蛋白、配体结构 |
| 事实关系边 | 知识图谱、逻辑推理 |
| 时间依赖边 | 实验流程、状态演化 |
| 路由边 | token-expert、node-subtree |

## 6.3 kNN 图构造算子 <a href="#section-003" id="section-003"></a>

对几何点 $$p_i$$，可构造：

$$
A_{ij}=1 \quad \text{if } j\in \operatorname{kNN}(p_i)
$$

## 6.4 半径图构造算子 <a href="#section-004" id="section-004"></a>

$$
A_{ij}=1 \quad \text{if } \|p_i-p_j\|_2 < r
$$

## 6.5 事实可达图构造算子 <a href="#section-005" id="section-005"></a>

$$
A_{ij}=1 \quad \text{if } \operatorname{Relation}(i,j)\in \mathcal{R}_{allowed}
$$

## 6.6 工程接口 <a href="#section-006" id="section-006"></a>

```text
GraphBuilder.build(encoded_objects) -> GraphState
```


---

[← 上一页](section-05.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-07.md)

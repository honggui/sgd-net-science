# 12. Layer 7：拓扑自适应重构层

**本页目录**

- [12.1 功能](#section-001)
- [12.2 h-refinement 节点分裂](#section-002)
- [12.3 p-refinement 局部升阶](#section-003)
- [12.4 hp-refinement 混合策略](#section-004)
- [12.5 子图 refinement](#section-005)
- [12.6 工程接口](#section-006)

---

## 12.1 功能 <a href="#section-001" id="section-001"></a>

拓扑重构层根据后验误差改变图结构。

它回答的问题是：

> 图中哪些节点要分裂？旧节点是否隔离？新节点如何继承邻居？边如何更新？

## 12.2 h-refinement 节点分裂 <a href="#section-002" id="section-002"></a>

若节点 $$i$$ 被分裂为 $$i_L,i_R$$：

$$
\mathcal{V}'=\mathcal{V}\cup\{i_L,i_R\}
$$

邻接继承：

$$
\forall j\in\mathcal{N}(i),\quad A'_{i_Lj}=A'_{ji_L}=1,\quad A'_{i_Rj}=A'_{ji_R}=1
$$

左右子节点内部连接：

$$
A'_{i_Li_R}=A'_{i_Ri_L}=1
$$

旧节点隔离：

$$
A'_{i,:}=0,\quad A'_{:,i}=0
$$

## 12.3 p-refinement 局部升阶 <a href="#section-003" id="section-003"></a>

不增加图节点，只增强局部表达能力：

$$
Adapter_i \leftarrow \operatorname{IncreaseRank}(Adapter_i)
$$

例如增大局部 MLP hidden dim、增加 LoRA rank 或增加树深。

## 12.4 hp-refinement 混合策略 <a href="#section-004" id="section-004"></a>

- 如果误差来自空间局部奇异：执行 h-refinement；
- 如果误差来自局部函数复杂度：执行 p-refinement；
- 如果两者都有：执行 hp-refinement。

## 12.5 子图 refinement <a href="#section-005" id="section-005"></a>

对高误差子图 $$\mathcal{S}\subset\mathcal{V}$$：

$$
\mathcal{G}'=\operatorname{RefineSubgraph}(\mathcal{G},\mathcal{S})
$$

适合蛋白口袋、材料局部缺陷、事实图谱争议区域。

## 12.6 工程接口 <a href="#section-006" id="section-006"></a>

```text
TopologyRefiner.refine(state, error_report) -> RefineResult
```


---

[← 上一页](section-13.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-15.md)

# 9b. Layer 4b：等变几何传播层

**本页目录**

- [9b.1 功能](#section-001)
- [9b.2 EGNN 基础消息算子](#section-002)
- [9b.3 CrystalGraphBuilder](#section-003)
- [9b.4 工程接口](#section-004)
- [9b.5 适用边界](#section-005)

---

## 9b.1 功能 <a href="#section-001" id="section-001"></a>

`EquivariantGNNLayer` 用于分子、晶体、蛋白、PDE 粒子场、机器人等显式 3D 几何任务。它回答的问题是：

> 当输入整体旋转、平移或镜像时，模型输出是否按同样的几何变换同步变化，并保持物理规律不变？

普通 TopoGNN 只处理连接关系；EGNN 同时处理：

- 不变标量：原子类型、电荷、质量、距离平方、能量密度；
- 等变向量：坐标、速度、力、位移；
- 结构元数据：晶胞、周期边界、空间群操作、缺陷 mask。

## 9b.2 EGNN 基础消息算子 <a href="#section-002" id="section-002"></a>

对边 $$(i,j)$$，定义相对距离不变量：

$$
r_{ij}=\|\mathbf{x}_i-\mathbf{x}_j\|^2
$$

消息：

$$
m_{ij}=\phi_e(h_i,h_j,r_{ij},e_{ij})
$$

坐标等变更新：

$$
\Delta \mathbf{x}_i=\sum_{j\in\mathcal{N}(i)}(\mathbf{x}_i-\mathbf{x}_j)\phi_x(m_{ij})
$$

$$
\mathbf{x}_i'=\mathbf{x}_i+\Delta \mathbf{x}_i
$$

标量更新：

$$
h_i'=\phi_h(h_i,\sum_j m_{ij})
$$

若输入坐标整体变换为 $$R\mathbf{x}+t$$，输出满足：

$$
\mathbf{x}'(R\mathbf{x}+t)=R\mathbf{x}'(\mathbf{x})+t
$$

## 9b.3 CrystalGraphBuilder <a href="#section-003" id="section-003"></a>

材料和晶体任务应优先构造几何图状态：

```text
CrystalGraphBuilder.build(
  cif_or_structure,
  radius_cutoff,
  pbc=True,
  space_group=True,
  defect_policy=None
) -> GeometryGraphState
```

```text
GeometryGraphState
  node_scalar
  node_coord
  edge_index
  edge_attr
  lattice_vectors
  pbc_offsets
  space_group_ops
  defect_masks
  source_tags
```

## 9b.4 工程接口 <a href="#section-004" id="section-004"></a>

```text
EquivariantGNNLayer.forward(
  node_scalar,
  node_coord,
  edge_index,
  edge_attr=None,
  lattice_vectors=None,
  pbc_offsets=None,
  vector_field=None
) -> node_scalar_new, node_coord_new, vector_field_new, geometry_trace
```

## 9b.5 适用边界 <a href="#section-005" id="section-005"></a>

| 适合启用 EGNN | 不必启用 EGNN |
|---|---|
| 分子/蛋白/配体 | 纯知识图谱 |
| 晶体/材料/半导体 | 文本事实图 |
| PDE 粒子场/网格点 | MoE token-expert 路由 |
| 机器人/刚体/点云 | 无显式几何的流程 DAG |


---

[← 上一页](section-09.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-11.md)

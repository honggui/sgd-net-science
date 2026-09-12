# 9c. Layer 4c：多度量/流形传播层（远期可选）

**本页目录**

- [9c.1 功能](#section-001)
- [9c.2 基础接口](#section-002)
- [9c.3 候选算子](#section-003)
- [9c.4 与 EGNN 的边界](#section-004)
- [9c.5 风险边界](#section-005)

---

## 9c.1 功能 <a href="#section-001" id="section-001"></a>

`Metric-aware Manifold Propagation Layer` 用于承接 `G-GeoS` 远期路线。它面向的不是普通三维坐标等变，而是层级、偏序、非对称可达性、混合曲率和外部世界模型 latent 到结构空间的度量映射。

典型适用场景包括：

| 场景 | 为什么需要多度量/流形 |
|---|---|
| 科学假设层级图 | 双曲或 Lorentz 几何更适合树状/层级结构 |
| 实验路径与子目标 | 从状态 A 到 B 的成本可能非对称，适合 quasi-metric |
| 知识图谱与偏序 cone | 上下位、包含、依赖关系可用偏序或双曲 cone 表达 |
| 混合拓扑图 | 同时存在层级、环和局部欧氏结构时可用 product manifold |
| WorldBridge / JSBO | 外部 latent 的度量需拉回到结构空间 |

## 9c.2 基础接口 <a href="#section-002" id="section-002"></a>

```text
MetricAwareManifoldLayer.forward(
  node_scalar,
  edge_index,
  edge_attr=None,
  manifold_state=None,
  metric_type=None,
  curvature=None,
  signature=None,
  asymmetry_flag=False
) -> node_scalar_new, manifold_state_new, metric_trace, manifold_trace
```

## 9c.3 候选算子 <a href="#section-003" id="section-003"></a>

| 算子 | 用途 |
|---|---|
| `hyperbolic_distance` | 层级图或树状结构距离 |
| `lorentz_inner_product` | Lorentz 模型中的稳定双曲内积 |
| `exp_map` / `log_map` | 流形与切空间映射 |
| `mobius_add` | 双曲空间中的向量组合 |
| `parallel_transport` | 流形上状态或梯度迁移 |
| `metric_pullback` | 将 latent 度量拉回结构空间 |
| `quasi_metric_gate` | 非对称可达性或方向性成本门控 |
| `reachability_distance` | 实验路径、因果链或子目标距离 |
| `curvature_estimate` | 局部曲率选择或混合曲率 gating |
| `magnitude_estimate` | 表征空间拓扑复杂度摘要 |

## 9c.4 与 EGNN 的边界 <a href="#section-004" id="section-004"></a>

| 能力 | EGNN / Geometry Layer | Metric-aware Manifold Layer |
|---|---|---|
| 主要对象 | 三维坐标、分子、晶体、粒子场 | 层级、偏序、可达性、混合曲率空间 |
| 核心对称性 | 平移/旋转/镜像等变 | 度量结构、曲率、非对称性或 chart 一致性 |
| 近期优先级 | P0/P1 | P2/P3 |
| Trace | `GeometryTrace` | `MetricTrace` / `ManifoldTrace` |
| 硬件路线 | Geometry Assist / Ada-GEO | G-GeoS 研究期权 |

## 9c.5 风险边界 <a href="#section-005" id="section-005"></a>

该层按任务对群、度量和流形的需求选择；软件模型实验可先行，不以 CModel 性能证明为前提。只有专用硬件固化需要 `MetricTrace` / `ManifoldTrace`、数值验证和端到端成本收益证据。


---

[← 上一页](section-10.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-12.md)

# 9c. Layer 4c: Multiple-Metric/Manifold Propagation Layer (Long-Term Option)

**On This Page**

- [9c.1 Function](#section-001)
- [9c.2 Basic Interface](#section-002)
- [9c.3 Candidate Operators](#section-003)
- [9c.4 Boundaries Relative to EGNN](#section-004)
- [9c.5 Risk Boundaries](#section-005)

---

## 9c.1 Function <a href="#section-001" id="section-001"></a>

The `Metric-aware Manifold Propagation Layer` supports the long-term `G-GeoS` roadmap. It targets hierarchies, partial orders, asymmetric reachability, mixed curvature, and metric mappings from external world-model latents to structural spaces, rather than ordinary equivariance of three-dimensional coordinates.

Typical application scenarios include:

| Scenario | Why multiple metrics/manifolds are needed |
|---|---|
| Scientific-hypothesis hierarchies | Hyperbolic or Lorentz geometry better suits tree-like/hierarchical structures |
| Experimental paths and subgoals | The cost from state A to B may be asymmetric, making a quasimetric appropriate |
| Knowledge graphs and partial-order cones | Superordinate/subordinate, inclusion, and dependency relations can be expressed by partial orders or hyperbolic cones |
| Mixed-topology graphs | A product manifold can be used when hierarchies, cycles, and local Euclidean structures coexist |
| WorldBridge / JSBO | External latent metrics need to be pulled back to structural space |

## 9c.2 Basic Interface <a href="#section-002" id="section-002"></a>

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

## 9c.3 Candidate Operators <a href="#section-003" id="section-003"></a>

| Operator | Uses |
|---|---|
| `hyperbolic_distance` | Distances in hierarchical graphs or tree structures |
| `lorentz_inner_product` | Stable hyperbolic inner products in the Lorentz model |
| `exp_map` / `log_map` | Mappings between manifolds and tangent spaces |
| `mobius_add` | Vector composition in hyperbolic space |
| `parallel_transport` | State or gradient transport on a manifold |
| `metric_pullback` | Pulling latent metrics back to structural space |
| `quasi_metric_gate` | Asymmetric reachability or directional-cost gating |
| `reachability_distance` | Distances for experimental paths, causal chains, or subgoals |
| `curvature_estimate` | Local-curvature selection or mixed-curvature gating |
| `magnitude_estimate` | Summary of representation-space topological complexity |

## 9c.4 Boundaries Relative to EGNN <a href="#section-004" id="section-004"></a>

| Capability | EGNN / Geometry Layer | Metric-aware Manifold Layer |
|---|---|---|
| Primary objects | 3D coordinates, molecules, crystals, particle fields | Hierarchies, partial orders, reachability, mixed-curvature spaces |
| Core symmetry | Translation/rotation/reflection equivariance | Metric structure, curvature, asymmetry, or chart consistency |
| Near-term priority | P0/P1 | P2/P3 |
| Trace | `GeometryTrace` | `MetricTrace` / `ManifoldTrace` |
| Hardware roadmap | Geometry Assist / Ada-GEO | G-GeoS research option |

## 9c.5 Risk Boundaries <a href="#section-005" id="section-005"></a>

This layer is selected according to the task's requirements for groups, metrics, and manifolds. Software-model experiments may proceed without prior CModel performance evidence. Only committing to dedicated hardware requires `MetricTrace` / `ManifoldTrace`, numerical validation, and evidence of end-to-end cost benefits.


---

[← Previous](section-10.md) · [Contents](../../../SUMMARY.md) · [Next →](section-12.md)

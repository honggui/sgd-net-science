# 1. Core Conclusion

**SGD-Net is an overall AI model architecture and also a composite model consisting of multiple pluggable submodules.**

More precisely:

> SGD-Net is neither an ordinary individual layer nor a single algorithm module. It is an overall neural architecture paradigm combining state-space evolution, graph-topology propagation, dynamic-tree adaptation, posterior error control, and stability projection.

Its hierarchy is analogous to that of a Transformer:

- The Transformer is the overall architecture;
- its internal modules include embedding, multi-head attention, FFN, LayerNorm, residual connections, and positional encoding.

The same applies to SGD-Net:

- SGD-Net is the overall architecture;
- its internal modules include `Input Graph Builder`, `SSM State Layer`, `GNN Topology Layer`, `Dynamic Tree Routing Layer`, `Posterior Error Estimator`, `Topology Refiner`, `Stability Projector`, and `Task Head`.

Following [New Idea Discussion_20260613_Equivariant GNNs and Self-Evolving Architectures](../../24.md) and [SGD-Harness Physical Conservation Sentinel and Equivariant Self-Evolution Loop](../../25.md), the recommended enhanced definition of SGD-Net is:

```text
SSM
+ Topological / Equivariant GNN
+ Geometry-aware Dynamic Tree
+ Physics Harness
+ Stability Projection
+ Retrospection
```

The ordinary `GNN Topology Layer` continues to serve purely topological tasks such as knowledge graphs, workflow graphs, and MoE routing. The added `EquivariantGNNLayer` serves molecules, crystals, PDE particle fields, robotics, and other tasks with explicit 3D geometry. `SGD-Harness` unifies posterior error, physical conservation, safety auditing, fallback, and retrospective-analysis entry points into a runtime conservation sentinel.

Following independent research materials (not included in this package), multiple-metric/non-Euclidean/quasimetric capabilities are further proposed as a long-term optional `Metric-aware Manifold Propagation Layer`. It does not replace TopoGNN or EGNN; instead, it is recorded and validated through `MetricTrace` / `ManifoldTrace` in hierarchical graphs, partially ordered knowledge spaces, experimental reachability, mixed-curvature representations, or world-model bridges.

Following independent research materials (not included in this package), AQAD's combination of quantization awareness, dynamic step sizes, static maximum shapes, dynamic masks, and early release is proposed for adaptation into SGD-Net Runtime mechanisms: `AdaptiveBudgetController`, `QuantizationAwareStateCache`, and `StaticMaxGraphMask`. These do not change SGD-Net's five core capabilities; they determine budgets and deployment forms for dynamic-tree refinement, SSM rollout, Harness rechecking, low-bit state caching, and fallback.

Following the latest synthesis in [SGD-Net and JEPA World-Model Integration and JSBO Bridge Operators](../../18.md), [SGD-Net Hybrid Solver and Training System and World-Model Research Roadmap](../../27.md), and independent research materials (not included in this package), a further system-level boundary is necessary: document `13` can serve as the **algorithm-kernel and operator SSOT**, but cannot by itself replace the hardware-system documents `11/16/30/31`, implementation plans `03-06`, the JSBO/LeJEPA theory document `18`, or the evidence contracts in `SGD-Trace与Benchmark规范/`. All scenarios in `00`–`31` should use layered support through an intrinsic core, extension modules, and external backends/peer IP.

Thus, the answer in one sentence is:

> SGD-Net is an overall AI model architecture whose implementable forms combine multiple submodules and layer operators; modules can be trimmed, replaced, or stacked according to the task.


---

[← Previous](../../13.md) · [Contents](../../../SUMMARY.md) · [Next →](section-02.md)

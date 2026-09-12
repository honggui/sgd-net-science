# 2. Architectural Boundaries of SGD-Net

## 2.1 What SGD-Net Is Not <a href="#section-001" id="section-001"></a>

SGD-Net is not merely:

1. an SSM/Mamba;
2. a GNN;
3. a differentiable decision tree;
4. an MoE router;
5. a postprocessor;
6. a PDE/PINN solver;
7. an AlphaFold/RFdiffusion plugin.

Any of these can be a component or application scenario of SGD-Net, but none alone represents SGD-Net.

## 2.2 What SGD-Net Is <a href="#section-002" id="section-002"></a>

SGD-Net is a composite neural architecture whose minimum definition includes five necessary capabilities:

| Capability | Corresponding module | Core requirement |
|---|---|---|
| Long-range state evolution | SSM / Mamba / recurrent state operator | Required |
| Topology/geometry-constrained propagation | Topological GNN / Equivariant GNN / message passing / graph operator | Required |
| Local adaptive routing | Dynamic Tree / geometry-aware grow-shrink / adaptive routing | Required |
| Posterior-error guidance | Posterior Error Estimator / SGD-Harness residual checker | Required |
| Stability constraints | Stability Projector / Lyapunov constraint | Required |

Keeping only SSM + GNN without a dynamic tree and posterior-error guidance yields something closer to a `State-space Graph Network`, rather than a complete SGD-Net.

Keeping only GNN + dynamic trees without an SSM omits long-range state evolution.

Screening errors only once at the output, without feeding them back to topology reconstruction or local refinement, likewise does not constitute a complete SGD-Net.


---

[← Previous](section-01.md) · [Contents](../../../SUMMARY.md) · [Next →](section-03.md)

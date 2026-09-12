# 3. Overall Layered Architecture

SGD-Net can be divided into nine levels:

```mermaid
flowchart TB
    A[Input Data] --> B[Layer 0: Input Encoder]
    B --> C[Layer 1: Graph Construction Layer]
    C --> D[Layer 2: Node/Edge Embedding Layer]
    D --> E[Layer 3: SSM State Evolution Layer]
  E --> F[Layer 4a: GNN Topology Propagation Layer]
  E --> F2[Layer 4b: Equivariant Geometry Propagation Layer]
  E --> F3[Layer 4c: Metric-aware Manifold Propagation Layer]
    F --> G[Layer 5: Dynamic Tree Routing Layer]
  F2 --> G
  F3 --> G
    G --> H[Layer 6: Posterior Error Estimation Layer]
  H --> HH[Layer 6b: SGD-Harness Physics/Safety Sentinel]
  HH -->|error > threshold| I[Layer 7: Topology Refinement Layer]
  HH --> AB[Runtime: Adaptive Budget Controller]
  AB --> I
  AB --> E
    I --> J[Layer 8: Stability Projection Layer]
    J --> E
  HH -->|converged| K[Layer 9: Task Output Head]
```

## 3.1 Static Backbone and Dynamic Control Loop <a href="#section-001" id="section-001"></a>

SGD-Net contains two paths:

1. **Backbone forward path**: input → encoding → SSM → TopoGNN/EGNN → Tree Routing → output.
2. **Dynamic control loop**: posterior error / Harness residual → topology reconstruction → stability projection → graph-state writeback.

The backbone path resembles an ordinary neural network; the dynamic control loop is the main difference between SGD-Net and ordinary SSMs/GNNs.


---

[← Previous](section-02.md) · [Contents](../../../SUMMARY.md) · [Next →](section-04.md)

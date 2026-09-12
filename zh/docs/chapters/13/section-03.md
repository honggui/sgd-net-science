# 3. 整体分层架构

SGD-Net 可分为 9 个层级：

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

## 3.1 静态主干与动态控制环 <a href="#section-001" id="section-001"></a>

SGD-Net 包含两条路径：

1. **主干前向路径**：输入 → 编码 → SSM → TopoGNN/EGNN → Tree Routing → 输出。
2. **动态控制环**：后验误差 / Harness 残差 → 拓扑重构 → 稳定投影 → 回写图状态。

主干路径类似普通神经网络；动态控制环是 SGD-Net 与普通 SSM/GNN 最大的差异。


---

[← 上一页](section-02.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-04.md)

# 26. Design Invariants

An SGD-Net implementation must maintain the following invariants:

1. The number of node-feature entries matches the number of graph nodes;
2. `edge_index` must not reference nonexistent nodes;
3. Isolated nodes do not participate in message passing;
4. Every active node has at least one dynamic or shared tree;
5. Split events record parent-child lineage;
6. Stability projection must not change tensor shapes;
7. Graph structure is revalidated after refinement;
8. All dynamic operations can be disabled for ablation experiments.


---

[← Previous](section-27.md) · [Contents](../../../SUMMARY.md) · [Next →](section-29.md)

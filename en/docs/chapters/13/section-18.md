# 16. Complete Forward Procedure

## 16.1 Inference Forward Pass <a href="#section-001" id="section-001"></a>

```text
Input raw data
  ↓
InputEncoder
  ↓
GraphBuilder
  ↓
NodeEdgeEmbedding
  ↓
for each SGDBlock:
    SSMStateEvolution
    GNNTopologyPropagation
    DynamicTreeRouting
  ↓
PosteriorErrorEstimator
  ↓
if error exceeds threshold:
    TopologyRefiner
    StabilityProjector
    repeat or local refine
else:
    TaskHead
  ↓
Output prediction + explanation + refinement log
```

## 16.2 Training Forward Pass <a href="#section-002" id="section-002"></a>

There are two training approaches.

### Approach 1: Fixed-Topology Training <a href="#section-003" id="section-003"></a>

First disable dynamic splitting and train only SSM/GNN/Tree soft routing:

$$
\mathcal{L}=\mathcal{L}_{task}+\lambda\mathcal{L}_{topo}+\mu\mathcal{L}_{stability}
$$

### Approach 2: Periodic Dynamic Refinement <a href="#section-004" id="section-004"></a>

Perform a structural update every $$K$$ steps:

```text
for epoch:
  for batch:
    forward
    compute loss
    backward
    optimizer.step
  if epoch % K == 0:
    estimate posterior error
    refine topology
    project stability
```


---

[← Previous](section-17.md) · [Contents](../../../SUMMARY.md) · [Next →](section-19.md)

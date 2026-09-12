# 16. 完整前向流程

## 16.1 推理前向 <a href="#section-001" id="section-001"></a>

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

## 16.2 训练前向 <a href="#section-002" id="section-002"></a>

训练时存在两种方式。

### 方式一：固定拓扑训练 <a href="#section-003" id="section-003"></a>

先关闭动态分裂，只训练 SSM/GNN/Tree soft routing：

$$
\mathcal{L}=\mathcal{L}_{task}+\lambda\mathcal{L}_{topo}+\mu\mathcal{L}_{stability}
$$

### 方式二：周期性动态 refinement <a href="#section-004" id="section-004"></a>

每隔 $$K$$ 步执行一次结构更新：

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

[← 上一页](section-17.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-19.md)

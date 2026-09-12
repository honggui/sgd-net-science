# 25. Complexity Analysis

**On This Page**

- [25.1 SSM Complexity](#section-001)
- [25.2 GNN Complexity](#section-002)
- [25.3 Dynamic Tree Routing Complexity](#section-003)
- [25.4 Topology Refinement Cost](#section-004)

---

## 25.1 SSM Complexity <a href="#section-001" id="section-001"></a>

For length $$L$$ and state dimension $$D_h$$:

$$
O(LD_h)
$$

If every graph node has a state stream:

$$
O(NLD_h)
$$

## 25.2 GNN Complexity <a href="#section-002" id="section-002"></a>

Sparse-graph message passing:

$$
O(|E|F)
$$

Dense adjacency:

$$
O(N^2F)
$$

Implementations should therefore migrate to sparse graphs as early as possible.

## 25.3 Dynamic Tree Routing Complexity <a href="#section-003" id="section-003"></a>

If each tree has depth $$d$$:

$$
O(NdF_{cmp})
$$

Here $$F_{cmp}$$ is the dimension of the compared features; the cost is generally much smaller than that of a full MLP.

## 25.4 Topology Refinement Cost <a href="#section-004" id="section-004"></a>

If at most $$K$$ nodes split per operation, the node count increases by $$2K$$. One must specify:

- maximum node count;
- maximum tree depth;
- maximum splits per step;
- split cooldown period;
- pruning policy.


---

[← Previous](section-26.md) · [Contents](../../../SUMMARY.md) · [Next →](section-28.md)

# 4. Unified Notation and Tensor Definitions

## 4.1 Basic Graph State <a href="#section-001" id="section-001"></a>

Let the current graph be:

$$
\mathcal{G}_t=(\mathcal{V}_t,\mathcal{E}_t,X_t,E_t,A_t)
$$

where:

| Symbol | Meaning | Typical shape |
|---|---|---|
| $$N_t$$ | Current node count | scalar |
| $$F$$ | Node-feature dimension | scalar |
| $$D_e$$ | Edge-feature dimension | scalar |
| $$X_t$$ | Node-feature matrix | $$[N_t,F]$$ |
| $$E_t$$ | Edge-feature matrix | $$[|\mathcal{E}_t|,D_e]$$ |
| $$A_t$$ | Adjacency matrix | $$[N_t,N_t]$$ |
| `edge_index` | Sparse edge indices | $$[2,|\mathcal{E}_t|]$$ |
| $$H_t$$ | SSM hidden state | $$[N_t,D_h]$$ or $$[B,N_t,D_h]$$ |
| $$T_i$$ | Dynamic tree inside node $$i$$ | object |
| $$\eta_i$$ | Posterior error of node $$i$$ | scalar |

## 4.2 Batch Shapes <a href="#section-002" id="section-002"></a>

With a batch dimension, we can write:

$$
X \in \mathbb{R}^{B\times N\times F}
$$

However, dynamic graph growth makes $$N$$ vary across samples, so sparse batching similar to PyTorch Geometric is recommended in implementation:

```text
x:          [total_nodes, feature_dim]
edge_index: [2, total_edges]
batch:      [total_nodes]
```


---

[← Previous](section-03.md) · [Contents](../../../SUMMARY.md) · [Next →](section-05.md)

# 12. Layer 7: Adaptive Topology Reconstruction Layer

**On This Page**

- [12.1 Function](#section-001)
- [12.2 h-Refinement: Node Splitting](#section-002)
- [12.3 p-Refinement: Local Order Increase](#section-003)
- [12.4 hp-Refinement: Hybrid Strategy](#section-004)
- [12.5 Subgraph Refinement](#section-005)
- [12.6 Implementation Interface](#section-006)

---

## 12.1 Function <a href="#section-001" id="section-001"></a>

The topology reconstruction layer changes graph structure based on posterior error.

It addresses the question:

> Which graph nodes should split? Should old nodes be isolated? How should new nodes inherit neighbors? How should edges be updated?

## 12.2 h-Refinement: Node Splitting <a href="#section-002" id="section-002"></a>

If node $$i$$ is split into $$i_L,i_R$$:

$$
\mathcal{V}'=\mathcal{V}\cup\{i_L,i_R\}
$$

Adjacency inheritance:

$$
\forall j\in\mathcal{N}(i),\quad A'_{i_Lj}=A'_{ji_L}=1,\quad A'_{i_Rj}=A'_{ji_R}=1
$$

Internal connection between left and right children:

$$
A'_{i_Li_R}=A'_{i_Ri_L}=1
$$

Isolation of the old node:

$$
A'_{i,:}=0,\quad A'_{:,i}=0
$$

## 12.3 p-Refinement: Local Order Increase <a href="#section-003" id="section-003"></a>

Increase local expressiveness without adding graph nodes:

$$
Adapter_i \leftarrow \operatorname{IncreaseRank}(Adapter_i)
$$

For example, increase the local MLP hidden dimension, LoRA rank, or tree depth.

## 12.4 hp-Refinement: Hybrid Strategy <a href="#section-004" id="section-004"></a>

- If the error comes from a local spatial singularity, perform h-refinement;
- if the error comes from local function complexity, perform p-refinement;
- if both are present, perform hp-refinement.

## 12.5 Subgraph Refinement <a href="#section-005" id="section-005"></a>

For a high-error subgraph $$\mathcal{S}\subset\mathcal{V}$$:

$$
\mathcal{G}'=\operatorname{RefineSubgraph}(\mathcal{G},\mathcal{S})
$$

This suits protein pockets, local material defects, and disputed regions of factual graphs.

## 12.6 Implementation Interface <a href="#section-006" id="section-006"></a>

```text
TopologyRefiner.refine(state, error_report) -> RefineResult
```


---

[← Previous](section-13.md) · [Contents](../../../SUMMARY.md) · [Next →](section-15.md)

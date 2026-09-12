# 6. Layer 1: Graph Construction Layer

**On This Page**

- [6.1 Function](#section-001)
- [6.2 Edge Construction Strategies](#section-002)
- [6.3 kNN Graph Construction Operator](#section-003)
- [6.4 Radius Graph Construction Operator](#section-004)
- [6.5 Factual Reachability Graph Construction Operator](#section-005)
- [6.6 Implementation Interface](#section-006)

---

## 6.1 Function <a href="#section-001" id="section-001"></a>

The graph construction layer converts encoded objects into a graph:

$$
\operatorname{GraphBuild}(Z_0) \rightarrow (X_0,E_0,A_0)
$$

The task determines the definitions of graph nodes and edges.

## 6.2 Edge Construction Strategies <a href="#section-002" id="section-002"></a>

Common edge types:

| Edge type | Uses |
|---|---|
| Geometric proximity edges | PDE meshes, spatial proximity in molecules, material lattices |
| Physical constraint edges | Boundary conditions, conservation relations, contact relations |
| Chemical bond edges | Molecular, protein, and ligand structures |
| Factual relation edges | Knowledge graphs, logical reasoning |
| Temporal dependency edges | Experimental workflows, state evolution |
| Routing edges | token-expert, node-subtree |

## 6.3 kNN Graph Construction Operator <a href="#section-003" id="section-003"></a>

For geometric points $$p_i$$, construct:

$$
A_{ij}=1 \quad \text{if } j\in \operatorname{kNN}(p_i)
$$

## 6.4 Radius Graph Construction Operator <a href="#section-004" id="section-004"></a>

$$
A_{ij}=1 \quad \text{if } \|p_i-p_j\|_2 < r
$$

## 6.5 Factual Reachability Graph Construction Operator <a href="#section-005" id="section-005"></a>

$$
A_{ij}=1 \quad \text{if } \operatorname{Relation}(i,j)\in \mathcal{R}_{allowed}
$$

## 6.6 Implementation Interface <a href="#section-006" id="section-006"></a>

```text
GraphBuilder.build(encoded_objects) -> GraphState
```


---

[← Previous](section-05.md) · [Contents](../../../SUMMARY.md) · [Next →](section-07.md)

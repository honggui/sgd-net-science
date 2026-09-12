# 10. Layer 5: Dynamic Tree Routing Layer

**On This Page**

- [10.1 Function](#section-001)
- [10.2 Per-Node Tree Structure](#section-002)
- [10.3 Hard Routing Operator](#section-003)
- [10.4 Soft Routing Operator](#section-004)
- [10.5 Local Expert Operator](#section-005)
- [10.6 Split Operator](#section-006)
- [10.7 Pruning Operator](#section-007)
- [10.8 Geometric Grow/Shrink Control](#section-008)
- [10.9 Input and Output Shapes](#section-009)
- [10.10 Implementation Interface](#section-010)

---

## 10.1 Function <a href="#section-001" id="section-001"></a>

The dynamic tree layer is one of SGD-Net's core distinguishing features. It is a local adaptive routing structure embedded within each graph node or subgraph, rather than ordinary decision-tree postprocessing.

It addresses the question:

> For a given node, which local branch should the current features follow? Should a finer local representation space be split off?

## 10.2 Per-Node Tree Structure <a href="#section-002" id="section-002"></a>

Each graph node $$i$$ has a corresponding tree $$T_i$$:

```text
Graph Node i
  └── DynamicTree T_i
        ├── predicate: x[k] <= threshold
        ├── left child
        └── right child
```

## 10.3 Hard Routing Operator <a href="#section-003" id="section-003"></a>

For tree node $$q$$, let the split feature be $$k_q$$ and the threshold be $$\tau_q$$:

$$
route_q(x)=
\begin{cases}
left(q), & x_{k_q}\le \tau_q \\
right(q), & x_{k_q}>\tau_q
\end{cases}
$$

After reaching leaf $$\ell$$, the output is:

$$
\tilde{x}_i=x_i+b_{\ell}
$$

Here $$b_{\ell}$$ is the leaf's local bias or local-expert parameter.

## 10.4 Soft Routing Operator <a href="#section-004" id="section-004"></a>

For differentiable training, sigmoid gating can be used:

$$
g_q(x)=\sigma\left(\frac{x_{k_q}-\tau_q}{T}\right)
$$

$$
Tree_q(x)=(1-g_q(x))Tree_{left}(x)+g_q(x)Tree_{right}(x)
$$

Here $$T$$ is a temperature parameter. Routing approaches hard routing as $$T\to 0$$.

## 10.5 Local Expert Operator <a href="#section-005" id="section-005"></a>

A leaf can contain a local MLP / adapter instead of only a bias:

$$
\tilde{x}_i=x_i+\operatorname{Adapter}_{\ell}(x_i)
$$

## 10.6 Split Operator <a href="#section-006" id="section-006"></a>

When the posterior error $$\eta_i$$ exceeds the threshold, apply the following to node $$i$$'s tree:

$$
T_i \leftarrow \operatorname{Split}(T_i,k^*,\tau^*)
$$

Split features can be selected using:

1. the dimension with the largest gradient;
2. the dimension most correlated with the residual;
3. the dimension with the largest variance;
4. random exploration;
5. a specification from task priors.

In geometric tasks, dynamic trees must preferentially read invariant scalars to avoid breaking equivariance through direct dependence on absolute coordinates. Candidate geometric split features include:

| Feature | Uses |
|---|---|
| $$\|\mathbf{x}_i-\mathbf{x}_j\|^2$$ | Local tension/compression |
| $$\|\mathbf{F}_i\|$$ | Force anomalies |
| local energy density | Refinement of high-energy regions |
| residual norm | Error hotspots |
| strain / curvature scalar | Lattice defects, cracks, phase transitions |
| $$\|\dot{h}_i\|$$ | Abrupt changes in SSM hidden states |
| uncertainty scalar | Exploratory pre-splitting |

## 10.7 Pruning Operator <a href="#section-007" id="section-007"></a>

When a subtree has persistently low contribution or high risk:

$$
T_i \leftarrow \operatorname{Prune}(T_i, subtree)
$$

Pruning conditions include:

- low access frequency;
- persistently nondecreasing residuals;
- excessively low confidence;
- identification as poisoned/anomalous;
- exceeding the computation budget.

## 10.8 Geometric Grow/Shrink Control <a href="#section-008" id="section-008"></a>

For EGNN / crystals / PDEs / particle fields, dynamic trees can perform local refinement and coarsening analogous to adaptive finite elements in addition to abstract routing:

```text
if conservation_check_invalid_or_failed:
    reject_candidate_or_use_declared_fallback
elif observation_missing_or_noise_dominated:
    request_observation_or_keep_unknown
elif persistent_residual_high and capacity_or_resolution_cause_supported:
    propose_isolated_refinement_with_migration
elif residual_low and merge_consistency_valid and benefit_expected:
    propose_isolated_merge
else:
    keep
# All candidates are automatically accepted between windows after benefit, regression, numerical, resource, and version validation.
```

Correspondences:

| Adaptive FEM | SGD-Net geometric dynamic tree |
|---|---|
| Mesh refinement | graph grow / node split |
| Mesh coarsening | graph shrink / node merge |
| Posterior error estimation | Harness residual / posterior error |
| Stability repair | Stability Projection / rollback |

Geometric dynamic trees must record every grow/shrink operation in `ControlTrace` and `geometry_trace` for subsequent CModel replay and hardware-opportunity analysis.

## 10.9 Input and Output Shapes <a href="#section-009" id="section-009"></a>

```text
Input:
  x:          [N, F]
  node_trees: List[DynamicTree]
Output:
  x_routed:   [N, F]
  route_info: List[RouteEvent]
```

## 10.10 Implementation Interface <a href="#section-010" id="section-010"></a>

```text
DynamicTreeRoutingLayer.forward(x, node_trees) -> x_routed, route_info
DynamicTreeRoutingLayer.split(node_id, reason) -> SplitEvent
DynamicTreeRoutingLayer.prune(node_id, subtree_id, reason) -> PruneEvent
```


---

[← Previous](section-11.md) · [Contents](../../../SUMMARY.md) · [Next →](section-13.md)

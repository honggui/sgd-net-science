# 19. Minimum Implementable Version: MVP

A minimal SGD-Net need not implement every complex capability from the outset. The recommended MVP includes:

1. a graph state with a dense adjacency matrix;
2. simplified GNN: $$D^{-1}AXW$$;
3. simplified SSM: $$h_t=Ah_{t-1}+Bx_t$$;
4. a hard decision tree per node;
5. scaled L2 error after pairing predictions with results that have arrived;
6. representation-node refinement on isolated candidates, complete migration, and acceptance between windows;
7. SVD or norm-based numerical regularization, plus independent checks of stability conditions;
8. node-level regression output.

MVP forward pass:

```text
x_gnn = D^{-1} A X W_g
h_new = A_s h + B_s x_gnn
x_tree = TreeRoute(h_new)
graph.x = x_tree
y = head(graph.x)
record_prediction(y, versions)
report = paired_error(prior_prediction_record, matched_observation)
if persistent_error_has_supported_refinement_cause(report):
    enqueue_isolated_refinement_candidate(graph, report)
# Do not change the active topology within this forward pass; report UNKNOWN if no matched observation is available.
```


---

[← Previous](section-20.md) · [Contents](../../../SUMMARY.md) · [Next →](section-22.md)

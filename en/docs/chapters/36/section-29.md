# 27. Complexity, Resources, and Experimental Construction

**On This Page**

- [27.1 Accounting from Tensor Sizes Rather Than Module Counts](#section-001)
- [27.2 Minimum Reproducible Research Sequence](#section-002)
- [27.3 Acceptance Matrix](#section-003)

---

## 27.1 Accounting from Tensor Sizes Rather Than Module Counts <a href="#section-001" id="section-001"></a>

Ignoring batch size, a dense two-layer message MLP of width d per edge costs approximately O(E d²); node projections/FFNs cost O(Nd²), and vector geometry O(Eqn). Recalculate from GEMM shapes when actual dimensions differ. D04 elementwise recurrence costs O(Ns), but input/gate projections may cost O(Ns(d+d_m+d_c)); the entire temporal block cannot be called O(Ns). Multiply the corresponding cost by L layers.

First-order local-graph sparsity must come from actual neighborhoods; dense graphs can have E=O(N²). C03 graph-construction costs cannot be ignored. K IM iterations add roughly K times the relevant block computation, plus history/preconditioners and backward linear solves. U02 dense attention costs approximately O(N²d); higher-order graphs and multiple candidates further enlarge working sets.

State storage for z alone is LNs·bytes. Total capacity also includes h/v, edges, input encodings, weights, gradients, optimizers, candidates, experience, and solver temporaries. The earlier 128 KiB example in document 36 describes only a particular z configuration, not total chip requirements. Account for concurrent peaks using tensor lifetimes, sharing, and copy-on-write.

## 27.2 Minimum Reproducible Research Sequence <a href="#section-002" id="section-002"></a>

Stage 1 trains one task with fixed structure and checks units, identity, masks, time, and readouts. Stage 2 selects an evidence-supported bottleneck and enables the corresponding IM, U, or temporary refinement, comparing at equal budgets. Stage 3 adds EV continual learning and independently measures new-task benefits and old-task forgetting. Stage 4 uses actual operator traces for hardware mapping rather than adding hardware cores merely because of algorithm names.

Use [49 Task Cards](../../49.md) for small mathematical tasks: function fitting and aliasing, parameterized diffusion, piecewise dynamics, delayed queries, and concept changes. Refer to 37–43 for actual domains. Small tasks test mechanisms and do not replace evidence from real-world or biological tasks.

## 27.3 Acceptance Matrix <a href="#section-003" id="section-003"></a>

| Check level | Key tests | Supported conclusions |
|---|---|---|
| Data/structure | Units, identity, masks, cross-sample edges, periodic images | Whether input semantics and layouts are consistent |
| Neural forward pass | Output shapes, finite values, permutation/geometric-transform differences | Implementation consistency of the selected configuration |
| Gradients/solving | Finite differences, forward/backward residuals, condition numbers, failure paths | Numerical reliability of selected local computations |
| Adaptation/migration | P/Q readouts, version isolation, optimizer slots, fallback | Whether structural changes preserve agreed semantics |
| Task capability | Independent data, out-of-domain tests, calibration, old tasks, ablations | Quality and learning benefits for specified tasks |
| Deployment | Concurrent resources, deadlines, total bytes, peaks, energy consumption | Execution costs for specified backends/workloads |

Passing numerical examples does not prove superiority to the state of the art; sampled equivariance tests do not replace formal derivations. Report training budgets, data sources, failure rates, and unavailable capabilities rather than select only successes.



---

[← Previous](section-28.md) · [Contents](../../../SUMMARY.md) · [Next →](../../13.md)

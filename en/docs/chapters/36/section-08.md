# 7. Initial Configuration and Reproducible Forward Order

Shared debugging configuration: d=s=128, 4 blocks, edge hidden dimension 64, MLP hidden width 128, dropout=0, SiLU activation, and scalar LayerNorm; vector branches are enabled by domain. This is an experimental starting point, not selected optimal hyperparameters. N/E/H are specified in each document and may be changed by configuration; changes require recalculation. Parameters are not shared across layers by default, but are shared across time steps.

Single-inference order: A validation and event selection → B encoding and fusion → C graph construction → D02/D03 once per layer, D04/D05 once when needed, and D06/D07 → E output → F evaluation. Subject to H03, perform finite spatial refinement or candidate recomputation → H04 checks → H05 output. Candidate rollouts start from independent F08 states; E01 predictions advance hypothetical time using predicted observations or state-encoder inputs, without accessing real future sensors. When real observations arrive, correct again from the actual trajectory.

Training first fixes encoders, graphs, and budgets while training task heads and base blocks; then trains embodiment/domain adapters, followed by P/Q, and finally benefit selectors. Retain the preceding version as a baseline whenever adding a module. Static models do not fabricate temporal losses to satisfy the overall architecture.

A weight bundle includes enabled units, parameter dictionaries, encoder versions, graph definitions, normalization constants, symmetry modes, loss weights, training splits, random seeds, calibration, and runtime limits. Data IDs, current memory, and candidate states are stored in runtime bundles rather than model weights.

## 7.1 Training Parameters and Enablement Rules <a href="#section-001" id="section-001"></a>

Use AdamW as the initial optimization baseline for differentiable parameters. Initial exploration may start with a learning rate of 1e-3 for new heads, 1e-4 for backbone fine-tuning, weight decay 1e-2, and gradient-norm limit 1. All are tunable starting parameters, not experimental conclusions. Declare decay groups for normalization parameters and biases separately. Frozen encoders have no optimizer state; training with energy derivatives requires precision and second-order backpropagation validation. Validation performance and a predetermined budget govern training duration; do not tune on the test set.

A01–A08 form the data framework for every instance; G01/G02/G03/G07 are the training foundation; H03/H04/H07/H08 are the deployment foundation. Each domain document lists its B–F component selection; others are disabled by default. L01/L08 declare stability scope and evidence, with other L units selected according to document 44. Prediction models and check/correction models are recorded separately. Stage-specific components such as generation heads, budget controllers, P/Q, and JEPA start disabled and are enabled only for their corresponding training/ablation stages. H05/H06 are enabled only on paths requiring device/external actions; ordinary prediction reports require no device operation.

The minimum model-construction sequence is: choose tasks and labels → select observation encoders → define graphs and symmetry → determine whether real temporal data exist → select probability/property heads → train at a fixed budget → add adaptive modules according to evidence. Separate endpoint heads in domain documents instantiate E-class units and do not introduce new implicit base layers.

## 7.2 Basic Operations Underlying the Units <a href="#section-002" id="section-002"></a>

| Operation family | Specific operations | Main location |
|---|---|---|
| Dense transforms | embedding, Linear/GEMM, bias, SiLU, sigmoid, softplus, residual addition | B, D, E |
| Masks and sets | gather/scatter, segment sum/mean, masked softmax, concat, pooling | A07, B07, C08, D03, E08 |
| Geometry | Coordinate differences, inner products/norms, scalar-vector multiplication, periodic images, unit-cell transforms | C05/C06, D07, E05 |
| Time | Δt updates, elementwise exp/multiply-add, scan, state cache | A04, D04/D05, F08 |
| Structural adaptation | index remap, split/merge, prolong/restrict, local-boundary gather | C04, F04—F07 |
| Probability and training | Distribution-parameter transforms, NLL, sampling, automatic differentiation, EMA, gradient clipping | E, F02, G |
| Numerics and control | Finite iterations, optional projection/retraction, deadline checks, version comparisons | D08, F03, H |

This is a decomposition into algorithmic operators, not a chip instruction set; multiple backends may implement the same operation. Complete operator instances, parameter counts, and FLOPs can only be extracted from the actual computational graph after domain configurations and upstream encoders are fixed.


---

[← Previous](section-07.md) · [Contents](../../../SUMMARY.md) · [Next →](section-09.md)

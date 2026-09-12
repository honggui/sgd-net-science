# Model component index

Shared JSON retains the original fields and provides English names, responsibilities, and operation descriptions under `translations.en`; IDs and status do not change with the language.

This page provides a searchable component registry. See the [unified architecture](../docs/36.md) for complete equations, construction details, and conditions.

**On this page**

- [Top-level components](#section-001)
- [Implicit solving and evolution](#section-002)
- [On-demand capability extensions](#section-003)
- [Mathematical evaluation interfaces](#section-004)

---

## Top-level components <a href="#section-001" id="section-001"></a>

[Machine-readable JSON](../../architecture/component_registry.json)

| ID | Name | Inputs, outputs, or responsibility |
|---|---|---|
| A01 | Contract parsing | Raw records → typed samples |
| A02 | Unit and coordinate conversion | Values/units/coordinate frames → canonical physical quantities |
| A03 | Entity binding | Observed objects → stable IDs/matching confidence |
| A04 | Temporal scheduling | Multi-source timestamps → current events and Δt |
| A05 | Missingness and quality masks | Observations/QC → validity/noise cues |
| A06 | Domain and condition inputs | Embodiment/species/sample/boundary conditions → condition fields |
| A07 | Bounded packing | Variable-length nodes/edges → tensors, masks, and indices |
| A08 | Provenance binding | Samples and preprocessing → raw-data references |
| B01 | Visual/volumetric encoder | Images, video, microscopy volumes → local features |
| B02 | Sequence encoder | Proteins/nucleic acids/text → token features |
| B03 | Atom/entity encoder | Elements, types, physical attributes → node scalars |
| B04 | Signal encoder | Tactile/joint/electrophysiological/spectral signals → event features |
| B05 | Omics encoder | Counts/spatial omics → cell or gene features |
| B06 | Condition adapter | Task goals and domain conditions → modulation vectors |
| B07 | Masked multimodal fusion | Entity-bound features → unified node features |
| B08 | JSBO structural readout | Latent representations and measurements → physical/domain states and errors |
| C01 | Graph schema management | Domain configuration → node/edge types and attribute constraints |
| C02 | Fixed structural graph | URDF/bonds/reference mesh → static edges |
| C03 | Dynamic interaction graph | Current state → spatial/contact/interaction edges |
| C04 | Inference hierarchy graph | Current computational representation → parent–child mappings and hierarchy |
| C05 | Geometric edge features | Coordinates/vectors → distances, relative vectors, and invariants |
| C06 | Periodic boundary unit | Cell/fractional coordinates → periodic images and neighborhoods |
| C07 | Symmetry/chirality policy | Domain type → E(3)/SE(3)/non-geometric mode |
| C08 | Graph validity and sparse layout | Graph → valid indices/CSR or edge lists |
| D01 | Type-preserving normalization/projection | Scalar and vector channels → block inputs |
| D02 | Message generation | Adjacent nodes, edges, and conditions → edge messages |
| D03 | Message reduction | Edge messages → node aggregates |
| D04 | Entity-wise temporal SSM | Inputs, previous memory, and Δt → predicted memory |
| D05 | Observation-innovation correction | Predictions and observations at the same time → corrected memory |
| D06 | Node residual FFN | Inputs/messages/memory → new node representations |
| D07 | EGNN geometric readout/update | Scalar messages and relative vectors → vectors or computational coordinates |
| D08 | Metric/manifold slot | Specified geometry and tangent-space states → constrained representations |
| E01 | State-transition head | State, action, and conditions → future-state distribution parameters |
| E02 | Observation-likelihood head | Latent state → observation-distribution parameters |
| E03 | Contact/task-effect head | State and goal → contact probabilities and effects |
| E04 | Embodiment action head | Target effects, state, and embodiment → native action chunks |
| E05 | Physical-property head | Structural representations → energy/forces/stress/fields |
| E06 | Biological/function head | Conditional representations → expression, phenotypes, events, or endpoints |
| E07 | Conditional-generation head | Noise, conditions, and representations → candidate samples |
| E08 | Readout and pooling | Node/graph/path representations → task summaries |
| F01 | Error-evidence extraction | Innovations/constraints/tool discrepancies → decomposed error signals |
| F02 | Calibration and out-of-distribution detection | Predictive distributions/validation set → calibration parameters and OOD scores |
| F03 | Benefit and budget selection | Errors, resources, and candidate operations → computational actions |
| F04 | Dynamic-tree routing and growth candidates | Computational actions → subgraph/expert/candidate selection and structural-change proposals |
| F05 | Prolongation of state | Coarse memory and geometry → fine memory |
| F06 | Restriction of state | Fine memory and geometry → coarse memory |
| F07 | Local refinement execution | Subgraph and boundary context → local updates |
| F08 | Candidate-state versioning | Observed/imagined states → isolated snapshots |
| G01 | Loss assembly | Masked labels/outputs → component training losses |
| G02 | Data sampling and splitting | Data catalog → training/validation/test/calibration batches |
| G03 | Hybrid optimizer | Losses and parameter types → weight updates |
| G04 | JEPA teacher/predictor | Context and target views → latent-space targets |
| G05 | Replay and retrospective analysis | Failures, negative results, and historical tasks → retraining data |
| G06 | Adaptation/distillation/merging | Domain adapters and reference tasks → new checkpoints |
| G07 | Independent evaluation | Frozen configuration and test set → metrics and confidence intervals |
| G08 | Active experiment selection | Candidate pool and outcomes → next-batch suggestions |
| H01 | Domain-tool adaptation | Requests → DFT/MD/simulation/external-model results |
| H02 | ELN/LIMS/data retrieval | Identity and query → records or literature evidence |
| H03 | Hard-budget management | Configuration/timing → available budget or termination |
| H04 | Domain-output checks | Candidates → pass/violation/unknown and reasons |
| H05 | Commit and rollback | Checked candidates → reports/actions and ACK |
| H06 | Device and experiment orchestration | Authorized operations → device states/results |
| H07 | Compilation and numerical configuration | Operator graph → execution plan |
| H08 | Auditing and monitoring | End-to-end events → replayable evidence chain |
| L01 | Stability contract and reference | Task/dynamics/domain → errors, assumptions, and checking mode |
| L02 | V/storage-function head | Errors, graph, and domain → V and its gradient when applicable |
| L03 | Drift/descent evaluation | Current/candidate states, inputs, and V → conditional residuals |
| L04 | Disturbance and supply budgets | Input/error bounds → permitted supply and robustness margins |
| L05 | State/action correction | Candidates, constraints, and budget → correction or infeasibility status |
| L06 | Topology-switch compatibility | P/Q or mode resets → comparability of V and switching outcomes |
| L07 | Stability training and counterexamples | Data/candidate V → losses and counterexample samples |
| L08 | Evidence and failure handling | Conditional residuals and proof/sampling records → status and fallback |

## Implicit solving and evolution <a href="#section-002" id="section-002"></a>

[Machine-readable JSON](../../architecture/implicit_evolution_subunits.json)

| ID | Name | Inputs, outputs, or responsibility |
|---|---|---|
| IM01 | Equation assembly | Mode, inputs, and frozen graph/parameters → R(z;u,θ,G)=0 and dimensions |
| IM02 | Initialization and preconditioning | Explicit results or valid historical snapshots → initial guess and preconditioner |
| IM03 | Bounded forward solve | Residual operator → approximate solution, iteration count, residual, and termination reason |
| IM04 | Residual and conditioning diagnostics | Iteration trajectory → tolerance assessment, numerical status, and applicable error bounds |
| IM05 | Gradient interface | Approximate root and loss gradient → implicit or unrolled gradients and accuracy records |
| IM06 | Acceptance and failure paths | Approximate solution/diagnostics → acceptance, return to explicit baseline, or unknown |
| EV01 | Prediction snapshots and innovations | Pre-observation prediction + later outcome → paired error signal |
| EV02 | Persistence/drift diagnostics | Multi-window errors and QC → candidate causes, confidence, and triggers |
| EV03 | Structural candidate generation | Causes and resources → split/widen/add-edge/expert/no-change candidates |
| EV04 | Parameter and state migration | Old structure → candidate parameters, memory, masks, and optimizer state |
| EV05 | Local training and competition | New outcomes and old-task replay → parent/candidate performance comparison |
| EV06 | Stability and equivariance review | Candidate structure → structural, numerical, symmetry, and task evidence |
| EV07 | Persistent acceptance and monitoring | Candidates passing predefined thresholds → new versions and degradation monitoring |
| EV08 | Pruning/merging/distillation | Long-term usage and effect records → compressed versions or candidate withdrawal |

## On-demand capability extensions <a href="#section-003" id="section-003"></a>

[Machine-readable JSON](../../architecture/generality_extensions.json)

| ID | Name | Inputs, outputs, or responsibility |
|---|---|---|
| U01 | Typed computational graph | Distinguish data graphs from operator graphs; Tensor/State/Control edges, shape/dtype, parameter sharing, and branch/loop semantics |
| U02 | Global relations and ordering | Multi-head QKV, row-wise softmax, causal/cross masks, positional encoding, and global or sparse connectivity |
| U03 | Addressable working memory | Exact-address reads/writes, content retrieval, KV cache, stack/sequence buffers, lifetimes, and expansion policies |
| U04 | Dense and discrete primitive library | Activations, convolutions, normalization, comparison/selection, gating, and tensor rearrangements required by the target model |
| U05 | Higher-order and representation policy | State choices for nodes/edge pairs/subgraphs/hyperedges; scalars/vectors/tensors and group actions |
| U06 | Probabilistic generation and objective library | Specialized configurations for autoregressive distributions, noise schedules, sampling, and variational/adversarial/value/policy objectives |
| U07 | Adjustable constraints and execution semantics | Observation/correction modes; constraint domains; compatibility paths preserving target outputs; random-number and precision contracts |
| U08 | Constructive import and conformance testing | Target graph/parameter import, operator coverage, output/gradient/state differences, and cost reports |

## Mathematical evaluation interfaces <a href="#section-004" id="section-004"></a>

[Machine-readable JSON](../../architecture/functional_space_interfaces.json)

| ID | Name | Inputs, outputs, or responsibility |
|---|---|---|
| FN01 | Task-space contract | X/Y, measurement operators, constraints, regularity, norms, well-posedness, and objectives |
| FN02 | Sampling and reconstruction | P_h/R_h, coordinates, cell volumes/quadrature weights, and cross-resolution mappings |
| FN03 | Operator/query readout | Output fields/trajectories from function observations and query locations; add local or global kernels as required by the task |
| FN04 | Residual quality and stability factors | Residual norms, empirical/reliable status, and provenance of condition numbers or stability constants |
| FN05 | Error–cost adaptation | Map candidate actions to expected error reduction/cost, and accept only after validation |

---

[← Previous](../docs/49.md) · [Full contents](../SUMMARY.md) · [Next →](../guide/rendering-check.md)

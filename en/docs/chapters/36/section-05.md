# 4. Complete Component and Unit Set: 9 Groups, 72 Registered Units

Notation: T trainable; D deterministic operation; X external tool/system; O optional. Units marked T/D must fix the implementation choice in configuration. Training parameters, runtime states, and external data are stored separately. Input → output in the tables defines unit responsibility; tensor specifications are in Section 5.

**On This Page**

- [Group A: Scientific Inputs and Entity Management](#section-001)
- [Group B: Encoding and Latent Representation Bridges](#section-002)
- [Group C: Graph Structure and Geometry](#section-003)
- [Group D: Stackable Spatiotemporal Kernel](#section-004)
- [Group E: Task Output and Generation Heads](#section-005)
- [Group F: Bounded Adaptation and Uncertainty](#section-006)
- [Group G: Training and Scientific Research Loop (Not Counted as Inference Layers)](#section-007)
- [Group H: Tools and Runtime (Not Counted as Model Parameters)](#section-008)
- [Group L: Lyapunov Stability and Correction](#section-009)

---

## Group A: Scientific Inputs and Entity Management <a href="#section-001" id="section-001"></a>

| ID | Unit | Input → output | Type and internal operations |
|---|---|---|---|
| A01 | Contract parsing | Raw records → typed samples | D; field validation, schema version |
| A02 | Unit and coordinate conversion | Values/units/coordinate systems → canonical physical quantities | D; conversion and transformation records, retaining original values |
| A03 | Entity binding | Observed objects → stable IDs/matching confidence | D/T; association, birth/death, occlusion labels |
| A04 | Time scheduling | Multisource timestamps → current event and Δt | D; out-of-order policy, maximum observation age |
| A05 | Missingness and quality masks | Observations/QC → validity/noise indicators | D; do not fill missing observations with true-zero labels |
| A06 | Domain and condition encoding inputs | Embodiment/species/sample/boundary conditions → condition fields | D; separate identifier fields from learnable conditions |
| A07 | Bounded packing | Variable-length nodes/edges → tensors, masks, indices | D; reject overflow or explicitly partition |
| A08 | Provenance binding | Samples and preprocessing → raw-data references | X; versions, splits, experimental batches |

## Group B: Encoding and Latent Representation Bridges <a href="#section-002" id="section-002"></a>

| ID | Unit | Input → output | Type and internal operations |
|---|---|---|---|
| B01 | Visual/volumetric encoder | Images, videos, microscopy volumes → local features | T; CNN/ViT/3D models may be frozen |
| B02 | Sequence encoder | Protein/nucleic acid/text → token features | T; domain-pretrained models, sequence position independent of time |
| B03 | Atom/entity encoder | Elements, types, physical attributes → node scalars | T; embedding, MLP, unit normalization |
| B04 | Signal encoder | Tactile/joint/electrophysiological/spectral signals → event features | T; local filtering, convolution, channel projection |
| B05 | Omics encoder | Counts/spatial omics → cell or gene features | T; library-size/batch conditions, no fabricated pairings |
| B06 | Condition adapter | Task objectives and domain conditions → modulation vectors | T; MLP, FiLM, or small adapters |
| B07 | Masked multimodal fusion | Entity-bound features → unified node features | T; projection, gating, optional cross-attention |
| B08 | JSBO structural readout | Latent representations and measurements → physical/domain states and errors | T; supervised decoding and unit restoration, not automatic proofs of laws |

## Group C: Graph Structure and Geometry <a href="#section-003" id="section-003"></a>

| ID | Unit | Input → output | Type and internal operations |
|---|---|---|---|
| C01 | Graph schema management | Domain configuration → node/edge types and attribute constraints | D; distinguish measured, prior, and predicted edges |
| C02 | Fixed structural graph | URDF/bonds/reference meshes → static edges | D; joints, bond orders, reference topology |
| C03 | Dynamic interaction graph | Current state → spatial/contact/interaction edges | D/T; neighborhood search, edge classification, including graph-construction costs |
| C04 | Inference hierarchy graph | Current computational representation → parent-child maps and hierarchies | D; not the addition or disappearance of physical objects |
| C05 | Geometric edge features | Coordinates/vectors → distances, relative vectors, and invariants | D; rotation/translation transformation rules |
| C06 | Periodic boundary unit | Unit cells/fractional coordinates → periodic images and neighborhoods | D/O; multiple images, cutoffs, and smooth envelopes |
| C07 | Symmetry/chirality policy | Domain type → E(3)/SE(3)/nongeometric mode | D/T; chiral features or specialized equivariant blocks |
| C08 | Graph validity and sparse layout | Graph → valid indices/CSR or edge lists | D; cross-sample isolation, edge/node masks |

## Group D: Stackable Spatiotemporal Kernel <a href="#section-004" id="section-004"></a>

| ID | Unit | Input → output | Type and internal operations |
|---|---|---|---|
| D01 | Type-preserving normalization/projection | Scalar and vector channels → block inputs | T; scalar LayerNorm; vector scaling by invariant norms |
| D02 | Message generation | Neighboring nodes, edges, conditions → edge messages | T; invariant MLPs; geometric and nongeometric modes |
| D03 | Message reduction | Edge messages → node aggregates | D; sum/mean, masks, degree normalization |
| D04 | Entity-temporal SSM | Inputs, old memory, Δt → predicted memory | T/O; selective gates, discretization, per-entity recurrence |
| D05 | Observation innovation correction | Predictions and observations at the same time → corrected memory | T/O; residual projection, gating, disabled when missing |
| D06 | Node residual FFN | Inputs/messages/memory → new node representations | T; MLP, residuals, optional dropout |
| D07 | EGNN geometric readout/update | Scalar messages and relative vectors → vectors or computational coordinates | T/O; scalar weights times directions, neighborhood reduction |
| D08 | Metric/manifold slot | Specified geometry and tangent-space states → constrained representations | T/D/O; metrics, retractions, etc.; disabled by default |

## Group E: Task Output and Generation Heads <a href="#section-005" id="section-005"></a>

| ID | Unit | Input → output | Type and internal operations |
|---|---|---|---|
| E01 | State-transition head | States, actions, conditions → future-state distribution parameters | T; increments, mode probabilities, scales, or mixture distributions |
| E02 | Observation likelihood head | Hidden states → observation distribution parameters | T; continuous/count/event distributions, specified by domain |
| E03 | Contact/task-effect head | States and objectives → contact probabilities, effects | T/O; node/edge classification, geometric effect readout |
| E04 | Embodiment action head | Target effects, states, embodiment → native action chunks | T/O; action masks, time spans, and feasibility labels |
| E05 | Physical-property head | Structural representations → energy/forces/stress/fields | T/O; conservative forces may be generated by automatic differentiation of energy |
| E06 | Biological/functional head | Conditional representations → expression, phenotypes, events, or endpoints | T/O; domain-specific likelihoods, without mixing different endpoints |
| E07 | Conditional generation head | Noise, conditions, representations → candidate samples | T/O; flow/diffusion/conditional generation, with an independent algorithmic clock |
| E08 | Readout and pooling | Node/graph/path representations → task summaries | T/D; sum, mean, attention, selected according to extensivity |

## Group F: Bounded Adaptation and Uncertainty <a href="#section-006" id="section-006"></a>

| ID | Unit | Input → output | Type and internal operations |
|---|---|---|---|
| F01 | Error evidence extraction | Innovations/constraints/tool discrepancies → separate error signals | D/T; unit, provenance, and identifiability labels |
| F02 | Calibration and out-of-domain detection | Predictive distributions/validation sets → calibration parameters and out-of-domain scores | T/D; independent calibration; no variance promises without a distribution |
| F03 | Benefit and budget selection | Errors, resources, candidate operations → computational actions | T/D; fixed rules first, offline benefit fitting later |
| F04 | Dynamic tree routing and growth candidates | Computational actions → subgraph/expert/candidate selection and structural-change plans | D/T; explicitly configure three tree semantics |
| F05 | Prolong state expansion | Coarse memory and geometry → fine memory | T/D/O; parent-child mapping, mass weighting, or learned mapping |
| F06 | Restrict state aggregation | Fine memory and geometry → coarse memory | T/D/O; validate jointly with F05 |
| F07 | Local refinement execution | Subgraphs and boundary context → local updates | T/D/O; call shared blocks or explicit/implicit domain solvers |
| F08 | Candidate-state versions | Observed/hypothetical states → isolated snapshots | D; candidate, parent, graph_version, clock |

## Group G: Training and Scientific Research Loop (Not Counted as Inference Layers) <a href="#section-007" id="section-007"></a>

| ID | Unit | Input → output | Type and internal operations |
|---|---|---|---|
| G01 | Loss assembly | Masked labels/outputs → separate training losses | D; dimensional normalization, label applicability domains |
| G02 | Data sampling and splitting | Data catalogs → training/validation/test/calibration batches | D; subject/time/family grouping, leakage prevention |
| G03 | Hybrid optimizer | Losses and parameter types → weight updates | X; gradient optimization; projection/manifold methods only for specific constraints |
| G04 | JEPA teacher/predictor | Context and target views → latent-space targets | T/O; frozen or EMA teacher, stop-gradient targets |
| G05 | Replay and retrospective analysis | Failures, negative results, historical tasks → retraining data | X; separate real records from model-synthesized records |
| G06 | Adaptation/distillation/merging | Domain adapters and reference tasks → new checkpoints | T/X; parameter-compatibility checks, regression, negative-transfer evaluation |
| G07 | Independent evaluation | Frozen configurations and test sets → metrics and confidence intervals | X; ablations, cost, calibration, and out-of-domain evaluation |
| G08 | Active experiment selection | Candidate pools and results → next-batch suggestions | T/D/X; multifidelity, replicates, measurement costs, not automatic authorization |

## Group H: Tools and Runtime (Not Counted as Model Parameters) <a href="#section-008" id="section-008"></a>

| ID | Unit | Input → output | Type and internal operations |
|---|---|---|---|
| H01 | Domain tool adaptation | Requests → DFT/MD/simulation/external-model results | X; versions, fidelity, timeouts, error semantics |
| H02 | ELN/LIMS/data retrieval | Identities and queries → records or literature evidence | X; hidden memory cannot replace original files |
| H03 | Hard budget management | Configuration/timing → available quota or abort | D; reserve checking and fallback resources; do not trust predicted WCET |
| H04 | Domain output checks | Candidates → pass/violation/unknown and reasons | D/X; geometry, units, ranges, measurement validity periods |
| H05 | Commit and fallback | Checked candidates → reports/actions and ACKs | X; candidate versions, action prefixes, duplicate-request handling |
| H06 | Device and experiment orchestration | Permitted operations → device states/results | X; protocol constraints and external feedback |
| H07 | Compilation and numerical configuration | Operator graphs → execution plans | X; precision, static limits, sparse layouts, Aegis mapping |
| H08 | Auditing and monitoring | End-to-end events → replayable evidence chains | X; original references, weight, graph, calibration, and error versions |

## Group L: Lyapunov Stability and Correction <a href="#section-009" id="section-009"></a>

This restores the explicit responsibility of the original Layer 8 and extends it into a coordinated training, forward, and validation path. L01/L08 declare applicability and evidence; other units are enabled by domain. Not every prediction requires V to decrease.

| ID | Unit | Input → output | Type and execution location |
|---|---|---|---|
| L01 | Stability contract and reference | Task/dynamics/domain → errors, assumptions, and check mode | D; before the forward pass; without a definition, mark not applicable or unknown |
| L02 | V/storage-function head | Errors, graph, domain → V and gradients when applicable | T/D; alongside prediction heads, independently versioned |
| L03 | Drift/descent evaluation | Current/candidate states, inputs, V → conditional residuals | D; check temporal updates, refinement, and switching separately |
| L04 | Disturbance and supply budgets | Input/error bounds → allowable supply and robust margins | D/T; separate estimates from verified upper bounds |
| L05 | State/action correction | Candidates, constraints, budgets → corrections or infeasible status | D/T; line search, constrained optimization, or validated solvers |
| L06 | Topology-switch compatibility | P/Q or mode resets → V comparability and switching results | D/T; before graph-version switching; connected to F05/F06/F08 |
| L07 | Stability training and counterexamples | Data/candidate V → losses and counterexample samples | T/X; connected to G01/G03/G07, mainly offline |
| L08 | Evidence and failure handling | Conditional residuals, proof/sampling records → status and fallback | D/X; connected to H03/H04/H05/H08 |

See [44 Stability Explained](../../44.md) for mathematical definitions, EGNN symmetry, SSM feedback, graph-switching conditions, and configurations for the seven domains.


---

[← Previous](section-04.md) · [Contents](../../../SUMMARY.md) · [Next →](section-06.md)
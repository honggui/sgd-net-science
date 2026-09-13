# Graph Memory, Model Equivalence, and Mathematics-Driven Software and Hardware Planning

For subsequent design, see [memory–reasoning consolidation and parallel generation](memory-reasoning-parallel-generation.md): real-time state updates, long-term consolidation, evidence retention, and isolation between parallel candidates and actual experience.

> 2026-09-13. Further revision of the SGD-Net/Transformer comparison. This document distinguishes data representations, learned models, and execution systems. It proposes research contracts and functional planning, not new model experiments, universality theorems, or chip implementations. Existing top-level component IDs are unchanged.

## 1. Correction: graphs can hold history; SSM is not the only memory <a id="section-001"></a>

“Graphs provide spatial relations and SSM provides history” only describes a simplified configuration without persistent history. A fuller division is: **persistent temporal graphs organize relations and history; GNN/EGNN operations read, reason over, and update representations; SSM optionally supplies recurrent state, compression, and dynamic prediction.** A graph need not have geometry; coordinates, metrics, types, and group actions define geometry separately.

| Memory carrier | What it can retain | Conditions and limitations |
|---|---|---|
| Persistent nodes, edges, and attributes | Entities, events, relations, state snapshots, provenance | Explicit writes and versions; current adjacency alone does not preserve the past |
| Temporal edges and paths | Event ordering, transitions, associations between experiences | Define temporal direction, repeated events, and identity; a spatial path is not history or automatically a causal chain |
| Node/edge recurrent state z | Task-relevant history summaries and dynamic responses | SSM/RNN/gates are options; fixed precision and capacity generally cannot retain arbitrary history losslessly |
| Raw records and indexes | Exact event content and ID/time-based reads | Records must actually be retained; approximate semantic retrieval does not guarantee exact queries |
| Parameter and structural updates | Regularities learned across experiences and long-term priors | Learning/consolidation need data and validation; not an event-by-event archive |

Temporal graphs combined with memory modules have direct precedents such as TGN; storing history in graphs is not itself a new invention.[Temporal Graph Networks](https://arxiv.org/abs/2006.10637) SSM is also more than a compressor attached to a GNN: it can model sequences independently or form node, edge, or global dynamic states. The project's simplified D04 instance is not full Mamba.[Mamba](https://arxiv.org/abs/2312.00752)

A counterexample: retaining only “A and B are currently connected” cannot distinguish connect–disconnect–reconnect from uninterrupted connection. Greater GNN depth cannot recover information absent from its input. Representing each event as a persistent node with time and provenance distinguishes the histories; efficient access still depends on indexing and reasoning paths.

**Making history easier to retain and consolidate** comprises three claims: graphs may organize relational experience well; explicit retrieval can make it accessible; training and consolidation may turn experience into generalizable knowledge. The first does not establish the other two.

## 2. A multilayer memory contract for SGD-Net <a id="section-002"></a>

One candidate organization is:

$$
\mathcal M_t=(\mathcal G^{hist}_{\le t},Z_t,\mathcal R_{\le t},\theta_t,\nu_t),
$$
$$
z_{i,t}=\Phi_\theta(z_{i,t^-},\operatorname{Read}(\mathcal G^{hist}_{\le t},i),o_{i,t},\Delta t),
\qquad y_t=\Psi_\theta(q_t,\operatorname{Read}(\mathcal M_t,q_t)).
$$

Ghist is the persistent history graph, R retained raw records, Z compressed state, theta parameters, and nu the version closure. Read must specify exact indexing, range queries, or approximate retrieval. This is an organizational template, not a claim that everything resides on chip, is differentiable, or is implemented.

Assign responsibilities to existing components: C01/C08 defines temporal graph schemas, types, and layouts; current C02/C03 structures are separate from historical event views; U03 defines model-addressable working memory; H02 retrieves persistent records; D04/D05 predicts and corrects state; G05/EV handles replay and consolidation; H08 retains operational evidence. The historical data graph, model operator graph, and audit dependency graph are distinct objects, not interchangeable because they are all graphs.

The proposed event flow is: ingest and label provenance → retain events/records according to policy → retrieve the subgraph visible at query time → update graph representations and optional SSM state → produce an evidence-linked output → consolidate, compress, or forget in an isolated candidate version. Pre-observation predictions must still be saved before D05 innovation is computed; imagined candidate trajectories must not be written as actual experience.

Fields to specify include entity_id, event_id, event time, ingestion time, relation type, provenance, observed/predicted status, graph/parameter/state versions, retention period, index mode, and deletion rules. Event time differs from ingestion time. Replaying what was knowable then requires historical visibility, not timestamps added later that leak information. These are proposed field semantics, not a frozen binary ABI.

Capacity accounting includes nodes, edges, attributes, temporal indexes, raw records, Z, parameters, candidates, and temporary workspaces. Compressing arbitrary T-bit histories into B<T bits of all available memory without rereading causes at least two histories to collide. Distributing bits across topology does not evade this limit. Growing graphs can grow capacity, but also storage, retrieval, and maintenance costs.

## 3. Can it perform Transformer tasks? Separate levels of equivalence <a id="section-003"></a>

Let f be a specified target mapping on a finite input domain, and H_S/H_T the SGD/Transformer function families with declared resources and components. Fix input information, output types, history access, error norms, precision, and resources before discussing equivalence.

| Equivalence level | What can be established | What does not follow |
|---|---|---|
| Task substitution | Different structures may meet the required task error | Intermediate tensors need not match; no experiments cover all tasks |
| Function approximation | Ask whether the target lies in the closure of a specified function family | Parameter existence does not imply learnability from finite samples or acceptable cost |
| Exact computational construction | U02/U03/U04/U08 can construct a finite Transformer's execution if all target operators, states, and ordering are supported | Conditional compatibility, not current implementation or an exclusive SGD-core advantage |
| Finite-precision numerical conformance | Compare outputs, gradients, and states under declared tolerances | Equal real-number algebra does not imply bitwise floating-point equality |
| Learning and generalization | Compare samples, optimization, and out-of-domain risk by task | Equal representable sets do not imply equal optimization or generalization |
| Comparable cost | Compare time, memory, energy, and amortized training at equal task quality | Simulation does not imply efficiency; universality does not mean fastest on every workload |

The answer is therefore: **many Transformer tasks may be achievable; a fully implemented compatibility subset can also execute specified Transformers by construction; the default fixed local GNN plus finite SSM configuration has no all-task coverage guarantee.**

Two limitations are concrete. First, a node with only k local message layers and no global edge/readout/history-summary bypass cannot depend on new information more than k graph hops away; add communication paths, hierarchy, global relations, or iterations. Second, first-order aggregation without identity/position/higher-order features may fail to distinguish different graphs; depth or width alone does not automatically remove structural indistinguishability.[GNN expressivity](https://arxiv.org/abs/1810.00826) Information compression along long paths can also create bottlenecks.[Over-squashing](https://arxiv.org/abs/2006.05205)

Transformer universal-approximation results likewise have continuity, domain, position-encoding, function-class, and approximation-metric conditions. They do not guarantee solving all problems at arbitrary length with finite resources.[Transformer approximation research](https://arxiv.org/abs/1912.10077) Conditional operator-by-operator embedding of a finite computation into a larger system does not establish one fixed parameter set that solves every task.

## 4. Fitting, interpolation, and simulation perspectives <a id="section-004"></a>

For model family H and error threshold epsilon, study:

$$
e_{app}(f,\mathcal H)=\inf_{g\in\mathcal H}\|f-g\|_{\mathcal Y},\qquad
C_{\mathcal H}(\varepsilon;f)=\inf_{g\in\mathcal H:\|f-g\|_{\mathcal Y}\le\varepsilon}\operatorname{Cost}(g).
$$

C is infinite for an empty feasible set. Cost must specify parameters, traffic, time, or a weighted measure. Actual learners also incur observation, statistical, optimization, and numerical errors; representation error is not total error. See [48 mathematical assessment](../docs/48.md).

| Mathematical view | Transformer | Candidate SGD-Net benefits and limitations |
|---|---|---|
| Nonlocal data-dependent kernels | Attention produces query-dependent relation weights | U02 can use the same kernel; default local graphs save edges but may omit distant effects |
| Local operators and composition | Masks or training can express local relations | Graphs directly specify neighborhoods for local entity interactions; benefits depend on graph correctness |
| Symmetric function subclasses | Ordinary sequence configurations do not guarantee physical equivariance; structure can be added | EGNN restricts the family to selected equivariant maps, reducing redundancy when appropriate; incorrect symmetry excludes the target |
| History functions | Attention accesses explicit context; reading still does not guarantee lossless decoding | Graphs retain queryable events and SSM compresses task state; storage/retrieval costs remain |
| Piecewise or multiscale approximation | Hierarchies, experts, or other extensions are possible | C04/F/EV offers local refinement and experts; boundary continuity, migration, and search costs require validation |
| Interpolation | Sufficient capacity may fit finite samples | The same is possible; lower interpolation error does not establish better generalization |
| Equation/dynamics simulation | Learn surrogates or embed solver loops | IM/L/domain heads explicitly organize solving and constraints; model error and stability conditions remain necessary |

Attention and graph messages overlap: attention can be viewed as data-weighted message aggregation over allowed edges, with GAT providing an established graph-attention precedent.[GAT](https://arxiv.org/abs/1710.10903) General GNN edge MLPs, sum aggregation, and higher-order states are not individually identical to a standard attention layer; this does not make the model families unconditionally equal.

Finite samples x_j do not uniquely identify an unknown function:

$$
f_2(x)=f_1(x)+c\prod_{j=1}^{m}(x-x_j)
$$

It matches f1 at all samples but generally differs elsewhere. Both architectures need regularity, coverage, structural priors, or new observations to determine extrapolation; a more complex graph cannot recover unidentifiable laws from nothing. Exact interpolation should not be the sole objective for noisy data.

Function-to-function tasks can be described through sampling, reconstruction, and operator families; neural-operator research provides a direct foundation.[Neural Operator](https://www.jmlr.org/papers/v24/21-1524.html) The inverse solution operator of a local PDE can still be globally coupled, as in elliptic boundary-value problems. Local physical equations do not imply that a fixed, shallow local GNN accurately solves the whole domain. Multiscale processing, global interaction, or iterative solving may be needed.

Simulation also requires derivative, conservation, long-horizon trajectory, or distributional evaluation. Good position fits do not imply accurate forces; small equation residuals do not imply small solution errors without stability constants and conditioning. Pointwise long-term trajectories and statistics are different targets in chaotic systems.

## 5. Efficiency starts from equal quality, not complexity labels <a id="section-005"></a>

Standard dense attention interactions cost approximately O(T²d); graph messages can cost O(Ed²), and SSM state updates exclude projection, graph-building, and retrieval costs. See [operator comparison](transformer-comparison.md). Dense graphs, growing history, and additional solver iterations can make SGD more expensive. Regular matrix operations can outperform irregular accesses with fewer FLOPs.

For specified workloads and resource models, a planning approximation is:

$$
t_{exec}\gtrsim\max(F/P_{eff},D/BW_{eff},t_{dep}),\qquad
E_{exec}\approx e_FF+e_DD+E_{control}+E_{idle}.
$$

F counts executed operations, D actual bytes moved across a specified memory level, and tdep a dependency-path lower bound. Effective rates and energy coefficients require measurement. This is bottleneck analysis, not a hard-real-time proof.[Roofline](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)

A fair comparison reports task error, out-of-domain risk, constraint violations, training/pretraining sources, training/search budgets, end-to-end P50/P95/P99, peak total state, J/task, failures, and fallback behavior. Persistent graphs and raw records count toward SGD memory, just as KV and other caches count toward Transformer memory. External retrieval and host solving must remain in the accounting.

## 6. Foundations for software and chip planning <a id="section-006"></a>

Mathematical computation should be the foundation, but only the first layer. The proposed sequence is: **task and observability → functions/constraints/precision → representation and algorithm → data structures and access → compilation plan → hardware primitives → quality/cost evidence.** This abstraction avoids hardware organized around model names, but must return to real workloads to prevent a feature-rich yet inefficient design.

Software should retain multiple representations: task contracts; typed model/state graphs; kernels and solver plans; layouts/loops/communication; device execution. Lowering must preserve shapes, axis meanings, masks, temporal visibility, state effects, precision, and failure semantics. MLIR is an existing foundation for multilevel IR, but adopting it does not automatically provide project-specific semantics.[MLIR](https://mlir.llvm.org/docs/Rationale/Rationale/)

| Mathematical/computational family | Shared workloads | Functional planning focus |
|---|---|---|
| Dense contraction and projection | FFN, QKV, message MLPs, low-rank maps | GEMM/GEMV/batched matmul, accumulation precision, small-matrix utilization |
| Sparse relations and indexing | Graph messages, sparse attention, history retrieval | gather/scatter, CSR/COO, segment boundaries, duplicate indices, layout conversion |
| Reductions and nonlinearities | Softmax, normalization, probability and error metrics | max/sum, exp/log, rsqrt, stable evaluation, empty-set behavior |
| Recurrence and state transformation | SSM, filtering, prefix computation | affine/associative scan, state access, clocks and versions; not every recurrence admits an efficient scan |
| Addressable persistent memory | Event graphs, KV, indexes, replay | Bounded batched reads, append/invalidate, tiered storage, temporal filtering, snapshot isolation |
| Geometry and tensor operations | Distances, inner products, equivariant readouts, fields | Type-compatible contractions, norms, coordinates/units, degenerate inputs |
| Solving and differentiation | Implicit layers, physical correction, training | matvec, dot, AXPY, residuals and iteration control, compiled JVP/VJP compositions, precision fallback |
| Adaptive control and measurement | Expert routing, growth, candidate execution | compare/select, shape buckets, candidate resources, observable costs, version migration |

Do not create a non-reusable model-named instruction for each of memory, reflection, and world models. Graph-query database semantics, solver selection, differentiation rules, and graph-update transactions primarily belong in software. Hardware supplies composable batched computation, addressing, movement, synchronization, and measurement. Specialize fused kernels when workloads show repeated, material cost.

Addition and multiplication compose many computations, but primitive coverage alone does not establish effective acceleration: graph pointer traversal, synchronization, and layouts may dominate latency. Conversely, a large specialized kernel with fixed masks, head widths, or state layouts may restrict generality. Use programmable common primitives plus workload-validated fused paths.

## 7. Concrete project improvements and validation order <a id="section-007"></a>

This revision assigns graph-memory clarification to existing C/U03/H02/D04/G05/H08 responsibilities, without adding top-level components. Candidate planning is:

1. Define distinct lifecycles for persistent history graphs, raw records, compressed state, and parameter memory. Establish retention and exact reads before investigating compression and forgetting.
2. Use one typed relation interface for fixed graph aggregation, sparse attention, and dense attention, preserving different semantics and backends.
3. Specify differentiability boundaries, update order, version closure, and failure paths for recurrence, implicit solving, and persistent state. A finite budget must not silently change semantics by truncating history.
4. Measure shared primitives, then composed graph/attention/SSM/solver workloads, then real domain tasks. Use useful bytes, waiting, and quality to choose chip resource proportions.

| Research experiment | Question it distinguishes |
|---|---|
| Identical final adjacency with different event order; delayed exact indexed queries | What graph snapshots, persistent graphs, and compressed states retain |
| Change distant new inputs while varying graph distance and depth | Local propagation reachability and benefits of global relations/hierarchy |
| Smooth/high-frequency/noisy fitting with independent-point and derivative tests | Interpolation, observation aliasing, representation error, and generalization |
| Parameterized fields and nonlocal boundary-value solutions; cross-grid tests | Discretization, global coupling, operator generalization, and solver error |
| Contact switching, sparse events, long-running streaming queries | Net benefits of dynamic graph/SSM/attention hybrids |
| Different state layouts and fusion strategies at fixed task quality | Mathematical equivalence, numerical error, and hardware efficiency |

These experiments are planned, not executed. Compare fixed graphs, persistent graphs, graph+SSM, graph+attention, the selected full hybrid, and suitable Transformer baselines. Share observable information and account for retrieval, training, and maintenance. There is no basis for declaring one architecture universally superior before experimentation.

**The project should pursue composable, verifiable scientific-computing models and execution systems whose adaptation is guided by error and cost.** This includes foundational questions about computational objects, knowledge sources, and capability boundaries, while requiring mathematical conditions, executable contracts, and experimental evidence for each claim. It is not a proof of general intelligence or a universally optimal chip.

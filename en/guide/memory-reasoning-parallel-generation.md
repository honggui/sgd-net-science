# Memory–Reasoning Consolidation and Parallel Generation Design

> 2026-09-13. Discussion conclusions translated into composition contracts and experiment plans for existing components. No new top-level component IDs or implementation, training, or speed results are claimed. Analogies about unified knowledge and action are explanatory, not cognitive equivalence or constraint guarantees.

## 1. Design assessment and scope <a id="section-001"></a>

Dynamic trees, graph-representation growth, persistent history, recurrent state, replay, and distillation can compose an experience–memory–reasoning–consolidation loop. Potential benefits are local adaptation, relational experience retrieval, and converting repeatedly effective slow reasoning into fast paths. No experiments currently establish these benefits or superiority over Transformers.

Generation is task-defined, not fixed to token-by-token autoregression. E04 already defines action chunks, E07 a continuous flow-matching instance, and E08 entity/query readouts. Discrete diffusion language generation still needs dedicated training and sampling contracts. See [existing generation heads](../docs/chapters/36/section-21.md), [continuous evolution](../docs/46.md), and [graph memory foundations](mathematical-computing-foundation.md).

## 2. From experience to reusable capability <a id="section-002"></a>

| Stage | Existing owners | Object written or changed | Retention conditions |
|---|---|---|---|
| Record experience | C01/C08, U03, H02/H08 | Events, relations, provenance, times, addressable records | Label raw evidence separately from predictions/imagination |
| Update current estimates | D04/D05, EV01 | Node/edge states and prediction errors | Save pre-observation predictions; update magnitude is not true error |
| Diagnose and specialize | EV02/EV03, F03/F04 | Routing, local experts, refinement candidates | Distinguish missing data, noise, drift, solving, and insufficient capacity |
| Learn candidates | EV04/EV05, G05/G06 | Candidate parameters, structure, state, optimizer | New data and old-task replay; isolate the active version |
| Accept and monitor | EV06/EV07, L, H04/H08 | Complete new version and applicability scope | Independent evidence, constraints, old tasks, and budget checks |
| Organize and consolidate | EV08, G05/G06 | Merged, pruned, or distilled fast paths | Revalidate rare cases, boundaries, old tasks, and new data |

Event-level state updates, within-task temporary routing, window-level parameter learning, and intergenerational consolidation operate on different timescales. They are not all unconditionally real-time updates. Correct erroneous observations through provenance-linked amendments; do not overwrite historical evidence to fit a new model.

Adding computational nodes does not create real entities; pruning an expert does not delete its associated experiences. Data retention/deletion and model compression are separate policies. Distillation fits teacher behavior, and the teacher may be wrong; continue evaluating against observations, long-tail cases, and domain constraints.

One candidate slow-to-fast objective is:

$$
\mathcal L_{fast}=\lambda_d\,\mathbb E_{q\sim D_{replay}}d(F_\phi(q),T_\nu(q))
+\lambda_y\mathcal L_{observed}+\lambda_r\mathcal L_{retention}.
$$

Register teacher version nu, replay distribution, distance d, weights, and observed labels; the objective does not guarantee constraints. Average loss must not hide rare or consequential errors. Outside its validated scope or under uncertainty, a fast path can fall back to retrieval, expert competition, or solving. Evaluate internalization by cost reduction at matched task quality, not pruning counts or self-scores.

## 3. Coordinating constraints, memory, and action <a id="section-003"></a>

Applicable constraints can enter types, equivariant representations, training objectives, and candidate generation; memory can retain their conditions and exceptions. L/H04 independently checks required outputs. Learned regularities are approximations and cannot replace every execution condition. Lyapunov conditions apply only to specified systems, domains, and switching assumptions; growth or distillation requires rechecking applicability.

This coordination means consistency between what is known, its evidence, its applicability, and execution. It does not imply human experience or permanent constraint compliance after sufficient training. Shared version closure, error feedback, and checkable boundaries matter more than the analogy.

## 4. Distinguish autoregression, DiT, and diffusion language generation <a id="section-004"></a>

Autoregressive decoding generates from a prefix sequentially, but within-layer tensor computation and training on known target sequences are not wholly serial. Original DiT uses a Transformer backbone for image latent diffusion; it does not directly turn a language decoder into a one-pass generator.[DiT](https://arxiv.org/abs/2212.09748)

LLaDA uses discrete masking and reverse generation; Dream 7B iteratively denoises sequences in parallel. Each round can process multiple positions, while rounds remain dependent. Parallelism does not imply independent output positions, fewer rounds than tokens, or lower latency.[LLaDA](https://arxiv.org/abs/2502.09992), [Dream 7B](https://arxiv.org/abs/2508.15487)

The project can select direct joint readout, action chunks, continuous flow/diffusion, discrete masked iterations, or block autoregression. Each requires separate training, declared output dependencies, and evaluation. Removing a causal mask does not complete a diffusion-model conversion.

## 5. SGD-Net parallel-generation configurations and gaps <a id="section-005"></a>

| Configuration | Ownership | Existing foundation | Still to specify |
|---|---|---|---|
| Direct entity/query readout | E03/E05/E06/E08 | Node, property, and field-query heads | Joint consistency, batching semantics, actual parallelism |
| Multistep action chunks | E04, H04/H05 | Fixed-horizon action candidates | Coupled constraints, execution prefixes, feedback replanning |
| Continuous joint generation | E07, U06, D/U02 | Mathematical flow-matching instance | Noise/path, steps/solver, constraints, convergence, budget |
| Discrete masked generation | U06, E07/E08, B02/U02 | Generation extensions and sequence/relation interfaces | Corruption process, loss weighting, vocabulary, length/EOS, sampling/remasking |
| Local adaptive correction | F03/F04, H03/H07 | Candidate selection and local computation | Active sets, dependency propagation, cache invalidation, global coordination frequency |

The discrete path can start with a conditional masked-prediction baseline:

$$
\tilde y\sim q_\tau(\tilde y\mid y),\qquad
\mathcal L_{mask}=\mathbb E\!\left[\sum_{i\in\mathcal I(\tilde y)}w(\tau,i)
\bigl(-\log p_\theta(y_i\mid\tilde y,c,\tau)\bigr)\right].
$$

This is a training template to implement, not automatically a particular diffusion model's exact likelihood or bound. Define q, mask set I, time sampling, and weights. Reproducing a paper requires its complete process and validation. Continuous real-valued flow-matching ODEs cannot directly replace discrete token generation.

Inference configurations should register output_schema, length_policy, noise_or_mask_schedule, update_set, round_limit, stop_rule, rng_state, dependency_policy, precision, budget, and validation_profile. These are proposed software fields, not new hardware ABI.

## 6. Execution contracts for parallel rounds, memory, and local correction <a id="section-006"></a>

A candidate update round can be written as:

$$
Y^{k+1}=\operatorname{Update}_{\mathcal A_k}
\left(Y^k,F_\theta(Y^k,c,\tau_k;\mathcal M^{snapshot})\right).
$$

Each round reads frozen Yk and a memory snapshot and produces the next buffer: one synchronous-update configuration. Asynchronous/in-place updates need separate semantics; thread order must not accidentally define the model. Physical time t, generation time tau, correction round k, and version nu are distinct. Denoising must not automatically advance D04 physical state, and candidates must not enter actual-experience records.

Updating only difficult regions is a candidate optimization, not a guarantee that frozen positions preserve global consistency. Graph paths require affected-neighborhood expansion or periodic global checks; changing one position in dense attention can affect all positions. Bidirectional generation generally cannot reuse an autoregressive KV cache unchanged. Dependencies determine validity; approximate reuse incurs separately measured quality error.

By default, freeze model, graph schema, and memory versions within one generation. EV learning runs in the candidate domain and takes effect on subsequent requests or declared boundaries. Changing graph structure between rounds requires explicit state migration, revalidation, and registration as a different configuration. Candidates can be computed in parallel, while external actions still follow permitted prefixes, ordering, and feedback.

## 7. Efficiency and acceptance plan <a id="section-007"></a>

$$
t_{AR}\approx\sum_{i=1}^{T}t_{step,i},\qquad
t_{parallel}\approx\sum_{k=1}^{K}t_{round,k}+t_{schedule}+t_{check}.
$$

A full-output round does not cost the same as incremental decoding; T/K is not an automatic speedup. Compare completion time, first usable result, memory, energy, and batch throughput at matched quality, conditions, and hardware. Include retrieval, remasking, global coordination, failed recomputation, and distillation training costs.

| Experiment | Controls and key measurements |
|---|---|
| Memory correction and continual learning | State-only updates, fixed-capacity training, persistent growth; new/old tasks, delayed labels, long tail, drift |
| Consolidation and pruning | Original slow path, distilled fast path, hybrid; independent-data error, constraints, fallback rates, total cost |
| Parallel generation | Autoregression, direct chunks, global iterations, local iterations; matched-quality latency/memory curves |
| Caching and synchronization | Full recomputation reference; single-position changes, randomized execution order, version switching, missing observations |
| Closed-loop execution | Separate chunk prediction from execution; prefix validity, subsequent feedback, replanning |

These are unexecuted experiment plans. Chip/runtime requirements are composable batched computation, segmented reductions, active masks, double buffering, dependency/version management, and cost measurement. Small dynamic active sets may reduce matrix utilization; workload traces should justify fused paths rather than new model-named instructions for unified knowledge/action or diffusion intelligence.

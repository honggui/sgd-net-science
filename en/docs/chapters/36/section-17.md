# 15. Detailed Construction of Group A: Turning Observations into Interpretable Inputs

**On This Page**

- [A01 Contract Parsing](#section-001)
- [A02 Unit and Coordinate Conversion](#section-002)
- [A03 Entity Binding](#section-003)
- [A04 Time Scheduling](#section-004)
- [A05 Missingness and Quality Masks](#section-005)
- [A06 Domain and Condition Encoding Inputs](#section-006)
- [A07 Bounded Packing](#section-007)
- [A08 Provenance Binding](#section-008)

---

## A01 Contract Parsing <a href="#section-001" id="section-001"></a>

**Role and interface.** Before entering any encoder, raw record r is parsed into schema-bearing (o,c,M,metadata), distinguishing continuous quantities, categories, counts, and unknowns at the source. A reference definition is

$$
\operatorname{Parse}(r;\Sigma)=\begin{cases}(o,c,M,\mu),&r\models\Sigma,\\
\operatorname{Invalid}(reason),&\text{otherwise}.\end{cases}
$$

**Construction and use.** Σ defines field types, axes, required items, admissible units, and ranges. A software parser validates structure, and A02 then converts units. This unit has no learned parameters and shares one version between training and inference. Missing fields generate masks or rejection according to the contract; an MLP cannot be relied on to guess protocols, and unknown required types cannot default to zero.

## A02 Unit and Coordinate Conversion <a href="#section-002" id="section-002"></a>

**Role and mathematics.** Ensure inputs to the same model are comparable at a common scale. For scalars with affine unit conversions and rigid coordinates, use

$$
u_{can}=\alpha u_{raw}+\beta,\qquad x_{can}=Qx_{raw}+b,\quad Q^TQ=I.
$$

**Construction and use.** A unit library supplies scalar scales/offsets. Coordinates must associate a frame, transformation version, and measurement time; vectors are multiplied only by Q, while points also add b. Normalization statistics are fixed from training data, retaining inverse transforms for B08/E. Different physical quantities cannot share an arbitrary scale; nonrigid transformations, stress tensors, and manifold coordinates require their own rules. Conversion is deterministic preprocessing; losses do not train the meanings of units.

## A03 Entity Binding <a href="#section-003" id="section-003"></a>

**Role and mathematics.** Associate new observations with historical entity memory to prevent z from being mixed across objects. For example, an assignment allowing unmatched items is:

$$
\pi^*=\arg\min_{\pi\in\Pi}\sum_i c_{i,\pi(i)}+c_{miss}N_{miss},
\quad c_{ij}=\lambda_xd_x(i,j)+\lambda_fd_f(i,j).
$$

**Construction and use.** Geometric gating, appearance distances, and time differences form the cost. Matching outputs IDs, confidence, and birth/disappearance events for D04 to retrieve states. Learned associators train on labeled matching targets; deterministic matching needs no training. Hard assignments cannot be assumed differentiable. Errors near thresholds should retain candidate identities or create new states rather than continue old memory from low-confidence matches.

## A04 Time Scheduling <a href="#section-004" id="section-004"></a>

**Role and mathematics.** Convert asynchronous sensor events into one physical update for an entity:

$$
t=\operatorname{MapClock}(t_{src};clock,epoch),\qquad
\Delta t_i=t-t_i^{last},\quad age=t_{decision}-t_{sample}.
$$

**Construction and use.** The scheduler declares out-of-order windows, mapping errors, late-result associations, and maximum age. Each source event may trigger D04 only once. Inference refinement k does not change last; hypothetical rollouts use independent clocks. Mark unknown when reliable clock mapping is unavailable; data-row numbers must not replace real biological time. Training should simulate deployment-visible timing and missingness patterns.

## A05 Missingness and Quality Masks <a href="#section-005" id="section-005"></a>

**Role and mathematics.** Distinguish not measured from measured as zero, for example through a masked loss

$$
\mathcal L_{obs}=\frac{\sum_iM_iw_i\ell(\hat y_i,y_i)}{\max(1,\sum_iM_i)},
\quad M_i\in\{0,1\}.
$$

**Construction and use.** Output data validity, measurement noise, and reasons for B07 fusion, D05 correction, and G01 training. If w provides confidence weighting, its source is required; the model must not learn to assign zero weights to difficult samples. With all observations missing, the loss has no observational contribution; predictions may use history but must indicate insufficient evidence. Pass feature fill values together with masks; fillers are not supervision labels.

## A06 Domain and Condition Encoding Inputs <a href="#section-006" id="section-006"></a>

**Role and mathematics.** Convert device embodiment, species, environment, and task objectives into interpretable condition tuples:

$$
c=(c_{morph},c_{env},c_{task},c_{boundary}),\qquad \tilde c=\operatorname{EncodeSchema}(c).
$$

**Construction and use.** A06 outputs continuous attributes, categories, and presence masks; B06 handles the actual trainable embeddings. Identity keys support splitting/tracking and are not prediction features by default. New dexterous hands can be described by link lengths, joint limits, and similar properties, rather than relying only on training hand-type IDs. Out-of-domain categories require an unknown path; adding condition fields does not automatically yield cross-embodiment generalization.

## A07 Bounded Packing <a href="#section-007" id="section-007"></a>

**Role and mathematics.** Convert variable-length graphs into valid tensors within a batch. In packed representations, the node offset is

$$
o_b=\sum_{r<b}N_r,\quad i_{packed}=o_b+i,\qquad N_b\leq N_{max},\ E_b\leq E_{max}.
$$

**Construction and use.** Offset edge indices synchronously and retain sample_id; padded paths generate masks for nodes, edges, and readouts. Overflow may be rejected or explicitly partitioned, with cross-partition context/merging rules. This unit performs deterministic layout conversion. If truncation loses physical neighbors, the error contract must reflect it; valid shapes do not substitute for task completeness.

## A08 Provenance Binding <a href="#section-008" id="section-008"></a>

**Role and mathematics.** Represent data-to-prediction dependencies as a directed provenance graph:

$$
\mu_o=(id_o,version_o,parents_o,transform_o,split_o,source_o).
$$

**Construction and use.** Samples retain raw-object references, processing versions, real/simulated origins, and batches. G02 isolates data by provenance families; H08 aggregates evidence. Digests or hashes can verify object consistency but do not prove content is true. There are no trainable parameters. Regenerable data must retain seeds, generators, and versions rather than only a natural-language summary.


---

[← Previous](section-16.md) · [Contents](../../../SUMMARY.md) · [Next →](section-18.md)

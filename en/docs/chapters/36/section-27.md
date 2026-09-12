# 25. How to Construct and Execute the Complete Model

**On This Page**

- [25.1 From Task to Trainable Instance](#section-001)
- [25.2 Forward Procedure for One Real Event](#section-002)
- [25.3 Training and Gradient Boundaries](#section-003)
- [25.4 How Review, Retrospective Analysis, and Reflection Affect the Model](#section-004)
- [25.5 A Construction Example with Checkable Dimensions](#section-005)

---

## 25.1 From Task to Trainable Instance <a href="#section-001" id="section-001"></a>

For a task with entity graphs and real-time observations, first fix A contracts and G02 splits; choose B encoders producing d-dimensional scalars and optional vectors; construct C02/C03 edges and C05 geometry; stack L D blocks; select E task heads and G01 losses; register Group L applicability and H resource conditions. This forms a fixed-structure baseline.

Then enable F temporary refinement, IM implicit layers, U attention/memory, or EV online adaptation in individual experiments. A shared component library does not mean seven domains share one weight set. They may share only general operators or pretrained encoders, while domain-specific losses and output heads train separately.

A minimal instance must fully specify input axes and units, identity/clocks, graph relations and symmetry groups, d/s/n/q/L, layer sharing, activations, output distributions, losses, data splits, optimizers, stopping rules, available extensions, resource limits, and evidence versions. Abstract generality is not a substitute. Section 7 settings such as d=s=128 are debugging starting points, not optimal configurations.

## 25.2 Forward Procedure for One Real Event <a href="#section-002" id="section-002"></a>

```text
Input: event o, current active version ν, real memory at each layer, existing pre-observation prediction records
1. A01–A08 check identity/units/time/provenance and build inputs from visible information.
2. Associate event results with prior predictions and save EV01 innovations; mark unknown if no valid prediction exists.
3. B encodes and fuses by entity; C constructs graphs, geometry, and valid indices.
4. For each layer: D01 → D02/D03 → D04 (one real Δt only)
           → D05 (one valid innovation) → D06 → selected D07/D08.
5. E outputs next-time/task candidates and distributions; apply L02–L05 to candidates when needed.
6. F01/F02 form separate diagnostics; F03 chooses no change or temporary refinement within H03 budgets.
7. If refining: Fork the state; F05/F06 migrate computational representations; F07 executes explicit or IM computation;
           do not advance real D04 again; select candidates after L/H checks.
8. H04 checks the exact final candidate; H05/H06 commit reports or permitted actions according to use.
9. F08/H08 save prediction snapshots, versions, provenance, and costs; later outcomes support the next pairing cycle.
```

Group L's location in the connection diagram is simplified. Actual checks require their objects: state drift can be checked only after that state is predicted, and action feasibility only after the action is generated. L02 may generate V alongside E but cannot claim validation before the relevant candidate is defined. Static configurations skip temporal/observation innovations, while generation τ loops run in independent candidates.

## 25.3 Training and Gradient Boundaries <a href="#section-003" id="section-003"></a>

Explicit D/B/E paths backpropagate through the actual execution graph; time series use registered BPTT/truncation schemes. IM05 provides implicit gradients under applicable conditions or explicit unrolled alternatives. Teacher targets use stop-gradient. Hard graph search, tree splitting, version release, and device execution do not directly backpropagate. When learning soft edges or routing, specify differences between relaxations and final hardening.

Online local learning occurs in independent candidates. Affected parameters, states, optimizers, and executable plans form a consistent version. Do not save only new weights while retaining old-shape states. Window-level acceptance may occur automatically, without separate human approval for every update satisfying established conditions. Large-scale backbone reconstruction and cross-domain merging may occur in intergenerational stages.

## 25.4 How Review, Retrospective Analysis, and Reflection Affect the Model <a href="#section-004" id="section-004"></a>

G05/H08 reconstruct information available at the time; EV02 supplies testable causes; E01/H01 compare alternatives; F03/EV03 select revisions; EV05–EV07 test them on subsequent real tasks. If information alone is insufficient, update memory/acquire observations; grow only when capacity is actually insufficient, and change numerical paths when solving is ill-conditioned.

Counterfactual results retain SIMULATED labels and cannot directly serve as evidence of real interventions. Assess reflection capability through subsequent benefits, recurrence of similar errors, correct correction rates, and old-task degradation rather than explanation length or self-ratings. See [50 Reflection Document](../../50.md) for detailed event contracts.


## 25.5 A Construction Example with Checkable Dimensions <a href="#section-005" id="section-005"></a>

To illustrate concrete component tensors, take a single batch B=1, N=128, E≤2048, L=4, d=s=128, d_m=64, d_c=16, edge attributes d_e=8, geometric scalars d_I=1, n=3, q=4. The message MLP hidden width is 128 here. This independent example configuration neither overrides other Section 7 defaults nor represents an optimal choice.

| Path | Input and parameter shapes | Output | Construction responsibility |
|---|---|---|---|
| B03/B07 | Project each modality to 128, then fuse after gating | h:[128,128] | Preserve modality/entity binding |
| C05 | x:[128,3]，edge_index:[2,E] | Squared distances:[E,1] | Only distances enter messages in this example |
| D02 | concat:[E,281]，281→128→64 | m:[E,64] | 281=2×128+1+8+16 |
| D03 | segment mean over 128 nodes | m_bar:[128,64] | Empty neighborhoods produce zero |
| D04 input | concat:[128,208]，W_u:128×208 | u/z:[128,128] | 208=128+64+16 |
| D05 | 6 invariant observation innovations, W_r:128×6 | Correction:[128,128] | Gate is exactly zero with all observations missing |
| D06 | concat:[128,336]，336→128→128 | h_plus:[128,128] | 336=128+64+128+16 |
| D07 | φ_x:64→1, scalars multiply r:[E,3] | ξ_plus:[128,3] | Store ξ separately from measured x |
| E05 | Node-wise 128→1, then sum | Scalar energy and optional coordinate gradients | Output invariant energy and equivariant forces separately |

This example builds scalar memory from geometric invariants. If actual task observations contain orientations or force vectors, add compatible vector encoding and correction instead of directly inserting arbitrary six-dimensional vector data. If E05 recomputes energy after each D07, recompute the relevant geometric features and retain automatic-differentiation paths; the readout must not ignore the final layer's coordinate update.

Initialization may use conventional scale-preserving linear-layer initialization and zero biases. Register D04 initial decay timescales in training-sequence units. New adapters/experts may start with zero residual outputs, but all weights throughout the network should not be zeroed. Cold-start z=0; A03 decides reuse or reset at reidentification. All deployment normalization constants come from training data; online statistical changes form independent candidate versions.

First check forward computations and derivatives of this fixed-shape instance, then add one implicit refinement block or local expert. Do not simultaneously change encoders, layer counts, solver iterations, data, and budgets and attribute the benefits to one component.


---

[← Previous](section-26.md) · [Contents](../../../SUMMARY.md) · [Next →](section-28.md)

# 18. Detailed Construction of Group D: Trainable Kernels of Spatiotemporal Blocks

**On This Page**

- [D01 Type-Preserving Normalization/Projection](#section-001)
- [D02 Message Generation](#section-002)
- [D03 Message Reduction](#section-003)
- [D04 Entity-Temporal SSM](#section-004)
- [D05 Observation Innovation Correction](#section-005)
- [D06 Node Residual FFN](#section-006)
- [D07 EGNN Geometric Readout/Update](#section-007)
- [D08 Metric/Manifold Slot](#section-008)

---

## D01 Type-Preserving Normalization/Projection <a href="#section-001" id="section-001"></a>

**Role and mathematics.** Scalars may use LayerNorm; vectors use invariant-norm normalization:

$$
\tilde h=\operatorname{LN}(h),\quad
s_v=\sqrt{\epsilon+q^{-1}\sum_a\|v_a\|^2},\quad
\tilde v_a=\sum_bW_{ab}v_b/s_v.
$$

**Construction and use.** W mixes channel axes without arbitrarily mixing spatial axes; vector biases cannot be fixed directions. Only true invariants may use ordinary LN in scalar paths. Downstream tasks drive parameter training. Widening changes LN's statistical dimension, so copying old weights does not guarantee an unchanged function; revalidate after migration.

## D02 Message Generation <a href="#section-002" id="section-002"></a>

**Role and mathematics.** For an edge with relation type r:

$$
m_{ij}=\phi_{m,r}([h_i,h_j,I_{ij},a_{ij},c]),\qquad m_{ij}\in\mathbb R^{d_m}.
$$

**Construction and use.** The reference φ is Linear→SiLU→Linear with input width 2d+d_I+d_e+d_c. Relations may use a shared backbone plus type embeddings instead of copying a large network for every edge type. Geometric modes send only invariants into scalar MLPs; nongeometric graphs use attributes according to their schema. Local-message training cannot automatically overcome first-order graph expressiveness limits; select higher-order structures according to U05 task counterexamples.

## D03 Message Reduction <a href="#section-003" id="section-003"></a>

**Role and mathematics.** The default node aggregation is

$$
\bar m_i=\frac{\sum_{j\in\mathcal N(i)}M_{ij}m_{ij}}{\max(1,\sum_jM_{ij})},
$$

or a direct sum. Field-integral configurations instead use Σ_jw_jκ_θ(i,j)v_j, where w_j are quadrature weights.

**Construction and use.** Mean, sum, and quadrature weights have different meanings and must be registered in the model bundle rather than freely interchanged at deployment. Empty neighborhoods produce zero messages, and masks prevent padding leakage. Fixed sum/mean is not QK attention; attention requires explicit normalization scope and masks. Training gradients distribute through reductions; hardware implementations must test repeated indices and differences in floating-point order.

## D04 Entity-Temporal SSM <a href="#section-004" id="section-004"></a>

**Role and mathematics.** This edition's reference instance maintains per-entity, per-layer states through bounded decay:

$$
u_i=W_u[h_i,\bar m_i,c],\ a=-\operatorname{softplus}(\alpha),\quad
\delta_i=\Delta t_i\operatorname{softplus}(W_\delta u_i+b_\delta),\quad
\bar a_i=\exp(\delta_i\odot a),
$$
$$
z_i^-=\bar a_i\odot z_i^{old}+(1-\bar a_i)\odot\phi_b(u_i).
$$

**Construction and use.** u, δ, a, and z all have dimension s. Parameters are independent across layers and shared across events; new entities use fixed state initialization. Train on causal sequences; truncated BPTT is allowed if its truncation length is registered. This exponential-smoothing example constrains only homogeneous diagonal terms; it is neither complete Mamba nor implicit root finding. Δt=0 should not advance memory, and spatial refinement must not repeat physical decay.

## D05 Observation Innovation Correction <a href="#section-005" id="section-005"></a>

**Role and mathematics.** Using the prediction for the same target saved before observation, form

$$
r_t=M_t\odot(y_t-\hat y_t^-),\qquad
z_t=z_t^-+\gamma(u_t,M_t)\odot W_r\operatorname{Scale}(r_t).
$$

**Construction and use.** W_r:R^{d_y}→R^s, and γ is an s-dimensional gate multiplied by an indicator of a valid current innovation, making it exactly zero when all measurements are missing. Multiple observation heads combine separately by units and provenance. Scalar h/z paths receive only invariant innovations; vector innovations enter vector states through equivariant projection and cannot be mixed into invariant channels through arbitrary W_r. Each layer may consume the same real innovation once. Do not update the prediction using y and then call the difference a prior error. Without a matched prediction, use a separate encoding path or mark the innovation unknown. Task losses train K/gating; these do not automatically become optimal Kalman gains or Bayesian posteriors.

## D06 Node Residual FFN <a href="#section-006" id="section-006"></a>

**Role and mathematics.** Combine current representations, spatial messages, and memory:

$$
h_i^+=h_i+W_o\phi_h([h_i,\bar m_i,z_i,c]).
$$

**Construction and use.** Nontemporal configurations omit z and define corresponding weight shapes separately rather than fill arbitrary history. The FFN may be a two-layer MLP with output width d; reference dropout=0, with training/inference differences recorded if enabled. Residuals retain a path for old information but do not guarantee stability or permanent nondegradation. New residual experts may zero-initialize their output projections to reduce switching jumps, but whole-model validation remains necessary.

## D07 EGNN Geometric Readout/Update <a href="#section-007" id="section-007"></a>

**Role and mathematics.** Generate scalar coefficients from invariant messages to update computational coordinates ξ:

$$
\xi_i^+=\xi_i+\eta\sum_jM_{ij}\phi_x(m_{ij})(\xi_i-\xi_j).
$$

Vector outputs may similarly use Σ_jφ_v(m_ij)r_ij, with independent scalar coefficients for each vector channel.

**Construction and use.** n is the configured dimension, φ_x outputs scalars, and η is a declared scale. Measured x remains an original record, while ξ is an inference variable. This structure satisfies E(n) equivariance when the graph, messages, and group action are compatible; see the derivation in Section 26. Coordinate/vector targets require losses of corresponding types. Arbitrary coordinate updates are neither physical integrators nor guarantees of conservative forces. E(n) already has high-dimensional precedents. [EGNN](https://proceedings.mlr.press/v139/satorras21a.html)

## D08 Metric/Manifold Slot <a href="#section-008" id="section-008"></a>

**Role and mathematics.** On a defined Riemannian manifold (M,g), one constrained update is

$$
\xi^+=\operatorname{Retr}_\xi(-\eta\operatorname{grad}_g\mathcal J),\qquad
\operatorname{grad}_g\mathcal J=G(\xi)^{-1}\nabla_\xi\mathcal J
$$

(The latter expression is restricted to the corresponding local coordinates with positive-definite G.)

**Construction and use.** Coordinate domains, tangent spaces, metrics, retractions, singularities, and gradient interfaces are required. Substituting an arbitrary feature distance does not constitute a manifold algorithm. Lorentz-type indefinite metrics do not satisfy the above positive-definite gradient guarantees and require separate geometry and constraint definitions. Disabled by default; compare with Euclidean baselines only when task evidence supports the need.



---

[← Previous](section-19.md) · [Contents](../../../SUMMARY.md) · [Next →](section-21.md)

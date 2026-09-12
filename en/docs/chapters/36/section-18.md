# 16. Detailed Construction of Group B: Connecting Modality Encoding to Measurable States

**On This Page**

- [B01 Visual/Volumetric Encoder](#section-001)
- [B02 Sequence Encoder](#section-002)
- [B03 Atom/Entity Encoder](#section-003)
- [B04 Signal Encoder](#section-004)
- [B05 Omics Encoder](#section-005)
- [B06 Condition Adapter](#section-006)
- [B07 Masked Multimodal Fusion](#section-007)
- [B08 JSBO Structural Readout](#section-008)

---

## B01 Visual/Volumetric Encoder <a href="#section-001" id="section-001"></a>

**Role and mathematics.** Convert local pixels or voxels into features usable by nodes:

$$
p_k=\operatorname{Patch}(I)_k,\quad b_k=f_{vis}(p_{1:K})_k,\quad
h_i^{vis}=\sum_k\alpha_{ik}W_vb_k.
$$

**Construction and use.** f_vis may be a CNN, ViT, or volumetric network. α comes from geometric projection, segmentation, or learned matching, with entities aligned through A03. Freeze the encoder while training heads first, then fine-tune according to validation benefits. Ordinary visual encoding lacks arbitrary E(n) equivariance; an equivariant EGNN backend alone does not prove equivariance of the entire pixel-to-action system. Retain cropping, scaling, and camera calibration.

## B02 Sequence Encoder <a href="#section-002" id="section-002"></a>

**Role and mathematics.** Separate sequence positional encoding from physical time:

$$
b_{1:T}=f_{seq}(\operatorname{Emb}(s_{1:T})+p_{1:T};M_{seq}).
$$

**Construction and use.** Specify tokenizers, residue/base indices, masks, and truncation; retain mappings when aggregating tokens into graph nodes. Freeze or fine-tune domain-pretrained encoders; autoregressive tasks use causal masks, with separate configurations for bidirectional representations. Calling an external encoder reuses its interface and does not mean SGD has reproduced its full computation. Length extensions also require memory and cost tests.

## B03 Atom/Entity Encoder <a href="#section-003" id="section-003"></a>

**Role and mathematics.** Encode discrete types and canonical attributes into scalar channels:

$$
h_i^{ent}=\phi_e([\operatorname{Emb}(type_i),\tilde a_i,M_i,c_i])\in\mathbb R^d.
$$

**Construction and use.** Element, joint, and cell types use separate type tables; continuous mass/charge attributes retain unit provenance. φ_e may be a two-layer MLP; entities of the same type share parameters, while configuration determines cross-domain sharing. Geometric modes do not directly concatenate absolute coordinates into invariant scalar channels. Task losses train scalar encodings; explicit unknown or attribute-based paths handle unknown types.

## B04 Signal Encoder <a href="#section-004" id="section-004"></a>

**Role and mathematics.** Generate current-event features from a finite causal window:

$$
b_t=\phi_s\!\left(\sum_{r=0}^{K-1}W_r(M_{t-r}\odot y_{t-r}),M_{t-K+1:t},\Delta t\right).
$$

**Construction and use.** Convolution/filtering first processes electrophysiological, tactile, or spectral signals, then outputs d-dimensional entity-bound representations. Filters may be fixed or learned, but real-time modes must not read future windows; differing sampling rates require anti-aliasing/resampling conditions. Record window latency. Static spectrum inputs do not create physical-time memory from an FFT axis.

## B05 Omics Encoder <a href="#section-005" id="section-005"></a>

**Role and mathematics.** One expression-representation example is

$$
\tilde y_{ig}=\log\!\left(1+s_0\frac{y_{ig}}{\max(1,L_i)}\right),\quad L_i=\sum_g y_{ig},
\quad h_i^{omics}=\phi_o(\tilde y_i,c_i,M_i).
$$

**Construction and use.** This transform is only for input encoding. If E06 uses a count likelihood, supervision targets remain raw counts, modeled through exposure/library offsets. Batch conditions may reduce technical differences but should not unconditionally erase real biological differences. Fix gene order, sequencing platforms, and missingness scope; G02 manages donor/experiment splits. Cross-sectional cells cannot be fabricated into one-to-one paired trajectories.

## B06 Condition Adapter <a href="#section-006" id="section-006"></a>

**Role and mathematics.** Condition modulation may use FiLM:

$$
(\gamma(c),\beta(c))=\phi_c(c),\qquad \tilde h_i=(1+\gamma(c))\odot h_i+\beta(c).
$$

**Construction and use.** Initialize the γ/β output layers to zero so the initial scalar path approximates the identity, then train on target-domain data. Invariant γ may scale vector channels, but arbitrary fixed spatial directions β cannot be added. Adapter parameters may be independently frozen, transferred, and updated online. New input conditions do not automatically infer unobserved mechanical or biological mechanisms.

## B07 Masked Multimodal Fusion <a href="#section-007" id="section-007"></a>

**Role and mathematics.** For modality m of the same entity, use normalized gating over valid modalities:

$$
\alpha_{im}=\frac{M_{im}\exp q_m(b_{im},c)}{\sum_rM_{ir}\exp q_r(b_{ir},c)},\qquad
h_i^0=\sum_m\alpha_{im}W_mb_{im}.
$$

**Construction and use.** When the denominator is zero, use an explicit missing/history path instead of softmax(-∞,…,-∞). First map every modality into a common d-dimensional space. Simulate missing modalities appropriately during training, but evaluate on real validation data. Cross-modal attention may replace gating while preserving entity correspondence, temporal visibility, and type rules; gating weights are not directly interpretable as causal contributions.

## B08 JSBO Structural Readout <a href="#section-008" id="section-008"></a>

**Role and mathematics.** Connect latent representations to measurable physical quantities:

$$
\hat u_i=T_{unit}^{-1}g_\theta(h_i,v_i,c),\qquad
\mathcal L_{bridge}=\sum_iM_i\|W_{scale}(\hat u_i-u_i^{obs})\|^2.
$$

**Construction and use.** Define separate heads, inverse unit transforms, and observation operators for position, velocity, concentration, and other quantities. Map states not directly observed to actual sensor quantities before computing losses. g's output type must be compatible with target symmetry. JSBO here is a project-specific structural bridge requiring data calibration; unsupervised latent vectors cannot automatically reveal real causal variables or laws.


---

[← Previous](section-17.md) · [Contents](../../../SUMMARY.md) · [Next →](section-19.md)

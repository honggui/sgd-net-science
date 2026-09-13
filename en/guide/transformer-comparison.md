# SGD-Net and Transformer: Operator Coverage, Architectural Differences, and Proposed Additions

> 2026-09-13. Review of the current public research architecture. “Defined” means a mathematical or interface definition in the documents, not verified code, trained weights, chip implementation, or performance. This document adds comparisons and candidate contracts; it does not change the 72 top-level components or make optional U extensions mandatory defaults.

## 1. Conclusion and evidence scope <a id="section-001"></a>

SGD-Net needs many basic operators used by Transformers: linear projections, nonlinearities, residuals, normalization, masks, reductions, and tensor layout operations. These also underpin graph messages, state gating, and domain readouts. Global or cross-modal attention and addressable history should be enabled according to task needs.

The most accurate assessment is: **basic computations have relatively concrete definitions; full attention and caching have assigned extension slots; execution contracts for modern Transformer variants remain incomplete; end-to-end compatibility has not been validated.** An interface to a pretrained encoder does not establish an internal Transformer implementation.

Project evidence includes the [D01–D08 kernels](../docs/chapters/36/section-20.md), [B encoding and fusion group](../docs/chapters/36/section-18.md), [U extensions](../docs/chapters/36/section-26.md), [47 generality boundaries](../docs/47.md), and [component registry](../architecture/README.md). The table uses three design statuses: A means a concrete mathematical definition exists, B means an optional capability is listed, and C means a specific contract still needs definition. None means implemented.

The comparison baseline is a standard dense Transformer, with variants such as RMSNorm, SwiGLU, RoPE, and GQA identified separately. These variants, MoE, and inference caches are not mandatory components of every Transformer.[Original Transformer paper](https://arxiv.org/abs/1706.03762)

## 2. Operator-by-operator comparison <a id="section-002"></a>

| Operator or function | SGD-Net use and ownership | Current design status and gap |
|---|---|---|
| Embedding, lookup, and input projection | B02/B03: text, residues, elements, and entity types | A; vocabulary, tokenization, unknown types, and input mapping must still be fixed per task |
| Linear / GEMM and bias | B group, D02 message MLP, D04 gate projections, D06 FFN, E output heads | A; geometric vectors require compatible channel mixing and bias rules |
| ReLU / GELU / SiLU / gated multiplication | D02 references Linear–SiLU–Linear; D04/D05 use gates | A for existing SiLU/gates; B for U04; C for a specified GELU approximation and SwiGLU composition |
| FFN and residual addition | D06 node residual FFN | A; its inputs include messages/states, so it is not a same-weight substitute for the original per-token Transformer FFN |
| LayerNorm / RMSNorm | D01 scalar LayerNorm; invariant norms for vectors | A for LayerNorm; C for RMSNorm, epsilon, statistical axes, and Pre/Post-Norm composition under U04/D01 |
| Q/K/V projection, head split and merge | U02, associated with B02/B07 and D02/D03 | B; specify head counts, dimensions, layouts, output projection, and weight mapping |
| QK dot product, scaling, row Softmax, AV | U02 relation computation for global entity interaction or cross-modal queries | B; default D03 sum/mean is not equivalent; Softmax in B07 modality gating is not full attention either |
| Causal, padding, and cross-modal masks | A-group validity, B02/B07, U02 visibility | A/B; broadcasting axes, Boolean meaning, fully masked rows, and future-information isolation need a unified C contract |
| Sequence position encoding / RoPE | B02 separates position from physical time; U02 provides ordering capability | A/B for general positions; C for RoPE rotation dimensions, frequencies, position offsets, and scaling |
| Cross-attention | Optional B07 fusion with U02 relation operations | B; Q comes from queries and K/V from the conditioning source; lengths need not match |
| KV cache, incremental append, and history reads | U03, associated with D04/F08/H02/H07 | B; C for valid lengths, batch reordering, version isolation, and invalidation; not the D04 recurrent state |
| GQA / MQA | U02 head mapping and U03 cache layout | C; multiple query heads sharing KV heads require an explicit mapping, not just narrower tensors |
| Token logits, log-softmax, cross-entropy, sampling | E08/U06/G01/G03 language or discrete-sequence task configurations | B; general E08 readout is not a complete LM head; specify vocabulary projection, shifted labels, ignored masks, and sampling |
| Dropout, backpropagation, precision, random state | D06 defaults to dropout=0; G training, U07/U08 conformance | Training designs exist; C for exact random paths, gradients, and inference disabling rules |
| reshape, transpose, gather, scatter, segmented reductions | U01/U04/H07; graph indices, heads, token/entity mapping | Uses and extension ownership exist; C for strides, duplicate indices, segment boundaries, and layout-conversion costs |
| FlashAttention-style tiled execution | Candidate U02 backend and H07 scheduling | C; an IO optimization for exact attention, not a new learning mechanism; no project kernel or speed result exists |

LayerNorm and RMSNorm differ: RMSNorm does not center by the mean and cannot unconditionally replace the normalization used by existing weights.[RMSNorm](https://arxiv.org/abs/1910.07467) SiLU alone is not SwiGLU; one bias-free SwiGLU FFN is:

$$
\operatorname{FFN}_{\mathrm{SwiGLU}}(H)=\bigl(\operatorname{SiLU}(HW_g)\odot HW_u\bigr)W_d.
$$

Both input projections, elementwise multiplication, and output projection must be defined together; hidden width must also enter the budget.[GLU Variants](https://arxiv.org/abs/2002.05202)

## 3. Why graph aggregation is not automatically attention <a id="section-003"></a>

For one head, let Q have length Tq and K/V length Tk:

$$
Q=H_qW_Q,\quad K=H_kW_K,\quad V=H_kW_V,
$$
$$
S_{ij}=q_i^Tk_j/\sqrt{d_k}+b_{ij}+M_{ij},\quad
A_{ij}=\frac{\exp(S_{ij}-m_i)}{\sum_{r\in\mathcal V_i}\exp(S_{ir}-m_i)},\quad
Y_i=\sum_{j\in\mathcal V_i}A_{ij}v_j.
$$

The valid-key set is V_i; m_i is its maximum row score. M is an additive mask with zero for allowed positions and negative infinity for prohibited ones; b is an optional relation bias. The formula requires a nonempty valid set. Fully masked rows must explicitly raise an error or follow a declared zero-output path, rather than evaluate undefined normalization. Head outputs are combined and projected.[PyTorch multihead attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html)

The project mapping is: D02/U02 produces relation scores and values; D03/U02 performs segmented maxima, exponential sums, and weighted reductions per query node and head. If the neighborhood excludes otherwise allowed keys, this is the selected sparse attention, not an equivalent implementation of dense attention.

A counterexample makes the distinction concrete: two scalar values, 0 and 2, yield a fixed mean of 1. Attention weights of 0.9 and 0.1 yield 0.2, and another query can produce different weights. Without query-dependent scores and row normalization, stacking ordinary sum/mean layers does not automatically turn that operator into attention.

Unify masks in the adapter: Boolean True in PyTorch SDPA permits participation, whereas Boolean masking True in MultiheadAttention prohibits it. Conversion is required. SDPA dropout probability must also explicitly be zero during inference.[SDPA interface](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

## 4. Architectural capabilities and costs <a id="section-004"></a>

| Dimension | Standard dense Transformer | Current SGD-Net research design |
|---|---|---|
| Information interaction | Query-dependent attention reads permitted tokens | Default graph messages; U02 can add global/sparse attention |
| Temporal memory | Autoregressive inference commonly retains historical K/V; history can also be recomputed | D04 is per-entity recurrent state; U03 optionally retains addressable history |
| Geometric priors | Ordinary token attention has no automatic physical rotation/translation equivariance guarantee | D07 has E(n)-equivariant structure when types, graph, and group action are compatible; encoders/readouts must also be compatible |
| Physical constraints | Basic blocks do not guarantee conservation or stability; these can be added | L/IM/domain heads provide conditional stability, implicit solving, and constraint interfaces, still requiring validation |
| Structural adaptation | Basic blocks are usually fixed; other research can add routing, pruning, and memory | F/EV designs local refinement, persistent growth, and migration; benefits remain unverified |
| Function and task preferences | Flexible content addressing supports cross-position associations; results depend on training and capacity | Explicit relations, state, and symmetry can provide task priors; incorrect priors can also restrict expression |
| Empirical status | The original paper and later work provide evidence for their respective tasks | Current publication comprises designs and validation plans; component counts do not establish superiority over Transformers |

The following is project cost analysis, not a performance measurement. For T tokens and total head width d, a standard dense attention layer has approximately O(T²d) interaction work, O(Td²) projection work, and O(Tdd_ff) FFN work. FlashAttention uses tiling to reduce intermediate attention-matrix storage and memory traffic; dense interaction arithmetic remains quadratic.[FlashAttention](https://arxiv.org/abs/2205.14135)

For N SGD nodes and E edges, a dense two-layer width-d MLP on every edge can cost O(Ed²); projected sparse dot-product attention can have O(Ed) relation work. These are different operations. D04 state updates cost O(Ns), but projections, graph construction, geometry, solver iterations, candidate trajectories, and training states must be counted separately.[Existing cost boundaries](../docs/chapters/36/section-29.md)

For conventional decoding caches with equal K/V widths, raw cache bytes are:

$$
B_{KV}=2LBT H_{kv}d_h b,\qquad B_z=LB N s b.
$$

L is layer count, B batch size, b bytes per element, Hkv KV head count, and dh head width. The second expression covers only this project's simplified z state. With fixed entity count, z need not grow with history length, but it is compressed memory and does not guarantee exact access to arbitrary history. GQA reduces KV head count while retaining attention computation for query heads.[GQA](https://arxiv.org/abs/2305.13245)

For L=24, B=1, T=4096, Hkv=8, dh=64, and b=2, raw KV is 192 MiB. With N=128 and s=128, the corresponding z is 0.75 MiB. This arithmetic example does not match task capability, other states, or quality; it is not evidence of a 256-fold memory advantage for equivalent models.

## 5. How to combine the architectures <a id="section-005"></a>

Continue the capability-first approach: retain local D02/D03 structural computation and D04 temporal state; add U02 when evidence identifies long-range entity relations or cross-modal query bottlenecks. Add U03 for exact historical queries or incremental autoregression. Text and protein sequences can reuse B02 encoders, but external-model contributions must be reported separately.

For physical AI, an equivariant attention branch can be constructed for subsequent validation:

$$
\alpha_{ij}=\operatorname{softmax}_{j\in\mathcal N(i)}
f_\theta(h_i,h_j,\|x_i-x_j\|^2,a_{ij}),\qquad
\Delta x_i=\sum_j\alpha_{ij}c_{ij}(x_i-x_j).
$$

Only when h, a, c, and attention scores are invariant under the selected group action, and neighborhood construction is compatible, does a scalar-weighted relative-vector update preserve the corresponding E(n) equivariance. This follows established equivariant-message construction principles; it is neither claimed as a project invention nor as a trained result.[EGNN](https://proceedings.mlr.press/v139/satorras21a.html)

Sequence RoPE acts on Q/K feature pairs to encode position relationships. It is not a physical-space E(n) equivariance constraint and cannot replace D07. Sequence position, physical time, and spatial coordinates should remain distinct types.[RoFormer](https://arxiv.org/abs/2104.09864)

Prioritize three configurations rather than adding more top-level IDs:

1. **Basic operators**: D01/D02/D06/U04 specify activation, normalization, bias, residual order, and tensor axes; distinguish Pre-Norm from Post-Norm when matching a target model.[EncoderLayer configuration](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoderLayer.html)
2. **Relation computation**: U02 specifies QKV, head layout, Softmax axis, masks, positions, empty-row behavior, and local/global edge sets.
3. **Incremental memory**: U03 specifies model/layer/session versions, position offsets, valid lengths, append, reorder, and invalidation. Old caches cannot be assumed valid after weights or position rules change.

A hybrid can fuse local messages, state, and global relations in its invariant scalar stream, using type-compatible readouts for vectors. Retrain and validate the hybrid; do not assume it preserves outputs from original Transformer weights. Exact Transformer migration, when required, uses a separate U08 compatibility configuration that bypasses additional geometry, state, or projection operations that would change outputs.

## 6. Validation and hardware implications <a id="section-006"></a>

This documentation update does not implement or run the following model experiments. They are proposed acceptance checks before claiming support:

| Acceptance check | Error to exclude |
|---|---|
| Same-parameter single/multihead forward and gradient comparison | Incorrect scaling, head split, transpose, normalization axis, or output projection |
| Modify future tokens, add padding, fully mask a row | Future leakage, padding contamination, NaNs, or undeclared empty-row behavior |
| Compare full-sequence computation with token-by-token cached decoding | Position offsets, duplicate append, session mixing, or missing cache invalidation |
| GQA head mapping and unequal-length cross-attention inputs | Incorrect broadcasting or applying self-attention layouts to cross-modal inputs |
| Rotation/translation/relabeling differences | Attention inputs, neighborhoods, or vector outputs violating type/equivariance conditions |
| Ablate fixed graph+SSM, then U02, then U03 | Inability to attribute gains to attention, memory, or additional budget |

Register tolerances by dtype, sequence length, and backend; compare outputs, gradients, cache states, and peak memory. Quality tests should share data splits, observable information, and explicit training/inference budgets. Report failure rates and end-to-end latency, not only GEMM throughput.

The Aegis-X recommendations are workload requirements, not a completed chip Spec audit. Under the [existing mapping](../docs/chapters/36/section-09.md), matrix units continue serving projections/FFN/QK/AV; vector/reduction operations cover exp, max/sum, rsqrt, activations, and masks; sparse/index operations and DMA handle graph relations, layouts, and state movement. Actual workload traces should determine whether tiled attention and paged KV deserve specialized optimization. Transformer popularity does not require every SGD configuration to carry a KV cache; retaining an SSM does not remove necessary matrix and reduction capabilities.

**Research priority: establish reliable shared operators and the U02 relation configuration, then enable U03 according to task evidence; evaluate the hybrid by quality–cost and constraint consistency.** This continues the [49 capability-first roadmap](../docs/49.md), without making complete Transformer simulation the project's overall goal again.

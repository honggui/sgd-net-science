# 11. Layer 6: Posterior Error Estimation Layer

**On This Page**

- [11.1 Function](#section-001)
- [11.2 General Error Form](#section-002)
- [11.3 Task Error](#section-003)
- [11.4 Topology Error](#section-004)
- [11.5 State-Drift Error](#section-005)
- [11.6 Stability Error](#section-006)
- [11.7 Confidence Error](#section-007)
- [11.8 Trigger Policy](#section-008)
- [11.8.1 Adaptive Budget Controller (Runtime Budget Controller)](#section-009)
- [11.8.2 QuantizationAwareStateCache and StaticMaxGraphMask](#section-010)
- [11.9 Implementation Interface](#section-011)

---

## 11.1 Function <a href="#section-001" id="section-001"></a>

The posterior error estimation layer determines whether SGD-Net should dynamically change its structure.

It addresses the question:

> Which nodes, edges, subgraphs, paths, or local regions currently have excessive error and require refinement, isolation, or pruning?

## 11.2 General Error Form <a href="#section-002" id="section-002"></a>

$$
\eta_i=\lambda_1\eta_i^{task}+\lambda_2\eta_i^{topo}+\lambda_3\eta_i^{state}+\lambda_4\eta_i^{stability}+\lambda_5\eta_i^{confidence}
$$

## 11.3 Task Error <a href="#section-003" id="section-003"></a>

For supervised tasks:

$$
\eta_i^{task}=\|\hat{y}_i-y_i\|_2
$$

For PDEs:

$$
\eta_i^{task}=\|\mathcal{R}(u_i)\|
$$

Here $$\mathcal{R}$$ is the PDE residual operator.

## 11.4 Topology Error <a href="#section-004" id="section-004"></a>

If model information propagation crosses an inadmissible boundary:

$$
\eta_i^{topo}=\sum_j \mathbb{I}[A_{ij}=0]\cdot s_{ij}
$$

Here $$s_{ij}$$ is the strength of the implicit association the model attempts to establish.

## 11.5 State-Drift Error <a href="#section-005" id="section-005"></a>

$$
\eta_i^{state}=\|h_{i,t}-h_{i,t-1}\|_2
$$

Alternatively, use KL divergence:

$$
\eta_i^{state}=D_{KL}(P_t(y_i)\parallel P_{t-1}(y_i))
$$

## 11.6 Stability Error <a href="#section-006" id="section-006"></a>

For example, a spectral radius exceeding its bound:

$$
\eta^{stability}=\max(0,\rho(W)-\rho_{max})
$$

## 11.7 Confidence Error <a href="#section-007" id="section-007"></a>

In biological structure prediction:

$$
\eta_i^{confidence}=1-\frac{pLDDT_i}{100}
$$

Or, for high-PAE regions:

$$
\eta_{ij}^{PAE}=\frac{PAE_{ij}}{PAE_{max}}
$$

## 11.8 Trigger Policy <a href="#section-008" id="section-008"></a>

```text
if conservation_check_invalid_or_failed:
    reject_candidate_or_use_declared_fallback
elif observation_missing_or_noise_dominated:
    request_observation_or_keep_unknown
elif persistent_residual_high and capacity_or_resolution_cause_supported:
    propose_isolated_refinement_with_migration
elif residual_low and merge_consistency_valid and benefit_expected:
    propose_isolated_merge
else:
    keep
# All candidates are automatically accepted between windows after benefit, regression, numerical, resource, and version validation.
```

A residual threshold is only a condition for generating candidates. pLDDT and PAE are model confidence/error prediction metrics, not independent real observations; their reduction must not be directly treated as a growth benefit or improvement relative to ground truth.

## 11.8.1 Adaptive Budget Controller (Runtime Budget Controller) <a href="#section-009" id="section-009"></a>

AQAD's dynamic-step-size idea can be generalized into an SGD-Net runtime budget controller. Its input is multichannel uncertainty rather than logits entropy alone:

$$
U_t = w_H \tilde{H}_t + w_R\|r_t\| + w_C(1-c_t) + w_S s_t
$$

Here $$\tilde{H}_t$$ is threshold-truncated entropy of the path/output distribution, $$r_t$$ is the Harness residual, $$c_t$$ is source or model confidence, and $$s_t$$ is state drift, quantization residual, or memory pressure.

At runtime, $$U_t$$ can be mapped to:

| Output budget | Controlled object |
|---|---|
| `refinement_depth` | Maximum dynamic-tree split/grow depth. |
| `rollout_steps` | Number of short-term SSM/world-model rollout steps. |
| `candidate_branch_count` | Number of local-expert, draft-tree, or experimental-hypothesis branches. |
| `verification_strength` | Rechecking strength of Harness, PPU, or explicit solvers. |
| `state_precision` | Quantization bit width and fallback precision for SSMState, GraphState, JEPALatent, or KV cache. |
| `fallback_threshold` | Threshold for returning high-risk samples to a conservative path. |

This controller schedules only performance, budget, and safety policies; entropy or residual thresholds cannot be presented as correctness proofs. Correctness still comes from Harness, StabilityProjector, Rollback, and fallback.

## 11.8.2 QuantizationAwareStateCache and StaticMaxGraphMask <a href="#section-010" id="section-010"></a>

Long-horizon agents, Retrospection replay, and dynamic-graph batches generate substantial state caches. The following can be introduced:

```text
QuantizationAwareStateCache {
  state_kind: SSMState | GraphState | JEPALatent | RetrospectionTrace | KVCache
  bit_width
  scale_granularity
  residual_statistics
  quality_guard
  fallback_precision
}
```

At the same time, use an AQAD-style static-maximum-shape strategy for dynamic-graph batches:

```text
physical: N_max / E_max / tree_depth_max
logical:  N_t / E_t / tree_depth_t
mask:     mask invalid nodes, edges, and tree branches
release:  release unused slots at iteration boundaries
```

Record `wasted_compute_ratio`, `early_release_slots`, `masked_branch_count`, and quality-degradation metrics; low-bit state caching must not be assumed lossless.

## 11.9 Implementation Interface <a href="#section-011" id="section-011"></a>

```text
PosteriorErrorEstimator.estimate(prediction_record, matched_observation=None, constraints=None) -> ErrorReport
```


---

[← Previous](section-12.md) · [Contents](../../../SUMMARY.md) · [Next →](section-14.md)

# 26. Properties That Can Be Established, Conditions, and Counterexamples

**On This Page**

- [26.1 EGNN Module Equivariance in Arbitrary n-Dimensional Euclidean Space](#section-001)
- [26.2 Distinguishing SSM Decay from Whole-System Stability](#section-002)
- [26.3 A Short Derivation of the Implicit Residual Bound](#section-003)
- [26.4 P/Q, Errors, and Adaptation Do Not Automatically Guarantee Convergence](#section-004)
- [26.5 Practical Boundaries of Generality and Learning Capability](#section-005)

---

## 26.1 EGNN Module Equivariance in Arbitrary n-Dimensional Euclidean Space <a href="#section-001" id="section-001"></a>

Let Q∈O(n), b∈R^n, and ξ'_i=Qξ_i+b. If graph support is determined by compatible invariants and h/a/c are invariant, then

$$
r'_{ij}=Qr_{ij},\quad \|r'_{ij}\|^2=\|r_{ij}\|^2,\quad m'_{ij}=m_{ij}.
$$

Substitute into D07:

$$
(\xi'_i)^+=Q\xi_i+b+\eta\sum_j\phi_x(m_{ij})Qr_{ij}
=Q\xi_i^++b.
$$

This module form therefore satisfies E(n) equivariance, with the same construction applicable for n≥3. The derivation does not prove that it still holds after adding arbitrary encoders, chirality terms, normalization, nonequivariant routing, hard tie-breaking, or numerical solving. Every path in the complete composition requires compatible group actions. Increasing n does not automatically satisfy complex physical constraints, and E(n) does not cover arbitrary manifold transformations.

## 26.2 Distinguishing SSM Decay from Whole-System Stability <a href="#section-002" id="section-002"></a>

With fixed input driving and identical gates for two D04 trajectories, their difference satisfies δz⁺=diag(a_bar)δz. Thus, if max|a_bar|≤q<1, then `||δz⁺||≤q||δz||`. But if u, gates, messages, or innovations depend on z, the full Jacobian includes those derivatives; diag(a_bar) alone is insufficient. At Δt=0, the decay factor is 1, so there is no strict contraction.

Only if `sup ||∂F/∂z||≤q<1` can be proved for the complete F on an invariant domain can the corresponding contraction be discussed. Sampled Jacobian estimates during learning are not global upper bounds. Genuinely unstable physical dynamics need not be made contractive; select stability objectives according to L01.

## 26.3 A Short Derivation of the Implicit Residual Bound <a href="#section-003" id="section-003"></a>

If f is a q<1 contraction and ζ* a fixed point in the same domain, then

$$
\|\zeta-\zeta^*\|\leq\|\zeta-f(\zeta)\|+\|f(\zeta)-f(\zeta^*)\|
\leq\|R(\zeta)\|+q\|\zeta-\zeta^*\|.
$$

Rearranging gives the IM04 bound. When q is close to 1, tiny residuals may still correspond to substantial errors. The bound is unavailable without contraction or other appropriate stability conditions. Multiple roots additionally require root-selection semantics.

## 26.4 P/Q, Errors, and Adaptation Do Not Automatically Guarantee Convergence <a href="#section-004" id="section-004"></a>

Under F06 conditions, QP=I preserves coarse representations; detail in the I−PQ component is still lost, so bidirectional losslessness does not follow. P/Q operator norms in the task norm affect disturbance amplification; graph switching also requires L06 long-term conditions. Adding experts or representation nodes merely enlarges the candidate representation family. Training may fail or even reduce generalization due to insufficient data.

FN total error may distinguish observation/discretization, representation, learning, and numerical terms in the same output norm; refinement addresses only some terms. The aliasing example where 0 and sin(2πmx) have identical samples on a finite grid shows that no model size can distinguish them from those samples alone. Adding observations, introducing justified function-class priors, and increasing network capacity are different measures.

## 26.5 Practical Boundaries of Generality and Learning Capability <a href="#section-005" id="section-005"></a>

Finite-precision B-bit memory cannot losslessly store all binary histories of length T>B, a state-counting limitation. U03 can expand addressable memory but still requires resources. First-order neighborhood aggregation without identity/position/higher-order features cannot distinguish a six-cycle from two triangles with identical initial values; adding layers of the same type does not automatically fix this. Counterexamples guide task-based capability selection, without requiring SGD to fully simulate every existing model.

These are conditional derivations and counterexamples for specific configurations, not a unified theorem for the entire 72-unit architecture. The core hypothesis still awaiting validation is that error-driven representation/computation adjustments yield benefits under specified tasks and budgets.


---

[← Previous](section-27.md) · [Contents](../../../SUMMARY.md) · [Next →](section-29.md)

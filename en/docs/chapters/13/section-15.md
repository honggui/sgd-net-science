# 13. Layer 8: Stability Projection Layer

**On This Page**

- [13.1 Function](#section-001)
- [13.2 Spectral-Radius Projection](#section-002)
- [13.3 Singular-Value Clipping](#section-003)
- [13.4 Norm Clipping](#section-004)
- [13.5 Lyapunov Energy-Descent Constraint](#section-005)
- [13.6 Implementation Interface](#section-006)

---

## 13.1 Function <a href="#section-001" id="section-001"></a>

The stability projection layer enforces specified numerical constraints and checks declared stability conditions; it does not itself prove nondivergence of the entire dynamic graph.

It addresses the question:

> Can new nodes, updated weights, or state-transition matrices break stability? If so, how can they be projected back into a safe set?

## 13.2 Spectral-Radius Projection <a href="#section-002" id="section-002"></a>

For a square state-transition matrix $$A$$ with $$0<\rho_{max}<1$$:

$$
A' = \frac{A}{\max(1,\rho(A)/\rho_{max})}
$$

This expression controls only the spectral radius of one matrix. It is not equivalent to Euclidean-norm contraction and does not cover arbitrary input-dependent switching. A common contraction condition requires checking a bound on the complete update map in the same norm.

## 13.3 Singular-Value Clipping <a href="#section-003" id="section-003"></a>

$$
W=U\Sigma V^T
$$

$$
\Sigma'=\operatorname{clip}(\Sigma,0,\sigma_{max})
$$

$$
W'=U\Sigma'V^T
$$

## 13.4 Norm Clipping <a href="#section-004" id="section-004"></a>

For node features:

$$
x_i'=x_i\cdot\min\left(1,\frac{c}{\|x_i\|_2}\right)
$$

## 13.5 Lyapunov Energy-Descent Constraint <a href="#section-005" id="section-005"></a>

Define a candidate Lyapunov function (not the actual physical energy of an arbitrary task):

$$
V_t=\frac{1}{2}h_t^TPh_t+\Pi(X_t,\mathcal{G}_t)
$$

If:

$$
V_{t+1}>V_t+\delta
$$

then:

- reduce the update step size;
- roll back topology refinement;
- prune anomalous branches;
- reproject the weights.

Here one must state the positive definiteness of P, a lower bound for the additional term, the reference equilibrium/trajectory, and the input-work or disturbance-supply term. Positive definiteness does not imply descent; repeatedly allowing increases by a fixed δ does not guarantee stability. Across topologies, a common V or bounded switching conditions are required; vector channels also require checking the group invariance of V. Rollback on failure applies only to candidate computational states and cannot undo real actions. See document 44 for details.

## 13.6 Implementation Interface <a href="#section-006" id="section-006"></a>

```text
StabilityProjector.project(state, params=None) -> GraphState
```


---

[← Previous](section-14.md) · [Contents](../../../SUMMARY.md) · [Next →](section-16.md)

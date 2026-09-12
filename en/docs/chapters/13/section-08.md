# 8. Layer 3: SSM State Evolution Layer

**On This Page**

- [8.1 Function](#section-001)
- [8.2 Basic State-Space Operator](#section-002)
- [8.3 Selective SSM Operator](#section-003)
- [8.4 Graph-Conditioned SSM](#section-004)
- [8.5 Stable Discretization](#section-005)
- [8.6 Input and Output Shapes](#section-006)
- [8.7 Implementation Interface](#section-007)
- [8.8 Residual-Gated SSM Enhancement](#section-008)

---

## 8.1 Function <a href="#section-001" id="section-001"></a>

The SSM layer performs long-range explicit recurrence for each node or the graph-wide state. A hidden state is not an implicit solve; the latter is provided separately by IM01–IM06.

It addresses the question:

> How is the current node state affected by historical states, time series, reasoning chains, or diffusion steps?

## 8.2 Basic State-Space Operator <a href="#section-002" id="section-002"></a>

Continuous form:

$$
\dot{h}(t)=Ah(t)+Bx(t), \qquad y(t)=Ch(t)+Dx(t)
$$

Discrete form:

$$
h_t=\bar{A}h_{t-1}+\bar{B}x_t
$$

$$
y_t=Ch_t+Dx_t
$$

## 8.3 Selective SSM Operator <a href="#section-003" id="section-003"></a>

In Mamba/Selective SSMs, state transitions may depend on the input:

$$
h_t=\bar{A}(x_t)h_{t-1}+\bar{B}(x_t)x_t
$$

$$
y_t=C(x_t)h_t+D(x_t)x_t
$$

In SGD-Net, each graph node can be viewed as a local state stream:

$$
h_{i,t}=\bar{A}_i(x_{i,t})h_{i,t-1}+\bar{B}_i(x_{i,t})x_{i,t}
$$

## 8.4 Graph-Conditioned SSM <a href="#section-004" id="section-004"></a>

SGD-Net can go further by conditioning state transitions on graph context:

$$
\tilde{x}_{i,t}=\operatorname{Agg}_{j\in\mathcal{N}(i)}(x_{j,t},e_{ij})
$$

$$
h_{i,t}=\bar{A}(x_{i,t},\tilde{x}_{i,t})h_{i,t-1}+\bar{B}(x_{i,t})x_{i,t}
$$

This makes the SSM a topology-modulated state-evolution operator, beyond a sequence operator alone.

## 8.5 Stable Discretization <a href="#section-005" id="section-005"></a>

A bilinear transform can be used:

$$
\bar{A}=\left(I-\frac{\Delta t}{2}A\right)^{-1}\left(I+\frac{\Delta t}{2}A\right)
$$

If $$\operatorname{Re}(\lambda(A))\le 0$$, this helps keep $$\rho(\bar{A})\le 1$$.

## 8.6 Input and Output Shapes <a href="#section-006" id="section-006"></a>

```text
Input:
  x:     [N, F]
  h:     [N, D_h]
Output:
  y:     [N, D_s]
  h_new: [N, D_h]
```

## 8.7 Implementation Interface <a href="#section-007" id="section-007"></a>

```text
SSMStateLayer.forward(x, h, graph_context=None) -> y, h_new
```

## 8.8 Residual-Gated SSM Enhancement <a href="#section-008" id="section-008"></a>

In the equivariant self-evolving version, the SSM maintains long-range states and can also act as a short-term physical modulator during inference, receiving residual signals from `SGD-Harness` or the `Posterior Error Estimator`:

$$
h_{t+1}=\bar{A}(x_t,r_t)h_t+\bar{B}(x_t)x_t+G(x_t,r_t)r_t
$$

where:

- $$r_t$$ embeds task, physical, conservation, geometric-validity, or source-confidence residuals;
- $$G(x_t,r_t)$$ is a residual-selective gate;
- high-residual regions allow the SSM to absorb short-term anomalies rapidly;
- low-residual regions retain stable states to avoid meaningless drift.

This forms three tracks of self-evolution:

| Track | Mechanism | Weight changes | Timescale |
|---|---|---|---|
| Fast track | Residual modulation of SSM hidden states | No weight changes | Milliseconds to minutes |
| Medium track | Dynamic Tree grow/shrink + topology refinement | Structure / local-adapter changes | Minutes to hours |
| Slow track | Retrospection replay / distillation / consolidation | Parameters may change | Hours to months |

Enhanced interface:

```text
SSMStateLayer.forward(
  x,
  h,
  graph_context=None,
  residual_context=None,
  gate_policy=None
) -> y, h_new, ssm_trace
```


---

[← Previous](section-07.md) · [Contents](../../../SUMMARY.md) · [Next →](section-09.md)

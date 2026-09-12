# 17. Total Loss Function Design

SGD-Net's training objective can be written as:

$$
\mathcal{L}=\mathcal{L}_{task}
+\lambda_g\mathcal{L}_{graph}
+\lambda_s\mathcal{L}_{state}
+\lambda_t\mathcal{L}_{tree}
+\lambda_e\mathcal{L}_{posterior}
+\lambda_p\mathcal{L}_{projection}
$$

**On This Page**

- [17.1 Task Loss](#section-001)
- [17.2 Graph Constraint Loss](#section-002)
- [17.3 State Stability Loss](#section-003)
- [17.4 Tree Complexity Regularization](#section-004)
- [17.5 Posterior Error Loss](#section-005)
- [17.6 Stability Projection Loss](#section-006)

---

## 17.1 Task Loss <a href="#section-001" id="section-001"></a>

Classification:

$$
\mathcal{L}_{task}=\operatorname{CE}(\hat{y},y)
$$

Regression:

$$
\mathcal{L}_{task}=\|\hat{y}-y\|_2^2
$$

## 17.2 Graph Constraint Loss <a href="#section-002" id="section-002"></a>

$$
\mathcal{L}_{graph}=\sum_{(i,j)\notin\mathcal{E}} s_{ij}^2
$$

Suppress inadmissible implicit connections.

## 17.3 State Stability Loss <a href="#section-003" id="section-003"></a>

$$
\mathcal{L}_{state}=\sum_t\max(0,\|h_t\|_2-c)^2
$$

## 17.4 Tree Complexity Regularization <a href="#section-004" id="section-004"></a>

$$
\mathcal{L}_{tree}=\alpha\cdot \#leaves+\beta\cdot depth(T)
$$

Prevent unbounded tree growth.

## 17.5 Posterior Error Loss <a href="#section-005" id="section-005"></a>

$$
\mathcal{L}_{posterior}=\sum_i \eta_i
$$

## 17.6 Stability Projection Loss <a href="#section-006" id="section-006"></a>

$$
\mathcal{L}_{projection}=\max(0,\rho(A)-\rho_{max})^2
$$


---

[← Previous](section-18.md) · [Contents](../../../SUMMARY.md) · [Next →](section-20.md)

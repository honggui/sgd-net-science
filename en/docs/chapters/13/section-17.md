# 15. SGD-Net Block: Stackable Basic Unit

In an actual model, Layers 3–5 can form a stackable block:

```text
SGDBlock:
  SSMStateLayer
  GNNTopologyLayer
  DynamicTreeRoutingLayer
  ResidualConnection
  Normalization
```

## 15.1 Block Forward Equations <a href="#section-001" id="section-001"></a>

$$
Z_1 = X + \operatorname{SSM}(X,H)
$$

$$
Z_2 = Z_1 + \operatorname{GNN}(Z_1,A,E)
$$

$$
Z_3 = Z_2 + \operatorname{TreeRoute}(Z_2,T)
$$

$$
X' = \operatorname{Norm}(Z_3)
$$

## 15.2 Multilayer Stacking <a href="#section-002" id="section-002"></a>

$$
X^{(l+1)},H^{(l+1)}=\operatorname{SGDBlock}^{(l)}(X^{(l)},H^{(l)},\mathcal{G})
$$

Finally:

$$
\hat{Y}=\operatorname{Head}(X^{(L)})
$$


---

[← Previous](section-16.md) · [Contents](../../../SUMMARY.md) · [Next →](section-18.md)

# 15. SGD-Net Block：可堆叠基本单元

实际模型中，可将 Layer 3-5 组成可堆叠 block：

```text
SGDBlock:
  SSMStateLayer
  GNNTopologyLayer
  DynamicTreeRoutingLayer
  ResidualConnection
  Normalization
```

## 15.1 Block 前向公式 <a href="#section-001" id="section-001"></a>

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

## 15.2 多层堆叠 <a href="#section-002" id="section-002"></a>

$$
X^{(l+1)},H^{(l+1)}=\operatorname{SGDBlock}^{(l)}(X^{(l)},H^{(l)},\mathcal{G})
$$

最终：

$$
\hat{Y}=\operatorname{Head}(X^{(L)})
$$


---

[← 上一页](section-16.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-18.md)

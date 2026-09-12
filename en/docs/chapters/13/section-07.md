# 7. Layer 2: Node and Edge Embedding Layer

**On This Page**

- [7.1 Node Embedding](#section-001)
- [7.2 Edge Embedding](#section-002)
- [7.3 Type Encoding](#section-003)

---

## 7.1 Node Embedding <a href="#section-001" id="section-001"></a>

Initial node features may come from numerical features, text embeddings, structural embeddings, or type embeddings.

$$
X_0 = \phi_v(Z_0)
$$

Here $$\phi_v$$ may be an MLP, an embedding table, a pretrained model, or a handcrafted feature mapping.

## 7.2 Edge Embedding <a href="#section-002" id="section-002"></a>

Edge features may include distance, type, direction, weight, and relation confidence.

$$
e_{ij}=\phi_e(r_{ij}, d_{ij}, type_{ij})
$$

## 7.3 Type Encoding <a href="#section-003" id="section-003"></a>

For heterogeneous graphs, node and edge types should be included:

$$
x_i = [x_i^{raw}; \operatorname{Embed}(type_i)]
$$

$$
e_{ij} = [e_{ij}^{raw}; \operatorname{Embed}(type_{ij})]
$$


---

[← Previous](section-06.md) · [Contents](../../../SUMMARY.md) · [Next →](section-08.md)

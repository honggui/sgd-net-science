# 7. Layer 2：节点与边嵌入层

**本页目录**

- [7.1 节点嵌入](#section-001)
- [7.2 边嵌入](#section-002)
- [7.3 类型编码](#section-003)

---

## 7.1 节点嵌入 <a href="#section-001" id="section-001"></a>

节点初始特征可来自数值特征、文本 embedding、结构 embedding 或类型 embedding。

$$
X_0 = \phi_v(Z_0)
$$

其中 $$\phi_v$$ 可是 MLP、Embedding Table、预训练模型或手工特征映射。

## 7.2 边嵌入 <a href="#section-002" id="section-002"></a>

边特征可包括距离、类型、方向、权重、关系置信度等。

$$
e_{ij}=\phi_e(r_{ij}, d_{ij}, type_{ij})
$$

## 7.3 类型编码 <a href="#section-003" id="section-003"></a>

对异构图，建议加入节点类型和边类型：

$$
x_i = [x_i^{raw}; \operatorname{Embed}(type_i)]
$$

$$
e_{ij} = [e_{ij}^{raw}; \operatorname{Embed}(type_{ij})]
$$


---

[← 上一页](section-06.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-08.md)

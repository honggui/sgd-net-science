# 9. Layer 4: GNN Topology Propagation Layer

**On This Page**

- [9.1 Function](#section-001)
- [9.2 General Message-Passing Operator](#section-002)
- [9.3 Topology Normalization Operator](#section-003)
- [9.4 Edge-Type-Aware Message Passing](#section-004)
- [9.5 Physically Conservative Message Operator](#section-005)
- [9.6 Input and Output Shapes](#section-006)
- [9.7 Implementation Interface](#section-007)

---

## 9.1 Function <a href="#section-001" id="section-001"></a>

The GNN layer propagates information under graph-topology constraints.

It addresses the question:

> From which neighbors may a node receive information? Which physical, chemical, factual, or spatial relations affect the current node?

## 9.2 General Message-Passing Operator <a href="#section-002" id="section-002"></a>

$$
m_{ij}^{(l)}=\psi_m(x_i^{(l)},x_j^{(l)},e_{ij};\theta_m)
$$

$$
m_i^{(l)}=\operatorname{AGG}_{j\in\mathcal{N}(i)} m_{ij}^{(l)}
$$

$$
x_i^{(l+1)}=\psi_u(x_i^{(l)},m_i^{(l)};\theta_u)
$$

where:

- $$\psi_m$$ is the message function;
- $$\operatorname{AGG}$$ may be sum, mean, max, or attention;
- $$\psi_u$$ is the node-update function.

## 9.3 Topology Normalization Operator <a href="#section-003" id="section-003"></a>

The simplest implementation can use:

$$
X^{(l+1)}=\sigma(D^{-1}AX^{(l)}W)
$$

where:

$$
D_{ii}=\sum_j A_{ij}
$$

## 9.4 Edge-Type-Aware Message Passing <a href="#section-004" id="section-004"></a>

In heterogeneous graphs, different edge types use different parameters:

$$
m_{ij}=W_{type(e_{ij})}x_j
$$

$$
x_i'=\sigma\left(W_sx_i+\sum_{j\in\mathcal{N}(i)}\alpha_{ij}W_{type(e_{ij})}x_j\right)
$$

## 9.5 Physically Conservative Message Operator <a href="#section-005" id="section-005"></a>

For certain physical tasks, antisymmetric messages can be designed:

$$
m_{ij}=-m_{ji}
$$

This makes conservation relations easier to express.

For example:

$$
m_{ij}=\phi(x_i,x_j,e_{ij})-\phi(x_j,x_i,e_{ji})
$$

## 9.6 Input and Output Shapes <a href="#section-006" id="section-006"></a>

```text
Input:
  x:          [N, F]
  edge_index: [2, E]
  edge_attr:  [E, D_e]
Output:
  x_out:      [N, F_g]
```

## 9.7 Implementation Interface <a href="#section-007" id="section-007"></a>

```text
GNNTopologyLayer.forward(x, edge_index, edge_attr=None) -> x_out
```


---

[← Previous](section-08.md) · [Contents](../../../SUMMARY.md) · [Next →](section-10.md)

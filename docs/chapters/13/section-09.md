# 9. Layer 4：GNN 拓扑传播层

**本页目录**

- [9.1 功能](#section-001)
- [9.2 通用消息传递算子](#section-002)
- [9.3 拓扑归一化算子](#section-003)
- [9.4 边类型感知消息传递](#section-004)
- [9.5 物理守恒型消息算子](#section-005)
- [9.6 输入输出形状](#section-006)
- [9.7 工程接口](#section-007)

---

## 9.1 功能 <a href="#section-001" id="section-001"></a>

GNN 层负责在图拓扑约束下传播信息。

它回答的问题是：

> 节点只能从哪些邻居接收信息？哪些物理、化学、事实或空间关系会影响当前节点？

## 9.2 通用消息传递算子 <a href="#section-002" id="section-002"></a>

$$
m_{ij}^{(l)}=\psi_m(x_i^{(l)},x_j^{(l)},e_{ij};\theta_m)
$$

$$
m_i^{(l)}=\operatorname{AGG}_{j\in\mathcal{N}(i)} m_{ij}^{(l)}
$$

$$
x_i^{(l+1)}=\psi_u(x_i^{(l)},m_i^{(l)};\theta_u)
$$

其中：

- $$\psi_m$$ 是消息函数；
- $$\operatorname{AGG}$$ 可为 sum、mean、max、attention；
- $$\psi_u$$ 是节点更新函数。

## 9.3 拓扑归一化算子 <a href="#section-003" id="section-003"></a>

最简实现可使用：

$$
X^{(l+1)}=\sigma(D^{-1}AX^{(l)}W)
$$

其中：

$$
D_{ii}=\sum_j A_{ij}
$$

## 9.4 边类型感知消息传递 <a href="#section-004" id="section-004"></a>

异构图中，不同边类型使用不同参数：

$$
m_{ij}=W_{type(e_{ij})}x_j
$$

$$
x_i'=\sigma\left(W_sx_i+\sum_{j\in\mathcal{N}(i)}\alpha_{ij}W_{type(e_{ij})}x_j\right)
$$

## 9.5 物理守恒型消息算子 <a href="#section-005" id="section-005"></a>

对某些物理任务，可设计反对称消息：

$$
m_{ij}=-m_{ji}
$$

从而更容易表达守恒关系。

例如：

$$
m_{ij}=\phi(x_i,x_j,e_{ij})-\phi(x_j,x_i,e_{ji})
$$

## 9.6 输入输出形状 <a href="#section-006" id="section-006"></a>

```text
Input:
  x:          [N, F]
  edge_index: [2, E]
  edge_attr:  [E, D_e]
Output:
  x_out:      [N, F_g]
```

## 9.7 工程接口 <a href="#section-007" id="section-007"></a>

```text
GNNTopologyLayer.forward(x, edge_index, edge_attr=None) -> x_out
```


---

[← 上一页](section-08.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-10.md)

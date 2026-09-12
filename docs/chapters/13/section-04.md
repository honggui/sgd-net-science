# 4. 统一符号与张量定义

## 4.1 基本图状态 <a href="#section-001" id="section-001"></a>

令当前图为：

$$
\mathcal{G}_t=(\mathcal{V}_t,\mathcal{E}_t,X_t,E_t,A_t)
$$

其中：

| 符号 | 含义 | 典型形状 |
|---|---|---|
| $$N_t$$ | 当前节点数 | scalar |
| $$F$$ | 节点特征维度 | scalar |
| $$D_e$$ | 边特征维度 | scalar |
| $$X_t$$ | 节点特征矩阵 | $$[N_t,F]$$ |
| $$E_t$$ | 边特征矩阵 | $$[|\mathcal{E}_t|,D_e]$$ |
| $$A_t$$ | 邻接矩阵 | $$[N_t,N_t]$$ |
| `edge_index` | 稀疏边索引 | $$[2,|\mathcal{E}_t|]$$ |
| $$H_t$$ | SSM 隐状态 | $$[N_t,D_h]$$ 或 $$[B,N_t,D_h]$$ |
| $$T_i$$ | 第 $$i$$ 个节点内部动态树 | object |
| $$\eta_i$$ | 第 $$i$$ 个节点后验误差 | scalar |

## 4.2 批处理形状 <a href="#section-002" id="section-002"></a>

若存在 batch，则可写为：

$$
X \in \mathbb{R}^{B\times N\times F}
$$

但动态图增长会导致不同样本的 $$N$$ 不一致，因此工程上更建议使用 PyTorch Geometric 类似的稀疏批处理：

```text
x:          [total_nodes, feature_dim]
edge_index: [2, total_edges]
batch:      [total_nodes]
```


---

[← 上一页](section-03.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-05.md)

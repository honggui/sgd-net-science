# 8. Layer 3：SSM 状态演化层

**本页目录**

- [8.1 功能](#section-001)
- [8.2 基础状态空间算子](#section-002)
- [8.3 选择性 SSM 算子](#section-003)
- [8.4 图条件 SSM](#section-004)
- [8.5 稳定离散化](#section-005)
- [8.6 输入输出形状](#section-006)
- [8.7 工程接口](#section-007)
- [8.8 Residual-gated SSM 增强](#section-008)

---

## 8.1 功能 <a href="#section-001" id="section-001"></a>

SSM 层负责对每个节点或全图状态进行长程显式递推；隐状态不等于隐式求解，后者另由 IM01—IM06 提供。

它回答的问题是：

> 当前节点状态如何受历史状态、时间序列、推理链或扩散步骤影响？

## 8.2 基础状态空间算子 <a href="#section-002" id="section-002"></a>

连续形式：

$$
\dot{h}(t)=Ah(t)+Bx(t), \qquad y(t)=Ch(t)+Dx(t)
$$

离散形式：

$$
h_t=\bar{A}h_{t-1}+\bar{B}x_t
$$

$$
y_t=Ch_t+Dx_t
$$

## 8.3 选择性 SSM 算子 <a href="#section-003" id="section-003"></a>

在 Mamba/Selective SSM 中，状态转移可依赖输入：

$$
h_t=\bar{A}(x_t)h_{t-1}+\bar{B}(x_t)x_t
$$

$$
y_t=C(x_t)h_t+D(x_t)x_t
$$

SGD-Net 中可将每个图节点看成一条局部状态流：

$$
h_{i,t}=\bar{A}_i(x_{i,t})h_{i,t-1}+\bar{B}_i(x_{i,t})x_{i,t}
$$

## 8.4 图条件 SSM <a href="#section-004" id="section-004"></a>

SGD-Net 更进一步，可让状态转移受图上下文影响：

$$
\tilde{x}_{i,t}=\operatorname{Agg}_{j\in\mathcal{N}(i)}(x_{j,t},e_{ij})
$$

$$
h_{i,t}=\bar{A}(x_{i,t},\tilde{x}_{i,t})h_{i,t-1}+\bar{B}(x_{i,t})x_{i,t}
$$

这使 SSM 不只是序列算子，而是受拓扑调制的状态演化算子。

## 8.5 稳定离散化 <a href="#section-005" id="section-005"></a>

可采用双线性变换：

$$
\bar{A}=\left(I-\frac{\Delta t}{2}A\right)^{-1}\left(I+\frac{\Delta t}{2}A\right)
$$

若 $$\operatorname{Re}(\lambda(A))\le 0$$，则有助于控制 $$\rho(\bar{A})\le 1$$。

## 8.6 输入输出形状 <a href="#section-006" id="section-006"></a>

```text
Input:
  x:     [N, F]
  h:     [N, D_h]
Output:
  y:     [N, D_s]
  h_new: [N, D_h]
```

## 8.7 工程接口 <a href="#section-007" id="section-007"></a>

```text
SSMStateLayer.forward(x, h, graph_context=None) -> y, h_new
```

## 8.8 Residual-gated SSM 增强 <a href="#section-008" id="section-008"></a>

在等变自进化版本中，SSM 不仅维护长程状态，还可作为推理期短时物理调制器，接收 `SGD-Harness` 或 `Posterior Error Estimator` 产生的残差信号：

$$
h_{t+1}=\bar{A}(x_t,r_t)h_t+\bar{B}(x_t)x_t+G(x_t,r_t)r_t
$$

其中：

- $$r_t$$ 是任务残差、物理残差、守恒残差、几何合法性残差或来源置信残差的 embedding；
- $$G(x_t,r_t)$$ 是残差选择性门控；
- 高残差区域允许 SSM 快速吸收短时异常；
- 低残差区域保持稳定状态，避免无意义漂移。

这形成三轨自进化：

| 轨道 | 机制 | 是否改权重 | 时间尺度 |
|---|---|---|---|
| 快轨 | SSM 隐状态残差调制 | 不改权重 | 毫秒到分钟 |
| 中轨 | Dynamic Tree grow/shrink + topology refinement | 改结构/局部 adapter | 分钟到小时 |
| 慢轨 | Retrospection replay / distillation / consolidation | 可改参数 | 小时到月 |

增强接口：

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

[← 上一页](section-07.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-09.md)

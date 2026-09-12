# 10. Layer 5：动态树路由层

**本页目录**

- [10.1 功能](#section-001)
- [10.2 每节点树结构](#section-002)
- [10.3 硬路由算子](#section-003)
- [10.4 软路由算子](#section-004)
- [10.5 局部专家算子](#section-005)
- [10.6 分裂算子](#section-006)
- [10.7 剪枝算子](#section-007)
- [10.8 几何 grow / shrink 控制](#section-008)
- [10.9 输入输出形状](#section-009)
- [10.10 工程接口](#section-010)

---

## 10.1 功能 <a href="#section-001" id="section-001"></a>

动态树层是 SGD-Net 的核心差异点之一。它不是普通决策树后处理，而是嵌入在每个图节点或子图内部的局部自适应路由结构。

它回答的问题是：

> 对某个节点，当前特征应该走哪条局部分支？是否需要分裂出更细的局部表示空间？

## 10.2 每节点树结构 <a href="#section-002" id="section-002"></a>

每个图节点 $$i$$ 对应一棵树 $$T_i$$：

```text
Graph Node i
  └── DynamicTree T_i
        ├── predicate: x[k] <= threshold
        ├── left child
        └── right child
```

## 10.3 硬路由算子 <a href="#section-003" id="section-003"></a>

对树节点 $$q$$，设分裂特征为 $$k_q$$，阈值为 $$\tau_q$$：

$$
route_q(x)=
\begin{cases}
left(q), & x_{k_q}\le \tau_q \\
right(q), & x_{k_q}>\tau_q
\end{cases}
$$

最终到达叶节点 $$\ell$$，输出：

$$
\tilde{x}_i=x_i+b_{\ell}
$$

其中 $$b_{\ell}$$ 是叶节点局部偏置或局部专家参数。

## 10.4 软路由算子 <a href="#section-004" id="section-004"></a>

为了可微训练，可使用 sigmoid 门控：

$$
g_q(x)=\sigma\left(\frac{x_{k_q}-\tau_q}{T}\right)
$$

$$
Tree_q(x)=(1-g_q(x))Tree_{left}(x)+g_q(x)Tree_{right}(x)
$$

其中 $$T$$ 是温度参数。$$T\to 0$$ 时趋近硬路由。

## 10.5 局部专家算子 <a href="#section-005" id="section-005"></a>

叶节点可不只是 bias，也可以是局部 MLP / adapter：

$$
\tilde{x}_i=x_i+\operatorname{Adapter}_{\ell}(x_i)
$$

## 10.6 分裂算子 <a href="#section-006" id="section-006"></a>

当后验误差 $$\eta_i$$ 超过阈值时，对节点 $$i$$ 的树执行：

$$
T_i \leftarrow \operatorname{Split}(T_i,k^*,\tau^*)
$$

分裂特征可按以下策略选择：

1. 最大梯度维度；
2. 最大残差相关维度；
3. 最大方差维度；
4. 随机探索；
5. 任务先验指定。

在几何任务中，动态树必须优先读取不变标量，避免直接依赖绝对坐标破坏等变性。可选几何分裂特征包括：

| 特征 | 用途 |
|---|---|
| $$\|\mathbf{x}_i-\mathbf{x}_j\|^2$$ | 局部拉伸/压缩 |
| $$\|\mathbf{F}_i\|$$ | 受力异常 |
| local energy density | 高能区域 refinement |
| residual norm | 误差热点 |
| strain / curvature scalar | 晶格缺陷、裂纹、相变 |
| $$\|\dot{h}_i\|$$ | SSM 隐状态突变 |
| uncertainty scalar | 探索性预分裂 |

## 10.7 剪枝算子 <a href="#section-007" id="section-007"></a>

当某个子树长期低贡献或高风险时：

$$
T_i \leftarrow \operatorname{Prune}(T_i, subtree)
$$

剪枝条件包括：

- 访问频次低；
- 残差持续不下降；
- 置信度过低；
- 被判定为投毒/异常；
- 超出计算预算。

## 10.8 几何 grow / shrink 控制 <a href="#section-008" id="section-008"></a>

对于 EGNN / 晶体 / PDE / 粒子场，动态树不仅负责抽象路由，也可负责类似自适应有限元的局部加密与粗化：

```text
if conservation_check_invalid_or_failed:
    reject_candidate_or_use_declared_fallback
elif observation_missing_or_noise_dominated:
    request_observation_or_keep_unknown
elif persistent_residual_high and capacity_or_resolution_cause_supported:
    propose_isolated_refinement_with_migration
elif residual_low and merge_consistency_valid and benefit_expected:
    propose_isolated_merge
else:
    keep
# 所有候选经收益、回归、数值、资源与版本验证后，窗口间自动接纳。
```

对应关系：

| 自适应 FEM | SGD-Net 几何动态树 |
|---|---|
| 网格加密 | graph grow / node split |
| 网格粗化 | graph shrink / node merge |
| 后验误差估计 | Harness residual / posterior error |
| 稳定性修复 | Stability Projection / rollback |

几何动态树必须把每次 grow/shrink 写入 `ControlTrace` 与 `geometry_trace`，用于后续 CModel replay 和硬件机会分析。

## 10.9 输入输出形状 <a href="#section-009" id="section-009"></a>

```text
Input:
  x:          [N, F]
  node_trees: List[DynamicTree]
Output:
  x_routed:   [N, F]
  route_info: List[RouteEvent]
```

## 10.10 工程接口 <a href="#section-010" id="section-010"></a>

```text
DynamicTreeRoutingLayer.forward(x, node_trees) -> x_routed, route_info
DynamicTreeRoutingLayer.split(node_id, reason) -> SplitEvent
DynamicTreeRoutingLayer.prune(node_id, subtree_id, reason) -> PruneEvent
```


---

[← 上一页](section-11.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-13.md)

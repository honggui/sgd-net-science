# 1. 核心结论

**SGD-Net 是一个整体 AI Model 架构，同时也是由多个可插拔子模块构成的复合模型。**

更准确地说：

> SGD-Net 不是单个普通 layer，也不是单一算法模块，而是一种“状态空间演化 + 图拓扑传播 + 动态树自适应 + 后验误差控制 + 稳定性投影”的整体神经架构范式。

它类似 Transformer 的层级关系：

- Transformer 是整体架构；
- 内部由 embedding、multi-head attention、FFN、LayerNorm、residual connection、position encoding 等模块组成。

SGD-Net 也是如此：

- SGD-Net 是整体架构；
- 内部由 `Input Graph Builder`、`SSM State Layer`、`GNN Topology Layer`、`Dynamic Tree Routing Layer`、`Posterior Error Estimator`、`Topology Refiner`、`Stability Projector`、`Task Head` 等模块构成。

在 [新Idea探讨_20260613_等变GNN与自进化架构](../../24-新Idea探讨_20260613_等变GNN与自进化架构.md) 和 [SGD-Harness物理守恒哨兵与等变自进化闭环](../../25-SGD-Harness物理守恒哨兵与等变自进化闭环.md) 之后，推荐把 SGD-Net 的增强定义进一步写成：

```text
SSM
+ Topological / Equivariant GNN
+ Geometry-aware Dynamic Tree
+ Physics Harness
+ Stability Projection
+ Retrospection
```

其中普通 `GNN Topology Layer` 继续服务知识图谱、流程图、MoE 路由等纯拓扑任务；新增的 `EquivariantGNNLayer` 服务分子、晶体、PDE 粒子场、机器人和其他显式 3D 几何任务；`SGD-Harness` 则把后验误差、物理守恒、安全审计、fallback 和复盘入口统一为运行时守恒哨兵。

在 独立研究材料（本包不附） 之后，进一步建议把多度量/非欧/准度量能力定义为远期可选的 `Metric-aware Manifold Propagation Layer`：它不替代 TopoGNN 或 EGNN，而是在层级图、偏序知识空间、实验可达性、混合曲率表征或世界模型 bridge 中，通过 `MetricTrace` / `ManifoldTrace` 记录和验证。

在 独立研究材料（本包不附） 之后，建议把 AQAD 的“量化感知 + 动态步长 + 静态最大形状 + 动态掩码 + early release”降维为 SGD-Net Runtime 机制：新增 `AdaptiveBudgetController`、`QuantizationAwareStateCache` 与 `StaticMaxGraphMask`。它们不改变 SGD-Net 的五类核心能力，而是决定动态树 refinement、SSM rollout、Harness 复核、低比特状态缓存和 fallback 的预算与部署形态。

在 [SGD-Net与JEPA世界模型融合及JSBO桥梁算子](../../18-SGD-Net与JEPA世界模型融合及JSBO桥梁算子.md)、[SGD-Net混合求解训练体系与世界模型研究路线](../../27-SGD-Net混合求解训练体系与世界模型研究路线.md) 和 独立研究材料（本包不附） 的最新梳理之后，还需要补充一个系统级边界：`13` 号文档可以作为**算法内核与算子 SSOT**，但不能单独替代 `11/16/30/31` 的硬件系统文档、`03-06` 的实现计划、`18` 的 JSBO/LeJEPA 理论文档或 `SGD-Trace与Benchmark规范/` 的证据合同。对 `00`～`31` 的全部场景，应采用“内生核心 + 扩展模块 + 外部后端/peer IP”的分层支持方式。

因此，一句话回答：

> SGD-Net 是一个整体 AI Model 架构；该架构的可实现形态由多个子模块和层算子组合而成，各模块可以按任务裁剪、替换或堆叠。


---

[← 上一页](../../13-SGD-Net模型架构与层算子详解.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-02.md)

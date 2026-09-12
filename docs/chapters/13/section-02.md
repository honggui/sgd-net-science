# 2. SGD-Net 的架构边界

## 2.1 SGD-Net 不是什么 <a href="#section-001" id="section-001"></a>

SGD-Net 不是：

1. 单纯的 SSM/Mamba；
2. 单纯的 GNN；
3. 单纯的可微决策树；
4. 单纯的 MoE router；
5. 单纯的后处理器；
6. 单纯的 PDE/PINN 求解器；
7. 单纯的 AlphaFold/RFdiffusion 插件。

这些都可以成为 SGD-Net 的组成部分或应用场景，但不能单独代表 SGD-Net。

## 2.2 SGD-Net 是什么 <a href="#section-002" id="section-002"></a>

SGD-Net 是一种复合神经架构，其最小定义包括五类必要能力：

| 能力 | 对应模块 | 是否核心 |
|---|---|---|
| 长程状态演化 | SSM / Mamba / recurrent state operator | 必须 |
| 拓扑/几何约束传播 | Topological GNN / Equivariant GNN / message passing / graph operator | 必须 |
| 局部自适应路由 | Dynamic Tree / geometry-aware grow-shrink / adaptive routing | 必须 |
| 后验误差驱动 | Posterior Error Estimator / SGD-Harness residual checker | 必须 |
| 稳定性约束 | Stability Projector / Lyapunov constraint | 必须 |

如果只保留 SSM + GNN，而没有动态树和后验误差驱动，则更像 `State-space Graph Network`，还不能称为完整 SGD-Net。

如果只保留 GNN + 动态树，而没有 SSM，则缺少长程状态演化能力。

如果只在输出端做一次错误筛选，而不反馈到拓扑重构或局部 refinement，也不能称为完整 SGD-Net。


---

[← 上一页](section-01.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-03.md)

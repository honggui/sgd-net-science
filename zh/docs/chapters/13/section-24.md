# 22. 与其他架构的关系

| 架构 | 与 SGD-Net 的关系 |
|---|---|
| Transformer | 可作为输入 encoder 或 task head，不是核心必需 |
| Mamba/SSM | SGD-Net 的状态演化核心之一 |
| GNN | SGD-Net 的拓扑传播核心之一 |
| Decision Tree | SGD-Net 的局部自适应核心之一 |
| MoE | 可由动态树扩展为层级专家路由 |
| PINN | 可作为 PDE 后验误差或任务损失的一种形式 |
| Neural Operator | 可作为科学计算 baseline 或局部 operator |
| AlphaFold/RFdiffusion | 可作为外部基础模型，被 SGD-Net 增强或诊断 |
| JEPA / I-JEPA / V-JEPA | 可作为外部世界模型 encoder/predictor，经 JSBO 桥梁转换为结构化 GraphState 或 PhysicalState |
| G-JEPA / G-GeoS | 可作为远期研究假设，把 JEPA latent prediction 与等变、多度量、可达性和物理 residual 结合；当前只进入 trace/benchmark，不作为已验证 AGI 结论 |


---

[← 上一页](section-23.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-25.md)

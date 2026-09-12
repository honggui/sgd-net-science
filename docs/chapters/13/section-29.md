# 27. 推荐开发顺序

为了支撑 `00`～`31` 的全部算法、框架和应用场景，开发顺序不应停留在早期 MVP，而应按“核心可跑 → 桥梁可接 → 求解可路由 → Trace 可审计 → 后端可替换 → 应用可验收”的节奏推进：

1. `GraphState` 与图构造；
2. 简化 GNN / EGNN 几何状态；
3. 简化 SSM / Mamba-style scan；
4. 动态树路由与 MoE router compatibility；
5. 后验误差估计与 SGD-Harness；
6. 节点分裂、拓扑重构与稳定性投影；
7. `SGDBlock` 封装与任务输出头；
8. `TraceEmitter`、`WorkloadTraceRecorder`、`ReplayBundle` 与基础 benchmark；
9. `JEPAAdapter`、`JEPALatentTensor`、`JSBOBridgeOperator`；
10. `SIGRegGaussianRegularizer`、`LeJEPAIdentifiabilityVerifier`、`GaussianityReport`；
11. `SolverRouter` 与 BP/Adam、Riemannian、ADMM/Prox、PPO/Actor-Critic、ES/BO、Filter family 的路由；
12. `OnlineAdaptationController` 与 EKF/UKF/Particle/Kalman 状态修正；
13. `CollectiveMergeManager` 与 FedAvg、Model Soup、Task Arithmetic、TIES、distillation 安全合并；
14. `DomainAdapterRegistry` 与 bio/materials/sdl/phys/chem/med/MoE adapter；
15. `BackendRegistry`、`PlacementPlanner`、`FallbackManager` 与 CPU/GPU/NPU/SGD-TPU/PPU/Bio-NMC/EICU/SNVE/BICU connector；
16. CModel shadow backend、硬件机会报告和 HLO-OP-IR / SGD-UXS lowering 合同；
17. AlphaFold/RFdiffusion/ESMFold、材料/DFT、PIEVO/SDL、JEPA/Reacher 等应用 benchmark；
18. 安全审计、来源标签、ShadowGraphTree 与人类在环控制；
19. PyTorch/PyG/Triton/CUDA/FPGA/CModel 的分层实现；
20. 以 独立研究材料（本包不附）、`SGD-Trace与Benchmark规范/` 和 CModel/FPGA 结果回填论文、专利和商业文档。


---

[← 上一页](section-28.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-30.md)

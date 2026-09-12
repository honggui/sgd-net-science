# 27. Recommended Development Order

To support all algorithms, frameworks, and application scenarios in `00`–`31`, development should progress beyond the early MVP in this sequence: runnable core → connectable bridges → routable solvers → auditable traces → replaceable backends → verifiable applications:

1. `GraphState` and graph construction;
2. Simplified GNN / EGNN geometric states;
3. simplified SSM / Mamba-style scan;
4. Dynamic tree routing and MoE router compatibility;
5. Posterior error estimation and SGD-Harness;
6. Node splitting, topology reconstruction, and stability projection;
7. `SGDBlock` encapsulation and task output heads;
8. `TraceEmitter`, `WorkloadTraceRecorder`, `ReplayBundle`, and basic benchmarks;
9. `JEPAAdapter`, `JEPALatentTensor`, `JSBOBridgeOperator`;
10. `SIGRegGaussianRegularizer`, `LeJEPAIdentifiabilityVerifier`, `GaussianityReport`;
11. `SolverRouter` and routing for BP/Adam, Riemannian, ADMM/Prox, PPO/Actor-Critic, ES/BO, and Filter families;
12. `OnlineAdaptationController` and EKF/UKF/Particle/Kalman state corrections;
13. `CollectiveMergeManager` and safe merging through FedAvg, Model Soup, Task Arithmetic, TIES, and distillation;
14. `DomainAdapterRegistry` and bio/materials/sdl/phys/chem/med/MoE adapters;
15. `BackendRegistry`, `PlacementPlanner`, `FallbackManager`, and CPU/GPU/NPU/SGD-TPU/PPU/Bio-NMC/EICU/SNVE/BICU connectors;
16. CModel shadow backend, hardware-opportunity reports, and HLO-OP-IR / SGD-UXS lowering contracts;
17. Application benchmarks for AlphaFold/RFdiffusion/ESMFold, materials/DFT, PIEVO/SDL, JEPA/Reacher, and others;
18. Safety audits, provenance labels, ShadowGraphTree, and human-in-the-loop control;
19. Layered implementations in PyTorch/PyG/Triton/CUDA/FPGA/CModel;
20. Update papers, patents, and business documents using independent research materials (not included in this package), `SGD-Trace与Benchmark规范/`, and CModel/FPGA results.


---

[← Previous](section-28.md) · [Contents](../../../SUMMARY.md) · [Next →](section-30.md)

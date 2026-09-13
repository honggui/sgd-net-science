# Summary

* [Languages / 语言入口](README.md)

## 中文


* [首页](zh/README.md)

## 中文 · 入门

* [从这里开始](zh/guide/start-here.md)
* [术语与约定](zh/guide/glossary.md)
* [研究状态](zh/guide/status.md)

## 中文 · 研究定位

* [01 · 论文与研究问题](zh/docs/01.md)
* [32 · 融合评估与创新边界](zh/docs/32.md)
* [47 · 通用性与能力边界](zh/docs/47.md)
* [Transformer 算子与架构对比](zh/guide/transformer-comparison.md)
* [图记忆、数学等价性与计算基础](zh/guide/mathematical-computing-foundation.md)

## 中文 · 核心模型架构

* [36 · 统一架构与组件全集](zh/docs/36.md)
  * [摘要与详细版阅读方法](zh/docs/chapters/36/section-01.md)
  * [1. 架构决策与阅读入口](zh/docs/chapters/36/section-02.md)
  * [2. 与已有材料的对应](zh/docs/chapters/36/section-03.md)
  * [3. 系统连接图](zh/docs/chapters/36/section-04.md)
  * [4. 组件与单元全集：9 组、72 个登记单元](zh/docs/chapters/36/section-05.md)
  * [5. 张量、状态和时钟合同](zh/docs/chapters/36/section-06.md)
  * [6. 一个块到底如何计算](zh/docs/chapters/36/section-07.md)
  * [7. 初始配置与可复现前向顺序](zh/docs/chapters/36/section-08.md)
  * [8. 从计算图到芯片](zh/docs/chapters/36/section-09.md)
  * [9. 验收与当前交付边界](zh/docs/chapters/36/section-10.md)
  * [10. 李亚普洛夫修订记录](zh/docs/chapters/36/section-11.md)
  * [11. 隐式求解与持续演化的显式接入（2026-09-12）](zh/docs/chapters/36/section-12.md)
  * [12. 能力优先与可选兼容扩展（2026-09-12 修订）](zh/docs/chapters/36/section-13.md)
  * [13. 函数/算子空间与数值误差接口（2026-09-12）](zh/docs/chapters/36/section-14.md)
  * [反思、回顾与复盘补充（2026-09-12）](zh/docs/chapters/36/section-15.md)
  * [14. 统一数学建模：任务、模型与适应过程](zh/docs/chapters/36/section-16.md)
  * [15. A 组详细构建：把观测变成可解释的输入](zh/docs/chapters/36/section-17.md)
  * [16. B 组详细构建：模态编码与可测量状态的连接](zh/docs/chapters/36/section-18.md)
  * [17. C 组详细构建：图、几何与表示层级](zh/docs/chapters/36/section-19.md)
  * [18. D 组详细构建：时空块的可训练内核](zh/docs/chapters/36/section-20.md)
  * [19. E 组详细构建：预测、生成与任务读出](zh/docs/chapters/36/section-21.md)
  * [20. F 组详细构建：误差驱动的有界适应](zh/docs/chapters/36/section-22.md)
  * [21. G 组详细构建：训练、复盘和持续学习](zh/docs/chapters/36/section-23.md)
  * [22. H 组详细构建：工具、执行与证据](zh/docs/chapters/36/section-24.md)
  * [23. L 组详细构建：稳定性、耗散与修正](zh/docs/chapters/36/section-25.md)
  * [24. 内部扩展的数学职责与接入位置](zh/docs/chapters/36/section-26.md)
  * [25. 完整模型如何构建和执行](zh/docs/chapters/36/section-27.md)
  * [26. 可以论证的性质、条件和反例](zh/docs/chapters/36/section-28.md)
  * [27. 复杂度、资源与实验构建](zh/docs/chapters/36/section-29.md)
* [13 · 层与算子详解](zh/docs/13.md)
  * [1. 核心结论](zh/docs/chapters/13/section-01.md)
  * [2. SGD-Net 的架构边界](zh/docs/chapters/13/section-02.md)
  * [3. 整体分层架构](zh/docs/chapters/13/section-03.md)
  * [4. 统一符号与张量定义](zh/docs/chapters/13/section-04.md)
  * [5. Layer 0：输入编码层](zh/docs/chapters/13/section-05.md)
  * [6. Layer 1：图构造层](zh/docs/chapters/13/section-06.md)
  * [7. Layer 2：节点与边嵌入层](zh/docs/chapters/13/section-07.md)
  * [8. Layer 3：SSM 状态演化层](zh/docs/chapters/13/section-08.md)
  * [9. Layer 4：GNN 拓扑传播层](zh/docs/chapters/13/section-09.md)
  * [9b. Layer 4b：等变几何传播层](zh/docs/chapters/13/section-10.md)
  * [9c. Layer 4c：多度量/流形传播层（远期可选）](zh/docs/chapters/13/section-11.md)
  * [10. Layer 5：动态树路由层](zh/docs/chapters/13/section-12.md)
  * [11. Layer 6：后验误差估计层](zh/docs/chapters/13/section-13.md)
  * [12. Layer 7：拓扑自适应重构层](zh/docs/chapters/13/section-14.md)
  * [13. Layer 8：稳定性投影层](zh/docs/chapters/13/section-15.md)
  * [14. Layer 9：任务输出头](zh/docs/chapters/13/section-16.md)
  * [15. SGD-Net Block：可堆叠基本单元](zh/docs/chapters/13/section-17.md)
  * [16. 完整前向流程](zh/docs/chapters/13/section-18.md)
  * [17. 总损失函数设计](zh/docs/chapters/13/section-19.md)
  * [18. 算子与 Layer 的关系](zh/docs/chapters/13/section-20.md)
  * [19. 最小可实现版本 MVP](zh/docs/chapters/13/section-21.md)
  * [20. 标准版 SGD-Net](zh/docs/chapters/13/section-22.md)
  * [21. 完整版 SGD-Net](zh/docs/chapters/13/section-23.md)
  * [22. 与其他架构的关系](zh/docs/chapters/13/section-24.md)
  * [23. 实现建议目录](zh/docs/chapters/13/section-25.md)
  * [24. 核心类接口草案](zh/docs/chapters/13/section-26.md)
  * [25. 复杂度分析](zh/docs/chapters/13/section-27.md)
  * [26. 设计不变量](zh/docs/chapters/13/section-28.md)
  * [27. 推荐开发顺序](zh/docs/chapters/13/section-29.md)
  * [29. 总结](zh/docs/chapters/13/section-30.md)
* [24 · 等变 GNN 与状态模型](zh/docs/24.md)
* [18 · JEPA 与世界模型桥接](zh/docs/18.md)
* [33 · 拓扑接触世界模型提案](zh/docs/33.md)

## 中文 · 数学、学习与持续演化

* [44 · 李亚普洛夫稳定性](zh/docs/44.md)
* [45 · 隐式求解](zh/docs/45.md)
* [46 · 动态生长与持续演化](zh/docs/46.md)
* [48 · 函数空间与数值误差](zh/docs/48.md)
* [27 · 混合求解与训练](zh/docs/27.md)
* [14 · 双闭环控制与认知类比](zh/docs/14.md)
* [17 · 复盘与自进化设计](zh/docs/17.md)
* [50 · 反思、回顾与复盘](zh/docs/50.md)
* [记忆—推理巩固与并行生成](zh/guide/memory-reasoning-parallel-generation.md)
* [25 · 物理约束与执行检查](zh/docs/25.md)

## 中文 · 领域模型与科学应用

* [35 · 跨领域扩展路线](zh/docs/35.md)
* [37 · 世界模型](zh/docs/37.md)
* [38 · 物理 AI 与灵巧手](zh/docs/38.md)
* [39 · 制药与分子](zh/docs/39.md)
* [40 · 生物与细胞](zh/docs/40.md)
* [41 · 材料](zh/docs/41.md)
* [42 · 神经行为](zh/docs/42.md)
* [43 · 信号与突触](zh/docs/43.md)
* [07 · 科学大模型扩展](zh/docs/07.md)
* [08 · 科研 Agent 与材料筛选](zh/docs/08.md)
* [15 · 具身实验与数字孪生](zh/docs/15.md)

## 中文 · 实现与实验验证

* [03 · 实现技术白皮书](zh/docs/03.md)
* [04 · 实现概要设计](zh/docs/04.md)
* [05 · 实现详细设计](zh/docs/05.md)
* [34 · 跨本体验证与成本](zh/docs/34.md)
* [49 · 任务卡与评估路线](zh/docs/49.md)

## 中文 · 附录

* [组件登记索引](zh/architecture/README.md)
* [公式与图示检查](zh/guide/rendering-check.md)
* [许可说明](zh/NOTICE.md)
* [接入与维护](zh/GITBOOK_SETUP.md)

* [在线阅读与发布维护](zh/guide/publishing.md)

* [双语文档结构与维护](zh/guide/bilingual.md)

## English


* [Home](en/README.md)

## English · Getting started

* [Start here](en/guide/start-here.md)
* [Terminology and reading conventions](en/guide/glossary.md)
* [Research status and contribution boundaries](en/guide/status.md)

## English · Research framing

* [01 · SGD-Net: An Error-Driven Adaptive Neural Architecture Combining State-Space Models, Topological/Equivariant Graph Neural Networks, and Adaptive Decision Trees](en/docs/01.md)
* [32 · SGD-Net Research Integration Assessment and Innovation Boundaries](en/docs/32.md)
* [47 · SGD-Net Generality Review and Constructible Model Coverage Proposal](en/docs/47.md)
* [SGD-Net and Transformer: Operator Coverage, Architectural Differences, and Proposed Additions](en/guide/transformer-comparison.md)
* [Graph Memory, Model Equivalence, and Mathematics-Driven Software and Hardware Planning](en/guide/mathematical-computing-foundation.md)

## English · Core model architecture

* [36 · SGD-Net Unified Foundation Model Architecture and Complete Component Set](en/docs/36.md)
  * [Abstract and Guide to the Detailed Edition](en/docs/chapters/36/section-01.md)
  * [1. Architectural Decisions and Reading Entry Points](en/docs/chapters/36/section-02.md)
  * [2. Correspondence with Existing Materials](en/docs/chapters/36/section-03.md)
  * [3. System Connection Diagram](en/docs/chapters/36/section-04.md)
  * [4. Complete Component and Unit Set: 9 Groups, 72 Registered Units](en/docs/chapters/36/section-05.md)
  * [5. Tensor, State, and Clock Contracts](en/docs/chapters/36/section-06.md)
  * [6. How a Block Actually Computes](en/docs/chapters/36/section-07.md)
  * [7. Initial Configuration and Reproducible Forward Order](en/docs/chapters/36/section-08.md)
  * [8. From Computational Graph to Chip](en/docs/chapters/36/section-09.md)
  * [9. Acceptance and Current Delivery Boundaries](en/docs/chapters/36/section-10.md)
  * [10. Lyapunov Revision Record](en/docs/chapters/36/section-11.md)
  * [11. Explicit Integration of Implicit Solving and Continual Evolution (2026-09-12)](en/docs/chapters/36/section-12.md)
  * [12. Capability Priority and Optional Compatibility Extensions (Revised 2026-09-12)](en/docs/chapters/36/section-13.md)
  * [13. Function/Operator Spaces and Numerical Error Interfaces (2026-09-12)](en/docs/chapters/36/section-14.md)
  * [Supplement on Reflection, Review, and Retrospective Analysis (2026-09-12)](en/docs/chapters/36/section-15.md)
  * [14. Unified Mathematical Modeling: Tasks, Models, and Adaptation](en/docs/chapters/36/section-16.md)
  * [15. Detailed Construction of Group A: Turning Observations into Interpretable Inputs](en/docs/chapters/36/section-17.md)
  * [16. Detailed Construction of Group B: Connecting Modality Encoding to Measurable States](en/docs/chapters/36/section-18.md)
  * [17. Detailed Construction of Group C: Graphs, Geometry, and Representation Hierarchies](en/docs/chapters/36/section-19.md)
  * [18. Detailed Construction of Group D: Trainable Kernels of Spatiotemporal Blocks](en/docs/chapters/36/section-20.md)
  * [19. Detailed Construction of Group E: Prediction, Generation, and Task Readouts](en/docs/chapters/36/section-21.md)
  * [20. Detailed Construction of Group F: Error-Driven Bounded Adaptation](en/docs/chapters/36/section-22.md)
  * [21. Detailed Construction of Group G: Training, Retrospective Analysis, and Continual Learning](en/docs/chapters/36/section-23.md)
  * [22. Detailed Construction of Group H: Tools, Execution, and Evidence](en/docs/chapters/36/section-24.md)
  * [23. Detailed Construction of Group L: Stability, Dissipation, and Correction](en/docs/chapters/36/section-25.md)
  * [24. Mathematical Responsibilities and Integration Points of Internal Extensions](en/docs/chapters/36/section-26.md)
  * [25. How to Construct and Execute the Complete Model](en/docs/chapters/36/section-27.md)
  * [26. Properties That Can Be Established, Conditions, and Counterexamples](en/docs/chapters/36/section-28.md)
  * [27. Complexity, Resources, and Experimental Construction](en/docs/chapters/36/section-29.md)
* [13 · SGD-Net Model Architecture and Layer Operators Explained](en/docs/13.md)
  * [1. Core Conclusion](en/docs/chapters/13/section-01.md)
  * [2. Architectural Boundaries of SGD-Net](en/docs/chapters/13/section-02.md)
  * [3. Overall Layered Architecture](en/docs/chapters/13/section-03.md)
  * [4. Unified Notation and Tensor Definitions](en/docs/chapters/13/section-04.md)
  * [5. Layer 0: Input Encoding Layer](en/docs/chapters/13/section-05.md)
  * [6. Layer 1: Graph Construction Layer](en/docs/chapters/13/section-06.md)
  * [7. Layer 2: Node and Edge Embedding Layer](en/docs/chapters/13/section-07.md)
  * [8. Layer 3: SSM State Evolution Layer](en/docs/chapters/13/section-08.md)
  * [9. Layer 4: GNN Topology Propagation Layer](en/docs/chapters/13/section-09.md)
  * [9b. Layer 4b: Equivariant Geometry Propagation Layer](en/docs/chapters/13/section-10.md)
  * [9c. Layer 4c: Multiple-Metric/Manifold Propagation Layer (Long-Term Option)](en/docs/chapters/13/section-11.md)
  * [10. Layer 5: Dynamic Tree Routing Layer](en/docs/chapters/13/section-12.md)
  * [11. Layer 6: Posterior Error Estimation Layer](en/docs/chapters/13/section-13.md)
  * [12. Layer 7: Adaptive Topology Reconstruction Layer](en/docs/chapters/13/section-14.md)
  * [13. Layer 8: Stability Projection Layer](en/docs/chapters/13/section-15.md)
  * [14. Layer 9: Task Output Head](en/docs/chapters/13/section-16.md)
  * [15. SGD-Net Block: Stackable Basic Unit](en/docs/chapters/13/section-17.md)
  * [16. Complete Forward Procedure](en/docs/chapters/13/section-18.md)
  * [17. Total Loss Function Design](en/docs/chapters/13/section-19.md)
  * [18. Relationship Between Operators and Layers](en/docs/chapters/13/section-20.md)
  * [19. Minimum Implementable Version: MVP](en/docs/chapters/13/section-21.md)
  * [20. Standard SGD-Net](en/docs/chapters/13/section-22.md)
  * [21. Full SGD-Net](en/docs/chapters/13/section-23.md)
  * [22. Relationship to Other Architectures](en/docs/chapters/13/section-24.md)
  * [23. Suggested Implementation Directory](en/docs/chapters/13/section-25.md)
  * [24. Draft Core Class Interfaces](en/docs/chapters/13/section-26.md)
  * [25. Complexity Analysis](en/docs/chapters/13/section-27.md)
  * [26. Design Invariants](en/docs/chapters/13/section-28.md)
  * [27. Recommended Development Order](en/docs/chapters/13/section-29.md)
  * [29. Summary](en/docs/chapters/13/section-30.md)
* [24 · Exploring New Ideas: Equivariant GNNs, SSMs, and Self-Evolution Mechanisms](en/docs/24.md)
* [18 · SGD-Net and JEPA World-Model Integration with the JSBO Bridge Operator: White Paper](en/docs/18.md)
* [33 · SGD-Net-TC: A Topology- and Contact-Conditioned World-Model Architecture Proposal](en/docs/33.md)

## English · Mathematics, learning, and continual evolution

* [44 · SGD-Net Lyapunov Stability Components and Domain Applicability Conditions](en/docs/44.md)
* [45 · SGD-Net Implicit-Solve Components and Training/Inference Paths](en/docs/45.md)
* [46 · SGD-Net Prior–Posterior-Driven Dynamic Growth and Continuous Evolution](en/docs/46.md)
* [48 · SGD-Net Assessment of Function Spaces, Operator Approximation, and Constrained Solution Spaces](en/docs/48.md)
* [27 · SGD-Net Hybrid-Solver Training System and World-Model Research Roadmap](en/docs/27.md)
* [14 · SGD-Net Dual-Loop Control and Brain-Inspired Cognitive Architecture](en/docs/14.md)
* [17 · SGD-Net Retrospective Analysis and Dynamic Self-Evolution Mechanisms: White Paper](en/docs/17.md)
* [50 · SGD-Net: Foundations and Closed-Loop Improvements for Reflection, Review, and Retrospective Analysis](en/docs/50.md)
* [Memory–Reasoning Consolidation and Parallel Generation Design](en/guide/memory-reasoning-parallel-generation.md)
* [25 · SGD-Harness Physical Conservation Sentinel and Equivariant Self-Evolution Loop](en/docs/25.md)

## English · Domain models and scientific applications

* [35 · SGD-Net Cross-Domain Extension Roadmap and Scientific Task Mapping](en/docs/35.md)
* [37 · SGD-WM: A Detailed Guide to World Model Construction](en/docs/37.md)
* [38 · SGD-TC: A Detailed Guide to Physical AI and Dexterous Hand Model Construction](en/docs/38.md)
* [39 · SGD-Mol: A Detailed Guide to Pharmaceutical and Molecular Model Construction](en/docs/39.md)
* [40 · SGD-Cell: A Detailed Guide to Biological and Cellular Model Construction](en/docs/40.md)
* [41 · SGD-Mat: A Detailed Guide to Materials Model Construction](en/docs/41.md)
* [42 · SGD-Net: A Detailed Guide to Neurobehavioral Model Construction](en/docs/42.md)
* [43 · SGD-Net: A Detailed Guide to Signaling–Synapse Model Construction](en/docs/43.md)
* [07 · SGD-Net for Scientific Large Models: AlphaFold 3, RFdiffusion, and ESMFold](en/docs/07.md)
* [08 · SGD-Net Multimodal Research Agent and Automated Materials Screening Proposal](en/docs/08.md)
* [15 · SGD-Ecosystem: Physical Embodiment, Differential Digital Twins, and Bidirectional Manifold Folding](en/docs/15.md)

## English · Implementation and experimental validation

* [03 · SGD-Net Python Implementation Technical White Paper](en/docs/03.md)
* [04 · SGD-Net Python Implementation High-Level Design](en/docs/04.md)
* [05 · SGD-Net Python Detailed Implementation Design](en/docs/05.md)
* [34 · SGD-Net-TC Validation Plan and Software/Hardware Mapping](en/docs/34.md)
* [49 · SGD-Net Capability-First Roadmap and Mathematical Evaluation Task Cards](en/docs/49.md)

## English · Appendices

* [Model component index](en/architecture/README.md)
* [Equation and diagram rendering checks](en/guide/rendering-check.md)
* [Public documentation notice](en/NOTICE.md)
* [GitBook setup and maintenance](en/GITBOOK_SETUP.md)

* [Online reading and publishing maintenance](en/guide/publishing.md)

* [Bilingual documentation structure and maintenance](en/guide/bilingual.md)

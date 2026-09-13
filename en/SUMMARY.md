# Summary

* [Home](README.md)

## Getting started

* [Start here](guide/start-here.md)
* [Terminology and reading conventions](guide/glossary.md)
* [Research status and contribution boundaries](guide/status.md)

## Research framing

* [01 · SGD-Net: An Error-Driven Adaptive Neural Architecture Combining State-Space Models, Topological/Equivariant Graph Neural Networks, and Adaptive Decision Trees](docs/01.md)
* [32 · SGD-Net Research Integration Assessment and Innovation Boundaries](docs/32.md)
* [47 · SGD-Net Generality Review and Constructible Model Coverage Proposal](docs/47.md)
* [SGD-Net and Transformer: Operator Coverage, Architectural Differences, and Proposed Additions](guide/transformer-comparison.md)
* [Graph Memory, Model Equivalence, and Mathematics-Driven Software and Hardware Planning](guide/mathematical-computing-foundation.md)

## Core model architecture

* [36 · SGD-Net Unified Foundation Model Architecture and Complete Component Set](docs/36.md)
  * [Abstract and Guide to the Detailed Edition](docs/chapters/36/section-01.md)
  * [1. Architectural Decisions and Reading Entry Points](docs/chapters/36/section-02.md)
  * [2. Correspondence with Existing Materials](docs/chapters/36/section-03.md)
  * [3. System Connection Diagram](docs/chapters/36/section-04.md)
  * [4. Complete Component and Unit Set: 9 Groups, 72 Registered Units](docs/chapters/36/section-05.md)
  * [5. Tensor, State, and Clock Contracts](docs/chapters/36/section-06.md)
  * [6. How a Block Actually Computes](docs/chapters/36/section-07.md)
  * [7. Initial Configuration and Reproducible Forward Order](docs/chapters/36/section-08.md)
  * [8. From Computational Graph to Chip](docs/chapters/36/section-09.md)
  * [9. Acceptance and Current Delivery Boundaries](docs/chapters/36/section-10.md)
  * [10. Lyapunov Revision Record](docs/chapters/36/section-11.md)
  * [11. Explicit Integration of Implicit Solving and Continual Evolution (2026-09-12)](docs/chapters/36/section-12.md)
  * [12. Capability Priority and Optional Compatibility Extensions (Revised 2026-09-12)](docs/chapters/36/section-13.md)
  * [13. Function/Operator Spaces and Numerical Error Interfaces (2026-09-12)](docs/chapters/36/section-14.md)
  * [Supplement on Reflection, Review, and Retrospective Analysis (2026-09-12)](docs/chapters/36/section-15.md)
  * [14. Unified Mathematical Modeling: Tasks, Models, and Adaptation](docs/chapters/36/section-16.md)
  * [15. Detailed Construction of Group A: Turning Observations into Interpretable Inputs](docs/chapters/36/section-17.md)
  * [16. Detailed Construction of Group B: Connecting Modality Encoding to Measurable States](docs/chapters/36/section-18.md)
  * [17. Detailed Construction of Group C: Graphs, Geometry, and Representation Hierarchies](docs/chapters/36/section-19.md)
  * [18. Detailed Construction of Group D: Trainable Kernels of Spatiotemporal Blocks](docs/chapters/36/section-20.md)
  * [19. Detailed Construction of Group E: Prediction, Generation, and Task Readouts](docs/chapters/36/section-21.md)
  * [20. Detailed Construction of Group F: Error-Driven Bounded Adaptation](docs/chapters/36/section-22.md)
  * [21. Detailed Construction of Group G: Training, Retrospective Analysis, and Continual Learning](docs/chapters/36/section-23.md)
  * [22. Detailed Construction of Group H: Tools, Execution, and Evidence](docs/chapters/36/section-24.md)
  * [23. Detailed Construction of Group L: Stability, Dissipation, and Correction](docs/chapters/36/section-25.md)
  * [24. Mathematical Responsibilities and Integration Points of Internal Extensions](docs/chapters/36/section-26.md)
  * [25. How to Construct and Execute the Complete Model](docs/chapters/36/section-27.md)
  * [26. Properties That Can Be Established, Conditions, and Counterexamples](docs/chapters/36/section-28.md)
  * [27. Complexity, Resources, and Experimental Construction](docs/chapters/36/section-29.md)
* [13 · SGD-Net Model Architecture and Layer Operators Explained](docs/13.md)
  * [1. Core Conclusion](docs/chapters/13/section-01.md)
  * [2. Architectural Boundaries of SGD-Net](docs/chapters/13/section-02.md)
  * [3. Overall Layered Architecture](docs/chapters/13/section-03.md)
  * [4. Unified Notation and Tensor Definitions](docs/chapters/13/section-04.md)
  * [5. Layer 0: Input Encoding Layer](docs/chapters/13/section-05.md)
  * [6. Layer 1: Graph Construction Layer](docs/chapters/13/section-06.md)
  * [7. Layer 2: Node and Edge Embedding Layer](docs/chapters/13/section-07.md)
  * [8. Layer 3: SSM State Evolution Layer](docs/chapters/13/section-08.md)
  * [9. Layer 4: GNN Topology Propagation Layer](docs/chapters/13/section-09.md)
  * [9b. Layer 4b: Equivariant Geometry Propagation Layer](docs/chapters/13/section-10.md)
  * [9c. Layer 4c: Multiple-Metric/Manifold Propagation Layer (Long-Term Option)](docs/chapters/13/section-11.md)
  * [10. Layer 5: Dynamic Tree Routing Layer](docs/chapters/13/section-12.md)
  * [11. Layer 6: Posterior Error Estimation Layer](docs/chapters/13/section-13.md)
  * [12. Layer 7: Adaptive Topology Reconstruction Layer](docs/chapters/13/section-14.md)
  * [13. Layer 8: Stability Projection Layer](docs/chapters/13/section-15.md)
  * [14. Layer 9: Task Output Head](docs/chapters/13/section-16.md)
  * [15. SGD-Net Block: Stackable Basic Unit](docs/chapters/13/section-17.md)
  * [16. Complete Forward Procedure](docs/chapters/13/section-18.md)
  * [17. Total Loss Function Design](docs/chapters/13/section-19.md)
  * [18. Relationship Between Operators and Layers](docs/chapters/13/section-20.md)
  * [19. Minimum Implementable Version: MVP](docs/chapters/13/section-21.md)
  * [20. Standard SGD-Net](docs/chapters/13/section-22.md)
  * [21. Full SGD-Net](docs/chapters/13/section-23.md)
  * [22. Relationship to Other Architectures](docs/chapters/13/section-24.md)
  * [23. Suggested Implementation Directory](docs/chapters/13/section-25.md)
  * [24. Draft Core Class Interfaces](docs/chapters/13/section-26.md)
  * [25. Complexity Analysis](docs/chapters/13/section-27.md)
  * [26. Design Invariants](docs/chapters/13/section-28.md)
  * [27. Recommended Development Order](docs/chapters/13/section-29.md)
  * [29. Summary](docs/chapters/13/section-30.md)
* [24 · Exploring New Ideas: Equivariant GNNs, SSMs, and Self-Evolution Mechanisms](docs/24.md)
* [18 · SGD-Net and JEPA World-Model Integration with the JSBO Bridge Operator: White Paper](docs/18.md)
* [33 · SGD-Net-TC: A Topology- and Contact-Conditioned World-Model Architecture Proposal](docs/33.md)

## Mathematics, learning, and continual evolution

* [44 · SGD-Net Lyapunov Stability Components and Domain Applicability Conditions](docs/44.md)
* [45 · SGD-Net Implicit-Solve Components and Training/Inference Paths](docs/45.md)
* [46 · SGD-Net Prior–Posterior-Driven Dynamic Growth and Continuous Evolution](docs/46.md)
* [48 · SGD-Net Assessment of Function Spaces, Operator Approximation, and Constrained Solution Spaces](docs/48.md)
* [27 · SGD-Net Hybrid-Solver Training System and World-Model Research Roadmap](docs/27.md)
* [14 · SGD-Net Dual-Loop Control and Brain-Inspired Cognitive Architecture](docs/14.md)
* [17 · SGD-Net Retrospective Analysis and Dynamic Self-Evolution Mechanisms: White Paper](docs/17.md)
* [50 · SGD-Net: Foundations and Closed-Loop Improvements for Reflection, Review, and Retrospective Analysis](docs/50.md)
* [25 · SGD-Harness Physical Conservation Sentinel and Equivariant Self-Evolution Loop](docs/25.md)

## Domain models and scientific applications

* [35 · SGD-Net Cross-Domain Extension Roadmap and Scientific Task Mapping](docs/35.md)
* [37 · SGD-WM: A Detailed Guide to World Model Construction](docs/37.md)
* [38 · SGD-TC: A Detailed Guide to Physical AI and Dexterous Hand Model Construction](docs/38.md)
* [39 · SGD-Mol: A Detailed Guide to Pharmaceutical and Molecular Model Construction](docs/39.md)
* [40 · SGD-Cell: A Detailed Guide to Biological and Cellular Model Construction](docs/40.md)
* [41 · SGD-Mat: A Detailed Guide to Materials Model Construction](docs/41.md)
* [42 · SGD-Net: A Detailed Guide to Neurobehavioral Model Construction](docs/42.md)
* [43 · SGD-Net: A Detailed Guide to Signaling–Synapse Model Construction](docs/43.md)
* [07 · SGD-Net for Scientific Large Models: AlphaFold 3, RFdiffusion, and ESMFold](docs/07.md)
* [08 · SGD-Net Multimodal Research Agent and Automated Materials Screening Proposal](docs/08.md)
* [15 · SGD-Ecosystem: Physical Embodiment, Differential Digital Twins, and Bidirectional Manifold Folding](docs/15.md)

## Implementation and experimental validation

* [03 · SGD-Net Python Implementation Technical White Paper](docs/03.md)
* [04 · SGD-Net Python Implementation High-Level Design](docs/04.md)
* [05 · SGD-Net Python Detailed Implementation Design](docs/05.md)
* [34 · SGD-Net-TC Validation Plan and Software/Hardware Mapping](docs/34.md)
* [49 · SGD-Net Capability-First Roadmap and Mathematical Evaluation Task Cards](docs/49.md)

## Appendices

* [Model component index](architecture/README.md)
* [Equation and diagram rendering checks](guide/rendering-check.md)
* [Public documentation notice](NOTICE.md)
* [GitBook setup and maintenance](GITBOOK_SETUP.md)

* [Online reading and publishing maintenance](guide/publishing.md)

* [Bilingual documentation structure and maintenance](guide/bilingual.md)

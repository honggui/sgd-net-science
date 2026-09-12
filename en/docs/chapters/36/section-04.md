# 3. System Connection Diagram

```mermaid
flowchart TD
    A[Observations Contracts Units Identity Time] --> B[Domain Encoders and JSBO]
    B --> C[Typed Entity and Interaction Graphs]
    C --> D[SGD Spatiotemporal Block Repeated L Times]
    D --> E[State Observation Action Domain-Property Heads]
    D --> L[Candidate V and Applicable Stability Checks]
    E --> L
    L --> F[Error Calibration and Finite Budgets]
    E --> F
    F --> R[Graph Refinement and P/Q State Migration]
    R --> D
    F --> Q[Acquire Observations or Call External Tools]
    Q --> A
    E --> H[Domain Checks Version Checks Commit]
    H --> O[Prediction Report or Device Execution]
    O --> J[Raw Results and Offline Replay]
    J --> G[Training Adaptation Independent Evaluation]
    G --> W[Frozen New Weight Version]
    W --> D
```

The feedback edges have different temporal meanings: R does not advance real time; Q may wait for new physical measurements; G generates candidate weights that take effect after agreed validation and version release, potentially including restricted online adaptation and offline consolidation. D/E are neural forward computations; L contains a trainable V and conditional checks/corrections, while H belongs to the external runtime.


---

[← Previous](section-03.md) · [Contents](../../../SUMMARY.md) · [Next →](section-05.md)

# 22. Relationship to Other Architectures

| Architecture | Relationship to SGD-Net |
|---|---|
| Transformer | Can serve as an input encoder or task head; not a core requirement |
| Mamba/SSM | One of SGD-Net's state-evolution cores |
| GNN | One of SGD-Net's topology-propagation cores |
| Decision Tree | One of SGD-Net's local-adaptation cores |
| MoE | Dynamic trees can extend it to hierarchical expert routing |
| PINN | Can serve as a form of PDE posterior error or task loss |
| Neural Operator | Can serve as a scientific-computing baseline or local operator |
| AlphaFold/RFdiffusion | Can serve as an external foundation model enhanced or diagnosed by SGD-Net |
| JEPA / I-JEPA / V-JEPA | Can serve as external world-model encoders/predictors, converted through JSBO bridges to structured GraphState or PhysicalState |
| G-JEPA / G-GeoS | Can serve as long-term research hypotheses combining JEPA latent prediction with equivariance, multiple metrics, reachability, and physical residuals; currently included only in traces/benchmarks, not as verified AGI conclusions |


---

[← Previous](section-23.md) · [Contents](../../../SUMMARY.md) · [Next →](section-25.md)

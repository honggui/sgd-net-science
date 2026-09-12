# 1. Architectural Decisions and Reading Entry Points

The shared architecture retains the name **SGD-Science-Core**. A model instance is defined as a collection of encoders + a domain graph definition + spatiotemporal blocks + domain heads + optional adaptive inference. SGD-Net-TC is its physical-interaction configuration, EGNN a geometric message-passing module, JEPA an optional representation-learning path, and GLS/Aegis-X external execution and deployment systems.

Of the three options, a single set of weights shared across all domains faces conflicts in scale, symmetry, and objectives, while seven entirely independent systems duplicate development. The chosen approach uses a shared component library, explicit domain configurations, and experimentally selected parameter sharing. Static materials or cross-sectional cell tasks may bypass the SSM; gene-association graphs without geometry may bypass EGNN. Component availability does not mean every model must enable all components.

| Domain construction document | Model instance and focus |
|---|---|
| [37 World Models](../../37.md) | SGD-WM: Conditional future distributions and partially observable states |
| [38 Physical AI](../../38.md) | SGD-TC: Embodiment, contact, actions, and closed loops |
| [39 AI Drug Discovery](../../39.md) | SGD-Mol: Structure enhancement and ranking of property and functional candidates |
| [40 AI Biology](../../40.md) | SGD-Cell: Conditional population responses and perturbation prediction |
| [41 AI Materials](../../41.md) | SGD-Mat: Periodic structures, potential energy, and multifidelity screening |
| [42 Neural and Behavioral Research](../../42.md) | SGD-Neuro: Neural–behavioral relationships and conditional information transfer |
| [43 Signaling and Synapse Research](../../43.md) | SGD-Synapse: Processing, signaling, and functional phenotypes |

The names identify configurations and do not claim external priority. Each domain document specifies data, specialized units, forward paths, losses, training order, and exit conditions.


---

[← Previous](section-01.md) · [Contents](../../../SUMMARY.md) · [Next →](section-03.md)

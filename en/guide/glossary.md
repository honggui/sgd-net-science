# Terminology and reading conventions

| Term | Meaning in this book |
|---|---|
| SGD-Net | State-space Graph-Dynamic Tree Network; here, SGD is not an abbreviation for the stochastic gradient descent optimizer |
| SSM | State-space model; internal state is distinct from solving an actual implicit equation |
| GNN / EGNN | Graph message passing and equivariant graph models under a specified group action |
| JEPA | Joint Embedding Predictive Architecture; predicting latent representations does not directly amount to predicting physical variables |
| JSBO | A candidate operator discussed in this project for bridging JEPA to a structured model; it still requires validation |
| Implicit layer | A layer whose output is defined by an equation or fixed point, requiring a solver and valid gradient conditions |
| Lyapunov | A mathematical tool for analyzing stability with respect to specified states, errors, or reference trajectories |
| Observation innovation | The discrepancy between an earlier prediction and a subsequently paired observation; it is not an arbitrary change between adjacent states |
| Representation refinement | A change to the computational representation, not the creation of real atoms, links, or biological entities |
| Candidate version | Experimental parameters, states, or structures kept separate from the active version |
| Domain unit | A task-specific configuration or refinement of a base component; it must not be counted again as a top-level component |

The top-level registry contains 72 units; IM, EV, U, and FN contain 27 internal interfaces; and the seven domains contain 56 domain units. These are design counts, not counts of implemented neural layers or tools.

---

[← Previous](start-here.md) · [Full contents](../SUMMARY.md) · [Next →](status.md)

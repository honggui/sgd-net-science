# 11. Explicit Integration of Implicit Solving and Continual Evolution (2026-09-12)

Earlier versions retained only refinement, routing, and replay interfaces without fully detailing solving and persistent-growth procedures. These are now supplemented by [45-SGD-Net Implicit Solve Components and Training/Inference Paths 20260912](../../45.md) and [46-SGD-Net Prior/Posterior-Driven Dynamic Growth and Continual Evolution 20260912](../../46.md).

Implicit path: current D/B representations → IM01–IM04 within F07 assemble and solve equations on a fixed graph → L/H checks → E outputs. G01/G03 handle implicit or truncated gradients through IM05, while IM06 handles failure paths. DEQ equilibria and implicit physical stepping are configured separately; SSM recurrence or distillation must not be called implicit root finding.

Evolution path: F01/F08 store pre-observation predictions → EV01/EV02 compare real results and persistent errors → F03/F04 propose splitting, growth, or no change → F05/F06/G06 migrate parameters and states → G01/G03/G05 train locally → C/L/G07 validate → G06/H08 accept persistent versions → subsequent pruning and distillation. Dimension expansion must handle all affected parameters, normalization, optimizer slots, and compiled shapes.

The full version should include this procedure definition; a minimal fixed-graph configuration serves as a baseline and does not replace the full version. Event-state updates, temporary within-task refinement, restricted online parameter/structure acceptance, and offline intergenerational consolidation are recorded separately. By default, parameter shapes and graph versions do not change within a critical control operation, an implicit solve, or a backward pass.

IM01–IM06 and EV01–EV08 refine the existing 72 base units internally without increasing the top-level count. Including a function in the architecture does not establish an implemented model, training, or evolutionary gains.



---

[← Previous](section-11.md) · [Contents](../../../SUMMARY.md) · [Next →](section-13.md)

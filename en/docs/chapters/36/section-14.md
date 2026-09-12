# 13. Function/Operator Spaces and Numerical Error Interfaces (2026-09-12)

[48 Mathematical Evaluation](../../48.md) defines task spaces, constraint-feasible and solution sets, discrete observations and reconstruction, approximation and learning errors, solution conditions, and architecture comparisons. [49 Task Cards](../../49.md) specifies implementation order and five planned experiment groups.

New internal interfaces comprise FN01 task-space contracts, FN02 observation discretization and reconstruction, FN03 operator-query readouts, FN04 residual quality and stability factors, and FN05 error–cost adaptation. These attach to existing A/B/C/E/F/G/L and IM/EV units; the 72 top-level IDs remain unchanged.

F01 should distinguish conditional error bounds, empirical proxies, and unknown errors; small residuals cannot directly be treated as accurate solutions. F04 proposes splitting/expert growth only when evidence supports a representation bottleneck. G08 may request new observations when information is insufficient, while F07 adjusts tolerances and iterations for solver error. P/Q migration also requires stability checks in the task norm.

For field tasks, E08 and other readouts should declare query locations and reconstruction rules. When discretization points change, check quadrature weights and cross-resolution consistency. Graph messages may be designed as integral-operator approximations, but ordinary mean aggregation does not automatically have that meaning. Vary model capacity, data volume, and numerical iterations separately to avoid conflating their benefits.



---

[← Previous](section-13.md) · [Contents](../../../SUMMARY.md) · [Next →](section-15.md)

# 5. Tensor, State, and Clock Contracts

For a single-domain batch, B is batch size, N/E are the valid nodes/edges in the batch, d the scalar hidden dimension, s the SSM state dimension, q the number of vector channels, and H the prediction horizon. Batches may use padded or packed representations, but every index must carry sample membership.

| Object | Shape/structure | Semantics |
|---|---|---|
| Node scalars h | B×N×d | Rotation-invariant attribute channels; no direct absolute xyz in geometric mode |
| Vectors v, coordinates x | B×N×q×n, B×N×n | n is the chosen Euclidean coordinate dimension; n=3 for 3D tasks; measured x is separate from computational coordinates ξ |
| Edges | edge_index[2,E], edge_attr[E,de] | Type, sample, periodic image, and validity |
| Memory z | B×N×s per block | Maintained by stable entity and physical time; not weights |
| Conditions c | B×dc or B×N×dc | Embodiment, environment, task conditions; identity keys are not used directly as generalization features |
| Predictions | B×H×N×dy or domain-specific shape | Mean/scale/category parameters and masks; separate heads |
| State bundle | h,z,x,mask,IDs,versions | observation/candidate; physical time, rollout_step, refine_iter, generation_step |

Clear or freeze memory when an entity exits; reuse after reidentification must depend on matching confidence. Residual states must not be shared across batches, animals, or samples. Reordering graph nodes should only reorder outputs without changing prediction content.

Static configurations disable D04/D05. D02/D03/D06 may repeat over the structure, but the repetition count is spatial message depth. Cross-sectional cells do not maintain states by data-row number; molecular sequences are encoded by B02, not used to simulate physical evolution by atom index.

## 5.1 High-Dimensional Equivariance and Generalized Geometry: Explicit Retention of the Original Roadmap <a href="#section-001" id="section-001"></a>

Verification supplement, 2026-09-12: the original document 26 already proposed the G-GeoS roadmap combining high-dimensional equivariance with non-Euclidean/multiple-metric/quasimetric geometry. The preceding 3D application examples do not abandon that scope. Four quantities must be distinguished: coordinate dimension n, scalar-feature dimension d, vector-channel count q, and constraint count k. Increasing d, q, or k does not automatically produce E(n) equivariance.

The ordinary Euclidean branch allows coordinates in R^n. Assuming, among other conditions, invariant scalar features, apply the common transformation x→Qx+b with Q∈O(n), b∈R^n: coordinate outputs must transform accordingly, vector outputs are multiplied only by Q, and scalar outputs remain unchanged. The distance and relative-vector formulas in Section 6.1 apply to this n-dimensional Euclidean branch. n may be 2, 3, or higher; it is not fundamentally restricted to n≥3. The original EGNN paper explicitly supports high-dimensional spaces, so extending 3D to nD is not itself claimed as an original contribution of this project.

Non-Euclidean, Lorentz, mixed-curvature, and quasimetric geometries are not obtained merely by changing n. For these branches, D08 requires separate declarations of spaces, metrics/pairings, transformation groups, state representations, mappings, and validation rules. One cannot mechanically replace squared Euclidean distance and retain the E(n) proof. Quasimetrics and asymmetric relations may not share the same isometry group.

Recommended configuration fields are geometry_mode, coordinate_dim, feature_dim, vector_channels, group_action, metric_type, signature, curvature, and constraint_spec. High-dimensional Euclidean branches mainly reside in C05/C07 and D02/D07; more general geometry resides in D08. Molecules, robots, and ordinary crystals are still modeled in real three-dimensional space. High-dimensional attributes retain their units and transformation semantics: concatenating temperature, expression level, time, and xyz does not allow them to rotate arbitrarily into one another.

Implementing high-dimensional branches requires permutation, orthogonal-transformation, and translation consistency checks for n=3 and n>3; non-Euclidean branches use their declared transformation groups and constraint tests. Recalculate coordinate, vector, geometric-reduction, and state-migration costs after changing n or q. Mathematical applicability does not establish existing training benefits or hardware support.


---

[← Previous](section-05.md) · [Contents](../../../SUMMARY.md) · [Next →](section-07.md)

# 9b. Layer 4b: Equivariant Geometry Propagation Layer

**On This Page**

- [9b.1 Function](#section-001)
- [9b.2 Basic EGNN Message Operator](#section-002)
- [9b.3 CrystalGraphBuilder](#section-003)
- [9b.4 Implementation Interface](#section-004)
- [9b.5 Applicability Boundaries](#section-005)

---

## 9b.1 Function <a href="#section-001" id="section-001"></a>

`EquivariantGNNLayer` serves tasks with explicit 3D geometry, including molecules, crystals, proteins, PDE particle fields, and robotics. It addresses the question:

> When the input undergoes a global rotation, translation, or reflection, does the model output follow the same geometric transformation while preserving physical laws?

Ordinary TopoGNN handles connectivity alone; EGNN also handles:

- invariant scalars: atom types, charges, masses, squared distances, and energy densities;
- equivariant vectors: coordinates, velocities, forces, and displacements;
- structural metadata: unit cells, periodic boundaries, space-group operations, and defect masks.

## 9b.2 Basic EGNN Message Operator <a href="#section-002" id="section-002"></a>

For edge $$(i,j)$$, define the relative-distance invariant:

$$
r_{ij}=\|\mathbf{x}_i-\mathbf{x}_j\|^2
$$

Message:

$$
m_{ij}=\phi_e(h_i,h_j,r_{ij},e_{ij})
$$

Equivariant coordinate update:

$$
\Delta \mathbf{x}_i=\sum_{j\in\mathcal{N}(i)}(\mathbf{x}_i-\mathbf{x}_j)\phi_x(m_{ij})
$$

$$
\mathbf{x}_i'=\mathbf{x}_i+\Delta \mathbf{x}_i
$$

Scalar update:

$$
h_i'=\phi_h(h_i,\sum_j m_{ij})
$$

If the input coordinates are globally transformed to $$R\mathbf{x}+t$$, the output satisfies:

$$
\mathbf{x}'(R\mathbf{x}+t)=R\mathbf{x}'(\mathbf{x})+t
$$

## 9b.3 CrystalGraphBuilder <a href="#section-003" id="section-003"></a>

Materials and crystal tasks should preferentially construct a geometric graph state:

```text
CrystalGraphBuilder.build(
  cif_or_structure,
  radius_cutoff,
  pbc=True,
  space_group=True,
  defect_policy=None
) -> GeometryGraphState
```

```text
GeometryGraphState
  node_scalar
  node_coord
  edge_index
  edge_attr
  lattice_vectors
  pbc_offsets
  space_group_ops
  defect_masks
  source_tags
```

## 9b.4 Implementation Interface <a href="#section-004" id="section-004"></a>

```text
EquivariantGNNLayer.forward(
  node_scalar,
  node_coord,
  edge_index,
  edge_attr=None,
  lattice_vectors=None,
  pbc_offsets=None,
  vector_field=None
) -> node_scalar_new, node_coord_new, vector_field_new, geometry_trace
```

## 9b.5 Applicability Boundaries <a href="#section-005" id="section-005"></a>

| EGNN is appropriate | EGNN is unnecessary |
|---|---|
| Molecules/proteins/ligands | Pure knowledge graphs |
| Crystals/materials/semiconductors | Textual fact graphs |
| PDE particle fields/mesh points | MoE token-expert routing |
| Robots/rigid bodies/point clouds | Workflow DAGs without explicit geometry |


---

[← Previous](section-09.md) · [Contents](../../../SUMMARY.md) · [Next →](section-11.md)

# 17. Detailed Construction of Group C: Graphs, Geometry, and Representation Hierarchies

**On This Page**

- [C01 Graph Schema Management](#section-001)
- [C02 Fixed Structural Graph](#section-002)
- [C03 Dynamic Interaction Graph](#section-003)
- [C04 Inference Hierarchy Graph](#section-004)
- [C05 Geometric Edge Features](#section-005)
- [C06 Periodic Boundary Unit](#section-006)
- [C07 Symmetry/Chirality Policy](#section-007)
- [C08 Graph Validity and Sparse Layout](#section-008)

---

## C01 Graph Schema Management <a href="#section-001" id="section-001"></a>

**Role and mathematics.** Define a typed graph

$$
\mathcal G=(V,E,\tau_V,\tau_E,a_V,a_E),\qquad
\tau_E(i,j)\in\mathcal R_{\tau_V(i),\tau_V(j)}.
$$

**Construction and use.** The schema specifies admissible node pairs, directions, edge attributes, and evidence sources. Separate measured edges, structural-prior edges, and predicted association edges; D02 may select parameters by relation type. The graph schema itself needs no gradients. Learned predicted edges cannot automatically be promoted to real chemical bonds or causal connections. Cross-sample connections are prohibited by default.

## C02 Fixed Structural Graph <a href="#section-002" id="section-002"></a>

**Role and mathematics.** External embodiment/structural definitions provide reference adjacency:

$$
A^{fix}_{ij}=\mathbf1\{(i,j)\in E_{reference}\}.
$$

**Construction and use.** Generate edges from joint chains, molecular bonds, or reference meshes, retaining attributes such as bond order/joint axes. Learn messages on that structure rather than learning to delete the underlying structure by default. Tasks with genuinely changing topology should declare reaction/fracture mechanisms and generate new versions. Fixed topology reduces graph-construction cost but cannot describe all nonlocal interactions; merge with C03 and deduplicate where necessary.

## C03 Dynamic Interaction Graph <a href="#section-003" id="section-003"></a>

**Role and mathematics.** For example, use radius neighborhoods and learned contact probabilities:

$$
E_t^{near}=\{(i,j):\|x_i-x_j\|<r_c\},\qquad
p_{ij}^{contact}=\sigma(\phi_{edge}(h_i,h_j,a_{ij})).
$$

**Construction and use.** Search generates candidate edges, while a classifier estimates contact/interaction; geometric support and confidence attributes remain separate. Discrete changes in edge search usually do not backpropagate gradients. Force-field tasks need smooth envelopes and neighbor buffers to handle cutoffs. Configure maximum degree, reconstruction intervals, complexity, and missing-edge errors. Edge probabilities cannot directly establish that physical contact has occurred.

## C04 Inference Hierarchy Graph <a href="#section-004" id="section-004"></a>

**Role and mathematics.** Establish mappings between coarse nodes I and fine nodes i:

$$
P\in\mathbb R^{N_f\times N_c},\quad Q\in\mathbb R^{N_c\times N_f},\qquad
\mathcal H=(\mathcal G_c,\mathcal G_f,parent,P,Q).
$$

**Construction and use.** F04 selects computational-representation refinement, C04 generates valid parent-child indices, and F05/F06 perform actual state mappings. Representation splitting does not create real atoms, fingers, or cells; entity identity and computational-node identity remain separate. Hierarchical graphs may first be fixed and validated, then selected by error signals. Increased node counts are not evidence of improved accuracy.

## C05 Geometric Edge Features <a href="#section-005" id="section-005"></a>

**Role and mathematics.** In E(n) mode, construct

$$
r_{ij}=x_i-x_j,\quad d_{ij}^2=r_{ij}^Tr_{ij},\quad
I_{ij}=(d_{ij}^2,v_{ia}^Tv_{jb},v_{ia}^Tr_{ij}).
$$

**Construction and use.** Relative vectors enter equivariant channels, while distances/inner products enter scalar messages. All vectors must share a coordinate system and transform by the same Q; normalize differing units first or label them separately. Angular denominators require handling degenerate lengths. The same inner-product rules apply for n≥3, but higher coordinate dimensions are neither additional arbitrary constraints nor a substitute for higher-order tensor representations.

## C06 Periodic Boundary Unit <a href="#section-006" id="section-006"></a>

**Role and mathematics.** Let the columns of unit-cell matrix H be lattice basis vectors and s be fractional coordinates:

$$
x_i=Hs_i,\quad r_{ij,k}=H(s_i-s_j+k),\ k\in\mathbb Z^n,
\qquad \|r_{ij,k}\|<r_c.
$$

**Construction and use.** Enumerate periodic images within the cutoff, retaining integer shifts and unit-cell versions. Simple coordinatewise rounding does not guarantee the nearest image in skewed cells. Reverse edges should have corresponding reverse shifts. Include the unit cell in automatic differentiation when computing stress. Discrete neighbor-list changes require smooth cutoffs and validation; fractional-coordinate differences cannot directly serve as Euclidean distances.

## C07 Symmetry/Chirality Policy <a href="#section-007" id="section-007"></a>

**Role and mathematics.** Specify outputs through an explicit group action:

$$
f(g\cdot x)=\rho_{out}(g)f(x).
$$

**Construction and use.** Select permutation, E(n), SE(n), or no geometric constraints. The triple product of ordered relative vectors, χ=r_1·(r_2×r_3), may help represent 3D chirality; χ changes sign under reflection. Neighbor ordering requires consistent rules or compatible aggregation; arbitrary indices must not break permutation symmetry. Scalars, vectors, and pseudoscalars differ. Higher-dimensional chirality needs appropriate representations rather than reusing the 3D cross product.

## C08 Graph Validity and Sparse Layout <a href="#section-008" id="section-008"></a>

**Role and mathematics.** Layouts such as CSR should satisfy

$$
ptr_0=0,\quad ptr_N=E,\quad ptr_i\leq ptr_{i+1},\quad
0\leq src_e,dst_e<N,\quad sample(src_e)=sample(dst_e).
$$

**Construction and use.** Declare rules for duplicate edges, empty neighborhoods, self-loops, and reverse edges; check node/edge mask consistency. Index transformations must synchronize attributes, periodic images, and state mappings. Report graph-construction failures before allowing out-of-bounds gathers. Sparse layouts have no training parameters, but order affects floating-point reductions; deterministic modes and tolerances require separate definitions.


---

[← Previous](section-18.md) · [Contents](../../../SUMMARY.md) · [Next →](section-20.md)

# 6. How a Block Actually Computes

**On This Page**

- [6.1 Scalar Graph Messages and the EGNN Geometric Branch](#section-001)
- [6.2 Entity-Time States and Observation Correction](#section-002)
- [6.3 Where JEPA, Trees, and Uncertainty Actually Belong](#section-003)

---

## 6.1 Scalar Graph Messages and the EGNN Geometric Branch <a href="#section-001" id="section-001"></a>

Geometric configurations use the following EGNN-style reference form; nongeometric graphs omit the distance term:

$$
m_{ij}=\phi_e(h_i,h_j,\|x_i-x_j\|^2,a_{ij},c_i,c_j),\qquad
\bar m_i=\sum_{j\in\mathcal N(i)}M_{ij}m_{ij}.
$$

$$\phi_e$$ is a shared MLP, by default a two-layer Linear–SiLU–Linear network with hidden width d. Edge types may use embeddings or separate small adapters; a complete independent network per edge type is not required. Masks are applied before reduction. Node updates concatenate h, aggregated messages, and temporal memory, project back to d through D06, and then add a residual connection.

The vector output can be written as

$$
\Delta\xi_i=\frac{1}{\max(1,|\mathcal N(i)|)}
\sum_jM_{ij}(x_i-x_j)\phi_x(m_{ij}).
$$

This output can update computational coordinates via $$\xi'=\xi+\Delta\xi$$ or serve as a vector prediction channel; it does not directly overwrite currently measured physical coordinates. The domain head determines geometric update mode, unit scales, and boundary conditions. This reference form draws on EGNN and is not a new equivariance contribution. [Original EGNN paper](https://proceedings.mlr.press/v139/satorras21a.html)

Scalar messages depend only on invariants, and vector outputs are built by multiplying relative vectors by scalars, supporting the corresponding equivariance. Gravity, velocity, and lattice inputs must transform together. One cannot concatenate xyz into an arbitrary MLP or normalize the three axes independently without constraints and still claim equivariance.

These conditions apply to the geometric module itself. Upstream image encoding, coordinate estimation, discrete graph construction, and downstream action conversion must each satisfy the relevant conditions before whole-network equivariance can be discussed; one EGNN layer does not prove equivariance of an entire multimodal system. Edge selection under tied nearest-neighbor distances must also be included in permutation checks.

**Chirality branch**: distance-based E(3) scalars cannot distinguish mirror-image configurations. C07 may select representations with stereochemical labels or directed-geometry/SE(3) modules that aggregate correctly under permutation rules. Original EGNN and chirality-enhanced variants are named separately and tested separately for rotations and reflections. Discrete CIP labels alone do not guarantee that generated coordinates satisfy the specified chirality; structural checks are still required.

## 6.2 Entity-Time States and Observation Correction <a href="#section-002" id="section-002"></a>

The first version uses an explicitly implementable simplified selective diagonal SSM, avoiding mislabeling a custom block as complete Mamba:

$$
u_{i,t}=W_u[h_{i,t}^{in},\bar m_{i,t},c_i],\quad
\delta_{i,t}=\operatorname{softplus}(W_\delta u_{i,t}+b_\delta)\Delta t_i,
$$
$$
a=-\operatorname{softplus}(\alpha),\quad \bar a_{i,t}=\exp(\delta_{i,t}\odot a),\quad
z^-_{i,t}=\bar a_{i,t}\odot z_{i,t-1}+(1-\bar a_{i,t})\odot\phi_b(u_{i,t}).
$$

Here $$a,z,u$$ all have dimension s, and $$W_u$$ projects the concatenated input to s. This state equation is an exponential-smoothing research example constraining only homogeneous diagonal decay; input and message feedback may still cause instability, so it is not a whole-system proof. D04 may be replaced by a standard Mamba module, but state axes, convolution caches, and computation must be registered again. [Original Mamba paper](https://arxiv.org/abs/2312.00752)

When a real observation arrives, compute the innovation $$r=y-\hat y^-$$ using the prior observation prediction for that same time:

$$
z_{i,t}=z^-_{i,t}+M^{obs}_{i,t}\odot K(u_{i,t})\,\operatorname{NormUnit}(r_{i,t}),\quad
h^{out}_{i,t}=h^{in}_{i,t}+W_o\phi_h([h^{in}_{i,t},\bar m_{i,t},z_{i,t}]).
$$

K can be implemented as residual projection plus s-dimensional gating, first mapping valid residuals to s. Without a matched observation, disable this term and retain unknown. NormUnit uses fixed physical scales from the training set rather than mixing different units in an arbitrary norm. Each block keeps its own z and advances only once per event. Refinement loops call spatial/refinement operators without repeatedly decaying an already updated temporal state.

## 6.3 Where JEPA, Trees, and Uncertainty Actually Belong <a href="#section-003" id="section-003"></a>

G04 predicts teacher target representations from context representations, with stop-gradient on the target branch. Avoiding collapse requires a complete validated recipe that retains its regularization/target updates, not merely a copied latent-space MSE. B08 still needs domain labels to read out physical quantities. Related work such as FACTS already combines graph memory and temporal modeling; the combination itself does not establish originality. [FACTS](https://arxiv.org/abs/2410.20922)

F04's `tree_mode` must select one of the following: `representation` refines graph resolution and uses F05/F06; `expert` routes computational experts and uses expert branches and merging; `candidate` searches action/structure/experiment candidates and uses F08. The three cannot share the phrase node splitting without defining the object involved. The base instance defaults to a fixed graph without trees; extension experiments enable them individually.

F05/F06 training includes coarse–fine round trips, dynamics consistency at a common physical scale, and readout consistency at the same time. Newly added details cannot be required to be recovered losslessly by coarse representations. F02's variance head expresses only specified distribution parameters; model uncertainty may additionally use disagreement among models. A single scale output cannot distinguish every error source.

F05/F06 must migrate h, z, masks, and entity mappings for every affected layer. Domain states such as coordinates and masses migrate under their own rules, not through the same unitless projection as hidden vectors. A representation tree refines only computational representations; expert-tree leaves produce outputs using the appropriate FFN/graph blocks and merge using declared gating weights, with expert capacity and load losses recorded separately. Candidate trees only isolate candidates and do not thereby claim improved spatial resolution.


---

[← Previous](section-06.md) · [Contents](../../../SUMMARY.md) · [Next →](section-08.md)

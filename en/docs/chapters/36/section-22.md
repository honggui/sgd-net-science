# 20. Detailed Construction of Group F: Error-Driven Bounded Adaptation

**On This Page**

- [F01 Error Evidence Extraction](#section-001)
- [F02 Calibration and Out-of-Domain Detection](#section-002)
- [F03 Benefit and Budget Selection](#section-003)
- [F04 Dynamic Tree Routing and Growth Candidates](#section-004)
- [F05 Prolong State Expansion](#section-005)
- [F06 Restrict State Aggregation](#section-006)
- [F07 Local Refinement Execution](#section-007)
- [F08 Candidate-State Versions](#section-008)

---

## F01 Error Evidence Extraction <a href="#section-001" id="section-001"></a>

**Role and mathematics.** Output separate terms rather than combine them into one intelligent error:

$$
e_{diag}=(e_{obs},e_{constraint},e_{solver},e_{calibration},e_{reference}),
\quad e_{obs}=\|W_{unit}M(y-\hat y^-)\|.
$$

**Construction and use.** Bind each term to units, masks, provenance, sample counts, and empirical/bounded/unknown labels. Without ground truth, do not generate assumed true errors. A single residual cannot fully separate unidentifiable observation, representation, and learning errors. F01 supplies causal evidence to EV02 and action support to F03 rather than directly determining growth. It shares numerical-reliability contracts with FN04.

## F02 Calibration and Out-of-Domain Detection <a href="#section-002" id="section-002"></a>

**Role and mathematics.** A classification-calibration example is

$$
p_T(k\mid x)=\operatorname{softmax}(l(x)/T)_k,\quad
T^*=\arg\min_{T>0}\sum_{(x,y)\in D_{cal}}-\log p_T(y\mid x).
$$

**Construction and use.** Fit T on an independent calibration set. Out-of-domain scores may use feature distances or model disagreement but require dedicated validation of detection and false-positive rates. Recalibrate after backbone or data-domain changes. Confidence is not an automatic guarantee about error probability; low-entropy models can still be wrong. Evaluate every uncertainty-triggered rule on distributions approximating deployment.

## F03 Benefit and Budget Selection <a href="#section-003" id="section-003"></a>

**Role and mathematics.** Select among a finite set of candidate actions b

$$
b^*=\arg\max_{b\in\mathcal B_{allowed}}\left[\widehat{\Delta Q}(e,b)-\lambda_CCost(b)\right],
\quad Cost(b)+Reserve\leq Budget.
$$

**Construction and use.** Candidates include no change, more iterations, additional observations, local refinement, and training, not only growth. First run a rule-based policy and record actual benefits, then train a benefit predictor. Its predictions cannot prove hard-budget compliance; H03 enforces measurement. No action is allowed when benefits are unknown or all negative. Register reward/cost units and the source of λ.

## F04 Dynamic Tree Routing and Growth Candidates <a href="#section-004" id="section-004"></a>

**Role and mathematics.** One soft path for an expert tree is

$$
p_{left}=\sigma(g_\theta(s)),\quad
\pi_\ell(s)=\prod_{v\in path(\ell)}p_v^{b_v}(1-p_v)^{1-b_v},
\quad \hat y=\sum_\ell\pi_\ell f_\ell(s).
$$

**Construction and use.** Geometric configurations use invariants for s. Threshold-based hard routing must handle nondifferentiability and boundary jumps. Representation trees refine graphs and call P/Q, expert trees change local functions, and candidate trees manage search copies; these are not interchangeable. Persistent splits undergo initialization, training, and competition through EV03–EV07. Larger trees are not automatically better; expansion limits and pruning rules are required.

## F05 Prolong State Expansion <a href="#section-005" id="section-005"></a>

**Role and mathematics.** For coarse and fine computational representations, reference linear mappings are

$$
z_f=Pz_c,\quad h_f=Ph_c+\delta_\theta(h_c,geometry),\qquad P\mathbf1=\mathbf1.
$$

**Construction and use.** Unit row sums in P preserve constants but do not guarantee conservation. Migrate z/h at every layer; detail heads may be zero-initialized. Vector-mapping coefficients must be invariant, while physical positions, masses, and labels follow their own unit rules. Train/check using common physical readouts and stable norms without requiring all fine states to be losslessly recoverable from coarse states. P does not change the number of real entities.

## F06 Restrict State Aggregation <a href="#section-006" id="section-006"></a>

**Role and mathematics.** For positive-definite W_f and full-column-rank P, use weighted least-squares restriction:

$$
Q=(P^TW_fP)^{-1}P^TW_f,\quad QP=I,\qquad z_c=Qz_f.
$$

**Construction and use.** Implement this through linear solves rather than explicit inversion. If rank-deficient or ill-conditioned, change the mapping, regularize, and register its properties again. QP=I preserves only the coarse subspace; it implies neither PQ=I nor lossless compression of arbitrary details. Train learned Q jointly with consistency and actual task losses, checking vector types, weights, and physical-quantity rules. Node averaging is not a universally conservative restriction.

## F07 Local Refinement Execution <a href="#section-007" id="section-007"></a>

**Role and mathematics.** For selected region Ω and frozen external boundary b:

$$
s_\Omega^{k+1}=\Phi_\theta(s_\Omega^k,b,c),\quad k<K_{max},
\qquad\text{或 }R_\theta(s_\Omega;b,c)=0.
$$

**Construction and use.** Configure explicit repeated blocks and IM implicit solves separately. Halo depth should cover the message receptive field; boundaries that may be coupled unknowns cannot be arbitrarily fixed. Refinement k does not advance real time; versions, units, and stopping rules remain fixed. Gradients follow the actual execution path when training shared blocks. Boundary truncation errors and solver failures enter F01, rather than reporting only decreases in local residuals.

## F08 Candidate-State Versions <a href="#section-008" id="section-008"></a>

**Role and mathematics.** A snapshot explicitly contains

$$
S_\nu=(h^{1:L},z^{1:L},x,IDs,M,\mathcal G,\nu_\theta,\nu_{cal},\nu_{layout},t),
\quad S^{candidate}=\operatorname{Fork}(S_\nu).
$$

**Construction and use.** Implementations may copy objects or share read-only objects, but candidate writes must not contaminate active trajectories. Retain parent and candidate clocks. Caches with mismatched graph versions cannot be loaded directly. This unit has no trainable parameters; candidate models in training also need independent optimizer states. Discarding a computational candidate only reclaims resources and cannot undo device actions that already occurred.


---

[← Previous](section-21.md) · [Contents](../../../SUMMARY.md) · [Next →](section-23.md)

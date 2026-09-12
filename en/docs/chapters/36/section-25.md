# 23. Detailed Construction of Group L: Stability, Dissipation, and Correction

**On This Page**

- [L01 Stability Contract and Reference](#section-001)
- [L02 V/Storage-Function Head](#section-002)
- [L03 Drift/Descent Evaluation](#section-003)
- [L04 Disturbance and Supply Budgets](#section-004)
- [L05 State/Action Correction](#section-005)
- [L06 Topology-Switch Compatibility](#section-006)
- [L07 Stability Training and Counterexamples](#section-007)
- [L08 Evidence and Failure Handling](#section-008)

---

## L01 Stability Contract and Reference <a href="#section-001" id="section-001"></a>

**Role and mathematics.** Specify state errors and conditions to check:

$$
e=\Psi(s,s_{ref}),\qquad
V(e^+)-V(e)\leq-\alpha(\Delta t)\|e\|^2+\sigma(\|w\|,\Delta t).
$$

**Construction and use.** Ψ may be a Euclidean difference, observer error, or valid local manifold error. Specify scope, domain, inputs, noise bounds, graph, and V versions. Internal-memory differences, numerical iterations, and physical tracking differ. Real-world systems may oscillate or be chaotic; raw states cannot be forced toward zero. Mark undefined static tasks not applicable rather than fabricate V for completeness.

## L02 V/Storage-Function Head <a href="#section-002" id="section-002"></a>

**Role and mathematics.** For normalized Euclidean errors, construct a positive-definite candidate:

$$
V_\psi(e)=e^T(R^TR+\epsilon I)e+\|g_\psi(e)-g_\psi(0)\|^2,\quad \epsilon>0.
$$

**Construction and use.** R/g may be trained. Version V separately from predictors, using gradients in L03/L05/L07. The expression guarantees V(0)=0 and positive definiteness, not descent. Geometric invariance additionally requires Q^TPQ=P or construction from invariant inputs; arbitrary R does not satisfy this. V values across graphs of different dimensions cannot be directly compared without common readouts or weight mappings.

## L03 Drift/Descent Evaluation <a href="#section-003" id="section-003"></a>

**Role and mathematics.** The deterministic discrete-condition residual is

$$
r_V=V(e^+)-V(e)+\alpha\|e\|^2-\sigma(\|w\|).
$$

**Construction and use.** When checking r_V≤tolerance, also record floating-point, measurement, and model-error budgets; changes in Δt require corresponding coefficients. Stochastic processes require conditional expected-drift checks and uncertainty estimation; finite-sample means are not pathwise proofs. Online computation can use only actually available errors. V for unknown true states remains an offline metric or uses validated observer bounds.

## L04 Disturbance and Supply Budgets <a href="#section-004" id="section-004"></a>

**Role and mathematics.** With external driving, use a dissipative form:

$$
V(s^+)-V(s)\leq Supply(u,y,\Delta t)-Dissipation(s,\Delta t)+\epsilon_{model}.
$$

**Construction and use.** Supply may be work, heat, or input gain, but must use correct physical quantities and units. Disturbance estimates differ from guaranteed upper bounds. Learning a noise scale does not establish a worst-case disturbance bound. This unit permits justified energy increases such as lifting and provides evidence-based input budgets for L03; budgets must not be arbitrarily enlarged until all candidates pass.

## L05 State/Action Correction <a href="#section-005" id="section-005"></a>

**Role and mathematics.** A correction to candidate a_nom can be written as

$$
\min_{a,\delta\geq0}\|a-a_{nom}\|_R^2+p\delta^2,
\quad \Delta V(a)\leq-\alpha\|e\|^2+\sigma+\delta,\quad a\in\mathcal U_{feasible}.
$$

**Construction and use.** This becomes a convex QP only when the structure meets the necessary conditions. General nonlinear constraints may be nonconvex, infeasible, or time out. δ relaxes strict descent and needs an explicit allowable amount; hard device constraints cannot be relaxed. Corrected outputs form new candidates that pass through H04/H05 again. Gradients through optimization layers require IM05 regularity conditions; hard rejection must not be misrepresented as differentiable.

## L06 Topology-Switch Compatibility <a href="#section-006" id="section-006"></a>

**Role and mathematics.** For migration P and the graphs before and after it, require

$$
V_{G'}(Ps)\leq\mu V_G(s)+\eta_{switch}.
$$

**Construction and use.** μ, η, and switching frequency jointly determine long-term properties. Even a small η at each switch cannot accumulate without limit while stability is claimed. Prefer a common V, common physical readouts, or conditional dwell-time/budget rules. Graph-summed energy must account for mass/volume weights; otherwise copied nodes artificially increase V. Failed checks retain the original active version, and candidate migration does not take effect.

## L07 Stability Training and Counterexamples <a href="#section-007" id="section-007"></a>

**Role and mathematics.** Construct training and counterexample searches over an explicit domain D:

$$
\mathcal L_V=\mathbb E_{e\sim D}[r_V(e)]_+,
\qquad e_{adv}\approx\arg\max_{e\in D}r_V(e).
$$

**Construction and use.** Train V and selected model parameters jointly with task losses, fixing one side where necessary to prevent objectives from masking each other. Feed counterexamples back into training while keeping final validation independent. Failure to find counterexamples through local optimization or random search is not a global proof. Task metrics should detect models that reduce stability losses by degenerating into nonprediction; optimizing only V is insufficient.

## L08 Evidence and Failure Handling <a href="#section-008" id="section-008"></a>

**Role and mathematics.** Bind conditions to evidence status:

$$
E_V=(scope,domain,modelVersion,VVersion,method,tolerance,result).
$$

**Construction and use.** Distinguish empirical sampling, numerical validation, and proofs with assumptions. Model/graph/precision changes may invalidate old evidence. H04/H05 consume explicit statuses rather than stability scores. Handle timeouts, unknowns, and violations separately; static tasks may be not applicable. This mechanism connects mathematical analysis to execution and does not establish chip certification or stability across all domains.



---

[← Previous](section-24.md) · [Contents](../../../SUMMARY.md) · [Next →](section-26.md)

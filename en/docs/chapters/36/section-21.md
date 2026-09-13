# 19. Detailed Construction of Group E: Prediction, Generation, and Task Readouts

**On This Page**

- [E01 State-Transition Head](#section-001)
- [E02 Observation Likelihood Head](#section-002)
- [E03 Contact/Task-Effect Head](#section-003)
- [E04 Embodiment Action Head](#section-004)
- [E05 Physical-Property Head](#section-005)
- [E06 Biological/Functional Head](#section-006)
- [E07 Conditional Generation Head](#section-007)
- [E08 Readout and Pooling](#section-008)

---

## E01 State-Transition Head <a href="#section-001" id="section-001"></a>

**Role and mathematics.** Convert hidden states and actions into conditional future distributions. One mixture-increment example is

$$
p(x_{t+1}\mid h_t,a_t,c)=\sum_{k=1}^{K}\pi_k\mathcal N(x_{t+1};x_t+\mu_k,\Sigma_k),
\quad \pi=\operatorname{softmax}(l),\quad \Sigma_k=L_kL_k^T+\epsilon I.
$$

**Construction and use.** Output mode weights, increments, and valid covariance parameters separately; deterministic tasks may use only a mean head. Actions and Δt must be conditions or handled by defined dynamics; rollouts recur in isolated states. Multistep training must register the ratio of teacher forcing to free rollout. Single-step accuracy does not guarantee long-horizon accuracy. Means and covariances in rotation-equivariant settings must use the appropriate representations; ordinary diagonal covariances on fixed axes lack arbitrary rotational consistency.

## E02 Observation Likelihood Head <a href="#section-002" id="section-002"></a>

**Role and mathematics.** Map hidden states to the actual measurement space. Continuous observations can be written as

$$
p(y\mid h)=\mathcal N(y;\mu_\theta(h),\operatorname{diag}(\sigma_\theta(h)^2)),\quad
\sigma=\operatorname{softplus}(s)+\epsilon.
$$

**Construction and use.** Select Gaussian, count, or event distributions by sensor type, train through negative log-likelihood, and handle missingness with A05. A diagonal Gaussian is a specific observation assumption, not a representation of all uncertainty. Geometric-vector distributions must respect group actions. Provide pre-observation prediction snapshots to D05; later labels must not retroactively contaminate the same prediction.

## E03 Contact/Task-Effect Head <a href="#section-003" id="section-003"></a>

**Role and mathematics.** For edge contacts and vector effects, construct separately

$$
p_{ij}=\sigma(w^Tm_{ij}+b),\qquad
\hat d_i=\sum_j\phi_d(m_{ij})r_{ij}.
$$

**Construction and use.** Separate actual contact labels from predicted-edge confidence. Registered weighted cross-entropy may address class imbalance, while effect vectors use scaled regression. Without contact labels, report only proxy targets such as geometric proximity. This head connects spatial representations to task conditions; high contact probability does not establish stable grasping or physical feasibility. Friction, torque, and execution latency require separate validation.

## E04 Embodiment Action Head <a href="#section-004" id="section-004"></a>

**Role and mathematics.** For scalar actuator components with finite upper and lower bounds, let

$$
a_{t:t+H_a-1}=M_a\odot\left[a_{mid}+a_{range}\odot\tanh g_a(h,c_{morph},goal)\right].
$$

**Construction and use.** Output action chunks of fixed duration and map them to native joints/actuators by embodiment; H05 commits only permitted prefixes. tanh supplies box constraints only, without guaranteeing velocity, collision, or contact feasibility. Rotational actions require valid representations; coordinatewise quaternion clipping is not an orientation constraint. Training may use imitation or explicit RL objectives, whose labels, exploration, and evaluation must remain distinct. Unchecked outputs are only candidates.

## E05 Physical-Property Head <a href="#section-005" id="section-005"></a>

**Role and mathematics.** One energy-consistent molecular/materials configuration is

$$
\hat E=\sum_i e_\theta(h_i),\qquad \hat F_i=-\nabla_{x_i}\hat E,
\quad \hat\sigma=\frac1{\Omega}\left.\frac{\partial\hat E((I+\varepsilon)H)}{\partial\varepsilon}\right|_{\varepsilon=0}.
$$

**Construction and use.** Specify stress signs, strain conventions, and how atoms deform with the unit cell. The equation follows one convention and cannot be compared directly across data sources. Force supervision involves energy derivatives with respect to coordinates, requiring mixed second-order derivatives during parameter training. Distance cutoffs and neighbor changes must be smoothed and validated. Good structural-energy fitting does not guarantee good forces; report energy, force, stress, and rollout separately, with suitable separate heads for nonconservative systems.

## E06 Biological/Functional Head <a href="#section-006" id="section-006"></a>

**Role and mathematics.** Count outputs may use a negative binomial distribution:

$$
\mu_{ig}=L_i\operatorname{softmax}_g(f_\theta(h_i,c)),\quad
Y_{ig}\sim NB(\mu_{ig},r_g),\quad
\operatorname{Var}(Y_{ig})=\mu_{ig}+\mu_{ig}^2/r_g.
$$

**Construction and use.** r_g>0 may be parameterized by softplus; L_i is exposure or known library size. This form assumes compositional means over the modeled gene set; use a different mean model when fixed totals are unsuitable. Phenotype/survival/endpoint targets use separate likelihoods and label time domains rather than sharing a unitless MSE across different endpoints. Conditional association predictions are not intervention effects or drug efficacy.

## E07 Conditional Generation Head <a href="#section-007" id="section-007"></a>

The 2026-09-13 integration contract is in [memory–reasoning consolidation and parallel generation](../../../guide/memory-reasoning-parallel-generation.md). A continuous example exists; discrete token generation still needs corruption, loss, and sampling definitions. Each round reads frozen snapshots; generation clocks must not advance D04 physical time. Local updates must handle dependencies and cache invalidation.

**Role and mathematics.** As an optional example, use conditional flow matching:

$$
x_\tau=(1-\tau)x_0+\tau x_1,\quad
\mathcal L_{FM}=\mathbb E\|v_\theta(x_\tau,\tau,c)-(x_1-x_0)\|^2,
\quad \frac{dx}{d\tau}=v_\theta(x,\tau,c).
$$

**Construction and use.** x_0 is noise and x_1 data; declare the coupling and noise distributions. τ is a generation clock independent of physical t. Inference integration produces candidates filtered by domain checks. This linear path need not stay on a manifold or within a constrained domain; constrained generation requires a redefined path. Discrete categories cannot be generated directly along this real-valued ODE and need an explicit discrete model. Diffusion modes configure noise schedules separately; a noise head alone is not a complete algorithm.

## E08 Readout and Pooling <a href="#section-008" id="section-008"></a>

**Role and mathematics.** Choose readouts consistent with the task quantity:

$$
y_{ext}=\sum_iw_i\phi(h_i),\quad
y_{int}=\frac{\sum_iw_i\phi(h_i)}{\sum_iw_i},\quad
\hat u(a,x_q)=g_\theta\!\left(x_q,\sum_iw_i\kappa_\theta(x_q,x_i)h_i\right).
$$

**Construction and use.** The first two expressions apply to the corresponding extensive/intensive quantities, with a positive denominator required. The third is a candidate field readout at query locations, with weights matching the discrete measure. Ordinary summary attention guarantees neither mass conservation nor cross-mesh consistency. Train with labels or query-point supervision and restore output scales through B08/A02. Empty graphs require explicit missingness rather than division by zero.


---

[← Previous](section-20.md) · [Contents](../../../SUMMARY.md) · [Next →](section-22.md)

# 21. Detailed Construction of Group G: Training, Retrospective Analysis, and Continual Learning

**On This Page**

- [G01 Loss Assembly](#section-001)
- [G02 Data Sampling and Splitting](#section-002)
- [G03 Hybrid Optimizer](#section-003)
- [G04 JEPA Teacher/Predictor](#section-004)
- [G05 Replay and Retrospective Analysis](#section-005)
- [G06 Adaptation/Distillation/Merging](#section-006)
- [G07 Independent Evaluation](#section-007)
- [G08 Active Experiment Selection](#section-008)

---

## G01 Loss Assembly <a href="#section-001" id="section-001"></a>

**Role and mathematics.** Combine task-defined dimensionless losses:

$$
\mathcal L=\sum_{r\in\mathcal T_{enabled}}\lambda_r\frac{\sum_iM_{ir}\ell_r(\hat y_{ir},y_{ir})}{\max(1,\sum_iM_{ir})}
+\lambda_{reg}\Omega(\theta).
$$

**Construction and use.** r may include likelihood, geometry, forces, JEPA, migration, and stability regularization; disabled objectives do not participate. Register scales and weights; do not tune λ on the test set. Calling a loss a functional is a choice of mathematical expression and does not automatically make it real physical energy, convex, or uniquely minimized. Conflicting objectives require monitoring separate metrics.

## G02 Data Sampling and Splitting <a href="#section-002" id="section-002"></a>

**Role and mathematics.** Partition sets by provenance group g:

$$
\mathcal G_{train}\cap\mathcal G_{test}=\varnothing,\qquad
D_{test}=\{x:g(x)\in\mathcal G_{test}\}.
$$

**Construction and use.** Groups may be animals, donors, molecular scaffolds, experimental batches, or time windows; augmented views and derived records inherit their original group. Fit normalization on training data and keep calibration separate. Randomly dispersing near-duplicate experiments does not establish out-of-domain generalization. Explicitly evaluate uncertainty in domains with insufficient samples; more random splits do not create independent samples.

## G03 Hybrid Optimizer <a href="#section-003" id="section-003"></a>

**Role and mathematics.** For Euclidean parameters, consider a reference AdamW update:

$$
m_k=\beta_1m_{k-1}+(1-\beta_1)g_k,\quad
v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^2,
$$
$$
\theta_{k+1}=(1-\eta\lambda)\theta_k-\eta\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon},
\quad \hat m_k=m_k/(1-\beta_1^k),\quad\hat v_k=v_k/(1-\beta_2^k).
$$

**Construction and use.** Gradients may come from explicit backpropagation or IM05. Manifold parameters use separately declared retractions, while hard tree structures are selected in the outer loop. Initialize optimizer slots for new parameters and migrate old slots under the mapping. Online training updates only candidate domains; active shapes must not change during backpropagation. Optimizers may run on companion CPUs/accelerators; location does not alter mathematical responsibility.

## G04 JEPA Teacher/Predictor <a href="#section-004" id="section-004"></a>

**Role and mathematics.** For example, with matched-view targets:

$$
\mathcal L_J=\|p_\theta(f_\theta(x_{context}))-\operatorname{sg}(f_{\bar\theta}(x_{target}))\|^2,
\quad\bar\theta\leftarrow\tau\bar\theta+(1-\tau)\theta.
$$

**Construction and use.** This expression is only a teacher–predictor skeleton. Use complete view, mask, target-normalization, and anti-collapse strategies and validate representation variance; EMA or stop-gradient alone does not guarantee noncollapse. The target branch must not leak into online inputs, and the teacher receives no gradient from this term. Bridging to physical states still requires B08 observation supervision; a small latent loss does not imply physical correctness.

## G05 Replay and Retrospective Analysis <a href="#section-005" id="section-005"></a>

**Role and mathematics.** Experience records should retain at least

$$
b_t=(o_{\leq t}^{available},\hat y_t^-,a_t,y_t,versions,source,cost),\quad
P(sample=b)\propto\epsilon+priority(b).
$$

**Construction and use.** Priority may consider repeated errors, representativeness, and rarity rather than replaying only failures or high-loss samples. If the objective is risk under the original distribution, nonuniform sampling requires corresponding weights or an explicit declaration that the training distribution changed. Store original events, causal hypotheses, and simulated counterfactuals in separate layers. Retrospective analysis first tests error sources, then proposes revisions; producing an explanation is not evidence of correct attribution.

## G06 Adaptation/Distillation/Merging <a href="#section-006" id="section-006"></a>

**Role and mathematics.** Combine current-task training with retention of earlier tasks:

$$
\mathcal L_{adapt}=\mathcal L_{new}(\theta')+
\lambda\mathbb E_{x\sim D_{replay}}D(p_{old}(\cdot|x),p_{new}(\cdot|x)).
$$

**Construction and use.** D may be KL divergence on applicable distributions or structural consistency. The old model is only a behavioral reference; do not force known errors to be copied. Adapters may use low-rank parameterization ΔW=UV^T, with rank and initialization registered. Models with different topologies cannot have parameters averaged elementwise; first align or distill into the target structure. Persistent acceptance uses new data and old-task thresholds. Low replay loss does not guarantee no forgetting.

## G07 Independent Evaluation <a href="#section-007" id="section-007"></a>

**Role and mathematics.** Estimate risk and resources on independent test units:

$$
\hat R=\frac1{n}\sum_{i=1}^{n}\ell_i,\qquad
Report=(\hat R,CI,latency,memory,energy,failure).
$$

**Construction and use.** Compute CI using genuinely independent units or grouped resampling; adjacent time points from one animal are not independent replicates. Report fixed-budget ablations, out-of-domain splits, and multiple random seeds. Without energy measurements, do not fabricate J/task. Repeated model selection on the same held-out set leads to overfitting and requires subsequent data or an independent final set.

## G08 Active Experiment Selection <a href="#section-008" id="section-008"></a>

**Role and mathematics.** Within an already permitted experiment/observation set, candidate scoring may be written as

$$
a^*=\arg\max_{a\in\mathcal A_{allowed}}
\left[\mathbb E(\Delta Q\mid a,D)-\lambda C(a)\right].
$$

**Construction and use.** Gains may come from information acquisition, endpoint improvements, or reduced model disagreement, but estimates depend on the current model and are not known real effects. Include replicate experiments, negative controls, and multifidelity costs. This unit proposes suggestions; H06 executes under existing experimental authorization. Active selection does not mean autonomously changing biological protocols or device limits.


---

[← Previous](section-22.md) · [Contents](../../../SUMMARY.md) · [Next →](section-24.md)

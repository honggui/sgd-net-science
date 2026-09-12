# 22. Detailed Construction of Group H: Tools, Execution, and Evidence

Group H's equations describe protocol invariants and resource constraints; they do not fabricate trainable neural layers for external systems.

**On This Page**

- [H01 Domain Tool Adaptation](#section-001)
- [H02 ELN/LIMS/Data Retrieval](#section-002)
- [H03 Hard Budget Management](#section-003)
- [H04 Domain Output Checks](#section-004)
- [H05 Commit and Fallback](#section-005)
- [H06 Device and Experiment Orchestration](#section-006)
- [H07 Compilation and Numerical Configuration](#section-007)
- [H08 Auditing and Monitoring](#section-008)

---

## H01 Domain Tool Adaptation <a href="#section-001" id="section-001"></a>

**Role and mathematics.** External solver calls return structured results:

$$
Tool(q;version,budget)\to(value,fidelity,error,status,provenance).
$$

**Construction and use.** Define input units, boundaries, randomness, and failure reasons separately for DFT, MD, simulations, and external models. Returned approximate solutions do not automatically become ground truth. Nondifferentiable tools may provide offline labels/candidate evaluations without participating in gradients by default. Surrogate replacements need independent error validation. Tool timeouts, network waiting, and data transfer count toward total task cost.

## H02 ELN/LIMS/Data Retrieval <a href="#section-002" id="section-002"></a>

**Role and mathematics.** Retrieval selects records from the set permitted by versions and permissions:

$$
\mathcal R_q=\operatorname{TopK}_{r\in D_{allowed}}score(q,r),\quad
result=(records,versions,sources).
$$

**Construction and use.** Support exact-ID lookup and semantic retrieval, recognizing their different guarantees. Approximate vector matches cannot replace arbitrary address queries. Model states store summaries; record systems still provide original evidence. Retrieval results are context candidates, not automatically facts or executable instructions. Exact retrieval needs no parameters; learned rankers register data and evaluation separately.

## H03 Hard Budget Management <a href="#section-003" id="section-003"></a>

**Role and mathematics.** Admission requires conservative resource constraints, for example

$$
T_{remain}\geq T_{candidate}^{bound}+T_{check}^{bound}+T_{fallback}^{bound},
\quad M_{active}+M_{candidate}+M_{reserve}\leq M_{limit}.
$$

**Construction and use.** Without reliable worst-case latency bounds, hard real-time guarantees cannot be claimed; deadline aborts and isolation only provide specified fallback paths. Measurement covers queues, transfers, nonpreemptible segments, and logging contention. F03 benefit predictions cannot alter budget facts. Training/retrospective analysis runs in background resource domains; exceeding budgets pauses candidates without affecting the active version or minimum required handling.

## H04 Domain Output Checks <a href="#section-004" id="section-004"></a>

**Role and mathematics.** Produce three-valued results for each proposition:

$$
check_j(cand)\in\{pass,violation,unknown\}.
$$

**Construction and use.** Check units, shapes, geometry, validity periods, constraints, and numerical evidence; permit the corresponding use only when all required propositions hold. Missing evidence returns unknown. No issue found is not automatically proof. Static research reports may retain unknowns for analysis, while specific contracts determine whether device actions are allowed. H04 does not promote numerical passes to scientific causal conclusions.

## H05 Commit and Fallback <a href="#section-005" id="section-005"></a>

**Role and mathematics.** Bind decisions to exact candidate versions:

$$
Commit=(requestID,candidateID,digest,epoch,decision),\qquad
Committed\not\Rightarrow Executed.
$$

**Construction and use.** Separate commit, device acceptance, and execution feedback. Handle retries with the same request identity according to protocol; communication uncertainty must not cause blind repetition of physical actions. Recheck modified candidates. Reports and device actions use different applicable contracts. Fallback may replace computational candidates or perform predefined actions but cannot undo experimental effects that already occurred.

## H06 Device and Experiment Orchestration <a href="#section-006" id="section-006"></a>

**Role and mathematics.** A protocol state machine can be represented as

$$
s_{dev}^{+}=\delta(s_{dev},command,feedback),\qquad command\in Allowed(s_{dev},protocol).
$$

**Construction and use.** Device adapters handle native units, ACK semantics, interlocks, and result attribution, retaining sample/session/protocol versions. Models may supply candidates, but actual execution must satisfy existing authorization and state conditions. Biological effects may be delayed and noisy; device ACKs are not experiment-success labels. This unit belongs to system orchestration, not additional inference capability.

## H07 Compilation and Numerical Configuration <a href="#section-007" id="section-007"></a>

**Role and mathematics.** Lower logical graphs into execution plans:

$$
Plan=Compile(\mathcal A,shapes,dtypes,layout,capabilities),\quad
\|F_{Plan}(x)-F_{ref}(x)\|\leq\epsilon_{profile}
$$

(The latter is a target requiring validation, not an automatic compilation guarantee.)

**Construction and use.** Explicitly list gather/scatter, GEMM, reductions, scan, automatic differentiation, and solver primitives. Dynamic structures may use shape buckets, precompiled libraries, or background compilation; unsupported cases report design gaps. Aegis-X scope is revised according to SGD-Net requirements, without fabricating actual capabilities. Reduced precision requires task/derivative error validation; weight errors alone are insufficient.

## H08 Auditing and Monitoring <a href="#section-008" id="section-008"></a>

**Role and mathematics.** Evidence events may form versioned dependency chains:

$$
e_k=(id_k,parent_k,time_k,objectVersions_k,decision_k,reason_k,source_k).
$$

**Construction and use.** Retain traceable relationships showing predictions precede outcomes and linking candidate training, checks, and releases. Digests may verify integrity, but hashes do not prove scientific truth. Explicitly record overflow/loss; large log writes must not block critical execution indefinitely. Auditing supports retrospective analysis and model rollback, without replacing runtime authorization or task evaluation.


---

[← Previous](section-23.md) · [Contents](../../../SUMMARY.md) · [Next →](section-25.md)

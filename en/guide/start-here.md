# Start here

SGD-Net Science is a collection of model architecture research documents exploring how to combine graph structure, temporal memory, geometric priors, and adaptive computation. The public release currently contains designs and validation plans, without an accompanying model implementation or training results.

**On this page**

- [A quick introduction](#section-001)
- [Choose a route for your goal](#section-002)
- [Distinctions to retain while reading](#section-003)

---

## A quick introduction <a href="#section-001" id="section-001"></a>

Read the paper's abstract, research questions, and limitations first, then the boundaries of generality, and finally the task cards. You do not need to study every component on a first reading; consult the complete component catalog when a specific question arises.

## Choose a route for your goal <a href="#section-002" id="section-002"></a>

| Your goal | Suggested route |
|---|---|
| Understand the overall proposal | 01 Paper → 32 Innovation boundaries → 36 Architecture guide |
| Examine mathematical feasibility | 48 Function spaces → 44 Stability → 45 Implicit solving |
| Study continual learning | 46 Dynamic growth → 50 Foundations of retrospective analysis → 17 Detailed mechanisms |
| Start an experimental implementation | 49 Task cards → 04 High-level design → 05 Detailed design → 13 Layer operators |
| Apply the architecture to scientific tasks | 35 Cross-domain roadmap → Relevant domain chapter → 49 Evaluation tasks |

## Distinctions to retain while reading <a href="#section-003" id="section-003"></a>

An available component need not always be enabled. Static tasks may not require temporal state, and graphs without physical geometry need not use equivariant coordinates. Treat theoretical conditions, training objectives, design pseudocode, and experimental results as distinct categories.

For method equations, check variables, units, norms, and assumptions. For growth or solving procedures, examine failure conditions and total cost. For biological mechanisms, distinguish predictive associations, simulations, and evidence from actual interventions.

- [01 Paper and research questions](../docs/01.md)
- [32 Integration assessment and innovation boundaries](../docs/32.md)
- [36 Unified architecture and component catalog](../docs/36.md)
- [47 Generality and capability boundaries](../docs/47.md)
- [48 Function spaces and numerical error](../docs/48.md)
- [44 Lyapunov stability](../docs/44.md)
- [45 Implicit solving](../docs/45.md)
- [46 Dynamic growth and continual evolution](../docs/46.md)
- [50 Reflection, review, and retrospective analysis](../docs/50.md)
- [17 Retrospective analysis and self-evolution design](../docs/17.md)
- [49 Task cards and evaluation roadmap](../docs/49.md)
- [04 High-level implementation design](../docs/04.md)
- [05 Detailed implementation design](../docs/05.md)
- [13 Layers and operators explained](../docs/13.md)
- [35 Cross-domain extension roadmap](../docs/35.md)

---

[← Previous](../README.md) · [Full contents](../SUMMARY.md) · [Next →](glossary.md)

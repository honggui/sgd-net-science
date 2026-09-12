# 5. Layer 0: Input Encoding Layer

**On This Page**

- [5.1 Function](#section-001)
- [5.2 Operator Definition](#section-002)
- [5.3 Implementation Interface](#section-003)

---

## 5.1 Function <a href="#section-001" id="section-001"></a>

The input encoding layer converts raw data into initial objects that can be represented as a graph.

Inputs differ by scenario:

| Scenario | Raw input | Encoded objects |
|---|---|---|
| PDE / scientific computing | Sample points, boundary conditions, physical quantities | Mesh points, physical features |
| Safe LLM | Text, entities, facts, retrieval results | Concept nodes, factual edges |
| Biological structures | PDB/mmCIF/FASTA/SMILES | residue, atom, ligand, chain |
| Materials discovery | CIF, VASP output, XRD, SEM | Lattice, element, process, and characterization nodes |
| MoE acceleration | Tokens, router logits, expert IDs | Token nodes, expert nodes |

## 5.2 Operator Definition <a href="#section-002" id="section-002"></a>

Input encoding operator:

$$
Z_0 = \operatorname{Encode}(\mathcal{D}_{raw};\theta_{enc})
$$

where:

- $$\mathcal{D}_{raw}$$ is the raw input;
- $$Z_0$$ is the initial set of structured tokens / objects;
- $$\theta_{enc}$$ denotes the encoding parameters.

## 5.3 Implementation Interface <a href="#section-003" id="section-003"></a>

```text
InputEncoder.encode(raw_input) -> EncodedObjects
```


---

[← Previous](section-04.md) · [Contents](../../../SUMMARY.md) · [Next →](section-06.md)

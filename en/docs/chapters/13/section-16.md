# 14. Layer 9: Task Output Head

**On This Page**

- [14.1 Function](#section-001)
- [14.2 Node-Level Output](#section-002)
- [14.3 Graph-Level Output](#section-003)
- [14.4 Path-Level Output](#section-004)

---

## 14.1 Function <a href="#section-001" id="section-001"></a>

The output head maps SGD-Net's final graph state to task results.

## 14.2 Node-Level Output <a href="#section-002" id="section-002"></a>

$$
\hat{y}_i=f_{head}(x_i)
$$

Suitable for node classification, local physical-quantity prediction, and residue-risk prediction.

## 14.3 Graph-Level Output <a href="#section-003" id="section-003"></a>

$$
\hat{y}=f_{head}(\operatorname{Readout}(X))
$$

Readout can be:

- mean pooling;
- sum pooling;
- attention pooling;
- set transformer;
- graph-level token.

## 14.4 Path-Level Output <a href="#section-004" id="section-004"></a>

In Safe LLM / Agent scenarios, a path can be output:

$$
path=(v_1,e_{12},v_2,\ldots,v_k)
$$

along with an explanation:

```text
answer
supporting_path
confidence
blocked_paths
refinement_events
```


---

[← Previous](section-15.md) · [Contents](../../../SUMMARY.md) · [Next →](section-17.md)

# 5. Layer 0：输入编码层

**本页目录**

- [5.1 功能](#section-001)
- [5.2 算子定义](#section-002)
- [5.3 工程接口](#section-003)

---

## 5.1 功能 <a href="#section-001" id="section-001"></a>

输入编码层负责将原始数据转换为可图化的初始对象。

不同场景的输入不同：

| 场景 | 原始输入 | 编码对象 |
|---|---|---|
| PDE / 科学计算 | 采样点、边界条件、物理量 | 网格点、物理特征 |
| Safe LLM | 文本、实体、事实、检索结果 | 概念节点、事实边 |
| 生物结构 | PDB/mmCIF/FASTA/SMILES | residue、atom、ligand、chain |
| 材料发现 | CIF、VASP 输出、XRD、SEM | 晶格、元素、工艺、表征节点 |
| MoE 加速 | token、router logits、expert id | token 节点、expert 节点 |

## 5.2 算子定义 <a href="#section-002" id="section-002"></a>

输入编码算子：

$$
Z_0 = \operatorname{Encode}(\mathcal{D}_{raw};\theta_{enc})
$$

其中：

- $$\mathcal{D}_{raw}$$ 是原始输入；
- $$Z_0$$ 是初始结构化 token / object 集合；
- $$\theta_{enc}$$ 是编码参数。

## 5.3 工程接口 <a href="#section-003" id="section-003"></a>

```text
InputEncoder.encode(raw_input) -> EncodedObjects
```


---

[← 上一页](section-04.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-06.md)

# 17. 总损失函数设计

SGD-Net 的训练目标可写为：

$$
\mathcal{L}=\mathcal{L}_{task}
+\lambda_g\mathcal{L}_{graph}
+\lambda_s\mathcal{L}_{state}
+\lambda_t\mathcal{L}_{tree}
+\lambda_e\mathcal{L}_{posterior}
+\lambda_p\mathcal{L}_{projection}
$$

**本页目录**

- [17.1 任务损失](#section-001)
- [17.2 图约束损失](#section-002)
- [17.3 状态稳定损失](#section-003)
- [17.4 树复杂度正则](#section-004)
- [17.5 后验误差损失](#section-005)
- [17.6 稳定性投影损失](#section-006)

---

## 17.1 任务损失 <a href="#section-001" id="section-001"></a>

分类：

$$
\mathcal{L}_{task}=\operatorname{CE}(\hat{y},y)
$$

回归：

$$
\mathcal{L}_{task}=\|\hat{y}-y\|_2^2
$$

## 17.2 图约束损失 <a href="#section-002" id="section-002"></a>

$$
\mathcal{L}_{graph}=\sum_{(i,j)\notin\mathcal{E}} s_{ij}^2
$$

抑制非法隐式连接。

## 17.3 状态稳定损失 <a href="#section-003" id="section-003"></a>

$$
\mathcal{L}_{state}=\sum_t\max(0,\|h_t\|_2-c)^2
$$

## 17.4 树复杂度正则 <a href="#section-004" id="section-004"></a>

$$
\mathcal{L}_{tree}=\alpha\cdot \#leaves+\beta\cdot depth(T)
$$

避免树无限增长。

## 17.5 后验误差损失 <a href="#section-005" id="section-005"></a>

$$
\mathcal{L}_{posterior}=\sum_i \eta_i
$$

## 17.6 稳定性投影损失 <a href="#section-006" id="section-006"></a>

$$
\mathcal{L}_{projection}=\max(0,\rho(A)-\rho_{max})^2
$$


---

[← 上一页](section-18.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-20.md)

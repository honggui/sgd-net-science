# 25. 复杂度分析

**本页目录**

- [25.1 SSM 复杂度](#section-001)
- [25.2 GNN 复杂度](#section-002)
- [25.3 动态树路由复杂度](#section-003)
- [25.4 拓扑 refinement 成本](#section-004)

---

## 25.1 SSM 复杂度 <a href="#section-001" id="section-001"></a>

对长度 $$L$$、状态维度 $$D_h$$：

$$
O(LD_h)
$$

若每个图节点都有状态流：

$$
O(NLD_h)
$$

## 25.2 GNN 复杂度 <a href="#section-002" id="section-002"></a>

稀疏图消息传递：

$$
O(|E|F)
$$

稠密邻接：

$$
O(N^2F)
$$

因此工程上应尽早迁移到稀疏图。

## 25.3 动态树路由复杂度 <a href="#section-003" id="section-003"></a>

若每棵树深度为 $$d$$：

$$
O(NdF_{cmp})
$$

其中 $$F_{cmp}$$ 为比较特征维度，通常远小于完整 MLP 成本。

## 25.4 拓扑 refinement 成本 <a href="#section-004" id="section-004"></a>

如果每次最多分裂 $$K$$ 个节点，节点数增加 $$2K$$。必须设置：

- 最大节点数；
- 最大树深；
- 每步最大分裂数；
- 分裂冷却时间；
- 剪枝策略。


---

[← 上一页](section-26.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-28.md)

# 19. 最小可实现版本 MVP

一个最小 SGD-Net 不需要一开始实现所有复杂能力。建议 MVP 包含：

1. 稠密邻接矩阵图状态；
2. 简化 GNN：$$D^{-1}AXW$$；
3. 简化 SSM：$$h_t=Ah_{t-1}+Bx_t$$；
4. 每节点硬决策树；
5. 预测与已到达结果配对后的尺度化 L2 误差；
6. 隔离候选上的表示节点细化、完整迁移与窗口间接纳；
7. SVD 或范数数值正则化，以及独立稳定性条件检查；
8. 节点级回归输出。

MVP 前向：

```text
x_gnn = D^{-1} A X W_g
h_new = A_s h + B_s x_gnn
x_tree = TreeRoute(h_new)
graph.x = x_tree
y = head(graph.x)
record_prediction(y, versions)
report = paired_error(prior_prediction_record, matched_observation)
if persistent_error_has_supported_refinement_cause(report):
    enqueue_isolated_refinement_candidate(graph, report)
# 不在本次前向内改变活动拓扑；无配对观测则报告 UNKNOWN。
```


---

[← 上一页](section-20.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-22.md)

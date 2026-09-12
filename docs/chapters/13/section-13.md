# 11. Layer 6：后验误差估计层

**本页目录**

- [11.1 功能](#section-001)
- [11.2 通用误差形式](#section-002)
- [11.3 任务误差](#section-003)
- [11.4 拓扑误差](#section-004)
- [11.5 状态漂移误差](#section-005)
- [11.6 稳定性误差](#section-006)
- [11.7 置信度误差](#section-007)
- [11.8 触发策略](#section-008)
- [11.8.1 Adaptive Budget Controller（运行时预算控制器）](#section-009)
- [11.8.2 QuantizationAwareStateCache 与 StaticMaxGraphMask](#section-010)
- [11.9 工程接口](#section-011)

---

## 11.1 功能 <a href="#section-001" id="section-001"></a>

后验误差估计层决定 SGD-Net 是否要动态改变结构。

它回答的问题是：

> 当前哪些节点、边、子图、路径或局部区域误差太高，需要 refinement、隔离或剪枝？

## 11.2 通用误差形式 <a href="#section-002" id="section-002"></a>

$$
\eta_i=\lambda_1\eta_i^{task}+\lambda_2\eta_i^{topo}+\lambda_3\eta_i^{state}+\lambda_4\eta_i^{stability}+\lambda_5\eta_i^{confidence}
$$

## 11.3 任务误差 <a href="#section-003" id="section-003"></a>

监督任务中：

$$
\eta_i^{task}=\|\hat{y}_i-y_i\|_2
$$

PDE 中：

$$
\eta_i^{task}=\|\mathcal{R}(u_i)\|
$$

其中 $$\mathcal{R}$$ 是 PDE residual operator。

## 11.4 拓扑误差 <a href="#section-004" id="section-004"></a>

若模型信息传播跨越非法边界：

$$
\eta_i^{topo}=\sum_j \mathbb{I}[A_{ij}=0]\cdot s_{ij}
$$

其中 $$s_{ij}$$ 是模型试图建立的隐式关联强度。

## 11.5 状态漂移误差 <a href="#section-005" id="section-005"></a>

$$
\eta_i^{state}=\|h_{i,t}-h_{i,t-1}\|_2
$$

或使用 KL 散度：

$$
\eta_i^{state}=D_{KL}(P_t(y_i)\parallel P_{t-1}(y_i))
$$

## 11.6 稳定性误差 <a href="#section-006" id="section-006"></a>

例如谱半径超限：

$$
\eta^{stability}=\max(0,\rho(W)-\rho_{max})
$$

## 11.7 置信度误差 <a href="#section-007" id="section-007"></a>

在生物结构预测中：

$$
\eta_i^{confidence}=1-\frac{pLDDT_i}{100}
$$

或对高 PAE 区域：

$$
\eta_{ij}^{PAE}=\frac{PAE_{ij}}{PAE_{max}}
$$

## 11.8 触发策略 <a href="#section-008" id="section-008"></a>

```text
if conservation_check_invalid_or_failed:
    reject_candidate_or_use_declared_fallback
elif observation_missing_or_noise_dominated:
    request_observation_or_keep_unknown
elif persistent_residual_high and capacity_or_resolution_cause_supported:
    propose_isolated_refinement_with_migration
elif residual_low and merge_consistency_valid and benefit_expected:
    propose_isolated_merge
else:
    keep
# 所有候选经收益、回归、数值、资源与版本验证后，窗口间自动接纳。
```

残差阈值只是候选生成条件。pLDDT、PAE 是模型置信/误差预测指标，不是独立真实观测；不得把其降低直接当作生长收益或真值改善。

## 11.8.1 Adaptive Budget Controller（运行时预算控制器） <a href="#section-009" id="section-009"></a>

AQAD 的动态步长思想可推广为 SGD-Net 的运行时预算控制器。其输入不是单一 logits 熵，而是多通道不确定性：

$$
U_t = w_H \tilde{H}_t + w_R\|r_t\| + w_C(1-c_t) + w_S s_t
$$

其中 $$\tilde{H}_t$$ 是路径/输出分布的阈值截断熵，$$r_t$$ 是 Harness 残差，$$c_t$$ 是来源或模型置信度，$$s_t$$ 是状态漂移、量化残差或 memory pressure。

运行时可将 $$U_t$$ 映射为：

| 输出预算 | 控制对象 |
|---|---|
| `refinement_depth` | 动态树 split/grow 最大深度。 |
| `rollout_steps` | SSM/世界模型短期 rollout 步数。 |
| `candidate_branch_count` | 局部专家、草稿树或实验假设分支数。 |
| `verification_strength` | Harness、PPU 或显式求解器复核强度。 |
| `state_precision` | SSMState、GraphState、JEPALatent 或 KV cache 的量化位宽与 fallback precision。 |
| `fallback_threshold` | 高风险样本切回保守路径的阈值。 |

该控制器只用于性能、预算和安全策略调度；不能把熵或残差阈值写成正确性证明。正确性仍来自 Harness、StabilityProjector、Rollback 与 fallback。

## 11.8.2 QuantizationAwareStateCache 与 StaticMaxGraphMask <a href="#section-010" id="section-010"></a>

长程 Agent、Retrospection replay 和动态图 batch 会产生大量状态缓存。可引入：

```text
QuantizationAwareStateCache {
  state_kind: SSMState | GraphState | JEPALatent | RetrospectionTrace | KVCache
  bit_width
  scale_granularity
  residual_statistics
  quality_guard
  fallback_precision
}
```

同时，对动态图 batch 采用 AQAD 风格的静态最大形状策略：

```text
physical: N_max / E_max / tree_depth_max
logical:  N_t / E_t / tree_depth_t
mask:     屏蔽无效节点、边、树分支
release:  iteration boundary 释放未用 slot
```

必须记录 `wasted_compute_ratio`、`early_release_slots`、`masked_branch_count` 与质量退化指标；低比特状态缓存不能默认视为无损。

## 11.9 工程接口 <a href="#section-011" id="section-011"></a>

```text
PosteriorErrorEstimator.estimate(prediction_record, matched_observation=None, constraints=None) -> ErrorReport
```


---

[← 上一页](section-12.md) · [全书目录](../../../SUMMARY.md) · [下一页 →](section-14.md)

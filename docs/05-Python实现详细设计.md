# SGD-Net Python 实现详细设计

> 公开研究版 · 2026-09-12。保留模型方法、公式和组件设计；本文描述研究方案，未据此声称已有训练结果、完整实现或芯片性能。章节编号保留原研究索引，缺号表示未随包发布的独立规划内容。


> 文档类型：详细设计\
> 版本：v0.1\
> 日期：2026-06-07

**本页目录**

- [1. 设计原则](#section-001)
- [2. 推荐目录结构](#section-002)
- [3. 核心数据结构](#section-003)
- [4. 核心模块详细设计](#section-008)
- [5. 关键流程设计](#section-038)
- [6. 异常处理](#section-041)
- [7. 测试设计](#section-042)
- [8. 性能设计](#section-049)
- [9. 可观测性设计](#section-053)
- [10. 配置项详细建议](#section-054)
- [11. 后续 PyTorch 迁移设计](#section-055)
- [12. 验收清单](#section-056)
- [13. 科学大模型扩展接口设计](#section-057)
- [14. 科研 Agent 扩展接口设计](#section-062)
- [16. 双闭环控制与认知接口设计](#section-066)
- [17. SGD-Ecosystem 接口设计](#section-072)
- [19. 反省复盘与动态自进化接口设计](#section-078)
- [20. JEPA / JSBO 世界模型桥梁接口设计](#section-087)
- [21. 应用先行、后端抽象与模拟器影子执行接口设计](#section-097)

---

## 1. 设计原则 <a href="#section-001" id="section-001"></a>

1. **先正确，后高性能**：第一阶段以 NumPy 原型验证算法闭环。
2. **核心策略可插拔**：误差估计、分裂策略、稳定投影都应可替换。
3. **动态图可追踪**：每次节点分裂、剪枝、隔离都必须记录谱系。
4. **可迁移到深度学习框架**：数据结构设计应便于迁移至 PyTorch / PyG。
5. **严格测试关键不变量**：尤其是邻接矩阵维度、对称性、旧节点隔离和特征范数。

## 2. 推荐目录结构 <a href="#section-002" id="section-002"></a>

```text
sgd-net/
  pyproject.toml
  README.md
  src/
    sgd_net/
      __init__.py
      core/
        graph_state.py
        dynamic_tree.py
        error_estimator.py
        topology_refiner.py
        stability_projector.py
      layers/
        gnn.py
        ssm.py
        tree_routing.py
      experiments/
        synthetic_demo.py
        pde_demo.py
        safe_llm_demo.py
      utils/
        logging.py
        random.py
        visualization.py
  tests/
    test_dynamic_tree.py
    test_graph_state.py
    test_topology_refiner.py
    test_stability_projector.py
    test_end_to_end.py
```

## 3. 核心数据结构 <a href="#section-003" id="section-003"></a>

### 3.1 `GraphState` <a href="#section-004" id="section-004"></a>

职责：保存当前动态图、节点特征、树结构和谱系。

建议字段：

```text
GraphState
  x: np.ndarray                    # shape: [num_nodes, feature_dim]
  adjacency: np.ndarray            # shape: [num_nodes, num_nodes]
  node_trees: list[DecisionTreeNode]
  active_mask: np.ndarray          # shape: [num_nodes]
  lineage: dict[int, list[int]]    # parent -> children
  metadata: dict
  layer_states: dict              # 各层 h/z、向量通道及时间游标
  version_bundle: dict            # 权重、图、状态布局、校准、可执行包
  entity_ids: list                # 稳定实体 ID，区别表示子节点 ID
```

关键不变量：

- `x.shape[0] == adjacency.shape[0] == adjacency.shape[1]`；
- `len(node_trees) == x.shape[0]`；
- 无向图场景下 `adjacency == adjacency.T`；
- 被隔离节点对应行列可为零，但节点记录保留；
- `active_mask` 与隔离/剪枝状态一致。

### 3.2 `DecisionTreeNode` <a href="#section-005" id="section-005"></a>

职责：表示 GNN 节点内部的一棵可分裂树。

建议字段：

```text
DecisionTreeNode
  is_leaf: bool
  depth: int
  max_depth: int
  split_feature_idx: int
  split_threshold: float
  left_child: DecisionTreeNode | None
  right_child: DecisionTreeNode | None
  local_bias: np.ndarray
  status: str
```

方法：

```text
route(feature) -> routed_feature, leaf_id
can_split() -> bool
split(feature_dim, rng) -> bool
prune(reason) -> None
```

分裂逻辑：

1. 若节点不是叶子，禁止重复分裂；
2. 若深度达到 `max_depth`，禁止分裂；
3. 创建左右子节点；
4. 左右子节点继承父节点局部偏置并加小扰动；
5. 父节点状态改为非叶子。

### 3.3 `ErrorReport` <a href="#section-006" id="section-006"></a>

职责：封装后验误差结果。

建议字段：

```text
ErrorReport
  residuals: np.ndarray
  candidate_nodes: np.ndarray        # 建议，不是已批准的拓扑修改
  threshold: float
  reasons: dict[int, str]
  metrics: dict
  prediction_id: str
  observed_at: float | None
  evidence_status: str            # MATCHED / MISSING / INVALID
  attribution: dict              # 观测、数值、模型及资源原因
```

### 3.4 `RefineResult` <a href="#section-007" id="section-007"></a>

职责：封装拓扑重构结果。

建议字段：

```text
RefineResult
  state: GraphState
  replaced_nodes: list[int]
  new_nodes: list[int]
  isolated_nodes: list[int]
  events: list[dict]
```

## 4. 核心模块详细设计 <a href="#section-008" id="section-008"></a>

### 4.1 GNN Layer <a href="#section-009" id="section-009"></a>

#### 功能 <a href="#section-010" id="section-010"></a>

执行简化图消息传递。

#### 输入 <a href="#section-011" id="section-011"></a>

- `x`: 节点特征矩阵；
- `adjacency`: 邻接矩阵。

#### 输出 <a href="#section-012" id="section-012"></a>

- `out`: 聚合后节点特征。

#### 初版算法 <a href="#section-013" id="section-013"></a>

```text
D_inv = diag(1 / (sum(A, axis=1) + eps))
A_hat = D_inv @ A
out = A_hat @ X
```

#### 注意事项 <a href="#section-014" id="section-014"></a>

- 对隔离节点，度为 0，需要加 `eps`；
- 后续可替换为带权消息函数；
- 若迁移 PyG，改用 `edge_index` 聚合。

### 4.2 SSM Layer <a href="#section-015" id="section-015"></a>

#### 功能 <a href="#section-016" id="section-016"></a>

按真实事件时间执行显式状态递推；隐式求解由 IM01—IM06 独立组件承担。

#### 初版设计 <a href="#section-017" id="section-017"></a>

NumPy 阶段可以先做轻量线性状态更新：

```text
h_next = A_bar @ h + B_bar @ x
```

静态任务可关闭时间状态；若只做线性特征变换，应标为普通线性层，不能据此验证时序记忆。

#### 稳定参数化 <a href="#section-018" id="section-018"></a>

- 连续时间生成矩阵采用已验证的 Hurwitz 参数化（特征值实部严格为负）；离散 `A_bar` 的稳定条件另行检查，不能混用；
- 在候选版本上执行适用的范数约束；逐步谱半径小于 1 不足以证明任意切换稳定；
- 保留 `A_bar` 的谱半径日志。

### 4.3 Tree Routing Layer <a href="#section-019" id="section-019"></a>

#### 功能 <a href="#section-020" id="section-020"></a>

对每个节点特征执行动态树路由。

#### 初版算法 <a href="#section-021" id="section-021"></a>

```text
for i in range(num_nodes):
    out[i], leaf_id = node_trees[i].route(x[i])
```

#### 可微版本扩展 <a href="#section-022" id="section-022"></a>

- 硬阈值改为 sigmoid 门控；
- 使用 temperature 控制软硬程度；
- 使用 Gumbel-Softmax 或 straight-through estimator。

### 4.4 Posterior Error Estimator <a href="#section-023" id="section-023"></a>

#### 功能 <a href="#section-024" id="section-024"></a>

计算已配对预测误差，给出候选节点及归因，不直接提交分裂。

#### NumPy 初版 <a href="#section-025" id="section-025"></a>

```text
if matched_observation is None:
    return ErrorReport(status="UNKNOWN", candidate_nodes=[])
residuals = scaled_norm(prediction_record.value - matched_observation.value)
candidate_nodes = diagnose_persistent_error(residuals, noise, missing_data, budget)
```

#### PDE 扩展 <a href="#section-026" id="section-026"></a>

- 使用有限差分近似 PDE 残差；
- 使用边界条件误差；
- 使用局部梯度或散度变化。

#### Safe LLM 扩展 <a href="#section-027" id="section-027"></a>

- 使用事实图谱路径违背；
- 使用 KL 散度；
- 使用检索置信度下降；
- 使用逻辑一致性检查器得分。

### 4.5 Topology Refiner <a href="#section-028" id="section-028"></a>

#### 功能 <a href="#section-029" id="section-029"></a>

按候选动作在隔离副本上构建新图，完成迁移和验证后才发布。

#### 输入 <a href="#section-030" id="section-030"></a>

- 当前 `GraphState`；
- `ErrorReport`；
- 分裂策略配置。

#### 输出 <a href="#section-031" id="section-031"></a>

- 更新后的 `GraphState`；
- `RefineResult`。

#### 详细流程 <a href="#section-032" id="section-032"></a>

1. 固定活动版本快照，先确认树模式、候选预算和误差归因。
2. 在候选副本建立子节点或专家；实体 ID 与表示节点 ID 分开。
3. 重建候选边、权重、掩码和谱系，按任务约束定义限制/延拓算子。
4. 迁移所有层 h/z、向量、边状态及时间游标；新增参数映射优化器统计，无法可靠迁移时显式重置并计入验证。
5. 生成匹配的布局/可执行包，检查迁移前后输出一致性、后续预测收益、旧任务回归与资源峰值。
6. 验证通过后窗口间发布完整版本包；失败保留原活动版本，待旧任务结束后回收旧包。

#### 邻接矩阵更新规则 <a href="#section-033" id="section-033"></a>

下面仅是无向表示图的玩具细化规则，不适用于实体图、守恒网格或有向因果图；实际任务需定义权重/体积限制与延拓、守恒以及输出一致性。所有操作只作用于候选副本。若旧节点为 `i`，新节点为 `l` 和 `r`，旧邻居集合为 $$N(i)$$：

- `A[l, r] = A[r, l] = 1`；
- 对任意 `n in N(i)`：
  - `A[l, n] = A[n, l] = 1`；
  - `A[r, n] = A[n, r] = 1`；
- `A[i, :] = 0`；
- `A[:, i] = 0`。

### 4.6 Stability Projector <a href="#section-034" id="section-034"></a>

#### 功能 <a href="#section-035" id="section-035"></a>

限制特征或权重爆炸。

#### SVD 裁剪 <a href="#section-036" id="section-036"></a>

```text
u, s, vh = svd(X, full_matrices=False)
s = clip(s, 0, max_singular_value)
X_projected = (u * s) @ vh
```

#### 其他策略 <a href="#section-037" id="section-037"></a>

- `NormClipProjector`：按行裁剪节点特征范数；
- `SpectralRadiusProjector`：裁剪状态转移矩阵谱半径；
- `LyapunovProjector`：按指定域、输入和步长检验候选 V 的下降或耗散不等式，失败则缩步或拒绝；外力输入下真实能量可上升；
- `NoOpProjector`：用于消融实验。

## 5. 关键流程设计 <a href="#section-038" id="section-038"></a>

### 5.1 事件处理与后台演化 <a href="#section-039" id="section-039"></a>

```text
function on_event(active_bundle, event):
    frozen = acquire_version(active_bundle)
    report = match_prior_prediction(event, immutable_prediction_log)
    state = update_observed_state_once(frozen, event)  # 真实时间只推进一次
    prediction = graph_ssm_tree_predict(frozen, state)
    append_prediction(prediction, event.available_inputs, frozen.versions)
    enqueue_candidate_if_justified(report, state, budget)
    return prediction, state, report

function adapt_at_window(snapshot, paired_history):
    candidate = propose_structure_or_parameter_update(snapshot, paired_history)
    migrate_all_states_and_optimizer(candidate)
    train_or_refine_with_bounded_budget(candidate)  # 单次前向/反向内固定图
    if validate_gain_regression_numerics_and_layout(candidate):
        publish_at_window_boundary(candidate)
    else:
        discard_candidate()
```

### 5.2 离线训练与在线观测分离 <a href="#section-040" id="section-040"></a>

离线监督训练在完成预测后用标签计算损失；目标标签不能进入对应的预测输入或预算选择。在线部署只使用已到达观测，迟到结果保留原预测版本配对。活动模型不在前向中被后台候选改写。隐式层在固定图和参数下求解并反传，失败状态与梯度有效性一起返回，见 45 号文档。

```text
for batch in offline_training_data:
    prediction = model(batch.available_inputs)
    loss = objective(prediction, batch.labels)
    backward_and_update_in_fixed_graph(loss)
    schedule_structure_candidate_between_steps()

for event in deployment_stream:
    prediction, state, report = on_event(active_bundle, event)
    log_prediction_and_report(prediction, report)
    run_bounded_background_adaptation_if_window_ready()
```

上述为接口设计，未提供可运行实现；候选发布与真实动作执行权限分别检查。

## 6. 异常处理 <a href="#section-041" id="section-041"></a>

| 异常 | 处理 |
|---|---|
| 邻接矩阵维度不一致 | 抛出 `GraphShapeError` |
| 分裂超过最大深度 | 跳过并记录事件 |
| 节点数量超过预算 | 禁止继续分裂，进入剪枝策略 |
| SVD 不收敛 | 报告该检查失败；范数裁剪只作数值降级，不继承原稳定性结论 |
| 残差 NaN/Inf | 触发隔离并回滚上一状态 |

## 7. 测试设计 <a href="#section-042" id="section-042"></a>

### 7.1 单元测试 <a href="#section-043" id="section-043"></a>

#### 动态树 <a href="#section-044" id="section-044"></a>

- 叶节点路由返回 `feature + local_bias`；
- 未达到最大深度时可分裂；
- 达到最大深度时不可分裂；
- 分裂后左右子节点存在。

#### 拓扑重构 <a href="#section-045" id="section-045"></a>

- 分裂后节点数量增加 2；
- 特征矩阵行数与邻接矩阵维度一致；
- 旧节点行列被清零；
- 新节点继承旧邻居；
- 新节点之间存在连边。

#### 稳定投影 <a href="#section-046" id="section-046"></a>

- 投影后奇异值不超过阈值；
- 投影不改变矩阵形状；
- NaN 输入能被检测。

### 7.2 集成测试 <a href="#section-047" id="section-047"></a>

- 构造 5 节点环图；
- 人为制造第 2 个节点残差；
- 执行一步 refine；
- 验证节点数从 5 增至 7；
- 验证旧节点被隔离；
- 验证残差事件被记录。

### 7.3 消融测试 <a href="#section-048" id="section-048"></a>

- 关闭稳定投影；
- 关闭旧节点隔离；
- 关闭动态树，仅保留 GNN；
- 不同分裂阈值；
- 不同最大树深度。

## 8. 性能设计 <a href="#section-049" id="section-049"></a>

### 8.1 NumPy 阶段 <a href="#section-050" id="section-050"></a>

适合节点数小于数千的实验。邻接矩阵为 $$N\times N$$，空间复杂度 $$O(N^2)$$。

### 8.2 PyG 阶段 <a href="#section-051" id="section-051"></a>

使用稀疏 `edge_index` 后，空间复杂度约为 $$O(|E|)$$。适合更大图和批处理。

### 8.3 动态增长控制 <a href="#section-052" id="section-052"></a>

必须配置：

- 最大节点数；
- 最大树深；
- 每步最大分裂节点数；
- 最小分裂间隔；
- 剪枝策略。

## 9. 可观测性设计 <a href="#section-053" id="section-053"></a>

每个 step 输出：

```text
step_id
num_nodes
num_edges
max_residual
mean_residual
split_nodes
new_nodes
isolated_nodes
feature_norm
max_singular_value
elapsed_ms
```

## 10. 配置项详细建议 <a href="#section-054" id="section-054"></a>

```text
GraphConfig:
  initial_nodes: int
  feature_dim: int
  graph_type: ring | grid | knn | knowledge

TreeConfig:
  max_depth: int
  split_temperature: float
  split_noise_std: float

RefineConfig:
  error_threshold: float
  max_nodes: int
  max_splits_per_step: int
  isolate_old_node: bool

StabilityConfig:
  projector: svd | norm | lyapunov | none
  max_singular_value: float
  max_feature_norm: float

RuntimeConfig:
  seed: int
  max_steps: int
  log_dir: str
```

## 11. 后续 PyTorch 迁移设计 <a href="#section-055" id="section-055"></a>

| NumPy 原型 | PyTorch 版本 |
|---|---|
| `np.ndarray` | `torch.Tensor` |
| 硬分裂 | 软门控 / Gumbel-Softmax |
| 手写邻接矩阵乘法 | `torch_sparse` / PyG MessagePassing |
| SVD 裁剪 | `torch.linalg.svd` 或谱归一化 |
| print 日志 | TensorBoard / structured logging |

## 12. 验收清单 <a href="#section-056" id="section-056"></a>

- [ ] `GraphState` 不变量测试通过；
- [ ] 动态树分裂测试通过；
- [ ] 拓扑重构测试通过；
- [ ] 稳定投影测试通过；
- [ ] 端到端示例可运行；
- [ ] 输出日志包含节点增长和残差变化；
- [ ] 文档中示例与代码保持一致。

## 13. 科学大模型扩展接口设计 <a href="#section-057" id="section-057"></a>

本节描述二期扩展类和接口，供后续实现 `SGD-Fold`、`SGD-Design`、`SGD-ESM` 与 `SGD-Scientist` 使用。

### 13.1 `BioStructureGraphAdapter` <a href="#section-058" id="section-058"></a>

职责：将生物分子结构预测输出转换为 SGD-Net 图状态。

建议接口：

```text
BioStructureGraphAdapter
  parse_structure(path_or_obj) -> BioStructure
  build_graph(structure) -> GraphState
  attach_confidence(graph, confidence_payload) -> GraphState
```

节点类型：

- protein residue；
- nucleotide；
- ligand atom；
- ion；
- modified residue atom；
- chain/interface pseudo node。

边类型：

- covalent bond；
- spatial contact；
- hydrogen bond；
- salt bridge；
- ligand-pocket edge；
- chain-interface edge。

### 13.2 `BioPosteriorErrorEstimator` <a href="#section-059" id="section-059"></a>

职责：计算生物结构后验误差。

建议字段：

```text
BioErrorReport
  plddt_low_regions
  pae_high_pairs
  clash_pairs
  chirality_violations
  pocket_risk
  interface_risk
  suggested_refinement_regions
```

误差来源包括 pLDDT、PAE、pTM、ipTM、几何违背、原子 clash、手性风险和配体口袋异常。

### 13.3 `DiffusionSSMAdapter` <a href="#section-060" id="section-060"></a>

职责：将扩散模型去噪步骤抽象为 SSM 状态轨迹。

建议接口：

```text
DiffusionSSMAdapter
  initialize_state(noise, condition) -> DiffusionState
  step(state, t) -> DiffusionState
  estimate_path_error(state) -> ErrorReport
  rollback(state, checkpoint_id) -> DiffusionState
```

适用场景：RFdiffusion、分子扩散生成、结构 refinement diffusion。

### 13.4 `ProteinDesignConstraintBreaker` <a href="#section-061" id="section-061"></a>

职责：在蛋白生成过程中检测不可行路径并执行断路。

触发条件：

- motif RMSD 超阈值；
- active site 几何违背；
- binding interface 崩塌；
- clash 或能量代理指标异常；
- 可合成性或表达性 proxy 过低。

动作：

- `continue`；
- `local_refine`；
- `rollback`；
- `prune_candidate`；
- `increase_sampling_budget`。

## 14. 科研 Agent 扩展接口设计 <a href="#section-062" id="section-062"></a>

### 14.1 `ScientificAgentStateTracker` <a href="#section-063" id="section-063"></a>

职责：维护科研任务长期状态。

```text
ScientificAgentStateTracker
  update_with_literature(paper) -> None
  update_with_simulation(result) -> None
  update_with_experiment(record) -> None
  summarize_state() -> AgentState
```

### 14.2 `MaterialExperimentGraphAdapter` <a href="#section-064" id="section-064"></a>

职责：将材料、工艺、仿真、表征和实验结果映射为图。

节点包括：

- material；
- composition；
- crystal structure；
- process condition；
- simulation result；
- characterization result；
- hypothesis；
- experiment action。

### 14.3 `AdaptiveExperimentTree` <a href="#section-065" id="section-065"></a>

职责：对候选材料和实验路径执行自适应分裂、剪枝和排序。

分裂依据：

- 仿真-实验偏差；
- 不确定性；
- 成本；
- 安全约束；
- 候选接近目标性能阈值。

## 16. 双闭环控制与认知接口设计 <a href="#section-066" id="section-066"></a>

本节根据 原始讨论材料（本包不附） 增加前馈预测、后验反馈、来源监控和知识分类相关详细设计。

### 16.1 `FeedforwardPredictor` <a href="#section-067" id="section-067"></a>

职责：提前预测未来若干步的高风险区域，为动态树预分裂、GNN 稀疏路由和硬件预热提供计划。

```text
FeedforwardPredictor
  predict(state: GraphState, horizon: int) -> PredictedTrajectory
  estimate_future_risk(traj: PredictedTrajectory) -> PredictedRiskReport
  plan_prewarm(report: PredictedRiskReport) -> ControlPlan
```

输出字段：

```text
ControlPlan
  nodes_to_prewarm: list[int]
  routes_to_cache: list[tuple[int, int]]
  trees_to_presplit: list[int]
  stability_level: conservative | normal | aggressive
  confidence: float
```

### 16.2 `PosteriorCorrector` <a href="#section-068" id="section-068"></a>

职责：以后验误差为硬保底控制信号，对前馈计划或主干输出进行校正。

```text
PosteriorCorrector
  correct(state: GraphState, report: ErrorReport) -> CorrectionPlan
```

可选动作：

- `keep`；
- `split`；
- `prune`；
- `isolate`；
- `rollback`；
- `increase_projection_strength`。

### 16.3 `ControlArbiter` <a href="#section-069" id="section-069"></a>

职责：决定前馈预测和后验反馈谁优先。

```text
ControlArbiter
  decide(pred_plan: ControlPlan, post_report: ErrorReport, source_report: SourceReport) -> ControlAction
```

推荐规则：

```text
if post_report.has_critical_violation:
    feedback control takes over
elif source_report.has_untrusted_source:
    quarantine or shadow-route
elif pred_plan.confidence > threshold and post_report.residual_low:
    use predictive prewarm plan
else:
    conservative refinement
```

### 16.4 `SourceMonitor` <a href="#section-070" id="section-070"></a>

职责：为节点、边、实验记录、仿真结果和推理轨迹附加来源标签。

```text
SourceTag
  source_type: real_experiment | simulation | prediction | literature | third_party | dream | synthetic
  confidence: float
  verification_status: verified | unverified | contradicted | promoted | quarantined
  provenance: list[EvidenceRef]
  timestamp: datetime
```

### 16.5 `KnowledgeOntologyManager` <a href="#section-071" id="section-071"></a>

职责：管理经验、教训和待核实知识。

```text
KnowledgeOntologyManager
  classify(evidence: EvidenceRecord) -> KnowledgeStatus
  promote(shadow_item) -> StableExperience
  demote(shadow_item) -> NegativeLesson
  quarantine(item, reason) -> None
```

知识状态：

- `local_positive_experience`：在特定条件范围内正确；
- `local_negative_lesson`：在特定条件范围内错误或高风险；
- `unverified_shadow`：知道但尚未验证；
- `contradicted`：被真实证据或高可信仿真证伪；
- `promoted`：经多轮验证后晋升。

## 17. SGD-Ecosystem 接口设计 <a href="#section-072" id="section-072"></a>

本节增加物理肉身、数字孪生和双向流形折叠接口。

### 17.1 `LabToken` <a href="#section-073" id="section-073"></a>

职责：统一表示实验动作、传感器观测、样品状态和安全事件。

```text
LabToken
  token_type: action | observation | sample | instrument | safety_event
  modality: synthesis | xrd | sem | tem | raman | eis | simulation
  payload: dict
  source_tag: SourceTag
  sample_id: str
  timestamp: datetime
  uncertainty: float
```

### 17.2 `WorldModel` <a href="#section-074" id="section-074"></a>

职责：差分数字孪生状态推进。

```text
WorldModel
  initialize(problem_spec) -> TwinState
  rollout(state: TwinState, action_plan, noise_config, num_samples) -> list[Trajectory]
  score_robustness(trajectories) -> RobustnessReport
```

### 17.3 `DualTrackGraph` <a href="#section-075" id="section-075"></a>

职责：将模拟节点和真实实验节点折叠到统一图状态。

```text
DualTrackGraph
  sim_nodes
  real_nodes
  bridge_edges
  align(sim_record, real_record) -> AlignmentResult
```

### 17.4 `CrossDomainErrorEstimator` <a href="#section-076" id="section-076"></a>

职责：计算虚实误差，例如仿真 XRD 与真实 XRD 差异、仿真性能与实验性能差异。

```text
CrossDomainErrorEstimator
  estimate(sim_observation, real_observation) -> CrossDomainErrorReport
```

### 17.5 `PromotionEngine` <a href="#section-077" id="section-077"></a>

职责：根据真实实验与数字孪生验证结果决定假说状态转换。

```text
PromotionEngine
  evaluate(shadow_hypothesis, evidence_records) -> PromotionDecision
  apply(decision, graph_state) -> GraphState
```

## 19. 反省复盘与动态自进化接口设计 <a href="#section-078" id="section-078"></a>

本节根据 原始讨论材料（本包不附） 增加 SGD-Retrospection 的核心接口。

### 19.1 `ReflectionEvent` <a href="#section-079" id="section-079"></a>

职责：记录一次值得复盘的惊奇事件或高价值常用处理。

```text
ReflectionEvent
  event_id: str
  graph_state_ref: str
  action: str
  prediction_ref: str
  observation_ref: str
  posterior_error: float
  stability_delta: float
  source_report: dict
  compute_cost: dict
  timestamp: datetime
```

### 19.2 `RetrospectionLayer` <a href="#section-080" id="section-080"></a>

职责：从后验误差和多级记忆中触发单步复盘、深度反省、动态整理和主动遗忘。

```text
RetrospectionLayer
  should_reflect(event: ReflectionEvent, state: GraphState) -> bool
  collect(event: ReflectionEvent) -> None
  run_offline_cycle(config: ReplayConfig) -> RetrospectionReport
  generate_actions(report: RetrospectionReport) -> list[RetrospectionAction]
```

### 19.3 `HotnessClusterer` <a href="#section-081" id="section-081"></a>

职责：识别高频、高收益、可信且稳定的常用处理路径。

```text
HotnessClusterer
  score(path_trace) -> HotnessScore
  cluster(events: list[ReflectionEvent]) -> list[HotPathCluster]
  select_candidates(clusters, threshold) -> list[FastPathCandidate]
```

推荐热度分数：

```text
hotness = frequency * expected_gain * trust * stability / (cost + eps)
```

### 19.4 `RetrospectionCompiler` <a href="#section-082" id="section-082"></a>

职责：将高成本显式处理整理为快速推理候选。

```text
RetrospectionCompiler
  extract_causal_core(cluster: HotPathCluster) -> CausalCore
  compile_fast_path(core: CausalCore) -> FastPathCandidate
  validate(candidate: FastPathCandidate, constraints) -> ValidationReport
```

### 19.5 `FastPathRegistry` <a href="#section-083" id="section-083"></a>

职责：管理已固化的快速推理链路。

```text
FastPathRegistry
  register(candidate: FastPathCandidate) -> str
  lookup(state: GraphState) -> FastPath | None
  update_score(path_id, feedback) -> None
  quarantine(path_id, reason) -> None
  prune(path_id, reason) -> None
```

### 19.6 `CounterfactualReplayGenerator` <a href="#section-084" id="section-084"></a>

职责：在离线阶段生成反事实扰动和极端工况样本。

```text
CounterfactualReplayGenerator
  perturb(event: ReflectionEvent, perturbation_spec) -> CounterfactualTrace
  batch_generate(events, policy) -> list[CounterfactualTrace]
  filter_physical(traces, constraints) -> list[CounterfactualTrace]
```

### 19.7 `DecayPruner` <a href="#section-085" id="section-085"></a>

职责：执行主动遗忘、路径衰减、压缩、归档和剪枝。

```text
DecayPruner
  score_memory_item(item) -> DecayScore
  make_forget_plan(items, budget) -> list[ForgetPlan]
  apply(plan: ForgetPlan) -> ForgetResult
```

### 19.8 安全约束 <a href="#section-086" id="section-086"></a>

反省复盘不能直接污染主干。所有快速路径晋升必须满足：

```text
validation_error < threshold
source_confidence > threshold
stability_check == pass
conflict_check == pass
audit_record exists
```

## 20. JEPA / JSBO 世界模型桥梁接口设计 <a href="#section-087" id="section-087"></a>

本节根据 原始讨论材料（本包不附） 增加 `sgd_net.jepa_bridge` 的核心接口。该模块用于接入 I-JEPA、V-JEPA 或其他自监督世界模型 latent，并将其转换为 SGD-Net 的结构保持状态。

### 20.1 `JEPALatentTensor` <a href="#section-088" id="section-088"></a>

职责：封装外部 JEPA 模型输出的隐空间表征及其元数据。

```text
JEPALatentTensor
  data: Tensor
  source_model: str
  modality: image | video | sensor | graph | text | multimodal
  mask_spec: dict
  positional_meta: dict
  action_meta: dict | None
  uncertainty: Tensor | None
  source_tag: SourceTag
```

### 20.2 `JEPAAdapter` <a href="#section-089" id="section-089"></a>

职责：将外部 JEPA encoder / predictor 封装为统一接口。

```text
JEPAAdapter
  encode_context(raw_input, mask_spec) -> JEPALatentTensor
  encode_target(raw_target, mask_spec) -> JEPALatentTensor
  predict_latent(context: JEPALatentTensor, action=None) -> JEPALatentTensor
```

适配器不应假设所有 JEPA 都是图像模型；实现上应允许 video、trajectory、sensor、graph 和 multimodal latent。

### 20.3 `JSBOBridgeOperator` <a href="#section-090" id="section-090"></a>

职责：实现 $$\Psi_{bridge}:\mathcal{H}_{J}\rightarrow\mathcal{M}_{SGD}$$。

```text
JSBOBridgeOperator
  project(latent: JEPALatentTensor, target_space: str) -> BridgeState
  to_graph_state(bridge: BridgeState) -> GraphState
  to_physical_state(bridge: BridgeState) -> PhysicalState
  report_constraints(bridge: BridgeState) -> BridgeReport
```

### 20.4 `MetricPullbackProjector` <a href="#section-091" id="section-091"></a>

职责：估计目标结构空间度量，并将其拉回到 JEPA 隐空间。

```text
MetricPullbackProjector
  estimate_metric(bridge_state: BridgeState) -> MetricTensor
  compute_jacobian(latent: JEPALatentTensor) -> Jacobian
  pullback(jacobian: Jacobian, metric: MetricTensor) -> LatentMetric
```

核心约束：

```text
G_latent = J_bridge.T @ M_struct @ J_bridge
```

### 20.5 `KoopmanProjector` <a href="#section-092" id="section-092"></a>

职责：约束 JEPA latent predictor 与结构空间动力学一致。

```text
KoopmanProjector
  estimate_operator(sequence: list[BridgeState]) -> KoopmanOperator
  consistency_loss(current, predicted, operator) -> float
  spectral_report(operator, mode: conservative | dissipative | controlled) -> SpectralReport
```

谱约束不能一刀切：保守系统可要求 $$|\lambda|\approx1$$，耗散系统应要求 $$|\lambda|\le1$$，受控系统则应结合 Lyapunov 上界判断。

### 20.6 `StructurePreservingProjector` <a href="#section-093" id="section-093"></a>

职责：根据场景执行辛结构、Lyapunov、能量或 PDE residual 约束。

```text
StructurePreservingProjector
  symplectic_residual(bridge: BridgeState) -> float
  lyapunov_residual(bridge: BridgeState) -> float
  pde_residual(bridge: BridgeState, boundary) -> float
  project(bridge: BridgeState, constraints) -> BridgeState
```

### 20.7 `BridgeCostEvaluator` <a href="#section-094" id="section-094"></a>

职责：汇总 JSBO 训练或校准损失。

```text
BridgeCostEvaluator
  prediction_loss(pred_latent, target_latent) -> float
  bridge_consistency_loss(pred_bridge, target_bridge) -> float
  physical_cost(bridge, observation) -> float
  collapse_score(latent_batch) -> CollapseReport
  total_loss(parts: dict, weights: dict) -> float
```

### 20.8 `JSBOCompiler` <a href="#section-095" id="section-095"></a>

职责：将桥梁算子 lowering 到 SGD-IR 或 XPU workload。

```text
JSBOCompiler
  trace(adapter: JEPAAdapter, sample_input) -> BridgeGraph
  lower_to_sgd_ir(graph: BridgeGraph) -> SGDIREntry
  profile(entry: SGDIREntry) -> WorkloadReport
```

### 20.9 `NarrativeDreamAdapter` 可选接口 <a href="#section-096" id="section-096"></a>

职责：在非工业创作模式中，把物理约束替换为叙事一致性约束。

```text
NarrativeDreamAdapter
  build_story_graph(corpus_or_outline) -> GraphState
  perturb_plot(graph, dream_noise) -> GraphState
  score_narrative(graph) -> NarrativeScore
  consolidate_draft(graph, feedback) -> StoryPlan
```

该接口不得进入工业安全控制主干，只能作为独立实验子系统。

## 21. 应用先行、后端抽象与模拟器影子执行接口设计 <a href="#section-097" id="section-097"></a>

本节根据 `SGD-Net应用先行与模拟器闭环/` 增加 runtime、backend、trace 和 simulator shadow 的核心接口。目标是让同一 SGD-Net 应用可以先跑在 CPU/GPU/NPU/TPU 等第三方后端上，再用 SGD-TPU 模拟器/CModel 重放同一 trace，定位硬件机会点和瓶颈。

### 21.1 `Backend` 统一接口 <a href="#section-098" id="section-098"></a>

职责：抽象 CPU、vendor accelerator、SGD-TPU simulator 和 hybrid backend。

```text
Backend
  name: str
  capabilities() -> BackendCapabilities
  compile(op_graph) -> BackendExecutable
  run(executable, inputs, trace_policy) -> BackendRunResult
  profile() -> BackendProfile
  replay(trace: WorkloadTrace) -> ReplayResult
  fallback(reason: str) -> Backend
```

### 21.2 `BackendCapabilities` <a href="#section-099" id="section-099"></a>

职责：描述后端支持哪些 op、dtype、layout、动态 shape 和 profiling 能力。

```text
BackendCapabilities
  supported_ops: list[str]
  supported_dtypes: list[str]
  preferred_layouts: list[str]
  dynamic_shape_support: bool
  sparse_support_level: none | basic | advanced
  control_flow_support_level: none | host | device
  profiling_support_level: none | coarse | detailed
```

### 21.3 `PlacementPlan` <a href="#section-100" id="section-100"></a>

职责：记录 runtime 为什么把某个 op 放到某个后端。

```text
PlacementPlan
  op_id: str
  primary_backend: str
  shadow_backend: str | None
  fallback_backend: str
  trace_policy: dict
  diff_policy: dict
  safety_policy: dict
  reason: str
```

### 21.4 `WorkloadTrace` <a href="#section-101" id="section-101"></a>

职责：把真实应用运行转换为可 replay、可 diff、可建模的 trace。

```text
WorkloadTrace
  trace_id: str
  app_name: str
  run_config: dict
  input_summary: dict
  quality_metrics: dict
  ops: list[OpTrace]
  backend_events: list[BackendEvent]
  memory_events: list[MemoryEvent]
  control_events: list[ControlEvent]
  artifacts: dict
```

### 21.5 `ShadowExecutor` <a href="#section-102" id="section-102"></a>

职责：vendor backend 主执行，SGD-TPU simulator backend 旁路执行同一 trace。

```text
ShadowExecutor
  run_primary(plan: PlacementPlan, inputs) -> BackendRunResult
  capture_trace(primary_result) -> WorkloadTrace
  run_shadow(trace: WorkloadTrace) -> ReplayResult
  compare(primary_result, shadow_result) -> DiffReport
```

### 21.6 `BottleneckAnalyzer` <a href="#section-103" id="section-103"></a>

职责：比较 vendor baseline 与 simulator backend，输出应用、runtime 或硬件建议。

```text
BottleneckAnalyzer
  analyze(vendor_report, sim_report, trace) -> BottleneckReport
  generate_roofline(report) -> RooflineArtifacts
  suggest_pgo(report) -> list[PgoSuggestion]
  suggest_hardware(report) -> HardwareSpecSuggestion
```

### 21.7 安全约束 <a href="#section-104" id="section-104"></a>

模拟器和未校准后端不得直接控制真实设备。安全规则：

```text
if backend == simulator and calibration_level < required:
    allow_shadow_only = true
if output_diff > threshold:
    quarantine_result()
if op.is_safety_critical:
    require_cpu_or_ppu_reference_check()
```


---

[← 上一页](04-Python实现概要设计.md) · [全书目录](../SUMMARY.md) · [下一页 →](34-SGD-Net-TC验证方案与软硬件映射20260912.md)

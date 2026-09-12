# 24. Draft Core Class Interfaces

```text
class SGDNet:
    encoder: InputEncoder
    graph_builder: GraphBuilder
    embedding: NodeEdgeEmbedding
    blocks: list[SGDBlock]
    predictor: FeedforwardPredictor
    arbiter: ControlArbiter
    error_estimator: PosteriorErrorEstimator
    refiner: TopologyRefiner
    projector: StabilityProjector
    memory_manager: MemoryManager
    source_monitor: SourceMonitor
    knowledge_manager: KnowledgeOntologyManager
    retrospection_layer: RetrospectionLayer
    fast_path_registry: FastPathRegistry
    jepa_adapter: JEPAAdapter | None
    jsbo_bridge: JSBOBridgeOperator | None
    identifiability_verifier: LeJEPAIdentifiabilityVerifier | None
    solver_router: SolverRouter | None
    online_adaptation: OnlineAdaptationController | None
    collective_merge: CollectiveMergeManager | None
    domain_adapter_registry: DomainAdapterRegistry | None
    experiment_policy: ExperimentActionPolicy | None
    trace_emitter: TraceEmitter | None
    placement_planner: PlacementPlanner | None
    backend_registry: BackendRegistry | None
    trace_recorder: WorkloadTraceRecorder | None
    shadow_executor: ShadowExecutor | None
    head: TaskHead

    forward(raw_input, event_context, frozen_version) -> SGDOutput
    observe_result(prediction_id, observation) -> ErrorReport
    adapt_window(snapshot, paired_history, budget) -> CandidateResult
```

```text
class SGDBlock:
    ssm: SSMStateLayer
    gnn: GNNTopologyLayer
    tree: DynamicTreeRoutingLayer
    norm: Normalization

    forward(state: GraphState) -> GraphState
```

```text
class SGDOutput:
    prediction
    graph_state
    error_report
    refinement_events
    stability_report
    explanation
```


---

[← Previous](section-25.md) · [Contents](../../../SUMMARY.md) · [Next →](section-27.md)

# Architectural migration

## Why the architecture changed

The initial repository was module-oriented: files were grouped by topic, but the public structure did not expose the actual computational path.

The migrated repository is pipeline-oriented. The directory structure and runtime orchestration now follow the flow shown in the architecture diagram.

## Materialized mapping

| Architecture stage | Materialized code |
|---|---|
| Data source adapters | ingestion/ |
| Canonicalization | schema.py |
| Timeline + causal preprocessing | preprocessing.py |
| Future-event targets | labels.py |
| Experiment isolation | splits.py |
| Learned forecast model | models.py |
| Layer A | twin_layers/layer_a.py |
| Layer B | twin_layers/layer_b.py |
| Layer C | twin_layers/layer_c.py |
| Three-layer runtime | twin.py |
| End-to-end orchestration | pipeline.py |
| Evidence/artifacts | evaluation.py, artifacts.py |
| Executable proof | cli.py --demo, tests/test_architecture.py |
| Trust gate | .github/workflows/ci.yml |

## Important implementation clarification

Layer A is currently a constrained prior initializer, not an XGBoost-trained physiological-parameter estimator. The architectural slot exists so a learned initializer can be introduced once defensible patient-level targets or proxy targets are available.

Layer C now consumes the learned Layer B forecast distribution and the Layer A uncertainty prior. This is the important architectural transition from a standalone simulator to a forward model attached to the inferred patient state.

## Trust boundary

1. Internal state: patient-specific latent representation.
2. Research prediction: probabilistic future-glucose trajectories and threshold-crossing probabilities.
3. Scenario simulation: explicitly hypothetical counterfactual trajectories.

No treatment recommendation is emitted by the architecture.
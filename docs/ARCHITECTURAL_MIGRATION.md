# Architectural migration

## Why the architecture changed

The initial repository was effectively module-oriented: files were grouped by topic, but the public documentation did not expose the actual computational path.

The migrated architecture is pipeline-oriented. Every major edge answers: what consumes this data, state, or artifact next?

## Before

~~~mermaid
flowchart LR
    SCHEMA["schema.py"] --- PRE["preprocessing.py"]
    LABEL["labels.py"] --- SPLIT["splits.py"]
    MODEL["models.py"] --- TRAIN["training.py"]
    TWIN["twin.py"] --- SIM["simulator.py"]
    EVAL["evaluation.py"] --- TEST["tests"]
~~~

That representation is easy to map to files but poor at explaining system behavior.

## After

~~~mermaid
flowchart TB
    DATA["OhioT1DM / DiaTrend / HUPA-UCM / Synthea"] --> CANON["Canonical schema"] --> PRE["5-min timeline + quality gates"] --> FEAT["Causal features + sequence windows"]
    PRE --> LABEL["Future hypo labels"]
    LABEL --> SPLIT["Patient-level LOSO / GroupKFold"]
    FEAT --> SPLIT
    SPLIT --> BASE["Baselines"]
    SPLIT --> TRAIN["Forecast model training"]
    TRAIN --> STATE["Layer B: dynamic patient state"]
    STATIC["Patient static profile"] --> PRIOR["Layer A: patient prior"]
    STATE --> SIM["Layer C: probabilistic forward simulator"]
    PRIOR --> SIM
    SIM --> TRAJ["Uncertainty-aware trajectories"]
    TRAJ --> RISK["30/60/120 min hypoglycemia probability"]
    SIM --> CF["Counterfactual scenario simulation"]
    RISK --> DASH["Dashboard"]
    CF --> DASH
    SPLIT --> EVAL["Patient-level evaluation"]
    RISK --> EVAL
    EVAL --> ART["Metrics / calibration / plots"]
    ART --> CARD["Validation + model/data cards"]
    TEST["pytest + leakage tests + ruff"] --> CI["GitHub Actions"]
    ART --> RG["Dataset-backed research gate"]
    RG --> CI
~~~

## Migration mapping

| Old conceptual unit | New architectural role |
|---|---|
| schema.py | canonical ingestion boundary |
| preprocessing.py | time-series normalization and causal feature boundary |
| labels.py | future-event target construction |
| splits.py | experiment isolation / leakage boundary |
| baselines.py | reference models |
| models.py | learned forecasting + state-estimation components |
| training.py | reproducible model fitting |
| twin.py | orchestration of patient-specific state and simulation |
| simulator.py | Layer C forward/counterfactual engine |
| evaluation.py | research evidence boundary |
| tests/ | implementation trust boundary |
| config.py | shared experiment/runtime dependencies |

## Important implementation clarification

Layer A is currently a constrained prior initializer, not an XGBoost-trained physiological-parameter estimator. The architectural slot exists so a learned prior can be added when defensible patient-level targets or proxy targets are defined. This prevents synthetic physiological parameters from being presented as measured truth.

## Trust boundary

1. Model state: internal latent representation of the patient.
2. Research predictions: future glucose trajectories and hypoglycemia probability.
3. Scenario simulations: explicitly hypothetical counterfactual trajectories.

No treatment recommendation is emitted by the architecture.
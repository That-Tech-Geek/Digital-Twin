# Personalized Glucose Digital Twin

A research implementation of a patient-specific digital twin for glucose forecasting and impending hypoglycemia risk estimation.

> **Research prototype:** this repository is not a medical device and does not provide treatment recommendations. Counterfactuals are simulations only.

## System architecture

~~~mermaid
flowchart TB
    subgraph S["Data sources"]
      OH["OhioT1DM<br/>primary development + LOSO"]
      DT["DiaTrend<br/>external validation"]
      HU["HUPA-UCM<br/>external multimodal validation"]
      SY["Synthea FHIR<br/>synthetic infrastructure/scenarios"]
    end
    OH --> SCH["Canonical schema<br/>schema.py"]
    DT --> SCH
    HU --> SCH
    SY --> SCH
    SCH --> PRE["5-min timeline + quality gates<br/>preprocessing.py"]
    PRE --> LAB["Future hypoglycemia labels<br/>labels.py"]
    PRE --> WIN["Causal feature windows<br/>slope · acceleration · IOB · context"]
    LAB --> SPL["Patient-level split integrity<br/>LOSO / GroupKFold<br/>splits.py"]
    SPL --> BASE["Baselines<br/>persistence · trend · static model"]
    SPL --> TRAIN["Forecast model training<br/>GRU / TCN<br/>training.py"]
    WIN --> TRAIN
    TRAIN --> STATE["Layer B — Dynamic state estimator<br/>twin_layers/layer_b.py"]
    SCH --> PRIOR["Layer A — Patient prior<br/>twin_layers/layer_a.py"]
    subgraph T["Personalized digital twin"]
      STATE --> SIM["Layer C — Probabilistic forward simulator<br/>twin_layers/layer_c.py"]
      PRIOR --> SIM
      SIM --> PATHS["Monte Carlo trajectories"]
      PATHS --> RISK["30 / 60 / 120 min<br/>P(glucose < 70)"]
      SIM --> CF["Counterfactual simulator<br/>carbs · exercise · insulin<br/>simulation only"]
    end
    PRE --> STATE
    SPL --> EVAL["Patient-level evaluation"]
    DT --> EVAL
    HU --> EVAL
    TRAIN --> EVAL
    RISK --> EVAL
    EVAL --> ART["Research artifacts<br/>metrics.json · predictions.csv · manifest"]
    ART --> CARD["Model card + data card"]
    RISK --> DASH["Research dashboard<br/>state · uncertainty · risk"]
    CF --> DASH
    subgraph C["Trust / CI gates"]
      TEST["pytest<br/>unit + architecture tests"]
      LEAK["Patient split integrity"]
      LINT["ruff"]
      E2E["dataset-free end-to-end demo"]
      CI["GitHub Actions"]
    end
    PRE -.->|"shared constants"| CFG["config.py"]
    TRAIN -.->|"seed / optimizer / model config"| CFG
    SIM -.->|"seed / trajectory count"| CFG
    TEST --> CI
    LEAK --> CI
    LINT --> CI
    E2E --> CI
~~~

## Materialized architecture

The architecture is executable through DigitalTwinPipeline rather than being documentation-only:

ingestion → canonicalization → 5-minute timeline → causal features + future labels → patient-level LOSO → GRU state estimator → Layer A prior + Layer B state → Layer C probabilistic simulator → risk/counterfactuals → artifacts

Clinical datasets remain external. The repository includes a deterministic synthetic fixture so the full dataflow can be exercised in CI without fabricating clinical evidence.

### Validation boundary

Algorithmic validation targets insulin-treated diabetes in longitudinal open cohorts, primarily T1D. India/T2D is a future deployment and prospective-validation pathway, not a claim of current clinical validation.

### Repository map

| Stage | Implementation |
|---|---|
| Source adapters | src/digital_twin/ingestion/ |
| Canonical schema | src/digital_twin/schema.py |
| Resampling / gap policy / features | src/digital_twin/preprocessing.py |
| Hypoglycemia labels | src/digital_twin/labels.py |
| Patient-level splits | src/digital_twin/splits.py |
| Baselines | src/digital_twin/baselines.py |
| Forecast model | src/digital_twin/models.py |
| Layer A | src/digital_twin/twin_layers/layer_a.py |
| Layer B | src/digital_twin/twin_layers/layer_b.py |
| Layer C | src/digital_twin/twin_layers/layer_c.py |
| Runtime twin | src/digital_twin/twin.py |
| End-to-end orchestration | src/digital_twin/pipeline.py |
| Research artifacts | src/digital_twin/artifacts.py |
| Evaluation | src/digital_twin/evaluation.py |
| CLI / demo | src/digital_twin/cli.py |
| Architecture specification | docs/ARCHITECTURE.md |
| Migration record | docs/ARCHITECTURAL_MIGRATION.md |

## Run locally

~~~bash
python -m pip install -e ".[dev,research]"
pytest -q
python -m digital_twin.cli --demo --output artifacts/demo
~~~

Full cohort reproduction requires locally available datasets under their applicable terms.
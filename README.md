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
    WIN --> STATE["Layer B — Dynamic state estimator<br/>GRU hidden state"]
    subgraph T["Personalized digital twin"]
      PRIOR["Layer A — Patient prior<br/>constrained physiological proxies"]
      STATE --> SIM["Layer C — Probabilistic forward simulator<br/>future glucose distribution"]
      PRIOR --> SIM
      STATE --> SIM
      SIM --> PATHS["Monte Carlo trajectories"]
      PATHS --> RISK["30 / 60 / 120 min<br/>P(glucose < 70)"]
      SIM --> CF["Counterfactual simulator<br/>carbs · exercise · insulin<br/>simulation only"]
    end
    SCH --> PRIOR
    TRAIN --> STATE
    PRE --> STATE
    DT --> EVAL["Patient-level evaluation"]
    HU --> EVAL
    SPL --> EVAL
    TRAIN --> EVAL
    RISK --> EVAL
    EVAL --> ART["Research artifacts<br/>metrics.json · plots · validation report"]
    ART --> CARD["Model card + data card"]
    RISK --> DASH["Research dashboard<br/>state · uncertainty · risk"]
    CF --> DASH
    subgraph C["Trust / CI gates"]
      TEST["pytest<br/>unit + integration"]
      LEAK["Patient split integrity<br/>train/test disjointness"]
      LINT["ruff"]
      RGATE["Dataset-backed research gate<br/>metrics / calibration / safety thresholds"]
      CI["GitHub Actions"]
    end
    PRE -.->|"shared constants"| CFG["config.py"]
    TRAIN -.->|"seed / optimizer / model config"| CFG
    SIM -.->|"seed / trajectory count"| CFG
    TEST --> CI
    LEAK --> CI
    LINT --> CI
    ART --> RGATE
    RGATE --> CI
~~~

## What this shows

raw longitudinal data → canonical 5-minute timeline → quality gates + causal features + future labels → patient-level validation split → baselines + learned forecasting/state estimation → Layer A patient prior + Layer B dynamic state → Layer C probabilistic forward simulation → uncertainty-aware trajectory risk → counterfactual simulation → evaluation + dashboard + research artifacts

### Validation boundary

Algorithmic validation targets insulin-treated diabetes in longitudinal open cohorts, primarily T1D. India/T2D is a future deployment and prospective-validation pathway, not a claim of current clinical validation.

### Repository map

| Stage | Implementation |
|---|---|
| Canonical schema | schema.py |
| Resampling / gap policy / features | preprocessing.py |
| Hypoglycemia labels | labels.py |
| Patient-level splits | splits.py |
| Baselines | baselines.py |
| Forecast/state models | models.py |
| Training | training.py |
| Twin/simulator | twin.py, simulator.py |
| Evaluation | evaluation.py |
| Architecture specification | docs/ARCHITECTURE.md |
| Migration record | docs/ARCHITECTURAL_MIGRATION.md |
| Validation protocol | docs/VALIDATION.md |
| Safety boundary | docs/SAFETY.md |

## Run locally

~~~bash
python -m pip install -e ".[dev,research]"
pytest -q
~~~

Full cohort reproduction requires locally available datasets under their applicable terms.
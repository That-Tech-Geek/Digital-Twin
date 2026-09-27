# Personalized Glucose Digital Twin

A research implementation of a patient-specific digital twin for glucose forecasting and impending hypoglycemia risk estimation.

> **Research prototype:** this repository is not a medical device and does not provide treatment recommendations. Counterfactuals are simulations only.

## System flow

```mermaid
flowchart TB
    %% -------------------- DATA --------------------
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
    PRE --> WIN["Sequence windows + causal features<br/>slope · acceleration · IOB · circadian"]
    LAB --> SPL["Patient-level split integrity<br/>LOSO / GroupKFold<br/>splits.py"]

    SPL --> BASE["Baselines<br/>persistence · linear trend · static model"]
    SPL --> TRAIN["Forecast training<br/>GRU / TCN<br/>training.py"]
    WIN --> TRAIN
    WIN --> STATE["Layer B — Dynamic state estimator<br/>GRU hidden state"]
    
    %% -------------------- TWIN --------------------
    subgraph T["Personalized digital twin"]
      PRIOR["Layer A — Patient prior<br/>constrained physiological proxies"]
      STATE --> SIM["Layer C — Probabilistic forward simulator<br/>trajectory distribution + Monte Carlo"]
      PRIOR --> SIM
      STATE --> SIM
      SIM --> RISK["30 / 60 / 120 min<br/>P(glucose < 70)"]
      SIM --> CF["Counterfactual simulator<br/>carbs · exercise · insulin<br/><b>simulation only</b>"]
    end

    TRAIN --> STATE
    PRE --> STATE
    SCH --> PRIOR

    RISK --> DASH["Research dashboard<br/>current state · uncertainty · risk"]
    CF --> DASH

    %% -------------------- EVALUATION --------------------
    SPL --> EVAL["Patient-level evaluation<br/>forecast + event metrics"]
    TRAIN --> EVAL
    RISK --> EVAL
    DT --> EVAL
    HU --> EVAL

    EVAL --> ART["Research artifacts<br/>metrics.json · plots<br/>validation report · model card"]

    %% -------------------- CI --------------------
    subgraph C["CI / trust gates"]
      TEST["pytest<br/>unit + integration tests"]
      LEAK["Patient split integrity<br/>no train/test patient overlap"]
      LINT["ruff"]
      CGATE["Dataset-backed research gate<br/>metrics / calibration / safety thresholds<br/><i>not run without cohort data</i>"]
    end

    PRE -.->|"config: 5-min step"| CFG["config.py<br/>shared experiment constants"]
    TRAIN -.->|"config: seed / LR / epochs"| CFG
    SIM -.->|"config: seed / N trajectories"| CFG
    TEST --> CI["GitHub Actions"]
    LEAK --> CI
    LINT --> CI
    ART --> CGATE
    CGATE --> CI

    CI -->|pass| MERGE["reviewable implementation"]

## What is currently implemented

The current stack contains canonicalization and quality gates, causal feature generation, future-event labeling, patient-level splitting, baseline models, probabilistic GRU/TCN forecasting, a constrained patient prior, stochastic trajectory simulation, counterfactual scenarios, evaluation utilities, safety/reproducibility documentation, tests, and GitHub Actions CI.

### Validation boundary

The open-data validation population is insulin-treated diabetes, primarily T1D in the available longitudinal cohorts. India/T2D is a future deployment and prospective-validation pathway, not a claim of current clinical validation.

### Repository map

| Area | Code |
|---|---|
| Schema / ingestion contract | `src/digital_twin/schema.py` |
| Preprocessing / features | `src/digital_twin/preprocessing.py` |
| Hypoglycemia labels | `src/digital_twin/labels.py` |
| Patient-level splits | `src/digital_twin/splits.py` |
| Baselines | `src/digital_twin/baselines.py` |
| Forecast models | `src/digital_twin/models.py` |
| Training | `src/digital_twin/training.py` |
| Patient prior + simulator | `src/digital_twin/twin.py`, `simulator.py` |
| Evaluation | `src/digital_twin/evaluation.py` |
| Dashboard | `src/digital_twin/dashboard/app.py` |
| Architecture | `docs/ARCHITECTURE.md` |
| Validation protocol | `docs/VALIDATION.md` |
| Safety boundary | `docs/SAFETY.md` |

## Run locally

```bash
python -m pip install -e ".[dev,research]"
pytest -q
python -m digital_twin.demo
streamlit run src/digital_twin/dashboard/app.py
```

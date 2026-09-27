# Architecture

The system is a real pipeline, not a collection of isolated modules:

**raw longitudinal data → canonical schema → quality gates → labels/features → patient-level split → forecast/event models → three-layer digital twin → probabilistic trajectories → hypoglycemia probability + counterfactual simulation → evaluation artifacts → dashboard**

## End-to-end flow

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
    PRE --> WIN["Causal feature windows<br/>glucose · slope · acceleration · IOB · context"]
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
    SPL --> EVAL["Patient-level evaluation"]
    DT --> EVAL
    HU --> EVAL
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

## Code mapping

| Flow stage | Implementation |
|---|---|
| Canonical schema | src/digital_twin/schema.py |
| Resampling / gaps / causal features | src/digital_twin/preprocessing.py |
| Hypoglycemia labels | src/digital_twin/labels.py |
| Patient-level validation splits | src/digital_twin/splits.py |
| Baselines | src/digital_twin/baselines.py |
| GRU / TCN | src/digital_twin/models.py |
| Training | src/digital_twin/training.py |
| Twin/simulation | src/digital_twin/twin.py, simulator.py |
| Evaluation | src/digital_twin/evaluation.py |
| CI | .github/workflows/ci.yml |

## Architectural boundary

Layer A currently implements a constrained patient prior initializer. It is not presented as an XGBoost-trained physiological estimator until defensible patient-level targets or proxy targets are available.

Layer B contains the dynamic patient state learned from longitudinal observations.

Layer C turns that state plus the patient prior into probabilistic glucose trajectories and counterfactual scenarios. Counterfactuals are simulations only and do not emit treatment recommendations.

## Research artifacts

Every dataset-backed experiment should terminate in explicit metrics, calibration diagnostics, plots, validation documentation, and updated model/data cards. Clinical datasets are not bundled with the repository.
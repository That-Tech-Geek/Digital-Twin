# Architecture

The system is a real pipeline, not a collection of isolated modules:

**raw longitudinal data → canonical schema → quality gates → labels/features → patient-level split → forecast/event models → three-layer digital twin → probabilistic trajectories → hypoglycemia probability + counterfactual simulation → evaluation artifacts → dashboard**

> The current implementation uses a constrained prior initializer for Layer A. It does **not** currently claim an XGBoost-trained physiological-parameter estimator. That can be introduced as a later ablation/model once appropriate patient-level targets or proxy targets are defined.

## System flow

```mermaid
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
      SIM --> PATHS["Monte Carlo trajectories<br/>N = configurable"]
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
    EVAL --> MET["metrics.json + plots<br/>forecast + event + calibration"]
    MET --> CARD["Validation report<br/>model card + data card"]

    RISK --> DASH["Research dashboard<br/>state · uncertainty · risk"]
    CF --> DASH

    subgraph C["Trust / CI gates"]
      TEST["pytest<br/>unit + integration"]
      LEAK["Patient split integrity<br/>train/test disjointness"]
      LINT["ruff"]
      RGATE["Dataset-backed research gate<br/>metrics / calibration / safety thresholds<br/><i>activated when cohort data are available</i>"]
      CI["GitHub Actions"]
    end

    PRE -.->|"shared constants"| CFG["config.py"]
    TRAIN -.->|"seed / optimizer / model config"| CFG
    SIM -.->|"seed / trajectory count"| CFG

    TEST --> CI
    LEAK --> CI
    LINT --> CI
    MET --> RGATE
    RGATE --> CI

## Code mapping

| Flow stage | Implementation |
|---|---|
| Canonical schema | `src/digital_twin/schema.py` |
| Resampling / gap policy / causal features | `src/digital_twin/preprocessing.py` |
| Future Level 1 / Level 2 labels | `src/digital_twin/labels.py` |
| Patient-level validation splits | `src/digital_twin/splits.py` |
| Baselines | `src/digital_twin/baselines.py` |
| GRU / TCN models | `src/digital_twin/models.py` |
| Training / Gaussian NLL | `src/digital_twin/training.py` |
| Layer A + twin wrapper | `src/digital_twin/models.py`, `twin.py` |
| Layer C simulator | `src/digital_twin/simulator.py` |
| Forecast/event metrics | `src/digital_twin/evaluation.py` |
| Dashboard | `src/digital_twin/dashboard/app.py` |
| CI | `.github/workflows/ci.yml` |

## Artifact flow

The intended research run ends in explicit artifacts rather than an opaque dashboard:

1. **Metrics:** forecast and event metrics at 30/60/120 minutes.
2. **Calibration:** Brier score and calibration diagnostics.
3. **Visualization:** forecast trajectories, uncertainty, and glucose error-zone analysis.
4. **Validation report:** cohort, split, configuration, metrics, limitations, and domain-shift notes.
5. **Model card / data card:** intended use, population boundary, limitations, provenance, and safety constraints.

The current public repository does not bundle restricted clinical datasets, so the dataset-backed research performance gate is deliberately distinct from ordinary CI.

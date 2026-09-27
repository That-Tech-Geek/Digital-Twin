# Architecture

The system is a real pipeline, not a collection of isolated modules:
**raw longitudinal data → canonical schema → quality gates → labels/features → patient-level split → forecast/state model → three-layer digital twin → probabilistic trajectories → hypoglycemia probability + counterfactual simulation → evaluation artifacts → dashboard**

## Code mapping

| Flow stage | Implementation |
|---|---|
| Source adapters | src/digital_twin/ingestion/ |
| Canonical schema | src/digital_twin/schema.py |
| Resampling / causal features | src/digital_twin/preprocessing.py |
| Future hypo labels | src/digital_twin/labels.py |
| Patient-level validation split | src/digital_twin/splits.py |
| Forecast model | src/digital_twin/models.py |
| Layer A | src/digital_twin/twin_layers/layer_a.py |
| Layer B | src/digital_twin/twin_layers/layer_b.py |
| Layer C | src/digital_twin/twin_layers/layer_c.py |
| Runtime twin | src/digital_twin/twin.py |
| End-to-end pipeline | src/digital_twin/pipeline.py |
| Artifacts | src/digital_twin/artifacts.py |
| Evaluation | src/digital_twin/evaluation.py |
| CI | .github/workflows/ci.yml |

## Architectural boundary

Layer A initializes constrained patient-specific physiological proxies from static context. These are model parameters, not measured clinical truth.

Layer B converts the longitudinal feature window into a dynamic latent patient state and a probabilistic future-glucose distribution.

Layer C consumes the Layer B distribution and Layer A uncertainty prior to generate Monte Carlo trajectories. The same forward engine powers observed forecasting and explicitly hypothetical counterfactual scenarios.

The runtime object in twin.py binds all three layers together. pipeline.py binds the runtime to ingestion, preprocessing, splitting, training and evaluation.

No treatment recommendation is emitted by the architecture.

## Research artifacts

Every dataset-backed experiment should terminate in explicit metrics, calibration diagnostics, predictions, run metadata, validation documentation, and updated model/data cards. Clinical datasets are not bundled with the repository.
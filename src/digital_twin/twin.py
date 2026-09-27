from dataclasses import dataclass
import numpy as np
from .contracts import ForecastOutput, TwinState
from .twin_layers import CounterfactualScenario, DynamicStateEstimator, PatientPrior, ProbabilisticForwardSimulator

@dataclass
class MaterializedPatientTwin:
    """The three-layer runtime object: prior + dynamic state + probabilistic simulator."""
    patient_id: str
    prior: PatientPrior
    state_estimator: DynamicStateEstimator
    simulator: ProbabilisticForwardSimulator
    state: TwinState|None=None

    def update(self, feature_window: np.ndarray) -> TwinState:
        self.state=self.state_estimator.encode(feature_window,self.patient_id)
        return self.state

    def forecast(self, feature_window: np.ndarray, horizon_steps=24, seed=42) -> ForecastOutput:
        self.update(feature_window)
        return self.simulator.forecast(feature_window,horizon_steps,seed)

    def simulate(self, feature_window: np.ndarray, scenario: CounterfactualScenario,
                 horizon_steps=24, seed=42) -> ForecastOutput:
        self.update(feature_window)
        return self.simulator.counterfactual(feature_window,scenario,horizon_steps,seed)

__all__=["MaterializedPatientTwin","CounterfactualScenario"]

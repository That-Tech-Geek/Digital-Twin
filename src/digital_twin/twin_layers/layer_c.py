from dataclasses import dataclass
import numpy as np
from ..config import SimulationConfig
from ..contracts import ForecastOutput
from .layer_b import DynamicStateEstimator

@dataclass(frozen=True)
class CounterfactualScenario:
    name: str
    carb_delta_g: float=0.0
    exercise_delta_min: float=0.0
    insulin_delta_u: float=0.0

class ProbabilisticForwardSimulator:
    def __init__(self, state_estimator: DynamicStateEstimator, config: SimulationConfig|None=None):
        self.state_estimator=state_estimator
        self.config=config or SimulationConfig()

    def _sample(self, mean, sigma, seed):
        rng=np.random.default_rng(seed)
        return rng.normal(mean[None,:,:],sigma[None,:,:],size=(self.config.n_trajectories,*mean.shape))

    def forecast(self, X: np.ndarray, horizon_steps: int, seed: int|None=None) -> ForecastOutput:
        mean,sigma=self.state_estimator.forecast_distribution(X,horizon_steps)
        trajectories=self._sample(mean,sigma,self.config.seed if seed is None else seed)
        return ForecastOutput(mean,sigma,trajectories)

    def counterfactual(self, X: np.ndarray, scenario: CounterfactualScenario, horizon_steps: int, seed: int|None=None):
        perturbed=X.copy()
        if scenario.carb_delta_g:
            perturbed[-1,5]+=scenario.carb_delta_g
        if scenario.insulin_delta_u:
            perturbed[-1,3]+=scenario.insulin_delta_u
        if scenario.exercise_delta_min:
            perturbed[-1,7]+=scenario.exercise_delta_min/20.0
        return self.forecast(perturbed,horizon_steps,seed)

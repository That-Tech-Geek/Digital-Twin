from dataclasses import dataclass
import numpy as np
from ..config import SimulationConfig
from ..contracts import ForecastOutput
from ..models import PriorParameters
from .layer_b import DynamicStateEstimator

@dataclass(frozen=True)
class CounterfactualScenario:
    name: str
    carb_delta_g: float=0.0
    exercise_delta_min: float=0.0
    insulin_delta_u: float=0.0

class ProbabilisticForwardSimulator:
    """Layer C: samples future glucose trajectories from Layer B conditioned on Layer A."""

    def __init__(self, state_estimator: DynamicStateEstimator,
                 prior: PriorParameters|None=None, config: SimulationConfig|None=None):
        self.state_estimator=state_estimator
        self.prior=prior
        self.config=config or SimulationConfig()

    @staticmethod
    def _batch(X: np.ndarray) -> np.ndarray:
        X=np.asarray(X,dtype=np.float32)
        if X.ndim==2:
            X=X[None,:,:]
        if X.ndim!=3:
            raise ValueError("feature window must have shape [steps, features] or [batch, steps, features]")
        return X

    def _sample(self, mean, sigma, seed):
        rng=np.random.default_rng(seed)
        if self.prior is not None:
            sigma_floor=1.0+4.0*float(self.prior.glucose_volatility)
            sigma=np.maximum(sigma,sigma_floor)
        return rng.normal(mean[None,:,:],sigma[None,:,:],
                          size=(self.config.n_trajectories,*mean.shape))

    def forecast(self, X: np.ndarray, horizon_steps: int, seed: int|None=None) -> ForecastOutput:
        X=self._batch(X)
        mean,sigma=self.state_estimator.forecast_distribution(X,horizon_steps)
        trajectories=self._sample(mean,sigma,self.config.seed if seed is None else seed)
        return ForecastOutput(mean,sigma,trajectories)

    def counterfactual(self, X: np.ndarray, scenario: CounterfactualScenario,
                       horizon_steps: int, seed: int|None=None):
        perturbed=self._batch(X).copy()
        if scenario.carb_delta_g:
            perturbed[:,-1,5]+=scenario.carb_delta_g
        if scenario.insulin_delta_u:
            perturbed[:,-1,3]+=scenario.insulin_delta_u
        if scenario.exercise_delta_min:
            perturbed[:,-1,7]+=scenario.exercise_delta_min/20.0
        return self.forecast(perturbed,horizon_steps,seed)

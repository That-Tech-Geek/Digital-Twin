from dataclasses import dataclass
import numpy as np
from .config import SimulationConfig
from .models import PriorParameters
@dataclass(frozen=True)
class Scenario:
    name:str; carb_delta_g:float=0.; exercise_delta_min:float=0.; insulin_delta_u:float=0.
def simulate_trajectories(current_glucose,slope,prior,scenario=Scenario("observed"),horizon_min=120,config=SimulationConfig()):
    rng=np.random.default_rng(config.seed); steps=horizon_min//5; paths=np.empty((config.n_trajectories,steps)); base=2+5*prior.glucose_volatility
    carb=.08*scenario.carb_delta_g*prior.carb_response; insulin=-.55*scenario.insulin_delta_u*prior.insulin_sensitivity; exercise=-.08*scenario.exercise_delta_min*prior.insulin_sensitivity
    for n in range(config.n_trajectories):
        g=float(current_glucose)
        for t in range(steps):
            drift=slope*5*np.exp(-t/18)+carb*np.exp(-t/30)+insulin*np.exp(-t/45)+exercise*np.exp(-t/60)
            g=float(np.clip(g+drift+rng.normal(0,base),20,600)); paths[n,t]=g
    return paths
def crossing_probability(paths,threshold=70): return float(np.mean(np.min(paths,axis=1)<threshold))

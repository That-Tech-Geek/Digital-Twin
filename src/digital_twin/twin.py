from dataclasses import dataclass
import numpy as np
from .models import initialize_prior
from .simulator import Scenario,simulate_trajectories,crossing_probability
from .config import SimulationConfig
@dataclass
class PatientTwin:
    patient_id:str; static_profile:dict; prior:object; current_glucose:float; current_slope:float; history:np.ndarray|None=None
    @classmethod
    def from_profile(cls,pid,profile,current_glucose,current_slope,history=None): return cls(pid,profile,initialize_prior(profile),float(current_glucose),float(current_slope),history)
    def forecast(self,horizon_min=120,n_trajectories=100,seed=42): return simulate_trajectories(self.current_glucose,self.current_slope,self.prior,horizon_min=horizon_min,config=SimulationConfig(n_trajectories,seed))
    def risk(self,horizon_min=60,threshold=70,seed=42): return crossing_probability(self.forecast(horizon_min,100,seed),threshold)
    def simulate(self,scenario,horizon_min=120,seed=42): return simulate_trajectories(self.current_glucose,self.current_slope,self.prior,scenario,horizon_min,SimulationConfig(100,seed))

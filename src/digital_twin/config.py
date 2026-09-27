from dataclasses import dataclass
from typing import Tuple
HORIZONS_MIN:Tuple[int,...]=(30,60,120)
STEP_MIN=5
@dataclass(frozen=True)
class SimulationConfig:
    n_trajectories:int=100
    seed:int=42
    sigma_floor:float=1e-3
@dataclass(frozen=True)
class TrainingConfig:
    seed:int=42
    batch_size:int=64
    epochs:int=10
    lr:float=1e-3
    weight_decay:float=1e-4
    hidden_size:int=64
    dropout:float=0.3

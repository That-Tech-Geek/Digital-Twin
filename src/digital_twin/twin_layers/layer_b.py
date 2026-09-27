import numpy as np
import torch
from ..models import GRUForecaster
from ..contracts import TwinState

class DynamicStateEstimator:
    def __init__(self, model: GRUForecaster):
        self.model=model

    @torch.no_grad()
    def encode(self, X: np.ndarray, patient_id: str) -> TwinState:
        self.model.eval()
        x=torch.as_tensor(X,dtype=torch.float32)
        _,hidden=self.model.encoder(x)
        state=hidden[0,-1].detach().cpu().numpy()
        return TwinState(patient_id, state, float(X[-1,0]), float(X[-1,1]) if X.shape[1]>1 else 0.0)

    @torch.no_grad()
    def forecast_distribution(self, X: np.ndarray, horizon_steps: int):
        self.model.eval()
        x=torch.as_tensor(X,dtype=torch.float32)
        mean,sigma=self.model(x,horizon_steps)[:,:,0],self.model(x,horizon_steps)[:,:,1]
        return mean.cpu().numpy(),sigma.cpu().numpy()

import numpy as np
import torch
from ..models import GRUForecaster
from ..contracts import TwinState

class DynamicStateEstimator:
    """Layer B: converts a longitudinal window into a latent patient state and forecast distribution."""

    def __init__(self, model: GRUForecaster):
        self.model=model

    @torch.no_grad()
    def encode(self, X: np.ndarray, patient_id: str) -> TwinState:
        self.model.eval()
        x=torch.as_tensor(X,dtype=torch.float32)
        _,hidden=self.model.encoder(x)
        state=hidden[0,-1].detach().cpu().numpy()
        current_glucose=float(X[0,-1,0])
        current_slope=float(X[0,-1,1]) if X.shape[-1]>1 else 0.0
        return TwinState(patient_id,state,current_glucose,current_slope)

    @torch.no_grad()
    def forecast_distribution(self, X: np.ndarray, horizon_steps: int):
        self.model.eval()
        x=torch.as_tensor(X,dtype=torch.float32)
        output=self.model(x,horizon_steps)
        return output[:,:,0].cpu().numpy(),output[:,:,1].cpu().numpy()

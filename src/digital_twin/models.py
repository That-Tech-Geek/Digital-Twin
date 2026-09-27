from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
class GRUForecaster(nn.Module):
    def __init__(self,input_size=11,hidden_size=64,dropout=0.3):
        super().__init__(); self.encoder=nn.GRU(input_size,hidden_size,batch_first=True); self.drop=nn.Dropout(dropout)
        self.bridge=nn.Linear(hidden_size,32); self.decoder=nn.GRU(1,32,batch_first=True); self.mu=nn.Linear(32,1); self.log_sigma=nn.Linear(32,1)
    def forward(self,x,horizon):
        _,h=self.encoder(x); dh=torch.tanh(self.bridge(self.drop(h[0]))).unsqueeze(0); cur=x[:,-1:,:1]; out=[]
        for _ in range(horizon):
            z,dh=self.decoder(cur,dh); mu=self.mu(z); sig=torch.nn.functional.softplus(self.log_sigma(z))+1e-3
            out.append(torch.cat([mu,sig],-1)); cur=mu.detach()
        return torch.cat(out,1)
class TCNForecaster(nn.Module):
    def __init__(self,input_size=11,hidden=64):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(input_size,hidden,3,padding=2),nn.ReLU(),nn.Conv1d(hidden,hidden,3,padding=4,dilation=2),nn.ReLU(),nn.Conv1d(hidden,32,3,padding=8,dilation=4),nn.ReLU()); self.head=nn.Linear(32,1)
    def forward(self,x,horizon): return self.head(self.net(x.transpose(1,2))[:,:,-1]).repeat(1,horizon).unsqueeze(-1)
@dataclass(frozen=True)
class PriorParameters:
    insulin_sensitivity:float; baseline_glucose:float; carb_response:float; hypo_susceptibility:float; circadian_risk_amplitude:float; glucose_volatility:float
def initialize_prior(static):
    bmi=float(static.get("bmi",25) or 25); a=float(static.get("hba1c",7) or 7); d=float(static.get("diabetes_duration",5) or 5); ph=float(static.get("prior_hypoglycemia_events",0) or 0)
    return PriorParameters(float(np.clip(1/(1+.03*(bmi-25)+.08*max(a-6,0)),0,1)),float(np.clip(100+8*(a-5.5),80,200)),float(np.clip(.35+.03*max(a-6,0)+.01*d,0,1)),float(np.clip(.15+.02*ph+.01*d,0,1)),.2,float(np.clip(.25+.03*max(a-6,0),0,1)))

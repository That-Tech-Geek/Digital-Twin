import random,numpy as np,torch
from .config import TrainingConfig
from .models import GRUForecaster
def seed_everything(seed=42): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
def gaussian_nll(out,target):
    mu=out[:,:,0]; sig=out[:,:,1].clamp_min(1e-3); return (.5*torch.log(2*torch.pi*sig**2)+(target-mu)**2/(2*sig**2)).mean()
def train_gru(X,y,config=TrainingConfig()):
    seed_everything(config.seed); model=GRUForecaster(X.shape[-1],config.hidden_size,config.dropout); opt=torch.optim.AdamW(model.parameters(),lr=config.lr,weight_decay=config.weight_decay)
    ds=torch.utils.data.TensorDataset(torch.tensor(X),torch.tensor(y)); loader=torch.utils.data.DataLoader(ds,batch_size=config.batch_size,shuffle=True)
    for _ in range(config.epochs):
        for xb,yb in loader:
            opt.zero_grad(); loss=gaussian_nll(model(xb,yb.shape[1]),yb); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
    return model
@torch.no_grad()
def predict_gru(model,X,horizon=None):
    model.eval(); o=model(torch.tensor(X),horizon or X.shape[1]); return o[:,:,0].numpy(),o[:,:,1].numpy()

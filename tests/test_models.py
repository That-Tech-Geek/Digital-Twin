import torch
from digital_twin.models import GRUForecaster
def test_gru_shape(): assert GRUForecaster()(torch.zeros(2,48,11),6).shape==(2,6,2)

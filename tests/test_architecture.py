import numpy as np
from digital_twin.demo_data import make_demo_dataset
from digital_twin.pipeline import DigitalTwinPipeline

def test_materialized_pipeline_prepares_all_stages():
    prepared=DigitalTwinPipeline().prepare(make_demo_dataset(n_patients=2,steps=80))
    for c in ("glucose_slope","glucose_accel","iob","hypo_l1_30m","hypo_l1_60m","hypo_l1_120m"):
        assert c in prepared.frame.columns

def test_materialized_pipeline_creates_windows():
    prepared=DigitalTwinPipeline().prepare(make_demo_dataset(n_patients=2,steps=100))
    X,y,meta=DigitalTwinPipeline().make_windows(prepared)
    assert X.ndim==3 and y.ndim==2 and len(meta)==len(X)
    assert set(m[0] for m in meta)=={"demo-0","demo-1"}

def test_counterfactual_engine_shape():
    from digital_twin.models import GRUForecaster
    from digital_twin.twin_layers import DynamicStateEstimator,ProbabilisticForwardSimulator,CounterfactualScenario
    import torch
    model=GRUForecaster()
    est=DynamicStateEstimator(model)
    sim=ProbabilisticForwardSimulator(est)
    X=np.zeros((1,48,11),dtype=np.float32)
    out=sim.counterfactual(X,CounterfactualScenario("carbs",carb_delta_g=30),24,42)
    assert out.mean.shape==(1,24) and out.trajectories.shape[0]==100

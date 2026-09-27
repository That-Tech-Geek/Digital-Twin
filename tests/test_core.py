import numpy as np,pandas as pd
from digital_twin.labels import add_hypoglycemia_labels
from digital_twin.preprocessing import compute_iob,add_derived_features
from digital_twin.splits import assert_disjoint
from digital_twin.twin import PatientTwin
def df():
    t=pd.date_range("2026-01-01",periods=80,freq="5min",tz="UTC"); return pd.DataFrame({"patient_id":"p1","timestamp":t,"glucose":np.linspace(120,60,80),"insulin":0.,"carbs":0.,"hr":70.,"activity":.2,"sleep_stage":0.})
def test_iob_nonnegative(): assert (compute_iob(df())>=0).all()
def test_level2_subset(): 
    x=add_hypoglycemia_labels(df(),(30,)).dropna(); assert (x["hypo_l2_30m"]<=x["hypo_l1_30m"]).all()
def test_split(): assert_disjoint(["a"],["b"])
def test_features(): assert add_derived_features(df())["time_sin"].notna().all()
def test_deterministic():
    p=PatientTwin.from_profile("p",{"bmi":25,"hba1c":7},110,-.1); assert np.allclose(p.forecast(60,20,123),p.forecast(60,20,123))

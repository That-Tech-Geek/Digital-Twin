import numpy as np
import streamlit as st

from digital_twin.config import TrainingConfig
from digital_twin.demo_data import make_demo_dataset
from digital_twin.pipeline import DigitalTwinPipeline
from digital_twin.twin_layers import CounterfactualScenario

st.set_page_config(page_title="Glucose Digital Twin",layout="wide")
st.title("Personalized Glucose Digital Twin")
st.caption("Research prototype. Forecasts and scenarios are not treatment recommendations.")

@st.cache_resource
def build_demo():
    pipe=DigitalTwinPipeline()
    prepared=pipe.prepare(make_demo_dataset())
    X,y,meta=pipe.make_windows(prepared)
    train_mask=np.asarray([m[0]!="demo-0" for m in meta])
    from digital_twin.training import train_gru
    from digital_twin.twin_layers import DynamicStateEstimator,PatientPrior,ProbabilisticForwardSimulator
    model=train_gru(X[train_mask],y[train_mask],TrainingConfig(epochs=1,batch_size=32))
    estimator=DynamicStateEstimator(model)
    prior=PatientPrior.from_profile({"bmi":25,"hba1c":7,"diabetes_duration":5,"prior_hypoglycemia_events":1})
    return pipe,prepared,X,y,meta,estimator,prior,ProbabilisticForwardSimulator(estimator,prior)

pipe,prepared,X,y,meta,estimator,prior,simulator=build_demo()
test_indices=[i for i,m in enumerate(meta) if m[0]=="demo-0"]
latest=X[test_indices[-1]]
state=estimator.encode(latest[None,:,:],"demo-0")
forecast=simulator.forecast(latest,24,seed=42)

st.subheader("Current twin state")
c1,c2,c3=st.columns(3)
c1.metric("Current glucose",f"{state.current_glucose:.1f} mg/dL")
c2.metric("Current slope",f"{state.current_slope:.3f} mg/dL/min")
c3.metric("Hidden state",str(state.hidden_state.size))

st.subheader("Forecast risk")
cols=st.columns(3)
for col,h in zip(cols,(30,60,120)):
    steps=h//5
    p=float(np.mean(np.min(forecast.trajectories[0,:,:steps],axis=1)<70))
    col.metric(f"P(<70) within {h} min",f"{p:.1%}")

st.subheader("Probabilistic forecast")
median=np.median(forecast.trajectories[0],axis=0)
low=np.percentile(forecast.trajectories[0],5,axis=0)
high=np.percentile(forecast.trajectories[0],95,axis=0)
st.line_chart({"P5":low,"Median":median,"P95":high})

st.subheader("Scenario simulator")
scenario_name=st.selectbox("Scenario",[
    "Observed",
    "+30 g carbohydrate",
    "+20 min moderate exercise",
    "-2 U insulin (hypothetical)"
])
scenario={
    "Observed":CounterfactualScenario("observed"),
    "+30 g carbohydrate":CounterfactualScenario("carbs",carb_delta_g=30),
    "+20 min moderate exercise":CounterfactualScenario("exercise",exercise_delta_min=20),
    "-2 U insulin (hypothetical)":CounterfactualScenario("insulin",insulin_delta_u=-2),
}[scenario_name]
if st.button("Simulate scenario"):
    out=simulator.counterfactual(latest,scenario,24,seed=42)
    p=float(np.mean(np.min(out.trajectories[0],axis=1)<70))
    st.metric("Simulated P(<70) within 120 min",f"{p:.1%}")
    st.line_chart({
        "P5":np.percentile(out.trajectories[0],5,axis=0),
        "Median":np.median(out.trajectories[0],axis=0),
        "P95":np.percentile(out.trajectories[0],95,axis=0)
    })
    st.warning("This is a hypothetical scenario simulation, not a treatment recommendation.")

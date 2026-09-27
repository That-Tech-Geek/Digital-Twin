from pathlib import Path
import numpy as np
import pandas as pd
from .artifacts import write_run_artifacts
from .config import TrainingConfig
from .contracts import PipelineResult, PreparedDataset
from .evaluation import classification_metrics, regression_metrics
from .labels import add_hypoglycemia_labels
from .preprocessing import add_derived_features, resample_patient, window_frame
from .schema import canonicalize
from .splits import assert_disjoint
from .training import seed_everything, train_gru
from .twin_layers import (
    CounterfactualScenario,
    DynamicStateEstimator,
    PatientPrior,
    ProbabilisticForwardSimulator,
)

FEATURES=("glucose","glucose_slope","glucose_accel","insulin","iob","carbs","hr",
          "activity","sleep_stage","time_sin","time_cos")
LABELS=("hypo_l1_30m","hypo_l1_60m","hypo_l1_120m",
        "hypo_l2_30m","hypo_l2_60m","hypo_l2_120m")

class DigitalTwinPipeline:
    """Orchestrates the architecture as an executable dataflow."""

    def prepare(self, raw: pd.DataFrame) -> PreparedDataset:
        canonical=canonicalize(raw)
        timeline=resample_patient(canonical)
        features=add_derived_features(timeline)
        labeled=add_hypoglycemia_labels(features)
        return PreparedDataset(labeled,FEATURES,LABELS)

    def make_windows(self, prepared: PreparedDataset):
        X,y,meta=window_frame(prepared.frame,history_steps=48,forecast_steps=24)
        return X,y,meta

    def fit_loso(self, prepared: PreparedDataset, held_out_patient: str,
                 output_dir: str|Path, config: TrainingConfig|None=None,
                 static_profile: dict|None=None) -> PipelineResult:
        config=config or TrainingConfig()
        X,y,meta=self.make_windows(prepared)
        groups=np.asarray([m[0] for m in meta])
        train_mask=groups!=held_out_patient
        test_mask=groups==held_out_patient
        test_patients={held_out_patient}
        if not test_mask.any():
            raise ValueError(f"No windows found for {held_out_patient}")
        train_patients=set(groups[train_mask])
        assert_disjoint(train_patients,test_patients)

        model=train_gru(X[train_mask],y[train_mask],config)
        state_estimator=DynamicStateEstimator(model)
        prior=PatientPrior.from_profile(static_profile or {
            "bmi":25,"hba1c":7,"diabetes_duration":5,"prior_hypoglycemia_events":0
        })
        simulator=ProbabilisticForwardSimulator(state_estimator,prior)

        Xtest=X[test_mask]
        ytest=y[test_mask]
        test_meta=[m for i,m in enumerate(meta) if test_mask[i]]
        pred_mu,pred_sigma=model_predict(state_estimator,Xtest,24)
        sampled=simulator.forecast(Xtest,24,seed=config.seed).trajectories

        metrics={}
        risk_rows={}
        for steps,h in ((6,30),(12,60),(24,120)):
            metrics[f"forecast_{h}m"]=regression_metrics(ytest[:,:steps],pred_mu[:,:steps])
            true_event=(np.min(ytest[:,:steps],axis=1)<70).astype(int)
            predicted_risk=np.mean(np.min(sampled[:,:,:steps],axis=2)<70,axis=0)
            risk_rows[str(h)]=classification_metrics(true_event,predicted_risk)
        metrics["risk"]=risk_rows

        latest=Xtest[-1]
        state=state_estimator.encode(latest[None,:,:],held_out_patient)
        forecast=simulator.forecast(latest[None,:,:],24,seed=config.seed)
        metrics["latest_twin"]={
            "30m":float(np.mean(np.min(forecast.trajectories[0,:,:6],axis=1)<70)),
            "60m":float(np.mean(np.min(forecast.trajectories[0,:,:12],axis=1)<70)),
            "120m":float(np.mean(np.min(forecast.trajectories[0,:,:24],axis=1)<70)),
        }

        counterfactual=simulator.counterfactual(
            latest,CounterfactualScenario("plus_30g_carbs",carb_delta_g=30),24,config.seed
        )
        metrics["counterfactual_demo"]={
            "scenario":"plus_30g_carbs",
            "risk_120m":float(np.mean(np.min(counterfactual.trajectories[0,:,:24],axis=1)<70)),
        }

        predictions=pd.DataFrame({
            "timestamp":[str(m[1]) for m in test_meta],
            "patient_id":[m[0] for m in test_meta],
            "glucose_t30":pred_mu[:,5],
            "glucose_t60":pred_mu[:,11],
            "glucose_t120":pred_mu[:,23],
        })
        metadata={
            "held_out_patient":held_out_patient,
            "train_patients":sorted(train_patients),
            "seed":config.seed,
            "n_test_windows":int(test_mask.sum()),
            "prior":prior.parameters.__dict__,
            "state_dim":int(state.hidden_state.size),
        }
        paths=write_run_artifacts(output_dir,metrics,metadata,predictions)
        return PipelineResult(metrics,paths,held_out_patient)

def model_predict(estimator,X,horizon):
    means=[]; sigmas=[]
    for row in X:
        mu,s=estimator.forecast_distribution(row[None,:,:],horizon)
        means.append(mu[0]); sigmas.append(s[0])
    return np.asarray(means),np.asarray(sigmas)

def run_demo(output_dir="artifacts/demo"):
    from .demo_data import make_demo_dataset
    seed_everything(42)
    pipe=DigitalTwinPipeline()
    prepared=pipe.prepare(make_demo_dataset())
    return pipe.fit_loso(prepared,"demo-0",output_dir,
                         TrainingConfig(epochs=1,batch_size=32))

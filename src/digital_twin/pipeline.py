from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from .artifacts import write_run_artifacts
from .config import TrainingConfig
from .contracts import PipelineResult, PreparedDataset
from .labels import add_hypoglycemia_labels
from .preprocessing import add_derived_features, resample_patient, window_frame
from .schema import canonicalize
from .splits import assert_disjoint
from .training import train_gru, seed_everything
from .twin_layers import DynamicStateEstimator, PatientPrior, ProbabilisticForwardSimulator, CounterfactualScenario
from .evaluation import regression_metrics

FEATURES=("glucose","glucose_slope","glucose_accel","insulin","iob","carbs","hr","activity","sleep_stage","time_sin","time_cos")
LABELS=("hypo_l1_30m","hypo_l1_60m","hypo_l1_120m","hypo_l2_30m","hypo_l2_60m","hypo_l2_120m")

class DigitalTwinPipeline:
    def prepare(self, raw: pd.DataFrame) -> PreparedDataset:
        canonical=canonicalize(raw)
        timeline=resample_patient(canonical)
        features=add_derived_features(timeline)
        labeled=add_hypoglycemia_labels(features)
        return PreparedDataset(labeled,FEATURES,LABELS)

    def make_windows(self, prepared: PreparedDataset):
        X,y,meta=window_frame(prepared.frame,history_steps=48,forecast_steps=24)
        return X,y,meta

    def fit_loso(self, prepared: PreparedDataset, held_out_patient: str, output_dir: str|Path,
                 config: TrainingConfig|None=None) -> PipelineResult:
        config=config or TrainingConfig()
        X,y,meta=self.make_windows(prepared)
        groups=np.asarray([m[0] for m in meta])
        test_patients={held_out_patient}
        train_mask=groups!=held_out_patient
        test_mask=groups==held_out_patient
        if not test_mask.any(): raise ValueError(f"No windows found for {held_out_patient}")
        train_patients=set(groups[train_mask]); assert_disjoint(train_patients,test_patients)
        model=train_gru(X[train_mask],y[train_mask],config)
        estimator=DynamicStateEstimator(model)
        simulator=ProbabilisticForwardSimulator(estimator)

        Xtest=X[test_mask]; ytest=y[test_mask]; test_meta=[m for i,m in enumerate(meta) if test_mask[i]]
        pred_mu,pred_sigma=model_predict(estimator,Xtest,24)
        metrics={}
        for steps,h in ((6,30),(12,60),(24,120)):
            metrics[f"forecast_{h}m"]=regression_metrics(ytest[:,:steps],pred_mu[:,:steps])
        profile={"bmi":25,"hba1c":7,"diabetes_duration":5,"prior_hypoglycemia_events":0}
        prior=PatientPrior.from_profile(profile)
        latest=Xtest[-1]
        state=estimator.encode(latest[None,:,:],"heldout")
        forecast=simulator.forecast(latest[None,:,:],24,seed=config.seed)
        metrics["risk"]={str(h):float(np.mean(np.min(forecast.trajectories[:,:h//5],axis=2)<70))
                         for h in (30,60,120)}
        predictions=pd.DataFrame({"timestamp":[str(m[1]) for m in test_meta[:len(pred_mu)]],
                                  "patient_id":[m[0] for m in test_meta[:len(pred_mu)]],
                                  "glucose_t30":pred_mu[:,5]})
        metadata={"held_out_patient":held_out_patient,"train_patients":sorted(train_patients),
                  "seed":config.seed,"n_test_windows":int(test_mask.sum()),
                  "prior":prior.parameters.__dict__,"state_dim":int(state.hidden_state.size)}
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
    df=make_demo_dataset()
    pipe=DigitalTwinPipeline(); prepared=pipe.prepare(df)
    result=pipe.fit_loso(prepared,"demo-0",output_dir,TrainingConfig(epochs=1,batch_size=32))
    return result

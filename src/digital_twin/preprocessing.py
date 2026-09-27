import numpy as np
import pandas as pd
from .config import STEP_MIN
def resample_patient(df,freq="5min"):
    rows=[]
    for pid,g in df.groupby("patient_id",sort=False):
        base=g.sort_values("timestamp").set_index("timestamp").resample(freq).asfreq()
        base["patient_id"]=pid
        base["glucose_observed"]=base["glucose"].notna().astype(int)
        for c in ("glucose","hr","activity"):
            base[c+"_missing"]=base[c].isna().astype(int)
            base[c]=base[c].interpolate(limit=int(15/STEP_MIN),limit_area="inside")
        for c in ("insulin","carbs"): base[c]=base[c].fillna(0.0)
        base["long_gap"]=base["glucose"].isna().astype(int)
        rows.append(base.reset_index())
    return pd.concat(rows,ignore_index=True) if rows else df.copy()
def add_derived_features(df):
    out=df.copy().sort_values(["patient_id","timestamp"])
    g=out.groupby("patient_id",group_keys=False)
    out["glucose_slope"]=g["glucose"].transform(lambda s:s.diff(3)/15.0)
    out["glucose_accel"]=g["glucose_slope"].transform(lambda s:s.diff(3)/15.0)
    minutes=out["timestamp"].dt.hour*60+out["timestamp"].dt.minute
    out["time_sin"]=np.sin(2*np.pi*minutes/1440); out["time_cos"]=np.cos(2*np.pi*minutes/1440)
    out["iob"]=compute_iob(out)
    return out.reset_index(drop=True)
def compute_iob(df,duration_min=300.0):
    decay=np.exp(-STEP_MIN/duration_min); iob=np.zeros(len(df))
    for i,x in enumerate(df["insulin"].fillna(0).astype(float)):
        iob[i]=(iob[i-1]*decay if i else 0)+x
    return pd.Series(iob,index=df.index,name="iob")
def causal_patient_normalize(df,columns):
    out=df.copy()
    for _,idx in out.groupby("patient_id").groups.items():
        sub=out.loc[idx].sort_values("timestamp")
        for c in columns:
            if c not in out: continue
            v=sub[c].astype(float); m=v.expanding().mean(); s=v.expanding(min_periods=2).std().fillna(1).clip(lower=1e-6)
            out.loc[sub.index,c+"_z"]=(v.to_numpy()-m.to_numpy())/s.to_numpy()
    return out
def window_frame(df,history_steps=48,forecast_steps=24):
    cols=["glucose","glucose_slope","glucose_accel","insulin","iob","carbs","hr","activity","sleep_stage","time_sin","time_cos"]
    X=[]; y=[]; meta=[]
    for pid,g in df.groupby("patient_id",sort=False):
        g=g.sort_values("timestamp").reset_index(drop=True)
        for end in range(history_steps,len(g)-forecast_steps+1):
            h=g.iloc[end-history_steps:end]; f=g.iloc[end:end+forecast_steps]
            if h["glucose"].isna().any() or f["glucose"].isna().any(): continue
            X.append(h[cols].fillna(0).to_numpy("float32")); y.append(f["glucose"].to_numpy("float32")); meta.append((pid,g.iloc[end]["timestamp"]))
    return np.asarray(X),np.asarray(y),meta

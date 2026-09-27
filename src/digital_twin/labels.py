import numpy as np
def add_hypoglycemia_labels(df,horizons=(30,60,120),step_min=5):
    out=df.copy().sort_values(["patient_id","timestamp"]).reset_index(drop=True)
    for h in horizons:
        steps=h//step_min; l1=np.full(len(out),np.nan); l2=np.full(len(out),np.nan)
        for _,idx in out.groupby("patient_id").groups.items():
            a=out.loc[idx,"glucose"].to_numpy(float)
            for j in range(len(a)):
                fut=a[j+1:j+1+steps]
                if len(fut)<steps or np.isnan(fut).any(): continue
                l1[idx[j]]=float(np.min(fut)<70); l2[idx[j]]=float(np.min(fut)<54)
        out[f"hypo_l1_{h}m"]=l1; out[f"hypo_l2_{h}m"]=l2
    return out

import numpy as np
import pandas as pd

def make_demo_dataset(n_patients=4,steps=180,seed=42):
    rng=np.random.default_rng(seed); rows=[]
    t=pd.date_range("2026-01-01",periods=steps,freq="5min",tz="UTC")
    for p in range(n_patients):
        g=110+rng.normal(0,5)
        values=[]
        for i in range(steps):
            meal=30.0 if i in (36,96,150) else 0.0
            bolus=3.0 if meal else 0.0
            drift=rng.normal(0,1.8)+0.03*meal-0.7*bolus
            g=float(np.clip(g+drift,45,240)); values.append(g)
        arr=np.asarray(values)
        rows.extend({"patient_id":f"demo-{p}","timestamp":t[i],"glucose":arr[i],
                     "insulin":3.0 if i in (36,96,150) else 0.0,
                     "carbs":30.0 if i in (36,96,150) else 0.0,
                     "hr":70.0+10.0*np.sin(i/18),"activity":0.3,
                     "sleep_stage":0.0} for i in range(steps))
    return pd.DataFrame(rows)

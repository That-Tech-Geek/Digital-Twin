import json
from pathlib import Path
import pandas as pd

GLUCOSE_CODES={"2339-0"}
HEART_RATE_CODES={"8867-4"}

def _code(resource):
    codings=resource.get("code",{}).get("coding",[])
    return {str(x.get("code")) for x in codings if x.get("code")}

def load_synthea_bundle(path: str|Path) -> pd.DataFrame:
    payload=json.loads(Path(path).read_text(encoding="utf-8"))
    entries=payload.get("entry",[])
    rows=[]
    patient_id="synthea-patient"
    for item in entries:
        r=item.get("resource",{})
        if r.get("resourceType")=="Patient":
            patient_id=str(r.get("id") or patient_id)
        if r.get("resourceType")!="Observation":
            continue
        codes=_code(r)
        ts=r.get("effectiveDateTime") or r.get("issued")
        value=r.get("valueQuantity",{}).get("value")
        if ts is None or value is None:
            continue
        row={"patient_id":str(r.get("subject",{}).get("reference",patient_id)).split("/")[-1],
             "timestamp":ts,"glucose":None,"insulin":0.0,"carbs":0.0,
             "hr":None,"activity":None,"sleep_stage":None}
        if codes & GLUCOSE_CODES: row["glucose"]=float(value)
        elif codes & HEART_RATE_CODES: row["hr"]=float(value)
        else: continue
        rows.append(row)
    if not rows:
        raise ValueError("No supported Synthea observations found")
    return pd.DataFrame(rows)

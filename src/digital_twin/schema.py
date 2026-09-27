from dataclasses import dataclass
import pandas as pd
REQUIRED_COLUMNS=("patient_id","timestamp","glucose","insulin","carbs","hr","activity","sleep_stage")
@dataclass(frozen=True)
class ValidationReport:
    rows:int
    patients:int
    invalid_rows:int
    warnings:tuple[str,...]
def validate_raw_frame(df:pd.DataFrame)->ValidationReport:
    missing=[c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    ts=pd.to_datetime(df["timestamp"],utc=True,errors="coerce")
    glucose=pd.to_numeric(df["glucose"],errors="coerce")
    invalid=int(ts.isna().sum()+((glucose.notna())&((glucose<20)|(glucose>600))).sum())
    warnings=("missing patient_id values",) if df["patient_id"].isna().any() else ()
    return ValidationReport(len(df),df["patient_id"].nunique(dropna=True),invalid,warnings)
def canonicalize(df):
    validate_raw_frame(df)
    out=df.copy()
    out["timestamp"]=pd.to_datetime(out["timestamp"],utc=True)
    for c in ("glucose","insulin","carbs","hr","activity","sleep_stage"): out[c]=pd.to_numeric(out[c],errors="coerce")
    return out.sort_values(["patient_id","timestamp"]).drop_duplicates(["patient_id","timestamp"]).reset_index(drop=True)

from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from ..schema import canonicalize, REQUIRED_COLUMNS

@dataclass(frozen=True)
class CSVSourceSpec:
    name: str
    column_map: dict[str,str] = field(default_factory=dict)

def load_csv_source(path: str|Path, spec: CSVSourceSpec) -> pd.DataFrame:
    raw=pd.read_csv(path)
    rename={src:dst for src,dst in spec.column_map.items() if src in raw.columns}
    out=raw.rename(columns=rename).copy()
    for c in REQUIRED_COLUMNS:
        if c not in out:
            out[c]=0.0 if c not in ("patient_id","timestamp") else None
    if out["patient_id"].isna().all():
        raise ValueError(f"{spec.name}: patient_id is required")
    if out["timestamp"].isna().all():
        raise ValueError(f"{spec.name}: timestamp is required")
    return canonicalize(out)

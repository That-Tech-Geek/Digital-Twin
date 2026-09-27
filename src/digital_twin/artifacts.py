import json
from pathlib import Path
from typing import Any
import pandas as pd

def write_run_artifacts(output_dir: str|Path, metrics: dict[str,Any], metadata: dict[str,Any],
                        predictions: pd.DataFrame|None=None) -> tuple[Path,...]:
    root=Path(output_dir); root.mkdir(parents=True,exist_ok=True)
    metrics_path=root/"metrics.json"
    metrics_path.write_text(json.dumps(metrics,indent=2,sort_keys=True,default=str),encoding="utf-8")
    manifest=root/"run_manifest.json"
    manifest.write_text(json.dumps(metadata,indent=2,sort_keys=True,default=str),encoding="utf-8")
    paths=[metrics_path,manifest]
    if predictions is not None:
        pred_path=root/"predictions.csv"
        predictions.to_csv(pred_path,index=False)
        paths.append(pred_path)
    return tuple(paths)

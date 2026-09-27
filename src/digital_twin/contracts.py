from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class PreparedDataset:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    label_columns: tuple[str, ...]

@dataclass(frozen=True)
class ForecastOutput:
    mean: np.ndarray
    sigma: np.ndarray
    trajectories: np.ndarray

@dataclass(frozen=True)
class TwinState:
    patient_id: str
    hidden_state: np.ndarray
    current_glucose: float
    current_slope: float

@dataclass(frozen=True)
class RunArtifact:
    path: Path
    kind: str

@dataclass(frozen=True)
class PipelineResult:
    metrics: dict[str, Any]
    artifact_paths: tuple[Path, ...]
    held_out_patient: str

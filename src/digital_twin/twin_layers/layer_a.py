from dataclasses import dataclass
from ..models import PriorParameters, initialize_prior

@dataclass(frozen=True)
class PatientPrior:
    parameters: PriorParameters

    @classmethod
    def from_profile(cls, profile: dict) -> "PatientPrior":
        return cls(initialize_prior(profile))

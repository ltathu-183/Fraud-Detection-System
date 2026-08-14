"""Single source of truth for the supported pipeline."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.70
    validation: float = 0.15
    policy: float = 0.075
    test: float = 0.075

    def __post_init__(self):
        if abs(sum(asdict(self).values()) - 1.0) > 1e-12:
            raise ValueError("Split fractions must sum to one")


@dataclass(frozen=True)
class PolicyConfig:
    min_recall: float = 0.80  # fraud triage coverage: REVIEW or DECLINE
    max_review_rate: float = 0.20
    max_false_decline_rate: float = 0.02
    threshold_grid_size: int = 101


@dataclass(frozen=True)
class PipelineConfig:
    random_state: int = 42
    split: SplitConfig = SplitConfig()
    policy: PolicyConfig = PolicyConfig()


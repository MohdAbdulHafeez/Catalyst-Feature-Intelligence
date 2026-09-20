from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class BenchmarkTask(StrEnum):
    """Supported supervised learning task types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class BenchmarkStatus(StrEnum):
    """Lifecycle state of a benchmark trial."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ScoreDirection(StrEnum):
    """Optimization direction for a benchmark metric."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class FoldResult(BaseModel):
    """Result captured from one cross-validation fold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fold: int = Field(ge=0)
    score: FiniteFloat
    fit_time_s: FiniteFloat = Field(ge=0)
    score_time_s: FiniteFloat = Field(ge=0)
    n_features: int = Field(ge=0)
    memory_bytes: int | None = Field(default=None, ge=0)


class TrialResult(BaseModel):
    """Aggregate result for one encoder/model benchmark candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_name: str = Field(min_length=1)
    task: BenchmarkTask
    metric_name: str = Field(min_length=1)
    status: BenchmarkStatus

    mean_score: FiniteFloat | None = None
    std_score: FiniteFloat | None = Field(default=None, ge=0)

    fit_time_total_s: FiniteFloat = Field(ge=0)
    score_time_total_s: FiniteFloat = Field(ge=0)

    mean_n_features: FiniteFloat | None = Field(default=None, ge=0)
    peak_n_features: int | None = Field(default=None, ge=0)

    folds: tuple[FoldResult, ...] = ()

    error_type: str | None = None
    error_message: str | None = None


class BenchmarkSummary(BaseModel):
    """Top-level metadata describing a benchmark execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task: BenchmarkTask
    metric_name: str = Field(min_length=1)
    score_direction: ScoreDirection
    cv_splits: int = Field(ge=2)
    shuffle: bool
    random_state: int | None


class BenchmarkResult(BaseModel):
    """Complete structured output from a benchmark execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: BenchmarkSummary
    trials: tuple[TrialResult, ...] = ()


DEFAULT_RANDOM_STATE: Final[int] = 42
DEFAULT_CV_SPLITS: Final[int] = 5

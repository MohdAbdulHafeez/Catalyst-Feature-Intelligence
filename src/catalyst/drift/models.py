from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

MISSING_CATEGORY_KEY: Final[str] = "<CATALYST::MISSING>"


class DriftMetric(StrEnum):
    """Canonical categorical drift metrics supported by CATALYST."""

    PSI = "psi"
    JENSEN_SHANNON = "jensen_shannon"
    TOTAL_VARIATION = "total_variation"
    UNSEEN_RATE = "unseen_rate"
    MISSING_RATE_DELTA = "missing_rate_delta"
    HHI_DELTA = "hhi_delta"
    TOP_CATEGORY_SHARE_DELTA = "top_category_share_delta"


class RobustnessStatus(StrEnum):
    """Lifecycle state of an encoder robustness evaluation."""

    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DriftConfig:
    """Configuration for numerical stability in categorical drift metrics."""

    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be greater than zero.")


class CategoryFrequency(BaseModel):
    """Observed frequency for one canonicalized categorical value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category_key: str = Field(min_length=1)
    category_label: str = Field(min_length=1)
    count: int = Field(ge=1)
    proportion: FiniteFloat = Field(ge=0.0, le=1.0)


class CategoricalDistributionProfile(BaseModel):
    """Distribution profile for one categorical feature."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_name: str = Field(min_length=1)
    row_count: int = Field(ge=1)
    non_missing_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    missing_rate: FiniteFloat = Field(ge=0.0, le=1.0)
    unique_category_count: int = Field(ge=0)
    categories: tuple[CategoryFrequency, ...] = ()


class CategoricalDriftMetrics(BaseModel):
    """Pairwise drift measurements between training and reference data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_name: str = Field(min_length=1)

    train_row_count: int = Field(ge=1)
    reference_row_count: int = Field(ge=1)

    train_unique_category_count: int = Field(ge=0)
    reference_unique_category_count: int = Field(ge=0)
    shared_category_count: int = Field(ge=0)
    new_category_count: int = Field(ge=0)

    unseen_rate: FiniteFloat = Field(ge=0.0, le=1.0)
    new_category_rate: FiniteFloat | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    train_missing_rate: FiniteFloat = Field(ge=0.0, le=1.0)
    reference_missing_rate: FiniteFloat = Field(ge=0.0, le=1.0)
    missing_rate_delta: FiniteFloat

    psi: FiniteFloat | None = Field(default=None, ge=0.0)
    jensen_shannon_divergence: FiniteFloat | None = Field(
        default=None,
        ge=0.0,
    )
    total_variation_distance: FiniteFloat | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    train_hhi: FiniteFloat | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    reference_hhi: FiniteFloat | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    hhi_delta: FiniteFloat | None = None

    train_top_category_share: FiniteFloat | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    reference_top_category_share: FiniteFloat | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    top_category_share_delta: FiniteFloat | None = None


class CategoricalDriftReport(BaseModel):
    """Drift measurements for multiple categorical features."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    features: tuple[CategoricalDriftMetrics, ...] = ()


class EncoderRobustnessMetrics(BaseModel):
    """Observed behavior of one encoder under a train/reference shift."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    encoder_name: str = Field(min_length=1)
    feature_name: str = Field(min_length=1)

    status: RobustnessStatus

    train_row_count: int = Field(ge=1)
    reference_row_count: int = Field(ge=1)

    unseen_rate: FiniteFloat = Field(ge=0.0, le=1.0)
    new_category_count: int = Field(ge=0)
    missing_rate_delta: FiniteFloat

    train_output_features: int | None = Field(default=None, ge=0)
    reference_output_features: int | None = Field(default=None, ge=0)
    output_feature_count_delta: int | None = None

    train_output_density: FiniteFloat | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    reference_output_density: FiniteFloat | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    output_density_delta: FiniteFloat | None = None

    train_non_finite_rate: FiniteFloat | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    reference_non_finite_rate: FiniteFloat | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    train_memory_bytes: int | None = Field(default=None, ge=0)
    reference_memory_bytes: int | None = Field(default=None, ge=0)

    error_type: str | None = None
    error_message: str | None = None


class EncoderRobustnessReport(BaseModel):
    """Robustness measurements for multiple encoder evaluations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    results: tuple[EncoderRobustnessMetrics, ...] = ()


DEFAULT_RANDOM_STATE: Final[int] = 42
DEFAULT_CV_SPLITS: Final[int] = 5

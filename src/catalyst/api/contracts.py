from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator


class DatasetFormat(StrEnum):
    CSV = "csv"
    PARQUET = "parquet"


class TaskType(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ApiModel(BaseModel):
    """Base transport model: strict fields and immutable request/response contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetReference(ApiModel):
    dataset_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    )


class DatasetUploadMetadata(ApiModel):
    dataset_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    filename: str = Field(min_length=1, max_length=255)
    format: DatasetFormat
    size_bytes: int = Field(ge=0)


class JobResponse(ApiModel):
    job_id: str = Field(min_length=1, max_length=128)
    status: JobStatus
    created_at: datetime


class ProfileRequest(ApiModel):
    dataset_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    target_column: str | None = Field(default=None, min_length=1, max_length=255)


class BenchmarkRequest(ApiModel):
    dataset_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    target_column: str = Field(min_length=1, max_length=255)
    task_type: TaskType
    metric: str = Field(min_length=1, max_length=64)
    categorical_columns: tuple[str, ...] = ()
    numerical_columns: tuple[str, ...] = ()
    cv: int = Field(default=5, ge=2, le=20)
    random_state: int = Field(default=42, ge=0)

    @field_validator("categorical_columns", "numerical_columns")
    @classmethod
    def validate_column_names(cls, columns: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(column.strip() for column in columns)
        if any(not column for column in cleaned):
            raise ValueError("Column names must not be empty.")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("Column names must be unique within each column group.")
        return cleaned

    @model_validator(mode="after")
    def validate_column_groups(self) -> Self:
        overlap = set(self.categorical_columns).intersection(self.numerical_columns)
        if overlap:
            raise ValueError(
                "A column cannot be declared as both categorical and numerical: "
                + ", ".join(sorted(overlap))
            )
        if (
            self.target_column in self.categorical_columns
            or self.target_column in self.numerical_columns
        ):
            raise ValueError("target_column must not be included in feature column groups.")
        return self


class OptimizationConstraints(ApiModel):
    minimum_performance: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)
    maximum_cost: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)
    maximum_risk: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)


class OptimizationWeights(ApiModel):
    performance_weight: FiniteFloat = Field(default=0.50, ge=0.0)
    cost_weight: FiniteFloat = Field(default=0.25, ge=0.0)
    risk_weight: FiniteFloat = Field(default=0.25, ge=0.0)

    @model_validator(mode="after")
    def validate_positive_total(self) -> Self:
        if self.performance_weight + self.cost_weight + self.risk_weight <= 0.0:
            raise ValueError("At least one optimization weight must be positive.")
        return self


class OptimizationRequest(ApiModel):
    benchmark_id: str = Field(min_length=1, max_length=128)
    constraints: OptimizationConstraints = Field(default_factory=OptimizationConstraints)
    weights: OptimizationWeights = Field(default_factory=OptimizationWeights)


class ApiError(ApiModel):
    code: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=512)


class HealthResponse(ApiModel):
    status: str = Field(min_length=1, max_length=32)
    version: str = Field(min_length=1, max_length=64)


class DecisionResponse(ApiModel):
    job_id: str = Field(min_length=1, max_length=128)
    status: JobStatus
    selected_candidate: str | None = None
    policy_fingerprint: str = Field(min_length=1, max_length=128)

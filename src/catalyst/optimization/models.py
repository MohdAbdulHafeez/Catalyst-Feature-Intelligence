from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class OptimizationEvidence(BaseModel):
    """Normalized evidence used for cost and multi-objective analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_name: str = Field(min_length=1)

    performance: FiniteFloat
    performance_direction: str = Field(min_length=1)

    feature_count: int = Field(ge=0)
    memory_bytes: int = Field(ge=0)
    fit_time_s: FiniteFloat = Field(ge=0)
    score_time_s: FiniteFloat = Field(ge=0)

    unseen_rate: FiniteFloat = Field(ge=0.0, le=1.0)
    non_finite_rate: FiniteFloat = Field(ge=0.0, le=1.0)
    missingness_delta_abs: FiniteFloat = Field(ge=0.0, le=1.0)

    cost_score: FiniteFloat = Field(ge=0.0, le=1.0)
    risk_score: FiniteFloat = Field(ge=0.0, le=1.0)


class ParetoPoint(BaseModel):
    """One candidate represented in the optimization objective space."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_name: str = Field(min_length=1)

    performance: FiniteFloat
    cost: FiniteFloat = Field(ge=0.0, le=1.0)
    risk: FiniteFloat = Field(ge=0.0, le=1.0)

    dominated: bool


class ParetoFrontier(BaseModel):
    """Result of deterministic Pareto dominance analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    points: tuple[ParetoPoint, ...] = ()

    @property
    def frontier(self) -> tuple[ParetoPoint, ...]:
        """Return non-dominated points in deterministic input order."""
        return tuple(point for point in self.points if not point.dominated)

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from catalyst.benchmark.models import BenchmarkStatus, ScoreDirection, TrialResult
from catalyst.drift.models import EncoderRobustnessMetrics, RobustnessStatus
from catalyst.optimization.models import OptimizationEvidence


@dataclass(frozen=True, slots=True)
class CostWeights:
    """Relative weights for computational cost components."""

    feature_weight: float = 0.35
    memory_weight: float = 0.35
    fit_time_weight: float = 0.20
    score_time_weight: float = 0.10

    def __post_init__(self) -> None:
        values = (
            self.feature_weight,
            self.memory_weight,
            self.fit_time_weight,
            self.score_time_weight,
        )

        if any(value < 0.0 for value in values):
            raise ValueError("Cost weights must be non-negative.")

        if sum(values) <= 0.0:
            raise ValueError("At least one cost weight must be positive.")


@dataclass(frozen=True, slots=True)
class RiskWeights:
    """Relative weights for observed robustness risk signals."""

    unseen_weight: float = 0.50
    non_finite_weight: float = 0.30
    missingness_weight: float = 0.20

    def __post_init__(self) -> None:
        values = (
            self.unseen_weight,
            self.non_finite_weight,
            self.missingness_weight,
        )

        if any(value < 0.0 for value in values):
            raise ValueError("Risk weights must be non-negative.")

        if sum(values) <= 0.0:
            raise ValueError("At least one risk weight must be positive.")


class OptimizationEvidenceBuilder:
    """
    Convert benchmark and robustness measurements into optimization evidence.

    Computational cost is normalized across the completed candidate
    population. Robustness signals are aggregated across all completed
    categorical features belonging to the same encoder.
    """

    def __init__(
        self,
        *,
        cost_weights: CostWeights | None = None,
        risk_weights: RiskWeights | None = None,
    ) -> None:
        self.cost_weights = cost_weights or CostWeights()
        self.risk_weights = risk_weights or RiskWeights()

    def build(
        self,
        trials: tuple[TrialResult, ...],
        robustness: tuple[EncoderRobustnessMetrics, ...],
        *,
        performance_direction: ScoreDirection = ScoreDirection.MAXIMIZE,
    ) -> tuple[OptimizationEvidence, ...]:
        """Build optimization evidence for completed benchmark trials."""
        completed_trials = tuple(
            trial
            for trial in trials
            if trial.status is BenchmarkStatus.COMPLETED and trial.mean_score is not None
        )

        if not completed_trials:
            return ()

        robustness_groups = self._group_robustness(robustness)
        raw_costs = tuple(_trial_cost_components(trial) for trial in completed_trials)
        normalized_costs = tuple(
            _weighted_cost(values, raw_costs, self.cost_weights) for values in raw_costs
        )

        evidence: list[OptimizationEvidence] = []

        for trial, cost_score in zip(
            completed_trials,
            normalized_costs,
            strict=True,
        ):
            encoder_name = _encoder_name_from_candidate(trial.candidate_name)
            robustness_items = robustness_groups.get(encoder_name, ())

            if robustness_items:
                (
                    unseen_rate,
                    non_finite_rate,
                    missingness_delta_abs,
                ) = _aggregate_risk_signals(robustness_items)
            else:
                # Conservative fallback: an unmeasured robustness signal must
                # not silently become an apparent zero-risk candidate.
                unseen_rate = 1.0
                non_finite_rate = 1.0
                missingness_delta_abs = 1.0

            risk_score = _weighted_risk(
                unseen_rate=unseen_rate,
                non_finite_rate=non_finite_rate,
                missingness_delta_abs=missingness_delta_abs,
                weights=self.risk_weights,
            )

            evidence.append(
                OptimizationEvidence(
                    candidate_name=trial.candidate_name,
                    performance=float(trial.mean_score),
                    performance_direction=performance_direction.value,
                    feature_count=int(trial.peak_n_features or 0),
                    memory_bytes=_total_memory(trial),
                    fit_time_s=float(trial.fit_time_total_s),
                    score_time_s=float(trial.score_time_total_s),
                    unseen_rate=unseen_rate,
                    non_finite_rate=non_finite_rate,
                    missingness_delta_abs=missingness_delta_abs,
                    cost_score=cost_score,
                    risk_score=risk_score,
                )
            )

        return tuple(evidence)

    @staticmethod
    def _group_robustness(
        robustness: tuple[EncoderRobustnessMetrics, ...],
    ) -> dict[str, tuple[EncoderRobustnessMetrics, ...]]:
        grouped: defaultdict[
            str,
            list[EncoderRobustnessMetrics],
        ] = defaultdict(list)

        for result in robustness:
            if result.status is RobustnessStatus.COMPLETED:
                grouped[result.encoder_name].append(result)

        return {encoder_name: tuple(results) for encoder_name, results in grouped.items()}


def _encoder_name_from_candidate(candidate_name: str) -> str:
    """Extract the encoder identifier from a benchmark candidate name."""
    return candidate_name.split("__", 1)[0]


def _aggregate_risk_signals(
    robustness: tuple[EncoderRobustnessMetrics, ...],
) -> tuple[float, float, float]:
    unseen_rate = float(np.mean([float(item.unseen_rate) for item in robustness]))
    non_finite_rate = float(
        np.mean([float(item.reference_non_finite_rate or 0.0) for item in robustness])
    )
    missingness_delta_abs = float(
        np.mean([abs(float(item.missing_rate_delta)) for item in robustness])
    )

    return (
        min(unseen_rate, 1.0),
        min(non_finite_rate, 1.0),
        min(missingness_delta_abs, 1.0),
    )


def _weighted_risk(
    *,
    unseen_rate: float,
    non_finite_rate: float,
    missingness_delta_abs: float,
    weights: RiskWeights,
) -> float:
    total_weight = weights.unseen_weight + weights.non_finite_weight + weights.missingness_weight

    return float(
        (
            weights.unseen_weight * unseen_rate
            + weights.non_finite_weight * non_finite_rate
            + weights.missingness_weight * missingness_delta_abs
        )
        / total_weight
    )


def _trial_cost_components(
    trial: TrialResult,
) -> tuple[float, float, float, float]:
    return (
        float(trial.mean_n_features or 0.0),
        float(_total_memory(trial)),
        float(trial.fit_time_total_s),
        float(trial.score_time_total_s),
    )


def _total_memory(trial: TrialResult) -> int:
    return int(sum(fold.memory_bytes or 0 for fold in trial.folds))


def _weighted_cost(
    values: tuple[float, float, float, float],
    population: tuple[tuple[float, float, float, float], ...],
    weights: CostWeights,
) -> float:
    normalized = tuple(
        _min_max_value(
            value,
            [row[index] for row in population],
        )
        for index, value in enumerate(values)
    )

    total_weight = (
        weights.feature_weight
        + weights.memory_weight
        + weights.fit_time_weight
        + weights.score_time_weight
    )

    return float(
        (
            weights.feature_weight * normalized[0]
            + weights.memory_weight * normalized[1]
            + weights.fit_time_weight * normalized[2]
            + weights.score_time_weight * normalized[3]
        )
        / total_weight
    )


def _min_max_value(
    value: float,
    population: list[float],
) -> float:
    minimum = min(population)
    maximum = max(population)

    if maximum == minimum:
        return 0.0

    return float((value - minimum) / (maximum - minimum))

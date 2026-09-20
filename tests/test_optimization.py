from __future__ import annotations

import pytest

from catalyst.benchmark.models import (
    BenchmarkStatus,
    BenchmarkTask,
    FoldResult,
    ScoreDirection,
    TrialResult,
)
from catalyst.drift.models import (
    EncoderRobustnessMetrics,
    RobustnessStatus,
)
from catalyst.optimization import (
    CostWeights,
    OptimizationEvidenceBuilder,
    RiskWeights,
    build_pareto_frontier,
)


def _trial(
    name: str,
    score: float,
    features: int,
    memory: int,
) -> TrialResult:
    return TrialResult(
        candidate_name=name,
        task=BenchmarkTask.CLASSIFICATION,
        metric_name="accuracy",
        status=BenchmarkStatus.COMPLETED,
        mean_score=score,
        std_score=0.01,
        fit_time_total_s=1.0,
        score_time_total_s=0.1,
        mean_n_features=float(features),
        peak_n_features=features,
        folds=(
            FoldResult(
                fold=0,
                score=score,
                fit_time_s=1.0,
                score_time_s=0.1,
                n_features=features,
                memory_bytes=memory,
            ),
        ),
    )


def _robustness(
    encoder_name: str,
    feature_name: str,
    unseen: float,
) -> EncoderRobustnessMetrics:
    return EncoderRobustnessMetrics(
        encoder_name=encoder_name,
        feature_name=feature_name,
        status=RobustnessStatus.COMPLETED,
        train_row_count=100,
        reference_row_count=100,
        unseen_rate=unseen,
        new_category_count=1 if unseen > 0 else 0,
        missing_rate_delta=0.0,
        train_output_features=4,
        reference_output_features=4,
        output_feature_count_delta=0,
        train_output_density=0.5,
        reference_output_density=0.5,
        output_density_delta=0.0,
        train_non_finite_rate=0.0,
        reference_non_finite_rate=0.0,
        train_memory_bytes=1000,
        reference_memory_bytes=1000,
    )


def test_cost_weights_require_positive_total_weight() -> None:
    with pytest.raises(ValueError, match="positive"):
        CostWeights(
            feature_weight=0.0,
            memory_weight=0.0,
            fit_time_weight=0.0,
            score_time_weight=0.0,
        )


def test_risk_weights_require_positive_total_weight() -> None:
    with pytest.raises(ValueError, match="positive"):
        RiskWeights(
            unseen_weight=0.0,
            non_finite_weight=0.0,
            missingness_weight=0.0,
        )


def test_evidence_builder_normalizes_costs() -> None:
    trials = (
        _trial("one_hot__logistic_regression", 0.85, 100, 10000),
        _trial("hashing__logistic_regression", 0.83, 16, 2000),
    )
    robustness = (
        _robustness("one_hot", "city", 0.0),
        _robustness("hashing", "city", 0.1),
    )

    evidence = OptimizationEvidenceBuilder().build(trials, robustness)

    assert len(evidence) == 2
    assert evidence[0].performance == pytest.approx(0.85)
    assert evidence[1].performance == pytest.approx(0.83)
    assert 0.0 <= evidence[0].cost_score <= 1.0
    assert 0.0 <= evidence[1].cost_score <= 1.0
    assert evidence[0].risk_score == pytest.approx(0.0)
    assert evidence[1].risk_score > 0.0


def test_robustness_is_aggregated_across_features() -> None:
    trials = (_trial("one_hot__logistic_regression", 0.85, 100, 10000),)
    robustness = (
        _robustness("one_hot", "city", 0.0),
        _robustness("one_hot", "segment", 0.4),
    )

    evidence = OptimizationEvidenceBuilder().build(trials, robustness)

    assert len(evidence) == 1
    assert evidence[0].unseen_rate == pytest.approx(0.2)


def test_missing_robustness_is_conservative() -> None:
    trials = (_trial("one_hot__logistic_regression", 0.85, 100, 10000),)

    evidence = OptimizationEvidenceBuilder().build(trials, ())

    assert evidence[0].unseen_rate == pytest.approx(1.0)
    assert evidence[0].non_finite_rate == pytest.approx(1.0)
    assert evidence[0].missingness_delta_abs == pytest.approx(1.0)
    assert evidence[0].risk_score == pytest.approx(1.0)


def test_failed_trials_are_excluded() -> None:
    failed = TrialResult(
        candidate_name="failed__logistic_regression",
        task=BenchmarkTask.CLASSIFICATION,
        metric_name="accuracy",
        status=BenchmarkStatus.FAILED,
        fit_time_total_s=0.0,
        score_time_total_s=0.0,
    )

    evidence = OptimizationEvidenceBuilder().build((failed,), ())

    assert evidence == ()


def test_performance_direction_is_preserved() -> None:
    trial = _trial("one_hot__logistic_regression", 0.85, 10, 1000)

    evidence = OptimizationEvidenceBuilder().build(
        (trial,),
        (),
        performance_direction=ScoreDirection.MAXIMIZE,
    )

    assert evidence[0].performance_direction == "maximize"


def test_pareto_frontier_identifies_non_dominated_candidates() -> None:
    trials = (
        _trial("high_performance__logistic_regression", 0.95, 100, 10000),
        _trial("balanced__logistic_regression", 0.90, 20, 2000),
        _trial("dominated__logistic_regression", 0.85, 100, 10000),
    )
    robustness = (
        _robustness("high_performance", "city", 0.0),
        _robustness("balanced", "city", 0.0),
        _robustness("dominated", "city", 0.2),
    )

    evidence = OptimizationEvidenceBuilder().build(trials, robustness)
    frontier = build_pareto_frontier(evidence)

    assert {point.candidate_name for point in frontier.frontier} == {
        "high_performance__logistic_regression",
        "balanced__logistic_regression",
    }

    dominated = next(
        point
        for point in frontier.points
        if point.candidate_name == "dominated__logistic_regression"
    )
    assert dominated.dominated is True


def test_pareto_requires_maximize_performance_direction() -> None:
    trial = _trial("candidate__logistic_regression", 0.85, 10, 1000)
    evidence = OptimizationEvidenceBuilder().build(
        (trial,),
        (),
        performance_direction=ScoreDirection.MINIMIZE,
    )

    with pytest.raises(ValueError, match="maximize"):
        build_pareto_frontier(evidence)

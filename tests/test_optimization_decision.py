from __future__ import annotations

import pytest

from catalyst.optimization.decision import (
    DecisionEngine,
    SelectionPolicy,
)
from catalyst.optimization.models import (
    OptimizationEvidence,
    ParetoFrontier,
    ParetoPoint,
)


def _evidence(
    name: str,
    performance: float,
    cost: float,
    risk: float,
) -> OptimizationEvidence:
    return OptimizationEvidence(
        candidate_name=name,
        performance=performance,
        performance_direction="maximize",
        feature_count=10,
        memory_bytes=10_000,
        fit_time_s=0.1,
        score_time_s=0.01,
        unseen_rate=risk,
        non_finite_rate=0.0,
        missingness_delta_abs=0.0,
        cost_score=cost,
        risk_score=risk,
    )


def _frontier(*names: str) -> ParetoFrontier:
    return ParetoFrontier(
        points=tuple(
            ParetoPoint(
                candidate_name=name,
                performance=0.8,
                cost=0.2,
                risk=0.2,
                dominated=False,
            )
            for name in names
        )
    )


def test_default_policy_selects_highest_utility() -> None:
    evidence = (
        _evidence("one_hot__logistic_regression", 0.90, 0.80, 0.80),
        _evidence("frequency__logistic_regression", 0.84, 0.20, 0.20),
    )
    decision = DecisionEngine().decide(
        evidence, _frontier(*(item.candidate_name for item in evidence))
    )

    assert decision.selected_candidate == "one_hot__logistic_regression"
    assert decision.is_feasible is True


def test_minimum_performance_excludes_candidate() -> None:
    evidence = (
        _evidence("a__logistic_regression", 0.70, 0.10, 0.10),
        _evidence("b__logistic_regression", 0.90, 0.90, 0.90),
    )
    policy = SelectionPolicy(minimum_performance=0.80)
    decision = DecisionEngine().decide(
        evidence, _frontier(*(item.candidate_name for item in evidence)), policy=policy
    )

    assert decision.selected_candidate == "b__logistic_regression"
    assert decision.considered_candidates[0].eligible is False
    assert decision.considered_candidates[0].exclusion_reason == "below_minimum_performance"


def test_maximum_cost_excludes_candidate() -> None:
    evidence = (
        _evidence("a__logistic_regression", 0.95, 0.90, 0.10),
        _evidence("b__logistic_regression", 0.80, 0.20, 0.20),
    )
    policy = SelectionPolicy(maximum_cost=0.50)
    decision = DecisionEngine().decide(
        evidence, _frontier(*(item.candidate_name for item in evidence)), policy=policy
    )

    assert decision.selected_candidate == "b__logistic_regression"


def test_maximum_risk_excludes_candidate() -> None:
    evidence = (
        _evidence("a__logistic_regression", 0.95, 0.10, 0.90),
        _evidence("b__logistic_regression", 0.80, 0.20, 0.20),
    )
    policy = SelectionPolicy(maximum_risk=0.50)
    decision = DecisionEngine().decide(
        evidence, _frontier(*(item.candidate_name for item in evidence)), policy=policy
    )

    assert decision.selected_candidate == "b__logistic_regression"


def test_no_feasible_candidate_returns_no_selection() -> None:
    evidence = (
        _evidence("a__logistic_regression", 0.70, 0.90, 0.90),
        _evidence("b__logistic_regression", 0.75, 0.80, 0.80),
    )
    policy = SelectionPolicy(minimum_performance=0.90)
    decision = DecisionEngine().decide(
        evidence, _frontier(*(item.candidate_name for item in evidence)), policy=policy
    )

    assert decision.selected_candidate is None
    assert decision.selected_utility is None
    assert decision.is_feasible is False


def test_only_pareto_frontier_candidates_are_considered() -> None:
    evidence = (
        _evidence("frontier__logistic_regression", 0.90, 0.30, 0.20),
        _evidence("dominated__logistic_regression", 0.85, 0.40, 0.30),
    )
    frontier = ParetoFrontier(
        points=(
            ParetoPoint(
                candidate_name="frontier__logistic_regression",
                performance=0.90,
                cost=0.30,
                risk=0.20,
                dominated=False,
            ),
            ParetoPoint(
                candidate_name="dominated__logistic_regression",
                performance=0.85,
                cost=0.40,
                risk=0.30,
                dominated=True,
            ),
        )
    )
    decision = DecisionEngine().decide(evidence, frontier)

    assert decision.frontier_candidate_names == ("frontier__logistic_regression",)
    assert len(decision.considered_candidates) == 1


def test_policy_fingerprint_is_reproducible() -> None:
    policy_a = SelectionPolicy(
        performance_weight=0.6,
        cost_weight=0.2,
        risk_weight=0.2,
        maximum_risk=0.4,
    )
    policy_b = SelectionPolicy(
        performance_weight=0.6,
        cost_weight=0.2,
        risk_weight=0.2,
        maximum_risk=0.4,
    )

    assert policy_a.fingerprint == policy_b.fingerprint


def test_invalid_selection_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="At least one selection weight"):
        SelectionPolicy(performance_weight=0.0, cost_weight=0.0, risk_weight=0.0)


def test_invalid_constraints_are_rejected() -> None:
    with pytest.raises(ValueError, match="maximum_cost"):
        SelectionPolicy(maximum_cost=1.5)

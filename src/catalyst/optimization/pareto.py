from __future__ import annotations

from catalyst.optimization.models import (
    OptimizationEvidence,
    ParetoFrontier,
    ParetoPoint,
)


def build_pareto_frontier(
    evidence: tuple[OptimizationEvidence, ...],
) -> ParetoFrontier:
    """
    Identify non-dominated candidates.

    Phase 7A assumes the benchmark performance metric is a maximize objective.
    Computational cost and robustness risk are minimize objectives.
    """
    for item in evidence:
        if item.performance_direction != "maximize":
            raise ValueError("Pareto analysis in Phase 7A requires a maximize performance metric.")

    points = tuple(
        ParetoPoint(
            candidate_name=item.candidate_name,
            performance=item.performance,
            cost=item.cost_score,
            risk=item.risk_score,
            dominated=False,
        )
        for item in evidence
    )

    dominated_flags = tuple(
        _is_dominated(candidate_index=index, points=points) for index in range(len(points))
    )

    return ParetoFrontier(
        points=tuple(
            point.model_copy(update={"dominated": dominated})
            for point, dominated in zip(
                points,
                dominated_flags,
                strict=True,
            )
        )
    )


def _is_dominated(
    candidate_index: int,
    points: tuple[ParetoPoint, ...],
) -> bool:
    candidate = points[candidate_index]

    for index, competitor in enumerate(points):
        if index == candidate_index:
            continue

        no_worse = (
            competitor.performance >= candidate.performance
            and competitor.cost <= candidate.cost
            and competitor.risk <= candidate.risk
        )

        strictly_better = (
            competitor.performance > candidate.performance
            or competitor.cost < candidate.cost
            or competitor.risk < candidate.risk
        )

        if no_worse and strictly_better:
            return True

    return False

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from catalyst.optimization.models import OptimizationEvidence, ParetoFrontier


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    """Explicit, reproducible policy for selecting from optimization evidence."""

    performance_weight: float = 0.50
    cost_weight: float = 0.25
    risk_weight: float = 0.25
    minimum_performance: float | None = None
    maximum_cost: float | None = None
    maximum_risk: float | None = None

    def __post_init__(self) -> None:
        weights = (
            self.performance_weight,
            self.cost_weight,
            self.risk_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("Selection weights must be non-negative.")
        if sum(weights) <= 0.0:
            raise ValueError("At least one selection weight must be positive.")

        if self.minimum_performance is not None and not isinstance(
            self.minimum_performance, (int, float)
        ):
            raise ValueError("minimum_performance must be numeric.")

        if self.maximum_cost is not None and not 0.0 <= self.maximum_cost <= 1.0:
            raise ValueError("maximum_cost must be between 0 and 1.")

        if self.maximum_risk is not None and not 0.0 <= self.maximum_risk <= 1.0:
            raise ValueError("maximum_risk must be between 0 and 1.")

    @property
    def fingerprint(self) -> str:
        payload = {
            "performance_weight": self.performance_weight,
            "cost_weight": self.cost_weight,
            "risk_weight": self.risk_weight,
            "minimum_performance": self.minimum_performance,
            "maximum_cost": self.maximum_cost,
            "maximum_risk": self.maximum_risk,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class DecisionCandidate:
    candidate_name: str
    performance: float
    cost: float
    risk: float
    utility: float
    eligible: bool
    exclusion_reason: str | None = None


@dataclass(frozen=True, slots=True)
class OptimizationDecision:
    """Deterministic decision output produced from explicit policy constraints."""

    selected_candidate: str | None
    selected_utility: float | None
    policy_fingerprint: str
    considered_candidates: tuple[DecisionCandidate, ...]
    frontier_candidate_names: tuple[str, ...]

    @property
    def is_feasible(self) -> bool:
        return self.selected_candidate is not None


class DecisionEngine:
    """Apply explicit constraints and weights to Pareto candidates."""

    def decide(
        self,
        evidence: tuple[OptimizationEvidence, ...],
        frontier: ParetoFrontier,
        *,
        policy: SelectionPolicy | None = None,
    ) -> OptimizationDecision:
        resolved_policy = policy or SelectionPolicy()
        frontier_names = tuple(point.candidate_name for point in frontier.frontier)
        frontier_name_set = set(frontier_names)

        evidence_by_name = {
            item.candidate_name: item
            for item in evidence
            if item.candidate_name in frontier_name_set
        }

        candidates: list[DecisionCandidate] = []
        for name in sorted(evidence_by_name):
            item = evidence_by_name[name]
            excluded = _exclusion_reason(item, resolved_policy)
            utility = _utility(item, tuple(evidence_by_name.values()), resolved_policy)

            candidates.append(
                DecisionCandidate(
                    candidate_name=name,
                    performance=float(item.performance),
                    cost=float(item.cost_score),
                    risk=float(item.risk_score),
                    utility=utility,
                    eligible=excluded is None,
                    exclusion_reason=excluded,
                )
            )

        eligible = [candidate for candidate in candidates if candidate.eligible]
        selected = max(
            eligible,
            key=lambda candidate: (
                candidate.utility,
                candidate.performance,
                candidate.candidate_name,
            ),
            default=None,
        )

        return OptimizationDecision(
            selected_candidate=selected.candidate_name if selected else None,
            selected_utility=selected.utility if selected else None,
            policy_fingerprint=resolved_policy.fingerprint,
            considered_candidates=tuple(candidates),
            frontier_candidate_names=frontier_names,
        )


def _exclusion_reason(
    item: OptimizationEvidence,
    policy: SelectionPolicy,
) -> str | None:
    if (
        policy.minimum_performance is not None
        and float(item.performance) < policy.minimum_performance
    ):
        return "below_minimum_performance"

    if policy.maximum_cost is not None and float(item.cost_score) > policy.maximum_cost:
        return "above_maximum_cost"

    if policy.maximum_risk is not None and float(item.risk_score) > policy.maximum_risk:
        return "above_maximum_risk"

    return None


def _utility(
    item: OptimizationEvidence,
    evidence: tuple[OptimizationEvidence, ...],
    policy: SelectionPolicy,
) -> float:
    """Normalize performance across the supplied evidence, then maximize utility."""

    performances = [float(candidate.performance) for candidate in evidence]
    minimum = min(performances)
    maximum = max(performances)
    performance = (
        0.0 if maximum == minimum else (float(item.performance) - minimum) / (maximum - minimum)
    )

    total_weight = policy.performance_weight + policy.cost_weight + policy.risk_weight
    return float(
        (
            policy.performance_weight * performance
            + policy.cost_weight * (1.0 - float(item.cost_score))
            + policy.risk_weight * (1.0 - float(item.risk_score))
        )
        / total_weight
    )

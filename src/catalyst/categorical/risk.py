from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .cardinality import CardinalityProfile
from .frequency import FrequencyDistributionProfile
from .ordinality import OrdinalityProfile


class RiskSeverity(StrEnum):
    """Severity assigned to a categorical-feature risk finding."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskType(StrEnum):
    """Machine-readable categorical data risk categories."""

    CARDINALITY = "high_cardinality"
    OHE_EXPLOSION = "ohe_explosion"
    RARE_CATEGORIES = "rare_categories"
    IMBALANCE = "distribution_imbalance"
    MISSINGNESS = "missingness"
    ORDINAL_AMBIGUITY = "ordinality_ambiguity"


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Thresholds for deterministic categorical risk findings."""

    high_cardinality_unique_count: int = 50
    extreme_cardinality_unique_count: int = 1_000
    high_cardinality_ratio: float = 0.05
    high_ohe_dimensions: int = 100
    critical_ohe_dimensions: int = 10_000
    high_rare_category_ratio: float = 0.50
    high_dominant_share: float = 0.90
    high_missing_fraction: float = 0.20
    ordinal_ambiguity_min_categories: int = 3

    def __post_init__(self) -> None:
        if self.high_cardinality_unique_count < 2:
            raise ValueError("high_cardinality_unique_count must be at least 2")
        if self.extreme_cardinality_unique_count < self.high_cardinality_unique_count:
            raise ValueError(
                "extreme_cardinality_unique_count must be >= high_cardinality_unique_count"
            )
        if not 0.0 <= self.high_cardinality_ratio <= 1.0:
            raise ValueError("high_cardinality_ratio must be between 0 and 1")
        if self.high_ohe_dimensions < 2:
            raise ValueError("high_ohe_dimensions must be at least 2")
        if self.critical_ohe_dimensions < self.high_ohe_dimensions:
            raise ValueError("critical_ohe_dimensions must be >= high_ohe_dimensions")
        if not 0.0 <= self.high_rare_category_ratio <= 1.0:
            raise ValueError("high_rare_category_ratio must be between 0 and 1")
        if not 0.0 <= self.high_dominant_share <= 1.0:
            raise ValueError("high_dominant_share must be between 0 and 1")
        if not 0.0 <= self.high_missing_fraction <= 1.0:
            raise ValueError("high_missing_fraction must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class RiskFinding:
    """One deterministic finding produced by the risk analyzer."""

    risk_type: RiskType
    severity: RiskSeverity
    score: float
    title: str
    detail: str
    recommended_actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CategoricalRiskProfile:
    """Aggregate categorical risk assessment with explainable findings."""

    feature_name: str
    overall_score: float
    overall_severity: RiskSeverity
    findings: tuple[RiskFinding, ...]


class CategoricalRiskAnalyzer:
    """Convert categorical intelligence metrics into explainable risk findings."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def analyze(
        self,
        cardinality: CardinalityProfile,
        frequency: FrequencyDistributionProfile,
        ordinality: OrdinalityProfile,
    ) -> CategoricalRiskProfile:
        findings: list[RiskFinding] = []

        if (
            cardinality.unique_count >= self.config.extreme_cardinality_unique_count
            or cardinality.cardinality_ratio >= self.config.high_cardinality_ratio * 4
        ):
            findings.append(
                RiskFinding(
                    risk_type=RiskType.CARDINALITY,
                    severity=RiskSeverity.CRITICAL,
                    score=1.0,
                    title="Extreme categorical cardinality",
                    detail=(
                        f"{cardinality.unique_count:,} unique categories or a "
                        f"{cardinality.cardinality_ratio:.1%} cardinality ratio may "
                        "make OHE inefficient."
                    ),
                    recommended_actions=(
                        "Avoid naive One-Hot Encoding.",
                        "Benchmark hashing or frequency encoding.",
                    ),
                )
            )
        elif (
            cardinality.unique_count >= self.config.high_cardinality_unique_count
            or cardinality.cardinality_ratio >= self.config.high_cardinality_ratio
        ):
            findings.append(
                RiskFinding(
                    risk_type=RiskType.CARDINALITY,
                    severity=RiskSeverity.HIGH,
                    score=0.75,
                    title="High categorical cardinality",
                    detail=(
                        f"{cardinality.unique_count:,} unique categories or a "
                        f"{cardinality.cardinality_ratio:.1%} cardinality ratio may "
                        "make OHE inefficient."
                    ),
                    recommended_actions=("Compare One-Hot against compact encodings.",),
                )
            )

        if cardinality.estimated_one_hot_features >= self.config.critical_ohe_dimensions:
            findings.append(
                RiskFinding(
                    risk_type=RiskType.OHE_EXPLOSION,
                    severity=RiskSeverity.CRITICAL,
                    score=1.0,
                    title="One-Hot dimensionality explosion",
                    detail=(
                        f"Naive OHE could create approximately "
                        f"{cardinality.estimated_one_hot_features:,} encoded features."
                    ),
                    recommended_actions=(
                        "Prefer compact encodings and benchmark memory cost before training.",
                    ),
                )
            )
        elif cardinality.estimated_one_hot_features >= self.config.high_ohe_dimensions:
            findings.append(
                RiskFinding(
                    risk_type=RiskType.OHE_EXPLOSION,
                    severity=RiskSeverity.MEDIUM,
                    score=0.50,
                    title="High One-Hot dimensionality",
                    detail=(
                        f"OHE would create approximately "
                        f"{cardinality.estimated_one_hot_features:,} encoded features."
                    ),
                    recommended_actions=("Estimate memory and benchmark compact alternatives.",),
                )
            )

        if cardinality.rare_category_ratio >= self.config.high_rare_category_ratio:
            findings.append(
                RiskFinding(
                    risk_type=RiskType.RARE_CATEGORIES,
                    severity=RiskSeverity.HIGH,
                    score=0.75,
                    title="Many rare categories",
                    detail=(
                        f"{cardinality.rare_category_ratio:.1%} of categories are currently "
                        "classified as rare under the configured thresholds."
                    ),
                    recommended_actions=(
                        "Evaluate rare-category grouping before high-dimensional encoding.",
                    ),
                )
            )

        if cardinality.dominant_category_share >= self.config.high_dominant_share:
            findings.append(
                RiskFinding(
                    risk_type=RiskType.IMBALANCE,
                    severity=RiskSeverity.MEDIUM,
                    score=0.50,
                    title="Highly concentrated category distribution",
                    detail=(
                        f"The dominant category represents "
                        f"{cardinality.dominant_category_share:.1%} of observations."
                    ),
                    recommended_actions=(
                        "Inspect class/category concentration before interpreting "
                        "downstream importance.",
                    ),
                )
            )

        if cardinality.missing_count > 0:
            missing_fraction = (
                cardinality.missing_count / cardinality.row_count if cardinality.row_count else 0.0
            )
            severity = (
                RiskSeverity.HIGH
                if missing_fraction >= self.config.high_missing_fraction
                else RiskSeverity.LOW
            )
            findings.append(
                RiskFinding(
                    risk_type=RiskType.MISSINGNESS,
                    severity=severity,
                    score=0.75 if severity is RiskSeverity.HIGH else 0.25,
                    title="Missing categorical values",
                    detail=(
                        f"{cardinality.missing_count:,} rows are missing this categorical feature "
                        f"({missing_fraction:.1%} of rows)."
                    ),
                    recommended_actions=("Define an explicit missing-category handling policy.",),
                )
            )

        if (
            not ordinality.order_detected
            and cardinality.unique_count >= self.config.ordinal_ambiguity_min_categories
        ):
            findings.append(
                RiskFinding(
                    risk_type=RiskType.ORDINAL_AMBIGUITY,
                    severity=RiskSeverity.INFO,
                    score=0.0,
                    title="Ordinal semantics not established",
                    detail=(
                        "No reliable ordering signal was detected from dtype or "
                        "supported value vocabularies."
                    ),
                    recommended_actions=(
                        "Treat the feature as nominal unless domain metadata establishes an order.",
                    ),
                )
            )

        overall_score = max((finding.score for finding in findings), default=0.0)
        overall_severity = max(
            (finding.severity for finding in findings),
            key=_severity_rank,
            default=RiskSeverity.INFO,
        )

        return CategoricalRiskProfile(
            feature_name=cardinality.feature_name,
            overall_score=overall_score,
            overall_severity=overall_severity,
            findings=tuple(findings),
        )


def _severity_rank(severity: RiskSeverity) -> int:
    return {
        RiskSeverity.INFO: 0,
        RiskSeverity.LOW: 1,
        RiskSeverity.MEDIUM: 2,
        RiskSeverity.HIGH: 3,
        RiskSeverity.CRITICAL: 4,
    }[severity]

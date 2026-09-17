from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, log
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class CardinalityConfig:
    """Configuration for categorical cardinality and rarity analysis."""

    rare_count_threshold: int = 5
    rare_share_threshold: float = 0.01
    include_missing_as_one_hot_dimension: bool = True

    def __post_init__(self) -> None:
        if self.rare_count_threshold < 1:
            raise ValueError("rare_count_threshold must be at least 1")
        if not 0.0 <= self.rare_share_threshold <= 1.0:
            raise ValueError("rare_share_threshold must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class CardinalityProfile:
    """Measured cardinality and category-concentration characteristics."""

    feature_name: str
    row_count: int
    non_null_count: int
    missing_count: int
    unique_count: int
    cardinality_ratio: float
    dominant_category_share: float
    entropy: float
    normalized_entropy: float
    effective_category_count: float
    singleton_category_count: int
    rare_category_count: int
    rare_category_ratio: float
    estimated_one_hot_features: int


class CardinalityAnalyzer:
    """Compute deterministic, model-agnostic cardinality statistics."""

    def __init__(self, config: CardinalityConfig | None = None) -> None:
        self.config = config or CardinalityConfig()

    def analyze(self, series: pd.Series, feature_name: str | None = None) -> CardinalityProfile:
        """Analyze one categorical pandas Series without mutating it."""
        if not isinstance(series, pd.Series):
            raise TypeError("series must be a pandas Series")

        name = feature_name if feature_name is not None else str(series.name)
        row_count = int(len(series))
        missing_count = int(series.isna().sum())
        non_null_count = row_count - missing_count

        if non_null_count == 0:
            return CardinalityProfile(
                feature_name=name,
                row_count=row_count,
                non_null_count=0,
                missing_count=missing_count,
                unique_count=0,
                cardinality_ratio=0.0,
                dominant_category_share=0.0,
                entropy=0.0,
                normalized_entropy=0.0,
                effective_category_count=0.0,
                singleton_category_count=0,
                rare_category_count=0,
                rare_category_ratio=0.0,
                estimated_one_hot_features=1 if (
                    missing_count > 0 and self.config.include_missing_as_one_hot_dimension
                ) else 0,
            )

        try:
            counts = series.dropna().value_counts(dropna=False)
        except TypeError as exc:
            raise TypeError(
                f"Categorical feature {name!r} contains unhashable values and cannot "
                "be analyzed as a categorical variable."
            ) from exc

        unique_count = int(len(counts))
        cardinality_ratio = unique_count / non_null_count

        probabilities = counts.to_numpy(dtype="float64") / non_null_count
        dominant_category_share = float(probabilities.max(initial=0.0))

        entropy = float(-sum(p * log(p) for p in probabilities if p > 0.0))
        normalized_entropy = (
            entropy / log(unique_count)
            if unique_count > 1 and entropy > 0.0
            else 0.0
        )
        effective_category_count = float(exp(entropy)) if unique_count else 0.0

        singleton_category_count = int((counts == 1).sum())

        # Use an adaptive absolute threshold so very small datasets do not mark
        # every observed category as rare simply because the configured absolute
        # threshold is larger than the sample size.
        adaptive_count_threshold = max(
            1,
            min(
                self.config.rare_count_threshold,
                ceil(non_null_count * self.config.rare_share_threshold),
            ),
        )
        rare_mask = counts <= adaptive_count_threshold
        rare_category_count = int(rare_mask.sum())
        rare_category_ratio = rare_category_count / unique_count if unique_count else 0.0

        estimated_one_hot_features = unique_count
        if missing_count > 0 and self.config.include_missing_as_one_hot_dimension:
            estimated_one_hot_features += 1

        return CardinalityProfile(
            feature_name=name,
            row_count=row_count,
            non_null_count=non_null_count,
            missing_count=missing_count,
            unique_count=unique_count,
            cardinality_ratio=cardinality_ratio,
            dominant_category_share=dominant_category_share,
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            effective_category_count=effective_category_count,
            singleton_category_count=singleton_category_count,
            rare_category_count=rare_category_count,
            rare_category_ratio=rare_category_ratio,
            estimated_one_hot_features=estimated_one_hot_features,
        )

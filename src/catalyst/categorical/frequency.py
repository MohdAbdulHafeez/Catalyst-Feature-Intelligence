from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class FrequencyDistributionConfig:
    """Configuration for category-frequency summaries."""

    top_k: int = 10

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")


@dataclass(frozen=True, slots=True)
class CategoryFrequency:
    """Frequency information for one observed category."""

    category: Any
    count: int
    share: float
    cumulative_share: float


@dataclass(frozen=True, slots=True)
class FrequencyDistributionProfile:
    """Compact description of category-frequency concentration."""

    feature_name: str
    categories: tuple[CategoryFrequency, ...]
    herfindahl_index: float
    top_3_share: float
    top_5_share: float
    top_10_share: float
    unique_category_share: float


class FrequencyDistributionAnalyzer:
    """Analyze how observations are distributed across categories."""

    def __init__(self, config: FrequencyDistributionConfig | None = None) -> None:
        self.config = config or FrequencyDistributionConfig()

    def analyze(
        self,
        series: pd.Series,
        feature_name: str | None = None,
    ) -> FrequencyDistributionProfile:
        """Analyze one categorical feature without modifying the input."""
        if not isinstance(series, pd.Series):
            raise TypeError("series must be a pandas Series")

        name = feature_name if feature_name is not None else str(series.name)
        non_null = series.dropna()
        non_null_count = int(len(non_null))

        if non_null_count == 0:
            return FrequencyDistributionProfile(
                feature_name=name,
                categories=(),
                herfindahl_index=0.0,
                top_3_share=0.0,
                top_5_share=0.0,
                top_10_share=0.0,
                unique_category_share=0.0,
            )

        try:
            counts = non_null.value_counts(dropna=False)
        except TypeError as exc:
            raise TypeError(
                f"Categorical feature {name!r} contains unhashable values and cannot "
                "be analyzed as a categorical variable."
            ) from exc

        shares = counts.to_numpy(dtype="float64") / non_null_count
        cumulative = shares.cumsum()

        categories = tuple(
            CategoryFrequency(
                category=_python_scalar(category),
                count=int(count),
                share=float(share),
                cumulative_share=float(cumulative_share),
            )
            for category, count, share, cumulative_share in zip(
                counts.index[: self.config.top_k],
                counts.to_numpy(dtype="int64")[: self.config.top_k],
                shares[: self.config.top_k],
                cumulative[: self.config.top_k],
                strict=True,
            )
        )

        herfindahl_index = float((shares**2).sum())

        return FrequencyDistributionProfile(
            feature_name=name,
            categories=categories,
            herfindahl_index=herfindahl_index,
            top_3_share=float(shares[:3].sum()),
            top_5_share=float(shares[:5].sum()),
            top_10_share=float(shares[:10].sum()),
            unique_category_share=float((counts == 1).sum() / len(counts)),
        )


def _python_scalar(value: Any) -> Any:
    """Convert NumPy/Pandas scalar labels to standard Python scalars."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value

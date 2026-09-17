from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class OrdinalityConfig:
    """Configuration for deterministic ordinal-signal detection."""

    minimum_ordered_categories: int = 3
    minimum_order_match_ratio: float = 0.75
    maximum_unmatched_fraction: float = 0.20

    def __post_init__(self) -> None:
        if self.minimum_ordered_categories < 2:
            raise ValueError("minimum_ordered_categories must be at least 2")
        if not 0.0 < self.minimum_order_match_ratio <= 1.0:
            raise ValueError("minimum_order_match_ratio must be in (0, 1]")
        if not 0.0 <= self.maximum_unmatched_fraction <= 1.0:
            raise ValueError("maximum_unmatched_fraction must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class OrdinalSignal:
    """One explainable signal supporting or rejecting ordinal interpretation."""

    source: str
    strength: str
    detail: str


@dataclass(frozen=True, slots=True)
class OrdinalityProfile:
    """Evidence-based profile of whether a categorical feature appears ordered."""

    feature_name: str
    is_ordered_categorical_dtype: bool
    order_detected: bool
    confidence: float
    ordered_categories: tuple[Any, ...]
    unmatched_categories: tuple[Any, ...]
    signals: tuple[OrdinalSignal, ...]


class OrdinalityAnalyzer:
    """Detect ordinal semantics without using the target variable."""

    _PATTERNS: tuple[tuple[tuple[str, ...], tuple[str, ...], str], ...] = (
        (
            (
                "very low",
                "low",
                "medium",
                "high",
                "very high",
            ),
            (),
            "intensity scale",
        ),
        (
            (
                "very dissatisfied",
                "dissatisfied",
                "neutral",
                "satisfied",
                "very satisfied",
            ),
            (),
            "satisfaction scale",
        ),
        (
            (
                "very poor",
                "poor",
                "average",
                "good",
                "very good",
            ),
            (),
            "quality scale",
        ),
        (
            ("beginner", "intermediate", "advanced", "expert"),
            (),
            "skill level",
        ),
        (
            ("primary", "middle school", "high school", "bachelor", "master", "phd"),
            (),
            "education progression",
        ),
        (
            ("bronze", "silver", "gold", "platinum"),
            (),
            "tier progression",
        ),
    )

    _NUMERIC_LEVEL_RE = re.compile(r"^(?:level|grade|stage|rank)[ _-]?(\d+)$", re.IGNORECASE)

    def __init__(self, config: OrdinalityConfig | None = None) -> None:
        self.config = config or OrdinalityConfig()

    def analyze(
        self,
        series: pd.Series,
        feature_name: str | None = None,
    ) -> OrdinalityProfile:
        """Analyze one categorical feature using only feature values and dtype."""
        if not isinstance(series, pd.Series):
            raise TypeError("series must be a pandas Series")

        name = feature_name if feature_name is not None else str(series.name)
        values = series.dropna()

        if isinstance(series.dtype, pd.CategoricalDtype) and series.dtype.ordered:
            categories = tuple(_python_scalar(value) for value in series.cat.categories)
            return OrdinalityProfile(
                feature_name=name,
                is_ordered_categorical_dtype=True,
                order_detected=True,
                confidence=1.0,
                ordered_categories=categories,
                unmatched_categories=(),
                signals=(
                    OrdinalSignal(
                        source="pandas_dtype",
                        strength="strong",
                        detail="Column uses an explicitly ordered pandas categorical dtype.",
                    ),
                ),
            )

        if values.empty:
            return OrdinalityProfile(
                feature_name=name,
                is_ordered_categorical_dtype=False,
                order_detected=False,
                confidence=0.0,
                ordered_categories=(),
                unmatched_categories=(),
                signals=(),
            )

        normalized = tuple(str(value).strip().casefold() for value in values.unique())

        best_order: tuple[str, ...] = ()
        best_source = ""
        best_match_ratio = 0.0

        for order, _, source in self._PATTERNS:
            observed = set(normalized)
            matched = [item for item in order if item in observed]
            match_ratio = len(matched) / len(observed) if observed else 0.0
            if (
                len(matched) >= self.config.minimum_ordered_categories
                and match_ratio > best_match_ratio
            ):
                best_order = tuple(item for item in order if item in observed)
                best_source = source
                best_match_ratio = match_ratio

        numeric_levels = self._numeric_level_order(normalized)
        if len(numeric_levels) >= self.config.minimum_ordered_categories:
            level_match_ratio = len(numeric_levels) / len(normalized)
            if level_match_ratio > best_match_ratio:
                best_order = numeric_levels
                best_source = "explicit numeric level labels"
                best_match_ratio = level_match_ratio

        if not best_order:
            return OrdinalityProfile(
                feature_name=name,
                is_ordered_categorical_dtype=False,
                order_detected=False,
                confidence=0.0,
                ordered_categories=(),
                unmatched_categories=tuple(_python_scalar(value) for value in values.unique()),
                signals=(
                    OrdinalSignal(
                        source="value-patterns",
                        strength="none",
                        detail="No supported ordinal vocabulary or ordered dtype was detected.",
                    ),
                ),
            )

        unmatched = tuple(
            _python_scalar(original)
            for original, normalized_value in zip(values.unique(), normalized, strict=True)
            if normalized_value not in best_order
        )
        unmatched_fraction = len(unmatched) / len(normalized) if normalized else 1.0

        if unmatched_fraction > self.config.maximum_unmatched_fraction:
            return OrdinalityProfile(
                feature_name=name,
                is_ordered_categorical_dtype=False,
                order_detected=False,
                confidence=best_match_ratio,
                ordered_categories=tuple(
                    _python_scalar(value)
                    for value in values.unique()
                    if str(value).strip().casefold() in best_order
                ),
                unmatched_categories=unmatched,
                signals=(
                    OrdinalSignal(
                        source="value-patterns",
                        strength="weak",
                        detail=(
                            f"{best_source} matched {best_match_ratio:.0%} of observed categories, "
                            f"but unmatched categories exceed the configured tolerance."
                        ),
                    ),
                ),
            )

        confidence = min(
            0.98,
            0.55 + 0.35 * best_match_ratio + 0.08 * (len(best_order) >= 4),
        )
        ordered_categories = tuple(
            _python_scalar(value)
            for value in values.unique()
            if str(value).strip().casefold() in best_order
        )
        ordered_categories = tuple(
            sorted(
                ordered_categories,
                key=lambda value: best_order.index(str(value).strip().casefold()),
            )
        )

        return OrdinalityProfile(
            feature_name=name,
            is_ordered_categorical_dtype=False,
            order_detected=True,
            confidence=confidence,
            ordered_categories=ordered_categories,
            unmatched_categories=unmatched,
            signals=(
                OrdinalSignal(
                    source="value-patterns",
                    strength="strong" if best_match_ratio >= 0.9 else "moderate",
                    detail=(
                        f"Detected {best_source}; {best_match_ratio:.0%} of observed "
                        "categories match the inferred ordering."
                    ),
                ),
            ),
        )

    @staticmethod
    def _numeric_level_order(values: tuple[str, ...]) -> tuple[str, ...]:
        matches: list[tuple[int, str]] = []
        for value in values:
            match = OrdinalityAnalyzer._NUMERIC_LEVEL_RE.match(value)
            if match:
                matches.append((int(match.group(1)), value))
        if not matches:
            return ()
        return tuple(value for _, value in sorted(matches))


def _python_scalar(value: Any) -> Any:
    """Convert NumPy/Pandas scalar values to standard Python scalars."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value

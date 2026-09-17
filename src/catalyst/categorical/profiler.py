from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from catalyst.profiling import ColumnSemanticType, SchemaProfiler

from .cardinality import CardinalityAnalyzer, CardinalityConfig, CardinalityProfile
from .frequency import (
    FrequencyDistributionAnalyzer,
    FrequencyDistributionConfig,
    FrequencyDistributionProfile,
)


@dataclass(frozen=True, slots=True)
class CategoricalProfilerConfig:
    """Combined configuration for categorical intelligence analysis."""

    cardinality: CardinalityConfig = CardinalityConfig()
    frequency: FrequencyDistributionConfig = FrequencyDistributionConfig()


@dataclass(frozen=True, slots=True)
class CategoricalFeatureProfile:
    """Combined intelligence profile for one categorical feature."""

    feature_name: str
    cardinality: CardinalityProfile
    frequency: FrequencyDistributionProfile


class CategoricalProfiler:
    """Build composable categorical feature intelligence from a DataFrame."""

    def __init__(
        self,
        config: CategoricalProfilerConfig | None = None,
        schema_profiler: SchemaProfiler | None = None,
    ) -> None:
        self.config = config or CategoricalProfilerConfig()
        self.schema_profiler = schema_profiler or SchemaProfiler()
        self.cardinality_analyzer = CardinalityAnalyzer(self.config.cardinality)
        self.frequency_analyzer = FrequencyDistributionAnalyzer(self.config.frequency)

    def profile(
        self,
        frame: pd.DataFrame,
        columns: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[CategoricalFeatureProfile, ...]:
        """Profile categorical columns in deterministic DataFrame column order."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")

        schema = self.schema_profiler.profile(frame)
        available = set(frame.columns)

        if columns is None:
            selected = list(schema.categorical_columns)
        else:
            selected = list(columns)
            unknown = [column for column in selected if column not in available]
            if unknown:
                raise KeyError(f"Unknown categorical feature(s): {unknown}")

            non_categorical = [
                column
                for column in selected
                if next(profile for profile in schema.columns if profile.name == column).semantic_type
                is not ColumnSemanticType.CATEGORICAL
            ]
            if non_categorical:
                raise ValueError(
                    "The following selected columns are not classified as categorical: "
                    f"{non_categorical}"
                )

        return tuple(
            self._profile_column(frame[column], column)
            for column in selected
        )

    def _profile_column(
        self,
        series: pd.Series,
        feature_name: str,
    ) -> CategoricalFeatureProfile:
        return CategoricalFeatureProfile(
            feature_name=feature_name,
            cardinality=self.cardinality_analyzer.analyze(series, feature_name),
            frequency=self.frequency_analyzer.analyze(series, feature_name),
        )

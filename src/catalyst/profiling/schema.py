from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pandas as pd


class ColumnSemanticType(StrEnum):
    """High-level semantic classification used by downstream ML components."""

    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """Immutable profile of one dataset column."""

    name: str
    semantic_type: ColumnSemanticType
    pandas_dtype: str
    row_count: int
    non_null_count: int
    missing_count: int
    missing_fraction: float
    unique_count: int
    unique_ratio: float
    is_constant: bool
    likely_identifier: bool
    identifier_likelihood_score: float
    identifier_signals: tuple[str, ...]
    sample_values: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Immutable schema and quality profile for a pandas DataFrame."""

    row_count: int
    column_count: int
    columns: tuple[ColumnProfile, ...]

    @property
    def numerical_columns(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns if c.semantic_type is ColumnSemanticType.NUMERIC)

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        return tuple(
            c.name for c in self.columns if c.semantic_type is ColumnSemanticType.CATEGORICAL
        )

    @property
    def datetime_columns(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns if c.semantic_type is ColumnSemanticType.DATETIME)

    @property
    def boolean_columns(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns if c.semantic_type is ColumnSemanticType.BOOLEAN)

    @property
    def text_columns(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns if c.semantic_type is ColumnSemanticType.TEXT)

    @property
    def identifier_columns(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns if c.likely_identifier)


@dataclass(frozen=True, slots=True)
class SchemaProfilerConfig:
    """Configuration for deterministic schema inference heuristics.

    Thresholds are heuristics, not probabilities. They are intentionally exposed
    so later experiments can tune them against benchmark datasets.
    """

    identifier_unique_ratio_threshold: float = 0.98
    identifier_name_signal_weight: float = 0.20
    identifier_uniqueness_weight: float = 0.80
    identifier_score_threshold: float = 0.82
    text_unique_ratio_threshold: float = 0.20
    text_mean_length_threshold: float = 40.0
    text_mean_tokens_threshold: float = 6.0
    sample_size: int = 5

    def __post_init__(self) -> None:
        if not 0.0 <= self.identifier_unique_ratio_threshold <= 1.0:
            raise ValueError("identifier_unique_ratio_threshold must be between 0 and 1")
        if not 0.0 <= self.identifier_name_signal_weight <= 1.0:
            raise ValueError("identifier_name_signal_weight must be between 0 and 1")
        if not 0.0 <= self.identifier_uniqueness_weight <= 1.0:
            raise ValueError("identifier_uniqueness_weight must be between 0 and 1")
        if not 0.0 <= self.identifier_score_threshold <= 1.0:
            raise ValueError("identifier_score_threshold must be between 0 and 1")
        if self.sample_size < 0:
            raise ValueError("sample_size must be non-negative")


class SchemaProfiler:
    """Infer stable, explainable column semantics and basic schema risks."""

    _IDENTIFIER_NAME_RE = re.compile(
        r"(^id$|^uuid$|(^|[_-])(id|uuid|identifier|key)$)",
        re.IGNORECASE,
    )

    def __init__(self, config: SchemaProfilerConfig | None = None) -> None:
        self.config = config or SchemaProfilerConfig()

    def profile(self, frame: pd.DataFrame) -> DatasetProfile:
        """Profile a DataFrame without mutating the input."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")

        columns = tuple(self._profile_column(frame, name) for name in frame.columns)
        return DatasetProfile(
            row_count=len(frame),
            column_count=len(frame.columns),
            columns=columns,
        )

    def _profile_column(self, frame: pd.DataFrame, name: Any) -> ColumnProfile:
        series = frame[name]
        row_count = len(series)
        missing_count = int(series.isna().sum())
        non_null_count = row_count - missing_count
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = unique_count / non_null_count if non_null_count else 0.0

        semantic_type = self._infer_semantic_type(series, unique_ratio)
        identifier_score, identifier_signals = self._identifier_score(
            str(name),
            unique_ratio,
            non_null_count,
            row_count,
        )

        sample = tuple(
            self._python_scalar(value) for value in series.dropna().head(self.config.sample_size)
        )

        return ColumnProfile(
            name=str(name),
            semantic_type=semantic_type,
            pandas_dtype=str(series.dtype),
            row_count=row_count,
            non_null_count=non_null_count,
            missing_count=missing_count,
            missing_fraction=missing_count / row_count if row_count else 0.0,
            unique_count=unique_count,
            unique_ratio=unique_ratio,
            is_constant=unique_count <= 1,
            likely_identifier=identifier_score >= self.config.identifier_score_threshold,
            identifier_likelihood_score=identifier_score,
            identifier_signals=tuple(identifier_signals),
            sample_values=sample,
        )

    def _infer_semantic_type(
        self,
        series: pd.Series,
        unique_ratio: float,
    ) -> ColumnSemanticType:
        dtype = series.dtype

        if pd.api.types.is_bool_dtype(dtype):
            return ColumnSemanticType.BOOLEAN

        if pd.api.types.is_datetime64_any_dtype(dtype):
            return ColumnSemanticType.DATETIME

        if pd.api.types.is_numeric_dtype(dtype):
            return ColumnSemanticType.NUMERIC

        if isinstance(dtype, pd.CategoricalDtype):
            return ColumnSemanticType.CATEGORICAL

        if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            non_null = series.dropna()
            if non_null.empty:
                return ColumnSemanticType.CATEGORICAL

            values = non_null.astype(str)
            mean_length = float(values.str.len().mean())
            mean_tokens = float(values.str.split().str.len().mean())

            if unique_ratio >= self.config.text_unique_ratio_threshold and (
                mean_length >= self.config.text_mean_length_threshold
                or mean_tokens >= self.config.text_mean_tokens_threshold
            ):
                return ColumnSemanticType.TEXT

            return ColumnSemanticType.CATEGORICAL

        return ColumnSemanticType.UNKNOWN

    def _identifier_score(
        self,
        name: str,
        unique_ratio: float,
        non_null_count: int,
        row_count: int,
    ) -> tuple[float, list[str]]:
        signals: list[str] = []

        uniqueness_signal = (
            1.0
            if unique_ratio >= self.config.identifier_unique_ratio_threshold
            else unique_ratio / self.config.identifier_unique_ratio_threshold
            if self.config.identifier_unique_ratio_threshold
            else 0.0
        )

        name_signal = 1.0 if self._IDENTIFIER_NAME_RE.search(name) else 0.0

        if unique_ratio >= self.config.identifier_unique_ratio_threshold:
            signals.append("near-unique values")

        if name_signal:
            signals.append("identifier-like column name")

        if row_count > 0 and non_null_count == row_count and unique_ratio == 1.0:
            signals.append("fully unique non-null column")

        score = (
            self.config.identifier_uniqueness_weight * uniqueness_signal
            + self.config.identifier_name_signal_weight * name_signal
        )

        return min(score, 1.0), signals

    @staticmethod
    def _python_scalar(value: Any) -> Any:
        """Convert NumPy/Pandas scalar values to ordinary Python values when possible."""
        if hasattr(value, "item"):
            try:
                return value.item()
            except (ValueError, TypeError):
                pass
        return value

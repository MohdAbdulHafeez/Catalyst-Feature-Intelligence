from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from catalyst.drift.metrics import (
    hhi,
    jensen_shannon_divergence,
    population_stability_index,
    total_variation_distance,
)
from catalyst.drift.models import (
    MISSING_CATEGORY_KEY,
    CategoricalDistributionProfile,
    CategoricalDriftMetrics,
    CategoricalDriftReport,
    CategoryFrequency,
    DriftConfig,
)


class DistributionProfiler:
    """Profile categorical value distributions from pandas Series."""

    def profile(
        self,
        series: pd.Series,
        *,
        feature_name: str | None = None,
    ) -> CategoricalDistributionProfile:
        """Build a distribution profile for one categorical feature."""
        if not isinstance(series, pd.Series):
            raise TypeError("series must be a pandas Series.")

        name = feature_name or (str(series.name) if series.name is not None else "")

        if not name:
            raise ValueError("feature_name must be provided for unnamed Series.")

        if series.empty:
            raise ValueError("Cannot profile an empty Series.")

        counts: Counter[str] = Counter()
        labels: dict[str, str] = {}
        missing_count = 0

        for value in series.tolist():
            category_key, category_label, is_missing = _canonicalize(value)

            if is_missing:
                missing_count += 1
                continue

            counts[category_key] += 1
            labels[category_key] = category_label

        non_missing_count = len(series) - missing_count
        denominator = max(non_missing_count, 1)

        categories = tuple(
            CategoryFrequency(
                category_key=category_key,
                category_label=labels[category_key],
                count=count,
                proportion=count / denominator,
            )
            for category_key, count in sorted(
                counts.items(),
                key=lambda item: item[0],
            )
        )

        return CategoricalDistributionProfile(
            feature_name=name,
            row_count=len(series),
            non_missing_count=non_missing_count,
            missing_count=missing_count,
            missing_rate=missing_count / len(series),
            unique_category_count=len(counts),
            categories=categories,
        )


class CategoricalDriftAnalyzer:
    """
    Compare categorical distributions between training and reference data.

    Distribution metrics exclude missing values so missingness changes are
    measured independently through missing-rate delta.
    """

    def __init__(self, config: DriftConfig | None = None) -> None:
        self.config = config or DriftConfig()
        self.profiler = DistributionProfiler()

    def compare_series(
        self,
        train: pd.Series,
        reference: pd.Series,
        *,
        feature_name: str | None = None,
    ) -> CategoricalDriftMetrics:
        """Compare one training Series with one reference Series."""
        train_profile = self.profiler.profile(
            train,
            feature_name=feature_name,
        )
        reference_profile = self.profiler.profile(
            reference,
            feature_name=train_profile.feature_name,
        )

        return self._compare_profiles(train_profile, reference_profile)

    def compare(
        self,
        train: pd.DataFrame,
        reference: pd.DataFrame,
        *,
        categorical_columns: tuple[str, ...],
    ) -> CategoricalDriftReport:
        """Compare selected categorical columns across two datasets."""
        _validate_dataframe(train, "train")
        _validate_dataframe(reference, "reference")

        if not categorical_columns:
            raise ValueError("At least one categorical column is required.")

        _validate_unique_columns(categorical_columns)
        _validate_columns_exist(train, categorical_columns, "train")
        _validate_columns_exist(reference, categorical_columns, "reference")

        metrics = tuple(
            self.compare_series(
                train[column],
                reference[column],
                feature_name=column,
            )
            for column in categorical_columns
        )

        return CategoricalDriftReport(features=metrics)

    def _compare_profiles(
        self,
        train: CategoricalDistributionProfile,
        reference: CategoricalDistributionProfile,
    ) -> CategoricalDriftMetrics:
        train_map = {category.category_key: category.proportion for category in train.categories}
        reference_map = {
            category.category_key: category.proportion for category in reference.categories
        }

        train_keys = set(train_map)
        reference_keys = set(reference_map)
        shared_keys = train_keys.intersection(reference_keys)
        new_keys = reference_keys.difference(train_keys)

        unseen_count = sum(
            category.count for category in reference.categories if category.category_key in new_keys
        )

        unseen_rate = unseen_count / reference.row_count

        new_category_rate = (
            len(new_keys) / reference.unique_category_count
            if reference.unique_category_count > 0
            else None
        )

        probabilities_train, probabilities_reference = _aligned_probabilities(
            train_map,
            reference_map,
        )

        psi: float | None
        js_divergence: float | None
        total_variation: float | None

        if (
            train.non_missing_count == 0
            or reference.non_missing_count == 0
            or probabilities_train.size == 0
        ):
            psi = None
            js_divergence = None
            total_variation = None
        else:
            psi = population_stability_index(
                probabilities_train,
                probabilities_reference,
                epsilon=self.config.epsilon,
            )
            js_divergence = jensen_shannon_divergence(
                probabilities_train,
                probabilities_reference,
                epsilon=self.config.epsilon,
            )
            total_variation = total_variation_distance(
                probabilities_train,
                probabilities_reference,
            )

        train_hhi = hhi(np.asarray(list(train_map.values()), dtype=float)) if train_map else None
        reference_hhi = (
            hhi(np.asarray(list(reference_map.values()), dtype=float)) if reference_map else None
        )

        hhi_delta = (
            reference_hhi - train_hhi
            if train_hhi is not None and reference_hhi is not None
            else None
        )

        train_top_share = _top_share(train_map.values())
        reference_top_share = _top_share(reference_map.values())

        top_share_delta = (
            reference_top_share - train_top_share
            if train_top_share is not None and reference_top_share is not None
            else None
        )

        return CategoricalDriftMetrics(
            feature_name=train.feature_name,
            train_row_count=train.row_count,
            reference_row_count=reference.row_count,
            train_unique_category_count=train.unique_category_count,
            reference_unique_category_count=reference.unique_category_count,
            shared_category_count=len(shared_keys),
            new_category_count=len(new_keys),
            unseen_rate=unseen_rate,
            new_category_rate=new_category_rate,
            train_missing_rate=train.missing_rate,
            reference_missing_rate=reference.missing_rate,
            missing_rate_delta=reference.missing_rate - train.missing_rate,
            psi=psi,
            jensen_shannon_divergence=js_divergence,
            total_variation_distance=total_variation,
            train_hhi=train_hhi,
            reference_hhi=reference_hhi,
            hhi_delta=hhi_delta,
            train_top_category_share=train_top_share,
            reference_top_category_share=reference_top_share,
            top_category_share_delta=top_share_delta,
        )


def _canonicalize(value: Any) -> tuple[str, str, bool]:
    missing = pd.isna(value)

    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return MISSING_CATEGORY_KEY, "<MISSING>", True

    value_type = f"{type(value).__module__}.{type(value).__qualname__}"

    return (
        f"{value_type}:{value!r}",
        str(value),
        False,
    )


def _aligned_probabilities(
    train: dict[str, float],
    reference: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(set(train).union(reference))

    if not keys:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )

    train_values = np.asarray(
        [train.get(key, 0.0) for key in keys],
        dtype=float,
    )
    reference_values = np.asarray(
        [reference.get(key, 0.0) for key in keys],
        dtype=float,
    )

    return train_values, reference_values


def _top_share(values: Any) -> float | None:
    probabilities = np.asarray(list(values), dtype=float)

    if probabilities.size == 0:
        return None

    return float(np.max(probabilities))


def _validate_dataframe(
    frame: pd.DataFrame,
    parameter_name: str,
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{parameter_name} must be a pandas DataFrame.")

    if frame.empty:
        raise ValueError(f"{parameter_name} must contain at least one row.")

    if frame.columns.duplicated().any():
        raise ValueError(f"{parameter_name} must not contain duplicate column names.")


def _validate_unique_columns(
    columns: tuple[str, ...],
) -> None:
    if len(columns) != len(set(columns)):
        raise ValueError("categorical_columns must not contain duplicates.")


def _validate_columns_exist(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    parameter_name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]

    if missing:
        raise ValueError(f"{parameter_name} is missing categorical columns: {missing}.")

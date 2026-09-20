from __future__ import annotations

import pandas as pd
import pytest

from catalyst.drift import (
    CategoricalDriftAnalyzer,
    DistributionProfiler,
)


def test_distribution_profile_counts_categories_and_missing_values() -> None:
    series = pd.Series(
        ["A", "A", "B", None, "B"],
        name="segment",
    )

    profile = DistributionProfiler().profile(series)

    assert profile.feature_name == "segment"
    assert profile.row_count == 5
    assert profile.non_missing_count == 4
    assert profile.missing_count == 1
    assert profile.missing_rate == pytest.approx(0.2)
    assert profile.unique_category_count == 2

    proportions = {category.category_label: category.proportion for category in profile.categories}

    assert proportions["A"] == pytest.approx(0.5)
    assert proportions["B"] == pytest.approx(0.5)


def test_distribution_profile_preserves_type_information() -> None:
    series = pd.Series([1, "1", 1, "1"], name="mixed")

    profile = DistributionProfiler().profile(series)

    assert profile.unique_category_count == 2
    assert sorted(category.count for category in profile.categories) == [2, 2]


def test_distribution_profiler_rejects_empty_series() -> None:
    with pytest.raises(ValueError, match="empty"):
        DistributionProfiler().profile(
            pd.Series([], dtype=object),
            feature_name="segment",
        )


def test_distribution_profiler_requires_feature_name_for_unnamed_series() -> None:
    with pytest.raises(ValueError, match="feature_name"):
        DistributionProfiler().profile(pd.Series(["A", "B"]))


def test_drift_identical_distributions_are_zero() -> None:
    train = pd.DataFrame(
        {
            "city": ["A", "A", "B", "B"],
        }
    )
    reference = train.copy()

    report = CategoricalDriftAnalyzer().compare(
        train,
        reference,
        categorical_columns=("city",),
    )

    metrics = report.features[0]

    assert metrics.new_category_count == 0
    assert metrics.unseen_rate == pytest.approx(0.0)
    assert metrics.missing_rate_delta == pytest.approx(0.0)
    assert metrics.psi == pytest.approx(0.0)
    assert metrics.jensen_shannon_divergence == pytest.approx(0.0)
    assert metrics.total_variation_distance == pytest.approx(0.0)
    assert metrics.hhi_delta == pytest.approx(0.0)
    assert metrics.top_category_share_delta == pytest.approx(0.0)


def test_drift_detects_new_categories_and_missingness_shift() -> None:
    train = pd.DataFrame(
        {
            "city": ["A", "A", "B", "B"],
        }
    )
    reference = pd.DataFrame(
        {
            "city": ["A", "B", "B", "C", None],
        }
    )

    metrics = (
        CategoricalDriftAnalyzer()
        .compare(
            train,
            reference,
            categorical_columns=("city",),
        )
        .features[0]
    )

    assert metrics.train_unique_category_count == 2
    assert metrics.reference_unique_category_count == 3
    assert metrics.shared_category_count == 2
    assert metrics.new_category_count == 1

    assert metrics.unseen_rate == pytest.approx(0.2)
    assert metrics.new_category_rate == pytest.approx(1 / 3)

    assert metrics.train_missing_rate == pytest.approx(0.0)
    assert metrics.reference_missing_rate == pytest.approx(0.2)
    assert metrics.missing_rate_delta == pytest.approx(0.2)

    assert metrics.psi is not None
    assert metrics.psi > 0.0

    assert metrics.jensen_shannon_divergence is not None
    assert metrics.jensen_shannon_divergence > 0.0

    assert metrics.total_variation_distance is not None
    assert metrics.total_variation_distance > 0.0


def test_drift_handles_all_missing_reference_values() -> None:
    train = pd.DataFrame({"city": ["A", "B", "A"]})
    reference = pd.DataFrame({"city": [None, None, None]})

    metrics = (
        CategoricalDriftAnalyzer()
        .compare(
            train,
            reference,
            categorical_columns=("city",),
        )
        .features[0]
    )

    assert metrics.reference_unique_category_count == 0
    assert metrics.unseen_rate == pytest.approx(0.0)
    assert metrics.reference_missing_rate == pytest.approx(1.0)

    assert metrics.psi is None
    assert metrics.jensen_shannon_divergence is None
    assert metrics.total_variation_distance is None


def test_drift_can_compare_multiple_features() -> None:
    train = pd.DataFrame(
        {
            "city": ["A", "A", "B", "B"],
            "segment": ["X", "Y", "X", "Y"],
        }
    )
    reference = pd.DataFrame(
        {
            "city": ["A", "B", "B", "C"],
            "segment": ["X", "X", "X", "Y"],
        }
    )

    report = CategoricalDriftAnalyzer().compare(
        train,
        reference,
        categorical_columns=("city", "segment"),
    )

    assert len(report.features) == 2
    assert [feature.feature_name for feature in report.features] == [
        "city",
        "segment",
    ]


def test_drift_rejects_missing_column() -> None:
    train = pd.DataFrame({"city": ["A", "B"]})
    reference = pd.DataFrame({"city": ["A", "B"]})

    with pytest.raises(ValueError, match="missing categorical columns"):
        CategoricalDriftAnalyzer().compare(
            train,
            reference,
            categorical_columns=("segment",),
        )


def test_drift_rejects_duplicate_requested_columns() -> None:
    train = pd.DataFrame({"city": ["A", "B"]})
    reference = pd.DataFrame({"city": ["A", "B"]})

    with pytest.raises(ValueError, match="must not contain duplicates"):
        CategoricalDriftAnalyzer().compare(
            train,
            reference,
            categorical_columns=("city", "city"),
        )

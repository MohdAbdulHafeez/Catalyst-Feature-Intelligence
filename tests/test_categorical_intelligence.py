import math

import pandas as pd
import pytest

from catalyst.categorical import (
    CardinalityAnalyzer,
    CardinalityConfig,
    CategoricalProfiler,
    FrequencyDistributionAnalyzer,
    FrequencyDistributionConfig,
)


def test_cardinality_metrics_are_computed_correctly() -> None:
    series = pd.Series(["A", "A", "B", "C", "C", None], name="segment")

    profile = CardinalityAnalyzer().analyze(series)

    assert profile.feature_name == "segment"
    assert profile.row_count == 6
    assert profile.non_null_count == 5
    assert profile.missing_count == 1
    assert profile.unique_count == 3
    assert profile.cardinality_ratio == pytest.approx(0.6)
    assert profile.dominant_category_share == pytest.approx(0.4)
    assert profile.singleton_category_count == 1
    assert profile.rare_category_count == 1
    assert profile.estimated_one_hot_features == 4
    assert profile.entropy == pytest.approx(
        -(0.4 * math.log(0.4) + 0.2 * math.log(0.2) + 0.4 * math.log(0.4))
    )
    assert profile.effective_category_count == pytest.approx(math.exp(profile.entropy))


def test_cardinality_empty_and_all_null_series() -> None:
    empty = pd.Series([], dtype="object", name="empty")
    all_null = pd.Series([None, None], dtype="object", name="all_null")

    empty_profile = CardinalityAnalyzer().analyze(empty)
    null_profile = CardinalityAnalyzer().analyze(all_null)

    assert empty_profile.unique_count == 0
    assert empty_profile.estimated_one_hot_features == 0

    assert null_profile.unique_count == 0
    assert null_profile.missing_count == 2
    assert null_profile.estimated_one_hot_features == 1


def test_cardinality_thresholds_are_configurable() -> None:
    series = pd.Series(["A"] * 10 + ["B"] * 2 + ["C"] * 1, name="category")

    profile = CardinalityAnalyzer(
        CardinalityConfig(rare_count_threshold=1, rare_share_threshold=0.05)
    ).analyze(series)

    assert profile.singleton_category_count == 1
    assert profile.rare_category_count == 1


def test_frequency_distribution_reports_top_categories() -> None:
    series = pd.Series(
        ["A"] * 6 + ["B"] * 2 + ["C"] + ["D"] + [None],
        name="category",
    )

    profile = FrequencyDistributionAnalyzer(FrequencyDistributionConfig(top_k=3)).analyze(series)

    assert [item.category for item in profile.categories] == ["A", "B", "C"]
    assert [item.count for item in profile.categories] == [6, 2, 1]
    assert profile.top_3_share == pytest.approx(0.9)
    assert profile.top_5_share == pytest.approx(1.0)
    assert profile.unique_category_share == pytest.approx(0.5)
    assert profile.herfindahl_index == pytest.approx(0.42)


def test_frequency_empty_series_is_safe() -> None:
    series = pd.Series([], dtype="object", name="category")
    profile = FrequencyDistributionAnalyzer().analyze(series)

    assert profile.categories == ()
    assert profile.herfindahl_index == 0.0
    assert profile.top_10_share == 0.0


def test_categorical_profiler_uses_schema_and_preserves_column_order() -> None:
    frame = pd.DataFrame(
        {
            "numeric": [1, 2, 3],
            "city": ["Hyderabad", "Pune", "Hyderabad"],
            "segment": pd.Series(["A", "B", "A"], dtype="category"),
        }
    )

    profiles = CategoricalProfiler().profile(frame)

    assert [profile.feature_name for profile in profiles] == ["city", "segment"]
    assert profiles[0].cardinality.unique_count == 2
    assert profiles[1].cardinality.unique_count == 2


def test_categorical_profiler_rejects_unknown_or_non_categorical_columns() -> None:
    frame = pd.DataFrame(
        {
            "age": [20, 21],
            "city": ["Hyderabad", "Pune"],
        }
    )
    profiler = CategoricalProfiler()

    with pytest.raises(KeyError, match="Unknown categorical feature"):
        profiler.profile(frame, columns=["missing"])

    with pytest.raises(ValueError, match="not classified as categorical"):
        profiler.profile(frame, columns=["age"])


def test_analyzers_reject_non_series_input() -> None:
    with pytest.raises(TypeError, match="pandas Series"):
        CardinalityAnalyzer().analyze(["A", "B"])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="pandas Series"):
        FrequencyDistributionAnalyzer().analyze(["A", "B"])  # type: ignore[arg-type]

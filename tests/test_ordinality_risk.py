import pandas as pd

from catalyst.categorical import (
    CategoricalProfiler,
    OrdinalityAnalyzer,
    RiskSeverity,
)


def test_explicit_ordered_dtype_is_strong_ordinal_signal() -> None:
    dtype = pd.CategoricalDtype(
        categories=["Low", "Medium", "High"],
        ordered=True,
    )
    series = pd.Series(["Low", "High", "Medium"], dtype=dtype, name="priority")

    profile = OrdinalityAnalyzer().analyze(series)

    assert profile.order_detected is True
    assert profile.is_ordered_categorical_dtype is True
    assert profile.confidence == 1.0
    assert profile.ordered_categories == ("Low", "Medium", "High")


def test_supported_value_vocabulary_can_infer_order() -> None:
    series = pd.Series(
        ["Very Poor", "Good", "Average", "Very Good", "Poor"],
        name="rating",
    )

    profile = OrdinalityAnalyzer().analyze(series)

    assert profile.order_detected is True
    assert profile.ordered_categories == (
        "Very Poor",
        "Poor",
        "Average",
        "Good",
        "Very Good",
    )
    assert profile.unmatched_categories == ()


def test_numeric_level_labels_can_infer_order() -> None:
    series = pd.Series(
        ["Level 3", "Level 1", "Level 2", "Level 4"],
        name="level",
    )

    profile = OrdinalityAnalyzer().analyze(series)

    assert profile.order_detected is True
    assert profile.ordered_categories == ("Level 1", "Level 2", "Level 3", "Level 4")


def test_unknown_values_can_prevent_ordinal_decision() -> None:
    series = pd.Series(
        ["Low", "Medium", "High", "Unknown", "Other"],
        name="priority",
    )

    profile = OrdinalityAnalyzer().analyze(series)

    assert profile.order_detected is False
    assert set(profile.unmatched_categories) == {"Unknown", "Other"}


def test_profiler_exposes_ordinality_and_risk() -> None:
    frame = pd.DataFrame(
        {
            "priority": ["Low"] * 80 + ["Medium"] * 15 + ["High"] * 5,
            "merchant": [f"merchant_{index}" for index in range(100)],
        }
    )

    profiles = CategoricalProfiler().profile(frame)

    priority = next(profile for profile in profiles if profile.feature_name == "priority")
    merchant = next(profile for profile in profiles if profile.feature_name == "merchant")

    assert priority.ordinality.order_detected is True
    assert merchant.cardinality.unique_count == 100
    assert merchant.risk.overall_severity in {
        RiskSeverity.MEDIUM,
        RiskSeverity.HIGH,
        RiskSeverity.CRITICAL,
    }


def test_ohe_explosion_is_reported_as_critical_for_extreme_cardinality() -> None:
    series = pd.Series([f"product_{index}" for index in range(12_000)], name="product")

    profiles = CategoricalProfiler().profile(pd.DataFrame({"product": series}))
    product = profiles[0]

    assert product.risk.overall_severity is RiskSeverity.CRITICAL
    assert any(
        finding.title == "One-Hot dimensionality explosion" for finding in product.risk.findings
    )


def test_missingness_is_reported() -> None:
    series = pd.Series(["A"] * 7 + [None] * 3, name="segment")

    profiles = CategoricalProfiler().profile(pd.DataFrame({"segment": series}))
    segment = profiles[0]

    assert any(finding.title == "Missing categorical values" for finding in segment.risk.findings)


def test_ordinal_ambiguity_is_informational() -> None:
    series = pd.Series(["A", "B", "C", "A"], name="nominal")

    profiles = CategoricalProfiler().profile(pd.DataFrame({"nominal": series}))
    nominal = profiles[0]

    assert any(
        finding.title == "Ordinal semantics not established"
        and finding.severity is RiskSeverity.INFO
        for finding in nominal.risk.findings
    )


def test_empty_categorical_feature_is_profiled_safely() -> None:
    frame = pd.DataFrame({"category": pd.Series([], dtype="object")})

    profiles = CategoricalProfiler().profile(frame)

    assert len(profiles) == 1
    assert profiles[0].feature_name == "category"
    assert profiles[0].cardinality.unique_count == 0
    assert profiles[0].risk.overall_severity is RiskSeverity.INFO

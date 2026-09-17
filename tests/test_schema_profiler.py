import pandas as pd
import pytest

from catalyst.profiling import SchemaProfiler


def test_profile_mixed_schema() -> None:
    frame = pd.DataFrame(
        {
            "age": [20, 21, 22, 23],
            "is_member": [True, False, True, True],
            "city": ["Hyderabad", "Pune", "Hyderabad", "Delhi"],
            "joined_at": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
            "description": [
                "A long description with several meaningful words for profiling.",
                "Another long description that contains enough tokens to be text.",
                "A third description with enough length to trigger text detection.",
                "The final long-form description should be classified as text.",
            ],
        }
    )

    profile = SchemaProfiler().profile(frame)

    assert profile.row_count == 4
    assert profile.column_count == 5
    assert "age" in profile.numerical_columns
    assert "is_member" in profile.boolean_columns
    assert "city" in profile.categorical_columns
    assert "joined_at" in profile.datetime_columns
    assert "description" in profile.text_columns


def test_profile_missingness_and_constant_column() -> None:
    frame = pd.DataFrame(
        {
            "category": ["A", None, "B", "A"],
            "constant": ["x", "x", "x", "x"],
        }
    )

    profile = SchemaProfiler().profile(frame)
    category = next(c for c in profile.columns if c.name == "category")
    constant = next(c for c in profile.columns if c.name == "constant")

    assert category.missing_count == 1
    assert category.missing_fraction == pytest.approx(0.25)
    assert category.unique_count == 2

    assert constant.is_constant is True
    assert constant.unique_count == 1


def test_identifier_detection_uses_multiple_signals() -> None:
    frame = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003", "C004", "C005"],
            "city": ["HYD", "HYD", "PUN", "DEL", "HYD"],
        }
    )

    profile = SchemaProfiler().profile(frame)
    customer_id = next(c for c in profile.columns if c.name == "customer_id")

    assert customer_id.likely_identifier is True
    assert "near-unique values" in customer_id.identifier_signals
    assert "identifier-like column name" in customer_id.identifier_signals
    assert "customer_id" in profile.identifier_columns


def test_invalid_input_is_rejected() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        SchemaProfiler().profile([1, 2, 3])  # type: ignore[arg-type]


def test_non_mutating_profile() -> None:
    frame = pd.DataFrame({"city": ["Hyderabad", "Pune"]})
    before = frame.copy(deep=True)

    SchemaProfiler().profile(frame)

    pd.testing.assert_frame_equal(frame, before)

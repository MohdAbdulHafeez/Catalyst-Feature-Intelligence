from __future__ import annotations

import pandas as pd
import pytest

from catalyst.drift import (
    EncoderRobustnessAnalyzer,
    RobustnessStatus,
)
from catalyst.encoders import EncoderKind


@pytest.fixture
def shifted_data() -> tuple[pd.Series, pd.Series]:
    train = pd.Series(
        ["A", "A", "B", "B", "A", "B"],
        name="city",
    )
    reference = pd.Series(
        ["A", "B", "C", "C", "A", None],
        name="city",
    )

    return train, reference


def test_one_hot_is_robust_to_unseen_categories(
    shifted_data: tuple[pd.Series, pd.Series],
) -> None:
    train, reference = shifted_data

    result = EncoderRobustnessAnalyzer().compare_series(
        train,
        reference,
        encoder_kind=EncoderKind.ONE_HOT,
    )

    assert result.status is RobustnessStatus.COMPLETED
    assert result.unseen_rate == pytest.approx(2 / 6)
    assert result.new_category_count == 1

    assert result.train_output_features == 2
    assert result.reference_output_features == 2
    assert result.output_feature_count_delta == 0

    assert result.reference_non_finite_rate == pytest.approx(0.0)
    assert result.error_message is None


def test_frequency_encoder_maps_unseen_categories_to_zero(
    shifted_data: tuple[pd.Series, pd.Series],
) -> None:
    train, reference = shifted_data

    result = EncoderRobustnessAnalyzer().compare_series(
        train,
        reference,
        encoder_kind=EncoderKind.FREQUENCY,
    )

    assert result.status is RobustnessStatus.COMPLETED
    assert result.unseen_rate == pytest.approx(2 / 6)
    assert result.reference_output_features == 1
    assert result.reference_non_finite_rate == pytest.approx(0.0)
    assert result.error_message is None


def test_ordinal_encoder_keeps_fixed_output_width(
    shifted_data: tuple[pd.Series, pd.Series],
) -> None:
    train, reference = shifted_data

    result = EncoderRobustnessAnalyzer().compare_series(
        train,
        reference,
        encoder_kind=EncoderKind.ORDINAL,
    )

    assert result.status is RobustnessStatus.COMPLETED
    assert result.train_output_features == 1
    assert result.reference_output_features == 1
    assert result.output_feature_count_delta == 0
    assert result.reference_non_finite_rate == pytest.approx(0.0)


def test_hashing_encoder_has_fixed_width_under_category_shift(
    shifted_data: tuple[pd.Series, pd.Series],
) -> None:
    train, reference = shifted_data

    result = EncoderRobustnessAnalyzer().compare_series(
        train,
        reference,
        encoder_kind=EncoderKind.HASHING,
        encoder_params={"n_features": 16},
    )

    assert result.status is RobustnessStatus.COMPLETED
    assert result.train_output_features == 16
    assert result.reference_output_features == 16
    assert result.output_feature_count_delta == 0
    assert result.reference_non_finite_rate == pytest.approx(0.0)


def test_target_mean_requires_training_target(
    shifted_data: tuple[pd.Series, pd.Series],
) -> None:
    train, reference = shifted_data

    result = EncoderRobustnessAnalyzer().compare_series(
        train,
        reference,
        encoder_kind=EncoderKind.TARGET_MEAN,
    )

    assert result.status is RobustnessStatus.FAILED
    assert result.error_type == "ValueError"
    assert result.error_message is not None
    assert "requires y_train" in result.error_message


def test_target_mean_uses_training_target_only(
    shifted_data: tuple[pd.Series, pd.Series],
) -> None:
    train, reference = shifted_data

    y_train = pd.Series(
        [0, 0, 1, 1, 0, 1],
        name="target",
    )

    result = EncoderRobustnessAnalyzer().compare_series(
        train,
        reference,
        encoder_kind=EncoderKind.TARGET_MEAN,
        y_train=y_train,
    )

    assert result.status is RobustnessStatus.COMPLETED
    assert result.train_output_features == 1
    assert result.reference_output_features == 1
    assert result.reference_non_finite_rate == pytest.approx(0.0)


def test_multi_feature_robustness_report() -> None:
    train = pd.DataFrame(
        {
            "city": ["A", "A", "B", "B"],
            "segment": ["X", "Y", "X", "Y"],
        }
    )

    reference = pd.DataFrame(
        {
            "city": ["A", "B", "C", "C"],
            "segment": ["X", "X", "Y", "Z"],
        }
    )

    report = EncoderRobustnessAnalyzer().compare(
        train,
        reference,
        categorical_columns=("city", "segment"),
        encoder_kind=EncoderKind.FREQUENCY,
    )

    assert len(report.results) == 2
    assert [result.feature_name for result in report.results] == [
        "city",
        "segment",
    ]

    assert all(result.status is RobustnessStatus.COMPLETED for result in report.results)

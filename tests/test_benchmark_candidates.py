from __future__ import annotations

import pandas as pd
import pytest

from catalyst.benchmark.candidates import (
    DEFAULT_CLASSIFICATION_MODEL,
    BenchmarkCandidate,
    EncoderSpec,
    build_pipeline,
    generate_classification_candidates,
)
from catalyst.encoders import EncoderKind


def test_generate_classification_candidates() -> None:
    candidates = generate_classification_candidates()

    assert len(candidates) == 5
    assert candidates[0].name == "one_hot__logistic_regression"
    assert candidates[-1].name == "hashing__logistic_regression"


def test_build_pipeline_is_sklearn_compatible() -> None:
    candidate = BenchmarkCandidate(
        name="frequency__logistic_regression",
        encoder=EncoderSpec(
            name="frequency",
            kind=EncoderKind.FREQUENCY,
            params={},
        ),
        model=DEFAULT_CLASSIFICATION_MODEL,
    )

    pipeline = build_pipeline(
        candidate,
        categorical_columns=("city",),
        numerical_columns=("age",),
    )

    X = pd.DataFrame(
        {
            "city": ["Hyderabad", "Delhi", "Hyderabad", "Mumbai", "Delhi", "Mumbai"],
            "age": [20, 21, 22, 23, 24, 25],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1])

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == len(X)
    assert pipeline.named_steps["preprocessor"].get_feature_names_out().size == 2


def test_build_pipeline_rejects_overlapping_columns() -> None:
    candidate = generate_classification_candidates()[0]

    with pytest.raises(ValueError, match="both categorical and numerical"):
        build_pipeline(
            candidate,
            categorical_columns=("city",),
            numerical_columns=("city",),
        )


def test_build_pipeline_rejects_duplicate_categorical_columns() -> None:
    candidate = generate_classification_candidates()[0]

    with pytest.raises(ValueError, match="categorical_columns"):
        build_pipeline(
            candidate,
            categorical_columns=("city", "city"),
        )


def test_target_encoder_is_inside_pipeline() -> None:
    candidates = generate_classification_candidates(
        encoder_specs=(
            EncoderSpec(
                name="target_mean",
                kind=EncoderKind.TARGET_MEAN,
                params={},
            ),
        )
    )

    pipeline = build_pipeline(
        candidates[0],
        categorical_columns=("city",),
        numerical_columns=("age",),
    )

    preprocessor = pipeline.named_steps["preprocessor"]

    assert preprocessor.transformers[0][0] == "categorical"
    assert preprocessor.transformers[0][2] == ["city"]
    assert preprocessor.transformers[0][1].requires_target is True


def test_build_pipeline_drops_columns_not_declared_for_preprocessing() -> None:
    candidate = BenchmarkCandidate(
        name="frequency__logistic_regression",
        encoder=EncoderSpec(
            name="frequency",
            kind=EncoderKind.FREQUENCY,
            params={},
        ),
        model=DEFAULT_CLASSIFICATION_MODEL,
    )

    pipeline = build_pipeline(
        candidate,
        categorical_columns=("city",),
        numerical_columns=("age",),
    )

    X = pd.DataFrame(
        {
            "city": ["Hyderabad", "Delhi", "Hyderabad", "Mumbai"],
            "age": [20, 21, 22, 23],
            "unselected_metadata": ["a", "b", "c", "d"],
        }
    )
    y = pd.Series([0, 1, 0, 1])

    pipeline.fit(X, y)

    feature_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())

    assert feature_names == ["city", "age"]
    assert "unselected_metadata" not in feature_names

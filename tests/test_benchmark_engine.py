from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from catalyst.benchmark import (
    BenchmarkConfig,
    BenchmarkEngine,
    BenchmarkStatus,
    BenchmarkTask,
)


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "city": [
                "Hyderabad",
                "Delhi",
                "Mumbai",
                "Chennai",
                "Hyderabad",
                "Delhi",
                "Mumbai",
                "Chennai",
                "Hyderabad",
                "Delhi",
                "Mumbai",
                "Chennai",
            ],
            "segment": [
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
            ],
            "age": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        }
    )
    y = pd.Series(
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        name="target",
    )

    return X, y


def test_engine_runs_all_default_candidates(
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = classification_data

    engine = BenchmarkEngine(
        BenchmarkConfig(
            cv_splits=3,
            shuffle=True,
            random_state=42,
        )
    )

    result = engine.run(
        X,
        y,
        categorical_columns=("city", "segment"),
        metric="accuracy",
    )

    assert result.summary.task is BenchmarkTask.CLASSIFICATION
    assert result.summary.metric_name == "accuracy"
    assert len(result.trials) == 5

    for trial in result.trials:
        assert trial.status is BenchmarkStatus.COMPLETED
        assert trial.mean_score is not None
        assert trial.std_score is not None
        assert trial.mean_n_features is not None
        assert trial.peak_n_features is not None
        assert len(trial.folds) == 3

        assert trial.fit_time_total_s >= 0
        assert trial.score_time_total_s >= 0

        assert all(fold.n_features > 0 for fold in trial.folds)
        assert all(fold.memory_bytes is not None for fold in trial.folds)


def test_engine_is_reproducible(
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = classification_data

    config = BenchmarkConfig(
        cv_splits=3,
        shuffle=True,
        random_state=42,
    )

    result_a = BenchmarkEngine(config).run(
        X,
        y,
        categorical_columns=("city", "segment"),
        metric="accuracy",
    )

    result_b = BenchmarkEngine(config).run(
        X,
        y,
        categorical_columns=("city", "segment"),
        metric="accuracy",
    )

    scores_a = [tuple(fold.score for fold in trial.folds) for trial in result_a.trials]
    scores_b = [tuple(fold.score for fold in trial.folds) for trial in result_b.trials]

    assert scores_a == scores_b


def test_engine_uses_numeric_columns_automatically(
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = classification_data

    result = BenchmarkEngine(BenchmarkConfig(cv_splits=3)).run(
        X,
        y,
        categorical_columns=("city", "segment"),
        metric="accuracy",
    )

    for trial in result.trials:
        assert trial.mean_n_features is not None
        assert trial.mean_n_features >= 1


def test_engine_accepts_explicit_numeric_columns(
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = classification_data

    result = BenchmarkEngine(BenchmarkConfig(cv_splits=3)).run(
        X,
        y,
        categorical_columns=("city", "segment"),
        numerical_columns=("age",),
        metric="accuracy",
    )

    assert all(trial.status is BenchmarkStatus.COMPLETED for trial in result.trials)


def test_target_mean_encoder_runs_inside_cv_pipeline(
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    from catalyst.benchmark.candidates import (
        BenchmarkCandidate,
        EncoderSpec,
        ModelKind,
        ModelSpec,
    )
    from catalyst.encoders import EncoderKind

    X, y = classification_data

    candidate = BenchmarkCandidate(
        name="target_mean__logistic_regression",
        encoder=EncoderSpec(
            name="target_mean",
            kind=EncoderKind.TARGET_MEAN,
            params={},
        ),
        model=ModelSpec(
            name="logistic_regression",
            kind=ModelKind.LOGISTIC_REGRESSION,
            params={"max_iter": 1000, "random_state": 42},
        ),
    )

    result = BenchmarkEngine(BenchmarkConfig(cv_splits=3)).run(
        X,
        y,
        categorical_columns=("city", "segment"),
        candidates=(candidate,),
        metric="accuracy",
    )

    trial = result.trials[0]

    assert trial.status is BenchmarkStatus.COMPLETED
    assert len(trial.folds) == 3
    assert trial.error_message is None


def test_engine_rejects_missing_categorical_column(
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = classification_data

    with pytest.raises(ValueError, match="not found in X"):
        BenchmarkEngine(BenchmarkConfig(cv_splits=3)).run(
            X,
            y,
            categorical_columns=("does_not_exist",),
        )


def test_engine_rejects_invalid_cv_split_count(
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = classification_data

    with pytest.raises(ValueError, match="cv_splits"):
        BenchmarkEngine(BenchmarkConfig(cv_splits=1)).run(
            X,
            y,
            categorical_columns=("city",),
        )


def test_engine_rejects_insufficient_class_frequency(
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = classification_data

    y = pd.Series(
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        name="target",
    )

    with pytest.raises(ValueError, match="Minimum class count"):
        BenchmarkEngine(BenchmarkConfig(cv_splits=7)).run(
            X,
            y,
            categorical_columns=("city",),
        )


def test_engine_rejects_single_class_target(
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = classification_data
    y = pd.Series(np.zeros(len(X), dtype=int))

    with pytest.raises(ValueError, match="at least two classes"):
        BenchmarkEngine(BenchmarkConfig(cv_splits=3)).run(
            X,
            y,
            categorical_columns=("city",),
        )


def test_engine_rejects_misaligned_target(
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = classification_data
    y = pd.Series([0, 1, 0])

    with pytest.raises(ValueError, match="same number"):
        BenchmarkEngine(BenchmarkConfig(cv_splits=3)).run(
            X,
            y,
            categorical_columns=("city",),
        )

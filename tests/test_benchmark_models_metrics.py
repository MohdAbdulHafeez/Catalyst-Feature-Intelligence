from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.linear_model import LinearRegression, LogisticRegression

from catalyst.benchmark.metrics import MetricDefinition, available_metrics, get_metric
from catalyst.benchmark.models import (
    BenchmarkStatus,
    BenchmarkTask,
    FoldResult,
    ScoreDirection,
    TrialResult,
)


def test_fold_result_rejects_negative_timing() -> None:
    with pytest.raises(ValueError):
        FoldResult(
            fold=0,
            score=0.8,
            fit_time_s=-1.0,
            score_time_s=0.1,
            n_features=4,
        )


def test_trial_result_is_typed_and_immutable() -> None:
    result = TrialResult(
        candidate_name="one_hot",
        task=BenchmarkTask.CLASSIFICATION,
        metric_name="accuracy",
        status=BenchmarkStatus.COMPLETED,
        mean_score=0.9,
        std_score=0.02,
        fit_time_total_s=0.4,
        score_time_total_s=0.1,
        mean_n_features=12.0,
        peak_n_features=12,
        folds=(
            FoldResult(
                fold=0,
                score=0.9,
                fit_time_s=0.2,
                score_time_s=0.05,
                n_features=12,
            ),
        ),
    )

    assert result.mean_score == 0.9

    with pytest.raises(ValidationError):
        result.mean_score = 0.8  # type: ignore[misc]


def test_classification_metric_registry() -> None:
    assert available_metrics(BenchmarkTask.CLASSIFICATION) == (
        "accuracy",
        "f1",
        "roc_auc",
    )


def test_regression_metric_registry() -> None:
    assert available_metrics(BenchmarkTask.REGRESSION) == (
        "r2",
        "mae",
        "rmse",
    )


def test_metric_alias_is_normalized() -> None:
    metric = get_metric("f1_score", BenchmarkTask.CLASSIFICATION)

    assert metric.name == "f1"
    assert metric.direction is ScoreDirection.MAXIMIZE


def test_accuracy_metric() -> None:
    class FixedEstimator:
        def predict(self, X: object) -> np.ndarray:
            return np.array([0, 1, 0, 1])

    X = pd.DataFrame({"x": [0, 1, 2, 3]})
    y = pd.Series([0, 1, 0, 1])

    metric = get_metric("accuracy", BenchmarkTask.CLASSIFICATION)

    assert metric.evaluate(FixedEstimator(), X, y) == pytest.approx(1.0)


def test_roc_auc_metric() -> None:
    X = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5]})
    y = pd.Series([0, 0, 0, 1, 1, 1])

    estimator = LogisticRegression(random_state=42)
    estimator.fit(X, y)

    metric = get_metric("roc_auc", BenchmarkTask.CLASSIFICATION)

    score = metric.evaluate(estimator, X, y)

    assert 0.0 <= score <= 1.0


def test_regression_metrics() -> None:
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0.0, 2.0, 4.0, 6.0])

    estimator = LinearRegression()
    estimator.fit(X, y)

    r2 = get_metric("r2", BenchmarkTask.REGRESSION)
    mae = get_metric("mae", BenchmarkTask.REGRESSION)
    rmse = get_metric("rmse", BenchmarkTask.REGRESSION)

    assert r2.evaluate(estimator, X, y) == pytest.approx(1.0)
    assert mae.evaluate(estimator, X, y) == pytest.approx(0.0)
    assert rmse.evaluate(estimator, X, y) == pytest.approx(0.0)


def test_metric_task_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="not supported"):
        get_metric("accuracy", BenchmarkTask.REGRESSION)


def test_unknown_metric_is_rejected() -> None:
    with pytest.raises(ValueError, match="not supported"):
        get_metric("does_not_exist", BenchmarkTask.CLASSIFICATION)


def test_metric_rejects_non_finite_scores() -> None:
    metric = MetricDefinition(
        name="invalid",
        task=BenchmarkTask.CLASSIFICATION,
        direction=ScoreDirection.MAXIMIZE,
        evaluator=lambda estimator, X, y: float("nan"),
        description="Metric used to verify non-finite score protection.",
    )

    with pytest.raises(ValueError, match="non-finite"):
        metric.evaluate(
            estimator=object(),
            X=pd.DataFrame({"x": [1, 2]}),
            y=pd.Series([0, 1]),
        )

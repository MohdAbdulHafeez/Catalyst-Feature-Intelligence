from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from catalyst.benchmark.models import BenchmarkTask, ScoreDirection

MetricEvaluator = Callable[[Any, Any, Any], float]


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """Immutable definition of a benchmark metric."""

    name: str
    task: BenchmarkTask
    direction: ScoreDirection
    evaluator: MetricEvaluator
    description: str

    def evaluate(self, estimator: Any, X: Any, y: Any) -> float:
        """Evaluate this metric against a fitted estimator."""
        score = float(self.evaluator(estimator, X, y))

        if not np.isfinite(score):
            raise ValueError(f"Metric '{self.name}' produced a non-finite score: {score!r}.")

        return score


def _accuracy(estimator: Any, X: Any, y: Any) -> float:
    return float(accuracy_score(y, estimator.predict(X)))


def _f1_weighted(estimator: Any, X: Any, y: Any) -> float:
    predictions = estimator.predict(X)
    return float(
        f1_score(
            y,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )


def _roc_auc(estimator: Any, X: Any, y: Any) -> float:
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(X))

        if probabilities.ndim != 2:
            raise ValueError("predict_proba must return a 2D array.")

        if probabilities.shape[1] == 2:
            return float(roc_auc_score(y, probabilities[:, 1]))

        return float(
            roc_auc_score(
                y,
                probabilities,
                multi_class="ovr",
                average="weighted",
            )
        )

    if hasattr(estimator, "decision_function"):
        decision = np.asarray(estimator.decision_function(X))

        if decision.ndim == 1:
            return float(roc_auc_score(y, decision))

        return float(
            roc_auc_score(
                y,
                decision,
                multi_class="ovr",
                average="weighted",
            )
        )

    raise TypeError("ROC AUC requires an estimator exposing predict_proba or decision_function.")


def _r2(estimator: Any, X: Any, y: Any) -> float:
    return float(r2_score(y, estimator.predict(X)))


def _mae(estimator: Any, X: Any, y: Any) -> float:
    return float(mean_absolute_error(y, estimator.predict(X)))


def _rmse(estimator: Any, X: Any, y: Any) -> float:
    predictions = estimator.predict(X)
    return float(sqrt(mean_squared_error(y, predictions)))


_METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        name="accuracy",
        task=BenchmarkTask.CLASSIFICATION,
        direction=ScoreDirection.MAXIMIZE,
        evaluator=_accuracy,
        description="Fraction of correctly classified samples.",
    ),
    MetricDefinition(
        name="f1",
        task=BenchmarkTask.CLASSIFICATION,
        direction=ScoreDirection.MAXIMIZE,
        evaluator=_f1_weighted,
        description="Weighted F1 score across classification classes.",
    ),
    MetricDefinition(
        name="roc_auc",
        task=BenchmarkTask.CLASSIFICATION,
        direction=ScoreDirection.MAXIMIZE,
        evaluator=_roc_auc,
        description="ROC AUC using probabilities or decision scores.",
    ),
    MetricDefinition(
        name="r2",
        task=BenchmarkTask.REGRESSION,
        direction=ScoreDirection.MAXIMIZE,
        evaluator=_r2,
        description="Coefficient of determination.",
    ),
    MetricDefinition(
        name="mae",
        task=BenchmarkTask.REGRESSION,
        direction=ScoreDirection.MINIMIZE,
        evaluator=_mae,
        description="Mean absolute error.",
    ),
    MetricDefinition(
        name="rmse",
        task=BenchmarkTask.REGRESSION,
        direction=ScoreDirection.MINIMIZE,
        evaluator=_rmse,
        description="Root mean squared error.",
    ),
)


_ALIASES: dict[str, str] = {
    "f1_score": "f1",
    "roc-auc": "roc_auc",
    "rocauc": "roc_auc",
    "r_squared": "r2",
    "r-squared": "r2",
    "mean_absolute_error": "mae",
    "root_mean_squared_error": "rmse",
}


def _normalize_metric_name(name: str) -> str:
    normalized = name.strip().lower()

    if not normalized:
        raise ValueError("Metric name must not be empty.")

    return _ALIASES.get(normalized, normalized)


def get_metric(name: str, task: BenchmarkTask) -> MetricDefinition:
    """Return a metric compatible with the requested task."""
    normalized_name = _normalize_metric_name(name)

    for metric in _METRICS:
        if metric.name == normalized_name and metric.task is task:
            return metric

    available = ", ".join(available_metrics(task))
    raise ValueError(
        f"Metric '{name}' is not supported for task '{task.value}'. Available metrics: {available}."
    )


def available_metrics(task: BenchmarkTask | None = None) -> tuple[str, ...]:
    """Return canonical metric names, optionally filtered by task."""
    names = [metric.name for metric in _METRICS if task is None or metric.task is task]
    return tuple(names)

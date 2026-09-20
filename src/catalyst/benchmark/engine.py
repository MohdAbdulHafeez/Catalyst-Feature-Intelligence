from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from catalyst.benchmark import candidates as candidate_module
from catalyst.benchmark.metrics import get_metric
from catalyst.benchmark.models import (
    DEFAULT_CV_SPLITS,
    DEFAULT_RANDOM_STATE,
    BenchmarkResult,
    BenchmarkStatus,
    BenchmarkSummary,
    BenchmarkTask,
    FoldResult,
    TrialResult,
)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Configuration controlling one benchmark execution."""

    cv_splits: int = DEFAULT_CV_SPLITS
    shuffle: bool = True
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        if self.cv_splits < 2:
            raise ValueError("cv_splits must be at least 2.")

        if not self.shuffle and self.random_state is not None:
            raise ValueError("random_state must be None when shuffle is False.")


class BenchmarkEngine:
    """
    Execute leakage-safe encoder benchmarks using stratified cross-validation.

    A brand-new sklearn pipeline is constructed for every candidate and every
    fold. This guarantees that supervised encoders such as target mean
    encoding are fitted exclusively on the training partition of that fold.
    """

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        self.config = config or BenchmarkConfig()

    def run(
        self,
        X: pd.DataFrame,
        y: Any,
        *,
        categorical_columns: tuple[str, ...],
        metric: str = "accuracy",
        numerical_columns: tuple[str, ...] | None = None,
        candidates: tuple[candidate_module.BenchmarkCandidate, ...] | None = None,
        task: BenchmarkTask = BenchmarkTask.CLASSIFICATION,
    ) -> BenchmarkResult:
        """Run the configured benchmark and return structured trial results."""
        self._validate_task(task)
        self._validate_input(X, y, categorical_columns)

        metric_definition = get_metric(metric, task)

        resolved_numerical = self._resolve_numerical_columns(
            X,
            categorical_columns,
            numerical_columns,
        )

        resolved_candidates = (
            candidates
            if candidates is not None
            else candidate_module.generate_classification_candidates()
        )

        self._validate_candidates(resolved_candidates)

        splitter = StratifiedKFold(
            n_splits=self.config.cv_splits,
            shuffle=self.config.shuffle,
            random_state=self.config.random_state,
        )

        y_array = np.asarray(y)

        self._validate_class_distribution(y_array)

        summary = BenchmarkSummary(
            task=task,
            metric_name=metric_definition.name,
            score_direction=metric_definition.direction,
            cv_splits=self.config.cv_splits,
            shuffle=self.config.shuffle,
            random_state=self.config.random_state,
        )

        trials = tuple(
            self._benchmark_candidate(
                candidate,
                X,
                y_array,
                categorical_columns=categorical_columns,
                numerical_columns=resolved_numerical,
                splitter=splitter,
                metric_definition=metric_definition,
            )
            for candidate in resolved_candidates
        )

        return BenchmarkResult(summary=summary, trials=trials)

    def _benchmark_candidate(
        self,
        candidate: candidate_module.BenchmarkCandidate,
        X: pd.DataFrame,
        y: np.ndarray,
        *,
        categorical_columns: tuple[str, ...],
        numerical_columns: tuple[str, ...],
        splitter: StratifiedKFold,
        metric_definition: Any,
    ) -> TrialResult:
        fold_results: list[FoldResult] = []
        total_fit_time = 0.0
        total_score_time = 0.0
        first_error: tuple[str, str] | None = None

        try:
            split_iterator = splitter.split(X, y)

            for fold_index, (train_idx, valid_idx) in enumerate(split_iterator):
                try:
                    X_train = X.iloc[train_idx]
                    X_valid = X.iloc[valid_idx]
                    y_train = y[train_idx]
                    y_valid = y[valid_idx]

                    # CRITICAL: a fresh pipeline is created for every fold.
                    pipeline = candidate_module.build_pipeline(
                        candidate,
                        categorical_columns=categorical_columns,
                        numerical_columns=numerical_columns,
                    )

                    fit_start = perf_counter()
                    pipeline.fit(X_train, y_train)
                    fit_time = perf_counter() - fit_start

                    score_start = perf_counter()
                    score = metric_definition.evaluate(
                        pipeline,
                        X_valid,
                        y_valid,
                    )
                    score_time = perf_counter() - score_start

                    preprocessor = pipeline.named_steps["preprocessor"]
                    transformed_valid = preprocessor.transform(X_valid)

                    n_features = int(len(preprocessor.get_feature_names_out()))
                    memory_bytes = _estimate_matrix_memory(transformed_valid)

                    fold_results.append(
                        FoldResult(
                            fold=fold_index,
                            score=score,
                            fit_time_s=fit_time,
                            score_time_s=score_time,
                            n_features=n_features,
                            memory_bytes=memory_bytes,
                        )
                    )

                    total_fit_time += fit_time
                    total_score_time += score_time

                except Exception as exc:
                    if first_error is None:
                        first_error = (
                            type(exc).__name__,
                            f"fold {fold_index}: {exc}",
                        )

            if first_error is not None:
                return TrialResult(
                    candidate_name=candidate.name,
                    task=metric_definition.task,
                    metric_name=metric_definition.name,
                    status=BenchmarkStatus.FAILED,
                    fit_time_total_s=total_fit_time,
                    score_time_total_s=total_score_time,
                    folds=tuple(fold_results),
                    error_type=first_error[0],
                    error_message=first_error[1],
                )

            if not fold_results:
                return TrialResult(
                    candidate_name=candidate.name,
                    task=metric_definition.task,
                    metric_name=metric_definition.name,
                    status=BenchmarkStatus.FAILED,
                    fit_time_total_s=total_fit_time,
                    score_time_total_s=total_score_time,
                    error_type="BenchmarkExecutionError",
                    error_message="No cross-validation fold completed.",
                )

            scores = np.asarray(
                [fold.score for fold in fold_results],
                dtype=float,
            )
            feature_counts = np.asarray(
                [fold.n_features for fold in fold_results],
                dtype=float,
            )

            return TrialResult(
                candidate_name=candidate.name,
                task=metric_definition.task,
                metric_name=metric_definition.name,
                status=BenchmarkStatus.COMPLETED,
                mean_score=float(np.mean(scores)),
                std_score=float(np.std(scores, ddof=0)),
                fit_time_total_s=total_fit_time,
                score_time_total_s=total_score_time,
                mean_n_features=float(np.mean(feature_counts)),
                peak_n_features=int(np.max(feature_counts)),
                folds=tuple(fold_results),
            )

        except Exception as exc:
            return TrialResult(
                candidate_name=candidate.name,
                task=metric_definition.task,
                metric_name=metric_definition.name,
                status=BenchmarkStatus.FAILED,
                fit_time_total_s=total_fit_time,
                score_time_total_s=total_score_time,
                folds=tuple(fold_results),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    def _resolve_numerical_columns(
        self,
        X: pd.DataFrame,
        categorical_columns: tuple[str, ...],
        numerical_columns: tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        if numerical_columns is not None:
            self._validate_columns_exist(
                X,
                numerical_columns,
                "numerical_columns",
            )

            overlap = set(categorical_columns).intersection(numerical_columns)
            if overlap:
                raise ValueError(
                    f"Columns cannot be both categorical and numerical: {sorted(overlap)}."
                )

            return numerical_columns

        categorical_set = set(categorical_columns)

        return tuple(
            column
            for column in X.columns
            if column not in categorical_set and pd.api.types.is_numeric_dtype(X[column])
        )

    @staticmethod
    def _validate_task(task: BenchmarkTask) -> None:
        if task is not BenchmarkTask.CLASSIFICATION:
            raise NotImplementedError(
                "The benchmark engine currently supports classification only."
            )

    @staticmethod
    def _validate_input(
        X: pd.DataFrame,
        y: Any,
        categorical_columns: tuple[str, ...],
    ) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if X.empty:
            raise ValueError("X must contain at least one row and one column.")

        if X.columns.duplicated().any():
            raise ValueError("X must not contain duplicate column names.")

        if not all(isinstance(column, str) for column in X.columns):
            raise TypeError("All X column names must be strings.")

        if not categorical_columns:
            raise ValueError("At least one categorical column is required.")

        if len(categorical_columns) != len(set(categorical_columns)):
            raise ValueError("categorical_columns must not contain duplicates.")

        BenchmarkEngine._validate_columns_exist(
            X,
            categorical_columns,
            "categorical_columns",
        )

        y_array = np.asarray(y)

        if y_array.ndim != 1:
            raise ValueError("y must be one-dimensional.")

        if len(y_array) != len(X):
            raise ValueError("X and y must contain the same number of rows.")

        if len(y_array) < 2:
            raise ValueError("At least two target observations are required.")

        if pd.isna(y_array).any():
            raise ValueError("y must not contain missing values.")

    @staticmethod
    def _validate_columns_exist(
        X: pd.DataFrame,
        columns: tuple[str, ...],
        parameter_name: str,
    ) -> None:
        missing = [column for column in columns if column not in X.columns]

        if missing:
            raise ValueError(f"{parameter_name} contains columns not found in X: {missing}.")

    def _validate_class_distribution(self, y: np.ndarray) -> None:
        _, counts = np.unique(y, return_counts=True)

        if len(counts) < 2:
            raise ValueError("Classification benchmarks require at least two classes.")

        minimum_class_count = int(np.min(counts))

        if minimum_class_count < self.config.cv_splits:
            raise ValueError(
                "Each class must contain at least cv_splits observations. "
                f"Minimum class count is {minimum_class_count}; "
                f"cv_splits is {self.config.cv_splits}."
            )

    @staticmethod
    def _validate_candidates(
        candidates: tuple[candidate_module.BenchmarkCandidate, ...],
    ) -> None:
        if not candidates:
            raise ValueError("At least one benchmark candidate is required.")

        names = [candidate.name for candidate in candidates]

        if len(names) != len(set(names)):
            raise ValueError("Benchmark candidate names must be unique.")


def _estimate_matrix_memory(matrix: Any) -> int:
    """
    Estimate the in-memory representation size of a transformed matrix.

    This is intentionally a representation-size proxy, not process peak RSS.
    It supports both dense NumPy arrays and scipy-like sparse matrices without
    requiring scipy to be imported directly by CATALYST.
    """
    if hasattr(matrix, "data") and hasattr(matrix, "indices") and hasattr(matrix, "indptr"):
        return int(
            np.asarray(matrix.data).nbytes
            + np.asarray(matrix.indices).nbytes
            + np.asarray(matrix.indptr).nbytes
        )

    return int(np.asarray(matrix).nbytes)

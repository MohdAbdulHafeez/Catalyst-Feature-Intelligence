from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from catalyst.drift.distribution import CategoricalDriftAnalyzer
from catalyst.drift.models import (
    EncoderRobustnessMetrics,
    EncoderRobustnessReport,
    RobustnessStatus,
)
from catalyst.encoders import EncoderKind, create_encoder


class EncoderRobustnessAnalyzer:
    """
    Measure categorical encoder behavior under distribution shift.

    The encoder is fitted only on the training feature and is then applied
    independently to training and reference data. Reference targets are never
    used during the robustness evaluation.
    """

    def compare_series(
        self,
        train: pd.Series,
        reference: pd.Series,
        *,
        encoder_kind: EncoderKind | str,
        encoder_params: dict[str, Any] | None = None,
        y_train: Any = None,
        feature_name: str | None = None,
    ) -> EncoderRobustnessMetrics:
        """Evaluate one encoder against one shifted categorical feature."""
        if not isinstance(train, pd.Series):
            raise TypeError("train must be a pandas Series.")

        if not isinstance(reference, pd.Series):
            raise TypeError("reference must be a pandas Series.")

        if train.empty or reference.empty:
            raise ValueError("train and reference must both contain at least one row.")

        name = feature_name or (str(train.name) if train.name is not None else "")

        if not name:
            raise ValueError("feature_name must be provided for unnamed Series.")

        encoder_name = EncoderKind(encoder_kind).value

        drift = CategoricalDriftAnalyzer().compare_series(
            train,
            reference,
            feature_name=name,
        )

        params = dict(encoder_params or {})

        try:
            encoder = create_encoder(
                EncoderKind(encoder_kind),
                **params,
            )

            train_frame = train.to_frame(name=name)
            reference_frame = reference.to_frame(name=name)

            if encoder.requires_target:
                if y_train is None:
                    raise ValueError(f"{encoder.__class__.__name__} requires y_train.")

                if len(y_train) != len(train):
                    raise ValueError("y_train must contain the same number of rows as train.")

                encoder.fit(train_frame, y_train)
            else:
                encoder.fit(train_frame)

            train_encoded = encoder.transform(train_frame)
            reference_encoded = encoder.transform(reference_frame)

            train_summary = _summarize_output(train_encoded)
            reference_summary = _summarize_output(reference_encoded)

            output_feature_count_delta = reference_summary.n_features - train_summary.n_features

            output_density_delta = (
                reference_summary.density - train_summary.density
                if train_summary.density is not None and reference_summary.density is not None
                else None
            )

            return EncoderRobustnessMetrics(
                encoder_name=encoder_name,
                feature_name=name,
                status=RobustnessStatus.COMPLETED,
                train_row_count=len(train),
                reference_row_count=len(reference),
                unseen_rate=drift.unseen_rate,
                new_category_count=drift.new_category_count,
                missing_rate_delta=drift.missing_rate_delta,
                train_output_features=train_summary.n_features,
                reference_output_features=reference_summary.n_features,
                output_feature_count_delta=output_feature_count_delta,
                train_output_density=train_summary.density,
                reference_output_density=reference_summary.density,
                output_density_delta=output_density_delta,
                train_non_finite_rate=train_summary.non_finite_rate,
                reference_non_finite_rate=reference_summary.non_finite_rate,
                train_memory_bytes=train_summary.memory_bytes,
                reference_memory_bytes=reference_summary.memory_bytes,
            )

        except Exception as exc:
            return EncoderRobustnessMetrics(
                encoder_name=encoder_name,
                feature_name=name,
                status=RobustnessStatus.FAILED,
                train_row_count=len(train),
                reference_row_count=len(reference),
                unseen_rate=drift.unseen_rate,
                new_category_count=drift.new_category_count,
                missing_rate_delta=drift.missing_rate_delta,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    def compare(
        self,
        train: pd.DataFrame,
        reference: pd.DataFrame,
        *,
        categorical_columns: tuple[str, ...],
        encoder_kind: EncoderKind | str,
        encoder_params: dict[str, Any] | None = None,
        y_train: Any = None,
    ) -> EncoderRobustnessReport:
        """Evaluate one encoder independently across multiple features."""
        if not isinstance(train, pd.DataFrame):
            raise TypeError("train must be a pandas DataFrame.")

        if not isinstance(reference, pd.DataFrame):
            raise TypeError("reference must be a pandas DataFrame.")

        if not categorical_columns:
            raise ValueError("At least one categorical column is required.")

        results = tuple(
            self.compare_series(
                train[column],
                reference[column],
                encoder_kind=encoder_kind,
                encoder_params=encoder_params,
                y_train=y_train,
                feature_name=column,
            )
            for column in categorical_columns
        )

        return EncoderRobustnessReport(results=results)


class _OutputSummary:
    """Internal representation summary for encoded output."""

    def __init__(
        self,
        *,
        n_features: int,
        density: float | None,
        non_finite_rate: float,
        memory_bytes: int,
    ) -> None:
        self.n_features = n_features
        self.density = density
        self.non_finite_rate = non_finite_rate
        self.memory_bytes = memory_bytes


def _summarize_output(matrix: Any) -> _OutputSummary:
    """Summarize dense or sparse encoder output without unnecessary densification."""
    shape = getattr(matrix, "shape", None)

    if shape is None or len(shape) != 2:
        array = np.asarray(matrix)

        if array.ndim != 2:
            raise ValueError("Encoder output must be a two-dimensional matrix.")

        total_values = int(array.size)
        non_finite_count = int((~np.isfinite(array)).sum())

        return _OutputSummary(
            n_features=int(array.shape[1]),
            density=(float(np.count_nonzero(array) / total_values) if total_values > 0 else 0.0),
            non_finite_rate=(non_finite_count / total_values if total_values > 0 else 0.0),
            memory_bytes=int(array.nbytes),
        )

    rows = int(shape[0])
    columns = int(shape[1])
    total_values = rows * columns

    if _is_sparse_matrix(matrix):
        data = np.asarray(matrix.data)

        non_finite_count = int((~np.isfinite(data)).sum())

        stored_values = int(data.size)

        return _OutputSummary(
            n_features=columns,
            density=(float(stored_values / total_values) if total_values > 0 else 0.0),
            non_finite_rate=(non_finite_count / total_values if total_values > 0 else 0.0),
            memory_bytes=_sparse_memory_bytes(matrix),
        )

    array = np.asarray(matrix)

    if array.ndim != 2:
        raise ValueError("Encoder output must be a two-dimensional matrix.")

    non_finite_count = int((~np.isfinite(array)).sum())

    return _OutputSummary(
        n_features=int(array.shape[1]),
        density=(float(np.count_nonzero(array) / total_values) if total_values > 0 else 0.0),
        non_finite_rate=(non_finite_count / total_values if total_values > 0 else 0.0),
        memory_bytes=int(array.nbytes),
    )


def _is_sparse_matrix(matrix: Any) -> bool:
    return all(hasattr(matrix, attribute) for attribute in ("data", "indices", "indptr"))


def _sparse_memory_bytes(matrix: Any) -> int:
    return int(
        np.asarray(matrix.data).nbytes
        + np.asarray(matrix.indices).nbytes
        + np.asarray(matrix.indptr).nbytes
    )

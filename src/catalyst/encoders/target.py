from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from .base import CategoricalEncoder


class TargetMeanCategoricalEncoder(CategoricalEncoder):
    """Smoothed target-mean encoder that learns only from fitted training rows.

    Put this transformer inside the same sklearn Pipeline as the estimator when
    evaluating with cross-validation. That makes each fold fit category
    statistics on its training partition only.
    """

    requires_target = True

    def __init__(
        self,
        *,
        smoothing: float = 10.0,
        min_samples_leaf: int = 1,
        handle_missing: str = "global_mean",
    ) -> None:
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.handle_missing = handle_missing

    def _fit(self, X: Any, y: Any = None) -> TargetMeanCategoricalEncoder:
        if self.smoothing <= 0:
            raise ValueError("smoothing must be greater than zero.")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")
        if self.handle_missing not in {"global_mean", "learned"}:
            raise ValueError("handle_missing must be 'global_mean' or 'learned'.")
        if y is None:
            raise ValueError("TargetMeanCategoricalEncoder requires y during fit.")

        frame = _as_frame(X)
        target = _as_numeric_target(y)
        if len(frame) != len(target):
            raise ValueError("X and y must contain the same number of rows.")
        if target.isna().any():
            raise ValueError("Target values cannot contain missing values.")

        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.global_mean_ = float(target.mean())
        self.mappings_: dict[str, dict[Any, float]] = {}

        target_array = target.to_numpy(dtype=np.float64, copy=False)
        for column in frame.columns:
            keys = frame[column].map(_category_key)
            stats = (
                pd.DataFrame({"key": keys, "target": target_array})
                .groupby("key")["target"]
                .agg(["mean", "count"])
            )
            counts = stats["count"].astype(float)
            weight = 1.0 / (1.0 + np.exp(-(counts - self.min_samples_leaf) / self.smoothing))
            smoothed = weight * stats["mean"] + (1.0 - weight) * self.global_mean_
            self.mappings_[str(column)] = {key: float(value) for key, value in smoothed.items()}

        self._is_fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "_is_fitted")
        frame = _as_frame(X)
        _validate_columns(frame, self.feature_names_in_)

        encoded = np.empty((len(frame), frame.shape[1]), dtype=np.float64)
        for index, column in enumerate(frame.columns):
            mapping = self.mappings_[str(column)]
            encoded[:, index] = [
                self.global_mean_
                if pd.isna(value) and self.handle_missing == "global_mean"
                else mapping.get(_category_key(value), self.global_mean_)
                for value in frame[column]
            ]
        return encoded


def _as_numeric_target(y: Any) -> pd.Series:
    series = pd.Series(y, copy=False)
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(
            "TargetMeanCategoricalEncoder currently requires a numeric target. "
            "Encode classification labels before fitting."
        )
    return series.astype(float)


def _category_key(value: Any) -> Any:
    if pd.isna(value):
        return ("__CATALYST_MISSING__",)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _validate_columns(frame: pd.DataFrame, fitted: np.ndarray) -> None:
    if list(frame.columns.astype(str)) != list(fitted.astype(str)):
        raise ValueError("Input columns must match the columns seen during fit.")


def _as_frame(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.Series):
        return X.to_frame()
    if isinstance(X, pd.DataFrame):
        return X
    raise TypeError("X must be a pandas Series or pandas DataFrame.")

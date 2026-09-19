from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from .base import CategoricalEncoder


class FrequencyCategoricalEncoder(CategoricalEncoder):
    """Encode categories using frequencies learned from the fit partition."""

    def __init__(self, *, normalize: bool = True, handle_missing: str = "frequency") -> None:
        self.normalize = normalize
        self.handle_missing = handle_missing

    def _fit(self, X: Any, y: Any = None) -> FrequencyCategoricalEncoder:
        if self.handle_missing not in {"frequency", "zero"}:
            raise ValueError("handle_missing must be 'frequency' or 'zero'.")

        frame = _as_frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.mappings_: dict[str, dict[Any, float]] = {}

        for column in frame.columns:
            series = frame[column]
            denominator = len(series) if self.normalize else 1
            mapping: dict[Any, float] = {}
            for category, count in series.value_counts(dropna=False).items():
                mapping[_category_key(category)] = float(count / denominator)
            self.mappings_[str(column)] = mapping

        self._is_fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "_is_fitted")
        frame = _as_frame(X)
        _validate_columns(frame, self.feature_names_in_)

        encoded = np.zeros((len(frame), frame.shape[1]), dtype=np.float64)
        for index, column in enumerate(frame.columns):
            mapping = self.mappings_[str(column)]
            encoded[:, index] = [
                0.0
                if pd.isna(value) and self.handle_missing == "zero"
                else mapping.get(_category_key(value), 0.0)
                for value in frame[column]
            ]
        return encoded


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

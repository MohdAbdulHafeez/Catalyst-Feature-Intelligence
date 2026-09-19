from __future__ import annotations

from contextlib import suppress
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.utils.validation import check_is_fitted

from .base import CategoricalEncoder


class HashingCategoricalEncoder(CategoricalEncoder):
    """Fixed-width hashing encoder suitable for high-cardinality categoricals."""

    def __init__(self, *, n_features: int = 64, alternate_sign: bool = False) -> None:
        self.n_features = n_features
        self.alternate_sign = alternate_sign

    def _fit(self, X: Any, y: Any = None) -> HashingCategoricalEncoder:
        if self.n_features < 2:
            raise ValueError("n_features must be at least 2.")

        frame = _as_frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.hasher_ = FeatureHasher(
            n_features=self.n_features,
            input_type="string",
            alternate_sign=self.alternate_sign,
        )
        self._is_fitted = True
        return self

    def transform(self, X: Any) -> Any:
        check_is_fitted(self, "_is_fitted")
        frame = _as_frame(X)
        _validate_columns(frame, self.feature_names_in_)

        token_rows = [
            [
                f"{column}={_stable_string(value)}"
                for column, value in zip(frame.columns, row, strict=True)
            ]
            for row in frame.itertuples(index=False, name=None)
        ]
        return self.hasher_.transform(token_rows)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "_is_fitted")
        return np.asarray([f"hash_{i}" for i in range(self.n_features)], dtype=object)


def _stable_string(value: Any) -> str:
    if pd.isna(value):
        return "__CATALYST_MISSING__"
    if hasattr(value, "item"):
        with suppress(TypeError, ValueError):
            value = value.item()
    return str(value)


def _validate_columns(frame: pd.DataFrame, fitted: np.ndarray) -> None:
    if list(frame.columns.astype(str)) != list(fitted.astype(str)):
        raise ValueError("Input columns must match the columns seen during fit.")


def _as_frame(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.Series):
        return X.to_frame()
    if isinstance(X, pd.DataFrame):
        return X
    raise TypeError("X must be a pandas Series or pandas DataFrame.")

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

from .base import CategoricalEncoder


class OrdinalCategoricalEncoder(CategoricalEncoder):
    """Ordinal encoder with explicit unknown and missing-value sentinels."""

    def __init__(
        self,
        *,
        unknown_value: int = -1,
        encoded_missing_value: int = -2,
    ) -> None:
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value

    def _fit(self, X: Any, y: Any = None) -> OrdinalCategoricalEncoder:
        frame = _as_frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=self.unknown_value,
            encoded_missing_value=self.encoded_missing_value,
        )
        self.encoder_.fit(frame)
        self._is_fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "_is_fitted")
        return self.encoder_.transform(_as_frame(X))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "_is_fitted")
        features = self.feature_names_in_ if input_features is None else np.asarray(input_features)
        if len(features) != len(self.feature_names_in_):
            raise ValueError("input_features must match the number of fitted features.")
        return np.asarray([str(name) for name in features], dtype=object)


def _as_frame(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.Series):
        return X.to_frame()
    if isinstance(X, pd.DataFrame):
        return X
    raise TypeError("X must be a pandas Series or pandas DataFrame.")

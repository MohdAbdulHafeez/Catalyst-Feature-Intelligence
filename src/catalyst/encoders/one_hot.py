from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from .base import CategoricalEncoder


class OneHotCategoricalEncoder(CategoricalEncoder):
    """CATALYST adapter around sklearn's OneHotEncoder."""

    def __init__(
        self,
        *,
        handle_unknown: str = "ignore",
        min_frequency: int | float | None = None,
        max_categories: int | None = None,
        sparse_output: bool = True,
    ) -> None:
        self.handle_unknown = handle_unknown
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.sparse_output = sparse_output

    def _fit(self, X: Any, y: Any = None) -> OneHotCategoricalEncoder:
        frame = _as_frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.encoder_ = OneHotEncoder(
            handle_unknown=self.handle_unknown,
            min_frequency=self.min_frequency,
            max_categories=self.max_categories,
            sparse_output=self.sparse_output,
        )
        self.encoder_.fit(frame)
        self._is_fitted = True
        return self

    def transform(self, X: Any) -> Any:
        check_is_fitted(self, "_is_fitted")
        return self.encoder_.transform(_as_frame(X))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "_is_fitted")
        return self.encoder_.get_feature_names_out(input_features)


def _as_frame(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.Series):
        return X.to_frame()
    if isinstance(X, pd.DataFrame):
        return X
    raise TypeError("X must be a pandas Series or pandas DataFrame.")

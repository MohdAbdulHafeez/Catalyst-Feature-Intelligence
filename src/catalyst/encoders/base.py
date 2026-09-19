from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class CategoricalEncoder(BaseEstimator, TransformerMixin, ABC):
    """Base sklearn-compatible contract for CATALYST categorical encoders."""

    requires_target: bool = False

    def fit(self, X: Any, y: Any = None) -> CategoricalEncoder:
        """Fit encoder state using only the supplied training partition."""
        self._validate_X(X)
        if self.requires_target and y is None:
            raise ValueError(f"{self.__class__.__name__} requires y during fit.")
        return self._fit(X, y)

    @abstractmethod
    def _fit(self, X: Any, y: Any = None) -> CategoricalEncoder:
        """Implementation-specific fitting logic."""

    @abstractmethod
    def transform(self, X: Any) -> Any:
        """Transform input using state learned during fit."""

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output names for encoders whose width equals input width."""
        check_is_fitted(self, "_is_fitted")
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        if input_features is None:
            raise ValueError("input_features is unavailable before fit metadata is established.")
        return np.asarray([str(feature) for feature in input_features], dtype=object)

    @staticmethod
    def _validate_X(X: Any) -> None:
        if not isinstance(X, (pd.Series, pd.DataFrame)):
            raise TypeError("X must be a pandas Series or pandas DataFrame.")

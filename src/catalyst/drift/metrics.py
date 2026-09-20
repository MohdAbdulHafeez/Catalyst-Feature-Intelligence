from __future__ import annotations

import numpy as np


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> float:
    """
    Calculate population stability index for aligned probability vectors.

    `expected` is the training/reference baseline and `actual` is the
    comparison distribution.
    """
    _validate_probability_vectors(expected, actual, epsilon)

    expected_safe = np.clip(expected, epsilon, None)
    actual_safe = np.clip(actual, epsilon, None)

    return float(np.sum((actual_safe - expected_safe) * np.log(actual_safe / expected_safe)))


def jensen_shannon_divergence(
    first: np.ndarray,
    second: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> float:
    """Calculate Jensen-Shannon divergence in natural-log units."""
    _validate_probability_vectors(first, second, epsilon)

    first_safe = _normalize(np.clip(first, epsilon, None))
    second_safe = _normalize(np.clip(second, epsilon, None))
    midpoint = 0.5 * (first_safe + second_safe)

    return float(
        0.5 * _kl_divergence(first_safe, midpoint) + 0.5 * _kl_divergence(second_safe, midpoint)
    )


def total_variation_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Calculate total variation distance between two distributions."""
    _validate_probability_vectors(first, second, epsilon=None)

    first_normalized = _normalize(first)
    second_normalized = _normalize(second)

    return float(0.5 * np.sum(np.abs(first_normalized - second_normalized)))


def hhi(probabilities: np.ndarray) -> float:
    """Calculate Herfindahl-Hirschman concentration index."""
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 1:
        raise ValueError("probabilities must be one-dimensional.")

    normalized = _normalize(probabilities)

    return float(np.sum(np.square(normalized)))


def _kl_divergence(
    probabilities: np.ndarray,
    reference: np.ndarray,
) -> float:
    return float(np.sum(probabilities * np.log(probabilities / reference)))


def _normalize(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))

    if total <= 0.0:
        raise ValueError("Probability vector must contain positive mass.")

    return values / total


def _validate_probability_vectors(
    first: np.ndarray,
    second: np.ndarray,
    epsilon: float | None,
) -> None:
    if epsilon is not None and epsilon <= 0.0:
        raise ValueError("epsilon must be greater than zero.")

    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)

    if first_array.ndim != 1 or second_array.ndim != 1:
        raise ValueError("Probability vectors must be one-dimensional.")

    if first_array.shape != second_array.shape:
        raise ValueError("Probability vectors must have identical shapes.")

    if not np.isfinite(first_array).all():
        raise ValueError("First probability vector contains non-finite values.")

    if not np.isfinite(second_array).all():
        raise ValueError("Second probability vector contains non-finite values.")

    if np.any(first_array < 0.0) or np.any(second_array < 0.0):
        raise ValueError("Probability vectors cannot contain negative values.")

    if float(np.sum(first_array)) <= 0.0:
        raise ValueError("First probability vector must contain positive mass.")

    if float(np.sum(second_array)) <= 0.0:
        raise ValueError("Second probability vector must contain positive mass.")

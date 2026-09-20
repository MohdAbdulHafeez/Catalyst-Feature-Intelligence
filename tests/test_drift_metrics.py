from __future__ import annotations

import numpy as np
import pytest

from catalyst.drift.metrics import (
    hhi,
    jensen_shannon_divergence,
    population_stability_index,
    total_variation_distance,
)


def test_psi_is_zero_for_identical_distributions() -> None:
    distribution = np.array([0.2, 0.3, 0.5])

    assert population_stability_index(
        distribution,
        distribution,
    ) == pytest.approx(0.0)


def test_js_divergence_is_zero_for_identical_distributions() -> None:
    distribution = np.array([0.2, 0.3, 0.5])

    assert jensen_shannon_divergence(
        distribution,
        distribution,
    ) == pytest.approx(0.0)


def test_total_variation_is_zero_for_identical_distributions() -> None:
    distribution = np.array([0.2, 0.3, 0.5])

    assert total_variation_distance(
        distribution,
        distribution,
    ) == pytest.approx(0.0)


def test_total_variation_is_bounded() -> None:
    first = np.array([1.0, 0.0])
    second = np.array([0.0, 1.0])

    assert total_variation_distance(first, second) == pytest.approx(1.0)


def test_hhi_uniform_distribution() -> None:
    distribution = np.array([0.25, 0.25, 0.25, 0.25])

    assert hhi(distribution) == pytest.approx(0.25)


def test_hhi_single_category() -> None:
    distribution = np.array([1.0])

    assert hhi(distribution) == pytest.approx(1.0)


def test_probability_vector_shape_must_match() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        population_stability_index(
            np.array([0.5, 0.5]),
            np.array([1.0]),
        )


def test_probability_vectors_cannot_be_negative() -> None:
    with pytest.raises(ValueError, match="negative"):
        jensen_shannon_divergence(
            np.array([1.0, -0.1]),
            np.array([0.5, 0.5]),
        )


def test_epsilon_must_be_positive_for_psi() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        population_stability_index(
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            epsilon=0.0,
        )

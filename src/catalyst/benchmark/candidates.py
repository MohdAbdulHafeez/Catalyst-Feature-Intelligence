from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from catalyst.encoders import EncoderKind, create_encoder


class ModelKind(StrEnum):
    """Supported baseline models for benchmark candidates."""

    LOGISTIC_REGRESSION = "logistic_regression"


@dataclass(frozen=True, slots=True)
class EncoderSpec:
    """Immutable configuration for one categorical encoder."""

    name: str
    kind: EncoderKind
    params: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Immutable configuration for one benchmark model."""

    name: str
    kind: ModelKind
    params: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BenchmarkCandidate:
    """Immutable encoder/model combination evaluated by the benchmark engine."""

    name: str
    encoder: EncoderSpec
    model: ModelSpec


DEFAULT_ENCODERS: tuple[EncoderSpec, ...] = (
    EncoderSpec(
        name="one_hot",
        kind=EncoderKind.ONE_HOT,
        params={},
    ),
    EncoderSpec(
        name="ordinal",
        kind=EncoderKind.ORDINAL,
        params={},
    ),
    EncoderSpec(
        name="frequency",
        kind=EncoderKind.FREQUENCY,
        params={},
    ),
    EncoderSpec(
        name="target_mean",
        kind=EncoderKind.TARGET_MEAN,
        params={},
    ),
    EncoderSpec(
        name="hashing",
        kind=EncoderKind.HASHING,
        params={},
    ),
)


DEFAULT_CLASSIFICATION_MODEL = ModelSpec(
    name="logistic_regression",
    kind=ModelKind.LOGISTIC_REGRESSION,
    params={
        "max_iter": 1000,
        "random_state": 42,
    },
)


def build_pipeline(
    candidate: BenchmarkCandidate,
    *,
    categorical_columns: tuple[str, ...],
    numerical_columns: tuple[str, ...] = (),
) -> Pipeline:
    """
    Build a fresh leakage-safe sklearn pipeline for one candidate.

    Categorical columns are transformed by the candidate encoder.
    Numerical columns are passed through unchanged.
    """
    if not categorical_columns:
        raise ValueError("At least one categorical column is required.")

    _validate_unique_columns(categorical_columns, "categorical_columns")
    _validate_unique_columns(numerical_columns, "numerical_columns")

    overlap = set(categorical_columns).intersection(numerical_columns)
    if overlap:
        raise ValueError(f"Columns cannot be both categorical and numerical: {sorted(overlap)}.")

    encoder = create_encoder(
        candidate.encoder.kind,
        **dict(candidate.encoder.params),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                encoder,
                list(categorical_columns),
            ),
        ],
        remainder="passthrough" if numerical_columns else "drop",
        verbose_feature_names_out=False,
    )

    model = _build_model(candidate.model)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def generate_classification_candidates(
    *,
    encoder_specs: tuple[EncoderSpec, ...] = DEFAULT_ENCODERS,
    model: ModelSpec = DEFAULT_CLASSIFICATION_MODEL,
) -> tuple[BenchmarkCandidate, ...]:
    """Generate deterministic classification benchmark candidates."""
    if not encoder_specs:
        raise ValueError("At least one encoder specification is required.")

    return tuple(
        BenchmarkCandidate(
            name=f"{encoder.name}__{model.name}",
            encoder=encoder,
            model=model,
        )
        for encoder in encoder_specs
    )


def _build_model(spec: ModelSpec) -> Any:
    if spec.kind is ModelKind.LOGISTIC_REGRESSION:
        return LogisticRegression(**dict(spec.params))

    raise ValueError(f"Unsupported model kind: {spec.kind!r}")


def _validate_unique_columns(
    columns: tuple[str, ...],
    parameter_name: str,
) -> None:
    if len(columns) != len(set(columns)):
        raise ValueError(f"{parameter_name} must not contain duplicates.")

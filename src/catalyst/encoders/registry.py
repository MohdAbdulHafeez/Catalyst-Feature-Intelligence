from __future__ import annotations

from enum import StrEnum
from typing import Any

from .base import CategoricalEncoder
from .frequency import FrequencyCategoricalEncoder
from .hashing import HashingCategoricalEncoder
from .one_hot import OneHotCategoricalEncoder
from .ordinal import OrdinalCategoricalEncoder
from .target import TargetMeanCategoricalEncoder


class EncoderKind(StrEnum):
    """Stable identifiers consumed by benchmark and API layers."""

    ONE_HOT = "one_hot"
    ORDINAL = "ordinal"
    FREQUENCY = "frequency"
    TARGET_MEAN = "target_mean"
    HASHING = "hashing"


_ENCODER_REGISTRY: dict[EncoderKind, type[CategoricalEncoder]] = {
    EncoderKind.ONE_HOT: OneHotCategoricalEncoder,
    EncoderKind.ORDINAL: OrdinalCategoricalEncoder,
    EncoderKind.FREQUENCY: FrequencyCategoricalEncoder,
    EncoderKind.TARGET_MEAN: TargetMeanCategoricalEncoder,
    EncoderKind.HASHING: HashingCategoricalEncoder,
}


def create_encoder(kind: EncoderKind | str, **kwargs: Any) -> CategoricalEncoder:
    """Instantiate a registered encoder using a stable public identifier."""
    normalized = EncoderKind(kind)
    return _ENCODER_REGISTRY[normalized](**kwargs)


def supported_encoders() -> tuple[EncoderKind, ...]:
    """Return supported encoder kinds in deterministic registry order."""
    return tuple(_ENCODER_REGISTRY)

"""Leakage-safe categorical encoder architecture for CATALYST."""

from catalyst.encoders.base import CategoricalEncoder
from catalyst.encoders.frequency import FrequencyCategoricalEncoder
from catalyst.encoders.hashing import HashingCategoricalEncoder
from catalyst.encoders.one_hot import OneHotCategoricalEncoder
from catalyst.encoders.ordinal import OrdinalCategoricalEncoder
from catalyst.encoders.registry import EncoderKind, create_encoder, supported_encoders
from catalyst.encoders.target import TargetMeanCategoricalEncoder

__all__ = [
    "CategoricalEncoder",
    "FrequencyCategoricalEncoder",
    "HashingCategoricalEncoder",
    "OneHotCategoricalEncoder",
    "OrdinalCategoricalEncoder",
    "TargetMeanCategoricalEncoder",
    "EncoderKind",
    "create_encoder",
    "supported_encoders",
]

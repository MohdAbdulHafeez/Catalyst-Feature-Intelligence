"""Benchmark contracts, candidates, and metrics for CATALYST."""

from catalyst.benchmark.candidates import (
    DEFAULT_CLASSIFICATION_MODEL,
    DEFAULT_ENCODERS,
    BenchmarkCandidate,
    EncoderSpec,
    ModelKind,
    ModelSpec,
    build_pipeline,
    generate_classification_candidates,
)
from catalyst.benchmark.engine import BenchmarkConfig, BenchmarkEngine
from catalyst.benchmark.metrics import available_metrics, get_metric
from catalyst.benchmark.models import (
    DEFAULT_CV_SPLITS,
    DEFAULT_RANDOM_STATE,
    BenchmarkResult,
    BenchmarkStatus,
    BenchmarkSummary,
    BenchmarkTask,
    FoldResult,
    ScoreDirection,
    TrialResult,
)

__all__ = [
    "BenchmarkCandidate",
    "BenchmarkResult",
    "BenchmarkStatus",
    "BenchmarkSummary",
    "BenchmarkTask",
    "DEFAULT_CLASSIFICATION_MODEL",
    "DEFAULT_CV_SPLITS",
    "DEFAULT_ENCODERS",
    "DEFAULT_RANDOM_STATE",
    "EncoderSpec",
    "FoldResult",
    "ModelKind",
    "ModelSpec",
    "ScoreDirection",
    "TrialResult",
    "available_metrics",
    "build_pipeline",
    "generate_classification_candidates",
    "get_metric",
    "BenchmarkConfig",
    "BenchmarkEngine",
]

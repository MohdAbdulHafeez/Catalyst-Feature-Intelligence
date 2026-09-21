from __future__ import annotations

import pytest
from pydantic import ValidationError

from catalyst.api.contracts import (
    BenchmarkRequest,
    DatasetFormat,
    DatasetReference,
    OptimizationRequest,
    OptimizationWeights,
    TaskType,
)


def test_dataset_reference_rejects_invalid_identifier() -> None:
    with pytest.raises(ValidationError):
        DatasetReference(dataset_id="contains spaces")


def test_upload_metadata_serializes_format() -> None:
    from catalyst.api.contracts import DatasetUploadMetadata

    payload = DatasetUploadMetadata(
        dataset_id="demo_01",
        filename="customers.parquet",
        format=DatasetFormat.PARQUET,
        size_bytes=1024,
    ).model_dump(mode="json")

    assert payload["format"] == "parquet"


def test_benchmark_request_validates_overlapping_feature_groups() -> None:
    with pytest.raises(ValidationError, match="both categorical and numerical"):
        BenchmarkRequest(
            dataset_id="demo",
            target_column="target",
            task_type=TaskType.CLASSIFICATION,
            metric="accuracy",
            categorical_columns=("city",),
            numerical_columns=("city",),
        )


def test_benchmark_request_rejects_target_in_feature_groups() -> None:
    with pytest.raises(ValidationError, match="target_column"):
        BenchmarkRequest(
            dataset_id="demo",
            target_column="target",
            task_type=TaskType.CLASSIFICATION,
            metric="accuracy",
            categorical_columns=("target",),
        )


def test_benchmark_request_rejects_duplicate_column_names() -> None:
    with pytest.raises(ValidationError, match="unique"):
        BenchmarkRequest(
            dataset_id="demo",
            target_column="target",
            task_type=TaskType.CLASSIFICATION,
            metric="accuracy",
            categorical_columns=("city", "city"),
        )


def test_benchmark_request_rejects_invalid_cv() -> None:
    with pytest.raises(ValidationError):
        BenchmarkRequest(
            dataset_id="demo",
            target_column="target",
            task_type=TaskType.CLASSIFICATION,
            metric="accuracy",
            cv=1,
        )


def test_optimization_weights_require_positive_total() -> None:
    with pytest.raises(ValidationError, match="positive"):
        OptimizationWeights(
            performance_weight=0.0,
            cost_weight=0.0,
            risk_weight=0.0,
        )


def test_optimization_request_has_explicit_defaults() -> None:
    request = OptimizationRequest(benchmark_id="bench_001")

    assert request.weights.performance_weight == 0.50
    assert request.weights.cost_weight == 0.25
    assert request.weights.risk_weight == 0.25
    assert request.constraints.maximum_risk is None


def test_transport_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        DatasetReference(dataset_id="demo", unexpected="value")


def test_api_models_are_immutable() -> None:
    request = DatasetReference(dataset_id="demo")

    with pytest.raises(ValidationError):
        request.dataset_id = "changed"


def test_json_round_trip() -> None:
    original = BenchmarkRequest(
        dataset_id="demo",
        target_column="target",
        task_type=TaskType.CLASSIFICATION,
        metric="f1",
        categorical_columns=("city", "segment"),
        numerical_columns=("age", "income"),
    )

    restored = BenchmarkRequest.model_validate_json(original.model_dump_json())

    assert restored == original

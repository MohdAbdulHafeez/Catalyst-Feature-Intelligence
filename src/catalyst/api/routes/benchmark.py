from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status

from catalyst.api.contracts import BenchmarkRequest, TaskType
from catalyst.api.ingestion import DatasetIngestionError
from catalyst.benchmark.engine import BenchmarkConfig, BenchmarkEngine
from catalyst.benchmark.models import BenchmarkResult, BenchmarkTask
from catalyst.profiling import SchemaProfiler

router = APIRouter(
    prefix="/api/v1/benchmark",
    tags=["benchmark"],
)


def _resolve_columns(
    frame: pd.DataFrame,
    target_column: str,
    requested_categorical: Sequence[str],
    requested_numerical: Sequence[str],
) -> tuple[list[str], list[str]]:
    if target_column not in frame.columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "invalid_target_column",
                "message": f"Target column '{target_column}' was not found.",
            },
        )

    profile = SchemaProfiler().profile(frame)

    available_categorical = set(profile.categorical_columns)
    available_numerical = set(profile.numerical_columns)

    categorical_columns = (
        list(dict.fromkeys(requested_categorical))
        if requested_categorical
        else [column for column in profile.categorical_columns if column != target_column]
    )

    numerical_columns = (
        list(dict.fromkeys(requested_numerical))
        if requested_numerical
        else [column for column in profile.numerical_columns if column != target_column]
    )

    unknown_categorical = [column for column in categorical_columns if column not in frame.columns]
    if unknown_categorical:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "unknown_categorical_column",
                "message": (
                    f"The following categorical columns were not found: {unknown_categorical}"
                ),
            },
        )

    non_categorical = [
        column for column in categorical_columns if column not in available_categorical
    ]
    if non_categorical:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "non_categorical_column",
                "message": (f"The following columns are not categorical: {non_categorical}"),
            },
        )

    unknown_numerical = [column for column in numerical_columns if column not in frame.columns]
    if unknown_numerical:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "unknown_numerical_column",
                "message": (f"The following numerical columns were not found: {unknown_numerical}"),
            },
        )

    non_numerical = [column for column in numerical_columns if column not in available_numerical]
    if non_numerical:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "non_numerical_column",
                "message": (f"The following columns are not numerical: {non_numerical}"),
            },
        )

    if target_column in categorical_columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "invalid_feature_columns",
                "message": ("The target column cannot also be a categorical feature."),
            },
        )

    if target_column in numerical_columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "invalid_feature_columns",
                "message": ("The target column cannot also be a numerical feature."),
            },
        )

    overlap = sorted(set(categorical_columns).intersection(numerical_columns))
    if overlap:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "overlapping_feature_columns",
                "message": (f"Columns cannot be both categorical and numerical: {overlap}"),
            },
        )

    return categorical_columns, numerical_columns


@router.post(
    "",
    response_model=BenchmarkResult,
    status_code=status.HTTP_200_OK,
)
def run_benchmark(
    payload: BenchmarkRequest,
    request: Request,
) -> BenchmarkResult:
    service = request.app.state.dataset_store

    try:
        frame = service.load_dataframe(payload.dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "dataset_not_found",
                "message": "Dataset not found.",
            },
        ) from exc
    except DatasetIngestionError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "dataset_not_found",
                "message": str(exc),
            },
        ) from exc

    if payload.task_type is not TaskType.CLASSIFICATION:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "unsupported_task_type",
                "message": ("The benchmark API currently supports classification only."),
            },
        )

    categorical_columns, numerical_columns = _resolve_columns(
        frame=frame,
        target_column=payload.target_column,
        requested_categorical=payload.categorical_columns,
        requested_numerical=payload.numerical_columns,
    )

    if not categorical_columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "no_categorical_features",
                "message": ("At least one categorical feature is required."),
            },
        )

    target = frame[payload.target_column]
    features = frame.drop(columns=[payload.target_column])

    config = BenchmarkConfig(
        cv_splits=payload.cv,
        shuffle=True,
        random_state=payload.random_state,
    )

    engine = BenchmarkEngine(config=config)

    try:
        result = engine.run(
            X=features,
            y=target,
            categorical_columns=tuple(categorical_columns),
            numerical_columns=tuple(numerical_columns),
            metric=payload.metric,
            task=BenchmarkTask.CLASSIFICATION,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "benchmark_validation_error",
                "message": str(exc),
            },
        ) from exc

    return result

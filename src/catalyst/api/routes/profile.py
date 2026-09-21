from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from catalyst.api.contracts import ProfileColumnResponse, ProfileRequest, ProfileResponse
from catalyst.api.ingestion import DatasetIngestionError
from catalyst.profiling import SchemaProfiler

router = APIRouter(prefix="/profile", tags=["profiling"])


@router.post("", response_model=ProfileResponse)
def profile_dataset(payload: ProfileRequest, request: Request) -> ProfileResponse:
    store = request.app.state.dataset_store

    try:
        manifest = store.get_manifest(payload.dataset_id)
        frame = store.load_dataframe(payload.dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found.",
        ) from exc
    except DatasetIngestionError:
        raise

    if payload.target_column is not None and payload.target_column not in frame.columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Target column '{payload.target_column}' was not found in the dataset.",
        )

    profile = SchemaProfiler().profile(frame)
    columns = tuple(
        ProfileColumnResponse(
            name=column.name,
            semantic_type=column.semantic_type.value,
            pandas_dtype=column.pandas_dtype,
            row_count=column.row_count,
            non_null_count=column.non_null_count,
            missing_count=column.missing_count,
            missing_fraction=column.missing_fraction,
            unique_count=column.unique_count,
            unique_ratio=column.unique_ratio,
            is_constant=column.is_constant,
            likely_identifier=column.likely_identifier,
            identifier_likelihood_score=column.identifier_likelihood_score,
            identifier_signals=column.identifier_signals,
            sample_values=column.sample_values,
        )
        for column in profile.columns
    )

    return ProfileResponse(
        dataset_id=manifest.dataset_id,
        filename=manifest.filename,
        format=manifest.format,
        row_count=profile.row_count,
        column_count=profile.column_count,
        columns=columns,
        numerical_columns=profile.numerical_columns,
        categorical_columns=profile.categorical_columns,
        boolean_columns=profile.boolean_columns,
        datetime_columns=profile.datetime_columns,
        text_columns=profile.text_columns,
        identifier_columns=profile.identifier_columns,
    )

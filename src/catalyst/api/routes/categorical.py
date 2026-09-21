from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Field, field_validator

from catalyst.api.contracts import DatasetFormat
from catalyst.api.ingestion import DatasetIngestionError
from catalyst.categorical import CategoricalProfiler

router = APIRouter(prefix="/categorical", tags=["categorical"])


class CategoricalIntelligenceRequest(BaseModel):
    """Strict request contract for categorical feature intelligence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    columns: tuple[str, ...] = ()

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, columns: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(column.strip() for column in columns)
        if any(not column for column in cleaned):
            raise ValueError("Column names must not be empty.")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("Column names must be unique.")
        return cleaned


class CategoricalIntelligenceResponse(BaseModel):
    """Frontend-safe serialization of the categorical intelligence result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1, max_length=128)
    filename: str = Field(min_length=1, max_length=255)
    format: DatasetFormat
    profile_count: int = Field(ge=0)
    profiles: tuple[dict[str, Any], ...] = ()


@router.post("", response_model=CategoricalIntelligenceResponse)
def analyze_categorical_features(
    payload: CategoricalIntelligenceRequest,
    request: Request,
) -> CategoricalIntelligenceResponse:
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

    try:
        profiles = CategoricalProfiler().profile(
            frame,
            columns=payload.columns or None,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    encoded_profiles = tuple(jsonable_encoder(profile) for profile in profiles)

    return CategoricalIntelligenceResponse(
        dataset_id=manifest.dataset_id,
        filename=manifest.filename,
        format=manifest.format,
        profile_count=len(encoded_profiles),
        profiles=encoded_profiles,
    )

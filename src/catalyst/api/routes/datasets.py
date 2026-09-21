from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from catalyst.api.contracts import DatasetUploadMetadata
from catalyst.api.ingestion import DatasetIngestionError

router = APIRouter(
    prefix="/datasets",
    tags=["datasets"],
)


@router.post(
    "",
    response_model=DatasetUploadMetadata,
    status_code=status.HTTP_201_CREATED,
)
def upload_dataset(
    request: Request,
    file: UploadFile = File(...),
) -> DatasetUploadMetadata:
    service = request.app.state.dataset_store
    dataset_id = service.create_dataset_id()

    try:
        manifest = service.save_upload(
            dataset_id=dataset_id,
            filename=file.filename or "",
            upload=file,
        )
    finally:
        file.file.close()

    return manifest.to_metadata()


@router.get(
    "/{dataset_id}",
    response_model=DatasetUploadMetadata,
)
def get_dataset(
    dataset_id: str,
    request: Request,
) -> DatasetUploadMetadata:
    service = request.app.state.dataset_store

    try:
        manifest = service.get_manifest(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found.",
        ) from exc
    except DatasetIngestionError:
        raise

    return manifest.to_metadata()

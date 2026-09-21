from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyarrow.parquet as pq

from catalyst.api.contracts import DatasetFormat, DatasetUploadMetadata

_DATASET_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_CHUNK_SIZE = 1024 * 1024


class DatasetIngestionError(ValueError):
    """Raised when an uploaded dataset cannot be safely ingested."""


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    filename: str
    format: DatasetFormat
    size_bytes: int
    row_count: int
    column_count: int
    columns: tuple[str, ...]
    created_at: datetime

    def to_metadata(self) -> DatasetUploadMetadata:
        return DatasetUploadMetadata(
            dataset_id=self.dataset_id,
            filename=self.filename,
            format=self.format,
            size_bytes=self.size_bytes,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "filename": self.filename,
            "format": self.format.value,
            "size_bytes": self.size_bytes,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": list(self.columns),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> DatasetManifest:
        return cls(
            dataset_id=str(payload["dataset_id"]),
            filename=str(payload["filename"]),
            format=DatasetFormat(str(payload["format"])),
            size_bytes=int(payload["size_bytes"]),
            row_count=int(payload["row_count"]),
            column_count=int(payload["column_count"]),
            columns=tuple(str(value) for value in payload["columns"]),
            created_at=datetime.fromisoformat(str(payload["created_at"])),
        )


class LocalDatasetStore:
    """Filesystem-backed dataset store with atomic file and manifest creation."""

    def __init__(self, root: Path, max_upload_bytes: int = 25 * 1024 * 1024) -> None:
        if max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be positive.")
        self.root = root
        self.max_upload_bytes = max_upload_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def create_dataset_id(self) -> str:
        return uuid4().hex

    def save_upload(
        self,
        *,
        dataset_id: str,
        filename: str,
        upload,
    ) -> DatasetManifest:
        _validate_dataset_id(dataset_id)
        safe_filename = _safe_filename(filename)
        dataset_format = _format_from_filename(safe_filename)

        dataset_dir = self.root / dataset_id
        dataset_dir.mkdir(parents=False, exist_ok=False)

        data_path = dataset_dir / f"data.{dataset_format.value}"
        temp_path = dataset_dir / f".upload-{uuid4().hex}.tmp"

        size_bytes = 0
        try:
            with temp_path.open("wb") as target:
                while True:
                    chunk = upload.file.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    size_bytes += len(chunk)
                    if size_bytes > self.max_upload_bytes:
                        raise DatasetIngestionError(
                            f"Dataset exceeds the {self.max_upload_bytes} byte upload limit."
                        )
                    target.write(chunk)

            temp_path.replace(data_path)
            manifest = _build_manifest(
                dataset_id=dataset_id,
                filename=safe_filename,
                dataset_format=dataset_format,
                data_path=data_path,
                size_bytes=size_bytes,
            )
            manifest_path = dataset_dir / "manifest.json"
            temp_manifest = dataset_dir / f".manifest-{uuid4().hex}.tmp"
            temp_manifest.write_text(
                json.dumps(manifest.to_json(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temp_manifest.replace(manifest_path)
            return manifest
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            if dataset_dir.exists():
                for child in dataset_dir.iterdir():
                    child.unlink()
                dataset_dir.rmdir()
            raise

    def get_manifest(self, dataset_id: str) -> DatasetManifest:
        _validate_dataset_id(dataset_id)
        manifest_path = self.root / dataset_id / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(dataset_id)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return DatasetManifest.from_json(payload)


def _validate_dataset_id(dataset_id: str) -> None:
    if not _DATASET_ID_PATTERN.fullmatch(dataset_id):
        raise DatasetIngestionError("Invalid dataset identifier.")


def _safe_filename(filename: str) -> str:
    candidate = Path(filename.replace("\\", "/")).name.strip()
    if not candidate or candidate in {".", ".."}:
        raise DatasetIngestionError("A valid dataset filename is required.")
    if len(candidate) > 255:
        raise DatasetIngestionError("Dataset filename is too long.")
    return candidate


def _format_from_filename(filename: str) -> DatasetFormat:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return DatasetFormat.CSV
    if suffix == ".parquet":
        return DatasetFormat.PARQUET
    raise DatasetIngestionError("Only .csv and .parquet datasets are supported.")


def _build_manifest(
    *,
    dataset_id: str,
    filename: str,
    dataset_format: DatasetFormat,
    data_path: Path,
    size_bytes: int,
) -> DatasetManifest:
    if dataset_format is DatasetFormat.CSV:
        row_count, columns = _inspect_csv(data_path)
    else:
        row_count, columns = _inspect_parquet(data_path)

    if not columns:
        raise DatasetIngestionError("Dataset must contain at least one column.")
    if row_count <= 0:
        raise DatasetIngestionError("Dataset must contain at least one data row.")

    return DatasetManifest(
        dataset_id=dataset_id,
        filename=filename,
        format=dataset_format,
        size_bytes=size_bytes,
        row_count=row_count,
        column_count=len(columns),
        columns=tuple(columns),
        created_at=datetime.now(UTC),
    )


def _inspect_csv(path: Path) -> tuple[int, tuple[str, ...]]:
    try:
        header = pd.read_csv(path, nrows=0)
        columns = tuple(str(column) for column in header.columns)
        row_count = 0
        for chunk in pd.read_csv(path, chunksize=100_000):
            row_count += len(chunk)
        return row_count, columns
    except Exception as exc:
        raise DatasetIngestionError(f"Invalid CSV dataset: {exc}") from exc


def _inspect_parquet(path: Path) -> tuple[int, tuple[str, ...]]:
    try:
        parquet = pq.ParquetFile(path)
        metadata = parquet.metadata
        columns = tuple(str(name) for name in parquet.schema_arrow.names)
        return int(metadata.num_rows), columns
    except Exception as exc:
        raise DatasetIngestionError(f"Invalid Parquet dataset: {exc}") from exc

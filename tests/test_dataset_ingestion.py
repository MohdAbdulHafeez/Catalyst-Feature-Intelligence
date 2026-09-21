from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from catalyst.api.app import create_app
from catalyst.api.config import ApiSettings


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = ApiSettings(
        title="CATALYST Test API",
        version="test-version",
        environment="test",
        cors_origins=("http://localhost:3000",),
        data_dir=tmp_path / "datasets",
        max_upload_bytes=5 * 1024 * 1024,
    )
    return TestClient(create_app(settings))


def test_csv_upload_creates_dataset(client: TestClient) -> None:
    response = client.post(
        "/api/v1/datasets",
        files={
            "file": (
                "customers.csv",
                b"age,city,target\n20,Hyderabad,0\n21,Mumbai,1\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["filename"] == "customers.csv"
    assert payload["format"] == "csv"
    assert payload["size_bytes"] > 0
    assert len(payload["dataset_id"]) == 32


def test_uploaded_dataset_can_be_retrieved(client: TestClient) -> None:
    upload = client.post(
        "/api/v1/datasets",
        files={"file": ("demo.csv", b"city,target\nA,0\nB,1\n", "text/csv")},
    )
    dataset_id = upload.json()["dataset_id"]

    response = client.get(f"/api/v1/datasets/{dataset_id}")

    assert response.status_code == 200
    assert response.json()["dataset_id"] == dataset_id
    assert response.json()["filename"] == "demo.csv"


def test_missing_dataset_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/datasets/does-not-exist")
    assert response.status_code == 404


def test_unsupported_extension_returns_422(client: TestClient) -> None:
    response = client.post(
        "/api/v1/datasets",
        files={"file": ("data.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 422


def test_malformed_csv_returns_422(client: TestClient) -> None:
    response = client.post(
        "/api/v1/datasets",
        files={"file": ("broken.csv", b'city,target\nA,"broken\n', "text/csv")},
    )
    assert response.status_code == 422


def test_empty_dataset_returns_422(client: TestClient) -> None:
    response = client.post(
        "/api/v1/datasets",
        files={"file": ("empty.csv", b"city,target\n", "text/csv")},
    )
    assert response.status_code == 422


def test_upload_size_limit_is_enforced(tmp_path: Path) -> None:
    settings = ApiSettings(data_dir=tmp_path / "datasets", max_upload_bytes=10)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/datasets",
            files={
                "file": (
                    "large.csv",
                    b"city,target\nA,123456789\n",
                    "text/csv",
                )
            },
        )

    assert response.status_code == 422
    assert "upload limit" in response.json()["message"]
    assert response.json()["code"] == "dataset_ingestion_error"


def test_filename_path_is_sanitized(client: TestClient) -> None:
    response = client.post(
        "/api/v1/datasets",
        files={
            "file": (
                "../../safe.csv",
                b"city,target\nA,1\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 201
    assert response.json()["filename"] == "safe.csv"


def test_dataset_storage_is_atomic_on_failure(tmp_path: Path) -> None:
    from catalyst.api.ingestion import DatasetIngestionError, LocalDatasetStore

    store = LocalDatasetStore(tmp_path / "datasets", max_upload_bytes=10)

    class FailingUpload:
        class File:
            def read(self, _size: int) -> bytes:
                return b"x" * 11

        file = File()

    with pytest.raises(DatasetIngestionError):
        store.save_upload(
            dataset_id=store.create_dataset_id(),
            filename="data.csv",
            upload=FailingUpload(),
        )

    assert list((tmp_path / "datasets").iterdir()) == []


def test_parquet_upload_creates_dataset(client: TestClient, tmp_path: Path) -> None:
    frame = pd.DataFrame({"city": ["A", "B"], "target": [0, 1]})
    parquet_path = tmp_path / "demo.parquet"
    frame.to_parquet(parquet_path, index=False)

    with parquet_path.open("rb") as handle:
        response = client.post(
            "/api/v1/datasets",
            files={"file": ("demo.parquet", handle, "application/octet-stream")},
        )

    assert response.status_code == 201
    assert response.json()["format"] == "parquet"

from __future__ import annotations

from pathlib import Path

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


def upload_dataset(client: TestClient) -> str:
    response = client.post(
        "/api/v1/datasets",
        files={
            "file": (
                "customers.csv",
                (
                    b"age,is_member,city,customer_id,target\n"
                    b"20,true,Hyderabad,C001,1\n"
                    b"21,false,Mumbai,C002,0\n"
                    b"22,true,Hyderabad,C003,1\n"
                    b"23,true,Delhi,C004,1\n"
                ),
                "text/csv",
            )
        },
    )
    assert response.status_code == 201
    return response.json()["dataset_id"]


def test_profile_endpoint_returns_schema_intelligence(client: TestClient) -> None:
    dataset_id = upload_dataset(client)
    response = client.post(
        "/api/v1/profile",
        json={"dataset_id": dataset_id, "target_column": "target"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_id"] == dataset_id
    assert payload["row_count"] == 4
    assert payload["column_count"] == 5
    assert payload["numerical_columns"] == ["age", "target"]
    assert payload["boolean_columns"] == ["is_member"]
    assert payload["categorical_columns"] == ["city", "customer_id"]
    assert payload["identifier_columns"] == ["customer_id"]


def test_profile_endpoint_returns_column_details(client: TestClient) -> None:
    dataset_id = upload_dataset(client)
    response = client.post(
        "/api/v1/profile",
        json={"dataset_id": dataset_id},
    )

    assert response.status_code == 200
    columns = {item["name"]: item for item in response.json()["columns"]}
    assert columns["city"]["semantic_type"] == "categorical"
    assert columns["customer_id"]["likely_identifier"] is True


def test_profile_endpoint_rejects_unknown_target(client: TestClient) -> None:
    dataset_id = upload_dataset(client)
    response = client.post(
        "/api/v1/profile",
        json={"dataset_id": dataset_id, "target_column": "missing_target"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "http_error"
    assert "was not found" in response.json()["message"]


def test_profile_endpoint_returns_404_for_missing_dataset(client: TestClient) -> None:
    response = client.post(
        "/api/v1/profile",
        json={"dataset_id": "missing"},
    )

    assert response.status_code == 404
    assert response.json()["code"] == "http_error"


def test_profile_is_repeatable(client: TestClient) -> None:
    dataset_id = upload_dataset(client)
    first = client.post("/api/v1/profile", json={"dataset_id": dataset_id})
    second = client.post("/api/v1/profile", json={"dataset_id": dataset_id})

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()

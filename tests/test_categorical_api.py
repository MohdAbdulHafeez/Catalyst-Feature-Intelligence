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
                "categorical.csv",
                (
                    b"age,city,segment,customer_id,target\n"
                    b"20,Hyderabad,Gold,C001,1\n"
                    b"21,Mumbai,Silver,C002,0\n"
                    b"22,Hyderabad,Gold,C003,1\n"
                    b"23,Delhi,Bronze,C004,1\n"
                ),
                "text/csv",
            )
        },
    )
    assert response.status_code == 201
    return response.json()["dataset_id"]


def test_categorical_endpoint_returns_unified_profiles(client: TestClient) -> None:
    dataset_id = upload_dataset(client)

    response = client.post(
        "/api/v1/categorical",
        json={"dataset_id": dataset_id},
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["dataset_id"] == dataset_id
    assert payload["profile_count"] == 3
    names = [profile["feature_name"] for profile in payload["profiles"]]
    assert names == ["city", "segment", "customer_id"]


def test_categorical_endpoint_exposes_cardinality_frequency_ordinality_and_risk(
    client: TestClient,
) -> None:
    dataset_id = upload_dataset(client)

    response = client.post(
        "/api/v1/categorical",
        json={"dataset_id": dataset_id, "columns": ["segment"]},
    )

    assert response.status_code == 200
    profile = response.json()["profiles"][0]

    assert profile["feature_name"] == "segment"
    assert "cardinality" in profile
    assert "frequency" in profile
    assert "ordinality" in profile
    assert "risk" in profile
    assert profile["cardinality"]["unique_count"] == 3


def test_categorical_endpoint_allows_explicit_subset(client: TestClient) -> None:
    dataset_id = upload_dataset(client)

    response = client.post(
        "/api/v1/categorical",
        json={"dataset_id": dataset_id, "columns": ["city", "segment"]},
    )

    assert response.status_code == 200
    assert response.json()["profile_count"] == 2
    assert [profile["feature_name"] for profile in response.json()["profiles"]] == [
        "city",
        "segment",
    ]


def test_categorical_endpoint_rejects_non_categorical_selection(
    client: TestClient,
) -> None:
    dataset_id = upload_dataset(client)

    response = client.post(
        "/api/v1/categorical",
        json={"dataset_id": dataset_id, "columns": ["age"]},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "http_error"
    assert "not classified as categorical" in response.json()["message"]


def test_categorical_endpoint_rejects_unknown_column(client: TestClient) -> None:
    dataset_id = upload_dataset(client)

    response = client.post(
        "/api/v1/categorical",
        json={"dataset_id": dataset_id, "columns": ["unknown"]},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "http_error"


def test_categorical_endpoint_returns_404_for_missing_dataset(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/categorical",
        json={"dataset_id": "missing"},
    )

    assert response.status_code == 404
    assert response.json()["code"] == "http_error"


def test_categorical_endpoint_is_repeatable(client: TestClient) -> None:
    dataset_id = upload_dataset(client)

    first = client.post(
        "/api/v1/categorical",
        json={"dataset_id": dataset_id},
    )
    second = client.post(
        "/api/v1/categorical",
        json={"dataset_id": dataset_id},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()


def test_categorical_request_rejects_duplicate_columns(client: TestClient) -> None:
    dataset_id = upload_dataset(client)

    response = client.post(
        "/api/v1/categorical",
        json={"dataset_id": dataset_id, "columns": ["city", "city"]},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"

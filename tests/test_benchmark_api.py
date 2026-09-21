from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from catalyst.api.app import create_app
from catalyst.api.config import ApiSettings
from catalyst.api.ingestion import LocalDatasetStore


@pytest.fixture
def benchmark_client(tmp_path):
    settings = ApiSettings(data_dir=tmp_path)
    app = create_app(settings=settings)

    store = LocalDatasetStore(tmp_path)

    csv_content = """city,segment,age,target
Hyderabad,A,21,0
Mumbai,B,24,1
Delhi,A,27,0
Pune,C,31,1
Chennai,B,26,0
Bengaluru,C,35,1
Hyderabad,B,23,0
Mumbai,A,29,1
Delhi,C,32,0
Pune,A,28,1
Chennai,C,25,0
Bengaluru,B,34,1
"""

    dataset_id = store.create_dataset_id()

    upload = UploadFile(
        filename="benchmark.csv",
        file=BytesIO(csv_content.encode("utf-8")),
    )

    store.save_upload(
        dataset_id=dataset_id,
        filename="benchmark.csv",
        upload=upload,
    )

    return TestClient(app), dataset_id


def _benchmark_payload(dataset_id: str) -> dict:
    return {
        "dataset_id": dataset_id,
        "target_column": "target",
        "task_type": "classification",
        "metric": "accuracy",
        "categorical_columns": ["city", "segment"],
        "numerical_columns": ["age"],
        "cv": 3,
        "random_state": 42,
    }


def test_benchmark_endpoint_returns_result(benchmark_client):
    client, dataset_id = benchmark_client

    response = client.post(
        "/api/v1/benchmark",
        json=_benchmark_payload(dataset_id),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["summary"]["task"] == "classification"
    assert body["summary"]["metric_name"] == "accuracy"
    assert body["summary"]["cv_splits"] == 3
    assert body["trials"]
    assert len(body["trials"]) >= 1

    for trial in body["trials"]:
        assert "candidate_name" in trial
        assert "status" in trial
        assert "mean_score" in trial


def test_benchmark_endpoint_infers_categorical_columns(
    benchmark_client,
):
    client, dataset_id = benchmark_client

    payload = _benchmark_payload(dataset_id)
    payload.pop("categorical_columns")

    response = client.post(
        "/api/v1/benchmark",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["summary"]["task"] == "classification"
    assert body["trials"]


def test_benchmark_endpoint_rejects_unknown_categorical_column(
    benchmark_client,
):
    client, dataset_id = benchmark_client

    payload = _benchmark_payload(dataset_id)
    payload["categorical_columns"] = ["does_not_exist"]

    response = client.post(
        "/api/v1/benchmark",
        json=payload,
    )

    assert response.status_code == 422


def test_benchmark_endpoint_rejects_non_categorical_column(
    benchmark_client,
):
    client, dataset_id = benchmark_client

    payload = _benchmark_payload(dataset_id)
    payload["categorical_columns"] = ["age"]

    response = client.post(
        "/api/v1/benchmark",
        json=payload,
    )

    assert response.status_code == 422


def test_benchmark_endpoint_rejects_target_column_missing(
    benchmark_client,
):
    client, dataset_id = benchmark_client

    payload = _benchmark_payload(dataset_id)
    payload["target_column"] = "missing_target"

    response = client.post(
        "/api/v1/benchmark",
        json=payload,
    )

    assert response.status_code == 422


def test_benchmark_endpoint_rejects_no_categorical_features(
    benchmark_client,
):
    client, dataset_id = benchmark_client

    payload = _benchmark_payload(dataset_id)
    payload["categorical_columns"] = ["age"]
    payload["numerical_columns"] = []

    response = client.post(
        "/api/v1/benchmark",
        json=payload,
    )

    assert response.status_code == 422


def test_benchmark_endpoint_rejects_missing_dataset(
    tmp_path,
):
    settings = ApiSettings(data_dir=tmp_path)
    client = TestClient(create_app(settings=settings))

    response = client.post(
        "/api/v1/benchmark",
        json=_benchmark_payload("missing-dataset-id"),
    )

    assert response.status_code == 404

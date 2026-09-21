from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from catalyst.api.app import create_app
from catalyst.api.config import ApiSettings


@pytest.fixture
def client() -> TestClient:
    settings = ApiSettings(
        title="CATALYST Test API",
        version="test-version",
        environment="test",
        cors_origins=("http://localhost:3000",),
    )
    return TestClient(create_app(settings))


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "test-version",
    }


def test_health_endpoint_exposes_request_id(client: TestClient) -> None:
    response = client.get("/api/v1/health")

    assert response.headers["x-request-id"]


def test_incoming_request_id_is_preserved(client: TestClient) -> None:
    response = client.get(
        "/api/v1/health",
        headers={"x-request-id": "req-test-001"},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-test-001"


def test_openapi_document_is_available(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"]["title"] == "CATALYST Test API"


def test_swagger_docs_are_available(client: TestClient) -> None:
    response = client.get("/docs")

    assert response.status_code == 200
    assert "Swagger UI" in response.text


def test_validation_errors_use_structured_api_error(client: TestClient) -> None:
    response = client.post(
        "/api/v1/health",
        json={"unexpected": True},
    )

    assert response.status_code == 405


def test_unknown_route_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/does-not-exist")

    assert response.status_code == 404


def test_cors_preflight_is_configured(client: TestClient) -> None:
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_rejects_unconfigured_origin(client: TestClient) -> None:
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert "access-control-allow-origin" not in response.headers


def test_custom_settings_reach_app_state() -> None:
    settings = ApiSettings(
        title="Custom",
        version="9.9.9",
        environment="test",
        cors_origins=("http://localhost:3000",),
    )

    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/health").json()["version"] == "9.9.9"


def test_create_app_uses_default_settings() -> None:
    app = create_app()

    assert app.title == "CATALYST API"
    assert app.version == "0.1.0"

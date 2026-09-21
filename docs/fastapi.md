# Phase 8B — FastAPI Application Shell

Phase 8B introduces the first executable backend boundary for CATALYST.

## Runtime architecture

```text
HTTP client
    │
    ▼
FastAPI application
    │
    ├── CORS middleware
    ├── Request-ID middleware
    ├── Structured validation/error handlers
    │
    ▼
/api/v1
    │
    └── /health
```

## Current endpoint

`GET /api/v1/health`

Response:

```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

Every HTTP response also receives an `X-Request-ID` header. If a client supplies one, it is preserved.

## Configuration

Supported environment variables:

- `CATALYST_API_TITLE`
- `CATALYST_API_VERSION`
- `CATALYST_ENV`
- `CATALYST_CORS_ORIGINS`

`CATALYST_CORS_ORIGINS` accepts a comma-separated list.

Example:

```text
CATALYST_CORS_ORIGINS=http://localhost:3000,https://example.com
```

CORS is intentionally allow-list based. There is no wildcard origin default.

## Run locally

After installing the API optional dependencies:

```powershell
py -3.11 -m uv run --extra api uvicorn catalyst.api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Scope

8B intentionally contains no dataset or ML execution endpoint yet.

8C adds dataset ingestion and storage boundaries. 8D wires the profiling/benchmark engines. 8E exposes drift, robustness, Pareto, and optimization decisions.

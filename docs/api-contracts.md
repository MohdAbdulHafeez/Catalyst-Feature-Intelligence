# Phase 8A — Backend Product Contract

Phase 8A establishes the transport boundary between the CATALYST intelligence engine and the future web application.

## Design principles

- **Strict contracts:** unknown fields are rejected.
- **Immutable DTOs:** request and response models cannot be mutated after validation.
- **Stable identifiers:** dataset and job IDs have a constrained transport format.
- **No business logic in API models:** the models validate transport-level structure; domain engines remain in `catalyst.*`.
- **Explicit optimization policy:** constraints and weights are carried in the request rather than hidden in backend code.
- **Framework-independent:** these contracts do not import FastAPI, so the core package remains usable without running a web server.

## Initial workflow contract

```text
Frontend
   │
   ├── Dataset upload metadata
   │
   ├── ProfileRequest
   │
   ├── BenchmarkRequest
   │
   └── OptimizationRequest
   │
   ▼
FastAPI transport layer
   │
   ▼
CATALYST domain engines
```

Phase 8B will add the FastAPI application and dataset ingestion. Phase 8C will wire profiling, benchmarking, drift/robustness, and optimization into API endpoints.

## Contract boundary

The backend should translate API DTOs into existing domain models instead of changing the existing intelligence engines to understand HTTP, JSON, or frontend concerns.

That separation keeps the ML core testable and lets the API evolve independently.

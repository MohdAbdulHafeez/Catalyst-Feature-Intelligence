# Phase 8E — Categorical Intelligence API

Phase 8E exposes CATALYST's existing categorical intelligence engine through the backend.

## Endpoint

`POST /api/v1/categorical`

Request:

```json
{
  "dataset_id": "dataset-id",
  "columns": ["city", "segment"]
}
```

`columns` is optional. When omitted, CATALYST profiles every column that the existing `SchemaProfiler` classifies as categorical.

## Response

Each profile contains the unified categorical intelligence result:

```text
feature
 ├── cardinality
 ├── frequency distribution
 ├── ordinality
 └── risk
```

The API serializes the existing domain result; it does not duplicate the categorical heuristics in the web layer.

## Selection behavior

Explicit column selection is strict:

- unknown columns return HTTP 422;
- non-categorical selections return HTTP 422;
- duplicate column names are rejected during request validation.

## Flow

```text
POST /datasets
       ↓
dataset_id
       ↓
POST /categorical
       ↓
LocalDatasetStore
       ↓
CategoricalProfiler
       ↓
Cardinality + Frequency + Ordinality + Risk
       ↓
Frontend-ready JSON
```

## Architecture

The ML core remains independent of FastAPI. The route only performs transport validation, artifact loading, domain-engine invocation, and JSON adaptation.

Benchmark execution is the next API layer. It will consume the selected target, categorical columns, numerical columns, and model/metric configuration and then expose the existing leakage-safe benchmark engine.

# Phase 8D — Dataset Profiling API

Phase 8D connects stored datasets to the existing CATALYST Schema Intelligence Engine.

## Endpoint

`POST /api/v1/profile`

Request:

```json
{
  "dataset_id": "dataset-id",
  "target_column": "target"
}
```

`target_column` is optional. When present, the API verifies that the column exists before invoking the profiler.

## Flow

```text
Stored Dataset
      ↓
LocalDatasetStore.load_dataframe()
      ↓
SchemaProfiler.profile()
      ↓
ProfileResponse
```

The API exposes row/column counts, column-level semantic types, missingness, uniqueness, constant/identifier signals, sample values, and grouped column names.

The API does not reimplement schema inference. It adapts the existing ML-core `SchemaProfiler` output into strict Pydantic response DTOs.

Filesystem paths are never returned to clients.

Repeated profiling requests for the same stored dataset return the same response under the same library/configuration versions.

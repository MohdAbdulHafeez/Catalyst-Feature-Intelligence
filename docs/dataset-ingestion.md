# Phase 8C — Dataset Ingestion

Phase 8C makes the backend accept real datasets while keeping storage and parsing concerns isolated from the ML intelligence engines.

## Supported formats

- CSV
- Parquet

The uploaded filename determines the accepted format. Other extensions are rejected.

## Upload flow

```text
Browser / API client
        │
        ▼
POST /api/v1/datasets
        │
        ├── filename validation
        ├── format validation
        ├── streaming size limit
        ├── atomic file write
        ├── parser validation
        └── manifest creation
        │
        ▼
.catalyst/data/datasets/<dataset_id>/
        ├── data.csv / data.parquet
        └── manifest.json
```

## Dataset identifiers

The backend generates a 32-character UUID4 hex dataset ID. Clients do not control the storage path.

## Safety boundaries

The ingestion layer:

- never uses the original filename as a filesystem path;
- sanitizes path components;
- streams uploads and enforces a configurable byte limit;
- writes through a temporary file and atomically replaces the final file;
- removes partial files after failed ingestion;
- validates that the dataset is parseable;
- requires at least one column and one data row.

The current default maximum upload size is 25 MiB.

## Endpoints

### Upload

`POST /api/v1/datasets`

Multipart form field: `file`

### Retrieve metadata

`GET /api/v1/datasets/{dataset_id}`

The response returns metadata, not the physical filesystem path.

## Configuration

```text
CATALYST_DATA_DIR=.catalyst/data/datasets
CATALYST_MAX_UPLOAD_BYTES=26214400
```

Phase 8D will consume the stored dataset through a controlled service boundary and connect ingestion to the existing schema/profiling engine.

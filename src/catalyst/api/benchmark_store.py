from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from catalyst.api.contracts import TaskType
from catalyst.benchmark.models import BenchmarkResult

_BENCHMARK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class BenchmarkArtifact(BaseModel):
    """Persisted record connecting a benchmark result to its source dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str = Field(
        min_length=1,
        max_length=128,
    )
    dataset_id: str = Field(
        min_length=1,
        max_length=128,
    )
    target_column: str = Field(
        min_length=1,
        max_length=255,
    )
    task_type: TaskType
    metric: str = Field(
        min_length=1,
        max_length=64,
    )
    categorical_columns: tuple[str, ...] = ()
    numerical_columns: tuple[str, ...] = ()
    result: BenchmarkResult


class BenchmarkArtifactStore:
    """Durable JSON-backed storage for completed benchmark executions."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, artifact: BenchmarkArtifact) -> BenchmarkArtifact:
        self._validate_id(artifact.benchmark_id)

        destination = self._path(artifact.benchmark_id)
        temporary = destination.with_suffix(".tmp")

        temporary.write_text(
            artifact.model_dump_json(indent=2),
            encoding="utf-8",
        )
        temporary.replace(destination)

        return artifact

    def get(self, benchmark_id: str) -> BenchmarkArtifact:
        self._validate_id(benchmark_id)

        path = self._path(benchmark_id)

        if not path.is_file():
            raise FileNotFoundError(f"Benchmark '{benchmark_id}' was not found.")

        return BenchmarkArtifact.model_validate_json(path.read_text(encoding="utf-8"))

    def _path(self, benchmark_id: str) -> Path:
        return self.root / f"{benchmark_id}.json"

    @staticmethod
    def _validate_id(benchmark_id: str) -> None:
        if not _BENCHMARK_ID_PATTERN.fullmatch(benchmark_id):
            raise ValueError("benchmark_id contains invalid characters.")

from __future__ import annotations

import pytest

from catalyst.api.benchmark_store import BenchmarkArtifactStore


def test_benchmark_store_rejects_unsafe_id(tmp_path) -> None:
    store = BenchmarkArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="invalid characters"):
        store.get("../outside")


def test_benchmark_store_rejects_missing_artifact(tmp_path) -> None:
    store = BenchmarkArtifactStore(tmp_path)

    with pytest.raises(FileNotFoundError, match="was not found"):
        store.get("missing-benchmark")

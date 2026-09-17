# CATALYST

## Categorical Feature Intelligence & Optimization Engine

CATALYST is a production-oriented ML platform for diagnosing categorical data,
benchmarking leakage-safe encoding strategies, analyzing category drift and
computational cost, and generating reproducible preprocessing pipelines.

> **Status:** Early development — ML core first, platform second.

### Product thesis

Real-world tabular ML rarely fails because an engineer does not know what
`OneHotEncoder` is. It fails because categorical features introduce decisions
around cardinality, rare categories, unknown categories, leakage, drift,
dimensionality, memory, and model compatibility.

CATALYST is being built to make those decisions measurable and reproducible.

### Planned workflow

```text
Dataset
  ↓
Schema Intelligence
  ↓
Categorical Diagnosis
  ↓
Encoding Candidate Generation
  ↓
Leakage-Safe Cross-Validation
  ↓
Model-Aware Benchmarking
  ↓
Cost / Robustness Analysis
  ↓
Encoding Optimization
  ↓
Reproducible Pipeline
  ↓
Inference Validation
```

### Engineering principles

- ML decisions must be evidence-backed.
- Supervised encoders must be leakage-safe.
- Train/test category drift must be observable.
- Benchmark results must be reproducible.
- The ML core must remain independent from the web application.
- Every meaningful engineering milestone is committed separately.

### Local development

Prerequisite: Python 3.11+ and `uv`.

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

### License

MIT

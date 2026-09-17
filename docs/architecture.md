# CATALYST Architecture

The first implementation phase is deliberately limited to the reusable ML core.
The API, worker and web layers will depend on this core rather than containing
business logic themselves.

```text
              Web / API
                  │
                  ▼
        ┌───────────────────┐
        │   CATALYST Core   │
        │ Profiling         │
        │ Categorical       │
        │ Encoders          │
        │ Diagnostics       │
        │ Benchmarking      │
        │ Optimization      │
        │ Pipelines         │
        └─────────┬─────────┘
                  │
                  ▼
          Pandas / NumPy / SKL
```

## Planned ML modules

- `profiling`: dataset/schema inspection
- `categorical`: cardinality, rarity, ordinality and identifier analysis
- `encoders`: common adapter interface for categorical encoders
- `diagnostics`: leakage, drift and robustness checks
- `benchmarking`: leakage-safe cross-validation experiments
- `optimization`: cost/performance selection
- `pipelines`: serialization and reproducible inference

## First milestone non-goals

- LLM-based encoder selection
- frontend work
- production deployment
- distributed benchmark execution

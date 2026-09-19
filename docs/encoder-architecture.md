# Leakage-Safe Encoder Architecture

Phase 4 establishes the encoder contract that the benchmark and optimization
layers will consume.

## Core contract

Every encoder follows the sklearn lifecycle:

```text
fit(X_train, y_train?)
        ↓
learn training state
        ↓
transform(X_new)
```

Supervised encoders explicitly declare `requires_target = True`.

### Leakage boundary

For target-based encoding, the transformer must be inside the same sklearn
`Pipeline` as the estimator during cross-validation:

```text
Fold
├── encoder.fit(X_train, y_train)
├── model.fit(encoded_train, y_train)
└── encoder.transform(X_valid) → model.predict(...)
```

Pre-fitting a target encoder on the full dataset before cross-validation would
allow validation targets to influence the representation.

## MVP encoders

- **One-Hot:** sklearn OneHotEncoder adapter with unknown-category handling.
- **Ordinal:** sklearn OrdinalEncoder adapter with explicit unknown/missing sentinels.
- **Frequency:** training-partition category frequencies; unseen categories map to zero.
- **Target mean:** smoothed numeric target statistics learned only during fit.
- **Hashing:** fixed-width FeatureHasher representation for high-cardinality data.

## Why CATALYST owns the interface

Third-party encoder APIs must not leak into the benchmark or optimization
layers. CATALYST controls a stable interface and can add new implementations
without changing downstream consumers.

## Scope boundary

The encoder layer transforms data. It does **not** select the best encoder.
Selection belongs to the benchmark and optimization engines, where predictive
performance, dimensionality, memory, latency, drift and leakage controls can be
measured together.

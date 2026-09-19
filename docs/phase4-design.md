# Phase 4 Design Decisions

## Custom adapters over direct third-party coupling

CATALYST owns the encoder abstraction so the later benchmark system can compare
implementations through one stable contract.

## No automatic selection yet

The encoder layer is responsible for trustworthy transformations. The benchmark
layer will measure strategies, and the optimization layer will make trade-offs
explicit. This separation prevents selection logic from becoming hidden inside
individual encoders.

## Target encoding is deliberately constrained

The MVP target encoder accepts numeric targets only. It does not silently encode
arbitrary class labels as numbers because that could introduce an unintended
ordering or target meaning. Multiclass target statistics can be added later as an
explicit design.

# Ordinality & Categorical Risk Intelligence

Phase 3 extends CATALYST from descriptive category statistics into explainable
semantic and risk signals.

## Ordinality

Ordinal inference deliberately avoids target information. It uses:

- explicitly ordered pandas categorical dtypes
- supported semantic vocabularies
- explicit numeric level labels such as `Level 1`, `Level 2`, etc.

The result includes the inferred order, unmatched categories, confidence and
the exact signal used.

This is a **signal**, not a universal truth. Domain metadata can override the
automatic inference in later versions.

## Risk findings

Current risk types:

- high categorical cardinality
- One-Hot dimensionality explosion
- many rare categories
- highly concentrated category distributions
- missing categorical values
- unresolved ordinal semantics

Risk findings include severity, a numeric score, an explanation and suggested
actions. The overall score is currently the maximum finding score rather than a
weighted average; this keeps a severe failure visible instead of allowing many
low-risk features to dilute it.

The risk engine is deterministic and configurable. Thresholds are intended for
benchmarking and later calibration against real datasets.


## Design rule

Ordinality detection is advisory. CATALYST must never silently convert an
ambiguous nominal variable into an ordinal representation. The signal is exposed
to the later decision engine, where domain overrides and benchmark evidence can
be incorporated.

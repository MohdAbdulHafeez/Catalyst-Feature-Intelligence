# Categorical Intelligence Engine

Phase 2 of CATALYST turns schema-level categorical detection into measurable
feature intelligence.

## Metrics

### Cardinality

For a feature with `k` observed categories and `n` non-null observations:

`cardinality_ratio = k / n`

The engine also measures:

- dominant category share
- Shannon entropy
- normalized entropy
- effective category count (`exp(entropy)`)
- singleton category count
- rare category count
- rare category ratio
- estimated One-Hot dimensions

### Frequency concentration

The frequency analyzer provides:

- top-k categories
- per-category counts
- per-category shares
- cumulative share
- Herfindahl-Hirschman concentration index
- top-3/top-5/top-10 cumulative shares
- unique-category share

These are descriptive signals. They do not select an encoder by themselves.

## Rare category definition

A category is currently flagged as rare using an adaptive absolute count
threshold derived from both the configured absolute threshold and the
configured relative-share threshold. This prevents tiny datasets from
classifying every category as rare merely because the absolute threshold is
larger than the sample size.

Both thresholds are configurable so later benchmark experiments can measure
their effect instead of relying on an unexplained hard-coded constant.

## Architectural boundary

This phase deliberately stops at feature intelligence. Encoder selection belongs
to later phases and must consume these measurements alongside leakage, drift,
model performance and computational cost.

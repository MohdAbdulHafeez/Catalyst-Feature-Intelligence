# CATALYST Optimization Intelligence

## Phase 7A

Phase 7 begins by combining benchmark performance, computational cost, and
encoder robustness observations into a common optimization evidence model.

The system does not immediately select a single encoder.

## Cost

Cost is derived from:

- transformed feature count
- transformed representation memory
- total fit time
- total score time

The components are normalized across the completed candidate population using
min-max scaling.

The resulting `cost_score` is bounded to `[0, 1]`.

## Robustness risk

The initial risk model uses observed encoder robustness signals:

- unseen category rate
- non-finite output rate
- absolute missingness-rate change

The signals are aggregated across the categorical features evaluated for the
same encoder. The weighted result is bounded to `[0, 1]`.

When robustness evidence is unavailable for a candidate, CATALYST uses a
conservative risk representation rather than silently treating missing
measurements as zero risk.

## Pareto analysis

Phase 7A models three objectives for classification benchmarks:

- maximize validation performance
- minimize computational cost
- minimize robustness risk

A candidate is dominated when another candidate is at least as good on all
three objectives and strictly better on at least one.

The Pareto frontier therefore preserves multiple observed trade-offs instead
of forcing a universal single choice.

## Design boundary

Phase 7A produces measurable optimization evidence and Pareto trade-offs.

It does not yet apply a user-specific objective function or automatically
select a final encoder.

Those behaviors belong to the next optimization stage.

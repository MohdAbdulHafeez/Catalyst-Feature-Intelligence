# CATALYST Optimization Decision Engine

Phase 7B converts Pareto-frontier evidence into a deterministic, policy-driven decision.

## Decision flow

1. Build optimization evidence from benchmark and robustness measurements.
2. Compute the Pareto frontier using performance, cost, and risk.
3. Ignore dominated candidates for final decision-making.
4. Apply explicit feasibility constraints:
   - minimum performance
   - maximum normalized cost
   - maximum risk
5. Compute a reproducible utility from configured weights.
6. Return the selected candidate plus complete decision evidence.

## Utility

For a candidate:

`U = (wp * P + wc * (1 - C) + wr * (1 - R)) / (wp + wc + wr)`

where:

- `P` is min-max normalized performance across the supplied evidence.
- `C` is normalized computational cost.
- `R` is aggregated robustness risk.
- `wp`, `wc`, and `wr` are explicit policy weights.

There is no hidden encoder preference in the decision engine. Changing the policy changes the decision and changes the policy fingerprint.

## Reproducibility

`SelectionPolicy.fingerprint` is a stable SHA-256-derived identifier for the exact policy configuration. The decision output includes this fingerprint so recommendations can be reproduced and audited.

## No-feasible-result behavior

When no Pareto candidate satisfies the configured constraints, the engine returns `selected_candidate=None`. It does not silently relax constraints.

## Scope

Phase 7B is intentionally limited to deterministic decision logic. Persistence, experiment replay, deployment packaging, and API orchestration belong to later phases.

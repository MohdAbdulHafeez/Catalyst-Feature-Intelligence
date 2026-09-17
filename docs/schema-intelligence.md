# Schema Intelligence Engine

The Schema Intelligence Engine is CATALYST's first ML-core component.

It performs deterministic, explainable inspection of a pandas DataFrame:

- semantic type inference
- missingness measurement
- uniqueness/cardinality signals
- constant-column detection
- identifier-risk detection
- representative sample extraction

The engine intentionally does **not** choose encoders yet. That belongs to the
Categorical Intelligence and Encoding Decision layers.

Identifier scores are heuristic scores, not probabilities. Every score is
accompanied by explicit signals so downstream UI/reporting can explain the
decision.

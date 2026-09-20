\# CATALYST Drift \& Robustness Intelligence



\## Phase 6A — Categorical Distribution Drift



Phase 6 begins by measuring how categorical feature distributions change

between a training dataset and a reference dataset.



The reference dataset can represent a serving window, production sample,

future validation set, or another population that should be compared with the

training distribution.



\## Distribution profile



For each categorical feature, CATALYST records:



\- row count

\- non-missing count

\- missing count

\- missing rate

\- unique category count

\- category frequencies

\- category proportions



Category identities are canonicalized with their Python type information so

values such as integer `1` and string `"1"` do not silently collapse into the

same category.



\## Drift measurements



Phase 6A exposes:



\### Population Stability Index



PSI measures the change between two categorical probability distributions.

Zero represents identical aligned distributions.



Zero-frequency bins are stabilized with a configurable epsilon before the

logarithmic calculation.



\### Jensen-Shannon divergence



Jensen-Shannon divergence provides a symmetric divergence measure derived from

the two distributions and their midpoint.



\### Total Variation distance



Total Variation distance measures the absolute distributional difference and is

bounded between zero and one.



\### Unseen category rate



The percentage of reference rows whose non-missing categorical value was not

observed in training.



This is measured separately from missingness.



\### Missingness shift



Reference missing rate minus training missing rate.



A positive value indicates more missing categorical values in the reference

population.



\### Concentration shift



HHI and top-category share are calculated for both populations and their

changes are retained.



These measurements help characterize cases where the set of categories remains

similar but the population becomes substantially more concentrated.



\## Design boundary



Phase 6A measures drift.



It does not yet assign risk severity, select an encoder, or make an optimization

decision.



Later CATALYST stages will combine:



\- benchmark performance

\- computational cost

\- distribution drift

\- unknown-category exposure

\- representation robustness



to support cost- and risk-aware optimization.




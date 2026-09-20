\# CATALYST Benchmark Engine



\## Purpose



The benchmark engine evaluates categorical encoding strategies under

leakage-safe cross-validation.



Phase 5 currently supports classification with Logistic Regression.



\## Leakage boundary



A fresh sklearn pipeline is constructed for every candidate and every

cross-validation fold.



For each fold:



1\. Split the dataset into training and validation partitions.

2\. Construct a new encoder/model pipeline.

3\. Fit the encoder and model using only the training partition.

4\. Transform and score the validation partition.

5\. Record benchmark measurements.



This is especially important for supervised encoders such as target mean

encoding.



\## Measurements



Each completed fold records:



\- validation score

\- fit time

\- score time

\- transformed feature count

\- transformed-matrix memory proxy



The engine aggregates fold results into:



\- mean score

\- score standard deviation

\- total fit time

\- total score time

\- mean feature count

\- peak feature count



The memory measurement is a representation-size proxy for the transformed

validation matrix. It is not process-level peak RSS.



\## Candidate isolation



A failure in one candidate does not abort the complete benchmark.



The failed candidate is returned with:



\- `status = failed`

\- exception type

\- error message

\- any folds that completed before the failure



This allows the benchmark matrix to remain usable even when an individual

encoder configuration is incompatible with a dataset.



\## Determinism



The default configuration uses:



\- 5 cross-validation folds

\- shuffled stratified splitting

\- random state `42`



The same dataset, candidates, metric, and configuration should reproduce the

same fold assignments and validation scores.



Runtime measurements are naturally expected to vary between executions.



\## Scope



Optimization and recommendation are intentionally outside the benchmark

engine.



The benchmark engine measures candidates.



Later CATALYST layers will use those measurements together with cost and risk

signals to produce optimization decisions.

